from types import SimpleNamespace

import pytest

from torch_mojo_backend import eager_kernels
from torch_mojo_backend.mojo_device import mojo_device_autograd as autograd


class _Token:
    def __init__(self, name: str):
        self.name = name

    def _contig(self):
        return self

    def __repr__(self) -> str:
        return self.name


def _patch_saved_and_views(
    monkeypatch: pytest.MonkeyPatch,
    saved: tuple[_Token, ...],
    views: list[tuple[str, tuple[int, ...]]],
) -> None:
    monkeypatch.setattr(autograd, "_restore_saved_mojo_tensors", lambda _ctx: saved)

    def contiguous_view(tensor: _Token, shape) -> _Token:
        normalized = tuple(shape)
        views.append((tensor.name, normalized))
        return _Token(f"view({tensor.name},{normalized})")

    monkeypatch.setattr(autograd, "_contiguous_view", contiguous_view)


def _query_gradient_context(has_dropout: bool) -> SimpleNamespace:
    saved_names = ["key", "value", "probabilities"]
    if has_dropout:
        saved_names.append("dropout_mask")
    return SimpleNamespace(
        saved_names=tuple(saved_names),
        needed_input_gradients=(True, False, False),
        query_shape=(1, 2, 3, 4),
        key_shape=(1, 2, 5, 4),
        value_shape=(1, 2, 5, 4),
        has_dropout=has_dropout,
        dropout_scale=1.25 if has_dropout else 1.0,
        scale=-0.5,
        is_causal=False,
    )


def test_sdpa_backward_bridge_is_lazy_registered() -> None:
    from torch_mojo_backend.eager_kernels import aten_fast

    assert aten_fast._SdpaBackwardExtension.MOJO_FILE.name == "sdpa_backward_ops.mojo"


def test_missing_sdpa_kernel_module_returns_not_handled_before_device_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from torch_mojo_backend.eager_kernels import aten_fast

    device = object()
    probabilities = SimpleNamespace(
        _dtype=aten_fast.DType.float32, _device=device, _shape=(2, 3)
    )
    grad = SimpleNamespace(
        _dtype=aten_fast.DType.float32, _device=device, _shape=(2, 3)
    )
    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_on_gpu", lambda _tensor: True)

    def forbidden_device_work(*_args, **_kwargs):
        raise AssertionError("missing SDPA bridge reached materialization/allocation")

    monkeypatch.setattr(aten_fast, "_tc", forbidden_device_work)
    monkeypatch.setattr(aten_fast, "_alloc", forbidden_device_work)

    monkeypatch.setattr(
        aten_fast,
        "_SDPA_BACKWARD_SOURCE_PATHS",
        (tmp_path / "missing_fable_kernel.mojo",),
    )

    def forbidden_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("absent SDPA kernel triggered lazy Mojo compilation")

    monkeypatch.setattr(
        eager_kernels.MOJO_EXTENSION_LOADER, "load_canonical", forbidden_load
    )
    result = aten_fast.fast_sdpa_dropout_softmax_backward(
        probabilities, grad, None, object(), 0.5
    )

    assert result is aten_fast.NOT_HANDLED


@pytest.mark.parametrize("has_dropout", [False, True])
def test_sdpa_query_gradient_prefers_fused_dropout_softmax_backward(
    monkeypatch: pytest.MonkeyPatch, has_dropout: bool
) -> None:
    ctx = _query_gradient_context(has_dropout)
    saved = tuple(_Token(name) for name in ctx.saved_names)
    views: list[tuple[str, tuple[int, ...]]] = []
    _patch_saved_and_views(monkeypatch, saved, views)
    not_handled = object()
    fused_calls = []

    def probability_gradient(grad, value):
        return _Token(f"dprob({grad.name},{value.name})")

    def fused(probabilities, grad, mask, dropout_scale, score_scale, causal, q_len):
        fused_calls.append(
            (probabilities, grad, mask, dropout_scale, score_scale, causal, q_len)
        )
        return _Token("dscores-fused")

    def query_gradient(scores, key):
        assert scores.name == "dscores-fused"
        return _Token(f"dquery({scores.name},{key.name})")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("fused-supported inputs reached the decomposition")

    fake_fast = SimpleNamespace(
        NOT_HANDLED=not_handled,
        fast_sdpa_backward=lambda *_args: not_handled,
        _fast_aten_bmm_transpose_b=probability_gradient,
        fast_sdpa_dropout_softmax_backward=fused,
        fast_aten_bmm=query_gradient,
        fast_aten_native_dropout_backward=forbidden,
        fast_aten_mul=forbidden,
        fast_aten_sum=forbidden,
        fast_aten_sub=forbidden,
    )
    monkeypatch.setattr(autograd, "_fast", lambda: fake_fast)

    result = autograd._ScaledDotProductAttentionAutograd.backward(
        ctx, _Token("grad_output")
    )

    assert result[0].name.startswith("view(dquery(dscores-fused")
    assert result[1:] == (None, None, None, None, None, None, None)
    assert len(fused_calls) == 1
    probabilities, grad, mask, dropout_scale, score_scale, causal, q_len = fused_calls[
        0
    ]
    assert causal is ctx.is_causal
    assert q_len == ctx.query_shape[2]
    assert probabilities.name.startswith("view(probabilities")
    assert grad.name.startswith("dprob(view(grad_output")
    assert (mask is not None) == has_dropout
    if mask is not None:
        assert mask.name.startswith("view(dropout_mask")
    assert dropout_scale == ctx.dropout_scale
    assert score_scale == ctx.scale


@pytest.mark.parametrize("has_dropout", [False, True])
def test_sdpa_fused_not_handled_preserves_decomposition_order(
    monkeypatch: pytest.MonkeyPatch, has_dropout: bool
) -> None:
    ctx = _query_gradient_context(has_dropout)
    saved = tuple(_Token(name) for name in ctx.saved_names)
    views: list[tuple[str, tuple[int, ...]]] = []
    _patch_saved_and_views(monkeypatch, saved, views)
    not_handled = object()
    operations = []

    def probability_gradient(grad, value):
        return _Token(f"dprob({grad.name},{value.name})")

    def dropout(grad, mask, scale):
        operations.append(("dropout", grad.name, mask.name, scale))
        return _Token("masked-dprob")

    def multiply(lhs, rhs):
        rhs_name = rhs.name if isinstance(rhs, _Token) else rhs
        operations.append(("mul", lhs.name, rhs_name))
        return _Token(f"mul-{len(operations)}")

    def reduce_sum(value, *, dim, keepdim):
        operations.append(("sum", value.name, tuple(dim), keepdim))
        return _Token("row-sum")

    def subtract(lhs, rhs):
        operations.append(("sub", lhs.name, rhs.name))
        return _Token("centered")

    def query_gradient(scores, key):
        operations.append(("bmm", scores.name, key.name))
        return _Token("dquery")

    fake_fast = SimpleNamespace(
        NOT_HANDLED=not_handled,
        fast_sdpa_backward=lambda *_args: not_handled,
        _fast_aten_bmm_transpose_b=probability_gradient,
        fast_sdpa_dropout_softmax_backward=lambda *_args: not_handled,
        fast_aten_native_dropout_backward=dropout,
        fast_aten_mul=multiply,
        fast_aten_sum=reduce_sum,
        fast_aten_sub=subtract,
        fast_aten_bmm=query_gradient,
    )
    monkeypatch.setattr(autograd, "_fast", lambda: fake_fast)

    result = autograd._ScaledDotProductAttentionAutograd.backward(
        ctx, _Token("grad_output")
    )

    names = [operation[0] for operation in operations]
    assert names == (
        ["dropout", "mul", "sum", "sub", "mul", "mul", "bmm"]
        if has_dropout
        else ["mul", "sum", "sub", "mul", "mul", "bmm"]
    )
    multiplies = [operation for operation in operations if operation[0] == "mul"]
    if has_dropout:
        assert multiplies[0][1] == "masked-dprob"
    else:
        assert multiplies[0][1].startswith("dprob(view(grad_output")
    assert multiplies[0][2].startswith("view(probabilities")
    assert multiplies[1][1] == "centered"
    assert multiplies[1][2].startswith("view(probabilities")
    assert multiplies[2][2] == ctx.scale
    assert result[0].name.startswith("view(dquery")


@pytest.mark.parametrize("has_dropout", [False, True])
def test_sdpa_value_only_gradient_skips_fused_score_gradient(
    monkeypatch: pytest.MonkeyPatch, has_dropout: bool
) -> None:
    names = ["probabilities"]
    if has_dropout:
        names.append("dropout_mask")
    ctx = SimpleNamespace(
        saved_names=tuple(names),
        needed_input_gradients=(False, False, True),
        query_shape=(1, 2, 3, 4),
        key_shape=(1, 2, 5, 4),
        value_shape=(1, 2, 5, 4),
        has_dropout=has_dropout,
        dropout_scale=1.25 if has_dropout else 1.0,
        scale=0.5,
    )
    saved = tuple(_Token(name) for name in names)
    views: list[tuple[str, tuple[int, ...]]] = []
    _patch_saved_and_views(monkeypatch, saved, views)
    not_handled = object()
    calls = []

    def dropout(probabilities, mask, scale):
        calls.append(("dropout", probabilities.name, mask.name, scale))
        return _Token("effective-probabilities")

    def transpose(value, dim0, dim1):
        calls.append(("transpose", value.name, dim0, dim1))
        return _Token(f"transpose-{len(calls)}")

    def bmm(lhs, rhs):
        calls.append(("bmm", lhs.name, rhs.name))
        return _Token("dvalue-transposed")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("value-only backward constructed dScores")

    fake_fast = SimpleNamespace(
        NOT_HANDLED=not_handled,
        fast_sdpa_backward=lambda *_args: not_handled,
        fast_aten_native_dropout_backward=dropout,
        fast_aten_transpose=transpose,
        fast_aten_bmm=bmm,
        _fast_aten_bmm_transpose_b=forbidden,
        fast_sdpa_dropout_softmax_backward=forbidden,
    )
    monkeypatch.setattr(autograd, "_fast", lambda: fake_fast)

    result = autograd._ScaledDotProductAttentionAutograd.backward(
        ctx, _Token("grad_output")
    )

    assert result[:2] == (None, None)
    assert result[2].name.startswith("view(view(transpose")
    assert [call[0] for call in calls] == (
        ["dropout", "transpose", "bmm", "transpose"]
        if has_dropout
        else ["transpose", "bmm", "transpose"]
    )
