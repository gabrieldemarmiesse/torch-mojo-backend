"""Host-only contracts for the vendored H100 BF16 FA4 integration."""

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from torch_mojo_backend.eager_kernels import aten_fast
from torch_mojo_backend.mojo_device import mojo_device_autograd as autograd


def _device(*, arch: str = "sm_90a") -> SimpleNamespace:
    return SimpleNamespace(api="cuda", architecture_name=arch, label="gpu", id=0)


def _tensor(
    name: str,
    shape=(2, 4, 128, 64),
    *,
    dtype=None,
    device=None,
    ptr: int = 100,
    strides=None,
    contiguous: bool | None = None,
) -> SimpleNamespace:
    shape = tuple(shape)
    strides = aten_fast._row_major_strides(shape) if strides is None else tuple(strides)
    if contiguous is None:
        contiguous = strides == aten_fast._row_major_strides(shape)
    return SimpleNamespace(
        name=name,
        _shape=shape,
        _strides=strides,
        _offset=0,
        _dtype=dtype or aten_fast.DType.bfloat16,
        _device=device or _device(),
        _ptr=ptr,
        _itemsize=(dtype or aten_fast.DType.bfloat16).size_in_bytes,
        _numel=math.prod(shape),
        _is_contiguous=contiguous,
        _holder=object(),
    )


def test_fa4_rejects_ineligible_regimes_before_loading_or_device_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_mojo_backend.eager_flash_attention as package

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an ineligible FA4 call reached device/compiler work")

    monkeypatch.setattr(package, "load_fa4_ops", forbidden)
    monkeypatch.setattr(aten_fast, "_fa4_native_bthd", forbidden)
    monkeypatch.setattr(aten_fast, "_alloc", forbidden)

    cases = [
        ((_tensor("q", dtype=aten_fast.DType.float32),), {}),
        ((_tensor("q", device=_device(arch="sm_80")),), {}),
        ((_tensor("q", shape=(2, 4, 96, 64)),), {}),
        ((_tensor("q", shape=(2, 4, 128, 32)),), {}),
        ((_tensor("q"),), {"is_causal": False}),
        ((_tensor("q"),), {"dropout_p": 0.1}),
        ((_tensor("q"),), {"attn_mask": object()}),
        ((_tensor("q"),), {"enable_gqa": True}),
    ]
    for (query,), overrides in cases:
        key = _tensor("k", dtype=query._dtype, device=query._device)
        value = _tensor("v", dtype=query._dtype, device=query._device)
        kwargs = {
            "attn_mask": None,
            "dropout_p": 0.0,
            "is_causal": True,
            "scale": None,
            "enable_gqa": False,
        }
        kwargs.update(overrides)
        assert (
            aten_fast.fast_fa4_bf16_d64_causal_forward(query, key, value, **kwargs)
            is aten_fast.NOT_HANDLED
        )


def test_fa4_forward_bridge_uses_dynamic_bthd_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_mojo_backend.eager_flash_attention as package

    device = _device()
    public = [
        _tensor(name, shape=(3, 12, 256, 64), device=device, ptr=ptr)
        for name, ptr in zip(("q", "k", "v"), (10, 20, 30), strict=True)
    ]
    native = {
        tensor.name: _tensor(
            f"{tensor.name}_native",
            shape=(3, 256, 12, 64),
            device=device,
            ptr=tensor._ptr + 100,
        )
        for tensor in public
    }
    allocations = []
    bridge_calls = []

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=1000 + len(allocations),
        )
        allocations.append(result)
        return result

    def transpose(tensor, dim0, dim1):
        assert (dim0, dim1) == (1, 2)
        shape = list(tensor._shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        return _tensor(
            "output", shape, dtype=tensor._dtype, device=device, ptr=tensor._ptr
        )

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(
        aten_fast, "_fa4_native_bthd", lambda tensor: native[tensor.name]
    )
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", transpose)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 9090)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_fwd_bf16_d64_causal=lambda *args: bridge_calls.append(args)
        ),
    )

    result = aten_fast.fast_fa4_bf16_d64_causal_forward(
        *public, is_causal=True, scale=0.125
    )
    output, logsumexp, q_native, k_native, v_native = result

    assert output._shape == (3, 12, 256, 64)
    assert logsumexp._shape == (3, 12, 256)
    assert (q_native, k_native, v_native) == tuple(native[name] for name in "qkv")
    assert [(item._shape, item._dtype) for item in allocations] == [
        ((3, 256, 12, 64), aten_fast.DType.bfloat16),
        ((3, 12, 256), aten_fast.DType.float32),
    ]
    assert bridge_calls == [
        (
            110,
            120,
            130,
            allocations[0]._ptr,
            allocations[1]._ptr,
            3,
            256,
            12,
            0.125,
            9090,
        )
    ]


def test_fa4_strided_layout_contract_is_strict() -> None:
    shape = (2, 256, 12, 64)
    token_stride = 3 * shape[2] * shape[3]
    strides = (shape[1] * token_stride, token_stride, 64, 1)
    eligible = _tensor(
        "q_native", shape=shape, ptr=0x1000, strides=strides, contiguous=False
    )
    assert aten_fast._fa4_strided_bthd_layout(eligible)

    invalid = []
    for updates in (
        {"_ptr": eligible._ptr + 2},
        {"_shape": (2, 255, 12, 64)},
        {"_shape": (2, 256, 12, 32)},
        {"_strides": (strides[0], strides[1], 128, 1)},
        {"_strides": (strides[0], 12 * 64 - 8, 64, 1)},
        {"_strides": (strides[0] + 8, strides[1], 64, 1)},
        {"_strides": (strides[0], strides[1], 64, 2)},
        {"_strides": (strides[0], strides[1] + 2, 64, 1)},
    ):
        candidate = SimpleNamespace(**vars(eligible))
        for name, value in updates.items():
            setattr(candidate, name, value)
        invalid.append(candidate)
    assert not any(aten_fast._fa4_strided_bthd_layout(tensor) for tensor in invalid)


def test_fa4_canonical_fused_qkv_uses_zero_copy_strided_forward_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_mojo_backend.eager_flash_attention as package

    device = _device()
    batch, heads, seqlen, head_dim = 2, 12, 256, 64
    token_stride = 3 * heads * head_dim
    batch_stride = seqlen * token_stride
    public_strides = (batch_stride, head_dim, token_stride, 1)
    pointers = (0x1000, 0x1600, 0x1C00)
    public = tuple(
        _tensor(
            name,
            shape=(batch, heads, seqlen, head_dim),
            device=device,
            ptr=ptr,
            strides=public_strides,
            contiguous=False,
        )
        for name, ptr in zip("qkv", pointers, strict=True)
    )
    allocations = []
    strided_calls = []

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=0x8000 + len(allocations) * 0x1000,
        )
        allocations.append(result)
        return result

    def transpose(tensor, dim0, dim1):
        assert (dim0, dim1) == (1, 2)
        shape = list(tensor._shape)
        strides = list(tensor._strides)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
        return _tensor(
            f"{tensor.name}_transpose",
            shape=shape,
            dtype=tensor._dtype,
            device=tensor._device,
            ptr=tensor._ptr,
            strides=strides,
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("canonical fused QKV reached a copy or old FA4 bridge")

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_tc", forbidden)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", transpose)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 9090)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_fwd_bf16_d64_causal=forbidden,
            flash_attention_fwd_bf16_d64_causal_strided_qkv=lambda *args: (
                strided_calls.append(args)
            ),
        ),
    )

    output, logsumexp, q_native, k_native, v_native = (
        aten_fast.fast_fa4_bf16_d64_causal_forward(*public, is_causal=True, scale=0.125)
    )

    physical_strides = (batch_stride, token_stride, head_dim, 1)
    assert output._shape == (batch, heads, seqlen, head_dim)
    assert logsumexp._shape == (batch, heads, seqlen)
    assert tuple(tensor._ptr for tensor in (q_native, k_native, v_native)) == pointers
    assert all(
        tensor._strides == physical_strides for tensor in (q_native, k_native, v_native)
    )
    assert strided_calls == [
        (
            pointers[0],
            *physical_strides,
            pointers[1],
            *physical_strides,
            pointers[2],
            *physical_strides,
            allocations[0]._ptr,
            allocations[1]._ptr,
            batch,
            seqlen,
            heads,
            0.125,
            9090,
        )
    ]


def test_fa4_unsupported_public_layout_copies_and_uses_contiguous_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_mojo_backend.eager_flash_attention as package

    device = _device()
    public = tuple(
        _tensor(name, shape=(2, 12, 256, 64), device=device, ptr=ptr)
        for name, ptr in zip("qkv", (0x1000, 0x2000, 0x3000), strict=True)
    )
    copied = []
    old_calls = []
    allocations = []

    def transpose(tensor, dim0, dim1):
        shape = list(tensor._shape)
        strides = list(tensor._strides)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
        return _tensor(
            tensor.name,
            shape=shape,
            dtype=tensor._dtype,
            device=tensor._device,
            ptr=tensor._ptr,
            strides=strides,
        )

    def make_contiguous(tensor):
        copied.append(tensor)
        return _tensor(
            f"{tensor.name}_copy",
            shape=tensor._shape,
            dtype=tensor._dtype,
            device=tensor._device,
            ptr=0x4000 + len(copied) * 0x1000,
        )

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=0x8000 + len(allocations) * 0x1000,
        )
        allocations.append(result)
        return result

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsupported QKV selected the strided FA4 bridge")

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(aten_fast, "_tc", make_contiguous)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", transpose)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 7070)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_fwd_bf16_d64_causal=lambda *args: old_calls.append(args),
            flash_attention_fwd_bf16_d64_causal_strided_qkv=forbidden,
        ),
    )

    result = aten_fast.fast_fa4_bf16_d64_causal_forward(
        *public, is_causal=True, scale=0.125
    )

    assert result is not aten_fast.NOT_HANDLED
    assert [tensor.name for tensor in copied] == ["q", "k", "v"]
    assert len(old_calls) == 1
    assert old_calls[0][:3] == (0x5000, 0x6000, 0x7000)


def test_direct_flash_aten_returns_real_lse_and_cuda_shaped_auxiliaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _device()
    query, key, value = (
        _tensor(name, shape=(2, 8, 256, 64), device=device, ptr=ptr)
        for name, ptr in zip(("q", "k", "v"), (10, 20, 30), strict=True)
    )
    output = _tensor("output", shape=query._shape, device=device, ptr=40)
    logsumexp = _tensor(
        "lse", shape=(2, 8, 256), dtype=aten_fast.DType.float32, device=device, ptr=50
    )
    physical = tuple(
        _tensor(name, shape=(2, 256, 8, 64), device=device, ptr=ptr)
        for name, ptr in zip(("qn", "kn", "vn"), (60, 70, 80), strict=True)
    )
    allocations = []

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(
        aten_fast,
        "fast_fa4_bf16_d64_causal_forward",
        lambda *_args: (output, logsumexp, *physical),
    )

    def alloc(shape, dtype, actual_device):
        allocations.append((tuple(shape), dtype, actual_device))
        return _tensor(
            f"alloc{len(allocations)}", shape=shape, dtype=dtype, device=actual_device
        )

    monkeypatch.setattr(aten_fast, "_alloc", alloc)

    result = aten_fast.fast_aten__scaled_dot_product_flash_attention(
        query, key, value, dropout_p=0.0, is_causal=True
    )

    assert result[:6] == (output, logsumexp, None, None, 256, 256)
    assert result[6]._dtype == aten_fast.DType.uint64
    assert result[7]._dtype == aten_fast.DType.uint64
    assert result[8]._shape == (0,)
    assert allocations == [
        ((2,), aten_fast.DType.uint64, device),
        ((), aten_fast.DType.uint64, device),
        ((0,), aten_fast.DType.bfloat16, device),
    ]
    assert (
        aten_fast.fast_aten__scaled_dot_product_flash_attention(
            query, key, value, dropout_p=0.0, is_causal=True, return_debug_mask=True
        )
        is aten_fast.NOT_HANDLED
    )


def test_direct_flash_backward_materializes_strided_logsumexp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = _device()
    shape = (2, 4, 128, 64)
    query, key, value = (
        _tensor(name, shape=shape, device=device, ptr=ptr)
        for name, ptr in zip(("q", "k", "v"), (0x1000, 0x2000, 0x3000), strict=True)
    )
    grad = _tensor("grad", shape=shape, device=device, ptr=0x4000)
    output = _tensor("output", shape=shape, device=device, ptr=0x5000)
    lse = _tensor(
        "lse",
        shape=(2, 4, 128),
        dtype=aten_fast.DType.float32,
        device=device,
        ptr=0x6000,
        strides=(4 * 128 * 2, 128 * 2, 2),
        contiguous=False,
    )
    contiguous_lse = _tensor(
        "lse_contiguous",
        shape=lse._shape,
        dtype=aten_fast.DType.float32,
        device=device,
        ptr=0x7000,
    )
    native = {
        tensor.name: _tensor(
            f"{tensor.name}_native",
            shape=(2, 128, 4, 64),
            device=device,
            ptr=tensor._ptr,
        )
        for tensor in (query, key, value)
    }
    backward_calls = []

    monkeypatch.setattr(aten_fast, "_t", lambda tensor: tensor)
    monkeypatch.setattr(
        aten_fast, "_fa4_native_bthd", lambda tensor: native[tensor.name]
    )

    def make_contiguous(tensor):
        assert tensor is lse
        return contiguous_lse

    monkeypatch.setattr(aten_fast, "_tc", make_contiguous)
    monkeypatch.setattr(
        aten_fast,
        "fast_fa4_bf16_d64_causal_backward",
        lambda *args: backward_calls.append(args) or "gradients",
    )

    result = aten_fast.fast_aten__scaled_dot_product_flash_attention_backward(
        grad,
        query,
        key,
        value,
        output,
        lse,
        None,
        None,
        128,
        128,
        0.0,
        True,
        None,
        None,
        scale=0.125,
    )

    assert result == "gradients"
    assert len(backward_calls) == 1
    assert backward_calls[0][4] is contiguous_lse


def test_fa4_combined_backward_bridge_allocates_exact_scratch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_mojo_backend.eager_flash_attention as package

    device = _device()
    q_native, k_native, v_native = [
        _tensor(name, shape=(2, 384, 4, 64), device=device, ptr=ptr)
        for name, ptr in zip(("q", "k", "v"), (11, 22, 33), strict=True)
    ]
    output = _tensor("output", shape=(2, 4, 384, 64), device=device, ptr=44)
    logsumexp = _tensor(
        "lse", shape=(2, 4, 384), dtype=aten_fast.DType.float32, device=device, ptr=55
    )
    grad_output = _tensor("grad", shape=(2, 4, 384, 64), device=device, ptr=66)
    physical = {
        "output": _tensor("out_native", shape=(2, 384, 4, 64), device=device, ptr=77),
        "grad": _tensor("dout_native", shape=(2, 384, 4, 64), device=device, ptr=88),
    }
    allocations = []
    bridge_calls = []

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=1000 + len(allocations),
        )
        allocations.append(result)
        return result

    def transpose(tensor, dim0, dim1):
        shape = list(tensor._shape)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        return _tensor(
            f"public_{tensor.name}",
            shape,
            dtype=tensor._dtype,
            device=device,
            ptr=tensor._ptr,
        )

    monkeypatch.setattr(
        aten_fast, "_fa4_native_bthd", lambda tensor: physical[tensor.name]
    )
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", transpose)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 8080)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_bwd_bf16_d64_causal=lambda *args: bridge_calls.append(args)
        ),
    )

    gradients = aten_fast.fast_fa4_bf16_d64_causal_backward(
        q_native, k_native, v_native, output, logsumexp, grad_output, 0.125
    )

    assert [gradient._shape for gradient in gradients] == [(2, 4, 384, 64)] * 3
    assert [(item._shape, item._dtype) for item in allocations] == [
        ((2, 384, 4, 64), aten_fast.DType.bfloat16),
        ((2, 384, 4, 64), aten_fast.DType.bfloat16),
        ((2, 384, 4, 64), aten_fast.DType.bfloat16),
        ((2, 4, 384), aten_fast.DType.float32),
        ((2, 4, 384), aten_fast.DType.float32),
        ((2 * 4 * 384 * 64,), aten_fast.DType.float32),
    ]
    assert bridge_calls == [
        (
            11,
            22,
            33,
            77,
            88,
            55,
            1000,
            1001,
            1002,
            1003,
            1004,
            1005,
            2,
            384,
            4,
            0.125,
            8080,
        )
    ]


def test_fa4_canonical_fused_qkv_uses_strided_backward_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch_mojo_backend.eager_flash_attention as package

    device = _device()
    batch, seqlen, heads, head_dim = 2, 256, 12, 64
    token_stride = 3 * heads * head_dim
    qkv_strides = (seqlen * token_stride, token_stride, head_dim, 1)
    qkv_pointers = (0x1000, 0x1600, 0x1C00)
    q_native, k_native, v_native = tuple(
        _tensor(
            name,
            shape=(batch, seqlen, heads, head_dim),
            device=device,
            ptr=ptr,
            strides=qkv_strides,
            contiguous=False,
        )
        for name, ptr in zip("qkv", qkv_pointers, strict=True)
    )
    public_shape = (batch, heads, seqlen, head_dim)
    public_strides = (seqlen * heads * head_dim, head_dim, heads * head_dim, 1)
    output = _tensor(
        "output",
        shape=public_shape,
        device=device,
        ptr=0x3000,
        strides=public_strides,
        contiguous=False,
    )
    grad_output = _tensor(
        "grad",
        shape=public_shape,
        device=device,
        ptr=0x4000,
        strides=public_strides,
        contiguous=False,
    )
    logsumexp = _tensor(
        "lse",
        shape=(batch, heads, seqlen),
        dtype=aten_fast.DType.float32,
        device=device,
        ptr=0x5000,
    )
    allocations = []
    strided_calls = []

    def alloc(shape, dtype, actual_device):
        result = _tensor(
            f"alloc{len(allocations)}",
            shape=shape,
            dtype=dtype,
            device=actual_device,
            ptr=0x8000 + len(allocations) * 0x1000,
        )
        allocations.append(result)
        return result

    def transpose(tensor, dim0, dim1):
        shape = list(tensor._shape)
        strides = list(tensor._strides)
        shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
        strides[dim0], strides[dim1] = strides[dim1], strides[dim0]
        return _tensor(
            f"{tensor.name}_transpose",
            shape=shape,
            dtype=tensor._dtype,
            device=tensor._device,
            ptr=tensor._ptr,
            strides=strides,
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("strided QKV reached a copy or old FA4 bridge")

    monkeypatch.setattr(aten_fast, "_tc", forbidden)
    monkeypatch.setattr(aten_fast, "_alloc", alloc)
    monkeypatch.setattr(aten_fast, "fast_aten_transpose", transpose)
    monkeypatch.setattr(aten_fast, "_ctx_ptr", lambda actual_device: 6060)
    monkeypatch.setattr(
        package,
        "load_fa4_ops",
        lambda: SimpleNamespace(
            flash_attention_bwd_bf16_d64_causal=forbidden,
            flash_attention_bwd_bf16_d64_causal_strided_qkv=lambda *args: (
                strided_calls.append(args)
            ),
        ),
    )

    gradients = aten_fast.fast_fa4_bf16_d64_causal_backward(
        q_native, k_native, v_native, output, logsumexp, grad_output, 0.125
    )

    assert [gradient._shape for gradient in gradients] == [public_shape] * 3
    assert strided_calls == [
        (
            qkv_pointers[0],
            *qkv_strides,
            qkv_pointers[1],
            *qkv_strides,
            qkv_pointers[2],
            *qkv_strides,
            output._ptr,
            grad_output._ptr,
            logsumexp._ptr,
            allocations[0]._ptr,
            allocations[1]._ptr,
            allocations[2]._ptr,
            allocations[3]._ptr,
            allocations[4]._ptr,
            allocations[5]._ptr,
            batch,
            seqlen,
            heads,
            0.125,
            6060,
        )
    ]


def test_fa4_eligible_backward_routes_to_native_flash_with_public_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An eligible FA4 call that needs gradients goes to the lower ATen pair.

    The custom SDPA Function used to own the FA4 saves itself; PyTorch's
    generated autograd owns them now, so the eligibility gate must hand the
    *public* BHTD q/k/v to ``aten::_scaled_dot_product_flash_attention`` (the
    physical BTHD copies are made inside the kernel bridge and are not what
    gets saved) and neither custom Function may be entered.
    """
    public = tuple(_tensor(name, ptr=index) for index, name in enumerate("qkv", 1))
    for tensor in public:
        tensor.requires_grad = True
    output = _tensor("output", ptr=200)
    lse = _tensor("lse", shape=(2, 4, 128), dtype=aten_fast.DType.float32, ptr=201)
    eligible_with = []
    flash_calls = []

    def eligible(*args):
        eligible_with.append(args)
        return public

    def flash(*args, **kwargs):
        flash_calls.append((args, kwargs))
        return (output, lse)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("an eligible FA4 call built a custom autograd node")

    monkeypatch.setattr(aten_fast, "_fa4_bf16_d64_causal_inputs", eligible)
    monkeypatch.setattr(
        torch.ops.aten._scaled_dot_product_flash_attention, "default", flash
    )
    monkeypatch.setattr(autograd._ScaledDotProductAttentionAutograd, "apply", forbidden)
    monkeypatch.setattr(autograd._FusedFlashAttentionAutograd, "apply", forbidden)

    actual = autograd._scaled_dot_product_attention_autograd(
        *public, None, 0.0, True, scale=0.125
    )

    assert actual is output
    assert eligible_with == [(*public, None, 0.0, True, 0.125, False)]
    assert flash_calls == [((*public, 0.0, True, False), {"scale": 0.125})]


def test_vendored_fa4_sources_have_no_torch_cuda_or_internal_sync() -> None:
    source_dir = Path(aten_fast.__file__).parents[1] / "eager_flash_attention"
    source_by_name = {path.name: path.read_text() for path in source_dir.glob("*.mojo")}
    sources = "\n".join(source_by_name.values())
    assert "torch.cuda" not in sources
    assert "ctx.synchronize()" not in sources

    # All five launches share compiled code by specialization and raw context
    # identity.  B/S/H, pointers, descriptors, and launch grids stay runtime
    # values, so changing model or batch shapes does not force recompilation.
    launchers = (
        source_by_name["fa4_fwd_launch.mojo"] + source_by_name["fa4_bwd_launch.mojo"]
    )
    cache = source_by_name["fa4_launch_cache.mojo"]
    assert launchers.count("enqueue_fa4_cached[") == 5
    assert ".compile_function[" not in launchers
    assert "_CTX{context_identity}" in cache
