"""torch.export on the mojo device.

A model whose parameters, buffers and inputs live on the device exports
(strict and non-strict), and the exported program runs on the device through
`ep.module()` and matches the eager CPU model: as exported, after
`run_decompositions` (core ATen, inference IR, a custom table), with dynamic
shapes in every spelling `torch.export` accepts, with pytree and keyword
inputs, with buffer mutations, with `torch.cond`, from `draft_export`, and
after a `torch.export.save` / `torch.export.load` round trip.
"""

import io
import re
from collections.abc import Callable

import pytest
import torch
from _pytest.mark.structures import ParameterSet
from torch import nn
from torch.export import Dim
from torch.fx.node import Target
from torch.utils import _pytree as pytree

aten = torch.ops.aten


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(37, 64)
        self.fc2 = nn.Linear(64, 19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(nn.functional.gelu(self.fc1(x)))


class _ConvBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.bn(self.conv(x)))


class _RunningSum(nn.Module):
    """A registered buffer updated in place by forward."""

    total: torch.Tensor
    scale: torch.Tensor

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(11, 11)
        self.register_buffer("total", torch.zeros(11))
        self.register_buffer("scale", torch.linspace(0.5, 1.5, 11))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.total.add_(x.sum(0))
        return self.lin(x) * self.scale + self.total


class _Derived(nn.Module):
    """Two inputs whose batch sizes are tied: y has twice the rows of x."""

    def __init__(self):
        super().__init__()
        self.mlp = _MLP()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.mlp(x).repeat(2, 1) + self.mlp(y)


class _Pytree(nn.Module):
    """Dict and tuple inputs, a keyword-only tensor, a nested output."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(13, 13)

    def forward(
        self,
        feats: dict[str, torch.Tensor],
        pair: tuple[torch.Tensor, torch.Tensor],
        *,
        scale: torch.Tensor,
    ) -> dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        h = self.lin(feats["a"]) + feats["b"] * scale
        return {"h": h, "parts": (pair[0] @ pair[1], h.sum(-1))}


class _Cond(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(9, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cond(
            x.sum() > 0,
            lambda t: torch.sigmoid(self.lin(t)),
            lambda t: self.lin(t).cos() * 2,
            (x,),
        )


def _seeded(cls: type[nn.Module]) -> nn.Module:
    torch.manual_seed(0)
    return cls()


def _pair(cls: type[nn.Module], device: str) -> tuple[nn.Module, nn.Module]:
    """The same freshly-initialized model on CPU and on `device`."""
    return _seeded(cls), _seeded(cls).to(device)


def _input(*shape: int, seed: int = 1) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def _close(got, want):
    """Compare a device pytree to a CPU pytree, leaf by leaf."""
    got_leaves, got_spec = pytree.tree_flatten(got)
    want_leaves, want_spec = pytree.tree_flatten(want)
    assert got_spec == want_spec
    for g, w in zip(got_leaves, want_leaves, strict=True):
        assert g.device.type != "cpu", "the program returned a CPU tensor"
        torch.testing.assert_close(g.cpu(), w, atol=1e-4, rtol=1e-4)


def _assert_on(ep: torch.export.ExportedProgram, device: str):
    """Every lifted parameter / buffer / constant and every placeholder's
    fake value reports the mojo device."""
    want = torch.device(device)
    assert ep.state_dict, "the program lifted no parameters or buffers"
    for name, t in list(ep.state_dict.items()) + list(ep.constants.items()):
        if isinstance(t, torch.Tensor):
            assert t.device == want, f"{name} is on {t.device}"
    placeholders = [n for n in ep.graph.nodes if n.op == "placeholder"]
    assert placeholders
    for node in placeholders:
        val = node.meta.get("val")
        if isinstance(val, torch.Tensor):
            assert val.device == want, f"placeholder {node.name} is on {val.device}"


def _targets(ep: torch.export.ExportedProgram) -> set[Target]:
    return {n.target for n in ep.graph.nodes if n.op == "call_function"}


@pytest.mark.parametrize("strict", [False, True], ids=["nonstrict", "strict"])
def test_export_mlp(mojo_gpu, strict):
    cpu, model = _pair(_MLP, mojo_gpu)
    x = _input(23, 37)
    ep = torch.export.export(model, (x.to(mojo_gpu),), strict=strict)
    _assert_on(ep, mojo_gpu)
    sig = ep.graph_signature
    assert list(sig.parameters) == [name for name, _ in model.named_parameters()]
    assert list(sig.buffers) == []
    assert list(sig.user_inputs) == ["x"]
    assert len(sig.user_outputs) == 1
    _close(ep.module()(x.to(mojo_gpu)), cpu(x))


def _gelu_erf(x: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    return x * 0.5 * (1 + torch.erf(x * 0.7071067811865476))


def _linear_mm(
    x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None
) -> torch.Tensor:
    out = torch.mm(x, w.t())
    return out if b is None else out + b


@pytest.mark.parametrize("table", ["core_aten", "inference", "default", "custom"])
def test_run_decompositions(mojo_gpu, table):
    """`None` is the core ATen opset, `{}` the inference IR (functional, no
    decomposition), `default_decompositions()` spelled out, and a table
    whose gelu and linear entries are overridden by hand-written ones.

    `aten.linear` is CompositeImplicitAutograd with a mojo kernel of its
    own: the core ATen and custom cases are the regression tests of
    `monkeypatching.decompose_composite_ops_with_mojo_kernels_in_export`,
    without which the program kept `aten.linear` whatever the table said.
    The inference IR preserves it, as on CPU."""
    cpu, model = _pair(_MLP, mojo_gpu)
    x = _input(23, 37)
    ep = torch.export.export(model, (x.to(mojo_gpu),))
    if table == "core_aten":
        ep = ep.run_decompositions(None)
        assert aten.linear.default not in _targets(ep)
    elif table == "inference":
        ep = ep.run_decompositions({})
        assert aten.linear.default in _targets(ep)
    elif table == "default":
        ep = ep.run_decompositions(torch.export.default_decompositions())
        assert aten.linear.default not in _targets(ep)
    else:
        decomp = torch.export.default_decompositions()
        decomp[aten.gelu.default] = _gelu_erf
        decomp[aten.linear.default] = _linear_mm
        ep = ep.run_decompositions(decomp)
        targets = _targets(ep)
        assert aten.gelu.default not in targets
        assert aten.erf.default in targets
        assert aten.linear.default not in targets
        assert aten.addmm.default not in targets
        assert aten.mm.default in targets
    _assert_on(ep, mojo_gpu)
    _close(ep.module()(x.to(mojo_gpu)), cpu(x))


def _cases_per_op() -> list[ParameterSet]:
    """(fn, inputs, runs) for every CompositeImplicitAutograd op the mojo
    device has a kernel for (and isclose's equal_nan branch, mish_backward
    with mixed dtypes), plus F.interpolate's nearest modes. `runs` is False
    where the decomposed program needs an `aten::index.Tensor` with several
    index tensors, which the device declines: those still compare graphs
    and assert the decline, so supporting it flips them."""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(4, 6, generator=g)
    special = x.clone()
    special[0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
    close = special + torch.randn(4, 6, generator=g) * 1e-6
    close[1, ::2] += 1
    grad = torch.randn(4, 6, generator=g)
    weight, bias = torch.randn(5, 6, generator=g), torch.randn(5, generator=g)
    cases = [
        pytest.param(nn.functional.linear, (x, weight, bias), True, id="linear"),
        pytest.param(torch.isclose, (special, close), True, id="isclose"),
        pytest.param(
            lambda a, b: torch.isclose(a, b, equal_nan=True),
            (special, close),
            True,
            id="isclose_equal_nan",
        ),
        pytest.param(torch.isfinite, (special,), True, id="isfinite"),
        pytest.param(aten.silu_backward, (grad, x), True, id="silu_backward"),
        pytest.param(aten.mish_backward, (grad, x), True, id="mish_backward"),
        pytest.param(
            aten.mish_backward, (grad, x.half()), True, id="mish_backward_f32_f16"
        ),
    ]
    # The `.default` upsamplings are the CIA overloads with a mojo kernel;
    # they have a CPU kernel too, so they stay whole. F.interpolate goes
    # through the `.vec` overloads, which decompose to index ops.
    for rank, size in ((1, [7]), (2, [7, 9]), (3, [7, 9, 4])):
        t = torch.randn(2, 3, *(5, 4, 3)[:rank], generator=g)
        for name, mode in (
            (f"upsample_nearest{rank}d", "nearest"),
            (f"_upsample_nearest_exact{rank}d", "nearest-exact"),
        ):
            op = getattr(aten, name).default
            cases.append(
                pytest.param(
                    lambda a, op=op, size=size: op(a, size), (t,), True, id=name
                )
            )
            cases.append(
                pytest.param(
                    lambda a, size=size, mode=mode: nn.functional.interpolate(
                        a, size=size, mode=mode
                    ),
                    (t,),
                    (rank, mode) == (2, "nearest"),
                    id=f"interpolate_{mode}_{rank}d",
                )
            )
    return cases


class _Call(nn.Module):
    def __init__(self, fn: Callable[..., torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.fn(*args)


def _decomposed(fn: Callable[..., torch.Tensor], args: tuple[torch.Tensor, ...]):
    ep = torch.export.export(_Call(fn), args).run_decompositions()
    nodes = [n for n in ep.graph.nodes if n.op == "call_function"]
    return ep, [(n.target, n.meta["val"].dtype) for n in nodes]


@pytest.mark.parametrize(("fn", "args", "runs"), _cases_per_op())
def test_run_decompositions_per_op(mojo_gpu, fn, args, runs):
    """A mojo program decomposes to the very graph a CPU program does: the
    same ops with the same dtypes, in the same order. A CUDA program, when
    there is one, does too: the mojo registration must not change it. Ops
    with a CPU kernel (silu_backward, mish_backward, the upsamplings) stay
    whole; isclose's decomposition has in-place ops that must come out
    functional."""
    cpu_ep, want = _decomposed(fn, args)
    ep, got = _decomposed(fn, tuple(a.to(mojo_gpu) for a in args))
    assert got == want
    if torch.cuda.is_available():
        assert _decomposed(fn, tuple(a.cuda() for a in args))[1] == want
    want_out = cpu_ep.module()(*args)
    if not runs:
        with pytest.raises(NotImplementedError, match="index.Tensor"):
            ep.module()(*(a.to(mojo_gpu) for a in args))
        return
    out = ep.module()(*(a.to(mojo_gpu) for a in args))
    assert out.dtype == want_out.dtype
    torch.testing.assert_close(
        out.cpu(), want_out, atol=1e-3, rtol=1e-3, equal_nan=True
    )


@pytest.mark.parametrize("train", [False, True], ids=["eval", "train"])
def test_export_conv_batchnorm(mojo_gpu, train):
    """Eval mode reads the running statistics; train mode normalizes with
    the batch statistics and updates the running buffers, which the
    exported module writes back into its own buffers."""
    cpu, model = _pair(_ConvBN, mojo_gpu)
    for m in (cpu, model):
        m.train(train)
        # non-trivial running stats, so eval mode does not read identities
        m.get_buffer("bn.running_mean").copy_(torch.linspace(-0.3, 0.4, 8))
        m.get_buffer("bn.running_var").copy_(torch.linspace(0.5, 2.0, 8))
    x = _input(4, 3, 13, 11)
    ep = torch.export.export(model, (x.to(mojo_gpu),))
    _assert_on(ep, mojo_gpu)
    assert list(ep.graph_signature.buffers) == [
        "bn.running_mean",
        "bn.running_var",
        "bn.num_batches_tracked",
    ]
    module = ep.module()
    _close(module(x.to(mojo_gpu)), cpu(x))
    got = dict(module.named_buffers())
    for name, want in cpu.named_buffers():
        assert got[name].device == torch.device(mojo_gpu)
        torch.testing.assert_close(got[name].cpu(), want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("ir", ["training", "inference"])
def test_export_buffer_mutation(mojo_gpu, ir):
    """The in-place buffer update is visible on the exported module's
    buffer after each call. The inference IR lifts it into a buffer
    mutation output of the signature."""
    cpu, model = _pair(_RunningSum, mojo_gpu)
    ep = torch.export.export(model, (_input(5, 11).to(mojo_gpu),))
    if ir == "inference":
        ep = ep.run_decompositions({})
        assert set(ep.graph_signature.buffers_to_mutate.values()) == {"total"}
    _assert_on(ep, mojo_gpu)
    module = ep.module()
    for seed in (2, 3):
        x = _input(5, 11, seed=seed)
        _close(module(x.to(mojo_gpu)), cpu(x))
        total = dict(module.named_buffers())["total"]
        _close(total, cpu.total)


def _single_input_spec(form: str, model: nn.Module, x: torch.Tensor):
    batch = Dim("batch", min=2, max=512)
    if form == "dict":
        return {"x": {0: batch}}
    if form == "tuple":
        return ({0: batch},)
    if form == "auto":
        return {"x": {0: Dim.AUTO}}
    if form == "dynamic":
        return {"x": {0: Dim.DYNAMIC}}
    if form == "shapes_collection":
        shapes = torch.export.ShapesCollection()
        shapes[x] = {0: batch}
        return shapes.dynamic_shapes(model, (x,))
    assert form == "additional_inputs"
    extra = torch.export.AdditionalInputs()
    extra.add((torch.empty(3, 37, device=x.device),))
    extra.add((torch.empty(100, 37, device=x.device),))
    return extra.dynamic_shapes(model, (x,))


def _needs(api: str) -> pytest.MarkDecorator:
    """Skip on a torch that predates `torch.export.<api>`."""
    return pytest.mark.skipif(
        not hasattr(torch.export, api), reason=f"torch.export.{api} is missing"
    )


@pytest.mark.parametrize(
    "form",
    [
        "dict",
        "tuple",
        "auto",
        "dynamic",
        pytest.param("shapes_collection", marks=_needs("ShapesCollection")),
        pytest.param("additional_inputs", marks=_needs("AdditionalInputs")),
    ],
)
def test_export_dynamic_batch(mojo_gpu, form):
    cpu, model = _pair(_MLP, mojo_gpu)
    x = _input(8, 37).to(mojo_gpu)
    ep = torch.export.export(
        model, (x,), dynamic_shapes=_single_input_spec(form, model, x)
    )
    _assert_on(ep, mojo_gpu)
    assert len(ep.range_constraints) == 1
    (vr,) = ep.range_constraints.values()
    bounded = form in ("dict", "tuple", "shapes_collection")
    if bounded:
        assert (vr.lower, vr.upper) == (2, 512)
    else:
        assert vr.lower <= 3 and vr.upper >= 512
    module = ep.module()
    for n in (2, 3, 8, 357, 512):
        x = _input(n, 37, seed=n)
        out = module(x.to(mojo_gpu))
        assert out.shape == (n, 19)
        _close(out, cpu(x))
    if bounded:
        # Outside the declared range the device program behaves exactly
        # like the same program exported on CPU: past the upper bound both
        # reject the call; below the lower bound torch does not check a
        # size-1 batch (0/1 are never guarded), so both run it.
        cpu_module = torch.export.export(
            cpu, (_input(8, 37),), dynamic_shapes=({0: Dim("batch", min=2, max=512)},)
        ).module()
        with pytest.raises(AssertionError, match="<= 512") as on_cpu:
            cpu_module(_input(513, 37))
        with pytest.raises(type(on_cpu.value), match="<= 512"):
            module(_input(513, 37).to(mojo_gpu))
        x = _input(1, 37)
        _close(module(x.to(mojo_gpu)), cpu_module(x))


def test_export_derived_dim(mojo_gpu):
    cpu, model = _pair(_Derived, mojo_gpu)
    d = Dim("d", min=1, max=256)
    ep = torch.export.export(
        model,
        (_input(4, 37).to(mojo_gpu), _input(8, 37).to(mojo_gpu)),
        dynamic_shapes={"x": {0: d}, "y": {0: 2 * d}},
    )
    _assert_on(ep, mojo_gpu)
    # d itself and the derived 2 * d
    assert sorted((vr.lower, vr.upper) for vr in ep.range_constraints.values()) == [
        (1, 256),
        (2, 512),
    ]
    module = ep.module()
    for n in (1, 7, 256):
        x, y = _input(n, 37, seed=n), _input(2 * n, 37, seed=n + 1)
        _close(module(x.to(mojo_gpu), y.to(mojo_gpu)), cpu(x, y))
    # y must have exactly 2 * len(x) rows. One row still broadcasts against
    # the 2 * len(x) rows of the other operand, so only the exported guard
    # rejects it: the same error as the CPU program's.
    cpu_module = torch.export.export(
        cpu,
        (_input(4, 37), _input(8, 37)),
        dynamic_shapes={"x": {0: d}, "y": {0: 2 * d}},
    ).module()
    x, y = _input(3, 37), _input(1, 37)
    with pytest.raises(AssertionError, match="Guard failed") as on_cpu:
        cpu_module(x, y)
    with pytest.raises(AssertionError, match=re.escape(str(on_cpu.value))):
        module(x.to(mojo_gpu), y.to(mojo_gpu))


@pytest.mark.parametrize("strict", [False, True], ids=["nonstrict", "strict"])
def test_export_pytree_and_kwargs(mojo_gpu, strict):
    cpu, model = _pair(_Pytree, mojo_gpu)

    def inputs(
        device: str,
    ) -> tuple[
        tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]],
        dict[str, torch.Tensor],
    ]:
        feats = {"a": _input(6, 13, seed=2), "b": _input(6, 13, seed=3)}
        pair = (_input(4, 5, seed=4), _input(5, 3, seed=5))
        args = (
            {k: v.to(device) for k, v in feats.items()},
            tuple(t.to(device) for t in pair),
        )
        return args, {"scale": _input(13, seed=6).to(device)}

    args, kwargs = inputs(mojo_gpu)
    ep = torch.export.export(model, args, kwargs, strict=strict)
    _assert_on(ep, mojo_gpu)
    assert list(ep.graph_signature.user_inputs) == [
        "feats_a",
        "feats_b",
        "pair_0",
        "pair_1",
        "scale",
    ]
    cpu_args, cpu_kwargs = inputs("cpu")
    _close(ep.module()(*args, **kwargs), cpu(*cpu_args, **cpu_kwargs))


def test_export_cond(mojo_gpu):
    cpu, model = _pair(_Cond, mojo_gpu)
    ep = torch.export.export(model, (_input(5, 9).to(mojo_gpu),))
    _assert_on(ep, mojo_gpu)
    assert torch.ops.higher_order.cond in _targets(ep)
    module = ep.module()
    for sign in (1.0, -1.0):  # take both branches
        x = _input(5, 9).abs() * sign
        _close(module(x.to(mojo_gpu)), cpu(x))


@_needs("draft_export")
def test_draft_export(mojo_gpu):
    cpu, model = _pair(_ConvBN, mojo_gpu)
    cpu.eval()
    model.eval()
    x = _input(2, 3, 9, 9)
    ep = torch.export.draft_export(model, (x.to(mojo_gpu),))
    assert isinstance(ep, torch.export.ExportedProgram)
    _assert_on(ep, mojo_gpu)
    _close(ep.module()(x.to(mojo_gpu)), cpu(x))


@pytest.mark.parametrize("target", ["path", "bytesio"])
@pytest.mark.parametrize("program", ["static", "dynamic", "decomposed"])
def test_export_save_load(mojo_gpu, tmp_path, program, target):
    cpu, model = _pair(_ConvBN, mojo_gpu)
    cpu.eval()
    model.eval()
    x = _input(2, 3, 9, 9)
    dynamic = {"x": {0: Dim("batch", min=1, max=64)}} if program == "dynamic" else None
    ep = torch.export.export(model, (x.to(mojo_gpu),), dynamic_shapes=dynamic)
    if program == "decomposed":
        ep = ep.run_decompositions()
    f = tmp_path / "conv_bn.pt2" if target == "path" else io.BytesIO()
    torch.export.save(ep, f)
    if isinstance(f, io.BytesIO):
        f.seek(0)
    loaded = torch.export.load(f)
    _assert_on(loaded, mojo_gpu)
    assert loaded.range_constraints.keys() == ep.range_constraints.keys()
    module = loaded.module()
    for n in (2, 5) if program == "dynamic" else (2,):
        x = _input(n, 3, 9, 9, seed=n)
        _close(module(x.to(mojo_gpu)), cpu(x))
