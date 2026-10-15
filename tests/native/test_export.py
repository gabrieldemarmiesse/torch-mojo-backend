"""torch.export on the mojo device.

A model whose parameters, buffers and inputs live on the device exports
(strict and non-strict), and the exported program runs on the device through
`ep.module()` and matches the eager CPU model: as exported, after
`run_decompositions` (core ATen, inference IR, a custom table), with dynamic
shapes in every spelling `torch.export` accepts, with pytree and keyword
inputs, with buffer mutations, with `torch.cond`, from `draft_export`, and
after a `torch.export.save` / `torch.export.load` round trip.
"""

import contextlib
import io
import re
from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch._decomp import core_aten_decompositions
from torch.export import Dim
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.node import Target
from torch.utils import _python_dispatch as python_dispatch, _pytree as pytree

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


class _IscloseLinear(nn.Module):
    """The CIA ops the mojo device has kernels for and CPU has not."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(6, 6)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.lin(x), torch.isclose(x, y), torch.isfinite(x)


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


# Since torch 2.10 (pytorch#164939), tracing decomposes a
# CompositeImplicitAutograd op only when the device has no kernel of its own
# (`autograd_would_have_decomposed`), so that export and compile trace what
# eager runs; `run_decompositions` follows it (pytorch#171897 closed as
# expected). A mojo program therefore keeps the CIA ops the mojo device has
# kernels for and CPU has not, whatever the table says. Torch
# 2.7-2.9 decompose every CIA op they trace.
def _kept_whole_on_mojo() -> list[torch._ops.OpOverload]:
    """The CIA ops a mojo program keeps whole that a CPU program decomposes:
    those with a mojo kernel and no CPU kernel, read from the dispatcher
    (so only once the mojo device has registered its kernels: call it from
    a test). Empty before torch 2.10."""
    if not hasattr(python_dispatch, "autograd_would_have_decomposed"):
        return []
    has_kernel = torch._C._dispatch_has_kernel_for_dispatch_key
    key = torch._C.DispatchKey
    ops = []
    for name in torch._C._dispatch_get_registrations_for_dispatch_key("PrivateUse1"):
        if has_kernel(name, key.CompositeImplicitAutograd) and not has_kernel(
            name, key.CPU
        ):
            namespace, _, qualified = name.partition("::")
            op_name, _, overload = qualified.partition(".")
            packet = getattr(getattr(torch.ops, namespace), op_name)
            ops.append(getattr(packet, overload or "default"))
    return ops


# Torch < 2.10 registers no AutogradPrivateUse1 fallback (pytorch#165315
# added it), so a CompositeImplicitAutograd op with no autograd formula of its
# own has no AutogradPrivateUse1 kernel once the mojo device registers a kernel
# for it: isclose and isfinite on a mojo tensor raise NotImplementedError,
# eagerly and while tracing.
_NEEDS_AUTOGRAD_FALLBACK = pytest.mark.xfail(
    not torch._C._dispatch_has_backend_fallback(
        torch._C.DispatchKey.AutogradPrivateUse1
    ),
    reason="torch < 2.10 has no AutogradPrivateUse1 fallback: isclose / isfinite"
    " have a mojo kernel and no autograd kernel there",
    strict=True,
)


def _table(
    name: str, device: str = "cpu"
) -> dict[torch._ops.OperatorBase, Callable[..., object]] | None:
    """The decomposition table `name`, or, for `device` "mojo", the table
    that reproduces on CPU what upstream gives a mojo program: the same one
    with the ops the mojo device keeps whole deleted (so kept whole)."""
    if name == "inference":
        return {}
    decomp = torch.export.default_decompositions()
    if name == "preserve_linear":
        del decomp[aten.linear.default]
    elif name == "custom":
        decomp[aten.gelu.default] = _gelu_erf
        decomp[aten.linear.default] = _linear_mm
    else:
        assert name in ("core_aten", "default", "core_aten_inference_mode")
    if device == "mojo":
        for op in _kept_whole_on_mojo():
            decomp.pop(op, None)
    elif name == "core_aten":
        return None
    return decomp


def _ops(ep: torch.export.ExportedProgram) -> list[Target]:
    return [n.target for n in ep.graph.nodes if n.op == "call_function"]


@pytest.mark.parametrize(
    "table",
    [
        "core_aten",
        "inference",
        "default",
        "custom",
        "preserve_linear",
        "core_aten_inference_mode",
    ],
)
def test_run_decompositions(mojo_gpu, table):
    """`None` is the core ATen opset, `{}` the inference IR (functional, no
    decomposition), `default_decompositions()` spelled out, a table whose
    gelu and linear entries are overridden by hand-written ones, one with
    linear deleted (so kept whole), and core ATen under inference_mode.

    The mojo program's graph is upstream's: the CPU program's under the
    same table less the ops the mojo device keeps whole (`aten.linear`
    here, from torch 2.10). Its values are the CPU model's."""
    cpu, model = _pair(_MLP, mojo_gpu)
    x = _input(23, 37)
    mode = (
        torch.inference_mode()
        if table == "core_aten_inference_mode"
        else contextlib.nullcontext()
    )
    with mode:
        ep = torch.export.export(model, (x.to(mojo_gpu),))
        ep = ep.run_decompositions(_table(table))
        want = torch.export.export(cpu, (x,)).run_decompositions(_table(table, "mojo"))
    assert _ops(ep) == _ops(want)
    keeps_linear = (
        table in ("inference", "preserve_linear")
        or aten.linear.default in _kept_whole_on_mojo()
    )
    assert (aten.linear.default in _targets(ep)) == keeps_linear
    if table == "custom":
        assert aten.gelu.default not in _targets(ep)
        assert aten.erf.default in _targets(ep)
    _assert_on(ep, mojo_gpu)
    _close(ep.module()(x.to(mojo_gpu)), cpu(x))


@_NEEDS_AUTOGRAD_FALLBACK
def test_make_fx_decomposes_kept_ops(mojo_gpu):
    """The workaround pytorch#171897 points to: re-trace the program with
    make_fx and a decomposition table, then export and decompose again. A
    mojo program then reaches the CPU program's core ATen graph."""
    cpu, model = _pair(_IscloseLinear, mojo_gpu)
    args = (_input(4, 6, seed=2), _input(4, 6, seed=3))
    on_device = tuple(a.to(mojo_gpu) for a in args)
    ep = torch.export.export(model, on_device)
    table = {
        op: decomp
        for op, decomp in core_aten_decompositions().items()
        if isinstance(op, torch._ops.OpOverload)
    }
    gm = make_fx(ep.module(), decomposition_table=table)(*on_device)
    ep = torch.export.export(gm, on_device).run_decompositions()
    want = torch.export.export(cpu, args).run_decompositions()
    assert _ops(ep) == _ops(want)
    assert not set(_kept_whole_on_mojo()) & _targets(ep)
    _close(ep.module()(*on_device), cpu(*args))


def _cases_per_op():
    """(fn, inputs) for every CompositeImplicitAutograd op the mojo device
    has a kernel for (and isclose's equal_nan branch, mish_backward with
    mixed dtypes), plus the nearest upsamplings and F.interpolate's nearest
    modes."""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(4, 6, generator=g)
    special = x.clone()
    special[0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")])
    close = special + torch.randn(4, 6, generator=g) * 1e-6
    close[1, ::2] += 1
    grad = torch.randn(4, 6, generator=g)
    weight, bias = torch.randn(5, 6, generator=g), torch.randn(5, generator=g)
    needs_fallback = {"marks": _NEEDS_AUTOGRAD_FALLBACK}
    cases = [
        pytest.param(nn.functional.linear, (x, weight, bias), id="linear"),
        pytest.param(torch.isclose, (special, close), id="isclose", **needs_fallback),
        pytest.param(
            lambda a, b: torch.isclose(a, b, equal_nan=True),
            (special, close),
            id="isclose_equal_nan",
            **needs_fallback,
        ),
        pytest.param(torch.isfinite, (special,), id="isfinite", **needs_fallback),
        pytest.param(aten.silu_backward, (grad, x), id="silu_backward"),
        pytest.param(aten.mish_backward, (grad, x), id="mish_backward"),
        pytest.param(aten.mish_backward, (grad, x.half()), id="mish_backward_f32_f16"),
    ]
    # The `.default` upsamplings have a CPU and a mojo kernel (no CIA one),
    # so they stay whole from torch 2.10. F.interpolate goes through the
    # `.vec` overloads, CIA with no mojo kernel, which decompose to index ops.
    for rank, size in ((1, [7]), (2, [7, 9]), (3, [7, 9, 4])):
        t = torch.randn(2, 3, *(5, 4, 3)[:rank], generator=g)
        for name, mode in (
            (f"upsample_nearest{rank}d", "nearest"),
            (f"_upsample_nearest_exact{rank}d", "nearest-exact"),
        ):
            op = getattr(aten, name).default
            cases.append(
                pytest.param(lambda a, op=op, size=size: op(a, size), (t,), id=name)
            )
            cases.append(
                pytest.param(
                    lambda a, size=size, mode=mode: nn.functional.interpolate(
                        a, size=size, mode=mode
                    ),
                    (t,),
                    id=f"interpolate_{mode}_{rank}d",
                )
            )
    return cases


def _needs_multi_index(ep: torch.export.ExportedProgram) -> bool:
    """The program calls `aten.index.Tensor` with an index anywhere but on
    the leading dim alone, which the mojo device declines."""
    for node in ep.graph.nodes:
        if node.target is aten.index.Tensor:
            present = [i for i, t in enumerate(node.args[1]) if t is not None]
            if present != [0]:
                return True
    return False


class _Call(nn.Module):
    def __init__(self, fn: Callable[..., torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.fn(*args)


def _decomposed(
    fn: Callable[..., torch.Tensor],
    args: tuple[torch.Tensor, ...],
    table: dict[torch._ops.OperatorBase, Callable[..., object]] | None = None,
):
    ep = torch.export.export(_Call(fn), args).run_decompositions(table)
    nodes = [n for n in ep.graph.nodes if n.op == "call_function"]
    # Some nodes carry no value (torch 2.7's _assert_tensor_metadata):
    # those compare by target alone.
    return ep, [(n.target, getattr(n.meta.get("val"), "dtype", None)) for n in nodes]


@pytest.mark.parametrize(("fn", "args"), _cases_per_op())
def test_run_decompositions_per_op(mojo_gpu, fn, args):
    """Core ATen on a one-op program. The mojo program's graph (ops and
    dtypes, in order) is upstream's: the CPU program's with the ops the
    mojo device keeps whole left whole. A CUDA program, when there is one,
    gets the CPU program's graph: registering the mojo device does not
    change it. Values and dtypes match CPU; isclose stays whole on mojo, or
    decomposes functionally on torch < 2.10. A program decomposed to an
    `aten.index.Tensor` with several index tensors (the nearest upsamplings
    but 2d nearest; the `.default` ones only before torch 2.10) asserts the
    device's decline instead, so supporting it flips them."""
    cpu_ep, want = _decomposed(fn, args)
    ep, got = _decomposed(fn, tuple(a.to(mojo_gpu) for a in args))
    assert got == _decomposed(fn, args, _table("core_aten", "mojo"))[1]
    if torch.cuda.is_available():
        assert _decomposed(fn, tuple(a.cuda() for a in args))[1] == want
    want_out = cpu_ep.module()(*args)
    if _needs_multi_index(ep):
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
        # (AssertionError from the guards since torch 2.9, RuntimeError
        # from the input-constraint hook before)
        with pytest.raises((AssertionError, RuntimeError), match="512") as on_cpu:
            cpu_module(_input(513, 37))
        with pytest.raises(type(on_cpu.value), match=re.escape(str(on_cpu.value))):
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
    with pytest.raises((AssertionError, RuntimeError)) as on_cpu:
        cpu_module(x, y)
    with pytest.raises(type(on_cpu.value), match=re.escape(str(on_cpu.value))):
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
