"""End-to-end checks of the native backend's core: factories, transfers,
views, fills, item, autograd, streams, events, RNG (public torch API only)."""

import pytest
import torch
from torch._dynamo.source import ConstantSource
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

from tests.native.conftest import side_stream_or_skip
from torch_mojo_backend import native
from torch_mojo_backend.native import device_module


def _arange(n: int, device: str) -> torch.Tensor:
    return torch.arange(n, dtype=torch.float32).to(device)


def test_registration_and_devices(mojo_device):
    assert native.is_registered()
    # 0 is a legitimate count on a box with no accelerator: there is no
    # CPU-backed mojo device to fall back to any more.
    assert device_module.device_count() >= 0


def test_empty_to_and_back(mojo_device):
    x = torch.empty(2, 3, device=mojo_device)
    assert x.device.type == "mojo" and x.dtype == torch.float32 and x.is_contiguous()
    a = _arange(6, mojo_device).reshape(2, 3)
    assert a.cpu().tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_add_mul_views_item(mojo_device):
    a = _arange(6, mojo_device).reshape(2, 3)
    b = torch.full((2, 3), 2.0).to(mojo_device)
    torch.testing.assert_close((a + b).cpu(), a.cpu() + 2)
    torch.testing.assert_close((a * b).cpu(), a.cpu() * 2)
    assert a.view(6).cpu().tolist() == list(range(6))
    assert a.reshape(3, 2).cpu().tolist() == [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
    assert a[1, 2].item() == 5.0
    assert a.t().contiguous().cpu().tolist() == a.cpu().t().tolist()


def test_fills_and_strided_copies(mojo_device):
    assert torch.zeros(3, device=mojo_device).cpu().tolist() == [0.0] * 3
    assert torch.ones(2, dtype=torch.int64, device=mojo_device).cpu().tolist() == [1, 1]
    z = torch.empty(4, device=mojo_device)
    z.fill_(7.5)
    assert z.cpu().tolist() == [7.5] * 4
    v = torch.zeros(3, 4, device=mojo_device)
    v[:, 1] = 5.0
    assert v.cpu()[:, 1].tolist() == [5.0] * 3
    w = torch.zeros(3, 4, device=mojo_device)
    w[:, 1].copy_(_arange(3, mojo_device))
    assert w.cpu()[:, 1].tolist() == [0.0, 1.0, 2.0]


@pytest.mark.parametrize("contiguous", [True, False])
def test_fill_keeps_the_scalar_tag(mojo_device, contiguous: bool):
    """`fill_` takes an ATen Scalar, not a double: a bool destination is
    filled on nonzero truth, an int64 one keeps every bit past 2**53, and
    `-0.0` keeps its sign. Both the contiguous (memset) and the strided
    (kernel) routes."""

    def target(dtype: torch.dtype) -> torch.Tensor:
        if contiguous:
            return torch.empty(4, dtype=dtype, device=mojo_device)
        return torch.empty(4, 2, dtype=dtype, device=mojo_device)[:, 1]

    for value in (0.5, -0.5, 2, -3, True):
        b = target(torch.bool)
        b.fill_(value)
        assert b.cpu().tolist() == [bool(value)] * 4, value

    b = target(torch.bool)
    b.fill_(0)
    assert b.cpu().tolist() == [False] * 4

    f = target(torch.float32)
    f.fill_(-0.0)
    assert torch.signbit(f.cpu()).all(), f.cpu()
    f.fill_(0.0)
    assert not torch.signbit(f.cpu()).any(), f.cpu()


@pytest.mark.parametrize("contiguous", [True, False])
def test_fill_keeps_int64_bits_past_2_53(mojo_gpu: str, contiguous: bool):
    """An integer Scalar reaches an int64 destination exactly; a Float64
    round-trip would round it. Past 2**53 the fill goes through a dense
    temporary (memset) plus a strided copy rather than the kernel, which
    only takes a Float64."""
    big = 2**60 + 1
    if contiguous:
        i = torch.empty(4, dtype=torch.int64, device=mojo_gpu)
    else:
        i = torch.empty(4, 2, dtype=torch.int64, device=mojo_gpu)[:, 1]
    i.fill_(big)
    assert i.cpu().tolist() == [big] * 4


def test_copy_into_a_broadcast_shaped_view(mojo_device):
    """`copy_` may hand `_copy_from` a source of another logical shape with
    the same element count (`dst(2,3).copy_(src(1,2,3))`); the strided copy
    kernel walks one shape, so the source is viewed as the destination's."""
    src = _arange(6, mojo_device).reshape(1, 2, 3)
    dense = torch.empty(2, 3, device=mojo_device)
    dense.copy_(src)
    assert dense.cpu().tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    strided = torch.zeros(2, 6, device=mojo_device)[:, ::2]
    strided.copy_(src)
    assert strided.cpu().tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert strided.cpu().sum() == src.cpu().sum()


def test_out_keeps_a_matching_targets_strides_and_offset(mojo_device):
    """torch's `resize_output`: an `out=` whose logical shape already
    matches is written where it lives, offset included. A slice of a larger
    buffer must not be re-laid-out at offset 0 over the start of its base."""
    base = torch.zeros(12, device=mojo_device)
    out = base[4:8]
    torch.arange(4, dtype=torch.float32, out=out)
    assert out.cpu().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert base.cpu().tolist() == [0.0] * 4 + [0.0, 1.0, 2.0, 3.0] + [0.0] * 4

    cat_base = torch.zeros(12, device=mojo_device)
    cat_out = cat_base[4:8]
    torch.cat([_arange(2, mojo_device), _arange(2, mojo_device) + 10], out=cat_out)
    assert cat_out.cpu().tolist() == [0.0, 1.0, 10.0, 11.0]
    assert cat_base.cpu().tolist() == [0.0] * 4 + [0.0, 1.0, 10.0, 11.0] + [0.0] * 4

    # an out of the wrong shape is resized (and keeps its storage offset)
    grow = torch.empty(0, device=mojo_device)
    torch.arange(5, dtype=torch.float32, out=grow)
    assert grow.cpu().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    cat_grow = torch.empty(0, device=mojo_device)
    torch.cat([_arange(3, mojo_device)], out=cat_grow)
    assert cat_grow.cpu().tolist() == [0.0, 1.0, 2.0]


def test_dtype_cast(mojo_device):
    m = _arange(12, mojo_device).reshape(3, 4)
    torch.testing.assert_close(
        m.to(torch.bfloat16).float().cpu(), m.cpu().to(torch.bfloat16).float()
    )


def test_error_paths(mojo_device):
    a = _arange(6, mojo_device).reshape(2, 3)
    with pytest.raises(RuntimeError):
        a * _arange(5, mojo_device)


def test_autograd_uses_aten_formulas(mojo_device):
    x = _arange(4, mojo_device).requires_grad_()
    y = torch.full((4,), 3.0).to(mojo_device).requires_grad_()
    (x * y).backward(torch.ones(4).to(mojo_device))
    assert x.grad is not None and y.grad is not None
    assert x.grad.cpu().tolist() == [3.0] * 4
    assert y.grad.cpu().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_streams_and_events(mojo_gpu):
    s = side_stream_or_skip(mojo_gpu)
    assert s.device.type == "mojo"
    assert s.stream_id != torch.accelerator.current_stream().stream_id
    a = _arange(6, mojo_gpu)
    with device_module.stream(s):
        e1 = torch.Event(device=mojo_gpu, enable_timing=True)
        e1.record()
        _ = a * a
        e2 = torch.Event(device=mojo_gpu, enable_timing=True)
        e2.record()
        assert torch.accelerator.current_stream().stream_id == s.stream_id
    e2.synchronize()
    assert e1.elapsed_time(e2) >= 0.0
    assert e2.query()
    torch.accelerator.synchronize()


def test_rng_state_and_generator(mojo_device):
    device_module.manual_seed_all(123)
    state = device_module.get_rng_state()
    assert state.dtype == torch.uint8 and state.numel() == 16
    assert int.from_bytes(bytes(state.tolist()[:8]), "little") == 123
    g = torch.Generator(device=mojo_device)
    g.manual_seed(5)
    assert g.initial_seed() == 5
    device_module.set_rng_state(state)
    assert device_module.get_rng_state().tolist() == state.tolist()


def test_device_oom_is_not_disguised_as_unsupported(mojo_gpu):
    """An allocation the device cannot satisfy must surface as an OOM
    carrying the allocator's own message -- not as `NotImplementedError`
    ("unsupported dtype/shape"), which would send the reader looking for a
    missing kernel, and not silently at some later synchronize.
    """
    with pytest.raises(Exception) as excinfo:  # noqa: B017 -- the point is WHICH type
        torch.empty(2**44, dtype=torch.float64, device=mojo_gpu).fill_(1.0)
    assert not isinstance(excinfo.value, NotImplementedError), excinfo.value
    assert isinstance(excinfo.value, (torch.OutOfMemoryError, RuntimeError)), (
        type(excinfo.value),
        excinfo.value,
    )


def test_view_ops_metadata_and_aliasing(mojo_device):
    """view / _unsafe_view / _reshape_alias / as_strided: shape, stride,
    storage offset, one shared storage, writes visible through the base."""
    base = _arange(24, mojo_device).reshape(4, 6)
    for v in (
        base.view(2, 12),
        torch.ops.aten._unsafe_view(base, [2, 12]),
        torch.ops.aten._reshape_alias(base, [2, 12], [12, 1]),
        base.as_strided((2, 12), (12, 1)),
    ):
        assert v.shape == (2, 12) and v.stride() == (12, 1)
        assert v.storage_offset() == 0 and v.data_ptr() == base.data_ptr()
        assert v.dtype == base.dtype and v.device == base.device
    sub = base[1:3, 2:5]
    assert sub.shape == (2, 3) and sub.stride() == (6, 1) and sub.storage_offset() == 8
    strided = base.as_strided((3, 2), (2, 3), 5)
    assert strided.stride() == (2, 3) and strided.storage_offset() == 5
    assert strided.cpu().tolist() == [[5.0, 8.0], [7.0, 10.0], [9.0, 12.0]]
    view = base.view(24)
    view[0] = 99.0
    assert base.cpu()[0, 0].item() == 99.0


def test_empty_strided_metadata(mojo_device):
    """empty_strided keeps the strides it was given; empty_like and a
    channels-last request keep theirs."""
    t = torch.empty_strided((3, 4), (1, 3), dtype=torch.bfloat16, device=mojo_device)
    assert t.shape == (3, 4) and t.stride() == (1, 3) and t.dtype == torch.bfloat16
    assert not t.is_contiguous()
    like = torch.empty_like(t)
    assert like.shape == (3, 4) and like.stride() == (1, 3)
    cl = torch.empty(
        (2, 3, 4, 5), device=mojo_device, memory_format=torch.channels_last
    )
    assert cl.stride() == (60, 1, 15, 3)
    assert torch.empty((), device=mojo_device).shape == ()
    assert torch.empty((0, 3), device=mojo_device).numel() == 0


def test_boxed_adapter_returns_undefined_tensors_for_masked_gradients(mojo_gpu):
    """bool[] argument, three returns, and the None-record rule: a masked-off
    gradient of a `Tensor` (not `Tensor?`) return must come back as an
    UNDEFINED tensor, which torch shows as None -- not as an error.

    Accelerators only: the layer-norm backward kernel has no CPU route."""
    x = _arange(6, mojo_gpu).reshape(2, 3)
    w = torch.full((3,), 1.0).to(mojo_gpu)
    bias = torch.zeros(3).to(mojo_gpu)
    out, mean, rstd = torch.ops.aten.native_layer_norm(x, [3], w, bias, 1e-5)
    grad = torch.ones_like(out)
    full = torch.ops.aten.native_layer_norm_backward(
        grad, x, [3], mean, rstd, w, bias, [True, True, True]
    )
    assert all(t is not None for t in full)
    masked = torch.ops.aten.native_layer_norm_backward(
        grad, x, [3], mean, rstd, w, bias, [True, False, False]
    )
    assert masked[0] is not None and masked[1] is None and masked[2] is None


def test_boxed_adapter_carries_every_argument_kind(mojo_device):
    """One call per record kind the boxed adapter converts (shim_dispatch.cpp,
    the Conv codes), so every kind has a live call."""
    a = _arange(6, mojo_device).reshape(2, 3)
    b = torch.full((2, 3), 2.0).to(mojo_device)

    # Scalar: int, float, bool -- the three tags a c10::Scalar can carry here
    assert torch.add(a, 2).cpu().tolist() == [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]
    assert torch.add(a, 0.5).cpu()[0, 0].item() == 0.5
    assert torch.add(a, True).cpu()[0, 0].item() == 1.0
    # int[] (a view's size), int? left None (as_strided's storage_offset)
    assert a.view([6]).cpu().tolist() == list(range(6))
    assert a.sum(dim=[0, 1]).item() == 15.0
    assert a.as_strided((3, 2), (2, 1)).cpu().tolist() == [
        [0.0, 1.0],
        [2.0, 3.0],
        [4.0, 5.0],
    ]
    # ScalarType?, Device?, bool? (pin_memory), MemoryFormat?
    e = torch.empty(
        (2, 3),
        dtype=torch.int64,
        device=mojo_device,
        memory_format=torch.contiguous_format,
    )
    assert e.dtype == torch.int64 and e.is_contiguous()
    # str
    torch.testing.assert_close(
        torch.nn.functional.gelu(a, approximate="tanh").cpu(),
        torch.nn.functional.gelu(a.cpu(), approximate="tanh"),
    )
    torch.testing.assert_close(
        torch.div(a, b, rounding_mode="floor").cpu(), a.cpu() // 2
    )
    # Tensor? left None (layer_norm without affine parameters)
    torch.testing.assert_close(
        torch.nn.functional.layer_norm(a, (3,)).cpu(),
        torch.nn.functional.layer_norm(a.cpu(), (3,)),
    )
    # Tensor?[] (advanced indexing, leading index only) and Tensor[] in and out
    idx = torch.tensor([1, 0]).to(mojo_device)
    assert a[idx].cpu().tolist() == [[3.0, 4.0, 5.0], [0.0, 1.0, 2.0]]
    assert torch.cat([a, b]).cpu().shape == (4, 3)
    parts = a.view(6).split_with_sizes([2, 4])
    assert [p.cpu().tolist() for p in parts] == [[0.0, 1.0], [2.0, 3.0, 4.0, 5.0]]
    # Generator
    g = torch.Generator(device=mojo_device)
    g.manual_seed(7)
    r1 = torch.rand(4, device=mojo_device, generator=g)
    g.manual_seed(7)
    torch.testing.assert_close(
        r1.cpu(), torch.rand(4, device=mojo_device, generator=g).cpu()
    )
    # Scalar return (_local_scalar_dense)
    assert a[1, 2].item() == 5.0


def test_boxed_adapter_error_kinds(mojo_device):
    """A declined op is a NotImplementedError, a real failure a plain
    RuntimeError, and a message from the boxed adapter names the op it
    called. `view` no longer passes through the adapter (ATen's own kernel
    serves it since #482), so its bad-shape error is ATen's."""
    a = _arange(6, mojo_device).reshape(2, 3)
    with pytest.raises(NotImplementedError) as declined:
        torch.empty(2, dtype=torch.complex64, device=mojo_device)
    assert "ScalarType" in str(declined.value)
    assert "aten::empty.memory_format" in str(declined.value)

    with pytest.raises(RuntimeError) as failed:
        a.view([4, 4])
    assert not isinstance(failed.value, NotImplementedError), failed.value
    assert "invalid for input of size 6" in str(failed.value)

    # a C++-side check inside the shim, not a declining kernel
    with pytest.raises(RuntimeError) as bounds:
        torch.ops.aten.as_strided(a, [2, 3], [3, 1], 20)
    assert not isinstance(bounds.value, NotImplementedError), bounds.value
    assert "storage" in str(bounds.value)


def test_boxed_adapter_finds_a_warm_plan_by_value(mojo_device):
    """A warm op's conversion plan is accepted by the hot-path identity check,
    so no call after the first re-interns it (shim_dispatch.cpp, Plan)."""
    a = _arange(6, mojo_device)
    ops = (lambda: a.add(1.0), lambda: a.mul(2.0), lambda: a.view(2, 3), a.sum)
    for op in ops:
        op()
    before = native.plan_builds()
    assert before > 0
    for _ in range(20):
        for op in ops:
            op()
    assert native.plan_builds() == before


def test_view_ops_guard_a_backed_symbolic_size(mojo_device):
    """A backed symbolic size specializes at the view boundary (`guard_int`)
    rather than being rejected, as the boxed adapter did for the same
    argument."""
    env = ShapeEnv()
    six = env.create_symintnode(
        env.create_symbol(6, ConstantSource("v"), dynamic_dim=DimDynamic.DYNAMIC),
        hint=6,
    )
    t = _arange(6, mojo_device)
    assert t.view([six]).shape == (6,)
    assert t.view([six]).cpu().tolist() == list(range(6))
    strided = t.as_strided([six - 3], [2], six - 6)
    assert strided.shape == (3,) and strided.storage_offset() == 0
    assert torch.ops.aten._reshape_alias(t, [six], [1]).shape == (6,)
