"""Native backend: data_movement group (see docs/native_backend.md and
torch_mojo_backend/native/mojo/ops_data_movement.mojo).

Public-API checks only (no `TorchMojoTensor`/`aten_fast`/old-eager
internals): every op is exercised through ordinary `torch` calls on tensors
living on a `mojo` device, and `call_checker` confirms the native op (not
some other route) actually ran.
"""

import pytest
import torch

from torch_mojo_backend import aten_functions, get_accelerators, register_mojo_devices

# The `mojo_device` fixture (tests/conftest.py) yields a "mojo:N" string but,
# unlike `mojo_gpu`, never registers the backend itself -- it assumes some
# other test using the `conf` fixture ran first in the same session. Running
# this file on its own needs the same idempotent call `mojo_gpu` makes.
register_mojo_devices()


def _fill(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A deterministic, non-RNG input: consecutive elements differ, so a
    kernel that reads/writes one element off is caught by VALUE, not just by
    shape."""
    numel = 1
    for extent in shape:
        numel *= extent
    base = torch.arange(numel, dtype=torch.int64) % 251
    if dtype.is_floating_point:
        base = base.to(torch.float32) / 256.0
    return base.to(dtype).view(shape)


# ---------------------------------------------------------------------------
# clone
# ---------------------------------------------------------------------------


def test_clone_contiguous(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_clone)
    x = _fill((3, 4), torch.float32)
    dev = x.to(mojo_device)
    cloned = dev.clone()
    torch.testing.assert_close(cloned.cpu(), x)
    # An independent allocation: mutating one leaves the other untouched.
    # (fill_ rather than add_: add.out is a different group's op.)
    cloned.fill_(99.0)
    torch.testing.assert_close(dev.cpu(), x)


def test_clone_strided(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_clone)
    x = _fill((5, 7), torch.float32)
    dev = x.to(mojo_device).t()
    cloned = dev.clone()
    # preserve_format, and a transpose IS dense: torch keeps the strides.
    assert cloned.stride() == x.t().clone().stride()
    torch.testing.assert_close(cloned.cpu(), x.t())


@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5])
def test_clone_every_rank(mojo_gpu, rank):
    """rank<=4 takes the PermuteCopy fast path, rank>4 the general one."""
    shape = tuple(range(2, 2 + rank))
    x = _fill(shape, torch.bfloat16)
    dev = x.to(mojo_gpu).permute(*reversed(range(rank)))
    torch.testing.assert_close(dev.clone().cpu(), x.permute(*reversed(range(rank))))


# ---------------------------------------------------------------------------
# memory formats (empty.memory_format / clone / _to_copy)
# ---------------------------------------------------------------------------

# (shape, memory format): rank 4 for channels_last, rank 5 for
# channels_last_3d, plus the degenerate extents whose strides torch still
# spells out in full (a size-1 channel, an empty batch).
_MEMORY_FORMATS = [
    ((2, 3, 4, 5), torch.channels_last),
    ((3, 7, 5, 11), torch.channels_last),
    ((2, 1, 4, 5), torch.channels_last),
    ((0, 3, 4, 5), torch.channels_last),
    ((2, 3, 4, 5, 2), torch.channels_last_3d),
    ((2, 5, 3, 7, 3), torch.channels_last_3d),
]


def _memory_format_id(val):
    if isinstance(val, torch.memory_format):
        return str(val).removeprefix("torch.")
    return "x".join(str(d) for d in val)


@pytest.mark.parametrize("shape,memory_format", _MEMORY_FORMATS, ids=_memory_format_id)
def test_empty_memory_format_strides_match_cpu(mojo_device, shape, memory_format):
    made = torch.empty(shape, device=mojo_device, memory_format=memory_format)
    assert made.stride() == torch.empty(shape, memory_format=memory_format).stride()
    assert made.is_contiguous(memory_format=memory_format)


@pytest.mark.parametrize(
    "factory", [torch.empty_like, torch.zeros_like, torch.ones_like, torch.rand_like]
)
def test_like_factories_keep_the_memory_format(mojo_device, factory):
    """ATen's `*_like` composites resolve the format themselves and land on
    `empty.memory_format` / `empty_strided`."""
    x = torch.arange(120.0).reshape(2, 3, 4, 5).to(memory_format=torch.channels_last)
    dev = x.to(mojo_device)
    assert factory(dev).stride() == factory(x).stride()
    for memory_format in (
        torch.preserve_format,
        torch.contiguous_format,
        torch.channels_last,
    ):
        got = factory(dev, memory_format=memory_format)
        assert got.stride() == factory(x, memory_format=memory_format).stride()


@pytest.mark.parametrize("shape,memory_format", _MEMORY_FORMATS, ids=_memory_format_id)
def test_clone_memory_format_matches_cpu(
    mojo_device, shape, memory_format, call_checker
):
    call_checker.register(aten_functions.aten_clone)
    x = _fill(shape, torch.float32)
    expected = x.clone(memory_format=memory_format)
    got = x.to(mojo_device).clone(memory_format=memory_format)
    assert got.stride() == expected.stride()
    assert got.is_contiguous(memory_format=memory_format)
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("shape,memory_format", _MEMORY_FORMATS, ids=_memory_format_id)
def test_to_memory_format_matches_cpu(mojo_device, shape, memory_format, call_checker):
    call_checker.register(aten_functions.aten__to_copy)
    x = _fill(shape, torch.float32)
    expected = x.to(memory_format=memory_format)
    got = x.to(mojo_device).to(memory_format=memory_format)
    assert got.stride() == expected.stride()
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("shape,memory_format", _MEMORY_FORMATS, ids=_memory_format_id)
def test_contiguous_memory_format_matches_cpu(mojo_device, shape, memory_format):
    """`Tensor.contiguous(memory_format=...)` is a composite torch lowers to
    `clone(memory_format)` -- but only when the tensor is not already in that
    format, so it also covers the no-copy answer."""
    x = _fill(shape, torch.float32)
    dev = x.to(mojo_device)
    for tensor, reference in (
        (dev, x),
        (dev.contiguous(memory_format=memory_format), x),
    ):
        got = tensor.contiguous(memory_format=memory_format)
        expected = reference.contiguous(memory_format=memory_format)
        assert got.stride() == expected.stride()
        torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.int64])
def test_to_dtype_and_memory_format_matches_cpu(mojo_device, dtype):
    x = _fill((2, 3, 4, 5), torch.float32)
    expected = x.to(dtype, memory_format=torch.channels_last)
    got = x.to(mojo_device).to(dtype, memory_format=torch.channels_last)
    assert got.stride() == expected.stride()
    assert got.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(got.cpu(), expected)


def test_memory_format_from_a_permuted_input(mojo_device):
    """A non-contiguous source: the strided read has to feed the strided
    write, neither of them row-major."""
    x = _fill((2, 3, 4, 5), torch.float32).permute(0, 2, 3, 1)
    dev = _fill((2, 3, 4, 5), torch.float32).to(mojo_device).permute(0, 2, 3, 1)
    for expected, got in (
        (
            x.clone(memory_format=torch.channels_last),
            dev.clone(memory_format=torch.channels_last),
        ),
        (
            x.to(torch.float16, memory_format=torch.channels_last),
            dev.to(torch.float16, memory_format=torch.channels_last),
        ),
        (x.contiguous(), dev.contiguous()),
    ):
        assert got.stride() == expected.stride()
        torch.testing.assert_close(got.cpu(), expected)


def test_preserve_format_keeps_the_input_layout(mojo_device):
    """torch's preserve_format: a non-overlapping-and-dense input keeps its
    strides exactly, anything else gets `infer_dense_strides`."""
    x = _fill((2, 3, 4, 5), torch.float32)
    dev = x.to(mojo_device)
    cases = [
        (
            x.to(memory_format=torch.channels_last),
            dev.to(memory_format=torch.channels_last),
        ),
        (x.permute(0, 2, 3, 1), dev.permute(0, 2, 3, 1)),
        # Not dense: a strided slice of a transpose.
        (x.reshape(24, 5).t()[:, ::2], dev.reshape(24, 5).t()[:, ::2]),
    ]
    for expected_src, got_src in cases:
        for expected, got in (
            (expected_src.clone(), got_src.clone()),  # clone
            (expected_src.half(), got_src.half()),  # _to_copy
        ):
            assert got.stride() == expected.stride()
            torch.testing.assert_close(got.cpu(), expected)


def test_channels_last_survives_a_device_round_trip(mojo_device):
    x = _fill((2, 3, 4, 5), torch.float32).to(memory_format=torch.channels_last)
    dev = x.to(mojo_device)
    assert dev.stride() == x.stride()
    back = dev.cpu()
    assert back.stride() == x.stride()
    assert torch.equal(back, x)


def test_channels_last_survives_a_move_between_mojo_devices():
    """The cross-device leg stages a contiguous buffer and lays the layout out
    again on the destination."""
    if len(list(get_accelerators())) < 2:
        pytest.skip("needs two mojo devices")
    x = _fill((2, 3, 4, 5), torch.float32).to(memory_format=torch.channels_last)
    moved = x.to("mojo:0").to("mojo:1")
    assert moved.stride() == x.stride()
    torch.testing.assert_close(moved.cpu(), x)


@pytest.mark.parametrize(
    "shape,memory_format",
    [((2, 3, 4), torch.channels_last), ((2, 3, 4, 5), torch.channels_last_3d)],
    ids=_memory_format_id,
)
def test_memory_format_rank_is_checked_like_torch(mojo_device, shape, memory_format):
    x = _fill(shape, torch.float32)
    dev = x.to(mojo_device)
    calls = [
        lambda t: torch.empty(shape, device=t.device, memory_format=memory_format),
        lambda t: t.clone(memory_format=memory_format),
        lambda t: t.to(torch.float16, memory_format=memory_format),
        lambda t: t.contiguous(memory_format=memory_format),
    ]
    for call in calls:
        with pytest.raises(RuntimeError) as cpu_error:
            call(x)
        with pytest.raises(RuntimeError) as mojo_error:
            call(dev)
        assert str(cpu_error.value) in str(mojo_error.value)


def test_empty_rejects_preserve_format_like_torch(mojo_device):
    with pytest.raises(RuntimeError) as cpu_error:
        torch.empty(3, memory_format=torch.preserve_format)
    with pytest.raises(RuntimeError) as mojo_error:
        torch.empty(3, device=mojo_device, memory_format=torch.preserve_format)
    assert str(cpu_error.value) in str(mojo_error.value)


# ---------------------------------------------------------------------------
# _to_copy: dtype casts (same device) and device moves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float32, torch.int64),
        (torch.int64, torch.int32),
        (torch.int32, torch.uint8),
        (torch.uint8, torch.bool),
        # Exotic pairs outside the fast CastSpec kernel: the host round trip.
        (torch.float32, torch.float64),
        (torch.float64, torch.float32),
        (torch.float32, torch.int16),
        (torch.int16, torch.int8),
        (torch.uint8, torch.uint16),
    ],
)
def test_to_copy_dtype_cast(mojo_device, src_dtype, dst_dtype, call_checker):
    call_checker.register(aten_functions.aten__to_copy)
    x = _fill((3, 5), src_dtype if not src_dtype.is_floating_point else torch.float32)
    x = x.to(src_dtype)
    dev = x.to(mojo_device)
    torch.testing.assert_close(dev.to(dst_dtype).cpu(), x.to(dst_dtype))


def test_to_copy_always_returns_a_fresh_tensor(mojo_gpu):
    # `Tensor.to(dtype)` short-circuits to `self` in Python when nothing
    # would change, without ever reaching `_to_copy`; call the aten op
    # directly to exercise its own "always a fresh tensor" contract.
    x = torch.randn(4).to(mojo_gpu)
    same = torch.ops.aten._to_copy.default(x, dtype=torch.float32)
    assert same.data_ptr() != x.data_ptr()


def test_to_copy_device_round_trip(mojo_device):
    x = _fill((4, 6), torch.float32)
    dev = x.to(mojo_device)
    back = dev.to("cpu")
    torch.testing.assert_close(back, x)


# ---------------------------------------------------------------------------
# cat
# ---------------------------------------------------------------------------


def test_cat_skips_legacy_empty(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_cat)
    empty = torch.empty(0)
    x = torch.randn(1, 12, 6, 8)
    result = torch.cat([empty.to(mojo_device), x.to(mojo_device)], dim=-2)
    torch.testing.assert_close(result.cpu(), torch.cat([empty, x], dim=-2))


_CAT_CASES = [
    ("single input", [(1000,)], 0),
    ("two aligned", [(4096,), (4096,)], 0),
    ("three", [(777,), (777,), (777,)], 0),
    ("past one batch", [(311,)] * 70, 0),
    ("wildly unequal", [(1,), (7,), (4096,), (3,), (10000,)], 0),
    ("odd lengths", [(12345,), (7,), (999,)], 0),
    ("zero along dim", [(0, 5), (3, 5)], 0),
    ("3-D middle dim", [(5, 2, 7), (5, 3, 7), (5, 4, 7)], 1),
    ("3-D trailing dim", [(5, 6, 2), (5, 6, 3), (5, 6, 4)], 2),
    ("3-D negative dim", [(5, 6, 2), (5, 6, 3)], -1),
]


@pytest.mark.parametrize(
    "shapes,dim",
    [case[1:] for case in _CAT_CASES],
    ids=[case[0] for case in _CAT_CASES],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_cat_batched(mojo_device, shapes, dim, dtype):
    host = [torch.randn(shape).to(dtype) for shape in shapes]
    device = [x.to(mojo_device) for x in host]
    torch.testing.assert_close(
        torch.cat(device, dim).cpu(), torch.cat(host, dim), rtol=0, atol=0
    )


def test_cat_strided_inputs(mojo_gpu):
    """A non-contiguous input takes the per-input strided-view path (the
    batched kernel only ever sees contiguous inputs)."""
    host = [torch.randn(64, 32), torch.randn(64, 32)]
    device = [x.to(mojo_gpu).t() for x in host]
    torch.testing.assert_close(
        torch.cat(device, 0).cpu(), torch.cat([x.t() for x in host], 0), rtol=0, atol=0
    )


def test_cat_offset_views_and_legacy_empty(mojo_gpu):
    host = [torch.randn(2048) for _ in range(4)]
    device = [x.to(mojo_gpu) for x in host]
    torch.testing.assert_close(
        torch.cat([x[3:1000] for x in device]).cpu(),
        torch.cat([x[3:1000] for x in host]),
        rtol=0,
        atol=0,
    )
    empty = torch.empty(0)
    mid = [torch.randn(4, 8), empty, torch.randn(3, 8)]
    torch.testing.assert_close(
        torch.cat([x.to(mojo_gpu) for x in mid], 0).cpu(),
        torch.cat(mid, 0),
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# stack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_stack(mojo_device, dim, call_checker):
    call_checker.register(aten_functions.aten_stack)
    host = [torch.randn(6, 5) for _ in range(5)]
    device = [x.to(mojo_device) for x in host]
    torch.testing.assert_close(
        torch.stack(device, dim).cpu(), torch.stack(host, dim), rtol=0, atol=0
    )


def test_stack_many_inputs(mojo_gpu):
    host = [torch.randn(1024) for _ in range(33)]
    device = [x.to(mojo_gpu) for x in host]
    torch.testing.assert_close(
        torch.stack(device, 0).cpu(), torch.stack(host, 0), rtol=0, atol=0
    )


# ---------------------------------------------------------------------------
# repeat
# ---------------------------------------------------------------------------

_REPEAT_CASES = [
    (357, 789, (2, 3)),
    (13, 7, (3, 5)),
    (64, 1024, (3, 2)),
    (1024, 64, (1, 16)),
    (33, 33, (2, 2)),
    (1, 100, (5, 7)),
    (17, 31, (1, 1)),
    (3, 4, (2, 3, 5)),
    (5, 6, (3, 1, 1)),
    (1, 3, (2000, 1)),
]
_REPEAT_IDS = [
    f"{r}x{c}_r{'x'.join(str(k) for k in reps)}" for r, c, reps in _REPEAT_CASES
]


@pytest.mark.parametrize("rows,cols,reps", _REPEAT_CASES, ids=_REPEAT_IDS)
def test_repeat_matches_torch(mojo_device, rows, cols, reps, call_checker):
    call_checker.register(aten_functions.aten_repeat)
    x = _fill((rows, cols), torch.float32)
    torch.testing.assert_close(
        x.to(mojo_device).repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int64, torch.uint8]
)
@pytest.mark.parametrize("rows,cols,reps", [(357, 789, (2, 3)), (100, 8, (1, 5))])
def test_repeat_every_element_size(mojo_gpu, dtype, rows, cols, reps):
    x = _fill((rows, cols), dtype)
    torch.testing.assert_close(
        x.to(mojo_gpu).repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
    )


@pytest.mark.parametrize("offset", [1, 2, 3, 4])
def test_repeat_offset_views_are_not_assumed_aligned(mojo_gpu, offset):
    rows, cols, reps = 357, 789, (2, 3)
    base = _fill((rows * cols + 4,), torch.float32)
    x = base[offset : offset + rows * cols].view(rows, cols)
    device = base.to(mojo_gpu)[offset : offset + rows * cols].view(rows, cols)
    torch.testing.assert_close(
        device.repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
    )


def test_repeat_degenerate_extents(mojo_device):
    """A zero repeat factor or a zero input extent is an empty output, not a
    launch and not a crash. `repeat(x, [])` on a 0-d tensor is a legal 0-d
    copy with no last dim to tile along."""
    for shape, reps in (((4, 5), (0, 2)), ((4, 5), (2, 0)), ((0, 5), (2, 3))):
        x = _fill(shape, torch.float32)
        torch.testing.assert_close(
            x.to(mojo_device).repeat(*reps).cpu(), x.repeat(*reps), rtol=0, atol=0
        )
    scalar = torch.tensor(1.25)
    torch.testing.assert_close(
        torch.ops.aten.repeat(scalar.to(mojo_device), []).cpu(),
        torch.ops.aten.repeat(scalar, []),
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# tril / triu
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("shape", [(5, 5), (357, 41), (2, 6, 4)])
def test_tril(mojo_device, shape, diagonal, call_checker):
    call_checker.register(aten_functions.aten_tril)
    x = _fill(shape, torch.float32)
    torch.testing.assert_close(x.to(mojo_device).tril(diagonal).cpu(), x.tril(diagonal))


@pytest.mark.parametrize("diagonal", [-2, -1, 0, 1, 2])
@pytest.mark.parametrize("shape", [(5, 5), (41, 357), (2, 6, 4)])
def test_triu(mojo_device, shape, diagonal, call_checker):
    call_checker.register(aten_functions.aten_triu)
    x = _fill(shape, torch.float32)
    torch.testing.assert_close(x.to(mojo_device).triu(diagonal).cpu(), x.triu(diagonal))


def test_triu_every_dtype(mojo_gpu):
    for dtype in (torch.bfloat16, torch.int64, torch.uint8, torch.bool):
        x = (_fill((6, 6), torch.int64) % 2).to(dtype)
        torch.testing.assert_close(x.to(mojo_gpu).triu(1).cpu(), x.triu(1))


# ---------------------------------------------------------------------------
# select_scatter
# ---------------------------------------------------------------------------


def test_select_scatter_basic(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_select_scatter)
    a = _fill((4, 5), torch.float32)
    src = torch.full((5,), -1.0)
    expected = a.select_scatter(src, 0, 2)
    dev = a.to(mojo_device).select_scatter(src.to(mojo_device), 0, 2)
    torch.testing.assert_close(dev.cpu(), expected)
    # `a` itself is untouched (select_scatter is functional).
    torch.testing.assert_close(a, _fill((4, 5), torch.float32))


def test_select_scatter_negative_dim_and_index(mojo_device):
    a = _fill((3, 4, 5), torch.float32)
    src = torch.full((3, 5), 7.0)
    expected = a.select_scatter(src, -2, -1)
    dev = a.to(mojo_device).select_scatter(src.to(mojo_device), -2, -1)
    torch.testing.assert_close(dev.cpu(), expected)


def test_select_scatter_casts_src(mojo_gpu):
    # float16 (not float64): the fast CastSpec kernel's dtype set is what
    # select_scatter's src-cast uses, matching the old eager path's
    # `_cast_tensor` (pre-gated on the same set, never a host round trip).
    a = _fill((4, 5), torch.float32)
    src = torch.full((5,), 9.0, dtype=torch.float16)
    expected = a.select_scatter(src, 0, 1)
    dev = a.to(mojo_gpu).select_scatter(src.to(mojo_gpu), 0, 1)
    torch.testing.assert_close(dev.cpu(), expected)


def test_select_scatter_rejects_a_src_of_the_wrong_shape(mojo_gpu):
    """`select_scatter_symint` checks `slice.sizes() == src.sizes()`: it does
    not broadcast, and neither may this backend."""
    a = _fill((4, 5), torch.float32).to(mojo_gpu)
    src = torch.tensor(9.0, device=mojo_gpu)  # 0-d against a (5,) slice
    with pytest.raises(RuntimeError, match="size equal to the slice"):
        a.select_scatter(src, 0, 1)


# ---------------------------------------------------------------------------
# scatter.src / scatter.value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_scatter_src(mojo_device, dim, call_checker):
    call_checker.register(aten_functions.aten_scatter_src)
    a = _fill((4, 5), torch.float32)
    # Indices with no duplicate destination within a scatter group: torch's
    # own semantics call a colliding scatter "nondeterministic" (whichever
    # write lands last wins), so a randint index could legitimately disagree
    # with the cpu reference by write-order alone -- that is not a kernel bug.
    # Each row (dim=1/-1) or column (dim=0) is instead a distinct sub-permutation.
    if dim == 0:
        index = torch.stack([torch.randperm(4)[:3] for _ in range(5)], dim=1)
    elif dim == 1:
        index = torch.stack([torch.randperm(5)[:4] for _ in range(3)], dim=0)
    else:
        index = torch.stack([torch.randperm(5)[:3] for _ in range(4)], dim=0)
    index = index.to(torch.int64)
    src = _fill(tuple(index.shape), torch.float32) + 100
    expected = a.scatter(dim, index, src)
    dev = a.to(mojo_device).scatter(dim, index.to(mojo_device), src.to(mojo_device))
    torch.testing.assert_close(dev.cpu(), expected)


def test_scatter_value(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_scatter_value)
    a = _fill((4, 5), torch.float32)
    index = torch.randint(0, 4, (2, 5)).to(torch.int64)
    expected = a.scatter(0, index, -3.5)
    dev = a.to(mojo_device).scatter(0, index.to(mojo_device), -3.5)
    torch.testing.assert_close(dev.cpu(), expected)


def test_scatter_value_bool(mojo_gpu):
    a = torch.zeros(4, 5, dtype=torch.bool)
    index = torch.randint(0, 4, (2, 5)).to(torch.int64)
    expected = a.scatter(0, index, True)
    dev = a.to(mojo_gpu).scatter(0, index.to(mojo_gpu), True)
    assert dev.cpu().tolist() == expected.tolist()


def test_scatter_rejects_an_out_of_range_index(mojo_gpu):
    """The kernel skips the write rather than scribbling outside the tensor,
    and raises the flag the host reads back after the launch."""
    a = torch.zeros(4, 5, device=mojo_gpu)
    src = torch.ones(2, 5, device=mojo_gpu)
    for bad in (4, -1, 1 << 40):
        index = torch.zeros(2, 5, dtype=torch.int64, device=mojo_gpu)
        index[0, 0] = bad
        with pytest.raises(RuntimeError, match="index out of range"):
            a.scatter(0, index, src)
    with pytest.raises(RuntimeError, match="index out of range"):
        index = torch.full((2, 5), 9, dtype=torch.int64, device=mojo_gpu)
        a.scatter(0, index, -3.5)
    # A valid scatter still works after the flagged one (the flag is fresh
    # per launch, not a sticky per-device bit).
    ok = torch.zeros(2, 5, dtype=torch.int64, device=mojo_gpu)
    torch.testing.assert_close(
        a.scatter(0, ok, src).cpu(), torch.zeros(4, 5).scatter(0, ok.cpu(), src.cpu())
    )


def test_scatter_rejects_an_index_bigger_than_self(mojo_gpu):
    """ATen's `scatter_shape_check`: index.size(d) <= self.size(d) off `dim`,
    and index.size(d) <= src.size(d) everywhere."""
    a = torch.zeros(4, 5, device=mojo_gpu)
    index = torch.zeros(2, 9, dtype=torch.int64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="smaller than self"):
        a.scatter(0, index, 1.0)
    index = torch.zeros(2, 5, dtype=torch.int64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="smaller than src"):
        a.scatter(0, index, torch.ones(1, 5, device=mojo_gpu))


def test_index_tensor_out_of_range_index_stays_in_bounds(mojo_gpu):
    """An out-of-range gather index is clamped in the kernel: the read stays
    inside the table, so the process survives a bad index instead of reading
    device memory it does not own."""
    table = _fill((4, 3), torch.float32).to(mojo_gpu)
    idx = torch.tensor([0, 99, -99], dtype=torch.int64, device=mojo_gpu)
    out = table[idx]
    assert tuple(out.shape) == (3, 3)
    torch.testing.assert_close(out[0].cpu(), table[0].cpu())
    # The clamped rows are unspecified in value but must be real table rows.
    rows = table.cpu().tolist()
    assert out[1].cpu().tolist() in rows
    assert out[2].cpu().tolist() in rows


def test_scatter_rejects_rank_beyond_4(mojo_gpu):
    a = torch.zeros(2, 2, 2, 2, 2).to(mojo_gpu)
    index = torch.zeros(2, 2, 2, 2, 2, dtype=torch.int64).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        a.scatter(0, index, 1.0)


# ---------------------------------------------------------------------------
# index.Tensor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
def test_index_tensor_gather_rows(mojo_device, idx_dtype, call_checker):
    call_checker.register(aten_functions.aten_index)
    x = _fill((10, 3, 4), torch.float32)
    idx = torch.tensor([3, 0, 7, 7], dtype=idx_dtype)
    expected = x[idx]
    dev = x.to(mojo_device)[idx.to(mojo_device)]
    torch.testing.assert_close(dev.cpu(), expected)


def test_index_tensor_negative_indices(mojo_gpu):
    x = _fill((6, 4), torch.float32)
    idx = torch.tensor([-1, -2, 0], dtype=torch.int64)
    expected = x[idx]
    dev = x.to(mojo_gpu)[idx.to(mojo_gpu)]
    torch.testing.assert_close(dev.cpu(), expected)


def test_index_tensor_bool_mask_full_rank(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_index)
    x = _fill((4, 5), torch.float32)
    mask = _fill((4, 5), torch.int64) % 3 == 0
    expected = x[mask]
    dev = x.to(mojo_device)[mask.to(mojo_device)]
    torch.testing.assert_close(dev.cpu(), expected)


def test_index_tensor_bool_mask_partial_rank(mojo_gpu):
    x = _fill((4, 5, 3), torch.float32)
    mask = torch.tensor([True, False, True, False])
    expected = x[mask]
    dev = x.to(mojo_gpu)[mask.to(mojo_gpu)]
    torch.testing.assert_close(dev.cpu(), expected)


def test_index_tensor_bool_mask_all_false(mojo_gpu):
    x = _fill((4, 5), torch.float32)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    expected = x[mask]
    dev = x.to(mojo_gpu)[mask.to(mojo_gpu)]
    assert dev.cpu().shape == expected.shape


# ---------------------------------------------------------------------------
# nonzero
# ---------------------------------------------------------------------------


def test_nonzero_basic(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_nonzero)
    x = torch.tensor([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=torch.float32)
    torch.testing.assert_close(x.to(mojo_device).nonzero().cpu(), x.nonzero())


def test_nonzero_all_zeros(mojo_device):
    x = torch.zeros(3, 3, dtype=torch.float32)
    torch.testing.assert_close(x.to(mojo_device).nonzero().cpu(), x.nonzero())


def test_nonzero_all_nonzero(mojo_device):
    x = torch.ones(2, 3, dtype=torch.float32)
    torch.testing.assert_close(x.to(mojo_device).nonzero().cpu(), x.nonzero())


@pytest.mark.parametrize("shape", [(2,), (3, 4), (2, 3, 4)])
def test_nonzero_shapes(mojo_gpu, shape):
    x = _fill(shape, torch.float32)
    x = x * (x > 0.5)  # scatter in some real zeros
    torch.testing.assert_close(x.to(mojo_gpu).nonzero().cpu(), x.nonzero())


@pytest.mark.parametrize("value", [0, 1])
def test_nonzero_scalar_has_no_coordinate_column(mojo_gpu, value):
    """A 0-d tensor has no coordinates: ATen reports (n, 0), not (n, 1)."""
    want = torch.nonzero(torch.tensor(value))
    got = torch.nonzero(torch.tensor(value, device=mojo_gpu))
    assert tuple(got.shape) == tuple(want.shape)
    assert got.cpu().tolist() == want.tolist()


def test_cat_out_keeps_a_matching_out_where_it_is(mojo_gpu):
    """`cat.out` must not resize an out that already has the right shape:
    resizing resets sizes, strides AND offset, which would send the result to
    the front of the base storage instead of into the caller's view."""
    base = torch.zeros(16, device=mojo_gpu)
    out = base[4:8]
    parts = [torch.ones(2, device=mojo_gpu), torch.full((2,), 2.0, device=mojo_gpu)]
    torch.cat(parts, 0, out=out)
    assert out.cpu().tolist() == [1.0, 1.0, 2.0, 2.0]
    assert base.cpu().tolist() == [0.0] * 4 + [1.0, 1.0, 2.0, 2.0] + [0.0] * 8


def test_cat_out_resizes_a_mismatching_out(mojo_gpu):
    out = torch.empty(0, device=mojo_gpu)
    parts = [torch.ones(2, device=mojo_gpu), torch.full((3,), 2.0, device=mojo_gpu)]
    torch.cat(parts, 0, out=out)
    assert out.cpu().tolist() == [1.0, 1.0, 2.0, 2.0, 2.0]


def test_nonzero_int_dtype(mojo_gpu):
    x = torch.tensor([1, 0, 3, 0, 5], dtype=torch.int64)
    torch.testing.assert_close(x.to(mojo_gpu).nonzero().cpu(), x.nonzero())


# ---------------------------------------------------------------------------
# set_.source_Tensor
# ---------------------------------------------------------------------------


def test_set_source_tensor_adopts_the_allocation(mojo_gpu):
    # Built on the host and uploaded (torch.arange(..., device=mojo) is the
    # factories group's op, not this one's) rather than through arange/add_
    # directly on the mojo device.
    destination = torch.zeros(8, device=mojo_gpu)
    source = (torch.arange(4, dtype=torch.float32) + 50).to(mojo_gpu)
    returned = destination.set_(source)  # ty: ignore[invalid-argument-type]
    assert returned is destination
    assert tuple(destination.shape) == (4,)
    assert destination.cpu().tolist() == [50.0, 51.0, 52.0, 53.0]
    # Sharing the allocation, not a copy of it.
    source.fill_(99.0)
    assert destination.cpu().tolist() == [99.0] * 4


def test_set_source_tensor_keeps_its_own_dtype(mojo_gpu):
    """Matches upstream `set_tensor_`'s `set_storage_keep_dtype`: self's own
    dtype survives, only storage/sizes/strides move -- a raw bit
    reinterpretation, same as real ATen. `Tensor.set_`'s Python method adds
    its own dtype-equality check ahead of the dispatcher for this overload,
    so this goes through the aten op directly, the way FSDP1's C++-side caller
    does."""
    destination = torch.zeros(4, dtype=torch.int32, device=mojo_gpu)
    source = torch.arange(4, dtype=torch.float32).to(mojo_gpu)
    torch.ops.aten.set_.source_Tensor(destination, source)
    assert destination.dtype == torch.int32
    assert destination.cpu().tolist() == [0, 1065353216, 1073741824, 1077936128]


# ---------------------------------------------------------------------------
# empty_permuted
# ---------------------------------------------------------------------------


def test_empty_permuted_shape_dtype_device(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_empty_permuted)
    out = torch.ops.aten.empty_permuted(
        [2, 3, 4], [1, 0, 2], dtype=torch.float16, device=mojo_device
    )
    assert tuple(out.shape) == (2, 3, 4)
    assert out.dtype == torch.float16
    assert out.device.type == "mojo"
    # physical_layout is the whole point of the op: dim 1 is outermost, then
    # dim 0, then dim 2 -- the same strides CPU torch produces.
    reference = torch.ops.aten.empty_permuted([2, 3, 4], [1, 0, 2])
    assert out.stride() == reference.stride()
    assert not out.is_contiguous()
    assert out.permute(1, 0, 2).is_contiguous()


@pytest.mark.parametrize(
    "layout", [[0, 1, 2], [2, 1, 0], [1, 2, 0], [0, 2, 1], [2, 0, 1]]
)
def test_empty_permuted_every_layout_matches_cpu(mojo_gpu, layout):
    out = torch.ops.aten.empty_permuted([2, 3, 4], layout, device=mojo_gpu)
    reference = torch.ops.aten.empty_permuted([2, 3, 4], layout)
    assert out.stride() == reference.stride()
    assert out.is_contiguous() == reference.is_contiguous()


def test_empty_permuted_rejects_a_bad_layout(mojo_gpu):
    with pytest.raises(RuntimeError, match="Duplicate dim"):
        torch.ops.aten.empty_permuted([2, 3], [0, 0], device=mojo_gpu)
    with pytest.raises(RuntimeError, match="physical_layout"):
        torch.ops.aten.empty_permuted([2, 3], [0], device=mojo_gpu)


# ---------------------------------------------------------------------------
# Cast exactness across every dtype pair, length and storage offset.
#
# `test_to_copy_dtype_cast` above is 11 fixed pairs at shape (3,5) from
# offset 0. The values here are `arange % 5`, whose period is coprime with
# every power-of-two vector width, so a rotated lane or an unwritten tail
# cannot pass; every comparison is exact.
# ---------------------------------------------------------------------------

_CAST_DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.uint8,
    torch.bool,
]


@pytest.mark.parametrize("src_dtype", _CAST_DTYPES)
def test_cast_is_exact_for_every_dtype_pair(mojo_gpu, src_dtype):
    for dst_dtype in _CAST_DTYPES:
        for numel in (1, 3, 17, 1027, 4099):
            for offset in (0, 1, 2, 3):
                base = torch.arange(numel + offset) % 5
                src_cpu = (base != 0) if src_dtype == torch.bool else base.to(src_dtype)
                view_cpu = src_cpu[offset : offset + numel]
                view = src_cpu.to(mojo_gpu)[offset : offset + numel]
                assert view.is_contiguous()
                assert torch.equal(view.to(dst_dtype).cpu(), view_cpu.to(dst_dtype)), (
                    src_dtype,
                    dst_dtype,
                    numel,
                    offset,
                )


@pytest.mark.parametrize("dst_dtype", [torch.bfloat16, torch.float16])
def test_float_narrowing_rounds_like_cpu(mojo_gpu, dst_dtype):
    """Rounding mode, not just range: 65_539 values at three base alignments,
    down and back up, exactly equal to CPU."""
    for numel in (1027, 65_539):
        for offset in (0, 1, 3):
            src_cpu = torch.randn(numel + offset, dtype=torch.float32) * 8.0
            view_cpu = src_cpu[offset:]
            view = src_cpu.to(mojo_gpu)[offset:]
            narrowed = view.to(dst_dtype)
            assert torch.equal(narrowed.cpu(), view_cpu.to(dst_dtype)), (numel, offset)
            assert torch.equal(
                narrowed.to(torch.float32).cpu(), view_cpu.to(dst_dtype).float()
            )


# ---------------------------------------------------------------------------
# cat / repeat / stack batching edges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
@pytest.mark.parametrize(
    "shapes,dim",
    [
        ([(5000,)] * 64, 0),  # exactly the per-launch segment cap
        ([(37,)] * 130, 0),  # two batches past it
        ([(2, 3, 5, 64), (2, 3, 1, 64), (2, 3, 9, 64)], 2),  # 4-D middle dim
        ([(3, 4), (100_000, 4), (1, 4)], 0),  # wildly unequal members
    ],
)
def test_cat_batching_edges(mojo_gpu, shapes, dim, dtype):
    parts = [_fill(shape, dtype) for shape in shapes]
    expected = torch.cat(parts, dim=dim)
    actual = torch.cat([p.to(mojo_gpu) for p in parts], dim=dim)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_cat_mixed_contiguity_in_one_list(mojo_gpu):
    """One launch must not assume every member has the same layout."""
    a = _fill((32, 64), torch.float32)
    b = _fill((64, 32), torch.float32)
    c = _fill((32, 64), torch.float32)
    expected = torch.cat([a, b.t(), c], dim=0)
    actual = torch.cat([a.to(mojo_gpu), b.to(mojo_gpu).t(), c.to(mojo_gpu)], dim=0)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,reps",
    [
        ((100, 1), (3, 7)),  # a single input column
        ((7, 128), (4, 1, 2)),  # more reps than input dims
        ((2, 3), (1000, 1)),
        ((1, 5000), (5000, 1)),
        ((1, 64), (1, 300)),
        ((64,), (5,)),  # rank-1 input
        ((64,), (2, 5)),  # rank-1 input, left-padded
        ((1, 8, 16), (2, 3, 4)),  # leading extent 1
        ((4, 8, 16), (2, 3, 4)),  # genuine rank-3 tile
        ((2, 3, 4, 5), (2, 2, 2, 2)),  # rank 4
    ],
)
def test_repeat_geometries(mojo_gpu, shape, reps):
    cpu = _fill(shape, torch.float32)
    torch.testing.assert_close(
        cpu.to(mojo_gpu).repeat(*reps).cpu(), cpu.repeat(*reps), rtol=0, atol=0
    )


def test_stack_opinfo_samples_do_not_corrupt_the_heap(mojo_gpu):
    """Regression for a SIGSEGV (exit 139), not just a wrong value.

    The destination-side narrow copy vectorized 4 elements wide on CPU and
    could issue one width-4 store PAST the destination allocation; it only
    corrupts when the heap is packed, so the faithful repro is the whole
    OpInfo sample sequence plus a value check.
    """
    from torch.testing._internal.common_methods_invocations import (  # noqa: PLC0415 -- importing op_db at module scope would pull torch's OpInfo database into every collection of this file
        op_db,
    )

    infos = [info for info in op_db if info.name == "stack"]
    assert infos, "no `stack` OpInfo in this torch"
    ran = 0
    for sample in infos[0].sample_inputs(torch.device("cpu"), torch.int64):
        parts = list(sample.input)
        expected = torch.stack(parts, *sample.args, **sample.kwargs)
        actual = torch.stack(
            [p.to(mojo_gpu) for p in parts], *sample.args, **sample.kwargs
        )
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        ran += 1
    assert ran > 0
