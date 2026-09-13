"""The reductions group of the native mojo backend, through public torch APIs.

Ported from the eager-mode reduction tests in `tests/test_eager_kernels.py`
(the reduce skeleton, var, cumsum, the arg-reductions, min.dim, the vector
norm) and `tests/test_mojo_device.py` (the vector norm's accumulation dtype).
Every check compares against stock CPU torch on the same values; the only
mojo-specific assertions are the dispatch ones at the bottom, which use the
backend's own boxed-kernel counters.
"""

import math

import pytest
import torch

from torch_mojo_backend import get_accelerators, native, register_mojo_devices
from torch_mojo_backend.native import device_module


@pytest.fixture(scope="session")
def registered():
    """The shared `mojo_device` fixture yields a device string without
    registering the backend; these tests need it up before the first op."""
    register_mojo_devices()
    return True


@pytest.fixture(params=["cpu", "gpu"])
def mojo_device(request, registered, mojo_gpu_available: bool):
    if request.param == "gpu":
        if not mojo_gpu_available:
            pytest.skip("You do not have a GPU supported by MAX")
        return "mojo:0"
    return str(device_module.cpu())


@pytest.fixture
def mojo_gpu(registered, mojo_gpu_available: bool) -> str:
    if not mojo_gpu_available:
        pytest.skip("You do not have a GPU supported by MAX")
    return "mojo:0"


# ---------------------------------------------------------------------------
# The generic reduction skeleton: one accumulator per op over one
# (outer, reduce, inner) geometry, so the cases below are written once and
# parametrized by the op rather than once per op.
# ---------------------------------------------------------------------------

_REDUCE_OPS = {
    "sum": lambda t, **kw: torch.sum(t, **kw),
    "mean": lambda t, **kw: torch.mean(t, **kw),
    "amax": lambda t, **kw: torch.amax(t, **kw),
    "amin": lambda t, **kw: torch.amin(t, **kw),
    "norm": lambda t, **kw: torch.linalg.vector_norm(t, **kw),
    "all": lambda t, **kw: torch.all(t, **kw),
    "any": lambda t, **kw: torch.any(t, **kw),
}


@pytest.mark.parametrize("op", list(_REDUCE_OPS))
@pytest.mark.parametrize(
    "shape,dim",
    [
        # Straddle the split threshold on the CONTIGUOUS axis; the split count
        # comes from the runtime SM count, so these bracket it on any card.
        ((1048583,), 0),  # one output, millions deep: maximum splitting
        ((3, 400001), 1),  # a few outputs: still split
        ((4099, 1031), 1),  # many outputs: fused, block per row
        ((5003, 37), 1),  # many SHORT rows: fused, warp per row
        # ... and on the STRIDED axis, where a non-trailing reduce dim is read
        # where it lies instead of being materialized transposed.
        ((400001, 3), 0),
        ((2, 65537), 0),
        ((1031, 4099), 0),
        ((37, 5003), 0),
        # Awkward rank-3 interiors: outer > 1 AND inner > 1.
        ((7, 129, 33), 1),
        ((3, 5, 7), 1),
    ],
)
def test_reduce_skeleton_layouts_match_cpu(mojo_gpu, shape, dim, op):
    fn = _REDUCE_OPS[op]
    if op in ("all", "any"):
        x = torch.rand(shape) < 0.5
    else:
        x = torch.rand(shape) * 0.9 + 0.05
    ours = fn(x.to(mojo_gpu), dim=dim).cpu()
    if op in ("all", "any"):
        torch.testing.assert_close(ours, fn(x, dim=dim))
    else:
        # fp64 reference on the same values: this measures the reduction
        # order, not the input dtype.
        expected = fn(x.double(), dim=dim)
        torch.testing.assert_close(ours.double(), expected, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("op", list(_REDUCE_OPS))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [0, 1])
def test_reduce_skeleton_nonfinite_matches_torch(mojo_gpu, op, dtype, dim):
    """NaN and +/-inf, per op, against stock torch rather than against a rule.

    The rules genuinely differ: amax/amin PROPAGATE NaN, sum and mean inherit
    it through the arithmetic, the L2 norm squares it, and any/all treat NaN
    as TRUTHY because their map is a nonzero test and not a comparison.
    """
    if op in ("all", "any") and dtype is torch.bfloat16:
        pytest.skip("any/all take the bool fast path; dtype is not the axis")
    nan, inf = float("nan"), float("inf")
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [nan, 2.0, 3.0, 4.0],
            [1.0, inf, 3.0, 4.0],
            [1.0, 2.0, -inf, 4.0],
            [inf, -inf, 3.0, 4.0],
            [nan, inf, -inf, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=dtype,
    )
    fn = _REDUCE_OPS[op]
    expected = fn(x, dim=dim)
    ours = fn(x.to(mojo_gpu), dim=dim).cpu()
    if ours.dtype == torch.bool:
        torch.testing.assert_close(ours, expected)
    else:
        torch.testing.assert_close(
            ours.float(), expected.float(), equal_nan=True, atol=1e-2, rtol=1e-2
        )


@pytest.mark.parametrize("op", list(_REDUCE_OPS))
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((3, 1), 1),  # reduce extent of exactly 1
        ((1, 3), 1),  # one output
        ((1, 1), 1),
        ((3, 0), 1),  # EMPTY reduce axis: identity, or torch's own error
        ((0, 3), 1),  # empty output
        ((3, 0), 0),
    ],
)
def test_reduce_skeleton_degenerate_extents_match_torch(mojo_gpu, shape, dim, op):
    """A zero-length reduce axis is where an identity-seeded accumulator and a
    zero-filled workspace part company: sum answers 0, the L2 norm 0, all true,
    any false, mean nan (0/0) — and amax/amin refuse, because torch refuses."""
    fn = _REDUCE_OPS[op]
    x = (torch.rand(shape) < 0.5) if op in ("all", "any") else torch.rand(shape)
    try:
        expected = fn(x, dim=dim)
    except (RuntimeError, IndexError) as exc:
        # The device declines the same cases; it reports its own error rather
        # than reproducing torch's message.
        with pytest.raises((type(exc), NotImplementedError)):
            fn(x.to(mojo_gpu), dim=dim)
        return
    ours = fn(x.to(mojo_gpu), dim=dim).cpu()
    if ours.dtype == torch.bool:
        torch.testing.assert_close(ours, expected)
    else:
        torch.testing.assert_close(ours, expected, equal_nan=True)


@pytest.mark.parametrize("op", ["all", "any"])
@pytest.mark.parametrize("size", [1, 255, 4096, (1 << 22) - 1, (1 << 22) + 1, 1 << 24])
def test_bool_full_reduce_settles_early_and_late(mojo_gpu, op, size):
    """Full any()/all() over bool, including the aten cap at 4.2M elements that
    separates the AnyBool/AllBool entry from the AnySpec/AllSpec one. Both the
    settled-immediately case (element 0 decides) and the settled-at-the-very-end
    case must give torch's answer."""
    fn = torch.all if op == "all" else torch.any
    seed = (
        torch.ones(size, dtype=torch.bool)
        if op == "all"
        else torch.zeros(size, dtype=torch.bool)
    )
    for position in (0, size // 2, size - 1, None):
        x = seed.clone()
        if position is not None:
            x[position] = op != "all"
        assert bool(fn(x.to(mojo_gpu)).cpu()) == bool(fn(x)), (
            f"{op} size={size} flipped at {position}"
        )


# ---------------------------------------------------------------------------
# sum / mean on both mojo devices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keepdim", [True, False])
def test_mean_trailing_dims(mojo_device, keepdim):
    x = torch.randn(1, 512, 7, 7)
    result = x.to(mojo_device).mean([-1, -2], keepdim=keepdim).cpu()
    torch.testing.assert_close(result, x.mean([-1, -2], keepdim=keepdim))


@pytest.mark.parametrize(
    ("shape", "dims", "keepdim"),
    [
        ((3, 5, 7), (0,), False),
        ((3, 5, 7), (0,), True),
        ((5, 3, 17), (0, 1), False),
        ((7, 17, 65), (0,), False),
        ((2, 3, 5, 7), (1, 2), False),
        ((2, 3, 5, 7), (1, 2), True),
        ((2, 257, 17), (1,), False),
    ],
)
def test_sum_contiguous_adjacent_dims(mojo_device, shape, dims, keepdim):
    """Adjacent reductions operate directly on contiguous storage, including a
    nonzero storage offset, and leave the input storage untouched."""
    elements = math.prod(shape)
    host_storage = torch.arange(elements + 2, dtype=torch.float32)
    expected_input = host_storage[1:-1].reshape(shape)
    device_storage = host_storage.to(mojo_device)
    device_input = device_storage[1:-1].view(shape)

    actual = device_input.sum(dim=dims, keepdim=keepdim)
    expected = expected_input.sum(dim=dims, keepdim=keepdim)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(device_storage.cpu(), host_storage, rtol=0, atol=0)


def test_sum_nonadjacent_or_strided_fallback(mojo_device):
    """Layouts outside the direct adjacent-dimension regime stay correct: the
    op permutes the reduce dims to the end and materializes once."""
    host = torch.randn(2, 3, 5, 7)
    device = host.to(mojo_device)

    torch.testing.assert_close(device.sum(dim=(0, 2)).cpu(), host.sum(dim=(0, 2)))

    host_strided = host.transpose(1, 2)
    device_strided = device.transpose(1, 2)
    torch.testing.assert_close(
        device_strided.sum(dim=(0, 2)).cpu(), host_strided.sum(dim=(0, 2))
    )
    torch.testing.assert_close(device_strided.sum(dim=3).cpu(), host_strided.sum(dim=3))


def test_reduction_split_tier(mojo_device):
    """Few outputs, huge reduce extent: the split-the-reduce-axis path. Integer
    valued floats keep every f32 partial sum exact, so an arbitrary split must
    reproduce the sequential answer bit for bit."""
    x = torch.randint(-4, 5, (1, 2**20 + 7)).float()
    xd = x.to(mojo_device)
    torch.testing.assert_close(xd.sum(-1).cpu(), x.sum(-1))
    torch.testing.assert_close(xd.amax(-1).cpu(), x.amax(-1))
    torch.testing.assert_close(torch.any(xd, -1).cpu(), torch.any(x, -1))

    y = torch.randint(-4, 5, (128, 2**20)).float()
    y[5] = 0.0  # give any() a False row
    yd = y.to(mojo_device)
    torch.testing.assert_close(yd.sum(-1).cpu(), y.sum(-1))
    torch.testing.assert_close(yd.amax(-1).cpu(), y.amax(-1))
    torch.testing.assert_close(torch.any(yd, -1).cpu(), torch.any(y, -1))


def test_anyall_nan_is_truthy(mojo_device):
    """torch treats NaN as truthy in any/all, in both launch regimes."""
    small_any = torch.zeros(2, 100)
    small_any[0, 0] = float("nan")
    small_all = torch.full((2, 100), float("nan"))
    huge_any = torch.zeros(1, 2**20 + 7)
    huge_any[0, 12345] = float("nan")
    huge_all = torch.ones(1, 2**20 + 7)
    huge_all[0, 999] = float("nan")
    for x in (small_any, huge_any):
        torch.testing.assert_close(
            torch.any(x.to(mojo_device), -1).cpu(), torch.any(x, -1)
        )
    for x in (small_all, huge_all):
        torch.testing.assert_close(
            torch.all(x.to(mojo_device), -1).cpu(), torch.all(x, -1)
        )


def test_sum_full_reduce_and_dtype_promotion(mojo_gpu):
    """`sum()` with no dim decomposes to sum.dim_IntList, and torch's promotion
    rules (bool / sub-int64 integers -> int64, an explicit dtype= casting the
    input BEFORE the accumulation) are applied on our side too."""
    x = torch.randn(16, 33)
    torch.testing.assert_close(
        x.to(mojo_gpu).sum().cpu(), x.sum(), rtol=2e-6, atol=2e-6
    )

    for dtype in (torch.bool, torch.uint8, torch.int32):
        if dtype is torch.bool:
            i = torch.randint(0, 2, (8, 16), dtype=dtype)
        else:
            i = torch.randint(0, 20, (8, 16), dtype=dtype)
        got = i.to(mojo_gpu).sum(dim=1)
        assert got.dtype == torch.int64 == i.sum(dim=1).dtype
        torch.testing.assert_close(got.cpu(), i.sum(dim=1))

    # dtype= casts first: 1.7 + 2.7 as int32 is 1 + 2, not 4.
    y = torch.tensor([[1.7, 2.7, 3.7, 0.2]])
    ours = torch.sum(y.to(mojo_gpu), dim=1, dtype=torch.float32)
    torch.testing.assert_close(ours.cpu(), torch.sum(y, dim=1, dtype=torch.float32))


@pytest.mark.parametrize("keepdim", [False, True])
def test_mean_and_any_out_variants(mojo_gpu, keepdim):
    """out= writes into the caller's tensor and returns it."""
    x = torch.randn(6, 9)
    xd = x.to(mojo_gpu)

    expected = x.mean(dim=1, keepdim=keepdim)
    out = torch.empty(expected.shape, dtype=torch.float32, device=mojo_gpu)
    returned = torch.mean(xd, dim=1, keepdim=keepdim, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected, rtol=2e-6, atol=2e-6)

    # The comparison runs on the host: `gt` belongs to another op group.
    mask = x > 0
    expected_any = torch.any(mask, dim=0, keepdim=keepdim)
    out_b = torch.empty(expected_any.shape, dtype=torch.bool, device=mojo_gpu)
    returned = torch.any(mask.to(mojo_gpu), dim=0, keepdim=keepdim, out=out_b)
    assert returned.data_ptr() == out_b.data_ptr()
    torch.testing.assert_close(out_b.cpu(), expected_any)


def test_out_variant_resizes_a_mismatching_out(mojo_gpu):
    """`resize_output` first, and by SHAPE: matching only the element count
    lets a (2, 3) result be poured into a (3, 2) destination, and the copy
    then reads the source through the destination's extents."""
    x = torch.randn(2, 3, 4)
    out = torch.empty(0, device=mojo_gpu)
    torch.mean(x.to(mojo_gpu), dim=2, out=out)
    assert tuple(out.shape) == (2, 3)
    torch.testing.assert_close(out.cpu(), x.mean(dim=2), rtol=2e-6, atol=2e-6)

    transposed = torch.empty(3, 2, device=mojo_gpu)
    torch.mean(x.to(mojo_gpu), dim=2, out=transposed)
    assert tuple(transposed.shape) == (2, 3)
    torch.testing.assert_close(transposed.cpu(), x.mean(dim=2), rtol=2e-6, atol=2e-6)


def test_out_variant_into_a_strided_destination(mojo_gpu):
    """A non-contiguous `out` cannot be written by the kernel directly, so the
    result is computed into a fresh buffer and copied across."""
    x = torch.randn(4, 8)
    storage = torch.zeros(4, 2, device=mojo_gpu)
    out = storage[:, 0]
    assert not out.is_contiguous()
    torch.mean(x.to(mojo_gpu), dim=1, out=out)
    torch.testing.assert_close(out.cpu(), x.mean(dim=1), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(storage[:, 1].cpu(), torch.zeros(4))


# ---------------------------------------------------------------------------
# amax / amin / max / min / min.dim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "shape,dim",
    [((357, 789), 1), ((357, 789), 0), ((4, 5, 6), (0, 1)), ((4, 5, 6), (0, 2))],
)
def test_amax_amin_layouts(mojo_device, shape, dim, keepdim):
    x = torch.randn(shape)
    xd = x.to(mojo_device)
    for fn in (torch.amax, torch.amin):
        ours = fn(xd, dim=dim, keepdim=keepdim)
        expected = fn(x, dim=dim, keepdim=keepdim)
        assert ours.shape == expected.shape
        torch.testing.assert_close(ours.cpu(), expected)


def test_max_and_min_full_reduction(mojo_device):
    x = torch.randn(37, 41)
    xd = x.to(mojo_device)
    torch.testing.assert_close(torch.max(xd).cpu(), torch.max(x))
    torch.testing.assert_close(torch.min(xd).cpu(), torch.min(x))
    ints = torch.randint(-100, 100, (5, 9), dtype=torch.int64)
    torch.testing.assert_close(torch.max(ints.to(mojo_device)).cpu(), torch.max(ints))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,dim",
    [((4099, 1031), 1), ((5003, 37), 1), ((1031, 4099), 0), ((1 << 20,), 0)],
)
def test_min_dim_layouts_and_ties(mojo_gpu, shape, dim, dtype):
    """min.dim rides the (value, index) arg-reduction, so it inherits the split
    path, the strided-axis kernel and first-occurrence tie-breaking."""
    x = (torch.rand(shape) * 0.9 + 0.05).to(dtype)
    flat = x.reshape(-1)
    flat[: flat.numel() // 3] = 0.05  # force ties
    values, indices = torch.min(x.to(mojo_gpu), dim=dim)
    exp_values, exp_indices = torch.min(x, dim=dim)
    torch.testing.assert_close(values.cpu(), exp_values)
    torch.testing.assert_close(indices.cpu(), exp_indices)


@pytest.mark.parametrize("dim", [0, 1])
def test_min_dim_propagates_nan_like_torch(mojo_gpu, dim):
    """torch's min.dim answers with the FIRST NaN, not the smallest number."""
    nan = float("nan")
    x = torch.tensor(
        [[1.0, nan, -7.0, 3.0], [nan, nan, 2.0, 1.0], [5.0, 4.0, 3.0, 2.0]]
    )
    values, indices = torch.min(x.to(mojo_gpu), dim=dim)
    exp_values, exp_indices = torch.min(x, dim=dim)
    torch.testing.assert_close(values.cpu(), exp_values, equal_nan=True)
    torch.testing.assert_close(indices.cpu(), exp_indices)


@pytest.mark.parametrize("keepdim", [False, True])
def test_min_dim_keepdim_and_out(mojo_gpu, keepdim):
    x = torch.randn(8, 300, 17)
    xd = x.to(mojo_gpu)
    for dim in (0, 1, 2):
        values, indices = torch.min(xd, dim=dim, keepdim=keepdim)
        exp_values, exp_indices = torch.min(x, dim=dim, keepdim=keepdim)
        assert values.shape == exp_values.shape
        torch.testing.assert_close(values.cpu(), exp_values)
        torch.testing.assert_close(indices.cpu(), exp_indices)

    exp_values, exp_indices = torch.min(x, dim=1, keepdim=keepdim)
    out_v = torch.empty(exp_values.shape, dtype=torch.float32, device=mojo_gpu)
    out_i = torch.empty(exp_indices.shape, dtype=torch.int64, device=mojo_gpu)
    got_v, got_i = torch.min(xd, dim=1, keepdim=keepdim, out=(out_v, out_i))
    assert got_v.data_ptr() == out_v.data_ptr()
    assert got_i.data_ptr() == out_i.data_ptr()
    torch.testing.assert_close(out_v.cpu(), exp_values)
    torch.testing.assert_close(out_i.cpu(), exp_indices)


def test_min_dim_out_into_a_strided_destination(mojo_gpu):
    """A non-contiguous pair of `out` tensors takes the copy path."""
    x = torch.randn(6, 11)
    exp_values, exp_indices = torch.min(x, dim=1)
    v_storage = torch.zeros(6, 2, device=mojo_gpu)
    i_storage = torch.zeros(6, 2, dtype=torch.int64, device=mojo_gpu)
    out_v, out_i = v_storage[:, 0], i_storage[:, 0]
    assert not out_v.is_contiguous() and not out_i.is_contiguous()
    torch.min(x.to(mojo_gpu), dim=1, out=(out_v, out_i))
    torch.testing.assert_close(out_v.cpu(), exp_values)
    torch.testing.assert_close(out_i.cpu(), exp_indices)
    torch.testing.assert_close(v_storage[:, 1].cpu(), torch.zeros(6))


def test_any_out_accepts_a_uint8_destination(mojo_gpu):
    """`any.out`'s dtype policy is bool-or-uint8, so a uint8 `out` is written
    through the cast kernel rather than refused."""
    mask = torch.randint(0, 2, (4, 7), dtype=torch.bool)
    expected = torch.any(mask, dim=1)
    out = torch.empty(4, dtype=torch.uint8, device=mojo_gpu)
    torch.any(mask.to(mojo_gpu), dim=1, out=out)
    torch.testing.assert_close(out.cpu(), expected.to(torch.uint8))
    with pytest.raises(RuntimeError):
        torch.any(
            mask.to(mojo_gpu),
            dim=1,
            out=torch.empty(4, dtype=torch.float32, device=mojo_gpu),
        )


def test_var_default_overloads_decompose_to_var_correction(mojo_gpu):
    """`var(unbiased=True)` and `var.dim` are ATen composites over
    var.correction, which is the one we register."""
    x = torch.randn(9, 13)
    xd = x.to(mojo_gpu)
    torch.testing.assert_close(torch.var(xd).cpu(), torch.var(x), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(
        torch.var(xd, dim=1, unbiased=False).cpu(),
        torch.var(x, dim=1, unbiased=False),
        rtol=2e-6,
        atol=2e-6,
    )


# ---------------------------------------------------------------------------
# argmin / argmax
#
# The arg-reduction splits the reduce axis across blocks and merges partials
# out of order, so each case below checks a rule an out-of-order merge could
# break: torch answers with the FIRST occurrence, and a NaN beats every number.
# ---------------------------------------------------------------------------

_ARGREDUCE_FNS = (torch.argmax, torch.argmin)
# Sizes that straddle the split decision (4096 elements is the smallest slice
# worth its own block).
_ARGREDUCE_SPLIT_SHAPES = (4095, 4096, 8193, 1 << 20)


def _assert_argreduce_matches(device, cpu_tensor, dim=None, device_tensor=None):
    """`device_tensor` lets a caller hand in a VIEW built on the device: a
    strided host tensor cannot cross `_copy_from` (core group), so a strided
    case transfers the contiguous base and slices it there."""
    if device_tensor is None:
        device_tensor = cpu_tensor.to(device)
    for fn in _ARGREDUCE_FNS:
        expected = fn(cpu_tensor) if dim is None else fn(cpu_tensor, dim=dim)
        actual = (
            fn(device_tensor) if dim is None else fn(device_tensor, dim=dim)
        ).cpu()
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected), f"{fn.__name__} dim={dim}"


@pytest.mark.parametrize("size", _ARGREDUCE_SPLIT_SHAPES)
def test_argreduce_full_reduction_across_split_sizes(mojo_gpu, size):
    generator = torch.Generator().manual_seed(20260811)
    _assert_argreduce_matches(mojo_gpu, torch.randn(size, generator=generator))


@pytest.mark.parametrize("size", _ARGREDUCE_SPLIT_SHAPES)
def test_argreduce_ties_take_the_lowest_index(mojo_gpu, size):
    _assert_argreduce_matches(mojo_gpu, torch.full((size,), 3.5))

    marked = torch.zeros(size)
    for position in (0, 1, size // 3, size // 2, size - 1):
        marked[position] = 1.0
    _assert_argreduce_matches(mojo_gpu, marked)

    marked = torch.zeros(size)
    for position in (2, size // 5, size - 2):
        marked[position] = -1.0
    _assert_argreduce_matches(mojo_gpu, marked)


@pytest.mark.parametrize("size", (1000, 1 << 20))
def test_argreduce_nan_beats_every_number(mojo_gpu, size):
    generator = torch.Generator().manual_seed(20260811)
    for position in (0, 3, size // 2, size - 1):
        values = torch.randn(size, generator=generator)
        values[position] = float("nan")
        _assert_argreduce_matches(mojo_gpu, values)
    _assert_argreduce_matches(mojo_gpu, torch.full((size,), float("nan")))


@pytest.mark.parametrize("size", (1000, 1 << 20))
def test_argreduce_saturated_identity_values(mojo_gpu, size):
    """A row of the identity element still has an answer: index 0."""
    _assert_argreduce_matches(mojo_gpu, torch.full((size,), float("-inf")))
    _assert_argreduce_matches(mojo_gpu, torch.full((size,), float("inf")))


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]
)
def test_argreduce_dtypes(mojo_gpu, dtype):
    generator = torch.Generator().manual_seed(20260811)
    if dtype.is_floating_point:
        values = torch.randn(3, 5000, generator=generator).to(dtype)
    else:
        values = torch.randint(-1000, 1000, (3, 5000), generator=generator, dtype=dtype)
    _assert_argreduce_matches(mojo_gpu, values)
    _assert_argreduce_matches(mojo_gpu, values, dim=1)
    _assert_argreduce_matches(mojo_gpu, values, dim=0)


@pytest.mark.parametrize(
    ("shape", "dim"),
    [
        ((4096, 8), 0),  # inner below the coalescing floor: materialized
        ((4096, 16), 0),  # first inner extent the strided kernel takes
        ((4096, 33), 0),  # ragged tail column
        ((357, 789), 0),  # awkward, and one block per column tile
        ((5, 7, 33), 1),  # middle dim of a rank-3 tensor
        ((2, 3, 4, 5), 1),
        ((1 << 12, 1 << 12), 0),  # wide enough to split the strided axis
        ((70000, 2, 32), 1),  # outer past the 65535 cap on grid.y / grid.z
    ],
)
def test_argreduce_strided_axis(mojo_gpu, shape, dim):
    """Non-trailing reduce dims: the strided kernel reads the source in place
    above the coalescing floor and the materialized route runs below it."""
    generator = torch.Generator().manual_seed(20260811)
    _assert_argreduce_matches(
        mojo_gpu, torch.randn(shape, generator=generator), dim=dim
    )
    _assert_argreduce_matches(mojo_gpu, torch.zeros(shape), dim=dim)

    with_nan = torch.randn(shape, generator=generator)
    with_nan[(0,) * (len(shape) - 1) + (1,)] = float("nan")
    _assert_argreduce_matches(mojo_gpu, with_nan, dim=dim)


def test_argreduce_views_and_keepdim(mojo_gpu):
    generator = torch.Generator().manual_seed(20260811)
    base = torch.randn(64, 128, generator=generator)
    device_base = base.to(mojo_gpu)
    views = [
        (base.t(), device_base.t()),
        (base[:, 3:70], device_base[:, 3:70]),
        (base[::2, ::3], device_base[::2, ::3]),
    ]
    for host_view, device_view in views:
        for dim in (None, 0, 1):
            _assert_argreduce_matches(
                mojo_gpu, host_view, dim=dim, device_tensor=device_view
            )

    values = torch.randn(8, 300, 17, generator=generator)
    device_values = values.to(mojo_gpu)
    for dim in (0, 1, 2):
        for fn in _ARGREDUCE_FNS:
            expected = fn(values, dim=dim, keepdim=True)
            actual = fn(device_values, dim=dim, keepdim=True).cpu()
            assert actual.shape == expected.shape
            assert torch.equal(actual, expected)


# ---------------------------------------------------------------------------
# var
# ---------------------------------------------------------------------------

_VAR_LAYOUTS = [
    ((357, 789), 1),  # contiguous reduce, awkward extents
    ((357, 789), 0),  # strided reduce (leading dim), awkward extents
    ((357, 789), None),  # full reduce
    ((4, 5, 6), 1),  # strided reduce with a real outer AND inner
    ((4, 5, 6), (1, 2)),  # adjacent trailing interval
    ((4, 5, 6), (0, 1)),  # adjacent leading interval
    ((4, 5, 6), (0, 2)),  # NON-adjacent: permuted and materialized
    ((1, 513), 0),  # reduced extent of 1
    ((513, 1), 0),  # inner of 1, single column
    ((16, 33), 1),  # rows shorter than one 16-byte pass per thread
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("shape,dim", _VAR_LAYOUTS)
def test_var_layouts_match_cpu(mojo_device, shape, dim, keepdim, dtype):
    x = torch.randn(shape).to(dtype)
    kwargs = {} if dim is None else {"dim": dim}
    result = torch.var(x.to(mojo_device), correction=1, keepdim=keepdim, **kwargs)
    expected = torch.var(x, correction=1, keepdim=keepdim, **kwargs)
    assert result.shape == expected.shape
    # equal_nan: a reduced extent of 1 with correction=1 is nan on both sides.
    torch.testing.assert_close(
        result.cpu(), expected, atol=2e-2, rtol=2e-2, equal_nan=True
    )


@pytest.mark.parametrize(
    "shape,dim",
    [
        ((1048583,), None),
        ((3, 400001), 1),
        ((400001, 3), 0),
        ((2, 65537), 0),
        ((4099, 1031), 1),
        ((1031, 4099), 0),
    ],
)
def test_var_split_regimes_match_cpu(mojo_gpu, shape, dim):
    x = torch.rand(shape) * 0.9 + 0.05
    kwargs = {} if dim is None else {"dim": dim}
    result = torch.var(x.to(mojo_gpu), correction=1, **kwargs)
    expected = torch.var(x.double(), correction=1, **kwargs)
    torch.testing.assert_close(result.cpu().double(), expected, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("shape,dim", [((1 << 22,), None), ((5, 300000), 1)])
def test_var_survives_large_mean_offset(mojo_gpu, shape, dim):
    """Single-pass moments must not cancel away the answer: the moments are
    taken about an assumed mean read from the slice itself."""
    x = torch.rand(shape) - 0.5 + 1e4
    kwargs = {} if dim is None else {"dim": dim}
    result = torch.var(x.to(mojo_gpu), correction=1, **kwargs).cpu().double()
    expected = torch.var(x.double(), correction=1, **kwargs)
    torch.testing.assert_close(result, expected, atol=0, rtol=1e-3)


@pytest.mark.parametrize("correction", [0, 1, 4, 5, 9])
def test_var_correction_at_or_above_extent_matches_torch(mojo_gpu, correction):
    # torch clamps the divisor at 0, so correction >= n is inf (or nan for a
    # constant sample), never a negative variance.
    x = torch.tensor([[1.0, 2.0, 3.0, 5.0], [7.0, 7.0, 7.0, 7.0]])
    result = torch.var(x.to(mojo_gpu), dim=1, correction=correction)
    expected = torch.var(x, dim=1, correction=correction)
    torch.testing.assert_close(result.cpu(), expected, equal_nan=True)


# ---------------------------------------------------------------------------
# linalg_vector_norm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,dim",
    [((4099, 1031), 1), ((5003, 37), 1), ((1031, 4099), 0), ((1 << 20,), 0)],
)
def test_vector_norm_matches_torch(mojo_gpu, shape, dim, dtype):
    """One pass (sum of squares, root in the finalize), not mul -> sum -> sqrt."""
    x = (torch.rand(shape) * 0.9 + 0.05).to(dtype)
    ours = torch.linalg.vector_norm(x.to(mojo_gpu), dim=dim).cpu()
    expected = torch.linalg.vector_norm(x.double(), dim=dim)
    torch.testing.assert_close(ours.double(), expected, atol=0, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vector_norm_with_an_accumulation_dtype(mojo_gpu, dtype):
    """clip_grad_norm_ asks for the norm in float32 explicitly."""
    cpu = torch.randn(4096, dtype=dtype)
    expected = torch.linalg.vector_norm(cpu, 2.0, dtype=torch.float32)
    got = torch.linalg.vector_norm(cpu.to(mojo_gpu), 2.0, dtype=torch.float32)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-5, atol=1e-4)


def test_vector_norm_out_and_strided_input(mojo_gpu):
    """Gradient clipping keeps its norm on-device: a strided operand, and an
    `out=` scalar the op must write in place."""
    contiguous = torch.linspace(-3.0, 4.0, 35).reshape(5, 7)
    strided = contiguous.t()
    assert not strided.is_contiguous()
    expected = torch.linalg.vector_norm(strided)

    # The transpose is taken ON the device: a strided host tensor cannot cross
    # `_copy_from` (core group).
    device_strided = contiguous.to(mojo_gpu).t()
    out = torch.empty((), dtype=torch.float32, device=mojo_gpu)
    out_ptr = out.data_ptr()
    returned = torch.linalg.vector_norm(device_strided, out=out)
    assert returned.data_ptr() == out_ptr
    torch.testing.assert_close(out.cpu(), expected)

    empty = torch.empty((0, 7), dtype=torch.float32).to(mojo_gpu)
    torch.testing.assert_close(
        torch.linalg.vector_norm(empty).cpu(), torch.tensor(0.0), rtol=0, atol=0
    )


# ---------------------------------------------------------------------------
# cumsum
# ---------------------------------------------------------------------------

_CUMSUM_SHAPES = [
    ((4096, 4096), 1),  # INNER, many lines
    ((4096, 4096), 0),  # OUTER, many lines
    ((357, 789), 1),  # INNER, awkward extents
    ((357, 789), 0),  # OUTER, awkward extents
    ((1, 200_000), 1),  # INNER, workspace (one very long line)
    ((8, 100_003), 1),  # INNER, workspace (few long lines, off-by-one)
    ((4, 1024 * 5), 1),  # INNER, workspace, exact tile multiple
    ((200_003, 4), 0),  # OUTER, narrow (few columns, many rows)
    ((1, 4096), 1),  # single line
    ((500, 1), 1),  # single column, INNER framing
    ((4096, 1), 0),  # single column, OUTER framing
    ((1, 1), 1),  # single element
    ((500, 256), 1),  # exact tile multiple (256 threads)
    ((500, 257), 1),  # one past a tile multiple
]

_CUMSUM_LOWP_RTOL = 3e-2
_CUMSUM_LOWP_ATOL = 3e-2


@pytest.mark.parametrize("shape,dim", _CUMSUM_SHAPES)
def test_cumsum_shapes_match_cpu(mojo_gpu, shape, dim):
    torch.manual_seed(0)
    x = torch.randn(shape)
    result = torch.cumsum(x.to(mojo_gpu), dim=dim).cpu().double()
    expected = torch.cumsum(x.double(), dim=dim)
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-2)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.int32, torch.int64]
)
@pytest.mark.parametrize("shape,dim", [((4096, 4096), 1), ((4096, 4096), 0)])
def test_cumsum_dtypes_match_cpu(mojo_gpu, shape, dim, dtype):
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(shape).to(dtype)
    else:
        x = torch.randint(-100, 100, shape, dtype=dtype)
    result = torch.cumsum(x.to(mojo_gpu), dim=dim)
    expected = torch.cumsum(x, dim=dim)
    assert result.dtype == expected.dtype
    if dtype.is_floating_point:
        torch.testing.assert_close(
            result.cpu(), expected, rtol=_CUMSUM_LOWP_RTOL, atol=_CUMSUM_LOWP_ATOL
        )
    else:
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.int32, torch.int64]
)
@pytest.mark.parametrize("shape,dim", [((4, 5120), 1), ((8, 100_003), 1)])
def test_cumsum_workspace_dtypes_match_cpu(mojo_gpu, shape, dim, dtype):
    """The long-line 3-pass workspace path (few rows), per dtype."""
    torch.manual_seed(0)
    if dtype.is_floating_point:
        x = torch.randn(shape).to(dtype)
    else:
        x = torch.randint(-100, 100, shape, dtype=dtype)
    result = torch.cumsum(x.to(mojo_gpu), dim=dim)
    expected = torch.cumsum(x, dim=dim)
    assert result.dtype == expected.dtype
    if dtype.is_floating_point:
        torch.testing.assert_close(
            result.cpu(), expected, rtol=_CUMSUM_LOWP_RTOL, atol=_CUMSUM_LOWP_ATOL
        )
    else:
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dim", [-1, -2])
def test_cumsum_negative_dim(mojo_gpu, dim):
    torch.manual_seed(0)
    x = torch.randn(64, 4096)
    result = torch.cumsum(x.to(mojo_gpu), dim=dim).cpu().double()
    expected = torch.cumsum(x.double(), dim=dim)
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=2e-2)


def test_cumsum_dtype_kwarg_casts_before_accumulating(mojo_gpu):
    """torch casts the input to `dtype` first, then accumulates in it."""
    x = torch.arange(1, 9, dtype=torch.int64).reshape(2, 4)
    result = torch.cumsum(x.to(mojo_gpu), dim=1, dtype=torch.float32)
    expected = torch.cumsum(x, dim=1, dtype=torch.float32)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result.cpu(), expected)

    y = torch.tensor([[1.7, 2.7, 3.7, 0.2]])
    result = torch.cumsum(y.to(mojo_gpu), dim=1, dtype=torch.int32)
    expected = torch.cumsum(y, dim=1, dtype=torch.int32)
    assert result.dtype == torch.int32
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.int32, torch.uint8, torch.bool])
def test_cumsum_integer_promotes_to_int64(mojo_gpu, dtype):
    """No dtype= kwarg: torch promotes every integer/bool cumsum to int64.
    int8/int16 are NOT covered: promoting them needs a cast the cast kernel
    does not dispatch on — a pre-existing gap `sum` has for the same reason."""
    if dtype == torch.bool:
        x = torch.randint(0, 2, (8, 16), dtype=torch.bool)
    else:
        x = torch.randint(0, 20, (8, 16), dtype=dtype)
    result = torch.cumsum(x.to(mojo_gpu), dim=1)
    expected = torch.cumsum(x, dim=1)
    assert result.dtype == torch.int64 == expected.dtype
    torch.testing.assert_close(result.cpu(), expected)


def test_cumsum_noncontiguous_input_materializes_correctly(mojo_gpu):
    torch.manual_seed(0)
    base = torch.randn(64, 128)
    x = base.t()
    assert not x.is_contiguous()
    result = torch.cumsum(base.to(mojo_gpu).t(), dim=1).cpu().double()
    expected = torch.cumsum(x.double(), dim=1)
    torch.testing.assert_close(result, expected, rtol=2e-3, atol=1e-2)


def test_cumsum_declines_middle_dim_on_rank3(mojo_gpu):
    """rank>=3 non-trailing dims are out of this kernel family's scope and must
    raise, not silently compute the wrong axis."""
    x = torch.randn(4, 5, 6).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.cumsum(x, dim=1)


def test_cumsum_new_dtype_and_dim_decline_on_the_cpu_device(registered):
    """dim=0 and bf16/f16 route through the CUDA-only fast kernels; on the MAX
    CPU device they must decline, not run an unmeasured kernel. int64 / int32 /
    float32 on the trailing dim is unaffected."""
    mojo_cpu = str(device_module.cpu())
    x32 = torch.randn(8, 16)
    with pytest.raises(NotImplementedError):
        torch.cumsum(x32.to(mojo_cpu), dim=0)
    xbf16 = torch.randn(8, 16).to(torch.bfloat16)
    with pytest.raises(NotImplementedError):
        torch.cumsum(xbf16.to(mojo_cpu), dim=1)
    result = torch.cumsum(x32.to(mojo_cpu), dim=1)
    torch.testing.assert_close(result.cpu(), torch.cumsum(x32, dim=1))


# ---------------------------------------------------------------------------
# declines and dispatch
# ---------------------------------------------------------------------------


def test_unsupported_inputs_raise_not_implemented(mojo_gpu):
    """Eager has no graph fallback: every gate the old fast path answered with
    NOT_HANDLED is an actionable NotImplementedError here."""
    scalar = torch.tensor(3.0).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        scalar.sum()
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(torch.randn(4, 5).to(mojo_gpu), ord=3)
    with pytest.raises(NotImplementedError):
        torch.mean(torch.randint(0, 4, (3, 4), dtype=torch.int64).to(mojo_gpu), dim=1)
    with pytest.raises(NotImplementedError):
        torch.argmax(torch.empty(0, 4).to(mojo_gpu), dim=1)
    with pytest.raises(NotImplementedError):
        torch.ops.aten.any.dims((torch.randn(3, 4) > 0).to(mojo_gpu), [])


def _bools() -> torch.Tensor:
    """A host-side bool tensor: comparisons belong to another op group, so
    these tests must not build their masks on the device."""
    return torch.randint(0, 2, (4, 5), dtype=torch.bool)


_EXPECTED_OVERLOADS = [
    ("aten::sum.dim_IntList", lambda d: torch.randn(4, 5).to(d).sum(dim=1)),
    ("aten::mean", lambda d: torch.randn(4, 5).to(d).mean()),
    ("aten::mean.dim", lambda d: torch.randn(4, 5).to(d).mean(dim=1)),
    (
        "aten::mean.out",
        lambda d: torch.mean(
            torch.randn(4, 5).to(d), dim=1, out=torch.empty(4, device=d)
        ),
    ),
    ("aten::amax", lambda d: torch.amax(torch.randn(4, 5).to(d), dim=1)),
    ("aten::amin", lambda d: torch.amin(torch.randn(4, 5).to(d), dim=1)),
    ("aten::max", lambda d: torch.max(torch.randn(4, 5).to(d))),
    ("aten::min", lambda d: torch.min(torch.randn(4, 5).to(d))),
    ("aten::min.dim", lambda d: torch.min(torch.randn(4, 5).to(d), dim=1)),
    (
        "aten::min.dim_min",
        lambda d: torch.min(
            torch.randn(4, 5).to(d),
            dim=1,
            out=(torch.empty(4, device=d), torch.empty(4, dtype=torch.int64, device=d)),
        ),
    ),
    ("aten::argmax", lambda d: torch.argmax(torch.randn(4, 5).to(d), dim=1)),
    ("aten::argmin", lambda d: torch.argmin(torch.randn(4, 5).to(d), dim=1)),
    ("aten::all", lambda d: torch.all(_bools().to(d))),
    ("aten::all.dim", lambda d: torch.all(_bools().to(d), dim=1)),
    ("aten::all.dims", lambda d: torch.ops.aten.all.dims(_bools().to(d), [0, 1])),
    ("aten::any", lambda d: torch.any(_bools().to(d))),
    ("aten::any.dim", lambda d: torch.any(_bools().to(d), dim=1)),
    ("aten::any.dims", lambda d: torch.ops.aten.any.dims(_bools().to(d), [0, 1])),
    (
        "aten::any.out",
        lambda d: torch.any(
            _bools().to(d), dim=1, out=torch.empty(4, dtype=torch.bool, device=d)
        ),
    ),
    ("aten::var.correction", lambda d: torch.var(torch.randn(4, 5).to(d), dim=1)),
    (
        "aten::linalg_vector_norm",
        lambda d: torch.linalg.vector_norm(torch.randn(4, 5).to(d), dim=1),
    ),
    (
        "aten::linalg_vector_norm.out",
        lambda d: torch.linalg.vector_norm(
            torch.randn(4, 5).to(d), dim=1, out=torch.empty(4, device=d)
        ),
    ),
    ("aten::cumsum", lambda d: torch.cumsum(torch.randn(4, 5).to(d), dim=1)),
]


@pytest.mark.parametrize(
    "op_name,call", _EXPECTED_OVERLOADS, ids=[n for n, _ in _EXPECTED_OVERLOADS]
)
def test_every_overload_dispatches_natively(mojo_gpu, op_name, call):
    """The boxed-kernel counter names the overload that actually ran, so this
    is the assertion that the registration (not an ATen decomposition) is what
    served the call."""
    native.op_counting(True)
    before = native.op_count(op_name)
    call(mojo_gpu)
    assert native.op_count(op_name) > before, f"{op_name} did not reach the backend"


def test_sum_default_overload_decomposes_to_dim_intlist(mojo_gpu):
    """`aten::sum` is left to ATen's composite, which calls sum.dim_IntList —
    the old registry did the same, so there is one sum implementation."""
    native.op_counting(True)
    before = native.op_count("aten::sum.dim_IntList")
    torch.randn(4, 5).to(mojo_gpu).sum()
    assert native.op_count("aten::sum.dim_IntList") > before
    assert native.op_count("aten::sum") == 0


def test_accelerator_count_is_sane(registered):
    assert len(list(get_accelerators())) >= 1


# ---------------------------------------------------------------------------
# var: the cancellation regimes. The tests above are all well-conditioned;
# none of them can reach the detector that recovers a slice whose moments
# cancel.
# ---------------------------------------------------------------------------

_VAR_N = 1 << 22


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("magnitude", [1e6, -1e6])
@pytest.mark.parametrize(
    "position", [0, 1, 8192, _VAR_N // 528, _VAR_N // 528 - 1, _VAR_N // 2, _VAR_N - 1]
)
def test_var_outlier_anywhere_stays_accurate(mojo_gpu, position, magnitude, dtype):
    """Position 0 is the catastrophic one: it IS the assumed mean, so
    `M2 = q - s**2/n` becomes a difference of two ~4e18 quantities. The
    reference is computed on the SAME quantized values, so this measures the
    kernel's summation, not the dtype."""
    x64 = torch.randn(
        _VAR_N, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    x64[position] = magnitude
    x = x64.to(dtype)
    expected = torch.var(x.double(), correction=1)
    result = torch.var(x.to(mojo_gpu), correction=1).cpu().double()
    rtol = 1e-6 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(result, expected, atol=0, rtol=rtol)


@pytest.mark.parametrize("dim", [0, 1])
def test_var_outlier_row_poisons_every_slice(mojo_gpu, dim):
    """Recovery has to be per output element, not one whole-tensor decision:
    here every slice contains an outlier."""
    x64 = torch.randn(
        2048, 2048, dtype=torch.float64, generator=torch.Generator().manual_seed(0)
    )
    if dim == 0:
        x64[0, :] = 1e6
    else:
        x64[:, 0] = 1e6
    x = x64.to(torch.float32)
    expected = torch.var(x.double(), dim=dim, correction=1)
    result = torch.var(x.to(mojo_gpu), dim=dim, correction=1).cpu().double()
    torch.testing.assert_close(result, expected, atol=0, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("value", [0.0, 3.0, -1e6])
@pytest.mark.parametrize("shape,dim", [((1 << 22,), None), ((512, 4096), 0)])
def test_var_constant_slice_is_exactly_zero(mojo_gpu, shape, dim, value, dtype):
    """The boundary of the cancellation detector: a ratio-based one divides
    by zero here. Exactly +0.0, never -0.0."""
    x = torch.full(shape, value, dtype=dtype)
    kwargs = {} if dim is None else {"dim": dim}
    result = torch.var(x.to(mojo_gpu), correction=1, **kwargs).cpu()
    assert bool((result == 0).all()), result
    assert not bool(result.signbit().any())
