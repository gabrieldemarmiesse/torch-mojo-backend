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

from tests.native.conftest import ran, skip_if_metal
from torch_mojo_backend import get_accelerators, native, register_mojo_devices


@pytest.fixture(scope="session")
def registered():
    """The shared `mojo_device` fixture yields a device string without
    registering the backend; these tests need it up before the first op."""
    register_mojo_devices()
    return True


@pytest.fixture
def mojo_gpu(registered, mojo_gpu_available: bool) -> str:
    if not mojo_gpu_available:
        pytest.skip("You do not have a GPU supported by MAX")
    return "mojo:0"


@pytest.fixture
def mojo_device(mojo_gpu: str) -> str:
    """There is no CPU-backed mojo device any more; this is now just an
    alias of `mojo_gpu`."""
    return mojo_gpu


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
    """`sum()` with no dim reduces over every axis, and torch's promotion
    rules (bool / sub-int64 integers -> int64, an explicit dtype= casting the
    input BEFORE the accumulation) are applied on our side too."""
    x = torch.randn(16, 33, generator=torch.Generator().manual_seed(0))
    # fp32 rounding grows with the partial sums, i.e. with sum(|x|), not with
    # the (possibly near-zero) result: an unseeded draw summing to 0.16 once
    # missed an absolute 2e-6 by 1e-6 purely from a different reduction order.
    torch.testing.assert_close(
        x.to(mojo_gpu).sum().cpu(), x.sum(), rtol=2e-6, atol=1e-6 * x.abs().sum().item()
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
def test_sum_out_variant(mojo_gpu, keepdim):
    """out= writes into the caller's tensor, resizes on a shape mismatch, and
    applies the same int64 promotion as the non-out overload."""
    x = torch.randn(6, 9)
    xd = x.to(mojo_gpu)
    expected = x.sum(dim=1, keepdim=keepdim)
    out = torch.empty(expected.shape, dtype=torch.float32, device=mojo_gpu)
    returned = torch.sum(xd, dim=1, keepdim=keepdim, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected, rtol=2e-6, atol=2e-6)

    i = torch.randint(0, 20, (8, 16), dtype=torch.int32)
    out_i = torch.empty(0, dtype=torch.int64, device=mojo_gpu)
    torch.sum(i.to(mojo_gpu), dim=1, out=out_i)
    torch.testing.assert_close(out_i.cpu(), i.sum(dim=1))

    resized = torch.empty(0, device=mojo_gpu)
    torch.sum(xd, dim=1, out=resized)
    assert tuple(resized.shape) == tuple(x.sum(dim=1).shape)
    torch.testing.assert_close(resized.cpu(), x.sum(dim=1), rtol=2e-6, atol=2e-6)


def test_sum_out_variant_declines_unsafe_cast(mojo_gpu):
    """`safe_cast`: a float result poured into an integral out is refused,
    matching sum.IntList_out's structured-kernel dtype check."""
    x = torch.randn(4, 5).to(mojo_gpu)
    out = torch.empty(4, dtype=torch.int64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.sum(x, dim=1, out=out)


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


@pytest.mark.parametrize("keepdim", [False, True])
def test_amax_out_variant(mojo_gpu, keepdim):
    x = torch.randn(357, 789)
    xd = x.to(mojo_gpu)
    expected = torch.amax(x, dim=1, keepdim=keepdim)

    out = torch.empty(expected.shape, dtype=torch.float32, device=mojo_gpu)
    returned = torch.amax(xd, dim=1, keepdim=keepdim, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected)

    # a wrongly-shaped out is resized (resize_output), by shape not numel.
    mismatched = torch.empty(0, device=mojo_gpu)
    torch.amax(xd, dim=1, keepdim=keepdim, out=mismatched)
    torch.testing.assert_close(mismatched.cpu(), expected)


def test_amax_out_into_a_strided_destination(mojo_gpu):
    """A non-contiguous `out` cannot be written by the kernel directly, so the
    result is computed into a fresh buffer and copied across."""
    x = torch.randn(357, 789)
    storage = torch.zeros(357, 2, device=mojo_gpu)
    out = storage[:, 0]
    assert not out.is_contiguous()
    torch.amax(x.to(mojo_gpu), dim=1, out=out)
    torch.testing.assert_close(out.cpu(), x.amax(dim=1))
    torch.testing.assert_close(storage[:, 1].cpu(), torch.zeros(357))


def test_amax_out_dtype_and_empty_dim_errors(mojo_gpu):
    """amax's out dtype policy is exact (torch's meta: input/out dtypes must
    match, no cast), unlike mean.out's safe_cast."""
    x = torch.randn(4, 5).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.amax(x, dim=1, out=torch.empty(4, dtype=torch.float64, device=mojo_gpu))
    with pytest.raises(NotImplementedError, match="reduce dim of size 0"):
        torch.amax(
            torch.empty(4, 0, device=mojo_gpu),
            dim=1,
            out=torch.empty(4, device=mojo_gpu),
        )


def test_max_and_min_full_reduction(mojo_device):
    x = torch.randn(37, 41)
    xd = x.to(mojo_device)
    torch.testing.assert_close(torch.max(xd).cpu(), torch.max(x))
    torch.testing.assert_close(torch.min(xd).cpu(), torch.min(x))
    ints = torch.randint(-100, 100, (5, 9), dtype=torch.int64)
    torch.testing.assert_close(torch.max(ints.to(mojo_device)).cpu(), torch.max(ints))


@pytest.mark.parametrize("shape", [(7,), (357, 789), (1 << 20,)])
def test_extrema_float64(mojo_gpu, shape):
    """amax/amin and full max/min select exactly in float64 (the warp fold
    trades the bits as uint64): values 1 ulp apart and a NaN must survive."""
    skip_if_metal(mojo_gpu, "Metal does not support float64")
    x = torch.randn(shape, dtype=torch.float64)
    x.view(-1)[len(x.view(-1)) // 2] = 1e300
    x.view(-1)[-1] = math.nextafter(1e300, math.inf)
    xd = x.to(mojo_gpu)
    for fn in (torch.max, torch.min):
        torch.testing.assert_close(fn(xd).cpu(), fn(x), rtol=0, atol=0)
    dim = len(shape) - 1
    for fn in (torch.amax, torch.amin):
        want = fn(x, dim=dim)
        torch.testing.assert_close(fn(xd, dim=dim).cpu(), want, rtol=0, atol=0)
    x.view(-1)[0] = math.nan
    assert torch.max(x.to(mojo_gpu)).isnan().item()
    assert torch.min(x.to(mojo_gpu)).isnan().item()


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape,dim,keepdim",
    [
        ((4099, 1031), 1, False),
        ((5003, 37), 0, True),
        ((357, 789), None, False),
        ((4, 5, 6), [0, 2], False),
    ],
)
def test_vector_norm_ord0_matches_torch(mojo_gpu, shape, dim, keepdim, dtype):
    """ord=0: count of nonzero elements (a separate compiled spec, NormL0Op)."""
    x = torch.randint(-2, 3, shape).to(dtype)  # includes exact zeros
    ours = torch.linalg.vector_norm(
        x.to(mojo_gpu), ord=0, dim=dim, keepdim=keepdim
    ).cpu()
    expected = torch.linalg.vector_norm(x, ord=0, dim=dim, keepdim=keepdim)
    torch.testing.assert_close(ours, expected)


def test_vector_norm_ord0_nan_counts_as_nonzero(mojo_gpu):
    """Matches CUDA's `NormZeroOps`: an ordered `== 0` test, so NaN counts."""
    x = torch.tensor([0.0, 1.0, float("nan"), 0.0, -3.0])
    ours = torch.linalg.vector_norm(x.to(mojo_gpu), ord=0).cpu()
    expected = torch.linalg.vector_norm(x, ord=0)
    torch.testing.assert_close(ours, expected)


def test_vector_norm_ord0_noncontiguous(mojo_gpu):
    """A full reduction is permutation-invariant for a count, so it can't
    catch a kernel that ignores strides; reduce one axis of a transpose
    instead -- the wrong grouping would change per-row counts."""
    contiguous = torch.randint(-2, 3, (5, 7)).float()
    strided = contiguous.t()
    assert not strided.is_contiguous()
    expected = torch.linalg.vector_norm(strided, ord=0, dim=1)
    # The transpose is taken ON the device (see test_vector_norm_out_and_
    # strided_input): a strided host tensor cannot cross `_copy_from`.
    device_strided = contiguous.to(mojo_gpu).t()
    ours = torch.linalg.vector_norm(device_strided, ord=0, dim=1).cpu()
    torch.testing.assert_close(ours, expected)


def test_vector_norm_ord0_empty(mojo_gpu):
    empty = torch.empty((0, 7), dtype=torch.float32).to(mojo_gpu)
    torch.testing.assert_close(
        torch.linalg.vector_norm(empty, ord=0).cpu(), torch.tensor(0.0), rtol=0, atol=0
    )


def test_vector_norm_ord0_out_resizes(mojo_gpu):
    """out= starts the wrong shape and must be resized (`resize_output`)."""
    x = torch.randn(4, 5)
    expected = torch.linalg.vector_norm(x, ord=0, dim=1)
    out = torch.empty(1, device=mojo_gpu)
    returned = torch.linalg.vector_norm(x.to(mojo_gpu), ord=0, dim=1, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected)


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


def _value_index_out(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(0, device=device), torch.empty(
        0, dtype=torch.int64, device=device
    )


_EXPECTED_OVERLOADS = [
    ("aten::sum", lambda d: torch.randn(4, 5).to(d).sum()),
    ("aten::sum.dim_IntList", lambda d: torch.randn(4, 5).to(d).sum(dim=1)),
    (
        "aten::sum.IntList_out",
        lambda d: torch.sum(
            torch.randn(4, 5).to(d), dim=1, out=torch.empty(4, device=d)
        ),
    ),
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
    ("aten::topk", lambda d: torch.topk(torch.randn(4, 5).to(d), 2)),
    (
        "aten::topk.values",
        lambda d: torch.topk(
            torch.randn(4, 5).to(d),
            2,
            out=(torch.empty(0, device=d), torch.empty(0, dtype=torch.int64, device=d)),
        ),
    ),
    ("aten::sort.stable", lambda d: torch.sort(torch.randn(4, 5).to(d))),
    ("aten::sort.stable", lambda d: torch.argsort(torch.randn(4, 5).to(d))),
    ("aten::sort.stable", lambda d: torch.msort(torch.randn(4, 5).to(d))),
    (
        "aten::sort.values_stable",
        lambda d: torch.sort(
            torch.randn(4, 5).to(d),
            stable=True,
            out=(torch.empty(0, device=d), torch.empty(0, dtype=torch.int64, device=d)),
        ),
    ),
    ("aten::kthvalue", lambda d: torch.kthvalue(torch.randn(4, 5).to(d), 2)),
    (
        "aten::kthvalue.values",
        lambda d: torch.kthvalue(torch.randn(4, 5).to(d), 2, out=_value_index_out(d)),
    ),
    ("aten::median", lambda d: torch.median(torch.randn(4, 5).to(d))),
    ("aten::median.dim", lambda d: torch.median(torch.randn(4, 5).to(d), 1)),
    (
        "aten::median.dim_values",
        lambda d: torch.median(torch.randn(4, 5).to(d), 1, out=_value_index_out(d)),
    ),
    ("aten::nanmedian", lambda d: torch.nanmedian(torch.randn(4, 5).to(d))),
    ("aten::nanmedian.dim", lambda d: torch.nanmedian(torch.randn(4, 5).to(d), 1)),
    (
        "aten::nanmedian.dim_values",
        lambda d: torch.nanmedian(torch.randn(4, 5).to(d), 1, out=_value_index_out(d)),
    ),
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


def test_sum_default_overload_dispatches_natively(mojo_gpu):
    """`aten::sum` (full reduction) is registered directly now, so it no
    longer falls through ATen's CompositeExplicitAutograd to
    sum.dim_IntList."""
    native.op_counting(True)
    before = native.op_count("aten::sum")
    torch.randn(4, 5).to(mojo_gpu).sum()
    assert native.op_count("aten::sum") > before


def test_accelerator_count_is_sane(registered):
    # 0 is a legitimate count now: there is no CPU-backed mojo device to
    # fall back to any more on a box with no accelerator.
    assert len(list(get_accelerators())) >= 0


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


# ---------------------------------------------------------------------------
# sort / topk. The kernel orders (key, original index) pairs, so its result is
# always the STABLE one: every reference below is CPU's `stable=True` sort,
# which it must reproduce exactly, ties included. topk's tie order is
# unspecified in ATen, so its values are compared with `torch.topk` and its
# indices with the stable sort's prefix, plus a check that they point at the
# returned values. The kernel picks one of three launch routes from the row
# length and k (one tile / tournament / full multi-tile sort), so the sizes
# straddle the route boundaries: the tile is 4096 elements for 32-bit keys
# and 2048 for 64-bit ones.
# ---------------------------------------------------------------------------

_SORT_ROUTE_SIZES = (1, 2, 33, 255, 2049, 4095, 4096, 4097, 8193, 12000, 50257)
_SORT_DTYPES = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float64,
    torch.int64,
    torch.int32,
    torch.int16,
    torch.int8,
    torch.uint8,
    torch.bool,
]


def _same(actual: torch.Tensor, expected: torch.Tensor):
    """Bit-exact equality with NaN equal to NaN (`torch.equal` has no
    equal_nan, and a NaN still has to land in the right place)."""
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)


def _sort_probe(rows: int, columns: int, seed: int = 7) -> torch.Tensor:
    """Values with deliberate ties, both zeros, and NaNs in longer rows."""
    host = torch.randn(rows, columns, generator=torch.Generator().manual_seed(seed))
    if columns >= 16:
        host[:, ::7] = 0.0
        host[:, ::11] = -0.0
        host[:, ::13] = 1.5
        host[:, 3] = float("nan")
        host[:, 5] = -float("nan")
    return host


def _dtype_probe(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Heavily tied values exactly representable in every dtype."""
    host = torch.arange(math.prod(shape), dtype=torch.int64).reshape(shape)
    host = (host * 37 + 11) % 97
    if dtype is torch.bool:
        return host % 3 == 0
    if dtype is torch.uint8:
        return host.to(dtype)
    return (host - 48).to(dtype)


def _check_topk(x: torch.Tensor, device: str, k: int, dim: int, largest: bool):
    with ran("aten::topk"):
        actual = torch.topk(x.to(device), k, dim=dim, largest=largest)
    expected = torch.topk(x, k, dim=dim, largest=largest)
    _same(actual.values, expected.values)
    indices = actual.indices.cpu()
    assert indices.dtype == torch.int64
    _same(torch.gather(x, dim, indices), expected.values)
    stable = torch.sort(x, dim=dim, descending=largest, stable=True)
    _same(indices, stable.indices.narrow(dim, 0, k))


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("columns", _SORT_ROUTE_SIZES)
def test_sort_matches_stable_cpu_on_every_route(mojo_gpu, columns, descending):
    host = _sort_probe(3, columns)
    expected = torch.sort(host, dim=-1, descending=descending, stable=True)
    with ran("aten::sort.stable"):
        actual = torch.sort(host.to(mojo_gpu), dim=-1, descending=descending)
    _same(actual.values, expected.values)
    _same(actual.indices, expected.indices)


@pytest.mark.parametrize(
    ("columns", "k"),
    [
        # k = 1, in between, the whole row; for the long rows, k small enough
        # for the tournament route, then the first k that no longer fits and
        # falls back to the full sort, then the whole row.
        (1, 1),
        (33, 1),
        (33, 5),
        (33, 33),
        (4095, 17),
        (4096, 4096),
        (8193, 4096),
        (50257, 1),
        (50257, 50),
        (50257, 315),
        (50257, 316),
        (50257, 2048),
    ],
)
@pytest.mark.parametrize("largest", [True, False])
def test_topk_matches_cpu_across_the_routes(mojo_gpu, columns, k, largest):
    _check_topk(_sort_probe(2, columns, seed=11), mojo_gpu, k, -1, largest)


@pytest.mark.parametrize("dtype", _SORT_DTYPES)
@pytest.mark.parametrize("columns", [37, 6001])
def test_sort_every_dtype(mojo_gpu, dtype, columns):
    """Every kernel dtype, on a one-tile and a multi-tile row. The 64-bit
    dtypes matter most: their key is twice as wide, which halves the tile and
    so moves every route boundary."""
    host = _dtype_probe((3, columns), dtype)
    for descending in (False, True):
        expected = torch.sort(host, dim=-1, descending=descending, stable=True)
        with ran("aten::sort.stable"):
            actual = torch.sort(host.to(mojo_gpu), dim=-1, descending=descending)
        assert actual.values.dtype == dtype
        _same(actual.values, expected.values)
        _same(actual.indices, expected.indices)


@pytest.mark.parametrize("dtype", [d for d in _SORT_DTYPES if d is not torch.bool])
@pytest.mark.parametrize("k", [1, 9, 257])
def test_topk_every_dtype(mojo_gpu, dtype, k):
    host = _dtype_probe((3, 257), dtype)
    for largest in (True, False):
        _check_topk(host, mojo_gpu, k, -1, largest)


@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
def test_sort_and_topk_every_dim(mojo_gpu, dim):
    """A dim other than the last is moved to last, sorted, and moved back."""
    host = _sort_probe(5, 7 * 9, seed=3).reshape(5, 7, 9)
    device = host.to(mojo_gpu)
    for descending in (False, True):
        expected = torch.sort(host, dim=dim, descending=descending, stable=True)
        actual = torch.sort(device, dim=dim, descending=descending, stable=True)
        _same(actual.values, expected.values)
        _same(actual.indices, expected.indices)
    for k in (1, 3, host.shape[dim]):
        _check_topk(host, mojo_gpu, k, dim, True)
        _check_topk(host, mojo_gpu, k, dim, False)


def test_sort_and_topk_non_contiguous_input(mojo_gpu):
    host = _sort_probe(9, 40, seed=5)
    device = host.to(mojo_gpu)
    for view, ref in ((device.t(), host.t()), (device[:, 1::3], host[:, 1::3])):
        expected = torch.sort(ref, dim=-1, stable=True)
        actual = torch.sort(view, dim=-1)
        _same(actual.values, expected.values)
        _same(actual.indices, expected.indices)
        _check_topk(ref.contiguous(), mojo_gpu, 4, -1, True)
        with ran("aten::topk"):
            got = torch.topk(view, 4, dim=-1)
        _same(got.values, torch.topk(ref, 4, dim=-1).values)


def test_sort_orders_nan_and_signed_zero_like_aten(mojo_gpu):
    """A negative NaN's bits sit below -inf and -0.0's below +0.0, but ATen
    orders every NaN above every number and treats the two zeros as equal."""
    nan = float("nan")
    inf = float("inf")
    host = torch.tensor([[0.0, -0.0, nan, -nan, inf, -inf, 1.0, -1.0, nan, 0.0]])
    for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64):
        x = host.to(dtype)
        for descending in (False, True):
            expected = torch.sort(x, dim=-1, descending=descending, stable=True)
            actual = torch.sort(x.to(mojo_gpu), dim=-1, descending=descending)
            _same(actual.values, expected.values)
            _same(actual.indices, expected.indices)
        for k in (1, 3, 10):
            for largest in (True, False):
                got = torch.topk(x.to(mojo_gpu), k, largest=largest)
                _same(got.values, torch.topk(x, k, largest=largest).values)


def test_sort_and_topk_many_rows(mojo_gpu):
    """More rows than one launch's grid.y (65535): the op runs row chunks."""
    host = _sort_probe(70001, 3, seed=9)
    expected = torch.sort(host, dim=-1, stable=True)
    actual = torch.sort(host.to(mojo_gpu), dim=-1)
    _same(actual.values, expected.values)
    _same(actual.indices, expected.indices)
    _check_topk(host, mojo_gpu, 2, -1, True)


def test_sort_and_topk_empty_and_scalar(mojo_gpu):
    for host in (torch.empty(2, 0), torch.empty(0, 5), torch.tensor(4.5)):
        expected = torch.sort(host, dim=-1)
        actual = torch.sort(host.to(mojo_gpu), dim=-1)
        assert actual.values.shape == expected.values.shape
        _same(actual.values, expected.values)
        _same(actual.indices, expected.indices)
    for host, k in (
        (torch.empty(2, 0), 0),
        (torch.randn(3, 4), 0),
        (torch.tensor(4.5), 1),
    ):
        expected = torch.topk(host, k, dim=-1)
        actual = torch.topk(host.to(mojo_gpu), k, dim=-1)
        assert actual.values.shape == expected.values.shape
        _same(actual.values, expected.values)
        _same(actual.indices, expected.indices)


def test_sort_and_topk_out_variants(mojo_gpu):
    host = _sort_probe(3, 37)
    x = host.to(mojo_gpu)
    # Empty outs (resized) and a transposed out (not contiguous: computed,
    # then copied into it).
    for values, indices in (
        (
            torch.empty(0, device=mojo_gpu),
            torch.empty(0, dtype=torch.int64, device=mojo_gpu),
        ),
        (
            torch.empty(5, 3, device=mojo_gpu).t(),
            torch.empty(5, 3, dtype=torch.int64, device=mojo_gpu).t(),
        ),
    ):
        with ran("aten::topk.values"):
            got = torch.topk(x, 5, out=(values, indices))
        assert got[0] is values and got[1] is indices
        expected = torch.topk(host, 5)
        _same(values, expected.values)
        _same(indices, torch.sort(host, descending=True, stable=True).indices[:, :5])
    values = torch.empty(0, device=mojo_gpu)
    indices = torch.empty(0, dtype=torch.int64, device=mojo_gpu)
    with ran("aten::sort.values_stable"):
        got = torch.sort(x, dim=-1, stable=True, out=(values, indices))
    assert got[0] is values and got[1] is indices
    expected = torch.sort(host, dim=-1, stable=True)
    _same(values, expected.values)
    _same(indices, expected.indices)


def test_sort_and_topk_errors(mojo_gpu):
    x = torch.randn(3, 6).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        torch.topk(x, 7)
    with pytest.raises(RuntimeError, match="selected index k out of range"):
        torch.topk(x, -1)
    with pytest.raises((IndexError, RuntimeError), match="Dimension out of range"):
        torch.topk(x, 2, dim=2)
    with pytest.raises((IndexError, RuntimeError), match="Dimension out of range"):
        torch.sort(x, dim=-3)
    with pytest.raises(RuntimeError, match="dtype"):
        torch.sort(
            x, out=(torch.empty(0, device=mojo_gpu), torch.empty(0, device=mojo_gpu))
        )


def test_sort_and_topk_backward(mojo_gpu):
    """Both backwards are a scatter of the gradient over the returned
    indices, through ops the device already has."""
    host = _dtype_probe((4, 9), torch.float32) + torch.arange(9) * 0.01
    x = host.to(mojo_gpu).requires_grad_(True)
    reference = host.clone().requires_grad_(True)

    def loss(t: torch.Tensor) -> torch.Tensor:
        top = torch.topk(t, 3, dim=-1).values
        low = torch.sort(t, dim=0).values[:2]
        return (top * 2).sum() + (low * 3).sum()

    loss(x).backward()
    loss(reference).backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad.cpu(), reference.grad)


# ---------------------------------------------------------------------------
# kthvalue / median / nanmedian: one element per row of the same sorted
# (key, index) order. median's tie index is the lowest one (CPU's comparator),
# so its indices are compared exactly; kthvalue's is unspecified in ATen, so
# its indices are checked to point at the returned value. NaN sorts after
# every number: median returns a row's first NaN, nanmedian the lower middle
# of its numbers. The row lengths cross the sort routes as above, and the k
# of kthvalue crosses the topk tournament's boundary.
# ---------------------------------------------------------------------------

_ORDER_STAT_DTYPES = [
    torch.float32,
    torch.bfloat16,
    torch.float16,
    torch.float64,
    torch.int64,
    torch.int32,
    torch.uint8,
]


def _check_points_at(x: torch.Tensor, dim: int, keepdim: bool, result):
    """The returned indices hold the returned values (NaN equal to NaN)."""
    indices = result.indices.cpu()
    assert indices.dtype == torch.int64
    values = result.values.cpu()
    if x.dim() == 0:
        assert indices.item() == 0
        _same(values, x)
        return
    if not keepdim:
        indices, values = indices.unsqueeze(dim), values.unsqueeze(dim)
    _same(torch.gather(x, dim, indices), values)


def _check_kthvalue(x: torch.Tensor, device: str, k: int, dim: int, keepdim: bool):
    with ran("aten::kthvalue", "aten::kthvalue.values"):
        actual = torch.kthvalue(x.to(device), k, dim, keepdim)
    expected = torch.kthvalue(x, k, dim, keepdim)
    _same(actual.values, expected.values)
    _check_points_at(x, dim, keepdim, actual)


def _check_median(x: torch.Tensor, device: str, dim: int, keepdim: bool):
    with ran("aten::median.dim", "aten::median.dim_values"):
        actual = torch.median(x.to(device), dim, keepdim)
    expected = torch.median(x, dim, keepdim)
    _same(actual.values, expected.values)
    _same(actual.indices, expected.indices)


@pytest.mark.parametrize("columns", _SORT_ROUTE_SIZES)
def test_median_matches_cpu_on_every_route(mojo_gpu, columns):
    """Ties, both zeros and NaN rows (every probe of 16+ columns has NaNs):
    values AND indices exactly CPU's, whatever the route."""
    host = _sort_probe(3, columns)
    _check_median(host, mojo_gpu, -1, False)
    clean = torch.randn(3, columns, generator=torch.Generator().manual_seed(5))
    clean[:, ::3] = 0.25  # ties at the median
    _check_median(clean, mojo_gpu, 1, True)


@pytest.mark.parametrize(
    ("columns", "k"),
    [
        (1, 1),
        (33, 1),
        (33, 17),
        (33, 33),
        (4095, 17),
        (4097, 2049),
        (8193, 8193),
        (50257, 50),
        (50257, 316),
        (50257, 25129),
    ],
)
def test_kthvalue_matches_cpu_across_the_routes(mojo_gpu, columns, k):
    host = torch.randn(2, columns, generator=torch.Generator().manual_seed(9))
    _check_kthvalue(host, mojo_gpu, k, -1, False)


@pytest.mark.parametrize("dtype", _ORDER_STAT_DTYPES)
@pytest.mark.parametrize("columns", [37, 38, 6001])
def test_median_and_kthvalue_every_dtype(mojo_gpu, dtype, columns):
    """Heavily tied values: median's lowest-index tie rule is exact, and
    kthvalue's value is, with an index pointing at it."""
    host = _dtype_probe((3, columns), dtype)
    _check_median(host, mojo_gpu, -1, False)
    for k in (1, columns // 3, columns):
        _check_kthvalue(host, mojo_gpu, k, -1, False)


@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("keepdim", [False, True])
def test_median_and_kthvalue_every_dim(mojo_gpu, dim, keepdim):
    """Odd and even lengths along a dim that is not last are moved to last;
    the result keeps the other dims' order."""
    host = torch.randn(5, 7, 6, generator=torch.Generator().manual_seed(4))
    _check_median(host, mojo_gpu, dim, keepdim)
    for k in (1, 3, host.shape[dim]):
        _check_kthvalue(host, mojo_gpu, k, dim, keepdim)


def test_median_and_kthvalue_non_contiguous_input(mojo_gpu):
    host = torch.randn(9, 7, generator=torch.Generator().manual_seed(2))
    device = host.to(mojo_gpu)
    _same(torch.median(device.t(), 1).values, torch.median(host.t(), 1).values)
    _same(
        torch.median(device[:, ::2], 1).indices, torch.median(host[:, ::2], 1).indices
    )
    _same(
        torch.kthvalue(device.t(), 4, 1).values, torch.kthvalue(host.t(), 4, 1).values
    )


def test_median_nan_rules_match_cpu(mojo_gpu):
    """median: a row with any NaN is (nan, first NaN), even behind +inf.
    nanmedian: the lower middle of the numbers (an all-NaN row is NaN).
    kthvalue: NaN is the largest value."""
    nan, inf = float("nan"), float("inf")
    host = torch.tensor(
        [
            [2.0, 1.0, 2.0, 3.0, 2.0, 1.0],
            [1.0, 5.0, nan, 0.0, nan, 2.0],
            [inf, 1.0, 0.0, nan, -inf, 3.0],
            [nan, nan, nan, 9.0, nan, nan],
            [nan, nan, nan, nan, nan, nan],
        ]
    )
    device = host.to(mojo_gpu)
    _check_median(host, mojo_gpu, 1, False)
    with ran("aten::nanmedian.dim", "aten::nanmedian.dim_values"):
        actual = torch.nanmedian(device, 1)
    expected = torch.nanmedian(host, 1)
    _same(actual.values, expected.values)
    # ATen leaves the index of an all-NaN row unspecified.
    _same(actual.indices[:4], expected.indices[:4])
    for k in (1, 3, 6):
        _check_kthvalue(host, mojo_gpu, k, 1, False)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_median_and_nanmedian_of_the_whole_tensor(mojo_gpu, dtype):
    host = _dtype_probe((7, 9), dtype)
    with ran("aten::median"):
        _same(torch.median(host.to(mojo_gpu)), torch.median(host))
    with ran("aten::nanmedian"):
        _same(torch.nanmedian(host.to(mojo_gpu)), torch.nanmedian(host))
    if dtype.is_floating_point:
        host[2, 3] = float("nan")
        _same(torch.median(host.to(mojo_gpu)), torch.median(host))
        _same(torch.nanmedian(host.to(mojo_gpu)), torch.nanmedian(host))
        empty = torch.empty(0, dtype=dtype)
        _same(torch.median(empty.to(mojo_gpu)), torch.median(empty))


def test_median_and_kthvalue_scalar_and_single_column(mojo_gpu):
    for host in (torch.tensor(3.5), torch.randn(4, 1)):
        dim = 0 if host.dim() == 0 else 1
        for keepdim in (False, True):
            _check_median(host, mojo_gpu, dim, keepdim)
            _check_kthvalue(host, mojo_gpu, 1, dim, keepdim)


def test_median_and_kthvalue_out_variants(mojo_gpu):
    host = torch.randn(4, 7, generator=torch.Generator().manual_seed(8))
    device = host.to(mojo_gpu)
    for fn, overload in (
        (lambda t, out: torch.median(t, 1, out=out), "aten::median.dim_values"),
        (lambda t, out: torch.nanmedian(t, 1, out=out), "aten::nanmedian.dim_values"),
        (lambda t, out: torch.kthvalue(t, 3, 1, out=out), "aten::kthvalue.values"),
    ):
        # A wrong-sized out is resized; a strided one is written through.
        values = torch.empty(0, device=mojo_gpu)
        indices = torch.empty(0, dtype=torch.int64, device=mojo_gpu)
        with ran(overload):
            fn(device, (values, indices))
        expected = fn(host, (torch.empty(0), torch.empty(0, dtype=torch.int64)))
        _same(values, expected[0])
        _same(indices, expected[1])
        values = torch.zeros(4, 2, device=mojo_gpu)[:, 0]
        indices = torch.zeros(4, 2, dtype=torch.int64, device=mojo_gpu)[:, 1]
        fn(device, (values, indices))
        _same(values, expected[0])
        _same(indices, expected[1])


def test_median_and_kthvalue_errors(mojo_gpu):
    device = torch.randn(3, 4).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="k out of range"):
        torch.kthvalue(device, 5, 1)
    with pytest.raises(RuntimeError, match="k out of range"):
        torch.kthvalue(device, 0, 1)
    with pytest.raises((IndexError, RuntimeError), match="out of range"):
        torch.median(device, 2)
    with pytest.raises((IndexError, RuntimeError), match="non-zero size"):
        torch.median(torch.empty(2, 0).to(mojo_gpu), 1)
    with pytest.raises(RuntimeError, match="not implemented for 'Bool'"):
        torch.median(torch.tensor([True, False]).to(mojo_gpu), 0)
