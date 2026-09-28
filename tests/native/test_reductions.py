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

from tests.native.conftest import is_metal, ran, skip_if_metal
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
    "nansum": lambda t, **kw: torch.nansum(t, **kw),
    "mean": lambda t, **kw: torch.mean(t, **kw),
    "prod": lambda t, **kw: torch.prod(t, **kw),
    "amax": lambda t, **kw: torch.amax(t, **kw),
    "amin": lambda t, **kw: torch.amin(t, **kw),
    "norm": lambda t, **kw: torch.linalg.vector_norm(t, **kw),
    "norminf": lambda t, **kw: torch.linalg.vector_norm(t, ord=math.inf, **kw),
    "norm_l1": lambda t, **kw: torch.linalg.vector_norm(t, ord=1, **kw),
    "norm_neginf": lambda t, **kw: torch.linalg.vector_norm(t, ord=float("-inf"), **kw),
    "norm_p3": lambda t, **kw: torch.linalg.vector_norm(t, ord=3, **kw),
    "norm_pneg": lambda t, **kw: torch.linalg.vector_norm(t, ord=-1.5, **kw),
    "all": lambda t, **kw: torch.all(t, **kw),
    "any": lambda t, **kw: torch.any(t, **kw),
    "count_nonzero": lambda t, **kw: torch.count_nonzero(t, **kw),
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
    if op in ("all", "any", "count_nonzero"):
        x = torch.rand(shape) < 0.5
    elif op == "prod":
        # Values near 1: a product over millions of elements from the [0.05,
        # 0.95) range used below underflows to 0 on BOTH legs (device and the
        # fp64 reference), which would pass trivially without checking that
        # the split/merge path actually combines partial products correctly.
        x = 1.0 + (torch.rand(shape) - 0.5) * 0.002
    else:
        x = torch.rand(shape) * 0.9 + 0.05
    ours = fn(x.to(mojo_gpu), dim=dim).cpu()
    if op in ("all", "any", "count_nonzero"):
        torch.testing.assert_close(ours, fn(x, dim=dim))
    else:
        # fp64 reference on the same values: this measures the reduction
        # order, not the input dtype.
        expected = fn(x.double(), dim=dim)
        if op == "prod":
            # Unlike a sum, a product compounds one float32 rounding error
            # PER MULTIPLY: the relative error random-walks as
            # sqrt(reduce extent) * eps32, ~1e-4 at the million-element end
            # of these shapes -- looser than the other ops' shared tolerance.
            torch.testing.assert_close(ours.double(), expected, atol=1e-5, rtol=3e-3)
        else:
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


@pytest.mark.parametrize("op", ["all", "any"])
@pytest.mark.parametrize("dim_kw", [{}, {"dim": 1}, {"dim": [0, 1]}])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bool, torch.int32, torch.float32])
def test_any_all_uint8_input_keeps_uint8_output(mojo_gpu, op, dim_kw, dtype):
    """torch's uint8 compatibility (ReduceOps.cpp, Note "[all, any :
    uint8 compatibility]"): a uint8 input's any/any.dim/any.dims (and all's)
    result stays uint8; every other truthy dtype still narrows to bool."""
    fn = torch.all if op == "all" else torch.any
    if dtype is torch.bool:
        x = torch.randint(0, 2, (3, 4, 5), dtype=dtype)
    else:
        x = torch.randint(0, 3, (3, 4, 5), dtype=dtype)
    xd = x.to(mojo_gpu)
    expected = fn(x, **dim_kw)
    got = fn(xd, **dim_kw)
    want_dtype = torch.uint8 if dtype is torch.uint8 else torch.bool
    assert got.dtype == expected.dtype == want_dtype
    torch.testing.assert_close(got.cpu(), expected)


def test_any_all_uint8_output_values_are_0_or_1(mojo_gpu):
    """The uint8-kept result carries the same 0/1 payload as the bool one,
    not the raw int32 accumulator."""
    x = torch.tensor([[2, 0, 0], [0, 0, 0], [3, 5, 0]], dtype=torch.uint8)
    xd = x.to(mojo_gpu)
    got_any = torch.any(xd, dim=1).cpu()
    assert got_any.dtype == torch.uint8
    torch.testing.assert_close(got_any, torch.tensor([1, 0, 1], dtype=torch.uint8))
    got_all = torch.all(xd, dim=1).cpu()
    assert got_all.dtype == torch.uint8
    torch.testing.assert_close(got_all, torch.tensor([0, 0, 0], dtype=torch.uint8))


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


def test_nansum_dtype_promotion(mojo_gpu):
    """Same promotion rules as `sum` (bool/int -> int64, `dtype=` casts before
    the reduction) -- integral inputs have no NaN to remove, so nansum and sum
    must agree on them exactly."""
    for dtype in (torch.bool, torch.uint8, torch.int32):
        if dtype is torch.bool:
            i = torch.randint(0, 2, (8, 16), dtype=dtype)
        else:
            i = torch.randint(0, 20, (8, 16), dtype=dtype)
        got = i.to(mojo_gpu).nansum(dim=1)
        assert got.dtype == torch.int64 == i.nansum(dim=1).dtype
        torch.testing.assert_close(got.cpu(), i.nansum(dim=1))

    # dtype= casts the input BEFORE the reduction: summing two 40000s in
    # float16 overflows to inf, but casting to float32 first (as torch does)
    # sums them to exactly 80000.
    y = torch.tensor([[40000.0, 40000.0]], dtype=torch.float16)
    ours = torch.nansum(y.to(mojo_gpu), dim=1, dtype=torch.float32)
    expected = torch.nansum(y, dim=1, dtype=torch.float32)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(ours.cpu(), expected)

    # An explicit dtype=int32 is declined (`_is_sum_dtype`, same as sum's own
    # dtype= gate) regardless of the input -- use an INTEGRAL input here so
    # this actually exercises that gate, rather than the floating-input ->
    # integral-dtype rejection below (which would fire first on `y`).
    i32 = torch.randint(0, 20, (4, 5), dtype=torch.int32)
    with pytest.raises(NotImplementedError):
        torch.nansum(i32.to(mojo_gpu), dim=1, dtype=torch.int32)

    # An explicit integral dtype on a floating input is declined outright:
    # this backend has no `nan_to_num` kernel to zero the NaN before the cast
    # would otherwise truncate it to an arbitrary integer.
    with pytest.raises(NotImplementedError):
        torch.nansum(y.to(mojo_gpu), dim=1, dtype=torch.int64)


def test_nansum_out_variant_and_noncontiguous(mojo_gpu):
    x = torch.tensor([[1.0, float("nan"), 3.0], [float("nan"), float("nan"), 6.0]])
    xd = x.to(mojo_gpu)

    expected = x.nansum(dim=1)
    out = torch.empty(0, dtype=torch.float32, device=mojo_gpu)
    returned = torch.nansum(xd, dim=1, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert tuple(out.shape) == tuple(expected.shape)
    torch.testing.assert_close(out.cpu(), expected)

    # Non-contiguous input: not an adjacent-ascending interval, so this takes
    # the permute + materialize path (`_middle_direct_ok` requires
    # contiguity), not the strided-axis kernel.
    y = torch.randn(5, 7)
    y[1, 2] = float("nan")
    yt = y.t()
    yt_device = y.to(mojo_gpu).t()
    torch.testing.assert_close(yt_device.nansum(dim=0).cpu(), yt.nansum(dim=0))


def test_nansum_out_computes_in_outs_dtype(mojo_gpu):
    """Same `_out_reduce_dtype`/`_promote_for_out_reduction` path as
    sum.IntList_out/mean.out: with no `dtype=`, nansum.out accumulates in
    `out`'s own dtype, not the input's -- summing two 40000s in float16
    overflows to inf, but widened to float32 first it is exactly 80000."""
    y = torch.tensor([40000.0, 40000.0], dtype=torch.float16)
    assert torch.isinf(y.sum())  # accumulating (and rounding) in float16 overflows
    yd = y.to(mojo_gpu)
    out = torch.empty((), dtype=torch.float32, device=mojo_gpu)
    torch.nansum(yd, dim=0, out=out)
    assert out.item() == 80000.0


def test_nansum_out_dtype_must_match_out_dtype(mojo_gpu):
    """An explicit `dtype=` that disagrees with `out`'s dtype raises, rather
    than silently using either one (same rule as sum.IntList_out/mean.out)."""
    y = torch.tensor([40000.0, 40000.0], dtype=torch.float16).to(mojo_gpu)
    with pytest.raises(RuntimeError):
        torch.nansum(
            y,
            dim=0,
            dtype=torch.float32,
            out=torch.empty((), dtype=torch.float16, device=mojo_gpu),
        )


def test_nansum_out_float64_out_and_nan(mojo_gpu):
    """A float64 `out` with no `dtype=`: NaN is zeroed and the rest sums in
    double (mirrors sum.IntList_out's cast-then-reduce, `nansum` semantics
    with a NaN present)."""
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = torch.randn(4, 5, dtype=torch.float64)
    x[0, 0] = float("nan")
    xd = x.to(mojo_gpu)
    out = torch.empty(4, dtype=torch.float64, device=mojo_gpu)
    torch.nansum(xd, dim=1, out=out)
    torch.testing.assert_close(out.cpu(), torch.nansum(x, dim=1))


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
        torch.int16,
        torch.int8,
        torch.uint8,
        torch.bool,
    ],
)
def test_count_nonzero_dtypes(mojo_gpu, dtype):
    """Every dtype the reduce_skeleton TRUTHY set accepts, output always
    int64 (torch's count_nonzero.dim_IntList_out(self, dim, out=out) result
    dtype)."""
    if dtype is torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    if dtype is torch.bool:
        x = torch.rand(6, 9) < 0.5
    elif dtype.is_floating_point:
        x = (torch.rand(6, 9) - 0.5).to(dtype)
    elif dtype is torch.uint8:
        x = torch.randint(0, 4, (6, 9), dtype=dtype)
    else:
        x = torch.randint(-3, 4, (6, 9), dtype=dtype)
    xd = x.to(mojo_gpu)
    for dim in (None, 0, 1, -1, (0, 1)):
        got = torch.count_nonzero(xd, dim=dim)
        expected = torch.count_nonzero(x, dim=dim)
        assert got.dtype == torch.int64
        torch.testing.assert_close(got.cpu(), expected)


def test_count_nonzero_empty_dim_list_reduces_all(mojo_gpu):
    """`count_nonzero.dim_IntList(self, [])` reduces every dim (unlike
    any.dims/all.dims, where an explicit empty list reduces nothing); this is
    also what `count_nonzero(self, dim=None)` redispatches to."""
    x = torch.randn(4, 5)
    xd = x.to(mojo_gpu)
    torch.testing.assert_close(
        torch.ops.aten.count_nonzero.dim_IntList(xd, []).cpu(),
        torch.ops.aten.count_nonzero.dim_IntList(x, []),
    )
    torch.testing.assert_close(torch.count_nonzero(xd).cpu(), torch.count_nonzero(x))


def test_count_nonzero_nan_counts_as_nonzero(mojo_device):
    """NaN is nonzero under `not (x == 0)` (matches any/all's rule)."""
    x = torch.tensor([[1.0, 0.0, float("nan"), 0.0, float("nan")]])
    torch.testing.assert_close(
        torch.count_nonzero(x.to(mojo_device), dim=1).cpu(),
        torch.count_nonzero(x, dim=1),
    )


def test_count_nonzero_noncontiguous_and_empty(mojo_device):
    """A deterministic zero/nonzero pattern whose per-row and per-column
    counts differ, reduced along both axes of a transposed (non-contiguous)
    view -- random normals are virtually always nonzero, so a stride bug
    that skips or duplicates elements would not change the count and would
    go undetected."""
    x = torch.tensor(
        [
            [1.0, 0.0, 3.0, 0.0, 5.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    xd = x.to(mojo_device)
    torch.testing.assert_close(
        torch.count_nonzero(xd.t(), dim=0).cpu(), torch.count_nonzero(x.t(), dim=0)
    )
    torch.testing.assert_close(
        torch.count_nonzero(xd.t(), dim=1).cpu(), torch.count_nonzero(x.t(), dim=1)
    )
    e = torch.empty(0)
    ed = e.to(mojo_device)
    torch.testing.assert_close(
        torch.count_nonzero(ed, dim=0).cpu(), torch.count_nonzero(e, dim=0)
    )


@pytest.mark.parametrize("dim", [None, [], 0, -1])
def test_count_nonzero_rank0(mojo_device, dim):
    """A rank-0 operand has one (virtual) element and no real axis to
    iterate: `_reduce_dims` normalizes any of these dim specs to empty and
    the skeleton reduces the single element (see `_reduce_dims`'s rank-0
    case)."""
    x = torch.tensor(5.0)
    xd = x.to(mojo_device)
    expected = torch.count_nonzero(x, dim=dim)
    got = torch.count_nonzero(xd, dim=dim)
    assert got.shape == expected.shape == ()
    torch.testing.assert_close(got.cpu(), expected)


def test_count_nonzero_rank0_out_of_range_dim_raises(mojo_device):
    x = torch.tensor(5.0)
    xd = x.to(mojo_device)
    match = (
        "Dimension out of range \\(expected to be in range of \\[-1, 0\\], but got 1\\)"
    )
    with pytest.raises(IndexError, match=match):
        torch.count_nonzero(x, dim=1)
    with pytest.raises(IndexError, match=match):
        torch.count_nonzero(xd, dim=1)


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


def test_sum_out_variant_into_int64_truncates_each_element_first(mojo_gpu):
    """Verified against stock torch: a float result poured into an int64 out
    with no `dtype=` is NOT a "sum then cast" -- torch truncates every
    element to int64 first (`ScalarType dtype = result.scalar_type();`, used
    as the reduction's own compute dtype), so this differs from
    `x.sum(dim=1).to(int64)` whenever fractional parts would otherwise
    accumulate before truncation."""
    x = torch.tensor([[0.9, 0.9, 0.9]])
    expected = x.to(torch.int64).sum(dim=1)
    assert expected.item() != x.sum(dim=1).to(torch.int64).item()  # the two must differ
    out = torch.empty(1, dtype=torch.int64, device=mojo_gpu)
    torch.sum(x.to(mojo_gpu), dim=1, out=out)
    torch.testing.assert_close(out.cpu(), expected)


def test_sum_out_variant_computes_in_outs_dtype_for_int_input_too(mojo_gpu):
    """`out`'s dtype is the compute dtype for ANY self, not just a floating
    one: an int64 self summed into a float32 `out` casts each element to
    float32 first, same as a float self does (test above). It must NOT
    accumulate in int64 (self's own dtype) and cast the sum down afterward.
    Verified on stock CUDA: [16777217, -16777216] (adjacent int64 values that
    collapse to the same float32) sums to 0.0 in float32, not 1.0 from an
    int64 sum cast down afterward."""
    x = torch.tensor([16777217, -16777216], dtype=torch.int64)
    xd = x.to(mojo_gpu)
    assert x.sum().item() == 1  # int64 sum-then-cast would give 1.0, not 0.0
    out = torch.empty((), dtype=torch.float32, device=mojo_gpu)
    torch.sum(xd, dim=0, out=out)
    assert out.item() == 0.0


def test_sum_out_variant_declines_dtypes_the_kernel_lacks(mojo_gpu):
    """SumSpec only accumulates in float16/bfloat16/float32/int64
    (`_is_sum_dtype`); bool/uint8 outs, which torch itself accepts for
    sum.out, are declined rather than silently mishandled."""
    x = torch.randn(4, 5).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.sum(x, dim=1, out=torch.empty(4, dtype=torch.bool, device=mojo_gpu))
    with pytest.raises(NotImplementedError):
        torch.sum(x, dim=1, out=torch.empty(4, dtype=torch.uint8, device=mojo_gpu))


def test_sum_out_computes_in_outs_dtype(mojo_gpu):
    """Same `_out_reduce_dtype`/`_promote_for_out_reduction` path as mean.out:
    with no `dtype=`, sum.IntList_out accumulates in `out`'s own dtype, not
    the input's (values picked so a float16 accumulation would round
    differently than the float32 one torch actually does)."""
    x = torch.tensor([1.0, 1.0009765625], dtype=torch.float16)
    xd = x.to(mojo_gpu)
    expected = x.float().sum(dim=0)
    assert expected.item() != x.sum(dim=0).float().item()  # the two must differ
    out = torch.empty((), dtype=torch.float32, device=mojo_gpu)
    torch.sum(xd, dim=0, out=out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_sum_out_float64_out_accumulates_in_double(mojo_gpu):
    """A float64 `out` casts every element to double THEN sums (verified on
    real CUDA against a naive `.double().sum()`, which a float32-then-cast
    accumulation would miss by more than fp64 rounding)."""
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    torch.manual_seed(1)
    x = ((torch.rand(200_000) * 2 - 1) * 1e-3 + 1.0).float()
    xd = x.to(mojo_gpu)
    expected = x.double().sum()
    assert expected.item() != x.sum().double().item()  # the two must differ
    out = torch.empty((), dtype=torch.float64, device=mojo_gpu)
    torch.sum(xd, dim=0, out=out)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    # a float64 self needs no promotion at all.
    x64 = torch.randn(357, 789, dtype=torch.float64)
    out2 = torch.empty(357, dtype=torch.float64, device=mojo_gpu)
    torch.sum(x64.to(mojo_gpu), dim=1, out=out2)
    torch.testing.assert_close(out2.cpu(), x64.sum(dim=1))


@pytest.mark.parametrize("fn_name", ["sum", "prod", "nansum", "mean"])
def test_reduce_narrows_a_float64_self_with_explicit_dtype(mojo_gpu, fn_name):
    """`dtype=torch.float32` on a float64 self is an explicit NARROWING cast
    (unlike linalg_vector_norm/norm, sum/prod/nansum/mean have no widen-only
    restriction -- verified on real CUDA that all four accept it). This is
    also the one case the float64-on-Metal decline must catch by the SELF's
    dtype, not just the requested target: `_is_castable` used to check only
    `dtype=`'s target, so a float64 self with a non-float64 explicit dtype
    reached the cast kernel on an Apple GPU instead of declining cleanly.

    Asserts the decline itself on Metal (`is_metal`), rather than skipping
    past it, so a Mac run actually exercises this fix instead of never
    reaching it."""
    fn = getattr(torch, fn_name)
    x = torch.randn(4, 5, dtype=torch.float64)
    if is_metal(mojo_gpu):
        with pytest.raises(NotImplementedError):
            fn(x.to(mojo_gpu), dim=1, dtype=torch.float32)
        return
    expected = fn(x, dim=1, dtype=torch.float32)
    got = fn(x.to(mojo_gpu), dim=1, dtype=torch.float32).cpu()
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)


def test_sum_out_dtype_must_match_out_dtype(mojo_gpu):
    """Same equality rule as mean.out: an explicit `dtype=` that disagrees
    with `out`'s dtype raises, rather than silently using either one."""
    x = torch.randn(4, 5).to(mojo_gpu)
    with pytest.raises(RuntimeError):
        torch.sum(
            x,
            dim=1,
            dtype=torch.float32,
            out=torch.empty(4, dtype=torch.float16, device=mojo_gpu),
        )


def test_sum_rounds_each_element_to_the_target_dtype_first(mojo_gpu):
    """`TORCH_IMPL_FUNC(sum_out)`'s CUDA path (`make_reduction_from_out_ty`)
    builds its TensorIterator from `out`'s own dtype directly, so every
    element is rounded to it BEFORE accumulating (`SumOp`'s own float32
    `acc_dtype` then sums those already-rounded values). Verified on an
    actual CUDA device: summing [1 + 2**-12, -1] with dtype=float16 rounds
    1 + 2**-12 down to 1.0 first, giving exactly 0 -- not 2**-12 from summing
    at full precision and rounding only the final scalar."""
    x = torch.tensor([1.0 + 2**-12, -1.0])
    xd = x.to(mojo_gpu)
    assert (x.sum().half().item(), x.half().sum().item()) == (2**-12, 0.0)

    out = torch.empty((), dtype=torch.float16, device=mojo_gpu)
    torch.sum(xd, dim=0, dtype=torch.float16, out=out)
    assert out.item() == 0.0

    out2 = torch.empty((), dtype=torch.float16, device=mojo_gpu)
    torch.sum(xd, dim=0, out=out2)  # dtype=None: same rule, dtype comes from `out`
    assert out2.item() == 0.0


def test_mean_rounds_each_element_to_the_target_dtype_first(mojo_gpu):
    """Confirmed on an actual CUDA device (not CPU torch, whose `mean_out`
    has a CPU-only `is_half_type` trick that avoids this): mean's CUDA path
    is the exact same `make_reduction_from_out_ty` machinery as sum (above),
    so it rounds every element to the target dtype BEFORE accumulating too --
    `torch.mean(torch.tensor([1 + 2**-12, -1], device="cuda"),
    dtype=torch.float16)` gives 0, not the CPU-only 2**-13. The mojo device
    mirrors CUDA, not CPU."""
    x = torch.tensor([1.0 + 2**-12, -1.0])
    xd = x.to(mojo_gpu)
    assert (x.mean().half().item(), x.half().mean().item()) == (2**-13, 0.0)

    out = torch.empty((), dtype=torch.float16, device=mojo_gpu)
    torch.mean(xd, dtype=torch.float16, out=out)  # -> mean.dtype_out
    assert out.item() == 0.0

    out2 = torch.empty((), dtype=torch.float16, device=mojo_gpu)
    torch.mean(xd, dim=0, out=out2)  # dtype=None: same rule -> mean.out
    assert out2.item() == 0.0


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


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16]
)
def test_mean_dtype_out_full_reduce(mojo_gpu, dtype):
    """`torch.mean(x, out=out)` with no `dim` dispatches to mean.dtype_out
    (verified against stock torch's own overload resolution), always a full
    reduce to a 0-d result regardless of input rank."""
    if dtype is torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = torch.randn(4, 6, 5, dtype=dtype)
    xd = x.to(mojo_gpu)
    expected = x.mean()
    out = torch.empty((), dtype=dtype, device=mojo_gpu)
    returned = torch.mean(xd, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected, rtol=2e-2, atol=2e-2)


def test_mean_dtype_out_casts_before_reducing(mojo_gpu):
    """`dtype=` promotes the input before reducing (float16/bfloat16/
    float32/float64, same as mean()/mean.out/mean.dim). These two
    values are picked so summing them AS float16 (cast-after-reduce, the bug)
    rounds to a different float16-representable pair than summing them AS
    float32 (cast-before-reduce, what torch does): a loose tolerance cannot
    tell the two apart, so this uses an exact comparison."""
    x = torch.tensor([1.0, 1.0009765625], dtype=torch.float16)
    xd = x.to(mojo_gpu)
    expected = x.mean(dtype=torch.float32)
    assert expected.item() != x.half().mean().float().item()  # the two must differ
    out = torch.empty((), dtype=torch.float32, device=mojo_gpu)
    torch.mean(xd, dtype=torch.float32, out=out)  # no dim -> mean.dtype_out
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_mean_dtype_out_resizes_a_mismatching_out(mojo_gpu):
    """Non-scalar `out` is resized to the full-reduce (0-d) shape."""
    x = torch.randn(3, 4)
    out = torch.empty(3, 4, device=mojo_gpu)
    torch.mean(x.to(mojo_gpu), out=out)
    assert tuple(out.shape) == ()
    torch.testing.assert_close(out.cpu(), x.mean())


def test_mean_dtype_out_rejects_integer_input(mojo_gpu):
    x = torch.randint(0, 10, (3, 4))
    out = torch.empty((), device=mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.mean(x.to(mojo_gpu), out=out)


def test_mean_out_computes_in_outs_dtype(mojo_gpu):
    """With no `dtype=`, torch computes mean.out/mean.dtype_out in `out`'s
    OWN dtype (`ReduceOps.cpp`: `ScalarType dtype = result.scalar_type();`),
    not the input's: a float16 input poured into a float32 `out` must match
    the float32-accumulated answer exactly, not a float16-rounded one poured
    into fp32 (the values below are picked so the two differ)."""
    x = torch.tensor([1.0, 1.0009765625], dtype=torch.float16)
    xd = x.to(mojo_gpu)
    expected = x.float().mean()
    assert expected.item() != x.mean().float().item()  # the two must differ

    out_dtype_out = torch.empty((), dtype=torch.float32, device=mojo_gpu)
    torch.mean(xd, out=out_dtype_out)  # -> mean.dtype_out
    torch.testing.assert_close(out_dtype_out.cpu(), expected, rtol=0, atol=0)

    out_dim_out = torch.empty((), dtype=torch.float32, device=mojo_gpu)
    torch.mean(xd, dim=0, out=out_dim_out)  # -> mean.out
    torch.testing.assert_close(out_dim_out.cpu(), expected, rtol=0, atol=0)


def test_mean_out_float64_out_accumulates_in_double(mojo_gpu):
    """A float64 `out` with no `dtype=` computes mean ENTIRELY in double
    (cast every element to double, then accumulate in double) -- verified on
    real CUDA: it does not merely round a float32-accumulated answer. Values
    picked so a float32 accumulation and a double one visibly differ."""
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    torch.manual_seed(1)
    x = ((torch.rand(200_000) * 2 - 1) * 1e-3 + 1.0).float()
    xd = x.to(mojo_gpu)
    expected = x.double().mean()
    assert expected.item() != x.mean().double().item()  # the two must differ

    out = torch.empty((), dtype=torch.float64, device=mojo_gpu)
    torch.mean(xd, out=out)  # -> mean.dtype_out
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    out2 = torch.empty((), dtype=torch.float64, device=mojo_gpu)
    torch.mean(xd, dim=0, out=out2)  # -> mean.out
    torch.testing.assert_close(out2.cpu(), expected, rtol=0, atol=0)


def test_mean_out_declines_mismatched_float64_out(mojo_gpu):
    """Unlike a float64 `out` with no `dtype=` (computed in double, above), a
    float32 input with an EXPLICIT `dtype=torch.float64` still declines here:
    `_out_reduce_dtype`'s policy takes `out`'s own dtype only when `dtype=`
    is not given, so this exercises `_check_out_dtype`'s exact-match rule."""
    x = torch.randn(4, 5).to(mojo_gpu)
    with pytest.raises(RuntimeError):
        torch.mean(
            x,
            dim=1,
            dtype=torch.float64,
            out=torch.empty(4, dtype=torch.float32, device=mojo_gpu),
        )


def test_mean_out_dtype_must_match_out_dtype(mojo_gpu):
    """torch requires an explicit `dtype=` to equal `out`'s dtype exactly and
    raises otherwise ("Expected out tensor to have dtype X, but got dtype Y
    instead"); it is not a safe-cast check."""
    x = torch.randn(4, 5).to(mojo_gpu)
    with pytest.raises(RuntimeError):
        torch.mean(
            x,
            dim=1,
            dtype=torch.float32,
            out=torch.empty(4, dtype=torch.float16, device=mojo_gpu),
        )
    with pytest.raises(RuntimeError):
        torch.mean(
            x,
            dtype=torch.float32,
            out=torch.empty((), dtype=torch.float16, device=mojo_gpu),
        )


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_mean_out_explicit_dtype_bypasses_the_int_input_check(mojo_gpu, dtype):
    """Verified on stock CUDA: an int/bool self with an explicit float
    `dtype=` is valid (self is cast to it before reducing) -- mean only
    requires self itself to be float/complex when `dtype=` is absent."""
    x = (
        torch.randint(0, 10, (4, 5), dtype=dtype)
        if dtype is not torch.bool
        else torch.randint(0, 2, (4, 5), dtype=dtype)
    )
    expected = x.float().mean(dim=0)
    out = torch.empty(5, dtype=torch.float32, device=mojo_gpu)
    torch.mean(x.to(mojo_gpu), dim=0, dtype=torch.float32, out=out)
    torch.testing.assert_close(out.cpu(), expected)


def test_mean_dtype_out_nan(mojo_gpu):
    x = torch.randn(4, 6)
    x[2, 3] = float("nan")
    out = torch.empty((), device=mojo_gpu)
    torch.mean(x.to(mojo_gpu), out=out)
    torch.testing.assert_close(out.cpu(), x.mean(), equal_nan=True)


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


def test_mean_out_declines_an_out_that_aliases_the_input(mojo_gpu):
    """See `_decline_aliasing_out` for why any aliasing `out=` declines."""
    x = torch.randn(3, 4, device=mojo_gpu)
    out = x.reshape(-1)[x.numel() :]
    with pytest.raises(NotImplementedError):
        torch.mean(x, dim=1, out=out)


# ---------------------------------------------------------------------------
# prod
# ---------------------------------------------------------------------------


def test_prod_full_reduce_and_dtype_promotion(mojo_gpu):
    """`prod()` (prod.default, no dim) and torch's bool/int -> int64 promotion,
    matching test_sum_full_reduce_and_dtype_promotion."""
    x = torch.rand(4, 5, generator=torch.Generator().manual_seed(0)) * 0.5 + 0.5
    torch.testing.assert_close(
        x.to(mojo_gpu).prod().cpu(), x.prod(), rtol=1e-5, atol=1e-6
    )

    for dtype in (torch.bool, torch.uint8, torch.int32):
        if dtype is torch.bool:
            i = torch.randint(0, 2, (3, 4), dtype=dtype)
        else:
            i = torch.randint(0, 3, (3, 4), dtype=dtype)
        got = i.to(mojo_gpu).prod(dim=1)
        assert got.dtype == torch.int64 == i.prod(dim=1).dtype
        torch.testing.assert_close(got.cpu(), i.prod(dim=1))

    # dtype= casts first: 1.7 * 2.7 * 3.7 as int64 is 1 * 2 * 3, not (int)16.983.
    # (an explicit dtype=int32 is declined -- see `_is_sum_dtype` -- so int64
    # is the dtype that exercises the cast-before-reduce rule here.)
    y = torch.tensor([[1.7, 2.7, 3.7]])
    ours = torch.prod(y.to(mojo_gpu), dim=1, dtype=torch.int64)
    torch.testing.assert_close(ours.cpu(), torch.prod(y, dim=1, dtype=torch.int64))


@pytest.mark.parametrize("keepdim", [False, True])
def test_prod_dim_int_and_out(mojo_gpu, keepdim):
    """prod.dim_int, and prod.int_out into a wrongly-shaped out that must be
    resized (`resize_output`, the same rule every out= op follows)."""
    x = torch.rand(6, 9) * 0.5 + 0.5
    xd = x.to(mojo_gpu)
    expected = x.prod(dim=1, keepdim=keepdim)
    ours = torch.prod(xd, dim=1, keepdim=keepdim)
    torch.testing.assert_close(ours.cpu(), expected, rtol=1e-5, atol=1e-6)

    out = torch.empty(0, device=mojo_gpu)
    returned = torch.prod(xd, dim=1, keepdim=keepdim, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert tuple(out.shape) == tuple(expected.shape)
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-6)


def test_prod_empty_reduce_axis_is_one(mojo_gpu):
    """An empty reduce axis contributes the identity: prod -> 1, unlike
    amax/amin, which torch refuses on an empty axis."""
    x = torch.empty(3, 0)
    expected = x.prod(dim=1)
    ours = torch.prod(x.to(mojo_gpu), dim=1).cpu()
    torch.testing.assert_close(ours, expected)

    # prod.default (no dim) over an all-empty-axes tensor is also the
    # identity, matching torch.
    y = torch.empty(0)
    torch.testing.assert_close(torch.prod(y.to(mojo_gpu)).cpu(), torch.prod(y))


def test_prod_noncontiguous(mojo_gpu):
    """A transposed operand exercises the permute-and-materialize fallback."""
    x = torch.rand(5, 8) * 0.5 + 0.5
    xt = x.t()
    assert not xt.is_contiguous()
    torch.testing.assert_close(
        xt.to(mojo_gpu).prod(dim=1).cpu(), xt.prod(dim=1), rtol=1e-5, atol=1e-6
    )


def test_prod_out_variant_computes_in_outs_dtype_for_int_input_too(mojo_gpu):
    """Same `_out_reduce_dtype`/`_promote_for_out_reduction` path as
    sum.IntList_out: with no `dtype=`, prod.int_out accumulates in `out`'s own
    dtype, not the input's. An int64 self must NOT be multiplied in int64
    (where 2**32 * 2**32 wraps to 0) and cast down afterward -- each element
    is cast to `out`'s dtype FIRST. Verified on stock CUDA: [2**32, 2**32] as
    int64 overflows to 0 in an int64 product, but 2**32 is an exact float32
    value and so is their product, giving exactly 2**64 in a float32 out."""
    x = torch.tensor([2**32, 2**32], dtype=torch.int64)
    assert x.prod().item() == 0  # the int64 product wraps to 0
    xd = x.to(mojo_gpu)
    out = torch.empty((), dtype=torch.float32, device=mojo_gpu)
    torch.prod(xd, dim=0, out=out)
    assert out.item() == 2.0**64


def test_prod_out_float64_out(mojo_gpu):
    """A float64 `out` with no `dtype=` casts every element to double, not
    float32, before multiplying: 2**24+1 is exact in double but rounds in
    float32, so a float32-then-cast accumulation would give a different
    (rounded) product than casting straight to double."""
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = torch.tensor([2**24 + 1, 2**24 + 1], dtype=torch.int64)
    expected = x.double().prod()
    assert expected.item() != x.float().prod().double().item()  # must differ
    out = torch.empty((), dtype=torch.float64, device=mojo_gpu)
    torch.prod(x.to(mojo_gpu), dim=0, out=out)
    assert out.item() == expected.item()

    x64 = torch.rand(4, 5, dtype=torch.float64) * 0.5 + 0.5
    out2 = torch.empty(4, dtype=torch.float64, device=mojo_gpu)
    torch.prod(x64.to(mojo_gpu), dim=1, out=out2)
    torch.testing.assert_close(out2.cpu(), torch.prod(x64, dim=1))


def test_prod_out_dtype_must_match_out_dtype(mojo_gpu):
    """Same equality rule as sum.IntList_out/mean.out: an explicit `dtype=`
    that disagrees with `out`'s dtype raises, rather than silently using
    either one."""
    x = torch.rand(4, 5).to(mojo_gpu) * 0.5 + 0.5
    with pytest.raises(RuntimeError):
        torch.prod(
            x,
            dim=1,
            dtype=torch.float32,
            out=torch.empty(4, dtype=torch.float16, device=mojo_gpu),
        )


def test_prod_out_variant_declines_dtypes_the_kernel_lacks(mojo_gpu):
    """ProdSpec only accumulates in float16/bfloat16/float32/int64
    (`_is_sum_dtype`, shared with sum); an int32 out -- which torch itself
    accepts for prod.int_out -- is declined rather than silently
    mishandled."""
    x = torch.rand(4, 5).to(mojo_gpu) * 0.5 + 0.5
    with pytest.raises(NotImplementedError):
        torch.prod(x, dim=1, out=torch.empty(4, dtype=torch.int32, device=mojo_gpu))


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
    match, no cast), unlike mean.out's safe_cast. A float32 self can't
    satisfy that exact match against a float64 out (verified on real CUDA:
    "Expected the dtype for input and out to match"), even though float64 is
    itself a supported amax dtype (see test_amax_out_float64 below)."""
    x = torch.randn(4, 5).to(mojo_gpu)
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.amax(x, dim=1, out=torch.empty(4, dtype=torch.float64, device=mojo_gpu))
    with pytest.raises(NotImplementedError, match="reduce dim of size 0"):
        torch.amax(
            torch.empty(4, 0, device=mojo_gpu),
            dim=1,
            out=torch.empty(4, device=mojo_gpu),
        )


def test_amax_out_float64(mojo_gpu):
    """A float64 self with a float64 out satisfies amax's exact-dtype policy
    (selection needs no accumulation dtype, so this is the whole float64
    story for amax/amin/max/min: EXTREMUM_DTYPES already covered them)."""
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = torch.randn(357, 789, dtype=torch.float64)
    out = torch.empty(357, dtype=torch.float64, device=mojo_gpu)
    torch.amax(x.to(mojo_gpu), dim=1, out=out)
    torch.testing.assert_close(out.cpu(), torch.amax(x, dim=1))


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.int64,
    ],
)
@pytest.mark.parametrize(
    "shape,dim", [((357, 789), 1), ((4, 5, 6), (0, 2)), ((37,), None)]
)
def test_amin_out(mojo_gpu, shape, dim, dtype, keepdim):
    """amin.out over dtypes/dims/keepdim, including the dim=None full reduce
    (empty dim list) and a non-contiguous input."""
    if dtype is torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    if dtype.is_floating_point:
        x = torch.randn(shape).to(dtype)
    else:
        x = torch.randint(-100, 100, shape, dtype=dtype)
    dim_args = (dim,) if dim is not None else ()
    expected = torch.amin(x, *dim_args, keepdim=keepdim)
    out = torch.empty(0, dtype=dtype, device=mojo_gpu)
    returned = torch.amin(x.to(mojo_gpu), *dim_args, out=out, keepdim=keepdim)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected)

    # non-contiguous input, out already correctly shaped
    if len(shape) >= 2:
        xt = x.transpose(0, 1)
        expected_t = torch.amin(xt, *dim_args, keepdim=keepdim)
        out2 = torch.empty(expected_t.shape, dtype=dtype, device=mojo_gpu)
        torch.amin(xt.to(mojo_gpu), *dim_args, out=out2, keepdim=keepdim)
        torch.testing.assert_close(out2.cpu(), expected_t)


def test_amin_out_resizes_a_mismatching_out(mojo_gpu):
    x = torch.randn(2, 3, 4)
    out = torch.empty(0, device=mojo_gpu)
    torch.amin(x.to(mojo_gpu), dim=2, out=out)
    assert tuple(out.shape) == (2, 3)
    torch.testing.assert_close(out.cpu(), torch.amin(x, dim=2))


def test_amin_out_rejects_a_mismatching_dtype(mojo_gpu):
    """amin's meta func requires out.dtype == self.dtype exactly (no dtype=
    kwarg exists to cast through)."""
    x = torch.randn(4, 7)
    with pytest.raises(RuntimeError):
        torch.amin(
            x.to(mojo_gpu),
            dim=1,
            out=torch.empty(4, dtype=torch.float64, device=mojo_gpu),
        )


def test_amin_out_empty_reduce_dim_raises(mojo_gpu):
    x = torch.empty(3, 0, 5)
    out = torch.empty(0, device=mojo_gpu)
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.amin(x.to(mojo_gpu), dim=1, out=out)


def test_amax_amin_empty_reduce_dim_refused_even_with_empty_output(mojo_gpu):
    """torch refuses a zero-length reduce dim EVEN WHEN the output itself is
    empty too: amin(empty(0, 0), dim=1) still raises "Expected reduction dim
    1 to have non-zero size", it does not just return an empty tensor."""
    x = torch.empty(0, 0)
    xd = x.to(mojo_gpu)
    for fn in (torch.amax, torch.amin):
        with pytest.raises((RuntimeError, NotImplementedError)):
            fn(xd, dim=1)
        with pytest.raises((RuntimeError, NotImplementedError)):
            fn(xd, dim=1, out=torch.empty(0, device=mojo_gpu))

    # sanity pair: reducing the EMPTY dim is refused regardless of which side
    # it's on ((0, 3) dim=0, (3, 0) dim=1); reducing the NON-EMPTY dim while
    # the other one happens to be empty is not an error (output is empty too).
    a = torch.empty(0, 3)
    b = torch.empty(3, 0)
    for fn in (torch.amax, torch.amin):
        with pytest.raises((RuntimeError, NotImplementedError)):
            fn(a.to(mojo_gpu), dim=0)
        with pytest.raises((RuntimeError, NotImplementedError)):
            fn(b.to(mojo_gpu), dim=1)
        torch.testing.assert_close(fn(a.to(mojo_gpu), dim=1).cpu(), fn(a, dim=1))
        torch.testing.assert_close(fn(b.to(mojo_gpu), dim=0).cpu(), fn(b, dim=0))


def test_max_and_min_full_reduction(mojo_device):
    x = torch.randn(37, 41)
    xd = x.to(mojo_device)
    torch.testing.assert_close(torch.max(xd).cpu(), torch.max(x))
    torch.testing.assert_close(torch.min(xd).cpu(), torch.min(x))
    ints = torch.randint(-100, 100, (5, 9), dtype=torch.int64)
    torch.testing.assert_close(torch.max(ints.to(mojo_device)).cpu(), torch.max(ints))


@pytest.mark.parametrize("shape", [(0,), (3, 0), (0, 3)])
def test_max_and_min_full_reduction_of_empty_refused(mojo_device, shape):
    """The extent==0 refusal reused by full max()/min() (every dim is
    reduced, so any zero dim makes the whole reduce extent 0) must stay
    unaffected by widening amax/amin's out= refusal to empty outputs too."""
    x = torch.empty(shape).to(mojo_device)
    for fn in (torch.max, torch.min):
        with pytest.raises((RuntimeError, NotImplementedError)):
            fn(x)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int64,
        torch.int32,
    ],
)
@pytest.mark.parametrize("shape", [(37, 41), (357, 789), (128,)])
def test_max_unary_out(mojo_device, shape, dtype):
    """`max.unary_out`: the full-reduction `out=` overload (`aten::max`
    itself has no `out=` form; torch routes `torch.max(x, out=t)` here)."""
    if dtype is torch.float64:
        skip_if_metal(mojo_device, "no float64 on Apple GPUs")
    if dtype.is_floating_point:
        x = torch.randn(shape).to(dtype)
    else:
        x = torch.randint(-100, 100, shape, dtype=dtype)
    xd = x.to(mojo_device)
    out = torch.empty((), dtype=dtype, device=mojo_device)
    returned = torch.max(xd, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), torch.max(x))


def test_max_unary_out_noncontiguous_and_wrongly_shaped_out(mojo_device):
    x = torch.randn(6, 11)
    xd = x.to(mojo_device)[:, ::2]
    assert not xd.is_contiguous()
    out = torch.empty(4, 4, device=mojo_device)  # wrong shape: resized to ()
    torch.max(xd, out=out)
    assert tuple(out.shape) == ()
    torch.testing.assert_close(out.cpu(), torch.max(x[:, ::2]))


def test_max_unary_out_propagates_nan(mojo_device):
    x = torch.tensor([1.0, float("nan"), -7.0])
    out = torch.empty((), device=mojo_device)
    torch.max(x.to(mojo_device), out=out)
    assert out.cpu().isnan().item()


def test_max_unary_out_declines_an_out_that_aliases_the_input(mojo_gpu):
    """See `_decline_aliasing_out` for why any aliasing `out=` declines."""
    x = torch.randn(6, device=mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.max(x, out=x[x.numel() :])
    with pytest.raises(NotImplementedError):
        torch.max(x, out=x)


def test_max_unary_out_requires_an_exact_dtype_match(mojo_gpu):
    """Unlike mean.out/any.out's safe_cast, max_all_kernel_impl's
    make_reduction on stock CUDA refuses ANY dtype mismatch, even a safe
    upcast (verified against stock CUDA torch: int64 -> float32 raises
    "provided dtype must match dtype of result")."""
    x = torch.randint(-100, 100, (9, 5), dtype=torch.int64)
    with pytest.raises(RuntimeError):
        torch.max(
            x.to(mojo_gpu), out=torch.empty((), dtype=torch.float32, device=mojo_gpu)
        )

    y = torch.randn(9, 5)
    with pytest.raises(RuntimeError):
        torch.max(
            y.to(mojo_gpu), out=torch.empty((), dtype=torch.int64, device=mojo_gpu)
        )

    out = torch.empty((), dtype=torch.int64, device=mojo_gpu)
    torch.max(x.to(mojo_gpu), out=out)
    torch.testing.assert_close(out.cpu(), torch.max(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_max_unary_out_refuses_empty_input(mojo_device, dtype):
    """Stock CUDA errors on this too (an internal assert in Reduce.cuh,
    verified on an H100), just not cleanly; declining is the closest match."""
    x = torch.empty((0, 5), dtype=dtype)
    out = torch.empty((), dtype=dtype, device=mojo_device)
    with pytest.raises(RuntimeError):
        torch.max(x.to(mojo_device), out=out)


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16, torch.int64]
)
def test_min_unary_out(mojo_gpu, dtype):
    """`torch.min(x, out=out)` dispatches to min.unary_out."""
    if dtype is torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    if dtype.is_floating_point:
        x = torch.randn(37, 41).to(dtype)
    else:
        x = torch.randint(-100, 100, (37, 41), dtype=dtype)
    xd = x.to(mojo_gpu)
    expected = torch.min(x)

    out = torch.empty((), dtype=dtype, device=mojo_gpu)
    returned = torch.min(xd, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected)

    # non-contiguous input
    strided = xd.t()
    out2 = torch.empty((), dtype=dtype, device=mojo_gpu)
    torch.min(strided, out=out2)
    torch.testing.assert_close(out2.cpu(), torch.min(x.t()))


def test_min_unary_out_propagates_nan(mojo_gpu):
    x = torch.tensor([1.0, float("nan"), -7.0, 3.0])
    out = torch.empty((), device=mojo_gpu)
    torch.min(x.to(mojo_gpu), out=out)
    assert out.cpu().isnan().item()


def test_min_unary_out_resizes_a_mismatching_out(mojo_gpu):
    """`resize_output` first: a non-scalar `out` is resized to `()`."""
    x = torch.randn(37, 41)
    out = torch.empty(5, 3, device=mojo_gpu)
    returned = torch.min(x.to(mojo_gpu), out=out)
    assert tuple(returned.shape) == ()
    torch.testing.assert_close(returned.cpu(), torch.min(x))


def test_min_unary_out_requires_an_exact_dtype_match(mojo_gpu):
    """Unlike mean.out/any.out's safe_cast, min_all_kernel_impl's
    make_reduction on stock CUDA refuses ANY dtype mismatch, even a safe
    upcast (verified against stock CUDA torch: int64 -> float32 raises
    "provided dtype must match dtype of result")."""
    x = torch.randint(-100, 100, (9, 5), dtype=torch.int64)
    with pytest.raises(RuntimeError):
        torch.min(
            x.to(mojo_gpu), out=torch.empty((), dtype=torch.float32, device=mojo_gpu)
        )

    y = torch.randn(9, 5)
    with pytest.raises(RuntimeError):
        torch.min(
            y.to(mojo_gpu), out=torch.empty((), dtype=torch.int64, device=mojo_gpu)
        )

    out = torch.empty((), dtype=torch.int64, device=mojo_gpu)
    torch.min(x.to(mojo_gpu), out=out)
    torch.testing.assert_close(out.cpu(), torch.min(x))


@pytest.mark.parametrize("shape", [(0,), (3, 0), (0, 3)])
def test_min_unary_out_empty_input_errors(mojo_gpu, shape):
    """min.unary_out rides `_full_extremum_dims`, so it inherits the same
    extent==0 refusal as full max()/min() (a zero-length reduce dim always
    refuses here, since a full reduction's output is never itself empty)."""
    out = torch.empty((), device=mojo_gpu)
    with pytest.raises((RuntimeError, NotImplementedError)):
        torch.min(torch.empty(shape, device=mojo_gpu), out=out)


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
    through the cast kernel rather than refused. Deterministic rows (an
    all-False row included) so both outcomes are exercised, unlike random
    zero-containing data which is almost always any=True per row."""
    mask = torch.tensor(
        [
            [False, False, False, False, False, False, False],
            [False, False, True, False, False, False, False],
            [True, True, True, True, True, True, True],
            [False, False, False, False, False, False, False],
        ]
    )
    expected = torch.any(mask, dim=1)
    assert expected.tolist() == [False, True, True, False]
    out = torch.empty(4, dtype=torch.uint8, device=mojo_gpu)
    torch.any(mask.to(mojo_gpu), dim=1, out=out)
    torch.testing.assert_close(out.cpu(), expected.to(torch.uint8))
    with pytest.raises(RuntimeError):
        torch.any(
            mask.to(mojo_gpu),
            dim=1,
            out=torch.empty(4, dtype=torch.float32, device=mojo_gpu),
        )


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.uint8, torch.int32, torch.float32, torch.float64]
)
def test_all_out_variants(mojo_gpu, keepdim, dtype):
    """all.out (int dim), all.dims_out (dim list) and all.all_out (no dim,
    full reduction) all round-trip through the out= tensor, reusing the
    same AllOp path as the non-out overloads. Random zero-containing data
    almost never has an all-True row, so the rows below are a deterministic
    mix of all-nonzero, all-zero and one-zero; the full/dims_out cases (one
    scalar each, so they can't show both outcomes in a single call) are
    checked against this mixed tensor (a real, non-vacuous False) and a
    second all-nonzero one (True)."""
    if dtype is torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    mixed = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 0, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
        ]
    ).to(dtype)
    all_true = torch.full_like(mixed, 1)
    xd = mixed.to(mojo_gpu)

    # `out=` fixes the result dtype explicitly (here bool), which the
    # bool-or-uint8 policy accepts regardless of the input's dtype -- unlike
    # the no-out overloads, which follow the uint8-input-keeps-uint8 rule.
    expected = torch.all(mixed, dim=1, keepdim=keepdim).bool()
    assert expected.reshape(-1).tolist() == [True, False, False, True, False, True]
    out = torch.empty(expected.shape, dtype=torch.bool, device=mojo_gpu)
    returned = torch.all(xd, dim=1, keepdim=keepdim, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected)

    for src, want in ((mixed, False), (all_true, True)):
        srcd = src.to(mojo_gpu)
        expected_dims = torch.all(src, dim=(0, 1), keepdim=keepdim).bool()
        assert expected_dims.item() == want
        out_dims = torch.empty(expected_dims.shape, dtype=torch.bool, device=mojo_gpu)
        returned = torch.all(srcd, dim=(0, 1), keepdim=keepdim, out=out_dims)
        assert returned.data_ptr() == out_dims.data_ptr()
        torch.testing.assert_close(out_dims.cpu(), expected_dims)

        expected_full = torch.all(src).bool()
        assert expected_full.item() == want
        out_full = torch.empty((), dtype=torch.bool, device=mojo_gpu)
        returned = torch.all(srcd, out=out_full)
        assert returned.data_ptr() == out_full.data_ptr()
        torch.testing.assert_close(out_full.cpu(), expected_full)


def test_all_out_accepts_a_uint8_destination(mojo_gpu):
    """all's out dtype policy is bool-or-uint8, mirroring any.out.
    Deterministic rows (an all-True row included) so both outcomes are
    exercised, unlike random zero-containing data which is almost always
    all=False per row."""
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [True, False, True, True, True, True, True],
            [False, False, False, False, False, False, False],
            [True, True, True, True, True, True, True],
        ]
    )
    expected = torch.all(mask, dim=1)
    assert expected.tolist() == [True, False, False, True]
    out = torch.empty(4, dtype=torch.uint8, device=mojo_gpu)
    torch.all(mask.to(mojo_gpu), dim=1, out=out)
    torch.testing.assert_close(out.cpu(), expected.to(torch.uint8))
    with pytest.raises(RuntimeError):
        torch.all(
            mask.to(mojo_gpu),
            dim=1,
            out=torch.empty(4, dtype=torch.float32, device=mojo_gpu),
        )


def test_all_out_resizes_and_handles_strided_and_noncontig(mojo_gpu):
    """all.out follows the same resize/strided-copy rules as mean.out /
    any.out (`_scalar_reduction_out`), exercised here for all's overloads."""
    x = torch.randint(0, 2, (2, 3, 4), dtype=torch.bool)
    xd = x.to(mojo_gpu)

    # Mismatching shape -> resize_output.
    out = torch.empty(0, dtype=torch.bool, device=mojo_gpu)
    torch.all(xd, dim=2, out=out)
    assert tuple(out.shape) == (2, 3)
    torch.testing.assert_close(out.cpu(), x.all(dim=2))

    # Non-contiguous destination -> computed into a fresh buffer and copied.
    storage = torch.zeros(2, 3, 2, dtype=torch.bool, device=mojo_gpu)
    strided_out = storage[:, :, 0]
    assert not strided_out.is_contiguous()
    torch.all(xd, dim=2, out=strided_out)
    torch.testing.assert_close(strided_out.cpu(), x.all(dim=2))
    torch.testing.assert_close(
        storage[:, :, 1].cpu(), torch.zeros(2, 3, dtype=torch.bool)
    )

    # Non-contiguous input works too.
    x_t = x.transpose(0, 1)
    out_t = torch.empty(x_t.shape[:-1], dtype=torch.bool, device=mojo_gpu)
    torch.all(x_t.to(mojo_gpu), dim=-1, out=out_t)
    torch.testing.assert_close(out_t.cpu(), x_t.all(dim=-1))

    # all.dims_out with an empty tensor.
    empty = torch.empty(0, 3, dtype=torch.bool, device=mojo_gpu)
    out_empty = torch.empty((), dtype=torch.bool, device=mojo_gpu)
    torch.all(empty, dim=(0, 1), out=out_empty)
    torch.testing.assert_close(
        out_empty.cpu(), torch.empty(0, 3, dtype=torch.bool).all(dim=(0, 1))
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.bool, torch.uint8, torch.int32, torch.float32, torch.float16, torch.float64],
)
def test_any_all_out_full_reduction(mojo_gpu, dtype):
    """`any.all_out`: no dim/keepdim args, always reduces to a 0-d result.

    The `out=` dtype is bool here on both sides, so the input-dtype-dependent
    result-dtype rule (`aten::any` w/o `out` keeps a uint8 input as uint8)
    does not come into play; that overload is unrelated to this one.

    Random data with `> 0.3` is True with 99.9997% probability over 30
    elements, so the "True" and "all-False" cases below are deterministic
    (one hot element, and an actual non-empty all-zero tensor) rather than
    relying on chance.
    """
    if dtype is torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = torch.zeros(5, 6, dtype=dtype)
    x[2, 3] = 1
    xd = x.to(mojo_gpu)
    out = torch.empty((), dtype=torch.bool, device=mojo_gpu)
    expected = torch.empty((), dtype=torch.bool)
    torch.any(x, out=expected)
    assert expected.item() is True
    returned = torch.any(xd, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected)

    # non-contiguous input
    xt = x.t()
    torch.any(xt, out=expected)
    torch.any(xt.to(mojo_gpu), out=out)
    torch.testing.assert_close(out.cpu(), expected)

    # A real, non-empty all-zero input (not the vacuous empty-tensor
    # identity below): the sum-of-truthiness ops must find no True anywhere.
    zeros_nonempty = torch.zeros(5, 6, dtype=dtype)
    torch.any(zeros_nonempty, out=expected)
    assert expected.item() is False
    torch.any(zeros_nonempty.to(mojo_gpu), out=out)
    torch.testing.assert_close(out.cpu(), expected)

    # all-False input, including the empty-tensor case (identity = False)
    zeros = torch.zeros(4, 0, dtype=dtype)
    torch.any(zeros, out=expected)
    torch.any(zeros.to(mojo_gpu), out=out)
    torch.testing.assert_close(out.cpu(), expected)


def test_any_all_out_uint8_destination_and_resize(mojo_gpu):
    """bool-or-uint8 dtype policy and `resize_output` apply to all_out too."""
    mask = torch.zeros(4, 7, dtype=torch.bool)
    mask[1, 3] = True
    expected = torch.any(mask)
    assert expected.item() is True
    out = torch.empty(4, dtype=torch.uint8, device=mojo_gpu)  # wrong shape too
    torch.any(mask.to(mojo_gpu), out=out)
    assert tuple(out.shape) == ()
    torch.testing.assert_close(out.cpu(), expected.to(torch.uint8))
    with pytest.raises(RuntimeError):
        torch.any(
            mask.to(mojo_gpu), out=torch.empty((), dtype=torch.float32, device=mojo_gpu)
        )


def test_any_all_out_nan_is_truthy(mojo_gpu):
    x = torch.zeros(2, 100)
    x[1, 50] = float("nan")
    out = torch.empty((), dtype=torch.bool, device=mojo_gpu)
    torch.any(x.to(mojo_gpu), out=out)
    torch.testing.assert_close(out.cpu(), torch.any(x))


def test_any_all_out_rank0_input(mojo_gpu):
    """A rank-0 input reduces over zero axes -- a legitimate no-op, not the
    user-requested empty dim list `any.dims`/`all.dims` decline."""
    x = torch.tensor(True)
    out = torch.empty((), dtype=torch.bool, device=mojo_gpu)
    torch.any(x.to(mojo_gpu), out=out)
    torch.testing.assert_close(out.cpu(), torch.any(x))
    torch.all(x.to(mojo_gpu), out=out)
    torch.testing.assert_close(out.cpu(), torch.all(x))


@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dims", [None, [0], [1], [0, 2], [-1, 0]])
def test_any_dims_out(mojo_gpu, dims, keepdim):
    """`any.dims_out`: optional int[] dim (None = full reduce), like
    `any.dims` but writing into `out`. A sparse deterministic hot-cell
    pattern (6 True cells out of 60) rather than random data, so every dims
    combination's output mixes True and False instead of "almost always
    True" (`any` over ~15-20 random booleans per fiber is True >99.99% of
    the time)."""
    x = torch.arange(60).reshape(3, 4, 5) % 11 == 0
    xd = x.to(mojo_gpu)
    expected = torch.any(x, dim=dims, keepdim=keepdim)
    out = torch.empty(expected.shape, dtype=torch.bool, device=mojo_gpu)
    returned = torch.ops.aten.any.dims_out(xd, dims, keepdim, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected)

    # non-contiguous input
    xt = x.transpose(0, 1)
    expected_t = torch.any(xt, dim=dims, keepdim=keepdim)
    out_t = torch.empty(expected_t.shape, dtype=torch.bool, device=mojo_gpu)
    torch.ops.aten.any.dims_out(xt.to(mojo_gpu), dims, keepdim, out=out_t)
    torch.testing.assert_close(out_t.cpu(), expected_t)


def test_any_dims_out_uint8_destination_and_resize(mojo_gpu):
    mask = torch.tensor(
        [
            [False, False, False, False, False, False, False],
            [False, True, False, False, False, False, False],
            [False, False, False, False, False, False, False],
            [False, False, False, False, False, False, True],
        ]
    )
    expected = torch.any(mask, dim=[1])
    assert expected.tolist() == [False, True, False, True]
    out = torch.empty(0, dtype=torch.uint8, device=mojo_gpu)  # wrong shape
    torch.ops.aten.any.dims_out(mask.to(mojo_gpu), [1], False, out=out)
    assert tuple(out.shape) == expected.shape
    torch.testing.assert_close(out.cpu(), expected.to(torch.uint8))


def test_any_dims_out_empty_dim_list_declines(mojo_gpu):
    """An explicit empty dim list is declined, matching `any.dims`
    (`test_unsupported_inputs_raise_not_implemented`)."""
    mask = torch.randint(0, 2, (3, 4), dtype=torch.bool)
    out = torch.empty(3, 4, dtype=torch.bool, device=mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.ops.aten.any.dims_out(mask.to(mojo_gpu), [], False, out=out)


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


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16]
)
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
    if dtype is torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
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


@pytest.mark.parametrize("dim", [None, 0, -1, []])
def test_vector_norm_ord0_rank0(mojo_gpu, dim):
    """ord=0 (`NormL0Spec`, count of nonzero) skips the size-one `abs`
    shortcut entirely, so this exercises the general reduce path at rank 0
    rather than the abs fast path the other ords take."""
    x = torch.tensor(-3.5)
    xd = x.to(mojo_gpu)
    expected = torch.linalg.vector_norm(x, ord=0, dim=dim)
    got = torch.linalg.vector_norm(xd, ord=0, dim=dim)
    assert got.shape == expected.shape == ()
    torch.testing.assert_close(got.cpu(), expected)


def test_vector_norm_ord0_size_one_reduce_is_not_abs(mojo_gpu):
    """The `_all_reduced_dims_size_one` abs shortcut is only valid for
    ord != 0 (LinearAlgebra.cpp special-cases ord=0 to `ne(0)` there, not
    `abs()`); a lone nonzero element must count as 1, not its magnitude."""
    x = torch.tensor([[5.0], [0.0], [-3.0]])
    expected = torch.linalg.vector_norm(x, ord=0, dim=1)
    got = torch.linalg.vector_norm(x.to(mojo_gpu), ord=0, dim=1)
    torch.testing.assert_close(got.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vector_norm_l1_with_an_accumulation_dtype(mojo_gpu, dtype):
    """Same `dtype=` cast-before-accumulate contract as ord=2."""
    cpu = torch.randn(4096, dtype=dtype)
    expected = torch.linalg.vector_norm(cpu, 1, dtype=torch.float32)
    got = torch.linalg.vector_norm(cpu.to(mojo_gpu), 1, dtype=torch.float32)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-5, atol=1e-4)


def test_vector_norm_l1_out_resizes_and_strided_input(mojo_gpu):
    contiguous = torch.linspace(-3.0, 4.0, 35).reshape(5, 7)
    strided = contiguous.t()
    expected = torch.linalg.vector_norm(strided, ord=1)

    device_strided = contiguous.to(mojo_gpu).t()
    out = torch.empty(0, dtype=torch.float32, device=mojo_gpu)
    returned = torch.linalg.vector_norm(device_strided, ord=1, out=out)
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected)

    empty = torch.empty((0, 7), dtype=torch.float32).to(mojo_gpu)
    torch.testing.assert_close(
        torch.linalg.vector_norm(empty, ord=1).cpu(), torch.tensor(0.0), rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_vector_norm_inf_matches_torch(mojo_gpu, dtype):
    """ord=+inf: max of |x|, distinct from ord=2's sum of squares. This
    selects one already-rounded element rather than accumulating, so it is
    exactly representable -- exact equality catches a selection error (wrong
    element/index) that a loose tolerance would hide."""
    x = (torch.randn(4099, 37) * 10).to(dtype)
    expected = torch.linalg.vector_norm(x.double(), ord=math.inf, dim=1)
    ours = torch.linalg.vector_norm(x.to(mojo_gpu), ord=math.inf, dim=1).cpu()
    torch.testing.assert_close(ours.double(), expected, atol=0, rtol=0)


def test_vector_norm_inf_with_an_accumulation_dtype(mojo_gpu):
    cpu = torch.randn(4096, dtype=torch.bfloat16)
    expected = torch.linalg.vector_norm(cpu, ord=math.inf, dtype=torch.float32)
    got = torch.linalg.vector_norm(cpu.to(mojo_gpu), ord=math.inf, dtype=torch.float32)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.cpu(), expected)


def test_vector_norm_declines_non_floating_input(mojo_gpu):
    """`TORCH_META_FUNC(linalg_vector_norm)` calls `checkFloatingOrComplex` on
    the INPUT's own dtype unconditionally, before ever looking at `dtype=`:
    an integer/bool input is rejected even though `dtype=float32` would make
    the cast well-defined."""
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(torch.randint(0, 9, (4, 5)).to(mojo_gpu), dim=1)
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(
            torch.randint(0, 9, (4, 5)).to(mojo_gpu), dim=1, dtype=torch.float32
        )
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(
            (torch.randn(4, 5) > 0).to(mojo_gpu), dim=1, dtype=torch.float32
        )


@pytest.mark.parametrize(
    "src_dtype,target_dtype",
    [
        (torch.float32, torch.float16),
        (torch.float32, torch.bfloat16),
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
    ],
)
def test_vector_norm_declines_narrowing_dtype(mojo_gpu, src_dtype, target_dtype):
    """`check_linalg_norm_dtype`'s `promoteTypes(self_dtype, dtype) == dtype`:
    among these three floats, only same-dtype or a target of float32 widens;
    every other pair narrows and torch rejects it."""
    x = torch.randn(4, 5, dtype=src_dtype)
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(x.to(mojo_gpu), dim=1, dtype=target_dtype)


def test_vector_norm_out_declines_non_floating_input(mojo_gpu):
    out = torch.empty(4, device=mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(
            torch.randint(0, 9, (4, 5)).to(mojo_gpu), dim=1, out=out
        )


@pytest.mark.parametrize(
    "shape,dim,keepdim",
    [((1,), 0, False), ((1, 1), None, False), ((5, 1), 1, False), ((5, 1), 1, True)],
)
def test_vector_norm_reduce_over_size_one_dims_uses_abs(mojo_gpu, shape, dim, keepdim):
    """torch's `is_reduce_over_1D_vector`: every REDUCED dim has extent 1
    (a kept dim may be any size), so the reduction is exactly `abs()`, not
    square-then-sqrt -- squaring 1e20 overflows float32 to inf even though
    abs(1e20) is exact (`linalg_vector_norm_out` in ATen's LinearAlgebra.cpp)."""
    x = torch.tensor([1e20, -1e20, 3.0, -1.0, 0.0][: shape[0]], dtype=torch.float32)
    x = x.reshape(shape)
    expected = torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)
    assert torch.isfinite(
        expected
    ).all()  # sanity: the reference itself must not be inf
    got = torch.linalg.vector_norm(x.to(mojo_gpu), dim=dim, keepdim=keepdim)
    torch.testing.assert_close(got.cpu(), expected)


def test_norm_reduce_over_size_one_dims_uses_abs(mojo_gpu):
    """Same fix, reached through the legacy `norm.ScalarOpt_dim` overload."""
    x = torch.tensor([[1e20], [-2.0], [3.0]], dtype=torch.float32)
    expected = torch.linalg.vector_norm(x, dim=1)
    got = torch.ops.aten.norm.ScalarOpt_dim(x.to(mojo_gpu), 2, [1], False)
    torch.testing.assert_close(got.cpu(), expected)


def test_vector_norm_size_one_reduce_out_success(mojo_gpu):
    """`_vector_norm_abs_out`'s success path (the decline tests below only
    cover rejection): a wrongly-shaped `out=` is resized, and a
    non-contiguous `out=` is computed into a fresh buffer and copied across,
    same as `_scalar_reduction_out`."""
    x = torch.tensor([[1e20], [-2.0], [3.0]], dtype=torch.float32)
    expected = torch.linalg.vector_norm(x, dim=1)
    xd = x.to(mojo_gpu)

    # Mismatching shape -> resize_out.
    out = torch.empty(0, dtype=torch.float32, device=mojo_gpu)
    returned = torch.linalg.vector_norm(xd, dim=1, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert tuple(out.shape) == (3,)
    torch.testing.assert_close(out.cpu(), expected)

    # Non-contiguous destination -> computed into a fresh buffer and copied.
    storage = torch.zeros(3, 2, device=mojo_gpu)
    strided_out = storage[:, 0]
    assert not strided_out.is_contiguous()
    torch.linalg.vector_norm(xd, dim=1, out=strided_out)
    torch.testing.assert_close(strided_out.cpu(), expected)
    torch.testing.assert_close(storage[:, 1].cpu(), torch.zeros(3))


def test_vector_norm_size_one_reduce_out_declines_aliasing_input(mojo_gpu):
    """Any `out=` sharing storage with the input is declined outright, before
    any resize is even considered -- not just when the current byte ranges
    overlap. `resize_out` can reallocate a shared allocation to grow `dst`,
    which would silently move the bytes `a`'s already-cached tensor info
    still points at (see `_decline_aliasing_out`'s docstring); declining
    every case uniformly avoids having to reason about which ones happen to
    be safe. Covers: a genuinely shifted overlap, a same-tensor "in-place via
    out=" call (`abs(x, out=x)` is fine on real torch, but this backend
    declines it too rather than special-case it), and a same-storage `out=`
    that merely reshapes the input (`x.squeeze(1)`, covering the exact same
    bytes as `x` at a different rank -- no resize needed, but still declined,
    since it still shares storage)."""
    b = torch.arange(6, dtype=torch.float32).to(mojo_gpu)
    inp = b[:-1].view(-1, 1)  # every reduced dim (dim=1) has extent 1
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(inp, dim=1, out=b[1:])

    x = torch.tensor([[1e20], [-2.0], [3.0]], dtype=torch.float32).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(x, dim=1, keepdim=True, out=x)

    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(x, dim=1, out=x.squeeze(1))


def test_vector_norm_size_one_reduce_out_declines_wrong_dtype(mojo_gpu):
    """Same `exact` out-dtype policy as the accumulator path (`_check_out_dtype`):
    confirmed on stock CUDA torch that an integer `out=` for a float result is
    rejected, not silently cast."""
    x = torch.tensor([[1e20]], dtype=torch.float32).to(mojo_gpu)
    out = torch.empty(1, dtype=torch.int64, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="can't be cast"):
        torch.linalg.vector_norm(x, dim=1, out=out)


def test_vector_norm_size_one_reduce_out_declines_aliasing_input_needing_resize(
    mojo_gpu,
):
    """The hazard `_decline_aliasing_out` exists for: a same-storage `out=`
    that does NOT currently overlap the input's bytes at all, but whose
    resize (growing it in place) would reallocate the storage `a` also reads
    through -- caught by the storage-identity check before resize is even
    attempted, not by comparing byte ranges (which would find no overlap
    here, before OR after resizing this particular pair of slices)."""
    base = torch.arange(4, dtype=torch.float32).to(mojo_gpu)
    inp = base[:3].view(3, 1)  # every reduced dim (dim=1) has extent 1
    out = base[3:]  # 1 element; the result needs 3 -- same storage as inp
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(inp, dim=1, out=out)


def test_vector_norm_size_one_reduce_out_declines_internal_overlap(mojo_gpu):
    """An `out=` with more than one logical element sharing one physical
    address (`.expand()`) must be declined, not silently collapse every
    reduced row into whichever write happens to land last."""
    x = torch.tensor([[1e20], [2.0], [3.0]], dtype=torch.float32).to(mojo_gpu)
    out = torch.empty(1, device=mojo_gpu).expand(3)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.linalg.vector_norm(x, dim=1, out=out)


def test_reduction_out_declines_internal_overlap(mojo_gpu):
    """Same check, in the general (non size-one-reduce) `_scalar_reduction_out`
    path every out= reduction shares -- stock CUDA torch actually tolerates
    this for `sum.out` (every aliased position ends up holding whichever
    result happened to be written last, which is never what the caller
    wanted), so this backend is intentionally stricter here."""
    x = torch.randn(4, 5).to(mojo_gpu)
    out = torch.empty(1, device=mojo_gpu).expand(4)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.sum(x, dim=1, out=out)


def test_reduction_out_internal_overlap_is_fine_when_empty(mojo_gpu):
    """An EMPTY out= is exempt regardless of its strides: nothing is written,
    so a degenerate `.expand()` over a zero extent can't collapse anything.
    Confirmed accepted on stock CUDA torch."""
    x = torch.randn(0, 3, 1).to(mojo_gpu)
    out = torch.empty(0, 1, device=mojo_gpu).expand(0, 3)
    result = torch.sum(x, dim=2, out=out)
    assert result.shape == (0, 3)


# ---------------------------------------------------------------------------
# norm (legacy overloads): all route through the same ord-2 path as
# linalg_vector_norm above, so these tests only need to check the schema
# plumbing (p=None/2, dim=[]/None/single/multi, dtype=, out=), not the math.
#
# These call `torch.ops.aten.norm.*` directly rather than the public
# `torch.norm`/`Tensor.norm`: on a strided PrivateUse1 tensor (ours),
# `torch/functional.py`'s `norm()` always redirects to
# `torch.linalg.vector_norm`/`matrix_norm`/`_VF.nuclear_norm` before the
# dispatcher is ever reached (confirmed with the boxed-kernel counters --
# `torch.norm(x.to(mojo_gpu), p=2, dim=1)` increments
# `aten::linalg_vector_norm`, never any `aten::norm.*`), so no public API
# reaches these overloads on this device.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p", [None, 2, 2.0])
def test_norm_scalar_defaults_to_p2_over_all_dims(mojo_gpu, p):
    x = torch.randn(4, 5)
    expected = torch.linalg.vector_norm(x)
    if p is None:
        got = torch.ops.aten.norm.Scalar(x.to(mojo_gpu))
    else:
        got = torch.ops.aten.norm.Scalar(x.to(mojo_gpu), p)
    torch.testing.assert_close(got.cpu(), expected)


def test_norm_scalaropt_dtype(mojo_gpu):
    x = torch.randn(4, 5, dtype=torch.bfloat16)
    expected = torch.linalg.vector_norm(x, dtype=torch.float32)
    got = torch.ops.aten.norm.ScalarOpt_dtype(x.to(mojo_gpu), None, dtype=torch.float32)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.cpu(), expected)


def test_vector_norm_inf_out_and_resize(mojo_gpu):
    """A wrongly-shaped `out` is resized, same as every other scalar
    reduction's out= path (`_scalar_reduction_out`)."""
    x = torch.randn(5, 7)
    expected = torch.linalg.vector_norm(x, ord=math.inf, dim=1)
    out = torch.empty(0, device=mojo_gpu)
    returned = torch.linalg.vector_norm(x.to(mojo_gpu), ord=math.inf, dim=1, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert tuple(out.shape) == (5,)
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize("dim", [0, 1])
def test_vector_norm_inf_refuses_empty_reduce_dim_with_empty_output(mojo_gpu, dim):
    """torch's meta check refuses ord=+inf over a zero-length reduce dim EVEN
    WHEN THE OUTPUT ITSELF IS EMPTY TOO (confirmed on live CPU torch:
    `vector_norm(empty(0, 0), ord=inf, dim=1)` still raises) -- same as
    amax/amin's `errors_on_empty_axis`, whose skeleton-generic guard
    (`_rowred_spec_into_go`) is unconditional on `reduce_n == 0`, not gated
    by the output count."""
    x = torch.empty(0, 0)
    with pytest.raises(RuntimeError):
        torch.linalg.vector_norm(x, ord=math.inf, dim=dim)
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(x.to(mojo_gpu), ord=math.inf, dim=dim)


def test_vector_norm_inf_declines_non_floating_input_with_dtype(mojo_gpu):
    """Same rule as ord=2 (`checkFloatingOrComplex` on the input's own dtype,
    unconditionally, before `dtype=` is ever consulted): int/bool inputs are
    declined for ord=+inf too, even with a `dtype=` that would make the cast
    well-defined."""
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(
            torch.randint(0, 9, (4, 5)).to(mojo_gpu), ord=math.inf, dim=1
        )
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(
            torch.randint(0, 9, (4, 5)).to(mojo_gpu),
            ord=math.inf,
            dim=1,
            dtype=torch.float32,
        )
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(
            (torch.randn(4, 5) > 0).to(mojo_gpu),
            ord=math.inf,
            dim=1,
            dtype=torch.float32,
        )


@pytest.mark.parametrize("dim", [None, 1, [0, 1], []])
@pytest.mark.parametrize("keepdim", [False, True])
def test_norm_scalaropt_dim(mojo_gpu, dim, keepdim):
    """An explicit empty dim list means "reduce every dim", same as dim=None,
    unlike any.dims/all.dims."""
    x = torch.randn(4, 5)
    dim_arg = [] if dim is None else ([dim] if isinstance(dim, int) else dim)
    expected = torch.linalg.vector_norm(
        x, dim=(None if dim is None else dim), keepdim=keepdim
    )
    got = torch.ops.aten.norm.ScalarOpt_dim(x.to(mojo_gpu), 2, dim_arg, keepdim)
    torch.testing.assert_close(got.cpu(), expected)


def test_norm_scalaropt_dim_dtype(mojo_gpu):
    x = torch.randn(4, 5, dtype=torch.bfloat16)
    expected = torch.linalg.vector_norm(x, dim=1, dtype=torch.float32)
    got = torch.ops.aten.norm.ScalarOpt_dim_dtype(
        x.to(mojo_gpu), None, [1], False, dtype=torch.float32
    )
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.cpu(), expected)


def test_norm_out_resizes_a_wrongly_shaped_out(mojo_gpu):
    x = torch.randn(4, 5)
    expected = torch.linalg.vector_norm(x, dim=1)
    out = torch.empty(1, dtype=torch.float32, device=mojo_gpu)  # wrong shape
    returned = torch.ops.aten.norm.out(x.to(mojo_gpu), 2, [1], False, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert out.shape == (4,)
    torch.testing.assert_close(out.cpu(), expected)


def test_norm_dtype_out(mojo_gpu):
    """Exercises the accumulation-dtype plumbing with dtype=float32."""
    x = torch.randn(4, 5, dtype=torch.bfloat16)
    expected = torch.linalg.vector_norm(x, dim=[0, 1], dtype=torch.float32)
    out = torch.empty((), dtype=torch.float32, device=mojo_gpu)
    returned = torch.ops.aten.norm.dtype_out(
        x.to(mojo_gpu), None, [0, 1], False, dtype=torch.float32, out=out
    )
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-5, atol=1e-4)


def test_norm_dtype_out_float64(mojo_gpu):
    """`dtype=torch.float64` widens from any of the three floats (verified on
    real CUDA); norm.dtype_out/linalg_vector_norm.out both accumulate the sum
    of squares in double, not float32-then-cast."""
    skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = torch.randn(4, 5, dtype=torch.bfloat16)
    expected = torch.linalg.vector_norm(x, dim=[0, 1], dtype=torch.float64)
    out = torch.empty((), dtype=torch.float64, device=mojo_gpu)
    returned = torch.ops.aten.norm.dtype_out(
        x.to(mojo_gpu), None, [0, 1], False, dtype=torch.float64, out=out
    )
    assert returned.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)

    # a float64 self needs no dtype= at all, and rejects narrowing back down.
    # Values close to 1.0 with a 2**-30 perturbation: below float32's own
    # precision there, so a kernel that accumulated in float32 -- rather
    # than genuinely in double -- would silently lose them and answer
    # measurably differently (verified: the two differ starting a few ulps
    # in), unlike bfloat16-sourced values above, which carry too little
    # precision to tell float32 and float64 accumulation apart at all.
    torch.manual_seed(0)
    x64 = 1.0 + (torch.rand(5000, dtype=torch.float64) * 2 - 1) * 2**-30
    expected64 = torch.linalg.vector_norm(x64, dim=0)
    assert (
        expected64.item()
        != torch.linalg.vector_norm(x64.float(), dim=0).double().item()
    )
    out2 = torch.empty((), dtype=torch.float64, device=mojo_gpu)
    torch.linalg.vector_norm(x64.to(mojo_gpu), dim=0, out=out2)
    torch.testing.assert_close(out2.cpu(), expected64, rtol=0, atol=0)
    with pytest.raises(RuntimeError):
        torch.linalg.vector_norm(x64.to(mojo_gpu), dim=0, dtype=torch.float32)


def test_norm_general_p_declines_float64(mojo_gpu):
    """NormPOp (any ord outside {0, 1, 2, +-inf}) keeps a plain float32
    accumulator, unlike the dedicated-ord accumulators, which all use
    `_float_acc` and so admit float64: a float64 self, or `dtype=
    torch.float64` on another float input, would silently compute in less
    precision than asked for (verified on real CUDA that this measurably
    differs), so both are declined -- on every device, not just Apple's."""
    x64 = torch.randn(4, 5, dtype=torch.float64).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(x64, ord=3, dim=1)
    x32 = torch.randn(4, 5, dtype=torch.float32).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(x32, ord=3, dim=1, dtype=torch.float64)


def test_norm_general_p(mojo_gpu):
    x = torch.randn(4, 5)
    d = x.to(mojo_gpu)
    torch.testing.assert_close(
        torch.ops.aten.norm.Scalar(d, 3).cpu(), torch.linalg.vector_norm(x, 3)
    )
    torch.testing.assert_close(
        torch.ops.aten.norm.ScalarOpt_dim(d, -1.5, [1], True).cpu(),
        torch.linalg.vector_norm(x, -1.5, dim=1, keepdim=True),
    )
    xb = x.to(torch.bfloat16)
    got = torch.ops.aten.norm.ScalarOpt_dtype(xb.to(mojo_gpu), 0.5, dtype=torch.float32)
    assert got.dtype == torch.float32
    torch.testing.assert_close(
        got.cpu(), torch.linalg.vector_norm(xb, 0.5, dtype=torch.float32)
    )
    got = torch.ops.aten.norm.ScalarOpt_dim_dtype(
        xb.to(mojo_gpu), 4, [0], False, dtype=torch.float32
    )
    torch.testing.assert_close(
        got.cpu(), torch.linalg.vector_norm(xb, 4, dim=0, dtype=torch.float32)
    )
    out = torch.empty(1, device=mojo_gpu)
    returned = torch.ops.aten.norm.out(d, 1.5, [1], False, out=out)
    assert returned.data_ptr() == out.data_ptr() and out.shape == (4,)
    torch.testing.assert_close(out.cpu(), torch.linalg.vector_norm(x, 1.5, dim=1))
    out = torch.empty(3, dtype=torch.float32, device=mojo_gpu)
    torch.ops.aten.norm.dtype_out(
        xb.to(mojo_gpu), -2, [0, 1], False, dtype=torch.float32, out=out
    )
    assert out.shape == ()
    torch.testing.assert_close(
        out.cpu(), torch.linalg.vector_norm(xb, -2, dtype=torch.float32)
    )


# ---------------------------------------------------------------------------
# linalg_vector_norm(ord=p) for any other p: NormPOp
# ---------------------------------------------------------------------------

_GENERAL_P = [3, 0.5, 1.5, -1, -2, 4]


@pytest.mark.parametrize("p", _GENERAL_P)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dim,keepdim", [(None, False), (1, False), (0, True), ((0, -1), False), (-1, True)]
)
def test_vector_norm_general_p_matches_torch(mojo_gpu, p, dtype, dim, keepdim):
    x = (torch.randn(6, 357, 79) * 2).to(dtype)
    ours = torch.linalg.vector_norm(x.to(mojo_gpu), ord=p, dim=dim, keepdim=keepdim)
    assert ours.dtype == dtype
    if dtype is torch.float32:
        # fp64 reference: CPU and CUDA float32 already differ by ~3e-5 here
        # (p=-2, where the elements nearest 0 dominate the sum).
        expected = torch.linalg.vector_norm(x.double(), ord=p, dim=dim, keepdim=keepdim)
        torch.testing.assert_close(ours.cpu().double(), expected, rtol=1e-4, atol=0)
    else:
        # Same-dtype reference: the half types overflow and go subnormal here.
        expected = torch.linalg.vector_norm(x, ord=p, dim=dim, keepdim=keepdim)
        torch.testing.assert_close(ours.cpu(), expected, rtol=1e-2, atol=0)


@pytest.mark.parametrize("p", _GENERAL_P)
def test_vector_norm_general_p_noncontiguous_and_split(mojo_gpu, p):
    base = torch.rand(789, 357) + 0.01
    ours = torch.linalg.vector_norm(base.to(mojo_gpu).t(), ord=p, dim=1).cpu()
    torch.testing.assert_close(
        ours, torch.linalg.vector_norm(base.t(), ord=p, dim=1), rtol=1e-5, atol=0
    )
    big = torch.rand(1 << 20) + 0.01  # one output: the split + merge path
    ours = torch.linalg.vector_norm(big.to(mojo_gpu), ord=p).cpu()
    expected = torch.linalg.vector_norm(big.double(), ord=p).float()
    torch.testing.assert_close(ours, expected, rtol=1e-4, atol=0)


@pytest.mark.parametrize("p", [*_GENERAL_P, float("nan")])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_vector_norm_general_p_nonfinite_and_zeros(mojo_gpu, p, dtype):
    """|0|^p = inf for p < 0 (so the norm is 0), inf and NaN go through pow."""
    nan, inf = float("nan"), float("inf")
    x = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [nan, 2.0, 3.0, 4.0],
            [1.0, inf, 3.0, 4.0],
            [1.0, 2.0, -inf, 4.0],
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
            [inf, inf, inf, inf],
            [-1.0, 1.0, -1.0, 1.0],
        ],
        dtype=dtype,
    )
    for dim in (0, 1):
        ours = torch.linalg.vector_norm(x.to(mojo_gpu), ord=p, dim=dim).cpu()
        expected = torch.linalg.vector_norm(x, ord=p, dim=dim)
        torch.testing.assert_close(ours, expected, equal_nan=True, rtol=1e-5, atol=0)


@pytest.mark.parametrize("p", [*_GENERAL_P, float("nan")])
def test_vector_norm_general_p_empty(mojo_gpu, p):
    """torch refuses p < 0 over an empty reduce dim (no identity); otherwise
    an empty input gives 0, even for p = NaN."""
    for shape, dim in (((3, 0), 1), ((3, 0), None), ((0, 5), 1), ((3, 0), 0)):
        x = torch.empty(shape)
        try:
            expected = torch.linalg.vector_norm(x, ord=p, dim=dim)
        except RuntimeError:
            assert p < 0
            with pytest.raises(NotImplementedError):
                torch.linalg.vector_norm(x.to(mojo_gpu), ord=p, dim=dim)
            continue
        ours = torch.linalg.vector_norm(x.to(mojo_gpu), ord=p, dim=dim).cpu()
        torch.testing.assert_close(ours, expected, rtol=0, atol=0)


@pytest.mark.parametrize("p", _GENERAL_P)
def test_vector_norm_general_p_size_one_reduce_is_abs(mojo_gpu, p):
    x = torch.tensor([[1e20], [-3.0], [0.0]])
    ours = torch.linalg.vector_norm(x.to(mojo_gpu), ord=p, dim=1).cpu()
    torch.testing.assert_close(ours, torch.linalg.vector_norm(x, ord=p, dim=1))


@pytest.mark.parametrize("p", _GENERAL_P)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_vector_norm_general_p_with_an_accumulation_dtype(mojo_gpu, p, dtype):
    x = torch.randn(64, 257).to(dtype)
    got = torch.linalg.vector_norm(x.to(mojo_gpu), p, dim=1, dtype=torch.float32)
    assert got.dtype == torch.float32
    torch.testing.assert_close(
        got.cpu(),
        torch.linalg.vector_norm(x, p, dim=1, dtype=torch.float32),
        rtol=1e-5,
        atol=0,
    )


@pytest.mark.parametrize("p", _GENERAL_P)
def test_vector_norm_general_p_out_resizes(mojo_gpu, p):
    x = torch.randn(5, 7)
    out = torch.empty(1, device=mojo_gpu)  # wrong shape
    returned = torch.linalg.vector_norm(x.to(mojo_gpu), ord=p, dim=1, out=out)
    assert returned.data_ptr() == out.data_ptr() and out.shape == (5,)
    torch.testing.assert_close(out.cpu(), torch.linalg.vector_norm(x, ord=p, dim=1))
    # A non-contiguous out= goes through the compute-then-copy route.
    wide = torch.zeros(5, 2, device=mojo_gpu)
    torch.linalg.vector_norm(x.to(mojo_gpu), ord=p, dim=1, out=wide[:, 0])
    torch.testing.assert_close(
        wide[:, 0].cpu(), torch.linalg.vector_norm(x, ord=p, dim=1)
    )
    assert (wide[:, 1].cpu() == 0).all()


def test_vector_norm_general_p_declines_integer_input(mojo_gpu):
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(torch.randint(0, 9, (4, 5)).to(mojo_gpu), ord=3)


# ---------------------------------------------------------------------------
# linalg_vector_norm(ord=-inf): min of |x|, NormNegInfOp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [torch.float64, torch.float32, torch.float16, torch.bfloat16]
)
@pytest.mark.parametrize(
    "shape,dim",
    [((4099, 1031), 1), ((5003, 37), 1), ((1031, 4099), 0), ((1 << 20,), 0)],
)
def test_vector_norm_neginf_matches_torch(mojo_gpu, shape, dim, dtype):
    """ord=-inf selects the min |x|, an already-rounded, exactly
    representable element (not an accumulation), so exact equality is the
    right bar -- a loose tolerance would hide a selection error."""
    if dtype is torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    x = (torch.rand(shape) * 0.9 + 0.05).to(dtype)
    ours = torch.linalg.vector_norm(x.to(mojo_gpu), ord=float("-inf"), dim=dim).cpu()
    expected = torch.linalg.vector_norm(x.double(), ord=float("-inf"), dim=dim)
    torch.testing.assert_close(ours.double(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dims", [(0,), (1,), (0, 1), (-1,), None])
def test_vector_norm_neginf_dims_and_keepdim(mojo_gpu, dims, keepdim):
    x = torch.rand(5, 7) * 0.9 + 0.05
    kwargs = {} if dims is None else {"dim": dims}
    expected = torch.linalg.vector_norm(x, ord=float("-inf"), keepdim=keepdim, **kwargs)
    ours = torch.linalg.vector_norm(
        x.to(mojo_gpu), ord=float("-inf"), keepdim=keepdim, **kwargs
    ).cpu()
    torch.testing.assert_close(ours, expected)


def test_vector_norm_neginf_noncontiguous(mojo_gpu):
    contiguous = torch.linspace(-3.0, 4.0, 35).reshape(5, 7)
    strided = contiguous.t()
    assert not strided.is_contiguous()
    expected = torch.linalg.vector_norm(strided, ord=float("-inf"))
    device_strided = contiguous.to(mojo_gpu).t()
    ours = torch.linalg.vector_norm(device_strided, ord=float("-inf")).cpu()
    torch.testing.assert_close(ours, expected)


def test_vector_norm_neginf_nan_propagates(mojo_gpu):
    x = torch.tensor([1.0, float("nan"), 2.0])
    expected = torch.linalg.vector_norm(x, ord=float("-inf"))
    ours = torch.linalg.vector_norm(x.to(mojo_gpu), ord=float("-inf")).cpu()
    torch.testing.assert_close(ours, expected, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vector_norm_neginf_with_an_accumulation_dtype(mojo_gpu, dtype):
    cpu = torch.randn(4096, dtype=dtype)
    expected = torch.linalg.vector_norm(cpu, float("-inf"), dtype=torch.float32)
    got = torch.linalg.vector_norm(cpu.to(mojo_gpu), float("-inf"), dtype=torch.float32)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-5, atol=1e-4)


def test_vector_norm_neginf_out_resizes(mojo_gpu):
    """A wrongly-shaped `out=` is resized, matching every ATen out= op."""
    x = torch.rand(5, 7) * 0.9 + 0.05
    expected = torch.linalg.vector_norm(x, ord=float("-inf"), dim=1)
    out = torch.empty(1, dtype=torch.float32, device=mojo_gpu)  # wrong shape
    returned = torch.linalg.vector_norm(
        x.to(mojo_gpu), ord=float("-inf"), dim=1, out=out
    )
    assert returned.data_ptr() == out.data_ptr()
    assert out.shape == (5,)
    torch.testing.assert_close(out.cpu(), expected)


def test_vector_norm_neginf_empty_reduce_dim_declines(mojo_gpu):
    """Unlike the L2 norm (identity 0), ord=-inf has no identity: torch
    refuses a reduction over a zero-length axis, and so must we -- even when
    the output itself is also empty (`_refuse_empty_extremum` looks only at
    the reduced dim's own extent)."""
    x = torch.empty(3, 0).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(x, ord=float("-inf"), dim=1)
    x00 = torch.empty(0, 0).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(x00, ord=float("-inf"), dim=1)
    # An empty OUTPUT is fine when the reduced dim itself is non-empty.
    empty_out = torch.empty(0, 7).to(mojo_gpu)
    torch.testing.assert_close(
        torch.linalg.vector_norm(empty_out, ord=float("-inf"), dim=1).cpu(),
        torch.empty(0),
    )


def test_vector_norm_neginf_declines_non_floating_input(mojo_gpu):
    """The dtype gate (`_vector_norm_operand`) is shared by every ord: an
    integer/bool input is rejected before `dtype=` is even looked at, same as
    the ord=2 path (`test_vector_norm_declines_non_floating_input`)."""
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(
            torch.randint(0, 9, (4, 5)).to(mojo_gpu), ord=float("-inf"), dim=1
        )
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(
            torch.randint(0, 9, (4, 5)).to(mojo_gpu),
            ord=float("-inf"),
            dim=1,
            dtype=torch.float32,
        )
    with pytest.raises(NotImplementedError):
        torch.linalg.vector_norm(
            (torch.randn(4, 5) > 0).to(mojo_gpu),
            ord=float("-inf"),
            dim=1,
            dtype=torch.float32,
        )


# ---------------------------------------------------------------------------
# rank-0 (0-d) operands: torch treats a 0-d tensor as if it were 1-d of size
# 1 (`maybe_wrap_dim`'s 0-d exception), so dim=0/-1/None/[] are all valid and
# reduce the single element, keepdim never changes the (always 0-d) shape,
# and any other dim is out of range. `_reduce_dims` normalizes every one of
# these specs to an empty dims list for a rank-0 operand. `_adjacent_reduce_geom`
# returns False for zero given dims (it never matches an empty interval), so
# the rank-0 case falls through to its caller's own fallback -- `_reduce_spec_geom`
# already computes (outer=1, reduce_n=1, inner=1) there, so most of these ops
# need no kernel-side change at all -- see `_ready_operand`'s early return
# and the `_argreduce_spec_into` rank-0 branch for the one spot that had no
# such fallback and needed one added.
# ---------------------------------------------------------------------------

_RANK0_GENERIC_OPS = [op for op in _REDUCE_OPS if op not in ("prod", "count_nonzero")]


@pytest.mark.parametrize("op", _RANK0_GENERIC_OPS)
@pytest.mark.parametrize("dim", [None, 0, -1, []])
@pytest.mark.parametrize("keepdim", [False, True])
def test_reduce_skeleton_rank0_matches_torch(mojo_gpu, op, dim, keepdim):
    fn = _REDUCE_OPS[op]
    x = torch.tensor(True) if op in ("all", "any") else torch.tensor(-3.5)
    kw = {"dim": dim, "keepdim": keepdim}
    expected = fn(x, **kw)
    ours = fn(x.to(mojo_gpu), **kw).cpu()
    assert ours.shape == expected.shape == ()
    assert ours.dtype == expected.dtype
    if ours.dtype == torch.bool:
        torch.testing.assert_close(ours, expected)
    else:
        torch.testing.assert_close(
            ours.double(), expected.double(), atol=1e-6, rtol=1e-4
        )


@pytest.mark.parametrize("op", _RANK0_GENERIC_OPS)
def test_reduce_skeleton_rank0_out_of_range_dim_declines(mojo_gpu, op):
    fn = _REDUCE_OPS[op]
    x = torch.tensor(True) if op in ("all", "any") else torch.tensor(-3.5)
    match = (
        "Dimension out of range \\(expected to be in range of \\[-1, 0\\], but got 1\\)"
    )
    with pytest.raises(IndexError, match=match):
        fn(x, dim=1)
    with pytest.raises(IndexError, match=match):
        fn(x.to(mojo_gpu), dim=1)


@pytest.mark.parametrize("op", _RANK0_GENERIC_OPS)
@pytest.mark.parametrize("dims", [[0, -1], [0, 0], [-1, -1]])
def test_reduce_skeleton_rank0_duplicate_dim_declines(mojo_gpu, op, dims):
    """Every entry of `dims` normalizes to the same (only) dim, 0, on a
    rank-0 operand: a second one is a duplicate, same as torch's own
    refusal (confirmed on real CUDA: "dim 0 appears multiple times")."""
    fn = _REDUCE_OPS[op]
    x = torch.tensor(True) if op in ("all", "any") else torch.tensor(-3.5)
    match = "dim 0 appears multiple times in the list of dims"
    with pytest.raises(RuntimeError, match=match):
        fn(x, dim=dims)
    with pytest.raises(RuntimeError, match=match):
        fn(x.to(mojo_gpu), dim=dims)


@pytest.mark.parametrize("op", _RANK0_GENERIC_OPS)
@pytest.mark.parametrize("dim", [5, -5])
def test_reduce_skeleton_rank2_out_of_range_int_dim_declines(mojo_gpu, op, dim):
    """Rank-2 counterpart of the rank-0 test above: range is [-2, 1]."""
    fn = _REDUCE_OPS[op]
    x = (
        torch.tensor([[True, False], [False, True]])
        if op in ("all", "any")
        else torch.randn(2, 3)
    )
    match = (
        r"Dimension out of range \(expected to be in range of \[-2, 1\], but got "
        + str(dim)
        + r"\)"
    )
    with pytest.raises(IndexError, match=match):
        fn(x, dim=dim)
    with pytest.raises(IndexError, match=match):
        fn(x.to(mojo_gpu), dim=dim)


@pytest.mark.parametrize("op", _RANK0_GENERIC_OPS)
def test_reduce_skeleton_rank2_out_of_range_dim_in_list_declines(mojo_gpu, op):
    fn = _REDUCE_OPS[op]
    x = (
        torch.tensor([[True, False], [False, True]])
        if op in ("all", "any")
        else torch.randn(2, 3)
    )
    match = (
        r"Dimension out of range \(expected to be in range of \[-2, 1\], but got 5\)"
    )
    with pytest.raises(IndexError, match=match):
        fn(x, dim=[0, 5])
    with pytest.raises(IndexError, match=match):
        fn(x.to(mojo_gpu), dim=[0, 5])


@pytest.mark.parametrize("op", _RANK0_GENERIC_OPS)
@pytest.mark.parametrize("dims", [[0, 0], [1, -1], [0, -2]])
def test_reduce_skeleton_rank2_duplicate_dim_declines(mojo_gpu, op, dims):
    """`[1, -1]` and `[0, -2]` are aliases of the same normalized dim on a
    rank-2 operand, same as a literal repeat."""
    fn = _REDUCE_OPS[op]
    x = (
        torch.tensor([[True, False], [False, True]])
        if op in ("all", "any")
        else torch.randn(2, 3)
    )
    norm = dims[0] if dims[0] >= 0 else dims[0] + 2
    match = f"dim {norm} appears multiple times in the list of dims"
    with pytest.raises(RuntimeError, match=match):
        fn(x, dim=dims)
    with pytest.raises(RuntimeError, match=match):
        fn(x.to(mojo_gpu), dim=dims)


@pytest.mark.parametrize("dim", [5, -5])
def test_prod_and_count_nonzero_rank2_out_of_range_dim_declines(mojo_gpu, dim):
    """`prod`/`count_nonzero` take their own path (`_RANK0_GENERIC_OPS`
    excludes them for unrelated keepdim/signature reasons) but share
    `_reduce_dims`/`_norm_dim`, so the same IndexError applies."""
    x = torch.randn(2, 3)
    match = (
        r"Dimension out of range \(expected to be in range of \[-2, 1\], but got "
        + str(dim)
        + r"\)"
    )
    with pytest.raises(IndexError, match=match):
        torch.prod(x, dim=dim)
    with pytest.raises(IndexError, match=match):
        torch.prod(x.to(mojo_gpu), dim=dim)
    with pytest.raises(IndexError, match=match):
        torch.count_nonzero(x, dim=dim)
    with pytest.raises(IndexError, match=match):
        torch.count_nonzero(x.to(mojo_gpu), dim=dim)


def test_count_nonzero_rank2_duplicate_dim_declines(mojo_gpu):
    x = torch.randn(2, 3)
    match = "dim 0 appears multiple times in the list of dims"
    with pytest.raises(RuntimeError, match=match):
        torch.count_nonzero(x, dim=[0, -2])
    with pytest.raises(RuntimeError, match=match):
        torch.count_nonzero(x.to(mojo_gpu), dim=[0, -2])


def test_var_rank2_out_of_range_and_duplicate_dim_declines(mojo_gpu):
    x = torch.randn(2, 3)
    match_oor = (
        r"Dimension out of range \(expected to be in range of \[-2, 1\], but got 5\)"
    )
    with pytest.raises(IndexError, match=match_oor):
        torch.var(x, dim=5)
    with pytest.raises(IndexError, match=match_oor):
        torch.var(x.to(mojo_gpu), dim=5)
    match_dup = "dim 0 appears multiple times in the list of dims"
    with pytest.raises(RuntimeError, match=match_dup):
        torch.var(x, dim=[0, -2])
    with pytest.raises(RuntimeError, match=match_dup):
        torch.var(x.to(mojo_gpu), dim=[0, -2])


@pytest.mark.parametrize("fn", [torch.argmax, torch.argmin])
def test_argreduce_rank2_out_of_range_dim_declines(mojo_gpu, fn):
    x = torch.randn(2, 3)
    match = (
        r"Dimension out of range \(expected to be in range of \[-2, 1\], but got 5\)"
    )
    with pytest.raises(IndexError, match=match):
        fn(x, dim=5)
    with pytest.raises(IndexError, match=match):
        fn(x.to(mojo_gpu), dim=5)


def test_min_dim_rank2_out_of_range_dim_declines(mojo_gpu):
    x = torch.randn(2, 3)
    match = (
        r"Dimension out of range \(expected to be in range of \[-2, 1\], but got 5\)"
    )
    with pytest.raises(IndexError, match=match):
        torch.min(x, dim=5)
    with pytest.raises(IndexError, match=match):
        torch.min(x.to(mojo_gpu), dim=5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32, torch.bool])
def test_sum_rank0_dtype_promotion(mojo_gpu, dtype):
    """The bool/sub-int64 -> int64 promotion applies at rank 0 exactly as it
    does at any other rank."""
    x = (
        torch.tensor(True, dtype=dtype)
        if dtype is torch.bool
        else torch.tensor(3, dtype=dtype)
    )
    xd = x.to(mojo_gpu)
    for dim in (None, 0, -1, []):
        expected = torch.sum(x, dim=dim)
        got = torch.sum(xd, dim=dim)
        assert got.dtype == expected.dtype
        torch.testing.assert_close(got.cpu(), expected)


def test_sum_out_rank0(mojo_gpu):
    x = torch.tensor(-3.5)
    xd = x.to(mojo_gpu)
    expected = torch.sum(x, dim=0)
    # A wrongly-shaped out= is resized, same as any other rank.
    out = torch.empty(5, device=mojo_gpu)
    returned = torch.sum(xd, dim=0, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert out.shape == ()
    torch.testing.assert_close(out.cpu(), expected)


def test_prod_rank0(mojo_gpu):
    """`prod()` (no dim) and `prod.dim_int` (dim required, 0-d exception)."""
    x = torch.tensor(1.3)
    xd = x.to(mojo_gpu)
    torch.testing.assert_close(torch.prod(xd).cpu(), torch.prod(x))
    for dim in (0, -1):
        torch.testing.assert_close(
            torch.prod(xd, dim=dim).cpu(), torch.prod(x, dim=dim)
        )
    match = (
        "Dimension out of range \\(expected to be in range of \\[-1, 0\\], but got 1\\)"
    )
    with pytest.raises(IndexError, match=match):
        torch.prod(x, dim=1)
    with pytest.raises(IndexError, match=match):
        torch.prod(xd, dim=1)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_max_and_min_full_reduction_rank0(mojo_device, dtype):
    if dtype == torch.float64:
        skip_if_metal(mojo_device, "no float64 on Apple GPUs")
    x = (
        torch.tensor(-3.5, dtype=dtype)
        if dtype.is_floating_point
        else torch.tensor(3, dtype=dtype)
    )
    xd = x.to(mojo_device)
    torch.testing.assert_close(torch.max(xd).cpu(), torch.max(x))
    torch.testing.assert_close(torch.min(xd).cpu(), torch.min(x))
    torch.testing.assert_close(torch.amax(xd, dim=0).cpu(), torch.amax(x, dim=0))
    torch.testing.assert_close(torch.amin(xd, dim=[]).cpu(), torch.amin(x, dim=[]))


def test_max_unary_out_rank0_resizes(mojo_device):
    """A wrongly-shaped out= is resized (confirmed on real CUDA, which warns
    but still resizes -- torch.Resize.cpp's deprecated auto-resize path)."""
    x = torch.tensor(-3.5)
    xd = x.to(mojo_device)
    expected = torch.max(x)
    out = torch.empty(5, device=mojo_device)
    returned = torch.max(xd, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert out.shape == ()
    torch.testing.assert_close(out.cpu(), expected)


@pytest.mark.parametrize("dim", [0, -1])
def test_min_dim_rank0(mojo_gpu, dim):
    x = torch.tensor(2.5)
    xd = x.to(mojo_gpu)
    exp_v, exp_i = torch.min(x, dim=dim)
    got_v, got_i = torch.min(xd, dim=dim)
    assert got_v.shape == exp_v.shape == ()
    torch.testing.assert_close(got_v.cpu(), exp_v)
    torch.testing.assert_close(got_i.cpu(), exp_i)


def test_min_dim_rank0_out_of_range_declines(mojo_gpu):
    x = torch.tensor(2.5)
    match = (
        "Dimension out of range \\(expected to be in range of \\[-1, 0\\], but got 1\\)"
    )
    with pytest.raises(IndexError, match=match):
        torch.min(x, dim=1)
    with pytest.raises(IndexError, match=match):
        torch.min(x.to(mojo_gpu), dim=1)


@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_argreduce_rank0(mojo_gpu, dim, keepdim):
    x = torch.tensor(2.5)
    xd = x.to(mojo_gpu)
    kw = {"keepdim": keepdim}
    if dim is not None:
        kw["dim"] = dim
    exp_max = torch.argmax(x, **kw)
    exp_min = torch.argmin(x, **kw)
    got_max = torch.argmax(xd, **kw)
    got_min = torch.argmin(xd, **kw)
    assert got_max.shape == exp_max.shape
    torch.testing.assert_close(got_max.cpu(), exp_max)
    torch.testing.assert_close(got_min.cpu(), exp_min)


def test_argreduce_rank0_out_of_range_declines(mojo_gpu):
    x = torch.tensor(2.5)
    match = (
        "Dimension out of range \\(expected to be in range of \\[-1, 0\\], but got 1\\)"
    )
    with pytest.raises(IndexError, match=match):
        torch.argmax(x, dim=1)
    with pytest.raises(IndexError, match=match):
        torch.argmax(x.to(mojo_gpu), dim=1)


@pytest.mark.parametrize("correction", [0, 1])
def test_var_correction_rank0(mojo_gpu, correction):
    """A single element has zero degrees of freedom above `correction=0`:
    torch answers 0 there and NaN (0/0) at `correction=1` (its own default),
    the same divide stock CUDA performs -- confirmed on an actual H100."""
    x = torch.tensor(2.5)
    xd = x.to(mojo_gpu)
    expected = torch.var(x, dim=0, correction=correction)
    got = torch.var(xd, dim=0, correction=correction)
    torch.testing.assert_close(got.cpu(), expected, equal_nan=True)


def test_var_correction_rank0_out_of_range_declines(mojo_gpu):
    x = torch.tensor(2.5)
    match = (
        "Dimension out of range \\(expected to be in range of \\[-1, 0\\], but got 1\\)"
    )
    with pytest.raises(IndexError, match=match):
        torch.var(x, dim=1)
    with pytest.raises(IndexError, match=match):
        torch.var(x.to(mojo_gpu), dim=1)


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


def test_cumsum_narrows_a_float64_self_with_explicit_dtype(mojo_gpu):
    """`cumsum(x_float64, dtype=torch.float32)`: the exact case that exposed
    `_is_castable` checking only `dtype=`'s target and not the float64 SELF
    -- `_promote`'s cast-then-accumulate reaches `_cast` with a float64
    source regardless of the (non-float64) target. Asserts the decline on
    Metal instead of skipping past it."""
    x = torch.randn(4, 5, dtype=torch.float64)
    if is_metal(mojo_gpu):
        with pytest.raises(NotImplementedError):
            torch.cumsum(x.to(mojo_gpu), dim=1, dtype=torch.float32)
        return
    expected = torch.cumsum(x, dim=1, dtype=torch.float32)
    got = torch.cumsum(x.to(mojo_gpu), dim=1, dtype=torch.float32).cpu()
    torch.testing.assert_close(got, expected)


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


@pytest.mark.parametrize("dim", [5, -5])
def test_cumsum_rank2_out_of_range_dim_declines(mojo_gpu, dim):
    x = torch.randn(4, 5)
    match = (
        r"Dimension out of range \(expected to be in range of \[-2, 1\], but got "
        + str(dim)
        + r"\)"
    )
    with pytest.raises(IndexError, match=match):
        torch.cumsum(x, dim=dim)
    with pytest.raises(IndexError, match=match):
        torch.cumsum(x.to(mojo_gpu), dim=dim)


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
    with pytest.raises(NotImplementedError):
        # int8/int16 promote to int64 through a cast the cast kernel does not
        # dispatch on (same gap sum.IntList_out/cumsum document elsewhere).
        torch.tensor(3, dtype=torch.int8).to(mojo_gpu).sum()
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
    (
        "aten::mean.dtype_out",
        lambda d: torch.mean(torch.randn(4, 5).to(d), out=torch.empty((), device=d)),
    ),
    ("aten::amax", lambda d: torch.amax(torch.randn(4, 5).to(d), dim=1)),
    ("aten::amin", lambda d: torch.amin(torch.randn(4, 5).to(d), dim=1)),
    ("aten::max", lambda d: torch.max(torch.randn(4, 5).to(d))),
    (
        "aten::max.unary_out",
        lambda d: torch.max(torch.randn(4, 5).to(d), out=torch.empty((), device=d)),
    ),
    ("aten::min", lambda d: torch.min(torch.randn(4, 5).to(d))),
    (
        "aten::min.unary_out",
        lambda d: torch.min(torch.randn(4, 5).to(d), out=torch.empty((), device=d)),
    ),
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
    (
        "aten::all.out",
        lambda d: torch.all(
            _bools().to(d), dim=1, out=torch.empty(4, dtype=torch.bool, device=d)
        ),
    ),
    (
        "aten::all.all_out",
        lambda d: torch.all(
            _bools().to(d), out=torch.empty((), dtype=torch.bool, device=d)
        ),
    ),
    (
        "aten::all.dims_out",
        lambda d: torch.all(
            _bools().to(d), dim=[0, 1], out=torch.empty((), dtype=torch.bool, device=d)
        ),
    ),
    ("aten::any", lambda d: torch.any(_bools().to(d))),
    ("aten::any.dim", lambda d: torch.any(_bools().to(d), dim=1)),
    ("aten::any.dims", lambda d: torch.ops.aten.any.dims(_bools().to(d), [0, 1])),
    (
        "aten::any.out",
        lambda d: torch.any(
            _bools().to(d), dim=1, out=torch.empty(4, dtype=torch.bool, device=d)
        ),
    ),
    (
        "aten::any.all_out",
        lambda d: torch.any(
            _bools().to(d), out=torch.empty((), dtype=torch.bool, device=d)
        ),
    ),
    (
        "aten::any.dims_out",
        lambda d: torch.ops.aten.any.dims_out(
            _bools().to(d), [0, 1], out=torch.empty((), dtype=torch.bool, device=d)
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
    (
        "aten::norm.Scalar",
        lambda d: torch.ops.aten.norm.Scalar(torch.randn(4, 5).to(d)),
    ),
    (
        "aten::norm.ScalarOpt_dtype",
        lambda d: torch.ops.aten.norm.ScalarOpt_dtype(
            torch.randn(4, 5).to(d), None, dtype=torch.float32
        ),
    ),
    (
        "aten::norm.ScalarOpt_dim",
        lambda d: torch.ops.aten.norm.ScalarOpt_dim(
            torch.randn(4, 5).to(d), 2, [1], False
        ),
    ),
    (
        "aten::norm.ScalarOpt_dim_dtype",
        lambda d: torch.ops.aten.norm.ScalarOpt_dim_dtype(
            torch.randn(4, 5).to(d), None, [1], False, dtype=torch.float32
        ),
    ),
    (
        "aten::norm.out",
        lambda d: torch.ops.aten.norm.out(
            torch.randn(4, 5).to(d), 2, [1], False, out=torch.empty(4, device=d)
        ),
    ),
    (
        "aten::norm.dtype_out",
        lambda d: torch.ops.aten.norm.dtype_out(
            torch.randn(4, 5).to(d),
            None,
            [1],
            False,
            dtype=torch.float32,
            out=torch.empty(4, dtype=torch.float32, device=d),
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
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
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
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
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


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.float64]
)
def test_sort_orders_nan_and_signed_zero_like_aten(mojo_gpu, dtype):
    """A negative NaN's bits sit below -inf and -0.0's below +0.0, but ATen
    orders every NaN above every number and treats the two zeros as equal."""
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
    nan = float("nan")
    inf = float("inf")
    x = torch.tensor([[0.0, -0.0, nan, -nan, inf, -inf, 1.0, -1.0, nan, 0.0]])
    x = x.to(dtype)
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
    with pytest.raises(IndexError, match="Dimension out of range"):
        torch.topk(x, 2, dim=2)
    with pytest.raises(IndexError, match="Dimension out of range"):
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
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "no float64 on Apple GPUs")
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
    with pytest.raises(IndexError, match="out of range"):
        torch.median(device, 2)
    with pytest.raises((IndexError, RuntimeError), match="non-zero size"):
        torch.median(torch.empty(2, 0).to(mojo_gpu), 1)
    with pytest.raises(RuntimeError, match="not implemented for 'Bool'"):
        torch.median(torch.tensor([True, False]).to(mojo_gpu), 0)
