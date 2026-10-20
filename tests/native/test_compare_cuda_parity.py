"""Comparisons, logical / bitwise / shift ops, maximum / minimum / fmax /
fmin, clamp, where, masked_fill and isclose on the mojo device against stock
CUDA, bit for bit (tests/native/cuda_oracle.py): every dtype, `out=`, in
place, Python-number and 0-d operands, type promotion, broadcasting and
strided operands, and the errors CUDA raises. Operands:
tests/native/binary_parity.py.
"""

import functools
import math
import operator
from collections.abc import Callable, Sequence

import pytest
import torch

from tests.native import binary_parity as bp
from tests.native.cuda_oracle import CudaOracle

_DTYPES = bp.ALL + bp.UINTS
_NAN = bp.nan_scalar(0x7FF8000000000123)
_SNAN = bp.nan_scalar(0x7FF0000000000077)


@functools.cache
def selection_pairs(dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """`bp.pairs` without dropping the float64 pairs of two distinct NaNs: a
    selection (max/min, clamp, where) returns one operand's bits as they are,
    so the pick is deterministic."""
    gen = torch.Generator().manual_seed(0)
    s = bp.specials(dtype)
    if dtype.is_complex:
        s = s[torch.randperm(len(s), generator=gen)[:200]]
    ra, rb = bp.random_values(dtype, 4096, gen), bp.random_values(dtype, 4096, gen)
    m = min(len(ra), len(rb))
    return (
        torch.cat([s.repeat_interleave(len(s)), ra[:m]]),
        torch.cat([s.repeat(len(s)), rb[:m]]),
    )


@functools.cache
def selection_triples(
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every triple of (up to 30) specials, NaNs of every payload kept, then
    random triples."""
    gen = torch.Generator().manual_seed(1)
    s = bp.specials(dtype)
    if len(s) > 30:
        nan = (
            s.isnan()
            if dtype.is_floating_point
            else torch.zeros_like(s, dtype=torch.bool)
        )
        rest = s[~nan]
        rest = rest[torch.randperm(len(rest), generator=gen)[: 30 - int(nan.sum())]]
        s = torch.cat([rest, s[nan]])
    k = len(s)
    r = [bp.random_values(dtype, 4096, gen) for _ in range(3)]
    m = min(len(x) for x in r)
    return (
        torch.cat([s.repeat_interleave(k * k), r[0][:m]]),
        torch.cat([s.repeat_interleave(k).repeat(k), r[1][:m]]),
        torch.cat([s.repeat(k * k), r[2][:m]]),
    )


def _no_half_nans(t: torch.Tensor) -> torch.Tensor:
    """`t` with float16 NaNs replaced by 1: a float16 NaN converted to
    float64 is CastSpec's known mismatch (test_half_nan_into_float64)."""
    if t.dtype != torch.float16:
        return t
    return torch.where(t.isnan(), torch.ones_like(t), t)


def _mask(n: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(5)
    return torch.rand(n, generator=gen) < 0.5


Cases = list[tuple[str, Callable[..., object], Sequence[object]]]


def _out_cases(fn, a, b, dtypes) -> Cases:
    """`fn(a, b, out=o)` for an empty `o` of each dtype."""

    def with_out(x, y, dt):
        o = torch.empty(0, dtype=dt, device=x.device)
        return fn(x, y, out=o), o

    return [
        (f"out_{bp.name(dt)}", lambda x, y, dt=dt: with_out(x, y, dt), (a, b))
        for dt in dtypes
    ]


def _zero_dim_cases(fn, a, b) -> Cases:
    """One operand a 0-d tensor, on the device or on the CPU, either side."""
    return [
        ("t-dev0d", lambda x, y: fn(x, y[7]), (a, b)),
        ("dev0d-t", lambda x, y: fn(x[7], y), (a, b)),
        ("t-cpu0d", lambda x, y: fn(x, y[7].cpu()), (a, b)),
        ("cpu0d-t", lambda x, y: fn(x[7].cpu(), y), (a, b)),
    ]


def _scalar_0d_cases(fn, a, s, *, left: bool = True, right: bool = True) -> Cases:
    """`s` as a 0-d tensor operand: on the CPU (CUDA's CPU-scalar path; built
    inside the call, as `assert_bitwise_equal` moves tensor arguments to the
    device) and on the device (an ordinary operand), right and/or left."""
    cases: Cases = []
    if right:
        cases.append((f"t0d={s!r}", lambda x, s=s: fn(x, torch.tensor(s)), (a,)))
        cases.append(
            (
                f"tdev0d={s!r}",
                lambda x, s=s: fn(x, torch.tensor(s, device=x.device)),
                (a,),
            )
        )
    if left:
        cases.append((f"0dt={s!r}", lambda x, s=s: fn(torch.tensor(s), x), (a,)))
        cases.append(
            (
                f"dev0dt={s!r}",
                lambda x, s=s: fn(torch.tensor(s, device=x.device), x),
                (a,),
            )
        )
    return cases


# -- comparisons -------------------------------------------------------------

_COMPARE = ["eq", "ne", "lt", "le", "gt", "ge"]
_COMPARE_SCALARS = (
    bp.FLOAT_SCALARS
    + bp.COMPLEX_SCALARS
    + [1.5, -0.5, math.nan, 300, -129, 65504, 65505.0, 2**31, -(2**31) - 1]
    + [2**40, 2**63 - 1, -(2**63), 16777217, 3.4e38, 1e-46, _SNAN]
)


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize("op", _COMPARE)
def test_compare_tensor_tensor(cuda_oracle: CudaOracle, op, dtype):
    fn = getattr(torch, op)
    a, b = bp.pairs(dtype)
    cases = bp.binary_cases(fn, dtype, inplace=op + "_")
    cases += _out_cases(fn, a, b, [torch.float32, torch.int64, torch.uint8])
    cases += _zero_dim_cases(fn, a, b)
    cases.append(("operator", getattr(operator, op), (a, b)))
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize("op", _COMPARE)
def test_compare_scalar(cuda_oracle: CudaOracle, op, dtype):
    fn = getattr(torch, op)
    a, _ = bp.pairs(dtype)
    cases = bp.scalar_cases(fn, dtype, _COMPARE_SCALARS, inplace=op + "_")
    if dtype == torch.float16:
        # a device complex 0-d meeting a half tensor: complex32 (the xfail below)
        cases = [c for c in cases if not (c[0].startswith("tdev0d=") and "j" in c[0])]
    for s in (2.5, 3, _NAN, True):
        # the scalar on the left: the reflected comparison
        cases += _scalar_0d_cases(fn, a, s, right=False)
        cases.append((f"op_st={s!r}", lambda y, s=s: getattr(operator, op)(s, y), (a,)))
    # Python ints beyond int64
    cases.append(("huge", lambda x: fn(x, 2**64 - 1), (a,)))
    cases.append(("huge_neg", lambda x: fn(x, -(2**63) - 1), (a,)))
    cases += _out_cases(lambda x, y, out: fn(x, 0.5, out=out), a, a, [torch.bool])
    cases += _out_cases(lambda x, y, out: fn(x, 2, out=out), a, a, [torch.float16])
    bp.check_all(cuda_oracle, cases)


@pytest.mark.xfail(
    strict=True,
    reason="float16 against a device complex 0-d tensor computes in complex32,"
    " a dtype the device does not hold",
)
@pytest.mark.parametrize("op", ["eq", "ne"])
def test_compare_half_vs_device_complex_0d(cuda_oracle: CudaOracle, op):
    a, _ = bp.pairs(torch.float16)
    fn = getattr(torch, op)
    cases = [
        (f"tdev0d={s!r}", lambda x, s=s: fn(x, torch.tensor(s, device=x.device)), (a,))
        for s in (1j, 2 + 3j, -0.5 + 0j)
    ]
    bp.check_all(cuda_oracle, cases)


_MIXED = [
    (torch.float16, torch.float32), (torch.bfloat16, torch.float16),
    (torch.int32, torch.float16), (torch.int64, torch.float32),
    (torch.uint8, torch.int8), (torch.bool, torch.int32),
    (torch.float32, torch.float64), (torch.int8, torch.bfloat16),
    (torch.float32, torch.complex64), (torch.float64, torch.complex64),
    (torch.bool, torch.float16), (torch.int64, torch.float64),
    (torch.uint8, torch.bool), (torch.int16, torch.int64),
    (torch.uint16, torch.int32), (torch.uint32, torch.int64),
]  # fmt: skip
_MIXED_IDS = [f"{bp.name(a)}-{bp.name(b)}" for a, b in _MIXED]


def _mixed_operands(da, db):
    a, _ = selection_pairs(da)
    _, b = selection_pairs(db)
    n = min(len(a), len(b))
    return a[:n], b[:n]


def _mixed_cases(fn, da, db, inplace: str | None = None) -> Cases:
    a, b = _mixed_operands(da, db)
    cases = [("tt", fn, (a, b)), *_zero_dim_cases(fn, a, b)]
    if inplace is not None:
        cases.append(("inplace", lambda x, y: getattr(x, inplace)(y), (a, b)))
    return cases


@pytest.mark.parametrize(("da", "db"), _MIXED, ids=_MIXED_IDS)
@pytest.mark.parametrize("op", _COMPARE)
def test_compare_mixed_dtypes(cuda_oracle: CudaOracle, op, da, db):
    bp.check_all(cuda_oracle, _mixed_cases(getattr(torch, op), da, db, op + "_"))


# -- logical -----------------------------------------------------------------

_LOGICAL = ["logical_and", "logical_or", "logical_xor"]


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize("op", _LOGICAL)
def test_logical(cuda_oracle: CudaOracle, op, dtype):
    fn = getattr(torch, op)
    a, b = bp.pairs(dtype)
    cases = bp.binary_cases(fn, dtype, inplace=op + "_")
    cases += _out_cases(fn, a, b, [torch.float32, torch.int64, torch.complex64])
    cases += _zero_dim_cases(fn, a, b)
    for s in (0, 2, 0.5, -0.0, _NAN, True, 1j):
        cases += _scalar_0d_cases(fn, a, s)
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize(("da", "db"), _MIXED, ids=_MIXED_IDS)
@pytest.mark.parametrize("op", _LOGICAL)
def test_logical_mixed_dtypes(cuda_oracle: CudaOracle, op, da, db):
    bp.check_all(cuda_oracle, _mixed_cases(getattr(torch, op), da, db, op + "_"))


# -- bitwise and shifts --------------------------------------------------------

_BITWISE = [
    ("bitwise_and", operator.and_),
    ("bitwise_or", operator.or_),
    ("bitwise_xor", operator.xor),
]
_BIT_SCALARS = [*bp.INT_SCALARS, 0x7F, -(2**31), 2**40, -(2**63), False]


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize(("op", "dunder"), _BITWISE, ids=[b[0] for b in _BITWISE])
def test_bitwise(cuda_oracle: CudaOracle, op, dunder, dtype):
    fn = getattr(torch, op)
    a, b = bp.pairs(dtype)
    cases = bp.binary_cases(fn, dtype, inplace=op + "_")
    cases += _zero_dim_cases(fn, a, b)
    cases += bp.scalar_cases(fn, dtype, _BIT_SCALARS, inplace=op + "_")
    cases += _out_cases(lambda x, y, out: fn(x, 3, out=out), a, a, [dtype])
    cases.append(("dunder", dunder, (a, b)))
    cases.append(("dunder_s", lambda x: dunder(x, 5), (a,)))
    cases.append(("dunder_rs", lambda x: dunder(5, x), (a,)))
    cases.append(("reverse_scalar", lambda x: fn(6, x), (a,)))
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize(("da", "db"), _MIXED, ids=_MIXED_IDS)
@pytest.mark.parametrize("op", [b[0] for b in _BITWISE])
def test_bitwise_mixed_dtypes(cuda_oracle: CudaOracle, op, da, db):
    bp.check_all(cuda_oracle, _mixed_cases(getattr(torch, op), da, db, op + "_"))


_SHIFTS = [
    ("left", torch.bitwise_left_shift, operator.lshift, "__ilshift__"),
    ("right", torch.bitwise_right_shift, operator.rshift, "__irshift__"),
]
_SHIFT_AMOUNTS = [0, 1, 2, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 100, 300]
_SHIFT_AMOUNTS = [*_SHIFT_AMOUNTS, -1, -5, -64, 2**40, -(2**63), True, 2.0, 2.5, -0.0]


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize(
    ("op", "fn", "dunder", "idunder"), _SHIFTS, ids=[s[0] for s in _SHIFTS]
)
def test_shift(cuda_oracle: CudaOracle, op, fn, dunder, idunder, dtype):
    a, b = bp.pairs(dtype)
    if not dtype.is_floating_point and not dtype.is_complex and dtype != torch.bool:
        # every shift amount of every operand, and some beyond the width
        amounts = bp.int_specials(dtype)
        b = torch.cat([b, amounts.repeat(len(a) // len(amounts) + 1)[: len(a)]])
        a = torch.cat([a, a])
    method = f"bitwise_{op}_shift_"
    cases: Cases = [
        ("tt", fn, (a, b)),
        *_out_cases(fn, a, b, [dtype]),
        ("inplace", lambda x, y: getattr(x, method)(y), (a, b)),
        ("dunder", dunder, (a, b)),
        ("idunder", lambda x, y: getattr(x, idunder)(y), (a, b)),
        *_zero_dim_cases(fn, a, b),
    ]
    for s in _SHIFT_AMOUNTS:
        cases.append((f"ts={s!r}", lambda x, s=s: fn(x, s), (a,)))
        cases += _scalar_0d_cases(fn, a, s, left=False)
        cases.append((f"dunder_s={s!r}", lambda x, s=s: dunder(x, s), (a,)))
        cases.append((f"idunder_s={s!r}", lambda x, s=s: getattr(x, idunder)(s), (a,)))
        cases.append((f"inplace_s={s!r}", lambda x, s=s: getattr(x, method)(s), (a,)))
    for s in (1, 3, -7, 2.0, 255):
        cases.append((f"st={s!r}", lambda y, s=s: fn(s, y), (b,)))
        cases.append((f"dunder_st={s!r}", lambda y, s=s: dunder(s, y), (b,)))
        cases += _scalar_0d_cases(fn, b, s, right=False)
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize(("da", "db"), _MIXED, ids=_MIXED_IDS)
@pytest.mark.parametrize("fn", [s[1] for s in _SHIFTS], ids=[s[0] for s in _SHIFTS])
def test_shift_mixed_dtypes(cuda_oracle: CudaOracle, fn, da, db):
    a, b = _mixed_operands(da, db)
    bp.check_all(cuda_oracle, [("tt", fn, (a, b)), *_zero_dim_cases(fn, a, b)])


# -- maximum / minimum / fmax / fmin --------------------------------------------

_MAXMIN = ["maximum", "minimum", "fmax", "fmin"]


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize("op", _MAXMIN)
def test_maxmin(cuda_oracle: CudaOracle, op, dtype):
    fn = getattr(torch, op)
    a, b = selection_pairs(dtype)
    if dtype == torch.float64 and op in ("fmax", "fmin"):
        # two tensor NaNs: CUDA's pick depends on the launch
        # (test_float64_fmax_two_nans_follow_cuda_launch)
        keep = ~bp.two_distinct_nans(a, b)
        a, b = a[keep], b[keep]
    cases: Cases = [("tt", fn, (a, b)), ("swapped", fn, (b, a))]
    cases += _out_cases(fn, a, b, [dtype])
    cases += _out_cases(fn, _no_half_nans(a), _no_half_nans(b), [torch.float64])
    cases += _zero_dim_cases(fn, a, b)
    for s in (2.5, -0.0, 0.0, _NAN, _SNAN, math.inf, 3, True, 1e30):
        if dtype == torch.float64 and op in ("fmax", "fmin") and s != s:
            # a device 0-d NaN against the tensor's NaNs: two tensor NaNs
            cases += _scalar_0d_cases(fn, a, s, left=True, right=True)[::2]
            continue
        cases += _scalar_0d_cases(fn, a, s)
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize(("da", "db"), _MIXED, ids=_MIXED_IDS)
@pytest.mark.parametrize("op", _MAXMIN)
def test_maxmin_mixed_dtypes(cuda_oracle: CudaOracle, op, da, db):
    fn = getattr(torch, op)
    if op in ("fmax", "fmin") and torch.float64 in (da, db):
        # two NaNs of different payloads in float64: CUDA's pick follows the
        # launch (test_float64_fmax_two_nans_follow_cuda_launch)
        a, b = _mixed_operands(da, db)
        keep = ~bp.two_distinct_nans(a.to(torch.float64), b.to(torch.float64))
        a, b = a[keep], b[keep]
        cases: Cases = [("tt", fn, (a, b)), *_zero_dim_cases(fn, a, b)]
        bp.check_all(cuda_oracle, cases)
        return
    bp.check_all(cuda_oracle, _mixed_cases(fn, da, db))


# -- clamp ---------------------------------------------------------------------

_BOUNDS = [
    (-1.5, 2), (-1, 2), (2, -1), (0, 0), (-0.0, 0.0), (0.0, -0.0),
    (math.nan, 2), (-1, math.nan), (_NAN, _SNAN), (_SNAN, None), (None, _NAN),
    (None, 0.5), (0.5, None), (None, None), (-math.inf, math.inf),
    (math.inf, -math.inf), (True, False), (300, -129), (-129, 300),
    (0.1, 0.2), (1e-40, 1e30), (-70000.0, 70000.0), (2**40, 2**41),
    (1 + 1j, 2), (False, True),
]  # fmt: skip
_CLAMP_DTYPES = bp.FLOATS + bp.INTS + bp.BOOL + bp.UINTS + bp.COMPLEX


@pytest.mark.parametrize("dtype", _CLAMP_DTYPES, ids=bp.name)
def test_clamp_scalar(cuda_oracle: CudaOracle, dtype):
    a, _ = selection_pairs(dtype)
    cases: Cases = []
    for lo, hi in _BOUNDS:
        tag = f"{lo!r},{hi!r}"
        cases.append(
            (f"clamp({tag})", lambda x, lo=lo, hi=hi: torch.clamp(x, lo, hi), (a,))
        )
        cases.append((f"clamp_({tag})", lambda x, lo=lo, hi=hi: x.clamp_(lo, hi), (a,)))

        def with_out(x, lo=lo, hi=hi):
            o = torch.empty(0, dtype=x.dtype, device=x.device)
            return torch.clamp(x, lo, hi, out=o), o

        cases.append((f"clamp_out({tag})", with_out, (a,)))
        if lo is not None:
            cases.append(
                (f"clamp_min({lo!r})", lambda x, lo=lo: torch.clamp_min(x, lo), (a,))
            )
            cases.append(
                (f"clamp_min_({lo!r})", lambda x, lo=lo: x.clamp_min_(lo), (a,))
            )
        if hi is not None:
            cases.append(
                (f"clamp_max({hi!r})", lambda x, hi=hi: torch.clamp_max(x, hi), (a,))
            )
            cases.append(
                (f"clamp_max_({hi!r})", lambda x, hi=hi: x.clamp_max_(hi), (a,))
            )

        # the same bounds as 0-d CPU tensors: the Tensor overloads
        def bound(v, device=None):
            return None if v is None else torch.tensor(v, device=device)

        cases.append(
            (
                f"clamp_cpu0d({tag})",
                lambda x, lo=lo, hi=hi: torch.clamp(x, bound(lo), bound(hi)),
                (a,),
            )
        )
        cases.append(
            (
                f"clamp_dev0d({tag})",
                lambda x, lo=lo, hi=hi: torch.clamp(
                    x, bound(lo, x.device), bound(hi, x.device)
                ),
                (a,),
            )
        )
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize("dtype", _CLAMP_DTYPES, ids=bp.name)
def test_clamp_tensor(cuda_oracle: CudaOracle, dtype):
    x, lo, hi = selection_triples(dtype)

    def with_out(v, lo, hi):
        o = torch.empty(0, dtype=v.dtype, device=v.device)
        return torch.clamp(v, lo, hi, out=o), o

    cases: Cases = [
        ("clamp", torch.clamp, (x, lo, hi)),
        ("clamp_min_kw", lambda v, lo: torch.clamp(v, min=lo), (x, lo)),
        ("clamp_max_kw", lambda v, hi: torch.clamp(v, max=hi), (x, hi)),
        ("clamp_min", torch.clamp_min, (x, lo)),
        ("clamp_max", torch.clamp_max, (x, hi)),
        ("clamp_", lambda v, lo, hi: v.clamp_(lo, hi), (x, lo, hi)),
        ("clamp_min_", lambda v, lo: v.clamp_min_(lo), (x, lo)),
        ("clamp_max_", lambda v, hi: v.clamp_max_(hi), (x, hi)),
        ("clamp_out", with_out, (x, lo, hi)),
        ("clamp_none", lambda v: torch.clamp(v, None, None), (x,)),
        ("dev0d", lambda v, lo, hi: torch.clamp(v, lo[3], hi[9]), (x, lo, hi)),
        ("mixed_0d", lambda v, lo: torch.clamp(v, lo[3], 2), (x, lo)),
        ("min_none", lambda v, hi: torch.clamp(v, None, hi), (x, hi)),
    ]

    def clamp_min_out(v, lo):
        o = torch.empty(0, dtype=v.dtype, device=v.device)
        return torch.clamp_min(v, lo, out=o), o

    def clamp_max_out(v, hi):
        o = torch.empty(0, dtype=v.dtype, device=v.device)
        return torch.clamp_max(v, hi, out=o), o

    cases += [("clamp_min_out", clamp_min_out, (x, lo))]
    cases += [("clamp_max_out", clamp_max_out, (x, hi))]
    bp.check_all(cuda_oracle, cases)


_CLAMP_MIXED = [
    (torch.int32, torch.float32), (torch.int64, torch.float16),
    (torch.float16, torch.float32), (torch.bfloat16, torch.float64),
    (torch.float32, torch.int64), (torch.uint8, torch.int8),
    (torch.bool, torch.float32), (torch.float32, torch.bool),
    (torch.int8, torch.int64), (torch.float64, torch.float16),
]  # fmt: skip


@pytest.mark.parametrize(
    ("dx", "dbound"),
    _CLAMP_MIXED,
    ids=[f"{bp.name(a)}-{bp.name(b)}" for a, b in _CLAMP_MIXED],
)
def test_clamp_mixed_dtypes(cuda_oracle: CudaOracle, dx, dbound):
    x, _, _ = selection_triples(dx)
    _, lo, hi = selection_triples(dbound)
    n = min(len(x), len(lo))
    x, lo, hi = _no_half_nans(x[:n]), _no_half_nans(lo[:n]), _no_half_nans(hi[:n])

    def with_out(v, lo, hi, dt):
        o = torch.empty(0, dtype=dt, device=v.device)
        return torch.clamp(v, lo, hi, out=o), o

    cases: Cases = [
        ("clamp", torch.clamp, (x, lo, hi)),
        ("clamp_min", torch.clamp_min, (x, lo)),
        ("clamp_max", torch.clamp_max, (x, hi)),
        ("clamp_", lambda v, lo, hi: v.clamp_(lo, hi), (x, lo, hi)),
        (
            "cpu0d",
            lambda v, lo, hi: torch.clamp(v, lo[3].cpu(), hi[9].cpu()),
            (x, lo, hi),
        ),
        ("out_x", lambda v, lo, hi: with_out(v, lo, hi, dx), (x, lo, hi)),
        ("out_f64", lambda v, lo, hi: with_out(v, lo, hi, torch.float64), (x, lo, hi)),
    ]
    bp.check_all(cuda_oracle, cases)


# -- where -----------------------------------------------------------------------


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
def test_where(cuda_oracle: CudaOracle, dtype):
    a, b = selection_pairs(dtype)
    c = _mask(len(a))

    def with_out(m, x, y, dt):
        o = torch.empty(0, dtype=dt, device=x.device)
        return torch.where(m, x, y, out=o), o

    cases: Cases = [
        ("tt", torch.where, (c, a, b)),
        ("method", lambda m, x, y: x.where(m, y), (c, a, b)),
        ("out", lambda m, x, y: with_out(m, x, y, dtype), (c, a, b)),
        ("out_f64", lambda m, x, y: with_out(m, x, y, torch.float64), (c, a, b)),
        ("dev0d_other", lambda m, x, y: torch.where(m, x, y[7]), (c, a, b)),
        ("dev0d_self", lambda m, x, y: torch.where(m, x[7], y), (c, a, b)),
        ("cpu0d_other", lambda m, x, y: torch.where(m, x, y[7].cpu()), (c, a, b)),
        ("cpu0d_self", lambda m, x, y: torch.where(m, x[7].cpu(), y), (c, a, b)),
        ("cond0d", lambda m, x, y: torch.where(m[1], x, y), (c, a, b)),
        ("cond_cpu0d", lambda m, x, y: torch.where(m[1].cpu(), x, y), (c, a, b)),
        ("uint8_cond", lambda m, x, y: torch.where(m.to(torch.uint8), x, y), (c, a, b)),
        ("float_cond", lambda m, x, y: torch.where(m.float(), x, y), (c, a, b)),
    ]
    for s in (2.5, 7, -0.0, True, _NAN, 1e30, 300, 1 + 2j):
        if dtype == torch.float16 and s in (1e30, 1 + 2j):
            continue  # test_where_half_scalars
        cases.append((f"ts={s!r}", lambda m, x, s=s: torch.where(m, x, s), (c, a)))
        cases.append((f"st={s!r}", lambda m, x, s=s: torch.where(m, s, x), (c, a)))
        cases.append(
            (f"t0d={s!r}", lambda m, x, s=s: torch.where(m, x, torch.tensor(s)), (c, a))
        )
        cases.append(
            (f"0dt={s!r}", lambda m, x, s=s: torch.where(m, torch.tensor(s), x), (c, a))
        )
        cases.append(
            (
                f"tdev0d={s!r}",
                lambda m, x, s=s: torch.where(m, x, torch.tensor(s, device=x.device)),
                (c, a),
            )
        )
    bp.check_all(cuda_oracle, cases)


@pytest.mark.xfail(
    strict=True,
    reason="where(cond, half, 1e30) raises on CUDA (scalar_tensor fills with a"
    " checked Half conversion; our scalar_tensor narrows unchecked to serve"
    " Python item assignment) and where(cond, half, 1+2j) is a complex32"
    " result, a dtype the device does not hold",
)
def test_where_half_scalars(cuda_oracle: CudaOracle):
    a, _ = selection_pairs(torch.float16)
    c = _mask(len(a))
    cases: Cases = []
    for s in (1e30, 1 + 2j):
        cases.append((f"ts={s!r}", lambda m, x, s=s: torch.where(m, x, s), (c, a)))
        cases.append((f"st={s!r}", lambda m, x, s=s: torch.where(m, s, x), (c, a)))
    bp.check_all(cuda_oracle, cases)


def test_where_scalars(cuda_oracle: CudaOracle):
    c = _mask(64)
    cases: Cases = []
    for s, t in [(2.5, 1), (1, 2), (True, False), (1j, 2), (_NAN, 0.5), (2, 3.0)]:
        cases.append((f"{s!r},{t!r}", lambda m, s=s, t=t: torch.where(m, s, t), (c,)))
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize(("da", "db"), _MIXED, ids=_MIXED_IDS)
def test_where_mixed_dtypes(cuda_oracle: CudaOracle, da, db):
    a, b = _mixed_operands(da, db)
    c = _mask(len(a))
    fn = torch.where
    bp.check_all(
        cuda_oracle,
        [
            ("tt", fn, (c, a, b)),
            ("dev0d", lambda m, x, y: fn(m, x, y[7]), (c, a, b)),
            ("0ddev", lambda m, x, y: fn(m, x[7], y), (c, a, b)),
            ("cpu0d", lambda m, x, y: fn(m, x, y[7].cpu()), (c, a, b)),
            ("0dcpu", lambda m, x, y: fn(m, x[7].cpu(), y), (c, a, b)),
        ],
    )


# -- masked_fill -------------------------------------------------------------------

_FILL_VALUES = [3, 2.5, -1, 0, True, False, 300, -129, _NAN, _SNAN, -0.0]
_FILL_VALUES = [*_FILL_VALUES, math.inf, 1e30, 70000.0, 1e-40, 2**40, 1 + 2j]


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
def test_masked_fill(cuda_oracle: CudaOracle, dtype):
    a, _ = selection_pairs(dtype)
    m = _mask(len(a))
    cases: Cases = []
    for v in _FILL_VALUES:
        cases.append((f"s={v!r}", lambda x, mm, v=v: x.masked_fill(mm, v), (a, m)))
        cases.append((f"s_={v!r}", lambda x, mm, v=v: x.masked_fill_(mm, v), (a, m)))
        cases.append(
            (
                f"cpu0d={v!r}",
                lambda x, mm, v=v: x.masked_fill(mm, torch.tensor(v)),
                (a, m),
            )
        )
        cases.append(
            (
                f"dev0d={v!r}",
                lambda x, mm, v=v: x.masked_fill(mm, torch.tensor(v, device=x.device)),
                (a, m),
            )
        )
        cases.append(
            (
                f"cpu0d_={v!r}",
                lambda x, mm, v=v: x.masked_fill_(mm, torch.tensor(v)),
                (a, m),
            )
        )
    cases += [
        ("dev0d", lambda x, mm: x.masked_fill(mm, x[5]), (a, m)),
        ("dev0d_", lambda x, mm, y: x.masked_fill_(mm, y[5]), (a, m, a.clone())),
        ("dev0d_other_dtype", lambda x, mm: x.masked_fill(mm, mm[1]), (a, m)),
        (
            "dev0d_f64",
            lambda x, mm, t: x.masked_fill(mm, t[0]),
            (a, m, torch.tensor([_NAN, 2.0])),
        ),
        ("dim1_value", lambda x, mm: x.masked_fill(mm, x[:1]), (a, m)),
        ("uint8_mask", lambda x, mm: x.masked_fill(mm.to(torch.uint8), 1), (a, m)),
        ("all_true", lambda x, mm: x.masked_fill(mm | True, 2), (a, m)),
        ("functional", lambda x, mm: torch.masked_fill(x, mm, 4), (a, m)),
    ]
    bp.check_all(cuda_oracle, cases)


def test_masked_fill_broadcast(cuda_oracle: CudaOracle):
    def base(device):
        gen = torch.Generator().manual_seed(3)
        return (
            bp.random_values(torch.float32, 4096, gen)[: 64 * 48]
            .reshape(64, 48)
            .to(device)
        )

    def mask(device, *shape):
        gen = torch.Generator().manual_seed(4)
        return (torch.rand(*shape, generator=gen) < 0.5).to(device)

    cases = [
        ("row", lambda d: base(d).masked_fill(mask(d, 48), 2.0)),
        ("col", lambda d: base(d).masked_fill(mask(d, 64, 1), _NAN)),
        ("inplace_row", lambda d: base(d).masked_fill_(mask(d, 1, 48), -1)),
        ("strided_self", lambda d: base(d)[::2, 1::3].masked_fill(mask(d, 32, 16), 5)),
        ("strided_mask", lambda d: base(d).masked_fill(mask(d, 48, 64).t(), 5)),
        ("inplace_strided", lambda d: base(d).t().masked_fill_(mask(d, 48, 64), 7)),
        ("self_broadcast", lambda d: base(d)[:1].masked_fill(mask(d, 64, 48), 7)),
        (
            "inplace_self_broadcast",
            lambda d: base(d)[:1].masked_fill_(mask(d, 64, 48), 7),
        ),
    ]
    failures = []
    for label, f in cases:
        try:
            cuda_oracle.assert_bitwise_equal_by_device(f)
        except AssertionError as error:
            failures.append(f"{label}: {error}")
    assert not failures, "\n".join(failures)


# -- isclose -----------------------------------------------------------------------

_TOLS = [(1e-5, 1e-8), (0.0, 0.0), (0.1, 0.0), (0.0, 0.5), (1e-3, 1e-3), (2.0, 1e30)]
_TOLS += [(math.inf, 0.0), (-1.0, 0.0), (0.0, -1e-9), (math.nan, 0.0)]


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
def test_isclose(cuda_oracle: CudaOracle, dtype):
    a, b = bp.pairs(dtype)
    # near neighbours, where the tolerance decides
    if dtype.is_floating_point or dtype.is_complex:
        near = a * (1 + 1e-3) + 1e-6
        a, b = torch.cat([a, a]), torch.cat([b, near.to(dtype)])
    cases: Cases = []
    for rtol, atol in _TOLS:
        for equal_nan in (False, True):
            tag = f"rtol={rtol},atol={atol},equal_nan={equal_nan}"
            cases.append(
                (
                    tag,
                    lambda x, y, r=rtol, t=atol, e=equal_nan: torch.isclose(
                        x, y, rtol=r, atol=t, equal_nan=e
                    ),
                    (a, b),
                )
            )
    cases.append(("cpu0d", lambda x, y: torch.isclose(x, y[3].cpu()), (a, b)))
    cases.append(("dev0d", lambda x, y: torch.isclose(x[3], y), (a, b)))
    cases.append(("other_dtype", lambda x: torch.isclose(x, x.to(torch.float64)), (a,)))
    cases.append(("allclose", lambda x, y: torch.allclose(x, y), (a, b)))
    cases.append(
        ("allclose_self", lambda x: torch.allclose(x, x, equal_nan=True), (a,))
    )
    bp.check_all(cuda_oracle, cases)


_HALF_NANS = bp.from_bits([0x7E00, 0xFE00, 0x7E01, 0x7C01], torch.float16)


def test_half_nan_into_float64(cuda_oracle: CudaOracle):
    """A float16 NaN meeting a float64 computation or out=."""
    big = torch.tensor([1.0, -2.0, 3.0, 0.5], dtype=torch.float64)

    def max_out(x, y):
        o = torch.empty(0, dtype=torch.float64, device=x.device)
        return torch.maximum(x, y, out=o), o

    bp.check_all(
        cuda_oracle,
        [
            ("maximum_out_f64", max_out, (_HALF_NANS, _HALF_NANS.flip(0))),
            ("clamp_f64", torch.clamp, (big, _HALF_NANS, _HALF_NANS)),
            ("where_f64", lambda x, y: torch.where(x > 0, x, y), (big, _HALF_NANS)),
        ],
    )


@pytest.mark.xfail(
    strict=True,
    reason="stock CUDA keeps one of two float64 NaN payloads in fmax / fmin by"
    " launch (binary_parity.two_distinct_nans): no fixed rule matches both",
)
def test_float64_fmax_two_nans_follow_cuda_launch(cuda_oracle: CudaOracle):
    nans = bp.from_bits([0x7FF8000000000000, 0xFFF8000000000000], torch.float64)
    a = nans[:1].repeat(4096 + 36)
    b = nans[1:].repeat(4096 + 36)
    cuda_oracle.assert_bitwise_equal(torch.fmax, a, b)


# -- layouts -----------------------------------------------------------------------


def _layout_cases(fn, dtype):
    def base(device, seed=3):
        gen = torch.Generator().manual_seed(seed)
        return bp.random_values(dtype, 4096, gen)[: 64 * 48].reshape(64, 48).to(device)

    return [
        ("bcast_row", lambda d: fn(base(d), base(d, 4)[:1, :])),
        ("bcast_outer", lambda d: fn(base(d)[:, :1], base(d, 4)[:1, :])),
        ("transposed", lambda d: fn(base(d)[:48, :48].t(), base(d, 4)[16:, :])),
        ("stepped", lambda d: fn(base(d)[::3, 1::2], base(d, 4)[1::3, ::2])),
        (
            "rank5",
            lambda d: fn(
                base(d).reshape(2, 4, 2, 3, 64), base(d, 4).reshape(2, 4, 2, 3, 64)
            ),
        ),
        (
            "offset",
            lambda d: fn(base(d).flatten()[1:3001], base(d, 4).flatten()[5:3005]),
        ),
    ]


_LAYOUT_OPS = {
    "eq": torch.eq,
    "lt": torch.lt,
    "logical_xor": torch.logical_xor,
    "maximum": torch.maximum,
    "fmin": torch.fmin,
    "clamp": lambda x, y: torch.clamp(x, y, y),
    "clamp_min": torch.clamp_min,
    "where": lambda x, y: torch.where(x > y, x, y),
    "masked_fill": lambda x, y: x.masked_fill(x > y, 2),
    "bitwise_and": torch.bitwise_and,
    "lshift": torch.bitwise_left_shift,
    "isclose": torch.isclose,
    "lt_": lambda x, y: x.lt_(y),
    "clamp_": lambda x, y: x.clamp_(min=y),
}


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.float64, torch.int32, torch.uint8],
    ids=bp.name,
)
@pytest.mark.parametrize("op", list(_LAYOUT_OPS))
def test_layouts(cuda_oracle: CudaOracle, op, dtype):
    fn = _LAYOUT_OPS[op]
    failures = []
    for label, f in _layout_cases(fn, dtype):
        try:
            cuda_oracle.assert_bitwise_equal_by_device(f)
        except AssertionError as error:
            failures.append(f"{label}: {error}")
    assert not failures, "\n".join(failures)
