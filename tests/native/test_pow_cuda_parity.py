"""pow / float_power on the mojo device against stock CUDA, bit for bit
(tests/native/cuda_oracle.py): every dtype, `out=`, in place, Python-number
and 0-d (CPU and device) operands on either side, the exponents PowKernel.cu
special-cases (0.5, -0.5, -1 as sqrt / rsqrt / reciprocal; 2, 3, -2 as
products, tested on the exponent rounded to the tensor dtype), Pow.cpp's
0 / 1 shortcuts, integer pow (negative exponents, wrapping), mixed dtypes,
broadcasting and strided operands, and the errors. Operands:
tests/native/binary_parity.py.
"""

import math

import pytest
import torch

from tests.native import binary_parity as bp
from tests.native.cuda_oracle import CudaOracle

_NAN = bp.nan_scalar(0x7FF8000000000123)
# Exponents: the special ones in every spelling, values that round onto them
# in float32 / the half types but not in double (PowKernel.cu tests the
# exponent converted to the tensor dtype for 2 / 3 / -2, the original Scalar
# for 0.5 / -0.5 / -1), overflowing ones (Scalar::to<Half> raises), and
# ordinary ones.
_EXPONENTS = [
    0.5, -0.5, -1, -1.0, 2, 2.0, 3, 3.0, -2, -2.0, 0, 0.0, -0.0, 1, 1.0,
    True, False, 1.5, 0.25, 4, 10, -3, 7, 2.5, 0.1, 1 / 3, 1e-40, 1e30, 1e39,
    70000.0, 300, -129, 2**40, 0.5000000001, 2.0000000001, 3.0001, -1.0000000001,
    1.0000001, math.inf, -math.inf, _NAN,
]  # fmt: skip
_BASES = [
    2, 2.0, 0.5, -2.0, 10, 0, 0.0, -0.0, 1, 1.0, True, False, -1, -1.0,
    math.e, math.inf, -math.inf, _NAN, 1e-40, 70000.0, 3,
]  # fmt: skip
_COMPLEX = [1j, 2 + 1j, -0.5 + 0j, 2 + 0j, 0.5 + 0j, 1 + 0j, 0j]
_DTYPES = bp.ALL + bp.UINTS


def _split_nan(t):
    """Where a complex value holds two NaNs of different payload, as a
    double-precision kernel sees it. Which of the two a double DMUL / DFMA
    keeps depends on the operand slots ptxas gave the instruction, and those
    change with the launch (the contiguous kernel, the broadcast 0-d device
    one, the scalar-base one): bp.two_distinct_nans, within one element."""
    if not t.dtype.is_complex:
        return torch.zeros(len(t), dtype=torch.bool)
    r = torch.view_as_real(t.to(torch.complex128))
    return bp.two_distinct_nans(r[..., 0].contiguous(), r[..., 1].contiguous())


def _ops(dtype):
    """`pairs(dtype)` without the `_split_nan` elements (complex dtypes)."""
    a, b = bp.pairs(dtype)
    keep = ~(_split_nan(a) | _split_nan(b))
    return a[keep], b[keep]


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
def test_tensor_tensor(cuda_oracle: CudaOracle, dtype):
    bp.check_all(
        cuda_oracle,
        bp.binary_cases(torch.pow, dtype, inplace="pow_", operands=_ops(dtype)),
    )


def _integral(s: object) -> bool:
    return isinstance(s, (bool, int))


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
def test_tensor_scalar(cuda_oracle: CudaOracle, dtype):
    exps, bases = _EXPONENTS, _BASES
    if dtype in bp.UINTS:
        # (their floating promotion: test_uint_float_promotion)
        exps = [e for e in exps if _integral(e)]
        bases = [e for e in bases if _integral(e)]
    bp.check_all(
        cuda_oracle,
        bp.scalar_cases(
            torch.pow,
            dtype,
            exps,
            inplace="pow_",
            reverse=bases,
            operands=_ops(dtype),
        ),
    )


@pytest.mark.xfail(
    strict=True,
    reason="the mojo device has no result type for uint16 / uint32 / uint64"
    " with a floating Scalar ('type promotion of dtypes 27 and 6', the shared"
    " promotion code in tmb/ops; add / mul fail the same way)",
)
@pytest.mark.parametrize("dtype", bp.UINTS, ids=bp.name)
def test_uint_float_promotion(cuda_oracle: CudaOracle, dtype):
    bp.check_all(
        cuda_oracle, bp.scalar_cases(torch.pow, dtype, [0.5, 2.5], reverse=[2.0])
    )


@pytest.mark.parametrize(
    "dtype", [d for d in _DTYPES if d != torch.float16], ids=bp.name
)
def test_complex_scalar(cuda_oracle: CudaOracle, dtype):
    bp.check_all(
        cuda_oracle,
        bp.scalar_cases(
            torch.pow,
            dtype,
            _COMPLEX,
            inplace="pow_",
            reverse=_COMPLEX,
            operands=_ops(dtype),
        ),
    )


def _out_cases(fn, dtype, operands=None):
    """`out=` with a Python-number operand on either side."""
    a, b = _ops(dtype) if operands is None else operands
    cases = []
    for s in (0.5, 2, -1.0, 1.5, 0, 1, True):
        cases.append(
            (
                f"out_ts={s!r}",
                lambda x, s=s: fn(x, s, out=torch.empty_like(fn(x, s))),
                (a,),
            )
        )
        cases.append(
            (
                f"out_st={s!r}",
                lambda y, s=s: fn(s, y, out=torch.empty_like(fn(s, y))),
                (b,),
            )
        )
    return cases


@pytest.mark.parametrize("dtype", bp.FLOATS + bp.INTS + bp.COMPLEX, ids=bp.name)
def test_scalar_out(cuda_oracle: CudaOracle, dtype):
    bp.check_all(cuda_oracle, _out_cases(torch.pow, dtype))


# (a NaN number: _fp_nan_cases, as float_power's 0-d DEVICE NaN would meet
# the tensor's NaNs in a float64 kernel, bp.two_distinct_nans)
_FP_EXPONENTS = [2, 0.5, -1, -0.5, 3, -2, 0, 1, 1.5, True, 1j]
_FP_BASES = [2, 0.5, -1.0, 1, 0, 1j]


def _fp_nan_cases(a, b):
    nan = _NAN
    fp = torch.float_power
    return [
        ("ts=nan", lambda x: fp(x, nan), (a,)),
        ("t0d=nan", lambda x: fp(x, torch.tensor(nan)), (a,)),
        ("st=nan", lambda y: fp(nan, y), (b,)),
        ("0dt=nan", lambda y: fp(torch.tensor(nan), y), (b,)),
    ]


def _no_half_nans(dtype):
    """`pairs(dtype)` without float16 NaNs, which float_power converts to
    float64 (see test_float_power_half_nan)."""
    a, b = _ops(dtype)
    if dtype != torch.float16:
        return a, b
    keep = ~(a.isnan() | b.isnan())
    return a[keep], b[keep]


@pytest.mark.parametrize("dtype", bp.FLOATS + bp.INTS + bp.COMPLEX, ids=bp.name)
def test_float_power(cuda_oracle: CudaOracle, dtype):
    ops = _no_half_nans(dtype)
    cases = bp.binary_cases(
        torch.float_power, dtype, inplace="float_power_", operands=ops
    )
    cases += bp.scalar_cases(
        torch.float_power,
        dtype,
        _FP_EXPONENTS,
        inplace="float_power_",
        reverse=_FP_BASES,
        operands=ops,
    )
    cases += _out_cases(torch.float_power, dtype, ops)
    a, b = ops
    if dtype.is_floating_point:
        cases += _fp_nan_cases(a, b)
    # an out of the wrong dtype raises
    cases.append(
        (
            "out_wrong_dtype",
            lambda x, y: torch.float_power(
                x, y, out=torch.empty_like(x, dtype=torch.float32)
            ),
            (a, b),
        )
    )
    bp.check_all(cuda_oracle, cases)


@pytest.mark.xfail(
    strict=True,
    reason="float_power converts a float16 NaN to float64 with CastSpec, which"
    " keeps its payload, where stock CUDA's cast goes through float32 (the"
    " canonical 0x7fffffffe0000000); casts belong to another group",
)
def test_float_power_half_nan(cuda_oracle: CudaOracle):
    nans = bp.from_bits([0x7E00, 0xFE00, 0x7E01, 0x7C01], torch.float16)
    cuda_oracle.assert_bitwise_equal(torch.float_power, nans, 2.0)


_MIXED = [
    (torch.float16, torch.float32), (torch.bfloat16, torch.float16),
    (torch.int32, torch.float16), (torch.int64, torch.float32),
    (torch.uint8, torch.int8), (torch.bool, torch.int32),
    (torch.float32, torch.float64), (torch.int8, torch.bfloat16),
    (torch.float32, torch.complex64), (torch.float64, torch.complex64),
    (torch.bfloat16, torch.complex128), (torch.bool, torch.float16),
    (torch.int32, torch.int64), (torch.uint8, torch.bool),
    (torch.int16, torch.float64),
]  # fmt: skip


@pytest.mark.parametrize(
    ("da", "db"), _MIXED, ids=[f"{bp.name(a)}-{bp.name(b)}" for a, b in _MIXED]
)
@pytest.mark.parametrize("op", ["pow", "float_power"])
def test_mixed_dtypes(cuda_oracle: CudaOracle, op, da, db):
    """Type promotion: two tensors, and each one as a 0-d tensor (device and
    CPU; a CPU one is a CPU scalar to PowKernel.cu), both orders."""
    fn = getattr(torch, op)
    a, _ = bp.pairs(da)
    _, b = bp.pairs(db)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    common = torch.result_type(a, b)
    keep = ~(_split_nan(a) | _split_nan(b))
    if common == torch.float64 or (op == "float_power" and not common.is_complex):
        # two tensor NaNs in a double kernel: bp.two_distinct_nans
        keep &= ~bp.two_distinct_nans(a.double(), b.double())
    if op == "float_power" and torch.float16 in (da, db):
        keep &= ~(a.isnan() | b.isnan())  # test_float_power_half_nan
    a, b = a[keep], b[keep]
    cases = [("tt", fn, (a, b)), ("tt_rev", fn, (b, a))]
    for k in (3, 7, 11, 40):
        cases += [
            (f"t-dev0d[{k}]", lambda x, y, k=k: fn(x, y[k]), (a, b)),
            (f"dev0d-t[{k}]", lambda x, y, k=k: fn(x[k], y), (a, b)),
            (f"t-cpu0d[{k}]", lambda x, y, k=k: fn(x, y[k].cpu()), (a, b)),
            (f"cpu0d-t[{k}]", lambda x, y, k=k: fn(x[k].cpu(), y), (a, b)),
            (f"rev_t-cpu0d[{k}]", lambda x, y, k=k: fn(y, x[k].cpu()), (a, b)),
            (f"rev_cpu0d-t[{k}]", lambda x, y, k=k: fn(y[k].cpu(), x), (a, b)),
        ]
    bp.check_all(cuda_oracle, cases)


def _layout_cases(fn, dtype, no_nan=False):
    def base(device):
        gen = torch.Generator().manual_seed(3)
        t = bp.random_values(dtype, 4096, gen)[: 64 * 48].reshape(64, 48)
        if no_nan:
            t = torch.nan_to_num(t, nan=1.5)
        return t.to(device)

    def small(device):
        # exponents in a range where pow is mostly finite
        gen = torch.Generator().manual_seed(4)
        t = torch.randint(-3, 5, (64, 48), generator=gen)
        if dtype.is_floating_point or dtype.is_complex:
            t = t + torch.rand(64, 48, generator=gen) * 0.5
        return t.to(dtype).to(device)

    return [
        ("bcast_row", lambda d: fn(base(d), small(d)[:1, :])),
        ("bcast_outer", lambda d: fn(base(d)[:, :1], small(d)[:1, :])),
        ("transposed", lambda d: fn(base(d)[:48, :48].t(), small(d)[16:, :])),
        ("stepped", lambda d: fn(base(d)[::3, 1::2], small(d)[1::3, ::2])),
        (
            "rank5",
            lambda d: fn(
                base(d).reshape(2, 4, 2, 3, 64), small(d).reshape(2, 4, 2, 3, 64)
            ),
        ),
        ("offset", lambda d: fn(base(d).flatten()[1:3001], small(d).flatten()[5:3005])),
        ("scalar_strided", lambda d: fn(base(d)[::2, 1::3], 2.5)),
        ("scalar_sq_strided", lambda d: fn(base(d).t(), 2)),
        ("scalar_base_strided", lambda d: fn(1.5, small(d)[::2, 1::3])),
        ("inplace_strided", lambda d: base(d)[::2, 1::3].pow_(small(d)[::2, 1::3])),
    ]


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32, torch.float16, torch.bfloat16, torch.float64,
        torch.int32, torch.int8, torch.complex64, torch.complex128,
    ],
    ids=bp.name,
)  # fmt: skip
@pytest.mark.parametrize("op", ["pow", "float_power"])
def test_layouts(cuda_oracle: CudaOracle, op, dtype):
    fn = getattr(torch, op)
    failures = []
    half_fp = op == "float_power" and dtype == torch.float16
    for label, f in _layout_cases(fn, dtype, no_nan=half_fp):
        if op == "float_power" and label == "inplace_strided":
            continue
        try:
            cuda_oracle.assert_bitwise_equal_by_device(f)
        except AssertionError as error:
            failures.append(f"{label}: {error}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=bp.name)
def test_half_every_base(cuda_oracle: CudaOracle, dtype):
    """Every 16-bit pattern as the base, against each kind of exponent."""
    x = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32).to(torch.int16).view(dtype)
    exps = [0.5, -0.5, -1, 2, 3, -2, 1.5, 1 / 3, 0.1, -2.75, 7, 65504.0]
    cases = [(f"ts={e!r}", lambda t, e=e: torch.pow(t, e), (x,)) for e in exps]
    cases += [
        (f"st={e!r}", lambda t, e=e: torch.pow(e, t), (x,)) for e in (2, 0.5, 10, -3.0)
    ]
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=bp.name)
def test_random_sweep(cuda_oracle: CudaOracle, dtype):
    """Many random pairs: positive bases over the whole range against
    exponents of every magnitude, and bases near 1 with large exponents."""
    gen = torch.Generator().manual_seed(11)
    n = 1 << 18
    fi = torch.finfo(dtype)
    lo, hi = math.log(fi.tiny * fi.eps), math.log(fi.max)
    x = torch.empty(n, dtype=torch.float64).uniform_(lo, hi, generator=gen).exp()
    y = torch.randn(n, dtype=torch.float64, generator=gen) * torch.exp(
        torch.empty(n, dtype=torch.float64).uniform_(-8, 6, generator=gen)
    )
    near = 1 + torch.randn(n // 4, dtype=torch.float64, generator=gen) * 1e-3
    big = torch.randn(n // 4, dtype=torch.float64, generator=gen) * 1e5
    x = torch.cat([x, near, -x[: n // 8]]).to(dtype)
    y = torch.cat([y, big, y[: n // 8].round()]).to(dtype)
    cases = [("tt", torch.pow, (x, y)), ("ts=0.37", lambda t: torch.pow(t, 0.37), (x,))]
    cases.append(("st=1.0001", lambda t: torch.pow(1.0001, t), (y,)))
    bp.check_all(cuda_oracle, cases)
