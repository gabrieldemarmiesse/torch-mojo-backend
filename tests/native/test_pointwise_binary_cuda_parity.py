"""The binary / ternary pointwise math ops on the mojo device against stock
CUDA, bit for bit (tests/native/cuda_oracle.py): atan2, hypot, copysign,
nextafter, logaddexp(2), ldexp, fmax / fmin, heaviside, gcd / lcm, lerp,
addcmul and addcdiv. Every dtype (CUDA's refusals included), `out=`, in
place, Python-number and 0-d CPU / device operands, mixed dtypes,
broadcasting and strided operands. Operands: tests/native/binary_parity.py.
"""

import math

import pytest
import torch

from tests.native import binary_parity as bp
from tests.native.cuda_oracle import CudaOracle

_DTYPES = bp.ALL + bp.UINTS

# op, function, in-place method (None: torch has none)
_BINARY = [
    ("atan2", torch.atan2, "atan2_"),
    ("hypot", torch.hypot, "hypot_"),
    ("copysign", torch.copysign, "copysign_"),
    ("nextafter", torch.nextafter, "nextafter_"),
    ("logaddexp", torch.logaddexp, None),
    ("logaddexp2", torch.logaddexp2, None),
    ("ldexp", torch.ldexp, "ldexp_"),
    ("fmax", torch.fmax, None),
    ("fmin", torch.fmin, None),
    ("heaviside", torch.heaviside, "heaviside_"),
    ("gcd", torch.gcd, "gcd_"),
    ("lcm", torch.lcm, "lcm_"),
]
_BINARY_IDS = [b[0] for b in _BINARY]

# A double whose rounding to float32 lands on a half / bfloat16 midpoint:
# c10 converts a CPU scalar to Half / BFloat16 through float.
_HALF_TIE = 1 + 2**-11 + 2**-40
_BF16_TIE = 1 + 2**-8 + 2**-40


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize(("op", "fn", "inplace"), _BINARY, ids=_BINARY_IDS)
def test_tensor_tensor(cuda_oracle: CudaOracle, op, fn, inplace, dtype):
    bp.check_all(cuda_oracle, bp.binary_cases(fn, dtype, inplace=inplace))


def _scalar_values(dtype: torch.dtype) -> list[object]:
    if dtype.is_complex:
        return [1j, 2 + 3j, -0.5 + 0.0j, complex(math.inf, 1.0)]
    if dtype.is_floating_point:
        return [
            2.0, 0.5, -1.0, 0.1, 0.0, -0.0, 1e-40, 1e30, 70000.0, math.inf,
            -math.inf, bp.nan_scalar(0x7FF8000000000123),
            bp.nan_scalar(0xFFF0000000000001), _HALF_TIE, _BF16_TIE, 3, -7,
        ]  # fmt: skip
    if dtype == torch.bool:
        return [True, False]
    return [2, 3, -1, 0, 7, 300, -129, 12]


@pytest.mark.parametrize(
    "dtype",
    bp.FLOATS + (torch.int32, torch.int64, torch.uint8, torch.complex64),
    ids=bp.name,
)
@pytest.mark.parametrize(("op", "fn", "inplace"), _BINARY, ids=_BINARY_IDS)
def test_zero_dim_operands(cuda_oracle: CudaOracle, op, fn, inplace, dtype):
    """A 0-d operand on either side: a CPU tensor (CUDA's kernels read it as
    a scalar, `iter.scalar_value`) of the tensor's dtype and of the wider
    dtypes that do not promote, and a device tensor (an ordinary operand)."""
    a, b = bp.pairs(dtype)
    cases = []
    wide = torch.complex128 if dtype.is_complex else torch.float64
    for s in _scalar_values(dtype):
        for sd in dict.fromkeys((dtype, wide if dtype.is_floating_point else dtype)):
            if (
                isinstance(s, float)
                and not (sd.is_floating_point or sd.is_complex)
                or isinstance(s, complex)
                and not sd.is_complex
            ):
                continue
            tag = f"{s!r}:{bp.name(sd)}"
            cases.append(
                (
                    f"t-cpu0d={tag}",
                    lambda x, s=s, sd=sd: fn(x, torch.tensor(s, dtype=sd)),
                    (a,),
                )
            )
            cases.append(
                (
                    f"cpu0d-t={tag}",
                    lambda y, s=s, sd=sd: fn(torch.tensor(s, dtype=sd), y),
                    (b,),
                )
            )
            if inplace is not None:
                cases.append(
                    (
                        f"inplace-cpu0d={tag}",
                        lambda x, s=s, sd=sd: getattr(x, inplace)(
                            torch.tensor(s, dtype=sd)
                        ),
                        (a,),
                    )
                )
    cases.append(("t-dev0d", lambda x, y: fn(x, y[5]), (a, b)))
    cases.append(("dev0d-t", lambda x, y: fn(x[5], y), (a, b)))
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
def test_copysign_scalar(cuda_oracle: CudaOracle, dtype):
    """copysign.Scalar (and its in-place / out= forms): the number is a
    wrapped scalar tensor."""
    scalars = bp.FLOAT_SCALARS + [_HALF_TIE, -_BF16_TIE, 1j]

    def with_out(x, s):
        o = torch.empty(0, dtype=torch.copysign(x, s).dtype, device=x.device)
        return torch.copysign(x, s, out=o), o

    cases = bp.scalar_cases(torch.copysign, dtype, scalars, inplace="copysign_")
    a, _ = bp.pairs(dtype)
    for s in scalars:
        cases.append((f"out={s!r}", lambda x, s=s: with_out(x, s), (a,)))
    bp.check_all(cuda_oracle, cases)


_MIXED = [
    (torch.float16, torch.float32), (torch.bfloat16, torch.float16),
    (torch.int32, torch.float16), (torch.int64, torch.float32),
    (torch.uint8, torch.int8), (torch.bool, torch.int32),
    (torch.float32, torch.float64), (torch.int8, torch.bfloat16),
    (torch.float32, torch.complex64), (torch.float64, torch.complex64),
    (torch.bfloat16, torch.complex128), (torch.bool, torch.float16),
    (torch.int16, torch.int64),
]  # fmt: skip


@pytest.mark.parametrize(
    ("da", "db"), _MIXED, ids=[f"{bp.name(a)}-{bp.name(b)}" for a, b in _MIXED]
)
@pytest.mark.parametrize(("op", "fn", "inplace"), _BINARY, ids=_BINARY_IDS)
def test_mixed_dtypes(cuda_oracle: CudaOracle, op, fn, inplace, da, db):
    """Type promotion: two tensors, and each one as a 0-d tensor (device and
    CPU), which only counts when its category is higher."""
    a, _ = bp.pairs(da)
    _, b = bp.pairs(db)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    cases = [
        ("tt", fn, (a, b)),
        ("t-dev0d", lambda x, y: fn(x, y[7]), (a, b)),
        ("dev0d-t", lambda x, y: fn(x[7], y), (a, b)),
        ("t-cpu0d", lambda x, y: fn(x, y[7].cpu()), (a, b)),
        ("cpu0d-t", lambda x, y: fn(x[7].cpu(), y), (a, b)),
    ]
    if inplace is not None:
        cases.append(("inplace", lambda x, y: getattr(x, inplace)(y), (a, b)))

    def with_out(x, y, dt):
        o = torch.empty(0, dtype=dt, device=x.device)
        return fn(x, y, out=o), o

    for dt in (torch.float32, torch.float64, torch.int64, torch.float16):
        if dt == torch.float64 and torch.promote_types(da, db) == torch.float16:
            continue  # test_half_nan_into_float64_out
        cases.append(
            (f"out={bp.name(dt)}", lambda x, y, dt=dt: with_out(x, y, dt), (a, b))
        )
    bp.check_all(cuda_oracle, cases)


def _layout_cases(fn, dtype):
    def base(device):
        gen = torch.Generator().manual_seed(3)
        return bp.random_values(dtype, 4096, gen)[: 64 * 48].reshape(64, 48).to(device)

    return [
        ("bcast_row", lambda d: fn(base(d), base(d)[:1, :])),
        ("bcast_outer", lambda d: fn(base(d)[:, :1], base(d)[:1, :])),
        ("transposed", lambda d: fn(base(d)[:48, :48].t(), base(d)[16:, :])),
        ("stepped", lambda d: fn(base(d)[::3, 1::2], base(d)[1::3, ::2])),
        (
            "rank5",
            lambda d: fn(
                base(d).reshape(2, 4, 2, 3, 64), base(d).reshape(2, 4, 2, 3, 64)
            ),
        ),
        ("offset", lambda d: fn(base(d).flatten()[1:3001], base(d).flatten()[5:3005])),
        ("empty", lambda d: fn(base(d)[:0], base(d)[:0])),
    ]


def _check_layouts(cuda_oracle: CudaOracle, cases):
    failures = []
    for label, f in cases:
        try:
            cuda_oracle.assert_bitwise_equal_by_device(f)
        except AssertionError as error:
            failures.append(f"{label}: {error}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.float64, torch.int32], ids=bp.name
)
@pytest.mark.parametrize(("op", "fn", "inplace"), _BINARY, ids=_BINARY_IDS)
def test_layouts(cuda_oracle: CudaOracle, op, fn, inplace, dtype):
    _check_layouts(cuda_oracle, _layout_cases(fn, dtype))


_EXP_DTYPES = (
    torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.bool,
)  # fmt: skip


def _exponents(dtype: torch.dtype, n: int) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.arange(n) % 2 == 0
    ii = torch.iinfo(dtype)
    gen = torch.Generator().manual_seed(5)
    e = torch.randint(-1100, 1100, (n,), generator=gen)
    small = torch.arange(n) % 80 - 40
    e = torch.where(torch.arange(n) % 2 == 0, e, small)
    e[:8] = torch.tensor([0, 1, -1, 2**31 - 1, -(2**31), 2**40, -(2**40), 127])
    return e.clamp(ii.min, ii.max).to(dtype)


@pytest.mark.parametrize("edtype", _EXP_DTYPES, ids=bp.name)
@pytest.mark.parametrize("dtype", bp.FLOATS, ids=bp.name)
def test_ldexp_integer_exponent(cuda_oracle: CudaOracle, dtype, edtype):
    """ldexp(float, integer): ATen's `_ldexp_int_exponent` (ldexp_stub, the
    C ::ldexp per element, the result in self's dtype)."""
    a, _ = bp.pairs(dtype)
    e = _exponents(edtype, len(a))

    def with_out(x, y, dt):
        o = torch.empty(0, dtype=dt, device=x.device)
        return torch.ldexp(x, y, out=o), o

    cases = [
        ("tt", torch.ldexp, (a, e)),
        ("inplace", lambda x, y: x.ldexp_(y), (a, e)),
        ("out", lambda x, y: with_out(x, y, dtype), (a, e)),
        ("out_f64", lambda x, y: with_out(x, y, torch.float64), (a, e)),
    ]
    if dtype == torch.float16:
        cases.pop()  # test_half_nan_into_float64_out
    cases += [
        ("out_int", lambda x, y: with_out(x, y, torch.int32), (a, e)),
        ("t-cpu0d", lambda x: torch.ldexp(x, torch.tensor(3, dtype=edtype)), (a,)),
        ("t-dev0d", lambda x, y: torch.ldexp(x, y[3]), (a, e)),
        ("bcast", lambda x, y: torch.ldexp(x[:64].reshape(8, 8), y[:8]), (a, e)),
    ]
    bp.check_all(cuda_oracle, cases)


# ---------------------------------------------------------------------------
# lerp
# ---------------------------------------------------------------------------

_WEIGHTS = [
    0.0, 0.25, 0.5, 0.49999997, 0.4999999999, 0.75, 1.0, -0.5, 2.0, -3.5,
    0.1, math.inf, -math.inf, bp.nan_scalar(0x7FF8000000000123), 1e-40, 3,
    _HALF_TIE / 2,
]  # fmt: skip


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
def test_lerp_scalar(cuda_oracle: CudaOracle, dtype):
    a, b = bp.pairs(dtype)
    weights = _WEIGHTS + ([0.3 + 0.4j, 1j, 0.2 - 0.1j] if dtype.is_complex else [])
    cases = []

    def with_out(x, y, w):
        o = torch.empty(0, dtype=x.dtype, device=x.device)
        return torch.lerp(x, y, w, out=o), o

    for w in weights:
        x, y = a, b
        if dtype == torch.float64 and isinstance(w, float) and math.isnan(w):
            keep = ~(a.isnan() | b.isnan())  # test_lerp_float64_nan_weight_lanes
            x, y = a[keep], b[keep]
        cases.append((f"w={w!r}", lambda x, y, w=w: torch.lerp(x, y, w), (x, y)))
        cases.append((f"inplace/w={w!r}", lambda x, y, w=w: x.lerp_(y, w), (x, y)))
        cases.append((f"out/w={w!r}", lambda x, y, w=w: with_out(x, y, w), (x, y)))
    bp.check_all(cuda_oracle, cases)


@pytest.mark.xfail(
    strict=True,
    reason="float64 lerp with a NaN Scalar weight and a NaN operand: Lerp.cu's"
    " vectorized SASS computes end - diff * (1 - w) as DFMA(diff, 1 - w, end) in"
    " lane 0 and DFMA(1 - w, diff, end) in lanes 1-3, so the kept NaN (1 - w's"
    " or diff's) depends on the element's lane; the mojo device keeps w's",
)
def test_lerp_float64_nan_weight_lanes(cuda_oracle: CudaOracle):
    nan = bp.from_bits([0x7FF8000000000001], torch.float64)
    a = torch.zeros(4096, dtype=torch.float64)
    b = nan.repeat(4096)
    cuda_oracle.assert_bitwise_equal(
        lambda x, y: torch.lerp(x, y, bp.nan_scalar(0x7FF8000000000123)), a, b
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
def test_lerp_tensor(cuda_oracle: CudaOracle, dtype):
    a, b, c = bp.triples(dtype)
    if dtype.is_floating_point:
        # weights around the branch point and in [0, 1]
        gen = torch.Generator().manual_seed(9)
        c = c.clone()
        c[1::3] = torch.rand(len(c[1::3]), generator=gen, dtype=torch.float64).to(dtype)
        c[2::7] = torch.tensor(0.5).to(dtype).nextafter(torch.tensor(0.0).to(dtype))

    def with_out(x, y, z):
        o = torch.empty(0, dtype=x.dtype, device=x.device)
        return torch.lerp(x, y, z, out=o), o

    cases = [
        ("ttt", torch.lerp, (a, b, c)),
        ("inplace", lambda x, y, z: x.lerp_(y, z), (a, b, c)),
        ("out", with_out, (a, b, c)),
        ("w-dev0d", lambda x, y, z: torch.lerp(x, y, z[11]), (a, b, c)),
        (
            "bcast",
            lambda x, y, z: torch.lerp(
                x[:64].reshape(8, 8), y[:8], z[:64].reshape(8, 8)
            ),
            (a, b, c),
        ),
        ("strided", lambda x, y, z: torch.lerp(x[::2], y[1::2], z[::2]), (a, b, c)),
    ]
    for w in (0.25, 0.75, 0.4999999999, bp.nan_scalar(0x7FF8000000000123)):
        a2, b2 = a, b
        if dtype == torch.float64 and math.isnan(w):
            keep = ~(a.isnan() | b.isnan())  # test_lerp_float64_nan_weight_lanes
            a2, b2 = a[keep], b[keep]
        for wd in (dtype, torch.float64, torch.float32):
            if dtype.is_complex and not wd.is_complex:
                continue
            cases.append(
                (
                    f"w-cpu0d={w!r}:{bp.name(wd)}",
                    lambda x, y, w=w, wd=wd: torch.lerp(
                        x, y, torch.tensor(w, dtype=wd)
                    ),
                    (a2, b2),
                )
            )
            cases.append(
                (
                    f"inplace-w-cpu0d={w!r}:{bp.name(wd)}",
                    lambda x, y, w=w, wd=wd: x.lerp_(y, torch.tensor(w, dtype=wd)),
                    (a2, b2),
                )
            )
    bp.check_all(cuda_oracle, cases)


def test_lerp_dtype_errors(cuda_oracle: CudaOracle):
    a, b = bp.pairs(torch.float32)
    h = a.half()
    w = torch.full_like(a, 0.3)
    cases = [
        ("end", torch.lerp, (a, h, w)),
        ("weight", torch.lerp, (a, b, w.double())),
        (
            "w0d_f64",
            lambda x, y: torch.lerp(x, y, torch.tensor(0.3, dtype=torch.float64)),
            (a, b),
        ),
        (
            "w0d_half",
            lambda x, y: torch.lerp(x, y, torch.tensor(0.3, dtype=torch.float16)),
            (a, b),
        ),
        ("w0d_int", lambda x, y: torch.lerp(x, y, torch.tensor(1)), (a, b)),
        ("w0d_cplx", lambda x, y: torch.lerp(x, y, torch.tensor(0.5j)), (a, b)),
        (
            "out_half",
            lambda x, y: torch.lerp(
                x, y, 0.3, out=torch.empty(0, dtype=torch.half, device=x.device)
            ),
            (a, b),
        ),
        (
            "out_f64_t",
            lambda x, y, z: torch.lerp(
                x, y, z, out=torch.empty(0, dtype=torch.float64, device=x.device)
            ),
            (a, b, w),
        ),
        (
            "half_w0d_f32",
            lambda x, y: torch.lerp(x.half(), y.half(), torch.tensor(0.3)),
            (a, b),
        ),
    ]
    bp.check_all(cuda_oracle, cases)


def test_lerp_layouts(cuda_oracle: CudaOracle):
    def scalar(x, y):
        return torch.lerp(x, y, 0.3)

    def tensor(x, y):
        return torch.lerp(x, y, (x - y).abs().clamp(max=1))

    for dtype in (torch.float32, torch.float16):
        _check_layouts(cuda_oracle, _layout_cases(scalar, dtype))
        _check_layouts(cuda_oracle, _layout_cases(tensor, dtype))


# ---------------------------------------------------------------------------
# addcmul / addcdiv
# ---------------------------------------------------------------------------


def _values(dtype: torch.dtype) -> list[object]:
    vals: list[object] = [1, 2, -1, 0, 1.0, 0.3, -2.5, 1e-40, True, False]
    vals += [math.inf, bp.nan_scalar(0x7FF8000000000123), 3e38, 70000.0]
    if dtype.is_complex:
        vals += [1 + 2j, 1j, 1 + 0j]
    else:
        vals += [0.5j, 1 + 0j]
    return vals


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize("op", ["addcmul", "addcdiv"])
def test_addc_tensors(cuda_oracle: CudaOracle, op, dtype):
    fn = getattr(torch, op)
    a, b, c = bp.triples(dtype)
    cases = []

    def with_out(x, y, z, v):
        o = torch.empty(0, dtype=x.dtype, device=x.device)
        return fn(x, y, z, value=v, out=o), o

    for v in _values(dtype):
        cases.append((f"v={v!r}", lambda x, y, z, v=v: fn(x, y, z, value=v), (a, b, c)))
        cases.append(
            (
                f"inplace/v={v!r}",
                lambda x, y, z, v=v: getattr(x, op + "_")(y, z, value=v),
                (a, b, c),
            )
        )
        cases.append(
            (f"out/v={v!r}", lambda x, y, z, v=v: with_out(x, y, z, v), (a, b, c))
        )
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize(
    "dtype",
    bp.FLOATS + (torch.int32, torch.int64, torch.uint8, torch.complex64, torch.bool),
    ids=bp.name,
)
@pytest.mark.parametrize("op", ["addcmul", "addcdiv"])
def test_addc_zero_dim(cuda_oracle: CudaOracle, op, dtype):
    """0-d operands: a CPU tensor2 is CUDA's scalar_tensor2 kernel (addcmul),
    a CPU self / tensor1 raises for addcmul; device 0-d tensors are
    ordinary operands."""
    fn = getattr(torch, op)
    a, b, c = bp.triples(dtype)
    wide = torch.complex128 if dtype.is_complex else torch.float64
    cases = []
    for s in _scalar_values(dtype)[:12]:
        for sd in dict.fromkeys((dtype, wide if dtype.is_floating_point else dtype)):
            tag = f"{s!r}:{bp.name(sd)}"
            for v in (1, 0.3, 2):
                cases.append(
                    (
                        f"t2-cpu0d={tag}/v={v}",
                        lambda x, y, s=s, sd=sd, v=v: fn(
                            x, y, torch.tensor(s, dtype=sd), value=v
                        ),
                        (a, b),
                    )
                )
            cases.append(
                (
                    f"inplace-t2-cpu0d={tag}",
                    lambda x, y, s=s, sd=sd: getattr(x, op + "_")(
                        y, torch.tensor(s, dtype=sd), value=0.3
                    ),
                    (a, b),
                )
            )
            cases.append(
                (
                    f"t1-cpu0d={tag}",
                    lambda x, z, s=s, sd=sd: fn(x, torch.tensor(s, dtype=sd), z),
                    (a, c),
                )
            )
            cases.append(
                (
                    f"self-cpu0d={tag}",
                    lambda y, z, s=s, sd=sd: fn(torch.tensor(s, dtype=sd), y, z),
                    (b, c),
                )
            )
    cases.extend(
        [
            ("t2-dev0d", lambda x, y, z: fn(x, y, z[3], value=0.3), (a, b, c)),
            ("t1-dev0d", lambda x, y, z: fn(x, y[3], z, value=0.3), (a, b, c)),
            ("self-dev0d", lambda x, y, z: fn(x[3], y, z, value=0.3), (a, b, c)),
        ]
    )
    bp.check_all(cuda_oracle, cases)


_ADDC_MIXED = [
    (torch.float32, torch.float16, torch.float16),
    (torch.float16, torch.float32, torch.float16),
    (torch.float16, torch.bfloat16, torch.float32),
    (torch.float32, torch.float32, torch.float64),
    (torch.int32, torch.float16, torch.float16),
    (torch.int64, torch.int32, torch.int8),
    (torch.float32, torch.int32, torch.int32),
    (torch.float32, torch.complex64, torch.float32),
    (torch.bool, torch.float16, torch.float16),
    (torch.uint8, torch.uint8, torch.int16),
]


@pytest.mark.parametrize(
    "dtypes", _ADDC_MIXED, ids=["-".join(bp.name(d) for d in ds) for ds in _ADDC_MIXED]
)
@pytest.mark.parametrize("op", ["addcmul", "addcdiv"])
def test_addc_mixed_dtypes(cuda_oracle: CudaOracle, op, dtypes):
    fn = getattr(torch, op)
    ops = [bp.triples(d)[k] for k, d in enumerate(dtypes)]
    n = min(len(t) for t in ops)
    a, b, c = (t[:n] for t in ops)

    def with_out(x, y, z, dt):
        o = torch.empty(0, dtype=dt, device=x.device)
        return fn(x, y, z, value=0.3, out=o), o

    cases = [
        ("ttt", lambda x, y, z: fn(x, y, z, value=0.3), (a, b, c)),
        ("ttt_v1", fn, (a, b, c)),
        ("inplace", lambda x, y, z: getattr(x, op + "_")(y, z, value=2), (a, b, c)),
        ("t2-cpu0d", lambda x, y, z: fn(x, y, z[5].cpu(), value=0.3), (a, b, c)),
        ("t2-dev0d", lambda x, y, z: fn(x, y, z[5], value=0.3), (a, b, c)),
    ]
    for dt in (torch.float32, torch.float64, torch.float16, torch.int64):
        cases.append(
            (
                f"out={bp.name(dt)}",
                lambda x, y, z, dt=dt: with_out(x, y, z, dt),
                (a, b, c),
            )
        )
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize("op", ["addcmul", "addcdiv"])
def test_addc_layouts(cuda_oracle: CudaOracle, op):
    fn = getattr(torch, op)

    def base(device, dtype, seed):
        gen = torch.Generator().manual_seed(seed)
        return bp.random_values(dtype, 4096, gen)[: 64 * 48].reshape(64, 48).to(device)

    cases = []
    for dtype in (torch.float32, torch.float16, torch.float64):

        def b3(d, dtype=dtype):
            return base(d, dtype, 1), base(d, dtype, 2), base(d, dtype, 3)

        cases += [
            (
                f"{bp.name(dtype)}/bcast",
                lambda d, b3=b3: fn(b3(d)[0], b3(d)[1][:1], b3(d)[2][:, :1], value=0.3),
            ),
            (
                f"{bp.name(dtype)}/transposed",
                lambda d, b3=b3: fn(
                    b3(d)[0][:48, :48].t(), b3(d)[1][16:], b3(d)[2][:48, :48]
                ),
            ),
            (
                f"{bp.name(dtype)}/stepped",
                lambda d, b3=b3: fn(
                    b3(d)[0][::3, 1::2],
                    b3(d)[1][1::3, ::2],
                    b3(d)[2][::3, ::2],
                    value=2,
                ),
            ),
            (
                f"{bp.name(dtype)}/rank6",
                lambda d, b3=b3: fn(
                    *(t.reshape(2, 2, 2, 2, 3, 64) for t in b3(d)), value=0.3
                ),
            ),
            (
                f"{bp.name(dtype)}/offset",
                lambda d, b3=b3: fn(
                    *(t.flatten()[k + 1 : k + 3001] for k, t in enumerate(b3(d)))
                ),
            ),
            (
                f"{bp.name(dtype)}/inplace_strided",
                lambda d, b3=b3: (
                    lambda x, y, z: (
                        getattr(x[::2], op + "_")(y[::2], z[1::2], value=0.3),
                        x,
                    )[1]
                )(*b3(d)),
            ),
        ]
    _check_layouts(cuda_oracle, cases)


@pytest.mark.xfail(
    strict=True,
    reason="a float16 NaN cast to float64 (the out= store of a float16 result):"
    " CUDA widens it through float, whose cvt.f32.f16 returns the canonical NaN"
    " (0x7fffffffe0000000); the mojo device's cast (CastSpec, the casts group's)"
    " keeps the payload",
)
def test_half_nan_into_float64_out(cuda_oracle: CudaOracle):
    a, b = bp.pairs(torch.float16)

    def with_out(x, y):
        o = torch.empty(0, dtype=torch.float64, device=x.device)
        return torch.hypot(x, y, out=o), o

    cuda_oracle.assert_bitwise_equal(with_out, a, b)
