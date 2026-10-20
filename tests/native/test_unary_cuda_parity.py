"""The basic unary ops and dtype conversions against stock CUDA, bit for bit
(the live oracle, tests/native/cuda_oracle.py): every dtype, the functional,
`out=` and in-place variants, strided and broadcast operands, and the
exception type wherever CUDA raises."""

from __future__ import annotations

import functools
import math
import struct
from collections.abc import Callable

import pytest
import torch

from tests.native.cuda_oracle import CudaOracle

aten = torch.ops.aten

_FLOAT8 = [
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
]
_INTS = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
_UINTS = [torch.uint16, torch.uint32, torch.uint64]
_FLOATS = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
# complex32 is left out: the mojo device holds no complex32 tensor at all
# (test_complex32_tensors_unsupported).
_COMPLEX = [torch.complex64, torch.complex128]
_DTYPES = [torch.bool, *_INTS, *_UINTS, *_FLOATS, *_COMPLEX, *_FLOAT8]


def _dtype_id(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


# Open mismatches on NaN bits (strict: a fix turns the entry into a failure
# that asks for its removal from this table).
_NAN_SIGN_REASON = (
    "NaN sign or payload of a complex result differs from stock CUDA: the "
    "double-precision exp/log/expm1/sgn and complex64 negation paths are not "
    "bit-exact on NaN yet"
)
_NAN_SIGN_XFAIL: dict[str, frozenset[tuple[str, str]]] = {
    "test_unary": frozenset(
        {
            ("conj_physical", "complex64"),
            ("neg", "complex64"),
            ("angle", "complex128"),
            ("exp", "complex128"),
            ("exp2", "complex128"),
            ("expm1", "complex128"),
            ("log", "complex128"),
            ("log2", "complex128"),
            ("log10", "complex128"),
            ("log1p", "complex128"),
            ("sgn", "complex128"),
        }
    ),
    "test_unary_out": frozenset(
        {
            ("conj_physical", "complex64"),
            ("neg", "complex64"),
            ("angle", "complex128"),
            ("exp", "complex128"),
            ("exp2", "complex128"),
            ("expm1", "complex128"),
            ("log", "complex128"),
            ("log2", "complex128"),
            ("log10", "complex128"),
            ("log1p", "complex128"),
            ("sgn", "complex128"),
        }
    ),
    "test_unary_inplace": frozenset(
        {
            ("conj_physical", "complex64"),
            ("neg", "complex64"),
            ("exp", "complex128"),
            ("exp2", "complex128"),
            ("expm1", "complex128"),
            ("log", "complex128"),
            ("log2", "complex128"),
            ("log10", "complex128"),
            ("log1p", "complex128"),
            ("sgn", "complex128"),
        }
    ),
    "test_unary_layouts": frozenset({("neg", "complex64")}),
}


def _xfail_nan_sign(
    request: pytest.FixtureRequest, test: str, op: str, dtype: torch.dtype
) -> None:
    if (op, _dtype_id(dtype)) in _NAN_SIGN_XFAIL.get(test, frozenset()):
        request.applymarker(pytest.mark.xfail(strict=True, reason=_NAN_SIGN_REASON))


def _hashed(count: int, seed: int) -> list[int]:
    """`count` pseudo-random 64-bit words (an LCG: reproducible anywhere)."""
    h = 0x9E3779B97F4A7C15 ^ seed
    out = []
    for _ in range(count):
        h = (h * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        out.append(h)
    return out


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def _f64_bits(v: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", v))[0]


# Where the CUDA routines switch algorithm or the result changes regime:
# 1 and its neighbours, the overflow / underflow thresholds of exp, exp2,
# expm1, sinh, cosh (float and double), pi / 2 multiples for the trig
# range reductions, tanh's and asin's polynomial switches, 2**23 / 2**52.
_CENTERS = [
    0.5,
    1.0,
    2.0,
    0.125,
    0.5625,
    0.625,
    0.6,
    0.55,
    0.9,
    0.98,
    1e-4,
    math.pi / 4,
    math.pi / 2,
    math.pi,
    3 * math.pi / 2,
    2 * math.pi,
    105615.0,
    48039.0,
    88.72283935546875,
    89.41598510742188,
    -87.33654022216797,
    -103.97207641601562,
    126.0,
    127.0,
    128.0,
    -126.0,
    -149.0,
    -150.0,
    38.0,
    44.0,
    9.0,
    20.0,
    709.782712893384,
    710.4758600739439,
    -708.3964185322641,
    -745.1332191019411,
    1023.0,
    1024.0,
    -1022.0,
    -1074.0,
    -1075.0,
    308.0,
    0.6931471805599453,
    2.0**23,
    2.0**24,
    2.0**31,
    2.0**52,
    2.0**53,
    2.0**63,
    2.0**64,
]


def _float32_bits() -> list[int]:
    """Non-negative float32 bit patterns (each also run negated)."""
    bits = {0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0x7FA00000, 0x7FC12345, 0x7F800000}
    for e in range(256):
        for m in (0, 1, 2, 0x200000, 0x400000, 0x400001, 0x600000, 0x7FFFFE, 0x7FFFFF):
            bits.add((e << 23) | m)
    for c in _CENTERS:
        b = _f32_bits(c)
        bits.update(range(b - 48, b + 49))
    # log-uniform magnitudes over the whole range, subnormals included
    for k in range(4096):
        bits.add(_f32_bits(10.0 ** (-45.5 + 84 * k / 4096)) & 0x7FFFFFFF)
    # integers and half-integers (rounding ties), and hashed patterns
    for k in range(-600, 601):
        bits.add(_f32_bits(abs(k) * 0.5))
    bits.update((h >> 33) for h in _hashed(12288, 1))
    return sorted(b for b in bits if b < (1 << 31))


def _float64_bits() -> list[int]:
    bits = {
        0x7FF0000000000001,
        0x7FF8000000000000,
        0x7FFFFFFFFFFFFFFF,
        0x7FF4000000000000,
        0x7FF8000012345678,
        0x7FF0000000000000,
    }
    for e in range(2048):
        for m in (0, 1, 1 << 51, (1 << 52) - 1, 0x8000000001):
            bits.add((e << 52) | m)
    for c in _CENTERS:
        b = _f64_bits(c)
        bits.update(range(b - 32, b + 33))
        b = _f64_bits(float(struct.unpack("<f", struct.pack("<f", c))[0]))
        bits.update(range(b - 4, b + 5))
    for k in range(4096):
        bits.add(_f64_bits(10.0 ** (-323.5 + 631.5 * k / 4096)))
    for k in range(-600, 601):
        bits.add(_f64_bits(abs(k) * 0.5))
    bits.update((h >> 1) for h in _hashed(8192, 2))
    return sorted(b for b in bits if b < (1 << 63))


def _signed(bits: list[int], width: int) -> list[int]:
    """`bits` and their negations, as the signed integers of a width."""
    sign = 1 << (width - 1)
    return bits + [b - sign for b in bits]


def _real_values(dtype: torch.dtype) -> torch.Tensor:
    """Input values of a real dtype (every pattern of a 1- or 2-byte float)."""
    if dtype == torch.bool:
        return torch.tensor([False, True, True, False])
    if dtype in _FLOAT8:
        return torch.arange(256, dtype=torch.int32).to(torch.uint8).view(dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return (
            torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32)
            .to(torch.int16)
            .view(dtype)
        )
    if dtype == torch.float32:
        signed = _signed(_float32_bits(), 32)
        return torch.tensor(signed, dtype=torch.int32).view(torch.float32)
    if dtype == torch.float64:
        signed = _signed(_float64_bits(), 64)
        return torch.tensor(signed, dtype=torch.int64).view(torch.float64)
    info = torch.iinfo(dtype)
    vals = {0, 1, 2, 3, 7, info.max, info.max - 1, info.min, info.min + 1}
    for k in range(info.bits):
        for d in (-1, 0, 1):
            vals.update((2**k + d, -(2**k) + d))
    # around the float32 / float64 integer limits
    for k in (24, 25, 31, 32, 53, 54, 63):
        vals.update(2**k + d for d in range(-3, 4))
        vals.update(-(2**k) + d for d in range(-3, 4))
    span = info.max - info.min + 1
    vals.update(info.min + h % span for h in _hashed(4096, info.bits))
    vals = sorted(v for v in vals if info.min <= v <= info.max)
    if dtype == torch.uint64:
        # no CPU tensor constructor for values >= 2**63: build from int64 bits
        return torch.tensor([v - (1 << 64) if v >= 1 << 63 else v for v in vals]).view(
            torch.uint64
        )
    return torch.tensor(vals, dtype=dtype)


def _complex_values(dtype: torch.dtype) -> torch.Tensor:
    part = torch.float32 if dtype == torch.complex64 else torch.float64
    specials = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -3.0, 1e-3, math.pi / 2]
    specials += [math.inf, -math.inf, math.nan, 10.0, -100.0, 40.0, 1e30, -1e-30]
    finfo = torch.finfo(part)
    specials += [finfo.max, -finfo.max, finfo.tiny, -finfo.tiny, finfo.eps]
    grid = torch.tensor(specials, dtype=torch.float64).to(part)
    re, im = torch.meshgrid(grid, grid, indexing="ij")
    real = _real_values(part)
    count = real.numel() // 2 * 2
    rnd = real[:count].reshape(2, -1)
    flipped = torch.stack((rnd[0].flip(0), rnd[1]))
    pairs = torch.cat(
        (torch.stack((re.flatten(), im.flatten())), rnd, flipped), dim=1
    ).T.contiguous()
    return torch.view_as_complex(pairs)


@functools.cache
def _cached_values(dtype: torch.dtype) -> torch.Tensor:
    if dtype in _COMPLEX:
        return _complex_values(dtype)
    return _real_values(dtype)


def _values(dtype: torch.dtype) -> torch.Tensor:
    return _cached_values(dtype).clone()


# ---------------------------------------------------------------------------
# The unary ops
# ---------------------------------------------------------------------------

_UNARY = [
    "abs",
    "acos",
    "acosh",
    "angle",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "bitwise_not",
    "ceil",
    "conj_physical",
    "cos",
    "cosh",
    "deg2rad",
    "exp",
    "exp2",
    "expm1",
    "fix",
    "floor",
    "frac",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "log",
    "log10",
    "log1p",
    "log2",
    "logical_not",
    "nan_to_num",
    "neg",
    "rad2deg",
    "reciprocal",
    "round",
    "rsqrt",
    "sgn",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "trunc",
]


def _packet(name: str, overload: str) -> Callable[..., object] | None:
    packet = getattr(aten, name, None)
    if packet is None or overload not in packet.overloads():
        return None
    return getattr(packet, overload)


def _functional(name: str) -> Callable[..., object]:
    fn = _packet(name, "default")
    assert fn is not None
    return fn


@pytest.mark.parametrize("dtype", _DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("op", _UNARY)
def test_unary(
    request: pytest.FixtureRequest, cuda_oracle: CudaOracle, op: str, dtype: torch.dtype
):
    _xfail_nan_sign(request, "test_unary", op, dtype)
    cuda_oracle.assert_bitwise_equal(_functional(op), _values(dtype))


def _result_dtype(cuda_oracle: CudaOracle, op: str, dtype: torch.dtype) -> torch.dtype:
    """The dtype CUDA's functional op returns for `dtype`, or a skip when it
    raises (the functional test covers that)."""
    fn = _functional(op)
    try:
        out = fn(torch.zeros(1, dtype=dtype, device=cuda_oracle.cuda))
    except (RuntimeError, NotImplementedError, TypeError):
        pytest.skip("CUDA's functional op raises for this dtype")
    assert isinstance(out, torch.Tensor)
    return out.dtype


@pytest.mark.parametrize("dtype", _DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("op", _UNARY)
def test_unary_out(
    request: pytest.FixtureRequest, cuda_oracle: CudaOracle, op: str, dtype: torch.dtype
):
    """`out=` of the result dtype and the input's shape, then an empty one of
    a wider dtype (resized; cast to, or refused)."""
    _xfail_nan_sign(request, "test_unary_out", op, dtype)
    fn = _packet(op, "out")
    if fn is None:
        pytest.skip(f"aten::{op} has no out overload")
    x = _values(dtype)
    result = _result_dtype(cuda_oracle, op, dtype)
    out = torch.zeros(x.shape, dtype=result)
    cuda_oracle.assert_bitwise_equal(lambda a, o: fn(a, out=o), x, out)
    wide = torch.complex128 if result.is_complex else torch.float64
    cuda_oracle.assert_bitwise_equal(
        lambda a, o: fn(a, out=o), x, torch.zeros(0, dtype=wide)
    )


@pytest.mark.parametrize("dtype", _DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("op", _UNARY)
def test_unary_inplace(
    request: pytest.FixtureRequest, cuda_oracle: CudaOracle, op: str, dtype: torch.dtype
):
    _xfail_nan_sign(request, "test_unary_inplace", op, dtype)
    fn = _packet(op + "_", "default")
    if fn is None:
        pytest.skip(f"aten::{op} has no in-place variant")
    cuda_oracle.assert_bitwise_equal(fn, _values(dtype))


_LAYOUT_DTYPES = [torch.float32, torch.bfloat16, torch.int32, torch.complex64]


@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("op", _UNARY)
def test_unary_layouts(
    request: pytest.FixtureRequest, cuda_oracle: CudaOracle, op: str, dtype: torch.dtype
):
    """Strided, transposed, expanded, 0-dim and empty inputs, and a strided
    `out=`."""
    _xfail_nan_sign(request, "test_unary_layouts", op, dtype)
    values = _values(dtype)[: 37 * 41 * 2]
    fn = _functional(op)

    def views(device: str) -> list[torch.Tensor]:
        x = values.to(device)
        grid = x.reshape(37, 82)
        return [
            grid[:, ::2],
            grid.T,
            grid[3:9, 5].unsqueeze(1).expand(6, 11),
            x[7].reshape(()),
            x[:0],
        ]

    cuda_oracle.assert_bitwise_equal_by_device(
        lambda d: [fn(v) for v in views(d)], what=f"{op} views"
    )
    out_fn = _packet(op, "out")
    if out_fn is None:
        return
    result = _result_dtype(cuda_oracle, op, dtype)

    def strided_out(device: str) -> torch.Tensor:
        src = values.to(device).reshape(37, 82)[:, 1::2]
        base = torch.zeros(41, 74, dtype=result, device=device)
        out_fn(src, out=base.T[:, ::2])
        return base

    cuda_oracle.assert_bitwise_equal_by_device(strided_out, what=f"{op} strided out")


_ROUND_FLOATS = [*_FLOATS, torch.int32, torch.int64, torch.uint8, torch.complex64]


@pytest.mark.parametrize("decimals", [-3, -1, 1, 2, 4, 7, 20])
@pytest.mark.parametrize("dtype", _ROUND_FLOATS, ids=_dtype_id)
def test_round_decimals(cuda_oracle: CudaOracle, dtype: torch.dtype, decimals: int):
    x = _values(dtype)
    cuda_oracle.assert_bitwise_equal(lambda a: torch.round(a, decimals=decimals), x)
    cuda_oracle.assert_bitwise_equal(
        lambda a, o: torch.round(a, decimals=decimals, out=o),
        x,
        torch.zeros(x.shape, dtype=dtype),
    )
    cuda_oracle.assert_bitwise_equal(lambda a: a.round_(decimals=decimals), x)


@pytest.mark.parametrize(
    ("nan", "posinf", "neginf"),
    [(1.5, 2.0, -3.0), (-0.0, None, None), (None, 1e300, -1e300), (7, 70000.0, None)],
    ids=["finite", "neg_zero", "huge", "big"],
)
@pytest.mark.parametrize(
    "dtype", [*_INTS, torch.bool, *_FLOATS, *_COMPLEX], ids=_dtype_id
)
def test_nan_to_num_args(
    cuda_oracle: CudaOracle,
    dtype: torch.dtype,
    nan: float | None,
    posinf: float | None,
    neginf: float | None,
):
    x = _values(dtype)
    cuda_oracle.assert_bitwise_equal(
        lambda a: torch.nan_to_num(a, nan, posinf, neginf), x
    )
    cuda_oracle.assert_bitwise_equal(
        lambda a, o: torch.nan_to_num(a, nan, posinf, neginf, out=o),
        x,
        torch.zeros(x.shape, dtype=dtype),
    )
    cuda_oracle.assert_bitwise_equal(lambda a: a.nan_to_num_(nan, posinf, neginf), x)


@pytest.mark.parametrize("dtype", _DTYPES, ids=_dtype_id)
def test_frexp(cuda_oracle: CudaOracle, dtype: torch.dtype):
    x = _values(dtype)
    cuda_oracle.assert_bitwise_equal(torch.frexp, x)
    cuda_oracle.assert_bitwise_equal(
        lambda a, m, e: torch.frexp(a, out=(m, e)),
        x,
        torch.zeros(x.shape, dtype=dtype),
        torch.zeros(x.shape, dtype=torch.int32),
    )


# ---------------------------------------------------------------------------
# dtype conversion: _to_copy and copy_
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dst", _DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("src", _DTYPES, ids=_dtype_id)
def test_to_copy(cuda_oracle: CudaOracle, src: torch.dtype, dst: torch.dtype):
    x = _values(src)
    cuda_oracle.assert_bitwise_equal(lambda a: a.to(dst), x)


@pytest.mark.parametrize("dst", _DTYPES, ids=_dtype_id)
@pytest.mark.parametrize("src", _DTYPES, ids=_dtype_id)
def test_copy_(cuda_oracle: CudaOracle, src: torch.dtype, dst: torch.dtype):
    """Into a contiguous destination, then into a strided one from a
    transposed source, then broadcast from one element."""
    x = _values(src)
    n = x.numel() // 2 * 2
    cuda_oracle.assert_bitwise_equal(
        lambda a, o: o.copy_(a), x, torch.zeros(x.shape, dtype=dst)
    )

    def strided(device: str) -> torch.Tensor:
        a = x[:n].to(device).reshape(2, -1).T
        base = torch.zeros(3, n // 2, 2, dtype=dst, device=device)
        base[1].copy_(a)
        return base

    cuda_oracle.assert_bitwise_equal_by_device(strided, what="strided copy_")

    def broadcast(device: str) -> torch.Tensor:
        return torch.zeros(5, 3, dtype=dst, device=device).copy_(x[n // 3].to(device))

    cuda_oracle.assert_bitwise_equal_by_device(broadcast, what="broadcast copy_")


@pytest.mark.xfail(
    strict=True,
    reason="complex32 (ComplexHalf) tensors cannot be created on the mojo"
    " device; stock CUDA runs these ops on them (jiterated in complex<float>)",
)
def test_complex32_tensors_unsupported(cuda_oracle: CudaOracle):
    x = torch.tensor([1 + 2j, -0.5j]).to(torch.complex32)
    cuda_oracle.assert_bitwise_equal(lambda a: (a.exp(), a.to(torch.complex64)), x)
