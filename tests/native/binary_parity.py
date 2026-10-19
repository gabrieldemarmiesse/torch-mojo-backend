"""Operands and checks shared by the binary-group CUDA parity tests
(tests/native/test_*_cuda_parity.py): elementwise ops of two or three
operands, held to stock CUDA bit for bit by the live oracle
(tests/native/cuda_oracle.py).

Inputs are deterministic: for each dtype a set of special values (signed
zeros, infinities, NaNs with payloads and both signs, signaling NaNs,
subnormals, the extremes, small integers and values around them), crossed
with itself, plus random pairs (random bit patterns, a log-uniform magnitude
sweep and small values). Python-scalar operands come from `SCALARS`.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence

import torch

from tests.native.cuda_oracle import CudaOracle

FLOATS = (torch.float32, torch.float64, torch.float16, torch.bfloat16)
INTS = (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
UINTS = (torch.uint16, torch.uint32, torch.uint64)
BOOL = (torch.bool,)
COMPLEX = (torch.complex64, torch.complex128)
ALL = FLOATS + INTS + BOOL + COMPLEX

_BITS = {
    torch.float32: (torch.int32, 32),
    torch.float64: (torch.int64, 64),
    torch.float16: (torch.int16, 16),
    torch.bfloat16: (torch.int16, 16),
}
# quiet / signaling NaNs of both signs, with payloads
_NANS = {
    torch.float32: [0x7FC00000, 0xFFC00000, 0x7FC00001, 0x7F800001, 0xFFBFFFFF, 0x7FFFFFFF],
    torch.float64: [
        0x7FF8000000000000, 0xFFF8000000000000, 0x7FF8000000000001,
        0x7FF0000000000001, 0xFFF7FFFFFFFFFFFF,
    ],
    torch.float16: [0x7E00, 0xFE00, 0x7E01, 0x7C01, 0xFDFF],
    torch.bfloat16: [0x7FC0, 0xFFC0, 0x7FC1, 0x7F81, 0xFFBF],
}  # fmt: skip


def name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def from_bits(bits: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """The float tensor of the given (unsigned) bit patterns."""
    int_dtype, width = _BITS[dtype]
    signed = [b - (1 << width) if b >= 1 << (width - 1) else b for b in bits]
    return torch.tensor(signed, dtype=torch.int64).to(int_dtype).view(dtype)


def float_specials(dtype: torch.dtype) -> torch.Tensor:
    fi = torch.finfo(dtype)
    sub = fi.smallest_normal * fi.eps  # the smallest subnormal
    vals = [
        0.0, 1.0, 0.5, 2.0, 3.0, 0.1, 1e-3, 7.0, 10.0, 100.0, 1 / 3, math.pi,
        1.5, 2.5, 0.75, 4.0, 1e4, 65504.0, 1e5, 1e20, fi.max, fi.tiny,
        fi.tiny * 4, sub, sub * 3, fi.eps, 1 + fi.eps, 1 - fi.eps / 2, math.inf,
    ]  # fmt: skip
    vals += [-v for v in vals]
    t = torch.tensor(vals, dtype=torch.float64).to(dtype)
    return torch.cat([t, from_bits(_NANS[dtype], dtype)])


def int_specials(dtype: torch.dtype) -> torch.Tensor:
    if dtype == torch.bool:
        return torch.tensor([False, True])
    ii = torch.iinfo(dtype)
    vals = [
        0, 1, -1, 2, -2, 3, -3, 5, 7, -7, 8, 15, 16, 31, 32, 33, 63, 64, 65,
        100, -100, 127, 128, 255, 256, ii.max, ii.min, ii.max - 1, ii.min + 1,
        ii.max // 2,
    ]  # fmt: skip
    return torch.tensor(sorted({v for v in vals if ii.min <= v <= ii.max}), dtype=dtype)


def complex_specials(dtype: torch.dtype) -> torch.Tensor:
    """Every pair of a subset of the component specials (NaNs included)."""
    real = torch.float32 if dtype == torch.complex64 else torch.float64
    s = float_specials(real)
    pick = torch.cat([s[:12], s[29:41], s[20:29], s[-4:]])
    return torch.complex(pick.repeat_interleave(len(pick)), pick.repeat(len(pick)))


def specials(dtype: torch.dtype) -> torch.Tensor:
    if dtype.is_complex:
        return complex_specials(dtype)
    if dtype.is_floating_point:
        return float_specials(dtype)
    return int_specials(dtype)


def random_values(dtype: torch.dtype, n: int, gen: torch.Generator) -> torch.Tensor:
    """About `n` values: random bit patterns, log-uniform magnitudes of both
    signs and small values for floats; uniform and small values for ints."""
    if dtype == torch.bool:
        return torch.randint(0, 2, (n,), generator=gen).bool()
    if dtype.is_complex:
        real = torch.float32 if dtype == torch.complex64 else torch.float64
        re, im = random_values(real, n, gen), random_values(real, n, gen)
        return torch.complex(re, im)
    if dtype.is_floating_point:
        int_dtype, width = _BITS[dtype]
        lo, hi = -(1 << (width - 1)), (1 << (width - 1)) - 1
        bits = torch.randint(lo, hi, (n // 2,), generator=gen, dtype=torch.int64)
        fi = torch.finfo(dtype)
        e = torch.empty(n - n // 2, dtype=torch.float64).uniform_(
            math.log(fi.tiny) / 2, math.log(fi.max) / 2, generator=gen
        )
        sign = torch.where(torch.rand(e.shape, generator=gen) < 0.5, -1.0, 1.0)
        small = torch.randn(n // 8, generator=gen, dtype=torch.float64) * 4
        return torch.cat(
            [
                bits.to(int_dtype).view(dtype),
                (e.exp() * sign).to(dtype),
                small.to(dtype),
            ]
        )
    small = torch.randint(-40, 40, (n // 4,), generator=gen)
    if dtype in (torch.int64, torch.uint64):
        wide = torch.randint(-(1 << 62), 1 << 62, (n,), generator=gen) * 2
        return torch.cat([wide, small]).to(dtype)
    ii = torch.iinfo(dtype)
    wide = torch.randint(ii.min, ii.max + 1, (n,), generator=gen, dtype=torch.int64)
    return torch.cat([wide, small]).to(dtype)


@functools.cache
def pairs(
    dtype: torch.dtype, n: int = 4096, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """(a, b): every pair of specials, then random pairs."""
    gen = torch.Generator().manual_seed(seed)
    s = specials(dtype)
    if dtype.is_complex:
        s = s[torch.randperm(len(s), generator=gen)[:200]]
    ra, rb = random_values(dtype, n, gen), random_values(dtype, n, gen)
    m = min(len(ra), len(rb))
    a = torch.cat([s.repeat_interleave(len(s)), ra[:m]])
    b = torch.cat([s.repeat(len(s)), rb[:m]])
    if dtype == torch.float64:
        keep = ~two_distinct_nans(a, b)
        a, b = a[keep], b[keep]
    return a, b


def two_distinct_nans(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Where `a` and `b` are both NaN and differ once quieted. Stock CUDA's
    float64 arithmetic keeps one of the two payloads by the operand slot
    ptxas gave each in that instruction, which changes with the launch: its
    mul is `DMUL a, b` in the vectorized body and `DMUL b, a` in the tail of
    the same kernel, so the same pair gives b's NaN at index 0 of a 4096-long
    tensor and a's in a 36-long one (test_binary_cuda_parity.py's
    test_float64_two_nans_follow_cuda_launch). The parity tests leave these
    pairs out."""
    quiet = 1 << 51
    abits = a.view(torch.int64) | quiet
    bbits = b.view(torch.int64) | quiet
    return a.isnan() & b.isnan() & (abits != bbits)


@functools.cache
def triples(
    dtype: torch.dtype, n: int = 4096, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(a, b, c): every triple of (up to 30) specials, then random triples."""
    gen = torch.Generator().manual_seed(seed)
    s = specials(dtype)
    keep = 24 if dtype.is_complex else 30
    if len(s) > keep:
        s = s[torch.randperm(len(s), generator=gen)[:keep]]
    k = len(s)
    r = [random_values(dtype, n, gen) for _ in range(3)]
    m = min(len(x) for x in r)
    a = torch.cat([s.repeat_interleave(k * k), r[0][:m]])
    b = torch.cat([s.repeat_interleave(k).repeat(k), r[1][:m]])
    c = torch.cat([s.repeat(k * k), r[2][:m]])
    if dtype == torch.float64:
        keep = ~(
            two_distinct_nans(a, b) | two_distinct_nans(a, c) | two_distinct_nans(b, c)
        )
        a, b, c = a[keep], b[keep], c[keep]
    return a, b, c


def nan_scalar(bits: int) -> float:
    """A Python float with the given float64 bit pattern (a NaN payload)."""
    return from_bits([bits], torch.float64).item()


# Python-number operands: ints, floats (exact, inexact in every float dtype,
# subnormal in float32, overflowing in the half types), bools, infinities, a
# NaN with a payload, and complex numbers.
FLOAT_SCALARS = [
    2.0, 0.5, -1.0, 0.1, 3, 0, -2, True, False, 1e-40, 1e30, 70000.0, 2.5,
    -0.0, math.inf, -math.inf, nan_scalar(0x7FF8000000000123),
]  # fmt: skip
INT_SCALARS = [2, 3, -1, 0, 7, 300, -129, True, 2.5, -0.0]
COMPLEX_SCALARS = [1j, 2 + 3j, -0.5 + 0.0j]


def check_all(
    oracle: CudaOracle,
    cases: Sequence[tuple[str, Callable[..., object], Sequence[object]]],
    **options: object,
):
    """Run every `(label, fn, args)` through `assert_bitwise_equal` and fail
    once at the end with all the mismatches, each under its label."""
    failures = []
    for label, fn, args in cases:
        try:
            oracle.assert_bitwise_equal(fn, *args, **options)  # ty: ignore[invalid-argument-type]
        except AssertionError as error:
            failures.append(f"{label}: {error}")
    assert not failures, f"{len(failures)} of {len(cases)} cases differ:\n" + "\n".join(
        failures
    )


def binary_cases(
    fn: Callable[..., torch.Tensor],
    dtype: torch.dtype,
    *,
    inplace: str | None = None,
    out: bool = True,
) -> list[tuple[str, Callable[..., object], Sequence[object]]]:
    """Tensor-tensor cases of `fn` over `pairs(dtype)`: functional, `out=`
    (an empty out of the result's dtype, resized by the op) and the method
    named `inplace`."""
    a, b = pairs(dtype)
    cases: list[tuple[str, Callable[..., object], Sequence[object]]] = [
        ("tt", fn, (a, b))
    ]
    if out:

        def with_out(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            o = torch.empty(0, dtype=fn(x, y).dtype, device=x.device)
            return fn(x, y, out=o), o

        cases.append(("out", with_out, (a, b)))
    if inplace is not None:
        cases.append(("inplace", lambda x, y: getattr(x, inplace)(y), (a, b)))
    return cases


def scalar_cases(
    fn: Callable[..., torch.Tensor],
    dtype: torch.dtype,
    scalars: Sequence[object],
    *,
    inplace: str | None = None,
    reverse: Sequence[object] = (),
    zero_dim: bool = True,
) -> list[tuple[str, Callable[..., object], Sequence[object]]]:
    """Tensor-with-number cases: `fn(t, s)`, the 0-d CPU tensor `torch.tensor(s)`
    in its place, the in-place method, and `fn(s, t)` for `reverse`."""
    a, b = pairs(dtype)
    cases: list[tuple[str, Callable[..., object], Sequence[object]]] = []
    for s in scalars:
        cases.append((f"ts={s!r}", lambda x, s=s: fn(x, s), (a,)))
        if zero_dim:
            cases.append((f"t0d={s!r}", fn, (a, torch.tensor(s))))
        if inplace is not None:
            cases.append(
                (f"inplace_s={s!r}", lambda x, s=s: getattr(x, inplace)(s), (a,))
            )
    for s in reverse:
        cases.append((f"st={s!r}", lambda y, s=s: fn(s, y), (b,)))
        if zero_dim:
            cases.append((f"0dt={s!r}", fn, (torch.tensor(s), b)))
    return cases
