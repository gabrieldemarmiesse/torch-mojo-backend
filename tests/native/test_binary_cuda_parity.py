"""add / sub / rsub / mul on the mojo device against
stock CUDA, bit for bit (tests/native/cuda_oracle.py): every dtype, `out=`,
in place, Python-number and 0-d CPU operands, alpha, mixed dtypes,
broadcasting and strided operands. Operands: tests/native/binary_parity.py.
"""

import pytest
import torch

from tests.native import binary_parity as bp
from tests.native.cuda_oracle import CudaOracle

_ARITH = [
    ("add", torch.add, "add_"),
    ("sub", torch.sub, "sub_"),
    ("mul", torch.mul, "mul_"),
    ("rsub", torch.rsub, None),
]
_DTYPES = bp.ALL + bp.UINTS


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize(("op", "fn", "inplace"), _ARITH, ids=[a[0] for a in _ARITH])
def test_tensor_tensor(cuda_oracle: CudaOracle, op, fn, inplace, dtype):
    out = op != "rsub"
    bp.check_all(cuda_oracle, bp.binary_cases(fn, dtype, inplace=inplace, out=out))


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize(("op", "fn", "inplace"), _ARITH, ids=[a[0] for a in _ARITH])
def test_tensor_scalar(cuda_oracle: CudaOracle, op, fn, inplace, dtype):
    scalars = bp.FLOAT_SCALARS
    if dtype != torch.float16:  # complex32: test_complex32_is_not_held
        scalars = scalars + bp.COMPLEX_SCALARS
    reverse = [2.0, 3, -0.5, bp.nan_scalar(0x7FF8000000000123)] if op != "rsub" else []
    bp.check_all(
        cuda_oracle,
        bp.scalar_cases(fn, dtype, scalars, inplace=inplace, reverse=reverse),
    )


_ALPHAS = [2, -1, 0, 3, 0.3, -2.5, 1e-40, 1e300, float("nan"), True, 1.0]


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize("op", ["add", "sub", "rsub"])
def test_alpha(cuda_oracle: CudaOracle, op, dtype):
    fn = getattr(torch, op)
    a, b = bp.pairs(dtype)
    cases = []
    alphas = _ALPHAS + ([1 + 2j] if dtype.is_complex else [])
    for alpha in alphas:
        cases.append(
            (f"tt/alpha={alpha!r}", lambda x, y, al=alpha: fn(x, y, alpha=al), (a, b))
        )
        for s in (0.1, 3, -2.5):
            cases.append(
                (
                    f"ts={s}/alpha={alpha!r}",
                    lambda x, s=s, al=alpha: fn(x, s, alpha=al),
                    (a,),
                )
            )
            cases.append(
                (
                    f"t0d={s}/alpha={alpha!r}",
                    lambda x, s=s, al=alpha: fn(x, torch.tensor(s), alpha=al),
                    (a,),
                )
            )
        if op != "rsub":
            inplace = getattr(torch.Tensor, op + "_")
            cases.append(
                (
                    f"inplace/alpha={alpha!r}",
                    lambda x, y, al=alpha: inplace(x, y, alpha=al),
                    (a, b),
                )
            )
            cases.append(
                (
                    f"inplace_s/alpha={alpha!r}",
                    lambda x, al=alpha: inplace(x, 0.1, alpha=al),
                    (a,),
                )
            )
    bp.check_all(cuda_oracle, cases)


_MIXED = [
    (torch.float16, torch.float32), (torch.bfloat16, torch.float16),
    (torch.int32, torch.float16), (torch.int64, torch.float32),
    (torch.uint8, torch.int8), (torch.bool, torch.int32),
    (torch.float32, torch.float64), (torch.int8, torch.bfloat16),
    (torch.float32, torch.complex64), (torch.float64, torch.complex64),
    (torch.bfloat16, torch.complex128), (torch.bool, torch.float16),
]  # fmt: skip


@pytest.mark.parametrize(
    ("da", "db"), _MIXED, ids=[f"{bp.name(a)}-{bp.name(b)}" for a, b in _MIXED]
)
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_mixed_dtypes(cuda_oracle: CudaOracle, op, da, db):
    """Type promotion: two tensors, and each one as a 0-d tensor (device and
    CPU), which only counts when its category is higher."""
    fn = getattr(torch, op)
    a, _ = bp.pairs(da)
    _, b = bp.pairs(db)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    wide = torch.promote_types(da, db)
    if wide in (torch.float64, torch.complex128):
        real = torch.float64
        both = torch.zeros(n, dtype=torch.bool)
        for pa in (
            (a.to(wide),) if not wide.is_complex else (a.to(wide).real, a.to(wide).imag)
        ):
            for pb in (
                (b.to(wide),)
                if not wide.is_complex
                else (b.to(wide).real, b.to(wide).imag)
            ):
                both |= bp.two_distinct_nans(pa.to(real), pb.to(real))
        a, b = a[~both], b[~both]
    bp.check_all(
        cuda_oracle,
        [
            ("tt", fn, (a, b)),
            ("t-dev0d", lambda x, y: fn(x, y[7]), (a, b)),
            ("dev0d-t", lambda x, y: fn(x[7], y), (a, b)),
            ("t-cpu0d", lambda x, y: fn(x, y[7].cpu()), (a, b)),
            ("cpu0d-t", lambda x, y: fn(x[7].cpu(), y), (a, b)),
        ],
    )


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
    ]


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.float64, torch.int32, torch.complex64],
    ids=bp.name,
)
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_layouts(cuda_oracle: CudaOracle, op, dtype):
    fn = getattr(torch, op)
    failures = []
    for label, f in _layout_cases(fn, dtype):
        try:
            cuda_oracle.assert_bitwise_equal_by_device(f)
        except AssertionError as error:
            failures.append(f"{label}: {error}")
    assert not failures, "\n".join(failures)


@pytest.mark.xfail(
    strict=True,
    reason="stock CUDA keeps one of two float64 NaN payloads by launch"
    " (binary_parity.two_distinct_nans): no fixed rule matches both arms",
)
def test_float64_two_nans_follow_cuda_launch(cuda_oracle: CudaOracle):
    nans = bp.from_bits([0x7FF8000000000000, 0xFFF8000000000000], torch.float64)
    # 4096 pairs fill the vectorized body, the 36 after them its tail
    a = nans[:1].repeat(4096 + 36)
    b = nans[1:].repeat(4096 + 36)
    cuda_oracle.assert_bitwise_equal(torch.mul, a, b)


def test_cuda_float64_two_nans_depend_on_the_launch(cuda_oracle: CudaOracle):
    """The premise of the xfail above, on stock CUDA alone: the same NaN pair
    multiplies to b's NaN in the vectorized body and to a's in the tail."""
    nans = bp.from_bits([0x7FF8000000000000, 0xFFF8000000000000], torch.float64)
    a = nans[:1].repeat(4096 + 36).to(cuda_oracle.cuda)
    b = nans[1:].repeat(4096 + 36).to(cuda_oracle.cuda)
    bits = torch.mul(a, b).cpu().view(torch.int64)
    assert bits[0] == nans[1:].view(torch.int64)[0]
    assert bits[-1] == nans[:1].view(torch.int64)[0]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="complex32 (float16 with a complex operand) is not held by the mojo"
    " device (tmb/ops/complex.mojo): it declines where CUDA computes",
)
@pytest.mark.parametrize("op", ["add", "sub", "mul"])
def test_complex32_is_not_held(cuda_oracle: CudaOracle, op):
    a, _ = bp.pairs(torch.float16)
    cuda_oracle.assert_bitwise_equal(lambda x: getattr(torch, op)(x, 1j), a)
