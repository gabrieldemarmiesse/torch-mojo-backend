"""div (true, trunc, floor), floor_divide, remainder and fmod on the mojo
device against stock CUDA, bit for bit (tests/native/cuda_oracle.py): every
dtype, `out=`, in place, Python-number and 0-d CPU operands on either side,
mixed dtypes, broadcasting and strided operands, integer division by zero
and error behaviour. Operands: tests/native/binary_parity.py.
"""

import functools

import pytest
import torch

from tests.native import binary_parity as bp
from tests.native.cuda_oracle import CudaOracle


def _div_mode(mode):
    def fn(x, y, **kw):
        return torch.div(x, y, rounding_mode=mode, **kw)

    def inplace(x, y):
        return x.div_(y, rounding_mode=mode)

    return fn, inplace


_TRUNC, _TRUNC_ = _div_mode("trunc")
_FLOOR, _FLOOR_ = _div_mode("floor")

# (name, function, in-place method, `function(number, tensor)` exists)
_OPS = [
    ("div", torch.div, torch.Tensor.div_, True),
    ("div_trunc", _TRUNC, _TRUNC_, True),
    ("div_floor", _FLOOR, _FLOOR_, True),
    ("floor_divide", torch.floor_divide, torch.Tensor.floor_divide_, True),
    ("remainder", torch.remainder, torch.Tensor.remainder_, True),
    ("fmod", torch.fmod, torch.Tensor.fmod_, False),
]
_IDS = [o[0] for o in _OPS]
_DTYPES = bp.FLOATS + bp.INTS + bp.BOOL + bp.UINTS


def _tt_cases(fn, inplace, dtype):
    a, b = bp.pairs(dtype)

    def with_out(x, y):
        o = torch.empty(0, dtype=fn(x, y).dtype, device=x.device)
        return fn(x, y, out=o), o

    return [
        ("tt", fn, (a, b)),
        ("out", with_out, (a, b)),
        ("inplace", inplace, (a, b)),
        # the divisor as a one-element device tensor (no CPU-scalar path)
        ("t-dev1", lambda x, y: fn(x, y[7:8]), (a, b)),
        ("t-dev0d", lambda x, y: fn(x, y[7]), (a, b)),
    ]


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize(("op", "fn", "inplace", "rev"), _OPS, ids=_IDS)
def test_tensor_tensor(cuda_oracle: CudaOracle, op, fn, inplace, rev, dtype):
    bp.check_all(cuda_oracle, _tt_cases(fn, inplace, dtype))


_INT_SCALARS = bp.INT_SCALARS + [-2, 5, -7, 1, 128, -32769, 1 << 40, -(1 << 33)]
_FLOAT_SCALARS = bp.FLOAT_SCALARS + [
    3.0, -2.5, 1e-310, 0.3, 1e300, -1e-5, 6e-8, 65504.0, 1 / 3,
    bp.nan_scalar(0xFFF0000000000001),
]  # fmt: skip


def _scalars(dtype):
    if dtype.is_floating_point:
        return _FLOAT_SCALARS
    return _INT_SCALARS + [0.5, -3.0, 1e-40, float("inf"), float("nan")]


@pytest.mark.parametrize("dtype", _DTYPES, ids=bp.name)
@pytest.mark.parametrize(("op", "fn", "inplace", "rev"), _OPS, ids=_IDS)
def test_tensor_scalar(cuda_oracle: CudaOracle, op, fn, inplace, rev, dtype):
    scalars = _scalars(dtype)
    cases = bp.scalar_cases(fn, dtype, scalars, reverse=scalars if rev else ())
    a, _ = bp.pairs(dtype)
    for s in scalars:
        cases.append((f"inplace_s={s!r}", lambda x, s=s: inplace(x, s), (a,)))
        cases.append(
            (f"inplace_0d={s!r}", lambda x, s=s: inplace(x, torch.tensor(s)), (a,))
        )
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize("dtype", [torch.float32, torch.int32], ids=bp.name)
@pytest.mark.parametrize(("op", "fn", "inplace", "rev"), _OPS, ids=_IDS)
def test_python_operators(cuda_oracle: CudaOracle, op, fn, inplace, rev, dtype):
    """`//`, `%` and `/` with a number on either side."""
    if op not in ("div", "floor_divide", "remainder"):
        pytest.skip("no Python operator")
    a, b = bp.pairs(dtype)
    binop = {
        "div": lambda x, y: x / y,
        "floor_divide": lambda x, y: x // y,
        "remainder": lambda x, y: x % y,
    }[op]
    cases = [("tt", binop, (a, b))]
    for s in (3, -2, 0.5, -0.0, 7.25):
        cases.append((f"ts={s}", lambda x, s=s: binop(x, s), (a,)))
        if op != "div":  # `number / tensor` is reciprocal-then-mul
            cases.append((f"st={s}", lambda y, s=s: binop(s, y), (b,)))
    bp.check_all(cuda_oracle, cases)


_MIXED = [
    (torch.float16, torch.float32), (torch.bfloat16, torch.float16),
    (torch.int32, torch.float16), (torch.int64, torch.float32),
    (torch.uint8, torch.int8), (torch.bool, torch.int32),
    (torch.float32, torch.float64), (torch.int8, torch.bfloat16),
    (torch.bool, torch.float16), (torch.int16, torch.int64),
    (torch.uint8, torch.bool), (torch.float64, torch.bfloat16),
]  # fmt: skip


@pytest.mark.parametrize(
    ("da", "db"), _MIXED, ids=[f"{bp.name(a)}-{bp.name(b)}" for a, b in _MIXED]
)
@pytest.mark.parametrize(("op", "fn", "inplace", "rev"), _OPS, ids=_IDS)
def test_mixed_dtypes(cuda_oracle: CudaOracle, op, fn, inplace, rev, da, db):
    """Type promotion: two tensors, and each one as a 0-d tensor (device and
    CPU), which only counts when its category is higher."""
    a, _ = bp.pairs(da)
    _, b = bp.pairs(db)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if torch.result_type(a, b) == torch.float64:
        # promoted, two different NaNs: the payload depends on the launch
        keep = ~bp.two_distinct_nans(a.double(), b.double())
        a, b = a[keep], b[keep]
    cases = [
        ("tt", fn, (a, b)),
        ("t-dev0d", lambda x, y: fn(x, y[7]), (a, b)),
        ("dev0d-t", lambda x, y: fn(x[7], y), (a, b)),
        ("t-cpu0d", lambda x, y: fn(x, y[7].cpu()), (a, b)),
        ("inplace", inplace, (a, b)),
    ]
    if rev:
        cases.append(("cpu0d-t", lambda x, y: fn(x[7].cpu(), y), (a, b)))
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize(
    ("dtype", "out_dtype"),
    [
        (torch.float32, torch.float64),
        (torch.float16, torch.float32),
        (torch.int32, torch.float32),
        (torch.int64, torch.int32),
        (torch.float32, torch.int32),
        (torch.int32, torch.float16),
    ],
    ids=lambda d: bp.name(d),
)
@pytest.mark.parametrize(("op", "fn", "inplace", "rev"), _OPS, ids=_IDS)
def test_out_dtype(cuda_oracle: CudaOracle, op, fn, inplace, rev, dtype, out_dtype):
    """`out=` of another dtype: computed in the common dtype, then cast (or
    refused when the cast would narrow its category)."""
    a, b = bp.pairs(dtype)

    def with_out(x, y, s=None):
        o = torch.empty(0, dtype=out_dtype, device=x.device)
        return fn(x, y if s is None else s, out=o), o

    bp.check_all(
        cuda_oracle,
        [
            ("tt", with_out, (a, b)),
            ("ts", lambda x, y: with_out(x, y, 3), (a, b)),
            ("ts_float", lambda x, y: with_out(x, y, -2.5), (a, b)),
        ],
    )


def _layout_cases(fn, dtype):
    def base(device):
        gen = torch.Generator().manual_seed(3)
        return bp.random_values(dtype, 4096, gen)[: 64 * 48].reshape(64, 48).to(device)

    return [
        ("bcast_row", lambda d: fn(base(d), base(d)[:1, :])),
        ("bcast_col", lambda d: fn(base(d)[:1, :], base(d)[:, 3:4])),
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
        ("strided_scalar", lambda d: fn(base(d)[::2, 1::3], 3)),
        ("strided_scalar_f", lambda d: fn(base(d).t(), -1.5)),
        ("empty", lambda d: fn(base(d)[:0], base(d)[:0])),
    ]


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.int32,
     torch.int64, torch.uint8],
    ids=bp.name,
)  # fmt: skip
@pytest.mark.parametrize(("op", "fn", "inplace", "rev"), _OPS, ids=_IDS)
def test_layouts(cuda_oracle: CudaOracle, op, fn, inplace, rev, dtype):
    failures = []
    for label, f in _layout_cases(fn, dtype):
        try:
            cuda_oracle.assert_bitwise_equal_by_device(f)
        except AssertionError as error:
            failures.append(f"{label}: {error}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "dtype",
    [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64],
    ids=bp.name,
)
@pytest.mark.parametrize(("op", "fn", "inplace", "rev"), _OPS, ids=_IDS)
def test_integer_division_by_zero(cuda_oracle: CudaOracle, op, fn, inplace, rev, dtype):
    """CUDA does not raise: an integer quotient or remainder by zero is what
    the GPU's division sequence returns (-1, or 2**32 - 1 for an int64 whose
    operands fit in 32 bits unsigned), and INT_MIN / -1 wraps."""
    ii = torch.iinfo(dtype)
    vals = [0, 1, -1, 5, -5, 100, ii.max, ii.min, ii.max - 1, ii.min + 1]
    if dtype == torch.int64:
        vals += [(1 << 32) - 1, 1 << 32, 1 << 31, -(1 << 31), 12345678901]
    a = torch.tensor([v for v in vals if ii.min <= v <= ii.max], dtype=dtype)
    z = torch.zeros_like(a)
    m1 = torch.full_like(a, -1) if dtype != torch.uint8 else torch.full_like(a, 255)
    bp.check_all(
        cuda_oracle,
        [
            ("tt0", fn, (a, z)),
            ("ts0", lambda x: fn(x, 0), (a,)),
            ("t0d0", lambda x: fn(x, torch.tensor(0)), (a,)),
            ("tt-1", fn, (a, m1)),
            ("ts-1", lambda x: fn(x, -1), (a,)),
            ("inplace0", lambda x: inplace(x, 0), (a,)),
        ]
        + ([("st0", lambda y: fn(0, y), (z,))] if rev else []),
    )


@pytest.mark.parametrize(("op", "fn", "inplace", "rev"), _OPS, ids=_IDS)
def test_errors(cuda_oracle: CudaOracle, op, fn, inplace, rev):
    """What CUDA refuses: float results into an integer `out=` / self, bool
    and unsigned dtypes outside the dispatch, unknown rounding modes, shape
    mismatches."""
    i = torch.arange(-6, 6, dtype=torch.int32)
    f = torch.linspace(-3, 3, 12)
    if op == "div":
        bp.check_all(
            cuda_oracle,
            [("bad_mode", lambda x: torch.div(x, 2, rounding_mode="round"), (f,))],
            check_message=True,
        )
    cases = [
        ("int_out_float", lambda x, y: fn(y, x, out=torch.empty_like(x)), (i, f)),
        ("inplace_int_by_float", lambda x, y: inplace(x, y), (i, f)),
        ("inplace_int_by_float_s", lambda x: inplace(x, 1.5), (i,)),
        ("bool_bool", fn, (i > 0, i < 3)),
        ("shape", lambda x: fn(x, x[:5]), (f,)),
        ("inplace_bcast", lambda x: inplace(x[:1], x), (f,)),
    ]
    bp.check_all(cuda_oracle, cases)


@functools.cache
def _halves_sweep(dtype):
    """Every finite bit pattern of a half dtype over a few divisors."""
    x = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32).to(torch.int16).view(dtype)
    divs = torch.tensor([3.0, -0.1, 7.5, 1e-3, -250.0, 0.6], dtype=torch.float32)
    return x.repeat(len(divs)), divs.to(dtype).repeat_interleave(len(x))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=bp.name)
@pytest.mark.parametrize(("op", "fn", "inplace", "rev"), _OPS, ids=_IDS)
def test_half_sweep(cuda_oracle: CudaOracle, op, fn, inplace, rev, dtype):
    """Every half bit pattern as the numerator, by tensors and by numbers."""
    a, b = _halves_sweep(dtype)
    cases = [("tt", fn, (a, b))]
    for s in (3.0, -0.1, 7.5, 1e-3, 0.6):
        cases.append((f"ts={s}", lambda x, s=s: fn(x, s), (a[: 1 << 16],)))
    bp.check_all(cuda_oracle, cases)


@pytest.mark.parametrize("s", [1e-40, 3e-40, 1e-38])
def test_bf16_fmod_denormal_scalar(cuda_oracle: CudaOracle, s):
    """A bfloat16 denormal CPU divisor: the kernel must get its bits."""
    a, _ = bp.pairs(torch.bfloat16)
    bp.check_all(
        cuda_oracle,
        [
            ("tt_s", lambda x: torch.fmod(x, s), (a,)),
            ("inplace_s", lambda x: x.fmod_(s), (a,)),
        ],
    )
