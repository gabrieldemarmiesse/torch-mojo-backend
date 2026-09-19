"""Tests for the native `unary` op group
(torch_mojo_backend/native/mojo/ops_unary.mojo): abs/neg/sign/relu, the
transcendental unary ops, ceil/floor, gelu(+backward), isnan/logical_not/
bitwise_not and fill.Scalar.

Public torch API only, run against the mojo device: `mojo_gpu`/`mojo_device`
(tests/conftest.py) call `register_mojo_devices()`, which now registers the
*native* backend (`torch_mojo_backend/mojo_device/register.py` ->
`native.register()`) rather than the old Python eager path.
"""

from collections.abc import Callable

import pytest
import torch
import torch.nn.functional as F

from tests.elementwise_cases import log1p_edge_input, log1p_rtol
from tests.native.conftest import skip_if_metal
from torch_mojo_backend import aten_functions, get_accelerators, native
from torch_mojo_backend.native import device_module


def _native_count(name: str) -> int:
    return native.op_count(f"aten::{name}")


def _reset_native_counts():
    native.op_counting(True)
    native.op_counts_reset()


def _tol(dtype: torch.dtype) -> tuple[float | None, float | None]:
    if dtype in (torch.float16, torch.bfloat16):
        return 3e-2, 3e-2
    return None, None


# The original native-dispatch sweep uses bfloat16. The launcher tests below
# exercise fp16, bf16 and fp32 across alignment, tail and output-view regimes.
SWEEP_DTYPE = torch.bfloat16

# (native op name, torch callable, input-domain key)
_UNARY_OPS = [
    ("abs", torch.abs, "signed"),
    ("acos", torch.acos, "unit"),
    ("asinh", torch.asinh, "signed"),
    ("atanh", torch.atanh, "unit"),
    ("cos", torch.cos, "signed"),
    ("cosh", torch.cosh, "signed"),
    ("erf", torch.erf, "signed"),
    ("exp", torch.exp, "signed"),
    ("log", torch.log, "positive"),
    ("log1p", torch.log1p, "above_neg1"),
    ("neg", torch.neg, "signed"),
    ("reciprocal", torch.reciprocal, "positive"),
    ("rsqrt", torch.rsqrt, "positive"),
    ("sigmoid", torch.sigmoid, "signed"),
    ("sign", torch.sign, "signed"),
    ("silu", F.silu, "signed"),
    ("sin", torch.sin, "signed"),
    ("sinh", torch.sinh, "signed"),
    ("sqrt", torch.sqrt, "positive"),
    ("tan", torch.tan, "unit"),
    ("tanh", torch.tanh, "signed"),
    ("relu", torch.relu, "signed"),
]


def _sample(domain: str, shape: tuple[int, ...]) -> torch.Tensor:
    torch.manual_seed(0)
    if domain == "unit":
        return torch.empty(shape, dtype=torch.float64).uniform_(-0.85, 0.85)
    if domain == "positive":
        return torch.empty(shape, dtype=torch.float64).uniform_(0.15, 4.0)
    if domain == "above_neg1":
        # log1p(x) = log(1 + x) needs x > -1; stay well clear of the pole.
        return torch.empty(shape, dtype=torch.float64).uniform_(-0.9, 4.0)
    return torch.randn(shape, dtype=torch.float64) * 2


@pytest.mark.parametrize("op_name,fn,domain", _UNARY_OPS)
def test_unary_matches_cpu(mojo_gpu, op_name, fn, domain):
    x64 = _sample(domain, (3, 5))
    expected = fn(x64).to(SWEEP_DTYPE)
    x = x64.to(SWEEP_DTYPE).to(mojo_gpu)
    _reset_native_counts()
    y = fn(x)
    assert y.device.type == "mojo"
    assert _native_count(op_name) > 0, f"aten::{op_name} did not run natively"
    rtol, atol = _tol(SWEEP_DTYPE)
    torch.testing.assert_close(y.cpu(), expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "op_name,fn", [("abs", torch.abs), ("exp", torch.exp), ("sign", torch.sign)]
)
def test_unary_on_cpu_and_gpu_device(mojo_device, op_name, fn):
    """The elementwise spec kernels run on the MAX CPU device too (mojo:cpu),
    not only accelerators — `mojo_device` covers both legs."""
    x64 = torch.randn(3, 5, dtype=torch.float64) * 2
    expected = fn(x64).to(torch.float32)
    x = x64.to(torch.float32).to(mojo_device)
    _reset_native_counts()
    y = fn(x)
    assert _native_count(op_name) > 0
    torch.testing.assert_close(y.cpu(), expected)


def test_unary_noncontiguous_input(mojo_gpu):
    x64 = torch.randn(4, 6, dtype=torch.float64)
    x_cpu = x64.to(torch.float32)
    x = x_cpu.to(mojo_gpu).t()
    assert not x.is_contiguous()
    y = torch.exp(x)
    torch.testing.assert_close(y.cpu(), torch.exp(x_cpu.t()))


_ALIGNED_UNARY_OPS = _UNARY_OPS + [
    ("ceil", torch.ceil, "signed"),
    ("floor", torch.floor, "signed"),
    ("log2", torch.log2, "positive"),
    ("gelu", F.gelu, "signed"),
    ("isnan", torch.isnan, "signed"),
]


@pytest.mark.parametrize("op_name,fn,domain", _ALIGNED_UNARY_OPS)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("layout", ("aligned", "offset_1", "strided"))
def test_unary_launcher_layouts(
    mojo_gpu: str,
    op_name: str,
    fn: Callable[[torch.Tensor], torch.Tensor],
    domain: str,
    dtype: torch.dtype,
    layout: str,
):
    """The vector launcher and its fallback handle every tail and view."""
    for count in (0, 1, 2, 3, 256, 257, 357 * 789):
        cpu = _sample(domain, (count,)).to(dtype)
        if layout == "aligned":
            x = cpu.to(mojo_gpu)
            if count:
                assert x.data_ptr() % (4 * x.element_size()) == 0
        elif layout == "offset_1":
            storage = torch.cat((torch.zeros(1, dtype=dtype), cpu)).to(mojo_gpu)
            x = storage[1:]
            if count:
                assert x.data_ptr() % (4 * x.element_size()) == x.element_size()
        else:
            storage = torch.stack((cpu, torch.zeros_like(cpu)), dim=1).to(mojo_gpu)
            x = storage[:, 0]
        _reset_native_counts()
        actual = fn(x)
        assert _native_count(op_name) > 0
        torch.testing.assert_close(actual.cpu(), fn(cpu))


@pytest.mark.parametrize(
    "op_name,fn,domain",
    [op for op in _ALIGNED_UNARY_OPS if op[0] not in {"silu", "gelu", "relu", "isnan"}],
)
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("input_offset", (0, 1))
def test_unary_launcher_offset_out(
    mojo_gpu: str,
    op_name: str,
    fn: Callable[[torch.Tensor], torch.Tensor],
    domain: str,
    dtype: torch.dtype,
    input_offset: int,
):
    """An aligned input cannot justify vector stores into an offset output."""
    cpu = _sample(domain, (257,)).to(dtype)
    input_storage = torch.cat((torch.zeros(input_offset, dtype=dtype), cpu)).to(
        mojo_gpu
    )
    x = input_storage[input_offset:]
    output_storage = torch.full((259,), 37.0, dtype=dtype, device=mojo_gpu)
    out = output_storage[1:-1]
    assert out.data_ptr() % (4 * out.element_size()) == out.element_size()
    result = getattr(torch, op_name)(x, out=out)
    assert result.data_ptr() == out.data_ptr()
    actual = output_storage.cpu()
    torch.testing.assert_close(actual[1:-1], fn(cpu))
    assert actual[0].item() == actual[-1].item() == 37.0


@pytest.mark.parametrize(
    "op_name,dtype",
    [("log2", torch.float64)]
    + [
        (name, dtype)
        for name in ("abs", "neg", "sign", "ceil", "floor")
        for dtype in (torch.int32, torch.uint8)
    ],
)
@pytest.mark.parametrize("offset", (0, 1))
def test_unary_launcher_other_dtypes(
    mojo_gpu: str, op_name: str, dtype: torch.dtype, offset: int
):
    """SIMD alignment is measured in elements for every supported dtype."""
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Metal does not support float64")
    cpu = (torch.arange(261) % 13 + 1).to(dtype)
    if dtype != torch.uint8 and op_name not in {"rsqrt", "log2"}:
        cpu[::2].neg_()
    x = cpu.to(mojo_gpu)[offset : offset + 257]
    expected = getattr(torch, op_name)(cpu[offset : offset + 257])
    actual = getattr(torch, op_name)(x)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize(
    "dtype", (torch.float16, torch.bfloat16, torch.float32, torch.bool)
)
@pytest.mark.parametrize("input_offset,output_offset", ((0, 0), (0, 1), (1, 0), (1, 1)))
def test_unary_launcher_bool_output(
    mojo_gpu: str, dtype: torch.dtype, input_offset: int, output_offset: int
):
    """Boolean vector stores must use the output's one-byte element size."""
    cpu = torch.tensor([0, -1, float("inf"), float("nan")], dtype=dtype).repeat(65)
    x = cpu.to(mojo_gpu)[input_offset : input_offset + 257]
    storage = torch.ones(259, dtype=torch.bool, device=mojo_gpu)
    out = storage[output_offset : output_offset + 257]
    torch.logical_not(x, out=out)
    expected = torch.ones(259, dtype=torch.bool)
    expected[output_offset : output_offset + 257] = torch.logical_not(
        cpu[input_offset : input_offset + 257]
    )
    torch.testing.assert_close(storage.cpu(), expected)


@pytest.mark.parametrize("op_name", ("exp", "log"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("offset", (0, 1))
def test_unary_launcher_exp_log_special_values(
    mojo_gpu: str, op_name: str, dtype: torch.dtype, offset: int
):
    cpu = torch.tensor(
        [-float("inf"), -1, -0.0, 0.0, 1, float("inf"), float("nan")], dtype=dtype
    ).repeat(37)
    storage = torch.cat((torch.zeros(offset, dtype=dtype), cpu)).to(mojo_gpu)
    actual = getattr(torch, op_name)(storage[offset:])
    torch.testing.assert_close(
        actual.cpu(), getattr(torch, op_name)(cpu), equal_nan=True
    )


@pytest.mark.parametrize("dtype", (torch.int32, torch.int64, torch.uint8, torch.bool))
@pytest.mark.parametrize(
    "input_offset,output_offset", ((0, 0), (0, 1), (1, 0), (1, 1), (2, 2))
)
def test_unary_launcher_bitwise_not_offsets(
    mojo_gpu: str, dtype: torch.dtype, input_offset: int, output_offset: int
):
    cpu = (torch.arange(260) % 17).to(dtype)
    if dtype in (torch.int32, torch.int64):
        cpu[::2].neg_()
    x = cpu.to(mojo_gpu)[input_offset : input_offset + 257]
    storage = torch.full((261,), 37, dtype=dtype, device=mojo_gpu)
    out = storage[output_offset : output_offset + 257]
    torch.bitwise_not(x, out=out)
    expected = torch.full((261,), 37, dtype=dtype)
    expected[output_offset : output_offset + 257] = torch.bitwise_not(
        cpu[input_offset : input_offset + 257]
    )
    torch.testing.assert_close(storage.cpu(), expected)


@pytest.mark.parametrize("dtype", (torch.int64, torch.float64))
def test_unary_launcher_predicate_offset_two(mojo_gpu: str, dtype: torch.dtype):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Metal does not support float64")
    cpu = (torch.arange(260) % 13).to(dtype)
    if dtype == torch.float64:
        cpu[::7] = float("nan")
    x = cpu.to(mojo_gpu)[2:259]
    assert x.data_ptr() % 16 == 0
    assert x.data_ptr() % 32 == 16
    torch.testing.assert_close(torch.isnan(x).cpu(), torch.isnan(cpu[2:259]))
    storage = torch.ones(261, dtype=torch.bool, device=mojo_gpu)
    out = storage[2:259]
    torch.logical_not(x, out=out)
    expected = torch.ones(261, dtype=torch.bool)
    expected[2:259] = torch.logical_not(cpu[2:259])
    torch.testing.assert_close(storage.cpu(), expected)


@pytest.mark.parametrize(
    "op_name,dtype,offset,count",
    [
        (*case, count)
        for case in [
            (name, dtype, 4)
            for name in ("gelu", "gelu_tanh")
            for dtype in (torch.float16, torch.bfloat16)
        ]
        + [
            ("log1p", torch.float32, 4),
            ("log2", torch.float32, 4),
            ("logical_not", torch.bool, 16),
            ("bitwise_not", torch.int32, 4),
        ]
        for count in (257, 513)
    ]
    + [
        ("acos", dtype, offset, count)
        for dtype in (torch.float16, torch.bfloat16)
        for offset in (0, 4, 8)
        for count in (0, 1, 15, 16, 17, 255, 256, 257, 513)
    ],
)
def test_unary_launcher_intermediate_alignment(
    mojo_gpu: str, op_name: str, dtype: torch.dtype, offset: int, count: int
):
    """Wider SIMD must retain the older alignment regimes and guarded tails."""
    if dtype in (torch.bool, torch.int32):
        cpu = (torch.arange(count) % 7 - 3).to(dtype)
    else:
        cpu = _sample("positive" if op_name.startswith("log") else "unit", (count,))
        cpu = cpu.to(dtype)
    input_storage = torch.cat((torch.zeros(offset, dtype=dtype), cpu)).to(mojo_gpu)
    x = input_storage[offset:]
    storage = torch.full((offset + count + 3,), 37, dtype=dtype, device=mojo_gpu)
    out = storage[offset : offset + count]
    if count:
        for tensor in (x, out):
            if offset:
                old_alignment = offset * tensor.element_size()
                assert tensor.data_ptr() % (2 * old_alignment) == old_alignment
            else:
                assert tensor.data_ptr() % 16 == 0
    if op_name.startswith("gelu"):
        approximate = "tanh" if op_name == "gelu_tanh" else "none"
        result = torch.ops.aten.gelu.out(x, approximate=approximate, out=out)
        reference = F.gelu(cpu, approximate=approximate)
    else:
        result = getattr(torch, op_name)(x, out=out)
        reference = getattr(torch, op_name)(cpu)
    assert result.data_ptr() == out.data_ptr()
    expected = torch.full((offset + count + 3,), 37, dtype=dtype)
    expected[offset : offset + count] = reference
    torch.testing.assert_close(storage.cpu(), expected)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("offset", (0, 1))
def test_unary_launcher_log1p_near_zero(mojo_gpu: str, dtype: torch.dtype, offset: int):
    cpu = log1p_edge_input(dtype)
    storage = torch.cat((torch.zeros(offset, dtype=dtype), cpu)).to(mojo_gpu)
    actual = torch.log1p(storage[offset:]).cpu()
    expected = torch.log1p(cpu.double()).to(dtype)
    torch.testing.assert_close(
        actual, expected, rtol=log1p_rtol(dtype), atol=0, equal_nan=True
    )
    zeros = cpu == 0
    torch.testing.assert_close(torch.signbit(actual[zeros]), torch.signbit(cpu[zeros]))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("offset", (0, 1))
def test_unary_launcher_asinh_zero(mojo_gpu: str, dtype: torch.dtype, offset: int):
    cpu = torch.tensor([-0.0, 0.0, -0.0, 0.0, 1.0], dtype=dtype)
    storage = torch.cat((torch.zeros(offset, dtype=dtype), cpu)).to(mojo_gpu)
    actual = torch.asinh(storage[offset:]).cpu()
    expected = torch.asinh(cpu)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.signbit(actual), torch.signbit(expected))


_DIRECT_OPS = [
    ("abs", torch.abs),
    ("neg", torch.neg),
    ("sign", torch.sign),
    ("relu", torch.relu),
]
_DIRECT_INT_DTYPES = (torch.int32, torch.uint8)


@pytest.mark.parametrize("dtype", _DIRECT_INT_DTYPES)
@pytest.mark.parametrize("op_name,fn", _DIRECT_OPS)
def test_direct_ops_int_dtypes(mojo_gpu, op_name, fn, dtype):
    lo, hi = (0, 6) if dtype == torch.uint8 else (-6, 6)
    x_cpu = torch.randint(lo, hi, (4, 5)).to(dtype)
    x = x_cpu.to(mojo_gpu)
    _reset_native_counts()
    y = fn(x)
    assert _native_count(op_name) > 0
    assert y.cpu().tolist() == fn(x_cpu).tolist()


@pytest.mark.parametrize(
    "op_name,fn", [("abs", torch.abs), ("exp", torch.exp), ("sigmoid", torch.sigmoid)]
)
def test_out_variant(mojo_gpu, op_name, fn):
    x64 = torch.randn(3, 4, dtype=torch.float64)
    expected = fn(x64).to(torch.float32)
    x = x64.to(torch.float32).to(mojo_gpu)
    out_name = f"{op_name}.out"

    # Ready out tensor (contiguous, right shape/dtype): compute straight into it.
    out1 = torch.empty(3, 4, device=mojo_gpu)
    _reset_native_counts()
    ret1 = fn(x, out=out1)
    assert _native_count(out_name) > 0
    assert ret1.data_ptr() == out1.data_ptr()
    torch.testing.assert_close(out1.cpu(), expected)

    # Non-contiguous out tensor: compute into a temporary then copy_strided_into.
    out2 = torch.empty(4, 3, device=mojo_gpu).t()
    assert not out2.is_contiguous()
    _reset_native_counts()
    fn(x, out=out2)
    assert _native_count(out_name) > 0
    # `.cpu()` on a strided mojo tensor would itself need a strided host
    # destination, which `_copy_from` (ops_core.mojo, outside this group)
    # does not support; materialize contiguous on-device first.
    torch.testing.assert_close(out2.contiguous().cpu(), expected)


def test_out_variant_resizes_a_mismatching_out(mojo_gpu):
    """Every `out=` op in ATen runs `resize_output` first; without it the copy
    back faces a shape it cannot satisfy."""
    x = torch.randn(3, 4, device=mojo_gpu)
    out = torch.empty(0, device=mojo_gpu)
    torch.abs(x, out=out)
    assert tuple(out.shape) == (3, 4)
    torch.testing.assert_close(out.cpu(), x.cpu().abs())
    # A correctly shaped `out` that is a view keeps its own offset.
    base = torch.zeros(16, device=mojo_gpu)
    view = base[4:8]
    torch.abs(torch.full((4,), -2.0, device=mojo_gpu), out=view)
    assert base.cpu().tolist() == [0.0] * 4 + [2.0] * 4 + [0.0] * 8


def test_out_variant_rejects_partial_overlap(mojo_gpu):
    x = torch.arange(8, dtype=torch.float32, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.neg(x[:-1], out=x[1:])
    # The same view is not an overlap: that is how the in-place ops work.
    torch.neg(x[:4], out=x[:4])


def test_relu_inplace(mojo_gpu):
    x64 = torch.randn(3, 4, dtype=torch.float64)
    expected = torch.relu(x64).to(torch.float32)
    x = x64.to(torch.float32).to(mojo_gpu)
    _reset_native_counts()
    ptr_before = x.data_ptr()
    ret = x.relu_()
    assert _native_count("relu_") > 0
    assert ret.data_ptr() == ptr_before
    torch.testing.assert_close(x.cpu(), expected)


def test_relu_inplace_noncontiguous(mojo_gpu):
    x64 = torch.randn(4, 3, dtype=torch.float64)
    expected = torch.relu(x64.t()).to(torch.float32)
    x = x64.to(torch.float32).to(mojo_gpu).t()
    assert not x.is_contiguous()
    ptr_before = x.data_ptr()
    x.relu_()
    assert x.data_ptr() == ptr_before  # in-place: same storage, no reallocation
    # See test_out_variant: `.cpu()` needs a contiguous source for this
    # backend's current `_copy_from` (outside this group).
    torch.testing.assert_close(x.contiguous().cpu(), expected)


@pytest.mark.parametrize("op_name,fn", [("ceil", torch.ceil), ("floor", torch.floor)])
def test_ceil_floor_float(mojo_gpu, op_name, fn):
    x64 = torch.randn(3, 5, dtype=torch.float64) * 3
    expected = fn(x64).to(torch.float32)
    x = x64.to(torch.float32).to(mojo_gpu)
    _reset_native_counts()
    y = fn(x)
    assert _native_count(op_name) > 0
    torch.testing.assert_close(y.cpu(), expected)


@pytest.mark.parametrize("op_name,fn", [("ceil", torch.ceil), ("floor", torch.floor)])
def test_ceil_floor_int_is_identity_copy(mojo_gpu, op_name, fn):
    """ceil/floor of an int tensor is the identity, but still functional: a
    fresh tensor, not the same object (matches aten_fast._int_unary_identity).
    """
    x_cpu = torch.randint(-5, 5, (3, 4), dtype=torch.int64)
    x = x_cpu.to(mojo_gpu)
    y = fn(x)
    assert y.data_ptr() != x.data_ptr()
    assert y.cpu().tolist() == x_cpu.tolist()


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_forward(mojo_gpu, approximate):
    x64 = torch.randn(3, 5, dtype=torch.float64) * 2
    expected = F.gelu(x64, approximate=approximate).to(SWEEP_DTYPE)
    x = x64.to(SWEEP_DTYPE).to(mojo_gpu)
    _reset_native_counts()
    y = F.gelu(x, approximate=approximate)
    assert _native_count("gelu") > 0
    rtol, atol = _tol(SWEEP_DTYPE)
    torch.testing.assert_close(y.cpu(), expected, rtol=rtol, atol=atol)


def test_gelu_out_variant(mojo_gpu):
    x64 = torch.randn(3, 5, dtype=torch.float64) * 2
    expected = F.gelu(x64, approximate="tanh").to(torch.float32)
    x = x64.to(torch.float32).to(mojo_gpu)
    out = torch.empty(3, 5, device=mojo_gpu)
    _reset_native_counts()
    torch.ops.aten.gelu.out(x, approximate="tanh", out=out)
    assert _native_count("gelu.out") > 0
    torch.testing.assert_close(out.cpu(), expected)


def test_gelu_invalid_approximate_declines(mojo_gpu):
    x = torch.randn(3).to(mojo_gpu)
    with pytest.raises(NotImplementedError):
        F.gelu(x, approximate="bogus")


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_backward_matches_cpu(mojo_gpu, approximate):
    x64 = (torch.randn(3, 5, dtype=torch.float64) * 2).requires_grad_()
    g64 = torch.randn(3, 5, dtype=torch.float64)
    F.gelu(x64, approximate=approximate).backward(g64)
    assert x64.grad is not None
    expected_grad = x64.grad.to(SWEEP_DTYPE)

    x = x64.detach().to(SWEEP_DTYPE).to(mojo_gpu).requires_grad_()
    g = g64.to(SWEEP_DTYPE).to(mojo_gpu)
    _reset_native_counts()
    F.gelu(x, approximate=approximate).backward(g)
    assert _native_count("gelu_backward") > 0
    assert x.grad is not None
    rtol, atol = _tol(SWEEP_DTYPE)
    torch.testing.assert_close(x.grad.cpu(), expected_grad, rtol=rtol, atol=atol)


def test_gelu_backward_declines_float16(mojo_gpu):
    x = torch.randn(3, 4, dtype=torch.float16).to(mojo_gpu).requires_grad_()
    y = F.gelu(x)
    with pytest.raises(NotImplementedError):
        y.backward(torch.ones_like(y))


def test_gelu_backward_declines_on_cpu_device(mojo_gpu):
    # `mojo_gpu` only guarantees registration + a real accelerator exists;
    # this test deliberately targets the MAX CPU device instead.
    cpu_device = f"mojo:{device_module.device_count() - 1}"
    x = torch.randn(3, 4).to(cpu_device).requires_grad_()
    y = F.gelu(x)
    with pytest.raises(NotImplementedError):
        y.backward(torch.ones_like(y))


def test_an_unregistered_op_raises_out_of_the_dispatcher(mojo_gpu):
    """An op with no PrivateUse1 kernel must raise, not abort the process.

    This was written for sigmoid/tanh/threshold backward, which the old
    eager path had to preflight from the FORWARD because a Python exception
    raised inside its autograd engine could kill the interpreter; the native
    backend has no such hazard and those three now have composed kernels
    (tests/native/test_composed.py). `masked_select` stands in as an op the
    backend genuinely does not implement -- the point is the failure mode,
    not which op it is.
    """
    x = torch.rand(4).to(mojo_gpu)
    mask = (x > 0.5).to(mojo_gpu)
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.masked_select(x, mask)


def test_isnan(mojo_gpu):
    x_cpu = torch.tensor([1.0, float("nan"), -float("inf"), 2.0])
    x = x_cpu.to(mojo_gpu)
    _reset_native_counts()
    y = torch.isnan(x)
    assert _native_count("isnan") > 0
    assert y.cpu().tolist() == torch.isnan(x_cpu).tolist()


def test_logical_not(mojo_gpu):
    x_cpu = torch.tensor([True, False, True, False])
    x = x_cpu.to(mojo_gpu)
    _reset_native_counts()
    y = torch.logical_not(x)
    assert _native_count("logical_not") > 0
    assert y.cpu().tolist() == torch.logical_not(x_cpu).tolist()


@pytest.mark.parametrize("dtype", [torch.int32, torch.uint8, torch.bool])
def test_bitwise_not(mojo_gpu, dtype):
    if dtype is torch.bool:
        x_cpu = torch.tensor([True, False, True])
    else:
        x_cpu = torch.randint(0, 20, (5,)).to(dtype)
    x = x_cpu.to(mojo_gpu)
    _reset_native_counts()
    y = torch.bitwise_not(x)
    assert _native_count("bitwise_not") > 0
    assert y.cpu().tolist() == torch.bitwise_not(x_cpu).tolist()


def test_fill_scalar_functional(mojo_gpu):
    x_cpu = torch.zeros(3, 4)
    x = x_cpu.to(mojo_gpu)
    _reset_native_counts()
    y = torch.fill(x, 7.5)
    assert _native_count("fill.Scalar") > 0
    assert y.data_ptr() != x.data_ptr()  # functional: does not alias self
    assert y.cpu().tolist() == torch.full((3, 4), 7.5).tolist()
    assert x.cpu().tolist() == [[0.0] * 4] * 3  # self left untouched


@pytest.mark.parametrize(
    "aten_fn,torch_fn",
    [
        (aten_functions.aten_abs, torch.abs),
        (aten_functions.aten_exp, torch.exp),
        (aten_functions.aten_sigmoid, torch.sigmoid),
        (aten_functions.aten_relu, torch.relu),
        (aten_functions.aten_isnan, torch.isnan),
    ],
)
def test_call_checker_confirms_native_dispatch(
    mojo_gpu, call_checker, aten_fn, torch_fn
):
    call_checker.register(aten_fn)
    x = torch.randn(3, 4).to(mojo_gpu)
    torch_fn(x)


def test_call_checker_bitwise_not(mojo_gpu, call_checker):
    call_checker.register(aten_functions.aten_bitwise_not)
    x = torch.randint(0, 10, (3, 4), dtype=torch.int32).to(mojo_gpu)
    torch.bitwise_not(x)


# --------------------------------------------------------------------------
# bf16 GELU exactness. The sweep above compares 15 values at atol 3e-2, which
# cannot see either failure mode below.
# --------------------------------------------------------------------------


@pytest.fixture
def mojo_h100(mojo_gpu):
    """H100 mojo device: the frozen bit patterns below are that card's."""
    accelerator = list(get_accelerators())[0]
    if accelerator.api != "cuda" or accelerator.architecture_name != "sm_90a":
        pytest.skip("the frozen GELU bit patterns were recorded on an H100")
    return mojo_gpu


def _finite_bf16_grid(limit: float) -> torch.Tensor:
    """Every finite bf16 value with |x| <= limit, as a contiguous tensor."""
    bits = torch.arange(1 << 16, dtype=torch.int32).to(torch.uint16)
    values = bits.view(torch.bfloat16)
    values = values[torch.isfinite(values) & (values.abs() <= limit)]
    return values.contiguous()


def _gelu_none_fp64(x: torch.Tensor) -> torch.Tensor:
    """x * Phi(x), written so it keeps its bits in the left tail.

    NOT `F.gelu(x.double())`: `1 + erf(x/sqrt(2))` is quantized by the fp64
    epsilon at 1.0, and by x = -8 half the significand is already gone.
    `erfc` keeps the small value small.
    """
    x = x.double()
    return torch.relu(x) - 0.5 * x.abs() * torch.erfc(x.abs() / 2.0**0.5)


@pytest.mark.xfail(
    strict=True,
    reason="the bf16 route computes `0.5*x*(1 + erf(x/sqrt2))`, whose sum is "
    "quantized by the fp32 epsilon at 1.0: below x = -5.2 it returns exactly "
    "-0.0, and gelu(-inf) is NaN instead of -0.0. The old eager path used "
    "`relu(x) - 0.5*|x|*erfc(|x|/sqrt2)`, which resolves the tail down to "
    "x = -13.7 and is within one ulp of the true function over the whole "
    "bf16 grid. Drop the marker when that form is back.",
)
def test_gelu_bf16_matches_a_double_reference_over_the_whole_grid(mojo_gpu):
    """Every finite bf16 input rounds to the same bf16 as the true function.

    12.5 is where the kernel's `exp2` reaches the smallest fp32 normal, so the
    last twelve bf16 inputs before the answer underflows bf16 altogether are
    outside the guarantee and excluded.
    """
    x_cpu = _finite_bf16_grid(12.5)
    assert x_cpu.numel() > 30000, x_cpu.numel()
    expected = _gelu_none_fp64(x_cpu).bfloat16()
    actual = F.gelu(x_cpu.to(mojo_gpu), approximate="none").cpu()

    actual_bits = actual.view(torch.int16).int()
    expected_bits = expected.view(torch.int16).int()
    worst = int((actual_bits - expected_bits).abs().max())
    assert worst <= 1, f"a bf16 result is {worst} ULP from the true exact GELU"
    # The survivors are genuine round-to-nearest ties, not a systematic bias.
    assert int((actual_bits != expected_bits).sum()) <= 8


@pytest.mark.xfail(
    strict=True,
    reason="the bf16 route computes `0.5*x*(1 + erf(x/sqrt2))`, whose sum is "
    "quantized by the fp32 epsilon at 1.0: below x = -5.2 it returns exactly "
    "-0.0, and gelu(-inf) is NaN instead of -0.0. The old eager path used "
    "`relu(x) - 0.5*|x|*erfc(|x|/sqrt2)`, which resolves the tail down to "
    "x = -13.7 and is within one ulp of the true function over the whole "
    "bf16 grid. Drop the marker when that form is back.",
)
def test_gelu_bf16_resolves_the_negative_tail(mojo_gpu):
    """Below x ~ -5.2 a form built on `0.5*x*(1+erf(x/sqrt2))` returns
    exactly 0: the sum has lost every bit of the answer."""
    x_cpu = torch.arange(-12.5, -5.0, 0.0625, dtype=torch.bfloat16)
    actual = F.gelu(x_cpu.to(mojo_gpu), approximate="none").cpu()
    assert bool((actual < 0).all()), "the negative tail collapsed to zero"
    # Strictly decreasing in x, i.e. the decay has the shape of the true
    # function and not of a floor or a plateau.
    assert bool((actual[1:].double() < actual[:-1].double()).all())
    torch.testing.assert_close(
        actual.double(), _gelu_none_fp64(x_cpu), rtol=8e-3, atol=0
    )


@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_bf16_special_values(mojo_h100, approximate):
    """Signed zero, non-finites and two mode probes, against frozen H100
    results. The `-inf` case is where the two forms diverge: CUDA's
    `0.5*x*(1 + erf(x/sqrt2))` reaches `-inf * 0` and returns NaN, while
    `relu(x) - 0.5*|x|*erfc(...)` never forms that product and gives the
    correct limit, -0.0. `approximate="none"` currently takes the first
    form (see the xfail on the two tests above), so its -inf case is
    xfailed here rather than the whole parametrization."""
    input_bits = torch.tensor(
        [
            0x0000,
            0x8000,
            0x7F80,
            0xFF80,
            0x7FC0,
            0x0001,
            0x8001,
            0x7F7F,
            0xFF7F,
            0x4005,
            0x4030,
        ],
        dtype=torch.uint16,
    )
    source = input_bits.view(torch.bfloat16)
    on_device = source.to(mojo_h100)
    actual = F.gelu(on_device, approximate=approximate).cpu()
    actual_bits = actual.view(torch.uint16)

    assert int(actual_bits[0]) == 0x0000
    assert int(actual_bits[1]) == 0x8000
    assert torch.isposinf(actual[2])
    if approximate == "tanh":
        assert torch.isnan(actual[3])
    elif int(actual_bits[3]) != 0x8000:
        pytest.xfail(
            "gelu(-inf, approximate='none') is NaN: the bf16 route forms "
            "`-inf * 0` through `0.5*x*(1 + erf)` instead of the erfc "
            "identity, whose limit is -0.0"
        )
    assert torch.isnan(actual[4])
    expected_probes = (0x4002, 0x402F) if approximate == "none" else (0x4003, 0x4030)
    assert tuple(int(value) for value in actual_bits[-2:]) == expected_probes
    torch.testing.assert_close(
        on_device.cpu().view(torch.int16), source.view(torch.int16), atol=0, rtol=0
    )


@pytest.mark.parametrize("approximate", ["none", "tanh"])
@pytest.mark.parametrize(
    "layout", ["contiguous", "transposed", "gapped", "empty", "fp32", "fp16"]
)
def test_gelu_forward_every_layout_and_dtype(mojo_gpu, approximate, layout):
    """The regimes around the bf16 direct route: other dtypes, a device
    transpose, a stride-2 view, and an empty tensor (which must not launch)."""
    if layout == "empty":
        x = torch.empty(0, 7, dtype=torch.bfloat16, device=mojo_gpu)
        out = F.gelu(x, approximate=approximate)
        assert out.shape == (0, 7) and out.dtype == torch.bfloat16
        return
    dtype = {"fp32": torch.float32, "fp16": torch.float16}.get(layout, torch.bfloat16)
    if layout == "transposed":
        cpu = torch.randn(7, 5).to(dtype)
        x, x_cpu = cpu.to(mojo_gpu).t(), cpu.t()
    elif layout == "gapped":
        cpu = torch.randn(71).to(dtype)
        x, x_cpu = cpu.to(mojo_gpu)[1:71:2], cpu[1:71:2]
    else:
        cpu = torch.randn(5, 7).to(dtype)
        x, x_cpu = cpu.to(mojo_gpu), cpu
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(
        F.gelu(x, approximate=approximate).cpu(),
        F.gelu(x_cpu.float(), approximate=approximate).to(dtype),
        rtol=rtol or 5e-5,
        atol=atol or 5e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_vector_body_tail_and_offset(
    mojo_gpu: str, dtype: torch.dtype, offset: int, approximate: str
):
    """The 4-wide route handles aligned bodies and unaligned scalar tails."""
    count = 1027
    source = torch.linspace(-5, 5, count + offset + 3).to(dtype)
    device_source = source.to(mojo_gpu)
    input_view = device_source[offset : offset + count]
    output = torch.full_like(source, -123).to(mojo_gpu)
    output_view = output[offset : offset + count]
    with torch.no_grad():
        torch.ops.aten.gelu.out(input_view, approximate=approximate, out=output_view)
    expected = torch.full_like(source, -123)
    expected[offset : offset + count] = F.gelu(
        source[offset : offset + count].float(), approximate=approximate
    ).to(dtype)
    tolerance = 3e-2 if dtype != torch.float32 else 5e-5
    torch.testing.assert_close(output.cpu(), expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(device_source.cpu(), source, rtol=0, atol=0)


@pytest.mark.parametrize("step", [1, 2])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_gelu_backward_runtime_layouts(mojo_gpu, approximate, step):
    """gelu_backward straight from aten, on operands that start at a nonzero
    offset (step 1) or skip every other element (step 2). The autograd test
    above only ever feeds it contiguous tensors from offset 0."""
    n = 257
    grad_backing = torch.linspace(2.0, -2.0, 519)
    input_backing = torch.linspace(-8.0, 8.0, 517)
    grad_cpu = grad_backing[1 : 1 + n * step : step]
    input_cpu = input_backing[2 : 2 + n * step : step]
    grad = grad_backing.to(mojo_gpu)[1 : 1 + n * step : step]
    x = input_backing.to(mojo_gpu)[2 : 2 + n * step : step]
    expected = torch.ops.aten.gelu_backward(
        grad_cpu, input_cpu, approximate=approximate
    )
    actual = torch.ops.aten.gelu_backward(grad, x, approximate=approximate)
    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-5, atol=5e-5)


# --------------------------------------------------------------------------
# fill_ / zero_ over every rank, length, alignment phase and layout.
#
# One dtype per element width (8/4/2/1 bytes), because the vector width the
# kernel picks is a function of the element size, the extent AND the runtime
# base address. Every case fills a VIEW of an 8192-element base and compares
# the WHOLE base, so a kernel writing outside the view is caught.
# --------------------------------------------------------------------------

_FILL_DTYPES = [torch.int64, torch.float32, torch.bfloat16, torch.bool]


def _fill_base(dtype: torch.dtype, device: str):
    if dtype == torch.bool:
        cpu = (torch.arange(8192) % 3) == 0
    elif dtype.is_floating_point:
        cpu = (torch.arange(8192, dtype=torch.float32) % 251 / 8.0).to(dtype)
    else:
        cpu = (torch.arange(8192) % 251).to(dtype)
    return cpu, cpu.to(device)


def _fill_value(dtype: torch.dtype):
    return True if dtype == torch.bool else 7


@pytest.mark.parametrize("dtype", _FILL_DTYPES)
@pytest.mark.parametrize("rank", [1, 2, 3, 4, 5, 6, 7, 8])
def test_fill_scalar_ranks_one_through_eight(mojo_device, dtype, rank):
    shape = (2,) * (rank - 1) + (5,)
    numel = 2 ** (rank - 1) * 5
    cpu, device = _fill_base(dtype, mojo_device)
    value = _fill_value(dtype)
    device[:numel].view(shape).fill_(value)
    cpu[:numel].view(shape).fill_(value)
    assert torch.equal(device.cpu(), cpu)


@pytest.mark.parametrize("dtype", _FILL_DTYPES)
def test_fill_scalar_lengths_exercise_the_scalar_tail(mojo_device, dtype):
    value = _fill_value(dtype)
    for length in (0, 1, 2, 3, 5, 7, 9, 15, 16, 17, 31, 33, 4099):
        cpu, device = _fill_base(dtype, mojo_device)
        device[:length].fill_(value)
        cpu[:length].fill_(value)
        assert torch.equal(device.cpu(), cpu), length


@pytest.mark.parametrize("dtype", _FILL_DTYPES)
def test_fill_scalar_every_alignment_phase(mojo_device, dtype):
    """Offsets 0..16 walk every 16-byte phase of the base address."""
    value = _fill_value(dtype)
    for offset in range(17):
        cpu, device = _fill_base(dtype, mojo_device)
        device[offset : offset + 333].fill_(value)
        cpu[offset : offset + 333].fill_(value)
        assert torch.equal(device.cpu(), cpu), offset


@pytest.mark.parametrize("dtype", _FILL_DTYPES)
@pytest.mark.parametrize(
    "layout",
    ["transpose", "column", "gapped", "block", "permuted", "expanded", "scalar"],
)
def test_fill_scalar_strided_layouts(mojo_device, dtype, layout):
    def view(t):
        if layout == "transpose":
            return t[:6000].view(60, 100).t()
        if layout == "column":
            return t[:6000].view(60, 100).t()[:, 3]
        if layout == "gapped":
            return t[:6000:7]
        if layout == "block":
            return t[:6000].view(60, 100)[10:50, 20:90]
        if layout == "permuted":
            return t[:5040].view(6, 7, 8, 15).permute(2, 0, 3, 1)
        if layout == "expanded":
            return t[:100].view(1, 100).expand(7, 100)
        return t[:1].view(())

    value = _fill_value(dtype)
    cpu, device = _fill_base(dtype, mojo_device)
    view(device).fill_(value)
    view(cpu).fill_(value)
    assert torch.equal(device.cpu(), cpu)


@pytest.mark.parametrize("dtype", _FILL_DTYPES)
def test_zero__through_a_transposed_view(mojo_device, dtype):
    cpu, device = _fill_base(dtype, mojo_device)
    device[:1234].view(2, 617).t().zero_()
    cpu[:1234].view(2, 617).t().zero_()
    assert torch.equal(device.cpu(), cpu)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int64, torch.int32, torch.uint8]
)
def test_fill_scalar_value_conversion_matches_cpu(mojo_device, dtype):
    """How a python value narrows into the destination dtype is ATen's rule,
    not the kernel's choice."""
    values = [0, 7, True, False, 0.0, 2.75]
    if dtype != torch.uint8:
        values += [-1.5, -3]
    if dtype.is_floating_point:
        values += [float("inf"), float("-inf"), float("nan")]
    for value in values:
        cpu = torch.zeros(9, dtype=dtype)
        device = cpu.to(mojo_device)
        cpu.fill_(value)
        device.fill_(value)
        got = device.cpu()
        if dtype.is_floating_point:
            assert torch.equal(got.isnan(), cpu.isnan()), value
            assert torch.equal(got.nan_to_num(0.0), cpu.nan_to_num(0.0)), value
        else:
            assert torch.equal(got, cpu), value


def test_fill_through_a_transposed_view_keeps_the_allocation(mojo_device):
    x = torch.zeros(4, 6, dtype=torch.bool).to(mojo_device)
    view = x.t()
    before = x.data_ptr()
    filled = view.fill_(True)
    assert filled.data_ptr() == before
    assert bool(x.cpu().all())


def _sqrt_ieee_input(count: int) -> torch.Tensor:
    edges = torch.tensor(
        [
            0,
            0x80000000,
            1,
            2,
            3,
            0x007FFFFF,
            0x00800000,
            0x00800001,
            0x80000001,
            0x807FFFFF,
            0x80800000,
            0x3F7FFFFF,
            0x3F800000,
            0x3F800001,
            0x3FFFFFFF,
            0x40000000,
            0x40000001,
            0x407FFFFF,
            0x40800000,
            0x40800001,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0x7F800001,
            0xFFC00001,
            0xBF800000,
        ],
        dtype=torch.int64,
    )
    indices = torch.arange(count, dtype=torch.int64)
    bits = (indices * 2654435761 + 13) & 0xFFFFFFFF
    slots = indices % 32
    special = slots < edges.numel()
    bits[special] = edges[slots[special]]
    return bits.to(torch.int32).view(torch.float32)


_SQRT_SPANS = (
    [(1029, src, dst, False) for src in range(4) for dst in range(4)]
    + [
        (size, 0, 0, False)
        for size in [
            0,
            1,
            2,
            3,
            4,
            5,
            7,
            15,
            16,
            17,
            255,
            256,
            257,
            1023,
            1024,
            1025,
            1026,
            1027,
            1028,
            4095,
            4096,
            4097,
        ]
    ]
    + [
        (size, offset, offset, alias)
        for offset in range(4)
        for size, alias in [(357 * 789, False), (357 * 789, True), (2, True)]
    ]
    + [(size, 0, 0, False) for size in [800, 3840000, 5120000, 40206400]]
)


@pytest.mark.parametrize("count,source_offset,destination_offset,alias", _SQRT_SPANS)
def test_sqrt_ieee_spans(
    mojo_gpu: str, count: int, source_offset: int, destination_offset: int, alias: bool
):
    host = _sqrt_ieee_input(count + 16)
    source_storage = host.to(mojo_gpu)
    source = source_storage[source_offset : source_offset + count]
    storage = source_storage if alias else torch.full_like(source_storage, 17)
    out = storage[destination_offset : destination_offset + count]
    version = out._version
    returned = source.sqrt_() if alias else torch.sqrt(source, out=out)
    assert returned.data_ptr() == out.data_ptr()
    assert out._version == version + 1
    expected = host.clone() if alias else torch.full_like(host, 17)
    expected[destination_offset : destination_offset + count] = (
        host[source_offset : source_offset + count].double().sqrt().float()
    )
    actual = storage.cpu()
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual.view(torch.int32)[~nan], expected.view(torch.int32)[~nan])
    if not alias:
        assert torch.equal(
            source_storage.cpu().view(torch.int32), host.view(torch.int32)
        )


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "strided"])
def test_sqrt_dtype_layout_fallback(mojo_gpu: str, dtype: torch.dtype, layout: str):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Metal does not support float64")
    host = torch.arange(1, 106, dtype=torch.float32).reshape(7, 15).to(dtype)
    source = host.to(mojo_gpu)
    if layout == "transposed":
        host, source = host.t(), source.t()
    elif layout == "strided":
        host, source = host[:, ::2], source[:, ::2]
    if dtype == torch.float64:
        # The pre-existing SqrtSpec contract supports f32/f16/bf16 only.
        with pytest.raises(NotImplementedError, match="dtype float64 is not supported"):
            torch.sqrt(source)
        return
    expected = host.double().sqrt().to(dtype)
    torch.testing.assert_close(torch.sqrt(source).cpu(), expected, rtol=0, atol=0)
    storage = torch.full(
        (source.shape[0], source.shape[1] * 2), -17.0, dtype=dtype, device=mojo_gpu
    )
    output = storage[:, ::2]
    torch.sqrt(source, out=output)
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert torch.all(storage[:, 1::2].cpu() == -17)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
def test_log2_domain(mojo_gpu: str, dtype: torch.dtype):
    if dtype == torch.float64:
        skip_if_metal(mojo_gpu, "Metal does not support float64")
    data = torch.tensor(
        [-1.0, -0.0, 0.0, 0.125, 1.0, 2.0, 3.0, float("inf"), float("nan")], dtype=dtype
    )
    result = torch.log2(data.to(mojo_gpu))
    assert result.device.type == "mojo"
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(
        result.cpu(), torch.log2(data), rtol=rtol, atol=atol, equal_nan=True
    )


@pytest.mark.parametrize("shape", [(), (0,), (3, 5)])
def test_log2_shapes(mojo_gpu: str, shape: tuple[int, ...]):
    data = torch.full(shape, 8.0)
    torch.testing.assert_close(torch.log2(data.to(mojo_gpu)).cpu(), torch.log2(data))


def test_log2_noncontiguous(mojo_gpu: str):
    data = torch.arange(1, 36, dtype=torch.float32).reshape(5, 7)
    result = torch.log2(data.to(mojo_gpu).transpose(0, 1))
    torch.testing.assert_close(result.cpu(), torch.log2(data.transpose(0, 1)))


@pytest.mark.parametrize("count", [1, 4, 5, 7, 8, 9, 1023, 1024, 1025, 357 * 789])
@pytest.mark.parametrize("input_offset,output_offset", [(0, 0), (1, 0), (0, 1), (3, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_log2_offset_tail(
    mojo_gpu: str, count: int, input_offset: int, output_offset: int, dtype: torch.dtype
):
    data = torch.linspace(0.125, 8.0, count + input_offset + 3).to(dtype)
    source = data.to(mojo_gpu)[input_offset : input_offset + count]
    storage = torch.full((count + output_offset + 3,), -12345.0, dtype=dtype).to(
        mojo_gpu
    )
    output = storage[output_offset : output_offset + count]
    returned = torch.log2(source, out=output)
    assert returned.data_ptr() == output.data_ptr()
    expected = torch.full((count + output_offset + 3,), -12345.0, dtype=dtype)
    expected[output_offset : output_offset + count] = torch.log2(
        data[input_offset : input_offset + count]
    )
    rtol, atol = (0.008, 1e-5) if dtype == torch.bfloat16 else (2e-6, 2e-6)
    torch.testing.assert_close(storage.cpu(), expected, rtol=rtol, atol=atol)


def test_log2_bfloat16_normals_and_domain(mojo_gpu: str):
    normals = torch.arange(0x0080, 0x7F80, dtype=torch.int16).view(torch.bfloat16)
    domain = torch.tensor(
        [0.0, -0.0, -1.0, 1.0, 2.0, float("inf"), -float("inf"), float("nan"), 0.5],
        dtype=torch.bfloat16,
    )
    data = torch.cat((normals, -normals, domain))
    torch.testing.assert_close(
        torch.log2(data.to(mojo_gpu)).cpu(),
        torch.log2(data),
        rtol=0.008,
        atol=1e-5,
        equal_nan=True,
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool])
def test_log2_integral_promotion(mojo_gpu: str, dtype: torch.dtype):
    data = torch.tensor([0, 1, 2, 8], dtype=dtype)
    result = torch.log2(data.to(mojo_gpu))
    assert result.dtype == torch.get_default_dtype()
    torch.testing.assert_close(result.cpu(), torch.log2(data))


def test_log2_out(mojo_gpu: str):
    data = torch.tensor([0.5, 1, 2, 8]).to(mojo_gpu)
    out = torch.empty(4).to(mojo_gpu)
    result = torch.log2(data, out=out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out.cpu(), torch.tensor([-1.0, 0.0, 1.0, 3.0]))
