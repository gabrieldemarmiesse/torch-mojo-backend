"""Binary arithmetic on the native mojo device: add/sub/mul/div, pow,
maximum/minimum, remainder, floor_divide, lerp, addcmul/addcdiv, clamp and
the logical/bitwise ops, with their in-place and `out=` variants.

Every check compares against the same computation on CPU torch through the
public API only; `CallChecker` (or `native.op_count` for the ops with no
`aten_functions` twin) asserts the native op actually ran.
"""

import contextlib
import math

import pytest
import torch

from tests.native.conftest import skip_if_metal
from torch_mojo_backend import aten_functions, native


@contextlib.contextmanager
def native_ran(*op_names: str):
    """Assert at least one of `op_names` ran as a native boxed kernel."""
    native.op_counting(True)
    before = {name: native.op_count(name) for name in op_names}
    yield
    assert any(native.op_count(name) > before[name] for name in op_names), (
        f"none of {op_names} ran natively"
    )


def _both(shape, dtype, device, *, low=1, high=9):
    """The same tensor on CPU and on the mojo device."""
    if dtype.is_floating_point:
        cpu = torch.randn(shape, dtype=torch.float32).to(dtype)
    else:
        cpu = torch.randint(low, high, shape, dtype=dtype)
    return cpu, cpu.to(device)


# --------------------------------------------------------------------------
# add / sub / mul / div
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_add_sub_mul_tensor(mojo_device, dtype, call_checker):
    call_checker.register(aten_functions.aten_add)
    a_cpu, a = _both((4, 5), dtype, mojo_device)
    b_cpu, b = _both((4, 5), dtype, mojo_device)
    torch.testing.assert_close((a + b).cpu(), a_cpu + b_cpu)
    torch.testing.assert_close((a - b).cpu(), a_cpu - b_cpu)
    torch.testing.assert_close((a * b).cpu(), a_cpu * b_cpu)


def test_div_float(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_div)
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    torch.testing.assert_close((a / b).cpu(), a_cpu / b_cpu)


def test_div_int_promotes_to_float(mojo_device):
    a_cpu, a = _both((6,), torch.int64, mojo_device)
    b_cpu, b = _both((6,), torch.int64, mojo_device)
    with native_ran("aten::div.Tensor"):
        out = a / b
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), a_cpu / b_cpu)
    with native_ran("aten::div.Tensor"):
        out_scalar = a / 2
    torch.testing.assert_close(out_scalar.cpu(), a_cpu / 2)


@pytest.mark.parametrize(
    "denominator_dtype", [torch.float16, torch.bfloat16, torch.float32]
)
def test_div_int_by_float_takes_the_denominator_dtype(mojo_gpu, denominator_dtype):
    """`torch.result_type(int64_tensor, half_tensor)` is half, not float32:
    true division promotes to a float, but to the RIGHT float."""
    a_cpu, a = _both((6,), torch.int64, mojo_gpu)
    b_cpu, b = _both((6,), denominator_dtype, mojo_gpu)
    with native_ran("aten::div.Tensor"):
        out = a / b
    assert out.dtype == torch.result_type(a_cpu, b_cpu) == denominator_dtype
    torch.testing.assert_close(out.cpu(), a_cpu / b_cpu, atol=1e-2, rtol=1e-2)


def test_div_int_by_float_scalar_uses_the_default_dtype(mojo_gpu):
    a_cpu, a = _both((6,), torch.int32, mojo_gpu)
    with native_ran("aten::div.Tensor"):
        out = a / 2.5
    assert out.dtype == torch.get_default_dtype()
    torch.testing.assert_close(out.cpu(), a_cpu / 2.5)


def test_div_int_honours_a_changed_default_dtype(mojo_gpu):
    """`promote_integer_inputs_to_float` reads `torch.get_default_dtype()`.
    The divide kernel does cover float64, but data_movement_ops' CAST_DTYPES
    does not, so an integer numerator cannot be lifted into it: the op must
    decline rather than quietly hand back float32."""
    _, a = _both((6,), torch.int64, mojo_gpu)
    torch.set_default_dtype(torch.float64)
    try:
        with pytest.raises(NotImplementedError, match="a cast from dtype"):
            a / a
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_add_sub_alpha_reduced_precision_rounds_like_cpu(mojo_gpu, dtype):
    """ATen's CPU add/sub kernel takes `alpha` as a scalar_t (rounded to the
    half type) and its scalar loop computes `a + alpha * b` one rounding at a
    time; that is the documented policy here, checked bit for bit. CPU torch
    itself is not a bit-exact oracle: its vectorized loop fuses the multiply
    and add into one rounding, so which loop a given element hits depends on
    the tensor size and the CPU's vector width. The conformance suite is the
    CPU-parity check, on the small samples where the two agree."""
    a = (torch.arange(64, dtype=torch.float32) / 7.0 - 4.0).to(dtype)
    b = (torch.arange(64, dtype=torch.float32) / 3.0 - 10.0).to(dtype)
    alpha = 1.0 / 3.0
    alpha_r = torch.tensor(alpha).to(dtype).float()  # alpha as a scalar_t
    ad, bd = a.to(mojo_gpu), b.to(mojo_gpu)

    got = torch.add(ad, bd, alpha=alpha)
    assert got.dtype == dtype
    expect = (a.float() + (alpha_r * b.float()).to(dtype).float()).to(dtype)
    torch.testing.assert_close(got.cpu(), expect, atol=0, rtol=0)
    got_sub = torch.sub(ad, bd, alpha=alpha)
    expect_sub = (a.float() - (alpha_r * b.float()).to(dtype).float()).to(dtype)
    torch.testing.assert_close(got_sub.cpu(), expect_sub, atol=0, rtol=0)
    # alpha kept in float32 gives a DIFFERENT tensor: without this the test
    # would pass on the implementation it is meant to reject.
    unrounded = (a.float() + (alpha * b.float()).to(dtype).float()).to(dtype)
    assert not torch.equal(unrounded, got.cpu())


def test_inplace_add_rejects_partial_overlap(mojo_gpu):
    """`x[1:].add_(x[:-1])` is a read/write race over one storage: ATen
    refuses it, and so must a backend that hands the kernel raw pointers."""
    x = torch.arange(8, dtype=torch.float32, device=mojo_gpu)
    with pytest.raises(RuntimeError, match="single memory location"):
        x[1:].add_(x[:-1])
    with pytest.raises(RuntimeError, match="single memory location"):
        x[:-1].mul_(x[1:])
    # The two allowed shapes still work: identical views and disjoint ones.
    x[:4].add_(x[:4])
    x[:2].add_(x[6:])


def test_add_alpha(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_add)
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    torch.testing.assert_close(
        torch.add(a, b, alpha=-2.5).cpu(), torch.add(a_cpu, b_cpu, alpha=-2.5)
    )
    torch.testing.assert_close(
        torch.sub(a, b, alpha=3).cpu(), torch.sub(a_cpu, b_cpu, alpha=3)
    )


def test_scalar_operands(mojo_device):
    """`x + 2` reaches the backend as add.Tensor with a wrapped 0-d CPU
    tensor; the scalar routes have to recognise it."""
    a_cpu, a = _both((5,), torch.float32, mojo_device)
    with native_ran("aten::add.Tensor"):
        torch.testing.assert_close((a + 2).cpu(), a_cpu + 2)
    torch.testing.assert_close((2 + a).cpu(), 2 + a_cpu)
    torch.testing.assert_close((a * 1.5).cpu(), a_cpu * 1.5)
    torch.testing.assert_close((a - 0.25).cpu(), a_cpu - 0.25)
    torch.testing.assert_close((a / 4).cpu(), a_cpu / 4)
    torch.testing.assert_close((a**2).cpu(), a_cpu**2)


def test_int_scalar_operands(mojo_device):
    a_cpu, a = _both((5,), torch.int64, mojo_device)
    with native_ran("aten::add.Tensor"):
        torch.testing.assert_close((a + 3).cpu(), a_cpu + 3)
    torch.testing.assert_close((a * 3).cpu(), a_cpu * 3)
    torch.testing.assert_close((a - 3).cpu(), a_cpu - 3)


def test_broadcasting(mojo_device):
    a_cpu, a = _both((3, 1, 5), torch.float32, mojo_device)
    b_cpu, b = _both((4, 5), torch.float32, mojo_device)
    with native_ran("aten::mul.Tensor"):
        out = a * b
    assert tuple(out.shape) == (3, 4, 5)
    torch.testing.assert_close(out.cpu(), a_cpu * b_cpu)


def test_strided_operands(mojo_device):
    a_cpu, a = _both((4, 6), torch.float32, mojo_device)
    b_cpu, b = _both((6, 4), torch.float32, mojo_device)
    with native_ran("aten::add.Tensor"):
        out = a + b.t()
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu.t())


def test_rank5_equal_shapes(mojo_device):
    """Above rank 4 the kernel takes a flat pass: equal shapes, contiguous."""
    a_cpu, a = _both((2, 2, 2, 2, 3), torch.float32, mojo_device)
    b_cpu, b = _both((2, 2, 2, 2, 3), torch.float32, mojo_device)
    with native_ran("aten::mul.Tensor"):
        out = a * b
    torch.testing.assert_close(out.cpu(), a_cpu * b_cpu)


def test_mixed_dtype_promotion(mojo_device):
    a_cpu, a = _both((4, 4), torch.float32, mojo_device)
    b_cpu, b = _both((4, 4), torch.bfloat16, mojo_device)
    with native_ran("aten::add.Tensor"):
        out = a + b
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu)
    i_cpu, i = _both((4, 4), torch.int32, mojo_device)
    j_cpu, j = _both((4, 4), torch.int64, mojo_device)
    mixed = i + j
    assert mixed.dtype == torch.int64
    torch.testing.assert_close(mixed.cpu(), i_cpu + j_cpu)


def test_non_contiguous_mixed_dtype(mojo_device):
    a_cpu, a = _both((4, 6), torch.float32, mojo_device)
    b_cpu, b = _both((6, 4), torch.bfloat16, mojo_device)
    out = a + b.t()
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu.t())


def test_unsupported_mix_declines(mojo_device):
    """A promotion the port does not cover declines (NotImplementedError),
    exactly where the old fast path returned NOT_HANDLED."""
    _, a = _both((4,), torch.float32, mojo_device)
    _, b = _both((4,), torch.int64, mojo_device)
    with pytest.raises(NotImplementedError):
        a + b


def test_shape_mismatch_raises(mojo_device):
    _, a = _both((4,), torch.float32, mojo_device)
    _, b = _both((5,), torch.float32, mojo_device)
    with pytest.raises(RuntimeError):
        a + b


# --------------------------------------------------------------------------
# in-place
# --------------------------------------------------------------------------


def test_inplace_tensor(mojo_device):
    a_cpu, a = _both((4, 5), torch.float32, mojo_device)
    b_cpu, b = _both((4, 5), torch.float32, mojo_device)
    with native_ran("aten::add_.Tensor"):
        a.add_(b)
    a_cpu.add_(b_cpu)
    torch.testing.assert_close(a.cpu(), a_cpu)
    with native_ran("aten::mul_.Tensor"):
        a.mul_(b)
    a_cpu.mul_(b_cpu)
    torch.testing.assert_close(a.cpu(), a_cpu)
    with native_ran("aten::sub_.Tensor"):
        a.sub_(b)
    a_cpu.sub_(b_cpu)
    torch.testing.assert_close(a.cpu(), a_cpu)


def test_inplace_scalar(mojo_device):
    a_cpu, a = _both((7,), torch.float32, mojo_device)
    with native_ran("aten::add_.Tensor"):
        a.add_(1.5)
    a_cpu.add_(1.5)
    torch.testing.assert_close(a.cpu(), a_cpu)
    a.mul_(2.0)
    a_cpu.mul_(2.0)
    torch.testing.assert_close(a.cpu(), a_cpu)
    a.sub_(0.5)
    a_cpu.sub_(0.5)
    torch.testing.assert_close(a.cpu(), a_cpu)
    a.add_(b := torch.full((7,), 2.0).to(mojo_device), alpha=3)
    a_cpu.add_(torch.full((7,), 2.0), alpha=3)
    assert b.shape == a.shape
    torch.testing.assert_close(a.cpu(), a_cpu)


def test_inplace_on_a_view(mojo_device):
    a_cpu, a = _both((4, 5), torch.float32, mojo_device)
    b_cpu, b = _both((4,), torch.float32, mojo_device)
    a[:, 1].add_(b)
    a_cpu[:, 1].add_(b_cpu)
    torch.testing.assert_close(a.cpu(), a_cpu)


# --------------------------------------------------------------------------
# out=
# --------------------------------------------------------------------------


def test_out_variants(mojo_device):
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    dest = torch.empty((3, 4), device=mojo_device)
    with native_ran("aten::add.out"):
        torch.add(a, b, out=dest)
    torch.testing.assert_close(dest.cpu(), a_cpu + b_cpu)
    with native_ran("aten::mul.out"):
        torch.mul(a, b, out=dest)
    torch.testing.assert_close(dest.cpu(), a_cpu * b_cpu)
    with native_ran("aten::sub.out"):
        torch.sub(a, b, out=dest)
    torch.testing.assert_close(dest.cpu(), a_cpu - b_cpu)
    with native_ran("aten::div.out"):
        torch.div(a, b, out=dest)
    torch.testing.assert_close(dest.cpu(), a_cpu / b_cpu)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    ("n", "in_offset", "out_offset"),
    [
        (16000, 0, 0),  # 16-byte aligned bases, no tail: the vector lanes
        (16000, 8, 8),  # aligned by offset (8 elements = 16 B in bf16, 32 B in fp32)
        (16003, 0, 0),  # ragged length: scalar lanes
        (16000, 1, 0),  # unaligned input, aligned output
        (16000, 0, 7),  # aligned input, unaligned output
        (16000, 4, 1),
    ],
)
def test_scalar_mul_out_into_bucket_view(mojo_device, dtype, n, in_offset, out_offset):
    """DDP's reducer: `mul_out(bucket_view, grad, 1/world)` with the view at
    whatever element offset the previous parameter left. Both bases 16-byte
    aligned with no tail takes the vector lanes; anything else the scalar
    ones."""
    src_cpu, src = _both((n + 16,), dtype, mojo_device)
    grad_cpu, grad = src_cpu[in_offset : in_offset + n], src[in_offset : in_offset + n]
    bucket = torch.zeros(n + 16, dtype=dtype, device=mojo_device)
    with native_ran("aten::mul.out"):
        torch.mul(grad, 1.0 / 16, out=bucket[out_offset : out_offset + n])
    expected = torch.zeros(n + 16, dtype=dtype)
    expected[out_offset : out_offset + n] = grad_cpu * (1.0 / 16)
    torch.testing.assert_close(bucket.cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_scalar_mul_out_aliasing_self(mojo_device, dtype):
    """`torch.mul(x, s, out=x)`: the exact alias the direct-write route must
    accept (a flat elementwise loop reads each element before writing it)."""
    x_cpu, x = _both((16000,), dtype, mojo_device)
    with native_ran("aten::mul.out"):
        torch.mul(x, 0.5, out=x)
    torch.testing.assert_close(x.cpu(), x_cpu * 0.5)


_SCALAR_MUL_PATTERNS = [
    0,
    0x80000000,
    1,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x00800000,
    0x80800000,
    0x3F800000,
    0x3F800001,
    0x3F7FFFFF,
    0xBF800001,
    0x7F7FFFFF,
    0xFF7FFFFF,
    0x7F800000,
    0xFF800000,
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0xFFFFFFFF,
]


def _check_scalar_mul_peel(
    device, size, source_offset, destination_offset, bits, inplace
):
    indices = torch.arange(size + 16, dtype=torch.int64)
    source_bits = indices * 2654435761
    for i, pattern in enumerate(_SCALAR_MUL_PATTERNS):
        source_bits[indices % 32 == i] = pattern
    source_bits = source_bits.to(torch.int32)
    host = source_bits.view(torch.float32)
    scalar = (
        torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(torch.float32).item()
    )
    source = host.to(device)
    destination = source if inplace else torch.full_like(source, 17.0)
    expected = host.clone() if inplace else torch.full_like(host, 17.0)
    value = source[source_offset : source_offset + size]
    output = destination[destination_offset : destination_offset + size]
    before, pointer = output._version, output.data_ptr()
    assert torch.mul(value, scalar, out=output) is output
    expected[destination_offset : destination_offset + size] = (
        host[source_offset : source_offset + size] * scalar
    )
    actual = destination.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    finite_or_inf = ~torch.isnan(expected)
    torch.testing.assert_close(
        actual.view(torch.int32)[finite_or_inf],
        expected.view(torch.int32)[finite_or_inf],
        rtol=0,
        atol=0,
    )
    assert output._version == before + 1
    assert output.data_ptr() == pointer
    if not inplace:
        torch.testing.assert_close(
            source.cpu().view(torch.int32), source_bits, rtol=0, atol=0
        )


@pytest.mark.parametrize("source_offset", range(4))
@pytest.mark.parametrize("destination_offset", range(4))
def test_scalar_mul_peel_alignment(mojo_gpu, source_offset, destination_offset):
    _check_scalar_mul_peel(
        mojo_gpu, 1025, source_offset, destination_offset, 0x41FCFB72, False
    )


@pytest.mark.parametrize(
    "size",
    [
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
        1029,
    ],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_scalar_mul_peel_boundaries(mojo_gpu, size, inplace):
    offset, scalar = (3, 0xBF800001) if inplace else (1, 0x3F800001)
    _check_scalar_mul_peel(mojo_gpu, size, offset, offset, scalar, inplace)


@pytest.mark.parametrize(
    "bits",
    [
        0,
        0x80000000,
        0x3F800000,
        0xBF800000,
        0x3F000000,
        0x40000000,
        1,
        0x00800000,
        0x7F800000,
        0xFF800000,
        0x7FC00000,
        0x7F7FFFFF,
    ],
)
@pytest.mark.parametrize("inplace", [False, True])
def test_scalar_mul_peel_special_values(mojo_gpu, bits, inplace):
    offset = int(inplace)
    _check_scalar_mul_peel(mojo_gpu, 1025, offset, offset, bits, inplace)


@pytest.mark.parametrize("offset", [0, 1, 3])
@pytest.mark.parametrize("inplace", [False, True])
def test_scalar_mul_peel_large(mojo_gpu, offset, inplace):
    _check_scalar_mul_peel(
        mojo_gpu,
        357 * 789,
        offset,
        offset,
        0xBF800001 if inplace else 0x41FCFB72,
        inplace,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("n,offset", [(0, 1), (1, 0), (357 * 789, 1)])
def test_mul_inplace_device_scalar_preserves_storage(mojo_gpu, dtype, n, offset):
    cpu = (torch.arange(n + offset + 3, dtype=torch.float32) % 29 - 14).to(dtype)
    base = cpu.to(mojo_gpu)
    value = base[offset : offset + n]
    scalar = torch.tensor(0.375, dtype=dtype, device=mojo_gpu)
    scalar_version, version, ptr = scalar._version, value._version, value.data_ptr()
    result = value.mul_(scalar)
    cpu[offset : offset + n].mul_(0.375)
    assert result is value
    assert value.data_ptr() == ptr
    assert value._version == version + 1
    assert scalar._version == scalar_version
    assert scalar.cpu().item() == 0.375
    torch.testing.assert_close(base.cpu(), cpu, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["strided", "promoted", "broadcast", "alias"])
def test_mul_inplace_device_scalar_fallbacks(mojo_gpu, layout):
    cpu = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    value = cpu.to(mojo_gpu)
    if layout == "strided":
        cpu, value = cpu[:, ::2], value[:, ::2]
    if layout == "alias":
        with pytest.raises(RuntimeError, match="single memory location|overlap"):
            value.mul_(value[0, 0])
        torch.testing.assert_close(value.cpu(), cpu)
        return
    dtype = torch.float16 if layout == "promoted" else torch.float32
    scalar_cpu = torch.tensor(0.375, dtype=dtype)
    if layout == "broadcast":
        scalar_cpu = scalar_cpu.expand(3, 1).clone()
    scalar = scalar_cpu.to(mojo_gpu)
    version = value._version
    value.mul_(scalar)
    cpu.mul_(scalar_cpu)
    assert value._version == version + 1
    torch.testing.assert_close(value.cpu(), cpu)


@pytest.mark.parametrize("divisor", [3.0, -0.03162277660168379, 7, 1e-20, 1e20])
@pytest.mark.parametrize("out_kind", ["functional", "offset", "alias", "strided"])
def test_div_host_scalar_direct(mojo_gpu, divisor, out_kind):
    cpu = torch.linspace(-17, 19, 359, dtype=torch.float32)
    source = cpu.to(mojo_gpu)
    source_before = source.cpu()
    expected = cpu / divisor
    if out_kind == "functional":
        result = source / divisor
    else:
        backing = torch.full((2 * cpu.numel() + 3,), 71.0, device=mojo_gpu)
        if out_kind == "alias":
            result = source
        elif out_kind == "strided":
            result = backing[1 : 2 * cpu.numel() + 1 : 2]
        else:
            result = backing[1 : cpu.numel() + 1]
        version, ptr = result._version, result.data_ptr()
        assert torch.div(source, divisor, out=result) is result
        assert result.data_ptr() == ptr
        assert result._version == version + 1
        if out_kind != "alias":
            expected_backing = torch.full_like(backing.cpu(), 71.0)
            if out_kind == "strided":
                expected_backing[1 : 2 * cpu.numel() + 1 : 2] = expected
            else:
                expected_backing[1 : cpu.numel() + 1] = expected
            torch.testing.assert_close(backing.cpu(), expected_backing)
    torch.testing.assert_close(result.cpu(), expected)
    if out_kind != "alias":
        torch.testing.assert_close(source.cpu(), source_before, rtol=0, atol=0)


@pytest.mark.parametrize("divisor", [0.0, -0.0, float("inf"), -float("inf")])
def test_div_host_scalar_special_values(mojo_gpu, divisor):
    cpu = torch.tensor([-float("inf"), -3.0, -0.0, 0.0, 5.0, float("inf")])
    torch.testing.assert_close(
        (cpu.to(mojo_gpu) / divisor).cpu(), cpu / divisor, equal_nan=True
    )


def test_out_resizes(mojo_device):
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    dest = torch.empty(0, device=mojo_device)
    torch.add(a, b, out=dest)
    assert tuple(dest.shape) == (3, 4)
    torch.testing.assert_close(dest.cpu(), a_cpu + b_cpu)


def test_out_aliasing_an_input(mojo_device):
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    torch.add(a, b, out=a)
    torch.testing.assert_close(a.cpu(), a_cpu + b_cpu)


def test_out_strided(mojo_device):
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    dest = torch.zeros((3, 8), device=mojo_device)[:, ::2]
    torch.add(a, b, out=dest)
    torch.testing.assert_close(dest.cpu(), a_cpu + b_cpu)


def test_div_out_mode(mojo_device):
    a_cpu, a = _both((6,), torch.float32, mojo_device)
    b_cpu, b = _both((6,), torch.float32, mojo_device)
    dest = torch.empty((6,), device=mojo_device)
    with native_ran("aten::div.out_mode"):
        torch.div(a, b, rounding_mode="floor", out=dest)
    torch.testing.assert_close(
        dest.cpu(), torch.div(a_cpu, b_cpu, rounding_mode="floor")
    )


# --------------------------------------------------------------------------
# div rounding modes, floor_divide, remainder
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["floor", "trunc"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_div_rounding_modes(mojo_device, mode, dtype):
    a_cpu, a = _both((8,), dtype, mojo_device, low=-9, high=9)
    b_cpu, b = _both((8,), dtype, mojo_device, low=1, high=5)
    with native_ran("aten::div.Tensor_mode"):
        out = torch.div(a, b, rounding_mode=mode)
    assert out.dtype == dtype
    torch.testing.assert_close(out.cpu(), torch.div(a_cpu, b_cpu, rounding_mode=mode))


def test_floor_divide(mojo_device):
    a_cpu, a = _both((8,), torch.int64, mojo_device, low=-9, high=9)
    b_cpu, b = _both((8,), torch.int64, mojo_device, low=1, high=5)
    with native_ran("aten::floor_divide"):
        out = a // b
    torch.testing.assert_close(out.cpu(), a_cpu // b_cpu)
    with native_ran("aten::floor_divide", "aten::floor_divide.Scalar"):
        out_scalar = a // 3
    torch.testing.assert_close(out_scalar.cpu(), a_cpu // 3)


def test_remainder(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_remainder)
    a_cpu, a = _both((8,), torch.float32, mojo_device)
    b_cpu, b = _both((8,), torch.float32, mojo_device, low=1, high=4)
    torch.testing.assert_close(torch.remainder(a, b).cpu(), a_cpu % b_cpu)
    torch.testing.assert_close(torch.remainder(a, 2.0).cpu(), a_cpu % 2.0)
    torch.testing.assert_close(
        torch.remainder(2.0, b).cpu(), torch.remainder(2.0, b_cpu)
    )


# --------------------------------------------------------------------------
# pow / maximum / minimum / clamp
# --------------------------------------------------------------------------


def test_pow(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_pow)
    # Positive base: the tensor-tensor exponent is real-valued only there
    # (and `abs` belongs to another op group, so build it on CPU).
    a_cpu = torch.rand(5) + 0.5
    a = a_cpu.to(mojo_device)
    torch.testing.assert_close(torch.pow(a, 2.0).cpu(), torch.pow(a_cpu, 2.0))
    e_cpu, e = _both((5,), torch.float32, mojo_device)
    with native_ran("aten::pow.Tensor_Tensor"):
        out = torch.pow(a, e)
    torch.testing.assert_close(out.cpu(), torch.pow(a_cpu, e_cpu))


def test_div_and_pow_float64(mojo_device, request):
    """float64 through the broadcast binary kernel.

    logic_ops' SPEC_BCAST_DTYPES carries float64, and `_binary_spec_into_go`
    asks of div/pow only that the dtype be floating -- so both ops run at
    full precision instead of declining. The scalar exponent takes that same
    broadcast route rather than elementwise_ops' PowScalarSpec, which is
    FLOAT_DTYPES only (and would raise, not narrow, on a float64 operand).

    `call_checker` is fetched lazily (not a plain fixture argument): its
    teardown unconditionally requires `register` to have been called, and a
    skip before that point must not have already forced that fixture's
    setup/teardown into existence.
    """
    skip_if_metal(mojo_device, "float64 is not supported on Apple GPU")
    call_checker = request.getfixturevalue("call_checker")
    call_checker.register(aten_functions.aten_div)
    base_cpu = torch.rand(4, 5, dtype=torch.float64) + 0.5
    other_cpu = torch.rand(4, 5, dtype=torch.float64) + 0.5
    base, other = base_cpu.to(mojo_device), other_cpu.to(mojo_device)

    got = base / other
    assert got.dtype == torch.float64
    torch.testing.assert_close(got.cpu(), base_cpu / other_cpu)
    torch.testing.assert_close((base / 2.5).cpu(), base_cpu / 2.5)

    with native_ran("aten::pow.Tensor_Tensor"):
        out = torch.pow(base, other)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out.cpu(), torch.pow(base_cpu, other_cpu))
    with native_ran("aten::pow.Tensor_Scalar"):
        scalar = torch.pow(base, 7.3)
    assert scalar.dtype == torch.float64
    torch.testing.assert_close(scalar.cpu(), torch.pow(base_cpu, 7.3))


def test_maximum_minimum(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_maximum)
    a_cpu, a = _both((4, 4), torch.float32, mojo_device)
    b_cpu, b = _both((4, 4), torch.float32, mojo_device)
    torch.testing.assert_close(torch.maximum(a, b).cpu(), torch.maximum(a_cpu, b_cpu))
    torch.testing.assert_close(torch.minimum(a, b).cpu(), torch.minimum(a_cpu, b_cpu))


def test_clamp(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_clamp)
    a_cpu, a = _both((10,), torch.float32, mojo_device)
    torch.testing.assert_close(a.clamp(-0.5, 0.5).cpu(), a_cpu.clamp(-0.5, 0.5))
    torch.testing.assert_close(a.clamp(min=0.0).cpu(), a_cpu.clamp(min=0.0))
    torch.testing.assert_close(a.clamp(max=0.0).cpu(), a_cpu.clamp(max=0.0))


# --------------------------------------------------------------------------
# the fast common-case route (`_b_fast_route` / `_b_fast_inplace_t`), against
# the cascade it stands in front of
# --------------------------------------------------------------------------


def _fast_operands(shape, dtype, device, layout):
    """One (cpu, device) pair per layout class the fast route sorts on."""
    if layout == "dense":
        return _both(shape, dtype, device)
    if layout == "transposed":
        cpu, dev = _both(tuple(reversed(shape)), dtype, device)
        return cpu.t(), dev.t()
    if layout == "strided":
        wide = tuple(shape[:-1]) + (shape[-1] * 2,)
        cpu, dev = _both(wide, dtype, device)
        return cpu[..., ::2], dev[..., ::2]
    if layout == "offset":
        flat_cpu, flat = _both((math.prod(shape) + 3,), dtype, device)
        return flat_cpu[3:].view(shape), flat[3:].view(shape)
    raise AssertionError(layout)


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int64]
)
@pytest.mark.parametrize("shape", [(0,), (), (5,), (3, 4), (2, 2, 2, 2, 2)])
@pytest.mark.parametrize("layout", ["dense", "transposed", "strided", "offset"])
def test_elementwise_tensor_routes_match_cpu(mojo_device, dtype, shape, layout):
    """Functional/in-place/out= add, sub and mul over every layout class."""
    if layout != "dense" and len(shape) != 2:
        pytest.skip("only the rank-2 shapes have a meaningful transpose/stride")
    a_cpu, a = _fast_operands(shape, dtype, mojo_device, layout)
    b_cpu, b = _fast_operands(shape, dtype, mojo_device, layout)
    for op in ("add", "sub", "mul"):
        want = getattr(torch, op)(a_cpu, b_cpu)
        with native_ran(f"aten::{op}.Tensor"):
            got = getattr(torch, op)(a, b)
        torch.testing.assert_close(got.cpu(), want)
        if layout == "dense":
            # A non-dense input is a layout CPU torch propagates through
            # TensorIterator and this device does not: its broadcast route
            # has always returned a contiguous result.
            assert got.stride() == want.stride()
        # out= into a fresh buffer, and out= aliasing each operand
        dest = torch.empty_like(a)
        assert getattr(torch, op)(a, b, out=dest) is dest
        torch.testing.assert_close(dest.cpu(), want)
        for which in ("self", "other"):
            alias_cpu = (a_cpu if which == "self" else b_cpu).clone()
            alias = (a if which == "self" else b).clone()
            getattr(torch, op)(
                *(alias, b) if which == "self" else (a, alias), out=alias
            )
            getattr(torch, op)(
                *(alias_cpu, b_cpu) if which == "self" else (a_cpu, alias_cpu),
                out=alias_cpu,
            )
            torch.testing.assert_close(alias.cpu(), alias_cpu)
        inplace_cpu, inplace = a_cpu.clone(), a.clone()
        with native_ran(f"aten::{op}_.Tensor"):
            assert getattr(inplace, op + "_")(b) is inplace
        getattr(inplace_cpu, op + "_")(b_cpu)
        torch.testing.assert_close(inplace.cpu(), inplace_cpu)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "other", ["equal", "row", "column", "zero_dim", "expanded", "wrapped", "scalar"]
)
def test_inplace_binary_broadcast_operands(mojo_device, dtype, other):
    """`self op= other` for every shape of `other` the kernel can broadcast."""
    a_cpu, a = _both((4, 6), dtype, mojo_device)
    if other == "equal":
        b_cpu, b = _both((4, 6), dtype, mojo_device)
    elif other == "row":
        b_cpu, b = _both((6,), dtype, mojo_device)
    elif other == "column":
        b_cpu, b = _both((4, 1), dtype, mojo_device)
    elif other == "zero_dim":
        b_cpu, b = _both((), dtype, mojo_device)
    elif other == "expanded":
        base_cpu, base = _both((1, 6), dtype, mojo_device)
        b_cpu, b = base_cpu.expand(4, 6), base.expand(4, 6)
    elif other == "wrapped":
        b_cpu = b = torch.tensor(0.5, dtype=dtype)  # a 0-d CPU wrapped number
    else:
        b_cpu = b = 0.5
    for op in ("mul", "add", "sub"):
        got_cpu, got = a_cpu.clone(), a.clone()
        version = got._version
        with native_ran(f"aten::{op}_.Tensor"):
            assert getattr(got, op + "_")(b) is got
        assert got._version == version + 1
        getattr(got_cpu, op + "_")(b_cpu)
        torch.testing.assert_close(got.cpu(), got_cpu)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_inplace_binary_keeps_self_storage(mojo_device, dtype):
    """The in-place route writes self's own bytes, at self's own offset."""
    storage_cpu = torch.linspace(-1, 1, 40)
    storage = storage_cpu.to(mojo_device)
    view_cpu, view = storage_cpu[7:31].view(4, 6), storage[7:31].view(4, 6)
    if dtype is not torch.float32:
        pytest.skip("the offset-view contract is dtype independent")
    b_cpu, b = _both((6,), dtype, mojo_device)
    pointer = view.data_ptr()
    view.mul_(b)
    view_cpu.mul_(b_cpu)
    assert view.data_ptr() == pointer
    torch.testing.assert_close(storage.cpu(), storage_cpu)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_scalar_and_mixed_dtype_routes_match_cpu(mojo_device, dtype):
    """Tensor-with-scalar add/sub/mul/div/pow, and the mixed-dtype add."""
    a_cpu, a = _both((3, 5), dtype, mojo_device)
    for value in (0.5, 2, -1.25):
        for op in ("add", "sub", "mul"):
            torch.testing.assert_close(
                getattr(torch, op)(a, value).cpu(), getattr(torch, op)(a_cpu, value)
            )
        # Default per-dtype tolerances: the scalar routes compute in float32
        # (and the mojo:cpu divide embeds the divisor in the tensor's own
        # dtype), so a low-precision result is an ulp from CPU torch's.
        torch.testing.assert_close((a / 0.97).cpu(), a_cpu / 0.97)
        torch.testing.assert_close(torch.pow(a, 3.0).cpu(), torch.pow(a_cpu, 3.0))
    other = torch.float32 if dtype is not torch.float32 else torch.bfloat16
    b_cpu, b = _both((3, 5), other, mojo_device)
    want = a_cpu + b_cpu
    got = a + b
    assert got.dtype == want.dtype
    torch.testing.assert_close(got.cpu(), want)


def test_inplace_binary_rejects_a_partially_overlapping_operand(mojo_device):
    """The fast route declines and the cascade still raises ATen's error."""
    storage = torch.arange(10, dtype=torch.float32, device=mojo_device)
    for op in ("mul_", "add_", "sub_"):
        with pytest.raises(RuntimeError, match="single memory location"):
            getattr(storage[1:], op)(storage[:-1])
    with pytest.raises(RuntimeError, match="single memory location"):
        torch.add(storage[:-1], storage[:-1], out=storage[1:])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_clamp_with_a_float_bound_promotes_an_integer_tensor(mojo_gpu, dtype):
    """ATen's clamp iterator promotes its inputs to a common dtype:
    `torch.clamp(int_tensor, min=0.5)` is a FLOAT tensor, not an integer one
    with the bound truncated away."""
    a_cpu, a = _both((10,), dtype, mojo_gpu, low=0, high=5)
    for kwargs in ({"min": 0.5}, {"max": 3.5}, {"min": 0.5, "max": 3.5}):
        want = a_cpu.clamp(**kwargs)
        got = a.clamp(**kwargs)
        assert got.dtype == want.dtype == torch.get_default_dtype()
        torch.testing.assert_close(got.cpu(), want)
    # An INTEGER bound keeps the tensor's own dtype, as it does on CPU.
    assert a.clamp(min=1).dtype == a_cpu.clamp(min=1).dtype == dtype
    torch.testing.assert_close(a.clamp(min=1).cpu(), a_cpu.clamp(min=1))


# --------------------------------------------------------------------------
# addcmul / addcdiv / lerp
# --------------------------------------------------------------------------


def test_addcmul_addcdiv(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_addcmul)
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 4), torch.float32, mojo_device)
    c_cpu, c = _both((3, 4), torch.float32, mojo_device, low=1, high=5)
    torch.testing.assert_close(
        torch.addcmul(a, b, c, value=0.5).cpu(),
        torch.addcmul(a_cpu, b_cpu, c_cpu, value=0.5),
    )
    torch.testing.assert_close(
        torch.addcdiv(a, b, c, value=2.0).cpu(),
        torch.addcdiv(a_cpu, b_cpu, c_cpu, value=2.0),
    )
    dest = torch.empty((3, 4), device=mojo_device)
    with native_ran("aten::addcmul.out"):
        torch.addcmul(a, b, c, value=0.5, out=dest)
    torch.testing.assert_close(
        dest.cpu(), torch.addcmul(a_cpu, b_cpu, c_cpu, value=0.5)
    )
    with native_ran("aten::addcdiv.out"):
        torch.addcdiv(a, b, c, value=2.0, out=dest)
    torch.testing.assert_close(
        dest.cpu(), torch.addcdiv(a_cpu, b_cpu, c_cpu, value=2.0)
    )


def test_addcmul_broadcast(mojo_device):
    a_cpu, a = _both((3, 4), torch.float32, mojo_device)
    b_cpu, b = _both((3, 1), torch.float32, mojo_device)
    c_cpu, c = _both((1, 4), torch.float32, mojo_device)
    with native_ran("aten::addcmul"):
        out = torch.addcmul(a, b, c, value=-1.0)
    torch.testing.assert_close(
        out.cpu(), torch.addcmul(a_cpu, b_cpu, c_cpu, value=-1.0)
    )


@pytest.mark.parametrize("operation", ["addcmul", "addcdiv", "lerp"])
@pytest.mark.parametrize("shape", [(0,), (7,), (17, 19)])
@pytest.mark.parametrize("value", [-0.25, 0.5 - 2**-30, 0.75])
def test_optimizer_inplace_offset_storage(mojo_gpu, operation, shape, value):
    count = math.prod(shape)
    storage_cpu = torch.linspace(-1.0, 1.0, count + 10)
    storage = storage_cpu.to(mojo_gpu)
    a_cpu = storage_cpu[3 : 3 + count].view(shape)
    a = storage[3 : 3 + count].view(shape)
    b_cpu = torch.linspace(0.25, 1.25, count).view(shape)
    c_cpu = torch.linspace(1.0, 2.0, count).view(shape)
    b, c = b_cpu.to(mojo_gpu), c_cpu.to(mojo_gpu)
    version = a._version
    native_name = "aten::lerp_.Scalar" if operation == "lerp" else f"aten::{operation}_"
    with native_ran(native_name):
        if operation == "lerp":
            result = a.lerp_(b, value)
            a_cpu.lerp_(b_cpu, value)
        else:
            result = getattr(a, operation + "_")(b, c, value=value)
            getattr(a_cpu, operation + "_")(b_cpu, c_cpu, value=value)
    assert result is a
    assert a._version == version + 1
    torch.testing.assert_close(storage.cpu(), storage_cpu)


@pytest.mark.parametrize("operation", ["addcmul", "addcdiv", "lerp"])
def test_optimizer_out_partial_overlap(mojo_gpu, operation):
    storage = torch.arange(10, dtype=torch.float32, device=mojo_gpu)
    a, dest = storage[:-1], storage[1:]
    b = torch.ones_like(a)
    with pytest.raises(RuntimeError, match="overlap|single memory location"):
        if operation == "lerp":
            torch.lerp(dest, a, 0.25, out=dest)
        else:
            getattr(torch, operation)(a, b, b, out=dest)


def _optimizer_update(a: torch.Tensor, b: torch.Tensor, operation: str) -> torch.Tensor:
    if operation == "lerp":
        return a.lerp_(b, 0.25)
    return getattr(a, operation + "_")(b, b, value=0.25)


@pytest.mark.parametrize("operation", ["addcmul", "addcdiv", "lerp"])
@pytest.mark.parametrize(
    "layout", ["broadcast", "transpose", "interleaved", "alias", "empty_expanded"]
)
def test_optimizer_inplace_layout_contract(mojo_device, operation, layout):
    host = torch.arange(1, 41, dtype=torch.float32).reshape(5, 8) / 40
    device = host.to(mojo_device)

    def views(base: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if layout == "broadcast":
            return base[:4], base[4:]
        if layout == "transpose":
            return base[:4].t(), base[4:].t()
        if layout == "interleaved":
            return base.flatten()[::2], base.flatten()[1::2]
        if layout == "alias":
            return base, base
        empty = base[:0].expand(3, 0, 8)
        return empty, empty

    a, b = views(device)
    expected_a, expected_b = views(host)
    version = a._version
    assert _optimizer_update(a, b, operation) is a
    _optimizer_update(expected_a, expected_b, operation)
    assert a._version == version + 1
    torch.testing.assert_close(device.cpu(), host)


@pytest.mark.parametrize("operation", ["addcmul", "addcdiv", "lerp"])
@pytest.mark.parametrize("invalid", ["grow", "expand", "partial", "transpose_alias"])
def test_optimizer_inplace_rejects_before_write(mojo_device, operation, invalid):
    expected = torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
    storage = expected.to(mojo_device)
    if invalid == "grow":
        a, b = storage[:1], storage
    elif invalid == "expand":
        a, b = storage[:1].expand(4, 4), storage
    elif invalid == "partial":
        a, b = storage.flatten()[1:], storage.flatten()[:-1]
    else:
        a, b = storage, storage.t()
    version = a._version
    with pytest.raises(RuntimeError, match="shape|memory location|overlap"):
        _optimizer_update(a, b, operation)
    assert a._version == version
    torch.testing.assert_close(storage.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["addcmul", "addcdiv", "lerp"])
def test_optimizer_inplace_autograd_contract(mojo_device, operation):
    host = torch.linspace(0.25, 1.0, 7, requires_grad=True)
    leaf = host.detach().to(mojo_device).requires_grad_()
    b_host = torch.full((7,), 0.5)
    b = b_host.to(mojo_device)
    for invalid in (leaf, leaf.view_as(leaf)):
        version = invalid._version
        with pytest.raises(RuntimeError, match="leaf"):
            _optimizer_update(invalid, b, operation)
        assert invalid._version == version
    torch.testing.assert_close(leaf.cpu(), host)
    out = _optimizer_update(leaf * 1.0, b, operation)
    expected = _optimizer_update(host * 1.0, b_host, operation)
    out.sum().backward()
    expected.sum().backward()
    assert leaf.grad is not None
    assert host.grad is not None
    torch.testing.assert_close(leaf.grad.cpu(), host.grad)


@pytest.mark.parametrize("operation", ["addcmul", "addcdiv"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("alias", [False, True])
def test_addc_out_contiguous(mojo_gpu, operation, dtype, alias):
    a_cpu = torch.linspace(-1, 1, 357, dtype=dtype)
    b_cpu = torch.linspace(0.25, 1.25, 357, dtype=dtype)
    c_cpu = torch.linspace(1, 2, 357, dtype=dtype)
    a, b, c = [t.to(mojo_gpu) for t in (a_cpu, b_cpu, c_cpu)]
    dest = a if alias else torch.empty_like(a)
    result = getattr(torch, operation)(a, b, c, value=0.125, out=dest)
    assert result is dest
    expected = getattr(torch, operation)(a_cpu, b_cpu, c_cpu, value=0.125)
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize("weight", [0.25, 0.75])
def test_lerp(mojo_device, weight):
    a_cpu, a = _both((6,), torch.float32, mojo_device)
    b_cpu, b = _both((6,), torch.float32, mojo_device)
    with native_ran("aten::lerp.Scalar"):
        out = torch.lerp(a, b, weight)
    torch.testing.assert_close(out.cpu(), torch.lerp(a_cpu, b_cpu, weight))
    dest = torch.empty((6,), device=mojo_device)
    with native_ran("aten::lerp.Scalar_out"):
        torch.lerp(a, b, weight, out=dest)
    torch.testing.assert_close(dest.cpu(), torch.lerp(a_cpu, b_cpu, weight))


# --------------------------------------------------------------------------
# logical / bitwise
# --------------------------------------------------------------------------


def test_logical_and_xor(mojo_device, call_checker):
    call_checker.register(aten_functions.aten_logical_and)
    a_cpu = torch.tensor([True, False, True, False])
    b_cpu = torch.tensor([True, True, False, False])
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    torch.testing.assert_close(
        torch.logical_and(a, b).cpu(), torch.logical_and(a_cpu, b_cpu)
    )
    torch.testing.assert_close(
        torch.logical_xor(a, b).cpu(), torch.logical_xor(a_cpu, b_cpu)
    )


def test_logical_mixed_dtypes(mojo_device):
    a_cpu = torch.tensor([0.0, 1.5, 0.0, -2.0])
    b_cpu = torch.tensor([1, 0, 0, 7], dtype=torch.int64)
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    with native_ran("aten::logical_and"):
        out = torch.logical_and(a, b)
    torch.testing.assert_close(out.cpu(), torch.logical_and(a_cpu, b_cpu))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_bitwise(mojo_device, dtype, call_checker):
    call_checker.register(aten_functions.aten_bitwise_and)
    if dtype == torch.bool:
        a_cpu = torch.tensor([True, False, True, False])
        b_cpu = torch.tensor([True, True, False, False])
    else:
        a_cpu = torch.tensor([1, 2, 3, 12], dtype=dtype)
        b_cpu = torch.tensor([3, 3, 1, 10], dtype=dtype)
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    torch.testing.assert_close((a & b).cpu(), a_cpu & b_cpu)
    torch.testing.assert_close((a | b).cpu(), a_cpu | b_cpu)
    torch.testing.assert_close((a ^ b).cpu(), a_cpu ^ b_cpu)


def test_bitwise_scalar(mojo_device):
    a_cpu = torch.tensor([1, 2, 3, 12], dtype=torch.int64)
    a = a_cpu.to(mojo_device)
    with native_ran("aten::bitwise_and.Scalar"):
        out = a & 3
    torch.testing.assert_close(out.cpu(), a_cpu & 3)


# --------------------------------------------------------------------------
# GPU-only routes
# --------------------------------------------------------------------------


def test_add_f32_bf16_fused_route(mojo_gpu):
    """FP32 + BF16 -> FP32 in one launch, no materialized BF16 operand."""
    a_cpu, a = _both((64, 32), torch.float32, mojo_gpu)
    b_cpu, b = _both((64, 32), torch.bfloat16, mojo_gpu)
    with native_ran("aten::add.Tensor"):
        out = a + b
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), a_cpu + b_cpu)


def test_autograd_through_binary_ops(mojo_gpu):
    x = torch.randn(4).to(mojo_gpu).requires_grad_()
    y = torch.randn(4).to(mojo_gpu).requires_grad_()
    # `.backward(grad)` rather than `.sum().backward()`: reductions are
    # another op group.
    ((x * y) + x * 2.0).backward(torch.ones(4).to(mojo_gpu))
    assert x.grad is not None and y.grad is not None
    torch.testing.assert_close(x.grad.cpu(), (y + 2.0).detach().cpu())
    torch.testing.assert_close(y.grad.cpu(), x.detach().cpu())


# --------------------------------------------------------------------------
# Numeric regressions: the operand regimes where a naive formula is wrong.
# Each of these reproduced a real bug; the ordinary tests above use benign
# operands that none of them can fire on.
# --------------------------------------------------------------------------


def test_remainder_bfloat16_near_zero_divisor(mojo_device):
    """A divisor at the smallest bf16 normal makes the quotient overflow.

    `a - trunc(a/b)*b` sends a/b to +/-inf here, though the true remainder is
    bounded by |b|. The kernel must reduce without forming the quotient.
    """
    a_cpu = torch.tensor([0.4922, -7.4375, 2.3125, 91.0], dtype=torch.bfloat16)
    b_cpu = torch.tensor([1.0, 1.0, 1.0, 1.1754943508222875e-38], dtype=torch.bfloat16)
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    torch.testing.assert_close(
        torch.remainder(a, b).cpu(), torch.remainder(a_cpu, b_cpu)
    )
    torch.testing.assert_close(
        torch.remainder(b, a).cpu(), torch.remainder(b_cpu, a_cpu)
    )


def test_remainder_float16_large_ratio(mojo_device):
    """|a/b| ~ 2.4e7 is past fp32's 2**24 integer grid: rounding the quotient
    to fp32 loses the remainder entirely (it comes back exactly 0)."""
    a_cpu = torch.tensor([23456.0], dtype=torch.float16)
    b_cpu = torch.tensor([0.0009937286376953125], dtype=torch.float16)
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    torch.testing.assert_close(
        torch.remainder(a, b).cpu(), torch.remainder(a_cpu, b_cpu)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_floor_divide_narrow_float_boundary(mojo_device, dtype):
    """-6.3125 / -1.0546875 is 5.985..., but rounds to 6.0 at bf16 precision.
    Dividing in the narrow dtype and flooring afterwards answers 6."""
    a_cpu = torch.tensor([-6.3125, 91.0, 2.3125, -5.2812, 357.0], dtype=dtype)
    b_cpu = torch.tensor([-1.0546875, 3.375, 8.5, 1.0547, 6.789], dtype=dtype)
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    torch.testing.assert_close(
        torch.floor_divide(a, b).cpu(), torch.floor_divide(a_cpu, b_cpu)
    )


def test_floor_divide_subnormal_quotient_underflow(mojo_device):
    """Floor must retain the sign of a quotient flushed to zero by Metal."""
    a_cpu = torch.tensor(
        [2.0**-126, -(2.0**-126), 0.0, -0.0, 1.0, -1.0], dtype=torch.bfloat16
    )
    b_cpu = torch.tensor(
        [-4.71875, -4.71875, -2.0, 2.0, -float("inf"), float("inf")],
        dtype=torch.bfloat16,
    )
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    expected = torch.floor_divide(a_cpu, b_cpu)
    assert expected[0].item() == -1.0, expected  # the CPU reference itself
    torch.testing.assert_close(torch.floor_divide(a, b).cpu(), expected)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_div_trunc_mode_keeps_the_narrow_quotient(mojo_device, dtype):
    """The same operands as floor_divide above, with rounding_mode="trunc":
    ATen does NOT widen here, so the answer is the rounded-then-truncated 6,
    not 5. The opposite fix from the floor case -- this is the check that
    stops someone "correcting" both."""
    a_cpu = torch.tensor([-6.3125, 91.0, 2.3125, -5.2812, 357.0], dtype=dtype)
    b_cpu = torch.tensor([-1.0546875, 3.375, 8.5, 1.0547, 6.789], dtype=dtype)
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    torch.testing.assert_close(
        torch.div(a, b, rounding_mode="trunc").cpu(),
        torch.div(a_cpu, b_cpu, rounding_mode="trunc"),
    )


@pytest.mark.parametrize("mode", ["floor", "trunc"])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_div_rounding_mode_negative_divisor(mojo_device, mode, dtype):
    """floor and trunc only disagree when the quotient is negative, so a
    positive-divisor test cannot tell them apart."""
    if dtype == torch.float64:
        skip_if_metal(mojo_device, "float64 is not supported on Apple GPU")
    a_cpu = torch.tensor([7, -7, 7, -7, 8, -8, 6, -6], dtype=dtype)
    b_cpu = torch.tensor([2, 2, -2, -2, 3, 3, -3, -3], dtype=dtype)
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    torch.testing.assert_close(
        torch.div(a, b, rounding_mode=mode).cpu(),
        torch.div(a_cpu, b_cpu, rounding_mode=mode),
    )


@pytest.mark.parametrize("mode", ["floor", "trunc"])
def test_div_scalar_mode_negative_scalar(mojo_device, mode):
    a_cpu = torch.tensor([7, -7, 8, -8, 9, -9], dtype=torch.int32)
    a = a_cpu.to(mojo_device)
    torch.testing.assert_close(
        torch.div(a, -2, rounding_mode=mode).cpu(),
        torch.div(a_cpu, -2, rounding_mode=mode),
    )


def test_lerp_scalar_weight_at_the_branch_boundary(mojo_device):
    """weight = 0.5 - 2**-30 narrows to exactly 0.5f, and ATen picks its
    stable formula from the NARROWED value; these operands separate the two
    branches, so the answer must be bit-exact, not merely close."""
    weight = 0.5 - 2.0**-30
    start_cpu = torch.tensor(
        [[-1.0687099695205688, -2.0, 3.0], [4.0, -5.0, 6.0]], dtype=torch.float32
    )
    end_cpu = torch.tensor([[2.028475284576416, 8.0, -3.0]], dtype=torch.float32)
    start, end = start_cpu.to(mojo_device), end_cpu.to(mojo_device)

    out = torch.empty_like(start)
    returned = torch.lerp(start, end, weight, out=out)
    assert returned is out
    torch.testing.assert_close(
        out.cpu(), torch.lerp(start_cpu, end_cpu, weight), rtol=0, atol=0
    )

    before = start.data_ptr()
    aliased = start.view(3, 2)
    assert start.lerp_(end, weight) is start
    assert start.data_ptr() == before
    expected = start_cpu.lerp_(end_cpu, weight)
    torch.testing.assert_close(start.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(aliased.cpu(), expected.view(3, 2), rtol=0, atol=0)
    torch.testing.assert_close(end.cpu(), end_cpu, rtol=0, atol=0)


@pytest.mark.parametrize("f32_first", [True, False])
@pytest.mark.parametrize(
    "shape", [(), (0,), (0, 5), (1,), (7,), (17, 65), (3, 5, 7), (2, 3, 5, 7, 11)]
)
def test_add_f32_bf16_fused_is_bit_exact(mojo_gpu, shape, f32_first):
    """Bit-exact, not merely close: that is what says the bf16 operand was
    widened in registers rather than materialized through a rounding cast."""
    f32_cpu = torch.randn(shape, dtype=torch.float32)
    bf16_cpu = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    f32, bf16 = f32_cpu.to(mojo_gpu), bf16_cpu.to(mojo_gpu)
    left, right = (f32, bf16) if f32_first else (bf16, f32)
    left_cpu, right_cpu = (f32_cpu, bf16_cpu) if f32_first else (bf16_cpu, f32_cpu)
    out = left + right
    assert out.dtype == torch.float32
    torch.testing.assert_close(out.cpu(), left_cpu + right_cpu, rtol=0, atol=0)


@pytest.mark.parametrize("f32_first", [True, False])
def test_add_f32_bf16_fused_on_offset_views(mojo_gpu, f32_first):
    """Same, off a storage offset: the fused route may not assume its
    operands start at offset 0."""
    f32_cpu = torch.randn(1106, dtype=torch.float32)
    bf16_cpu = torch.randn(1106, dtype=torch.float32).to(torch.bfloat16)
    f32, bf16 = f32_cpu.to(mojo_gpu)[1:], bf16_cpu.to(mojo_gpu)[1:]
    left, right = (f32, bf16) if f32_first else (bf16, f32)
    left_cpu, right_cpu = (
        (f32_cpu[1:], bf16_cpu[1:]) if f32_first else (bf16_cpu[1:], f32_cpu[1:])
    )
    torch.testing.assert_close(
        (left + right).cpu(), left_cpu + right_cpu, rtol=0, atol=0
    )


@pytest.mark.parametrize("shape", [(), (1,), (5,), (3, 7)])
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int32]
)
def test_inplace_scalar_every_dtype_and_rank(mojo_device, dtype, shape):
    """add_/mul_ with a python scalar. The integer dtype is here because it
    must keep falling through to the functional-plus-copy-back path."""
    scalars = (3, -1, 2) if dtype == torch.int32 else (2.5, -1, 0.0)
    for scalar in scalars:
        if dtype.is_floating_point:
            cpu = torch.randn(shape, dtype=torch.float32).to(dtype)
        else:
            cpu = torch.randint(-9, 9, shape, dtype=dtype)
        x = cpu.to(mojo_device)
        assert x.add_(scalar) is x
        torch.testing.assert_close(x.cpu(), cpu.add_(scalar), rtol=2e-2, atol=2e-2)
        assert x.mul_(scalar) is x
        torch.testing.assert_close(x.cpu(), cpu.mul_(scalar), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("scalar", [0.9, 1.0001, -0.33333])
@pytest.mark.parametrize("strided", [False, True])
def test_mul_inplace_preserves_scalar_precision(
    mojo_device: str, dtype: torch.dtype, scalar: float, strided: bool
):
    cpu = (torch.arange(515, dtype=torch.float32) / 37 - 7).to(dtype)
    actual = cpu.to(mojo_device)
    if strided:
        cpu, actual = cpu[1::2], actual[1::2]
    assert actual.mul_(scalar) is actual
    torch.testing.assert_close(actual.cpu(), cpu.mul_(scalar), rtol=0, atol=0)


def test_add_above_last_level_cache(mojo_gpu):
    """24_000_003 fp32 elements: past a 256 MiB L2, and not a multiple of the
    4-element vector, so the scalar tail rides the streaming grid too. The
    only case that reaches the arm covering the vector slots exactly once
    instead of the capped-block grid."""
    n = 24_000_003
    left_cpu = torch.arange(n, dtype=torch.float32) % 1021 - 510.0
    right_cpu = torch.arange(n, dtype=torch.float32) % 733 - 366.0
    out = left_cpu.to(mojo_gpu) + right_cpu.to(mojo_gpu)
    torch.testing.assert_close(out.cpu(), left_cpu + right_cpu, rtol=0, atol=0)
