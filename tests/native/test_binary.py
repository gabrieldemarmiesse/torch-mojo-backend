"""Binary arithmetic on the native mojo device: add/sub/mul/div, pow,
maximum/minimum, remainder, floor_divide, lerp, addcmul/addcdiv, clamp and
the logical/bitwise ops, with their in-place and `out=` variants.

Every check compares against the same computation on CPU torch through the
public API only; `CallChecker` (or `native.op_count` for the ops with no
`aten_functions` twin) asserts the native op actually ran.
"""

import contextlib

import pytest
import torch

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
def test_add_sub_alpha_reduced_precision_uses_opmath(mojo_gpu, dtype):
    """ATen's add/sub functor runs in `opmath_type<scalar_t>` (float32 for
    both half types) and rounds ONCE, at the store. Materializing `alpha * b`
    in the input dtype first rounds twice, and the two answers really differ:
    exact equality against the single-rounding reference is what separates
    them."""
    a = (torch.arange(64, dtype=torch.float32) / 7.0 - 4.0).to(dtype)
    b = (torch.arange(64, dtype=torch.float32) / 3.0 - 10.0).to(dtype)
    alpha = 1.0 / 3.0
    ad, bd = a.to(mojo_gpu), b.to(mojo_gpu)

    got = torch.add(ad, bd, alpha=alpha)
    assert got.dtype == dtype
    torch.testing.assert_close(
        got.cpu(), (a.float() + alpha * b.float()).to(dtype), atol=0, rtol=0
    )
    got_sub = torch.sub(ad, bd, alpha=alpha)
    torch.testing.assert_close(
        got_sub.cpu(), (a.float() - alpha * b.float()).to(dtype), atol=0, rtol=0
    )
    # The double-rounded answer is a DIFFERENT tensor: without this the test
    # would pass on the implementation it is meant to reject.
    double_rounded = (a.float() + (alpha * b.float()).to(dtype).float()).to(dtype)
    assert not torch.equal(double_rounded, got.cpu())


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


def test_div_and_pow_float64(mojo_device, call_checker):
    """float64 through the broadcast binary kernel.

    logic_ops' SPEC_BCAST_DTYPES carries float64, and `_binary_spec_into_go`
    asks of div/pow only that the dtype be floating -- so both ops run at
    full precision instead of declining. The scalar exponent takes that same
    broadcast route rather than elementwise_ops' PowScalarSpec, which is
    FLOAT_DTYPES only (and would raise, not narrow, on a float64 operand).
    """
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
    """2**-126 / -4.71875 is an fp32 SUBNORMAL; flushing it to zero answers 0
    where the floor is -1. bf16-only: fp16's normal range bottoms out at
    2**-14, far above the fp32 subnormal cliff."""
    a_cpu = torch.tensor([2.0**-126], dtype=torch.bfloat16)
    b_cpu = torch.tensor([-4.71875], dtype=torch.bfloat16)
    a, b = a_cpu.to(mojo_device), b_cpu.to(mojo_device)
    expected = torch.floor_divide(a_cpu, b_cpu)
    assert expected.item() == -1.0, expected  # the CPU reference itself
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
