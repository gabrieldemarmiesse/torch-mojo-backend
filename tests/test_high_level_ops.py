import math

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo import mark_dynamic

from torch_mojo_backend import mojo_backend
from torch_mojo_backend.testing import (
    Conf,
    check_functions_are_equivalent,
    check_outputs,
)

# torch.testing.assert_close's defaults for float32.
_FP32_RTOL = 1.3e-6
_FP32_ATOL = 1e-5

# TF32 keeps 10 explicit mantissa bits, so its unit roundoff is 2**-11.
_TF32_UNIT_ROUNDOFF = 2.0**-11


def tf32_gemm_atol(k: int, *, scale: float = 1.0) -> float:
    """Absolute error budget for one float32 GEMM of inner size ``k`` executed
    on TF32 tensor cores, for the unit-normal operands these tests use.

    MAX lowers a float32 ``mm``/``bmm`` to ``rmo.matmul``, whose GPU kernel
    rounds both operands to TF32 and accumulates in fp32. There is no opt-out
    to reach for: ``max/kernels/src/linalg/matmul/gpu/__init__.mojo``
    compile-asserts that ``use_tf32=False`` is wired up for the SM100 dispatch
    only, and ``max.graph.ops.matmul`` takes no precision argument at all.
    torch eager meanwhile keeps ``torch.get_float32_matmul_precision() ==
    "highest"``, i.e. IEEE fp32, so the two are *expected* to disagree on an
    NVIDIA GPU. The compiled result is in fact bit-identical (0 ulp, measured
    on H100 for several shapes) to torch's own output once
    ``torch.backends.cuda.matmul.allow_tf32`` is turned on, which is what says
    the lowering is right and only the arithmetic is coarser.

    Rounding both operands bounds one output element's error by
    ``2 * u * sum_k |a_ik * b_kj|`` with ``u = 2**-11 = 4.9e-4``; the fp32
    accumulation contributes a further ``~k * 6e-8``, which is negligible
    beside it. ``|a*b|`` for unit normals has mean ``2/pi`` and standard
    deviation ``sqrt(1 - (2/pi)**2)``, so that sum stays under its mean plus
    five standard deviations for any draw of these (unseeded) inputs.
    ``scale`` is whatever constant multiplies the product afterwards: addmm's
    ``alpha``, attention's ``1/sqrt(head_dim)``.

    Measured over 300 random draws per shape, the budget lands 2.6x-4.2x above
    the worst error actually observed. It stays tight enough to fail on a bug:
    replaying the same draws through a broken addmm exceeds it by 70x (dropped
    ``beta``), 409x (dropped ``alpha``) or 634x (the two swapped), and even a
    silent downgrade of the GEMM from TF32 to bfloat16 overshoots by 2.9x.
    """
    mean_abs = 2.0 / math.pi
    sd_abs = math.sqrt(1.0 - mean_abs**2)
    sum_abs_products = mean_abs * k + 5.0 * sd_abs * math.sqrt(k)
    return 2.0 * _TF32_UNIT_ROUNDOFF * scale * sum_abs_products


def gemm_tolerances(
    device: str, k: int, *, scale: float = 1.0
) -> tuple[float | None, float | None]:
    """``(rtol, atol)`` for a comparison whose result goes through a float32 GEMM.

    Only the NVIDIA path is loosened, and only through ``atol``: cancellation
    leaves the *relative* error of a near-zero output element unbounded (a
    64x64x64 product shows 54% on one element), so ``rtol`` keeps torch's fp32
    default and the whole TF32 budget goes into ``atol``. On CPU nothing is
    truncated, so ``None, None`` leaves the strict defaults in force.
    """
    if not device.startswith("cuda"):
        return None, None
    return _FP32_RTOL, max(_FP32_ATOL, tf32_gemm_atol(k, scale=scale))


def test_basic_addition(conf: Conf):
    def fn(x, y):
        return x + y

    a = torch.randn(3)
    b = torch.randn(3)

    check_outputs(fn, conf, [a, b])


def test_iadd(conf: Conf):
    def fn(x, y):
        x += y
        return x

    a = torch.randn(3)
    b = torch.randn(3)

    check_outputs(fn, conf, [a, b])


def test_t_method(conf: Conf):
    def fn(x):
        return x.t()

    a = torch.randn(3, 4)

    check_outputs(fn, conf, [a])


def test_t_function(conf: Conf):
    def fn(x):
        return torch.t(x)

    a = torch.randn(3, 4)

    check_outputs(fn, conf, [a])


def test_new_ones(device: str):
    def fn(x):
        return x.new_ones((3, 3))

    a = torch.randn(3)

    check_functions_are_equivalent(fn, device, [a])


def test_new_ones_device(device: str):
    def fn(x):
        return x.new_ones((3, 3), device=torch.device(device))

    a = torch.randn(3)

    check_functions_are_equivalent(fn, "cpu", [a])


def test_new_ones_dtype(device: str):
    def fn(x):
        return x.new_ones((3, 3), dtype=torch.uint8)

    a = torch.randn(3)

    check_functions_are_equivalent(fn, device, [a])


def test_operator_add(conf: Conf, tensor_shapes: tuple):
    def fn(x, y):
        return x + y

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a, b])


def test_subtraction(conf: Conf, tensor_shapes: tuple):
    def fn(x, y):
        return x - y

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a, b])


def test_subtraction_different_dtypes(conf: Conf, tensor_shapes: tuple):
    def fn(x, y):
        return x - y

    a = torch.randn(tensor_shapes, dtype=torch.float32)
    b = torch.randint(0, 10, tensor_shapes, dtype=torch.int64)

    check_outputs(fn, conf, [a, b])


def test_multiplication(conf: Conf, tensor_shapes: tuple):
    def fn(x, y):
        return x * y

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a, b])


def test_multiplication_int32(conf: Conf, tensor_shapes: tuple):
    def fn(x, y):
        return x * y

    a = torch.randint(0, 10, size=tensor_shapes, dtype=torch.int32)
    b = torch.randint(0, 10, size=tensor_shapes, dtype=torch.int32)

    check_outputs(fn, conf, [a, b])


def test_division(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x / y

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes) + 1.0  # Avoid division by zero

    check_functions_are_equivalent(fn, device, [a, b])


def test_floor_division(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x // y

    a = torch.randn(tensor_shapes) * 10
    b = torch.randn(tensor_shapes).abs() + 1.0  # Avoid division by zero

    check_functions_are_equivalent(fn, device, [a, b])


def test_power(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x**y

    a = torch.randn(tensor_shapes).abs() + 0.1  # Avoid negative base
    b = torch.randn(tensor_shapes) * 2  # Keep exponent reasonable

    check_functions_are_equivalent(fn, device, [a, b])


def test_modulo(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return x % y

    a = torch.randn(tensor_shapes) * 10
    b = torch.randn(tensor_shapes).abs() + 1.0  # Avoid division by zero

    check_functions_are_equivalent(fn, device, [a, b])


def test_abs(conf: Conf, tensor_shapes: tuple):
    def fn(x):
        return torch.abs(x)

    a = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a])


def test_floor(conf: Conf, tensor_shapes: tuple):
    def fn(x):
        return torch.floor(x)

    # Use a mix of positive and negative values to test floor properly
    a = torch.randn(tensor_shapes) * 10  # Scale to get larger values

    check_outputs(fn, conf, [a])


def test_floor_edge_cases(conf: Conf):
    """Test floor with specific edge cases"""

    def fn(x):
        return torch.floor(x)

    # Test with specific values to ensure floor behavior is correct
    test_cases = [
        torch.tensor([2.7, -2.7, 3.0, -3.0, 0.5, -0.5, 0.0]),  # Mixed cases
        torch.tensor([1.9999, -1.9999, 10.1, -10.1]),  # Near integers
        torch.tensor([100.9, -100.9, 0.1, -0.1]),  # Large and small values
    ]

    for test_tensor in test_cases:
        check_outputs(fn, conf, [test_tensor])


def test_tensor_floor_method(conf: Conf):
    """Test tensor.floor() method"""

    def fn(x):
        return x.floor()

    x = torch.randn(3, 4) * 5  # Scale values for better floor testing
    check_outputs(fn, conf, [x])


def test_cos(conf: Conf, tensor_shapes: tuple):
    def fn(x):
        return torch.cos(x)

    a = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a])


def test_sin(conf: Conf, tensor_shapes: tuple):
    def fn(x):
        return torch.sin(x)

    a = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a])


def test_tanh(conf: Conf, tensor_shapes: tuple):
    def fn(x):
        return torch.tanh(x)

    a = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a])


def test_sign(conf: Conf, tensor_shapes: tuple):
    def fn(x):
        return torch.sign(x)

    # Test with mixed positive, negative, and zero values
    a = torch.randn(tensor_shapes)
    # Ensure we have a mix of positive, negative, and zero
    if a.numel() >= 3:
        a.view(-1)[:3] = torch.tensor([-1.0, 0.0, 1.0])

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sign_different_dtypes(conf: Conf, dtype):
    def fn(x):
        return torch.sign(x)

    a = torch.tensor([-2.5, -1.0, 0.0, 1.0, 2.5], dtype=dtype)

    check_outputs(fn, conf, [a])


def test_atanh(conf: Conf, tensor_shapes: tuple):
    def fn(x):
        return torch.atanh(x)

    # atanh is defined for |x| < 1, so we need to ensure our test values are in this range
    a = torch.rand(tensor_shapes) * 1.8 - 0.9  # Values in range (-0.9, 0.9)

    check_outputs(fn, conf, [a])


def test_outer(conf: Conf):
    def fn(x, y):
        return torch.outer(x, y)

    # torch.outer requires 1D tensors
    a = torch.randn(5)
    b = torch.randn(3)

    check_outputs(fn, conf, [a, b])


def test_log1p_basic(conf: Conf):
    """Test basic log1p functionality"""

    def fn(x):
        return torch.log1p(x)

    # log1p domain is x > -1, use values in range (-0.5, 2.0) for safety
    a = torch.rand(3, 4) * 2.5 - 0.5  # Range (-0.5, 2.0)
    check_outputs(fn, conf, [a])


def test_log1p_small_values(conf: Conf):
    """Test log1p with small values where it's most beneficial"""

    def fn(x):
        return torch.log1p(x)

    # Test with very small values where log1p is numerically superior to log(1+x)
    test_cases = [
        torch.tensor([0.0, 0.1, 0.01, 0.001]),  # Small positive values
        torch.tensor([-0.1, -0.01, -0.001, -0.0001]),  # Small negative values
        torch.tensor([1e-6, -1e-6, 1e-10, -1e-10]),  # Very small values
        torch.rand(2, 3) * 0.2 - 0.1,  # Random small values in (-0.1, 0.1)
    ]
    for test_tensor in test_cases:
        check_outputs(fn, conf, [test_tensor])


def test_log_basic(conf: Conf):
    """Test basic log functionality"""

    def fn(x):
        return torch.log(x)

    # Use positive values only since log is only defined for positive numbers
    a = torch.rand(3, 4) + 0.1  # Range (0.1, 1.1) to avoid values too close to zero
    check_outputs(fn, conf, [a])


def test_log_various_ranges(conf: Conf):
    """Test log with various value ranges"""

    def fn(x):
        return torch.log(x)

    test_cases = [
        torch.tensor([1.0, 2.0, 10.0, 100.0]),  # Simple positive values
        torch.tensor([0.1, 0.5, 1.5, 5.0]),  # Mixed small and medium values
        torch.tensor(
            [math.e, 1.0, math.e**2, math.e**0.5]
        ),  # Values with known log results
        torch.rand(2, 3) * 10 + 0.1,  # Random positive values in range (0.1, 10.1)
    ]

    for test_tensor in test_cases:
        check_outputs(fn, conf, [test_tensor])


def test_log1p_various_ranges(conf: Conf):
    """Test log1p with various value ranges"""

    def fn(x):
        return torch.log1p(x)

    test_cases = [
        torch.tensor([0.0, 1.0, 2.0, 10.0]),  # Simple values
        torch.tensor([math.e - 1, 0.0, math.e**2 - 1]),  # Values with known results
        torch.tensor([-0.5, -0.9, -0.99, -0.999]),  # Negative values approaching -1
        torch.rand(2, 3) * 5 - 0.5,  # Random values in (-0.5, 4.5)
    ]
    for test_tensor in test_cases:
        check_outputs(fn, conf, [test_tensor])


def test_log_edge_cases(conf: Conf):
    """Test log with edge cases"""

    def fn(x):
        return torch.log(x)

    # Test with edge values (avoiding zero and negative values)
    test_cases = [
        torch.tensor([1.0]),  # log(1) = 0
        torch.tensor([math.e]),  # log(e) = 1
        torch.tensor([1e-5, 1e5]),  # Very small and very large positive values
        torch.tensor([0.001, 1000.0]),  # Range of magnitudes
    ]

    for test_tensor in test_cases:
        check_outputs(fn, conf, [test_tensor])


def test_log1p_edge_cases(conf: Conf):
    """Test log1p with edge cases"""

    def fn(x):
        return torch.log1p(x)

    # Test with edge values (avoiding -1 which gives -inf)
    test_cases = [
        torch.tensor([0.0]),  # log1p(0) = 0
        torch.tensor([math.e - 1]),  # log1p(e-1) = 1
        torch.tensor([-0.9999, 0.9999]),  # Close to domain boundary and symmetric
        torch.tensor([100.0, 1000.0]),  # Large positive values
    ]
    for test_tensor in test_cases:
        check_outputs(fn, conf, [test_tensor])


def test_isnan_basic(conf: Conf):
    """Test basic isnan functionality"""

    def fn(x):
        return torch.isnan(x)

    # Create tensor with mix of normal values and NaNs
    a = torch.tensor([1.0, float("nan"), 3.0, float("nan"), 5.0])
    check_outputs(fn, conf, [a])


def test_isnan_no_nan(conf: Conf):
    """Test isnan with tensor containing no NaNs"""

    def fn(x):
        return torch.isnan(x)

    # Regular tensor with no NaNs
    a = torch.randn(3, 4)
    check_outputs(fn, conf, [a])


def test_isnan_all_nan(conf: Conf):
    """Test isnan with tensor containing all NaNs"""

    def fn(x):
        return torch.isnan(x)

    # Tensor with all NaNs
    a = torch.full((2, 3), float("nan"))
    check_outputs(fn, conf, [a])


def test_isnan_edge_cases(conf: Conf):
    """Test isnan with various edge cases"""

    def fn(x):
        return torch.isnan(x)

    # Test with inf, -inf, 0, negative numbers, and NaN
    test_cases = [
        torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan")]),
        torch.tensor([[1.0, float("nan")], [float("inf"), -2.5]]),
        torch.tensor(
            [1e10, -1e10, 1e-10, float("nan")]
        ),  # Very large and small numbers
    ]

    for test_tensor in test_cases:
        check_outputs(fn, conf, [test_tensor])


def test_tensor_log1p_method(conf: Conf):
    """Test tensor.log1p() method"""

    def fn(x):
        return x.log1p()

    # Use range where log1p is well-defined (x > -1)
    x = torch.rand(3, 4) * 3 - 0.5  # Range (-0.5, 2.5)

    check_outputs(fn, conf, [x])


def test_tensor_log_method(conf: Conf):
    """Test tensor.log() method"""

    def fn(x):
        return x.log()

    # Positive values for log domain
    x = torch.rand(3, 4) * 5 + 0.5  # Range (0.5, 5.5)

    check_outputs(fn, conf, [x])


def test_tensor_isnan_method(conf: Conf):
    """Test tensor.isnan() method"""

    def fn(x):
        return x.isnan()

    # Mix of NaN and regular values
    x = torch.tensor([1.0, float("nan"), -3.5, float("nan"), 0.0])
    check_outputs(fn, conf, [x])


def test_stack_1d(conf: Conf):
    # Test 1D tensors
    def fn_1d(a, b):
        return torch.stack([a, b], dim=0)

    a1d = torch.randn(2)
    b1d = torch.randn(2)
    check_outputs(fn_1d, conf, [a1d, b1d])


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_stack_2d(conf: Conf, dim: int):
    def fn(a, b, c):
        return torch.stack([a, b, c], dim=dim)

    # Create tensors with same shape for stacking
    a = torch.randn(3, 4)
    b = torch.randn(3, 4)
    c = torch.randn(3, 4)

    check_outputs(fn, conf, [a, b, c])


def test_stack_3d(conf: Conf):
    # Test 3D tensors
    def fn_3d(a, b):
        return torch.stack([a, b], dim=0)

    a3d = torch.randn(2, 3, 4)
    b3d = torch.randn(2, 3, 4)
    check_outputs(fn_3d, conf, [a3d, b3d])


@pytest.mark.parametrize("func", [min, max])
def test_builtin_min_max(device: str, func):
    """Only works with a single dimension."""

    def fn(x):
        return func(x)

    a = torch.randn((9,))

    check_functions_are_equivalent(fn, device, [a])


@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("func", [torch.amin, torch.amax])
@pytest.mark.parametrize(
    "shapes,dims",
    [
        ((3, 4), (0,)),
        ((5, 6, 2), (0, 2)),
        ((8,), (0,)),
        ((2, 3, 4), (-1,)),
        ((2, 3, 4), -1),
        ((2, 3, 4), None),
    ],
)
def test_torch_amin_amax_single_element_options(
    conf: Conf, shapes, dims, keepdim, func
):
    """Only works with a single element."""

    def fn(x):
        return func(x, dims, keepdim=keepdim)

    a = torch.randn(shapes)

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("shapes,dims", [((8,), 0), ((2, 3, 4), -1), ((2, 3, 4), None)])
def test_torch_argmax(conf: Conf, shapes, dims, keepdim):
    """Test argmax with various dimensions and keepdim options."""

    def fn(x):
        return torch.argmax(x, dim=dims, keepdim=keepdim)

    a = torch.randn(shapes)

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize("shapes", [(8,), (3, 4), (2, 3, 4), (5, 6, 2, 3)])
def test_torch_argmax_no_dim(conf: Conf, shapes):
    """Test argmax with only tensor argument (no dim parameter)."""

    def fn(x):
        return torch.argmax(x)

    a = torch.randn(shapes)

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("shapes,dims", [((8,), 0), ((2, 3, 4), -1), ((2, 3, 4), None)])
def test_torch_argmin(conf: Conf, shapes, dims, keepdim):
    """Test argmin with various dimensions and keepdim options."""

    def fn(x):
        return torch.argmin(x, dim=dims, keepdim=keepdim)

    a = torch.randn(shapes)

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize("shapes", [(8,), (3, 4), (2, 3, 4), (5, 6, 2, 3)])
def test_torch_argmin_no_dim(conf: Conf, shapes):
    """Test argmin with only tensor argument (no dim parameter)."""

    def fn(x):
        return torch.argmin(x)

    a = torch.randn(shapes)

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize("func", [torch.min, torch.max])
@pytest.mark.parametrize("shapes", [(8,), (3, 4), (2, 3, 4), (5, 6, 2, 3)])
def test_torch_max_single_value(conf: Conf, shapes, func):
    def fn(x):
        return func(x)

    a = torch.randn(shapes)

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize("func", [torch.min, torch.max])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("shapes,dims", [((8,), 0), ((2, 3, 4), -1)])
def test_torch_max_with_dim(device: str, shapes, dims, keepdim, func):
    def fn(x):
        return func(x, dim=dims, keepdim=keepdim)

    a = torch.randn(shapes)

    check_functions_are_equivalent(fn, device, [a])


@pytest.mark.parametrize("func", [torch.min, torch.max])
@pytest.mark.parametrize("shapes,dims", [((8,), 0), ((2, 3, 4), -1)])
def test_torch_max_with_dim_positional(device: str, shapes, dims, func):
    def fn(x):
        return func(x, dims)

    a = torch.randn(shapes)

    check_functions_are_equivalent(fn, device, [a])


@pytest.mark.parametrize("func", [torch.min, torch.max])
def test_torch_max_elementwise(conf: Conf, tensor_shapes: tuple, func):
    def fn(x, y):
        return func(x, y)

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a, b])


@pytest.mark.parametrize("func", [torch.minimum, torch.maximum])
def test_minimum_maximum(conf: Conf, tensor_shapes: tuple, func):
    """Only works with elementwise min/max of two tensors."""

    def fn(x, y):
        return func(x, y)

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a, b])


def test_relu(conf: Conf, tensor_shapes: tuple):
    def fn(x):
        return F.relu(x)

    a = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a])


def test_cat(conf: Conf, tensor_shapes: tuple):
    def fn(x, y):
        return torch.cat([x, y], dim=0)

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a, b])


def test_cat_skips_legacy_empty(conf: Conf):
    """torch.cat ignores the legacy 1-D empty tensor (e.g. uninitialized
    KV-caches), even when its rank differs from the other inputs."""

    def fn(empty, x, y):
        return torch.cat([empty, x, y], dim=2)

    empty = torch.empty(0)
    x = torch.randn(1, 3, 4, 5)
    y = torch.randn(1, 3, 6, 5)

    check_outputs(fn, conf, [empty, x, y])


def test_combination_add_mul(conf: Conf, tensor_shapes: tuple):
    def fn(x, y, z):
        return (x + y) * z

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)
    c = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a, b, c])


def test_combination_sub_div(conf: Conf, tensor_shapes: tuple):
    def fn(x, y, z):
        return (x - y) / z

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)
    c = torch.randn(tensor_shapes) + 1.0  # Avoid division by zero

    check_outputs(fn, conf, [a, b, c])


def test_combination_trig_arithmetic(conf: Conf, tensor_shapes: tuple):
    def fn(x, y):
        return torch.sin(x) + torch.cos(y)

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a, b])


def test_combination_abs_mul_add(conf: Conf, tensor_shapes: tuple):
    def fn(x, y, z):
        return torch.abs(x) * y + z

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)
    c = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a, b, c])


def test_combination_pow_mod(device: str, tensor_shapes: tuple):
    def fn(x, y):
        return (x**2) % y

    a = torch.randn(tensor_shapes).abs() + 0.1
    b = torch.randn(tensor_shapes).abs() + 1.0

    check_functions_are_equivalent(fn, device, [a, b])


def test_complex_combination(conf: Conf, tensor_shapes: tuple):
    def fn(x, y, z):
        return torch.abs(torch.sin(x) * y + torch.cos(z))

    a = torch.randn(tensor_shapes)
    b = torch.randn(tensor_shapes)
    c = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a, b, c])


def test_scalar_shapes(conf: Conf):
    def fn(x, y):
        return x + y * 2

    a = torch.randn(())  # Scalar tensor
    b = torch.randn(())

    check_outputs(fn, conf, [a, b])


def test_broadcasting_compatible(conf: Conf):
    def fn(x, y):
        return x + y

    a = torch.randn(5, 1)
    b = torch.randn(1, 5)

    check_outputs(fn, conf, [a, b])


def test_conv2d_basic(conf: Conf):
    """Test basic conv2d with default parameters"""

    def fn(x, w):
        return F.conv2d(x, w)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv2d_with_bias(conf: Conf):
    """Test conv2d with bias"""

    def fn(x, w, b):
        return F.conv2d(x, w, b)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.randn(out_channels)

    check_outputs(fn, conf, [x, w, b])


def test_conv2d_stride_int(conf: Conf):
    """Test conv2d with integer stride"""

    def fn(x, w):
        return F.conv2d(x, w, stride=2)

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv2d_stride_tuple(conf: Conf):
    """Test conv2d with tuple stride"""

    def fn(x, w):
        return F.conv2d(x, w, stride=(2, 3))

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv2d_padding_int(conf: Conf):
    """Test conv2d with integer padding"""

    def fn(x, w):
        return F.conv2d(x, w, padding=1)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv2d_padding_tuple(conf: Conf):
    """Test conv2d with tuple padding"""

    def fn(x, w):
        return F.conv2d(x, w, padding=(1, 2))

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv2d_dilation_int(conf: Conf):
    def fn(x, w):
        return F.conv2d(x, w, dilation=2)

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv2d_dilation_tuple(conf: Conf):
    """Test conv2d with tuple dilation"""

    def fn(x, w):
        return F.conv2d(x, w, dilation=(2, 3))

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv2d_all_params(conf: Conf):
    """Test conv2d with all parameters specified"""

    def fn(x, w, b):
        return F.conv2d(x, w, b, stride=2, padding=1, dilation=1)

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.randn(out_channels)

    check_outputs(fn, conf, [x, w, b])


def test_conv2d_1x1_kernel(conf: Conf):
    """Test conv2d with 1x1 kernel (pointwise convolution)"""

    def fn(x, w):
        return F.conv2d(x, w)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels = 4

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, 1, 1)

    check_outputs(fn, conf, [x, w])


def test_conv2d_large_kernel(conf: Conf):
    """Test conv2d with larger kernel"""

    def fn(x, w):
        return F.conv2d(x, w, padding=2)

    batch_size, in_channels, height, width = 2, 3, 16, 16
    out_channels, kernel_size = 4, 5

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv2d_asymmetric_kernel(conf: Conf):
    """Test conv2d with asymmetric kernel"""

    def fn(x, w):
        return F.conv2d(x, w, padding=(1, 2))

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels = 4

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, 3, 5)  # 3x5 kernel

    check_outputs(fn, conf, [x, w])


@pytest.mark.parametrize("size", [(1, 1, 4, 4), (3, 8, 32, 32), (2, 16, 64, 64)])
def test_conv2d_different_input_sizes(conf: Conf, size: tuple):
    """Test conv2d with different input tensor sizes"""

    def fn(x, w):
        return F.conv2d(x, w, padding=1)

    # Test various input sizes
    batch_size, in_channels, height, width = size
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    # TensorFloat-32 tensor cores are used by default, lowering precision
    check_outputs(fn, conf, [x, w], rtol=3e-5, atol=3e-5)


def test_conv2d_edge_cases(conf: Conf):
    """Test conv2d edge cases"""

    # Single pixel output
    def fn1(x, w):
        return F.conv2d(x, w)

    batch_size, in_channels = 1, 2
    out_channels, kernel_size = 3, 3

    x = torch.randn(batch_size, in_channels, 3, 3)  # Exactly kernel size
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)

    check_outputs(fn1, conf, [x, w])


def test_conv1d_basic(conf: Conf):
    """Test basic conv1d with default parameters"""

    def fn(x, w):
        return F.conv1d(x, w)

    batch_size, in_channels, length = 2, 3, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(out_channels, in_channels, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv1d_with_bias(conf: Conf):
    """Test conv1d with bias"""

    def fn(x, w, b):
        return F.conv1d(x, w, b)

    batch_size, in_channels, length = 2, 3, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(out_channels, in_channels, kernel_size)
    b = torch.randn(out_channels)

    check_outputs(fn, conf, [x, w, b])


def test_conv1d_stride(conf: Conf):
    """Test conv1d with stride"""

    def fn(x, w):
        return F.conv1d(x, w, stride=2)

    batch_size, in_channels, length = 2, 3, 32
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(out_channels, in_channels, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv1d_padding(conf: Conf):
    """Test conv1d with padding"""

    def fn(x, w):
        return F.conv1d(x, w, padding=2)

    batch_size, in_channels, length = 2, 3, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(out_channels, in_channels, kernel_size)

    check_outputs(fn, conf, [x, w])


@pytest.mark.xfail(reason="Dilation not implemented yet on max")
def test_conv1d_dilation(conf: Conf):
    """Test conv1d with dilation"""

    def fn(x, w):
        return F.conv1d(x, w, dilation=2)

    batch_size, in_channels, length = 2, 3, 32
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(out_channels, in_channels, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv1d_all_params(conf: Conf):
    """Test conv1d with all parameters specified"""

    def fn(x, w, b):
        return F.conv1d(x, w, b, stride=2, padding=1, dilation=1)

    batch_size, in_channels, length = 2, 3, 32
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(out_channels, in_channels, kernel_size)
    b = torch.randn(out_channels)

    check_outputs(fn, conf, [x, w, b])


def test_conv1d_1x1_kernel(conf: Conf):
    """Test conv1d with 1x1 kernel (pointwise convolution)"""

    def fn(x, w):
        return F.conv1d(x, w)

    batch_size, in_channels, length = 2, 3, 16
    out_channels, kernel_size = 4, 1

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(out_channels, in_channels, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv1d_large_kernel(conf: Conf):
    """Test conv1d with larger kernel"""

    def fn(x, w):
        return F.conv1d(x, w, padding=2)

    batch_size, in_channels, length = 2, 3, 16
    out_channels, kernel_size = 4, 5

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(out_channels, in_channels, kernel_size)

    check_outputs(fn, conf, [x, w])


@pytest.mark.parametrize("size", [(1, 1, 8), (3, 8, 32), (2, 16, 64)])
def test_conv1d_different_input_sizes(conf: Conf, size: tuple):
    """Test conv1d with different input tensor sizes"""

    def fn(x, w):
        return F.conv1d(x, w, padding=1)

    batch_size, in_channels, length = size
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(out_channels, in_channels, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv1d_edge_cases(conf: Conf):
    """Test conv1d edge cases"""

    # Single element output
    def fn1(x, w):
        return F.conv1d(x, w)

    batch_size, in_channels = 1, 2
    out_channels, kernel_size = 3, 3

    x = torch.randn(batch_size, in_channels, 3)  # Exactly kernel size
    w = torch.randn(out_channels, in_channels, kernel_size)

    check_outputs(fn1, conf, [x, w])


def test_conv_transpose1d_basic(conf: Conf):
    """Test basic conv_transpose1d"""

    def fn(x, w):
        return F.conv_transpose1d(x, w)

    batch_size, in_channels, length = 2, 3, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(in_channels, out_channels, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose1d_with_bias(conf: Conf):
    """Test conv_transpose1d with bias"""

    def fn(x, w, b):
        return F.conv_transpose1d(x, w, b)

    batch_size, in_channels, length = 2, 3, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(in_channels, out_channels, kernel_size)
    b = torch.randn(out_channels)

    check_outputs(fn, conf, [x, w, b])


def test_conv_transpose1d_stride(conf: Conf):
    """Test conv_transpose1d with stride"""

    def fn(x, w):
        return F.conv_transpose1d(x, w, stride=2)

    batch_size, in_channels, length = 2, 3, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(in_channels, out_channels, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose1d_padding(conf: Conf):
    """Test conv_transpose1d with padding"""

    def fn(x, w):
        return F.conv_transpose1d(x, w, padding=1)

    batch_size, in_channels, length = 2, 3, 16
    out_channels, kernel_size = 3, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(in_channels, out_channels, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose1d_output_padding(conf: Conf):
    """Test conv_transpose1d with output_padding"""

    def fn(x, w):
        return F.conv_transpose1d(x, w, stride=2, output_padding=1)

    batch_size, in_channels, length = 2, 3, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(in_channels, out_channels, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose1d_all_params(conf: Conf):
    """Test conv_transpose1d with all parameters"""

    def fn(x, w, b):
        return F.conv_transpose1d(x, w, b, stride=2, padding=1, output_padding=1)

    batch_size, in_channels, length = 2, 3, 16
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(in_channels, out_channels, kernel_size)
    b = torch.randn(out_channels)

    check_outputs(fn, conf, [x, w, b])


def test_conv1d_grouped(conf: Conf):
    """Test grouped conv1d (groups > 1)"""

    def fn(x, w, b):
        return F.conv1d(x, w, b, groups=2)

    batch_size, in_channels, length = 2, 4, 16
    out_channels, kernel_size = 6, 3
    groups = 2

    x = torch.randn(batch_size, in_channels, length)
    w = torch.randn(out_channels, in_channels // groups, kernel_size)
    b = torch.randn(out_channels)

    check_outputs(fn, conf, [x, w, b])


def test_conv1d_depthwise(conf: Conf):
    """Test depthwise conv1d (groups == in_channels)"""

    def fn(x, w):
        return F.conv1d(x, w, stride=2, padding=1, groups=4)

    batch_size, in_channels, length = 2, 4, 16
    kernel_size = 3
    groups = 4

    x = torch.randn(batch_size, in_channels, length)
    # depthwise: out_channels == in_channels, weight (C, 1, K)
    w = torch.randn(in_channels, in_channels // groups, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose1d_grouped(conf: Conf):
    """Test grouped conv_transpose1d (groups > 1)"""

    def fn(x, w, b):
        return F.conv_transpose1d(x, w, b, stride=2, groups=2)

    batch_size, in_channels, length = 2, 4, 16
    out_channels, kernel_size = 6, 3
    groups = 2

    x = torch.randn(batch_size, in_channels, length)
    # transposed weight shape: (in_channels, out_channels // groups, K)
    w = torch.randn(in_channels, out_channels // groups, kernel_size)
    b = torch.randn(out_channels)

    check_outputs(fn, conf, [x, w, b])


def test_conv2d_grouped(conf: Conf):
    """Test grouped conv2d (groups > 1)"""

    def fn(x, w):
        return F.conv2d(x, w, groups=3)

    batch_size, in_channels, height, width = 2, 6, 8, 8
    out_channels, kernel_size = 9, 3
    groups = 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose2d_basic(conf: Conf):
    """Test basic conv_transpose2d"""

    def fn(x, w):
        return F.conv_transpose2d(x, w)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(in_channels, out_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose2d_with_bias(conf: Conf):
    """Test conv_transpose2d with bias"""

    def fn(x, w, b):
        return F.conv_transpose2d(x, w, b)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(in_channels, out_channels, kernel_size, kernel_size)
    b = torch.randn(out_channels)

    check_outputs(fn, conf, [x, w, b])


def test_conv_transpose2d_stride_int(conf: Conf):
    """Test conv_transpose2d with integer stride"""

    def fn(x, w):
        return F.conv_transpose2d(x, w, stride=2)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(in_channels, out_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose2d_stride_tuple(conf: Conf):
    """Test conv_transpose2d with tuple stride"""

    def fn(x, w):
        return F.conv_transpose2d(x, w, stride=(2, 1))

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(in_channels, out_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose2d_padding_int(conf: Conf):
    """Test conv_transpose2d with integer padding"""

    def fn(x, w):
        return F.conv_transpose2d(x, w, padding=1)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(in_channels, out_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose2d_padding_tuple(conf: Conf):
    """Test conv_transpose2d with tuple padding"""

    def fn(x, w):
        return F.conv_transpose2d(x, w, padding=(1, 2))

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(in_channels, out_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose2d_output_padding(conf: Conf):
    """Test conv_transpose2d with output_padding"""

    def fn(x, w):
        return F.conv_transpose2d(x, w, stride=2, output_padding=1)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(in_channels, out_channels, kernel_size, kernel_size)

    check_outputs(fn, conf, [x, w])


def test_conv_transpose2d_all_params(conf: Conf):
    """Test conv_transpose2d with all parameters"""

    def fn(x, w, b):
        return F.conv_transpose2d(x, w, b, stride=2, padding=1, output_padding=1)

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(in_channels, out_channels, kernel_size, kernel_size)
    b = torch.randn(out_channels)

    check_outputs(fn, conf, [x, w, b])


def test_conv2d_combined_with_other_ops(conf: Conf):
    """Test conv2d combined with other operations"""

    def fn(x, w, b, y):
        conv_out = F.conv2d(x, w, b, padding=1)
        return conv_out + y

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 4, 3

    x = torch.randn(batch_size, in_channels, height, width)
    w = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    b = torch.randn(out_channels)
    # y should have same shape as conv output: (2, 4, 8, 8)
    y = torch.randn(batch_size, out_channels, height, width)

    check_outputs(fn, conf, [x, w, b, y])


def test_embedding_basic(conf: Conf):
    """Test basic embedding lookup"""

    def fn(indices, weight):
        return F.embedding(indices, weight)

    vocab_size, embedding_dim = 10, 5
    seq_length = 4

    # Create indices tensor (LongTensor)
    indices = torch.randint(0, vocab_size, (seq_length,))
    weight = torch.randn(vocab_size, embedding_dim)

    check_outputs(fn, conf, [indices, weight])


def test_embedding_2d_indices(conf: Conf):
    """Test embedding with 2D indices (batch processing)"""

    def fn(indices, weight):
        return F.embedding(indices, weight)

    vocab_size, embedding_dim = 20, 8
    batch_size, seq_length = 3, 6

    indices = torch.randint(0, vocab_size, (batch_size, seq_length))
    weight = torch.randn(vocab_size, embedding_dim)

    check_outputs(fn, conf, [indices, weight])


def test_embedding_3d_indices(conf: Conf):
    """Test embedding with 3D indices"""

    def fn(indices, weight):
        return F.embedding(indices, weight)

    vocab_size, embedding_dim = 15, 4
    batch_size, seq_length, depth = 2, 3, 4

    indices = torch.randint(0, vocab_size, (batch_size, seq_length, depth))
    weight = torch.randn(vocab_size, embedding_dim)

    check_outputs(fn, conf, [indices, weight])


def test_embedding_single_index(conf: Conf):
    """Test embedding with single index (scalar)"""

    def fn(indices, weight):
        return F.embedding(indices, weight)

    vocab_size, embedding_dim = 10, 3

    indices = torch.tensor(5)  # Scalar tensor
    weight = torch.randn(vocab_size, embedding_dim)

    check_outputs(fn, conf, [indices, weight])


def test_embedding_combined_with_other_ops(conf: Conf):
    """Test embedding combined with other operations"""

    def fn(indices, weight, bias):
        embedded = F.embedding(indices, weight)
        return embedded + bias

    vocab_size, embedding_dim = 10, 5
    seq_length = 4

    indices = torch.randint(0, vocab_size, (seq_length,))
    weight = torch.randn(vocab_size, embedding_dim)
    bias = torch.randn(embedding_dim)

    check_outputs(fn, conf, [indices, weight, bias])


def test_embedding_with_padding_idx(conf: Conf):
    """Test embedding with padding_idx parameter"""

    def fn(indices, weight):
        return F.embedding(indices, weight, padding_idx=0)

    vocab_size, embedding_dim = 8, 4

    # Include padding index (0) in the indices
    indices = torch.tensor([[0, 1, 2, 0, 3], [4, 0, 5, 6, 0]])
    weight = torch.randn(vocab_size, embedding_dim)

    check_outputs(fn, conf, [indices, weight])


def test_embedding_padding_idx_different_values(conf: Conf):
    """Test embedding with different padding_idx values"""

    def fn_pad_0(indices, weight):
        return F.embedding(indices, weight, padding_idx=0)

    def fn_pad_2(indices, weight):
        return F.embedding(indices, weight, padding_idx=2)

    vocab_size, embedding_dim = 6, 3

    indices_0 = torch.tensor([0, 1, 3, 0])  # Using 0 as padding
    indices_2 = torch.tensor([1, 2, 4, 2])  # Using 2 as padding
    weight = torch.randn(vocab_size, embedding_dim)

    check_outputs(fn_pad_0, conf, [indices_0, weight])
    check_outputs(fn_pad_2, conf, [indices_2, weight])


def test_embedding_padding_idx_scalar(conf: Conf):
    """Test embedding with padding_idx on scalar indices"""

    def fn(indices, weight):
        return F.embedding(indices, weight, padding_idx=0)

    vocab_size, embedding_dim = 5, 3

    indices = torch.tensor(0)  # Scalar padding index
    weight = torch.randn(vocab_size, embedding_dim)

    check_outputs(fn, conf, [indices, weight])


def test_tensor_slice_basic(conf: Conf):
    def fn(x):
        return x[1:3]  # Basic slice along first dimension

    x = torch.randn(5, 4)

    check_outputs(fn, conf, [x])


def test_tensor_slice_2d(conf: Conf):
    def fn(x):
        return x[1:3, 0:2]  # Slice along both dimensions

    x = torch.randn(5, 4)

    check_outputs(fn, conf, [x])


def test_tensor_slice_negative_index(conf: Conf):
    def fn(x):
        return x[-2:]  # Negative slice

    x = torch.randn(5, 3)

    check_outputs(fn, conf, [x])


def test_tensor_slice_with_step(conf: Conf):
    def fn(x):
        return x[1:10:2]  # Negative slice

    x = torch.randn(20, 20)

    check_outputs(fn, conf, [x])


def test_torch_matmul(conf: Conf):
    def fn(x, w):
        return torch.matmul(x, w)

    x = torch.randn(1, 3, 8)
    w = torch.randn(8, 16)

    check_outputs(fn, conf, [x, w])


def test_to_float(conf: Conf):
    def fn(x):
        return x.float()

    x = torch.randint(0, 10, (5,))

    check_outputs(fn, conf, [x])


def test_expand_basic(conf: Conf):
    """Test basic expand operation"""

    def fn(x):
        return x.expand(3, 4)

    x = torch.randn(1, 4)

    check_outputs(fn, conf, [x])


def test_expand_with_negative_one(conf: Conf):
    """Test expand with -1 (keep dimension unchanged)"""

    def fn(x):
        return x.expand(-1, 5)

    x = torch.randn(3, 1)

    check_outputs(fn, conf, [x])


def test_expand_multiple_dims(conf: Conf):
    """Test expand on tensor with multiple dimensions"""

    def fn(x):
        return x.expand(2, 3, 4)

    x = torch.randn(1, 1, 4)

    check_outputs(fn, conf, [x])


def test_expand_same_size(conf: Conf):
    """Test expand to same size (should be no-op)"""

    def fn(x):
        return x.expand(2, 3)

    x = torch.randn(2, 3)

    check_outputs(fn, conf, [x])


def test_expand_add_dimensions(conf: Conf):
    """Test expand adding new leading dimensions"""

    def fn(x):
        return x.expand(2, 3, 4)

    x = torch.randn(4)  # 1D tensor

    check_outputs(fn, conf, [x])


def test_expand_mixed_operations(conf: Conf):
    """Test expand combined with arithmetic operations"""

    def fn(x, y):
        expanded_x = x.expand(2, 3)
        return expanded_x + y

    x = torch.randn(1, 3)
    y = torch.randn(2, 3)

    check_outputs(fn, conf, [x, y])


def test_expand_with_scalar_broadcast(conf: Conf):
    """Test expand from scalar dimension"""

    def fn(x):
        return x.expand(5, 5)

    x = torch.randn(1, 1)

    check_outputs(fn, conf, [x])


def test_expand_complex_pattern(conf: Conf):
    """Test expand with complex dimension pattern"""

    def fn(x):
        return x.expand(2, -1, 4, -1)

    x = torch.randn(1, 3, 1, 5)

    check_outputs(fn, conf, [x])


def test_transpose_2d(conf: Conf):
    """Test basic transpose on 2D tensor"""

    def fn(x):
        return x.transpose(0, 1)

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_transpose_3d_first_last(conf: Conf):
    """Test transpose swapping first and last dimensions on 3D tensor"""

    def fn(x):
        return x.transpose(0, 2)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_transpose_3d_middle_dims(conf: Conf):
    """Test transpose swapping middle dimensions on 3D tensor"""

    def fn(x):
        return x.transpose(1, 2)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_transpose_negative_dims(conf: Conf):
    """Test transpose with negative dimension indices"""

    def fn(x):
        return x.transpose(-2, -1)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_transpose_same_dim(conf: Conf):
    """Test transpose with same dimension (should be no-op)"""

    def fn(x):
        return x.transpose(1, 1)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_transpose_4d(conf: Conf):
    """Test transpose on 4D tensor"""

    def fn(x):
        return x.transpose(1, 3)

    x = torch.randn(2, 3, 4, 5)

    check_outputs(fn, conf, [x])


def test_transpose_batch_dimension(conf: Conf):
    """Test transpose involving batch dimension"""

    def fn(x):
        return x.transpose(0, 1)

    x = torch.randn(8, 16, 32)

    check_outputs(fn, conf, [x])


def test_transpose_with_arithmetic(conf: Conf):
    """Test transpose combined with arithmetic operations"""

    def fn(x, y):
        x_t = x.transpose(0, 1)
        return x_t + y

    x = torch.randn(3, 4)
    y = torch.randn(4, 3)

    check_outputs(fn, conf, [x, y])


def test_transpose_multiple_ops(conf: Conf):
    """Test multiple transpose operations"""

    def fn(x):
        # First transpose: (2, 3, 4) -> (2, 4, 3)
        x1 = x.transpose(1, 2)
        # Second transpose: (2, 4, 3) -> (4, 2, 3)
        x2 = x1.transpose(0, 1)
        return x2

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_transpose_with_other_methods(conf: Conf):
    """Test transpose combined with other tensor methods"""

    def fn(x):
        x_t = x.transpose(0, 1)
        return x_t.expand(-1, 5, -1)

    x = torch.randn(1, 3, 4)

    check_outputs(fn, conf, [x])


def test_transpose_scalar_like(conf: Conf):
    """Test transpose on tensor with singleton dimensions"""

    def fn(x):
        return x.transpose(0, 2)

    x = torch.randn(1, 3, 1)

    check_outputs(fn, conf, [x])


def test_tensor_cos_method(conf: Conf):
    """Test tensor.cos() method"""

    def fn(x):
        return x.cos()

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_sin_method(conf: Conf):
    """Test tensor.sin() method"""

    def fn(x):
        return x.sin()

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_atanh_method(conf: Conf):
    """Test tensor.atanh() method"""

    def fn(x):
        return x.atanh()

    # atanh is defined for |x| < 1, so we need values in this range
    x = torch.rand(3, 4) * 1.8 - 0.9  # Values in range (-0.9, 0.9)

    check_outputs(fn, conf, [x])


def test_tensor_cos_sin_combined(conf: Conf):
    """Test combining tensor.cos() and tensor.sin() methods"""

    def fn(x):
        return x.cos() + x.sin()

    x = torch.randn(2, 3)

    check_outputs(fn, conf, [x])


def test_tensor_cos_with_arithmetic(conf: Conf):
    """Test tensor.cos() combined with arithmetic operations"""

    def fn(x, y):
        return x.cos() * y

    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    check_outputs(fn, conf, [x, y])


def test_tensor_sin_with_arithmetic(conf: Conf):
    """Test tensor.sin() combined with arithmetic operations"""

    def fn(x, y):
        return x.sin() - y

    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    check_outputs(fn, conf, [x, y])


def test_tensor_cos_sin_chained(conf: Conf):
    """Test chained tensor.cos().sin() operations"""

    def fn(x):
        return x.cos().sin()

    x = torch.randn(2, 3)

    check_outputs(fn, conf, [x])


def test_tensor_trig_with_transpose(conf: Conf):
    """Test tensor trigonometric methods with transpose"""

    def fn(x):
        return x.transpose(0, 1).cos()

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_cos_sin_different_shapes(conf: Conf, tensor_shapes: tuple):
    """Test tensor.cos() and tensor.sin() with different tensor shapes"""

    def fn_cos(x):
        return x.cos()

    def fn_sin(x):
        return x.sin()

    x = torch.randn(tensor_shapes)

    check_outputs(fn_cos, conf, [x])
    check_outputs(fn_sin, conf, [x])


def test_tensor_pow_method(device: str):
    """Test tensor.pow() method"""

    def fn(x, y):
        return x.pow(y)

    x = torch.randn(3, 4).abs() + 0.1  # Avoid negative base
    y = torch.randn(3, 4) * 2  # Keep exponent reasonable

    check_functions_are_equivalent(fn, device, [x, y])


def test_tensor_pow_scalar_exponent(conf: Conf):
    """Test tensor.pow() with scalar exponent"""

    def fn(x):
        return x.pow(2)

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_pow_negative_exponent(conf: Conf):
    """Test tensor.pow() with negative exponent"""

    def fn(x):
        return x.pow(-2)

    x = torch.randn(3, 4).abs() + 1.0  # Avoid division by zero

    check_outputs(fn, conf, [x])


def test_tensor_pow_fractional_exponent(conf: Conf):
    """Test tensor.pow() with fractional exponent"""

    def fn(x):
        return x.pow(0.5)  # Square root

    x = torch.randn(3, 4).abs() + 0.1  # Ensure positive for square root

    check_outputs(fn, conf, [x])


def test_tensor_pow_with_arithmetic(conf: Conf):
    """Test tensor.pow() combined with arithmetic operations"""

    def fn(x, y, z):
        return x.pow(y) + z

    x = torch.randn(3, 4).abs() + 0.1
    y = torch.randn(3, 4) * 2
    z = torch.randn(3, 4)

    check_outputs(fn, conf, [x, y, z])


def test_tensor_pow_chained(conf: Conf):
    """Test chained tensor.pow() operations"""

    def fn(x):
        return x.pow(2).pow(0.5)  # Should be approximately x

    x = torch.randn(3, 4).abs() + 0.1

    check_outputs(fn, conf, [x])


def test_tensor_pow_broadcast(conf: Conf):
    """Test tensor.pow() with broadcasting"""

    def fn(x, y):
        return x.pow(y)

    x = torch.randn(3, 4).abs() + 0.1
    y = torch.randn(1, 4) * 2

    check_outputs(fn, conf, [x, y])


def test_tensor_pow_different_shapes(conf: Conf, tensor_shapes: tuple):
    """Test tensor.pow() with different tensor shapes"""

    def fn(x):
        return x.pow(2)

    x = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [x])


def test_tensor_pow_with_other_methods(conf: Conf):
    """Test tensor.pow() combined with other tensor methods"""

    def fn(x):
        return x.transpose(0, 1).pow(2).cos()

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_change_device_to_cpu(device: str):
    """Test changing device to CPU"""

    def fn(x):
        return x.to("cpu")

    x = torch.randn(1, 3)

    check_functions_are_equivalent(fn, device, [x])


def test_change_device_to_cpu_by_device(device: str):
    """Test changing device to CPU"""

    def fn(x):
        return x.to(torch.device("cpu"))

    x = torch.randn(1, 3)

    check_functions_are_equivalent(fn, device, [x])


def test_change_device_to_cuda(device: str, cuda_available: bool):
    """Test changing device to CUDA"""
    if not cuda_available:
        pytest.skip("CUDA not available")

    def fn(x):
        return x.to("cuda")

    x = torch.randn(1, 3)

    check_functions_are_equivalent(fn, device, [x])


def test_change_device_to_cuda_by_device(device: str, cuda_available: bool):
    """Test changing device to CUDA"""
    if not cuda_available:
        pytest.skip("CUDA not available")

    def fn(x):
        return x.to(torch.device("cuda"))

    x = torch.randn(1, 3)

    check_functions_are_equivalent(fn, device, [x])


def test_to_with_dtype_keyword(device: str):
    """Test tensor.to() with dtype keyword argument"""

    def fn(x):
        return x.to(dtype=torch.float32)

    x = torch.randint(0, 10, (2, 3))

    check_functions_are_equivalent(fn, device, [x])


def test_to_with_device_keyword(device: str):
    """Test tensor.to() with device keyword argument"""

    def fn(x):
        return x.to(device="cpu")

    x = torch.randn(2, 3)

    check_functions_are_equivalent(fn, device, [x])


def test_to_with_device_dtype_keywords(device: str):
    """Test tensor.to() with both device and dtype keyword arguments"""

    def fn(x):
        return x.to(device="cpu", dtype=torch.float32)

    x = torch.randint(0, 10, (2, 3))

    check_functions_are_equivalent(fn, device, [x])


def test_to_with_torch_device_object(device: str):
    """Test tensor.to() with torch.device object"""

    def fn(x):
        return x.to(torch.device("cpu"))

    x = torch.randn(2, 3)

    check_functions_are_equivalent(fn, device, [x])


def test_to_with_torch_device_object_cuda(device: str, cuda_available: bool):
    """Test tensor.to() with torch.device object for CUDA"""
    if not cuda_available:
        pytest.skip("CUDA not available")

    def fn(x):
        return x.to(torch.device("cuda:0"))

    x = torch.randn(2, 3)

    check_functions_are_equivalent(fn, device, [x])


def test_to_with_dtype_positional(device: str):
    """Test tensor.to() with dtype as positional argument"""

    def fn(x):
        return x.to(torch.float32)

    x = torch.randint(0, 10, (2, 3))

    check_functions_are_equivalent(fn, device, [x])


def test_to_dtype_conversion_int_to_float(device: str):
    """Test converting integer tensor to float"""

    def fn(x):
        return x.to(dtype=torch.float32)

    x = torch.randint(-5, 5, (3, 4))

    check_functions_are_equivalent(fn, device, [x])


def test_to_dtype_conversion_float_to_int(device: str):
    """Test converting float tensor to int"""

    def fn(x):
        return x.to(dtype=torch.int32)

    x = torch.randn(3, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_to_dtype_conversion_double_to_float(device: str):
    """Test converting double tensor to float"""

    def fn(x):
        return x.to(dtype=torch.float32)

    x = torch.randn(3, 4, dtype=torch.float64)

    check_functions_are_equivalent(fn, device, [x])


def test_to_combined_with_operations(device: str):
    """Test tensor.to() combined with other operations"""

    def fn(x, y):
        x_converted = x.to(dtype=torch.float32)
        return x_converted + y

    x = torch.randint(0, 10, (3, 4))
    y = torch.randn(3, 4)

    check_functions_are_equivalent(fn, device, [x, y])


def test_to_device_transfer_with_computation(device: str):
    """Test device transfer followed by computation"""

    def fn(x):
        x_cpu = x.to("cpu")
        return x_cpu * 2

    x = torch.randn(3, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_autocast_enter_exit(conf: Conf):
    """Test autocast enter and exit functionality"""

    def fn(x):
        with torch.amp.autocast("cuda", enabled=True):
            return x + 1.0

    x = torch.randn(2, 3)

    # Test on CPU device only as autocast behavior may vary
    check_outputs(fn, conf, [x])


def test_complex_to_operations(device: str):
    """Test complex combinations of .to() operations"""

    def fn(x):
        # Convert to float first, then back to int
        x_float = x.to(dtype=torch.float32)
        result = x_float * 2.5
        return result.to(dtype=torch.int32)

    x = torch.randint(1, 5, (2, 3))

    check_functions_are_equivalent(fn, device, [x])


def test_mean_no_dim(conf: Conf, tensor_shapes: tuple):
    """Test mean without specifying dimensions (reduce all)"""

    def fn(x):
        return torch.mean(x)

    a = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a])


def test_mean_single_dim(device: str, tensor_shapes: tuple):
    """Test mean with single dimension"""

    def fn(x):
        return torch.mean(x, dim=1)

    a = torch.randn(tensor_shapes) if len(tensor_shapes) > 1 else torch.randn(3, 4)

    check_functions_are_equivalent(fn, device, [a])


def test_mean_negative_dim(device: str, tensor_shapes: tuple):
    """Test mean with negative dimension"""

    def fn(x):
        return torch.mean(x, dim=-1)

    a = torch.randn(tensor_shapes)

    check_functions_are_equivalent(fn, device, [a])


def test_mean_keepdim_true(device: str, tensor_shapes: tuple):
    """Test mean with keepdim=True"""

    def fn(x):
        return torch.mean(x, dim=1, keepdim=True)

    a = torch.randn(tensor_shapes) if len(tensor_shapes) > 1 else torch.randn(3, 4)

    check_functions_are_equivalent(fn, device, [a])


def test_mean_multiple_dims(device: str):
    """Test mean with multiple dimensions"""

    def fn(x):
        return torch.mean(x, dim=(1, 2))

    a = torch.randn(2, 3, 4)

    check_functions_are_equivalent(fn, device, [a])


def test_mean_multiple_dims_keepdim(device: str):
    """Test mean with multiple dimensions and keepdim=True"""

    def fn(x):
        return torch.mean(x, dim=(0, 2), keepdim=True)

    a = torch.randn(2, 3, 4)

    check_functions_are_equivalent(fn, device, [a])


def test_tensor_mean_method(conf: Conf, tensor_shapes: tuple):
    """Test tensor.mean() method"""

    def fn(x):
        return x.mean()

    a = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [a])


def test_tensor_mean_method_with_dim(device: str, tensor_shapes: tuple):
    """Test tensor.mean(dim) method"""

    def fn(x):
        return x.mean(dim=1)

    a = torch.randn(tensor_shapes) if len(tensor_shapes) > 1 else torch.randn(3, 4)

    check_functions_are_equivalent(fn, device, [a])


def test_mean_3d_tensor(device: str):
    def fn(x):
        return torch.mean(x, dim=1)

    a = torch.randn(2, 3, 4)

    check_functions_are_equivalent(fn, device, [a])


def test_mean_3d_tensor_change_dtype(device: str):
    def fn(x):
        return torch.mean(x, dim=1, dtype=torch.float32)

    a = torch.randn(2, 3, 4).to(torch.int32)

    check_functions_are_equivalent(fn, device, [a])


def test_mean_combined_with_arithmetic(device: str, tensor_shapes: tuple):
    """Test mean combined with arithmetic operations"""

    def fn(x, y):
        mean_x = torch.mean(x, dim=-1, keepdim=True)
        return mean_x + y

    a = torch.randn(tensor_shapes)
    # Create y with compatible shape for broadcasting
    if len(tensor_shapes) == 0:
        y = torch.randn(())
    else:
        y_shape = list(tensor_shapes)
        y_shape[-1] = 1  # Make last dimension 1 for broadcasting
        y = torch.randn(y_shape)

    check_functions_are_equivalent(fn, device, [a, y])


def test_rsqrt_function(conf: Conf):
    """Test torch.rsqrt() function"""

    def fn(x):
        return torch.rsqrt(x)

    x = torch.randn(3, 4).abs() + 0.1  # Ensure positive values for rsqrt

    check_outputs(fn, conf, [x])


def test_sqrt_function(conf: Conf):
    """Test torch.sqrt() function"""

    def fn(x):
        return torch.sqrt(x)

    x = torch.randn(3, 4).abs() + 0.01  # Ensure positive values for sqrt

    check_outputs(fn, conf, [x])


def test_tensor_rsqrt_method(conf: Conf):
    """Test tensor.rsqrt() method"""

    def fn(x):
        return x.rsqrt()

    x = torch.randn(3, 4).abs() + 0.1  # Ensure positive values

    check_outputs(fn, conf, [x])


def test_tensor_sqrt_method(conf: Conf):
    """Test tensor.sqrt() method"""

    def fn(x):
        return x.sqrt()

    x = torch.randn(3, 4).abs() + 0.01  # Ensure positive values

    check_outputs(fn, conf, [x])


def test_rsqrt_different_shapes(conf: Conf, tensor_shapes: tuple):
    """Test rsqrt with different tensor shapes"""

    def fn(x):
        return torch.rsqrt(x)

    x = torch.randn(tensor_shapes).abs() + 0.1

    check_outputs(fn, conf, [x])


def test_sqrt_different_shapes(conf: Conf, tensor_shapes: tuple):
    """Test sqrt with different tensor shapes"""

    def fn(x):
        return torch.sqrt(x)

    x = torch.randn(tensor_shapes).abs() + 0.01

    check_outputs(fn, conf, [x])


def test_rsqrt_with_ones(conf: Conf):
    """Test rsqrt with tensor of ones (should return ones)"""

    def fn(x):
        return torch.rsqrt(x)

    x = torch.ones(2, 3)

    check_outputs(fn, conf, [x])


def test_sqrt_with_ones(conf: Conf):
    """Test sqrt with tensor of ones (should return ones)"""

    def fn(x):
        return torch.sqrt(x)

    x = torch.ones(2, 3)

    check_outputs(fn, conf, [x])


def test_rsqrt_with_powers_of_two(conf: Conf):
    """Test rsqrt with powers of 2 for exact mathematical results"""

    def fn(x):
        return torch.rsqrt(x)

    x = torch.tensor([1.0, 4.0, 16.0, 64.0])  # Powers of 2

    check_outputs(fn, conf, [x])


def test_sqrt_with_perfect_squares(conf: Conf):
    """Test sqrt with perfect squares for exact mathematical results"""

    def fn(x):
        return torch.sqrt(x)

    x = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])  # Perfect squares

    check_outputs(fn, conf, [x])


def test_rsqrt_sqrt_relationship(conf: Conf):
    """Test mathematical relationship: rsqrt(x) * sqrt(x) should equal x"""

    def fn(x):
        return torch.rsqrt(x) * torch.sqrt(x)

    x = torch.randn(3, 4).abs() + 0.1

    check_outputs(fn, conf, [x])


def test_rsqrt_combined_with_arithmetic(conf: Conf):
    """Test rsqrt combined with arithmetic operations"""

    def fn(x, y):
        rsqrt_x = torch.rsqrt(x)
        return rsqrt_x + y

    x = torch.randn(3, 4).abs() + 0.1
    y = torch.randn(3, 4)

    check_outputs(fn, conf, [x, y])


def test_sqrt_combined_with_arithmetic(conf: Conf):
    """Test sqrt combined with arithmetic operations"""

    def fn(x, y):
        sqrt_x = torch.sqrt(x)
        return sqrt_x * y

    x = torch.randn(3, 4).abs() + 0.01
    y = torch.randn(3, 4)

    check_outputs(fn, conf, [x, y])


def test_chained_sqrt_rsqrt_operations(conf: Conf):
    """Test chained sqrt and rsqrt operations"""

    def fn(x):
        # This should approximately equal x (with small numerical errors)
        return torch.sqrt(torch.rsqrt(x)).pow(2)

    x = torch.randn(3, 4).abs() + 0.1

    check_outputs(fn, conf, [x])


def test_rsqrt_with_trigonometric_functions(conf: Conf):
    """Test rsqrt combined with trigonometric functions"""

    def fn(x):
        rsqrt_x = torch.rsqrt(x)
        return torch.sin(rsqrt_x)

    x = torch.randn(3, 4).abs() + 0.1

    check_outputs(fn, conf, [x])


def test_sqrt_with_trigonometric_functions(conf: Conf):
    """Test sqrt combined with trigonometric functions"""

    def fn(x):
        sqrt_x = torch.sqrt(x)
        return torch.cos(sqrt_x)

    x = torch.randn(3, 4).abs() + 0.01

    check_outputs(fn, conf, [x])


def test_tensor_methods_chain_sqrt_rsqrt(conf: Conf):
    """Test chaining tensor methods with sqrt and rsqrt"""

    def fn(x):
        return x.abs().sqrt().rsqrt()

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_sqrt_rsqrt_with_transpose(conf: Conf):
    """Test sqrt and rsqrt with transpose operations"""

    def fn(x):
        x_t = x.transpose(0, 1)
        return torch.sqrt(x_t) + torch.rsqrt(x_t + 0.1)

    x = torch.randn(3, 4).abs() + 0.01

    check_outputs(fn, conf, [x])


def test_sqrt_rsqrt_broadcasting(conf: Conf):
    """Test sqrt and rsqrt with broadcasting"""

    def fn(x, y):
        sqrt_x = torch.sqrt(x)
        rsqrt_y = torch.rsqrt(y)
        return sqrt_x + rsqrt_y

    x = torch.randn(3, 1).abs() + 0.01
    y = torch.randn(1, 4).abs() + 0.1

    check_outputs(fn, conf, [x, y])


def test_linear_basic(conf: Conf):
    """Test basic linear function without bias"""

    def fn(input, weight):
        return F.linear(input, weight)

    in_features, out_features = 4, 3
    batch_size = 2

    input = torch.randn(batch_size, in_features)
    weight = torch.randn(out_features, in_features)

    check_outputs(fn, conf, [input, weight])


def test_linear_with_bias(conf: Conf):
    """Test linear function with bias"""

    def fn(input, weight, bias):
        return F.linear(input, weight, bias)

    in_features, out_features = 4, 3
    batch_size = 2

    input = torch.randn(batch_size, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)

    check_outputs(fn, conf, [input, weight, bias])


def test_linear_small_dimensions(conf: Conf):
    """Test linear function with small dimensions"""

    def fn(input, weight):
        return F.linear(input, weight)

    in_features, out_features = 8, 16
    batch_size = 3

    input = torch.randn(batch_size, in_features)
    weight = torch.randn(out_features, in_features)

    # TensorFloat-32 tensor cores are used by default, lowering precision
    check_outputs(fn, conf, [input, weight], atol=1e-2, rtol=1e-2)


def test_linear_medium_dimensions(conf: Conf):
    """Test linear function with medium dimensions"""

    def fn(input, weight):
        return F.linear(input, weight)

    in_features, out_features = 32, 10
    batch_size = 3

    input = torch.randn(batch_size, in_features)
    weight = torch.randn(out_features, in_features)

    check_outputs(fn, conf, [input, weight])


def test_linear_single_dimension(conf: Conf):
    """Test linear function with single dimensions"""

    def fn(input, weight):
        return F.linear(input, weight)

    in_features, out_features = 1, 1
    batch_size = 3

    input = torch.randn(batch_size, in_features)
    weight = torch.randn(out_features, in_features)

    check_outputs(fn, conf, [input, weight])


def test_linear_3d_input(conf: Conf):
    """Test linear function with 3D input (batch, sequence, features)"""

    def fn(input, weight, bias):
        return F.linear(input, weight, bias)

    batch_size, seq_length = 2, 5
    in_features, out_features = 8, 6

    input = torch.randn(batch_size, seq_length, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)

    check_outputs(fn, conf, [input, weight, bias])


def test_linear_4d_input(conf: Conf):
    """Test linear function with 4D input (..., features)"""

    def fn(input, weight, bias):
        return F.linear(input, weight, bias=bias)

    batch_size, height, width = 2, 3, 4
    in_features, out_features = 7, 5

    input = torch.randn(batch_size, height, width, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)

    check_outputs(fn, conf, [input, weight, bias])


def test_linear_1d_input(conf: Conf):
    """Test linear function with 1D input (just features)"""

    def fn(input, weight, bias):
        return F.linear(input, weight, bias)

    in_features, out_features = 6, 4

    input = torch.randn(in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)

    check_outputs(fn, conf, [input, weight, bias])


def test_linear_chained(conf: Conf):
    """Test chained linear functions (simple MLP)"""

    def fn(input, weight1, bias1, weight2, bias2):
        hidden = F.linear(input, weight1, bias1)
        output = F.linear(hidden, weight2, bias2)
        return output

    in_features, hidden_features, out_features = 4, 6, 2
    batch_size = 3

    input = torch.randn(batch_size, in_features)
    weight1 = torch.randn(hidden_features, in_features)
    bias1 = torch.randn(hidden_features)
    weight2 = torch.randn(out_features, hidden_features)
    bias2 = torch.randn(out_features)

    check_outputs(fn, conf, [input, weight1, bias1, weight2, bias2])


def test_linear_broadcasting(conf: Conf):
    """Test linear function with broadcasting scenarios"""

    def fn(input, weight, bias):
        return F.linear(input, weight, bias)

    in_features, out_features = 4, 3
    batch_size, seq_length = 2, 5

    # Test with different batch shapes
    input = torch.randn(batch_size, seq_length, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)  # Should broadcast across batch and sequence dims

    check_outputs(fn, conf, [input, weight, bias])


def test_linear_single_feature(conf: Conf):
    """Test linear function with single input/output feature"""

    def fn(input, weight, bias):
        return F.linear(input, weight, bias)

    in_features, out_features = 1, 1
    batch_size = 3

    input = torch.randn(batch_size, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)

    check_outputs(fn, conf, [input, weight, bias])


def test_linear_large_dimensions(conf: Conf):
    """Test linear function with larger dimensions"""

    def fn(input, weight):
        return F.linear(input, weight)

    in_features, out_features = 128, 64
    batch_size = 4

    input = torch.randn(batch_size, in_features)
    weight = torch.randn(out_features, in_features)

    # TensorFloat-32 tensor cores are used by default, lowering precision
    check_outputs(fn, conf, [input, weight], atol=1e-2, rtol=1e-2)


def test_linear_with_transpose(conf: Conf):
    """Test linear function combined with transpose operations"""

    def fn(input, weight, bias):
        # Apply linear first, then transpose the result
        linear_out = F.linear(input, weight, bias)
        return linear_out.transpose(0, 1)  # Transpose output dimensions

    in_features, out_features = 6, 4
    batch_size = 3

    input = torch.randn(batch_size, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.randn(out_features)

    check_outputs(fn, conf, [input, weight, bias])


def test_linear_zero_bias(conf: Conf):
    """Test linear function with zero bias"""

    def fn(input, weight, bias):
        return F.linear(input, weight, bias)

    in_features, out_features = 5, 3
    batch_size = 2

    input = torch.randn(batch_size, in_features)
    weight = torch.randn(out_features, in_features)
    bias = torch.zeros(out_features)  # Zero bias

    check_outputs(fn, conf, [input, weight, bias])


def test_tensor_view_basic(conf: Conf):
    """Test basic tensor.view() operation"""

    def fn(x):
        return x.view(6, 4)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_view_with_negative_one(conf: Conf):
    """Test tensor.view() with -1 (infer dimension)"""

    def fn(x):
        return x.view(-1, 4)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_view_flatten(conf: Conf):
    """Test tensor.view() to flatten tensor"""

    def fn(x):
        return x.view(-1)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_view_2d_to_3d(conf: Conf):
    """Test tensor.view() from 2D to 3D"""

    def fn(x):
        return x.view(2, 3, 4)

    x = torch.randn(6, 4)

    check_outputs(fn, conf, [x])


def test_tensor_view_3d_to_2d(conf: Conf):
    """Test tensor.view() from 3D to 2D"""

    def fn(x):
        return x.view(6, -1)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_view_same_shape(conf: Conf):
    """Test tensor.view() with same shape (no-op)"""

    def fn(x):
        return x.view(2, 3, 4)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_view_single_dimension(conf: Conf):
    """Test tensor.view() creating single dimension"""

    def fn(x):
        return x.view(24, 1)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_view_multiple_negative_one_dimensions(conf: Conf):
    """Test tensor.view() with multiple inferred dimensions"""

    def fn(x):
        return x.view(2, -1, 2)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_view_with_arithmetic(conf: Conf):
    """Test tensor.view() combined with arithmetic operations"""

    def fn(x, y):
        x_reshaped = x.view(-1, 4)
        return x_reshaped + y

    x = torch.randn(3, 2, 4)
    y = torch.randn(6, 4)

    check_outputs(fn, conf, [x, y])


def test_tensor_view_chained_operations(conf: Conf):
    """Test chained tensor.view() operations"""

    def fn(x):
        return x.view(6, 4).view(2, 12).view(-1)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_view_with_transpose(conf: Conf):
    """Test tensor.view() combined with transpose"""

    def fn(x):
        # This should work since we're not changing the transpose result's shape
        x_t = x.transpose(0, 1)
        return x_t.view(3, 2, 4)  # Same total shape, just explicit dimensions

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_view_scalar_like(conf: Conf):
    """Test tensor.view() with scalar-like tensors"""

    def fn(x):
        return x.view(1, 1, 1)

    x = torch.randn(1)

    check_outputs(fn, conf, [x])


def test_tensor_view_large_dimensions(conf: Conf):
    """Test tensor.view() with larger dimensions"""

    def fn(x):
        return x.view(8, -1)

    x = torch.randn(2, 4, 16)

    check_outputs(fn, conf, [x])


def test_tensor_view_with_other_methods(conf: Conf):
    """Test tensor.view() combined with other tensor methods"""

    def fn(x):
        return x.abs().view(-1, 4).cos()

    x = torch.randn(3, 2, 4)

    check_outputs(fn, conf, [x])


def test_tensor_view_broadcasting_prep(conf: Conf):
    """Test tensor.view() for broadcasting preparation"""

    def fn(x, y):
        x_reshaped = x.view(2, 3, 1)
        return x_reshaped + y

    x = torch.randn(6)
    y = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x, y])


def test_tensor_contiguous_basic(conf: Conf):
    """Test basic tensor.contiguous() operation"""

    def fn(x):
        return x.contiguous()

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_contiguous_with_transpose(conf: Conf):
    """Test tensor.contiguous() after transpose"""

    def fn(x):
        x_t = x.transpose(0, 1)
        return x_t.contiguous()

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_contiguous_view_chain(conf: Conf):
    """Test tensor.contiguous().view() chain"""

    def fn(x):
        x_t = x.transpose(0, 1)
        return x_t.contiguous().view(-1, 4)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_unsqueeze_basic(conf: Conf):
    """Test basic tensor.unsqueeze() operation"""

    def fn(x):
        return x.unsqueeze(0)

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_unsqueeze_middle_dim(conf: Conf):
    """Test tensor.unsqueeze() in middle dimension"""

    def fn(x):
        return x.unsqueeze(1)

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_unsqueeze_last_dim(conf: Conf):
    """Test tensor.unsqueeze() at last dimension"""

    def fn(x):
        return x.unsqueeze(-1)

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_unsqueeze_negative_dim(conf: Conf):
    """Test tensor.unsqueeze() with negative dimension"""

    def fn(x):
        return x.unsqueeze(-2)

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_unsqueeze_multiple_ops(conf: Conf):
    """Test multiple tensor.unsqueeze() operations"""

    def fn(x):
        return x.unsqueeze(0).unsqueeze(-1)

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_tensor_unsqueeze_with_view(conf: Conf):
    """Test tensor.unsqueeze() combined with view()"""

    def fn(x):
        x_unsq = x.unsqueeze(1)  # (2, 3) -> (2, 1, 3)
        return x_unsq.view(2, 3)  # (2, 1, 3) -> (2, 3)

    x = torch.randn(2, 3)

    check_outputs(fn, conf, [x])


def test_tensor_unsqueeze_1d_tensor(conf: Conf):
    """Test tensor.unsqueeze() on 1D tensor"""

    def fn(x):
        return x.unsqueeze(0)

    x = torch.randn(5)

    check_outputs(fn, conf, [x])


def test_tensor_unsqueeze_scalar(conf: Conf):
    """Test tensor.unsqueeze() on scalar tensor"""

    def fn(x):
        return x.unsqueeze(0)

    x = torch.randn(())

    check_outputs(fn, conf, [x])


def test_unary_negation(conf: Conf):
    """Test unary negation operator (-x)"""

    def fn(x):
        return -x

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_negation_with_arithmetic(conf: Conf):
    """Test negation combined with arithmetic operations"""

    def fn(x, y):
        return -x + y

    x = torch.randn(3, 4)
    y = torch.randn(3, 4)

    check_outputs(fn, conf, [x, y])


def test_double_negation(conf: Conf):
    """Test double negation (-(-x))"""

    def fn(x):
        return -(-x)

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


def test_negation_different_shapes(conf: Conf, tensor_shapes: tuple):
    """Test negation with different tensor shapes"""

    def fn(x):
        return -x

    x = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [x])


def test_max_pool2d_basic(conf: Conf):
    """Test basic max_pool2d operation"""

    def fn(x):
        return F.max_pool2d(x, kernel_size=2)

    batch_size, channels, height, width = 2, 3, 8, 8
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_max_pool2d_with_stride(conf: Conf):
    """Test max_pool2d with custom stride"""

    def fn(x):
        return F.max_pool2d(x, kernel_size=3, stride=2)

    batch_size, channels, height, width = 2, 4, 12, 12
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_max_pool2d_with_padding(conf: Conf):
    """Test max_pool2d with padding"""

    def fn(x):
        return F.max_pool2d(x, kernel_size=2, padding=1)

    batch_size, channels, height, width = 2, 3, 6, 6
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_max_pool2d_asymmetric_kernel(conf: Conf):
    """Test max_pool2d with asymmetric kernel"""

    def fn(x):
        return F.max_pool2d(x, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))

    batch_size, channels, height, width = 2, 3, 8, 9
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_max_pool2d_various_sizes(conf: Conf):
    """Test max_pool2d with various input sizes"""

    def fn(x):
        return F.max_pool2d(x, kernel_size=2, stride=2)

    # Test different sizes
    for height, width in [(16, 16), (32, 24), (7, 11)]:
        batch_size, channels = 1, 2
        x = torch.randn(batch_size, channels, height, width)

        check_outputs(fn, conf, [x])


def test_adaptive_avg_pool2d_global(device: str):
    """Test adaptive_avg_pool2d with (1, 1) output (global pooling)"""

    def fn(x):
        return F.adaptive_avg_pool2d(x, (1, 1))

    batch_size, channels, height, width = 2, 3, 8, 8
    x = torch.randn(batch_size, channels, height, width)

    check_functions_are_equivalent(fn, device, [x])


def test_adaptive_avg_pool2d_7x7(conf: Conf):
    """Test adaptive_avg_pool2d with (7, 7) output like in VGG"""

    def fn(x):
        return F.adaptive_avg_pool2d(x, (7, 7))

    batch_size, channels, height, width = 2, 512, 14, 14
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_adaptive_avg_pool2d_various_outputs(conf: Conf):
    """Test adaptive_avg_pool2d with various output sizes"""

    def fn_2x2(x):
        return F.adaptive_avg_pool2d(x, (2, 2))

    def fn_4x4(x):
        return F.adaptive_avg_pool2d(x, (4, 4))

    batch_size, channels, height, width = 2, 64, 16, 16
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn_2x2, conf, [x])
    check_outputs(fn_4x4, conf, [x])


def test_avg_pool2d_basic(conf: Conf):
    """Test basic avg_pool2d with 2x2 kernel"""

    def fn(x):
        return F.avg_pool2d(x, kernel_size=2)

    batch_size, channels, height, width = 1, 3, 8, 8
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_avg_pool2d_with_stride(conf: Conf):
    """Test avg_pool2d with custom stride"""

    def fn(x):
        return F.avg_pool2d(x, kernel_size=3, stride=2)

    batch_size, channels, height, width = 2, 16, 10, 10
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_avg_pool2d_with_padding(conf: Conf):
    """Test avg_pool2d with padding"""

    def fn(x):
        return F.avg_pool2d(x, kernel_size=2, stride=2, padding=1)

    batch_size, channels, height, width = 1, 8, 6, 6
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_avg_pool2d_asymmetric_kernel(conf: Conf):
    """Test avg_pool2d with asymmetric kernel size"""

    def fn(x):
        return F.avg_pool2d(x, kernel_size=(2, 3), stride=(1, 2))

    batch_size, channels, height, width = 1, 4, 8, 9
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_avg_pool2d_various_sizes(conf: Conf):
    """Test avg_pool2d with various input sizes and parameters"""

    def fn_small(x):
        return F.avg_pool2d(x, kernel_size=2, stride=1)

    def fn_large(x):
        return F.avg_pool2d(x, kernel_size=4, stride=4)

    # Small input
    x_small = torch.randn(1, 8, 5, 5)
    check_outputs(fn_small, conf, [x_small])

    # Larger input
    x_large = torch.randn(2, 32, 16, 16)
    check_outputs(fn_large, conf, [x_large])


def test_flatten_basic(conf: Conf):
    """Test basic flatten operation"""

    def fn(x):
        return torch.flatten(x, start_dim=1)

    batch_size, channels, height, width = 2, 3, 4, 5
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_flatten_different_start_dims(conf: Conf):
    """Test flatten with different start dimensions"""

    def fn_start_0(x):
        return torch.flatten(x, start_dim=0)

    def fn_start_2(x):
        return torch.flatten(x, start_dim=2)

    x = torch.randn(2, 3, 4, 5)

    check_outputs(fn_start_0, conf, [x])
    check_outputs(fn_start_2, conf, [x])


def test_flatten_with_end_dim(conf: Conf):
    """Test flatten with specific end dimension"""

    def fn(x):
        return torch.flatten(x, start_dim=1, end_dim=2)

    x = torch.randn(2, 3, 4, 5)

    check_outputs(fn, conf, [x])


def test_flatten_negative_dims(conf: Conf):
    """Test flatten with negative dimensions"""

    def fn(x):
        return torch.flatten(x, start_dim=-2, end_dim=-1)

    x = torch.randn(2, 3, 4, 5)

    check_outputs(fn, conf, [x])


def test_dropout_inference(conf: Conf):
    """Test dropout in inference mode (should be no-op)"""

    def fn(x):
        return F.dropout(x, p=0.5, training=False)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_dropout_different_probabilities(conf: Conf):
    """Test dropout with different dropout probabilities in inference"""

    def fn_p01(x):
        return F.dropout(x, p=0.1, training=False)

    def fn_p05(x):
        return F.dropout(x, p=0.5, training=False)

    def fn_p09(x):
        return F.dropout(x, p=0.9, training=False)

    x = torch.randn(3, 4, 5)

    check_outputs(fn_p01, conf, [x])
    check_outputs(fn_p05, conf, [x])
    check_outputs(fn_p09, conf, [x])


def test_combined_vgg_like_ops(conf: Conf):
    """Test combining VGG-like operations together"""

    def fn(x, weight, bias):
        # Simulate a VGG-like block
        conv_out = F.conv2d(x, weight, bias, padding=1)
        relu_out = F.relu(conv_out)
        pool_out = F.max_pool2d(relu_out, kernel_size=2, stride=2)
        return pool_out

    batch_size, in_channels, height, width = 2, 3, 8, 8
    out_channels, kernel_size = 64, 3

    x = torch.randn(batch_size, in_channels, height, width)
    weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size)
    bias = torch.randn(out_channels)

    check_outputs(fn, conf, [x, weight, bias])


def test_max_pool2d_ceil_mode(conf: Conf):
    """Test max_pool2d with ceil_mode=True"""

    def fn(x):
        return F.max_pool2d(x, kernel_size=2, stride=2, ceil_mode=True)

    # Use odd spatial dimensions to test ceil_mode effect
    batch_size, channels, height, width = 2, 3, 7, 7
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_max_pool2d_with_conv2d_chain(conf: Conf):
    """Test max_pool2d chained with conv2d operations"""

    def fn(x, weight1, bias1, weight2, bias2):
        conv1 = F.conv2d(x, weight1, bias1)
        pool1 = F.max_pool2d(conv1, kernel_size=2)
        conv2 = F.conv2d(pool1, weight2, bias2)
        pool2 = F.max_pool2d(conv2, kernel_size=2)
        return pool2

    batch_size, in_channels = 2, 3
    hidden_channels, out_channels = 16, 32
    height, width = 16, 16

    x = torch.randn(batch_size, in_channels, height, width)
    weight1 = torch.randn(hidden_channels, in_channels, 3, 3)
    bias1 = torch.randn(hidden_channels)
    weight2 = torch.randn(out_channels, hidden_channels, 3, 3)
    bias2 = torch.randn(out_channels)

    # TensorFloat-32 tensor cores are used by default, lowering precision
    check_outputs(fn, conf, [x, weight1, bias1, weight2, bias2], atol=1e-3, rtol=1e-3)


def test_flatten_after_pooling(conf: Conf):
    """Test flatten operation after pooling (common CNN pattern)"""

    def fn(x):
        pooled = F.adaptive_avg_pool2d(x, (4, 4))
        flattened = torch.flatten(pooled, 1)
        return flattened

    batch_size, channels, height, width = 3, 64, 12, 12
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


@pytest.mark.xfail(reason="Dropout training mode not implemented yet")
def test_dropout_training_mode(conf: Conf):
    """Test dropout in training mode (should raise NotImplementedError)"""

    def fn(x):
        return F.dropout(x, p=0.5, training=True)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


@pytest.mark.xfail(reason="max_pool2d with return_indices not implemented yet")
def test_max_pool2d_return_indices(conf: Conf):
    """Test max_pool2d with return_indices=True (should raise NotImplementedError)"""

    def fn(x):
        return F.max_pool2d(x, kernel_size=2, return_indices=True)

    batch_size, channels, height, width = 2, 3, 4, 4
    x = torch.randn(batch_size, channels, height, width)

    check_outputs(fn, conf, [x])


def test_tril_basic(conf: Conf):
    """Test basic tril operation on 2D tensor"""

    def fn(x):
        return torch.tril(x)

    x = torch.randn(4, 4)
    check_outputs(fn, conf, [x])


def test_tril_with_positive_diagonal(conf: Conf):
    """Test tril with positive diagonal offset"""

    def fn(x):
        return torch.tril(x, diagonal=1)

    x = torch.randn(5, 5)
    check_outputs(fn, conf, [x])


def test_tril_with_negative_diagonal(conf: Conf):
    """Test tril with negative diagonal offset"""

    def fn(x):
        return torch.tril(x, diagonal=-1)

    x = torch.randn(4, 4)
    check_outputs(fn, conf, [x])


def test_tril_rectangular_matrix_tall(conf: Conf):
    def fn(x):
        return torch.tril(x)

    # Test both tall and wide matrices
    x_tall = torch.randn(6, 4)

    check_outputs(fn, conf, [x_tall])


def test_tril_rectangular_matrix_wide(conf: Conf):
    def fn(x):
        return torch.tril(x)

    # Test both tall and wide matrices
    x_wide = torch.randn(3, 7)
    check_outputs(fn, conf, [x_wide])


def test_tril_3_dimensions(conf: Conf):
    """Test tril on 3D tensor (should apply tril to each 2D slice)"""

    def fn(x):
        return torch.tril(x)

    x = torch.randn(2, 4, 6)  # 2 slices of 4x4 matrices
    check_outputs(fn, conf, [x])


def test_tril_4_dimensions(conf: Conf):
    """Test tril on 4D tensor (should apply tril to each 2D slice)"""

    def fn(x):
        return torch.tril(x)

    x = torch.randn(2, 3, 4, 5)  # 2 batches of 3 slices of 4x4 matrices
    check_outputs(fn, conf, [x])


def test_tril_int32(conf: Conf):
    """Test tril with float32 tensors"""

    def fn(x):
        return torch.tril(x)

    # Test with float32 (main supported type)
    x_float32 = torch.randint(0, 5, (3, 3), dtype=torch.int32)
    check_outputs(fn, conf, [x_float32])


def test_split_basic(conf: Conf):
    """Test basic tensor splitting"""

    def fn(x):
        return torch.split(x, 2, 0)

    x = torch.randn(6, 4)
    check_outputs(fn, conf, [x])


def test_split_uneven_second_dim(conf: Conf):
    """Test tensor splitting with uneven split sizes"""

    def fn(x):
        return torch.split(x, 3, 1)

    # 7 elements split by 3 should give splits of [3, 3, 1]
    x = torch.randn(2, 7)
    check_outputs(fn, conf, [x])


def test_split_uneven(conf: Conf):
    """Test tensor splitting with uneven split sizes"""

    def fn(x):
        return torch.split(x, 3, 0)

    # 7 elements split by 3 should give splits of [3, 3, 1]
    x = torch.randn(7, 4)
    check_outputs(fn, conf, [x])


def test_split_different_dims(conf: Conf):
    """Test tensor splitting along different dimensions"""

    def fn_dim0(x):
        return torch.split(x, 2, 0)

    def fn_dim1(x):
        return torch.split(x, 3, 1)

    x = torch.randn(4, 6)

    check_outputs(fn_dim0, conf, [x])
    check_outputs(fn_dim1, conf, [x])


def test_split_single_element(conf: Conf):
    """Test tensor splitting into single elements"""

    def fn(x):
        return torch.split(x, 1, 0)

    x = torch.randn(3, 2)
    check_outputs(fn, conf, [x])


def test_chunk_basic(conf: Conf):
    """Test basic tensor chunking"""

    def fn(x):
        return torch.chunk(x, 2, 0)

    x = torch.randn(6, 4)
    check_outputs(fn, conf, [x])


def test_chunk_uneven(conf: Conf):
    """Test tensor chunking with uneven split"""

    def fn(x):
        return torch.chunk(x, 3, 0)

    x = torch.randn(8, 4)  # 8 doesn't divide evenly by 3
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("chunks", [2, 3, 4])
def test_chunk_different_num_chunks(conf: Conf, chunks: int):
    """Test tensor chunking with different number of chunks"""

    def fn(x):
        return torch.chunk(x, chunks, 0)

    x = torch.randn(12, 8)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_chunk_different_dims(conf: Conf, dim: int):
    """Test tensor chunking along different dimensions"""

    def fn(x):
        return torch.chunk(x, 2, dim)

    x = torch.randn(4, 6)
    check_outputs(fn, conf, [x])


def test_chunk_3d_tensor(conf: Conf):
    """Test tensor chunking on 3D tensor"""

    def fn(x):
        return torch.chunk(x, 2, 1)

    x = torch.randn(2, 8, 3)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("shapes", [(8,), (3, 4), (2, 3, 4)])
def test_torch_clamp_both_bounds(conf: Conf, shapes):
    """Test torch.clamp with both min and max bounds."""

    def fn(x):
        return torch.clamp(x, min=-0.5, max=0.5)

    a = torch.randn(shapes)

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize("shapes", [(8,), (3, 4), (2, 3, 4)])
def test_torch_clamp_min_only(conf: Conf, shapes):
    """Test torch.clamp with only min bound."""

    def fn(x):
        return torch.clamp(x, min=-1.0)

    a = torch.randn(shapes)

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize("shapes", [(8,), (3, 4), (2, 3, 4)])
def test_torch_clamp_max_only(conf: Conf, shapes):
    """Test torch.clamp with only max bound."""

    def fn(x):
        return torch.clamp(x, max=1.0)

    a = torch.randn(shapes)

    check_outputs(fn, conf, [a])


def test_torch_clamp_tensor_bounds(device: str, tensor_shapes: tuple):
    """Test torch.clamp with tensor bounds (min and max as tensors)."""

    def fn(x, min_tensor, mojo_tensor):
        return torch.clamp(x, min=min_tensor, max=mojo_tensor)

    a = torch.randn(tensor_shapes)
    min_tensor = torch.full(tensor_shapes, -0.5)
    mojo_tensor = torch.full(tensor_shapes, 0.5)

    check_functions_are_equivalent(fn, device, [a, min_tensor, mojo_tensor])


def test_torch_clamp_edge_cases(conf: Conf):
    """Test torch.clamp edge cases with specific values."""

    def fn_identical_bounds(x):
        # When min equals max, all values should be set to that value
        return torch.clamp(x, min=0.5, max=0.5)

    def fn_inverted_bounds(x):
        # When min > max, PyTorch sets all values to max
        return torch.clamp(x, min=1.0, max=0.5)

    a = torch.tensor([-2.0, -0.5, 0.0, 0.5, 1.0, 2.0])

    check_outputs(fn_identical_bounds, conf, [a])
    check_outputs(fn_inverted_bounds, conf, [a])


def test_torch_arange_single_arg_int(device: str):
    def fn():
        return torch.arange(5, device=torch.device(device))

    check_functions_are_equivalent(fn, device, [])


def test_torch_arange_two_args(device: str):
    def fn():
        return torch.arange(1, 8, device=torch.device(device))

    check_functions_are_equivalent(fn, device, [])


def test_torch_arange_two_args_negative(device: str):
    def fn():
        return torch.arange(-3, 2, device=torch.device(device))

    check_functions_are_equivalent(fn, device, [])


def test_torch_arange_three_args(device: str):
    def fn():
        return torch.arange(0, 10, 2, device=torch.device(device))

    check_functions_are_equivalent(fn, device, [])


def test_torch_arange_negative_step(device: str):
    def fn():
        return torch.arange(10, 0, -1, device=torch.device(device))

    check_functions_are_equivalent(fn, device, [])


def test_torch_arange_negative_step_invert_bounds(device: str):
    def fn():
        return torch.arange(10, 0, -1, device=torch.device(device))

    check_functions_are_equivalent(fn, device, [])


def test_torch_arange_force_dtype_float(device: str):
    def fn():
        return torch.arange(5, dtype=torch.float32, device=torch.device(device))

    check_functions_are_equivalent(fn, device, [])


def test_torch_arange_force_dtype_int(device: str):
    def fn():
        return torch.arange(5, dtype=torch.int32, device=torch.device(device))

    check_functions_are_equivalent(fn, device, [])


def test_layer_norm_basic(device: str):
    def fn(x):
        return F.layer_norm(x, normalized_shape=(10,))

    input_tensor = torch.randn(5, 10)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_layer_norm_with_weight_bias(device: str):
    def fn(x, weight, bias):
        return F.layer_norm(x, normalized_shape=(10,), weight=weight, bias=bias)

    input_tensor = torch.randn(5, 10)
    weight = torch.randn(10)
    bias = torch.randn(10)
    check_functions_are_equivalent(fn, device, [input_tensor, weight, bias])


def test_layer_norm_multidim(device: str):
    def fn(x):
        return F.layer_norm(x, normalized_shape=(3, 4))

    input_tensor = torch.randn(2, 5, 3, 4)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_layer_norm_custom_eps(device: str):
    def fn(x):
        return F.layer_norm(x, normalized_shape=(10,), eps=1e-6)

    input_tensor = torch.randn(5, 10)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_gelu_basic(conf: Conf):
    def fn(x):
        return F.gelu(x)

    input_tensor = torch.randn(5, 10)
    check_outputs(fn, conf, [input_tensor])


def test_gelu_tanh_approx(conf: Conf):
    def fn(x):
        return F.gelu(x, approximate="tanh")

    input_tensor = torch.randn(5, 10)
    check_outputs(fn, conf, [input_tensor])


def test_gelu_negative_values(conf: Conf):
    def fn(x):
        return F.gelu(x)

    input_tensor = torch.randn(5, 10) - 2.0  # Mostly negative values
    check_outputs(fn, conf, [input_tensor])


def test_softmax_basic(device: str):
    def fn(x):
        return F.softmax(x, dim=-1)

    input_tensor = torch.randn(5, 10)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_softmax_dim_0(device: str):
    def fn(x):
        return F.softmax(x, dim=0)

    input_tensor = torch.randn(5, 10)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_softmax_dim_1(device: str):
    def fn(x):
        return F.softmax(x, dim=1)

    input_tensor = torch.randn(5, 10)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_softmax_multidim(device: str):
    def fn(x):
        return F.softmax(x, dim=2)

    input_tensor = torch.randn(3, 4, 5)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_softmax_negative_dim(device: str):
    def fn(x):
        return F.softmax(x, dim=-2)

    input_tensor = torch.randn(3, 4, 5)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_sum_basic(conf: Conf):
    def fn(x):
        return torch.sum(x)

    input_tensor = torch.randn(5, 10)
    check_outputs(fn, conf, [input_tensor])


def test_sum_with_dim(conf: Conf):
    def fn(x):
        return torch.sum(x, dim=1)

    input_tensor = torch.randn(5, 10)
    check_outputs(fn, conf, [input_tensor])


def test_sum_with_keepdim(conf: Conf):
    def fn(x):
        return torch.sum(x, dim=1, keepdim=True)

    input_tensor = torch.randn(5, 10)
    check_outputs(fn, conf, [input_tensor])


def test_sum_multiple_dims(conf: Conf):
    def fn(x):
        return torch.sum(x, dim=[1, 2])

    input_tensor = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [input_tensor])


def test_sum_multiple_dims_keepdim(conf: Conf):
    def fn(x):
        return torch.sum(x, dim=[1, 2], keepdim=True)

    input_tensor = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [input_tensor])


def test_sum_negative_dim(conf: Conf):
    def fn(x):
        return torch.sum(x, dim=-1)

    input_tensor = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [input_tensor])


def test_cumsum_basic(conf: Conf):
    """Test basic cumsum operation along dimension 0"""

    def fn(x):
        return torch.cumsum(x, dim=0)

    input_tensor = torch.randn(5, 3)
    check_outputs(fn, conf, [input_tensor])


def test_cumsum_different_dims(conf: Conf):
    """Test cumsum along different dimensions"""

    def fn_dim0(x):
        return torch.cumsum(x, dim=0)

    def fn_dim1(x):
        return torch.cumsum(x, dim=1)

    def fn_dim2(x):
        return torch.cumsum(x, dim=2)

    input_tensor = torch.randn(3, 4, 5)

    check_outputs(fn_dim0, conf, [input_tensor])
    check_outputs(fn_dim1, conf, [input_tensor])
    check_outputs(fn_dim2, conf, [input_tensor])


def test_cumsum_negative_dim(conf: Conf):
    """Test cumsum with negative dimension index"""

    def fn(x):
        return torch.cumsum(x, dim=-1)

    input_tensor = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [input_tensor])


def test_cumsum_with_dtype(conf: Conf):
    """Test cumsum with dtype conversion"""

    def fn(x):
        return torch.cumsum(x, dim=1, dtype=torch.float32)

    # Start with integer tensor to test dtype conversion
    input_tensor = torch.randint(
        0, 10, (4, 6)
    ).float()  # Convert to float for consistency
    check_outputs(fn, conf, [input_tensor])


def test_cumsum_1d_tensor(conf: Conf):
    """Test cumsum on 1D tensor"""

    def fn(x):
        return torch.cumsum(x, dim=0)

    input_tensor = torch.randn(10)
    check_outputs(fn, conf, [input_tensor])


def test_cumsum_large_tensor(conf: Conf):
    """Test cumsum on larger tensor"""

    def fn(x):
        return torch.cumsum(x, dim=1)

    input_tensor = torch.randn(8, 16, 4)
    check_outputs(fn, conf, [input_tensor])


def test_masked_fill_basic(conf: Conf):
    def fn(x, mask):
        return x.masked_fill(mask, -float("inf"))

    input_tensor = torch.randn(5, 10)
    mask = torch.randint(0, 2, (5, 10), dtype=torch.bool)
    check_outputs(fn, conf, [input_tensor, mask])


def test_masked_fill_scalar_value(conf: Conf):
    def fn(x, mask):
        return x.masked_fill(mask, 0.0)

    input_tensor = torch.randn(5, 10)
    mask = torch.randint(0, 2, (5, 10), dtype=torch.bool)
    check_outputs(fn, conf, [input_tensor, mask])


def test_masked_fill_broadcast(conf: Conf):
    def fn(x, mask):
        return x.masked_fill(mask, 99.0)

    input_tensor = torch.randn(3, 4, 5)
    mask = torch.randint(0, 2, (4, 5), dtype=torch.bool)
    check_outputs(fn, conf, [input_tensor, mask])


def test_reshape_basic(conf: Conf):
    def fn(x):
        return x.reshape(6, 4)

    input_tensor = torch.randn(3, 8)
    check_outputs(fn, conf, [input_tensor])


def test_reshape_flatten(conf: Conf):
    def fn(x):
        return x.reshape(-1)

    input_tensor = torch.randn(2, 3, 4)
    check_outputs(fn, conf, [input_tensor])


def test_reshape_with_negative_dim(conf: Conf):
    def fn(x):
        return x.reshape(2, -1, 3)

    input_tensor = torch.randn(2, 4, 3)
    check_outputs(fn, conf, [input_tensor])


def test_unbind_basic(conf: Conf):
    def fn(x):
        # unbind returns a tuple, so we convert to list for testing
        return list(x.unbind(dim=0))

    input_tensor = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [input_tensor])


def test_unbind_different_dim(conf: Conf):
    def fn(x):
        return list(x.unbind(dim=1))

    input_tensor = torch.randn(2, 3, 4)
    check_outputs(fn, conf, [input_tensor])


def test_unbind_negative_dim(conf: Conf):
    def fn(x):
        return list(x.unbind(dim=-1))

    input_tensor = torch.randn(2, 3, 4)
    check_outputs(fn, conf, [input_tensor])


def test_repeat_basic(conf: Conf):
    def fn(x):
        return x.repeat(2, 3)

    input_tensor = torch.randn(2, 3)
    check_outputs(fn, conf, [input_tensor])


@pytest.mark.parametrize("repeats", [(2,), (2, 3), (1, 2, 3), (2, 1, 3)])
def test_repeat_different_repeats(conf: Conf, repeats):
    def fn(x):
        return x.repeat(*repeats)

    # Create tensor with appropriate shape for the repeat operation
    if len(repeats) == 1:
        input_tensor = torch.randn(3)
    elif len(repeats) == 2:
        input_tensor = torch.randn(2, 3)
    elif len(repeats) == 3:
        input_tensor = torch.randn(2, 3, 4)
    else:
        input_tensor = torch.randn(2, 3)

    check_outputs(fn, conf, [input_tensor])


def test_repeat_1d(conf: Conf):
    def fn(x):
        return x.repeat(3)

    input_tensor = torch.randn(4)
    check_outputs(fn, conf, [input_tensor])


def test_repeat_2d(conf: Conf):
    def fn(x):
        return x.repeat(2, 3)

    input_tensor = torch.randn(2, 4)
    check_outputs(fn, conf, [input_tensor])


def test_repeat_3d(conf: Conf):
    def fn(x):
        return x.repeat(2, 1, 3)

    input_tensor = torch.randn(1, 3, 2)
    check_outputs(fn, conf, [input_tensor])


def test_repeat_interleave_basic(device: str):
    def fn(x):
        return x.repeat_interleave(2, dim=0)

    input_tensor = torch.randn(3, 4)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_repeat_interleave_different_dim(device: str):
    def fn(x):
        return x.repeat_interleave(3, dim=1)

    input_tensor = torch.randn(2, 4, 5)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_repeat_interleave_negative_dim(device: str):
    def fn(x):
        return x.repeat_interleave(2, dim=-1)

    input_tensor = torch.randn(3, 4, 5)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_torch_full_basic(device: str):
    def fn():
        return torch.full((3, 4), -float("inf"))

    check_functions_are_equivalent(fn, device, [])


def test_torch_full_with_dtype(device: str):
    def fn():
        return torch.full((2, 3), 5.5, dtype=torch.float32)

    check_functions_are_equivalent(fn, device, [])


def test_torch_triu_basic(conf: Conf):
    def fn(x):
        return torch.triu(x)

    input_tensor = torch.randn(4, 4)
    check_outputs(fn, conf, [input_tensor])


def test_torch_triu_with_diagonal(conf: Conf):
    def fn(x):
        return torch.triu(x, diagonal=1)

    input_tensor = torch.randn(3, 3)
    check_outputs(fn, conf, [input_tensor])


@pytest.mark.parametrize("dynamic_dim", [0, 1])
def test_torch_triu_with_diagonal_dynamic_dim(conf: Conf, dynamic_dim: int):
    def fn(x):
        return torch.triu(x, diagonal=1)

    input_tensor = torch.randn(3, 5)
    mark_dynamic(input_tensor, dynamic_dim)
    check_outputs(fn, conf, [input_tensor])


def test_silu_activation(device: str):
    def fn(x):
        return F.silu(x)

    input_tensor = torch.randn(3, 4, 5)
    check_functions_are_equivalent(fn, device, [input_tensor])


def test_mse_loss_default_reduction(device: str):
    """Test MSE loss with default mean reduction"""

    def fn(input_tensor, target):
        return F.mse_loss(input_tensor, target)

    input_tensor = torch.randn(3, 4)
    target = torch.randn(3, 4)
    check_functions_are_equivalent(fn, device, [input_tensor, target])


def test_mse_loss_mean_reduction(device: str):
    """Test MSE loss with explicit mean reduction"""

    def fn(input_tensor, target):
        return F.mse_loss(input_tensor, target, reduction="mean")

    input_tensor = torch.randn(2, 5)
    target = torch.randn(2, 5)
    check_functions_are_equivalent(fn, device, [input_tensor, target])


def test_mse_loss_sum_reduction(device: str):
    """Test MSE loss with sum reduction"""

    def fn(input_tensor, target):
        return F.mse_loss(input_tensor, target, reduction="sum")

    input_tensor = torch.randn(3, 3)
    target = torch.randn(3, 3)
    check_functions_are_equivalent(fn, device, [input_tensor, target])


def test_mse_loss_none_reduction(device: str):
    """Test MSE loss with no reduction (returns element-wise squared differences)"""

    def fn(input_tensor, target):
        return F.mse_loss(input_tensor, target, reduction="none")

    input_tensor = torch.randn(2, 3)
    target = torch.randn(2, 3)
    check_functions_are_equivalent(fn, device, [input_tensor, target])


def test_mse_loss_1d(device: str):
    """Test MSE loss on 1D tensors"""

    def fn(input_tensor, target):
        return F.mse_loss(input_tensor, target)

    input_tensor = torch.randn(10)
    target = torch.randn(10)
    check_functions_are_equivalent(fn, device, [input_tensor, target])


def test_mse_loss_3d(device: str):
    """Test MSE loss on 3D tensors"""

    def fn(input_tensor, target):
        return F.mse_loss(input_tensor, target)

    input_tensor = torch.randn(2, 3, 4)
    target = torch.randn(2, 3, 4)
    check_functions_are_equivalent(fn, device, [input_tensor, target])


def test_addmm_basic(device: str):
    """Test basic torch.addmm operation"""

    def fn(bias, mat1, mat2):
        return torch.addmm(bias, mat1, mat2)

    bias = torch.randn(3, 4)
    mat1 = torch.randn(3, 5)
    mat2 = torch.randn(5, 4)
    # `beta * bias` is exact; only `mat1 @ mat2` (inner size 5) is truncated.
    rtol, atol = gemm_tolerances(device, 5)
    check_functions_are_equivalent(fn, device, [bias, mat1, mat2], rtol=rtol, atol=atol)


def test_addmm_with_alpha_beta(device: str):
    """Test torch.addmm with custom alpha and beta parameters"""

    def fn(bias, mat1, mat2):
        return torch.addmm(bias, mat1, mat2, alpha=2.0, beta=0.5)

    bias = torch.randn(3, 4)
    mat1 = torch.randn(3, 5)
    mat2 = torch.randn(5, 4)
    # `alpha` multiplies the truncated product, so it scales the budget too.
    # `beta = 0.5` is a power of two, so `beta * bias` stays exact.
    rtol, atol = gemm_tolerances(device, 5, scale=2.0)
    check_functions_are_equivalent(fn, device, [bias, mat1, mat2], rtol=rtol, atol=atol)


def test_addmm_different_shapes(device: str):
    """Test torch.addmm with different matrix shapes"""

    def fn(bias, mat1, mat2):
        return torch.addmm(bias, mat1, mat2)

    bias = torch.randn(2, 8)
    mat1 = torch.randn(2, 6)
    mat2 = torch.randn(6, 8)
    rtol, atol = gemm_tolerances(device, 6)
    check_functions_are_equivalent(fn, device, [bias, mat1, mat2], rtol=rtol, atol=atol)


def test_addmm_broadcast_bias(conf: Conf):
    """Test torch.addmm with bias that needs broadcasting"""

    def fn(bias, mat1, mat2):
        return torch.addmm(bias, mat1, mat2)

    bias = torch.randn(4)  # Will broadcast to (3, 4)
    mat1 = torch.randn(3, 5)
    mat2 = torch.randn(5, 4)
    check_outputs(fn, conf, [bias, mat1, mat2])


def test_bmm_basic(conf: Conf):
    """Test basic torch.bmm operation (batch matrix multiplication)"""

    def fn(input, mat2):
        return torch.bmm(input, mat2)

    # Create batch matrix multiplication inputs: [batch, n, m] x [batch, m, p] = [batch, n, p]
    batch_size, n, m, p = 4, 3, 5, 6
    input = torch.randn(batch_size, n, m)
    mat2 = torch.randn(batch_size, m, p)

    check_outputs(fn, conf, [input, mat2])


def test_bmm_different_batch_sizes(conf: Conf):
    """Test torch.bmm with different batch sizes and matrix dimensions"""

    def fn(input, mat2):
        return torch.bmm(input, mat2)

    # Different batch size and matrix dimensions
    batch_size, n, m, p = 2, 4, 3, 7
    input = torch.randn(batch_size, n, m)
    mat2 = torch.randn(batch_size, m, p)

    check_outputs(fn, conf, [input, mat2])


def test_bmm_single_batch(conf: Conf):
    """Test torch.bmm with single batch dimension"""

    def fn(input, mat2):
        return torch.bmm(input, mat2)

    # Single batch case
    input = torch.randn(1, 2, 4)
    mat2 = torch.randn(1, 4, 3)

    check_outputs(fn, conf, [input, mat2])


def test_exp(device: str):
    def fn(input):
        return torch.exp(input)

    # Single batch case
    input = torch.randn(1, 2, 4)

    check_functions_are_equivalent(fn, device, [input])


def test_group_norm(device: str):
    """Test F.group_norm with various configurations"""

    def fn(input):
        return F.group_norm(input, 2)

    # Test case: 4 channels, 2 groups
    batch_size, channels, height, width = 2, 4, 8, 8
    input = torch.randn(batch_size, channels, height, width)

    check_functions_are_equivalent(fn, device, [input])


def test_group_norm_with_weight_bias(device: str):
    """Test F.group_norm with weight and bias parameters"""

    def fn(input, weight, bias):
        return F.group_norm(input, 3, weight, bias)

    # Test case: 6 channels, 3 groups
    batch_size, channels, height, width = 1, 6, 4, 4
    input = torch.randn(batch_size, channels, height, width)
    weight = torch.randn(channels)
    bias = torch.randn(channels)

    check_functions_are_equivalent(fn, device, [input, weight, bias])


def test_group_norm_eps(device: str):
    """Test F.group_norm with custom eps parameter"""

    def fn(input):
        return F.group_norm(input, 4, eps=1e-6)

    # Test case: 8 channels, 4 groups
    batch_size, channels, height, width = 1, 8, 2, 2
    input = torch.randn(batch_size, channels, height, width)

    check_functions_are_equivalent(fn, device, [input])


def test_logical_not_bool(conf: Conf):
    """Test torch.logical_not with boolean tensors"""

    def fn(x):
        return torch.logical_not(x)

    # Test with boolean tensor
    a = torch.tensor([True, False, True, False])

    check_outputs(fn, conf, [a])


def test_logical_not_numeric(conf: Conf):
    """Test torch.logical_not with numeric tensors"""

    def fn(x):
        return torch.logical_not(x)

    # Test with numeric tensor (non-zero values are treated as True)
    a = torch.tensor([0, 1, 2, -1, 0.0])

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize("shapes", [(3, 4), (2, 3, 4), (5,)])
def test_logical_not_shapes(conf: Conf, shapes):
    """Test torch.logical_not with different tensor shapes"""

    def fn(x):
        return torch.logical_not(x)

    # Test with various shapes and numeric values
    a = torch.randint(0, 2, shapes, dtype=torch.int32)

    check_outputs(fn, conf, [a])


def test_logical_xor_bool(conf: Conf):
    """Test torch.logical_xor with boolean tensors"""

    def fn(x, y):
        return torch.logical_xor(x, y)

    # Test with boolean tensors - various XOR combinations
    a = torch.tensor([True, False, True, False])
    b = torch.tensor([True, True, False, False])

    check_outputs(fn, conf, [a, b])


def test_logical_xor_numeric(conf: Conf):
    """Test torch.logical_xor with numeric tensors"""

    def fn(x, y):
        return torch.logical_xor(x, y)

    # Test with numeric tensors (non-zero treated as True)
    a = torch.tensor([0, 1, 2, -1, 0.0])
    b = torch.tensor([0, 0, 1, -1, 3.0])

    check_outputs(fn, conf, [a, b])


def test_logical_xor_mixed_types(conf: Conf):
    """Test torch.logical_xor with mixed boolean and numeric tensors"""

    def fn(x, y):
        return torch.logical_xor(x, y)

    # Test boolean tensor with numeric tensor
    a = torch.tensor([True, False, True, False])
    b = torch.tensor([0, 1, 0, 2])

    check_outputs(fn, conf, [a, b])


@pytest.mark.parametrize("shapes", [(3, 4), (2, 3, 4), (5,)])
def test_logical_xor_shapes(conf: Conf, shapes):
    """Test torch.logical_xor with different tensor shapes"""

    def fn(x, y):
        return torch.logical_xor(x, y)

    # Test with various shapes
    a = torch.randint(0, 2, shapes, dtype=torch.int32)
    b = torch.randint(0, 2, shapes, dtype=torch.int32)

    check_outputs(fn, conf, [a, b])


def test_logical_xor_edge_cases(conf: Conf):
    """Test torch.logical_xor with edge cases"""

    def fn(x, y):
        return torch.logical_xor(x, y)

    # Test with specific edge case values
    test_cases = [
        # All False XOR All False = All False
        (torch.tensor([False, False, False]), torch.tensor([False, False, False])),
        # All True XOR All False = All True
        (torch.tensor([True, True, True]), torch.tensor([False, False, False])),
        # All True XOR All True = All False
        (torch.tensor([True, True, True]), torch.tensor([True, True, True])),
        # Mixed numeric values
        (torch.tensor([0.0, -1.0, 2.5]), torch.tensor([0.0, 1.0, 0.0])),
    ]

    for a, b in test_cases:
        check_outputs(fn, conf, [a, b])


def test_any_basic(conf: Conf):
    """Test torch.any basic functionality"""

    def fn(x):
        return torch.any(x)

    # Test with tensor containing non-zero values (should return True)
    a = torch.tensor([0, 0, 1, 0])
    check_outputs(fn, conf, [a])

    # Test with tensor containing all zeros (should return False)
    b = torch.tensor([0, 0, 0, 0])
    check_outputs(fn, conf, [b])


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_any_with_dim(device: str, dim: int):
    """Test torch.any with dimension parameter"""

    def fn(x):
        return torch.any(x, dim=dim)

    # Test with 2D tensor
    a = torch.tensor([[1, 0], [0, 0], [0, 1]])

    check_functions_are_equivalent(fn, device, [a])


def test_any_keepdim(device: str):
    """Test torch.any with keepdim parameter"""

    def fn(x):
        return torch.any(x, dim=1, keepdim=True)

    # Test with 2D tensor and keepdim=True
    a = torch.tensor([[1, 0], [0, 0], [0, 1]])

    check_functions_are_equivalent(fn, device, [a])


@pytest.mark.parametrize("shapes", [(3, 4), (2, 3, 4), (5,)])
def test_any_shapes(conf: Conf, shapes):
    """Test torch.any with different tensor shapes"""

    def fn(x):
        return torch.any(x)

    # Test with various shapes and random binary values
    a = torch.randint(0, 2, shapes, dtype=torch.int32)

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize(
    "dtype",
    [torch.int8, torch.int32, torch.int64, torch.float32, torch.float64, torch.bool],
)
def test_any_dtypes(conf: Conf, dtype):
    """Test torch.any with different data types"""

    def fn(x):
        return torch.any(x)

    if dtype == torch.bool:
        # For boolean dtype, create tensor with True/False values
        a = torch.tensor([True, False, False, True], dtype=dtype)
    elif dtype.is_floating_point:
        # For float dtypes, use values including 0.0 and non-zero
        a = torch.tensor([0.0, 1.5, 0.0, -2.3], dtype=dtype)
    else:
        # For integer dtypes, use mix of zero and non-zero values
        a = torch.tensor([0, 1, 0, -2], dtype=dtype)

    check_outputs(fn, conf, [a])


@pytest.mark.parametrize("dtype", [torch.int32, torch.float32, torch.bool])
@pytest.mark.parametrize("dim", [0, 1])
def test_any_dtypes_with_dim(device: str, dtype, dim):
    """Test torch.any with different data types and dimensions"""

    def fn(x):
        return torch.any(x, dim=dim)

    if dtype == torch.bool:
        # For boolean dtype
        a = torch.tensor([[True, False], [False, False], [True, True]], dtype=dtype)
    elif dtype == torch.float32:
        # For float dtype
        a = torch.tensor([[1.0, 0.0], [0.0, 0.0], [2.5, -1.5]], dtype=dtype)
    else:
        # For integer dtype
        a = torch.tensor([[1, 0], [0, 0], [2, -1]], dtype=dtype)

    check_functions_are_equivalent(fn, device, [a])


@pytest.mark.parametrize("tensor_shapes", [(2, 3), (1, 5), (4,), (2, 3, 4)])
def test_full_like_basic(conf: Conf, tensor_shapes: tuple):
    """Test torch.full_like with different shapes and fill values"""

    def fn(x):
        return torch.full_like(x, 5.0)

    x = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("fill_value", [0, 1, -1, 3.14, -2.5])
def test_full_like_different_values(conf: Conf, fill_value):
    """Test torch.full_like with different fill values"""

    def fn(x):
        return torch.full_like(x, fill_value)

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.int32, torch.int64]
)
def test_full_like_dtype(conf: Conf, dtype):
    """Test torch.full_like with different dtype specifications"""

    def fn(x):
        return torch.full_like(x, 7, dtype=dtype)

    x = torch.randn(2, 3)

    check_outputs(fn, conf, [x])


def test_full_like_scalar_tensor(conf: Conf):
    """Test torch.full_like with scalar tensor"""

    def fn(x):
        return torch.full_like(x, 42.0)

    x = torch.tensor(1.0)

    check_outputs(fn, conf, [x])


def test_full_like_zero_fill(conf: Conf):
    """Test torch.full_like with zero fill value"""

    def fn(x):
        return torch.full_like(x, 0)

    x = torch.randn(2, 5, 3)

    check_outputs(fn, conf, [x])


def test_full_like_negative_fill(conf: Conf):
    """Test torch.full_like with negative fill value"""

    def fn(x):
        return torch.full_like(x, -10)

    x = torch.randn(4, 2)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("tensor_shapes", [(1, 1), (10,), (2, 2, 2, 2)])
def test_full_like_edge_cases(conf: Conf, tensor_shapes: tuple):
    """Test torch.full_like with edge case shapes"""

    def fn(x):
        return torch.full_like(x, 100)

    x = torch.ones(tensor_shapes)

    check_outputs(fn, conf, [x])


def test_scaled_dot_product_attention_basic(device: str):
    """Test basic scaled dot-product attention"""

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    batch_size, seq_len, embed_dim = 2, 8, 64
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    check_functions_are_equivalent(fn, device, [query, key, value])


def test_scaled_dot_product_attention_with_mask(device: str):
    """Test scaled dot-product attention with attention mask"""

    def fn(query, key, value, attn_mask):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask
        )

    batch_size, seq_len, embed_dim = 2, 4, 32
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    # Create a simple causal mask (lower triangular)
    attn_mask = torch.tril(torch.ones(seq_len, seq_len))
    attn_mask = attn_mask.masked_fill(attn_mask == 0, float("-inf"))
    attn_mask = attn_mask.masked_fill(attn_mask == 1, 0.0)

    check_functions_are_equivalent(fn, device, [query, key, value, attn_mask])


def test_scaled_dot_product_attention_different_kv_length(device: str):
    """Test scaled dot-product attention with different key/value sequence length"""

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    batch_size, q_len, kv_len, embed_dim = 2, 6, 10, 64
    query = torch.randn(batch_size, q_len, embed_dim)
    key = torch.randn(batch_size, kv_len, embed_dim)
    value = torch.randn(batch_size, kv_len, embed_dim)

    check_functions_are_equivalent(fn, device, [query, key, value])


def test_scaled_dot_product_attention_multihead(device: str):
    """Test scaled dot-product attention with multiple heads"""

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    batch_size, num_heads, seq_len, head_dim = 2, 8, 12, 64
    query = torch.randn(batch_size, num_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    check_functions_are_equivalent(fn, device, [query, key, value])


def test_scaled_dot_product_attention_with_dropout_mask(device: str):
    """Test scaled dot-product attention with dropout mask"""

    def fn(query, key, value):
        # Note: for testing we use dropout_p=0 to ensure deterministic results
        dropout_p = 0.0  # No dropout for deterministic testing
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p
        )

    batch_size, seq_len, embed_dim = 2, 6, 32
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    check_functions_are_equivalent(fn, device, [query, key, value])


def test_scaled_dot_product_attention_small_dimensions(device: str):
    """Test scaled dot-product attention with small dimensions"""

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    batch_size, seq_len, embed_dim = 1, 3, 16
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    # A 3D sdpa decomposes to two `bmm`s. At batch size 1 MAX takes the TF32
    # tensor-core GEMM for them (see `tf32_gemm_atol`); the batched shapes in
    # the neighbouring tests land on a MAX kernel that stays in fp32, which is
    # why only this one needs a budget. The budget is the `q @ k^T` error --
    # inner size `embed_dim`, scaled by `1/sqrt(embed_dim)` -- doubled:
    # perturbing the scores by `d` moves `softmax` by at most `2*d` in l1, and
    # the value `bmm` then averages rows of `value` with weights summing to
    # one rather than amplifying. Worst error over 300 draws: 1.5e-3 against
    # the 1.25e-2 budget.
    rtol, atol = gemm_tolerances(device, embed_dim, scale=2.0 / math.sqrt(embed_dim))
    check_functions_are_equivalent(
        fn, device, [query, key, value], rtol=rtol, atol=atol
    )


def test_scaled_dot_product_attention_single_token(device: str):
    """Test scaled dot-product attention with single token sequence"""

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    batch_size, seq_len, embed_dim = 2, 1, 32
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    check_functions_are_equivalent(fn, device, [query, key, value])


@pytest.mark.parametrize("embed_dim", [32, 64, 128])
def test_scaled_dot_product_attention_different_embed_dims(device: str, embed_dim: int):
    """Test scaled dot-product attention with different embedding dimensions"""

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    batch_size, seq_len = 2, 8
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    check_functions_are_equivalent(fn, device, [query, key, value])


def test_scaled_dot_product_attention_is_causal(device: str):
    """Test scaled dot-product attention with causal masking"""

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=True
        )

    batch_size, seq_len, embed_dim = 2, 6, 32
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    check_functions_are_equivalent(fn, device, [query, key, value])


def test_scaled_dot_product_attention_cross_attention(device: str):
    """Test scaled dot-product attention for cross-attention (encoder-decoder)"""

    def fn(query, key, value):
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    batch_size = 2
    tgt_len, src_len = 8, 12  # Different lengths for cross-attention
    embed_dim = 64

    query = torch.randn(batch_size, tgt_len, embed_dim)  # From decoder
    key = torch.randn(batch_size, src_len, embed_dim)  # From encoder
    value = torch.randn(batch_size, src_len, embed_dim)  # From encoder

    check_functions_are_equivalent(fn, device, [query, key, value])


def test_scaled_dot_product_attention_with_scale(device: str):
    """Test scaled dot-product attention with custom scale"""

    def fn(query, key, value):
        custom_scale = 0.125  # Custom scale instead of 1/sqrt(embed_dim)
        return torch.nn.functional.scaled_dot_product_attention(
            query, key, value, scale=custom_scale
        )

    batch_size, seq_len, embed_dim = 2, 4, 32
    query = torch.randn(batch_size, seq_len, embed_dim)
    key = torch.randn(batch_size, seq_len, embed_dim)
    value = torch.randn(batch_size, seq_len, embed_dim)

    check_functions_are_equivalent(fn, device, [query, key, value])


@pytest.mark.parametrize("dims", [(0, 1), (1, 0)])
def test_permute_2d(conf: Conf, dims: tuple):
    """Test torch.permute with 2D tensors"""

    def fn(x):
        return x.permute(dims)

    x = torch.randn(3, 4)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize(
    "dims", [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
)
def test_permute_3d(conf: Conf, dims: tuple):
    """Test torch.permute with 3D tensors - all permutations"""

    def fn(x):
        return x.permute(dims)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_permute_4d_nchw_to_nhwc(conf: Conf):
    """Test common permutation: NCHW to NHWC format conversion"""

    def fn(x):
        # Convert from [N, C, H, W] to [N, H, W, C]
        return x.permute(0, 2, 3, 1)

    x = torch.randn(2, 3, 8, 8)  # Batch=2, Channels=3, Height=8, Width=8

    check_outputs(fn, conf, [x])


def test_permute_4d_nhwc_to_nchw(conf: Conf):
    """Test common permutation: NHWC to NCHW format conversion"""

    def fn(x):
        # Convert from [N, H, W, C] to [N, C, H, W]
        return x.permute(0, 3, 1, 2)

    x = torch.randn(2, 8, 8, 3)  # Batch=2, Height=8, Width=8, Channels=3

    check_outputs(fn, conf, [x])


def test_permute_with_negative_indices(conf: Conf):
    """Test torch.permute with negative dimension indices"""

    def fn(x):
        # Equivalent to (0, 2, 1) for 3D tensor
        return x.permute(0, -1, -2)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_permute_identity(conf: Conf):
    """Test permute with identity permutation (no change)"""

    def fn(x):
        return x.permute(0, 1, 2)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_permute_reverse_order(conf: Conf):
    """Test permute that reverses all dimensions"""

    def fn(x):
        return x.permute(2, 1, 0)

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


@pytest.mark.parametrize("tensor_shapes", [(5,), (1, 10), (2, 3, 4), (1, 2, 3, 4, 5)])
def test_permute_different_shapes(conf: Conf, tensor_shapes: tuple):
    """Test permute with various tensor shapes"""

    def fn(x):
        # Create a permutation that reverses the dimensions
        dims = tuple(range(len(x.shape) - 1, -1, -1))
        return x.permute(dims)

    x = torch.randn(tensor_shapes)

    check_outputs(fn, conf, [x])


def test_permute_with_arithmetic(conf: Conf):
    """Test permute combined with arithmetic operations"""

    def fn(x, y):
        x_permuted = x.permute(1, 0, 2)
        y_permuted = y.permute(1, 0, 2)
        return x_permuted + y_permuted

    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x, y])


def test_permute_chain(conf: Conf):
    """Test chaining multiple permute operations"""

    def fn(x):
        # Apply permute twice
        step1 = x.permute(2, 1, 0)  # Reverse all dims
        step2 = step1.permute(2, 1, 0)  # Reverse again (should be back to original)
        return step2

    x = torch.randn(2, 3, 4)

    check_outputs(fn, conf, [x])


def test_permute_transpose_equivalent(conf: Conf):
    """Test that permute(1, 0) is equivalent to transpose for 2D tensors"""

    def fn(x):
        return x.permute(1, 0)

    x = torch.randn(3, 4)

    # This should be equivalent to x.t() or x.transpose(0, 1)
    check_outputs(fn, conf, [x])


def test_permute_1d_identity(conf: Conf):
    """Test permute with 1D tensor (identity operation)"""

    def fn(x):
        # 1D tensors can only be permuted with (0,)
        return x.permute(0)

    x = torch.randn(5)

    check_outputs(fn, conf, [x])


def test_movedim_basic(conf: Conf):
    """Test basic movedim operation - move single dimension"""

    def fn(x):
        return torch.movedim(x, 0, 2)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


@pytest.mark.parametrize(
    "source,destination", [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]
)
def test_movedim_different_positions(conf: Conf, source: int, destination: int):
    """Test movedim with different source and destination positions"""

    def fn(x):
        return torch.movedim(x, source, destination)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


def test_movedim_negative_indices(conf: Conf):
    """Test movedim with negative indices"""

    def fn(x):
        return torch.movedim(x, -1, 0)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


def test_movedim_multiple_dims(conf: Conf):
    """Test movedim with multiple dimensions"""

    def fn(x):
        return torch.movedim(x, [0, 1], [2, 0])

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


def test_movedim_4d_tensor(conf: Conf):
    """Test movedim on 4D tensor (common in computer vision)"""

    def fn(x):
        # Move channels from dim 1 to dim 3 (NCHW to NHWC)
        return torch.movedim(x, 1, 3)

    x = torch.randn(2, 3, 4, 5)  # N, C, H, W
    check_outputs(fn, conf, [x])


def test_movedim_identity(conf: Conf):
    """Test movedim identity operation"""

    def fn(x):
        return torch.movedim(x, 1, 1)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])


def test_movedim_2d_tensor(conf: Conf):
    """Test movedim on 2D tensor"""

    def fn(x):
        return torch.movedim(x, 0, 1)

    x = torch.randn(3, 4)
    check_outputs(fn, conf, [x])


def test_optimizer(conf: Conf):
    # Create two identical models
    torch.manual_seed(42)
    model_compiled = torch.nn.Sequential(
        torch.nn.Linear(8, 8, False), torch.nn.Linear(8, 8, False)
    )

    torch.manual_seed(42)
    model_uncompiled = torch.nn.Sequential(
        torch.nn.Linear(8, 8, False), torch.nn.Linear(8, 8, False)
    )

    # Same input for both
    torch.manual_seed(123)
    input_data = torch.rand(8)

    # Forward and backward for compiled version
    output_compiled = model_compiled(input_data)
    output_compiled.sum().backward()

    # Forward and backward for uncompiled version
    output_uncompiled = model_uncompiled(input_data)
    output_uncompiled.sum().backward()

    # Create optimizers
    opt_compiled = torch.optim.Adam(model_compiled.parameters(), lr=0.01)
    opt_uncompiled = torch.optim.Adam(model_uncompiled.parameters(), lr=0.01)

    # Run optimizer step with compilation
    @torch.compile(backend=mojo_backend)
    def fn_compiled():
        opt_compiled.step()

    fn_compiled()

    # Run optimizer step without compilation
    opt_uncompiled.step()

    # Compare results: parameters should be the same after first optimizer step
    for p_compiled, p_uncompiled in zip(
        model_compiled.parameters(), model_uncompiled.parameters()
    ):
        torch.testing.assert_close(p_compiled, p_uncompiled)

    # Run a second forward/backward pass and optimizer step
    torch.manual_seed(456)
    input_data_2 = torch.rand(8)

    # Second forward and backward for compiled version
    output_compiled_2 = model_compiled(input_data_2)
    output_compiled_2.sum().backward()

    # Second forward and backward for uncompiled version
    output_uncompiled_2 = model_uncompiled(input_data_2)
    output_uncompiled_2.sum().backward()

    # Second optimizer step with compilation
    fn_compiled()

    # Second optimizer step without compilation
    opt_uncompiled.step()

    # Compare results: parameters should be the same after second optimizer step
    for p_compiled, p_uncompiled in zip(
        model_compiled.parameters(), model_uncompiled.parameters()
    ):
        torch.testing.assert_close(p_compiled, p_uncompiled)


# TODO: support list as input too
def test_aten_index_select_basic(conf: Conf):
    """Test basic torch.index_select operation"""

    def fn(x, indices):
        return x[indices]

    x = torch.randn(5, 3)
    idx = torch.tensor([1, 2, 3], dtype=torch.int32)

    check_outputs(fn, conf, [x, idx])


def test_aten_index_select_second_dim(conf: Conf):
    """Test basic torch.index_select operation"""

    def fn(x, indices):
        return x[:, indices]

    x = torch.randn(5, 10)
    idx = torch.tensor([1, 2, 3], dtype=torch.int32)

    check_outputs(fn, conf, [x, idx])


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_aten_index_select_multiple_dtypes(conf: Conf, dtype):
    """Test basic torch.index_select operation"""

    def fn(x, indices):
        return x[:, indices]

    x = torch.randn(5, 10)
    idx = torch.tensor([1, 2, 3], dtype=dtype)

    check_outputs(fn, conf, [x, idx])


def test_aten_index_select_multiple_dims(conf: Conf):
    """Test basic torch.index_select operation"""

    def fn(x, indices1, indices2):
        return x[indices1, indices2, indices2]

    x = torch.randn(4, 5, 6, 7, 8)
    idx1 = torch.tensor([1, 2, 3], dtype=torch.int32)
    idx2 = torch.tensor([0, 2, 4], dtype=torch.int32)

    check_outputs(fn, conf, [x, idx1, idx2])


def test_aten_index_select_multiple_dims_start_nonzero(conf: Conf):
    """Test basic torch.index_select operation"""

    def fn(x, indices1, indices2):
        return x[:, indices1, indices2, indices2]

    x = torch.randn(4, 5, 6, 7, 8)
    idx1 = torch.tensor([1, 2, 3], dtype=torch.int32)
    idx2 = torch.tensor([0, 2, 4], dtype=torch.int32)

    check_outputs(fn, conf, [x, idx1, idx2])


def test_nonzero_function(device: str):
    """Test torch.nonzero() function"""

    def fn(x):
        return torch.nonzero(x)

    # Test with a simple tensor that has some zeros
    x = torch.tensor([[1, 0, 2], [0, 3, 0], [4, 0, 5]], dtype=torch.float32)

    check_functions_are_equivalent(fn, device, [x])


def test_nonzero_all_zeros(device: str):
    """Test torch.nonzero() with all zeros"""

    def fn(x):
        return torch.nonzero(x)

    x = torch.zeros(3, 3, dtype=torch.float32)

    check_functions_are_equivalent(fn, device, [x])


def test_nonzero_all_nonzeros(device: str):
    """Test torch.nonzero() with all non-zero values"""

    def fn(x):
        return torch.nonzero(x)

    x = torch.ones(2, 3, dtype=torch.float32)

    check_functions_are_equivalent(fn, device, [x])


def test_tensor_nonzero_method(device: str):
    """Test tensor.nonzero() method"""

    def fn(x):
        return x.nonzero()

    x = torch.tensor([1, 0, 3, 0, 5], dtype=torch.float32)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("tensor_shapes", [(2,), (3, 4), (2, 3, 4)])
def test_nonzero_different_shapes(device: str, tensor_shapes: tuple):
    """Test nonzero with different tensor shapes"""

    def fn(x):
        return torch.nonzero(x)

    # Create a tensor with some zeros and non-zeros
    x = torch.randn(*tensor_shapes)
    # Set some elements to zero
    if len(tensor_shapes) == 1:
        x[0] = 0
    elif len(tensor_shapes) == 2:
        x[0, 0] = 0
        x[1, 1] = 0
    else:
        x[0, 0, 0] = 0
        x[1, 1, 1] = 0

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_nearest_upsampling_2d(device: str):
    """Test F.interpolate with nearest upsampling for 2D tensors"""

    def fn(x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    # Input shape: [batch, channels, height, width]
    x = torch.randn(2, 3, 4, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_bilinear_upsampling_2d(device: str):
    """Test F.interpolate with bilinear upsampling for 2D tensors"""

    def fn(x):
        return torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=False
        )

    # Input shape: [batch, channels, height, width]
    x = torch.randn(2, 3, 4, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_nearest_downsampling_2d(device: str):
    """Test F.interpolate with nearest downsampling for 2D tensors"""

    def fn(x):
        return torch.nn.functional.interpolate(x, scale_factor=0.5, mode="nearest")

    # Input shape: [batch, channels, height, width]
    x = torch.randn(2, 3, 8, 8)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_with_size_2d(device: str):
    """Test F.interpolate with explicit output size for 2D tensors"""

    def fn(x):
        return torch.nn.functional.interpolate(x, size=(16, 16), mode="nearest")

    # Input shape: [batch, channels, height, width]
    x = torch.randn(2, 3, 8, 8)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_bilinear_with_size_2d(device: str):
    """Test F.interpolate with bilinear and explicit output size"""

    def fn(x):
        return torch.nn.functional.interpolate(
            x, size=(12, 12), mode="bilinear", align_corners=False
        )

    # Input shape: [batch, channels, height, width]
    x = torch.randn(2, 3, 6, 6)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_linear_1d(device: str):
    """Test F.interpolate with linear interpolation for 1D tensors"""

    def fn(x):
        return torch.nn.functional.interpolate(
            x, scale_factor=2, mode="linear", align_corners=False
        )

    # Input shape: [batch, channels, length]
    x = torch.randn(2, 3, 8)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_nearest_3d(device: str):
    """Test F.interpolate with nearest interpolation for 3D tensors"""

    def fn(x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")

    # Input shape: [batch, channels, depth, height, width]
    x = torch.randn(1, 2, 4, 4, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_trilinear_3d(device: str):
    """Test F.interpolate with trilinear interpolation for 3D tensors"""

    def fn(x):
        return torch.nn.functional.interpolate(
            x, scale_factor=1.5, mode="trilinear", align_corners=False
        )

    # Input shape: [batch, channels, depth, height, width]
    x = torch.randn(1, 2, 4, 4, 4)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("scale_factor", [0.5, 1.5, 2.0, 3.0])
def test_interpolate_different_scale_factors(device: str, scale_factor: float):
    """Test F.interpolate with various scale factors"""

    def fn(x):
        return torch.nn.functional.interpolate(
            x, scale_factor=scale_factor, mode="nearest"
        )

    # Input shape: [batch, channels, height, width]
    x = torch.randn(1, 2, 8, 8)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
def test_interpolate_different_modes_2d(device: str, mode: str):
    """Test F.interpolate with different interpolation modes for 2D"""

    def fn(x):
        if mode == "bilinear":
            return torch.nn.functional.interpolate(
                x, scale_factor=2, mode=mode, align_corners=False
            )
        else:
            return torch.nn.functional.interpolate(x, scale_factor=2, mode=mode)

    # Input shape: [batch, channels, height, width]
    x = torch.randn(2, 3, 4, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_align_corners_true(device: str):
    """Test F.interpolate with align_corners=True"""

    def fn(x):
        return torch.nn.functional.interpolate(
            x, scale_factor=2, mode="bilinear", align_corners=True
        )

    # Input shape: [batch, channels, height, width]
    x = torch.randn(2, 3, 4, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_small_tensor(device: str):
    """Test F.interpolate with small tensor dimensions"""

    def fn(x):
        return torch.nn.functional.interpolate(x, size=(8, 8), mode="nearest")

    # Input shape: [batch, channels, height, width]
    x = torch.randn(1, 1, 2, 2)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_single_channel(device: str):
    """Test F.interpolate with single channel input"""

    def fn(x):
        return torch.nn.functional.interpolate(x, scale_factor=3, mode="nearest")

    # Input shape: [batch, channels, height, width]
    x = torch.randn(1, 1, 4, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_asymmetric_scaling(device: str):
    """Test F.interpolate with different scaling factors for different dimensions"""

    def fn(x):
        return torch.nn.functional.interpolate(x, size=(6, 12), mode="nearest")

    # Input shape: [batch, channels, height, width]
    x = torch.randn(2, 3, 3, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_large_upsampling(device: str):
    """Test F.interpolate with large upsampling factor"""

    def fn(x):
        return torch.nn.functional.interpolate(x, scale_factor=4, mode="nearest")

    # Input shape: [batch, channels, height, width]
    x = torch.randn(1, 2, 3, 3)

    check_functions_are_equivalent(fn, device, [x])


@pytest.mark.parametrize("size", [(5, 5), (7, 9), (16, 16)])
def test_interpolate_various_output_sizes(device: str, size: tuple):
    """Test F.interpolate with various output sizes"""

    def fn(x):
        return torch.nn.functional.interpolate(x, size=size, mode="nearest")

    # Input shape: [batch, channels, height, width]
    x = torch.randn(1, 2, 4, 4)

    check_functions_are_equivalent(fn, device, [x])


def test_interpolate_preserve_aspect_ratio(conf: Conf):
    """Test F.interpolate preserving aspect ratio with rectangular input"""

    def fn(x):
        return torch.nn.functional.interpolate(
            x, size=(8, 16), mode="bilinear", align_corners=False
        )

    # Input shape: [batch, channels, height, width] - rectangular
    x = torch.randn(2, 3, 4, 8)

    check_outputs(fn, conf, [x])


def test_empty_tensor(device: str):
    """Test operations on empty tensors"""

    def fn(x):
        empty_tensor = torch.empty_like(x)
        empty_tensor.copy_(x)
        return empty_tensor + 1

    x = torch.ones((2, 3, 4))

    check_functions_are_equivalent(fn, device, [x])


def test_inplace_zero_tensor(device: str):
    """Test operations on empty tensors"""

    def fn(x):
        empty_tensor = torch.zeros_like(x)
        empty_tensor.copy_(x)
        return empty_tensor + 1

    x = torch.ones((2, 3, 4))

    check_functions_are_equivalent(fn, device, [x])
