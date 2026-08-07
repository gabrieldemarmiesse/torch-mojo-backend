import std.math as math
import extensibility as compiler
from extensibility import ElementwiseBinaryOp


@compiler.register("gelu_backward")
struct GeluBackwardNoneKernel(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](grad_output: SIMD[dtype, width], input: SIMD[dtype, width]) -> SIMD[
        dtype, width
    ]:
        comptime assert (
            dtype.is_floating_point()
        ), "gelu_backward requires floating point dtype"

        # Exact GELU backward using error function
        # Formula: grad = dy * (CDF + x * PDF)
        # where CDF = 0.5 * (1 + erf(x * M_SQRT1_2))
        #       PDF = (M_2_SQRTPI * M_SQRT1_2 * 0.5) * exp(-0.5 * x²)

        # Constants from PyTorch implementation
        comptime M_SQRT1_2 = 0.7071067811865476  # sqrt(1/2) = 1/sqrt(2)
        comptime PDF_CONSTANT = 0.39894228040143276  # M_2_SQRTPI * M_SQRT1_2 * 0.5

        x = input
        grad_out = grad_output

        # Compute CDF term: 0.5 * (1 + erf(x * M_SQRT1_2))
        cdf = 0.5 * (1.0 + math.erf(x * M_SQRT1_2))

        # Compute PDF term: PDF_CONSTANT * exp(-0.5 * x²)
        x_squared = x * x
        pdf = PDF_CONSTANT * math.exp(-0.5 * x_squared)

        # Gradient: grad_out * (CDF + x * PDF)
        return grad_out * (cdf + x * pdf)


@compiler.register("gelu_backward_tanh")
struct GeluBackwardTanhKernel(ElementwiseBinaryOp):
    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](grad_output: SIMD[dtype, width], input: SIMD[dtype, width]) -> SIMD[
        dtype, width
    ]:
        comptime assert (
            dtype.is_floating_point()
        ), "gelu_backward requires floating point dtype"

        # Tanh approximation backward
        # Formula: grad = dy * (left_derivative + right_derivative)
        # See PyTorch CUDA implementation for details

        # Constants from PyTorch implementation
        comptime k_Beta = 0.7978845608028654  # sqrt(2) * sqrt(2/π) * 0.5
        comptime k_Kappa = 0.044715

        x = input
        grad_out = grad_output

        # Compute inner = kBeta * (x + kKappa * x³)
        x_squared = x * x
        x_cubed = x_squared * x
        inner = k_Beta * (x + k_Kappa * x_cubed)
        tanh_inner = math.tanh(inner)

        # Left term derivatives
        left = 0.5 * x
        right = 1.0 + tanh_inner
        left_derivative = 0.5 * right

        # Right term derivatives
        tanh_derivative = 1.0 - tanh_inner * tanh_inner
        inner_derivative = k_Beta * (1.0 + 3.0 * k_Kappa * x_squared)
        right_derivative = left * tanh_derivative * inner_derivative

        # Total gradient
        return grad_out * (left_derivative + right_derivative)
