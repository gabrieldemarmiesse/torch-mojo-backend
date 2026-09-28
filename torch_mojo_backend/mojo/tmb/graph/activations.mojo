import std.math as math
import extensibility as compiler
from extensibility import ElementwiseBinaryOp, InputTensor, OutputTensor
from max.gpu.host import DeviceContext
from std.sys.info import size_of
from std.utils.coord import Coord

from tmb.kernels.common.gpu_elementwise import elementwise
from tmb.kernels.common.pointwise_math import param_dtype, pointwise


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

        var x = input
        var grad_out = grad_output

        # Compute CDF term: 0.5 * (1 + erf(x * M_SQRT1_2))
        var cdf = 0.5 * (1.0 + math.erf(x * M_SQRT1_2))

        # Compute PDF term: PDF_CONSTANT * exp(-0.5 * x²)
        var x_squared = x * x
        var pdf = PDF_CONSTANT * math.exp(-0.5 * x_squared)

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

        var x = input
        var grad_out = grad_output

        # Compute inner = kBeta * (x + kKappa * x³)
        var x_squared = x * x
        var x_cubed = x_squared * x
        var inner = k_Beta * (x + k_Kappa * x_cubed)
        var tanh_inner = math.tanh(inner)

        # Left term derivatives
        var left = 0.5 * x
        var right = 1.0 + tanh_inner
        var left_derivative = 0.5 * right

        # Right term derivatives
        var tanh_derivative = 1.0 - tanh_inner * tanh_inner
        var inner_derivative = k_Beta * (1.0 + 3.0 * k_Kappa * x_squared)
        var right_derivative = left * tanh_derivative * inner_derivative

        # Total gradient
        return grad_out * (left_derivative + right_derivative)


@compiler.register("native_glu_backward_b")
struct NativeGluBackwardB:
    """The second gradient half of glu_backward, `(1 - sigmoid(b)) * sigmoid(b) *
    grad * a`: the mojo device's own `glu_backward_b` pointwise kind over
    three flat operands of one shape (the elementwise traits stop at two
    inputs, so this is a plain custom op launching the shared launcher)."""

    @staticmethod
    def execute[
        dtype: DType, //, target: StaticString
    ](
        output: OutputTensor[dtype=dtype, rank=1, ...],
        grad: InputTensor[dtype=dtype, rank=1, ...],
        a: InputTensor[dtype=dtype, rank=1, ...],
        b: InputTensor[dtype=dtype, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            var numel = output.dim_size(0)
            if numel == 0:
                return
            var out_ptr = output.unsafe_ptr()
            var g_ptr = grad.unsafe_ptr()
            var a_ptr = a.unsafe_ptr()
            var b_ptr = b.unsafe_ptr()

            @always_inline
            @__parameter
            @__copy_capture(out_ptr, g_ptr, a_ptr, b_ptr)
            def func[w: Int, alignment: Int = 1](idx: Coord):
                var i = Int(idx[0].value())
                out_ptr.unsafe_store[width=w](
                    i,
                    pointwise["glu_backward_b", dtype, dtype, w](
                        g_ptr.unsafe_load[width=w](i),
                        a_ptr.unsafe_load[width=w](i),
                        b_ptr.unsafe_load[width=w](i),
                        SIMD[param_dtype[dtype](), 4](0),
                    ),
                )

            elementwise[
                func,
                simd_width=16 // size_of[dtype](),
                target="gpu",
                _trace_description="glu_backward_b",
                _heavy=True,
            ](Coord(numel), ctx)
        else:
            raise Error("native_glu_backward_b runs on an accelerator only")
