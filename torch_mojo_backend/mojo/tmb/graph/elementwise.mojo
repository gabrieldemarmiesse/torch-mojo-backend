import extensibility
from extensibility import (
    ElementwiseUnaryMixedOp,
    InputTensor,
    OutputTensor,
    foreach,
)
from max.gpu.host import DeviceContext
from std.utils.coord import Coord

from tmb.kernels.common.unary_math import (
    elementwise_polygamma,
    elementwise_predicate,
    elementwise_unary,
)


struct ElementwiseOp[kind: StaticString](ElementwiseUnaryMixedOp):
    # Concrete registrations are generated from one template. MAX 26.5's
    # elementwise fusion lowering does not forward parent-struct parameters.
    @staticmethod
    def elementwise[
        dtype: DType,
        out_dtype: DType,
        width: SIMDLength,
    ](x: SIMD[dtype, width]) -> SIMD[out_dtype, width]:
        comptime if (
            Self.kind == "isinf"
            or Self.kind == "isnan"
            or Self.kind == "logical_not"
            or Self.kind == "signbit"
        ):
            comptime assert out_dtype == DType.bool, "expected boolean output"
            return elementwise_predicate[Self.kind](x).cast[out_dtype]()
        else:
            comptime assert (
                out_dtype == dtype
            ), "expected matching input/output dtypes"
            return elementwise_unary[Self.kind](x).cast[out_dtype]()


@extensibility.register("polygamma")
struct Polygamma:
    """The polygamma function of order n >= 2 (n = 0 and n = 1 take the
    digamma / trigamma elementwise kinds). `n`, the int64 order, is a
    parameter: a float operand of the input dtype would round it (bfloat16
    257 -> 256, float32 16777217 -> 16777216), flipping the sign of the
    result."""

    @staticmethod
    def execute[
        n: Int,
        target: StaticString,
    ](
        output: OutputTensor,
        x: InputTensor[dtype=output.dtype, rank=output.rank, ...],
        ctx: DeviceContext,
    ) raises:
        comptime assert (
            output.dtype.is_floating_point()
        ), "polygamma takes floats"

        @__parameter
        @__copy_capture(x)
        @always_inline
        def lane[width: Int](idx: Coord) -> SIMD[output.dtype, width]:
            return elementwise_polygamma[n, True](x.load[width](idx))

        foreach[lane, target=target](output, ctx)
