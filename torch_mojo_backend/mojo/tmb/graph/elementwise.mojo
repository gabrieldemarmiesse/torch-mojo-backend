import extensibility
from extensibility import ElementwiseBinaryOp, ElementwiseUnaryMixedOp

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
struct Polygamma(ElementwiseBinaryOp):
    """The polygamma function of order n >= 2, with n as a (broadcast) float
    operand; n = 0 and n = 1 take the digamma / trigamma elementwise kinds."""

    @staticmethod
    def elementwise[
        dtype: DType,
        width: SIMDLength,
    ](n: SIMD[dtype, width], x: SIMD[dtype, width]) -> SIMD[dtype, width]:
        comptime assert dtype.is_floating_point(), "polygamma takes floats"
        return elementwise_polygamma(n, x)
