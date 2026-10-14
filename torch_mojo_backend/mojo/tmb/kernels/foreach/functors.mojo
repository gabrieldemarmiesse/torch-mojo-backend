"""The element math of the foreach family: one function, `foreach_math`, that
every `aten::_foreach_*` overload's kernel calls with a comptime `kind`.

ATen's CUDA foreach kernels (ForeachFunctors.cuh) widen each element to
`opmath_t` (float32 for the half types and float32, double for float64, the
integer itself for integer dtypes), apply a functor, and round once on the
store. The functors are the per-tensor ops' own math, so the bodies here are
calls into the shared math modules the single-tensor kernels use, never a
copy of them:

  * the unary kinds are `unary_math.elementwise_unary` -- `_foreach_exp` and
    `exp` share one definition of exp, with its half-precision rounding point
    (unary ops get the raw dtype vector: that function does its own widening);
  * maximum / minimum (clamp_min / clamp_max), lerp and pow are
    `pointwise_math.pointwise` and `pow_math.torch_pow`;
  * addcmul / addcdiv are `pointwise_math.pointwise_op_impl`, the port of
    DeviceAddCmulCdiv.cuh that CUDA's per-tensor and foreach kernels share;
  * add / sub / mul / div are single opmath instructions (an fma for an
    alpha-scaled operand, which is what nvcc contracts `a + alpha * b` to).

Operands: `a` is the element of the first list in its own dtype; `b` and `c`
are already in opmath, whatever they came from (a second / third list, a
host Scalar, a 0-d device tensor); `alpha` is the opmath scalar parameter
(add / sub's alpha, addcmul's value), 1 where the overload has none.
"""

from tmb.kernels.common.pointwise_math import (
    param_dtype,
    pointwise,
    pointwise_op_impl,
)
from tmb.kernels.common.unary_math import elementwise_unary


# The unary kinds, each `unary_math.elementwise_unary`'s kind of the same name.
comptime FOREACH_UNARY_KINDS = ["sqrt"]

# Everything else `foreach_math` computes. `nonfinite_unscale` is GradScaler's
# `_amp_foreach_non_finite_check_and_unscale_`, whose flag write the kernel
# does itself (`apply.mojo`): here it is only the multiply.
comptime FOREACH_OTHER_KINDS = [
    "add",
    "mul",
    "addcmul",
    "addcdiv",
    "lerp",
    "nonfinite_unscale",
]


@always_inline
def is_foreach_unary[kind: StaticString]() -> Bool:
    comptime for k in FOREACH_UNARY_KINDS:
        comptime if kind == k:
            return True
    return False


@always_inline
def is_foreach_kind[kind: StaticString]() -> Bool:
    comptime if is_foreach_unary[kind]():
        return True
    comptime for k in FOREACH_OTHER_KINDS:
        comptime if kind == k:
            return True
    return False


@always_inline
def foreach_opmath[dtype: DType]() -> DType:
    """ATen's `opmath_type`: float32 for the half types, the dtype itself
    otherwise (a bool list reaches here as int32, see `apply.mojo`)."""
    comptime if dtype == DType.float16 or dtype == DType.bfloat16:
        return DType.float32
    else:
        return dtype


@always_inline
def foreach_math[
    kind: StaticString, dtype: DType, n: Int
](
    a: SIMD[dtype, n],
    b: SIMD[foreach_opmath[dtype](), n],
    c: SIMD[foreach_opmath[dtype](), n],
    alpha: Scalar[foreach_opmath[dtype]()],
) -> SIMD[dtype, n]:
    """`kind` of one vector of elements, rounded once to `dtype`."""
    comptime op = foreach_opmath[dtype]()
    comptime if is_foreach_unary[kind]():
        return elementwise_unary[kind](a)
    else:
        var x = a.cast[op]()
        var r: SIMD[op, n]
        comptime if kind == "add":
            # BinaryOpListAlphaFunctor: op(a, alpha * b), contracted.
            comptime if op.is_floating_point():
                r = SIMD[op, n](alpha).fma(b, x)
            else:
                r = x + alpha * b
        elif kind == "mul" or kind == "nonfinite_unscale":
            r = x * b
        elif kind == "addcmul":
            r = pointwise_op_impl[True](x, b, c, alpha)
        elif kind == "addcdiv":
            r = pointwise_op_impl[False](x, b, c, alpha)
        elif kind == "lerp":
            # Lerp.h through the per-tensor lerp's own math: the weight is
            # `c`, a third list or a splatted Scalar.
            r = pointwise["lerp", op, op, n](
                x, b, c, SIMD[param_dtype[op](), 4](0)
            )
        else:
            comptime assert False, "no foreach math for this kind"
        return r.cast[dtype]()
