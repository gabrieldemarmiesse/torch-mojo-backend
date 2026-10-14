"""The launch contract of the foreach family, shared by the kernels and the
ops layer (`tmb/ops/foreach.mojo`), which encodes it: no device code here.

An overload's operand variant is one int, so it travels as a single
`-D FE_VARIANT=<v>` define:
  bits 0-1  input tensor lists (1, 2 or 3)
  bit  2    out-of-place (a fresh output list; else the first list is written)
  bits 3-4  the per-tensor host scalar's role: FE_ROLE_*
  bit  5    the second operand is a 0-d device tensor of the list dtype
"""

comptime FE_OUT = 4
comptime FE_DEVICE_B = 32
comptime FE_ROLE_NONE = 0
comptime FE_ROLE_ALPHA = 1  # the alpha / value parameter
comptime FE_ROLE_B = 2  # splatted into the second operand
comptime FE_ROLE_C = 3  # splatted into the third operand


@always_inline
def fe_variant(
    lists: Int, out_of_place: Bool, role: Int, device_b: Bool
) -> Int:
    return (
        lists
        + (FE_OUT if out_of_place else 0)
        + role * 8
        + (FE_DEVICE_B if device_b else 0)
    )


@always_inline
def fe_lists[V: Int]() -> Int:
    return V & 3


@always_inline
def fe_out_of_place[V: Int]() -> Bool:
    return (V & FE_OUT) != 0


@always_inline
def fe_role[V: Int]() -> Int:
    return (V >> 3) & 3


@always_inline
def fe_device_b[V: Int]() -> Bool:
    return (V & FE_DEVICE_B) != 0


def foreach_apple_supported(kind: StringSlice, v: Int) -> Bool:
    """The overloads `apple_kernels.mojo` has a float32 launch for: the host
    routes nothing else to the multi-tensor apply on Apple, and the kernel
    side evaluates the same function at compile time."""
    var inplace_b = fe_variant(1, False, FE_ROLE_B, False)
    return (
        ((kind == "mul" or kind == "add" or kind == "div") and v == inplace_b)
        or (kind == "mul" and v == fe_variant(1, False, FE_ROLE_NONE, True))
        or (
            (kind == "addcmul" or kind == "addcdiv")
            and v == fe_variant(3, False, FE_ROLE_ALPHA, False)
        )
        or (kind == "lerp" and v == fe_variant(2, False, FE_ROLE_C, False))
        or (kind == "sqrt" and v == fe_variant(1, True, FE_ROLE_NONE, False))
        or (
            kind == "nonfinite_unscale"
            and v == fe_variant(1, False, FE_ROLE_NONE, False)
        )
    )


# The foreach reductions (reduce.mojo): the reduction kinds the `norm` / `max`
# builds take as FE_NORM, and the chunk the host sizes the partials scratch
# by -- CUDA's kChunkSize.
comptime FE_REDUCE_CHUNK = 65_536
comptime FR_L0 = 0
comptime FR_L1 = 1
comptime FR_L2 = 2
comptime FR_LINF = 3
comptime FR_MAX = 4
comptime FR_LNEGINF = 5  # the -inf norm: min of |x|
comptime FR_LP = 6  # any other order p: sum(|x|^p) (and its 1/p root)
