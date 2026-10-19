"""Fixed-width random-kernel metadata with a supported Metal launch ABI.

Bare SIMD kernel arguments (even inside a register-passable struct) retain
invalid vector argument metadata in AIR. Array encodes its indices as
an aggregate, like the existing Metal foreach launch metadata.
"""

from tmb.kernels.common.op_utils import Arg, _raw_tuple_int, _raw_tuple_len

comptime I64x8 = Array[Int64, 8]


@always_inline
def _join_u64(lo: Int, hi: Int) -> UInt64:
    return UInt64(lo) | (UInt64(hi) << 32)


@always_inline
def _tuple_i64x8(t: Arg) -> I64x8:
    var out = I64x8(fill=Int64(0))
    var n = _raw_tuple_len(t)
    for i in range(min(n, 8)):
        out[i] = Int64(_raw_tuple_int(t, i))
    return out^
