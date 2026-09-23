"""The embedding gather (`Gather0`), shared by the eager `Gather0` op and any
MAX custom op that wants the same kernel. Kept out of the family's entry
module for the reason softmax_rows_kernels.mojo states: that module exports
`tmb_call`, which one MAX compilation unit may import only once.
"""

from max.gpu.host import DeviceContext
from std.utils.coord import Coord

from tmb.kernels.common.op_utils import (
    _make_ptr,
    _parallel_for,
)


# ---------------------------------------------------------------------------
# Embedding lookup: out[i] = weight[indices[i // row_len] * row_len +
# i % row_len]. This is gather along dim 0 of a 2D weight table.
#
# `num_rows` is the table's row count: a row index outside [0, num_rows)
# would read arbitrary device memory, so it is clamped into range. Clamped
# rather than reported because a host-visible error flag costs a device
# synchronization on every embedding lookup -- the first op of every
# transformer forward -- and CUDA_KERNEL_ASSERT has no portable equivalent
# across CUDA / HIP / Metal. The value produced for an invalid index is
# unspecified; the memory access is not.
# ---------------------------------------------------------------------------


@always_inline
def _gather0[
    dtype: DType, idx_dtype: DType
](
    out_addr: Int,
    weight_addr: Int,
    indices_addr: Int,
    num_indices: Int,
    row_len: Int,
    num_rows: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var weight_ptr = _make_ptr[dtype](weight_addr)
    var indices_ptr = _make_ptr[idx_dtype](indices_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, weight_ptr, indices_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var row = Int(indices_ptr[unsafe_offset=i // row_len])
        if row < 0:
            row = 0
        elif row >= num_rows:
            row = num_rows - 1
        out_ptr[unsafe_offset=i] = weight_ptr[
            unsafe_offset=row * row_len + i % row_len
        ]

    _parallel_for[func](num_indices * row_len, ctx)
