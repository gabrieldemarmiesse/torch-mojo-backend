"""Block-wide reductions through shared memory, for the kernels whose
accumulator dtype (float64 among them) the warp-shuffle `block.sum` of
max.gpu.primitives cannot lower."""

from max.gpu import thread_idx
from max.gpu.sync import barrier
from std.memory import stack_allocation


@always_inline
def block_sum[dtype: DType, threads: Int](v: Scalar[dtype]) -> Scalar[dtype]:
    """Sum of `v` over a block of exactly `threads` (a power of two)
    threads, returned to every thread: a shared-memory tree, so float64
    works where the warp-shuffle `block.sum` has no 64-bit lowering. Every
    thread of the block must call it."""
    var sh = stack_allocation[
        threads, dtype, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    sh[unsafe_offset=tid] = v
    barrier()
    var stride = threads // 2
    while stride > 0:
        if tid < stride:
            sh[unsafe_offset=tid] += sh[unsafe_offset=tid + stride]
        barrier()
        stride //= 2
    var total = sh[unsafe_offset=0]
    barrier()  # the next call may reuse this shared buffer
    return total
