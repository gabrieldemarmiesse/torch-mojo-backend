# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/device/symmetric/gin_scratch.h
#
# The three device kernels the inter-node hop needs on top of
# the intra-node kernels of device/symmetric/, which are untouched.
#
# None of them synchronizes with anything: stream order does it. Each runs
# after the host callback that put the network data in place (transport/net.mojo)
# and before the intra-node collective that consumes the result, so there is
# no flag protocol here and no peer pointer -- every address is inside this
# rank's own region or its own user buffers.

from max.gpu.host import DeviceContext, DeviceStream
from max.gpu import MAX_THREADS_PER_BLOCK_METADATA, global_idx, grid_dim
from std.utils import StaticTuple
from std.sys import size_of

from tmb.ccl.device.common import _enqueue_cached
from tmb.ccl.device.symmetric.data_ops import _copy_bytes
from tmb.ccl.include.device import BLOCK


comptime _UNROLL = 4
comptime _MAX_BLOCKS = 432


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_internode_inbox_add_{dtype}")
def _inbox_add_kernel[
    dtype: DType, W: Int
](
    shard: Pointer[Scalar[dtype], MutAnyOrigin],
    inbox: Pointer[UInt8, MutAnyOrigin],
    count: Int64,
    slot_bytes: Int64,
    npeers_i: Int32,
):
    """`shard += sum of the npeers inbox slots`, elementwise.

    The node-local sum of this rank's shard is already in `shard` (that is
    what `reduce_scatter_stage` left there); each remote node's node-local
    sum of the SAME shard has landed in one inbox slot. Adding them makes
    `shard` the global sum, which `allgather_finish` then spreads.

    Accumulated in the wire dtype, like the rest of the hierarchical path:
    the remote sums arrived rounded already, so a wider accumulator here
    would buy nothing.
    """
    _inbox_add_body[dtype, W](
        shard,
        inbox,
        Int(count),
        Int(slot_bytes),
        Int(npeers_i),
        Int(global_idx.x),
        Int(grid_dim.x) * BLOCK,
    )


@always_inline
def _inbox_add_body[
    dtype: DType, W: Int
](
    shard: Pointer[Scalar[dtype], MutAnyOrigin],
    inbox: Pointer[UInt8, MutAnyOrigin],
    n: Int,
    sb: Int,
    npeers: Int,
    tid: Int,
    stride: Int,
):
    """`_inbox_add_kernel`'s body, shared with the fused inter-node kernel
    (all_reduce_gin.mojo), which runs it once per chunk on its own
    grid-stride slice."""
    var nv = n // W

    for v in range(tid, nv, stride):
        var acc = shard.unsafe_load[width=W, alignment=16](v * W)
        for j in range(npeers):
            var src = inbox.unsafe_offset(j * sb).unsafe_bitcast[
                Scalar[dtype]
            ]()
            acc += src.unsafe_load[width=W, alignment=16](v * W)
        shard.unsafe_store[width=W, alignment=16](v * W, acc)

    for i in range(nv * W + tid, n, stride):
        var acc = shard[unsafe_offset=i]
        for j in range(npeers):
            var src = inbox.unsafe_offset(j * sb).unsafe_bitcast[
                Scalar[dtype]
            ]()
            acc += src[unsafe_offset=i]
        shard[unsafe_offset=i] = acc


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name("ccl_internode_copy_bytes")
def _copy_kernel(
    dst: Pointer[UInt8, MutAnyOrigin],
    src: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int64,
):
    """Staging copy: a user buffer is not in the registered region, so
    broadcast and allgather have to move their payload in and out of it."""
    _copy_bytes[_UNROLL](
        dst,
        src,
        Int(nbytes),
        Int(global_idx.x),
        Int(grid_dim.x) * BLOCK,
    )


# ===-------------------------------------------------------------------=== #
# Launchers
# ===-------------------------------------------------------------------=== #


def _blocks_for(nbytes: Int) -> Int:
    return min(_MAX_BLOCKS, max(1, (nbytes // 16 + BLOCK - 1) // BLOCK))


def inbox_add[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    shard_ptr: Int,
    inbox_ptr: Int,
    count: Int,
    slot_bytes: Int,
    npeers: Int,
) raises:
    """Enqueue `shard += sum(inbox slots)` on `stream`."""
    if count <= 0 or npeers <= 0:
        return
    comptime W = 16 // size_of[dtype]()
    _enqueue_cached[_inbox_add_kernel[dtype, W]](
        ctx,
        stream,
        String(t"ib_add_{dtype}"),
        _blocks_for(count * size_of[dtype]()),
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=shard_ptr),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=inbox_ptr),
        Int64(count),
        Int64(slot_bytes),
        Int32(npeers),
    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK))
)
@__name(t"ccl_internode_inbox_sum_out_{dtype}_v{W}")
def _inbox_sum_out_kernel[
    dtype: DType, W: Int
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    partial: Pointer[Scalar[dtype], MutAnyOrigin],
    inbox: Pointer[UInt8, MutAnyOrigin],
    count: Int64,
    slot_bytes: Int64,
    npeers_i: Int32,
):
    comptime accum = DType.float32 if (
        dtype == DType.bfloat16 or dtype == DType.float16
    ) else dtype
    var tid = Int(global_idx.x)
    var stride = Int(grid_dim.x) * BLOCK
    var n = Int(count)
    for v in range(tid, n // W, stride):
        var acc = partial.unsafe_load[width=W](v * W).cast[accum]()
        for j in range(Int(npeers_i)):
            var src = inbox.unsafe_offset(j * Int(slot_bytes)).unsafe_bitcast[
                Scalar[dtype]
            ]()
            acc += src.unsafe_load[width=W](v * W).cast[accum]()
        dst.unsafe_store[width=W](v * W, acc.cast[dtype]())
    for i in range(n // W * W + tid, n, stride):
        var acc = partial[unsafe_offset=i].cast[accum]()
        for j in range(Int(npeers_i)):
            var src = inbox.unsafe_offset(j * Int(slot_bytes)).unsafe_bitcast[
                Scalar[dtype]
            ]()
            acc += src[unsafe_offset=i].cast[accum]()
        dst[unsafe_offset=i] = acc.cast[dtype]()


def inbox_sum_out[
    dtype: DType
](
    ctx: DeviceContext,
    stream: DeviceStream,
    out_ptr: Int,
    partial_ptr: Int,
    inbox_ptr: Int,
    count: Int,
    slot_bytes: Int,
    npeers: Int,
) raises:
    """Sum node partials straight into user memory, including offset views."""
    comptime W = 16 // size_of[dtype]()
    _enqueue_cached[_inbox_sum_out_kernel[dtype, W]](
        ctx,
        stream,
        String(t"ib_sum_out_{dtype}"),
        _blocks_for(count * size_of[dtype]()),
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=out_ptr),
        Pointer[Scalar[dtype], MutAnyOrigin](unsafe_from_address=partial_ptr),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=inbox_ptr),
        Int64(count),
        Int64(slot_bytes),
        Int32(npeers),
    )


def copy_bytes(
    ctx: DeviceContext, stream: DeviceStream, dst: Int, src: Int, nbytes: Int
) raises:
    if nbytes <= 0:
        return
    _enqueue_cached[_copy_kernel](
        ctx,
        stream,
        "ib_copy",
        _blocks_for(nbytes),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=dst),
        Pointer[UInt8, MutAnyOrigin](unsafe_from_address=src),
        Int64(nbytes),
    )
