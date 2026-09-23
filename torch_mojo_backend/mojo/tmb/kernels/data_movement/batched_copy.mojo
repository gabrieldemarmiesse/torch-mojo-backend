"""Batched contiguous copies, converting between any two copy dtypes.

Descriptors are passed by value, with no device-side scratch allocation. Each
segment has independent source/destination pointers and a runtime element
count. The kernel is specialized on the (source, destination) dtype pair;
bool travels as its uint8 storage. Tiles measured on H100.
"""
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.collections import InlineArray
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from std.sys import size_of
from max.gpu.host import DeviceContext
from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr
from tmb.kernels.data_movement.copy_segments import (
    CopyTileSegment,
    copy_segment_index,
)

comptime COPY_BATCH_CAP = 64
comptime COPY_BATCH_THREADS = 256
# Accesses per thread per tile, whatever their width: 2, 4 and 8 measured
# 4 best on H100 for f32->bf16, bf16->bf16 and f32->f32 lists.
comptime COPY_BATCH_ITEMS = 4


struct CopySeg(CopyTileSegment, DevicePassable):
    comptime device_type: AnyType = Self
    var src: Int
    var dst: Int
    var size: Int
    var head: Int
    var tile_end: Int

    def __init__(
        out self, src: Int, dst: Int, size: Int, head: Int, tile_end: Int
    ):
        self.src = src
        self.dst = dst
        self.size = size
        self.head = head
        self.tile_end = tile_end

    @always_inline
    def tile_limit(self) -> Int:
        return self.tile_end

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "CopySeg"


@always_inline
def _storage[dt: DType]() -> DType:
    return DType.uint8 if dt == DType.bool else dt


@always_inline
def _vec[src: DType, dst: DType]() -> Int:
    """Elements per access: 16 bytes of the wider operand."""
    return 16 // max(size_of[_storage[src]()](), size_of[_storage[dst]()]())


@always_inline
def _tile[V: Int]() -> Int:
    """Elements one block moves per tile with `V`-wide accesses."""
    return COPY_BATCH_THREADS * COPY_BATCH_ITEMS * V


@always_inline
def _convert[
    src: DType, dst: DType, S: DType, D: DType, w: Int
](v: SIMD[S, w]) -> SIMD[D, w]:
    """`src` -> `dst` on their storage dtypes `S` / `D` (uint8 for bool)."""
    comptime if dst == DType.bool:
        return rebind[SIMD[D, w]](v.ne(0).cast[DType.uint8]())
    elif S == D:
        return rebind[SIMD[D, w]](v)
    elif (dst == DType.float16 or dst == DType.bfloat16) and (
        src != DType.float32
    ):
        # c10::Half and c10::BFloat16 are built from float: round through it.
        return v.cast[DType.float32]().cast[D]()
    else:
        return v.cast[D]()


@always_inline
def _head[S: DType, D: DType, V: Int](src: Int, dst: Int, size: Int) -> Int:
    """Scalar elements to peel so both pointers sit on a `V`-access boundary,
    or -1 when the two never reach one together."""
    comptime SB = V * size_of[S]()
    var head = min(((SB - src % SB) % SB) // size_of[S](), size)
    if (dst + head * size_of[D]()) % (V * size_of[D]()) != 0:
        return -1
    return head


@__name(t"copy_batched_contig_{src}_{dst}_v{V}")
def _copy_batch_kernel[
    src: DType, dst: DType, V: Int
](
    segs: InlineArray[CopySeg, COPY_BATCH_CAP],
    count_arg: Int64,
    tiles_arg: Int64,
):
    """`V`-wide accesses after each segment's `head` scalar elements. One
    width per kernel: a kernel carrying every width's route measured at twice
    the registers and half the occupancy."""
    comptime S = _storage[src]()
    comptime D = _storage[dst]()
    comptime TILE = _tile[V]()
    var tile = Int(block_idx.x)
    while tile < Int(tiles_arg):
        var lo = copy_segment_index(segs, Int(count_arg), tile)
        var seg = segs[lo]
        var first = 0 if lo == 0 else segs[lo - 1].tile_end
        var start = (tile - first) * TILE
        var sp = _make_ptr[S](seg.src)
        var dp = _make_ptr[D](seg.dst)
        var lane = Int(thread_idx.x)
        if start == 0 and lane < seg.head:
            dp[unsafe_offset=lane] = _convert[src, dst, S, D, 1](
                sp[unsafe_offset=lane]
            )
        comptime for k in range(COPY_BATCH_ITEMS):
            var i = seg.head + start + (lane + k * COPY_BATCH_THREADS) * V
            if i + V <= seg.size:
                dp.unsafe_store[width=V, alignment=V * size_of[D]()](
                    i,
                    _convert[src, dst, S, D, V](
                        sp.unsafe_load[width=V, alignment=V * size_of[S]()](i)
                    ),
                )
            elif i < seg.size:
                for j in range(i, seg.size):
                    dp[unsafe_offset=j] = _convert[src, dst, S, D, 1](
                        sp[unsafe_offset=j]
                    )
        tile += Int(grid_dim.x)


@__name(t"copy_batched_contig_{src}_{dst}_small")
def _copy_batch_small_kernel[
    src: DType, dst: DType
](segs: InlineArray[CopySeg, COPY_BATCH_CAP]):
    comptime S = _storage[src]()
    comptime D = _storage[dst]()
    var seg = segs[Int(block_idx.x)]
    var sp = _make_ptr[S](seg.src)
    var dp = _make_ptr[D](seg.dst)
    var i = Int(thread_idx.x)
    while i < seg.size:
        dp[unsafe_offset=i] = _convert[src, dst, S, D, 1](sp[unsafe_offset=i])
        i += Int(block_dim.x)


def copy_batched[
    src: DType, dst: DType
](
    srcs: List[Int], dsts: List[Int], sizes: List[Int], ctx: DeviceContext
) raises:
    """Copy `sizes[i]` elements from `srcs[i]` to `dsts[i]` for every i, in
    ceil(len / COPY_BATCH_CAP) launches. The spans must not overlap."""
    comptime S = _storage[src]()
    comptime D = _storage[dst]()
    comptime VEC = _vec[src, dst]()
    var index = 0
    while index < len(sizes):
        var segs = InlineArray[CopySeg, COPY_BATCH_CAP](
            fill=CopySeg(0, 0, 0, 0, 0)
        )
        var count = 0
        var largest = 0
        # The launch takes the widest access every one of its segments allows.
        var width = VEC
        while index < len(sizes) and count < COPY_BATCH_CAP:
            var n = sizes[index]
            if n > 0:
                largest = max(largest, n)
                segs[count] = CopySeg(srcs[index], dsts[index], n, 0, 0)
                count += 1
                comptime for step in range(5):  # VEC <= 16: 16, 8, 4, 2
                    comptime V = VEC >> step
                    comptime if V > 1:
                        if (
                            width == V
                            and _head[S, D, V](srcs[index], dsts[index], n) < 0
                        ):
                            width = V // 2
            index += 1
        if count == 0:
            continue
        comptime for step in range(5):
            comptime V = VEC >> step
            comptime if V >= 1:
                if width == V:
                    comptime TILE = _tile[V]()
                    if largest <= TILE:
                        _enqueue_cached[_copy_batch_small_kernel[src, dst]](
                            ctx,
                            String(t"copy_batched_contig_{src}_{dst}_small"),
                            count,
                            1,
                            1,
                            COPY_BATCH_THREADS,
                            segs,
                        )
                    else:
                        var tiles = 0
                        for i in range(count):
                            segs[i].head = _head[S, D, V](
                                segs[i].src, segs[i].dst, segs[i].size
                            )
                            tiles += ceildiv(segs[i].size, TILE)
                            segs[i].tile_end = tiles
                        _enqueue_cached[_copy_batch_kernel[src, dst, V]](
                            ctx,
                            String(t"copy_batched_contig_{src}_{dst}_v{V}"),
                            min(tiles, 1 << 22),
                            1,
                            1,
                            COPY_BATCH_THREADS,
                            segs,
                            Int64(count),
                            Int64(tiles),
                        )
