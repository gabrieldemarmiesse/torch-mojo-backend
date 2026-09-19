"""Batched contiguous F32-to-BF16 copies; tiles measured on Hopper.

Descriptors are passed by value, with no device-side scratch allocation. Each
segment has independent source/destination pointers and a runtime element count.
"""
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.collections import InlineArray
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceContext
from op_utils import _enqueue_cached, _make_ptr
from copy_segments import CopyTileSegment, copy_segment_index

comptime COPY_CAST_CAP = 64
comptime COPY_CAST_THREADS = 256
comptime COPY_CAST_TILE = 2048


struct CopyCastSeg(CopyTileSegment, DevicePassable):
    comptime device_type: AnyType = Self
    var src: Int
    var dst: Int
    var size: Int
    var tile_end: Int

    def __init__(out self, src: Int, dst: Int, size: Int, tile_end: Int):
        self.src = src
        self.dst = dst
        self.size = size
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
        return "CopyCastSeg"


@__name("copy_batched_contig_f32_bf16_v2")
def _copy_cast_kernel(
    segs: InlineArray[CopyCastSeg, COPY_CAST_CAP],
    count_arg: Int64,
    tiles_arg: Int64,
):
    var tile = Int(block_idx.x)
    while tile < Int(tiles_arg):
        var lo = copy_segment_index(segs, Int(count_arg), tile)
        var seg = segs[lo]
        var first = 0 if lo == 0 else segs[lo - 1].tile_end
        var start = (tile - first) * COPY_CAST_TILE
        var src = _make_ptr[DType.float32](seg.src)
        var dst = _make_ptr[DType.bfloat16](seg.dst)
        var lane = Int(thread_idx.x)
        var head = (seg.dst // 2) % 2
        if (seg.src // 4) % 2 == head:
            comptime for k in range(COPY_CAST_TILE // (COPY_CAST_THREADS * 2)):
                var i = head + start + (lane + k * COPY_CAST_THREADS) * 2
                if i + 1 < seg.size:
                    dst.unsafe_store[width=2, alignment=4](
                        i,
                        src.unsafe_load[width=2, alignment=8](i).cast[
                            DType.bfloat16
                        ](),
                    )
            if lane == 0:
                if start == 0 and head == 1:
                    dst[unsafe_offset=0] = src[unsafe_offset=0].cast[
                        DType.bfloat16
                    ]()
                var tail = head + ((seg.size - head) // 2) * 2
                if (
                    tail < seg.size
                    and tail >= start
                    and tail < start + COPY_CAST_TILE
                ):
                    dst[unsafe_offset=tail] = src[unsafe_offset=tail].cast[
                        DType.bfloat16
                    ]()
        else:
            comptime for k in range(COPY_CAST_TILE // COPY_CAST_THREADS):
                var i = start + lane + k * COPY_CAST_THREADS
                if i < seg.size:
                    dst[unsafe_offset=i] = src[unsafe_offset=i].cast[
                        DType.bfloat16
                    ]()
        tile += Int(grid_dim.x)


@__name("copy_batched_contig_f32_bf16_small")
def _copy_cast_small_kernel(segs: InlineArray[CopyCastSeg, COPY_CAST_CAP]):
    var seg = segs[Int(block_idx.x)]
    var src = _make_ptr[DType.float32](seg.src)
    var dst = _make_ptr[DType.bfloat16](seg.dst)
    var i = Int(thread_idx.x)
    while i < seg.size:
        dst[unsafe_offset=i] = src[unsafe_offset=i].cast[DType.bfloat16]()
        i += Int(block_dim.x)


def copy_batched_cast(
    srcs: List[Int], dsts: List[Int], sizes: List[Int], ctx: DeviceContext
) raises:
    var index = 0
    while index < len(sizes):
        var segs = InlineArray[CopyCastSeg, COPY_CAST_CAP](
            fill=CopyCastSeg(0, 0, 0, 0)
        )
        var count = 0
        var tiles = 0
        var largest = 0
        while index < len(sizes) and count < COPY_CAST_CAP:
            var n = sizes[index]
            if n > 0:
                largest = max(largest, n)
                tiles += ceildiv(n, COPY_CAST_TILE)
                segs[count] = CopyCastSeg(srcs[index], dsts[index], n, tiles)
                count += 1
            index += 1
        if tiles > 0 and largest <= COPY_CAST_TILE:
            _enqueue_cached[_copy_cast_small_kernel](
                ctx,
                count,
                1,
                1,
                COPY_CAST_THREADS,
                segs,
            )
        elif tiles > 0:
            _enqueue_cached[_copy_cast_kernel](
                ctx,
                min(tiles, 1 << 22),
                1,
                1,
                COPY_CAST_THREADS,
                segs,
                Int64(count),
                Int64(tiles),
            )
