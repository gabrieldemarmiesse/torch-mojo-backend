from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.collections import InlineArray
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.math import ceildiv
from max.gpu.host import DeviceContext
from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr
from tmb.kernels.data_movement.copy_segments import (
    CopyTileSegment,
    copy_segment_index,
)

# 80*48=3840 parameter bytes, leaving 256 bytes in a conservative 4KiB parameter budget.
comptime COPY_ROWS_CAP = 80
comptime COPY_ROWS_THREADS = 256
comptime COPY_ROWS_TILE = 2048
# The short-row launch regime was measured on H100; other targets stay gated.
comptime COPY_SMALL_ROWS = 8


struct CopyRect(CopyTileSegment, DevicePassable):
    comptime device_type: AnyType = Self
    var src: Int
    var dst: Int
    var rows: Int
    var cols: Int
    var pitch: Int
    var tile_end: Int

    def __init__(
        out self,
        src: Int,
        dst: Int,
        rows: Int,
        cols: Int,
        pitch: Int,
        tile_end: Int,
    ):
        self.src = src
        self.dst = dst
        self.rows = rows
        self.cols = cols
        self.pitch = pitch
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
        return "CopyRect"


@__name("copy_batched_rows_u16_small")
def _copy_rows_small_kernel(segs: InlineArray[CopyRect, COPY_ROWS_CAP]):
    var seg = segs[Int(block_idx.x)]
    var src = _make_ptr[DType.uint16](seg.src)
    var dst = _make_ptr[DType.uint16](seg.dst)
    var row = Int(block_idx.y)
    if row < seg.rows:
        var col = Int(thread_idx.x)
        while col < seg.cols:
            dst[unsafe_offset=row * seg.cols + col] = src[
                unsafe_offset=row * seg.pitch + col
            ]
            col += Int(block_dim.x)


@__name("copy_batched_rows_u16_v8")
def _copy_rows_kernel(
    segs: InlineArray[CopyRect, COPY_ROWS_CAP],
    count_arg: Int64,
    tiles_arg: Int64,
):
    var tile = Int(block_idx.x)
    while tile < Int(tiles_arg):
        var lo = copy_segment_index(segs, Int(count_arg), tile)
        var seg = segs[lo]
        var first = 0 if lo == 0 else segs[lo - 1].tile_end
        var local_tile = tile - first
        var src = _make_ptr[DType.uint16](seg.src)
        var dst = _make_ptr[DType.uint16](seg.dst)
        var lane = Int(thread_idx.x)
        if seg.cols > COPY_ROWS_TILE:
            var tiles_per_row = (
                seg.cols + COPY_ROWS_TILE - 1
            ) // COPY_ROWS_TILE
            var row = local_tile // tiles_per_row
            var start = (local_tile % tiles_per_row) * COPY_ROWS_TILE
            if (
                seg.src % 16 == 0
                and seg.dst % 16 == 0
                and seg.cols % 8 == 0
                and seg.pitch % 8 == 0
            ):
                var col = start + lane * 8
                if col + 8 <= seg.cols:
                    dst.unsafe_store[width=8, alignment=16](
                        row * seg.cols + col,
                        src.unsafe_load[width=8, alignment=16](
                            row * seg.pitch + col
                        ),
                    )
            else:
                comptime for k in range(COPY_ROWS_TILE // COPY_ROWS_THREADS):
                    var col = start + lane + k * COPY_ROWS_THREADS
                    if col < seg.cols:
                        dst[unsafe_offset=row * seg.cols + col] = src[
                            unsafe_offset=row * seg.pitch + col
                        ]
        else:
            comptime for k in range(COPY_ROWS_TILE // COPY_ROWS_THREADS):
                var i = (
                    local_tile * COPY_ROWS_TILE + lane + k * COPY_ROWS_THREADS
                )
                if i < seg.rows * seg.cols:
                    var row = i // seg.cols
                    var col = i % seg.cols
                    dst[unsafe_offset=i] = src[
                        unsafe_offset=row * seg.pitch + col
                    ]
        tile += Int(grid_dim.x)


def copy_batched_rows(
    srcs: List[Int],
    dsts: List[Int],
    rows: List[Int],
    cols: List[Int],
    pitches: List[Int],
    ctx: DeviceContext,
) raises:
    var index = 0
    while index < len(cols):
        var segs = InlineArray[CopyRect, COPY_ROWS_CAP](
            fill=CopyRect(0, 0, 0, 0, 0, 0)
        )
        var count = 0
        var tiles = 0
        var small = True
        var max_rows = 0
        while index < len(cols) and count < COPY_ROWS_CAP:
            var r = rows[index]
            var c = cols[index]
            if r > 0 and c > 0:
                small = small and r <= COPY_SMALL_ROWS and c <= COPY_ROWS_TILE
                max_rows = max(max_rows, r)
                tiles += r * ceildiv(
                    c, COPY_ROWS_TILE
                ) if c > COPY_ROWS_TILE else ceildiv(r * c, COPY_ROWS_TILE)
                segs[count] = CopyRect(
                    srcs[index], dsts[index], r, c, pitches[index], tiles
                )
                count += 1
            index += 1
        if tiles > 0 and small:
            _enqueue_cached[_copy_rows_small_kernel](
                ctx,
                count,
                max_rows,
                1,
                COPY_ROWS_THREADS,
                segs,
            )
        elif tiles > 0:
            _enqueue_cached[_copy_rows_kernel](
                ctx,
                min(tiles, 1 << 22),
                1,
                1,
                COPY_ROWS_THREADS,
                segs,
                Int64(count),
                Int64(tiles),
            )
