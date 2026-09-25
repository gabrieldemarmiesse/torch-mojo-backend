"""Batched rectangle copies, converting between any two copy dtypes.

One launch moves up to COPY_BATCH_CAP rectangles: `rows` rows of `cols`
elements, `src_pitch` / `dst_pitch` elements apart. That one shape is
`_foreach_copy_` (one row per pair), `cat.out` (every input's rows into
strided output rows) and `split_with_sizes_copy` (strided input rows into
every output). Descriptors are passed by value, with no device-side scratch
allocation. The kernel is specialized on the (source, destination) dtype
pair; bool travels as its uint8 storage. Tiles measured on H100.
"""
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.collections import Array
from max.gpu import block_idx, grid_dim, thread_idx
from std.math import ceildiv
from std.sys import size_of
from max.gpu.host import DeviceContext
from tmb.kernels.common.dtype_convert import convert, storage_dtype
from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr
from tmb.kernels.data_movement.copy_segments import (
    CopyTileSegment,
    copy_segment_index,
)

# 80 * 48 = 3840 parameter bytes, inside a conservative 4 KiB budget: one
# launch covers an 80-piece split, which two launches measured twice as slow.
comptime COPY_BATCH_CAP = 80
comptime COPY_BATCH_THREADS = 256
comptime COPY_BATCH_MAX_BLOCKS = 1 << 22
# Rows per rectangle up to which a launch of short rows takes the small
# kernel, one block per row (the 16-bit split kernel's measured regime).
comptime COPY_SMALL_ROWS = 8
# Largest row pitch, in elements, a rectangle may have.
comptime COPY_BATCH_MAX_PITCH = (1 << 31) - 1


struct CopyRect(CopyTileSegment, DevicePassable):
    comptime device_type: AnyType = Self
    var src: Int
    var dst: Int
    var cols: Int
    # 32 bits hold these (callers keep pitches below 2^31, see
    # COPY_BATCH_MAX_PITCH): a launch past 2^31 rows or tiles would move more
    # than 2^41 elements.
    var src_pitch: Int32
    var dst_pitch: Int32
    var rows: Int32
    var head: Int32  # scalar elements peeled so both row starts are aligned
    var row_tiles: Int32  # tiles across one row
    var tile_end: Int32  # exclusive prefix sum of tiles over the launch

    def __init__(
        out self,
        src: Int,
        dst: Int,
        rows: Int,
        cols: Int,
        src_pitch: Int,
        dst_pitch: Int,
    ):
        self.src = src
        self.dst = dst
        self.rows = Int32(rows)
        self.cols = cols
        self.src_pitch = Int32(src_pitch)
        self.dst_pitch = Int32(dst_pitch)
        self.head = 0
        self.row_tiles = 0
        self.tile_end = 0

    @always_inline
    def tile_limit(self) -> Int:
        return Int(self.tile_end)

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "CopyRect"


@always_inline
def _vec[src: DType, dst: DType]() -> Int:
    """Elements per access: 16 bytes of the wider operand."""
    return 16 // max(
        size_of[storage_dtype[src]()](), size_of[storage_dtype[dst]()]()
    )


@always_inline
def _items[S: DType, V: Int]() -> Int:
    """Accesses per thread per tile. Four at every vector width, and at
    least 16 source bytes so the scalar route still amortizes the segment
    lookup."""
    return max(4, 16 // (V * size_of[S]()))


@always_inline
def _tile[S: DType, V: Int]() -> Int:
    """Elements of one row a block moves per tile with `V`-wide accesses."""
    return COPY_BATCH_THREADS * _items[S, V]() * V


@always_inline
def _head[S: DType, D: DType, V: Int](rect: CopyRect) -> Int:
    """Scalar elements to peel so every row of both sides starts its vector
    run on a `V`-access boundary, or -1 when they never reach one together.
    """
    comptime SB = V * size_of[S]()
    var head = min(((SB - rect.src % SB) % SB) // size_of[S](), rect.cols)
    if (rect.dst + head * size_of[D]()) % (V * size_of[D]()) != 0:
        return -1
    if Int(rect.rows) > 1 and (
        Int(rect.src_pitch) % V != 0 or Int(rect.dst_pitch) % V != 0
    ):
        return -1
    return head


@__name(t"copy_batched_rect_{src}_{dst}_v{V}")
def _copy_rect_kernel[
    src: DType, dst: DType, V: Int
](rects: Array[CopyRect, COPY_BATCH_CAP], count_arg: Int64, tiles_arg: Int64):
    """`V`-wide accesses after each row's `head` scalar elements, one tile
    of one row per block iteration. One width per kernel, and no loop over
    rows inside a tile: either measured at twice the registers and half the
    occupancy."""
    comptime S = storage_dtype[src]()
    comptime D = storage_dtype[dst]()
    comptime ITEMS = _items[S, V]()
    comptime TILE = _tile[S, V]()
    var lane = Int(thread_idx.x)
    var tile = Int(block_idx.x)
    while tile < Int(tiles_arg):
        var lo = copy_segment_index(rects, Int(count_arg), tile)
        var rect = rects[lo]
        var local = tile - (0 if lo == 0 else Int(rects[lo - 1].tile_end))
        # Tile counts fit 32 bits (the grid is capped at 2^22 blocks and a
        # launch at a few billion tiles); a 64-bit divide costs registers.
        var chunk = local
        var src_addr = rect.src
        var dst_addr = rect.dst
        if rect.rows > 1:
            var row = Int(UInt32(local) // UInt32(rect.row_tiles))
            chunk = local - row * Int(rect.row_tiles)
            src_addr += row * Int(rect.src_pitch) * size_of[S]()
            dst_addr += row * Int(rect.dst_pitch) * size_of[D]()
        var start = chunk * TILE
        var sp = _make_ptr[S](src_addr)
        var dp = _make_ptr[D](dst_addr)
        var head = Int(rect.head)
        if start == 0 and lane < head:
            dp[unsafe_offset=lane] = convert[src, dst, S, D, 1](
                sp[unsafe_offset=lane]
            )
        comptime for k in range(ITEMS):
            var i = head + start + (lane + k * COPY_BATCH_THREADS) * V
            if i + V <= rect.cols:
                dp.unsafe_store[width=V, alignment=V * size_of[D]()](
                    i,
                    convert[src, dst, S, D, V](
                        sp.unsafe_load[width=V, alignment=V * size_of[S]()](i)
                    ),
                )
            elif i < rect.cols:
                for j in range(i, rect.cols):
                    dp[unsafe_offset=j] = convert[src, dst, S, D, 1](
                        sp[unsafe_offset=j]
                    )
        tile += Int(grid_dim.x)


@__name(t"copy_batched_rect_{src}_{dst}_small")
def _copy_rect_small_kernel[
    src: DType, dst: DType
](rects: Array[CopyRect, COPY_BATCH_CAP]):
    """One block per rectangle row, scalar: for launches whose rows all fit
    in one tile, where the tile lookup and vector route measured slower."""
    comptime S = storage_dtype[src]()
    comptime D = storage_dtype[dst]()
    var rect = rects[Int(block_idx.x)]
    var row = Int(block_idx.y)
    if row < Int(rect.rows):
        var src_addr = rect.src
        var dst_addr = rect.dst
        if row > 0:
            src_addr += row * Int(rect.src_pitch) * size_of[S]()
            dst_addr += row * Int(rect.dst_pitch) * size_of[D]()
        var sp = _make_ptr[S](src_addr)
        var dp = _make_ptr[D](dst_addr)
        var i = Int(thread_idx.x)
        while i < rect.cols:
            dp[unsafe_offset=i] = convert[src, dst, S, D, 1](
                sp[unsafe_offset=i]
            )
            i += COPY_BATCH_THREADS


@__name(t"copy_batched_rect_{src}_{dst}_narrow")
def _copy_rect_narrow_kernel[
    src: DType, dst: DType
](rects: Array[CopyRect, COPY_BATCH_CAP], count_arg: Int64, tiles_arg: Int64):
    """Scalar copies over each rectangle's rows x cols as one run, for many
    rows narrower than a block: a tile per row would idle nearly every
    thread (65536 rows of 1-2 elements measured 100x slower that way)."""
    comptime S = storage_dtype[src]()
    comptime D = storage_dtype[dst]()
    comptime TILE = _tile[S, 1]()
    var lane = Int(thread_idx.x)
    var tile = Int(block_idx.x)
    while tile < Int(tiles_arg):
        var lo = copy_segment_index(rects, Int(count_arg), tile)
        var rect = rects[lo]
        var local = tile - (0 if lo == 0 else Int(rects[lo - 1].tile_end))
        var total = Int(rect.rows) * rect.cols
        var sp = _make_ptr[S](rect.src)
        var dp = _make_ptr[D](rect.dst)
        comptime for k in range(TILE // COPY_BATCH_THREADS):
            var i = local * TILE + lane + k * COPY_BATCH_THREADS
            if i < total:
                var row = Int(UInt32(i) // UInt32(rect.cols))
                var col = i - row * rect.cols
                dp[unsafe_offset=row * Int(rect.dst_pitch) + col] = convert[
                    src, dst, S, D, 1
                ](sp[unsafe_offset=row * Int(rect.src_pitch) + col])
        tile += Int(grid_dim.x)


def copy_batched[
    src: DType, dst: DType
](
    srcs: List[Int],
    dsts: List[Int],
    rows: List[Int],
    cols: List[Int],
    src_pitches: List[Int],
    dst_pitches: List[Int],
    ctx: DeviceContext,
) raises:
    """Copy rectangle i (`rows[i]` x `cols[i]` elements, rows `src_pitches[i]`
    / `dst_pitches[i]` elements apart) from `srcs[i]` to `dsts[i]` for every
    i, in ceil(len / COPY_BATCH_CAP) launches. The spans must not overlap."""
    comptime S = storage_dtype[src]()
    comptime D = storage_dtype[dst]()
    comptime VEC = _vec[src, dst]()
    var index = 0
    while index < len(cols):
        var rects = Array[CopyRect, COPY_BATCH_CAP](
            fill=CopyRect(0, 0, 0, 0, 0, 0)
        )
        var count = 0
        var largest = 0
        var max_rows = 0
        var max_total = 0
        # The launch takes the widest access every one of its rectangles allows.
        var width = VEC
        while index < len(cols) and count < COPY_BATCH_CAP:
            if rows[index] > 0 and cols[index] > 0:
                # A one-row rectangle never reads its pitches, which need
                # not fit their 32 bits (a 1-D span past 2^31 elements).
                var one_row = rows[index] == 1
                var rect = CopyRect(
                    srcs[index],
                    dsts[index],
                    rows[index],
                    cols[index],
                    0 if one_row else src_pitches[index],
                    0 if one_row else dst_pitches[index],
                )
                rects[count] = rect
                count += 1
                largest = max(largest, rect.cols)
                max_rows = max(max_rows, Int(rect.rows))
                max_total = max(max_total, Int(rect.rows) * rect.cols)
                comptime for step in range(5):  # VEC <= 16: 16, 8, 4, 2
                    comptime V = VEC >> step
                    comptime if V > 1:
                        if width == V and _head[S, D, V](rect) < 0:
                            width = V // 2
            index += 1
        if count == 0:
            continue
        if (
            max_rows > COPY_SMALL_ROWS
            and largest < COPY_BATCH_THREADS
            and max_total <= COPY_BATCH_MAX_PITCH
        ):
            comptime TILE = _tile[S, 1]()
            var tiles = 0
            for i in range(count):
                tiles += ceildiv(Int(rects[i].rows) * rects[i].cols, TILE)
                rects[i].tile_end = Int32(tiles)
            _enqueue_cached[_copy_rect_narrow_kernel[src, dst]](
                ctx,
                min(tiles, COPY_BATCH_MAX_BLOCKS),
                1,
                1,
                COPY_BATCH_THREADS,
                rects,
                Int64(count),
                Int64(tiles),
            )
            continue
        if largest <= _tile[S, VEC]() and max_rows <= COPY_SMALL_ROWS:
            _enqueue_cached[_copy_rect_small_kernel[src, dst]](
                ctx,
                count,
                max_rows,
                1,
                COPY_BATCH_THREADS,
                rects,
            )
            continue
        comptime for step in range(5):
            comptime V = VEC >> step
            comptime if V >= 1:
                if width == V:
                    comptime TILE = _tile[S, V]()
                    var tiles = 0
                    for i in range(count):
                        rects[i].head = Int32(_head[S, D, V](rects[i]))
                        var row_tiles = ceildiv(rects[i].cols, TILE)
                        rects[i].row_tiles = Int32(row_tiles)
                        tiles += Int(rects[i].rows) * row_tiles
                        rects[i].tile_end = Int32(tiles)
                    _enqueue_cached[_copy_rect_kernel[src, dst, V]](
                        ctx,
                        min(tiles, COPY_BATCH_MAX_BLOCKS),
                        1,
                        1,
                        COPY_BATCH_THREADS,
                        rects,
                        Int64(count),
                        Int64(tiles),
                    )
