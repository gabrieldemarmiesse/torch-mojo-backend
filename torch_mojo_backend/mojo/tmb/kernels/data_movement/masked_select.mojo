"""Stream compaction for `masked_select`: a per-tile mask count, then a
per-tile scan that writes the selected elements in row-major order.

The output size depends on the mask's contents, so the op runs in two
launches with a host step between them: `MaskedSelectCount` writes one
count per MS_TILE elements, the host turns them into exclusive offsets (and
the total, which sizes the output), and `MaskedSelectCompact` rescans each
tile and writes its selected elements from the tile's offset on. Both
inputs arrive contiguous and of one shape (the op broadcasts them first).
Elements move as raw bits, so one kernel per element width serves every
dtype; the mask is bool's uint8 storage.
"""
from std.gpu import block_idx, grid_dim, thread_idx
from std.memory import AddressSpace, stack_allocation
from max.gpu.sync import barrier
from tmb.kernels.common.op_utils import (
    Argv,
    _device_sm_count,
    _enqueue_cached,
    _make_ptr,
    _raw_ctx,
    _raw_int,
)

comptime MS_THREADS = 256
comptime MS_ITEMS = 4  # consecutive elements per thread
comptime MS_TILE = MS_THREADS * MS_ITEMS
# Resident tiles per SM before the grid strides; portable, not fitted.
comptime MS_BLOCKS_PER_SM = 8


@always_inline
def _thread_count(
    mask: Pointer[UInt8, MutAnyOrigin], base: Int, n: Int
) -> Int32:
    var c = Int32(0)
    for j in range(MS_ITEMS):
        var i = base + j
        if i < n and mask[unsafe_offset=i] != 0:
            c += 1
    return c


@__name("masked_select_count_t1024")
def _count(
    mask: Pointer[UInt8, MutAnyOrigin],
    counts: Pointer[Int64, MutAnyOrigin],
    n64: Int64,
    tiles64: Int64,
):
    var n = Int(n64)
    var tiles = Int(tiles64)
    var sums = stack_allocation[
        MS_THREADS, Int32, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var tile = Int(block_idx.x)
    while tile < tiles:
        sums[unsafe_offset=tid] = _thread_count(
            mask, tile * MS_TILE + tid * MS_ITEMS, n
        )
        barrier()
        var s = MS_THREADS // 2
        while s > 0:
            if tid < s:
                sums[unsafe_offset=tid] += sums[unsafe_offset=tid + s]
            barrier()
            s //= 2
        if tid == 0:
            counts[unsafe_offset=tile] = Int64(sums[unsafe_offset=0])
        barrier()
        tile += Int(grid_dim.x)


@__name("masked_select_compact_t1024_" + String(dt))
def _compact[
    dt: DType
](
    src: Pointer[Scalar[dt], MutAnyOrigin],
    mask: Pointer[UInt8, MutAnyOrigin],
    offsets: Pointer[Int64, MutAnyOrigin],
    dst: Pointer[Scalar[dt], MutAnyOrigin],
    n64: Int64,
    tiles64: Int64,
):
    var n = Int(n64)
    var tiles = Int(tiles64)
    var scan = stack_allocation[
        MS_THREADS, Int32, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var tile = Int(block_idx.x)
    while tile < tiles:
        var base = tile * MS_TILE + tid * MS_ITEMS
        var mine = _thread_count(mask, base, n)
        scan[unsafe_offset=tid] = mine
        barrier()
        # Hillis-Steele inclusive scan over the block's thread counts.
        var step = 1
        while step < MS_THREADS:
            var left = Int32(0)
            if tid >= step:
                left = scan[unsafe_offset=tid - step]
            barrier()
            scan[unsafe_offset=tid] += left
            barrier()
            step *= 2
        var pos = Int(offsets[unsafe_offset=tile]) + Int(
            scan[unsafe_offset=tid] - mine
        )
        for j in range(MS_ITEMS):
            var i = base + j
            if i < n and mask[unsafe_offset=i] != 0:
                dst[unsafe_offset=pos] = src[unsafe_offset=i]
                pos += 1
        barrier()
        tile += Int(grid_dim.x)


def _tiles(n: Int, handed: Int) raises -> Int:
    """The tile count of `n` elements, checked against the op's own: the op
    sized the offsets buffer with it, and its tile constant mirrors
    MS_TILE."""
    var tiles = (n + MS_TILE - 1) // MS_TILE
    if tiles != handed:
        raise Error("MaskedSelect: the op's tile size does not match MS_TILE")
    return tiles


def _blocks(tiles: Int, sms: Int) -> Int:
    return max(1, min(tiles, max(1, sms) * MS_BLOCKS_PER_SM))


def masked_select_count_dispatcher(argv: Argv, argc: Int) raises:
    """Slots: mask, counts, n, tiles, ctx."""
    if argc != 5:
        raise Error("MaskedSelectCount expects five slots")
    var mask = _make_ptr[DType.uint8](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var counts = _make_ptr[DType.int64](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var n = _raw_int(argv[unsafe_offset=2])
    var tiles = _tiles(n, _raw_int(argv[unsafe_offset=3]))
    var ctx = _raw_ctx(argv[unsafe_offset=4])
    _enqueue_cached[_count](
        ctx,
        _blocks(tiles, _device_sm_count(ctx)),
        1,
        1,
        MS_THREADS,
        mask,
        counts,
        Int64(n),
        Int64(tiles),
    )


def _launch_compact[dt: DType](argv: Argv, n: Int, tiles: Int) raises:
    var src = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=0])
    ).as_unsafe_any_origin()
    var mask = _make_ptr[DType.uint8](
        _raw_int(argv[unsafe_offset=1])
    ).as_unsafe_any_origin()
    var offsets = _make_ptr[DType.int64](
        _raw_int(argv[unsafe_offset=2])
    ).as_unsafe_any_origin()
    var dst = _make_ptr[dt](
        _raw_int(argv[unsafe_offset=3])
    ).as_unsafe_any_origin()
    var ctx = _raw_ctx(argv[unsafe_offset=7])
    _enqueue_cached[_compact[dt]](
        ctx,
        _blocks(tiles, _device_sm_count(ctx)),
        1,
        1,
        MS_THREADS,
        src,
        mask,
        offsets,
        dst,
        Int64(n),
        Int64(tiles),
    )


def masked_select_compact_dispatcher(argv: Argv, argc: Int) raises:
    """Slots: src, mask, offsets, out, n, tiles, itemsize, ctx."""
    if argc != 8:
        raise Error("MaskedSelectCompact expects eight slots")
    var n = _raw_int(argv[unsafe_offset=4])
    var tiles = _tiles(n, _raw_int(argv[unsafe_offset=5]))
    var itemsize = _raw_int(argv[unsafe_offset=6])
    if itemsize == 1:
        _launch_compact[DType.uint8](argv, n, tiles)
    elif itemsize == 2:
        _launch_compact[DType.uint16](argv, n, tiles)
    elif itemsize == 4:
        _launch_compact[DType.uint32](argv, n, tiles)
    elif itemsize == 8:
        _launch_compact[DType.uint64](argv, n, tiles)
    else:
        raise Error("MaskedSelectCompact: unsupported element size")
