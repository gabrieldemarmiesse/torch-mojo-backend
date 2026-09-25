# ===----------------------------------------------------------------------=== #
# Segmented (per-row) sort and selection for aten::sort and aten::topk.
#
# Everything here sorts (key, original-index) PAIRS under one total order:
# the key is an order-preserving unsigned image of the value, and the index
# breaks every tie.  That single choice buys three things at once -- the
# order is a strict total order, so a bitonic network (which is not a stable
# algorithm) still produces exactly the STABLE answer ATen's
# `sort(stable=True)` promises; ties are broken the same way on every GPU;
# and `topk` composes, because the top-k of a total order can be taken
# tile-by-tile and re-merged.
#
# Three routes, chosen by the op (tmb/ops/reductions.mojo, which also sizes
# the workspace):
#
#   0  one row fits one tile: a single block sorts the whole row in shared
#      memory.  One launch (plus the gather).
#   1  topk tournament: every tile emits its own top-k, and one final block
#      sorts the concatenation.  Two launches, and the row is read once --
#      the regime `generate()` lives in (k tiny, row = a vocabulary).
#   2  full sort of a long row: tile sort, then the bitonic stages that
#      cross tiles, with every step below the tile size fused back into
#      shared memory.
#
# Every size is a runtime argument; the only compile-time numbers are the
# tile (a shared-memory budget) and the block width.
#
# `SortSelect` runs the same ascending routes and then, instead of gathering
# the first k, reads ONE position per row (`_select_kernel`): kthvalue's
# k-th smallest, median's lower middle, nanmedian's lower middle of the
# non-NaN prefix.  The total order is also what makes those answers ATen's:
# ties resolve to the lowest index of the tied value (CPU's
# `nth_element` comparator), and every NaN sorts last, lowest index first,
# so the first NaN of a row is one binary search over the sorted keys away.
# ===----------------------------------------------------------------------=== #

from std.bit import next_power_of_two
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_dim,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.memory import AddressSpace, bitcast, stack_allocation
from std.sys import bit_width_of
from std.sys.info import has_accelerator
from std.utils.static_tuple import StaticTuple
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier

from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    GS_THREADS,
    _enqueue_cached,
    _gs_blocks,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _spec_dispatcher13,
    _spec_dispatcher14,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)


# bool rides the uint8 specialization (the op passes uint8 for it).
comptime SORT_DTYPES: List[DType] = [
    DType.float32,
    DType.float64,
    DType.bfloat16,
    DType.float16,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
]

# The key is the smallest unsigned type that holds the value's bit pattern.
comptime _KEY_DTYPE[dtype: DType] = (
    DType.uint64 if bit_width_of[dtype]() == 64 else DType.uint32
)

# Elements one block sorts in shared memory.  Sized so key+index stay within
# 32 KiB -- CUDA's static limit is 48 KiB but Metal's is 32 KiB, and this
# kernel is one source for every backend.  Mirrored by SORT_TILE_32/64 in
# tmb/ops/reductions.mojo, which picks the route and sizes the workspace.
comptime _TILE[KT: DType] = 2048 if KT == DType.uint64 else 4096

# Threads per block for the sorting kernels.  A bitonic pass is
# barrier-bound, not arithmetic-bound: 1024 threads cut a 4096-element tile
# from 8 iterations per step to 2, and the 32 KiB tile still leaves two
# blocks per SM resident.  (Measured on H100 PCIe; the 256-thread grid this
# replaced was ~2x slower on every recorded shape.)
comptime SORT_THREADS = 1024


@always_inline
def _bitonic_partner(pair: Int, stride: Int) -> Int:
    """The lower element of bitonic pair number `pair` at this stride.

    Enumerating PAIRS rather than elements is what keeps every lane busy: an
    `i ^ j; if partner > i` formulation leaves half the threads idle in every
    step, and there are ~78 steps in a 4096-element tile.
    """
    return ((pair & ~(stride - 1)) << 1) | (pair & (stride - 1))


# Padding lanes carry an index no real element can have, which is what keeps
# them after every real element whatever their key compares to.
comptime _SENTINEL_IDX = Int32(2147483647)


@always_inline
def _float_key[
    dtype: DType,
    KT: DType,
    UT: DType,
    abs_mask: Int,
    inf_bits: Int,
    sign_shift: Int,
](v: Scalar[dtype]) -> Scalar[KT]:
    """The monotone float-to-unsigned map, with ATen's two exceptions.

    Every NaN takes the maximum key whatever its sign says (ATen orders NaN
    above every number, but a negative NaN's bits sit below -inf), and
    negative zero takes positive zero's key (the two compare equal, so only
    the index may separate them).

    Both are recognized from the bit pattern, never from `v != v` or `v == 0`:
    those are float predicates, and a fast-math GPU compilation is free to
    fold the NaN test to a constant false -- which it did, and which left
    negative NaN sorting below -inf.
    """
    var raw = bitcast[UT, 1](v)
    var magnitude = raw & Scalar[UT](abs_mask)
    if magnitude.gt(Scalar[UT](inf_bits))[0]:
        return ~Scalar[KT](0)
    var bits = Scalar[UT](0) if magnitude.eq(0)[0] else raw
    var shift = Scalar[UT](sign_shift)
    return (bits ^ ((-(bits >> shift)) | (Scalar[UT](1) << shift))).cast[KT]()


@always_inline
def _sort_key[dtype: DType, KT: DType](v: Scalar[dtype]) -> Scalar[KT]:
    """The order-preserving unsigned image of `v`: sorting the keys sorts the
    values, so one comparator serves every dtype."""
    comptime if dtype == DType.float64:
        return _float_key[
            dtype,
            KT,
            DType.uint64,
            0x7FFF_FFFF_FFFF_FFFF,
            0x7FF0_0000_0000_0000,
            63,
        ](v)
    elif dtype == DType.float32:
        return _float_key[
            dtype, KT, DType.uint32, 0x7FFF_FFFF, 0x7F80_0000, 31
        ](v)
    elif dtype == DType.float16:
        return _float_key[dtype, KT, DType.uint16, 0x7FFF, 0x7C00, 15](v)
    elif dtype == DType.bfloat16:
        # bfloat16 keeps float32's exponent range, so its infinity pattern is
        # not float16's.
        return _float_key[dtype, KT, DType.uint16, 0x7FFF, 0x7F80, 15](v)
    elif dtype == DType.int64:
        return (v.cast[DType.uint64]() ^ (UInt64(1) << UInt64(63))).cast[KT]()
    elif dtype == DType.int32:
        return (v.cast[DType.uint32]() ^ UInt32(0x8000_0000)).cast[KT]()
    elif dtype == DType.int16:
        return (v.cast[DType.uint16]() ^ UInt16(0x8000)).cast[KT]()
    elif dtype == DType.int8:
        return (v.cast[DType.uint8]() ^ UInt8(0x80)).cast[KT]()
    else:
        return v.cast[KT]()


@always_inline
def _sentinel_key[KT: DType, descending: Bool]() -> Scalar[KT]:
    """The key that loses to every real key in this direction."""
    comptime if descending:
        return Scalar[KT](0)
    else:
        return ~Scalar[KT](0)


@always_inline
def _after[
    KT: DType, descending: Bool
](ka: Scalar[KT], ia: Int32, kb: Scalar[KT], ib: Int32) -> Bool:
    """Whether `(ka, ia)` belongs after `(kb, ib)` in the output order."""
    if ka.ne(kb)[0]:
        comptime if descending:
            return ka.lt(kb)[0]
        else:
            return ka.gt(kb)[0]
    return ia.gt(ib)[0]


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SORT_THREADS))
)
@__name(t"segsort_tile_{dtype}_{descending}_pairs{from_pairs}_t{SORT_THREADS}")
def _sort_tile_kernel[
    dtype: DType, KT: DType, descending: Bool, from_pairs: Bool
](
    out_keys: Pointer[Scalar[KT], MutAnyOrigin],
    out_idx: Pointer[Scalar[DType.int32], MutAnyOrigin],
    in_vals: Pointer[Scalar[dtype], ImmutAnyOrigin],
    in_keys: Pointer[Scalar[KT], ImmutAnyOrigin],
    in_idx: Pointer[Scalar[DType.int32], ImmutAnyOrigin],
    in_row_len_arg: Int64,
    out_row_len_arg: Int64,
    emit_arg: Int64,
    span_arg: Int64,
    dir_global_arg: Int64,
):
    """Sort one `span`-wide tile of one row in shared memory, and emit its
    first `emit` pairs.

    `from_pairs` reads already-keyed pairs (the tournament's final round);
    otherwise the values are keyed on load and indexed by position.
    `dir_global` takes each step's direction from the element's position in
    the whole row rather than in the tile -- what the full sort needs so the
    sorted tiles alternate direction and form bitonic sequences.
    """
    comptime TILE = _TILE[KT]
    var in_row_len = Int(in_row_len_arg)
    var out_row_len = Int(out_row_len_arg)
    var emit = Int(emit_arg)
    var span = Int(span_arg)
    var dir_global = Int(dir_global_arg) != 0

    var s_key = stack_allocation[TILE, KT, address_space=AddressSpace.SHARED]()
    var s_idx = stack_allocation[
        TILE, DType.int32, address_space=AddressSpace.SHARED
    ]()

    var row = Int(block_idx.y)
    var tile = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var base = tile * span

    var load = tid
    while load < span:
        var pos = base + load
        if pos < in_row_len:
            comptime if from_pairs:
                s_key[unsafe_offset=load] = in_keys[
                    unsafe_offset=row * in_row_len + pos
                ]
                s_idx[unsafe_offset=load] = in_idx[
                    unsafe_offset=row * in_row_len + pos
                ]
            else:
                s_key[unsafe_offset=load] = _sort_key[dtype, KT](
                    in_vals[unsafe_offset=row * in_row_len + pos]
                )
                s_idx[unsafe_offset=load] = Int32(pos)
        else:
            s_key[unsafe_offset=load] = _sentinel_key[KT, descending]()
            s_idx[unsafe_offset=load] = _SENTINEL_IDX
        load += SORT_THREADS
    barrier()

    var pairs = span >> 1
    var k = 2
    while k <= span:
        var j = k >> 1
        while j > 0:
            var p = tid
            while p < pairs:
                var t = _bitonic_partner(p, j)
                var partner = t + j
                var gi = (base + t) if dir_global else t
                var ki = s_key[unsafe_offset=t]
                var ii = s_idx[unsafe_offset=t]
                var kp = s_key[unsafe_offset=partner]
                var ip = s_idx[unsafe_offset=partner]
                var swap_needed = _after[KT, descending](ki, ii, kp, ip) if (
                    gi & k
                ) == 0 else _after[KT, descending](kp, ip, ki, ii)
                if swap_needed:
                    s_key[unsafe_offset=t] = kp
                    s_idx[unsafe_offset=t] = ip
                    s_key[unsafe_offset=partner] = ki
                    s_idx[unsafe_offset=partner] = ii
                p += SORT_THREADS
            barrier()
            j >>= 1
        k <<= 1

    var store = tid
    while store < emit:
        var dst = row * out_row_len + tile * emit + store
        out_keys[unsafe_offset=dst] = s_key[unsafe_offset=store]
        out_idx[unsafe_offset=dst] = s_idx[unsafe_offset=store]
        store += SORT_THREADS


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SORT_THREADS))
)
@__name(t"segsort_bitonic_step_{KT}_{descending}_t{SORT_THREADS}")
def _bitonic_step_kernel[
    KT: DType, descending: Bool
](
    keys: Pointer[Scalar[KT], MutAnyOrigin],
    idx: Pointer[Scalar[DType.int32], MutAnyOrigin],
    row_pow2_arg: Int64,
    j_arg: Int64,
    k_arg: Int64,
):
    """One bitonic compare-exchange step whose partners cross tiles."""
    var row_pow2 = Int(row_pow2_arg)
    var j = Int(j_arg)
    var k = Int(k_arg)
    var base = Int(block_idx.y) * row_pow2
    var pairs = row_pow2 >> 1
    var p = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while p < pairs:
        var i = _bitonic_partner(p, j)
        var partner = i + j
        var ki = keys[unsafe_offset=base + i]
        var ii = idx[unsafe_offset=base + i]
        var kp = keys[unsafe_offset=base + partner]
        var ip = idx[unsafe_offset=base + partner]
        var swap_needed = _after[KT, descending](ki, ii, kp, ip) if (
            i & k
        ) == 0 else _after[KT, descending](kp, ip, ki, ii)
        if swap_needed:
            keys[unsafe_offset=base + i] = kp
            idx[unsafe_offset=base + i] = ip
            keys[unsafe_offset=base + partner] = ki
            idx[unsafe_offset=base + partner] = ii
        p += stride


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SORT_THREADS))
)
@__name(t"segsort_merge_local_{KT}_{descending}_t{SORT_THREADS}")
def _bitonic_merge_local_kernel[
    KT: DType, descending: Bool
](
    keys: Pointer[Scalar[KT], MutAnyOrigin],
    idx: Pointer[Scalar[DType.int32], MutAnyOrigin],
    row_pow2_arg: Int64,
    k_arg: Int64,
):
    """Every bitonic step below the tile size, fused into shared memory."""
    comptime TILE = _TILE[KT]
    var row_pow2 = Int(row_pow2_arg)
    var k = Int(k_arg)

    var s_key = stack_allocation[TILE, KT, address_space=AddressSpace.SHARED]()
    var s_idx = stack_allocation[
        TILE, DType.int32, address_space=AddressSpace.SHARED
    ]()

    var tile = Int(block_idx.x)
    var base = Int(block_idx.y) * row_pow2 + tile * TILE
    var tid = Int(thread_idx.x)

    var load = tid
    while load < TILE:
        s_key[unsafe_offset=load] = keys[unsafe_offset=base + load]
        s_idx[unsafe_offset=load] = idx[unsafe_offset=base + load]
        load += SORT_THREADS
    barrier()

    comptime PAIRS = TILE >> 1
    var j = TILE >> 1
    while j > 0:
        var p = tid
        while p < PAIRS:
            var t = _bitonic_partner(p, j)
            var partner = t + j
            var gi = tile * TILE + t
            var ki = s_key[unsafe_offset=t]
            var ii = s_idx[unsafe_offset=t]
            var kp = s_key[unsafe_offset=partner]
            var ip = s_idx[unsafe_offset=partner]
            var swap_needed = _after[KT, descending](ki, ii, kp, ip) if (
                gi & k
            ) == 0 else _after[KT, descending](kp, ip, ki, ii)
            if swap_needed:
                s_key[unsafe_offset=t] = kp
                s_idx[unsafe_offset=t] = ip
                s_key[unsafe_offset=partner] = ki
                s_idx[unsafe_offset=partner] = ii
            p += SORT_THREADS
        barrier()
        j >>= 1

    var store = tid
    while store < TILE:
        keys[unsafe_offset=base + store] = s_key[unsafe_offset=store]
        idx[unsafe_offset=base + store] = s_idx[unsafe_offset=store]
        store += SORT_THREADS


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
@__name(t"segsort_gather_{dtype}_t{GS_THREADS}")
def _gather_kernel[
    dtype: DType
](
    out_vals: Pointer[Scalar[dtype], MutAnyOrigin],
    out_idx: Pointer[Scalar[DType.int64], MutAnyOrigin],
    in_vals: Pointer[Scalar[dtype], ImmutAnyOrigin],
    sorted_idx: Pointer[Scalar[DType.int32], ImmutAnyOrigin],
    n_arg: Int64,
    pad_arg: Int64,
    out_k_arg: Int64,
):
    """Materialize the answer: values by gather, indices widened to int64."""
    var n = Int(n_arg)
    var pad = Int(pad_arg)
    var out_k = Int(out_k_arg)
    var row = Int(block_idx.y)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while i < out_k:
        var src = Int(sorted_idx[unsafe_offset=row * pad + i])
        out_vals[unsafe_offset=row * out_k + i] = in_vals[
            unsafe_offset=row * n + src
        ]
        out_idx[unsafe_offset=row * out_k + i] = Int64(src)
        i += stride


# What `_select_kernel` reads from each sorted row.
comptime SELECT_KTH = 0  # the given position
comptime SELECT_MEDIAN = 1  # the given position, or the first NaN if any
comptime SELECT_NANMEDIAN = 2  # the lower middle of the non-NaN prefix


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GS_THREADS))
)
@__name(t"segsort_select_{dtype}_m{mode}_t{GS_THREADS}")
def _select_kernel[
    dtype: DType, KT: DType, mode: Int
](
    out_vals: Pointer[Scalar[dtype], MutAnyOrigin],
    out_idx: Pointer[Scalar[DType.int64], MutAnyOrigin],
    in_vals: Pointer[Scalar[dtype], ImmutAnyOrigin],
    keys: Pointer[Scalar[KT], ImmutAnyOrigin],
    sorted_idx: Pointer[Scalar[DType.int32], ImmutAnyOrigin],
    rows_arg: Int64,
    n_arg: Int64,
    pad_arg: Int64,
    pos_arg: Int64,
):
    """One (value, int64 index) per ascending-sorted row.

    Only a float row can hold the all-ones key below `n` (`_float_key` gives
    it to NaN alone; the padding lanes sit at `n` and after), so the NaN
    modes are compiled for float dtypes only and the others read `pos`.
    """
    var rows = Int(rows_arg)
    var n = Int(n_arg)
    var pad = Int(pad_arg)
    var pos = Int(pos_arg)
    var row = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * Int(block_dim.x)
    while row < rows:
        var base = row * pad
        var p = pos
        comptime if mode != SELECT_KTH and dtype.is_floating_point():
            var lo = 0
            var hi = n
            while lo < hi:
                var mid = (lo + hi) >> 1
                if keys[unsafe_offset=base + mid].eq(~Scalar[KT](0))[0]:
                    hi = mid
                else:
                    lo = mid + 1
            comptime if mode == SELECT_MEDIAN:
                if lo < n:
                    p = lo
            else:
                p = (lo - 1) >> 1 if lo > 0 else 0
        var src = Int(sorted_idx[unsafe_offset=base + p])
        out_vals[unsafe_offset=row] = in_vals[unsafe_offset=row * n + src]
        out_idx[unsafe_offset=row] = Int64(src)
        row += stride


@always_inline
def _sort_pairs_gpu[
    dtype: DType, KT: DType, descending: Bool
](
    in_vals: Pointer[Scalar[dtype], ImmutAnyOrigin],
    keys: Pointer[Scalar[KT], MutAnyOrigin],
    scratch: Pointer[Scalar[DType.int32], MutAnyOrigin],
    rows: Int,
    n: Int,
    out_k: Int,
    pad: Int,
    route: Int,
    ctx: DeviceContext,
) raises:
    """Leave each row's first `out_k` sorted (key, index) pairs at the start
    of its `pad`-wide slice of `keys` / `scratch`."""
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        comptime TILE = _TILE[KT]
        comptime tile_sort = _sort_tile_kernel[dtype, KT, descending, False]
        comptime pair_sort = _sort_tile_kernel[dtype, KT, descending, True]
        var immutable_keys = keys.as_imm()
        var immutable_scratch = scratch.as_imm()

        if route == 0:
            if pad > TILE:
                raise Error("sort route 0 requires a row that fits one tile")
            _enqueue_cached[tile_sort](
                ctx,
                1,
                rows,
                1,
                SORT_THREADS,
                keys,
                scratch,
                in_vals,
                immutable_keys,
                immutable_scratch,
                Int64(n),
                Int64(pad),
                Int64(pad),
                Int64(pad),
                Int64(0),
            )
        elif route == 1:
            var tiles = (n + TILE - 1) // TILE
            _enqueue_cached[tile_sort](
                ctx,
                tiles,
                rows,
                1,
                SORT_THREADS,
                keys,
                scratch,
                in_vals,
                immutable_keys,
                immutable_scratch,
                Int64(n),
                Int64(pad),
                Int64(out_k),
                Int64(TILE),
                Int64(0),
            )
            var span = next_power_of_two(pad)
            if span > TILE:
                raise Error("topk tournament needs one final tile")
            # In place is safe: this launch is one block per row, and every
            # load completes at the barrier before the first store.
            _enqueue_cached[pair_sort](
                ctx,
                1,
                rows,
                1,
                SORT_THREADS,
                keys,
                scratch,
                in_vals,
                immutable_keys,
                immutable_scratch,
                Int64(pad),
                Int64(pad),
                Int64(out_k),
                Int64(span),
                Int64(0),
            )
        elif route == 2:
            var tiles = pad // TILE
            _enqueue_cached[tile_sort](
                ctx,
                tiles,
                rows,
                1,
                SORT_THREADS,
                keys,
                scratch,
                in_vals,
                immutable_keys,
                immutable_scratch,
                Int64(n),
                Int64(pad),
                Int64(TILE),
                Int64(TILE),
                Int64(1),
            )
            var step_pairs = pad >> 1
            var step_blocks = max(
                1, min((step_pairs + SORT_THREADS - 1) // SORT_THREADS, 1024)
            )
            var k = TILE * 2
            while k <= pad:
                var j = k >> 1
                while j >= TILE:
                    _enqueue_cached[_bitonic_step_kernel[KT, descending]](
                        ctx,
                        step_blocks,
                        rows,
                        1,
                        SORT_THREADS,
                        keys,
                        scratch,
                        Int64(pad),
                        Int64(j),
                        Int64(k),
                    )
                    j >>= 1
                _enqueue_cached[_bitonic_merge_local_kernel[KT, descending]](
                    ctx,
                    tiles,
                    rows,
                    1,
                    SORT_THREADS,
                    keys,
                    scratch,
                    Int64(pad),
                    Int64(k),
                )
                k <<= 1
        else:
            raise Error("unknown sort route " + String(route))


@always_inline
def _sort_gpu[
    dtype: DType, KT: DType, descending: Bool
](
    out_vals: Pointer[Scalar[dtype], MutAnyOrigin],
    out_idx: Pointer[Scalar[DType.int64], MutAnyOrigin],
    in_vals: Pointer[Scalar[dtype], ImmutAnyOrigin],
    keys: Pointer[Scalar[KT], MutAnyOrigin],
    scratch: Pointer[Scalar[DType.int32], MutAnyOrigin],
    rows: Int,
    n: Int,
    out_k: Int,
    pad: Int,
    route: Int,
    ctx: DeviceContext,
) raises:
    _sort_pairs_gpu[dtype, KT, descending](
        in_vals, keys, scratch, rows, n, out_k, pad, route, ctx
    )
    comptime if has_accelerator():
        _enqueue_cached[_gather_kernel[dtype]](
            ctx,
            _gs_blocks(out_k),
            rows,
            1,
            GS_THREADS,
            out_vals,
            out_idx,
            in_vals,
            scratch.as_imm(),
            Int64(n),
            Int64(pad),
            Int64(out_k),
        )


@always_inline
def _dispatch_sort[
    dtype: DType
](
    out_vals_addr: Int,
    out_idx_addr: Int,
    in_addr: Int,
    keys_addr: Int,
    scratch_addr: Int,
    rows: Int,
    n: Int,
    out_k: Int,
    pad: Int,
    route: Int,
    descending: Bool,
    ctx: DeviceContext,
) raises:
    comptime KT = _KEY_DTYPE[dtype]
    var out_vals = _make_ptr[dtype](out_vals_addr).as_unsafe_any_origin()
    var out_idx = _make_ptr[DType.int64](out_idx_addr).as_unsafe_any_origin()
    var in_vals = _make_ptr[dtype](in_addr).as_unsafe_any_origin().as_imm()
    var keys = _make_ptr[KT](keys_addr).as_unsafe_any_origin()
    var scratch = _make_ptr[DType.int32](scratch_addr).as_unsafe_any_origin()
    if descending:
        _sort_gpu[dtype, KT, True](
            out_vals,
            out_idx,
            in_vals,
            keys,
            scratch,
            rows,
            n,
            out_k,
            pad,
            route,
            ctx,
        )
    else:
        _sort_gpu[dtype, KT, False](
            out_vals,
            out_idx,
            in_vals,
            keys,
            scratch,
            rows,
            n,
            out_k,
            pad,
            route,
            ctx,
        )


@always_inline
def _dispatch_select[
    dtype: DType
](
    out_vals_addr: Int,
    out_idx_addr: Int,
    in_addr: Int,
    keys_addr: Int,
    scratch_addr: Int,
    rows: Int,
    n: Int,
    out_k: Int,
    pad: Int,
    route: Int,
    pos: Int,
    mode: Int,
    ctx: DeviceContext,
) raises:
    comptime KT = _KEY_DTYPE[dtype]
    var out_vals = _make_ptr[dtype](out_vals_addr).as_unsafe_any_origin()
    var out_idx = _make_ptr[DType.int64](out_idx_addr).as_unsafe_any_origin()
    var in_vals = _make_ptr[dtype](in_addr).as_unsafe_any_origin().as_imm()
    var keys = _make_ptr[KT](keys_addr).as_unsafe_any_origin()
    var scratch = _make_ptr[DType.int32](scratch_addr).as_unsafe_any_origin()
    _sort_pairs_gpu[dtype, KT, False](
        in_vals, keys, scratch, rows, n, out_k, pad, route, ctx
    )
    comptime if has_accelerator():
        comptime for m in range(3):
            if mode == m:
                _enqueue_cached[_select_kernel[dtype, KT, m]](
                    ctx,
                    _gs_blocks(rows),
                    1,
                    1,
                    GS_THREADS,
                    out_vals,
                    out_idx,
                    in_vals,
                    keys.as_imm(),
                    scratch.as_imm(),
                    Int64(rows),
                    Int64(n),
                    Int64(pad),
                    Int64(pos),
                )


def _select_go(
    out_vals_obj: Arg,
    out_idx_obj: Arg,
    in_obj: Arg,
    keys_obj: Arg,
    scratch_obj: Arg,
    rows_obj: Arg,
    n_obj: Arg,
    out_k_obj: Arg,
    pad_obj: Arg,
    route_obj: Arg,
    pos_obj: Arg,
    mode_obj: Arg,
    dtype_obj: Arg,
    ctx_obj: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var mode = _raw_int(mode_obj)
    if mode < 0 or mode > SELECT_NANMEDIAN:
        raise Error("unknown sort select mode " + String(mode))
    var handled = False
    comptime for dt in SORT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _dispatch_select[dt](
                    _raw_int(out_vals_obj),
                    _raw_int(out_idx_obj),
                    _raw_int(in_obj),
                    _raw_int(keys_obj),
                    _raw_int(scratch_obj),
                    _raw_int(rows_obj),
                    _raw_int(n_obj),
                    _raw_int(out_k_obj),
                    _raw_int(pad_obj),
                    _raw_int(route_obj),
                    _raw_int(pos_obj),
                    mode,
                    _raw_ctx(ctx_obj),
                )
                handled = True
    if not handled:
        raise Error(
            "unsupported dtype specialization for sort select: " + String(dtype)
        )


def _sort_go(
    out_vals_obj: Arg,
    out_idx_obj: Arg,
    in_obj: Arg,
    keys_obj: Arg,
    scratch_obj: Arg,
    rows_obj: Arg,
    n_obj: Arg,
    out_k_obj: Arg,
    pad_obj: Arg,
    route_obj: Arg,
    descending_obj: Arg,
    dtype_obj: Arg,
    ctx_obj: Arg,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var handled = False
    comptime for dt in SORT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _dispatch_sort[dt](
                    _raw_int(out_vals_obj),
                    _raw_int(out_idx_obj),
                    _raw_int(in_obj),
                    _raw_int(keys_obj),
                    _raw_int(scratch_obj),
                    _raw_int(rows_obj),
                    _raw_int(n_obj),
                    _raw_int(out_k_obj),
                    _raw_int(pad_obj),
                    _raw_int(route_obj),
                    Bool(_raw_int(descending_obj)),
                    _raw_ctx(ctx_obj),
                )
                handled = True
    if not handled:
        raise Error(
            "unsupported dtype specialization for sort: " + String(dtype)
        )


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["Sort"]():
            _spec_dispatcher13[_sort_go, "Sort"](argv, argc)
            return 0
        comptime if _op_on["SortSelect"]():
            _spec_dispatcher14[_select_go, "SortSelect"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
