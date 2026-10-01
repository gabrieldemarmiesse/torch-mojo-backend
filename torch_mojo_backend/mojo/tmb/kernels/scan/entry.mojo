# ===----------------------------------------------------------------------=== #
# Cumulative scans along one dimension: cumsum (every layout the nn family's
# block prefix sums do not cover), cumprod, logcumsumexp, and cummax / cummin
# with their int64 indices (aten::_cummax_helper / _cummin_helper).
#
# GEOMETRY. The operand is contiguous and viewed as (outer, n, inner): element
# (o, r, i) sits at `(o * n + r) * inner + i`, and the scan runs over `r`. The
# output (and the indices, for cummax / cummin) is a contiguous buffer of the
# same shape. The op side (tmb/ops/scans.mojo) materializes a strided operand
# and copies into a strided `out=`.
#
# TWO KERNELS, one algebra (`ScanOp`):
#
#   `_scan_lines_kernel`  one thread per line, walking it sequentially. With
#                         inner > 1 neighbouring threads own neighbouring
#                         columns, so every step is a coalesced load; it also
#                         serves short rows (inner == 1, n small).
#   `_scan_rows_kernel`   inner == 1 and a row long enough to matter: one block
#                         per row, `SCAN_THREADS`-element tiles scanned in
#                         shared memory (Hillis-Steele) and threaded together by
#                         a running carry.
#
# FOLLOW-UP (performance): a long scan dim over FEW lines still runs one block
# (or one thread) per line, e.g. `cumprod(randn(10**6))` is a single block
# walking 4000 tiles. The nn family's three-pass workspace scan does this for
# cumsum; generalizing it to `ScanOp` (chunk totals -> scan of the totals ->
# re-scan seeded with each chunk's prefix) is the fix.
#
# ACCUMULATION. Every combine computes in float32 for the half dtypes
# (float64 in float64; integers in their own dtype, int64 wrapping like
# torch), and the value scans (cumsum, cumprod, logcumsumexp) store every
# combined value in the element dtype, as CUDA's `scan_dim<scalar_t>` keeps it
# in all three of its routes (ScanUtils.cuh: the outer-dim kernel's running
# `scalar_t acc`, the innermost kernel's `scalar_t` shared tile, cub's
# accumulator of `std::plus<scalar_t>`): a half cumsum of ones saturates at
# 2048, a half cumprod that overflows stays inf. The orders are CUDA's too:
# the outer dim runs one sequential line per thread (`_scan_lines_kernel`),
# the innermost dim the same Sklansky tiles with the carry folded into each
# tile's first element (`_scan_rows_sklansky_kernel`); a 1-D half/bf16 scan,
# which CUDA hands to cub, follows cub's tile structure
# (`_scan_1d_cub_kernel`).

# TIES AND NaN. cummax / cummin follow ATen's `scan_dim_with_indices` (and CPU's
# `cummax_cummin_helper`): a later element replaces the running extremum when
# it is NaN, or when the running value is not NaN and the new one compares
# `>=` (max) / `<=` (min) -- so ties report the LAST index, and a NaN sticks
# with each later NaN taking over its index. logcumsumexp combines with
# `_log_add_exp_helper` (LogcumsumexpKernel.cu), NaN and infinities included.
# ===----------------------------------------------------------------------=== #

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import has_accelerator, size_of
from std.utils.numerics import isinf, isnan, max_finite, max_or_inf
from std.utils.numerics import min_finite, min_or_neg_inf
from std.utils.static_tuple import StaticTuple

from tmb.kernels.common.libdevice_port import (
    nv_exp,
    nv_expf,
    nv_log1p,
    nv_log1pf,
)
from tmb.kernels.common.op_utils import (
    Arg,
    Argv,
    _enqueue_cached,
    _enqueue_cached_2d,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _spec_dispatcher9,
)
from tmb.kernels.common.variant_gates import (
    ErrBuf,
    NO_OP_COMPILED,
    _dtype_arg_on,
    _op_on,
    _tmb_entry_error,
)

comptime SCAN_THREADS = 256

# ScanUtils.cuh `scan_dim`'s three routes (mirrored in tmb/ops/scans.mojo).
comptime ROUTE_1D = 0
comptime ROUTE_INNERMOST = 1
comptime ROUTE_OUTER = 2

# A row shorter than this is scanned by one thread: a 256-thread block would
# leave most of its lanes idle on the only tile it has.
comptime SCAN_ROW_BLOCK_MIN = 64

# grid-stride caps (rows of the block kernel, lines of the thread kernel).
comptime SCAN_MAX_BLOCKS = 65535

comptime SUM_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int64,
    DType.int32,
]

comptime FLOAT_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
]

# bool rides uint8 (same storage, 0 / 1).
comptime SELECT_DTYPES: List[DType] = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
]


@always_inline
def _wide[dt: DType]() -> DType:
    """float32 for the half dtypes and float32, float64 for float64, the
    integer dtype itself otherwise."""
    comptime if dt.is_floating_point() and dt != DType.float64:
        return DType.float32
    else:
        return dt


trait ScanOp:
    """One scan's algebra. `combine(a, b)` merges an earlier `a` with a later
    `b`; for the selecting scans, `take_later(cur, nxt)` decides whether the
    later element replaces the running one (its index travels with it)."""

    comptime name: StaticString
    comptime dtypes: List[DType]
    comptime with_index: Bool
    comptime rounds: Bool
    """Does CUDA keep the running value in the element dtype (rounding a
    half/bfloat16 value after every combine)?"""

    @staticmethod
    def acc_dtype[dt: DType]() -> DType:
        ...

    @staticmethod
    def identity[acc: DType]() -> Scalar[acc]:
        ...

    @staticmethod
    def combine[acc: DType](a: Scalar[acc], b: Scalar[acc]) -> Scalar[acc]:
        ...

    @staticmethod
    def take_later[acc: DType](cur: Scalar[acc], nxt: Scalar[acc]) -> Bool:
        ...


struct SumScan(ScanOp):
    comptime name = "sum"
    comptime dtypes = SUM_DTYPES
    comptime with_index = False
    comptime rounds = True

    @staticmethod
    def acc_dtype[dt: DType]() -> DType:
        return _wide[dt]()

    @staticmethod
    def identity[acc: DType]() -> Scalar[acc]:
        return Scalar[acc](0)

    @staticmethod
    def combine[acc: DType](a: Scalar[acc], b: Scalar[acc]) -> Scalar[acc]:
        return a + b

    @staticmethod
    def take_later[acc: DType](cur: Scalar[acc], nxt: Scalar[acc]) -> Bool:
        return True


struct ProdScan(ScanOp):
    comptime name = "prod"
    comptime dtypes = SUM_DTYPES
    comptime with_index = False
    comptime rounds = True

    @staticmethod
    def acc_dtype[dt: DType]() -> DType:
        return _wide[dt]()

    @staticmethod
    def identity[acc: DType]() -> Scalar[acc]:
        return Scalar[acc](1)

    @staticmethod
    def combine[acc: DType](a: Scalar[acc], b: Scalar[acc]) -> Scalar[acc]:
        return a * b

    @staticmethod
    def take_later[acc: DType](cur: Scalar[acc], nxt: Scalar[acc]) -> Bool:
        return True


@always_inline
def _exp[acc: DType](x: Scalar[acc]) -> Scalar[acc]:
    comptime if acc == DType.float64:
        return nv_exp(x.cast[DType.float64]()).cast[acc]()
    else:
        return nv_expf(x.cast[DType.float32]()).cast[acc]()


@always_inline
def _log1p[acc: DType](x: Scalar[acc]) -> Scalar[acc]:
    comptime if acc == DType.float64:
        return nv_log1p(x.cast[DType.float64]()).cast[acc]()
    else:
        return nv_log1pf(x.cast[DType.float32]()).cast[acc]()


struct LogSumExpScan(ScanOp):
    """`_log_add_exp_helper` (LogcumsumexpKernel.cu): NaN propagates, two
    equal infinities return the operand unchanged (no inf - inf)."""

    comptime name = "logsumexp"
    comptime dtypes = FLOAT_DTYPES
    comptime with_index = False
    comptime rounds = True

    @staticmethod
    def acc_dtype[dt: DType]() -> DType:
        return _wide[dt]()

    @staticmethod
    def identity[acc: DType]() -> Scalar[acc]:
        return min_or_neg_inf[acc]()

    @staticmethod
    def combine[acc: DType](a: Scalar[acc], b: Scalar[acc]) -> Scalar[acc]:
        var nan_a = isnan(a)
        var nan_b = isnan(b)
        var lo = b if nan_b else (a if nan_a else (b if b < a else a))
        var hi = b if nan_b else (a if nan_a else (b if b > a else a))
        var finite_lo = not isnan(lo) and not isinf(lo)
        if lo != hi or finite_lo:
            return _log1p[acc](_exp[acc](lo - hi)) + hi
        return a

    @staticmethod
    def take_later[acc: DType](cur: Scalar[acc], nxt: Scalar[acc]) -> Bool:
        return True


@always_inline
def _lowest[acc: DType]() -> Scalar[acc]:
    comptime if acc.is_floating_point():
        return min_or_neg_inf[acc]()
    else:
        return min_finite[acc]()


@always_inline
def _highest[acc: DType]() -> Scalar[acc]:
    comptime if acc.is_floating_point():
        return max_or_inf[acc]()
    else:
        return max_finite[acc]()


struct MaxScan(ScanOp):
    comptime name = "max"
    comptime dtypes = SELECT_DTYPES
    comptime with_index = True
    comptime rounds = False

    @staticmethod
    def acc_dtype[dt: DType]() -> DType:
        return dt

    @staticmethod
    def identity[acc: DType]() -> Scalar[acc]:
        return _lowest[acc]()

    @staticmethod
    def combine[acc: DType](a: Scalar[acc], b: Scalar[acc]) -> Scalar[acc]:
        return b if Self.take_later[acc](a, b) else a

    @staticmethod
    def take_later[acc: DType](cur: Scalar[acc], nxt: Scalar[acc]) -> Bool:
        comptime if acc.is_floating_point():
            return isnan(nxt) or (not isnan(cur) and nxt >= cur)
        else:
            return nxt >= cur


struct MinScan(ScanOp):
    comptime name = "min"
    comptime dtypes = SELECT_DTYPES
    comptime with_index = True
    comptime rounds = False

    @staticmethod
    def acc_dtype[dt: DType]() -> DType:
        return dt

    @staticmethod
    def identity[acc: DType]() -> Scalar[acc]:
        return _highest[acc]()

    @staticmethod
    def combine[acc: DType](a: Scalar[acc], b: Scalar[acc]) -> Scalar[acc]:
        return b if Self.take_later[acc](a, b) else a

    @staticmethod
    def take_later[acc: DType](cur: Scalar[acc], nxt: Scalar[acc]) -> Bool:
        comptime if acc.is_floating_point():
            return isnan(nxt) or (not isnan(cur) and nxt <= cur)
        else:
            return nxt <= cur


@always_inline
def _merge[
    Op: ScanOp, dtype: DType, acc: DType
](a_v: Scalar[acc], a_i: Int64, b_v: Scalar[acc], b_i: Int64) -> Tuple[
    Scalar[acc], Int64
]:
    """(earlier a) then (later b): a value scan combines, a selecting scan
    keeps whichever element wins together with its index. A rounding scan
    (`Op.rounds`) stores every combined value in the element dtype, as
    CUDA's `scan_dim<scalar_t>` does: a half cumprod that overflows to inf
    stays inf."""
    comptime if Op.with_index:
        if Op.take_later[acc](a_v, b_v):
            return (b_v, b_i)
        return (a_v, a_i)
    else:
        var c = Op.combine[acc](a_v, b_v)
        comptime if Op.rounds and dtype != acc:
            c = c.cast[dtype]().cast[acc]()
        return (c, a_i)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SCAN_THREADS))
)
@__name(t"scan_lines_{Op.name}_{dtype}")
def _scan_lines_kernel[
    Op: ScanOp, dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    idx_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    lines_arg: Int64,
    n_arg: Int64,
    inner_arg: Int64,
):
    """One thread per (outer, inner) line, walking the scan dim in order."""
    comptime acc = Op.acc_dtype[dtype]()
    var lines = Int(lines_arg)
    var n = Int(n_arg)
    var inner = Int(inner_arg)
    var t = Int(block_idx.x) * SCAN_THREADS + Int(thread_idx.x)
    var stride = Int(grid_dim.x) * SCAN_THREADS
    while t < lines:
        var o = t // inner
        var i = t - o * inner
        var base = o * n * inner + i
        var v = Op.identity[acc]()
        var vi = Int64(-1)
        for r in range(n):
            var off = base + r * inner
            var x = in_ptr[unsafe_offset=off].cast[acc]()
            var m = _merge[Op, dtype, acc](v, vi, x, Int64(r))
            v = m[0]
            vi = m[1]
            out_ptr[unsafe_offset=off] = v.cast[dtype]()
            comptime if Op.with_index:
                idx_ptr[unsafe_offset=off] = vi
        t += stride


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SCAN_THREADS))
)
@__name(t"scan_rows_block_{Op.name}_{dtype}_t{SCAN_THREADS}")
def _scan_rows_kernel[
    Op: ScanOp, dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    idx_ptr: Pointer[Scalar[DType.int64], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    n_arg: Int64,
):
    """One block per contiguous row: SCAN_THREADS-element tiles, each scanned
    in shared memory and offset by the carry of the tiles before it."""
    comptime acc = Op.acc_dtype[dtype]()
    var rows = Int(rows_arg)
    var n = Int(n_arg)
    var tid = Int(thread_idx.x)
    var sv = stack_allocation[
        SCAN_THREADS, acc, address_space=AddressSpace.SHARED
    ]()
    var si = stack_allocation[
        SCAN_THREADS, DType.int64, address_space=AddressSpace.SHARED
    ]()
    var row = Int(block_idx.x)
    while row < rows:
        var base = row * n
        var carry_v = Op.identity[acc]()
        var carry_i = Int64(-1)
        var start = 0
        while start < n:
            var j = start + tid
            var v = Op.identity[acc]()
            var vi = Int64(-1)
            if j < n:
                v = in_ptr[unsafe_offset=base + j].cast[acc]()
                vi = Int64(j)
            sv[unsafe_offset=tid] = v
            si[unsafe_offset=tid] = vi
            barrier()
            var off = 1
            while off < SCAN_THREADS:
                var cv = sv[unsafe_offset=tid]
                var ci = si[unsafe_offset=tid]
                if tid >= off:
                    var m = _merge[Op, dtype, acc](
                        sv[unsafe_offset=tid - off],
                        si[unsafe_offset=tid - off],
                        cv,
                        ci,
                    )
                    cv = m[0]
                    ci = m[1]
                barrier()
                sv[unsafe_offset=tid] = cv
                si[unsafe_offset=tid] = ci
                barrier()
                off *= 2
            var r = _merge[Op, dtype, acc](
                carry_v, carry_i, sv[unsafe_offset=tid], si[unsafe_offset=tid]
            )
            if j < n:
                out_ptr[unsafe_offset=base + j] = r[0].cast[dtype]()
                comptime if Op.with_index:
                    idx_ptr[unsafe_offset=base + j] = r[1]
            var c = _merge[Op, dtype, acc](
                carry_v,
                carry_i,
                sv[unsafe_offset=SCAN_THREADS - 1],
                si[unsafe_offset=SCAN_THREADS - 1],
            )
            carry_v = c[0]
            carry_i = c[1]
            # Every lane has read the tile's last slot before the next tile
            # overwrites the shared buffers.
            barrier()
            start += SCAN_THREADS
        row += Int(grid_dim.x)


comptime CUDA_SCAN_THREADS = 512


@always_inline
def _cuda_log_threads_x(num_rows: Int, row_size: Int) -> Int:
    """ScanUtils.cuh `get_log_num_threads_x_inner_scan<uint32_t>`, its
    unsigned wrap-around included (a negative `9 + diff` clamps to 9)."""
    var lx = UInt32(0)
    var ly = UInt32(0)
    while (UInt32(1) << lx) < UInt32(row_size):
        lx += 1
    while (UInt32(1) << ly) < UInt32(num_rows):
        ly += 1
    var diff = lx - ly  # uint32 arithmetic, as in ATen
    var l = (UInt32(9) + diff) / UInt32(2)
    return Int(min(max(l, UInt32(4)), UInt32(9)))


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(CUDA_SCAN_THREADS)
    )
)
@__name(t"scan_rows_sklansky_{Op.name}_{dtype}")
def _scan_rows_sklansky_kernel[
    Op: ScanOp, dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    n_arg: Int64,
    log_x_arg: Int64,
):
    """ScanUtils.cuh `tensor_kernel_scan_innermost_dim_impl`, step for step:
    block (2^log_x, 512 / 2^log_x), one row per y-lane, tiles of 2 * 2^log_x
    elements scanned by the Sklansky network after the previous tiles' total
    is folded into the tile's FIRST element. With `Op.rounds` every combine
    stores the element dtype, so a half scan rounds exactly where CUDA's
    `scan_dim<scalar_t>` does."""
    comptime acc = Op.acc_dtype[dtype]()
    var rows = Int(rows_arg)
    var n = Int(n_arg)
    var log_x = Int(log_x_arg)
    var nx = 1 << log_x
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var ny = CUDA_SCAN_THREADS // nx
    var smem = stack_allocation[
        2 * CUDA_SCAN_THREADS, acc, address_space=AddressSpace.SHARED
    ]()
    var off = ty * 2 * nx
    var block_row = Int(block_idx.x) * ny
    while block_row < rows:
        var row = block_row + ty
        var exists = row < rows
        var total = Op.identity[acc]()
        var base = row * n
        var col0 = 0
        while col0 < n:
            var c1 = col0 + tx
            var c2 = col0 + nx + tx
            if exists:
                smem[unsafe_offset=off + tx] = (
                    in_ptr[unsafe_offset=base + c1].cast[acc]() if c1
                    < n else Op.identity[acc]()
                )
                smem[unsafe_offset=off + nx + tx] = (
                    in_ptr[unsafe_offset=base + c2].cast[acc]() if c2
                    < n else Op.identity[acc]()
                )
                if tx == 0:
                    smem[unsafe_offset=off + 0] = _merge[Op, dtype, acc](
                        total, Int64(-1), smem[unsafe_offset=off + 0], Int64(-1)
                    )[0]
            barrier()
            for m in range(log_x + 1):
                if exists:
                    # 32-bit index math with a mask for `tx % 2^m`: the
                    # 64-bit modulo was most of the step's cost.
                    var mu = UInt32(m)
                    var t32 = UInt32(tx)
                    var sz = UInt32(1) << mu
                    var a = ((t32 >> mu) << (mu + 1)) | sz
                    var ti = Int(a + (t32 & (sz - 1)))
                    var si = Int(a - 1)
                    smem[unsafe_offset=off + ti] = _merge[Op, dtype, acc](
                        smem[unsafe_offset=off + si],
                        Int64(-1),
                        smem[unsafe_offset=off + ti],
                        Int64(-1),
                    )[0]
                barrier()
            if exists:
                if c1 < n:
                    out_ptr[unsafe_offset=base + c1] = smem[
                        unsafe_offset=off + tx
                    ].cast[dtype]()
                if c2 < n:
                    out_ptr[unsafe_offset=base + c2] = smem[
                        unsafe_offset=off + nx + tx
                    ].cast[dtype]()
            total = smem[unsafe_offset=off + 2 * nx - 1]
            barrier()
            col0 += 2 * nx
        block_row += ny * Int(grid_dim.x)


# cub's DeviceScan agent for a 2-byte accumulator (c10::Half / BFloat16 are
# not cub "primitive" types, so sm90 takes the default tuning: 128 threads x
# 15 nominal 4-byte items, scaled by MemBoundScaling to 30 items of 2 bytes).
comptime CUB_THREADS = 128
comptime CUB_ITEMS = 30
comptime CUB_WARPS = CUB_THREADS // 32
comptime CUB_TILE = CUB_THREADS * CUB_ITEMS

# The tile-prefix pass: one block; up to CUB_SEQ_TILES tiles their prefixes
# are taken in order by one thread (cub's look-back resolved in order, what
# makes the result bit-exact against CUDA), past it in parallel chunks (cub's
# decoupled look-back associates them by timing there anyway).
comptime CUB_PREFIX_THREADS = 1024
comptime CUB_SEQ_TILES = 64


@always_inline
def _cub_block_prefix[
    Op: ScanOp, dtype: DType, acc: DType
](agg: Scalar[acc], tid: Int) -> Tuple[Scalar[acc], Bool, Scalar[acc]]:
    """cub's BLOCK_SCAN_WARP_SCANS over one aggregate per thread: Kogge-Stone
    inside each warp (shfl_up 1..16), the warp totals combined in order. Every
    thread gets (its exclusive prefix, whether it has one, the block total)."""
    var aggs = stack_allocation[
        CUB_THREADS, acc, address_space=AddressSpace.SHARED
    ]()
    var warp_tot = stack_allocation[
        CUB_WARPS, acc, address_space=AddressSpace.SHARED
    ]()
    var lane = tid % 32
    var warp = tid // 32
    aggs[unsafe_offset=tid] = agg
    barrier()
    var v = agg
    var off = 1
    while off < 32:
        var other = aggs[unsafe_offset=tid - off] if lane >= off else v
        barrier()
        if lane >= off:
            v = _merge[Op, dtype, acc](other, Int64(-1), v, Int64(-1))[0]
        aggs[unsafe_offset=tid] = v
        barrier()
        off *= 2
    if lane == 31:
        warp_tot[unsafe_offset=warp] = v
    barrier()
    var excl = aggs[unsafe_offset=tid - 1] if lane > 0 else v
    var has_excl = lane > 0
    if warp > 0:
        var wp = warp_tot[unsafe_offset=0]
        for w in range(1, warp):
            wp = _merge[Op, dtype, acc](
                wp, Int64(-1), warp_tot[unsafe_offset=w], Int64(-1)
            )[0]
        if has_excl:
            excl = _merge[Op, dtype, acc](wp, Int64(-1), excl, Int64(-1))[0]
        else:
            excl = wp
        has_excl = True
    var block_tot = warp_tot[unsafe_offset=0]
    for w in range(1, CUB_WARPS):
        block_tot = _merge[Op, dtype, acc](
            block_tot, Int64(-1), warp_tot[unsafe_offset=w], Int64(-1)
        )[0]
    barrier()  # the shared slots are reused by the next call
    return (excl, has_excl, block_tot)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(CUB_THREADS))
)
@__name(t"scan_1d_cub_tile_agg_{Op.name}_{dtype}")
def _scan_1d_tile_agg_kernel[
    Op: ScanOp, dtype: DType
](
    ws_ptr: Pointer[Scalar[Op.acc_dtype[dtype]()], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    n_arg: Int64,
):
    """Pass 1: every tile's aggregate (cub's block total), in parallel."""
    comptime acc = Op.acc_dtype[dtype]()
    var n = Int(n_arg)
    var tid = Int(thread_idx.x)
    var t = Int(block_idx.x)
    var t0 = t * CUB_TILE
    var w = min(CUB_TILE, n - t0)
    var tile = stack_allocation[
        CUB_TILE, acc, address_space=AddressSpace.SHARED
    ]()
    # The tile, coalesced (cub's warp-transposed load), into shared memory:
    # 16-byte vectors, all issued before any is used, when the tile is full
    # and aligned (the scalar loop otherwise).
    comptime V = 16 // size_of[dtype]()
    comptime NVEC = CUB_TILE // V
    if w == CUB_TILE and (Int(in_ptr) + t0 * size_of[dtype]()) % 16 == 0:
        comptime for j in range(ceildiv(NVEC, CUB_THREADS)):
            var vi = j * CUB_THREADS + tid
            if vi < NVEC:
                var vv = in_ptr.unsafe_load[width=V, alignment=16](
                    t0 + vi * V
                ).cast[acc]()
                comptime for e in range(V):
                    tile[unsafe_offset=vi * V + e] = vv[e]
    else:
        for k in range(tid, CUB_TILE, CUB_THREADS):
            tile[unsafe_offset=k] = (
                in_ptr[unsafe_offset=t0 + k].cast[acc]() if k
                < w else Op.identity[acc]()
            )
    barrier()
    var cnt = max(0, min(CUB_ITEMS, w - tid * CUB_ITEMS))
    # The thread's 30 blocked items, reduced in order, from registers (the
    # tail past `w` is the identity, which every combine passes through
    # exactly, so a full unrolled chain equals cub's `cnt`-long one).
    var items = SIMD[acc, 32](Op.identity[acc]())
    comptime for k in range(CUB_ITEMS):
        items[k] = tile[unsafe_offset=tid * CUB_ITEMS + k]
    var agg = items[0]
    comptime for k in range(1, CUB_ITEMS):
        agg = _merge[Op, dtype, acc](agg, Int64(-1), items[k], Int64(-1))[0]
    var r = _cub_block_prefix[Op, dtype, acc](agg, tid)
    if tid == 0:
        ws_ptr[unsafe_offset=t] = r[2]


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(CUB_PREFIX_THREADS)
    )
)
@__name(t"scan_1d_cub_tile_prefix_{Op.name}_{dtype}")
def _scan_1d_tile_prefix_kernel[
    Op: ScanOp, dtype: DType
](
    ws_ptr: Pointer[Scalar[Op.acc_dtype[dtype]()], MutAnyOrigin],
    tiles_arg: Int64,
):
    """Pass 2: the tile aggregates at ws[0, tiles) -> the exclusive tile
    prefixes at ws[tiles + i] (i >= 1; tile 0 has none)."""
    comptime acc = Op.acc_dtype[dtype]()
    var tiles = Int(tiles_arg)
    var tid = Int(thread_idx.x)
    if tiles <= CUB_SEQ_TILES:
        if tid == 0:
            var p = ws_ptr[unsafe_offset=0]
            for i in range(1, tiles):
                ws_ptr[unsafe_offset=tiles + i] = p
                p = _merge[Op, dtype, acc](
                    p, Int64(-1), ws_ptr[unsafe_offset=i], Int64(-1)
                )[0]
        return
    var part = stack_allocation[
        CUB_PREFIX_THREADS, acc, address_space=AddressSpace.SHARED
    ]()
    var chunk = ceildiv(tiles, CUB_PREFIX_THREADS)
    var lo = min(tid * chunk, tiles)
    var hi = min(lo + chunk, tiles)
    var tot = Op.identity[acc]()
    for i in range(lo, hi):
        tot = (
            ws_ptr[unsafe_offset=i] if i
            == lo else _merge[Op, dtype, acc](
                tot, Int64(-1), ws_ptr[unsafe_offset=i], Int64(-1)
            )[0]
        )
    part[unsafe_offset=tid] = tot
    barrier()
    var off = 1
    var v = tot
    while off < CUB_PREFIX_THREADS:
        var other = part[unsafe_offset=tid - off] if tid >= off else v
        barrier()
        if tid >= off:
            v = _merge[Op, dtype, acc](other, Int64(-1), v, Int64(-1))[0]
        part[unsafe_offset=tid] = v
        barrier()
        off *= 2
    # Chunks before this one, then this chunk's tiles in order.
    var has = tid > 0 and lo > 0
    var p = part[unsafe_offset=tid - 1] if tid > 0 else Op.identity[acc]()
    for i in range(lo, hi):
        if has:
            ws_ptr[unsafe_offset=tiles + i] = p
            p = _merge[Op, dtype, acc](
                p, Int64(-1), ws_ptr[unsafe_offset=i], Int64(-1)
            )[0]
        else:
            p = ws_ptr[unsafe_offset=i]
            has = True


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(CUB_THREADS))
)
@__name(t"scan_1d_cub_tile_scan_{Op.name}_{dtype}")
def _scan_1d_tile_scan_kernel[
    Op: ScanOp, dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    ws_ptr: Pointer[Scalar[Op.acc_dtype[dtype]()], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    n_arg: Int64,
    tiles_arg: Int64,
):
    """Pass 3: every tile rescanned in parallel, each thread's 30 items in
    order from its exclusive prefix (tile prefix, then the block's
    warp-scan prefix), stored back coalesced."""
    comptime acc = Op.acc_dtype[dtype]()
    var n = Int(n_arg)
    var tiles = Int(tiles_arg)
    var tid = Int(thread_idx.x)
    var t = Int(block_idx.x)
    var t0 = t * CUB_TILE
    var w = min(CUB_TILE, n - t0)
    var tile = stack_allocation[
        CUB_TILE, acc, address_space=AddressSpace.SHARED
    ]()
    # The tile, coalesced (cub's warp-transposed load), into shared memory:
    # 16-byte vectors, all issued before any is used, when the tile is full
    # and aligned (the scalar loop otherwise).
    comptime V = 16 // size_of[dtype]()
    comptime NVEC = CUB_TILE // V
    if w == CUB_TILE and (Int(in_ptr) + t0 * size_of[dtype]()) % 16 == 0:
        comptime for j in range(ceildiv(NVEC, CUB_THREADS)):
            var vi = j * CUB_THREADS + tid
            if vi < NVEC:
                var vv = in_ptr.unsafe_load[width=V, alignment=16](
                    t0 + vi * V
                ).cast[acc]()
                comptime for e in range(V):
                    tile[unsafe_offset=vi * V + e] = vv[e]
    else:
        for k in range(tid, CUB_TILE, CUB_THREADS):
            tile[unsafe_offset=k] = (
                in_ptr[unsafe_offset=t0 + k].cast[acc]() if k
                < w else Op.identity[acc]()
            )
    barrier()
    var cnt = max(0, min(CUB_ITEMS, w - tid * CUB_ITEMS))
    # The thread's 30 blocked items, reduced in order, from registers (the
    # tail past `w` is the identity, which every combine passes through
    # exactly, so a full unrolled chain equals cub's `cnt`-long one).
    var items = SIMD[acc, 32](Op.identity[acc]())
    comptime for k in range(CUB_ITEMS):
        items[k] = tile[unsafe_offset=tid * CUB_ITEMS + k]
    var agg = items[0]
    comptime for k in range(1, CUB_ITEMS):
        agg = _merge[Op, dtype, acc](agg, Int64(-1), items[k], Int64(-1))[0]
    var r = _cub_block_prefix[Op, dtype, acc](agg, tid)
    var excl = r[0]
    var has_excl = r[1]
    if t > 0:
        var tp = ws_ptr[unsafe_offset=tiles + t]
        excl = _merge[Op, dtype, acc](tp, Int64(-1), excl, Int64(-1))[
            0
        ] if has_excl else tp
        has_excl = True
    var run = excl if has_excl else Op.identity[acc]()
    comptime for k in range(CUB_ITEMS):
        run = _merge[Op, dtype, acc](run, Int64(-1), items[k], Int64(-1))[0]
        tile[unsafe_offset=tid * CUB_ITEMS + k] = run
    _ = cnt
    barrier()
    if w == CUB_TILE and (Int(out_ptr) + t0 * size_of[dtype]()) % 16 == 0:
        comptime for j in range(ceildiv(NVEC, CUB_THREADS)):
            var vi = j * CUB_THREADS + tid
            if vi < NVEC:
                var vv = SIMD[dtype, V]()
                comptime for e in range(V):
                    vv[e] = tile[unsafe_offset=vi * V + e].cast[dtype]()
                out_ptr.unsafe_store[alignment=16](t0 + vi * V, vv)
    else:
        for k in range(tid, w, CUB_THREADS):
            out_ptr[unsafe_offset=t0 + k] = tile[unsafe_offset=k].cast[dtype]()


@always_inline
def _scan_1d_cub[
    Op: ScanOp, dtype: DType
](
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    inp: Pointer[Scalar[dtype], ImmutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    """A 1-D half/bf16 scan in the combine order of the cub DeviceScan CUDA
    runs for it (ScanUtils.cuh `scan_dim` -> `cuda::cub::inclusive_scan`),
    every combine rounded to the element dtype: tiles of 128 threads x 30
    blocked items, three passes (tile aggregates; tile prefixes; per-tile
    rescan). Bit-exact against CUDA while the tile prefixes are taken in
    order (up to CUB_SEQ_TILES tiles, ~245k elements); past it cub's
    decoupled look-back associates them by timing, and so do we, in
    parallel chunks."""
    comptime acc = Op.acc_dtype[dtype]()
    var tiles = ceildiv(n, CUB_TILE)
    var ws = ctx.enqueue_create_buffer[acc](2 * tiles)
    var ws_ptr = ws.unsafe_ptr().as_unsafe_any_origin()
    _enqueue_cached[_scan_1d_tile_agg_kernel[Op, dtype]](
        ctx, tiles, 1, 1, CUB_THREADS, ws_ptr, inp, Int64(n)
    )
    if tiles > 1:
        _enqueue_cached[_scan_1d_tile_prefix_kernel[Op, dtype]](
            ctx, 1, 1, 1, CUB_PREFIX_THREADS, ws_ptr, Int64(tiles)
        )
    _enqueue_cached[_scan_1d_tile_scan_kernel[Op, dtype]](
        ctx, tiles, 1, 1, CUB_THREADS, dst, ws_ptr, inp, Int64(n), Int64(tiles)
    )
    _ = ws^  # a stream-ordered free after the kernels


@always_inline
def _scan[
    Op: ScanOp, dtype: DType
](
    out_addr: Int,
    idx_addr: Int,
    in_addr: Int,
    outer: Int,
    n: Int,
    inner: Int,
    route: Int,
    ctx: DeviceContext,
) raises:
    """`route` is the path ScanUtils.cuh `scan_dim` takes for this tensor:
    ROUTE_1D when the scan dim holds every element (cub), ROUTE_INNERMOST
    when it is the tensor's last dim, ROUTE_OUTER otherwise -- decided on the
    dims, not on the geometry (a trailing size-1 dim makes `inner == 1`
    without making the scan dim the last one)."""
    var out = _make_ptr[dtype](out_addr).as_unsafe_any_origin()
    var idx = _make_ptr[DType.int64](idx_addr).as_unsafe_any_origin()
    var inp = _make_ptr[dtype](in_addr).as_unsafe_any_origin().as_imm()
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        comptime if not Op.with_index and Op.rounds and (
            dtype == DType.float16 or dtype == DType.bfloat16
        ):
            if route == ROUTE_1D:
                _scan_1d_cub[Op, dtype](out, inp, n, ctx)
                return
        comptime if not Op.with_index:
            if route != ROUTE_OUTER and inner == 1:
                # CUDA's innermost-dim scan (also standing in for its cub
                # route of a 1-D scan of a dtype that does not round).
                var log_x = _cuda_log_threads_x(outer, n)
                var ny = CUDA_SCAN_THREADS >> log_x
                _enqueue_cached_2d[_scan_rows_sklansky_kernel[Op, dtype]](
                    ctx,
                    min(ceildiv(outer, ny), SCAN_MAX_BLOCKS),
                    1,
                    1,
                    1 << log_x,
                    ny,
                    out,
                    inp,
                    Int64(outer),
                    Int64(n),
                    Int64(log_x),
                )
                return
        if Op.with_index and inner == 1 and n >= SCAN_ROW_BLOCK_MIN:
            _enqueue_cached[_scan_rows_kernel[Op, dtype]](
                ctx,
                min(outer, SCAN_MAX_BLOCKS),
                1,
                1,
                SCAN_THREADS,
                out,
                idx,
                inp,
                Int64(outer),
                Int64(n),
            )
            return
        var lines = outer * inner
        _enqueue_cached[_scan_lines_kernel[Op, dtype]](
            ctx,
            min(ceildiv(lines, SCAN_THREADS), SCAN_MAX_BLOCKS),
            1,
            1,
            SCAN_THREADS,
            out,
            idx,
            inp,
            Int64(lines),
            Int64(n),
            Int64(inner),
        )


def _scan_go[
    Op: ScanOp
](
    out_o: Arg,
    idx_o: Arg,
    in_o: Arg,
    outer_o: Arg,
    n_o: Arg,
    inner_o: Arg,
    route_o: Arg,
    dtype_o: Arg,
    ctx_o: Arg,
) raises:
    """Slots: output, indices (0 unless cummax / cummin), operand, outer, n,
    inner, CUDA's route (`_scan`), the operand's dtype code, the device
    context."""
    var dtype = _raw_dtype_int(dtype_o)
    var outer = _raw_int(outer_o)
    var n = _raw_int(n_o)
    var inner = _raw_int(inner_o)
    if outer * n * inner == 0:
        return
    comptime if Op.with_index:
        if _raw_int(idx_o) == 0:
            raise Error("mojo scan: missing indices buffer")
    comptime for dt in Op.dtypes:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _scan[Op, dt](
                    _raw_int(out_o),
                    _raw_int(idx_o),
                    _raw_int(in_o),
                    outer,
                    n,
                    inner,
                    _raw_int(route_o),
                    _raw_ctx(ctx_o),
                )
                return
    raise Error("mojo scan ", Op.name, ": unsupported dtype ", dtype)


@export
def tmb_call(argv: Argv, argc: Int, err: ErrBuf, errcap: Int) abi("C") -> Int32:
    """C entry of this family: one kernel per build (see `OP`).
    Slots are described in op_utils (`Arg`); errors come back as (rc=1, message).
    """
    try:
        comptime if _op_on["ScanSum"]():
            _spec_dispatcher9[_scan_go[SumScan], "ScanSum"](argv, argc)
            return 0
        comptime if _op_on["ScanProd"]():
            _spec_dispatcher9[_scan_go[ProdScan], "ScanProd"](argv, argc)
            return 0
        comptime if _op_on["ScanLogSumExp"]():
            _spec_dispatcher9[_scan_go[LogSumExpScan], "ScanLogSumExp"](
                argv, argc
            )
            return 0
        comptime if _op_on["ScanMax"]():
            _spec_dispatcher9[_scan_go[MaxScan], "ScanMax"](argv, argc)
            return 0
        comptime if _op_on["ScanMin"]():
            _spec_dispatcher9[_scan_go[MinScan], "ScanMin"](argv, argc)
            return 0
        raise Error(NO_OP_COMPILED)
    except e:
        return _tmb_entry_error(err, errcap, e)
