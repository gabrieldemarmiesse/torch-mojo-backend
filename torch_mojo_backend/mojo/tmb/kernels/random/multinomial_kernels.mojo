# ===----------------------------------------------------------------------=== #
# aten::multinomial's two kernels (the op is in tmb/ops/random.mojo).
#
# `_check_kernel` validates every row of the probability matrix in one read --
# a negative entry, an inf/NaN entry, a row that sums to zero -- and leaves a
# bit set in one int32 flag the op reads back, so a bad distribution raises
# ATen's own message synchronously, as CPU torch does, instead of
# `_assert_async`'s device-side abort.
#
# `_draw_kernel` is the WITH-replacement sampler (n_sample > 1; ATen's fast
# path covers the other cases and the op runs it through the registered
# exponential_ / div / argmax / topk). One block per row builds the row's
# inclusive CDF in a workspace, then every thread inverts it for its samples:
# a uniform u in [0, 1 - 2^-24] times the CDF's own last entry lands strictly
# below that entry, and a bisection that keeps `cdf[hi] > x` and
# `lo == 0 or cdf[lo - 1] <= x` ends on a category whose CDF step is
# positive. A zero-probability category's step is exactly zero within one
# thread's sequential chunk (x + 0 == x); only at a chunk boundary could
# rounding give it a sliver, so a hit on a zero entry moves to the nearest
# positive one. Zero-probability categories are therefore never drawn.
# ===----------------------------------------------------------------------=== #

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.memory import AddressSpace, bitcast, stack_allocation
from std.sys.info import has_accelerator
from std.utils.static_tuple import StaticTuple
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier

from tmb.kernels.common.op_utils import _enqueue_cached, _make_ptr
from tmb.kernels.random.philox import curand4, curand_ctr, curand_key

# Threads per row block, for both kernels.
comptime MN_THREADS = 256
# Rows beyond this run grid-stride (CUDA/Metal cap grid.x generously, but
# one wave of blocks is plenty for a row-per-block reduction).
comptime MN_MAX_BLOCKS = 65535

# The flag bits `_check_kernel` sets (mirrored in tmb/ops/random.mojo).
comptime MN_BAD_NEGATIVE = 1
comptime MN_BAD_NONFINITE = 2
comptime MN_BAD_SUM = 4

comptime _TWO_POW_M24 = 5.9604644775390625e-08  # 2^-24


@always_inline
def mn_acc[dtype: DType]() -> DType:
    """ATen's accumulation type for the row sums."""
    return DType.float64 if dtype == DType.float64 else DType.float32


@always_inline
def _magnitude_vs_inf[acc: DType](v: Scalar[acc]) -> Int:
    """-1 finite, 0 infinite, 1 NaN -- read from the bits: a fast-math GPU
    compile may fold a float `v != v` to false."""
    comptime if acc == DType.float64:
        var m = bitcast[DType.uint64, 1](v) & 0x7FFF_FFFF_FFFF_FFFF
        comptime INF = UInt64(0x7FF0_0000_0000_0000)
        return -1 if m.lt(INF)[0] else (0 if m.eq(INF)[0] else 1)
    else:
        var m = bitcast[DType.uint32, 1](v) & 0x7FFF_FFFF
        comptime INF = UInt32(0x7F80_0000)
        return -1 if m.lt(INF)[0] else (0 if m.eq(INF)[0] else 1)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(MN_THREADS))
)
@__name(t"multinomial_check_{dtype}_t{MN_THREADS}")
def _check_kernel[
    dtype: DType
](
    flag: Pointer[Scalar[DType.int32], MutAnyOrigin],
    probs: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    n_arg: Int64,
):
    comptime ACC = mn_acc[dtype]()
    var rows = Int(rows_arg)
    var n = Int(n_arg)
    var s_sum = stack_allocation[
        MN_THREADS, ACC, address_space=AddressSpace.SHARED
    ]()
    var s_bad = stack_allocation[
        MN_THREADS, DType.int32, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    while row < rows:
        var total = Scalar[ACC](0)
        var bad = Int32(0)
        var i = tid
        while i < n:
            var v = probs[unsafe_offset=row * n + i].cast[ACC]()
            var kind = _magnitude_vs_inf(v)
            # CPU's sampler tests `val >= 0` first, so NaN and -inf report
            # as a negative entry and only +inf as the infinite one.
            if kind == 1 or v.lt(0)[0]:
                bad |= MN_BAD_NEGATIVE
            elif kind == 0:
                bad |= MN_BAD_NONFINITE
            else:
                total += v
            i += MN_THREADS
        s_sum[unsafe_offset=tid] = total
        s_bad[unsafe_offset=tid] = bad
        barrier()
        var width = MN_THREADS // 2
        while width > 0:
            if tid < width:
                s_sum[unsafe_offset=tid] += s_sum[unsafe_offset=tid + width]
                s_bad[unsafe_offset=tid] |= s_bad[unsafe_offset=tid + width]
            barrier()
            width >>= 1
        if tid == 0:
            var code = s_bad[unsafe_offset=0]
            if code == 0 and not s_sum[unsafe_offset=0].gt(0)[0]:
                code = MN_BAD_SUM
            if code != 0:
                flag[unsafe_offset=0] = code
        barrier()  # the next row reuses the shared arrays
        row += Int(grid_dim.x)


@always_inline
def _positive[dtype: DType](v: Scalar[dtype]) -> Bool:
    return v.cast[mn_acc[dtype]()]().gt(0)[0]


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(MN_THREADS))
)
@__name(t"multinomial_cdf_draw_{dtype}_t{MN_THREADS}")
def _draw_kernel[
    dtype: DType
](
    dst: Pointer[Scalar[DType.int64], MutAnyOrigin],
    cdf: Pointer[Scalar[mn_acc[dtype]()], MutAnyOrigin],
    probs: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    n_arg: Int64,
    n_sample_arg: Int64,
    seed: UInt64,
    offset: UInt64,
):
    comptime ACC = mn_acc[dtype]()
    var rows = Int(rows_arg)
    var n = Int(n_arg)
    var n_sample = Int(n_sample_arg)
    var key = curand_key(seed)
    var ctr = curand_ctr(offset, 0)
    var s_part = stack_allocation[
        MN_THREADS, ACC, address_space=AddressSpace.SHARED
    ]()
    var tid = Int(thread_idx.x)
    var chunk = (n + MN_THREADS - 1) // MN_THREADS
    var start = min(tid * chunk, n)
    var end = min(start + chunk, n)
    var row = Int(block_idx.x)
    while row < rows:
        var base = row * n
        var local = Scalar[ACC](0)
        for i in range(start, end):
            local += probs[unsafe_offset=base + i].cast[ACC]()
        s_part[unsafe_offset=tid] = local
        barrier()
        # Hillis-Steele inclusive scan of the per-thread chunk sums.
        var step = 1
        while step < MN_THREADS:
            var add = Scalar[ACC](0)
            if tid >= step:
                add = s_part[unsafe_offset=tid - step]
            barrier()
            s_part[unsafe_offset=tid] += add
            barrier()
            step <<= 1
        var running = Scalar[ACC](0)
        if tid > 0:
            running = s_part[unsafe_offset=tid - 1]
        for i in range(start, end):
            running += probs[unsafe_offset=base + i].cast[ACC]()
            cdf[unsafe_offset=base + i] = running
        barrier()  # the whole row's CDF is visible to the block

        var total = cdf[unsafe_offset=base + n - 1]
        var s = tid
        while s < n_sample:
            var e = row * n_sample + s
            var words = curand4(ctr, key, UInt64(e >> 2))
            var w = words[e & 3]
            var u = (w >> 8).cast[ACC]() * Scalar[ACC](_TWO_POW_M24)
            var x = u * total
            var lo = 0
            var hi = n - 1
            while lo < hi:
                var mid = (lo + hi) >> 1
                if cdf[unsafe_offset=base + mid].gt(x)[0]:
                    hi = mid
                else:
                    lo = mid + 1
            if not _positive(probs[unsafe_offset=base + lo]):
                var j = lo + 1
                while j < n and not _positive(probs[unsafe_offset=base + j]):
                    j += 1
                if j == n:
                    j = lo - 1
                    while j > 0 and not _positive(
                        probs[unsafe_offset=base + j]
                    ):
                        j -= 1
                lo = j
            dst[unsafe_offset=e] = Int64(lo)
            s += MN_THREADS
        barrier()  # the next row reuses the shared scan
        row += Int(grid_dim.x)


def enqueue_multinomial_check[
    dtype: DType
](ctx: DeviceContext, flag: Int, probs: Int, rows: Int, n: Int) raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[_check_kernel[dtype]](
            ctx,
            max(1, min(rows, MN_MAX_BLOCKS)),
            1,
            1,
            MN_THREADS,
            _make_ptr[DType.int32](flag).as_unsafe_any_origin(),
            _make_ptr[dtype](probs).as_unsafe_any_origin().as_imm(),
            Int64(rows),
            Int64(n),
        )


def enqueue_multinomial_draw[
    dtype: DType
](
    ctx: DeviceContext,
    dst: Int,
    cdf: Int,
    probs: Int,
    rows: Int,
    n: Int,
    n_sample: Int,
    seed: UInt64,
    offset: UInt64,
) raises:
    comptime if not has_accelerator():
        raise Error("no GPU accelerator available at compile time")
    else:
        _enqueue_cached[_draw_kernel[dtype]](
            ctx,
            max(1, min(rows, MN_MAX_BLOCKS)),
            1,
            1,
            MN_THREADS,
            _make_ptr[DType.int64](dst).as_unsafe_any_origin(),
            _make_ptr[mn_acc[dtype]()](cdf).as_unsafe_any_origin(),
            _make_ptr[dtype](probs).as_unsafe_any_origin().as_imm(),
            Int64(rows),
            Int64(n),
            Int64(n_sample),
            seed,
            offset,
        )
