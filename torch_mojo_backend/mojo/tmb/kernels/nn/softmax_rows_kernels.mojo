"""Row-wise softmax kernels, shared by the eager `SoftmaxSpec` / `SoftmaxRows`
ops and any MAX custom op that wants the same kernel.

They live apart from `tmb/kernels/nn/entry.mojo` for one reason: that module
carries the family's `@export tmb_call`, and one MAX compilation unit that
imports two such entry modules (a GEMM from `tmb/kernels/matmul/entry.mojo`
next to a softmax from the nn one) fails to compile on the duplicate export.
Kernel bodies go here; the `_go` glue and `tmb_call` stay in the entry.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from std.gpu.primitives import warp
from std.math import (
    ceildiv,
    exp,
    floor,
)
from std.sys.info import (
    has_accelerator,
    has_apple_gpu_accelerator,
    size_of,
)
from std.utils.coord import Coord
from std.utils.numerics import (
    min_finite,
    min_or_neg_inf,
)
from std.utils.static_tuple import StaticTuple
from layout import (
    TileTensor,
    row_major,
)
from nn.softmax import softmax

from tmb.kernels.random.dropout_kernels import _philox4x32_10
from tmb.kernels.common.op_utils import (
    Arg,
    _enqueue_cached,
    _make_ptr,
    _parallel_for,
    _raw_ctx,
    _raw_f64,
    _raw_int,
)


# ---------------------------------------------------------------------------
# Row-wise softmax with optional scaling and causal masking, for attention.
# Input is (rows, cols) where rows = batch * q_len. With causal=1, row r
# (query index r % q_len) only attends to columns j <= r % q_len — the
# top-left-aligned tril(ones(L, S)) mask that torch's sdpa is_causal=True
# specifies; masked columns get probability 0.
#
# The CUDA/ROCm GPU path delegates to modular's `nn.softmax.softmax`, which
# runs an online single-pass kernel (2 input reads + 1 write) — less HBM
# traffic than a hand-written 4-pass block kernel — and a warp-shuffle kernel
# for short rows (cols <= WARP_SIZE: 32 NVIDIA, 64 AMD). `scale` and the
# causal mask are folded into the input lambda: the value is read and scaled
# in float32 (for scale == 1 the round-trip back to `dtype` is exact), and
# masked columns are fed as -inf so their softmax weight is exactly 0,
# matching the CPU branch below. Because
# `allowed = min(cols, r % q_len + 1) >= 1`, no causal row is ever fully
# masked. The CPU MAX device keeps the explicit per-row loop below.
#
# On Apple GPUs the library kernel is ~10x off the bandwidth roofline (2.65 ms
# for 12288x256 f32 on this machine's ~100 GB/s part), so Metal gets custom
# kernels instead:
#
#   * `_softmax_rows_warp_kernel`: one simdgroup (32 lanes) per row, the whole
#     row held in registers (`_APPLE_SM_MAX_VPT` SIMD[V] slots per lane), so the row
#     is read exactly once and only up to the causal boundary — shuffles for
#     the max/sum, no shared memory, no re-read. Handles rows up to
#     32 * _APPLE_SM_MAX_VPT * V elements; V = 16B vectors when the pointers are
#     16B-aligned and cols is a multiple of V, else scalar (V = 1).
#   * `_softmax_rows_block_kernel`: generic fallback for longer or unaligned
#     rows — one thread block per row, online (m, s) accumulation in one read
#     (see `_log_softmax_rows_block_kernel` for the min_finite-not-inf
#     rationale), block.max/block.sum combine, then an output pass that
#     re-reads via the cache and zero-fills past the causal boundary.
# ---------------------------------------------------------------------------

comptime _APPLE_SM_WARPS_PER_BLOCK = 8
comptime _APPLE_SM_MAX_VPT = 8  # per-lane register slots in the warp kernel
comptime _APPLE_SM_BIG_ROW_BYTES = 25_000  # 1024-thread blocks above this


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_APPLE_SM_WARPS_PER_BLOCK * WARP_SIZE)
    )
)
@__name(t"softmax_rows_warp_{dtype}_{V}")
def _softmax_rows_warp_kernel[
    dtype: DType, V: Int
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    scale: Float32,
    causal_arg: Int64,
    q_len_arg: Int64,
):
    # One warp per row; requires cols % V == 0 and
    # cols <= WARP_SIZE * _APPLE_SM_MAX_VPT * V (host-checked). Lanes at or past
    # the causal boundary carry min_finite (finite, so `exp` below stays in
    # range even when the row max is tiny) and are explicitly zeroed before
    # the sum, so no sentinel arithmetic can leak into the result.
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var causal = Int(causal_arg)
    var q_len = Int(q_len_arg)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _APPLE_SM_WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _APPLE_SM_WARPS_PER_BLOCK
    var n_vec = cols // V
    while row < rows:
        var base = row * cols
        var allowed = cols
        if causal != 0:
            allowed = min(cols, row % q_len + 1)

        # Read pass: the row (up to the causal boundary) into registers,
        # scaled, with a per-lane running max.
        var vals = StaticTuple[SIMD[DType.float32, V], _APPLE_SM_MAX_VPT]()
        var m_t = min_finite[DType.float32]()
        comptime for k in range(_APPLE_SM_MAX_VPT):
            var j0 = (lane + k * WARP_SIZE) * V
            var x = SIMD[DType.float32, V](min_finite[DType.float32]())
            if j0 < allowed:  # implies j0 < cols, i.e. a full in-row vector
                var raw = (
                    in_ptr.unsafe_load[width=V, alignment=V * size_of[dtype]()](
                        base + j0
                    ).cast[DType.float32]()
                    * scale
                )
                if j0 + V > allowed:
                    comptime for li in range(V):
                        if j0 + li < allowed:
                            x[li] = raw[li]
                else:
                    x = raw
            vals[k] = x
            m_t = max(m_t, x.reduce_max())
        var row_m = warp.max(m_t)

        # exp pass in registers; masked lanes forced to exactly 0.
        var s_t = Float32(0)
        comptime for k in range(_APPLE_SM_MAX_VPT):
            var j0 = (lane + k * WARP_SIZE) * V
            var e = SIMD[DType.float32, V](0)
            if j0 < allowed:
                e = exp(vals[k] - row_m)
                if j0 + V > allowed:
                    comptime for li in range(V):
                        if j0 + li >= allowed:
                            e[li] = 0
            vals[k] = e
            s_t += e.reduce_add()
        var inv = Float32(1) / warp.sum(s_t)

        # Write pass: probabilities, zeros past the causal boundary.
        comptime for k in range(_APPLE_SM_MAX_VPT):
            var v = lane + k * WARP_SIZE
            if v < n_vec:
                out_ptr.unsafe_store[width=V, alignment=V * size_of[dtype]()](
                    base + v * V, (vals[k] * inv).cast[dtype]()
                )
        row += row_stride


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(threads))
)
@__name(t"softmax_rows_block_{dtype}_{threads}_{vec}")
def _softmax_rows_block_kernel[
    dtype: DType, threads: Int, vec: Bool
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    scale: Float32,
    causal_arg: Int64,
    q_len_arg: Int64,
):
    # One thread block per row (grid-stride over rows), any rows/cols >= 1.
    # `vec = True` requires 16B-aligned base pointers (host-checked); rows
    # whose start is then still unaligned get a scalar head, exactly like
    # `_log_softmax_rows_block_kernel`.
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var causal = Int(causal_arg)
    var q_len = Int(q_len_arg)
    comptime V = 16 // size_of[dtype]()
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    while row < rows:
        var base = row * cols
        var allowed = cols
        if causal != 0:
            allowed = min(cols, row % q_len + 1)

        # ---- Pass 1: online (max, sum) over the allowed prefix, read once.
        var m_t = min_finite[DType.float32]()
        var s_t = Float32(0)

        comptime if vec:
            var head = (V - (base % V)) % V
            if head > cols:
                head = cols
            var n_vec_a = 0
            if allowed > head:
                n_vec_a = (allowed - head) // V

            var m_vec = SIMD[DType.float32, V](min_finite[DType.float32]())
            var s_vec = SIMD[DType.float32, V](0)
            var v = tid
            while v < n_vec_a:
                var x = (
                    in_ptr.unsafe_load[width=V, alignment=16](
                        base + head + v * V
                    ).cast[DType.float32]()
                    * scale
                )
                var nm = max(m_vec, x)
                s_vec = s_vec * exp(m_vec - nm) + exp(x - nm)
                m_vec = nm
                v += threads
            m_t = m_vec.reduce_max()
            s_t = (s_vec * exp(m_vec - m_t)).reduce_add()

            # Scalar head plus the partial vector at the causal boundary.
            var jh = tid
            while jh < min(head, allowed):
                var x = (
                    in_ptr[unsafe_offset=base + jh].cast[DType.float32]()
                    * scale
                )
                var nm = max(m_t, x)
                s_t = s_t * exp(m_t - nm) + exp(x - nm)
                m_t = nm
                jh += threads
            var jt = head + n_vec_a * V + tid
            while jt < allowed:
                var x = (
                    in_ptr[unsafe_offset=base + jt].cast[DType.float32]()
                    * scale
                )
                var nm = max(m_t, x)
                s_t = s_t * exp(m_t - nm) + exp(x - nm)
                m_t = nm
                jt += threads
        else:
            var j = tid
            while j < allowed:
                var x = (
                    in_ptr[unsafe_offset=base + j].cast[DType.float32]() * scale
                )
                var nm = max(m_t, x)
                s_t = s_t * exp(m_t - nm) + exp(x - nm)
                m_t = nm
                j += threads

        var block_m = block.max[block_size=threads, broadcast=True](m_t)
        var block_s = block.sum[block_size=threads, broadcast=True](
            s_t * exp(m_t - block_m)
        )
        var inv = Float32(1) / block_s

        # ---- Pass 2: write probabilities, zeros past the boundary. The
        # allowed prefix is re-read through the cache.
        comptime if vec:
            var head = (V - (base % V)) % V
            if head > cols:
                head = cols
            var n_vec_c = (cols - head) // V
            var jh = tid
            while jh < head:
                var y = Scalar[dtype](0)
                if jh < allowed:
                    var x = (
                        in_ptr[unsafe_offset=base + jh].cast[DType.float32]()
                        * scale
                    )
                    y = (exp(x - block_m) * inv).cast[dtype]()
                out_ptr[unsafe_offset=base + jh] = y
                jh += threads
            var v = tid
            while v < n_vec_c:
                var j0 = head + v * V
                var y = SIMD[dtype, V](0)
                if j0 < allowed:
                    var x = (
                        in_ptr.unsafe_load[width=V, alignment=16](
                            base + j0
                        ).cast[DType.float32]()
                        * scale
                    )
                    var e = exp(x - block_m) * inv
                    if j0 + V > allowed:
                        comptime for li in range(V):
                            if j0 + li >= allowed:
                                e[li] = 0
                    y = e.cast[dtype]()
                out_ptr.unsafe_store[width=V, alignment=16](base + j0, y)
                v += threads
            var jt = head + n_vec_c * V + tid
            while jt < cols:
                var y = Scalar[dtype](0)
                if jt < allowed:
                    var x = (
                        in_ptr[unsafe_offset=base + jt].cast[DType.float32]()
                        * scale
                    )
                    y = (exp(x - block_m) * inv).cast[dtype]()
                out_ptr[unsafe_offset=base + jt] = y
                jt += threads
        else:
            var j = tid
            while j < allowed:
                var x = (
                    in_ptr[unsafe_offset=base + j].cast[DType.float32]() * scale
                )
                out_ptr[unsafe_offset=base + j] = (exp(x - block_m) * inv).cast[
                    dtype
                ]()
                j += threads
            var jz = allowed + tid
            while jz < cols:
                out_ptr[unsafe_offset=base + jz] = Scalar[dtype](0)
                jz += threads

        row += Int(grid_dim.x)


@always_inline
def _softmax_rows_apple[
    dtype: DType
](
    out_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: Pointer[Scalar[dtype], MutUntrackedOrigin],
    rows: Int,
    cols: Int,
    scale: Float32,
    causal: Int,
    q_len: Int,
    ctx: DeviceContext,
) raises:
    """Regime dispatch for the Apple-GPU row-softmax kernels."""
    comptime V = 16 // size_of[dtype]()
    var mout = out_ptr.as_unsafe_any_origin()
    var min_ = in_ptr.as_unsafe_any_origin().as_imm()
    var aligned = Int(out_ptr) % 16 == 0 and Int(in_ptr) % 16 == 0
    var warp_grid = min(
        (rows + _APPLE_SM_WARPS_PER_BLOCK - 1) // _APPLE_SM_WARPS_PER_BLOCK,
        32768,
    )
    if aligned and cols % V == 0 and cols <= WARP_SIZE * _APPLE_SM_MAX_VPT * V:
        _enqueue_cached[_softmax_rows_warp_kernel[dtype, V]](
            ctx,
            warp_grid,
            1,
            1,
            _APPLE_SM_WARPS_PER_BLOCK * WARP_SIZE,
            mout,
            min_,
            Int64(rows),
            Int64(cols),
            scale,
            Int64(causal),
            Int64(q_len),
        )
    elif cols <= WARP_SIZE * _APPLE_SM_MAX_VPT:
        _enqueue_cached[_softmax_rows_warp_kernel[dtype, 1]](
            ctx,
            warp_grid,
            1,
            1,
            _APPLE_SM_WARPS_PER_BLOCK * WARP_SIZE,
            mout,
            min_,
            Int64(rows),
            Int64(cols),
            scale,
            Int64(causal),
            Int64(q_len),
        )
    else:
        var block_grid = min(rows, 32768)
        if cols * size_of[dtype]() > _APPLE_SM_BIG_ROW_BYTES:
            if aligned:
                _enqueue_cached[_softmax_rows_block_kernel[dtype, 1024, True]](
                    ctx,
                    block_grid,
                    1,
                    1,
                    1024,
                    mout,
                    min_,
                    Int64(rows),
                    Int64(cols),
                    scale,
                    Int64(causal),
                    Int64(q_len),
                )
            else:
                _enqueue_cached[_softmax_rows_block_kernel[dtype, 1024, False]](
                    ctx,
                    block_grid,
                    1,
                    1,
                    1024,
                    mout,
                    min_,
                    Int64(rows),
                    Int64(cols),
                    scale,
                    Int64(causal),
                    Int64(q_len),
                )
        else:
            if aligned:
                _enqueue_cached[_softmax_rows_block_kernel[dtype, 256, True]](
                    ctx,
                    block_grid,
                    1,
                    1,
                    256,
                    mout,
                    min_,
                    Int64(rows),
                    Int64(cols),
                    scale,
                    Int64(causal),
                    Int64(q_len),
                )
            else:
                _enqueue_cached[_softmax_rows_block_kernel[dtype, 256, False]](
                    ctx,
                    block_grid,
                    1,
                    1,
                    256,
                    mout,
                    min_,
                    Int64(rows),
                    Int64(cols),
                    scale,
                    Int64(causal),
                    Int64(q_len),
                )


# ---------------------------------------------------------------------------
# Fused causal softmax + native dropout for the Apple SDPA forward: one
# launch produces the pre-dropout probabilities (saved for backward), the
# dropped/rescaled probabilities (consumed by the value BMM), and the bool
# keep-mask. The probabilities never make a DRAM round-trip between softmax
# and dropout, saving one full read of the (rows, cols) matrix plus a launch
# versus SoftmaxRows + NativeDropoutF32.
#
# RNG contract: identical to native_dropout_kernels (Philox4x32-10, element
# i belongs to block (base_offset + i // 4), lane i % 4, keep iff
# u32 < floor(Float32(1 - p) * 2^32)). cols % 4 == 0 and 16B-aligned
# pointers (host-checked) make each vec4 exactly one Philox block, so the
# mask and dropped values are byte-identical to the unfused path — dropout
# is applied across the whole row, including the zeros past the causal
# boundary, exactly like composed dropout.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_APPLE_SM_WARPS_PER_BLOCK * WARP_SIZE)
    )
)
@__name("softmax_rows_dropout_warp_f32")
def _softmax_rows_dropout_warp_kernel(
    probs_ptr: Pointer[Scalar[DType.float32], MutAnyOrigin],
    pdrop_ptr: Pointer[Scalar[DType.float32], MutAnyOrigin],
    mask_ptr: Pointer[Scalar[DType.bool], MutAnyOrigin],
    in_ptr: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    scale: Float32,
    causal_arg: Int64,
    q_len_arg: Int64,
    seed: UInt64,
    base_offset: UInt64,
    threshold: UInt64,
    keep_scale: Float32,
):
    # Same structure as `_softmax_rows_warp_kernel[float32, 4]` (see there
    # for the masking rationale), plus the Philox dropout epilogue.
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var causal = Int(causal_arg)
    var q_len = Int(q_len_arg)
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _APPLE_SM_WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _APPLE_SM_WARPS_PER_BLOCK
    var n_vec = cols // 4
    while row < rows:
        var base = row * cols
        var allowed = cols
        if causal != 0:
            allowed = min(cols, row % q_len + 1)

        var vals = StaticTuple[SIMD[DType.float32, 4], _APPLE_SM_MAX_VPT]()
        var m_t = min_finite[DType.float32]()
        comptime for k in range(_APPLE_SM_MAX_VPT):
            var j0 = (lane + k * WARP_SIZE) * 4
            var x = SIMD[DType.float32, 4](min_finite[DType.float32]())
            if j0 < allowed:
                var raw = (
                    in_ptr.unsafe_load[width=4, alignment=16](base + j0) * scale
                )
                if j0 + 4 > allowed:
                    comptime for li in range(4):
                        if j0 + li < allowed:
                            x[li] = raw[li]
                else:
                    x = raw
            vals[k] = x
            m_t = max(m_t, x.reduce_max())
        var row_m = warp.max(m_t)

        var s_t = Float32(0)
        comptime for k in range(_APPLE_SM_MAX_VPT):
            var j0 = (lane + k * WARP_SIZE) * 4
            var e = SIMD[DType.float32, 4](0)
            if j0 < allowed:
                e = exp(vals[k] - row_m)
                if j0 + 4 > allowed:
                    comptime for li in range(4):
                        if j0 + li >= allowed:
                            e[li] = 0
            vals[k] = e
            s_t += e.reduce_add()
        var inv = Float32(1) / warp.sum(s_t)

        comptime for k in range(_APPLE_SM_MAX_VPT):
            var v = lane + k * WARP_SIZE
            if v < n_vec:
                var y = vals[k] * inv
                probs_ptr.unsafe_store[width=4, alignment=16](base + v * 4, y)
                var rnd = _philox4x32_10(
                    base_offset + UInt64(base // 4 + v), seed
                )
                var keep_bits = (
                    rnd.cast[DType.uint64]() - SIMD[DType.uint64, 4](threshold)
                ) >> 63
                pdrop_ptr.unsafe_store[width=4, alignment=16](
                    base + v * 4,
                    y * keep_bits.cast[DType.float32]() * keep_scale,
                )
                mask_ptr.unsafe_bitcast[Scalar[DType.uint8]]().unsafe_store[
                    alignment=4
                ](base + v * 4, keep_bits.cast[DType.uint8]())
        row += row_stride


def _softmax_rows_dropout_go(
    probs_ptr_obj: Arg,
    pdrop_ptr_obj: Arg,
    mask_ptr_obj: Arg,
    in_ptr_obj: Arg,
    rows_obj: Arg,
    cols_obj: Arg,
    scale_obj: Arg,
    causal_obj: Arg,
    q_len_obj: Arg,
    p_obj: Arg,
    seed_lo_obj: Arg,
    seed_hi_obj: Arg,
    offset_lo_obj: Arg,
    offset_hi_obj: Arg,
    device_context_ptr: Arg,
) raises:
    var probs_addr = _raw_int(probs_ptr_obj)
    var pdrop_addr = _raw_int(pdrop_ptr_obj)
    var mask_addr = _raw_int(mask_ptr_obj)
    var in_addr = _raw_int(in_ptr_obj)
    var rows = _raw_int(rows_obj)
    var cols = _raw_int(cols_obj)
    var scale = Float32(_raw_f64(scale_obj))
    var causal = _raw_int(causal_obj)
    var q_len = _raw_int(q_len_obj)
    var p = _raw_f64(p_obj)
    var seed = UInt64(_raw_int(seed_lo_obj)) | (
        UInt64(_raw_int(seed_hi_obj)) << 32
    )
    var base_offset = UInt64(_raw_int(offset_lo_obj)) | (
        UInt64(_raw_int(offset_hi_obj)) << 32
    )
    var ctx = _raw_ctx(device_context_ptr)

    enqueue_softmax_rows_dropout_f32(
        probs_addr,
        pdrop_addr,
        mask_addr,
        in_addr,
        rows,
        cols,
        scale,
        causal,
        q_len,
        p,
        seed,
        base_offset,
        ctx,
    )


@always_inline
def enqueue_softmax_rows_dropout_f32(
    probs_addr: Int,
    pdrop_addr: Int,
    mask_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    scale: Float32,
    causal: Int,
    q_len: Int,
    p: Float64,
    seed: UInt64,
    base_offset: UInt64,
    ctx: DeviceContext,
) raises:
    comptime if has_apple_gpu_accelerator():
        # The Python caller gates on all of this; re-checked here because the
        # dispatcher cannot report failure.
        if (
            not (p > 0.0 and p < 1.0)
            or cols % 4 != 0
            or cols > WARP_SIZE * _APPLE_SM_MAX_VPT * 4
            or probs_addr % 16 != 0
            or pdrop_addr % 16 != 0
            or mask_addr % 4 != 0
            or in_addr % 16 != 0
        ):
            raise Error("SoftmaxRowsDropoutF32: unsupported configuration")
        # Same threshold arithmetic as native_dropout_kernels (Float64
        # subtraction, one narrowing, all 32 random bits compared).
        var keep_f32 = Float32(1.0 - p)
        var keep_scale = Float32(1.0) / keep_f32
        var threshold = (Float64(keep_f32) * 4294967296.0).cast[DType.uint64]()
        var warp_grid = min(
            (rows + _APPLE_SM_WARPS_PER_BLOCK - 1) // _APPLE_SM_WARPS_PER_BLOCK,
            32768,
        )
        _enqueue_cached[_softmax_rows_dropout_warp_kernel](
            ctx,
            warp_grid,
            1,
            1,
            _APPLE_SM_WARPS_PER_BLOCK * WARP_SIZE,
            _make_ptr[DType.float32](probs_addr).as_unsafe_any_origin(),
            _make_ptr[DType.float32](pdrop_addr).as_unsafe_any_origin(),
            _make_ptr[DType.bool](mask_addr).as_unsafe_any_origin(),
            _make_ptr[DType.float32](in_addr).as_unsafe_any_origin().as_imm(),
            Int64(rows),
            Int64(cols),
            scale,
            Int64(causal),
            Int64(q_len),
            seed,
            base_offset,
            threshold,
            keep_scale,
        )
    else:
        raise Error("SoftmaxRowsDropoutF32 is Apple-GPU only")


# One warp per row, so a causal row's shorter extent costs proportionally less
# and the hardware balances the very uneven per-row work across blocks.
comptime _SM_WARPS_PER_BLOCK = 8 if WARP_SIZE <= 32 else 4
comptime _SM_BLOCK = _SM_WARPS_PER_BLOCK * WARP_SIZE
comptime _SM_MAX_GRID = 1 << 20
comptime _SM_VECTOR_BYTES = 16


@always_inline
def _softmax_warp_rows[
    dtype: DType, causal: Bool, VEC: Int
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    scale: Float32,
    q_len: Int,
):
    """Row softmax with the causal mask folded into the row extent.

    A causal row's masked columns are fed `-inf` by the reference kernel and
    come out exactly zero, so the extent `min(cols, query + 1)` carries all the
    information: this kernel reads only that prefix and writes the zeros
    directly.  The reduction is an online single-read max+sum in float32; the
    store pass re-reads the row from L1/L2.
    """
    comptime F32 = DType.float32
    comptime ALIGN = VEC * size_of[dtype]()
    var lane = Int(lane_id())
    var row = Int(block_idx.x) * _SM_WARPS_PER_BLOCK + Int(warp_id())
    var row_stride = Int(grid_dim.x) * _SM_WARPS_PER_BLOCK

    while row < rows:
        var limit = cols
        comptime if causal:
            limit = min(cols, row % q_len + 1)
        var base = row * cols
        var vec_limit = (limit // VEC) * VEC

        # Pass 1: online max and sum over the live prefix.  The running max
        # starts at the most negative *finite* float, not at -inf: a lane or a
        # vector slot with no live column keeps the sentinel, and the rescaling
        # below subtracts two maxima, so -inf would give `-inf - -inf = NaN` and
        # poison the whole row through the warp reduction.  With a finite
        # sentinel the same expression is `exp(0) = 1` times a zero sum.
        var run_max = SIMD[F32, VEC](Float32.MIN_FINITE)
        var run_sum = SIMD[F32, VEC](0.0)
        var col = lane * VEC
        while col < vec_limit:
            var x = (
                in_ptr.unsafe_load[width=VEC, alignment=ALIGN](base + col).cast[
                    F32
                ]()
                * scale
            )
            var new_max = max(run_max, x)
            run_sum = run_sum * exp(run_max - new_max) + exp(x - new_max)
            run_max = new_max
            col += WARP_SIZE * VEC
        # Ragged remainder: fewer than VEC columns, at most one per lane.
        var tail = vec_limit + lane
        var tail_max = Float32.MIN_FINITE
        var tail_sum = Float32(0.0)
        if tail < limit:
            tail_max = in_ptr[unsafe_offset=base + tail].cast[F32]() * scale
            tail_sum = 1.0

        # Lane-local then warp-wide combination of the (max, sum) pairs.
        var lane_max = max(run_max.reduce_max(), tail_max)
        var lane_sum = (run_sum * exp(run_max - lane_max)).reduce_add() + (
            tail_sum * exp(tail_max - lane_max)
        )
        # The row bound is warp-uniform, so every lane reaches the shuffles.
        var row_max = warp.max(lane_max)
        var inv = 1.0 / warp.sum(lane_sum * exp(lane_max - row_max))

        # Pass 2: the probabilities.  The row is L1/L2-resident from pass 1.
        col = lane * VEC
        while col < vec_limit:
            var x = (
                in_ptr.unsafe_load[width=VEC, alignment=ALIGN](base + col).cast[
                    F32
                ]()
                * scale
            )
            out_ptr.unsafe_store[width=VEC, alignment=ALIGN](
                base + col, (exp(x - row_max) * inv).cast[dtype]()
            )
            col += WARP_SIZE * VEC
        if tail < limit:
            out_ptr[unsafe_offset=base + tail] = (
                exp(
                    in_ptr[unsafe_offset=base + tail].cast[F32]() * scale
                    - row_max
                )
                * inv
            ).cast[dtype]()

        comptime if causal:
            var zero_head = min(cols, ceildiv(limit, VEC) * VEC)
            if limit + lane < zero_head:
                out_ptr[unsafe_offset=base + limit + lane] = Scalar[dtype](0)
            col = zero_head + lane * VEC
            while col + VEC <= cols:
                out_ptr.unsafe_store[width=VEC, alignment=ALIGN](
                    base + col, SIMD[dtype, VEC](Scalar[dtype](0))
                )
                col += WARP_SIZE * VEC
            # `cols` and `zero_head` are both multiples of VEC in the wide
            # regime; VEC == 1 makes `zero_head == limit`, so nothing is left.
        row += row_stride


@__name(t"softmax_rows_warp_{dtype}_c{causal}_v{VEC}")
def _softmax_warp_kernel[
    dtype: DType, causal: Bool, VEC: Int
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows_arg: Int64,
    cols_arg: Int64,
    scale: Float32,
    q_len_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var rows = Int(rows_arg)
    var cols = Int(cols_arg)
    var q_len = Int(q_len_arg)
    _softmax_warp_rows[dtype, causal, VEC](
        out_ptr, in_ptr, rows, cols, scale, q_len
    )


@always_inline
def _enqueue_softmax_warp[
    dtype: DType, causal: Bool, VEC: Int
](
    out_ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: Pointer[Scalar[dtype], ImmutAnyOrigin],
    rows: Int,
    cols: Int,
    scale: Float32,
    q_len: Int,
    ctx: DeviceContext,
) raises:
    _enqueue_cached[_softmax_warp_kernel[dtype, causal, VEC]](
        ctx,
        min(ceildiv(rows, _SM_WARPS_PER_BLOCK), _SM_MAX_GRID),
        1,
        1,
        _SM_BLOCK,
        out_ptr,
        in_ptr,
        Int64(rows),
        Int64(cols),
        scale,
        Int64(q_len),
    )


@always_inline
def _softmax_rows[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    rows: Int,
    cols: Int,
    scale: Float32,
    causal: Int,
    q_len: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    comptime if has_apple_gpu_accelerator():
        _softmax_rows_apple[dtype](
            out_ptr, in_ptr, rows, cols, scale, causal, q_len, ctx
        )
    elif has_accelerator():
        # Causal regime: half the score matrix is masked and the reference
        # kernel still reads, exponentiates and writes all of it.  A
        # warp-per-row kernel whose extent is the live prefix does the same
        # arithmetic over half the bytes.  Rows must be long enough for a
        # warp to vectorize and short enough that one warp per row is not
        # itself the bottleneck; both are runtime comparisons.
        comptime WIDE = _SM_VECTOR_BYTES // size_of[dtype]()
        if causal != 0 and cols <= 8192 and rows >= 256:
            var wide_ok = (
                cols % WIDE == 0
                and Int(out_ptr) % _SM_VECTOR_BYTES == 0
                and Int(in_ptr) % _SM_VECTOR_BYTES == 0
            )
            if wide_ok:
                _enqueue_softmax_warp[dtype, True, WIDE](
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_imm(),
                    rows,
                    cols,
                    scale,
                    q_len,
                    ctx,
                )
            else:
                _enqueue_softmax_warp[dtype, True, 1](
                    out_ptr.as_unsafe_any_origin(),
                    in_ptr.as_unsafe_any_origin().as_imm(),
                    rows,
                    cols,
                    scale,
                    q_len,
                    ctx,
                )
            return

        @parameter
        @always_inline
        @__copy_capture(in_ptr)
        def input_fn[
            _simd_width: Int
        ](coords: Coord) -> SIMD[dtype, _simd_width]:
            var r = Int(coords[0].value())
            var c = Int(coords[1].value())
            var v = (
                in_ptr.unsafe_load[width=_simd_width](r * cols + c).cast[
                    DType.float32
                ]()
                * scale
            )
            if causal != 0:
                var allowed = min(cols, r % q_len + 1)

                comptime for lane in range(_simd_width):
                    if c + lane >= allowed:
                        v[lane] = min_or_neg_inf[DType.float32]()
            # Known trade-off: for float16 input with scale > 1 this
            # f32 -> dtype round-trip can overflow to +inf where the old
            # all-f32 kernel didn't (unreachable with the default
            # 1/sqrt(head_dim) scales).
            return v.cast[dtype]()

        softmax[dtype, 1, 2, input_fn, target="gpu"](
            Coord(rows, cols),
            TileTensor(out_ptr, row_major(rows, cols)),
            1,
            ctx,
        )
    else:
        raise Error("no GPU accelerator available at compile time")
