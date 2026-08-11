"""Pure-Mojo Apple simdgroup-matrix transposed-A batched GEMM for eager SDPA
backward.

Computes, per batch slice ``z``::

    C[z, s, d] = sum_l A[z, l, s] * B[z, l, d]        (C = A^T @ B)

with A stored (k x m) row-major and B stored (k x n) row-major — exactly the
layouts SDPA backward has on hand for ``dV = P_drop^T @ dO`` and
``dK = dS^T @ Q``. The logical A transpose happens in the fragment loads
(per-lane scalar pairs, the same pattern the fat Apple GEMM uses for a
transposed B), so no permute copies are ever materialized.

Two SDPA-specific fusions:

* ``MASKED``: every A element is multiplied on load by a bool keep-mask of
  the same (k x m) layout and by ``drop_scale`` — reconstructing
  ``P_drop = P * mask * drop_scale`` for dV without a dropout-backward pass
  or an intermediate buffer.
* ``CAUSAL``: A comes from a causal SDPA matrix (P or dScores) whose row l
  is zero for columns s > l, so the l-reduction for the output row tile
  [row_base, row_base + SG_M) starts at l = row_base — halving A/B traffic
  and FLOPs at q_len == kv_len. Requires A's rows to be zero-filled past
  the causal boundary at least up to the next multiple of SG_M (the Apple
  causal softmax zero-fills whole rows; the causal fused dScores kernel
  zero-fills a 64-aligned band).

One threadgroup owns a 64x64 C tile (4 simdgroups, each a 32x32 subtile of
8x8 fragments), batched via ``block_idx.z``. All shapes stay runtime values
with fully guarded ragged edges; only the tile geometry is comptime. Apple
GPU only — the 8x8 simdgroup-matrix primitive does not exist elsewhere, and
other targets raise so callers keep their composed route.
"""

from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys import llvm_intrinsic
from std.sys.info import has_apple_gpu_accelerator
from std.utils.static_tuple import StaticTuple

from op_utils import _enqueue_cached

# Apple 8x8 simdgroup-matrix primitives (vendored from
# linalg.matmul.gpu.apple.matmul_8x8, whose helpers are not exported —
# matmul_ops.mojo carries the same vendored copy).
comptime MMA8_DIM = 8
comptime FRAG8 = 2  # 8x8 = 64 elems / 32 lanes = 2 per lane


@always_inline
def _frag8_layout(lane: Int) -> Tuple[Int, Int]:
    """Apple 8x8 simdgroup-matrix per-lane layout: lane owns
    (row, col_base) and (row, col_base + 1)."""
    return (
        ((lane & 6) >> 1) + ((lane & 16) >> 2),
        ((lane & 1) << 1) + ((lane & 8) >> 1),
    )


@always_inline
def _mma8x8(
    a: SIMD[DType.float32, FRAG8],
    b: SIMD[DType.float32, FRAG8],
    c: SIMD[DType.float32, FRAG8],
) -> SIMD[DType.float32, FRAG8]:
    """One 8x8x8 simdgroup-matrix multiply-accumulate: D = A @ B + C."""
    return llvm_intrinsic[
        "llvm.air.simdgroup_matrix_8x8_multiply_accumulate",
        SIMD[DType.float32, FRAG8],
    ](a, b, c)


comptime _BLOCK_M = 64
comptime _BLOCK_N = 64
comptime _SG_COLS = 2
comptime _SG_M = 32
comptime _SG_N = 32
comptime _NT_M = _SG_M // MMA8_DIM
comptime _NT_N = _SG_N // MMA8_DIM
comptime _THREADS = 128


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_THREADS))
)
@__name(t"sdpa_ta_gemm_f32_m{MASKED}_c{CAUSAL}")
def _sdpa_ta_gemm_kernel[
    MASKED: Bool, CAUSAL: Bool
](
    c_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    mask_base: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    m_arg: Int64,
    n_arg: Int64,
    k_arg: Int64,
    drop_scale: Float32,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var m = Int(m_arg)
    var n = Int(n_arg)
    var k = Int(k_arg)
    comptime F32 = DType.float32

    var bz = Int(block_idx.z)
    var c_ptr = c_base + bz * m * n
    var a_ptr = a_base + bz * k * m
    var b_ptr = b_base + bz * k * n
    var mask_ptr = mask_base + bz * k * m

    var lane = Int(lane_id())
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(warp_id())
    var row_base = Int(block_idx.y) * _BLOCK_M + (sg // _SG_COLS) * _SG_M
    var col_base = Int(block_idx.x) * _BLOCK_N + (sg % _SG_COLS) * _SG_N
    var interior = (row_base + _SG_M <= m) and (col_base + _SG_N <= n)

    # Causal: A columns [row_base, row_base + SG_M) are zero for every
    # reduction row l < row_base, so start there. row_base is a multiple of
    # SG_M, keeping the start 8-slab-aligned.
    var k_start = 0
    comptime if CAUSAL:
        k_start = min(k, row_base)

    var accum = InlineArray[SIMD[F32, FRAG8], _NT_M * _NT_N](
        fill=SIMD[F32, FRAG8](0)
    )

    # One 8-slab with every bound checked: ragged M/N subtiles and the K
    # tail. Interior full slabs never come through here.
    @always_inline
    @parameter
    def _slab_guarded(
        kk: Int, mut acc: InlineArray[SIMD[F32, FRAG8], _NT_M * _NT_N]
    ):
        var afrag = InlineArray[SIMD[F32, FRAG8], _NT_M](uninitialized=True)
        comptime for mi in range(_NT_M):
            # Logical A row = stored A column index.
            var grow = row_base + mi * MMA8_DIM + frow
            var af = SIMD[F32, FRAG8](0)
            if grow < m:
                comptime for s in range(FRAG8):
                    var kl = kk + fcol + s
                    if kl < k:
                        var av = a_ptr[kl * m + grow]
                        comptime if MASKED:
                            av = (
                                av
                                * mask_ptr[kl * m + grow].cast[F32]()
                                * drop_scale
                            )
                        af[s] = av
            afrag[mi] = af
        var bfrag = InlineArray[SIMD[F32, FRAG8], _NT_N](uninitialized=True)
        comptime for ni in range(_NT_N):
            var bf = SIMD[F32, FRAG8](0)
            if kk + frow < k:
                comptime for s in range(FRAG8):
                    var gj = col_base + ni * MMA8_DIM + fcol + s
                    if gj < n:
                        bf[s] = b_ptr[(kk + frow) * n + gj]
            bfrag[ni] = bf
        comptime for mi in range(_NT_M):
            comptime for ni in range(_NT_N):
                acc[mi * _NT_N + ni] = _mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * _NT_N + ni]
                )

    # Unguarded fragment loads for one interior 8-slab. The logical A
    # transpose makes A's two fragment slots differ by a whole stored row
    # (stride m) — per-lane scalar pairs, like the fat kernel's transposed B.
    @always_inline
    @parameter
    def _load_a_fast(
        ap0: UnsafePointer[Scalar[F32], ImmutAnyOrigin],
        mp0: UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin],
    ) -> InlineArray[SIMD[F32, FRAG8], _NT_M]:
        var afrag = InlineArray[SIMD[F32, FRAG8], _NT_M](uninitialized=True)
        comptime for mi in range(_NT_M):
            var p = ap0 + mi * MMA8_DIM
            var af = SIMD[F32, FRAG8](0)
            comptime for s in range(FRAG8):
                af[s] = p[s * m]
            comptime if MASKED:
                var mp = mp0 + mi * MMA8_DIM
                comptime for s in range(FRAG8):
                    af[s] = af[s] * mp[s * m].cast[F32]() * drop_scale
            afrag[mi] = af
        return afrag^

    @always_inline
    @parameter
    def _load_b_fast(
        bp0: UnsafePointer[Scalar[F32], ImmutAnyOrigin],
    ) -> InlineArray[SIMD[F32, FRAG8], _NT_N]:
        var bfrag = InlineArray[SIMD[F32, FRAG8], _NT_N](uninitialized=True)
        comptime for ni in range(_NT_N):
            bfrag[ni] = (bp0 + ni * MMA8_DIM).load[width=FRAG8]()
        return bfrag^

    @always_inline
    @parameter
    def _mma_block(
        afrag: InlineArray[SIMD[F32, FRAG8], _NT_M],
        bfrag: InlineArray[SIMD[F32, FRAG8], _NT_N],
        mut acc: InlineArray[SIMD[F32, FRAG8], _NT_M * _NT_N],
    ):
        comptime for mi in range(_NT_M):
            comptime for ni in range(_NT_N):
                acc[mi * _NT_N + ni] = _mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * _NT_N + ni]
                )

    var full_end = k_start + ((k - k_start) // MMA8_DIM) * MMA8_DIM
    if interior:
        # Software-pipelined pointer-increment loop: the next slab's
        # fragments are in flight while the current slab's mmas issue.
        var ap = a_ptr + (k_start + fcol) * m + row_base + frow
        var mp = mask_ptr + (k_start + fcol) * m + row_base + frow
        var bp = b_ptr + (k_start + frow) * n + col_base + fcol
        var nslabs = (k - k_start) // MMA8_DIM
        if nslabs > 0:
            var cura = _load_a_fast(ap, mp)
            var curb = _load_b_fast(bp)
            for _ in range(nslabs - 1):
                ap += MMA8_DIM * m
                mp += MMA8_DIM * m
                bp += MMA8_DIM * n
                var nxta = _load_a_fast(ap, mp)
                var nxtb = _load_b_fast(bp)
                _mma_block(cura, curb, accum)
                cura = nxta^
                curb = nxtb^
            _mma_block(cura, curb, accum)
        if full_end < k:
            _slab_guarded(full_end, accum)
    else:
        var kk = k_start
        while kk < k:
            _slab_guarded(kk, accum)
            kk += MMA8_DIM

    comptime for mi in range(_NT_M):
        var grow = row_base + mi * MMA8_DIM + frow
        if grow < m:
            comptime for ni in range(_NT_N):
                var gcol = col_base + ni * MMA8_DIM + fcol
                var frag = accum[mi * _NT_N + ni]
                if gcol + FRAG8 <= n:
                    c_ptr.store(grow * n + gcol, frag)
                elif gcol < n:
                    c_ptr[grow * n + gcol] = frag[0]


def enqueue_sdpa_ta_gemm_f32(
    c_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    mask: Optional[UnsafePointer[Scalar[DType.bool], ImmutAnyOrigin]],
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    causal: Bool,
    drop_scale: Float64,
    ctx: DeviceContext,
) raises:
    """Enqueue C[z] = A[z]^T @ B[z] (A stored (k, m), B (k, n), C (m, n)).

    A mask pointer selects the MASKED specialization (A elements multiplied
    by the keep-mask and ``drop_scale`` on load). ``causal`` selects the
    reduction cut at the output row tile — see the module docstring for the
    zero-band contract that makes it exact.
    """
    comptime if not has_apple_gpu_accelerator():
        raise Error("sdpa_ta_gemm_f32 is Apple-GPU only")
    else:
        if batch <= 0 or m <= 0 or n <= 0 or k <= 0:
            return
        var gx = ceildiv(n, _BLOCK_N)
        var gy = ceildiv(m, _BLOCK_M)
        var scale_f32 = Float32(drop_scale)
        # The unmasked kernels never dereference the mask pointer; alias A
        # so the argument stays a valid translated device pointer.
        var mask_arg = a_ptr.bitcast[Scalar[DType.bool]]()
        if mask:
            mask_arg = mask.value()
        if mask:
            if causal:
                _enqueue_cached[_sdpa_ta_gemm_kernel[True, True]](
                    ctx,
                    "sdpa_ta_gemm_f32_m1_c1",
                    gx,
                    gy,
                    batch,
                    _THREADS,
                    c_ptr,
                    a_ptr,
                    b_ptr,
                    mask_arg,
                    Int64(m),
                    Int64(n),
                    Int64(k),
                    scale_f32,
                )
            else:
                _enqueue_cached[_sdpa_ta_gemm_kernel[True, False]](
                    ctx,
                    "sdpa_ta_gemm_f32_m1_c0",
                    gx,
                    gy,
                    batch,
                    _THREADS,
                    c_ptr,
                    a_ptr,
                    b_ptr,
                    mask_arg,
                    Int64(m),
                    Int64(n),
                    Int64(k),
                    scale_f32,
                )
        elif causal:
            _enqueue_cached[_sdpa_ta_gemm_kernel[False, True]](
                ctx,
                "sdpa_ta_gemm_f32_m0_c1",
                gx,
                gy,
                batch,
                _THREADS,
                c_ptr,
                a_ptr,
                b_ptr,
                mask_arg,
                Int64(m),
                Int64(n),
                Int64(k),
                scale_f32,
            )
        else:
            _enqueue_cached[_sdpa_ta_gemm_kernel[False, False]](
                ctx,
                "sdpa_ta_gemm_f32_m0_c0",
                gx,
                gy,
                batch,
                _THREADS,
                c_ptr,
                a_ptr,
                b_ptr,
                mask_arg,
                Int64(m),
                Int64(n),
                Int64(k),
                scale_f32,
            )
