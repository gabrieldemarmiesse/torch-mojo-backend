# ===----------------------------------------------------------------------=== #
# Fast eager-mode matmul kernels for mojo_device.
#
# `Matmul` / `Bmm` run pure-Mojo GEMM kernels (shared-memory tiled with
# per-thread register tiles, plus a bandwidth-oriented small-M variant) so
# the fast path works with only the NVIDIA driver — no cuBLAS. All kernels
# accumulate in float32 and handle dynamic shapes with edge guards.
#
# `Matmul` / `MatmulBiasSpec` / `Bmm` also run on the CPU MAX device, via
# modular's production CPU matmul (`linalg.matmul.matmul` target="cpu"; see
# `_cpu_gemm` below), since there is no graph fallback to lean on anymore.
# ===----------------------------------------------------------------------=== #

from std.math import ceildiv
from std.memory import alloc, stack_allocation
from std.os import abort
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_dim,
    block_idx,
    grid_dim,
    lane_id,
    thread_idx,
    warp_id,
)
from std.gpu.compute.mma import mma
from std.gpu.memory import (
    AddressSpace,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
)
from std.gpu.host import (
    DeviceAttribute,
    DeviceBuffer,
    DeviceContext,
    FuncAttribute,
)
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys.info import (
    _accelerator_arch,
    _has_sm_9x,
    has_accelerator,
    has_apple_gpu_accelerator,
    is_amd_gpu,
    simd_width_of,
    size_of,
)
from std.utils.coord import Coord as StdCoord
from std.utils.static_tuple import StaticTuple

from std.algorithm.functional import elementwise, parallelize

from layout import Coord, TileTensor, row_major
from layout.tensor_core import get_mma_shape

from linalg.gemv import gemv_gpu
from linalg.matmul import matmul as cpu_lib_matmul
from linalg.matmul.gpu import multistage_gemm_kernel
from linalg.utils import elementwise_epilogue_type
from linalg.utils_gpu import MatmulConfig
from std.sys import llvm_intrinsic

# Apple 8x8 simdgroup-matrix primitives (vendored from
# linalg.matmul.gpu.apple.matmul_8x8, whose helpers are not exported).
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


from std.gpu.primitives.grid_controls import PDLLevel

from std.python._cpython import PyObjectPtr, Py_ssize_t

from std.utils.index import Index, IndexList

from apple_gemm_nn_kernels import (
    apple_nn_direct_enqueue,
    apple_nn_smem_enqueue,
)
from apple_gemm_nt_kernels import (
    apple_nt_direct_enqueue,
    apple_nt_smem_enqueue,
)
from apple_gemm_tn_kernels import apple_tn_gemm_enqueue
from gemm_splitk_common import TARGET_BLOCKS, _ksplit_reduce_kernel
from tn_f32_gemm_kernels import try_enqueue_tn_f32_gemm
from op_utils import (
    FLOAT_DTYPES,
    MAX_RANK,
    TensorSpec,
    _copy_strided,
    _enqueue_cached,
    _get_ctx,
    _gs_blocks,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_int,
    _raw_tuple_len,
    _scratch_contig,
    _spec_dispatcher4,
    _spec_dispatcher5,
    _spec_dispatcher6,
    _spec_ptr,
    _spec_unsupported,
)

from variant_gates import (
    _dtype_arg_on,
    _dtype_supported,
    _op_on,
    _register_call,
)


# ---------------------------------------------------------------------------
# Tiled GEMM kernel: each block computes a BM x BN tile of C using shared
# memory K-slabs and a TM x TN register tile per thread. Batched via
# block_idx.z. Accumulates in float32.
#
#   C[z, m, n] = A[z, m, k] @ B[z, k, n]     (transpose_b=False)
#   C[z, m, n] = A[z, m, k] @ B[z, n, k]^T   (transpose_b=True)
#
# Split-K: grid.z = batch * ksplits; each split covers a K-chunk and, when
# ksplits > 1, accumulates its partial result into a zero-initialized C
# with fp32 atomics. This keeps enough blocks in flight for the K-rich,
# MN-poor shapes convolution lowers to (e.g. 512x49 @ k=4608).
# ---------------------------------------------------------------------------

# `TARGET_BLOCKS` (the blocks-in-flight target for the split-K heuristics
# below) and the split-K reduce kernel live in `gemm_splitk_common.mojo`,
# shared with the NVIDIA fp32 TN dispatch (`tn_f32_gemm_kernels.mojo`).


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((BM // TM) * (BN // TN))
    ),
    `nvvm.minctasm`=SIMDLength(2),
)
@__name(t"pure_gemm_tiled_{dtype}_{BM}x{BN}x{BK}_tb{transpose_b}")
def _gemm_tiled_kernel[
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    transpose_b: Bool,
](
    c_base: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
):
    comptime THREADS = (BM // TM) * (BN // TN)
    comptime assert (BM * BK) % THREADS == 0, "A tile not evenly loadable"
    comptime assert (BK * BN) % THREADS == 0, "B tile not evenly loadable"
    comptime LA = (BM * BK) // THREADS
    comptime LB = (BK * BN) // THREADS

    var bz = block_idx.z // ksplits
    var ks = block_idx.z % ksplits
    # Round the chunk to a BK multiple: k_start must stay vector-aligned
    # for the vectorized/cp.async load paths (16B-aligned addresses).
    var kchunk = ceildiv(ceildiv(k, ksplits), BK) * BK
    var k_start = ks * kchunk
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [batch * ksplits, m, n] workspace and each
    # split writes its own slice; a reduce kernel sums them afterwards.
    # a_bstride is 0 when A is shared across the batch (conv weights).
    var c_ptr = c_base + block_idx.z * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var bm = block_idx.y * BM
    var bn = block_idx.x * BN
    var tid = thread_idx.x

    # A slab stored transposed (As[kk][mm]) so the per-thread TM-wide frag
    # load is contiguous; B slab stored as Bs[kk][nn].
    var a_smem = stack_allocation[
        BK * BM, dtype, address_space=AddressSpace.SHARED
    ]()
    var b_smem = stack_allocation[
        BK * BN, dtype, address_space=AddressSpace.SHARED
    ]()

    var tn0 = (tid % (BN // TN)) * TN
    var tm0 = (tid // (BN // TN)) * TM

    var acc = InlineArray[SIMD[DType.float32, TN], TM](
        fill=SIMD[DType.float32, TN](0)
    )

    for kt in range(k_start, k_end, BK):
        # Cooperative loads, zero-padded on every edge.
        comptime for t in range(LA):
            var i = t * THREADS + tid
            var mm = i // BK
            var kk = i % BK
            var row = bm + mm
            var col = kt + kk
            var val = Scalar[dtype](0)
            if row < m and col < k_end:
                val = a_ptr[row * k + col]
            a_smem[kk * BM + mm] = val

        comptime for t in range(LB):
            var i = t * THREADS + tid
            var val = Scalar[dtype](0)

            comptime if transpose_b:
                var nn = i // BK
                var kk = i % BK
                var row = bn + nn  # row of B (n, k)
                var col = kt + kk
                if row < n and col < k_end:
                    val = b_ptr[row * k + col]
                b_smem[kk * BN + nn] = val
            else:
                var kk = i // BN
                var nn = i % BN
                var row = kt + kk  # row of B (k, n)
                var col = bn + nn
                if row < k_end and col < n:
                    val = b_ptr[row * n + col]
                b_smem[kk * BN + nn] = val

        barrier()

        # Runtime loop on kk: full comptime unrolling of the slab kept ~190
        # live registers (1 block/SM); this stays ~2x lower.
        for kk in range(BK):
            var a_frag = a_smem.load[width=TM](kk * BM + tm0).cast[
                DType.float32
            ]()
            var b_frag = b_smem.load[width=TN](kk * BN + tn0).cast[
                DType.float32
            ]()
            comptime for i in range(TM):
                acc[i] = b_frag.fma(SIMD[DType.float32, TN](a_frag[i]), acc[i])

        barrier()

    comptime for i in range(TM):
        var row = bm + tm0 + i
        if row < m:
            var out = acc[i].cast[dtype]()
            if bn + tn0 + TN <= n:
                c_ptr.store(row * n + bn + tn0, out)
            else:
                comptime for j in range(TN):
                    var col = bn + tn0 + j
                    if col < n:
                        c_ptr[row * n + col] = out[j]


# ---------------------------------------------------------------------------
# Pipelined float32 GEMM kernel: double-buffered shared memory with a
# software pipeline — the next K-slab is fetched (B via cp.async straight
# into shared memory, A staged through registers so it can be stored
# transposed) while the current slab is computed. VEC=4 uses float4 global
# accesses and requires the relevant leading dimension % 4 == 0.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((BM // TM) * (BN // TN))
    ),
    `nvvm.minctasm`=SIMDLength(2 if BM >= 128 else 3),
)
@__name(t"pure_gemm_pipe_{BM}x{BN}x{BK}_va{VEC_A}_vb{VEC_B}_tb{transpose_b}")
def _gemm_pipe_kernel[
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    VEC_A: Int,
    VEC_B: Int,
    transpose_b: Bool,
](
    c_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
):
    comptime F32 = DType.float32
    comptime THREADS = (BM // TM) * (BN // TN)
    comptime LA = (BM * BK) // THREADS  # A elements per thread per slab
    comptime LB = (BK * BN) // THREADS  # B elements per thread per slab
    comptime NA = LA // VEC_A  # A vectors per thread
    comptime NB = LB // VEC_B  # B vectors per thread
    comptime assert (BM * BK) % (THREADS * VEC_A) == 0
    comptime assert (BK * BN) % (THREADS * VEC_B) == 0

    var bz = block_idx.z // ksplits
    var ks = block_idx.z % ksplits
    # Round the chunk to a BK multiple: k_start must stay vector-aligned
    # for the vectorized/cp.async load paths (16B-aligned addresses).
    var kchunk = ceildiv(ceildiv(k, ksplits), BK) * BK
    var k_start = ks * kchunk
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [batch * ksplits, m, n] workspace and each
    # split writes its own slice; a reduce kernel sums them afterwards.
    # a_bstride is 0 when A is shared across the batch (conv weights).
    var c_ptr = c_base + block_idx.z * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var bm = block_idx.y * BM
    var bn = block_idx.x * BN
    var tid = thread_idx.x

    # Double-buffered slabs. A is stored transposed (As[kk][mm]) so the
    # compute fragment load is a contiguous vector; B as Bs[kk][nn].
    var a_smem = stack_allocation[
        2 * BK * BM, F32, address_space=AddressSpace.SHARED
    ]()
    var b_smem = stack_allocation[
        2 * BK * BN, F32, address_space=AddressSpace.SHARED
    ]()

    var tn0 = (tid % (BN // TN)) * TN
    var tm0 = (tid // (BN // TN)) * TM

    var acc = InlineArray[SIMD[F32, TN], TM](fill=SIMD[F32, TN](0))
    var a_regs = InlineArray[SIMD[F32, VEC_A], NA](uninitialized=True)
    var b_regs = InlineArray[SIMD[F32, VEC_B], NB](uninitialized=True)

    var nslabs = ceildiv(k_end - k_start, BK)

    # ---- helpers ----------------------------------------------------------

    @always_inline
    @parameter
    @__copy_capture(a_ptr, bm, k_end)
    def _load_a_regs(kt: Int, mut regs: InlineArray[SIMD[F32, VEC_A], NA]):
        # Guarded (row, k) loads, zero-padded past m / k_end.
        comptime for t in range(NA):
            var ci = t * THREADS + tid
            var mm = ci // (BK // VEC_A)
            var ck = (ci % (BK // VEC_A)) * VEC_A
            var row = bm + mm
            var col = kt + ck
            var vec = SIMD[F32, VEC_A](0)
            if row < m:
                if col + VEC_A <= k_end:
                    vec = a_ptr.load[width=VEC_A](row * k + col)
                elif col < k_end:
                    for u in range(k_end - col):
                        vec[u] = a_ptr[row * k + col + u]
            regs[t] = vec

    @always_inline
    @parameter
    @__copy_capture(a_smem, tid)
    def _store_a_smem(buf: Int, regs: InlineArray[SIMD[F32, VEC_A], NA]):
        var base = a_smem + buf * BK * BM
        comptime for t in range(NA):
            var ci = t * THREADS + tid
            var mm = ci // (BK // VEC_A)
            var ck = (ci % (BK // VEC_A)) * VEC_A
            comptime for u in range(VEC_A):
                base[(ck + u) * BM + mm] = regs[t][u]

    @always_inline
    @parameter
    @__copy_capture(b_ptr, bn, n, k, k_end)
    def _cpasync_b(kt: Int, buf: Int):
        # B (k, n) row-major: chunks along n, straight into Bs[kk][nn].
        var base = b_smem + buf * BK * BN
        comptime for t in range(NB):
            var ci = t * THREADS + tid
            var kk = ci // (BN // VEC_B)
            var cn = (ci % (BN // VEC_B)) * VEC_B
            var row = kt + kk
            var col = bn + cn
            var bytes: Int32 = 0
            if row < k_end:
                bytes = Int32(max(0, min(VEC_B, n - col)) * 4)
            var src_off = (row * n + col) if bytes > 0 else 0
            async_copy[VEC_B * 4, fill=Scalar[F32](0)](
                (b_ptr + src_off).address_space_cast[AddressSpace.GLOBAL](),
                (base + kk * BN + cn).address_space_cast[AddressSpace.SHARED](),
                src_size=bytes,
            )

    @always_inline
    @parameter
    @__copy_capture(b_ptr, bn, n, k, k_end)
    def _load_b_regs(kt: Int, mut regs: InlineArray[SIMD[F32, VEC_B], NB]):
        # transpose_b: B is (n, k) row-major; chunks along k like A.
        comptime for t in range(NB):
            var ci = t * THREADS + tid
            var nn = ci // (BK // VEC_B)
            var ck = (ci % (BK // VEC_B)) * VEC_B
            var row = bn + nn
            var col = kt + ck
            var vec = SIMD[F32, VEC_B](0)
            if row < n:
                if col + VEC_B <= k_end:
                    vec = b_ptr.load[width=VEC_B](row * k + col)
                elif col < k_end:
                    for u in range(k_end - col):
                        vec[u] = b_ptr[row * k + col + u]
            regs[t] = vec

    @always_inline
    @parameter
    @__copy_capture(b_smem, tid)
    def _store_b_smem(buf: Int, regs: InlineArray[SIMD[F32, VEC_B], NB]):
        var base = b_smem + buf * BK * BN
        comptime for t in range(NB):
            var ci = t * THREADS + tid
            var nn = ci // (BK // VEC_B)
            var ck = (ci % (BK // VEC_B)) * VEC_B
            comptime for u in range(VEC_B):
                base[(ck + u) * BN + nn] = regs[t][u]

    @always_inline
    @parameter
    def _fetch(kt: Int, buf: Int):
        comptime if transpose_b:
            _load_b_regs(kt, b_regs)
            _store_b_smem(buf, b_regs)
        else:
            _cpasync_b(kt, buf)
            async_copy_commit_group()
        _load_a_regs(kt, a_regs)
        _store_a_smem(buf, a_regs)

    # ---- prologue ---------------------------------------------------------

    if nslabs == 0:
        return
    _fetch(k_start, 0)

    comptime if not transpose_b:
        async_copy_wait_group(0)
    barrier()

    # ---- main loop --------------------------------------------------------

    for s in range(nslabs):
        var cur = s % 2
        var nxt = (s + 1) % 2
        var has_next = s + 1 < nslabs

        # Issue next slab's fetches so they overlap with this slab's math.
        if has_next:
            comptime if not transpose_b:
                _cpasync_b(k_start + (s + 1) * BK, nxt)
                async_copy_commit_group()
            _load_a_regs(k_start + (s + 1) * BK, a_regs)

            comptime if transpose_b:
                _load_b_regs(k_start + (s + 1) * BK, b_regs)

        var a_base_s = a_smem + cur * BK * BM
        var b_base_s = b_smem + cur * BK * BN
        for kk in range(BK):
            var a_frag = a_base_s.load[width=TM](kk * BM + tm0)
            var b_frag = b_base_s.load[width=TN](kk * BN + tn0)
            comptime for i in range(TM):
                acc[i] = b_frag.fma(SIMD[F32, TN](a_frag[i]), acc[i])

        if has_next:
            comptime if not transpose_b:
                async_copy_wait_group(0)
            _store_a_smem(nxt, a_regs)

            comptime if transpose_b:
                _store_b_smem(nxt, b_regs)
        barrier()

    # ---- epilogue ---------------------------------------------------------

    comptime for i in range(TM):
        var row = bm + tm0 + i
        if row < m:
            if bn + tn0 + TN <= n:
                c_ptr.store(row * n + bn + tn0, acc[i])
            else:
                comptime for j in range(TN):
                    var col = bn + tn0 + j
                    if col < n:
                        c_ptr[row * n + col] = acc[i][j]


# ---------------------------------------------------------------------------
# 3-stage cp.async float32 GEMM (transpose_b=False only): both operands are
# fetched straight into shared memory with cp.async, two slabs ahead of the
# compute — enough in-flight distance to cover L2 latency on the L2-resident
# deep-K shapes convolution produces. A is stored row-major (As[mm][kk],
# cp.async cannot transpose); the compute loop reads A as per-row scalar
# broadcasts.
# ---------------------------------------------------------------------------

comptime PIPE3_STAGES = 4


# One thread per element: out (k, m) <- in (m, k). Used to transpose the
# small activation matrix for the skinny-M transposed-B GEMM path below
# (96 KB for a GPT-2 decode step; ~1 µs).
@__name("pure_transpose_small")
def _transpose_small_kernel(
    out_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    k: Int,
):
    var i = block_idx.x * 256 + thread_idx.x
    if i >= m * k:
        return
    var mm = i // k
    var kk = i % k
    out_ptr[kk * m + mm] = in_ptr[i]


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((BM // TM) * (BN // TN))
    ),
    `nvvm.minctasm`=SIMDLength(MINB if MINB > 0 else (2 if BM >= 128 else 3)),
)
@__name(
    t"pure_gemm_pipe3_{BM}x{BN}x{BK}_va{VEC_A}_vb{VEC_B}_ct{CT}_s{STAGES}_at{AT}_mb{MINB}_bs{BIAS}"
)
def _gemm_pipe3_kernel[
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    VEC_A: Int,
    VEC_B: Int,
    CT: Bool = False,
    STAGES: Int = PIPE3_STAGES,
    AT: Bool = False,
    MINB: Int = -1,
    BIAS: Bool = False,
](
    # MINB: min resident blocks/SM hint (__launch_bounds__ second arg);
    # -1 keeps the historical default (2 for BM >= 128 else 3).
    # BIAS: fuse `c[row, col] += bias[col]` into the epilogue (the ks == 0
    # split only, so split-K partial sums receive the bias exactly once).
    c_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
    bias_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
):
    comptime F32 = DType.float32
    comptime THREADS = (BM // TM) * (BN // TN)
    comptime NA = (BM * BK) // (THREADS * VEC_A)
    comptime NB = (BK * BN) // (THREADS * VEC_B)
    comptime assert (BM * BK) % (THREADS * VEC_A) == 0
    comptime assert (BK * BN) % (THREADS * VEC_B) == 0
    comptime assert not (BIAS and CT), "bias fusion not supported with CT"

    var bz = block_idx.z // ksplits
    var ks = block_idx.z % ksplits
    # Round the chunk to a BK multiple: k_start must stay vector-aligned
    # for the vectorized/cp.async load paths (16B-aligned addresses).
    var kchunk = ceildiv(ceildiv(k, ksplits), BK) * BK
    var k_start = ks * kchunk
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [batch * ksplits, m, n] workspace and each
    # split writes its own slice; a reduce kernel sums them afterwards.
    # a_bstride is 0 when A is shared across the batch (conv weights).
    var c_ptr = c_base + block_idx.z * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var bm = block_idx.y * BM
    var bn = block_idx.x * BN
    var tid = thread_idx.x

    var a_smem = stack_allocation[
        STAGES * BM * BK, F32, address_space=AddressSpace.SHARED
    ]()
    var b_smem = stack_allocation[
        STAGES * BK * BN, F32, address_space=AddressSpace.SHARED
    ]()

    var tn0 = (tid % (BN // TN)) * TN
    var tm0 = (tid // (BN // TN)) * TM

    var acc = InlineArray[SIMD[F32, TN], TM](fill=SIMD[F32, TN](0))

    var nslabs = ceildiv(k_end - k_start, BK)
    if nslabs == 0:
        return

    @always_inline
    @parameter
    def _fetch(s: Int):
        var buf = s % STAGES
        var kt = k_start + s * BK
        var a_dst = a_smem + buf * BM * BK
        var b_dst = b_smem + buf * BK * BN

        comptime if AT:
            # A^T (k, m) row-major: chunks along m into As[kk][mm], so the
            # compute fragment load is a contiguous vector (no scalar
            # broadcasts). Requires m % VEC_A == 0 for cp.async alignment.
            comptime for t in range(NA):
                var ci = t * THREADS + tid
                var kk = ci // (BM // VEC_A)
                var cm = (ci % (BM // VEC_A)) * VEC_A
                var row = kt + kk
                var col = bm + cm
                var bytes: Int32 = 0
                if row < k_end:
                    bytes = Int32(max(0, min(VEC_A, m - col)) * 4)
                var src_off = (row * m + col) if bytes > 0 else 0
                async_copy[VEC_A * 4, fill=Scalar[F32](0)](
                    (a_ptr + src_off).address_space_cast[AddressSpace.GLOBAL](),
                    (a_dst + kk * BM + cm).address_space_cast[
                        AddressSpace.SHARED
                    ](),
                    src_size=bytes,
                )
        else:
            # A (m, k) row-major: chunks along k into As[mm][kk].
            comptime for t in range(NA):
                var ci = t * THREADS + tid
                var mm = ci // (BK // VEC_A)
                var ck = (ci % (BK // VEC_A)) * VEC_A
                var row = bm + mm
                var col = kt + ck
                var bytes: Int32 = 0
                if row < m:
                    bytes = Int32(max(0, min(VEC_A, k_end - col)) * 4)
                var src_off = (row * k + col) if bytes > 0 else 0
                async_copy[VEC_A * 4, fill=Scalar[F32](0)](
                    (a_ptr + src_off).address_space_cast[AddressSpace.GLOBAL](),
                    (a_dst + mm * BK + ck).address_space_cast[
                        AddressSpace.SHARED
                    ](),
                    src_size=bytes,
                )

        # B (k, n) row-major: chunks along n into Bs[kk][nn].
        comptime for t in range(NB):
            var ci = t * THREADS + tid
            var kk = ci // (BN // VEC_B)
            var cn = (ci % (BN // VEC_B)) * VEC_B
            var row = kt + kk
            var col = bn + cn
            var bytes: Int32 = 0
            if row < k_end:
                bytes = Int32(max(0, min(VEC_B, n - col)) * 4)
            var src_off = (row * n + col) if bytes > 0 else 0
            async_copy[VEC_B * 4, fill=Scalar[F32](0)](
                (b_ptr + src_off).address_space_cast[AddressSpace.GLOBAL](),
                (b_dst + kk * BN + cn).address_space_cast[
                    AddressSpace.SHARED
                ](),
                src_size=bytes,
            )

    # Prologue: fill the pipeline (STAGES - 1 slabs in flight).
    for st in range(min(STAGES - 1, nslabs)):
        _fetch(st)
        async_copy_commit_group()
    for _ in range(nslabs, STAGES - 1):
        async_copy_commit_group()

    for s in range(nslabs):
        # Wait until the group fetched for slab s has landed.
        async_copy_wait_group(Int32(STAGES - 2))
        barrier()

        var buf = s % STAGES
        var a_base_s = a_smem + buf * BM * BK
        var b_base_s = b_smem + buf * BK * BN

        # Fully unrolled with register-double-buffered fragments: the
        # loads for slab column kk + 1 issue before the FMAs of column
        # kk, so the shared-memory load latency hides behind the math
        # instead of stalling every iteration.
        var a_cur: SIMD[F32, TM]
        comptime if AT:
            a_cur = a_base_s.load[width=TM](tm0)
        else:
            a_cur = SIMD[F32, TM](0)
            comptime for i in range(TM):
                a_cur[i] = a_base_s[(tm0 + i) * BK]
        var b_cur = b_base_s.load[width=TN](tn0)
        comptime for kk in range(BK - 1):
            var a_nxt: SIMD[F32, TM]
            comptime if AT:
                a_nxt = a_base_s.load[width=TM]((kk + 1) * BM + tm0)
            else:
                a_nxt = SIMD[F32, TM](0)
                comptime for i in range(TM):
                    a_nxt[i] = a_base_s[(tm0 + i) * BK + kk + 1]
            var b_nxt = b_base_s.load[width=TN]((kk + 1) * BN + tn0)
            comptime for i in range(TM):
                acc[i] = b_cur.fma(SIMD[F32, TN](a_cur[i]), acc[i])
            a_cur = a_nxt
            b_cur = b_nxt
        comptime for i in range(TM):
            acc[i] = b_cur.fma(SIMD[F32, TN](a_cur[i]), acc[i])

        # Release this buffer, then refill it with the slab ahead.
        barrier()
        if s + STAGES - 1 < nslabs:
            _fetch(s + STAGES - 1)
        async_copy_commit_group()

    comptime if CT:
        # C is stored transposed, (n, m) row-major: used by the skinny-M
        # transposed-B path, which computes C^T = B @ A^T with the roles
        # of the operands swapped. Rows (the kernel's m) are contiguous in
        # C, so each column's TM values store as one vector.
        comptime for j in range(TN):
            var col = bn + tn0 + j
            if col < n:
                var row0 = bm + tm0
                var out = SIMD[F32, TM](0)
                comptime for i in range(TM):
                    out[i] = acc[i][j]
                if row0 + TM <= m:
                    c_ptr.store(col * m + row0, out)
                else:
                    comptime for i in range(TM):
                        if row0 + i < m:
                            c_ptr[col * m + row0 + i] = out[i]
    else:
        comptime if BIAS:
            var bias_frag = SIMD[F32, TN](0)
            if ks == 0:
                if bn + tn0 + TN <= n:
                    bias_frag = bias_ptr.load[width=TN](bn + tn0)
                else:
                    comptime for j in range(TN):
                        if bn + tn0 + j < n:
                            bias_frag[j] = bias_ptr[bn + tn0 + j]
            comptime for i in range(TM):
                acc[i] = acc[i] + bias_frag
        comptime for i in range(TM):
            var row = bm + tm0 + i
            if row < m:
                if bn + tn0 + TN <= n:
                    c_ptr.store(row * n + bn + tn0, acc[i])
                else:
                    comptime for j in range(TN):
                        var col = bn + tn0 + j
                        if col < n:
                            c_ptr[row * n + col] = acc[i][j]


# ---------------------------------------------------------------------------
# Small-M GEMM kernel (m <= MR): one thread per output column, streaming B
# at full bandwidth with MR row accumulators in registers. Batched via
# block_idx.z.
# ---------------------------------------------------------------------------

comptime SMALLM_THREADS = 256
comptime SMALLM_MR = 8


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(SMALLM_THREADS)),
    `nvvm.minctasm`=SIMDLength(2),
)
@__name(t"pure_gemm_smallm_{dtype}_tb{transpose_b}")
def _gemm_smallm_kernel[
    dtype: DType,
    transpose_b: Bool,
](
    c_base: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
):
    var col = block_idx.x * SMALLM_THREADS + thread_idx.x
    if col >= n:
        return

    var bz = block_idx.z // ksplits
    var ks = block_idx.z % ksplits
    var kchunk = ceildiv(k, ksplits)
    var k_start = ks * kchunk
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [batch * ksplits, m, n] workspace and each
    # split writes its own slice; a reduce kernel sums them afterwards.
    # a_bstride is 0 when A is shared across the batch (conv weights).
    var c_ptr = c_base + block_idx.z * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var acc = InlineArray[Scalar[DType.float32], SMALLM_MR](
        fill=Scalar[DType.float32](0)
    )

    # Unroll k by 4 so each thread keeps several B loads in flight — the
    # kernel is B-bandwidth-bound and often runs at low occupancy.
    comptime KU = 4
    var k4 = k_start + ((k_end - k_start) // KU) * KU

    @always_inline
    @parameter
    def _load_b(kk: Int) -> Scalar[DType.float32]:
        comptime if transpose_b:
            return b_ptr[col * k + kk].cast[DType.float32]()
        else:
            return b_ptr[kk * n + col].cast[DType.float32]()

    for kt in range(k_start, k4, KU):
        var bv = InlineArray[Scalar[DType.float32], KU](uninitialized=True)
        comptime for u in range(KU):
            bv[u] = _load_b(kt + u)
        comptime for u in range(KU):
            comptime for r in range(SMALLM_MR):
                if r < m:
                    acc[r] = (
                        a_ptr[r * k + kt + u]
                        .cast[DType.float32]()
                        .fma(bv[u], acc[r])
                    )

    for kk in range(k4, k_end):
        var bv = _load_b(kk)
        comptime for r in range(SMALLM_MR):
            if r < m:
                acc[r] = a_ptr[r * k + kk].cast[DType.float32]().fma(bv, acc[r])

    comptime for r in range(SMALLM_MR):
        if r < m:
            c_ptr[r * n + col] = acc[r].cast[dtype]()


# ---------------------------------------------------------------------------
# MI300X FP32 MFMA GEMM.  The portable kernels above use scalar FFMAs and
# were tuned for NVIDIA; on gfx942 they leave the matrix cores idle.  MAX's
# public multistage wrapper selects a transposed-B AMD schedule that requires
# static N/K layouts.  Enqueue the underlying pure-Mojo MFMA kernel directly
# instead: its tile configuration is static, while M/N/K and all edge guards
# remain runtime values.  This gives one compiled kernel per shape regime,
# not one per model dimension, and never calls vendor BLAS.
# ---------------------------------------------------------------------------


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
)
@__name(t"amd_dynamic_mfma_edge_{dtype}_tb{transpose_b}_bias{fuse_bias}")
def _amd_dynamic_mfma_edge_kernel[
    dtype: DType, transpose_b: Bool, fuse_bias: Bool
](
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    bias: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    n: Int,
    k: Int,
    row_start: Int,
    row_count: Int,
    col_start: Int,
    col_count: Int,
):
    """Compute a partial M/N edge without padding or a workspace.

    The MFMA launch handles only complete BMxBN tiles. At most one thin row
    strip and one thin column strip remain for this scalar cleanup kernel.
    """
    var idx = block_idx.x * 256 + thread_idx.x
    if idx >= row_count * col_count:
        return
    var row = row_start + idx // col_count
    var col = col_start + idx % col_count
    var acc = Scalar[DType.float32](0)
    var k4 = (k // 4) * 4
    for kk in range(0, k4, 4):
        comptime for u in range(4):
            var bv = b[col * k + kk + u] if transpose_b else b[
                (kk + u) * n + col
            ]
            acc = (
                a[row * k + kk + u]
                .cast[DType.float32]()
                .fma(bv.cast[DType.float32](), acc)
            )
    for kk in range(k4, k):
        var bv = b[col * k + kk] if transpose_b else b[kk * n + col]
        acc = (
            a[row * k + kk]
            .cast[DType.float32]()
            .fma(bv.cast[DType.float32](), acc)
        )
    comptime if fuse_bias:
        acc += bias[col].cast[DType.float32]()
    c[row * n + col] = acc.cast[dtype]()


@always_inline
def _amd_dynamic_mfma_gemm[
    dtype: DType,
    BM: Int,
    BN: Int,
    WM: Int,
    WN: Int,
    transpose_b: Bool,
    fuse_bias: Bool,
    BLOCK_K: Int = 32,
    WARP_K_PARTITIONS: Int = 1,
    K_GROUP_SIZE: Int = 1,
    STAGES: Int = 2,
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    bias_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    # --- gfx942 transposed-B geometry guard -------------------------------
    # MAX's two-stage `multistage_gemm_kernel` miscompiles on gfx942 for a set
    # of transposed-B block/warp decompositions: one MFMA k-step's contribution
    # is dropped from part of the accumulator, and which part varies per
    # workgroup.  It reproduces with K == BLOCK_K, where the K loop performs no
    # global-to-LDS prefetch at all and therefore cannot race, so it is a
    # code-generation defect and not a synchronization bug; three or four
    # pipeline stages also make it disappear.  Every observed failure has a
    # warp tile only one MMA wide in some dimension (WM == 16 or WN == 16),
    # so require at least a 2x2 MMA warp tile whenever B is read transposed.
    # See optimization_journal.md, "Change 10".
    comptime MMA_DIM = get_mma_shape[dtype, DType.float32]()[0]
    comptime if transpose_b:
        comptime assert WM >= 2 * MMA_DIM and WN >= 2 * MMA_DIM, (
            "transposed-B multistage geometries with a single-MMA warp tile are"
            " miscompiled on gfx942; use WM, WN >= 2 * mma_dim"
        )
        # The B tile copy distributes `min(threads, BN*BLOCK_K/simd)` threads
        # over BN rows; a row count that does not divide BN drops or duplicates
        # rows of B.
        #
        # Both factors have to be the kernel's, not this host function's.
        # `multistage_mma` receives `num_threads_per_warp_k_part`, that is
        # `config.num_threads()` divided by the partition count, so the
        # partition count must not appear here at all. And its `simd_size` is
        # `simd_width_of[a_type]()` evaluated for the *device*, a 16-byte gfx942
        # vector, whereas the same call in this host `def` would return the
        # build machine's vector width.
        comptime DEVICE_SIMD = 16 // size_of[dtype]()
        comptime B_COPY_ROWS = min(
            (BM // WM) * (BN // WN) * 64,
            BN * BLOCK_K // DEVICE_SIMD,
        ) * DEVICE_SIMD // BLOCK_K
        comptime assert (
            BN % B_COPY_ROWS == 0
        ), "transposed-B tile copy row count must divide BN"

    comptime F32 = DType.float32
    var c_ptr = _make_ptr[dtype](c_addr)
    var c = TileTensor(c_ptr, row_major(Coord(m, n)))
    var a = TileTensor(
        _make_ptr[dtype](a_addr).as_immutable(), row_major(Coord(m, k))
    )

    @always_inline
    @parameter
    @__copy_capture(c_ptr, n)
    def _edge_store[
        value_dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], value: SIMD[value_dtype, width]):
        var row = Int(coords[0])
        var col = Int(coords[1])
        var off = row * n + col
        if col + width <= n:
            c_ptr.store[width=width, alignment=4](off, value.cast[dtype]())
        else:
            comptime for i in range(width):
                if col + i < n:
                    c_ptr[off + i] = value[i].cast[dtype]()

    comptime edge_epilogue = Optional[elementwise_epilogue_type](_edge_store)

    var bias_ptr = _make_ptr[dtype](bias_addr).as_immutable()

    @always_inline
    @parameter
    @__copy_capture(c_ptr, bias_ptr, n)
    def _bias_store[
        value_dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], value: SIMD[value_dtype, width]):
        var row = Int(coords[0])
        var col = Int(coords[1])
        var off = row * n + col
        if col + width <= n:
            var result = (
                value.cast[F32]() + bias_ptr.load[width=width](col).cast[F32]()
            )
            c_ptr.store[width=width, alignment=4](off, result.cast[dtype]())
        else:
            comptime for i in range(width):
                if col + i < n:
                    c_ptr[off + i] = (
                        value[i].cast[F32]() + bias_ptr[col + i].cast[F32]()
                    ).cast[dtype]()

    comptime bias_epilogue = Optional[elementwise_epilogue_type](_bias_store)

    comptime config = MatmulConfig[dtype, dtype, dtype, transpose_b](
        block_tile_shape=Index(BM, BN, BLOCK_K),
        warp_tile_shape=Index(WM, WN, BLOCK_K),
        mma_shape=get_mma_shape[dtype, DType.float32](),
        num_pipeline_stages=STAGES,
        num_warp_k_partitions=WARP_K_PARTITIONS,
        k_group_size=K_GROUP_SIZE,
    )
    var b = TileTensor(
        _make_ptr[dtype](b_addr).as_immutable(),
        row_major(Coord(n, k)) if transpose_b else row_major(Coord(k, n)),
    )

    comptime kernel = multistage_gemm_kernel[
        CLT=c.LayoutType,
        ALT=a.LayoutType,
        BLT=b.LayoutType,
        c_linear_idx_type=c.linear_idx_type,
        a_linear_idx_type=a.linear_idx_type,
        b_linear_idx_type=b.linear_idx_type,
        config=config,
        elementwise_lambda_fn=(bias_epilogue if fuse_bias else edge_epilogue),
    ]
    var n_full = (n // BN) * BN
    if n_full > 0:
        ctx.enqueue_function[kernel](
            c,
            a,
            b,
            grid_dim=(n_full // BN, ceildiv(m, BM)),
            block_dim=config.block_dim(),
            shared_mem_bytes=config.shared_mem_usage(),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(config.shared_mem_usage())
            ),
        )
    var c_raw = c_ptr.as_unsafe_any_origin()
    var a_raw = _make_ptr[dtype](a_addr).as_unsafe_any_origin().as_immutable()
    var b_raw = _make_ptr[dtype](b_addr).as_unsafe_any_origin().as_immutable()
    if n_full < n:
        var col_count = n - n_full
        _enqueue_cached[
            _amd_dynamic_mfma_edge_kernel[dtype, transpose_b, fuse_bias]
        ](
            ctx,
            String(
                t"amd_dynamic_mfma_edge_{dtype}_tb{transpose_b}_bias{fuse_bias}"
            ),
            ceildiv(m * col_count, 256),
            1,
            1,
            256,
            c_raw,
            a_raw,
            b_raw,
            bias_ptr.as_unsafe_any_origin(),
            n,
            k,
            0,
            m,
            n_full,
            col_count,
        )


# ---------------------------------------------------------------------------
# Global split-K for the underfilled deep-K regime.
#
# A weight-gradient GEMM has a huge contraction extent and a small output:
# (2304, 768, 49152) yields 108 output tiles of 128x128, which is 0.36 workgroups
# per CU on a 304-CU part, so two thirds of the device is idle for the whole K
# loop.  Partitioning K across `parts` workgroups per output tile multiplies the
# grid by `parts`; each partition accumulates its own K slab in FP32 into its own
# workspace slice, and a second pass sums the slices and rounds once to the
# output dtype.
#
# This is expressible around the stock `multistage_gemm_kernel` for the same
# reason the batch index is: the core takes its output-tile coordinates from
# `block_idx.x`/`.y` only, so `block_idx.z` is free, and it takes K from B's
# runtime extent, so a slab is a pointer offset plus a shorter extent -- exactly
# what MAX's own `multistage_gemm_split_k_kernel` does (`_multistage_gemm_gpu.mojo`,
# which is unreachable here because it reads N and K from *static* shapes).
#
# Only `transpose_b == False` can express a slab: a `[K, N]` operand's slab is
# `parts` whole rows, but an `[N, K]` operand's slab is a column range of every
# row, whose row stride is the full K and therefore not `row_major`.  The route
# declines otherwise.
#
# Accumulation stays FP32 throughout: the partial sums are the multistage core's
# own FP32 accumulators, written to an FP32 workspace, and the only rounding to
# the output dtype is the last one in the reduction -- the same single rounding
# the unsplit route performs.
# ---------------------------------------------------------------------------


@__name(t"amd_splitk_mfma_{dtype}_{BM}x{BN}x{BLOCK_K}")
def _amd_splitk_mfma_kernel[
    dtype: DType,
    BM: Int,
    BN: Int,
    BLOCK_K: Int,
    config: MatmulConfig[dtype, dtype, DType.float32, False],
](
    ws_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    k_per: Int,
):
    """One K slab per `block_idx.z`, into that slab's own FP32 workspace slice.

    `k_per` is a multiple of `BLOCK_K` and `parts * k_per == k`, both enforced by
    the dispatch, so every slab is a whole number of K tiles and the loader never
    reads past an operand.
    """
    var z = Int(block_idx.z)
    var k_off = z * k_per
    var ws_ptr = ws_base + z * m * n
    var c = TileTensor(ws_ptr, row_major(Coord(m, n)))
    # A keeps its true row stride `k` and is merely offset along the contraction
    # axis; B's slab is `k_per` whole rows, so its extent carries the slab length
    # and the core's K loop follows it.
    var a = TileTensor(a_base + k_off, row_major(Coord(m, k)))
    var b = TileTensor(b_base + k_off * n, row_major(Coord(k_per, n)))

    @always_inline
    @parameter
    def _store[
        value_dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], value: SIMD[value_dtype, width]):
        ws_ptr.store[width=width, alignment=size_of[DType.float32]()](
            Int(coords[0]) * n + Int(coords[1]),
            value.cast[DType.float32](),
        )

    multistage_gemm_kernel[
        config=config,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](_store),
    ](c, a, b)


@__name(t"amd_splitk_reduce_{dtype}_v{VEC}")
def _splitk_reduce_kernel[
    dtype: DType, VEC: Int
](
    c_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    total: Int,
    parts: Int,
    vec_count: Int,
):
    """`c[i] = sum over slabs of ws[slab, i]`, summed and rounded once in FP32.
    """
    var index = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    var j = index
    while j < vec_count:
        var acc = SIMD[DType.float32, VEC](0)
        for p in range(parts):
            acc += ws_ptr.load[width=VEC, alignment=VEC * 4](
                p * total + j * VEC
            )
        c_ptr.store[width=VEC, alignment=VEC * size_of[dtype]()](
            j * VEC, acc.cast[dtype]()
        )
        j += gstride
    var tail = vec_count * VEC + index
    while tail < total:
        var acc = Scalar[DType.float32](0)
        for p in range(parts):
            acc += ws_ptr[p * total + tail]
        c_ptr[tail] = acc.cast[dtype]()
        tail += gstride


@always_inline
def _amd_splitk_mfma_gemm[
    dtype: DType,
    BM: Int,
    BN: Int,
    WM: Int,
    WN: Int,
    BLOCK_K: Int = 32,
    STAGES: Int = 2,
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    parts: Int,
    ctx: DeviceContext,
) raises:
    comptime config = MatmulConfig[dtype, dtype, DType.float32, False](
        block_tile_shape=Index(BM, BN, BLOCK_K),
        warp_tile_shape=Index(WM, WN, BLOCK_K),
        mma_shape=get_mma_shape[dtype, DType.float32](),
        num_pipeline_stages=STAGES,
    )
    var total = m * n
    # Bounded by the dispatch rule: `tiles * parts` stays within a few
    # workgroups per CU, so `parts * m * n <= 4 * CUs * BM * BN` -- under 80 MB
    # of FP32 at any tile this route selects.  An ordinary per-call temporary,
    # freed on the same stream.
    var ws = ctx.enqueue_create_buffer[DType.float32](parts * total)
    ctx.enqueue_function[
        _amd_splitk_mfma_kernel[dtype, BM, BN, BLOCK_K, config]
    ](
        ws.unsafe_ptr().as_unsafe_any_origin(),
        _make_ptr[dtype](a_addr).as_unsafe_any_origin().as_immutable(),
        _make_ptr[dtype](b_addr).as_unsafe_any_origin().as_immutable(),
        m,
        n,
        k,
        k // parts,
        grid_dim=(n // BN, ceildiv(m, BM), parts),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.shared_mem_usage())
        ),
    )
    # The wide reduction needs a vector-aligned output; the caller has checked
    # the base address, and the kernel's own scalar tail covers an element count
    # the vector does not divide.
    comptime VEC = 16 // size_of[DType.float32]()
    _enqueue_cached[_splitk_reduce_kernel[dtype, VEC]](
        ctx,
        String(t"amd_splitk_reduce_{dtype}_v{VEC}"),
        _gs_blocks(total // VEC),
        1,
        1,
        256,
        _make_ptr[dtype](c_addr).as_unsafe_any_origin(),
        ws.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
        total,
        parts,
        total // VEC,
    )
    _ = ws^


@always_inline
def _splitk_parts(tiles: Int, cus: Int, k: Int, bk: Int) -> Int:
    """How many K slabs the runtime grid fill and the runtime K admit.

    Doubling the slab count doubles the grid, so it pays only while the output
    tiles alone cannot keep the CUs busy; past a few workgroups per CU the extra
    workspace traffic is pure cost.  Every slab must be a whole number of K tiles
    (`k % (parts * bk) == 0`) or the slabs would not cover K exactly, and each
    slab must keep enough contraction indices that its own pipeline fill and
    epilogue stay a small fraction of its work.
    """
    var parts = 1
    while (
        parts < 16
        and tiles * 2 * parts <= 4 * cus
        and k % (2 * parts * bk) == 0
        and k // (2 * parts) >= 2048
    ):
        parts *= 2
    # The loop only bounds the split from above. It must also be bounded from
    # below, because the divisibility and slab-depth conditions can stop the
    # doubling early and leave a split whose *product* still does not fill the
    # device -- and this clause preempts the in-workgroup-partition route, which
    # was tuned for exactly that underfilled regime, so a short split is worse
    # than not splitting at all.
    #
    # Measured against the pre-split-K dispatch, one case per process (the
    # harness runs cases in one process and later cases read low if earlier ones
    # warmed the device, which is how an earlier version of this table showed a
    # 49% win that isolation turned into a 14% loss):
    #
    #   tiles*parts   shape (m, n, k)          change
    #    24           128,  768,  8192         +32.1%
    #    72           768,  768,  8256         +70.6%
    #   128           512, 1024,  8192         +13.7%
    #   144           768,  768,  8192         +14.0%
    #   288           768,  768, 16384         -36.4%
    #   384          1024, 1536, 12288         -31.3%
    #   512          4096, 1024,  8192         -30.6%
    #   512          2048, 2048,  8192          -6.0%
    #   576           768,  768, 49152         -48.8%
    #
    # Every loss sits below 0.5 workgroups per CU and every win at or above
    # 0.95, so the floor is three quarters of one workgroup per CU. It separates
    # all nine, and it is the slab count rather than the tile count that decides:
    # 768x768 loses at k = 8192 and wins at k = 16384 on an identical grid of
    # output tiles.
    if tiles * parts * 4 < cus * 3:
        return 1
    return parts


# ---------------------------------------------------------------------------
# Batched MI300X MFMA GEMM.
#
# `multistage_gemm_kernel` derives its output-tile coordinates from block_idx.x
# and block_idx.y only, so block_idx.z is free to carry the batch index: a thin
# wrapper offsets the three operand pointers by their runtime batch strides and
# calls the same core.  That is exactly how MAX's own `batched_matmul_kernel_gpu`
# batches it (`max/kernels/src/linalg/bmm.mojo`, the `is_nvidia_gpu()` branch);
# the AMD branch there routes to `AMDMatmul`, which needs static N/K and is
# therefore unreachable from an eager backend holding runtime extents.
#
# Every extent, stride and batch stride stays a runtime value; only the tile
# geometry is compile-time, exactly as in the unbatched dispatch above.
# ---------------------------------------------------------------------------


# Causal regimes for the batched route.  A top-left-aligned causal mask makes
# the score matrix live only where `column <= row`, which the three attention
# GEMM shapes see in three different places:
#
#   CAUSAL_NONE   the ordinary dense GEMM.
#   CAUSAL_OUT    the output itself is the masked matrix (Q @ K^T and dO @ V^T).
#                 A whole output tile above the diagonal contributes nothing and
#                 is not computed *or written*: its consumer reads only the live
#                 prefix of every row.
#   CAUSAL_A_ROWS A is the masked matrix and K is its column axis (P @ V and
#                 dS @ K).  Output row block `r` can only see contraction
#                 indices below `(r + 1) * BM`, so the K loop is shortened.
#   CAUSAL_B_COLS B is the masked matrix and K is its row axis (dO^T @ P and
#                 Q^T @ dS).  Output column block `c` can only see contraction
#                 indices from `c * BN` up, so the K loop starts late.
#
# The last two are exact whatever the consumer does, because the skipped
# contraction indices multiply exact zeros, and they are what production selects.
#
# CAUSAL_OUT is implemented and exercised by `bench_attention_bmm --causal=1`, but
# it is deliberately *not* selected: at the attention shapes it measures slower
# than the dense GEMM (895.05 -> 901.91 us), because these GEMMs are bound by the
# rate at which the hardware can launch workgroups rather than by the work inside
# one, so removing work from a workgroup that is still launched buys nothing. See
# optimization_journal.md, diagnostic experiment AA. It also leaves the masked
# half of its output undefined, so a caller would additionally have to know its
# consumer reads no more than each row's live prefix.
comptime CAUSAL_NONE = 0
comptime CAUSAL_OUT = 1
comptime CAUSAL_A_ROWS = 2
comptime CAUSAL_B_COLS = 3


@__name(
    t"amd_batched_mfma_{dtype}_tb{transpose_b}_{BM}x{BN}x{BLOCK_K}_cz{CAUSAL}"
)
def _amd_batched_mfma_kernel[
    dtype: DType,
    transpose_b: Bool,
    BM: Int,
    BN: Int,
    BLOCK_K: Int,
    CAUSAL: Int,
    config: MatmulConfig[dtype, dtype, dtype, transpose_b],
](
    c_base: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
):
    """One batch element per block_idx.z, sharing the 2-D multistage core."""
    comptime assert (
        CAUSAL == CAUSAL_NONE or CAUSAL == CAUSAL_OUT or not transpose_b
    ), "the contraction-side causal regimes need a [K, N] B operand"
    var z = Int(block_idx.z)
    var row_block = Int(block_idx.y)
    var col_block = Int(block_idx.x)
    var c_ptr = c_base + z * c_bstride

    # A shortened or late-starting K range, in elements.  BM and BN are multiples
    # of BLOCK_K and the dispatch guarantees `k % 32 == 0`, so every bound below
    # is a whole number of K tiles and the loader never reads past the operand:
    # it consumes exactly `ceildiv(k_live, BLOCK_K)` tiles from the offset base.
    var k_off = 0
    var k_live = k
    comptime if CAUSAL == CAUSAL_OUT:
        # Entirely above the diagonal: nothing to accumulate, and the consumer
        # never reads it.
        if col_block * BN >= (row_block + 1) * BM:
            return
    comptime if CAUSAL == CAUSAL_A_ROWS:
        k_live = min(k, (row_block + 1) * BM)
    comptime if CAUSAL == CAUSAL_B_COLS:
        k_off = min(k, col_block * BN)
        k_live = k - k_off

    var c = TileTensor(c_ptr, row_major(Coord(m, n)))
    # `multistage_gemm_kernel` takes M from C and both N and K from B, so a
    # shortened K only has to be expressed in B's extent; A keeps its true row
    # stride and is merely offset along the contraction axis.
    var a = TileTensor(
        a_base + z * a_bstride + (k_off if not transpose_b else 0),
        row_major(Coord(m, k)),
    )
    var b = TileTensor(
        b_base + z * b_bstride + k_off * n,
        row_major(Coord(n, k)) if transpose_b else row_major(Coord(k_live, n)),
    )

    # The core's epilogue path already guards `row < m and col < n` before it
    # calls this, and the launch only covers whole BN column tiles, so no
    # further bound check is needed here.  It exists at all because the core's
    # own store path assumes an NVIDIA staging layout for half-float output.
    @always_inline
    @parameter
    def _store[
        value_dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], value: SIMD[value_dtype, width]):
        c_ptr.store[width=width, alignment=size_of[dtype]()](
            Int(coords[0]) * n + Int(coords[1]), value.cast[dtype]()
        )

    comptime if CAUSAL == CAUSAL_B_COLS:
        # A column block past the last live contraction index has an all-zero
        # output tile.  Write it rather than skip it: unlike CAUSAL_OUT this
        # region is inside the live part of the result.
        if k_live <= 0:
            var tid = Int(thread_idx.x)
            var threads = Int(block_dim.x)
            var row_end = min(m, (row_block + 1) * BM)
            var col_end = min(n, (col_block + 1) * BN)
            var index = tid
            var total = BM * BN
            while index < total:
                var row = row_block * BM + index // BN
                var col = col_block * BN + index % BN
                if row < row_end and col < col_end:
                    c_ptr[row * n + col] = Scalar[dtype](0)
                index += threads
            return

    multistage_gemm_kernel[
        config=config,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](_store),
    ](c, a, b)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(256))
)
@__name(t"amd_batched_mfma_edge_{dtype}_tb{transpose_b}_cz{CAUSAL}")
def _amd_batched_mfma_edge_kernel[
    dtype: DType, transpose_b: Bool, CAUSAL: Int
](
    c_base: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    c_bstride: Int,
    a_bstride: Int,
    b_bstride: Int,
    col_start: Int,
    col_count: Int,
):
    """Scalar cleanup for the trailing N columns of every batch element.

    The causal regimes are applied per element here rather than per tile, which
    is exact and needs no divisibility between the tile and the extents.
    """
    var idx = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if idx >= m * col_count:
        return
    var z = Int(block_idx.y)
    var row = idx // col_count
    var col = col_start + idx % col_count
    var a = a_base + z * a_bstride
    var b = b_base + z * b_bstride
    var k_begin = 0
    var k_end = k
    comptime if CAUSAL == CAUSAL_OUT:
        if col > row:
            (c_base + z * c_bstride)[row * n + col] = Scalar[dtype](0)
            return
    comptime if CAUSAL == CAUSAL_A_ROWS:
        k_end = min(k, row + 1)
    comptime if CAUSAL == CAUSAL_B_COLS:
        k_begin = min(k, col)
    var acc = Scalar[DType.float32](0)
    for kk in range(k_begin, k_end):
        var bv = b[col * k + kk] if transpose_b else b[kk * n + col]
        acc = (
            a[row * k + kk]
            .cast[DType.float32]()
            .fma(bv.cast[DType.float32](), acc)
        )
    (c_base + z * c_bstride)[row * n + col] = acc.cast[dtype]()


@always_inline
def _amd_batched_mfma_gemm[
    dtype: DType,
    BM: Int,
    BN: Int,
    WM: Int,
    WN: Int,
    transpose_b: Bool,
    BLOCK_K: Int = 32,
    WARP_K_PARTITIONS: Int = 1,
    STAGES: Int = 2,
    CAUSAL: Int = CAUSAL_NONE,
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ctx: DeviceContext,
) raises:
    # Same gfx942 transposed-B geometry preconditions as the unbatched route;
    # see `_amd_dynamic_mfma_gemm` and optimization_journal.md "Change 10".
    comptime MMA_DIM = get_mma_shape[dtype, DType.float32]()[0]
    comptime if transpose_b:
        comptime assert WM >= 2 * MMA_DIM and WN >= 2 * MMA_DIM, (
            "transposed-B multistage geometries with a single-MMA warp tile are"
            " miscompiled on gfx942; use WM, WN >= 2 * mma_dim"
        )
        comptime B_COPY_ROWS = min(
            (BM // WM) * (BN // WN) * WARP_K_PARTITIONS * 64,
            BN * BLOCK_K // simd_width_of[dtype](),
        ) * simd_width_of[dtype]() // BLOCK_K
        comptime assert (
            BN % B_COPY_ROWS == 0
        ), "transposed-B tile copy row count must divide BN"

    comptime config = MatmulConfig[dtype, dtype, dtype, transpose_b](
        block_tile_shape=Index(BM, BN, BLOCK_K),
        warp_tile_shape=Index(WM, WN, BLOCK_K),
        mma_shape=get_mma_shape[dtype, DType.float32](),
        num_pipeline_stages=STAGES,
        num_warp_k_partitions=WARP_K_PARTITIONS,
    )
    var c_ptr = _make_ptr[dtype](c_addr).as_unsafe_any_origin()
    var a_ptr = _make_ptr[dtype](a_addr).as_unsafe_any_origin().as_immutable()
    var b_ptr = _make_ptr[dtype](b_addr).as_unsafe_any_origin().as_immutable()
    var n_full = (n // BN) * BN
    if n_full > 0:
        ctx.enqueue_function[
            _amd_batched_mfma_kernel[
                dtype, transpose_b, BM, BN, BLOCK_K, CAUSAL, config
            ]
        ](
            c_ptr,
            a_ptr,
            b_ptr,
            m,
            n,
            k,
            m * n,
            a_bstride,
            n * k,
            grid_dim=(n_full // BN, ceildiv(m, BM), batch),
            block_dim=config.block_dim(),
            shared_mem_bytes=config.shared_mem_usage(),
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(config.shared_mem_usage())
            ),
        )
    if n_full < n:
        var col_count = n - n_full
        _enqueue_cached[
            _amd_batched_mfma_edge_kernel[dtype, transpose_b, CAUSAL]
        ](
            ctx,
            String(t"amd_batched_mfma_edge_{dtype}_tb{transpose_b}_cz{CAUSAL}"),
            ceildiv(m * col_count, 256),
            batch,
            1,
            256,
            c_ptr,
            a_ptr,
            b_ptr,
            m,
            n,
            k,
            m * n,
            a_bstride,
            n * k,
            n_full,
            col_count,
        )


@always_inline
def _batched_mfma_tile[
    dtype: DType, transpose_b: Bool, CAUSAL: Int = CAUSAL_NONE
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ctx: DeviceContext,
) raises:
    """Tile selection for the batched route, from the runtime operand shape.

    The batch alone gives the grid several workgroups per CU at any useful tile,
    so unlike the unbatched large-m regime the choice is not a grid-fill
    question: take the largest macro tile that both extents admit.  A tile wider
    than N would leave every column to the scalar cleanup kernel, which measured
    22.2 ms against 0.9 ms for a fitting tile, so BN never exceeds N.

    BN must also *divide* N, not merely fit inside it.  The MFMA core covers
    `(n // BN) * BN` columns and the residual goes to
    `_amd_batched_mfma_edge_kernel`, one thread per output element with a full
    runtime K loop -- about 34x the MFMA cost per column on the same table.
    Choosing BN by magnitude alone therefore turned any N that is not a tile
    multiple into a cliff: at N = 96 a 64-wide tile leaves a third of every row
    to the scalar path.  The caller guarantees `n % 32 == 0` and declines
    otherwise, so one of the three widths always divides N and the cleanup kernel
    handles only the M edge.

    `BLOCK_K` is 16 for float32 rather than 32 because two pipeline stages of a
    128x128x32 float32 tile request the entire 64 KB gfx942 LDS budget, which
    diagnostic experiment V showed does not launch.  At 16 it is 32 KB, and it is
    the same K depth the unbatched float32 deep-K route already uses.
    """
    comptime BK = 16 if dtype == DType.float32 else 32
    var wide_m = m >= 128
    if n % 128 == 0:
        if wide_m:
            _amd_batched_mfma_gemm[
                dtype, 128, 128, 32, 64, transpose_b, BK, 1, 2, CAUSAL
            ](c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx)
        else:
            _amd_batched_mfma_gemm[
                dtype, 64, 128, 32, 64, transpose_b, BK, 1, 2, CAUSAL
            ](c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx)
    elif n % 64 == 0:
        if wide_m:
            _amd_batched_mfma_gemm[
                dtype, 128, 64, 32, 64, transpose_b, BK, 1, 2, CAUSAL
            ](c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx)
        else:
            _amd_batched_mfma_gemm[
                dtype, 64, 64, 32, 32, transpose_b, BK, 1, 2, CAUSAL
            ](c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx)
    else:
        # n % 32 == 0 by the caller's guarantee.
        if wide_m:
            _amd_batched_mfma_gemm[
                dtype, 128, 32, 32, 32, transpose_b, BK, 1, 2, CAUSAL
            ](c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx)
        else:
            _amd_batched_mfma_gemm[
                dtype, 64, 32, 32, 32, transpose_b, BK, 1, 2, CAUSAL
            ](c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx)


@always_inline
def _causal_bmm_dispatch(
    dtype: DType,
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Int,
    causal: Int,
    ctx: DeviceContext,
) raises:
    """Batched GEMM that may skip contraction indices a causal mask kills.

    Raises when the causal MFMA route is not selectable, rather than quietly
    substituting the ordinary dense batched GEMM.  The substitution would be
    numerically fine, but it would also make this the last word for every
    device: the Python caller turns a refusal into `None` and then runs its own
    chain, which on sm_90a reaches the tuned BF16/TF32 batched bridges that a
    silent fallback here would preempt.  Every caller's fallback yields a dense
    result, which is a correct superset of all three causal regimes.

    `CAUSAL_OUT` leaves the masked half of the output unwritten, so the caller
    asks for it only when its consumer reads the live prefix of each row -- see
    `_sdpa_math_forward_with_dropout` and `_ScaledDotProductAttentionAutograd`.
    """
    # The MFMA tile calls must sit INSIDE the arch comptime gate: a runtime
    # `if eligible` does not stop instantiation, and these kernels use mma
    # ops that break the whole module build on Metal (same hazard as the
    # AmdBf16Tune dispatcher above).
    comptime if has_accelerator():
        comptime if _accelerator_arch() == "amdgpu:gfx942":
            var eligible = (
                batch > 1
                and m >= 64
                and n >= 64
                and n % 32 == 0
                and k % 32 == 0
            )
            if eligible:
                var handled = False
                comptime for dt in [DType.float32, DType.bfloat16]:
                    if dtype == dt:
                        if transpose_b != 0:
                            if causal == CAUSAL_OUT:
                                _batched_mfma_tile[dt, True, CAUSAL_OUT](
                                    c_addr,
                                    a_addr,
                                    b_addr,
                                    batch,
                                    m,
                                    n,
                                    k,
                                    m * k,
                                    ctx,
                                )
                                handled = True
                        elif causal == CAUSAL_A_ROWS:
                            _batched_mfma_tile[dt, False, CAUSAL_A_ROWS](
                                c_addr,
                                a_addr,
                                b_addr,
                                batch,
                                m,
                                n,
                                k,
                                m * k,
                                ctx,
                            )
                            handled = True
                        elif causal == CAUSAL_B_COLS:
                            _batched_mfma_tile[dt, False, CAUSAL_B_COLS](
                                c_addr,
                                a_addr,
                                b_addr,
                                batch,
                                m,
                                n,
                                k,
                                m * k,
                                ctx,
                            )
                            handled = True
                        elif causal == CAUSAL_OUT:
                            _batched_mfma_tile[dt, False, CAUSAL_OUT](
                                c_addr,
                                a_addr,
                                b_addr,
                                batch,
                                m,
                                n,
                                k,
                                m * k,
                                ctx,
                            )
                            handled = True
                if handled:
                    return
    raise Error(
        "causal batched GEMM is not selectable here: it needs gfx942, batch >"
        " 1, m and n >= 64, k % 32 == 0, and a causal mode this layout"
        " implements"
    )


# ---------------------------------------------------------------------------
# Hand-written NT (transposed-B) MFMA GEMM for gfx942 / CDNA3.
#
#   C[m, n] = sum_k A[m, k] * B[n, k]        A is (m, k), B is (n, k)
#
# Why this exists at all.  MAX's `multistage_gemm_kernel` cannot make both
# halves of the data path wide for this layout: with `transpose_b=True` its B
# LDS tile is (BN, BK), which gives k-contiguous LDS fragments but 64-byte
# global rows; with `transpose_b=False` it is (BK, BN), which gives wide global
# rows but lowers every MFMA B fragment to four 2-byte LDS reads.  Counters on
# the best configuration of the second route measured 2.30 LDS instructions and
# 11.08 bank-conflict cycles per MFMA against 4 cycles of per-CU MFMA
# throughput -- the LDS pipe oversubscribed 3.3x -- and that is the whole gap.
# See optimization_journal.md, "The remaining 3.17x, quantified".
#
# For NT both operands are already k-major in memory, and an MFMA operand wants
# exactly four elements adjacent along k, so a k-major LDS tile makes BOTH the
# global rows and the LDS fragment reads wide.  Every fragment is one
# `ds_read_b64`; nothing is transposed anywhere.
#
# Layout choices, all of them forced:
#
# * `v_mfma_f32_32x32x8bf16_1k` puts, in lane L, register p, the element
#   D[8*(p//4) + (p%4) + 4*(L//32), L%32]; its A operand wants
#   A[L%32, 4*(L//32) + j] and its B operand B[4*(L//32) + j, L%32].  Feeding
#   the A operand from the A tile and the B operand from the B tile therefore
#   gives an accumulator whose LANE index is n and whose REGISTER index is m.
#   A lane's sixteen values are sixteen rows of ONE column, so a store
#   instruction covers 32 consecutive n for a fixed row: 64 contiguous bytes per
#   half-wave, fully coalesced, with no LDS round trip in the epilogue.  The
#   other assignment (accumulator lane = m) would store four contiguous n per
#   lane spread over 32 rows, i.e. 16-byte fragments of 32 different cache
#   lines.
# * LDS rows are `BK + 4` elements.  A fragment read is `ds_read_b64` at
#   `(L%32)*PAD + c`, i.e. dword `(L%32)*(PAD/2) + c/2`, and `PAD/2 == 2 (mod 4)`
#   makes lanes 0..15 hit sixteen distinct even banks while the second dword of
#   each b64 fills the odd ones, so a 16-lane LDS cycle covers all 32 banks
#   exactly once.  `BK + 4` with `BK` a multiple of 8 always satisfies it.  This
#   is the same derivation as `flash_attention_fwd_kernels.mojo`, which measures
#   `SQ_LDS_BANK_CONFLICT == 0`.
# * The k loop is a two-stage software pipeline: the next tile's global loads
#   are issued BEFORE the current tile's MFMAs, so the load latency sits under
#   the matrix work, and with `STAGES == 2` the LDS write goes to the other
#   buffer so one barrier per iteration suffices.
#
# `EXACT` drops every edge guard when the runtime extents happen to divide the
# tile.  The masked instantiation is the general path: out-of-range rows and k
# columns are zero-filled in LDS, which contributes exactly nothing to any
# accumulator, and out-of-range outputs are not stored.  Both are selected from
# the runtime shape; no extent is ever compiled in.
# ---------------------------------------------------------------------------


@always_inline
def _nt_sched_fence():
    """Stop the machine scheduler moving anything across this point.

    `barrier()` alone is not sufficient on this pin: with a plain `barrier()`
    between the MFMA sequence that reads an LDS tile and the stores that refill
    it, a single-buffered k loop returns wrong results, and adding this fence on
    both sides of the barrier makes it correct.  Whatever the underlying defect
    is, `s_barrier` is not ordering the LDS traffic around it, so every phase
    boundary in the kernel below is fenced explicitly.
    """
    llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))


@always_inline
def _nt_barrier():
    """A phase boundary: no instruction and no LDS access crosses it."""
    _nt_sched_fence()
    barrier()
    _nt_sched_fence()


comptime NT_MAX_PARTS = 64  # most K slabs the split-K route will consider
comptime NT_MMA = 32  # both non-contraction extents of the MFMA tile
comptime NT_MMA_K = 8  # contraction extent of one MFMA
comptime NT_VEC = 8  # elements per global load and per LDS write
# k steps of MFMA issued before the next stage's refill, at BK / NT_MMA_K == 4
# steps per k tile.  Half is measured best for the layouts with a native
# operand; see `FILL_AT` in the kernel.
comptime NT_MID_FILL = 2
# The two-tile body (`WBODY`) wants the refill later still, and where depends on
# the layout: with both operands k-major the refill is pure `ds_write` and three
# k steps of four is measured best, with both native it carries the pair
# interleave and two is.  Both are 10-13% ahead of the one-tile body, and both
# are measured one shape per process -- see the journal.
comptime NT_BODY2_FILL = 3
comptime NT_BODY2_FILL_NATIVE = 2
# Bytes per microsecond the vectorized 2-D transpose sustains (journal change 28
# measured 3.10-3.74 TB/s on exactly these shapes) and FLOP per microsecond the
# k-major MFMA core sustains.  Both are used only to compare two routes' costs,
# never to predict a time.
comptime NT_T2_BYTES_PER_US = 3_300_000
comptime NT_GEMM_FLOP_PER_US = 500_000_000


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32((BM // WM) * (BN // WN) * 64)
    ),
)
@__name(
    t"nt_mfma_{dtype}_{otype}_{BM}x{BN}x{BK}_w{WM}x{WN}_s{STAGES}_z{SWIZZLE}_l{MASK_LOAD}_t{MASK_STORE}_a{A_KMAJOR}_b{B_KMAJOR}_p{SPLITK}"
)
def _nt_mfma_kernel[
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    STAGES: Int,
    SWIZZLE: Bool,
    MASK_LOAD: Bool,
    MASK_STORE: Bool,
    A_KMAJOR: Bool = True,
    B_KMAJOR: Bool = True,
    SPLITK: Bool = False,
    otype: DType = dtype,
    PAIR: Bool = not A_KMAJOR and not B_KMAJOR,
    FILL_AT: Int = 0,
    WBODY: Bool = False,
](
    c: UnsafePointer[Scalar[otype], MutAnyOrigin],
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    k_per: Int,
    xcds: Int,
):
    comptime WAVES_N = BN // WN
    comptime THREADS = (BM // WM) * WAVES_N * 64
    comptime MT = WM // NT_MMA
    comptime NTL = WN // NT_MMA
    comptime KSTEPS = BK // NT_MMA_K
    # LDS row stride.  `SWIZZLE` trades the four-element pad for an XOR
    # permutation of the four-element chunks within a row, which costs no LDS at
    # all and is what lets two pipeline stages of a 256x256x32 tile fit in the
    # 64 KB budget exactly.  `SW_SHIFT`/`SW_MASK` are derived below.
    comptime PAD = BK if SWIZZLE else BK + 4
    comptime CHUNKS = BK // 4  # four-element (one `ds_read_b64`) chunks per row
    # A row occupies `BK/2` dwords.  When that is 16 mod 32 the row index itself
    # already splits the 32 banks in two halves and the permutation only has to
    # be injective within a parity class, so it is driven by `row >> 1`; when it
    # is 0 mod 32 the row contributes nothing and the permutation must be
    # injective over all 16 rows of an LDS cycle.
    comptime SW_SHIFT = 1 if (BK // 2) % 32 == 16 else 0
    comptime SW_MASK = CHUNKS - 1
    comptime TPR = BK // NT_VEC  # threads that cover one tile row
    comptime ROWS = THREADS // TPR  # tile rows filled per pass
    # A NON-k-major operand lives in a `(k, X)` buffer, so its global rows run
    # along the non-contraction axis.  Turning its LDS tile k-major would put the
    # transpose on the GLOBAL side, where a lane can only fetch two bytes per row
    # and the load instruction count grows eightfold; leaving the tile native
    # puts it on the LDS side, where the tile is read four times over but each
    # access is cheap.
    #
    # The layout is `BK/2` rows of `2 * BX` elements, holding the k PAIR
    # `(2p, 2p+1)` of column `x` adjacent at `2x`.  That is what makes the
    # fragment read whole: an MFMA operand is four consecutive k of one column,
    # i.e. two adjacent pairs, so it is exactly two `ds_read_b32` whose two dwords
    # ARE the operand register pair -- no shifting, no `v_perm`, no packing at
    # all.  The one-element-per-row alternative needs four `ds_read_u16` and the
    # packing to go with them; measured with both operands native, that is 1.79x
    # the VALU count and 100 bytes of scratch per thread.
    #
    # Both accesses are bank-optimal with no padding and no permutation.  A
    # fragment read is dword `p * BX + x`: `BX` is a multiple of 32 so the pair
    # row drops out, the `lo` half-wave covers all 32 banks through `x`, and the
    # `hi` half-wave covers them again at different addresses -- 64 dwords over
    # 32 banks in the two cycles 256 bytes take anyway.  A fill write is
    # `ds_write_b128` at dword `p * BX + x` with `x` a multiple of NT_HVEC, which
    # is eight lanes per bank over 1024 bytes: also exactly the floor.
    #
    # Whether to pair is MEASURED, not assumed, and it splits on how many
    # operands are native.  With both native (TN) the pair layout is worth
    # 40%: 257 -> 360 TFLOP/s at (2304, 768, 49152) and 343 -> 517 at
    # (50304, 768, 49152), and it is what turns the copy-free transposed-A route
    # from 15-18% SLOWER than materializing A^T into 15-25% faster.  With one
    # native operand it is a wash that goes the wrong way on the shapes that
    # matter here: -0.5 to -2.9% on the grid-filling data gradients but +3 to
    # +4.9% on the split-K weight gradients (403.6 -> 423.3 us of GEMM at
    # (2304, 768, 49152)), because it trades one 16-byte global load for two
    # 8-byte ones and an interleave to halve LDS reads that were not binding.
    comptime NT_HVEC = NT_VEC // 2 if PAIR else NT_VEC
    comptime NROWS = 2 if PAIR else 1  # k rows one fill vector spans
    comptime XPA = BM // NT_HVEC  # threads covering one native-A fill row
    comptime XPB = BN // NT_HVEC
    comptime LRA = THREADS // XPA if XPA <= THREADS else 1  # fill rows per pass
    comptime LRB = THREADS // XPB if XPB <= THREADS else 1
    comptime APASS = (
        ceildiv(BM, ROWS) if A_KMAJOR else ceildiv(BK // NROWS, LRA)
    )
    comptime BPASS = (
        ceildiv(BN, ROWS) if B_KMAJOR else ceildiv(BK // NROWS, LRB)
    )
    comptime SA = BM * (PAD if A_KMAJOR else BK)
    comptime SB = BN * (PAD if B_KMAJOR else BK)

    # `WBODY` runs TWO k tiles per trip, one out of each LDS stage, and refills
    # each stage immediately after the barrier that proves every wave has read
    # it.  The stage pointers then never swap, so both stages' `ds_read` and
    # `ds_write` addresses are compile-time offsets from one base.  It is a
    # per-layout choice, measured: on (49152, 768, 3072) it is worth 13.2% with
    # both operands k-major and 10.2% with both native, and it LOSES 2% with one
    # of each.  It is also only worth having together with `FILL_AT >= 2`: at
    # `FILL_AT = 0` the same body measures 8.4% SLOWER than the one-tile body on
    # (49152, 768, 768), because each trip then has two refills with no queued
    # matrix work in front of either.
    #
    # Loading both k tiles in ONE global load per row -- 128 bytes, which is the
    # whole cache line a 64-byte k-major tile row half-fills -- was built and
    # measured on top of this and is NOT here: 480 us against 460 for the plain
    # two-tile body at (49152, 768, 3072).  The line reuse it was meant to
    # capture is real (`FETCH_SIZE` 482 MB at k = 3072 against 380 at k = 3104,
    # with an identical L1 miss count), but a second staging vector per operand
    # costs more than it recovers.
    comptime assert not (
        WBODY and MASK_LOAD
    ), "the two-tile body needs BK to divide the slab exactly"
    comptime assert (
        not WBODY or STAGES == 2
    ), "the two-tile body fills both pipeline stages per trip"
    comptime assert (
        0 <= FILL_AT and FILL_AT <= KSTEPS
    ), "the refill must land on a k step of the tile"
    comptime assert BK % NT_VEC == 0, "BK must be a multiple of the load width"
    comptime assert (
        WM % NT_MMA == 0 and WN % NT_MMA == 0
    ), "the warp tile must be a whole number of MFMA tiles"
    comptime assert THREADS >= TPR, "BK is wider than the block can cover"
    comptime assert STAGES == 1 or STAGES == 2, "STAGES is 1 or 2"
    comptime if PAIR:
        comptime assert BK % 2 == 0, "a paired native tile pairs adjacent k"
        comptime assert (
            NT_MMA_K % 4 == 0
        ), "an MFMA fragment must be a whole number of k pairs"
    comptime if not A_KMAJOR:
        comptime assert BM % 32 == 0, "a native tile row must span the banks"
        comptime assert (
            XPA <= THREADS and THREADS % XPA == 0
        ), "the block must cover a whole number of native-A pair rows"
    comptime if not B_KMAJOR:
        comptime assert BN % 32 == 0, "a native tile row must span the banks"
        comptime assert (
            XPB <= THREADS and THREADS % XPB == 0
        ), "the block must cover a whole number of native-B pair rows"
    comptime if SWIZZLE:
        # The permutation is a function of `row` only through
        # `(row >> SW_SHIFT) & SW_MASK`, so every row offset a fill pass or an
        # MFMA tile adds must be a multiple of its period for one precomputed
        # value per lane to stay valid.
        comptime SW_PERIOD = (SW_MASK + 1) << SW_SHIFT
        comptime assert (BK // 2) % 32 == 0 or (
            BK // 2
        ) % 32 == 16, "the swizzle is derived for LDS rows of 16 or 32 dwords"
        comptime assert (
            NT_MMA % SW_PERIOD == 0
        ), "the MFMA tile stride must be a whole number of swizzle periods"
        comptime assert (
            THREADS // TPR
        ) % SW_PERIOD == 0, (
            "the fill pass stride must be a whole number of swizzle periods"
        )

    comptime if is_amd_gpu():
        var smem = stack_allocation[
            STAGES * (SA + SB),
            dtype,
            address_space=AddressSpace.SHARED,
            alignment=16,
        ]()

        var tid = Int(thread_idx.x)
        var wave = tid // 64
        var lane = tid % 64
        var lo = lane % 32  # the MFMA's non-contraction index
        var hi = lane // 32  # which half-wave, i.e. which k quad
        var wm0 = (wave // WAVES_N) * WM
        var wn0 = (wave % WAVES_N) * WN
        # Which k slab this workgroup owns.  `k_per` is a whole number of k tiles
        # and `(parts - 1) * k_per < k <= parts * k_per`, both enforced by the
        # dispatch, so the slabs cover K exactly once and no loader reads past an
        # operand.  The LAST slab is short whenever `k_per` does not divide K; see
        # `n_kt` below.
        var slab = Int(block_idx.z) if SPLITK else 0

        # Output-tile coordinates.  With `xcds > 1` the linear workgroup index is
        # remapped so that each XCD receives a CONTIGUOUS band of the natural
        # ordering instead of every `xcds`-th tile: gfx942 dispatches workgroup i
        # to XCD `i % xcds`, so without this the tiles that share an A row block
        # land on different XCDs and each one's private L2 fetches its own copy.
        var m0 = Int(block_idx.y) * BM
        var n0 = Int(block_idx.x) * BN
        if xcds > 1:
            var nx = Int(grid_dim.x)
            # With k slabs on block_idx.z the dispatch order is z-major, so the
            # linear index the hardware maps to an XCD counts the slabs too; a
            # remap that ignored z would scatter each output tile's slabs across
            # XCDs and defeat the point, since those slabs share both operands'
            # row blocks.
            var plane = nx * Int(grid_dim.y)
            var total = plane * Int(grid_dim.z)
            var wgid = (
                Int(block_idx.z) * plane
                + Int(block_idx.y) * nx
                + Int(block_idx.x)
            )
            # Unlike MAX's own `_xcd_wgm_swizzle` this handles a tile count that
            # is not a multiple of the XCD count: XCDs below the remainder take
            # one extra tile, which keeps the map a bijection for every runtime
            # grid instead of falling back to row-major.
            var per = total // xcds
            var rem = total % xcds
            var xcd = wgid % xcds
            var slot = wgid // xcds
            var base = xcd * (per + 1) if xcd < rem else rem + xcd * per
            wgid = base + slot
            m0 = ((wgid % plane) // nx) * BM
            n0 = (wgid % nx) * BN
            comptime if SPLITK:
                slab = wgid // plane
        comptime if not MASK_LOAD and not MASK_STORE:
            # Edge tiles are SHIFTED BACK to end at the extent rather than
            # masked.  The masked instantiation costs 120 bytes of scratch per
            # thread -- the guards and the sixteen-register rounding block do not
            # coexist in 128 VGPRs -- and measures 2.2-3.2x slower, far more than
            # the 7% the transposed-B route recorded.  A shifted tile recomputes
            # rows another workgroup also computes, but with the same k loop, the
            # same MFMA order and therefore the same FP32 accumulation, so both
            # store bit-identical bytes to the overlap and the redundant work is
            # one tile out of `ceildiv(m, BM) * ceildiv(n, BN)`.  The caller only
            # selects this instantiation when `m >= BM` and `n >= BN`; the k edge
            # cannot be shifted and still selects the masked one.
            if m0 + BM > m:
                m0 = m - BM
            if n0 + BN > n:
                n0 = n - BN

        # Tile fill, k-major operand: one thread reads NT_VEC contiguous k of one
        # row, so a whole row of BK elements is 2*BK bytes read by TPR adjacent
        # threads.
        var trow = tid // TPR
        var tcol = (tid % TPR) * NT_VEC
        var pass_step = ROWS * k
        # Tile fill, native operand: one thread reads NT_HVEC contiguous
        # non-contraction elements from EACH of the two k rows of one pair, so
        # `BX / NT_HVEC` adjacent threads cover a whole pair row and the block
        # covers `THREADS / (BX / NT_HVEC)` pairs per pass.
        var akr = tid // XPA
        var axc = (tid % XPA) * NT_HVEC
        var bkr = tid // XPB
        var bxc = (tid % XPB) * NT_HVEC
        var k0 = slab * k_per
        var a_ptr = a + (
            (m0 + trow) * k
            + tcol
            + k0 if A_KMAJOR else (k0 + NROWS * akr) * m
            + m0
            + axc
        )
        var b_ptr = b + (
            (n0 + trow) * k
            + tcol
            + k0 if B_KMAJOR else (k0 + NROWS * bkr) * n
            + n0
            + bxc
        )
        var a_pass = pass_step if A_KMAJOR else NROWS * LRA * m
        var b_pass = pass_step if B_KMAJOR else NROWS * LRB * n
        var a_step = BK if A_KMAJOR else BK * m
        var b_step = BK if B_KMAJOR else BK * n
        # Where this thread's NT_VEC elements land in an LDS row.  Unswizzled
        # that is just the k offset; swizzled it is two four-element chunks at
        # XOR-permuted positions, so the store below is two `ds_write_b64`
        # instead of one `ds_write_b128` -- the same bytes and the same LDS
        # cycles, and no conditional half-swap.
        var lds_off = trow * PAD + tcol
        var sw_w = (trow >> SW_SHIFT) & SW_MASK
        var chunk0 = tcol // 4

        comptime LDSPtr = type_of(smem)

        var acc = stack_allocation[MT * NTL * 16, DType.float32]()
        comptime for i in range(MT * NTL):
            acc.store(i * 16, SIMD[DType.float32, 16](0))

        var areg = stack_allocation[APASS * NT_VEC, dtype]()
        var breg = stack_allocation[BPASS * NT_VEC, dtype]()
        comptime RegPtr = type_of(areg)

        @always_inline
        @parameter
        def _load(kt: Int):
            """Global -> registers for k tile `kt`, then advance the pointers.

            A pass covers `ROWS` tile rows.  When `ROWS` does not divide the tile
            extent the last pass is partial, and its guard is compile-time
            selected so the full passes stay branch-free.
            """
            var kok = True
            comptime if MASK_LOAD:
                kok = k0 + kt * BK + tcol < k
            comptime if A_KMAJOR:
                comptime for p in range(APASS):
                    var live = kok
                    comptime if (p + 1) * ROWS > BM:
                        live = live and trow + p * ROWS < BM
                    comptime if MASK_LOAD:
                        live = live and m0 + trow + p * ROWS < m
                    var v = SIMD[dtype, NT_VEC](0)
                    if live:
                        v = a_ptr.load[width=NT_VEC](p * pass_step)
                    areg.store(p * NT_VEC, v)
            else:
                comptime for p in range(APASS):
                    var live = True
                    comptime if (p + 1) * LRA > BK // NROWS:
                        live = akr + p * LRA < BK // NROWS
                    comptime if MASK_LOAD:
                        # `m % NT_VEC == 0` is a precondition of the native-A
                        # route, so a fill vector at an aligned offset is wholly
                        # inside the extent or wholly outside it: one guard, no
                        # per-element tail.  A tail written as predicated scalar
                        # loads costs 144 bytes of scratch per thread and
                        # measures 3.2x slower.
                        live = (
                            live
                            and m0 + axc < m
                            and k0
                            + kt * BK
                            + NROWS * (akr + p * LRA)
                            + NROWS
                            - 1
                            < k
                        )
                    var v0 = SIMD[dtype, NT_HVEC](0)
                    if live:
                        v0 = a_ptr.load[width=NT_HVEC](p * a_pass)
                    comptime if PAIR:
                        var v1 = SIMD[dtype, NT_HVEC](0)
                        if live:
                            v1 = a_ptr.load[width=NT_HVEC](p * a_pass + m)
                        areg.store(p * NT_VEC, v0.interleave(v1))
                    else:
                        areg.store(p * NT_VEC, v0)
            comptime if B_KMAJOR:
                comptime for p in range(BPASS):
                    var live = kok
                    comptime if (p + 1) * ROWS > BN:
                        live = live and trow + p * ROWS < BN
                    comptime if MASK_LOAD:
                        live = live and n0 + trow + p * ROWS < n
                    var v = SIMD[dtype, NT_VEC](0)
                    if live:
                        v = b_ptr.load[width=NT_VEC](p * pass_step)
                    breg.store(p * NT_VEC, v)
            else:
                comptime for p in range(BPASS):
                    var live = True
                    comptime if (p + 1) * LRB > BK // NROWS:
                        live = bkr + p * LRB < BK // NROWS
                    comptime if MASK_LOAD:
                        live = (
                            live
                            and n0 + bxc < n
                            and k0
                            + kt * BK
                            + NROWS * (bkr + p * LRB)
                            + NROWS
                            - 1
                            < k
                        )
                    var v0 = SIMD[dtype, NT_HVEC](0)
                    if live:
                        v0 = b_ptr.load[width=NT_HVEC](p * b_pass)
                    comptime if PAIR:
                        var v1 = SIMD[dtype, NT_HVEC](0)
                        if live:
                            v1 = b_ptr.load[width=NT_HVEC](p * b_pass + n)
                        breg.store(p * NT_VEC, v0.interleave(v1))
                    else:
                        breg.store(p * NT_VEC, v0)
            a_ptr += a_step
            b_ptr += b_step

        @always_inline
        @parameter
        def _write_row(
            dst: LDSPtr,
            row_base: Int,
            regs: RegPtr,
            reg_off: Int,
        ):
            comptime if SWIZZLE:
                comptime for h in range(NT_VEC // 4):
                    dst.store(
                        row_base + 4 * ((chunk0 + h) ^ sw_w),
                        regs.load[width=4](reg_off + 4 * h),
                    )
            else:
                dst.store(row_base + tcol, regs.load[width=NT_VEC](reg_off))

        # The interleaved k pair lands contiguously at `2 * x` of pair row `p`,
        # so the whole NT_VEC is still one `ds_write_b128`.
        @always_inline
        @parameter
        def _write_native(
            dst: LDSPtr,
            pair: Int,
            xcol: Int,
            w: Int,
            regs: RegPtr,
            reg_off: Int,
        ):
            comptime if PAIR:
                dst.store(
                    pair * 2 * w + 2 * xcol, regs.load[width=NT_VEC](reg_off)
                )
            else:
                # Unpaired, the fragment's four elements are `BX` apart and
                # `4 * BX * 2` bytes is a multiple of 128, so the `hi` half-wave
                # would land on the banks the `lo` half already holds.  Permuting
                # the row by `x ^ (32 * ((k >> 2) & 1))` costs no LDS and makes
                # them disjoint, and `(k >> 2) & 1` is exactly the reader's `hi`.
                dst.store(
                    pair * w + (xcol ^ (32 * ((pair >> 2) & 1))),
                    regs.load[width=NT_VEC](reg_off),
                )

        @always_inline
        @parameter
        def _fill(pa: LDSPtr, pb: LDSPtr):
            comptime if A_KMAJOR:
                comptime for p in range(APASS):
                    comptime if (p + 1) * ROWS <= BM:
                        _write_row(
                            pa, (trow + p * ROWS) * PAD, areg, p * NT_VEC
                        )
                    else:
                        if trow + p * ROWS < BM:
                            _write_row(
                                pa, (trow + p * ROWS) * PAD, areg, p * NT_VEC
                            )
            else:
                comptime for p in range(APASS):
                    comptime if (p + 1) * LRA <= BK // NROWS:
                        _write_native(
                            pa, akr + p * LRA, axc, BM, areg, p * NT_VEC
                        )
                    else:
                        if akr + p * LRA < BK // NROWS:
                            _write_native(
                                pa, akr + p * LRA, axc, BM, areg, p * NT_VEC
                            )
            comptime if B_KMAJOR:
                comptime for p in range(BPASS):
                    comptime if (p + 1) * ROWS <= BN:
                        _write_row(
                            pb, (trow + p * ROWS) * PAD, breg, p * NT_VEC
                        )
                    else:
                        if trow + p * ROWS < BN:
                            _write_row(
                                pb, (trow + p * ROWS) * PAD, breg, p * NT_VEC
                            )
            else:
                comptime for p in range(BPASS):
                    comptime if (p + 1) * LRB <= BK // NROWS:
                        _write_native(
                            pb, bkr + p * LRB, bxc, BN, breg, p * NT_VEC
                        )
                    else:
                        if bkr + p * LRB < BK // NROWS:
                            _write_native(
                                pb, bkr + p * LRB, bxc, BN, breg, p * NT_VEC
                            )

        # Every fragment of the whole k tile, read from LDS in one burst before
        # any MFMA issues.  `MT * KSTEPS + NTL * KSTEPS` `ds_read_b64` is only
        # (MT + NTL) * BK / 2 VGPRs -- 32 for a 64x64 warp tile at BK = 32 --
        # and it removes the per-k-step LDS dependency entirely: without it each
        # k step's MFMAs wait on that step's own reads, and the whole tile is a
        # chain of four such waits.
        var af = stack_allocation[MT * KSTEPS * 4, dtype]()
        var bf = stack_allocation[NTL * KSTEPS * 4, dtype]()

        @always_inline
        @parameter
        def _read_frags[S0: Int, S1: Int](pa: LDSPtr, pb: LDSPtr):
            # A fragment is the four k elements `8s + 4*hi ..+4` of one tile row,
            # i.e. chunk `2s + hi`, and `2s | hi == 2s ^ hi`, so the swizzled
            # position is `4 * ((2s) ^ (hi ^ sw))` with `2s` a compile-time
            # constant and `hi ^ sw` one precomputed value per lane.
            var abase = (wm0 + lo) * PAD
            var bbase = (wn0 + lo) * PAD
            var t0 = hi ^ ((lo >> SW_SHIFT) & SW_MASK)
            comptime if not SWIZZLE:
                abase += 4 * hi
                bbase += 4 * hi
            # Native side: the fragment is two adjacent k pairs of one column,
            # so it is two `SIMD[dtype, 2]` whose concatenation IS the operand
            # register pair.  Pair row `4s + 2*hi`, column `wx0 + 32i + lo`; the
            # `hi` term and the lane column are loop-invariant.
            var anb = 4 * hi * BM + (NROWS * (wm0 + lo) if PAIR else lo)
            var bnb = 4 * hi * BN + (NROWS * (wn0 + lo) if PAIR else lo)
            var aq = wm0 // NT_MMA
            var bq = wn0 // NT_MMA
            comptime for s in range(S0, S1):
                var kof = s * NT_MMA_K
                comptime if SWIZZLE:
                    kof = 4 * ((2 * s) ^ t0)
                comptime for i in range(MT):
                    comptime if A_KMAJOR:
                        af.store(
                            (s * MT + i) * 4,
                            pa.load[width=4](abase + i * NT_MMA * PAD + kof),
                        )
                    else:
                        var base = (
                            s * NT_MMA_K * BM
                            + anb
                            + (
                                2
                                * i
                                * NT_MMA if PAIR else NT_MMA
                                * ((aq + i) ^ hi)
                            )
                        )
                        comptime if PAIR:
                            af.store(
                                (s * MT + i) * 4,
                                pa.load[width=2](base).join(
                                    pa.load[width=2](base + 2 * BM)
                                ),
                            )
                        else:
                            var v = SIMD[dtype, 4]()
                            comptime for e in range(4):
                                v[e] = pa[base + e * BM]
                            af.store((s * MT + i) * 4, v)
                comptime for j in range(NTL):
                    comptime if B_KMAJOR:
                        bf.store(
                            (s * NTL + j) * 4,
                            pb.load[width=4](bbase + j * NT_MMA * PAD + kof),
                        )
                    else:
                        var base = (
                            s * NT_MMA_K * BN
                            + bnb
                            + (
                                2
                                * j
                                * NT_MMA if PAIR else NT_MMA
                                * ((bq + j) ^ hi)
                            )
                        )
                        comptime if PAIR:
                            bf.store(
                                (s * NTL + j) * 4,
                                pb.load[width=2](base).join(
                                    pb.load[width=2](base + 2 * BN)
                                ),
                            )
                        else:
                            var v = SIMD[dtype, 4]()
                            comptime for e in range(4):
                                v[e] = pb[base + e * BN]
                            bf.store((s * NTL + j) * 4, v)

        @always_inline
        @parameter
        def _do_mma[S0: Int, S1: Int]():
            comptime for s in range(S0, S1):
                comptime for i in range(MT):
                    comptime for j in range(NTL):
                        var d = acc.load[width=16]((i * NTL + j) * 16)
                        mma(
                            d,
                            af.load[width=4]((s * MT + i) * 4),
                            bf.load[width=4]((s * NTL + j) * 4),
                            d,
                        )
                        acc.store((i * NTL + j) * 16, d)

        var ca = smem
        var cb = smem + SA
        var na = smem + (SA + SB if STAGES == 2 else 0)
        var nb = na + SA

        # k tiles THIS workgroup runs.  Under `SPLITK` the slab length `k_per` is a
        # whole number of k tiles but need NOT divide the tile count: the last slab
        # takes whatever is left.  That is what frees the slab count to be chosen
        # for grid fill instead of for divisibility -- at (2304, 768, 49152) the
        # 27 output tiles want eleven slabs to cover 304 CUs, and 11 divides
        # neither 1536 nor anything near it.  The dispatch guarantees the short
        # slab is non-empty, and even whenever `WBODY` needs it to be.
        var n_kt = ceildiv(k, BK)
        comptime if SPLITK:
            var kt_per = k_per // BK
            n_kt = min(kt_per, n_kt - slab * kt_per)
        comptime if WBODY:
            # Two k tiles per trip, one out of each LDS stage.  Each stage's
            # refill is separated from the read of that same stage by a barrier,
            # and from the read of the OTHER stage by the next barrier, so the
            # two barriers per trip each do double duty and the barrier rate is
            # one per k tile -- exactly the one-tile body's.  What changes is
            # that a stage is refilled with the tile it will serve NEXT TRIP
            # rather than the one it serves later this trip, so the two stage
            # pointers are fixed and every LDS address is a compile-time offset
            # from one base.  The MFMA sequence still straddles each refill at
            # `FILL_AT`.
            _load(0)
            _fill(ca, cb)
            _load(1)
            _fill(na, nb)
            _nt_barrier()
            var wkt = 0
            while wkt + 2 < n_kt:
                _load(wkt + 2)
                _read_frags[0, KSTEPS](ca, cb)
                _do_mma[0, FILL_AT]()
                # Every wave has now read stage 0, and stage 1's refill from the
                # previous trip becomes visible here.
                _nt_barrier()
                _fill(ca, cb)
                _do_mma[FILL_AT, KSTEPS]()
                # The next tile goes into the same registers stage 0's refill
                # just consumed, and is consumed by stage 1's refill a whole
                # MFMA group and a barrier later.
                _load(wkt + 3)
                _read_frags[0, KSTEPS](na, nb)
                _do_mma[0, FILL_AT]()
                # Every wave has now read stage 1, and stage 0's refill above
                # becomes visible for the next trip.
                _nt_barrier()
                _fill(na, nb)
                _do_mma[FILL_AT, KSTEPS]()
                wkt += 2
            # The last trip's refill of stage 1 is the one write in the loop with
            # no barrier after it, so the tail needs one before it reads that
            # stage -- a wave reads fragments its neighbours wrote.  Without it
            # the all-ones and chunk gates still pass (stale LDS holds the same
            # value) and `--pattern-check=1`, whose operands are nonzero only on
            # the K edges, fails on every shape.
            _read_frags[0, KSTEPS](ca, cb)
            _do_mma[0, KSTEPS]()
            _nt_barrier()
            _read_frags[0, KSTEPS](na, nb)
            _do_mma[0, KSTEPS]()
        else:
            _load(0)
            _fill(ca, cb)
            _nt_barrier()

            var kt = 0
            while kt + 1 < n_kt:
                # Order matters and is measured, not assumed.  This tile's
                # fragments leave LDS into registers first, in one burst, and the
                # next tile's global loads are ISSUED next so their latency runs
                # under the MFMAs.  Reading the fragments one k step ahead of the
                # MFMA that needs them instead of in a burst -- which halves their
                # peak register footprint -- measures 3.4% slower weighted, and
                # putting the whole MFMA sequence AFTER the refill, which is what
                # the transposed-B work measured, is what the split below improves
                # on.
                #
                # Unrolling this by two so that both LDS stage bases become
                # compile-time offsets -- which removes all 28 `v_lshl_add_u32`
                # address instructions per wave per k tile, since the whole 64 KB
                # budget fits the 16-bit `ds_read` immediate -- measures 8-20%
                # SLOWER on every NN shape, with or without the global loads pinned
                # at the top of the body.  The address VALU is not on the critical
                # path, and the tight single-body loop is what lets the scheduler
                # issue a tile's global loads a whole body ahead of the `ds_write`
                # that consumes them.
                _read_frags[0, KSTEPS](ca, cb)
                _load(kt + 1)
                comptime if STAGES == 1:
                    # One buffer: every wave must be done reading it before it is
                    # refilled.
                    _nt_barrier()
                    _fill(ca, cb)
                    _do_mma[0, KSTEPS]()
                else:
                    # `FILL_AT` places the refill of the next stage inside the MFMA
                    # sequence rather than in front of all of it.  At `FILL_AT = 0`
                    # the refill's three `ds_write`s and the `s_waitcnt vmcnt(0)`
                    # that precedes them run before the matrix pipe has any queued
                    # work, and the `s_waitcnt lgkmcnt(0)` the barrier needs is a
                    # tail with nothing to cover it.  Issued halfway through, they
                    # go out after about a thousand cycles of MFMA -- long enough
                    # that the global loads have landed -- and retire under the
                    # remaining half.
                    #
                    # Where the optimum is depends on the body's instruction mix,
                    # so it is a per-route parameter, measured one case per process:
                    # for NN (`_dense_mfma_route`, B native) the weighted ratio is
                    # 1.085 at 0, 1.085 after one k step of four, **1.057** after
                    # two and 1.086 after three; for NT (`_nt_mfma_route`, both
                    # operands k-major, which reads twice as many `ds_read_b64` and
                    # runs no `v_perm`) two measures 1.116 against 1.109 at 0.
                    _do_mma[0, FILL_AT]()
                    _fill(na, nb)
                    _do_mma[FILL_AT, KSTEPS]()
                _nt_barrier()
                comptime if STAGES == 2:
                    var ta = ca
                    ca = na
                    na = ta
                    var tb = cb
                    cb = nb
                    nb = tb
                kt += 1
            _read_frags[0, KSTEPS](ca, cb)
            _do_mma[0, KSTEPS]()

        # Every accumulator is read and rounded HERE, unconditionally and in
        # straight-line code, and the fence below stops any of it from being
        # sunk into the guarded stores.  This is not decoration.  On this pin
        # gfx942 miscompiles a read of an MFMA destination that the scheduler
        # has placed too close to the MFMA: with the rounding left inside the
        # edge guard, the last two of the sixteen accumulator registers of the
        # FIRST accumulator read back their pre-MFMA value, so every output in
        # that 4x32 block is short by exactly one k step -- reproduced at
        # (128, 128, 768), where every guard is trivially true, so only the
        # branch differs.  `s_barrier` does not fix it and
        # `llvm.amdgcn.sched.barrier` does, which is what identifies it as
        # scheduling rather than synchronization.  It is the same defect family
        # as optimization_journal.md "Defect analysis D6", where MAX's
        # multistage kernel dropped one MFMA k step into two of four
        # accumulator elements and a deeper pipeline made it vanish.
        # Whether the whole accumulator set can be read at once.  With a BF16
        # output the rounded copy is 32 VGPRs on top of the 64 live
        # accumulators; with the FP32 workspace of the split-K route it is 64
        # more, and 64 + 64 plus the addresses does not fit 128 VGPRs -- the
        # instantiation reported **148 bytes of scratch per thread**.  So a
        # wide output dtype reads ONE accumulator at a time, 16 registers, each
        # read fenced on both sides, which keeps the invariant that matters
        # (no read of an MFMA destination adjacent to a branch) at a sixteenth
        # of the peak register cost.
        comptime OUT_ALL = size_of[otype]() <= size_of[dtype]()
        _nt_sched_fence()
        var out = stack_allocation[(MT * NTL if OUT_ALL else 1) * 16, otype]()
        comptime if OUT_ALL:
            comptime for i in range(MT * NTL):
                out.store(i * 16, acc.load[width=16](i * 16).cast[otype]())
        llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))

        # Register p of accumulator (i, j) is row
        # `wm0 + 32i + 8*(p//4) + (p%4) + 4*hi`, column `wn0 + 32j + lo`, so one
        # store instruction writes 32 consecutive columns of one row per
        # half-wave: 64 contiguous bytes, fully coalesced.
        # Under SPLITK each slab owns its own FP32 plane of the workspace; the
        # only rounding to the output dtype is the one the reduction performs,
        # exactly as in the unsplit path.
        var cp = c + (slab * m * n if SPLITK else 0)
        var cbase = (m0 + wm0 + 4 * hi) * n + n0 + wn0 + lo
        # Distance to the edge, so each guard is one compare against a
        # compile-time offset instead of a recomputed address.
        var rmax = m - (m0 + wm0 + 4 * hi)
        var cmax = n - (n0 + wn0 + lo)
        comptime for i in range(MT):
            comptime for j in range(NTL):
                comptime dc = j * NT_MMA
                comptime if not OUT_ALL:
                    _nt_sched_fence()
                    out.store(
                        0, acc.load[width=16]((i * NTL + j) * 16).cast[otype]()
                    )
                    llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](
                        Int32(0)
                    )
                comptime OFF = (i * NTL + j) * 16 if OUT_ALL else 0
                comptime if not MASK_STORE:
                    comptime for p in range(16):
                        comptime dr = i * NT_MMA + 8 * (p // 4) + (p % 4)
                        cp[cbase + dr * n + dc] = out[OFF + p]
                else:
                    if dc < cmax:
                        comptime for p in range(16):
                            comptime dr = i * NT_MMA + 8 * (p // 4) + (p % 4)
                            if dr < rmax:
                                cp[cbase + dr * n + dc] = out[OFF + p]


# Bytes per CU cycle the epilogue and the split-K workspace move.  Fitted from
# two points of ONE shape -- 256x256 at 8 and at 32 slabs on (2304, 768, 49152),
# which differ in workspace bytes by 4x and in wave count by 3x -- and the two
# constants that come out are 4.05 TB/s and 1.67 GHz, both physically sensible
# and both consistent with what change 27 measured for the reduction alone
# (4.6 TB/s out of the Infinity Cache) and with the 1395-1700 MHz `rocm-smi`
# reports under load.  See `_nt_ktile_cyc` for what the pair predicts.
comptime NT_BYTES_PER_CYC = 2425
# Fewest k tiles a split-K slab may hold.  A workgroup pays for two k tiles of
# global loads and LDS fills before its loop and for rounding sixteen
# accumulators after it, none of which `_nt_plan_cost` charges, so a slab of a
# few tiles is nearly all prologue -- and at that point this tile is the wrong
# kernel for the shape, not merely the wrong slab count.  Measured against the
# routes this one declines to, NN, one shape per process: a slab of 4 loses at
# (512, 3072, 768) 48.4 -> 57.5 us and at (512, 768, 768) 28.6 -> 43.5, while a
# slab of 12 wins at (2048, 768, 3072) 102.8 -> 72.0 and of 16 at
# (4096, 768, 3072) 109.5 -> 91.1.  The floor sits between them.  The old rule
# here -- a slab of at least 1024 k ELEMENTS, i.e. 32 tiles -- excluded those two
# wins along with the losses.
comptime NT_MIN_SLAB = 8
# Fewest k tiles of total work per CU for the split-K arm to be offered at all.
# See `_nt_best_parts`; the measured crossover is between 1.9 and 7.6.
comptime NT_MIN_WORK = 4


@always_inline
def _nt_slab_tiles(ktiles: Int, parts: Int) -> Int:
    """k tiles in every split-K slab but the last, for `parts` slabs of K.

    Even, because both k-loop bodies have to apply to it and the two-tile one
    needs an even count.  Rounding UP is what makes the last slab the short one;
    a `parts` whose rounded-up slab length would leave that last slab empty is
    rejected by the plan search -- `(parts - 1) * slab >= ktiles` -- because the
    reduction reads every plane and an empty slab would feed it a plane nobody
    wrote, and the two-tile body would read k tiles outside its own slab.
    """
    return ktiles if parts <= 1 else max(2, 2 * ceildiv(ktiles, 2 * parts))


@always_inline
def _nt_ktile_cyc(bm: Int, bn: Int) -> Int:
    """CU cycles one k tile of a `bm x bn` macro tile costs, at BK=32, warp 64x64.

    Counted from the geometry, exactly as diagnostic experiment AD counted it for
    256x256 (2048 MFMA + 1024 LDS read + 256 LDS write + 512 VMEM = 3840 against
    3995 measured):

    * MFMA.  A 64x64 warp tile is four 32x32x8 MFMAs per k step and four k steps,
      i.e. 512 cycles per wave per k tile, and a CU issues them on four SIMDs --
      so the tile costs `ceil(waves / 4)` rounds of 512.  The CEILING is not a
      detail: a 15-wave workgroup (320x192) occupies four rounds for the work of
      3.75 and measures exactly that, 16% off the model that divides instead.
    * LDS reads.  Each wave reads its own A and B slab once per k tile, 16
      `ds_read_b64` at 4 cycles.
    * LDS writes and VMEM.  The fill moves `(bm + bn) * BK * 2` bytes through a
      128 byte/cycle LDS port and through 64-byte global requests, which is
      `1.5 * (bm + bn)` cycles together.

    Measured against this model, one shape per process on (2304, 768, 49152),
    net of the workspace traffic the caller charges separately: 256x256 and
    128x384 hit it to within 0.5% at every slab count tried, and 256x192, 384x128,
    320x192, 192x320, 128x256 and 128x128 fall 7-39% short of it.  Only the two
    that hit it are instantiated, which is what makes the model a fair comparator
    rather than a guess -- see the journal for the whole table.
    """
    var waves = (bm // 64) * (bn // 64)
    return ceildiv(waves, 4) * 512 + waves * 64 + 3 * (bm + bn) // 2


@always_inline
def _nt_plan_cost(
    bm: Int,
    bn: Int,
    m: Int,
    n: Int,
    obytes: Int,
    ktiles: Int,
    parts: Int,
    cus: Int,
) -> Int:
    """CU cycles this (macro tile, slab count) plan costs on this runtime shape.

    One workgroup of these tiles is resident per CU, so `parts` slabs of a grid of
    `tiles` output tiles serialize in `ceil(tiles * parts / cus)` waves of
    `_nt_slab_tiles` k tiles each -- which is why a slab count that leaves a
    ragged last wave can beat a smaller one that fits in a single partial wave,
    and why the grid FILL rather than the tile size is what the choice turns on.
    The traffic term is the epilogue's store plus, when K is split, the FP32
    workspace written once and read once.  Every term is a runtime value.
    """
    var tiles = ceildiv(m, bm) * ceildiv(n, bn)
    var traffic = m * n * obytes
    if parts > 1:
        traffic += 2 * parts * m * n * 4
    # Scaled by `NT_BYTES_PER_CYC` so the comparison needs no division: this runs
    # on the host once per GEMM call, and an eager step issues 75 of them.
    return (
        ceildiv(tiles * parts, cus)
        * _nt_slab_tiles(ktiles, parts)
        * _nt_ktile_cyc(bm, bn)
        * NT_BYTES_PER_CYC
        + traffic
    )


@always_inline
def _nt_best_parts(
    bm: Int,
    bn: Int,
    m: Int,
    n: Int,
    obytes: Int,
    ktiles: Int,
    cus: Int,
    splittable: Bool,
    min_parts: Int,
) -> Int:
    """The slab count `_nt_plan_cost` minimizes for this macro tile, or 0 if the
    tile is no candidate for this shape at all.

    A tile is only a candidate when it fits inside the output: below that the
    kernel needs its edge-masked instantiation, which spills and measures 2.2-3.2x
    slower, so the caller is better off declining to the routes tuned for small
    grids.  `parts == 1` is likewise only offered when this tile's own output grid
    covers the device -- an unsplit wave that leaves CUs idle is exactly what
    those other routes beat by 19-49% (change 24) -- and when the caller allows
    it: `min_parts` is how a tile that is only instantiated in its split form
    says so.
    """
    if m < bm or n < bn:
        return 0
    var best = 0
    var chosen = 0
    var tiles = ceildiv(m, bm) * ceildiv(n, bn)
    var lo = max(min_parts, 1 if tiles >= cus else 2)
    var hi = NT_MAX_PARTS if splittable else 1
    # Splitting K is what covers the device when the output grid cannot, but it
    # cannot conjure WORK: below a few k tiles per CU the whole GEMM is prologue
    # and epilogue at this tile size, and the routes tuned for few output rows
    # win instead.  Measured, NN, one shape per process, against those routes:
    # this tile loses at 0.5, 1.0 and 1.9 k tiles per CU ((512, 768, 768)
    # 28.6 -> 48.2 us, (1024, 768, 768) 43.2 -> 51.1, (512, 3072, 768)
    # 47.7 -> 58.3) and wins from 7.6 up ((2048, 768, 3072) 102.1 -> 72.1,
    # (1024, 2304, 3072) 105.7 -> 81.3, (4096, 2304, 768) 109.2 -> 86.9).
    if lo > 1 and tiles * ktiles < NT_MIN_WORK * cus:
        return 0
    # Everything that does not depend on `parts` is hoisted, so a candidate costs
    # two integer divisions.
    var unit = _nt_ktile_cyc(bm, bn) * NT_BYTES_PER_CYC
    var store = m * n * obytes
    var plane = m * n * 8  # FP32 workspace, written once and read once
    for parts in range(lo, hi + 1):
        var slab = _nt_slab_tiles(ktiles, parts)
        if parts > 1 and ((parts - 1) * slab >= ktiles or slab < NT_MIN_SLAB):
            continue
        var cost = ceildiv(tiles * parts, cus) * slab * unit + store
        if parts > 1:
            cost += parts * plane
        if chosen == 0 or cost < best:
            best = cost
            chosen = parts
    return chosen


@always_inline
def _nt_mfma_gemm[
    dtype: DType,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    STAGES: Int = 2,
    SWIZZLE: Bool = False,
    A_KMAJOR: Bool = True,
    B_KMAJOR: Bool = True,
    SPLITK: Bool = False,
    otype: DType = dtype,
    PAIR: Bool = not A_KMAJOR and not B_KMAJOR,
    FILL_AT: Int = 0,
    WBODY: Bool = False,
    NOMASK: Bool = False,
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    parts: Int,
    xcds: Int,
    ctx: DeviceContext,
) raises:
    """Enqueue the MFMA GEMM, dropping the edge guards when they are unused.

    Both instantiations handle every runtime shape correctly; the guarded one
    spills and measures 2.2-3.2x slower, so the unguarded one shifts an edge tile
    back to end at the extent instead (see the kernel) and the guarded one is
    reached only by a k that BK does not divide.  `NOMASK` says the caller has
    already established that it cannot be -- `m >= BM`, `n >= BN` and `BK` divides
    `k` -- so it is not instantiated at all; that is worth having because every
    instantiation of this kernel costs eager-extension compile time.

    Under `SPLITK` the grid gains a z dimension of `parts` k slabs and the output
    is `parts` FP32 planes for the caller's reduction.  A slab is a whole number
    of k tiles -- `_nt_slab_tiles` derives it -- and the last slab is short when
    that count does not divide K; the caller must have used the same helper, so
    that the plane count it reduces is the plane count the grid writes.
    """
    comptime THREADS = (BM // WM) * (BN // WN) * 64
    var ktiles = ceildiv(k, BK)
    var kt_per = _nt_slab_tiles(ktiles, parts) if SPLITK else ktiles
    var k_per = kt_per * BK
    # k tiles the LAST slab runs.  The two-tile body needs an even count in every
    # slab, so it needs this one even too, and `_nt_slab_tiles` keeps `kt_per`
    # even -- which makes the remainder even exactly when the total is.
    var kt_last = ktiles - (parts - 1) * kt_per
    var grid = (ceildiv(n, BN), ceildiv(m, BM), parts)

    @always_inline
    @parameter
    def _go[MASKED: Bool, BODY2: Bool]() raises:
        ctx.enqueue_function[
            _nt_mfma_kernel[
                dtype,
                BM,
                BN,
                BK,
                WM,
                WN,
                STAGES,
                SWIZZLE,
                MASKED,
                MASKED,
                A_KMAJOR,
                B_KMAJOR,
                SPLITK,
                otype,
                PAIR,
                FILL_AT,
                BODY2,
            ]
        ](
            _make_ptr[otype](c_addr),
            _make_ptr[dtype](a_addr).as_immutable(),
            _make_ptr[dtype](b_addr).as_immutable(),
            m,
            n,
            k,
            k_per,
            xcds,
            grid_dim=grid,
            block_dim=(THREADS,),
        )

    if NOMASK or (m >= BM and n >= BN and k % BK == 0):
        # The wide-k body needs an EVEN number of k tiles in every slab, which
        # is a runtime property of the shape, so both bodies are instantiated
        # and the launch picks between them.  Both the full slabs and the short
        # last one have to qualify.
        comptime if WBODY:
            if kt_per % 2 == 0 and kt_per >= 2 and kt_last % 2 == 0:
                _go[False, True]()
            else:
                _go[False, False]()
        else:
            _go[False, False]()
    else:
        comptime if not NOMASK:
            _go[True, False]()


@always_inline
def _nt_mfma_route[
    dtype: DType
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Select an NT MFMA geometry for a runtime shape, or decline.

    One geometry wins on every grid-filled shape measured: 256x256x32 with a
    64x64 warp tile, sixteen wave64 per workgroup, two swizzled LDS stages.  The
    reason it wins is occupancy, not arithmetic intensity -- 64x64 is the widest
    warp tile whose sixteen accumulators still fit the 128 VGPRs a wave gets at
    four waves per SIMD, and the swizzle is what makes two stages of that tile
    exactly 65536 bytes, the whole gfx942 LDS budget for the one resident
    workgroup those sixteen waves already are.  Wider warp tiles measure 24-50%
    slower (128x64: 331, 128x128: 219 TFLOP/s against 434), and one padded LDS
    stage instead of two swizzled ones measures 13% slower.

    A smaller tile is selected only when the macro tile would exceed the output,
    and the route declines below that so the caller keeps its existing path
    rather than launching a workgroup that computes mostly nothing.
    """
    # A 16-byte vector load along k needs both operands 16-byte aligned, and
    # since the row stride IS k, `k % 8 == 0` is what makes every row of every
    # tile start aligned too.
    if k % NT_VEC != 0 or a_addr % 16 != 0 or b_addr % 16 != 0:
        return False
    # gfx942 dispatches workgroup i to XCD `i % xcds`, and each XCD has its own
    # L2, so the kernel remaps the linear workgroup index to give every XCD a
    # contiguous band of output tiles.  CDNA3 chiplets are 38 CUs, so the runtime
    # CU count gives the chiplet count; a wrong value costs locality only,
    # because the remap is a bijection for any value >= 1.  Measured, one case
    # per process: attn_c_proj 144.6 -> 138.2 us, mlp_c_proj 584.7 -> 530.2,
    # mlp_c_fc 554.1 -> 532.2, attn_c_attn 399.8 -> 406.2, lm_head 8588 -> 8619.
    var cus = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var xcds = max(1, cus // 38)
    # Only one workgroup of this tile is resident per CU -- sixteen wave64 and
    # the whole LDS budget -- so an output grid that does not cover the device
    # leaves CUs idle for the entire k loop with no second workgroup to fill
    # them.  Take the largest tile whose grid still covers every CU, and decline
    # below that so the caller keeps its existing route: measured, the 256x256
    # tile is 19-49% SLOWER than that route at (4096, 768, 3072),
    # (2048, 768, 3072) and (1024, 2304, 3072), whose grids are 48, 24 and 36
    # workgroups on 304 CUs.  Both the tile count and the CU count are runtime
    # values.
    if ceildiv(m, 256) * ceildiv(n, 256) >= cus:
        _nt_mfma_gemm[
            dtype,
            256,
            256,
            32,
            64,
            64,
            2,
            True,
            True,
            True,
            False,
            dtype,
            False,
            NT_BODY2_FILL,
            True,
        ](c_addr, a_addr, b_addr, m, n, k, 1, xcds, ctx)
        return True
    if ceildiv(m, 128) * ceildiv(n, 128) >= cus:
        _nt_mfma_gemm[
            dtype,
            128,
            128,
            32,
            64,
            64,
            2,
            True,
            True,
            True,
            False,
            dtype,
            False,
            NT_BODY2_FILL,
            True,
        ](c_addr, a_addr, b_addr, m, n, k, 1, xcds, ctx)
        return True
    return False


@always_inline
def _dense_mfma_route[
    dtype: DType, A_KMAJOR: Bool, B_KMAJOR: Bool
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """The same MFMA core for any combination of dense operand layouts.

    `A_KMAJOR` says A is `(m, k)` and `B_KMAJOR` says B is `(n, k)`; the other
    way round the operand lives in a `(k, X)` buffer, its global rows run along
    the non-contraction axis, and its LDS tile keeps that native shape.  See
    `_nt_mfma_kernel` for why that costs nothing in LDS bytes or cycles -- four
    `ds_read_u16` per fragment move the same 512 bytes in the same four LDS
    cycles as one `ds_read_b64`, and the row permutation keeps both halves of the
    wave in disjoint banks.  Measured with `SQ_LDS_BANK_CONFLICT`: zero.

    Three layouts reach this: NT (both k-major), NN (B native) and TN (both
    native, the weight gradient reading A straight out of its `(k, m)` buffer
    with no materialization at all).

    One workgroup of a tile this size is resident per CU, so an output grid that
    does not cover the device leaves CUs idle for the whole k loop.  Two macro
    tiles and every slab count are searched together by `_nt_plan_cost`, and when
    no plan is left the route declines and the caller keeps its own.
    """
    # Whether the native operand's LDS tile holds k PAIRS.  Pairing turns a
    # fragment into two `ds_read_b32` whose dwords ARE the operand register
    # pair -- half the LDS read cycles of the one-element-per-row layout, which
    # needs four `ds_read_u16` (a two-byte read still occupies a whole bank
    # slot, so 64 lanes take two LDS cycles for 128 bytes) plus two `v_perm_b32`
    # to pack.  It costs one extra global load instruction per native operand,
    # the same bytes and the same 64-byte request count.  With BOTH operands
    # native it is worth 40% and is never in doubt; with ONE it is measured, and
    # it splits on the regime -- see the journal.
    comptime if A_KMAJOR and not B_KMAJOR and size_of[dtype]() == 2:
        # The MIXED layout is the one the two-tile body loses on, and a k-major B
        # is one transpose away.  Measured one shape per process, this route
        # against `_nt_mfma_route` on the same extents:
        #   (49152,  768,   768)  138.9 -> 128.6 us
        #   (49152,  768,  2304)  353.0 -> 341.4
        #   (49152,  768,  3072)  462.6 -> 447.1
        #   (49152, 3072,   768)  522.4 -> 481.7
        #   (49152,  768, 50304) 6729.8 -> 6109.1
        # i.e. 3.3-9.2%, against a transpose of `4 * n * k` bytes.  Both sides
        # scale with `n * k`, so the whole condition reduces to a bound on m:
        # requiring the transpose to cost under a hundredth of the GEMM (a
        # quarter of the smallest measured gain) is
        #   4 * n * k / T2 <= 0.01 * 2 * m * n * k / GR   <=>   m >= 200 GR / T2
        # which is about 30300 on this part.  The arithmetic is unchanged -- the
        # k loop visits the same k in the same order -- so the outputs are
        # bit-identical to this route's.
        var xcus = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        if (
            m * NT_T2_BYTES_PER_US >= 200 * NT_GEMM_FLOP_PER_US
            and k % NT_VEC == 0
            and n % NT_VEC == 0
            and a_addr % 16 == 0
            and b_addr % 16 == 0
            and ceildiv(m, 256) * ceildiv(n, 256) >= xcus
        ):
            var bt = ctx.enqueue_create_buffer[dtype](n * k)
            _copy_strided[DType.uint16](
                Int(bt.unsafe_ptr()),
                b_addr,
                _rm2(n, k),
                _st2(k, 1),
                _st2(1, n),
                ctx,
            )
            var handled = _nt_mfma_route[dtype](
                c_addr, a_addr, Int(bt.unsafe_ptr()), m, n, k, ctx
            )
            _ = bt^
            if handled:
                return True
    # Whether the native operand's LDS tile holds k PAIRS.
    comptime PAIR_FILL = not A_KMAJOR or not B_KMAJOR
    comptime PAIR_SPLIT = not A_KMAJOR and not B_KMAJOR
    # The two-tile body pays when both operands have the SAME layout and loses
    # when they differ -- measured on (49152, 768, 3072): 530 -> 460 us with both
    # k-major, 531 -> 477 with both native, and 468 -> 478 (best of four refill
    # positions) with one of each.  So the mixed-layout NN keeps the one-tile
    # body and its own refill position.
    comptime BODY2 = A_KMAJOR == B_KMAJOR
    comptime FILL = NT_BODY2_FILL_NATIVE if BODY2 else NT_MID_FILL
    # Each operand's 16-byte rows run along its own contiguous axis, so it is
    # that axis's length that has to keep every tile row aligned.
    var alen = k if A_KMAJOR else m
    var blen = k if B_KMAJOR else n
    if (
        alen % NT_VEC != 0
        or blen % NT_VEC != 0
        or a_addr % 16 != 0
        or b_addr % 16 != 0
    ):
        return False
    var cus = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var xcds = max(1, cus // 38)
    var ktiles = ceildiv(k, 32)
    # Whether K can be split at all: a slab has to be a whole number of k tiles,
    # and the reduction reads the workspace 16 bytes at a time.
    var splittable = k % 32 == 0 and c_addr % 8 == 0
    # Plan search: for each candidate macro tile, every slab count up to
    # `NT_MAX_PARTS`, ranked by `_nt_plan_cost` on grid fill, wave quantization
    # and traffic.  Which plan wins is a property of the runtime shape, not of
    # this workload.  A tile is only a candidate when it fits inside the output,
    # because the edge-masked instantiation it would otherwise need spills and
    # measures 2.2-3.2x slower (see `_nt_mfma_gemm`).  `parts == 1` is only offered
    # when the tile's own grid covers the device: a single unsplit workgroup wave
    # that leaves CUs idle is what the routes below this one are for, and measured
    # 19-49% better there.
    var sq_parts = _nt_best_parts(
        256, 256, m, n, size_of[dtype](), ktiles, cus, splittable, 1
    )
    # The second macro tile, and the two conditions the sweep put on it.
    #
    # 128x384 is the only other shape of fifteen swept that reaches
    # `_nt_ktile_cyc` (diagnostic experiment AF), which is what makes it safe to
    # rank against 256x256 with a model rather than a table.  It is offered to the
    # PURE layouts only: in the MIXED layout, whose fill is one k tile at a time
    # and whose refill sits mid-sequence, it is 22% off its bound, and at
    # (768, 768, 49152) the model puts it 3% ahead of 256x256 while it measures
    # 175.2 us against 151.3.  A cost model cannot see that; the candidate set is
    # what carries it.  And it is offered with K SPLIT only (`min_parts = 2`),
    # which is the form measured -- (2304, 1152, 49152), an n that 256 quantizes
    # by 11% and 384 divides: 620.6 -> 566.9 us against this route, one shape per
    # process, interleaved.
    #
    # With the slab count free, 256x256 wins every shape nanoGPT itself issues, so
    # this tile is here for the shapes a different sequence length or hidden size
    # brings, not for that workload.
    var wide_parts = _nt_best_parts(
        128, 384, m, n, size_of[dtype](), ktiles, cus, splittable, 2
    ) if BODY2 else 0
    var b_wide = sq_parts == 0
    if sq_parts != 0 and wide_parts != 0:
        b_wide = _nt_plan_cost(
            128, 384, m, n, size_of[dtype](), ktiles, wide_parts, cus
        ) < _nt_plan_cost(
            256, 256, m, n, size_of[dtype](), ktiles, sq_parts, cus
        )
    var b_parts = wide_parts if b_wide else sq_parts
    if b_parts == 0:
        return False

    @always_inline
    @parameter
    def _launch[BM: Int, BN: Int, SPLIT_ONLY: Bool = False]() raises:
        comptime if not SPLIT_ONLY:
            if b_parts == 1:
                _nt_mfma_gemm[
                    dtype,
                    BM,
                    BN,
                    32,
                    64,
                    64,
                    2,
                    True,
                    A_KMAJOR,
                    B_KMAJOR,
                    False,
                    dtype,
                    PAIR_FILL,
                    FILL,
                    BODY2,
                ](c_addr, a_addr, b_addr, m, n, k, 1, xcds, ctx)
                return
        var ws = ctx.enqueue_create_buffer[DType.float32](b_parts * m * n)
        _nt_mfma_gemm[
            dtype,
            BM,
            BN,
            32,
            64,
            64,
            2,
            True,
            A_KMAJOR,
            B_KMAJOR,
            True,
            DType.float32,
            PAIR_SPLIT,
            FILL,
            BODY2,
            True,  # the plan search guarantees the tile fits and BK divides k
        ](
            Int(ws.unsafe_ptr()),
            a_addr,
            b_addr,
            m,
            n,
            k,
            b_parts,
            xcds,
            ctx,
        )
        comptime VEC = 16 // size_of[DType.float32]()
        _enqueue_cached[_splitk_reduce_kernel[dtype, VEC]](
            ctx,
            String(t"amd_splitk_reduce_{dtype}_v{VEC}"),
            _gs_blocks(m * n // VEC),
            1,
            1,
            256,
            _make_ptr[dtype](c_addr).as_unsafe_any_origin(),
            ws.unsafe_ptr().as_unsafe_any_origin().as_immutable(),
            m * n,
            b_parts,
            m * n // VEC,
        )
        _ = ws^

    # The mixed layout never selects the second tile, so it does not instantiate
    # it either.
    comptime if BODY2:
        if b_wide:
            _launch[128, 384, True]()
        else:
            _launch[256, 256]()
    else:
        _launch[256, 256]()
    return True


@always_inline
def _leading_trivial(a: TensorSpec) -> Bool:
    """True when only the two innermost dims of `a` are non-trivial."""
    for d in range(MAX_RANK - 2):
        if a.shape[d] != 1:
            return False
    return True


@always_inline
def _tn_mfma_route[
    dtype: DType
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """`C(m,n) = A(m,k) @ B(k,n)` with A read straight from its `(k, m)` buffer.

    This is the weight gradient, and it is the one layout where the existing
    dispatch had no kernel at all: it materialized A^T first, which costs a full
    read and write of A that PyTorch-ROCm never pays.  Both operands are native
    here, which is exactly the case the LDS tile shape above was written for.
    """
    return _dense_mfma_route[dtype, False, False](
        c_addr, a_addr, b_addr, m, n, k, ctx
    )


@always_inline
def _amd_bf16_large_m[
    dtype: DType, transpose_b: Bool, fuse_bias: Bool
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    bias_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    """Macro-tile selection for the many-output-rows regime.

    Pick the largest macro tile whose output-tile count still fills the device:
    a 128x128 tile has four times the arithmetic intensity of 32x128 but a
    quarter of the tiles, so it only pays once the grid covers every CU several
    times over.  Both the tile count and the CU count are runtime values.
    """
    var cus = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var tiles = ceildiv(m, 128) * ceildiv(n, 128)
    var wide = ceildiv(n, 128) >= 12
    if tiles >= 2 * cus or (wide and tiles >= cus):
        _amd_dynamic_mfma_gemm[dtype, 128, 128, 32, 64, transpose_b, fuse_bias](
            c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx
        )
    elif n >= 1536:
        _amd_dynamic_mfma_gemm[dtype, 64, 128, 32, 64, transpose_b, fuse_bias](
            c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx
        )
    else:
        _amd_dynamic_mfma_gemm[dtype, 32, 128, 32, 64, transpose_b, fuse_bias](
            c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx
        )


@always_inline
def _rm2(rows: Int, cols: Int) -> IndexList[MAX_RANK]:
    """Rank-MAX_RANK padded shape for a rows x cols matrix."""
    var out = IndexList[MAX_RANK](1)
    out[MAX_RANK - 2] = rows
    out[MAX_RANK - 1] = cols
    return out


@always_inline
def _st2(row_stride: Int, col_stride: Int) -> IndexList[MAX_RANK]:
    var out = IndexList[MAX_RANK](0)
    out[MAX_RANK - 2] = row_stride
    out[MAX_RANK - 1] = col_stride
    return out


@always_inline
def _amd_dynamic_mfma_dispatch[
    dtype: DType, transpose_b: Bool, fuse_bias: Bool = False
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    bias_addr: Int,
    ctx: DeviceContext,
) raises -> Bool:
    # The cutoffs describe reusable workload regimes. All tensor dimensions
    # stay runtime-dynamic inside every selected kernel.
    if a_bstride == 0 or k % 32 != 0:
        return False
    if batch != 1:
        # Batched: block_idx.z carries the batch index (see
        # `_amd_batched_mfma_gemm`).  The batch alone gives the grid several
        # workgroups per CU at any useful tile, so the tile is chosen from the
        # operand shape rather than from grid fill -- unlike the unbatched
        # large-m regime below.  A tile wider than N would leave every column
        # to the scalar cleanup kernel, so BN never exceeds N.
        comptime if fuse_bias:
            return False
        else:
            # n % 32 lets `_batched_mfma_tile` pick a BN that divides N, so the
            # scalar cleanup kernel only ever handles the M edge. Without it the
            # residual columns cost about 34x the MFMA per column, which made a
            # non-multiple N slower than the edge-masked tiled path this declines
            # to.
            if m < 64 or n < 64 or n % 32 != 0:
                return False
            _batched_mfma_tile[dtype, transpose_b](
                c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx
            )
            return True
    if m < 64:
        return False
    comptime if dtype == DType.bfloat16:
        # A K-dominant shape has few output columns per row block, so the
        # large-m macro tile has to beat the in-workgroup K partition on row
        # count alone. That crossover is at m = 2048, not 1024: measured on
        # transposed-B n = 768 k = 3072, the partitioned 32x32 route is 68%
        # faster at m = 1024 and 35% faster at m = 1536, breaks even at
        # m = 2048, and loses by 29% at m = 4096. Both branches keep every
        # extent runtime-dynamic; only which regime is selected depends on the
        # runtime shape.
        var deep_k = k >= 2048 and k >= 2 * n
        # Global split-K, for a deep contraction whose output tiles alone cannot
        # fill the device.  This is the weight-gradient regime: at
        # (2304, 768, 49152) the 128x128 output grid is 108 workgroups on a
        # 304-CU part, and no tile choice fixes that because the shortage is
        # output area, not tile size.  Slabs of K are the only axis left.
        #
        # Measured on the four such shapes the step issues (25 warmups, 100
        # synchronized iterations, A already contiguous so the numbers exclude the
        # transposed-operand copy): 2003 -> 1068 us, 742 -> 380, 2022 -> 1394 and
        # 2046 -> 1394 us, i.e. 87-114 -> 138-166 TFLOP/s, which is the same band
        # the well-filled forward and data-gradient shapes already reach.
        # `lm_head_wgrad`, whose 2358 tiles already fill the device, measures
        # 21126 us unsplit and 20989 with eight slabs, and the rule below leaves
        # it unsplit.
        #
        # `fuse_bias` is excluded because a bias would be added once per slab;
        # the reduction is where it would have to go, and no weight gradient has
        # one.  The C alignment condition is what lets the reduction use 16-byte
        # accesses; the `n % 128` condition removes the N-edge kernel entirely.
        comptime if not transpose_b and not fuse_bias:
            # Both operands dense, B in its native `(k, n)` layout: the same
            # hand-written MFMA core the transposed-B route uses, with B's LDS
            # tile left native instead of turned k-major.  It declines unless its
            # output grid covers the device, so the split-K and macro-tile routes
            # below still own the underfilled shapes.
            if _dense_mfma_route[dtype, True, False](
                c_addr, a_addr, b_addr, m, n, k, ctx
            ):
                return True
            if k >= 8192 and m >= 128 and n % 128 == 0 and c_addr % 8 == 0:
                var splitk_parts = _splitk_parts(
                    ceildiv(m, 128) * (n // 128),
                    ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT),
                    k,
                    32,
                )
                if splitk_parts > 1:
                    _amd_splitk_mfma_gemm[dtype, 128, 128, 32, 64](
                        c_addr, a_addr, b_addr, m, n, k, splitk_parts, ctx
                    )
                    return True
        if m >= 2048 or (m >= 1024 and not deep_k):
            comptime if transpose_b and not fuse_bias:
                # Both operands are k-major here, which is the one layout MAX's
                # multistage core cannot serve with wide global rows AND wide
                # LDS fragments at the same time.  `_nt_mfma_route` is a
                # hand-written kernel that has both; it measures 397-443
                # TFLOP/s against 152-175 for the route below, so try it first
                # and keep the B^T materialization only for the shapes it
                # declines.
                if _nt_mfma_route[dtype](c_addr, a_addr, b_addr, m, n, k, ctx):
                    return True
            comptime if transpose_b:
                # A [N, K] operand is read as BN rows of BLOCK_K elements, so
                # each global load touches only BLOCK_K * 2 = 64 bytes of a
                # 128-byte line, and the same tile is re-read by every row
                # block.  Materializing B^T once costs 4 * n * k bytes of
                # bandwidth -- three orders of magnitude less than the GEMM at
                # these row counts -- and the [K, N] kernel then runs about
                # 1.4x faster.  Below m = 1024 the GEMM is too small to
                # amortize the copy, and that regime keeps the [N, K] kernel.
                var bt = ctx.enqueue_create_buffer[dtype](k * n)
                _copy_strided[dtype](
                    Int(bt.unsafe_ptr()),
                    b_addr,
                    _rm2(k, n),
                    _st2(n, 1),
                    _st2(1, k),
                    ctx,
                )
                _amd_bf16_large_m[dtype, False, fuse_bias](
                    c_addr,
                    a_addr,
                    Int(bt.unsafe_ptr()),
                    bias_addr,
                    m,
                    n,
                    k,
                    ctx,
                )
                _ = bt^
            else:
                _amd_bf16_large_m[dtype, transpose_b, fuse_bias](
                    c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx
                )
            return True
        if k >= 2048 and k >= 2 * n:
            # Deep-K. A wide N still has enough output columns to feed a
            # 32x128 tile, which beats the in-workgroup K partition once the
            # column count stops being the scarce resource.
            if n >= 2048:
                _amd_dynamic_mfma_gemm[
                    dtype, 32, 128, 32, 64, transpose_b, fuse_bias
                ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
                return True
            # In-workgroup K partitions: the K loop is long and the 32x32
            # output grid alone can leave most SIMDs with less than one
            # resident wave, so extra waves hide global-load latency for more
            # than the in-workgroup reduction costs.
            #
            # The partition count may only divide a K the kernel can split
            # exactly. `multistage_gemm_kernel` starts partition p at K tile
            # `(k / BLOCK_K) // parts * p` and gives every partition
            # `ceildiv(k / parts, BLOCK_K)` tiles, both floor divisions, so
            # unless `(k / BLOCK_K) % parts == 0` the partitions overlap on
            # some tiles and leave the last ones uncovered -- silently wrong
            # output, not a crash. With BLOCK_K == 32 that means k % 128 for
            # four partitions and k % 64 for two. Step down to a count this
            # runtime k admits rather than restricting which k reaches here.
            #
            # The count is also an occupancy decision, not a fixed choice:
            # extra partitions only pay while the output grid alone cannot keep
            # every CU busy, and past that they add reduction work to an
            # already-occupied device. Measured on transposed-B n = 768,
            # k = 3072, four partitions win by 8% at m = 512 (384 tiles against
            # 304 CUs) and lose by 6% at m = 1024 (768 tiles). Both the tile
            # count and the CU count are runtime values.
            var partition_cus = ctx.get_attribute(
                DeviceAttribute.MULTIPROCESSOR_COUNT
            )
            var partition_tiles = ceildiv(m, 32) * ceildiv(n, 32)
            if k % 128 == 0 and partition_tiles < 2 * partition_cus:
                _amd_dynamic_mfma_gemm[
                    dtype, 32, 32, 32, 32, transpose_b, fuse_bias, 32, 4
                ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
            elif k % 64 == 0:
                _amd_dynamic_mfma_gemm[
                    dtype, 32, 32, 32, 32, transpose_b, fuse_bias, 32, 2
                ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
            else:
                _amd_dynamic_mfma_gemm[
                    dtype, 32, 32, 32, 32, transpose_b, fuse_bias, 32, 1
                ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
            return True
        comptime if not transpose_b:
            if n >= 512 and n <= 2304 and k >= 512 and k % 64 == 0:
                _amd_dynamic_mfma_gemm[
                    dtype, 32, 32, 32, 32, transpose_b, fuse_bias, 32, 2
                ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
                return True
            if n > 2304 and n < 8192 and k >= 512:
                _amd_dynamic_mfma_gemm[
                    dtype, 64, 32, 32, 32, transpose_b, fuse_bias
                ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
                return True
    if m >= 128 and n >= 8192 and k >= 128:
        _amd_dynamic_mfma_gemm[dtype, 128, 128, 64, 64, transpose_b, fuse_bias](
            c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx
        )
    elif k >= 2048 and k >= 2 * n:
        comptime if transpose_b:
            comptime if dtype == DType.float32:
                # Same exact-split requirement as the BF16 deep-K route above,
                # at BLOCK_K == 16: four partitions need k % 64, two need
                # k % 32. The outer guard only promises k % 32.
                if k % 64 == 0:
                    _amd_dynamic_mfma_gemm[
                        dtype, 32, 32, 32, 32, transpose_b, fuse_bias, 16, 4
                    ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
                else:
                    _amd_dynamic_mfma_gemm[
                        dtype, 32, 32, 32, 32, transpose_b, fuse_bias, 16, 2
                    ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
            else:
                # A 16x16 warp tile is one of the miscompiled transposed-B
                # geometries (see the guard in `_amd_dynamic_mfma_gemm`).
                _amd_dynamic_mfma_gemm[
                    dtype, 32, 32, 32, 32, transpose_b, fuse_bias
                ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        else:
            _amd_dynamic_mfma_gemm[
                dtype, 32, 32, 16, 16, transpose_b, fuse_bias
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
    else:
        comptime if transpose_b:
            # 32x64 with a 16x32 warp tile is the miscompiled transposed-B
            # geometry that produced the silently wrong nanoGPT forward GEMMs;
            # a 32x32 warp tile computes the same tile correctly.
            _amd_dynamic_mfma_gemm[
                dtype, 32, 64, 32, 32, transpose_b, fuse_bias
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        else:
            _amd_dynamic_mfma_gemm[
                dtype, 32, 64, 16, 32, transpose_b, fuse_bias
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
    return True


# Benchmark-only MFMA configuration sweep. This is deliberately not called by
# eager dispatch: it allows reusable runtime-shape schedules to be compared in
# one compiled extension build before a single winning hypothesis changes
# production. The body is gated on gfx942 at compile time: this function is
# non-parameterized, so it is always codegen'd, and instantiating the MFMA
# multistage GEMM on non-AMD targets (Apple in particular has no `mma`)
# breaks the whole module build.
def _amd_bf16_tune_dispatcher(
    c_obj: PythonObject,
    a_obj: PythonObject,
    b_obj: PythonObject,
    bias_obj: PythonObject,
    # (m, n, k, config_id)
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    comptime if _accelerator_arch() != "amdgpu:gfx942":
        raise Error("AmdBf16Tune requires an AMD gfx942 accelerator")
    else:
        var c_addr = Int(py=c_obj)
        var a_addr = Int(py=a_obj)
        var b_addr = Int(py=b_obj)
        var bias_addr = Int(py=bias_obj)
        var m = Int(py=params[0])
        var n = Int(py=params[1])
        var k = Int(py=params[2])
        var cfg = Int(py=params[3])
        var ctx = _get_ctx(device_context_ptr)

        if cfg == 0:
            _amd_dynamic_mfma_gemm[DType.bfloat16, 32, 64, 16, 32, False, True](
                c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx
            )
        elif cfg == 1:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 32, 128, 16, 64, False, True
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 2:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 32, 128, 32, 64, False, True
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 3:
            _amd_dynamic_mfma_gemm[DType.bfloat16, 64, 64, 32, 32, False, True](
                c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx
            )
        elif cfg == 4:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 64, 128, 32, 64, False, True
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 5:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 96, 64, 48, 32, False, True, 64
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 6:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 128, 64, 64, 32, False, True
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 7:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 32, 32, 32, 32, False, True, 64, 2
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 8:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 32, 64, 32, 32, False, True, 64, 2
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 9:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 64, 32, 32, 32, False, True, 64, 2
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 15:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 32, 32, 32, 32, False, True, 32, 2
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 16:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 32, 32, 32, 32, False, True, 64, 4
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 17:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 32, 32, 32, 32, False, True, 64, 2, 2
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 21:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 32, 64, 16, 32, False, True, 64, 1, 2
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        elif cfg == 18:
            _amd_dynamic_mfma_gemm[DType.bfloat16, 16, 32, 16, 32, False, True](
                c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx
            )
        elif cfg == 19:
            _amd_dynamic_mfma_gemm[DType.bfloat16, 16, 64, 16, 32, False, True](
                c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx
            )
        elif cfg == 20:
            _amd_dynamic_mfma_gemm[DType.bfloat16, 64, 32, 32, 32, False, True](
                c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx
            )
        elif cfg == 22:
            _amd_dynamic_mfma_gemm[
                DType.bfloat16, 128, 32, 64, 32, False, True
            ](c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx)
        else:
            raise Error("unknown AMD BF16 MFMA tune config")


# ---------------------------------------------------------------------------
# Dispatch: pick a kernel/config from the runtime shape.
# ---------------------------------------------------------------------------


@always_inline
def _enqueue_pipe[
    BM: Int, BN: Int, BK: Int, TM: Int, TN: Int, transpose_b: Bool
](
    ctx: DeviceContext,
    gx: Int,
    gy: Int,
    gz: Int,
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
) raises:
    comptime THREADS = (BM // TM) * (BN // TN)
    var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
    var a = (
        _make_ptr[DType.float32](a_addr).as_unsafe_any_origin().as_immutable()
    )
    var b = (
        _make_ptr[DType.float32](b_addr).as_unsafe_any_origin().as_immutable()
    )
    var va4 = k % 4 == 0  # A (and B when transposed) is loaded along k

    comptime if transpose_b:
        if va4:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 4, 4, True]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v44_tb1"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        else:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 1, 1, True]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v11_tb1"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
    elif BM >= 128:
        # Compute-bound fat tiles: 2-stage with transposed-A vector frags.
        var vb4 = n % 4 == 0  # B is loaded along n
        if va4 and vb4:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 4, 4, False]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v44_tb0"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        elif va4:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 4, 1, False]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v41_tb0"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        elif vb4:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 1, 4, False]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v14_tb0"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        else:
            _enqueue_cached[_gemm_pipe_kernel[BM, BN, BK, TM, TN, 1, 1, False]](
                ctx,
                String(t"gemm_pipe_{BM}x{BN}_v11_tb0"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
    else:
        # Latency-bound thin tiles: 3-stage cp.async pipeline.
        var vb4 = n % 4 == 0  # B is loaded along n
        if va4 and vb4:
            _enqueue_cached[
                _gemm_pipe3_kernel[
                    BM, BN, BK, TM, TN, 4, 4, False, PIPE3_STAGES, False, 4
                ]
            ](
                ctx,
                String(t"gemm_pipe3_{BM}x{BN}_v44_mb4"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
                a,
            )
        elif va4:
            _enqueue_cached[
                _gemm_pipe3_kernel[
                    BM, BN, BK, TM, TN, 4, 1, False, PIPE3_STAGES, False, 4
                ]
            ](
                ctx,
                String(t"gemm_pipe3_{BM}x{BN}_v41_mb4"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
                a,
            )
        elif vb4:
            _enqueue_cached[
                _gemm_pipe3_kernel[
                    BM, BN, BK, TM, TN, 1, 4, False, PIPE3_STAGES, False, 4
                ]
            ](
                ctx,
                String(t"gemm_pipe3_{BM}x{BN}_v14_mb4"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
                a,
            )
        else:
            _enqueue_cached[
                _gemm_pipe3_kernel[
                    BM, BN, BK, TM, TN, 1, 1, False, PIPE3_STAGES, False, 4
                ]
            ](
                ctx,
                String(t"gemm_pipe3_{BM}x{BN}_v11_mb4"),
                gx,
                gy,
                gz,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
                a,
            )


@always_inline
def _gemm_enqueue[
    dtype: DType, transpose_b: Bool
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        comptime if (
            dtype in (DType.float32, DType.bfloat16)
            and _accelerator_arch() == "amdgpu:gfx942"
        ):
            if _amd_dynamic_mfma_dispatch[dtype, transpose_b](
                c_addr,
                a_addr,
                b_addr,
                batch,
                m,
                n,
                k,
                a_bstride,
                0,
                ctx,
            ):
                return

        var a = _make_ptr[dtype](a_addr).as_unsafe_any_origin().as_immutable()
        var b = _make_ptr[dtype](b_addr).as_unsafe_any_origin().as_immutable()

        # Skinny-M float32 paths (decode-step GEMMs: m = batch <= 32). The
        # BM=64 tiles waste half of every block there; the BM=32 pipe3 tile
        # (128-thread blocks, k-chunk floor 96) beats cuBLAS SGEMM on the
        # GPT-2 decode shapes. For transposed-B (linear/lm_head), compute
        # C^T = B @ A^T instead so B streams via cp.async at full rate.
        comptime if dtype == DType.float32:
            # Apple: everything fatter than the skinny band runs the 64x64
            # simdgroup-matrix tile — the SIMT pipe kernels below are tuned
            # for NVIDIA and run 3-15x behind torch MPS on Metal. Shapes with
            # enough K-depth and mostly-interior tiles take the threadgroup-
            # staged variant (single wide load per element per block); short-K
            # or edge-dominated shapes keep the leaner direct-load kernel.
            # Degenerate n stays on the SIMT path (a 64-wide tile would be
            # mostly idle).
            comptime if has_apple_gpu_accelerator():
                if m > 32 and n >= 16:
                    comptime if not transpose_b:
                        # NN layout: dedicated kernels (see
                        # apple_gemm_nn_kernels.mojo). Both use supertile
                        # rasterization (SWIZZLE=4) — without it these
                        # DRAM-bound shapes are bimodal, allocation-address
                        # dependent, up to 2x slower. Mid-K keeps the
                        # direct-load kernel (staging store/barrier overhead
                        # outweighs once-per-block loads until the K loop is
                        # long); deep-K takes the staged kernel.
                        if k >= 192 and m >= 96 and n >= 96:
                            if k <= 640:
                                apple_nn_direct_enqueue[64, 64, 2, 2, 4](
                                    c_addr,
                                    a_addr,
                                    b_addr,
                                    batch,
                                    m,
                                    n,
                                    k,
                                    a_bstride,
                                    ctx,
                                )
                            else:
                                apple_nn_smem_enqueue[
                                    64, 64, 16, 2, 2, 2, True, 4
                                ](
                                    c_addr,
                                    a_addr,
                                    b_addr,
                                    batch,
                                    m,
                                    n,
                                    k,
                                    a_bstride,
                                    ctx,
                                )
                        else:
                            _apple8_fat_enqueue[transpose_b](
                                c_addr,
                                a_addr,
                                b_addr,
                                batch,
                                m,
                                n,
                                k,
                                a_bstride,
                                ctx,
                            )
                        return
                    # NT layout (linear forward x @ W^T, SDPA's Q @ K^T):
                    # dedicated kernels, see apple_gemm_nt_kernels.mojo.
                    # Staging pays off from k = 256 up for every MN tile
                    # shape, narrow N included (2048x65x384 runs 25% faster
                    # staged): B's k-contiguous rows come in as float4s and
                    # are transposed on the threadgroup store, so the mma
                    # sweep never issues the stride-k scalar pairs the direct
                    # kernel needs for B fragments. Below that the K loop is
                    # too short to amortize the staging barriers and the
                    # direct-load kernel wins (SDPA head dims land there).
                    # Unlike NN, supertile rasterization loses here — see the
                    # NT file header.
                    comptime if transpose_b:
                        if k >= 256:
                            apple_nt_smem_enqueue[64, 64, 16, 2, 2, 2, True, 0](
                                c_addr,
                                a_addr,
                                b_addr,
                                batch,
                                m,
                                n,
                                k,
                                a_bstride,
                                ctx,
                            )
                        else:
                            apple_nt_direct_enqueue[64, 64, 2, 2, 0](
                                c_addr,
                                a_addr,
                                b_addr,
                                batch,
                                m,
                                n,
                                k,
                                a_bstride,
                                ctx,
                            )
                    return
            comptime if transpose_b:
                # Apple M1-M4's stock 8x8 simdgroup-matrix kernel uses a
                # 64-row tile. The 32-row variant keeps all four simdgroups
                # doing useful work on skinny-M projections (decode-step
                # lm_head), for any m in the skinny band — the kernel edge-
                # masks partial tiles.
                comptime if has_apple_gpu_accelerator():
                    if (
                        batch == 1
                        and m > SMALLM_MR
                        and m <= 32
                        and n >= 8192
                        and k % 16 == 0
                    ):
                        _apple8_enqueue[False, True](
                            c_addr, a_addr, b_addr, 0, m, n, k, ctx
                        )
                        return
                # Fat C^T path for medium-m transposed-B (lm_head at large
                # batch): batch==1 only, grid must fill the GPU on its own.
                if (
                    batch == 1
                    and m > 32
                    and k % 4 == 0
                    and m % 4 == 0
                    and ceildiv(n, 64) * ceildiv(m, 64) >= 342
                ):
                    _ct_enqueue[64, 64, 16, 8, 8, 3, 6](
                        c_addr, a_addr, b_addr, m, n, k, ctx
                    )
                    return
            if m > SMALLM_MR and m <= 32:
                comptime if transpose_b:
                    # CT path: batch==1 only (no batched A^T scratch), and
                    # only when the weight-row grid alone fills the GPU.
                    # This C-transpose tile is tuned for NVIDIA/AMD. Its
                    # 3-stage pipeline needs 36 KiB of threadgroup memory,
                    # above Apple GPUs' 32 KiB limit. Apple either took the
                    # simdgroup-matrix path above or uses the regular
                    # transposed-B fallback below for smaller outputs.
                    comptime if not has_apple_gpu_accelerator():
                        if (
                            batch == 1
                            and k % 4 == 0
                            and m % 4 == 0
                            and ceildiv(n, 64) >= 128
                        ):
                            _ct_enqueue[64, 32, 32, 4, 4, 3](
                                c_addr, a_addr, b_addr, m, n, k, ctx
                            )
                            return
                    if k % 4 == 0:
                        _tune_enqueue[32, 64, 16, 4, 4, True, False](
                            c_addr,
                            a_addr,
                            b_addr,
                            batch,
                            m,
                            n,
                            k,
                            a_bstride,
                            96,
                            ctx,
                        )
                        return
                else:
                    if k % 4 == 0 and n % 4 == 0:
                        _tune_enqueue[32, 64, 16, 4, 4, False, True](
                            c_addr,
                            a_addr,
                            b_addr,
                            batch,
                            m,
                            n,
                            k,
                            a_bstride,
                            96,
                            ctx,
                        )
                        return

        # Choose the kernel/tile config, then a split-K factor that brings
        # the block count up to a few per SM (float32 only).
        var gx: Int
        var gy: Int
        var kcap: Int

        var use_t128 = False
        if m > SMALLM_MR and m >= 96 and n >= 96:
            # Only use the fat tile when its grid alone fills a wave;
            # otherwise the thinner tiles' extra blocks hide more latency.
            use_t128 = ceildiv(n, 128) * ceildiv(m, 128) * batch >= 114
        var use_n32 = (
            m > SMALLM_MR
            and not use_t128
            and n <= 96
            and dtype == DType.float32
        )

        if m <= SMALLM_MR:
            gx = ceildiv(n, SMALLM_THREADS)
            gy = 1
            kcap = ceildiv(k, 64)
        elif use_t128:
            gx = ceildiv(n, 128)
            gy = ceildiv(m, 128)
            kcap = ceildiv(k, 64)
        elif use_n32:
            gx = ceildiv(n, 32)
            gy = ceildiv(m, 64)
            kcap = ceildiv(k, 192)
        else:
            gx = ceildiv(n, 64)
            gy = ceildiv(m, 64)
            kcap = ceildiv(k, 192)

        var ksplits = 1
        if dtype == DType.float32:
            var base = gx * gy * batch
            if base < TARGET_BLOCKS // 2:
                ksplits = min(min(ceildiv(TARGET_BLOCKS, base), kcap), 32)

        # Split-K partials go to a stream-ordered workspace and a final
        # reduce kernel sums them into C (plain stores, deterministic).
        var ws = Optional[DeviceBuffer[DType.float32]](None)
        var c_target = c_addr
        if ksplits > 1:
            ws = ctx.enqueue_create_buffer[DType.float32](
                batch * ksplits * m * n
            )
            c_target = Int(ws.value().unsafe_ptr())
        var c = _make_ptr[dtype](c_target).as_unsafe_any_origin()

        if m <= SMALLM_MR:
            _enqueue_cached[_gemm_smallm_kernel[dtype, transpose_b]](
                ctx,
                String(t"gemm_smallm_{dtype}_tb{transpose_b}"),
                gx,
                gy,
                batch * ksplits,
                SMALLM_THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        elif use_t128:
            comptime if dtype == DType.float32:
                _enqueue_pipe[128, 128, 16, 8, 8, transpose_b](
                    ctx,
                    gx,
                    gy,
                    batch * ksplits,
                    c_target,
                    a_addr,
                    b_addr,
                    m,
                    n,
                    k,
                    a_bstride,
                    ksplits,
                )
            else:
                _enqueue_cached[
                    _gemm_tiled_kernel[dtype, 128, 128, 16, 8, 8, transpose_b]
                ](
                    ctx,
                    String(t"gemm_t128_{dtype}_tb{transpose_b}"),
                    gx,
                    gy,
                    batch * ksplits,
                    256,
                    c,
                    a,
                    b,
                    m,
                    n,
                    k,
                    a_bstride,
                    ksplits,
                )
        elif use_n32:
            comptime if dtype == DType.float32:
                _enqueue_pipe[64, 32, 16, 4, 4, transpose_b](
                    ctx,
                    gx,
                    gy,
                    batch * ksplits,
                    c_target,
                    a_addr,
                    b_addr,
                    m,
                    n,
                    k,
                    a_bstride,
                    ksplits,
                )
        else:
            comptime if dtype == DType.float32:
                _enqueue_pipe[64, 64, 16, 4, 4, transpose_b](
                    ctx,
                    gx,
                    gy,
                    batch * ksplits,
                    c_target,
                    a_addr,
                    b_addr,
                    m,
                    n,
                    k,
                    a_bstride,
                    ksplits,
                )
            else:
                _enqueue_cached[
                    _gemm_tiled_kernel[dtype, 64, 64, 16, 4, 4, transpose_b]
                ](
                    ctx,
                    String(t"gemm_t64_{dtype}_tb{transpose_b}"),
                    gx,
                    gy,
                    batch * ksplits,
                    256,
                    c,
                    a,
                    b,
                    m,
                    n,
                    k,
                    a_bstride,
                    ksplits,
                )
        if ksplits > 1:
            var total = batch * m * n
            var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
            var ws_ptr = (
                _make_ptr[DType.float32](c_target)
                .as_unsafe_any_origin()
                .as_immutable()
            )
            _enqueue_cached[_ksplit_reduce_kernel](
                ctx,
                String("ksplit_reduce"),
                ceildiv(total, 1024),
                1,
                1,
                256,
                c_out,
                ws_ptr,
                m * n,
                ksplits,
                total,
            )
        # Keep the workspace alive until its free is enqueued after the
        # reduce (stream-ordered).
        _ = ws^
    else:
        raise Error("no GPU accelerator available at compile time")


@always_inline
def _gemm_transb_dispatch[
    dtype: DType
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    transpose_b: Int,
    ctx: DeviceContext,
) raises:
    if transpose_b != 0:
        _gemm_enqueue[dtype, True](
            c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx
        )
    else:
        _gemm_enqueue[dtype, False](
            c_addr, a_addr, b_addr, batch, m, n, k, a_bstride, ctx
        )


# ---------------------------------------------------------------------------
# Single-token decode GEMV: modular's linalg.gemv.gemv_gpu for the m == 1,
# batch == 1 slice (both transpose_b orientations). Benchmarked equal-or-
# faster than our smallm kernel on every decode shape (f32/f16/bf16, aligned
# and unaligned k, tb=0 and tb=1), so it replaces smallm for m == 1. gemv's
# fast GEMV_SPLIT_K path fires for m==1 && transpose_b && dtype in
# {bf16, f16, fp8} && k % simd_width == 0; other cases take its GEMV_KERNEL /
# GEVM_KERNEL, which are still a single well-coalesced launch (vs our
# split-K's buffer alloc + reduce), hence the consistent win.
# ---------------------------------------------------------------------------


@always_inline
def _gemv_enqueue[
    dtype: DType, transpose_b: Bool
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        var c_ptr = _make_ptr[dtype](c_addr).as_unsafe_any_origin()
        var a_ptr = (
            _make_ptr[dtype](a_addr).as_unsafe_any_origin().as_immutable()
        )
        var b_ptr = (
            _make_ptr[dtype](b_addr).as_unsafe_any_origin().as_immutable()
        )
        var c = TileTensor(c_ptr, row_major(m, n))
        var a = TileTensor(a_ptr, row_major(m, k))
        comptime if transpose_b:
            var b = TileTensor(b_ptr, row_major(n, k))
            gemv_gpu[transpose_b=True, pdl_level=PDLLevel.OFF](c, a, b, ctx)
        else:
            var b = TileTensor(b_ptr, row_major(k, n))
            gemv_gpu[transpose_b=False, pdl_level=PDLLevel.OFF](c, a, b, ctx)
    else:
        raise Error("no GPU accelerator available at compile time")


@always_inline
def _gemv_dtype_dispatch(
    dtype: DType,
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Int,
    ctx: DeviceContext,
) raises:
    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                # The pinned MAX AMD GEVM launch grid divides N by 16 even
                # though a gfx942 wavefront emits 64 columns.  That launches 4x
                # too many blocks and writes beyond the output.  Keep every
                # dtype off that unsafe path on MI300X; the existing GEMM path
                # is correct for m == 1 and also avoids bf16's unsupported fdot2.
                comptime if _accelerator_arch() == "amdgpu:gfx942":
                    _gemm_transb_dispatch[dt](
                        c_addr,
                        a_addr,
                        b_addr,
                        1,
                        m,
                        n,
                        k,
                        m * k,
                        transpose_b,
                        ctx,
                    )
                else:
                    if transpose_b != 0:
                        _gemv_enqueue[dt, True](
                            c_addr, a_addr, b_addr, m, n, k, ctx
                        )
                    else:
                        _gemv_enqueue[dt, False](
                            c_addr, a_addr, b_addr, m, n, k, ctx
                        )
                handled = True
    if not handled:
        raise Error("unsupported dtype for gemv: " + String(dtype))


# ---------------------------------------------------------------------------
# Tuning entry point (float32 only): run one GEMM with an explicit tile
# config + split-K chunk floor, selected at runtime. Used by offline sweeps
# to pick the dispatch configs; not called by the backend.
# ---------------------------------------------------------------------------


@always_inline
# ---------------------------------------------------------------------------
# Cached raw-pointer variant of the Apple 8x8 simdgroup-matrix GEMM
# (linalg.matmul.gpu.apple), for float32 row-major C = A @ B [+ bias],
# including B stored transposed as (N, K).
# Differences from the stock kernel: thin function with plain pointer/int
# arguments so it launches through `_enqueue_cached` (the stock TileTensor +
# closure-epilogue form only fits the uncached per-call compile path), an
# optional bias row folded into the store, and split-K over `grid.z`
# (each z-slice writes its k-chunk partial to a workspace; a cached reduce
# sums them and applies the bias). Split-K is what keeps skinny-M
# projections (a 32-row A against a wide weight) from running a dozen
# threadgroups on a 10-40 core GPU.
# ---------------------------------------------------------------------------


def _apple8_gemm_kernel[
    HAS_BIAS: Bool, SPLIT: Bool, TRANSPOSE_B: Bool
](
    c_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    ksplits: Int,
):
    comptime BLOCK_M = 32
    comptime BLOCK_N = 64
    comptime SG_M = BLOCK_M // 2
    comptime SG_N = BLOCK_N // 2
    comptime NT_M = SG_M // MMA8_DIM
    comptime NT_N = SG_N // MMA8_DIM

    var lane = Int(lane_id())
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(warp_id())
    var row_base = Int(block_idx.y) * BLOCK_M + (sg // 2) * SG_M
    var col_base = Int(block_idx.x) * BLOCK_N + (sg % 2) * SG_N
    var interior = (row_base + SG_M <= m) and (col_base + SG_N <= n)

    # This z-slice's k-chunk, in whole 8-wide slabs (dispatch gates k % 8).
    var slabs = k // MMA8_DIM
    var chunk = (slabs + ksplits - 1) // ksplits
    var ks0 = Int(block_idx.z) * chunk
    var ks1 = min(slabs, ks0 + chunk)

    var accum = InlineArray[SIMD[DType.float32, FRAG8], NT_M * NT_N](
        fill=SIMD[DType.float32, FRAG8](0)
    )

    for ks in range(ks0, ks1):
        var kk = ks * MMA8_DIM
        var afrag = InlineArray[SIMD[DType.float32, FRAG8], NT_M](
            uninitialized=True
        )
        comptime for mi in range(NT_M):
            var grow = row_base + mi * MMA8_DIM + frow
            if interior or grow < m:
                afrag[mi] = (a_ptr + grow * k + kk + fcol).load[width=FRAG8]()
            else:
                afrag[mi] = SIMD[DType.float32, FRAG8](0)
        var bfrag = InlineArray[SIMD[DType.float32, FRAG8], NT_N](
            uninitialized=True
        )
        comptime for ni in range(NT_N):
            comptime if TRANSPOSE_B:
                var bf = SIMD[DType.float32, FRAG8](0)
                comptime for s in range(FRAG8):
                    var gj = col_base + ni * MMA8_DIM + fcol + s
                    if interior or gj < n:
                        bf[s] = b_ptr[gj * k + kk + frow]
                bfrag[ni] = bf
            else:
                var krow = kk + frow
                var gj = col_base + ni * MMA8_DIM + fcol
                if interior or gj + 1 < n:
                    bfrag[ni] = (b_ptr + krow * n + gj).load[width=FRAG8]()
                else:
                    var bf = SIMD[DType.float32, FRAG8](0)
                    if gj < n:
                        bf[0] = b_ptr[krow * n + gj]
                    bfrag[ni] = bf
        comptime for mi in range(NT_M):
            comptime for ni in range(NT_N):
                accum[mi * NT_N + ni] = _mma8x8(
                    afrag[mi], bfrag[ni], accum[mi * NT_N + ni]
                )

    var out_base = Int(block_idx.z) * m * n
    comptime for mi in range(NT_M):
        comptime for ni in range(NT_N):
            var frag = accum[mi * NT_N + ni]
            comptime for s in range(FRAG8):
                var grow = row_base + mi * MMA8_DIM + frow
                var gcol = col_base + ni * MMA8_DIM + fcol + s
                if grow < m and gcol < n:
                    var v = frag[s]
                    comptime if SPLIT:
                        c_ptr[out_base + grow * n + gcol] = v
                    else:
                        comptime if HAS_BIAS:
                            v += bias_ptr[gcol]
                        c_ptr[grow * n + gcol] = v


def _apple8_reduce_kernel[
    HAS_BIAS: Bool
](
    c_ptr: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    ws_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    bias_ptr: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    mn: Int,
    n: Int,
    ksplits: Int,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < mn:
        var acc = Float32(0)
        for z in range(ksplits):
            acc += ws_ptr[z * mn + i]
        comptime if HAS_BIAS:
            acc += bias_ptr[i % n]
        c_ptr[i] = acc
        i += gstride


@always_inline
def _apple8_enqueue[
    HAS_BIAS: Bool, TRANSPOSE_B: Bool = False
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    bias_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    var gx = ceildiv(n, 64)
    var gy = ceildiv(m, 32)
    var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
    var a = (
        _make_ptr[DType.float32](a_addr).as_unsafe_any_origin().as_immutable()
    )
    var b = (
        _make_ptr[DType.float32](b_addr).as_unsafe_any_origin().as_immutable()
    )
    var bias = (
        _make_ptr[DType.float32](bias_addr if bias_addr != 0 else a_addr)
        .as_unsafe_any_origin()
        .as_immutable()
    )
    var slabs = k // MMA8_DIM
    var ksplits = 1
    # Split K only when the N-tile grid is far below GPU saturation
    # (~2 threadgroups per core); each shard costs an m*n partials
    # round-trip, so a grid that already fills the cores never splits.
    if gx * gy < 24:
        ksplits = min(min(ceildiv(TARGET_BLOCKS, gx * gy), slabs), 8)
    if ksplits == 1:
        _enqueue_cached[_apple8_gemm_kernel[HAS_BIAS, False, TRANSPOSE_B]](
            ctx,
            String(t"apple8_gemm_b{HAS_BIAS}_tb{TRANSPOSE_B}"),
            gx,
            gy,
            1,
            128,
            c,
            a,
            b,
            bias,
            m,
            n,
            k,
            1,
        )
        return
    var ws = ctx.enqueue_create_buffer[DType.float32](ksplits * m * n)
    var ws_mut = UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
        unsafe_from_address=Int(ws.unsafe_ptr())
    ).as_unsafe_any_origin()
    _enqueue_cached[_apple8_gemm_kernel[False, True, TRANSPOSE_B]](
        ctx,
        String(t"apple8_gemm_split_tb{TRANSPOSE_B}"),
        gx,
        gy,
        ksplits,
        128,
        ws_mut,
        a,
        b,
        bias,
        m,
        n,
        k,
        ksplits,
    )
    var mn = m * n
    _enqueue_cached[_apple8_reduce_kernel[HAS_BIAS]](
        ctx,
        String(t"apple8_reduce_b{HAS_BIAS}"),
        max(1, min((mn + 255) // 256, 512)),
        1,
        1,
        256,
        c,
        ws_mut.as_immutable(),
        bias,
        mn,
        n,
        ksplits,
    )
    _ = ws^  # dropped now: the stream-ordered free lands after the reduce


# ---------------------------------------------------------------------------
# Fat-tile Apple 8x8 simdgroup-matrix GEMM: the m > 32 float32 workhorse on
# Metal (the NVIDIA SIMT pipe kernels above run 3-15x behind torch MPS
# there). 64x64 block, 4 simdgroups each owning a 32x32 subtile (4x4 grid of
# 8x8 fragments). Operands stream DRAM -> registers directly — no
# threadgroup staging, which measurably degrades matmul on Apple GPUs —
# with a pointer-increment unguarded loop for fully-interior subtiles and a
# guarded slab for ragged M/N edges and the K tail (any m/n/k works).
# Batched via block_idx.z; split-K over grid.z like `_gemm_tiled_kernel`
# (partials to a workspace, summed by `_ksplit_reduce_kernel`).
#
# TRANSPOSE_A reads A stored (K, M) row-major — C = A^T @ B without ever
# materializing the transpose (the layout `mm` with a transposed-view LHS,
# i.e. linear-backward's dW = dY^T @ X, has on hand). The logical transpose
# lives in the fragment loads: per-lane scalar pairs a whole stored row
# apart, the same pattern TRANSPOSE_B uses for B.
# ---------------------------------------------------------------------------


def _apple8_fat_kernel[
    SPLIT: Bool,
    TRANSPOSE_B: Bool,
    BLOCK_M: Int = 64,
    BLOCK_N: Int = 64,
    SG_ROWS: Int = 2,
    SG_COLS: Int = 2,
    CAUSAL: Int = 0,
    TRANSPOSE_A: Bool = False,
](
    c_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
):
    comptime SG_M = BLOCK_M // SG_ROWS
    comptime SG_N = BLOCK_N // SG_COLS
    comptime NT_M = SG_M // MMA8_DIM
    comptime NT_N = SG_N // MMA8_DIM

    var bz = Int(block_idx.z) // ksplits
    var ks = Int(block_idx.z) % ksplits
    # Round the k-chunk to whole 8-slabs so only the last split sees a tail.
    var kchunk = ceildiv(ceildiv(k, ksplits), MMA8_DIM) * MMA8_DIM
    var k_start = min(k, ks * kchunk)
    var k_end = min(k, k_start + kchunk)

    # With ksplits > 1, C is a [batch * ksplits, m, n] workspace and each
    # split writes its own slice. a_bstride is 0 when A is batch-shared.
    var c_ptr = c_base + Int(block_idx.z) * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var lane = Int(lane_id())
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(warp_id())
    var row_base = Int(block_idx.y) * BLOCK_M + (sg // SG_COLS) * SG_M
    var col_base = Int(block_idx.x) * BLOCK_N + (sg % SG_COLS) * SG_N
    var interior = (row_base + SG_M <= m) and (col_base + SG_N <= n)

    # SDPA causal specializations (row i of a batch slice only interacts with
    # columns/positions j <= i; see `_bmm_causal_go`):
    #   CAUSAL == 1 (scores = Q @ K^T): simdgroup tiles strictly above the
    #   diagonal are fully masked downstream — skip them entirely, leaving
    #   that part of C unwritten (the Apple row-softmax never reads past the
    #   causal boundary).
    #   CAUSAL == 2 (out = P @ V): P's columns past the boundary are exactly
    #   zero, so the K loop can stop at the tile's last row + 1. SG_M is a
    #   multiple of MMA8_DIM, so the cut point stays 8-slab-aligned and never
    #   creates a new K tail.
    comptime if CAUSAL == 1:
        if col_base >= row_base + SG_M:
            return
    comptime if CAUSAL == 2:
        k_end = max(k_start, min(k_end, row_base + SG_M))

    var accum = InlineArray[SIMD[DType.float32, FRAG8], NT_M * NT_N](
        fill=SIMD[DType.float32, FRAG8](0)
    )

    # One 8-slab with every bound checked: ragged M/N subtiles and the K
    # tail (kk + 8 > k_end). Interior full slabs never come through here.
    @always_inline
    @parameter
    def _slab_guarded(
        kk: Int, mut acc: InlineArray[SIMD[DType.float32, FRAG8], NT_M * NT_N]
    ):
        var afrag = InlineArray[SIMD[DType.float32, FRAG8], NT_M](
            uninitialized=True
        )
        comptime for mi in range(NT_M):
            var grow = row_base + mi * MMA8_DIM + frow
            var af = SIMD[DType.float32, FRAG8](0)
            if grow < m:
                comptime for s in range(FRAG8):
                    if kk + fcol + s < k_end:
                        comptime if TRANSPOSE_A:
                            # A (K, M): logical row = stored column.
                            af[s] = a_ptr[(kk + fcol + s) * m + grow]
                        else:
                            af[s] = a_ptr[grow * k + kk + fcol + s]
            afrag[mi] = af
        var bfrag = InlineArray[SIMD[DType.float32, FRAG8], NT_N](
            uninitialized=True
        )
        comptime for ni in range(NT_N):
            var bf = SIMD[DType.float32, FRAG8](0)
            comptime if TRANSPOSE_B:
                if kk + frow < k_end:
                    comptime for s in range(FRAG8):
                        var gj = col_base + ni * MMA8_DIM + fcol + s
                        if gj < n:
                            bf[s] = b_ptr[gj * k + kk + frow]
            else:
                if kk + frow < k_end:
                    comptime for s in range(FRAG8):
                        var gj = col_base + ni * MMA8_DIM + fcol + s
                        if gj < n:
                            bf[s] = b_ptr[(kk + frow) * n + gj]
            bfrag[ni] = bf
        comptime for mi in range(NT_M):
            comptime for ni in range(NT_N):
                acc[mi * NT_N + ni] = _mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * NT_N + ni]
                )

    # Unguarded fragment loads for one 8-slab: vector A loads, vector B loads
    # for (K, N) storage, per-n-row scalar pairs for transposed (N, K).
    @always_inline
    @parameter
    def _load_a_fast(
        ap0: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    ) -> InlineArray[SIMD[DType.float32, FRAG8], NT_M]:
        var afrag = InlineArray[SIMD[DType.float32, FRAG8], NT_M](
            uninitialized=True
        )
        comptime for mi in range(NT_M):
            comptime if TRANSPOSE_A:
                # A (K, M): the fragment's two slots differ by one stored
                # row (stride m) — per-lane scalar pairs, like transposed B.
                var p = ap0 + mi * MMA8_DIM
                var af = SIMD[DType.float32, FRAG8](0)
                comptime for s in range(FRAG8):
                    af[s] = p[s * m]
                afrag[mi] = af
            else:
                afrag[mi] = (ap0 + mi * MMA8_DIM * k).load[width=FRAG8]()
        return afrag

    @always_inline
    @parameter
    def _load_b_fast(
        bp0: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    ) -> InlineArray[SIMD[DType.float32, FRAG8], NT_N]:
        var bfrag = InlineArray[SIMD[DType.float32, FRAG8], NT_N](
            uninitialized=True
        )
        comptime for ni in range(NT_N):
            comptime if TRANSPOSE_B:
                # B is (N, K): the fragment's two slots differ in n-row.
                var p = bp0 + ni * MMA8_DIM * k
                var bf = SIMD[DType.float32, FRAG8](0)
                bf[0] = p[0]
                bf[1] = p[k]
                bfrag[ni] = bf
            else:
                bfrag[ni] = (bp0 + ni * MMA8_DIM).load[width=FRAG8]()
        return bfrag

    @always_inline
    @parameter
    def _mma_block(
        afrag: InlineArray[SIMD[DType.float32, FRAG8], NT_M],
        bfrag: InlineArray[SIMD[DType.float32, FRAG8], NT_N],
        mut acc: InlineArray[SIMD[DType.float32, FRAG8], NT_M * NT_N],
    ):
        comptime for mi in range(NT_M):
            comptime for ni in range(NT_N):
                acc[mi * NT_N + ni] = _mma8x8(
                    afrag[mi], bfrag[ni], acc[mi * NT_N + ni]
                )

    var full_end = k_start + ((k_end - k_start) // MMA8_DIM) * MMA8_DIM
    if interior:
        # Software-pipelined pointer-increment loop: the next slab's
        # fragments are in flight while the current slab's mmas issue, so a
        # single simdgroup hides most of the device-load latency itself.
        var ap: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]
        var astep: Int
        comptime if TRANSPOSE_A:
            ap = a_ptr + (k_start + fcol) * m + row_base + frow
            astep = MMA8_DIM * m
        else:
            ap = a_ptr + (row_base + frow) * k + fcol + k_start
            astep = MMA8_DIM
        var bp: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin]
        var bstep: Int
        comptime if TRANSPOSE_B:
            bp = b_ptr + (col_base + fcol) * k + frow + k_start
            bstep = MMA8_DIM
        else:
            bp = b_ptr + (k_start + frow) * n + col_base + fcol
            bstep = MMA8_DIM * n
        var nslabs = (k_end - k_start) // MMA8_DIM
        if nslabs > 0:
            var cura = _load_a_fast(ap)
            var curb = _load_b_fast(bp)
            for _ in range(nslabs - 1):
                ap += astep
                bp += bstep
                var nxta = _load_a_fast(ap)
                var nxtb = _load_b_fast(bp)
                _mma_block(cura, curb, accum)
                cura = nxta
                curb = nxtb
            _mma_block(cura, curb, accum)
        if full_end < k_end:
            _slab_guarded(full_end, accum)
    else:
        var kk = k_start
        while kk < k_end:
            _slab_guarded(kk, accum)
            kk += MMA8_DIM

    comptime for mi in range(NT_M):
        var grow = row_base + mi * MMA8_DIM + frow
        if grow < m:
            comptime for ni in range(NT_N):
                var gcol = col_base + ni * MMA8_DIM + fcol
                var frag = accum[mi * NT_N + ni]
                if gcol + FRAG8 <= n:
                    c_ptr.store(grow * n + gcol, frag)
                elif gcol < n:
                    c_ptr[grow * n + gcol] = frag[0]


@always_inline
def _apple8_fat_enqueue[
    TRANSPOSE_B: Bool,
    BLOCK_M: Int = 64,
    BLOCK_N: Int = 64,
    SG_ROWS: Int = 2,
    SG_COLS: Int = 2,
    STAGED: Bool = False,
    CAUSAL: Int = 0,
    TRANSPOSE_A: Bool = False,
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ctx: DeviceContext,
) raises:
    comptime assert (not STAGED) or (
        BLOCK_M == 64 and BLOCK_N == 64 and SG_ROWS == 2 and SG_COLS == 2
    ), "the staged kernel is fixed at the 64x64 / 4-simdgroup geometry"
    comptime assert (
        not STAGED
    ) or CAUSAL == 0, (
        "causal specializations exist only for the direct-load kernel"
    )
    comptime assert not (
        TRANSPOSE_A and TRANSPOSE_B
    ), "transposed-A GEMM reads B as (K, N) row-major"
    comptime assert (
        not TRANSPOSE_A
    ) or CAUSAL == 0, "causal specializations exist only for the dense-A kernel"
    comptime THREADS = SG_ROWS * SG_COLS * 32
    var gx = ceildiv(n, BLOCK_N)
    var gy = ceildiv(m, BLOCK_M)
    var blocks = gx * gy * batch
    var slabs = ceildiv(k, MMA8_DIM)
    var ksplits = 1
    # Split K only when the MN grid alone cannot keep ~2 threadgroups per
    # core busy; each shard costs an m*n partials round-trip.
    if blocks < TARGET_BLOCKS // 2:
        ksplits = min(min(ceildiv(TARGET_BLOCKS, blocks), slabs), 8)
    var a = (
        _make_ptr[DType.float32](a_addr).as_unsafe_any_origin().as_immutable()
    )
    var b = (
        _make_ptr[DType.float32](b_addr).as_unsafe_any_origin().as_immutable()
    )
    if ksplits == 1:
        var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
        comptime if STAGED:
            _enqueue_cached[
                _apple8_smem_kernel[False, TRANSPOSE_B, TRANSPOSE_A]
            ](
                ctx,
                String(t"apple8_smem_tb{TRANSPOSE_B}_ta{TRANSPOSE_A}"),
                gx,
                gy,
                batch,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                1,
            )
        else:
            _enqueue_cached[
                _apple8_fat_kernel[
                    False,
                    TRANSPOSE_B,
                    BLOCK_M,
                    BLOCK_N,
                    SG_ROWS,
                    SG_COLS,
                    CAUSAL,
                    TRANSPOSE_A,
                ]
            ](
                ctx,
                String(
                    t"apple8_fat_tb{TRANSPOSE_B}_{BLOCK_M}x{BLOCK_N}"
                    t"_c{CAUSAL}_ta{TRANSPOSE_A}"
                ),
                gx,
                gy,
                batch,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                1,
            )
        return
    var ws = ctx.enqueue_create_buffer[DType.float32](batch * ksplits * m * n)
    var ws_mut = UnsafePointer[Scalar[DType.float32], MutUntrackedOrigin](
        unsafe_from_address=Int(ws.unsafe_ptr())
    ).as_unsafe_any_origin()
    comptime if STAGED:
        _enqueue_cached[_apple8_smem_kernel[True, TRANSPOSE_B, TRANSPOSE_A]](
            ctx,
            String(t"apple8_smem_split_tb{TRANSPOSE_B}_ta{TRANSPOSE_A}"),
            gx,
            gy,
            batch * ksplits,
            THREADS,
            ws_mut,
            a,
            b,
            m,
            n,
            k,
            a_bstride,
            ksplits,
        )
    else:
        _enqueue_cached[
            _apple8_fat_kernel[
                True,
                TRANSPOSE_B,
                BLOCK_M,
                BLOCK_N,
                SG_ROWS,
                SG_COLS,
                CAUSAL,
                TRANSPOSE_A,
            ]
        ](
            ctx,
            String(
                t"apple8_fat_split_tb{TRANSPOSE_B}_{BLOCK_M}x{BLOCK_N}"
                t"_c{CAUSAL}_ta{TRANSPOSE_A}"
            ),
            gx,
            gy,
            batch * ksplits,
            THREADS,
            ws_mut,
            a,
            b,
            m,
            n,
            k,
            a_bstride,
            ksplits,
        )
    var total = batch * m * n
    var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
    _enqueue_cached[_ksplit_reduce_kernel](
        ctx,
        String("ksplit_reduce"),
        ceildiv(total, 1024),
        1,
        1,
        256,
        c_out,
        ws_mut.as_immutable(),
        m * n,
        ksplits,
        total,
    )
    _ = ws^  # dropped now: the stream-ordered free lands after the reduce


# ---------------------------------------------------------------------------
# Threadgroup-staged variant of the fat Apple simdgroup-matrix GEMM: 64x64
# block, BK=16 K-stages double-buffered through threadgroup memory. Each
# element is loaded from device memory once per block (the direct-load
# kernel above loads A and B twice each — once per simdgroup row/column)
# with wide float4 accesses, and the store->barrier->compute pipeline keeps
# the next stage's device loads in flight during the current stage's mmas
# with a single barrier per stage. Edge guards live only in the cooperative
# fill (OOB elements are zeroed), so the mma loop is always unguarded.
#
# TRANSPOSE_A reads A stored (K, M) row-major (C = A^T @ B, no materialized
# transpose): the fill loads wide k-row segments of A and scatters them
# transposed into the same threadgroup layout the mma sweep always reads —
# so unlike the direct-load kernel, every A element still moves from device
# memory in a single float4 load.
# ---------------------------------------------------------------------------


def _apple8_smem_kernel[
    SPLIT: Bool, TRANSPOSE_B: Bool, TRANSPOSE_A: Bool = False
](
    c_base: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    a_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    b_base: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    ksplits: Int,
):
    comptime BLOCK_M = 64
    comptime BLOCK_N = 64
    comptime BK = 16
    comptime THREADS = 128
    # Pad the A row stride (and B for symmetry) to stagger threadgroup
    # banks across the 8 fragment rows.
    comptime LDA = BK + 2
    comptime LDB = BLOCK_N + 2
    comptime SG_M = BLOCK_M // 2
    comptime SG_N = BLOCK_N // 2
    comptime NT_M = SG_M // MMA8_DIM  # 4
    comptime NT_N = SG_N // MMA8_DIM  # 4
    comptime AV = (BLOCK_M * BK) // (THREADS * 4)  # float4 fills per thread
    comptime BV = (BK * BLOCK_N) // (THREADS * 4)

    var bz = Int(block_idx.z) // ksplits
    var ks = Int(block_idx.z) % ksplits
    var kchunk = ceildiv(ceildiv(k, ksplits), BK) * BK
    var k_start = min(k, ks * kchunk)
    var k_end = min(k, k_start + kchunk)

    var c_ptr = c_base + Int(block_idx.z) * m * n
    var a_ptr = a_base + bz * a_bstride
    var b_ptr = b_base + bz * k * n

    var bm = Int(block_idx.y) * BLOCK_M
    var bn = Int(block_idx.x) * BLOCK_N
    var tid = Int(thread_idx.x)

    var a_smem = stack_allocation[
        2 * BLOCK_M * LDA, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var b_smem = stack_allocation[
        2 * BK * LDB, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var lane = Int(lane_id())
    var fl = _frag8_layout(lane)
    var frow = fl[0]
    var fcol = fl[1]
    var sg = Int(warp_id())
    var sgm = (sg // 2) * SG_M
    var sgn = (sg % 2) * SG_N

    var accum = InlineArray[SIMD[DType.float32, FRAG8], NT_M * NT_N](
        fill=SIMD[DType.float32, FRAG8](0)
    )

    # Device -> register fill for one BK stage at k-offset kt (OOB -> 0).
    @always_inline
    @parameter
    def _fill_regs(
        kt: Int,
        mut a_regs: InlineArray[SIMD[DType.float32, 4], AV],
        mut b_regs: InlineArray[SIMD[DType.float32, 4], BV],
    ):
        comptime for q in range(AV):
            var fid = q * THREADS + tid
            comptime if TRANSPOSE_A:
                # A (K, M): a straight m-run of this k-row; scattered
                # transposed on store.
                var kk = fid >> 4
                var mq = (fid & 15) * 4
                var row = kt + kk
                var col = bm + mq
                if row < k_end and col + 4 <= m:
                    a_regs[q] = (a_ptr + row * m + col).load[width=4]()
                else:
                    var v = SIMD[DType.float32, 4](0)
                    if row < k_end:
                        comptime for j in range(4):
                            if col + j < m:
                                v[j] = a_ptr[row * m + col + j]
                    a_regs[q] = v
            else:
                var mm = fid >> 2
                var kq = (fid & 3) * 4
                var row = bm + mm
                var col = kt + kq
                if row < m and col + 4 <= k_end:
                    a_regs[q] = (a_ptr + row * k + col).load[width=4]()
                else:
                    var v = SIMD[DType.float32, 4](0)
                    if row < m:
                        comptime for j in range(4):
                            if col + j < k_end:
                                v[j] = a_ptr[row * k + col + j]
                    a_regs[q] = v
        comptime for q in range(BV):
            var fid = q * THREADS + tid
            comptime if TRANSPOSE_B:
                # B (N, K): read a k-run of this n-row; scattered on store.
                var nn = fid >> 2
                var kq = (fid & 3) * 4
                var row = bn + nn
                var col = kt + kq
                if row < n and col + 4 <= k_end:
                    b_regs[q] = (b_ptr + row * k + col).load[width=4]()
                else:
                    var v = SIMD[DType.float32, 4](0)
                    if row < n:
                        comptime for j in range(4):
                            if col + j < k_end:
                                v[j] = b_ptr[row * k + col + j]
                    b_regs[q] = v
            else:
                # B (K, N): a straight row-segment copy.
                var kk = fid >> 4
                var nq = (fid & 15) * 4
                var row = kt + kk
                var col = bn + nq
                if row < k_end and col + 4 <= n:
                    b_regs[q] = (b_ptr + row * n + col).load[width=4]()
                else:
                    var v = SIMD[DType.float32, 4](0)
                    if row < k_end:
                        comptime for j in range(4):
                            if col + j < n:
                                v[j] = b_ptr[row * n + col + j]
                    b_regs[q] = v

    # Register -> threadgroup store for buffer `buf`.
    @always_inline
    @parameter
    def _store_smem(
        buf: Int,
        a_regs: InlineArray[SIMD[DType.float32, 4], AV],
        b_regs: InlineArray[SIMD[DType.float32, 4], BV],
    ):
        var a_dst = a_smem + buf * BLOCK_M * LDA
        var b_dst = b_smem + buf * BK * LDB
        comptime for q in range(AV):
            var fid = q * THREADS + tid
            comptime if TRANSPOSE_A:
                var kk = fid >> 4
                var mq = (fid & 15) * 4
                comptime for j in range(4):
                    a_dst[(mq + j) * LDA + kk] = a_regs[q][j]
            else:
                var mm = fid >> 2
                var kq = (fid & 3) * 4
                a_dst.store(mm * LDA + kq, a_regs[q])
        comptime for q in range(BV):
            var fid = q * THREADS + tid
            comptime if TRANSPOSE_B:
                var nn = fid >> 2
                var kq = (fid & 3) * 4
                comptime for j in range(4):
                    b_dst[(kq + j) * LDB + nn] = b_regs[q][j]
            else:
                var kk = fid >> 4
                var nq = (fid & 15) * 4
                b_dst.store(kk * LDB + nq, b_regs[q])

    # Two 8-slab mma sweeps over threadgroup buffer `buf`.
    @always_inline
    @parameter
    def _compute(
        buf: Int,
        mut acc: InlineArray[SIMD[DType.float32, FRAG8], NT_M * NT_N],
    ):
        var a_src = a_smem + buf * BLOCK_M * LDA + (sgm + frow) * LDA + fcol
        var b_src = b_smem + buf * BK * LDB + frow * LDB + sgn + fcol
        comptime for kc in range(BK // MMA8_DIM):
            var afrag = InlineArray[SIMD[DType.float32, FRAG8], NT_M](
                uninitialized=True
            )
            comptime for mi in range(NT_M):
                afrag[mi] = (a_src + mi * MMA8_DIM * LDA + kc * MMA8_DIM).load[
                    width=FRAG8
                ]()
            var bfrag = InlineArray[SIMD[DType.float32, FRAG8], NT_N](
                uninitialized=True
            )
            comptime for ni in range(NT_N):
                bfrag[ni] = (b_src + kc * MMA8_DIM * LDB + ni * MMA8_DIM).load[
                    width=FRAG8
                ]()
            comptime for mi in range(NT_M):
                comptime for ni in range(NT_N):
                    acc[mi * NT_N + ni] = _mma8x8(
                        afrag[mi], bfrag[ni], acc[mi * NT_N + ni]
                    )

    var stages = ceildiv(k_end - k_start, BK)
    var a_regs = InlineArray[SIMD[DType.float32, 4], AV](uninitialized=True)
    var b_regs = InlineArray[SIMD[DType.float32, 4], BV](uninitialized=True)
    if stages > 0:
        _fill_regs(k_start, a_regs, b_regs)
    for t in range(stages):
        # Writing buffer t%2 is safe without a leading barrier: its previous
        # readers all passed the barrier below in iteration t-1.
        _store_smem(t & 1, a_regs, b_regs)
        barrier()
        if t + 1 < stages:
            _fill_regs(k_start + (t + 1) * BK, a_regs, b_regs)
        _compute(t & 1, accum)

    comptime for mi in range(NT_M):
        var grow = bm + sgm + mi * MMA8_DIM + frow
        if grow < m:
            comptime for ni in range(NT_N):
                var gcol = bn + sgn + ni * MMA8_DIM + fcol
                var frag = accum[mi * NT_N + ni]
                if gcol + FRAG8 <= n:
                    c_ptr.store(grow * n + gcol, frag)
                elif gcol < n:
                    c_ptr[grow * n + gcol] = frag[0]


def _tune_enqueue[
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    transpose_b: Bool,
    pipe3: Bool,
    STAGES: Int = PIPE3_STAGES,
    MINB: Int = -1,
    BIAS: Bool = False,
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    kchunk_min: Int,
    ctx: DeviceContext,
    bias_addr: Int = 0,
) raises:
    comptime if has_accelerator():
        comptime THREADS = (BM // TM) * (BN // TN)
        var gx = ceildiv(n, BN)
        var gy = ceildiv(m, BM)
        var ksplits = 1
        var base = gx * gy * batch
        var kcap = ceildiv(k, kchunk_min)
        if base < TARGET_BLOCKS // 2:
            ksplits = min(min(ceildiv(TARGET_BLOCKS, base), kcap), 32)
        var ws = Optional[DeviceBuffer[DType.float32]](None)
        var c_target = c_addr
        if ksplits > 1:
            ws = ctx.enqueue_create_buffer[DType.float32](
                batch * ksplits * m * n
            )
            c_target = Int(ws.value().unsafe_ptr())
        var c = _make_ptr[DType.float32](c_target).as_unsafe_any_origin()
        var a = (
            _make_ptr[DType.float32](a_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        var b = (
            _make_ptr[DType.float32](b_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        comptime if pipe3:
            var bias = (
                _make_ptr[DType.float32](
                    bias_addr if bias_addr != 0 else a_addr
                )
                .as_unsafe_any_origin()
                .as_immutable()
            )
            _enqueue_cached[
                _gemm_pipe3_kernel[
                    BM, BN, BK, TM, TN, 4, 4, False, STAGES, False, MINB, BIAS
                ]
            ](
                ctx,
                String(
                    t"tune_p3_{BM}x{BN}x{BK}_{TM}{TN}_s{STAGES}_mb{MINB}_bs{BIAS}"
                ),
                gx,
                gy,
                batch * ksplits,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
                bias,
            )
        else:
            _enqueue_cached[
                _gemm_pipe_kernel[BM, BN, BK, TM, TN, 4, 4, transpose_b]
            ](
                ctx,
                String(t"tune_p_{BM}x{BN}x{BK}_{TM}{TN}_tb{transpose_b}"),
                gx,
                gy,
                batch * ksplits,
                THREADS,
                c,
                a,
                b,
                m,
                n,
                k,
                a_bstride,
                ksplits,
            )
        if ksplits > 1:
            var total = batch * m * n
            var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
            var ws_ptr = (
                _make_ptr[DType.float32](c_target)
                .as_unsafe_any_origin()
                .as_immutable()
            )
            _enqueue_cached[_ksplit_reduce_kernel](
                ctx,
                String("ksplit_reduce"),
                ceildiv(total, 1024),
                1,
                1,
                256,
                c_out,
                ws_ptr,
                m * n,
                ksplits,
                total,
            )
        _ = ws^
    else:
        raise Error("no GPU accelerator available at compile time")


# Skinny-M transposed-B GEMM as C^T = B @ A^T: materialize A^T (m x k ->
# k x m, one tiny kernel), then run the cp.async tb0 pipeline with the
# operand roles swapped — B (n, k) row-major streams as the "A" operand at
# full bandwidth — and the CT epilogue writes C (m, n) directly.
@always_inline
def _ct_enqueue[
    BM: Int,
    BN: Int,
    BK: Int,
    TM: Int,
    TN: Int,
    STAGES: Int = PIPE3_STAGES,
    MINB: Int = -1,
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        comptime THREADS = (BM // TM) * (BN // TN)
        var at_buf = ctx.enqueue_create_buffer[DType.float32](m * k)
        var at_addr = Int(at_buf.unsafe_ptr())
        var a_in = (
            _make_ptr[DType.float32](a_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        var at_out = _make_ptr[DType.float32](at_addr).as_unsafe_any_origin()
        _enqueue_cached[_transpose_small_kernel](
            ctx,
            String("transpose_small"),
            ceildiv(m * k, 256),
            1,
            1,
            256,
            at_out,
            a_in,
            m,
            k,
        )
        var c = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
        var wa = (
            _make_ptr[DType.float32](b_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        var at = (
            _make_ptr[DType.float32](at_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        # Swapped roles: kernel-m = n (weight rows), kernel-n = m.
        _enqueue_cached[
            _gemm_pipe3_kernel[
                BM, BN, BK, TM, TN, 4, 4, True, STAGES, False, MINB
            ]
        ](
            ctx,
            String(t"gemm_ct_{BM}x{BN}x{BK}_{TM}{TN}_s{STAGES}_mb{MINB}"),
            ceildiv(m, BN),
            ceildiv(n, BM),
            1,
            THREADS,
            c,
            wa,
            at,
            n,
            m,
            k,
            n * k,
            1,
            at,
        )
        _ = at_buf^
    else:
        raise Error("no GPU accelerator available at compile time")


# Medium-M FFMA-dense path: transpose the (small) activation A once so
# BOTH operand slabs stream via cp.async and both compute fragments are
# contiguous vector loads (CUTLASS-SIMT-style instruction density).
@always_inline
def _pipe3t_enqueue[
    BM: Int, BN: Int, BK: Int, TM: Int, TN: Int, STAGES: Int, MINB: Int = -1
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    kchunk_min: Int,
    ctx: DeviceContext,
) raises:
    comptime if has_accelerator():
        comptime THREADS = (BM // TM) * (BN // TN)
        var at_buf = ctx.enqueue_create_buffer[DType.float32](m * k)
        var at_addr = Int(at_buf.unsafe_ptr())
        var a_in = (
            _make_ptr[DType.float32](a_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        var at_out = _make_ptr[DType.float32](at_addr).as_unsafe_any_origin()
        _enqueue_cached[_transpose_small_kernel](
            ctx,
            String("transpose_small"),
            ceildiv(m * k, 256),
            1,
            1,
            256,
            at_out,
            a_in,
            m,
            k,
        )
        var gx = ceildiv(n, BN)
        var gy = ceildiv(m, BM)
        var ksplits = 1
        var base = gx * gy
        var kcap = ceildiv(k, kchunk_min)
        if base < TARGET_BLOCKS // 2:
            ksplits = min(min(ceildiv(TARGET_BLOCKS, base), kcap), 32)
        var ws = Optional[DeviceBuffer[DType.float32]](None)
        var c_target = c_addr
        if ksplits > 1:
            ws = ctx.enqueue_create_buffer[DType.float32](ksplits * m * n)
            c_target = Int(ws.value().unsafe_ptr())
        var c = _make_ptr[DType.float32](c_target).as_unsafe_any_origin()
        var at = (
            _make_ptr[DType.float32](at_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        var b = (
            _make_ptr[DType.float32](b_addr)
            .as_unsafe_any_origin()
            .as_immutable()
        )
        _enqueue_cached[
            _gemm_pipe3_kernel[
                BM, BN, BK, TM, TN, 4, 4, False, STAGES, True, MINB
            ]
        ](
            ctx,
            String(t"gemm_p3t_{BM}x{BN}x{BK}_{TM}{TN}_s{STAGES}_mb{MINB}"),
            gx,
            gy,
            ksplits,
            THREADS,
            c,
            at,
            b,
            m,
            n,
            k,
            m * k,
            ksplits,
            at,
        )
        if ksplits > 1:
            var total = m * n
            var c_out = _make_ptr[DType.float32](c_addr).as_unsafe_any_origin()
            var ws_ptr = (
                _make_ptr[DType.float32](c_target)
                .as_unsafe_any_origin()
                .as_immutable()
            )
            _enqueue_cached[_ksplit_reduce_kernel](
                ctx,
                String("ksplit_reduce"),
                ceildiv(total, 1024),
                1,
                1,
                256,
                c_out,
                ws_ptr,
                m * n,
                ksplits,
                total,
            )
        _ = ws^
        _ = at_buf^
    else:
        raise Error("no GPU accelerator available at compile time")


def _matmul_tune_dispatcher(
    c_buffer: PythonObject,
    a_buffer: PythonObject,
    b_buffer: PythonObject,
    # (m, n, k, transpose_b, cfg, kchunk_min); float32 only, k % 4 == 0
    # and (n % 4 == 0 when transpose_b == 0) required.
    params: PythonObject,
    device_context_ptr: PythonObject,
) raises:
    var c_addr = Int(py=c_buffer._data_ptr())
    var a_addr = Int(py=a_buffer._data_ptr())
    var b_addr = Int(py=b_buffer._data_ptr())
    var m = Int(py=params[0])
    var n = Int(py=params[1])
    var k = Int(py=params[2])
    var transpose_b = Int(py=params[3])
    var cfg = Int(py=params[4])
    var kmin = Int(py=params[5])
    var ctx = _get_ctx(device_context_ptr)

    if transpose_b == 0:
        if cfg == 1:
            _tune_enqueue[32, 64, 16, 4, 4, False, True](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 2:
            _tune_enqueue[64, 64, 16, 4, 4, False, True](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 3:
            _tune_enqueue[128, 128, 16, 8, 8, False, False](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 4:
            _tune_enqueue[128, 64, 8, 8, 8, False, True, 5](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 5:
            _tune_enqueue[128, 128, 8, 8, 8, False, True, 5](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 6:
            _tune_enqueue[128, 64, 8, 8, 8, False, True, 7](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 7:
            _pipe3t_enqueue[128, 64, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 8:
            _pipe3t_enqueue[128, 128, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 9:
            _pipe3t_enqueue[128, 64, 16, 8, 8, 3](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 10:
            _pipe3t_enqueue[64, 64, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 11:
            _tune_enqueue[64, 64, 16, 4, 4, False, True, 4, 4](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 12:
            _tune_enqueue[64, 64, 16, 4, 4, False, True, 4, 5](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 13:
            _pipe3t_enqueue[64, 64, 8, 8, 4, 5, 6](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 14:
            _pipe3t_enqueue[64, 64, 8, 8, 4, 6, 8](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 15:
            _pipe3t_enqueue[128, 64, 8, 8, 8, 5, 4](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        elif cfg == 16:
            _pipe3t_enqueue[64, 128, 8, 8, 8, 5, 6](
                c_addr, a_addr, b_addr, m, n, k, kmin, ctx
            )
        else:
            raise Error("unknown tb0 tune cfg")
    else:
        if cfg == 1:
            _tune_enqueue[32, 64, 16, 4, 4, True, False](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, kmin, ctx
            )
        elif cfg == 2:
            _ct_enqueue[64, 32, 32, 4, 4, 3](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 3:
            _ct_enqueue[64, 32, 16, 4, 4](c_addr, a_addr, b_addr, m, n, k, ctx)
        elif cfg == 4:
            _ct_enqueue[128, 64, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 5:
            _ct_enqueue[128, 128, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 6:
            _ct_enqueue[64, 64, 8, 8, 8, 5](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 7:
            _ct_enqueue[64, 64, 8, 8, 8, 5, 6](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 8:
            _ct_enqueue[64, 128, 8, 8, 8, 5, 4](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        elif cfg == 9:
            _ct_enqueue[64, 64, 16, 8, 8, 3, 6](
                c_addr, a_addr, b_addr, m, n, k, ctx
            )
        else:
            raise Error("unknown tb1 tune cfg")


@always_inline
def _gemm_dtype_dispatch(
    dtype: DType,
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    transpose_b: Int,
    c_off: Int,  # element offsets into c/a/b
    a_off: Int,
    b_off: Int,
    ctx: DeviceContext,
) raises:
    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                # c_off/a_off/b_off are element offsets; scale to bytes for the
                # raw addresses (4 for float32, 2 for the 16-bit dtypes).
                comptime elem_size = size_of[dt]()
                _gemm_transb_dispatch[dt](
                    c_addr + c_off * elem_size,
                    a_addr + a_off * elem_size,
                    b_addr + b_off * elem_size,
                    batch,
                    m,
                    n,
                    k,
                    a_bstride,
                    transpose_b,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast matmul: " + String(dtype))


# ---------------------------------------------------------------------------
# CPU GEMM: routes through modular's production CPU matmul
# (`linalg.matmul.matmul` with target="cpu" — tuned packed/tiled inner
# kernels selected by dtype+ISA, fp32 accumulation, dynamic shapes). Honors
# the same transpose_b / element-offset / a-broadcast (a_bstride == 0) /
# row-broadcast bias semantics as the GPU dispatch so callers don't need to
# special-case the device.
#
# Bias (broadcast over rows, any batch):
#   * f32 output: matmul into C, then `_bias_add_row` (exact fp32 add).
#   * f16/bf16 output: matmul into an fp32 scratch, then a fused
#     (scratch + bias) -> cast pass. This preserves our historical semantics
#     of forming A@B + bias in fp32 and casting to the output dtype exactly
#     once — the library's epilogue lambda would instead add bias *after* the
#     accumulator is cast down to the output dtype (KERN-2790 scratch path).
#
# Degenerate shapes go through `_cpu_gemm_naive` (the old parallel-loop
# kernel) instead of the library: the library's CPU route special-cases
# n == 1 into gemv[parallelize=True] WITHOUT forwarding the DeviceContext
# (segfaults in our embedded runtime) and, under transpose_b, builds the
# gemv rhs with length b.dim[0]() == 1 instead of k
# (linalg/matmul/cpu/impl.mojo:421-427 @ the pinned nightly). k == 0 would
# compute num_tasks == 0 and leave C unwritten (it is currently rejected
# upstream by the Python dispatcher, so that guard is just insurance).
# ---------------------------------------------------------------------------


@always_inline
def _cpu_gemm_naive[
    dtype: DType
](
    c_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    a_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    b_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    transpose_b: Bool,
    bias_addr: Int,  # 0 means no bias
    ctx: DeviceContext,
) raises:
    """Correctness-grade GEMM (fp32 accumulate, bias in fp32, single cast)
    for the degenerate shapes the library CPU matmul mishandles."""
    var has_bias = bias_addr != 0
    var bias_ptr = _make_ptr[dtype](bias_addr) if has_bias else c_ptr
    var b_batch_stride = (n * k) if transpose_b else (k * n)

    @always_inline
    @parameter
    @__copy_capture(c_ptr, a_ptr, b_ptr, bias_ptr)
    def row_func(row: Int):
        var bz = row // m
        var mm = row % m
        var a_row = a_ptr + bz * a_bstride + mm * k
        var c_row = c_ptr + row * n
        var b_base = b_ptr + bz * b_batch_stride
        for j in range(n):
            var acc = Float32(0)
            if transpose_b:
                var b_row = b_base + j * k
                for kk in range(k):
                    acc += (
                        a_row[kk].cast[DType.float32]()
                        * b_row[kk].cast[DType.float32]()
                    )
            else:
                for kk in range(k):
                    acc += (
                        a_row[kk].cast[DType.float32]()
                        * b_base[kk * n + j].cast[DType.float32]()
                    )
            if has_bias:
                acc += bias_ptr[j].cast[DType.float32]()
            c_row[j] = acc.cast[dtype]()

    parallelize[row_func](batch * m, ctx)


@always_inline
def _cpu_matmul_one[
    ab_dtype: DType, c_dtype: DType, transpose_b: Bool
](
    c_ptr: UnsafePointer[Scalar[c_dtype], MutUntrackedOrigin],
    a_ptr: UnsafePointer[Scalar[ab_dtype], MutUntrackedOrigin],
    b_ptr: UnsafePointer[Scalar[ab_dtype], MutUntrackedOrigin],
    m: Int,
    n: Int,
    k: Int,
    ctx: DeviceContext,
) raises:
    # C(m, n) = A(m, k) @ B; B is (k, n) row-major when not transposed,
    # (n, k) row-major when transposed. Dynamic dims via runtime `row_major`
    # (same construction the vendor path uses); the library entry derives
    # kernel_type_m from a.static_shape[0] (UNKNOWN -> 0) and, when c_dtype is
    # f16/bf16, wraps the accumulation in an fp32 scratch itself.
    var c = TileTensor(c_ptr, row_major(m, n))
    var a = TileTensor(a_ptr, row_major(m, k))
    comptime if transpose_b:
        var b = TileTensor(b_ptr, row_major(n, k))
        cpu_lib_matmul[transpose_b=True, target="cpu"](c, a, b, ctx=ctx)
    else:
        var b = TileTensor(b_ptr, row_major(k, n))
        cpu_lib_matmul[transpose_b=False, target="cpu"](c, a, b, ctx=ctx)


@always_inline
def _cpu_gemm[
    dtype: DType
](
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    transpose_b: Bool,
    c_off: Int,
    a_off: Int,
    b_off: Int,
    bias_addr: Int,  # 0 means no bias
    ctx: DeviceContext,
) raises:
    var c_base = _make_ptr[dtype](c_addr) + c_off
    var a_base = _make_ptr[dtype](a_addr) + a_off
    var b_base = _make_ptr[dtype](b_addr) + b_off
    var has_bias = bias_addr != 0

    # Shapes the library CPU matmul mishandles (see the section comment):
    # n == 1 (gemv special case: ctx not forwarded -> segfault; wrong rhs
    # length under transpose_b) and k == 0 (C left unwritten). Route those
    # through the naive kernel, which honors the same bias / batch /
    # broadcast semantics.
    if n == 1 or k == 0:
        _cpu_gemm_naive[dtype](
            c_base,
            a_base,
            b_base,
            batch,
            m,
            n,
            k,
            a_bstride,
            transpose_b,
            bias_addr,
            ctx,
        )
        return

    # B is (k, n) row-major when transpose_b is False, (n, k) row-major
    # ("transposed") when True — the same layouts the GPU kernels assume.
    var b_batch_stride = (n * k) if transpose_b else (k * n)

    comptime if dtype == DType.float32:
        # fp32: matmul each batch straight into C, then optional fp32 bias
        # (broadcast over all batch * m rows).
        for bz in range(batch):
            var c_ptr = c_base + bz * (m * n)
            var a_ptr = a_base + bz * a_bstride
            var b_ptr = b_base + bz * b_batch_stride
            if transpose_b:
                _cpu_matmul_one[dtype, dtype, True](
                    c_ptr, a_ptr, b_ptr, m, n, k, ctx
                )
            else:
                _cpu_matmul_one[dtype, dtype, False](
                    c_ptr, a_ptr, b_ptr, m, n, k, ctx
                )
        if has_bias:
            _bias_add_row[dtype](
                c_addr + c_off * size_of[dtype](),
                bias_addr,
                batch * m * n,
                n,
                ctx,
            )
    else:
        # f16 / bf16.
        if has_bias:
            # Accumulate A@B in an fp32 scratch, then add bias in fp32 and
            # cast to the output dtype exactly once.
            var total = batch * m * n
            var scratch = alloc[Scalar[DType.float32]](total)
            var bias_ptr = _make_ptr[dtype](bias_addr)

            @always_inline
            @parameter
            @__copy_capture(scratch, c_base, bias_ptr)
            def add_cast_func[width: Int, alignment: Int = 1](idx: StdCoord):
                var i = Int(idx[0].value())
                var acc = scratch[i] + bias_ptr[i % n].cast[DType.float32]()
                c_base[i] = acc.cast[dtype]()

            try:
                for bz in range(batch):
                    var s_ptr = scratch + bz * (m * n)
                    var a_ptr = a_base + bz * a_bstride
                    var b_ptr = b_base + bz * b_batch_stride
                    if transpose_b:
                        _cpu_matmul_one[dtype, DType.float32, True](
                            s_ptr, a_ptr, b_ptr, m, n, k, ctx
                        )
                    else:
                        _cpu_matmul_one[dtype, DType.float32, False](
                            s_ptr, a_ptr, b_ptr, m, n, k, ctx
                        )
                elementwise[add_cast_func, simd_width=1](StdCoord(total), ctx)
            finally:
                scratch.free()
        else:
            for bz in range(batch):
                var c_ptr = c_base + bz * (m * n)
                var a_ptr = a_base + bz * a_bstride
                var b_ptr = b_base + bz * b_batch_stride
                if transpose_b:
                    _cpu_matmul_one[dtype, dtype, True](
                        c_ptr, a_ptr, b_ptr, m, n, k, ctx
                    )
                else:
                    _cpu_matmul_one[dtype, dtype, False](
                        c_ptr, a_ptr, b_ptr, m, n, k, ctx
                    )


@always_inline
def _cpu_gemm_dtype_dispatch(
    dtype: DType,
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    a_bstride: Int,
    transpose_b: Bool,
    c_off: Int,
    a_off: Int,
    b_off: Int,
    bias_addr: Int,
    ctx: DeviceContext,
) raises:
    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _cpu_gemm[dt](
                    c_addr,
                    a_addr,
                    b_addr,
                    batch,
                    m,
                    n,
                    k,
                    a_bstride,
                    transpose_b,
                    c_off,
                    a_off,
                    b_off,
                    bias_addr,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for CPU matmul: " + String(dtype))


def _matmul_go(
    out_ptr: PyObjectPtr,
    a_ptr: PyObjectPtr,
    b_ptr: PyObjectPtr,
    # (m, n, k, transpose_b) or, with element offsets into the three
    # buffers (grouped convolution), (m, n, k, transpose_b, c_off, a_off,
    # b_off).
    params: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var c_addr = _raw_int(out_ptr)
    var a_addr = _raw_int(a_ptr)
    var b_addr = _raw_int(b_ptr)
    var m = _raw_tuple_int(params, 0)
    var n = _raw_tuple_int(params, 1)
    var k = _raw_tuple_int(params, 2)
    var transpose_b = _raw_tuple_int(params, 3)
    var c_off = 0
    var a_off = 0
    var b_off = 0
    if _raw_tuple_len(params) > 4:
        c_off = _raw_tuple_int(params, 4)
        a_off = _raw_tuple_int(params, 5)
        b_off = _raw_tuple_int(params, 6)
    var ctx = _raw_ctx(device_context_ptr)

    if ctx.api() == "cpu":
        _cpu_gemm_dtype_dispatch(
            dtype,
            c_addr,
            a_addr,
            b_addr,
            1,
            m,
            n,
            k,
            m * k,
            transpose_b != 0,
            c_off,
            a_off,
            b_off,
            0,
            ctx,
        )
        return

    # Single-token (m == 1) decode GEMV: modular's linalg.gemv.gemv_gpu is a
    # single-launch, well-coalesced kernel that beats our smallm split-K path
    # on every decode shape (2-3x for f32, 6-18x for bf16/f16 where our
    # split-K is gated off). Zero-offset calls only: linear/mm always, and
    # also a grouped conv's first group (s=0, g=0), which is safe because
    # that sub-block is contiguous at the base pointers. Nonzero offsets
    # keep the offset-aware GEMM path.
    if m == 1 and c_off == 0 and a_off == 0 and b_off == 0:
        _gemv_dtype_dispatch(
            dtype, c_addr, a_addr, b_addr, m, n, k, transpose_b, ctx
        )
        return

    _gemm_dtype_dispatch(
        dtype,
        c_addr,
        a_addr,
        b_addr,
        1,
        m,
        n,
        k,
        m * k,
        transpose_b,
        c_off,
        a_off,
        b_off,
        ctx,
    )


def _bmm_go(
    out_ptr: PyObjectPtr,
    a_ptr: PyObjectPtr,
    b_ptr: PyObjectPtr,
    # (batch, m, n, k, transpose_b) or (batch, m, n, k, transpose_b,
    # a_shared) — a_shared=1 broadcasts a single (m, k) A across the batch
    # (batched convolution with shared weights).
    params: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var c_addr = _raw_int(out_ptr)
    var a_addr = _raw_int(a_ptr)
    var b_addr = _raw_int(b_ptr)
    var batch = _raw_tuple_int(params, 0)
    var m = _raw_tuple_int(params, 1)
    var n = _raw_tuple_int(params, 2)
    var k = _raw_tuple_int(params, 3)
    var transpose_b = _raw_tuple_int(params, 4)
    var a_bstride = m * k
    if _raw_tuple_len(params) > 5 and _raw_tuple_int(params, 5) != 0:
        a_bstride = 0
    var ctx = _raw_ctx(device_context_ptr)

    if ctx.api() == "cpu":
        _cpu_gemm_dtype_dispatch(
            dtype,
            c_addr,
            a_addr,
            b_addr,
            batch,
            m,
            n,
            k,
            a_bstride,
            transpose_b != 0,
            0,
            0,
            0,
            0,
            ctx,
        )
        return

    _gemm_dtype_dispatch(
        dtype,
        c_addr,
        a_addr,
        b_addr,
        batch,
        m,
        n,
        k,
        a_bstride,
        transpose_b,
        0,
        0,
        0,
        ctx,
    )


# ---------------------------------------------------------------------------
# Causal-aware float32 batched GEMMs for the Apple SDPA forward: identical
# math to Bmm on the positions that matter, but exploiting the causal
# structure of the (batch, q_len, kv_len) score/probability matrices. Row i
# of each batch slice only attends to columns j <= i, so
#   mode 1 (scores = Q @ K^T, transpose_b): tiles strictly above the diagonal
#     are skipped and left UNWRITTEN — callers must never read C past the
#     causal boundary (the Apple row-softmax doesn't);
#   mode 2 (out = P @ V): P's columns past the boundary are exactly zero, so
#     the reduction stops at the tile's last row + 1.
# Roughly halves both FLOPs and traffic at q_len == kv_len. Apple GPU only —
# other targets raise and the Python caller keeps the plain Bmm path.
# ---------------------------------------------------------------------------


def _bmm_causal_go(
    out_ptr: PyObjectPtr,
    a_ptr: PyObjectPtr,
    b_ptr: PyObjectPtr,
    # (batch, m, n, k, transpose_b, causal_mode)
    params: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var dtype = _raw_dtype_int(dtype_obj)
    var c_addr = _raw_int(out_ptr)
    var a_addr = _raw_int(a_ptr)
    var b_addr = _raw_int(b_ptr)
    var batch = _raw_tuple_int(params, 0)
    var m = _raw_tuple_int(params, 1)
    var n = _raw_tuple_int(params, 2)
    var k = _raw_tuple_int(params, 3)
    var transpose_b = _raw_tuple_int(params, 4)
    var causal_mode = _raw_tuple_int(params, 5)
    var ctx = _raw_ctx(device_context_ptr)

    comptime if has_apple_gpu_accelerator():
        if (
            dtype != DType.float32
            or ctx.api() == "cpu"
            or causal_mode < 1
            or causal_mode > 2
        ):
            raise Error("BmmCausalF32: unsupported configuration")
        if causal_mode == 1:
            if transpose_b != 0:
                _apple8_fat_enqueue[True, 64, 64, 2, 2, False, 1](
                    c_addr, a_addr, b_addr, batch, m, n, k, m * k, ctx
                )
            else:
                _apple8_fat_enqueue[False, 64, 64, 2, 2, False, 1](
                    c_addr, a_addr, b_addr, batch, m, n, k, m * k, ctx
                )
        else:
            if transpose_b != 0:
                _apple8_fat_enqueue[True, 64, 64, 2, 2, False, 2](
                    c_addr, a_addr, b_addr, batch, m, n, k, m * k, ctx
                )
            else:
                _apple8_fat_enqueue[False, 64, 64, 2, 2, False, 2](
                    c_addr, a_addr, b_addr, batch, m, n, k, m * k, ctx
                )
    else:
        raise Error("BmmCausalF32 is Apple-GPU only")


def _bmm_causal_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _bmm_causal_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# In-place row-broadcast bias add: out[i] += bias[i % cols]. Used as the
# addmm / conv-bias epilogue.
# ---------------------------------------------------------------------------


@always_inline
def _bias_add_row[
    dtype: DType
](
    out_addr: Int, bias_addr: Int, total: Int, cols: Int, ctx: DeviceContext
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var bias_ptr = _make_ptr[dtype](bias_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, bias_ptr)
    def func[width: Int, alignment: Int = 1](idx: StdCoord):
        var i = Int(idx[0].value())
        out_ptr[i] = out_ptr[i] + bias_ptr[i % cols]

    if ctx.api() == "cpu":
        elementwise[func, simd_width=1](StdCoord(total), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](StdCoord(total), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


def _matmul_bias_run(
    dtype: DType,
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    bias_addr: Int,
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Int,
    ctx: DeviceContext,
) raises:
    """The MatmulBiasSpec tier ladder: CPU library / m==1 gemv+bias /
    f32 fused-epilogue / GEMM+bias."""
    if ctx.api() == "cpu":
        _cpu_gemm_dtype_dispatch(
            dtype,
            c_addr,
            a_addr,
            b_addr,
            1,
            m,
            n,
            k,
            m * k,
            transpose_b != 0,
            0,
            0,
            0,
            bias_addr,
            ctx,
        )
        return

    # Single-token (m == 1) decode: gemv_gpu + row-broadcast bias. gemv beats
    # our smallm split-K path on every decode shape; the bias add is the same
    # cheap epilogue either way. No unsupported-dtype raise needed here:
    # _gemv_dtype_dispatch already raised for anything outside FLOAT_DTYPES.
    if m == 1:
        _gemv_dtype_dispatch(
            dtype, c_addr, a_addr, b_addr, m, n, k, transpose_b, ctx
        )
        comptime for dt in FLOAT_DTYPES:
            comptime if _dtype_arg_on[0, dt]():
                if dtype == dt:
                    _bias_add_row[dt](c_addr, bias_addr, m * n, n, ctx)
        return

    # Apple: skinny-M float32 projections take the vendored cached
    # simdgroup-matrix kernel (32-row tile, bias folded in the store,
    # split-K when the N-tile grid alone cannot fill the GPU). This branch
    # is absent from CUDA/ROCm builds, which retain the existing pure-Mojo
    # SIMT dispatch and tuning constants exactly.
    comptime if has_apple_gpu_accelerator():
        if (
            dtype == DType.float32
            and transpose_b == 0
            and m > SMALLM_MR
            and m <= 32
            and k % MMA8_DIM == 0
        ):
            _apple8_enqueue[True](
                c_addr, a_addr, b_addr, bias_addr, m, n, k, ctx
            )
            return
        # Fat tiles: the simdgroup-matrix GEMM plus a separate row-broadcast
        # bias pass still beats the NVIDIA-tuned fused-bias pipe3 tile here.
        if dtype == DType.float32 and m > 32 and n >= 16:
            if transpose_b != 0:
                _gemm_enqueue[DType.float32, True](
                    c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
                )
            else:
                _gemm_enqueue[DType.float32, False](
                    c_addr, a_addr, b_addr, 1, m, n, k, m * k, ctx
                )
            _bias_add_row[DType.float32](c_addr, bias_addr, m * n, n, ctx)
            return

    # Large-M gfx942 workloads use the dynamic pure-Mojo MFMA kernels and
    # fold the row-broadcast bias into the store epilogue. This check
    # precedes the portable SIMT fused-bias route below.
    comptime if _accelerator_arch() == "amdgpu:gfx942":
        if dtype == DType.float32:
            var used_mfma = _amd_dynamic_mfma_dispatch[
                DType.float32, True, True
            ](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, bias_addr, ctx
            ) if transpose_b != 0 else _amd_dynamic_mfma_dispatch[
                DType.float32, False, True
            ](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, bias_addr, ctx
            )
            if used_mfma:
                return
        if dtype == DType.bfloat16:
            var used_mfma = _amd_dynamic_mfma_dispatch[
                DType.bfloat16, True, True
            ](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, bias_addr, ctx
            ) if transpose_b != 0 else _amd_dynamic_mfma_dispatch[
                DType.bfloat16, False, True
            ](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, bias_addr, ctx
            )
            if used_mfma:
                return

    # Fused-epilogue path: the float32 pipe3 tile kernels add the bias in
    # the GEMM epilogue, saving a full extra read+write of C per call.
    if (
        dtype == DType.float32
        and transpose_b == 0
        and m > SMALLM_MR
        and k % 4 == 0
        and n % 4 == 0
    ):
        if m <= 32:
            _tune_enqueue[
                32, 64, 16, 4, 4, False, True, PIPE3_STAGES, -1, True
            ](c_addr, a_addr, b_addr, 1, m, n, k, m * k, 96, ctx, bias_addr)
        else:
            _tune_enqueue[64, 64, 16, 4, 4, False, True, PIPE3_STAGES, 4, True](
                c_addr, a_addr, b_addr, 1, m, n, k, m * k, 192, ctx, bias_addr
            )
        return

    _gemm_dtype_dispatch(
        dtype,
        c_addr,
        a_addr,
        b_addr,
        1,
        m,
        n,
        k,
        m * k,
        transpose_b,
        0,
        0,
        0,
        ctx,
    )
    var handled = False
    comptime for dt in FLOAT_DTYPES:
        comptime if _dtype_arg_on[0, dt]():
            if dtype == dt:
                _bias_add_row[dt](c_addr, bias_addr, m * n, n, ctx)
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast bias add: " + String(dtype))


# METH_FASTCALL wrappers for the hot dispatchers (raw CPython unpack; the
# internal Python callers guarantee argument types, and raise sites are
# unsupported-dtype guards gated upstream).


def _matmul_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _matmul_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _bmm_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _bmm_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


def _causal_bmm_go(
    out_ptr: PyObjectPtr,
    a_ptr: PyObjectPtr,
    b_ptr: PyObjectPtr,
    # (batch, m, n, k, transpose_b, causal_mode)
    params: PyObjectPtr,
    dtype_obj: PyObjectPtr,
    device_context_ptr: PyObjectPtr,
) raises:
    var ctx = _raw_ctx(device_context_ptr)
    var causal = _raw_tuple_int(params, 5)
    if causal < CAUSAL_NONE or causal > CAUSAL_B_COLS:
        raise Error("CausalBmm: unknown causal mode ", causal)
    _causal_bmm_dispatch(
        _raw_dtype_int(dtype_obj),
        _raw_int(out_ptr),
        _raw_int(a_ptr),
        _raw_int(b_ptr),
        _raw_tuple_int(params, 0),
        _raw_tuple_int(params, 1),
        _raw_tuple_int(params, 2),
        _raw_tuple_int(params, 3),
        _raw_tuple_int(params, 4),
        causal,
        ctx,
    )


# ---------------------------------------------------------------------------
# TensorSpec entries (docs/tensor_spec_design.md): the matmul-family
# prologue — shape/dtype/contiguity gates, m/n/k geometry, output alloc and
# tier dispatch — in one boundary call. The GEMV-vs-GEMM tier ladder stays
# in the shared dispatch helpers below (tier choice on scalar fields is
# Mojo's job); the FAMILY choice (matmul vs linear-with-bias vs bmm) stays
# in Python. Failed checks raise a real NotImplementedError, which declines
# the spec route; operand layout is not one of those checks, since
# `_matmul_spec_operands_launch` handles strided operands itself.
# ---------------------------------------------------------------------------


@always_inline
def _matmul_spec_checks(
    a: TensorSpec, b: TensorSpec, transpose_b: Bool
) raises -> Tuple[Int, Int, Int]:
    """Common gates + (m, n, k) for a @ b (b rank-2, a rank >= 2 with the
    leading dims flattened into m, exactly like fast_aten_linear)."""
    if a.dtype != b.dtype:
        raise Error("mojo spec matmul: operand dtypes differ")
    if a.ctx_ptr != b.ctx_ptr:
        raise Error("mojo spec matmul: operands on different devices")
    if not _dtype_supported[FLOAT_DTYPES](a.dtype):
        raise Error("mojo spec matmul: unsupported dtype ", a.dtype)
    if a.rank < 2 or b.rank != 2:
        raise Error("mojo spec matmul: bad ranks")
    var k = a.shape[MAX_RANK - 1]
    var m = a.numel // k if k > 0 else 0
    var n: Int
    var kb: Int
    if transpose_b:
        n = b.shape[MAX_RANK - 2]
        kb = b.shape[MAX_RANK - 1]
    else:
        kb = b.shape[MAX_RANK - 2]
        n = b.shape[MAX_RANK - 1]
    if kb != k:
        raise Error("mojo spec matmul: inner dims differ")
    if m == 0 or n == 0 or k == 0:
        raise Error("mojo spec matmul: zero-sized dim")
    return (m, n, k)


@always_inline
def _matmul_spec_launch(
    dtype: DType,
    c_addr: Int,
    a_addr: Int,
    b_addr: Int,
    bias_addr: Int,
    has_bias: Bool,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Int,
    ctx: DeviceContext,
) raises:
    """The spec ops' tier launch over raw operand addresses. batch > 1 is
    the bmm shape (no bias, no GEMV tier — mirrors the classic entries)."""
    if has_bias:
        _matmul_bias_run(
            dtype, c_addr, a_addr, b_addr, bias_addr, m, n, k, transpose_b, ctx
        )
        return
    if ctx.api() == "cpu":
        _cpu_gemm_dtype_dispatch(
            dtype,
            c_addr,
            a_addr,
            b_addr,
            batch,
            m,
            n,
            k,
            m * k,
            transpose_b != 0,
            0,
            0,
            0,
            0,
            ctx,
        )
        return
    if batch == 1 and m == 1:
        _gemv_dtype_dispatch(
            dtype, c_addr, a_addr, b_addr, m, n, k, transpose_b, ctx
        )
        return
    _gemm_dtype_dispatch(
        dtype,
        c_addr,
        a_addr,
        b_addr,
        batch,
        m,
        n,
        k,
        m * k,
        transpose_b,
        0,
        0,
        0,
        ctx,
    )


@always_inline
def _apple_ta_spec_route(
    a: TensorSpec,
    b_addr: Int,
    has_bias: Bool,
    c_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Apple f32 fast path for a strided `a` that is exactly the 2D
    transpose view of a contiguous (k, m) matrix — strides (1, m), i.e.
    linear-backward's dW = dY^T @ X. The transposed-A simdgroup-matrix GEMM
    consumes the stored (k, m) layout directly, so the permute-copy
    materialization (a full extra pass over A plus a launch) is skipped.
    Returns False when the operand isn't that pattern (caller keeps the
    scratch-copy fallback)."""
    if (
        has_bias
        or batch != 1
        or transpose_b != 0
        or a.dtype != DType.float32
        or a.rank != 2
        or a.strides[MAX_RANK - 2] != 1
        or a.strides[MAX_RANK - 1] != m
        or ctx.api() == "cpu"
    ):
        return False
    # Dedicated TN kernels: same 8x8-fragment design as the shared fat
    # kernel, with the block height picked from the reduction depth and the
    # grid size (see apple_gemm_tn_kernels.mojo for the measurements).
    apple_tn_gemm_enqueue(c_addr, a.ptr, b_addr, 1, m, n, k, m * k, ctx)
    return True


@always_inline
def _matmul_spec_operands_launch(
    a: TensorSpec,
    b: TensorSpec,
    bias_addr: Int,
    has_bias: Bool,
    c_addr: Int,
    batch: Int,
    m: Int,
    n: Int,
    k: Int,
    transpose_b: Int,
    ctx: DeviceContext,
) raises:
    """Launch with Mojo-side temporaries for strided operands: the hot
    (contiguous) path is branch-only; a strided a/b materializes into a
    scratch buffer that lives until its launch is enqueued."""
    if a.contig and b.contig:
        _matmul_spec_launch(
            a.dtype,
            c_addr,
            a.ptr,
            b.ptr,
            bias_addr,
            has_bias,
            batch,
            m,
            n,
            k,
            transpose_b,
            ctx,
        )
    elif a.contig:
        var tmp_b = _scratch_contig(b, ctx)
        _matmul_spec_launch(
            a.dtype,
            c_addr,
            a.ptr,
            Int(tmp_b.unsafe_ptr()),
            bias_addr,
            has_bias,
            batch,
            m,
            n,
            k,
            transpose_b,
            ctx,
        )
        _ = tmp_b^
    elif b.contig:
        comptime if has_apple_gpu_accelerator():
            if _apple_ta_spec_route(
                a, b.ptr, has_bias, c_addr, batch, m, n, k, transpose_b, ctx
            ):
                return
        # A strided A whose two innermost strides are `(1, lda)` is a transposed
        # view: A^T is the dense `(k, lda)` buffer, and this is the weight
        # gradient.  The MFMA route reads it in place, which removes a full read
        # and write of A that PyTorch-ROCm never pays -- measured against
        # materializing A^T and running the dense route, one shape per process:
        # (2304, 768, 49152) 575.8 -> 483.6 us, (768, 768, 49152) 197.7 -> 171.7,
        # (3072, 768, 49152) 661.4 -> 509.1, (768, 3072, 49152) 527.1 -> 508.8,
        # (50304, 768, 49152) 9600.5 -> 7369.0.  It declines whenever its grid
        # cannot cover the device even split along K, and then the copy runs as
        # before.
        comptime if _accelerator_arch() == "amdgpu:gfx942":
            # `_accelerator_arch()` is the build target, not the context: a
            # gfx942 build also serves the CPU mojo device, whose context has
            # no multiprocessor count and cannot take a GPU launch.  The Apple
            # route above declines CPU for the same reason.
            if (
                ctx.api() != "cpu"
                and not has_bias
                and batch == 1
                and transpose_b == 0
                and a.dtype == DType.bfloat16
                and a.strides[MAX_RANK - 2] == 1
                and a.strides[MAX_RANK - 1] == m
                and _leading_trivial(a)
            ):
                if _tn_mfma_route[DType.bfloat16](
                    c_addr, a.ptr, b.ptr, m, n, k, ctx
                ):
                    return
        # NVIDIA sm_90, strict fp32: the same transposed-dense-A trigger as
        # the two routes above (linear-backward's dW = dY^T @ X).  The
        # dedicated CUDA-core FFMA kernels (tn_f32_gemm_kernels.mojo) read
        # the stored (k, m) layout in place, skipping the `_scratch_contig`
        # materialization (a full extra pass over A plus a launch).  Gated
        # at runtime to compute capability 9.0 exactly inside
        # `try_enqueue_tn_f32_gemm` — its tile/wave constants are fitted to
        # H100's 114 SMs — so every other NVIDIA part keeps the copy path
        # below, and it declines regimes where the copy path is better
        # (m == 1 stays on the GEMV-after-copy route).
        comptime if _has_sm_9x():
            comptime if _dtype_arg_on[0, DType.float32]():
                if (
                    ctx.api() == "cuda"
                    and not has_bias
                    and batch == 1
                    and transpose_b == 0
                    and a.dtype == DType.float32
                    and a.strides[MAX_RANK - 2] == 1
                    and a.strides[MAX_RANK - 1] == m
                    and _leading_trivial(a)
                ):
                    if try_enqueue_tn_f32_gemm(
                        c_addr, a.ptr, b.ptr, m, n, k, ctx
                    ):
                        return
        var tmp_a = _scratch_contig(a, ctx)
        _matmul_spec_launch(
            a.dtype,
            c_addr,
            Int(tmp_a.unsafe_ptr()),
            b.ptr,
            bias_addr,
            has_bias,
            batch,
            m,
            n,
            k,
            transpose_b,
            ctx,
        )
        _ = tmp_a^
    else:
        var tmp_a = _scratch_contig(a, ctx)
        var tmp_b = _scratch_contig(b, ctx)
        _matmul_spec_launch(
            a.dtype,
            c_addr,
            Int(tmp_a.unsafe_ptr()),
            Int(tmp_b.unsafe_ptr()),
            bias_addr,
            has_bias,
            batch,
            m,
            n,
            k,
            transpose_b,
            ctx,
        )
        _ = tmp_a^
        _ = tmp_b^


def _matmul_spec_into_go(
    a_o: PyObjectPtr, b_o: PyObjectPtr, tb_o: PyObjectPtr, out_o: PyObjectPtr
) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    ref b = _spec_ptr(b_o)[]
    var transpose_b = _raw_int(tb_o)
    var geom = _matmul_spec_checks(a, b, transpose_b != 0)
    var m = geom[0]
    var n = geom[1]
    var k = geom[2]

    var ctx = a.ctx()
    var numel = m * n
    var nbytes = numel * a.itemsize
    _ = nbytes
    if out.numel != m * n or not out.contig or out.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec into: output buffer mismatch")
    if out.dtype != a.dtype:
        raise Error("mojo spec into: output dtype mismatch")
    var addr = out.ptr

    _matmul_spec_operands_launch(
        a, b, 0, False, addr, 1, m, n, k, transpose_b, ctx
    )

    # out shape: a's leading dims with the last dim replaced by n.
    var oshape = a.shape
    oshape[MAX_RANK - 1] = n


def _matmul_bias_spec_into_go(
    a_o: PyObjectPtr,
    b_o: PyObjectPtr,
    bias_o: PyObjectPtr,
    tb_o: PyObjectPtr,
    out_o: PyObjectPtr,
) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    ref b = _spec_ptr(b_o)[]
    ref bias = _spec_ptr(bias_o)[]
    var transpose_b = _raw_int(tb_o)
    var geom = _matmul_spec_checks(a, b, transpose_b != 0)
    var m = geom[0]
    var n = geom[1]
    var k = geom[2]
    if bias.dtype != a.dtype:
        raise Error("mojo spec matmul: bias dtype differs")
    if bias.rank != 1 or bias.numel != n:
        raise Error("mojo spec matmul: bias must be a length-n vector")

    var ctx = a.ctx()
    var numel = m * n
    var nbytes = numel * a.itemsize
    _ = nbytes
    if out.numel != m * n or not out.contig or out.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec into: output buffer mismatch")
    if out.dtype != a.dtype:
        raise Error("mojo spec into: output dtype mismatch")
    var addr = out.ptr
    if bias.contig:
        _matmul_spec_operands_launch(
            a, b, bias.ptr, True, addr, 1, m, n, k, transpose_b, ctx
        )
    else:
        var tmp_bias = _scratch_contig(bias, ctx)
        _matmul_spec_operands_launch(
            a,
            b,
            Int(tmp_bias.unsafe_ptr()),
            True,
            addr,
            1,
            m,
            n,
            k,
            transpose_b,
            ctx,
        )
        _ = tmp_bias^

    var oshape = a.shape
    oshape[MAX_RANK - 1] = n


def _bmm_spec_into_go(
    a_o: PyObjectPtr, b_o: PyObjectPtr, tb_o: PyObjectPtr, out_o: PyObjectPtr
) raises:
    ref a = _spec_ptr(a_o)[]
    ref out = _spec_ptr(out_o)[]
    ref b = _spec_ptr(b_o)[]
    var transpose_b = _raw_int(tb_o)

    if a.dtype != b.dtype:
        raise Error("mojo spec bmm: operand dtypes differ")
    if a.ctx_ptr != b.ctx_ptr:
        raise Error("mojo spec bmm: operands on different devices")
    if not _dtype_supported[FLOAT_DTYPES](a.dtype):
        raise Error("mojo spec bmm: unsupported dtype ", a.dtype)
    if a.rank != 3 or b.rank != 3:
        raise Error("mojo spec bmm: rank != 3")
    var batch = a.shape[MAX_RANK - 3]
    var m = a.shape[MAX_RANK - 2]
    var k = a.shape[MAX_RANK - 1]
    if b.shape[MAX_RANK - 3] != batch:
        raise Error("mojo spec bmm: batch dims differ")
    var n: Int
    var kb: Int
    if transpose_b != 0:
        n = b.shape[MAX_RANK - 2]
        kb = b.shape[MAX_RANK - 1]
    else:
        kb = b.shape[MAX_RANK - 2]
        n = b.shape[MAX_RANK - 1]
    if kb != k:
        raise Error("mojo spec bmm: inner dims differ")
    if batch == 0 or m == 0 or n == 0 or k == 0:
        raise Error("mojo spec bmm: zero-sized dim")

    var ctx = a.ctx()
    var numel = batch * m * n
    var nbytes = numel * a.itemsize
    _ = nbytes
    if out.numel != batch * m * n or not out.contig or out.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec into: output buffer mismatch")
    if out.dtype != a.dtype:
        raise Error("mojo spec into: output dtype mismatch")
    var addr = out.ptr
    _matmul_spec_operands_launch(
        a, b, 0, False, addr, batch, m, n, k, transpose_b, ctx
    )

    var oshape = IndexList[MAX_RANK](1)
    oshape[MAX_RANK - 3] = batch
    oshape[MAX_RANK - 2] = m
    oshape[MAX_RANK - 1] = n


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_matmul_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("matmul_ops")
        comptime if _op_on["MatmulSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[_matmul_spec_into_go, "MatmulSpec"],
                docstring="(a_spec, b_spec, transpose_b, out_spec)",
            )
        comptime if _op_on["MatmulBiasSpec"]():
            _register_call(
                b,
                _spec_dispatcher5[_matmul_bias_spec_into_go, "MatmulBiasSpec"],
                docstring="(a_spec, b_spec, bias_spec, transpose_b, out_spec)",
            )
        comptime if _op_on["BmmSpec"]():
            _register_call(
                b,
                _spec_dispatcher4[_bmm_spec_into_go, "BmmSpec"],
                docstring="(a_spec, b_spec, transpose_b, out_spec)",
            )
        comptime if _op_on["Matmul"]():
            _register_call(
                b,
                _matmul_dispatcher,
                docstring=(
                    "C = A @ B (row-major, optional transposed B); pure Mojo"
                    " tiled kernels on GPU, modular's linalg matmul on CPU"
                ),
            )
        comptime if _op_on["Bmm"]():
            _register_call(
                b,
                _bmm_dispatcher,
                docstring=(
                    "batched C = A @ B (rank 3, optional transposed B); pure"
                    " Mojo tiled kernels on GPU, modular's linalg matmul on CPU"
                ),
            )
        comptime if _op_on["BmmCausalF32"]():
            _register_call(
                b,
                _bmm_causal_dispatcher,
                docstring=(
                    "causal-structured batched f32 GEMM for SDPA on Apple GPUs:"
                    " mode 1 skips (and leaves unwritten) score tiles above the"
                    " diagonal, mode 2 cuts the reduction at the causal"
                    " boundary"
                ),
            )
        comptime if _op_on["CausalBmm"]():
            _register_call(
                b,
                _spec_dispatcher6[_causal_bmm_go, "CausalBmm"],
                docstring=(
                    "(out_ptr, a_ptr, b_ptr, (batch, m, n, k, transpose_b,"
                    " causal_mode), dtype, context_ptr); batched C = A @ B that"
                    " skips the contraction indices a top-left-aligned causal"
                    " mask kills. Mode 1 leaves the masked half of the output"
                    " unwritten (its consumer must read only each row's live"
                    " prefix), modes 2 and 3 are exact for any consumer"
                ),
            )
        b.def_function[_matmul_tune_dispatcher](
            "MatmulTune",
            docstring=(
                "GEMM with explicit tile cfg + split-K floor — tuning only"
            ),
        )
        b.def_function[_amd_bf16_tune_dispatcher](
            "AmdBf16Tune",
            docstring="BF16 MFMA tile sweep — benchmarking only",
        )
        return b.finalize()
    except e:
        abort(t"failed to create matmul_ops python module: {e}")
