"""Fused causal flash-attention backward for gfx942. THE FILE TO OPTIMIZE.

The contract below is frozen; `harness/nanogpt_train/bench_flash_attention_bwd.mojo`
calls exactly this entry point and defines acceptance.

`enqueue_flash_attention_bwd` dispatches between two implementations:

* a pair of MFMA kernels, `_bwd_dkv_mfma` and `_bwd_dq_mfma`, selected when the
  device is HIP, the dtype is a half float and `head_dim % 8 == 0`;
* the original one-block-per-row baselines, which stay as the general correct
  path for every other shape and device.

## The math, stated once

With `S = scale * Q K^T` masked causally, `P = softmax(S)` row-wise, `O = P V`:

    D_i  = sum_d dO[i,d] * O[i,d]           (one scalar per query row)
    dP   = dO V^T
    dS   = P * (dP - D)                      (row-broadcast D)
    dV   = P^T dO
    dQ   = scale * dS K
    dK   = scale * dS^T Q

`P` is not passed in and must not be materialized for the whole matrix: recover
it as `exp(S - L)` where `L` is the per-row log-sum-exp handed in by the forward,
which is why `softmax_lse` is an argument.

## Invariants any replacement must keep

* `batch`, `heads`, `seq_q`, `seq_kv` and `head_dim` are RUNTIME values, and so
  is every STRIDE. Tile shapes, pipeline depth and head-dim regimes may be
  compile-time as long as a runtime dispatch picks between them and every
  runtime head_dim is handled.
* The five READ operands -- `grad_output`, `query`, `key`, `value`, `out_fwd` --
  each carry their own `RowStrides` (batch, head, seq) triple, because the real
  caller hands over `x.view(B, T, H, D).transpose(1, 2)` views and materializing
  them cost 1.83 ms/step of pure copy. `head_dim`'s stride is not carried: it
  must be 1, since that is the axis every vectorized load and every LDS row fill
  runs along. `softmax_lse` is dense.
* `dq`, `dk` and `dv` are WRITTEN through their own `RowStrides` for the same
  reason, so that the `transpose(1, 2)` sitting between the attention and
  `x.view(B, T, H, D)` on the way IN has a free backward. A `[B, H, T, D]`
  gradient stored `[B, T, H, D]` transposes to a contiguous tensor, so the
  reshape autograd performs on it is a view rather than a gather; three dense
  gradients per layer cost nanoGPT 1.38 ms/step of `clone` (journal D11).
  head_dim's stride must be 1 here too: the epilogues store eight contiguous
  head-dim elements at a time.
* No allocation, no host transfer, no synchronization: enqueue on the caller's
  `DeviceContext` and return. Multiple kernel launches are fine.
* Write only to `dq`, `dk`, `dv`. Everything else is read-only. Those three are
  WRITE-ONLY and arrive UNINITIALIZED: a replacement must store every live
  element, because the bridge no longer zeros them (that cost 728 us/step and
  bought nothing). Accumulate in registers and store, as both kernels here do.
  A kernel that genuinely needs to accumulate in global memory must zero them
  itself AND avoid two workgroups touching one element without atomics --
  silently losing a contribution is the classic defect in this family.
* Accumulate in FP32, store once in `dtype`.
* Seed any running max with `Float32.MIN_FINITE`, never `Float32.MIN` -- the
  latter is -inf, so a lane that never runs its loop body computes
  `exp(-inf - -inf)` = nan, which poisons a whole row through a reduction. That
  defect has shipped three times in this repository. `-Float32.MAX` fails the
  same way, since `Float32.MAX` is inf.
* `is_causal` masks strictly above the diagonal aligned to the BOTTOM right, so
  query row `q` attends key indices `0 ..= q`, whatever `seq_kv` is --
  PyTorch's TOP-LEFT alignment. Equivalently, key `j` is attended by query
  rows `j ..< seq_q`.
  Both directions matter here, because dK/dV iterate the other way round.

## Why there are two kernels, and which way each one transposes its score tile

dQ accumulates over keys for a fixed query; dK and dV accumulate over queries for
a fixed key. Owning one output per workgroup is what makes the accumulation
race-free with no atomics, so the two loop orders become two kernels. The price
is that S and dP are computed twice -- seven GEMMs where a single fused pass
would do five -- and that is the cheapest correct arrangement here, because dq
is a half-float buffer and a half-float atomic accumulation would violate
"accumulate in FP32, store once".

Everything else follows from the MFMA fragment algebra. For
`v_mfma_f32_32x32x8bf16_1k` the accumulator holds, in lane `L` register `p`, the
element at row `8*(p//4) + (p%4) + 4*(L//32)`, column `L%32`; both the A operand
(`A[m, k]`, `m = L%32`, `k = 4*(L//32)+j`) and the B operand (`B[k, n]`,
`n = L%32`, `k = 4*(L//32)+j`) have the SAME register layout. So an accumulator
can be fed straight back in as an operand whose `k` axis is the accumulator's
ROW axis and whose free axis is the accumulator's COLUMN axis -- with a bf16 cast
and a slice, no shuffle and no LDS round trip. That single fact fixes both
orientations:

* `_bwd_dq_mfma` contracts the second GEMM over kv, so it wants `k = kv`, so the
  score accumulator must have kv along its rows: it computes `S^T[kv, q]`,
  exactly like the fused forward. `dQ^T[d, q] = K^T[d, kv] dS^T[kv, q]`, with
  `K^T` staged transposed in LDS and `dS^T` coming out of the registers that the
  first two GEMMs just wrote. A lane owns one query, so `L` and `D` are per-lane
  scalars and there is no block reduction anywhere.
* `_bwd_dkv_mfma` contracts its second GEMMs over q, so it wants `k = q`, so the
  score accumulator must have q along its rows: it computes `S[q, kv]`, the
  other way round. `dV^T[d, kv] = dO^T[d, q] P[q, kv]` and
  `dK^T[d, kv] = Q^T[d, q] dS[q, kv]`. A lane owns one key, and `L` and `D` are
  now per-accumulator-ROW, so they are staged in LDS as an interleaved
  `[-L*log2(e), D]` pair per query and read back with one broadcast
  `ds_read_b64` per row.

The contraction over q is why `_bwd_dkv_mfma` needs Q and dO in BOTH layouts:
an MFMA operand always holds four elements adjacent along `k`, so contracting
over q needs q contiguous in both operands, while the score GEMM that produces
`S[q, kv]` needs `Q[q, d]` with d contiguous. There is no single LDS layout that
serves both, so the tile loop stages four tiles: `Q`, `Q^T`, `dO`, `dO^T`. The
transposing loader is the fused forward's V loader -- four scalar reads at one
head-dim column, which is 128 contiguous bytes per wavefront and one conflict-
free `ds_write_b64`.

## D_i without a scratch buffer

The usual flash-attention backward runs a preprocess kernel that writes
`D = rowsum(dO * O)` to a `[batch, heads, seq_q]` workspace. The contract here
forbids allocation, so both kernels recompute it from `O`. In `_bwd_dq_mfma` it
is free: the lane already holds its own `dO` fragments, so it loads the matching
`O` fragments, multiplies, and finishes with the same `shuffle_xor(.., 32)` that
pairs the two half-waves owning a query. In `_bwd_dkv_mfma` it costs one extra
global tile read of `O` per query tile, reduced across the `head_dim/8` lanes
that share a row.

## LDS layout

Every tile is row-major with the row length padded to `X + 4` elements, the same
rule the fused forward uses: an MFMA fragment read is `ds_read_b64` at
`(L%32)*PAD + c`, i.e. dword `(L%32)*(PAD/2) + c/2`, and `PAD/2 == 2 (mod 4)`
makes `{(L%32)*(PAD/2) mod 32}` sixteen distinct even banks over lanes 0..15,
so a 16-lane LDS cycle covers all 32 banks exactly once.

## The gfx942 MFMA-read hazard

Reading an MFMA destination register near a branch can silently drop a k step on
this target (journal D6/D8). Both epilogues therefore read and round every
accumulator ONCE, unconditionally, in straight-line code, fenced with
`llvm.amdgcn.sched.barrier` on both sides, before any guarded store runs.
"""

from flash_attention_fwd_kernels import RowStrides, _is_dense
from max.gpu.sync import barrier
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    grid_dim,
    thread_idx,
)
from max.gpu.compute.mma import mma
from max.gpu.host import DeviceContext
from std.memory import AddressSpace
from max.gpu.primitives import block
from std.gpu.primitives.warp import shuffle_xor
from std.math import ceildiv, exp, exp2
from std.memory import bitcast, stack_allocation
from std.sys.info import is_amd_gpu
from std.sys.intrinsics import llvm_intrinsic
from std.utils.static_tuple import StaticTuple

comptime THREADS = 256
comptime MAX_HEAD_DIM = 256

# log2(e): the exponent runs in log2 units so it is a bare `v_exp_f32`.
comptime LOG2E = Float32(1.4426950408889634)

# Row of the 32x32 MFMA accumulator held by register `p`, before the
# `4 * (lane // 32)` half-wave shift. Hardware-determined.
comptime ACC_ROWS = SIMD[DType.int32, 16](
    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
)

# A masked score is a finite -1e30, and a row that does not exist gets a finite
# -1e30 in place of `-lse * log2(e)`; both drive `exp2` to an exact zero without
# any lane ever evaluating `exp(-inf - -inf)`.
comptime NEG_BIG = SIMD[DType.float32, 16](-1.0e30)
comptime NEG_ROW = Float32(-1.0e30)


@always_inline
def _pack_half[
    dtype: DType, w: Int
](x: SIMD[DType.float32, w]) -> SIMD[dtype, w]:
    """Round FP32 to `dtype` cheaply.

    gfx942 has no `v_cvt_pk_bf16_f32`, so `SIMD.cast` to bfloat16 expands into a
    six-instruction round-to-nearest-even sequence per element. Everything
    rounded here is finite -- `exp2` of a non-positive number, or a product of
    it with a difference of two dot products -- so adding half an ULP to the bit
    pattern and truncating rounds to nearest for both signs at one add plus a
    byte selection.
    """
    comptime if dtype == DType.bfloat16:
        var bits = bitcast[DType.uint32, w](x) + SIMD[DType.uint32, w](0x8000)
        return bitcast[dtype, w](
            bitcast[DType.uint16, 2 * w](bits).deinterleave()[1]
        )
    else:
        return x.cast[dtype]()


@always_inline
def _sched_fence() -> None:
    """Stop the scheduler moving an MFMA-destination read across this point."""
    llvm_intrinsic["llvm.amdgcn.sched.barrier", NoneType](Int32(0))


@__name(t"fa_bwd_dq_baseline_{dtype}")
def _bwd_dq_baseline[
    dtype: DType
](
    dq: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    out_fwd: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    g_st: RowStrides,
    q_st: RowStrides,
    k_st: RowStrides,
    v_st: RowStrides,
    o_st: RowStrides,
    dq_st: RowStrides,
    seq_q_arg: Int64,
    seq_kv_arg: Int64,
    head_dim_arg: Int64,
    scale: Float32,
    causal_arg: Int64,
):
    """One block per (batch, head, query row) accumulating dQ for that row."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var seq_q = Int(seq_q_arg)
    var seq_kv = Int(seq_kv_arg)
    var head_dim = Int(head_dim_arg)
    var causal = Int(causal_arg)
    var tid = Int(thread_idx.x)
    var qi = Int(block_idx.x)
    var head = Int(block_idx.y)
    var batch = Int(block_idx.z)
    var heads = Int(grid_dim.y)

    var bh = batch * heads + head
    # Every operand, dQ included, walks its own strides.
    var g_row = batch * g_st.batch + head * g_st.head + qi * g_st.seq
    var q_row = batch * q_st.batch + head * q_st.head + qi * q_st.seq
    var o_row = batch * o_st.batch + head * o_st.head + qi * o_st.seq
    var k_base = batch * k_st.batch + head * k_st.head
    var v_base = batch * v_st.batch + head * v_st.head
    var dq_row = batch * dq_st.batch + head * dq_st.head + qi * dq_st.seq

    var acc = stack_allocation[
        MAX_HEAD_DIM, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var q_s = stack_allocation[
        MAX_HEAD_DIM, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var do_s = stack_allocation[
        MAX_HEAD_DIM, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var d = tid
    while d < head_dim:
        q_s[d] = query[q_row + d].cast[DType.float32]()
        do_s[d] = grad_output[g_row + d].cast[DType.float32]()
        acc[d] = 0.0
        d += THREADS
    barrier()

    # D_i = sum_d dO[i,d] * O[i,d], one scalar for this row.
    var partial = Float32(0.0)
    var dd = tid
    while dd < head_dim:
        partial += do_s[dd] * out_fwd[o_row + dd].cast[DType.float32]()
        dd += THREADS
    var row_d = block.sum[block_size=THREADS](partial)
    var row_lse = lse[bh * seq_q + qi]

    var limit = seq_kv
    if causal != 0:
        limit = min(seq_kv, qi + 1)

    # Every thread walks every key, so the whole block agrees on ds; the
    # head_dim axis is what is split across threads.
    for j in range(limit):
        var krow = k_base + j * k_st.seq
        var vrow = v_base + j * v_st.seq
        var sp = Float32(0.0)
        var dp = Float32(0.0)
        var e = tid
        while e < head_dim:
            sp += q_s[e] * key[krow + e].cast[DType.float32]()
            dp += do_s[e] * value[vrow + e].cast[DType.float32]()
            e += THREADS
        var s_dot = block.sum[block_size=THREADS](sp)
        var dp_dot = block.sum[block_size=THREADS](dp)
        var p = exp(s_dot * scale - row_lse)
        var ds = p * (dp_dot - row_d)
        var e2 = tid
        while e2 < head_dim:
            acc[e2] += ds * key[krow + e2].cast[DType.float32]() * scale
            e2 += THREADS
        barrier()

    var o = tid
    while o < head_dim:
        dq[dq_row + o] = acc[o].cast[dtype]()
        o += THREADS


@__name(t"fa_bwd_dkv_baseline_{dtype}")
def _bwd_dkv_baseline[
    dtype: DType
](
    dk: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    out_fwd: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    g_st: RowStrides,
    q_st: RowStrides,
    k_st: RowStrides,
    v_st: RowStrides,
    o_st: RowStrides,
    dk_st: RowStrides,
    dv_st: RowStrides,
    seq_q_arg: Int64,
    seq_kv_arg: Int64,
    head_dim_arg: Int64,
    scale: Float32,
    causal_arg: Int64,
):
    """One block per (batch, head, key row) accumulating dK and dV for that row.

    Owning a key row per workgroup is what makes the accumulation race-free
    without atomics: every contribution to `dk[j]` and `dv[j]` comes from this
    block alone.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var seq_q = Int(seq_q_arg)
    var seq_kv = Int(seq_kv_arg)
    var head_dim = Int(head_dim_arg)
    var causal = Int(causal_arg)
    var tid = Int(thread_idx.x)
    var j = Int(block_idx.x)
    var head = Int(block_idx.y)
    var batch = Int(block_idx.z)
    var heads = Int(grid_dim.y)

    var bh = batch * heads + head
    # Every operand, dK and dV included, walks its own strides.
    var k_row = batch * k_st.batch + head * k_st.head + j * k_st.seq
    var v_row = batch * v_st.batch + head * v_st.head + j * v_st.seq
    var g_base = batch * g_st.batch + head * g_st.head
    var q_base = batch * q_st.batch + head * q_st.head
    var o_base = batch * o_st.batch + head * o_st.head
    var dk_row = batch * dk_st.batch + head * dk_st.head + j * dk_st.seq
    var dv_row = batch * dv_st.batch + head * dv_st.head + j * dv_st.seq

    var acc_k = stack_allocation[
        MAX_HEAD_DIM, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var acc_v = stack_allocation[
        MAX_HEAD_DIM, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var k_s = stack_allocation[
        MAX_HEAD_DIM, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var v_s = stack_allocation[
        MAX_HEAD_DIM, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var d = tid
    while d < head_dim:
        k_s[d] = key[k_row + d].cast[DType.float32]()
        v_s[d] = value[v_row + d].cast[DType.float32]()
        acc_k[d] = 0.0
        acc_v[d] = 0.0
        d += THREADS
    barrier()

    # Key j is attended by query rows from `first_q` upward.
    var first_q = 0
    if causal != 0:
        first_q = j

    for qi in range(first_q, seq_q):
        var q_row = q_base + qi * q_st.seq
        var g_row = g_base + qi * g_st.seq
        var o_row = o_base + qi * o_st.seq
        var sp = Float32(0.0)
        var dp = Float32(0.0)
        var dsum = Float32(0.0)
        var e = tid
        while e < head_dim:
            var dov = grad_output[g_row + e].cast[DType.float32]()
            sp += k_s[e] * query[q_row + e].cast[DType.float32]()
            dp += dov * v_s[e]
            dsum += dov * out_fwd[o_row + e].cast[DType.float32]()
            e += THREADS
        var s_dot = block.sum[block_size=THREADS](sp)
        var dp_dot = block.sum[block_size=THREADS](dp)
        var row_d = block.sum[block_size=THREADS](dsum)
        var p = exp(s_dot * scale - lse[bh * seq_q + qi])
        var ds = p * (dp_dot - row_d)
        var e2 = tid
        while e2 < head_dim:
            acc_v[e2] += p * grad_output[g_row + e2].cast[DType.float32]()
            acc_k[e2] += ds * query[q_row + e2].cast[DType.float32]() * scale
            e2 += THREADS
        barrier()

    var o = tid
    while o < head_dim:
        dk[dk_row + o] = acc_k[o].cast[dtype]()
        dv[dv_row + o] = acc_v[o].cast[dtype]()
        o += THREADS


# Without the flat-work-group-size metadata the AMDGPU backend assumes up to
# 1024 threads per block, caps the kernel at 128 VGPRs and spills every
# accumulator to scratch. 256 threads is four wave64, one per SIMD.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS)),
    `rocdl.waves_per_eu`=SIMDSize(WAVES_PER_EU),
)
@__name(t"fa_bwd_dq_mfma_{dtype}_h{HD}_n{BN}_q{QT}_x{EXACT}_d{DENSE}")
def _bwd_dq_mfma[
    dtype: DType,
    HD: Int,
    BN: Int,
    QT: Int,
    EXACT: Bool,
    DENSE: Bool,
    WAVES_PER_EU: Int,
    IGLP: Int,
    PEEL: Bool,
](
    dq: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    out_fwd: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    g_st: RowStrides,
    q_st: RowStrides,
    k_st: RowStrides,
    v_st: RowStrides,
    o_st: RowStrides,
    dq_st: RowStrides,
    seq_q_arg: Int64,
    seq_kv_arg: Int64,
    head_dim_arg: Int64,
    scale: Float32,
    causal_arg: Int64,
):
    """dQ only: one workgroup owns `4 * 32 * QT` query rows and walks kv tiles.

    The score tile is transposed (`S^T[kv, q]`), so a lane owns one query, `L`
    and `D` are per-lane scalars, and `dS^T` feeds the `dQ^T = K^T dS^T` GEMM
    directly out of the accumulator registers.

    `DENSE` asserts the five read operands are row-major `[b, h, seq, head_dim]`
    and restores the compile-time row offsets in the K and V loaders; see the
    same parameter on the forward kernel for the measurement behind it. It says
    nothing about `dq_st`: the epilogue's store address was already a runtime
    multiply by `head_dim`, so reading the row stride out of `dq_st` instead
    costs nothing.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var seq_q = Int(seq_q_arg)
    var seq_kv = Int(seq_kv_arg)
    var head_dim = Int(head_dim_arg)
    var causal = Int(causal_arg)
    comptime NW = 4
    comptime BM = NW * 32 * QT
    comptime KSTEPS = HD // 8  # k-steps of the score / dP GEMMs (over d)
    comptime KVT = BN // 32  # kv MFMA tiles per score tile
    comptime DT = HD // 32  # head-dim MFMA tiles of dQ
    comptime KPAD = HD + 4  # LDS row length of the row-major K and V tiles
    comptime TPAD = BN + 4  # LDS row length of the K^T tile
    comptime OPAD = 36  # LDS row length of the epilogue staging tile
    comptime KITERS = BN * HD // 8 // THREADS
    comptime TITERS = HD * (BN // 4) // THREADS
    comptime OPASSES = BM // 64
    comptime LDS_ELEMS = max(2 * BN * KPAD + HD * TPAD, BM * OPAD)

    comptime if is_amd_gpu():
        var smem = stack_allocation[
            LDS_ELEMS, dtype, address_space=AddressSpace.SHARED, alignment=16
        ]()
        var k_smem = smem
        var v_smem = smem + BN * KPAD
        var kt_smem = smem + 2 * BN * KPAD

        var tid = Int(thread_idx.x)
        var wave = tid // 64
        var lane = tid % 64
        var lo = lane % 32  # index along the MFMA's non-K axis
        var hi = lane // 32  # which half-wave, i.e. which K quad

        var heads = Int(grid_dim.y)
        var bz = Int(block_idx.z)
        var by = Int(block_idx.y)
        var bh = bz * heads + by
        # Every base follows its own operand's strides, dQ included; only the
        # log-sum-exp is dense.
        # STRIDE FIRST, INDEX SECOND, and on this kernel that is worth 109
        # us/layer in the production build: see "The masked tile's schedule is
        # chaotic" in `flash_attention_fwd_kernels`. This kernel has the same
        # peeled masked/unmasked tile structure as the fused forward and the same
        # sensitivity.
        var dq_base = dq_st.batch * bz + dq_st.head * by
        var dqss = dq_st.seq
        var g_base = bz * g_st.batch + by * g_st.head
        var q_base = bz * q_st.batch + by * q_st.head
        var o_base = bz * o_st.batch + by * o_st.head
        var k_base = bz * k_st.batch + by * k_st.head
        var v_base = bz * v_st.batch + by * v_st.head
        var gss = g_st.seq
        var qss = q_st.seq
        var oss = o_st.seq
        var kss = k_st.seq
        var vss = v_st.seq
        comptime if DENSE:
            var q_dense = bh * seq_q * head_dim
            g_base = q_dense
            q_base = q_dense
            o_base = q_dense
            k_base = bh * seq_kv * head_dim
            v_base = k_base
            gss = head_dim
            qss = head_dim
            oss = head_dim
            kss = head_dim
            vss = head_dim
            comptime if EXACT:
                gss = HD
                qss = HD
                oss = HD
                kss = HD
                vss = HD
        var lse_base = bh * seq_q

        # Heaviest query block first: under a causal mask the last block does
        # `grid_dim.x` times the work of the first, and blocks are dispatched
        # x-fastest, so reversing x starts the long pole at t=0.
        var q_block = (Int(grid_dim.x) - 1 - Int(block_idx.x)) * BM

        # Top-left causal alignment, matching PyTorch: the diagonal is q == kv
        # regardless of the length difference. This is 0 whenever
        # seq_q == seq_kv, so every square self-attention shape is
        # bit-identical to before.
        var delta = 0
        var q_hi = min(q_block + BM - 1, seq_q - 1)
        var lim_hi = seq_kv
        var lim_lo = seq_kv
        if causal != 0:
            lim_hi = min(seq_kv, q_hi + delta + 1)
            lim_lo = min(seq_kv, q_block + delta + 1)
        var n_tiles = ceildiv(max(lim_hi, 0), BN)
        var n_safe = max(lim_lo, 0) // BN

        # ---- Q and dO fragments, resident for the whole kv loop. Lane L holds
        # query `L%32`, head-dim `8s + 4*(L//32)`.  The `O` fragments live only
        # long enough to form `D`.
        var q_frag = stack_allocation[QT * KSTEPS * 4, dtype]()
        var do_frag = stack_allocation[QT * KSTEPS * 4, dtype]()
        var nlse = stack_allocation[QT, DType.float32]()
        var dval = stack_allocation[QT, DType.float32]()
        var qlim = stack_allocation[QT, DType.int32]()

        var q_row0 = q_block + wave * (32 * QT)
        comptime for qt in range(QT):
            var qg = q_row0 + qt * 32 + lo
            var live = qg < seq_q
            var qrow = q_base + qg * qss + 4 * hi
            var grow = g_base + qg * gss + 4 * hi
            var orow = o_base + qg * oss + 4 * hi
            var dsum = Float32(0.0)
            comptime for s in range(KSTEPS):
                var qv = SIMD[dtype, 4](0)
                var gv = SIMD[dtype, 4](0)
                var ov = SIMD[dtype, 4](0)
                var ok = live
                comptime if not EXACT:
                    ok = ok and (8 * s + 4 * hi < head_dim)
                if ok:
                    qv = query.load[width=4](qrow + 8 * s)
                    gv = grad_output.load[width=4](grow + 8 * s)
                    ov = out_fwd.load[width=4](orow + 8 * s)
                q_frag.store((qt * KSTEPS + s) * 4, qv)
                do_frag.store((qt * KSTEPS + s) * 4, gv)
                dsum += (
                    gv.cast[DType.float32]() * ov.cast[DType.float32]()
                ).reduce_add()
            # Lanes L and L^32 hold the two head-dim halves of one query.
            dsum += shuffle_xor(dsum, 32)
            var nl = NEG_ROW
            if live:
                nl = -lse[lse_base + qg] * LOG2E
            else:
                dsum = 0.0
            nlse[qt] = nl
            dval[qt] = dsum
            var lm = seq_kv
            if causal != 0:
                lm = min(seq_kv, qg + delta + 1)
            qlim[qt] = Int32(lm)

        var dq_acc = stack_allocation[DT * QT * 16, DType.float32]()
        comptime for i in range(DT * QT):
            dq_acc.store(i * 16, SIMD[DType.float32, 16](0))

        var sl = scale * LOG2E

        comptime KROW_STEP = THREADS // (HD // 8)
        comptime TKV_STEP = THREADS // HD
        var k_row0 = tid // (HD // 8)
        var k_col = (tid % (HD // 8)) * 8
        var t_col = tid % HD
        var t_kvg0 = tid // HD
        var k_ptr = key + (k_base + k_row0 * kss + k_col)
        var v_ptr = value + (v_base + k_row0 * vss + k_col)
        var k_tile_step = BN * kss
        var v_tile_step = BN * vss
        var k_lds0 = k_row0 * KPAD + k_col
        var t_lds0 = t_col * TPAD + t_kvg0 * 4
        var last_row = max(seq_kv - 1, 0)

        @always_inline
        @parameter
        def _tile[MASKED: Bool](t: Int):
            var kv0 = t * BN
            var full = True
            comptime if MASKED:
                full = kv0 + BN <= seq_kv

            # ---- Stage K and V row-major. Every global read of the tile is
            # ISSUED BEFORE any of them is consumed, so the tile pays one memory
            # round trip rather than one per load-then-write group.
            var kreg = stack_allocation[KITERS * 8, dtype]()
            var vreg = stack_allocation[KITERS * 8, dtype]()
            if full:
                comptime for ci in range(KITERS):
                    var kvals = SIMD[dtype, 8](0)
                    var vvals = SIMD[dtype, 8](0)
                    comptime if EXACT:
                        kvals = k_ptr.load[width=8](ci * KROW_STEP * kss)
                        vvals = v_ptr.load[width=8](ci * KROW_STEP * vss)
                    else:
                        if k_col < head_dim:
                            kvals = k_ptr.load[width=8](ci * KROW_STEP * kss)
                            vvals = v_ptr.load[width=8](ci * KROW_STEP * vss)
                    kreg.store(ci * 8, kvals)
                    vreg.store(ci * 8, vvals)
            else:
                comptime for ci in range(KITERS):
                    var row = min(kv0 + k_row0 + ci * KROW_STEP, last_row)
                    var kvals = SIMD[dtype, 8](0)
                    var vvals = SIMD[dtype, 8](0)
                    if EXACT or k_col < head_dim:
                        kvals = key.load[width=8](k_base + row * kss + k_col)
                        vvals = value.load[width=8](v_base + row * vss + k_col)
                    kreg.store(ci * 8, kvals)
                    vreg.store(ci * 8, vvals)
            comptime for ci in range(KITERS):
                k_smem.store(
                    k_lds0 + ci * KROW_STEP * KPAD, kreg.load[width=8](ci * 8)
                )
                v_smem.store(
                    k_lds0 + ci * KROW_STEP * KPAD, vreg.load[width=8](ci * 8)
                )

            k_ptr += k_tile_step
            v_ptr += v_tile_step
            barrier()

            # ---- K^T comes out of the row-major K tile, not out of a second
            # pass over global memory: four `ds_read_u16` at one head-dim column
            # are 128 contiguous bytes per wavefront and cost no memory round
            # trip, where the transposing global loader cost four more of them.
            comptime for ti in range(TITERS):
                var kv0l = (t_kvg0 + ti * TKV_STEP) * 4
                var tv = SIMD[dtype, 4](0)
                comptime for j in range(4):
                    tv[j] = k_smem[(kv0l + j) * KPAD + t_col]
                kt_smem.store(t_lds0 + ti * TKV_STEP * 4, tv)
            barrier()

            llvm_intrinsic["llvm.amdgcn.iglp.opt", NoneType](Int32(IGLP))
            comptime for kt in range(KVT):
                var abase = (kt * 32 + lo) * KPAD + 4 * hi
                comptime for qt in range(QT):
                    # ---- GEMM 1: S^T[kv, q] = sum_d K[kv, d] * Q[q, d]
                    var s_acc = SIMD[DType.float32, 16](0)
                    comptime for s in range(KSTEPS):
                        mma(
                            s_acc,
                            k_smem.load[width=4](abase + 8 * s),
                            q_frag.load[width=4]((qt * KSTEPS + s) * 4),
                            s_acc,
                        )
                    # ---- GEMM 2: dP^T[kv, q] = sum_d V[kv, d] * dO[q, d]
                    var p_acc = SIMD[DType.float32, 16](0)
                    comptime for s in range(KSTEPS):
                        mma(
                            p_acc,
                            v_smem.load[width=4](abase + 8 * s),
                            do_frag.load[width=4]((qt * KSTEPS + s) * 4),
                            p_acc,
                        )

                    comptime if MASKED:
                        # `kv` of element p is `kv0 + 32*kt + 4*hi + ACC_ROWS[p]`
                        # and the causal bound is one scalar per lane.
                        var kvv = (
                            SIMD[DType.int32, 16](kv0 + kt * 32 + 4 * hi)
                            + ACC_ROWS
                        )
                        s_acc = kvv.lt(SIMD[DType.int32, 16](qlim[qt])).select(
                            s_acc, NEG_BIG
                        )
                    var pv = exp2(
                        s_acc.fma(
                            SIMD[DType.float32, 16](sl),
                            SIMD[DType.float32, 16](nlse[qt]),
                        )
                    )
                    var dsv = (
                        pv
                        * (p_acc - SIMD[DType.float32, 16](dval[qt]))
                        * SIMD[DType.float32, 16](scale)
                    )
                    var ds_frag = _pack_half[dtype, 16](dsv)

                    # ---- GEMM 3: dQ^T[d, q] += sum_kv K^T[d, kv] * dS^T[kv, q]
                    comptime for g in range(4):
                        var b = ds_frag.slice[4, offset=g * 4]()
                        comptime for dt in range(DT):
                            var acc = dq_acc.load[width=16]((dt * QT + qt) * 16)
                            mma(
                                acc,
                                kt_smem.load[width=4](
                                    (dt * 32 + lo) * TPAD
                                    + kt * 32
                                    + 8 * g
                                    + 4 * hi
                                ),
                                b,
                                acc,
                            )
                            dq_acc.store((dt * QT + qt) * 16, acc)

            barrier()

        var t = 0
        comptime if PEEL:
            while t < n_safe:
                _tile[False](t)
                t += 1
        while t < n_tiles:
            _tile[True](t)
            t += 1

        # ---- Epilogue. Read and round every accumulator once, in straight-line
        # code, fenced on both sides: see the module docstring on the gfx942
        # MFMA-destination hazard.
        _sched_fence()
        var packed = stack_allocation[DT * QT * 16, dtype]()
        comptime for i in range(DT * QT):
            packed.store(
                i * 16, _pack_half[dtype, 16](dq_acc.load[width=16](i * 16))
            )
        _sched_fence()

        # dQ^T lives with one query per lane and 16 scattered head-dim rows, so
        # it is staged through LDS one 32-column slab at a time and read back in
        # row-major order for coalesced global stores.
        comptime for dt in range(DT):
            barrier()
            comptime for qt in range(QT):
                var row = wave * (32 * QT) + qt * 32 + lo
                var ob = packed.load[width=16]((dt * QT + qt) * 16)
                comptime for g in range(4):
                    smem.store(
                        row * OPAD + 8 * g + 4 * hi,
                        ob.slice[4, offset=g * 4](),
                    )
            barrier()
            comptime for pz in range(OPASSES):
                var r = pz * 64 + tid // 4
                var cc = (tid % 4) * 8
                var qg = q_block + r
                var dcol = dt * 32 + cc
                var store = qg < seq_q
                comptime if not EXACT:
                    store = store and dcol < head_dim
                if store:
                    dq.store(
                        dq_base + qg * dqss + dcol,
                        smem.load[width=8](r * OPAD + cc),
                    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS)),
    `rocdl.waves_per_eu`=SIMDSize(WAVES_PER_EU),
)
@__name(t"fa_bwd_dkv_mfma_{dtype}_h{HD}_m{BM}_k{KT}_x{EXACT}_d{DENSE}")
def _bwd_dkv_mfma[
    dtype: DType,
    HD: Int,
    BM: Int,
    KT: Int,
    EXACT: Bool,
    DENSE: Bool,
    WAVES_PER_EU: Int,
    IGLP: Int,
](
    dk: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    out_fwd: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    g_st: RowStrides,
    q_st: RowStrides,
    k_st: RowStrides,
    v_st: RowStrides,
    o_st: RowStrides,
    dk_st: RowStrides,
    dv_st: RowStrides,
    seq_q_arg: Int64,
    seq_kv_arg: Int64,
    head_dim_arg: Int64,
    scale: Float32,
    causal_arg: Int64,
):
    """dK and dV: one workgroup owns `4 * 32 * KT` key rows and walks q tiles.

    The score tile runs the other way round from the dQ kernel (`S[q, kv]`, so a
    lane owns one KEY), because the two output GEMMs contract over q and an MFMA
    operand's `k` axis must be the accumulator's row axis.

    `DENSE` asserts the five read operands are row-major `[b, h, seq, head_dim]`
    and restores the compile-time row offsets in the Q/dO/O loader; see the same
    parameter on the forward kernel for the measurement behind it. It says
    nothing about `dk_st` and `dv_st`: the epilogue's store address was already a
    runtime multiply by `head_dim`.
    """
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var seq_q = Int(seq_q_arg)
    var seq_kv = Int(seq_kv_arg)
    var head_dim = Int(head_dim_arg)
    var causal = Int(causal_arg)
    comptime NW = 4
    comptime BN = NW * 32 * KT
    comptime MT = BM // 32  # query MFMA tiles per score tile
    comptime KSTEPS = HD // 8  # k-steps of the score / dP GEMMs (over d)
    comptime DT = HD // 32  # head-dim MFMA tiles of dK and dV
    comptime QPAD = HD + 4  # LDS row length of the row-major Q and dO tiles
    comptime TPAD = BM + 4  # LDS row length of the Q^T and dO^T tiles
    comptime OPAD = 36  # LDS row length of the epilogue staging tile
    comptime QITERS = BM * HD // 8 // THREADS
    comptime TITERS = HD * (BM // 4) // THREADS
    comptime RSTEP = THREADS // (HD // 8)
    comptime LPR = HD // 8  # lanes that share one row in the row-major loader
    comptime OPASSES = BN // 64
    comptime LDS_ELEMS = max(2 * BM * QPAD + 2 * HD * TPAD, BN * OPAD)

    comptime if is_amd_gpu():
        var smem = stack_allocation[
            LDS_ELEMS, dtype, address_space=AddressSpace.SHARED, alignment=16
        ]()
        var q_smem = smem
        var do_smem = smem + BM * QPAD
        var qt_smem = smem + 2 * BM * QPAD
        var dot_smem = smem + 2 * BM * QPAD + HD * TPAD
        # `[-lse * log2(e), D]` per query row of the tile, read back as one
        # broadcast `ds_read_b64` per accumulator row.
        var aux = stack_allocation[
            2 * BM, DType.float32, address_space=AddressSpace.SHARED
        ]()

        var tid = Int(thread_idx.x)
        var wave = tid // 64
        var lane = tid % 64
        var lo = lane % 32
        var hi = lane // 32

        var heads = Int(grid_dim.y)
        var bz = Int(block_idx.z)
        var by = Int(block_idx.y)
        var bh = bz * heads + by
        # Every base follows its own operand's strides, dK and dV included;
        # only the log-sum-exp is dense.
        var dk_base = dk_st.batch * bz + dk_st.head * by
        var dv_base = dv_st.batch * bz + dv_st.head * by
        var dkss = dk_st.seq
        var dvss = dv_st.seq
        var g_base = bz * g_st.batch + by * g_st.head
        var q_base = bz * q_st.batch + by * q_st.head
        var o_base = bz * o_st.batch + by * o_st.head
        var k_base = bz * k_st.batch + by * k_st.head
        var v_base = bz * v_st.batch + by * v_st.head
        var gss = g_st.seq
        var qss = q_st.seq
        var oss = o_st.seq
        var kss = k_st.seq
        var vss = v_st.seq
        comptime if DENSE:
            var kv_dense = bh * seq_kv * head_dim
            g_base = bh * seq_q * head_dim
            q_base = g_base
            o_base = g_base
            k_base = kv_dense
            v_base = kv_dense
            gss = head_dim
            qss = head_dim
            oss = head_dim
            kss = head_dim
            vss = head_dim
            comptime if EXACT:
                gss = HD
                qss = HD
                oss = HD
                kss = HD
                vss = HD
        var lse_base = bh * seq_q

        # Natural x order already puts the heaviest block first: under a causal
        # mask kv tile 0 is attended by every query.
        var kv_block = Int(block_idx.x) * BN
        # Top-left causal alignment, matching PyTorch: the diagonal is q == kv
        # regardless of the length difference. This is 0 whenever
        # seq_q == seq_kv, so every square self-attention shape is
        # bit-identical to before.
        var delta = 0

        # ---- K and V fragments, resident. Lane L holds key
        # `kv_block + 32*(wave*KT + kt) + L%32`, head-dim `8s + 4*(L//32)`.
        var k_frag = stack_allocation[KT * KSTEPS * 4, dtype]()
        var v_frag = stack_allocation[KT * KSTEPS * 4, dtype]()
        var kv_lane = kv_block + wave * (32 * KT) + lo
        comptime for kt in range(KT):
            var kvg = kv_lane + kt * 32
            var live = kvg < seq_kv
            var krow = k_base + kvg * kss + 4 * hi
            var vrow = v_base + kvg * vss + 4 * hi
            comptime for s in range(KSTEPS):
                var kv4 = SIMD[dtype, 4](0)
                var vv4 = SIMD[dtype, 4](0)
                var ok = live
                comptime if not EXACT:
                    ok = ok and (8 * s + 4 * hi < head_dim)
                if ok:
                    kv4 = key.load[width=4](krow + 8 * s)
                    vv4 = value.load[width=4](vrow + 8 * s)
                k_frag.store((kt * KSTEPS + s) * 4, kv4)
                v_frag.store((kt * KSTEPS + s) * 4, vv4)

        var dk_acc = stack_allocation[DT * KT * 16, DType.float32]()
        var dv_acc = stack_allocation[DT * KT * 16, DType.float32]()
        comptime for i in range(DT * KT):
            dk_acc.store(i * 16, SIMD[DType.float32, 16](0))
            dv_acc.store(i * 16, SIMD[DType.float32, 16](0))

        var sl = scale * LOG2E

        # ---- Which query tiles touch this key block.
        var n_tiles = ceildiv(seq_q, BM)
        var t_first = 0
        var t_safe = 0
        if causal != 0:
            t_first = max(0, kv_block - delta) // BM
            t_safe = ceildiv(max(0, kv_block + BN - 1 - delta), BM)
        var t_full = seq_q // BM
        var ta = min(t_first, n_tiles)
        var tb = min(max(t_safe, ta), n_tiles)
        var tc = min(max(t_full, tb), n_tiles)

        comptime TQ_STEP = THREADS // HD
        var r_row0 = tid // (HD // 8)
        var r_col = (tid % (HD // 8)) * 8
        var t_col = tid % HD
        var t_qg0 = tid // HD
        var r_lds0 = r_row0 * QPAD + r_col
        var t_lds0 = t_col * TPAD + t_qg0 * 4
        var last_q = max(seq_q - 1, 0)
        var row_lane = tid % LPR

        @always_inline
        @parameter
        def _tile[MASKED: Bool](t: Int):
            var q0 = t * BM
            var full = True
            comptime if MASKED:
                full = q0 + BM <= seq_q

            # ---- Every global read of the tile is ISSUED BEFORE any of them is
            # consumed. With one workgroup per CU and one wave per SIMD there is
            # nothing else to hide a memory round trip behind, so a loader that
            # interleaves loads with the LDS writes that consume them pays the
            # full latency once per group; batching pays it once per tile.
            var qreg = stack_allocation[QITERS * 8, dtype]()
            var greg = stack_allocation[QITERS * 8, dtype]()
            var oreg = stack_allocation[QITERS * 8, dtype]()
            var lval = NEG_ROW
            comptime for ci in range(QITERS):
                var grow = q0 + r_row0 + ci * RSTEP
                if not full:
                    grow = min(grow, last_q)
                var qv = SIMD[dtype, 8](0)
                var gv = SIMD[dtype, 8](0)
                var ov = SIMD[dtype, 8](0)
                var ok = True
                comptime if not EXACT:
                    ok = r_col < head_dim
                if ok:
                    qv = query.load[width=8](q_base + grow * qss + r_col)
                    gv = grad_output.load[width=8](g_base + grow * gss + r_col)
                    ov = out_fwd.load[width=8](o_base + grow * oss + r_col)
                qreg.store(ci * 8, qv)
                greg.store(ci * 8, gv)
                oreg.store(ci * 8, ov)
            if tid < BM:
                var qg = q0 + tid
                if qg < seq_q:
                    lval = -lse[lse_base + qg] * LOG2E

            # ---- Stage Q and dO row-major, and the per-row `[-lse*log2e, D]`.
            comptime for ci in range(QITERS):
                var lrow = r_row0 + ci * RSTEP
                var gv = greg.load[width=8](ci * 8)
                q_smem.store(
                    r_lds0 + ci * RSTEP * QPAD, qreg.load[width=8](ci * 8)
                )
                do_smem.store(r_lds0 + ci * RSTEP * QPAD, gv)
                var part = (
                    gv.cast[DType.float32]()
                    * oreg.load[width=8](ci * 8).cast[DType.float32]()
                ).reduce_add()
                part += shuffle_xor(part, 1)
                part += shuffle_xor(part, 2)
                part += shuffle_xor(part, 4)
                comptime if LPR > 8:
                    part += shuffle_xor(part, 8)
                comptime if LPR > 16:
                    part += shuffle_xor(part, 16)
                if row_lane == 0:
                    aux[2 * lrow + 1] = part
            if tid < BM:
                aux[2 * tid] = lval
            barrier()

            # ---- Q^T and dO^T come out of the row-major tiles, not out of a
            # second pass over global memory: four `ds_read_u16` at one head-dim
            # column are 128 contiguous bytes per wavefront and cost no memory
            # round trip at all, where the transposing global loader cost four
            # more of them per tile.
            comptime for ti in range(TITERS):
                var q0l = (t_qg0 + ti * TQ_STEP) * 4
                var qv4 = SIMD[dtype, 4](0)
                var gv4 = SIMD[dtype, 4](0)
                comptime for j in range(4):
                    qv4[j] = q_smem[(q0l + j) * QPAD + t_col]
                    gv4[j] = do_smem[(q0l + j) * QPAD + t_col]
                qt_smem.store(t_lds0 + ti * TQ_STEP * 4, qv4)
                dot_smem.store(t_lds0 + ti * TQ_STEP * 4, gv4)
            barrier()

            llvm_intrinsic["llvm.amdgcn.iglp.opt", NoneType](Int32(IGLP))
            comptime for mt in range(MT):
                var abase = (mt * 32 + lo) * QPAD + 4 * hi
                # `-lse*log2e` and `D` for the sixteen accumulator ROWS.
                var nl_v = SIMD[DType.float32, 16](0)
                var dd_v = SIMD[DType.float32, 16](0)
                comptime for p in range(16):
                    comptime dr = 8 * (p // 4) + (p % 4)
                    var pair = aux.load[width=2](2 * (mt * 32 + dr + 4 * hi))
                    nl_v[p] = pair[0]
                    dd_v[p] = pair[1]

                comptime for kt in range(KT):
                    # ---- GEMM 1: S[q, kv] = sum_d Q[q, d] * K[kv, d]
                    var s_acc = SIMD[DType.float32, 16](0)
                    comptime for s in range(KSTEPS):
                        mma(
                            s_acc,
                            q_smem.load[width=4](abase + 8 * s),
                            k_frag.load[width=4]((kt * KSTEPS + s) * 4),
                            s_acc,
                        )
                    # ---- GEMM 2: dP[q, kv] = sum_d dO[q, d] * V[kv, d]
                    var p_acc = SIMD[DType.float32, 16](0)
                    comptime for s in range(KSTEPS):
                        mma(
                            p_acc,
                            do_smem.load[width=4](abase + 8 * s),
                            v_frag.load[width=4]((kt * KSTEPS + s) * 4),
                            p_acc,
                        )

                    comptime if MASKED:
                        # Key `kv` is attended by query `q` iff
                        # `q >= kv - delta`; `kv` is one scalar per lane and the
                        # sixteen `q` are the accumulator rows.
                        var qv = (
                            SIMD[DType.int32, 16](q0 + mt * 32 + 4 * hi)
                            + ACC_ROWS
                        )
                        var need = SIMD[DType.int32, 16](
                            kv_lane + kt * 32 - delta
                        )
                        s_acc = qv.ge(need).select(s_acc, NEG_BIG)
                    var pv = exp2(s_acc.fma(SIMD[DType.float32, 16](sl), nl_v))
                    var dsv = (
                        pv * (p_acc - dd_v) * SIMD[DType.float32, 16](scale)
                    )
                    var p_frag = _pack_half[dtype, 16](pv)
                    var ds_frag = _pack_half[dtype, 16](dsv)

                    # ---- GEMM 3/4: dV^T[d, kv] += dO^T[d, q] * P[q, kv]
                    #                dK^T[d, kv] += Q^T[d, q] * dS[q, kv]
                    comptime for g in range(4):
                        var bp = p_frag.slice[4, offset=g * 4]()
                        var bd = ds_frag.slice[4, offset=g * 4]()
                        comptime for dt in range(DT):
                            var toff = (
                                (dt * 32 + lo) * TPAD + mt * 32 + 8 * g + 4 * hi
                            )
                            var va = dv_acc.load[width=16]((dt * KT + kt) * 16)
                            mma(va, dot_smem.load[width=4](toff), bp, va)
                            dv_acc.store((dt * KT + kt) * 16, va)
                            var ka = dk_acc.load[width=16]((dt * KT + kt) * 16)
                            mma(ka, qt_smem.load[width=4](toff), bd, ka)
                            dk_acc.store((dt * KT + kt) * 16, ka)

            barrier()

        var t = ta
        while t < tb:
            _tile[True](t)
            t += 1
        while t < tc:
            _tile[False](t)
            t += 1
        while t < n_tiles:
            _tile[True](t)
            t += 1

        # ---- Epilogue: read and round every accumulator once, fenced.
        _sched_fence()
        var pk_v = stack_allocation[DT * KT * 16, dtype]()
        var pk_k = stack_allocation[DT * KT * 16, dtype]()
        comptime for i in range(DT * KT):
            pk_v.store(
                i * 16, _pack_half[dtype, 16](dv_acc.load[width=16](i * 16))
            )
            pk_k.store(
                i * 16, _pack_half[dtype, 16](dk_acc.load[width=16](i * 16))
            )
        _sched_fence()

        comptime for pass_id in range(2):
            comptime for dt in range(DT):
                barrier()
                comptime for kt in range(KT):
                    var row = wave * (32 * KT) + kt * 32 + lo
                    var ob = pk_v.load[width=16]((dt * KT + kt) * 16)
                    comptime if pass_id == 1:
                        ob = pk_k.load[width=16]((dt * KT + kt) * 16)
                    comptime for g in range(4):
                        smem.store(
                            row * OPAD + 8 * g + 4 * hi,
                            ob.slice[4, offset=g * 4](),
                        )
                barrier()
                comptime for pz in range(OPASSES):
                    var r = pz * 64 + tid // 4
                    var cc = (tid % 4) * 8
                    var kvg = kv_block + r
                    var dcol = dt * 32 + cc
                    var store = kvg < seq_kv
                    comptime if not EXACT:
                        store = store and dcol < head_dim
                    if store:
                        var vals = smem.load[width=8](r * OPAD + cc)
                        comptime if pass_id == 0:
                            dv.store(dv_base + kvg * dvss + dcol, vals)
                        else:
                            dk.store(dk_base + kvg * dkss + dcol, vals)


def _enqueue_mfma_pair[
    dtype: DType,
    HD: Int,
    BN: Int,
    QT: Int,
    BM: Int,
    KT: Int,
    EXACT: Bool,
    WPE_Q: Int = 1,
    WPE_KV: Int = 1,
    IGLP: Int = 1,
    PEEL: Bool = True,
](
    dq: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dk: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    out_fwd: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    softmax_lse: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    g_st: RowStrides,
    q_st: RowStrides,
    k_st: RowStrides,
    v_st: RowStrides,
    o_st: RowStrides,
    dq_st: RowStrides,
    dk_st: RowStrides,
    dv_st: RowStrides,
    batch: Int,
    heads: Int,
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    causal: Int,
    ctx: DeviceContext,
) raises:
    # `DENSE` is a property of the READ operands only; see `_bwd_dq_mfma`.
    var dense = (
        _is_dense(g_st, heads, seq_q, head_dim)
        and _is_dense(q_st, heads, seq_q, head_dim)
        and _is_dense(o_st, heads, seq_q, head_dim)
        and _is_dense(k_st, heads, seq_kv, head_dim)
        and _is_dense(v_st, heads, seq_kv, head_dim)
    )
    if dense:
        ctx.enqueue_function[
            _bwd_dkv_mfma[dtype, HD, BM, KT, EXACT, True, WPE_KV, IGLP]
        ](
            dk,
            dv,
            grad_output,
            query,
            key,
            value,
            out_fwd,
            softmax_lse,
            g_st,
            q_st,
            k_st,
            v_st,
            o_st,
            dk_st,
            dv_st,
            Int64(seq_q),
            Int64(seq_kv),
            Int64(head_dim),
            scale,
            Int64(causal),
            grid_dim=(ceildiv(seq_kv, 4 * 32 * KT), heads, batch),
            block_dim=(THREADS,),
        )
        ctx.enqueue_function[
            _bwd_dq_mfma[dtype, HD, BN, QT, EXACT, True, WPE_Q, IGLP, PEEL]
        ](
            dq,
            grad_output,
            query,
            key,
            value,
            out_fwd,
            softmax_lse,
            g_st,
            q_st,
            k_st,
            v_st,
            o_st,
            dq_st,
            Int64(seq_q),
            Int64(seq_kv),
            Int64(head_dim),
            scale,
            Int64(causal),
            grid_dim=(ceildiv(seq_q, 4 * 32 * QT), heads, batch),
            block_dim=(THREADS,),
        )
        return
    ctx.enqueue_function[
        _bwd_dkv_mfma[dtype, HD, BM, KT, EXACT, False, WPE_KV, IGLP]
    ](
        dk,
        dv,
        grad_output,
        query,
        key,
        value,
        out_fwd,
        softmax_lse,
        g_st,
        q_st,
        k_st,
        v_st,
        o_st,
        dk_st,
        dv_st,
        Int64(seq_q),
        Int64(seq_kv),
        Int64(head_dim),
        scale,
        Int64(causal),
        grid_dim=(ceildiv(seq_kv, 4 * 32 * KT), heads, batch),
        block_dim=(THREADS,),
    )
    ctx.enqueue_function[
        _bwd_dq_mfma[dtype, HD, BN, QT, EXACT, False, WPE_Q, IGLP, PEEL]
    ](
        dq,
        grad_output,
        query,
        key,
        value,
        out_fwd,
        softmax_lse,
        g_st,
        q_st,
        k_st,
        v_st,
        o_st,
        dq_st,
        Int64(seq_q),
        Int64(seq_kv),
        Int64(head_dim),
        scale,
        Int64(causal),
        grid_dim=(ceildiv(seq_q, 4 * 32 * QT), heads, batch),
        block_dim=(THREADS,),
    )


def enqueue_flash_attention_bwd[
    dtype: DType
](
    dq: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dk: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    grad_output: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    out_fwd: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    softmax_lse: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    g_st: RowStrides,
    q_st: RowStrides,
    k_st: RowStrides,
    v_st: RowStrides,
    o_st: RowStrides,
    dq_st: RowStrides,
    dk_st: RowStrides,
    dv_st: RowStrides,
    batch: Int,
    heads: Int,
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    is_causal: Bool,
    ctx: DeviceContext,
) raises:
    """THE FROZEN ENTRY POINT.

    Shapes:
      `dq`, `query`, `grad_output`, `out_fwd`  `[batch, heads, seq_q, head_dim]`
      `dk`, `dv`, `key`, `value`              `[batch, heads, seq_kv, head_dim]`
      `softmax_lse`                            `[batch, heads, seq_q]`, FP32

    The five READ operands are addressed through `g_st`, `q_st`, `k_st`, `v_st`
    and `o_st`, the element strides of their batch, head and seq axes, in that
    operand order, and the three WRITTEN ones through `dq_st`, `dk_st` and
    `dv_st`. Every head_dim axis must have stride 1 -- it is the axis every
    vectorized load and every vectorized store runs along -- and a caller holding
    a layout where it is not must decline rather than pass one here.
    `softmax_lse` is dense.

    `dense_strides(heads, seq, head_dim)` builds the triple for a plain
    row-major `[batch, heads, seq, head_dim]` tensor; `bthd_strides` builds the
    one for `[batch, heads, seq, head_dim]` stored `[batch, seq, heads,
    head_dim]`, the layout that makes the caller's `transpose(1, 2)` free.

    `softmax_lse[i]` is the natural log of the sum of `exp(S[i, :] - max)` plus
    that max, i.e. the row log-sum-exp, so `P[i, j] = exp(S[i, j] - lse[i])`.

    `dq`, `dk` and `dv` arrive zeroed. Every extent AND every stride is a
    runtime value.
    """
    if batch <= 0 or heads <= 0 or seq_q <= 0 or seq_kv <= 0 or head_dim <= 0:
        return
    if head_dim > MAX_HEAD_DIM:
        raise Error(
            "flash attention backward: head_dim ",
            head_dim,
            " exceeds the staged maximum ",
            MAX_HEAD_DIM,
            (
                "; a replacement must raise the staging or route this head_dim"
                " to a path that handles it"
            ),
        )
    var causal = 1 if is_causal else 0

    # The MFMA path needs a half-float MFMA and 8-element vector loads along
    # head_dim. Everything else falls through to the general baselines.
    comptime if dtype.is_half_float():
        if ctx.api() == "hip" and head_dim % 8 == 0 and head_dim <= 128:
            if head_dim == 64:
                _enqueue_mfma_pair[dtype, 64, 64, 1, 64, 1, True](
                    dq,
                    dk,
                    dv,
                    grad_output,
                    query,
                    key,
                    value,
                    out_fwd,
                    softmax_lse,
                    g_st,
                    q_st,
                    k_st,
                    v_st,
                    o_st,
                    dq_st,
                    dk_st,
                    dv_st,
                    batch,
                    heads,
                    seq_q,
                    seq_kv,
                    head_dim,
                    scale,
                    causal,
                    ctx,
                )
                return
            if head_dim < 64:
                _enqueue_mfma_pair[dtype, 64, 64, 1, 64, 1, False](
                    dq,
                    dk,
                    dv,
                    grad_output,
                    query,
                    key,
                    value,
                    out_fwd,
                    softmax_lse,
                    g_st,
                    q_st,
                    k_st,
                    v_st,
                    o_st,
                    dq_st,
                    dk_st,
                    dv_st,
                    batch,
                    heads,
                    seq_q,
                    seq_kv,
                    head_dim,
                    scale,
                    causal,
                    ctx,
                )
                return
            if head_dim == 128:
                _enqueue_mfma_pair[dtype, 128, 64, 1, 32, 1, True](
                    dq,
                    dk,
                    dv,
                    grad_output,
                    query,
                    key,
                    value,
                    out_fwd,
                    softmax_lse,
                    g_st,
                    q_st,
                    k_st,
                    v_st,
                    o_st,
                    dq_st,
                    dk_st,
                    dv_st,
                    batch,
                    heads,
                    seq_q,
                    seq_kv,
                    head_dim,
                    scale,
                    causal,
                    ctx,
                )
                return
            _enqueue_mfma_pair[dtype, 128, 64, 1, 32, 1, False](
                dq,
                dk,
                dv,
                grad_output,
                query,
                key,
                value,
                out_fwd,
                softmax_lse,
                g_st,
                q_st,
                k_st,
                v_st,
                o_st,
                dq_st,
                dk_st,
                dv_st,
                batch,
                heads,
                seq_q,
                seq_kv,
                head_dim,
                scale,
                causal,
                ctx,
            )
            return

    ctx.enqueue_function[_bwd_dq_baseline[dtype]](
        dq,
        grad_output,
        query,
        key,
        value,
        out_fwd,
        softmax_lse,
        g_st,
        q_st,
        k_st,
        v_st,
        o_st,
        dq_st,
        Int64(seq_q),
        Int64(seq_kv),
        Int64(head_dim),
        scale,
        Int64(causal),
        grid_dim=(seq_q, heads, batch),
        block_dim=(THREADS,),
    )
    ctx.enqueue_function[_bwd_dkv_baseline[dtype]](
        dk,
        dv,
        grad_output,
        query,
        key,
        value,
        out_fwd,
        softmax_lse,
        g_st,
        q_st,
        k_st,
        v_st,
        o_st,
        dk_st,
        dv_st,
        Int64(seq_q),
        Int64(seq_kv),
        Int64(head_dim),
        scale,
        Int64(causal),
        grid_dim=(seq_kv, heads, batch),
        block_dim=(THREADS,),
    )
