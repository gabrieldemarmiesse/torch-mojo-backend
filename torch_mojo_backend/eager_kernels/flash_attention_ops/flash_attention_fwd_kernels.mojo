"""Fused causal flash-attention forward for gfx942. THE FILE TO OPTIMIZE.

The contract below is frozen; `harness/nanogpt_train/bench_flash_attention.mojo`
calls exactly this entry point and defines acceptance.

`enqueue_flash_attention_fwd` dispatches between two implementations:

* `_fa_mfma`, an MFMA flash-attention kernel for CDNA (gfx942), selected when
  the device is HIP and `head_dim % 8 == 0`;
* `_flash_attention_fwd_baseline`, the original one-block-per-query-row kernel,
  which stays as the general correct path for every other shape and device.

Invariants any replacement must keep:

* `batch`, `heads`, `seq_q`, `seq_kv` and `head_dim` are RUNTIME values, and so
  is every STRIDE. Tile shapes, pipeline depth and head-dim regimes may be
  compile-time as long as a runtime dispatch picks between them and every
  runtime head_dim is handled.
* Q, K and V are read through a `RowStrides` triple rather than assumed dense,
  because the real caller hands over `x.view(B, T, H, D).transpose(1, 2)` and
  materializing that cost 1.83 ms/step of pure copy. `head_dim`'s stride is NOT
  carried: it must be 1, since that is the axis the 8-element global loads and
  the LDS tile fills run along. The log-sum-exp is freshly allocated by the
  caller and stays dense.
* `output` is WRITTEN through its own `RowStrides` triple for the same reason.
  PyTorch's own flash attention returns a tensor shaped `[B, H, T, D]` but
  stored `[B, T, H, D]`, so the universal re-assembly idiom
  `y.transpose(1, 2).contiguous().view(B, T, C)` is a free no-op; returning
  dense BHSD instead cost nanoGPT 1.84 ms/step of gather (journal D11). The
  head_dim axis must still have stride 1 -- the epilogue store is an 8-element
  vector along it.
* No allocation, no host transfer, no synchronization: enqueue on the caller's
  `DeviceContext` and return.
* Never write outside `output`. Q, K and V are read-only.
* Numerics: accumulate in FP32, store once in `dtype`. Seed a running max with
  `Float32.MIN_FINITE`, never `Float32.MIN` -- the latter is -inf, and an idle
  lane then computes `exp(-inf - -inf)` = nan, which poisons the whole row
  through the block reduction. That defect has been shipped twice in this
  repository; the harness's NaN counter exists because of it.
* `is_causal` masks strictly above the diagonal aligned to the BOTTOM right, so
  query row `q` attends to key indices `0 ..= q`, whatever `seq_kv` is. That is
  PyTorch's convention for `is_causal=True` and it matters when
  `seq_kv != seq_q`.

## Why the score tile is computed transposed

Everything about the MFMA kernel follows from one decision: the first GEMM
produces `S^T = K @ Q^T` (kv along M, queries along N) rather than `S = Q @ K^T`.

The `v_mfma_f32_32x32x8bf16_1k` accumulator puts, in lane `L`, register `p`,
the element at row `8*(p//4) + (p%4) + 4*(L//32)`, column `L%32`. Its B operand
wants, in lane `L`, `B[4*(L//32) + j, L%32]` for `j in 0..3`. So for the
transposed tile:

* a lane owns ONE query (column `L%32`) and sixteen kv rows. The online-softmax
  state is a single scalar per lane, the causal bound is a single scalar per
  lane, and the row reduction is `reduce_max` over 16 registers plus one
  exchange with lane `L^32`. No LDS reduction, no block reduction.
* accumulator register group `g` (registers `4g..4g+3`) holds rows
  `8g + 4*(L//32) + 0..3` of column `L%32`, which is EXACTLY the B operand of
  the next MFMA for the K-strip `[8g, 8g+8)`. `P^T` therefore feeds the second
  GEMM straight out of the accumulator registers with a bf16 cast and a slice:
  no shuffle, no LDS round trip. On CDNA4 (`MMA_K=16`) the same chain needs a
  key permutation that cancels against `ds_read_b64_tr_b16`; at `MMA_K=8` the
  permutation is the identity.

The price is that the second GEMM, `O^T = V^T @ P^T`, needs V with kv
contiguous, because an MFMA operand always holds four elements adjacent along
K. `ds_read_b64_tr_b16` is CDNA4-only, so V is transposed on the way into LDS
instead: each thread reads four kv values at one head-dim column (a fully
coalesced 2-bytes-per-lane global read, 128 contiguous bytes per wavefront) and
writes them as one 8-byte LDS store into a `[head_dim][BN]` tile.

## LDS layout

Both LDS tiles are row-major with the row length padded to `X + 4` elements.
That is not arbitrary. An MFMA fragment read is `ds_read_b64` at
`(L%32)*PAD + c`, i.e. dword `(L%32)*(PAD/2) + c/2`; the bank is that value mod
32. `PAD/2 == 2 (mod 4)` makes `{(L%32)*(PAD/2) mod 32}` sixteen distinct even
banks over lanes 0..15, and the second dword of each `b64` fills the odd banks,
so a 16-lane LDS cycle covers all 32 banks exactly once. `X + 4` with `X` a
multiple of 8 always satisfies it, and keeps rows 8-byte aligned.
`SQ_LDS_BANK_CONFLICT` measures 0.

## What binds this kernel, measured

On the nanogpt case (48 x 12 x 1024, head_dim 64, causal) the kernel measures
511.7 us/layer against PyTorch-ROCm's fused `attn_fwd` at 679.80.  An MFMA-only
microbenchmark on this device sustains 810 TFLOP/s -- with ONE independent
accumulator chain it still reaches 662 -- so the 77.3 GFLOP of this case has a
realistic floor near 97 us and neither matrix throughput nor accumulator depth
is the constraint.  Removing one stage at a time from the tile loop gives:

    MFMA + bf16 pack + barriers, no memory      190 us
    + the LDS fragment reads                    281 us   (+90)
    + the global K/V loads                      354 us   (+76)
    + the online softmax                        512 us   (+187, of which the
                                                          causal mask 70 and
                                                          exp2 65)

The stages ADD; nothing overlaps.  That is a latency profile, not a throughput
one, and it is why the usual levers did nothing here: occupancy is inert
(waves-per-eu 1/2/3/4 measure 530.8/535.5/534.1/531.6), a register prefetch of
the next K/V tile made it worse (576 vs 550), and hoisting the fragment reads
out of the MFMA sequence made it worse (515 vs 513).  What did work was letting
the AMDGPU scheduler interleave the clusters itself (`iglp.opt`) and deleting
instructions outright.

## The masked tile's schedule is chaotic, and it costs 10-20%

BEWARE when A/B-ing the STRIDED-READ arm of this kernel or of the backward's dQ
kernel. Their absolute time is partly a compilation lottery. Three harness
binaries built from ONE kernel source, differing only in which cases their case
list holds, run the nanogpt shape in the production layout at 562, 608 and 674
us/layer of GPU time (rocprofv3, same kernel symbol, same launch geometry). The
unmasked tile loop is instruction-for-instruction identical across those builds;
what moves is the AMDGPU schedule of the MASKED tile -- MFMA placement around
the barrier and the accumulator correction -- and that tile is 2 of ~9 per block
on this shape.

Two consequences, both paid for in measurements:

* A harness A/B on the strided arm has a +-10% floor. Only the end-to-end step
  time settles anything, and the step time is what the journal quotes.
* Trivia becomes load-bearing. Writing an output base as `stride * index` rather
  than `index * stride` -- two commutative multiplies, same value -- is worth
  109 us/layer on the backward's dQ kernel in the production build (9869 ->
  8821 us/step over twelve layers). Both harnesses carry a `nanogpt_bthd` case,
  the acceptance shape in the production layout, so a compiler release that
  re-rolls these dice is at least visible.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    barrier,
    block_idx,
    grid_dim,
    thread_idx,
)
from std.gpu.compute.mma import mma
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block
from std.gpu.primitives.warp import shuffle_xor
from std.math import ceildiv, exp, exp2, log
from std.memory import bitcast, stack_allocation
from std.sys.info import is_amd_gpu
from std.sys.intrinsics import llvm_intrinsic
from std.utils.static_tuple import StaticTuple

comptime THREADS = 256
# Scores for one KV tile, one per thread.
comptime BK = THREADS
# Largest head dimension the LDS staging below is sized for. A larger runtime
# head_dim is a legal input; the dispatch must route it somewhere correct.
comptime MAX_HEAD_DIM = 256

# log2(e): the softmax runs in log2 units so the exponential is a bare
# `v_exp_f32`.
comptime LOG2E = Float32(1.4426950408889634)

# Row of the 32x32 MFMA accumulator held by register `p`, before the
# `4 * (lane // 32)` half-wave shift. Hardware-determined.
comptime ACC_ROWS = SIMD[DType.int32, 16](
    0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
)

# The running max is kept in RAW accumulator units so `scale * log2(e)` folds
# into the exponent's `v_pk_fma_f32` instead of costing a separate multiply.
# A masked score is set to a finite -1e30 and the max is floored at -9e29, so a
# row with no live key at all (possible only when `seq_kv < seq_q` under a
# causal mask) yields `exp2(-1e29 * sl)` = 0 rather than `exp2(0)` = 1: the
# fully-masked row still reduces to an exact zero output, and no lane ever
# evaluates `exp(-inf - -inf)`.  `run_m` is still SEEDED with
# `Float32.MIN_FINITE`, which drives the first tile's correction to zero.
comptime NEG_BIG = SIMD[DType.float32, 16](-1.0e30)
comptime RAW_FLOOR = Float32(-9.0e29)


struct RowStrides(
    DevicePassable,
    ImplicitlyCopyable,
    TrivialRegisterPassable,
):
    """Element strides of one `[batch, heads, seq, head_dim]` attention operand.

    The `head_dim` stride is deliberately absent: it must be 1. Every loader
    here reads eight contiguous head-dim elements per lane, or fills an LDS row
    along head-dim, and the epilogue stores eight contiguous head-dim elements,
    so a strided innermost axis is not something this kernel can express -- a
    caller holding such a layout must decline rather than guess.

    Everything else is free. The three strides are runtime values, so a dense
    `[B, H, T, D]` tensor and the `x.view(B, T, H, D).transpose(1, 2)` a
    transformer actually produces are the same code path.
    """

    comptime device_type: AnyType = Self

    var batch: Int
    var head: Int
    var seq: Int

    def __init__(out self, batch: Int, head: Int, seq: Int):
        self.batch = batch
        self.head = head
        self.seq = seq

    def _to_device_type(
        self,
        mut encoder: Some[DeviceTypeEncoder],
        target: MutOpaquePointer[_],
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "RowStrides"


@always_inline
def dense_strides(heads: Int, seq: Int, head_dim: Int) -> RowStrides:
    """The strides of a dense row-major `[batch, heads, seq, head_dim]`."""
    return RowStrides(heads * seq * head_dim, seq * head_dim, head_dim)


@always_inline
def bthd_strides(heads: Int, seq: Int, head_dim: Int) -> RowStrides:
    """`[batch, heads, seq, head_dim]` logically, `[batch, seq, heads, head_dim]`
    in memory.

    The layout PyTorch's flash attention returns, and the one that makes
    `y.transpose(1, 2).contiguous()` a no-op. Element `(b, h, s, d)` sits at
    `((b*seq + s)*heads + h)*head_dim + d`.
    """
    return RowStrides(seq * heads * head_dim, head_dim, heads * head_dim)


@always_inline
def _pack_half[
    dtype: DType, w: Int
](x: SIMD[DType.float32, w]) -> SIMD[dtype, w]:
    """Round FP32 to `dtype` cheaply.

    gfx942 has no `v_cvt_pk_bf16_f32`, so `SIMD.cast` to bfloat16 expands into
    a six-instruction round-to-nearest-even sequence per element (NaN test,
    tie-break bias, select, pack).  Measured in the tile loop that was 209 of
    538 VALU instructions.  The scores here are `exp2` of a non-positive number
    and the outputs are a convex combination of V rows, so none of them is NaN
    and the even/away tie-break is irrelevant: adding half an ULP to the bit
    pattern and truncating rounds to nearest for both signs and costs one add
    plus the pack.
    """
    comptime if dtype == DType.bfloat16:
        var bits = bitcast[DType.uint32, w](x) + SIMD[DType.uint32, w](0x8000)
        # Deinterleave rather than shift-and-narrow: the high half of each
        # little-endian pair already IS the bfloat16, so this is a pure byte
        # selection and lowers to `v_perm_b32` instead of shift/shift/or.
        return bitcast[dtype, w](
            bitcast[DType.uint16, 2 * w](bits).deinterleave()[1]
        )
    else:
        return x.cast[dtype]()


@__name(t"flash_attention_fwd_baseline_{dtype}")
def _flash_attention_fwd_baseline[
    dtype: DType
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lse: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q_st: RowStrides,
    k_st: RowStrides,
    v_st: RowStrides,
    o_st: RowStrides,
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    causal: Int,
):
    """One block per (batch, head, query row). Correct, not fast."""
    var tid = Int(thread_idx.x)
    var qi = Int(block_idx.x)
    var head = Int(block_idx.y)
    var batch = Int(block_idx.z)
    var heads = Int(grid_dim.y)

    # Every operand, the output included, walks its own strides; only the
    # log-sum-exp is a dense `[batch, heads, seq_q]` allocation.
    var bh = batch * heads + head
    var q_row = batch * q_st.batch + head * q_st.head + qi * q_st.seq
    var k_base = batch * k_st.batch + head * k_st.head
    var v_base = batch * v_st.batch + head * v_st.head
    var o_row = batch * o_st.batch + head * o_st.head + qi * o_st.seq

    var q_smem = stack_allocation[
        MAX_HEAD_DIM, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var acc_smem = stack_allocation[
        MAX_HEAD_DIM, DType.float32, address_space=AddressSpace.SHARED
    ]()
    var s_smem = stack_allocation[
        BK, DType.float32, address_space=AddressSpace.SHARED
    ]()

    var d = tid
    while d < head_dim:
        q_smem[d] = query[q_row + d].cast[DType.float32]()
        acc_smem[d] = 0.0
        d += THREADS
    barrier()

    # Bottom-right aligned causal mask, PyTorch's `is_causal=True` convention.
    var limit = seq_kv
    if causal != 0:
        limit = min(seq_kv, qi + 1)

    var running_max = Float32.MIN_FINITE
    var running_sum = Float32(0.0)

    var kv_start = 0
    while kv_start < limit:
        var j = kv_start + tid
        var s = Float32.MIN_FINITE
        if j < limit:
            var krow = k_base + j * k_st.seq
            var dot = Float32(0.0)
            for e in range(head_dim):
                dot += q_smem[e] * key[krow + e].cast[DType.float32]()
            s = dot * scale
        s_smem[tid] = s
        barrier()

        var tile_max = block.max[block_size=THREADS](s)
        var new_max = max(running_max, tile_max)
        var correction = exp(running_max - new_max)

        var p = exp(s_smem[tid] - new_max) if j < limit else Float32(0.0)
        s_smem[tid] = p
        var tile_sum = block.sum[block_size=THREADS](p)
        barrier()

        running_sum = running_sum * correction + tile_sum
        running_max = new_max

        # acc[d] = acc[d] * correction + sum_j p_j * v[j][d]
        var dd = tid
        while dd < head_dim:
            var partial = Float32(0.0)
            for t in range(BK):
                var jj = kv_start + t
                if jj < limit:
                    partial += (
                        s_smem[t]
                        * value[v_base + jj * v_st.seq + dd].cast[
                            DType.float32
                        ]()
                    )
            acc_smem[dd] = acc_smem[dd] * correction + partial
            dd += THREADS
        barrier()
        kv_start += BK

    var inv = 1.0 / running_sum
    if tid == 0:
        # P = exp(S - L) with L the row log-sum-exp; `running_max` is already
        # the scaled score's max. A fully masked row contributes no key, so its
        # L is never read back -- store a finite 0 rather than -inf + log(0).
        var l = Float32(0.0)
        if running_sum > 0.0 and running_max > RAW_FLOOR:
            l = running_max + log(running_sum)
        lse[bh * seq_q + qi] = l
    var do = tid
    while do < head_dim:
        output[o_row + do] = (acc_smem[do] * inv).cast[dtype]()
        do += THREADS


# Without the flat-work-group-size metadata the AMDGPU backend assumes up to
# 1024 threads per block, which caps the kernel at 128 VGPRs and spills every
# accumulator to scratch. 256 threads is four wave64, one per SIMD.
@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(THREADS)),
    `rocdl.waves_per_eu`=SIMDSize(WAVES_PER_EU),
)
@__name(t"fa_mfma_{dtype}_h{HD}_n{BN}_q{QT}_x{EXACT}_d{DENSE}")
def _fa_mfma[
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
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lse: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q_st: RowStrides,
    k_st: RowStrides,
    v_st: RowStrides,
    o_st: RowStrides,
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    causal: Int,
):
    """MFMA flash-attention forward, four wave64 per workgroup.

    `HD` is the compile-time padded head dimension (a runtime `head_dim <= HD`
    is zero-filled in LDS and in the Q fragments, which contributes nothing to
    any score and nothing to any stored output column). `BN` is the KV tile,
    `QT` the number of 32-query MFMA tiles each wave owns, so the workgroup
    covers `4 * 32 * QT` query rows. `EXACT` asserts `head_dim == HD` and drops
    the per-element head-dim bound checks from the loaders.

    `DENSE` asserts that `q_st`, `k_st` and `v_st` are exactly the row-major
    strides of `[batch, heads, seq, head_dim]`, which lets every row offset in
    the loaders go back to being a compile-time immediate. It says nothing about
    `o_st`: the epilogue's store address was already a runtime multiply by
    `head_dim`, so reading the row stride out of `o_st` instead costs nothing
    and there is no arm worth specializing. It is not cosmetic:
    the general form measures 535.1 us/layer against 511.2 on the nanogpt case,
    because the V loader issues sixteen scalar global reads per tile and each
    one then needs its own address add instead of an immediate offset. The
    dispatch picks the specialization whenever the strides happen to be dense.

    The dense arm still measures 515.9 rather than 511.2, and that 0.9% is NOT
    the indexing: a build whose body never mentions the strides at all measures
    the same, and the gap scales with wave count rather than launch count, so it
    is the three extra structs in the kernarg segment. Splitting into two kernel
    entry points so the dense one keeps the old argument list does not recover
    it either (517.8) -- the `@always_inline` body then costs what the kernargs
    did. Both routes were measured; see the journal entry for this change.
    """
    comptime NW = 4
    comptime BM = NW * 32 * QT
    comptime KSTEPS = HD // 8  # k-steps of the Q@K^T GEMM
    comptime KVT = BN // 32  # kv MFMA tiles per score tile
    comptime DT = HD // 32  # head-dim MFMA tiles of the output
    comptime PKS = BN // 8  # k-steps of the P@V GEMM
    comptime KPAD = HD + 4  # LDS row length of the K tile
    comptime VPAD = BN + 4  # LDS row length of the V^T tile
    comptime OPAD = 36  # LDS row length of the output staging tile
    comptime KITERS = BN * HD // 8 // THREADS
    comptime VITERS = HD * (BN // 4) // THREADS
    comptime OPASSES = BM // 64
    comptime LDS_ELEMS = max(BN * KPAD + HD * VPAD, BM * OPAD)

    comptime if is_amd_gpu():
        var smem = stack_allocation[
            LDS_ELEMS, dtype, address_space=AddressSpace.SHARED, alignment=16
        ]()
        var k_smem = smem
        var v_smem = smem + BN * KPAD

        var tid = Int(thread_idx.x)
        var wave = tid // 64
        var lane = tid % 64
        var lo = lane % 32  # index along the MFMA's non-K axis
        var hi = lane // 32  # which half-wave, i.e. which K quad

        var heads = Int(grid_dim.y)
        var bz = Int(block_idx.z)
        var by = Int(block_idx.y)
        var bh = bz * heads + by
        # Every base follows its own operand's strides; only the log-sum-exp is
        # dense.
        var q_base = bz * q_st.batch + by * q_st.head
        var k_base = bz * k_st.batch + by * k_st.head
        var v_base = bz * v_st.batch + by * v_st.head
        comptime if DENSE:
            q_base = bh * seq_q * head_dim
            k_base = bh * seq_kv * head_dim
            v_base = k_base

        # The row strides. Under `DENSE` these fold to `HD` whenever the head
        # dimension is exact, which is what restores the immediate offsets.
        var qss = q_st.seq
        var kss = k_st.seq
        var vss = v_st.seq
        comptime if DENSE:
            qss = head_dim
            kss = head_dim
            vss = head_dim
            comptime if EXACT:
                qss = HD
                kss = HD
                vss = HD

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
        # Tiles below this index are live for EVERY query in the block, so they
        # need no mask and no bound check.
        var n_safe = max(lim_lo, 0) // BN

        # ---- Q fragments: the B operand of S^T = K @ Q^T, resident for the
        # whole kv loop. Lane L holds query `L%32`, head-dim `8s + 4*(L//32)`.
        var q_frag = stack_allocation[QT * KSTEPS * 4, dtype]()
        var q_row0 = q_block + wave * (32 * QT)
        comptime for qt in range(QT):
            var qg = q_row0 + qt * 32 + lo
            var live = qg < seq_q
            var qrow = q_base + qg * qss + 4 * hi
            comptime for s in range(KSTEPS):
                var v = SIMD[dtype, 4](0)
                comptime if EXACT:
                    if live:
                        v = query.load[width=4](qrow + 8 * s)
                else:
                    if live and 8 * s + 4 * hi < head_dim:
                        v = query.load[width=4](qrow + 8 * s)
                q_frag.store((qt * KSTEPS + s) * 4, v)

        # ---- Online-softmax state: one scalar per lane per query tile.
        var run_m = stack_allocation[QT, DType.float32]()
        var run_s = stack_allocation[QT, DType.float32]()
        var qlim = stack_allocation[QT, DType.int32]()
        comptime for qt in range(QT):
            run_m[qt] = Float32.MIN_FINITE
            run_s[qt] = 0.0
            var qg = q_row0 + qt * 32 + lo
            var lm = seq_kv
            if causal != 0:
                lm = min(seq_kv, qg + delta + 1)
            qlim[qt] = Int32(lm)

        var s_acc = stack_allocation[KVT * QT * 16, DType.float32]()
        var p_frag = stack_allocation[KVT * QT * 16, dtype]()
        var o_acc = stack_allocation[DT * QT * 16, DType.float32]()
        comptime for i in range(DT * QT):
            o_acc.store(i * 16, SIMD[DType.float32, 16](0))

        var sl = scale * LOG2E

        # The row stride is folded into a pointer bump so the tile index never
        # enters an address. What the loaders still pay for when it is a runtime
        # value is the WITHIN-tile row offsets: `ci * KROW_STEP * kss` and
        # `(4*vi_group + j) * vss` are compile-time immediates only under
        # `DENSE`, and there are eighteen of them per tile. That is the whole of
        # the 535.1-vs-511.2 gap, and the reason `DENSE` exists.
        comptime KROW_STEP = THREADS // (HD // 8)
        comptime VKV_STEP = THREADS // HD
        var k_row0 = tid // (HD // 8)
        var k_col = (tid % (HD // 8)) * 8
        var v_col = tid % HD
        var v_kvg0 = tid // HD
        var k_ptr = key + (k_base + k_row0 * kss + k_col)
        var v_ptr = value + (v_base + v_kvg0 * 4 * vss + v_col)
        var k_tile_step = BN * kss
        var v_tile_step = BN * vss
        var k_lds0 = k_row0 * KPAD + k_col
        var v_lds0 = v_col * VPAD + v_kvg0 * 4
        # Beyond `seq_kv` the loaders read a clamped, in-bounds duplicate row
        # rather than branching per lane: every such score is replaced by the
        # mask before it reaches the running max, so the staged values are never
        # observed. Only the final tile of a sequence takes this path.
        var last_row = max(seq_kv - 1, 0)

        # The tile loop is peeled at `n_safe`.  Below it every query in the
        # block sees every key of the tile and the tile lies wholly inside
        # `seq_kv`, so the causal compare, the select and the clamped loader all
        # vanish at compile time rather than sitting behind a branch.  Leaving
        # them behind a runtime `if` cost 70 us/layer on the nanogpt case even
        # though only two tiles per block ever take it.
        @always_inline
        @parameter
        def _tile[MASKED: Bool](t: Int):
            var kv0 = t * BN
            var full = True
            comptime if MASKED:
                full = kv0 + BN <= seq_kv

            # ---- Stage K row-major and V transposed. ----
            if full:
                comptime for ci in range(KITERS):
                    var vals = SIMD[dtype, 8](0)
                    comptime if EXACT:
                        vals = k_ptr.load[width=8](ci * KROW_STEP * kss)
                    else:
                        if k_col < head_dim:
                            vals = k_ptr.load[width=8](ci * KROW_STEP * kss)
                    k_smem.store(k_lds0 + ci * KROW_STEP * KPAD, vals)
                comptime for vi in range(VITERS):
                    var vv = SIMD[dtype, 4](0)
                    var live = True
                    comptime if not EXACT:
                        live = v_col < head_dim
                    if live:
                        comptime for j in range(4):
                            vv[j] = v_ptr[(vi * VKV_STEP * 4 + j) * vss]
                    v_smem.store(v_lds0 + vi * VKV_STEP * 4, vv)
            else:
                comptime for ci in range(KITERS):
                    var row = min(kv0 + k_row0 + ci * KROW_STEP, last_row)
                    var vals = SIMD[dtype, 8](0)
                    if EXACT or k_col < head_dim:
                        vals = key.load[width=8](k_base + row * kss + k_col)
                    k_smem.store(k_lds0 + ci * KROW_STEP * KPAD, vals)
                comptime for vi in range(VITERS):
                    var vv = SIMD[dtype, 4](0)
                    if EXACT or v_col < head_dim:
                        var g = kv0 + (v_kvg0 + vi * VKV_STEP) * 4
                        comptime for j in range(4):
                            vv[j] = value[
                                v_base + min(g + j, last_row) * vss + v_col
                            ]
                    v_smem.store(v_lds0 + vi * VKV_STEP * 4, vv)

            k_ptr += k_tile_step
            v_ptr += v_tile_step
            barrier()

            llvm_intrinsic["llvm.amdgcn.iglp.opt", NoneType](Int32(IGLP))
            # ---- GEMM 1: S^T[kv, q] = sum_d K[kv, d] * Q[q, d] ----
            comptime for i in range(KVT * QT):
                s_acc.store(i * 16, SIMD[DType.float32, 16](0))
            var kbase = lo * KPAD + 4 * hi
            comptime for s in range(KSTEPS):
                var afrag = stack_allocation[KVT * 4, dtype]()
                comptime for kt in range(KVT):
                    afrag.store(
                        kt * 4,
                        k_smem.load[width=4](kbase + kt * 32 * KPAD + 8 * s),
                    )
                comptime for qt in range(QT):
                    var b = q_frag.load[width=4]((qt * KSTEPS + s) * 4)
                    comptime for kt in range(KVT):
                        var d = s_acc.load[width=16]((kt * QT + qt) * 16)
                        mma(d, afrag.load[width=4](kt * 4), b, d)
                        s_acc.store((kt * QT + qt) * 16, d)

            # ---- Online softmax, entirely within a lane plus one exchange ----
            comptime for qt in range(QT):
                var tmax = Float32.MIN_FINITE
                comptime for kt in range(KVT):
                    var v16 = s_acc.load[width=16]((kt * QT + qt) * 16)
                    comptime if MASKED:
                        # `kv` of element p is `kv0 + 32*kt + 4*hi + ACC_ROWS[p]`
                        # and the causal bound is one scalar per lane, so the
                        # whole mask is one compare against a constant vector.
                        var kvv = (
                            SIMD[DType.int32, 16](kv0 + kt * 32 + 4 * hi)
                            + ACC_ROWS
                        )
                        v16 = kvv.lt(SIMD[DType.int32, 16](qlim[qt])).select(
                            v16, NEG_BIG
                        )
                    s_acc.store((kt * QT + qt) * 16, v16)
                    tmax = max(tmax, v16.reduce_max())
                # Lanes L and L^32 hold the same query column.
                tmax = max(tmax, shuffle_xor(tmax, 32))
                var nm = max(max(run_m[qt], tmax), RAW_FLOOR)
                var corr = exp2((run_m[qt] - nm) * sl)
                run_m[qt] = nm
                var negm = -nm * sl

                var psum = Float32(0)
                comptime for kt in range(KVT):
                    var pv = exp2(
                        s_acc.load[width=16]((kt * QT + qt) * 16).fma(
                            SIMD[DType.float32, 16](sl),
                            SIMD[DType.float32, 16](negm),
                        )
                    )
                    p_frag.store((kt * QT + qt) * 16, _pack_half[dtype, 16](pv))
                    psum += pv.reduce_add()
                run_s[qt] = run_s[qt] * corr + psum
                comptime for dt in range(DT):
                    var acc = o_acc.load[width=16]((dt * QT + qt) * 16)
                    o_acc.store((dt * QT + qt) * 16, acc * corr)

            # ---- GEMM 2: O^T[d, q] += sum_kv V^T[d, kv] * P^T[kv, q] ----
            var vbase = lo * VPAD + 4 * hi
            comptime for ks in range(PKS):
                var vfrag = stack_allocation[DT * 4, dtype]()
                comptime for dt in range(DT):
                    vfrag.store(
                        dt * 4,
                        v_smem.load[width=4](vbase + dt * 32 * VPAD + 8 * ks),
                    )
                comptime for qt in range(QT):
                    var b = p_frag.load[width=4](
                        ((ks // 4) * QT + qt) * 16 + (ks % 4) * 4
                    )
                    comptime for dt in range(DT):
                        var d = o_acc.load[width=16]((dt * QT + qt) * 16)
                        mma(d, vfrag.load[width=4](dt * 4), b, d)
                        o_acc.store((dt * QT + qt) * 16, d)

            barrier()

        var t = 0
        comptime if PEEL:
            while t < n_safe:
                _tile[False](t)
                t += 1
        while t < n_tiles:
            _tile[True](t)
            t += 1

        # ---- Epilogue. O^T lives with one query per lane and 16 scattered
        # head-dim rows, so it is staged through LDS one 32-column slab at a
        # time and read back in row-major order for coalesced global stores.
        var inv = stack_allocation[QT, DType.float32]()
        comptime for qt in range(QT):
            var tot = run_s[qt] + shuffle_xor(run_s[qt], 32)
            var r = Float32(0)
            if tot > 0.0 and run_m[qt] > RAW_FLOOR:
                r = 1.0 / tot
            inv[qt] = r
            # `run_m` is the max of the RAW dot product and the exponent used is
            # `(s_raw - run_m) * scale`, so the natural-log row log-sum-exp is
            # `run_m * scale + ln(tot)`. Lanes L and L^32 hold the same query
            # (both `tmax` and `tot` are reduced across the pair), so exactly
            # one half-wave stores it.
            var qg_l = q_block + wave * (32 * QT) + qt * 32 + lo
            if hi == 0 and qg_l < seq_q:
                var l = Float32(0.0)
                if tot > 0.0 and run_m[qt] > RAW_FLOOR:
                    l = run_m[qt] * scale + log(tot)
                lse[bh * seq_q + qg_l] = l

        # STRIDE FIRST, INDEX SECOND. That reads naturally, and on this target
        # it is also LOAD-BEARING -- see the note at the top of this file on the
        # masked tile's schedule. Written `bz * o_st.batch + by * o_st.head`, the
        # SAME source, the dQ backward kernel measures 109 us/layer worse in the
        # production build. Read here rather than next to the read bases so that
        # nothing about the output layout is live across the tile loop.
        var o_base = o_st.batch * bz + o_st.head * by
        var oss = o_st.seq

        comptime for dt in range(DT):
            barrier()
            comptime for qt in range(QT):
                var row = wave * (32 * QT) + qt * 32 + lo
                var acc = o_acc.load[width=16]((dt * QT + qt) * 16) * inv[qt]
                var ob = _pack_half[dtype, 16](acc)
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
                    output.store(
                        o_base + qg * oss + dcol,
                        smem.load[width=8](r * OPAD + cc),
                    )


@always_inline
def _is_dense(st: RowStrides, heads: Int, seq: Int, head_dim: Int) -> Bool:
    """Whether `st` is exactly row-major `[batch, heads, seq, head_dim]`."""
    return (
        st.seq == head_dim
        and st.head == seq * head_dim
        and st.batch == heads * seq * head_dim
    )


def _enqueue_fa_mfma[
    dtype: DType,
    HD: Int,
    BN: Int,
    QT: Int,
    EXACT: Bool,
    WAVES_PER_EU: Int = 1,
    IGLP: Int = 1,
    PEEL: Bool = True,
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lse: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q_st: RowStrides,
    k_st: RowStrides,
    v_st: RowStrides,
    o_st: RowStrides,
    batch: Int,
    heads: Int,
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    is_causal: Bool,
    ctx: DeviceContext,
) raises:
    # `DENSE` is a property of the READ operands only; see `_fa_mfma`.
    var dense = (
        _is_dense(q_st, heads, seq_q, head_dim)
        and _is_dense(k_st, heads, seq_kv, head_dim)
        and _is_dense(v_st, heads, seq_kv, head_dim)
    )
    if dense:
        ctx.enqueue_function[
            _fa_mfma[dtype, HD, BN, QT, EXACT, True, WAVES_PER_EU, IGLP, PEEL]
        ](
            output,
            lse,
            query,
            key,
            value,
            q_st,
            k_st,
            v_st,
            o_st,
            seq_q,
            seq_kv,
            head_dim,
            scale,
            1 if is_causal else 0,
            grid_dim=(ceildiv(seq_q, 4 * 32 * QT), heads, batch),
            block_dim=(THREADS,),
        )
        return
    ctx.enqueue_function[
        _fa_mfma[dtype, HD, BN, QT, EXACT, False, WAVES_PER_EU, IGLP, PEEL]
    ](
        output,
        lse,
        query,
        key,
        value,
        q_st,
        k_st,
        v_st,
        o_st,
        seq_q,
        seq_kv,
        head_dim,
        scale,
        1 if is_causal else 0,
        grid_dim=(ceildiv(seq_q, 4 * 32 * QT), heads, batch),
        block_dim=(THREADS,),
    )


def enqueue_flash_attention_fwd[
    dtype: DType
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lse: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q_st: RowStrides,
    k_st: RowStrides,
    v_st: RowStrides,
    o_st: RowStrides,
    batch: Int,
    heads: Int,
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    is_causal: Bool,
    ctx: DeviceContext,
) raises:
    """THE FROZEN ENTRY POINT. Q, K, V and output are `[batch, heads, seq, head_dim]`;
    `seq` is `seq_q` for Q and output, `seq_kv` for K and V.

    Q, K and V are read through `q_st`, `k_st` and `v_st`, and `output` is
    WRITTEN through `o_st`: the element strides of their batch, head and seq
    axes. The head_dim axis must have stride 1 in all four -- it is the axis
    every vectorized load and the epilogue's vectorized store run along -- and a
    caller holding a layout where it is not must decline rather than pass one
    here. `lse` is dense.

    `dense_strides(heads, seq, head_dim)` builds the triple for a plain
    row-major `[batch, heads, seq, head_dim]` tensor; `bthd_strides` builds the
    one for `[batch, heads, seq, head_dim]` stored `[batch, seq, heads,
    head_dim]`, which is what PyTorch's flash attention returns and what makes
    `y.transpose(1, 2).contiguous()` free.

    Every extent AND every stride is a runtime value. Dispatch on them as you
    like; keep this signature and keep every runtime input correct.
    """
    if batch <= 0 or heads <= 0 or seq_q <= 0 or seq_kv <= 0 or head_dim <= 0:
        return
    if head_dim > MAX_HEAD_DIM:
        raise Error(
            "flash attention forward: head_dim ",
            head_dim,
            " exceeds the staged maximum ",
            MAX_HEAD_DIM,
            (
                "; a replacement must either raise the staging or route this"
                " head_dim to a path that handles it"
            ),
        )

    # The MFMA path needs a half-float MFMA and 8-element vector loads along
    # head_dim. Everything else falls through to the general baseline.
    comptime if dtype.is_half_float():
        if ctx.api() == "hip" and head_dim % 8 == 0:
            if head_dim == 64:
                _enqueue_fa_mfma[dtype, 64, 64, 1, True](
                    output,
                    lse,
                    query,
                    key,
                    value,
                    q_st,
                    k_st,
                    v_st,
                    o_st,
                    batch,
                    heads,
                    seq_q,
                    seq_kv,
                    head_dim,
                    scale,
                    is_causal,
                    ctx,
                )
                return
            if head_dim < 64:
                _enqueue_fa_mfma[dtype, 64, 64, 1, False](
                    output,
                    lse,
                    query,
                    key,
                    value,
                    q_st,
                    k_st,
                    v_st,
                    o_st,
                    batch,
                    heads,
                    seq_q,
                    seq_kv,
                    head_dim,
                    scale,
                    is_causal,
                    ctx,
                )
                return
            if head_dim == 128:
                _enqueue_fa_mfma[dtype, 128, 64, 1, True, 1, 1, False](
                    output,
                    lse,
                    query,
                    key,
                    value,
                    q_st,
                    k_st,
                    v_st,
                    o_st,
                    batch,
                    heads,
                    seq_q,
                    seq_kv,
                    head_dim,
                    scale,
                    is_causal,
                    ctx,
                )
                return
            if head_dim < 128:
                _enqueue_fa_mfma[dtype, 128, 64, 1, False, 1, 1, False](
                    output,
                    lse,
                    query,
                    key,
                    value,
                    q_st,
                    k_st,
                    v_st,
                    o_st,
                    batch,
                    heads,
                    seq_q,
                    seq_kv,
                    head_dim,
                    scale,
                    is_causal,
                    ctx,
                )
                return
            if head_dim == 256:
                _enqueue_fa_mfma[dtype, 256, 32, 1, True, 1, 1, False](
                    output,
                    lse,
                    query,
                    key,
                    value,
                    q_st,
                    k_st,
                    v_st,
                    o_st,
                    batch,
                    heads,
                    seq_q,
                    seq_kv,
                    head_dim,
                    scale,
                    is_causal,
                    ctx,
                )
                return
            _enqueue_fa_mfma[dtype, 256, 32, 1, False, 1, 1, False](
                output,
                lse,
                query,
                key,
                value,
                q_st,
                k_st,
                v_st,
                o_st,
                batch,
                heads,
                seq_q,
                seq_kv,
                head_dim,
                scale,
                is_causal,
                ctx,
            )
            return

    ctx.enqueue_function[_flash_attention_fwd_baseline[dtype]](
        output,
        lse,
        query,
        key,
        value,
        q_st,
        k_st,
        v_st,
        o_st,
        seq_q,
        seq_kv,
        head_dim,
        scale,
        1 if is_causal else 0,
        grid_dim=(seq_q, heads, batch),
        block_dim=(THREADS,),
    )
