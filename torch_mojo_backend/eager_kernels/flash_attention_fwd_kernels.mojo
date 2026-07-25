"""Fused causal flash-attention forward for gfx942. THE FILE TO OPTIMIZE.

The contract below is frozen; `harness/nanogpt_train/bench_flash_attention.mojo`
calls exactly this entry point and defines acceptance. What is inside is a
correct, deliberately unoptimized baseline: one 256-thread block per query row,
K and V read straight from global memory, scores staged in LDS, online softmax
with a block reduction per KV tile. It is right, and it is slow.

The optimization target is `enqueue_flash_attention_fwd`'s runtime, nothing else.

Invariants any replacement must keep:

* `batch`, `heads`, `seq_q`, `seq_kv` and `head_dim` are RUNTIME values. Tile
  shapes, pipeline depth and head-dim regimes may be compile-time as long as a
  runtime dispatch picks between them and every runtime head_dim is handled.
* No allocation, no host transfer, no synchronization: enqueue on the caller's
  `DeviceContext` and return.
* Never write outside `output`. Q, K and V are read-only.
* Numerics: accumulate in FP32, store once in `dtype`. Seed a running max with
  `Float32.MIN_FINITE`, never `Float32.MIN` -- the latter is -inf, and an idle
  lane then computes `exp(-inf - -inf)` = nan, which poisons the whole row
  through the block reduction. That defect has been shipped twice in this
  repository; the harness's NaN counter exists because of it.
* `is_causal` masks strictly above the diagonal aligned to the BOTTOM right, so
  query row `q` attends to key indices `0 ..= q + (seq_kv - seq_q)`. That is
  PyTorch's convention for `is_causal=True` and it matters when
  `seq_kv != seq_q`.
"""

from std.gpu import barrier, block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block
from std.math import ceildiv, exp
from std.memory import stack_allocation

comptime THREADS = 256
# Scores for one KV tile, one per thread.
comptime BK = THREADS
# Largest head dimension the LDS staging below is sized for. A larger runtime
# head_dim is a legal input; the dispatch must route it somewhere correct.
comptime MAX_HEAD_DIM = 256


@__name(t"flash_attention_fwd_baseline_{dtype}")
def _flash_attention_fwd_baseline[
    dtype: DType
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
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

    # [batch, heads, seq, head_dim] row-major.
    var qkv_head = (batch * heads + head) * head_dim
    var q_base = qkv_head * seq_q + qi * head_dim
    var kv_base = qkv_head * seq_kv

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
        q_smem[d] = query[q_base + d].cast[DType.float32]()
        acc_smem[d] = 0.0
        d += THREADS
    barrier()

    # Bottom-right aligned causal mask, PyTorch's `is_causal=True` convention.
    var limit = seq_kv
    if causal != 0:
        limit = min(seq_kv, qi + (seq_kv - seq_q) + 1)

    var running_max = Float32.MIN_FINITE
    var running_sum = Float32(0.0)

    var kv_start = 0
    while kv_start < limit:
        var j = kv_start + tid
        var s = Float32.MIN_FINITE
        if j < limit:
            var krow = kv_base + j * head_dim
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
                        * value[kv_base + jj * head_dim + dd].cast[
                            DType.float32
                        ]()
                    )
            acc_smem[dd] = acc_smem[dd] * correction + partial
            dd += THREADS
        barrier()
        kv_start += BK

    var inv = 1.0 / running_sum
    var do = tid
    while do < head_dim:
        output[q_base + do] = (acc_smem[do] * inv).cast[dtype]()
        do += THREADS


def enqueue_flash_attention_fwd[
    dtype: DType
](
    output: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    query: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    key: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    value: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    batch: Int,
    heads: Int,
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    is_causal: Bool,
    ctx: DeviceContext,
) raises:
    """THE FROZEN ENTRY POINT. Q, K, V and output are `[batch, heads, seq, head_dim]`
    row-major and dense; `seq` is `seq_q` for Q and output, `seq_kv` for K and V.

    Every extent is a runtime value. Dispatch on them as you like; keep this
    signature and keep every runtime input correct.
    """
    if batch <= 0 or heads <= 0 or seq_q <= 0 or seq_kv <= 0 or head_dim <= 0:
        return
    if head_dim > MAX_HEAD_DIM:
        raise Error(
            "flash attention forward: head_dim ",
            head_dim,
            " exceeds the staged maximum ",
            MAX_HEAD_DIM,
            "; a replacement must either raise the staging or route this"
            " head_dim to a path that handles it",
        )
    ctx.enqueue_function[_flash_attention_fwd_baseline[dtype]](
        output,
        query,
        key,
        value,
        seq_q,
        seq_kv,
        head_dim,
        scale,
        1 if is_causal else 0,
        grid_dim=(seq_q, heads, batch),
        block_dim=(THREADS,),
    )
