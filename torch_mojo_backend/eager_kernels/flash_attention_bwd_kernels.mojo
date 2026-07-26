"""Fused causal flash-attention backward for gfx942. THE FILE TO OPTIMIZE.

The contract below is frozen; `harness/nanogpt_train/bench_flash_attention_bwd.mojo`
calls exactly this entry point and defines acceptance. What is inside is a
correct, deliberately unoptimized baseline: two kernels, one block per query row
for dQ and one block per key row for dK/dV, everything read from global memory.
It is right, and it is slow.

There is no reference implementation to port. MAX has no attention backward
anywhere (`flash_attention_bwd`, `mha_bwd`, `attention_backward` all have zero
hits under `max/`), and this repository's own `eager_flash_attention/fa4_bwd_*`
is written against H100 wgmma/TMA and will not compile for this target. The
forward counterpart in `flash_attention_fwd_kernels.mojo` is the closest useful
model, and it is a good one: it beats PyTorch-ROCm's fused forward by 25%.

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
which is why `softmax_lse` is an argument. That is the same information PyTorch's
own `_scaled_dot_product_flash_attention_backward` receives.

## Invariants any replacement must keep

* `batch`, `heads`, `seq_q`, `seq_kv` and `head_dim` are RUNTIME values. Tile
  shapes, pipeline depth and head-dim regimes may be compile-time as long as a
  runtime dispatch picks between them and every runtime head_dim is handled.
* No allocation, no host transfer, no synchronization: enqueue on the caller's
  `DeviceContext` and return. Multiple kernel launches are fine.
* Write only to `dq`, `dk`, `dv`. Everything else is read-only. The three output
  buffers arrive ZEROED, so a kernel may accumulate into them, but it must then
  be launched such that no two workgroups accumulate into the same element
  without atomics -- silently losing a contribution is the classic defect here.
* Accumulate in FP32, store once in `dtype`.
* Seed any running max with `Float32.MIN_FINITE`, never `Float32.MIN` -- the
  latter is -inf, so a lane that never runs its loop body computes
  `exp(-inf - -inf)` = nan, which poisons a whole row through a reduction. That
  defect has shipped three times in this repository. `-Float32.MAX` fails the
  same way, since `Float32.MAX` is inf.
* `is_causal` masks strictly above the diagonal aligned to the BOTTOM right, so
  query row `q` attends key indices `0 ..= q + (seq_kv - seq_q)`. Equivalently,
  key `j` is attended by query rows `max(0, j - (seq_kv - seq_q)) ..< seq_q`.
  Both directions matter here, because dK/dV iterate the other way round.
"""

from std.gpu import barrier, block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.gpu.primitives import block
from std.math import ceildiv, exp
from std.memory import stack_allocation

comptime THREADS = 256
comptime MAX_HEAD_DIM = 256


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
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    causal: Int,
):
    """One block per (batch, head, query row) accumulating dQ for that row."""
    var tid = Int(thread_idx.x)
    var qi = Int(block_idx.x)
    var head = Int(block_idx.y)
    var batch = Int(block_idx.z)
    var heads = Int(grid_dim.y)

    var bh = batch * heads + head
    var q_row = bh * seq_q * head_dim + qi * head_dim
    var kv_base = bh * seq_kv * head_dim

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
        do_s[d] = grad_output[q_row + d].cast[DType.float32]()
        acc[d] = 0.0
        d += THREADS
    barrier()

    # D_i = sum_d dO[i,d] * O[i,d], one scalar for this row.
    var partial = Float32(0.0)
    var dd = tid
    while dd < head_dim:
        partial += do_s[dd] * out_fwd[q_row + dd].cast[DType.float32]()
        dd += THREADS
    var row_d = block.sum[block_size=THREADS](partial)
    var row_lse = lse[bh * seq_q + qi]

    var limit = seq_kv
    if causal != 0:
        limit = min(seq_kv, qi + (seq_kv - seq_q) + 1)

    # Every thread walks every key, so the whole block agrees on ds; the
    # head_dim axis is what is split across threads.
    for j in range(limit):
        var krow = kv_base + j * head_dim
        var sp = Float32(0.0)
        var dp = Float32(0.0)
        var e = tid
        while e < head_dim:
            sp += q_s[e] * key[krow + e].cast[DType.float32]()
            dp += do_s[e] * value[krow + e].cast[DType.float32]()
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
        dq[q_row + o] = acc[o].cast[dtype]()
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
    seq_q: Int,
    seq_kv: Int,
    head_dim: Int,
    scale: Float32,
    causal: Int,
):
    """One block per (batch, head, key row) accumulating dK and dV for that row.

    Owning a key row per workgroup is what makes the accumulation race-free
    without atomics: every contribution to `dk[j]` and `dv[j]` comes from this
    block alone.
    """
    var tid = Int(thread_idx.x)
    var j = Int(block_idx.x)
    var head = Int(block_idx.y)
    var batch = Int(block_idx.z)
    var heads = Int(grid_dim.y)

    var bh = batch * heads + head
    var kv_row = bh * seq_kv * head_dim + j * head_dim
    var q_base = bh * seq_q * head_dim

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
        k_s[d] = key[kv_row + d].cast[DType.float32]()
        v_s[d] = value[kv_row + d].cast[DType.float32]()
        acc_k[d] = 0.0
        acc_v[d] = 0.0
        d += THREADS
    barrier()

    # Key j is attended by query rows from `first_q` upward.
    var first_q = 0
    if causal != 0:
        first_q = max(0, j - (seq_kv - seq_q))

    for qi in range(first_q, seq_q):
        var q_row = q_base + qi * head_dim
        var sp = Float32(0.0)
        var dp = Float32(0.0)
        var dsum = Float32(0.0)
        var e = tid
        while e < head_dim:
            var dov = grad_output[q_row + e].cast[DType.float32]()
            sp += k_s[e] * query[q_row + e].cast[DType.float32]()
            dp += dov * v_s[e]
            dsum += dov * out_fwd[q_row + e].cast[DType.float32]()
            e += THREADS
        var s_dot = block.sum[block_size=THREADS](sp)
        var dp_dot = block.sum[block_size=THREADS](dp)
        var row_d = block.sum[block_size=THREADS](dsum)
        var p = exp(s_dot * scale - lse[bh * seq_q + qi])
        var ds = p * (dp_dot - row_d)
        var e2 = tid
        while e2 < head_dim:
            acc_v[e2] += p * grad_output[q_row + e2].cast[DType.float32]()
            acc_k[e2] += ds * query[q_row + e2].cast[DType.float32]() * scale
            e2 += THREADS
        barrier()

    var o = tid
    while o < head_dim:
        dk[kv_row + o] = acc_k[o].cast[dtype]()
        dv[kv_row + o] = acc_v[o].cast[dtype]()
        o += THREADS


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

    Layouts, all dense row-major:
      `dq`, `query`, `grad_output`, `out_fwd`  `[batch, heads, seq_q, head_dim]`
      `dk`, `dv`, `key`, `value`              `[batch, heads, seq_kv, head_dim]`
      `softmax_lse`                            `[batch, heads, seq_q]`, FP32

    `softmax_lse[i]` is the natural log of the sum of `exp(S[i, :] - max)` plus
    that max, i.e. the row log-sum-exp, so `P[i, j] = exp(S[i, j] - lse[i])`.

    `dq`, `dk` and `dv` arrive zeroed. Every extent is a runtime value.
    """
    if batch <= 0 or heads <= 0 or seq_q <= 0 or seq_kv <= 0 or head_dim <= 0:
        return
    if head_dim > MAX_HEAD_DIM:
        raise Error(
            "flash attention backward: head_dim ",
            head_dim,
            " exceeds the staged maximum ",
            MAX_HEAD_DIM,
            "; a replacement must raise the staging or route this head_dim to a"
            " path that handles it",
        )
    var causal = 1 if is_causal else 0
    ctx.enqueue_function[_bwd_dq_baseline[dtype]](
        dq,
        grad_output,
        query,
        key,
        value,
        out_fwd,
        softmax_lse,
        seq_q,
        seq_kv,
        head_dim,
        scale,
        causal,
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
        seq_q,
        seq_kv,
        head_dim,
        scale,
        causal,
        grid_dim=(seq_kv, heads, batch),
        block_dim=(THREADS,),
    )
