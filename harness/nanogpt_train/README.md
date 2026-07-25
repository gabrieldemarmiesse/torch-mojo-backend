# nanoGPT training-step kernel harnesses (MI300X, gfx942)

Frozen workload: nanoGPT's GPT-2 124M (12 layers, 12 heads, 768 embedding,
`bias=False`, `vocab_size=50304`), batch 48, block 1024, BF16 autocast, fused
AdamW, gradient clipping 1.0, eager execution, no `torch.compile`.

Measured on an MI300X VF with PyTorch 2.11.0+rocm7.2 and MAX
26.5.0.dev2026072306:

| backend | step | tokens/s |
|---|---:|---:|
| PyTorch-ROCm | 156.64 ms | 313 799 |
| eager Mojo device, initial | 921.39 ms | 53 346 |
| eager Mojo device, after the Linear GEMM work | 661.86 ms | 74 264 |

Both backends are GPU-bound (trace idle 0.01%), so GPU kernel time is the unit
of comparison. Current ranked gap per step, from
`current_bench_train/comparison_v2/nanogpt_train_kernel_gap.csv`:

| target | mojo | rocm | ratio | share of gap |
|---|---:|---:|---:|---:|
| SDPA (backward) | 239.28 ms | 33.95 ms | 7.05x | 40.2% |
| Linear GEMM (backward) | 186.22 ms | 43.65 ms | 4.27x | 27.9% |
| SDPA (forward) | 105.07 ms | 9.08 ms | 11.57x | 18.8% |
| Linear GEMM (forward) | 72.29 ms | 24.03 ms | 3.01x | 9.4% |
| Copies / dtype casts | 26.03 ms | 11.44 ms | 2.28x | 2.9% |
| everything else | 32.11 ms | 34.87 ms | 0.92x | 0% |

The initial table, before the Linear GEMM work, was SDPA backward 239.24,
Linear GEMM backward 346.89, Linear GEMM forward 171.84, SDPA forward 105.10.
Mojo is already *faster* than PyTorch-ROCm on LayerNorm forward and backward,
GELU backward, fused AdamW, gradient clipping and concat.

Acceptance for this work: **mojo/rocm <= 1.02** on the per-step weighted total
of a target, with every correctness case passing.

## Why these harnesses are pure Mojo

`torch_mojo_backend/eager_kernels/__init__.py` content-addresses `__mojocache__`
over *all* `.mojo` files in that directory, so editing one kernel recompiles all
twenty CPython extension modules — about ten minutes per iteration. A
standalone `mojo build` of a harness that imports the same production modules
takes about **two seconds**:

```bash
uv run --no-sync mojo build harness/nanogpt_train/bench_linear_gemm.mojo \
    -I torch_mojo_backend/eager_kernels -o /tmp/bench_linear_gemm
/tmp/bench_linear_gemm --targets harness/nanogpt_train/rocm_gemm_targets.csv
```

The harness calls the real dispatch entry points, so an accepted improvement in
the harness is an improvement in production. Re-validate through Python only at
the end of a change, not per iteration.

## Files

- `bench_linear_gemm.mojo` — every Linear GEMM the step issues (forward, data
  gradient, weight gradient), timed against the PyTorch-ROCm reference, with
  two independent correctness checks.
- `rocm_gemm_targets.csv` — PyTorch-ROCm reference times, regenerate with
  `uv run --no-sync python scripts/rocm_gemm_reference.py --output <path>`
  (25 warmups, 100 synchronized iterations each).
- `bench_attention_bmm.mojo` — the six batched GEMMs of the eager SDPA
  decomposition, timed against `torch.bmm` for the identical batched shape.
- `rocm_attention_bmm_targets.csv`, `rocm_attention_targets.csv` — regenerate
  both with `uv run --no-sync python scripts/rocm_attention_reference.py
  --output-op <path> --output-bmm <path>`.

## Correctness gates

Both are exact equality tests, not tolerance tests, and both inspect **every**
output element via a device-side reduction (no multi-GB host transfer):

1. **Default (all ones).** `A = B = 1`, so every output must equal K rounded to
   BF16. The FP32 accumulator holds every K used here exactly; the BF16 *store*
   keeps only 8 significand bits, so 768, 2304, 3072 and 49152 come back exact
   while K = 50304 rounds to 50176 (it needs 9). The gate compares against the
   rounded value, so it stays an exact equality test and still catches a tile
   never written, written twice, or a truncated K loop.
2. **`--pattern-check 1`.** Operands are `{-1, 0, 1}` on the first and last four
   K indices and zero elsewhere. Outputs stay small integers, so equality still
   holds exactly, while values now vary along m, n *and* k — this is what
   catches swapped indices, wrong strides, and a transposed operand read with
   the wrong layout. The reference is the pattern's closed form recomputed in
   the checking kernel, sharing no code with the GEMM.

Run both. A case that fails either one has no meaningful timing.

## State of the dispatch (2026-07-25)

`_amd_dynamic_mfma_dispatch` in `matmul_ops.mojo` was originally tuned for GPT-2
*decode* shapes (M around 512), and at training shapes (M = 49152) it was both
wrong and slow. Both are recorded in `optimization_journal.md` under "nanoGPT
training-step MI300X journal"; the summary of what changed and what did not:

**The four wrong cases are fixed.** Two unrelated defects:

- `attn_c_attn_fwd`, `attn_c_proj_fwd`, `mlp_c_fc_fwd` (about 24.85% of outputs
  wrong each) hit a code-generation defect in MAX's two-stage
  `multistage_gemm_kernel` on gfx942 for `transpose_b=True`: one MFMA k-step's
  contribution is dropped from part of the accumulator, non-deterministically
  per workgroup. It reproduces at `K == BLOCK_K`, where the K loop performs no
  global-to-LDS prefetch and no race is possible, and disappears with three or
  four pipeline stages. `_amd_dynamic_mfma_gemm` now refuses, at compile time,
  every transposed-B geometry whose warp tile is one MMA wide in some dimension
  (`WM, WN >= 2 * mma_dim`) plus any whose B-copy row count does not divide BN.
- `lm_head_dgrad` (100% wrong, "short by exactly 128") was a gate defect, not a
  kernel defect: K = 50304 is not representable in BF16 and rounds to 50176.
  See the correctness-gates section above.

How far that miscompile reached, corrected after review: it is **not** confined
to `k < 2048`. On the pre-fix kernels `m = 1024, n = 2304, k = 3072`
transposed-B was also wrong (589824 of 2359296 outputs). The reachable set is
whichever branch of the old dispatch selected a warp tile one MMA wide, so it is
a property of the branch and geometry rather than a single interval in k. Most
consequentially, `m = 512, n = 768, k = 768` transposed-B was wrong, and that is
**GPT-2's own decode linear forward** — this defect silently corrupted GPT-2
*inference* on gfx942, not only training.

Two further defects came out of an adversarial review of that work and are also
fixed; see `optimization_journal.md` findings R1-R3.

- **R1, silent wrong answers.** Raising the deep-K route to four in-workgroup K
  partitions requires `(k / BLOCK_K) % partitions == 0`, because MAX splits K by
  floor division. At `BLOCK_K = 32` that is `k % 128`, but the dispatch only
  guarded `k % 32`, so every `k` that is 64 mod 128 in that regime was silently
  wrong — `k = 2112`, `2240`, `2880` all produced 100% wrong output and all were
  correct before the change. `k = 2880` is a production hidden size. The
  partition count now steps down to what the runtime `k` admits, and the same
  requirement is enforced on the FP32 route (`BLOCK_K = 16`, so `k % 64`).
- **R2, a 31-70% speed regression.** The `m >= 1024` large-macro-tile branch was
  taken before the deep-K branch could be considered, so K-dominant narrow-N
  shapes above m = 1024 lost the partitioned route. Measured crossover is
  m = 2048, not 1024. Fixed by yielding to the deep-K branch for
  `1024 <= m < 2048` when K dominates, and by choosing the partition count from
  the runtime output-tile count against the runtime CU count. The whole band is
  now within +-0.8% of its pre-change timings, with the m = 4096 gain (-28.7%)
  intact.

**Speed: per-step weighted total 506.9 -> 257.4 ms against ROCm's 71.6, ratio
7.076 -> 3.593.** Mojo now reaches 73-178 TFLOP/s where ROCm reaches 367-612.
What was done: a runtime-fill-driven macro tile (128x128x32 with an 8-warp
32x64 decomposition once the output-tile count covers every CU twice),
materializing B^T for m >= 1024 so the GEMM reads 256-byte global rows, four
in-workgroup K partitions in the deep-K regime, and a tiled LDS transpose that
took the strided-operand materialization from 141 to 12 ms/step.

**The remaining 3.6x is the data movement inside MAX's multistage core**, and it
is measured rather than guessed. On the best configuration, counters give 2.26
LDS instructions per MFMA and 4.87 bank-conflict cycles per LDS instruction.
The reason is in the source: with `transpose_b=False` the B LDS tile is
`(BK, BN)`, so each MFMA B fragment is four elements strided by BN and
`_load_b_amd` lowers it to four 2-byte reads -- 32 of the 38 LDS instructions
per warp per k-tile. The two operand layouts trade the two halves of the
problem and this kernel cannot have both: `(N,K)` gives k-contiguous LDS (one
`ds_read_b64` per fragment) but 64-byte global rows, `(K,N)` gives 256-byte
global rows but the 4x scalar fragment reads. Closing it needs a kernel that
transposes during the global-to-LDS store, plus a global split-K for the
weight-gradient shapes, whose 432-workgroup grids leave 0.71 waves per SIMD.
Neither is expressible around the stock kernel; both are new kernel work.

Two leads that were **closed** rather than taken:

1. Macro tiles above 128x128 (Tensile picks `MT256x224x64`, `MT256x256x32`,
   `MT192x256x32`, `MT128x512x32`, `MT512x128x64`) are all 13-60% *slower*
   here, and 256x256x32 at two pipeline stages does not launch at all: it
   requests the whole 64 KB gfx942 LDS budget.
2. MAX's own tuned AMD entry points are unreachable under the runtime-extent
   rule. `AMDMatmul` takes N and K from static shapes and bakes static strides
   into its loaders; ping-pong / 4-wave / 4-wave-split-K are CDNA4 (BF16 MMA
   16x16x32) and want 128 KB of LDS; `warp_specialized_matmul` takes M, N, K as
   compile-time parameters; and `_matmul_gpu` falls back to hipBLASLt when N/K
   are not static.

Note also that `_amd_dynamic_mfma_dispatch` returns `False` for `batch != 1`,
which is why attention's batched GEMMs fall back to the scalar-FFMA
`pure_gemm_tiled` kernel; that belongs to the SDPA target, not this one.

## State of the SDPA target (2026-07-25)

PyTorch-ROCm runs **one fused flash-attention kernel per direction** (AOTriton
`attn_fwd` and `bwd_kernel_fuse`) and never materializes the score matrix.
Measured standalone at the nanoGPT shapes: **8.158 ms/step forward, 28.966
ms/step backward** (679.80 and 2413.83 us per layer, 12 layers). In situ the
profile attributes 9.083 and 33.953.

The eager Mojo device instead runs a **math decomposition**, because the repo's
FA4 kernels are gated to H100: `_fa4_bf16_d64_causal_inputs` in `aten_fast.py`
requires `device.api == "cuda"` and `architecture_name == "sm_90a"`. On gfx942
`_sdpa_math_forward_with_dropout` runs `bmm(Q, K^T)` → `SoftmaxRows` →
`bmm(P, V)`, materializing a `[576, 1024, 1024]` BF16 score matrix — **1.208 GB
per layer**, twice (scores and probabilities). The backward adds four more
batched GEMMs, two transpose materializations, and an unfused softmax-backward
chain.

Three leads, in descending measured size.

**1. The batched GEMMs never touch the matrix cores.**
`_amd_dynamic_mfma_dispatch` opens with
`if batch != 1 or a_bstride == 0 or m < 64 or k % 32 != 0: return False`, so
every attention BMM falls through to the portable scalar-FFMA
`pure_gemm_tiled` kernel. `bench_attention_bmm.mojo` measures **167.06 ms/step
against `torch.bmm`'s 36.60 ms** (4.56x), at 32-36 TFLOP/s where Tensile
reaches 122-178. This is the largest single item and the harness for it is
ready.

**2. The fused softmax-backward kernel is FP32-only.**
`fast_sdpa_dropout_softmax_backward` in `aten_fast.py` returns `NOT_HANDLED`
unless `probs._dtype == DType.float32`, so this BF16 workload takes the
five-kernel fallback: broadcast multiply, row reduction, broadcast subtract,
multiply, scale. In the profile that is `logic_ops__bin_bcast_kernel` 50.7 ms +
`logic_ops__bin_flat_vec_kernel` 23.7 ms + `reduce_rows_block_bfloat16`
14.7 ms + `elementwise_r1_w1` 11.2 ms ≈ **100 ms/step**, against three passes
over a 1.208 GB tensor if fused.

This is not a flag to relax: the kernels in
`sdpa_dropout_softmax_backward_kernels.mojo` are written against
`UnsafePointer[Scalar[DType.float32]]` throughout, so the work is parameterizing
all three on dtype (keeping the FP32 accumulator for the row reduction, which
BF16 cannot carry) and widening the Python bridge with it. Contained, and worth
about 100 ms/step — the best value-per-line item in this target.

**3. Neither of the above is sufficient — the full-square decomposition cannot
reach 1.02.** Optimistic floor, using the *spec* 5.3 TB/s HBM bandwidth (real
bandwidth is lower, so the true floor is higher):

| component | floor | basis |
|---|---:|---|
| six batched GEMMs | 36.60 ms | measured `torch.bmm` |
| softmax forward | 5.5 ms | read scores + write probs, 29.0 GB |
| softmax backward, perfectly fused | 8.2 ms | read P, read dP, write dS, 43.5 GB |
| **total** | **50.3 ms** | vs PyTorch-ROCm's **37.1 ms** |

So a decomposition that materializes the whole square is stuck at about 1.36x
even with a perfect batched GEMM and a perfect fused softmax backward. Two ways
out, both real work:

- **Fuse.** MAX has a flash-attention forward with genuine AMD CDNA support in
  `/tmp/modular/max/kernels/src/nn/attention/gpu/mha.mojo` — `flash_attention`
  at line 337, with `has_amd_gpu_accelerator()` branches, "AMD bf16: 4 warps
  (256 threads) with 16x16 MMA", `BN = 128` for AMD, and a long-context CDNA
  prefill gate. Check before committing to it: whether its `MHAConfig` and
  `q_layout` can carry a runtime sequence length and batch (depth and head
  count as compile-time regimes with a runtime dispatch are acceptable under
  AGENTS.md rule 4; sequence length and batch are not), and whether it reaches
  any vendor library. There is **no flash-attention backward anywhere in MAX** —
  grep confirms it. The backward is the larger half (28.97 vs 8.16 ms), so a
  fused backward is new kernel work either way; the repo's own
  `eager_flash_attention/fa4_bwd_*.mojo` is a structural template but is written
  against H100 wgmma/TMA.
- **Exploit causality.** nanoGPT calls SDPA with `is_causal=True`, so half the
  score matrix is masked, and the decomposition currently computes, softmaxes
  and re-reads all of it. Skipping fully-masked blocks roughly halves every term
  above, giving a floor near 25 ms — below PyTorch-ROCm's 37.1 ms. That makes a
  causal-block-skipping decomposition viable in principle without full fusion.

If you implement a fused path, wiring it into production is part of the work:
the orchestration lives in Python (`_sdpa_math_forward_with_dropout` in
`aten_fast.py` and `_ScaledDotProductAttentionAutograd` in
`mojo_device/mojo_device_autograd.py`), so a fused Mojo kernel that nothing
calls is not a result. Extend the harness with the new route rather than
replacing the decomposed one, so both stay measurable side by side. The
`rocm_attention_targets.csv` whole-op numbers and the correctness gates are
frozen; the harness itself is yours to extend.

## Profiling

`scripts/rocprof_kernels.py` wraps `rocprofv3` and works identically on a Mojo
binary and a PyTorch script:

```bash
# per-kernel duration, grid/block, VGPR/SGPR, LDS, scratch
uv run --no-sync python scripts/rocprof_kernels.py -- \
    /tmp/bench_linear_gemm --case=mlp_c_fc_fwd

# hardware counters (MFMA issue, LDS conflicts, HBM traffic)
uv run --no-sync python scripts/rocprof_kernels.py \
    --pmc "SQ_INSTS_MFMA SQ_LDS_BANK_CONFLICT FETCH_SIZE WRITE_SIZE" -- \
    /tmp/bench_linear_gemm --case=mlp_c_fc_fwd

# what Tensile does for the same shape, for comparison
uv run --no-sync python scripts/rocprof_kernels.py --filter Cijk -- \
    /root/rocm-venv/bin/python /tmp/reference_gemm.py
```

Counter *values* are collected on an SR-IOV virtual function: treat them as
relative indicators between candidates. Counter *durations* are inflated by
serialization, so always time with `--kernel-trace` (the default) or with the
harness itself, never from a `--pmc` run.

`rocprof-compute` (ROCm's closest analogue to Nsight Compute, with roofline and
derived metrics) needs its own interpreter:

```bash
PATH=/root/rocprof-compute-venv/bin:/opt/rocm/bin:$PATH rocprof-compute --version
```

## Rules that bind any change here

`AGENTS.md` is authoritative; the parts that bite hardest on this work:

- No vendor libraries. No hipBLASLt, rocBLAS, Tensile, MIOpen, or anything that
  only exists in a GPU PyTorch install. `pip install torch` (CPU) plus this
  package must use the GPU on its own.
- Tensor extents stay **runtime** values. Tile and pipeline regimes may be
  compile-time; M, N, K, and strides may not. Several dispatch regimes selected
  by runtime shape are correct; a kernel specialized to 49152 x 2304 x 768 is
  not, and neither is a fast path plus a slow fallback for everything else.
- Do not read refcounts, cache history across calls, or write into input buffers
  unless the ATen signature says to.
- Do not allocate your own pool or arena; ordinary allocation only.
- Python extension modules cannot be part of the answer: no C++ PyTorch API.
- Type-hint every Python function you touch, arguments and return.
