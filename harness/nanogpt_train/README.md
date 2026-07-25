# nanoGPT training-step kernel harnesses (MI300X, gfx942)

Frozen workload: nanoGPT's GPT-2 124M (12 layers, 12 heads, 768 embedding,
`bias=False`, `vocab_size=50304`), batch 48, block 1024, BF16 autocast, fused
AdamW, gradient clipping 1.0, eager execution, no `torch.compile`.

Measured on an MI300X VF with PyTorch 2.11.0+rocm7.2 and MAX
26.5.0.dev2026072306:

| backend | step | tokens/s |
|---|---:|---:|
| PyTorch-ROCm | 156.63 ms | 313 805 |
| eager Mojo device | 921.39 ms | 53 346 |

Both backends are GPU-bound (trace idle 0.01%), so GPU kernel time is the unit
of comparison. Ranked gap per step, from
`current_bench_train/comparison/nanogpt_train_kernel_gap.csv`:

| target | mojo | rocm | ratio | share of gap |
|---|---:|---:|---:|---:|
| Linear GEMM (backward) | 346.89 ms | 43.65 ms | 7.95x | 39.3% |
| SDPA (backward) | 239.24 ms | 33.95 ms | 7.05x | 26.6% |
| Linear GEMM (forward) | 171.84 ms | 24.03 ms | 7.15x | 19.2% |
| SDPA (forward) | 105.10 ms | 9.08 ms | 11.57x | 12.4% |
| Copies / dtype casts | 26.00 ms | 11.44 ms | 2.27x | 1.9% |
| everything else | 33.28 ms | 34.87 ms | 0.95x | 0% |

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

## Known defects in the current dispatch (2026-07-24)

`_amd_dynamic_mfma_dispatch` in `matmul_ops.mojo` was tuned for GPT-2 *decode*
shapes (M around 512). At training shapes (M = 49152) it is both wrong and slow.

**Four of fifteen GEMMs compute wrong results.** These are silent: nanoGPT still
trains and reports a plausible loss.

| case | m | n | k | ta | tb | wrong outputs | selected config |
|---|---:|---:|---:|:-:|:-:|---:|---|
| `attn_c_attn_fwd` | 49152 | 2304 | 768 | 0 | 1 | 24.85% | `BM=32 BN=64 WM=16 WN=32 BK=32` |
| `attn_c_proj_fwd` | 49152 | 768 | 768 | 0 | 1 | 24.83% | same |
| `mlp_c_fc_fwd` | 49152 | 3072 | 768 | 0 | 1 | 24.85% | same |
| `lm_head_dgrad` | 49152 | 768 | 50304 | 0 | 0 | 100%, short by exactly 128 of K | `BM=32 BN=32 WM=32 WN=32 BK=32 WARP_K=2` |

The 24.85% figure is close to one quarter, and the failing config has exactly
four warps per block ((BM/WM) x (BN/WN) = 2 x 2), which points at one warp's
output. `lm_head_dgrad` being short by exactly 128 = 2 x BK x WARP_K points at a
K-tail or pipeline-drain bound. Both are hypotheses, not conclusions.

**Every GEMM is far off the ROCm reference**, even the correct ones: Mojo
reaches 38-149 TFLOP/s where ROCm reaches 367-612 TFLOP/s.

Two structural leads, both visible in the harness output:

1. **The transposed operand is materialized.** `fast_aten_linear_backward`
   builds `grad_output.transpose(0, 1)` as a view, and
   `_matmul_spec_operands_launch` sees a non-contiguous operand and calls
   `_scratch_contig` -> `_copy_strided`. That copy costs **141 ms/step**, 28% of
   all Mojo GEMM time, and PyTorch-ROCm pays none of it: Tensile reads the
   strided view directly. `lm_head_wgrad` alone spends 73 ms of its 98 ms in the
   copy. `bf16_gemm_tn_v4_kernels.mojo` already exists but is gated to
   `sm_90a`; the AMD dispatch has no `transpose_a` parameter at all.
2. **The macro tiles are far smaller than Tensile's.** The AMD dispatch never
   selects more than `128x128x64`, and picks `32x64x32` or `32x32x32` for most
   training shapes. For the same GEMMs Tensile selects `MT256x224x64`,
   `MT256x256x32`, `MT192x256x32`, `MT128x512x32`, `MT512x128x64` with
   `MI16x16x1`. At M = 49152 there is abundant parallelism, so small tiles buy
   nothing and cost arithmetic intensity.

Nothing here is a required solution — they are the measurements. Note also that
`_amd_dynamic_mfma_dispatch` returns `False` for `batch != 1`, which is why
attention's batched GEMMs fall back to the scalar-FFMA `pure_gemm_tiled` kernel;
that belongs to the SDPA target, not this one.

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
