# Optimization backlog

The standing list of kernels and code paths in this repository that are **not
fully optimized**. It is an audit deliverable, not a plan: every entry names the
file and line it is about, says what runs today, why that is not the best we can
do, what the finished version looks like, and how the next person measures it.

**Audited**: 2026-08-01, branch `kernel-call-queue`, working tree with the
in-flight review fixes applied (`git status` dirty). Read-only source audit plus
one host query (`torch.get_float32_matmul_precision()`). Machine: 1x NVIDIA H100
PCIe (`sm_90a`), PyTorch 2.11.0+cu130.

**Line numbers drift.** They are from the audit snapshot, and several of these
files were being edited concurrently while this was written. Every citation also
names the symbol; if a line does not match, grep the name.

**Honesty rules used here** (AGENTS.md "no silent caps" spirit):

* Every number is labelled with where it came from. Numbers taken from
  `optimization_journal.md` were measured on **MI300X (gfx942)** unless stated,
  and do not transfer to NVIDIA or Apple.
* Anything I did not measure and could not find a measurement for is written as
  **UNMEASURED**. There are a lot of those and that is the honest state.
* [Not audited](#not-audited) lists what this pass did **not** cover.

**Constraints every proposal here respects** (AGENTS.md): no CuBLAS / CuDNN /
RocBLAS / vendor BLAS of any kind; flexible kernels with runtime dispatch, never
shape-hardcoded kernels plus a fallback; never "change the model"; eager ops may
read only shape, stride, dtype and pointer, never refcounts.

---

## Priority table

Impact and effort are estimates. "Impact" is about the workloads this repo
benchmarks (nanoGPT training, GPT-2 decode, resnet/vgg forward) unless the row
says otherwise. **Every item in this document appears in this table**, roughly
ordered by estimated value; the tail below the `G9` row is the lower tier, not a
different kind of item.

| # | Item | Area | Est. impact | Est. effort | Blocked by |
|---|---|---|---|---|---|
| [G1](#g1) | Python contiguity gate strands the Mojo strided-operand matmul routes (incl. the gfx942 copy-free TN route and the Apple TA route) | GEMM | **High** on gfx942 (3.5–23.2% per weight-gradient GEMM, measured); unknown elsewhere | Low (Python only) | — |
| [G2](#g2) | No fp32 GEMM reads a strided A; the new backstop copies instead | GEMM | High for fp32 training | High (new kernel) | G1 should land first |
| [G3](#g3) | Default `float32_matmul_precision="highest"` turns the TF32 bridge off, so default fp32 matmul is SIMT FFMA | GEMM | High for fp32 users on sm_90a | Medium (needs an fp32-exact tensor-core path, not a gate relaxation) | — |
| [G4](#g4) | Tensor-core GEMM exists only for `sm_90a` and `gfx942` (+ Apple simdgroup); every other GPU runs SIMT FFMA | GEMM | **High** on A100 / Ada / Blackwell / other CDNA / RDNA | High | hardware access |
| [A1](#a1) | SDPA outside the two fused gates materializes the full `[B*H, Tq, Tkv]` probability matrix and saves it for backward | Attention | **High** (memory *and* time) off gfx942 / sm_90a-bf16-d64 | High | — |
| [A3](#a3) | Fused SDPA backward is Apple-only; CUDA/ROCm run the composed sequence with two extra full transposes for dV | Attention | Medium–High | Medium | — |
| [D1](#d1) | The hot rank≤4 materialize path lacks the 16-byte vectorized transpose the rank≤8 path already has | Data movement | Medium (2.1–2.4 → 3.57 TB/s on the transposes, gfx942-measured) | **Low** (port an existing kernel) | — |
| [P1](#p1) | Warm-path regression from the variant loader: 101 ms/step vs 60 ms on `main` | Compile pipeline | **High** — the largest single eager regression currently on record | Medium | — |
| [R1](#r1) | ~~Non-trailing reductions materialize a whole permuted copy of the input~~ **DONE**: one accumulator-parametrized skeleton over (outer, reduce, inner), strided-axis kernel for every scalar reduction | Reductions | Medium | Medium | — |
| [O2](#o2) | ~~Five `_foreach_*` batched kernels are Metal-only; CUDA/ROCm fall back to one launch per tensor~~ **DONE**: one descriptor-batched kernel body for the whole elementwise family, chunk size sized from the device | Optimizer | Medium (non-fused `AdamW(foreach=True)`) | Low–Medium | — |
| [C2](#c2) | Grouped convolution issues `N × groups` GEMM launches from a Python loop | Conv | Medium for depthwise/grouped nets | Medium | — |
| [C1](#c1) | Convolution is im2col + GEMM with a full `(N, C·kh·kw, out_h·out_w)` temporary | Conv | Medium (memory-bound at large spatial) | High | no conv benchmark exists |
| [N1](#n1) | LayerNorm backward is fp32-only and GPU-only; the forward accepts fp16/bf16 | Normalization | Medium (blocks pure-bf16 training) | Medium | — |
| [D3](#d3) | ~~Fused 3-way `cat` is Metal-only~~ **DONE**: batched N-input `CatN` | Data movement | Low–Medium | Low | — |
| [E1](#e1) | Divide-by-scalar has no scalar spec: costs an extra 0-d fill allocation + launch | Elementwise | Low–Medium (very high call count) | **Low** | — |
| [D2](#d2) | The generic rank≤8 strided copy is a scalar kernel with an 8-deep div/mod chain and no dimension collapsing | Data movement | Medium for rank>4 workloads | Medium | — |
| [X1](#x1) | Launch-geometry constants fitted to one card and applied to all (`_LLC_BYTES`, `TARGET_BLOCKS`, `_TARGET_BLOCKS`) | Cross-cutting | UNMEASURED off gfx942 | Low–Medium | needs the other cards |
| [G6](#g6) | The medium-M fp32 SIMT path allocates an `m×k` transpose scratch on every call | GEMM | UNMEASURED | Medium | — |
| [N4](#n4) | RMSNorm has no fused op in either backend | Normalization | Medium for modern LLMs | Medium | — |
| [A5](#a5) | `attn_mask` is a hard decline in eager SDPA | Attention | High for HF models | Medium | — |
| [D5](#d5) | `aten::index` only handles a single index tensor on dim 0 | Data movement | Medium | Medium | — |
| [R2](#r2) | ~~`linalg_vector_norm` is composed: 3 launches + an input-sized temporary~~ **DONE**: `NormSpec` / `NormL2Op`, one pass | Reductions | Low | Low | — |
| [C3](#c3) | eager conv backward rides the materialized im2col route: 1.4-23x cuDNN; forward still 2-D-only (no conv1d/3d/transposed) | Conv | High for vision *training* | High | — |
| [N2](#n2) | BatchNorm training and GroupNorm backward are absent in eager | Normalization | High for vision training (blocks it) | Medium | — |
| [Q1](#q1) | Graph backend hand-decomposes softmax / log_softmax instead of using MAX's fused ops | Graph | Low–Medium | **Low** | verify MAX does not already re-fuse |
| [Q3](#q3) | Graph `max_pool2d_with_indices` returns the values as the indices | Graph | Correctness bug | Low | — |
| [P2](#p2) | Strict-FIFO call queue: one cold variant blocks every warm launch behind it | Compile pipeline | Cold-start only | Medium | — |
| [P3](#p3) | Ops whose Mojo gates are not mirrored in Python drain the queue on every call | Compile pipeline | Low–Medium | Low per op | — |
| [D4](#d4) | `nonzero` round-trips through the host | Data movement | Low (rare op, but a full drain) | Medium | — |
| [E4](#e4) | GELU forward has ~0.3 ms/step of ALU above its memory bound (gfx942-measured) | Elementwise | Low | Medium | — |
| [G9](#g9) | The last 5.3% of every gfx942 GEMM is a tile-count / CU-count residue; only Stream-K reaches it | GEMM | ~0.45 ms/step on gfx942 (modelled) | High | — |
| [A2](#a2) | FA4's gate is bf16 + `head_dim == 64` + causal + `seq % 128 == 0` only | Attention | Medium (Llama-class `head_dim = 128`, GQA) | Medium | — |
| [A4](#a4) | The causal batched GEMM that skips masked contraction indices is gfx942-only | Attention | UNMEASURED on NVIDIA | Medium | — |
| [A6](#a6) | The fused decode kernel declines `kv_len > 4096` and causal/multi-token decode | Attention | Medium for long-context decode | Medium | — |
| [O1](#o1) | Fused AdamW is fp32-homogeneous only, and raises instead of declining | Optimizer | Blocks bf16 master weights | Medium | — |
| [O3](#o3) | `clip_grad_norm_` (`_foreach_norm` / `_foreach_mul_.Tensor`) is fp32-only | Optimizer | Low–Medium | Low | — |
| [E2](#e2) | `native_dropout` is fp32 + GPU only | Elementwise | Blocks bf16/fp16 dropout | Low | — |
| [G7](#g7) | `addmm` declines the fused-bias bridges for `beta`/`alpha` != 1; rank>2 linear needs full contiguity | GEMM | UNMEASURED, workload-dependent | Low–Medium | — |
| [C4](#c4) | Pooling has no `ceil_mode` and no backward | Conv | Blocks torchvision configs and vision training | Low–Medium | — |
| [D7](#d7) | rank>4 binary elementwise operands are pre-materialized | Data movement | UNMEASURED | Medium | D2 would subsume it |
| [D6](#d6) | `_LLC_BYTES` = MI300X Infinity Cache used as the residency threshold everywhere | Data movement | UNMEASURED off gfx942 | Low | part of X1 |
| [G5](#g5) | `TARGET_BLOCKS = 342` hardcoded split-K target for every non-Apple GPU | GEMM | UNMEASURED off H100/MI300X | Low | part of X1 |
| [N3](#n3) | `_TARGET_BLOCKS = 1280` is an unlabelled fitted constant | Normalization | UNMEASURED | Low | part of X1 |
| [R3](#r3) | `cumsum` is trailing-dim only, 3 dtypes, refuses `dtype=` | Reductions | Coverage | Low–Medium | — |
| [E3](#e3) | fp64 declines in every structured op (norm/conv/pool/SDPA/matmul) | Elementwise | Coverage | Medium | — |
| [G8](#g8) | No fp64 GEMM at all | GEMM | Coverage | Medium | part of E3 |
| [Q2](#q2) | Graph GroupNorm is reimplemented and returns `NotImplementedError` objects for mean/rstd | Graph | Low + a latent hazard | Low | — |
| [Q4](#q4) | Graph SDPA with `attn_mask` decomposes to full-matrix attention | Graph | Medium for HF models | Medium | — |
| [Q5](#q5) | Graph conv/pool insert explicit NCHW↔NHWC permutes | Graph | UNMEASURED, possibly zero | Low | verify MAX does not elide them |
| [P4](#p4) | Cold start is 11.6 s and `__mojocache__` cannot be shipped | Compile pipeline | First-run only | Low | — |

---

## GEMM / matmul

### G1
**The Python contiguity gate strands every strided-operand route the Mojo matmul kernel already has**

* **What.** `_spec_matmul_out_shape` and `_try_spec_matmul`,
  `torch_mojo_backend/eager_kernels/aten_fast.py:1876` and `:1993`, versus
  `_matmul_spec_operands_launch`,
  `torch_mojo_backend/eager_kernels/matmul_ops/matmul_ops.mojo:6793`.
* **Current implementation.** The Mojo side of the matmul spec family is fully
  layout-aware. `_matmul_spec_operands_launch` has four arms — both operands
  contiguous, strided B, strided A, both strided — and inside the strided-A arm
  there are two *copy-free* routes: `_apple_ta_spec_route`
  (`matmul_ops.mojo:6755`) and, for gfx942 bf16, `_tn_mfma_route`
  (`matmul_ops.mojo:3353`, taken at `matmul_ops.mojo:6857-6867` when A's two
  innermost strides are `(1, m)`, B is contiguous and there is no bias). Only if
  those decline does it build a Mojo-side scratch copy.

  The Python side never lets any of that run. `_spec_matmul_out_shape` returns
  `None` unless every operand is contiguous (`aten_fast.py:1893`), which makes
  `_MatmulSpecExtension.expected_output_specs` raise, which makes
  `_submit_spec_matmul` (`aten_fast.py:1975`) return `None`. That gate arrived
  with commit `1d82629` ("Into launches for every remaining spec family");
  before it, `_try_spec_matmul` passed the raw specs — strides and all —
  straight to `MatmulSpec`. The correctness backstop in the current working tree
  (`aten_fast.py:2009-2036`) restores *correctness* by calling `_tc()` in
  Python, i.e. it now always copies, on every architecture.
* **Why it is not optimal.** Three separate losses. (a) The gfx942 copy-free TN
  route — the one that made the weight-gradient GEMMs 3.5–23.2% faster, measured
  shape by shape in `optimization_journal.md` "Change 29" — is dead code
  reachable from no Python caller. (b) The Apple TA route is dead the same way.
  (c) Where a copy really is needed it is now made in Python (an extra
  `TorchMojoTensor`, an extra queue item, an extra keep-alive) instead of in the
  Mojo scratch arm written for it. All three arms of
  `_matmul_spec_operands_launch` after the first are currently unreachable.
* **What the optimized version looks like.** `_spec_matmul_out_shape` keeps a
  contiguity requirement only for the *queued* eligibility answer where a launch
  genuinely cannot fall back, and grows an explicit predicate for the layouts the
  Mojo side is known to accept (dense row-major, or physically transposed with
  innermost strides `(1, ld)`). Operands matching that predicate go through
  untouched and `_matmul_spec_operands_launch` decides; everything else keeps the
  Python backstop. Settle it with a test that asserts a transposed-A `mm` reaches
  the Mojo path, not by inspection.
* **Expected win.** On gfx942 bf16 weight gradients: 3.5–23.2% per launch,
  measured (`optimization_journal.md`, Change 29 table). On Apple the TA route's
  value is **UNMEASURED** here. On NVIDIA it removes one full read+write of A per
  strided GEMM; magnitude **UNMEASURED**.
* **How to measure it.** `uv run --no-sync python bench_gemm.py` for the
  shape-exact GEMM cases, and
  `uv run --no-sync python bench_nanogpt_train.py --device mojo` with
  `profile_nanogpt_train_aten.py` / `compare_nanogpt_train_kernels.py` for
  per-kernel attribution of a whole training step. On gfx942,
  `scripts/rocprof_kernels.py --kernel-trace` gives per-kernel durations
  directly.

### G2
**No fp32 GEMM reads a strided A, so fp32 `nn.Linear` backward pays a materialization**

* **What.** `fast_aten_linear_backward`, `aten_fast.py:7295` (weight gradient
  formed at `:7371`), landing in `fast_aten_mm`, `aten_fast.py:7209`, and the
  backstop at `aten_fast.py:2009-2036`.
* **Current implementation.** `fast_aten_linear_backward` forms the weight
  gradient as `mm(transpose(grad, 0, 1), input)`. The transpose is a zero-copy
  view, so `mm` receives a non-contiguous A. Three bridges are tried:
  `_try_gemm16_mm` (`:6861` — bf16/f16 only, and it *does* accept a physically
  transposed 2-D operand via `_tf32_dense_2d_layout`), `_try_tf32_gemm` (`:6935`,
  off at default precision — see [G3](#g3)), and `_try_spec_matmul`. For fp32 all
  three decline and the backstop materializes A.
* **Why it is not optimal.** A full read + write of the gradient matrix, per
  Linear, per step, purely to change layout. bf16 autocast never pays it (the
  bf16 bridge takes the transposed operand directly), which is exactly why every
  benchmark in this repo stayed green while plain fp32 training aborted the
  process — the abort is fixed by the backstop, the copy is not.
* **What the optimized version looks like.** A layout-flag fp32 GEMM: one
  compile-time `TRANSPOSE_A` arm over the existing dynamic tile machinery,
  selected at runtime from the operand's strides, exactly the way `TRANSPOSE_B`
  already is. `matmul_ops.mojo` already carries a `TRANSPOSE_A` parameter on the
  Apple and split kernels (`:4781`, `:5004`, `:5188`); the work is extending it
  to the portable fp32 tiles and to whichever tensor-core path the architecture
  has. Shapes stay runtime (AGENTS.md rule 4).
* **Expected win.** Removes one `m×k` fp32 read+write per Linear backward.
  Magnitude **UNMEASURED**; a bandwidth model puts it at roughly one extra pass
  over the gradient, which for nanoGPT-scale shapes is a few percent of the GEMM
  and more for short-K shapes.
* **How to measure it.** Add fp32 cases to `bench_gemm.py` (it reads its cases
  from CSV, so this is a data change), then
  `bench_nanogpt_train.py --device mojo --dtype float32` against `--device cuda`.
  A tight repro is three lines timing `nn.Linear(...).float()` forward+backward
  on `mojo:0`.

### G3
**The TF32 bridge is gated off at PyTorch's default `float32_matmul_precision`**

* **What.** `_try_tf32_gemm`, `aten_fast.py:6948`; `_try_tf32_bmm`,
  `aten_fast.py:7083`; `_try_tf32_linear`, `aten_fast.py:7179`. All three open
  with `if torch.get_float32_matmul_precision() == "highest": return None`.
* **Current implementation.** Verified on this host: the default is `"highest"`.
  So on an out-of-the-box install the TF32 bridge — the only sm_90a fp32 route
  that uses tensor cores *and* accepts arbitrary dense 2-D layouts — never runs.
  fp32 matmul falls to `MatmulSpec` → `_gemm_dtype_dispatch`
  (`matmul_ops.mojo:5854`) → the portable SIMT/FFMA pipe tiles.
* **Why it is not optimal.** The default-precision fp32 user gets no tensor-core
  path at all on an H100, and additionally loses the layout flexibility that
  causes [G2](#g2). The gate itself is correct and must not simply be removed:
  TF32 drops mantissa bits and `"highest"` is the user asking for full fp32.
* **What the optimized version looks like.** Two independent pieces. (1) Keep the
  gate and give `"highest"` its own tensor-core path — 3xTF32 / split-float
  emulation (three TF32 products reconstructing near-fp32 accuracy) is the
  standard technique and is expressible with Mojo `mma`; it belongs behind the
  same runtime dispatch as everything else. (2) Independently, make the *SIMT*
  fp32 path competitive, since it is also what every non-sm_90a NVIDIA card uses
  ([G4](#g4)).
* **Expected win.** **UNMEASURED.** Upper bound is the ratio between the FFMA
  SIMT tiles and a tensor-core path at the same shapes; on H100 that ratio is
  large (TF32 tensor-core throughput is ~8x the FP32 FMA rate), but 3xTF32 costs
  three products, so the realistic ceiling is ~2–3x on compute-bound shapes and
  nothing on memory-bound ones.
* **How to measure it.** `bench_gemm.py` with fp32 cases, comparing
  `torch.set_float32_matmul_precision("highest")` against `"high"` on the same
  shapes — that A/B isolates exactly this bridge today.

### G4
**Tensor-core GEMM exists only for `sm_90a` and `gfx942`; every other GPU runs the SIMT path**

* **What.** Host gates: `aten_fast.py:6880` and `:7029` (`architecture_name !=
  "sm_90a"` for the bf16 GEMM and BMM bridges), `:6962` / `:7095` (same for
  TF32). Device gates: `matmul_ops.mojo:4025` (`_accelerator_arch() ==
  "amdgpu:gfx942"` inside `_gemm_enqueue`, `:4007`), `:6498` (gfx942 fused-bias
  MFMA inside `_matmul_bias_run`, `:6416`), `:4056` (Apple simdgroup kernels).
  Everything else lands on `_gemm_dtype_dispatch` (`matmul_ops.mojo:5854`) and
  the `_tune_enqueue` / `_pipe3t_enqueue` FFMA tiles.
* **Current implementation.** Three architectures have hand-written
  matrix-engine GEMMs; all others share one portable SIMT implementation that
  the source itself describes as "CUTLASS-SIMT-style"
  (`matmul_ops.mojo:5620-5622`).
* **Why it is not optimal.** A100 (`sm_80`), Ada (`sm_89`), Blackwell
  (`sm_100`/`sm_120`), MI250 (`gfx90a`), MI355 and every RDNA part run fp32 and
  bf16 matmul with no matrix engine at all. For bf16 this is order-of-magnitude
  territory: the SIMT path issues no `mma`/`MFMA`/`WMMA`.
* **What the optimized version looks like.** One `mma`-based dtype-parametric
  core keyed on *capability* predicates rather than exact architecture strings,
  with the hand-tuned `sm_90a` and `gfx942` routes kept as specializations on
  top. `tf32_gemm_kernels.mojo:62` already demonstrates the capability style
  (`is_nvidia_gpu() and _is_sm_9x_or_newer()`); the architecture-string equality
  checks are the ones to replace.
* **Expected win.** **UNMEASURED** — none of those parts is in this machine.
  Large by inspection for bf16.
* **How to measure it.** `bench_gemm.py` on the target card. Before touching
  anything, `uv run python scripts/compare_kernel_asm.py --before <baseline>
  --after . --accelerator sm_80` (and `gfx90a`, `sm_100`) to prove which
  architectures a change moves — and read the host dispatch by hand as well,
  because launch geometry does not appear in an assembly diff (AGENTS.md).

### G5
**`TARGET_BLOCKS = 342` is a hardcoded blocks-in-flight target for every non-Apple GPU**

* **What.** `matmul_ops.mojo:152`.
* **Current implementation.** `comptime TARGET_BLOCKS = 80 if
  has_apple_gpu_accelerator() else 342`. It drives the split-K factor for the
  portable SIMT GEMM. 342 = 3 × 114 (H100 SM count); MI300X has 304 CUs.
* **Why it is not optimal.** A compile-time constant standing in for a runtime
  device property. On a 46-SM RTX 2000 Ada or a 108-SM A100 the split-K heuristic
  over- or under-splits, and every extra shard costs an `m*n` fp32 workspace
  write plus a read-back in the reduce pass (the comment there says so).
  `op_utils/__init__.mojo:179` already shows the right pattern
  (`ctx.default_device_info.sm_count`).
* **What the optimized version looks like.** Derive the target from `sm_count` at
  launch, keeping the Apple arm's much lower multiplier — the comment records a
  real reason for it (few cores, expensive partials).
* **Expected win.** **UNMEASURED**; zero on H100 and MI300X by construction,
  possibly material on smaller parts.
* **How to measure it.** `bench_gemm.py` on a part whose SM count is far from
  114/304. This is a *host-side* change: `scripts/compare_kernel_asm.py` will
  report no kernel difference, which is precisely the trap AGENTS.md warns about.

### G6
**The medium-M fp32 SIMT path allocates an `m×k` transpose scratch on every call**

* **What.** `_pipe3t_enqueue`, `matmul_ops.mojo:5622` (buffer creation and the
  `transpose_small` launch immediately below it).
* **Current implementation.** To get both operand slabs streaming via `cp.async`,
  the (small) activation A is transposed into a freshly created device buffer,
  then the GEMM runs.
* **Why it is not optimal.** One allocation and one full read+write of A per
  call, on a path that exists precisely because the shapes are latency-sensitive.
  It is the same class of work [G2](#g2) is trying to delete elsewhere.
* **What the optimized version looks like.** A `TRANSPOSE_A` arm of the pipe
  kernel reading A from its native layout (the gfx942 MFMA core already has one,
  `_tn_mfma_route`), keeping the copy only where the strides genuinely cannot be
  expressed.
* **Expected win.** **UNMEASURED.** Bounded by one pass over A, which in the
  medium-M regime is the smaller operand — likely low single-digit percent, so
  confirm before spending effort.
* **How to measure it.** `bench_gemm.py` restricted to medium-M fp32 cases, plus
  `ncu` / `scripts/rocprof_kernels.py` to watch the `transpose_small` kernel
  disappear.

### G7
**Bridge decline conditions that push a whole GEMM onto the slow path**

* **What.** `fast_aten_addmm`, `aten_fast.py:7220`; bias validation in
  `_try_gemm16_mm` / `_try_tf32_gemm`, `aten_fast.py:6893` and `:6975`; the
  rank>2 contiguity requirement in `_try_gemm16_linear` / `_try_tf32_linear`,
  `aten_fast.py:7150` and `:7188`.
* **Current implementation.** `addmm` declines the fused-bias bridges entirely
  unless `beta == 1 and alpha == 1`. The bridges require the bias to be a
  contiguous 1-D tensor of exactly `(n,)` in the operand dtype. A rank>2 linear
  input must be fully contiguous or it loses the bridge (and then, per
  [G1](#g1), loses everything else too).
* **Why it is not optimal.** `beta`/`alpha` are two scalars the epilogue could
  fold for free; declining on them sends the whole GEMM to a different
  implementation. The rank>2 contiguity requirement drops any activation that is
  a slice or a permuted view out of the tensor-core path.
* **What the optimized version looks like.** `alpha`/`beta` as *runtime* scalars
  in the existing bias epilogue — they do not select different code, so per
  `docs/kernel_call_queue.md` they must be runtime data, not defines. For rank>2,
  flatten the leading dims whenever the trailing two are dense, which is strictly
  weaker than full contiguity.
* **Expected win.** **UNMEASURED.** Coverage-shaped: zero for workloads that
  never hit these, large for the ones that do (`beta`/`alpha` appear in attention
  variants and some HF layers).
* **How to measure it.** `tests/test_eager_kernels.py` for correctness; a script
  timing `torch.addmm(bias, a, b, beta=0.5)` against `beta=1.0` on `mojo:0` shows
  the cliff directly.

### G8
**No fp64 GEMM at all**

* **What.** `comptime FLOAT_DTYPES = [float32, float16, bfloat16]`,
  `torch_mojo_backend/eager_kernels/op_utils/__init__.mojo:27`, consumed by
  `_gemm_dtype_dispatch` (`matmul_ops.mojo:5854`). Python-side `_FLOAT_DTYPES`
  (`aten_fast.py:222`) likewise excludes `float64`.
* **Current implementation.** fp64 elementwise works (`_SPEC_FLOAT_DTYPES`,
  `aten_fast.py:1550`, includes it); fp64 matmul does not.
* **Why it is not optimal.** A scientific-computing user hits a hard
  `NotImplementedError` on `a @ b`. A coverage hole rather than a speed hole, but
  it lives in the GEMM family.
* **What the optimized version looks like.** fp64 added to the CPU library
  dispatch (which already has a general matmul) and to the portable SIMT tiles;
  GPU fp64 needs the Apple guard treatment (`_parallel_for_dt`,
  `op_utils/__init__.mojo:478`) because Metal has no fp64.
* **Expected win.** N/A (coverage).
* **How to measure it.** A `torch.float64` case in
  `tests/test_aten_functions.py`.

### G9
**The residual 5.3% of every gfx942 GEMM**

* **What.** Recorded in `optimization_journal.md`, "What is left, and why the
  tile shape cannot reach it".
* **Current implementation.** Macro tiles with extents that are multiples of 64
  give these shapes tile counts of the form `2^a·3^b`, never divisible by 19, and
  `304 = 16 × 19` — so the last wave is always short by exactly 1/19.
* **Why it is not optimal.** ~5.3% of every affected launch, structurally
  unreachable by any tile-shape search.
* **What the optimized version looks like.** Stream-K: persistent workgroups
  splitting the k-loop across an exactly-device-sized grid, plus a fixup
  reduction. The journal prices it: fixup traffic is `nsplits · BM · BN · 4`
  bytes, ~155 MB for 304 workgroups of a 256×256 tile, ~38 µs at 4.05 TB/s. It
  pays only above ~720 µs per launch — 3 of the 75 GEMM launches in a nanoGPT
  step, of which 2 have the fill to gain; the other 72 would lose.
* **Expected win.** ~0.45 ms/step on gfx942, **modelled, not measured**
  (`lm_head_dgrad` +288 µs, `lm_head_wgrad` +164 µs).
* **How to measure it.** `bench_nanogpt_train.py --device mojo` on MI300X with
  `scripts/rocprof_kernels.py --kernel-trace`, gating Stream-K on the same
  per-launch threshold the model derives.

---

## Attention

### A1
**Every SDPA outside the two fused gates materializes the full probability matrix**

* **What.** `fast_aten_scaled_dot_product_attention`, `aten_fast.py:7578`;
  `_sdpa_math_forward_with_dropout`, `aten_fast.py:5471`;
  `_ScaledDotProductAttentionAutograd`,
  `torch_mojo_backend/mojo_device/mojo_device_autograd.py:266`.
* **Current implementation.** Four tiers in order. FA4
  (`_fa4_bf16_d64_causal_inputs`, `aten_fast.py:5737`) requiring `api == "cuda"`,
  `sm_90a` (`:5763-5764`), bf16 on all three operands, rank 4,
  `is_causal is True`, `dropout_p == 0`, no mask, `seqlen % 128 == 0` and
  `head_dim == 64` (`:5774`). Then the fused gfx942 kernels (`_fused_fa_inputs`,
  `aten_fast.py:6066`) requiring `api == "hip"` and `gfx942` (`:6096-6097`).
  Then a decode special case (`AttnDecodeSpec`, `aten_fast.py:7654`). Then the
  math decomposition, which allocates `scores` and `probs`, each
  `(B·H, Tq, Tkv)`, and hands `probs` to autograd to save.
* **Why it is not optimal.** Two `O(B·H·T²)` tensors allocated and written per
  layer per step, plus the saved one held for the whole backward. At nanoGPT 124M
  / batch 48 / block 1024 the journal measures the decomposition at 29.854 ms
  forward against the fused kernel's 6.100 (gfx942) — ~5x, before counting the
  memory pressure. On an H100 every fp16 model, every fp32 model, every
  `head_dim != 64`, every non-causal prefill and every seqlen not a multiple of
  128 lands here.
* **What the optimized version looks like.** One dtype-parametric,
  head_dim-parametric flash-attention forward+backward with runtime dispatch on
  head_dim regime and mask mode, replacing the *fallback* rather than adding a
  fourth gate. `flash_attention_fwd_kernels.mojo` already declares the right
  contract (runtime shapes and strides, per-operand `RowStrides`, no allocation)
  and already contains `_flash_attention_fwd_baseline`
  (`flash_attention_fwd_kernels.mojo:268`) as the general path — but that
  baseline is one block per query row and is only reachable on gfx942 with
  `head_dim % 8 != 0`, so in practice it has never been what runs.
* **Expected win.** gfx942-measured 29.854 → 6.100 ms forward and 46.153 →
  28.896 ms backward at that one workload. The equivalent H100 number for the
  shapes FA4 declines is **UNMEASURED**.
* **How to measure it.** `bench_nanogpt_train.py` with
  `--dtype float16`/`float32` (which forces the decomposition on both backends),
  plus `profile_nanogpt_train_aten.py` and `compare_nanogpt_train_kernels.py` —
  the latter exists specifically to attribute the SDPA autograd node's GPU time,
  which the profiler records as a `user_annotation` rather than an `aten::` row.

### A2
**FA4's eligibility gate is extremely narrow**

* **Update (head_dim=128, this PR).** `head_dim` is now a compile-time
  regime with 64 (GPT-2-class) and 128 (Llama-class) both instantiated and
  gated in: `_fa4_16bit_d64_causal_inputs` / `_fa4_strided_bthd_layout` /
  `_fa4_symbol` (`aten_fast.py`) derive dtype+head_dim symbol selection at
  the call site, and `fa4_ops.mojo` exports
  `flash_attention_{fwd,bwd}_{bf16,f16}_d128_causal{,_strided_qkv,_bhsd}`
  alongside the pre-existing d64 family (BM/warpgroup-count/register-budget
  constants were already head_dim-parametric in `fa4_fwd_common.mojo` /
  `fa4_bwd_common.mojo` before this landed). GQA and non-multiple-of-128
  sequences are still open below.
* **What.** `_fa4_16bit_d64_causal_inputs`, `aten_fast.py:6482` (the name
  predates f16/d128 support and is now imprecise, kept for call-site
  stability).
* **Current implementation.** bf16 or f16, `head_dim in (64, 128)`,
  `seqlen % 128 == 0`, `is_causal is True` (identity, not truthiness), no
  dropout, no mask, no GQA, `sm_90a`.
* **Why it is not optimal.** GPT-2 (`head_dim = 64`) and Llama-class
  (`head_dim = 128`) models now fit; anything with GQA, any other head_dim,
  and any non-multiple-of-128 sequence still fall all the way to [A1](#a1).
* **What the optimized version looks like.** A masked tail for
  `seqlen % 128 != 0`, and a GQA arm that indexes the KV head instead of
  materializing a repeat. These are compile-time *regimes* selected at runtime,
  which is what AGENTS.md rule 4 asks for — not hardcoded shapes.
* **Expected win.** head_dim=128: measured, see `benchmarks/baselines.html`
  (`B1H16S4096D128`). GQA / ragged-seqlen: **UNMEASURED**, bounded above by
  the [A1](#a1) ratio for the shapes they would newly claim.
* **How to measure it.** `tests/test_fa4_host_wiring.py` and
  `tests/test_eager_kernels.py` (`test_fa4_*[...-d128]`) cover the gate
  itself; for timing, `benchmarks/test_attention.py -k D128` and a
  `head_dim = 128` variant of `bench_nanogpt_train.py`.

### A3
**The fused SDPA backward is Apple-only**

* **What.** `fast_sdpa_backward`, `aten_fast.py:6565` — declines unless
  `probs._device.api == "metal"` (`:6604`) and fp32. Kernel side:
  `sdpa_backward_ops/sdpa_backward_gemm_kernels.mojo:281`,
  `raise Error("sdpa_ta_gemm_f32 is Apple-GPU only")`.
* **Current implementation.** On Metal: five launches, no permute copies, no
  separate dropout-backward pass. On CUDA and ROCm (outside the gfx942 fused
  node), `mojo_device_autograd.py:404-540` runs the composed sequence, which for
  dV alone does `transpose(grad3, 1, 2)` → `_contiguous_view` and then a second
  transpose → `_contiguous_view` (`:417-439`) — two full materializations of an
  `O(B·H·T·D)` tensor per layer.
* **Why it is not optimal.** The transposed-A GEMM those copies exist to avoid is
  exactly what `_tn_mfma_route` (gfx942) and a `TRANSPOSE_A` fp32 kernel
  ([G2](#g2)) provide. The dropout backward is a separate full pass over the
  probability matrix that the fused kernel folds into an operand load.
* **What the optimized version looks like.** Promote `sdpa_ta_gemm_f32` to a
  dtype- and architecture-parametric transposed-A batched GEMM and let the fused
  backward's gate become "GPU + float dtype" rather than "Metal + fp32". The
  causal-band contract it relies on is already documented in the module docstring
  and is device-independent.
* **Expected win.** **UNMEASURED** off Metal. Each removed materialization is one
  pass over a `B·H·T·D` tensor per layer.
* **How to measure it.** `bench_nanogpt_train.py --device mojo` on a workload
  where FA4 declines (e.g. `head_dim = 96`), then
  `compare_nanogpt_train_kernels.py` to watch the transpose kernels vanish.

### A4
**The causal batched GEMM that skips masked contraction indices is gfx942-only**

* **What.** `_try_sdpa_causal_bmm`, `aten_fast.py:5408`, declines unless
  `architecture_name == "gfx942"` (`:5448`). Its Apple counterpart is a separate
  entry point, `_bmm_causal_go` / `BmmCausalF32`, `matmul_ops.mojo:6321`, which
  raises `"BmmCausalF32 is Apple-GPU only"` at `:6369`.
* **Current implementation.** Two device-specific causal-BMM implementations, no
  NVIDIA one. On CUDA the SDPA backward runs dense GEMMs over a matrix that is
  half zeros by construction.
* **Why it is not optimal.** Up to half the contraction work multiplies exact
  zeros. The honest counterweight is already in the source
  (`aten_fast.py:5546-5553`, journal experiment AA): the *output-side* causal
  regime measured **slower** on gfx942 (895.05 → 901.77 µs) because those GEMMs
  are dispatch-bound, not work-bound. Only the two contraction-side regimes are
  worth porting.
* **What the optimized version looks like.** A `causal_mode` parameter on the
  portable batched GEMM with the same runtime dispatch the gfx942 path uses, plus
  the measured decline the journal records so it does not fire where dispatch
  binds.
* **Expected win.** **UNMEASURED** on NVIDIA. On gfx942 the contraction-side
  regimes were accepted as wins (journal Change 18); the output-side one was
  rejected with numbers.
* **How to measure it.** `bench_nanogpt_train.py` +
  `compare_nanogpt_train_kernels.py` on a workload whose SDPA falls to the
  decomposition, with a control that forces the causal flag off.

### A5
**`attn_mask` is a hard decline in eager SDPA**

* **What.** `fast_aten_scaled_dot_product_attention`, `aten_fast.py:7611`
  (`attn_mask is None` inside the eligibility conjunction); the autograd node
  raises explicitly at `mojo_device_autograd.py:271-275`.
* **Current implementation.** `aten::scaled_dot_product_attention` is registered
  for the mojo device (`mojo_device_aten_ops.py`), which suppresses
  PyTorch's own composite decomposition, and then declines any masked call —
  producing a `NotImplementedError` rather than a slow path.
* **Why it is not optimal.** Padded-batch inference and most HuggingFace encoder
  models always pass a mask. There is no slow-but-correct route at all.
* **What the optimized version looks like.** At minimum a masked arm of
  `_sdpa_math_forward_with_dropout` — add the mask to the scores before the
  softmax, exactly as the graph backend does at `aten_functions.py:478-500` — so
  masked SDPA becomes *slow* instead of *unsupported*. Better: an additive-mask
  operand on the flash kernels.
* **Expected win.** N/A (coverage). Removing the raise is the point.
* **How to measure it.** `tests/test_sdpa_host_wiring.py` for the gate;
  `run_hf_mojo.py` for a real masked model end to end.

### A6
**The fused decode kernel's gate excludes long contexts and causal decode**

* **What.** `aten_fast.py:7637-7648`.
* **Current implementation.** `AttnDecodeSpec` (`:7654`) requires `q_len == 1`
  (`:7637`), `not is_causal`, `head_dim % 4 == 0 and head_dim <= 256`,
  `kv_len <= 4096` (`:7641`), and unit innermost strides on q/k/v.
* **Why it is not optimal.** `kv_len <= 4096` is a hard cliff: a decode past 4k
  context silently drops to the math decomposition, which even for `q_len == 1`
  allocates a `(B·H, 1, kv_len)` scores tensor and runs three launches instead of
  one. Speculative decoding and multi-token decode (`q_len` 2–8) also miss.
* **What the optimized version looks like.** A split-KV variant (partition
  `kv_len`, combine with a log-sum-exp merge) so context length stops being a
  gate, plus a small-`q_len` regime.
* **Expected win.** **UNMEASURED.** The decode kernel is described in-source as
  saving two launches and all scratch buffers; long-context decode currently pays
  both.
* **How to measure it.** `uv run --no-sync python bench_gpt2_kernels.py mojo 256
  8192` — the script takes `kv_len` as its third argument, so the cliff is one
  command away — and `bench_gpt2_batch.py` for end-to-end throughput.

---

## Normalization

### N1
**LayerNorm backward is fp32-only and GPU-only, while the forward is not**

* **What.** `fast_aten_native_layer_norm_backward`, `aten_fast.py:4349` (dtype
  gate at `:4376-4380`), versus `fast_aten_native_layer_norm`,
  `aten_fast.py:4236`, which accepts all of `_FLOAT_DTYPES`. The kernel is fp32
  by construction:
  `normalization_backward_ops/normalization_backward_params.mojo` types every
  pointer as `Scalar[DType.float32]`.
* **Current implementation.** fp16/bf16 LayerNorm forward succeeds; its backward
  returns `NOT_HANDLED`, and the registration
  (`mojo_device_aten_ops.py`) turns that into a raise.
* **Why it is not optimal.** Under bf16 autocast this is invisible (autocast runs
  LayerNorm in fp32), which is why nanoGPT never hit it. A model trained in pure
  bf16 — increasingly the norm — cannot run its backward at all.
* **What the optimized version looks like.** dtype-parametric kernels with fp32
  accumulation, matching what the forward already does, plus a CPU arm.
* **Expected win.** N/A for speed under autocast; unblocks pure-bf16/fp16
  training.
* **How to measure it.** `tests/test_eager_kernels.py` parametrized over
  `torch.bfloat16`; `bench_nanogpt_train.py` with autocast disabled and the model
  cast to bf16.

### N2
**BatchNorm and GroupNorm BACKWARD are absent in eager (the BatchNorm training
forward now exists)**

* **What.** `aten::native_batch_norm_backward` and
  `aten::native_group_norm_backward` are not registered at all — the file has
  exactly one `_register_missing`, for `aten::_adaptive_avg_pool2d_backward`.
* **Current implementation.** The BatchNorm TRAINING FORWARD landed with the
  shared-moments normalization work: `_fast_batch_norm_training` in
  `aten_fast.py` over
  `normalization_forward_ops/batch_norm_kernels.mojo`, which reduces
  `{0, 2, 3}` in place through `op_utils._moments_scan_contig` and produces
  `save_mean` / `save_invstd` plus the ATen running-statistic update
  (measured 0.19-0.85x stock CUDA). Because its backward does not exist,
  `mojo_device_native_batch_norm` refuses a training call whose inputs require
  grad IN THE FORWARD — a raise from inside the autograd engine aborts the
  process on this backend rather than raising. GroupNorm likewise has a fused
  forward and no backward, and `mojo_device_native_group_norm` now refuses a
  grad-requiring call in the forward for the same reason — with no `training`
  flag to key on, that means every such call.
* **Why it is not optimal.** ResNet / VGG / DenseNet *training* on the mojo
  device is still blocked — now by the missing backward rather than the
  missing forward; the `demo_scripts/` vision examples are inference-only for
  this reason.
* **What the optimized version looks like.** The two backwards, written like
  the existing layer-norm pair (`normalization_backward_dx.mojo` +
  `normalization_backward_params.mojo`), which already solve the identical
  "per-channel parameter reduction across a large outer extent" problem; the
  batch-norm one can walk the NCHW geometry exactly as its forward does.
  Removing the forward preflight is part of that change.
* **Expected win.** N/A (coverage). Unblocks vision training.
* **How to measure it.** `tests/test_aten_functions.py` for correctness; a
  resnet-18 training-step benchmark modelled on `bench_nanogpt_train.py` for
  speed — none exists today, see [Not audited](#not-audited).

### N3
**`_TARGET_BLOCKS = 1280` in the LayerNorm-backward parameter reduction is an unlabelled fitted constant**

* **What.** `normalization_backward_ops/normalization_backward_params.mojo:34`,
  used at `:227` (`num_chunks = max(1, _TARGET_BLOCKS // col_blocks)`).
* **Current implementation.** A bare `comptime` integer with no comment naming
  the device it was fitted on. 1280 ≈ 4 × 304 CUs (MI300X); on an H100's 114 SMs
  that asks for ~11 blocks per SM.
* **Why it is not optimal.** AGENTS.md is explicit: "When a tuning constant was
  measured on one architecture, say so where it is defined, the way
  `LSM_BLOCKS_PER_CU` and `LSM_L2_BUDGET` do." This one does not. It also cannot
  be caught by `compare_kernel_asm.py`, because it only changes grid geometry.
* **What the optimized version looks like.** Derived from
  `ctx.default_device_info.sm_count` at launch, or at minimum documented with the
  card and the sweep behind it.
* **Expected win.** **UNMEASURED**; possibly zero on gfx942 by construction.
* **How to measure it.** Sweep the constant on the LayerNorm-backward shapes with
  `scripts/rocprof_kernels.py` (ROCm) or `ncu` (NVIDIA) and record the table next
  to the definition.

### N4
**RMSNorm has no fused op in either backend**

* **What.** No `rms_norm` anywhere under `torch_mojo_backend/` (verified by
  grep). `demo_scripts/gemma3.py` hand-rolls it.
* **Current implementation.** `nn.RMSNorm` (or a hand-written one) decomposes
  into roughly six ops — `pow`, `mean`, `add`, `rsqrt`, `mul`, `mul` — each a
  separate launch with its own full-size intermediate.
* **Why it is not optimal.** Every modern LLM (Llama, Gemma, Qwen, Mistral) uses
  RMSNorm at every layer, twice. Six memory-bound launches where one suffices.
* **What the optimized version looks like.** `RmsNormSpec` in
  `normalization_forward_ops` plus its backward, plus the `aten::rms_norm` /
  `_fused_rms_norm` registrations. The existing
  `normalization_forward_kernels.mojo` LayerNorm kernel is the same problem minus
  the mean subtraction. Graph side: route to `max.graph.ops.rms_norm`, which
  already exists.
* **Expected win.** **UNMEASURED** here. A 6-launch → 1-launch collapse on a
  memory-bound op normally approaches a 6x reduction in that op's traffic; what
  fraction of a step that is depends entirely on the model.
* **How to measure it.** `demo_scripts/gemma3.py` end to end, and
  `profile_gpt2_generate_aten.py`-style ATen profiling on a Llama-shaped model.

---

## Reductions

### R1
**Non-trailing reductions materialize a full permuted copy of the input** —
**DONE** (generic reduction skeleton)

> **Closed (2026-08-11).** `reduce_skeleton.mojo` is one comptime-parametrized
> reduction over an accumulator (`ReduceOp`: identity, map, combine, finalize)
> and one `(outer, reduce, inner)` geometry, with a contiguous-axis kernel, a
> STRIDED-axis kernel and a split-the-reduce-axis policy derived from the
> runtime SM count. `sum`, `mean`, `amax`, `amin`, `max`, `min`, `any`, `all`
> and the new L2 norm are one accumulator each; `min.dim` moved onto the
> arg-reduction's (value, index) payload, which already had both. Every one of
> them now reads an adjacent non-trailing reduce interval WHERE IT LIES
> (`aten_fast._REDUCE_MIDDLE_DIRECT` lists them), so the permuted copy is gone
> for the whole family rather than for fp32 `sum` alone. Measured on an H100
> PCIe against stock CUDA, ours/stock device time: `sum` bf16 `dim=0`
> 4.87 -> 0.90, `amax` bf16 `dim=0` 3.33 -> 0.81, `any` bool `dim=0`
> 3.41 -> 0.68, `min.dim` f32 `dim=0` 1.34 -> 0.60. The full-reduction end
> moved with it (one block for the whole tensor was the other half of the same
> defect): `sum` bf16 16.7M 13.08 -> 0.85, `max` bf16 7.76 -> 0.58.
> **Still deferred**: non-ADJACENT dim sets (e.g. `dim=(0, 2)` of a rank-3)
> still permute and materialize in Python, and so do genuinely strided
> operands; matmul's strided-operand copies and conv's im2col workspace are
> untouched.

### R2
**`linalg_vector_norm` is a three-launch composition with an input-sized
temporary** — **DONE** (`NormSpec`)

> **Closed (2026-08-11).** `NormL2Op` in `reduce_skeleton.mojo` maps each
> element to `x*x` and folds the square root into the accumulator's finalize,
> so the op is one pass with no temporary, over any dim set the skeleton
> accepts, in f32/f16/bf16 (it was fp32-only). Measured on an H100 PCIe,
> ours/stock device time: 357x789 f32 10.61 -> 0.86, 16.7M f32 6.13 -> 0.96;
> bf16 went from "declined, decomposed by torch" to 0.79-0.82.

### R3
**`cumsum` is trailing-dim only, three dtypes, and refuses `dtype=`**

* **What.** `fast_aten_cumsum`, `aten_fast.py:5101` — declines unless
  `dtype is None`, the dtype is int64/int32/float32, and `dim % rank == rank - 1`;
  `CumsumSpec` at `:5112`.
* **Current implementation.** A single trailing-axis scan.
* **Why it is not optimal.** bf16/fp16 cumsum and any non-trailing axis are hard
  declines and the registration raises. Unlike a reduction there is no
  permute-and-retry fallback here at all.
* **What the optimized version looks like.** dtype parametrization plus the
  transpose-materialize-transpose shape softmax already uses
  (`aten_fast.py:7705`) as an interim, then a strided scan kernel.
* **Expected win.** N/A (coverage).
* **How to measure it.** `tests/test_aten_functions.py`.

---

## Data movement

### D1
**The hot rank≤4 materialize path is missing the vectorized transpose the rank≤8 path has**

* **What.** `_permute_copy`, `data_movement_ops/data_movement_ops.mojo:243` — its
  batched-transpose arm enqueues `_transpose2d_kernel` only (`:411`). Compare
  `_copy_strided`, `op_utils/__init__.mojo:707` — its equivalent arm at `:794`
  tries `_transpose2d_vec_kernel` (`op_utils/__init__.mojo:564`) first and only
  then falls back to `_transpose2d_kernel` (`:631`). The caller that picks between
  the two entry points is `TorchMojoTensor._materialize_contiguous`,
  `mojo_device/torch_mojo_tensor.py:450`: rank ≤ 4 goes to `PermuteCopy`,
  rank > 4 to `CopyStrided` (`:461`).
* **Current implementation.** `_transpose2d_kernel` moves one element per thread
  per access (128-byte runs per wave). `_transpose2d_vec_kernel` moves 16 bytes
  per access (256-byte runs, eightfold fewer requests) whenever
  `rows % VEC == 0`, `cols % VEC == 0`, `src_ld % VEC == 0` and both pointers are
  16-byte aligned — properties of strides and the allocator, not of a particular
  shape.
* **Why it is not optimal.** The rank≤4 path is described in its own comment as
  the *hot* one — "attention q/k/v transposes, expand". So the fast kernel is
  wired into the cold path and the slow kernel into the hot one. The
  `_permute_copy` transpose arm's gate (`s2 == 1 and s3 >= d2`) is the same
  predicate the vec kernel needs, and typical attention shapes (head_dim 64,
  seq 1024, bf16 → VEC 8) satisfy every alignment condition.
* **What the optimized version looks like.** Lift the vec-kernel attempt out of
  `_copy_strided` into a shared helper in `op_utils` and call it from both. It is
  a code move: both call sites already compute `rows`, `cols`, `src_ld`, `batch`
  and both batch strides.
* **Expected win.** The `op_utils` comment records the two kernels at **2.1–2.4
  TB/s vs 3.57 TB/s** on the weight-gradient operands — and says so as a gfx942
  measurement. That is the honest bound: ~50% on the transposes this path
  performs, **measured on gfx942 only**, and only for shapes meeting the
  alignment gate.
* **How to measure it.** `profile_nanogpt_train_aten.py` +
  `compare_nanogpt_train_kernels.py`, watching the `dm_permute_*` /
  `transpose2d_*` rows; or `scripts/rocprof_kernels.py --kernel-trace` on
  `bench_nanogpt_train.py --device mojo`, which reports per-kernel medians
  directly.

### D2
**The generic strided copy is a scalar kernel with an 8-deep div/mod chain and no dimension collapsing**

* **What.** `_copy_strided_kernel`, `op_utils/__init__.mojo:491`; dispatched by
  `_copy_strided`, `:707`; reached from `_copy_strided_into`,
  `mojo_device/torch_mojo_tensor.py:752`, and from `_materialize_contiguous` for
  rank > 4 (`:461`).
* **Current implementation.** One element per thread. Every element recomputes
  its coordinates with a `MAX_RANK - 1` (seven) deep `%` / `/` chain over runtime
  extents, then does one scalar load and one scalar store. The only fast arms are
  the two transpose kernels, which require every dim outside the trailing three
  to be exactly 1 (`outer_trivial`, `op_utils/__init__.mojo:762`).
* **Why it is not optimal.** Three things at once: (a) scalar accesses even when
  the innermost extent is contiguous on both sides; (b) no collapsing of adjacent
  dimensions whose strides are already compatible — a rank-6 copy that is really
  a rank-2 problem still pays six divisions per element; (c) integer division by
  runtime values is itself expensive on GPUs. `_permute_copy` already solves (a)
  properly for rank ≤ 4 with its `_try_run_gather[V32/V16/V8]` ladder
  (`data_movement_ops.mojo:287`); the rank≤8 path never got it.
* **What the optimized version looks like.** A stride-normalizing prologue that
  merges adjacent compatible dims down to the smallest equivalent rank, then the
  `_permute_copy` ladder (run gather → tiled transpose → scalar) run on the
  collapsed problem. The scalar kernel stays as the true general case; most real
  tensors would stop reaching it.
* **Expected win.** **UNMEASURED.** Bounded by the same 2.1 → 3.6+ TB/s spread
  the transposes show, plus the removed divisions.
* **How to measure it.** A microbenchmark over `x.permute(...).contiguous()` for
  rank 5–8 shapes on `mojo:0`; `tests/test_eager_kernels.py` for correctness.
  `_copy_strided` is also what `.contiguous()`, `copy_` into views and `expand`
  materialization all use, so a regression here is broad — keep the scalar arm.
* **Half of this now exists next door.** The fill family took the same
  treatment and shipped it: `_collapse_fill_layout` (`op_utils/__init__.mojo`)
  is exactly the stride-normalizing prologue described above, and `_fill_contig`
  is the vector ladder that follows it. The *shape* of the fix transfers; the
  prologue itself does not, because a fill collapses a SET of addresses — it may
  drop stride-0 dimensions and reorder dimensions freely — while a copy must
  merge the source and destination strides jointly and preserve the
  correspondence between the two index spaces. Read it as a worked example, not
  as a function to call.

### D3
**The fused three-source `cat` is Metal-only** — **DONE** (batched `CatN`)

* **What it was.** `fast_aten_cat` fused exactly two contiguous inputs
  (`Cat2`) everywhere and three (`Cat3`) on Metal only; every other input
  count cost one `NarrowCopyDst` launch per input.
* **What replaced it.** The N-source variant this entry asked for:
  `CatN` in `data_movement_ops.mojo`, one launch per `CAT_SEG_CAP` (64)
  inputs, taking the runtime descriptor array the foreach kernels take, with
  Apple on the pointer-argument variant the Metal ABI requires. `Cat2` and
  `Cat3` are gone; the per-input loop remains only for non-contiguous inputs
  (the KV-cache gather) and the CPU device. The vector width now follows the
  dtype and the runtime alignment (16 bytes: 8 bf16 elements, 4 fp32) rather
  than a fixed four elements, so bf16 stopped copying at half rate.
* **Measured** (H100 PCIe, `benchmarks/test_data_movement.py`, ours/stock
  device time): `cat` 64x262144 bf16 9.347 → 0.988, 2x8388608 bf16
  7.087 → 0.991, 64x262144 f32 5.435 → 0.982, 2x8388608 f32
  4.117 → 0.997; `stack` improved with it (16.086 → 1.311,
  7.969 → 1.085). ~1.75 TB/s of this card's 2.04 TB/s peak.
* **What is still open.** Input lists longer than 64 pay one launch per 64
  (512 x 2048 bf16: 2.7x stock, launch-latency bound), and lists whose row
  lengths are not 16-byte multiples run the element-wise instantiation
  (37 x 12345 bf16: 1.4x). Both are small-tensor regimes.

### D4
**`nonzero` round-trips through the host**

* **What.** `fast_aten_nonzero`, `aten_fast.py:4009` —
  `t._to_cpu_tensor().nonzero()` then `TorchMojoTensor._from_cpu(...)` (`:4015`).
* **Current implementation.** Full device→host copy, CPU `nonzero`, full
  host→device copy. Because the payload is read from Python this also drains the
  entire call queue (`docs/kernel_call_queue.md`, "Host reads drain").
* **Why it is not optimal.** The output *shape* is data-dependent, so one
  synchronization is unavoidable — but only one, on a single integer. Today the
  whole tensor crosses the bus twice.
* **What the optimized version looks like.** Device-side count kernel → one
  4-byte D2H of the count → allocate → device-side compaction (prefix sum +
  scatter). One sync on a scalar instead of two full transfers.
* **Expected win.** **UNMEASURED**, and `nonzero` is rare in the benchmarked
  workloads. The queue drain may matter more than the copy.
* **How to measure it.** A microbenchmark on a large boolean mask; watch the
  drain with `TORCH_MOJO_BACKEND_TRACE=1`.

### D5
**`aten::index` handles only a single index tensor on dimension 0**

* **What.** `fast_aten_index`, `aten_fast.py:3840` — `if len(non_none) != 1 or
  non_none[0][0] != 0: return NOT_HANDLED` (`:3846`).
* **Current implementation.** Row gather along dim 0 via `GatherRows`; everything
  else raises.
* **Why it is not optimal.** `x[:, idx]`, `x[i, j]`, boolean masking and
  multi-tensor advanced indexing are all common and all hard failures. There is
  no slow path.
* **What the optimized version looks like.** A general gather taking the
  broadcast index shape and per-dim index pointers as runtime data (rank ≤ 8,
  the descriptor style `CopyStrided` already uses), plus the boolean-mask case
  routed through `nonzero` + gather.
* **Expected win.** N/A (coverage).
* **How to measure it.** `tests/test_aten_functions.py` with parametrized index
  patterns.

### D6
**`_LLC_BYTES = 256 MiB` is MI300X's Infinity Cache, applied to every architecture**

* **What.** `op_utils/__init__.mojo:160`, consumed by `_bw_flat_blocks` (`:186`)
  and neighbouring `_bw_blocks` (`:167`), which choose the grid for every
  bandwidth-bound launch.
* **Current implementation.** Above 256 MiB of traffic the grid covers the slots
  exactly; below it a 4096-block cap (or `2 × sm_count`) is kept because the
  operands are assumed cache-resident. The comment documents the sweep — on
  gfx942.
* **Why it is not optimal.** H100 has 50 MB of L2, an RTX 2000 Ada 32 MB, an
  M-series GPU a different hierarchy again. On those parts the "resident" arm is
  selected for working sets that are not resident, choosing a deliberately
  undersized grid for a streaming copy. This is exactly the failure mode
  AGENTS.md describes: identical device assembly, different launch geometry.
* **What the optimized version looks like.** Query the device's last-level cache
  size at launch — MAX exposes device attributes, and
  `gemm16_v3_kernels.mojo:1701` already calls `ctx.get_attribute(...)` — or at
  minimum re-fit per architecture behind a `comptime if` and record each sweep
  next to it.
* **Expected win.** **UNMEASURED** off gfx942.
* **How to measure it.** The cast/binary-add microbenchmark the existing comment
  describes (TB/s of read+write traffic across a traffic sweep), re-run on the
  target card. `scripts/compare_kernel_asm.py` will show nothing — read the
  dispatch by hand.

### D7
**Rank>4 operands are pre-materialized for every binary elementwise op**

* **What.** `_try_spec_binary_into`, `aten_fast.py:1325`, the `_tc(a)` / `_tc(b)`
  at `:1347-1349`; documented in `_try_spec_binary`'s docstring
  (`aten_fast.py:1403`, line `:1412`, "rank>4 operands are pre-materialized").
* **Current implementation.** The spec's flat broadcast path is rank ≤ 4; above
  that both operands are copied contiguous first.
* **Why it is not optimal.** A 5-D strided add copies both operands before adding
  them. Rank-5 tensors are not exotic (grouped-conv layouts, `(B, T, H, D)` plus
  a head axis, video).
* **What the optimized version looks like.** Extend the binary spec's coordinate
  decode to `MAX_RANK` — the machinery exists, `CopyStrided` is already rank-8 —
  or apply the dimension-collapsing prologue proposed in [D2](#d2), which would
  make most rank-5+ tensors rank ≤ 4 before the gate is consulted.
* **Expected win.** **UNMEASURED.**
* **How to measure it.** Time `a + b` for a rank-5 strided pair on `mojo:0`.

---

## Convolution and pooling

### C1
**Convolution is im2col + GEMM with a full column-matrix temporary**

* **What.** `fast_aten_convolution`, `aten_fast.py:7426`; the `Im2col` launch at
  `:7471` and its `(n, ckk, cols)` allocation just above it; the `Bmm` at
  `:7502`; the separate `BiasAddChan` pass at `:7552`.
* **Current implementation.** Unless the kernel is exactly 1x1 / stride-1 — in
  which case the NCHW input *is* the column matrix (`:7465`) — an
  `(N, C·kh·kw, out_h·out_w)` buffer is allocated and filled, one batched GEMM
  runs against it, then a third pass adds the bias.
* **Why it is not optimal.** The temporary is `kh·kw` times the input, written
  once and read once. For a 3x3 conv that is 9x the input in extra traffic and 9x
  the peak activation memory. The bias is a full extra pass over the output.
* **What the optimized version looks like.** Implicit-GEMM convolution: the
  GEMM's A-operand loader computes the im2col address on the fly instead of
  reading a materialized matrix, with the bias folded into the store epilogue —
  `_matmul_bias_run` (`matmul_ops.mojo:6416`) already has a fused-bias epilogue.
  Shapes stay runtime; only head_dim-style regimes are compile-time.
* **Expected win.** **UNMEASURED** here. Eliminates `kh·kw ×` the input in
  write+read traffic plus one output-sized pass for the bias; largest for big
  spatial extents and small channel counts.
* **How to measure it.** `uv run --no-sync python bench_mojo.py resnet` (forward
  latency, cuda vs mojo:0 vs mojo:1) and `demo_scripts/vgg.py` /
  `demo_scripts/densenet.py`. There is no per-kernel conv benchmark in the repo —
  see [Not audited](#not-audited).

### C2
**Grouped convolution issues `N × groups` GEMM launches from a Python loop**

* **What.** `aten_fast.py:7524` — `for s in range(n): for g in range(groups):
  _call_mojo(_MatmulExtension, "Matmul", ...)`.
* **Current implementation.** One offset `Matmul` per (sample, group) pair.
* **Why it is not optimal.** A depthwise conv has `groups == C`; batch 32 with
  C = 256 is 8192 Python-side launches for one layer, each through the full
  `_call_mojo` → prepare → queue path. This is dispatch cost, not kernel cost,
  and it scales with batch size.
* **What the optimized version looks like.** One batched GEMM whose batch axis is
  `n × groups`, with per-matrix element offsets as runtime strides — the `Bmm`
  entry point already takes batch strides and an `a_bstride == 0` broadcast form,
  which the `groups == 1` arm right above already uses. Depthwise specifically
  (`c_per_group == 1`) deserves a direct kernel rather than a GEMM at all.
* **Expected win.** **UNMEASURED.** Reduces `N × groups` launches to 1.
* **How to measure it.** `bench_mojo.py` on a MobileNet/EfficientNet-style
  (depthwise-heavy) model, and `TORCH_MOJO_BACKEND_TRACE=1` to count launches.

### C3
**The eager conv backward rides the materialized im2col route**

* **What.** `fast_aten_convolution_backward` and `fast_aten_convolution`,
  `aten_fast.py`. Both decline transposed convolutions; the forward also
  declines rank-3 (conv1d) and rank-5 (conv3d) inputs, while the backward
  already serves rank-3 through a size-1 H lift.
* **Current implementation.** `aten::convolution_backward` is registered and
  computes all three gradients: the bias gradient as one reduction, the weight
  gradient as a transposed-B GEMM against the im2col matrix plus a reduce over
  the batch, the data gradient as a shared-A GEMM plus a `Col2im` gather. The
  implicit-GEMM route of [C1](#c1) is FORWARD-ONLY, so both spatial
  gradients pay a full column-buffer write and read, which is what the
  forward's own profiling found was 89% of its device time before that route
  existed.
* **Why it is not optimal.** Measured on an H100 PCIe against cuDNN
  (`benchmarks/baselines.html`, 12 entries each): the data gradient runs
  1.40-15.12x stock (median 10.58) and the weight gradient 3.27-23.25x
  (median 11.71), both worst on the deep-C small-spatial stages; the bias
  gradient is 0.39-3.21x (median 0.90), i.e. usually FASTER, since it is a
  plain reduction. The weight gradient additionally materializes an
  `(N, out_c, C*R*S)` partials buffer, because none of the GEMM routes here
  takes a `beta=1` accumulator — bigger than the column buffer itself whenever
  `out_c > OH*OW`, i.e. in exactly those deep stages.
* **What the optimized version looks like.** A patch-major im2col variant (a
  transposed write, so a different kernel rather than a different index) folds
  N into the weight GEMM's K, which removes both the partials buffer and the
  batch reduce and turns `N` small-K GEMMs into one deep-K GEMM. Beyond that,
  an implicit-GEMM data/weight gradient in the shape of the forward's TMA
  im2col route would remove the column buffer entirely.
* **Expected win.** **UNMEASURED**, but the forward's own materialized-route
  profile bounds it: the column traffic this removes was 89% of that route's
  device time.
* **How to measure it.** `benchmarks/test_vision.py::test_conv2d_backward`,
  whose baseline keys are `convolution_backward/<dtype>/<shape>/{dgrad, wgrad,
  bgrad}` — one node per gradient, so a change to one of the three kernels is
  attributable. Then a resnet-18 training step.

### C4
**Pooling has no `ceil_mode` and no backward**

* **What.** `fast_aten_max_pool2d_with_indices`, `aten_fast.py:5135`
  (`not ceil_mode` at `:5148`); `fast_aten_avg_pool2d`, `:5185` (`not ceil_mode`
  at `:5203`). `aten::max_pool2d_with_indices_backward` and
  `aten::avg_pool2d_backward` are unregistered;
  `aten::_adaptive_avg_pool2d_backward` is explicitly `_register_missing`
  (`mojo_device_aten_ops.py`).
* **Current implementation.** Forward-only, floor-mode-only, rank-4-only. All
  three pooling forwards refuse a grad-requiring call in the FORWARD (see
  [N2](#n2)); `_register_missing` on the backward was not enough, because that
  raise still happens inside the autograd node and aborts the process.
* **Why it is not optimal.** `ceil_mode=True` appears in common torchvision
  configurations; the missing backwards are the third blocker for vision
  training.
* **What the optimized version looks like.** `ceil_mode` is an output-extent
  formula change plus edge clamping — a small kernel edit. The backwards are
  scatter-adds keyed on the saved indices (max) or a uniform spread (avg).
* **Expected win.** N/A (coverage).
* **How to measure it.** `tests/test_aten_functions.py`.

---

## Optimizer and foreach

### O1
**Fused AdamW is fp32-homogeneous only**

* **What.** `fast_aten__fused_adamw`, `aten_fast.py:880`; the dtype check at
  `:947` (`tensor._dtype != DType.float32` → `RuntimeError`) and the literal
  `0,  # homogeneous FP32 parameters, gradients, and optimizer state` at `:1007`.
* **Current implementation.** Every parameter, gradient, moment and step counter
  must be contiguous fp32 on one device, or the op raises.
* **Why it is not optimal.** Two gaps. (a) Mixed-precision training with bf16
  *master* weights (no fp32 copy) cannot use the fused optimizer. (b) The failure
  mode is a `RuntimeError` rather than `NOT_HANDLED`, so PyTorch cannot fall back
  to its own foreach path (which is now the fast one, see [O2](#o2)).
* **What the optimized version looks like.** A per-tensor dtype field in the
  descriptor record — the ABI already carries a "homogeneous" flag slot, so the
  heterogeneous encoding was anticipated — with fp32 accumulation inside the
  kernel, and decline-not-raise for anything still unsupported.
* **Expected win.** N/A for fp32 users; unblocks bf16-master-weight training.
* **How to measure it.** `tests/test_eager_optimizer_ops.py`;
  `bench_nanogpt_train.py` with a bf16-parameter variant.

### O2
**Five batched `_foreach_*` kernels are Metal-only**

* **What.** `_foreach_metal_f32`, `aten_fast.py:586` — returns `None` unless
  `device.api == "metal"` (`:611`). Its callers:
  `fast_aten__foreach_mul__scalar` (`:692`),
  `fast_aten__foreach_div__scalarlist` (`:723`),
  `fast_aten__foreach_lerp__scalar` (`:733`),
  `fast_aten__foreach_addcmul__scalar` (`:809`),
  `fast_aten__foreach_addcdiv__scalarlist` (`:824`) — all via
  `_foreach_scalar_inplace` (`:668`). Only `fast_aten__foreach_add__scalar`
  (`:712`) has a generic route (`_fast__foreach_add__scalar_generic`, `:462`),
  added by journal Change 45. `_foreach_norm` (`:329`) and `_foreach_mul_.Tensor`
  (`:512`) are device-general but fp32-only.
* **Current implementation.** On CUDA and ROCm those five decline, and
  `_register_foreach_inplace` (`mojo_device/aten_ops/foreach.py`) redispatches to
  ATen's generic decomposition — one launch per tensor.
* **Why it is not optimal.** `torch.optim.AdamW(foreach=True)` — PyTorch's
  default when `fused` is not requested — uses exactly `_foreach_mul_`,
  `_foreach_addcmul_`, `_foreach_addcdiv_`, `_foreach_div_` and `_foreach_lerp_`.
  For nanoGPT's 75 parameter tensors that is ~375 launches per step instead of
  five. The journal priced the analogous case: `_foreach_add_.Scalar` was 76
  launches / 379 µs against ROCm's 10.4 µs, and Change 45 collapsed it to one.
* **What the optimized version looks like.** Replace the Metal gate with "GPU +
  contiguous + homogeneous fp32". The `ForeachScalarOp` kernel
  (`optimizer_ops/optimizer_ops.mojo`) already takes a runtime descriptor array;
  what is Metal-specific is the 8-slot batching, not the operation.
* **Expected win.** By analogy with Change 45, ~370 µs/step on gfx942 — **that is
  an analogy, not a measurement of these five ops**.
* **How to measure it.** `bench_nanogpt_train.py` with the optimizer configured
  `fused=False, foreach=True`, plus `compare_nanogpt_train_kernels.py`'s
  optimizer group.

**Five batched `_foreach_*` kernels are Metal-only** — **DONE** (one
descriptor-batched body for the whole family)

* **What it was.** `_foreach_metal_f32` returned `None` unless
  `device.api == "metal"`, so `_foreach_mul_.Scalar`,
  `_foreach_div_.ScalarList`, `_foreach_lerp_.Scalar`,
  `_foreach_addcmul_.Scalar`, `_foreach_addcdiv_.ScalarList` and
  `_foreach_sqrt` all declined on CUDA and ROCm and
  `_register_foreach_inplace` redispatched to ATen's decomposition — one
  launch per tensor. Only `_foreach_add_.Scalar` and `_foreach_mul_.Tensor`
  had device-general descriptor kernels of their own.
* **What replaced it.** `foreach_batched_kernels.mojo`: ONE kernel body,
  comptime-parametrized by dtype and by op (mul/add/div by a per-tensor host
  scalar, multiply by a 0-d device scalar, lerp, addcmul, addcdiv, sqrt),
  reached through one bridge `_foreach_ew_go[op]` and one Python launch
  helper. The six separate bridges it replaced (`ForeachScalarOp`,
  `ForeachLerpScalar`, `ForeachAddcOp`, `ForeachSqrt`, `ForeachMulTensor` and
  `elementwise_ops`' `ForeachAddScalar`) are gone, as is the hand-tuned
  `foreach_mul_tensor_f32_aligned_streaming_ilp8_t128_v8`. Apple regroups the
  same descriptors into the fixed-arity Metal launches, which is the only
  part that ever needed to be Metal-specific.
* **The Metal gate was not the whole cost.** The chunk size was fixed at
  65_536 elements, which put the 4x1_048_576 list on 64 blocks and the
  16x65_536 list on 16 — on 114 SMs. It is now a runtime launch argument
  derived from the element count and the runtime SM count, which is where
  most of the win came from: `_foreach_add_.Scalar` and
  `_foreach_mul_.Tensor` already had a single-launch descriptor path and
  still went from 1.85/1.94 to 0.84/0.81.
* **Measured** (H100 PCIe, `benchmarks/test_foreach.py`, ours/stock device
  time, 4x1_048_576 then 16x65_536): `_foreach_mul_.Scalar` 1.76/4.31 ->
  0.84/0.39, `_foreach_add_.Scalar` 1.85/1.95 -> 0.84/0.39,
  `_foreach_div_.ScalarList` 1.19/2.88 -> 0.59/0.29, `_foreach_lerp_.Scalar`
  4.55/6.88 -> 0.85/0.44, `_foreach_addcmul_.Scalar` 2.43/4.55 ->
  0.84/0.44, `_foreach_addcdiv_.ScalarList` 2.36/3.36 -> 0.81/0.35,
  `_foreach_mul_.Tensor` 1.94/2.04 -> 0.81/0.37, `_foreach_sqrt`
  1.05/1.57 -> 0.53/0.30. (The four `.Scalar`/`.ScalarList` "before" numbers
  are freshly measured; the recorded baselines for `_foreach_mul_.Scalar`
  (8.03/6.91) and `_foreach_div_.ScalarList` (5.24/4.43) could not be
  reproduced at the commit that held them.)
* **What is still open.** mul/add take every float dtype; div, lerp, addc*,
  sqrt and mul.Tensor are still fp32-only, because the sequential kernels
  they have to stay bit-compatible with do not all widen to fp32 the same
  way. Lists longer than `FEW_DESC_CAP` (64) still pay one launch per 64.

### O3
**`clip_grad_norm_` is fp32-only end to end**

* **What.** `fast_aten__foreach_norm`, `aten_fast.py:329` (dtype check at `:352`);
  `fast_aten__foreach_mul__tensor`, `:512` (dtype check at `:541`); the
  sequential fallback at `:381`.
* **Current implementation.** Both decline for non-fp32 and fall back to
  `foreach_norm_sequential_fallback` or ATen redispatch — one launch per
  parameter.
* **Why it is not optimal.** Under bf16 autocast the *gradients* are fp32
  (because the parameters are), so nanoGPT is fine; a pure-bf16 model is not.
* **What the optimized version looks like.** dtype parametrization of
  `ForeachL2Norm` and `ForeachMulTensor` with fp32 accumulation.
* **Expected win.** **UNMEASURED.**
* **How to measure it.** `tests/test_eager_optimizer_ops.py`;
  `bench_nanogpt_train.py` with a bf16-parameter model.

---

## Elementwise and dtype coverage

### E1
**Divide-by-scalar has no scalar spec: an extra allocation and launch per call**

* **What.** `fast_aten_div`, `aten_fast.py:2498`; the fill it ends up taking,
  `_try_spec_binary_into`, `aten_fast.py:1361` ("One scalar operand: embed it as
  a queued 0-d fill"). Compare `fast_aten_add` (`:2336`), `fast_aten_sub`
  (`:2450`) and `fast_aten_mul` (`:2468`), which all try `_try_spec_scalar`
  (`:2039`) first.
* **Current implementation.** `x / 8.0` finds no `DivScalarSpec`, so
  `_try_spec_binary` embeds the scalar as a 0-D `FillSpec` result — an allocation
  plus a kernel launch — and then runs the broadcast binary.
* **Why it is not optimal.** Two launches and one allocation where one launch
  suffices, on one of the most frequently called ops there is (`x / sqrt(d)` in
  attention, `grad / world_size`, every mean-style normalization).
* **What the optimized version looks like.** `DivScalarSpec` and
  `DivScalarIntSpec` in `elementwise_ops` mirroring `MulScalarSpec` exactly, plus
  an `RDivScalarSpec` for `scalar / x`. Do **not** implement it as a multiply by
  the reciprocal — that changes the numerics.
* **Expected win.** **UNMEASURED.** One launch plus one allocation per call; the
  value is proportional to how CPU-bound the workload is, and the GPT-2 decode
  profile has batch-1 decode CPU-bound on both backends.
* **How to measure it.** `bench_gpt2_kernels.py`, and
  `profile_gpt2_generate_aten.py` / `compare_gpt2_aten_profiles.py`
  (per-ATen-op self time, which is where per-call overhead shows up).

### E2
**`native_dropout` is fp32 and GPU only**

* **What.** `fast_aten_native_dropout`, `aten_fast.py:4121` (`not _on_gpu(a)` at
  `:4133`, `a._dtype != DType.float32` at `:4134`);
  `fast_aten_native_dropout_backward`, `:4199` (same, `:4207`).
* **Current implementation.** One fused Philox kernel for fp32 GPU; anything else
  declines and the registration raises.
* **Why it is not optimal.** bf16/fp16 dropout is the norm in mixed-precision
  training. There is no CPU path either.
* **What the optimized version looks like.** dtype parametrization of the
  existing kernel — the RNG is dtype-independent, only the store changes — plus a
  CPU arm.
* **Expected win.** N/A (coverage).
* **How to measure it.** `tests/test_eager_kernels.py`.

### E3
**`_FLOAT_DTYPES` excludes float64, so most structured ops decline it**

* **What.** `aten_fast.py:222`, used to gate layer norm, group norm, batch norm,
  convolution, pooling, SDPA, upsample and the matmul family.
  `_SPEC_FLOAT_DTYPES` (`:1550`) *does* include float64, so elementwise and some
  reductions work.
* **Current implementation.** fp64 works for elementwise, not for anything
  structured.
* **Why it is not optimal.** Inconsistent coverage that is hard to predict from
  outside. Apple GPUs genuinely cannot do fp64 (guarded at
  `op_utils/__init__.mojo:478` and at `aten_fast.py:2591`, `:3904`, `:7848`,
  `:7870`) — but CUDA and ROCm can.
* **What the optimized version looks like.** fp64 added to the structured
  kernels' dtype lists, behind the same `has_apple_gpu_accelerator()` comptime
  guard `_parallel_for_dt` already establishes.
* **Expected win.** N/A (coverage).
* **How to measure it.** `tests/test_aten_functions.py` parametrized over
  `torch.float64`.

### E4
**GELU forward sits ~0.3 ms/step above its memory bound**

* **What.** `fast_aten_gelu`, `aten_fast.py:2749`;
  `activation_forward_ops/activation_forward_kernels.mojo` (the exact-branch
  polynomial, journal Change 44).
* **Current implementation.** After Change 44 the exact GELU is one branch with
  no divide: ~197 µs/call against a 175.5 µs/call memory floor on gfx942.
* **Why it is not optimal.** ~0.3 ms/step of pure ALU above the bound; the
  journal's own note says a degree-7 fit over a shorter range would take one FMA
  off, worth about 7% of the compute.
* **What the optimized version looks like.** A shorter-range polynomial fit, with
  an explicit recorded decision about whether to match CPU torch bit for bit.
* **Expected win.** ~7% of GELU's compute, i.e. a fraction of 0.3 ms/step on
  gfx942 — a journal estimate, not a measurement of the proposed fit.
* **How to measure it.** `compare_nanogpt_train_kernels.py`'s GELU group; the
  kernel is isolated enough that `scripts/rocprof_kernels.py --kernel-trace`
  gives a direct before/after.

---

## Compile pipeline and dispatch

### P1
**Warm-path regression from the variant loader: 101 ms/step vs 60 ms on `main`**

> **Update 2026-08-02.** Largely fixed; see `docs/fast_eager_design.md`
> "Compile granularity" for the full table. Measured on H100:
> batch 48 now at **parity** (193.6 vs 193.8 ms), batch 12 at
> **63.6 vs 60.0 ms** (was 66.5 in the pre-fix working tree). The three
> fixes: rule-4 thread-switch synchronize restricted to queue-path
> launches (was ~9 ms/step of full-device barriers), an equal-shape fast
> path around the doubled `torch.broadcast_shapes` in the binary spec
> route (14.2 µs/call), and memoized canonical defines in every
> `*SpecExtension` (2.4 µs/call). The rest of this entry describes the
> remaining ~3.6 ms/step structural gap at batch 12: Python-side output
> allocation, the dispatch bracket (+0.94 µs/op), and prepare/submit
> machinery (~13 µs vs main's ~4 µs for a warm binary add).

* **What.** Recorded in `docs/fast_eager_design.md`, "Compile granularity":
  *"The warm path is currently a regression, not a win. Steady-state training
  step with everything cached: 101 ms on this branch vs 60 ms on main."* The cost
  lives in `eager_kernels/__init__.py` (source-path resolution, define
  normalization) and `eager_kernels/call_queue.py` (queue handling).
* **Current implementation.** Every kernel call resolves its source path,
  normalizes and canonically orders its defines, hashes them, looks up the module
  cache, prepares a queue item and enqueues it — per launch, warm or not.
* **Why it is not optimal.** A 68% steady-state regression on the primary
  training benchmark, self-declared in the design doc. Cold start went
  56.3 s → 11.6 s, which is the trade being made, but the warm path is what runs
  for the other 99.99% of a training job.
* **What the optimized version looks like.** Memoize the whole
  (op, dtypes, flags) → resolved-module decision behind a single dict lookup
  keyed on a cheap-to-build tuple, so a warm launch does no path resolution, no
  define normalization and no hashing. The `_BRIDGE_AVAILABILITY` cache in the
  current working tree (`aten_fast.py:188-202`, memoizing `is_file()` stats worth
  ~9 µs for one tuple) is the same idea applied to a much smaller cost, and is
  the template.
* **Expected win.** Up to the full 41 ms/step gap, i.e. back to ~60 ms — that is
  the measured *size of the regression*, not a measurement of any proposed fix.
* **How to measure it.** `uv run --no-sync python bench_nanogpt_train.py --device
  mojo` with a warm `__mojocache__`, against the same command on `main`.
  `TORCH_MOJO_BACKEND_KERNEL_QUEUE=0` isolates the queue's share from the
  loader's.

### P2
**Strict FIFO: one cold variant blocks every warm launch behind it**

* **What.** `docs/kernel_call_queue.md`, "Queue ordering" rule 1;
  `eager_kernels/call_queue.py`.
* **Current implementation.** `pump()` launches the longest *ready prefix*. A
  queued item whose extension is still compiling stalls everything enqueued after
  it, warm or not.
* **Why it is not optimal.** During cold start — and after any new shape/dtype
  combination appears mid-run — the device idles behind a compile even when the
  next twenty launches are ready and independent.
* **What the optimized version looks like.** Dependency-aware release: an item
  may launch ahead of a blocked predecessor when their keep-alive sets are
  disjoint. The queue already retains every input and output per item (rule 3),
  so the buffer sets needed for the check are in hand. This carries real
  correctness risk — the FIFO rule exists because a consumer must not observe a
  buffer before its producer launches — so the aliasing check must be exact, not
  heuristic.
* **Expected win.** Cold start only; **UNMEASURED**. Steady state is unaffected
  because everything is warm.
* **How to measure it.** `bench_nanogpt_train.py` with `__mojocache__` wiped,
  first-step timing; `TORCH_MOJO_BACKEND_TRACE=1` prints build start/finish
  timestamps to line up against launches.

### P3
**Ops whose Mojo-side gates are not mirrored in Python drain the queue on every call**

* **What.** `_try_spec_unary`, `aten_fast.py:1763` — the
  `ok = False  # e.g. CumsumSpec: constraints not mirrored, stay sync` branch at
  `:1784` and the `force_sync=True` at `:1793`. Same shape in `_try_spec_reduce`
  (`:1800`, `force_sync=True` at `:1830`) and `_try_spec_matmul` (via
  `_submit_spec_matmul`, `:1975`). `mojo_device_aten_ops.py` comments name
  Cumsum, BatchNorm and AttnDecode as keeping the legacy sync form.
* **Current implementation.** `force_sync=True` calls `_call_queue.drain()` — a
  full pipeline flush — then executes inline.
* **Why it is not optimal.** A drain is not just "this op is synchronous"; it
  stalls every *other* pending launch too. `AttnDecodeSpec` in particular is on
  the hot decode path and runs once per layer per token.
* **What the optimized version looks like.** Mirror each op's Mojo-side
  eligibility in Python — the design calls this "Python eligibility stays
  conservatively exact per family" — so it can take the Into form. The three
  named ops need one predicate each.
* **Expected win.** **UNMEASURED.** Proportional to how much work is pending when
  the drain happens.
* **How to measure it.** `bench_gpt2_kernels.py` (decode shapes) and
  `bench_gpt2_batch.py`; `tests/test_call_queue.py` already asserts which ops
  queue and which drain, so the change is directly testable.

### P4
**Cold start is 11.6 s for the first nanoGPT step, and `__mojocache__` cannot be shipped**

> **Hazard found 2026-08-02 (pre-existing, reproduced on the pre-cleanup
> tree).** With a fully cold `__mojocache__` and a workload that performs
> no host reads (e.g. `bench_nanogpt_train.py`'s warmup loop), the queue's
> run-ahead is unbounded: rule 3 retains every input, output, and
> intermediate while any build is still in flight, so several warm-up
> steps of transients stay live at once — observed 71.5 GB across 3,244
> blocks at batch 12 before a 36 MB `memAlloc` failed and the process
> aborted. Real training loops survive because their eval/logging reads
> drain the queue. **Fixed 2026-08-03**: the queue now meters each item's
> retained device bytes and the enqueue that crosses the budget
> (`TORCH_MOJO_BACKEND_QUEUE_BUDGET_MB`, default 8192; 0 disables) drains
> first, waiting builds out. The same cold read-free loop now completes
> in 22 s with an 18.6 GB peak (was: abort at 71.5 GB retained), and the
> warm path is untouched — the meter runs only on the queued branch,
> which a warm pass never takes.

* **What.** `docs/fast_eager_design.md`, "Measured (H100 PCIe, 24-core host)":
  11.6 s first step with the build pool, 56.3 s without.
* **Current implementation.** One `.so` per exact specialization, built on demand
  on a background pool sized by `_pool_size()` (RAM / 5 GiB, cores / 3, cap 16).
  Caches are per-machine: `mojo build` defaults to `-march=native` and detects the
  GPU arch, so a prebuilt `.so` can SIGILL elsewhere.
* **Why it is not optimal.** 11.6 s is the honest cost of the design and already
  5x better than the alternative, but there is no warmup command, so every
  container image and every CI run pays it. The doc itself proposes the fix.
* **What the optimized version looks like.** An explicit `warmup` entry point
  that enumerates and builds the variants a named workload needs, plus CI caching
  of `__mojocache__` keyed on source hash + toolchain version **+ runner
  hardware**.
* **Expected win.** Moves 11.6 s from first-run to build time. Not a throughput
  win.
* **How to measure it.** Wipe `__mojocache__`, time the first step of
  `bench_nanogpt_train.py --device mojo`, with `TORCH_MOJO_BACKEND_TRACE=1`.

---

## Graph backend (`torch.compile(backend=mojo_backend)`)

### Q1
**softmax / log_softmax are hand-decomposed instead of using MAX's fused ops**

* **What.** `aten_softmax`, `torch_mojo_backend/aten_functions.py:691`;
  `aten__log_softmax`, `:719`.
* **Current implementation.** Explicit `amax` → subtract → `exp` → `sum` →
  divide, built op by op in the MAX graph.
* **Why it is not optimal.** MAX has `ops.softmax` / `ops.logsoftmax`, which
  lower to a fused kernel. The eager path already uses a fused `SoftmaxSpec`
  (`aten_fast.py:7710`), so the two backends disagree on the same op. Whether
  MAX's graph compiler re-fuses the decomposition is **unverified**.
* **What the optimized version looks like.** Call `max.graph.ops.softmax` /
  `logsoftmax` directly, keeping the `half_to_float` cast handling.
* **Expected win.** **UNMEASURED**, and possibly zero if MAX already fuses it —
  check before spending effort.
* **How to measure it.** `uv run --no-sync python bench_gpt2_compile.py mojo
  compile-max`; `TORCH_MOJO_BACKEND_VERBOSE=1` dumps the graph.

### Q2
**GroupNorm is reimplemented in the graph backend and returns `NotImplementedError` objects for mean/rstd**

* **What.** `aten_native_group_norm`, `aten_functions.py:2735`;
  `torch_group_norm_equivalent`, `:2777`.
* **Current implementation.** A reshape-and-reduce composition. It guesses `H`
  and `W` from `sqrt(HxW)` and falls back to `(HxW, 1)` when that is not square
  (`:2757`). Mean and rstd are returned as `NotImplementedError` *instances*
  inside the output tuple.
* **Why it is not optimal.** MAX has `ops.group_norm`. The eager path has a fused
  `GroupNorm` kernel that returns real mean/rstd (`aten_fast.py:5285`). The
  spatial-shape guess is a latent hazard for any consumer that needs the real
  `H`/`W`; the normalization itself is over `channels_per_group × HxW`, so the
  guess is harmless *today* — but it is load-bearing on that fact.
* **What the optimized version looks like.** Route to `max.graph.ops.group_norm`,
  return real statistics, delete the shape guess.
* **Expected win.** **UNMEASURED**; the correctness tidy is the stronger
  argument.
* **How to measure it.** `tests/test_aten_functions.py`; `demo_scripts/vgg.py`
  and `densenet.py` through `torch.compile`.

### Q3
**Graph `max_pool2d_with_indices` returns the values as the indices**

* **What.** `aten_max_pool2d_with_indices`, `aten_functions.py:2503`; the
  returned tuple at `:2540` — `forward_result,  # This is wrong but needed for
  eager mode` — with the intended `NotImplementedError` commented out, under a
  `# TODO: re-enable notimplementederror` at `:2500`.
* **Current implementation.** The second output is the first output.
* **Why it is not optimal.** Any consumer of the indices (max-unpool, the pooling
  backward) gets silently wrong data. It is masked by xfails rather than fixed.
  The eager backend already has a correct `MaxPool2dWithIndices` kernel
  (`aten_fast.py:5135`) that can serve as the reference.
* **What the optimized version looks like.** Either compute real indices in the
  graph, or restore the `NotImplementedError` so it fails loudly.
* **Expected win.** N/A (correctness).
* **How to measure it.** `tests/test_aten_functions.py`, comparing indices
  against CPU torch.

### Q4
**Graph SDPA with `attn_mask` decomposes to full-matrix attention**

* **What.** `aten_functions.py:478` onward — when `attn_mask is not None` it
  builds `scores = matmul(query, key^T) * scale`, adds the mask, softmaxes and
  matmuls again.
* **Current implementation.** Flash attention is used only for the unmasked case.
* **Why it is not optimal.** The same `O(T²)` materialization as [A1](#a1), for
  the case most HuggingFace models take. Unlike eager, at least it *works*.
* **What the optimized version looks like.** An additive-mask operand on MAX's
  flash attention if it supports one; else the causal form when the mask is
  recognizably causal.
* **Expected win.** **UNMEASURED.**
* **How to measure it.** `run_hf_mojo.py` and `bench_gpt2_compile.py`.

### Q5
**Graph conv and pooling insert explicit NCHW↔NHWC permutes**

* **What.** `aten_convolution`, `aten_functions.py:1664` (`input.permute([0, 2,
  3, 1])`, the weight permute below it, and the result permute back);
  `aten_max_pool2d_with_indices`, `:2522`.
* **Current implementation.** MAX's conv wants NHWC/RSCF; PyTorch hands over
  NCHW/KCRS; the adapter permutes on both sides of every call.
* **Why it is not optimal.** Up to three layout copies per convolution *if they
  survive compilation*. Whether MAX's graph compiler folds them into the conv's
  layout selection is **unverified** — this is the most likely false positive in
  this document.
* **What the optimized version looks like.** Verify first. If they survive,
  propagate NHWC across a run of convolutions and permute only at the boundaries.
* **Expected win.** **UNMEASURED** and possibly zero.
* **How to measure it.** `TORCH_MOJO_BACKEND_VERBOSE=1` to dump the graph, then
  `bench_mojo.py resnet` and `demo_scripts/vgg.py` through `torch.compile`.

---

## Cross-cutting: architecture gating

Every route below is written for one architecture. On any other, the listed
fallback runs. This is the answer to "which GPUs run an unoptimized path".

| Route | Gate | file:line | Everything else runs |
|---|---|---|---|
| BF16 dense GEMM bridge | `cuda` + `sm_90a` | `aten_fast.py:6880` | SIMT FFMA `MatmulSpec` |
| BF16 batched (BMM) bridge | `cuda` + `sm_90a` | `aten_fast.py:7029` | SIMT FFMA `BmmSpec` |
| TF32 GEMM / BMM / linear bridge | `cuda` + `sm_90a`, **and** precision ≠ `highest` | `aten_fast.py:6962`, `:7095`; precision at `:6948`, `:7083`, `:7179` | SIMT FFMA `MatmulSpec` |
| MFMA dynamic GEMM (fp32 + bf16) | `amdgpu:gfx942` | `matmul_ops.mojo:4025`, `:6498` | SIMT FFMA tiles |
| Copy-free transposed-A GEMM (`_tn_mfma_route`) | `amdgpu:gfx942` + bf16 | `matmul_ops.mojo:6857-6867` | Python-side copy ([G1](#g1)) |
| Apple simdgroup GEMM (NN/NT/TN) | `has_apple_gpu_accelerator()` | `matmul_ops.mojo:4056`, `:6416` region | SIMT FFMA tiles (the source says 3–15x behind torch MPS) |
| Apple transposed-A spec route | `has_apple_gpu_accelerator()` | `matmul_ops.mojo:6755` | copy |
| FA4 flash attention (fwd + bwd) | `cuda` + `sm_90a` + bf16/f16 + d64/d128 + causal + `seq % 128 == 0` | `aten_fast.py:6482` | math decomposition ([A1](#a1)) |
| Fused flash attention (fwd + bwd) | `hip` + `gfx942` | `aten_fast.py:6096-6097` | math decomposition ([A1](#a1)) |
| Causal batched GEMM (`_try_sdpa_causal_bmm`) | `gfx942` | `aten_fast.py:5448` | dense BMM |
| Causal BMM (`BmmCausalF32`) | Apple | `matmul_ops.mojo:6321`, raise at `:6369` | dense BMM |
| Fused SDPA backward (5 launches) | `metal` + fp32 | `aten_fast.py:6604`; kernel raise at `sdpa_backward_gemm_kernels.mojo:281` | composed sequence + 2 transposes ([A3](#a3)) |
| Fused softmax+dropout forward | `metal` + fp32 | `aten_fast.py:5581` | separate softmax + dropout |
| Fused 3-way `cat` | `metal` | `aten_fast.py:3621` | 3 `NarrowCopyDst` launches |
| `fast_aten_add` spec-bypass fast path | `metal` | `aten_fast.py:2364` | spec path |
| Foreach elementwise family, fixed-arity 8-slot ABI | `has_apple_gpu_accelerator()` | `foreach_batched_kernels.mojo` (`foreach_ew_enqueue`) | the descriptor-batched kernel (same body, same descriptors) |
| Apple vectorized D2D copy | `has_apple_gpu_accelerator()` | `tensor_holder.mojo:326` | `enqueue_copy_from` |
| Apple permute rows4 / rowloop | `has_apple_gpu_accelerator()` | `data_movement_ops.mojo:321` | generic gather ladder |
| Apple fused AdamW variant | `has_apple_gpu_accelerator()` | `optimizer_kernels.mojo:339` | generic fused AdamW |
| Apple layer-norm forward variant | `has_apple_gpu_accelerator()` | `normalization_forward_kernels.mojo:558` | generic |
| Embedding-backward table regime | **not** Apple | `embedding_backward_kernels.mojo:637` | per-row atomics path |
| 32-byte permute gather | **not** Apple (documented driver wedge) | `data_movement_ops.mojo:377` | 16-byte then 8-byte |
| fp64 on GPU | **not** Apple | `op_utils/__init__.mojo:478` | raises on Apple |

### X1
**Tuning constants fitted to one card and applied to all**

* **What.** The roster below; each entry is a `comptime` constant that selects
  grid geometry or a dispatch threshold.
* **Current implementation.** Each was swept on one card and then compiled into
  every build. Some record the sweep beside them, some do not.
* **Why it is not optimal.** They share a failure mode: they change *launch
  geometry*, so `scripts/compare_kernel_asm.py` reports byte-identical device
  code and gives a false all-clear — AGENTS.md calls this out explicitly, with
  the log-softmax grid floor as the worked example. A change tuned on one card
  can therefore silently retarget another with no diff to show for it.

| Constant | file:line | Fitted on | Labelled in source? |
|---|---|---|---|
| `_LLC_BYTES = 256 MiB` | `op_utils/__init__.mojo:160` | gfx942 (Infinity Cache) | yes, with a full sweep table |
| `_BW_MAX_BLOCKS` | `op_utils/__init__.mojo:163` | gfx942 | partially |
| `TARGET_BLOCKS = 342` | `matmul_ops.mojo:152` | 3 × H100 SM count ≈ MI300X CU count | partially — the comment names H100's 114 SMs but the constant is used everywhere |
| `_TARGET_BLOCKS = 1280` | `normalization_backward_params.mojo:34` | unstated (≈ 4 × 304) | **no** |
| `_REG_BLOCKS_PER_CU = 4` | `softmax_backward_kernels.mojo:64` | gfx942 | yes, with the sweep |
| `_NOSMEM_THREADS = 1024` | `softmax_backward_kernels.mojo:59` | gfx942 | yes, with the sweep |
| `TN_TARGET_BLOCKS = 80` | `apple_gemm_tn_kernels.mojo:80` | Apple | yes (Apple-only file) |
| `NT_TARGET_BLOCKS = 80` | `apple_gemm_nt_kernels.mojo:71` | Apple | yes (Apple-only file) |

* **What the optimized version looks like.** Derive from device attributes where
  one exists — `ctx.default_device_info.sm_count`
  (`op_utils/__init__.mojo:179`) and `ctx.get_attribute(DeviceAttribute...)`
  (`gemm16_v3_kernels.mojo:1701`) are both already used in this tree — and
  where a constant must stay fixed, state the card and the sweep next to it.
* **Expected win.** **UNMEASURED** on every card except the one each was fitted
  on.
* **How to measure it.** Per-constant sweeps on the target card: `ncu` on NVIDIA,
  `scripts/rocprof_kernels.py --kernel-trace` on AMD.

---

## Not audited

Stated explicitly so nobody reads this document as complete.

* **No performance measurements were taken for this document.** The only thing
  executed was `torch.get_float32_matmul_precision()`. The GPU is shared with
  other agents and the brief asked for a short leash on it. Every "expected win"
  is therefore either a citation of an existing measurement (labelled with the
  card it was taken on) or **UNMEASURED**.
* **Nothing was verified on gfx942, Apple, or any NVIDIA part other than
  `sm_90a`.** All AMD and Apple claims are read from source and from
  `optimization_journal.md`.
* **The MAX/Mojo library itself was not audited.** Whether MAX's graph compiler
  fuses the decompositions in [Q1](#q1) or elides the permutes in [Q5](#q5) is
  unverified, and those two items may be worth zero.
* **`eager_flash_attention/` (the FA4 module, ~6 kLOC of Mojo) was read only at
  the entry-point and gating level.** Its kernel bodies —
  `fa4_fwd_kernel.mojo`, `fa4_bwd_kernel.mojo`, `fa4_wgmma_f16.mojo` — were not
  reviewed for optimization opportunities.
* **`gemm16_matmul_ops/` (~7 kLOC across five files) was read only at the dispatch
  level.** The v3/v4 kernel bodies and their tile-selection heuristics were not
  reviewed. `optimization_journal.md` is the authority there and states the
  gfx942 GEMM core is close to exhausted.
* **CPU-device performance was not audited at all.** Several kernels have a CPU
  arm (`ctx.api() == "cpu"`) using `elementwise[..., simd_width=1]`, e.g.
  `_copy_strided` (`op_utils/__init__.mojo:728`) and `_permute_copy`
  (`data_movement_ops.mojo:261`). Whether that is a real cost on the CPU mojo
  device is unmeasured.
* **The graph backend (`aten_functions.py`, ~4 kLOC) was sampled, not swept.**
  Only softmax, group norm, convolution, pooling and SDPA were examined; the
  other ~200 op implementations were not.
* **Op *coverage* was inventoried only where it intersected performance.** No
  full "which ATen ops raise on the mojo device" list was produced. Holes noted
  in passing: `topk`, `sort`, `gather`, `index_select`, `scatter_add`, `div` with
  `rounding_mode` (`aten_fast.py:2498`), `softmax` with `dtype=`
  (`aten_fast.py:7714`), `conv1d`/`conv3d`/transposed conv, `ceil_mode` pooling,
  and the vision-training backwards.
* **No benchmark exists in this repo for convolution or for vision training.**
  The `bench_*.py` family covers GEMM, GPT-2 decode and nanoGPT training only.
  Items [C1](#c1)–[C4](#c4) and [N2](#n2) therefore have no measurement harness
  at all; building one is a prerequisite for working on them.
* **Numerical accuracy was out of scope.** This document is about speed and
  coverage; the one correctness item included ([Q3](#q3)) is here because it is
  masked by xfails rather than tracked.
