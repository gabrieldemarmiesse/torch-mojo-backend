# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

This is a PyTorch backend implementation using Modular's MAX framework. The project demonstrates how to create custom PyTorch compilation backends that bridge PyTorch operations to MAX/Mojo implementations. It also has support for eager mode.

## Setup

Make sure a copy of the repository https://github.com/modular/modular and https://github.com/pytorch/pytorch is available for you to grep things and explore. If you cannot find them locally, clone them in `/tmp`, checkout to the right branch/commit (the same as the one in the pyproject.toml).

## Common Commands

```bash
# Run tests (with parallel execution)
uv run pytest -n 15

# Run specific test file
uv run pytest tests/test_compiler.py

# Run with profiling enabled
TORCH_MOJO_BACKEND_PROFILE=1 uv run pytest tests/test_compiler.py

# Run with verbose output (shows graph structures)
TORCH_MOJO_BACKEND_VERBOSE=1 uv run pytest tests/test_compiler.py

# Run linter/formatter
uv run ruff check .
uv run ruff format .

# Or use pre-commit for all checks
uvx pre-commit run --all-files
```

Always use uv to run commands to ensure the correct environment is activated. Never run python directly. Never run the whole test suite because it's too slow. Only run tests for the specific files you are working on.


## Development Notes
- **Code Quality**: Uses Ruff for linting/formatting with Python 3.11+ target and pyupgrade rules
- **Debugging Tools**:
  - Environment variables for profiling and verbose output
  - Graph visualization when `TORCH_MOJO_BACKEND_VERBOSE=1`
  - Eager-mode kernel builds and launches are described in
    `docs/kernel_call_queue.md`. `TORCH_MOJO_BACKEND_KERNEL_QUEUE=0` is the kill
    switch (every kernel builds and launches inline),
    `TORCH_MOJO_BACKEND_FORCE_KERNEL_QUEUE=1` turns the queue back on under the
    test suite, and build timings print by default
    (`TORCH_MOJO_BACKEND_TRACE=0` silences them).
- **Model Examples**: `demo_scripts/` contains examples showing real-world usage:
  - GPT-2, Gemma3 (LLM models)
  - VGG, DenseNet (vision models)
  - `no_graph_breaks.py` (example demonstrating graph compilation without breaks)


## Performance-regression benchmarks (`benchmarks/`)

`benchmarks/` is a pytest suite that measures the mojo device against stock
PyTorch (CUDA on NVIDIA, ROCm on AMD, MPS on Apple) and compares the
device-time ratio ours/stock against `benchmarks/baselines.html` — one
git-tracked file holding the baselines of every hardware configuration side
by side, **and** the viewer that renders them (open it in a browser, off
disk: the data is a JSON block inside the page, so there is nothing to
fetch and nothing to serve). Each (op, shape, layout, dtype) case is its own
test node and the node id is its key in the JSON, so the list of failing test names is the
list of regressed kernel regimes. A test fails when its ratio regresses more
than 8% versus the recorded baseline, passes when this hardware has never
been measured, and the whole suite skips cleanly when there is no
accelerator. GPU **device time only** is measured (never wall time), the
two legs are interleaved in ABBA order (ref,ours | ours,ref) so a monotonic
clock ramp cancels to first order, every case is pre-warmed, and
`/tmp/gpu_lock_0.lock` is flocked around GPU work.

```bash
# Full suite (read-only: never writes baselines). Run serially, never -n.
uv run pytest benchmarks/

# Subsets are ordinary pytest selection — no markers, no custom taxonomy
uv run pytest benchmarks/test_gemm.py -k "TN and bf16"
uv run pytest "benchmarks/test_gemm.py::test_mm[S7_768x50304x49152-TN-bf16]"

# Record a baseline on new hardware, or update after an optimization
# (writes new entries and >8% improvements; +/-8% dead band never churns)
uv run pytest benchmarks/ --update-baselines

# Accept a >8% regression after an INTENTIONAL trade-off, for chosen nodes only
uv run pytest benchmarks/ -k "..." --update-baselines=force

# Where are we far behind stock PyTorch? (reads only the JSON, no GPU needed)
uv run python benchmarks/report.py --worst 20

# Two GPU-free reconciliations: every registered op is benchmarked or has a
# documented skip, and every recorded baseline still has a test node
uv run pytest benchmarks/test_coverage.py
```

`baselines.html` is one file with two halves. The `<script
type="application/json" id="baselines">` block at the bottom is the data,
machine-written by `--update-baselines`; everything above it is the viewer,
hand-written source you edit like any other file. Writes splice only the
block, so a benchmark run never touches the viewer and a viewer change
never touches the numbers.

The tree is hardware → aten op → dtype → shape → layout, and the data is
**measurements only** — one ratio per op/dtype/shape/layout leaf, nothing
derived from them. Per-branch min/median/max are computed
while reading: the viewer prints them on every row of its collapsible tree
(click a branch to open it), and `report.py` prints them in the terminal.
The page also takes another baselines file as a query parameter —
`baselines.html?url=<url>` — so one machine's file renders another
machine's numbers (a raw file URL, a CI artifact, another branch) without a
checkout, and accepts a dropped `baselines.html` or raw `.json` too. Note
that github.com shows `.html` as source rather than rendering it: download
the file, or route it through a raw-HTML viewer such as
`htmlpreview.github.io`.

Updates merge per entry: a run that measured three cases changes those
three lines of `baselines.html` and nothing else, so a PR touching one op
updates that op's numbers without rerunning the rest. Do not gate CI on
this suite: CI machines may have no GPU — except
`benchmarks/test_coverage.py`, which needs no accelerator and is safe to
gate on: it reconciles the registered ops against what is benchmarked,
and every recorded baseline against the test nodes that address it (a
shape id, dtype id or op token IS a baseline key, so renaming one without
renaming its recorded entries silently retires them — that check is what
catches it).


## To add support for an op

To add support for a new ATen operation, follow this test-driven development process:

### Step 1: Research the Operation
Ask a subagent to explore the PyTorch codebase `pytorch` (clone it if it's not there yet, the venv is not enough, it might not have the C++ code) and look for:
- The signature of the ATen function
- The meaning of inputs and outputs
- Any important behavioral details
- Request a full report with this information

You can skip this step if the user provided the signature and the details of the operation in the initial request.

### Step 2: Write Unit Tests
Write unit tests in `test_aten_functions.py` using this op directly:
- Place tests somewhere in the middle of the file to avoid merge conflicts
- Use `pytest.mark.parametrize` to test multiple input data types and shapes
- Test edge cases and different parameter combinations

You shoud check in the unit test that the aten function has been called with this pattern:

```python
def test_aten_min_no_dim(conf: Conf, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_min)

    def fn(x):
        return aten.min(x)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])
```

### Step 3: Run Tests (Expected to Fail)
Run the unit tests:
```bash
uv run pytest tests/test_aten_functions.py::test_your_new_op -v
```
You should see an error message explaining that the ATen op is not supported.

### Step 4: Add Operation Signature to aten_functions.py
- Find the alphabetically correct position in `aten_functions.py`
- Add a comment with the full ATen operation signature
- **IMPORTANT**: The file is sorted alphabetically and must remain this way

Example:
```python
# aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
```

### Step 5: Research MAX Implementation
Ask a subagent to explore the directory `../modular/max` to find:
- MAX functions that do something similar (sometimes there are direct equivalents)
- Functions that can be composed to re-implement the operation
- Check models created with MAX for usage examples
- Look in `kernels.py` for complex operation implementations
- Request a full report of useful functions with descriptions of inputs/outputs

### Step 6: Implement the Operation
Write the ATen operation implementation in `aten_functions.py` just below the signature comment:

**Important**: `aten_functions.py` serves the **torch.compile backend only**
(the mojo-device eager mode has its own fast implementations — see Step 7).
The implementation must still accept both value types the compile backend can
feed it:
- `TensorValue` (symbolic tensors, real graph building)
- `MaxEagerTensor` (MAX's eager interpreter — the test suite runs with
  `MAX_USE_EAGER_INTERPRETER=1`)

Use the type hint `MaxTensor = TensorValue | MaxEagerTensor` for tensor parameters.

Example implementation:
```python
# aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
def aten__log_softmax(
    self: MaxTensor, dim: int, half_to_float: bool
) -> MaxTensor:
    # Implementation using MAX operations that works for both modes
    return F.log_softmax(self, axis=dim)
```

### Step 7: Register for Eager Mode Execution
Eager mode has **no graph fallback**: every op is either bound to a fast
implementation (Mojo kernels over raw pointers) or raises
`NotImplementedError`. (The old `wrap_for_mojo_device` wrapper no longer
exists.) Two places are involved:

1. Write the fast implementation `fast_aten_<op>` in
   `torch_mojo_backend/eager_kernels/aten_fast.py`. It receives
   `TorchMojoTensor`s (a Mojo `TensorHolder` ownership token plus `_ptr` /
   `_shape` / `_strides` / `_offset` / `_dtype` / `_device`) and runs one or
   a few kernel calls from an `eager_kernels/<family>/` extension. View ops
   are zero-copy wrapper math (no kernel at all). Return the `NOT_HANDLED`
   sentinel to decline inputs you don't handle — the registration turns it
   into an actionable `NotImplementedError`.
2. Bind it in `torch_mojo_backend/mojo_device/mojo_device_aten_ops.py`, in
   alphabetical order within the file:

   ```python
   _register_fast("aten::<op>", "fast_aten_<op>")
   ```

   Related helpers: `_out_variant(...)` wraps a functional fast impl as an
   `out=` variant; `_register_foreach_inplace(...)` covers `_foreach_*_`
   ops; operations requiring custom device handling (like
   `aten::_copy_from`) use `@register_aten_op("aten::<op>")` on a
   hand-written function directly.

If the op needs a new Mojo kernel, add it to the matching
`eager_kernels/<family>/<family>.mojo` (variant-gated: the loader compiles
one specialization per (OP, DTYPE) on first use and caches it in
`__mojocache__`). You may import kernels from the modular repo inside the
`.mojo` file (`from nn import ...`) only if they don't call
CuBLAS/CuDNN/rocBLAS underneath. If a fully dynamic-shape function is not
available in the modular repo, write the kernel yourself.
If you have access to multiple gpus, the aten function should work on all those gpus.

Name kernels after the algorithm, never after a model or a workload. The
`@__name` string is what CUPTI, torch.profiler, Nsight and rocprof print, so a
user profiling their own model reads it: `bf16_gemm_tn_ws_m64n128_tma_col_a_s3`
tells them what ran, `nanogpt_bf16_gemm_...` tells them something false. The
house grammar is `<algorithm-or-route>_<layout/regime>_<dtype>_<tuning params>`
with no project prefix — `pure_gemm_pipe3_...`, `amd_splitk_mfma_...`,
`fa_mfma_...`, `lsm_bwd_...`. Recording the workload a tuning constant was
fitted to belongs in a comment next to the constant, not in the kernel name.

### Step 8: Re-run Tests
Run the unit tests again and verify they pass:
```bash
uv run pytest tests/test_aten_functions.py::test_your_new_op -v
```

Test both execution modes if applicable:
- Graph mode via `torch.compile(backend=mojo_backend)`
- Eager mode via tensors on `torch.device("mojo")`

### Step 9: Run Linter
Make sure to run the linter:
```bash
uvx pre-commit run --all-files
```

**Do not run the whole test suite** as it takes too long. Only run tests for the specific operation you added.

### Summary: Implementation Checklist
When adding an operation, you typically update **three places**:
1. **`aten_functions.py`**: torch.compile backend implementation (MAX ops composition)
2. **`eager_kernels/aten_fast.py`**: fast eager implementation over
   `TorchMojoTensor` (plus a Mojo kernel in `eager_kernels/<family>/` when needed)
3. **`mojo_device_aten_ops.py`**: the `_register_fast(...)` binding for eager mode

This ensures the operation works in both `torch.compile()` and on the `mojo` device.


## Type hints

Every function written must have type hints in its signature: annotate all
arguments and the return type. Do not put type hints in the function body
(no annotated local variables like `x: Foo = ...`).

Use `: object` only when no other option is possible. Prefer precise types,
unions, `Protocol`s (e.g. `MojoTensorLike` for payload-level helpers), or a
`TypeVar` for pass-through functions.

These hints are enforced at runtime by beartype, but only while running the
unit tests: `tests/conftest.py` sets `TORCH_MOJO_BACKEND_BEARTYPE=1`, and
production imports leave it off so no wrapper frames are added to hot paths.
Set the variable explicitly to enable or disable checking in other contexts.

## To find the correct type hints for a function
It may be hard to find the correct type hints for a function. What you should do in this case is:
1) Add an obviously wrong type hint, for example datetime.timezone in an aten function.
2) Run an existing unit test that calls this function.
3) Beartype will throw an error and give the name of the type being actually passed to the function.
4) Replace the type hint by the type given by beartype.
5) Run the unit test again to check that it works.
6) Run the whole test suite to verify that the type hint shouldn't be wider.

## Rules about the eager mode

Read this especially if you're an agent doing code review.

1) The user should be able to use `my_tensor.to("mojo")` and use their gpu, even if they have a CPU-only install of PyTorch. That means that when writing kernels, we can't use CuBLAS, CuDNN, RocBLAS, or any other lib that would be available only if torch-gpu was installed. We want to stand on our own legs. We can use and import mojo functions from the modular repository (`from nn import ...`) but only if it's not calling CuBLAS, CuDNN... underneath. Our motto should be "pip install torch-mojo-backend with the minimal pytorch install (cpu) and use your gpu.".
2) We cannot use the C++ interface of pytorch. We use JIT compilation to compile extensions only when they're first called. We want to be compatible with many PyTorch versions and we don't want to force the user to install a C++ compiler. So we must use Python extensions in mojo.
3) You cannot use information about the tensors other than the shape, stride, dtype, pointer in the eager mode. While it's tempting to "keep a history of some past op to do fused ops", it will not improve the performance for all workflows. Pytorch uses Aten ops and decompositions, so sometimes, if you want to implement a fused op, you might want to target a higher level aten function, before it gets decomposed. Aten ops that are not implemented are decomposed automatically in pytorch. 
4) Do not write kernels that work only for a very specific shape. Input shapes should be dynamic to avoid recompiles. While it's tempting to make things faster, a user trying a slightly different shape will not benefit from the optimisations of this kernels. It's fine to write different kernels for different regimes (e.g. a kernel for big shape, small shapes, square shapes, rectangular, power of two, etc...) and then do dynamic dispatch based on the input shapes. It's not because we optimize for a given model that we can hardcode at compile-time all the shapes of the kenels to make it faster. So do multiple flexible kernels + dispatch, do not do kernels for hardcoded shapes + fallback.
5) When asked to optimize a model, the answer should never be "change the code of the model". The model is user-defined, we have no control over it. We just control what we do with the tensors we're given by pytorch.
6) Do not over-allocate or write your own memory allocation. It's the job of Modular to write a good memory allocator. Allocate normally what you need, and do not write in the memory of the input tensors, unless the signature of the aten function specifies that it's what we should do. For example, at::add_ expects us to write in the input tensor memory, and it's a valid use case to re-use the inputs. Our ops should not read the refcounts and try to use this information to change the logic of the kernel. A refcount is only there to free the memory when needed.

## To check a kernel change against a GPU you do not have

You will normally be optimizing on whatever GPU is in the machine, but the
kernels are one source compiled for every backend, so a change tuned on one
architecture can silently retarget another that you cannot benchmark. Do not
guess and do not assume "I only touched the AMD path" — check it, because
Mojo cross-compiles and the check is cheap:

```bash
uv run python scripts/compare_kernel_asm.py \
    --before /path/to/baseline_worktree --after . --accelerator sm_90a
```

`mojo build --emit asm --target-accelerator <arch>` needs no such device
present. It writes host assembly to `-o` and one sidecar per GPU kernel beside
it (`.ptx` for NVIDIA, `.amdgcn` for AMD, `.ll` for Metal). The script builds
both trees, pairs kernels by name, masks the mangling hash and reports which
ones differ. It takes about a minute per module against roughly ten for one
end-to-end benchmark leg. `mojo --print-supported-accelerators` lists the
architecture names.

Read the result as follows:

- **No kernel differs** — the change cannot have moved that architecture's
  device code, and no measurement there is owed. This is the common answer when
  a new route is properly gated behind
  `comptime if _accelerator_arch() == "amdgpu:gfx942"`.
- **Some kernel differs** — that architecture is in scope. Either gate the
  change, or say plainly in the PR that its effect there is unmeasured. A
  reviewer with the other card can then measure exactly those kernels.

**This covers device code only, and that is the trap.** Launch geometry lives in
host code: grid size, block count, and the thresholds that choose between
kernels. Anything keyed off `sm_count`, an L2 budget or a blocks-per-CU constant
can retarget a GPU while emitting byte-identical assembly. A real example from
this repo: flooring the log-softmax forward grid at `4 * sm_count` moved an H100
from 114 blocks to 528 without changing one PTX instruction. So diff the
assembly *and* read the dispatch by hand; a clean assembly diff is not a clean
bill of health.

When a tuning constant was measured on one architecture, say so where it is
defined, the way `LSM_BLOCKS_PER_CU` and `LSM_L2_BUDGET` do. The next agent
needs to know which numbers are portable and which were fitted to one card.


## Optimizing a kernel

**The acceptance bar is 10%.** A kernel is done when its device time is
within 10% of stock PyTorch on the same shape — i.e. `ours / torch <= 1.10`.
That is the only pass/fail criterion. It is not 2%: an earlier revision of
this file said 2%, it was too strict, and agents have since quoted the old
number from memory and reported passing cells as failures. If you are about
to judge a result against any threshold other than 10%, re-read this line.

The roofline is a **diagnostic, not a gate**. It tells you whether a
remaining gap is physics or engineering; it never decides whether the work is
done. A kernel inside 10% of PyTorch is finished even at 50% of roofline,
because PyTorch itself is often nowhere near roofline.

When optimizing a kernel, you should make a harness for a subagent A to work on. The harness should include:
- Unit tests for the kernels (outside the main test suite), the unit tests should acquire the flock `/tmp/gpu_lock_{gpu_id}.lock`.
- A benchmark script in pure mojo, that measures the performance of the kernel on different input shapes (no more than 6), requiring at least one non-round/awkward shape (e.g. 357×789). The benchmark should lock the GPU frequency if possible. The benchmark should use a flock in /tmp/gpu_lock_{gpu_id}.lock to avoid using the gpu at the same time as other benchmarks. `ncu` or rocprof or equivalent should be given to the agent.
- The harness should only measure the total gpu time, not the wall time as the wall time can be worked on later on.
- The harness should also include reference numbers, so roofline estimate, and the performance of stock pytorch on the same input shapes (device time too, not wall time).
- The scope of the agent should be as limited as possible, for example, if writing a gemm, the agent should only take care of the TN variant, or NT but not all variants. It should only focus on one dtype. (if multiple dtypes are needed, we'll do the dance Agent A, Agent B for dtype1 and then Agent A Agent B for dtype2, etc..., with a bit of luck for dtype2, agent A can reuse the code of dtype1 and just change the dtype, which will be easy to integrate later by agent B by parametrizing the code).
- The agent should write the kernel outside the codebase, but the agent can import code from it, or import code from the modular repo. This is to avoid having the agent work on integrating the kernel into the codebase, the agent should only focus on the kernel itself. 
- The performance work is done when the kernel is within 10% of the speed of stock pytorch on the same input shapes (device time), as stated at the top of this section. Keep the roofline estimate in the harness as a diagnostic only.

A small agent A2 should be used for a quick code review, notably just check that the kernel respects the rules of the eager mode and the tests are passing. No need for a very smart model here.

When subagent A is done, a subagent B should start to integrate the kernel into the codebase. The subagent B should first:
- Add those harness tests in the ` tests/test_eager_kernels.py` file.
- Add benchmarks for the `benchmarks/` directory.  `ncu` or rocprof or equivalent should be given to the agent.
- Export the ptx/asm of the kernel into a temporary directory.
- Then integrate the kernel into the codebase, make sure the tests are passing and the benchmarks are as good as before. The subagent B can also generate the ptx/asm of its implementation to help, even if it doesn't need to match exactly the ptx/asm of the agent A kernel. The subagent B should take the decision to either use metaprogramming to adapt a kernel already in the codebase to perform the work of the new kernel given some specific parameter, or to write new code. Duplication should be avoided if possible so if the agent find out that some function/piece of code is already used in the codebase, the agent should perform a refactoring to reuse this code.
- The agent B should not degrade the performance of existing kernels, dtypes, shapes, other gpus, etc...

When subagent B is done, a subagent C should do a code review of B and run benchmarks to make sure that the performance has not regressed for other dtypes, shapes, similar kernels, other gpus etc... Cross-compilation can be used to check that the assembly/ptx didn't change much. But worst case scenario, other gpus are available to run benchmarks through ssh.

When agents A, B and C are all done, all the temporary files of agent A should be removed and a commit should be made.

## Benchmark harness notes

Hard-won facts from previous kernel-optimization engagements on this codebase.
Believe them before rediscovering them at GPU-hour prices.

### Measuring

- One GPU, many agents: EVERY GPU-touching command (harness runs, `ncu`,
  `pytest`, one-off probes) goes through `flock /tmp/gpu_lock_{gpu_id}.lock`.
  Never compile under the lock — `mojo build` takes minutes and needs no GPU;
  build first, then take the lock to run.
- Lock the clocks before taking reference numbers (`nvidia-smi -lgc` on
  NVIDIA; pick a sustainable frequency below max so power throttling cannot
  bite mid-suite). Record the chosen clock next to the numbers, recompute the
  roofline at that clock, and reset with `nvidia-smi -rgc` when the
  engagement ends.
- Time two ways: per-launch (sync every iteration; includes host enqueue
  overhead) and streamed (N back-to-back launches, one sync). The streamed
  number is the one comparable to torch-profiler GPU time and to the stock
  pytorch reference.
- Thermal ordering bias is real on power-limited cards: a config benchmarked
  later in a hot run can read several percent slower (~8% observed on hot
  GEMM shapes). Never conclude from a single in-run ordering — alternate
  baseline/candidate order between runs, or compare both against an absolute
  reference. Inside one case `benchmarks/` already does this (ABBA), and the
  reason is worth borrowing for hand-rolled harnesses: under a fixed A,B,A,B
  order the second leg is always measured one burst later than the first, so
  a steady ramp tilts every pair the same way. That bias is invisible to any
  scatter-based error bar — a measured example converged to "0.000%
  uncertainty" on a number 2% wrong.
- Before optimizing anything, the harness baseline leg must call the actual
  production entry points and reproduce the in-process production numbers
  (within ~1%). If it does not, the harness is measuring something else and
  every later conclusion is suspect.
- Fill inputs with a deterministic hash (not an RNG) so the host-side
  correctness check can recompute exact expected values on sampled output
  elements. Accumulate the host reference in fp64 and calibrate the tolerance
  to fp32 accumulation error (grows roughly like sqrt(k) for tiled sums).
  A tolerance failure after a restructured accumulation is a kernel bug, not
  a tolerance bug.
- `ncu` replays kernels: keep iteration counts tiny under profiling, and
  prefer a short `--metrics` list (stall ratios, registers per thread,
  pipe/memory pct-of-peak) over `--set full` while iterating.
- A negative result ("this approach cannot go faster") needs the same rigor
  as a positive one: profiler evidence for the limiter and the measured
  numbers of each failed variant, recorded so the next agent does not
  re-explore them.

### Writing kernels (Mojo/GPU gotchas)

- GPU closures capture module-local `var`s BY REFERENCE = garbage on device.
  Everything a kernel reads must be a function parameter or `@__copy_capture`.
  This has produced flaky, mostly-wrong outputs that PASSED a standalone
  repro.
- Mojo shared-memory vector loads/stores default to align-4 and can reach PTX
  as scalar `ld.shared.b32`. Pass `alignment=16` explicitly on 16-byte
  accesses, then check the PTX actually vectorized.
- cp.async needs 16B-aligned SOURCES. Gate the 16B path on the runtime base
  pointer alignment, not just on `dim % 4 == 0` — offset views break the
  latter-only reasoning — and fall back to 4-byte variants, or it silently
  corrupts.
- Fully comptime-unrolled inner loops blow past the register budget and
  collapse occupancy. Use runtime loops plus launch-bounds metadata
  (`@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=...)`, `nvvm.minctasm`).
- `Atomic.fetch_add` defaults to sequential ordering (catastrophic on hot
  paths). For split-K, prefer a workspace plus a separate reduce kernel:
  deterministic, and measured several times faster than atomics on deep-K
  shapes.
- Deep-K, L2-resident shapes want 3-4 pipeline stages, not 2.
- Grid and split-K choices should fill whole waves: a partial extra wave can
  cost far more than it computes. Derive wave sizes from the runtime SM count
  and document, next to its definition, every constant that was fitted to one
  card (per the cross-compilation section above).

### Build & integration gotchas

- Define-gating silently no-ops standalone builds: the gates in
  `variant_gates.mojo` (`_op_on`, `_dtype_arg_on`, ...) read compiler defines
  and gate OFF when the define is absent. A standalone harness binary that
  calls a define-gated dispatcher (e.g. `_matmul_spec_launch`) without the
  matching `-D DTYPE_ARG_0=<dtype>` will decline or raise at runtime. Either
  pass the defines, or call the comptime-parameterized functions directly
  (e.g. `_gemm_enqueue[dtype, ...]`), which need no defines.
- Directory shadows module on the import path: with
  `-I torch_mojo_backend/eager_kernels`, a family directory such as
  `matmul_ops/` resolves as a package and shadows the `matmul_ops.mojo` file
  inside it. Import as `from matmul_ops.matmul_ops import ...`.
- Cache/source registration for new `.mojo` files: the extension loader
  hashes the import closure of each extension's entry `.mojo` file, so a new
  file imported (even transitively) from the entry module invalidates the
  compile cache automatically. Verify it anyway: touch the new file and
  confirm the source hash changes and a recompile happens. Only optional
  bridges enumerated in explicit `*_SOURCE_PATHS` lists in `aten_fast.py`
  need manual registration.
- `ctx.enqueue_function[f]` re-runs `compile_function` on every call (tens to
  hundreds of microseconds for large kernels). That is acceptable in a
  throwaway harness; production code must use the `_enqueue_cached` pattern
  from `op_utils`, like the rest of the eager kernels.
