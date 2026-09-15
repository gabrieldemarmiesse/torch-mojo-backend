# Validating the native backend on AMD (MI300A / ROCm)

This is a hands-off brief for an agent with an AMD cluster. Everything below
was verified on NVIDIA H100 (CUDA), on the MAX CPU device, and on an Apple M4
(Metal); nothing has run on AMD since the eager mode was rewritten as a native
PyTorch backend (branch `mojo-native-backend`, PR #458,
`docs/native_backend.md` is the design). The kernel families under
`torch_mojo_backend/eager_kernels/` are the ones the old path ran on MI300A;
what is new and unverified on AMD is everything above them: the C++ shim, the
Mojo runtime (`native/mojo/`), every op body (`ops_<group>.mojo`), the HIP
half of the vendor bindings, the process group over RCCL, the Triton HIP
driver, and the prebuilt libraries.

Work through the sections in order, time-box each, and write the report as
you go: a partial report with exact commands and numbers is the deliverable,
not a green suite. Never weaken a test to make it pass: a genuine AMD gap is
recorded as a skip or decline that names the platform and the reason, a bug
gets a fix in its own commit with a test, and everything else is reported.
Read `AGENTS.md` first (house rules: `uv run`, type hints, terse comments,
the eager-mode rules, `flock` around GPU work).

## 1. Setup

```bash
git clone -b mojo-native-backend https://github.com/gabrieldemarmiesse-ai-agent/torch-mojo-backend.git
cd torch-mojo-backend && uv sync          # once; afterwards always `uv run --no-sync`
uv run --no-sync python -c "import torch; print(torch.__version__)"   # must be a +cpu build
```

- **torch must be the CPU wheel.** A CUDA wheel makes every HIP kernel load
  walk 3 GB of NVIDIA libraries (first step 15 s instead of 1 s, measured on
  MI300A), and a ROCm wheel puts a second HIP runtime in the process. If
  `uv sync` pulled the CUDA build, `uv pip install torch --index-url
  https://download.pytorch.org/whl/cpu` into the venv. `register_mojo_devices()`
  warns when it sees either.
- Put the checkout, its `.venv` and the kernel cache
  (`TORCH_MOJO_BACKEND_CACHE_DIR`; it defaults to `~/.cache`) on the fast
  parallel filesystem (scratch), not on a slow home; kernel loads read every mapped
  library.
- `export MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1` on APUs: MAX's default
  allocator reserves ~115 GB of host RAM per process at first use.
- ROCm: MAX dlopens `libamdhip64.so` / `libhsa-runtime64.so` from `$ROCM_PATH`
  or `/opt/rocm` and RCCL (`librccl.so.1`) from the same place
  (`TORCH_MOJO_BACKEND_RCCL_LIB` overrides). `module load rocm` if needed.
  A `GLIBCXX_3.4.30 not found` means the system libstdc++ predates GCC 12: put
  a newer one on `LD_LIBRARY_PATH`.
- Compiles: the Mojo compiler's own cache is redirected to node-local scratch
  by the package (`compiler_env()`); keep `TMPDIR` local. Set
  `TORCH_MOJO_BACKEND_CACHE_DIR` to a scratch directory shared by your jobs so
  builds are done once. Never compile under the GPU lock.
- Every GPU-touching command: `flock /tmp/gpu_lock_0.lock <command>` on the
  node. SLURM exposes GPUs through `ROCR_VISIBLE_DEVICES`; one `torchrun` per
  node with `--nproc-per-node` = GPUs per node, the package slices the
  variable per rank.
- `TORCH_MOJO_BACKEND_TRACE=1` prints every build with its `-D` set and its
  time; keep it on while validating.

**First contact** (record every line of output in the report):

```bash
uv run --no-sync mojo --print-supported-accelerators | grep -i gfx
flock /tmp/gpu_lock_0.lock uv run --no-sync python -c "
import time, torch
from torch_mojo_backend import register_mojo_devices, get_accelerators
t = time.time(); register_mojo_devices(); print('registered in %.1fs' % (time.time() - t))
print(get_accelerators(), torch.mojo.device_count())            # api must be hip
x = torch.ones(3, device='mojo:0') * 2; print(x.cpu().tolist())  # first op: builds an extension + a kernel
a = torch.randn(256, 256); b = torch.randn(256, 256)
print((torch.mm(a.to('mojo:0'), b.to('mojo:0')).cpu() - a @ b).abs().max().item())
s = torch.mojo.Stream(); e = torch.mojo.Event(enable_timing=True)
with torch.mojo.stream(s): y = (a.to('mojo:0') * 3).sum(); e.record()
e.synchronize(); print('stream/event ok', y.item())
print(torch.mojo.stream_native_handle(torch.mojo.current_stream()))   # the raw hipStream_t
"
```

Expected: `api=hip` devices, the shim (~4 s) and the Mojo base library (~5 s)
built, the op and matmul correct, a non-zero stream handle. The HIP vendor
bindings (`native/mojo/vendor.mojo`: `hipEvent*`, `hipStreamWaitEvent`,
`hipStreamQuery`, `AsyncRT_DeviceStream_hip_stream` resolved by name at run
time) have never been exercised; a failure here is finding number one.

Then warm the cache once, outside the lock (~15 to 25 min on H100, report the
AMD time):

```bash
uv run --no-sync python -c "from torch_mojo_backend import register_mojo_devices, native; register_mojo_devices(); native.prebuild_ops()"
```

## 2. Runtime and op groups (`tests/native/`)

Run serially (`-p no:cacheprovider`, no `-n`), under the lock, one file per
command so a hang is attributable. Reference counts are the H100's.

| file | what it checks | H100 |
|---|---|---|
| `test_bringup.py` | device, fills, strided copies, casts, streams/events, autograd formulas | 32 pass |
| `test_loader.py`, `test_register_retry.py`, `test_prebuilt.py` | on-demand builds, caching, failed registration is retryable, prebuilt selection | all pass |
| `test_stream_ordering.py` | side streams, events, `torch.accelerator.synchronize`, device futures | 4 pass |
| `test_profiler.py` | legacy profiler device time per op, torch.profiler trace | pass |
| `test_binary.py test_unary.py test_compare.py test_reductions.py test_data_movement.py test_factories.py test_foreach.py test_nn.py test_matmul.py test_composed.py` | the op groups against CPU torch | 1811 pass, 0 fail (ten files) |
| `test_attention.py` | flash attention forward/backward, SDPA dispatch with autograd and autocast | pass |

Kernels with an explicit gfx942 route (the ones most likely to have
drifted while untested) live in `ops_attention.mojo`, `ops_matmul.mojo`,
`flash_attention_ops/`, `nn_ops`, `reduction_ops`, `data_movement_ops`,
`activation_forward_ops`; the lifetime-keepalive sweep touched
`ops_attention.mojo`'s gfx942 path without an AMD run. Give `test_attention.py`
and `test_matmul.py` a careful read of every failure.

`tests/native/conftest.py` has `skip_if_metal` and `side_stream_or_skip`
for Metal's gaps; if AMD has a gap of its own, add the same kind of helper
(named for the platform, with the reason) rather than a bare skip.

## 3. The whole suite and the conformance tables

```bash
flock /tmp/gpu_lock_0.lock uv run --no-sync pytest tests -q -p no:cacheprovider --ignore=tests/multinode   # ~50 min serial on H100
```

Reference (H100, tree 8e964c4): 3587 passed, 287 skipped, 127 xfailed, 11
failed (all eleven since fixed). Report the AMD counts and every failing node
with one line of its error.

Then the OpInfo conformance suite. Its tables (`conformance/known_unsupported.py`)
are measured, not hand-written: the base tables are the H100's (`sm_90a`) and
each other accelerator carries a delta. AMD has none yet, so many nodes will
fail with "declared unsupported ... and it now PASSES" or the reverse; that
is expected and is exactly what the regeneration fixes:

```bash
flock /tmp/gpu_lock_0.lock uv run --no-sync python conformance/regenerate_known_unsupported.py --records $SCRATCH/records_amd -n 8 > regen_amd.log 2>&1
uv run --no-sync python conformance/write_accelerator_delta.py regen_amd.log          # dry run: how many operators differ
uv run --no-sync python conformance/write_accelerator_delta.py regen_amd.log --write   # _ACCELERATOR_DELTAS[<accelerator_key()>], "gfx942" on MI300A
uv run --no-sync ruff format conformance/known_unsupported.py
flock /tmp/gpu_lock_0.lock uv run --no-sync pytest conformance/test_opinfo.py -q -n 8 -p no:cacheprovider   # must be 0 failed
```

The regeneration refuses `--write` off the base accelerator on purpose;
`write_accelerator_delta.py` records only the per-operator differences. Read
the `+ declare` list in `regen_amd.log` before committing it: an operator that
passes on H100 and CPU but not on AMD is a bug report first and a table entry
second. A node that fails in a way the regeneration does not classify as
"absent" (a numerical mismatch, a wrong error message) stays a failure and
needs a fix or an explicit decline.

## 4. Distributed (RCCL, then mojoccl)

Single node, `N` = GPUs on the node (4 on an MI300A node):

```bash
flock /tmp/gpu_lock_0.lock uv run --no-sync pytest tests/test_distributed.py -q -p no:cacheprovider          # H100 x2: 39 passed, 1 skipped
flock /tmp/gpu_lock_0.lock uv run --no-sync torchrun --standalone --nproc-per-node=$N demo_scripts/nanogpt_ddp.py --device mojo --nanogpt-path <nanoGPT checkout> --data-dir <shakespeare> --batch-size 12 --block-size 1024 --max-iters 40 --log-interval 1 --eval-interval 0 --seed 1337
TORCH_MOJO_BACKEND_CCL=mojo  ...same two commands...    # the in-repo Mojo collectives over the NCCL C ABI
```

Expect `NCCL_DEBUG=INFO` to name RCCL, the trace to say which library the
process group resolved, and the loss curve to match the single-GPU one at
the same seed. `batch_isend_irecv`, coalescing and `all_gather_single` /
`reduce_scatter_single` are covered by the test file. The multi-node
protocol used on Adastra is `tests/multinode/e2e_three_stacks_adastra.sh`
(2 x 4 MI300A, Slingshot, `module load aws-ofi-rccl`); its paths are site
variables at the top. Run it if two nodes are available; its "stock" leg
needs a separate venv with the ROCm torch wheel (`torch-rocm` in the
script), the other two legs use this venv.

## 5. Triton on HIP

The Triton driver for the mojo device has a HIP variant that has never run
(`triton_driver.py`, `_hip_driver_class`). It needs only the `triton` wheel
(both GPU backends are inside it) next to the CPU torch:

```bash
uv pip install triton          # into the venv; no ROCm torch
flock /tmp/gpu_lock_0.lock uv run --no-sync pytest tests/native/test_triton.py -q -p no:cacheprovider     # H100: 8 passed
```

Then a package written against `torch.cuda`: `uv pip install liger-kernel`,
and on mojo tensors run `LigerRMSNormFunction.apply(h, w, 1e-6, 0.0,
"llama", True)` forward and backward against the CPU formula, plus
`triton.testing.do_bench` on it. On NVIDIA the package's own
`device.type == "cuda"` branches fell through to its generic paths and it
worked unchanged; report what happens on AMD. Things to check by hand: the
target Triton picks (`gfx942` via the HIP driver's `get_device_properties`),
that the launch's stream is the mojo current stream's `hipStream_t`, and a
launch under `with torch.mojo.device(1):` on a second GPU.

## 6. TorchInductor (expected to need work)

```bash
flock /tmp/gpu_lock_0.lock uv run --no-sync pytest tests/native/test_inductor.py -q -p no:cacheprovider     # H100: 8 passed
```

`torch_mojo_backend/inductor.py` registers the mojo Triton target as a
subclass of Triton's `CUDABackend` (`monkeypatching.py`,
`register_the_mojo_triton_target`) and points Triton at MAX's `ptxas`
(`_ptxas.py`); both are NVIDIA-specific and the docs say so. Report what
breaks; fix only if it is under an hour (an AMD target alias of
`triton.backends.amd.compiler.HIPBackend` keyed by the accelerator api is the
obvious shape), otherwise leave it documented as NVIDIA-only.

## 7. Compiled HIP extensions and `cuda_interop` (low priority)

`torch_mojo_backend/cuda_interop.py` runs a package's compiled kernels on
mojo tensors by aliasing memory through DLPack and putting `torch.cuda` on
the mojo stream. Its ROCm side is written from sources only: DLPack retags
to `kDLROCM`, `torch.cuda.ExternalStream` wraps a `hipStream_t`. It needs a
ROCm torch wheel, which means two HIP runtimes in one process (MAX's from
`/opt/rocm`, torch's bundled one), a combination the distributed backend
refuses. In a **separate** venv with the ROCm torch wheel:

```bash
PYTHONPATH=<checkout> flock /tmp/gpu_lock_0.lock <rocm-venv>/bin/python -m pytest tests/native/test_cuda_interop.py -q -p no:cacheprovider     # H100 CUDA venv: 28 passed
```

The two facts to establish, even if the suite fails early: whether a
pointer allocated by MAX is readable by a torch HIP kernel (`as_cuda(t)` then
`t_cuda * 2`), and whether MAX's device ordinals match `torch.cuda`'s.
causal-conv1d has a HIP branch in its `setup.py` (`--offload-arch`); building
it from source and running `causal_conv1d_fn` through `call_cuda` is optional.

## 8. Prebuilt libraries

The wheel ships the C++ shim (per torch series) and the Mojo base library
prebuilt; the base library is claimed accelerator-agnostic (it probes the
api at run time) and CPU-baseline (`x86-64-v3`). On the AMD box:

```bash
uv run --no-sync python scripts/build_prebuilt.py --report --out prebuilt-out      # see --help; builds shims for 2.7..2.14 CPU wheels and the base library
uv build && uv venv /tmp/wheelvenv && uv pip install --python /tmp/wheelvenv/bin/python dist/*.whl "torch==2.11.*" --index-url https://download.pytorch.org/whl/cpu
/tmp/wheelvenv/bin/python scripts/smoke_prebuilt_wheel.py      # hides every C++ compiler; must say "using prebuilt" twice, then run an op on the CPU device
```

Then, under the lock, register from that venv on a GPU node and run one op
on `mojo:0`: the prebuilt base library, built with no accelerator in sight,
must drive the MI300A. If a library from the CI artifact (built on GitHub's
runners) is available, install that wheel instead of building: it is the
stronger test. `TORCH_MOJO_BACKEND_PREBUILT=0` must fall back to compiling.

## 9. Performance

Stock reference: a separate venv with the ROCm torch wheel, `--device cuda`.
Our leg: this venv, `--device mojo`. Same node, same GPU, ABBA order
(stock, mojo, mojo, stock), clocks locked if the site allows (`rocm-smi
--setperfdeterminism` or the site's equivalent; record what you used).

```bash
# single GPU, tok/s over steps 20..60, batch 1 4 12 48
python demo_scripts/nanogpt_ddp.py --device <cuda|mojo> --nanogpt-path ... --data-dir ... --batch-size <B> --block-size 1024 --max-iters 60 --log-interval 1 --eval-interval 0 --seed 1337   # via torchrun --standalone --nproc-per-node=1
# DDP on the node's GPUs, batch 48 and 12 per rank, five interleaved rounds
```

H100 reference: native/cuda = 0.90, 0.94, 0.96, 1.02 tok/s at batch 1, 4,
12, 48; DDP 16 ranks 1.006 (RCCL) and 1.012 (mojoccl) at batch 48. Report the
AMD ratios the same way, with GPU model, ROCm version, clocks and the two
torch builds. `benchmarks/` (per-op device-time ratios against stock torch,
`benchmarks/baselines.html`) needs stock torch in the same process; try
`uv run pytest benchmarks/ --update-baselines` from the ROCm-torch venv with
`PYTHONPATH` at the checkout and report whether the two HIP runtimes
coexist; if not, say so and stop there. `benchmarks/test_coverage.py` needs
no GPU and must pass.

## 10. Report

One markdown file, committed on a branch `amd-validation` pushed to the same
fork, plus a comment on PR #458 pointing at it. Sections in this order:

1. Environment: node, GPU model, ROCm version, `mojo --version`, torch
   version and build, MAX version, the `MODULAR_*` and
   `TORCH_MOJO_BACKEND_*` variables in effect.
2. A table per section above: command, wall time, passed/failed/skipped, and
   for each failure one line naming the test, the error and your
   classification (MAX/HIP limitation, our bug, test assumption, not run).
3. The conformance delta commit and the `+ declare` list with the reason for
   each operator.
4. Performance tables with the ABBA raw numbers, not only the ratios.
5. Fixes made, each as its own commit with a test, and the H100-side check
   you did not do (say so plainly; a reviewer with the other card will).
6. What you did not run and why.

Time budget: setup and first contact 1 h, sections 2 and 3 half a day
(compiles dominate: the first call of every op builds an extension, ~6 s
each on H100), section 4 two hours, 5 and 6 two hours, 7 and 8 optional,
9 two hours. Do not stop on a failure you cannot fix in an hour: record it,
skip that section's dependents, continue.
