# Native backend on AMD MI300A (ROCm 6.4.3): validation report

Run of `agents_docs/amd_validation_plan.md` on the Adastra cluster (CINES),
2026-09-13, branch `mojo-native-backend` at 588f776. Everything below was run
by an agent on one exclusive MI300A node; the sections follow the plan.

## At a glance

| section | outcome |
|---|---|
| 1 setup / first contact | works: shim 6 s, base library 13 s, api = hip, 4 GPUs; every HIP vendor binding worked at first try; the process exit segfault with the VMM knob is Modular's known bug |
| 2 runtime + op groups (`tests/native/`) | about 2170 passed across the 17 files after the fixes (per-file table in section 2); the only real failures were the three findings below; the fp32 GEMM bug was found by the very first `torch.mm` |
| 3 whole suite | 3741 passed, 342 skipped, 123 xfailed, 36 xpassed, **2 failed** (RCCL 2-rank workers inside the suite: the documented APU test-order pitfall; the same file passes 40 / 40 standalone). Fifth launch: two were OOM-killed by my own agents' memory use, one was scancel'ed by an agent |
| 3 conformance | no AMD delta needed except two operators that decline empty tensors; 13 one-ulp nodes anchored to float64 per accelerator; **2 failed** (`pow`, `__rpow__` float32: a real 14-ulp precision gap). Both closed afterwards on the H100 side, see "Reviewer follow-up" |
| 4 distributed | RCCL 40 / 40, mojoccl 7 / 7, 4-rank nanoGPT on both, losses match |
| 5 Triton on HIP | works after one fix (second-device launch); liger-kernel RMSNorm correct and benchmarked through the HIP driver |
| 6 Inductor | made to work on HIP within the hour: 8 / 8 |
| 7 cuda_interop (ROCm torch wheel) | the two HIP runtimes coexist; 27 passed, 1 skipped; ordinals match; `benchmarks/` could not complete (compile-bound) |
| 8 prebuilt libraries | shims for torch 2.7 to 2.14 and the base library build; the wheel's prebuilt base library drives the MI300A; the compile fallback works |
| 9 performance | ours / stock ROCm torch: 1.05 to 1.12 on one GPU, 1.05 (RCCL) and 1.03 (mojoccl) on 4-rank DDP at batch 48, 1.10 / 1.07 at batch 12 |
| fixes | 5 bugs fixed + Inductor HIP support + conformance instrument, each its own commit with a test; sm_90a assembly unchanged for the two kernel fixes; nothing run on an H100 |
| open | run-to-run nondeterminism of our training losses (stock is bit-reproducible here); the mojoccl batch-48 bimodality |

## 1. Environment

| item | value |
|---|---|
| node | Adastra `a1018` (SLURM job 5408061, `--constraint=MI300 --exclusive`), 4x AMD Instinct MI300A (gfx942, APU: 128 GB HBM per device shared with the host), 501 GB host-visible RAM |
| kernel / driver | Linux 5.14.0-570.120.1.el9_6, amdgpu 6.12.12 |
| ROCm | 6.4.3 (`/opt/rocm`, hipconfig 6.4.43484), RCCL 2.22.3 from `/opt/rocm/lib/librccl.so.1` |
| mojo | Mojo 1.0.0 (ed45d567), `mojo-compiler` 1.0.0 |
| MAX | max / max-core / max-mojo-libs 26.5.0 |
| torch | 2.11.0+cpu (the CPU wheel; `uv sync` had resolved `2.11.0+cu130` and was corrected with `uv pip install --reinstall-package torch torch==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu`; a bare `torch==2.11.*` is a no-op because the CUDA build satisfies it) |
| triton | 3.6.0 (pulled by the CUDA wheel, kept for section 5) |
| filesystems | checkout, `.venv`, uv cache and `TORCH_MOJO_BACKEND_CACHE_DIR` on `/lus/scratch` (the `/lus/work` and login-node `/tmp` uv cache were unusable: see notes) |
| env | `ROCM_PATH=/opt/rocm`, `LD_LIBRARY_PATH=/opt/cray/pe/gcc-libs:/opt/rocm/lib:...` (GLIBCXX_3.4.30), `ROCR_VISIBLE_DEVICES=0,1,2,3` (set by SLURM), `TORCH_MOJO_BACKEND_CACHE_DIR=$SCRATCH/native-amd/cache`, `TORCH_MOJO_BACKEND_TRACE=1`, `TMPDIR=/tmp` (node-local) |
| `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM` | **unset for every single-process run** and `=1` only for the multi-rank DDP runs. With it set, every process segfaults at exit (exit code 139, HSA runtime atexit; MAX 26.5 + ROCm 6.4.3, known from the RCCL work, `demo_scripts/nanogpt_ddp.py` ends in `os._exit(0)` for that reason), which would turn every subprocess-based test into a failure. Without it one process reserves ~115 GB of the APU's memory at first use, which is fine for one process on a 501 GB node. |
| clocks | not locked: `rocm-smi --setperfdeterminism` needs root on this site; sclk level 1 (1506 MHz) / mclk 1300 MHz at rest |

Setup notes:
- `uv sync` on the login node failed twice out of the default uv cache
  (`/tmp/<user>/.cache/uv`, 30 GB): `ModuleNotFoundError: hatchling.build`
  then `trove_classifiers has no attribute classifiers`, both truncated
  archives. A fresh `UV_CACHE_DIR` on scratch fixed it.

## First contact

`mojo --print-supported-accelerators` lists `amdgpu:gfx942` (and the other
gfx targets). The first-contact script of the plan, under the lock:

| step | result |
|---|---|
| `register_mojo_devices()` | C++ shim built in 6.20 s, Mojo base library in 12.72 s, "native mojo backend ready in 15.29s (5 devices)", registered in 15.6 s |
| `get_accelerators()` | 4 `Device(type=gpu)` + the CPU device; accelerator identity `hip:gpu,hip:gpu,hip:gpu,hip:gpu,cpu:cpu` (api = hip) |
| `torch.ones(3) * 2` | `[2.0, 2.0, 2.0]`; first op 31.3 s (five extension builds: `empty.memory_format` 4.0 s, `fill_.Scalar` 6.5 s, `mul.Tensor` 7.6 s, `elementwise_ops MulScalarSpec` 5.6 s, `_to_copy` 7.4 s) |
| fp32 `torch.mm` 256x256 | **max abs error 21.7 against CPU: wrong** (see finding 1); `MatmulSpec` fp32 spec compiled in 344 s |
| stream / event | `stream/event ok`, `sum` correct (`SumSpec` 7.3 s, `_local_scalar_dense` 4.1 s) |
| `stream_native_handle` | 207391536 (non-zero `hipStream_t`) |
| process exit | exit code 139: the segfault in the HSA runtime's atexit handler that `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1` causes (pre-existing, Modular's; the variable was set for this run only) |

Total 7 min 26 s wall, of which 344 s the one fp32 GEMM specialization.
The HIP vendor bindings (`hipEvent*`, `hipStreamWaitEvent`, ...) resolved and
worked at first try: `test_bringup.py` and `test_stream_ordering.py` below
pass.

Cache warm (`native.prebuild_ops()`, outside the lock): **29 min 39 s** wall
on this node, 239 extensions built in that process (1771 s of compile;
concurrent test runs built the rest); each op extension is 4 to 8 s, the
kernel specializations built on first call are the expensive part (GEMM:
fp32 344 s, bf16 485 s, fp16 120 s; elementwise / reduction 5 to 8 s).

### Finding 1: fp32 GEMM is wrong on gfx942 (pre-existing)

`torch.mm` with float32 operands returns wrong values for m, n >= 64;
bfloat16 and float16 are correct, tiny fp32 shapes are correct:

| dtype | 256x256x256 | 64x64x64 | 8x8x8 | 3x5x7 | 128x1024x512 |
|---|---|---|---|---|---|
| float32 (max abs err / ref max) | 24 / 66 | 9.8 / 34 | 4.8e-7 | 1.2e-7 | 45.9 / 142 |
| bfloat16 | 0.235 / 68 | 0.062 / 28 | 0.021 | 0.015 | 0.355 / 133 |
| float16 | 0.029 / 85 | 0.0078 / 30 | 0.0018 | 0.0009 | 0.034 / 150 |

The old eager path (upstream main ba926d9, its own warmed checkout on this
node) gives the same fp32 numbers (24 / 9.8 / ok / 42.9), so this is a bug in
the shared kernel family `eager_kernels/matmul_ops/` (gfx942 fp32 route of
`_amd_dynamic_mfma_dispatch`), not in the native backend, and it predates
the branch: the aten-level test suites had never been run on MI300A (only the
distributed tests and bf16 nanoGPT had). Root cause and fix: see "Fixes".

## 2. Runtime and op groups (`tests/native/`)

One `pytest` per file, serial, under the lock, no VMM knob. Wall time
includes waiting for the lock behind other steps and every first-call
compile; the pytest time is the suite's own.

| file | AMD result | pytest time | H100 (plan) |
|---|---|---|---|
| `test_bringup.py` | 28 passed | 155 s | 32 pass (28 collected on this tree) |
| `test_loader.py` | 7 passed | 283 s | pass |
| `test_register_retry.py` | 1 passed | 7 s | pass |
| `test_prebuilt.py` | 18 passed | 2 s | pass |
| `test_stream_ordering.py` | 4 passed | 15 s | 4 pass |
| `test_profiler.py` | 2 passed | 10 s | pass |
| `test_binary.py` | 183 passed | 495 s | |
| `test_unary.py` | 241 passed, 2 skipped (GELU bit patterns recorded on H100), 2 xfailed | 308 s | |
| `test_compare.py` | 203 passed | 2253 s | |
| `test_reductions.py` | **15 failed**, 432 passed, 4 skipped (before the cumsum fix; 40/40 cumsum cases pass after it, see finding 2) | 441 s | |
| `test_data_movement.py` | 360 passed | 467 s | |
| `test_factories.py` | 97 passed, 2 skipped (CPU-device-only case; no native GPU reference on this accelerator) | 75 s | |
| `test_foreach.py` | 30 passed, 3 xfailed, 4 xpassed (stale non-strict xfails about `linalg_vector_norm` not being registered yet) | 107 s | |
| `test_nn.py` | 346 passed, 2 skipped (rank > 4 batch norm is accelerator-only, CPU-device case), 1 xfailed | 374 s | |
| `test_composed.py` | 25 passed | 59 s | |
| `test_matmul.py` | 153 passed, 9 skipped (pure-Mojo tensor-core fast paths require an H100); with the fp32 fix. A first attempt was killed by my 90 min cap at 64 of 162 tests: serial first-call compiles of the GEMM specializations (15 built, 2 to 8 min each). Warming the specializations with `pytest -n 12` outside the lock took 11 min for matmul + attention + composed together, then the file ran under the lock in 21 min | 1280 s | |
| `test_attention.py` | **3 failed**, 20 passed, 10 skipped ("the FA4 kernels are compiled for sm_90a"). The three failures are H100 assumptions, see finding 3 | 532 s | pass |

### Finding 2: cumsum bf16/f16 and outer-dim routes were declined on HIP

All 15 `test_reductions.py` failures were `NotImplementedError: cumsum ...`:
`op_cumsum` (`native/mojo/ops_reductions.mojo`) declined bfloat16 / float16
and the dim-0-of-rank-2 route on every device whose api is not `cuda`, with
the comment "only ever MEASURED on NVIDIA". Measured on MI300A: every
declined case is correct (40 of 40 cumsum tests, integer cumsum bit-exact,
float errors at accumulation level), because HIP runs the portable
one-thread-per-line kernels of `nn_ops.mojo` (the NVIDIA `block.prefix_sum`
fast path is gated separately, inside the kernels, on `ctx.api() == "cuda"`
and is untouched). The gate is widened to `cuda or hip`; Metal and the CPU
device keep the old surface. Commit 3f609ed.

## 8. Prebuilt libraries

Compute nodes have no internet here, so the shim builds (which create a
throwaway venv per torch series) ran on the login node; everything else on
the node.

| step | result |
|---|---|
| `scripts/build_prebuilt.py --torch 2.7 ... 2.14` | shims for torch 2.7, 2.8, 2.9, 2.10, 2.11, 2.12, 2.13, 2.14 (268 to 284 KiB each, glibc >= 2.32, cxx11abi1) and the base library `libtmb_backend-max26.5.0-linux-x86_64.so` (333 KiB, glibc >= 2.34); ~15 min including the wheel downloads. `--report` lists all nine. Note: `--report` alone on an empty out dir raises "no manifest entries" (it never builds). |
| `uv build` | `torch_mojo_backend-0.3.1-py3-none-any.whl` 1.93 MB (13.0 MB / 161 files uncompressed, of which 2.6 MB prebuilt libraries) |
| wheel venv | `uv pip install dist/*.whl "torch==2.11.0+cpu" --extra-index-url https://download.pytorch.org/whl/cpu`. The plan's `--index-url` form fails: it hides PyPI and `max==26.5` is not on the torch index. |
| `scripts/smoke_prebuilt_wheel.py` | "using prebuilt C++ shim for torch 2.11" and "using prebuilt Mojo base library for MAX 26.5.0", backend ready in 4.12 s, `ones(3) * 2` on the CPU device OK, rc 0 |
| register from the wheel venv on the GPU node | prebuilt shim used, ready in 2.23 s, `(arange(6)*2+1).sum()` = 36 and a 2x3 fp32 `mm` exact on `mojo:0`: the base library built with no accelerator in sight drives the MI300A |
| `TORCH_MOJO_BACKEND_PREBUILT=0` | ready in 15.49 s (both compiled), op correct: the fallback works |
| `benchmarks/test_coverage.py` | 2 passed (no GPU) |

The CI-artifact wheel was not available on this box, so the wheel tested is
the one built here.

### Finding 3: the attention tests assume the H100's FA4 gates

`test_flash_attention_declines_float32`, `test_fused_sdp_choice_efficient_for_inference`
and `test_fused_sdp_choice_math_when_grad_is_needed` encode FA4's limits (no
float32, backward only at a full seqlen). The gfx942 fused flash kernels
(`flash_attention_ops/`, `_fused_fa_plan` in `ops_attention.mojo`) are
instantiated for float32, bfloat16 and float16 with forward and backward, so
on this device `_fused_sdp_choice` legitimately answers `flash` where the
H100 answers `efficient` or `math`, and the fp32 flash call runs instead of
raising. See "Fixes" for the numerical check of the fp32 route and the
arch-aware test expectations.

## 4. Distributed (RCCL, then mojoccl)

Second node `a1020` (job 5408315), 4 MI300A, `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1`
for the torchrun legs only, `NCCL_DEBUG=INFO`.

| command | result |
|---|---|
| `pytest tests/test_distributed.py` (RCCL, 2-rank torchrun workers) | **40 passed** in 799 s (H100 x2: 39 passed, 1 skipped) |
| `torchrun --nproc-per-node=4 nanogpt_ddp.py --device mojo` batch 12, 40 steps, RCCL | trace `collectives via /lus/home/softs/rocm/6.4.3/lib/librccl.so.1 (rccl version 22203)`; first step 1133 s (48 kernel builds, 1131 s of compile, cold cache for the 4-rank shapes), then 705 to 720k tok/s; loss 11.0074 -> 6.0455 at step 40 |
| same, 1 rank, batch 12 | 190k tok/s, loss 6.0858 at step 40 (a different global batch, so a different curve by construction; the stock-vs-mojo same-configuration comparison is in section 9) |
| `pytest tests/test_distributed.py -k mojo` (`TORCH_MOJO_BACKEND_CCL=mojo`) | 7 passed in 106 s |
| 4-rank nanoGPT, mojoccl | trace `collectives via .../cache/libmojoccl.hash-68bf71f23b7ee946.so (mojoccl version 23102)`; 695 to 706k tok/s; losses equal to the RCCL run to 1e-4 through step 20 (6.2451 vs 6.2450), 6.0595 vs 6.0455 at step 40 |

The multi-node protocol (`tests/multinode/e2e_three_stacks_adastra.sh`) was
not run: the two allocations were single nodes used for the sequential and
the parallel halves of this validation, not a node pair.

## 5. Triton on HIP

`triton` 3.6.0 (the wheel the CUDA torch had pulled) next to the CPU torch,
`liger-kernel` 0.8.2.

| command | result |
|---|---|
| `pytest tests/native/test_triton.py` | **2 failed**, 5 passed, 1 skipped (CUDA driver contexts). Failures: (1) `test_triton_launch_on_a_second_device`: `Triton Error [HIP]: Code: 101, invalid device ordinal` from the HIP driver's launch under `with device_module.device(1)`: a real bug in the never-run `_hip_driver_class`, see "Fixes"; (2) `test_driver_is_installed_on_import_after_registration` asserts the class name `MojoCudaDriver`; on HIP it is `MojoHipDriver`: test assumption |
| liger `LigerRMSNormFunction.apply(h, w, 1e-6, 0.0, "llama", True)` on `mojo:0`, (64, 2048) | forward max err 1.4e-6, dX 7.2e-7, dW 7.6e-6 against the CPU formula (ref max 9.9 / 4.0 / 30.7); `triton.testing.do_bench` 0.0177 ms; `driver.active` is `MojoHipDriver`, target `GPUTarget(backend='hip', arch='gfx942', warp_size=64)`; the launch stream handle 189223616 equals `torch.mojo.stream_native_handle(current_stream())`. The package's `device.type == "cuda"` branches fell through to its generic paths unchanged, as on NVIDIA |
| same under `with torch.mojo.device(1)` on `mojo:1` | the second-device bug above; after the fix: forward 1.4e-6, dX 7.2e-7, dW 7.6e-6, do_bench 0.059 ms, launch stream == mojo `hipStream_t` |
| after the fix (commit 5a7eb6a) | `test_triton.py`: 7 passed, 1 skipped (CUDA driver contexts) |

## 6. TorchInductor

`pytest tests/native/test_inductor.py`: **7 failed**, 1 passed. Every failure
is `OSError: libcuda.so.1: cannot open shared object file` from
`inductor.py::_device_properties`, reached through Inductor's
`DeviceProperties.create`: the device interface reads compute capability and
SM count from libcuda, the mojo Triton target is an alias of
`CUDABackend`, and `_ptxas.apply_triton_default()` is NVIDIA-only, as the
docs say. Fixed within the hour (commit b3c9e27): device properties come from the HIP Triton driver's own `get_device_properties` on hip (no libcuda), the mojo Triton target aliases `triton.backends.amd.compiler.HIPBackend` when the accelerator api is hip, ptxas is skipped on hip, and `raise_if_triton_unavailable` looks for the `amd` backend. After: **8 passed** (cold 349 s, warm 19 to 25 s); `test_monkeypatching_is_centralized.py` still passes; ruff and ty clean. The CUDA path is the same code behind an api check.

## 3. The whole suite and the conformance tables

The conformance regeneration (`regenerate_known_unsupported.py --records ... -n 4`,
under the lock, with `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1` so four
workers fit: its recorder writes one file per process and tolerates the
exit segfault) ran in 115 s once the specializations were warm:
14 failed, 1112 passed, 236 skipped, 1424 xfailed. `write_accelerator_delta.py`:
**0 operators differ from the base tables** for `test_matches_cpu` and 0 for
`test_errors_match`, so no `_ACCELERATOR_DELTAS["gfx942"]` entry is
needed and there is no `+ declare` list: every operator the H100 tables
declare unsupported is unsupported here too, and nothing declared
unsupported passes. (The plan expected many differences; the native backend
declines op by op in Mojo, not per architecture, which is why the tables
carry over.)

The 14 failures are numerical, one bf16/f16 ulp or an fp32 summation-order
difference against tolerances that the H100 happens to meet:

| node | mismatch |
|---|---|
| bmm float32 | 1 of 250 elements, abs 1.44e-5 (1e-5 allowed) |
| pow, __rpow__ float32 | rel 1.4e-6 to 1.6e-6 (1.3e-6 allowed) on 1 to 3 large elements |
| log_softmax, masked_log_softmax bf16 / f16 | 2 to 3 of 25 elements, abs 3.8e-3 (bf16) / 4.3e-4 (f16); the CPU reference is exactly 0 there |
| addr float16 | 1 of 50, rel 1.045e-3 (1e-3 allowed) |
| batch_norm, instance_norm bf16 / f16 | 1 to 7 of 125, one ulp |
| conv2d bf16 / f16 | 1 element, one ulp |

The suite's existing instrument for this class is `_FP64_ANCHORED` in
`conformance/test_opinfo.py` (as accurate as torch against a float64
reference, within 2x). Two things were done with it, both measured on GPU 3
of `a1018`:

- `assert_close_fp64_anchored` (`torch_mojo_backend/testing.py`) used to drop
  the whole tensor to the default bar when any element of the reference was
  non-finite (a masked `-inf`, a NaN from a negative base to a fractional
  power) and crashed on an empty reference; it now anchors the finite
  elements and compares the rest with the default bar (commit 99a675f; the
  helper's own tests still pass).
- the anchored set became accelerator-keyed (`_FP64_ANCHORED_BY_ACCELERATOR`,
  merged by `known_unsupported.accelerator_key()`, which returns `gfx942`
  here; the H100 set is untouched), listing 13 of the 14 nodes (commit
  7542add + 99a675f). With it: bmm, addr, batch_norm, instance_norm and
  conv2d pass the anchored bar.

Second regeneration after that (`regen_amd2.log`, 205 s): 6 failed, 1120
passed, 236 skipped, 1424 xfailed, and now **2 operators differ from the
base** for `test_matches_cpu`: `log_softmax` and `masked_log_softmax` in
bfloat16 / float16 raise `NotImplementedError: softmax of an empty tensor`
on the `(5, 0, 0)` sample (`ops_nn.mojo` declines every empty softmax; the
one-ulp failure on an earlier sample used to hide it). The base table
already declares `log_softmax` float32 for the same reason and the MAX CPU
device's delta declares exactly these two operators in all three dtypes,
so `write_accelerator_delta.py --write` records them under
`_ACCELERATOR_DELTAS["gfx942"]` (the accelerator key string this tree
produces; the plan's `amdgpu:gfx942` spelling is what the docstring says,
`accelerator_key()` returns the bare architecture name). Whether the H100
also raises on that sample for bf16/f16 is not known from here.

What stays failing after the delta: **`pow` float32 and `__rpow__` float32**,
a real precision gap (finding 5): the fp64-anchored bar cannot accept a
result 14 ulp from the float64 answer when torch is at 0.5 ulp, and a table
entry only excuses an exception. 

| command | result |
|---|---|
| `regenerate_known_unsupported.py --records ... -n 4` (first, before the anchoring) | 14 failed, 1112 passed, 236 skipped, 1424 xfailed; 0 operators differ |
| same, after the anchoring and the helper fix | 6 failed, 1120 passed; 2 operators differ (`log_softmax`, `masked_log_softmax`) |
| `write_accelerator_delta.py regen_amd2.log --write` + `ruff format` | `_ACCELERATOR_DELTAS["gfx942"]` with those two operators, commit a6fe537 |
| `pytest conformance/test_opinfo.py -n 4` (VMM=1, GPU 3) | **2 failed** (`pow` float32, `__rpow__` float32), 1120 passed, 236 skipped, 1428 xfailed in 109 s |
| `pytest conformance/test_known_unsupported.py` | 7 passed |

The plan's "must be 0 failed" is not met by two nodes, both the `pow`
precision gap of finding 5.

The whole suite (`pytest tests --ignore=tests/multinode`, serial, one
process) was OOM-killed twice by the node's global OOM killer, at 24% and
at 335 s, while my agents held three other MAX contexts on the same node
(each reserves ~115 GB of the APU's 501 GB at first use; `free` does not
show those reservations). Its fourth launch, alone on the node, is reported
below.

Fifth launch (`a1020`, `ROCR_VISIBLE_DEVICES=1,2,3`, alone on the node
except the stuck benchmark process on GPU 0 until 05:50):

| command | wall | result |
|---|---|---|
| `pytest tests -q --ignore=tests/multinode` (serial, one process, with all the fixes in the tree) | 2 h 31 min (H100: ~50 min; the difference is first-call compiles of the specializations this run met first) | **2 failed**, 3741 passed, 342 skipped, 123 xfailed, 36 xpassed (H100 reference, tree 8e964c4: 3587 passed, 287 skipped, 127 xfailed, 11 failed) |

The two failures are `tests/test_distributed.py::test_two_rank_nccl[collectives-vendor]`
and `[stress-vendor]`: the RCCL worker dies with `ncclGroupEnd failed:
unhandled cuda error`. Both passed in the standalone run of the file
(section 4: 40 passed) and the same two modes pass with mojoccl inside the
whole suite; classification: **test-order interaction on the APU**, the
pitfall `test_distributed.py`'s own docstring describes (the pytest
process has by then created mojo tensors, so MAX holds its ~115 GB
reservation on the GPU rank 0 is pinned to, and RCCL's P2P setup on that
GPU fails). The file guards its own single-process probe against it by
using a subprocess, but cannot guard against the 3000 tests before it. Not
a backend bug; on an APU run `test_distributed.py` in its own process. The
36 xpassed are non-strict xfails that the branch has since fixed (the
`foreach` ones of section 2 among them).

## 9. Performance

Node `a1020` (4x MI300A), clocks not locked (no site permission). Stock:
the separate venv with `torch 2.9.1+rocm6.4` (the newest ROCm wheel that runs
at full speed on this 6.4.3 driver: the rocm7.1 wheel of torch 2.11 is 600x
slower here, measured on 2026-09-10), `--device cuda`, its bundled RCCL. Ours:
this venv (`torch 2.11.0+cpu`, MAX 26.5), `--device mojo`, system RCCL or
mojoccl, `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1`. nanoGPT 124M, bf16,
block 1024, 60 steps, seed 1337, the same `demo_scripts/nanogpt_ddp.py` on
both legs. Throughput = tokens / time over steps 20..60 (step time from the
per-step tok/s the script logs), one number per leg; the per-step warm-up of
step 1 is excluded. Every leg is a fresh process.

Single GPU (`ROCR_VISIBLE_DEVICES=0`), ABBA = stock, mojo, mojo, stock:

| batch | stock (A1, A2) | mojo (B1, B2) | mojo / stock |
|---|---|---|---|
| 1 | 54.6k, 55.0k | 59.5k, 59.7k | **1.09** |
| 4 | 118.6k, 119.7k | 128.1k, 128.2k | **1.075** |
| 12 | 170.1k, 169.4k | 189.8k, 189.3k | **1.12** |
| 48 | 213.0k, 212.6k | 223.9k, 224.2k | **1.05** |

H100 reference from the plan: 0.90, 0.94, 0.96, 1.02.

DDP, 4 ranks, three rounds of (stock, mojo+RCCL, mojoccl, mojoccl, mojo+RCCL, stock),
tok/s aggregate over the 4 ranks, all six legs listed in run order:

| per-rank batch | stock | mojo + RCCL | mojo + mojoccl | RCCL / stock | mojoccl / stock |
|---|---|---|---|---|---|
| 48 | 826.3k, 827.1k, 829.4k, 828.6k, 825.0k, 821.1k (mean 826.3k) | 870.8k, 866.7k, 870.5k, 866.1k, 869.4k, 869.5k (mean 868.8k) | 867.9k, 834.3k, 835.8k, 865.0k, 827.8k, 866.4k (mean 849.5k) | **1.051** | **1.028** |
| 12 | 655.4k, 652.0k, 653.8k, 654.1k, 653.9k, 656.1k (mean 654.2k) | 722.7k, 714.4k, 716.1k, 719.4k, 719.0k, 719.9k (mean 718.6k) | 699.4k, 682.1k, 702.6k, 703.5k, 701.4k, 701.4k (mean 698.4k) | **1.098** | **1.068** |

H100 reference: 16 ranks 1.006 (RCCL) and 1.012 (mojoccl) at batch 48.

Observations:
- mojoccl at batch 48 alternates between two levels (~866k and ~830k) from
  run to run; RCCL does not. Not investigated here.
- The stock leg's loss at step 60 is bit-identical across its six runs
  (5.6215 at batch 48); our legs vary from run to run at the 1e-2 level
  (5.57 to 5.69 at batch 48; RCCL and mojoccl alike), so some kernel on our
  side is run-to-run nondeterministic (a split-K or attention-backward
  reduction order is the usual suspect). Stock CUDA/ROCm torch is not
  deterministic in general either, but here it was. Not investigated.
- Every leg completed (52 of 52, rc 0). Warm start latency of our legs is 5
  to 20 s against 35 to 60 s for the stock leg (its ROCm wheel imports from
  Lustre); the compile-everything cold start of the first-ever run was ~19
  min (section 4).

`benchmarks/` against stock torch in the same process: see section 7.
`benchmarks/test_coverage.py`: 2 passed.

## 7. Compiled HIP extensions and `cuda_interop`

A separate venv: a copy of the stock `torch 2.9.1+rocm6.4` venv plus MAX
26.5 and the package's dependencies, with `PYTHONPATH` at the checkout
(the package pins `torch>=2.10`, which the shim honours for 2.9 since it
ships a 2.9 shim; nothing complained). Two HIP runtimes are in the
process (MAX's from `/opt/rocm`, torch's bundled one) and
`register_mojo_devices()` warns about it, as designed.

| check | result |
|---|---|
| registration with the ROCm wheel | OK: `torch.cuda.device_count()` 4, `torch.mojo.device_count()` 5 |
| a MAX-allocated pointer read by a torch HIP kernel | `as_cuda(t)` inside `on_mojo_stream(i)`, then `t_cuda * 2`: `[2, 4, ..., 16]` correct on every device |
| device ordinals | `mojo:i` aliases to `cuda:i` for i in 0..3, each a distinct MI300A uuid |
| `pytest tests/native/test_cuda_interop.py` | **27 passed, 1 skipped** (`causal_conv1d` not installed) in 302 s; H100 CUDA venv: 28 passed. The ROCm side written from sources (`kDLROCM` retag, `torch.cuda.ExternalStream` over a `hipStream_t`) works unchanged |
| `as_cuda` outside `on_mojo_stream` | raises the documented ordering error (my first probe tripped it) |
| `pytest benchmarks/ --update-baselines` from that venv | did not produce a measurement: the full run spent its 2 h cap compiling 24 specializations into that venv's own cache; `benchmarks/test_gemm.py` alone, with the cache seeded from the main one (same kernel-family hashes), sat on its first case `test_mm[S1_4096x4096x4096-NN-bf16]` for the whole 1 h cap without finishing it and without building anything new. Not diagnosed (the plan said to stop there); the two runtimes do coexist for the interop tests above, so the hang is specific to the benchmark harness (a torch.cuda event / MAX stream interaction is the first suspect). `benchmarks/baselines.html` therefore has no gfx942 entries; `benchmarks/test_coverage.py` passes |

causal-conv1d was not built from source (optional in the plan).

### Finding 4: bf16 / fp16 flash-attention backward wrong on partial tiles (gfx942)

Found by the fp32 probe written for finding 3, which also ran bf16:
`_scaled_dot_product_flash_attention(...)[0].sum().backward()` on
BHSD inputs against the float64 math SDPA on CPU:

| dtype | b h s d causal | out | dq | dk | dv | ref max |
|---|---|---|---|---|---|---|
| float32 | 2 4 200 64 F | 1.0e-6 | 1.2e-6 | 8.9e-7 | 8.4e-7 | 0.90 |
| bfloat16 | 2 4 200 64 F | 2.0e-3 | 3.4e-3 | **9.3e-2** | **6.2e-2** | 0.77 |
| bfloat16 | 1 2 8 8 F | 3.0e-3 | 4.8e-3 | **1.65** | **1.35** | 1.23 |
| bfloat16 | 1 2 128 64 T | 6.7e-3 | 8.2e-3 | 1.2e-2 | 1.4e-2 | 3.17 |

Root cause (`flash_attention_bwd_kernels.mojo`, `_bwd_dkv_mfma`,
`_tile[MASKED=True]`): the causal bound on the scores was applied whenever
the tile was the masked variant, but that variant is also the path for a
partial last query tile and for a sequence shorter than one 64-row tile,
regardless of `is_causal`, so every query-before-key term of the keys in
that tile was dropped from dK and dV for non-causal inputs. dQ and the
forward fold the causal flag into a per-query limit and were right; fp32
takes the baseline kernels (`is_half_float` gate) whose bound respects the
flag. One expression fixed it (the bound applies only when causal). Grid of
107 (shape, dtype, causal) cases: 24 wrong before, 0 after. Regression test
`test_fused_flash_backward_partial_tail_gfx942` (8 ids). sm_90a assembly: 0
of 34 kernels differ. The code-only Codex reader and the debugging agent
reached the same root cause independently. Commit 506e2c9.

## Fixes

Each fix is its own commit on `amd-validation` with a test. None was run on
an H100: the NVIDIA-side check is the cross-compiled sm_90a assembly compare
for the two kernel fixes, and an api gate for the three Python ones. A
reviewer with an H100 should run `tests/native/test_matmul.py`,
`test_reductions.py`, `test_attention.py`, `test_triton.py` and
`test_inductor.py` on it.

| commit | fix | test | H100-side check |
|---|---|---|---|
| 6598e2b | fp32 GEMM on gfx942: 32x32 warp tile for fp32 in the non-transposed and deep-K geometries; comptime guard `transpose_b or float32` against single-MMA warp tiles | `test_mm_float32_tensor_core_regime` (3 shapes), extended in ba372d6 to the deep-K geometry and `addmm` | `compare_kernel_asm.py --accelerator sm_90a`: 0 of 396 kernels differ; the route is under `comptime if gfx942` |
| 3f609ed | cumsum bf16/f16 and outer-dim routes enabled on HIP (the fast NVIDIA kernels stay gated on `ctx.api() == "cuda"` inside `nn_ops.mojo`) | 40 / 40 cumsum tests on MI300A | host-side gate only: `cuda` behaviour unchanged, Metal and CPU keep the old surface |
| b3c9e27 | Inductor on HIP: device properties from the HIP Triton driver, `HIPBackend` alias for the mojo target, no ptxas | `test_inductor.py` 8 / 8 | api-gated; CUDA path is the same code |
| 5a7eb6a | Triton HIP driver: `hipSetDevice` guard around `load_binary` (shared `_MojoUtils` wrapper; CUDA keeps its context push/pop) | `test_triton.py` 7 passed, 1 skipped; liger on `mojo:1` | CUDA path unchanged in behaviour (refactored wrapper) |
| 506e2c9 | flash-attention backward: causal bound only when causal in the half-float dK/dV tail tile | `test_fused_flash_backward_partial_tail_gfx942` (8 ids) | sm_90a: 0 of 34 kernels differ (the MFMA kernels are under `is_amd_gpu()`) |
| 2a93241 | review follow-ups: `GPUTarget` hint, checked `hipSetDevice` on restore, cumsum comment | ruff, ty | none needed |
| ba372d6 | `test_matmul.py`: deep-K fp32 and fused-bias cases | 10 / 10 | none needed |

A code-only review by Codex (`gpt-6-astra`) of the first four commits found
them mergeable and produced the follow-ups; its one substantive caveat is
performance: the fp32 non-transposed blocks now run 128 threads (deep-K: 64)
instead of 256, which is unmeasured: the benchmarks run of section 7 did
not complete, so the fp32 GEMM ratio against stock torch is still owed.

### Finding 5: float32 `pow` is 14 to 46 ulp off on gfx942 (not fixed)

Route: `aten::pow.Tensor_Tensor` -> `PowSpec` (`logic_ops.mojo`) and
`pow.Tensor_Scalar` -> `PowScalarSpec` (`elementwise_ops.mojo`), both
`std.math.pow` -> the Mojo stdlib's `SIMD._powf_scalar`: integral exponents
by repeated squaring, otherwise fp32 `exp(y * log(x))`. On AMD that is
`v_exp_f32` after a software (Cephes) fp32 log; half an ulp of the fp32
product `y * log(x)` is the same *relative* error in the result: 1.1e-6 at
`y * log(x) ~ 18.8`, i.e. 14 ulp, growing with the magnitude. NVIDIA's fp32
log takes a hardware log2 route in the same stdlib, which is why the H100
sits inside torch's rtol 1.3e-6 and the MI300A outside (and why the MAX CPU
device declares `pow` float32 in its delta). Measured with
`scripts/pow_probe.py` (1M elements, ulps against float64): max 46, mean
3.8, 2.1% of elements beyond 16 ulp; torch CPU max 0.585; integral
exponents fine (the untouched `_powi` path, 7 ulp at y = 105).

Two accurate replacements were written on branch `amd-pow-candidates`
(a fp64 `exp(y log x)` with an exact-coefficient log, and a lean variant
with `v_rcp_f64` + Newton and a `v_exp_f32` tail): both reach 0.5 to 0.84
ulp and pass the two conformance nodes, but on a 16M-element streamed
timing the tensor-scalar kernel (`PowScalarSpec`, whose body handles several
lanes per thread) costs +42 to 47% for a fractional scalar exponent and up
to +19% for `pow(x, 2.0)`, the RMSNorm / MSE case, because the per-lane fp64
path either serializes the lanes (out of line) or inflates the register
footprint (inlined). Not merged. The lead for a next attempt is to
vectorize the fp64 path across the lanes with SIMD float64 and a select
instead of a per-lane branch. `scripts/pow_probe.py` and
`scripts/pow_timing.py` are committed (5f18d69) so the next attempt starts
from the numbers.

Observation, not a fix: on the deep-K fp32 shapes (k = 2080, 4128) both the
gfx942 kernel and the CPU mojo device sit ~2e-4 from torch's blocked fp32
sum on a few elements, about 4x torch's own distance from float64: the
kernels accumulate in k order. Within fp32 expectations, but the
fp64-anchored bar of the conformance suite would not accept it.

## Reviewer follow-up (H100 side, 2026-09-14)

Done from the NVIDIA cluster on this branch before merging it:

- `scripts/compare_kernel_asm.py --accelerator sm_90a`: 0 of 290
  specializations differ between `mojo-native-backend` and this branch.
- H100, the files this branch touches (`test_matmul.py`, `test_reductions.py`,
  `test_attention.py`, `test_triton.py`, `test_inductor.py`): 662 passed,
  12 skipped; `test_nn.py`, `test_binary.py`, `test_unary.py`,
  `test_composed.py` after the follow-ups below: see the PR.
- The H100 conformance suite had 15 failures of its own on the same nodes
  this report anchored for gfx942 (never seen before: CPU-only CI, and the
  sm_90a base tables were regenerated without a re-run). Fixed here:
  fifteen `sm_90a` nodes anchored the same way; float32 `pow` and `__rpow__`
  now go through float64 in both kernels (`logic_ops` BOP_POW,
  `elementwise_ops` SOP_POW; not on Apple GPUs), which costs nothing
  measurable on the H100 (`scripts/pow_timing.py`: 71.0 us before and after
  on 16M elements, memory-bound) and should close finding 5 on MI300A too
  (unverified there); `softmax` of an empty tensor returns an empty tensor
  instead of declining, so the `log_softmax` / `masked_log_softmax` delta
  entries recorded above went with the decline (the gfx942 key stays, empty:
  a re-run there is owed). Softmax and pow entries regenerated on the H100
  and the MAX CPU device; both suites 0 failed.
- gpt-6-astra reviewed the diff: one P1 (the anchored comparison accepted a
  NaN where torch and the reference are finite, and let a non-finite torch
  result poison the error budget: fixed, with host-only tests) and four P2s
  (the two probe scripts ran GPU work on import, now behind `main()`; a
  failed `hipSetDevice` restore replaced the loader's exception, now a note
  on it; the wall-time timing is labelled as such; and this report carried
  two copies of sections 3 to 6 and a claim that section 7 measured fp32
  GEMM: consolidated).

## 10. Not run, and how this was run

Not run:
- The multi-node protocol (`tests/multinode/e2e_three_stacks_adastra.sh`):
  the two allocations were single nodes used in parallel (suites on one,
  distributed / perf on the other), never a pair.
- The CI-artifact wheel: not available on this box; the wheel tested in
  section 8 was built here.
- causal-conv1d from source (optional).
- `benchmarks/` beyond `test_gemm.py`: the full suite compiled for two hours
  into the ROCm venv's own cache without reaching a measurement (24
  specializations); only the GEMM file was rerun with a seeded cache, see
  section 7.
- The H100 side of every fix (section "Fixes").
- Clock locking (no permission).

How: one exclusive MI300A node (`a1018`, 10 h) for the serial suites, a
second one (`a1020`, 10 h) taken after 3.5 h for sections 4 to 9 in
parallel; every GPU command under a per-GPU `flock`; kernel specializations
warmed with `pytest -n 12` / `-n 16` outside the lock before the timed
runs (the serial first-call compiles were the dominant cost: GEMM
specializations 2 to 8 min each, ~40 of them). Five bugs were hunted with a
debugging agent and a code-only Codex reader in parallel; the reader's
diagnosis landed first and was verified by the probe for the Triton loader,
the flash-attention backward and the pow route. Two things went wrong with
that parallelism and are worth knowing: three agent processes plus the
suite exceeded the APU's memory twice (each MAX context reserves ~115 GB of
the 501 GB and `free` does not show it; the global OOM killer took the
suite), and one agent's `scancel` loop killed the fourth whole-suite launch
at 91% and a regeneration step. The `register_mojo_devices()` side effect on
the C environment (`PYTHONEXECUTABLE` set to the system interpreter without
`VIRTUAL_ENV`, `:` prepended to `PYTHONPATH`) made subprocess-based tests
fail with "No module named torch" when pytest was started with the bare venv
python instead of `uv run`; and `uv run python /path/script.py` resolves an
editable install to the checkout, not to a worktree on `PYTHONPATH`, unless
`PYTHONPATH` is set explicitly (checked with `native._KERNELS_DIR`).
