# GPT-2 XL FSDP2 throughput

## Current optimization snapshot

Commit `2194826`, measured on 2026-09-18 on an exclusive node
(`par2dc5-ai-prd-cl02s04dgx27`, Slurm job 257966, H100 80GB HBM3, driver
570.211.01). Same training configuration as below. Each row is the median of
six synchronized 10-step windows: three per launch, two launches per stack,
in CUDA/Mojo/MojoCCL then MojoCCL/Mojo/CUDA order, nothing else on the node,
every window retained. Raw records:
`current_bench_train/fsdp2_opt/fresh/runs/q2_{cuda,mojo,mojoccl}_{a,b}.json`.

### Two GPUs (sequence length 1024)

| Configuration | Tokens/s | vs CUDA | Window range (tokens/s) |
|---|---:|---:|---:|
| Stock PyTorch CUDA + NCCL | 9,062.8 | 100% | 8,289.2–9,093.4 |
| Torch Mojo + NCCL | 9,141.7 | 100.9% | 9,046.6–9,253.1 |
| Torch Mojo + MojoCCL | 9,186.1 | 101.4% | 8,198.6–9,267.1 |

### Eight GPUs (sequence length 1024, batch 1 per GPU)

Ten windows per stack (five per launch, two launches, palindromic order);
raw records `.../runs/r8_{cuda,mojo,mojoccl}_{a,b}.json`.

| Configuration | Tokens/s | vs CUDA | Window range (tokens/s) |
|---|---:|---:|---:|
| Stock PyTorch CUDA + NCCL | 35,582.6 | 100% | 35,357.9–36,043.0 |
| Torch Mojo + NCCL | 34,648.6 | 97.4% | 31,404.0–34,977.2 |
| Torch Mojo + MojoCCL | 34,399.4 | 96.7% | 31,208.4–34,592.8 |

Every mojo launch has one window ~10% below the others (the last one in
three launches, the first in one); CUDA launches do not. Unexplained; the
medians above include those windows.

### Two nodes, sixteen GPUs (sequence length 1024, batch 1 per GPU)

Nodes `par2dc5-ai-prd-cl02s03dgx28` + `dgx29` (Slurm job 259332, InfiniBand,
64 CPUs per node), commit `096979a0` (hierarchical multi-node reduce-scatter
with the streaming kernel, per-NIC all-gather with in-kernel RDMA release,
rank gate). 18 windows per stack: three palindromes of CUDA, NCCL, MojoCCL,
MojoCCL, NCCL, CUDA, three windows per launch; raw records under
`~/projects/tmp/fsdp2-2node/results/int4_xl/` (outside the repo).

| Configuration | Tokens/s | vs CUDA | Window range (tokens/s) |
|---|---:|---:|---:|
| Stock PyTorch CUDA + NCCL | 70,819.0 | 100% | 61,649.5–71,346.7 |
| Torch Mojo + NCCL | 68,692.2 | 97.0% | 60,166.0–69,621.3 |
| Torch Mojo + MojoCCL | 67,173.7 | 94.9% | 60,854.9–70,288.5 |

MojoCCL is at 97.8% of NCCL as the collective library. Two-node windows
spread ±7% on every stack (each stack has ~1 in 6 windows 10% below its
median), so single six-window runs on this node pair have read NCCL
anywhere from 95.4% to 100.5% and MojoCCL from 92.6% to 96.6%; only long
interleaved runs like this one are comparable, and 2–3% differences are at
the noise floor. The step before the hierarchical reduce-scatter was
14,608.6 tok/s (20.9%) with MojoCCL.

What closed the gap from the 84.8% snapshot below (both stacks are
host-bound: GPU compute-stream time is ~110 ms of a ~224 ms step, so every
change is host work per op or per collective):

- The process group returns an event-backed `Work` instead of building a
  device Future per collective (all_gather_into_tensor 105 → 71 µs in-model).
- The boxed-kernel adapter decides a schema's argument conversions once, reads
  a tensor's metadata in one call, and lets ATen's own device-agnostic kernels
  serve `view`/`_unsafe_view`/`_reshape_alias`/`as_strided`.
- The kernel launch cache is keyed by the kernel function's compile-time
  identity: a warm launch formats and allocates nothing; `KernelCall` uses
  inline storage.
- Functional elementwise ops take one fast route for the common case;
  `mul_`/`sub_` write in place; `addmm` adds its bias inside the op.
- MojoCCL reduce-scatter is one push+reduce kernel per call (was ~122 chunked
  all-reduces and a host synchronize per call); all-gather reads its local
  contribution once. Per-collective device time is within 10% of NCCL at every
  FSDP2 size for 2 ranks; at 8 ranks the once-per-step root collectives are
  1.11–1.16x NCCL.

Remaining at 8 GPUs (under investigation): the optimizer phase's host time is
still a few ms behind CUDA per step, and MojoCCL's comm kernels contend with
compute for SMs (its grid is far larger than NCCL's).

Correctness: the two-rank distributed suite (NCCL and MojoCCL: collectives,
DDP parity, stream ordering, stress, abort, both FSDP2 modes, chunked
reduce-scatter, AVG overflow) and the targeted native suites pass; GPT-2 124M
prints the same five-step loss trajectory under both collective libraries as
before these changes.

## Initial measurements

Measured on 2026-09-17 with two NVIDIA H100 80GB HBM3 GPUs connected by NVLink,
on an exclusive Slurm node (`par2dc5-ai-prd-cl02s02dgx23`, job 256161).
These are the initial end-to-end training measurements taken before the
throughput optimizations, including the correctness-first MojoCCL reduce-scatter.

### Initial results

Throughput is aggregate input tokens/second across both GPUs. Higher is better.
Each result is the median of six synchronized 10-step windows: three windows
from each of two independent torchrun launches, with the order reversed in the
second round. The range includes every window; no samples were discarded.

#### Sequence length 1024

| Configuration | Tokens/s | Step time (ms) | Window range (tokens/s) |
|---|---:|---:|---:|
| Stock PyTorch CUDA + NCCL | 9,150.4 | 223.82 | 8,689.3–9,167.5 |
| Torch Mojo + NCCL | 6,088.0 | 336.40 | 5,741.0–6,110.3 |
| Torch Mojo + MojoCCL | 4,114.8 | 497.72 | 4,023.3–4,130.4 |

#### Sequence length 64

| Configuration | Tokens/s | Step time (ms) | Window range (tokens/s) |
|---|---:|---:|---:|
| Stock PyTorch CUDA + NCCL | 572.1 | 223.73 | 570.4–573.7 |
| Torch Mojo + NCCL | 362.7 | 352.88 | 340.4–364.2 |
| Torch Mojo + MojoCCL | 259.1 | 493.97 | 254.2–260.9 |

At sequence length 1024, Mojo + NCCL delivers 66.5% of the stock CUDA
throughput. Mojo + MojoCCL delivers 45.0% of stock CUDA and 67.6% of
Mojo + NCCL throughput (32.4% lower throughput when replacing NCCL).

## Training configuration

- Standard Hugging Face GPT-2 XL: 48 layers, 25 heads, width 1600,
  vocabulary 50257, 1,557,611,200 parameters; random initialization.
- Batch size 1 per GPU, global batch size 2; fixed rank-specific synthetic
  tokens, dropout disabled, no activation checkpointing or accumulation.
- FSDP2 wraps each transformer block and the root; default resharding.
- BF16 block parameters, FP32 gradient reduction, and BF16 autocast.
  The root retains FP32 embedding/head parameters on every configuration,
  matching the current Mojo embedding-backward requirement.
- AdamW with learning rate 1e-4, default betas/epsilon/weight decay,
  `foreach=False`, gradient norm clipping at 1.0 with `foreach=False`.
- Eager execution, without `torch.compile`; Transformers selects its SDPA
  interface on both devices, with each device using its current dispatch.
  Supported Hopper inputs use Mojo's flash-attention path; other inputs use
  its math decomposition. The original timing runs did not capture traces.
- PyTorch 2.11.0+cu128, Transformers 5.4.0, MAX/Mojo 26.5.0,
  NCCL 2.28.9 for both vendor-library configurations; driver 570.211.01.
- Same CPU allocation and `OMP_NUM_THREADS=1`; physical GPUs 0 and 1,
  protected by both `/tmp/gpu_lock_0.lock` and `/tmp/gpu_lock_1.lock`.
- GPU clock locking was denied by the driver. Five-second telemetry samples
  with nonzero GPU utilization all reported 1980 MHz SM clocks; no tuning
  or clock settings were changed.

Five complete training steps warm up each process before timing. Each timed
window includes forward, backward, clipping, optimizer update, and gradient
clearing. Initialization, compilation, warmup, loss checks, and logging are
excluded. The device is synchronized before and after each window; a CPU
Gloo control group selects the maximum elapsed time across ranks. The token
count is `world_size * batch_size * sequence_length * steps`.

All runs completed with finite losses. Synthetic repeated batches make this
a throughput comparison, not an assessment of model quality. These results
apply to the stated configuration, not maximum throughput after batch-size,
optimizer, or compiler tuning. The initial MojoCCL reduce-scatter performed
extra communication and synchronized the CPU (see the snapshot above for the
current one).

## Reproduction

Use a CUDA-compatible PyTorch environment (this machine needs the CUDA 12.8
wheel; the project environment’s CUDA 13 wheel cannot initialize its driver).
Pin the workspace on `PYTHONPATH` when using an external environment to avoid
accidentally importing another editable checkout.

```bash
export CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1
export PYTHONPATH="$PWD"
export TORCH_MOJO_BACKEND_CCL=vendor
flock /tmp/gpu_lock_0.lock flock /tmp/gpu_lock_1.lock \
  uv run --no-project --python /path/to/cu128-venv/bin/python \
  python -m torch.distributed.run --standalone --nproc-per-node=2 \
  demo_scripts/gpt2_fsdp2.py --model gpt2-xl --device cuda \
  --dtype bfloat16 --sequence-length 1024 --batch-size 1 \
  --benchmark --warmup 5 --steps 10 --windows 3 --output cuda.json
```

For Mojo + NCCL, use `--device mojo` with `TORCH_MOJO_BACKEND_CCL=vendor`.
For Mojo + MojoCCL, use `--device mojo` with `TORCH_MOJO_BACKEND_CCL=mojo`.
Repeat with `--sequence-length 64`, then reverse the configuration and sequence
order for the second round. The JSON output preserves all timed windows.

The working tree was based on `46ce5ba8da4361eb24b71a28f252eedf4dbed740`,
with the FSDP2/MojoCCL support and benchmark additions present. The source
snapshot and raw run records are retained locally under
`current_bench_train/fsdp2_throughput/`.

## Profiling

Nsight Systems identified excess copies and launches as well as slower
compute kernels. The optimizer now launches 4,640 kernels per rank-step,
matching CUDA, after removing 580 redundant scalar-fill launches. Gradient
clipping's 581 device-to-device copies were reduced to one, and batched FSDP
split-copy reduced 1,152 row-copy launches to 96. These counts describe the
captured phases; overlapping GPU durations must not be added as step time.

Remaining measured targets include BF16 NT matrix multiplications, FP32
activation backward and square root, mixed-dtype gradient packing, and two
large root FP32 copies that take about 1.75 ms each in the generic strided
copy implementation. Longer NCCL GPU spans include waiting for other ranks;
they do not by themselves demonstrate a collective-bandwidth bottleneck.

Nsight Compute hardware counters are unavailable on this allocation
(`ERR_NVGPUCTRPERM`). Kernel timing uses Nsight Systems; exported PTX and
assembler diagnostics revealed serialization and register spills in some
GEMM variants. The experimental GEMM changes have not passed the per-kernel
acceptance checks and are not included in the reported throughput.

`--profile profiles/` writes per-rank PyTorch traces, operator tables, and
cProfile data after the timed windows. `--nsys` marks the timed windows for
Nsight Systems capture and labels forward, backward, clipping, optimizer,
and gradient clearing. Use a CUDA-enabled torch environment for Nsight:

```bash
nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --export=sqlite \
  -o fsdp2_mojo \
  uv run --no-project --python /path/to/cu128-venv/bin/python \
  python -m torch.distributed.run --standalone --nproc-per-node=2 \
  demo_scripts/gpt2_fsdp2.py --model gpt2-xl --device mojo \
  --dtype bfloat16 --sequence-length 1024 --batch-size 1 \
  --benchmark --warmup 5 --steps 5 --windows 1 --nsys
```

Acquire the same GPU locks as in the timing command. Profiling adds overhead;
use separate unprofiled runs for throughput comparisons.
