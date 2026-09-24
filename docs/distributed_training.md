# Distributed training

The `mojo` device has its own `torch.distributed` backend, also called
`"mojo"`, which runs collectives on mojo tensors through the GPU vendor's
collective library: NCCL on NVIDIA and RCCL on AMD. A DDP or FSDP2 script
written for CUDA needs only a few changed lines, and you launch it with
`torchrun` as usual. You don't need a CUDA or ROCm build of PyTorch.

Read [Accelerator API](accelerator_api.md) first if you don't yet know how to
get a tensor onto the device.

## What is supported

| | NVIDIA | AMD | Apple |
|---|---|---|---|
| Collective library | NCCL | RCCL (from your ROCm install) | none |
| DDP (`DistributedDataParallel`) | single node and multi-node | single node and multi-node | not supported |
| FSDP2 (`fully_shard`) | single node and multi-node | not validated yet | not supported |

- Run one process per GPU and launch with `torchrun`, either directly or
  under SLURM. Every node needs the same setup.
- GPU collectives need Linux on x86-64. On other architectures (Grace Hopper
  nodes are aarch64, for example), `init_process_group` raises
  `NotImplementedError`.
- The multi-GPU runs behind this page used H100 (NVIDIA) and MI300A (AMD)
  GPUs, in eager mode. `torch.compile` isn't supported on the mojo device yet
  (see the [home page](index.md)).
- Tensor and pipeline parallelism, 2-D device meshes (HSDP), FSDP1
  (`FullyShardedDataParallel`), and DDP's `find_unused_parameters=True` and
  `static_graph=True` are not covered by tests yet.

## Installing the collective library

Install the package itself as described in
[Installation](index.md#installation). Distributed training also needs the
vendor's collective library on every node:

=== "NVIDIA"

    The backend loads `libnccl.so.2` from the first of these that exists:

    1. the path in `TORCH_MOJO_BACKEND_NCCL_LIB`;
    2. an installed NCCL wheel (`nvidia-nccl-cu12` or `nvidia-nccl-cu13`);
    3. `libnccl.so.2` on the system library path.

    The CUDA builds of torch on PyPI already depend on one of these NCCL
    wheels, so they bring NCCL with them. With the CPU build of torch,
    install NCCL yourself:

    ```bash
    pip install torch-mojo-backend
    pip install "nvidia-nccl-cu12>=2.27"
    ```

    NCCL has to work with your NVIDIA driver: the CUDA 13 wheel
    (`nvidia-nccl-cu13`) needs driver 580 or newer.

=== "AMD"

    RCCL ships with ROCm, so there is nothing extra to install. The backend
    loads `librccl.so.1` from the first of these that exists:

    1. the path in `TORCH_MOJO_BACKEND_RCCL_LIB`;
    2. the ROCm install whose HIP runtime is already loaded in the process;
    3. `$ROCM_PATH/lib`, then `/opt/rocm/lib`;
    4. `librccl.so.1` on the system library path.

    Using the RCCL from the same ROCm as the HIP runtime keeps a single HIP
    runtime in the process. If your ROCm isn't where the backend looks (for
    example, a cluster module), set `ROCM_PATH` for every rank.

    Install the CPU build of torch:

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install torch-mojo-backend
    ```

    The other builds cause problems on AMD:

    - The default PyPI build of torch is a CUDA build. It loads gigabytes of
      NVIDIA libraries, and the HIP runtime rescans all of them every time it
      loads a kernel, so each first-use kernel load takes about 10 times
      longer.
    - A ROCm build of torch loads its own HIP runtime next to the one this
      backend uses, and RCCL can't work with buffers from two runtimes.

    `register_mojo_devices()` warns you if it detects either build.

To see which library a run loaded, check stderr. Each rank prints one line
when `init_process_group` runs:

```text
[TRACE] collectives via /.../site-packages/nvidia/nccl/lib/libnccl.so.2 (nccl version 23102)
```

The number is the library's integer version code (23102 means 2.31.2). Set
`TORCH_MOJO_BACKEND_TRACE=0` to hide this line and the kernel build timings.

## A minimal DDP script

```python title="train_ddp.py"
from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()  # first: give this process its own GPU

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()


def main():
    dist.init_process_group(backend="mojo")
    rank = dist.get_rank()
    device = torch.device("mojo")

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(32, 128), torch.nn.GELU(), torch.nn.Linear(128, 2)
    ).to(device)
    model = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Each rank draws its own batches.
    generator = torch.Generator().manual_seed(1000 + rank)
    for step in range(1, 101):
        x = torch.randn(64, 32, generator=generator)
        y = (x.sum(dim=1) > 0).long()  # a toy two-class problem
        x, y = x.to(device), y.to(device)

        loss = torch.nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # DDP all-reduces the gradients during backward
        optimizer.step()

        if step % 20 == 0:
            mean_loss = loss.detach().clone()
            dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)
            if rank == 0:
                print(f"step {step:3d}  loss {mean_loss.item():.4f}")

    if rank == 0:  # torch.save needs CPU tensors
        state = {k: v.cpu() for k, v in model.module.state_dict().items()}
        torch.save(state, "checkpoint.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

To run it on every GPU of one machine:

```bash
torchrun --standalone --nproc-per-node=8 train_ddp.py
```

The differences from a CUDA script:

1. `use_local_rank_gpu()` runs first. It sets the vendor's visibility
   variable so that each torchrun worker sees exactly one GPU, chosen by
   `LOCAL_RANK`; `"mojo"` (that is, `mojo:0`) is then always that worker's
   GPU. If a launcher such as SLURM already exported a list of several GPUs
   (`CUDA_VISIBLE_DEVICES=0,...,7`, or `ROCR_VISIBLE_DEVICES` on AMD), each
   worker keeps only its own entry, and a list with a single entry is left
   alone. Outside torchrun (no `LOCAL_RANK`), the call does nothing.
   Call it before `register_mojo_devices()` and before anything touches the
   GPU: the GPU list is read only once per process, so a later call has no
   effect.
2. `register_mojo_devices()` registers both the device and the `"mojo"`
   process-group backend.
3. `init_process_group(backend="mojo")` replaces `"nccl"`, and creates the
   communicator right away, on the current GPU. The same group also handles
   CPU tensors (object collectives, CPU `all_reduce`) through a private gloo
   group, so you don't need a `"cpu:gloo,..."` string.
   `init_process_group(device_id=0)` and
   `init_process_group(device_id=torch.device("mojo", 0))` work too.
4. `DDP(model)` takes no `device_ids`. Move your inputs to the device
   yourself. DDP's default `broadcast_buffers=True` works, and if your
   buffers never change (a causal mask, for example), passing
   `broadcast_buffers=False` saves one broadcast per forward pass.
5. Checkpoints are saved from CPU copies, because calling `torch.save`
   directly on mojo tensors raises `NotImplementedError`.

The first run compiles each kernel the first time it's used, so the first
steps are slow. Later runs load the kernels from the on-disk cache.

### Porting an existing CUDA script

| Stock PyTorch on CUDA | mojo device |
|---|---|
| `torch.cuda.set_device(local_rank)` | `use_local_rank_gpu()` at the top of the script |
| `dist.init_process_group("nccl")` | `register_mojo_devices()`, then `dist.init_process_group("mojo")` |
| `model.to(f"cuda:{local_rank}")` | `model.to("mojo")` |
| `DDP(model, device_ids=[local_rank])` | `DDP(model)` |
| `torch.autocast("cuda", dtype=torch.bfloat16)` | `torch.autocast("mojo", dtype=torch.bfloat16)` |
| `torch.cuda.manual_seed_all(seed + rank)` | `torch.mojo.manual_seed_all(seed + rank)` |
| `torch.cuda.synchronize()` | `torch.accelerator.synchronize()` |

Autocast, per-rank device seeds and fused AdamW all work under DDP. For a
complete example with bf16 autocast, gradient clipping and evaluation, see
the repository's nanoGPT DDP demo,
[`demo_scripts/nanogpt_ddp.py`](https://github.com/gabrieldemarmiesse/torch-mojo-backend/blob/main/demo_scripts/nanogpt_ddp.py).

### One script for CUDA and mojo

If the rest of the script only refers to `device`, the same file runs on
stock CUDA as well:

```python
from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import torch
import torch.distributed as dist

import torch_mojo_backend

torch_mojo_backend.register_mojo_devices()  # remove this line to use stock CUDA

device = torch.accelerator.current_accelerator()  # mojo, or cuda without the line above
dist.init_process_group(backend=dist.get_default_backend_for_device(device))  # "mojo" or "nccl"
```

On stock CUDA, `use_local_rank_gpu()` also leaves each rank one visible GPU,
so `cuda` is the right device there too. For autocast, write
`torch.autocast(device.type, ...)`. [Accelerator API](accelerator_api.md)
covers `torch.accelerator` in more detail.

## FSDP2

Use PyTorch's `fully_shard` with a mojo device mesh. The usual FSDP2 rules
apply: shard the blocks before the root module, and create the optimizer
after sharding.

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# after use_local_rank_gpu(), register_mojo_devices() and init_process_group("mojo")
mesh = init_device_mesh("mojo", (dist.get_world_size(),))
model = MyModel().to("mojo")
for block in model.blocks:
    fully_shard(block, mesh=mesh)
fully_shard(model, mesh=mesh)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
```

- The multi-GPU tests use `foreach=False` for the optimizer, as above.
- `torch.distributed.checkpoint` (`dcp.save` / `dcp.load`) can save and
  restore a sharded model together with its optimizer state.
- For bf16 training, `MixedPrecisionPolicy(param_dtype=torch.bfloat16,
  reduce_dtype=torch.float32)` on the blocks works. Leave the module that
  owns the token embedding (usually the root) in fp32, because the device's
  embedding backward currently needs fp32 gradients.
- [`demo_scripts/gpt2_fsdp2.py`](https://github.com/gabrieldemarmiesse/torch-mojo-backend/blob/main/demo_scripts/gpt2_fsdp2.py)
  trains GPT-2 124M or 1.5B (XL) this way, with no download needed.

## Collectives

All tensors in one call must be on the same mojo device, or all on the
CPU. The table covers mojo tensors; CPU tensors always go through gloo.

| `torch.distributed` call | NCCL / RCCL (default) | Mojo collectives |
|---|---|---|
| `all_reduce` | SUM, AVG, MIN, MAX, PRODUCT | SUM, AVG |
| `broadcast` | yes | yes |
| `all_gather`, `all_gather_into_tensor` | yes (equal-sized tensors) | yes |
| `reduce_scatter`, `reduce_scatter_tensor` | yes | SUM, AVG |
| `reduce` | yes | no |
| `all_to_all`, `all_to_all_single` | yes, uneven splits included | no |
| `gather`, `scatter` | yes | no |
| `send`, `recv`, `isend`, `irecv` | yes | no |
| `batch_isend_irecv` | implemented, not covered by the tests | no |
| `barrier` | yes | yes |
| `all_gather_object`, `broadcast_object_list`, other object collectives | yes | yes |

The vendor libraries handle `bool`, `uint8`, `int8`, `int32`, `uint32`,
`int64`, `uint64`, `float16`, `bfloat16`, `float32` and `float64`. `bool`
follows `ProcessGroupNCCL`'s rules: SUM and MAX act as a logical OR, PRODUCT
and MIN act as a logical AND, and AVG raises `TypeError`.
`ReduceOp.PREMUL_SUM` isn't supported. Mojo collectives reduce only
`float32`, `float16`, `bfloat16`, `int32` and `int64`, but they broadcast and
gather any of the dtypes above.

With `async_op=True`, a call returns a `Work` with the same stream semantics
as NCCL: `work.wait()` makes the *current stream* wait for the collective
without blocking the host, while `work.is_completed()` and
`work.get_future()` behave as usual. Passing a finite timeout,
`work.wait(timeout=...)`, raises `RuntimeError`. If you drop your references
to a tensor while a collective still uses it, its memory isn't reused until
the collective finishes. [Streams and events](streams_and_events.md) explains
how streams work on the device.

!!! note
    Consuming a collective's result on the stream that issued it gives the
    most reliable results, and it is what DDP, FSDP2 and blocking calls do.
    The tests also cover waiting from a different stream, but that pattern
    has failed intermittently in the past, and the root cause was never
    found.

## Multi-node

Run one `torchrun` per node, with the same rendezvous arguments on every
node:

```bash
torchrun --nnodes=2 --nproc-per-node=8 \
    --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:29500 \
    --rdzv-id=$JOB_ID train_ddp.py
```

Under SLURM, start one task per node and let torchrun start the per-GPU
workers:

```bash
#!/bin/bash
#SBATCH --nodes=2 --ntasks-per-node=1 --gpus-per-node=8 --exclusive
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
srun torchrun --nnodes="$SLURM_JOB_NUM_NODES" --nproc-per-node=8 \
    --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:29500" \
    --rdzv-id="$SLURM_JOB_ID" train_ddp.py
```

The backend needs no extra setup for multi-node runs: communicators bootstrap
through the torchrun rendezvous, and `use_local_rank_gpu()` splits the GPU
list that SLURM exports among the workers of each node.

NCCL's and RCCL's own environment variables, such as `NCCL_DEBUG`,
`NCCL_SOCKET_IFNAME` and `NCCL_IB_HCA`, still apply. The library itself
reads them, so they work exactly as with stock PyTorch, and so does any
network plugin your cluster needs, such as `aws-ofi-rccl` on HPE Slingshot.
PyTorch's `TORCH_NCCL_*` variables configure `ProcessGroupNCCL`, which this
backend doesn't use, so they have no effect.

Warm the kernel cache first. Every rank compiles the kernels it's missing
the first time it uses them, and on a cold cache the first step can take
minutes. Put the cache on a filesystem that every node sees
(`TORCH_MOJO_BACKEND_CACHE_DIR`, which defaults to the user's cache
directory), preferably a fast one, and do a short single-process run
(`--nproc-per-node=1`, a few steps) before the big launch.
`torch-mojo-backend cache dir` prints where the cache is.

??? note "AMD MI300A (APU): device memory is host memory"
    By default, MAX reserves about 90% of the device's memory in each
    process at the first allocation. On an MI300A that memory is the node's
    RAM, so four ranks per node leave little for anything else, and
    processes can be killed for lack of memory while compiling kernels.
    Setting MAX's `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM=1` makes each
    process hold only what it uses. With MAX 26.5 and ROCm 6.4.3, though,
    the process then segfaults during interpreter exit, after your script
    has finished. To avoid this, end the script with `os._exit(0)` once
    `destroy_process_group()` and checkpointing are done, as
    `demo_scripts/nanogpt_ddp.py` does. Without the variable, warm the
    kernel cache with a single process before launching four ranks per node.

## Mojo collectives (experimental)

Setting `TORCH_MOJO_BACKEND_CCL=mojo` replaces NCCL/RCCL with the project's
own collectives, an implementation of NCCL's C API written in Mojo that
covers the node-to-node hop too. Your script and the process group stay the
same; only the loaded library changes, and the `[TRACE] collectives via ...`
line then names `libmojoccl.so`. The library is compiled the first time you
use it, which takes about a minute, and is then cached with the kernels.
NCCL/RCCL remain the default.

```bash
TORCH_MOJO_BACKEND_CCL=mojo torchrun --standalone --nproc-per-node=8 train_ddp.py
```

Set the variable on every rank of every node. Consider Mojo collectives
when NCCL or RCCL isn't available, or when you want to try a collective
stack that doesn't come from the GPU vendor. They cover what DDP and FSDP2
need, and the project has trained with them on H100 (DDP and
FSDP2, up to two nodes over InfiniBand) and on MI300A (DDP, up to two nodes
over Slingshot).

Limitations:

- `all_reduce`, `broadcast`, `all_gather`, `reduce_scatter`, `barrier` and
  the object collectives work. `reduce`, point-to-point, `all_to_all`,
  `gather` and `scatter` raise. See the [Collectives](#collectives) table.
- Reductions support only SUM and AVG, on the five dtypes listed under
  [Collectives](#collectives).
- `all_reduce` needs a tensor that starts at a 16-byte-aligned address.
  Every freshly allocated tensor does, but a view such as `t[1:]` may not,
  and then fails with
  `RuntimeError: ncclAllReduce failed: ... invalid argument`. Call
  `.clone()` on the view first.
- Mojo collectives support at most 8 GPUs per node and 16 nodes.
- Multi-node runs need InfiniBand with GPUDirect RDMA (`nvidia_peermem`)
  through libibverbs, or a libfabric provider with RMA and `FI_HMEM` support
  (HPE Slingshot's `cxi`). The ranks also need a routable network interface
  for the TCP bootstrap.
- A late peer causes an error instead of a hang. A rank that waits for a
  peer longer than `MOJOCCL_IB_TIMEOUT_S` (60 s by default) gives up: it
  prints a line naming the collective and the peer it waited for, and its
  next collective raises `RuntimeError`. If one rank can legitimately reach
  a collective much later than the others, raise the timeout.

The variables in this table are the only tuning knobs, and the defaults
suit the hardware the library was measured on:

| Variable | Default | Effect |
|---|---|---|
| `MOJOCCL_SOCKET_IFNAME` | an up, non-loopback interface with a default route | network interface for the TCP bootstrap |
| `MOJOCCL_BOOTSTRAP_TIMEOUT_S` | 120 | seconds communicator creation waits for every rank to check in |
| `MOJOCCL_IB_TIMEOUT_S` | 60 | seconds a collective waits for its peers before failing |
| `MOJOCCL_NET` | InfiniBand if an active port exists, else libfabric | `verbs` or `fabric` forces the node-to-node transport |
| `MOJOCCL_IB_HCA` | chosen by GPU/NIC locality | InfiniBand adapter to use |
| `MOJOCCL_FABRIC_DOMAIN` | chosen by GPU/NIC locality | libfabric domain (NIC) to use, such as `cxi0` |
| `MOJOCCL_FABRIC_PROVIDER` | `cxi` | libfabric provider |
| `MOJOCCL_LIBFABRIC` | system loader, then the Cray install | path to `libfabric.so.1` |
| `MOJOCCL_REGION_MB` | 64 on gfx942 (MI300 series), 256 elsewhere | size of each rank's staging buffer, in MiB |
| `MOJOCCL_NVLS` | on where every GPU supports it | `0` disables the NVLink SHARP (multicast) path |
| `MOJOCCL_IB_TRACE` | `0` | `1` prints what each rank negotiated (transport, device, ports) |

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `TORCH_MOJO_BACKEND_CCL` | unset (vendor library) | `mojo` selects the [Mojo collectives](#mojo-collectives-experimental) |
| `TORCH_MOJO_BACKEND_NCCL_LIB` | wheel, then system | absolute path of the `libnccl.so.2` to load |
| `TORCH_MOJO_BACKEND_RCCL_LIB` | ROCm search | absolute path of the `librccl.so.1` to load |
| `ROCM_PATH` | `/opt/rocm` | ROCm install to take the HIP runtime and RCCL from |
| `TORCH_MOJO_BACKEND_CACHE_DIR` | user cache directory | where compiled kernels (and `libmojoccl.so`) are cached |
| `TORCH_MOJO_BACKEND_TRACE` | on | `0` hides the `[TRACE]` lines (library in use, build timings) |

`register_mojo_devices()` warns about any `TORCH_MOJO_BACKEND_*` or
`MOJOCCL_*` variable that it doesn't recognise, and suggests the real name
when one is close, so a typo doesn't go unnoticed.

## Troubleshooting

### `no NCCL/RCCL library found`

`init_process_group` couldn't find the library. On NVIDIA, install
`nvidia-nccl-cu12` or set `TORCH_MOJO_BACKEND_NCCL_LIB`. On AMD, the error
message still suggests `nvidia-nccl-cu12`, but what's missing there is
`librccl.so.1` from ROCm: set `ROCM_PATH` to your ROCm install or
`TORCH_MOJO_BACKEND_RCCL_LIB` to the library. See
[Installing the collective library](#installing-the-collective-library).

### NCCL fails to initialise on an older driver

The NCCL that was found is built for a newer CUDA than your driver
supports. For example, the `nvidia-nccl-cu13` wheel that recent CUDA builds
of torch install needs driver 580 or newer. The `[TRACE] collectives
via ...` line shows which file was loaded. Either point
`TORCH_MOJO_BACKEND_NCCL_LIB` at an NCCL that works with your driver, or use
the CPU build of torch together with `nvidia-nccl-cu12`.

### All ranks use the same GPU, or communicator creation fails

`use_local_rank_gpu()` wasn't called, or was called after the GPU had
already been initialised. Make it the first thing the script does. If it
raises `LOCAL_RANK=... but CUDA_VISIBLE_DEVICES=... lists only N devices`,
you started more processes per node than there are visible GPUs.

### A run hangs

NCCL and RCCL collectives have no watchdog here. The `timeout=` of
`init_process_group` applies only to the rendezvous and to CPU work
(`barrier`, object collectives). When a worker crashes, torchrun stops the
others. A hang with every process still alive usually means the ranks
issued different collectives, or issued them in a different order.
`NCCL_DEBUG=INFO` shows how the communicators were set up, and
`py-spy dump --pid <pid>` shows where each rank is waiting. With Mojo
collectives, a rank stops waiting after `MOJOCCL_IB_TIMEOUT_S` and its next
collective raises, so the run fails instead of hanging.

### The first training step takes minutes

Kernels compile the first time each one is used; see
[warm the kernel cache first](#multi-node). Later runs reuse the cache.

### `ValueError: the mojo distributed backend handles mojo and cpu tensors`

A tensor from another device, such as `cuda`, reached a collective. With a
CUDA build of torch installed, check that the model and inputs went to
`"mojo"` (or to `torch.accelerator.current_accelerator()`), not to `"cuda"`.

### `NotImplementedError` from `torch.save`

Mojo tensors can't be serialised directly. Copy them to the CPU first:
`{k: v.cpu() for k, v in model.state_dict().items()}`. For FSDP2 sharded
state, use `torch.distributed.checkpoint`, which works as it is.

### `Time stats are currently only collected for CPU and CUDA devices`

DDP prints this warning once for any device other than CPU or CUDA. You can
ignore it.
