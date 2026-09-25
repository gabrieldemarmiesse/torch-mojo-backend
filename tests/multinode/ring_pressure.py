"""Regression test: the calling thread outrunning the inter-node work ring.

`transport/net.mojo` keeps its exchanges in a ring of `WORK_SLOTS` (512) work
items, and nothing about the API tells a caller to stay inside it: the
enqueue path only puts kernels on a stream, so the calling thread runs as far
ahead of the GPU -- and therefore of the network -- as torch will let it.

DDP's `_sync_module_states` is the shape that does it: dozens of broadcasts
back to back with no synchronisation anywhere. On InfiniBand an exchange
retires in ~20 us and the host never gets far enough ahead to notice; on
Slingshot a broadcast exchange of a real parameter tensor costs milliseconds,
and `DDP(model)` on nanoGPT-124M died at construction on every rank of the
receiving node with "the inter-node work ring wrapped with an exchange still
in flight" (job 5393676, 2 nodes x 4 MI300A over cxi).

This reproduces that deterministically and in seconds: N broadcasts of one
tensor, issued without a single synchronize, then one synchronize at the end.
Against the code before `_await_ring_slot` it fails on every rank at exchange
513; against the code with it, all N complete and the trace reports how far
the host got ahead and how often it had to wait.

    N=1500 MIB=4 torchrun --nnodes=2 --nproc-per-node=4 ... ring_pressure.py

Measured (2 nodes x 4 MI300A, cxi): host ran up to 513 exchanges ahead and
waited for a ring slot on 846-880 of the 1500 exchanges, payload correct on
every rank. `MOJOCCL_IB_TRACE=1` prints both counters.
"""

from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from torch_mojo_backend import register_mojo_devices  # noqa: E402

register_mojo_devices()
dist.init_process_group(backend="mojo")
rank = dist.get_rank()
world = dist.get_world_size()
n_bcast = int(os.environ.get("N", "1500"))
mib = float(os.environ.get("MIB", "4"))
dev = torch.device("mojo")
n = int(mib * 2**20) // 4
x = torch.full((n,), float(rank + 1), dtype=torch.float32, device=dev)

dist.barrier()
torch.accelerator.synchronize()
t0 = time.perf_counter()
# No synchronize inside the loop: this is the point of the test.
for _ in range(n_bcast):
    dist.broadcast(x, 0)
torch.accelerator.synchronize()
dt = time.perf_counter() - t0
got = x.to("cpu")
ok = bool((got == 1.0).all())
print(
    f"[rank {rank}] {n_bcast} x {mib:g} MiB broadcasts in {dt:.2f}s "
    f"({dt / n_bcast * 1e6:.0f} us each) payload={'OK' if ok else 'WRONG'}",
    flush=True,
)
dist.destroy_process_group()
sys.stdout.flush()
if os.environ.get("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM"):
    os._exit(0)
