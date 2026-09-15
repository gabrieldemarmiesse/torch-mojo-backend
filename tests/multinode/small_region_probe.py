"""Allreduces the pipeline cuts into many chunks must still be correct.

`MOJOCCL_REGION_MB=1` makes every multi-node chunk a few hundred KiB, so a
129 MiB allreduce is more than `WORK_SLOTS` (512) chunks -- past what the
fused kernel, which files every chunk's exchange before it launches, can
take; `_do_allreduce` has to hand that call to the split schedule, and a 20
MiB one (still dozens of chunks) stays fused. A wrong dispatch is a hang
(the host waiting on a ring slot only its own unlaunched kernel would
free), which is why the sizes straddle the bound. At the default region the
same sizes straddle `MOJOCCL_FUSED_BIG_MB` instead, so the 129 MiB call runs
the big grid. Both are checked against the analytic sum::

    MOJOCCL_REGION_MB=1 TORCH_MOJO_BACKEND_CCL=mojo torchrun --nnodes=2 ... \\
        tests/multinode/small_region_probe.py

`SIZES_MIB` (default `20,129`) overrides the sizes.
"""

import os
import sys

from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import datetime  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from torch_mojo_backend import register_mojo_devices  # noqa: E402


def main():
    register_mojo_devices()
    dist.init_process_group(backend="mojo", timeout=datetime.timedelta(seconds=600))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("mojo", torch.accelerator.current_device_index())
    sizes = [int(s) for s in os.environ.get("SIZES_MIB", "20,129").split(",")]
    bad = []
    for mib in sizes:
        n = mib << 18  # fp32 elements
        x = torch.full((n,), float(rank + 1), dtype=torch.float32, device=device)
        x[::7] = float(rank)  # not a constant, so a misplaced slice shows
        dist.all_reduce(x)
        torch.accelerator.synchronize()
        want = torch.full((n,), float(world * (world + 1) // 2), dtype=torch.float32)
        want[::7] = float(world * (world - 1) // 2)
        got = x.cpu()
        if not torch.equal(got, want):
            wrong = int((got != want).sum())
            bad.append(f"{mib} MiB: {wrong} of {n} elements wrong")
        del x
    verdict = "OK" if not bad else "FAIL " + "; ".join(bad)
    print(f"[rank {rank}] small_region payload={verdict}", flush=True)
    dist.destroy_process_group()
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
