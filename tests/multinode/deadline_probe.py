"""One rank stops answering; the others must fail loudly, in time, not hang.

Rank 0 sleeps past `MOJOCCL_IB_TIMEOUT_S` before its second allreduce. Every
other rank's second allreduce hits the device deadline inside the fused
inter-node kernel, latches the fault, and the THIRD allreduce (the reporting
call is one collective behind) raises `RuntimeError`. Each such rank also
checks that the error arrived within the deadline plus the kernel's 1 s grid
grace plus slack, and a watchdog thread kills the process if nothing
returns at all -- an unbounded spin is the failure this probe exists for and
must not look like a hang of the probe itself::

    MOJOCCL_IB_TIMEOUT_S=3 torchrun --nnodes=2 --nproc-per-node=8 ... \\
        tests/multinode/deadline_probe.py

Every rank prints a `probe:` verdict and exits 0 only on the expected
outcome (rank 0 included: after its sleep its own allreduce waits for peers
that already gave up and must fail the same way), so torchrun's exit code
is the result. With `MOJOCCL_REGION_MB=1 --size-mib 129`, the payload
exceeds the fused work-ring capacity and exercises the split wait kernel.
"""

import argparse
import os
import sys
import threading
import time

from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import datetime  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from torch_mojo_backend import register_mojo_devices  # noqa: E402

GRID_GRACE_S = 1.0  # all_reduce_gin.mojo's _GRID_GRACE_NS
SLACK_S = 5.0


def _watchdog(rank: int, budget_s: float):
    time.sleep(budget_s)
    print(
        f"[rank {rank}] probe: FAIL, watchdog fired after {budget_s:.0f} s", flush=True
    )
    os._exit(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mib", type=int, default=32)
    args = parser.parse_args()
    register_mojo_devices()
    dist.init_process_group(backend="mojo", timeout=datetime.timedelta(seconds=300))
    rank = dist.get_rank()
    timeout_s = float(os.environ.get("MOJOCCL_IB_TIMEOUT_S", "60"))
    sleep_s = timeout_s * 3 + 5
    threading.Thread(
        target=_watchdog, args=(rank, sleep_s + timeout_s * 4 + 30), daemon=True
    ).start()
    device = torch.device("mojo", torch.accelerator.current_device_index())
    x = torch.ones(args.size_mib << 18, dtype=torch.float32, device=device)
    dist.all_reduce(x)
    torch.accelerator.synchronize()
    if rank == 0:
        time.sleep(sleep_s)
    t0 = time.perf_counter()
    err = None
    try:
        dist.all_reduce(x)
        torch.accelerator.synchronize()
        dist.all_reduce(x)
        torch.accelerator.synchronize()
    except RuntimeError as e:
        err = e
    elapsed = time.perf_counter() - t0
    # Rank 0 is not exempt: its own second allreduce waits for peers that
    # have already given up, hits the same deadline, and its third must raise.
    if err is None:
        print(
            f"[rank {rank}] probe: FAIL, no deadline error after {elapsed:.1f} s",
            flush=True,
        )
        sys.exit(1)
    if "remote" not in str(err).lower() and "deadline" not in str(err).lower():
        print(
            f"[rank {rank}] probe: FAIL, unexpected error: {str(err)[:160]}", flush=True
        )
        sys.exit(1)
    bound = timeout_s + GRID_GRACE_S + SLACK_S
    if elapsed > bound:
        print(
            f"[rank {rank}] probe: FAIL, deadline raised late: {elapsed:.1f} s"
            f" > {bound:.1f} s",
            flush=True,
        )
        sys.exit(1)
    print(
        f"[rank {rank}] probe: deadline raised as expected after {elapsed:.1f} s"
        f" (bound {bound:.1f}): {str(err)[:100]}",
        flush=True,
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
