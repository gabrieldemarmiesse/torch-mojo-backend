"""Host-side enqueue cost of one ``dist.all_reduce``, both CCLs.

What the DDP reducer actually pays on the autograd thread: the wall time of
the Python call itself, with nothing synchronized. A collective that costs
250 us of host time cannot be issued more than four times a millisecond, and
GPT-2 XL's backward issues 146 of them, so this number -- not the collective's
device time -- is what decides whether the compute stream starves.

Launch (16 ranks, 2 nodes), once per library::

    TORCH_MOJO_BACKEND_CCL=mojo torchrun --nnodes=2 --nproc-per-node=8 ... \
        tests/multinode/enqueue_bench.py

Env: ``SIZES_MIB`` (comma-separated, default ``0.000004,27,168``), ``CALLS``
(default 100), ``DTYPE`` (``float32``/``bfloat16``), ``WARMUP`` (default 10).

Prints one ``RESULT enqueue`` line per size from rank 0, with the median, the
mean and the 90th percentile of the per-call host time; the spread is what
says whether the launch queue backed up partway through the burst.
"""

import os

from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import datetime  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from torch_mojo_backend import register_mojo_devices  # noqa: E402


def main():
    register_mojo_devices()
    dist.init_process_group(backend="mojo", timeout=datetime.timedelta(seconds=300))
    rank = dist.get_rank()
    ccl = os.environ.get("TORCH_MOJO_BACKEND_CCL", "vendor")
    calls = int(os.environ.get("CALLS", "100"))
    warmup = int(os.environ.get("WARMUP", "10"))
    dtype = getattr(torch, os.environ.get("DTYPE", "float32"))
    sizes = [
        float(s) for s in os.environ.get("SIZES_MIB", "0.000004,27,168").split(",")
    ]
    device = torch.device("mojo", torch.accelerator.current_device_index())

    for mib in sizes:
        item = torch.empty(0, dtype=dtype).element_size()
        numel = max(1, int(mib * 1024 * 1024) // item)
        x = torch.ones(numel, dtype=dtype, device=device)
        for _ in range(warmup):
            dist.all_reduce(x)
        torch.accelerator.synchronize()
        dist.barrier()

        per_call = []
        t0 = time.perf_counter()
        for _ in range(calls):
            a = time.perf_counter()
            dist.all_reduce(x)
            per_call.append((time.perf_counter() - a) * 1e6)
        burst = (time.perf_counter() - t0) * 1e6
        torch.accelerator.synchronize()
        drain = (time.perf_counter() - t0) * 1e6
        dist.barrier()

        if rank == 0:
            first = per_call[0]
            per_call.sort()
            print(
                f"RESULT enqueue ccl={ccl} dtype={dtype} size_mib={mib:g}"
                f" calls={calls}"
                f" median_us={statistics.median(per_call):8.1f}"
                f" mean_us={statistics.fmean(per_call):8.1f}"
                f" p90_us={per_call[int(0.9 * len(per_call))]:8.1f}"
                f" first_us={first:8.1f}"
                f" burst_ms={burst / 1000:8.2f}"
                f" drain_ms={drain / 1000:8.2f}",
                flush=True,
            )
        del x

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
