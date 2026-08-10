"""Allreduce bandwidth benchmark for the mojo eager collectives.

Reports algorithm bandwidth (bytes moved per second per rank) and bus
bandwidth (algbw * 2*(n-1)/n, the nccl-tests metric that is comparable
across world sizes and to NCCL's own benchmarks).

Example:
    MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas \
        uv run demo_scripts/multi_gpu_allreduce_bench.py
"""

import argparse
import time

import torch

from torch_mojo_backend import distributed as mojo_dist
from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.mojo_device.torch_mojo_tensor import get_ordered_accelerators


def sync_all(world: int) -> None:
    for i in range(world):
        torch.mojo.synchronize(i)


def bench_size(world: int, numel: int, dtype: torch.dtype, iters: int) -> float:
    tensors = [
        torch.randn(numel, dtype=torch.float32).to(dtype).to(f"mojo:{i}")
        for i in range(world)
    ]
    mojo_dist.all_reduce_out(tensors)  # warmup: signals + kernel compile
    sync_all(world)
    start = time.perf_counter()
    for _ in range(iters):
        outs = mojo_dist.all_reduce_out(tensors)
    sync_all(world)
    elapsed = (time.perf_counter() - start) / iters
    del outs
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-devices", type=int, default=None)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    register_mojo_devices()
    gpu_count = sum(acc.label == "gpu" for acc in get_ordered_accelerators())
    if gpu_count < 2:
        raise SystemExit("needs at least 2 MAX GPUs")
    max_devices = min(args.max_devices or gpu_count, gpu_count)
    dtype = getattr(torch, args.dtype)
    itemsize = torch.tensor([], dtype=dtype).element_size()

    worlds = [n for n in (2, 4, 8) if n <= max_devices]
    sizes_bytes = [1 << p for p in range(15, 29)]  # 32 KiB .. 256 MiB

    print(f"dtype={args.dtype} iters={args.iters}")
    header = f"{'bytes':>12}" + "".join(
        f"{f'{n}gpu algbw':>14}{f'{n}gpu busbw':>14}" for n in worlds
    )
    print(header + "   (GB/s)")
    print("-" * len(header))
    for nbytes in sizes_bytes:
        numel = nbytes // itemsize
        row = f"{nbytes:>12,}"
        for world in worlds:
            seconds = bench_size(world, numel, dtype, args.iters)
            algbw = nbytes / seconds / 1e9
            busbw = algbw * 2 * (world - 1) / world
            row += f"{algbw:>14.1f}{busbw:>14.1f}"
        print(row)


if __name__ == "__main__":
    main()
