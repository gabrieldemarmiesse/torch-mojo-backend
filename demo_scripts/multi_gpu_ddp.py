"""Data-parallel training on the mojo device with standard DDP.

One process, one thread per GPU rank, gradient allreduce on Modular's
pure-Mojo P2P comm kernels — no NCCL, no torchrun, works with a CPU-only
PyTorch install.

Example:
    MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas \
        uv run demo_scripts/multi_gpu_ddp.py --world-size 8 --steps 20
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_mojo_backend import distributed as mojo_dist
from torch_mojo_backend import register_mojo_devices


def make_model(dim: int, hidden: int) -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, dim),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2048, help="per-rank batch")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument(
        "--bucket-cap-mb",
        type=int,
        default=25,
        help="DDP gradient bucket size (smaller = more, earlier overlap)",
    )
    args = parser.parse_args()

    register_mojo_devices()
    losses: dict[int, list[float]] = {}
    timings: dict[int, float] = {}

    def worker(rank: int, world_size: int, pg: mojo_dist.MojoProcessGroup) -> None:
        device = f"mojo:{rank}"
        model = make_model(args.dim, args.hidden).to(device)
        ddp = DDP(
            model, process_group=pg, device_ids=None, bucket_cap_mb=args.bucket_cap_mb
        )
        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

        torch.manual_seed(1000 + rank)
        x = torch.randn(args.batch, args.dim).to(device)
        y = torch.randn(args.batch, args.dim).to(device)

        # Warmup step compiles every kernel involved.
        optimizer.zero_grad()
        ((ddp(x) - y) ** 2).mean().backward()
        optimizer.step()
        torch.mojo.synchronize(rank)
        pg.barrier().wait()

        start = time.perf_counter()
        rank_losses = []
        for _ in range(args.steps):
            optimizer.zero_grad()
            loss = ((ddp(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            rank_losses.append(loss.detach())
        torch.mojo.synchronize(rank)
        pg.barrier().wait()
        timings[rank] = time.perf_counter() - start
        losses[rank] = [float(loss.cpu()) for loss in rank_losses]

    world = args.world_size
    mojo_dist.spawn(worker, world_size=world)
    world = len(timings)

    wall = max(timings.values())
    samples = args.steps * args.batch * world
    mode = "comm-stream (async)" if mojo_dist._ASYNC_COMM_ENABLED else "sync"
    print(
        f"world_size={world} steps={args.steps} per-rank batch={args.batch} "
        f"bucket_cap_mb={args.bucket_cap_mb} allreduce={mode}"
    )
    print(f"wall: {wall:.3f}s   throughput: {samples / wall:,.0f} samples/s")
    for rank in range(world):
        print(f"rank {rank} loss: {losses[rank][0]:.4f} -> {losses[rank][-1]:.4f}")


if __name__ == "__main__":
    main()
