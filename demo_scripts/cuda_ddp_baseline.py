"""torch-cuda DDP baseline mirroring demo_scripts/multi_gpu_ddp.py.

Standard PyTorch data parallelism — one PROCESS per GPU, NCCL allreduce —
with the exact model, shapes, optimizer and timing methodology of the mojo
DDP demo, so the two stacks can be compared number for number.

This script does NOT import torch_mojo_backend and cannot run in the
project venv on this cluster (its torch is a cu130 build; the driver is
CUDA 12.8). Run it from a venv with driver-compatible cuda wheels, e.g.:

    uv venv ~/.cache/claude-cuda-baseline/venv --python 3.12
    uv pip install --python ~/.cache/claude-cuda-baseline/venv/bin/python \
        torch --index-url https://download.pytorch.org/whl/cu128

    srun --gres=gpu:8 ... \
        ~/.cache/claude-cuda-baseline/venv/bin/torchrun --standalone \
        --nproc-per-node=8 demo_scripts/cuda_ddp_baseline.py --steps 20
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


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
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2048, help="per-rank batch")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--bucket-cap-mb", type=int, default=25)
    args = parser.parse_args()

    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl" if world > 1 else "gloo")
    device = f"cuda:{rank}"

    model = make_model(args.dim, args.hidden).to(device)
    ddp = (
        DDP(model, device_ids=[rank], bucket_cap_mb=args.bucket_cap_mb)
        if world > 1
        else model
    )
    optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-3)

    torch.manual_seed(1000 + rank)
    x = torch.randn(args.batch, args.dim).to(device)
    y = torch.randn(args.batch, args.dim).to(device)

    # Warmup step, exactly as the mojo demo counts it.
    optimizer.zero_grad()
    ((ddp(x) - y) ** 2).mean().backward()
    optimizer.step()
    torch.cuda.synchronize(rank)
    dist.barrier()

    start = time.perf_counter()
    rank_losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        loss = ((ddp(x) - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        rank_losses.append(loss.detach())
    torch.cuda.synchronize(rank)
    dist.barrier()
    elapsed = time.perf_counter() - start

    elapsed_max = torch.tensor([elapsed], device=device)
    if world > 1:
        dist.all_reduce(elapsed_max, op=dist.ReduceOp.MAX)
    wall = float(elapsed_max)
    if rank == 0:
        samples = args.steps * args.batch * world
        print(
            f"world_size={world} steps={args.steps} per-rank batch={args.batch} "
            f"bucket_cap_mb={args.bucket_cap_mb} allreduce=nccl "
            f"torch={torch.__version__}"
        )
        print(f"wall: {wall:.3f}s   throughput: {samples / wall:,.0f} samples/s")
        print(
            f"rank 0 loss: {float(rank_losses[0].cpu()):.4f} -> "
            f"{float(rank_losses[-1].cpu()):.4f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
