"""Distributed nanoGPT training on the mojo eager device (DDP over NCCL/RCCL).

Launch with torchrun — single node:

    uv run torchrun --standalone --nproc-per-node=8 \
        demo_scripts/nanogpt_ddp.py --nanogpt-path ~/nanoGPT \
        --data-dir ~/nanoGPT/data/shakespeare

multi node (SLURM):

    srun uv run torchrun --nnodes=$SLURM_NNODES --nproc-per-node=8 \
        --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:29500 \
        --rdzv-id=$SLURM_JOB_ID demo_scripts/nanogpt_ddp.py ...

nanoGPT's ``model.py`` is imported unmodified from ``--nanogpt-path`` (clone
https://github.com/karpathy/nanoGPT). Data is nanoGPT's ``prepare.py`` output
(train.bin/val.bin uint16 token files). The training step is nanoGPT's:
forward, backward, clip_grad_norm_, AdamW, zero_grad — under bf16 autocast.
"""

import os

# One GPU per torchrun worker, decided before anything initializes the GPU
# runtime or MAX (slices a SLURM-style "0,1,...,7" CUDA_VISIBLE_DEVICES —
# ROCR_VISIBLE_DEVICES on AMD — by LOCAL_RANK). Optional so --device cuda
# works as a baseline in a stock-torch venv (CUDA or ROCm, which also calls
# its devices "cuda").
try:
    from torch_mojo_backend.distributed import use_local_rank_gpu

    use_local_rank_gpu()
except ImportError:
    pass

import argparse
import datetime
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nanogpt-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="mojo", choices=["mojo", "cuda"])
    parser.add_argument("--batch-size", type=int, default=12, help="per-rank")
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--vocab-size", type=int, default=50304)
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=0, help="0 = off")
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        len(data) - block_size - 1, (batch_size,), generator=generator
    )
    x = torch.stack(
        [torch.from_numpy(data[s : s + block_size].astype(np.int64)) for s in starts]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[s + 1 : s + 1 + block_size].astype(np.int64))
            for s in starts
        ]
    )
    return x.to(device), y.to(device)


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    device = args.device

    if device == "mojo":
        from torch_mojo_backend import (  # noqa: PLC0415 -- optional, like the import at the top: --device cuda must run in a stock-torch venv
            register_mojo_devices,
        )

        register_mojo_devices()
        backend = "mojo"
    else:
        backend = "nccl"
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=10))

    def log(msg: str):
        if rank == 0:
            print(msg, flush=True)

    log(f"world={world} device={device} dtype={args.dtype}")

    # Identical model init everywhere (DDP re-broadcasts rank 0 anyway);
    # rank-offset seed for on-device randomness (dropout).
    torch.manual_seed(args.seed)
    if device == "mojo":
        from torch_mojo_backend.mojo_device import (  # noqa: PLC0415 -- same optional import as above
            torch_mojo_device_module,
        )

        torch_mojo_device_module.manual_seed_all(args.seed + rank)
    else:
        torch.cuda.manual_seed_all(args.seed + rank)

    sys.path.insert(0, str(args.nanogpt_path))
    from model import (  # ty: ignore[unresolved-import] -- nanoGPT, from --nanogpt-path  # noqa: PLC0415 -- reachable only via the sys.path insert above
        GPT,
        GPTConfig,
    )

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )
    model = GPT(config).to(device)
    model.train()
    ddp_model = DDP(model, broadcast_buffers=False)

    # nanoGPT's own optimizer config; "cuda" selects the fused AdamW path,
    # which the mojo device implements (aten::_fused_adamw_).
    optimizer = model.configure_optimizers(
        args.weight_decay, args.lr, (0.9, 0.95), "cuda"
    )
    for group in optimizer.param_groups:
        group["fused"] = True

    train_data = np.memmap(args.data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(args.data_dir / "val.bin", dtype=np.uint16, mode="r")
    generator = torch.Generator().manual_seed(args.seed * 1000 + rank)

    autocast = (
        torch.amp.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    def estimate_loss(data: np.memmap) -> float:
        model.eval()
        total = 0.0
        with torch.no_grad():
            for _ in range(args.eval_iters):
                x, y = get_batch(
                    data, args.batch_size, args.block_size, device, generator
                )
                with autocast:
                    _, loss = model(x, y)
                total += loss.item()
        model.train()
        # On the device so the NCCL cuda baseline can reduce it too.
        mean = torch.tensor([total / args.eval_iters]).to(device)
        dist.all_reduce(mean, op=dist.ReduceOp.AVG)
        return mean.item()

    tokens_per_step = world * args.batch_size * args.block_size
    t_start = time.perf_counter()
    t_window = t_start
    for step in range(1, args.max_iters + 1):
        x, y = get_batch(
            train_data, args.batch_size, args.block_size, device, generator
        )
        with autocast:
            _, loss = ddp_model(x, y)
        loss.backward()
        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            avg = loss.detach().float().clone()
            dist.all_reduce(avg, op=dist.ReduceOp.AVG)
            torch.accelerator.synchronize()
            now = time.perf_counter()
            tps = tokens_per_step * args.log_interval / (now - t_window)
            t_window = now
            log(
                f"step {step:5d} | loss {avg.item():.4f} | "
                f"{tps / 1e3:8.1f}k tok/s | {now - t_start:7.1f}s"
            )
        if args.eval_interval and step % args.eval_interval == 0:
            log(f"step {step:5d} | val loss {estimate_loss(val_data):.4f}")

    torch.accelerator.synchronize()
    total_s = time.perf_counter() - t_start
    log(
        f"done: {args.max_iters} steps, {args.max_iters * tokens_per_step / 1e6:.1f}M "
        f"tokens, avg {args.max_iters * tokens_per_step / total_s / 1e3:.1f}k tok/s"
    )
    if args.out_dir is not None and rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(state, args.out_dir / "ckpt.pt")
        log(f"saved checkpoint to {args.out_dir / 'ckpt.pt'}")
    dist.barrier()
    dist.destroy_process_group()
    vmm = os.environ.get("MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_VMM", "").lower()
    if device == "mojo" and vmm in ("1", "true", "yes", "on"):
        # MAX's on-demand (VMM) device allocator is what makes an APU such as
        # the MI300A usable with one rank per GPU (docs/distributed.md), but
        # with MAX 26.5 + ROCm 6.4.3 the HSA runtime segfaults in its atexit
        # teardown of the VMM mappings, after Python has finished. Everything
        # above ran and was checkpointed; skip the C exit handlers.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
