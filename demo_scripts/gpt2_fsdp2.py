"""Train GPT-2 124M or XL with FSDP2 on the mojo or CUDA device.

    TORCH_MOJO_BACKEND_CCL=mojo uv run torchrun --standalone \
        --nproc-per-node=2 demo_scripts/gpt2_fsdp2.py --model gpt2

Use --model gpt2-xl for the 1.5B model. Models are initialized from scratch
with the standard GPT-2 architecture; no weights or dataset are downloaded.
A fixed, rank-specific synthetic token batch makes this a reproducible
forward/backward/AdamW smoke test, not a language-model quality evaluation.

Add --benchmark --warmup 5 --steps 10 --windows 3 to measure aggregate
training tokens/second after warmup. --device cuda uses stock PyTorch/NCCL;
--device mojo uses TORCH_MOJO_BACKEND_CCL=vendor (NCCL) or mojo (MojoCCL).
The same precision policy and unfused AdamW apply to both devices.
"""

# ruff: noqa: E402 -- pin the GPU before torch/MAX initialization
from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import argparse
import cProfile
import datetime
import json
import math
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from transformers import GPT2Config, GPT2LMHeadModel

from torch_mojo_backend import register_mojo_devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gpt2", "gpt2-xl"], default="gpt2")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["mojo", "cuda"], default="mojo")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument(
        "--nsys", action="store_true", help="capture timed windows with Nsight Systems"
    )
    parser.add_argument(
        "--profile", type=Path, help="write per-rank traces after timing"
    )
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or not 2 <= args.sequence_length <= 1024:
        parser.error(
            "steps/batch-size must be positive and sequence-length must be 2..1024"
        )
    if args.warmup < 1 or args.windows < 1:
        parser.error("warmup and windows must be positive")
    return args


def main():
    args = parse_args()
    if args.device == "mojo":
        register_mojo_devices()
    else:
        torch.cuda.set_device(0)  # use_local_rank_gpu exposes one GPU per rank
    dist.init_process_group(
        "mojo" if args.device == "mojo" else "nccl",
        timeout=datetime.timedelta(minutes=15),
    )
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(123)
    layers, heads, width = (12, 12, 768) if args.model == "gpt2" else (48, 25, 1600)
    config = GPT2Config()
    config.n_layer, config.n_head, config.n_embd = layers, heads, width
    config.resid_pdrop = config.embd_pdrop = config.attn_pdrop = 0.0
    config.use_cache = False
    model = GPT2LMHeadModel(config)
    torch.nn.Module.to(model, args.device)
    model.train()
    parameters = sum(p.numel() for p in model.parameters())
    mesh = init_device_mesh(args.device, (world,))
    policy = (
        MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        if args.dtype == "bfloat16"
        else MixedPrecisionPolicy()
    )
    for block in model.transformer.h:
        fully_shard(block, mesh=mesh, mp_policy=policy)
    # The root owns the tied embedding/head parameters. Keep them in fp32:
    # the device's embedding backward currently requires fp32 gradients.
    fully_shard(model, mesh=mesh)
    assert all(isinstance(p, DTensor) for p in model.parameters())
    local_parameters = sum(
        p.to_local().numel() for p in model.parameters() if isinstance(p, DTensor)
    )
    assert local_parameters < parameters or world == 1
    print(
        f"rank={rank} model={args.model} parameters={parameters} local_parameters={local_parameters}",
        flush=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    generator = torch.Generator().manual_seed(9000 + rank)
    tokens = torch.randint(
        config.vocab_size, (args.batch_size, args.sequence_length), generator=generator
    ).to(args.device)
    if args.benchmark:
        benchmark(args, model, optimizer, tokens, parameters)
        dist.destroy_process_group()
        return
    losses = []
    norms = []
    for step in range(args.steps):
        # The device's autocast policy keeps normalization in fp32 while
        # FSDP's policy controls parameter communication and gradient reduction.
        with torch.autocast(
            args.device, dtype=torch.bfloat16, enabled=args.dtype == "bfloat16"
        ):
            loss = model(tokens, labels=tokens).loss
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        global_norm = (
            norm.full_tensor().item() if isinstance(norm, DTensor) else norm.item()
        )
        assert math.isfinite(global_norm) and global_norm > 0, global_norm
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = loss.detach().float().clone()
        dist.all_reduce(mean_loss, op=dist.ReduceOp.AVG)
        value = mean_loss.item()
        assert math.isfinite(value), value
        losses.append(value)
        norms.append(global_norm)
        if rank == 0:
            print(
                f"step={step + 1} loss={value:.6f} grad_norm={global_norm:.6f}",
                flush=True,
            )
    if len(losses) > 1:
        assert losses[-1] < losses[0], losses
    assert all(isinstance(p, DTensor) for p in model.parameters())
    if rank == 0:
        result = dict(
            model=args.model,
            collectives=os.environ.get("TORCH_MOJO_BACKEND_CCL", "vendor"),
            parameters=parameters,
            world_size=world,
            dtype=args.dtype,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            losses=losses,
            gradient_norms=norms,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("FSDP2 training OK", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def benchmark(
    args: argparse.Namespace,
    model: GPT2LMHeadModel,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    parameters: int,
):
    """Measure complete training steps after JIT and optimizer-state warmup.

    Every window is synchronized; the slowest rank determines throughput.
    Loss checks and reporting use a CPU control group outside the timed region.
    Both devices use identical FSDP precision, clipping, and AdamW settings.
    """
    control = dist.new_group(backend="gloo")
    device_module = torch.get_device_module(args.device)

    def phase(name: str):
        if args.nsys:
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push(name)

    def step() -> torch.Tensor:
        if args.nsys:
            torch.cuda.nvtx.range_push("forward")
        with torch.autocast(
            args.device, dtype=torch.bfloat16, enabled=args.dtype == "bfloat16"
        ):
            loss = model(tokens, labels=tokens).loss
        phase("backward")
        loss.backward()
        phase("clip_grad_norm")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        phase("optimizer")
        optimizer.step()
        phase("zero_grad")
        optimizer.zero_grad(set_to_none=True)
        if args.nsys:
            torch.cuda.nvtx.range_pop()
        return loss.detach()

    warmup_losses = []
    for index in range(args.warmup):
        value = step().item()
        assert math.isfinite(value), value
        warmup_losses.append(value)
        if dist.get_rank() == 0:
            print(f"warmup={index + 1} loss={value:.6f}", flush=True)
    elapsed_windows = []
    losses = []
    if args.nsys:
        torch.cuda.profiler.start()
        torch.cuda.nvtx.range_push("fsdp2_timed_windows")
    for index in range(args.windows):
        device_module.synchronize()
        dist.barrier(group=control)
        start = time.perf_counter()
        for _ in range(args.steps):
            loss = step()
        device_module.synchronize()
        elapsed = torch.tensor(time.perf_counter() - start, dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=control)
        elapsed_windows.append(elapsed.item())
        value = loss.float().cpu()
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=control)
        value = value.item() / dist.get_world_size()
        assert math.isfinite(value), value
        losses.append(value)
        if dist.get_rank() == 0:
            print(
                f"window={index + 1} seconds={elapsed.item():.6f} loss={value:.6f}",
                flush=True,
            )
    if args.nsys:
        torch.cuda.nvtx.range_pop()
        torch.cuda.profiler.stop()
    token_count = (
        args.steps * args.batch_size * args.sequence_length * dist.get_world_size()
    )
    throughputs = [token_count / elapsed for elapsed in elapsed_windows]
    if dist.get_rank() == 0:
        result = dict(
            model=args.model,
            device=args.device,
            collectives=(
                os.environ.get("TORCH_MOJO_BACKEND_CCL", "vendor")
                if args.device == "mojo"
                else "nccl"
            ),
            torch_version=torch.__version__,
            attention_implementation=model.config._attn_implementation,
            parameters=parameters,
            world_size=dist.get_world_size(),
            dtype=args.dtype,
            root_parameter_dtype="float32",
            reduction_dtype="float32",
            sequence_length=args.sequence_length,
            batch_size_per_rank=args.batch_size,
            warmup_steps=args.warmup,
            steps_per_window=args.steps,
            window_seconds=elapsed_windows,
            tokens_per_second=throughputs,
            median_tokens_per_second=statistics.median(throughputs),
            warmup_losses_rank0=warmup_losses,
            final_losses=losses,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    if args.profile:
        profile_step(args.profile, step, device_module.synchronize)
    dist.destroy_process_group(control)


def profile_step(
    directory: Path, step: Callable[[], torch.Tensor], synchronize: Callable[[], None]
):
    """Capture diagnostics separately so profiler overhead cannot affect timing."""
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / f"rank{dist.get_rank()}"
    synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        step()
        synchronize()
    prof.export_chrome_trace(str(prefix) + ".json")
    averages = prof.key_averages(group_by_input_shape=True)
    for kind in ("cpu", "device"):
        Path(str(prefix) + f"_{kind}.txt").write_text(
            averages.table(sort_by=f"self_{kind}_time_total", row_limit=80)
        )
    cpu = cProfile.Profile()
    cpu.runcall(step)
    synchronize()
    cpu.dump_stats(str(prefix) + ".pstats")


if __name__ == "__main__":
    main()
