"""Time one nanoGPT training step on PyTorch-ROCm and on the eager Mojo device.

The workload is frozen: nanoGPT's GPT-2 124M configuration (12 layers, 12
heads, 768 embedding, ``bias=False``, ``vocab_size=50304``), batch 48, block
1024, BF16 autocast, AdamW, gradient clipping at 1.0, no ``torch.compile`` and
no gradient accumulation.  One step is forward, backward, ``clip_grad_norm_``,
``optimizer.step()`` and ``zero_grad(set_to_none=True)``, exactly the body of
nanoGPT's ``train.py`` loop.

nanoGPT's own ``model.py`` is imported unmodified from ``--nanogpt-path``; the
only harness-side substitution is synthetic token data, so no dataset download
is needed and both backends see identical inputs.

Usage:
    uv run --no-sync python bench_nanogpt_train.py --device cuda
    uv run --no-sync python bench_nanogpt_train.py --device mojo
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import torch

DEFAULT_NANOGPT_PATH = Path("/root/nanoGPT")

# nanoGPT train.py defaults for the 124M single-GPU run.
WEIGHT_DECAY = 1e-1
LEARNING_RATE = 6e-4
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
VOCAB_SIZE = 50304

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def import_nanogpt(nanogpt_path: Path) -> tuple[type, type]:
    """Import ``GPTConfig`` and ``GPT`` from an unmodified nanoGPT checkout."""
    if not (nanogpt_path / "model.py").is_file():
        raise FileNotFoundError(
            f"{nanogpt_path}/model.py not found; clone "
            "https://github.com/karpathy/nanoGPT and pass --nanogpt-path"
        )
    if str(nanogpt_path) not in sys.path:
        sys.path.insert(0, str(nanogpt_path))
    from model import GPT, GPTConfig

    return GPTConfig, GPT


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def make_synchronize(device: str) -> Callable[[], None]:
    """Return a backend-appropriate full-queue drain for the timed device."""
    if device.startswith("mojo"):
        return lambda: torch.accelerator.synchronize()
    if device.startswith("cuda"):
        return lambda: torch.cuda.synchronize()
    return lambda: None


def autocast_context(device: str, dtype: torch.dtype) -> AbstractContextManager[None]:
    """BF16 autocast on the timed device, or no autocast for float32 runs."""
    if dtype is torch.float32:
        return nullcontext()
    device_type = torch.device(device).type
    return torch.amp.autocast(device_type=device_type, dtype=dtype)


def make_batches(
    batch_size: int, block_size: int, device: str, count: int, seed: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Pre-stage identical synthetic ``(inputs, targets)`` pairs on the device.

    Data loading is not part of the measured step, so the batches are created
    once on the CPU with a fixed seed and copied to the device.  Both backends
    therefore consume bit-identical token ids.
    """
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(count):
        tokens = torch.randint(
            0, VOCAB_SIZE, (batch_size, block_size + 1), generator=generator
        )
        inputs = tokens[:, :-1].contiguous().to(device)
        targets = tokens[:, 1:].contiguous().to(device)
        batches.append((inputs, targets))
    return batches


def cycle(
    items: list[tuple[torch.Tensor, torch.Tensor]],
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield from items


def build_model(
    nanogpt_path: Path, args: argparse.Namespace, device: str
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Instantiate nanoGPT's GPT and its own configured AdamW on the device."""
    GPTConfig, GPT = import_nanogpt(nanogpt_path)
    torch.manual_seed(args.seed)
    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=VOCAB_SIZE,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )
    model = GPT(config)
    model.to(device)
    model.train()

    # nanoGPT selects fused AdamW only for device_type == "cuda".  The flag
    # lets both backends be measured with the same optimizer implementation.
    if args.fused_adamw == "auto":
        optimizer_device_type = torch.device(device).type
    elif args.fused_adamw == "on":
        optimizer_device_type = "cuda"
    else:
        optimizer_device_type = "cpu"
    optimizer = model.configure_optimizers(
        WEIGHT_DECAY, LEARNING_RATE, BETAS, optimizer_device_type
    )
    if args.fused_adamw == "on":
        # configure_optimizers only inspects the string; make sure the groups
        # really are fused for a non-cuda timed device.
        for group in optimizer.param_groups:
            group["fused"] = True
    return model, optimizer


def training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    context: AbstractContextManager[None],
    grad_clip: float,
) -> torch.Tensor:
    """The body of nanoGPT's train.py iteration, without gradient accumulation."""
    with context:
        _, loss = model(inputs, targets)
    loss.backward()
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nanogpt-path", type=Path, default=DEFAULT_NANOGPT_PATH)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--fused-adamw", choices=("auto", "on", "off"), default="on")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--print-loss",
        action="store_true",
        help="report the final loss (adds a device-to-host sync outside timing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("mojo"):
        from torch_mojo_backend import register_mojo_devices

        register_mojo_devices()

    dtype = DTYPES[args.dtype]
    synchronize = make_synchronize(args.device)
    context = autocast_context(args.device, dtype)

    model, optimizer = build_model(args.nanogpt_path, args, args.device)
    batches = make_batches(
        args.batch_size, args.block_size, args.device, args.batches, args.seed
    )
    stream = cycle(batches)

    for _ in range(args.warmup):
        inputs, targets = next(stream)
        training_step(model, optimizer, inputs, targets, context, args.grad_clip)
    synchronize()

    samples = []
    loss = None
    for _ in range(args.iters):
        inputs, targets = next(stream)
        start = time.perf_counter()
        loss = training_step(model, optimizer, inputs, targets, context, args.grad_clip)
        synchronize()
        samples.append((time.perf_counter() - start) * 1e3)

    median = statistics.median(samples)
    tokens = args.batch_size * args.block_size
    result = {
        "device": args.device,
        "torch": torch.__version__,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "bias": args.bias,
        "fused_adamw": args.fused_adamw,
        "iters": args.iters,
        "step_ms_median": median,
        "step_ms_p10": percentile(samples, 0.10),
        "step_ms_p90": percentile(samples, 0.90),
        "step_ms_min": min(samples),
        "tokens_per_step": tokens,
        "tokens_per_second": tokens / (median / 1e3),
    }
    if args.print_loss and loss is not None:
        result["final_loss"] = float(loss.detach().cpu())

    print(json.dumps(result, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
