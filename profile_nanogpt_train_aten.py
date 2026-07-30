"""Profile one nanoGPT training step by ATen operator, per backend.

Reuses the frozen workload from ``bench_nanogpt_train.py``: nanoGPT's GPT-2
124M, batch 48, block 1024, BF16 autocast, fused AdamW, gradient clipping at
1.0, eager execution.  Emits

* ``aten_gpu_time_step.csv`` / ``aten_gpu_time_step_by_shape.csv`` -- self GPU
  time per ATen operator (and per operator + input shape) for the timed steps,
* ``trace_step.json`` -- a chrome trace,
* ``nanogpt_train_profile_summary.json`` -- environment, workload, wall-clock
  step time, GPU idle statistics and a forward/backward/clip/optimizer split
  measured by synchronized subtraction outside the profiler.

Runs on either vendor.  ``--device cuda`` means the vendor's own backend, which
is ROCm on a HIP build and CUDA on a CUDA one; PyTorch names the profiler
activity and columns "CUDA" in both cases.  Both CPU and CUDA activities are
required so GPU kernels correlate back to their ATen ranges.

Usage:
    uv run --no-sync python profile_nanogpt_train_aten.py --device cuda \
        --output-dir current_bench_train/vendor
    uv run --no-sync python profile_nanogpt_train_aten.py --device mojo \
        --output-dir current_bench_train/mojo
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import torch
from tabulate import tabulate
from torch.profiler import ProfilerActivity, profile

from bench_nanogpt_train import (
    DEFAULT_NANOGPT_PATH,
    DTYPES,
    GRAD_CLIP,
    autocast_context,
    build_model,
    cycle,
    make_batches,
    make_synchronize,
    training_step,
)


def self_gpu_us(event: object) -> float:
    """Return self GPU time in microseconds across profiler API versions."""
    value = getattr(event, "self_device_time_total", None)
    if value is not None:
        return float(value)
    return float(getattr(event, "self_cuda_time_total", 0.0))


def phase_events(profiler: profile, group_by_input_shape: bool = False) -> list[object]:
    averages = profiler.key_averages(group_by_input_shape=group_by_input_shape)
    return [
        event
        for event in averages
        if event.key.startswith("aten::") and self_gpu_us(event) > 0
    ]


def gpu_idle_stats(profiler: profile) -> dict[str, float]:
    """Measure gaps in the union of GPU activities visible in the trace."""
    intervals = sorted(
        (event.time_range.start, event.time_range.end)
        for event in profiler.events()
        if str(event.device_type) in {"DeviceType.CUDA", "DeviceType.PrivateUse1"}
        and event.time_range.end > event.time_range.start
    )
    if not intervals:
        return {
            "gpu_activity_count": 0,
            "span_us": 0.0,
            "busy_us": 0.0,
            "idle_us": 0.0,
            "idle_pct": 0.0,
            "gap_count": 0,
            "gaps_over_50us": 0,
            "max_gap_us": 0.0,
        }

    merged = []
    current_start, current_end = intervals[0]
    gaps = []
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            gaps.append(start - current_end)
            current_start, current_end = start, end
    merged.append((current_start, current_end))

    span = merged[-1][1] - merged[0][0]
    busy = sum(end - start for start, end in merged)
    return {
        "gpu_activity_count": len(intervals),
        "span_us": span,
        "busy_us": busy,
        "idle_us": max(span - busy, 0.0),
        "idle_pct": 100.0 * max(span - busy, 0.0) / span if span else 0.0,
        "gap_count": len(gaps),
        "gaps_over_50us": sum(gap >= 50.0 for gap in gaps),
        "max_gap_us": max(gaps, default=0.0),
    }


def render_op_table(events: list[object], total_us: float, row_limit: int) -> str:
    rows = []
    for rank, event in enumerate(
        sorted(events, key=self_gpu_us, reverse=True)[:row_limit], start=1
    ):
        event_us = self_gpu_us(event)
        rows.append(
            [
                rank,
                event.key,
                event.count,
                f"{event_us / 1000:.3f}",
                f"{100 * event_us / total_us:.2f}",
                f"{event_us / max(event.count, 1):.2f}",
            ]
        )
    return tabulate(
        rows,
        headers=["rank", "ATen op", "calls", "self GPU ms", "step %", "avg us/call"],
        tablefmt="simple",
        disable_numparse=True,
    )


def render_shape_table(events: list[object], total_us: float, row_limit: int) -> str:
    rows = []
    for rank, event in enumerate(
        sorted(events, key=self_gpu_us, reverse=True)[:row_limit], start=1
    ):
        event_us = self_gpu_us(event)
        rows.append(
            [
                rank,
                event.key,
                repr(event.input_shapes),
                event.count,
                f"{event_us / 1000:.3f}",
                f"{100 * event_us / total_us:.2f}",
                f"{event_us / max(event.count, 1):.2f}",
            ]
        )
    return tabulate(
        rows,
        headers=[
            "rank",
            "ATen op",
            "input shapes",
            "calls",
            "self GPU ms",
            "step %",
            "avg us/call",
        ],
        tablefmt="simple",
        disable_numparse=True,
    )


def write_op_csv(path: Path, events: list[object], total_us: float, steps: int) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "aten_op",
                "calls",
                "calls_per_step",
                "self_gpu_time_us",
                "self_gpu_us_per_step",
                "self_gpu_pct",
                "avg_us_per_call",
            ]
        )
        for event in sorted(events, key=self_gpu_us, reverse=True):
            event_us = self_gpu_us(event)
            writer.writerow(
                [
                    event.key,
                    event.count,
                    f"{event.count / steps:.2f}",
                    f"{event_us:.1f}",
                    f"{event_us / steps:.1f}",
                    f"{100 * event_us / total_us:.2f}",
                    f"{event_us / max(event.count, 1):.2f}",
                ]
            )


def write_shape_csv(
    path: Path, events: list[object], total_us: float, steps: int
) -> None:
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "aten_op",
                "input_shapes",
                "calls",
                "calls_per_step",
                "self_gpu_time_us",
                "self_gpu_us_per_step",
                "self_gpu_pct",
                "avg_us_per_call",
            ]
        )
        for event in sorted(events, key=self_gpu_us, reverse=True):
            event_us = self_gpu_us(event)
            writer.writerow(
                [
                    event.key,
                    repr(event.input_shapes),
                    event.count,
                    f"{event.count / steps:.2f}",
                    f"{event_us:.1f}",
                    f"{event_us / steps:.1f}",
                    f"{100 * event_us / total_us:.2f}",
                    f"{event_us / max(event.count, 1):.2f}",
                ]
            )


def measure_phase_split(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    context: AbstractContextManager[None],
    synchronize: Callable[[], None],
    grad_clip: float,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """Time cumulative prefixes of the step and report per-phase medians.

    Each prefix is measured with its own synchronized loop, so the phase
    numbers are wall-clock and free of profiler overhead.  Backward, clip and
    optimizer times come from differences between adjacent prefixes.
    """

    def forward_only(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with context:
            _, loss = model(inputs, targets)
        return loss

    def through_backward(inputs: torch.Tensor, targets: torch.Tensor) -> None:
        forward_only(inputs, targets).backward()

    def through_clip(inputs: torch.Tensor, targets: torch.Tensor) -> None:
        through_backward(inputs, targets)
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    def full_step(inputs: torch.Tensor, targets: torch.Tensor) -> None:
        training_step(model, optimizer, inputs, targets, context, grad_clip)

    def time_prefix(body: Callable[[torch.Tensor, torch.Tensor], object]) -> float:
        stream = cycle(batches)
        for _ in range(warmup):
            body(*next(stream))
            optimizer.zero_grad(set_to_none=True)
        synchronize()
        samples = []
        for _ in range(iters):
            inputs, targets = next(stream)
            start = time.perf_counter()
            body(inputs, targets)
            synchronize()
            samples.append((time.perf_counter() - start) * 1e3)
            optimizer.zero_grad(set_to_none=True)
        return statistics.median(samples)

    forward_ms = time_prefix(forward_only)
    backward_ms = time_prefix(through_backward)
    clip_ms = time_prefix(through_clip)
    step_ms = time_prefix(full_step)
    return {
        "forward_ms": forward_ms,
        "forward_backward_ms": backward_ms,
        "through_clip_ms": clip_ms,
        "full_step_ms": step_ms,
        "backward_ms": backward_ms - forward_ms,
        "clip_grad_norm_ms": clip_ms - backward_ms,
        "optimizer_ms": step_ms - clip_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", choices=["cuda", "mojo"])
    parser.add_argument("--nanogpt-path", type=Path, default=DEFAULT_NANOGPT_PATH)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bias", action="store_true", default=False)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16")
    parser.add_argument("--grad-clip", type=float, default=GRAD_CLIP)
    parser.add_argument("--fused-adamw", choices=("auto", "on", "off"), default="on")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profiled-steps", type=int, default=3)
    parser.add_argument("--timed-iters", type=int, default=10)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--row-limit", type=int, default=45)
    parser.add_argument("--skip-phase-split", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the GPU")
    hardware_name = torch.cuda.get_device_name(0)
    # `--device cuda` is the vendor's own backend, whichever vendor that is:
    # a ROCm build answers to the same `torch.cuda` API as a CUDA one, so the
    # only difference is what to call it in the report.
    vendor = "ROCm" if torch.version.hip is not None else "CUDA"

    if args.device == "cuda":
        execution_backend = f"PyTorch {vendor}"
    else:
        from torch_mojo_backend import get_accelerators, register_mojo_devices

        register_mojo_devices()
        max_device = list(get_accelerators())[0]
        if "gpu" not in str(max_device).lower():
            raise RuntimeError(f"Expected MAX accelerator 0 to be a GPU: {max_device}")
        execution_backend = f"Mojo/MAX ({max_device})"

    synchronize = make_synchronize(args.device)
    dtype = DTYPES[args.dtype]
    context = autocast_context(args.device, dtype)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Environment: "
        f"torch={torch.__version__}, "
        f"{vendor.lower()}={torch.version.hip or torch.version.cuda}, "
        f"hardware={hardware_name}, execution_backend={execution_backend}"
    )
    model, optimizer = build_model(args.nanogpt_path, args, args.device)
    batches = make_batches(
        args.batch_size, args.block_size, args.device, args.batches, args.seed
    )
    stream = cycle(batches)
    print(
        f"Workload: backend={execution_backend}, batch={args.batch_size}, "
        f"block={args.block_size}, layers={args.n_layer}, heads={args.n_head}, "
        f"embd={args.n_embd}, bias={args.bias}, dtype={args.dtype}, "
        f"fused_adamw={args.fused_adamw}, grad_clip={args.grad_clip}"
    )

    for _ in range(args.warmup):
        training_step(model, optimizer, *next(stream), context, args.grad_clip)
    synchronize()

    samples = []
    for _ in range(args.timed_iters):
        inputs, targets = next(stream)
        start = time.perf_counter()
        training_step(model, optimizer, inputs, targets, context, args.grad_clip)
        synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    step_ms = statistics.median(samples)
    tokens = args.batch_size * args.block_size
    print(
        f"Unprofiled step: median {step_ms:.3f} ms, "
        f"{tokens / (step_ms / 1e3):.1f} tok/s"
    )

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(activities=activities, record_shapes=True) as prof:
        for _ in range(args.profiled_steps):
            training_step(model, optimizer, *next(stream), context, args.grad_clip)
        synchronize()

    by_op = phase_events(prof)
    by_shape = phase_events(prof, group_by_input_shape=True)
    # Self times on ATen rows partition attributed GPU kernel time.  Raw GPU
    # kernel rows in key_averages duplicate that attribution and are therefore
    # deliberately excluded from the denominator.
    total_us = sum(self_gpu_us(event) for event in by_op)
    if total_us <= 0:
        raise RuntimeError("No GPU time was attributed to ATen ops")

    print("\n================ training step: by ATen op ================")
    print(render_op_table(by_op, total_us, args.row_limit))
    print("\n================ training step: by ATen op + input shape ================")
    print(render_shape_table(by_shape, total_us, args.row_limit))

    write_op_csv(
        args.output_dir / "aten_gpu_time_step.csv", by_op, total_us, args.profiled_steps
    )
    write_shape_csv(
        args.output_dir / "aten_gpu_time_step_by_shape.csv",
        by_shape,
        total_us,
        args.profiled_steps,
    )
    prof.export_chrome_trace(str(args.output_dir / "trace_step.json"))
    idle = gpu_idle_stats(prof)
    print(
        f"\nATen-attributed self GPU time = {total_us / 1000:.3f} ms over "
        f"{args.profiled_steps} steps ({total_us / 1000 / args.profiled_steps:.3f} "
        "ms/step)"
    )
    print(
        f"trace GPU span = {idle['span_us'] / 1000:.3f} ms, "
        f"union busy = {idle['busy_us'] / 1000:.3f} ms, "
        f"idle = {idle['idle_pct']:.2f}% "
        f"({idle['gaps_over_50us']} gaps >= 50 us, max {idle['max_gap_us']:.1f} us)"
    )

    phases = {}
    if not args.skip_phase_split:
        phases = measure_phase_split(
            model,
            optimizer,
            batches,
            context,
            synchronize,
            args.grad_clip,
            warmup=3,
            iters=args.timed_iters,
        )
        print("\n================ phase split (wall clock) ================")
        print(
            tabulate(
                [
                    ["forward", f"{phases['forward_ms']:.3f}"],
                    ["backward", f"{phases['backward_ms']:.3f}"],
                    ["clip_grad_norm_", f"{phases['clip_grad_norm_ms']:.3f}"],
                    ["optimizer.step", f"{phases['optimizer_ms']:.3f}"],
                    ["full step", f"{phases['full_step_ms']:.3f}"],
                ],
                headers=["phase", "ms"],
                tablefmt="simple",
                disable_numparse=True,
            )
        )

    summary = {
        "environment": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "cuda": torch.version.cuda,
            "hardware": hardware_name,
            "execution_backend": execution_backend,
            "torch_device": args.device,
        },
        "workload": {
            "model": "nanoGPT GPT-2 124M",
            "batch_size": args.batch_size,
            "block_size": args.block_size,
            "n_layer": args.n_layer,
            "n_head": args.n_head,
            "n_embd": args.n_embd,
            "bias": args.bias,
            "dtype": args.dtype,
            "fused_adamw": args.fused_adamw,
            "grad_clip": args.grad_clip,
            "tokens_per_step": tokens,
        },
        "unprofiled": {
            "step_ms_median": step_ms,
            "tokens_per_second": tokens / (step_ms / 1e3),
        },
        "profile": {
            "profiled_steps": args.profiled_steps,
            "self_gpu_us_total": total_us,
            "self_gpu_us_per_step": total_us / args.profiled_steps,
            "gpu_trace": idle,
        },
        "phases_ms": phases,
    }
    with (args.output_dir / "nanogpt_train_profile_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()
