"""Per-op device-time profile of a GPT-2 XL training step (forward + backward).

Runs nanoGPT's GPT-2 XL (48 layers, 25 heads, 1600 embd, block 1024) for a few
real training steps (bf16 autocast, fused AdamW) to warm every kernel and
allocate optimizer state, then profiles K iterations of forward + backward with
the legacy autograd profiler, which reports device time per op on both backends
(torch's CUDA events on "cuda", the shim's ProfilerStubs on "mojo" — see
agents_docs/native_backend.md, "Profiling"). The optimizer runs only during warmup so
its ops stay out of the profile; zero_grad(set_to_none=True) between profiled
steps keeps each backward identical to a fresh training step's.

Usage:
    flock /tmp/gpu_lock_0.lock uv run --no-sync python \
        current_bench/profile_gpt2xl_ops.py --device cuda --json-out /tmp/cuda.json
    flock /tmp/gpu_lock_0.lock uv run --no-sync python \
        current_bench/profile_gpt2xl_ops.py --device mojo --json-out /tmp/mojo.json

    # memory/fit probe only (exit 0 iff the batch runs full training steps)
    flock /tmp/gpu_lock_0.lock uv run --no-sync python \
        current_bench/profile_gpt2xl_ops.py --device cuda --batch-size 16 --probe

Compare the two JSONs with compare_op_profiles.py.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from bench_nanogpt_train import (
    autocast_context,
    build_model,
    cycle,
    make_batches,
    make_synchronize,
    training_step,
)

from torch_mojo_backend import register_mojo_devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nanogpt-path", type=Path, default=Path("/root/nanoGPT"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=1024)
    # GPT-2 XL
    parser.add_argument("--n-layer", type=int, default=48)
    parser.add_argument("--n-head", type=int, default=25)
    parser.add_argument("--n-embd", type=int, default=1600)
    parser.add_argument("--dropout", type=float, default=0.0)
    # Real GPT-2 XL has biases (nanoGPT's from-scratch default is bias=False).
    parser.add_argument("--bias", action="store_true", default=True)
    parser.add_argument("--no-bias", dest="bias", action="store_false")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--fused-adamw", choices=("auto", "on", "off"), default="on")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="only check the batch size fits: run warmup full steps, report, exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.device.startswith("mojo"):
        register_mojo_devices()

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    synchronize = make_synchronize(args.device)
    context = autocast_context(args.device, dtype)

    model, optimizer = build_model(args.nanogpt_path, args, args.device)
    batches = make_batches(args.batch_size, args.block_size, args.device, 2, args.seed)
    stream = cycle(batches)

    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # Warmup: full training steps (fwd, bwd, clip, fused AdamW, zero_grad) so
    # every kernel is built/warm and optimizer state is allocated — the memory
    # footprint during profiling is the real training footprint.
    for _ in range(args.warmup):
        inputs, targets = next(stream)
        training_step(model, optimizer, inputs, targets, context, 1.0)
    synchronize()

    if args.device.startswith("cuda"):
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        print(f"peak allocated after warmup full steps: {peak_gib:.2f} GiB")

    def fwd_bwd():
        inputs, targets = next(stream)
        with context:
            _, loss = model(inputs, targets)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)

    # Timed (unprofiled) fwd+bwd steps for a step-time reference.
    samples = []
    for _ in range(args.timed_steps):
        start = time.perf_counter()
        fwd_bwd()
        synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    step_ms = statistics.median(samples)
    print(f"fwd+bwd median: {step_ms:.1f} ms over {args.timed_steps} steps")

    if args.probe:
        print("PROBE_OK")
        return

    with torch.autograd.profiler.profile(
        use_device=torch.device(args.device).type, record_shapes=True
    ) as prof:
        for _ in range(args.profile_steps):
            fwd_bwd()
        synchronize()

    rows = []
    for e in prof.key_averages(group_by_input_shape=True):
        rows.append(
            {
                "key": e.key,
                "input_shapes": str(e.input_shapes),
                "count": e.count,
                "self_device_us": e.self_device_time_total,
                "device_us": e.device_time_total,
                "self_cpu_us": e.self_cpu_time_total,
                "cpu_us": e.cpu_time_total,
            }
        )

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
        "profile_steps": args.profile_steps,
        "fwd_bwd_ms_median": step_ms,
        "rows": rows,
    }
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=25))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
