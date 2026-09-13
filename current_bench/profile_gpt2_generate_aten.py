"""Profile the real batched GPT-2 eager-generation path by ATen operator.

This instruments the same Hugging Face ``model.generate`` path, prompt,
greedy sampling, and dynamic KV cache used by ``bench_gpt2_batch.py``.  The
first generation iteration is prompt prefill; generation then resumes from
the returned DynamicCache for the remaining cached-decode iterations.

On ROCm, PyTorch continues to call the profiler activity and table columns
"CUDA".  Both CPU and CUDA activities are required for GPU kernel time to be
correlated back to the CPU-side ATen ranges.

Usage:
    uv run --no-sync python profile_gpt2_generate_aten.py --device cuda
    uv run --no-sync python profile_gpt2_generate_aten.py \
        --device mojo --output-dir mojo_profile
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import transformers
from tabulate import tabulate
from torch.profiler import ProfilerActivity, profile
from transformers import AutoModelForCausalLM, AutoTokenizer

from torch_mojo_backend import get_accelerators, register_mojo_devices
from torch_mojo_backend.native import device_module

PROMPT = "Here is how quantum computing works: "


def self_gpu_us(event) -> float:
    """Return self GPU time in microseconds across profiler API versions."""
    value = getattr(event, "self_device_time_total", None)
    if value is not None:
        return float(value)
    return float(getattr(event, "self_cuda_time_total", 0.0))


def make_prompt(
    tokenizer, batch_size: int, prompt_len: int, device: str
) -> torch.Tensor:
    """Create the benchmark's repeated fixed prompt at the requested length."""
    ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    if prompt_len != ids.shape[1]:
        repeats = math.ceil(prompt_len / ids.shape[1])
        ids = ids.repeat(1, repeats)[:, :prompt_len]
    return ids.repeat(batch_size, 1).to(device)


def generation_kwargs(tokenizer, new_tokens: int) -> dict:
    """The generation settings from bench_gpt2_batch.py."""
    return {
        "max_new_tokens": new_tokens,
        "min_new_tokens": new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }


def generate_full(model, prompt_ids, tokenizer, new_tokens: int):
    return model.generate(prompt_ids, **generation_kwargs(tokenizer, new_tokens))


def profile_prefill(model, prompt_ids, tokenizer, activities, synchronize):
    """Profile HF generate's first iteration and return its live KV cache."""
    kwargs = generation_kwargs(tokenizer, 1)
    kwargs["return_dict_in_generate"] = True
    with profile(activities=activities, record_shapes=True) as prof:
        output = model.generate(prompt_ids, **kwargs)
        synchronize()
    return prof, output.sequences, output.past_key_values


def profile_decode(
    model,
    input_ids,
    past_key_values,
    tokenizer,
    decode_tokens: int,
    activities,
    synchronize,
):
    """Resume HF generation and profile every remaining cached-decode step."""
    with profile(activities=activities, record_shapes=True) as prof:
        output = model.generate(
            input_ids,
            past_key_values=past_key_values,
            **generation_kwargs(tokenizer, decode_tokens),
        )
        synchronize()
    return prof, output


def phase_events(profiler, group_by_input_shape=False):
    averages = profiler.key_averages(group_by_input_shape=group_by_input_shape)
    return [
        event
        for event in averages
        if event.key.startswith("aten::") and self_gpu_us(event) > 0
    ]


def gpu_idle_stats(profiler) -> dict:
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
    idle = max(span - busy, 0.0)
    return {
        "gpu_activity_count": len(intervals),
        "span_us": span,
        "busy_us": busy,
        "idle_us": idle,
        "idle_pct": 100.0 * idle / span if span else 0.0,
        "gap_count": len(gaps),
        "gaps_over_50us": sum(gap >= 50.0 for gap in gaps),
        "max_gap_us": max(gaps, default=0.0),
    }


def render_op_table(events, total_us: float, row_limit: int) -> str:
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
        headers=["rank", "ATen op", "calls", "self GPU ms", "phase %", "avg us/call"],
        tablefmt="simple",
        disable_numparse=True,
    )


def render_shape_table(events, total_us: float, row_limit: int) -> str:
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
            "phase %",
            "avg us/call",
        ],
        tablefmt="simple",
        disable_numparse=True,
    )


def write_op_csv(path: Path, events, total_us: float):
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["aten_op", "calls", "self_gpu_time_us", "self_gpu_pct", "avg_us_per_call"]
        )
        for event in sorted(events, key=self_gpu_us, reverse=True):
            event_us = self_gpu_us(event)
            writer.writerow(
                [
                    event.key,
                    event.count,
                    f"{event_us:.1f}",
                    f"{100 * event_us / total_us:.2f}",
                    f"{event_us / max(event.count, 1):.2f}",
                ]
            )


def write_shape_csv(path: Path, events, total_us: float):
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "aten_op",
                "input_shapes",
                "calls",
                "self_gpu_time_us",
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
                    f"{event_us:.1f}",
                    f"{100 * event_us / total_us:.2f}",
                    f"{event_us / max(event.count, 1):.2f}",
                ]
            )


def dump_phase(profiler, tag: str, output_dir: Path, row_limit: int):
    by_op = phase_events(profiler)
    by_shape = phase_events(profiler, group_by_input_shape=True)

    # Self times on ATen rows partition attributed GPU kernel time.  Raw GPU
    # kernel rows in key_averages duplicate that attribution and are therefore
    # deliberately excluded from the denominator.
    total_us = sum(self_gpu_us(event) for event in by_op)
    if total_us <= 0:
        raise RuntimeError(f"No GPU time was attributed to ATen ops in {tag}")

    print(f"\n================ {tag}: by ATen op ================")
    print(render_op_table(by_op, total_us, row_limit))
    print(f"\n================ {tag}: by ATen op + input shape ================")
    print(render_shape_table(by_shape, total_us, row_limit))

    write_op_csv(output_dir / f"aten_gpu_time_{tag}.csv", by_op, total_us)
    write_shape_csv(
        output_dir / f"aten_gpu_time_{tag}_by_shape.csv", by_shape, total_us
    )
    profiler.export_chrome_trace(str(output_dir / f"trace_{tag}.json"))
    idle = gpu_idle_stats(profiler)

    print(f"{tag}: total ATen-attributed self GPU time = {total_us / 1000:.3f} ms")
    print(
        f"{tag}: trace GPU span = {idle['span_us'] / 1000:.3f} ms, "
        f"union busy = {idle['busy_us'] / 1000:.3f} ms, "
        f"idle = {idle['idle_pct']:.2f}% "
        f"({idle['gaps_over_50us']} gaps >= 50 us, "
        f"max {idle['max_gap_us']:.1f} us)"
    )
    return total_us, idle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda", choices=["cuda", "mojo"])
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=8,
        help="8 is the tokenized length of bench_gpt2_batch.py's fixed prompt",
    )
    parser.add_argument("--new-tokens", type=int, default=200)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--sampling", default="greedy", choices=["greedy"])
    parser.add_argument("--row-limit", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.new_tokens < 2:
        raise ValueError("--new-tokens must be at least 2 for a decode phase")
    if torch.version.hip is None:
        raise RuntimeError(
            f"A ROCm PyTorch build is required; found torch {torch.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the ROCm GPU")
    hardware_name = torch.cuda.get_device_name(0)
    if "MI300X" not in hardware_name:
        raise RuntimeError(f"Expected an MI300X, found {hardware_name}")

    if args.device == "cuda":
        execution_backend = "PyTorch ROCm"
        execution_device = "cuda"
        synchronize = torch.cuda.synchronize
    else:
        register_mojo_devices()
        max_device = list(get_accelerators())[0]
        if "gpu" not in str(max_device).lower():
            raise RuntimeError(f"Expected MAX accelerator 0 to be a GPU: {max_device}")

        def synchronize():
            device_module.synchronize(0)

        execution_backend = f"Mojo/MAX ({max_device})"
        execution_device = "mojo"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        args.dtype
    ]

    print(
        "Environment: "
        f"torch={torch.__version__}, hip={torch.version.hip}, "
        f"cuda_available={torch.cuda.is_available()}, hardware={hardware_name}, "
        f"execution_backend={execution_backend}, transformers={transformers.__version__}"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = (
        AutoModelForCausalLM.from_pretrained(args.model)
        .eval()
        .to(execution_device, dtype=dtype)
    )
    prompt_ids = make_prompt(
        tokenizer, args.batch_size, args.prompt_len, execution_device
    )
    decode_steps = args.new_tokens - 1
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    print(
        f"Workload: backend={execution_backend}, model={args.model}, "
        f"batch={args.batch_size}, "
        f"prompt_len={prompt_ids.shape[1]}, new_tokens={args.new_tokens}, "
        f"dtype={args.dtype}, sampling={args.sampling}, "
        f"cache={model.generation_config.cache_implementation or 'dynamic'}"
    )
    print(
        "Phase boundary: 1 prompt-prefill generation iteration + "
        f"{decode_steps} cached-decode iterations = {args.new_tokens} new tokens"
    )

    with torch.no_grad():
        # One complete generation covers every growing-cache decode shape.
        warmup_output = generate_full(model, prompt_ids, tokenizer, args.new_tokens)
        synchronize()
        del warmup_output

        # The benchmark number is from a separate, unprofiled full generation.
        synchronize()
        start = time.perf_counter()
        timed_output = generate_full(model, prompt_ids, tokenizer, args.new_tokens).to(
            "cpu"
        )
        synchronize()
        wall_seconds = time.perf_counter() - start
        generated_per_sequence = timed_output.shape[1] - prompt_ids.shape[1]
        generated_tokens = args.batch_size * generated_per_sequence
        throughput = generated_tokens / wall_seconds
        print(
            f"Unprofiled generation: {wall_seconds:.3f} s, "
            f"{generated_tokens} tokens, {throughput:.1f} tok/s aggregate, "
            f"{throughput / args.batch_size:.2f} tok/s/sequence"
        )

        # Plain profile context A: HF's prompt-prefill iteration.
        prof_prefill, generated_ids, past_key_values = profile_prefill(
            model, prompt_ids, tokenizer, activities, synchronize
        )
        prefill_us, prefill_idle = dump_phase(
            prof_prefill, "prefill", args.output_dir, args.row_limit
        )

        # Plain profile context B: resume the same HF DynamicCache for all
        # remaining iterations.  No profiler schedule or prof.step() is used.
        prof_decode, profile_output = profile_decode(
            model,
            generated_ids,
            past_key_values,
            tokenizer,
            decode_steps,
            activities,
            synchronize,
        )
        decode_us, decode_idle = dump_phase(
            prof_decode, "decode", args.output_dir, args.row_limit
        )

    if profile_output.shape[1] - prompt_ids.shape[1] != args.new_tokens:
        raise RuntimeError("Split profiled generation produced the wrong length")
    if not torch.equal(timed_output, profile_output.to("cpu")):
        raise RuntimeError(
            "Split profiled generation did not match uninterrupted generation"
        )
    print("Correctness: split profiled output matches uninterrupted generation")

    combined_us = prefill_us + decode_us
    print(
        "\nGPU time split (ATen self time): "
        f"prefill {prefill_us / 1000:.3f} ms "
        f"({100 * prefill_us / combined_us:.2f}%) | "
        f"decode {decode_us / 1000:.3f} ms "
        f"({100 * decode_us / combined_us:.2f}%)"
    )
    print(
        f"Cached decode average: {decode_us / 1000 / decode_steps:.3f} "
        f"ms/step over {decode_steps} growing-cache steps; "
        f"combined average: {combined_us / 1000 / args.new_tokens:.3f} "
        "ms/generated token"
    )

    summary = {
        "environment": {
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "hardware": hardware_name,
            "execution_backend": execution_backend,
            "torch_device": execution_device,
            "transformers": transformers.__version__,
        },
        "workload": {
            "model": args.model,
            "batch_size": args.batch_size,
            "prompt_len": args.prompt_len,
            "new_tokens": args.new_tokens,
            "cached_decode_steps": decode_steps,
            "dtype": args.dtype,
            "sampling": args.sampling,
            "prompt": PROMPT,
        },
        "unprofiled": {
            "wall_seconds": wall_seconds,
            "generated_tokens": generated_tokens,
            "tokens_per_second": throughput,
        },
        "profile": {
            "prefill_self_gpu_us": prefill_us,
            "decode_self_gpu_us": decode_us,
            "decode_ms_per_cached_step": decode_us / 1000 / decode_steps,
            "combined_ms_per_generated_token": combined_us / 1000 / args.new_tokens,
            "prefill_gpu_trace": prefill_idle,
            "decode_gpu_trace": decode_idle,
        },
    }
    with (args.output_dir / "gpt2_aten_profile_summary.json").open("w") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    main()
