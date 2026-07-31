"""Op-by-op nanoGPT training-step comparison between PyTorch-ROCm and Mojo.

Both backends spend essentially 100% of the step inside GPU kernels, so the
trustworthy unit of comparison is GPU kernel time, not the profiler's ATen
self-time rows.  ATen self time is also incomplete on the Mojo side: the
eager SDPA path runs under a custom autograd function
(``_ScaledDotProductAttentionAutograd``, or ``_FusedFlashAttentionAutograd``
once the fused gfx942 kernels claim the shape), which the profiler records as a
``user_annotation`` rather than an ``aten::`` row, so an ATen-only view silently
drops it.

This script therefore reads the chrome traces written by
``profile_nanogpt_train_aten.py``, attributes every GPU kernel to the deepest
enclosing CPU range through the profiler's ``External id`` correlation, and
folds those ranges into functional groups that mean the same thing on both
backends.  Every kernel is accounted for: anything no group claims is reported
in a residual table instead of being dropped.

Usage:
    uv run --no-sync python compare_nanogpt_train_kernels.py \
        --rocm-dir current_bench_train/rocm --mojo-dir current_bench_train/mojo \
        --output-dir current_bench_train/comparison
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from tabulate import tabulate

# (label, pass, ROCm range names, Mojo range names).  A range is the deepest
# enclosing profiler range for a kernel: an ATen op name, or one of the Mojo
# autograd-function annotations.  ``pass`` is "forward", "backward" or "any";
# it is what separates ROCm's three Linear GEMMs, which all dispatch through
# aten::mm regardless of pass.
GROUPS: list[tuple[str, str, set[str], set[str]]] = [
    ("Linear GEMM (forward)", "forward", {"aten::mm"}, {"aten::linear"}),
    (
        "Linear GEMM (backward)",
        "backward",
        {"aten::mm"},
        {"aten::linear_backward", "aten::linear"},
    ),
    (
        "SDPA (forward)",
        "any",
        {"aten::_flash_attention_forward", "aten::scaled_dot_product_attention"},
        {"_ScaledDotProductAttentionAutograd", "_FusedFlashAttentionAutograd"},
    ),
    (
        "SDPA (backward)",
        "any",
        {"aten::_flash_attention_backward"},
        {
            "_ScaledDotProductAttentionAutogradBackward",
            "_FusedFlashAttentionAutogradBackward",
        },
    ),
    (
        "Copies / dtype casts / layout",
        "any",
        {"aten::copy_", "aten::_to_copy", "aten::clone", "aten::contiguous"},
        {"aten::copy_", "aten::_to_copy", "aten::clone", "aten::contiguous"},
    ),
    (
        "LayerNorm (forward)",
        "any",
        {"aten::native_layer_norm"},
        {"aten::native_layer_norm"},
    ),
    (
        "LayerNorm (backward)",
        "any",
        {"aten::native_layer_norm_backward"},
        {"aten::native_layer_norm_backward"},
    ),
    (
        "Cross entropy (forward)",
        "any",
        {"aten::_log_softmax", "aten::nll_loss_forward"},
        {"aten::_log_softmax", "aten::nll_loss_forward"},
    ),
    (
        "Cross entropy (backward)",
        "any",
        {"aten::_log_softmax_backward_data", "aten::nll_loss_backward"},
        {"aten::_log_softmax_backward_data", "aten::nll_loss_backward"},
    ),
    ("GELU (forward)", "any", {"aten::gelu"}, {"aten::gelu"}),
    ("GELU (backward)", "any", {"aten::gelu_backward"}, {"aten::gelu_backward"}),
    (
        "Residual / elementwise add",
        "any",
        {"aten::add", "aten::add_", "aten::mul", "aten::mul_"},
        {"aten::add", "aten::add_", "aten::mul", "aten::mul_"},
    ),
    (
        "Embedding",
        "any",
        {"aten::embedding", "aten::embedding_dense_backward"},
        {"aten::embedding", "aten::embedding_dense_backward"},
    ),
    (
        "Concat / stack",
        "any",
        {"aten::cat", "aten::stack"},
        {"aten::cat", "aten::stack"},
    ),
    (
        # READ THIS TOGETHER WITH "Cross entropy (backward)". A kernel is
        # attributed to its DEEPEST enclosing CPU range, so PyTorch-ROCm's
        # zeroing of the logits-sized gradient -- 8 x aten::fill_ on
        # [49152, 50304], 2006.7 us/step, nested inside aten::nll_loss_backward
        # -- lands here rather than in cross entropy, while the Mojo device
        # fuses the same zeroing into its nll_loss_backward kernel and so shows
        # nothing here at all. Taken separately the two rows read as a 2.1 ms
        # win plus a 2.8 ms loss; taken together the truth is
        # 7253.5 us against 6431.1 (1.13x), and the only real gap is
        # _log_softmax_backward_data at 5272.8 against 4095.2 (1.29x).
        "Zero / fill grads",
        "any",
        {"aten::fill_", "aten::zero_", "aten::zeros", "aten::zeros_like"},
        {"aten::fill_", "aten::zero_", "aten::zeros", "aten::zeros_like"},
    ),
    (
        "Optimizer (fused AdamW)",
        "any",
        {"aten::_fused_adamw_", "aten::_foreach_add_"},
        {"aten::_fused_adamw_", "aten::_foreach_add_"},
    ),
    (
        "Gradient clipping",
        "any",
        {
            "aten::_foreach_norm",
            "aten::_foreach_mul_",
            "aten::linalg_vector_norm",
            "aten::clamp",
            "aten::reciprocal",
        },
        {
            "aten::_foreach_norm",
            "aten::_foreach_mul_",
            "aten::linalg_vector_norm",
            "aten::clamp",
            "aten::reciprocal",
        },
    ),
    (
        "Misc reductions / indexing",
        "any",
        {"aten::sum", "aten::gather", "aten::arange", "aten::ones_like"},
        {"aten::sum", "aten::gather", "aten::arange", "aten::ones_like"},
    ),
]

# Ranges opened by PyTorch's autograd engine.  A CPU range nested inside one
# belongs to the backward pass, which is how forward and backward uses of the
# same ATen op (ROCm runs all three Linear GEMMs through aten::mm) are told
# apart.  Shapes cannot do it: a data-gradient GEMM has the same operand
# geometry as a forward GEMM.
_AUTOGRAD_PREFIX = "autograd::engine::evaluate_function"
BACKWARD_SUFFIX = "#backward"


@dataclass
class RangeStats:
    name: str
    us: float = 0.0
    kernels: int = 0
    kernel_names: dict[str, float] = field(default_factory=dict)


def load_trace(path: Path) -> list[dict[str, object]]:
    with path.open() as file:
        data = json.load(file)
    return data["traceEvents"] if isinstance(data, dict) else data


def deepest_ranges(events: list[dict[str, object]]) -> dict[int, tuple[str, bool]]:
    """Map each profiler External id to its deepest CPU range and pass.

    Ranges are nested by time on a single thread, so a per-thread stack
    recovers the ancestry the chrome trace only encodes implicitly.  The pass
    flag is true when the range or any of its ancestors was opened by the
    autograd engine.
    """
    by_thread: dict[object, list[dict[str, object]]] = {}
    for event in events:
        if event.get("cat") not in ("cpu_op", "user_annotation"):
            continue
        if event.get("args", {}).get("External id") is None:
            continue
        by_thread.setdefault(event.get("tid"), []).append(event)

    ranges: dict[int, tuple[str, bool]] = {}
    for thread_events in by_thread.values():
        # Outer ranges start first and, on ties, last longer.
        thread_events.sort(
            key=lambda event: (event["ts"], -float(event.get("dur", 0.0)))
        )
        stack: list[tuple[float, bool]] = []  # (end timestamp, inside autograd)
        for event in thread_events:
            start = float(event["ts"])
            end = start + float(event.get("dur", 0.0))
            while stack and stack[-1][0] < start:
                stack.pop()
            inherited = stack[-1][1] if stack else False
            is_backward = inherited or event["name"].startswith(_AUTOGRAD_PREFIX)
            stack.append((end, is_backward))
            ranges[event["args"]["External id"]] = (event["name"], is_backward)
    return ranges


def attribute(events: list[dict[str, object]]) -> tuple[dict[str, RangeStats], float]:
    """Sum GPU kernel time per owning CPU range, tagged forward or backward."""
    ranges = deepest_ranges(events)
    stats: dict[str, RangeStats] = {}
    total = 0.0
    for event in events:
        if event.get("cat") != "kernel":
            continue
        duration = float(event.get("dur", 0.0))
        total += duration
        external_id = event.get("args", {}).get("External id")
        name, is_backward = ranges.get(external_id, ("<<unattributed>>", False))
        owner = name + BACKWARD_SUFFIX if is_backward else name
        entry = stats.setdefault(owner, RangeStats(name=owner))
        entry.us += duration
        entry.kernels += 1
        kernel_name = str(event.get("name", "?"))
        entry.kernel_names[kernel_name] = (
            entry.kernel_names.get(kernel_name, 0.0) + duration
        )
    return stats, total


def scale(stats: dict[str, RangeStats], steps: int) -> None:
    for entry in stats.values():
        entry.us /= steps
        entry.kernels = round(entry.kernels / steps)
        for name in list(entry.kernel_names):
            entry.kernel_names[name] /= steps


def matching_keys(
    stats: dict[str, RangeStats], names: set[str], pass_name: str
) -> set[str]:
    """Keys in ``stats`` whose base name and pass match the group's request."""
    keys = set()
    for key in stats:
        is_backward = key.endswith(BACKWARD_SUFFIX)
        base = key[: -len(BACKWARD_SUFFIX)] if is_backward else key
        if base not in names:
            continue
        if pass_name == "forward" and is_backward:
            continue
        if pass_name == "backward" and not is_backward:
            continue
        keys.add(key)
    return keys


def group_total(stats: dict[str, RangeStats], keys: set[str]) -> tuple[float, int]:
    return (sum(stats[key].us for key in keys), sum(stats[key].kernels for key in keys))


def top_kernels(stats: dict[str, RangeStats], keys: set[str], count: int) -> str:
    merged: dict[str, float] = {}
    for key in keys:
        for kernel_name, duration in stats[key].kernel_names.items():
            merged[kernel_name] = merged.get(kernel_name, 0.0) + duration
    ordered = sorted(merged.items(), key=lambda item: item[1], reverse=True)[:count]
    return ", ".join(
        f"{shorten_kernel(name)} {duration / 1000:.1f}ms" for name, duration in ordered
    )


def shorten_kernel(name: str) -> str:
    cleaned = name.split("(")[0].replace("void ", "").replace("at::native::", "")
    if "Cijk_" in cleaned:
        return "Tensile:" + cleaned.split("UserArgs_")[-1].split("_SN")[0]
    # Mojo kernel names carry a trailing content hash; it adds no information.
    parts = cleaned.rsplit("_", 1)
    if (
        len(parts) == 2
        and len(parts[1]) >= 8
        and all(character in "0123456789abcdef" for character in parts[1])
    ):
        cleaned = parts[0]
    return cleaned[:44]


def format_ratio(mojo_us: float, rocm_us: float) -> str:
    if rocm_us <= 0:
        return "n/a" if mojo_us <= 0 else "inf"
    return f"{mojo_us / rocm_us:.2f}x"


def build_rows(
    rocm: dict[str, RangeStats], mojo: dict[str, RangeStats]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows = []
    claimed_rocm: set[str] = set()
    claimed_mojo: set[str] = set()
    for label, pass_name, rocm_names, mojo_names in GROUPS:
        rocm_keys = matching_keys(rocm, rocm_names, pass_name)
        mojo_keys = matching_keys(mojo, mojo_names, pass_name)
        rocm_us, rocm_kernels = group_total(rocm, rocm_keys)
        mojo_us, mojo_kernels = group_total(mojo, mojo_keys)
        claimed_rocm |= rocm_keys
        claimed_mojo |= mojo_keys
        if rocm_us <= 0 and mojo_us <= 0:
            continue
        rows.append(
            {
                "label": label,
                "rocm_us": rocm_us,
                "mojo_us": mojo_us,
                "rocm_kernels": rocm_kernels,
                "mojo_kernels": mojo_kernels,
                "gap_us": mojo_us - rocm_us,
                "rocm_top": top_kernels(rocm, rocm_keys, 2),
                "mojo_top": top_kernels(mojo, mojo_keys, 2),
            }
        )

    residual = []
    for name in sorted((set(rocm) - claimed_rocm) | (set(mojo) - claimed_mojo)):
        rocm_us = rocm[name].us if name in rocm else 0.0
        mojo_us = mojo[name].us if name in mojo else 0.0
        residual.append(
            {
                "label": f"[residual] {name}",
                "rocm_us": rocm_us,
                "mojo_us": mojo_us,
                "rocm_kernels": rocm[name].kernels if name in rocm else 0,
                "mojo_kernels": mojo[name].kernels if name in mojo else 0,
                "gap_us": mojo_us - rocm_us,
                "rocm_top": top_kernels(rocm, {name} & set(rocm), 1),
                "mojo_top": top_kernels(mojo, {name} & set(mojo), 1),
            }
        )
    return rows, residual


def render(rows: list[dict[str, object]], title: str, total_gap: float) -> str:
    ordered = sorted(rows, key=lambda row: row["gap_us"], reverse=True)
    table = []
    cumulative = 0.0
    for rank, row in enumerate(ordered, start=1):
        if row["gap_us"] > 0:
            cumulative += row["gap_us"]
        table.append(
            [
                rank,
                row["label"],
                row["rocm_kernels"],
                row["mojo_kernels"],
                f"{row['rocm_us'] / 1000:.3f}",
                f"{row['mojo_us'] / 1000:.3f}",
                format_ratio(row["mojo_us"], row["rocm_us"]),
                f"{row['gap_us'] / 1000:+.3f}",
                f"{100 * cumulative / total_gap:.1f}" if total_gap > 0 else "n/a",
            ]
        )
    return f"\n================ {title} ================\n" + tabulate(
        table,
        headers=[
            "rank",
            "target",
            "rocm K/step",
            "mojo K/step",
            "rocm ms",
            "mojo ms",
            "ratio",
            "gap ms",
            "cum % gap",
        ],
        tablefmt="simple",
        disable_numparse=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocm-dir", type=Path, required=True)
    parser.add_argument("--mojo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=3, help="profiled steps per trace")
    parser.add_argument("--target-ratio", type=float, default=1.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rocm_events = load_trace(args.rocm_dir / "trace_step.json")
    mojo_events = load_trace(args.mojo_dir / "trace_step.json")

    rocm, rocm_total = attribute(rocm_events)
    mojo, mojo_total = attribute(mojo_events)
    scale(rocm, args.steps)
    scale(mojo, args.steps)
    rocm_total /= args.steps
    mojo_total /= args.steps

    rows, residual = build_rows(rocm, mojo)
    all_rows = rows + residual
    total_gap = sum(row["gap_us"] for row in all_rows if row["gap_us"] > 0)

    print(render(all_rows, "nanoGPT training step: ranked GPU-kernel gap", total_gap))
    accounted_rocm = sum(row["rocm_us"] for row in all_rows)
    accounted_mojo = sum(row["mojo_us"] for row in all_rows)
    print(
        f"\nROCm  {rocm_total / 1000:.3f} ms/step of GPU kernels "
        f"({accounted_rocm / 1000:.3f} ms accounted)\n"
        f"Mojo  {mojo_total / 1000:.3f} ms/step of GPU kernels "
        f"({accounted_mojo / 1000:.3f} ms accounted)\n"
        f"Overall {format_ratio(mojo_total, rocm_total)}, "
        f"gap to close {total_gap / 1000:.3f} ms/step"
    )

    failing = [
        row
        for row in all_rows
        if row["rocm_us"] > 0 and row["mojo_us"] / row["rocm_us"] > args.target_ratio
    ]
    failing.sort(key=lambda row: row["gap_us"], reverse=True)
    print(
        f"\n{len(failing)} targets above the {args.target_ratio:.2f}x acceptance "
        "ratio, in impact order:"
    )
    running = 0.0
    for rank, row in enumerate(failing, start=1):
        running += max(row["gap_us"], 0.0)
        print(
            f"  {rank}. {row['label']}: {row['mojo_us'] / 1000:.3f} vs "
            f"{row['rocm_us'] / 1000:.3f} ms "
            f"({format_ratio(row['mojo_us'], row['rocm_us'])}), "
            f"gap {row['gap_us'] / 1000:+.3f} ms, "
            f"cumulative {100 * running / total_gap:.1f}% of the gap"
        )
        if row["mojo_top"]:
            print(f"       mojo kernels: {row['mojo_top']}")
        if row["rocm_top"]:
            print(f"       rocm kernels: {row['rocm_top']}")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "nanogpt_train_kernel_gap.csv").open(
            "w", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "target",
                    "rocm_kernels_per_step",
                    "mojo_kernels_per_step",
                    "rocm_us_per_step",
                    "mojo_us_per_step",
                    "ratio",
                    "gap_us_per_step",
                    "rocm_top_kernels",
                    "mojo_top_kernels",
                ]
            )
            for row in sorted(all_rows, key=lambda row: row["gap_us"], reverse=True):
                ratio = row["mojo_us"] / row["rocm_us"] if row["rocm_us"] > 0 else ""
                writer.writerow(
                    [
                        row["label"],
                        row["rocm_kernels"],
                        row["mojo_kernels"],
                        f"{row['rocm_us']:.1f}",
                        f"{row['mojo_us']:.1f}",
                        f"{ratio:.4f}" if ratio != "" else "",
                        f"{row['gap_us']:.1f}",
                        row["rocm_top"],
                        row["mojo_top"],
                    ]
                )
        with (args.output_dir / "nanogpt_train_kernel_gap.json").open("w") as file:
            json.dump(
                {
                    "rocm_us_per_step": rocm_total,
                    "mojo_us_per_step": mojo_total,
                    "overall_ratio": mojo_total / rocm_total if rocm_total else None,
                    "positive_gap_us_per_step": total_gap,
                    "target_ratio": args.target_ratio,
                    "work_order": [
                        {
                            "target": row["label"],
                            "rocm_us_per_step": row["rocm_us"],
                            "mojo_us_per_step": row["mojo_us"],
                            "ratio": row["mojo_us"] / row["rocm_us"],
                            "gap_us_per_step": row["gap_us"],
                        }
                        for row in failing
                    ],
                },
                file,
                indent=2,
            )


if __name__ == "__main__":
    main()
