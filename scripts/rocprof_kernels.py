#!/usr/bin/env python3
"""Per-kernel GPU profiling for any command, ROCm's answer to ``ncu``.

Wraps ``rocprofv3`` and reads its SQLite result database with the standard
library only, so it runs under any interpreter and needs no extra packages.
It works identically on a standalone Mojo benchmark binary and on a PyTorch
script, which makes Mojo-versus-ROCm kernel comparisons directly comparable.

Two modes, combinable:

* ``--kernel-trace`` (default): one row per distinct kernel with call count,
  median/min/max duration, grid and block geometry, VGPR/SGPR counts, LDS and
  scratch usage.  This is the launch-geometry and occupancy-input view.
* ``--pmc "COUNTER ..."``: hardware counters per kernel, summed over the
  dispatches of that kernel and also reported per dispatch.  Counters are read
  on this SR-IOV virtual function, so treat absolute values as relative
  indicators between candidate kernels rather than as machine-wide totals.

Examples:
    # Launch geometry and register/LDS usage of a Mojo benchmark binary
    uv run --no-sync python scripts/rocprof_kernels.py -- /tmp/bench_layernorm --rows 49152 --cols 768

    # Same for the ROCm reference, filtered to the interesting kernel
    uv run --no-sync python scripts/rocprof_kernels.py --filter layer_norm -- \
        python /tmp/reference_layernorm.py

    # Memory and MFMA counters
    uv run --no-sync python scripts/rocprof_kernels.py \
        --pmc "SQ_WAVES SQ_INSTS_MFMA FETCH_SIZE WRITE_SIZE" -- /tmp/bench_gemm

Useful gfx942 counters:
    SQ_WAVES                 waves launched
    SQ_INSTS_MFMA            MFMA instructions issued
    SQ_INSTS_VALU            vector ALU instructions
    SQ_INSTS_LDS             LDS instructions
    SQ_LDS_BANK_CONFLICT     LDS bank conflict cycles
    FETCH_SIZE / WRITE_SIZE  HBM bytes read / written (KB)
    GRBM_GUI_ACTIVE          GPU busy cycles
    SQ_BUSY_CU_CYCLES        per-CU busy cycles
Run ``rocprofv3 --list-avail`` for the full list.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROCM_BIN = Path("/opt/rocm/bin")


def find_rocprofv3() -> str:
    found = shutil.which("rocprofv3")
    if found:
        return found
    candidate = ROCM_BIN / "rocprofv3"
    if candidate.is_file():
        return str(candidate)
    for root in sorted(Path("/opt").glob("rocm-*"), reverse=True):
        candidate = root / "bin" / "rocprofv3"
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("rocprofv3 not found; is ROCm installed?")


@dataclass
class KernelStats:
    name: str
    durations_ns: list[float] = field(default_factory=list)
    grid: tuple[int, int, int] = (0, 0, 0)
    block: tuple[int, int, int] = (0, 0, 0)
    vgpr: int = 0
    accum_vgpr: int = 0
    sgpr: int = 0
    lds: int = 0
    scratch: int = 0
    counters: dict[str, float] = field(default_factory=dict)


def run_rocprofv3(
    command: list[str], output_dir: Path, pmc: str | None, kernel_trace: bool
) -> None:
    argv = [find_rocprofv3()]
    if kernel_trace or not pmc:
        argv.append("--kernel-trace")
    if pmc:
        argv += ["--pmc", *pmc.split()]
    argv += ["-d", str(output_dir), "--"]
    argv += command
    environment = dict(os.environ)
    # rocprofv3 needs its own ROCm libraries ahead of anything a venv injects.
    completed = subprocess.run(argv, env=environment)
    if completed.returncode != 0:
        raise SystemExit(
            f"profiled command exited with {completed.returncode}: {' '.join(command)}"
        )


def result_databases(output_dir: Path) -> list[Path]:
    return sorted(output_dir.rglob("*_results.db"))


def load_kernels(database: Path, name_filter: str | None) -> dict[str, KernelStats]:
    connection = sqlite3.connect(database)
    stats: dict[str, KernelStats] = {}
    query = """
        select name, duration, grid_x, grid_y, grid_z,
               workgroup_x, workgroup_y, workgroup_z,
               vgpr_count, accum_vgpr_count, sgpr_count, lds_size, scratch_size
        from kernels
    """
    for row in connection.execute(query):
        name = row[0]
        if name_filter and name_filter not in name:
            continue
        entry = stats.setdefault(name, KernelStats(name=name))
        entry.durations_ns.append(float(row[1]))
        entry.grid = (row[2], row[3], row[4])
        entry.block = (row[5], row[6], row[7])
        entry.vgpr, entry.accum_vgpr, entry.sgpr = row[8], row[9], row[10]
        entry.lds, entry.scratch = row[11], row[12]

    try:
        counter_query = """
            select name, counter_name, sum(counter_value), count(*)
            from pmc_events group by name, counter_name
        """
        for name, counter_name, total, dispatches in connection.execute(counter_query):
            if name_filter and name_filter not in name:
                continue
            entry = stats.setdefault(name, KernelStats(name=name))
            entry.counters[counter_name] = total / max(dispatches, 1)
    except sqlite3.OperationalError:
        pass  # kernel-trace-only run has no pmc_events table
    connection.close()
    return stats


def shorten(name: str, width: int) -> str:
    """Keep kernel names readable: drop C++ template noise, then truncate."""
    cleaned = name.split("(")[0]
    cleaned = cleaned.replace("void ", "").replace("at::native::", "")
    if len(cleaned) <= width:
        return cleaned
    return cleaned[: width - 3] + "..."


def render(stats: dict[str, KernelStats], top: int, name_width: int) -> str:
    ordered = sorted(
        stats.values(), key=lambda entry: sum(entry.durations_ns), reverse=True
    )[:top]
    counter_names: list[str] = []
    for entry in ordered:
        for counter in entry.counters:
            if counter not in counter_names:
                counter_names.append(counter)

    headers = [
        "kernel",
        "calls",
        "total us",
        "median us",
        "min us",
        "grid",
        "block",
        "vgpr",
        "sgpr",
        "lds",
        "scr",
    ] + counter_names
    rows = []
    for entry in ordered:
        durations = entry.durations_ns or [0.0]
        rows.append(
            [
                shorten(entry.name, name_width),
                str(len(entry.durations_ns)),
                f"{sum(durations) / 1e3:.1f}",
                f"{statistics.median(durations) / 1e3:.2f}",
                f"{min(durations) / 1e3:.2f}",
                "x".join(str(value) for value in entry.grid),
                "x".join(str(value) for value in entry.block),
                str(entry.vgpr + entry.accum_vgpr),
                str(entry.sgpr),
                str(entry.lds),
                str(entry.scratch),
            ]
            + [f"{entry.counters.get(counter, 0.0):.0f}" for counter in counter_names]
        )

    widths = [
        max(len(str(header)), *(len(row[index]) for row in rows))
        if rows
        else len(str(header))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(
            str(header).ljust(widths[index]) for index, header in enumerate(headers)
        )
    ]
    lines.append("  ".join("-" * width for width in widths))
    for row in rows:
        lines.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pmc", default=None, help="space-separated counter names")
    parser.add_argument("--kernel-trace", action="store_true", default=False)
    parser.add_argument(
        "--filter", default=None, help="substring a kernel must contain"
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--name-width", type=int, default=64)
    parser.add_argument(
        "--keep",
        type=Path,
        default=None,
        help="keep the rocprofv3 output in this directory instead of a temp dir",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("usage: rocprof_kernels.py [options] -- <command> [args...]")

    with tempfile.TemporaryDirectory() as temporary:
        output_dir = args.keep if args.keep is not None else Path(temporary)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_rocprofv3(command, output_dir, args.pmc, args.kernel_trace)
        databases = result_databases(output_dir)
        if not databases:
            raise SystemExit(f"rocprofv3 produced no result database in {output_dir}")
        merged: dict[str, KernelStats] = {}
        for database in databases:
            for name, entry in load_kernels(database, args.filter).items():
                if name in merged:
                    merged[name].durations_ns.extend(entry.durations_ns)
                    merged[name].counters.update(entry.counters)
                else:
                    merged[name] = entry
        if not merged:
            raise SystemExit(
                "no kernels matched"
                + (f" filter {args.filter!r}" if args.filter else "")
            )
        print()
        print(render(merged, args.top, args.name_width))
        total_us = sum(sum(entry.durations_ns) for entry in merged.values()) / 1e3
        print(f"\n{len(merged)} distinct kernels, {total_us:.1f} us total GPU time")


if __name__ == "__main__":
    main()
