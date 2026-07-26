"""Run bench_linear_gemm one case per process and weight the per-variant totals.

The harness runs its whole list in one process, and a later case reads low when
an earlier one warmed the device (optimization_journal.md, the NT measurement
note).  This driver launches one process per case, parses the single data row,
and recomputes the NN / NT / TN weighted ratios from those isolated numbers.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

TARGETS = Path("harness/nanogpt_train/rocm_gemm_targets.csv")


def variant_of(transpose_a: str, transpose_b: str) -> str:
    if transpose_a != "0":
        return "TN"
    if transpose_b != "0":
        return "NT"
    return "NN"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="/tmp/bench_gemm")
    parser.add_argument("--targets", default=str(TARGETS))
    parser.add_argument("--variants", default="NN,NT,TN")
    parser.add_argument("--iterations", default="100")
    parser.add_argument("--warmup", default="25")
    parser.add_argument("--extra", default="")
    args = parser.parse_args()

    wanted = set(args.variants.split(","))
    with open(args.targets, newline="") as handle:
        rows = list(csv.DictReader(handle))

    totals: dict[str, list[float]] = {
        "NN": [0.0, 0.0],
        "NT": [0.0, 0.0],
        "TN": [0.0, 0.0],
    }
    print(
        f"{'case':<22}{'variant':<8}{'mojo_us':>10}{'rocm_us':>10}"
        f"{'ratio':>8}{'TFLOP/s':>9}{'copy_us':>10}  correct"
    )
    for row in rows:
        variant = variant_of(row["transpose_a"], row["transpose_b"])
        if variant not in wanted:
            continue
        cmd = [
            args.binary,
            f"--targets={args.targets}",
            f"--case={row['label']}",
            f"--iterations={args.iterations}",
            f"--warmup={args.warmup}",
        ]
        if args.extra:
            cmd += args.extra.split()
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(
                f"{row['label']:<22}{variant:<8}  FAILED: {out.stdout.strip()[-400:]}"
            )
            print(out.stderr.strip()[-400:])
            return 1
        data = [
            line
            for line in out.stdout.splitlines()
            if line.strip().startswith(row["label"])
        ]
        if not data:
            print(f"{row['label']:<22}{variant:<8}  NO ROW\n{out.stdout}")
            return 1
        fields = data[0].split()
        mojo_us = float(fields[7])
        rocm_us = float(fields[8])
        tflops = float(fields[10])
        copy_us = float(fields[11])
        correct = " ".join(fields[12:])
        calls = float(row["calls_per_step"])
        totals[variant][0] += mojo_us * calls
        totals[variant][1] += rocm_us * calls
        print(
            f"{row['label']:<22}{variant:<8}{mojo_us:>10.2f}{rocm_us:>10.2f}"
            f"{mojo_us / rocm_us:>8.3f}{tflops:>9.1f}{copy_us:>10.2f}  {correct}"
        )
    print()
    grand = [0.0, 0.0]
    for variant in ("NN", "NT", "TN"):
        mojo, rocm = totals[variant]
        if rocm == 0.0:
            continue
        grand[0] += mojo
        grand[1] += rocm
        print(
            f"  {variant}  mojo {mojo / 1000.0:9.3f} ms   rocm {rocm / 1000.0:9.3f} ms"
            f"   ratio {mojo / rocm:6.3f}"
        )
    if grand[1] > 0.0:
        print(
            f"  ALL mojo {grand[0] / 1000.0:9.3f} ms   rocm {grand[1] / 1000.0:9.3f} ms"
            f"   ratio {grand[0] / grand[1]:6.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
