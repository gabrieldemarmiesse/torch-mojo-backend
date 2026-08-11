"""Where are we far behind stock PyTorch? Sort baselines by ratio.

Reads only benchmarks/baselines.html — no accelerator needed.

    uv run python benchmarks/report.py                # every hardware config
    uv run python benchmarks/report.py --hw H100      # configs matching a substring
    uv run python benchmarks/report.py --worst 20     # top offenders only
    uv run python benchmarks/report.py -k mm/tf32     # entry paths matching a substring

Entries print as their baseline tree path op/dtype/shape/layout, so -k
matches any axis ("bf16", "mm", "S7", "TN") or a path fragment ("mm/tf32").

For the same data as a collapsible tree with per-branch min/median/max,
open that same benchmarks/baselines.html in a browser: it carries both the
ratios and the viewer that renders them.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_lib import baselines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hw", default="", help="only configs whose key contains this")
    parser.add_argument("-k", default="", help="only entries whose key contains this")
    parser.add_argument(
        "--worst", type=int, default=0, help="show only the N worst entries per config"
    )
    args = parser.parse_args()

    if not baselines.BASELINES_PATH.exists():
        print(f"no baseline file at {baselines.BASELINES_PATH}; nothing to report")
        return 0
    data = baselines.load()
    configs = {
        key: cfg
        for key, cfg in sorted(data.configs.items())
        if args.hw.lower() in key.lower()
    }
    if not configs:
        print("no baselines recorded for the requested hardware; nothing to report")
        return 0

    # The torch version is the last " | torch <v>" component of the config
    # key; two torch versions of the same box are two separate configs that
    # sort next to each other.  Count versions per hardware so a superseded
    # block is visible instead of silently doubling the report.
    versions_per_hw = {}
    for key in configs:
        hw_part, _, _ = key.rpartition(" | torch ")
        versions_per_hw[hw_part or key] = versions_per_hw.get(hw_part or key, 0) + 1

    for key, cfg in configs.items():
        leaves = cfg.leaves()
        results = {
            str(bench_key): ratio
            for bench_key, ratio in leaves.items()
            if args.k.lower() in str(bench_key).lower()
        }
        # min/median/max are derived, so they are computed here rather than
        # stored in the file (see bench_lib/baselines.py).
        ratios = list(leaves.values())
        if not ratios:
            print(f"\n{key}  (block holds no ratio leaf at all)")
            continue
        print(
            f"\n{key}  ({len(results)} entries; all-leaf min/median/max "
            f"{min(ratios):.3f}/{statistics.median(ratios):.3f}/"
            f"{max(ratios):.3f})"
        )
        hw_part, _, _ = key.rpartition(" | torch ")
        if versions_per_hw.get(hw_part or key, 0) > 1:
            print(
                "  NOTE: several torch versions coexist for this hardware; "
                "ratios are only comparable within one version. Once every "
                "machine of this hardware has reseeded under the newer torch, "
                "delete the superseded config block in a reviewed commit."
            )
        if not results:
            print("  no matching entries")
            continue
        ranked = sorted(results.items(), key=lambda item: -item[1])
        if args.worst:
            ranked = ranked[: args.worst]
        print("  ratio ours/stock (device time); > 1 means we are slower")
        for entry, ratio in ranked:
            print(f"  {ratio:8.3f}  {entry}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
