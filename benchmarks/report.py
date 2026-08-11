"""Where are we far behind stock PyTorch? Sort baselines by ratio.

Reads only benchmarks/baselines.json — no accelerator needed.

    uv run python benchmarks/report.py                # every hardware config
    uv run python benchmarks/report.py --hw H100      # configs matching a substring
    uv run python benchmarks/report.py --worst 20     # top offenders only
    uv run python benchmarks/report.py -k tf32        # entries matching a substring
"""

from __future__ import annotations

import argparse
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
        for key, cfg in sorted(data.get("configs", {}).items())
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
        results = {
            entry: ratio
            for entry, ratio in cfg.get("results", {}).items()
            if args.k.lower() in entry.lower()
        }
        print(f"\n{key}  ({len(results)} entries)")
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
