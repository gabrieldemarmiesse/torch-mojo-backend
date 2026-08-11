"""The single git-tracked baseline file and its merge discipline.

Layout of benchmarks/baselines.json:

    {
      "format": 2,
      "configs": {
        "NVIDIA H100 PCIe | cuda | kineto-device-time | torch 2.8.0+cu128": {
          "results": {
            "test_gemm.py::test_mm[S1_4096x4096x4096-NN-bf16]": 1.023,
            ...
          }
        },
        "AMD Instinct MI300X | rocm | kineto-device-time | torch 2.8.0+rocm6.4": { ... }
      }
    }

Why this shape:

* ONE file for every hardware configuration, so H100 / MI300X / Apple /
  CPU baselines are reviewed side by side.
* The value is the RATIO ours/stock (device time), never an absolute
  time: ratios are hardware-normalized and survive clock drift.  ALL the
  context a ratio needs — which hardware, against which stock backend,
  timed how, on which torch build — lives in the config key.  The torch
  version is part of the KEY, not a field: the ratio's denominator IS
  stock PyTorch, so a torch upgrade changes the meaning of every ratio.
  Keying on it makes an upgraded box an unseen config ("never measured":
  pass and record), exactly like new hardware, instead of silently
  re-stamping numbers measured under the old build.  No timestamps, no
  commit stamps.
* Keys inside "results" are pytest node ids (relative to benchmarks/), so
  a failing test name and its baseline entry are the same string.
* Serialization is canonical (sorted keys, indent 2, one entry per line,
  trailing newline).  Because every writer re-emits the same canonical
  form, a partial update rewrites untouched entries byte-for-byte, each
  hardware config occupies one contiguous sorted block, and two machines
  updating different configs touch disjoint regions that git merges
  cleanly.

Writes go through merge_write(): read the file fresh under an exclusive
flock, update ONLY the measured entries, write atomically (tmp + rename).
A run that measured three relu cases can therefore never clobber the GEMM
numbers it did not run, nor entries written by a concurrent run.

Superseded torch versions: after a fleet-wide torch upgrade the old
config block keeps protecting nothing (no machine runs under that key any
more).  Prune it manually in a reviewed commit — delete the old
"... | torch <old>" block once every machine of that hardware has
reseeded under the new version.  Keep both blocks only while machines of
the same hardware genuinely run different torch builds.

Known git-merge limitation (deliberate, documented rather than solved):
two machines each adding a BRAND-NEW config block concurrently can
conflict, because both insert lines into the same gap of the sorted
"configs" object (updating existing, distinct configs merges cleanly —
those edits touch disjoint regions).  JSON's no-trailing-comma rule makes
this insertion-point conflict unavoidable in a single file.  To resolve:
keep BOTH added config blocks (union them, fixing the comma on the last
block), then re-canonicalize so the next writer diffs cleanly:

    uv run python -c "import sys; sys.path.insert(0, 'benchmarks'); \\
        from bench_lib import baselines; \\
        baselines.BASELINES_PATH.write_text( \\
            baselines.canonical_dump(baselines.load()))"
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

FORMAT_VERSION = 2

BASELINES_PATH = Path(__file__).resolve().parent.parent / "baselines.json"


def empty() -> dict:
    return {"format": FORMAT_VERSION, "configs": {}}


def load(path: Path = BASELINES_PATH) -> dict:
    if not path.exists():
        return empty()
    with path.open() as fh:
        data = json.load(fh)
    if data.get("format") != FORMAT_VERSION:
        raise ValueError(f"{path}: unknown baseline format {data.get('format')!r}")
    return data


def lookup(data: dict, hw_key: str, entry_key: str) -> float | None:
    config = data.get("configs", {}).get(hw_key)
    if config is None:
        return None
    value = config.get("results", {}).get(entry_key)
    return None if value is None else float(value)


def canonical_dump(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def merge_write(
    hw_key: str, entries: dict[str, float], path: Path = BASELINES_PATH
) -> None:
    """Merge `entries` into the file; every other entry survives byte-for-byte."""
    if not entries:
        return
    lock_path = path.with_suffix(".json.lock")
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            data = load(path)  # fresh read under the lock, not a cached copy
            config = data.setdefault("configs", {}).setdefault(hw_key, {"results": {}})
            results = config.setdefault("results", {})
            for key, ratio in entries.items():
                results[key] = round(float(ratio), 3)
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(canonical_dump(data))
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
