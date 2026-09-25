#!/usr/bin/env python3
"""Summarize a tests/multinode/run_two_node_checks.sbatch job log.

Reads the SLURM job log named by that sbatch script's ``-o`` (e.g.
``/home/gabriel/ddp_work/logs/mojoccl_2node_<jobid>.log``) and prints one
markdown report with three tables:

- allreduce bench (``ar_bench_gpt2.py`` ``RESULT ...`` lines): per dtype and
  size, the vendor/mojo median device time and the mojo/vendor ratio.
- worker checks (``=== ddp_worker ccl=... mode=... exit: N ===``, and the
  same for ``fsdp_worker``, whose modes are listed as ``fsdp_worker:<mode>``):
  per (ccl, mode) pass/fail, from the leg's exit code.
- nanoGPT DDP training (``=== nanogpt_ddp ccl=... leg=... ===`` plus the
  script's own ``step NNNNN | ...`` lines): per (ccl, leg) the last logged
  step's loss and tok/s, the val loss if an eval ran, and pass/fail.

Stdlib only: this runs on the SLURM login node (no GPU, no torch import).

    uv run --no-sync python tests/multinode/summarize.py \\
        /home/gabriel/ddp_work/logs/mojoccl_2node_<jobid>.log
"""

import argparse
import statistics
import sys
from pathlib import Path
from re import Match, Pattern, compile as re_compile

_RESULT_RE: Pattern[str] = re_compile(
    r"^RESULT ccl=(?P<ccl>\S+) dtype=(?P<dtype>\S+) size_mib=\s*(?P<mib>\d+) "
    r"median_us=\s*(?P<median>[\d.]+) min_us=\s*(?P<min>[\d.]+) "
    r"busbw_gbs=\s*(?P<busbw>[\d.]+)"
)
_WORKER_HEADER_RE: Pattern[str] = re_compile(
    r"^=== (?P<worker>ddp|fsdp)_worker ccl=(?P<ccl>\S+) mode=(?P<mode>\S+) "
    r"\(16 ranks\) ===$"
)
_WORKER_EXIT_RE: Pattern[str] = re_compile(
    r"^=== (?P<worker>ddp|fsdp)_worker ccl=(?P<ccl>\S+) mode=(?P<mode>\S+) "
    r"exit: (?P<rc>\d+) ===$"
)


def _worker_mode(m: Match[str]) -> str:
    """ddp_worker modes keep their bare names; fsdp_worker's are prefixed."""
    mode = m.group("mode")
    return mode if m.group("worker") == "ddp" else f"fsdp_worker:{mode}"


_TRAIN_HEADER_RE: Pattern[str] = re_compile(
    r"^=== nanogpt_ddp ccl=(?P<ccl>\S+) leg=(?P<leg>\S+) ===$"
)
_TRAIN_EXIT_RE: Pattern[str] = re_compile(
    r"^=== nanogpt_ddp ccl=(?P<ccl>\S+) leg=(?P<leg>\S+) exit: (?P<rc>\d+) ===$"
)
_STEP_RE: Pattern[str] = re_compile(
    r"^step\s+(?P<step>\d+)\s*\|\s*loss\s+(?P<loss>-?[\d.]+)\s*\|\s*"
    r"(?P<toks>[\d.]+)k tok/s\s*\|\s*(?P<secs>[\d.]+)s"
)
_VAL_RE: Pattern[str] = re_compile(
    r"^step\s+(?P<step>\d+)\s*\|\s*val loss\s+(?P<val>-?[\d.]+)"
)
_SKIP_RE: Pattern[str] = re_compile(r"^=== SKIP (?P<what>.+) \(RUN_MOJO=0\) ===$")

BenchKey = tuple[str, str, int]  # (ccl, dtype, size_mib)
WorkerKey = tuple[str, str]  # (ccl, mode)
TrainKey = tuple[str, str]  # (ccl, leg)


def parse(lines: list[str]) -> dict[str, object]:
    bench = dict[BenchKey, list[float]]()
    worker = dict[WorkerKey, int | None]()
    train = dict[TrainKey, dict[str, float | int | None]]()
    skipped = list[str]()
    current_train = None

    for raw in lines:
        line = raw.rstrip("\n")

        m = _SKIP_RE.match(line)
        if m:
            skipped.append(m.group("what"))
            continue

        m = _RESULT_RE.match(line)
        if m:
            key = (m.group("ccl"), m.group("dtype"), int(m.group("mib")))
            bench.setdefault(key, []).append(float(m.group("median")))
            continue

        m = _WORKER_HEADER_RE.match(line)
        if m:
            worker.setdefault((m.group("ccl"), _worker_mode(m)), None)
            continue
        m = _WORKER_EXIT_RE.match(line)
        if m:
            worker[(m.group("ccl"), _worker_mode(m))] = int(m.group("rc"))
            continue

        m = _TRAIN_HEADER_RE.match(line)
        if m:
            key = (m.group("ccl"), m.group("leg"))
            train.setdefault(
                key,
                {
                    "rc": None,
                    "step": None,
                    "loss": None,
                    "toks": None,
                    "val_loss": None,
                },
            )
            current_train = key
            continue
        m = _TRAIN_EXIT_RE.match(line)
        if m:
            key = (m.group("ccl"), m.group("leg"))
            train.setdefault(
                key,
                {
                    "rc": None,
                    "step": None,
                    "loss": None,
                    "toks": None,
                    "val_loss": None,
                },
            )
            train[key]["rc"] = int(m.group("rc"))
            if current_train == key:
                current_train = None
            continue

        if current_train is not None:
            m = _STEP_RE.match(line)
            if m:
                step = int(m.group("step"))
                d = train[current_train]
                if d["step"] is None or step >= d["step"]:
                    d["step"] = step
                    d["loss"] = float(m.group("loss"))
                    d["toks"] = float(m.group("toks"))
                continue
            m = _VAL_RE.match(line)
            if m:
                train[current_train]["val_loss"] = float(m.group("val"))
                continue

    return {"bench": bench, "worker": worker, "train": train, "skipped": skipped}


def _fmt(x: float | int | None, spec: str = ".1f") -> str:
    return format(x, spec) if isinstance(x, (int, float)) else "N/A"


def render_bench(bench: dict[BenchKey, list[float]]) -> str:
    sizes = dict[tuple[str, int], dict[str, float]]()
    for (ccl, dtype, mib), medians in bench.items():
        sizes.setdefault((dtype, mib), {})[ccl] = statistics.median(medians)

    lines = [
        "| dtype | size (MiB) | vendor median (us) | mojo median (us) | mojo/vendor |",
        "|---|---|---|---|---|",
    ]
    for dtype, mib in sorted(sizes, key=lambda k: (k[0], k[1])):
        cell = sizes[(dtype, mib)]
        vendor = cell.get("vendor")
        mojo = cell.get("mojo")
        ratio = mojo / vendor if (vendor and mojo) else None
        lines.append(
            f"| {dtype} | {mib} | {_fmt(vendor)} | {_fmt(mojo)} | {_fmt(ratio, '.3f')} |"
        )
    if len(lines) == 2:
        lines.append("| _(no RESULT lines found)_ | | | | |")
    return "\n".join(lines)


def render_worker(worker: dict[WorkerKey, int | None]) -> str:
    lines = ["| ccl | mode | result |", "|---|---|---|"]
    for ccl, mode in sorted(worker):
        rc = worker[(ccl, mode)]
        status = "PASS" if rc == 0 else ("FAIL" if rc is not None else "NO EXIT LOGGED")
        lines.append(
            f"| {ccl} | {mode} | {status} (rc={_fmt(rc, 'd') if rc is not None else 'N/A'}) |"
        )
    if len(lines) == 2:
        lines.append("| _(no worker legs found)_ | | |")
    return "\n".join(lines)


def render_train(train: dict[TrainKey, dict[str, float | int | None]]) -> str:
    lines = [
        "| ccl | leg | step | loss | val loss | tok/s | result |",
        "|---|---|---|---|---|---|---|",
    ]
    for ccl, leg in sorted(train):
        d = train[(ccl, leg)]
        rc = d["rc"]
        status = "PASS" if rc == 0 else ("FAIL" if rc is not None else "NO EXIT LOGGED")
        toks = f"{d['toks']:.1f}k" if isinstance(d["toks"], (int, float)) else "N/A"
        lines.append(
            f"| {ccl} | {leg} | {_fmt(d['step'], 'd') if d['step'] is not None else 'N/A'} "
            f"| {_fmt(d['loss'], '.4f')} | {_fmt(d['val_loss'], '.4f')} | {toks} | {status} |"
        )
    if len(lines) == 2:
        lines.append("| _(no nanogpt_ddp legs found)_ | | | | | | |")
    return "\n".join(lines)


def render(parsed: dict[str, object], source: str) -> str:
    bench = parsed["bench"]
    worker = parsed["worker"]
    train = parsed["train"]
    skipped = parsed["skipped"]
    assert isinstance(bench, dict)
    assert isinstance(worker, dict)
    assert isinstance(train, dict)
    assert isinstance(skipped, list)

    parts = [f"# mojoccl two-node summary\n\nSource: `{source}`\n"]
    if skipped:
        parts.append(
            "Skipped legs (RUN_MOJO=0): " + ", ".join(sorted(set(skipped))) + "\n"
        )
    parts.append("## Allreduce bench (ar_bench_gpt2.py)\n")
    parts.append(render_bench(bench) + "\n")
    parts.append("## worker checks (16 ranks)\n")
    parts.append(render_worker(worker) + "\n")
    parts.append("## nanoGPT DDP training (40 steps)\n")
    parts.append(render_train(train) + "\n")
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "logs",
        nargs="+",
        type=Path,
        help="job log path(s) from the sbatch job's -o file",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="write markdown here instead of stdout"
    )
    args = parser.parse_args()

    lines = list[str]()
    for path in args.logs:
        lines.extend(path.read_text(errors="replace").splitlines())

    parsed = parse(lines)
    report = render(parsed, ", ".join(str(p) for p in args.logs))

    if args.out is not None:
        args.out.write_text(report)
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
