"""Tables from e2e_three_stacks logs. Every run is verified before it counts: the library
that ran (loader trace line for mojo stacks), the network transport (NCCL's NET/IB line, or
the OFI plugin's "Selected provider is <cxi|efa|...>" line on libfabric fabrics such as
Slingshot, for the NCCL/RCCL stacks; mojoccl has no TCP data path, its transport is verified
by a separate job), and the NUMA binding. Warm-up runs are discarded. Mean +- 95% CI over
rounds (t, n-1 dof).

usage: summarize_three_stacks.py JOB [LOGDIR] [WORLD]
env: E2E_STOCK_NAME (row label of the stock stack), E2E_SETUP (the "16 ranks on 2x8 H100"
phrase of the table caption), E2E_MODEL (the model phrase, default nanoGPT-124M), E2E_TAG (the
log-name prefix the sbatch wrote, default e2e_three_stacks).
"""

import glob
import os
import re
import statistics
import sys

job = sys.argv[1]
LOGDIR = sys.argv[2] if len(sys.argv) > 2 else "/home/gabriel/ddp_work/logs"
WORLD = int(sys.argv[3]) if len(sys.argv) > 3 else 16
SETUP = os.environ.get("E2E_SETUP", "16 ranks on 2x8 H100")
MODEL = os.environ.get("E2E_MODEL", "nanoGPT-124M")
TAG = os.environ.get("E2E_TAG", "e2e_three_stacks")
T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}
NAMES = {
    "stock": os.environ.get("E2E_STOCK_NAME", "stock CUDA torch 2.11 + NCCL"),
    "mojo_nccl": "mojo backend + NCCL",
    "mojo_mojoccl": "mojo backend + mojoccl (all Mojo)",
}
runs, problems = {}, []
for f in sorted(glob.glob(f"{LOGDIR}/{TAG}_{job}_bs*_*_*.log")):
    m = re.search(rf"{job}_bs(\d+)_(\d+)_(\w+)\.log", f)
    assert m is not None, f
    bs, i, cfg = int(m.group(1)), int(m.group(2)), m.group(3)
    if cfg.startswith("warmup"):
        continue
    text = open(f).read()
    tps, el = {}, {}
    for s in re.finditer(
        r"step\s+(\d+) \| loss ([\d.]+) \|\s+([\d.]+)k tok/s \|\s+([\d.]+)s", text
    ):
        k = int(s.group(1))
        tps[k] = float(s.group(3))
        el[k] = float(s.group(4))
    tag = f"bs{bs} run {i} {cfg}"
    if 30 not in tps:
        problems.append(f"{tag}: incomplete")
        continue
    lib = re.findall(r"collectives via (\S+) \(([^)]*)\)", text)
    ib = len(re.findall(r"NET/IB", text)) + len(
        re.findall(r"NET/OFI Selected provider is \w+", text)
    )
    sock = len(re.findall(r"NET/Socket", text))
    numa = len(re.findall(r"rank_bind\[numa\]", text))
    compact = len(re.findall(r"rank_bind\[compact\]", text))
    if cfg == "stock":
        if lib:
            problems.append(f"{tag}: the package was importable in the stock run")
        if ib == 0:
            problems.append(f"{tag}: no NET/IB or NET/OFI provider line from NCCL")
    elif cfg == "mojo_nccl":
        if not lib or not re.search(r"lib[nr]ccl", lib[0][0]):
            problems.append(f"{tag}: library line says {lib[:1]}")
        if ib == 0:
            problems.append(f"{tag}: no NET/IB or NET/OFI provider line from NCCL")
    elif cfg == "mojo_mojoccl":
        if not lib or "mojoccl" not in lib[0][1]:
            problems.append(f"{tag}: library line says {lib[:1]}")
    if sock:
        problems.append(f"{tag}: NCCL reports NET/Socket ({sock} lines)")
    if numa < WORLD or compact:
        problems.append(f"{tag}: binding numa={numa} compact={compact}")
    runs.setdefault((bs, cfg), []).append(
        (
            statistics.mean(tps[k] for k in range(20, 31)),
            el[1],
            [tps[k] for k in range(20, 31)],
        )
    )


def ci(xs):
    return (
        T[len(xs)] * statistics.stdev(xs) / len(xs) ** 0.5
        if len(xs) > 1
        else float("nan")
    )


print(
    "VERIFICATION:",
    "all runs verified" if not problems else "\n  " + "\n  ".join(problems),
)
for bs in sorted({b for b, _ in runs}, reverse=True):
    ref = [x[0] for x in runs[(bs, "stock")]]
    rm, rc = statistics.mean(ref), ci(ref)
    n = len(ref)
    print(
        f"\nBatch {bs}x1024 per rank, {SETUP}, {MODEL}, bf16 autocast, 30 steps, {n} interleaved rounds after a discarded warm-up; +- is a 95% CI over rounds:\n"
    )
    print(
        f"| stack | mean tokens/s, steps 20-30 | ratio vs {NAMES['stock']} | step 1 (init) |"
    )
    print("|---|---|---|---|")
    for cfg in ("stock", "mojo_nccl", "mojo_mojoccl"):
        r = runs.get((bs, cfg), [])
        t = [x[0] for x in r]
        s1 = [x[1] for x in r]
        if not t:
            print(f"| {NAMES[cfg]} | no runs | | |")
            continue
        tm, tc = statistics.mean(t), ci(t)
        ratio = tm / rm
        rci = ratio * ((tc / tm) ** 2 + (rc / rm) ** 2) ** 0.5
        print(
            f"| {NAMES[cfg]} | {tm:.0f}k +- {tc:.0f}k | {ratio:.3f} +- {rci:.3f} | {statistics.mean(s1):.1f} s |"
        )
    per_step = {
        cfg: [v for x in runs.get((bs, cfg), []) for v in x[2]] for cfg in NAMES
    }
    print(
        "median per-step tokens/s over all rounds: "
        + ", ".join(
            f"{NAMES[c]} {statistics.median(v):.0f}k" for c, v in per_step.items() if v
        )
    )
