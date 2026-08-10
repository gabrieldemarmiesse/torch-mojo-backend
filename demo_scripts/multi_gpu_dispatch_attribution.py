"""Attribute the multi-GPU dispatch ceiling: interpreter-side or runtime-side?

`multi_gpu_dispatch_bench.py` measured ~0.5 scaling efficiency at 8 GPUs,
but both of its modes (roundrobin, threads) run in ONE process, so the GIL,
per-op Python dispatch cost and MAX AsyncRT/driver contention are
confounded. This bench runs the SAME per-device op mix as N separate
processes, one GPU each (`CUDA_VISIBLE_DEVICES=i`), started together
through a stdin barrier:

- processes scale ~1.0  -> the ceiling is interpreter-side (GIL / per-op
  Python cost / shared locks); GIL-release, nogil and lock-sharding levers
  pay off.
- processes also degrade -> the ceiling is runtime/driver-side; only
  "fewer launches" directions pay (SPMD fan-out in Mojo, record/replay,
  whole-step multi-device graphs).

Example:
    MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas \
        uv run demo_scripts/multi_gpu_dispatch_attribution.py --max-devices 8
"""

import argparse
import os
import subprocess
import sys
import time

from multi_gpu_dispatch_bench import (
    OPS_PER_STEP,
    make_state,
    run_roundrobin,
    run_threads,
    step,
    sync_all,
)


def child(steps: int, batch: int, dim: int) -> None:
    """One rank: warm up on the only visible GPU, then run timed steps.

    Prints READY, blocks on stdin for the GO line (the cross-process start
    barrier), runs, prints the elapsed wall seconds.
    """
    from torch_mojo_backend import register_mojo_devices

    register_mojo_devices()
    state = make_state("mojo:0", batch, dim)
    # A fresh process pays per-variant .so loads, allocator growth and
    # first-launch device sync on its first pass — run a whole warmup
    # round so the timed section measures steady-state dispatch only
    # (the in-process modes get the same treatment from the parent's
    # warmup + earlier legs).
    for _ in range(max(5, steps // 5)):
        step(state)
    sync_all(1)
    print("READY", flush=True)
    if sys.stdin.readline().strip() != "GO":
        raise SystemExit("expected GO")
    sync_all(1)
    start = time.perf_counter()
    for _ in range(steps):
        step(state)
    sync_all(1)
    print(f"ELAPSED {time.perf_counter() - start}", flush=True)


def run_processes(num_devices: int, steps: int, batch: int, dim: int) -> float:
    """N single-GPU processes running the op mix concurrently; wall time is
    the slowest rank (they start together off the stdin barrier)."""
    env = dict(os.environ)
    procs = []
    for i in range(num_devices):
        rank_env = env | {"CUDA_VISIBLE_DEVICES": str(i)}
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--child",
                    f"--steps={steps}",
                    f"--batch={batch}",
                    f"--dim={dim}",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=rank_env,
                text=True,
                cwd=os.path.dirname(__file__),
            )
        )
    for proc in procs:
        line = proc.stdout.readline().strip()
        if line != "READY":
            raise SystemExit(f"child failed before READY: {line!r}")
    start = time.perf_counter()
    for proc in procs:
        proc.stdin.write("GO\n")
        proc.stdin.flush()
    elapsed = []
    for proc in procs:
        line = proc.stdout.readline().strip()
        if not line.startswith("ELAPSED "):
            raise SystemExit(f"child failed after GO: {line!r}")
        elapsed.append(float(line.split()[1]))
        proc.stdin.close()
        proc.wait()
    wall = time.perf_counter() - start
    # Prefer the barrier-to-last-report wall time (includes any straggler),
    # but report per-rank spread too.
    spread = max(elapsed) - min(elapsed)
    print(f"    per-rank elapsed spread: {spread * 1e3:.0f} ms")
    return wall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--max-devices", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument(
        "--small",
        action="store_true",
        help="dispatch-bound tiny shapes: per-op host overhead probe",
    )
    args = parser.parse_args()

    if args.small:
        args.batch, args.dim = 32, 64

    if args.child:
        child(args.steps, args.batch, args.dim)
        return

    from torch_mojo_backend import register_mojo_devices
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
        get_ordered_accelerators,
    )

    register_mojo_devices()
    gpu_count = sum(acc.label == "gpu" for acc in get_ordered_accelerators())
    if gpu_count == 0:
        raise SystemExit("no MAX GPU available")
    max_devices = min(args.max_devices or gpu_count, gpu_count)
    device_counts = sorted({1, 2, 4, max_devices})
    device_counts = [n for n in device_counts if n <= max_devices]

    print(
        f"op mix: {OPS_PER_STEP} dispatched ops/step, "
        f"batch={args.batch} dim={args.dim}, {args.steps} steps/device"
    )

    all_states = [
        make_state(f"mojo:{i}", args.batch, args.dim) for i in range(max_devices)
    ]
    for state in all_states:
        step(state)
    sync_all(max_devices)

    header = (
        f"{'mode':<12}{'devices':>8}{'wall s':>10}{'steps/s/dev':>13}"
        f"{'us/op (host)':>14}{'efficiency':>12}"
    )
    print(header)
    print("-" * len(header))
    for mode in ("roundrobin", "threads", "processes"):
        baseline = None
        for n in device_counts:
            if mode == "roundrobin":
                elapsed = run_roundrobin(all_states[:n], args.steps)
            elif mode == "threads":
                elapsed = run_threads(all_states[:n], args.steps)
            else:
                elapsed = run_processes(n, args.steps, args.batch, args.dim)
            steps_per_s = args.steps / elapsed
            us_per_op = elapsed / (args.steps * OPS_PER_STEP * n) * 1e6
            baseline = baseline if baseline is not None else elapsed
            print(
                f"{mode:<12}{n:>8}{elapsed:>10.3f}{steps_per_s:>13.1f}"
                f"{us_per_op:>14.2f}{baseline / elapsed:>12.2f}"
            )


if __name__ == "__main__":
    main()
