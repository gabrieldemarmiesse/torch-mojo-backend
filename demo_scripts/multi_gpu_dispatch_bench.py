"""Host-dispatch budget microbench for single-process multi-GPU eager mode.

One Python process drives N mojo GPUs with a transformer-block-flavored op
mix (M0 of docs/multi_gpu_training_plan.md). If host dispatch keeps every
queue fed, wall time for N devices doing the same per-device work stays flat
(efficiency ~= 1.0); efficiency well below 1.0 on realistic shapes means the
GIL/dispatch path is the bottleneck and M2's thread-per-rank DDP would not
scale.

Two issue orders are measured:
- roundrobin: one host thread interleaves op issue across devices (upper
  bound on a single dispatcher feeding N queues).
- threads: one host thread per device (the M2 DDP execution model; measures
  GIL-contended dispatch).

Use --small to switch to dispatch-bound tiny shapes, which turns the
benchmark into a direct per-op host-overhead probe.

Example:
    MODULAR_NVPTX_COMPILER_PATH=/usr/local/cuda/bin/ptxas \
        uv run demo_scripts/multi_gpu_dispatch_bench.py --max-devices 8
"""

import argparse
import threading
import time

import torch
import torch.nn.functional as F

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.mojo_device.torch_mojo_tensor import get_ordered_accelerators

OPS_PER_STEP = 7  # mm, add, gelu, mm, add, layer_norm, add


def make_state(device: str, batch: int, dim: int) -> dict[str, torch.Tensor]:
    generator_state = torch.random.get_rng_state()
    torch.manual_seed(0)
    state = {
        "x": torch.randn(batch, dim),
        "w1": torch.randn(dim, 4 * dim) * 0.02,
        "b1": torch.zeros(4 * dim),
        "w2": torch.randn(4 * dim, dim) * 0.02,
        "b2": torch.zeros(dim),
        "ln_w": torch.ones(dim),
        "ln_b": torch.zeros(dim),
    }
    torch.random.set_rng_state(generator_state)
    return {k: v.to(device) for k, v in state.items()}


def step(state: dict[str, torch.Tensor]) -> torch.Tensor:
    x = state["x"]
    h = x @ state["w1"]
    h = h + state["b1"]
    h = F.gelu(h)
    h = h @ state["w2"]
    h = h + state["b2"]
    h = F.layer_norm(h, (h.shape[-1],), state["ln_w"], state["ln_b"])
    x = x + h
    state["x"] = x
    return x


def sync_all(num_devices: int) -> None:
    for i in range(num_devices):
        torch.mojo.synchronize(i)


def run_roundrobin(states: list[dict[str, torch.Tensor]], steps: int) -> float:
    num_devices = len(states)
    sync_all(num_devices)
    start = time.perf_counter()
    for _ in range(steps):
        for state in states:
            step(state)
    sync_all(num_devices)
    return time.perf_counter() - start


def run_threads(states: list[dict[str, torch.Tensor]], steps: int) -> float:
    num_devices = len(states)
    sync_all(num_devices)
    barrier = threading.Barrier(num_devices + 1)

    def worker(state: dict[str, torch.Tensor]) -> None:
        barrier.wait()  # start together
        for _ in range(steps):
            step(state)
        barrier.wait()  # everyone done issuing

    threads = [
        threading.Thread(target=worker, args=(state,), daemon=True) for state in states
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    start = time.perf_counter()
    barrier.wait()
    for thread in threads:
        thread.join()
    sync_all(num_devices)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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

    register_mojo_devices()
    gpu_count = sum(acc.label == "gpu" for acc in get_ordered_accelerators())
    if gpu_count == 0:
        raise SystemExit("no MAX GPU available")
    max_devices = min(args.max_devices or gpu_count, gpu_count)

    device_counts = sorted({1, 2, max_devices} | {n for n in (4,) if n <= max_devices})
    device_counts = [n for n in device_counts if n <= max_devices]

    print(
        f"op mix: {OPS_PER_STEP} dispatched ops/step, "
        f"batch={args.batch} dim={args.dim}, {args.steps} steps/device"
    )

    all_states = [
        make_state(f"mojo:{i}", args.batch, args.dim) for i in range(max_devices)
    ]
    # Warmup: one step everywhere compiles every kernel per device.
    for state in all_states:
        step(state)
    sync_all(max_devices)

    baselines: dict[str, float] = {}
    header = (
        f"{'mode':<12}{'devices':>8}{'wall s':>10}{'steps/s/dev':>13}"
        f"{'us/op (host)':>14}{'efficiency':>12}"
    )
    print(header)
    print("-" * len(header))
    for mode, runner in (("roundrobin", run_roundrobin), ("threads", run_threads)):
        for n in device_counts:
            states = all_states[:n]
            elapsed = runner(states, args.steps)
            steps_per_s = args.steps / elapsed
            total_ops = args.steps * OPS_PER_STEP * n
            us_per_op = elapsed / total_ops * 1e6
            baseline = baselines.setdefault(mode, elapsed)
            efficiency = baseline / elapsed
            print(
                f"{mode:<12}{n:>8}{elapsed:>10.3f}{steps_per_s:>13.1f}"
                f"{us_per_op:>14.2f}{efficiency:>12.2f}"
            )


if __name__ == "__main__":
    main()
