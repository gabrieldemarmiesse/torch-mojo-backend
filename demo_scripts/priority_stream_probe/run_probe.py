"""Runner for the priority-stream probe.

Empirically answers: can Modular's pure-Mojo allreduce run on a
raised-priority DeviceStream today, from a --emit shared-lib Python
extension?

Steps:
  1. import torch_mojo_backend FIRST (pins the process so mojo extension
     loads survive later max.nn imports), register mojo devices.
  2. Load priority_probe.so via ExtensionFileLoader.
  3. Query stream_priority_range on both devices.
  4. Minimal claim: capture-free iota kernel via compile_function +
     create_stream(priority=greatest) + DeviceStream.enqueue_function.
  5. SUM allreduce (1-stage and 2-stage wrapper kernels) on priority side
     streams; verify out[i] == in[0] + in[1] on both devices.
  6. Overlap experiment: many-wave spin kernel on the default streams,
     allreduce on side streams at priority 0 vs greatest; compare time
     until the collective completes.
"""

import importlib.machinery
import importlib.util
import os
import sys

PROBE_DIR = os.path.dirname(os.path.abspath(__file__))

import torch  # noqa: E402

from torch_mojo_backend import register_mojo_devices  # noqa: E402


def load_probe():
    path = os.path.join(PROBE_DIR, "priority_probe.so")
    loader = importlib.machinery.ExtensionFileLoader("priority_probe", path)
    spec = importlib.util.spec_from_file_location("priority_probe", path, loader=loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def main() -> int:
    register_mojo_devices()

    from torch_mojo_backend import eager_kernels
    from torch_mojo_backend.mojo_device import deferred_compile
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
        get_ordered_accelerators,
    )

    gpu_count = sum(acc.label == "gpu" for acc in get_ordered_accelerators())
    if gpu_count < 2:
        print("FAIL: need >= 2 MAX GPUs, found", gpu_count)
        return 1

    # Force the backend's own comm_ops extension resident first: it pins the
    # process (modular-5) and its signal_header_bytes sizes the buffers.
    header = int(eager_kernels.comm_ops.signal_header_bytes())
    print(f"signal header bytes: {header}")

    probe = load_probe()
    print("probe .so loaded OK")

    world = 2
    numel = 1 << 22  # 4 Mi float32 = 16 MiB per rank
    nbytes = numel * 4

    torch.manual_seed(0)
    cpu_inputs = [torch.randn(numel, dtype=torch.float32) for _ in range(world)]
    inputs = [cpu_inputs[i].to(f"mojo:{i}") for i in range(world)]
    outputs = [
        torch.zeros(numel, dtype=torch.float32, device=f"mojo:{i}")
        for i in range(world)
    ]

    # Signal buffers: header + world * payload, zeroed, drained + synced
    # before any collective (mirrors distributed._Signals).
    sig_nbytes = header + world * nbytes
    signals = [
        torch.zeros(sig_nbytes, dtype=torch.uint8, device=f"mojo:{i}")
        for i in range(world)
    ]
    deferred_compile.drain()
    for i in range(world):
        torch.mojo.synchronize(i)

    devices = [t._device for t in inputs]
    ctx_ptrs = tuple(eager_kernels._ctx_ptr(d) for d in devices)
    in_ptrs = tuple(t._ptr for t in inputs)
    out_ptrs = tuple(t._ptr for t in outputs)
    sig_ptrs = tuple(t._ptr for t in signals)

    expected = cpu_inputs[0] + cpu_inputs[1]
    failures = []

    # --- 1. priority range -------------------------------------------------
    ranges = [probe.stream_priority_range(ctx_ptrs[i]) for i in range(world)]
    print(f"stream priority ranges (least, greatest): {ranges}")
    greatest = int(ranges[0][1])

    # --- 2. minimal claim: capture-free kernel on a priority stream --------
    scratch = torch.zeros(1024, dtype=torch.float32, device="mojo:0")
    deferred_compile.drain()
    torch.mojo.synchronize(0)
    probe.probe_simple(ctx_ptrs[0], scratch._ptr, 1024, greatest)
    got = scratch.cpu()
    want = torch.arange(1024, dtype=torch.float32)
    ok = torch.equal(got, want)
    print(
        f"probe_simple (iota on priority={greatest} stream): {'PASS' if ok else 'FAIL'}"
    )
    if not ok:
        failures.append("probe_simple")

    # --- 3. allreduce correctness on priority streams ----------------------
    for use_2stage in (False, True):
        for out in outputs:
            out.zero_()
        deferred_compile.drain()
        for i in range(world):
            torch.mojo.synchronize(i)
        t = probe.priority_all_reduce(
            in_ptrs, out_ptrs, sig_ptrs, ctx_ptrs, numel, (use_2stage, greatest, 0, 0)
        )
        for i in range(world):
            torch.mojo.synchronize(i)
        stage = "2-stage" if use_2stage else "1-stage"
        all_ok = True
        for i in range(world):
            got = outputs[i].cpu()
            match = torch.allclose(got, expected, rtol=1e-5, atol=1e-5)
            all_ok &= match
            if not match:
                diff = (got - expected).abs().max().item()
                print(f"  rank {i}: max abs diff {diff}")
        print(
            f"allreduce {stage} on priority={greatest} streams: "
            f"{'PASS' if all_ok else 'FAIL'} "
            f"(comm done in {float(t[0]):.2f} ms)"
        )
        if not all_ok:
            failures.append(f"allreduce-{stage}")

    # --- 4. does priority matter? -----------------------------------------
    # Spin kernel: many waves of short blocks on the default stream; the
    # collective (few blocks) is enqueued on the side stream after the ready
    # event. With raised priority its blocks should jump the wave queue.
    spin_blocks = 32768
    spin_iters = 20000
    print(
        f"\noverlap experiment: spin({spin_blocks} blocks x {spin_iters} "
        f"iters) on default stream, allreduce (1-stage) on side stream"
    )
    results = {}
    for prio in (0, greatest):
        # Warmup (compiles cached, signals already used -> counters are
        # monotonic, reuse is fine).
        probe.priority_all_reduce(
            in_ptrs, out_ptrs, sig_ptrs, ctx_ptrs, numel, (False, prio, 0, 0)
        )
        times = []
        for _ in range(3):
            t = probe.priority_all_reduce(
                in_ptrs,
                out_ptrs,
                sig_ptrs,
                ctx_ptrs,
                numel,
                (False, prio, spin_blocks, spin_iters),
            )
            times.append((float(t[0]), float(t[1])))
        best = min(times, key=lambda x: x[0])
        results[prio] = best
        print(
            f"  priority={prio:>3}: comm done at {best[0]:8.2f} ms, "
            f"spin done at {best[1]:8.2f} ms  "
            f"(all runs: {[f'{a:.1f}/{b:.1f}' for a, b in times]})"
        )
    for i in range(world):
        torch.mojo.synchronize(i)

    t0_comm, t0_all = results[0]
    tp_comm, tp_all = results[greatest]
    if tp_comm < 0.5 * t0_comm:
        print(
            f"priority effect: YES - greatest priority completes the "
            f"collective {t0_comm / tp_comm:.1f}x sooner than priority 0"
        )
    else:
        print(
            f"priority effect: WEAK/NONE - {t0_comm:.2f} ms (prio 0) vs "
            f"{tp_comm:.2f} ms (prio {greatest})"
        )

    # Sanity: outputs still correct after the overlap runs.
    ok = all(
        torch.allclose(outputs[i].cpu(), expected, rtol=1e-5, atol=1e-5)
        for i in range(world)
    )
    print(f"post-overlap correctness: {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("post-overlap")

    print()
    if failures:
        print(f"FAIL: {failures}")
        return 1
    print("PASS: allreduce ran on raised-priority DeviceStreams with correct results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
