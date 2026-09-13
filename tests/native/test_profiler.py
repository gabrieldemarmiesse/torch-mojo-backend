"""Profiling on the mojo device: the shim registers torch's PrivateUse1
ProfilerStubs over the backend's timed events (docs/native_backend.md,
"Profiling"), so the legacy profiler reports device time per op; the Kineto
profiler records the CPU-side timeline and exports a Chrome trace."""

import json

import torch
from torch.profiler import ProfilerActivity, profile


def _mul_loop(device: str):
    a = torch.ones(512, 512, device=device)
    b = torch.full((512, 512), 2.0, device=device)
    for _ in range(4):
        a = a * b
    torch.accelerator.synchronize()


def test_legacy_profiler_reports_device_time_per_op(mojo_gpu):
    _mul_loop(mojo_gpu)  # warm the kernel build outside the profiled region
    with torch.autograd.profiler.profile(use_device="mojo") as prof:
        _mul_loop(mojo_gpu)
    rows = {e.key: e for e in prof.key_averages()}
    assert "aten::mul" in rows
    mul = rows["aten::mul"]
    assert mul.count >= 4
    assert mul.self_device_time_total > 0
    table = prof.key_averages().table(sort_by="self_device_time_total", row_limit=5)
    assert "Self MOJO" in table


def test_kineto_profiler_records_ops_and_exports_a_trace(mojo_gpu, tmp_path):
    _mul_loop(mojo_gpu)
    activities = [ProfilerActivity.CPU]
    if hasattr(ProfilerActivity, "PrivateUse1"):
        activities.append(ProfilerActivity.PrivateUse1)
    with profile(activities=activities) as prof:
        _mul_loop(mojo_gpu)
    names = {e.key for e in prof.key_averages()}
    assert "aten::mul" in names
    trace = tmp_path / "trace.json"
    prof.export_chrome_trace(str(trace))
    events = json.loads(trace.read_text())["traceEvents"]
    assert any(e.get("name") == "aten::mul" for e in events)
