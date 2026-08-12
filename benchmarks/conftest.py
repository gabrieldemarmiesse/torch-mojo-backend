"""Performance-regression benchmark suite: pytest wiring.

One pytest test node per benchmark case.  The ratio against stock
PyTorch is stored in benchmarks/baselines.html under the axis-tree path
hardware -> aten op -> dtype -> shape -> layout, derived from the node's own
axes (bench_lib/check.py:_bench_key): op from the test function name
(overridable with @pytest.mark.bench_op), dtype/shape from the
parametrize ids, layout from the layout axis or the sentinel "contig".
Selection is plain pytest (-k, node ids, file paths) — there is no
marker taxonomy here.  Pass/fail and update rules live in
bench_lib/check.py; measurement discipline in bench_lib/measure.py; the
baseline file contract in bench_lib/baselines.py.

Run it serially (no -n): every case takes the GPU flock and interleaves
two legs on the device, so parallel workers would only fight each other.
"""

from __future__ import annotations

import os

os.environ.setdefault("MODULAR_TELEMETRY_ENABLED", "0")

from pathlib import Path

import pytest
import torch
from bench_lib.check import Bench, bench_key, update_mode
from bench_lib.hw import Hardware, detect

# Set to a path to make collection write every benchmark node's baseline
# key there, one per line, and measure nothing.  test_coverage.py drives
# this to reconcile the recorded baselines against the suite: the keys come
# from the real bench_key(), so the reconciliation cannot disagree with
# what a measuring run would write.
KEY_DUMP_ENV = "TORCH_MOJO_BACKEND_BENCH_DUMP_KEYS"


def pytest_collection_finish(session: pytest.Session) -> None:
    dump = os.environ.get(KEY_DUMP_ENV)
    if not dump:
        return
    keys = [
        bench_key(item)
        for item in session.items
        if isinstance(item, pytest.Function) and "bench" in item.fixturenames
    ]
    Path(dump).write_text("".join(f"{key}\n" for key in keys))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "bench_op(name): store this benchmark under the given op token in "
        "baselines.html instead of the test function name (needed when the "
        'aten name is not a Python identifier, e.g. "add.Tensor")',
    )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-baselines",
        nargs="?",
        const="improve",
        default=None,
        choices=("improve", "force"),
        help=(
            "Write measured ratios into benchmarks/baselines.html: new entries "
            "and >8%% improvements. '=force' also accepts >8%% regressions "
            "(after an intentional performance trade-off)."
        ),
    )


@pytest.fixture(scope="session")
def hw() -> Hardware:
    hardware = detect()
    if hardware is None:
        pytest.skip(
            "no accelerator available (set TORCH_MOJO_BACKEND_BENCH_CPU=1 "
            "to benchmark the CPU configuration)"
        )
    return hardware


@pytest.fixture(scope="session")
def mojo_device(hw: Hardware) -> torch.device:
    from torch_mojo_backend import register_mojo_devices

    register_mojo_devices()
    return torch.device("mojo")


@pytest.fixture(autouse=True)
def deterministic_operands() -> None:
    # Same operand data in every process: a reproducible measurement should
    # feed reproducible inputs.  Tested on S1-tf32: data content is NOT the
    # driver of the residual ~2% between-process drift (that is allocator /
    # memory-layout state), but seeding removes it as a variable for the
    # power-bound cases where GEMM device time is weakly data-dependent.
    torch.manual_seed(0)


@pytest.fixture
def bench(
    request: pytest.FixtureRequest, hw: Hardware, mojo_device: torch.device
) -> Bench:
    return Bench(request, hw, mojo_device)


def pytest_terminal_summary(
    terminalreporter, exitstatus: int, config: pytest.Config
) -> None:
    notes = getattr(config, "_bench_notes", [])
    if not notes:
        return
    terminalreporter.section("benchmark baselines")
    for note in notes:
        terminalreporter.write_line(note)
    if update_mode(config) is None and any("NOT recorded" in n for n in notes):
        terminalreporter.write_line(
            "hint: rerun with --update-baselines (or "
            "TORCH_MOJO_BACKEND_BENCH_UPDATE=1) to write these into "
            "benchmarks/baselines.html"
        )
