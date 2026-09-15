"""Peer-copy fault/route observability; every hook runs in a fresh process.

Tensor operations use public torch APIs. The cached test-only native hook
observes allocator retirement and injects failures after a real DMA. A pipe
holds the destination stream to distinguish event ordering from a host wait.

TORCH_MOJO_BACKEND_TEST_PEER_COPY accepts comma-separated modes: trace (routes),
host (force staging), enable_error (fail peer enable), audit (allocations),
submit_error (fail after DMA), drain_error (also fail cleanup), gate (hold DMA
on TORCH_MOJO_BACKEND_TEST_PEER_GATE_FD). Route and allocation logs accompany
every mode; unset disables all hooks. Settings are cached at initialization.
"""

import ctypes
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import pytest
import torch

from torch_mojo_backend import get_accelerators, register_mojo_devices


@pytest.fixture(autouse=True)
def two_gpus():
    if len(get_accelerators()) - 1 < 2:
        pytest.skip("requires two mojo GPUs")


def _run(scenario: str, mode: str, *args: str) -> str:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), scenario, *args],
        cwd=Path(__file__).resolve().parents[2],
        env={
            **os.environ,
            "TORCH_MOJO_BACKEND_TEST_PEER_COPY": f"trace,{mode}",
            "PYTHONUNBUFFERED": "1",
        },
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode == 77:
        pytest.skip("requires a CUDA peer-capable pair for controlled DMA tests")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "P2P_DONE" in result.stdout, result.stdout
    return result.stdout


def _cases(output: str) -> list[list[str]]:
    return [
        part.split("P2P_END", 1)[0].splitlines()
        for part in output.split("P2P_CASE\n")[1:]
    ]


@pytest.mark.parametrize("mode", ["submit_error", "drain_error"])
@pytest.mark.parametrize("layout", ["contiguous", "transpose"])
@pytest.mark.parametrize("operation", ["to", "copy"])
def test_peer_partial_submission_lifetime(mode: str, layout: str, operation: str):
    output = _run("fault", mode, layout, operation)
    cases = _cases(output)
    assert len(cases) == 8, output
    for lines in cases:
        submitted = next(
            i for i, line in enumerate(lines) if line.startswith("P2P_SUBMITTED ")
        )
        _, dst, src = lines[submitted].split()
        tail = lines[submitted + 1 :]
        original = next(
            line.split()[1] for line in lines if line.startswith("P2P_ORIGINAL ")
        )
        drains = [i for i, line in enumerate(tail) if line.startswith("P2P_DRAINED ")]
        frees = [i for i, line in enumerate(tail) if line.startswith("P2P_FREE ")]
        if mode == "submit_error":
            assert len(drains) == 2, "missing error cleanup drains:\n" + "\n".join(
                lines
            )
            assert frees and max(drains) < min(frees), "storage retired before cleanup"
            released = {
                line.split()[2] for line in tail if line.startswith("P2P_FREE ")
            }
            assert {original, src, dst} <= released, (
                "source/staging/destination not released"
            )
        else:
            assert len(drains) == 1, "source drain must still be attempted"
            assert not frees, "storage reused after an unprovable completion"
            retained = {
                line.split()[2] for line in tail if line.startswith("P2P_RETAIN ")
            }
            assert {original, src, dst} <= retained, (
                "source/staging/destination not quarantined"
            )


@pytest.mark.parametrize("layout", ["contiguous", "transpose"])
def test_peer_failed_to_releases_destination(layout: str):
    output = _run("fault", "submit_error", layout, "to")
    cases = _cases(output)
    assert len(cases) == 8, output
    live = {}
    for lines in cases:
        for line in lines:
            fields = line.split()
            if fields[0] == "P2P_ALLOC":
                _, device, ptr, size = fields
                assert (device, ptr) not in live, "allocator reused a live allocation"
                live[device, ptr] = int(size)
            elif fields[0] == "P2P_FREE":
                live.pop((fields[1], fields[2]))
        assert not live, f"failed .to leaked allocated bytes: {live}"


def test_peer_contiguous_to_borrows_source():
    output = _run("borrow", "audit")
    cases = _cases(output)
    assert len(cases) == 2, output
    for lines in cases:
        source = next(
            line.split()[1] for line in lines if line.startswith("P2P_SOURCE_DEVICE ")
        )
        allocations = [line.split() for line in lines if line.startswith("P2P_ALLOC ")]
        assert any(fields[1] != source for fields in allocations), (
            "missing destination allocation observation"
        )
        assert not [fields for fields in allocations if fields[1] == source], (
            "contiguous same-dtype .to allocated source-side storage: "
            + "\n".join(lines)
        )


@pytest.mark.parametrize("operation", ["to", "copy"])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_peer_returns_before_destination_completes(operation: str, non_blocking: bool):
    _run("gate", "gate", operation, str(int(non_blocking)))


def _require_peer_pair():
    # Independent driver query; a missing direct backend route must not skip.
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        sys.exit(77)
    assert driver.cuInit(0) == 0
    for dst, src in [(1, 0), (0, 1)]:
        capable = ctypes.c_int()
        assert driver.cuDeviceCanAccessPeer(ctypes.byref(capable), dst, src) == 0
        if not capable.value:
            sys.exit(77)


def _fault(layout: str, operation: str):
    for src, dst in [("mojo:0", "mojo:1"), ("mojo:1", "mojo:0")]:
        owner, current, target = (torch.Stream(device=d) for d in (src, src, dst))
        cpu = (torch.arange(1021 * 4093, dtype=torch.float32) % 251).reshape(1021, 4093)
        for _ in range(4):
            print("P2P_CASE", flush=True)
            with owner:
                source = cpu.to(src)
            print("P2P_ORIGINAL", source.data_ptr(), flush=True)
            current.wait_stream(owner)
            with current, target:
                expected = cpu
                if layout == "transpose":
                    source, expected = source.t(), cpu.t()
                out = torch.empty(expected.shape, device=dst)
                with pytest.raises(RuntimeError, match="injected failure after DMA"):
                    if operation == "to":
                        source.to(dst, non_blocking=True)
                    else:
                        out.copy_(source, non_blocking=True)
                del source
                # Check the partially submitted copy after dropping its source.
                if operation == "copy":
                    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                del out
            with owner:
                torch.empty_like(cpu, device=src).fill_(-77)
            owner.synchronize()
            target.synchronize()
            print("P2P_END", flush=True)


def _borrow():
    cpu = torch.arange(8 * 1024 * 1024, dtype=torch.float32)
    for src, dst in [("mojo:0", "mojo:1"), ("mojo:1", "mojo:0")]:
        source = cpu.to(src)
        print("P2P_CASE", flush=True)
        print("P2P_SOURCE_DEVICE", source.device.index, flush=True)
        actual = source.to(dst)
        print("P2P_END", flush=True)
        torch.testing.assert_close(actual.cpu(), cpu, rtol=0, atol=0)
        del source, actual


def _gated_transfer(
    source: torch.Tensor,
    target: torch.Tensor,
    src_stream: torch.Stream,
    dst_stream: torch.Stream,
    operation: str,
    non_blocking: bool,
) -> tuple[torch.Tensor, torch.Event]:
    with src_stream, dst_stream:
        if operation == "to":
            target = source.to(target.device, non_blocking=non_blocking)
        else:
            target.copy_(source, non_blocking=non_blocking)
        done = torch.Event()
        done.record(dst_stream)
        return target, done


def _gate(operation: str, non_blocking: bool, write_fd: int):
    cpu = torch.arange(1024 * 1024, dtype=torch.float32)
    for src, dst in [("mojo:0", "mojo:1"), ("mojo:1", "mojo:0")]:
        a, b = torch.Stream(device=src), torch.Stream(device=dst)
        with a, b:
            source, target = cpu.to(src), torch.empty_like(cpu, device=dst)
        # Warm the exact op before the deadline, including a cold JIT cache.
        os.write(write_fd, b"w")
        warm, event = _gated_transfer(source, target, a, b, operation, non_blocking)
        event.synchronize()
        del warm, event
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _gated_transfer, source, target, a, b, operation, non_blocking
            )
            returned = False
            try:
                try:
                    actual, done = future.result(timeout=15)
                    returned = True
                except TimeoutError:
                    actual = target
                if returned:
                    assert not done.query(), "destination gate was not active"
            finally:
                os.write(write_fd, b"x")
            # Release even on regression; the old host wait can now finish.
            future.result(timeout=30)
        assert returned, "peer transfer waited on the host for destination completion"
        with b:
            torch.testing.assert_close(actual.cpu(), cpu, rtol=0, atol=0)
            assert actual[-1].item() == cpu[-1].item()


if __name__ == "__main__":
    scenario = sys.argv[1]
    read_fd, write_fd = os.pipe()
    os.environ["TORCH_MOJO_BACKEND_TEST_PEER_GATE_FD"] = str(read_fd)
    register_mojo_devices()
    _require_peer_pair()
    # The native setting must stay cached after initialization.
    os.environ.pop("TORCH_MOJO_BACKEND_TEST_PEER_COPY")
    if scenario == "fault":
        _fault(sys.argv[2], sys.argv[3])
    elif scenario == "borrow":
        _borrow()
    else:
        _gate(sys.argv[2], bool(int(sys.argv[3])), write_fd)
    os.close(read_fd)
    os.close(write_fd)
    print("P2P_DONE", flush=True)
