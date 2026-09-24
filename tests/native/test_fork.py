"""fork() after the mojo device is up (agents_docs/native_backend.md, "Fork").

The MAX runtime is not fork-safe: its worker threads and device contexts do
not exist in a forked child, and a device call there waits forever on a
thread that is gone (measured: `DeviceContext.synchronize` under `_to_copy`,
a futex never signalled). The shim installs a pthread_atfork child handler
at registration so the child gets CUDA's answer instead: `_is_in_bad_fork()`
is True, `torch.manual_seed` skips the device, and any allocation or op
raises a RuntimeError that names the 'spawn' start method. What DataLoader
relies on keeps working: `torch.accelerator.is_available()` and
`device_count()` are reads of registration state, and forked workers that
only touch CPU tensors run normally.
"""

from __future__ import annotations

import gc
import os
import select
import signal
import subprocess
import sys
import textwrap
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from torch_mojo_backend import native
from torch_mojo_backend.native import device_module  # what torch.mojo is

_CHILD_TIMEOUT_S = 120


def _in_forked_child(fn: Callable[[], str], timeout: int = _CHILD_TIMEOUT_S) -> str:
    """fn's return value from a forked child, or a failure if the child hangs
    (which is what this fix turns into an error)."""
    r, w = os.pipe()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # multi-threaded fork
        pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            out = fn()
        except BaseException as e:  # noqa: BLE001 -- reported to the parent, whatever it is
            out = f"EXC {type(e).__name__}: {e}"
        os.write(w, out.encode())
        os._exit(0)
    os.close(w)
    chunks = []
    while True:
        ready, _, _ = select.select([r], [], [], timeout)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            os.close(r)
            pytest.fail(
                f"the forked child hung for {timeout}s; partial output: "
                + b"".join(chunks).decode()
            )
        chunk = os.read(r, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    os.close(r)
    os.waitpid(pid, 0)
    return b"".join(chunks).decode()


def test_device_use_in_a_forked_child_raises_instead_of_hanging(mojo_device: str):
    count = torch.accelerator.device_count()
    assert not device_module._is_in_bad_fork()

    def child() -> str:
        parts = [
            f"bad_fork={device_module._is_in_bad_fork()}",
            f"available={torch.accelerator.is_available()}",
            f"count={torch.accelerator.device_count()}",
        ]
        torch.manual_seed(1)  # skips the device in a bad fork, must not raise
        try:
            torch.ones(3, device=mojo_device)
            parts.append("op=no error")
        except RuntimeError as e:
            parts.append(f"op=raised spawn={'spawn' in str(e)}")
        return " ".join(parts)

    out = _in_forked_child(child)
    assert out == f"bad_fork=True available=True count={count} op=raised spawn=True"
    assert not device_module._is_in_bad_fork()  # the parent is untouched


def test_is_pinned_in_a_forked_child_returns_false(mojo_device: str):
    """Pinned queries refuse both MAX and CUDA runtime access without raising."""
    with device_module.device(mojo_device):
        pinned = torch.empty(8).pin_memory()
        factory_pinned = torch.empty(8, pin_memory=True)
        pageable = torch.empty(8)
        assert pinned.is_pinned()
        assert factory_pinned.is_pinned()

        def child() -> str:
            return str(
                [tensor.is_pinned() for tensor in (pinned, factory_pinned, pageable)]
            )

        assert _in_forked_child(child) == "[False, False, False]"
        assert pinned.is_pinned()
        assert factory_pinned.is_pinned()


def test_forked_pin_queries_cover_empty_views_aliases_and_devices(mojo_device: str):
    with device_module.device(mojo_device):
        base = torch.arange(17).pin_memory()
        tensors = [
            torch.arange(17),
            base,
            base[3:],
            base[17:],
            torch.frombuffer(memoryview(base.numpy())[8:], dtype=torch.int64),
            torch.empty(0).pin_memory(),
            torch.empty(17, pin_memory=True),
            torch.empty(17, device=mojo_device),
            torch.empty(0, device=mojo_device),
            torch.empty(17, device="meta"),
        ]
        expected = [False, True, True, True, True, False, True, False, False, False]
        assert [tensor.is_pinned() for tensor in tensors] == expected

        def child() -> str:
            for _ in range(20):
                if any(tensor.is_pinned() for tensor in tensors):
                    return "inherited storage reported pinned"
            return "all queries false"

        assert _in_forked_child(child) == "all queries false"
        assert [tensor.is_pinned() for tensor in tensors] == expected
        assert torch.equal(base, torch.arange(17))


def test_forked_child_can_destroy_inherited_pinned_storage(mojo_device: str):
    with device_module.device(mojo_device):
        # The list is the only owner in each process. Clearing it really calls
        # the deleter; os._exit alone would hide unsafe runtime destruction.
        owned = [torch.arange(257).pin_memory(), torch.empty(0).pin_memory()]

        def child() -> str:
            owned.clear()
            gc.collect()
            try:
                torch.arange(3).pin_memory()
            except RuntimeError as error:
                return f"destroyed; repin requires spawn={'spawn' in str(error)}"
            return "repin unexpectedly succeeded"

        assert _in_forked_child(child) == "destroyed; repin requires spawn=True"
        assert owned[0].is_pinned()
        assert torch.equal(owned[0], torch.arange(257))


def test_forked_pinned_deleter_does_not_acquire_inherited_mutex(mojo_device: str):
    acquired = Event()
    release = Event()
    shim = native.shim()

    def hold_allocator_lock():
        # Controlled allocator activity: guarantee fork happens while another
        # thread owns the lock, without changing any hook or dereferencing data.
        shim.tmb_lock()
        try:
            acquired.set()
            assert release.wait(30), "parent did not release allocator lock"
        finally:
            shim.tmb_unlock()

    with device_module.device(mojo_device):
        owned = [torch.arange(17).pin_memory()]

        def child() -> str:
            assert not owned[0].is_pinned()
            owned.clear()
            gc.collect()
            return "destroyed without inherited lock"

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(hold_allocator_lock)
            try:
                assert acquired.wait(10), "allocator lock was not acquired"
                assert (
                    _in_forked_child(child, timeout=10)
                    == "destroyed without inherited lock"
                )
            finally:
                release.set()
            future.result(timeout=10)
        assert owned[0].is_pinned()
        assert torch.equal(owned[0], torch.arange(17))


def test_fork_during_concurrent_pinned_allocations_does_not_lock_child(
    mojo_device: str,
):
    active = Event()
    stop = Event()

    def churn():
        with device_module.device(mojo_device):
            while not stop.is_set():
                block = torch.arange(257).pin_memory()
                assert block.is_pinned()
                active.set()

    with device_module.device(mojo_device):
        pinned = torch.arange(17).pin_memory()
        pageable = torch.arange(17)

        def child() -> str:
            return str([pinned.is_pinned(), pageable.is_pinned()])

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(churn) for _ in range(2)]
            try:
                assert active.wait(30), "allocator threads never started"
                for _ in range(3):
                    assert _in_forked_child(child) == "[False, False]"
            finally:
                stop.set()
            for future in futures:
                future.result(timeout=30)
        assert pinned.is_pinned()
        assert torch.equal(pinned, torch.arange(17))


def test_fork_before_import_allows_fresh_child_pinned_allocator(
    mojo_gpu_available: bool,
):
    if not mojo_gpu_available:
        pytest.skip("You do not have a GPU supported by MAX")
    # This outer interpreter has imported neither torch nor MAX before fork.
    script = textwrap.dedent("""
        import os
        import signal

        child = os.fork()
        if child == 0:
            signal.alarm(120)
            import torch
            from torch_mojo_backend import register_mojo_devices
            from torch_mojo_backend.native import device_module

            register_mojo_devices()
            assert not device_module._is_in_bad_fork()
            with device_module.device(0):
                for pinned in (torch.arange(17).pin_memory(),
                               torch.empty(17, device='cpu', pin_memory=True),
                               torch.empty_like(torch.arange(17), device='cpu', pin_memory=True)):
                    assert pinned.is_pinned()
            del pinned
            print('fresh pinning succeeded', flush=True)
            raise SystemExit(0)
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0, status
    """)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONEXECUTABLE")
    }
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "fresh pinning succeeded" in proc.stdout


@pytest.mark.filterwarnings("ignore:This process:DeprecationWarning")
@pytest.mark.parametrize("pin_memory", [False, True])
def test_dataloader_with_forked_workers_after_registration(pin_memory: bool):
    ds = TensorDataset(torch.arange(64.0).view(16, 4))
    loader = DataLoader(
        ds,
        batch_size=4,
        num_workers=2,
        multiprocessing_context="fork",
        pin_memory=pin_memory,
        timeout=_CHILD_TIMEOUT_S,  # a hung worker fails instead of blocking
    )
    assert sum(batch[0].shape[0] for batch in loader) == 16
