"""Unit tests for basic native mojo device functionality.

The device is a real PrivateUse1 backend (see docs/native_backend.md):
tensors are plain, unwrapped `torch.Tensor`s over MAX device memory, with no
`TorchMojoTensor` wrapper, no `_holder`/`_ptr`/`_mojo_strides` payload
attributes, and no Python-level transfer bookkeeping (`_PENDING_H2D`,
dlpack pending exports, `cuda_peer`) to introspect -- all of that lived in
the old eager Python path and is gone. Tests that used to poke at those
internals are rewritten here at the public torch API level, or dropped with
a one-line reason (see the docstring of each removed test's replacement, and
the migration report).
"""

import ctypes
import gc
import io
import math
import multiprocessing
import os
import subprocess
import sys
import textwrap
import time
import warnings
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from threading import Barrier, Event, Timer
from typing import Generic, NamedTuple, TypeVar

import numpy as np
import pytest
import torch
from torch.optim.optimizer import _default_to_fused_or_foreach
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataloader import _MultiProcessingDataLoaderIter

from tests.native.conftest import side_stream_or_skip, skip_if_metal
from torch_mojo_backend import (
    get_accelerators,
    mojo_backend,
    native,
    register_mojo_devices,
)
from torch_mojo_backend.native import device_module

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_max_device():
    """Setup mojo_device for all tests"""
    register_mojo_devices()


def test_mojo_is_the_default_torch_accelerator():
    assert torch.accelerator.current_accelerator(check_available=True) == torch.device(
        "mojo"
    )


def test_torch_accelerator_synchronize_dispatches_and_validates_device():
    original_device = device_module.current_device()
    try:
        torch.accelerator.synchronize()
        torch.accelerator.synchronize("mojo")
        torch.accelerator.synchronize(0)
        if torch.accelerator.device_count() > 1:
            device_module.set_device(1)
            torch.accelerator.synchronize()
    finally:
        device_module.set_device(original_device)

    with pytest.raises(ValueError, match="doesn't match the current accelerator mojo"):
        torch.accelerator.synchronize("cpu")


def test_tensor_to_max_device(mojo_device):
    """Test converting regular tensor to mojo_device"""
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    mojo_tensor = cpu_tensor.to(mojo_device)
    assert mojo_tensor.device.type == "mojo"
    assert mojo_tensor.shape == (3,)
    assert mojo_tensor.dtype == torch.float32


def test_max_tensor_to_cpu(mojo_device):
    """Test converting MaxTensor back to CPU"""
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    mojo_tensor = cpu_tensor.to(mojo_device)
    result = mojo_tensor.to("cpu")
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, cpu_tensor)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::arange.start_out")
def test_factory_arange(mojo_device):
    """Test torch.arange with mojo_device"""
    tensor = torch.arange(5, device=mojo_device)
    assert tensor.device.type == "mojo"
    assert tensor.shape == (5,)
    cpu_result = tensor.to("cpu")
    expected = torch.arange(5)
    torch.testing.assert_close(cpu_result, expected)


@pytest.mark.xfail(
    strict=False, reason="op not ported yet: aten::uniform_ (torch.rand's RNG kernel)"
)
def test_factory_rand(mojo_device):
    """Test torch.rand with mojo_device"""
    tensor = torch.rand(3, 4, device=mojo_device)
    assert tensor.device.type == "mojo"
    assert tensor.shape == (3, 4)
    cpu_result = tensor.to("cpu")
    assert torch.all(cpu_result >= 0)
    assert torch.all(cpu_result <= 1)


def test_factory_empty(mojo_device):
    """Test torch.empty with mojo_device"""
    tensor = torch.empty(2, 3, device=mojo_device)
    assert tensor.device.type == "mojo"
    assert tensor.shape == (2, 3)


def test_device_string_variations():
    """Test different mojo device string formats"""
    t1 = torch.tensor([1.0]).to("mojo")
    assert t1.device.type == "mojo"
    t2 = torch.tensor([1.0]).to("mojo:0")
    assert t2.device.type == "mojo"


def test_indexless_mojo_device_uses_and_restores_current_device():
    """An indexless mojo target follows the current device (see also the
    more focused version of this test in test_mojo_device_runtime.py)."""
    if device_module.device_count() < 2:
        pytest.skip("requires two Mojo devices, including the MAX CPU device")

    original_index = device_module.current_device()
    alternate_index = (original_index + 1) % device_module.device_count()
    try:
        device_module.set_device(alternate_index)
        empty_tensor = torch.empty(1, device="mojo")
        assert empty_tensor.device.type == "mojo"
        assert empty_tensor.device.index == alternate_index
    finally:
        device_module.set_device(original_index)
    assert device_module.current_device() == original_index


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::abs.out (torch's tensor-repr formatter "
    "needs it, along with isfinite/masked_select, to choose a print style)",
)
def test_tensor_properties(mojo_device):
    """A native mojo tensor is a plain torch.Tensor: shape/dtype/device and
    the standard torch repr all just work, with no custom wrapper needed."""
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    mojo_tensor = original.to(mojo_device)

    assert mojo_tensor.shape == (2, 2)
    assert mojo_tensor.dtype == torch.float64
    assert mojo_tensor.device == torch.device(mojo_device)

    repr_str = repr(mojo_tensor)
    assert mojo_device in repr_str
    assert "float64" in repr_str


def test_round_trip_conversion(mojo_device):
    """Test CPU -> mojo_device -> CPU round trip"""
    original = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mojo_tensor = original.to(mojo_device)
    result = mojo_tensor.to("cpu")
    torch.testing.assert_close(result, original)


def test_non_blocking_cpu_to_mojo_transfers(mojo_device):
    """Both to() and copy_() honor PyTorch's non_blocking transfer API."""
    source = torch.arange(4096, dtype=torch.float32)

    via_to = source.to(mojo_device, non_blocking=True)
    via_copy = torch.empty_like(via_to)
    via_copy.copy_(source, non_blocking=True)

    torch.accelerator.synchronize(mojo_device)
    torch.testing.assert_close(via_to.cpu(), source)
    torch.testing.assert_close(via_copy.cpu(), source)


def test_non_blocking_cpu_source_lifetime(mojo_device):
    """An upload retains its snapshot after the temporary CPU source dies."""
    source = torch.arange(1 << 20, dtype=torch.int32)
    expected = source.clone()
    uploaded = source.to(mojo_device, non_blocking=True)
    assert uploaded.device.type == "mojo"

    del source
    # Encourage the CPU allocator to reuse the released storage while the H2D
    # operation may still be queued.
    for _ in range(8):
        torch.empty_like(expected).fill_(-1)

    torch.accelerator.synchronize(mojo_device)
    torch.testing.assert_close(uploaded.cpu(), expected)


def test_non_blocking_h2d_does_not_drain_prior_gpu_work(mojo_gpu: str):
    """An async upload returns without waiting for older default-stream work."""
    a = torch.full((4096, 4096), 1.0, device=mojo_gpu)
    b = torch.full((4096, 4096), 2.0, device=mojo_gpu)
    torch.accelerator.synchronize(mojo_gpu)

    # Establish a conservative duration for the work placed ahead of H2D:
    # enough launches that the device time dwarfs the enqueue cost.
    def burst() -> torch.Tensor:
        out = a * b
        for _ in range(31):
            out = out * b
        return out

    # The first staged upload of a process pays the pinned-buffer setup
    # (hundreds of ms); the steady state is what the check is about.
    _ = torch.arange(16).to(mojo_gpu, non_blocking=True)
    _ = burst()
    torch.accelerator.synchronize(mojo_gpu)
    reference_seconds, return_seconds = [], []
    source = torch.arange(4096)
    for trial in range(4):
        # Alternate reference/transfer order (ABBA), then compare medians.
        for reference in (True, False) if trial % 2 == 0 else (False, True):
            if reference:
                started = time.perf_counter()
                _ = burst()
                torch.accelerator.synchronize(mojo_gpu)
                reference_seconds.append(time.perf_counter() - started)
            else:
                delayed = burst()
                started = time.perf_counter()
                uploaded = source.to(mojo_gpu, non_blocking=True)
                return_seconds.append(time.perf_counter() - started)
                torch.accelerator.synchronize(mojo_gpu)
                torch.testing.assert_close(uploaded.cpu(), source)
                assert delayed.shape == (4096, 4096)

    assert median(return_seconds) < median(reference_seconds) * 0.5


def _empty_mojo_pinned_like(source: torch.Tensor) -> torch.Tensor:
    """Use our allocator even when a CUDA wheel's factories prefer CUDA."""
    if torch.cuda.is_available():
        return torch.empty_like(source).pin_memory()
    return torch.empty_like(source, pin_memory=True)


@pytest.mark.parametrize("free_immediately", [False, True])
@pytest.mark.parametrize("api", ["copy", "to"])
def test_non_blocking_mojo_to_cpu_does_not_drain_prior_gpu_work(
    mojo_gpu: str, free_immediately: bool, api: str
):
    """Async D2H (including free) returns without draining queued kernels.

    Both an explicit pinned copy destination and to("cpu", non_blocking=True)
    use Mojo's allocator, even with a CUDA-enabled torch wheel.
    Many elementwise multiplies make the queued GPU time dwarf dispatch cost;
    the generous margin keeps this robust on a busy shared GPU.
    """
    if free_immediately and get_accelerators()[0].api != "cuda":
        pytest.skip("MAX host callbacks require CUDA; free synchronizes otherwise")
    a = torch.full((4096, 4096), 1.0, device=mojo_gpu)
    b = torch.full((4096, 4096), 2.0, device=mojo_gpu)
    expected = torch.arange(1 << 20, dtype=torch.float32)
    source = expected.to(mojo_gpu)
    with device_module.device(mojo_gpu):
        downloaded = _empty_mojo_pinned_like(expected)

    # Warm both the async-download path and the queued kernel before timing.
    downloaded.copy_(source, non_blocking=True)
    _ = source.to("cpu", non_blocking=True)
    _ = a * b
    torch.accelerator.synchronize(mojo_gpu)

    def burst() -> torch.Tensor:
        out = a * b
        for _ in range(199):
            out = a * b
        return out

    reference_seconds, return_seconds = [], []
    for trial in range(4):
        # Alternate reference/transfer order (ABBA), then compare medians.
        for reference in (True, False) if trial % 2 == 0 else (False, True):
            if reference:
                started = time.perf_counter()
                _ = burst()
                torch.accelerator.synchronize(mojo_gpu)
                reference_seconds.append(time.perf_counter() - started)
            else:
                with device_module.device(mojo_gpu):
                    downloaded = _empty_mojo_pinned_like(expected)
                delayed = burst()
                started = time.perf_counter()
                if api == "copy":
                    downloaded.copy_(source, non_blocking=True)
                else:
                    downloaded = source.to("cpu", non_blocking=True)
                assert downloaded.is_pinned()
                if free_immediately:
                    del downloaded
                return_seconds.append(time.perf_counter() - started)
                # A host deschedule can let the queue finish at any point;
                # timing the transfer is the assertion, not an event query.
                torch.accelerator.synchronize(mojo_gpu)
                if not free_immediately:
                    torch.testing.assert_close(downloaded, expected)
                assert delayed.shape == (4096, 4096)

    assert median(return_seconds) < median(reference_seconds) * 0.5


def test_non_blocking_strided_d2h_survives_source_destruction(mojo_gpu: str):
    """A strided D2H download outlives its source tensor and storage."""
    expected = torch.arange(1 << 20, dtype=torch.int32)
    storage = torch.stack((expected, -expected), dim=1).to(mojo_gpu)
    source = storage[:, 0]
    downloaded = source.to("cpu", non_blocking=True)

    del source, storage
    torch.accelerator.synchronize(mojo_gpu)
    torch.testing.assert_close(downloaded, expected)


def test_non_blocking_d2h_survives_destination_destruction(mojo_gpu: str):
    """An automatically pinned download can be freed while DMA is pending."""
    a = torch.full((4096, 4096), 1.0, device=mojo_gpu)
    b = torch.full((4096, 4096), 2.0, device=mojo_gpu)
    source = torch.arange(1 << 20, dtype=torch.float32).to(mojo_gpu)
    torch.accelerator.synchronize(mojo_gpu)

    delayed = [a * b for _ in range(8)]
    downloaded = source.to("cpu", non_blocking=True)
    del downloaded

    torch.accelerator.synchronize(mojo_gpu)
    assert all(result.shape == (4096, 4096) for result in delayed)
    # A later, ordinary transfer must still be correct (no corruption left
    # behind by the dropped in-flight one).
    again = source.to("cpu")
    torch.testing.assert_close(again, source.cpu())


def test_non_blocking_pageable_download(mojo_device: str):
    """Mojo guarantees completed pageable copy_ values even with the true flag."""
    expected = torch.arange(1 << 20, dtype=torch.int32)
    source = expected.to(mojo_device)
    downloaded = torch.empty_like(expected)
    assert not downloaded.is_pinned()
    downloaded.copy_(source, non_blocking=True)
    torch.testing.assert_close(downloaded, expected)
    downloaded.zero_()
    downloaded.copy_(source, non_blocking=True)
    torch.testing.assert_close(downloaded, expected)


@pytest.mark.parametrize("non_blocking", [False, True])
def test_pinned_transfer_interior_pointer(mojo_device: str, non_blocking: bool):
    """Direct copies recognize offset views of the same pinned allocation."""
    expected = torch.arange(4096, dtype=torch.float32)
    with device_module.device(mojo_device):
        host = torch.full((4096 + 32,), -1.0).pin_memory()
        source = host[16:-16]
        source.copy_(expected)
        uploaded = source.to(mojo_device, non_blocking=non_blocking)
        if non_blocking:
            torch.accelerator.synchronize(mojo_device)
        source.zero_()
        source.copy_(uploaded, non_blocking=non_blocking)
        if non_blocking:
            torch.accelerator.synchronize(mojo_device)
        torch.testing.assert_close(source, expected)
        assert torch.all(host[:16] == -1) and torch.all(host[-16:] == -1)


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_non_blocking_pinned_download_layout_and_dtype(
    mojo_device: str, strided: bool, dtype: torch.dtype
):
    """Host relayout/casting consumes a completed, blocking staged download."""
    expected = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
    source = expected.to(mojo_device)
    with device_module.device(mojo_device):
        storage = torch.full((32, 65), -1, dtype=dtype).pin_memory()
        destination = (
            storage[:, 1::2] if strided else storage.flatten()[1:1025].view(32, 32)
        )
        destination.copy_(source, non_blocking=True)
        # Only the dense, same-dtype route needs an explicit wait.
        if not strided and dtype == expected.dtype:
            torch.accelerator.synchronize(mojo_device)
        torch.testing.assert_close(destination, expected.to(dtype))
        if strided:
            assert torch.all(storage[:, ::2] == -1)


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_non_blocking_pinned_transfer_survives_host_destruction(
    mojo_device: str, direction: str
):
    """Free a pinned block used on two streams; neither may lose its DMA data."""
    expected = torch.arange(1 << 20, dtype=torch.float32)
    is_cpu = torch.device(mojo_device) == device_module.cpu()
    streams = (
        [device_module.current_stream(mojo_device)]
        if is_cpu
        else [side_stream_or_skip(mojo_device) for _ in range(2)]
    )
    with device_module.device(mojo_device):
        host = torch.cat((expected, expected)).pin_memory()
        source = expected.to(mojo_device)
        a = torch.full((2048, 2048), 1.0, device=mojo_device)
        b = torch.full_like(a, 2.0)
        _ = a * b  # compile before queueing delayed copies
        torch.accelerator.synchronize(mojo_device)
        uploaded = []
        delayed = []
        for index, stream in enumerate(streams):
            with device_module.stream(stream):
                if not is_cpu:
                    delayed.extend(a * b for _ in range(64))
                view = host[index * expected.numel() : (index + 1) * expected.numel()]
                if direction == "upload":
                    # Repeated use of one stream must be deduplicated.
                    uploaded.append(view.to(mojo_device, non_blocking=True))
                    uploaded.append(view.to(mojo_device, non_blocking=True))
                else:
                    view.copy_(source, non_blocking=True)
                    view.copy_(source, non_blocking=True)
                del view
        del host
        # The churn below is what proves the deferred free: had the block gone
        # back for reuse before its DMA finished, one of these would hold the
        # transferred values instead of -1. (Asking is_pinned() about the freed
        # address cannot show that -- once CUDA is initialized it claims MAX's
        # page-locked pointers too, so the answer depends on test order.)
        churn = [torch.full((2 << 20,), -1.0).pin_memory() for _ in range(4)]
        for stream in streams:
            stream.synchronize()
        assert all(torch.all(block == -1) for block in churn)
        for tensor in uploaded:
            torch.testing.assert_close(tensor.cpu(), expected)
        torch.testing.assert_close(source.cpu(), expected)
        assert all(result.shape == a.shape for result in delayed)


@contextmanager
def _held_transfer_stream(stream: torch.Stream) -> Iterator[Event]:
    """A real driver callback gates DMA; a watchdog bounds failed tests.

    Mutating before releasing this gate is ordered, not a race with DMA.
    The callback uses no device API or backend mutex. Keep it alive until
    stream completion, including exception unwinding. Hold only one callback
    at a time: CUDA may serialize callback execution across streams.
    """
    if get_accelerators()[stream.device_index].api != "cuda":
        pytest.skip("the independent stream gate uses CUDA host callbacks")
    driver = ctypes.CDLL("libcuda.so.1")
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    driver.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, callback_type, ctypes.c_void_p]
    driver.cuLaunchHostFunc.restype = ctypes.c_int
    entered, release, expired = Event(), Event(), Event()

    def wait_for_release(unused: int | None):
        entered.set()
        if not release.wait(30):
            expired.set()

    callback = callback_type(wait_for_release)
    assert (
        driver.cuLaunchHostFunc(
            device_module.stream_native_handle(stream), callback, None
        )
        == 0
    )
    try:
        assert entered.wait(10), "driver did not enter stream gate"
        yield release
    finally:
        release.set()
        stream.synchronize()
    assert not expired.is_set(), "stream gate watchdog expired"


@pytest.mark.parametrize("api", ["copy", "to"])
@pytest.mark.parametrize("host_class", ["owned", "pageable", "foreign", "other-device"])
@pytest.mark.parametrize("offset", [0, 1])
def test_non_blocking_pinned_upload_skips_staging_memcpy(
    mojo_gpu: str, api: str, host_class: str, offset: int
):
    """Direct DMA sees the post-mutation value; staging preserves the snapshot.

    The held stream makes the source mutation happen before DMA can start,
    including a one-byte-offset pointer. Timing cannot distinguish these routes.
    """
    with device_module.device(mojo_gpu):
        storage = torch.full((4099,), 17, dtype=torch.uint8)
        if host_class == "owned":
            storage = storage.pin_memory()
        elif host_class == "foreign":
            if not torch.cuda.is_available():
                pytest.skip("requires CUDA's foreign pinned allocator")
            storage = torch.full_like(storage, 17, pin_memory=True)
        elif host_class == "other-device":
            with device_module.device(device_module.cpu()):
                storage = storage.pin_memory()
        source = storage[offset : offset + 4096]
        destination = source.to(mojo_gpu, non_blocking=True)
        destination.copy_(source, non_blocking=True)
        stream = device_module.current_stream(mojo_gpu)
        stream.synchronize()
        with _held_transfer_stream(stream):
            if api == "to":
                destination = source.to(mojo_gpu, non_blocking=True)
            else:
                destination.copy_(source, non_blocking=True)
            source.fill_(29)
        expected = torch.full_like(source, 29 if host_class == "owned" else 17)
        assert not destination.is_pinned()
        torch.testing.assert_close(destination.cpu(), expected)
        assert torch.all(storage[offset + 4096 :] == 17)
        assert torch.all(storage[:offset] == 17)


@pytest.mark.parametrize("api", ["copy", "to"])
@pytest.mark.parametrize("non_blocking", [False, True])
@pytest.mark.parametrize("streams", ["default", "side", "cross"])
@pytest.mark.parametrize("conversion", ["none", "dtype", "transpose"])
def test_download_then_upload_orders_pinned_reads(
    mojo_gpu: str, api: str, non_blocking: bool, streams: str, conversion: str
):
    """A queued D2H feeds H2D, including CPU conversion and explicit waits.

    Raw reads stay stream-ordered; a CPU cast/relayout must synchronize before
    reading. Both blocking and nonblocking uploads must see downloaded values.
    """
    with device_module.device(mojo_gpu):
        expected = torch.arange(1 << 20, dtype=torch.float32).reshape(1024, 1024)
        host = torch.full_like(expected, -17).pin_memory()
        source = expected.to(mojo_gpu)
        read_view = host.t() if conversion == "transpose" else host
        dtype = torch.float64 if conversion == "dtype" else host.dtype
        wanted = expected.t().contiguous() if conversion == "transpose" else expected
        destination = torch.empty(read_view.shape, dtype=dtype, device=mojo_gpu)
        # Warm both spellings and all conversions before holding the stream.
        destination.copy_(read_view, non_blocking=non_blocking)
        _ = read_view.to(mojo_gpu, dtype=dtype, non_blocking=non_blocking)
        host.copy_(source, non_blocking=True)
        torch.accelerator.synchronize(mojo_gpu)
        host.fill_(-17)
        first = (
            device_module.current_stream(mojo_gpu)
            if streams == "default"
            else side_stream_or_skip(mojo_gpu)
        )
        second = side_stream_or_skip(mojo_gpu) if streams == "cross" else first
        with _held_transfer_stream(first) as release:
            with device_module.stream(first):
                host.copy_(source, non_blocking=True)
                ready = first.record_event()
            assert not ready.query()
            # Blocking calls and CPU conversion legitimately wait for DMA.
            timer = Timer(0.2, release.set)
            timer.start()
            try:
                with device_module.stream(second):
                    second.wait_event(ready)
                    if api == "copy":
                        destination.copy_(read_view, non_blocking=non_blocking)
                    else:
                        destination = read_view.to(
                            mojo_gpu, dtype=dtype, non_blocking=non_blocking
                        )
                    destination.record_stream(second)
            finally:
                timer.join()
        second.synchronize()
        torch.testing.assert_close(destination.cpu(), wanted.to(dtype))


@pytest.mark.parametrize("non_blocking", [False, True])
@pytest.mark.parametrize("api", ["copy", "to"])
@pytest.mark.parametrize("side_stream", [False, True])
@pytest.mark.parametrize("pinned", [False, True])
def test_upload_then_download_orders_values(
    mojo_gpu: str, non_blocking: bool, api: str, side_stream: bool, pinned: bool
):
    """H2D then D2H uses the chosen stream in either transfer spelling."""
    with device_module.device(mojo_gpu):
        expected = torch.arange(8192, dtype=torch.int32)
        host = expected.pin_memory() if pinned else expected.clone()
        downloaded = torch.empty_like(expected)
        if pinned:
            downloaded = downloaded.pin_memory()
        stream = (
            side_stream_or_skip(mojo_gpu)
            if side_stream
            else device_module.current_stream(mojo_gpu)
        )
        with device_module.stream(stream):
            uploaded = host.to(mojo_gpu, non_blocking=non_blocking)
            if api == "copy":
                result = downloaded.copy_(uploaded, non_blocking=non_blocking)
                assert result is downloaded
            else:
                downloaded = uploaded.to("cpu", non_blocking=non_blocking)
                assert downloaded.is_pinned() == non_blocking
            if not non_blocking or (api == "copy" and not pinned):
                torch.testing.assert_close(downloaded, expected)
            stream.synchronize()
        torch.testing.assert_close(downloaded, expected)


@pytest.mark.parametrize("direction", ["upload", "download"])
@pytest.mark.parametrize("host_class", ["owned", "pageable", "foreign", "other-device"])
@pytest.mark.parametrize("strided", [False, True])
def test_blocking_transfer_completes_selected_stream(
    mojo_gpu: str, direction: str, host_class: str, strided: bool
):
    """Blocking downloads and direct uploads finish reading caller memory."""
    with device_module.device(mojo_gpu):
        expected = torch.arange(4096, dtype=torch.float32).reshape(64, 64)
        host = expected.clone()
        if host_class == "owned":
            host = host.pin_memory()
        elif host_class == "foreign":
            if not torch.cuda.is_available():
                pytest.skip("requires CUDA's foreign pinned allocator")
            host = torch.empty_like(host, pin_memory=True).copy_(expected)
        elif host_class == "other-device":
            with device_module.device(device_module.cpu()):
                host = host.pin_memory()
        storage = expected.to(mojo_gpu)
        gpu = storage.t() if strided else storage
        source, destination = (host, gpu) if direction == "upload" else (gpu, host)
        destination.copy_(source)  # warm the selected path
        stream = device_module.current_stream(mojo_gpu)
        with _held_transfer_stream(stream) as release:
            pending = stream.record_event()
            timer = Timer(0.2, release.set)
            timer.start()
            try:
                destination.copy_(source, non_blocking=False)
                if direction == "download" or host_class == "owned":
                    assert pending.query(), "blocking copy returned before its stream"
                if direction == "download":
                    torch.testing.assert_close(
                        host, expected.t() if strided else expected
                    )
                else:
                    host.zero_()  # Staged uploads must already own a snapshot.
            finally:
                timer.join()
        if direction == "upload":
            torch.testing.assert_close(gpu.cpu(), expected)


@pytest.mark.parametrize("api", ["copy", "copy_strided", "to", "arange"])
def test_blocking_pageable_upload_does_not_drain_prior_gpu_work(
    mojo_gpu: str, api: str
):
    """The host snapshot satisfies blocking semantics without draining DMA."""
    expected = torch.arange(2**54, 2**54 + 16, dtype=torch.int64)
    source = expected.clone()
    assert not source.is_pinned()
    destination = torch.empty((16, 2), dtype=source.dtype, device=mojo_gpu)[:, 0]
    if api != "copy_strided":
        destination = torch.empty_like(source, device=mojo_gpu)

    def upload() -> torch.Tensor:
        if api == "arange":
            # Large integer factories use the internal pageable staging path.
            return torch.arange(2**54, 2**54 + 16, dtype=torch.int64, device=mojo_gpu)
        if api == "to":
            return source.to(mojo_gpu, non_blocking=False)
        return destination.copy_(source, non_blocking=False)

    uploaded = upload()  # Warm compilation and the staging allocator.
    stream = device_module.current_stream(mojo_gpu)
    stream.synchronize()
    return_seconds = []
    hold_seconds = 0.3
    for _ in range(4):
        source.copy_(expected)
        with _held_transfer_stream(stream) as release:
            timer = Timer(hold_seconds, release.set)
            timer.start()
            try:
                started = time.perf_counter()
                uploaded = upload()
                return_seconds.append(time.perf_counter() - started)
                source.zero_()
            finally:
                timer.join()
        torch.testing.assert_close(uploaded.cpu(), expected)
    assert median(return_seconds) < hold_seconds * 0.5


@pytest.mark.parametrize("non_blocking", [False, True])
@pytest.mark.parametrize("pin_memory", [False, True])
def test_to_cpu_explicit_pin_memory(
    mojo_device: str,
    non_blocking: bool,
    pin_memory: bool,
    pin_allocator_probe: "_PinAllocatorProbe",
):
    """_to_copy exposes pin_memory; Tensor.to's Python overloads do not.

    Upstream overwrites the pin_memory option with non_blocking for strided
    accelerator-to-CPU transfers, including pin_memory=True on blocking output.
    """
    expected = torch.arange(257, dtype=torch.float32)
    source = expected.to(mojo_device)
    with device_module.device(device_module.cpu()):
        result = torch.ops.aten._to_copy.default(
            source,
            device=torch.device("cpu"),
            pin_memory=pin_memory,
            non_blocking=non_blocking,
        )
        assert result.is_pinned() == non_blocking
        if result.is_pinned():
            assert pin_allocator_probe.uses_mojo_allocator(result)
        assert device_module.current_device() == device_module.cpu().index
        if not non_blocking:
            torch.testing.assert_close(result, expected)
        torch.accelerator.synchronize(mojo_device)
    torch.testing.assert_close(result, expected)


def test_blocking_to_cpu_ignores_explicit_pin_memory(mojo_device: str):
    """Upstream pin_out overwrites even an explicit true pin_memory option."""
    expected = torch.arange(257, dtype=torch.int32)
    result = torch.ops.aten._to_copy.default(
        expected.to(mojo_device),
        device=torch.device("cpu"),
        layout=torch.strided,
        pin_memory=True,
        non_blocking=False,
    )
    assert not result.is_pinned()
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("transpose", [False, True])
def test_to_cpu_device_conversion_stays_asynchronous(
    mojo_gpu: str, dtype: torch.dtype, transpose: bool
):
    """Unlike copy_'s host cast, to() casts/relayouts on GPU and stays async."""
    with device_module.device(mojo_gpu):
        expected = torch.arange(1024, dtype=torch.float32).reshape(32, 32)
        if transpose:
            expected = expected.t()
        source = expected.to(mojo_gpu)
        _ = source.to("cpu", dtype=dtype, non_blocking=True)
        stream = device_module.current_stream(mojo_gpu)
        stream.synchronize()
        with _held_transfer_stream(stream):
            pending = stream.record_event()
            with device_module.device(device_module.cpu()):
                downloaded = source.to("cpu", dtype=dtype, non_blocking=True)
                assert device_module.current_device() == device_module.cpu().index
            assert downloaded.is_pinned()
            assert not pending.query()
        assert downloaded.stride() == expected.stride()
        torch.testing.assert_close(downloaded, expected.to(dtype))


def test_to_cpu_host_conversion_is_deliberately_blocking(mojo_gpu: str):
    """Float64 to() uses our host cast fallback; CUDA can cast it on the GPU."""
    skip_if_metal(mojo_gpu, "Metal does not support float64")
    with device_module.device(mojo_gpu):
        expected = torch.arange(257, dtype=torch.float32)
        source = expected.to(mojo_gpu)
        _ = source.to("cpu", dtype=torch.float64, non_blocking=True)
        stream = device_module.current_stream(mojo_gpu)
        stream.synchronize()
        with _held_transfer_stream(stream) as release:
            pending = stream.record_event()
            timer = Timer(0.2, release.set)
            timer.start()
            try:
                output = source.to("cpu", dtype=torch.float64, non_blocking=True)
                assert pending.query()
                torch.testing.assert_close(output, expected.double())
            finally:
                timer.join()
        torch.accelerator.synchronize(mojo_gpu)
        assert output.is_pinned()
        torch.testing.assert_close(output, expected.double())


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.float32,
        torch.int64,
        torch.float64,
    ],
)
@pytest.mark.parametrize("shape", [(), (0,), (3, 0, 5), (17,), (257,)])
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_transfer_raw_bytes_and_empty_storage(
    mojo_device: str,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    pinned: bool,
    non_blocking: bool,
):
    """Both APIs preserve every supported element width, scalar rank and empties."""
    if dtype == torch.float64:
        skip_if_metal(mojo_device, "Metal does not support float64")
    expected = (torch.arange(math.prod(shape)) % 31).to(dtype).reshape(shape)
    with device_module.device(mojo_device):
        host = expected.pin_memory() if pinned else expected.clone()
        uploaded = host.to(mojo_device, non_blocking=non_blocking)
        downloaded = uploaded.to("cpu", non_blocking=non_blocking)
        torch.accelerator.synchronize(mojo_device)
        assert downloaded.shape == shape
        assert downloaded.dtype == dtype
        assert torch.equal(
            downloaded.reshape(-1).view(torch.uint8),
            expected.reshape(-1).view(torch.uint8),
        )
        if not math.prod(shape):
            assert downloaded.untyped_storage().data_ptr() == 0
            assert not downloaded.is_pinned()
        uploaded.copy_(host, non_blocking=non_blocking)
        result = host.copy_(uploaded, non_blocking=non_blocking)
        assert result is host
        torch.accelerator.synchronize(mojo_device)
        assert torch.equal(
            host.reshape(-1).view(torch.uint8), expected.reshape(-1).view(torch.uint8)
        )


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (3,),
        (255,),
        (256,),
        (4095,),
        (4096,),
        (4097,),
        (357, 789),
        (8 * 1024 * 1024 + 3,),
    ],
)
@pytest.mark.parametrize("non_blocking", [False, True])
def test_transfer_byte_boundaries_and_buffer_alias(
    mojo_device: str, shape: tuple[int, ...], non_blocking: bool
):
    """An independent storage beginning one byte into a pinned block is eligible."""
    count = math.prod(shape)
    expected = (torch.arange(count) % 251).to(torch.uint8).reshape(shape)
    with device_module.device(mojo_device):
        storage = torch.full((count + 2,), 253, dtype=torch.uint8).pin_memory()
        alias = torch.frombuffer(
            storage.numpy(), dtype=torch.uint8, count=count, offset=1
        ).reshape(shape)
        assert alias.untyped_storage().data_ptr() == storage.data_ptr() + 1
        assert alias.is_pinned()
        alias.copy_(expected)
        uploaded = alias.to(mojo_device, non_blocking=non_blocking)
        torch.accelerator.synchronize(mojo_device)
        alias.zero_()
        alias.copy_(uploaded, non_blocking=non_blocking)
        torch.accelerator.synchronize(mojo_device)
        assert torch.equal(alias, expected)
        assert storage[0] == 253 and storage[-1] == 253
        # Empty views keep storage pinning while zero-element copies do nothing.
        empty = storage[count + 2 :]
        assert empty.is_pinned()
        empty.copy_(empty.to(mojo_device), non_blocking=non_blocking)
        assert storage[0] == 253 and storage[-1] == 253


@pytest.mark.parametrize("layout", ["transpose", "gapped", "expanded", "channels-last"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("non_blocking", [False, True])
def test_transfer_layout_and_conversion_canaries(
    mojo_device: str, layout: str, dtype: torch.dtype, non_blocking: bool
):
    """Host relayout/cast snapshots before return and never overwrites gaps."""
    if dtype == torch.float64:
        skip_if_metal(mojo_device, "Metal does not support float64")
    with device_module.device(mojo_device):
        storage = torch.arange(512, dtype=torch.float32).pin_memory()
        if layout == "transpose":
            host = storage[:256].view(16, 16).t()
        elif layout == "gapped":
            host = storage[1::2].view(16, 16)
        elif layout == "expanded":
            host = storage[:16].view(1, 16).expand(16, 16)
        else:
            host = storage[:256].view(2, 4, 4, 8).permute(0, 3, 1, 2)
        expected = host.clone().to(dtype)
        backing = torch.full((*host.shape, 2), -999, dtype=dtype, device=mojo_device)
        destination = backing[..., 1]
        destination.copy_(host, non_blocking=non_blocking)
        via_to = host.to(mojo_device, dtype=dtype, non_blocking=non_blocking)
        host.fill_(-13)
        torch.testing.assert_close(destination.cpu(), expected)
        torch.testing.assert_close(via_to.cpu(), expected)
        assert torch.all(backing[..., 0].cpu() == -999)
        cpu_backing = torch.full(
            (*host.shape, 2), -999, dtype=torch.float32
        ).pin_memory()
        cpu_destination = cpu_backing[..., 1]
        ptr, strides = cpu_destination.data_ptr(), cpu_destination.stride()
        result = cpu_destination.copy_(destination, non_blocking=non_blocking)
        assert result is cpu_destination
        assert cpu_destination.data_ptr() == ptr and cpu_destination.stride() == strides
        assert cpu_destination.is_pinned()
        # A strided host destination requires blocking host work for either flag.
        torch.testing.assert_close(cpu_destination, expected.float())
        assert torch.all(cpu_backing[..., 0] == -999)


@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.float16, torch.bfloat16, torch.float64]
)
def test_transfer_conversion_value_boundaries(mojo_device: str, dtype: torch.dtype):
    """Host and device casts preserve defined rounding, signs, NaNs and infinities."""
    if dtype == torch.float64:
        skip_if_metal(mojo_device, "Metal does not support float64")
    source = torch.tensor(
        [
            0.0,
            -0.0,
            1.00390625,
            -1.00390625,
            255.5,
            -128.5,
            float("inf"),
            float("-inf"),
            float("nan"),
        ]
    )
    with device_module.device(mojo_device):
        host = source.pin_memory()
        result = host.to(mojo_device, dtype=dtype, non_blocking=True)
        expected = source.to(dtype)
        torch.testing.assert_close(result.cpu(), expected, equal_nan=True)
        gpu = source.to(mojo_device)
        output = torch.empty_like(source, dtype=dtype).pin_memory()
        output.copy_(gpu, non_blocking=True)
        torch.testing.assert_close(output, expected, equal_nan=True)
        if dtype != torch.bool:
            assert torch.equal(torch.signbit(output[:2]), torch.signbit(expected[:2]))


@pytest.mark.parametrize("non_blocking", [False, True])
@pytest.mark.parametrize("src_pinned", [False, True])
@pytest.mark.parametrize("dst_pinned", [False, True])
def test_host_to_host_copy_pinning_is_storage_property(
    mojo_device: str, non_blocking: bool, src_pinned: bool, dst_pinned: bool
):
    """CPU copies finish immediately for every pinned/pageable pairing."""
    with device_module.device(mojo_device):
        source = torch.arange(257)
        destination = torch.empty_like(source)
        if src_pinned:
            source = source.pin_memory()
        if dst_pinned:
            destination = destination.pin_memory()
        assert destination.copy_(source, non_blocking=non_blocking) is destination
        torch.testing.assert_close(destination, source)
        assert destination.is_pinned() == dst_pinned


@pytest.mark.parametrize("direction", ["upload", "download"])
@pytest.mark.parametrize("first_to_finish", [0, 1])
def test_transfer_final_offset_owner_waits_for_both_streams(
    mojo_gpu: str, direction: str, first_to_finish: int
):
    """One completion cannot reclaim a block still used by another stream."""
    with device_module.device(mojo_gpu):
        expected = torch.arange(8192, dtype=torch.float32)
        host = torch.cat(
            (torch.tensor([-1.0]), expected, expected, torch.tensor([-1.0]))
        ).pin_memory()
        streams = [side_stream_or_skip(mojo_gpu), side_stream_or_skip(mojo_gpu)]
        source = expected.to(mojo_gpu)
        outputs = [torch.empty_like(source) for _ in streams]
        for output in outputs:
            output.copy_(expected, non_blocking=True)
        # Warm the download registration before entering either driver gate.
        host[1:8193].copy_(source, non_blocking=True)
        torch.accelerator.synchronize(mojo_gpu)
        delayed = 1 - first_to_finish
        with _held_transfer_stream(streams[delayed]):
            for index, stream in enumerate(streams):
                with device_module.stream(stream):
                    view = (
                        host[1:8193]
                        if direction == "upload"
                        else host[1 + index * 8192 : 1 + (index + 1) * 8192]
                    )
                    if direction == "upload":
                        outputs[index].copy_(view, non_blocking=True)
                        outputs[index].record_stream(stream)
                    else:
                        view.copy_(source, non_blocking=True)
                    del view
            streams[first_to_finish].synchronize()
            assert not streams[delayed].query()
            # The first use is complete at destruction; the other must keep
            # the backing alive. Do not wait for a free callback while the
            # gate holds CUDA's callback worker.
            del host
            replacements = [torch.full((16386,), -23.0).pin_memory() for _ in range(8)]
        streams[first_to_finish].synchronize()
        assert all(torch.all(block == -23) for block in replacements)
        for output in outputs:
            torch.testing.assert_close(output.cpu(), expected)


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_transfer_gpu_endpoint_destruction_and_device_churn(
    mojo_gpu: str, direction: str
):
    """Both device allocations and the host block outlive their queued DMA."""
    with device_module.device(mojo_gpu):
        expected = torch.arange(8192, dtype=torch.float32)
        host = expected.pin_memory()
        stream = side_stream_or_skip(mojo_gpu)
        with device_module.stream(stream):
            gpu = expected.to(mojo_gpu)
            host.copy_(gpu, non_blocking=True)
            stream.synchronize()
            with _held_transfer_stream(stream):
                if direction == "upload":
                    gpu.copy_(host, non_blocking=True)
                else:
                    host.copy_(gpu, non_blocking=True)
                del gpu
                replacements = [
                    torch.full_like(expected, -31, device=mojo_gpu) for _ in range(8)
                ]
        torch.testing.assert_close(host, expected)
        for replacement in replacements:
            assert torch.all(replacement.cpu() == -31)


def test_blocking_transfer_does_not_synchronize_other_streams(mojo_gpu: str):
    """Blocking means the selected stream; unrelated stream work stays pending."""
    with device_module.device(mojo_gpu):
        host = torch.arange(4096, dtype=torch.float32).pin_memory()
        gpu = host.to(mojo_gpu)
        other = side_stream_or_skip(mojo_gpu)
        with _held_transfer_stream(other):
            pending = other.record_event()
            gpu.copy_(host)
            downloaded = gpu.to("cpu")
            torch.testing.assert_close(downloaded, host)
            assert not pending.query()


@pytest.mark.parametrize("non_blocking", [False, True])
def test_max_cpu_transfer_is_synchronous(non_blocking: bool):
    """The MAX CPU device completes host DMA and conversion before returning."""
    expected = torch.arange(257, dtype=torch.float32)
    with device_module.device("mojo:0"):
        host = expected.pin_memory()
    cpu = device_module.cpu()
    uploaded = host.to(cpu, dtype=torch.float64, non_blocking=non_blocking)
    host.zero_()
    result = uploaded.to("cpu", non_blocking=non_blocking)
    torch.testing.assert_close(result, expected.double())
    host.copy_(uploaded, non_blocking=non_blocking)
    torch.testing.assert_close(host, expected)


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_transfer_repeated_use_keeps_last_fence(mojo_gpu: str, direction: str):
    """A completed first use must not free storage before a later use finishes."""
    with device_module.device(mojo_gpu):
        first = torch.full((8192,), 13.0)
        second = torch.full_like(first, 29.0)
        host = first.pin_memory()
        gpu = first.to(mojo_gpu)
        next_gpu = second.to(mojo_gpu)
        stream = device_module.current_stream(mojo_gpu)
        host.copy_(gpu, non_blocking=True)
        stream.synchronize()
        host.copy_(second)
        with _held_transfer_stream(stream):
            if direction == "upload":
                gpu.copy_(host, non_blocking=True)
            else:
                host.copy_(next_gpu, non_blocking=True)
                # The write and read must share a lifetime and be ordered.
                gpu.copy_(host, non_blocking=True)
            with device_module.device(device_module.cpu()):
                del host
            replacements = [torch.full_like(first, -1).pin_memory() for _ in range(8)]
        torch.testing.assert_close(gpu.cpu(), second)
        assert all(torch.all(block == -1) for block in replacements)


@pytest.mark.parametrize("non_blocking", [False, True])
def test_transfer_broadcast_and_overlapping_destination(
    mojo_device: str, non_blocking: bool
):
    """Public copy_ broadcasts sources and rejects definite overlap before writes."""
    source = torch.tensor([[1.0], [2.0], [3.0]]).pin_memory()
    gpu = torch.full((3, 7), -1.0, device=mojo_device)
    gpu.copy_(source, non_blocking=non_blocking)
    torch.testing.assert_close(gpu.cpu(), source.expand(3, 7))
    host = torch.full((1,), -19.0).pin_memory()
    expanded = host.expand(3, 7)
    with pytest.raises(RuntimeError, match="single memory location|overlap"):
        expanded.copy_(gpu, non_blocking=non_blocking)
    assert host.item() == -19


@pytest.mark.parametrize("non_blocking", [False, True])
def test_transfer_lazy_negative_and_to_identity(mojo_device: str, non_blocking: bool):
    """Logical negative views survive transfers; same-device to() is not a fence."""
    expected = torch.arange(257, dtype=torch.float32)
    host = expected.pin_memory()
    negative = torch.ops.aten._neg_view.default(host)
    gpu = negative.to(mojo_device, non_blocking=non_blocking)
    torch.testing.assert_close(gpu.cpu(), -expected)
    assert gpu.to(mojo_device, non_blocking=non_blocking) is gpu
    copied = gpu.to(mojo_device, copy=True, non_blocking=non_blocking)
    assert copied.data_ptr() != gpu.data_ptr()
    torch.testing.assert_close(copied.cpu(), -expected)
    gpu_negative = torch.ops.aten._neg_view.default(gpu)
    downloaded = gpu_negative.to("cpu", non_blocking=non_blocking)
    torch.accelerator.synchronize(mojo_device)
    torch.testing.assert_close(downloaded, expected)


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_transfer_unsupported_complex_is_explicit(mojo_device: str, dtype: torch.dtype):
    """Complex host pinning is supported; device transfer has a clear dtype boundary."""
    host = torch.tensor([1 + 2j, -3 - 4j], dtype=dtype).pin_memory()
    expected = host.clone()
    with pytest.raises(NotImplementedError, match="dtype|ScalarType"):
        host.conj().to(mojo_device, non_blocking=True)
    torch.testing.assert_close(host, expected)


@pytest.mark.parametrize("reverse", [False, True])
def test_transfer_other_gpu_requires_snapshot_and_host_wait(reverse: bool):
    """A GPU event wait cannot order a host snapshot across allocation devices."""
    if len(get_accelerators()) < 3:
        pytest.skip("requires two GPUs plus the MAX CPU device")
    first, second = ("mojo:1", "mojo:0") if reverse else ("mojo:0", "mojo:1")
    expected = torch.arange(8192, dtype=torch.float32)
    with device_module.device(first):
        host = torch.full_like(expected, -1).pin_memory()
        source = expected.to(first)
        host.copy_(source, non_blocking=True)
        done = device_module.current_stream(first).record_event()
    done.synchronize()  # Host staging on the other device needs a host wait.
    with device_module.device(second):
        result = host.to(second, non_blocking=True)
        host.zero_()
        torch.testing.assert_close(result.cpu(), expected)
        host.copy_(result, non_blocking=True)
        torch.testing.assert_close(host, expected)


def test_non_blocking_foreign_pinned_transfers(mojo_gpu: str, cuda_available: bool):
    """CUDA's pinned allocator cannot track DMA on Mojo streams."""
    if not cuda_available:
        pytest.skip("requires CUDA's pinned allocator")
    expected = torch.arange(1 << 20, dtype=torch.float32)
    pinned = torch.empty_like(expected, pin_memory=True)
    pinned.copy_(expected)
    uploaded = pinned.to(mojo_gpu, non_blocking=True)
    pinned.zero_()  # The upload must have taken a snapshot.
    torch.testing.assert_close(uploaded.cpu(), expected)
    pinned.copy_(uploaded, non_blocking=True)
    torch.testing.assert_close(pinned, expected)  # D2H must already be complete.


def test_non_blocking_pinned_other_device(mojo_gpu: str):
    """A block pinned on the MAX CPU device takes the conservative GPU path."""
    expected = torch.arange(1 << 20, dtype=torch.float32)
    with device_module.device(device_module.cpu()):
        host = expected.pin_memory()
    with device_module.device(mojo_gpu):
        uploaded = host.to(mojo_gpu, non_blocking=True)
        host.zero_()
        torch.testing.assert_close(uploaded.cpu(), expected)
        host.copy_(uploaded, non_blocking=True)
        torch.testing.assert_close(host, expected)


def test_same_device_d2d_does_not_drain_prior_gpu_work(mojo_gpu: str):
    """Contiguous and strided D2D copies stay queued on the device stream."""
    a = torch.full((4096, 4096), 1.0, device=mojo_gpu)
    b = torch.full((4096, 4096), 2.0, device=mojo_gpu)
    elements = 1 << 20
    expected = torch.arange(elements, dtype=torch.float32)

    contiguous_source = expected.to(mojo_gpu)
    contiguous_destination = torch.empty_like(contiguous_source)

    strided_source_storage = torch.stack((expected, -expected), dim=1).to(mojo_gpu)
    strided_source = strided_source_storage[:, 0]
    strided_destination_storage = torch.empty_like(strided_source_storage)
    strided_destination = strided_destination_storage[:, 1]

    # Warm every copy path before measuring Python return latency.
    contiguous_destination.copy_(contiguous_source)
    strided_destination.copy_(strided_source)
    repeats = 200
    for _ in range(repeats):
        _ = a * b
    torch.accelerator.synchronize(mojo_gpu)

    started = time.perf_counter()
    for _ in range(repeats):
        _ = a * b
    torch.accelerator.synchronize(mojo_gpu)
    mul_seconds = time.perf_counter() - started

    for layout, destination, source in (
        ("contiguous", contiguous_destination, contiguous_source),
        ("strided", strided_destination, strided_source),
    ):
        delayed = [a * b for _ in range(repeats)]
        started = time.perf_counter()
        destination.copy_(source)
        copy_return_seconds = time.perf_counter() - started

        assert copy_return_seconds < mul_seconds * 0.5, layout
        torch.accelerator.synchronize(mojo_gpu)
        torch.testing.assert_close(destination.cpu(), expected)
        assert all(result.shape == (4096, 4096) for result in delayed)


def test_dtype_preservation(mojo_device):
    """Test that dtypes are preserved during conversion"""
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        original = torch.tensor([1, 2, 3], dtype=dtype)
        mojo_tensor = original.to(mojo_device)
        result = mojo_tensor.to("cpu")
        assert result.dtype == dtype
        torch.testing.assert_close(result, original)


def test_multiple_conversions():
    """Test multiple to() calls don't cause issues"""
    tensor = torch.tensor([1.0, 2.0])
    max1 = tensor.to("mojo")
    max2 = max1.to("mojo")  # Should return same tensor
    cpu1 = max2.to("cpu")
    cpu2 = cpu1.to("cpu")  # Should work normally
    torch.testing.assert_close(cpu2, tensor)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::sub.Tensor")
def test_multiple_conversions_arithmetic():
    """Operate on the round-tripped tensors: a same-device sub then square."""
    tensor = torch.tensor([1.0, 2.0])
    max1 = tensor.to("mojo")
    max2 = max1.to("mojo")
    diff = max1 - max2
    squared = diff * diff
    summed = torch.sum(squared)
    assert summed.to("cpu").item() == 0


def test_pin_memory_preserves_values_and_reuses_storage(mojo_device: str):
    with device_module.device(mojo_device):
        source = torch.arange(64, dtype=torch.float32)
        pinned = source.pin_memory()
        assert pinned.device.type == "cpu"
        assert pinned.is_pinned()
        torch.testing.assert_close(pinned, source)
        assert pinned.pin_memory().data_ptr() == pinned.data_ptr()


@pytest.mark.parametrize("entry", ["empty", "empty_like"])
@pytest.mark.parametrize("cuda_first", [False, True], ids=["cold", "cuda-initialized"])
def test_first_pinned_factory_query_after_registration(
    mojo_device: str, entry: str, cuda_first: bool
):
    """Registration makes the first factory's is_pinned query reach our hook."""
    if cuda_first and not torch.cuda.is_available():
        pytest.skip(
            "initializing CUDA before registration requires runtime CUDA availability"
        )
    script = textwrap.dedent(
        """
        import sys
        import torch
        from torch_mojo_backend import register_mojo_devices
        from torch_mojo_backend.native import device_module

        if sys.argv[3] == 'True':
            torch.cuda.init()
        register_mojo_devices()
        with device_module.device(sys.argv[1]):
            if sys.argv[2] == 'empty':
                pinned = torch.empty(8, device='cpu', pin_memory=True)
            else:
                pinned = torch.empty_like(torch.empty(8), device='cpu', pin_memory=True)
            assert pinned.is_pinned()
        """
    )
    # MAX's Python interop sets these to its own prefix in the parent.
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONEXECUTABLE")
    }
    proc = subprocess.run(
        [sys.executable, "-c", script, mojo_device, entry, str(cuda_first)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("size", [0, 8])
def test_pinned_empty(mojo_device: str, size: int):
    with device_module.device(mojo_device):
        for pinned in (
            torch.empty(size, pin_memory=True),
            torch.empty(size).pin_memory(),
        ):
            assert pinned.device.type == "cpu"
            # Zero bytes means a null storage pointer; stock CUDA also says False.
            assert pinned.is_pinned() is (size != 0)
        assert not torch.empty(size).is_pinned()


def test_pinned_entry_points_agree(mojo_device: str):
    with device_module.device(mojo_device):
        source = torch.arange(8, dtype=torch.float32)
        factory_pinned = torch.empty(8, pin_memory=True)
        factory_pinned.copy_(source)
        method_pinned = source.pin_memory()
        # Factories may choose CUDA while Tensor.pin_memory() chooses Mojo.
        for pinned in (factory_pinned, method_pinned):
            assert pinned.is_pinned()
            assert pinned.pin_memory().data_ptr() == pinned.data_ptr()
            torch.testing.assert_close(pinned.to(mojo_device).cpu(), source)


_PIN_ENTRY_POINTS = ("method", "empty", "empty_like")


def _pin_via(source: torch.Tensor, entry: str) -> torch.Tensor:
    if entry == "method":
        return source.pin_memory()
    if entry == "empty":
        result = torch.empty(
            source.shape, dtype=source.dtype, device="cpu", pin_memory=True
        )
    else:
        assert entry == "empty_like"
        result = torch.empty_like(source, device="cpu", pin_memory=True)
    result.copy_(source)
    return result


class _PinAllocatorProbe:
    def __init__(self, path: Path):
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.lib.uses_mojo_allocator.argtypes = [ctypes.c_void_p]
        self.lib.uses_mojo_allocator.restype = ctypes.c_bool
        self.lib.is_pinned_address.argtypes = [ctypes.c_void_p]
        self.lib.is_pinned_address.restype = ctypes.c_bool

    def uses_mojo_allocator(self, tensor: torch.Tensor) -> bool:
        return self.lib.uses_mojo_allocator(tensor.untyped_storage()._cdata)

    def is_pinned_address(self, address: int) -> bool:
        return self.lib.is_pinned_address(address)


@pytest.fixture(scope="module")
def pin_allocator_probe(tmp_path_factory: pytest.TempPathFactory) -> _PinAllocatorProbe:
    """Inspect ATen's allocator identity, independently of our pin registry.

    Python exposes neither StorageImpl::allocator nor DataPtr's deleter. This
    read-only probe uses ATen's public C++ interface and keeps its Tensor alive.
    """
    directory = tmp_path_factory.mktemp("pin-allocator-probe")
    source = directory / "probe.cpp"
    source.write_text("""
        #include <ATen/Context.h>
        #include <c10/core/StorageImpl.h>
        extern "C" bool uses_mojo_allocator(c10::StorageImpl* storage) {
            return storage->allocator() ==
                at::globalContext().getPinnedMemoryAllocator(
                    c10::DeviceType::PrivateUse1);
        }
        extern "C" bool is_pinned_address(const void* pointer) {
            return at::globalContext().isPinnedPtr(
                pointer, c10::DeviceType::PrivateUse1);
        }
    """)
    output = directory / ("probe.dylib" if sys.platform == "darwin" else "probe.so")
    torch_lib = Path(torch.__file__).parent / "lib"
    proc = subprocess.run(
        [
            *native._cxx(),
            native._cxx_standard(),
            "-shared",
            "-fPIC",
            "-O0",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            *native._torch_include_flags(),
            str(source),
            "-o",
            str(output),
            f"-L{torch_lib}",
            f"-Wl,-rpath,{torch_lib}",
            "-ltorch_cpu",
            "-lc10",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return _PinAllocatorProbe(output)


@pytest.mark.parametrize("entry", _PIN_ENTRY_POINTS)
@pytest.mark.parametrize(
    "configuration", ["cpu-only-wheel", "cuda-available", "cuda-unavailable"]
)
def test_pinned_allocator_provenance_follows_runtime_cuda_availability(
    mojo_device: str,
    pin_allocator_probe: _PinAllocatorProbe,
    entry: str,
    configuration: str,
):
    cuda_available = torch.cuda.is_available()
    if configuration == "cpu-only-wheel":
        if torch.backends.cuda.is_built():
            pytest.skip(
                "requires a CPU-only PyTorch wheel; this wheel includes CUDA/ROCm"
            )
        assert not cuda_available
    else:
        if not torch.backends.cuda.is_built():
            pytest.skip("requires a CUDA/ROCm PyTorch wheel")
        expected_available = configuration == "cuda-available"
        if cuda_available != expected_available:
            pytest.skip(f"requires runtime CUDA availability={expected_available}")
    with device_module.device(mojo_device):
        source = torch.arange(257, dtype=torch.int64)
        pinned = _pin_via(source, entry)
        assert pinned.device == torch.device("cpu")
        assert pinned.is_pinned()
        assert pin_allocator_probe.uses_mojo_allocator(pinned) is (
            entry == "method" or not cuda_available
        )
        assert pinned.pin_memory() is pinned
        torch.testing.assert_close(pinned, source, rtol=0, atol=0)


@pytest.mark.parametrize("entry", _PIN_ENTRY_POINTS)
@pytest.mark.parametrize("shape", [(0,), (3, 0, 5)], ids=["vector", "middle-zero"])
def test_pinned_zero_byte_storage_has_null_pointer(
    mojo_device: str, entry: str, shape: tuple[int, ...]
):
    with device_module.device(mojo_device):
        pinned = _pin_via(torch.empty(shape, device="cpu"), entry)
        assert pinned.shape == shape
        assert pinned.untyped_storage().nbytes() == 0
        assert pinned.untyped_storage().data_ptr() == 0
        assert not pinned.is_pinned()


def test_pinned_empty_strided_zero_byte_storage(mojo_device: str):
    with device_module.device(mojo_device):
        pinned = torch.empty_strided(
            (3, 0, 5), (100, 17, 2), device="cpu", pin_memory=True
        )
        assert pinned.stride() == (100, 17, 2)
        assert pinned.untyped_storage().nbytes() == 0
        assert pinned.untyped_storage().data_ptr() == 0
        assert not pinned.is_pinned()


@pytest.mark.parametrize("entry", _PIN_ENTRY_POINTS)
@pytest.mark.parametrize("offset", [0, 7, 17], ids=["start", "interior", "end"])
def test_pinned_empty_view_retains_nonempty_storage(
    mojo_device: str, entry: str, offset: int
):
    with device_module.device(mojo_device):
        base = _pin_via(torch.arange(17), entry)
        view = base[offset:offset]
        assert view.numel() == 0
        assert view.storage_offset() == offset
        assert view.untyped_storage().nbytes() == 17 * base.element_size()
        assert view.untyped_storage().data_ptr() == base.data_ptr() != 0
        assert view.is_pinned()
        assert view.pin_memory() is view
        del base
        gc.collect()
        assert view.is_pinned()


@pytest.mark.parametrize("entry", _PIN_ENTRY_POINTS)
@pytest.mark.parametrize("shape", [(), (1,), (17,)], ids=["scalar", "one", "tail"])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.float32,
        torch.int64,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
def test_pinned_all_element_widths_preserve_exact_bytes(
    mojo_device: str, entry: str, shape: tuple[int, ...], dtype: torch.dtype
):
    values = torch.arange(math.prod(shape), dtype=torch.int64)
    if dtype == torch.bool:
        source = (values % 2 == 0).reshape(shape)
    elif dtype.is_complex:
        source = torch.complex(values.double() + 0.25, -values.double() - 0.5)
        source = source.to(dtype).reshape(shape)
    else:
        source = (values - 7).to(dtype).reshape(shape)
    with device_module.device(mojo_device):
        pinned = _pin_via(source, entry)
        assert pinned.shape == shape
        assert pinned.dtype == dtype
        assert pinned.is_pinned()
        assert (
            pinned.untyped_storage().nbytes() == source.numel() * source.element_size()
        )
        assert torch.equal(
            pinned.reshape(-1).view(torch.uint8), source.reshape(-1).view(torch.uint8)
        )
        assert pinned.data_ptr() != source.data_ptr()
        assert not source.is_pinned()


@pytest.mark.parametrize("entry", _PIN_ENTRY_POINTS)
@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (255,),
        (256,),
        (257,),
        (4095,),
        (4096,),
        (4097,),
        (357, 789),
        (8 * 1024 * 1024 + 3,),
    ],
    ids=[
        "three",
        "255",
        "256",
        "257",
        "page-minus",
        "page",
        "page-plus",
        "awkward",
        "large",
    ],
)
def test_pinned_size_boundaries_preserve_every_byte(
    mojo_device: str, entry: str, shape: tuple[int, ...]
):
    source = torch.arange(math.prod(shape), dtype=torch.int64) * 37 + 11
    source = source.to(torch.uint8).reshape(shape)
    with device_module.device(mojo_device):
        pinned = _pin_via(source, entry)
        assert pinned.is_pinned()
        assert pinned.untyped_storage().nbytes() == source.numel()
        assert torch.equal(pinned, source)


@pytest.mark.parametrize("entry", _PIN_ENTRY_POINTS)
def test_pinned_cuda_driver_recognizes_first_and_last_page(mojo_gpu: str, entry: str):
    if get_accelerators()[int(mojo_gpu.split(":")[1])].api != "cuda":
        pytest.skip("independent page-registration probe requires the CUDA driver")
    cuda = ctypes.CDLL("libcuda.so.1")
    query = cuda.cuPointerGetAttribute
    query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    query.restype = ctypes.c_int
    with device_module.device(mojo_gpu):
        pinned = _pin_via(torch.arange(4 * 1024 * 1024 + 17).to(torch.uint8), entry)
        for offset in (0, 1, 4095, 4096, pinned.numel() - 1):
            memory_type = ctypes.c_uint()
            # CU_POINTER_ATTRIBUTE_MEMORY_TYPE=2, CU_MEMORYTYPE_HOST=1.
            assert query(ctypes.byref(memory_type), 2, pinned.data_ptr() + offset) == 0
            assert memory_type.value == 1


@pytest.mark.parametrize("kind", ["transpose", "gapped", "offset", "as-strided"])
def test_pin_memory_preserves_strides_and_independent_storage(
    mojo_device: str, kind: str
):
    base = torch.arange(128, dtype=torch.int64)
    source = {
        "transpose": base.reshape(8, 16).t(),
        "gapped": base.reshape(8, 16)[1::2, 1::3],
        "offset": base[1:18],
        "as-strided": base.as_strided((3, 4), (31, 3), 5),
    }[kind]
    expected = source.clone()
    with device_module.device(mojo_device):
        pinned = source.pin_memory()
        assert pinned.is_pinned()
        assert pinned.stride() == source.stride()
        assert pinned.storage_offset() == 0
        span = 1 + sum(
            (size - 1) * stride
            for size, stride in zip(source.shape, source.stride(), strict=True)
        )
        assert pinned.untyped_storage().nbytes() == span * source.element_size()
        assert torch.equal(pinned, expected)
        pinned.fill_(-731)
        assert torch.equal(source, expected)
        source.fill_(923)
        assert torch.equal(pinned, torch.full_like(expected, -731))
        assert not source.is_pinned()


@pytest.mark.parametrize(
    "kind", ["offset", "end", "strided", "transpose", "overlap", "detach", "buffer"]
)
def test_pinned_alias_outlives_base_and_preserves_canaries(mojo_device: str, kind: str):
    with device_module.device(mojo_device):
        expected = torch.arange(65, dtype=torch.uint8)
        base = expected.pin_memory()
        if kind == "offset":
            view = base[1:18]
        elif kind == "end":
            view = base[49:]
        elif kind == "strided":
            view = base[1:50:3]
        elif kind == "transpose":
            view = base[1:49].view(6, 8).t()
        elif kind == "overlap":
            view = base[3:4].expand(17)
        elif kind == "detach":
            view = base.detach()
        else:
            # NumPy's buffer retains its Tensor owner. Unlike an ordinary slice,
            # this storage really begins at an interior byte of the allocation.
            view = torch.frombuffer(memoryview(base.numpy())[1:18], dtype=torch.uint8)
            assert view.untyped_storage().data_ptr() == base.data_ptr() + 1
        snapshot = view.clone()
        guard = base.detach()
        clone = view.clone()
        assert not clone.is_pinned()
        assert view.pin_memory() is view
        del base
        gc.collect()
        pressure = [
            torch.full((65,), i + 100, dtype=torch.uint8).pin_memory() for i in range(8)
        ]
        assert view.is_pinned()
        assert torch.equal(view, snapshot)
        # Mutating only the addressed elements must leave all other bytes intact.
        if kind != "overlap":
            offsets = view.data_ptr() - guard.data_ptr()
            indices = torch.empty_strided(view.shape, view.stride(), dtype=torch.int64)
            for index in np.ndindex(tuple(view.shape)):
                indices[index] = offsets + sum(
                    i * s for i, s in zip(index, view.stride(), strict=True)
                )
            expected[indices.reshape(-1)] = 231
            view.fill_(231)
            assert torch.equal(guard, expected)
        for i, block in enumerate(pressure):
            assert torch.equal(block, torch.full_like(block, i + 100))


def test_pin_memory_overlapping_pageable_view_errors_but_pinned_view_is_identity(
    mojo_device: str,
):
    with device_module.device(mojo_device):
        source = torch.tensor([37.0]).expand(3, 5)
        with pytest.raises(
            RuntimeError, match="more than one element.*single memory location"
        ):
            source.pin_memory()
        assert torch.equal(source, torch.full((3, 5), 37.0))
        pinned = torch.tensor([37.0]).pin_memory().expand(3, 5)
        assert pinned.is_pinned()
        assert pinned.pin_memory() is pinned
        assert pinned.stride() == (0, 0)


@pytest.mark.parametrize(
    "kind", ["transpose", "gapped", "expanded", "channels-last", "channels-last-3d"]
)
@pytest.mark.parametrize(
    "memory_format", [torch.preserve_format, torch.contiguous_format]
)
def test_pinned_empty_like_memory_format(
    mojo_device: str, kind: str, memory_format: torch.memory_format
):
    templates = {
        "transpose": torch.arange(24).reshape(4, 6).t(),
        "gapped": torch.arange(48).reshape(6, 8)[::2, ::2],
        "expanded": torch.tensor([31]).expand(3, 4),
        "channels-last": torch.arange(120)
        .reshape(2, 3, 4, 5)
        .contiguous(memory_format=torch.channels_last),
        "channels-last-3d": torch.arange(360)
        .reshape(2, 3, 3, 4, 5)
        .contiguous(memory_format=torch.channels_last_3d),
    }
    source = templates[kind]
    expected = torch.empty_like(source, pin_memory=False, memory_format=memory_format)
    with device_module.device(mojo_device):
        pinned = torch.empty_like(
            source, device="cpu", pin_memory=True, memory_format=memory_format
        )
        assert pinned.is_pinned()
        assert pinned.stride() == expected.stride()
        assert pinned.data_ptr() != source.data_ptr()
        pinned.copy_(source)
        assert torch.equal(pinned, source)


@pytest.mark.parametrize("entry", ["empty", "empty_like"])
@pytest.mark.parametrize("pin_option", [None, False], ids=["omitted", "false"])
@pytest.mark.parametrize("pinned_template", [False, True])
def test_unpinned_factories_do_not_inherit_template_pinning(
    mojo_device: str, entry: str, pin_option: bool | None, pinned_template: bool
):
    with device_module.device(mojo_device):
        source = torch.arange(17)
        if pinned_template:
            source = source.pin_memory()
        if entry == "empty_like":
            result = (
                torch.empty_like(source, device="cpu")
                if pin_option is None
                else torch.empty_like(source, device="cpu", pin_memory=False)
            )
        else:
            result = (
                torch.empty(17, device="cpu")
                if pin_option is None
                else torch.empty(17, device="cpu", pin_memory=False)
            )
        assert not result.is_pinned()
        assert source.is_pinned() is pinned_template


@pytest.mark.parametrize("kind", ["ordinary", "numpy", "shared"])
def test_pin_query_rejects_interleaved_pageable_allocations(
    mojo_device: str, kind: str
):
    with device_module.device(mojo_device):
        pinned = [torch.arange(n).pin_memory() for n in (1, 17, 4097)]
        for n in (0, 1, 3, 4097):
            if kind == "numpy":
                source = torch.from_numpy(np.arange(n, dtype=np.int64))
            else:
                source = torch.arange(n)
                if kind == "shared":
                    source.share_memory_()
            assert not source.is_pinned()
        assert all(block.is_pinned() for block in pinned)


@pytest.mark.parametrize("shape", [(0,), (17,)])
def test_mojo_tensor_is_not_host_pinned_and_cannot_be_pinned(
    mojo_device: str, shape: tuple[int, ...]
):
    tensor = torch.empty(shape, device=mojo_device)
    assert not tensor.is_pinned()
    with pytest.raises(RuntimeError, match="only dense CPU tensors can be pinned"):
        tensor.pin_memory()
    assert tensor.device == torch.device(mojo_device)


@pytest.mark.parametrize("target", ["meta", "cuda"])
@pytest.mark.parametrize("size", [0, 17])
def test_other_non_cpu_tensors_are_not_host_pinned(target: str, size: int):
    if target == "cuda" and not torch.cuda.is_available():
        pytest.skip("non-CPU CUDA tensor requires runtime CUDA availability")
    tensor = torch.empty(size, device=target)
    assert not tensor.is_pinned()
    with pytest.raises(RuntimeError, match="only dense CPU tensors can be pinned"):
        tensor.pin_memory()


@pytest.mark.parametrize("entry", ["empty", "empty_like", "empty_strided"])
def test_pinned_factory_rejects_mojo_output(mojo_device: str, entry: str):
    source = torch.empty(3, device=mojo_device)
    with pytest.raises(RuntimeError, match="[Pp]in|CPU"):
        if entry == "empty":
            torch.empty(3, device=mojo_device, pin_memory=True)
        elif entry == "empty_strided":
            torch.empty_strided((3,), (2,), device=mojo_device, pin_memory=True)
        else:
            torch.empty_like(source, pin_memory=True)


@pytest.mark.parametrize("query_device", ["mojo", "mojo:0", "cpu", "meta"])
def test_is_pinned_explicit_device_type_routing(mojo_device: str, query_device: str):
    with device_module.device(mojo_device):
        pinned = torch.arange(17).pin_memory()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            if query_device.startswith("mojo"):
                assert pinned.is_pinned(device=query_device)
                assert pinned.pin_memory(device=query_device) is pinned
            else:
                # Context::isPinnedPtr rejects non-accelerator device types
                # with false before looking up a hook (Context.h).
                assert not pinned.is_pinned(device=query_device)
        assert device_module.current_device() == int(mojo_device.split(":")[1])


def test_is_pinned_explicit_cuda_bypasses_mojo_hook():
    # A MAX CPU HostBuffer is known to Mojo but is not CUDA-registered. Using
    # it distinguishes the hooks even on a machine with a working CUDA wheel.
    if torch.cuda.is_available():
        torch.cuda.init()  # Context::isPinnedPtr otherwise has its own cold guard.
    with device_module.device(device_module.device_count() - 1):
        pinned = torch.arange(17).pin_memory()
        assert pinned.is_pinned()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            assert not pinned.is_pinned(device="cuda")
            assert pinned.is_pinned(device="mojo")
        assert pinned.is_pinned()


def test_pinned_query_does_not_depend_on_current_device(mojo_device: str):
    original = device_module.current_device()
    with device_module.device(mojo_device):
        pinned = torch.arange(17).pin_memory()
    for index in range(device_module.device_count()):
        with device_module.device(index):
            assert pinned.is_pinned()
            assert device_module.current_device() == index
    assert device_module.current_device() == original


@pytest.mark.parametrize(
    "layout",
    [
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    ],
)
@pytest.mark.parametrize("empty", [False, True], ids=["nonempty", "empty"])
def test_sparse_pin_memory_pins_components(
    mojo_device: str, layout: torch.layout, empty: bool
):
    dense = (
        torch.zeros((4, 4), dtype=torch.float64)
        if empty
        else torch.tensor(
            [
                [0.0, 2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 5.0],
                [0.0, 0.0, 7.0, 0.0],
                [11.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
    )
    blocksize = (2, 2) if layout in (torch.sparse_bsr, torch.sparse_bsc) else None
    source = dense.to_sparse(layout=layout, blocksize=blocksize)
    with device_module.device(mojo_device):
        pinned = source.pin_memory()
        assert pinned.layout == layout
        assert pinned.device == torch.device("cpu")
        assert torch.equal(pinned.to_dense(), dense)
        assert pinned.is_pinned() is (not empty)
        if layout == torch.sparse_coo:
            parts = (pinned._indices(), pinned._values())
            originals = (source._indices(), source._values())
        elif layout in (torch.sparse_csr, torch.sparse_bsr):
            parts = (pinned.crow_indices(), pinned.col_indices(), pinned.values())
            originals = (source.crow_indices(), source.col_indices(), source.values())
        else:
            parts = (pinned.ccol_indices(), pinned.row_indices(), pinned.values())
            originals = (source.ccol_indices(), source.row_indices(), source.values())
        for part, original in zip(parts, originals, strict=True):
            assert part.is_pinned() is (part.untyped_storage().nbytes() != 0)
            assert not original.is_pinned()
            assert torch.equal(part, original)
            if part.numel():
                assert part.data_ptr() != original.data_ptr()
        if not empty:
            assert pinned.pin_memory() is pinned
        else:
            assert torch.equal(pinned.pin_memory().to_dense(), dense)


@pytest.mark.parametrize("entry", _PIN_ENTRY_POINTS)
def test_pinned_python_threads_allocate_disjoint_live_blocks(
    mojo_device: str, entry: str
):
    # Fresh first use and a process deadline: a deadlocked allocator must not
    # hang pytest in ThreadPoolExecutor.__exit__ waiting for its worker threads.
    script = textwrap.dedent("""
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        import torch
        from torch_mojo_backend import register_mojo_devices
        from torch_mojo_backend.native import device_module

        register_mojo_devices()
        barrier = Barrier(4, timeout=60)

        def allocate(worker: int) -> list[torch.Tensor]:
            with device_module.device(sys.argv[1]):
                barrier.wait()
                blocks = []
                for i in range(8):
                    source = torch.arange(257) + worker * 10000 + i * 300
                    if sys.argv[2] == 'method':
                        pinned = source.pin_memory()
                    elif sys.argv[2] == 'empty':
                        pinned = torch.empty(257, dtype=source.dtype, device='cpu', pin_memory=True)
                        pinned.copy_(source)
                    else:
                        pinned = torch.empty_like(source, device='cpu', pin_memory=True)
                        pinned.copy_(source)
                    assert pinned.is_pinned()
                    blocks.append(pinned)
                assert device_module.current_device() == int(sys.argv[1].split(':')[1])
            return blocks

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(allocate, range(4)))
        ranges = sorted((block.data_ptr(), block.data_ptr() + block.untyped_storage().nbytes())
                        for blocks in results for block in blocks)
        assert all(end <= start for (_, end), (start, _) in zip(ranges, ranges[1:]))
        for worker, blocks in enumerate(results):
            for i, block in enumerate(blocks):
                assert block.is_pinned()
                assert torch.equal(block, torch.arange(257) + worker * 10000 + i * 300)
    """)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONEXECUTABLE")
    }
    proc = subprocess.run(
        [sys.executable, "-c", script, mojo_device, entry],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


@pytest.mark.parametrize("kind", ["negative-size", "overflow", "overflow-strides"])
def test_invalid_pinned_allocation_arithmetic_errors_and_recovers(
    mojo_device: str, kind: str
):
    with device_module.device(mojo_device):
        with pytest.raises(RuntimeError, match="negative|overflow|Storage size"):
            if kind == "negative-size":
                torch.empty((-1, 3), device="cpu", pin_memory=True)
            elif kind == "overflow":
                torch.empty(
                    (2**62, 8), dtype=torch.float64, device="cpu", pin_memory=True
                )
            else:
                torch.empty_strided(
                    (3, 3),
                    (2**62, 1),
                    dtype=torch.float64,
                    device="cpu",
                    pin_memory=True,
                )
        pinned = torch.arange(17).pin_memory()
        assert pinned.is_pinned()
        assert torch.equal(pinned, torch.arange(17))


def test_method_pinned_storage_cannot_grow(mojo_device: str):
    with device_module.device(mojo_device):
        pinned = torch.arange(17).pin_memory()
        storage = pinned.untyped_storage()
        assert not storage.resizable()
        with pytest.raises(RuntimeError, match="not resizable"):
            storage.resize_(storage.nbytes() + 4096)
        assert pinned.is_pinned()
        assert torch.equal(pinned, torch.arange(17))


def test_pinned_pointer_query_exact_live_range_on_max_cpu(
    pin_allocator_probe: _PinAllocatorProbe,
):
    with device_module.device(device_module.device_count() - 1):
        pinned = torch.arange(257, dtype=torch.uint8).pin_memory()
        base = pinned.data_ptr()
        end = base + pinned.untyped_storage().nbytes()
        # These are address-only hook queries, including the two out-of-bounds
        # addresses. They never construct a readable tensor or touch those bytes.
        assert pin_allocator_probe.is_pinned_address(base)
        assert pin_allocator_probe.is_pinned_address(end - 1)
        assert not pin_allocator_probe.is_pinned_address(base - 1)
        assert not pin_allocator_probe.is_pinned_address(end)
        assert not pin_allocator_probe.is_pinned_address(0)
        assert torch.equal(pinned, torch.arange(257).to(torch.uint8))


@pytest.mark.parametrize(
    "dtype_name",
    [
        "uint16",
        "uint32",
        "uint64",
        "complex32",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    ],
)
def test_pinned_additional_installed_host_dtypes(mojo_device: str, dtype_name: str):
    if not hasattr(torch, dtype_name):
        pytest.skip(f"installed PyTorch does not expose torch.{dtype_name}")
    dtype = getattr(torch, dtype_name)
    source = torch.arange(1, 18, dtype=torch.float32).to(dtype)
    with device_module.device(mojo_device):
        pinned = source.pin_memory()
        assert pinned.dtype == dtype
        assert pinned.is_pinned()
        assert torch.equal(pinned.view(torch.uint8), source.view(torch.uint8))


def test_pinned_method_preserves_autograd(mojo_device: str):
    with device_module.device(mojo_device):
        source = torch.arange(1, 18, dtype=torch.float64, requires_grad=True)
        pinned = source.pin_memory()
        assert pinned.is_pinned()
        (pinned * pinned).sum().backward()
        assert source.grad is not None
        assert torch.equal(source.grad, 2 * source.detach())
        with torch.inference_mode():
            inference = torch.arange(17).pin_memory()
            assert inference.is_pinned()
            assert torch.equal(inference, torch.arange(17))


@pytest.mark.parametrize("entry", _PIN_ENTRY_POINTS)
def test_pinned_factories_without_runtime_cuda(
    pin_allocator_probe: _PinAllocatorProbe, entry: str
):
    # Hiding accelerators exercises a CUDA wheel with hasCUDA()==false. The
    # same test also works with a CPU-only wheel; no wheel-branding oracle.
    script = textwrap.dedent("""
        import ctypes
        import sys
        import torch
        from torch_mojo_backend import register_mojo_devices
        from torch_mojo_backend.native import device_module

        assert not torch.cuda.is_available()
        register_mojo_devices()
        assert device_module.device_count() == 1
        probe = ctypes.CDLL(sys.argv[1])
        probe.uses_mojo_allocator.argtypes = [ctypes.c_void_p]
        probe.uses_mojo_allocator.restype = ctypes.c_bool
        source = torch.arange(17)
        if sys.argv[2] == 'method':
            pinned = source.pin_memory()
        elif sys.argv[2] == 'empty':
            pinned = torch.empty(17, dtype=source.dtype, device='cpu', pin_memory=True)
        else:
            pinned = torch.empty_like(source, device='cpu', pin_memory=True)
        assert pinned.is_pinned()
        assert probe.uses_mojo_allocator(pinned.untyped_storage()._cdata)
        assert pinned.pin_memory() is pinned
        for batch in torch.utils.data.DataLoader([source], batch_size=None, pin_memory=True):
            assert batch.is_pinned()
            assert torch.equal(batch, source)
    """)
    if any(device.api == "metal" for device in get_accelerators()):
        pytest.skip("CUDA/HIP visibility variables cannot hide a Metal accelerator")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONEXECUTABLE")
    }
    env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, "-c", script, str(pin_allocator_probe.path), entry],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def _cuda_pinned_allocation_device(tensor: torch.Tensor) -> int | None:
    """Driver registration ordinal; None for the MAX CPU's host allocation."""
    cuda = ctypes.CDLL("libcuda.so.1")
    query = cuda.cuPointerGetAttribute
    query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64]
    query.restype = ctypes.c_int
    ordinal = ctypes.c_int(-1)
    status = query(ctypes.byref(ordinal), 9, tensor.data_ptr())
    if status == 1:  # CUDA_ERROR_INVALID_VALUE: no registration for this pointer.
        return None
    assert status == 0
    return ordinal.value


@pytest.mark.parametrize("two_gpus", [False, True], ids=["gpu-vs-max-cpu", "two-gpus"])
def test_explicit_pin_device_index_does_not_override_current_device(
    mojo_gpu: str, two_gpus: bool
):
    if get_accelerators()[0].api != "cuda":
        pytest.skip("allocation-device conformance probe requires the CUDA driver")
    if two_gpus and device_module.device_count() < 3:
        pytest.skip("requires two physical Mojo GPUs; MAX CPU is not a second GPU")
    other = 1 if two_gpus else device_module.device_count() - 1
    for current, argument in ((0, other), (other, 0)):
        with device_module.device(current), warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pinned = torch.arange(17).pin_memory(device=f"mojo:{argument}")
            assert pinned.is_pinned(device=f"mojo:{argument}")
            assert device_module.current_device() == current
            expected = None if current == device_module.device_count() - 1 else current
            assert _cuda_pinned_allocation_device(pinned) == expected


@pytest.mark.parametrize("two_gpus", [False, True], ids=["gpu-vs-max-cpu", "two-gpus"])
def test_pinned_threads_keep_their_own_allocation_device(mojo_gpu: str, two_gpus: bool):
    if get_accelerators()[0].api != "cuda":
        pytest.skip("thread allocation-device probe requires the CUDA driver")
    if two_gpus and device_module.device_count() < 3:
        pytest.skip("requires two physical Mojo GPUs")
    other = 1 if two_gpus else device_module.device_count() - 1
    barrier = Barrier(2, timeout=30)

    def allocate(index: int) -> torch.Tensor:
        with device_module.device(index):
            barrier.wait()
            result = torch.arange(17).pin_memory()
            barrier.wait()
            assert device_module.current_device() == index
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        gpu, alternate = list(executor.map(allocate, (0, other)))
    assert _cuda_pinned_allocation_device(gpu) == 0
    assert _cuda_pinned_allocation_device(alternate) == (other if two_gpus else None)
    assert gpu.is_pinned() and alternate.is_pinned()


@pytest.mark.parametrize("entry", _PIN_ENTRY_POINTS)
def test_pinned_allocation_failure_preserves_source_and_recovers(
    mojo_device: str, entry: str, pinned_host_failure_preload: Path
):
    if not sys.platform.startswith("linux"):
        pytest.skip(
            "controlled address-space exhaustion needs Linux /proc and RLIMIT_AS"
        )
    script = textwrap.dedent("""
        import ctypes
        import gc
        import os
        import resource
        import sys
        from pathlib import Path
        import torch
        from torch_mojo_backend import register_mojo_devices
        from torch_mojo_backend.native import device_module

        torch.set_num_threads(1)
        register_mojo_devices()
        with device_module.device(sys.argv[1]):
            for warm in (torch.empty(17).pin_memory(), torch.empty(17, device='cpu', pin_memory=True)):
                assert warm.is_pinned()
            del warm
            # Allocate the input and oracle before imposing the limit. The
            # request is a budgeted 128 MiB, not an arbitrary enormous size.
            source = torch.arange(32 * 1024 * 1024, dtype=torch.int32)
            expected = source.clone()
            gc.collect()
            pages = int(Path('/proc/self/statm').read_text().split()[0])
            current = pages * resource.getpagesize()
            old_limit = resource.getrlimit(resource.RLIMIT_AS)
            ceiling = current + 8 * 1024 * 1024
            if old_limit[1] != resource.RLIM_INFINITY:
                ceiling = min(ceiling, old_limit[1])
            message = ''
            # MAX may serve allocations from reserved address space, so
            # RLIMIT_AS alone is not a fault injector for its HostBuffers.
            # Interpose only its documented createHostBuffer C ABI; CUDA
            # factories still fail against the address-space limit below.
            os.environ['TMB_TEST_FAIL_HOST_BUFFER'] = '1'
            resource.setrlimit(resource.RLIMIT_AS, (ceiling, old_limit[1]))
            try:
                try:
                    if sys.argv[2] == 'method':
                        source.pin_memory()
                    elif sys.argv[2] == 'empty':
                        torch.empty(source.shape, dtype=source.dtype, device='cpu', pin_memory=True)
                    else:
                        torch.empty_like(source, device='cpu', pin_memory=True)
                except RuntimeError as error:
                    message = str(error)
            finally:
                resource.setrlimit(resource.RLIMIT_AS, old_limit)
                del os.environ['TMB_TEST_FAIL_HOST_BUFFER']
            assert message, 'controlled allocation unexpectedly succeeded'
            assert any(word in message.lower() for word in ('allocat', 'memory')), message
            probe = ctypes.CDLL(sys.argv[3])
            expected_faults = int(sys.argv[2] == 'method' or not torch.cuda.is_available())
            assert probe.tmb_test_host_buffer_failures() == expected_faults
            assert not source.is_pinned()
            assert torch.equal(source, expected)
            for recovered in (torch.arange(17).pin_memory(),
                              torch.empty(17, device='cpu', pin_memory=True),
                              torch.empty_like(torch.arange(17), device='cpu', pin_memory=True)):
                assert recovered.is_pinned()
            print('allocation failure recovered', flush=True)
    """)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONEXECUTABLE")
    }
    env["LD_PRELOAD"] = str(pinned_host_failure_preload) + (
        ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            mojo_device,
            entry,
            str(pinned_host_failure_preload),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "allocation failure recovered" in proc.stdout


def test_non_blocking_download_falls_back_after_pinned_allocation_failure(
    mojo_device: str, pinned_host_failure_preload: Path
):
    """Only automatic download pinning falls back, with completed CPU values."""
    script = textwrap.dedent("""
        import ctypes
        import os
        import sys
        from contextlib import nullcontext
        from threading import Timer
        import torch
        from tests.test_mojo_device import _held_transfer_stream
        from torch_mojo_backend import get_accelerators, register_mojo_devices
        from torch_mojo_backend.native import device_module

        torch.set_num_threads(1)
        register_mojo_devices()
        device = sys.argv[1]
        probe = ctypes.CDLL(sys.argv[2])
        with device_module.device(device):
            expected = torch.arange(1 << 20, dtype=torch.int32)
            source = expected.to(device)
            warm = source.to('cpu', non_blocking=True)
            torch.accelerator.synchronize(device)
            assert warm.is_pinned()
            del warm
            stream = device_module.current_stream(device)
            cuda = get_accelerators()[stream.device_index].api == 'cuda'
            gate = _held_transfer_stream(stream) if cuda else nullcontext(None)
            with gate as release:
                pending = stream.record_event() if cuda else None
                timer = Timer(0.2, release.set) if release is not None else None
                if timer is not None:
                    timer.start()
                os.environ['TMB_TEST_FAIL_HOST_BUFFER'] = '1'
                try:
                    result = source.to('cpu', non_blocking=True)
                    assert probe.tmb_test_host_buffer_failures() == 1
                    assert not result.is_pinned()
                    if pending is not None:
                        assert pending.query(), 'pageable fallback must finish DMA'
                    torch.testing.assert_close(result, expected)
                finally:
                    del os.environ['TMB_TEST_FAIL_HOST_BUFFER']
                    if timer is not None:
                        timer.join()
            recovered = source.to('cpu', non_blocking=True)
            torch.accelerator.synchronize(device)
            assert recovered.is_pinned()
            torch.testing.assert_close(recovered, expected)
            # Transfer failures must propagate both with and without fallback.
            for fail_allocation in (False, True):
                os.environ['TMB_TEST_FAIL_DOWNLOAD'] = '1'
                if fail_allocation:
                    os.environ['TMB_TEST_FAIL_HOST_BUFFER'] = '1'
                try:
                    try:
                        source.to('cpu', non_blocking=True)
                    except RuntimeError as error:
                        assert 'injected download failure' in str(error), str(error)
                    else:
                        raise AssertionError('download error was swallowed')
                finally:
                    del os.environ['TMB_TEST_FAIL_DOWNLOAD']
                    os.environ.pop('TMB_TEST_FAIL_HOST_BUFFER', None)
            print('download allocation fallback recovered', flush=True)
    """)
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTHONHOME", "PYTHONEXECUTABLE")
    }
    env["LD_PRELOAD"] = str(pinned_host_failure_preload) + (
        ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, mojo_device, str(pinned_host_failure_preload)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "download allocation fallback recovered" in proc.stdout


@pytest.fixture(scope="module")
def pinned_host_failure_preload(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Subprocess-only faults at MAX's C ABI, with no production test switch."""
    if not sys.platform.startswith("linux"):
        pytest.skip("MAX allocation fault injection requires Linux LD_PRELOAD")
    directory = tmp_path_factory.mktemp("host-buffer-failure")
    source = directory / "fail_host_buffer.cpp"
    source.write_text("""
        #include <dlfcn.h>
        #include <stddef.h>
        #include <stdlib.h>
        #include <string.h>
        static int failures = 0;
        extern "C" int tmb_test_host_buffer_failures() { return failures; }
        extern "C" const char* AsyncRT_DeviceContext_createHostBuffer(
            void** result, void** pointer, const void* context,
            size_t count, size_t itemsize) {
            if (getenv("TMB_TEST_FAIL_HOST_BUFFER")) {
                ++failures;
                *result = nullptr;
                *pointer = nullptr;
                // MAX consumes errors with AsyncRT_DeviceContext_strfree.
                return strdup("injected host memory allocation failure");
            }
            using Create = const char* (*)(void**, void**, const void*, size_t, size_t);
            auto create = reinterpret_cast<Create>(dlsym(
                RTLD_NEXT, "AsyncRT_DeviceContext_createHostBuffer"));
            if (!create) return strdup("test interposer could not find MAX host allocator");
            return create(result, pointer, context, count, itemsize);
        }
        extern "C" const char* AsyncRT_DeviceContext_DtoH_async(
            const void* context, void* destination, const void* source) {
            if (getenv("TMB_TEST_FAIL_DOWNLOAD"))
                return strdup("injected download failure");
            using Download = const char* (*)(const void*, void*, const void*);
            auto download = reinterpret_cast<Download>(dlsym(
                RTLD_NEXT, "AsyncRT_DeviceContext_DtoH_async"));
            if (!download) return strdup("test interposer could not find MAX download");
            return download(context, destination, source);
        }
    """)
    output = directory / "fail_host_buffer.so"
    proc = subprocess.run(
        [*native._cxx(), "-shared", "-fPIC", str(source), "-o", str(output), "-ldl"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return output


def test_pinned_view_is_sole_surviving_storage_owner(mojo_device: str):
    with device_module.device(mojo_device):
        base = torch.arange(257, dtype=torch.int64).pin_memory()
        view = base[3::7]
        expected = torch.arange(257)[3::7]
        del base
        gc.collect()
        pressure = [
            torch.full((257,), -i - 17, dtype=torch.int64).pin_memory()
            for i in range(8)
        ]
        assert view.is_pinned()
        assert torch.equal(view, expected)
        assert all(block.is_pinned() for block in pressure)


@pytest.mark.parametrize("strided", [False, True])
def test_pinned_round_trip(mojo_device: str, strided: bool):
    with device_module.device(mojo_device):
        source = torch.arange(128, dtype=torch.float32).reshape(8, 16)
        if strided:
            source = source[1::2, 1::2]
        pinned = source.pin_memory()
        assert pinned.is_pinned()
        assert pinned.stride() == source.stride()
        uploaded = pinned.to(mojo_device)
        assert not uploaded.is_pinned()
        torch.testing.assert_close(uploaded.cpu(), source)
        downloaded = torch.empty_like(pinned, pin_memory=True)
        downloaded.copy_(uploaded)
        assert downloaded.is_pinned()
        torch.testing.assert_close(downloaded, source)


def _host_pointer_alias(address: int) -> torch.Tensor:
    """A non-owning storage for pointer queries; never read its contents."""
    return torch.frombuffer(
        (ctypes.c_uint8 * 1).from_address(address), dtype=torch.uint8
    )


def test_pinned_interior_pointers(mojo_device: str):
    with device_module.device(mojo_device):
        pinned = torch.arange(64, dtype=torch.float32).pin_memory()
        view = pinned[16:]
        base = pinned.data_ptr()
        end = base + pinned.numel() * pinned.element_size()
        assert base < view.data_ptr() < end
        assert view.is_pinned()
        # A slice queries the storage base; these storages start inside it.
        for address in (base + 1, view.data_ptr(), end - 1):
            assert _host_pointer_alias(address).is_pinned()
        del pinned
        assert view.is_pinned()


def _current_rss_bytes() -> int | None:
    """Current resident memory, or None when procfs does not expose it."""
    try:
        status = Path("/proc/self/status").read_text()
    except FileNotFoundError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return None


@pytest.mark.parametrize("async_use", [False, True])
def test_pinned_allocations_are_released(mojo_device: str, async_use: bool):
    """Catch leaked HostBuffers that were not destroyed after their tensors died.

    Current RSS bounds retained memory after churn regardless of earlier peaks.
    Without procfs, the churn and fresh-allocation checks still run.
    """
    with device_module.device(mojo_device):
        # Tensor.pin_memory() uses our allocator even with a CUDA torch wheel.
        source = torch.zeros(4 * 1024 * 1024, dtype=torch.uint8)
        uploaded = torch.empty_like(source, device=mojo_device)
        warm = source.pin_memory()
        if async_use:
            uploaded.copy_(warm, non_blocking=True)
        del warm
        torch.accelerator.synchronize(mojo_device)
        before = _current_rss_bytes()
        # Churn 256 MiB in total with only 32 MiB of live pinned blocks.
        for _ in range(8):
            blocks = [source.pin_memory() for _ in range(8)]
            while blocks:
                block = blocks.pop(len(blocks) // 2)
                if async_use:
                    uploaded.copy_(block, non_blocking=True)
                del block
                assert all(live.is_pinned() for live in blocks)
        torch.accelerator.synchronize(mojo_device)
        after = _current_rss_bytes()
        if before is not None and after is not None:
            assert after - before < 128 * 1024 * 1024
        fresh = torch.empty(8).pin_memory()
        assert fresh.is_pinned()


@pytest.mark.filterwarnings("ignore:pin_memory_device is deprecated:UserWarning")
def test_pinned_dataloader(mojo_device: str):
    with device_module.device(mojo_device):
        source = torch.arange(32, dtype=torch.float32).reshape(8, 4)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(source),
            batch_size=2,
            num_workers=0,
            pin_memory=True,
            pin_memory_device=mojo_device,
        )
        batches = [batch[0] for batch in loader]
        assert all(batch.is_pinned() for batch in batches)
        torch.testing.assert_close(torch.cat(batches), source)


@pytest.mark.parametrize("workers", ["none", "fork", "spawn"])
def test_pinned_dataloader_worker_modes_preserve_values_and_method_provenance(
    mojo_device: str, workers: str, pin_allocator_probe: _PinAllocatorProbe
):
    if workers != "none" and workers not in multiprocessing.get_all_start_methods():
        pytest.skip(f"multiprocessing start method {workers!r} is unavailable")
    source = (torch.arange(68) * 37 + 11).reshape(17, 4)
    with device_module.device(mojo_device):
        loader = DataLoader(
            TensorDataset(source),
            batch_size=3,
            num_workers=0 if workers == "none" else 2,
            multiprocessing_context=None if workers == "none" else workers,
            pin_memory=True,
            timeout=0 if workers == "none" else 60,
        )
        batches = [batch[0] for batch in loader]
        assert len(batches) == 6
        assert all(
            batch.device.type == "cpu" and batch.is_pinned() for batch in batches
        )
        assert all(pin_allocator_probe.uses_mojo_allocator(batch) for batch in batches)
        assert torch.equal(torch.cat(batches), source)
        assert not source.is_pinned()


class _NamedPinSample(NamedTuple):
    value: torch.Tensor
    label: str


_Sample = TypeVar("_Sample")


class _SinglePinSample(Dataset[_Sample], Generic[_Sample]):
    def __init__(self, sample: _Sample):
        self.sample = sample

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> _Sample:
        if index != 0:
            raise IndexError(index)
        return self.sample


@dataclass
class _CustomPinSample:
    value: torch.Tensor
    pin_calls: int = 0

    def pin_memory(self) -> "_CustomPinSample":
        return _CustomPinSample(self.value.pin_memory(), self.pin_calls + 1)


def test_pinned_dataloader_nested_and_custom_batches(mojo_device: str):
    source = {
        "tensor": torch.arange(17),
        "list": [torch.tensor(31), "text", 19],
        "tuple": (torch.tensor([7, 11]), "pair"),
        "named": _NamedPinSample(torch.tensor([29]), "named"),
        "empty": torch.empty(3, 0, 5),
        "custom": _CustomPinSample(torch.tensor([43, 47])),
    }
    with device_module.device(mojo_device):
        batch = next(
            iter(DataLoader(_SinglePinSample(source), batch_size=None, pin_memory=True))
        )
        assert torch.equal(batch["tensor"], source["tensor"])
        assert batch["tensor"].is_pinned()
        assert batch["list"][0].shape == ()
        assert batch["list"][0].item() == 31 and batch["list"][0].is_pinned()
        assert batch["list"][1:] == ["text", 19]
        # Upstream intentionally turns ordinary tuples into lists.
        assert isinstance(batch["tuple"], list)
        assert torch.equal(batch["tuple"][0], torch.tensor([7, 11]))
        assert batch["tuple"][0].is_pinned() and batch["tuple"][1] == "pair"
        assert isinstance(batch["named"], _NamedPinSample)
        assert batch["named"].value.is_pinned()
        assert batch["named"].label == "named"
        assert batch["empty"].shape == (3, 0, 5) and not batch["empty"].is_pinned()
        assert isinstance(batch["custom"], _CustomPinSample)
        assert batch["custom"].pin_calls == 1
        assert batch["custom"].value.is_pinned()
        assert torch.equal(batch["custom"].value, torch.tensor([43, 47]))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_pinned_dataloader_pin_error_reaches_consumer_and_recovers(
    mojo_device: str, num_workers: int
):
    source = torch.tensor([37.0]).expand(3, 5)
    with device_module.device(mojo_device):
        loader = DataLoader(
            _SinglePinSample(source),
            batch_size=None,
            num_workers=num_workers,
            pin_memory=True,
            timeout=60 if num_workers else 0,
        )
        iterator = iter(loader)
        try:
            with pytest.raises(
                RuntimeError, match="more than one element.*single memory location"
            ):
                next(iterator)
        finally:
            if num_workers:
                assert isinstance(iterator, _MultiProcessingDataLoaderIter)
                iterator._shutdown_workers()
            del iterator, loader
            gc.collect()
        assert torch.equal(source, torch.full((3, 5), 37.0))
        batch = next(
            iter(
                DataLoader(
                    _SinglePinSample(torch.arange(17)), batch_size=None, pin_memory=True
                )
            )
        )
        assert batch.is_pinned()
        assert torch.equal(batch, torch.arange(17))


def test_pinned_dataloader_persistent_workers_and_early_shutdown(mojo_device: str):
    source = torch.arange(68).reshape(17, 4)
    with device_module.device(mojo_device):
        for _ in range(2):
            loader = DataLoader(
                TensorDataset(source),
                batch_size=3,
                num_workers=2,
                persistent_workers=True,
                pin_memory=True,
                timeout=60,
            )
            iterator = iter(loader)
            first = next(iterator)[0]
            assert first.is_pinned()
            assert torch.equal(first, source[:3])
            for _ in range(2):
                batches = [batch[0] for batch in loader]
                assert all(batch.is_pinned() for batch in batches)
                assert torch.equal(torch.cat(batches), source)
            del iterator, loader
            gc.collect()
            # A delivered batch survives its loader and pinning thread.
            assert first.is_pinned()
            assert torch.equal(first, source[:3])


def test_pinned_dataloader_foreign_cuda_batch_keeps_allocator(
    pin_allocator_probe: _PinAllocatorProbe,
):
    if not torch.cuda.is_available():
        pytest.skip(
            "foreign CUDA-owned pinned batch requires runtime CUDA availability"
        )
    source = torch.empty(17, dtype=torch.int64, device="cpu", pin_memory=True)
    source.copy_(torch.arange(17))
    assert not pin_allocator_probe.uses_mojo_allocator(source)
    batch = next(
        iter(DataLoader(_SinglePinSample(source), batch_size=None, pin_memory=True))
    )
    assert batch is source
    assert batch.is_pinned()
    assert not pin_allocator_probe.uses_mojo_allocator(batch)
    assert torch.equal(batch, torch.arange(17))


def test_pinned_dataloader_thread_keeps_captured_device(mojo_gpu: str):
    if get_accelerators()[0].api != "cuda":
        pytest.skip("pinning-thread allocation-device probe requires the CUDA driver")
    if device_module.device_count() < 3:
        pytest.skip(
            "requires two physical Mojo GPUs for the pinning thread's captured device"
        )
    with device_module.device(1):
        loader = DataLoader(
            TensorDataset(torch.arange(68).reshape(17, 4)),
            batch_size=3,
            num_workers=2,
            pin_memory=True,
            timeout=60,
        )
        iterator = iter(loader)
        try:
            with device_module.device(0):
                batches = [batch[0] for batch in iterator]
                assert all(
                    _cuda_pinned_allocation_device(batch) == 1 for batch in batches
                )
                assert device_module.current_device() == 0
        finally:
            del iterator, loader
            gc.collect()
        assert torch.equal(torch.cat(batches), torch.arange(68).reshape(17, 4))


def test_module_to_mojo_preserves_tied_parameters(mojo_device):
    """See the fuller version of this test in test_model_state_conversion.py
    (module conversion owns the tied-weight contract there); kept here too
    since it also exercises plain `.to(device)` on an `nn.Module`."""

    class TiedWeights(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.projection = torch.nn.Linear(8, 16, bias=False)
            self.projection.weight = self.embedding.weight

    module = TiedWeights()
    module.to(mojo_device)

    assert module.embedding.weight is module.projection.weight
    assert module.embedding.weight.device.type == "mojo"
    assert len(list(module.parameters())) == 1


def test_mojo_parameters_enable_foreach_optimizer_selection(mojo_device):
    parameter = torch.nn.Parameter(torch.ones(8)).to(mojo_device)
    fused, foreach = _default_to_fused_or_foreach([parameter], differentiable=False)
    assert not fused
    assert foreach


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::mul.out (AdamW's foreach/single-tensor "
    "step needs the out= arithmetic ops)",
)
@pytest.mark.parametrize("foreach", [None, True, False])
def test_mojo_adamw_step_matches_cpu(mojo_gpu_available, foreach):
    """The optimizer path used by nanoGPT must update parameters and moments."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    mojo_gpu = "mojo:0"
    initial = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)
    cpu_parameter = torch.nn.Parameter(initial.clone())
    mojo_parameter = torch.nn.Parameter(initial.to(mojo_gpu))
    cpu_optimizer = torch.optim.AdamW(
        [cpu_parameter],
        lr=0.025,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=foreach,
    )
    mojo_optimizer = torch.optim.AdamW(
        [mojo_parameter],
        lr=0.025,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=foreach,
    )

    for grad in (
        torch.tensor([0.25, -0.5, 0.75, -1.0]),
        torch.tensor([-0.125, 0.25, -0.375, 0.5]),
    ):
        cpu_parameter.grad = grad.clone()
        mojo_parameter.grad = grad.to(mojo_gpu)
        cpu_optimizer.step()
        mojo_optimizer.step()

    torch.accelerator.synchronize(mojo_gpu)
    torch.testing.assert_close(mojo_parameter.cpu(), cpu_parameter)
    cpu_state = cpu_optimizer.state[cpu_parameter]
    mojo_state = mojo_optimizer.state[mojo_parameter]
    for name in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(mojo_state[name].cpu(), cpu_state[name])
    assert mojo_state["step"].item() == cpu_state["step"].item() == 2


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::set_.source_Storage (torch.load's "
    "tensor rebuild path needs it before map_location moves the value to cpu)",
)
def test_mojo_checkpoint_resumes_through_portable_cpu_state(mojo_gpu_available):
    """The nanoGPT resume path loads CPU state, then moves it normally."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    mojo_gpu = "mojo:0"
    model = torch.nn.Linear(3, 2).to(mojo_gpu)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=None)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    checkpoint_bytes = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        checkpoint_bytes,
    )
    checkpoint_bytes.seek(0)
    checkpoint = torch.load(checkpoint_bytes, map_location="cpu")
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["model"].values())

    resumed_model = torch.nn.Linear(3, 2)
    resumed_model.load_state_dict(checkpoint["model"])
    resumed_model.to(mojo_gpu)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(), lr=0.01, foreach=None
    )
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    for state in resumed_optimizer.state.values():
        assert state["step"].device.type == "cpu"
        assert state["exp_avg"].device == torch.device(mojo_gpu)
        assert state["exp_avg_sq"].device == torch.device(mojo_gpu)

    for parameter in resumed_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    resumed_optimizer.step()
    torch.accelerator.synchronize(mojo_gpu)
    assert all(
        torch.isfinite(parameter.cpu()).all()
        for parameter in resumed_model.parameters()
    )


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::linalg_vector_norm.out (foreach L2-norm "
    "clip needs the reduction)",
)
@pytest.mark.parametrize("foreach", [None, True, False])
def test_mojo_clip_grad_norm_matches_cpu(mojo_gpu_available, foreach):
    """nanoGPT's FP32 gradient clipping uses the foreach L2-norm path."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    mojo_gpu = "mojo:0"
    cpu_parameters = [
        torch.nn.Parameter(torch.zeros(3)),
        torch.nn.Parameter(torch.zeros(2, 2)),
    ]
    mojo_parameters = [
        torch.nn.Parameter(parameter.detach().to(mojo_gpu))
        for parameter in cpu_parameters
    ]
    gradients = (
        torch.tensor([3.0, 4.0, -2.0]),
        torch.tensor([[1.0, -2.0], [2.0, -1.0]]),
    )
    for cpu_parameter, mojo_parameter, gradient in zip(
        cpu_parameters, mojo_parameters, gradients, strict=True
    ):
        cpu_parameter.grad = gradient.clone()
        mojo_parameter.grad = gradient.to(mojo_gpu)

    expected_norm = torch.nn.utils.clip_grad_norm_(
        cpu_parameters, 1.25, foreach=foreach
    )
    actual_norm = torch.nn.utils.clip_grad_norm_(mojo_parameters, 1.25, foreach=foreach)
    torch.accelerator.synchronize(mojo_gpu)

    torch.testing.assert_close(actual_norm.cpu(), expected_norm)
    for actual, expected in zip(mojo_parameters, cpu_parameters, strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad.cpu(), expected.grad)


def test_device_ordering_gpu_first_cpu_last():
    """The mojo device convention: index 0 is the first GPU (when present),
    the highest index is the MAX CPU device. `get_accelerators()` (MAX's own
    accelerator list, unrelated to the old TorchMojoTensor/torch_mojo_tensor
    machinery) carries the `.label` used to check this without any private
    import."""
    accelerators = list(get_accelerators())
    assert len(accelerators) > 0
    assert accelerators[-1].label == "cpu"
    assert device_module.cpu() == torch.device(f"mojo:{len(accelerators) - 1}")

    gpu_labels = [a.label for a in accelerators if a.label == "gpu"]
    if gpu_labels:
        assert accelerators[0].label == "gpu"
        t_gpu = torch.tensor([1.0]).to("mojo")
        assert t_gpu.device.type == "mojo" and t_gpu.device.index == 0
    cpu_index = len(accelerators) - 1
    t_cpu = torch.tensor([1.0]).to(f"mojo:{cpu_index}")
    assert t_cpu.device.type == "mojo"


# Original tests from the existing file
def function_equivalent_on_both_devices(
    func, device, *args, rtol=1e-4, atol=1e-4, **kwargs
):
    # This helper checks forward values only. Keeping the first forward's
    # autograd graph alive while the same closure-owned module moves back to
    # CPU adds a legitimate TensorImpl reference, which PyTorch's required
    # swap-on-conversion path rejects. Avoid manufacturing that unrelated
    # lifetime condition in forward-equivalence tests.
    with torch.no_grad():
        out1 = func(*args, device=device, **kwargs)
        out2 = func(*args, device="cpu", **kwargs)
    if isinstance(out1, list | tuple):
        assert type(out1) is type(out2)
    else:
        assert isinstance(out1, torch.Tensor)
        assert isinstance(out2, torch.Tensor)
        out1 = [out1]
        out2 = [out2]

    out1 = [o.to("cpu") for o in out1]

    for i, (o1, o2) in enumerate(zip(out1, out2)):
        assert o1.device == o2.device, f"Issue with output {i}"
        assert o1.shape == o2.shape, f"Issue with output {i}"
        assert o1.dtype == o2.dtype, f"Issue with output {i}"
        assert torch.allclose(o1, o2, rtol=rtol, atol=atol), f"Issue with output {i}"


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::sqrt.out")
def test_mojo_device_basic(mojo_device):
    def do_sqrt(device):
        a = torch.arange(4, device=device, dtype=torch.float32)
        return torch.sqrt(a)

    function_equivalent_on_both_devices(do_sqrt, mojo_device)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::sqrt.out")
def test_mojo_device_basic_arange_sqrt(mojo_device):
    a = torch.arange(4, device=mojo_device, dtype=torch.float32)
    sqrt_result = torch.sqrt(a)
    result_cpu = sqrt_result.to("cpu")
    assert torch.allclose(
        result_cpu, torch.tensor([0.0, 1.0, 1.4142, 1.7320]), atol=1e-4
    )
    b = torch.arange(4, device=mojo_device, dtype=torch.float32)
    chained = sqrt_result + b
    chained_cpu = chained.to("cpu")
    assert torch.allclose(
        chained_cpu, torch.tensor([0.0, 2.0, 3.4142, 4.7320]), atol=1e-4
    )


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::arange.start_out")
def test_device_creation(mojo_device):
    torch_device = torch.device(mojo_device)
    arr = torch.arange(4, device=torch_device, dtype=torch.float32)
    arr_cpu = arr.to("cpu")
    assert torch.allclose(arr_cpu, torch.tensor([0.0, 1.0, 2.0, 3.0]), atol=1e-4)


def test_device_basic_full(mojo_device):
    def do_full(device):
        a = torch.full((2, 3), 7.0, device=device, dtype=torch.float32)
        return a

    function_equivalent_on_both_devices(do_full, mojo_device)


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::normal_ (randn inputs) and conv2d itself",
)
def test_convolution_2d(mojo_device):
    input_tensor_cpu = torch.randn(1, 3, 32, 32, device="cpu")
    weight_cpu = torch.randn(6, 3, 5, 5, device="cpu")
    bias_cpu = torch.randn(6, device="cpu")

    def do_convolution(device):
        input_tensor = input_tensor_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        return torch.nn.functional.conv2d(
            input_tensor, weight, bias=bias, stride=1, padding=2
        )

    function_equivalent_on_both_devices(do_convolution, mojo_device)


def test_simple_module(mojo_device):
    """No forward pass here (see test_custom_module for that): just that
    `.to(device)` on a module carries its weight over correctly."""
    linear = torch.nn.Linear(4, 8)

    def run_module(device):
        my_linear = linear.to(device)
        return my_linear.weight

    function_equivalent_on_both_devices(run_module, mojo_device)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::addmm.out")
def test_custom_module(mojo_device):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 8)

        def forward(self, x):
            return self.linear(x)

    module = MyModule()
    input_tensor = torch.randn(2, 4)

    def run_module(device):
        in_device_module = module.to(device)
        in_device_input_tensor = input_tensor.to(device)
        return in_device_module(in_device_input_tensor)

    function_equivalent_on_both_devices(run_module, mojo_device, rtol=1e-3, atol=1e-3)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::addmm.out")
def test_custom_module_with_seqential(mojo_device):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 8)

        def forward(self, x):
            return self.linear(x)

    module = torch.nn.Sequential(MyModule())
    input_tensor = torch.randn(2, 4)

    def run_module(device):
        in_device_module = module.to(device)
        in_device_input_tensor = input_tensor.to(device)
        return in_device_module(in_device_input_tensor)

    function_equivalent_on_both_devices(run_module, mojo_device, rtol=1e-3, atol=1e-3)


def test_compile_with_max_device(mojo_device):
    """The torch.compile(backend=mojo_backend) graph path is independent of
    the native eager device: it lowers straight to a MAX graph rather than
    dispatching aten ops one at a time, so it is unaffected by which native
    ops have landed yet. Its outputs come back as mojo tensors through
    `mojo_device/dlpack.py`'s kDLExtDev capsule, not through any wrapper."""

    @torch.compile(backend=mojo_backend)
    def do_sqrt(device):
        a = torch.arange(4, device=device, dtype=torch.float32)
        return torch.sqrt(a)

    function_equivalent_on_both_devices(do_sqrt, mojo_device)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a second backend to copy from"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to_mojo_from_another_backend(mojo_gpu, dtype):
    """A tensor on another accelerator must transfer, not fault.

    Regression test for a real segfault: `_to_copy`/`_copy_from` used to
    branch on "is it our own wrapper tensor", not "is it on the host", so
    with a GPU torch installed they also received a CUDA source and
    dereferenced its pointer as HOST memory. Found by the OpInfo `to`
    samples, which include a cross-device target tensor.
    """
    src = torch.randn(3, 5, device="cuda", dtype=dtype)
    moved = src.to(mojo_gpu)
    assert moved.device.type == "mojo"
    assert moved.dtype == dtype
    assert torch.equal(moved.cpu(), src.cpu())

    cast = src.to(mojo_gpu, dtype=torch.float32)
    assert cast.dtype == torch.float32
    assert torch.allclose(cast.cpu(), src.float().cpu())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a second backend to copy from"
)
def test_copy_into_mojo_from_another_backend(mojo_gpu):
    """The same trap on the `copy_` path, including the broadcasting arm."""
    dest = torch.zeros(3, 5, device=mojo_gpu)
    src = torch.randn(3, 5, device="cuda")
    dest.copy_(src)
    assert torch.equal(dest.cpu(), src.cpu())

    wide = torch.zeros(4, 3, device=mojo_gpu)
    row = torch.arange(3, device="cuda", dtype=torch.float32)
    wide.copy_(row)
    assert torch.equal(wide.cpu(), row.cpu().expand(4, 3).contiguous())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a second backend to copy from"
)
def test_same_gpu_transfer_skips_the_host(mojo_gpu):
    """The transfer must not go through host memory when both live on one GPU.

    Asserted by timing rather than by values, because a host bounce is
    CORRECT -- just two PCIe crossings instead of one HBM copy.
    """
    big = torch.randn(1 << 26, device="cuda")  # 256 MB
    big.to(mojo_gpu)
    torch.accelerator.synchronize(mojo_gpu)

    start = time.perf_counter()
    moved = big.to(mojo_gpu)
    torch.accelerator.synchronize(mojo_gpu)
    on_device = time.perf_counter() - start

    start = time.perf_counter()
    bounced = big.cpu().to(mojo_gpu)
    torch.accelerator.synchronize(mojo_gpu)
    through_host = time.perf_counter() - start

    assert torch.equal(moved.cpu(), bounced.cpu())
    assert on_device * 10 < through_host, (
        f"expected the on-device route; {on_device * 1e3:.2f}ms on device vs "
        f"{through_host * 1e3:.2f}ms through the host"
    )


@pytest.mark.xfail(
    strict=False, reason="op not ported yet: aten::linalg_vector_norm.out"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vector_norm_with_an_accumulation_dtype(mojo_gpu_available, dtype):
    """FSDP1's clip_grad_norm_ asks for the norm in float32 explicitly."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    cpu = torch.randn(4096, dtype=dtype)
    expected = torch.linalg.vector_norm(cpu, 2.0, dtype=torch.float32)
    got = torch.linalg.vector_norm(cpu.to("mojo:0"), 2.0, dtype=torch.float32)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-5, atol=1e-4)


def test_data_assignment_moves_the_payload(mojo_gpu_available):
    """``x.data = y`` has to move the allocation, not just TensorImpl metadata.

    FSDP1 swaps a flat parameter between its sharded and unsharded buffers
    with exactly this assignment.
    """
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    destination = torch.zeros(8, device="mojo:0")
    source = torch.full((8,), 100.0, device="mojo:0")
    destination.data = source
    assert destination.cpu().tolist() == source.cpu().tolist()

    # A strided view at a non-zero offset must survive intact: FSDP hands
    # every original parameter a slice of the flat parameter this way.
    base = torch.arange(16, dtype=torch.float32).to("mojo:0")
    view = base[4:12].reshape(2, 4)
    holder = torch.zeros(1, device="mojo:0")
    holder.data = view
    assert tuple(holder.shape) == (2, 4)
    assert holder.stride() == view.stride()
    assert holder.cpu().tolist() == [[4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]


def test_as_strided_zero_copy_view(mojo_gpu_available):
    """A strided view at an offset, and the allocation-bounds refusal.

    as_strided is the one view op whose arguments are unconstrained by the
    input's own shape, so the backend must bound-check against the real
    allocation rather than trust the caller (DDP's Reducer builds gradient
    bucket views with it).
    """
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    cpu = torch.arange(24, dtype=torch.float32)
    dev = cpu.to("mojo:0")
    view = torch.as_strided(dev, (3, 4), (8, 2), 1)
    assert torch.equal(view.cpu(), torch.as_strided(cpu, (3, 4), (8, 2), 1))
    # Zero-copy: writes through the base are visible in the view.
    dev.fill_(1.0)
    torch.testing.assert_close(
        view.cpu(), torch.as_strided(torch.ones(24), (3, 4), (8, 2), 1)
    )
    # A layout that would reach past the allocation must be refused: a real
    # error (TORCH_CHECK in the C++ shim), not silently accepted.
    with pytest.raises(RuntimeError, match="sizes/strides reach"):
        torch.as_strided(dev, (5, 4), (8, 2), 1)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::set_.source_Tensor")
def test_set_source_tensor_adopts_the_allocation(mojo_gpu_available):
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    destination = torch.zeros(8, device="mojo:0")
    source = torch.full((4,), 50.0, device="mojo:0")
    returned = destination.set_(source)  # ty: ignore[invalid-argument-type] -- torch's stub lacks the Tensor overload
    assert returned is destination
    assert tuple(destination.shape) == (4,)
    assert destination.cpu().tolist() == [50.0, 50.0, 50.0, 50.0]
    # Sharing the allocation, not a copy of it.
    source.fill_(1.0)
    assert destination.cpu().tolist() == [1.0, 1.0, 1.0, 1.0]
