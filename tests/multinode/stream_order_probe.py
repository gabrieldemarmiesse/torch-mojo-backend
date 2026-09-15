"""Exercise mojoccl's C ABI on caller-owned torch CUDA streams.

Use a CUDA-enabled torch environment, MOJOCCL_LIBRARY from the main checkout's
build, and MOJOCCL_STATE_PROBE_LIBRARY built from selftest/comm_state_probe.mojo.
Gloo only distributes the NCCL unique ID; collectives call the C ABI directly
because the mojo process group uses its own comm stream.
"""

import ctypes
import datetime
import os
import threading
import time

import torch
import torch.distributed as dist


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def check(rc: int):
    if rc:
        raise RuntimeError(f"C ABI returned error {rc}")


def main():
    timer = threading.Timer(240, lambda: os._exit(3))
    timer.daemon = True
    timer.start()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=180))
    rank, world = dist.get_rank(), dist.get_world_size()
    lib = ctypes.CDLL(os.environ["MOJOCCL_LIBRARY"])
    state_probe = ctypes.CDLL(os.environ["MOJOCCL_STATE_PROBE_LIBRARY"])
    driver = ctypes.CDLL("libcuda.so.1")
    ptr = ctypes.c_void_p
    lib.ncclGetUniqueId.argtypes = [ctypes.POINTER(UniqueId)]
    lib.ncclCommInitRank.argtypes = [
        ctypes.POINTER(ptr),
        ctypes.c_int,
        UniqueId,
        ctypes.c_int,
    ]
    lib.ncclAllReduce.argtypes = [
        ptr,
        ptr,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ptr,
        ptr,
    ]
    lib.ncclBroadcast.argtypes = [
        ptr,
        ptr,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ptr,
        ptr,
    ]
    lib.ncclAllGather.argtypes = [ptr, ptr, ctypes.c_size_t, ctypes.c_int, ptr, ptr]
    lib.ncclCommDestroy.argtypes = [ptr]
    lib.ncclCommAbort.argtypes = [ptr]
    lib.ncclCommGetAsyncError.argtypes = [ptr, ctypes.POINTER(ctypes.c_int)]
    state_probe.mojoccl_test_resources_released.argtypes = [ptr]
    driver.cuStreamCreate.argtypes = [ctypes.POINTER(ptr), ctypes.c_uint]
    driver.cuStreamDestroy_v2.argtypes = [ptr]
    uid = UniqueId()
    if rank == 0:
        check(lib.ncclGetUniqueId(ctypes.byref(uid)))
    blob = [bytes(uid)]
    dist.broadcast_object_list(blob, src=0)
    uid = UniqueId.from_buffer_copy(blob[0])
    comm = ptr()
    check(lib.ncclCommInitRank(ctypes.byref(comm), world, uid, rank))
    default = torch.cuda.default_stream()
    assert default.cuda_stream == 0
    n = 357 * 792  # Awkward, but 16-byte aligned for every rank's shard.
    inputs = [
        torch.full((n,), rank + j + 1, device="cuda", dtype=torch.float32)
        for j in range(32)
    ]
    outputs = [
        torch.empty(n * (world if j % 3 == 2 else 1), device="cuda") for j in range(32)
    ]
    torch.cuda.synchronize()
    submitting = threading.Event()
    stop_poll = threading.Event()
    poll_ready = threading.Event()
    poll_errors = []
    overlapping_polls = []

    def poll():
        try:
            torch.cuda.set_device(local_rank)
            while not stop_poll.is_set():
                active = submitting.is_set()
                error = ctypes.c_int(-1)
                check(lib.ncclCommGetAsyncError(comm, ctypes.byref(error)))
                check(error.value)
                if active and submitting.is_set():
                    overlapping_polls.append(1)
                poll_ready.set()
        except Exception as exc:
            poll_errors.append(str(exc))
            poll_ready.set()

    def issue(j: int, stream: torch.cuda.Stream):
        src, dst = inputs[j].data_ptr(), outputs[j].data_ptr()
        submitting.set()
        try:
            if j % 3 == 0:
                check(lib.ncclAllReduce(src, dst, n, 7, 0, comm, stream.cuda_stream))
            elif j % 3 == 1:
                check(lib.ncclBroadcast(src, dst, n, 7, 0, comm, stream.cuda_stream))
            else:
                check(lib.ncclAllGather(src, dst, n, 7, comm, stream.cuda_stream))
        finally:
            submitting.clear()

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    assert poll_ready.wait(30), "poll thread did not start"
    for j in range(256):
        issue(j % 32, default)
    stop_poll.set()
    poller.join(timeout=30)
    assert not poller.is_alive(), "poll thread did not stop"
    assert not poll_errors, poll_errors
    assert overlapping_polls, "no polls overlapped collective submission"
    print(
        f"rank {rank}: concurrent_poll PASS ({len(overlapping_polls)} overlapping polls)",
        flush=True,
    )

    for j in range(0, 32, 4):
        raw = ptr()
        check(driver.cuStreamCreate(ctypes.byref(raw), 1))  # NON_BLOCKING
        side = torch.cuda.ExternalStream(raw.value, device=local_rank)
        with torch.cuda.stream(default):
            torch.cuda._sleep(20_000_000)
        issue(j, default)
        issue(j + 1, side)
        issue(j + 2, default)
        issue(j + 3, side)
        side.synchronize()
        check(driver.cuStreamDestroy_v2(raw))
        del side
        # No new stream is created before polling; the destroyed handle cannot
        # be recycled. The next iteration starts on default stream 0 again.
        error = ctypes.c_int()
        check(lib.ncclCommGetAsyncError(comm, ctypes.byref(error)))
        check(error.value)
    issue(0, default)  # The last side stream has been destroyed.
    torch.cuda.synchronize()
    for j, result in enumerate(outputs):
        if j % 3 == 0:
            expected = torch.full((n,), world * (world - 1) // 2 + world * (j + 1))
        elif j % 3 == 1:
            expected = torch.full((n,), j + 1)
        else:
            expected = torch.arange(j + 1, j + world + 1).repeat_interleave(n)
        torch.testing.assert_close(result.cpu(), expected.float(), rtol=0, atol=0)
    check(lib.ncclCommDestroy(comm))
    print(
        f"rank {rank}: stream_order PASS (default/side/default, destroyed sides, all three collectives)",
        flush=True,
    )
    uid = UniqueId()
    if rank == 0:
        check(lib.ncclGetUniqueId(ctypes.byref(uid)))
    blob = [bytes(uid)]
    dist.broadcast_object_list(blob, src=0)
    uid = UniqueId.from_buffer_copy(blob[0])
    check(lib.ncclCommInitRank(ctypes.byref(comm), world, uid, rank))
    raw = ptr()
    check(driver.cuStreamCreate(ctypes.byref(raw), 1))
    side = torch.cuda.ExternalStream(raw.value, device=local_rank)
    issue(0, side)
    side.synchronize()
    check(driver.cuStreamDestroy_v2(raw))
    del side
    dist.barrier()
    assert state_probe.mojoccl_test_resources_released(comm) == 0
    start = time.monotonic()
    check(lib.ncclCommAbort(comm))
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"abort took {elapsed:.3f}s after side-stream destruction"
    assert state_probe.mojoccl_test_resources_released(comm) == 1, (
        "abort leaked resources"
    )
    check(lib.ncclCommDestroy(comm))
    print(
        f"rank {rank}: destroyed_stream_abort PASS ({elapsed:.3f}s, released)",
        flush=True,
    )
    dist.destroy_process_group()
    timer.cancel()


if __name__ == "__main__":
    main()
