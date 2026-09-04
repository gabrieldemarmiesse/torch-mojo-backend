"""A c10d ProcessGroup for the mojo eager device, backed by NCCL or RCCL.

Design (see docs/distributed.md for the full story):

- Pure-Python ``torch.distributed.ProcessGroup`` subclass. In torch 2.11 a
  Python PG *replaces* the whole process group (distributed_c10d.py:2193-2198),
  so no gloo backend can be composed in by ``init_process_group`` — object
  collectives and ``barrier()`` therefore hand us **CPU** tensors
  (``_get_object_coll_device`` falls back to "cpu" when ``_device_types`` is
  empty). Every override dispatches on the tensor's device type and delegates
  CPU tensors to a private ``ProcessGroupGloo``.

- Collectives go through the NCCL C API — NCCL on NVIDIA, RCCL on AMD, one
  ctypes binding for both (``distributed/nccl.py``). "NCCL" below means
  whichever of the two the process loaded.

- NCCL collectives run on a dedicated comm stream (a side stream,
  ``mojo_device/device_streams.py``) for compute/communication overlap: it
  waits for the default stream first (so producers are ordered before the
  collective) and the collective is enqueued. Every tensor a collective
  touches is fenced via ``device_streams.record_use`` (see that module's
  docstring) and recorded as pending in ``mojo_device/comm_fence.py``.
  ``TORCH_MOJO_BACKEND_COMM_STREAM=0`` falls back to the default stream
  itself (ordering free, zero overlap) — also the automatic path for
  collectives needing default-stream copies AFTER the collective
  (non-contiguous outputs, list-form allgather/reduce_scatter,
  gather/scatter/alltoall staging). Both paths drain the kernel-call queue
  first (rule 1 in device_streams.py).

- Work objects wrap an already-completed ``torch.futures.Future`` holding the
  output tensors, on both paths, so ``wait()`` is a host-side no-op. Never
  pass ``devices=`` to that Future: with a device list it routes through the
  stub PythonDeviceGuard whose ``deviceCount() == 1`` and performs an
  out-of-bounds write for device indices >= 1.

  Completing at enqueue time rather than when the collective's end event
  fires is what keeps the host free, and it is the whole point of
  ``comm_fence``. Stock ``ProcessGroupNCCL`` does the same and pays for it
  with a device-typed future whose ``wait()`` makes the *current stream*
  wait — a C++ DeviceGuardImpl this backend cannot ship. Instead the device
  ordering is inserted lazily: the first default-stream op touching a
  collective's buffer makes the default stream wait on the comm stream.
  For DDP that lands in ``finalize_backward``, where the Reducer first reads
  a reduced bucket — after every backward kernel is already enqueued — so
  overlap is unchanged while the host runs ahead into the bucket→grad
  copies, ``clip_grad_norm_`` and the optimizer step. Blocking the host on
  the future instead cost ~2 ms/step of exposed GPU idle: nanoGPT 124M on
  32 H100s went 10.89 -> 11.10 M tok/s when it stopped doing so.

- The DDP Reducer calls ``allreduce`` through the C++ trampoline and then
  ``Work.get_future()`` (default_comm_hooks.cpp) — both supported by
  ``_create_work_from_future``. Gradient scaling (/world_size) happens inside
  the Reducer before the collective, so ``allreduce`` here is a plain SUM.

One process drives exactly one GPU (torchrun layout). Pin the visible GPU
per rank *before* MAX enumerates devices (``CUDA_VISIBLE_DEVICES`` on NVIDIA,
``ROCR_VISIBLE_DEVICES``/``HIP_VISIBLE_DEVICES`` on AMD) — see
``torch_mojo_backend.distributed.use_local_rank_gpu()``.
"""

import datetime
import os
import sys
import threading
import traceback
from collections.abc import Callable
from typing import cast

import torch
import torch.distributed as dist
from max.driver import Device
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    GatherOptions,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
    _create_work_from_future,  # ty: ignore[unresolved-import] -- exists at runtime, absent from the stub
)
from torch.distributed import PrefixStore, Store, Work

from torch_mojo_backend.distributed import nccl
from torch_mojo_backend.mojo_device import (
    comm_fence,
    deferred_compile,
    torch_mojo_device_module,
)
from torch_mojo_backend.mojo_device.device_streams import get_stream, record_use
from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
    TorchMojoTensor,
    find_equivalent_max_device,
)

_NCCL_DTYPE_OF: dict[torch.dtype, int] = {
    torch.int8: nccl.NCCL_INT8,
    torch.uint8: nccl.NCCL_UINT8,
    torch.bool: nccl.NCCL_UINT8,
    torch.int32: nccl.NCCL_INT32,
    torch.uint32: nccl.NCCL_UINT32,
    torch.int64: nccl.NCCL_INT64,
    torch.uint64: nccl.NCCL_UINT64,
    torch.float16: nccl.NCCL_FLOAT16,
    torch.float32: nccl.NCCL_FLOAT32,
    torch.float64: nccl.NCCL_FLOAT64,
    torch.bfloat16: nccl.NCCL_BFLOAT16,
}

# torch.bool reduces as uint8: SUM/MAX behave as logical OR, MIN as AND —
# the same convention ProcessGroupNCCL uses.


def _nccl_dtype(dtype: torch.dtype) -> int:
    try:
        return _NCCL_DTYPE_OF[dtype]
    except KeyError:
        raise TypeError(f"dtype {dtype} is not supported by the mojo NCCL/RCCL backend")


def _nccl_red_op(op: ReduceOp | ReduceOp.RedOpType) -> int:
    if op == ReduceOp.SUM:
        return nccl.NCCL_SUM
    if op == ReduceOp.PRODUCT:
        return nccl.NCCL_PROD
    if op == ReduceOp.MAX:
        return nccl.NCCL_MAX
    if op == ReduceOp.MIN:
        return nccl.NCCL_MIN
    if op == ReduceOp.AVG:
        return nccl.NCCL_AVG
    raise NotImplementedError(
        f"ReduceOp {op} is not supported by the mojo NCCL/RCCL backend"
    )


def _ptr_of(tensor: torch.Tensor) -> int:
    """Device pointer of a tensor `_is_cpu` already classified as mojo."""
    assert isinstance(tensor, TorchMojoTensor), tensor.device
    return tensor._ptr


def _completed_work(result: list[torch.Tensor]) -> Work:
    future = torch.futures.Future[
        list[torch.Tensor]
    ]()  # no devices= — see module docstring
    future.set_result(result)
    return _create_work_from_future(future)


def _loud(fn: Callable[..., Work]) -> Callable[..., Work]:
    """Print the traceback before propagating.

    An exception that escapes into the autograd engine through a C++ backward
    hook on this backend can terminate the process without any Python
    traceback (see mojo_device/aten_ops/autograd_preflight.py). Printing here
    guarantees the root cause is visible even in that worst case.
    """

    name = getattr(fn, "__name__", repr(fn))

    def wrapper(self: "MojoProcessGroup", *args: object, **kwargs: object) -> Work:
        try:
            return fn(self, *args, **kwargs)
        except Exception:
            print(
                f"[torch-mojo-backend] rank {self.rank()}: error in "
                f"MojoProcessGroup.{name}:",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            sys.stderr.flush()
            raise

    wrapper.__name__ = name
    return wrapper


class MojoProcessGroup(dist.ProcessGroup):
    """NCCL/RCCL-backed process group for ``mojo`` tensors, gloo for CPU tensors."""

    def __init__(
        self, store: Store, rank: int, world_size: int, timeout: datetime.timedelta
    ):
        # The 2-arg base init: the 3-arg (store, rank, size) overload is a
        # pybind FACTORY init, and factory inits cannot construct the
        # PyProcessGroup trampoline alias a Python subclass needs — it fails
        # with "returned holder-wrapped instance is not an alias instance".
        # Every in-tree Python PG (test_c10d_pypg, multi_threaded_pg) does
        # this too; the C++-side store stays null and we keep it here instead.
        super().__init__(rank, world_size)  # ty: ignore[missing-argument, invalid-argument-type]
        self._store = store
        self._timeout = timeout
        self._group_name = ""
        # Communicators keyed by mojo device index, created lazily at the
        # first device collective (a collective, blocking rendezvous — every
        # rank reaches it in the same order because collectives are SPMD).
        # The library (NCCL or RCCL) is chosen then too, from the device api
        # of the first mojo tensor seen: one process drives one GPU vendor.
        self._ccl: nccl.CclLibrary | None = None
        self._comms: dict[int, nccl.NcclComm] = {}
        self._streams: dict[int, int] = {}
        self._max_devices: dict[int, Device] = {}
        self._comm_seq = 0
        self._comm_lock = threading.Lock()
        self._device_current = threading.local()
        # Comm stream (a side stream) for compute/communication overlap;
        # TORCH_MOJO_BACKEND_COMM_STREAM=0 pins collectives to the default
        # stream instead (no overlap, simplest possible ordering).
        self._comm_stream_enabled = (
            os.environ.get("TORCH_MOJO_BACKEND_COMM_STREAM", "1") != "0"
        )
        # Private CPU backend: torch will not compose gloo around a Python PG
        # (see module docstring), so CPU tensors are our job too.
        # The stub declares the collectives on ProcessGroup only; the gloo
        # backend has the same methods at runtime.
        self._gloo = cast(
            dist.ProcessGroup,
            dist.ProcessGroupGloo(
                PrefixStore("mojo-cpu-gloo", store), rank, world_size, timeout
            ),
        )

    # -- plumbing ------------------------------------------------------------

    def getBackendName(self) -> str:
        return "mojo"

    def getGroupName(self) -> str:
        # The C++ accessor raises while no C++ backend is registered, so keep
        # the name Python-side (same workaround as test_c10d_pypg.py).
        return self._group_name

    def setGroupName(self, name: str):
        self._group_name = name

    def _set_group_name(self, name: str):
        self._group_name = name

    def shutdown(self):
        # Nothing waits for a comm-stream collective any more (see
        # _stream_work), so drain them here before the communicators they
        # run on are destroyed.
        self._quiesce_comm_streams()
        for comm in self._comms.values():
            comm.destroy()
        self._comms.clear()

    def abort(self):
        # An aborted communicator never finishes its work: drop the pending
        # fences rather than let a later op wait on the comm stream forever.
        for index in self._max_devices:
            comm_fence.discard(index)
        for comm in self._comms.values():
            comm.abort()
        self._comms.clear()

    def _quiesce_comm_streams(self):
        """Complete every in-flight comm-stream collective, fences included."""
        for index, max_device in self._max_devices.items():
            if self._comm_stream_enabled:
                get_stream(max_device, "nccl").synchronize()
            comm_fence.discard(index)

    def _ensure_device_current(self, api: str, ordinal: int):
        # Both libraries resolve the target GPU from per-thread runtime state
        # (the current CUDA context, the current HIP device), and DDP calls us
        # from the autograd thread while init usually runs on the main thread
        # — so re-assert per thread, once.
        if getattr(self._device_current, "ordinal", None) != ordinal:
            nccl.set_current_device(api, ordinal)
            self._device_current.ordinal = ordinal

    def _device_state(self, tensor: torch.Tensor) -> tuple[nccl.NcclComm, int, int]:
        """(communicator, default-stream native handle, device index) for a tensor."""
        index = tensor.device.index
        if index is None:
            index = torch_mojo_device_module.current_device()
        assert isinstance(tensor, TorchMojoTensor)  # _is_cpu classified it
        api = tensor._device.api
        # Which physical GPU owns the allocation is read off the POINTER
        # (cuda_peer / hip_peer), never assumed from an ordinal.
        ordinal = nccl.device_ordinal(api, _ptr_of(tensor))
        if ordinal is None:
            raise RuntimeError(
                "could not resolve which GPU owns a mojo tensor (device api "
                f"{api!r}); the mojo distributed backend supports NVIDIA (NCCL) "
                "and AMD (RCCL) GPUs"
            )
        self._ensure_device_current(api, ordinal)
        with self._comm_lock:
            comm = self._comms.get(index)
            if comm is None:
                if self._ccl is None:
                    self._ccl = nccl.load(api)
                comm = self._init_comm(index)
                self._comms[index] = comm
                max_device = find_equivalent_max_device(torch.device("mojo", index))
                self._max_devices[index] = max_device
                self._streams[index] = max_device.default_stream.native_stream_handle
        return comm, self._streams[index], index

    def _stream_work(
        self,
        index: int,
        result: list[torch.Tensor],
        fenced: tuple[torch.Tensor, ...],
        enqueue: Callable[[int], None],
    ) -> Work | None:
        """Run ``enqueue(stream_handle)`` on the comm stream, overlapped.

        Returns None when the comm-stream path is unavailable — the caller
        then runs the same ``enqueue`` on the default stream. Eligibility is
        the caller's job: only collectives needing NO default-stream work
        after the NCCL call (no copy-back into non-contiguous outputs) may
        come here, because the default stream is ordered after the comm
        stream lazily, at the first consumer, not here.

        ``fenced`` lists every tensor the collective reads or writes. Each is
        recorded against the comm stream (``device_streams.record_use``) so
        its free is ordered after this collective, and marked pending in
        ``comm_fence`` so its first default-stream use is too.
        """
        if not self._comm_stream_enabled:
            return None
        comm_stream = get_stream(self._max_devices[index], "nccl")
        self._drained()  # producers must be ON the stream before we fence it
        comm_stream.wait_default_stream()
        enqueue(comm_stream.handle)
        for tensor in fenced:
            assert isinstance(tensor, TorchMojoTensor)
            record_use(tensor._holder, comm_stream)
        comm_fence.mark_pending(
            index, comm_stream, cast(tuple[TorchMojoTensor, ...], fenced)
        )
        return _completed_work(result)

    def _fence_default(self, index: int):
        """Drain, and order the default stream after the comm stream.

        NCCL and RCCL require every rank to EXECUTE a communicator's
        operations in issue order. When comm-stream and default-stream
        collectives mix, this fence keeps device execution order equal to
        issue order. (The reverse direction is ``wait_default_stream`` in
        ``_stream_work``.)
        Unconditional, unlike ``comm_fence``: a comm-stream collective no
        consumer has touched yet is still pending on the device.
        """
        self._drained()
        if self._comm_stream_enabled and index in self._max_devices:
            get_stream(self._max_devices[index], "nccl").make_default_stream_wait()
            comm_fence.discard(index)

    def _init_comm(self, index: int) -> nccl.NcclComm:
        assert self._ccl is not None  # loaded by _device_state
        key = f"nccl-uid-{self._comm_seq}"
        self._comm_seq += 1
        if self.rank() == 0:
            unique_id = self._ccl.get_unique_id()
            self._store.set(key, unique_id)  # ty: ignore[invalid-argument-type] -- the stub says str; the binding takes bytes too
        else:
            unique_id = bytes(self._store.get(key))
            if len(unique_id) != nccl.NCCL_UNIQUE_ID_BYTES:
                raise RuntimeError(
                    f"bad {self._ccl.name} unique id from store key {key}"
                )
        return self._ccl.init_rank(self.size(), unique_id, self.rank())

    def _drained(self):
        """Launch every queued mojo kernel so the stream sees all producers."""
        deferred_compile.drain()

    def _dense(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor if tensor.is_contiguous() else tensor.contiguous()

    def _is_cpu(self, tensor: torch.Tensor) -> bool:
        if tensor.device.type == "cpu":
            return True
        if tensor.device.type != "mojo":
            raise RuntimeError(
                f"the mojo process group cannot handle tensors on {tensor.device}"
            )
        if isinstance(tensor, TorchMojoTensor) and tensor._device.api not in (
            "cuda",
            "hip",
        ):
            # mojo:<last> is MAX's host device (device_count counts it too);
            # NCCL/RCCL cannot touch host memory and gloo cannot touch mojo
            # wrappers, so refuse loudly rather than guess.
            raise NotImplementedError(
                "collectives on the mojo host device are not supported; use a "
                "GPU mojo device or a plain CPU tensor"
            )
        return False

    # -- collectives ---------------------------------------------------------

    @_loud
    def allreduce(
        self, tensors: list[torch.Tensor], opts: AllreduceOptions = AllreduceOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.allreduce(tensors, opts)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        op = _nccl_red_op(opts.reduceOp)
        comm, stream, index = self._device_state(tensor)
        staged = self._dense(tensor)

        def enqueue(handle: int):
            comm.all_reduce(
                _ptr_of(staged),
                _ptr_of(staged),
                staged.numel(),
                _nccl_dtype(staged.dtype),
                op,
                handle,
            )

        if staged is tensor:
            work = self._stream_work(index, tensors, (tensor,), enqueue)
            if work is not None:
                return work
        self._fence_default(index)
        enqueue(stream)
        if staged is not tensor:
            tensor.copy_(staged)
        return _completed_work(tensors)

    @_loud
    def allreduce_coalesced(
        self,
        tensors: list[torch.Tensor],
        opts: AllreduceCoalescedOptions = AllreduceCoalescedOptions(),
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.allreduce_coalesced(tensors, opts)
        op = _nccl_red_op(opts.reduceOp)
        comm, stream, index = self._device_state(tensors[0])
        staged = [self._dense(t) for t in tensors]

        def enqueue(handle: int):
            comm.group_start()
            for s in staged:
                comm.all_reduce(
                    _ptr_of(s), _ptr_of(s), s.numel(), _nccl_dtype(s.dtype), op, handle
                )
            comm.group_end()

        if all(s is t for s, t in zip(staged, tensors)):
            work = self._stream_work(index, tensors, tuple(tensors), enqueue)
            if work is not None:
                return work
        self._fence_default(index)
        enqueue(stream)
        for original, s in zip(tensors, staged):
            if s is not original:
                original.copy_(s)
        return _completed_work(tensors)

    @_loud
    def broadcast(
        self, tensors: list[torch.Tensor], opts: BroadcastOptions = BroadcastOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.broadcast(tensors, opts)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        comm, stream, index = self._device_state(tensor)
        staged = self._dense(tensor)
        root = int(opts.rootRank)

        def enqueue(handle: int):
            comm.broadcast(
                _ptr_of(staged),
                _ptr_of(staged),
                staged.numel(),
                _nccl_dtype(staged.dtype),
                root,
                handle,
            )

        if staged is tensor:
            work = self._stream_work(index, tensors, (tensor,), enqueue)
            if work is not None:
                return work
        self._fence_default(index)
        enqueue(stream)
        if staged is not tensor:
            tensor.copy_(staged)
        return _completed_work(tensors)

    @_loud
    def reduce(
        self, tensors: list[torch.Tensor], opts: ReduceOptions = ReduceOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.reduce(tensors, opts)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        comm, stream, index = self._device_state(tensor)
        staged = self._dense(tensor)
        op = _nccl_red_op(opts.reduceOp)
        root = int(opts.rootRank)

        def enqueue(handle: int):
            comm.reduce(
                _ptr_of(staged),
                _ptr_of(staged),
                staged.numel(),
                _nccl_dtype(staged.dtype),
                op,
                root,
                handle,
            )

        if staged is tensor:
            work = self._stream_work(index, tensors, (tensor,), enqueue)
            if work is not None:
                return work
        self._fence_default(index)
        enqueue(stream)
        if staged is not tensor:
            tensor.copy_(staged)
        return _completed_work(tensors)

    @_loud
    def allgather(
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions = AllgatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.allgather(output_tensors, input_tensors, opts)
        self._one_per_rank(input_tensors)
        source = self._dense(input_tensors[0])
        outputs = output_tensors[0]
        flat = torch.empty(
            self.size() * source.numel(), dtype=source.dtype, device=source.device
        )
        comm, stream, index = self._device_state(source)
        self._fence_default(index)
        comm.all_gather(
            _ptr_of(source),
            _ptr_of(flat),
            source.numel(),
            _nccl_dtype(source.dtype),
            stream,
        )
        for peer, out in enumerate(outputs):
            chunk = flat.narrow(0, peer * source.numel(), source.numel())
            out.copy_(chunk.view(out.shape))
        return _completed_work(outputs)

    @_loud
    def _allgather_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        opts: AllgatherOptions = AllgatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensor):
            return self._gloo._allgather_base(output_tensor, input_tensor, opts)
        source = self._dense(input_tensor)
        if output_tensor.numel() != source.numel() * self.size():
            raise ValueError("all_gather_into_tensor output has the wrong size")
        dest = self._dense(output_tensor)
        comm, stream, index = self._device_state(source)

        def enqueue(handle: int):
            comm.all_gather(
                _ptr_of(source),
                _ptr_of(dest),
                source.numel(),
                _nccl_dtype(source.dtype),
                handle,
            )

        if dest is output_tensor:
            # A staged (pre-copied) SOURCE is fine on the comm-stream path:
            # the copy rides the default stream before the fence. Only
            # post-collective copy-backs disqualify.
            work = self._stream_work(
                index, [output_tensor], (source, output_tensor), enqueue
            )
            if work is not None:
                return work
        self._fence_default(index)
        enqueue(stream)
        if dest is not output_tensor:
            output_tensor.copy_(dest)
        return _completed_work([output_tensor])

    @_loud
    def allgather_into_tensor_coalesced(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions = AllgatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.allgather_into_tensor_coalesced(
                output_tensors, input_tensors, opts
            )
        comm, stream, index = self._device_state(input_tensors[0])
        staged_in = [self._dense(t) for t in input_tensors]
        staged_out = [self._dense(t) for t in output_tensors]

        def enqueue(handle: int):
            comm.group_start()
            for source, dest in zip(staged_in, staged_out):
                comm.all_gather(
                    _ptr_of(source),
                    _ptr_of(dest),
                    source.numel(),
                    _nccl_dtype(source.dtype),
                    handle,
                )
            comm.group_end()

        if all(s is t for s, t in zip(staged_out, output_tensors)):
            work = self._stream_work(
                index, output_tensors, (*staged_in, *output_tensors), enqueue
            )
            if work is not None:
                return work
        self._fence_default(index)
        enqueue(stream)
        for original, s in zip(output_tensors, staged_out):
            if s is not original:
                original.copy_(s)
        return _completed_work(output_tensors)

    @_loud
    def reduce_scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.reduce_scatter(output_tensors, input_tensors, opts)
        output = output_tensors[0]
        inputs = input_tensors[0]
        if len(inputs) != self.size():
            raise ValueError("reduce_scatter expects world_size input tensors")
        count = output.numel()
        flat = torch.empty(
            self.size() * count, dtype=output.dtype, device=output.device
        )
        for peer, chunk in enumerate(inputs):
            flat.narrow(0, peer * count, count).copy_(chunk.reshape(-1))
        dest = self._dense(output)
        comm, stream, index = self._device_state(dest)
        self._fence_default(index)
        comm.reduce_scatter(
            _ptr_of(flat),
            _ptr_of(dest),
            count,
            _nccl_dtype(dest.dtype),
            _nccl_red_op(opts.reduceOp),
            stream,
        )
        if dest is not output:
            output.copy_(dest)
        return _completed_work(output_tensors)

    @_loud
    def _reduce_scatter_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensor):
            return self._gloo._reduce_scatter_base(output_tensor, input_tensor, opts)
        source = self._dense(input_tensor)
        dest = self._dense(output_tensor)
        if source.numel() != dest.numel() * self.size():
            raise ValueError("reduce_scatter_tensor input has the wrong size")
        comm, stream, index = self._device_state(dest)
        op = _nccl_red_op(opts.reduceOp)

        def enqueue(handle: int):
            comm.reduce_scatter(
                _ptr_of(source),
                _ptr_of(dest),
                dest.numel(),
                _nccl_dtype(dest.dtype),
                op,
                handle,
            )

        if dest is output_tensor:
            work = self._stream_work(
                index, [output_tensor], (source, output_tensor), enqueue
            )
            if work is not None:
                return work
        self._fence_default(index)
        enqueue(stream)
        if dest is not output_tensor:
            output_tensor.copy_(dest)
        return _completed_work([output_tensor])

    @_loud
    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.reduce_scatter_tensor_coalesced(
                output_tensors, input_tensors, opts
            )
        comm, stream, index = self._device_state(output_tensors[0])
        op = _nccl_red_op(opts.reduceOp)
        staged_in = [self._dense(t) for t in input_tensors]
        staged_out = [self._dense(t) for t in output_tensors]

        def enqueue(handle: int):
            comm.group_start()
            for source, dest in zip(staged_in, staged_out):
                comm.reduce_scatter(
                    _ptr_of(source),
                    _ptr_of(dest),
                    dest.numel(),
                    _nccl_dtype(dest.dtype),
                    op,
                    handle,
                )
            comm.group_end()

        if all(s is t for s, t in zip(staged_out, output_tensors)):
            work = self._stream_work(
                index, output_tensors, (*staged_in, *output_tensors), enqueue
            )
            if work is not None:
                return work
        self._fence_default(index)
        enqueue(stream)
        for original, s in zip(output_tensors, staged_out):
            if s is not original:
                original.copy_(s)
        return _completed_work(output_tensors)

    @_loud
    def alltoall_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts: AllToAllOptions = AllToAllOptions(),
    ) -> Work:
        if self._is_cpu(output_tensor):
            return self._gloo.alltoall_base(
                output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts
            )
        world = self.size()
        source = self._dense(input_tensor)
        dest = self._dense(output_tensor)
        row = source.numel() // max(source.shape[0], 1) if source.dim() else 1
        if not input_split_sizes:
            input_split_sizes = [source.shape[0] // world] * world
        if not output_split_sizes:
            output_split_sizes = [dest.shape[0] // world] * world
        dtype = _nccl_dtype(source.dtype)
        comm, stream, index = self._device_state(source)
        self._fence_default(index)
        comm.group_start()
        send_offset = 0
        recv_offset = 0
        itemsize = source.element_size()
        for peer in range(world):
            send_count = input_split_sizes[peer] * row
            recv_count = output_split_sizes[peer] * row
            comm.send(
                _ptr_of(source) + send_offset * itemsize,
                send_count,
                dtype,
                peer,
                stream,
            )
            comm.recv(
                _ptr_of(dest) + recv_offset * itemsize, recv_count, dtype, peer, stream
            )
            send_offset += send_count
            recv_offset += recv_count
        comm.group_end()
        if dest is not output_tensor:
            output_tensor.copy_(dest)
        return _completed_work([output_tensor])

    @_loud
    def alltoall(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: AllToAllOptions = AllToAllOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.alltoall(output_tensors, input_tensors, opts)
        comm, stream, index = self._device_state(input_tensors[0])
        staged_in = [self._dense(t) for t in input_tensors]
        staged_out = [self._dense(t) for t in output_tensors]
        self._fence_default(index)
        comm.group_start()
        for peer in range(self.size()):
            source = staged_in[peer]
            dest = staged_out[peer]
            comm.send(
                _ptr_of(source), source.numel(), _nccl_dtype(source.dtype), peer, stream
            )
            comm.recv(
                _ptr_of(dest), dest.numel(), _nccl_dtype(dest.dtype), peer, stream
            )
        comm.group_end()
        for original, s in zip(output_tensors, staged_out):
            if s is not original:
                original.copy_(s)
        return _completed_work(output_tensors)

    @_loud
    def gather(
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: GatherOptions = GatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.gather(output_tensors, input_tensors, opts)
        root = int(opts.rootRank)
        source = self._dense(input_tensors[0])
        dtype = _nccl_dtype(source.dtype)
        comm, stream, index = self._device_state(source)
        result: list[torch.Tensor] = []
        if self.rank() == root:
            outputs = output_tensors[0]
            # The root's own contribution is a local device copy, not a
            # self-send: same convention as ProcessGroupNCCL.
            outputs[root].copy_(source.view(outputs[root].shape))
            staged_out = [self._dense(t) for t in outputs]
            self._fence_default(index)
            comm.group_start()
            for peer, dest in enumerate(staged_out):
                if peer != root:
                    comm.recv(_ptr_of(dest), dest.numel(), dtype, peer, stream)
            comm.group_end()
            for original, s in zip(outputs, staged_out):
                if s is not original:
                    original.copy_(s)
            result = outputs
        else:
            self._fence_default(index)
            comm.group_start()
            comm.send(_ptr_of(source), source.numel(), dtype, root, stream)
            comm.group_end()
        return _completed_work(result)

    @_loud
    def scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ScatterOptions = ScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.scatter(output_tensors, input_tensors, opts)
        root = int(opts.rootRank)
        dest = self._dense(output_tensors[0])
        dtype = _nccl_dtype(dest.dtype)
        comm, stream, index = self._device_state(dest)
        if self.rank() == root:
            sources = [self._dense(t) for t in input_tensors[0]]
            output_tensors[0].copy_(sources[root].view(output_tensors[0].shape))
            self._fence_default(index)
            comm.group_start()
            for peer, source in enumerate(sources):
                if peer != root:
                    comm.send(_ptr_of(source), source.numel(), dtype, peer, stream)
            comm.group_end()
        else:
            self._fence_default(index)
            comm.group_start()
            comm.recv(_ptr_of(dest), dest.numel(), dtype, root, stream)
            comm.group_end()
            if dest is not output_tensors[0]:
                output_tensors[0].copy_(dest)
        return _completed_work([output_tensors[0]])

    @_loud
    def send(self, tensors: list[torch.Tensor], dst_rank: int, tag: int) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.send(tensors, dst_rank, tag)
        self._one_per_rank(tensors)
        source = self._dense(tensors[0])
        comm, stream, index = self._device_state(source)

        def enqueue(handle: int):
            comm.send(
                _ptr_of(source),
                source.numel(),
                _nccl_dtype(source.dtype),
                dst_rank,
                handle,
            )

        work = self._stream_work(index, tensors, (source,), enqueue)
        if work is not None:
            return work
        self._fence_default(index)
        enqueue(stream)
        return _completed_work(tensors)

    @_loud
    def recv(self, tensors: list[torch.Tensor], src_rank: int, tag: int) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.recv(tensors, src_rank, tag)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        dest = self._dense(tensor)
        comm, stream, index = self._device_state(dest)

        def enqueue(handle: int):
            comm.recv(
                _ptr_of(dest), dest.numel(), _nccl_dtype(dest.dtype), src_rank, handle
            )

        if dest is tensor:
            work = self._stream_work(index, tensors, (tensor,), enqueue)
            if work is not None:
                return work
        self._fence_default(index)
        enqueue(stream)
        if dest is not tensor:
            tensor.copy_(dest)
        return _completed_work(tensors)

    @_loud
    def barrier(self, opts: BarrierOptions = BarrierOptions()) -> Work:
        # Complete this rank's device work first, then rendezvous over gloo,
        # for the "everything before me is done everywhere" meaning users
        # expect. torch.mojo.synchronize() alone would not cover in-flight
        # collectives on the comm stream.
        if self._comms:
            torch_mojo_device_module.synchronize()
            self._quiesce_comm_streams()
        return self._gloo.barrier(opts)

    def _one_per_rank(self, tensors: list[torch.Tensor]):
        if len(tensors) != 1:
            raise NotImplementedError(
                "the mojo backend runs one process per GPU; multi-device-per-rank "
                f"collectives are not supported (got {len(tensors)} tensors)"
            )


def create_mojo_process_group(
    store: Store, rank: int, world_size: int, timeout: datetime.timedelta
) -> MojoProcessGroup:
    """The creator function registered with torch.distributed.Backend."""
    return MojoProcessGroup(store, rank, world_size, timeout)


# NVIDIA has one visibility variable. AMD has two levels: ROCR_VISIBLE_DEVICES
# masks at the HSA level and HIP_VISIBLE_DEVICES (or CUDA_VISIBLE_DEVICES, which
# the HIP runtime reads as its fallback) indexes INTO the ROCR-visible set.
_HSA_LEVEL = "ROCR_VISIBLE_DEVICES"
_RUNTIME_LEVEL = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")


def _visible_entries(var: str) -> list[str] | None:
    """The comma list in `var`, or None when it is unset or empty."""
    visible = os.environ.get(var)
    if visible is None:
        return None
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    return entries or None


def _keep_local_rank_entry(var: str, entries: list[str], rank_index: int):
    if len(entries) <= 1:
        return  # already pinned, e.g. one srun task per GPU
    if rank_index >= len(entries):
        raise RuntimeError(
            f"LOCAL_RANK={rank_index} but {var}={os.environ[var]!r} lists only "
            f"{len(entries)} devices"
        )
    os.environ[var] = entries[rank_index]


def use_local_rank_gpu():
    """Pin this torchrun worker to its GPU via the vendor's visibility variable.

    Call as early as possible — before any mojo tensor is created and before
    MAX enumerates devices (enumeration is cached per process). With exactly
    one visible GPU per rank, ``mojo``/``mojo:0`` is always the right device,
    the phantom ``privateuseone:0`` TensorImpl index is always truthful, and
    each process binds a single CUDA context / HIP device.

    SLURM (and other launchers) often pre-set the visibility variable to the
    whole allocation — ``CUDA_VISIBLE_DEVICES=0,...,7`` on NVIDIA,
    ``ROCR_VISIBLE_DEVICES=0,...,3`` on AMD; in that case each rank keeps
    only its LOCAL_RANK-th entry. A single already-pinned entry is left
    alone (one srun task per GPU pins that way). When no variable is set at
    all, both the CUDA and HIP ones are set to LOCAL_RANK, whichever runtime
    turns out to be present.

    Only one level is ever narrowed. When ``ROCR_VISIBLE_DEVICES`` is
    present it is the level that gets the rank's entry, and any runtime-level
    list (SLURM's gres plugin exports ``CUDA_VISIBLE_DEVICES`` next to it by
    default) is rewritten to ``"0"``, the only index that exists in a
    one-entry HSA set — narrowing both levels by rank would compose to no
    visible GPU at all for every rank but 0.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    try:
        rank_index = int(local_rank)
    except ValueError:
        raise RuntimeError(f"LOCAL_RANK={local_rank!r} is not an integer") from None
    hsa = _visible_entries(_HSA_LEVEL)
    if hsa is not None:
        _keep_local_rank_entry(_HSA_LEVEL, hsa, rank_index)
        for var in _RUNTIME_LEVEL:
            if _visible_entries(var) is not None:
                os.environ[var] = "0"
        return
    pinned = False
    for var in _RUNTIME_LEVEL:
        entries = _visible_entries(var)
        if entries is None:
            continue
        _keep_local_rank_entry(var, entries, rank_index)
        pinned = True
    if not pinned:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank_index)
        os.environ["HIP_VISIBLE_DEVICES"] = str(rank_index)
