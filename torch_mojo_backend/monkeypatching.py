"""Every monkeypatch this project applies, in one place.

A monkeypatch here means replacing or mutating, at runtime, an attribute of
a module or class we do not own -- mostly PyTorch internals that have no
extension point yet. Each patch is one function with a docstring saying what
upstream lacks, so that the patch can be turned into an upstream PR and
deleted from here. Nothing else in the package may patch a third-party
module; keep new patches in this file
(``tests/test_monkeypatching_is_centralized.py`` enforces that for
``torch``-rooted assignments).

Device patch installers are called by ``register.register_mojo_devices``.
The Dynamo stream-metadata patch is installed at package import, before
tracing can begin, including for CUDA-only users. Official registration APIs (``torch.library.impl``, the
PrivateUse1 backend module, ``torch.__future__`` toggles) are not
monkeypatches and stay in ``mojo_device/register.py``.
"""

import functools
import json
import math
import os
import pickle
import sys
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, cast

import torch
import torch._dynamo.guards as dynamo_guards
import torch._dynamo.utils as dynamo_utils
import torch._tensor_str
import torch.distributed.distributed_c10d as c10d
import torch.distributed.tensor as dtensor
import torch.fx.traceback
import torch.utils._triton
from torch._dynamo import source as dynamo_source
from torch._dynamo.symbolic_convert import InstructionTranslator
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.lazy import LazyVariableTracker
from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

from torch_mojo_backend.distributed import flight_recorder
from torch_mojo_backend.native import device_module

try:
    from torch._dynamo.variables import streams as dynamo_streams
except ImportError:  # Older torch releases predate generic stream tracing.
    dynamo_streams = None

if TYPE_CHECKING:
    from torch._dynamo.variables.streams import (
        StreamContextVariable,
        StreamVariable,
        SymbolicStreamState,
    )

    # Optional Triton dependency: its wheels are unavailable on macOS.
    from triton.backends.driver import DriverBase


def preserve_dynamo_stream_contexts():
    """Dynamo's single stream annotation loses outer contexts on other devices.

    Retain the full context stack in extra metadata for our backend. Install
    before tracing, including when CUDA users do not register the mojo device.
    The upstream stream annotation stays intact for other backends.
    """
    if dynamo_streams is None:
        return
    context = dynamo_streams.StreamContextVariable
    original_enter = context.enter
    if getattr(original_enter, "_torch_mojo_backend", False):
        return

    def enter(
        self: "StreamContextVariable", tx: InstructionTranslator, *args: VariableTracker
    ) -> VariableTracker:
        custom = torch.fx.traceback.get_current_meta().get("custom", {})
        indices = custom.get("torch_mojo_streams", ())
        # Newer Dynamo renamed this field; writing target_values there is
        # silently ignored by the annotation context's enter implementation.
        attribute = "annotation" if hasattr(self, "annotation") else "target_values"
        annotation = dict(getattr(self, attribute))
        annotation["torch_mojo_streams"] = (
            *indices,
            self.get_stream().user_object_index,
        )
        setattr(self, attribute, annotation)
        return original_enter(self, tx, *args)

    enter._torch_mojo_backend = True  # ty: ignore[unresolved-attribute] -- idempotent patch marker
    context.enter = enter


def fix_dynamo_current_stream_device():
    """Dynamo 2.11 falls back to the default accelerator's stream when an
    explicitly requested device has no entry in its symbolic stream stack.
    After PrivateUse1 registration that turns CUDA stream queries into mojo
    streams. Its runtime CurrentStreamSource helper also ignores the device
    type, and its CUDA handler treats no device argument as the default
    accelerator. Preserve the requested CUDA device in all three places.
    """
    state = getattr(dynamo_streams, "SymbolicStreamState", None)
    if (
        dynamo_streams is None
        or state is None
        or getattr(dynamo_utils.get_current_stream, "_torch_mojo_backend", False)
    ):
        return
    original_get = dynamo_utils.get_current_stream
    original_current = state.cur_stream
    handlers = TorchInGraphFunctionVariable._get_handlers()
    original_cuda = handlers[torch.cuda.current_stream]

    @wraps(original_get)
    def get_current_stream(device: torch.device) -> torch.Stream:
        if device.type == "cuda":
            stream = torch.cuda.current_stream(device)
            # Match accelerator.current_stream's generic object. Letting
            # Dynamo guard these temporary CUDA subclass wrappers triggers
            # weakref_dealloc crashes in torch 2.11 (also with backend=eager).
            return torch.Stream(
                stream_id=stream.stream_id,
                device_index=stream.device_index,
                device_type=stream.device_type,
            )
        return original_get(device)

    def cur_stream(
        self: "SymbolicStreamState", device: torch.device | None = None
    ) -> "StreamVariable":
        if device is None:
            return original_current(self, device)
        actual = get_current_stream(device)
        for stream in reversed(self.cur_stream_stack):
            if stream.device == actual.device:
                return stream
        return cast(
            "StreamVariable",
            LazyVariableTracker.create(
                actual, source=dynamo_source.CurrentStreamSource(actual.device)
            ),
        )

    def cuda_current_stream(
        self: TorchInGraphFunctionVariable,
        tx: InstructionTranslator,
        *args: VariableTracker,
        **kwargs: VariableTracker,
    ) -> VariableTracker:
        if len(args) + len(kwargs) > 1 or (kwargs and "device" not in kwargs):
            return original_cuda(self, tx, *args, **kwargs)
        value = args[0] if args else kwargs.get("device")
        requested = value.as_python_constant() if value is not None else None
        actual = torch.cuda.current_stream(requested)
        return original_cuda(self, tx, ConstantVariable.create(actual.device))

    get_current_stream._torch_mojo_backend = True  # ty: ignore[unresolved-attribute] -- idempotent patch marker
    dynamo_utils.get_current_stream = get_current_stream  # ty: ignore[invalid-assignment] -- torch's internal/public Stream return aliases differ in its stubs
    dynamo_guards.get_current_stream = get_current_stream  # ty: ignore[invalid-assignment] -- same Stream alias gap as utils above
    if dynamo_guards._CLOSURE_VARS is not None:
        dynamo_guards._CLOSURE_VARS["___get_current_stream"] = get_current_stream
    state.cur_stream = cur_stream
    handlers[torch.cuda.current_stream] = cuda_current_stream


def fix_privateuse1_dlpack_device_type():
    """`Tensor.__dlpack_device__` doesn't recognize a *renamed* PrivateUse1
    backend.

    ``torch/_tensor.py``'s ``Tensor.__dlpack_device__`` maps a PrivateUse1
    tensor to DLPack's ``kDLExtDev`` by comparing ``self.device.type``
    against the string literal ``"privateuse1"`` -- so after
    ``torch.utils.rename_privateuse1_backend("mojo")`` it never matches, and
    every mojo tensor's ``__dlpack_device__()`` raises ``ValueError("Unknown
    device type mojo for Dlpack")``. Two other call sites in that very same
    file (the ``__cuda_array_interface__`` gate) correctly compare against
    ``torch._C._get_privateuse1_backend_name()`` instead of the literal;
    this one method just didn't get the memo.

    ``Tensor.__dlpack__`` itself (the capsule export) is unaffected --
    ATen's C++ DLConvertor keys off the ``DeviceType`` enum, not the
    Python-visible name -- so only the device-query half needs patching.
    ``torch_compile_backend/compiler.py``'s ``fast_from_dlpack`` routes
    around this bug for its own zero-copy exchange (it never calls
    ``__dlpack_device__``), but plain ``torch.utils.dlpack`` /
    ``max.driver.Buffer.from_dlpack(t)`` usage elsewhere (user code,
    ``test_compile_mojo_device.py``) goes through the single-arg DLPack
    protocol, which calls ``__dlpack_device__()`` first and needs this fix.
    """
    original = torch.Tensor.__dlpack_device__
    if getattr(original, "_torch_mojo_backend", False):
        return

    from torch.utils.dlpack import (  # noqa: PLC0415 -- mirrors the private import inside the method being patched
        DLDeviceType,
    )

    @wraps(original)
    def __dlpack_device__(self: torch.Tensor) -> tuple[int, int]:
        if self.device.type == torch._C._get_privateuse1_backend_name():
            index = self.device.index if self.device.index is not None else 0
            return (DLDeviceType.kDLExtDev, index)
        return original(self)

    __dlpack_device__._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    torch.Tensor.__dlpack_device__ = (  # ty: ignore[invalid-assignment]
        __dlpack_device__
    )


def stage_mojo_checkpoint_tensors():
    """DCP's overlapping CPU loader stages only its preferred accelerator.

    With CUDA torch installed, `_OverlappingCpuLoader` selects CUDA even
    when saving Mojo tensors. Its `_refill` leaves other accelerators on
    device, and the filesystem writer then rejects them. Stage Mojo data
    synchronously when the selected loader stream cannot stage it; retain
    upstream's overlap for tensors on that stream's device type.
    """
    if not torch.distributed.is_available():
        return
    from torch.distributed.checkpoint import (  # noqa: PLC0415 -- checkpoint requires optional distributed support, guarded above
        filesystem,
    )

    original = filesystem._OverlappingCpuLoader.__init__
    if getattr(original, "_torch_mojo_backend", False):
        return

    def initialize(
        self: filesystem._OverlappingCpuLoader,
        resolve_fun: Callable[[object], torch.Tensor],
        stream: torch.Stream | None = None,
        inflight_threshhold: int = 1_000_000,
    ):
        original(self, resolve_fun, stream, inflight_threshhold)

        def resolve_data(item: object) -> torch.Tensor:
            tensor = resolve_fun(item)
            if tensor.device.type == "mojo" and self.device_type != "mojo":
                return tensor.detach().cpu()
            return tensor

        self.resolve_fun = resolve_data

    functools.update_wrapper(initialize, original)
    initialize._torch_mojo_backend = True  # ty: ignore[unresolved-attribute] -- idempotent patch marker
    filesystem._OverlappingCpuLoader.__init__ = initialize


def print_apple_gpu_tensors_from_the_host():
    """Tensor printing assumes a float64-capable device.

    ``torch/_tensor_str.py``'s ``_Formatter`` runs ``isfinite``, ``abs`` and
    ``min``/``max`` on the device, in float64 (``tensor_totype``), falling
    back to float32 only for a hardcoded list (MPS, fp64-less XPU, MAIA).
    Apple GPUs have no float64, so ``repr`` of any float tensor on a mojo
    Metal device raised. Format a host copy there instead, as torch already
    does for XLA/lazy/IPU/MTIA tensors; the ``device=`` suffix is computed by
    the caller before this runs, so it still names the mojo device.
    """
    original = torch._tensor_str._tensor_str
    if getattr(original, "_torch_mojo_backend", False):
        return

    @wraps(original)
    def _tensor_str(self: torch.Tensor, indent: int) -> str:
        if (
            self.device.type == "mojo"
            and device_module.get_device_properties(self.device).api == "metal"
        ):
            self = self.cpu()
        return original(self, indent)

    _tensor_str._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    torch._tensor_str._tensor_str = _tensor_str  # ty: ignore[invalid-assignment]


def fix_batch_isend_irecv_for_python_process_groups():
    """`batch_isend_irecv` never coalesces for a Python `ProcessGroup`
    subclass, and a bidirectional exchange then deadlocks.

    ``torch/distributed/distributed_c10d.py``'s ``batch_isend_irecv`` gates
    its NCCL-style coalescing on ``type(group) is ProcessGroup``. The mojo
    backend IS a Python subclass of ``ProcessGroup`` (it has to be: the
    C++ ``Backend`` cannot be implemented in Python), so the check is False
    however capable the backend is, and every operation in the list goes out
    as its own NCCL group. A two-rank exchange then deadlocks on the device:
    each rank's comm stream holds ``[send(->peer), recv(<-peer)]``, and the
    send cannot retire until the peer posts its recv, which sits behind that
    peer's own send. The very next line asks the backend whether it
    ``supports_coalescing``, which is the real question; ``isinstance`` is
    what the type check means. Everything else is torch's own code path,
    called unchanged.
    """
    original = c10d.batch_isend_irecv
    if getattr(original, "_torch_mojo_backend", False):
        return

    @wraps(original)
    def batch_isend_irecv(p2p_op_list: list[c10d.P2POp]) -> list[c10d.Work]:
        c10d._check_p2p_op_list(p2p_op_list)
        group = p2p_op_list[0].group or c10d._get_default_group()
        device = p2p_op_list[0].tensor.device
        coalesces = (
            type(group) is not torch.distributed.ProcessGroup
            and isinstance(group, torch.distributed.ProcessGroup)
            and group._get_backend(device).supports_coalescing
        )
        if not coalesces:
            return original(p2p_op_list)
        with c10d._coalescing_manager(group, device, async_ops=True) as manager:
            for op in p2p_op_list:
                peer = "group_dst" if op.op is c10d.isend else "group_src"
                op.op(op.tensor, group=op.group, tag=op.tag, **{peer: op.group_peer})
        return manager.works

    batch_isend_irecv._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    c10d.batch_isend_irecv = batch_isend_irecv  # ty: ignore[invalid-assignment]
    torch.distributed.batch_isend_irecv = batch_isend_irecv  # ty: ignore[invalid-assignment]


def serve_nccl_trace_dumps_from_the_mojo_backend():
    """`torch._C._distributed_c10d._dump_nccl_trace` and
    `_dump_nccl_trace_json` only read ProcessGroupNCCL's flight recorder.

    They are what a CUDA user calls for an on-demand dump, but they read a
    C++ singleton only ProcessGroupNCCL records into (FlightRecorderCuda.cpp)
    and exist only in CUDA builds (init.cpp, `USE_C10D_NCCL`). While a mojo
    process group is alive, both return its recorder's dump
    (distributed/flight_recorder.py), in the same format, on every build --
    unless ProcessGroupNCCL recorded too: then torch's dump comes back
    untouched (one dump cannot hold both, the analyzer aligns a rank by its
    first record id), and ours is the dump files (timeout, debug pipe). With
    no mojo group alive -- none yet, all destroyed, the watchdog off,
    TORCH_FR_BUFFER_SIZE=0 -- they are torch's own. `_dump_fr_trace*` (the
    generic recorder Gloo and FlightRecorderHook use) is left alone.
    """
    c10d_c = torch._C._distributed_c10d  # ty: ignore[possibly-missing-submodule] -- loaded by torch.distributed, imported above
    if _NCCL_TRACE_ORIGINALS:
        return
    trace = getattr(c10d_c, "_dump_nccl_trace", None)
    trace_json = getattr(c10d_c, "_dump_nccl_trace_json", None)
    _NCCL_TRACE_ORIGINALS.extend((trace, trace_json))

    def _dump_nccl_trace(
        includeCollectives: bool | None = None,  # noqa: N803 -- torch's keyword names
        includeStackTraces: bool | None = None,  # noqa: N803
        onlyActive: bool | None = None,  # noqa: N803
    ) -> bytes:
        if trace is not None:
            theirs = trace(includeCollectives, includeStackTraces, onlyActive)
            if not flight_recorder.active() or _recorded(pickle.loads(theirs)):
                return theirs
        return flight_recorder.dump_nccl_trace(
            includeCollectives, includeStackTraces, onlyActive
        )

    def _dump_nccl_trace_json(
        includeCollectives: bool | None = None,  # noqa: N803 -- torch's keyword names
        onlyActive: bool | None = None,  # noqa: N803
    ) -> bytes:
        if trace_json is not None:
            theirs = trace_json(includeCollectives, onlyActive)
            if not flight_recorder.active() or _recorded(json.loads(theirs)):
                return theirs
        return flight_recorder.dump_nccl_trace_json(includeCollectives, onlyActive)

    c10d_c._dump_nccl_trace = _dump_nccl_trace
    c10d_c._dump_nccl_trace_json = _dump_nccl_trace_json


def _recorded(dump: dict[str, object]) -> bool:
    """ProcessGroupNCCL's recorder holds something."""
    return bool(dump.get("entries") or dump.get("pg_config"))


# torch's `_dump_nccl_trace` / `_json` (None where the build has none), kept
# by the patch above, which also uses it as its done-once flag.
_NCCL_TRACE_ORIGINALS: list[Callable[..., bytes] | None] = []


def let_dtensor_differentiate_a_native_linear():
    """DTensor has no sharding rule for ``aten.linear_backward``, which a
    backend that implements ``aten.linear`` itself produces.

    ``aten::linear`` is CompositeImplicitAutograd, but a backend kernel for
    it (the mojo device has a fused one) makes autograd record
    ``derivatives.yaml``'s ``linear`` formula, whose backward is
    ``aten::linear_backward``. DTensor wrappers carry the device's autograd
    key, so tensor parallelism on such a device reaches that op, and
    ``torch/distributed/tensor/_dispatch.py`` has neither a strategy nor a
    decomposition for it (``linear`` itself decomposes, being composite).
    The handler below is the formula in DTensor ops: ``grad @ weight``,
    ``grad.T @ input`` and the row sum over the flattened leading dims, the
    same way ``_custom_op_handlers`` already does ``convolution_backward``.
    """
    dispatcher = dtensor.DTensor._op_dispatcher
    op = torch.ops.aten.linear_backward.default
    if op in dispatcher._custom_op_handlers:
        return

    def linear_backward(
        op_call: torch._ops.OpOverload,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> tuple[torch.Tensor | None, ...]:
        input, grad, weight, mask = args
        assert isinstance(input, torch.Tensor)
        assert isinstance(grad, torch.Tensor)
        assert isinstance(weight, torch.Tensor)
        assert isinstance(mask, list | tuple)
        # explicit row counts: reshape(-1, 0) is ambiguous for empty features
        rows = math.prod(grad.shape[:-1])
        grad_2d = grad.reshape(rows, grad.shape[-1])
        return (
            torch.matmul(grad, weight) if mask[0] else None,
            grad_2d.t() @ input.reshape(rows, input.shape[-1]) if mask[1] else None,
            grad_2d.sum(0) if mask[2] else None,
        )

    dispatcher._custom_op_handlers[op] = linear_backward


def add_mojo_to_the_inductor_gpu_types():
    """Inductor decides "is this an accelerator?" against a hardcoded list.

    ``torch/_inductor/utils.py``'s ``GPU_TYPES = ["cuda", "mps", "xpu",
    "mtia"]`` feeds ``is_gpu()``, and there is no registry for it the way
    there is for the device interface and the codegen backend -- so a
    PrivateUse1 backend reads as a CPU however completely it registers.
    What that costs: no device guard around the generated kernels
    (``device_need_guard``), CPU loop-merging heuristics in the scheduler,
    no alignment-driven input cloning, and the "which device does this
    pointwise op belong to" scan in ``lowering.py`` never picking ours.

    Appending is safe here because ``is_gpu`` reads the module global and
    the two modules that ``from ... import GPU_TYPES`` bind the same list
    object.

    The list is process-wide and ``get_gpu_type()`` asserts at most one of
    its entries is available, so "mojo" and a working ``torch.cuda`` cannot
    both be in it: with both, that assert fires in the autotuning
    subprocess setup and the profiler benchmarking, for CUDA workloads as
    much as for ours. There is no per-graph answer to give it, so this
    patch stands aside on a torch with a working CUDA/ROCm build --
    ``inductor.enable_inductor()`` refuses outright there, and says why.
    """
    if torch.cuda.is_available():
        return
    from torch._inductor import (  # noqa: PLC0415 -- 0.87 s of sympy, measured; every register_mojo_devices() would otherwise pay it
        utils as inductor_utils,
    )

    if "mojo" not in inductor_utils.GPU_TYPES:
        inductor_utils.GPU_TYPES.append("mojo")


def let_has_triton_see_the_mojo_device():
    """`torch.utils._triton.has_triton` asks a hardcoded device dict.

    Its ``triton_supported_devices = {"cuda", "xpu", "cpu", "mtia"}`` is a
    local of the function, so a PrivateUse1 backend cannot add itself: with
    a CPU torch wheel every entry answers "not available", ``has_triton()``
    is False, and Inductor then refuses to build a Triton scheduling for
    our device (``scheduler.py``'s ``create_backend`` raises
    ``TritonMissing``). The fix upstream is one more entry in that dict,
    fed by the registered device interfaces.

    The rebinding loop is the price of ``from torch.utils._triton import
    has_triton``: a dozen torch modules hold the original by value.
    """
    from torch_mojo_backend.native import (  # noqa: PLC0415 -- `torch.mojo`, imported here so registering the devices does not pull in the native backend
        device_module,
    )

    original = torch.utils._triton.has_triton
    if getattr(original, "_torch_mojo_backend", False):
        return

    @functools.cache
    def has_triton() -> bool:
        return original() or (
            torch.utils._triton.has_triton_package() and device_module.is_available()
        )

    has_triton._torch_mojo_backend = True  # ty: ignore[unresolved-attribute]
    torch.utils._triton.has_triton = has_triton
    for module in list(sys.modules.values()):
        if getattr(module, "has_triton", None) is original:
            module.has_triton = has_triton  # ty: ignore[unresolved-attribute] -- setting an attribute torch bound by value


def register_the_mojo_triton_target(driver: "type[DriverBase]", api: str):
    """Triton picks a compiler backend by the name inside the target, and
    Inductor puts the torch device type there.

    ``torch/_inductor/runtime/triton_heuristics.py`` builds
    ``GPUTarget(compile_meta["device_type"], ...)`` from
    ``DeviceProperties.type``, which is the torch device type -- "mojo".
    ``triton.compiler.compiler.make_backend`` then asks every registered
    backend ``supports_target(target)``, and NVIDIA's (or AMD's) answers
    only to "cuda" (or "hip"), so compilation dies with "0 compatible
    backends". Registering an alias of the vendor backend under our name is
    what an out-of-tree Triton backend would do through the
    ``triton.backends`` entry point; only ``supports_target`` differs, the
    target's ``arch`` (compute capability, or the gfx string) is what the
    backend actually compiles against. `api` (``accelerator_api()``) picks
    which vendor backend to alias -- NVIDIA's ``CUDABackend`` for "cuda",
    AMD's ``HIPBackend`` for "hip".

    The entry also carries `driver` (`triton_driver.driver_class()`), because
    the same dict is what `torch/_inductor/runtime/triton_helpers.py`'s
    ``set_driver_to_gpu`` -- run at the import of every generated kernel
    module -- scans for a backend whose driver ``is_active()``. The vendor
    driver says no (it asks ``torch.cuda.is_available()`` / the HIP
    equivalent), so without an entry of ours the import of the generated
    module raises "Could not find an active GPU backend".
    """
    import triton.backends  # noqa: PLC0415 -- triton is optional
    from triton.backends.compiler import (  # noqa: PLC0415 -- triton is optional
        GPUTarget,
    )

    if "mojo" in triton.backends.backends:
        return

    if api == "hip":
        from triton.backends.amd.compiler import (  # noqa: PLC0415 -- triton is optional
            HIPBackend,
        )

        class MojoBackend(HIPBackend):
            @staticmethod
            def supports_target(target: GPUTarget) -> bool:
                return getattr(target, "backend", None) == "mojo"
    else:
        from triton.backends.nvidia.compiler import (  # noqa: PLC0415 -- triton is optional
            CUDABackend,
        )

        class MojoBackend(CUDABackend):
            @staticmethod
            def supports_target(target: GPUTarget) -> bool:
                return getattr(target, "backend", None) == "mojo"

    triton.backends.backends["mojo"] = triton.backends.Backend(
        compiler=MojoBackend, driver=driver
    )


def compile_inductor_kernels_in_process():
    """An Inductor compile worker is a fresh interpreter that imports only
    torch, and nothing can teach it about an out-of-tree device.

    ``torch/_inductor/compile_worker/__main__.py`` (started as a subprocess
    by default) imports torch and triton and nothing else, so our Triton
    backend registration is absent there and importing a generated kernel
    module dies inside ``torch/_inductor/runtime/triton_helpers.py``'s
    ``set_driver_to_gpu`` with "Could not find an active GPU backend".
    Upstream has no worker-startup import hook, so the compiles run in this
    process instead; on the graphs measured here that is also the faster of
    the two, the pool costing more to start than these compiles take.

    ``TORCHINDUCTOR_WORKER_START=fork`` is the other way out -- a forked
    worker inherits the registration and works -- and is left to the user,
    forking a process that already holds a GPU context being its own risk.
    An explicit ``TORCHINDUCTOR_COMPILE_THREADS`` also stands.
    """
    from torch._inductor import (  # noqa: PLC0415 -- see add_mojo_to_the_inductor_gpu_types
        config as inductor_config,
    )

    if (
        inductor_config.worker_start_method != "fork"
        and "TORCHINDUCTOR_COMPILE_THREADS" not in os.environ
    ):
        inductor_config.compile_threads = 1
