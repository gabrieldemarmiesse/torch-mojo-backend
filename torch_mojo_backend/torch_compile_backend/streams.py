"""Keep stream/event orchestration outside AOTAutograd's tensor tracing.

Dynamo represents stream contexts as node metadata and stream/event methods
as ordinary Python calls. AOT executes those calls while tracing and drops
them from its tensor graph. Partition *before* AOT, retaining the calls in
the runtime wrapper and compiling each contiguous tensor region separately.
This is backend partitioning, so it also works with ``fullgraph=True``.
"""

import ctypes
import functools
import threading
import weakref
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from contextvars import ContextVar
from typing import cast

import max._core
import max.driver
import torch
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_flatten, tree_leaves, tree_unflatten

try:
    from torch._dynamo import graph_bytecode_inputs
except ImportError:  # PyTorch before the generic stream tracing API.
    graph_bytecode_inputs = None

RegionCompiler = Callable[[torch.fx.GraphModule, list[object]], Callable[..., object]]

# Concrete torch contexts exist only for the forward invocation. Backward
# executes on the autograd engine's current stream and uses ordinary fences.
ACTIVE_STREAM_CONTEXTS: ContextVar[tuple[tuple[torch.Stream, ...], ...]] = ContextVar(
    "torch_mojo_active_stream_contexts", default=((),)
)
_STREAM_POOL_LOCK = threading.Lock()


@functools.cache
def _stream_api() -> ctypes.CDLL:
    # MAX's Python DeviceQueue creates a separate context, not a member of
    # the context's indexed stream pool. Use the same runtime API as Mojo's
    # DeviceContext.create_stream until MAX exposes it in Python.
    api = ctypes.CDLL(max._core.__file__)
    api.AsyncRT_DeviceContext_numStreams.argtypes = [ctypes.c_void_p]
    api.AsyncRT_DeviceContext_numStreams.restype = ctypes.c_int
    api.AsyncRT_DeviceContext_createStream.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    api.AsyncRT_DeviceContext_createStream.restype = ctypes.c_char_p
    api.AsyncRT_DeviceContext_selectStream.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    api.AsyncRT_DeviceContext_selectStream.restype = ctypes.c_char_p
    api.AsyncRT_DeviceContext_release.argtypes = [ctypes.c_void_p]
    api.AsyncRT_DeviceContext_release.restype = None
    api.AsyncRT_DeviceContext_enqueue_wait_for_context.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    api.AsyncRT_DeviceContext_enqueue_wait_for_context.restype = ctypes.c_char_p
    return api


def _release_stream_views(api: ctypes.CDLL, views: dict[int, list[int]]):
    for contexts in views.values():
        for context in contexts:
            api.AsyncRT_DeviceContext_release(context)


class MaxStreamPool:
    def __init__(self, devices: list[max.driver.Device], count: int):
        self.views: dict[int, list[int]] = {}
        if count <= 1:
            return
        api = _stream_api()
        self._finalizer = weakref.finalize(self, _release_stream_views, api, self.views)
        ensure_side_streams(devices, count)
        for device in devices:
            if (
                device.label == "cpu"
                or cast("max.driver.Accelerator", device).api == "metal"
            ):
                continue
            contexts = self.views.setdefault(device.id, [])
            for index in range(1, count):
                view = ctypes.c_void_p()
                error = api.AsyncRT_DeviceContext_selectStream(
                    ctypes.byref(view), device._device_context_ptr(), index
                )
                if error:
                    raise RuntimeError(error.decode())
                assert view.value is not None
                contexts.append(view.value)

    def fence(self, buffers: list[max.driver.Buffer], *, entering: bool):
        if not self.views:
            return
        api = _stream_api()
        # All producer dependencies are already enqueued by the DLPack
        # imports. Fence each source context once, even for several tokens.
        sources = {
            (buffer.device.id, buffer.device._device_context_ptr())
            for buffer in buffers
            if buffer.device.label != "cpu"
        }
        for device_id, source in sources:
            for side in self.views.get(device_id, ()):
                destination, producer = (side, source) if entering else (source, side)
                error = api.AsyncRT_DeviceContext_enqueue_wait_for_context(
                    destination, producer
                )
                if error:
                    raise RuntimeError(error.decode())


def ensure_side_streams(devices: list[max.driver.Device], count: int):
    if count <= 1:
        return
    api = _stream_api()
    with _STREAM_POOL_LOCK:
        for device in devices:
            if (
                device.label == "cpu"
                or cast("max.driver.Accelerator", device).api == "metal"
            ):
                continue
            context = device._device_context_ptr()
            for _ in range(api.AsyncRT_DeviceContext_numStreams(context), count):
                stream = ctypes.c_void_p()
                error = api.AsyncRT_DeviceContext_createStream(
                    ctypes.byref(stream), 0, context
                )
                if error:
                    raise RuntimeError(error.decode())
            # Streams are owned by the context, just as in Mojo's non-owning
            # DeviceStream wrapper; no separate destruction is needed.


def resolve_streams(indices: tuple[int, ...]) -> tuple[torch.Stream, ...]:
    if not indices:
        return ()
    if graph_bytecode_inputs is None:
        raise RuntimeError("Compiled stream contexts require PyTorch 2.11 or newer")
    streams = tuple(
        graph_bytecode_inputs.get_external_object_by_index(index) for index in indices
    )
    if not all(isinstance(stream, torch.Stream) for stream in streams):
        raise RuntimeError("Compiled stream index no longer names a torch.Stream")
    return streams


def enter_streams(streams: tuple[torch.Stream, ...]) -> ExitStack:
    contexts = ExitStack()
    try:
        for stream in streams:
            # Generic Stream.__enter__ rejects CUDA after PrivateUse1 registration.
            context = (
                torch.cuda.stream(cast("torch.cuda.Stream", stream))
                if stream.device.type == "cuda"
                else stream
            )
            contexts.enter_context(context)
    except BaseException:
        contexts.close()
        raise
    return contexts


def current_stream(device: torch.device) -> torch.Stream:
    # After PrivateUse1 registration torch.accelerator names the mojo device.
    if device.type == "cuda":
        return torch.cuda.current_stream(device)
    return torch.accelerator.current_stream(device)


def _streams_of(contexts: Iterable[tuple[torch.Stream, ...]]) -> list[torch.Stream]:
    return list(dict.fromkeys(stream for streams in contexts for stream in streams))


def wait_for_contexts(contexts: Iterable[tuple[torch.Stream, ...]]):
    """Order each device's current stream after every stream of `contexts`."""
    for stream in _streams_of(contexts):
        current_stream(stream.device).wait_stream(stream)


def publish_to_contexts(contexts: Iterable[tuple[torch.Stream, ...]]):
    """Order every stream of `contexts` after its device's current stream."""
    for stream in _streams_of(contexts):
        stream.wait_stream(current_stream(stream.device))


def max_stream_id(node: torch.fx.Node) -> int:
    return node.meta.get("custom", {}).get("torch_mojo_max_stream", 0)


def _is_stream_object(node: torch.fx.Node) -> bool:
    return isinstance(node.meta.get("example_value"), torch.Stream | torch.Event)


def _is_orchestration(node: torch.fx.Node) -> bool:
    if _is_stream_object(node):
        return True
    if any(_is_stream_object(arg) for arg in node.all_input_nodes):
        return True
    return str(node.target).startswith("streams.")


def stream_indices(node: torch.fx.Node) -> tuple[int, ...]:
    custom = node.meta.get("custom", {})
    if "torch_mojo_streams" in custom:
        return custom["torch_mojo_streams"]
    index = custom.get("stream")
    return () if index is None else (index,)


def uses_streams(graph: torch.fx.Graph) -> bool:
    return any(
        bool(stream_indices(node)) or _is_orchestration(node) for node in graph.nodes
    )


class _StreamRegion(torch.nn.Module):
    def __init__(self, graph: torch.fx.GraphModule, indices: tuple[int, ...]):
        super().__init__()
        self.graph = graph
        self.indices = indices

    def _run(self, args: tuple[object, ...]) -> object:
        return self.graph(*args)

    def forward(self, *args: object) -> object:
        # Resolve on every invocation: streams constructed inside the user's
        # function are recreated by Dynamo's bytecode preamble on every call.
        with enter_streams(resolve_streams(self.indices)):
            return self._run(args)


class _CompiledRegion(_StreamRegion):
    def __init__(self, graph: torch.fx.GraphModule, compile_region: RegionCompiler):
        super().__init__(graph, ())
        output = next(node for node in graph.graph.nodes if node.op == "output")
        leaves, self.output_spec = tree_flatten(output.args[0])
        output.args = (tuple(leaves),)
        graph.recompile()
        self.context_indices = list(
            dict.fromkeys(
                stream_indices(node)
                for node in graph.graph.nodes
                if node.op in ("call_function", "call_method")
            )
        )
        # Applying AOT's autograd Function under the original torch context
        # preserves the stream seen by backward hooks and gradient accumulation.
        # Training partitions have one context; inference may combine several.
        if len(self.context_indices) == 1:
            self.indices = self.context_indices[0]
        placements = {
            indices: i + 1 for i, indices in enumerate(self.context_indices) if indices
        }
        for node in graph.graph.nodes:
            # Placement now belongs to this region's runtime context. AOT's
            # backward stream pass must not resolve Dynamo's temporary object
            # table after the forward invocation has ended.
            node.meta = dict(node.meta)
            placement = placements.get(stream_indices(node), 0)
            if "custom" in node.meta:
                node.meta["custom"] = dict(node.meta["custom"])
                node.meta["custom"].pop("stream", None)
                node.meta["custom"].pop("torch_mojo_streams", None)
            node.meta.setdefault("custom", {})["torch_mojo_max_stream"] = placement
        # Use Dynamo's symbolic examples now, not concrete runtime arguments
        # on the first call: the latter silently specializes dynamic shapes.
        examples = [
            node.meta.get("example_value", node.meta.get("val"))
            for node in graph.graph.nodes
            if node.op == "placeholder"
        ]
        self.compiled = compile_region(graph, examples)

    def _run(self, args: tuple[object, ...]) -> object:
        contexts = tuple(resolve_streams(indices) for indices in self.context_indices)
        token = ACTIVE_STREAM_CONTEXTS.set(tuple(dict.fromkeys(((), *contexts))))
        try:
            result = self.compiled(*args)
        finally:
            ACTIVE_STREAM_CONTEXTS.reset(token)
        if len(contexts) > 1:
            # Several contexts run on the ambient stream, where AOT's epilogue
            # also copies mutated inputs back, after the graph's own exit
            # handoff. The graph already waited for the ambient stream, so
            # this adds only that epilogue to what the contexts wait for.
            publish_to_contexts(contexts)
        assert isinstance(result, list | tuple)
        return tree_unflatten(list(result), self.output_spec)


def compile_stream_graph(
    gm: torch.fx.GraphModule, compile_region: RegionCompiler
) -> torch.fx.GraphModule:
    partitions = {}
    regions = {}
    partition = -1
    previous = None
    training = any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for node in gm.graph.nodes
        for value in tree_leaves(node.meta.get("example_value", node.meta.get("val")))
    )
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        control = _is_orchestration(node)
        # A fused AOT autograd Function has one torch stream identity. Combining
        # distinct contexts there changes backward-hook behavior, even when
        # numerical gradients are correct. Keep training's original boundaries.
        key = (control, stream_indices(node) if control or training else ())
        if control or key != previous:
            partition += 1
        partitions[node] = partition
        regions[partition] = key
        previous = None if control else key

    # split_module searches example values for symbolic shapes; torch.Stream
    # and torch.Event are not tensor/symbolic values accepted by that search.
    examples = {
        node: node.meta.pop("example_value")
        for node in gm.graph.nodes
        if _is_stream_object(node)
    }
    try:
        result = split_module(gm, gm, partitions.__getitem__, keep_original_order=True)
    finally:
        for node, value in examples.items():
            node.meta["example_value"] = value
    for partition, (control, indices) in regions.items():
        name = f"submod_{partition}"
        graph = result.get_submodule(name)
        assert isinstance(graph, torch.fx.GraphModule)
        region = (
            _StreamRegion(graph, indices)
            if control
            else _CompiledRegion(graph, compile_region)
        )
        result.set_submodule(name, region)
    return result
