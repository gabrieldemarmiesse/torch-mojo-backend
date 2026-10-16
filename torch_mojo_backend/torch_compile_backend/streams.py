"""Keep stream/event orchestration outside AOTAutograd's tensor tracing.

Dynamo represents stream contexts as node metadata and stream/event methods
as ordinary Python calls. AOT executes those calls while tracing and drops
them from its tensor graph. Partition *before* AOT, retaining the calls in
the runtime wrapper and compiling each contiguous tensor region separately.
This is backend partitioning, so it also works with ``fullgraph=True``.
"""

from collections.abc import Callable
from contextlib import ExitStack

import torch
from torch.fx.passes.split_module import split_module
from torch.utils._pytree import tree_flatten, tree_unflatten

try:
    from torch._dynamo import graph_bytecode_inputs
except ImportError:  # PyTorch before the generic stream tracing API.
    graph_bytecode_inputs = None

RegionCompiler = Callable[[torch.fx.GraphModule, list[object]], Callable[..., object]]


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
        if not self.indices:
            return self._run(args)
        # Resolve on every invocation: streams constructed inside the user's
        # function are recreated by Dynamo's bytecode preamble on every call.
        if graph_bytecode_inputs is None:
            raise RuntimeError("Compiled stream contexts require PyTorch 2.11 or newer")
        with ExitStack() as contexts:
            for index in self.indices:
                stream = graph_bytecode_inputs.get_external_object_by_index(index)
                if not isinstance(stream, torch.Stream):
                    raise RuntimeError(
                        "Compiled stream index no longer names a torch.Stream"
                    )
                # Generic Stream.__enter__ rejects CUDA after PrivateUse1
                # registration. The CUDA-specific context accepts both classes.
                context = (
                    torch.cuda.stream(stream)
                    if stream.device.type == "cuda"
                    else stream
                )
                contexts.enter_context(context)
            return self._run(args)


class _CompiledRegion(_StreamRegion):
    def __init__(
        self,
        graph: torch.fx.GraphModule,
        compile_region: RegionCompiler,
        indices: tuple[int, ...],
    ):
        super().__init__(graph, indices)
        output = next(node for node in graph.graph.nodes if node.op == "output")
        leaves, self.output_spec = tree_flatten(output.args[0])
        output.args = (tuple(leaves),)
        graph.recompile()
        for node in graph.graph.nodes:
            # Placement now belongs to this region's runtime context. AOT's
            # backward stream pass must not resolve Dynamo's temporary object
            # table after the forward invocation has ended.
            node.meta = dict(node.meta)
            if "custom" in node.meta:
                node.meta["custom"] = dict(node.meta["custom"])
                node.meta["custom"].pop("stream", None)
                node.meta["custom"].pop("torch_mojo_streams", None)
        # Use Dynamo's symbolic examples now, not concrete runtime arguments
        # on the first call: the latter silently specializes dynamic shapes.
        examples = [
            node.meta.get("example_value", node.meta.get("val"))
            for node in graph.graph.nodes
            if node.op == "placeholder"
        ]
        self.compiled = compile_region(graph, examples)

    def _run(self, args: tuple[object, ...]) -> object:
        result = self.compiled(*args)
        assert isinstance(result, list | tuple)
        return tree_unflatten(list(result), self.output_spec)


def compile_stream_graph(
    gm: torch.fx.GraphModule, compile_region: RegionCompiler
) -> torch.fx.GraphModule:
    partitions = {}
    regions = {}
    partition = -1
    previous = None
    for node in gm.graph.nodes:
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        control = _is_orchestration(node)
        key = (control, stream_indices(node))
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
            else _CompiledRegion(graph, compile_region, indices)
        )
        result.set_submodule(name, region)
    return result
