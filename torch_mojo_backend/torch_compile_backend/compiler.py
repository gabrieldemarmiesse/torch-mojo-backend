import datetime as dt
import functools
import time
import traceback
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, cast

import max.driver
import max.graph.value
import torch
from functorch.compile import make_boxed_func
from max import engine
from max.experimental.torch.torch import torch_dtype_to_max
from max.graph import DeviceRef, Graph, KernelLibrary, ops as max_ops
from torch._dynamo.backends.common import aot_autograd
from torch._subclasses.fake_tensor import unset_fake_temporarily

from torch_mojo_backend.aten_functions import (
    CURRENT_FX_NODE,
    DECOMPOSITION_TABLE,
    MAPPING_TORCH_ATEN_TO_MOJO,
    torch_device_to_max_device,
)
from torch_mojo_backend.flags import profiling_enabled, verbose_enabled
from torch_mojo_backend.mojo_device import dlpack as mojo_dlpack
from torch_mojo_backend.native import device_module
from torch_mojo_backend.torch_compile_backend import debug
from torch_mojo_backend.torch_compile_backend.utils import (
    get_accelerators,
    get_error_message,
    get_fully_qualified_name,
)


class MojoCompilerError(Exception):
    pass


@dataclass
class GlobalMaxObjects:
    session: engine.InferenceSession
    kernel_library: KernelLibrary


_global_max_objects: GlobalMaxObjects | None = None

paths_to_mojo_kernels = [Path(__file__).parent.parent / "mojo_kernels"]


def global_max_objects() -> GlobalMaxObjects:
    global _global_max_objects
    if _global_max_objects is None:
        kernel_library = KernelLibrary()
        kernel_library.load_paths(paths_to_mojo_kernels)
        session = engine.InferenceSession(devices=list(get_accelerators()))
        debug.set_print_options(session)

        _global_max_objects = GlobalMaxObjects(
            session=session, kernel_library=kernel_library
        )
    return _global_max_objects


def gather_stats_on_graph(gm: torch.fx.GraphModule):
    # count the number of times we see each function.
    # print and sort alphabetically.
    function_counts: dict[str, int] = {}
    for node in gm.graph.nodes:
        if node.op == "call_function" or node.op == "call_method":
            name = get_fully_qualified_name(node.target)
            function_counts.setdefault(name, 0)
            function_counts[name] += 1
    sorted_counts = sorted(function_counts.items(), key=lambda x: x[1], reverse=True)
    print("Function call counts:")
    for name, count in sorted_counts:
        print(f"{name}: {count}")


class TensorsBook:
    def __init__(self):
        self.tensors: dict[str, Any] = {}

    def __setitem__(self, name: str, tensor: object):
        self.tensors[name] = tensor

    def convert_to_max(self, something: object) -> object:
        if isinstance(something, torch.fx.Node):
            input_tensor = self.tensors[something.name]
            if isinstance(input_tensor, NotImplementedError):
                raise input_tensor
            return input_tensor
        elif isinstance(something, str):
            return something
        elif isinstance(something, int):
            return something
        elif isinstance(something, float):
            return something
        elif isinstance(something, slice):
            return slice(
                self.convert_to_max(something.start),
                self.convert_to_max(something.stop),
                self.convert_to_max(something.step),
            )
        elif isinstance(something, torch.fx.immutable_collections.immutable_list):
            return [self.convert_to_max(x) for x in something]
        elif isinstance(something, tuple):
            return tuple(self.convert_to_max(x) for x in something)
        elif isinstance(something, torch.device):
            return something
        elif isinstance(something, torch.dtype):
            return something
        elif isinstance(something, torch.layout):
            return something
        elif isinstance(something, torch.memory_format):
            return something
        elif isinstance(something, NotImplementedError):
            raise something
        elif something is None:
            return None
        elif something == ...:
            return ...
        elif isinstance(something, torch.nn.Module):
            return something
        elif isinstance(something, torch._ops.OpOverload):
            return something
        raise ValueError(f"Unsupported type when reading the graph: {type(something)}")


def fetch_attr(gm: torch.fx.GraphModule, target: str) -> object:
    """Fetch an attribute from the Module hierarchy of self.gm.
    Args:
        target (str): The fully-qualified name of the attribute to fetch
    """
    target_atoms = target.split(".")
    attr_itr = gm
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[: i + 1])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


class OutputBlueprintKind(Enum):
    NONE = 1
    TENSOR = 2
    DIM = 3


class _GraphFactory:
    def __init__(
        self,
        replace_inputs: dict[str, torch.Tensor] = {},
        force_device: DeviceRef | None = None,
    ):
        """Creates the MAX graph according to the input fx graph.

        Create a new instance for each new graph to be created.
        Args:
            replace_inputs (dict): A mapping from placeholder op to an actual tensor.
                With this information, we can remove graph inputs and use a constant instead.
                This is mainly useful to "freeze" the parameters of the model, because pytorch
                very often assumes that parameters are graph inputs. Max prefers constants. It's
                also a nicer UX for inference.
            force_device (DeviceRef | None): If provided, forces all graph inputs and constants
                to be on this device.
        """
        self.names_to_input_idx: dict[str, int] = {}
        self.shape_names_to_input_dim: dict[str, tuple[str, int]] = {}
        self.graph_inputs: list[max.graph.value.TensorType] = []
        self.graph: Graph | None = None
        # Whether the graph context manager is currently entered. MAX ties the
        # realization context to the active graph, so a graph that is entered
        # but never exited (e.g. graph construction raises) poisons every
        # subsequent eager realization with "Can't realize from a graph
        # context". We track this to guarantee the graph is always exited.
        self._graph_open = False
        self.tensor_book = TensorsBook()
        # Link the shape expressions (names) to the node names
        self.expression_to_node_name: dict[str, str] = {}
        self.replace_inputs = replace_inputs
        self.force_device = force_device

    def initialize_graph(self):
        if self.graph is not None:
            raise RuntimeError("Graph has already been initialized.")

        self.graph = Graph(
            "torch_mojo_backend",
            input_types=self.graph_inputs,
            kernel_library=global_max_objects().kernel_library,
        ).__enter__()
        self._graph_open = True
        # Let's fill the tensor book
        for tensor_name, idx in self.names_to_input_idx.items():
            self.tensor_book[tensor_name] = self.graph.inputs[idx]
        for shape_name, (tensor_name, dim_idx) in self.shape_names_to_input_dim.items():
            self.tensor_book[shape_name] = self.tensor_book.tensors[tensor_name].shape[
                dim_idx
            ]
        for input_name, tensor in self.replace_inputs.items():
            self.tensor_book[input_name] = max_ops.constant(
                tensor,
                dtype=torch_dtype_to_max(tensor.dtype),
                device=self.get_max_device(tensor),
            )

    def get_max_device(self, tensor: torch.Tensor) -> DeviceRef:
        if self.force_device is not None:
            return self.force_device
        # torch_device_to_max_device also handles the eager "mojo" device.
        return torch_device_to_max_device(tensor.device)

    def handle_placeholder(self, node: torch.fx.Node):
        if node.name in self.replace_inputs:
            # We short-circuit this input and use a constant instead.
            # We still have to place it in the graph inputs list because
            # at this point we don't have an active graph yet.
            # We'll register all the constants when initializing the graph.
            # TODO: add some validation in case the names in self.replace_inputs are not
            # in the graph.
            return

        if "example_value" in node.meta:
            example_value = node.meta["example_value"]
        elif "val" in node.meta:
            example_value = node.meta["val"]
        if isinstance(example_value, torch.SymInt):
            self.expression_to_node_name[example_value.node.expr.name] = node.name
        if isinstance(example_value, torch.Tensor | torch.nn.Parameter):
            shape = []
            for dim_idx, dim in enumerate(example_value.shape):
                if isinstance(dim, torch.SymInt):
                    shape.append(str(dim))
                    self.shape_names_to_input_dim[
                        self.expression_to_node_name[str(dim)]
                    ] = (node.name, dim_idx)
                elif isinstance(dim, int):
                    shape.append(dim)
                else:
                    raise TypeError(
                        f"Unsupported dimension type {type(dim)} for input {node.name} at index {dim_idx}"
                    )
            self.graph_inputs.append(
                max.graph.value.TensorType(
                    dtype=torch_dtype_to_max(example_value.dtype),
                    shape=shape,
                    device=self.get_max_device(example_value),
                )
            )
            self.names_to_input_idx[node.name] = len(self.graph_inputs) - 1

    def handle_call_function(self, node_idx: int, node: torch.fx.Node):
        func_args = [self.tensor_book.convert_to_max(x) for x in node.args]
        func_kwargs = {
            k: self.tensor_book.convert_to_max(v) for k, v in node.kwargs.items()
        }
        if isinstance(
            node.target, torch._higher_order_ops.auto_functionalize.AutoFunctionalizedV2
        ):
            # This is a torch-mojo-backend custom op. Let's add it to the graph.
            # (no graph break here)
            key = func_args[0]
            normalized_name = str(key).removesuffix(".default")
            func_to_execute = MAPPING_TORCH_ATEN_TO_MOJO[normalized_name]
            # without hidden keys
            input_tensors = [v for k, v in func_kwargs.items() if not k.startswith("_")]
            # AutoFunctionalizedV2's `_all_bases` kwarg is the list of mutated
            # tensor arguments (torch._higher_order_ops.auto_functionalize).
            all_bases = func_kwargs["_all_bases"]
            assert isinstance(all_bases, list)
            # We pray the gods that the order is correct here
            # because we only work with positional arguments
            self.tensor_book[node.name] = func_to_execute(*all_bases, *input_tensors)
            return
        key = node.target

        # TODO: refactor this
        if (
            isinstance(key, torch._ops.OpOverload)
            and key not in MAPPING_TORCH_ATEN_TO_MOJO
            and key.overloadpacket in MAPPING_TORCH_ATEN_TO_MOJO
        ):
            key = key.overloadpacket

        if key not in MAPPING_TORCH_ATEN_TO_MOJO:
            raise MojoCompilerError(
                "The aten function is not supported by the Mojo backend yet. "
                + get_error_message(node, node_idx, func_args, func_kwargs)
                + "You can try to write it yourself and insert it in the MAPPING_TORCH_ATEN_TO_MOJO dictionary."
            )
        try:
            mapping_func = MAPPING_TORCH_ATEN_TO_MOJO[key]
            token = CURRENT_FX_NODE.set(node)
            try:
                func_output = mapping_func(*func_args, **func_kwargs)
            finally:
                CURRENT_FX_NODE.reset(token)
        except Exception as e:
            raise MojoCompilerError(
                get_error_message(node, node_idx, func_args, func_kwargs)
                + "There was an error when executing the function. See the original error below. \n"
                f"{e}\n"
                f"{traceback.format_exc()}"
            )
        debug.add_prints(node_idx, str(node.target), func_output)

        self.tensor_book[node.name] = func_output

    def handle_get_attr(self, node: torch.fx.Node):
        owning_module = node.graph.owning_module
        assert owning_module is not None
        assert isinstance(node.target, str)
        attr_value = fetch_attr(owning_module, node.target)
        if isinstance(attr_value, torch.Tensor):
            # A tensor constant embedded in the graph (e.g. dynamo's
            # lift_fresh of a torch.tensor(...) created inside the traced
            # function). Bake it into the MAX graph as a constant. The
            # compiler runs under AOTAutograd's fake mode, which must not
            # intercept the reads of the real constant.
            device = self.get_max_device(attr_value)
            with unset_fake_temporarily():
                host = attr_value.detach().cpu()
                attr_value = max_ops.constant(
                    host, dtype=torch_dtype_to_max(host.dtype), device=device
                )
        self.tensor_book[node.name] = attr_value

    def handle_output(
        self, node: torch.fx.Node
    ) -> list[tuple[OutputBlueprintKind, int | None]]:
        """Handles the output node and returns the output blueprint.

        The blueprint indicates what the final output should look like, as
        opposed to what the MAX graph will return.
        The blueprint is the same size as the final output and
        NONE means that the output is None,
        TENSOR means that the output is a tensor (and the index in the MAX output list),
        DIM means that the output is a dimension (int) of a tensor (and the index in the MAX output list).
        Note that for DIM outputs, we'll need to convert the MAX tensor to an int at runtime,
        because MAX assumes that if your ouput is a Dim(), then you want a max tensor
        as output, not a simple python int.
        """
        # create_graph always calls initialize_graph (which sets self.graph)
        # before any node reaches handle_output.
        assert self.graph is not None
        # Elements are whatever convert_to_max returns for a graph-output
        # leaf: in practice always a previously-converted MAX value (the
        # Dim branch below is split out separately) -- narrowed with a
        # single cast at the `graph.output` call below rather than
        # scattering isinstance checks convert_to_max's broad return
        # doesn't support per-branch.
        output_tensors: list[object] = []

        # None outputs can be required. So we remember here if
        # we want an output tensor (and we reccord the tensor position)
        # or if we want None.
        output_blueprint: list[tuple[OutputBlueprintKind, int | None]] = []

        # An "output" fx node's args[0] is the traced function's actual
        # return value structure, always a list/tuple of Nodes/constants
        # (aot_autograd graphs always produce a flat tuple).
        output_args = node.args[0]
        assert isinstance(output_args, list | tuple)
        for x in output_args:
            converted = self.tensor_book.convert_to_max(x)
            if converted is None:
                output_blueprint.append((OutputBlueprintKind.NONE, None))
            elif isinstance(converted, max.graph.Dim):
                # position of the output tensor
                output_blueprint.append((OutputBlueprintKind.DIM, len(output_tensors)))
                output_tensors.append(converted)
            else:
                # position of the output tensor
                output_blueprint.append(
                    (OutputBlueprintKind.TENSOR, len(output_tensors))
                )
                output_tensors.append(converted)
        # Store the none indices for runtime handling
        self.graph.output(
            *cast("list[max.graph.value.TensorValueLike]", output_tensors)
        )
        self.graph.__exit__(None, None, None)
        self._graph_open = False
        return output_blueprint

    def create_graph(
        self, graph: torch.fx.Graph
    ) -> tuple[Graph, list[tuple[OutputBlueprintKind, int | None]]]:
        output_blueprint = None
        try:
            for node_idx, node in enumerate(graph.nodes):
                if node.op == "placeholder":
                    self.handle_placeholder(node)
                    continue

                if not self.graph:
                    self.initialize_graph()

                if node.op in ("call_function", "call_method"):
                    self.handle_call_function(node_idx, node)
                elif node.op == "get_attr":
                    self.handle_get_attr(node)
                elif node.op == "output":
                    output_blueprint = self.handle_output(node)
                else:
                    raise ValueError(f"Unsupported node type: {node.op}")
            if output_blueprint is None:
                raise ValueError(
                    "No output node found in the graph, this should never happen."
                )
        except BaseException:
            # If graph construction fails after the graph was entered, exit it so
            # the leaked graph realization context doesn't break later eager ops.
            if self._graph_open and self.graph is not None:
                self.graph.__exit__(None, None, None)
                self._graph_open = False
            raise
        # handle_output (which ran to set output_blueprint above) asserts
        # self.graph is set.
        assert self.graph is not None
        return self.graph, output_blueprint


def _graph_uses_mojo_device(
    gm: torch.fx.GraphModule, example_inputs: list[object]
) -> bool:
    """Whether this graph computes on the eager "mojo" device.

    Checked at compile time (inputs may be fake tensors; factory-only graphs
    have no tensor inputs, so node metas are scanned too) to decide how the
    MAX output buffers must be wrapped at runtime.
    """
    for t in example_inputs:
        if isinstance(t, torch.Tensor) and t.device.type == "mojo":
            return True
    for node in gm.graph.nodes:
        val = node.meta.get("val", node.meta.get("example_value"))
        if isinstance(val, torch.Tensor) and val.device.type == "mojo":
            return True
    return False


@functools.cache
def _mojo_accelerators() -> tuple[max.driver.Device, ...]:
    """The concrete MAX devices backing each `mojo:<index>`, in the same
    order (GPUs, then the MAX CPU device) the native backend's
    `device.mojo` assigns them -- see `get_accelerators()`."""
    return tuple(get_accelerators())


def _max_device_for_mojo(device: torch.device) -> max.driver.Device:
    """The concrete `max.driver.Device` a `mojo:<index>` torch device maps to."""
    accelerators = _mojo_accelerators()
    index = device.index if device.index is not None else 0
    if index >= len(accelerators):
        raise ValueError(f"Invalid mojo device index {index}")
    return accelerators[index]


def _mojo_index_for_max_device(device: max.driver.Device) -> int:
    """The inverse of `_max_device_for_mojo`: which `mojo:<index>` a MAX
    device (as reported by a MAX output buffer) corresponds to."""
    for index, accelerator in enumerate(_mojo_accelerators()):
        if accelerator.label == "cpu" and device.label == "cpu":
            return index
        if accelerator.label == device.label and accelerator.id == device.id:
            return index
    raise ValueError(f"MAX device {device} has no corresponding mojo index")


def _max_device_for_cuda(device: torch.device) -> max.driver.Device:
    """The concrete MAX accelerator a real (non-mojo) `cuda`/`hip` torch
    device maps to: the GPUs among `_mojo_accelerators()`, in order."""
    gpu_accelerators = [a for a in _mojo_accelerators() if a.label == "gpu"]
    index = device.index if device.index is not None else 0
    if index >= len(gpu_accelerators):
        raise RuntimeError(f"GPU index {index} not available in MAX")
    return gpu_accelerators[index]


def _mojo_tensor_from_buffer(buffer: max.driver.Buffer) -> torch.Tensor:
    """Zero-copy wrap of a MAX output buffer as a plain `mojo`-device tensor.

    The `mojo` device is a real (renamed) PrivateUse1 backend: torch's C++
    DLPack importer maps DLPack's kDLExtDev device-type code straight to
    `at::Device(DeviceType::PrivateUse1, index)` (aten/src/ATen/DLConvertor.cpp)
    regardless of the backend's Python-visible rename, so a capsule tagged
    (kDLExtDev, this mojo index) imports zero-copy through the public
    `torch.from_dlpack` (see `mojo_dlpack.make_capsule_privateuse1`). The
    capsule keeps `buffer` (a normal Python object) alive until torch frees
    the imported storage -- the same refcount-based lifetime a MAX buffer
    already has on its own.
    """
    index = _mojo_index_for_max_device(buffer.device)
    capsule = mojo_dlpack.make_capsule_privateuse1(
        buffer, buffer._data_ptr(), tuple(buffer.shape), buffer.dtype, index
    )
    return torch.from_dlpack(capsule)


def _dim_buffer_to_cpu_tensor(buffer: max.driver.Buffer) -> torch.Tensor:
    """A DIM output as a CPU torch tensor (works without a CUDA-enabled torch)."""
    if buffer.device.label != "cpu":
        buffer = buffer.to(max.driver.CPU())
    return torch.from_dlpack(buffer)


class BaseMaxCompiler:
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: list[object],
        mode: str | None = None,
    ):
        self.gm = gm
        self.mojo_outputs = _graph_uses_mojo_device(gm, example_inputs)
        if profiling_enabled():
            compiler_start = time.time_ns()
        if verbose_enabled():
            print(f"Graph has {len(gm.graph.nodes)} nodes.")
            gather_stats_on_graph(gm)
            gm.graph.print_tabular()

        graph, self.output_blueprint = _GraphFactory().create_graph(gm.graph)
        if verbose_enabled():
            print(graph)
        if profiling_enabled():
            graph_defined_time = time.time_ns()
        self.model = global_max_objects().session.load(graph)
        if profiling_enabled():
            compiling_done_time = time.time_ns()
            defining = dt.timedelta(
                microseconds=(graph_defined_time - compiler_start) / 1000
            )
            print(f"Defining the Max graph in {defining}")
            compiling = dt.timedelta(
                microseconds=(compiling_done_time - graph_defined_time) / 1000
            )
            print(f"Compiling the Max graph in {compiling}")

    def reconstruct_from_blueprint(
        self, max_ouptputs: list[torch.Tensor]
    ) -> list[torch.Tensor | int | float | None]:
        result: list[torch.Tensor | int | float | None] = []
        for kind, index in self.output_blueprint:
            if kind is OutputBlueprintKind.NONE:
                result.append(None)
            elif kind is OutputBlueprintKind.TENSOR:
                assert index is not None
                result.append(max_ouptputs[index])
            elif kind is OutputBlueprintKind.DIM:
                assert index is not None
                result.append(max_ouptputs[index].item())
        return result

    def __call__(self, *args: object) -> list[torch.Tensor | int | float | None]:
        # Detach tensors to avoid gradient tracking issues with DLpack
        if profiling_enabled():
            start_inference_time = time.time_ns()
        input_tensors = [
            _cached_buffer_for(x) for x in args if isinstance(x, torch.Tensor)
        ]
        outputs = self.model.execute(*input_tensors)
        if self.mojo_outputs:
            # The graph computes on the mojo device: adopt the MAX output
            # buffers zero-copy as eager mojo tensors (DIM outputs become
            # CPU tensors, `.item()`-ed in reconstruct_from_blueprint).
            dim_indices = {
                index
                for kind, index in self.output_blueprint
                if kind is OutputBlueprintKind.DIM
            }
            tensor_outputs = [
                _dim_buffer_to_cpu_tensor(x)
                if i in dim_indices
                else _mojo_tensor_from_buffer(x)
                for i, x in enumerate(outputs)
            ]
        else:
            tensor_outputs = [torch.from_dlpack(x) for x in outputs]

        debug.debug_graph_if_required(self.gm, args)

        result = self.reconstruct_from_blueprint(tensor_outputs)

        if profiling_enabled():
            end_inference_time = time.time_ns()
            inference_duration = dt.timedelta(
                microseconds=(end_inference_time - start_inference_time) / 1000
            )
            print(f"Running the Max graph in {inference_duration}")
        return result


# Cross-call Buffer cache. Graph inputs are dominated by parameters, which
# are the SAME tensor objects on every call of a compiled graph; converting
# each of them through DLPack every call costs ~10-20us per tensor. Cache
# the imported Buffer keyed by tensor identity, guarded by the data pointer
# (catches storage reallocation, e.g. `param.data = ...`), evicted when the
# tensor dies. Buffers alias the tensor memory, so in-place updates
# (optimizer steps) are seen without invalidation.
# Quoted: `weakref.finalize` is not subscriptable at runtime.
_buffer_cache: "dict[int, tuple[max.driver.Buffer, int, weakref.finalize[[int], torch.Tensor]]]" = {}


def _evict_buffer(tensor_id: int, /):
    _buffer_cache.pop(tensor_id, None)


def _cached_buffer_for(t: torch.Tensor) -> max.driver.Buffer:
    if not t.is_contiguous():
        # A MAX Buffer is dense row-major, and the graph input type only
        # carries a shape, so a strided input has to be materialized. The
        # copy is a fresh allocation on every call and is deliberately NOT
        # cached: the cache exists to alias input memory so that in-place
        # updates stay visible, which a frozen copy could not honor.
        return fast_from_dlpack(t.detach().contiguous())

    key = id(t)
    entry = _buffer_cache.get(key)
    if entry is not None:
        buffer, ptr, _finalizer = entry
        if ptr == t.data_ptr():
            return buffer
    buffer = fast_from_dlpack(t.detach())
    finalizer = weakref.finalize(t, _evict_buffer, key)
    _buffer_cache[key] = (buffer, t.data_ptr(), finalizer)
    return buffer


GraphValue = torch.Tensor | int | float | None


def boxed_func(
    gm: torch.fx.GraphModule, example_inputs: list[object], mode: str | None = None
) -> Callable[[list[GraphValue]], list[GraphValue]]:
    return make_boxed_func(BaseMaxCompiler(gm, example_inputs, mode).__call__)


class mojo_backend:
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: list[object]):
        self.func_to_execute = aot_autograd(
            fw_compiler=boxed_func, decompositions=DECOMPOSITION_TABLE
        )(gm, example_inputs)

    def __call__(self, *args: object) -> list[torch.Tensor | int | float | None]:
        result = self.func_to_execute(*args)
        if isinstance(result, tuple):
            return list(result)
        return result


def dummy_compiler(
    gm: torch.fx.GraphModule, example_inputs: list[object]
) -> Callable[[list[GraphValue]], object]:
    return make_boxed_func(gm.forward)  # returns whatever the graph returns


# Can be used to check if it's the fault of the max backend or not.
dummy_backend = aot_autograd(fw_compiler=dummy_compiler)


# Taken from torch.py in max.
# Torch `__dlpack__(stream=...)` has substantial overhead.
# - Manually retrieving and syncing the stream drops dlpack marshalling
#   from ~60us per tensor to ~15us per tensor.
# - Further optimizations are possible. Moving more of this behavior
#   into a single C++ ffi call can drop overhead to ~2us.
# - Generally users shouldn't be putting this marshalling into their
#   inner loop. Gains are much more substantial for larger graphs
#   which can take advantage of MAX's automatic kernel fusion.
def fast_from_dlpack(t: torch.Tensor) -> max.driver.Buffer:
    if t.device.type == "cuda":
        stream = torch.cuda.current_stream(t.device).cuda_stream
        # _from_dlpack wants a concrete driver Device, not the graph-building
        # DeviceRef torch_device_to_max_device returns.
        device = _max_device_for_cuda(t.device)
        data = t.__dlpack__()
        try:
            return max.driver.Buffer._from_dlpack(data, device, stream)
        except Exception:
            # This approach fails when passing the tensor across threads.
            # Fall back to letting torch slowly sync streams.
            return max.driver.Buffer.from_dlpack(t)
    if t.device.type == "mojo":
        # `Tensor.__dlpack_device__` (torch/_tensor.py) checks the device
        # type string literal "privateuse1", not the *renamed* backend name
        # ("mojo") -- a real torch gap for a renamed PrivateUse1 backend, so
        # it (and the 1-arg `Buffer.from_dlpack`, which calls it first) raise
        # "Unknown device type mojo for Dlpack". `Tensor.__dlpack__()` itself
        # is unaffected (its C++ implementation keys off the DeviceType enum,
        # not the name), so build the capsule through it directly and hand
        # MAX the device/stream explicitly, exactly like the CUDA branch
        # above. No fallback: unlike CUDA, `Buffer.from_dlpack(t)` would hit
        # the very same broken device query.
        device = _max_device_for_mojo(t.device)
        if device.label == "cpu":
            # The explicit-device/stream `_from_dlpack` overload below is
            # GPU-only (it raises "unsupported device type in dlpack
            # implementation" for a CPU `device`), and the MAX-CPU mojo
            # device's memory is already host RAM, so there is nothing to
            # gain from a cleverer path: hand MAX a real torch CPU tensor
            # (a plain host-to-host copy, not the zero-copy exchange the
            # GPU case below gets).
            return max.driver.Buffer.from_dlpack(t.cpu())
        # the vendor handle of the current mojo stream (torch.Stream's own
        # native_handle exists only from torch 2.11)
        stream = device_module.stream_native_handle(
            torch.accelerator.current_stream(t.device)
        )
        data = t.__dlpack__()
        return max.driver.Buffer._from_dlpack(data, device, stream)
    return max.driver.Buffer.from_dlpack(t)
