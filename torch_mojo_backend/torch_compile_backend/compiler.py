import time
import traceback
import weakref
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import max.driver
import max.graph.value
import torch
from functorch.compile import make_boxed_func
from max import engine
from max.experimental.torch.torch import torch_dtype_to_max
from max.graph import DeviceRef, Graph, KernelLibrary
from max.graph import ops as max_ops
from torch._dynamo.backends.common import aot_autograd

from torch_mojo_backend.aten_functions import (
    CURRENT_FX_NODE,
    DECOMPOSITION_TABLE,
    torch_device_to_max_device,
)
from torch_mojo_backend.flags import profiling_enabled, verbose_enabled
from torch_mojo_backend.torch_compile_backend import debug
from torch_mojo_backend.torch_compile_backend.utils import (
    get_error_message,
    get_fully_qualified_name,
)

from ..aten_functions import MAPPING_TORCH_ATEN_TO_MOJO
from .utils import get_accelerators


class MojoCompilerError(Exception):
    pass


import datetime as dt


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
    function_counts = {}
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

    def __setitem__(self, name: str, tensor):
        self.tensors[name] = tensor

    def convert_to_max(self, something):
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


def fetch_attr(gm: torch.fx.GraphModule, target: str):
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
            # We pray the gods that the order is correct here
            # because we only work with positional arguments
            self.tensor_book[node.name] = func_to_execute(
                *func_kwargs["_all_bases"], *input_tensors
            )
            return
        key = node.target

        # TODO: refactor this
        if (
            key not in MAPPING_TORCH_ATEN_TO_MOJO
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
        attr_value = fetch_attr(node.graph.owning_module, node.target)
        if isinstance(attr_value, torch.Tensor):
            # A tensor constant embedded in the graph (e.g. dynamo's
            # lift_fresh of a torch.tensor(...) created inside the traced
            # function). Bake it into the MAX graph as a constant. The
            # compiler runs under AOTAutograd's fake mode, which must not
            # intercept the reads of the real constant.
            from torch._subclasses.fake_tensor import unset_fake_temporarily

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
        output_tensors = []

        # None outputs can be required. So we remember here if
        # we want an output tensor (and we reccord the tensor position)
        # or if we want None.
        output_blueprint: list[tuple[OutputBlueprintKind, int | None]] = []

        for x in node.args[0]:
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
        self.graph.output(*output_tensors)
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
        return self.graph, output_blueprint


def _graph_uses_mojo_device(gm: torch.fx.GraphModule, example_inputs: list) -> bool:
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


def _mojo_tensor_from_buffer(buffer: max.driver.Buffer) -> torch.Tensor:
    """Zero-copy wrap of a MAX output buffer as an eager mojo tensor.

    The buffer itself becomes the wrapper's ownership token (`_holder`):
    MAX buffers release their device memory when the last Python reference
    drops, exactly like the eager TensorHolder.
    """
    from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
        TorchMojoTensor,
        _row_major_strides,
    )

    shape = tuple(buffer.shape)
    return TorchMojoTensor._make(
        buffer,
        buffer._data_ptr(),
        shape,
        _row_major_strides(shape),
        0,
        buffer.dtype,
        buffer.device,
        contiguous=True,
    )


def _dim_buffer_to_cpu_tensor(buffer: max.driver.Buffer) -> torch.Tensor:
    """A DIM output as a CPU torch tensor (works without a CUDA-enabled torch)."""
    if buffer.device.label != "cpu":
        buffer = buffer.to(max.driver.CPU())
    return torch.from_dlpack(buffer)


class BaseMaxCompiler:
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: list, mode=None):
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
        result = []
        for kind, index in self.output_blueprint:
            if kind is OutputBlueprintKind.NONE:
                result.append(None)
            elif kind is OutputBlueprintKind.TENSOR:
                result.append(max_ouptputs[index])
            elif kind is OutputBlueprintKind.DIM:
                result.append(max_ouptputs[index].item())
        return result

    def __call__(self, *args) -> list[torch.Tensor | int | float | None]:
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
_buffer_cache: dict[int, tuple] = {}


def _evict_buffer(tensor_id: int) -> None:
    _buffer_cache.pop(tensor_id, None)


def _data_ptr_of(t: torch.Tensor) -> int:
    ptr = getattr(t, "_ptr", None)  # TorchMojoTensor
    if ptr is not None:
        return ptr
    return t.data_ptr()


def _cached_buffer_for(t: torch.Tensor):
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
        if ptr == _data_ptr_of(t):
            return buffer
    buffer = fast_from_dlpack(t.detach())
    finalizer = weakref.finalize(t, _evict_buffer, key)
    _buffer_cache[key] = (buffer, _data_ptr_of(t), finalizer)
    return buffer


def boxed_func(*args, **kwargs):
    return make_boxed_func(BaseMaxCompiler(*args, **kwargs).__call__)


class mojo_backend:
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: list):
        self.func_to_execute = aot_autograd(
            fw_compiler=boxed_func, decompositions=DECOMPOSITION_TABLE
        )(gm, example_inputs)

    def __call__(self, *args) -> list[torch.Tensor | int | float | None]:
        result = self.func_to_execute(*args)
        if isinstance(result, tuple):
            return list(result)
        return result


def dummy_compiler(gm: torch.fx.GraphModule, example_inputs: list):
    return make_boxed_func(gm.forward)


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
        device = torch_device_to_max_device(t.device)
        data = t.__dlpack__()
        try:
            return max.driver.Buffer._from_dlpack(data, device, stream)
        except Exception:
            # This approach fails when passing the tensor across threads.
            # Fall back to letting torch slowly sync streams.
            return max.driver.Buffer.from_dlpack(t)
    return max.driver.Buffer.from_dlpack(t)
