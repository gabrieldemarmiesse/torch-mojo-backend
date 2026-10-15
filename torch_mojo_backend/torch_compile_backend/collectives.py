"""`_c10d_functional` collectives inside a MAX graph.

Dynamo traces `torch.distributed` calls in compiled code (functional
collectives, `dist.all_reduce` and friends, DTensor redistributions) as a
collective node and a later `wait_tensor` node. Both become custom ops of
`tmb/graph/collectives.mojo`, which run the collective through the rank's
mojo process group -- MojoCCL, NCCL or RCCL, whichever eager mode uses --
on its comm stream, so compute between the start and the wait overlaps it.

The ops find the process group through a host int64 graph input per group
(`GroupHandles.buffer`), not through constants: addresses differ from one
process to the next and the graph must not.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import max.driver
import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    StaticDim,
    SymbolicDim,
    TensorType,
    TensorValue,
    ops,
)

from torch_mojo_backend.distributed.process_group import (
    MojoProcessGroup,
    _nccl_dtype,
    _nccl_red_op,
)
from torch_mojo_backend.native import device_module
from torch_mojo_backend.types import MaxTensor

_functional = torch.ops._c10d_functional
_autograd = torch.ops._c10d_functional_autograd

# tmb/backend/pg.mojo GRAPH_* kinds
_ALLREDUCE = 0
_ALLGATHER = 1
_REDUCE_SCATTER = 2
_BROADCAST = 3

_REDUCE_OPS = {
    "sum": dist.ReduceOp.SUM,
    "avg": dist.ReduceOp.AVG,
    "product": dist.ReduceOp.PRODUCT,
    "min": dist.ReduceOp.MIN,
    "max": dist.ReduceOp.MAX,
}


class CollectiveLoweringError(Exception):
    pass


def _group_name(node: torch.fx.Node) -> str:
    """Every collective's last positional argument is its group name."""
    name = node.args[-1]
    if not isinstance(name, str):
        raise CollectiveLoweringError(f"{node.target}: no group name in {node.args}")
    return name


def _mojo_device_index(node: torch.fx.Node) -> int:
    val = node.meta.get("val")
    if isinstance(val, list | tuple):
        val = val[0]
    if not isinstance(val, torch.Tensor) or val.device.type != "mojo":
        device = val.device if isinstance(val, torch.Tensor) else None
        raise CollectiveLoweringError(
            f"{node.target} on {device}: collectives in a compiled graph run "
            "on the mojo device only"
        )
    index = val.device.index
    return index if index is not None else device_module.current_device()


@dataclass
class GroupHandles:
    """What the graph ops of one process group read at run time."""

    group: MojoProcessGroup
    device_index: int
    buffer: max.driver.Buffer

    @staticmethod
    def resolve(group_name: str, device_index: int) -> GroupHandles:
        group = c10d._resolve_process_group(c10d.GroupName(group_name))
        if not isinstance(group, MojoProcessGroup):
            group = group._get_backend(torch.device("mojo", device_index))
        if not isinstance(group, MojoProcessGroup):
            raise CollectiveLoweringError(
                f"process group {group_name!r} is {type(group).__name__}: "
                "collectives in a graph compiled for the mojo device need the "
                'mojo process group (dist.init_process_group("mojo") or a '
                "device mesh of the mojo device)"
            )
        # Collective, like every communicator creation: every rank compiles
        # the same graph at the same point of the program.
        group._ensure(device_index)
        start, wait = group._core.graph_entries
        handles = np.array(
            [group._handle, device_index, start, wait, group._core.get_error],
            dtype=np.int64,
        )
        return GroupHandles(group, device_index, max.driver.Buffer.from_numpy(handles))


@dataclass
class _Pending:
    """A started collective, written on the comm stream until the graph
    waits for it: into `buffer` (loaded, then viewed as `shape`, after the
    wait), or -- when MAX cannot load a buffer of that shape -- into
    `started`, which the wait copies. `input` is what the wait keeps alive."""

    input: TensorValue
    handles: TensorValue
    shape: list[Dim]
    buffer: BufferValue | None = None
    started: TensorValue | None = None
    token: BufferValue | None = None


class GraphCollectives:
    """The collectives of one FX graph. `scan` runs before the MAX graph
    exists (its inputs are fixed at creation), `bind` once the handle inputs
    are values, then `lower` for every collective node in graph order."""

    def __init__(self, nodes: list[torch.fx.Node]):
        self.handles: dict[str, GroupHandles] = {}
        for node in nodes:
            name = _group_name(node)
            if name not in self.handles:
                self.handles[name] = GroupHandles.resolve(
                    name, _mojo_device_index(node)
                )
        self._values: dict[str, TensorValue] = {}
        self._input_symbols: set[str] = set()
        self._token: BufferValue | None = None

    @staticmethod
    def scan(graph: torch.fx.Graph) -> GraphCollectives | None:
        calls = [n for n in graph.nodes if n.op == "call_function"]
        for node in calls:
            if node.target in _UNSUPPORTED:
                raise CollectiveLoweringError(_UNSUPPORTED[node.target])
        nodes = [n for n in calls if n.target in _START_OPS]
        return GraphCollectives(nodes) if nodes else None

    def input_types(self) -> list[TensorType]:
        return [TensorType(DType.int64, [5], DeviceRef.CPU()) for _ in self.handles]

    def runtime_inputs(self) -> list[max.driver.Buffer]:
        return [h.buffer for h in self.handles.values()]

    def bind(self, inputs: Sequence[TensorValue], handles: Sequence[TensorValue]):
        self._values = dict(zip(self.handles, handles, strict=True))
        self._input_symbols = {
            str(d) for x in inputs for d in x.shape if isinstance(d, SymbolicDim)
        }

    def loadable(self, shape: Sequence[Dim]) -> bool:
        """Whether MAX can load a graph buffer of `shape`: every dim static
        or a graph input's symbol."""
        return all(
            isinstance(d, StaticDim)
            or (isinstance(d, SymbolicDim) and str(d) in self._input_symbols)
            for d in shape
        )

    def token(self, device: DeviceRef) -> BufferValue:
        """The buffer that chains the ops of the `_out`/`_copy` variants."""
        if self._token is None:
            self._token = ops.buffer_create(BufferType(DType.int8, [1], device))
        return self._token

    def handles_value(self, node: torch.fx.Node) -> TensorValue:
        return self._values[_group_name(node)]

    def lower(self, node: torch.fx.Node, args: list[object]) -> object:
        if node.target == _functional.wait_tensor.default:
            return wait(args[0])
        return _START_OPS[node.target](self, node, *args)


def _start(
    collectives: GraphCollectives,
    node: torch.fx.Node,
    x: TensorValue,
    shape: list[Dim],
    kind: int,
    arg: int,
    buffer_shape: list[Dim] | None = None,
) -> _Pending:
    """`buffer_shape` is the same memory as `shape`, spelled with loadable
    dims (all-gather's `[group, n, ...]` for `[group * n, ...]`)."""
    handles = collectives.handles_value(node)
    parameters: dict[str, bool | int | str | DType] = {
        "kind": kind,
        "nccl_dtype": _nccl_dtype(_torch_dtype(node)),
        "arg": arg,
    }
    buffer_shape = buffer_shape or shape
    if collectives.loadable(buffer_shape):
        buffer = ops.buffer_create(BufferType(x.dtype, buffer_shape, x.device))
        ops.inplace_custom(
            "c10d_start", x.device, [buffer, x, handles], parameters=parameters
        )
        return _Pending(x, handles, shape, buffer=buffer)
    token = collectives.token(x.device)
    (started,) = ops.inplace_custom(
        "c10d_start_out",
        x.device,
        [token, x, handles],
        out_types=[TensorType(x.dtype, shape, x.device)],
        parameters=parameters,
    )
    return _Pending(x, handles, shape, started=started.tensor, token=token)


def wait(value: object) -> object:
    """The result of a started collective, once the graph stream waited
    for it; anything else (an already-waited value) as is."""
    if isinstance(value, list):
        return [wait(v) for v in value]
    if not isinstance(value, _Pending):
        return value
    device = value.input.device
    if value.buffer is not None:
        ops.inplace_custom(
            "c10d_wait", device, [value.buffer, value.input, value.handles]
        )
        loaded = ops.buffer_load(value.buffer)
        return loaded if loaded.shape == value.shape else loaded.reshape(value.shape)
    assert value.started is not None and value.token is not None
    (result,) = ops.inplace_custom(
        "c10d_wait_copy",
        device,
        [value.token, value.started, value.input, value.handles],
        out_types=[value.started.type],
    )
    return result.tensor


def _torch_dtype(node: torch.fx.Node) -> torch.dtype:
    val = node.meta["val"]
    if isinstance(val, list | tuple):
        val = val[0]
    assert isinstance(val, torch.Tensor)
    return val.dtype


def _red_op(node: torch.fx.Node, reduce_op: str) -> int:
    op = _REDUCE_OPS.get(reduce_op.lower())
    if op is None:
        raise CollectiveLoweringError(
            f"reduce op {reduce_op!r} is not supported in a compiled graph"
        )
    return _nccl_red_op(op, _torch_dtype(node))


def _all_reduce(
    c: GraphCollectives, node: torch.fx.Node, input: MaxTensor, reduce_op: str, _: str
) -> _Pending:
    x = TensorValue(input)
    return _start(c, node, x, list(x.shape), _ALLREDUCE, _red_op(node, reduce_op))


def _all_gather(
    c: GraphCollectives, node: torch.fx.Node, input: MaxTensor, group_size: int, _: str
) -> _Pending:
    x = TensorValue(input)
    shape = [x.shape[0] * group_size, *x.shape[1:]]
    return _start(c, node, x, shape, _ALLGATHER, 0, [Dim(group_size), *x.shape])


def _reduce_scatter(
    c: GraphCollectives,
    node: torch.fx.Node,
    input: MaxTensor,
    reduce_op: str,
    group_size: int,
    _: str,
) -> _Pending:
    x = TensorValue(input)
    shape = [x.shape[0] // group_size, *x.shape[1:]]
    return _start(c, node, x, shape, _REDUCE_SCATTER, _red_op(node, reduce_op))


def _broadcast(
    c: GraphCollectives, node: torch.fx.Node, input: MaxTensor, src: int, _: str
) -> _Pending:
    x = TensorValue(input)
    return _start(c, node, x, list(x.shape), _BROADCAST, src)


def _coalesced(one: Callable[..., _Pending]) -> Callable[..., list[_Pending]]:
    def lower(
        c: GraphCollectives, node: torch.fx.Node, inputs: list[MaxTensor], *rest: object
    ) -> list[_Pending]:
        return [one(c, node, x, *rest) for x in inputs]

    return lower


_START_OPS: dict[object, Callable[..., object]] = {
    _functional.all_reduce.default: _all_reduce,
    _functional.all_reduce_coalesced.default: _coalesced(_all_reduce),
    _functional.all_gather_into_tensor.default: _all_gather,
    _functional.all_gather_into_tensor_coalesced.default: _coalesced(_all_gather),
    _functional.reduce_scatter_tensor.default: _reduce_scatter,
    _functional.reduce_scatter_tensor_coalesced.default: _coalesced(_reduce_scatter),
    _functional.broadcast.default: _broadcast,
    _autograd.all_gather_into_tensor.default: _all_gather,
    _autograd.reduce_scatter_tensor.default: _reduce_scatter,
}

_UNSUPPORTED = {
    op: "all_to_all_single is not supported in a graph compiled for the mojo device yet"
    for op in (
        _functional.all_to_all_single.default,
        _autograd.all_to_all_single.default,
    )
}

COLLECTIVE_OPS = frozenset({*_START_OPS, _functional.wait_tensor.default})
