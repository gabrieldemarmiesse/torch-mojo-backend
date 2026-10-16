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
from operator import getitem

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
    TensorType,
    TensorValue,
    ops,
)

from torch_mojo_backend import native
from torch_mojo_backend.distributed.process_group import (
    MojoProcessGroup,
    _nccl_dtype,
    _nccl_red_op,
)
from torch_mojo_backend.native import device_module
from torch_mojo_backend.types import MaxTensor

_functional = torch.ops._c10d_functional
_autograd = torch.ops._c10d_functional_autograd
WAIT_TENSOR = _functional.wait_tensor.default

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


def _mojo_device_indices(node: torch.fx.Node) -> set[int]:
    """The mojo device of every output (one per operand of a coalesced op)."""
    val = node.meta.get("val")
    indices = set()
    for v in val if isinstance(val, list | tuple) else [val]:
        if not isinstance(v, torch.Tensor) or v.device.type != "mojo":
            device = v.device if isinstance(v, torch.Tensor) else None
            raise CollectiveLoweringError(
                f"{node.target} on {device}: collectives in a compiled graph run "
                "on the mojo device only"
            )
        index = v.device.index
        indices.add(index if index is not None else device_module.current_device())
    return indices


@dataclass
class GroupHandles:
    """What the graph ops of one process group read at run time."""

    group: MojoProcessGroup
    device_index: int
    buffer: max.driver.Buffer
    address: int

    def check_alive(self):
        """The graph holds the group's address: refuse to run once the group
        was shut down (destroy_process_group) or rebuilt, instead of handing
        the ops freed memory."""
        if self.group._handle != self.address:
            raise RuntimeError(
                f"process group {self.group.getGroupName()!r} was destroyed after "
                "this graph was compiled; compile it again (torch._dynamo.reset())"
            )

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
        address = group._handle
        assert address is not None  # _ensure created the communicator
        start, wait = group._core.graph_entries
        handles = np.array(
            [address, device_index, start, wait, group._core.get_error], dtype=np.int64
        )
        return GroupHandles(
            group, device_index, max.driver.Buffer.from_numpy(handles), address
        )


@dataclass
class _Pending:
    """A started collective: `started` is written on the comm stream until
    the graph waits for it, `input` is read there until then; `result` is
    the waited value, once there is one (a second wait reuses it)."""

    started: TensorValue
    input: TensorValue
    handles: TensorValue
    token: BufferValue
    result: TensorValue | None = None


class GraphCollectives:
    """The collectives of one FX graph. `scan` runs before the MAX graph
    exists (its inputs are fixed at creation), `bind` once the handle inputs
    are values, then `lower` for every collective node in graph order."""

    def __init__(self, nodes: list[torch.fx.Node]):
        # One device per graph: MAX chains in-place ops per device, so the
        # program order every rank must follow holds within one device only.
        devices = set().union(*(_mojo_device_indices(n) for n in nodes))
        if len(devices) > 1:
            raise CollectiveLoweringError(
                f"collectives on mojo devices {sorted(devices)} in one compiled "
                "graph: a graph's collectives must all run on one device"
            )
        (device,) = devices
        self.handles: dict[str, GroupHandles] = {}
        for node in nodes:
            name = _group_name(node)
            if name not in self.handles:
                self.handles[name] = GroupHandles.resolve(name, device)
        self._values: dict[str, TensorValue] = {}
        self._token: BufferValue | None = None
        self._started: list[_Pending] = []

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
        for h in self.handles.values():
            h.check_alive()
        return [h.buffer for h in self.handles.values()]

    def bind(self, handles: Sequence[TensorValue]):
        self._values = dict(zip(self.handles, handles, strict=True))

    def token(self, device: DeviceRef) -> BufferValue:
        """The buffer that chains every start and wait on the device chain."""
        if self._token is None:
            self._token = ops.buffer_create(BufferType(DType.int8, [1], device))
        return self._token

    def handles_value(self, node: torch.fx.Node) -> TensorValue:
        return self._values[_group_name(node)]

    def start(
        self, node: torch.fx.Node, args: list[object]
    ) -> _Pending | list[_Pending]:
        return _START_OPS[node.target](self, node, *args)

    def started(self, pending: _Pending):
        self._started.append(pending)

    def synchronize(self) -> list[str]:
        """Host-wait for every collective the graph's process groups issued:
        after a failed execute, before MAX can hand out memory a collective
        it started still reads or writes. Returns the groups that could not
        be drained (their memory is not safe to reuse)."""
        failed = []
        for name, h in self.handles.items():
            if h.group._handle != h.address:
                continue
            if h.group._core.synchronize_comm(h.group._handle, h.device_index):
                failed.append(f"{name!r}: {native.last_error()}")
        return failed

    def finish(self):
        """Wait for every collective the graph started and nothing waited
        for (dynamo's dead-code pass drops the wait of an unused result,
        never the collective): its output and input must stay allocated
        until the transfer is over, and every start needs its wait."""
        for pending in self._started:
            wait(pending)


def _start(
    collectives: GraphCollectives,
    node: torch.fx.Node,
    x: TensorValue,
    dtype: torch.dtype,
    shape: list[Dim],
    kind: int,
    arg: int,
) -> _Pending:
    handles = collectives.handles_value(node)
    token = collectives.token(x.device)
    (started,) = ops.inplace_custom(
        "c10d_start",
        x.device,
        [token, x, handles],
        out_types=[TensorType(x.dtype, shape, x.device)],
        parameters={"kind": kind, "nccl_dtype": _nccl_dtype(dtype), "arg": arg},
    )
    pending = _Pending(started.tensor, x, handles, token)
    collectives.started(pending)
    return pending


def wait(value: object) -> object:
    """The result of a started collective, once the graph stream waited
    for it; anything else as is. A `wait_tensor` on a graph input (an
    eager collective's result passed in) is waited for before the graph
    runs: `BaseMaxCompiler.__call__`, `waited_inputs`."""
    if isinstance(value, list):
        return [wait(v) for v in value]
    if not isinstance(value, _Pending):
        return value
    if value.result is None:
        (copied,) = ops.inplace_custom(
            "c10d_wait",
            value.input.device,
            [value.token, value.started, value.input, value.handles],
            out_types=[value.started.type],
        )
        value.result = copied.tensor
    return value.result


def _is_view(node: torch.fx.Node) -> bool:
    target = node.target
    return target == torch.ops.aten._unsafe_view.default or (
        isinstance(target, torch._ops.OpOverload) and target.is_view
    )


def _wait_source(node: object) -> torch.fx.Node | bool:
    """What a `wait_tensor` on `node` waits for: through views, selections
    (`getitem` of a coalesced collective or of a split) and earlier waits,
    the graph input it reads (a placeholder), True for a collective of the
    graph, False for anything else."""
    while isinstance(node, torch.fx.Node):
        if node.op == "placeholder":
            return node
        if node.op != "call_function" or not node.args:
            return False
        if node.target in _START_OPS:
            return True
        if not (node.target in (getitem, WAIT_TENSOR) or _is_view(node)):
            return False
        node = node.args[0]
    return False


def waited_inputs(graph: torch.fx.Graph) -> list[int]:
    """Positions of the graph inputs a `wait_tensor` reads, directly or
    through views: eager collectives' results (an AsyncCollectiveTensor
    passed to compiled code traces to one), which the caller waits for
    before the graph runs. A wait on anything that is neither cannot be
    honored, and is refused."""
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    waited = set()
    for n in graph.nodes:
        if n.op != "call_function" or n.target != WAIT_TENSOR:
            continue
        source = _wait_source(n.args[0])
        if source is False:
            raise CollectiveLoweringError(
                f"wait_tensor on {n.args[0]}, which is neither a collective of "
                "this graph nor a graph input (or a view of one)"
            )
        if isinstance(source, torch.fx.Node):
            waited.add(source)
    return [i for i, p in enumerate(placeholders) if p in waited]


def _dtypes(node: torch.fx.Node) -> list[torch.dtype]:
    """The torch dtype of each output (one per input of a coalesced op,
    whose inputs may differ)."""
    val = node.meta["val"]
    vals = val if isinstance(val, list | tuple) else [val]
    assert all(isinstance(v, torch.Tensor) for v in vals)
    return [v.dtype for v in vals]


def _red_op(dtype: torch.dtype, reduce_op: str) -> int:
    op = _REDUCE_OPS.get(reduce_op.lower())
    if op is None:
        raise CollectiveLoweringError(
            f"reduce op {reduce_op!r} is not supported in a compiled graph"
        )
    return _nccl_red_op(op, dtype)


# One collective each: (collectives, node, input, its torch dtype, the op's
# remaining arguments, group name last).


def _all_reduce(
    c: GraphCollectives,
    node: torch.fx.Node,
    input: MaxTensor,
    dtype: torch.dtype,
    reduce_op: str,
    _: str,
) -> _Pending:
    x = TensorValue(input)
    op = _red_op(dtype, reduce_op)
    return _start(c, node, x, dtype, list(x.shape), _ALLREDUCE, op)


def _all_gather(
    c: GraphCollectives,
    node: torch.fx.Node,
    input: MaxTensor,
    dtype: torch.dtype,
    group_size: int,
    _: str,
) -> _Pending:
    x = TensorValue(input)
    # a 0-d input gathers into [group_size]
    shape = [x.shape[0] * group_size, *x.shape[1:]] if x.rank else [Dim(group_size)]
    return _start(c, node, x, dtype, shape, _ALLGATHER, 0)


def _reduce_scatter(
    c: GraphCollectives,
    node: torch.fx.Node,
    input: MaxTensor,
    dtype: torch.dtype,
    reduce_op: str,
    group_size: int,
    _: str,
) -> _Pending:
    x = TensorValue(input)
    shape = [x.shape[0] // group_size, *x.shape[1:]]
    op = _red_op(dtype, reduce_op)
    return _start(c, node, x, dtype, shape, _REDUCE_SCATTER, op)


def _broadcast(
    c: GraphCollectives,
    node: torch.fx.Node,
    input: MaxTensor,
    dtype: torch.dtype,
    src: int,
    _: str,
) -> _Pending:
    x = TensorValue(input)
    return _start(c, node, x, dtype, list(x.shape), _BROADCAST, src)


def _single(one: Callable[..., _Pending]) -> Callable[..., _Pending]:
    def lower(
        c: GraphCollectives, node: torch.fx.Node, input: MaxTensor, *rest: object
    ) -> _Pending:
        return one(c, node, input, _dtypes(node)[0], *rest)

    return lower


def _coalesced(one: Callable[..., _Pending]) -> Callable[..., list[_Pending]]:
    def lower(
        c: GraphCollectives, node: torch.fx.Node, inputs: list[MaxTensor], *rest: object
    ) -> list[_Pending]:
        dtypes = _dtypes(node)
        return [one(c, node, x, d, *rest) for x, d in zip(inputs, dtypes, strict=True)]

    return lower


_START_OPS: dict[object, Callable[..., _Pending | list[_Pending]]] = {
    _functional.all_reduce.default: _single(_all_reduce),
    _functional.all_reduce_coalesced.default: _coalesced(_all_reduce),
    _functional.all_gather_into_tensor.default: _single(_all_gather),
    _functional.all_gather_into_tensor_coalesced.default: _coalesced(_all_gather),
    _functional.reduce_scatter_tensor.default: _single(_reduce_scatter),
    _functional.reduce_scatter_tensor_coalesced.default: _coalesced(_reduce_scatter),
    _functional.broadcast.default: _single(_broadcast),
    _autograd.all_gather_into_tensor.default: _single(_all_gather),
    _autograd.reduce_scatter_tensor.default: _single(_reduce_scatter),
}

_UNSUPPORTED = {
    op: "all_to_all_single is not supported in a graph compiled for the mojo device yet"
    for op in (
        _functional.all_to_all_single.default,
        _autograd.all_to_all_single.default,
    )
}

START_OPS = frozenset(_START_OPS)
