"""`_c10d_functional` collectives as MAX graph custom ops.

A traced collective is a start and a `wait_tensor`. `c10d_start` enqueues
it, through the process group of the native backend (`tmb/backend/pg.mojo`,
`tmb_pg_graph_start`), on the communicator's comm stream after the graph
stream's work so far, writing into its own output; `c10d_wait` makes the
graph stream wait for it and copies that output into the result its
consumers read. The wait reads the start's output and the collective's
input, so MAX keeps both allocated until the transfer is over, and the
copy is what orders every consumer after the wait: a MAX op cannot hand an
input back as its output. In-graph buffers would avoid the copy, but MAX
26.6 gives two live `buffer_create` buffers of one type the same memory
(measured). Both ops are chained through `token`, a buffer nothing reads,
so MAX keeps them in program order on the device chain -- every rank
issues its collectives in the same order -- while the compute between a
start and its wait overlaps the transfer.

The backend is another library with its own globals: these ops reach it
only through `handles`, a host int64 graph input of
`[process group, mojo device index, graph_start, graph_wait, tmb_get_error]`
(addresses, fed at every execution so the graph itself stays the same in
every process).
"""
import extensibility as compiler
from extensibility import InputTensor, OutputTensor
from extensibility import _MutableInputTensor as MutableInputTensor
from max.gpu.host import DeviceBuffer, DeviceContext

comptime StartFn = def(
    Int, Int32, Int32, Int, Int, Int, Int32, Int32, Int
) thin abi("C") -> Int32
comptime WaitFn = def(Int, Int32, Int, Int) thin abi("C") -> Int32
comptime ErrorFn = def() thin abi("C") -> Int

# `handles` slots
comptime _PG = 0
comptime _DEVICE = 1
comptime _START = 2
comptime _WAIT = 3
comptime _ERROR = 4

# tmb/backend/pg.mojo GRAPH_* kinds
comptime REDUCE_SCATTER = 2


@always_inline
def _fn[F: TrivialRegisterPassable](addr: Int) -> F:
    var a = addr
    return Pointer(to=a).unsafe_bitcast[F]()[]


def _handle(
    handles: InputTensor[dtype=DType.int64, rank=1, ...], i: Int
) -> Int:
    return Int(handles.unsafe_ptr()[unsafe_offset=i])


def _check(
    rc: Int32,
    handles: InputTensor[dtype=DType.int64, rank=1, ...],
    what: StaticString,
) raises:
    if rc != 0:
        var msg = _fn[ErrorFn](_handle(handles, _ERROR))()
        raise Error(
            what,
            " failed: ",
            String(
                unsafe_from_utf8_ptr=Pointer[UInt8, MutUntrackedOrigin](
                    unsafe_from_address=msg
                )
            ) if msg else String("(no message)"),
        )


@always_inline
def _stream(ctx: DeviceContext) raises -> Int:
    return Int(ctx.stream()._handle.value())


def _start[
    kind: Int, nccl_dtype: Int, arg: Int
](
    output: Int,
    input: Int,
    count: Int,
    handles: InputTensor[dtype=DType.int64, rank=1, ...],
    ctx: DeviceContext,
) raises:
    # Nothing to transfer: no call, so no completion event keyed by an
    # address MAX may hand to another empty output.
    if count == 0:
        return
    _check(
        _fn[StartFn](_handle(handles, _START))(
            _handle(handles, _PG),
            Int32(_handle(handles, _DEVICE)),
            Int32(kind),
            input,
            output,
            count,
            Int32(nccl_dtype),
            Int32(arg),
            _stream(ctx),
        ),
        handles,
        "c10d_start",
    )


def _wait(
    output: Int,
    count: Int,
    handles: InputTensor[dtype=DType.int64, rank=1, ...],
    ctx: DeviceContext,
) raises:
    if count == 0:  # `_start` issued nothing
        return
    _check(
        _fn[WaitFn](_handle(handles, _WAIT))(
            _handle(handles, _PG),
            Int32(_handle(handles, _DEVICE)),
            output,
            _stream(ctx),
        ),
        handles,
        "c10d_wait",
    )


@compiler.register("c10d_start")
struct C10dStart:
    @staticmethod
    def execute[
        dtype: DType,
        out_rank: Int,
        in_rank: Int,
        //,
        kind: Int,
        nccl_dtype: Int,
        arg: Int,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=out_rank, ...],
        token: MutableInputTensor,
        input: InputTensor[dtype=dtype, rank=in_rank, ...],
        handles: InputTensor[dtype=DType.int64, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            _start[kind, nccl_dtype, arg](
                Int(output.unsafe_ptr()),
                Int(input.unsafe_ptr()),
                output.size() if kind == REDUCE_SCATTER else input.size(),
                handles,
                ctx,
            )
        else:
            raise Error("graph collectives run on an accelerator only")


@compiler.register("c10d_wait")
struct C10dWait:
    @staticmethod
    def execute[
        dtype: DType,
        rank: Int,
        in_dtype: DType,
        in_rank: Int,
        //,
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=rank, ...],
        token: MutableInputTensor,
        started: InputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=in_dtype, rank=in_rank, ...],
        handles: InputTensor[dtype=DType.int64, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            _wait(Int(started.unsafe_ptr()), started.size(), handles, ctx)
            # TODO: drop this copy once MAX keeps live in-graph buffers apart
            # (agents_docs/optimization_backlog.md, Q6).
            var n = output.size()
            if n > 0:
                ctx.enqueue_copy(
                    DeviceBuffer[dtype](
                        ctx, output.unsafe_ptr(), n, owning=False
                    ),
                    DeviceBuffer[dtype](
                        ctx, started.unsafe_ptr(), n, owning=False
                    ),
                )
        else:
            raise Error("graph collectives run on an accelerator only")
