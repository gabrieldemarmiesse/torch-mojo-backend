"""`_c10d_functional` collectives as MAX graph custom ops.

A traced collective is a start and a `wait_tensor`. `c10d_start` enqueues
it, through the process group of the native backend (`tmb/backend/pg.mojo`,
`tmb_pg_graph_start`), on the communicator's comm stream after the graph
stream's work so far; `c10d_wait` makes the graph stream wait for it. Both
write into one MAX buffer the Python lowering
(`torch_compile_backend/collectives.py`) creates and loads after the wait,
and both are in-place ops, so MAX keeps them in program order on the
device chain -- every rank issues its collectives in the same order -- while
the compute between them overlaps the transfer. The wait also takes the
collective's input, so MAX cannot release it while the comm stream reads it.
`c10d_start_out` / `c10d_wait_copy` are the same pair for an output MAX
cannot hold in a buffer (see there).

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
    handles: InputTensor[dtype=DType.int64, rank=1, ...],
    ctx: DeviceContext,
) raises:
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
    """Into a graph buffer the lowering loads after `c10d_wait`."""

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
        output: MutableInputTensor[dtype=dtype, rank=out_rank, ...],
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
        output: MutableInputTensor[dtype=dtype, rank=rank, ...],
        input: InputTensor[dtype=in_dtype, rank=in_rank, ...],
        handles: InputTensor[dtype=DType.int64, rank=1, ...],
        ctx: DeviceContext,
    ) raises:
        comptime if target == "gpu":
            _wait(Int(output.unsafe_ptr()), handles, ctx)
        else:
            raise Error("graph collectives run on an accelerator only")


# MAX loads a graph buffer only when every dim of its shape is static or a
# graph input's symbol ('mo.mutable.load' fails to verify otherwise, even for
# a symbol a `rebind` introduces), so an output shaped like `s0 // 2` cannot
# be a buffer. These two variants let MAX allocate it as an ordinary output
# instead; they stay on the device chain through `token`, a buffer nothing
# reads, and the wait hands its consumers a copy -- a tensor output cannot
# alias the start's.


@compiler.register("c10d_start_out")
struct C10dStartOut:
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


@compiler.register("c10d_wait_copy")
struct C10dWaitCopy:
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
            _wait(Int(started.unsafe_ptr()), handles, ctx)
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
