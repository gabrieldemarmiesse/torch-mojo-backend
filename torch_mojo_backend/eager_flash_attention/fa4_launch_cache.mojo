"""Context-safe compiled-function cache for the FA4 host launchers."""

from std.builtin.device_passable import DevicePassable
from std.collections import OptionalReg
from std.ffi import _get_global_or_null, external_call
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.host.device_context import _DumpPath
from std.memory import OpaquePointer, alloc


@always_inline
def enqueue_fa4_cached[
    declared_arg_types: TypeList[Trait=AnyType, ...],
    //,
    func: def(* args: * declared_arg_types) thin -> None,
    *Ts: DevicePassable,
    use_external_stream: Bool,
    dump_asm: _DumpPath = False,
](
    ctx: DeviceContext,
    context_identity: Int,
    stream_opaque: OpaquePointer[MutAnyOrigin],
    key: String,
    grid: Tuple[Int, Int, Int],
    threads: Int,
    shared_mem_bytes: Int,
    func_attribute: OptionalReg[FuncAttribute],
    *args: *Ts,
) raises:
    # `ctx.id()` is a device ordinal, so it is not sufficient when two MAX
    # contexts target the same GPU. The caller supplies its stable raw
    # DeviceContext address as the cache identity.
    var name = String(
        t"TMB_FA4_DEVICE_FUNCTION_V1_{key}"
        t"_SM{shared_mem_bytes}_CTX{context_identity}"
    )
    comptime FuncT = type_of(ctx.compile_function[func]())

    if global_ptr := _get_global_or_null(name):
        var fptr = global_ptr.value().bitcast[FuncT]()
        comptime if use_external_stream:
            var stream = ctx.create_external_stream(stream_opaque)
            stream.enqueue_function(
                fptr[],
                *args,
                grid_dim=grid,
                block_dim=(threads,),
                shared_mem_bytes=shared_mem_bytes,
            )
        else:
            ctx.enqueue_function(
                fptr[],
                *args,
                grid_dim=grid,
                block_dim=(threads,),
                shared_mem_bytes=shared_mem_bytes,
            )
        return

    var compiled = ctx.compile_function[
        func,
        dump_asm=dump_asm,
    ](func_attribute=func_attribute)
    var fptr = alloc[FuncT](1)
    fptr.init_pointee_move(compiled^)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name),
        fptr.bitcast[NoneType](),
    )

    comptime if use_external_stream:
        var stream = ctx.create_external_stream(stream_opaque)
        stream.enqueue_function(
            fptr[],
            *args,
            grid_dim=grid,
            block_dim=(threads,),
            shared_mem_bytes=shared_mem_bytes,
        )
    else:
        ctx.enqueue_function(
            fptr[],
            *args,
            grid_dim=grid,
            block_dim=(threads,),
            shared_mem_bytes=shared_mem_bytes,
        )
