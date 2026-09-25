# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/transport/p2p.cc
#
# MAX's own allocator memory cannot be exported with legacy IPC (measured:
# cuIpcGetMemHandle rc=1 on enqueue_create_buffer memory -- see
# agents_docs/mojo_collectives_feasibility.md in the main worktree, section 5.6), so
# this library owns raw driver allocations (cuMemAlloc_v2 / hipExtMallocWith-
# Flags) for its communication regions and shares them with legacy IPC
# (cuIpc*/hipIpc*). Mirrors proto/ipc_probe.mojo's driver-call shape, called
# through the library MAX already dlopened -- no C shim, no NCCL/RCCL.

from max.gpu.host import DeviceBuffer, DeviceContext
from std.ffi import OwnedDLHandle

from tmb.ccl.misc.cudawrap import AMD, _check


comptime FN_ALLOC = "hipExtMallocWithFlags" if AMD else "cuMemAlloc_v2"
comptime FN_GET_HANDLE = "hipIpcGetMemHandle" if AMD else "cuIpcGetMemHandle"
comptime FN_OPEN_HANDLE = "hipIpcOpenMemHandle" if AMD else "cuIpcOpenMemHandle"
comptime FN_CLOSE_HANDLE = "hipIpcCloseMemHandle" if AMD else "cuIpcCloseMemHandle"
comptime FN_FREE = "hipFree" if AMD else "cuMemFree_v2"


# CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS == hipIpcMemLazyEnablePeerAccess == 1
comptime IPC_LAZY_PEER: UInt32 = 1
# hipDeviceMallocUncached: cross-agent flag buffers must be uncached on AMD
# (RCCL's own precondition for polled P2P flags); NVIDIA needs no such flag.
comptime HIP_DEVICE_MALLOC_UNCACHED: UInt32 = 0x3

comptime HANDLE_BYTES = 64


def alloc_region(lib: OwnedDLHandle, nbytes: Int) raises -> Int:
    """cuMemAlloc_v2 / hipExtMallocWithFlags(hipDeviceMallocUncached, size).

    Returns the base device address. Not zeroed -- the caller zeros the
    signal-area prefix via `zero_bytes` (region_init).
    """
    var base: Int = 0
    comptime if AMD:
        _check(
            lib.get_function[Int32](FN_ALLOC)(
                Pointer(to=base), nbytes, HIP_DEVICE_MALLOC_UNCACHED
            ),
            FN_ALLOC,
        )
    else:
        _check(
            lib.get_function[Int32](FN_ALLOC)(Pointer(to=base), nbytes),
            FN_ALLOC,
        )
    return base


def free_region(lib: OwnedDLHandle, addr: Int) raises:
    _check(lib.get_function[Int32](FN_FREE)(addr), FN_FREE)


def get_handle(
    lib: OwnedDLHandle, addr: Int, out_bytes: Pointer[UInt8, MutAnyOrigin]
) raises:
    """cuIpcGetMemHandle / hipIpcGetMemHandle: fills `out_bytes[0:64]`."""
    _check(
        lib.get_function[Int32](FN_GET_HANDLE)(out_bytes, addr), FN_GET_HANDLE
    )


def open_handle(
    lib: OwnedDLHandle, handle: Pointer[UInt8, MutAnyOrigin]
) raises -> Int:
    """cuIpcOpenMemHandle / hipIpcOpenMemHandle(..., LazyEnablePeerAccess).

    `CUipcMemHandle`/`hipIpcMemHandle_t` is a 64-byte struct passed BY VALUE
    (SysV MEMORY class: no integer register, pushed on the stack as 8 qwords)
    after the two real leading args (pdptr -> rdi, flags -> esi) -- the exact
    shim proto/ipc_probe.mojo uses and measured working (§5.3): four dummy
    Int64 args exhaust rdx/rcx/r8/r9, then the 8 handle qwords land on the
    stack where a real C caller would place them. `std.ffi` has no C-struct
    ABI yet (MOCO-3692/3709).
    """
    var opn = lib.get_function[Int32](FN_OPEN_HANDLE)
    var h64 = handle.unsafe_bitcast[UInt64]()
    var ptr: Int = 0
    _check(
        opn(
            Pointer(to=ptr),
            IPC_LAZY_PEER,
            Int64(0),
            Int64(0),
            Int64(0),
            Int64(0),
            h64[unsafe_offset=0],
            h64[unsafe_offset=1],
            h64[unsafe_offset=2],
            h64[unsafe_offset=3],
            h64[unsafe_offset=4],
            h64[unsafe_offset=5],
            h64[unsafe_offset=6],
            h64[unsafe_offset=7],
        ),
        FN_OPEN_HANDLE,
    )
    return ptr


def close_handle(lib: OwnedDLHandle, addr: Int) raises:
    _check(lib.get_function[Int32](FN_CLOSE_HANDLE)(addr), FN_CLOSE_HANDLE)


def zero_bytes(ctx: DeviceContext, addr: Int, nbytes: Int) raises:
    """Zero-fill `nbytes` at a raw device address, blocking.

    Blocking (unlike every per-collective op here) is fine: this runs once
    per rank, at communicator creation, on `ctx`'s own queue -- while later
    collectives run on the caller-supplied external stream, so without a
    synchronize here the first collective could race this region_init.
    """
    var buf = DeviceBuffer[DType.uint8](
        ctx,
        Pointer[Scalar[DType.uint8], MutUntrackedOrigin](
            unsafe_from_address=addr
        ),
        nbytes,
        owning=False,
    )
    ctx.enqueue_memset(buf, 0)
    ctx.synchronize()
