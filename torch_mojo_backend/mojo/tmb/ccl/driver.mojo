# Vendor CUDA/HIP driver shims for mojoccl.
#
# MAX's own allocator memory cannot be exported with legacy IPC (measured:
# cuIpcGetMemHandle rc=1 on enqueue_create_buffer memory -- see
# agents_docs/mojo_collectives_feasibility.md in the main worktree, section 5.6), so
# this library owns raw driver allocations (cuMemAlloc_v2 / hipExtMallocWith-
# Flags) for its communication regions and shares them with legacy IPC
# (cuIpc*/hipIpc*). Mirrors proto/ipc_probe.mojo's driver-call shape, called
# through the library MAX already dlopened -- no C shim, no NCCL/RCCL.

from std.ffi import OwnedDLHandle
from std.memory.alloc import unsafe_alloc
from max.gpu.host import DeviceContext, DeviceBuffer
from std.sys import has_amd_gpu_accelerator

comptime AMD = has_amd_gpu_accelerator()
comptime DRIVER_LIB = "libamdhip64.so" if AMD else "libcuda.so.1"
comptime FN_ALLOC = "hipExtMallocWithFlags" if AMD else "cuMemAlloc_v2"
comptime FN_GET_HANDLE = "hipIpcGetMemHandle" if AMD else "cuIpcGetMemHandle"
comptime FN_OPEN_HANDLE = "hipIpcOpenMemHandle" if AMD else "cuIpcOpenMemHandle"
comptime FN_CLOSE_HANDLE = "hipIpcCloseMemHandle" if AMD else "cuIpcCloseMemHandle"
comptime FN_FREE = "hipFree" if AMD else "cuMemFree_v2"
comptime FN_GET_DEVICE = "hipGetDevice" if AMD else "cuCtxGetDevice"
comptime FN_LAUNCH_HOST_FUNC = (
    "hipLaunchHostFunc" if AMD else "cuLaunchHostFunc"
)
comptime FN_PCI_BUS_ID = (
    "hipDeviceGetPCIBusId" if AMD else "cuDeviceGetPCIBusId"
)
comptime FN_STREAM_QUERY = "hipStreamQuery" if AMD else "cuStreamQuery"
comptime FN_DEVICE_ATTRIBUTE = (
    "hipDeviceGetAttribute" if AMD else "cuDeviceGetAttribute"
)
comptime FN_HOST_ALLOC = "hipHostMalloc" if AMD else "cuMemHostAlloc"
comptime FN_HOST_FREE = "hipHostFree" if AMD else "cuMemFreeHost"
comptime FN_HOST_DEVPTR = (
    "hipHostGetDevicePointer" if AMD else "cuMemHostGetDevicePointer_v2"
)
# PORTABLE | DEVICEMAP, spelled the same in both APIs
# (CU_MEMHOSTALLOC_PORTABLE|CU_MEMHOSTALLOC_DEVICEMAP,
# hipHostMallocPortable|hipHostMallocMapped).
comptime HOST_ALLOC_FLAGS: UInt32 = 3
# CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS == hipIpcMemLazyEnablePeerAccess == 1
comptime IPC_LAZY_PEER: UInt32 = 1
# hipDeviceMallocUncached: cross-agent flag buffers must be uncached on AMD
# (RCCL's own precondition for polled P2P flags); NVIDIA needs no such flag.
comptime HIP_DEVICE_MALLOC_UNCACHED: UInt32 = 0x3

comptime HANDLE_BYTES = 64


def open_driver() raises -> OwnedDLHandle:
    return OwnedDLHandle(DRIVER_LIB)


def _check(rc: Int32, what: String) raises:
    if rc != 0:
        raise Error(what + " failed, rc=" + String(rc))


struct CompletionEvent(Movable):
    """Owned ordering event; MAX's DeviceEvent has no nonblocking query."""

    var lib: OwnedDLHandle
    var handle: Int

    def __init__(out self, ctx: DeviceContext) raises:
        self.lib = open_driver()
        self.handle = 0
        comptime name = "hipEventCreateWithFlags" if AMD else "cuEventCreate"
        with ctx.push_context():
            _check(
                self.lib.get_function[Int32](name)(
                    Pointer(to=self.handle), UInt32(2)  # DISABLE_TIMING
                ),
                name,
            )

    def __deinit__(deinit self):
        try:
            self.release()
        except e:
            print("mojoccl: completion event cleanup failed:", e)

    def release(mut self) raises:
        if self.handle != 0:
            comptime name = "hipEventDestroy" if AMD else "cuEventDestroy_v2"
            _check(self.lib.get_function[Int32](name)(self.handle), name)
            self.handle = 0

    def record(self, stream: Int64) raises:
        comptime name = "hipEventRecord" if AMD else "cuEventRecord"
        _check(self.lib.get_function[Int32](name)(self.handle, stream), name)

    def wait_on(self, stream: Int64) raises:
        comptime name = "hipStreamWaitEvent" if AMD else "cuStreamWaitEvent"
        _check(
            self.lib.get_function[Int32](name)(stream, self.handle, UInt32(0)),
            name,
        )

    def synchronize(self) raises:
        comptime name = "hipEventSynchronize" if AMD else "cuEventSynchronize"
        _check(self.lib.get_function[Int32](name)(self.handle), name)

    def query(self) raises -> Bool:
        comptime name = "hipEventQuery" if AMD else "cuEventQuery"
        var rc = self.lib.get_function[Int32](name)(self.handle)
        if rc == 600:  # CUDA_ERROR_NOT_READY == hipErrorNotReady
            return False
        _check(rc, name)
        return True

    def done(self) -> Bool:
        try:
            return self.query()
        except:
            return False


def current_device_ordinal(lib: OwnedDLHandle) raises -> Int:
    """The GPU this thread already has current.

    NCCL's binding contract is "the device current on the calling thread"
    (cudaSetDevice / hipSetDevice); nccl.py's set_current_device runs this
    before ncclCommInitRank, matching real NCCL/RCCL -- so this queries
    rather than sets, unlike proto/ipc_probe.mojo's standalone `make_current`.
    """
    var dev: Int32 = 0
    _check(
        lib.get_function[Int32](FN_GET_DEVICE)(Pointer(to=dev)), FN_GET_DEVICE
    )
    return Int(dev)


def device_attribute(lib: OwnedDLHandle, attr: Int, ordinal: Int) raises -> Int:
    """`cuDeviceGetAttribute` / `hipDeviceGetAttribute`, or a negative driver
    rc; raises only if the symbol is missing. Every caller treats "could not
    ask" as "not supported"."""
    var v: Int32 = -1
    var rc = lib.get_function[Int32](FN_DEVICE_ATTRIBUTE)(
        Pointer(to=v), Int32(attr), Int32(ordinal)
    )
    if rc != 0:
        return -Int(rc)
    return Int(v)


# CU_DEVICE_ATTRIBUTE_ / hipDeviceAttributeDirectManagedMemAccessFromHost.
# HIP's is 13 in every ROCm from 5.7 to 7.2: hip_runtime_api.h keeps a retired
# entry as `hipDeviceAttributeUnused<n>`, so that block never renumbers.
comptime ATTR_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 13 if AMD else 101


def direct_managed_mem_access(lib: OwnedDLHandle, ordinal: Int) -> Bool:
    """Whether the host accesses this GPU's managed memory directly: an APU
    (MI300A), not a discrete GPU (MI300X). RCCL's test for its multi-node
    grid (`_node_grids`). False, with one line saying so, if the driver will
    not answer, which keeps the discrete-GPU grids."""
    try:
        var v = device_attribute(
            lib, ATTR_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST, ordinal
        )
        if v >= 0:
            return v != 0
        print(
            "mojoccl: DirectManagedMemAccessFromHost query failed, rc=",
            -v,
            "; using the discrete-GPU collective grids",
        )
    except e:
        print("mojoccl:", FN_DEVICE_ATTRIBUTE, "unavailable:", e)
    return False


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


def launch_host_func(
    lib: OwnedDLHandle, stream: Int, func_addr: Int, user_data: Int
) raises:
    """cuLaunchHostFunc / hipLaunchHostFunc on a RAW stream handle.

    The ordering primitive the inter-node hop stands on: a host function
    enqueued on the caller's stream runs after everything enqueued before it
    has completed, and everything enqueued after it waits for it to return.
    That is what lets `internode.mojo` post its RDMA writes knowing the
    reduce-scatter's output is final, and lets the add kernel start knowing
    the peers' shards have landed.

    Contract (identical in both vendors' docs): the callback must not call
    into the driver. This library's callback only touches libibverbs and its
    own host memory, never cu*/hip*.
    """
    _check(
        lib.get_function[Int32](FN_LAUNCH_HOST_FUNC)(
            stream, func_addr, user_data
        ),
        FN_LAUNCH_HOST_FUNC,
    )


def stream_done(lib: OwnedDLHandle, stream: Int) -> Bool:
    """`cuStreamQuery`/`hipStreamQuery` on a RAW stream handle: has everything
    enqueued on it completed?

    `ncclCommAbort` polls this instead of synchronizing. A synchronize is
    exactly the unbounded wait abort must not do, and the whole point of the
    abort word is that the spin kernels are already leaving; a poll turns
    "quiesced" into something abort can put a deadline on. Anything but
    success (`CUDA_ERROR_NOT_READY`, or a sticky error from a launch that
    faulted) reads as not-done and the caller's deadline ends the wait.
    """
    try:
        return lib.get_function[Int32](FN_STREAM_QUERY)(stream) == 0
    except:
        return False


def device_pci_bus_id(lib: OwnedDLHandle, ordinal: Int) raises -> String:
    """The GPU's PCI address, e.g. `0000:1b:00.0`, lowercased.

    Used to pair a rank with the IB HCA nearest its GPU (internode.mojo);
    both vendors expose the same call with the same signature.
    """
    var buf = unsafe_alloc[UInt8](32)
    for i in range(32):
        buf[unsafe_offset=i] = 0
    var rc = lib.get_function[Int32](FN_PCI_BUS_ID)(
        buf, Int32(32), Int32(ordinal)
    )
    if rc != 0:
        return String("")
    var s = String("")
    var i = 0
    while i < 31 and buf[unsafe_offset=i] != 0:
        var c = Int(buf[unsafe_offset=i])
        if c >= 65 and c <= 90:
            c += 32
        s += chr(c)
        i += 1
    return s^


def alloc_host(lib: OwnedDLHandle, nbytes: Int) raises -> Int:
    """Pinned, device-mapped host memory: the proxy thread's mailbox.

    The two words the GPU and the progress thread exchange live here --
    written by a one-thread kernel with a system-scope release and read by
    the CPU (and the reverse). Pageable memory would not do: the device
    mapping is what lets a kernel touch it at all.
    """
    var p: Int = 0
    _check(
        lib.get_function[Int32](FN_HOST_ALLOC)(
            Pointer(to=p), nbytes, HOST_ALLOC_FLAGS
        ),
        FN_HOST_ALLOC,
    )
    return p


def host_device_ptr(lib: OwnedDLHandle, host_addr: Int) raises -> Int:
    """The address a kernel must use for `alloc_host` memory.

    Equal to the host address under unified addressing on both vendors, but
    asked for rather than assumed.
    """
    var d: Int = 0
    _check(
        lib.get_function[Int32](FN_HOST_DEVPTR)(
            Pointer(to=d), host_addr, UInt32(0)
        ),
        FN_HOST_DEVPTR,
    )
    return d


def free_host(lib: OwnedDLHandle, addr: Int) raises:
    _check(lib.get_function[Int32](FN_HOST_FREE)(addr), FN_HOST_FREE)
