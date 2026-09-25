# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/misc/cudawrap.cc
#
# Vendor CUDA/HIP driver shims for mojoccl.

from std.sys import has_amd_gpu_accelerator
from std.ffi import OwnedDLHandle
from std.memory.alloc import unsafe_alloc


comptime AMD = has_amd_gpu_accelerator()
comptime DRIVER_LIB = "libamdhip64.so" if AMD else "libcuda.so.1"


comptime FN_GET_DEVICE = "hipGetDevice" if AMD else "cuCtxGetDevice"


comptime FN_PCI_BUS_ID = (
    "hipDeviceGetPCIBusId" if AMD else "cuDeviceGetPCIBusId"
)


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


def open_driver() raises -> OwnedDLHandle:
    return OwnedDLHandle(DRIVER_LIB)


def _check(rc: Int32, what: String) raises:
    if rc != 0:
        raise Error(what + " failed, rc=" + String(rc))


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


def device_pci_bus_id(lib: OwnedDLHandle, ordinal: Int) raises -> String:
    """The GPU's PCI address, e.g. `0000:1b:00.0`, lowercased.

    Used to pair a rank with the IB HCA nearest its GPU (proxy.mojo);
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
