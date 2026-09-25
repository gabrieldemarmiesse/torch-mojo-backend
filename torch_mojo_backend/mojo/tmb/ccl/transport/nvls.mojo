# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/transport/nvls.cc
#
# CUDA VMM + NVSwitch-multicast bring-up for the region device/all_reduce.mojo
# reduces through, called into libcuda.so.1 with the `OwnedDLHandle`
# misc/cudawrap.mojo already opened (MAX exposes none of this API).
#
# The sequence mirrors NCCL's `nccl:src/transport/nvls.cc` and is the one the
# prototype measured working end to end (`nvls/mcsetup.mojo`, its RESULTS.md
# section 2):
#
#   local rank 0 : cuMulticastGetGranularity -> cuMulticastCreate -> export as
#                  a POSIX fd, send it to the node's other ranks
#   all          : cuMemImportFromShareableHandle, cuMulticastAddDevice(mine)
#   BARRIER        (every device must be in the team before any memory binds)
#   all          : cuMemCreate(own physical memory, POSIX-fd handle type,
#                  GPUDirect-RDMA-capable so an HCA can register it)
#                  cuMulticastBindMem(mc, 0, mine, 0, size)  -- every rank at
#                  multicast offset 0, so ONE multicast address covers the
#                  node's eight distinct physical allocations
#                  cuMemAddressReserve/Map/SetAccess twice: once against the
#                  multicast handle (the address `multimem.*` takes) and once
#                  against my own memory (ordinary loads and stores)
#   all          : export my own handle as an fd too and send it to each local
#                  peer, import theirs, map them -- this replaces
#                  cuIpcOpenMemHandle, which cannot open VMM memory
#   BARRIER        (nobody issues a multimem instruction before all have mapped)
#
# Granularity: the bound size must be a multiple of what
# `cuMulticastGetGranularity` reports, and on H100 that is 512 MiB for
# RECOMMENDED against 2 MiB for MINIMUM. This asks for MINIMUM, unlike NCCL,
# so that `MOJOCCL_REGION_MB` keeps meaning what it says -- at RECOMMENDED the
# default 256 MiB region (128 KiB + 2 x 256 MiB) rounds up to a 1 GiB
# allocation per rank, and even a deliberately tiny test region costs 512 MiB.
# MINIMUM and RECOMMENDED measured the same on H100 (agents_docs/distributed.md),
# so production uses MINIMUM to avoid rounding up that much memory.

from std.os import getenv
from std.ffi import OwnedDLHandle
from std.memory.alloc import unsafe_alloc

from tmb.ccl.device.all_reduce import nvls_min_bytes
from tmb.ccl.env_vars import MOJOCCL_NVLS
from tmb.ccl.include.device import MAX_WORLD
from tmb.ccl.include.transport import NvlsRegion
from tmb.ccl.misc.cudawrap import device_attribute
from tmb.ccl.os.linux_ipcsocket import scm_exchange_fds, socket_path
from tmb.ccl.transport.multicast import (
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
    MSG_KIND_UC,
    NVLS_AVAILABLE,
    _cu,
    _mc_prop,
)


comptime NVLS_FD_TIMEOUT_S: Float64 = 30.0
"""Bound on one fd hand-off. The whole bring-up is 150-230 ms when it works,
so 30 s only ever fires when a local peer died mid-init -- and it has to fire,
or the surviving ranks block in `recvmsg` forever."""


def _nvls_enabled() -> Bool:
    """`MOJOCCL_NVLS=0` turns the multicast path off, region and all.

    Folded into the round-1 capability word, so one rank setting it takes the
    whole communicator back to the unicast kernels -- a communicator where
    some ranks bound a multicast region and others did not is not a
    configuration, it is a crash.
    """
    return getenv(MOJOCCL_NVLS, String("1")) != String("0")


def _nvls_min_bytes() -> Int:
    """Message size at or above which a single-node allreduce goes through the
    switch (48 MiB -- the measured crossover,
    see `NVLS_MIN_BYTES` in device/all_reduce.mojo)."""
    return nvls_min_bytes()


def _nvls_recommended_granularity() -> Bool:
    """Use MINIMUM multicast granularity: it allocates what the region asks
    for. MINIMUM and RECOMMENDED measured the same on H100; see
    agents_docs/distributed.md's NVLS measurements."""
    return False


# CUmemAllocationType
comptime CU_MEM_ALLOCATION_TYPE_PINNED = 1
# CUmemLocationType
comptime CU_MEM_LOCATION_TYPE_DEVICE = 1
# CUmemAccess_flags
comptime CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
# CUmulticastGranularity_flags / CUmemAllocationGranularity_flags
comptime GRANULARITY_MINIMUM = 0
comptime GRANULARITY_RECOMMENDED = 1
# CUdevice_attribute
comptime ATTR_MULTIPROCESSOR_COUNT = 16
comptime ATTR_MULTICAST_SUPPORTED = 132
comptime ATTR_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110


def _align_up(x: Int, a: Int) -> Int:
    return (x + a - 1) // a * a


def sm_count(lib: OwnedDLHandle, ordinal: Int) -> Int:
    """SMs on this device, for the NVLS grid (which must be fully resident).
    0 if the driver refuses to say, which `nvls_blocks` reads as "use the
    fitted default"."""
    comptime if not NVLS_AVAILABLE:
        return 0
    try:
        var n = device_attribute(lib, ATTR_MULTIPROCESSOR_COUNT, ordinal)
        return n if n > 0 else 0
    except:
        return 0


# ===-------------------------------------------------------------------=== #
# Driver property blocks, laid out by hand over UInt64 words (field offsets
# from cuda.h; `std.ffi` has no C-struct ABI).
# ===-------------------------------------------------------------------=== #


def _mem_prop(
    lib: OwnedDLHandle, ordinal: Int
) raises -> Pointer[UInt64, MutUntrackedOrigin]:
    """CUmemAllocationProp{type, requestedHandleTypes, location, win32, flags}.

    `allocFlags.gpuDirectRDMACapable` is set whenever the device supports it,
    as NCCL does (nccl:src/include/alloc.h:329) and as MAX's own arena does:
    nvidia_peermem refuses to pin a VMM chunk created without it, so
    `ibv_reg_mr` on the unicast mapping returned NULL (EFAULT) until this was
    set (agents_docs/mojo_collectives_nvls_results.md §4).
    """
    var p = unsafe_alloc[UInt64](4)
    p[unsafe_offset=0] = UInt64(CU_MEM_ALLOCATION_TYPE_PINNED) | (
        UInt64(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) << 32
    )
    p[unsafe_offset=1] = UInt64(CU_MEM_LOCATION_TYPE_DEVICE) | (
        UInt64(UInt32(ordinal)) << 32
    )
    p[unsafe_offset=2] = 0
    # CUmemAllocationFlags{compressionType: u8, gpuDirectRDMACapable: u8, usage: u16}
    var rdma = (
        device_attribute(
            lib, ATTR_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED, ordinal
        )
        == 1
    )
    p[unsafe_offset=3] = UInt64(1) << 8 if rdma else UInt64(0)
    return p


def _access_desc(ordinal: Int) -> Pointer[UInt64, MutUntrackedOrigin]:
    """CUmemAccessDesc{location{type,id}, flags}."""
    var p = unsafe_alloc[UInt64](2)
    p[unsafe_offset=0] = UInt64(CU_MEM_LOCATION_TYPE_DEVICE) | (
        UInt64(UInt32(ordinal)) << 32
    )
    p[unsafe_offset=1] = UInt64(CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
    return p


# ===-------------------------------------------------------------------=== #
# Capability probe and sizing
# ===-------------------------------------------------------------------=== #


def multicast_granularity(
    lib: OwnedDLHandle,
    world: Int,
    ordinal: Int,
    want_bytes: Int,
    recommended: Bool,
) raises -> Int:
    """The alignment both the multicast object and the physical allocation
    need; the region is rounded up to a multiple of it. See the header for why
    `recommended` defaults to False."""
    var probe = _mc_prop(world, max(want_bytes, 1))
    var g_mc: Int = 0
    _cu(
        lib,
        lib.get_function[Int32]("cuMulticastGetGranularity")(
            Pointer(to=g_mc),
            probe,
            Int32(
                GRANULARITY_RECOMMENDED if recommended else GRANULARITY_MINIMUM
            ),
        ),
        "cuMulticastGetGranularity",
    )
    var g_mem: Int = 0
    _cu(
        lib,
        lib.get_function[Int32]("cuMemGetAllocationGranularity")(
            Pointer(to=g_mem),
            _mem_prop(lib, ordinal),
            Int32(GRANULARITY_RECOMMENDED),
        ),
        "cuMemGetAllocationGranularity",
    )
    var g = max(g_mc, g_mem)
    if g <= 0:
        raise Error("mojoccl: multicast granularity came back as " + String(g))
    return g


def multicast_capable(
    lib: OwnedDLHandle, ordinal: Int, world: Int, probe_create: Bool
) -> Bool:
    """Can this rank take the NVLS path?

    Attribute first (`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED`), then -- on the
    one rank per node that will actually create the object -- a real
    `cuMulticastCreate` of a granularity-sized object, released immediately.
    The create is where a broken fabric-manager configuration shows up (NCCL
    calls its bind "where we normally see issues if the system NVLS/Multicast
    support is broken", nccl:src/transport/nvls.cc:369), and finding out here
    costs 88 us and lets every rank agree to fall back BEFORE anything is
    allocated.
    """
    comptime if not NVLS_AVAILABLE:
        return False
    try:
        if device_attribute(lib, ATTR_MULTICAST_SUPPORTED, ordinal) != 1:
            return False
        if not probe_create:
            return True
        var g = multicast_granularity(lib, world, ordinal, 1, False)
        var h: UInt64 = 0
        var rc = lib.get_function[Int32]("cuMulticastCreate")(
            Pointer(to=h), _mc_prop(world, g)
        )
        if rc != 0:
            return False
        _ = lib.get_function[Int32]("cuMemRelease")(h)
        return True
    except:
        return False


# ===-------------------------------------------------------------------=== #
# The region
# ===-------------------------------------------------------------------=== #


def _map(
    lib: OwnedDLHandle, handle: UInt64, size: Int, gran: Int, ordinal: Int
) raises -> Int:
    """Reserve a VA range, map `handle` into it, and grant this device
    read/write. A step that fails releases what the earlier steps took, so
    the caller only ever owns a fully mapped VA or nothing."""
    var va: Int = 0
    _cu(
        lib,
        lib.get_function[Int32]("cuMemAddressReserve")(
            Pointer(to=va), size, gran, Int(0), UInt64(0)
        ),
        "cuMemAddressReserve",
    )
    var rc = lib.get_function[Int32]("cuMemMap")(
        va, size, Int(0), handle, UInt64(0)
    )
    if rc != 0:
        _ = lib.get_function[Int32]("cuMemAddressFree")(va, size)
        _cu(lib, rc, "cuMemMap")
    rc = lib.get_function[Int32]("cuMemSetAccess")(
        va, size, _access_desc(ordinal), Int(1)
    )
    if rc != 0:
        _unmap(lib, va, size)
        _cu(lib, rc, "cuMemSetAccess")
    return va


def _unmap(lib: OwnedDLHandle, va: Int, size: Int) raises:
    if va == 0:
        return
    _ = lib.get_function[Int32]("cuMemUnmap")(va, size)
    _ = lib.get_function[Int32]("cuMemAddressFree")(va, size)


def nvls_teardown(
    mut region: NvlsRegion, lib: OwnedDLHandle, ordinal: Int
) raises:
    """Release everything the bring-up took, in reverse, and blank the struct
    so a second call is a no-op. Tolerates a partly built region, which is
    what lets the failure path call it too."""
    comptime if not NVLS_AVAILABLE:
        return
    for r in range(MAX_WORLD):
        if region.peer_va[r] != region.uc:
            _unmap(lib, region.peer_va[r], region.size)
        if region.peer_handle[r] != 0:
            _ = lib.get_function[Int32]("cuMemRelease")(region.peer_handle[r])
        region.peer_va[r] = 0
        region.peer_handle[r] = 0
    _unmap(lib, region.uc, region.size)
    _unmap(lib, region.mc, region.size)
    if region.mc_handle != 0 and region.mem_handle != 0:
        _ = lib.get_function[Int32]("cuMulticastUnbind")(
            region.mc_handle, Int32(ordinal), Int(0), region.size
        )
    if region.mem_handle != 0:
        _ = lib.get_function[Int32]("cuMemRelease")(region.mem_handle)
    if region.mc_handle != 0:
        _ = lib.get_function[Int32]("cuMemRelease")(region.mc_handle)
    region.uc = 0
    region.mc = 0
    region.mem_handle = 0
    region.mc_handle = 0
    region.size = 0


def nvls_bind_and_map(
    lib: OwnedDLHandle,
    libc: OwnedDLHandle,
    dir: String,
    magic: UInt64,
    local_rank: Int,
    local_world: Int,
    ordinal: Int,
    sock: Int,
    timeout_s: Float64,
    mut region: NvlsRegion,
) raises:
    """Steps 4-6: bind my physical memory into the object, map it twice, and
    trade unicast handles with the node's other ranks.

    The peers' mappings replace `cuIpcOpenMemHandle`, which cannot open VMM
    memory; this is also what NCCL's P2P transport does with driver >= 12.0
    (nccl:src/transport/p2p.cc:267-327). The caller barriers after this: no
    rank may issue a multimem instruction before every rank has mapped.
    """
    comptime if not NVLS_AVAILABLE:
        raise Error("mojoccl: NVLS is CUDA-only")
    var size = region.size
    var memh: UInt64 = 0
    _cu(
        lib,
        lib.get_function[Int32]("cuMemCreate")(
            Pointer(to=memh), size, _mem_prop(lib, ordinal), UInt64(0)
        ),
        "cuMemCreate",
    )
    region.mem_handle = memh
    _cu(
        lib,
        lib.get_function[Int32]("cuMulticastBindMem")(
            region.mc_handle, Int(0), memh, Int(0), size, UInt64(0)
        ),
        "cuMulticastBindMem",
    )
    region.mc = _map(lib, region.mc_handle, size, region.granularity, ordinal)
    region.uc = _map(lib, memh, size, region.granularity, ordinal)
    region.peer_va[local_rank] = region.uc

    # Trade unicast handles, sending and receiving at the same time: see
    # `scm_exchange_fds` for why "everyone sends, then everyone receives"
    # cannot be safe on a host with a small `net.unix.max_dgram_qlen`.
    # One export covers every peer -- SCM_RIGHTS installs a new descriptor in
    # each receiver, and a descriptor already in flight stays valid after the
    # sender closes it.
    var paths = List[String]()
    for r in range(local_world):
        if r != local_rank:
            paths.append(socket_path(dir, magic, r))
    var fd: Int32 = -1
    _cu(
        lib,
        lib.get_function[Int32]("cuMemExportToShareableHandle")(
            Pointer(to=fd),
            memh,
            Int32(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
            UInt64(0),
        ),
        "cuMemExportToShareableHandle(unicast)",
    )
    try:
        var got = scm_exchange_fds(
            libc, sock, paths, Int(fd), MSG_KIND_UC, local_rank, timeout_s
        )
        for i in range(len(got)):
            var g = got[i]
            if g[1] != MSG_KIND_UC or g[2] < 0 or g[2] >= local_world:
                _ = libc.get_function[Int32]("close")(Int32(g[0]))
                raise Error(
                    "mojoccl: unexpected datagram on the fd socket, kind "
                    + String(g[1])
                    + " tag "
                    + String(g[2])
                )
            var ph: UInt64 = 0
            var rc = lib.get_function[Int32]("cuMemImportFromShareableHandle")(
                Pointer(to=ph),
                g[0],
                Int32(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
            )
            _ = libc.get_function[Int32]("close")(Int32(g[0]))
            _cu(lib, rc, "cuMemImportFromShareableHandle(peer)")
            # Recorded before `_map` can raise, so teardown releases it.
            region.peer_handle[g[2]] = ph
            region.peer_va[g[2]] = _map(
                lib, ph, size, region.granularity, ordinal
            )
    except e:
        _ = libc.get_function[Int32]("close")(fd)
        raise e
    _ = libc.get_function[Int32]("close")(fd)
    for r in range(local_world):
        if region.peer_va[r] == 0:
            raise Error(
                "mojoccl: local rank " + String(r) + " never sent its region"
            )
