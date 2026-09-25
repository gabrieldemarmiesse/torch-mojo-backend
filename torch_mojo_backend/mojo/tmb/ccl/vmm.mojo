# CUDA VMM + NVSwitch-multicast bring-up for the region nvls_kernels.mojo
# reduces through, called into libcuda.so.1 with the `OwnedDLHandle`
# driver.mojo already opened (MAX exposes none of this API).
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
# fd transport is SCM_RIGHTS over an AF_UNIX SOCK_DGRAM socket, what NCCL does
# (nccl:src/os/linux_ipcsocket.cc:171-245). NOT pidfd_getfd: that needs
# PTRACE_MODE_ATTACH, i.e. /proc/sys/kernel/yama/ptrace_scope <= 1, and it is 1
# on stock Ubuntu and 2 or 3 on hardened images -- a collective that silently
# loses NVLS to a sysctl is worse than 70 lines of hand-laid-out `sendmsg`.
# `std.ffi` has no C-struct ABI, so msghdr/cmsghdr/sockaddr_un are written out
# over UInt64 words, field offsets from the glibc headers.
#
# Granularity: the bound size must be a multiple of what
# `cuMulticastGetGranularity` reports, and on H100 that is 512 MiB for
# RECOMMENDED against 2 MiB for MINIMUM. This asks for MINIMUM, unlike NCCL,
# so that `MOJOCCL_REGION_MB` keeps meaning what it says -- at RECOMMENDED the
# default 256 MiB region (128 KiB + 2 x 256 MiB) rounds up to a 1 GiB
# allocation per rank, and even a deliberately tiny test region costs 512 MiB.
# MINIMUM and RECOMMENDED measured the same on H100 (agents_docs/distributed.md),
# so production uses MINIMUM to avoid rounding up that much memory.

from std.ffi import OwnedDLHandle, external_call
from std.memory.alloc import unsafe_alloc
from std.sys import has_amd_gpu_accelerator
from std.time import perf_counter_ns, sleep
from std.utils import StaticTuple

from tmb.ccl.collectives_kernels import MAX_WORLD
from tmb.ccl.driver import device_attribute

comptime LIBC = "libc.so.6"

# CUmemAllocationHandleType
comptime CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1
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

# libc / linux
comptime AF_UNIX: Int32 = 1
comptime SOCK_DGRAM: Int32 = 2
comptime SOL_SOCKET: Int32 = 1
comptime SO_RCVTIMEO: Int32 = 20
comptime SO_SNDTIMEO: Int32 = 21
comptime SCM_RIGHTS = 1
# MSG_DONTWAIT: every sendmsg/recvmsg below is one non-blocking attempt, and
# the wait is the caller's deadline loop. A blocking `sendmsg` on an AF_UNIX
# datagram socket waits for room in the RECEIVER's queue, which is bounded by
# `net.unix.max_dgram_qlen`; with eight ranks each sending to the seven others
# before receiving anything, a small qlen wedges the whole node's bring-up in
# a syscall no deadline can reach.
comptime MSG_DONTWAIT: Int32 = 0x40
comptime EAGAIN: Int32 = 11

comptime MSG_KIND_MC = 0
"""Payload word 0 of the datagram carrying the multicast object's fd."""
comptime MSG_KIND_UC = 1
"""...and of the one carrying a peer's own memory handle; word 1 is that
peer's local rank."""

comptime NVLS_AVAILABLE = not has_amd_gpu_accelerator()
"""Every entry point below is CUDA-only; HIP has no multicast equivalent and
RCCL none either, so an AMD build never reaches them."""


def _errno() -> Int32:
    return external_call["__errno_location", Pointer[Int32, MutAnyOrigin]]()[
        unsafe_offset=0
    ]


def _cu(lib: OwnedDLHandle, rc: Int32, what: String) raises:
    if rc != 0:
        raise Error("mojoccl: " + what + " failed, driver rc=" + String(rc))


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


def _mc_prop(world: Int, size: Int) -> Pointer[UInt64, MutUntrackedOrigin]:
    """CUmulticastObjectProp{numDevices, size, handleTypes, flags}, 32 B."""
    var p = unsafe_alloc[UInt64](4)
    p[unsafe_offset=0] = UInt64(UInt32(world))  # numDevices +0, pad +4
    p[unsafe_offset=1] = UInt64(size)  # size +8
    p[unsafe_offset=2] = UInt64(
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )  # handleTypes +16
    p[unsafe_offset=3] = 0  # flags +24
    return p


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
# fd transport: SCM_RIGHTS over AF_UNIX SOCK_DGRAM
# ===-------------------------------------------------------------------=== #


def socket_path(dir: String, magic: UInt64, local_rank: Int) -> String:
    """Where local rank `local_rank` of the communicator `magic` listens.

    Derived, not exchanged: every rank of a node computes the same name from
    the unique id it was already given, so the TCP bootstrap carries no extra
    round. The magic is per-`ncclGetUniqueId`, so two communicators alive at
    once do not collide.
    """
    return dir + "/mojoccl-" + hex(magic) + "-" + String(local_rank) + ".sock"


def _sockaddr(path: String) raises -> Pointer[UInt8, MutUntrackedOrigin]:
    """sockaddr_un{u16 family; char path[108]}, zero filled."""
    var sa = unsafe_alloc[UInt8](112)
    for i in range(112):
        sa[unsafe_offset=i] = 0
    sa.unsafe_bitcast[UInt16]()[unsafe_offset=0] = UInt16(AF_UNIX)
    var b = path.as_bytes()
    if len(b) >= 107:
        raise Error("mojoccl: unix socket path too long: " + path)
    for i in range(len(b)):
        sa[unsafe_offset=2 + i] = b[i]
    return sa


def scm_bind(
    libc: OwnedDLHandle, path: String, timeout_s: Float64
) raises -> Int:
    """Create and bind a datagram socket that receives one fd per message.

    `SO_RCVTIMEO` matters: without it a rank whose peer died during bring-up
    blocks in `recvmsg` forever, and the communicator hangs instead of
    failing.
    """
    _ = libc.get_function[Int32]("unlink")(_sockaddr(path).unsafe_offset(2))
    var s = libc.get_function[Int32]("socket")(AF_UNIX, SOCK_DGRAM, Int32(0))
    if s < 0:
        raise Error("mojoccl: socket(AF_UNIX) failed")
    if libc.get_function[Int32]("bind")(s, _sockaddr(path), Int32(110)) != 0:
        _ = libc.get_function[Int32]("close")(s)
        raise Error("mojoccl: bind(" + path + ") failed")
    var tv = unsafe_alloc[Int64](2)  # struct timeval
    tv[unsafe_offset=0] = Int64(timeout_s)
    tv[unsafe_offset=1] = 0
    _ = libc.get_function[Int32]("setsockopt")(
        s, SOL_SOCKET, SO_RCVTIMEO, tv, Int32(16)
    )
    return Int(s)


def scm_unbind(libc: OwnedDLHandle, sock: Int, path: String) raises:
    if sock > 0:
        _ = libc.get_function[Int32]("close")(Int32(sock))
    _ = libc.get_function[Int32]("unlink")(_sockaddr(path).unsafe_offset(2))


def _send_socket(libc: OwnedDLHandle, timeout_s: Float64) raises -> Int32:
    """A datagram socket for sending fds. `SO_SNDTIMEO` is a backstop only --
    every send below passes `MSG_DONTWAIT` -- but it is what bounds a send
    that somehow blocks anyway."""
    var s = libc.get_function[Int32]("socket")(AF_UNIX, SOCK_DGRAM, Int32(0))
    if s < 0:
        raise Error("mojoccl: socket(AF_UNIX) failed")
    var tv = unsafe_alloc[Int64](2)  # struct timeval
    tv[unsafe_offset=0] = Int64(timeout_s)
    tv[unsafe_offset=1] = 0
    _ = libc.get_function[Int32]("setsockopt")(
        s, SOL_SOCKET, SO_SNDTIMEO, tv, Int32(16)
    )
    return s


def _fd_msghdr(
    sa: Pointer[UInt8, MutUntrackedOrigin], fd: Int, kind: Int, tag: Int
) -> Pointer[UInt64, MutUntrackedOrigin]:
    """`struct msghdr` (56 B) carrying one SCM_RIGHTS descriptor and an
    8-byte `(kind, tag)` payload, to the address in `sa`."""
    var payload = unsafe_alloc[UInt32](2)
    payload[unsafe_offset=0] = UInt32(kind)
    payload[unsafe_offset=1] = UInt32(tag)
    var iov = unsafe_alloc[UInt64](2)  # struct iovec
    iov[unsafe_offset=0] = UInt64(Int(payload))
    iov[unsafe_offset=1] = 8
    var ctl = unsafe_alloc[UInt64](3)  # CMSG_SPACE(sizeof(int)) == 24
    ctl[unsafe_offset=0] = 20  # cmsg_len = CMSG_LEN(4)
    ctl[unsafe_offset=1] = UInt64(SOL_SOCKET) | (UInt64(SCM_RIGHTS) << 32)
    ctl[unsafe_offset=2] = UInt64(UInt32(fd))  # the fd itself, at CMSG_DATA
    var msg = unsafe_alloc[UInt64](7)
    msg[unsafe_offset=0] = UInt64(Int(sa))  # msg_name
    msg[unsafe_offset=1] = 110  # msg_namelen (u32 at +8)
    msg[unsafe_offset=2] = UInt64(Int(iov))  # msg_iov
    msg[unsafe_offset=3] = 1  # msg_iovlen
    msg[unsafe_offset=4] = UInt64(Int(ctl))  # msg_control
    msg[unsafe_offset=5] = 24  # msg_controllen
    msg[unsafe_offset=6] = 0  # msg_flags
    return msg


def _send_once(
    libc: OwnedDLHandle, sock: Int32, msg: Pointer[UInt64, MutUntrackedOrigin]
) raises -> Int32:
    """One `MSG_DONTWAIT` sendmsg. 0 on success, else the errno -- EAGAIN for
    "the peer's queue is full", ENOENT/ECONNREFUSED for "it has not bound
    yet"; both are retried by the caller under its deadline."""
    if libc.get_function[Int]("sendmsg")(sock, msg, MSG_DONTWAIT) >= 0:
        return 0
    return _errno()


def scm_send(
    libc: OwnedDLHandle,
    path: String,
    fd: Int,
    kind: Int,
    tag: Int,
    timeout_s: Float64,
) raises:
    """Send `fd`, plus an 8-byte `(kind, tag)` payload, to the socket at
    `path`. Retries while the peer has not bound yet, or while its datagram
    queue is full, until `timeout_s` -- and never blocks in the syscall, so
    the deadline is real."""
    var s = _send_socket(libc, timeout_s)
    var msg = _fd_msghdr(_sockaddr(path), fd, kind, tag)
    var deadline = perf_counter_ns() + Int(timeout_s * 1.0e9)
    while True:
        if _send_once(libc, s, msg) == 0:
            break
        if perf_counter_ns() > deadline:
            _ = libc.get_function[Int32]("close")(s)
            raise Error("mojoccl: sendmsg to " + path + " timed out")
        sleep(0.005)
    _ = libc.get_function[Int32]("close")(s)


def scm_try_recv(libc: OwnedDLHandle, sock: Int) raises -> Tuple[Int, Int, Int]:
    """One non-blocking `recvmsg`. Returns `(-1, 0, 0)` when nothing is
    queued."""
    return _recv_impl(libc, sock, MSG_DONTWAIT)


def scm_recv(libc: OwnedDLHandle, sock: Int) raises -> Tuple[Int, Int, Int]:
    """Receive one fd and its `(kind, tag)`; returns `(fd, kind, tag)`.
    Blocking, bounded by the `SO_RCVTIMEO` `scm_bind` set."""
    return _recv_impl(libc, sock, Int32(0))


def _recv_impl(
    libc: OwnedDLHandle, sock: Int, flags: Int32
) raises -> Tuple[Int, Int, Int]:
    var payload = unsafe_alloc[UInt32](2)
    payload[unsafe_offset=0] = 0
    payload[unsafe_offset=1] = 0
    var iov = unsafe_alloc[UInt64](2)
    iov[unsafe_offset=0] = UInt64(Int(payload))
    iov[unsafe_offset=1] = 8
    var ctl = unsafe_alloc[UInt64](3)
    for i in range(3):
        ctl[unsafe_offset=i] = 0
    var msg = unsafe_alloc[UInt64](7)
    msg[unsafe_offset=0] = 0
    msg[unsafe_offset=1] = 0
    msg[unsafe_offset=2] = UInt64(Int(iov))
    msg[unsafe_offset=3] = 1
    msg[unsafe_offset=4] = UInt64(Int(ctl))
    msg[unsafe_offset=5] = 24
    msg[unsafe_offset=6] = 0
    if libc.get_function[Int]("recvmsg")(Int32(sock), msg, flags) < 0:
        if flags == MSG_DONTWAIT and _errno() == EAGAIN:
            return Tuple(-1, 0, 0)
        raise Error("mojoccl: recvmsg on the fd socket failed or timed out")
    if ctl[unsafe_offset=0] != 20:
        raise Error("mojoccl: datagram carried no SCM_RIGHTS control message")
    return Tuple(
        Int(UInt32(ctl[unsafe_offset=2] & 0xFFFFFFFF)),
        Int(payload[unsafe_offset=0]),
        Int(payload[unsafe_offset=1]),
    )


def scm_exchange_fds(
    libc: OwnedDLHandle,
    sock: Int,
    paths: List[String],
    my_fd: Int,
    kind: Int,
    tag: Int,
    timeout_s: Float64,
) raises -> List[Tuple[Int, Int, Int]]:
    """Send `my_fd` to every path AND receive one datagram from each, with the
    two interleaved.

    The all-to-all round of the NVLS bring-up cannot send everything first:
    `local_world` ranks each pushing `local_world - 1` datagrams before they
    read one can fill every queue at once, and on a host with a small
    `net.unix.max_dgram_qlen` a blocking sender then waits on a receiver who
    is itself blocked sending. Both halves here are one non-blocking attempt
    per turn, so a rank that cannot send drains its own queue instead -- and
    the absolute deadline covers the whole exchange, not one syscall.
    Returns `(fd, kind, tag)` per datagram; the caller validates and closes.
    """
    var send_sock = _send_socket(libc, timeout_s)
    var msgs = List[Pointer[UInt64, MutUntrackedOrigin]]()
    for i in range(len(paths)):
        msgs.append(_fd_msghdr(_sockaddr(paths[i]), my_fd, kind, tag))
    var got = List[Tuple[Int, Int, Int]]()
    var sent = 0
    var deadline = perf_counter_ns() + Int(timeout_s * 1.0e9)
    while sent < len(paths) or len(got) < len(paths):
        var moved = False
        if sent < len(paths):
            if _send_once(libc, send_sock, msgs[sent]) == 0:
                sent += 1
                moved = True
        if len(got) < len(paths):
            var r = scm_try_recv(libc, sock)
            if r[0] >= 0:
                got.append(r)
                moved = True
        if moved:
            continue
        if perf_counter_ns() > deadline:
            _ = libc.get_function[Int32]("close")(send_sock)
            for i in range(len(got)):
                _ = libc.get_function[Int32]("close")(Int32(got[i][0]))
            raise Error(
                "mojoccl: the fd exchange timed out after sending "
                + String(sent)
                + " and receiving "
                + String(len(got))
                + " of "
                + String(len(paths))
            )
        sleep(0.0005)
    _ = libc.get_function[Int32]("close")(send_sock)
    return got^


# ===-------------------------------------------------------------------=== #
# The region
# ===-------------------------------------------------------------------=== #


struct NvlsRegion(Movable):
    """One rank's slice of a node-wide multicast allocation.

    `mc` is the multicast VA -- only `multimem.*` may touch it. `uc` is a plain
    mapping of the same physical bytes and is what every unicast kernel, the
    staging copies and the flag spin use; it is also what `regions[local_rank]`
    holds, so collectives_kernels.mojo never learns that the region changed.
    """

    var mc: Int
    var uc: Int
    var size: Int
    var granularity: Int
    var mc_handle: UInt64
    var mem_handle: UInt64
    var peer_va: StaticTuple[Int, MAX_WORLD]
    var peer_handle: StaticTuple[UInt64, MAX_WORLD]

    def __init__(out self):
        self.mc = 0
        self.uc = 0
        self.size = 0
        self.granularity = 0
        self.mc_handle = 0
        self.mem_handle = 0
        self.peer_va = StaticTuple[Int, MAX_WORLD](fill=0)
        self.peer_handle = StaticTuple[UInt64, MAX_WORLD](fill=0)


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


def nvls_create_and_share(
    lib: OwnedDLHandle,
    libc: OwnedDLHandle,
    dir: String,
    magic: UInt64,
    local_rank: Int,
    local_world: Int,
    ordinal: Int,
    size: Int,
    gran: Int,
    sock: Int,
    timeout_s: Float64,
    mut region: NvlsRegion,
) raises:
    """Steps 1-3 of the bring-up: the multicast object exists on every rank and
    every device has joined it. The caller barriers after this, then calls
    `nvls_bind_and_map` -- the split is where the "every device in the team
    before any memory is bound" rule lives.
    """
    comptime if not NVLS_AVAILABLE:
        raise Error("mojoccl: NVLS is CUDA-only")
    region.size = size
    region.granularity = gran
    var mch: UInt64 = 0
    if local_rank == 0:
        _cu(
            lib,
            lib.get_function[Int32]("cuMulticastCreate")(
                Pointer(to=mch), _mc_prop(local_world, size)
            ),
            "cuMulticastCreate",
        )
        # Owned by the region from this line on, so a failure in the export
        # or the fd hand-off below is released by `nvls_teardown`.
        region.mc_handle = mch
        var fd: Int32 = -1
        _cu(
            lib,
            lib.get_function[Int32]("cuMemExportToShareableHandle")(
                Pointer(to=fd),
                mch,
                Int32(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                UInt64(0),
            ),
            "cuMemExportToShareableHandle(multicast)",
        )
        try:
            for r in range(1, local_world):
                scm_send(
                    libc,
                    socket_path(dir, magic, r),
                    Int(fd),
                    MSG_KIND_MC,
                    0,
                    timeout_s,
                )
        except e:
            _ = libc.get_function[Int32]("close")(fd)
            raise e
        _ = libc.get_function[Int32]("close")(fd)
    else:
        var got = scm_recv(libc, sock)
        if got[1] != MSG_KIND_MC:
            _ = libc.get_function[Int32]("close")(Int32(got[0]))
            raise Error(
                "mojoccl: expected the multicast fd, got kind " + String(got[1])
            )
        var rc = lib.get_function[Int32]("cuMemImportFromShareableHandle")(
            Pointer(to=mch),
            got[0],
            Int32(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
        )
        _ = libc.get_function[Int32]("close")(Int32(got[0]))
        _cu(lib, rc, "cuMemImportFromShareableHandle(multicast)")
        region.mc_handle = mch
    _cu(
        lib,
        lib.get_function[Int32]("cuMulticastAddDevice")(mch, Int32(ordinal)),
        "cuMulticastAddDevice",
    )


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
