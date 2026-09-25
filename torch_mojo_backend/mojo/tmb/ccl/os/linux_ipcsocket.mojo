# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/os/linux_ipcsocket.cc
#
# fd transport is SCM_RIGHTS over an AF_UNIX SOCK_DGRAM socket, what NCCL does
# (nccl:src/os/linux_ipcsocket.cc:171-245). NOT pidfd_getfd: that needs
# PTRACE_MODE_ATTACH, i.e. /proc/sys/kernel/yama/ptrace_scope <= 1, and it is 1
# on stock Ubuntu and 2 or 3 on hardened images -- a collective that silently
# loses NVLS to a sysctl is worse than 70 lines of hand-laid-out `sendmsg`.
# `std.ffi` has no C-struct ABI, so msghdr/cmsghdr/sockaddr_un are written out
# over UInt64 words, field offsets from the glibc headers.

from std.ffi import external_call, OwnedDLHandle
from std.time import perf_counter_ns, sleep
from std.memory.alloc import unsafe_alloc


comptime DEFAULT_SOCKET_DIR = "/tmp"
"""Where the node-local AF_UNIX sockets that carry the VMM/multicast file
descriptors are bound. Node-local by definition -- a
shared filesystem would work too but buys nothing, since the ranks that talk
over it are on one host. NCCL puts its own at /tmp as well
(nccl:src/os/linux_ipcsocket.cc)."""


def _socket_dir() -> String:
    return String(DEFAULT_SOCKET_DIR)


comptime LIBC = "libc.so.6"


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


def _errno() -> Int32:
    return external_call["__errno_location", Pointer[Int32, MutAnyOrigin]]()[
        unsafe_offset=0
    ]


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
