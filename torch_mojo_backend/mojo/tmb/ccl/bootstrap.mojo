# TCP socket bootstrap for mojoccl's ncclCommInitRank -- single node and
# multi node alike (it replaced the /dev/shm rendezvous directory, which
# could never span hosts).
#
# Shape copied from NCCL: the process that calls ncclGetUniqueId opens a
# listening socket and encodes {ipv4, port, random magic} in the 128-byte
# ncclUniqueId (nccl:src/bootstrap.cc:495-529 puts the root's
# ncclSocketAddress + a magic in the id the same way); that id then travels
# through the caller's OWN rendezvous -- here the c10d store in
# distributed/process_group.py, unchanged. Every rank's ncclCommInitRank
# decodes it, connects to the root, and the root relays.
#
# One primitive is enough: `bootstrap_allgather` of a fixed-size blob. Round
# 1 gathers host identity so every rank derives the same node/local_rank
# table by itself (no root-side decision to broadcast); round 2 gathers the
# IPC handle + IB connection data; round 3 is a 4-byte barrier that closes
# init. NCCL's root does one allgather too (`bootstrapAllGather`,
# nccl:src/bootstrap.cc:1080) -- it just runs a ring, which is not worth the
# code at 16 ranks of 256 bytes.

from std.ffi import _get_global_or_null, external_call
from std.memory.alloc import unsafe_alloc
from std.os import getenv, listdir
from std.random import random_ui64
from std.time import perf_counter_ns, sleep

from tmb.ccl.env_vars import MOJOCCL_SOCKET_IFNAME

comptime UID_BYTES = 128
comptime HANDLE_BYTES = 64

# "MOJOCCL2" -- the id format tag. Bumped from the /dev/shm-path id (which
# carried no tag at all), so a stale id from a mismatched build is rejected
# with a message instead of being parsed as an address.
comptime UID_TAG: UInt64 = 0x4D_4F_4A_4F_43_43_4C_32

# <sys/socket.h>, <netinet/in.h>, <netinet/tcp.h>, <sys/ioctl.h>: verified on
# this machine with a gcc offsetof/value dump rather than taken from memory.
comptime AF_INET: Int32 = 2
comptime SOCK_STREAM: Int32 = 1
comptime SOCK_DGRAM: Int32 = 2
comptime SOL_SOCKET: Int32 = 1
comptime SO_REUSEADDR: Int32 = 2
comptime SO_RCVTIMEO: Int32 = 20
comptime SO_SNDTIMEO: Int32 = 21
comptime IPPROTO_TCP: Int32 = 6
comptime TCP_NODELAY: Int32 = 1
comptime SIOCGIFADDR: UInt64 = 0x8915
comptime MSG_NOSIGNAL: Int32 = 0x4000
comptime EINTR: Int32 = 4
comptime EAGAIN: Int32 = 11
comptime ECONNREFUSED: Int32 = 111
comptime SO_ERROR: Int32 = 4
comptime EINPROGRESS: Int32 = 115
comptime EALREADY: Int32 = 114
comptime EISCONN: Int32 = 106
# SOCK_NONBLOCK, the Linux flag `socket(2)` and `accept4(2)` take in their
# type argument. Every socket this module opens carries it, so no blocking
# syscall can outlive the deadline the caller set: the wait happens in
# `poll(2)` with a computed timeout instead, and SO_RCVTIMEO/SO_SNDTIMEO stay
# on as a backstop for the cases poll cannot see (a socket wedged in the
# kernel, a revents race).
comptime SOCK_NONBLOCK: Int32 = 0x800
comptime POLLIN: Int32 = 1
comptime POLLOUT: Int32 = 4
# sizeof(struct sockaddr_in) == 16: sin_family u16 @0, sin_port u16 @2 (both
# network order), sin_addr u32 @4, 8 zero bytes. sizeof(struct ifreq) == 40:
# ifr_name[16] @0, then the sockaddr at 16 (so the IPv4 word is at 20).
comptime SOCKADDR_IN_BYTES = 16
comptime IFREQ_BYTES = 40

# Per-socket send/recv timeout. Short enough that a wedged peer is noticed
# quickly, long enough that a slice is not spent on syscall churn; the loops
# below re-arm it until the caller's own deadline expires.
comptime SOCK_SLICE_S: Int64 = 5


@always_inline
def _alloc[T: AnyType](n: Int) -> Pointer[T, MutAnyOrigin]:
    """`unsafe_alloc` with the origin the FFI signatures here declare.

    Every buffer this module allocates is small, lives for the duration of
    one `ncclCommInitRank`, and is handed to libc by address; rebinding to
    `MutAnyOrigin` once here keeps the call sites free of casts.
    """
    return Pointer[T, MutAnyOrigin](unsafe_from_address=Int(unsafe_alloc[T](n)))


def _errno() -> Int32:
    return external_call["__errno_location", Pointer[Int32, MutAnyOrigin]]()[
        unsafe_offset=0
    ]


def _close(fd: Int32):
    if fd >= 0:
        _ = external_call["close", Int32](fd)


def _set_timeout(fd: Int32, optname: Int32, seconds: Int64) raises:
    """SO_RCVTIMEO / SO_SNDTIMEO: struct timeval {i64 tv_sec, i64 tv_usec}."""
    var tv = _alloc[Int64](2)
    tv[unsafe_offset=0] = seconds
    tv[unsafe_offset=1] = 0
    var rc = external_call["setsockopt", Int32](
        fd, SOL_SOCKET, optname, tv, UInt32(16)
    )
    if rc != 0:
        raise Error(
            "mojoccl: setsockopt(timeout) failed, errno=" + String(_errno())
        )


def _get_int_opt(fd: Int32, level: Int32, optname: Int32) raises -> Int32:
    var v = _alloc[Int32](1)
    v[unsafe_offset=0] = 0
    var l = _alloc[UInt32](1)
    l[unsafe_offset=0] = 4
    if external_call["getsockopt", Int32](fd, level, optname, v, l) != 0:
        raise Error("mojoccl: getsockopt failed, errno=" + String(_errno()))
    return v[unsafe_offset=0]


def _wait_ready(fd: Int32, events: Int32, deadline_ns: Int) raises -> Bool:
    """Wait until `fd` is ready (or its peer hung up), bounded by an ABSOLUTE
    deadline. False means the deadline passed.

    This is what bounds the blocking, rather than a check after the syscall
    came back: `connect` on a dropped SYN retries inside the kernel for over
    two minutes, and by the time it returns, a shorter bootstrap deadline has
    long expired. `struct pollfd` is {i32 fd, i16 events, i16 revents}.
    """
    var pfd = _alloc[Int32](2)
    while True:
        var left = deadline_ns - perf_counter_ns()
        if left <= 0:
            return False
        var ms = left // 1_000_000 + 1
        if ms > Int(SOCK_SLICE_S) * 1000:
            ms = Int(SOCK_SLICE_S) * 1000
        pfd[unsafe_offset=0] = fd
        pfd[unsafe_offset=1] = events & 0xFFFF
        var rc = external_call["poll", Int32](pfd, UInt64(1), Int32(ms))
        if rc > 0:
            return True
        if rc < 0:
            var e = _errno()
            if e != EINTR:
                raise Error("mojoccl: poll() failed, errno=" + String(e))


def _connect_deadline(
    fd: Int32, sa: Pointer[UInt8, MutAnyOrigin], deadline_ns: Int
) raises -> Bool:
    """Non-blocking `connect`, `poll` for writability inside the deadline,
    then `SO_ERROR` for the verdict -- the three steps a bounded TCP connect
    needs. False means "not connected" (refused, unreachable, or out of
    time); the caller decides whether to retry or to give up."""
    var rc = external_call["connect", Int32](fd, sa, UInt32(SOCKADDR_IN_BYTES))
    if rc == 0:
        return True
    var e = _errno()
    if e == EISCONN:
        return True
    if e != EINPROGRESS and e != EALREADY and e != EINTR:
        return False
    if not _wait_ready(fd, POLLOUT, deadline_ns):
        return False
    return _get_int_opt(fd, SOL_SOCKET, SO_ERROR) == 0


def _set_int_opt(fd: Int32, level: Int32, optname: Int32, value: Int32) raises:
    var v = _alloc[Int32](1)
    v[unsafe_offset=0] = value
    var rc = external_call["setsockopt", Int32](
        fd, level, optname, v, UInt32(4)
    )
    if rc != 0:
        raise Error("mojoccl: setsockopt failed, errno=" + String(_errno()))


def _fill_sockaddr(
    sa: Pointer[UInt8, MutAnyOrigin], addr_be: UInt32, port_be: UInt16
):
    for i in range(SOCKADDR_IN_BYTES):
        sa[unsafe_offset=i] = 0
    # sin_family is host-order u16; AF_INET == 2 fits in the low byte on the
    # little-endian targets this library builds for.
    sa[unsafe_offset=0] = 2
    sa[unsafe_offset=1] = 0
    var p = sa.unsafe_bitcast[UInt16]()
    p[unsafe_offset=1] = port_be
    var a = sa.unsafe_bitcast[UInt32]()
    a[unsafe_offset=1] = addr_be


# ===-------------------------------------------------------------------=== #
# Local IPv4 discovery (NCCL_SOCKET_IFNAME's analogue)
# ===-------------------------------------------------------------------=== #


def _ipv4_of_iface(name: String) -> UInt32:
    """SIOCGIFADDR on a throwaway UDP socket; 0 if the interface has no IPv4."""
    if name.byte_length() == 0 or name.byte_length() > 15:
        return 0
    var fd = external_call["socket", Int32](AF_INET, SOCK_DGRAM, Int32(0))
    if fd < 0:
        return 0
    var req = _alloc[UInt8](IFREQ_BYTES)
    for i in range(IFREQ_BYTES):
        req[unsafe_offset=i] = 0
    var nb = name.as_bytes()
    for i in range(len(nb)):
        req[unsafe_offset=i] = nb[i]
    var rc = external_call["ioctl", Int32](fd, SIOCGIFADDR, req)
    _close(fd)
    if rc != 0:
        return 0
    return req.unsafe_bitcast[UInt32]()[unsafe_offset=5]  # byte 20


def _default_route_ifaces() -> List[String]:
    """Interfaces carrying a default route, from /proc/net/route (columns:
    Iface Destination Gateway ...; Destination 00000000 == default)."""
    var out = List[String]()
    var text: String
    try:
        with open("/proc/net/route", "r") as f:
            text = f.read()
    except:
        return out^
    var first = True
    for line in text.split("\n"):
        if first:
            first = False  # header
            continue
        var cols = List[String]()
        for c in String(line).split("\t"):
            if String(c).byte_length() > 0:
                cols.append(String(c))
        if len(cols) < 2:
            continue
        if cols[1] == "00000000":
            out.append(cols[0])
    return out^


def _up_ifaces() -> List[String]:
    var out = List[String]()
    var names: List[String]
    try:
        names = listdir("/sys/class/net")
    except:
        return out^
    for n in names:
        if String(n) == "lo":
            continue
        try:
            with open("/sys/class/net/" + String(n) + "/operstate", "r") as f:
                if String(f.read().strip()) != "up":
                    continue
        except:
            continue
        out.append(String(n))
    return out^


def local_ipv4() raises -> UInt32:
    """The address this rank publishes in the unique id, network order.

    MOJOCCL_SOCKET_IFNAME wins (NCCL_SOCKET_IFNAME's analogue, without the
    ^/= prefix grammar); otherwise the first UP non-loopback interface that
    carries a default route and has an IPv4 address; otherwise any UP
    non-loopback IPv4 interface.
    """
    var want = getenv(MOJOCCL_SOCKET_IFNAME, "")
    if want.byte_length() > 0:
        var a = _ipv4_of_iface(want)
        if a == 0:
            raise Error(
                "mojoccl: MOJOCCL_SOCKET_IFNAME="
                + want
                + " has no IPv4 address"
            )
        return a
    for n in _default_route_ifaces():
        var a = _ipv4_of_iface(String(n))
        if a != 0:
            return a
    for n in _up_ifaces():
        var a = _ipv4_of_iface(String(n))
        if a != 0:
            return a
    raise Error(
        "mojoccl: no usable IPv4 interface found; set MOJOCCL_SOCKET_IFNAME"
    )


def format_ipv4(addr_be: UInt32) -> String:
    var b = (addr_be >> 0) & 0xFF
    var c = (addr_be >> 8) & 0xFF
    var d = (addr_be >> 16) & 0xFF
    var e = (addr_be >> 24) & 0xFF
    return String(b) + "." + String(c) + "." + String(d) + "." + String(e)


# ===-------------------------------------------------------------------=== #
# Host identity
# ===-------------------------------------------------------------------=== #


def host_hash() -> UInt64:
    """A 64-bit id of this physical host, equal on every rank of a node and
    different across nodes. /etc/machine-id first (stable, always present on
    these images), boot_id next, hostname last."""
    var text = String("")
    for p in ["/etc/machine-id", "/proc/sys/kernel/random/boot_id"]:
        try:
            with open(String(p), "r") as f:
                text = String(f.read().strip())
            if text.byte_length() > 0:
                break
        except:
            continue
    if text.byte_length() == 0:
        var buf = _alloc[UInt8](256)
        for i in range(256):
            buf[unsafe_offset=i] = 0
        _ = external_call["gethostname", Int32](buf, UInt64(255))
        var n = 0
        while n < 255 and buf[unsafe_offset=n] != 0:
            n += 1
        text = String(capacity_bytes=n)
        for i in range(n):
            text += chr(Int(buf[unsafe_offset=i]))
    # FNV-1a: any stable 64-bit mix is fine, this one needs no table.
    var h: UInt64 = 0xCBF29CE484222325
    for byte in text.as_bytes():
        h = (h ^ UInt64(byte)) * 0x100000001B3
    return h


# ===-------------------------------------------------------------------=== #
# The 128-byte unique id
# ===-------------------------------------------------------------------=== #


def _encode_id(
    out_bytes: Pointer[UInt8, MutAnyOrigin],
    addr_be: UInt32,
    port_be: UInt16,
    magic: UInt64,
):
    for i in range(UID_BYTES):
        out_bytes[unsafe_offset=i] = 0
    var w = out_bytes.unsafe_bitcast[UInt64]()
    w[unsafe_offset=0] = UID_TAG
    w[unsafe_offset=1] = magic
    out_bytes.unsafe_bitcast[UInt32]()[unsafe_offset=4] = addr_be
    out_bytes.unsafe_bitcast[UInt16]()[unsafe_offset=10] = port_be


def decode_id(
    in_bytes: Pointer[UInt8, MutAnyOrigin],
) raises -> Tuple[UInt32, UInt16, UInt64]:
    """`(addr_be, port_be, magic)` out of a unique id."""
    var w = in_bytes.unsafe_bitcast[UInt64]()
    if w[unsafe_offset=0] != UID_TAG:
        raise Error(
            "mojoccl: unique id is not this library's (tag mismatch); a stale"
            " id or one from a different CCL implementation"
        )
    return Tuple(
        in_bytes.unsafe_bitcast[UInt32]()[unsafe_offset=4],
        in_bytes.unsafe_bitcast[UInt16]()[unsafe_offset=10],
        w[unsafe_offset=1],
    )


def _root_global_name(magic: UInt64) -> String:
    return String("MOJOCCL_ROOT_") + String(magic)


def _remember_root(magic: UInt64, fd: Int32):
    """The listening socket ncclGetUniqueId opened has to survive until this
    process's own ncclCommInitRank runs (a different call, no shared state
    otherwise). Stashed in the compiler-runtime global table, the same
    process-global mechanism collectives_kernels.mojo caches DeviceFunctions
    in, keyed by the id's magic so several communicators can be in flight."""
    var slot = _alloc[Int64](1)
    slot[unsafe_offset=0] = Int64(fd)
    var name = _root_global_name(magic)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(name), slot.unsafe_bitcast[NoneType]()
    )


def _take_root(magic: UInt64) -> Int32:
    """The listening fd for `magic` if this process created the id, else -1.
    Consumed: the slot is set to -1 so a second communicator on the same id
    cannot re-accept on it."""
    var name = _root_global_name(magic)
    var g = _get_global_or_null(name)
    if not g:
        return -1
    var slot = g.value().unsafe_bitcast[Int64]()
    var fd = Int32(slot[unsafe_offset=0])
    slot[unsafe_offset=0] = -1
    return fd


def make_unique_id(out_bytes: Pointer[UInt8, MutAnyOrigin]) raises:
    """Bind a listening socket and encode where it is (ncclGetUniqueId)."""
    var addr = local_ipv4()
    var fd = external_call["socket", Int32](
        AF_INET, SOCK_STREAM | SOCK_NONBLOCK, Int32(0)
    )
    if fd < 0:
        raise Error("mojoccl: socket() failed, errno=" + String(_errno()))
    try:
        _set_int_opt(fd, SOL_SOCKET, SO_REUSEADDR, 1)
        var sa = _alloc[UInt8](SOCKADDR_IN_BYTES)
        _fill_sockaddr(sa, addr, 0)  # port 0: let the kernel choose
        if external_call["bind", Int32](fd, sa, UInt32(SOCKADDR_IN_BYTES)) != 0:
            raise Error("mojoccl: bind() failed, errno=" + String(_errno()))
        # A backlog of 1024 covers every rank connecting at once (NCCL uses
        # 16384, nccl:src/misc/socket.cc:571; 1024 is far past 8 nodes x 8).
        if external_call["listen", Int32](fd, Int32(1024)) != 0:
            raise Error("mojoccl: listen() failed, errno=" + String(_errno()))
        var slen = _alloc[UInt32](1)
        slen[unsafe_offset=0] = UInt32(SOCKADDR_IN_BYTES)
        if external_call["getsockname", Int32](fd, sa, slen) != 0:
            raise Error("mojoccl: getsockname() failed")
        var port_be = sa.unsafe_bitcast[UInt16]()[unsafe_offset=1]
        var magic = random_ui64(1, UInt64.MAX)
        _encode_id(out_bytes, addr, port_be, magic)
        _remember_root(magic, fd)
    except e:
        _close(fd)
        raise e


# ===-------------------------------------------------------------------=== #
# The connection
# ===-------------------------------------------------------------------=== #


struct BootstrapConn(Movable):
    """Root: `fds[r]` is the socket to rank r (own slot -1) and `listen_fd`
    is still open. Non-root: `fds` holds the one socket to the root."""

    var rank: Int
    var nranks: Int
    var is_root: Bool
    var listen_fd: Int32
    var fds: List[Int32]

    def __init__(
        out self, rank: Int, nranks: Int, is_root: Bool, listen_fd: Int32
    ):
        self.rank = rank
        self.nranks = nranks
        self.is_root = is_root
        self.listen_fd = listen_fd
        self.fds = List[Int32](length=nranks if is_root else 1, fill=-1)

    def close(mut self):
        for i in range(len(self.fds)):
            _close(self.fds[i])
            self.fds[i] = -1
        _close(self.listen_fd)
        self.listen_fd = -1


def _send_all(
    fd: Int32, buf: Pointer[UInt8, MutAnyOrigin], nbytes: Int, deadline_ns: Int
) raises:
    var done = 0
    while done < nbytes:
        # Checked before every syscall, not only when one returns EAGAIN: a
        # peer trickling bytes makes progress forever otherwise.
        if not _wait_ready(fd, POLLOUT, deadline_ns):
            raise Error("mojoccl: bootstrap send timed out")
        var n = external_call["send", Int64](
            fd,
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(buf) + done),
            UInt64(nbytes - done),
            MSG_NOSIGNAL,
        )
        if n > 0:
            done += Int(n)
            continue
        var e = _errno()
        if n < 0 and (e == EINTR or e == EAGAIN):
            if perf_counter_ns() > deadline_ns:
                raise Error("mojoccl: bootstrap send timed out")
            continue
        raise Error("mojoccl: bootstrap send failed, errno=" + String(e))


def _recv_all(
    fd: Int32, buf: Pointer[UInt8, MutAnyOrigin], nbytes: Int, deadline_ns: Int
) raises:
    var done = 0
    while done < nbytes:
        if not _wait_ready(fd, POLLIN, deadline_ns):
            raise Error("mojoccl: bootstrap recv timed out")
        var n = external_call["recv", Int64](
            fd,
            Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(buf) + done),
            UInt64(nbytes - done),
            Int32(0),
        )
        if n > 0:
            done += Int(n)
            continue
        if n == 0:
            raise Error("mojoccl: bootstrap peer closed the connection")
        var e = _errno()
        if e == EINTR or e == EAGAIN:
            if perf_counter_ns() > deadline_ns:
                raise Error("mojoccl: bootstrap recv timed out")
            continue
        raise Error("mojoccl: bootstrap recv failed, errno=" + String(e))


def bootstrap_connect(
    uid: Pointer[UInt8, MutAnyOrigin],
    rank: Int,
    nranks: Int,
    timeout_s: Float64,
) raises -> BootstrapConn:
    """Every rank's first act in ncclCommInitRank: reach the root.

    Non-root ranks connect and announce {magic, rank}; the root accepts
    `nranks - 1` of those and files each fd by the rank it announced. A
    mismatched magic is refused rather than trusted -- two communicators
    bootstrapping at once on one node would otherwise cross wires.
    """
    var decoded = decode_id(uid)
    var addr_be = decoded[0]
    var port_be = decoded[1]
    var magic = decoded[2]
    var deadline_ns = perf_counter_ns() + Int(timeout_s * 1.0e9)
    var listen_fd = _take_root(magic)
    var conn = BootstrapConn(rank, nranks, listen_fd >= 0, listen_fd)
    try:
        _rendezvous(
            conn, rank, nranks, addr_be, port_be, magic, deadline_ns, timeout_s
        )
    except e:
        # Every exit that is not the happy one closes the sockets here: this
        # function's caller never sees the `BootstrapConn` when it raises, so
        # a listener left open holds the unique id's port for the life of the
        # process and a per-rank socket leaves its peer blocked until its own
        # deadline instead of failing fast on a closed connection.
        conn.close()
        raise e
    return conn^


def _rendezvous(
    mut conn: BootstrapConn,
    rank: Int,
    nranks: Int,
    addr_be: UInt32,
    port_be: UInt16,
    magic: UInt64,
    deadline_ns: Int,
    timeout_s: Float64,
) raises:
    """`bootstrap_connect`'s body, split out so one `except` closes every
    socket on every failure path."""
    var hello = _alloc[UInt8](16)
    var hw = hello.unsafe_bitcast[UInt64]()

    if conn.is_root:
        _set_timeout(conn.listen_fd, SO_RCVTIMEO, SOCK_SLICE_S)
        var got = 0
        while got < nranks - 1:
            if not _wait_ready(conn.listen_fd, POLLIN, deadline_ns):
                raise Error(
                    "mojoccl: bootstrap timed out with "
                    + String(got + 1)
                    + " of "
                    + String(nranks)
                    + " ranks connected"
                )
            var cfd = external_call["accept4", Int32](
                conn.listen_fd, Int64(0), Int64(0), SOCK_NONBLOCK
            )
            if cfd < 0:
                var e = _errno()
                if e == EINTR or e == EAGAIN:
                    continue
                raise Error("mojoccl: accept4() failed, errno=" + String(e))
            try:
                _set_timeout(cfd, SO_RCVTIMEO, SOCK_SLICE_S)
                _set_timeout(cfd, SO_SNDTIMEO, SOCK_SLICE_S)
                _set_int_opt(cfd, IPPROTO_TCP, TCP_NODELAY, 1)
                _recv_all(cfd, hello, 16, deadline_ns)
            except e:
                _close(cfd)
                raise e
            var peer_magic = hw[unsafe_offset=0]
            var peer_rank = Int(hw[unsafe_offset=1])
            if peer_magic != magic or peer_rank < 0 or peer_rank >= nranks:
                _close(cfd)
                continue  # not ours: another communicator's straggler
            if conn.fds[peer_rank] >= 0:
                _close(cfd)
                raise Error(
                    "mojoccl: two ranks announced rank " + String(peer_rank)
                )
            conn.fds[peer_rank] = cfd
            got += 1
        return

    # Non-root: connect, retrying while the root is still coming up.
    while True:
        var fd = external_call["socket", Int32](
            AF_INET, SOCK_STREAM | SOCK_NONBLOCK, Int32(0)
        )
        if fd < 0:
            raise Error("mojoccl: socket() failed, errno=" + String(_errno()))
        var sa = _alloc[UInt8](SOCKADDR_IN_BYTES)
        _fill_sockaddr(sa, addr_be, port_be)
        var connected = False
        try:
            connected = _connect_deadline(fd, sa, deadline_ns)
        except e:
            _close(fd)
            raise e
        if connected:
            try:
                _set_timeout(fd, SO_RCVTIMEO, SOCK_SLICE_S)
                _set_timeout(fd, SO_SNDTIMEO, SOCK_SLICE_S)
                _set_int_opt(fd, IPPROTO_TCP, TCP_NODELAY, 1)
            except e:
                _close(fd)
                raise e
            conn.fds[0] = fd
            hw[unsafe_offset=0] = magic
            hw[unsafe_offset=1] = UInt64(rank)
            _send_all(fd, hello, 16, deadline_ns)
            return
        var err = _errno()
        _close(fd)
        if perf_counter_ns() > deadline_ns:
            raise Error(
                "mojoccl: could not reach the bootstrap root at "
                + format_ipv4(addr_be)
                + ":"
                + String(Int(port_be >> 8) | (Int(port_be & 0xFF) << 8))
                + " within "
                + String(timeout_s)
                + "s (errno="
                + String(err)
                + "); check MOJOCCL_SOCKET_IFNAME reaches every node"
            )
        sleep(0.02)


def bootstrap_allgather(
    mut conn: BootstrapConn,
    blob: Pointer[UInt8, MutAnyOrigin],
    nbytes: Int,
    all_out: Pointer[UInt8, MutAnyOrigin],
    timeout_s: Float64,
) raises:
    """`all_out[r*nbytes : (r+1)*nbytes]` gets rank r's `blob`, on every rank.

    Root-relayed: N-1 sends in, one N*nbytes table out. At 16 ranks x 256 B
    that is 4 KiB per rank, three times per communicator.
    """
    var deadline_ns = perf_counter_ns() + Int(timeout_s * 1.0e9)
    var total = conn.nranks * nbytes
    if conn.is_root:
        for i in range(nbytes):
            all_out[unsafe_offset=conn.rank * nbytes + i] = blob[
                unsafe_offset=i
            ]
        for r in range(conn.nranks):
            if r == conn.rank:
                continue
            _recv_all(
                conn.fds[r],
                Pointer[UInt8, MutAnyOrigin](
                    unsafe_from_address=Int(all_out) + r * nbytes
                ),
                nbytes,
                deadline_ns,
            )
        for r in range(conn.nranks):
            if r != conn.rank:
                _send_all(conn.fds[r], all_out, total, deadline_ns)
    else:
        _send_all(conn.fds[0], blob, nbytes, deadline_ns)
        _recv_all(conn.fds[0], all_out, total, deadline_ns)


def bootstrap_barrier(mut conn: BootstrapConn, timeout_s: Float64) raises:
    """A 4-byte allgather used for what its bytes are not: everyone has
    arrived. Init's last step -- after it, a peer's region is zeroed, its IPC
    handles are open and its QPs are in RTS, so the first collective may
    write flags into it."""
    var one = _alloc[UInt8](4)
    for i in range(4):
        one[unsafe_offset=i] = UInt8(1)
    var all = _alloc[UInt8](4 * conn.nranks)
    bootstrap_allgather(conn, one, 4, all, timeout_s)


# ===-------------------------------------------------------------------=== #
# Topology, derived identically on every rank from the round-1 table
# ===-------------------------------------------------------------------=== #


struct Topology(Movable):
    var nnodes: Int
    var local_world: Int
    var my_node: Int
    var my_local_rank: Int
    var node_of: List[Int]  # global rank -> node index
    var local_rank_of: List[Int]  # global rank -> local rank
    var rank_at: List[Int]  # node * local_world + local_rank -> global rank

    def __init__(out self, nranks: Int):
        self.nnodes = 1
        self.local_world = nranks
        self.my_node = 0
        self.my_local_rank = 0
        self.node_of = List[Int](length=nranks, fill=0)
        self.local_rank_of = List[Int](length=nranks, fill=0)
        self.rank_at = List[Int](length=nranks, fill=0)


def derive_topology(host_hashes: List[UInt64], rank: Int) raises -> Topology:
    """Node index = order of first appearance scanning ranks 0..n-1; local
    rank = how many earlier ranks share the host. Pure function of the
    gathered table, so every rank computes the same answer with no round
    trip -- and the DDP invariant (equal ranks per node) is checked here,
    where the error message can name the offending node."""
    var n = len(host_hashes)
    var topo = Topology(n)
    var node_hash = List[UInt64]()
    topo.nnodes = 0
    for r in range(n):
        var h = host_hashes[r]
        var idx = -1
        for j in range(len(node_hash)):
            if node_hash[j] == h:
                idx = j
                break
        if idx < 0:
            idx = len(node_hash)
            node_hash.append(h)
        topo.node_of[r] = idx
    topo.nnodes = len(node_hash)
    var counts = List[Int](length=topo.nnodes, fill=0)
    for r in range(n):
        var nd = topo.node_of[r]
        topo.local_rank_of[r] = counts[nd]
        counts[nd] += 1
    topo.local_world = counts[0]
    for j in range(topo.nnodes):
        if counts[j] != topo.local_world:
            raise Error(
                "mojoccl: every node must contribute the same number of ranks"
                " (node 0 has "
                + String(topo.local_world)
                + ", node "
                + String(j)
                + " has "
                + String(counts[j])
                + "); launch with a uniform --nproc-per-node"
            )
    topo.my_node = topo.node_of[rank]
    topo.my_local_rank = topo.local_rank_of[rank]
    topo.rank_at = List[Int](length=topo.nnodes * topo.local_world, fill=0)
    for r in range(n):
        topo.rank_at[
            topo.node_of[r] * topo.local_world + topo.local_rank_of[r]
        ] = r
    return topo^
