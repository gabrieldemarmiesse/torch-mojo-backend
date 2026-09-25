# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/misc/socket.cc
#   also:     https://github.com/NVIDIA/nccl/blob/master/src/os/linux.cc

from std.ffi import external_call
from std.os import getenv, listdir
from std.time import perf_counter_ns

from tmb.ccl.env_vars import MOJOCCL_SOCKET_IFNAME
from tmb.ccl.misc.utils import _alloc


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
