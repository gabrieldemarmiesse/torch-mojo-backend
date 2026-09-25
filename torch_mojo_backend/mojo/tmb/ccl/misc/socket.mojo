# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/misc/socket.cc

from std.ffi import external_call
from std.os import getenv
from std.time import perf_counter_ns

from tmb.ccl.env_vars import MOJOCCL_SOCKET_IFNAME
from tmb.ccl.os.linux import (
    EAGAIN,
    EINTR,
    MSG_NOSIGNAL,
    POLLIN,
    POLLOUT,
    SOCKADDR_IN_BYTES,
    _default_route_ifaces,
    _errno,
    _ipv4_of_iface,
    _up_ifaces,
    _wait_ready,
)


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
