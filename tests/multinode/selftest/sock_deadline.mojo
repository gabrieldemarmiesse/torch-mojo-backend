# The bootstrap and fd-transport deadlines, without a GPU, without IB and
# without a peer.
#
# Both transports wait on sockets, and both used to run a BLOCKING syscall
# first and check the deadline only after it returned. A dropped SYN keeps
# `connect` inside the kernel for over two minutes whatever
# MOJOCCL_BOOTSTRAP_TIMEOUT_S says, and a `sendmsg` into a full AF_UNIX
# datagram queue never returns at all -- which is how eight ranks that all
# send before they receive wedge an NVLS bring-up on a host with a small
# `net.unix.max_dgram_qlen`.
#
# Every case here is "this call must FAIL, and it must fail near its
# deadline": too fast is a bug (the deadline is not being honoured), too slow
# is the bug these cases exist for (a syscall the deadline cannot reach).
# Nothing needs a peer -- the peers are deliberately absent -- so this runs
# anywhere, in one process, in a few seconds.
#
#   uv run --no-sync mojo build tests/multinode/selftest/sock_deadline.mojo \
#       -I torch_mojo_backend/mojo -o /tmp/sock_deadline
#   /tmp/sock_deadline

from std.ffi import OwnedDLHandle, external_call
from std.memory.alloc import unsafe_alloc
from std.random import random_ui64
from std.time import perf_counter_ns

from tmb.ccl.nccl import UID_BYTES
from tmb.ccl.bootstrap import _encode_id, bootstrap_connect, make_unique_id
from tmb.ccl.misc.socket import local_ipv4
from tmb.ccl.transport.multicast import MSG_KIND_UC
from tmb.ccl.os.linux_ipcsocket import (
    _fd_msghdr,
    _send_once,
    _send_socket,
    _sockaddr,
    scm_bind,
    scm_exchange_fds,
    scm_send,
    scm_unbind,
    socket_path,
)

comptime DEADLINE_S: Float64 = 1.5
"""Short on purpose: every case below is expected to reach it."""

comptime SLACK_S: Float64 = 5.0
"""How far past the deadline a bounded failure may still land. Generous --
the point is to separate "bounded" from "blocked in a syscall for minutes",
not to measure the scheduler."""

comptime BLACKHOLE_IP = "203.0.113.9"
"""TEST-NET-3 (RFC 5737): reserved for documentation, so nothing routes it.
Whether the SYN is dropped or refused depends on the network in front of the
test, and either is fine here: `bootstrap_connect` retries until its deadline
in both cases, so the expected elapsed time is the same one."""


def _elapsed_s(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1.0e9


def _check_bounded(mut bad: Int, name: String, t: Float64):
    """A case passes when it came back, and came back near the deadline."""
    if t < DEADLINE_S * 0.5:
        print("FAIL", name, "returned after", t, "s, before its deadline")
        bad += 1
    elif t > DEADLINE_S + SLACK_S:
        print("FAIL", name, "took", t, "s -- the deadline did not bound it")
        bad += 1
    else:
        print("ok  ", name, t, "s")


def _ipv4_be(dotted: String) raises -> UInt32:
    """`inet_addr(3)`: a dotted quad as the network-order u32 the unique id
    carries."""
    var b = dotted.as_bytes()
    var c = unsafe_alloc[UInt8](len(b) + 1)
    for i in range(len(b)):
        c[unsafe_offset=i] = b[i]
    c[unsafe_offset=len(b)] = 0
    return external_call["inet_addr", UInt32](c)


def _closed_port_be() raises -> UInt16:
    """A port nothing listens on: bind one, read it back, close it."""
    comptime AF_INET: Int32 = 2
    comptime SOCK_STREAM: Int32 = 1
    var fd = external_call["socket", Int32](AF_INET, SOCK_STREAM, Int32(0))
    if fd < 0:
        raise Error("socket() failed")
    var sa = unsafe_alloc[UInt8](16)
    for i in range(16):
        sa[unsafe_offset=i] = 0
    sa[unsafe_offset=0] = 2
    if external_call["bind", Int32](fd, sa, UInt32(16)) != 0:
        _ = external_call["close", Int32](fd)
        raise Error("bind() failed")
    var slen = unsafe_alloc[UInt32](1)
    slen[unsafe_offset=0] = 16
    if external_call["getsockname", Int32](fd, sa, slen) != 0:
        _ = external_call["close", Int32](fd)
        raise Error("getsockname() failed")
    var port = sa.unsafe_bitcast[UInt16]()[unsafe_offset=1]
    _ = external_call["close", Int32](fd)
    return port


def _uid_pointing_at(addr_be: UInt32, port_be: UInt16) -> Int:
    """A well-formed unique id for an endpoint of this test's choosing. The
    magic is random and belongs to no listener in this process, so
    `bootstrap_connect` takes the non-root path and really connects."""
    var uid = unsafe_alloc[UInt8](UID_BYTES)
    var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(uid))
    _encode_id(p, addr_be, port_be, random_ui64(1, UInt64.MAX))
    return Int(uid)


def _case_root_never_reached(mut bad: Int) raises:
    """Rank 0 of a two-rank communicator, with rank 1 never launched: the
    accept loop must give up at the deadline."""
    var uid = unsafe_alloc[UInt8](UID_BYTES)
    var p = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(uid))
    make_unique_id(p)
    var t0 = perf_counter_ns()
    try:
        var conn = bootstrap_connect(p, 0, 2, DEADLINE_S)
        conn.close()
        print("FAIL root-never-reached returned a connection")
        bad += 1
        return
    except:
        pass
    _check_bounded(bad, "root: peer never connects", _elapsed_s(t0))


def _case_connect(
    mut bad: Int, name: String, addr_be: UInt32, port: UInt16
) raises:
    var p = Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=_uid_pointing_at(addr_be, port)
    )
    var t0 = perf_counter_ns()
    try:
        var conn = bootstrap_connect(p, 1, 2, DEADLINE_S)
        conn.close()
        print("FAIL", name, "connected to an endpoint that does not exist")
        bad += 1
        return
    except:
        pass
    _check_bounded(bad, name, _elapsed_s(t0))


def _fill_queue(libc: OwnedDLHandle, path: String) raises -> Int:
    """Fill `path`'s receive queue to `net.unix.max_dgram_qlen`, the state a
    node with few free slots is in when eight ranks all send at once.

    It takes several sending sockets: a datagram in flight is charged to its
    SENDER's send buffer, so one socket runs out (a few hundred here) long
    before the receiver's queue does. A brand-new socket that cannot push a
    single datagram is the signal that the limit reached is the receiver's.
    The sockets are deliberately left open -- closing them would give the
    charge back and drain the very condition the case needs.
    """
    var fd = libc.get_function[Int32]("dup")(Int32(0))
    var n = 0
    for _ in range(16):
        var sock = _send_socket(libc, 1.0)
        var msg = _fd_msghdr(_sockaddr(path), Int(fd), MSG_KIND_UC, 0)
        var pushed = 0
        while pushed < 100000 and _send_once(libc, sock, msg) == 0:
            pushed += 1
        n += pushed
        if pushed == 0:
            break
    _ = libc.get_function[Int32]("close")(fd)
    return n


def _case_full_queue(mut bad: Int) raises:
    """`scm_send` into a queue nobody drains. It must time out; a blocking
    `sendmsg` here waits for room that is never coming."""
    var libc = OwnedDLHandle("libc.so.6")
    var magic = random_ui64(1, UInt64.MAX)
    var path = socket_path("/tmp", magic, 0)
    var sock = scm_bind(libc, path, DEADLINE_S)
    var filled = _fill_queue(libc, path)
    if filled == 0:
        print("FAIL full-queue could not fill the receive queue at all")
        bad += 1
        scm_unbind(libc, sock, path)
        return
    var fd = libc.get_function[Int32]("dup")(Int32(0))
    var t0 = perf_counter_ns()
    try:
        scm_send(libc, path, Int(fd), MSG_KIND_UC, 0, DEADLINE_S)
        print("FAIL full-queue accepted a send into a full queue")
        bad += 1
    except:
        _check_bounded(
            bad,
            "scm_send: receiver queue full (" + String(filled) + ")",
            _elapsed_s(t0),
        )
    _ = libc.get_function[Int32]("close")(fd)
    scm_unbind(libc, sock, path)


def _case_exchange_no_peer(mut bad: Int) raises:
    """`scm_exchange_fds` with a peer that never binds its socket: bounded,
    and it says how far it got."""
    var libc = OwnedDLHandle("libc.so.6")
    var magic = random_ui64(1, UInt64.MAX)
    var path = socket_path("/tmp", magic, 0)
    var sock = scm_bind(libc, path, DEADLINE_S)
    var paths = List[String]()
    paths.append(socket_path("/tmp", magic, 1))  # nobody ever binds this
    var fd = libc.get_function[Int32]("dup")(Int32(0))
    var t0 = perf_counter_ns()
    try:
        var got = scm_exchange_fds(
            libc, sock, paths, Int(fd), MSG_KIND_UC, 0, DEADLINE_S
        )
        print("FAIL exchange-no-peer returned", len(got), "descriptors")
        bad += 1
    except:
        _check_bounded(
            bad, "scm_exchange_fds: peer never binds", _elapsed_s(t0)
        )
    _ = libc.get_function[Int32]("close")(fd)
    scm_unbind(libc, sock, path)


def main() raises:
    var bad = 0
    _case_root_never_reached(bad)
    _case_connect(
        bad, "connect: nothing listening", local_ipv4(), _closed_port_be()
    )
    _case_connect(
        bad,
        "connect: unroutable address",
        _ipv4_be(BLACKHOLE_IP),
        UInt16(0x3930),  # port 12345, network order
    )
    _case_full_queue(bad)
    _case_exchange_no_peer(bad)
    if bad == 0:
        print("sock_deadline PASS")
    else:
        print("sock_deadline FAIL", bad)
        raise Error("sock_deadline: " + String(bad) + " case(s) failed")
