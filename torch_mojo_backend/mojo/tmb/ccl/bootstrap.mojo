# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/bootstrap.cc
#
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
from std.random import random_ui64
from std.time import perf_counter_ns, sleep

from tmb.ccl.misc.socket import (
    _fill_sockaddr,
    _recv_all,
    _send_all,
    format_ipv4,
    local_ipv4,
)
from tmb.ccl.misc.utils import _alloc
from tmb.ccl.nccl import UID_BYTES
from tmb.ccl.os.linux import (
    AF_INET,
    EAGAIN,
    EINTR,
    IPPROTO_TCP,
    POLLIN,
    SOCKADDR_IN_BYTES,
    SOCK_NONBLOCK,
    SOCK_SLICE_S,
    SOCK_STREAM,
    SOL_SOCKET,
    SO_RCVTIMEO,
    SO_REUSEADDR,
    SO_SNDTIMEO,
    TCP_NODELAY,
    _close,
    _connect_deadline,
    _errno,
    _set_int_opt,
    _set_timeout,
    _wait_ready,
)


comptime HANDLE_BYTES = 64

# "MOJOCCL2" -- the id format tag. Bumped from the /dev/shm-path id (which
# carried no tag at all), so a stale id from a mismatched build is rejected
# with a message instead of being parsed as an address.
comptime UID_TAG: UInt64 = 0x4D_4F_4A_4F_43_43_4C_32


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
    process-global mechanism device/common.mojo caches DeviceFunctions
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
