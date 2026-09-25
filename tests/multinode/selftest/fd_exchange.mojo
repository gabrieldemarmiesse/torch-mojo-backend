# The AF_UNIX SCM_RIGHTS fd transport, without a GPU.
#
# `transport/multicast.mojo` and `transport/nvls.mojo` hand the multicast
# object and every rank's own VMM handle to its node-mates as file
# descriptors over an AF_UNIX SOCK_DGRAM socket, with the
# msghdr / cmsghdr / sockaddr_un structs laid out by hand over UInt64 words
# because `std.ffi` has no C-struct ABI. Getting one of those offsets wrong
# does not fail loudly -- `sendmsg` succeeds and the control message is
# dropped, or an fd arrives that belongs to something else -- so it is worth a
# test that needs neither CUDA nor a multicast-capable machine.
#
# This runs the shipped functions (`socket_path`, `scm_bind`, `scm_send`,
# `scm_recv`, `scm_exchange_fds`, `scm_unbind`) over ordinary file descriptors
# and checks that what the receiver reads THROUGH the descriptor is what the
# sender wrote, in the same rounds and with the same (kind, tag) dispatch
# production uses: one-to-all for the multicast handle, then all-to-all for
# the unicast ones -- round 2 sending everything before receiving anything,
# round 3 through `scm_exchange_fds`, which interleaves the two and is what
# `nvls_bind_and_map` actually calls. Round 3 is the shape that survives a
# host whose `net.unix.max_dgram_qlen` is smaller than `local_world`; that a
# full queue makes the send fail rather than block is checked, deterministic
# and peer-free, by sock_deadline.mojo.
#
#   uv run --no-sync mojo build tests/multinode/selftest/fd_exchange.mojo \
#       -I torch_mojo_backend/mojo -o /tmp/fd_exchange
#   for r in 0 1 2 3 4 5 6 7; do /tmp/fd_exchange $r 8 /tmp/fdx 1234 & done; wait

from std.ffi import OwnedDLHandle
from std.memory.alloc import unsafe_alloc
from std.pathlib import Path
from std.sys import argv
from std.time import perf_counter_ns, sleep

from tmb.ccl.transport.multicast import MSG_KIND_MC, MSG_KIND_UC
from tmb.ccl.os.linux_ipcsocket import (
    scm_bind,
    scm_exchange_fds,
    scm_recv,
    scm_send,
    scm_unbind,
    socket_path,
)

comptime TIMEOUT_S: Float64 = 30.0
comptime PAYLOAD = 64


def _barrier(dir: String, tag: String, rank: Int, world: Int) raises:
    """File rendezvous. Deliberately not the socket path: `open()` on a bound
    unix socket fails with ENXIO, which is how the prototype first lost an
    afternoon."""
    with open(dir + "/" + tag + String(rank), "w") as f:
        f.write(String("1"))
    var t0 = perf_counter_ns()
    for r in range(world):
        while not Path(dir + "/" + tag + String(r)).exists():
            sleep(0.01)
            if perf_counter_ns() - t0 > 60_000_000_000:
                raise Error("barrier " + tag + " timed out")


def _make_fd(libc: OwnedDLHandle, path: String, fill: Int) raises -> Int:
    """A file of `PAYLOAD` bytes of `fill`, opened read-only; its fd is what
    travels. `fill` stays ASCII: `chr` encodes as UTF-8, so 0xA5 would be
    written as two bytes and the check would fail on the test, not on the
    transport."""
    var buf = String("")
    for _ in range(PAYLOAD):
        buf += chr(fill)
    with open(path, "w") as f:
        f.write(buf)
    var cpath = unsafe_alloc[UInt8](len(path.as_bytes()) + 1)
    for i in range(len(path.as_bytes())):
        cpath[unsafe_offset=i] = path.as_bytes()[i]
    cpath[unsafe_offset=len(path.as_bytes())] = 0
    var fd = libc.get_function[Int32]("open")(cpath, Int32(0))  # O_RDONLY
    if fd < 0:
        raise Error("open(" + path + ") failed")
    return Int(fd)


def _read_fd(libc: OwnedDLHandle, fd: Int) raises -> Int:
    """First byte behind `fd`, read through the descriptor itself."""
    var buf = unsafe_alloc[UInt8](PAYLOAD)
    buf[unsafe_offset=0] = 0
    var n = libc.get_function[Int]("pread")(Int32(fd), buf, PAYLOAD, Int(0))
    if n <= 0:
        raise Error("pread through the received fd returned " + String(n))
    return Int(buf[unsafe_offset=0])


def main() raises:
    var a = argv()
    var rank = Int(String(a[1]))
    var world = Int(String(a[2]))
    var dir = String(a[3])
    var magic = UInt64(Int(String(a[4])))
    var libc = OwnedDLHandle("libc.so.6")
    var bad = 0

    var spath = socket_path(dir, magic, rank)
    var sock = scm_bind(libc, spath, TIMEOUT_S)
    _barrier(dir, "bound", rank, world)

    # Round 1: rank 0's "multicast" fd to everyone, one to all.
    comptime MC_FILL = 0x5A  # "Z"
    if rank == 0:
        var fd = _make_fd(libc, dir + "/mcfile", MC_FILL)
        for r in range(1, world):
            scm_send(
                libc, socket_path(dir, magic, r), fd, MSG_KIND_MC, 0, TIMEOUT_S
            )
        _ = libc.get_function[Int32]("close")(Int32(fd))
    else:
        var got = scm_recv(libc, sock)
        if got[1] != MSG_KIND_MC:
            print("rank", rank, "round 1 kind", got[1], "expected", MSG_KIND_MC)
            bad += 1
        var v = _read_fd(libc, got[0])
        if v != MC_FILL:
            print("rank", rank, "round 1 payload", v, "expected", MC_FILL)
            bad += 1
        _ = libc.get_function[Int32]("close")(Int32(got[0]))
    _barrier(dir, "round1", rank, world)

    # Round 2: every rank's own fd to every other rank, all to all, with the
    # sender's identity carried in the tag -- datagrams from several senders
    # arrive in an arbitrary order, so the tag is the only thing that says
    # whose region a descriptor is.
    var mine = _make_fd(libc, dir + "/ucfile" + String(rank), 0x41 + rank)
    for r in range(world):
        if r == rank:
            continue
        scm_send(
            libc, socket_path(dir, magic, r), mine, MSG_KIND_UC, rank, TIMEOUT_S
        )
    var seen = List[Int](length=world, fill=0)
    for _ in range(world - 1):
        var got = scm_recv(libc, sock)
        if got[1] != MSG_KIND_UC or got[2] < 0 or got[2] >= world:
            print("rank", rank, "round 2 kind", got[1], "tag", got[2])
            bad += 1
            continue
        var v = _read_fd(libc, got[0])
        if v != 0x41 + got[2]:
            print("rank", rank, "got", v, "from tag", got[2])
            bad += 1
        seen[got[2]] += 1
        _ = libc.get_function[Int32]("close")(Int32(got[0]))
    for r in range(world):
        var want = 0 if r == rank else 1
        if seen[r] != want:
            print("rank", rank, "saw", seen[r], "descriptors from", r)
            bad += 1
    _barrier(dir, "round2", rank, world)

    # Round 3: the same all-to-all through the shipped helper, which sends and
    # receives at the same time. `_barrier` first, so every rank is listening
    # before any starts -- the helper's own deadline is what covers a peer
    # that never binds, and that case belongs to sock_deadline.mojo, not here.
    var paths = List[String]()
    for r in range(world):
        if r != rank:
            paths.append(socket_path(dir, magic, r))
    var got3 = scm_exchange_fds(
        libc, sock, paths, mine, MSG_KIND_UC, rank, TIMEOUT_S
    )
    if len(got3) != world - 1:
        print("rank", rank, "round 3 got", len(got3), "want", world - 1)
        bad += 1
    var seen3 = List[Int](length=world, fill=0)
    for i in range(len(got3)):
        var g = got3[i]
        if g[1] != MSG_KIND_UC or g[2] < 0 or g[2] >= world:
            print("rank", rank, "round 3 kind", g[1], "tag", g[2])
            bad += 1
            continue
        var v = _read_fd(libc, g[0])
        if v != 0x41 + g[2]:
            print("rank", rank, "round 3 got", v, "from tag", g[2])
            bad += 1
        seen3[g[2]] += 1
        _ = libc.get_function[Int32]("close")(Int32(g[0]))
    for r in range(world):
        var want3 = 0 if r == rank else 1
        if seen3[r] != want3:
            print("rank", rank, "round 3 saw", seen3[r], "from", r)
            bad += 1

    _ = libc.get_function[Int32]("close")(Int32(mine))

    scm_unbind(libc, sock, spath)
    if bad == 0:
        print("rank", rank, "PASS")
    else:
        print("rank", rank, "FAIL", bad)
        raise Error("fd_exchange failed on rank " + String(rank))
