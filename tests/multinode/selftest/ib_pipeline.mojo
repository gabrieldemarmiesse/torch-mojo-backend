# Self-test for the pipelined transport: several exchanges in flight at once,
# inbox slot groups reused only against an explicit credit, on a host with
# InfiniBand and no GPU. The GPU schedule of `mojoccl._do_allreduce` is
# reproduced exactly -- submit chunk k, consume chunk k-(depth-1) -- with the
# calling thread standing in for the stream, so `credit_upto` takes the same
# values it does in production.
#
# What this catches that ib_bringup cannot: a slot group rewritten before its
# consumer read it (wrong bytes, since the pattern carries the sequence
# number), and a credit that is never sent or never counted (a hang, which is
# why the run is bounded well past `nslots` exchanges).
from std.ffi import OwnedDLHandle
from std.memory.alloc import unsafe_alloc
from std.sys import argv
from std.time import sleep

from tmb.ccl.nccl import UID_BYTES
from tmb.ccl.bootstrap import (
    bootstrap_allgather,
    bootstrap_barrier,
    bootstrap_connect,
    make_unique_id,
)
from tmb.ccl.init import derive_topology
from tmb.ccl.misc.utils import host_hash
from tmb.ccl.transport.net import (
    CREDIT_AREA_BYTES,
    IB_BLOB_BYTES,
    ib_connect,
    ib_local_info,
    ib_npeers,
    ib_submit_now,
    ib_wait_now,
)
from tmb.ccl.proxy import ib_setup, ib_teardown

comptime P8 = Pointer[UInt8, MutAnyOrigin]


def _p(n: Int) -> P8:
    var p = P8(unsafe_from_address=Int(unsafe_alloc[UInt8](n)))
    for i in range(n):
        p[unsafe_offset=i] = 0
    return p


def main() raises:
    var a = argv()
    var rank = Int(String(a[1]))
    var nranks = Int(String(a[2]))
    var uid_path = String(a[3])
    var nslots = Int(String(a[4])) if len(a) > 4 else 5
    var depth = Int(String(a[5])) if len(a) > 5 else 4
    var nexch = Int(String(a[6])) if len(a) > 6 else 200
    if depth > nslots:
        raise Error("depth must be <= nslots or the credit window deadlocks")

    var uid = _p(UID_BYTES)
    if rank == 0:
        make_unique_id(uid)
        var t = String("")
        for i in range(UID_BYTES):
            t += String(Int(uid[unsafe_offset=i])) + " "
        with open(uid_path, "w") as f:
            f.write(t)
    else:
        while True:
            try:
                var text: String
                with open(uid_path, "r") as f:
                    text = f.read()
                var k = 0
                for part in text.split(" "):
                    if String(part).byte_length() == 0:
                        continue
                    uid[unsafe_offset=k] = UInt8(Int(String(part)))
                    k += 1
                if k == UID_BYTES:
                    break
            except:
                pass
            sleep(0.02)

    var conn = bootstrap_connect(uid, rank, nranks, 30.0)
    # One "node" per rank, as in ib_bringup: that is what gives every rank a
    # queue pair to every other.
    var b1 = _p(16)
    b1.unsafe_bitcast[UInt64]()[unsafe_offset=0] = host_hash() + UInt64(rank)
    var t1 = _p(16 * nranks)
    bootstrap_allgather(conn, b1, 16, t1, 30.0)
    var hashes = List[UInt64]()
    for r in range(nranks):
        hashes.append(t1.unsafe_bitcast[UInt64]()[unsafe_offset=2 * r])
    var topo = derive_topology(hashes, rank)

    var nbytes = 16 * 1024
    var slot_bytes = nbytes
    var npeers_max = nranks - 1
    var group = npeers_max * slot_bytes
    var credit_off = 0
    var inbox0 = CREDIT_AREA_BYTES
    var src = inbox0 + nslots * group
    var region_bytes = src + depth * nbytes + 4096
    var region = _p(region_bytes)
    var driver = OwnedDLHandle("libc.so.6")
    var ib = ib_setup(
        driver,
        0,
        topo.my_local_rank,
        topo.local_world,
        topo.my_node,
        topo.nnodes,
        Int(region),
        region_bytes,
        nslots,
        credit_off,
        _synchronous_test=True,
    )
    var b2 = _p(IB_BLOB_BYTES)
    ib_local_info(ib, b2)
    var t2 = _p(IB_BLOB_BYTES * nranks)
    bootstrap_allgather(conn, b2, IB_BLOB_BYTES, t2, 30.0)
    var peer_rank_of_node = List[Int]()
    for j in range(topo.nnodes):
        peer_rank_of_node.append(topo.rank_at[j * topo.local_world])
    ib_connect(ib, t2, IB_BLOB_BYTES, peer_rank_of_node)
    bootstrap_barrier(conn, 30.0)
    var npeers = ib_npeers(ib)
    print(
        "rank",
        rank,
        "peers",
        npeers,
        "slots",
        nslots,
        "depth",
        depth,
        "exchanges",
        nexch,
        "RTS",
    )

    # Each in-flight exchange gets its own send buffer, the way each pipeline
    # chunk gets its own arena: the NIC may still be reading chunk k's bytes
    # while chunk k+1 is being filled.
    var bad = 0
    var consumed = 0
    for k in range(nexch + depth - 1):
        if k < nexch:
            var seq = k + 1
            var sbuf = src + (k % depth) * nbytes
            for i in range(nbytes):
                region[unsafe_offset=sbuf + i] = UInt8(
                    (rank * 31 + seq * 7 + i) & 0xFF
                )
            var inbox_base = inbox0 + (seq % nslots) * group
            ib_submit_now(
                ib,
                Int(region) + sbuf,
                nbytes,
                inbox_base,
                slot_bytes,
                True,
                npeers,
                Int(region) + inbox_base,
                seq,
                consumed,
            )
        var j = k - (depth - 1)
        if j >= 0:
            var seq = j + 1
            var inbox_base = inbox0 + (seq % nslots) * group
            ib_wait_now(ib, seq)
            for q in range(npeers):
                var sender = q if q < topo.my_node else q + 1
                for i in range(nbytes):
                    var want = UInt8((sender * 31 + seq * 7 + i) & 0xFF)
                    if (
                        region[unsafe_offset=inbox_base + q * slot_bytes + i]
                        != want
                    ):
                        bad += 1
            consumed = seq
            if seq % 50 == 0 or seq < 3:
                print("rank", rank, "consumed", seq, "bad", bad)
    ib_teardown(ib)
    conn.close()
    print("rank", rank, "PASS" if bad == 0 else "FAIL")
