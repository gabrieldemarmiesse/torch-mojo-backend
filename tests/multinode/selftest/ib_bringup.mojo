# Two-process bring-up self-test for bootstrap + ibverbs + internode, on a
# host that has InfiniBand but no GPU: the "region" is registered host
# memory and the exchange runs inline instead of from a stream callback.
from std.ffi import OwnedDLHandle, external_call
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
    ib_exchange_now,
    ib_local_info,
    ib_npeers,
)
from tmb.ccl.proxy import ib_setup, ib_teardown

comptime P8 = Pointer[UInt8, MutAnyOrigin]
comptime CAP = 1 << 20
comptime SIGNAL = 128 * 1024


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
    # Pretend each rank is its own node (that is what an inter-node QP set
    # needs); local_world == 1.
    var b1 = _p(16)
    b1.unsafe_bitcast[UInt64]()[unsafe_offset=0] = host_hash() + UInt64(rank)
    var t1 = _p(16 * nranks)
    bootstrap_allgather(conn, b1, 16, t1, 30.0)
    var hashes = List[UInt64]()
    for r in range(nranks):
        hashes.append(t1.unsafe_bitcast[UInt64]()[unsafe_offset=2 * r])
    var topo = derive_topology(hashes, rank)
    print("rank", rank, "nnodes", topo.nnodes, "local_world", topo.local_world)

    var region_bytes = SIGNAL + 3 * CAP
    var region = _p(region_bytes)
    var net_off = SIGNAL + 2 * CAP
    var driver = OwnedDLHandle("libc.so.6")  # no GPU here; PCI probe just fails
    # Two inbox slot groups: enough that consecutive exchanges alternate, and
    # the credit for e-2 is always in hand by the time e is submitted, so
    # this test stays a transport test. The credit protocol proper (several
    # exchanges in flight, reuse gated on a peer's credit) is ib_pipeline.
    var ib = ib_setup(
        driver,
        0,
        topo.my_local_rank,
        topo.local_world,
        topo.my_node,
        topo.nnodes,
        Int(region),
        region_bytes,
        2,
        net_off,
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
    print("rank", rank, "peers", ib_npeers(ib), "connected")

    # Many exchanges, alternating inbox halves and payload sizes, well past
    # RECV_DEPTH so a leaked recv WR would show up as a hang.
    var npeers = ib_npeers(ib)
    var nbytes = 64 * 1024
    var inbox0 = net_off + CREDIT_AREA_BYTES
    var slot_bytes = nbytes
    var half = npeers * slot_bytes
    var src = inbox0 + 2 * half  # scratch above both slot groups
    var bad = 0
    var nseq = 401
    for seq in range(1, nseq):
        for i in range(nbytes):
            region[unsafe_offset=src + i] = UInt8(
                (rank * 31 + seq * 7 + i) & 0xFF
            )
        var inbox_base = inbox0 + (seq % 2) * half
        ib_exchange_now(
            ib,
            Int(region) + src,
            nbytes,
            inbox_base,
            slot_bytes,
            True,
            npeers,
            Int(region) + inbox_base,
            seq,
            seq - 1,  # the previous exchange was checked before this one
        )
        for j in range(npeers):
            var sender = j if j < topo.my_node else j + 1
            for i in range(nbytes):
                var want = UInt8((sender * 31 + seq * 7 + i) & 0xFF)
                if (
                    region[unsafe_offset=inbox_base + j * slot_bytes + i]
                    != want
                ):
                    bad += 1
        if seq % 100 == 0 or seq < 3:
            print(
                "rank", rank, "seq", seq, "checked", npeers, "slots, bad", bad
            )
    ib_teardown(ib)
    conn.close()
    print("rank", rank, "PASS" if bad == 0 else "FAIL")
