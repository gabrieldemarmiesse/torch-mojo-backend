from std.ffi import external_call
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
from tmb.ccl.misc.socket import format_ipv4, local_ipv4
from tmb.ccl.misc.utils import host_hash


def _p8(n: Int) -> Pointer[UInt8, MutAnyOrigin]:
    return Pointer[UInt8, MutAnyOrigin](
        unsafe_from_address=Int(unsafe_alloc[UInt8](n))
    )


def main() raises:
    var a = argv()
    var rank = Int(String(a[1]))
    var nranks = Int(String(a[2]))
    var uid_path = String(a[3])
    var fake_host = Int(String(a[4]))  # pretend-node index for topology tests

    var uid = _p8(UID_BYTES)
    if rank == 0:
        make_unique_id(uid)
        var s = String("")
        for i in range(UID_BYTES):
            s += String(Int(uid[unsafe_offset=i])) + " "
        with open(uid_path, "w") as f:
            f.write(s)
        print("rank 0 ip", format_ipv4(local_ipv4()))
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
    # round 1: host hash (faked so one box can pretend to be several nodes)
    var blob = _p8(16)
    var bw = blob.unsafe_bitcast[UInt64]()
    bw[unsafe_offset=0] = host_hash() + UInt64(fake_host)
    bw[unsafe_offset=1] = UInt64(rank)
    var all = _p8(16 * nranks)
    bootstrap_allgather(conn, blob, 16, all, 30.0)
    var aw = all.unsafe_bitcast[UInt64]()
    var hashes = List[UInt64]()
    for r in range(nranks):
        hashes.append(aw[unsafe_offset=2 * r])
        if Int(aw[unsafe_offset=2 * r + 1]) != r:
            raise Error("rank mismatch in gathered table at " + String(r))
    var topo = derive_topology(hashes, rank)
    # round 2: a 256-byte blob, checked
    var b2 = _p8(256)
    for i in range(256):
        b2[unsafe_offset=i] = UInt8((rank * 7 + i) & 0xFF)
    var a2 = _p8(256 * nranks)
    bootstrap_allgather(conn, b2, 256, a2, 30.0)
    for r in range(nranks):
        for i in range(256):
            if Int(a2[unsafe_offset=r * 256 + i]) != ((r * 7 + i) & 0xFF):
                raise Error("round-2 payload mismatch")
    bootstrap_barrier(conn, 30.0)
    conn.close()
    print(
        "rank",
        rank,
        "nnodes",
        topo.nnodes,
        "local_world",
        topo.local_world,
        "my_node",
        topo.my_node,
        "my_local_rank",
        topo.my_local_rank,
        "rank_at[0]",
        topo.rank_at[0],
        "OK",
    )
