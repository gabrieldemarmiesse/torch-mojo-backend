# The one piece of the libfabric transport that host memory cannot exercise:
# registering a region the GPU driver allocated, and writing into it from
# another process's NIC.
#
# `ib_bringup` and `ib_pipeline` run the whole engine over a plain malloc'd
# region, which on the cxi provider registers as FI_HMEM_SYSTEM. Production
# hands `ib_setup` a `transport/p2p.mojo` `alloc_region` allocation
# (hipExtMallocWithFlags(hipDeviceMallocUncached) / cuMemAlloc_v2), which has
# to register as FI_HMEM_ROCR (or FI_HMEM_CUDA) instead -- a different code
# path in the provider, a different kernel driver, and the thing most likely
# to be missing from a libfabric build. This is the smallest program that
# touches it.
#
#     fabric_hmem <rank> <nranks> <uid-file> [nexchanges]
#
# Each rank is its own "node", so every rank exchanges with every other. It
# needs a GPU (one HIP/CUDA context per rank) and must be BUILT ON A NODE
# THAT HAS ONE: `misc/cudawrap.mojo` picks libamdhip64 vs libcuda at compile time
# from the build host's accelerator.
#
# What it checks is that every exchange RETIRES: the payload write landed,
# every peer's notification arrived, this rank's own writes completed and the
# flush read came back, with `ib_error` still zero. It does NOT check the
# bytes -- the region is device memory and this test has no stream to copy it
# back with. Byte-level correctness of an RMA write into
# hipExtMallocWithFlags memory over cxi was checked separately, in C.
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
from tmb.ccl.misc.utils import host_hash, P8, alloc_bytes
from tmb.ccl.transport.p2p import alloc_region, free_region
from tmb.ccl.misc.cudawrap import open_driver
from tmb.ccl.transport.net import (
    CREDIT_AREA_BYTES,
    IB_BLOB_BYTES,
    ib_connect,
    ib_error,
    ib_exchange_now,
    ib_local_info,
    ib_npeers,
)
from tmb.ccl.proxy import ib_setup, ib_teardown


def main() raises:
    var a = argv()
    var rank = Int(String(a[1]))
    var nranks = Int(String(a[2]))
    var uid_path = String(a[3])
    var nexch = Int(String(a[4])) if len(a) > 4 else 50

    var uid = alloc_bytes(UID_BYTES)
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
    var b1 = alloc_bytes(16)
    b1.unsafe_bitcast[UInt64]()[unsafe_offset=0] = host_hash() + UInt64(rank)
    var t1 = alloc_bytes(16 * nranks)
    bootstrap_allgather(conn, b1, 16, t1, 30.0)
    var hashes = List[UInt64]()
    for r in range(nranks):
        hashes.append(t1.unsafe_bitcast[UInt64]()[unsafe_offset=2 * r])
    var topo = derive_topology(hashes, rank)

    var nbytes = 64 * 1024
    var slot_bytes = nbytes
    var npeers_max = nranks - 1
    var net_off = 0
    var inbox0 = CREDIT_AREA_BYTES
    var half = npeers_max * slot_bytes
    var src = inbox0 + 2 * half
    var region_bytes = src + nbytes + 4096

    # The real thing: a driver allocation, not a malloc.
    var lib = open_driver()
    var region = alloc_region(lib, region_bytes)
    print("rank", rank, "region", region_bytes // 1024, "KiB at", region)

    var ib = ib_setup(
        lib,
        0,
        topo.my_local_rank,
        topo.local_world,
        topo.my_node,
        topo.nnodes,
        region,
        region_bytes,
        2,
        net_off,
        _synchronous_test=True,
    )
    var b2 = alloc_bytes(IB_BLOB_BYTES)
    ib_local_info(ib, b2)
    var t2 = alloc_bytes(IB_BLOB_BYTES * nranks)
    bootstrap_allgather(conn, b2, IB_BLOB_BYTES, t2, 30.0)
    var peer_rank_of_node = List[Int]()
    for j in range(topo.nnodes):
        peer_rank_of_node.append(topo.rank_at[j * topo.local_world])
    ib_connect(ib, t2, IB_BLOB_BYTES, peer_rank_of_node)
    bootstrap_barrier(conn, 30.0)
    var npeers = ib_npeers(ib)
    print("rank", rank, "peers", npeers, "connected on device memory")

    var failed = 0
    for seq in range(1, nexch + 1):
        var inbox_base = inbox0 + (seq % 2) * half
        try:
            ib_exchange_now(
                ib,
                region + src,
                nbytes,
                inbox_base,
                slot_bytes,
                True,
                npeers,
                region + inbox_base,
                seq,
                seq - 1,
            )
        except e:
            print("rank", rank, "exchange", seq, "failed:", e)
            failed += 1
            break
        if seq % 25 == 0 or seq < 3:
            print("rank", rank, "exchange", seq, "retired")
    if ib_error(ib) != 0:
        print("rank", rank, "transport error", ib_error(ib))
        failed += 1
    ib_teardown(ib)
    free_region(lib, region)
    conn.close()
    print("rank", rank, "PASS" if failed == 0 else "FAIL")

    if failed:
        raise Error("fabric_hmem transport failure")
