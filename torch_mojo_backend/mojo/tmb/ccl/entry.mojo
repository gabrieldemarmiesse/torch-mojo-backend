# Rewrite of: none -- the export table of libnccl (NCCL exports these via NCCL_API in the files named in each shim's module). Closest: https://github.com/NVIDIA/nccl/blob/master/src/nccl.h.in
#
# mojoccl: a Mojo shared library exporting NCCL's C ABI (nccl.h), so
# torch_mojo_backend/distributed/nccl.py can dlopen it exactly like
# libnccl.so.2/librccl.so.1 (TORCH_MOJO_BACKEND_CCL=mojo), and so external
# tools (nccl-tests) can link it as a libnccl.so drop-in.
#
# Signatures, enum values and ncclResult_t codes are pinned to
# /home/gabriel/projects/nccl/src/nccl.h.in (2.31.2) -- the source of truth
# is nccl.py's `_declare()`, which this library's exports were written
# against. AllReduce/Broadcast/AllGather/ReduceScatter are real; Reduce/
# Send/Recv return ncclInvalidUsage (GPT-2 DDP needs only the first three;
# tests/ddp_worker.py skips the checks that need them when
# TORCH_MOJO_BACKEND_CCL=mojo is set).
#
# One process per GPU. Within a node, one region per rank of raw
# driver-owned memory -- MAX's own allocator memory cannot be shared across
# processes at all (see agents_docs/mojo_collectives_feasibility.md, section 5.6) --
# and the collectives of device/symmetric/ run over the peer mappings.
# Across nodes, GPUDirect RDMA written here over libibverbs (transport/net_ib/,
# transport/net.mojo): no vendor collective library takes part at any level.
#
# This module is only the export table (NCCL's src/libnccl.map): one
# `@export` shim per symbol, forwarding to the implementation in the module
# that mirrors the NCCL file defining it. An `@export` is emitted only from
# the module being built, so the shims have to live here.

from tmb.ccl.init import (
    ncclGetVersion as _ncclGetVersion,
    ncclGetErrorString as _ncclGetErrorString,
    ncclGetUniqueId as _ncclGetUniqueId,
    ncclCommInitRank as _ncclCommInitRank,
    ncclCommDestroy as _ncclCommDestroy,
    ncclCommAbort as _ncclCommAbort,
    ncclCommGetAsyncError as _ncclCommGetAsyncError,
    ncclCommCount as _ncclCommCount,
    ncclCommUserRank as _ncclCommUserRank,
)
from tmb.ccl.group import (
    ncclGroupStart as _ncclGroupStart,
    ncclGroupEnd as _ncclGroupEnd,
)
from tmb.ccl.collectives import (
    ncclAllReduce as _ncclAllReduce,
    ncclBroadcast as _ncclBroadcast,
    ncclAllGather as _ncclAllGather,
    ncclReduce as _ncclReduce,
    ncclReduceScatter as _ncclReduceScatter,
    ncclSend as _ncclSend,
    ncclRecv as _ncclRecv,
)


# ---- src/init.cc (init.mojo)


@export
def ncclGetVersion(version: Pointer[Int32, MutAnyOrigin]) abi("C") -> Int32:
    return _ncclGetVersion(version)


@export
def ncclGetErrorString(
    result: Int32,
) abi("C") -> Pointer[UInt8, ImmStaticOrigin]:
    return _ncclGetErrorString(result)


@export
def ncclGetUniqueId(uid_out: Pointer[UInt8, MutAnyOrigin]) abi("C") -> Int32:
    return _ncclGetUniqueId(uid_out)


# `ncclUniqueId` is a 128-byte struct passed by value: the comment above
# init.mojo's ncclCommInitRank explains how this signature spells it.
@export
def ncclCommInitRank(
    comm_out: Pointer[Int64, MutAnyOrigin],
    nranks: Int32,
    rank: Int32,
    _r1: Int64,
    _r2: Int64,
    _r3: Int64,
    id0: UInt64,
    id1: UInt64,
    id2: UInt64,
    id3: UInt64,
    id4: UInt64,
    id5: UInt64,
    id6: UInt64,
    id7: UInt64,
    id8: UInt64,
    id9: UInt64,
    id10: UInt64,
    id11: UInt64,
    id12: UInt64,
    id13: UInt64,
    id14: UInt64,
    id15: UInt64,
) abi("C") -> Int32:
    return _ncclCommInitRank(
        comm_out,
        nranks,
        rank,
        _r1,
        _r2,
        _r3,
        id0,
        id1,
        id2,
        id3,
        id4,
        id5,
        id6,
        id7,
        id8,
        id9,
        id10,
        id11,
        id12,
        id13,
        id14,
        id15,
    )


@export
def ncclCommDestroy(comm: Int64) abi("C") -> Int32:
    return _ncclCommDestroy(comm)


@export
def ncclCommAbort(comm: Int64) abi("C") -> Int32:
    return _ncclCommAbort(comm)


@export
def ncclCommGetAsyncError(
    comm: Int64, err_out: Pointer[Int32, MutAnyOrigin]
) abi("C") -> Int32:
    return _ncclCommGetAsyncError(comm, err_out)


@export
def ncclCommCount(
    comm: Int64, count_out: Pointer[Int32, MutAnyOrigin]
) abi("C") -> Int32:
    return _ncclCommCount(comm, count_out)


@export
def ncclCommUserRank(
    comm: Int64, rank_out: Pointer[Int32, MutAnyOrigin]
) abi("C") -> Int32:
    return _ncclCommUserRank(comm, rank_out)


# ---- src/group.cc (group.mojo)


@export
def ncclGroupStart() abi("C") -> Int32:
    return _ncclGroupStart()


@export
def ncclGroupEnd() abi("C") -> Int32:
    return _ncclGroupEnd()


# ---- src/collectives.cc (collectives.mojo)


@export
def ncclAllReduce(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return _ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream)


@export
def ncclBroadcast(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    root: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return _ncclBroadcast(
        sendbuff, recvbuff, count, datatype, root, comm, stream
    )


@export
def ncclAllGather(
    sendbuff: Int64,
    recvbuff: Int64,
    sendcount: Int64,
    datatype: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return _ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream)


@export
def ncclReduce(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    root: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return _ncclReduce(
        sendbuff, recvbuff, count, datatype, op, root, comm, stream
    )


@export
def ncclReduceScatter(
    sendbuff: Int64,
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    op: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return _ncclReduceScatter(
        sendbuff, recvbuff, count, datatype, op, comm, stream
    )


@export
def ncclSend(
    sendbuff: Int64,
    count: Int64,
    datatype: Int32,
    peer: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return _ncclSend(sendbuff, count, datatype, peer, comm, stream)


@export
def ncclRecv(
    recvbuff: Int64,
    count: Int64,
    datatype: Int32,
    peer: Int32,
    comm: Int64,
    stream: Int64,
) abi("C") -> Int32:
    return _ncclRecv(recvbuff, count, datatype, peer, comm, stream)
