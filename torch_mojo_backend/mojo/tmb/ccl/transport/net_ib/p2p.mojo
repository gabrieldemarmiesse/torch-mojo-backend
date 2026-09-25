# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/transport/net_ib/p2p.cc

from std.sys import size_of

from tmb.ccl.include.ibvcore import (
    IBV_WC_RDMA_READ,
    IBV_WC_RDMA_WRITE,
    IBV_WC_RECV_RDMA_WITH_IMM,
    IBV_WC_SUCCESS,
    SZ_WC,
    WC_IMM_DATA,
    WC_OPCODE,
    WC_QP_NUM,
    WC_STATUS,
    WC_VENDOR_ERR,
    WC_WR_ID,
)
from tmb.ccl.include.ibvwrap import poll_cq, post_recv, post_send
from tmb.ccl.include.plugin.nccl_net import (
    NC_FLUSH,
    NC_OTHER,
    NC_RECV,
    NC_SEND,
    NetCompletion,
)
from tmb.ccl.misc.ibvwrap import (
    be32,
    build_read_wr,
    build_recv_wr,
    build_write_wr,
)
from tmb.ccl.misc.utils import ld32, ld64, ldu32
from tmb.ccl.transport.net_ib.connect import VerbsNet, _b


# ---- the data path -------------------------------------------------------


def vrb_post_payload(
    mut v: VerbsNet,
    peer: Int,
    local_addr: Int,
    nbytes: Int,
    remote_addr: Int,
    remote_key: UInt64,
    immediate: UInt32,
    seq: Int,
) -> Int:
    """One signaled RDMA_WRITE_WITH_IMM: payload and immediate in a single
    operation, which is the thing the libfabric transport has to build out
    of two."""
    build_write_wr(
        _b(v.wr),
        _b(v.sge),
        seq,
        local_addr,
        v.lkey,
        nbytes,
        remote_addr,
        UInt32(remote_key),
        immediate,
        True,
        True,
    )
    return Int(post_send(v.qps[peer], _b(v.wr), _b(v.bad)))


def vrb_post_imm(
    mut v: VerbsNet,
    peer: Int,
    remote_addr: Int,
    remote_key: UInt64,
    nbytes: Int,
    immediate: UInt32,
    seq: Int,
) -> Int:
    """An immediate with nothing to say: an UNSIGNALED short write into the
    peer's credit landing pad. The bytes are never read -- the immediate is
    the message -- but a real address is needed because a zero-length RDMA
    write is not worth relying on across HCAs. Unsignaled because a
    completion here would be indistinguishable from a data send, and the
    data sends are what reclaims the send queue; a failed credit still
    raises a completion with a bad status."""
    build_write_wr(
        _b(v.wr),
        _b(v.sge),
        seq,
        v.flush_host,
        v.flush_lkey,
        nbytes,
        remote_addr,
        UInt32(remote_key),
        immediate,
        True,
        False,
    )
    return Int(post_send(v.qps[peer], _b(v.wr), _b(v.bad)))


def vrb_post_flush(
    mut v: VerbsNet, remote_addr: Int, nbytes: Int, seq: Int
) -> Int:
    build_read_wr(
        _b(v.wr),
        _b(v.sge),
        seq,
        v.flush_host,
        v.flush_lkey,
        nbytes,
        remote_addr,
        v.rkey,
    )
    return Int(post_send(v.flush_qp, _b(v.wr), _b(v.bad)))


def _peer_of_qpn(v: VerbsNet, qpn: UInt32) -> Int:
    for k in range(len(v.qpns)):
        if v.qpns[k] == qpn:
            return k
    return -1


def vrb_poll(mut v: VerbsNet, comps: Int, max_comps: Int) -> Int:
    """Up to `max_comps` completions, translated into `NetCompletion`s.

    Reposting a consumed receive work request happens here rather than in
    the engine: it is the one piece of per-completion bookkeeping that is
    purely about verbs. A receive slot lost is a later RNR the peer retries
    forever (IB_RNR_RETRY = 7), i.e. a silent hang, so a failed repost is
    reported as a failed completion.
    """
    var n = Int(poll_cq(v.cq, min(max_comps, 16), _b(v.wc)))
    if n < 0:
        var c = NetCompletion()
        c.status = 2000 - n
        Pointer[NetCompletion, MutAnyOrigin](unsafe_from_address=comps)[] = c^
        return 1
    for i in range(n):
        var w = _b(v.wc + i * SZ_WC)
        var c = NetCompletion()
        if Int32(ld32(w, WC_STATUS)) != IBV_WC_SUCCESS:
            c.status = 1000 + ld32(w, WC_STATUS) * 1000 + ld32(w, WC_VENDOR_ERR)
        else:
            var op = Int32(ld32(w, WC_OPCODE))
            c.peer = _peer_of_qpn(v, ldu32(w, WC_QP_NUM))
            if op == IBV_WC_RECV_RDMA_WITH_IMM:
                c.kind = NC_RECV
                c.imm = be32(ldu32(w, WC_IMM_DATA))
                if c.peer >= 0:
                    build_recv_wr(_b(v.rwr), 0)
                    if post_recv(v.qps[c.peer], _b(v.rwr), _b(v.bad)) != 0:
                        c.status = 5
            elif op == IBV_WC_RDMA_WRITE:
                # Only data writes are signaled (credits are not), and their
                # wr_id is the exchange number.
                c.kind = NC_SEND
                c.wr_id = ld64(w, WC_WR_ID)
            elif op == IBV_WC_RDMA_READ:
                c.kind = NC_FLUSH
            else:
                c.kind = NC_OTHER
        Pointer[NetCompletion, MutAnyOrigin](
            unsafe_from_address=comps + i * size_of[NetCompletion]()
        )[] = (c^)
    return n
