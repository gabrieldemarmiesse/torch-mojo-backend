# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/transport/net_ib/connect.cc

from std.os import getenv

from tmb.ccl.env_vars import MOJOCCL_IB_HCA, MOJOCCL_IB_RELAXED_ORDERING
from tmb.ccl.graph.topo import pci_pick
from tmb.ccl.include.ibvcore import (
    IBV_ACCESS_LOCAL_WRITE,
    IBV_ACCESS_REMOTE_READ,
    IBV_ACCESS_REMOTE_WRITE,
    IBV_QPS_INIT,
    IBV_QPS_RTR,
    IBV_QPS_RTS,
    IBV_QPT_RC,
    MR_LKEY,
    MR_RKEY,
    PA_ACTIVE_MTU,
    PA_LID,
    QA_ACCESS_FLAGS,
    QA_AH_DGID,
    QA_AH_DLID,
    QA_AH_HOP_LIMIT,
    QA_AH_IS_GLOBAL,
    QA_AH_PORT_NUM,
    QA_AH_SGID_INDEX,
    QA_AH_SL,
    QA_DEST_QP_NUM,
    QA_MAX_DEST_RD_ATOMIC,
    QA_MAX_RD_ATOMIC,
    QA_MIN_RNR_TIMER,
    QA_PATH_MTU,
    QA_PKEY_INDEX,
    QA_PORT_NUM,
    QA_QP_STATE,
    QA_RETRY_CNT,
    QA_RNR_RETRY,
    QA_RQ_PSN,
    QA_SQ_PSN,
    QA_TIMEOUT,
    QIA_MAX_RECV_SGE,
    QIA_MAX_RECV_WR,
    QIA_MAX_SEND_SGE,
    QIA_MAX_SEND_WR,
    QIA_QP_TYPE,
    QIA_RECV_CQ,
    QIA_SEND_CQ,
    QP_MASK_INIT,
    QP_MASK_RTR,
    QP_MASK_RTS,
    SZ_PORT_ATTR,
    SZ_QP_ATTR,
    SZ_QP_INIT_ATTR,
    SZ_RECV_WR,
    SZ_SEND_WR,
    SZ_SGE,
    SZ_WC,
)
from tmb.ccl.include.ibvwrap import post_recv
from tmb.ccl.include.plugin.nccl_net import MAX_NODES
from tmb.ccl.misc.ibvwrap import Ibv, build_recv_wr, qp_number
from tmb.ccl.misc.utils import (
    P8,
    alloc_bytes,
    ld16,
    ld32,
    ldu32,
    st16,
    st32,
    st64,
    st8,
    stu32,
    stu64,
)
from tmb.ccl.transport.net_ib.init import list_ib_ports


# ---- QP attribute values, all from NCCL ----------------------------------
# nccl:src/transport/net_ib/connect.cc:440,:502 -- both PSNs are hardcoded
# 0 and no PSN is exchanged, so neither is this library's wire format.
comptime IB_PSN: Int32 = 0
comptime IB_MIN_RNR_TIMER: UInt8 = 12  # connect.cc:443
comptime IB_TIMEOUT: UInt8 = 20  # connect.cc:496, NCCL_IB_TIMEOUT default
comptime IB_RETRY_CNT: UInt8 = 7  # connect.cc:497, NCCL_IB_RETRY_CNT default
comptime IB_RNR_RETRY: UInt8 = 7  # connect.cc:498, hardcoded (= infinite)
comptime IB_HOP_LIMIT: UInt8 = 255  # connect.cc:452,:476


# ===-------------------------------------------------------------------=== #
# QP bring-up
# ===-------------------------------------------------------------------=== #


def create_rc_qp(
    ibv: Ibv, pd: Int, cq: Int, max_send_wr: Int, max_recv_wr: Int
) raises -> Int:
    var ia = alloc_bytes(SZ_QP_INIT_ATTR)
    st64(ia, QIA_SEND_CQ, cq)
    st64(ia, QIA_RECV_CQ, cq)
    st32(ia, QIA_MAX_SEND_WR, Int32(max_send_wr))
    st32(ia, QIA_MAX_RECV_WR, Int32(max_recv_wr))
    st32(ia, QIA_MAX_SEND_SGE, 1)
    st32(ia, QIA_MAX_RECV_SGE, 1)
    st32(ia, QIA_QP_TYPE, IBV_QPT_RC)
    var qp = ibv.create_qp(pd, ia)
    if qp == 0:
        raise Error("mojoccl: ibv_create_qp failed")
    return qp


def qp_to_init(ibv: Ibv, qp: Int, port: Int) raises:
    var a = alloc_bytes(SZ_QP_ATTR)
    st32(a, QA_QP_STATE, IBV_QPS_INIT)
    st16(a, QA_PKEY_INDEX, 0)
    st8(a, QA_PORT_NUM, UInt8(port))
    # Both directions on one QP pair: this rank writes into the peer's
    # region and the peer writes into this one, so REMOTE_WRITE is needed
    # on both ends (NCCL splits it because its QPs are one-directional).
    st32(
        a,
        QA_ACCESS_FLAGS,
        IBV_ACCESS_LOCAL_WRITE
        | IBV_ACCESS_REMOTE_WRITE
        | IBV_ACCESS_REMOTE_READ,
    )
    var rc = ibv.modify_qp(qp, a, QP_MASK_INIT)
    if rc != 0:
        raise Error("mojoccl: ibv_modify_qp(INIT) failed, rc=" + String(rc))


def qp_to_rtr(
    ibv: Ibv,
    qp: Int,
    dest_qpn: UInt32,
    dlid: Int,
    mtu: Int,
    port: Int,
    remote_gid: P8,
    local_gid_index: Int,
    global_route: Bool,
) raises:
    var a = alloc_bytes(SZ_QP_ATTR)
    st32(a, QA_QP_STATE, IBV_QPS_RTR)
    st32(a, QA_PATH_MTU, Int32(mtu))
    stu32(a, QA_DEST_QP_NUM, dest_qpn)
    st32(a, QA_RQ_PSN, IB_PSN)
    st8(a, QA_MAX_DEST_RD_ATOMIC, 1)
    st8(a, QA_MIN_RNR_TIMER, IB_MIN_RNR_TIMER)
    st16(a, QA_AH_DLID, UInt16(dlid))
    st8(a, QA_AH_SL, 0)
    st8(a, QA_AH_PORT_NUM, UInt8(port))
    if global_route:
        # Only when the two ports are on different IB subnets -- NCCL's rule
        # (connect.cc:456-458); a single-subnet fabric never takes this.
        st8(a, QA_AH_IS_GLOBAL, 1)
        for i in range(16):
            a[unsafe_offset=QA_AH_DGID + i] = remote_gid[unsafe_offset=i]
        st8(a, QA_AH_SGID_INDEX, UInt8(local_gid_index))
        st8(a, QA_AH_HOP_LIMIT, IB_HOP_LIMIT)
    var rc = ibv.modify_qp(qp, a, QP_MASK_RTR)
    if rc != 0:
        raise Error("mojoccl: ibv_modify_qp(RTR) failed, rc=" + String(rc))


def qp_to_rts(ibv: Ibv, qp: Int) raises:
    var a = alloc_bytes(SZ_QP_ATTR)
    st32(a, QA_QP_STATE, IBV_QPS_RTS)
    st32(a, QA_SQ_PSN, IB_PSN)
    st8(a, QA_TIMEOUT, IB_TIMEOUT)
    st8(a, QA_RETRY_CNT, IB_RETRY_CNT)
    st8(a, QA_RNR_RETRY, IB_RNR_RETRY)
    st8(a, QA_MAX_RD_ATOMIC, 1)
    var rc = ibv.modify_qp(qp, a, QP_MASK_RTS)
    if rc != 0:
        raise Error("mojoccl: ibv_modify_qp(RTS) failed, rc=" + String(rc))


# ===-------------------------------------------------------------------=== #
# The transport, in the shape `transport/net.mojo`'s engine drives
# ===-------------------------------------------------------------------=== #
#
# `transport/net_ofi.mojo` offers the same six operations over a completely
# different API (post payload, post immediate, post flush, poll, local info,
# add peer); the engine calls one or the other and never learns which
# library is underneath. Everything below is the verbs half, moved here
# unchanged from the engine when the second transport arrived -- it is the
# code the two-node H100 + InfiniBand measurements were taken with.

# Recv WRs kept posted per peer QP. Each RDMA_WRITE_WITH_IMM consumes one --
# data and credits alike; the engine reposts every one it consumes, so the
# depth only has to cover the burst a peer can produce while this rank is
# elsewhere: `nslots` data messages plus `nslots` credits, times a wide
# margin.
comptime RECV_DEPTH = 64
comptime CQ_SIZE = 1024
comptime SEND_WR_DEPTH = 64


struct VerbsNet(Movable):
    """Everything the libibverbs transport owns, per communicator."""

    var ibv: Ibv
    var hca: String
    var ctx: Int
    var port: Int
    var pd: Int
    var mr: Int
    var lkey: UInt32
    var rkey: UInt32
    var cq: Int
    var qps: List[Int]  # one RC queue pair per peer, indexed as IbState.peers
    var qpns: List[UInt32]
    var flush_qp: Int
    var flush_mr: Int
    var flush_host: Int
    var flush_lkey: UInt32
    # Scratch the work-request builders write into, reused across posts.
    var wr: Int
    var sge: Int
    var rwr: Int
    var bad: Int
    var wc: Int

    def __init__(out self, var ibv: Ibv):
        self.ibv = ibv^
        self.hca = String("")
        self.ctx = 0
        self.port = 0
        self.pd = 0
        self.mr = 0
        self.lkey = 0
        self.rkey = 0
        self.cq = 0
        self.qps = List[Int]()
        self.qpns = List[UInt32]()
        self.flush_qp = 0
        self.flush_mr = 0
        self.flush_host = 0
        self.flush_lkey = 0
        self.wr = Int(alloc_bytes(SZ_SEND_WR))
        self.sge = Int(alloc_bytes(SZ_SGE))
        self.rwr = Int(alloc_bytes(SZ_RECV_WR))
        self.bad = Int(alloc_bytes(16))
        self.wc = Int(alloc_bytes(SZ_WC * 16))


@always_inline
def _b(addr: Int) -> P8:
    return P8(unsafe_from_address=addr)


def vrb_setup(
    gpu_bdf: String,
    local_rank: Int,
    nnodes: Int,
    region: Int,
    region_bytes: Int,
) raises -> VerbsNet:
    """Open an HCA, register the region, create every queue pair (in INIT).

    The QPs cannot reach RTR until the peers' `(qpn, lid, gid)` have been
    gathered, so `vrb_connect_peer` finishes the job.
    """
    var ibv = Ibv()
    var want = getenv(MOJOCCL_IB_HCA, "")
    var ports = list_ib_ports(ibv, want)
    if len(ports) == 0:
        raise Error(
            "mojoccl: no ACTIVE InfiniBand port found"
            + (
                " matching MOJOCCL_IB_HCA=" + want if want.byte_length()
                > 0 else ""
            )
            + "; a multi-node communicator needs one"
        )
    var paths = List[String]()
    for i in range(len(ports)):
        paths.append("/sys/class/infiniband/" + ports[i].name + "/device")
    var pick = pci_pick(paths, gpu_bdf, local_rank)
    ref port = ports[pick]
    # These nodes carry ~10 IB HCAs and every rank opened all of them to
    # read their ports; hold only the one this rank will use.
    for i in range(len(ports)):
        if ports[i].ctx != port.ctx:
            ibv.close_device(ports[i].ctx)

    var v = VerbsNet(ibv^)
    v.hca = String(port.name)
    v.ctx = port.ctx
    v.port = port.port
    try:
        v.pd = v.ibv.alloc_pd(v.ctx)
        if v.pd == 0:
            raise Error("mojoccl: ibv_alloc_pd failed on " + v.hca)
        var ro = getenv(MOJOCCL_IB_RELAXED_ORDERING, "1") != "0"
        var acc = (
            IBV_ACCESS_LOCAL_WRITE
            | IBV_ACCESS_REMOTE_WRITE
            | IBV_ACCESS_REMOTE_READ
        )
        v.mr = v.ibv.reg_mr_relaxed(
            v.pd, region, region_bytes, acc
        ) if ro else v.ibv.reg_mr(v.pd, region, region_bytes, acc)
        if v.mr == 0:
            raise Error(
                "mojoccl: ibv_reg_mr of the "
                + String(region_bytes // (1024 * 1024))
                + " MiB device region failed on "
                + v.hca
                + "; is nvidia_peermem (or the ROCm equivalent) loaded?"
            )
        var mrp = _b(v.mr)
        v.lkey = ldu32(mrp, MR_LKEY)
        v.rkey = ldu32(mrp, MR_RKEY)

        # Host landing pad for the flush read, and the source of the 4-byte
        # credit writes.
        v.flush_host = Int(alloc_bytes(4096))
        v.flush_mr = v.ibv.reg_mr(
            v.pd, v.flush_host, 4096, IBV_ACCESS_LOCAL_WRITE
        )
        if v.flush_mr == 0:
            raise Error("mojoccl: ibv_reg_mr of the flush buffer failed")
        v.flush_lkey = ldu32(_b(v.flush_mr), MR_LKEY)

        v.cq = v.ibv.create_cq(v.ctx, CQ_SIZE)
        if v.cq == 0:
            raise Error("mojoccl: ibv_create_cq failed")

        for _ in range(nnodes - 1):
            var qp = create_rc_qp(
                v.ibv, v.pd, v.cq, SEND_WR_DEPTH, RECV_DEPTH + 8
            )
            # Recorded before `qp_to_init` can raise, so the unwind below
            # destroys it.
            v.qps.append(qp)
            v.qpns.append(qp_number(qp))
            qp_to_init(v.ibv, qp, v.port)
        v.flush_qp = create_rc_qp(v.ibv, v.pd, v.cq, SEND_WR_DEPTH, 8)
        qp_to_init(v.ibv, v.flush_qp, v.port)
    except e:
        vrb_teardown(v)
        raise e
    return v^


def vrb_port_lid(v: VerbsNet) raises -> Int:
    """The port's LID, straight from `ibv_query_port`.

    A nonzero return code (not the same thing as a `try/except` --
    `query_port`'s own C call never raises, it returns an errno) used to be
    silently discarded, reading LID 0 out of `pa`'s zeroed scratch. For the
    self-connected flush QP that 0 is not a sentinel anyone downstream
    checks; it just quietly modifies the flush QP with the wrong address.
    Raise instead.
    """
    var pa = alloc_bytes(SZ_PORT_ATTR)
    var rc = v.ibv.query_port(v.ctx, v.port, pa)
    if rc != 0:
        raise Error("mojoccl: ibv_query_port failed, rc=" + String(rc))
    return ld16(pa, PA_LID)


def vrb_port_mtu(v: VerbsNet) raises -> Int:
    """The port's active MTU (see `vrb_port_lid` for why a failed query
    raises rather than reading 0 out of zeroed scratch)."""
    var pa = alloc_bytes(SZ_PORT_ATTR)
    var rc = v.ibv.query_port(v.ctx, v.port, pa)
    if rc != 0:
        raise Error("mojoccl: ibv_query_port failed, rc=" + String(rc))
    return ld32(pa, PA_ACTIVE_MTU)


# ---- the bootstrap blob --------------------------------------------------
#
#   +0   u64 region base VA
#   +8   u32 rkey
#   +12  u32 lid
#   +16  u32 active_mtu (ibv_mtu enum)
#   +20  u32 number of QPs that follow
#   +24  u32 qpn[MAX_NODES]     -- indexed by the PEER's node
#   +24+4*MAX_NODES  u8 gid[16]

comptime VRB_BLOB_QPN = 24
comptime VRB_BLOB_GID = 24 + 4 * MAX_NODES


def vrb_local_info(v: VerbsNet, blob: P8, nodes: List[Int], region: Int) raises:
    """Fill this rank's half of the bootstrap blob. `nodes[i]` is the node
    queue pair `i` was created for; `region` is the virtual address peers
    write into (InfiniBand RMA always addresses by virtual address)."""
    stu64(blob, 0, UInt64(region))
    stu32(blob, 8, v.rkey)
    stu32(blob, 12, UInt32(vrb_port_lid(v)))
    stu32(blob, 16, UInt32(vrb_port_mtu(v)))
    stu32(blob, 20, UInt32(len(v.qps)))
    for i in range(len(v.qps)):
        stu32(blob, VRB_BLOB_QPN + 4 * nodes[i], v.qpns[i])


def vrb_blob_base(blob: P8) -> Int:
    return Int(blob.unsafe_bitcast[UInt64]()[unsafe_offset=0])


def vrb_blob_key(blob: P8) -> UInt64:
    return UInt64(ldu32(blob, 8))


def vrb_connect_peer(
    mut v: VerbsNet, peer_index: Int, node: Int, my_node: Int, blob: P8
) raises:
    """Move peer `peer_index`'s queue pair to RTS from its blob, then
    pre-post its receive work requests."""
    var lid = ld32(blob, 12)
    var mtu = ld32(blob, 16)
    # The peer's QP for MY node, not for its own.
    var dest_qpn = ldu32(blob, VRB_BLOB_QPN + 4 * my_node)
    var gid = _b(Int(blob) + VRB_BLOB_GID)
    if lid == 0:
        raise Error(
            "mojoccl: peer on node "
            + String(node)
            + " reported LID 0 -- its HCA port is not on an InfiniBand"
            " fabric this library can address"
        )
    qp_to_rtr(
        v.ibv,
        v.qps[peer_index],
        dest_qpn,
        lid,
        min(vrb_port_mtu(v), mtu),
        v.port,
        gid,
        0,
        False,
    )
    qp_to_rts(v.ibv, v.qps[peer_index])
    for _ in range(RECV_DEPTH):
        build_recv_wr(_b(v.rwr), 0)
        if post_recv(v.qps[peer_index], _b(v.rwr), _b(v.bad)) != 0:
            raise Error("mojoccl: ibv_post_recv failed while pre-posting")


def vrb_connect_flush(mut v: VerbsNet) raises:
    """The flush queue pair talks to itself."""
    var gid0 = alloc_bytes(16)
    qp_to_rtr(
        v.ibv,
        v.flush_qp,
        qp_number(v.flush_qp),
        vrb_port_lid(v),
        vrb_port_mtu(v),
        v.port,
        gid0,
        0,
        False,
    )
    qp_to_rts(v.ibv, v.flush_qp)


def vrb_teardown(mut v: VerbsNet):
    """Release every ibverbs resource `vrb_setup` may have created -- shared
    by teardown of a live communicator and by `vrb_setup`'s own failure path
    (a later step raised after an earlier one succeeded), which is why every
    field is zero-guarded."""
    try:
        for i in range(len(v.qps)):
            v.ibv.destroy_qp(v.qps[i])
        v.qps.clear()
        if v.flush_qp != 0:
            v.ibv.destroy_qp(v.flush_qp)
            v.flush_qp = 0
        if v.cq != 0:
            v.ibv.destroy_cq(v.cq)
            v.cq = 0
        if v.flush_mr != 0:
            v.ibv.dereg_mr(v.flush_mr)
            v.flush_mr = 0
        if v.mr != 0:
            v.ibv.dereg_mr(v.mr)
            v.mr = 0
        if v.pd != 0:
            v.ibv.dealloc_pd(v.pd)
            v.pd = 0
        if v.ctx != 0:
            v.ibv.close_device(v.ctx)
            v.ctx = 0
    except e:
        # Best-effort teardown: nothing is left to undo, but say what failed.
        print("mojoccl: verbs teardown step failed (ignored):", e)
