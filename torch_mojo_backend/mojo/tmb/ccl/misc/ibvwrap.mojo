# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/misc/ibvwrap.cc
#   also:     https://github.com/NVIDIA/nccl/blob/master/src/misc/ibvsymbols.cc
#   also:     https://github.com/NVIDIA/nccl/blob/master/src/include/ibvwrap.h
#   also:     https://github.com/NVIDIA/nccl/blob/master/src/include/ibvcore.h
#
# libibverbs bindings for mojoccl's inter-node hop -- a narrow RC/RDMA-WRITE
# client, no vendor collective library anywhere.
#
# Two calling conventions, because libibverbs has two:
#
#  * control path (open/query/alloc/reg/create/modify/destroy) -- real
#    exported symbols, reached with OwnedDLHandle.get_function. Runs a
#    couple of dozen times per communicator, at init.
#  * data path (post_send / post_recv / poll_cq) -- NOT exported: they are
#    `static inline` in <infiniband/verbs.h> and dispatch through
#    `qp->context->ops.post_send` etc. NCCL does not include the inline
#    either, it hand-writes the same dereference
#    (nccl:src/include/ibvwrap.h:61,78,88), and so does this file: load the
#    8-byte function pointer at a fixed offset from the ibv_context and call
#    it. Zero dlsym on the hot path.
#
# Every struct offset below was dumped with gcc offsetof against
# /usr/include/infiniband/verbs.h on this machine (rdma-core 1.14.54.0 /
# MLNX OFED 24.10) and the ops-table offsets were additionally read back
# from a live mlx5 context; enum values likewise. Nothing here is from
# memory. x86-64 SysV: the structs are built in raw byte buffers because
# std.ffi still has no C-struct ABI (MOCO-3692).

from std.ffi import OwnedDLHandle

from tmb.ccl.misc.utils import P8, as_fn, ld64, ldu32, st32, st64, stu32


# ---- enum values ---------------------------------------------------------
comptime IBV_QPS_INIT: Int32 = 1
comptime IBV_QPS_RTR: Int32 = 2
comptime IBV_QPS_RTS: Int32 = 3
comptime IBV_QPT_RC: Int32 = 2

comptime IBV_ACCESS_LOCAL_WRITE: Int32 = 1
comptime IBV_ACCESS_REMOTE_WRITE: Int32 = 2
comptime IBV_ACCESS_REMOTE_READ: Int32 = 4
# 1<<20, and above the range the versioned `ibv_reg_mr@IBVERBS_1.1` will
# accept -- reaching it needs `ibv_reg_mr_iova2@IBVERBS_1.8`, which is the
# only reason NCCL calls that entry point at all
# (nccl:src/transport/net_ib/reg.cc:41-46).
comptime IBV_ACCESS_RELAXED_ORDERING: Int32 = 0x100000

# The three composite attr masks, spelled out at nccl:src/transport/net_ib/
# connect.cc:377 (INIT), :435+441 (RTR), :492+500 (RTS).
comptime QP_MASK_INIT: Int32 = 0x39  # STATE|ACCESS_FLAGS|PKEY_INDEX|PORT
comptime QP_MASK_RTR: Int32 = 0x129181
comptime QP_MASK_RTS: Int32 = 0x12E01

comptime IBV_WR_RDMA_WRITE: Int32 = 0
comptime IBV_WR_RDMA_WRITE_WITH_IMM: Int32 = 1
comptime IBV_WR_RDMA_READ: Int32 = 4
comptime IBV_SEND_SIGNALED: Int32 = 2

comptime IBV_WC_SUCCESS: Int32 = 0
comptime IBV_WC_RDMA_WRITE: Int32 = 1
comptime IBV_WC_RDMA_READ: Int32 = 2
comptime IBV_WC_RECV_RDMA_WITH_IMM: Int32 = 129

comptime IBV_PORT_ACTIVE: Int32 = 4
comptime IBV_LINK_LAYER_INFINIBAND: Int = 1

# ---- struct sizes and field offsets --------------------------------------
comptime SZ_PORT_ATTR = 56
comptime SZ_QP_INIT_ATTR = 64
comptime SZ_QP_ATTR = 144
comptime SZ_SGE = 16
comptime SZ_SEND_WR = 128
comptime SZ_RECV_WR = 32
comptime SZ_WC = 48

# struct ibv_context: the ops table is at +8; these are absolute offsets of
# the four live data-path slots (verified against a live mlx5 context).
comptime CTX_POLL_CQ = 96
comptime CTX_POST_SEND = 208
comptime CTX_POST_RECV = 216

comptime DEV_NAME = 24  # struct ibv_device.name[64]
comptime QP_CONTEXT = 0  # struct ibv_qp.context  (also cq.context, mr.context)
comptime QP_QP_NUM = 52
comptime MR_LKEY = 36
comptime MR_RKEY = 40

comptime PA_STATE = 0  # struct ibv_port_attr
comptime PA_ACTIVE_MTU = 8
comptime PA_GID_TBL_LEN = 12
comptime PA_LID = 34
comptime PA_LINK_LAYER = 46

comptime QIA_SEND_CQ = 8  # struct ibv_qp_init_attr
comptime QIA_RECV_CQ = 16
comptime QIA_MAX_SEND_WR = 32
comptime QIA_MAX_RECV_WR = 36
comptime QIA_MAX_SEND_SGE = 40
comptime QIA_MAX_RECV_SGE = 44
comptime QIA_QP_TYPE = 52

comptime QA_QP_STATE = 0  # struct ibv_qp_attr
comptime QA_PATH_MTU = 8
comptime QA_RQ_PSN = 20
comptime QA_SQ_PSN = 24
comptime QA_DEST_QP_NUM = 28
comptime QA_ACCESS_FLAGS = 32
comptime QA_AH_DGID = 56  # ah_attr.grh.dgid
comptime QA_AH_SGID_INDEX = 76
comptime QA_AH_HOP_LIMIT = 77
comptime QA_AH_DLID = 80
comptime QA_AH_SL = 82
comptime QA_AH_IS_GLOBAL = 85
comptime QA_AH_PORT_NUM = 86
comptime QA_PKEY_INDEX = 120
comptime QA_MAX_RD_ATOMIC = 126
comptime QA_MAX_DEST_RD_ATOMIC = 127
comptime QA_MIN_RNR_TIMER = 128
comptime QA_PORT_NUM = 129
comptime QA_TIMEOUT = 130
comptime QA_RETRY_CNT = 131
comptime QA_RNR_RETRY = 132

comptime SGE_ADDR = 0
comptime SGE_LENGTH = 8
comptime SGE_LKEY = 12

comptime WR_ID = 0  # struct ibv_send_wr (and ibv_recv_wr for the first 4)
comptime WR_NEXT = 8
comptime WR_SG_LIST = 16
comptime WR_NUM_SGE = 24
comptime WR_OPCODE = 28
comptime WR_SEND_FLAGS = 32
comptime WR_IMM_DATA = 36
comptime WR_RDMA_REMOTE_ADDR = 40
comptime WR_RDMA_RKEY = 48

comptime WC_WR_ID = 0  # struct ibv_wc
comptime WC_STATUS = 8
comptime WC_OPCODE = 12
comptime WC_VENDOR_ERR = 16
comptime WC_BYTE_LEN = 20
comptime WC_IMM_DATA = 24
comptime WC_QP_NUM = 28


# ===-------------------------------------------------------------------=== #
# The library handle (control path)
# ===-------------------------------------------------------------------=== #


struct Ibv(Movable):
    """The dlopened libibverbs.so.1.

    Control-path calls go through `get_function`; the data path never
    touches this struct.
    """

    var lib: OwnedDLHandle

    def __init__(out self) raises:
        self.lib = OwnedDLHandle("libibverbs.so.1")

    def get_device_list(self, out_n: P8) raises -> Int:
        return Int(self.lib.get_function[Int64]("ibv_get_device_list")(out_n))

    def free_device_list(self, list_addr: Int) raises:
        _ = self.lib.get_function[NoneType]("ibv_free_device_list")(list_addr)

    def open_device(self, dev: Int) raises -> Int:
        return Int(self.lib.get_function[Int64]("ibv_open_device")(dev))

    def close_device(self, ctx: Int) raises:
        _ = self.lib.get_function[Int32]("ibv_close_device")(ctx)

    def query_port(self, ctx: Int, port: Int, out_attr: P8) raises -> Int32:
        """The exported `ibv_query_port@IBVERBS_1.1` fills only the first 48
        of the 56 bytes (the `_compat_ibv_port_attr` layout; the extended
        entry point is `static inline`). Everything read here -- state@0,
        active_mtu@8, lid@34, link_layer@46 -- is inside those 48, and
        `alloc_bytes` zeroed the rest."""
        return self.lib.get_function[Int32]("ibv_query_port")(
            ctx, UInt8(port), out_attr
        )

    def query_gid(
        self, ctx: Int, port: Int, index: Int, out_gid: P8
    ) raises -> Int32:
        return self.lib.get_function[Int32]("ibv_query_gid")(
            ctx, UInt8(port), Int32(index), out_gid
        )

    def alloc_pd(self, ctx: Int) raises -> Int:
        return Int(self.lib.get_function[Int64]("ibv_alloc_pd")(ctx))

    def dealloc_pd(self, pd: Int) raises:
        _ = self.lib.get_function[Int32]("ibv_dealloc_pd")(pd)

    def reg_mr(
        self, pd: Int, addr: Int, length: Int, access: Int32
    ) raises -> Int:
        """Plain `ibv_reg_mr@IBVERBS_1.1`."""
        return Int(
            self.lib.get_function[Int64]("ibv_reg_mr")(
                pd, addr, UInt64(length), access
            )
        )

    def reg_mr_relaxed(
        self, pd: Int, addr: Int, length: Int, access: Int32
    ) raises -> Int:
        """`ibv_reg_mr_iova2` with IBV_ACCESS_RELAXED_ORDERING, falling back
        to `reg_mr`.

        PCIe relaxed ordering is what lets the NIC's writes into GPU memory
        retire out of order; without it a GPUDirect RDMA transfer runs at a
        fraction of link rate on this class of machine. NCCL turns it on by
        default (`NCCL_IB_PCI_RELAXED_ORDERING=2`,
        nccl:src/transport/net_ib/init.cc:11,141-150) and reaches it through
        `ibv_reg_mr_iova2` for the same ABI reason: the older entry point
        silently drops access bits above 0xFFFFF.

        The iova passed is the address itself -- the identity mapping NCCL
        also uses -- so remote addresses stay plain virtual addresses.
        Ordering of the DATA against its COMPLETION is not what RO relaxes
        and not what this library relies on: the flush read in
        transport/net.mojo is what makes the payload visible, and it is posted
        after the completion either way.
        """
        try:
            var mr = Int(
                self.lib.get_function[Int64]("ibv_reg_mr_iova2")(
                    pd,
                    addr,
                    UInt64(length),
                    UInt64(addr),
                    UInt32(access | IBV_ACCESS_RELAXED_ORDERING),
                )
            )
            if mr != 0:
                return mr
        except:
            # IBVERBS_1.8 absent (an old rdma-core): relaxed ordering is simply
            # unavailable, so register without it.
            return self.reg_mr(pd, addr, length, access)
        return self.reg_mr(pd, addr, length, access)

    def dereg_mr(self, mr: Int) raises:
        _ = self.lib.get_function[Int32]("ibv_dereg_mr")(mr)

    def create_cq(self, ctx: Int, cqe: Int) raises -> Int:
        return Int(
            self.lib.get_function[Int64]("ibv_create_cq")(
                ctx, Int32(cqe), Int64(0), Int64(0), Int32(0)
            )
        )

    def destroy_cq(self, cq: Int) raises:
        _ = self.lib.get_function[Int32]("ibv_destroy_cq")(cq)

    def create_qp(self, pd: Int, init_attr: P8) raises -> Int:
        return Int(self.lib.get_function[Int64]("ibv_create_qp")(pd, init_attr))

    def destroy_qp(self, qp: Int) raises:
        _ = self.lib.get_function[Int32]("ibv_destroy_qp")(qp)

    def modify_qp(self, qp: Int, attr: P8, mask: Int32) raises -> Int32:
        return self.lib.get_function[Int32]("ibv_modify_qp")(qp, attr, mask)


# ===-------------------------------------------------------------------=== #
# Data path -- ctx->ops.<fn>, no dlsym
# ===-------------------------------------------------------------------=== #


@always_inline
def post_send(qp: Int, wr: P8, bad_wr: P8) -> Int32:
    """`qp->context->ops.post_send(qp, wr, &bad_wr)`; 0 or an errno."""
    var f = as_fn[def(Int, P8, P8) thin abi("C") -> Int32](
        ld64(
            P8(
                unsafe_from_address=ld64(P8(unsafe_from_address=qp), QP_CONTEXT)
            ),
            CTX_POST_SEND,
        )
    )
    return f(qp, wr, bad_wr)


@always_inline
def post_recv(qp: Int, wr: P8, bad_wr: P8) -> Int32:
    var f = as_fn[def(Int, P8, P8) thin abi("C") -> Int32](
        ld64(
            P8(
                unsafe_from_address=ld64(P8(unsafe_from_address=qp), QP_CONTEXT)
            ),
            CTX_POST_RECV,
        )
    )
    return f(qp, wr, bad_wr)


@always_inline
def poll_cq(cq: Int, num_entries: Int, wc: P8) -> Int32:
    """Number of completions written into `wc`, or negative on error."""
    var f = as_fn[def(Int, Int32, P8) thin abi("C") -> Int32](
        ld64(
            P8(
                unsafe_from_address=ld64(P8(unsafe_from_address=cq), QP_CONTEXT)
            ),
            CTX_POLL_CQ,
        )
    )
    return f(cq, Int32(num_entries), wc)


# ===-------------------------------------------------------------------=== #
# Work-request builders
# ===-------------------------------------------------------------------=== #


def build_write_wr(
    wr: P8,
    sge: P8,
    wr_id: Int,
    local_addr: Int,
    lkey: UInt32,
    nbytes: Int,
    remote_addr: Int,
    rkey: UInt32,
    immediate: UInt32,
    with_imm: Bool,
    signaled: Bool,
):
    """One unchained RDMA_WRITE[_WITH_IMM]. `wr`/`sge` are caller-owned
    scratch reused across calls, so every field is written every time
    rather than relying on what was there before."""
    st64(sge, SGE_ADDR, local_addr)
    stu32(sge, SGE_LENGTH, UInt32(nbytes))
    stu32(sge, SGE_LKEY, lkey)
    for i in range(SZ_SEND_WR):
        wr[unsafe_offset=i] = 0
    st64(wr, WR_ID, wr_id)
    st64(wr, WR_NEXT, 0)
    st64(wr, WR_SG_LIST, Int(sge))
    st32(wr, WR_NUM_SGE, 1)
    st32(
        wr,
        WR_OPCODE,
        IBV_WR_RDMA_WRITE_WITH_IMM if with_imm else IBV_WR_RDMA_WRITE,
    )
    st32(wr, WR_SEND_FLAGS, IBV_SEND_SIGNALED if signaled else 0)
    # imm_data is __be32 on both sides (nccl:...:p2p.cc:157 htobe32, :653
    # be32toh). Byte-swapped here so the receiver's plain load reads it back.
    stu32(wr, WR_IMM_DATA, _bswap32(immediate))
    st64(wr, WR_RDMA_REMOTE_ADDR, remote_addr)
    stu32(wr, WR_RDMA_RKEY, rkey)


def build_read_wr(
    wr: P8,
    sge: P8,
    wr_id: Int,
    local_addr: Int,
    lkey: UInt32,
    nbytes: Int,
    remote_addr: Int,
    rkey: UInt32,
):
    """A signaled RDMA_READ -- the GPUDirect flush (see transport/net.mojo)."""
    st64(sge, SGE_ADDR, local_addr)
    stu32(sge, SGE_LENGTH, UInt32(nbytes))
    stu32(sge, SGE_LKEY, lkey)
    for i in range(SZ_SEND_WR):
        wr[unsafe_offset=i] = 0
    st64(wr, WR_ID, wr_id)
    st64(wr, WR_SG_LIST, Int(sge))
    st32(wr, WR_NUM_SGE, 1)
    st32(wr, WR_OPCODE, IBV_WR_RDMA_READ)
    st32(wr, WR_SEND_FLAGS, IBV_SEND_SIGNALED)
    st64(wr, WR_RDMA_REMOTE_ADDR, remote_addr)
    stu32(wr, WR_RDMA_RKEY, rkey)


def build_recv_wr(wr: P8, wr_id: Int):
    """An EMPTY receive WR (`sg_list = NULL, num_sge = 0`), which is what
    NCCL posts too (nccl:src/transport/net_ib/common.cc:88): its only job is
    to be consumed by an incoming RDMA_WRITE_WITH_IMM so a completion with
    the immediate appears on the CQ. The payload went straight to the
    registered region."""
    for i in range(SZ_RECV_WR):
        wr[unsafe_offset=i] = 0
    st64(wr, WR_ID, wr_id)


@always_inline
def _bswap32(v: UInt32) -> UInt32:
    return (
        ((v & 0xFF) << 24)
        | ((v & 0xFF00) << 8)
        | ((v >> 8) & 0xFF00)
        | ((v >> 24) & 0xFF)
    )


@always_inline
def be32(v: UInt32) -> UInt32:
    """Host <-> big-endian for the 32-bit immediate (involutive)."""
    return _bswap32(v)


def qp_number(qp: Int) -> UInt32:
    return ldu32(P8(unsafe_from_address=qp), QP_QP_NUM)
