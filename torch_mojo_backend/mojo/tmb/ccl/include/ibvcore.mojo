# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/include/ibvcore.h
#
# Every struct offset below was dumped with gcc offsetof against
# /usr/include/infiniband/verbs.h on this machine (rdma-core 1.14.54.0 /
# MLNX OFED 24.10) and the ops-table offsets were additionally read back
# from a live mlx5 context; enum values likewise. Nothing here is from
# memory. x86-64 SysV: the structs are built in raw byte buffers because
# std.ffi still has no C-struct ABI (MOCO-3692).


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
