# libfabric bindings for mojoccl's inter-node hop -- the second transport,
# for fabrics that libibverbs cannot see. Written for and measured against
# the HPE Slingshot `cxi` provider (Adastra/CINES MI300A nodes have four
# /dev/cxi NICs and no /dev/infiniband at all), but nothing here is
# cxi-specific beyond the provider preference.
#
# Same two calling conventions as `ibverbs.mojo`, for the same reason:
#
#  * control path -- real exported symbols (fi_getinfo, fi_freeinfo,
#    fi_fabric, fi_version, fi_strerror), reached with
#    OwnedDLHandle.get_function.
#  * data path -- NOT exported. fi_writemsg, fi_sendmsg, fi_recv, fi_read,
#    fi_cq_read, fi_mr_regattr, fi_ep_bind, fi_close ... are all `static
#    inline` in <rdma/fi_*.h> and dispatch through the ops table hanging off
#    the object: `ep->rma->writemsg`, `ep->msg->sendmsg`, `cq->ops->read`,
#    `domain->mr->regattr`, `fid->ops->bind`. This file hand-writes those
#    dereferences over raw byte offsets, exactly as ibverbs.mojo does for
#    `qp->context->ops.post_send`.
#
# Every offset, size and constant below was dumped by
# tests/multinode/selftest/fabric_abi.c compiled against
# /opt/cray/libfabric/2.2.0rc1/include (libfabric 2.2, x86-64 SysV), and
# tests/multinode/selftest/fabric_abi.mojo re-checks the Mojo copies against
# that program's output. Nothing here is from memory.
#
# WHAT THE cxi PROVIDER DOES NOT HAVE, and what this file does instead:
#
#  * `fi_writedata` -- RDMA-write-with-immediate, the operation the verbs
#    transport is built on -- is NOT implemented by cxi
#    (libfabric:prov/cxi/src/cxip_rma.c, `cxip_ep_rma_ops.writedata =
#    fi_no_rma_writedata`, and `cxip_rma_writemsg` rejects
#    FI_REMOTE_CQ_DATA because it is not in CXIP_WRITEMSG_ALLOWED_FLAGS,
#    prov/cxi/include/cxip.h). Measured: `fi_writedata` returns -FI_ENOSYS
#    and `fi_writemsg(..., FI_REMOTE_CQ_DATA)` returns -FI_EBADFLAGS on
#    Adastra's cxi 2.2.0rc1.
#    So one write-with-immediate becomes TWO operations: a plain
#    `fi_writemsg` of the payload, then a zero-length `fi_sendmsg` carrying
#    the immediate as remote CQ data. cxi does support FI_REMOTE_CQ_DATA on
#    the MESSAGE path (CXIP_TX_OP_FLAGS, and cq_data_size is 8).
#  * ordering between those two: the message must not be delivered before
#    the payload it announces. `FI_FENCE` ("the fenced operation ... will be
#    deferred until all previous operations targeting the same peer
#    endpoint have completed", fi_endpoint(3)) is set on the first
#    notification of an exchange; cxi implements it as a hardware
#    C_CMD_CQ_FENCE on the transmit command queue
#    (prov/cxi/src/cxip_cmdq.c), which drains everything issued earlier, so
#    the remaining notifications of the same exchange are ordered by queue
#    position alone. FI_FENCE has to be requested in `caps` or cxi returns
#    -FI_EINVAL (prov/cxi/src/cxip_msg.c:956).
#  * `FI_SOURCE` is not offered, so a completion does not say who sent it.
#    The 64-bit remote CQ data carries `(node index << 32) | immediate`
#    instead; the 32-bit immediate keeps exactly the meaning it has on the
#    verbs path (bit 31 credit, bits 0..30 the exchange counter).
#  * `mr_mode` is FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_ENDPOINT and does
#    NOT contain FI_MR_VIRT_ADDR: remote addresses are OFFSETS into the
#    peer's memory region, not virtual addresses; the key is chosen by the
#    provider (`fi_mr_key`), and every MR must be bound to the endpoint and
#    enabled before use. Both addressing modes are handled at runtime
#    (`FabricNet.virt_addr`) rather than assumed.

from std.ffi import OwnedDLHandle
from std.os import getenv
from std.sys import size_of
from std.time import perf_counter_ns, sleep

from tmb.ccl.env_vars import (
    MOJOCCL_FABRIC_DOMAIN,
    MOJOCCL_FABRIC_PROVIDER,
    MOJOCCL_LIBFABRIC,
)
from tmb.ccl.netutil import (
    P8,
    alloc_bytes,
    as_fn,
    c_string,
    free_bytes,
    ld32,
    ld64,
    ldu32,
    ldu64,
    pci_pick,
    read_c_string,
    st32,
    st64,
    stu32,
    stu64,
    NC_FLUSH,
    NC_OTHER,
    NC_RECV,
    NC_SEND,
    NetCompletion,
)

# ---- struct sizes (fabric_abi.c) -----------------------------------------
comptime SZ_FI_INFO = 120
comptime SZ_FI_TX_ATTR = 80
comptime SZ_FI_RX_ATTR = 64
comptime SZ_FI_EP_ATTR = 96
comptime SZ_FI_DOMAIN_ATTR = 208
comptime SZ_FI_FABRIC_ATTR = 32
comptime SZ_FI_CQ_ATTR = 40
comptime SZ_FI_AV_ATTR = 48
comptime SZ_FI_MR_ATTR = 112
comptime SZ_FI_MSG_RMA = 64
comptime SZ_FI_MSG = 48
comptime SZ_FI_RMA_IOV = 24
comptime SZ_IOVEC = 16
comptime SZ_CQ_ENTRY = 40  # struct fi_cq_data_entry
comptime SZ_CQ_ERR = 88  # struct fi_cq_err_entry

# ---- struct fi_info ------------------------------------------------------
comptime INFO_NEXT = 0
comptime INFO_CAPS = 8
comptime INFO_MODE = 16
comptime INFO_ADDR_FORMAT = 24
comptime INFO_SRC_ADDRLEN = 32
comptime INFO_SRC_ADDR = 48
comptime INFO_TX_ATTR = 72
comptime INFO_RX_ATTR = 80
comptime INFO_EP_ATTR = 88
comptime INFO_DOMAIN_ATTR = 96
comptime INFO_FABRIC_ATTR = 104

comptime TXA_CAPS = 0
comptime TXA_SIZE = 48
comptime RXA_CAPS = 0
comptime RXA_SIZE = 48
comptime EPA_TYPE = 0
comptime EPA_MAX_MSG_SIZE = 16
comptime DA_NAME = 8
comptime DA_THREADING = 16
comptime DA_MR_MODE = 36
comptime DA_MR_KEY_SIZE = 40
comptime DA_CQ_DATA_SIZE = 48
comptime FA_NAME = 8
comptime FA_PROV_NAME = 16
comptime FA_API_VERSION = 28

# ---- attribute structs we build ------------------------------------------
comptime CQA_SIZE = 0
comptime CQA_FORMAT = 16
comptime CQA_WAIT_OBJ = 20
comptime AVA_TYPE = 0
comptime AVA_COUNT = 8

comptime MRA_MR_IOV = 0
comptime MRA_IOV_COUNT = 8
comptime MRA_ACCESS = 16
comptime MRA_OFFSET = 24
comptime MRA_REQUESTED_KEY = 32
comptime MRA_IFACE = 64

comptime IOV_BASE = 0
comptime IOV_LEN = 8
comptime RMAIOV_ADDR = 0
comptime RMAIOV_LEN = 8
comptime RMAIOV_KEY = 16

comptime MSGRMA_MSG_IOV = 0
comptime MSGRMA_DESC = 8
comptime MSGRMA_IOV_COUNT = 16
comptime MSGRMA_ADDR = 24
comptime MSGRMA_RMA_IOV = 32
comptime MSGRMA_RMA_IOV_COUNT = 40
comptime MSGRMA_CONTEXT = 48

comptime MSG_MSG_IOV = 0
comptime MSG_DESC = 8
comptime MSG_IOV_COUNT = 16
comptime MSG_ADDR = 24
comptime MSG_CONTEXT = 32
comptime MSG_DATA = 40

# ---- completions ---------------------------------------------------------
comptime CQE_OP_CONTEXT = 0
comptime CQE_FLAGS = 8
comptime CQE_LEN = 16
comptime CQE_DATA = 32
comptime CQERR_ERR = 56
comptime CQERR_PROV_ERRNO = 60

# ---- the fid_* objects and their ops tables ------------------------------
comptime FID_OPS = 16  # struct fid.ops
comptime FIDEP_CM = 32  # struct fid_ep.cm
comptime FIDEP_MSG = 40
comptime FIDEP_RMA = 48
comptime FIDMR_MEM_DESC = 24
comptime FIDMR_KEY = 32
comptime FIDDOM_OPS = 24
comptime FIDDOM_MR = 32
comptime FIDCQ_OPS = 24
comptime FIDAV_OPS = 24
comptime FIDFAB_OPS = 24

comptime OPS_CLOSE = 8  # struct fi_ops
comptime OPS_BIND = 16
comptime OPS_CONTROL = 24
comptime FABOPS_DOMAIN = 8
comptime DOMOPS_AV_OPEN = 8
comptime DOMOPS_CQ_OPEN = 16
comptime DOMOPS_ENDPOINT = 24
comptime MROPS_REGATTR = 24
comptime CQOPS_READ = 8
comptime CQOPS_READERR = 24
comptime AVOPS_INSERT = 8
comptime CMOPS_GETNAME = 16
comptime RMAOPS_READ = 8
comptime RMAOPS_WRITEMSG = 48
comptime MSGOPS_RECV = 8
comptime MSGOPS_SENDMSG = 48

# ---- capability / flag / enum constants ----------------------------------
comptime FI_MSG: UInt64 = 1 << 1
comptime FI_RMA: UInt64 = 1 << 2
comptime FI_READ: UInt64 = 1 << 8
comptime FI_WRITE: UInt64 = 1 << 9
comptime FI_RECV: UInt64 = 1 << 10
comptime FI_SEND: UInt64 = 1 << 11
comptime FI_TRANSMIT: UInt64 = FI_SEND  # fabric.h: `#define FI_TRANSMIT FI_SEND`
comptime FI_REMOTE_READ: UInt64 = 1 << 12
comptime FI_REMOTE_WRITE: UInt64 = 1 << 13
comptime FI_REMOTE_CQ_DATA: UInt64 = 1 << 17
comptime FI_FENCE: UInt64 = 1 << 21
comptime FI_COMPLETION: UInt64 = 1 << 24
comptime FI_DELIVERY_COMPLETE: UInt64 = 1 << 28
comptime FI_CONTEXT: UInt64 = 1 << 59
comptime FI_CONTEXT2: UInt64 = 1 << 52
comptime FI_LOCAL_COMM: UInt64 = 1 << 51
comptime FI_REMOTE_COMM: UInt64 = 1 << 52
comptime FI_HMEM: UInt64 = 1 << 47

comptime FI_MR_LOCAL: Int32 = 1 << 2
comptime FI_MR_VIRT_ADDR: Int32 = 1 << 4
comptime FI_MR_ALLOCATED: Int32 = 1 << 5
comptime FI_MR_PROV_KEY: Int32 = 1 << 6
comptime FI_MR_ENDPOINT: Int32 = 1 << 9
comptime FI_MR_HMEM: Int32 = 1 << 10

comptime FI_EP_RDM: Int32 = 3
comptime FI_AV_TABLE: Int32 = 2
comptime FI_CQ_FORMAT_DATA: Int32 = 3
comptime FI_WAIT_NONE: Int32 = 0
comptime FI_THREAD_SAFE: Int32 = 1
comptime FI_HMEM_SYSTEM = 0
comptime FI_HMEM_ROCR = 2
comptime FI_HMEM_CUDA = 1
comptime FI_ADDR_UNSPEC: UInt64 = 0xFFFF_FFFF_FFFF_FFFF
comptime FI_ENABLE: Int32 = 6

comptime FI_EAGAIN = 11
comptime FI_ENOMEM = 12
comptime FI_EAVAIL = 259
comptime FI_ENOSYS = 38

comptime FAB_SETUP_RETRY_S: Float64 = 30.0
"""How long to keep asking when the endpoint bring-up says -FI_ENOMEM.

A budget rather than a try count, because the condition this covers is a
node-wide window rather than a collision between the ranks: measured on
Adastra, eight tries over 2.5 s all failed on all four of a node's ranks,
and the same batch passed a run later. 30 s is under the bootstrap's own
deadline, so a node that spends its whole budget retrying still meets its
peers rather than timing the job out."""

comptime FAB_SETUP_BACKOFF_US = 20_000
comptime FAB_SETUP_BACKOFF_MAX_US = 1_000_000
"""Doubling backoff, 20 ms .. 1 s. Nothing here is fitted to a measurement;
it is only "ask often at first, then stop burning a core"."""

comptime FI_VERSION_2_2: UInt32 = (2 << 16) | 2
comptime FI_VERSION_1_5: UInt32 = (1 << 16) | 5

# The caps this transport asks for. Deliberately NO primary modifiers
# (FI_READ/FI_WRITE/FI_SEND/...): libfabric hands back every modifier when
# none is named, and naming a subset would silently drop the rest -- the
# same reasoning as aws-ofi-nccl's `get_hints`
# (aws-ofi-nccl:src/nccl_ofi_rdma.cpp). FI_FENCE has to be named because cxi
# checks `txc->attr.caps & FI_FENCE` before honouring the flag.
comptime WANT_CAPS: UInt64 = (
    FI_MSG | FI_RMA | FI_HMEM | FI_LOCAL_COMM | FI_REMOTE_COMM | FI_FENCE
)
# The mr_modes this transport knows how to satisfy. The provider answers
# with the subset it actually requires.
comptime WANT_MR_MODE: Int32 = (
    FI_MR_LOCAL
    | FI_MR_HMEM
    | FI_MR_VIRT_ADDR
    | FI_MR_ALLOCATED
    | FI_MR_PROV_KEY
    | FI_MR_ENDPOINT
)

# Where the CQ data splits: low 32 bits are the immediate the engine
# understands, high 32 bits are the sender's node index (cxi has no
# FI_SOURCE).
comptime CQ_DATA_NODE_SHIFT = 32

# Completion contexts are opaque 64-bit values, not pointers -- legal
# because `fi_getinfo` is required to return mode without FI_CONTEXT /
# FI_CONTEXT2 (checked in `fab_setup`, which raises otherwise). The engine
# needs (what kind of operation, which peer, which exchange) back out of a
# completion, so they are packed into one word.
comptime CTX_KIND_SHIFT = 56
comptime CTX_PEER_SHIFT = 40
comptime CTX_SEQ_MASK = (1 << 40) - 1

comptime EP_NAME_MAX = 80  # what `IB_BLOB_BYTES` leaves for an endpoint name
comptime FLUSH_PAD_BYTES = 4096
comptime FLUSH_EP_CAPS: UInt64 = FI_RMA | FI_READ
"""All the flush endpoint does is initiate one RMA read. No FI_MSG, no
receive and no remote access: it never receives, and the read's target is
the data endpoint, which serves the region's MR."""
comptime RECV_BUF_BYTES = 64
# Completion-queue depth. Far above what can be outstanding (flow control
# caps it at `nslots` exchanges, each worth two transmit completions and two
# receive completions per peer) and deliberately not larger: the cxi
# provider sizes its hugetlbfs event-queue allocation from this, and these
# nodes have no static hugepage pool.
comptime CQ_DEPTH = 2048


@always_inline
def _pack_ctx(kind: Int, peer: Int, seq: Int) -> Int:
    return (
        (kind << CTX_KIND_SHIFT)
        | (peer << CTX_PEER_SHIFT)
        | (seq & CTX_SEQ_MASK)
    )


@always_inline
def _ctx_kind(c: Int) -> Int:
    return (c >> CTX_KIND_SHIFT) & 0xFF


@always_inline
def _ctx_peer(c: Int) -> Int:
    return (c >> CTX_PEER_SHIFT) & 0xFFFF


@always_inline
def _ctx_seq(c: Int) -> Int:
    return c & CTX_SEQ_MASK


# ===-------------------------------------------------------------------=== #
# The library handle (control path)
# ===-------------------------------------------------------------------=== #

# Tried in order. The soname first (an LD_LIBRARY_PATH set by `module load
# libfabric` finds it), then the Cray PE location Adastra installs it at
# even when nothing is loaded.
comptime FABRIC_SONAME = "libfabric.so.1"
comptime FABRIC_CRAY_PATH = "/opt/cray/libfabric/2.2.0rc1/lib64/libfabric.so.1"


struct Fab(Movable):
    """The dlopened libfabric.so.1. Only the exported entry points."""

    var lib: OwnedDLHandle

    def __init__(out self) raises:
        var want = getenv(MOJOCCL_LIBFABRIC, "")
        if want.byte_length() > 0:
            self.lib = OwnedDLHandle(want)
            return
        try:
            self.lib = OwnedDLHandle(FABRIC_SONAME)
        except:
            self.lib = OwnedDLHandle(FABRIC_CRAY_PATH)

    def version(self) raises -> UInt32:
        return self.lib.get_function[UInt32]("fi_version")()

    def getinfo(
        self, version: UInt32, flags: UInt64, hints: P8, out_info: P8
    ) raises -> Int32:
        """`fi_getinfo(version, NULL, NULL, flags, hints, &info)`."""
        return self.lib.get_function[Int32]("fi_getinfo")(
            version, Int64(0), Int64(0), flags, hints, out_info
        )

    def freeinfo(self, info: Int) raises:
        _ = self.lib.get_function[NoneType]("fi_freeinfo")(info)

    def dupinfo(self, info: Int) raises -> Int:
        """`fi_dupinfo(info)`: a deep copy, freed with `freeinfo`; 0 if the
        library could not allocate it."""
        return Int(self.lib.get_function[Int64]("fi_dupinfo")(info))

    def fabric(self, attr: Int, out_fabric: P8) raises -> Int32:
        return self.lib.get_function[Int32]("fi_fabric")(
            attr, out_fabric, Int64(0)
        )

    def strerror(self, err: Int) raises -> String:
        """`fi_strerror` takes a POSITIVE error number; every libfabric call
        here returns the negative of one."""
        return read_c_string(
            Int(self.lib.get_function[Int64]("fi_strerror")(Int32(err))), 256
        )


# ===-------------------------------------------------------------------=== #
# Data path -- the ops tables, no dlsym
# ===-------------------------------------------------------------------=== #


@always_inline
def _slot(obj: Int, table_off: Int, fn_off: Int) -> Int:
    """The function pointer at `fn_off` in the ops table `obj` points at
    from `table_off`."""
    return ld64(
        P8(unsafe_from_address=ld64(P8(unsafe_from_address=obj), table_off)),
        fn_off,
    )


@always_inline
def fi_close(fid: Int) -> Int32:
    """`fid->ops->close(fid)`."""
    if fid == 0:
        return 0
    return as_fn[def(Int) thin abi("C") -> Int32](
        _slot(fid, FID_OPS, OPS_CLOSE)
    )(fid)


@always_inline
def fi_bind(fid: Int, bfid: Int, flags: UInt64) -> Int32:
    """`fid->ops->bind(fid, bfid, flags)` -- `fi_ep_bind` and `fi_mr_bind`
    are both this one inline."""
    return as_fn[def(Int, Int, UInt64) thin abi("C") -> Int32](
        _slot(fid, FID_OPS, OPS_BIND)
    )(fid, bfid, flags)


@always_inline
def fi_enable(fid: Int) -> Int32:
    """`fi_control(fid, FI_ENABLE, NULL)` -- `fi_enable` (endpoint) and
    `fi_mr_enable` are both this one inline."""
    return as_fn[def(Int, Int32, Int64) thin abi("C") -> Int32](
        _slot(fid, FID_OPS, OPS_CONTROL)
    )(fid, FI_ENABLE, Int64(0))


@always_inline
def fi_domain(fabric: Int, info: Int, out_domain: P8) -> Int32:
    return as_fn[def(Int, Int, P8, Int64) thin abi("C") -> Int32](
        _slot(fabric, FIDFAB_OPS, FABOPS_DOMAIN)
    )(fabric, info, out_domain, Int64(0))


@always_inline
def fi_cq_open(domain: Int, attr: P8, out_cq: P8) -> Int32:
    return as_fn[def(Int, P8, P8, Int64) thin abi("C") -> Int32](
        _slot(domain, FIDDOM_OPS, DOMOPS_CQ_OPEN)
    )(domain, attr, out_cq, Int64(0))


@always_inline
def fi_av_open(domain: Int, attr: P8, out_av: P8) -> Int32:
    return as_fn[def(Int, P8, P8, Int64) thin abi("C") -> Int32](
        _slot(domain, FIDDOM_OPS, DOMOPS_AV_OPEN)
    )(domain, attr, out_av, Int64(0))


@always_inline
def fi_endpoint(domain: Int, info: Int, out_ep: P8) -> Int32:
    return as_fn[def(Int, Int, P8, Int64) thin abi("C") -> Int32](
        _slot(domain, FIDDOM_OPS, DOMOPS_ENDPOINT)
    )(domain, info, out_ep, Int64(0))


@always_inline
def fi_mr_regattr(domain: Int, attr: P8, flags: UInt64, out_mr: P8) -> Int32:
    """`domain->mr->regattr(&domain->fid, attr, flags, &mr)`; `&domain->fid`
    is the domain pointer itself (fid is the first member)."""
    return as_fn[def(Int, P8, UInt64, P8) thin abi("C") -> Int32](
        _slot(domain, FIDDOM_MR, MROPS_REGATTR)
    )(domain, attr, flags, out_mr)


@always_inline
def fi_mr_key(mr: Int) -> UInt64:
    return ldu64(P8(unsafe_from_address=mr), FIDMR_KEY)


@always_inline
def fi_mr_desc(mr: Int) -> Int:
    return ld64(P8(unsafe_from_address=mr), FIDMR_MEM_DESC)


@always_inline
def fi_getname(ep: Int, addr: P8, addrlen: P8) -> Int32:
    """`ep->cm->getname(&ep->fid, addr, &addrlen)`."""
    return as_fn[def(Int, P8, P8) thin abi("C") -> Int32](
        _slot(ep, FIDEP_CM, CMOPS_GETNAME)
    )(ep, addr, addrlen)


@always_inline
def fi_av_insert(av: Int, addr: P8, count: Int, out_addrs: P8) -> Int32:
    return as_fn[def(Int, P8, Int64, P8, UInt64, Int64) thin abi("C") -> Int32](
        _slot(av, FIDAV_OPS, AVOPS_INSERT)
    )(av, addr, Int64(count), out_addrs, UInt64(0), Int64(0))


@always_inline
def fi_cq_read(cq: Int, buf: P8, count: Int) -> Int:
    """Number of completions written into `buf`, or -FI_EAGAIN / -FI_EAVAIL
    / another negative libfabric error."""
    return Int(
        as_fn[def(Int, P8, Int64) thin abi("C") -> Int64](
            _slot(cq, FIDCQ_OPS, CQOPS_READ)
        )(cq, buf, Int64(count))
    )


@always_inline
def fi_cq_readerr(cq: Int, buf: P8) -> Int:
    return Int(
        as_fn[def(Int, P8, UInt64) thin abi("C") -> Int64](
            _slot(cq, FIDCQ_OPS, CQOPS_READERR)
        )(cq, buf, UInt64(0))
    )


@always_inline
def fi_writemsg(ep: Int, msg: P8, flags: UInt64) -> Int:
    return Int(
        as_fn[def(Int, P8, UInt64) thin abi("C") -> Int64](
            _slot(ep, FIDEP_RMA, RMAOPS_WRITEMSG)
        )(ep, msg, flags)
    )


@always_inline
def fi_read(
    ep: Int,
    buf: Int,
    nbytes: Int,
    desc: Int,
    src_addr: UInt64,
    addr: UInt64,
    key: UInt64,
    context: Int,
) -> Int:
    return Int(
        as_fn[
            def(
                Int, Int, Int64, Int, UInt64, UInt64, UInt64, Int
            ) thin abi("C") -> Int64
        ](_slot(ep, FIDEP_RMA, RMAOPS_READ))(
            ep, buf, Int64(nbytes), desc, src_addr, addr, key, context
        )
    )


@always_inline
def fi_sendmsg(ep: Int, msg: P8, flags: UInt64) -> Int:
    return Int(
        as_fn[def(Int, P8, UInt64) thin abi("C") -> Int64](
            _slot(ep, FIDEP_MSG, MSGOPS_SENDMSG)
        )(ep, msg, flags)
    )


@always_inline
def fi_recv(
    ep: Int, buf: Int, nbytes: Int, desc: Int, src_addr: UInt64, context: Int
) -> Int:
    return Int(
        as_fn[def(Int, Int, Int64, Int, UInt64, Int) thin abi("C") -> Int64](
            _slot(ep, FIDEP_MSG, MSGOPS_RECV)
        )(ep, buf, Int64(nbytes), desc, src_addr, context)
    )


# ===-------------------------------------------------------------------=== #
# The transport state
# ===-------------------------------------------------------------------=== #

# Completions are buffered here rather than handed straight to the engine,
# because a post that hits -FI_EAGAIN has to make progress to free a command
# slot -- and with FI_PROGRESS_MANUAL the only way to progress is to read the
# completion queue. Anything read during such a retry would otherwise be
# lost, so everything goes through this ring and `fab_poll` pops from it.
comptime BACKLOG_CAP = 1024
# How long a post may keep retrying -FI_EAGAIN before it is called a failure.
# Flow control already bounds what is outstanding (`nslots` exchanges), so
# this is only reached when something is genuinely wedged.
comptime EAGAIN_SPINS = 1_000_000


struct FabricNet(Movable):
    """Everything the libfabric transport owns, per communicator.

    One RDM endpoint (two when the second comes up: it only issues the
    flush read, `_flush_ep_open`), one completion queue for both
    directions, one
    address vector holding this rank's own address (for the flush read) and
    one entry per remote node. Two memory regions: the communicator's device
    region, and a host scratch area holding the flush landing pad and the
    receive buffers.
    """

    var fab: Fab
    var info_list: Int  # head of the fi_getinfo list, for fi_freeinfo
    var info: Int  # the entry chosen for this rank (inside that list)
    var fabric: Int
    var domain: Int
    var ep: Int
    var cq: Int
    var av: Int
    var mr: Int
    var mr_key: UInt64
    var mr_desc: Int
    var host: Int
    var host_bytes: Int
    var host_mr: Int
    var host_desc: Int
    var domain_name: String
    var prov_name: String
    var api_version: UInt32
    var addrlen: Int
    var my_name: Int  # EP_NAME_MAX scratch holding fi_getname's answer
    var virt_addr: Bool  # FI_MR_VIRT_ADDR: remote addresses are VAs, not offsets
    var need_endpoint_mr: Bool
    var flush_ep: Int  # 0 if `_flush_ep_open` failed: the flush uses `ep`
    var flush_mr: Int  # its landing pad's MR under FI_MR_ENDPOINT, or 0
    var flush_desc: Int
    var flush_info: Int  # `fi_dupinfo` copy flush_ep was opened from, or 0
    var hmem_iface: Int
    var recv_depth: Int
    var self_addr: UInt64
    var addrs: List[UInt64]  # fi_addr_t per peer, parallel to IbState.peers
    var peer_of_node: List[Int]  # node index -> peer index, -1 if none
    var my_node: Int
    var region: Int
    var last_error: Int
    # Scratch for the structs every post builds (caller-owned, reused).
    var iov: Int
    var rma_iov: Int
    var desc_slot: Int
    var msg_rma: Int
    var msg: Int
    var cqe: Int
    var cqerr: Int
    var lenbuf: Int
    var addrbuf: Int
    # The completion ring.
    var backlog: Int
    var bl_head: Int
    var bl_count: Int

    def __init__(out self, var fab: Fab, region: Int, my_node: Int):
        self.fab = fab^
        self.info_list = 0
        self.info = 0
        self.fabric = 0
        self.domain = 0
        self.ep = 0
        self.cq = 0
        self.av = 0
        self.mr = 0
        self.mr_key = 0
        self.mr_desc = 0
        self.host = 0
        self.host_bytes = 0
        self.host_mr = 0
        self.host_desc = 0
        self.domain_name = String("")
        self.prov_name = String("")
        self.api_version = 0
        self.addrlen = 0
        self.my_name = Int(alloc_bytes(EP_NAME_MAX))
        self.virt_addr = False
        self.need_endpoint_mr = False
        self.flush_ep = 0
        self.flush_mr = 0
        self.flush_desc = 0
        self.flush_info = 0
        self.hmem_iface = FI_HMEM_SYSTEM
        self.recv_depth = 0
        self.self_addr = FI_ADDR_UNSPEC
        self.addrs = List[UInt64]()
        self.peer_of_node = List[Int]()
        self.my_node = my_node
        self.region = region
        self.last_error = 0
        self.iov = Int(alloc_bytes(SZ_IOVEC))
        self.rma_iov = Int(alloc_bytes(SZ_FI_RMA_IOV))
        self.desc_slot = Int(alloc_bytes(8))
        self.msg_rma = Int(alloc_bytes(SZ_FI_MSG_RMA))
        self.msg = Int(alloc_bytes(SZ_FI_MSG))
        self.cqe = Int(alloc_bytes(SZ_CQ_ENTRY * 16))
        self.cqerr = Int(alloc_bytes(SZ_CQ_ERR))
        self.lenbuf = Int(alloc_bytes(8))
        self.addrbuf = Int(alloc_bytes(8))
        self.backlog = Int(alloc_bytes(BACKLOG_CAP * _size_of_completion()))
        self.bl_head = 0
        self.bl_count = 0


@always_inline
def _size_of_completion() -> Int:
    """`NetCompletion` is five machine words; the ring is a byte buffer of
    them rather than a `List` so it can be indexed without bounds checks
    from the progress thread."""
    return 5 * 8


@always_inline
def _bl_slot(f: FabricNet, i: Int) -> P8:
    return P8(
        unsafe_from_address=f.backlog
        + (i % BACKLOG_CAP) * _size_of_completion()
    )


def _bl_push(
    mut f: FabricNet,
    kind: Int,
    peer: Int,
    immediate: UInt32,
    wr_id: Int,
    status: Int,
):
    if f.bl_count >= BACKLOG_CAP:
        # Cannot happen while flow control holds (the ring is far larger than
        # the number of operations that can be outstanding), and dropping a
        # completion would be a silent hang -- latch it instead.
        f.last_error = -1
        return
    var s = _bl_slot(f, f.bl_head + f.bl_count)
    st64(s, 0, kind)
    st64(s, 8, peer)
    stu32(s, 16, immediate)
    st64(s, 24, wr_id)
    st64(s, 32, status)
    f.bl_count += 1


def _bl_pop(
    mut f: FabricNet, out_comp: Pointer[NetCompletion, MutAnyOrigin]
) -> Bool:
    if f.bl_count == 0:
        return False
    var s = _bl_slot(f, f.bl_head)
    var c = NetCompletion()
    c.kind = ld64(s, 0)
    c.peer = ld64(s, 8)
    c.imm = ldu32(s, 16)
    c.wr_id = ld64(s, 24)
    c.status = ld64(s, 32)
    out_comp[] = c^
    f.bl_head = (f.bl_head + 1) % BACKLOG_CAP
    f.bl_count -= 1
    return True


# ===-------------------------------------------------------------------=== #
# Bring-up
# ===-------------------------------------------------------------------=== #


def _build_hints(prov: String) -> P8:
    """A `struct fi_info` filled in as hints, with its five attribute
    sub-structs allocated and linked.

    `fi_allocinfo` would do this, but it is a `static inline` wrapper around
    `fi_dupinfo(NULL)` and the result would then have to be freed with
    `fi_freeinfo`; building it here keeps the ownership trivial (these
    buffers are never freed, a few hundred bytes per communicator) and keeps
    every field this file sets visible in one place.
    """
    var hints = alloc_bytes(SZ_FI_INFO)
    var tx = alloc_bytes(SZ_FI_TX_ATTR)
    var rx = alloc_bytes(SZ_FI_RX_ATTR)
    var ep = alloc_bytes(SZ_FI_EP_ATTR)
    var dom = alloc_bytes(SZ_FI_DOMAIN_ATTR)
    var fabattr = alloc_bytes(SZ_FI_FABRIC_ATTR)
    st64(hints, INFO_TX_ATTR, Int(tx))
    st64(hints, INFO_RX_ATTR, Int(rx))
    st64(hints, INFO_EP_ATTR, Int(ep))
    st64(hints, INFO_DOMAIN_ATTR, Int(dom))
    st64(hints, INFO_FABRIC_ATTR, Int(fabattr))
    stu64(hints, INFO_CAPS, WANT_CAPS)
    stu64(hints, INFO_MODE, 0)
    st32(ep, EPA_TYPE, FI_EP_RDM)
    st32(dom, DA_MR_MODE, WANT_MR_MODE)
    st32(dom, DA_THREADING, FI_THREAD_SAFE)
    if prov.byte_length() > 0:
        st64(fabattr, FA_PROV_NAME, Int(c_string(String(prov))))
    return hints


def _sub_string(info: Int, attr_off: Int, name_off: Int) -> String:
    """A `char *` inside one of `fi_info`'s attribute sub-structs. Two
    dereferences: `fi_info` holds a POINTER to the sub-struct, which holds
    the pointer to the string."""
    if info == 0:
        return String("")
    var attr = ld64(P8(unsafe_from_address=info), attr_off)
    if attr == 0:
        return String("")
    return read_c_string(ld64(P8(unsafe_from_address=attr), name_off), 128)


def _info_domain_name(info: Int) -> String:
    return _sub_string(info, INFO_DOMAIN_ATTR, DA_NAME)


def _info_prov_name(info: Int) -> String:
    return _sub_string(info, INFO_FABRIC_ATTR, FA_PROV_NAME)


def fabric_available() -> Bool:
    """True if libfabric opens and offers at least one RMA-capable RDM
    provider. Used by the backend auto-selection in `internode.mojo`; the
    resources it opens are released before it returns.
    """
    try:
        var fab = Fab()
        var out = alloc_bytes(8)
        var api = _api_version(fab)
        var rc = fab.getinfo(api, 0, _build_hints(String("")), out)
        var info = ld64(out, 0)
        if rc != 0 or info == 0:
            return False
        fab.freeinfo(info)
        return True
    except:
        return False


def _api_version(fab: Fab) raises -> UInt32:
    """The API version to ask `fi_getinfo` for: the library's own, capped at
    the 2.2 headers every offset in this file was verified against.

    Asking for a version NEWER than the library has fails outright
    (-FI_ENOSYS), and asking for one older changes documented semantics
    (mr_mode is an enum rather than a bitmask below 1.5), so neither a
    hardcoded constant nor `FI_VERSION(1,5)` is right.
    """
    var v = fab.version()
    if v > FI_VERSION_2_2:
        v = FI_VERSION_2_2
    if v < FI_VERSION_1_5:
        raise Error(
            "mojoccl: libfabric is too old (reports version "
            + String(v >> 16)
            + "."
            + String(v & 0xFFFF)
            + "); 1.5 or newer is needed for the mr_mode bitmask"
        )
    return v


def _check(f: FabricNet, rc: Int, what: String) raises:
    if rc == 0:
        return
    var msg: String
    try:
        msg = f.fab.strerror(-rc if rc < 0 else rc)
    except:
        # No text for this code: the numeric rc below is the message.
        msg = String("")
    var hint = String("")
    if rc == -FI_ENOMEM:
        # This is memory pressure on the NODE, essentially always, and the
        # kernel says so if you ask it. Traced on Adastra (2x4 MI300A,
        # Slingshot/cxi) from `fi_enable failed, rc=-12` all the way down:
        #
        #   cxil_map: write error                       (libcxi, to /dev/cxi)
        #   cxip_ep_ctrl_init: Failed to allocate TX EQ resources, ret: -12
        #   python: page allocation failure: order:7,
        #           mode:GFP_KERNEL|__GFP_COMP|__GFP_ZERO
        #     cass_nta_alloc / cass_nta_init / cass_ac_alloc [cxi_ss1]
        #     cxi_map / cxi_user_atu_map / ucxi_write      [cxi_user]
        #
        # The driver needs 512 KiB (order 7) of PHYSICALLY CONTIGUOUS kernel
        # memory for the address context's translation table, and the
        # kernel's own Mem-Info at the failure showed why it could not have
        # it: of the four NUMA nodes, none had a single free block at order 7
        # -- "0*64kB 0*128kB ..." on node 0, whose free total was 355 MB
        # against a watermark min of 353 MB. On an APU the GPU's memory IS
        # system memory, so four ranks of MAX and torch had taken essentially
        # all of it.
        #
        # So `free -g` DURING the run is the check, not before it: the same
        # node reads 480 of 501 GB free when idle. Earlier work here recorded
        # this as an "intermittent cxi condition" on those idle readings and
        # ruled out NIC objects (`cxi_service list` showed 0 of 2047 EQs in
        # use), RLIMIT_MEMLOCK and the hugetlb knobs -- all correctly, none
        # of them was it -- and also measured that it is NOT the node's four
        # ranks racing (staggering them 2 s apart left 5 runs in 8 failing,
        # the same as unstaggered) and that it follows the NODE (four
        # consecutive failures on one node while its partner passed every
        # time, and both fresh nodes passing).
        #
        # `fab_setup` retries for a while because the pressure does ease as
        # ranks free their staging buffers, and some ranks do get through on
        # a later try. It is a mitigation, not a fix: the fix is to leave the
        # node some memory.
        hint = String(
            "; this is almost always node memory pressure rather than a NIC"
            " or provider fault -- on an APU the GPU's memory IS system"
            " memory, so check free memory DURING the run (an idle reading"
            " proves nothing) and look in dmesg for `page allocation"
            " failure: order:7 ... cass_nta_alloc [cxi_ss1]`, which is the"
            " cxi driver failing to find 512 KiB of contiguous kernel memory"
            " for an address context. /proc/buddyinfo and the kernel's"
            " Mem-Info dump show the per-NUMA high-order counts that decide"
            " it. Setup retries for 30 seconds, waiting for"
            " the pressure to ease"
        )
    raise Error(
        "mojoccl: "
        + what
        + " failed, rc="
        + String(rc)
        + " ("
        + msg
        + ")"
        + hint
    )


def _reg_mr(
    mut f: FabricNet,
    addr: Int,
    nbytes: Int,
    iface: Int,
    out_key: P8,
    ep: Int = 0,
    access: UInt64 = FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE,
) raises -> Int:
    """One `fi_mr_regattr`, bound to the endpoint (`ep`, default the data
    endpoint) and enabled if the provider asked for FI_MR_ENDPOINT. Returns
    the fid_mr, or a negative libfabric error from the registration. A
    failed bind or enable closes the MR before raising, so a caller that
    catches the error has nothing to clean up."""
    var attr = alloc_bytes(SZ_FI_MR_ATTR)
    var iov = alloc_bytes(SZ_IOVEC)
    st64(iov, IOV_BASE, addr)
    st64(iov, IOV_LEN, nbytes)
    st64(attr, MRA_MR_IOV, Int(iov))
    st64(attr, MRA_IOV_COUNT, 1)
    stu64(attr, MRA_ACCESS, access)
    stu64(attr, MRA_OFFSET, 0)
    stu64(attr, MRA_REQUESTED_KEY, 0)
    st32(attr, MRA_IFACE, Int32(iface))
    var out = alloc_bytes(8)
    var rc = fi_mr_regattr(f.domain, attr, 0, out)
    var mr = ld64(out, 0)
    free_bytes(out)
    if rc == 0 and f.need_endpoint_mr:
        # FI_MR_ENDPOINT: the key is only valid once the MR is attached to
        # the endpoint that will serve it, and reading it before
        # `fi_mr_enable` gives FI_KEY_NOTAVAIL.
        try:
            _check(
                f, Int(fi_bind(mr, ep if ep != 0 else f.ep, 0)), "fi_mr_bind"
            )
            _check(f, Int(fi_enable(mr)), "fi_mr_enable")
        except e:
            _ = fi_close(mr)
            free_bytes(iov)
            free_bytes(attr)
            raise e
    # Scratch of this call only: the provider copies the attributes, and the
    # bind and enable above ran with them still live.
    free_bytes(iov)
    free_bytes(attr)
    if rc != 0:
        stu64(out_key, 0, 0)
        return Int(rc)  # negative: a libfabric error, not a fid_mr
    stu64(out_key, 0, fi_mr_key(mr))
    return mr


def _flush_ep_open(mut st: FabricNet) raises:
    """Bring up the flush endpoint and, under FI_MR_ENDPOINT, its landing
    pad's MR. On an error the caller runs `_flush_ep_close`, which undoes
    whatever part of this ran, and the flush stays on the data endpoint.

    On the data endpoint the flush read sits in the transmit queue behind
    whatever was posted since -- the next exchange's multi-megabyte write --
    so exchange e retires only once e+1's payload has gone out. The verbs
    path keeps its flush on a separate QP for the same reason (NCCL's
    gpuFlush QP). An optimization, never a requirement.

    Same domain (so the same NIC and PCIe function, which is what makes its
    read a flush of that NIC's earlier writes), same CQ and AV as the data
    endpoint. It is opened from a copy of the data endpoint's `fi_info`
    with the capabilities cut to `FLUSH_EP_CAPS` and none on the receive
    side, and binds the CQ for transmit completions only: the endpoint
    should cost the provider no receive resources at all.
    """
    var fi = st.fab.dupinfo(st.info)
    if fi == 0:
        raise Error("mojoccl: fi_dupinfo of the endpoint info returned NULL")
    st.flush_info = fi
    var fp = P8(unsafe_from_address=fi)
    stu64(fp, INFO_CAPS, FLUSH_EP_CAPS)
    var tx = ld64(fp, INFO_TX_ATTR)
    if tx != 0:
        stu64(P8(unsafe_from_address=tx), TXA_CAPS, FLUSH_EP_CAPS)
    var rx = ld64(fp, INFO_RX_ATTR)
    if rx != 0:
        stu64(P8(unsafe_from_address=rx), RXA_CAPS, 0)
    var o = alloc_bytes(8)
    var rc = Int(fi_endpoint(st.domain, fi, o))
    var ep = ld64(o, 0)
    free_bytes(o)
    _check(st, rc, "fi_endpoint(flush)")
    st.flush_ep = ep
    _check(
        st,
        Int(fi_bind(st.flush_ep, st.cq, FI_TRANSMIT)),
        "fi_ep_bind(flush cq)",
    )
    _check(st, Int(fi_bind(st.flush_ep, st.av, 0)), "fi_ep_bind(flush av)")
    _check(st, Int(fi_enable(st.flush_ep)), "fi_enable(flush)")
    if st.need_endpoint_mr:
        # The landing pad, registered again for the flush endpoint. Local
        # access only: it is the destination of this endpoint's own read.
        var keybuf = alloc_bytes(8)
        try:
            var fmr = _reg_mr(
                st,
                st.host,
                FLUSH_PAD_BYTES,
                FI_HMEM_SYSTEM,
                keybuf,
                st.flush_ep,
                FI_READ,
            )
            if fmr <= 0:
                _check(st, fmr, "fi_mr_regattr of the flush landing pad")
            st.flush_mr = fmr
            st.flush_desc = fi_mr_desc(st.flush_mr)
        finally:
            # Never sent: the pad is only this endpoint's own read target.
            free_bytes(keybuf)


def _flush_ep_close(mut f: FabricNet):
    """Undo `_flush_ep_open`, whatever part of it ran, and point the flush
    back at the data endpoint's landing pad. Zero-guarded: also part of
    `fab_teardown`."""
    _ = fi_close(f.flush_mr)
    f.flush_mr = 0
    _ = fi_close(f.flush_ep)
    f.flush_ep = 0
    if f.flush_info != 0:
        try:
            f.fab.freeinfo(f.flush_info)
        except e:
            # Best-effort; the pointer is dropped either way.
            print("mojoccl: fi_freeinfo failed (ignored):", e)
        f.flush_info = 0
    f.flush_desc = f.host_desc


def _fab_setup_once(
    gpu_bdf: String,
    local_rank: Int,
    my_node: Int,
    nnodes: Int,
    region: Int,
    region_bytes: Int,
) raises -> FabricNet:
    """One attempt at `fab_setup`; see it for the retry this sits inside.

    Open one NIC, register the region, bring the endpoint up.

    Unlike the verbs path there is no connection to establish: an FI_EP_RDM
    endpoint is connectionless, so everything except "where are the peers"
    happens here and `fab_add_peer` only fills the address vector.
    """
    var fab = Fab()
    var api = _api_version(fab)
    var out = alloc_bytes(8)
    var prov = getenv(MOJOCCL_FABRIC_PROVIDER, "cxi")
    var rc = fab.getinfo(api, 0, _build_hints(String(prov)), out)
    if rc != 0 or ld64(out, 0) == 0:
        # No cxi (or whatever was asked for): let libfabric pick any
        # provider that satisfies the same hints.
        rc = fab.getinfo(api, 0, _build_hints(String("")), out)
        if rc != 0 or ld64(out, 0) == 0:
            raise Error(
                "mojoccl: fi_getinfo found no libfabric provider offering"
                " FI_RMA|FI_MSG|FI_HMEM on an FI_EP_RDM endpoint (rc="
                + String(rc)
                + "); a multi-node communicator needs one"
            )

    var st = FabricNet(fab^, region, my_node)
    st.info_list = ld64(out, 0)
    st.api_version = api
    try:
        # ---- choose the NIC -------------------------------------------
        var infos = List[Int]()
        var names = List[String]()
        var paths = List[String]()
        var p = st.info_list
        while p != 0:
            infos.append(p)
            var dn = _info_domain_name(p)
            names.append(String(dn))
            paths.append("/sys/class/cxi/" + dn + "/device")
            p = ld64(P8(unsafe_from_address=p), INFO_NEXT)
        var want = getenv(MOJOCCL_FABRIC_DOMAIN, "")
        var pick = -1
        if want.byte_length() > 0:
            for i in range(len(names)):
                if names[i] == want:
                    pick = i
            if pick < 0:
                raise Error(
                    "mojoccl: MOJOCCL_FABRIC_DOMAIN="
                    + want
                    + " matches none of the "
                    + String(len(names))
                    + " libfabric domains this provider offers"
                )
        else:
            pick = pci_pick(paths, gpu_bdf, local_rank)
        st.info = infos[pick]
        st.domain_name = String(names[pick])
        st.prov_name = _info_prov_name(st.info)

        # ---- everything this transport assumes about the provider ------
        var ip = P8(unsafe_from_address=st.info)
        var mode = ldu64(ip, INFO_MODE)
        if (mode & (FI_CONTEXT | FI_CONTEXT2)) != 0:
            raise Error(
                "mojoccl: provider "
                + st.prov_name
                + " requires FI_CONTEXT/FI_CONTEXT2, which means every"
                " operation's context must be provider-owned memory; this"
                " transport packs (kind, peer, exchange) into the context"
                " word instead"
            )
        var dom_attr = ld64(ip, INFO_DOMAIN_ATTR)
        var da = P8(unsafe_from_address=dom_attr)
        var cq_data_size = ld64(da, DA_CQ_DATA_SIZE)
        if cq_data_size < 8:
            raise Error(
                "mojoccl: provider "
                + st.prov_name
                + " carries only "
                + String(cq_data_size)
                + " bytes of remote CQ data; this transport packs the sender's"
                " node index above the 32-bit immediate and needs 8"
            )
        if (ldu64(ip, INFO_CAPS) & FI_FENCE) == 0:
            raise Error(
                "mojoccl: provider "
                + st.prov_name
                + " will not honour FI_FENCE; this transport announces a"
                " shard with a separate message and has nothing else to keep"
                " that message behind the payload it announces (and the"
                " engine's per-peer send accounting relies on the same fence"
                " to keep completions in exchange order)"
            )
        var mr_mode = Int32(ld32(da, DA_MR_MODE))
        st.virt_addr = (mr_mode & FI_MR_VIRT_ADDR) != 0
        st.need_endpoint_mr = (mr_mode & FI_MR_ENDPOINT) != 0

        # ---- fabric / domain / cq / av / endpoint ----------------------
        var o = alloc_bytes(8)
        _check(
            st, Int(st.fab.fabric(ld64(ip, INFO_FABRIC_ATTR), o)), "fi_fabric"
        )
        st.fabric = ld64(o, 0)
        _check(st, Int(fi_domain(st.fabric, st.info, o)), "fi_domain")
        st.domain = ld64(o, 0)

        var cqa = alloc_bytes(SZ_FI_CQ_ATTR)
        st64(cqa, CQA_SIZE, CQ_DEPTH)
        st32(cqa, CQA_FORMAT, FI_CQ_FORMAT_DATA)
        st32(cqa, CQA_WAIT_OBJ, FI_WAIT_NONE)
        _check(st, Int(fi_cq_open(st.domain, cqa, o)), "fi_cq_open")
        st.cq = ld64(o, 0)

        var ava = alloc_bytes(SZ_FI_AV_ATTR)
        st32(ava, AVA_TYPE, FI_AV_TABLE)
        st64(ava, AVA_COUNT, nnodes + 1)
        _check(st, Int(fi_av_open(st.domain, ava, o)), "fi_av_open")
        st.av = ld64(o, 0)

        _check(st, Int(fi_endpoint(st.domain, st.info, o)), "fi_endpoint")
        st.ep = ld64(o, 0)
        # One completion queue for both directions, like the verbs path's
        # single CQ: the engine classifies by completion flags, not by queue.
        _check(
            st, Int(fi_bind(st.ep, st.cq, FI_SEND | FI_RECV)), "fi_ep_bind(cq)"
        )
        _check(st, Int(fi_bind(st.ep, st.av, 0)), "fi_ep_bind(av)")
        _check(st, Int(fi_enable(st.ep)), "fi_enable")

        # ---- this endpoint's address -----------------------------------
        st64(P8(unsafe_from_address=st.lenbuf), 0, EP_NAME_MAX)
        _check(
            st,
            Int(
                fi_getname(
                    st.ep,
                    P8(unsafe_from_address=st.my_name),
                    P8(unsafe_from_address=st.lenbuf),
                )
            ),
            "fi_getname",
        )
        st.addrlen = ld64(P8(unsafe_from_address=st.lenbuf), 0)
        if st.addrlen > EP_NAME_MAX:
            raise Error(
                "mojoccl: this provider's endpoint address is "
                + String(st.addrlen)
                + " bytes and the bootstrap blob carries "
                + String(EP_NAME_MAX)
            )

        # ---- host scratch: the flush landing pad and the recv buffers --
        var npeers = nnodes - 1
        var rx_size = ld64(
            P8(unsafe_from_address=ld64(ip, INFO_RX_ATTR)), RXA_SIZE
        )
        st.recv_depth = max(64, npeers * 32)
        if rx_size > 0 and st.recv_depth > rx_size:
            st.recv_depth = Int(rx_size)
        st.host_bytes = FLUSH_PAD_BYTES + st.recv_depth * RECV_BUF_BYTES
        st.host = Int(alloc_bytes(st.host_bytes))
        var keybuf = alloc_bytes(8)
        var hmr = _reg_mr(st, st.host, st.host_bytes, FI_HMEM_SYSTEM, keybuf)
        if hmr <= 0:
            _check(st, hmr, "fi_mr_regattr of the host scratch")
        st.host_mr = hmr
        st.host_desc = fi_mr_desc(st.host_mr)
        st.flush_desc = st.host_desc
        try:
            _flush_ep_open(st)
        except e:
            # Never fatal: the data endpoint flushes correctly, only later.
            # A second endpoint is a second cxi address context, and the
            # node memory pressure `_check` describes can deny it after the
            # first one succeeded.
            _flush_ep_close(st)
            print(
                (
                    "mojoccl: flush endpoint unavailable, flushing on the"
                    " data endpoint --"
                ),
                e,
            )

        # ---- the communicator's region ---------------------------------
        # Ask the provider which accelerator interface can register this
        # pointer, then try host memory. This preserves the measured auto
        # order and never guesses that an accelerator pointer is host memory.
        var ifaces = List[Int]()
        ifaces.append(FI_HMEM_ROCR)
        ifaces.append(FI_HMEM_CUDA)
        ifaces.append(FI_HMEM_SYSTEM)
        var mr = 0
        var last = 0
        for iface in ifaces:
            var r = _reg_mr(st, region, region_bytes, iface, keybuf)
            if r > 0:
                mr = r
                st.hmem_iface = iface
                break
            last = r
        if mr <= 0:
            _check(
                st,
                last,
                "fi_mr_regattr of the "
                + String(region_bytes // (1024 * 1024))
                + " MiB region on "
                + st.domain_name
                + " (is this libfabric built with FI_HMEM support for this"
                " accelerator?)",
            )
        st.mr = mr
        st.mr_key = ldu64(keybuf, 0)
        st.mr_desc = fi_mr_desc(st.mr)

        # ---- this rank's own address, for the flush read ---------------
        var ab = alloc_bytes(8)
        var n = fi_av_insert(st.av, P8(unsafe_from_address=st.my_name), 1, ab)
        if Int(n) != 1:
            raise Error(
                "mojoccl: fi_av_insert of this rank's own address returned "
                + String(n)
            )
        st.self_addr = ldu64(ab, 0)

        for _ in range(nnodes):
            st.peer_of_node.append(-1)
    except e:
        fab_teardown(st)
        raise e
    return st^


def _fab_setup_retry_s() -> Float64:
    """Bound endpoint retries under transient provider memory pressure."""
    return FAB_SETUP_RETRY_S


def _is_enomem(e: Error) -> Bool:
    """Whether an error out of `_fab_setup_once` is libfabric's -FI_ENOMEM.

    Matched on the message rather than on a returned code because every
    failure in the bring-up comes out of `_check`, which has already turned
    the code into a String. `_check` writes exactly one "rc=<n>" per message
    and -12 is FI_ENOMEM, so the token is unambiguous; it is spelled here
    once so that changing `_check`'s wording breaks in one place.
    """
    return String(e).find("rc=" + String(-FI_ENOMEM)) >= 0


def fab_setup(
    gpu_bdf: String,
    local_rank: Int,
    my_node: Int,
    nnodes: Int,
    region: Int,
    region_bytes: Int,
) raises -> FabricNet:
    """`_fab_setup_once`, retried while the provider says -FI_ENOMEM.

    What that error means is in `_check`: the node has run out of high-order
    contiguous kernel memory and the cxi driver cannot build an address
    context. Measured on Adastra with four MI300A ranks per node, it hit one
    node in 5 runs of 8 while its partner passed every time.

    Retrying is worth doing because the pressure eases -- ranks free staging
    buffers as they go, and in two of four failing runs some of the ranks
    that failed the first attempt got through a later one. It is not a fast
    flap, though: eight tries over 2.5 s all failed on all four ranks of a
    node, so the budget is a wall-clock one (`FAB_SETUP_RETRY_S`,
    30 s) rather than a try count, and it sits inside the bootstrap's own
    120 s deadline so a node that spends it all still meets its peers.

    `_fab_setup_once` unwinds its own resources on the way out
    (`fab_teardown` in its except), so each try starts from nothing --
    including a fresh `fi_getinfo`, since the teardown frees the info list.

    UNVERIFIED: how often 30 s is enough. The retry was written after the
    only node that reproduced the failure went out of allocation, so what is
    measured is the 2.5 s version (partial recoveries, above); the 30 s
    budget is an extrapolation from it.
    """
    var budget_ns = Int(_fab_setup_retry_s() * 1.0e9)
    var deadline = perf_counter_ns() + budget_ns
    var backoff_us = FAB_SETUP_BACKOFF_US
    var attempt = 0
    while True:
        attempt += 1
        try:
            return _fab_setup_once(
                gpu_bdf, local_rank, my_node, nnodes, region, region_bytes
            )
        except e:
            if perf_counter_ns() >= deadline or not _is_enomem(e):
                raise e
            print(
                (
                    "mojoccl: libfabric endpoint bring-up failed with"
                    " -FI_ENOMEM (attempt"
                ),
                attempt,
                "), retrying for up to",
                budget_ns // 1_000_000_000,
                "s more --",
                e,
            )
            sleep(Float64(backoff_us) / 1.0e6)
            backoff_us = min(backoff_us * 2, FAB_SETUP_BACKOFF_MAX_US)
    # Unreachable: the loop only leaves by returning or by re-raising.


# ===-------------------------------------------------------------------=== #
# The bootstrap blob
# ===-------------------------------------------------------------------=== #
#
# Laid out inside the same fixed-size per-rank blob the verbs path uses (see
# `IB_BLOB_BYTES` in internode.mojo); only one of the two transports is ever
# active in a job, so the two layouts do not have to coexist.
#
#   +0   u64 region base VA   (meaningful only under FI_MR_VIRT_ADDR)
#   +8   u64 memory-region key (fi_mr_key, provider-chosen on cxi)
#   +16  u32 endpoint address length
#   +20  u32 flags: bit 0 = this rank's provider uses virtual addressing
#   +24  u8  endpoint address[EP_NAME_MAX]   (fi_getname, 8 bytes on cxi)

comptime BLOB_BASE = 0
comptime BLOB_KEY = 8
comptime BLOB_ADDRLEN = 16
comptime BLOB_FLAGS = 20
comptime BLOB_NAME = 24
comptime BLOB_FLAG_VIRT_ADDR: UInt32 = 1


def fab_local_info(f: FabricNet, blob: P8):
    stu64(blob, BLOB_BASE, UInt64(f.region))
    stu64(blob, BLOB_KEY, f.mr_key)
    stu32(blob, BLOB_ADDRLEN, UInt32(f.addrlen))
    stu32(blob, BLOB_FLAGS, BLOB_FLAG_VIRT_ADDR if f.virt_addr else 0)
    var src = P8(unsafe_from_address=f.my_name)
    for i in range(f.addrlen):
        blob[unsafe_offset=BLOB_NAME + i] = src[unsafe_offset=i]


def fab_blob_base(f: FabricNet, blob: P8) -> Int:
    """The base a remote address is measured from. Zero unless the provider
    uses virtual addressing: with FI_MR_ALLOCATED and no FI_MR_VIRT_ADDR --
    what cxi reports -- an RMA target address is an OFFSET into the peer's
    registered region, so the engine's region-relative offsets go on the
    wire unchanged."""
    return Int(ldu64(blob, BLOB_BASE)) if f.virt_addr else 0


def fab_blob_key(blob: P8) -> UInt64:
    return ldu64(blob, BLOB_KEY)


def fab_add_peer(mut f: FabricNet, peer_index: Int, node: Int, blob: P8) raises:
    """Insert one peer's endpoint address into the address vector."""
    var alen = Int(ldu32(blob, BLOB_ADDRLEN))
    if alen == 0 or alen > EP_NAME_MAX:
        raise Error(
            "mojoccl: peer on node "
            + String(node)
            + " published a "
            + String(alen)
            + "-byte libfabric endpoint address; this rank expects 1.."
            + String(EP_NAME_MAX)
        )
    if ((ldu32(blob, BLOB_FLAGS) & BLOB_FLAG_VIRT_ADDR) != 0) != f.virt_addr:
        raise Error(
            "mojoccl: peer on node "
            + String(node)
            + " and this rank disagree about FI_MR_VIRT_ADDR; the two ranks"
            " are on different libfabric providers"
        )
    var name = alloc_bytes(EP_NAME_MAX)
    for i in range(alen):
        name[unsafe_offset=i] = blob[unsafe_offset=BLOB_NAME + i]
    var ab = alloc_bytes(8)
    var n = fi_av_insert(f.av, name, 1, ab)
    if Int(n) != 1:
        raise Error(
            "mojoccl: fi_av_insert for node "
            + String(node)
            + " returned "
            + String(n)
        )
    while len(f.addrs) <= peer_index:
        f.addrs.append(FI_ADDR_UNSPEC)
    f.addrs[peer_index] = ldu64(ab, 0)
    f.peer_of_node[node] = peer_index


def fab_post_recvs(mut f: FabricNet) raises:
    """Pre-post the receive buffers the peers' notifications land on.

    Unlike the verbs path there is no per-peer queue: one endpoint receives
    every peer's messages, so the depth has to cover what ALL peers can
    produce while this rank is elsewhere, not what one peer can. The buffers
    themselves are never read -- the 64-bit remote CQ data is the whole
    message, and every notification is zero-length -- but a receive has to
    be posted for one to be delivered."""
    for i in range(f.recv_depth):
        var rc = fi_recv(
            f.ep,
            f.host + FLUSH_PAD_BYTES + i * RECV_BUF_BYTES,
            RECV_BUF_BYTES,
            f.host_desc,
            FI_ADDR_UNSPEC,
            _pack_ctx(NC_RECV, 0, i),
        )
        if rc != 0:
            _check(f, rc, "fi_recv while pre-posting")


# ===-------------------------------------------------------------------=== #
# The data path
# ===-------------------------------------------------------------------=== #


def _fab_drain(mut f: FabricNet):
    """One non-blocking `fi_cq_read` into the completion ring.

    Also the only thing that makes progress on a FI_PROGRESS_MANUAL
    provider, which is why the -FI_EAGAIN retry loops below call it: the
    command slot a retry is waiting for is freed by processing the event
    queue, and reading the completion queue is what processes it.
    """
    var buf = P8(unsafe_from_address=f.cqe)
    var n = fi_cq_read(f.cq, buf, 16)
    if n == -FI_EAGAIN:
        return
    if n == -FI_EAVAIL:
        var e = P8(unsafe_from_address=f.cqerr)
        for i in range(SZ_CQ_ERR):
            e[unsafe_offset=i] = 0
        _ = fi_cq_readerr(f.cq, e)
        # Same shape as the verbs path's status encoding, so `ib_error`'s
        # number stays readable: 1000 + status*1000 + vendor error.
        _bl_push(
            f,
            NC_OTHER,
            -1,
            0,
            0,
            1000 + ld32(e, CQERR_ERR) * 1000 + ld32(e, CQERR_PROV_ERRNO),
        )
        return
    if n < 0:
        _bl_push(f, NC_OTHER, -1, 0, 0, 2000 - n)
        return
    for i in range(n):
        var c = P8(unsafe_from_address=f.cqe + i * SZ_CQ_ENTRY)
        var flags = ldu64(c, CQE_FLAGS)
        var ctx = ld64(c, CQE_OP_CONTEXT)
        if (flags & FI_RECV) != 0:
            var data = ldu64(c, CQE_DATA)
            var node = Int(data >> CQ_DATA_NODE_SHIFT)
            var peer = -1
            if node >= 0 and node < len(f.peer_of_node):
                peer = f.peer_of_node[node]
            _bl_push(f, NC_RECV, peer, UInt32(data & 0xFFFF_FFFF), 0, 0)
            # Repost the buffer this completion consumed. A receive lost here
            # is a peer's notification that never arrives, i.e. a hang, so it
            # is latched rather than ignored.
            var rc = fi_recv(
                f.ep,
                f.host + FLUSH_PAD_BYTES + _ctx_seq(ctx) * RECV_BUF_BYTES,
                RECV_BUF_BYTES,
                f.host_desc,
                FI_ADDR_UNSPEC,
                ctx,
            )
            if rc != 0:
                _bl_push(f, NC_OTHER, -1, 0, 0, 3000 - rc)
        elif (flags & FI_READ) != 0:
            _bl_push(f, NC_FLUSH, -1, 0, 0, 0)
        elif (flags & FI_WRITE) != 0:
            _bl_push(f, NC_SEND, _ctx_peer(ctx), 0, _ctx_seq(ctx), 0)
        else:
            # A notification or credit send completing: nothing to account
            # for, but it is progress.
            _bl_push(f, NC_OTHER, -1, 0, 0, 0)


def _retry(mut f: FabricNet, rc0: Int) -> Bool:
    """True if a post that returned `rc0` should be tried again."""
    if rc0 != -FI_EAGAIN:
        return False
    _fab_drain(f)
    return True


def fab_post_write(
    mut f: FabricNet,
    peer: Int,
    local_addr: Int,
    nbytes: Int,
    remote_addr: Int,
    remote_key: UInt64,
    seq: Int,
) -> Int:
    """The payload half of one write-with-immediate: a plain RMA write.

    `fi_writemsg` rather than `fi_write` only so FI_COMPLETION can be named
    explicitly; the completion is what tells the engine the NIC has finished
    reading the source buffer.
    """
    var iov = P8(unsafe_from_address=f.iov)
    st64(iov, IOV_BASE, local_addr)
    st64(iov, IOV_LEN, nbytes)
    var riov = P8(unsafe_from_address=f.rma_iov)
    stu64(riov, RMAIOV_ADDR, UInt64(remote_addr))
    st64(riov, RMAIOV_LEN, nbytes)
    stu64(riov, RMAIOV_KEY, remote_key)
    st64(P8(unsafe_from_address=f.desc_slot), 0, f.mr_desc)
    var m = P8(unsafe_from_address=f.msg_rma)
    st64(m, MSGRMA_MSG_IOV, f.iov)
    st64(m, MSGRMA_DESC, f.desc_slot)
    st64(m, MSGRMA_IOV_COUNT, 1)
    stu64(m, MSGRMA_ADDR, f.addrs[peer])
    st64(m, MSGRMA_RMA_IOV, f.rma_iov)
    st64(m, MSGRMA_RMA_IOV_COUNT, 1)
    st64(m, MSGRMA_CONTEXT, _pack_ctx(NC_SEND, peer, seq))
    for _ in range(EAGAIN_SPINS):
        # FI_DELIVERY_COMPLETE, not cxi's default transmit-complete: the
        # notification that follows is FI_FENCEd behind THIS operation's
        # completion, and a transmit-complete write has only left the
        # initiator; cxi sets the target-side flush bit only under
        # delivery-complete (prov/cxi/src/cxip_rma.c). Without it the peer can
        # see the immediate before the payload is in its memory.
        var rc = fi_writemsg(f.ep, m, FI_COMPLETION | FI_DELIVERY_COMPLETE)
        if rc == 0:
            return 0
        if not _retry(f, rc):
            return rc
    return -FI_EAGAIN


def fab_post_imm(
    mut f: FabricNet, peer: Int, immediate: UInt32, fence: Bool, seq: Int
) -> Int:
    """The immediate half: a zero-length message whose 64-bit remote CQ data
    is `(my node index << 32) | imm`.

    `fence` must be set on the first notification that follows a payload
    write, and the cxi command-queue fence then covers the rest of the same
    exchange by queue position (see this file's header). A credit needs no
    fence: it announces nothing that was written.
    """
    var m = P8(unsafe_from_address=f.msg)
    st64(m, MSG_MSG_IOV, 0)
    st64(m, MSG_DESC, 0)
    st64(m, MSG_IOV_COUNT, 0)
    stu64(m, MSG_ADDR, f.addrs[peer])
    st64(m, MSG_CONTEXT, _pack_ctx(NC_OTHER, peer, seq))
    stu64(
        m,
        MSG_DATA,
        (UInt64(f.my_node) << CQ_DATA_NODE_SHIFT) | UInt64(immediate),
    )
    var flags = FI_REMOTE_CQ_DATA | FI_COMPLETION
    if fence:
        flags |= FI_FENCE
    for _ in range(EAGAIN_SPINS):
        var rc = fi_sendmsg(f.ep, m, flags)
        if rc == 0:
            return 0
        if not _retry(f, rc):
            return rc
    return -FI_EAGAIN


def fab_post_flush(
    mut f: FabricNet, remote_off: Int, nbytes: Int, seq: Int
) -> Int:
    """A short read of this rank's OWN region, from its own address -- the
    same role as the verbs self-connected QP's RDMA_READ.

    Whether it is still needed here is not obvious and was not assumed. The
    verbs argument is that the completion lands in host memory and the
    payload in the GPU's BAR, two PCIe destinations with no ordering between
    them. Under this transport the arrival is announced by a FENCED message,
    and fi_endpoint(3) says a fenced operation is deferred until previous
    operations to that peer have COMPLETED -- which should already place the
    payload in memory. No cxi provider documentation was found that promises
    it, and on an MI300A "device memory" is host-attached HBM anyway, so the
    read stays: it costs about a microsecond and it is the difference
    between an argument and a guarantee. The flush is always enabled.

    It goes on its own endpoint when one came up (`_flush_ep_open`;
    measured in agents_docs/distributed.md).
    """
    for _ in range(EAGAIN_SPINS):
        var rc = fi_read(
            f.flush_ep if f.flush_ep != 0 else f.ep,
            f.host,
            nbytes,
            f.flush_desc,
            f.self_addr,
            UInt64(remote_off if not f.virt_addr else f.region + remote_off),
            f.mr_key,
            _pack_ctx(NC_FLUSH, 0, seq),
        )
        if rc == 0:
            return 0
        if not _retry(f, rc):
            return rc
    return -FI_EAGAIN


def fab_poll(mut f: FabricNet, comps: Int, max_comps: Int) -> Int:
    """Up to `max_comps` completions into the caller's `NetCompletion`
    array, in the shape the engine understands."""
    _fab_drain(f)
    var n = 0
    while n < max_comps:
        if not _bl_pop(
            f,
            Pointer[NetCompletion, MutAnyOrigin](
                unsafe_from_address=comps + n * size_of[NetCompletion]()
            ),
        ):
            break
        n += 1
    if f.last_error != 0 and n < max_comps:
        var c = NetCompletion()
        c.status = 4000
        Pointer[NetCompletion, MutAnyOrigin](
            unsafe_from_address=comps + n * size_of[NetCompletion]()
        )[] = (c^)
        f.last_error = 0
        n += 1
    return n


def fab_iface_name(f: FabricNet) -> String:
    """The FI_HMEM interface the region actually registered under -- the one
    thing about this transport that cannot be predicted from the
    environment, since registration asks the provider rather
    than guessing (see `_fab_setup_once`)."""
    if f.hmem_iface == FI_HMEM_ROCR:
        return String("rocr")
    if f.hmem_iface == FI_HMEM_CUDA:
        return String("cuda")
    return String("system")


def fab_describe(f: FabricNet) -> String:
    """One line for `MOJOCCL_IB_TRACE=1`: what was negotiated, in the terms
    a libfabric user would grep for."""
    return (
        "hmem="
        + fab_iface_name(f)
        + " key="
        + hex(f.mr_key)
        + (" virt_addr" if f.virt_addr else " offset_addr")
        + " addrlen="
        + String(f.addrlen)
        + " recv_depth="
        + String(f.recv_depth)
        + (" flush_ep=own" if f.flush_ep != 0 else " flush_ep=data")
    )


def fab_teardown(mut f: FabricNet):
    """Close everything, in reverse order of creation and zero-guarded --
    this runs both on a live communicator's teardown and from `fab_setup`'s
    own failure path, where only some of it exists."""
    _flush_ep_close(f)
    _ = fi_close(f.mr)
    f.mr = 0
    _ = fi_close(f.host_mr)
    f.host_mr = 0
    _ = fi_close(f.ep)
    f.ep = 0
    _ = fi_close(f.cq)
    f.cq = 0
    _ = fi_close(f.av)
    f.av = 0
    _ = fi_close(f.domain)
    f.domain = 0
    _ = fi_close(f.fabric)
    f.fabric = 0
    if f.info_list != 0:
        try:
            f.fab.freeinfo(f.info_list)
        except e:
            # Best-effort teardown; the pointer is dropped either way.
            print("mojoccl: fi_freeinfo failed (ignored):", e)
        f.info_list = 0
        f.info = 0
