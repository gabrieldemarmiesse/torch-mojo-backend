# Checks every libfabric struct offset, size and constant that
# `transport/net_ofi.mojo` hard-codes against what the C compiler says the installed
# headers actually contain.
#
#     gcc -O0 -I <libfabric-prefix>/include \
#         -o /tmp/fabric_abi tests/multinode/selftest/fabric_abi.c
#     /tmp/fabric_abi > /tmp/fabric_abi.txt
#     mojo build tests/multinode/selftest/fabric_abi.mojo \
#         -I torch_mojo_backend/mojo -o /tmp/fabric_abi_check
#     /tmp/fabric_abi_check /tmp/fabric_abi.txt      # must print PASS
#
# Needs no libfabric at run time, no NIC and no peers: it compares two lists
# of numbers. What it is for is the thing that cannot be caught any other
# way -- `std.ffi` has no C-struct ABI (MOCO-3692), so a wrong offset in
# transport/net_ofi.mojo does not fail to compile and often does not fail to run; it
# passes a garbage pointer to the NIC.
from std.sys import argv

from tmb.ccl.transport.net_ofi import (
    AVA_COUNT,
    AVA_TYPE,
    CMOPS_GETNAME,
    CQA_FORMAT,
    CQA_SIZE,
    CQA_WAIT_OBJ,
    CQE_DATA,
    CQE_FLAGS,
    CQE_LEN,
    CQE_OP_CONTEXT,
    CQERR_ERR,
    CQERR_PROV_ERRNO,
    CQOPS_READ,
    CQOPS_READERR,
    DA_CQ_DATA_SIZE,
    DA_MR_KEY_SIZE,
    DA_MR_MODE,
    DA_NAME,
    DA_THREADING,
    DOMOPS_AV_OPEN,
    DOMOPS_CQ_OPEN,
    DOMOPS_ENDPOINT,
    EPA_MAX_MSG_SIZE,
    EPA_TYPE,
    FA_API_VERSION,
    FA_NAME,
    FA_PROV_NAME,
    FABOPS_DOMAIN,
    FI_ADDR_UNSPEC,
    FI_AV_TABLE,
    FI_COMPLETION,
    FI_CONTEXT,
    FI_CONTEXT2,
    FI_CQ_FORMAT_DATA,
    FI_EAGAIN,
    FI_EAVAIL,
    FI_ENABLE,
    FI_EP_RDM,
    FI_FENCE,
    FI_HMEM,
    FI_HMEM_CUDA,
    FI_HMEM_ROCR,
    FI_HMEM_SYSTEM,
    FI_LOCAL_COMM,
    FI_MR_ALLOCATED,
    FI_MR_ENDPOINT,
    FI_MR_HMEM,
    FI_MR_LOCAL,
    FI_MR_PROV_KEY,
    FI_MR_VIRT_ADDR,
    FI_MSG,
    FI_READ,
    FI_RECV,
    FI_REMOTE_CQ_DATA,
    FI_REMOTE_COMM,
    FI_REMOTE_READ,
    FI_REMOTE_WRITE,
    FI_RMA,
    FI_SEND,
    FI_VERSION_1_5,
    FI_VERSION_2_2,
    FI_WAIT_NONE,
    FI_WRITE,
    FID_OPS,
    FIDAV_OPS,
    FIDCQ_OPS,
    FIDDOM_MR,
    FIDDOM_OPS,
    FIDEP_CM,
    FIDEP_MSG,
    FIDEP_RMA,
    FIDFAB_OPS,
    FIDMR_KEY,
    FIDMR_MEM_DESC,
    INFO_ADDR_FORMAT,
    INFO_CAPS,
    INFO_DOMAIN_ATTR,
    INFO_EP_ATTR,
    INFO_FABRIC_ATTR,
    INFO_MODE,
    INFO_NEXT,
    INFO_RX_ATTR,
    INFO_SRC_ADDR,
    INFO_SRC_ADDRLEN,
    INFO_TX_ATTR,
    IOV_BASE,
    IOV_LEN,
    MRA_ACCESS,
    MRA_IFACE,
    MRA_IOV_COUNT,
    MRA_MR_IOV,
    MRA_OFFSET,
    MRA_REQUESTED_KEY,
    MROPS_REGATTR,
    MSG_ADDR,
    MSG_CONTEXT,
    MSG_DATA,
    MSG_DESC,
    MSG_IOV_COUNT,
    MSG_MSG_IOV,
    MSGOPS_RECV,
    MSGOPS_SENDMSG,
    MSGRMA_ADDR,
    MSGRMA_CONTEXT,
    MSGRMA_DESC,
    MSGRMA_IOV_COUNT,
    MSGRMA_MSG_IOV,
    MSGRMA_RMA_IOV,
    MSGRMA_RMA_IOV_COUNT,
    OPS_BIND,
    OPS_CLOSE,
    OPS_CONTROL,
    RMAIOV_ADDR,
    RMAIOV_KEY,
    RMAIOV_LEN,
    RMAOPS_READ,
    RMAOPS_WRITEMSG,
    RXA_SIZE,
    SZ_CQ_ENTRY,
    SZ_CQ_ERR,
    SZ_FI_AV_ATTR,
    SZ_FI_CQ_ATTR,
    SZ_FI_DOMAIN_ATTR,
    SZ_FI_EP_ATTR,
    SZ_FI_FABRIC_ATTR,
    SZ_FI_INFO,
    SZ_FI_MR_ATTR,
    SZ_FI_MSG,
    SZ_FI_MSG_RMA,
    SZ_FI_RMA_IOV,
    SZ_FI_RX_ATTR,
    SZ_FI_TX_ATTR,
    SZ_IOVEC,
    TXA_SIZE,
)


struct Pair(Copyable, ImplicitlyCopyable, Movable):
    var name: String
    var value: UInt64

    def __init__(out self, var name: String, value: UInt64):
        self.name = name^
        self.value = value


def _expected() -> List[Pair]:
    """Every number `transport/net_ofi.mojo` believes, paired with the name
    `fabric_abi.c` prints it under."""
    var e = List[Pair]()
    e.append(Pair(String("SZ_fi_info"), SZ_FI_INFO))
    e.append(Pair(String("SZ_fi_tx_attr"), SZ_FI_TX_ATTR))
    e.append(Pair(String("SZ_fi_rx_attr"), SZ_FI_RX_ATTR))
    e.append(Pair(String("SZ_fi_ep_attr"), SZ_FI_EP_ATTR))
    e.append(Pair(String("SZ_fi_domain_attr"), SZ_FI_DOMAIN_ATTR))
    e.append(Pair(String("SZ_fi_fabric_attr"), SZ_FI_FABRIC_ATTR))
    e.append(Pair(String("SZ_fi_cq_attr"), SZ_FI_CQ_ATTR))
    e.append(Pair(String("SZ_fi_av_attr"), SZ_FI_AV_ATTR))
    e.append(Pair(String("SZ_fi_mr_attr"), SZ_FI_MR_ATTR))
    e.append(Pair(String("SZ_fi_msg_rma"), SZ_FI_MSG_RMA))
    e.append(Pair(String("SZ_fi_msg"), SZ_FI_MSG))
    e.append(Pair(String("SZ_fi_rma_iov"), SZ_FI_RMA_IOV))
    e.append(Pair(String("SZ_iovec"), SZ_IOVEC))
    e.append(Pair(String("SZ_fi_cq_data_entry"), SZ_CQ_ENTRY))
    e.append(Pair(String("SZ_fi_cq_err_entry"), SZ_CQ_ERR))
    e.append(Pair(String("INFO_next"), INFO_NEXT))
    e.append(Pair(String("INFO_caps"), INFO_CAPS))
    e.append(Pair(String("INFO_mode"), INFO_MODE))
    e.append(Pair(String("INFO_addr_format"), INFO_ADDR_FORMAT))
    e.append(Pair(String("INFO_src_addrlen"), INFO_SRC_ADDRLEN))
    e.append(Pair(String("INFO_src_addr"), INFO_SRC_ADDR))
    e.append(Pair(String("INFO_tx_attr"), INFO_TX_ATTR))
    e.append(Pair(String("INFO_rx_attr"), INFO_RX_ATTR))
    e.append(Pair(String("INFO_ep_attr"), INFO_EP_ATTR))
    e.append(Pair(String("INFO_domain_attr"), INFO_DOMAIN_ATTR))
    e.append(Pair(String("INFO_fabric_attr"), INFO_FABRIC_ATTR))
    e.append(Pair(String("TXA_size"), TXA_SIZE))
    e.append(Pair(String("RXA_size"), RXA_SIZE))
    e.append(Pair(String("EPA_type"), EPA_TYPE))
    e.append(Pair(String("EPA_max_msg_size"), EPA_MAX_MSG_SIZE))
    e.append(Pair(String("DA_name"), DA_NAME))
    e.append(Pair(String("DA_threading"), DA_THREADING))
    e.append(Pair(String("DA_mr_mode"), DA_MR_MODE))
    e.append(Pair(String("DA_mr_key_size"), DA_MR_KEY_SIZE))
    e.append(Pair(String("DA_cq_data_size"), DA_CQ_DATA_SIZE))
    e.append(Pair(String("FA_name"), FA_NAME))
    e.append(Pair(String("FA_prov_name"), FA_PROV_NAME))
    e.append(Pair(String("FA_api_version"), FA_API_VERSION))
    e.append(Pair(String("CQA_size"), CQA_SIZE))
    e.append(Pair(String("CQA_format"), CQA_FORMAT))
    e.append(Pair(String("CQA_wait_obj"), CQA_WAIT_OBJ))
    e.append(Pair(String("AVA_type"), AVA_TYPE))
    e.append(Pair(String("AVA_count"), AVA_COUNT))
    e.append(Pair(String("MRA_mr_iov"), MRA_MR_IOV))
    e.append(Pair(String("MRA_iov_count"), MRA_IOV_COUNT))
    e.append(Pair(String("MRA_access"), MRA_ACCESS))
    e.append(Pair(String("MRA_offset"), MRA_OFFSET))
    e.append(Pair(String("MRA_requested_key"), MRA_REQUESTED_KEY))
    e.append(Pair(String("MRA_iface"), MRA_IFACE))
    e.append(Pair(String("IOV_base"), IOV_BASE))
    e.append(Pair(String("IOV_len"), IOV_LEN))
    e.append(Pair(String("RMAIOV_addr"), RMAIOV_ADDR))
    e.append(Pair(String("RMAIOV_len"), RMAIOV_LEN))
    e.append(Pair(String("RMAIOV_key"), RMAIOV_KEY))
    e.append(Pair(String("MSGRMA_msg_iov"), MSGRMA_MSG_IOV))
    e.append(Pair(String("MSGRMA_desc"), MSGRMA_DESC))
    e.append(Pair(String("MSGRMA_iov_count"), MSGRMA_IOV_COUNT))
    e.append(Pair(String("MSGRMA_addr"), MSGRMA_ADDR))
    e.append(Pair(String("MSGRMA_rma_iov"), MSGRMA_RMA_IOV))
    e.append(Pair(String("MSGRMA_rma_iov_count"), MSGRMA_RMA_IOV_COUNT))
    e.append(Pair(String("MSGRMA_context"), MSGRMA_CONTEXT))
    e.append(Pair(String("MSG_msg_iov"), MSG_MSG_IOV))
    e.append(Pair(String("MSG_desc"), MSG_DESC))
    e.append(Pair(String("MSG_iov_count"), MSG_IOV_COUNT))
    e.append(Pair(String("MSG_addr"), MSG_ADDR))
    e.append(Pair(String("MSG_context"), MSG_CONTEXT))
    e.append(Pair(String("MSG_data"), MSG_DATA))
    e.append(Pair(String("CQE_op_context"), CQE_OP_CONTEXT))
    e.append(Pair(String("CQE_flags"), CQE_FLAGS))
    e.append(Pair(String("CQE_len"), CQE_LEN))
    e.append(Pair(String("CQE_data"), CQE_DATA))
    e.append(Pair(String("CQERR_err"), CQERR_ERR))
    e.append(Pair(String("CQERR_prov_errno"), CQERR_PROV_ERRNO))
    e.append(Pair(String("FID_ops"), FID_OPS))
    e.append(Pair(String("FIDEP_cm"), FIDEP_CM))
    e.append(Pair(String("FIDEP_msg"), FIDEP_MSG))
    e.append(Pair(String("FIDEP_rma"), FIDEP_RMA))
    e.append(Pair(String("FIDMR_mem_desc"), FIDMR_MEM_DESC))
    e.append(Pair(String("FIDMR_key"), FIDMR_KEY))
    e.append(Pair(String("FIDDOM_ops"), FIDDOM_OPS))
    e.append(Pair(String("FIDDOM_mr"), FIDDOM_MR))
    e.append(Pair(String("FIDCQ_ops"), FIDCQ_OPS))
    e.append(Pair(String("FIDAV_ops"), FIDAV_OPS))
    e.append(Pair(String("FIDFAB_ops"), FIDFAB_OPS))
    e.append(Pair(String("OPS_close"), OPS_CLOSE))
    e.append(Pair(String("OPS_bind"), OPS_BIND))
    e.append(Pair(String("OPS_control"), OPS_CONTROL))
    e.append(Pair(String("FABOPS_domain"), FABOPS_DOMAIN))
    e.append(Pair(String("DOMOPS_av_open"), DOMOPS_AV_OPEN))
    e.append(Pair(String("DOMOPS_cq_open"), DOMOPS_CQ_OPEN))
    e.append(Pair(String("DOMOPS_endpoint"), DOMOPS_ENDPOINT))
    e.append(Pair(String("MROPS_regattr"), MROPS_REGATTR))
    e.append(Pair(String("CQOPS_read"), CQOPS_READ))
    e.append(Pair(String("CQOPS_readerr"), CQOPS_READERR))
    e.append(Pair(String("AVOPS_insert"), 8))
    e.append(Pair(String("CMOPS_getname"), CMOPS_GETNAME))
    e.append(Pair(String("RMAOPS_read"), RMAOPS_READ))
    e.append(Pair(String("RMAOPS_writemsg"), RMAOPS_WRITEMSG))
    e.append(Pair(String("MSGOPS_recv"), MSGOPS_RECV))
    e.append(Pair(String("MSGOPS_sendmsg"), MSGOPS_SENDMSG))
    e.append(Pair(String("FI_MSG"), FI_MSG))
    e.append(Pair(String("FI_RMA"), FI_RMA))
    e.append(Pair(String("FI_READ"), FI_READ))
    e.append(Pair(String("FI_WRITE"), FI_WRITE))
    e.append(Pair(String("FI_RECV"), FI_RECV))
    e.append(Pair(String("FI_SEND"), FI_SEND))
    e.append(Pair(String("FI_REMOTE_READ"), FI_REMOTE_READ))
    e.append(Pair(String("FI_REMOTE_WRITE"), FI_REMOTE_WRITE))
    e.append(Pair(String("FI_REMOTE_CQ_DATA"), FI_REMOTE_CQ_DATA))
    e.append(Pair(String("FI_FENCE"), FI_FENCE))
    e.append(Pair(String("FI_COMPLETION"), FI_COMPLETION))
    e.append(Pair(String("FI_CONTEXT"), FI_CONTEXT))
    e.append(Pair(String("FI_CONTEXT2"), FI_CONTEXT2))
    e.append(Pair(String("FI_LOCAL_COMM"), FI_LOCAL_COMM))
    e.append(Pair(String("FI_REMOTE_COMM"), FI_REMOTE_COMM))
    e.append(Pair(String("FI_HMEM"), FI_HMEM))
    e.append(Pair(String("FI_MR_LOCAL"), UInt64(FI_MR_LOCAL)))
    e.append(Pair(String("FI_MR_VIRT_ADDR"), UInt64(FI_MR_VIRT_ADDR)))
    e.append(Pair(String("FI_MR_ALLOCATED"), UInt64(FI_MR_ALLOCATED)))
    e.append(Pair(String("FI_MR_PROV_KEY"), UInt64(FI_MR_PROV_KEY)))
    e.append(Pair(String("FI_MR_ENDPOINT"), UInt64(FI_MR_ENDPOINT)))
    e.append(Pair(String("FI_MR_HMEM"), UInt64(FI_MR_HMEM)))
    e.append(Pair(String("FI_EP_RDM"), UInt64(FI_EP_RDM)))
    e.append(Pair(String("FI_AV_TABLE"), UInt64(FI_AV_TABLE)))
    e.append(Pair(String("FI_CQ_FORMAT_DATA"), UInt64(FI_CQ_FORMAT_DATA)))
    e.append(Pair(String("FI_WAIT_NONE"), UInt64(FI_WAIT_NONE)))
    e.append(Pair(String("FI_HMEM_SYSTEM"), UInt64(FI_HMEM_SYSTEM)))
    e.append(Pair(String("FI_HMEM_ROCR"), UInt64(FI_HMEM_ROCR)))
    e.append(Pair(String("FI_ADDR_UNSPEC"), FI_ADDR_UNSPEC))
    e.append(Pair(String("FI_ENABLE"), UInt64(FI_ENABLE)))
    e.append(Pair(String("FI_EAVAIL"), UInt64(FI_EAVAIL)))
    e.append(Pair(String("FI_EAGAIN"), UInt64(FI_EAGAIN)))
    e.append(Pair(String("FI_VERSION_1_5"), UInt64(FI_VERSION_1_5)))
    e.append(Pair(String("FI_VERSION_2_2"), UInt64(FI_VERSION_2_2)))
    e.append(Pair(String("FI_HMEM_CUDA"), UInt64(FI_HMEM_CUDA)))
    return e^


def _parse_u64(s: String) raises -> UInt64:
    """`Int(String)` overflows on FI_ADDR_UNSPEC (2^64-1), which is exactly
    the constant a signed parse must not mangle."""
    var v: UInt64 = 0
    var b = s.as_bytes()
    if len(b) == 0:
        raise Error("fabric_abi: empty number")
    for i in range(len(b)):
        var d = Int(b[i]) - 48
        if d < 0 or d > 9:
            raise Error("fabric_abi: not a number: " + s)
        v = v * 10 + UInt64(d)
    return v


def main() raises:
    var a = argv()
    if len(a) < 2:
        raise Error(
            "usage: fabric_abi <output-of-fabric_abi.c>  (gcc -I"
            " <libfabric>/include -o fabric_abi fabric_abi.c && ./fabric_abi >"
            " out.txt)"
        )
    var text: String
    with open(String(a[1]), "r") as fh:
        text = String(fh.read())
    var names = List[String]()
    var values = List[UInt64]()
    for raw in text.split("\n"):
        var line = String(raw).strip()
        if line.byte_length() == 0:
            continue
        var parts = line.split(" ")
        if len(parts) != 2:
            continue
        names.append(String(parts[0]))
        values.append(_parse_u64(String(parts[1])))
    print("read", len(names), "constants from", String(a[1]))

    var bad = 0
    var checked = 0
    for p in _expected():
        var found = -1
        for i in range(len(names)):
            if names[i] == p.name:
                found = i
                break
        if found < 0:
            print("MISSING from the C output:", p.name)
            bad += 1
            continue
        checked += 1
        if values[found] != p.value:
            print(
                "MISMATCH",
                p.name,
                "mojo",
                p.value,
                "!= C",
                values[found],
            )
            bad += 1
    print("checked", checked, "constants,", bad, "wrong")
    print("PASS" if bad == 0 else "FAIL")
