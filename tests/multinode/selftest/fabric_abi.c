/* Prints every libfabric struct size, field offset and constant that
 * torch_mojo_backend/mojo/tmb/ccl/transport/net_ofi.mojo hard-codes.
 *
 * libfabric's data path is `static inline` in <rdma/fi_*.h> and dispatches
 * through the fid_* ops tables (fi_writedata is ep->rma->writedata, fi_cq_read
 * is cq->ops->read, ...), exactly the way libibverbs' data path dispatches
 * through ibv_context->ops -- so, exactly like include/ibvwrap.mojo, the Mojo bindings
 * reproduce those dereferences by hand over raw byte offsets. `std.ffi` still
 * has no C-struct ABI (MOCO-3692), so every offset below is a number in the
 * Mojo source, and this program is what those numbers are checked against:
 *
 *     gcc -O0 -I <libfabric-prefix>/include -o fabric_abi fabric_abi.c
 *     ./fabric_abi > /tmp/fabric_abi.txt
 *     ./fabric_abi_check /tmp/fabric_abi.txt      # the Mojo side, must PASS
 *
 * Output is one `NAME VALUE` line per constant, sorted by nothing in
 * particular -- fabric_abi_check.mojo looks names up, it does not diff
 * positionally.
 */
#include <stdio.h>
#include <stddef.h>
#include <sys/uio.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_errno.h>

#define P(name, value) printf("%s %llu\n", name, (unsigned long long)(value))
#define SZ(t) P("SZ_" #t, sizeof(struct t))
#define OFF(prefix, t, f) P(prefix "_" #f, offsetof(struct t, f))

int main(void)
{
	/* ---- struct sizes ---- */
	SZ(fi_info);
	SZ(fi_tx_attr);
	SZ(fi_rx_attr);
	SZ(fi_ep_attr);
	SZ(fi_domain_attr);
	SZ(fi_fabric_attr);
	SZ(fi_cq_attr);
	SZ(fi_av_attr);
	SZ(fi_mr_attr);
	SZ(fi_msg_rma);
	SZ(fi_msg);
	SZ(fi_rma_iov);
	SZ(fi_cq_data_entry);
	SZ(fi_cq_err_entry);
	SZ(fid);
	SZ(fid_ep);
	SZ(fid_mr);
	SZ(fid_domain);
	SZ(fid_cq);
	SZ(fid_av);
	SZ(fid_fabric);
	P("SZ_iovec", sizeof(struct iovec));
	P("SZ_fi_addr_t", sizeof(fi_addr_t));

	/* ---- struct fi_info ---- */
	OFF("INFO", fi_info, next);
	OFF("INFO", fi_info, caps);
	OFF("INFO", fi_info, mode);
	OFF("INFO", fi_info, addr_format);
	OFF("INFO", fi_info, src_addrlen);
	OFF("INFO", fi_info, dest_addrlen);
	OFF("INFO", fi_info, src_addr);
	OFF("INFO", fi_info, dest_addr);
	OFF("INFO", fi_info, handle);
	OFF("INFO", fi_info, tx_attr);
	OFF("INFO", fi_info, rx_attr);
	OFF("INFO", fi_info, ep_attr);
	OFF("INFO", fi_info, domain_attr);
	OFF("INFO", fi_info, fabric_attr);
	OFF("INFO", fi_info, nic);

	/* ---- the attribute structs fi_getinfo hints are built in ---- */
	OFF("TXA", fi_tx_attr, caps);
	OFF("TXA", fi_tx_attr, mode);
	OFF("TXA", fi_tx_attr, op_flags);
	OFF("TXA", fi_tx_attr, inject_size);
	OFF("TXA", fi_tx_attr, size);
	OFF("TXA", fi_tx_attr, iov_limit);
	OFF("TXA", fi_tx_attr, rma_iov_limit);
	OFF("RXA", fi_rx_attr, caps);
	OFF("RXA", fi_rx_attr, mode);
	OFF("RXA", fi_rx_attr, op_flags);
	OFF("RXA", fi_rx_attr, size);
	OFF("EPA", fi_ep_attr, type);
	OFF("EPA", fi_ep_attr, max_msg_size);
	OFF("DA", fi_domain_attr, domain);
	OFF("DA", fi_domain_attr, name);
	OFF("DA", fi_domain_attr, threading);
	OFF("DA", fi_domain_attr, av_type);
	OFF("DA", fi_domain_attr, mr_mode);
	OFF("DA", fi_domain_attr, mr_key_size);
	OFF("DA", fi_domain_attr, cq_data_size);
	OFF("DA", fi_domain_attr, caps);
	OFF("DA", fi_domain_attr, mr_cnt);
	OFF("FA", fi_fabric_attr, fabric);
	OFF("FA", fi_fabric_attr, name);
	OFF("FA", fi_fabric_attr, prov_name);
	OFF("FA", fi_fabric_attr, prov_version);
	OFF("FA", fi_fabric_attr, api_version);
	OFF("CQA", fi_cq_attr, size);
	OFF("CQA", fi_cq_attr, flags);
	OFF("CQA", fi_cq_attr, format);
	OFF("CQA", fi_cq_attr, wait_obj);
	OFF("CQA", fi_cq_attr, signaling_vector);
	OFF("CQA", fi_cq_attr, wait_cond);
	OFF("CQA", fi_cq_attr, wait_set);
	OFF("AVA", fi_av_attr, type);
	OFF("AVA", fi_av_attr, rx_ctx_bits);
	OFF("AVA", fi_av_attr, count);
	OFF("AVA", fi_av_attr, ep_per_node);
	OFF("AVA", fi_av_attr, name);
	OFF("AVA", fi_av_attr, map_addr);
	OFF("AVA", fi_av_attr, flags);
	OFF("MRA", fi_mr_attr, mr_iov);
	OFF("MRA", fi_mr_attr, iov_count);
	OFF("MRA", fi_mr_attr, access);
	OFF("MRA", fi_mr_attr, offset);
	OFF("MRA", fi_mr_attr, requested_key);
	OFF("MRA", fi_mr_attr, context);
	OFF("MRA", fi_mr_attr, auth_key_size);
	OFF("MRA", fi_mr_attr, auth_key);
	OFF("MRA", fi_mr_attr, iface);
	OFF("MRA", fi_mr_attr, device);
	OFF("MRA", fi_mr_attr, hmem_data);
	OFF("MRA", fi_mr_attr, page_size);
	OFF("MRA", fi_mr_attr, base_mr);
	OFF("MRA", fi_mr_attr, sub_mr_cnt);
	P("IOV_base", offsetof(struct iovec, iov_base));
	P("IOV_len", offsetof(struct iovec, iov_len));
	OFF("RMAIOV", fi_rma_iov, addr);
	OFF("RMAIOV", fi_rma_iov, len);
	OFF("RMAIOV", fi_rma_iov, key);
	OFF("MSGRMA", fi_msg_rma, msg_iov);
	OFF("MSGRMA", fi_msg_rma, desc);
	OFF("MSGRMA", fi_msg_rma, iov_count);
	OFF("MSGRMA", fi_msg_rma, addr);
	OFF("MSGRMA", fi_msg_rma, rma_iov);
	OFF("MSGRMA", fi_msg_rma, rma_iov_count);
	OFF("MSGRMA", fi_msg_rma, context);
	OFF("MSGRMA", fi_msg_rma, data);
	OFF("MSG", fi_msg, msg_iov);
	OFF("MSG", fi_msg, desc);
	OFF("MSG", fi_msg, iov_count);
	OFF("MSG", fi_msg, addr);
	OFF("MSG", fi_msg, context);
	OFF("MSG", fi_msg, data);

	/* ---- completions ---- */
	OFF("CQE", fi_cq_data_entry, op_context);
	OFF("CQE", fi_cq_data_entry, flags);
	OFF("CQE", fi_cq_data_entry, len);
	OFF("CQE", fi_cq_data_entry, buf);
	OFF("CQE", fi_cq_data_entry, data);
	OFF("CQERR", fi_cq_err_entry, op_context);
	OFF("CQERR", fi_cq_err_entry, flags);
	OFF("CQERR", fi_cq_err_entry, len);
	OFF("CQERR", fi_cq_err_entry, buf);
	OFF("CQERR", fi_cq_err_entry, data);
	OFF("CQERR", fi_cq_err_entry, tag);
	OFF("CQERR", fi_cq_err_entry, olen);
	OFF("CQERR", fi_cq_err_entry, err);
	OFF("CQERR", fi_cq_err_entry, prov_errno);
	OFF("CQERR", fi_cq_err_entry, err_data);
	OFF("CQERR", fi_cq_err_entry, err_data_size);
	OFF("CQERR", fi_cq_err_entry, src_addr);

	/* ---- the fid_* objects and their ops tables (the data path) ---- */
	OFF("FID", fid, fclass);
	OFF("FID", fid, context);
	OFF("FID", fid, ops);
	P("FIDEP_fid", offsetof(struct fid_ep, fid));
	OFF("FIDEP", fid_ep, ops);
	OFF("FIDEP", fid_ep, cm);
	OFF("FIDEP", fid_ep, msg);
	OFF("FIDEP", fid_ep, rma);
	OFF("FIDMR", fid_mr, mem_desc);
	OFF("FIDMR", fid_mr, key);
	OFF("FIDDOM", fid_domain, ops);
	OFF("FIDDOM", fid_domain, mr);
	OFF("FIDCQ", fid_cq, ops);
	OFF("FIDAV", fid_av, ops);
	OFF("FIDFAB", fid_fabric, ops);
	OFF("FIDFAB", fid_fabric, api_version);
	OFF("OPS", fi_ops, close);
	OFF("OPS", fi_ops, bind);
	OFF("OPS", fi_ops, control);
	OFF("FABOPS", fi_ops_fabric, domain);
	OFF("DOMOPS", fi_ops_domain, av_open);
	OFF("DOMOPS", fi_ops_domain, cq_open);
	OFF("DOMOPS", fi_ops_domain, endpoint);
	OFF("MROPS", fi_ops_mr, reg);
	OFF("MROPS", fi_ops_mr, regattr);
	OFF("CQOPS", fi_ops_cq, read);
	OFF("CQOPS", fi_ops_cq, readerr);
	OFF("CQOPS", fi_ops_cq, strerror);
	OFF("AVOPS", fi_ops_av, insert);
	OFF("CMOPS", fi_ops_cm, getname);
	OFF("RMAOPS", fi_ops_rma, read);
	OFF("RMAOPS", fi_ops_rma, readmsg);
	OFF("RMAOPS", fi_ops_rma, write);
	OFF("RMAOPS", fi_ops_rma, writemsg);
	OFF("RMAOPS", fi_ops_rma, writedata);
	OFF("MSGOPS", fi_ops_msg, recv);
	OFF("MSGOPS", fi_ops_msg, recvmsg);
	OFF("MSGOPS", fi_ops_msg, send);
	OFF("MSGOPS", fi_ops_msg, sendmsg);
	OFF("MSGOPS", fi_ops_msg, senddata);

	/* ---- capability / flag / enum constants ---- */
	P("FI_MSG", FI_MSG);
	P("FI_RMA", FI_RMA);
	P("FI_READ", FI_READ);
	P("FI_WRITE", FI_WRITE);
	P("FI_RECV", FI_RECV);
	P("FI_SEND", FI_SEND);
	P("FI_REMOTE_READ", FI_REMOTE_READ);
	P("FI_REMOTE_WRITE", FI_REMOTE_WRITE);
	P("FI_REMOTE_CQ_DATA", FI_REMOTE_CQ_DATA);
	P("FI_COMPLETION", FI_COMPLETION);
	P("FI_FENCE", FI_FENCE);
	P("FI_MORE", FI_MORE);
	P("FI_INJECT", FI_INJECT);
	P("FI_DELIVERY_COMPLETE", FI_DELIVERY_COMPLETE);
	P("FI_TRANSMIT_COMPLETE", FI_TRANSMIT_COMPLETE);
	P("FI_HMEM", FI_HMEM);
	P("FI_RMA_EVENT", FI_RMA_EVENT);
	P("FI_SOURCE", FI_SOURCE);
	P("FI_LOCAL_COMM", FI_LOCAL_COMM);
	P("FI_REMOTE_COMM", FI_REMOTE_COMM);
	P("FI_MR_LOCAL", FI_MR_LOCAL);
	P("FI_MR_RAW", FI_MR_RAW);
	P("FI_MR_VIRT_ADDR", FI_MR_VIRT_ADDR);
	P("FI_MR_ALLOCATED", FI_MR_ALLOCATED);
	P("FI_MR_PROV_KEY", FI_MR_PROV_KEY);
	P("FI_MR_MMU_NOTIFY", FI_MR_MMU_NOTIFY);
	P("FI_MR_RMA_EVENT", FI_MR_RMA_EVENT);
	P("FI_MR_ENDPOINT", FI_MR_ENDPOINT);
	P("FI_MR_HMEM", FI_MR_HMEM);
	P("FI_EP_RDM", FI_EP_RDM);
	P("FI_AV_MAP", FI_AV_MAP);
	P("FI_AV_TABLE", FI_AV_TABLE);
	P("FI_CQ_FORMAT_DATA", FI_CQ_FORMAT_DATA);
	P("FI_HMEM_SYSTEM", FI_HMEM_SYSTEM);
	P("FI_HMEM_CUDA", FI_HMEM_CUDA);
	P("FI_HMEM_ROCR", FI_HMEM_ROCR);
	P("FI_ADDR_UNSPEC", FI_ADDR_UNSPEC);
	P("FI_ENABLE", FI_ENABLE);
	P("FI_EAVAIL", FI_EAVAIL);
	P("FI_EAGAIN", FI_EAGAIN);
	P("FI_ENOSYS", FI_ENOSYS);
	P("FI_EBADFLAGS", FI_EBADFLAGS);
	P("FI_WAIT_NONE", FI_WAIT_NONE);
	P("FI_ADDR_CXI", FI_ADDR_CXI);
	P("FI_CLASS_EP", FI_CLASS_EP);
	P("FI_CONTEXT", FI_CONTEXT);
	P("FI_CONTEXT2", FI_CONTEXT2);
	P("FI_VERSION_1_5", FI_VERSION(1, 5));
	P("FI_VERSION_2_2", FI_VERSION(2, 2));
	P("FI_MAJOR_VERSION", FI_MAJOR_VERSION);
	P("FI_MINOR_VERSION", FI_MINOR_VERSION);
	return 0;
}
