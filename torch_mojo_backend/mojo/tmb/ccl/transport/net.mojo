# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/transport/net.cc
#
# The inter-node hop: GPUDirect RDMA, no vendor collective library. One
# connection per remote node, to the rank holding the SAME local_rank there
# -- so the 8 ranks of a node drive 8 independent NICs and each rank only
# ever exchanges its own 1/local_world shard.
#
# TWO TRANSPORTS, ONE ENGINE. Everything below the "progress engine" heading
# is transport-independent; the six operations it needs (post a payload,
# post an immediate, post the flush read, poll completions, fill the
# bootstrap blob, attach a peer) are implemented twice:
#
#   * `transport/net_ib/` -- InfiniBand, an RC queue pair per peer, one
#     RDMA_WRITE_WITH_IMM per shard. This is the path the two-node H100
#     numbers were measured on and it is unchanged.
#   * `transport/net_ofi.mojo` -- HPE Slingshot through the `cxi` provider (Adastra
#     nodes have four /dev/cxi NICs and no InfiniBand at all). One
#     connectionless RDM endpoint; since cxi implements no
#     write-with-immediate, a shard is an RMA write followed by a FENCED
#     zero-length message carrying the immediate as remote CQ data. See that
#     file's header.
#
# Which one is used is decided at `ib_setup` time by `MOJOCCL_NET`, or by
# what the machine actually has. The names in this file kept their `ib_`
# prefix: they are the transport's API to `init.mojo`/`enqueue.mojo` and renaming them
# would have churned every call site for nothing.
#
# The GPUDirect flush. Seeing the RDMA_WRITE_WITH_IMM completion does NOT
# mean the payload is visible in GPU memory: the completion lands in host
# memory and the payload in the GPU's BAR, two different PCIe destinations
# with no ordering between them. A read from the GPU BAR flushes the posted
# writes ahead of it, so an exchange finishes with a 4-byte RDMA_READ of
# the inbox over a self-connected QP -- exactly NCCL's `gpuFlush` QP
# (nccl:src/transport/net_ib/p2p.cc:589-602). Measured 1.9 us.
#
# Flow control: EXPLICIT CREDITS. The inbox is carved once into `nslots`
# fixed slot groups and exchange e lands in group `e % nslots`, so peer B
# writes the same bytes at e and e+nslots. What makes reuse safe is a credit,
# not stream order: after my consumer kernel has read group g, my proxy
# RDMA-writes every peer a cumulative "I have consumed through exchange c"
# counter, and a peer may post exchange e only once every receiver has
# released the group e will land in (`c >= e - nslots`). That is NCCL's
# head/tail pair in miniature (nccl:src/transport/net.cc, the
# recvNetHead/step counters; nccl:src/device/prims_simple.h for the device
# side of the same idea).
#
# It replaces an earlier double-buffer-by-parity argument that derived
# reuse safety from stream order -- "B cannot reach e+2 before receiving my
# e+1 data, which I send only after my own add kernel for e". That chain
# holds only when exactly one exchange is in flight; the pipelined schedule
# (enqueue.mojo `_do_allreduce`) issues the reduce-scatter of later chunks
# before the add of earlier ones and breaks it. The credit is the
# replacement proof, and it is a proof rather than a timing margin.
#
# Two things survive from the old argument unchanged. Every exchange must
# still be all-to-all, because an arrival tally of `nrecv` is what completes
# one -- a rank with nothing to contribute sends EMPTY_SHARD_BYTES
# (enqueue.mojo). And every slot group must be a FIXED byte range for every
# exchange alike, never sized from the message in flight (`_inbox_base` in
# include/comm.mojo carries the case that broke). The immediate carries the
# exchange counter and a credit bit, so an arrival is tallied against its own
# exchange and a credit is never mistaken for data.
#
# `credit_upto` is how the host tells the engine what has been consumed
# without a further kernel: it is the number of consumer kernels already
# enqueued on the stream ahead of this exchange's request kernel. When the
# proxy observes the request, that kernel has run, so every kernel enqueued
# before it has completed -- the same stream-order argument that makes the
# shard final at that point.
#
# The inbox lives in the region's own network area, never aliased onto the
# intra-node staging: a peer node writes it as soon as ITS reduce-scatter is
# done, which is ordered against neither mine nor a local peer still reading
# the previous generation (see init.mojo's ncclCommInitRank).

from std.atomic import Atomic, Ordering
from std.os import getenv
from std.time import perf_counter_ns
from std.sys import size_of
from std.memory.alloc import unsafe_alloc
from max.gpu.host import DeviceContext, DeviceStream
from std.ffi import OwnedDLHandle, external_call
from std.utils import StaticTuple

from tmb.ccl.env_vars import MOJOCCL_IB_TIMEOUT_S, MOJOCCL_IB_TRACE
from tmb.ccl.include.nccl_device.gin.proxy.gin_proxy import (
    proxy_request,
    proxy_wait,
)
from tmb.ccl.include.plugin.nccl_net import (
    MAX_NODES,
    NC_FLUSH,
    NC_RECV,
    NC_SEND,
    NetCompletion,
)
from tmb.ccl.misc.cudawrap import free_host, open_driver
from tmb.ccl.misc.strongstream import launch_host_func
from tmb.ccl.misc.utils import P8, alloc_bytes
from tmb.ccl.plugin.net import NET_FABRIC, NET_VERBS
from tmb.ccl.transport.net_ib.connect import (
    VerbsNet,
    vrb_blob_base,
    vrb_blob_key,
    vrb_connect_flush,
    vrb_connect_peer,
    vrb_local_info,
    vrb_teardown,
)
from tmb.ccl.transport.net_ib.p2p import (
    vrb_poll,
    vrb_post_flush,
    vrb_post_imm,
    vrb_post_payload,
)
from tmb.ccl.transport.net_ofi import (
    FabricNet,
    fab_add_peer,
    fab_blob_base,
    fab_blob_key,
    fab_describe,
    fab_local_info,
    fab_poll,
    fab_post_flush,
    fab_post_imm,
    fab_post_recvs,
    fab_post_write,
    fab_teardown,
)


comptime WORK_SLOTS = 512
# Completions pulled out of the transport per engine step.
comptime COMP_BATCH = 16
comptime DEFAULT_IB_TIMEOUT_S: Float64 = 60.0


comptime BATCH_POLL_NS = 2_000_000
"""How long the proxy keeps yielding rather than sleeping once a pipelined
reduce-scatter or all-gather has released one chunk and more are due. The
20 us nanosleep wakes after a median 75 us on this cluster (Nsight, 2x8
H100), and a root reduce-scatter overlapped 11-13 of them: fused root fp32
1263 -> 1181 us, block 617 -> 534 (NCCL 1051 / 522). Bounded so a stalled
peer costs a burst of yields, not a hot core. GPT-2 XL FSDP2 on 2x8 H100
measured 66.8k tok/s with it against 65.1k without. NVIDIA-only because
that is where it was measured; AMD keeps the plain backoff."""


# Bytes of the inbox read back by the flush; any read of the destination
# device flushes the writes ahead of it, the size is irrelevant.
comptime FLUSH_BYTES = 4

# Largest inbox slot-group count the engine will accept; bounds the arrival
# tally and, through it, how many exchanges may be outstanding at once.
comptime PIPE_MAX_SLOTS = 16
# Immediate layout: bit 31 marks a credit, bits 0..30 carry the exchange
# counter. 2^31 exchanges is ~10^4 DDP training runs, and the counter never
# wraps within one communicator.
comptime IMM_CREDIT_BIT: UInt32 = 0x8000_0000
comptime IMM_SEQ_MASK: UInt32 = 0x7FFF_FFFF
# Credit landing pad: one 64-byte line per sender in the region's credit
# area, so two peers' credits never share a cache line. The bytes are never
# read -- the immediate is the message -- but a real address is needed
# because a zero-length RDMA write is not worth relying on across HCAs.
comptime CREDIT_SLOT_BYTES = 64
comptime CREDIT_AREA_BYTES = 4096
comptime CREDIT_PAYLOAD_BYTES = 4

# Payload a rank sends when it has nothing to contribute to an exchange:
# a broadcast, or an allreduce whose shard table leaves this rank empty
# (7 of 8 local ranks on DDP's 4-byte AVG allreduce). Every exchange has to
# be all-to-all because an arrival tally of `npeers` is what completes one,
# so a rank that stayed silent would hang its peers.
comptime EMPTY_SHARD_BYTES = 16

comptime OP_UNKNOWN = 0
comptime OP_ALLREDUCE = 1
comptime OP_BROADCAST = 2
comptime OP_ALLGATHER = 3
comptime OP_REDUCE_SCATTER = 4
"""Which collective an exchange belongs to, carried in its work item for the
stall messages only. `enqueue.mojo` passes it to `ib_enqueue_request`."""


def _op_name(kind: Int) -> String:
    if kind == OP_ALLREDUCE:
        return String("allreduce")
    if kind == OP_BROADCAST:
        return String("broadcast")
    if kind == OP_ALLGATHER:
        return String("allgather")
    if kind == OP_REDUCE_SCATTER:
        return String("reduce_scatter")
    return String("?")


# The proxy mailbox: 64-bit words the GPU and the progress thread pass
# exchange counters through, plus a stop word the host sets at teardown. A
# cache line apart so the GPU's writes to REQUEST never invalidate the line
# the CPU is writing DONE into.
#
# REQUEST and CONSUMED are written by the device, DONE by the thread, STOP by
# the host. CONSUMED exists for the fused kernel (all_reduce_gin.mojo): it is
# "my consumer for exchange e has RUN", the credit that frees the inbox slot
# group, published from the kernel rather than inferred on the host from the
# order kernels were enqueued in.
comptime MB_REQUEST = 0
comptime MB_DONE = 64
comptime MB_STOP = 128
comptime MB_CONSUMED = 192
comptime MB_BYTES = 256


struct IbPeer(Copyable, Movable):
    """One remote node, as the engine sees it.

    Deliberately transport-free: the queue pair (verbs) or address-vector
    entry (libfabric) that actually reaches this peer lives in the
    transport's own state, indexed by this peer's position in
    `IbState.peers`. `remote_base` is what a region offset is measured from
    on the wire -- the peer's virtual address under InfiniBand and under a
    FI_MR_VIRT_ADDR provider, and 0 under a provider whose RMA targets are
    offsets into the registered region (what cxi reports).
    """

    var node: Int
    var remote_base: Int
    var remote_key: UInt64

    def __init__(out self, node: Int, remote_base: Int, remote_key: UInt64):
        self.node = node
        self.remote_base = remote_base
        self.remote_key = remote_key


struct IbWork(Copyable, Movable):
    """One inter-node exchange, described by the host for the engine.

    Lives in a ring inside `IbState` indexed by `(seq-1) % WORK_SLOTS`: the
    counter is dense and starts at 1, so a slot is implied by the sequence
    number and no queue is needed. A slot is refilled only once the engine
    has set `status` away from 0.
    """

    var state: Int
    var send_addr: Int
    var send_bytes: Int
    var send_node_stride: Int  # zero broadcasts; otherwise one payload per node
    var inbox_base: Int  # region offset of this exchange's inbox slot group
    var slot_bytes: Int
    var do_send: Int
    var nrecv: Int
    var flush_addr: Int
    var seq: Int
    var status: Int  # 0 running, 1 done, 2 failed
    # Highest exchange whose consumer kernel is already enqueued ahead of
    # this one's request on the stream -- the credit the engine publishes
    # when it picks this exchange up. See the header's flow-control note.
    var credit_upto: Int
    # 1 once this exchange's data write is posted to every peer. The exchange
    # is not done until each of those sends has completed on ITS queue pair
    # (`IbState.send_done`), or the consumer kernel could overwrite a buffer
    # the NIC is still reading for a slower peer.
    var sent: Int
    var t0: Int  # perf_counter_ns when it was posted, for the trace
    # What the host was issuing when it filled this slot -- kind (see
    # `_op_name`), which chunk of how many, and the chunk's element count.
    # Stall messages ("exchange 78, allreduce chunk 3 of 14, 1703936
    # elements") and the proxy's between-chunk polling (`BATCH_POLL_NS`).
    var op_kind: Int
    var op_chunk: Int
    var op_nchunks: Int
    var op_numel: Int

    def __init__(out self):
        self.state = 0
        self.send_addr = 0
        self.send_bytes = 0
        self.send_node_stride = 0
        self.inbox_base = 0
        self.slot_bytes = 0
        self.do_send = 0
        self.nrecv = 0
        self.flush_addr = 0
        self.seq = 0
        self.status = 1
        self.credit_upto = 0
        self.sent = 0
        self.t0 = 0
        self.op_kind = OP_UNKNOWN
        self.op_chunk = 0
        self.op_nchunks = 0
        self.op_numel = 0


struct IbState(Movable):
    """Everything the inter-node hop owns, per communicator."""

    # Exactly one of `vrb` / `fab` is non-zero: the address of a heap
    # `VerbsNet` or `FabricNet`. Held as raw addresses rather than as two
    # struct fields because constructing either one dlopens its library, and
    # on a Slingshot node there is no libibverbs to dlopen at all.
    var net: Int
    var vrb: Int
    var fab: Int
    var netdev: String  # the HCA name, or "<provider>:<domain>"
    var peers: List[IbPeer]
    var do_flush: Bool
    var region: Int
    var my_node: Int
    var nnodes: Int
    var exchanges: Int
    var error: Int
    var timeout_ns: Int
    # --- the progress engine (see `ib_drive`) ---
    var nslots: Int  # inbox slot groups; exchange e lands in `e % nslots`
    var credit_off: Int  # region offset of the credit landing pad
    var request_seq: Int  # highest exchange handed to the engine
    var posted_seq: Int  # highest exchange whose data writes are posted
    var done_seq: Int  # highest exchange fully arrived, sent and flushed
    var flush_seq: Int  # exchange whose flush read is outstanding (0: none)
    var flush_done: Int  # flush-read completions seen and not yet consumed
    var flush_t0: Int
    # Per peer, the highest exchange whose data write has completed on that
    # peer's queue pair (the wr_id of the completion is the exchange number).
    # Per QP, not one global count: RC completes in order on ONE queue pair
    # only, so with three or more nodes a completion for e+1 towards a fast
    # peer can land before the completion for e towards a slow one, and a
    # global tally would call e's sends finished while the NIC still reads
    # e's source for the slow peer.
    var send_done: List[Int]
    var credit_sent: Int  # highest credit published to the peers
    # Highest "my consumer has run" the fused kernel published in MB_CONSUMED.
    # Device-attested, unlike `consumed_enqueued`, so it may be used directly
    # (see `ib_drive`), and monotone because one thread stores it in order.
    var credit_device: Int
    var tally: List[Int]  # arrivals, indexed by `seq % nslots`
    var credit_recv: List[Int]  # per peer, highest credit it published
    var last_progress_ns: Int
    var stall_seq: Int  # exchange the last credit stall was counted against
    var n_credit_stalls: Int
    # How far the calling thread got ahead of the engine, and how often it
    # had to wait for a ring slot (`_await_ring_slot`). Written only by the
    # calling thread.
    var max_ahead: Int
    var n_ring_waits: Int
    # Time-weighted breakdown of what the engine is waiting on, sampled once
    # per `ib_drive` step: the head exchange is blocked by flow control, by
    # the peer's data not having arrived, or by this rank's own writes not
    # having completed. Engine-thread only.
    var t_blocked_credit_ns: Int
    var t_blocked_arrive_ns: Int
    var t_blocked_sends_ns: Int
    var t_last_sample_ns: Int
    # Host-side: highest exchange whose consumer kernel has been ENQUEUED.
    # Snapshotted into each work item as `credit_upto` (see
    # `ib_note_consumed`); never touched by the engine thread.
    var consumed_enqueued: Int
    # Scratch buffers and the work ring are held as raw addresses: a
    # `Pointer[..., MutAnyOrigin]` cannot be a struct field, and these
    # outlive every borrow anyway (allocated once, freed never -- a few
    # hundred bytes per communicator).
    var comps: Int  # NetCompletion[COMP_BATCH], filled by the transport
    var ts: Int  # struct timespec scratch for the idle nanosleep
    # A second timespec, for the calling thread's back-off in
    # `_await_ring_slot`: `ts` belongs to the progress thread and the two
    # would otherwise write the same 16 bytes.
    var ts_host: Int
    var works: Int
    var work_next: Int
    var mailbox: Int  # pinned host address
    var mailbox_dev: Int  # the same memory as a kernel addresses it
    var error_word: Int  # region + error_offset, for the wait kernel
    # The communicator's pinned STATUS PAGE (the abort word is its first
    # word), twice: as the progress thread addresses it, and as a kernel
    # does. The device mapping is for `proxy_wait`, which has to leave a spin
    # the host cannot interrupt any other way; the host address is what stops
    # this rank putting an unproduced shard on the wire (`_comm_stopped`).
    # Both 0 until `ib_set_abort_word` runs.
    var status_host: Int
    var abort_dev: Int
    var proxy: Bool
    var thread_id: Int
    var trace: Bool
    var t_post_ns: Int
    var t_wait_ns: Int
    var t_flush_ns: Int
    var n_exchanges: Int
    # Highest exchange a PEER has sent data for. A peer only sends
    # exchange e once its own GPU released e, and every rank of a
    # communicator calls `ib_next_seq` the same number of times in the
    # same order, so `peer_seq_seen > request_seq` means MY GPU is the
    # one that has not got there. That is the only thing the silent side
    # of a stall knows about it, and without it only the noticing side
    # ever prints (see the watchdog in `ib_drive`).
    var peer_seq_seen: Int
    var watchdog_said: Int

    def __init__(
        out self,
        net: Int,
        region: Int,
        my_node: Int,
        nnodes: Int,
        nslots: Int,
        credit_off: Int,
    ):
        self.net = net
        self.vrb = 0
        self.fab = 0
        self.netdev = String("")
        self.peers = List[IbPeer]()
        self.do_flush = True
        self.region = region
        self.my_node = my_node
        self.nnodes = nnodes
        self.exchanges = 0
        self.error = 0
        self.timeout_ns = Int(DEFAULT_IB_TIMEOUT_S * 1.0e9)
        self.nslots = nslots
        self.credit_off = credit_off
        self.request_seq = 0
        self.posted_seq = 0
        self.done_seq = 0
        self.flush_seq = 0
        self.flush_done = 0
        self.flush_t0 = 0
        self.send_done = List[Int]()
        self.credit_sent = 0
        self.credit_device = 0
        self.tally = List[Int]()
        for _ in range(nslots):
            self.tally.append(0)
        self.credit_recv = List[Int]()
        self.last_progress_ns = 0
        self.stall_seq = 0
        self.n_credit_stalls = 0
        self.max_ahead = 0
        self.n_ring_waits = 0
        self.t_blocked_credit_ns = 0
        self.t_blocked_arrive_ns = 0
        self.t_blocked_sends_ns = 0
        self.t_last_sample_ns = 0
        self.consumed_enqueued = 0
        self.comps = Int(alloc_bytes(COMP_BATCH * size_of[NetCompletion]()))
        self.ts = Int(alloc_bytes(16))
        self.ts_host = Int(alloc_bytes(16))
        self.works = Int(unsafe_alloc[IbWork](WORK_SLOTS))
        var wp = Pointer[IbWork, MutAnyOrigin](unsafe_from_address=self.works)
        for i in range(WORK_SLOTS):
            wp[unsafe_offset=i] = IbWork()
        self.work_next = 0
        self.mailbox = 0
        self.mailbox_dev = 0
        self.error_word = region
        self.status_host = 0
        self.abort_dev = 0
        # Fused + proxy is the measured production schedule on H100 and
        # MI300A (Adastra job 5417296, 2026-09-15: disabling it loses 22.31%
        # in full-model ABBA). Only synchronous selftests opt out below.
        self.proxy = True
        self.thread_id = 0
        self.trace = getenv(MOJOCCL_IB_TRACE, "0") != "0"
        self.t_post_ns = 0
        self.t_wait_ns = 0
        self.t_flush_ns = 0
        self.n_exchanges = 0
        self.peer_seq_seen = 0
        self.watchdog_said = 0


@always_inline
def _st(ib: Int) -> Pointer[IbState, MutAnyOrigin]:
    return Pointer[IbState, MutAnyOrigin](unsafe_from_address=ib)


@always_inline
def _vn(st: IbState) -> Pointer[VerbsNet, MutAnyOrigin]:
    """The libibverbs transport. Only valid when `st.net == NET_VERBS`."""
    return Pointer[VerbsNet, MutAnyOrigin](unsafe_from_address=st.vrb)


@always_inline
def _fn(st: IbState) -> Pointer[FabricNet, MutAnyOrigin]:
    """The libfabric transport. Only valid when `st.net == NET_FABRIC`."""
    return Pointer[FabricNet, MutAnyOrigin](unsafe_from_address=st.fab)


@always_inline
def _comp(st: IbState, i: Int) -> Pointer[NetCompletion, MutAnyOrigin]:
    return Pointer[NetCompletion, MutAnyOrigin](
        unsafe_from_address=st.comps + i * size_of[NetCompletion]()
    )


# `IbState.error` and `IbWork.status` are written by the proxy thread (or the
# internal callback thread) inside `ib_drive` and read by
# the calling thread -- `ib_error` from the torch-facing calling thread,
# `ib_enqueue`'s ring-reuse check from the same -- with no other
# synchronization between the two. Every access goes through these two
# helpers rather than a plain field read/write so that relationship is a
# real release/acquire pair, not two threads racing a plain `Int`.
@always_inline
def _load_atomic_i(p: Pointer[Int, MutAnyOrigin]) -> Int:
    return Int(
        Atomic[Scalar[DType.int64]].load[ordering=Ordering.ACQUIRE](
            p.unsafe_bitcast[Int64]()
        )
    )


@always_inline
def _store_atomic_i(p: Pointer[Int, MutAnyOrigin], v: Int):
    Atomic[Scalar[DType.int64]].store[ordering=Ordering.RELEASE](
        p.unsafe_bitcast[Int64](), Int64(v)
    )


@always_inline
def _err_ptr(mut st: IbState) -> Pointer[Int, MutAnyOrigin]:
    return Pointer(to=st.error).unsafe_origin_cast[MutAnyOrigin]()


@always_inline
def _status_ptr(mut w: IbWork) -> Pointer[Int, MutAnyOrigin]:
    return Pointer(to=w.status).unsafe_origin_cast[MutAnyOrigin]()


# ===-------------------------------------------------------------------=== #
# The progress engine
# ===-------------------------------------------------------------------=== #
#
# One non-blocking step function, `ib_drive`, shared by all three drivers:
# the progress thread, the internal stream callback and the
# GPU-free self-tests. They differ only in who advances `request_seq` (the
# mailbox, the callback, the calling thread) and who reads `done_seq`.
#
# The engine keeps several exchanges in flight. Per step it publishes any
# credit the next exchange carries, posts that exchange if flow control
# allows, drains the completion queue, and retires exchanges in sequence
# order. Retiring is strictly in order even though arrivals are not: RC
# ordering is per queue pair, so peer B's message for e+1 can overtake peer
# C's for e, and `done_seq` is what the GPU waits on.


@always_inline
def _work(st: IbState, seq: Int) -> Pointer[IbWork, MutAnyOrigin]:
    """The ring slot exchange `seq` lives in."""
    return Pointer[IbWork, MutAnyOrigin](
        unsafe_from_address=st.works
        + ((seq - 1) % WORK_SLOTS) * size_of[IbWork]()
    )


@always_inline
def _sender_slot(st: IbState, peer_node: Int) -> Int:
    """Where MY message sits among the receiver's per-sender slots: senders
    are indexed by node, compacted past the receiver's own node."""
    return st.my_node if st.my_node < peer_node else st.my_node - 1


def _release_on_error(mut st: IbState):
    """An exchange failed: release everything outstanding rather than leave
    the GPU's spin kernels (or a host waiter) hanging past the point where
    `ncclCommGetAsyncError` could report it."""
    while st.done_seq < st.request_seq:
        st.done_seq += 1
        ref w = _work(st, st.done_seq)[]
        _store_atomic_i(_status_ptr(w), 2)
    st.flush_seq = 0


def _send_credits(mut st: IbState, upto: Int) -> Bool:
    """Publish "I have consumed through exchange `upto`" to every peer.

    Credits are cumulative, so only the newest is ever on the wire: one
    immediate per peer, carrying the credit bit and the number. On the verbs
    path that immediate rides an unsignaled 4-byte write into the peer's
    credit landing pad, because RDMA_WRITE_WITH_IMM is the only way to send
    one; on the libfabric path it is a zero-length message and the landing
    pad goes unused. Unlike a data write, a credit needs no fence: it
    announces nothing that was written.
    """
    if upto <= st.credit_sent:
        return False
    var imm = IMM_CREDIT_BIT | (UInt32(upto) & IMM_SEQ_MASK)
    for i in range(len(st.peers)):
        ref p = st.peers[i]
        var rc = 0
        if st.net == NET_VERBS:
            rc = vrb_post_imm(
                _vn(st)[],
                i,
                p.remote_base
                + st.credit_off
                + _sender_slot(st, p.node) * CREDIT_SLOT_BYTES,
                p.remote_key,
                CREDIT_PAYLOAD_BYTES,
                imm,
                upto,
            )
        else:
            rc = fab_post_imm(_fn(st)[], i, imm, False, upto)
        if rc != 0:
            _store_atomic_i(_err_ptr(st), 8)
            return False
    st.credit_sent = upto
    return True


@always_inline
def _can_post(st: IbState, seq: Int) -> Bool:
    """Flow control: exchange `seq` lands in slot group `seq % nslots`, whose
    previous occupant was `seq - nslots`, so every peer must have released
    that one first."""
    var need = seq - st.nslots
    if need <= 0:
        return True
    for i in range(len(st.credit_recv)):
        if st.credit_recv[i] < need:
            return False
    return True


def _post_data(mut st: IbState, mut w: IbWork) -> Bool:
    """Post this exchange's payload, plus its immediate, to every peer.

    One operation per peer on verbs (RDMA_WRITE_WITH_IMM); two on
    libfabric, and in two separate passes rather than interleaved: the cxi
    provider has no write-with-immediate, so the immediate is a message that
    must not overtake the payload it announces, and the fence that orders it
    (`FI_FENCE` on the first one) drains everything already posted. Writes
    first, then notifications, means one fence per exchange instead of one
    per peer -- with the fence between the two passes the remaining
    notifications are ordered by command-queue position alone.
    """
    if w.do_send == 0 or w.send_bytes <= 0:
        return True
    var imm = UInt32(w.seq) & IMM_SEQ_MASK
    for i in range(len(st.peers)):
        ref p = st.peers[i]
        var raddr = (
            p.remote_base
            + w.inbox_base
            + _sender_slot(st, p.node) * w.slot_bytes
        )
        var rc = 0
        if st.net == NET_VERBS:
            rc = vrb_post_payload(
                _vn(st)[],
                i,
                w.send_addr + p.node * w.send_node_stride,
                w.send_bytes,
                raddr,
                p.remote_key,
                imm,
                w.seq,
            )
        else:
            rc = fab_post_write(
                _fn(st)[],
                i,
                w.send_addr + p.node * w.send_node_stride,
                w.send_bytes,
                raddr,
                p.remote_key,
                w.seq,
            )
        if rc != 0:
            _store_atomic_i(_err_ptr(st), 1)
            return False
    if st.net == NET_FABRIC:
        for i in range(len(st.peers)):
            if fab_post_imm(_fn(st)[], i, imm, i == 0, w.seq) != 0:
                _store_atomic_i(_err_ptr(st), 1)
                return False
    w.sent = 1
    return True


@always_inline
def _sends_done(st: IbState, mut w: IbWork) -> Bool:
    """Every peer has completed this exchange's data write.

    `send_done[i]` is a running maximum, so `>= seq` is only a valid test if
    a LATER exchange's completion cannot arrive before this one's. On verbs
    that is RC ordering: completions on one queue pair are in order. On
    libfabric completion order is not promised in general, but the fence
    that separates exchange e's notifications from everything posted after
    them also separates e's writes from e+1's -- e+1's write command does
    not start until e's writes have completed -- so the same property holds,
    and `fab_setup` refuses a provider that will not honour FI_FENCE."""
    if w.sent == 0:
        return True
    for i in range(len(st.send_done)):
        if st.send_done[i] < w.seq:
            return False
    return True


def _consume_wc(mut st: IbState, c: NetCompletion) -> Int:
    """Account for one completion; -1 for a failed one (error recorded).

    Every completion is classified here, never "the one this exchange is
    waiting for": an arrival for a later exchange can land at any time, and
    a loop that dropped it would lose a tally somebody is waiting on. The
    transport has already normalized it -- which peer, which immediate,
    which exchange -- and has already reposted whatever receive resource the
    completion consumed.
    """
    if c.status != 0:
        _store_atomic_i(_err_ptr(st), c.status)
        return -1
    if c.kind == NC_RECV:
        var seq = Int(c.imm & IMM_SEQ_MASK)
        if (c.imm & IMM_CREDIT_BIT) != 0:
            if c.peer >= 0 and st.credit_recv[c.peer] < seq:
                st.credit_recv[c.peer] = seq
        else:
            st.tally[seq % st.nslots] += 1
            if seq > st.peer_seq_seen:
                st.peer_seq_seen = seq
        return 0
    if c.kind == NC_SEND:
        # The wr_id / context of a data write is the exchange number, dense
        # and increasing per peer.
        if c.peer >= 0 and st.send_done[c.peer] < c.wr_id:
            st.send_done[c.peer] = c.wr_id
        return 0
    if c.kind == NC_FLUSH:
        st.flush_done += 1
    return 0


def _net_poll(mut st: IbState) -> Int:
    """Up to COMP_BATCH completions from whichever transport is open."""
    if st.net == NET_VERBS:
        return vrb_poll(_vn(st)[], st.comps, COMP_BATCH)
    return fab_poll(_fn(st)[], st.comps, COMP_BATCH)


def _post_flush(mut st: IbState, seq: Int, flush_addr: Int) -> Int:
    if st.net == NET_VERBS:
        return vrb_post_flush(_vn(st)[], flush_addr, FLUSH_BYTES, seq)
    # libfabric reads this rank's own region through its own address vector
    # entry, and (unless the provider uses virtual addressing) by offset.
    return fab_post_flush(_fn(st)[], flush_addr - st.region, FLUSH_BYTES, seq)


comptime BLOCK_CREDIT = 0
comptime BLOCK_ARRIVE = 1
comptime BLOCK_SENDS = 2


def _sample_block(mut st: IbState, reason: Int):
    """Attribute the time since the previous engine step to what is blocking.

    Time-weighted rather than counted: "122 credit stalls" says flow control
    held something back 122 times, not whether that cost a microsecond or a
    tenth of a second, and the two look identical in a counter. Sampled at
    the top of each step and charged to whatever the head exchange was
    waiting for -- the engine spins, so the samples are dense.
    """
    var now = perf_counter_ns()
    var dt = now - st.t_last_sample_ns
    st.t_last_sample_ns = now
    # A first sample, or one across an idle gap where the engine slept, says
    # nothing about a wait; only charge plausible spin intervals.
    if dt <= 0 or dt > 1_000_000:
        return
    if reason == BLOCK_CREDIT:
        st.t_blocked_credit_ns += dt
    elif reason == BLOCK_ARRIVE:
        st.t_blocked_arrive_ns += dt
    else:
        st.t_blocked_sends_ns += dt


def _advance(mut st: IbState) -> Bool:
    """Retire every exchange that is complete, in sequence order.

    An exchange is complete when all `nrecv` peers' messages for it have
    arrived, its own send to EVERY peer has completed (the NIC is done
    reading the buffer the consumer kernel and the next reduce-scatter will
    overwrite) and its GPUDirect flush read has come back.
    """
    var moved = False
    while True:
        if st.flush_seq != 0:
            if st.flush_done <= 0:
                break
            st.flush_done -= 1
            st.t_flush_ns += perf_counter_ns() - st.flush_t0
            st.done_seq = st.flush_seq
            st.flush_seq = 0
            _retire(st, st.done_seq)
            moved = True
            continue
        if st.done_seq >= st.posted_seq:
            break
        var e = st.done_seq + 1
        ref w = _work(st, e)[]
        var idx = e % st.nslots
        if st.tally[idx] < w.nrecv:
            _sample_block(st, BLOCK_ARRIVE)
            break
        if not _sends_done(st, w):
            _sample_block(st, BLOCK_SENDS)
            break
        st.tally[idx] -= w.nrecv
        if w.nrecv > 0 and w.flush_addr != 0 and st.do_flush:
            # Seeing the arrivals does NOT mean the payload is visible in GPU
            # memory: the completion lands in host memory and the payload in
            # the GPU's BAR. A read of the destination flushes the writes
            # ahead of it -- NCCL's gpuFlush QP,
            # nccl:src/transport/net_ib/p2p.cc:589-602.
            if _post_flush(st, e, w.flush_addr) != 0:
                _store_atomic_i(_err_ptr(st), 4)
                return moved
            st.flush_seq = e
            st.flush_t0 = perf_counter_ns()
            moved = True
            continue
        st.done_seq = e
        _retire(st, e)
        moved = True
    return moved


def _retire(mut st: IbState, seq: Int):
    ref w = _work(st, seq)[]
    st.t_wait_ns += perf_counter_ns() - w.t0
    st.n_exchanges += 1
    _store_atomic_i(_status_ptr(w), 1)


def ib_drive(mut st: IbState) -> Bool:
    """One non-blocking step of the progress engine; True if anything moved.

    Order matters: credits go out BEFORE this step's own flow-control check,
    or two ranks that arrive at the same exchange together would each wait
    for a credit the other is holding back.
    """
    if _load_atomic_i(_err_ptr(st)) != 0:
        _release_on_error(st)
        return False
    var moved = False
    # The fused kernel publishes "my consumer for e has run" itself, so a
    # credit can be due with nothing of our own left to post. Send it anyway:
    # a peer blocked on `_can_post` is waiting for exactly this, and riding
    # the next request out (the only way the split path had) would make that
    # wait last until this rank's host issued another collective.
    if _send_credits(st, st.credit_device):
        moved = True
    if st.posted_seq < st.request_seq:
        var e = st.posted_seq + 1
        ref w = _work(st, e)[]
        if _send_credits(st, w.credit_upto):
            moved = True
        if _can_post(st, e):
            var t0 = perf_counter_ns()
            if not _post_data(st, w):
                _release_on_error(st)
                return False
            w.t0 = t0
            st.posted_seq = e
            st.t_post_ns += perf_counter_ns() - t0
            moved = True
        else:
            _sample_block(st, BLOCK_CREDIT)
            if st.stall_seq != e:
                # Counted once per exchange, not once per spin: a nonzero
                # number in the trace means flow control, not the network,
                # held a chunk back, which is the knob INBOX_SLOTS turns.
                st.stall_seq = e
                st.n_credit_stalls += 1
    var n = _net_poll(st)
    for i in range(n):
        if _consume_wc(st, _comp(st, i)[]) < 0:
            _release_on_error(st)
            return False
        moved = True
    if _advance(st):
        moved = True
    if moved:
        st.last_progress_ns = perf_counter_ns()
        st.t_last_sample_ns = st.last_progress_ns
        st.watchdog_said = 0
    elif st.request_seq > st.done_seq:
        # Nothing outstanding can move and nothing has moved for a whole
        # timeout: a peer is gone, or a credit was lost.
        if perf_counter_ns() - st.last_progress_ns > st.timeout_ns:
            # Say where the engine got to before giving up. This costs one
            # print on a path that ends the communicator anyway, and it is
            # the only chance to see the state: `ib_report` runs at teardown,
            # which a rank that dies on `ncclCommGetAsyncError` never reaches.
            # The reader wants to know WHICH side stopped -- an engine with
            # everything posted and nothing arrived is waiting for a peer's
            # GPU, one short of credits is waiting for a peer's consumer.
            print(
                "mojoccl: inter-node engine gave up after",
                st.timeout_ns // 1_000_000_000,
                "s with no progress, at",
                _ring_state(st, st.request_seq),
                "| blocked ms: credit",
                Float64(st.t_blocked_credit_ns) / 1.0e6,
                "arrival",
                Float64(st.t_blocked_arrive_ns) / 1.0e6,
                "own sends",
                Float64(st.t_blocked_sends_ns) / 1.0e6,
            )
            _store_atomic_i(_err_ptr(st), 3)
            _release_on_error(st)
    elif st.peer_seq_seen > st.request_seq:
        # THE SILENT SIDE OF A STALL. Nothing is outstanding here -- every
        # exchange this rank's GPU asked for is retired -- yet a peer has
        # already sent data for a LATER exchange, so the peer's GPU got
        # somewhere mine has not. Without this the branch above never arms
        # (it needs `request_seq > done_seq`), so the node that is actually
        # stuck says nothing and only its peers time out and print, which is
        # the wrong half of the picture.
        #
        # DIAGNOSTIC ONLY: it prints once per stall episode and never touches
        # the error word. A rank whose host legitimately spends a minute
        # between collectives is behind its peers for a good reason, and
        # failing it here would turn a slow run into a broken one.
        if (
            st.watchdog_said == 0
            and perf_counter_ns() - st.last_progress_ns > st.timeout_ns
        ):
            st.watchdog_said = 1
            print(
                "mojoccl: inter-node engine idle for",
                st.timeout_ns // 1_000_000_000,
                "s while peers ran ahead -- THIS rank's GPU has not released",
                _ring_state(st, st.request_seq + 1),
                (
                    "| the stream is stuck in a kernel before this exchange's"
                    " request (an intra-node barrier, a wait for an earlier"
                    " exchange, or work the host has not enqueued yet)"
                ),
            )
    return moved


def _ib_progress(user: OpaquePointer[MutAnyOrigin]) abi("C"):
    """Internal `cuLaunchHostFunc` exchange implementation.

    Runs on a driver-owned thread with the stream stalled behind it, so it
    must never call the CUDA/HIP driver, and it cannot pipeline: it drives
    the engine until its own exchange is done. Kept as a fallback, and as
    the thing the proxy thread is measured against: on this cluster the
    driver's stop-the-stream / wake-a-thread / restart round trip costs
    about 480 us per exchange (job 234035: a 1 MiB two-node allreduce took
    496 us against 24 us on one node, and the transfer in it is 3 us).
    """
    ref w = Pointer[IbWork, MutAnyOrigin](unsafe_from_address=Int(user))[]
    ref st = _st(w.state)[]
    _drive_until(st, w.seq)


def _drive_until(mut st: IbState, seq: Int):
    """Run the engine until exchange `seq` is retired (or the engine fails).
    The stall deadline inside `ib_drive` is what ends this if a peer never
    answers."""
    if seq > st.request_seq and not _comm_stopped(st):
        st.request_seq = seq
        st.last_progress_ns = perf_counter_ns()
    while st.done_seq < seq:
        if _load_atomic_i(_err_ptr(st)) != 0:
            _release_on_error(st)
            return
        _ = ib_drive(st)


@always_inline
def _mb(st: IbState, off: Int) -> Pointer[UInt64, MutAnyOrigin]:
    return Pointer[UInt64, MutAnyOrigin](unsafe_from_address=st.mailbox + off)


@always_inline
def _comm_stopped(st: IbState) -> Bool:
    """The word `abort_raised` (device/common.mojo) tests, read from the host:
    a cache line here, a PCIe round trip from a kernel (see `_proxy_main`)."""
    if st.status_host == 0:
        return False
    return (
        Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
            Pointer[UInt64, MutAnyOrigin](unsafe_from_address=st.status_host)
        )
        != 0
    )


def ib_set_abort_word(ib: Int, abort_dev: Int, abort_host: Int):
    """Hand the transport the communicator's pinned status page, whose first
    word is the abort word -- once as a kernel addresses it and once as this
    process does (init.mojo owns it: a single-node communicator has one
    too, and there is no IB state there to hold it)."""
    if ib == 0:
        return
    _st(ib)[].abort_dev = abort_dev
    _st(ib)[].status_host = abort_host


def _callback_address() -> Int:
    var f: def(OpaquePointer[MutAnyOrigin]) thin abi("C") -> None = _ib_progress
    return Pointer(to=f).unsafe_bitcast[Int]()[]


# ===-------------------------------------------------------------------=== #
# Setup
# ===-------------------------------------------------------------------=== #


def _ib_timeout_s() -> Float64:
    var s = getenv(MOJOCCL_IB_TIMEOUT_S, String(DEFAULT_IB_TIMEOUT_S))
    try:
        return Float64(s)
    except:
        return DEFAULT_IB_TIMEOUT_S


def _nanosleep_ns(ts_addr: Int, ns: Int):
    """`nanosleep(2)` for `ns` nanoseconds; `struct timespec{tv_sec,tv_nsec}`,
    16 bytes on this ABI, in the caller-owned scratch at `ts_addr` (an
    allocation per call here leaked 16 bytes per idle iteration, ~3 GB/h at
    the 20 us quantum). Best-effort: an interrupted sleep just returns early,
    which only means the next mailbox check happens a bit sooner."""
    var ts = Pointer[Int64, MutAnyOrigin](unsafe_from_address=ts_addr)
    ts[unsafe_offset=0] = 0
    ts[unsafe_offset=1] = Int64(ns)
    _ = external_call["nanosleep", Int32](ts, Int64(0))


def ib_local_info(ib: Int, out_blob: P8) raises:
    """Fill this rank's transport half of the bootstrap blob.

    The two transports need different things on the wire (queue-pair
    numbers, LID and MTU for verbs; an endpoint address, a memory key and
    the addressing mode for libfabric) and only one of them is ever active
    in a job, so they share the fixed-size slot rather than coexisting in
    it. Each layout is documented next to its writer, in `transport/net_ib/connect.mojo` and
    `transport/net_ofi.mojo`.
    """
    ref st = _st(ib)[]
    if st.net == NET_VERBS:
        var nodes = List[Int]()
        for i in range(len(st.peers)):
            nodes.append(st.peers[i].node)
        vrb_local_info(_vn(st)[], out_blob, nodes, st.region)
    else:
        fab_local_info(_fn(st)[], out_blob)


# 24 bytes of header, one queue-pair number per node, one GID: the larger of
# the two layouts, and what every rank's slot in the round-2 all-gather is.
comptime IB_BLOB_BYTES = 24 + 4 * MAX_NODES + 16


def ib_connect(
    ib: Int,
    blobs: P8,
    blob_stride: Int,
    peer_rank_of_node: List[Int],
) raises:
    """Attach every peer from the gathered table.

    `peer_rank_of_node[j]` is the global rank on node j holding this rank's
    local_rank -- the one this rank pairs with. `blobs` is the whole round-2
    table; `blob_stride` its per-rank size. On verbs this drives each queue
    pair to RTS and pre-posts its receives; on libfabric, where an FI_EP_RDM
    endpoint is connectionless, it inserts each peer into the address vector
    and posts the shared receive buffers once at the end.
    """
    ref st = _st(ib)[]
    for i in range(len(st.peers)):
        var j = st.peers[i].node
        var b = P8(
            unsafe_from_address=Int(blobs) + peer_rank_of_node[j] * blob_stride
        )
        if st.net == NET_VERBS:
            st.peers[i].remote_base = vrb_blob_base(b)
            st.peers[i].remote_key = vrb_blob_key(b)
            vrb_connect_peer(_vn(st)[], i, j, st.my_node, b)
        else:
            st.peers[i].remote_base = fab_blob_base(_fn(st)[], b)
            st.peers[i].remote_key = fab_blob_key(b)
            fab_add_peer(_fn(st)[], i, j, b)
    if st.net == NET_VERBS:
        vrb_connect_flush(_vn(st)[])
    else:
        fab_post_recvs(_fn(st)[])


# ===-------------------------------------------------------------------=== #
# Per-collective use
# ===-------------------------------------------------------------------=== #


def ib_npeers(ib: Int) -> Int:
    return len(_st(ib)[].peers)


def ib_error(ib: Int) -> Int:
    ref st = _st(ib)[]
    return _load_atomic_i(_err_ptr(st))


def ib_next_seq(ib: Int) -> Int:
    """Consume one exchange counter. Every rank of the communicator calls
    this the same number of times in the same order, so the slot group an
    exchange lands in, the credit window and the immediate that tags an
    arrival all agree across nodes."""
    ref st = _st(ib)[]
    st.exchanges += 1
    return st.exchanges


def ib_reserve_seqs(ib: Int, n: Int) -> Int:
    """Consume `n` consecutive exchange counters and return the first.

    `ib_next_seq` for a collective that knows up front how many exchanges it
    will make -- the fused multi-node allreduce, which fills all of its work
    items on the host and then launches one kernel that publishes the
    counters in order."""
    ref st = _st(ib)[]
    st.exchanges += n
    return st.exchanges - n + 1


def ib_mailbox_dev(ib: Int) -> StaticTuple[Int, 3]:
    """Device addresses of `(MB_REQUEST, MB_DONE, MB_CONSUMED)`.

    The fused kernel writes the first and the third and spins on the second,
    doing from inside the collective what `proxy_request` / `proxy_wait` did
    as kernels of their own."""
    ref st = _st(ib)[]
    return StaticTuple[Int, 3](
        st.mailbox_dev + MB_REQUEST,
        st.mailbox_dev + MB_DONE,
        st.mailbox_dev + MB_CONSUMED,
    )


def ib_uses_proxy(ib: Int) -> Bool:
    """Whether the progress thread drives the engine. Always true in
    production; only synchronous transport probes disable it."""
    return _st(ib)[].proxy


def ib_timeout_ns(ib: Int) -> Int:
    return _st(ib)[].timeout_ns


def ib_prepare_request(
    ib: Int,
    send_addr: Int,
    send_bytes: Int,
    inbox_base: Int,
    slot_bytes: Int,
    do_send: Bool,
    nrecv: Int,
    flush_addr: Int,
    seq: Int,
    op_kind: Int = OP_UNKNOWN,
    op_chunk: Int = 0,
    op_nchunks: Int = 0,
    op_numel: Int = 0,
    send_node_stride: Int = 0,
) raises:
    """Describe exchange `seq` for the engine without releasing it.

    `ib_enqueue_request` minus the release: the fused kernel publishes the
    counter itself when its reduce-scatter for that chunk has completed, so
    the host only has to have the work item ready before the kernel runs.
    Filling every chunk's item before the launch is what makes the whole
    collective one `cuLaunchKernelEx`.

    The credit carried here is the host's own `consumed_enqueued`, which on
    the fused path never advances (the kernel publishes credits through
    MB_CONSUMED instead); it is still the right value for a communicator that
    mixes fused allreduces with the unfused broadcast/allgather paths.
    """
    ref st = _st(ib)[]
    _fill_work(
        st,
        ib,
        send_addr,
        send_bytes,
        inbox_base,
        slot_bytes,
        do_send,
        nrecv,
        flush_addr,
        seq,
        st.consumed_enqueued,
        True,
        op_kind,
        op_chunk,
        op_nchunks,
        op_numel,
        send_node_stride,
    )


def ib_note_consumed(ib: Int, seq: Int):
    """Record that the consumer kernel for exchange `seq` has just been
    ENQUEUED on the stream.

    The number is carried into the next exchange's work item as
    `credit_upto` and published to the peers when the engine picks that
    exchange up -- at which point the request kernel has run and therefore
    every kernel enqueued before it, this consumer included, has completed.
    Callers enqueue the consumer first and call this second.

    That argument is stream order, so every exchange of one communicator has
    to be enqueued on ONE stream in issue order. The engine's dense sequence
    counter already required that (`_work` derives a ring slot from the
    number); `credit_upto` is the second thing that does.
    """
    ref st = _st(ib)[]
    if seq > st.consumed_enqueued:
        st.consumed_enqueued = seq


def _ring_state(st: IbState, seq: Int) -> String:
    """How far behind the engine is, for the ring-pressure messages.

    Every number but the work item's status is engine-owned and read here
    without synchronisation: this only ever builds a diagnostic string, and a
    counter that is one step stale in an error message costs nothing.
    """
    var s = String("")
    s += "exchange " + String(seq)
    ref w = _work(st, seq)[]
    if w.seq == seq and w.op_kind != OP_UNKNOWN:
        s += " (" + _op_name(w.op_kind)
        s += " chunk " + String(w.op_chunk + 1)
        s += " of " + String(w.op_nchunks)
        s += ", " + String(w.op_numel) + " elements)"
    s += " (ring slot " + String((seq - 1) % WORK_SLOTS) + " of "
    s += String(WORK_SLOTS) + "); engine at request "
    s += String(st.request_seq) + ", posted " + String(st.posted_seq)
    s += ", done " + String(st.done_seq)
    s += "; host is " + String(seq - st.done_seq) + " exchanges ahead"
    s += "; credits sent " + String(st.credit_sent) + ", received"
    for i in range(len(st.credit_recv)):
        s += " " + String(st.credit_recv[i])
    s += "; sends done"
    for i in range(len(st.send_done)):
        s += " " + String(st.send_done[i])
    s += "; stalls " + String(st.n_credit_stalls)
    s += "; peers have sent through " + String(st.peer_seq_seen)
    return s^


def _await_ring_slot(mut st: IbState, seq: Int) raises:
    """Back-pressure: wait until exchange `seq - WORK_SLOTS` has retired.

    The work ring is `WORK_SLOTS` deep and the calling thread fills it
    without ever touching the GPU, so how far ahead of the network it can get
    is bounded by nothing but how fast torch enqueues. One rank of a
    broadcast receives every byte while its node-mates receive sixteen, so on
    a slow-enough fabric the receiver's host reaches slot `seq % WORK_SLOTS`
    while the exchange that used it last is still on the wire. That is
    ordinary back-pressure, not an error: wait for the slot.

    Waiting here cannot deadlock. The engine is driven by the progress thread
    (default) or by an internal stream callback -- neither
    needs this thread, and the stream already holds every kernel the
    outstanding exchanges need. The inline self-test path is the one caller
    that drives the engine itself, and it does not come through here (see
    `ib_submit_now`).

    Bounded by `MOJOCCL_IB_TIMEOUT_S`, the same deadline every other wait in
    this library uses: a slot that never frees is a peer that stopped
    answering, and the message says how far behind the engine got.
    """
    ref w = _work(st, seq)[]
    var ahead = seq - st.done_seq
    if ahead > st.max_ahead:
        st.max_ahead = ahead
    if _load_atomic_i(_status_ptr(w)) != 0:
        return
    st.n_ring_waits += 1
    var deadline = perf_counter_ns() + st.timeout_ns
    while _load_atomic_i(_status_ptr(w)) == 0:
        var err = _load_atomic_i(_err_ptr(st))
        if err != 0:
            raise Error(
                "mojoccl: the inter-node transport failed (error "
                + String(err)
                + ") while the host waited for a work-ring slot at "
                + _ring_state(st, seq)
            )
        if perf_counter_ns() > deadline:
            # No exchange may have reached the engine (request == done),
            # so its own progress deadline need not fire. Publish this host
            # timeout as a transport failure for the C ABI and proxy. Only
            # the proxy releases outstanding work; preserve any earlier error.
            var expected = Int64(0)
            _ = Atomic[Scalar[DType.int64]].compare_exchange[
                success_ordering=Ordering.RELEASE,
                failure_ordering=Ordering.RELAXED,
            ](_err_ptr(st).unsafe_bitcast[Int64](), expected, 3)
            raise Error(
                "mojoccl: waited "
                + String(st.timeout_ns // 1_000_000_000)
                + "s for the inter-node work ring to free a slot and it never"
                " did, at "
                + _ring_state(st, seq)
                + "; MOJOCCL_IB_TRACE=1 for the per-exchange timings"
            )
        _ = external_call["sched_yield", Int32]()
        _nanosleep_ns(st.ts_host, 20_000)


def _fill_work(
    mut st: IbState,
    ib: Int,
    send_addr: Int,
    send_bytes: Int,
    inbox_base: Int,
    slot_bytes: Int,
    do_send: Bool,
    nrecv: Int,
    flush_addr: Int,
    seq: Int,
    credit_upto: Int,
    may_wait: Bool,
    op_kind: Int = OP_UNKNOWN,
    op_chunk: Int = 0,
    op_nchunks: Int = 0,
    op_numel: Int = 0,
    send_node_stride: Int = 0,
) raises:
    ref w = _work(st, seq)[]
    if may_wait:
        _await_ring_slot(st, seq)
    elif _load_atomic_i(_status_ptr(w)) == 0:
        # `ib_submit_now`: the caller is the only thing driving the engine, so
        # blocking here would deadlock rather than back off.
        raise Error(
            "mojoccl: the inter-node work ring wrapped with an exchange still"
            " in flight at "
            + _ring_state(st, seq)
        )
    w.state = ib
    w.send_addr = send_addr
    w.send_bytes = send_bytes
    w.send_node_stride = send_node_stride
    w.inbox_base = inbox_base
    w.slot_bytes = slot_bytes
    w.do_send = 1 if do_send else 0
    w.nrecv = nrecv
    w.flush_addr = flush_addr
    w.seq = seq
    w.credit_upto = credit_upto
    w.sent = 0
    w.t0 = 0
    w.op_kind = op_kind
    w.op_chunk = op_chunk
    w.op_nchunks = op_nchunks
    w.op_numel = op_numel
    _store_atomic_i(_status_ptr(w), 0)


def ib_enqueue_request(
    ib: Int,
    driver: OwnedDLHandle,
    ctx: DeviceContext,
    stream: DeviceStream,
    raw_stream: Int,
    send_addr: Int,
    send_bytes: Int,
    inbox_base: Int,
    slot_bytes: Int,
    do_send: Bool,
    nrecv: Int,
    flush_addr: Int,
    seq: Int,
    op_kind: Int = OP_UNKNOWN,
    op_chunk: Int = 0,
    op_nchunks: Int = 0,
    op_numel: Int = 0,
    send_node_stride: Int = 0,
) raises:
    """Release exchange `seq` to the network, at this point in stream order.

    Enqueued right after the kernel that produced its payload. With the
    proxy thread (default) it is a one-thread kernel storing `seq` into the
    pinned mailbox -- nothing else, no status check: `_proxy_main` holds the
    guard against releasing a shard nobody produced, and says there why that
    is the stronger place for it -- and the stream runs on: several exchanges
    may be in flight, and `ib_enqueue_wait` is what eventually stops the
    stream.
    The internal non-proxy implementation uses a `cuLaunchHostFunc` that
    runs the whole exchange inline -- correct with the same schedule, but
    with no overlap and several hundred microseconds of driver latency per
    exchange.
    """
    ref st = _st(ib)[]
    _fill_work(
        st,
        ib,
        send_addr,
        send_bytes,
        inbox_base,
        slot_bytes,
        do_send,
        nrecv,
        flush_addr,
        seq,
        st.consumed_enqueued,
        True,
        op_kind,
        op_chunk,
        op_nchunks,
        op_numel,
        send_node_stride,
    )
    if st.proxy:
        proxy_request(ctx, stream, st.mailbox_dev + MB_REQUEST, seq)
        return
    launch_host_func(
        driver,
        raw_stream,
        _callback_address(),
        st.works + ((seq - 1) % WORK_SLOTS) * size_of[IbWork](),
    )


def ib_enqueue_wait(
    ib: Int, ctx: DeviceContext, stream: DeviceStream, seq: Int
) raises:
    """Hold the stream until exchange `seq` has been retired, so the kernel
    enqueued next may read the inbox. A no-op on the internal callback
    path, where `ib_enqueue_request`'s callback already waited."""
    ref st = _st(ib)[]
    if not st.proxy:
        return
    proxy_wait(
        ctx,
        stream,
        st.mailbox_dev + MB_DONE,
        st.error_word,
        st.abort_dev,
        seq,
        st.timeout_ns,
    )


def ib_submit_now(
    ib: Int,
    send_addr: Int,
    send_bytes: Int,
    inbox_base: Int,
    slot_bytes: Int,
    do_send: Bool,
    nrecv: Int,
    flush_addr: Int,
    seq: Int,
    credit_upto: Int,
) raises:
    """Hand one exchange to the engine from the calling thread, without
    waiting for it.

    The GPU-free self-tests use this (with `ib_wait_now`) to keep several
    exchanges in flight and exercise the credit protocol past the slot
    count, which is the case a leaked credit turns into a hang. Never
    correct inside a collective: there the exchange's position in stream
    order is the whole ordering argument.
    """
    ref st = _st(ib)[]
    _fill_work(
        st,
        ib,
        send_addr,
        send_bytes,
        inbox_base,
        slot_bytes,
        do_send,
        nrecv,
        flush_addr,
        seq,
        credit_upto,
        False,
    )
    if seq > st.request_seq:
        st.request_seq = seq
        st.last_progress_ns = perf_counter_ns()


def ib_wait_now(ib: Int, seq: Int) raises:
    """Drive the engine on the calling thread until exchange `seq` is done."""
    ref st = _st(ib)[]
    _drive_until(st, seq)
    if _load_atomic_i(_err_ptr(st)) != 0:
        raise Error(
            "mojoccl: inline exchange failed, ib error "
            + String(_load_atomic_i(_err_ptr(st)))
        )


def ib_exchange_now(
    ib: Int,
    send_addr: Int,
    send_bytes: Int,
    inbox_base: Int,
    slot_bytes: Int,
    do_send: Bool,
    nrecv: Int,
    flush_addr: Int,
    seq: Int,
    credit_upto: Int,
) raises:
    """One exchange, submitted and waited for on the calling thread.

    The bring-up self-test uses it: the transport can then be exercised on
    a host with InfiniBand but no GPU (registered host memory, no stream to
    hang kernels on), which is where the bootstrap/QP/immediate wiring is
    cheapest to debug. Pass `_synchronous_test=True` to `ib_setup`, since the proxy
    mailbox needs a driver that can pin host memory.
    """
    ib_submit_now(
        ib,
        send_addr,
        send_bytes,
        inbox_base,
        slot_bytes,
        do_send,
        nrecv,
        flush_addr,
        seq,
        credit_upto,
    )
    ib_wait_now(ib, seq)


def ib_report(ib: Int):
    ref st = _st(ib)[]
    if not st.trace or st.n_exchanges == 0:
        return
    print(
        "mojoccl net:",
        "verbs" if st.net == NET_VERBS else "fabric",
        st.netdev,
        "peers",
        len(st.peers),
        "slots",
        st.nslots,
        "exchanges",
        st.n_exchanges,
        "credit stalls",
        st.n_credit_stalls,
        "| host ran up to",
        st.max_ahead,
        "exchanges ahead, waited for a ring slot",
        st.n_ring_waits,
        "times",
        "| blocked ms: credit",
        Float64(st.t_blocked_credit_ns) / 1.0e6,
        "arrival",
        Float64(st.t_blocked_arrive_ns) / 1.0e6,
        "own sends",
        Float64(st.t_blocked_sends_ns) / 1.0e6,
        "| mean us post",
        Float64(st.t_post_ns) / Float64(st.n_exchanges) / 1000.0,
        "in flight",
        Float64(st.t_wait_ns) / Float64(st.n_exchanges) / 1000.0,
        "flush",
        Float64(st.t_flush_ns) / Float64(st.n_exchanges) / 1000.0,
    )
    if st.net == NET_FABRIC:
        print("mojoccl net:", fab_describe(_fn(st)[]))


def _teardown_ib_resources(mut st: IbState):
    """Release every transport and host resource `ib_setup` may have created.

    Shared by `ib_teardown` (a live communicator) and `ib_setup`'s own
    failure path (a later step raised after an earlier one already
    succeeded) -- both leave `st` in the same "some fields non-zero, some
    still their zero default" shape, and every field here is zero-guarded
    for exactly that reason.
    """
    if st.mailbox != 0:
        # Pinned, device-mapped host memory: a scarce OS resource, unlike the
        # few hundred bytes of plain heap this struct also holds. Safe here
        # and only here -- the progress thread is joined (or, from
        # `ib_setup`'s failure path, never started) and, for a live
        # communicator, the caller synchronized the stream the spin kernels
        # were on. `open_driver` re-opens an already-loaded library, so it
        # costs a refcount.
        try:
            free_host(open_driver(), st.mailbox)
        except e:
            # Best effort on a teardown path; the mailbox is dropped either way.
            print("mojoccl: freeing the proxy mailbox failed (ignored):", e)
        st.mailbox = 0
        st.mailbox_dev = 0
    if st.vrb != 0:
        vrb_teardown(_vn(st)[])
        st.vrb = 0
    if st.fab != 0:
        fab_teardown(_fn(st)[])
        st.fab = 0
