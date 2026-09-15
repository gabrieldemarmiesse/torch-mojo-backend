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
#   * `ibverbs.mojo` -- InfiniBand, an RC queue pair per peer, one
#     RDMA_WRITE_WITH_IMM per shard. This is the path the two-node H100
#     numbers were measured on and it is unchanged.
#   * `libfabric.mojo` -- HPE Slingshot through the `cxi` provider (Adastra
#     nodes have four /dev/cxi NICs and no InfiniBand at all). One
#     connectionless RDM endpoint; since cxi implements no
#     write-with-immediate, a shard is an RMA write followed by a FENCED
#     zero-length message carrying the immediate as remote CQ data. See that
#     file's header.
#
# Which one is used is decided at `ib_setup` time by `MOJOCCL_NET`, or by
# what the machine actually has. The names in this file kept their `ib_`
# prefix: they are the transport's API to `mojoccl.mojo` and renaming them
# would have churned every call site for nothing.
#
# Ordering. The GPU cannot post verbs and the NIC cannot wait on a kernel,
# so a CPU thread stands between them and the stream is what sequences it:
#
#     [reduce_scatter_stage]  my shard is final in my stage_out
#     [proxy_request]         one thread releases the exchange counter into
#                             a pinned mailbox
#       ~ progress thread ~   post one RDMA_WRITE_WITH_IMM per peer, poll
#                             the CQ until every peer's shard has landed in
#                             my inbox, flush (below), release `done`
#     [proxy_wait]            one thread spins until `done` catches up
#     [inbox_add]             shard += the peers' shards
#     [allgather_finish]      spread the global sum
#
# The obvious alternative -- one `cuLaunchHostFunc` doing all of it inline
# -- was written first, shipped, and measured: it costs about 480 us per
# exchange on this cluster, because the driver has to stop the stream, wake
# a thread and restart it, and that delay does NOT cancel between the two
# nodes (each side ends up measuring the other's dispatch jitter; both
# reported ~390 us of "waiting for the peer" on a transfer worth 3 us). It
# survives behind `MOJOCCL_IB_PROXY=0`: same exchange body, one fewer core
# burned, several hundred microseconds slower.
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
# (mojoccl.mojo `_do_allreduce`) issues the reduce-scatter of later chunks
# before the add of earlier ones and breaks it. The credit is the
# replacement proof, and it is a proof rather than a timing margin.
#
# Two things survive from the old argument unchanged. Every exchange must
# still be all-to-all, because an arrival tally of `nrecv` is what completes
# one -- a rank with nothing to contribute sends EMPTY_SHARD_BYTES
# (mojoccl.mojo). And every slot group must be a FIXED byte range for every
# exchange alike, never sized from the message in flight (`_inbox_base` in
# mojoccl.mojo carries the case that broke). The immediate carries the
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
# the previous generation (see ncclCommInitRank).

from std.ffi import OwnedDLHandle, external_call
from std.memory.alloc import unsafe_alloc
from std.sys import size_of
from std.os import getenv
from std.time import perf_counter_ns, sleep
from std.utils import StaticTuple
from max.gpu.host import DeviceContext, DeviceStream

from std.atomic import Atomic, Ordering

from driver import (
    alloc_host,
    device_pci_bus_id,
    free_host,
    host_device_ptr,
    launch_host_func,
    open_driver,
)
from internode_kernels import proxy_request, proxy_wait
from ibverbs import (
    VerbsNet,
    verbs_available,
    vrb_blob_base,
    vrb_blob_key,
    vrb_connect_flush,
    vrb_connect_peer,
    vrb_local_info,
    vrb_poll,
    vrb_post_flush,
    vrb_post_imm,
    vrb_post_payload,
    vrb_setup,
    vrb_teardown,
)
from libfabric import (
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
    fab_setup,
    fab_teardown,
    fabric_available,
)
from netutil import (
    MAX_NODES,
    NC_FLUSH,
    NC_RECV,
    NC_SEND,
    NetCompletion,
    P8,
    alloc_bytes,
)

# Which transport `ib_setup` opened. Runtime rather than comptime: one
# build of libmojoccl.so has to run on an InfiniBand cluster and on a
# Slingshot one, and `MOJOCCL_NET` has to be able to force either.
comptime NET_VERBS = 0
comptime NET_FABRIC = 1

comptime WORK_SLOTS = 512
# Completions pulled out of the transport per engine step.
comptime COMP_BATCH = 16
comptime DEFAULT_IB_TIMEOUT_S: Float64 = 60.0
# ncclCommAbort's bound on waiting for the progress thread: long enough for
# it to notice MB_STOP (at most one idle-backoff quantum plus one engine
# step -- `ib_drive` never blocks), short enough that abort's documented
# "don't wait" contract still holds even if the thread is wedged somewhere
# this bound does not anticipate.
comptime IB_ABORT_JOIN_TIMEOUT_S: Float64 = 2.0
# Default idle-wait quantum for the progress thread between exchanges (see
# `_proxy_main`). An exchange takes ~280 us at the DDP bucket, so a few tens
# of us of wake-up latency between exchanges is cheap; measured end to end
# (job 234072, 2x8 H100) an unconditional hot spin here cost nanoGPT DDP
# ~20% of its steady-state tok/s against real NCCL, competing for a core/SMT
# sibling with the ~760-aten-op-per-step host dispatch of the training loop.
comptime DEFAULT_IB_PROXY_IDLE_US: Int = 20
# `cpu_set_t` size for `sched_getaffinity`/`pthread_setaffinity_np` on this
# ABI: 128 bytes (1024 bits), shared by the default-pin CPU scan and the
# explicit-CPU pin below.
comptime CPU_SET_BYTES = 128
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
"""Which collective an exchange belongs to, carried in its work item for the
stall messages only. `mojoccl.mojo` passes it to `ib_enqueue_request`."""


def _op_name(kind: Int) -> String:
    if kind == OP_ALLREDUCE:
        return String("allreduce")
    if kind == OP_BROADCAST:
        return String("broadcast")
    if kind == OP_ALLGATHER:
        return String("allgather")
    return String("?")


# The proxy mailbox: 64-bit words the GPU and the progress thread pass
# exchange counters through, plus a stop word the host sets at teardown. A
# cache line apart so the GPU's writes to REQUEST never invalidate the line
# the CPU is writing DONE into.
#
# REQUEST and CONSUMED are written by the device, DONE by the thread, STOP by
# the host. CONSUMED exists for the fused kernel (internode_fused.mojo): it is
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
    # Diagnostics only: when a stall is reported, "the GPU has not released
    # exchange 78" is a lot more useful as "exchange 78, allreduce chunk 3 of
    # 14, 1703936 elements", and the two ends of a stall can be compared.
    var op_kind: Int
    var op_chunk: Int
    var op_nchunks: Int
    var op_numel: Int

    def __init__(out self):
        self.state = 0
        self.send_addr = 0
        self.send_bytes = 0
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
        self.do_flush = getenv("MOJOCCL_FABRIC_FLUSH", "1") != "0"
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
        self.proxy = getenv("MOJOCCL_IB_PROXY", "1") != "0"
        self.thread_id = 0
        self.trace = getenv("MOJOCCL_IB_TRACE", "0") != "0"
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
# `MOJOCCL_IB_PROXY=0` callback thread) inside `ib_drive` and read by
# the calling thread -- `ib_error` from the torch-facing calling thread,
# `ib_enqueue`'s ring-reuse check from the same -- with no other
# synchronization between the two. Every access goes through these two
# helpers rather than a plain field read/write so that relationship is a
# real release/acquire pair, not two threads racing a plain `Int`.
@always_inline
def _load_atomic_i(p: Pointer[Int, MutAnyOrigin]) -> Int:
    return Int(
        Atomic[DType.int64].load[ordering=Ordering.ACQUIRE](
            p.unsafe_bitcast[Int64]()
        )
    )


@always_inline
def _store_atomic_i(p: Pointer[Int, MutAnyOrigin], v: Int):
    Atomic[DType.int64].store[ordering=Ordering.RELEASE](
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
# the progress thread, the `MOJOCCL_IB_PROXY=0` stream callback and the
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
                w.send_addr,
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
                w.send_addr,
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
    """`cuLaunchHostFunc` entry point -- the MOJOCCL_IB_PROXY=0 path.

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
    """The word `collectives_kernels.abort_raised` tests, read from the host:
    a cache line here, a PCIe round trip from a kernel (see `_proxy_main`)."""
    if st.status_host == 0:
        return False
    return (
        Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
            Pointer[UInt64, MutAnyOrigin](unsafe_from_address=st.status_host)
        )
        != 0
    )


def _proxy_main(arg: OpaquePointer[MutAnyOrigin]) abi("C"):
    """The progress thread: `ib_drive` in a loop, with the mailbox on both
    ends.

    `MB_REQUEST` is the highest exchange the stream has released (a
    one-thread kernel's release store); `MB_DONE` is the highest one
    retired, which the matching spin kernel waits for. Several exchanges may
    be in flight between the two.

    Hard-spins like NCCL's proxy only WHILE SOMETHING IS OUTSTANDING --
    otherwise it backs off (`sched_yield` once, then a short `nanosleep`)
    instead of burning a full core on a mailbox word that is not going to
    change for a while. A training step's host-side dispatch (hundreds of
    aten launches on the Python main thread) shares this core's SMT sibling,
    and measured end to end (job 234072, 2x8 H100) an unconditional hot spin
    here cost nanoGPT DDP ~20% of its steady-state tok/s against real NCCL.
    `MOJOCCL_IB_PROXY_IDLE_US` tunes the backoff quantum.

    THE STOPPED-COMMUNICATOR GUARD LIVES HERE, not in the request kernel.
    Once a kernel on this communicator has given up (`publish_fault` raises
    the abort word) the reduce-scatter behind `req` may never have run, and
    posting it would ship whatever the arena holds as data. Refusing to
    advance `request_seq` sends nothing: the peer's engine times out with a
    message instead of a wrong number. `ncclCommAbort` is the same case.

    Checking here is STRICTLY STRONGER than in the kernel: the request kernel
    publishes `seq` with a release store into pinned memory and the load
    above acquires it, so everything the stream did before that kernel is
    visible by the time `req` is read, including any fault latched before
    the payload existed. And it is a cache line here, against a PCIe round
    trip per exchange on the device, on the critical path between the
    reduce-scatter and the network post.
    """
    ref st = _st(Int(arg))[]
    var idle_ns = _proxy_idle_ns()
    var published = 0
    st.last_progress_ns = perf_counter_ns()
    while True:
        if (
            Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
                _mb(st, MB_STOP)
            )
            != 0
        ):
            return
        var consumed = Int(
            Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
                _mb(st, MB_CONSUMED)
            )
        )
        if consumed > st.credit_device:
            st.credit_device = consumed
        var req = Int(
            Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
                _mb(st, MB_REQUEST)
            )
        )
        if req > st.request_seq and not _comm_stopped(st):
            st.request_seq = req
            st.last_progress_ns = perf_counter_ns()
        var moved = ib_drive(st)
        if st.done_seq > published:
            # Published even on failure: the spin kernels must be released or
            # the stream hangs past the point where the error can be
            # reported.
            published = st.done_seq
            Atomic[DType.uint64].store[ordering=Ordering.RELEASE](
                _mb(st, MB_DONE), UInt64(published)
            )
        if moved or st.done_seq < st.request_seq:
            continue
        _ = external_call["sched_yield", Int32]()
        if Atomic[DType.uint64].load[ordering=Ordering.ACQUIRE](
            _mb(st, MB_REQUEST)
        ) <= UInt64(st.request_seq):
            _nanosleep_ns(st.ts, idle_ns)


def _proxy_address() -> Int:
    var f: def(OpaquePointer[MutAnyOrigin]) thin abi("C") -> None = _proxy_main
    return Pointer(to=f).unsafe_bitcast[Int]()[]


def _set_thread_affinity(tid: Int, cpu: Int) raises:
    """`pthread_setaffinity_np` to a single CPU. `cpu_set_t` is a 128-byte
    (1024-bit) bitmask on this ABI; only the one bit for `cpu` is set."""
    var byte_idx = cpu // 8
    if cpu < 0 or byte_idx >= CPU_SET_BYTES:
        raise Error(
            "mojoccl: MOJOCCL_IB_PROXY_CPU=" + String(cpu) + " out of range"
        )
    var mask = alloc_bytes(CPU_SET_BYTES)
    mask[unsafe_offset=byte_idx] = UInt8(1) << UInt8(cpu % 8)
    var rc = external_call["pthread_setaffinity_np", Int32](
        Int64(tid), UInt64(CPU_SET_BYTES), mask
    )
    if rc != 0:
        raise Error("mojoccl: pthread_setaffinity_np failed, rc=" + String(rc))


def _cpu_in_mask(mask: P8, cpu: Int) -> Bool:
    if cpu < 0 or cpu // 8 >= CPU_SET_BYTES:
        return False
    return (mask[unsafe_offset=cpu // 8] >> UInt8(cpu % 8)) & 1 != 0


def _mask_cpus_desc(mask: P8) -> List[Int]:
    """CPU ids set in the affinity `mask`, highest first."""
    var out = List[Int]()
    for cpu in range(CPU_SET_BYTES * 8 - 1, -1, -1):
        if _cpu_in_mask(mask, cpu):
            out.append(cpu)
    return out^


def _smt_sibling_free(cpu: Int, reserved: List[Int]) -> Bool:
    """Best-effort: True unless `cpu`'s hyperthread sibling is itself one of
    `reserved` -- i.e. would put two ranks' progress threads on one physical
    core's shared execution units. "Free" only means "not another rank's
    proxy pin": Python thread placement isn't ours to observe, so that's the
    only sibling contention this can detect. Reads
    `topology/thread_siblings_list` (comma-separated CPU ids/ranges) once; a
    missing file (no HT, non-Linux sysfs layout, permission) reads as free
    rather than blocking the pick.
    """
    var path = (
        "/sys/devices/system/cpu/cpu"
        + String(cpu)
        + "/topology/thread_siblings_list"
    )
    var text: String
    try:
        with open(path, "r") as f:
            text = String(f.read().strip())
    except:
        return True
    for part in text.split(","):
        var s = String(part)
        if s.byte_length() == 0:
            continue
        var pieces = s.split("-")
        var lo_s = String(pieces[0])
        var hi_s = lo_s if len(pieces) < 2 else String(pieces[1])
        try:
            var lo = Int(lo_s)
            var hi = Int(hi_s)
            for sib in range(lo, hi + 1):
                if sib != cpu:
                    for r in reserved:
                        if r == sib:
                            return False
        except:
            continue
    return True


def _default_proxy_cpu(local_rank: Int, local_world: Int) -> Int:
    """Default pin when `MOJOCCL_IB_PROXY_CPU` is unset, or -1 for "don't
    pin".

    torchrun gives every rank of a node the same affinity mask, so taking
    that mask's CPUs in descending order and indexing by `local_rank` spreads
    the `local_world` progress threads over the top `local_world` CPUs, out
    of the way of Python's dispatch threads (unpinned, so scheduled wherever
    the OS puts them across the whole mask). Needs the mask to hold at least
    `2 * local_world` CPUs, so pinning never claims a CPU Python is likely to
    need; a small mask (few cores, or an explicit `--cpus-per-task` slice)
    leaves the thread unpinned rather than fighting over a scarce core.

    Among those candidates, prefer ones whose SMT sibling is not itself
    another rank's pin (`_smt_sibling_free`): reorder sibling-free CPUs to
    the front, then reuse the same descending/by-rank rule. Every rank
    derives this reordering from the same mask and `local_world` alone, so
    it needs no coordination and never assigns two ranks the same CPU.
    """
    if local_world < 1 or local_rank < 0 or local_rank >= local_world:
        return -1
    var mask = alloc_bytes(CPU_SET_BYTES)
    var rc = external_call["sched_getaffinity", Int32](
        Int32(0), UInt64(CPU_SET_BYTES), mask
    )
    if rc != 0:
        return -1
    var desc = _mask_cpus_desc(mask)
    if len(desc) < 2 * local_world:
        return -1
    var naive = desc[local_rank]
    var reserved = List[Int]()
    for r in range(local_world):
        reserved.append(desc[r])
    if _smt_sibling_free(naive, reserved):
        return naive
    var ordered = List[Int]()
    var dirty = List[Int]()
    for cpu in desc:
        if _smt_sibling_free(cpu, reserved):
            ordered.append(cpu)
        else:
            dirty.append(cpu)
    for cpu in dirty:
        ordered.append(cpu)
    return ordered[local_rank]


def _start_proxy(ib: Int, local_rank: Int, local_world: Int) raises:
    ref st = _st(ib)[]
    var tid = unsafe_alloc[Int64](1)
    tid[unsafe_offset=0] = 0
    var rc = external_call["pthread_create", Int32](
        tid, Int64(0), _proxy_address(), ib
    )
    if rc != 0:
        raise Error("mojoccl: pthread_create failed, rc=" + String(rc))
    st.thread_id = Int(tid[unsafe_offset=0])
    # A best-effort placement hint, not load-bearing for correctness, so a
    # bad CPU index or a failed syscall only prints rather than failing
    # communicator init. MOJOCCL_IB_PROXY_CPU=none opts out of the default
    # policy below (unpinned, like every release before this one); any other
    # value overrides it with an exact CPU.
    var cpu_s = getenv("MOJOCCL_IB_PROXY_CPU", "")
    if cpu_s == "none":
        return
    var cpu: Int
    if cpu_s.byte_length() > 0:
        try:
            cpu = Int(cpu_s)
        except e:
            print("mojoccl: MOJOCCL_IB_PROXY_CPU pinning failed:", e)
            return
    else:
        cpu = _default_proxy_cpu(local_rank, local_world)
        if cpu < 0:
            if st.trace:
                print(
                    (
                        "mojoccl: progress thread left unpinned (affinity mask"
                        " too small for"
                    ),
                    local_world,
                    "local ranks)",
                )
            return
    try:
        _set_thread_affinity(st.thread_id, cpu)
        if st.trace:
            print("mojoccl: progress thread pinned to cpu", cpu)
    except e:
        print("mojoccl: MOJOCCL_IB_PROXY_CPU pinning failed:", e)


def _stop_proxy(mut st: IbState):
    if st.thread_id == 0:
        return
    Atomic[DType.uint64].store[ordering=Ordering.RELEASE](_mb(st, MB_STOP), 1)
    _ = external_call["pthread_join", Int32](st.thread_id, Int64(0))
    st.thread_id = 0


def ib_set_abort_word(ib: Int, abort_dev: Int, abort_host: Int):
    """Hand the transport the communicator's pinned status page, whose first
    word is the abort word -- once as a kernel addresses it and once as this
    process does (mojoccl.mojo owns it: a single-node communicator has one
    too, and there is no IB state there to hold it)."""
    if ib == 0:
        return
    _st(ib)[].abort_dev = abort_dev
    _st(ib)[].status_host = abort_host


def ib_signal_abort(ib: Int) -> Bool:
    """`ncclCommAbort`'s hook: raise MB_STOP and reclaim the progress thread
    without ncclCommDestroy's unbounded wait. True once the thread is gone
    (or there never was one), which is the caller's licence to tear the
    transport down: `ib_teardown` joins unconditionally, so releasing the
    queue pairs while this thread is still running would both reintroduce the
    unbounded wait and pull memory out from under it.

    Left unsignaled, the thread spins on the mailbox forever (nothing else
    ever sets MB_STOP for it) and burns one CPU core for the rest of the
    process. Unlike `_stop_proxy`, the join here is bounded
    (`IB_ABORT_JOIN_TIMEOUT_S`) with `pthread_tryjoin_np`, polled rather than
    blocking: abort must not hang because a dead peer left this rank's
    thread waiting for a completion that will never come -- `ib_drive` never
    blocks, so the loop notices MB_STOP within one step; this bound is only
    insurance against the case that doesn't anticipate.
    A thread this gives up on is simply left running; it exits on its own
    once it next checks MB_STOP, and the process exiting reclaims it either
    way.
    """
    if ib == 0:
        return True
    ref st = _st(ib)[]
    if st.thread_id == 0:
        return True
    Atomic[DType.uint64].store[ordering=Ordering.RELEASE](_mb(st, MB_STOP), 1)
    var tid = st.thread_id
    var deadline = perf_counter_ns() + Int(IB_ABORT_JOIN_TIMEOUT_S * 1.0e9)
    var retval = unsafe_alloc[Int64](1)
    while perf_counter_ns() < deadline:
        var rc = external_call["pthread_tryjoin_np", Int32](tid, retval)
        if rc == 0:
            st.thread_id = 0
            return True
        sleep(0.001)
    return False


def _callback_address() -> Int:
    var f: def(OpaquePointer[MutAnyOrigin]) thin abi("C") -> None = _ib_progress
    return Pointer(to=f).unsafe_bitcast[Int]()[]


# ===-------------------------------------------------------------------=== #
# Setup
# ===-------------------------------------------------------------------=== #


def _select_backend() raises -> Int:
    """Which transport to open: `MOJOCCL_NET` wins, else what is present.

    "Present" means the library opens AND has something usable behind it --
    a login node with a Mellanox card and a compute node with four Slingshot
    NICs and no /dev/infiniband both give an unambiguous answer, and a
    machine with neither gets the same error message this library has always
    given. Verbs is tried first because it is the measured path.
    """
    var want = getenv("MOJOCCL_NET", "")
    if want == "verbs":
        return NET_VERBS
    if want == "fabric":
        return NET_FABRIC
    if want.byte_length() > 0:
        raise Error(
            "mojoccl: MOJOCCL_NET must be `verbs` or `fabric`, got " + want
        )
    if verbs_available():
        return NET_VERBS
    if fabric_available():
        return NET_FABRIC
    raise Error(
        "mojoccl: no ACTIVE InfiniBand port found (libibverbs) and no"
        " libfabric provider offering FI_RMA|FI_MSG|FI_HMEM on an FI_EP_RDM"
        " endpoint; a multi-node communicator needs one of the two."
        " MOJOCCL_NET=verbs|fabric forces a choice, MOJOCCL_LIBFABRIC points"
        " at a libfabric.so.1 that is not on the loader path"
    )


def ib_setup(
    driver: OwnedDLHandle,
    ordinal: Int,
    local_rank: Int,
    local_world: Int,
    my_node: Int,
    nnodes: Int,
    region: Int,
    region_bytes: Int,
    nslots: Int,
    credit_off: Int,
) raises -> Int:
    """Open a NIC and register the region on whichever transport this
    machine has.

    Returns the address of a heap `IbState`. The peers cannot be reached
    until their blobs have been gathered, so bring-up is split: this, then
    `ib_local_info` / `ib_connect`.

    `nslots` is how many fixed inbox slot groups the caller carved out of the
    region and therefore how many exchanges may be outstanding; `credit_off`
    is the region offset of the credit landing pad. Both must be identical on
    every rank -- they are part of the wire layout. `local_world` only feeds
    the progress thread's default CPU pin (`_default_proxy_cpu`).
    """
    if nslots < 1 or nslots > PIPE_MAX_SLOTS:
        raise Error(
            "mojoccl: nslots must be in 1.."
            + String(PIPE_MAX_SLOTS)
            + ", got "
            + String(nslots)
        )
    if nnodes > MAX_NODES:
        raise Error(
            "mojoccl: this transport addresses at most "
            + String(MAX_NODES)
            + " nodes, got "
            + String(nnodes)
        )
    # Only a hint for NIC affinity: a driver without the symbol, or a device
    # that will not report one, falls back to round-robin.
    var gpu_bdf: String
    try:
        gpu_bdf = device_pci_bus_id(driver, ordinal)
    except:
        # No PCI hint from this driver: NIC affinity falls back to round-robin.
        gpu_bdf = String("")

    var st = IbState(
        _select_backend(), region, my_node, nnodes, nslots, credit_off
    )
    st.timeout_ns = Int(_ib_timeout_s() * 1.0e9)

    # Every step below can raise after an earlier one already allocated a
    # real transport or host resource -- unwind whatever got that far
    # instead of leaking it.
    try:
        if st.net == NET_VERBS:
            var v = vrb_setup(gpu_bdf, local_rank, nnodes, region, region_bytes)
            st.netdev = String(v.hca)
            var vh = unsafe_alloc[VerbsNet](1)
            vh.unsafe_write(v^)
            st.vrb = Int(vh)
        else:
            var f = fab_setup(
                gpu_bdf, local_rank, my_node, nnodes, region, region_bytes
            )
            st.netdev = f.prov_name + ":" + f.domain_name
            var fh = unsafe_alloc[FabricNet](1)
            fh.unsafe_write(f^)
            st.fab = Int(fh)
        for j in range(nnodes):
            if j == my_node:
                continue
            st.peers.append(IbPeer(j, 0, 0))
            st.credit_recv.append(0)
            st.send_done.append(0)

        if st.proxy:
            st.mailbox = alloc_host(driver, MB_BYTES)
            st.mailbox_dev = host_device_ptr(driver, st.mailbox)
            for i in range(MB_BYTES // 8):
                Pointer[UInt64, MutAnyOrigin](unsafe_from_address=st.mailbox)[
                    unsafe_offset=i
                ] = 0
    except e:
        _teardown_ib_resources(st)
        raise e

    var holder = unsafe_alloc[IbState](1)
    holder.unsafe_write(st^)
    if _st(Int(holder))[].proxy:
        try:
            _start_proxy(Int(holder), local_rank, local_world)
        except e:
            _teardown_ib_resources(_st(Int(holder))[])
            raise e
    return Int(holder)


def _ib_timeout_s() -> Float64:
    var s = getenv("MOJOCCL_IB_TIMEOUT_S", String(DEFAULT_IB_TIMEOUT_S))
    try:
        return Float64(s)
    except:
        return DEFAULT_IB_TIMEOUT_S


def _proxy_idle_ns() -> Int:
    """`MOJOCCL_IB_PROXY_IDLE_US`, read once at thread start (not from inside
    the progress-thread loop -- a `getenv` per idle iteration would defeat
    the point of backing off)."""
    var s = getenv("MOJOCCL_IB_PROXY_IDLE_US", String(DEFAULT_IB_PROXY_IDLE_US))
    var us = DEFAULT_IB_PROXY_IDLE_US
    try:
        var parsed = Int(s)
        if parsed > 0:
            us = parsed
    except:
        # Unparseable value: keep the default quantum.
        us = DEFAULT_IB_PROXY_IDLE_US
    return us * 1000


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
    it. Each layout is documented next to its writer, in `ibverbs.mojo` and
    `libfabric.mojo`.
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
    """Whether the progress thread is driving the engine. False means
    `MOJOCCL_IB_PROXY=0`, where the exchange runs inside a stream callback and
    the fused kernel cannot be used -- there is no point in the stream for the
    callback to run at."""
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
    (default) or, under `MOJOCCL_IB_PROXY=0`, by a stream callback -- neither
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
) raises:
    """Release exchange `seq` to the network, at this point in stream order.

    Enqueued right after the kernel that produced its payload. With the
    proxy thread (default) it is a one-thread kernel storing `seq` into the
    pinned mailbox -- nothing else, no status check: `_proxy_main` holds the
    guard against releasing a shard nobody produced, and says there why that
    is the stronger place for it -- and the stream runs on: several exchanges
    may be in flight, and `ib_enqueue_wait` is what eventually stops the
    stream.
    Without the proxy (`MOJOCCL_IB_PROXY=0`) it is a `cuLaunchHostFunc` that
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
    enqueued next may read the inbox. A no-op on the `MOJOCCL_IB_PROXY=0`
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
    cheapest to debug -- run it with `MOJOCCL_IB_PROXY=0`, since the proxy
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


def ib_teardown(ib: Int):
    if ib == 0:
        return
    ref st = _st(ib)[]
    _stop_proxy(st)
    ib_report(ib)
    _teardown_ib_resources(st)
