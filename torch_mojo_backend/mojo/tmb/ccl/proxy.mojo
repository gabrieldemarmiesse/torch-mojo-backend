# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/proxy.cc
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
# remains an internal callback implementation of the same exchange body.
# Production uses the proxy; host-only tests drive the exchange directly.

from std.atomic import Atomic, Ordering
from std.ffi import external_call, OwnedDLHandle
from std.sys import has_nvidia_gpu_accelerator
from std.time import perf_counter_ns, sleep
from std.memory.alloc import unsafe_alloc

from tmb.ccl.include.plugin.nccl_net import MAX_NODES
from tmb.ccl.misc.cudawrap import alloc_host, device_pci_bus_id, host_device_ptr
from tmb.ccl.misc.utils import P8, alloc_bytes
from tmb.ccl.plugin.net import NET_VERBS, _select_backend
from tmb.ccl.transport.net import (
    BATCH_POLL_NS,
    IbPeer,
    IbState,
    MB_BYTES,
    MB_CONSUMED,
    MB_DONE,
    MB_REQUEST,
    MB_STOP,
    OP_ALLGATHER,
    OP_REDUCE_SCATTER,
    PIPE_MAX_SLOTS,
    _comm_stopped,
    _err_ptr,
    _ib_timeout_s,
    _load_atomic_i,
    _mb,
    _nanosleep_ns,
    _st,
    _teardown_ib_resources,
    _work,
    ib_drive,
    ib_report,
)
from tmb.ccl.transport.net_ib.connect import VerbsNet, vrb_setup
from tmb.ccl.transport.net_ofi import FabricNet, fab_setup


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
    The idle backoff quantum is fixed at 20 us.

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
    # Last exchange of the multi-chunk collective being released, and how
    # long to keep polling for its next chunk instead of sleeping.
    var batch_end = 0
    var batch_deadline = 0
    st.last_progress_ns = perf_counter_ns()
    while True:
        if (
            Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
                _mb(st, MB_STOP)
            )
            != 0
        ):
            return
        var consumed = Int(
            Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
                _mb(st, MB_CONSUMED)
            )
        )
        if consumed > st.credit_device:
            st.credit_device = consumed
        var req = Int(
            Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
                _mb(st, MB_REQUEST)
            )
        )
        if req > st.request_seq and not _comm_stopped(st):
            st.request_seq = req
            st.last_progress_ns = perf_counter_ns()
            comptime if has_nvidia_gpu_accelerator():
                batch_end = 0
                # Read before ib_drive can retire and recycle this ring slot.
                ref work = _work(st, req)[]
                if (
                    work.seq == req
                    and (
                        work.op_kind == OP_REDUCE_SCATTER
                        or work.op_kind == OP_ALLGATHER
                    )
                    and 0 <= work.op_chunk < work.op_nchunks - 1
                ):
                    batch_end = req + work.op_nchunks - 1 - work.op_chunk
                    batch_deadline = st.last_progress_ns + BATCH_POLL_NS
        var moved = ib_drive(st)
        if st.done_seq > published:
            # Published even on failure: the spin kernels must be released or
            # the stream hangs past the point where the error can be
            # reported.
            published = st.done_seq
            Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
                _mb(st, MB_DONE), UInt64(published)
            )
        if moved or st.done_seq < st.request_seq:
            continue
        _ = external_call["sched_yield", Int32]()
        comptime if has_nvidia_gpu_accelerator():
            # Between chunks of one pipelined reduce-scatter or all-gather
            # the next request is microseconds away and a nanosleep is not
            # (see BATCH_POLL_NS): keep yielding until it lands.
            if (
                st.request_seq < batch_end
                and perf_counter_ns() < batch_deadline
                and not _comm_stopped(st)
                and _load_atomic_i(_err_ptr(st)) == 0
            ):
                continue
        if Atomic[Scalar[DType.uint64]].load[ordering=Ordering.ACQUIRE](
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
            "mojoccl: progress thread cpu=" + String(cpu) + " out of range"
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
    """Topology-based proxy placement, or -1 when pinning would crowd ranks.

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
    # communicator init. Placement follows the process affinity mask.
    var cpu = _default_proxy_cpu(local_rank, local_world)
    if cpu < 0:
        if st.trace:
            print(
                (
                    "mojoccl: progress thread left unpinned (affinity mask too"
                    " small for"
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
        print("mojoccl: progress thread pinning failed:", e)


def _stop_proxy(mut st: IbState):
    if st.thread_id == 0:
        return
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
        _mb(st, MB_STOP), 1
    )
    _ = external_call["pthread_join", Int32](st.thread_id, Int64(0))
    st.thread_id = 0


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
    Atomic[Scalar[DType.uint64]].store[ordering=Ordering.RELEASE](
        _mb(st, MB_STOP), 1
    )
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
    _synchronous_test: Bool = False,
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

    `_synchronous_test` is private to the ib_bringup, ib_pipeline and
    fabric_hmem selftests. They drive `ib_drive` from the calling thread;
    starting a proxy would race that driver. The two host-only probes also
    pass libc as their driver and cannot allocate a GPU-visible mailbox.
    Production always uses the proxy; this is not an environment setting.
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
    st.proxy = not _synchronous_test
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


def _proxy_idle_ns() -> Int:
    """Idle quantum; no measured improvement justifies changing 20 us."""
    return DEFAULT_IB_PROXY_IDLE_US * 1000


def ib_teardown(ib: Int):
    if ib == 0:
        return
    ref st = _st(ib)[]
    _stop_proxy(st)
    ib_report(ib)
    _teardown_ib_resources(st)
