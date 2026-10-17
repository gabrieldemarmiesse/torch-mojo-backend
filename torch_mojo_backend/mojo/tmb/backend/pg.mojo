"""Process-group core: NCCL / RCCL / mojoccl collectives on a per-device comm
stream.

torch's `MojoProcessGroup` (distributed/process_group.py) is a thin adapter:
it resolves data pointers, stages non-contiguous tensors with torch ops on
the comm stream and wraps a device-typed torch Future whose completion events
are recorded on the comm stream, so consumers on any stream wait exactly as
with ProcessGroupNCCL. Communicators, the comm stream and the library calls
live here. The three libraries share the NCCL C ABI; the 128-byte unique id
is passed the way the x86-64 SysV ABI lays out a by-value struct: 16 words
after the register arguments (see tmb/ccl/init.mojo's ncclCommInitRank; Python
refuses other architectures).

The entries are reached from Python outside the boxed adapter, so each one
holds the backend mutex over its whole body itself (`with Locked():`).

Every entry that enqueues collective work calls `PG.track` after it, which
is what the collective watchdog (watchdog.mojo) watches; a new one must too.
Each takes the flight recorder's tag word of the Python call (`tag`, 0 if
none) and hands it to `track` (`group_end` to `group_fr_mark`): the entries
it causes record it.
"""
from std.collections import Dict
from std.ffi import OwnedDLHandle, c_char, external_call
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns, sleep

from tmb.backend.device import (
    _add_stream,
    be,
    current_stream,
    dev,
    set_error,
    stream_ctx,
)
from tmb.backend.flight_recorder import (
    fr_after_fork,
    fr_init,
    fr_member,
    fr_pg,
    fr_ring,
    fr_selftest,
    fr_snapshot,
)
from tmb.backend.vendor import raw_stream_of
from tmb.backend.watchdog import (
    ACTION_OFF,
    InCall,
    OP_ABORT,
    OP_ALLGATHER,
    OP_ALLREDUCE,
    OP_BROADCAST,
    OP_GROUP_END,
    OP_RECV,
    OP_REDUCE,
    OP_REDUCE_SCATTER,
    OP_SEND,
    CommFn,
    Entry,
    call_fn,
    call_fp,
    group_enter,
    group_exit,
    group_add,
    group_fr_mark,
    group_open,
    watch_add,
    watch_claim,
    watch_commit,
    watch_create,
    watch_deadline_ns,
    watch_destroy,
    watch_drain,
    watch_fault,
    watch_note,
    watch_exit,
    watch_test_limit,
    watch_next_cpos,
    watch_process_exiting,
    watch_set_fr,
)

comptime UID_BYTES = 128


struct Locked:
    """Holds the backend mutex (shim tmb_lock) for a `with` block.

    Only `with Locked():` takes the lock: constructing one and binding it to
    a local does not. Mojo destroys a value right after its LAST USE, so a
    `var _lock = Locked()` that nothing touches again is destroyed on the
    spot -- a lock-in-`__init__` / unlock-in-`__deinit__` guard would have
    unlocked before the body it was meant to protect ever ran (verified: an
    unused guard printed its destructor message before the body's).
    `__enter__` / `__exit__` are scoped to the block instead, and run on the
    exceptional exit too.
    """

    def __init__(out self):
        pass

    def __enter__(mut self):
        external_call["tmb_lock", NoneType]()

    def __exit__(mut self):
        external_call["tmb_unlock", NoneType]()


@fieldwise_init
struct Comm(Copyable, Movable):
    var handle: Int64
    var device: Int
    var stream: Int  # comm stream id on the device
    var raw: Int  # its vendor stream handle (what the library enqueues on)
    var tracker: Int  # its watchdog ring (watchdog.mojo), 0 when unwatched


comptime CollFn = def(Int, Int, Int, Int32, Int32, Int64, Int64) thin abi(
    "C"
) -> Int32
comptime ReduceFn = def(
    Int, Int, Int, Int32, Int32, Int32, Int64, Int64
) thin abi("C") -> Int32
comptime GatherFn = def(Int, Int, Int, Int32, Int64, Int64) thin abi(
    "C"
) -> Int32
comptime P2pFn = def(Int, Int, Int32, Int32, Int64, Int64) thin abi(
    "C"
) -> Int32
comptime GroupFn = def() thin abi("C") -> Int32


def _symbol(lib: OwnedDLHandle, name: String) raises -> Int:
    var s = lib.get_symbol[NoneType](name)
    if not s:
        raise Error("collectives library lacks ", name)
    return Int(s.value())


struct PG(Movable):
    var lib: OwnedDLHandle
    var rank: Int
    var world: Int
    var comms: Dict[Int, Comm]
    # entry points resolved once (a dlsym per collective was measurable)
    var f_allreduce: Int
    var f_broadcast: Int
    var f_reduce: Int
    var f_allgather: Int
    var f_reduce_scatter: Int
    var f_send: Int
    var f_recv: Int
    var f_group_start: Int
    var f_group_end: Int
    # Collectives issued from compiled MAX graphs (tmb_pg_graph_start/_wait):
    # idle vendor events per device, and the completion event of every
    # collective still waiting for its graph wait, keyed by its output.
    var graph_events: Dict[Int, List[Int]]
    var graph_pending: Dict[Int, Int]
    var watch: Int  # the collective watchdog (watchdog.mojo), 0 when off
    var timeout_ns: Int  # init_process_group's timeout, 0 when unknown
    var closing: Bool  # destroyed or being destroyed: every entry fails
    var fr: Int  # its flight recorder state (flight_recorder.mojo), 0 if off

    def __init__(out self, path: String, rank: Int, world: Int) raises:
        self.lib = OwnedDLHandle(path)
        self.rank = rank
        self.world = world
        self.comms = Dict[Int, Comm]()
        self.f_allreduce = _symbol(self.lib, "ncclAllReduce")
        self.f_broadcast = _symbol(self.lib, "ncclBroadcast")
        self.f_reduce = _symbol(self.lib, "ncclReduce")
        self.f_allgather = _symbol(self.lib, "ncclAllGather")
        self.f_reduce_scatter = _symbol(self.lib, "ncclReduceScatter")
        self.f_send = _symbol(self.lib, "ncclSend")
        self.f_recv = _symbol(self.lib, "ncclRecv")
        self.f_group_start = _symbol(self.lib, "ncclGroupStart")
        self.f_group_end = _symbol(self.lib, "ncclGroupEnd")
        self.graph_events = Dict[Int, List[Int]]()
        self.graph_pending = Dict[Int, Int]()
        self.watch = 0
        self.timeout_ns = 0
        self.closing = False
        self.fr = 0

    def symbol(self, name: String) raises -> Int:
        return _symbol(self.lib, name)

    def error_string(self, rc: Int32) -> String:
        try:
            var p = self.lib.get_function[Pointer[UInt8, MutUntrackedOrigin]](
                "ncclGetErrorString"
            )(rc)
            return String(unsafe_from_utf8_ptr=p)
        except e:
            return "error " + String(rc)

    def check(self, rc: Int32, what: StaticString) raises:
        if rc != 0:
            raise Error(what, ": ", self.error_string(rc))

    def open_or_raise(self) raises:
        if self.closing:
            raise Error("the process group is being destroyed")

    def comm(self, device: Int) raises -> Comm:
        self.open_or_raise()
        var c = self.comms.find(device)
        if not c:
            raise Error(
                "no communicator on mojo:",
                device,
                " (init_device first, or it was aborted)",
            )
        return c.value().copy()

    def graph_event(mut self, device: Int) raises -> Int:
        """An idle event of `device` (vendor driver handle, no timing)."""
        var idle = self.graph_events.find(device)
        if idle and len(idle.value()) > 0:
            var ev = self.graph_events[device].pop()
            return ev
        return be()[].vendor.value().event_create(dev(device)[].ctx, False)

    def graph_release(mut self, device: Int, ev: Int):
        """Back to the pool. A wait enqueued on the event already captured
        its last record, so the event may be recorded again right away."""
        var pool = self.graph_events.pop(device, List[Int]())
        pool.append(ev)
        self.graph_events[device] = pool^

    def track(
        mut self,
        c: Comm,
        op: Int,
        count: Int,
        dtype: Int32,
        root: Int = -1,
        redop: Int = -1,
        tag: Int = 0,
    ):
        """Hand what a library call just enqueued on `c`'s comm stream to the
        watchdog (and the flight recorder: `tag` is the Python call's tag
        word, 0 if none). Call it right after every successful call that enqueues
        collective work there (`root` is the root or peer, `redop` the
        reduction). Inside an NCCL group -- this thread's, on this library,
        whichever process group opened it -- the calls are folded into one
        entry committed by the outermost ncclGroupEnd."""
        if c.tracker == 0:
            return
        var d = Int(dtype)
        var e = Entry(
            0,
            op,
            1,
            count,
            d,
            root,
            redop,
            call_fp(op, count, d, root, redop),
            0,
            0,
            watch_next_cpos(c.tracker, op),
        )
        if group_open(self.watch, self.f_group_start):
            # first: an operation before a block's first member is recorded
            # before the member, as it was issued
            group_add(self.f_group_start, self.watch, c.tracker, c.raw, e, tag)
            fr_member(
                self.fr,
                op,
                count,
                d,
                root,
                op == OP_SEND or op == OP_RECV,
                c.tracker,
                tag,
            )
        else:
            watch_commit(self.watch, c.tracker, c.raw, e^, tag)

    def sync_in(self, c: Comm) raises:
        """The comm stream waits for everything the caller's current stream has
        enqueued so far (inputs are ready, outputs are no longer being read)."""
        var cur = current_stream(c.device)
        if cur != c.stream:
            stream_ctx(c.device, c.stream).enqueue_wait_for(
                stream_ctx(c.device, cur)
            )


comptime PGP = Pointer[PG, MutUntrackedOrigin]


@always_inline
def _pg(p: Int) -> PGP:
    return PGP(unsafe_from_address=p)


def tmb_pg_create(
    path: Pointer[c_char, MutUntrackedOrigin], rank: Int32, world: Int32
) abi("C") -> Int:
    with Locked():
        try:
            var box = unsafe_alloc[PG](1)
            box.unsafe_write(
                PG(
                    String(unsafe_from_utf8_ptr=path.unsafe_bitcast[UInt8]()),
                    Int(rank),
                    Int(world),
                )
            )
            return Int(box)
        except e:
            set_error(String(e))
            return 0


comptime UNWATCHED_DRAIN_NS = 1_800_000_000_000  # torch's default timeout


def _stream_idle(c: Comm, deadline_ns: Int) -> Bool:
    """Wait, bounded, until the comm stream has nothing left to run."""
    try:
        ref v = be()[].vendor.value()
        with dev(c.device)[].ctx.push_context():
            while not v.stream_query(c.raw):
                if perf_counter_ns() > deadline_ns:
                    return False
                sleep(0.0002)
        return True
    except:
        return False


def tmb_pg_destroy(p: Int) abi("C"):
    """Tear the communicators down once this rank's collectives completed --
    ncclCommAbort, local and never waiting for the peers, as
    ProcessGroupNCCL's shutdown does: a peer still busy (writing its final
    checkpoint) must not make this rank's teardown time out. A collective
    that does not complete is the watchdog's (it reports, and exits in abort
    mode); past its deadline the communicators are torn down anyway.

    Three steps. Under the backend mutex -- so every native call already on
    this group has finished submitting -- the group is marked closing (later
    entries fail with an error) and its communicators are snapshotted. The
    drains then run without the mutex: a thread exiting the interpreter
    (freeing tensors) must not block behind them. The aborts and freeing
    take the mutex again. The group's shell is kept, not freed, so a call
    that raced the destroy gets that error rather than freed memory."""
    if p == 0:
        return
    var box = _pg(p)
    var comms = List[Comm]()
    var w: Int
    with Locked():
        if box[].closing:
            return
        box[].closing = True
        w = box[].watch
        for item in box[].comms.items():
            comms.append(item.value.copy())
    # Unwatched communicators (the watchdog off, or its table full) still
    # finish what they started before the local abort.
    var bound = box[].timeout_ns if box[].timeout_ns > 0 else UNWATCHED_DRAIN_NS
    for c in comms:
        if c.tracker == 0 and not _stream_idle(c, perf_counter_ns() + bound):
            print(
                "torch-mojo-backend: collectives on device",
                c.device,
                "did not finish within the timeout; aborting them",
            )
    if w != 0 and not watch_drain(w, perf_counter_ns() + watch_deadline_ns(w)):
        watch_note(
            w,
            (
                "collectives of this process group are unaccounted for;"
                " aborting its communicators without waiting for them"
            ),
        )
    with Locked():
        try:
            var f = call_fn[CommFn](box[].symbol("ncclCommAbort"))
            for c in comms:
                if watch_claim(c.tracker):
                    with InCall(w, OP_ABORT, c.device):
                        _ = f(c.handle)
            if be()[].vendor:
                for item in box[].graph_events.items():
                    for ev in item.value:
                        be()[].vendor.value().event_destroy(ev)
                for item in box[].graph_pending.items():
                    be()[].vendor.value().event_destroy(item.value)
                box[].graph_events = Dict[Int, List[Int]]()
                box[].graph_pending = Dict[Int, Int]()
        except e:
            set_error(String(e))
        watch_destroy(w)
        box[].watch = 0
        box[].comms = Dict[Int, Comm]()


def tmb_pg_version(p: Int) abi("C") -> Int32:
    with Locked():
        try:
            _pg(p)[].open_or_raise()
            var v: Int32 = 0
            _pg(p)[].check(
                _pg(p)[].lib.get_function[Int32]("ncclGetVersion")(
                    Pointer(to=v)
                ),
                "ncclGetVersion",
            )
            return v
        except e:
            set_error(String(e))
            return -1


def tmb_pg_unique_id(
    p: Int, buf: Pointer[UInt8, MutUntrackedOrigin]
) abi("C") -> Int32:
    with Locked():
        try:
            _pg(p)[].open_or_raise()
            _pg(p)[].check(
                _pg(p)[].lib.get_function[Int32]("ncclGetUniqueId")(buf),
                "ncclGetUniqueId",
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_init_device(
    p: Int, device: Int32, uid: Pointer[UInt8, MutUntrackedOrigin]
) abi("C") -> Int32:
    """Create the communicator of this rank on mojo:`device` (one per device)
    and its dedicated comm stream. Collective across ranks."""
    with Locked():
        try:
            var pg = _pg(p)
            pg[].open_or_raise()
            if pg[].comms.find(Int(device)):
                raise Error("mojo:", device, " already has a communicator")
            var d = dev(Int(device))
            var sid = _add_stream(d, 0)
            var raw = d[].raw[sid]
            if raw == 0:
                raise Error(
                    "no vendor stream handle for the comm stream (NCCL needs"
                    " one)"
                )
            var w = uid.unsafe_bitcast[UInt64]()
            var handle: Int64 = 0
            var rc: Int32
            with d[].ctx.push_context():  # ncclCommInitRank binds the communicator to the current device
                rc = pg[].lib.get_function[Int32]("ncclCommInitRank")(
                    Pointer(to=handle),
                    Int32(pg[].world),
                    Int32(pg[].rank),
                    Int64(0),
                    Int64(0),
                    Int64(0),
                    w[unsafe_offset=0],
                    w[unsafe_offset=1],
                    w[unsafe_offset=2],
                    w[unsafe_offset=3],
                    w[unsafe_offset=4],
                    w[unsafe_offset=5],
                    w[unsafe_offset=6],
                    w[unsafe_offset=7],
                    w[unsafe_offset=8],
                    w[unsafe_offset=9],
                    w[unsafe_offset=10],
                    w[unsafe_offset=11],
                    w[unsafe_offset=12],
                    w[unsafe_offset=13],
                    w[unsafe_offset=14],
                    w[unsafe_offset=15],
                )
            pg[].check(rc, "ncclCommInitRank")
            var tracker = 0
            if pg[].watch != 0:
                tracker = watch_add(pg[].watch, Int(device), handle, d[].ctx)
            pg[].comms[Int(device)] = Comm(
                handle, Int(device), sid, raw, tracker
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_comm_stream(p: Int, device: Int32) abi("C") -> Int64:
    with Locked():
        try:
            return Int64(_pg(p)[].comm(Int(device)).stream)
        except e:
            set_error(String(e))
            return -1


def tmb_pg_allreduce(
    p: Int,
    device: Int32,
    ptr: Int,
    count: Int,
    dtype: Int32,
    op: Int32,
    tag: Int64,
) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            var rc: Int32
            with InCall(pg[].watch, OP_ALLREDUCE, c.device):
                rc = call_fn[CollFn](pg[].f_allreduce)(
                    ptr, ptr, count, dtype, op, c.handle, Int64(c.raw)
                )
            pg[].check(rc, "ncclAllReduce")
            pg[].track(
                c, OP_ALLREDUCE, count, dtype, redop=Int(op), tag=Int(tag)
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_broadcast(
    p: Int,
    device: Int32,
    ptr: Int,
    count: Int,
    dtype: Int32,
    root: Int32,
    tag: Int64,
) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            var rc: Int32
            with InCall(pg[].watch, OP_BROADCAST, c.device):
                rc = call_fn[CollFn](pg[].f_broadcast)(
                    ptr, ptr, count, dtype, root, c.handle, Int64(c.raw)
                )
            pg[].check(rc, "ncclBroadcast")
            pg[].track(
                c, OP_BROADCAST, count, dtype, root=Int(root), tag=Int(tag)
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_reduce(
    p: Int,
    device: Int32,
    ptr: Int,
    count: Int,
    dtype: Int32,
    op: Int32,
    root: Int32,
    tag: Int64,
) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            var rc: Int32
            with InCall(pg[].watch, OP_REDUCE, c.device):
                rc = call_fn[ReduceFn](pg[].f_reduce)(
                    ptr, ptr, count, dtype, op, root, c.handle, Int64(c.raw)
                )
            pg[].check(rc, "ncclReduce")
            pg[].track(
                c,
                OP_REDUCE,
                count,
                dtype,
                root=Int(root),
                redop=Int(op),
                tag=Int(tag),
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_allgather(
    p: Int,
    device: Int32,
    send: Int,
    recv: Int,
    count: Int,
    dtype: Int32,
    tag: Int64,
) abi("C") -> Int32:
    """recv[world * count] <- every rank's send[count]."""
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            var rc: Int32
            with InCall(pg[].watch, OP_ALLGATHER, c.device):
                rc = call_fn[GatherFn](pg[].f_allgather)(
                    send, recv, count, dtype, c.handle, Int64(c.raw)
                )
            pg[].check(rc, "ncclAllGather")
            pg[].track(c, OP_ALLGATHER, count, dtype, tag=Int(tag))
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_reduce_scatter(
    p: Int,
    device: Int32,
    send: Int,
    recv: Int,
    count: Int,
    dtype: Int32,
    op: Int32,
    tag: Int64,
) abi("C") -> Int32:
    """recv[count] <- reduce of send[world * count] chunk `rank`."""
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            var rc: Int32
            with InCall(pg[].watch, OP_REDUCE_SCATTER, c.device):
                rc = call_fn[CollFn](pg[].f_reduce_scatter)(
                    send, recv, count, dtype, op, c.handle, Int64(c.raw)
                )
            pg[].check(rc, "ncclReduceScatter")
            pg[].track(
                c, OP_REDUCE_SCATTER, count, dtype, redop=Int(op), tag=Int(tag)
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_send(
    p: Int,
    device: Int32,
    ptr: Int,
    count: Int,
    dtype: Int32,
    peer: Int32,
    tag: Int64,
) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            var rc: Int32
            with InCall(pg[].watch, OP_SEND, c.device):
                rc = call_fn[P2pFn](pg[].f_send)(
                    ptr, count, dtype, peer, c.handle, Int64(c.raw)
                )
            pg[].check(rc, "ncclSend")
            pg[].track(c, OP_SEND, count, dtype, root=Int(peer), tag=Int(tag))
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_recv(
    p: Int,
    device: Int32,
    ptr: Int,
    count: Int,
    dtype: Int32,
    peer: Int32,
    tag: Int64,
) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            var rc: Int32
            with InCall(pg[].watch, OP_RECV, c.device):
                rc = call_fn[P2pFn](pg[].f_recv)(
                    ptr, count, dtype, peer, c.handle, Int64(c.raw)
                )
            pg[].check(rc, "ncclRecv")
            pg[].track(c, OP_RECV, count, dtype, root=Int(peer), tag=Int(tag))
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_group_start(p: Int) abi("C") -> Int32:
    with Locked():
        try:
            _pg(p)[].open_or_raise()
            _pg(p)[].check(
                call_fn[GroupFn](_pg(p)[].f_group_start)(), "ncclGroupStart"
            )
            group_enter(_pg(p)[].f_group_start)
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_group_end(p: Int, tag: Int64) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            pg[].open_or_raise()
            for c in pg[].comms.values():  # its operation (`tag`) ends
                group_fr_mark(
                    pg[].f_group_start, pg[].watch, c.tracker, Int(tag)
                )
            var rc: Int32
            with InCall(pg[].watch, OP_GROUP_END, -1):
                rc = call_fn[GroupFn](pg[].f_group_end)()
            # ncclGroupEnd leaves the group even when it fails
            group_exit(pg[].f_group_start, rc == 0)
            pg[].check(rc, "ncclGroupEnd")
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_async_error(p: Int, device: Int32) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            var err: Int32 = 0
            pg[].check(
                pg[].lib.get_function[Int32]("ncclCommGetAsyncError")(
                    c.handle, Pointer(to=err)
                ),
                "ncclCommGetAsyncError",
            )
            return err
        except e:
            set_error(String(e))
            return -1


def tmb_pg_abort(p: Int, device: Int32) abi("C") -> Int32:
    """Abort and forget the device's communicator (vendor NCCL destroys it)."""
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            _ = pg[].comms.pop(Int(device))
            if not watch_claim(c.tracker):
                return 0  # the watchdog is aborting it (and exiting)
            var rc: Int32
            with InCall(pg[].watch, OP_ABORT, c.device):
                rc = call_fn[CommFn](pg[].symbol("ncclCommAbort"))(c.handle)
            pg[].check(rc, "ncclCommAbort")
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_synchronize_comm(p: Int, device: Int32) abi("C") -> Int32:
    """Host-wait for every collective issued so far on the device's comm stream.
    """
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            stream_ctx(c.device, c.stream).synchronize()
            return 0
        except e:
            set_error(String(e))
            return 1


# --- collectives in compiled graphs --------------------------------------------
#
# A MAX graph (torch.compile(backend=mojo_backend)) runs on a stream of MAX's
# own, handed to its custom ops; tmb/graph/collectives.mojo calls these two
# entries through the addresses `pg_vtable` exposes. The collective runs on
# the communicator's comm stream -- the one eager collectives use, so eager
# and compiled collectives of a rank reach the communicator in program order
# -- between two event joins: the comm stream waits for the graph stream
# before it (inputs ready), and `tmb_pg_graph_wait` makes the graph stream
# wait for it at the graph's `wait_tensor`, so compute between the two
# overlaps the transfer.

comptime GRAPH_ALLREDUCE = 0
comptime GRAPH_ALLGATHER = 1
comptime GRAPH_REDUCE_SCATTER = 2
comptime GRAPH_BROADCAST = 3


def _graph_op(kind: Int) -> Int:
    if kind == GRAPH_ALLGATHER:
        return OP_ALLGATHER
    if kind == GRAPH_REDUCE_SCATTER:
        return OP_REDUCE_SCATTER
    if kind == GRAPH_BROADCAST:
        return OP_BROADCAST
    return OP_ALLREDUCE


def _need_vendor() raises:
    if not be()[].vendor:
        raise Error("collectives in a compiled graph need the vendor driver")


def tmb_pg_graph_start(
    p: Int,
    device: Int32,
    kind: Int32,
    send: Int,
    recv: Int,
    count: Int,
    dtype: Int32,
    arg: Int32,
    stream: Int,
) abi("C") -> Int32:
    """Enqueue one collective of a compiled graph on the comm stream after
    the work `stream` (an AsyncRT stream handle) has enqueued so far. `arg`
    is the reduction op, or the root of a broadcast; `count` is per rank
    (the input of an all-gather, the output of a reduce-scatter). `recv`
    keys the completion event `tmb_pg_graph_wait` joins on."""
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            # An entry for this address can only be stale (an execute failed
            # between a start and its wait, and MAX's memory plan handed the
            # address out again): a live one's buffer is still allocated.
            # The comm stream already orders that collective before this one.
            var stale = pg[].graph_pending.pop(recv, -1)
            if stale != -1:
                pg[].graph_release(c.device, stale)
            _need_vendor()
            var graph_raw = raw_stream_of(be()[].vendor.value(), stream)
            var ready = pg[].graph_event(c.device)
            try:
                be()[].vendor.value().event_record(ready, graph_raw)
                be()[].vendor.value().stream_wait_event(c.raw, ready)
            except e:
                pg[].graph_release(c.device, ready)
                raise e^
            pg[].graph_release(c.device, ready)
            var rc: Int32
            var what: StaticString
            var op = _graph_op(Int(kind))
            with InCall(pg[].watch, op, c.device):
                if kind == GRAPH_ALLREDUCE:
                    what = "ncclAllReduce"
                    rc = call_fn[CollFn](pg[].f_allreduce)(
                        send, recv, count, dtype, arg, c.handle, Int64(c.raw)
                    )
                elif kind == GRAPH_ALLGATHER:
                    what = "ncclAllGather"
                    rc = call_fn[GatherFn](pg[].f_allgather)(
                        send, recv, count, dtype, c.handle, Int64(c.raw)
                    )
                elif kind == GRAPH_REDUCE_SCATTER:
                    what = "ncclReduceScatter"
                    rc = call_fn[CollFn](pg[].f_reduce_scatter)(
                        send, recv, count, dtype, arg, c.handle, Int64(c.raw)
                    )
                elif kind == GRAPH_BROADCAST:
                    what = "ncclBroadcast"
                    rc = call_fn[CollFn](pg[].f_broadcast)(
                        send, recv, count, dtype, arg, c.handle, Int64(c.raw)
                    )
                else:
                    raise Error("unknown graph collective kind ", kind)
            pg[].check(rc, what)
            # Watched like the eager entries: a hung collective issued from a
            # compiled graph is reported the same way.
            if kind == GRAPH_BROADCAST:
                pg[].track(c, op, count, dtype, root=Int(arg))
            elif kind == GRAPH_ALLGATHER:
                pg[].track(c, op, count, dtype)
            else:
                pg[].track(c, op, count, dtype, redop=Int(arg))
            var done = pg[].graph_event(c.device)
            try:
                be()[].vendor.value().event_record(done, c.raw)
            except e:
                pg[].graph_release(c.device, done)
                raise e^
            pg[].graph_pending[recv] = done
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_watchdog(
    p: Int,
    timeout_ms: Int64,
    action: Int32,
    notify_fd: Int32,
    ack_fd: Int32,
    ack_wait_ms: Int64,
) abi("C") -> Int32:
    """Take the group's timeout and, unless `action` is off, start its
    collective watchdog (watchdog.mojo); communicators created from now on
    are watched. `notify_fd` / `ack_fd` are the pipe ends of the Python
    desync reporter, -1 for none."""
    with Locked():
        try:
            var pg = _pg(p)
            pg[].open_or_raise()
            pg[].timeout_ns = Int(timeout_ms) * 1_000_000
            if Int(action) == ACTION_OFF:
                return 0
            if pg[].watch != 0:
                raise Error("the watchdog is already running")
            if not be()[].vendor:
                raise Error("the watchdog needs the CUDA/HIP driver's events")
            ref v = be()[].vendor.value()
            pg[].watch = watch_create(
                Int(timeout_ms) * 1_000_000,
                Int(action),
                Int(notify_fd),
                Int(ack_fd),
                Int(ack_wait_ms) * 1_000_000,
                pg[].rank,
                pg[].world,
                _symbol(v.lib, v.names.event_record),
                _symbol(v.lib, v.names.event_query),
                pg[].symbol("ncclCommAbort"),
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_graph_wait(
    p: Int, device: Int32, recv: Int, stream: Int
) abi("C") -> Int32:
    """The graph stream waits for the collective that writes `recv`."""
    with Locked():
        try:
            var pg = _pg(p)
            pg[].open_or_raise()
            _need_vendor()
            var graph_raw = raw_stream_of(be()[].vendor.value(), stream)
            var done = pg[].graph_pending[recv]
            be()[].vendor.value().stream_wait_event(graph_raw, done)
            _ = pg[].graph_pending.pop(recv)
            pg[].graph_release(Int(device), done)
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_watchdog_exit(p: Int) abi("C"):
    """Interpreter exit: let the collectives in flight finish (a hung one is
    reported), then stop the watchdog thread before MAX and the driver tear
    down under it. No backend mutex (exit may be reached with it held by a
    hung thread); the Python side serializes it against destroy."""
    if p != 0:
        watch_exit(_pg(p)[].watch)


def tmb_pg_watchdog_test_limit(p: Int, device: Int32, entries: Int32) abi("C"):
    """Test hook: at most `entries` collectives in flight on the device's
    communicator, so a test can fill the queue."""
    with Locked():
        try:
            watch_test_limit(_pg(p)[].comm(Int(device)).tracker, Int(entries))
        except e:
            set_error(String(e))


def tmb_pg_watchdog_test_fault(p: Int, device: Int32) abi("C"):
    """Test hook: flag a driver error on the device's communicator, as a
    failed event record would."""
    with Locked():
        try:
            watch_fault(_pg(p)[].comm(Int(device)).tracker, -2)
        except e:
            set_error(String(e))


def tmb_pg_watchdog_process_exiting() abi("C"):
    """Interpreter exit: every watch -- a group being shut down on another
    thread included -- takes driver teardown for a quiet stop."""
    watch_process_exiting()


def tmb_pg_fr_init(size: Int64) abi("C") -> Int:
    """The flight recorder's ring, made with `size` entries on the first
    call: the address of its header, whose word 2 is the tag word
    flight_recorder.py writes. 0 when `size` is 0."""
    with Locked():
        return fr_init(Int(size))


def tmb_pg_fr_attach(p: Int, pg_id: Int64) abi("C") -> Int32:
    """Record the group's collectives in the flight recorder as `pg_id`.
    Needs the ring and the group's watchdog (the recorder follows it)."""
    with Locked():
        try:
            var pg = _pg(p)
            pg[].open_or_raise()
            var ring = fr_ring()
            if ring == 0 or pg[].watch == 0:
                raise Error(
                    "the flight recorder needs its ring and the watchdog"
                )
            if pg[].fr == 0:
                pg[].fr = fr_pg(ring, Int(pg_id))
                watch_set_fr(pg[].watch, pg[].fr)
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_fr_snapshot(buf: Int, cap: Int64) abi("C") -> Int64:
    """Copy up to `cap` flight recorder entries, oldest first, into `buf`
    (flight_recorder.mojo F_WORDS int64 words each). No backend mutex: a
    dump must work while a hung thread holds it."""
    return Int64(fr_snapshot(buf, Int(cap)))


def tmb_pg_fr_selftest() abi("C") -> Int64:
    """Test hook: flight_recorder.mojo `fr_selftest` (0 when it passes)."""
    return Int64(fr_selftest())


def tmb_pg_fr_after_fork() abi("C"):
    """A forked child: flight_recorder.mojo `fr_after_fork`."""
    fr_after_fork()


comptime PG_VTABLE_SLOTS = 30


def pg_vtable() -> Pointer[Int, MutUntrackedOrigin]:
    """Function addresses for the Python adapter, in this fixed order:
    0 create, 1 destroy, 2 version, 3 unique_id, 4 init_device, 5 comm_stream,
    6 allreduce, 7 broadcast, 8 reduce, 9 allgather, 10 reduce_scatter,
    11 send, 12 recv, 13 group_start, 14 group_end, 15 async_error, 16 abort,
    17 synchronize_comm, 18 graph_start, 19 graph_wait, 20 watchdog,
    21 watchdog_exit, 22 watchdog_test_fault, 23 watchdog_test_limit,
    24 watchdog_process_exiting, 25 fr_init, 26 fr_attach, 27 fr_snapshot,
    28 fr_selftest, 29 fr_after_fork.
    """
    var t = unsafe_alloc[Int](PG_VTABLE_SLOTS)
    var f0: def(Pointer[c_char, MutUntrackedOrigin], Int32, Int32) thin abi(
        "C"
    ) -> Int = tmb_pg_create
    var f1: def(Int) thin abi("C") -> None = tmb_pg_destroy
    var f2: def(Int) thin abi("C") -> Int32 = tmb_pg_version
    var f3: def(Int, Pointer[UInt8, MutUntrackedOrigin]) thin abi(
        "C"
    ) -> Int32 = tmb_pg_unique_id
    var f4: def(Int, Int32, Pointer[UInt8, MutUntrackedOrigin]) thin abi(
        "C"
    ) -> Int32 = tmb_pg_init_device
    var f5: def(Int, Int32) thin abi("C") -> Int64 = tmb_pg_comm_stream
    var f6: def(Int, Int32, Int, Int, Int32, Int32, Int64) thin abi(
        "C"
    ) -> Int32 = tmb_pg_allreduce
    var f7: def(Int, Int32, Int, Int, Int32, Int32, Int64) thin abi(
        "C"
    ) -> Int32 = tmb_pg_broadcast
    var f8: def(Int, Int32, Int, Int, Int32, Int32, Int32, Int64) thin abi(
        "C"
    ) -> Int32 = tmb_pg_reduce
    var f9: def(Int, Int32, Int, Int, Int, Int32, Int64) thin abi(
        "C"
    ) -> Int32 = tmb_pg_allgather
    var f10: def(Int, Int32, Int, Int, Int, Int32, Int32, Int64) thin abi(
        "C"
    ) -> Int32 = tmb_pg_reduce_scatter
    var f11: def(Int, Int32, Int, Int, Int32, Int32, Int64) thin abi(
        "C"
    ) -> Int32 = tmb_pg_send
    var f12: def(Int, Int32, Int, Int, Int32, Int32, Int64) thin abi(
        "C"
    ) -> Int32 = tmb_pg_recv
    var f13: def(Int) thin abi("C") -> Int32 = tmb_pg_group_start
    var f14: def(Int, Int64) thin abi("C") -> Int32 = tmb_pg_group_end
    var f15: def(Int, Int32) thin abi("C") -> Int32 = tmb_pg_async_error
    var f16: def(Int, Int32) thin abi("C") -> Int32 = tmb_pg_abort
    var f17: def(Int, Int32) thin abi("C") -> Int32 = tmb_pg_synchronize_comm
    var f18: def(Int, Int32, Int32, Int, Int, Int, Int32, Int32, Int) thin abi(
        "C"
    ) -> Int32 = tmb_pg_graph_start
    var f19: def(Int, Int32, Int, Int) thin abi(
        "C"
    ) -> Int32 = tmb_pg_graph_wait
    var f20: def(Int, Int64, Int32, Int32, Int32, Int64) thin abi(
        "C"
    ) -> Int32 = tmb_pg_watchdog
    var f21: def(Int) thin abi("C") -> None = tmb_pg_watchdog_exit
    var f22: def(Int, Int32) thin abi("C") -> None = tmb_pg_watchdog_test_fault
    var f23: def(Int, Int32, Int32) thin abi(
        "C"
    ) -> None = tmb_pg_watchdog_test_limit
    var f24: def() thin abi("C") -> None = tmb_pg_watchdog_process_exiting
    var f25: def(Int64) thin abi("C") -> Int = tmb_pg_fr_init
    var f26: def(Int, Int64) thin abi("C") -> Int32 = tmb_pg_fr_attach
    var f27: def(Int, Int64) thin abi("C") -> Int64 = tmb_pg_fr_snapshot
    var f28: def() thin abi("C") -> Int64 = tmb_pg_fr_selftest
    var f29: def() thin abi("C") -> None = tmb_pg_fr_after_fork
    t[unsafe_offset=0] = Pointer(to=f0).unsafe_bitcast[Int]()[]
    t[unsafe_offset=1] = Pointer(to=f1).unsafe_bitcast[Int]()[]
    t[unsafe_offset=2] = Pointer(to=f2).unsafe_bitcast[Int]()[]
    t[unsafe_offset=3] = Pointer(to=f3).unsafe_bitcast[Int]()[]
    t[unsafe_offset=4] = Pointer(to=f4).unsafe_bitcast[Int]()[]
    t[unsafe_offset=5] = Pointer(to=f5).unsafe_bitcast[Int]()[]
    t[unsafe_offset=6] = Pointer(to=f6).unsafe_bitcast[Int]()[]
    t[unsafe_offset=7] = Pointer(to=f7).unsafe_bitcast[Int]()[]
    t[unsafe_offset=8] = Pointer(to=f8).unsafe_bitcast[Int]()[]
    t[unsafe_offset=9] = Pointer(to=f9).unsafe_bitcast[Int]()[]
    t[unsafe_offset=10] = Pointer(to=f10).unsafe_bitcast[Int]()[]
    t[unsafe_offset=11] = Pointer(to=f11).unsafe_bitcast[Int]()[]
    t[unsafe_offset=12] = Pointer(to=f12).unsafe_bitcast[Int]()[]
    t[unsafe_offset=13] = Pointer(to=f13).unsafe_bitcast[Int]()[]
    t[unsafe_offset=14] = Pointer(to=f14).unsafe_bitcast[Int]()[]
    t[unsafe_offset=15] = Pointer(to=f15).unsafe_bitcast[Int]()[]
    t[unsafe_offset=16] = Pointer(to=f16).unsafe_bitcast[Int]()[]
    t[unsafe_offset=17] = Pointer(to=f17).unsafe_bitcast[Int]()[]
    t[unsafe_offset=18] = Pointer(to=f18).unsafe_bitcast[Int]()[]
    t[unsafe_offset=19] = Pointer(to=f19).unsafe_bitcast[Int]()[]
    t[unsafe_offset=20] = Pointer(to=f20).unsafe_bitcast[Int]()[]
    t[unsafe_offset=21] = Pointer(to=f21).unsafe_bitcast[Int]()[]
    t[unsafe_offset=22] = Pointer(to=f22).unsafe_bitcast[Int]()[]
    t[unsafe_offset=23] = Pointer(to=f23).unsafe_bitcast[Int]()[]
    t[unsafe_offset=24] = Pointer(to=f24).unsafe_bitcast[Int]()[]
    t[unsafe_offset=25] = Pointer(to=f25).unsafe_bitcast[Int]()[]
    t[unsafe_offset=26] = Pointer(to=f26).unsafe_bitcast[Int]()[]
    t[unsafe_offset=27] = Pointer(to=f27).unsafe_bitcast[Int]()[]
    t[unsafe_offset=28] = Pointer(to=f28).unsafe_bitcast[Int]()[]
    t[unsafe_offset=29] = Pointer(to=f29).unsafe_bitcast[Int]()[]
    return t
