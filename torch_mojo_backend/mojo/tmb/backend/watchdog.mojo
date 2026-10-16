"""Collective watchdog of the process group (pg.mojo): turns a hang into a
report and a non-zero exit, ProcessGroupNCCL's contract.

Every collective pg.mojo issues is committed right after the library call
enqueued it: one vendor event recorded on the comm stream and one slot of a
per-communicator queue (sequence number, op, arguments, host issue time). A
watchdog thread per process group polls the events oldest first; the first
one still pending past `init_process_group(timeout=...)` is reported and,
unless the action is "report", the communicators are aborted and the
process exits. A library call that itself blocks on the host (a full queue,
a lazy connection to a peer that never comes, a first-use kernel compile)
is caught by the in-call mark `InCall` leaves for the duration of the call.

Threads. Every producer runs under the backend mutex (pg.mojo's `Locked`),
so the queue has one producer at a time; it never waits for the GPU (an
issuing thread that waited could be the one whose next call lets the peer
make progress). The watchdog thread never takes that mutex -- the thread
that hung may be holding it. The queue grows by segments, up to a table of
NSEG of them in flight; a fully retired segment is remapped, never freed before the
watch, so every slot read is of live memory; slots are seqlocked (every
field an atomic word), so a read racing a reuse is detected, not torn. `tail` only moves forward, by
compare-exchange: the watchdog retires entries, and so does the issuer, at
most once per segment and without waiting (`_retire_now`).

Ownership. A communicator is aborted or destroyed by whoever wins
`watch_claim` (explicit abort, destroy, or the watchdog's abort), never by
two of them.
"""
from std.atomic import Atomic, Ordering
from std.ffi import _get_global_or_null, external_call, get_errno
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns, sleep

from max.gpu.host import DeviceContext

from tmb.backend.device import be
from tmb.backend.flight_recorder import fr_commit, fr_done
from tmb.backend.vendor import NOT_READY

comptime SEG = 1024  # slots per queue segment
comptime NSEG = 1024  # segments mapped at most: ~1M collectives in flight
comptime MAX_TRACKERS = 64  # communicators over a group's life (abort + re-init)
comptime POLL_S = 0.1  # ProcessGroupNCCL's watchdog sleeps 100 ms too
comptime HISTORY = 8  # completed collectives listed in a report
comptime ABORT_WAIT_NS = 10_000_000_000  # bound on ncclCommAbort before _exit
comptime WRITE_WAIT_NS = 2_000_000_000  # bound on one report write
comptime EXIT_WAIT_NS = 2_000_000_000  # interpreter exit: in-flight work's grace
comptime EXIT_ACK_NS = 3_000_000_000  # ...and its desync report's window
comptime PIPE_BUF = 4096  # one write at most
comptime EINTR = Int32(4)
comptime EAGAIN = Int32(11)
comptime PUSH_FAILURES = 3  # context pushes failing in a row = a device fault
# Driver errors that mean the process is tearing down (same codes on HIP).
comptime DEINITIALIZED = Int32(4)
comptime INVALID_CONTEXT = Int32(201)
comptime CONTEXT_DESTROYED = Int32(709)

comptime ACTION_OFF = 0
comptime ACTION_REPORT = 1
comptime ACTION_ABORT = 2

comptime OP_ALLREDUCE = 1
comptime OP_BROADCAST = 2
comptime OP_REDUCE = 3
comptime OP_ALLGATHER = 4
comptime OP_REDUCE_SCATTER = 5
comptime OP_SEND = 6
comptime OP_RECV = 7
comptime OP_SENDRECV = 8  # a group of sends and receives
comptime OP_GROUP = 9  # a group mixing other operations
comptime OP_GROUP_END = 10  # in-call marks only, from here on
comptime OP_ABORT = 12

# Watch words (release/acquire, shared with the watchdog thread).
comptime W_STOP = 0
comptime W_NTRACK = 1
comptime W_CALL_NS = 2  # host start of a library call in progress, 0 if none
comptime W_CALL_OP = 3
comptime W_CALL_DEV = 4
# word 5 is unused (the reaper keeps its deadline in a cell of its own)
comptime W_TEARDOWN = 6  # interpreter exit: driver teardown is a quiet stop
comptime W_JOINER = 7  # 0, 1 while one stopper joins the thread, 2 joined
comptime W_WORDS = 8

# How a report reaches stderr.
comptime OUT_PLAIN = 0  # a file, or stderr reopened non-blocking
comptime OUT_SEND = 1  # a socket: send(MSG_DONTWAIT)
comptime OUT_POLL = 2  # a blocking fd we could not reopen: poll(POLLOUT) first

# Tracker words.
comptime T_HEAD = 0  # entries committed (producer)
comptime T_TAIL = 1  # entries retired (compare-exchange)
comptime T_DONE = 2  # sequence number of the newest retired entry
comptime T_STATE = 3  # 0 live, 1 claimed for abort/destroy
comptime T_UNWATCHED = 4  # 1: the watchdog skips it
comptime T_FAULT = 5  # driver error seen by either side, 0 if none
comptime T_SEQ = 6  # sequence number of the newest commit (producer)
comptime T_CPOS = 7  # compared collective calls so far (producer)
comptime T_WORDS = 8

# Slot words: S_VER is the seqlock (odd while the producer writes).
comptime S_VER = 0
comptime S_SEQ = 1
comptime S_OP = 2
comptime S_NOPS = 3
comptime S_COUNT = 4
comptime S_DTYPE = 5
comptime S_ROOT = 6
comptime S_REDOP = 7
comptime S_FP = 8
comptime S_ISSUED = 9
comptime S_EVENT = 10  # created on the slot's first use, then fixed
comptime S_CPOS = 11
comptime S_FR = 12  # its flight recorder entry (record id + 1), 0 if none
comptime S_WORDS = 13

comptime WordsP = Pointer[Int64, MutUntrackedOrigin]
comptime EventRecordFn = def(Int, Int) thin abi("C") -> Int32
comptime EventQueryFn = def(Int) thin abi("C") -> Int32
comptime CommFn = def(Int64) thin abi("C") -> Int32
comptime ThreadFn = def(OpaquePointer[MutAnyOrigin]) thin abi("C") -> None


@always_inline
def _ld(p: WordsP, i: Int) -> Int:
    return Int(
        Atomic[Scalar[DType.int64]].load[ordering=Ordering.ACQUIRE](
            p.unsafe_offset(i)
        )
    )


@always_inline
def _st(p: WordsP, i: Int, v: Int):
    Atomic[Scalar[DType.int64]].store[ordering=Ordering.RELEASE](
        p.unsafe_offset(i), Int64(v)
    )


@always_inline
def _cas(p: WordsP, i: Int, old: Int, new: Int) -> Bool:
    var expected = Int64(old)
    return Atomic[Scalar[DType.int64]].compare_exchange[
        success_ordering=Ordering.ACQUIRE_RELEASE,
        failure_ordering=Ordering.ACQUIRE,
    ](p.unsafe_offset(i), expected, Int64(new))


def _st_max(p: WordsP, i: Int, v: Int):
    while True:
        var cur = _ld(p, i)
        if cur >= v or _cas(p, i, cur, v):
            return


@always_inline
def call_fn[F: TrivialRegisterPassable](addr: Int) -> F:
    """A C function pointer, resolved once, as a callable."""
    var a = addr
    return Pointer(to=a).unsafe_bitcast[F]()[]


def _words(n: Int) -> WordsP:
    var p = unsafe_alloc[Int64](n)
    for i in range(n):
        p[unsafe_offset=i] = 0
    return WordsP(unsafe_from_address=Int(p))


def _spawn(f: ThreadFn, arg: Int) -> Int:
    """pthread_create; 0 on failure."""
    var tid = unsafe_alloc[Int64](1)
    tid[unsafe_offset=0] = 0
    var fp = f
    var rc = external_call["pthread_create", Int32](
        tid, Int64(0), Pointer(to=fp).unsafe_bitcast[Int]()[], arg
    )
    var t = Int(tid[unsafe_offset=0]) if rc == 0 else 0
    tid.unsafe_free()
    return t


def _sleep_until(deadline_ns: Int):
    """Sleep to an absolute deadline; a signal only shortens one step."""
    while True:
        var left = deadline_ns - perf_counter_ns()
        if left <= 0:
            return
        sleep(min(Float64(left) / 1e9, 0.05))


def op_name(op: Int) -> StaticString:
    if op == OP_ALLREDUCE:
        return "allreduce"
    if op == OP_BROADCAST:
        return "broadcast"
    if op == OP_REDUCE:
        return "reduce"
    if op == OP_ALLGATHER:
        return "allgather"
    if op == OP_REDUCE_SCATTER:
        return "reduce_scatter"
    if op == OP_SEND:
        return "send"
    if op == OP_RECV:
        return "recv"
    if op == OP_SENDRECV:
        return "send/recv"
    if op == OP_GROUP_END:
        return "ncclGroupEnd"
    if op == OP_ABORT:
        return "ncclCommAbort"
    return "group"


def dtype_name(d: Int) -> StaticString:
    """ncclDataType_t."""
    if d == 0:
        return "int8"
    if d == 1:
        return "uint8"
    if d == 2:
        return "int32"
    if d == 3:
        return "uint32"
    if d == 4:
        return "int64"
    if d == 5:
        return "uint64"
    if d == 6:
        return "float16"
    if d == 7:
        return "float32"
    if d == 8:
        return "float64"
    if d == 9:
        return "bfloat16"
    if d == 10:
        return "float8_e4m3"
    if d == 11:
        return "float8_e5m2"
    return "?"


def redop_name(r: Int) -> StaticString:
    """ncclRedOp_t."""
    if r == 0:
        return "sum"
    if r == 1:
        return "prod"
    if r == 2:
        return "max"
    if r == 3:
        return "min"
    if r == 4:
        return "avg"
    return "?"


@always_inline
def _mix(h: UInt64, x: Int) -> UInt64:
    """FNV-1a step: a fingerprint of the calls in order, so two ranks whose
    summaries match but whose arguments differ do not compare equal."""
    return (h ^ UInt64(x & 0xFFFFFFFF)) * 1099511628211


def call_fp(op: Int, count: Int, dtype: Int, root: Int, redop: Int) -> UInt64:
    var h = UInt64(14695981039346656037)
    h = _mix(h, op)
    h = _mix(h, count)
    h = _mix(h, count >> 32)
    h = _mix(h, dtype)
    h = _mix(h, root + 1)
    return _mix(h, redop + 1)


@fieldwise_init
struct Entry(Copyable, Movable):
    var seq: Int
    var op: Int
    var nops: Int  # library calls folded into it (an NCCL group)
    var count: Int  # elements, summed over a group
    var dtype: Int
    var root: Int  # root or peer, -1 when none
    var redop: Int  # -1 when none
    var fp: UInt64  # fingerprint of every call's arguments, in order
    var issued_ns: Int
    var event: Int  # vendor event, created on the slot's first use, reused
    # Position among the calls every rank issues identically (the five
    # compared collectives, grouped or not): what ranks compare. 0 if none.
    var cpos: Int


def _empty_entry() -> Entry:
    return Entry(0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0)


struct Tracker(Movable):
    """One communicator's queue. `g_entry` is the producer's; `hist`,
    `reported`, `fault_reported` and `push_failures` the watchdog's."""

    var words: WordsP
    var segs: WordsP  # NSEG segment addresses, 0 until first used
    var hist: Pointer[Entry, MutUntrackedOrigin]  # last HISTORY retired
    var hist_n: Int
    var id: Int
    var device: Int
    var handle: Int64
    var ctx: DeviceContext
    var g_entry: Entry
    var reported: Int
    var fault_reported: Bool
    var published: Bool  # watchdog's: an incident line went out for it
    var push_failures: Int
    var seg_lo: Int  # producer's: oldest index segment still mapped
    var seg_pool: List[Int]  # producer's: retired segments, for reuse
    var inflight_limit: Int  # entries in flight at most (tests lower it)
    var overflowing: Bool  # producer's: commits are being dropped
    var zero_seg: WordsP  # what an unmapped segment reads as

    def __init__(
        out self, id: Int, device: Int, handle: Int64, ctx: DeviceContext
    ):
        self.words = _words(T_WORDS)
        self.segs = _words(NSEG)
        var hist = unsafe_alloc[Entry](HISTORY)
        for i in range(HISTORY):
            hist.unsafe_offset(i).unsafe_write(_empty_entry())
        self.hist = Pointer[Entry, MutUntrackedOrigin](
            unsafe_from_address=Int(hist)
        )
        self.hist_n = 0
        self.id = id
        self.device = device
        self.handle = handle
        self.ctx = ctx
        self.g_entry = _empty_entry()
        self.reported = 0
        self.fault_reported = False
        self.published = False
        self.push_failures = 0
        self.seg_lo = 0
        self.seg_pool = List[Int]()
        self.inflight_limit = (NSEG - 1) * SEG
        self.overflowing = False
        self.zero_seg = _words(SEG * S_WORDS)


@always_inline
def _slot(tr: Tracker, i: Int) -> WordsP:
    """Slot `i`'s words. A segment moved away under a racing reader reads as
    the zero segment (event 0, version 0); the reader then finds the tail
    moved and discards what it read."""
    var seg = _ld(tr.segs, (i // SEG) % NSEG)
    if seg == 0:
        seg = Int(tr.zero_seg)
    return WordsP(unsafe_from_address=seg + (i % SEG) * S_WORDS * 8)


def _read(tr: Tracker, i: Int) -> Optional[Entry]:
    """A seqlocked snapshot of slot `i`; None when the producer was writing
    it (the caller then re-checks `tail` and gives up for this round)."""
    var s = _slot(tr, i)
    var v = _ld(s, S_VER)
    if v & 1:
        return None
    var e = Entry(
        _ld(s, S_SEQ),
        _ld(s, S_OP),
        _ld(s, S_NOPS),
        _ld(s, S_COUNT),
        _ld(s, S_DTYPE),
        _ld(s, S_ROOT),
        _ld(s, S_REDOP),
        UInt64(_ld(s, S_FP)),
        _ld(s, S_ISSUED),
        _ld(s, S_EVENT),
        _ld(s, S_CPOS),
    )
    if _ld(s, S_VER) != v:
        return None
    return e^


comptime TrackerP = Pointer[Tracker, MutUntrackedOrigin]


struct Watch(Movable):
    var words: WordsP
    var trackers: Pointer[Int, MutUntrackedOrigin]  # MAX_TRACKERS addresses
    var timeout_ns: Int
    var action: Int
    var notify_fd: Int  # the Python desync reporter reads lines here
    var ack_fd: Int  # ...and writes the incident's id byte here when done
    var ack_wait_ns: Int
    var rank: Int
    var world: Int
    var f_record: Int
    var f_query: Int
    var f_abort: Int
    var thread: Int
    var reaper_cell: Int  # the last armed reaper's cell, 0 if none
    var err_fd: Int  # where reports go: stderr, reopened non-blocking
    var err_mode: Int  # OUT_*
    var groups: Int  # the process-wide NCCL group table
    var incident: Int  # watchdog's: reports so far
    var reported_call: Int  # watchdog's: in-call mark already reported
    var warned_full: Bool
    var warned_overflow: Bool
    var fr: Int  # flight_recorder.mojo `fr_pg`, 0 when not recording

    def __init__(
        out self,
        timeout_ns: Int,
        action: Int,
        notify_fd: Int,
        ack_fd: Int,
        ack_wait_ns: Int,
        rank: Int,
        world: Int,
        f_record: Int,
        f_query: Int,
        f_abort: Int,
    ):
        self.words = _words(W_WORDS)
        var t = unsafe_alloc[Int](MAX_TRACKERS)
        for i in range(MAX_TRACKERS):
            t[unsafe_offset=i] = 0
        self.trackers = Pointer[Int, MutUntrackedOrigin](
            unsafe_from_address=Int(t)
        )
        self.timeout_ns = timeout_ns
        self.action = action
        self.notify_fd = notify_fd
        self.ack_fd = ack_fd
        self.ack_wait_ns = ack_wait_ns
        self.rank = rank
        self.world = world
        self.f_record = f_record
        self.f_query = f_query
        self.f_abort = f_abort
        self.thread = 0
        self.reaper_cell = 0
        self.err_fd = 2
        self.err_mode = OUT_PLAIN
        self.groups = Int(_groups())
        self.incident = 0
        self.reported_call = 0
        self.warned_full = False
        self.warned_overflow = False
        self.fr = 0


comptime WatchP = Pointer[Watch, MutUntrackedOrigin]


@always_inline
def _watch(w: Int) -> WatchP:
    return WatchP(unsafe_from_address=w)


@always_inline
def _tracker(t: Int) -> TrackerP:
    return TrackerP(unsafe_from_address=t)


# --- NCCL groups: per thread and per library, like NCCL's own depth ----------


@fieldwise_init
struct Level(Copyable, Movable):
    var thread: Int
    var lib: Int
    var depth: Int
    var pending: List[Int]  # (watch, tracker, raw stream) triples


@fieldwise_init
struct Groups(Movable):
    var levels: List[Level]
    var exiting: WordsP  # [1]: the interpreter is exiting (all watches)


comptime GROUPS_GLOBAL = "TMB_PG_GROUPS"


def _groups() -> Pointer[Groups, MutUntrackedOrigin]:
    """Process-wide, under the backend mutex: a group opened through one
    process group's library also holds back another group's collectives."""
    var p = _get_global_or_null(GROUPS_GLOBAL)
    if p:
        return p.value().unsafe_bitcast[Groups]()
    var box = unsafe_alloc[Groups](1)
    box.unsafe_write(Groups(List[Level](), _words(1)))
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(GROUPS_GLOBAL), box.unsafe_bitcast[NoneType]()
    )
    return Pointer[Groups, MutUntrackedOrigin](unsafe_from_address=Int(box))


comptime GroupsP = Pointer[Groups, MutUntrackedOrigin]


def _level(g: GroupsP, lib: Int) -> Int:
    """Index of this thread's open group on `lib`, -1 when none."""
    ref levels = g[].levels
    if len(levels) == 0:
        return -1
    var me = external_call["pthread_self", Int]()
    for i in range(len(levels)):
        if levels[i].thread == me and levels[i].lib == lib:
            return i
    return -1


def watch_process_exiting():
    """Interpreter exit: every watch, a group being shut down on another
    thread included, takes driver teardown for a quiet stop."""
    _st(_groups()[].exiting, 0, 1)


def _tearing_down(wa: Watch) -> Bool:
    return (
        _ld(wa.words, W_TEARDOWN) != 0
        or _ld(GroupsP(unsafe_from_address=wa.groups)[].exiting, 0) != 0
    )


def group_open(w: Int, lib: Int) -> Bool:
    """On the hot path: the table's address is cached in the watch (the
    global lookup by name costs about a microsecond)."""
    return _level(GroupsP(unsafe_from_address=_watch(w)[].groups), lib) >= 0


def group_enter(lib: Int):
    var i = _level(_groups(), lib)
    if i >= 0:
        _groups()[].levels[i].depth += 1
        return
    _groups()[].levels.append(
        Level(external_call["pthread_self", Int](), lib, 1, List[Int]())
    )


def group_exit(lib: Int, submitted: Bool):
    """The library's ncclGroupEnd returned: at the outermost level, commit
    every tracker's folded calls (all were submitted just now)."""
    var i = _level(_groups(), lib)
    if i < 0:
        return
    ref levels = _groups()[].levels
    levels[i].depth -= 1
    if levels[i].depth > 0:
        return
    var pending = levels[i].pending.copy()
    _ = levels.pop(i)
    for j in range(0, len(pending), 3):
        _group_commit(pending[j], pending[j + 1], pending[j + 2], submitted)


def group_note(lib: Int, w: Int, t: Int, raw_stream: Int):
    var g = GroupsP(unsafe_from_address=_watch(w)[].groups)
    var i = _level(g, lib)
    if i < 0:
        return
    ref pending = g[].levels[i].pending
    for j in range(0, len(pending), 3):
        if pending[j + 1] == t:
            return
    pending.append(w)
    pending.append(t)
    pending.append(raw_stream)


def groups_forget(w: Int):
    """A watch going away: drop its trackers from every open group."""
    ref levels = _groups()[].levels
    for i in range(len(levels)):
        var kept = List[Int]()
        ref pending = levels[i].pending
        for j in range(0, len(pending), 3):
            if pending[j] != w:
                kept.append(pending[j])
                kept.append(pending[j + 1])
                kept.append(pending[j + 2])
        levels[i].pending = kept^


# --- producer side (under the backend mutex) ---------------------------------


struct InCall:
    """Marks a library call in progress for the watchdog (`with InCall(...):`
    around the call): a call blocked on the host never reaches a commit."""

    var w: Int
    var op: Int
    var device: Int

    def __init__(out self, w: Int, op: Int, device: Int):
        self.w = w
        self.op = op
        self.device = device

    @always_inline
    def __enter__(mut self):
        if self.w != 0:
            var words = _watch(self.w)[].words
            _st(words, W_CALL_OP, self.op)
            _st(words, W_CALL_DEV, self.device)
            _st(words, W_CALL_NS, perf_counter_ns())

    @always_inline
    def __exit__(mut self):
        if self.w != 0:
            _st(_watch(self.w)[].words, W_CALL_NS, 0)


def watch_create(
    timeout_ns: Int,
    action: Int,
    notify_fd: Int,
    ack_fd: Int,
    ack_wait_ns: Int,
    rank: Int,
    world: Int,
    f_record: Int,
    f_query: Int,
    f_abort: Int,
) raises -> Int:
    """Allocate the group's watch and start its thread."""
    var box = unsafe_alloc[Watch](1)
    box.unsafe_write(
        Watch(
            timeout_ns,
            action,
            notify_fd,
            ack_fd,
            ack_wait_ns,
            rank,
            world,
            f_record,
            f_query,
            f_abort,
        )
    )
    var w = Int(box)
    _watch(w)[].err_fd = _report_fd(2)
    _watch(w)[].err_mode = _out_mode(_watch(w)[].err_fd)
    var thread = _spawn(_watch_main, w)
    if thread == 0:
        if _watch(w)[].err_fd > 2:
            _ = external_call["close", Int32](Int32(_watch(w)[].err_fd))
        var moved = box.unsafe_take_pointee()
        box.unsafe_free()
        _ = moved^
        raise Error("watchdog: pthread_create failed")
    _watch(w)[].thread = thread
    return w


def watch_add(w: Int, device: Int, handle: Int64, ctx: DeviceContext) -> Int:
    """A tracker for a new communicator, or 0 when the table is full (the
    communicator then goes unwatched, which is said on stderr)."""
    ref wa = _watch(w)[]
    var n = _ld(wa.words, W_NTRACK)
    if n >= MAX_TRACKERS:
        if not wa.warned_full:
            wa.warned_full = True
            _err(
                wa,
                _prefix(wa.rank)
                + "watchdog: more than "
                + String(MAX_TRACKERS)
                + " communicators in one process group; the new ones are"
                " not watched\n",
            )
        return 0
    var box = unsafe_alloc[Tracker](1)
    box.unsafe_write(Tracker(n, device, handle, ctx))
    var t = Int(box)
    wa.trackers[unsafe_offset=n] = t
    _st(wa.words, W_NTRACK, n + 1)  # publishes the slot
    return t


def watch_claim(t: Int) -> Bool:
    """Take the right to abort or destroy the tracker's communicator. False:
    another path (an abort, a destroy, the watchdog) already has it."""
    if t == 0:
        return True
    var ok = _cas(_tracker(t)[].words, T_STATE, 0, 1)
    _st(_tracker(t)[].words, T_UNWATCHED, 1)
    return ok


def watch_set_fr(w: Int, fp: Int):
    """Record this group's collectives in the flight recorder from now on
    (`fp`: flight_recorder.mojo `fr_pg`). Under the backend mutex."""
    if w != 0:
        _watch(w)[].fr = fp


def watch_fault(t: Int, rc: Int):
    """A driver error on either side: the watchdog reports it, and a destroy
    aborts the communicator instead of trusting its queue."""
    if t != 0:
        _ = _cas(_tracker(t)[].words, T_FAULT, 0, rc)


@fieldwise_init
struct Boundary(Copyable, Movable):
    var end: Int  # first entry not complete (head when all are)
    var rc: Int32  # a driver error other than not-ready, 0 if none


def _boundary(
    tr: Tracker, query: EventQueryFn, tail: Int, head: Int
) -> Boundary:
    """Where completion stops in [tail, head). Events of one stream complete
    in order, so the newest is queried first (all done: one query retires
    the lot) and a binary search finds the rest: O(log n) driver calls per
    round, not one per collective. Slot events never change once created."""
    var newest = _ld(_slot(tr, head - 1), S_EVENT)
    if newest == 0:
        return Boundary(tail, 0)  # raced a segment move: nothing this round
    var rc = query(newest)
    if rc == 0:
        return Boundary(head, 0)
    if rc != NOT_READY:
        return Boundary(tail, rc)
    var lo = tail
    var hi = head - 1
    while lo < hi:
        var mid = (lo + hi) // 2
        var ev = _ld(_slot(tr, mid), S_EVENT)
        if ev == 0:
            return Boundary(lo, 0)
        rc = query(ev)
        if rc == 0:
            lo = mid + 1
        elif rc == NOT_READY:
            hi = mid
        else:
            return Boundary(lo, rc)
    return Boundary(lo, 0)


def _fr_retired(w: Int, tr: Tracker, lo: Int, hi: Int):
    """Stamp the flight recorder entries of [lo, hi) completed: called
    before the tail moves past them (their slots are reusable after)."""
    var fp = _watch(w)[].fr
    if fp == 0:
        return
    var now = perf_counter_ns()
    for i in range(lo, hi):
        fr_done(fp, _ld(_slot(tr, i), S_FR), now)


def _retire_now(w: Int, t: Int):
    """Producer-side retirement, once per segment and never waiting: one
    query of the newest event (all done: everything retires), else the
    binary search. Keeps a fast issuer from growing the queue (and creating
    events) between two polls of the watchdog. Entries retired here are not
    in the report's history."""
    ref tr = _tracker(t)[]
    var tail = _ld(tr.words, T_TAIL)
    var head = _ld(tr.words, T_HEAD)
    if tail == head:
        return
    try:
        var b: Boundary
        with tr.ctx.push_context():
            b = _boundary(
                tr, call_fn[EventQueryFn](_watch(w)[].f_query), tail, head
            )
        if b.end > tail:
            var seq = _ld(_slot(tr, b.end - 1), S_SEQ)
            _fr_retired(w, tr, tail, b.end)
            if _cas(tr.words, T_TAIL, tail, b.end):
                _st_max(tr.words, T_DONE, seq)
    except:
        return  # nothing retired: the watchdog sees the same driver state


def _unmap_retired(mut tr: Tracker, k: Int):
    """Unmap every fully retired segment below index segment `k` into the
    pool, so the mapped range only spans what is in flight."""
    var tail = _ld(tr.words, T_TAIL)
    while tr.seg_lo < k and (tr.seg_lo + 1) * SEG <= tail:
        var seg = _ld(tr.segs, tr.seg_lo % NSEG)
        _st(tr.segs, tr.seg_lo % NSEG, 0)
        if seg != 0:
            tr.seg_pool.append(seg)
        tr.seg_lo += 1


def _map_segment(w: Int, t: Int, k: Int):
    """Give index segment `k` memory before its first slot is written: a
    retired segment (its slots and events are reused, so a steady state
    creates no event), else a new one. The memory is never freed before the
    watch, and a watchdog read that raced the move sees the tail moved and
    discards what it read (an unmapped segment reads as the zero segment)."""
    ref tr = _tracker(t)[]
    if _ld(tr.segs, k % NSEG) != 0:
        return
    if tr.seg_lo < k and (tr.seg_lo + 1) * SEG > _ld(tr.words, T_TAIL):
        _retire_now(w, t)
    _unmap_retired(tr, k)
    var seg: Int
    if len(tr.seg_pool) > 0:
        seg = tr.seg_pool.pop()
    else:
        var fresh = unsafe_alloc[Int64](SEG * S_WORDS)
        for i in range(SEG * S_WORDS):
            fresh[unsafe_offset=i] = 0
        seg = Int(fresh)
    _st(tr.segs, k % NSEG, seg)


@always_inline
def _compared(op: Int) -> Bool:
    """Collectives every rank issues with the same arguments."""
    return op >= OP_ALLREDUCE and op <= OP_REDUCE_SCATTER


@always_inline
def _full(tr: Tracker, head: Int) -> Bool:
    return (
        head - _ld(tr.words, T_TAIL) >= tr.inflight_limit
        or head // SEG - tr.seg_lo >= NSEG - 1
    )


def watch_commit(w: Int, t: Int, raw_stream: Int, var e: Entry):
    """Record the completion point of what was just enqueued on the comm
    stream as the tracker's next collective. Never waits for the GPU: the
    queue grows by a segment instead; only with a whole table of segments
    in flight are commits dropped, until enough complete. A driver error
    goes to the watchdog (`watch_fault`)."""
    ref tr = _tracker(t)[]
    var fr = fr_commit(
        _watch(w)[].fr,
        e.op,
        e.nops,
        e.count,
        e.dtype,
        e.root,
        e.op >= OP_SEND and e.op <= OP_SENDRECV,
    )
    if _ld(tr.words, T_FAULT) != 0 or _ld(tr.words, T_STATE) != 0:
        return
    var seq = _ld(tr.words, T_SEQ) + 1
    _st(tr.words, T_SEQ, seq)
    var head = _ld(tr.words, T_HEAD)
    if _full(tr, head):
        _retire_now(w, t)
        _unmap_retired(tr, head // SEG)
        if _full(tr, head):
            if not tr.overflowing:
                tr.overflowing = True
                watch_note(
                    w,
                    String(
                        "more than ",
                        tr.inflight_limit,
                        (
                            " collectives in flight on one communicator; the"
                            " newest ones are not watched until some complete"
                        ),
                    ),
                )
            return
    if tr.overflowing:
        tr.overflowing = False
        watch_note(w, "collectives completed; watching new ones again")
    _map_segment(w, t, head // SEG)
    var s = _slot(tr, head)
    var event = _ld(s, S_EVENT)
    if event == 0:
        try:
            event = be()[].vendor.value().event_create(tr.ctx, False)
        except:
            watch_fault(t, -1)
            return
        _st(s, S_EVENT, event)
    var rc = call_fn[EventRecordFn](_watch(w)[].f_record)(event, raw_stream)
    if rc != 0:
        watch_fault(t, Int(rc))
        return
    var v = _ld(s, S_VER)
    _st(s, S_VER, v + 1)
    _st(s, S_SEQ, seq)
    _st(s, S_OP, e.op)
    _st(s, S_NOPS, e.nops)
    _st(s, S_COUNT, e.count)
    _st(s, S_DTYPE, e.dtype)
    _st(s, S_ROOT, e.root)
    _st(s, S_REDOP, e.redop)
    _st(s, S_FP, Int(e.fp & 0x3FFFFFFFFFFFFFFF))
    _st(s, S_ISSUED, perf_counter_ns())
    _st(s, S_CPOS, e.cpos)
    _st(s, S_FR, fr)
    _st(s, S_VER, v + 2)
    _st(tr.words, T_HEAD, head + 1)


def watch_next_cpos(t: Int, op: Int) -> Int:
    """The collective position of one library call, counted per call before
    any group folds it: a compared collective inside a group mixing in
    point-to-point still counts, so grouping never shifts positions. 0 for
    calls that do not compare."""
    if t == 0 or not _compared(op):
        return 0
    ref tr = _tracker(t)[]
    var cpos = _ld(tr.words, T_CPOS) + 1
    _st(tr.words, T_CPOS, cpos)
    return cpos


def watch_test_limit(t: Int, entries: Int):
    """Test hook: at most `entries` collectives in flight (the libraries
    themselves stop near 1024, far below the real limit)."""
    if t != 0:
        _tracker(t)[].inflight_limit = max(min(entries, (NSEG - 1) * SEG), 1)


def watch_group_add(t: Int, var e: Entry):
    """A library call inside an NCCL group: nothing is enqueued before the
    outermost ncclGroupEnd, so it is folded into one entry committed there."""
    ref g = _tracker(t)[].g_entry
    if g.nops == 0:
        g = e.copy()  # a group of one hashes like the call itself
        return
    if g.op != e.op:
        var sr = (
            g.op == OP_SEND or g.op == OP_RECV or g.op == OP_SENDRECV
        ) and (e.op == OP_SEND or e.op == OP_RECV)
        g.op = OP_SENDRECV if sr else OP_GROUP
    if g.cpos == 0:
        g.cpos = e.cpos  # the group's first compared call
    if g.root != e.root:
        g.root = -1
    if g.redop != e.redop:
        g.redop = -1
    g.nops += e.nops
    g.count += e.count
    g.fp = _mix(_mix(g.fp, Int(e.fp & 0xFFFFFFFF)), Int(e.fp >> 32))


def _group_commit(w: Int, t: Int, raw_stream: Int, submitted: Bool):
    ref tr = _tracker(t)[]
    var g = tr.g_entry.copy()
    tr.g_entry = _empty_entry()
    if g.nops != 0 and submitted:
        watch_commit(w, t, raw_stream, g^)  # a claimed tracker skips it


def _drain(w: Int, t: Int, deadline_ns: Int) -> Bool:
    """Retire everything in flight, waiting for it (a destroy's flush).
    False on a driver error or when the deadline passed first."""
    ref tr = _tracker(t)[]
    var query = call_fn[EventQueryFn](_watch(w)[].f_query)
    while True:
        var tail = _ld(tr.words, T_TAIL)
        var head = _ld(tr.words, T_HEAD)
        if tail == head:
            return True
        var b: Boundary
        try:
            with tr.ctx.push_context():
                b = _boundary(tr, query, tail, head)
        except:
            return False
        if b.rc != 0:
            return False
        if b.end > tail:
            var seq = _ld(_slot(tr, b.end - 1), S_SEQ)
            _fr_retired(w, tr, tail, b.end)
            if _cas(tr.words, T_TAIL, tail, b.end):
                _st_max(tr.words, T_DONE, seq)
            continue
        if perf_counter_ns() > deadline_ns:
            return False
        sleep(0.0002)


def watch_drain(w: Int, deadline_ns: Int) -> Bool:
    """Wait until every watched communicator has nothing in flight (a
    destroy's flush), bounded. The watchdog keeps running meanwhile, so a
    hung collective is still reported (and, in abort mode, the process
    exits). A communicator whose tracking failed is not drained: its queue
    proves nothing, so the destroy aborts instead."""
    ref wa = _watch(w)[]
    var drained = True
    var until = deadline_ns
    if _ld(wa.words, W_STOP) != 0:
        until = 0  # nobody would report a hang: one look, no waiting
    for i in range(_ld(wa.words, W_NTRACK)):
        var t = wa.trackers[unsafe_offset=i]
        ref tr = _tracker(t)[]
        if _ld(tr.words, T_STATE) != 0:
            continue  # claimed: an abort or the watchdog owns it
        if _ld(tr.words, T_FAULT) != 0 or _ld(tr.words, T_UNWATCHED) != 0:
            drained = False
            continue
        if not _drain(w, t, until):
            drained = False
    return drained


def watch_note(w: Int, msg: String):
    """One line on stderr, prefixed with the rank."""
    ref wa = _watch(w)[]
    _err(wa, _prefix(wa.rank) + "watchdog: " + msg + "\n")


def watch_deadline_ns(w: Int) -> Int:
    """How long a destroy waits for in-flight collectives: past this, the
    watchdog has reported (and in abort mode exited)."""
    ref wa = _watch(w)[]
    return wa.timeout_ns + wa.ack_wait_ns + ABORT_WAIT_NS


def watch_stop(w: Int):
    """Stop and join the thread. Any number of callers on any threads: one
    joins, the others wait until it has."""
    if w == 0:
        return
    ref wa = _watch(w)[]
    _st(wa.words, W_STOP, 1)
    if _cas(wa.words, W_JOINER, 0, 1):
        if wa.thread != 0:
            _ = external_call["pthread_join", Int32](wa.thread, Int64(0))
        _st(wa.words, W_JOINER, 2)
        return
    while _ld(wa.words, W_JOINER) != 2:
        sleep(0.001)


def _pending(w: Int) -> Bool:
    """Watched work not finished: a library call blocked on the host, a
    collective in flight, a driver error not reported yet."""
    ref wa = _watch(w)[]
    if _ld(wa.words, W_CALL_NS) != 0:
        return True
    for i in range(_ld(wa.words, W_NTRACK)):
        ref tr = _tracker(wa.trackers[unsafe_offset=i])[]
        if _ld(tr.words, T_STATE) != 0 or _ld(tr.words, T_UNWATCHED) != 0:
            continue
        if _ld(tr.words, T_FAULT) != 0 and not tr.fault_reported:
            return True
        if _ld(tr.words, T_TAIL) != _ld(tr.words, T_HEAD):
            return True
    return False


def watch_exit(w: Int):
    """Interpreter exit (Ctrl-C, sys.exit on a hung job included): give the
    watched work EXIT_WAIT_NS to finish, then stop the thread before MAX and
    the driver tear down under it. Work still pending then is reported --
    unless it already was (report mode) -- and the communicators aborted,
    under a reaper in either mode (an abort can stall behind a stalled
    kernel); in abort mode the process exits 1. Never waits for the
    timeout."""
    if w == 0:
        return
    ref wa = _watch(w)[]
    _st(wa.words, W_TEARDOWN, 1)
    var deadline = perf_counter_ns() + EXIT_WAIT_NS
    while perf_counter_ns() < deadline and _ld(wa.words, W_STOP) == 0:
        if not _pending(w):
            break
        sleep(0.01)
    var stopped_before = _ld(wa.words, W_STOP) != 0
    watch_stop(w)
    if stopped_before:
        return
    # The thread is gone: retire what completed meanwhile, then look again.
    for i in range(_ld(wa.words, W_NTRACK)):
        var t = wa.trackers[unsafe_offset=i]
        if _ld(_tracker(t)[].words, T_STATE) == 0:
            _retire_now(w, t)
    if not _pending(w):
        return
    if _exit_report(w):
        return
    # Everything pending was reported already (report mode): only abort.
    watch_note(
        w,
        (
            "exiting with reported collectives still pending; aborting the"
            " communicators"
        ),
    )
    if _arm_reaper(w, ABORT_WAIT_NS):
        _abort_all(w)
        _cancel_reaper(w)


def _exit_report(w: Int) -> Bool:
    """Report the first pending work not reported yet; False when all of it
    was."""
    ref wa = _watch(w)[]
    var now = perf_counter_ns()
    var call_ns = _ld(wa.words, W_CALL_NS)
    if call_ns != 0 and call_ns != wa.reported_call:
        wa.reported_call = call_ns
        _host_timeout(
            w,
            _ld(wa.words, W_CALL_OP),
            _ld(wa.words, W_CALL_DEV),
            call_ns,
            now,
            True,
        )
        return True
    for i in range(_ld(wa.words, W_NTRACK)):
        var t = wa.trackers[unsafe_offset=i]
        ref tr = _tracker(t)[]
        if _ld(tr.words, T_STATE) != 0 or _ld(tr.words, T_UNWATCHED) != 0:
            continue
        var fault = _ld(tr.words, T_FAULT)
        if fault != 0 and not tr.fault_reported:
            _fault(w, t, "an event record", fault, now, True)
            return True
        var tail = _ld(tr.words, T_TAIL)
        var head = _ld(tr.words, T_HEAD)
        if tail == head:
            continue
        var oldest = _read(tr, tail)
        if oldest and oldest.value().seq == tr.reported:
            continue  # reported already
        _device_timeout(w, t, tail, head, now, True)
        return True
    return False


def _free_segment(seg: Int):
    var words = WordsP(unsafe_from_address=seg)
    for j in range(SEG):
        var ev = _ld(words, j * S_WORDS + S_EVENT)
        if ev != 0:
            try:
                be()[].vendor.value().event_destroy(ev)
            except:
                continue  # a driver already torn down released it
    words.unsafe_free()


def watch_destroy(w: Int):
    """Stop the thread, then free every tracker, segment and event."""
    if w == 0:
        return
    groups_forget(w)
    watch_stop(w)
    ref wa = _watch(w)[]
    for i in range(_ld(wa.words, W_NTRACK)):
        var box = _tracker(wa.trackers[unsafe_offset=i])
        for k in range(NSEG):
            var seg = _ld(box[].segs, k)
            if seg == 0:
                continue
            _free_segment(seg)
        for k in range(len(box[].seg_pool)):
            _free_segment(box[].seg_pool[k])
        box[].segs.unsafe_free()
        box[].hist.unsafe_free()
        box[].zero_seg.unsafe_free()
        box[].words.unsafe_free()
        var moved = box.unsafe_take_pointee()
        box.unsafe_free()
        _ = moved^
    if wa.err_fd > 2:
        _ = external_call["close", Int32](Int32(wa.err_fd))
    wa.trackers.unsafe_free()
    wa.words.unsafe_free()
    var wbox = _watch(w)
    var moved = wbox.unsafe_take_pointee()
    wbox.unsafe_free()
    _ = moved^


# --- the watchdog thread -------------------------------------------------------


def _report_fd(fd: Int) -> Int:
    """Where reports to `fd` go: a pipe or a terminal reopened through /proc
    with O_NONBLOCK (a new open file description, so the process's own
    stderr stays blocking); `fd` itself otherwise -- a seekable file, a
    socket, or a pipe the reopen is refused on (a root-owned 0600 pipe in a
    container running as another user). Never no output: see `_out_mode`."""
    if external_call["lseek", Int](Int32(fd), Int(0), Int32(1)) >= 0:
        return fd
    var path = String("/proc/self/fd/") + String(fd)
    comptime FLAGS = 0o1 | 0o4000 | 0o2000000  # WRONLY|NONBLOCK|CLOEXEC
    var nfd = external_call["open", Int32, num_fixed_args=2](
        path.as_c_string_span().ptr(), Int32(FLAGS), Int32(0)
    )
    return Int(nfd) if nfd >= 0 else fd


def _out_mode(fd: Int) -> Int:
    """How `_write` must treat the report fd: as is (a reopened fd or a
    seekable file, whose writes do not block), with send(MSG_DONTWAIT) (a
    socket), or poll(POLLOUT) before each chunk (a blocking fd that could
    not be reopened: a chunk of at most PIPE_BUF goes through once the pipe
    polls writable, another writer racing for the room being the residual
    risk; in abort mode the reaper bounds it)."""
    if fd != 2 or external_call["lseek", Int](Int32(fd), Int(0), Int32(1)) >= 0:
        return OUT_PLAIN
    return OUT_SEND if _is_socket(fd) else OUT_POLL


def _is_socket(fd: Int) -> Bool:
    var kind = Int32(0)
    var size = UInt32(4)
    return (
        external_call["getsockopt", Int32](
            Int32(fd), Int32(1), Int32(3), Pointer(to=kind), Pointer(to=size)
        )
        == 0
    )  # SOL_SOCKET, SO_TYPE


def _err(wa: Watch, s: String):
    _write(wa.err_fd, s, wa.err_mode)


def _write(fd: Int, s: String, mode: Int = OUT_PLAIN):
    """Bounded and best effort, never longer than WRITE_WAIT_NS: PIPE_BUF
    chunks to a non-blocking fd (or sent with MSG_DONTWAIT to a socket); a
    full one (EAGAIN) waits in poll(2) for room, an interrupted write
    retries, and any other error, hang-up or invalid fd ends the write -- a
    report must never keep the process alive. fd -1: no output."""
    if fd < 0:
        return
    var p = Int(s.unsafe_ptr())
    var left = s.byte_length()
    var deadline = perf_counter_ns() + WRITE_WAIT_NS
    var pfd = unsafe_alloc[Int32](2)
    while left > 0:
        if mode == OUT_POLL and not _writable(pfd, fd, deadline):
            break
        var buf = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=p)
        var n: Int
        if mode == OUT_SEND:
            n = external_call["send", Int](
                Int32(fd), buf, min(left, PIPE_BUF), Int32(0x40)
            )  # MSG_DONTWAIT
        else:
            n = external_call["write", Int](fd, buf, min(left, PIPE_BUF))
        if n > 0:
            p += n
            left -= n
            continue
        var err = get_errno().value if n < 0 else Int32(0)
        if err == EINTR:
            continue
        if n < 0 and err != EAGAIN:
            break
        var wait = deadline - perf_counter_ns()
        if wait <= 0:
            break
        pfd[unsafe_offset=0] = Int32(fd)
        pfd[unsafe_offset=1] = 4  # POLLOUT in the low half, revents 0
        var ready = external_call["poll", Int32](
            pfd, UInt64(1), Int32(wait // 1_000_000 + 1)
        )
        if ready > 0 and (pfd[unsafe_offset=1] >> 16) & 0x38 != 0:
            break  # POLLERR / POLLHUP / POLLNVAL
    pfd.unsafe_free()


def _writable(
    pfd: Pointer[Int32, MutUntrackedOrigin], fd: Int, deadline: Int
) -> Bool:
    """poll(POLLOUT) until the fd takes a chunk, or the deadline."""
    while True:
        var wait = deadline - perf_counter_ns()
        if wait <= 0:
            return False
        pfd[unsafe_offset=0] = Int32(fd)
        pfd[unsafe_offset=1] = 4  # POLLOUT in the low half, revents 0
        var ready = external_call["poll", Int32](
            pfd, UInt64(1), Int32(wait // 1_000_000 + 1)
        )
        if ready > 0:
            return (pfd[unsafe_offset=1] >> 16) & 0x38 == 0  # no ERR/HUP/NVAL


def _secs(ns: Int) -> String:
    var tenths = ns // 100_000_000
    return String(tenths // 10) + "." + String(tenths % 10) + " s"


def _prefix(rank: Int) -> String:
    return "[rank " + String(rank) + "] "


def _describe(e: Entry) -> String:
    var s = (
        String(op_name(e.op))
        + " "
        + String(e.count)
        + " x "
        + String(dtype_name(e.dtype))
    )
    if e.root >= 0:
        var peer = e.op == OP_SEND or e.op == OP_RECV or e.op == OP_SENDRECV
        s += (" peer " if peer else " root ") + String(e.root)
    if e.redop >= 0:
        s += " " + String(redop_name(e.redop))
    if e.nops > 1:
        s += " (" + String(e.nops) + " calls in one NCCL group)"
    return s


def _watch_main(arg: OpaquePointer[MutAnyOrigin]) abi("C"):
    var w = Int(arg)
    while _ld(_watch(w)[].words, W_STOP) == 0:
        sleep(POLL_S)
        if _ld(_watch(w)[].words, W_STOP) != 0 or not _poll(w):
            break


def _reaper_main(arg: OpaquePointer[MutAnyOrigin]) abi("C"):
    """Exits at the deadline, writing nothing first: a write is exactly what
    may be stuck. Reads only its own cell [deadline, cancelled], never the
    watch, which may be freed while it sleeps."""
    var cell = WordsP(unsafe_from_address=Int(arg))
    _sleep_until(_ld(cell, 0))
    if _ld(cell, 1) == 0:
        external_call["_exit", NoneType](Int32(1))


def _arm_reaper(w: Int, ns: Int) -> Bool:
    """A thread that `_exit(1)`s in `ns` unless `_cancel_reaper` runs first.
    Its cell is leaked on purpose (16 bytes per arming, which is rare): the
    thread is never joined and may outlive the watch."""
    var cell = _words(2)
    _st(cell, 0, perf_counter_ns() + ns)
    _watch(w)[].reaper_cell = Int(cell)
    return _spawn(_reaper_main, Int(cell)) != 0


def _cancel_reaper(w: Int):
    """Unless another thread is still inside a library call (an abort under
    the in-call mark that may hang): then the deadline stays armed."""
    ref wa = _watch(w)[]
    if wa.reaper_cell != 0 and _ld(wa.words, W_CALL_NS) == 0:
        _st(WordsP(unsafe_from_address=wa.reaper_cell), 1, 1)


def _abort_all(w: Int):
    var abort = call_fn[CommFn](_watch(w)[].f_abort)
    ref wa = _watch(w)[]
    for i in range(_ld(wa.words, W_NTRACK)):
        var t = wa.trackers[unsafe_offset=i]
        if watch_claim(t):
            _ = abort(_tracker(t)[].handle)


def _poll(w: Int) -> Bool:
    """One round; False stops the thread (the process is tearing down)."""
    ref wa = _watch(w)[]
    var now = perf_counter_ns()
    var call_ns = _ld(wa.words, W_CALL_NS)
    if wa.reported_call != 0 and call_ns != wa.reported_call:
        wa.reported_call = 0  # report mode: the blocked call returned
        if wa.notify_fd >= 0:
            _write(wa.notify_fd, "resolved tid=-1\n")
    if call_ns != 0 and now - call_ns > wa.timeout_ns:
        var op = _ld(wa.words, W_CALL_OP)
        var device = _ld(wa.words, W_CALL_DEV)
        if _ld(wa.words, W_CALL_NS) == call_ns and call_ns != wa.reported_call:
            wa.reported_call = call_ns
            _host_timeout(w, op, device, call_ns, now)
    for i in range(_ld(wa.words, W_NTRACK)):
        if not _poll_tracker(w, wa.trackers[unsafe_offset=i], now):
            return False
    return True


def _resolve(wa: Watch, mut tr: Tracker):
    """Take the communicator's published incident out of the rank's index
    (report mode: it completed, or the communicator was claimed)."""
    if tr.published and wa.notify_fd >= 0:
        _write(wa.notify_fd, "resolved tid=" + String(tr.id) + "\n")
    tr.published = False


def _teardown(rc: Int32) -> Bool:
    return (
        rc == DEINITIALIZED or rc == INVALID_CONTEXT or rc == CONTEXT_DESTROYED
    )


def _poll_tracker(w: Int, t: Int, now: Int) -> Bool:
    ref wa = _watch(w)[]
    ref tr = _tracker(t)[]
    if _ld(tr.words, T_STATE) != 0 or _ld(tr.words, T_UNWATCHED) != 0:
        if _ld(tr.words, T_STATE) != 0:
            tr.reported = 0  # claimed (aborted, destroyed): retire its record
            _resolve(wa, tr)
        return True
    var fault = _ld(tr.words, T_FAULT)
    if fault != 0:
        _fault(w, t, "an event record", fault, now)
        return True
    if tr.reported != 0 and _ld(tr.words, T_DONE) >= tr.reported:
        tr.reported = 0  # report mode: the reported collective completed
        _resolve(wa, tr)
    var tail = _ld(tr.words, T_TAIL)
    var head = _ld(tr.words, T_HEAD)
    if tail == head:
        return True
    var b: Boundary
    try:
        var query = call_fn[EventQueryFn](wa.f_query)
        with tr.ctx.push_context():
            b = _boundary(tr, query, tail, head)
        tr.push_failures = 0
    except:
        # MAX and the driver tear down at interpreter exit: that is a stop,
        # not a fault (the atexit hook normally stops the thread first).
        if _ld(wa.words, W_STOP) != 0 or _tearing_down(wa):
            return False
        tr.push_failures += 1
        if tr.push_failures >= PUSH_FAILURES:
            _fault(w, t, "a context push", -1, now)
        return True
    var rc = b.rc
    if _teardown(rc):
        if (
            rc != DEINITIALIZED
            and _ld(wa.words, W_STOP) == 0
            and not _tearing_down(wa)
        ):
            watch_note(
                w,
                String(
                    "stopping: driver error ",
                    rc,
                    " (the device context is gone) on device ",
                    tr.device,
                ),
            )
        return False
    if _ld(tr.words, T_TAIL) != tail:
        return True  # the issuer retired entries meanwhile: next round
    if b.end > tail:
        # Snapshot what the report needs of the entries retired now before
        # publishing the tail: once it moves, the producer may reuse them.
        var done = List[Entry]()
        for k in range(max(b.end - HISTORY, tail), b.end):
            var e = _read(tr, k)
            if e:
                done.append(e.value().copy())
        var last_seq = _ld(
            _slot(tr, b.end - 1), S_SEQ
        )  # before: reusable after
        _fr_retired(w, tr, tail, b.end)
        if not _cas(tr.words, T_TAIL, tail, b.end):
            return True  # a drain retired entries: look again next round
        _st_max(tr.words, T_DONE, last_seq)
        for k in range(len(done)):
            tr.hist[unsafe_offset=tr.hist_n % HISTORY] = done[k].copy()
            tr.hist_n += 1
        tail = b.end
    if rc != 0 and rc != NOT_READY:
        _fault(w, t, "an event query", Int(rc), now)
        return True
    if tail == head:
        return True
    var oldest = _read(tr, tail)
    if not oldest or _ld(tr.words, T_TAIL) != tail:
        return True
    if (
        now - oldest.value().issued_ns > wa.timeout_ns
        and oldest.value().seq != tr.reported
    ):
        _device_timeout(w, t, tail, head, now)
    return True


def _pending(tr: Tracker, tail: Int, head: Int) -> List[Entry]:
    """Up to HISTORY in-flight entries from the oldest; empty when a slot
    was being rewritten or the tail moved (the report is then moot)."""
    var out = List[Entry]()
    for i in range(tail, min(head, tail + HISTORY)):
        var e = _read(tr, i)
        if not e:
            return List[Entry]()
        out.append(e.value().copy())
    if _ld(tr.words, T_TAIL) != tail:
        return List[Entry]()
    return out^


def _history(
    tr: Tracker, pending: List[Entry], more: Int, now: Int, rank: Int
) -> String:
    var p = _prefix(rank)
    var s = (
        p
        + "  last collectives issued on device "
        + String(tr.device)
        + ", oldest first:\n"
    )
    for k in range(max(tr.hist_n - HISTORY, 0), tr.hist_n):
        ref e = tr.hist[unsafe_offset=k % HISTORY]
        s += p + "    #" + String(e.seq) + " " + _describe(e.copy())
        s += "  completed\n"
    for k in range(len(pending)):
        s += p + "    #" + String(pending[k].seq) + " "
        s += _describe(pending[k].copy())
        s += "  pending for " + _secs(now - pending[k].issued_ns)
        s += "  <-- timed out\n" if k == 0 else "\n"
    if more > 0:
        s += p + "    ... and " + String(more) + " more pending\n"
    return s


def _line(
    wa: Watch,
    tr: Tracker,
    e: Entry,
    age_ns: Int,
    kind: StaticString,
    at_exit: Bool = False,
    fr: Int = 0,
) -> String:
    """The machine-readable line for the Python desync reporter: `id` is the
    incident, `tid` the communicator (-1: a call blocked on the host)."""
    return (
        "id="
        + String(wa.incident)
        + " tid="
        + String(-1 if kind == "host" else tr.id)
        + " seq="
        + String(e.seq)
        + " cpos="
        + String(e.cpos)
        + " done="
        + String(_ld(tr.words, T_DONE))
        + " last="
        + String(_ld(tr.words, T_SEQ))
        + " op="
        + String(op_name(e.op))
        + " count="
        + String(e.count)
        + " dtype="
        + String(dtype_name(e.dtype))
        + " root="
        + String(e.root)
        + " redop="
        + String(e.redop)
        + " nops="
        + String(e.nops)
        + " fp="
        + String(e.fp)
        + " device="
        + String(tr.device)
        + " pending_ms="
        + String(age_ns // 1_000_000)
        + " where="
        + String(kind)
        + " fr="
        + String(fr)
        + (" exit=1" if at_exit else "")
        + "\n"
    )


def _device_timeout(
    w: Int, t: Int, tail: Int, head: Int, now: Int, at_exit: Bool = False
):
    ref wa = _watch(w)[]
    ref tr = _tracker(t)[]
    var pending = _pending(tr, tail, head)
    if len(pending) == 0:
        return
    tr.reported = pending[0].seq
    var age = now - pending[0].issued_ns
    var msg = _prefix(wa.rank)
    if at_exit:
        msg += (
            "Watchdog: the process is exiting with collective #"
            + String(pending[0].seq)
            + " ("
            + _describe(pending[0].copy())
            + ") on device "
            + String(tr.device)
            + " still in flight, "
            + _secs(age)
            + " after it was issued.\n"
        )
    else:
        msg += (
            "Watchdog caught collective operation timeout: collective #"
            + String(pending[0].seq)
            + " ("
            + _describe(pending[0].copy())
            + ") on device "
            + String(tr.device)
            + " has not completed "
            + _secs(age)
            + " after it was issued (timeout "
            + _secs(wa.timeout_ns)
            + ", the timeout= of init_process_group).\n"
        )
    msg += _history(tr, pending, head - tail - len(pending), now, wa.rank)
    wa.incident += 1
    tr.published = True
    var line = _line(
        wa,
        tr,
        pending[0].copy(),
        age,
        "device",
        at_exit,
        _ld(_slot(tr, tail), S_FR),
    )
    _fail(w, msg^, line, at_exit)


def _host_timeout(
    w: Int, op: Int, device: Int, call_ns: Int, now: Int, at_exit: Bool = False
):
    ref wa = _watch(w)[]
    var msg = _prefix(wa.rank)
    if at_exit:
        msg += (
            "Watchdog: the process is exiting while the "
            + String(op_name(op))
            + " call has been blocked on the host inside the collectives"
            " library for "
            + _secs(now - call_ns)
            + ".\n"
        )
    else:
        msg += (
            "Watchdog caught collective operation timeout: the "
            + String(op_name(op))
            + " call has been blocked on the host inside the collectives"
            " library for "
            + _secs(now - call_ns)
            + " (timeout "
            + _secs(wa.timeout_ns)
            + ", the timeout= of init_process_group; a first-use kernel"
            " compile inside the call counts too).\n"
        )
    wa.incident += 1
    var line = String("")
    for i in range(_ld(wa.words, W_NTRACK)):
        ref tr = _tracker(wa.trackers[unsafe_offset=i])[]
        if _ld(tr.words, T_STATE) != 0:
            continue
        if device >= 0 and tr.device != device:
            continue
        var e = _empty_entry()
        e.op = op
        e.seq = _ld(tr.words, T_SEQ) + 1  # the call being made (approx.)
        var tail = _ld(tr.words, T_TAIL)
        var head = _ld(tr.words, T_HEAD)
        msg += _history(tr, _pending(tr, tail, head), 0, now, wa.rank)
        line = _line(wa, tr, e^, now - call_ns, "host", at_exit)
        break
    _fail(w, msg^, line, at_exit)


def _fault(
    w: Int,
    t: Int,
    what: StaticString,
    rc: Int,
    now: Int,
    at_exit: Bool = False,
):
    """A driver error while watching: the device is in an error state, so
    peers may wait on this rank forever. Same policy as a timeout; in report
    mode the communicator is left unwatched, and its destroy aborts it."""
    ref wa = _watch(w)[]
    ref tr = _tracker(t)[]
    watch_fault(t, rc)
    if tr.fault_reported:
        return
    tr.fault_reported = True
    var msg = (
        _prefix(wa.rank)
        + "Watchdog: "
        + String(what)
        + " on device "
        + String(tr.device)
        + " failed with driver error "
        + String(rc)
        + "; the device is in an error state and collectives on it cannot"
        " complete.\n"
    )
    wa.incident += 1
    var e = _empty_entry()
    e.seq = _ld(tr.words, T_DONE) + 1
    tr.published = True
    var line = _line(wa, tr, e^, 0, "error", at_exit)
    _fail(w, msg^, line, at_exit)
    _st(tr.words, T_UNWATCHED, 1)


def _fail(w: Int, var msg: String, line: String, at_exit: Bool = False):
    """Report, give the Python desync reporter its bounded window, then abort
    and exit -- or, with the "report" action, return and keep watching. In
    abort mode a reaper thread armed first `_exit`s at a deadline whatever
    blocks below; when it cannot be started, the process exits at once."""
    ref wa = _watch(w)[]
    var p = _prefix(wa.rank)
    var reaper = False
    if wa.action == ACTION_ABORT or at_exit:
        var wait = min(
            wa.ack_wait_ns, EXIT_ACK_NS
        ) if at_exit else wa.ack_wait_ns
        reaper = _arm_reaper(w, wait + ABORT_WAIT_NS)
        if not reaper and wa.action == ACTION_ABORT:
            external_call["_exit", NoneType](Int32(1))  # nothing else bounds it
    msg += (
        p
        + "  Every rank must issue the same collectives in the same order"
        " with the same arguments; a rank that issued a different or an"
        " extra one, or stopped issuing, is the usual cause.\n"
    )
    if wa.action == ACTION_ABORT:
        msg += (
            p
            + "  Aborting the communicators and exiting with status 1"
            " (TORCH_MOJO_BACKEND_PG_WATCHDOG=report only reports).\n"
        )
    elif at_exit:
        msg += p + "  Aborting the communicators so the process can exit.\n"
    _err(wa, msg)
    if wa.notify_fd >= 0 and line.byte_length() > 0:
        _write(wa.notify_fd, line)
        var wait = min(
            wa.ack_wait_ns, EXIT_ACK_NS
        ) if at_exit else wa.ack_wait_ns
        _wait_ack(wa.ack_fd, wait, wa.incident)
    if wa.action != ACTION_ABORT:
        if at_exit and reaper:  # an abort can stall behind a stalled kernel
            _abort_all(w)
            _cancel_reaper(w)
        return
    # ncclCommAbort releases the device spins. Exit right behind it: the
    # hung thread it releases must not run on.
    _abort_all(w)
    external_call["_exit", NoneType](Int32(1))


def ack_byte(incident: Int) -> UInt8:
    """What the reporter writes back for an incident (never 0)."""
    return UInt8(incident % 255 + 1)


def _wait_ack(fd: Int, wait_ns: Int, incident: Int):
    """Wait, bounded, for this incident's byte on the ack pipe; bytes of
    earlier incidents acknowledged late are skipped. Bounded because the
    reporter needs the GIL, which the hung thread may hold."""
    if fd < 0:
        return
    var want = ack_byte(incident)
    var pfd = unsafe_alloc[Int32](2)
    var deadline = perf_counter_ns() + wait_ns
    while True:
        var left = deadline - perf_counter_ns()
        if left <= 0:
            break
        pfd[unsafe_offset=0] = Int32(fd)
        pfd[unsafe_offset=1] = 1  # POLLIN in the low half, revents 0
        var rc = external_call["poll", Int32](
            pfd, UInt64(1), Int32(left // 1_000_000 + 1)
        )
        if rc <= 0:
            continue  # timeout (loop exits) or EINTR
        var byte = UInt8(0)
        var n = external_call["read", Int](fd, Pointer(to=byte), 1)
        if n <= 0 or byte == want:
            break  # this incident's ack, or the reporter is gone
    pfd.unsafe_free()
