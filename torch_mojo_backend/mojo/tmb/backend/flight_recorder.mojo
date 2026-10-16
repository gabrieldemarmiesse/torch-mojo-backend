"""Flight recorder of the process group: the last TORCH_FR_BUFFER_SIZE
collectives of the process, completed ones included, for the dumps
`torch_mojo_backend/distributed/flight_recorder.py` writes in the format of
PyTorch's flight recorder (torch/csrc/distributed/c10d/FlightRecorder.hpp).

One ring per process, like torch's. The watchdog's commit (`watch_commit`)
writes an entry for every collective it commits, and `PG.track` one for
every operation of a coalescing block; the watchdog stamps an entry
completed when it retires the collective. The dump reads without the
backend mutex (a hung thread may hold it), so every slot is seqlocked and
every field an atomic word. A slot has two kinds of writers -- the producer
(under the backend mutex) and the stampers (the watchdog, an issuer
retiring) -- so its version doubles as a lock: a writer moves it from even
to odd by compare-exchange, checks the record id under it, writes, and
moves it to the next even value. A stamp can therefore only land on the
entry it is for, whatever the ring did meanwhile.

The Python side describes the torch-level operation in flight: every
pg.mojo entry takes a tag word, passed down to the entries it causes -- a
record id, which the dump joins with its Python description, and the
coalescing structure (member, first, end, point-to-point). A group's entry
is recorded at its process group's group_end, with that call's tag, even
when the commit it completes with comes later (an enclosing group).
"""
from std.atomic import Atomic, Ordering
from std.ffi import _get_global_or_null, external_call
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns

comptime FR_GLOBAL = "TMB_PG_FLIGHT_RECORDER"

# Tag word (flight_recorder.py `Recorder.begin` builds it).
comptime TAG_MASK = (1 << 40) - 1  # the Python record's id, 0 if none
comptime TAG_MEMBER = 1 << 40  # one operation of a coalescing block
comptime TAG_FIRST = 1 << 41  # ...its first
comptime TAG_END = 1 << 42  # the block's end
comptime TAG_P2P = 1 << 43  # point-to-point (overrides the native op)

# Entry flags.
comptime FL_P2P = 1
comptime FL_MEMBER = 2
comptime FL_END = 4
comptime FL_UNTRACKED = 8  # the watchdog does not track it: no stamp comes

# Header words.
comptime H_SIZE = 0
comptime H_NEXT = 1  # entries recorded so far: the next record id
comptime H_WORDS = 2

# Slot words; a snapshot copies them as they are (flight_recorder.py).
comptime F_VER = 0  # odd while a writer holds the slot
comptime F_ID = 1  # record id + 1, 0 while empty
comptime F_PG = 2
comptime F_CSEQ = 3
comptime F_PSEQ = 4
comptime F_OPID = 5
comptime F_OP = 6  # watchdog.mojo OP_*
comptime F_COUNT = 7
comptime F_DTYPE = 8  # ncclDataType_t
comptime F_ROOT = 9  # root or peer (group rank), -1 if none
comptime F_NOPS = 10
comptime F_FLAGS = 11
comptime F_CREATED = 12  # CLOCK_MONOTONIC ns
comptime F_THREAD = 13  # pthread_self
comptime F_TAG = 14
comptime F_COMM = 15  # the communicator's tracker (watchdog.mojo), 0 if none
comptime F_DONE = 16  # CLOCK_MONOTONIC ns of completion, 0 while pending
comptime F_WORDS = 17

# Per process group (`fr_pg`), producer-only.
comptime P_RING = 0
comptime P_ID = 1
comptime P_CSEQ = 2
comptime P_PSEQ = 3
comptime P_OPID = 4
comptime P_GROUP_C = 5  # a coalescing block bumped the collective seq
comptime P_GROUP_P = 6  # ...the point-to-point seq
comptime P_WORDS = 7

comptime FrP = Pointer[Int64, MutUntrackedOrigin]


@always_inline
def _ld(p: FrP, i: Int) -> Int:
    return Int(
        Atomic[Scalar[DType.int64]].load[ordering=Ordering.ACQUIRE](
            p.unsafe_offset(i)
        )
    )


@always_inline
def _st(p: FrP, i: Int, v: Int):
    Atomic[Scalar[DType.int64]].store[ordering=Ordering.RELEASE](
        p.unsafe_offset(i), Int64(v)
    )


@always_inline
def _cas(p: FrP, i: Int, old: Int, new: Int) -> Bool:
    var expected = Int64(old)
    return Atomic[Scalar[DType.int64]].compare_exchange[
        success_ordering=Ordering.ACQUIRE_RELEASE,
        failure_ordering=Ordering.ACQUIRE,
    ](p.unsafe_offset(i), expected, Int64(new))


def _zeroed(n: Int) -> FrP:
    var p = unsafe_alloc[Int64](n)
    for i in range(n):
        p[unsafe_offset=i] = 0
    return FrP(unsafe_from_address=Int(p))


@always_inline
def _slot(ring: FrP, rid: Int) -> FrP:
    return FrP(
        unsafe_from_address=Int(ring)
        + (H_WORDS + (rid % _ld(ring, H_SIZE)) * F_WORDS) * 8
    )


@always_inline
def _pg(fp: Int) -> FrP:
    return FrP(unsafe_from_address=fp)


@always_inline
def _ring_of(fp: Int) -> FrP:
    return FrP(unsafe_from_address=_ld(_pg(fp), P_RING))


def _lock(s: FrP) -> Int:
    """Take the slot (its version odd); the even version it had."""
    var spins = 0
    while True:
        var v = _ld(s, F_VER)
        if v & 1 == 0 and _cas(s, F_VER, v, v + 1):
            return v
        spins += 1
        if spins > 64:  # the holder was preempted: let it run
            _ = external_call["sched_yield", Int32]()


def fr_ring() -> Int:
    """The process's ring, 0 when none was made."""
    var p = _get_global_or_null(FR_GLOBAL)
    if p:
        return Int(p.value())
    return 0


def _new_ring(size: Int) -> FrP:
    var words = _zeroed(H_WORDS + size * F_WORDS)
    _st(words, H_SIZE, size)
    return words


def fr_init(size: Int) -> Int:
    """The ring (its header address), made on the first call with `size`
    entries; later sizes are ignored."""
    var ring = fr_ring()
    if ring != 0 or size <= 0:
        return ring
    var words = _new_ring(size)
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(FR_GLOBAL), words.unsafe_bitcast[NoneType]()
    )
    return Int(words)


def fr_pg(ring: Int, pg_id: Int) -> Int:
    """A process group's recording state (never freed: a few words)."""
    var p = _zeroed(P_WORDS)
    _st(p, P_RING, ring)
    _st(p, P_ID, pg_id)
    return Int(p)


def _record(
    fp: Int,
    op: Int,
    nops: Int,
    count: Int,
    dtype: Int,
    root: Int,
    p2p_op: Bool,
    untracked: Bool,
    comm: Int,
    tag_word: Int,
) -> Int:
    """Seq ids as ProcessGroupNCCL counts them: `collective_seq_id` once per
    collective or coalescing block, `p2p_seq_id` once per point-to-point op
    or block, `op_id` per operation. Returns record id + 1."""
    var pg = _pg(fp)
    var ring = _ring_of(fp)
    var tag = tag_word
    if tag & TAG_END and _ld(pg, P_GROUP_C) == 0 and _ld(pg, P_GROUP_P) == 0:
        # Another group's block ended (this group's commit rides its
        # group_end): an ordinary entry, which that record does not describe.
        tag = 0
    var p2p = (tag & TAG_P2P != 0) if tag & TAG_MASK != 0 else p2p_op
    var flags = FL_P2P if p2p else 0
    if untracked:
        flags |= FL_UNTRACKED
    if tag & TAG_MEMBER:
        flags |= FL_MEMBER
        if tag & TAG_FIRST:
            _st(pg, P_GROUP_C, 0)
            _st(pg, P_GROUP_P, 0)
        _st(pg, P_OPID, _ld(pg, P_OPID) + 1)
        var bumped = P_GROUP_P if p2p else P_GROUP_C
        if _ld(pg, bumped) == 0:
            _st(pg, bumped, 1)
            var seq = P_PSEQ if p2p else P_CSEQ
            _st(pg, seq, _ld(pg, seq) + 1)
    elif tag & TAG_END:
        flags |= FL_END
        _st(pg, P_GROUP_C, 0)
        _st(pg, P_GROUP_P, 0)
    else:
        _st(pg, P_OPID, _ld(pg, P_OPID) + 1)
        var seq = P_PSEQ if p2p else P_CSEQ
        _st(pg, seq, _ld(pg, seq) + 1)
    var rid = _ld(ring, H_NEXT)
    var s = _slot(ring, rid)
    var v = _lock(s)
    _st(s, F_ID, rid + 1)
    _st(s, F_PG, _ld(pg, P_ID))
    _st(s, F_CSEQ, _ld(pg, P_CSEQ))
    _st(s, F_PSEQ, _ld(pg, P_PSEQ))
    _st(s, F_OPID, _ld(pg, P_OPID))
    _st(s, F_OP, op)
    _st(s, F_COUNT, count)
    _st(s, F_DTYPE, dtype)
    _st(s, F_ROOT, root)
    _st(s, F_NOPS, nops)
    _st(s, F_FLAGS, flags)
    _st(s, F_CREATED, perf_counter_ns())
    _st(s, F_THREAD, external_call["pthread_self", Int]())
    _st(s, F_TAG, tag & TAG_MASK)
    _st(s, F_COMM, comm)
    _st(s, F_DONE, 0)
    _st(s, F_VER, v + 2)
    _st(ring, H_NEXT, rid + 1)
    return rid + 1


def fr_commit(
    fp: Int,
    op: Int,
    nops: Int,
    count: Int,
    dtype: Int,
    root: Int,
    p2p_op: Bool,
    comm: Int,
    tag: Int,
) -> Int:
    """The entry of a collective the watchdog commits on tracker `comm` (a
    call, or a group's calls at its group_end), described by `tag`; record
    id + 1, 0 when not recording. Under the backend mutex."""
    if fp == 0:
        return 0
    return _record(fp, op, nops, count, dtype, root, p2p_op, False, comm, tag)


def fr_untracked(fp: Int, rid1: Int):
    """The watchdog dropped entry `rid1` (a full queue, a fault, a failed
    event): no stamp will come. Under the backend mutex."""
    if fp == 0 or rid1 == 0:
        return
    var s = _slot(_ring_of(fp), rid1 - 1)
    var v = _lock(s)
    if _ld(s, F_ID) == rid1:
        _st(s, F_FLAGS, _ld(s, F_FLAGS) | FL_UNTRACKED)
    _st(s, F_VER, v + 2)


def fr_member(
    fp: Int,
    op: Int,
    count: Int,
    dtype: Int,
    root: Int,
    p2p_op: Bool,
    comm: Int,
    tag: Int,
):
    """A library call inside an NCCL group: its own entry only when it is an
    operation of a coalescing block (`tag`). Like ProcessGroupNCCL's, these
    entries are never stamped; the block's end entry is. Under the backend
    mutex."""
    if fp != 0 and tag & TAG_MEMBER:
        _ = _record(fp, op, 1, count, dtype, root, p2p_op, False, comm, tag)


def fr_done(fp: Int, rid1: Int, now: Int):
    """Stamp entry `rid1` (record id + 1) completed at `now`. Any thread:
    the id is checked under the slot's lock, so a late stamp of a slot the
    ring reused is dropped; the first stamp of an entry stays."""
    if fp == 0 or rid1 == 0:
        return
    var s = _slot(_ring_of(fp), rid1 - 1)
    if _ld(s, F_ID) != rid1:
        return  # reused already: no need to lock
    var v = _lock(s)
    if _ld(s, F_ID) == rid1 and _ld(s, F_DONE) == 0:
        _st(s, F_DONE, max(now, 1))
    _st(s, F_VER, v + 2)


def _release_all(ring: FrP):
    for i in range(_ld(ring, H_SIZE)):
        var s = _slot(ring, i)
        var v = _ld(s, F_VER)
        if v & 1:
            _st(s, F_VER, v + 1)


def fr_after_fork():
    """In a forked child: a slot whose writer was another thread of the
    parent would stay locked forever; release every one."""
    var r = fr_ring()
    if r != 0:
        _release_all(FrP(unsafe_from_address=r))


def fr_snapshot(buf: Int, cap: Int) -> Int:
    """Copy the ring's entries, oldest first, into `buf` (`cap` entries of
    F_WORDS words); the number copied. Lock-free: an entry being written is
    retried (yielding to its writer after a few tries), and left out only
    if it stays busy."""
    var r = fr_ring()
    if r == 0:
        return 0
    return _snapshot(FrP(unsafe_from_address=r), buf, cap)


def _snapshot(ring: FrP, buf: Int, cap: Int) -> Int:
    var out = FrP(unsafe_from_address=buf)
    var size = _ld(ring, H_SIZE)
    var next = _ld(ring, H_NEXT)
    var n = 0
    for rid in range(max(next - size, 0), next):
        if n >= cap:
            break
        var s = _slot(ring, rid)
        for attempt in range(64):
            if attempt >= 8:
                _ = external_call["sched_yield", Int32]()
            var v = _ld(s, F_VER)
            if v & 1:
                continue
            var o = out.unsafe_offset(n * F_WORDS)
            for k in range(F_WORDS):
                o[unsafe_offset=k] = Int64(_ld(s, k))
            if _ld(s, F_VER) != v:
                continue
            if Int(o[unsafe_offset=F_ID]) == rid + 1:
                n += 1
            break  # consistent: kept, or already a newer entry's
    return n


def fr_selftest() -> Int:
    """The ring's rules on a private ring of 4 slots: 0 when they hold, else
    the number of the first check that failed (test hook)."""
    var ring = _new_ring(4)
    var fp = fr_pg(Int(ring), 7)
    var r1 = fr_commit(fp, 1, 1, 8, 7, -1, False, 11, 0)
    fr_done(fp, r1, 100)
    if _ld(_slot(ring, r1 - 1), F_DONE) != 100:
        return 1
    fr_done(fp, r1, 200)  # the first stamp stays
    if _ld(_slot(ring, r1 - 1), F_DONE) != 100:
        return 2
    var r = r1
    for _ in range(4):  # lap the ring: r1's slot is the newest entry's
        r = fr_commit(fp, 1, 1, 8, 7, -1, False, 11, 0)
    fr_done(fp, r1, 300)  # a late stamp of the slot's previous entry
    if _ld(_slot(ring, r - 1), F_DONE) != 0:
        return 3
    fr_done(fp, r, 400)
    if _ld(_slot(ring, r - 1), F_DONE) != 400:
        return 4
    if _ld(_slot(ring, r - 1), F_CSEQ) != 5:
        return 5
    # A coalescing block of two sends, tagged: one p2p seq, op ids per op.
    fr_member(fp, 6, 8, 7, 1, True, 11, 9 | TAG_MEMBER | TAG_FIRST | TAG_P2P)
    fr_member(fp, 6, 8, 7, 1, True, 11, 9 | TAG_MEMBER | TAG_P2P)
    var end = fr_commit(fp, 8, 2, 16, 7, -1, True, 11, 10 | TAG_END | TAG_P2P)
    var e = _slot(ring, end - 1)
    if _ld(e, F_PSEQ) != 1 or _ld(e, F_FLAGS) != FL_END | FL_P2P:
        return 6
    if _ld(_slot(ring, end - 2), F_FLAGS) != FL_MEMBER | FL_P2P:
        return 7
    if _ld(e, F_OPID) != 7 or _ld(e, F_TAG) != 10:
        return 8
    # An END tag reaching a group with no open block: an ordinary entry.
    var other = fr_pg(Int(ring), 8)
    var plain = fr_commit(
        other, 1, 1, 8, 7, -1, False, 12, 10 | TAG_END | TAG_P2P
    )
    var p = _slot(ring, plain - 1)
    if _ld(p, F_FLAGS) != 0 or _ld(p, F_TAG) != 0 or _ld(p, F_CSEQ) != 1:
        return 9
    # A slot left locked (a forked child's view of a parent's writer).
    _st(p, F_VER, _ld(p, F_VER) + 1)
    _release_all(ring)
    if _ld(p, F_VER) & 1:
        return 10
    fr_untracked(fp, plain)
    if _ld(p, F_FLAGS) != FL_UNTRACKED or _ld(p, F_VER) & 1:
        return 11
    var buf = unsafe_alloc[Int64](4 * F_WORDS)
    var n = _snapshot(ring, Int(buf), 4)
    var first = Int(buf[unsafe_offset=F_ID])
    buf.unsafe_free()
    _pg(fp).unsafe_free()
    _pg(other).unsafe_free()
    ring.unsafe_free()
    if n != 4 or first != plain - 3:
        return 12
    return 0
