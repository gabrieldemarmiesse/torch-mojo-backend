"""Flight recorder of the process group: the last TORCH_FR_BUFFER_SIZE
collectives of the process, completed ones included, for the dumps
`torch_mojo_backend/distributed/flight_recorder.py` writes in the format of
PyTorch's flight recorder (torch/csrc/distributed/c10d/FlightRecorder.hpp).

One ring per process, like torch's. The watchdog's commit (`watch_commit`)
writes an entry for every collective it tracks, and `PG.track` one for every
member of a coalescing block; the watchdog stamps an entry completed when it
retires the collective. Writers hold the backend mutex, so there is one at a
time; the dump reads without it (a hung thread may hold it), so every slot
is seqlocked and every field an atomic word.

The Python side describes the torch-level operation in flight: before a
collective it writes a tag word into the ring's header (`H_TAG`), which the
entries it causes record, and the dump joins the two. The tag word carries a
hash of the writing thread, so an entry made on another thread meanwhile
does not take it.
"""
from std.atomic import Atomic, Ordering
from std.ffi import _get_global_or_null, external_call
from std.memory.alloc import unsafe_alloc
from std.time import perf_counter_ns

comptime FR_GLOBAL = "TMB_PG_FLIGHT_RECORDER"

# Tag word (flight_recorder.py `_tag_word` builds it).
comptime TAG_MASK = (1 << 40) - 1  # the Python record's id, 0 if none
comptime TAG_MEMBER = 1 << 40  # one operation of a coalescing block
comptime TAG_FIRST = 1 << 41  # ...its first
comptime TAG_END = 1 << 42  # the block's end
comptime TAG_P2P = 1 << 43  # point-to-point (overrides the native op)
comptime TAG_THREAD_SHIFT = 44
comptime TAG_THREAD_MASK = (1 << 19) - 1

# Entry flags.
comptime FL_P2P = 1
comptime FL_MEMBER = 2
comptime FL_END = 4

# Header words.
comptime H_SIZE = 0
comptime H_NEXT = 1  # entries recorded so far: the next record id
comptime H_TAG = 2
comptime H_WORDS = 3

# Slot words; a snapshot copies them as they are (flight_recorder.py).
comptime F_VER = 0
comptime F_ID = 1  # record id + 1, 0 while empty
comptime F_PG = 2
comptime F_CSEQ = 3
comptime F_PSEQ = 4
comptime F_OPID = 5
comptime F_OP = 6  # watchdog.mojo OP_*
comptime F_COUNT = 7
comptime F_DTYPE = 8  # ncclDataType_t
comptime F_ROOT = 9
comptime F_NOPS = 10
comptime F_FLAGS = 11
comptime F_CREATED = 12  # CLOCK_MONOTONIC ns
comptime F_THREAD = 13  # pthread_self
comptime F_TAG = 14
comptime F_DONE = 15  # record id + 1 once completed (the stamper's)
comptime F_DONE_NS = 16
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


def fr_ring() -> Int:
    """The process's ring, 0 when none was made."""
    var p = _get_global_or_null(FR_GLOBAL)
    if p:
        return Int(p.value())
    return 0


def fr_init(size: Int) -> Int:
    """The ring (its header address, where the tag word is), made on the
    first call with `size` entries; later sizes are ignored."""
    var ring = fr_ring()
    if ring != 0 or size <= 0:
        return ring
    var words = _zeroed(H_WORDS + size * F_WORDS)
    _st(words, H_SIZE, size)
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


@always_inline
def _thread_hash(t: Int) -> Int:
    return ((t >> 12) ^ (t >> 31)) & TAG_THREAD_MASK


@always_inline
def _tag(ring: FrP) -> Int:
    """This thread's tag word, its thread bits dropped; 0 if none."""
    var t = _ld(ring, H_TAG)
    var me = external_call["pthread_self", Int]()
    if (t >> TAG_THREAD_SHIFT) & TAG_THREAD_MASK != _thread_hash(me):
        return 0
    return t & ((1 << TAG_THREAD_SHIFT) - 1)


def _record(
    fp: Int, op: Int, nops: Int, count: Int, dtype: Int, root: Int, p2p_op: Bool
) -> Int:
    """Seq ids as ProcessGroupNCCL counts them: `collective_seq_id` once per
    collective or coalescing block, `p2p_seq_id` once per point-to-point op
    or block, `op_id` per operation. Returns record id + 1."""
    var pg = FrP(unsafe_from_address=fp)
    var ring = FrP(unsafe_from_address=_ld(pg, P_RING))
    var tag = _tag(ring)
    var p2p = (tag & TAG_P2P != 0) if tag & TAG_MASK != 0 else p2p_op
    var flags = FL_P2P if p2p else 0
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
    var v = _ld(s, F_VER)
    _st(s, F_VER, v + 1)
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
    _st(s, F_DONE, 0)
    _st(s, F_DONE_NS, 0)
    _st(s, F_VER, v + 2)
    _st(ring, H_NEXT, rid + 1)
    return rid + 1


def fr_commit(
    fp: Int, op: Int, nops: Int, count: Int, dtype: Int, root: Int, p2p_op: Bool
) -> Int:
    """The entry of a collective the watchdog tracks (a call, or a group at
    its end); record id + 1, 0 when not recording. Under the backend mutex.
    """
    if fp == 0:
        return 0
    return _record(fp, op, nops, count, dtype, root, p2p_op)


def fr_member(
    fp: Int, op: Int, count: Int, dtype: Int, root: Int, p2p_op: Bool
):
    """A library call inside an NCCL group: its own entry only when it is an
    operation of a coalescing block (the block's end entry, at the group's
    commit, carries the completion). Under the backend mutex."""
    if fp == 0:
        return
    var ring = FrP(unsafe_from_address=_ld(FrP(unsafe_from_address=fp), P_RING))
    if _tag(ring) & TAG_MEMBER:
        _ = _record(fp, op, 1, count, dtype, root, p2p_op)


def fr_done(fp: Int, rid1: Int, now: Int):
    """Stamp entry `rid1` (record id + 1) completed at `now`, unless the ring
    has reused its slot. Any thread; a stale stamp of a reused slot names the
    old id, which the dump ignores."""
    if fp == 0 or rid1 == 0:
        return
    var ring = FrP(unsafe_from_address=_ld(FrP(unsafe_from_address=fp), P_RING))
    var s = _slot(ring, rid1 - 1)
    if _ld(s, F_ID) != rid1:
        return
    _st(s, F_DONE_NS, now)
    _st(s, F_DONE, rid1)


def fr_snapshot(buf: Int, cap: Int) -> Int:
    """Copy the ring's entries, oldest first, into `buf` (`cap` entries of
    F_WORDS words); the number copied. Lock-free: an entry being rewritten
    is retried a few times, then left out."""
    var r = fr_ring()
    if r == 0:
        return 0
    var ring = FrP(unsafe_from_address=r)
    var out = FrP(unsafe_from_address=buf)
    var size = _ld(ring, H_SIZE)
    var next = _ld(ring, H_NEXT)
    var n = 0
    for rid in range(max(next - size, 0), next):
        if n >= cap:
            break
        var s = _slot(ring, rid)
        for _ in range(8):
            var v = _ld(s, F_VER)
            if v & 1:
                continue
            var o = out.unsafe_offset(n * F_WORDS)
            for k in range(F_WORDS):
                o[unsafe_offset=k] = Int64(_ld(s, k))
            if _ld(s, F_VER) == v and Int(o[unsafe_offset=F_ID]) == rid + 1:
                n += 1
                break
    return n
