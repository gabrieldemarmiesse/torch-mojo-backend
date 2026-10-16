"""Flight recorder of the mojo process group, in PyTorch's format.

ProcessGroupNCCL keeps the last ``TORCH_FR_BUFFER_SIZE`` collectives of the
process in a ring and dumps it, one pickle per rank, when a collective times
out (torch/csrc/distributed/c10d/FlightRecorder.hpp, version "2.10");
``python -m torch.distributed.flight_recorder.fr_trace <dir>`` then matches
the ranks' dumps and names the collective they disagree on. This module
writes the same dumps for the mojo backend, so that analyzer reads them.

The ring lives in the Mojo core (tmb/backend/flight_recorder.mojo): the
watchdog writes an entry for every collective it commits -- compiled-graph
ones included -- and stamps it completed when it retires it. This side
describes the torch-level operation of each eager call (name, shapes,
dtypes, Python stack) in a record whose id it writes into its thread's tag
word of the ring header before the call, and joins the two when it dumps.
Recording follows the watchdog: with TORCH_MOJO_BACKEND_PG_WATCHDOG=off
nothing is recorded.

Dumps on a timeout are coordinated through the c10d store, as
ProcessGroupNCCL's are, plus a generation: the rank whose watchdog fired
takes the next one, every process dumps once per generation it has not
served and acknowledges it, and the rank that fired waits for those
acknowledgements before it aborts.
"""

from __future__ import annotations

import ctypes
import itertools
import json
import os
import pickle
import struct
import sys
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import PrefixStore, Store
from torch._C._profiler import CapturedTraceback, gather_traceback, symbolize_tracebacks

from torch_mojo_backend.distributed.store_worker import StoreWorker

if TYPE_CHECKING:
    from torch_mojo_backend.distributed.process_group import MojoProcessGroup

VERSION = "2.10"  # FlightRecorder.hpp's version_val, the analyzer's input
DEFAULT_BUFFER_SIZE = 2000
DUMP_KEY = "exception_dump"  # ProcessGroupNCCL's kStoreDumpKey
GEN_KEY = "mojo_fr/generation"  # ours: one per signal, so peers dump again
ACK_KEY = "mojo_fr/dumped"  # ours: <generation>/<global rank>

# tmb/backend/flight_recorder.mojo: the words of a snapshot entry (F_*), its
# flags (FL_*) and the tag word's bits (TAG_*).
F_WORDS = 17
F_ID = 1
F_PG = 2
F_CSEQ = 3
F_PSEQ = 4
F_OPID = 5
F_OP = 6
F_COUNT = 7
F_DTYPE = 8
F_ROOT = 9
F_FLAGS = 11
F_CREATED = 12
F_THREAD = 13
F_TAG = 14
F_COMM = 15
F_DONE = 16
FL_P2P, FL_MEMBER, FL_END, FL_UNTRACKED = 1, 2, 4, 8
TAG_MASK = (1 << 40) - 1
TAG_MEMBER, TAG_FIRST, TAG_END, TAG_P2P = 1 << 40, 1 << 41, 1 << 42, 1 << 43

# watchdog.mojo OP_* -> the name ProcessGroupNCCL records, for entries no
# eager call described (compiled-graph collectives).
_NATIVE_NAMES = {
    1: "all_reduce",
    2: "broadcast",
    3: "reduce",
    4: "_all_gather_base",
    5: "_reduce_scatter_base",
}
_SEND, _RECV = 6, 7
# ncclDataType_t -> c10::toString(ScalarType)
_NCCL_DTYPE_NAMES = {
    0: "Char",
    1: "Byte",
    2: "Int",
    3: "UInt32",
    4: "Long",
    5: "UInt64",
    6: "Half",
    7: "Float",
    8: "Double",
    9: "BFloat16",
    10: "Float8_e4m3fn",
    11: "Float8_e5m2",
}
_DTYPE_NAMES = {
    torch.float32: "Float",
    torch.float64: "Double",
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.int8: "Char",
    torch.uint8: "Byte",
    torch.int16: "Short",
    torch.int32: "Int",
    torch.int64: "Long",
    torch.bool: "Bool",
    torch.complex64: "ComplexFloat",
    torch.complex128: "ComplexDouble",
    torch.float8_e4m3fn: "Float8_e4m3fn",
    torch.float8_e5m2: "Float8_e5m2",
}


def _first(*values: str | None) -> str | None:
    """torch's getCvar*: the first of the aliases that is set wins."""
    return next((v for v in values if v is not None), None)


def _int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "y", "yes", "on")


def buffer_size() -> int:
    size = _first(
        os.environ.get("TORCH_FR_BUFFER_SIZE"),
        os.environ.get("TORCH_NCCL_TRACE_BUFFER_SIZE"),
    )
    return max(_int(size, DEFAULT_BUFFER_SIZE), 0)


def dump_on_timeout() -> bool:
    return _bool(os.environ.get("TORCH_NCCL_DUMP_ON_TIMEOUT"), True)


def dump_wait_s() -> float:
    """How long the rank whose watchdog fired waits for the others' dumps."""
    wait = _int(os.environ.get("TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC"), 15000)
    return max(wait, 0) / 1000


def coord_check_s() -> float:
    return max(_int(os.environ.get("TORCH_NCCL_COORD_CHECK_MILSEC"), 1000), 10) / 1000


def include_stack_traces() -> bool:
    return _bool(os.environ.get("TORCH_INCLUDE_STACK_TRACE"), True)


def include_only_active() -> bool:
    return _bool(os.environ.get("TORCH_INCLUDE_ONLY_ACTIVE"), False)


def dump_prefix() -> str:
    """FlightRecorder.cpp's DebugInfoWriter location."""
    prefix = _first(
        os.environ.get("TORCH_FR_DUMP_TEMP_FILE"),
        os.environ.get("TORCH_NCCL_DEBUG_INFO_TEMP_FILE"),
    )
    if prefix:
        return prefix
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(cache, "torch", "comm_lib_trace_rank_")


def _dtype_names(dtypes: list[torch.dtype]) -> list[str]:
    return [_DTYPE_NAMES.get(d) or str(d)[6:] for d in dtypes]


# Frames of this package's adapter and of dynamo's `disable` wrapper: the
# stack starts where ProcessGroupNCCL's would, in torch.distributed.
_OWN_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep
_DYNAMO_DIR = os.sep + os.path.join("torch", "_dynamo") + os.sep


def _symbolize(tracebacks: list[CapturedTraceback]) -> list[list[dict[str, Any]]]:
    out = []
    for frames in symbolize_tracebacks(tracebacks):
        i = 0
        while i < len(frames) and (
            frames[i]["filename"].startswith(_OWN_DIR)
            or _DYNAMO_DIR in frames[i]["filename"]
        ):
            i += 1
        out.append(frames[i:])
    return out


class Shaped(Protocol):
    """What a record keeps of a tensor."""

    @property
    def shape(self) -> torch.Size: ...

    @property
    def dtype(self) -> torch.dtype: ...


class Flat(NamedTuple):
    """The stacked buffer ProcessGroupNCCL records for a list collective
    (`newLikeFlat`: [len(list), *shape])."""

    shape: torch.Size
    dtype: torch.dtype


def flat(tensors: Sequence[torch.Tensor]) -> Flat:
    return Flat(torch.Size([len(tensors), *tensors[0].shape]), tensors[0].dtype)


# An eager call, as ProcessGroupNCCL's `record()` sees it: tag, name, input
# shapes and dtypes, output shapes and dtypes, stack. A tuple of what the
# tensors already hold, so recording builds nothing (the names come at dump).
_Record = tuple[
    int,
    str,
    list[torch.Size],
    list[torch.dtype],
    list[torch.Size],
    list[torch.dtype],
    CapturedTraceback,
]


@dataclass
class _Group:
    """A process group as the dump's pg_config shows it."""

    name: str
    desc: str
    ranks: list[int]
    rank: int  # this process's rank in it
    timeout_ms: int


@dataclass
class _Entry:
    """One ring entry from a snapshot."""

    rid: int
    pg: int
    cseq: int
    pseq: int
    op_id: int
    op: int
    count: int
    dtype: int
    root: int
    flags: int
    created_ns: int
    thread: int
    tag: int
    comm: int
    done_ns: int | None


class Recorder:
    """The process's recorder: the native ring plus the eager records."""

    def __init__(self, size: int, snapshot: Callable[[int, int], int], lib_name: str):
        self.size = size
        self._snapshot = snapshot
        self._prefix = lib_name + ":"
        self.comm_lib_version = ""
        # Indexed by tag, twice the ring: an eager call may cause no entry.
        self._records: list[_Record | None] = [None] * (2 * size)
        self._tags = itertools.count(1)
        self._pg_ids = itertools.count(0)
        self._groups: dict[int, _Group] = {}
        self._live: dict[int, weakref.ref[MojoProcessGroup]] = {}

    # ---- recording (the issuing thread) ------------------------------------

    def add_group(self, group: MojoProcessGroup) -> int:
        pg_id = next(self._pg_ids)
        self._live[pg_id] = weakref.ref(group)
        return pg_id

    def after_fork(self):
        """A forked child: the parent's groups are not its own."""
        self._live.clear()

    def retire(self, pg_id: int):
        """A group is destroyed: keep its names for the dumps (its entries
        stay in the ring), but it no longer makes this backend's recorder
        the one `_dump_nccl_trace` answers for."""
        self._group(pg_id)
        self._live.pop(pg_id, None)

    def begin(
        self, name: str, inputs: Sequence[Shaped], outputs: Sequence[Shaped], flags: int
    ) -> int:
        """Describe the eager call about to issue; the tag word its pg.mojo
        entries take (`flags` included), whose id the ring entries they
        cause record."""
        tag = next(self._tags)
        shapes = [t.shape for t in inputs]
        dtypes = [t.dtype for t in inputs]
        if outputs is not inputs:
            out_shapes = [t.shape for t in outputs]
            out_dtypes = [t.dtype for t in outputs]
        else:
            out_shapes, out_dtypes = shapes, dtypes
        records = self._records
        records[tag % len(records)] = (
            tag,
            name,
            shapes,
            dtypes,
            out_shapes,
            out_dtypes,
            gather_traceback(True, False, False),
        )
        return tag | flags

    # ---- dumping (any thread) ----------------------------------------------

    def active(self) -> bool:
        """A mojo group that is not destroyed records."""
        return bool(self._live)

    def peer_ranks(self) -> set[int]:
        """The global ranks of this process's live groups: the ranks that
        dump on a signal."""
        ranks: set[int] = set()
        for pg_id in list(self._live):
            group = self._group(pg_id)
            if group is not None:
                ranks.update(group.ranks)
        return ranks

    def _group(self, pg_id: int) -> _Group | None:
        """A group's names and ranks, read from the live group (the names
        are set after it is built) and kept for after it is destroyed."""
        ref = self._live.get(pg_id)
        group = ref() if ref is not None else None
        if group is not None:
            try:
                ranks = dist.get_process_group_ranks(group)
            except Exception:
                ranks = list(range(group.size()))
            timeout = getattr(group, "_timeout", None)
            self._groups[pg_id] = _Group(
                group.getGroupName(),
                getattr(group, "group_desc", "undefined") or "undefined",
                ranks,
                group.rank(),
                int(timeout.total_seconds() * 1000) if timeout is not None else 0,
            )
        return self._groups.get(pg_id)

    def entries(self) -> list[_Entry]:
        buf = (ctypes.c_int64 * (self.size * F_WORDS))()
        n = self._snapshot(ctypes.addressof(buf), self.size)
        out = []
        for i in range(n):
            w = buf[i * F_WORDS : (i + 1) * F_WORDS]
            out.append(
                _Entry(
                    w[F_ID] - 1,
                    w[F_PG],
                    w[F_CSEQ],
                    w[F_PSEQ],
                    w[F_OPID],
                    w[F_OP],
                    w[F_COUNT],
                    w[F_DTYPE],
                    w[F_ROOT],
                    w[F_FLAGS],
                    w[F_CREATED],
                    w[F_THREAD],
                    w[F_TAG],
                    w[F_COMM],
                    w[F_DONE] or None,
                )
            )
        return out

    @staticmethod
    def _record(records: list[_Record | None], tag: int) -> _Record | None:
        r = records[tag % len(records)] if tag else None
        return r if r is not None and r[0] == tag else None

    def stack(self, rid1: int) -> list[dict[str, Any]]:
        """The Python stack of the eager call behind entry `rid1` (record
        id + 1), innermost first; [] when there is none."""
        for e in self.entries():
            if e.rid + 1 == rid1:
                record = self._record(self._records, e.tag)
                if record is not None:
                    return _symbolize([record[6]])[0]
        return []

    def dump(
        self,
        include_collectives: bool = True,
        include_stack_traces: bool = True,
        only_active: bool = False,
    ) -> dict[str, Any]:
        """FlightRecorderDetail.hpp `dump()`'s dictionary. Entries of a group
        this process never knew are left out: the analyzer needs every
        entry's group in pg_config."""
        # A name is the newest group's of that name (re-created after a
        # destroy): an older one's entries would be read against its ranks.
        groups: dict[int, _Group] = {}
        newest: dict[str, int] = {}
        for pg in sorted({*self._live, *self._groups}):
            g = self._group(pg)
            if g is not None:
                groups[pg] = g
                newest[g.name] = pg
        groups = {pg: g for pg, g in groups.items() if newest[g.name] == pg}
        entries = [e for e in self.entries() if e.pg in groups]
        snapshot = list(self._records)
        records = {
            e.tag: r
            for e in entries
            if (r := self._record(snapshot, e.tag)) is not None
        }
        offset = time.time_ns() - time.monotonic_ns()
        threads = {t.ident: t.name for t in threading.enumerate()}
        # A collective the watchdog could not track (its queue full, its
        # event failed) has no stamp; a later collective on the same
        # communicator completing means it did (its comm stream runs in
        # order). A coalescing block's operations, like ProcessGroupNCCL's,
        # have none.
        done: list[int | None] = [e.done_ns for e in entries]
        later: dict[tuple[int, int], int] = {}
        for i in range(len(entries) - 1, -1, -1):
            e = entries[i]
            if e.flags & FL_MEMBER or not e.comm:
                continue
            if e.flags & FL_UNTRACKED:
                done[i] = later.get((e.pg, e.comm))
            elif e.done_ns is not None:
                later[(e.pg, e.comm)] = e.done_ns
        stacks: dict[int, list[dict[str, Any]]] = {}
        if include_collectives and include_stack_traces:
            tags = list(records)
            stacks = dict(zip(tags, _symbolize([records[t][6] for t in tags])))
        status: dict[str, dict[str, int]] = {}
        out: list[dict[str, Any]] = []
        for e, done_ns in zip(entries, done):
            g = groups[e.pg]
            st = status.setdefault(
                str(e.pg),
                {
                    "last_enqueued_collective": -1,
                    "last_started_collective": -1,
                    "last_completed_collective": -1,
                },
            )
            if not e.flags & (FL_P2P | FL_MEMBER):
                st["last_enqueued_collective"] = e.cseq
                if done_ns is not None:
                    st["last_completed_collective"] = e.cseq
            if only_active and done_ns is not None:
                continue
            entry: dict[str, Any] = {}
            if include_stack_traces:
                entry["frames"] = stacks.get(e.tag, [])
            entry["record_id"] = e.rid
            entry["pg_id"] = e.pg
            entry["process_group"] = (g.name, g.desc)
            entry["thread_name"] = threads.get(e.thread, "")
            entry["thread_id"] = str(e.thread)
            entry["collective_seq_id"] = e.cseq
            entry["p2p_seq_id"] = e.pseq
            entry["op_id"] = e.op_id
            record = records.get(e.tag)
            if e.flags & FL_END:
                name = record[1] if record is not None else "coalesced"
                sizes: list[list[int]] = []
                in_sizes, in_dtypes, out_sizes, out_dtypes = sizes, [], sizes, []
            elif record is not None:
                _, name, shapes, dtypes, out_shapes, out_dtypes_, _ = record
                in_sizes = [list(x) for x in shapes]
                in_dtypes = _dtype_names(dtypes)
                out_sizes = [list(x) for x in out_shapes]
                out_dtypes = _dtype_names(out_dtypes_)
            else:
                name, in_sizes, out_sizes = self._native(e, g)
                in_dtypes = out_dtypes = [_NCCL_DTYPE_NAMES.get(e.dtype, str(e.dtype))]
            entry["profiling_name"] = self._prefix + name
            entry["time_created_ns"] = e.created_ns + offset
            entry["input_sizes"] = in_sizes
            entry["input_dtypes"] = in_dtypes
            entry["output_sizes"] = out_sizes
            entry["output_dtypes"] = out_dtypes
            entry["state"] = "completed" if done_ns is not None else "scheduled"
            entry["time_discovered_started_ns"] = None
            entry["time_discovered_completed_ns"] = (
                done_ns + offset if done_ns is not None else None
            )
            entry["retired"] = done_ns is not None
            entry["timeout_ms"] = g.timeout_ms
            entry["is_p2p"] = bool(e.flags & FL_P2P)
            out.append(entry)
        result: dict[str, Any] = {
            "version": VERSION,
            "pg_config": {
                g.name: {"name": g.name, "desc": g.desc, "ranks": str(g.ranks)}
                for g in groups.values()
            },
            "comm_lib_version": self.comm_lib_version,
            "pg_status": status,
        }
        if include_collectives:
            result["entries"] = out
        return result

    @staticmethod
    def _native(e: _Entry, g: _Group) -> tuple[str, list[list[int]], list[list[int]]]:
        """Name and sizes of an entry no eager call described: a
        compiled-graph collective (its element count as its shape), or a
        call whose record was lost. Point-to-point names carry the ranks the
        analyzer parses."""
        if e.op == _SEND:
            return f"send {g.rank}->{e.root}", [[e.count]], [[e.count]]
        if e.op == _RECV:
            return f"recv {g.rank}<-{e.root}", [[e.count]], [[e.count]]
        n_in = n_out = e.count
        if e.op == 4:  # all-gather: count is per rank
            n_out = e.count * len(g.ranks)
        elif e.op == 5:  # reduce-scatter: count is the output's
            n_in = e.count * len(g.ranks)
        return _NATIVE_NAMES.get(e.op, "coalesced"), [[n_in]], [[n_out]]


def as_json(dump: dict[str, Any]) -> bytes:
    """FlightRecorderDetail.hpp `dump_json()`: no frames, the pg_status
    values as strings, a null time as 0, no "entries" when empty."""
    out = dict(dump)
    out["pg_status"] = {
        pg: {k: str(v) for k, v in st.items()} for pg, st in dump["pg_status"].items()
    }
    entries = [
        {
            k: (0 if v is None and k.startswith("time_") else v)
            for k, v in e.items()
            if k != "frames"
        }
        for e in dump.get("entries", [])
    ]
    out.pop("entries", None)
    if entries:
        out["entries"] = entries
    out["nccl_comm_state"] = {}
    return json.dumps(out, sort_keys=True, separators=(",", ":")).encode()


def empty_dump(include_collectives: bool = True) -> dict[str, Any]:
    out: dict[str, Any] = {
        "version": VERSION,
        "pg_config": {},
        "comm_lib_version": "",
        "pg_status": {},
    }
    if include_collectives:
        out["entries"] = []
    return out


_RECORDER: list[Recorder] = []
_LOCK = threading.Lock()
# flight_recorder.mojo `fr_after_fork`, for the fork hook below
_AFTER_FORK: list[Callable[[], None]] = []


def recorder() -> Recorder | None:
    return _RECORDER[0] if _RECORDER else None


def active() -> bool:
    """A mojo process group records: the dump is this backend's."""
    rec = recorder()
    return rec is not None and rec.active()


def make_recorder(
    init: Callable[[int], int],
    snapshot: Callable[[int, int], int],
    after_fork: Callable[[], None],
    lib_name: str,
) -> Recorder | None:
    """The process's recorder, made on first use; None when
    TORCH_FR_BUFFER_SIZE is 0."""
    with _LOCK:
        if _RECORDER:
            return _RECORDER[0]
        size = buffer_size()
        if size == 0 or not init(size):
            return None
        _RECORDER.append(Recorder(size, snapshot, lib_name))
        _AFTER_FORK.append(after_fork)
        return _RECORDER[0]


def global_rank(fallback: int) -> int:
    return dist.get_rank() if dist.is_initialized() else fallback


_WRITE_LOCK = threading.Lock()


def write_dump(rank: int, stacks: bool) -> str | None:
    """Write this process's dump, the pickle ProcessGroupNCCL writes, to
    `<prefix><rank>`; the path, or None without a recorder. Atomic and
    serialized: a temporary file of a unique name without the prefix in it
    (the analyzer picks files by it) is renamed into place."""
    rec = recorder()
    if rec is None:
        return None
    with _WRITE_LOCK:
        data = pickle.dumps(rec.dump(True, stacks, include_only_active()), protocol=2)
        path = dump_prefix() + str(rank)
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        tmp = os.path.join(directory, f".mojo-fr-{uuid.uuid4().hex}.tmp")
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return path


def dump_now(rank: int, why: str, stacks: bool) -> bool:
    """Write this process's dump; whether it was published."""
    try:
        path = write_dump(rank, stacks)
    except Exception as error:
        sys.stderr.write(f"[rank {rank}] flight recorder: dump failed: {error!r}\n")
        return False
    if path is None:
        return False
    sys.stderr.write(f"[rank {rank}] flight recorder: {why}, dumped to {path}\n")
    sys.stderr.flush()
    return True


def unprefixed(store: Store) -> Store:
    """The store under every PrefixStore: where ProcessGroupNCCL's
    `globalStore_` signals (getUnderlyingNonPrefixStore)."""
    while isinstance(store, PrefixStore):
        store = store.underlying_store
    return store


def store_key(store: Store) -> tuple[object, ...]:
    """Which store this is: groups of one job reach the same TCPStore (or
    FileStore) through different Python objects."""
    host, port = getattr(store, "host", None), getattr(store, "port", None)
    if host is not None and port is not None:
        return ("tcp", host, port)
    path = getattr(store, "path", None)
    if path is not None:
        return ("file", path)
    return ("object", id(store))


# ---- the dump signal --------------------------------------------------------


def signal_dump(client: Store, rank: int) -> int:
    """broadcastDumpSignal, plus a new generation (returned): the key torch
    polls, the signaller's global rank as a native int. The generation is
    taken first, so a poll that sees the key (it reads the key first) sees
    the generation too."""
    gen = client.add(GEN_KEY, 1)
    client.set(DUMP_KEY, struct.pack("=i", rank))  # ty: ignore[invalid-argument-type] -- the stub says str; Store.set takes bytes too, and torch's value is binary
    return gen


def current_generation(client: Store) -> int:
    return client.add(GEN_KEY, 0)


def signaller(client: Store) -> int:
    """The rank in torch's dump key, -1 when absent (a signaller that died
    between its generation and the key): `get` would wait for it."""
    if not client.check([DUMP_KEY]):
        return -1
    return struct.unpack("=i", bytes(client.get(DUMP_KEY)))[0]


def ack_dump(client: Store, rank: int, gens: Iterable[int]):
    for gen in gens:
        client.set(f"{ACK_KEY}/{gen}/{rank}", "1")


def wait_for_dumps(
    client: Store, ranks: Sequence[int], gen: int, deadline: float
) -> list[int]:
    """Wait until every rank of `ranks` acknowledged generation `gen` or the
    deadline (monotonic); the ranks that did not. One request per poll."""
    keys = [f"{ACK_KEY}/{gen}/{r}" for r in ranks]
    while time.monotonic() < deadline:
        if client.check(keys):
            return []
        time.sleep(0.1)
    return [r for r, key in zip(ranks, keys) if not client.check([key])]


_MAX_BACKOFF_S = 60.0


def _poll_backoff_cap() -> float:
    """Polls of an unanswering store back off, but never past a quarter of
    the dump wait: the firing rank waits that long for this rank's dump."""
    return max(min(_MAX_BACKOFF_S, dump_wait_s() / 4), coord_check_s())


class _StoreState:
    """The dump signals of one store incarnation, for this process: every
    generation is done (dumped), pending (a dump in progress), failed
    (dumped in vain: retried after a backoff) or to serve. A generation not
    seen before is served at once, whatever the backoff of failed ones. A
    dump written whose acknowledgement did not get through is acknowledged
    again at the next poll, without a new dump."""

    def __init__(self, base: int | None, torch_seen: bool):
        self.base = base  # every generation <= base is done; None: unread
        self.first = base  # the store's generation when first watched
        self.high = base  # the highest generation ever claimed
        self.done: set[int] = set()
        self.pending: set[int] = set()
        self.failed: set[int] = set()
        self.unacked: set[int] = set()
        # torch's own key (ProcessGroupNCCL's signal): "seen" when it was
        # there before, or dumped for; "unseen" until then
        self.torch = "seen" if torch_seen else "unseen"
        self.torch_tried = False
        self.dump_backoff = 0.0
        self.dump_retry_at = 0.0
        self.poll_backoff = 0.0
        self.poll_at = 0.0

    def claim(self, upto: int) -> tuple[int, ...]:
        if self.base is None:
            return ()
        top = max(self.high if self.high is not None else self.base, self.base)
        gens = set(range(top + 1, upto + 1))
        if self.failed and time.monotonic() >= self.dump_retry_at:
            gens |= self.failed
            self.failed = set()
        gens -= self.done | self.pending
        self.high = max(top, upto)
        self.pending |= gens
        return tuple(sorted(gens))

    def finish(self, gens: Iterable[int], dumped: bool, acked: bool = True):
        gens = set(gens)
        self.pending -= gens
        if not dumped:
            self.failed |= gens
            self.backoff(False)
            return
        self.backoff(True)
        self.failed -= gens
        self.done |= gens
        assert self.base is not None
        while self.base + 1 in self.done:
            self.base += 1
            self.done.discard(self.base)
        if not acked:
            self.unacked |= gens

    def backoff(self, ok: bool):
        """Back retries of a failing dump off, doubling up to a minute."""
        if ok:
            self.dump_backoff = 0.0
            return
        self.dump_backoff = min(
            max(2 * self.dump_backoff, coord_check_s()), _MAX_BACKOFF_S
        )
        self.dump_retry_at = time.monotonic() + self.dump_backoff


class _Poller:
    """ProcessGroupNCCL's heartbeat monitor, for the dump only: polls the
    store of the watched groups for a new dump generation (or torch's own
    dump key appearing, which a ProcessGroupNCCL in the job sets without
    one), and the debug pipe. One per process.

    It follows the groups: each attaches its store and detaches at
    shutdown; the store polled is the newest attached one. The signal state
    is kept per store (address) and incarnation: a generation below the
    state's base means the server was restarted, and starts a new state. A
    store's worker lives as long as its state, and an answer is only ever
    applied to the state it was asked for. Store operations run on a worker
    the monitor abandons on overrun, at most one abandoned at a time; an
    unanswering store is polled less often, up to a quarter of the dump
    wait."""

    def __init__(self):
        self.lock = threading.RLock()
        # group -> its unprefixed store's key, the store, this process's
        # global rank
        self.groups: dict[int, tuple[tuple[object, ...], Store, int]] = {}
        self.states: dict[tuple[object, ...], _StoreState] = {}
        self.pipe = -1
        self.pipe_path = ""

    def attach(self, group: int, store: Store, rank: int):
        """`rank` is global: a group created before the default group is
        registered is the default group, whose ranks are the global ones."""
        base = unprefixed(store)
        key = store_key(base)
        # Generations of an earlier job on this store are not ours: the
        # baseline is read before the store is watched, published with it
        # (the first poll takes it when this read fails).
        read: tuple[int, bool] | None = None
        if dump_on_timeout():
            try:
                seen = base.check([DUMP_KEY])
                read = (current_generation(base), seen)
            except Exception:
                read = None
        with self.lock:
            state = self.states.get(key)
            # A state that went (its last group detached meanwhile), or one
            # of a server since restarted at this address: a new one.
            if state is None or (
                read is not None and state.base is not None and read[0] < state.base
            ):
                self.states[key] = (
                    _StoreState(*read) if read else _StoreState(None, False)
                )
            self.groups[group] = (key, base, rank)
            self._pipe(rank)  # before the caller's first collective

    def detach(self, group: int):
        with self.lock:
            entry = self.groups.pop(group, None)
            if entry is not None and all(
                k != entry[0] for k, _, _ in self.groups.values()
            ):
                self.states.pop(entry[0], None)

    def state(self, key: tuple[object, ...]) -> _StoreState | None:
        with self.lock:
            return self.states.get(key)

    def _pipe(self, rank: int):
        stem = os.environ.get("TORCH_NCCL_DEBUG_INFO_PIPE_FILE", "")
        path = f"{stem}{rank}.pipe" if stem else ""
        if path == self.pipe_path:
            return
        if self.pipe >= 0:
            os.close(self.pipe)
            self.pipe = -1
        self.pipe_path = path
        if not path:
            return
        try:
            if os.path.exists(path):
                os.unlink(path)
            os.mkfifo(path, 0o666)
            self.pipe = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as error:
            sys.stderr.write(
                f"[rank {rank}] flight recorder: no debug pipe {path}: {error!r}\n"
            )

    def after_fork(self):
        """A forked child: no monitor thread, no groups, a fresh lock."""
        self.lock = threading.RLock()
        self.groups.clear()
        self.states.clear()

    def run(self):
        # one worker per store state, closed when the state goes
        workers: dict[int, tuple[_StoreState, StoreWorker]] = {}
        while True:
            time.sleep(coord_check_s())
            with self.lock:
                current = next(reversed(self.groups.values()), None)
                pipe = self.pipe
                live = {id(st): st for st in self.states.values()}
                state = self.states.get(current[0]) if current else None
            for sid in [i for i, (st, _) in workers.items() if live.get(i) is not st]:
                workers.pop(sid)[1].close()
            if current is None:
                continue
            _, store, rank = current
            if pipe >= 0:
                try:
                    if os.read(pipe, 4096):
                        dump_now(rank, "dump requested through the debug pipe", True)
                except OSError:
                    pass  # nothing written (EAGAIN), or reopened meanwhile
            if not dump_on_timeout() or state is None:
                continue
            if id(state) not in workers:
                workers[id(state)] = (
                    state,
                    StoreWorker(
                        store, quiet=True, name="mojo-pg-fr-store", max_abandoned=1
                    ),
                )
            self.poll(workers[id(state)][1], current[0], state, rank)

    def poll(
        self,
        worker: StoreWorker,
        key: tuple[object, ...],
        state: _StoreState,
        rank: int,
    ):
        if worker.stuck or time.monotonic() < state.poll_at:
            return  # an abandoned job still blocks: try again once it returns
        bound = coord_check_s() + 2
        # The key first: `signal_dump` takes the generation before it sets
        # the key, so a key seen here comes with its generation.
        seen = worker.run(lambda c: (c.check([DUMP_KEY]), current_generation(c)), bound)
        with self.lock:
            if self.states.get(key) is not state:
                return  # its groups went, or it was replaced: the answer is moot
            if seen is None:
                if not worker.stuck:  # an error, not an overrun: back off
                    state.poll_backoff = min(
                        max(2 * state.poll_backoff, coord_check_s()),
                        _poll_backoff_cap(),
                    )
                    state.poll_at = time.monotonic() + state.poll_backoff
                return
            state.poll_backoff = 0.0
            has_key, gen = seen
            if state.base is None or gen < state.base:
                # No baseline at attach, or a server restarted at this
                # address (its counter is below what was served): a new
                # state from this read, and a new worker with it.
                self.states[key] = _StoreState(gen, has_key)
                return
            if gen != state.first:
                state.torch = "seen"  # the key is (also) ours: generations serve it
            gens = state.claim(gen)
            torch_signal = (
                not gens
                and has_key
                and state.torch == "unseen"
                and (not state.torch_tried or time.monotonic() >= state.dump_retry_at)
            )
            unacked = tuple(sorted(state.unacked))
        if unacked and worker.run(lambda c: ack_dump(c, rank, unacked) or True, bound):
            with self.lock:
                state.unacked -= set(unacked)
        if gens:
            who = worker.run(signaller, bound)
            why = f"rank {who if who is not None else -1} signalled a dump"
            dumped = dump_now(rank, why, include_stack_traces())
            acked = dumped and (
                worker.run(lambda c: ack_dump(c, rank, gens) or True, bound) is True
            )
            with self.lock:
                state.finish(gens, dumped, acked)
        elif torch_signal:
            # ProcessGroupNCCL's broadcastDumpSignal: no generation, no ack.
            dumped = dump_now(
                rank, "a NCCL process group signalled a dump", include_stack_traces()
            )
            with self.lock:
                state.torch_tried = True
                if dumped:
                    state.torch = "seen"
                state.backoff(dumped)


_POLLER: list[_Poller] = []


def attach_poller(group: int, store: Store, rank: int):
    """The process's monitor, started with the first watched group."""
    if not (dump_on_timeout() or os.environ.get("TORCH_NCCL_DEBUG_INFO_PIPE_FILE")):
        return
    with _LOCK:
        start = not _POLLER
        if start:
            _POLLER.append(_Poller())
        poller = _POLLER[0]
    poller.attach(group, store, rank)
    if start:
        threading.Thread(
            target=poller.run, name="mojo-pg-fr-monitor", daemon=True
        ).start()


def detach_poller(group: int):
    if _POLLER:
        _POLLER[0].detach(group)


def claim(store: Store, upto: int) -> tuple[int, ...]:
    """This process's claim on the generations up to `upto` of `store` (see
    `_StoreState`); without a monitor every generation is new."""
    if not _POLLER:
        return (upto,)
    poller = _POLLER[0]
    with poller.lock:
        state = poller.states.get(store_key(store))
        return state.claim(upto) if state is not None else (upto,)


def finish(store: Store, gens: Iterable[int], dumped: bool, acked: bool):
    if _POLLER:
        poller = _POLLER[0]
        with poller.lock:
            state = poller.states.get(store_key(store))
            if state is not None:
                state.finish(gens, dumped, acked)


def _after_fork_in_child():
    """Locks another thread of the parent held stay held in the child, and
    the monitor thread is gone: start over (the parent's groups are not the
    child's), and release the ring's slots a parent thread was writing."""
    global _LOCK, _WRITE_LOCK
    _LOCK = threading.Lock()
    _WRITE_LOCK = threading.Lock()
    for poller in _POLLER:
        poller.after_fork()
    _POLLER.clear()
    for rec in _RECORDER:
        rec.after_fork()
    for release in _AFTER_FORK:
        release()


os.register_at_fork(after_in_child=_after_fork_in_child)


def dump_nccl_trace(
    includeCollectives: bool | None = None,  # noqa: N803 -- torch's keyword names
    includeStackTraces: bool | None = None,  # noqa: N803
    onlyActive: bool | None = None,  # noqa: N803
) -> bytes:
    """`torch._C._distributed_c10d._dump_nccl_trace` for the mojo backend:
    the pickled dump of this process."""
    collectives = True if includeCollectives is None else includeCollectives
    rec = recorder()
    dump = (
        rec.dump(
            collectives,
            True if includeStackTraces is None else includeStackTraces,
            bool(onlyActive),
        )
        if rec is not None
        else empty_dump(collectives)
    )
    return pickle.dumps(dump, protocol=2)


def dump_nccl_trace_json(
    includeCollectives: bool | None = None,  # noqa: N803 -- torch's keyword names
    onlyActive: bool | None = None,  # noqa: N803
) -> bytes:
    """`torch._C._distributed_c10d._dump_nccl_trace_json` for the mojo
    backend."""
    collectives = True if includeCollectives is None else includeCollectives
    rec = recorder()
    dump = (
        rec.dump(collectives, False, bool(onlyActive))
        if rec is not None
        else empty_dump(collectives)
    )
    return as_json(dump)
