"""Flight recorder of the mojo process group, in PyTorch's format.

ProcessGroupNCCL keeps the last ``TORCH_FR_BUFFER_SIZE`` collectives of the
process in a ring and dumps it, one pickle per rank, when a collective times
out (torch/csrc/distributed/c10d/FlightRecorder.hpp, version "2.10");
``python -m torch.distributed.flight_recorder.fr_trace <dir>`` then matches
the ranks' dumps and names the collective they disagree on. This module
writes the same dumps for the mojo backend, so that analyzer reads them.

The ring lives in the Mojo core (tmb/backend/flight_recorder.mojo): the
watchdog writes an entry for every collective it tracks -- compiled-graph
ones included -- and stamps it completed when it retires it. This side
describes the torch-level operation of each eager call (name, shapes,
dtypes, Python stack) in a record whose id it writes into the ring's tag
word before the call, and joins the two when it dumps. Recording follows
the watchdog: with TORCH_MOJO_BACKEND_PG_WATCHDOG=off nothing is recorded.
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
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import PrefixStore, Store
from torch._C._profiler import CapturedTraceback, gather_traceback, symbolize_tracebacks

if TYPE_CHECKING:
    from torch_mojo_backend.distributed.process_group import MojoProcessGroup

VERSION = "2.10"  # FlightRecorder.hpp's version_val, the analyzer's input
DEFAULT_BUFFER_SIZE = 2000
DUMP_KEY = "exception_dump"  # ProcessGroupNCCL's kStoreDumpKey
ACK_KEY = "mojo_fr/dumped"  # ours: a rank wrote its dump (torch only sleeps)

# tmb/backend/flight_recorder.mojo: the words of a snapshot entry (F_*), its
# flags (FL_*), the tag word's bits (TAG_*) and the tag's header word.
F_WORDS = 17
F_ID = 1
F_PG = 2
F_CSEQ = 3
F_PSEQ = 4
F_OPID = 5
F_OP = 6
F_COUNT = 7
F_DTYPE = 8
F_FLAGS = 11
F_CREATED = 12
F_THREAD = 13
F_TAG = 14
F_DONE = 15
F_DONE_NS = 16
FL_P2P, FL_MEMBER, FL_END = 1, 2, 4
TAG_MASK = (1 << 40) - 1
TAG_MEMBER, TAG_FIRST, TAG_END, TAG_P2P = 1 << 40, 1 << 41, 1 << 42, 1 << 43
TAG_THREAD_SHIFT = 44
TAG_THREAD_MASK = (1 << 19) - 1
H_TAG = 2  # header word of the tag

# watchdog.mojo OP_* -> the name ProcessGroupNCCL records, for entries no
# eager call described (compiled-graph collectives).
_NATIVE_NAMES = {
    1: "all_reduce",
    2: "broadcast",
    3: "reduce",
    4: "_all_gather_base",
    5: "_reduce_scatter_base",
    6: "send",
    7: "recv",
}
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


def _thread_bits() -> int:
    """flight_recorder.mojo `_thread_hash` of this thread (its pthread_self
    is CPython's thread ident), placed in the tag word."""
    t = threading.get_ident()
    return (((t >> 12) ^ (t >> 31)) & TAG_THREAD_MASK) << TAG_THREAD_SHIFT


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
    flags: int
    created_ns: int
    thread: int
    tag: int
    done_ns: int | None


class Recorder:
    """The process's recorder: the native ring plus the eager records."""

    def __init__(
        self, size: int, cell: int, snapshot: Callable[[int, int], int], lib_name: str
    ):
        self.size = size
        self._tag_cell = ctypes.c_int64.from_address(cell + 8 * H_TAG)
        self._snapshot = snapshot
        self._prefix = lib_name + ":"
        self.comm_lib_version = ""
        # Indexed by tag, twice the ring: an eager call may cause no entry.
        self._records: list[_Record | None] = [None] * (2 * size)
        self._tags = itertools.count(1)
        self._pg_ids = itertools.count(0)
        self._groups: dict[int, _Group] = {}
        self._live: dict[int, weakref.ref[MojoProcessGroup]] = {}
        self.dumped_on_signal = False

    # ---- recording (the issuing thread) ------------------------------------

    def add_group(self, group: MojoProcessGroup) -> int:
        pg_id = next(self._pg_ids)
        self._live[pg_id] = weakref.ref(group)
        return pg_id

    def remember(self, pg_id: int):
        """Keep a group's names for the dumps made after it is destroyed."""
        self._group(pg_id)

    def begin(
        self,
        name: str,
        inputs: Sequence[torch.Tensor],
        outputs: Sequence[torch.Tensor],
        flags: int,
    ):
        """Describe the eager call about to issue: the ring entries it
        causes take this record's id from the tag word."""
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
        self._tag_cell.value = tag | flags | _thread_bits()

    def clear(self):
        self._tag_cell.value = 0

    # ---- dumping (any thread) ----------------------------------------------

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
                int(timeout.total_seconds() * 1000) if timeout is not None else 0,
            )
        return self._groups.get(pg_id)

    def entries(self) -> list[_Entry]:
        buf = (ctypes.c_int64 * (self.size * F_WORDS))()
        n = self._snapshot(ctypes.addressof(buf), self.size)
        out = []
        for i in range(n):
            w = buf[i * F_WORDS : (i + 1) * F_WORDS]
            rid1 = w[F_ID]
            out.append(
                _Entry(
                    rid1 - 1,
                    w[F_PG],
                    w[F_CSEQ],
                    w[F_PSEQ],
                    w[F_OPID],
                    w[F_OP],
                    w[F_COUNT],
                    w[F_DTYPE],
                    w[F_FLAGS],
                    w[F_CREATED],
                    w[F_THREAD],
                    w[F_TAG],
                    w[F_DONE_NS] if w[F_DONE] == rid1 else None,
                )
            )
        return out

    def _record(self, records: list[_Record | None], tag: int) -> _Record | None:
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
        """FlightRecorderDetail.hpp `dump()`'s dictionary."""
        entries = self.entries()
        snapshot = list(self._records)
        records = {
            e.tag: r
            for e in entries
            if (r := self._record(snapshot, e.tag)) is not None
        }
        offset = time.time_ns() - time.monotonic_ns()
        threads = {t.ident: t.name for t in threading.enumerate()}
        groups = {pg: self._group(pg) for pg in {e.pg for e in entries}}
        groups.update({pg: self._group(pg) for pg in list(self._live)})
        # A coalescing block's operations complete with its end entry.
        done: list[int | None] = [e.done_ns for e in entries]
        block_end: dict[int, int | None] = {}
        for i in range(len(entries) - 1, -1, -1):
            e = entries[i]
            if e.flags & FL_END:
                block_end[e.pg] = e.done_ns
            elif e.flags & FL_MEMBER:
                done[i] = block_end.get(e.pg)
        stacks: dict[int, list[dict[str, Any]]] = {}
        if include_collectives and include_stack_traces:
            tags = list(records)
            symbolized = _symbolize([records[tag][6] for tag in tags])
            stacks = dict(zip(tags, symbolized))
        status: dict[str, dict[str, int]] = {}
        out: list[dict[str, Any]] = []
        for e, done_ns in zip(entries, done):
            g = groups.get(e.pg)
            st = status.setdefault(
                str(e.pg),
                {
                    "last_enqueued_collective": -1,
                    "last_started_collective": -1,
                    "last_completed_collective": -1,
                },
            )
            if not e.flags & FL_P2P:
                st["last_enqueued_collective"] = e.cseq
                if done_ns is not None:
                    st["last_completed_collective"] = e.cseq
            if only_active and done_ns is not None:
                continue
            record = records.get(e.tag)
            size = len(g.ranks) if g is not None else 1
            entry: dict[str, Any] = {}
            if include_stack_traces:
                entry["frames"] = stacks.get(e.tag, [])
            entry["record_id"] = e.rid
            entry["pg_id"] = e.pg
            entry["process_group"] = (
                (g.name, g.desc) if g is not None else (str(e.pg), "undefined")
            )
            entry["thread_name"] = threads.get(e.thread, "")
            entry["thread_id"] = str(e.thread)
            entry["collective_seq_id"] = e.cseq
            entry["p2p_seq_id"] = e.pseq
            entry["op_id"] = e.op_id
            if record is not None:
                _, name, shapes, dtypes, out_shapes, out_dtypes, _ = record
                entry["profiling_name"] = self._prefix + name
                entry["time_created_ns"] = e.created_ns + offset
                entry["input_sizes"] = [list(x) for x in shapes]
                entry["input_dtypes"] = _dtype_names(dtypes)
                entry["output_sizes"] = [list(x) for x in out_shapes]
                entry["output_dtypes"] = _dtype_names(out_dtypes)
            else:
                n_in, n_out = e.count, e.count
                if e.op == 4:  # all-gather: count is per rank
                    n_out = e.count * size
                elif e.op == 5:  # reduce-scatter: count is the output's
                    n_in = e.count * size
                dtype = [_NCCL_DTYPE_NAMES.get(e.dtype, str(e.dtype))]
                entry["profiling_name"] = self._prefix + _NATIVE_NAMES.get(
                    e.op, "coalesced"
                )
                entry["time_created_ns"] = e.created_ns + offset
                entry["input_sizes"] = [[n_in]]
                entry["input_dtypes"] = dtype
                entry["output_sizes"] = [[n_out]]
                entry["output_dtypes"] = dtype
            if e.flags & FL_END:
                for key in ("input_sizes", "input_dtypes", "output_sizes"):
                    entry[key] = []
                entry["output_dtypes"] = []
            entry["state"] = "completed" if done_ns is not None else "scheduled"
            entry["time_discovered_started_ns"] = None
            entry["time_discovered_completed_ns"] = (
                done_ns + offset if done_ns is not None else None
            )
            entry["retired"] = done_ns is not None
            entry["timeout_ms"] = g.timeout_ms if g is not None else 0
            entry["is_p2p"] = bool(e.flags & FL_P2P)
            out.append(entry)
        result: dict[str, Any] = {
            "version": VERSION,
            "pg_config": {
                g.name: {"name": g.name, "desc": g.desc, "ranks": str(g.ranks)}
                for g in groups.values()
                if g is not None
            },
            "comm_lib_version": self.comm_lib_version,
            "pg_status": status,
        }
        if include_collectives:
            result["entries"] = out
        return result


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


def recorder() -> Recorder | None:
    return _RECORDER[0] if _RECORDER else None


def make_recorder(
    init: Callable[[int], int], snapshot: Callable[[int, int], int], lib_name: str
) -> Recorder | None:
    """The process's recorder, made on first use; None when
    TORCH_FR_BUFFER_SIZE is 0."""
    with _LOCK:
        if _RECORDER:
            return _RECORDER[0]
        size = buffer_size()
        if size == 0:
            return None
        cell = init(size)
        if not cell:
            return None
        _RECORDER.append(Recorder(size, cell, snapshot, lib_name))
        return _RECORDER[0]


def global_rank(fallback: int) -> int:
    return dist.get_rank() if dist.is_initialized() else fallback


def write_dump(rank: int) -> str | None:
    """Write this process's dump, the pickle ProcessGroupNCCL writes, to
    `<prefix><rank>`; the path, or None without a recorder. Atomic: a
    temporary file without the prefix in its name (the analyzer picks files
    by it) is renamed into place."""
    rec = recorder()
    if rec is None:
        return None
    data = pickle.dumps(
        rec.dump(True, include_stack_traces(), include_only_active()), protocol=2
    )
    path = dump_prefix() + str(rank)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".mojo-fr-{os.getpid()}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def unprefixed(store: Store) -> Store:
    """The store under every PrefixStore: where ProcessGroupNCCL's
    `globalStore_` signals (getUnderlyingNonPrefixStore)."""
    while isinstance(store, PrefixStore):
        store = store.underlying_store
    return store


def signal_dump(client: Store, rank: int):
    """broadcastDumpSignal: the key every rank polls, the signaller's global
    rank as a native int."""
    client.set(DUMP_KEY, struct.pack("=i", rank))  # ty: ignore[invalid-argument-type] -- the stub says str; Store.set takes bytes too, and torch's value is binary


def ack_dump(client: Store, rank: int):
    client.set(f"{ACK_KEY}/{rank}", "1")


def wait_for_dumps(client: Store, world: int, deadline: float) -> list[int]:
    """Wait until every rank wrote its dump or the deadline (monotonic);
    the ranks that did not."""
    keys = [f"{ACK_KEY}/{r}" for r in range(world)]
    while True:
        missing = [r for r, k in enumerate(keys) if not client.check([k])]
        if not missing or time.monotonic() >= deadline:
            return missing
        time.sleep(0.1)


def dump_now(rank: int, why: str):
    try:
        path = write_dump(rank)
    except Exception as error:
        sys.stderr.write(f"[rank {rank}] flight recorder: dump failed: {error!r}\n")
        return
    if path is not None:
        sys.stderr.write(f"[rank {rank}] flight recorder: {why}, dumped to {path}\n")
        sys.stderr.flush()


class _Poller:
    """ProcessGroupNCCL's heartbeat monitor, for the dump only: polls the
    store for another rank's dump signal and the debug pipe file. One per
    process, on its own store client."""

    def __init__(self, store: Store, rank: int, watch_store: bool):
        self.rank = rank
        self.store = store
        self.watch_store = watch_store
        self.pipe = -1
        stem = os.environ.get("TORCH_NCCL_DEBUG_INFO_PIPE_FILE", "")
        if stem:
            path = f"{stem}{rank}.pipe"
            try:
                if os.path.exists(path):
                    os.unlink(path)
                os.mkfifo(path, 0o666)
                self.pipe = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError as error:
                sys.stderr.write(
                    f"[rank {rank}] flight recorder: no debug pipe {path}: {error!r}\n"
                )

    def run(self):
        client = None
        while True:
            time.sleep(coord_check_s())
            if self.pipe >= 0:
                try:
                    if os.read(self.pipe, 4096):
                        dump_now(self.rank, "dump requested through the debug pipe")
                except BlockingIOError:
                    pass
                except OSError:
                    self.pipe = -1
            rec = recorder()
            if not self.watch_store or rec is None or rec.dumped_on_signal:
                continue
            try:
                if client is None:
                    clone = getattr(self.store, "clone", None)
                    client = clone() if clone is not None else self.store
                if not client.check([DUMP_KEY]):
                    continue
                signaller = struct.unpack("=i", bytes(client.get(DUMP_KEY)))[0]
            except Exception:
                client = None  # the store went away (the job is ending): retry
                continue
            rec.dumped_on_signal = True
            dump_now(self.rank, f"rank {signaller} signalled a dump")
            try:
                ack_dump(client, self.rank)
            except Exception:
                client = None


_POLLER: list[_Poller] = []


def start_poller(store: Store, rank: int):
    """Once per process, with the first watched group's store."""
    watch_store = dump_on_timeout()
    with _LOCK:
        if _POLLER or not (
            watch_store or os.environ.get("TORCH_NCCL_DEBUG_INFO_PIPE_FILE")
        ):
            return
        poller = _Poller(unprefixed(store), rank, watch_store)
        _POLLER.append(poller)
    threading.Thread(target=poller.run, name="mojo-pg-fr-monitor", daemon=True).start()


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
