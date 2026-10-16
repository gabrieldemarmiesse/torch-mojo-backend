"""The `mojo` torch.distributed backend: a thin adapter over the Mojo
process-group core (tmb/backend/pg.mojo).

Every collective runs on a per-device comm stream: Mojo makes that stream
wait for the caller's current stream, issues the NCCL / RCCL / mojoccl call
on it, and the adapter records the touched buffers on it (their release is
then fenced) and returns a Work tied to an event on the comm stream, so
whoever waits on it, or on its device Future (the DDP reducer), orders their
own stream after the collective without blocking the host — the contract
ProcessGroupNCCL implements. CPU tensors go to a private gloo group.
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import datetime
import functools
import inspect
import os
import platform
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from typing import TypeVar, cast

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    AllgatherOptions,
    AllreduceCoalescedOptions,
    AllreduceOptions,
    AllToAllOptions,
    BarrierOptions,
    BroadcastOptions,
    GatherOptions,
    PrefixStore,
    ReduceOp,
    ReduceOptions,
    ReduceScatterOptions,
    ScatterOptions,
    Store,
    Work,
)

from torch_mojo_backend import native
from torch_mojo_backend.distributed import flight_recorder, nccl
from torch_mojo_backend.distributed.store_worker import StoreWorker as _StoreWorker
from torch_mojo_backend.native import device_module

_PRIVATEUSE1 = 20  # c10::DeviceType::PrivateUse1

_NCCL_DTYPE_OF: dict[torch.dtype, int] = {
    torch.int8: nccl.NCCL_INT8,
    torch.uint8: nccl.NCCL_UINT8,
    torch.bool: nccl.NCCL_UINT8,
    torch.int32: nccl.NCCL_INT32,
    torch.uint32: nccl.NCCL_UINT32,
    torch.int64: nccl.NCCL_INT64,
    torch.uint64: nccl.NCCL_UINT64,
    torch.float16: nccl.NCCL_FLOAT16,
    torch.float32: nccl.NCCL_FLOAT32,
    torch.float64: nccl.NCCL_FLOAT64,
    torch.bfloat16: nccl.NCCL_BFLOAT16,
}


def _nccl_dtype(dtype: torch.dtype) -> int:
    try:
        return _NCCL_DTYPE_OF[dtype]
    except KeyError:
        raise TypeError(
            f"dtype {dtype} is not supported by the mojo NCCL/RCCL backend"
        ) from None


def _nccl_red_op(
    op: ReduceOp | ReduceOp.RedOpType, dtype: torch.dtype = torch.float32
) -> int:
    if dtype == torch.bool:
        # bool reduces as uint8: SUM and PRODUCT become MAX and MIN (logical
        # or / and, no overflow); AVG has no meaning -- ProcessGroupNCCL's rules.
        if op == ReduceOp.SUM or op == ReduceOp.MAX:
            return nccl.NCCL_MAX
        if op == ReduceOp.PRODUCT or op == ReduceOp.MIN:
            return nccl.NCCL_MIN
        raise TypeError(f"ReduceOp {op} is not supported for bool tensors")
    if op == ReduceOp.SUM:
        return nccl.NCCL_SUM
    if op == ReduceOp.PRODUCT:
        return nccl.NCCL_PROD
    if op == ReduceOp.MAX:
        return nccl.NCCL_MAX
    if op == ReduceOp.MIN:
        return nccl.NCCL_MIN
    if op == ReduceOp.AVG:
        return nccl.NCCL_AVG
    raise NotImplementedError(
        f"ReduceOp {op} is not supported by the mojo NCCL/RCCL backend"
    )


# The collective entry points `_loud` wrapped, for `_never_traced`.
_LOUD: set[Callable[..., object]] = set()

# TORCH_MOJO_BACKEND_PG_WATCHDOG -> pg.mojo's watchdog action
_WATCHDOG_ACTIONS = {"off": 0, "0": 0, "report": 1, "abort": 2}
# How long a rank whose collective timed out waits for the other ranks'
# reports before printing the desync report (bounded by the timeout too).
_DESYNC_WAIT_S = 10.0
_DESYNC_KEY = "mojo_watchdog"
# Operations every rank issues with the same arguments: only these compare
# across ranks (a send pairs with a differently-shaped recv).
_COLLECTIVE_OPS = frozenset(
    ("allreduce", "broadcast", "reduce", "allgather", "reduce_scatter")
)

# Every pipe end of every watchdog in this process. A forked child (a
# DataLoader worker) closes its copies, or the reporter never sees EOF;
# creating, registering and closing them happen under the lock fork takes.
_FORK_LOCK = threading.Lock()
_WATCHDOG_FDS: set[int] = set()


class _Live:
    """A process group whose watchdog thread runs, for the exit hook.
    `users` counts exit-hook calls in flight on its handle: shutdown() waits
    for them before destroying it, without holding the lock across its own
    (possibly long) drain."""

    def __init__(self, core: _Core, handle: int):
        self.core = core
        self.handle = handle
        self.users = 0
        self.dead = False  # destroyed by a __del__ on the hook's own thread


_LIVE_WATCHDOGS: dict[int, _Live] = {}
# Any core, for the process-wide exit flag (set before any group is looked
# at, so a group being shut down on another thread is covered too).
_EXIT_CORE: list[_Core] = []
# The thread running the exit hook: a shutdown() on it (cyclic GC running a
# __del__ inside the hook) must not wait for the hook's own use.
_EXIT_THREAD: list[int] = []


def _before_fork():
    _FORK_LOCK.acquire()


def _after_fork_in_parent():
    _FORK_LOCK.release()


def _after_fork_in_child():
    for fd in _WATCHDOG_FDS:
        with contextlib.suppress(OSError):
            os.close(fd)
    _WATCHDOG_FDS.clear()
    _LIVE_WATCHDOGS.clear()  # the threads did not survive the fork
    global _LIVE_LOCK
    _LIVE_LOCK = threading.Condition(threading.RLock())  # its owner may be gone
    _FORK_LOCK.release()


os.register_at_fork(
    before=_before_fork,
    after_in_parent=_after_fork_in_parent,
    after_in_child=_after_fork_in_child,
)


def _watchdog_pipe() -> tuple[int, int]:
    with _FORK_LOCK:
        read, write = os.pipe()
        _WATCHDOG_FDS.update((read, write))
    return read, write


def _close_watchdog_fd(fd: int):
    with _FORK_LOCK:
        _WATCHDOG_FDS.discard(fd)
        with contextlib.suppress(OSError):
            os.close(fd)


# Guards _LIVE_WATCHDOGS and each entry's `users` (reentrant: a __del__ may
# run inside); held only for bookkeeping, never across a native call.
_LIVE_LOCK = threading.Condition(threading.RLock())


@atexit.register
def _stop_watchdogs():
    """Give the collectives in flight a short grace, report what is still
    pending (a hung job stopped by Ctrl-C or sys.exit included) and join the
    watchdog threads before MAX and the driver tear down: one still polling
    then would take the teardown for a device fault."""
    for core in _EXIT_CORE[:1]:
        core.watchdog_process_exiting()
    _EXIT_THREAD.append(threading.get_ident())
    with _LIVE_LOCK:
        keys = list(_LIVE_WATCHDOGS)
    for key in keys:
        with _LIVE_LOCK:  # a group shut down meanwhile is gone from it
            live = _LIVE_WATCHDOGS.get(key)
            if live is None:
                continue
            live.users += 1
        try:
            if not live.dead:
                live.core.watchdog_exit(live.handle)
        finally:
            with _LIVE_LOCK:
                live.users -= 1
                _LIVE_LOCK.notify_all()
    with _LIVE_LOCK:
        _LIVE_WATCHDOGS.clear()


def _watchdog_action() -> int:
    value = os.environ.get("TORCH_MOJO_BACKEND_PG_WATCHDOG", "abort").strip().lower()
    try:
        return _WATCHDOG_ACTIONS[value]
    except KeyError:
        raise ValueError(
            f"TORCH_MOJO_BACKEND_PG_WATCHDOG={value!r}: expected abort, report or off"
        ) from None


def _parse_report(line: str) -> dict[str, str]:
    """watchdog.mojo's machine line: `key=value` pairs."""
    return dict(item.split("=", 1) for item in line.split() if "=" in item)


_REDOP_NAMES = {"0": "sum", "1": "prod", "2": "max", "3": "min", "4": "avg"}


def _describe(report: dict[str, str]) -> str:
    where = report.get("where")
    if where == "host":
        return f"{report.get('op')} call blocked on the host"
    if where == "error":
        return "a device error"
    text = f"{report.get('op')} {report.get('count')} x {report.get('dtype')}"
    if report.get("root", "-1") != "-1":
        peer = report.get("op") in ("send", "recv", "send/recv")
        text += f" {'peer' if peer else 'root'} {report['root']}"
    redop = report.get("redop", "-1")
    if redop != "-1":
        text += " " + _REDOP_NAMES.get(redop, f"redop {redop}")
    if int(report.get("nops", "1")) > 1:
        text += f" ({report['nops']} calls in one NCCL group)"
    return text


def _ranks(ranks: list[int]) -> str:
    shown = ", ".join(map(str, ranks[:16]))
    return shown + (f" and {len(ranks) - 16} more" if len(ranks) > 16 else "")


def _set_index(client: Store, rank: int, gen: int, names: list[str]) -> bool:
    """Rewrite the rank's index to `names` unless a newer generation is
    already there: a write that lands late (its worker was abandoned)
    never rolls a newer index back. compare_set makes it atomic."""
    key = _index_key(rank)
    new = " ".join([str(gen), *names])
    for _ in range(8):
        current = bytes(client.get(key)).decode() if client.check([key]) else ""
        if current and int(current.split()[0]) >= gen:
            return True
        if bytes(client.compare_set(key, current, new)).decode() == new:
            return True
    return False


def _index_key(rank: int) -> str:
    """A rank's index: a generation, then the names of its active incident
    records; rewritten (never deleted) as incidents start and end."""
    return f"{_DESYNC_KEY}/{rank}"


def _collect(
    store: Store, world: int, deadline: float
) -> dict[int, list[dict[str, str]]]:
    """The active incident records of every rank, waiting until each rank has
    one or the deadline (time.monotonic) passes. Records are immutable and
    indexes only ever rewritten, so a key read here never disappears under
    the read."""
    index = [_index_key(r) for r in range(world)]
    while time.monotonic() < deadline:
        if store.check(index) and all(
            len(bytes(v).split()) > 1 for v in store.multi_get(index)
        ):
            break
        time.sleep(0.2)
    present = [r for r in range(world) if store.check([index[r]])]
    names = store.multi_get([index[r] for r in present]) if present else []
    listed = [
        (r, n) for r, v in zip(present, names) for n in bytes(v).decode().split()[1:]
    ]
    lines = store.multi_get([n for _, n in listed]) if listed else []
    return {
        r: [
            _parse_report(bytes(v).decode())
            for (q, _), v in zip(listed, lines)
            if q == r
        ]
        for r in present
    }


def _pick(records: list[dict[str, str]], mine: dict[str, str]) -> dict[str, str]:
    """The record of the same communicator as this rank's, else the first."""
    for report in records:
        if report.get("tid") == mine.get("tid"):
            return report
    return records[0]


def _comparable(report: dict[str, str]) -> bool:
    """Only the collectives every rank issues identically compare: their
    position counts nothing else (no point-to-point, no grouping)."""
    return (
        report.get("where") == "device"
        and report.get("op") in _COLLECTIVE_OPS
        and report.get("cpos", "0") != "0"
    )


def _desync_lines(
    mine: dict[str, str], reports: dict[int, list[dict[str, str]]], world: int
) -> list[str]:
    seq = int(mine["seq"])
    cpos = int(mine.get("cpos", "0"))
    my_span = (cpos, cpos + int(mine.get("nops", "1")) - 1)
    where_mine = f" (collective #{cpos})" if _comparable(mine) else ""
    out = [
        f"desync report for #{seq}{where_mine} ({_describe(mine)}), from the "
        "ranks whose watchdog fired:"
    ]
    same, missing, diverged = [], [], []
    for r in range(world):
        records = reports.get(r)
        if not records:
            missing.append(r)
            continue
        report = _pick(records, mine)
        k = int(report.get("seq", "0"))
        if not (_comparable(report) and _comparable(mine)):
            out.append(
                f"  rank {r}: waiting in #{k} {_describe(report)} (not compared:"
                " point-to-point, a mixed group or not a device wait)"
            )
            continue
        c = int(report["cpos"])
        span = (c, c + int(report.get("nops", "1")) - 1)
        if span == my_span and report.get("fp") == mine.get("fp"):
            same.append(r)
            continue
        if span != my_span and span[0] <= my_span[1] and my_span[0] <= span[1]:
            # The same calls, grouped differently: their fingerprints differ
            # by construction.
            out.append(
                f"  rank {r}: waiting in collective #{c} {_describe(report)}"
                " (not compared: grouped differently)"
            )
            continue
        diverged.append(r)
        if span == my_span:
            where = "the same position, different arguments"
        else:
            where = "behind" if c < cpos else "ahead"
        out.append(
            f"  rank {r}: waiting in collective #{c} {_describe(report)}  <-- {where}"
        )
    if same:
        out.append(f"  waiting in this same collective: ranks {_ranks(same)}")
    if missing:
        out.append(
            f"  no report: ranks {_ranks(missing)}. Their watchdog did not fire: "
            "usually they never issued this collective or are blocked outside "
            "a GPU collective of this group (on the host, in another process "
            "group, or exited); a rank whose report could not be sent looks "
            "the same. Look there first; `py-spy dump --pid <pid>` shows where "
            "a process is."
        )
    if diverged:
        out.append(
            "  The ranks marked above issued a different collective, or a "
            "different number of them, before this one."
        )
    if not (missing or diverged):
        out.append(
            "  Inconclusive: every rank that reported is waiting where this one "
            "is, or in work that does not compare across ranks. Either it is "
            "stuck itself (a peer GPU or link failure), or the ranks diverged "
            "in a way the arguments do not show."
        )
    return out


def _report_desync(
    line: str,
    store: _StoreWorker,
    rank: int,
    world: int,
    wait_s: float,
    published: dict[str, str] | None = None,
    gen: int = 1,
    incarnation: str = "0",
):
    """Publish this rank's stuck collective as an incident record (index
    generation `gen`; the key carries the reporter's `incarnation`, so a
    late write of a previous incarnation of the group never lands on this
    one's record), collect the other ranks' for at most `wait_s` (1 s for a
    report made at interpreter exit), and print which ranks diverged.
    `published` maps this rank's communicators to their active record."""
    mine = _parse_report(line)
    if mine.get("exit") == "1":
        wait_s = min(wait_s, 1.0)
    deadline = time.monotonic() + wait_s
    active = published if published is not None else {}
    name = f"{_DESYNC_KEY}/{rank}/{incarnation}/{mine.get('id')}"
    active[mine.get("tid", "0")] = name
    names = list(active.values())

    def publish(client: Store) -> bool:
        client.set(name, line)
        return _set_index(client, rank, gen, names)

    reports: dict[int, list[dict[str, str]]] = {}
    if store.run(publish, deadline - time.monotonic(), gen):
        reports = (
            store.run(
                lambda client: _collect(client, world, deadline),
                deadline - time.monotonic() + 2,
            )
            or {}
        )
    reports[rank] = [mine]
    out = _desync_lines(mine, reports, world)
    prefix = f"[rank {rank}] "
    sys.stderr.write("".join(prefix + text + "\n" for text in out))
    sys.stderr.flush()


def _desync_reporter(
    notify_fd: int,
    ack_fd: int,
    store: Store,
    rank: int,
    world: int,
    wait_s: float,
    fr_store: Store | None = None,
):
    """Thread body: report each incident line of the Mojo watchdog and
    acknowledge it with its incident's byte; a `resolved` line takes that
    communicator's record out of this rank's index (report mode). With
    `fr_store` (the store under every prefix), each incident also dumps the
    flight recorder of every rank first. Ends, with the index emptied and
    its store workers closed, when the process group closes the pipe."""
    published: dict[str, str] = {}
    worker = _StoreWorker(store)
    fr_worker = _StoreWorker(fr_store) if fr_store is not None else None
    pending = b""
    gen = 0
    incarnation = f"{time.time_ns():x}{os.getpid():x}"

    def next_gen() -> int:
        # The clock, not a count from 1: a group re-created under the same
        # name on the same store must outrank the last incarnation's index.
        nonlocal gen
        gen = max(gen + 1, time.time_ns())
        worker.latest = gen  # queued index updates of older ones are dropped
        return gen

    def rewrite_index():
        g, names = next_gen(), list(published.values())
        worker.run(lambda client: _set_index(client, rank, g, names), 2.0, g)

    try:
        while chunk := os.read(notify_fd, 4096):
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for raw in lines:
                line = raw.decode().strip()
                fields = _parse_report(line)
                if line.startswith("resolved"):
                    if published.pop(fields.get("tid", ""), None) is not None:
                        rewrite_index()
                    continue
                try:
                    fr_gen = 0  # not `gen`: next_gen() owns that one
                    if fr_worker is not None and fr_store is not None:
                        fr_gen = _fr_incident(fields, fr_worker, fr_store, rank)
                    _report_desync(
                        line,
                        worker,
                        rank,
                        world,
                        wait_s,
                        published,
                        next_gen(),
                        incarnation,
                    )
                    if fr_worker is not None:
                        _fr_wait_for_peers(fields, fr_worker, rank, fr_gen)
                except Exception:
                    traceback.print_exc(file=sys.stderr)
                finally:
                    with contextlib.suppress(Exception):
                        sys.stdout.flush()  # the process may _exit next
                    with contextlib.suppress(OSError):
                        os.write(ack_fd, bytes([int(fields.get("id", "0")) % 255 + 1]))
        if published:
            published.clear()
            rewrite_index()
    finally:
        worker.close()
        if fr_worker is not None:
            fr_worker.close()
        _close_watchdog_fd(notify_fd)
        _close_watchdog_fd(ack_fd)


def _fr_incident(
    fields: dict[str, str], worker: _StoreWorker, store: Store, rank: int
) -> int:
    """This rank's watchdog fired: signal every rank to dump its flight
    recorder (ProcessGroupNCCL's broadcastDumpSignal, plus a new generation,
    returned; 0 when the store did not answer), dump this one's and print
    the Python stack of the collective that timed out. Its acknowledgement
    also covers the generations this process had yet to serve."""
    grank = flight_recorder.global_rank(rank)
    gen = worker.run(lambda client: flight_recorder.signal_dump(client, grank), 2.0)
    claimed = flight_recorder.claim(store, gen) if gen else None
    dumped = flight_recorder.dump_now(
        grank, "collective timeout", flight_recorder.include_stack_traces()
    )
    acked = False
    if gen and dumped:  # an acknowledgement means a file is there
        assert claimed is not None
        mine = sorted({*claimed.gens, gen})
        acked = (
            worker.run(
                lambda client: flight_recorder.ack_dump(client, grank, mine) or True,
                2.0,
            )
            is True
        )
    if claimed is not None:
        flight_recorder.finish(claimed, dumped, acked)
    rec = flight_recorder.recorder()
    frames = rec.stack(int(fields.get("fr", "0"))) if rec is not None else []
    if frames:
        lines = [
            f"[rank {rank}] flight recorder: Python stack of that collective, "
            "innermost first:\n"
        ] + [
            f'[rank {rank}]   File "{f["filename"]}", line {f["line"]}, in {f["name"]}\n'
            for f in frames
        ]
        sys.stderr.write("".join(lines))
        sys.stderr.flush()
    return gen or 0


def _fr_wait_for_peers(
    fields: dict[str, str], worker: _StoreWorker, rank: int, gen: int
):
    """Give the other ranks TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC to dump
    generation `gen` before this one aborts (ProcessGroupNCCL sleeps 4x
    that instead)."""
    rec = flight_recorder.recorder()
    if not gen or rec is None:
        return
    # The ranks of this process's groups: in a job whose default group is
    # gloo, ranks without a mojo group never dump.
    ranks = sorted(rec.peer_ranks())
    wait = 0.5 if fields.get("exit") == "1" else flight_recorder.dump_wait_s()
    deadline = time.monotonic() + wait
    missing = worker.run(
        lambda client: flight_recorder.wait_for_dumps(client, ranks, gen, deadline),
        wait + 2.0,
    )
    if missing:
        sys.stderr.write(
            f"[rank {rank}] flight recorder: no dump from ranks {_ranks(missing)} "
            f"within {wait:.1f} s\n"
        )
        sys.stderr.flush()


def _loud(method: Callable[..., object]) -> Callable[..., object]:
    """torch swallows exceptions raised inside a Python ProcessGroup method
    called from the autograd thread; print the traceback before re-raising.

    A failure inside a coalescing block also unwinds that block: torch's
    `_coalescing_manager` has no `finally`, so `_end_coalescing` never runs
    after a raise and the communicator would stay inside an open NCCL group
    with deferred copy-backs queued against operations that were never
    submitted.
    """

    @functools.wraps(method)
    def wrapper(self: MojoProcessGroup, *args: object, **kwargs: object) -> object:
        try:
            return method(self, *args, **kwargs)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            self._abort_coalescing()
            raise
        finally:
            self._fr_tls.tag = 0

    _LOUD.add(wrapper)
    return wrapper


_C = TypeVar("_C", bound=type)
# Private methods of MojoProcessGroup that torch.distributed calls itself.
_C10D_HOOKS = frozenset(
    {"_get_backend", "_set_group_name", "_start_coalescing", "_end_coalescing"}
)


def _never_traced(cls: _C) -> _C:
    """Keep dynamo out of every method of `cls`.

    torch calls a Python process group (and its Work) back from C++, so when
    compiled code runs a collective eagerly -- after a graph break, or a
    DDP/FSDP hook dynamo does not trace -- the method's frame starts under
    dynamo's frame hook, which then tries to compile it (and fails on
    `record_stream`, which functionalization rejects). ProcessGroupNCCL is
    C++ and never meets the hook; this makes ours behave the same.

    Only the methods torch calls are wrapped -- the public ones, the `_loud`
    collectives and `_C10D_HOOKS` -- not their helpers: `disable` is
    recursive, and each wrapper costs ~0.4 us on the eager path."""
    for name, attr in list(vars(cls).items()):
        if inspect.isfunction(attr) and (
            not name.startswith("_") or attr in _LOUD or name in _C10D_HOOKS
        ):
            setattr(cls, name, torch.compiler.disable(attr))
    return cls


@_never_traced
class _StreamWork(Work):
    """Completion is a point on the comm stream: `wait()` makes the waiter's
    current stream follow it. The device Future is only built for callers
    that ask for one."""

    def __init__(self, group: MojoProcessGroup, index: int, result: list[torch.Tensor]):
        super().__init__()
        self._group = group
        self._index = index
        self._result = result
        self._event = group._ready[index].record_event()
        self._future: torch.futures.Future[list[torch.Tensor]] | None = None

    def wait(self, timeout: datetime.timedelta | None = None) -> bool:
        if timeout:  # timedelta(0) is c10d's kNoTimeout
            raise RuntimeError(
                "the mojo process group does not implement wait() with a finite timeout"
            )
        current = torch.accelerator.current_stream(self._index)
        current.wait_event(self._event)
        for t in self._result:  # as a device Future does for its waiter
            t.record_stream(current)
        return True

    def is_completed(self) -> bool:
        return self._event.query()

    def result(self) -> list[torch.Tensor]:
        return self._result

    def get_future(self) -> torch.futures.Future[list[torch.Tensor]]:
        if self._future is None:
            # set_result records the Future's events on the current stream. The
            # comm stream's tail may already hold later collectives, so a side
            # stream that follows this work's event only carries them.
            side = self._group._future_stream(self._index)
            side.wait_event(self._event)
            self._future = torch.futures.Future(
                devices=[torch.device("mojo", self._index)]
            )
            with device_module.stream(side):
                self._future.set_result(self._result)
        return self._future


class _Core:
    """The tmb_pg_* entries of the Mojo backend, taken from its vtable
    (functions of an imported Mojo module are not exported symbols)."""

    def __init__(self):
        lib = native.backend_lib()
        lib.tmb_pg_vtable.restype = ctypes.c_void_p
        table = ctypes.cast(lib.tmb_pg_vtable(), ctypes.POINTER(ctypes.c_void_p))
        i32, i64, vp, sz = (
            ctypes.c_int32,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_size_t,
        )
        # the order of pg.mojo's pg_vtable
        self.create = ctypes.CFUNCTYPE(vp, ctypes.c_char_p, i32, i32)(table[0])
        self.destroy = ctypes.CFUNCTYPE(None, vp)(table[1])
        self.version = ctypes.CFUNCTYPE(i32, vp)(table[2])
        self.unique_id = ctypes.CFUNCTYPE(i32, vp, vp)(table[3])
        self.init_device = ctypes.CFUNCTYPE(i32, vp, i32, vp)(table[4])
        self.comm_stream = ctypes.CFUNCTYPE(i64, vp, i32)(table[5])
        # the collectives and group_end take the flight recorder's tag last
        self.allreduce = ctypes.CFUNCTYPE(i32, vp, i32, vp, sz, i32, i32, i64)(table[6])
        self.broadcast = ctypes.CFUNCTYPE(i32, vp, i32, vp, sz, i32, i32, i64)(table[7])
        self.reduce = ctypes.CFUNCTYPE(i32, vp, i32, vp, sz, i32, i32, i32, i64)(
            table[8]
        )
        self.allgather = ctypes.CFUNCTYPE(i32, vp, i32, vp, vp, sz, i32, i64)(table[9])
        self.reduce_scatter = ctypes.CFUNCTYPE(i32, vp, i32, vp, vp, sz, i32, i32, i64)(
            table[10]
        )
        self.send = ctypes.CFUNCTYPE(i32, vp, i32, vp, sz, i32, i32, i64)(table[11])
        self.recv = ctypes.CFUNCTYPE(i32, vp, i32, vp, sz, i32, i32, i64)(table[12])
        self.group_start = ctypes.CFUNCTYPE(i32, vp)(table[13])
        self.group_end = ctypes.CFUNCTYPE(i32, vp, i64)(table[14])
        self.async_error = ctypes.CFUNCTYPE(i32, vp, i32)(table[15])
        self.abort = ctypes.CFUNCTYPE(i32, vp, i32)(table[16])
        self.synchronize_comm = ctypes.CFUNCTYPE(i32, vp, i32)(table[17])
        # called by the compiled-graph ops (tmb/graph/collectives.mojo), not here
        self.graph_entries = (table[18], table[19])
        self.get_error = ctypes.cast(native.shim().tmb_get_error, vp).value
        self.watchdog = ctypes.CFUNCTYPE(i32, vp, i64, i32, i32, i32, i64)(table[20])
        self.watchdog_exit = ctypes.CFUNCTYPE(None, vp)(table[21])
        self.watchdog_test_fault = ctypes.CFUNCTYPE(None, vp, i32)(table[22])
        self.watchdog_test_limit = ctypes.CFUNCTYPE(None, vp, i32, i32)(table[23])
        self.watchdog_process_exiting = ctypes.CFUNCTYPE(None)(table[24])
        self.fr_init = ctypes.CFUNCTYPE(vp, i64)(table[25])
        self.fr_attach = ctypes.CFUNCTYPE(i32, vp, i64)(table[26])
        self.fr_snapshot = ctypes.CFUNCTYPE(i64, vp, i64)(table[27])
        self.fr_selftest = ctypes.CFUNCTYPE(i64)(table[28])
        self.fr_after_fork = ctypes.CFUNCTYPE(None)(table[29])

    def check(self, rc: int, what: str):
        if rc != 0:
            raise RuntimeError(f"{what} failed: {native.last_error()}")


@_never_traced
class MojoProcessGroup(dist.ProcessGroup):
    def __init__(
        self, store: Store, rank: int, world_size: int, timeout: datetime.timedelta
    ):
        super().__init__(rank, world_size)  # ty: ignore[missing-argument, invalid-argument-type] -- the 2-arg base ctor is the one a Python subclass can use
        self._store = store
        self._timeout = timeout
        # CPU tensors (object collectives, barriers on CPU groups) go to gloo.
        self._gloo = cast(
            dist.ProcessGroup,  # the gloo stub declares none of the collectives; the base class does
            dist.ProcessGroupGloo(
                PrefixStore("mojo-cpu-gloo", store), rank, world_size, timeout
            ),
        )
        self._handle = None
        self._watchdog_fds: tuple[int, int] | None = None
        self._handle_pid = 0
        self._ready: dict[int, torch.Stream] = {}
        self._future_streams: dict[int, torch.Stream] = {}
        self._seq = 0
        self._group_name = ""
        self._coalescing: int | None = (
            None  # device index while batch_isend_irecv coalesces
        )
        self._coalesced_tensors: list[torch.Tensor] = []
        self._coalescing_thread: int | None = None
        # Work a coalesced call cannot do yet: inside a group nothing is
        # submitted until group_end, so copy-backs and record_stream fences
        # wait there. Holding the tensors also keeps every staging buffer
        # alive until the operations that read and write it are submitted.
        self._coalesced_records: list[torch.Tensor] = []
        self._coalesced_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        # Flight recorder (flight_recorder.py), None when not recording; the
        # point-to-point flags of the open coalescing block's operations.
        self._fr: flight_recorder.Recorder | None = None
        self._fr_block: list[bool] | None = None
        self._fr_pg_id = -1
        # The tag word of the call in progress on each thread, which every
        # pg.mojo entry takes (flight_recorder.mojo `fr_set_tag`).
        self._fr_tls = threading.local()
        # Communicator creation is collective: do it now, while every rank is
        # here, on the rank's current device (one visible GPU per torchrun rank).
        # A machine with no mojo device at all gets no communicator: only the
        # gloo delegation for CPU tensors works.
        if device_module.device_count() > 0:
            current = device_module.current_device()
            if device_module.get_device_properties(current).api in ("cuda", "hip"):
                self._ensure(current)

    @functools.cached_property
    def _core(self) -> _Core:
        # The native communicator ABI is irrelevant to CPU/Gloo collectives.
        if platform.machine() not in ("x86_64", "AMD64"):
            raise NotImplementedError(
                "the mojo process group passes ncclUniqueId with the x86-64 SysV layout; "
                f"{platform.machine()} is not supported yet"
            )
        return _Core()

    def _initialize_collectives(self):
        self._path = nccl.library_path()
        self._handle = self._core.create(
            str(self._path).encode(), self.rank(), self.size()
        )
        if not self._handle:
            raise RuntimeError(
                f"could not load the collectives library: {native.last_error()}"
            )
        self._handle_pid = os.getpid()
        version = self._core.version(self._handle)
        vendor = nccl.vendor_name()
        name = "mojoccl" if nccl.uses_mojoccl() else vendor
        self._fr_lib = ("rccl" if vendor == "rccl" else "nccl", version)
        if os.environ.get("TORCH_MOJO_BACKEND_TRACE", "1") != "0":
            print(
                f"[TRACE] collectives via {self._path} ({name} version {version})",
                file=sys.stderr,
                flush=True,
            )
        self._start_watchdog()

    def _start_watchdog(self):
        """pg.mojo's collective watchdog, on `init_process_group(timeout=...)`,
        plus the thread that writes its desync report through the store."""
        action = _watchdog_action()
        seconds = self._timeout.total_seconds()
        timeout_ms = max(min(int(seconds * 1000), 10**12), 0)
        if action == _WATCHDOG_ACTIONS["off"] or seconds <= 0:
            # No watchdog; the timeout still bounds destroy's drain.
            self._core.check(
                self._core.watchdog(self._handle, timeout_ms, 0, -1, -1, 0),
                "setting the process group timeout",
            )
            return
        wait_s = min(_DESYNC_WAIT_S, seconds)
        fr = flight_recorder.make_recorder(
            self._core.fr_init,
            self._core.fr_snapshot,
            self._core.fr_after_fork,
            self._fr_lib[0],
        )
        fr_dumps = fr is not None and flight_recorder.dump_on_timeout()
        # Before the reporter acknowledges an incident: the signal and the
        # acknowledgement (2 s each at most), this rank's dump, and the wait
        # for the other ranks' (its own 2 s of store slack included).
        fr_wait_s = flight_recorder.dump_wait_s() + 20 if fr_dumps else 0
        notify_r, notify_w = _watchdog_pipe()
        try:
            ack_r, ack_w = _watchdog_pipe()
        except OSError:
            _close_watchdog_fd(notify_r)
            _close_watchdog_fd(notify_w)
            raise
        os.set_blocking(notify_w, False)  # the watchdog's writes never block
        rc = self._core.watchdog(
            self._handle,
            timeout_ms,
            action,
            notify_w,
            ack_r,
            int((wait_s + 5 + fr_wait_s) * 1000),
        )
        if rc != 0:
            for fd in (notify_r, notify_w, ack_r, ack_w):
                _close_watchdog_fd(fd)
            self._core.check(rc, "starting the collective watchdog")
        self._watchdog_fds = (notify_w, ack_r)
        if fr is not None:
            pg_id = fr.add_group(self)
            if self._core.fr_attach(self._handle, pg_id) == 0:
                self._fr_pg_id = pg_id
                v = self._fr_lib[1]
                fr.comm_lib_version = f"{v // 10000}.{v // 100 % 100}.{v % 100}"
                self._fr = fr
                flight_recorder.attach_poller(
                    id(self), self._store, flight_recorder.global_rank(self.rank())
                )
            else:  # the watchdog runs on; this group is just not recorded
                fr.retire(pg_id)
                sys.stderr.write(
                    f"flight recorder off for this process group: {native.last_error()}\n"
                )
        with _LIVE_LOCK:
            _LIVE_WATCHDOGS[id(self)] = _Live(self._core, cast(int, self._handle))
            if not _EXIT_CORE:
                _EXIT_CORE.append(self._core)
        threading.Thread(
            target=_desync_reporter,
            args=(
                notify_r,
                ack_w,
                self._store,
                self.rank(),
                self.size(),
                wait_s,
                flight_recorder.unprefixed(self._store) if fr_dumps else None,
            ),
            name="mojo-pg-desync-report",
            daemon=True,
        ).start()

    # ---- plumbing ------------------------------------------------------------

    def getBackendName(self) -> str:
        return "mojo"

    # A Python ProcessGroup subclass is its own backend: torch's
    # batch_isend_irecv and _coalescing_manager look the backend up with
    # _get_backend(device) and read supports_coalescing on it, and the C++
    # lookup would fail because no backend is registered for "mojo".
    supports_coalescing = True
    supports_splitting = False
    _supports_time_estimate = False

    def _get_backend(self, device: torch.device) -> MojoProcessGroup:
        return self

    def _set_group_name(self, name: str):
        self._group_name = name

    def getGroupName(self) -> str:
        return self._group_name

    def setGroupName(self, name: str):
        self._group_name = name

    def shutdown(self):
        if self._handle_pid != os.getpid():
            # A forked child's copy: the communicators, streams, watchdog and
            # flight recorder monitor belong to the parent (fork closed the
            # pipe copies; flight_recorder.py's fork hook dropped the rest).
            self._handle = None
            self._watchdog_fds = None
            self._ready.clear()
            return
        if self._fr is not None and self._handle is not None:
            self._fr.retire(self._fr_pg_id)
            flight_recorder.detach_poller(id(self))
        handle, self._handle = self._handle, None
        fds, self._watchdog_fds = self._watchdog_fds, None
        self._ready.clear()
        with _LIVE_LOCK:
            live = _LIVE_WATCHDOGS.pop(id(self), None)
            if live is not None and threading.get_ident() in _EXIT_THREAD:
                live.dead = True  # the hook's use on this thread is not running
            else:
                while live is not None and live.users:  # the hook's, bounded
                    _LIVE_LOCK.wait()
        if handle:
            self._core.destroy(handle)  # joins the watchdog thread
        for fd in fds or ():  # the reporter thread sees EOF and ends
            _close_watchdog_fd(fd)

    def _test_watchdog_fault(self, index: int):
        """Test hook: the watchdog sees a driver error on this device's
        communicator, as after a failed event record."""
        self._core.watchdog_test_fault(self._handle, index)

    def _test_watchdog_limit(self, index: int, entries: int):
        """Test hook: the watchdog keeps at most `entries` collectives in
        flight on this device's communicator, then drops commits."""
        self._core.watchdog_test_limit(self._handle, index, entries)

    def abort(self):
        for index in list(self._ready):
            self._core.check(self._core.abort(self._handle, index), "ncclCommAbort")
            del self._ready[index]

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def _is_cpu(self, tensor: torch.Tensor) -> bool:
        device = tensor.device
        if device.type == "cpu":
            return True
        if device.type != "mojo":
            raise ValueError(
                f"the mojo distributed backend handles mojo and cpu tensors, got {tensor.device}"
            )
        return False

    @staticmethod
    def _index(tensor: torch.Tensor) -> int:
        return (
            tensor.device.index
            if tensor.device.index is not None
            else device_module.current_device()
        )

    def _ensure(self, index: int) -> torch.Stream:
        """One communicator per device, bootstrapped through the c10d store.
        The store key is rank-consistent (creation order), never the local
        device index, which differs across ranks without visibility pinning."""
        stream = self._ready.get(index)
        if stream is not None:
            return stream
        if self._coalescing_open():
            raise RuntimeError("cannot create a communicator inside a coalescing block")
        api = device_module.get_device_properties(index).api
        if api not in ("cuda", "hip"):
            raise NotImplementedError(
                f"mojo GPU collectives require CUDA or HIP, got {api}; "
                "CPU tensors can use Gloo"
            )
        if self._handle is None:
            self._initialize_collectives()
        key = f"mojo-ccl-unique-id-{self._seq}"
        self._seq += 1
        if self.rank() == 0:
            buf = ctypes.create_string_buffer(nccl.NCCL_UNIQUE_ID_BYTES)
            self._core.check(self._core.unique_id(self._handle, buf), "ncclGetUniqueId")
            unique_id = buf.raw
            self._store.set(key, unique_id)  # ty: ignore[invalid-argument-type] -- the binding takes bytes too
        else:
            unique_id = bytes(self._store.get(key))
            if len(unique_id) != nccl.NCCL_UNIQUE_ID_BYTES:
                raise RuntimeError(f"bad unique id from store key {key}")
        buf = ctypes.create_string_buffer(unique_id, nccl.NCCL_UNIQUE_ID_BYTES)
        self._core.check(
            self._core.init_device(self._handle, index, buf), "ncclCommInitRank"
        )
        stream_id = self._core.comm_stream(self._handle, index)
        stream = torch.Stream(
            stream_id=stream_id, device_index=index, device_type=_PRIVATEUSE1
        )
        self._ready[index] = stream
        return stream

    def _work(self, index: int, result: list[torch.Tensor]) -> Work:
        return _StreamWork(self, index, result)

    def _future_stream(self, index: int) -> torch.Stream:
        stream = self._future_streams.get(index)
        if stream is None:
            stream = torch.Stream(device=torch.device("mojo", index))
            self._future_streams[index] = stream
        return stream

    def _stage_in(self, index: int, tensor: torch.Tensor) -> torch.Tensor:
        """A dense copy of a non-contiguous input, made on the comm stream after
        it waited for the current stream, so its release never fences compute."""
        if tensor.is_contiguous():
            return tensor
        comm = self._ready[index]
        comm.wait_stream(torch.accelerator.current_stream(index))
        with device_module.stream(comm):
            staged = tensor.contiguous()
        return staged

    def _stage_out(self, index: int, tensor: torch.Tensor) -> torch.Tensor:
        """A fresh dense buffer for a non-contiguous output, on the comm stream."""
        if tensor.is_contiguous():
            return tensor
        with device_module.stream(self._ready[index]):
            staged = torch.empty_like(tensor, memory_format=torch.contiguous_format)
        return staged

    def _record(self, index: int, *tensors: torch.Tensor):
        """Inputs (original and staged): touched by the comm stream, not modified."""
        if self._coalescing_open():
            self._coalesced_records.extend(tensors)
            return
        comm = self._ready[index]
        previous = None
        for t in tensors:
            if t is not previous:  # a contiguous input is its own staging buffer
                t.record_stream(comm)
            previous = t

    def _finish(self, index: int, *pairs: tuple[torch.Tensor, torch.Tensor]):
        """Outputs: copy staged results back on the comm stream and record
        every buffer on it so its release is fenced.

        Inside a coalescing block the operation that fills `staged` is only
        submitted by `group_end`, so both the copy-back and the fence are
        deferred to `_end_coalescing`; copying now would read a buffer the
        collective has not written yet, and a `record_stream` event recorded
        before submission fences nothing.
        """
        if self._coalescing_open():
            self._coalesced_pairs.extend(pairs)
            return
        comm = self._ready[index]
        if any(staged is not tensor for tensor, staged in pairs):
            with device_module.stream(comm):
                for tensor, staged in pairs:
                    if staged is not tensor:
                        tensor.copy_(staged)
                        staged.record_stream(comm)
        for tensor, _ in pairs:
            tensor.record_stream(comm)

    @staticmethod
    def _one_per_rank(tensors: list[torch.Tensor]):
        if len(tensors) != 1:
            raise ValueError(
                "the mojo backend takes exactly one tensor per rank per call"
            )

    def _same_device(self, tensors: list[torch.Tensor]) -> int:
        if not tensors:
            raise ValueError("no tensors")
        first = tensors[0]
        if self._is_cpu(first):
            raise ValueError(
                "all tensors of a collective must be on the same mojo device"
            )
        device = first.device
        for t in tensors:
            if t is not first and t.device != device:
                raise ValueError(
                    "all tensors of a collective must be on the same mojo device"
                )
        return self._index(first)

    @contextlib.contextmanager
    def _group(self):
        """An NCCL group; the end call runs on every exit path so a failed
        submission never leaves the communicator inside an open group."""
        if self._coalescing_open():
            yield
            return
        self._core.check(self._core.group_start(self._handle), "ncclGroupStart")
        try:
            yield
        finally:
            self._core.check(
                self._core.group_end(self._handle, self._fr_tag()), "ncclGroupEnd"
            )

    def _op(self, dtype: torch.dtype, op: ReduceOp | ReduceOp.RedOpType) -> int:
        return _nccl_red_op(op, dtype)

    def _fr_op(
        self,
        name: str,
        inputs: Sequence[flight_recorder.Shaped],
        outputs: Sequence[flight_recorder.Shaped],
        p2p: bool = False,
    ):
        """Describe the call about to issue to the flight recorder, under
        ProcessGroupNCCL's name for it (`_loud` clears it after the call)."""
        fr = self._fr
        if fr is None:
            return
        flags = flight_recorder.TAG_P2P if p2p else 0
        block = self._fr_block
        if block is not None and self._coalescing_thread == threading.get_ident():
            flags |= flight_recorder.TAG_MEMBER
            if not block:
                flags |= flight_recorder.TAG_FIRST
            block.append(p2p)
        self._fr_tls.tag = fr.begin(name, inputs, outputs, flags)

    def _fr_tag(self) -> int:
        return getattr(self._fr_tls, "tag", 0)

    def _fr_block_end(self):
        """The end entry of a coalescing block (ProcessGroupNCCL's
        `nccl:coalesced`), committed by its group_end."""
        block, self._fr_block = self._fr_block, None
        if self._fr is not None and block is not None:
            p2p = flight_recorder.TAG_P2P if any(block) else 0
            end = flight_recorder.TAG_END | p2p
            self._fr_tls.tag = self._fr.begin("coalesced", [], [], end)

    def _result(self, index: int, tensors: list[torch.Tensor]) -> Work:
        if self._coalescing_open():
            self._coalesced_tensors.extend(tensors)
            return self._work(
                index, tensors
            )  # placeholder: ordering comes from _end_coalescing's Work
        return self._work(index, tensors)

    # ---- coalescing (torch.distributed.batch_isend_irecv / _coalescing_manager)

    def _coalescing_open(self) -> bool:
        """Whether the caller is inside this group's coalescing block. NCCL
        groups belong to the thread that opened them, so another thread must
        not join (or silently skip) an open block."""
        if self._coalescing is None:
            return False
        if threading.get_ident() != self._coalescing_thread:
            raise RuntimeError(
                "a coalescing block of this process group is open on another thread"
            )
        return True

    def _start_coalescing(self, device: torch.device):
        if self._coalescing is not None:
            raise RuntimeError("coalescing blocks do not nest")
        index = (
            device.index if device.index is not None else device_module.current_device()
        )
        self._ensure(index)
        self._core.check(self._core.group_start(self._handle), "ncclGroupStart")
        self._coalescing = index
        self._coalescing_thread = threading.get_ident()
        self._coalesced_tensors = []
        self._coalesced_records = []
        self._coalesced_pairs = []
        self._fr_block = []

    def _end_coalescing(self, device: torch.device) -> Work:
        """Submit the group, then run the work every call inside it deferred:
        the copy-backs and fences first (on the comm stream), then the Work
        whose completion events follow them."""
        index = (
            self._coalescing
            if self._coalescing is not None
            else self._index_of_device(device)
        )
        self._coalescing = None
        self._coalescing_thread = None
        tensors, self._coalesced_tensors = self._coalesced_tensors, []
        records, self._coalesced_records = self._coalesced_records, []
        pairs, self._coalesced_pairs = self._coalesced_pairs, []
        # Taken out of the group state first: a failed group_end must leave no
        # deferred work behind for the next block to apply.
        self._fr_block_end()
        try:
            self._core.check(
                self._core.group_end(self._handle, self._fr_tag()), "ncclGroupEnd"
            )
        finally:
            self._fr_tls.tag = 0
        self._record(index, *records)
        self._finish(index, *pairs)
        return self._work(index, tensors)

    def _abort_coalescing(self):
        """Unwind a coalescing block that raised. Closing the NCCL group
        submits what was queued (it does not cancel it), so the deferred
        copy-backs and fences still run and the staged buffers stay referenced
        until then; only then is the block's state dropped."""
        if self._coalescing is None:
            return
        index = self._coalescing
        self._coalescing = None
        self._coalescing_thread = None
        self._coalesced_tensors = []
        records, self._coalesced_records = self._coalesced_records, []
        pairs, self._coalesced_pairs = self._coalesced_pairs, []
        try:
            self._fr_block_end()
            self._core.check(
                self._core.group_end(self._handle, self._fr_tag()), "ncclGroupEnd"
            )
            self._record(index, *records)
            self._finish(index, *pairs)
        except Exception:  # the original failure is the one being propagated
            traceback.print_exc(file=sys.stderr)

    @staticmethod
    def _index_of_device(device: torch.device) -> int:
        return (
            device.index if device.index is not None else device_module.current_device()
        )

    # ---- collectives -----------------------------------------------------------

    @_loud
    def allreduce(
        self, tensors: list[torch.Tensor], opts: AllreduceOptions = AllreduceOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.allreduce(tensors, opts)
        self._fr_op("all_reduce", tensors, tensors)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        index = self._same_device(tensors)
        self._ensure(index)
        dtype, op = _nccl_dtype(tensor.dtype), self._op(tensor.dtype, opts.reduceOp)
        staged = self._stage_in(index, tensor)
        self._core.check(
            self._core.allreduce(
                self._handle,
                index,
                staged.data_ptr(),
                staged.numel(),
                dtype,
                op,
                self._fr_tag(),
            ),
            "ncclAllReduce",
        )
        self._finish(index, (tensor, staged))
        return self._result(index, tensors)

    @_loud
    def allreduce_coalesced(
        self,
        tensors: list[torch.Tensor],
        opts: AllreduceCoalescedOptions = AllreduceCoalescedOptions(),
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.allreduce_coalesced(tensors, opts)
        self._fr_op("allreduce_coalesced", tensors, tensors)
        index = self._same_device(tensors)
        self._ensure(index)
        plan = [
            (t, _nccl_dtype(t.dtype), self._op(t.dtype, opts.reduceOp)) for t in tensors
        ]
        staged = [self._stage_in(index, t) for t in tensors]
        with self._group():
            for (t, dtype, op), s in zip(plan, staged):
                self._core.check(
                    self._core.allreduce(
                        self._handle,
                        index,
                        s.data_ptr(),
                        s.numel(),
                        dtype,
                        op,
                        self._fr_tag(),
                    ),
                    "ncclAllReduce",
                )
        self._finish(index, *zip(tensors, staged))
        return self._result(index, tensors)

    @_loud
    def broadcast(
        self, tensors: list[torch.Tensor], opts: BroadcastOptions = BroadcastOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.broadcast(tensors, opts)
        self._fr_op("broadcast", tensors, tensors)
        self._one_per_rank(tensors)
        if opts.rootTensor != 0:
            raise ValueError("rootTensor must be 0 with one tensor per rank")
        tensor = tensors[0]
        index = self._same_device(tensors)
        self._ensure(index)
        dtype = _nccl_dtype(tensor.dtype)
        staged = self._stage_in(index, tensor)
        self._core.check(
            self._core.broadcast(
                self._handle,
                index,
                staged.data_ptr(),
                staged.numel(),
                dtype,
                opts.rootRank,
                self._fr_tag(),
            ),
            "ncclBroadcast",
        )
        self._finish(index, (tensor, staged))
        return self._result(index, tensors)

    @_loud
    def reduce(
        self, tensors: list[torch.Tensor], opts: ReduceOptions = ReduceOptions()
    ) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.reduce(tensors, opts)
        self._fr_op("reduce", tensors, tensors)
        self._one_per_rank(tensors)
        tensor = tensors[0]
        index = self._same_device(tensors)
        self._ensure(index)
        dtype, op = _nccl_dtype(tensor.dtype), self._op(tensor.dtype, opts.reduceOp)
        staged = self._stage_in(index, tensor)
        self._core.check(
            self._core.reduce(
                self._handle,
                index,
                staged.data_ptr(),
                staged.numel(),
                dtype,
                op,
                opts.rootRank,
                self._fr_tag(),
            ),
            "ncclReduce",
        )
        self._finish(index, (tensor, staged))
        return self._result(index, tensors)

    def _allgather_flat(
        self, index: int, output: torch.Tensor, input: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Base all-gather: output (world * input.numel()) <- every rank's input."""
        if output.numel() != input.numel() * self.size():
            raise ValueError(
                "all_gather output must hold world_size copies of the input"
            )
        if output.dtype != input.dtype:
            raise ValueError("all_gather output dtype must match the input")
        dtype = _nccl_dtype(input.dtype)
        staged_in = self._stage_in(index, input)
        staged_out = self._stage_out(index, output)
        self._core.check(
            self._core.allgather(
                self._handle,
                index,
                staged_in.data_ptr(),
                staged_out.data_ptr(),
                staged_in.numel(),
                dtype,
                self._fr_tag(),
            ),
            "ncclAllGather",
        )
        return staged_in, staged_out

    @_loud
    def _allgather_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        opts: AllgatherOptions = AllgatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensor):
            return self._gloo._allgather_base(output_tensor, input_tensor, opts)
        self._fr_op("_all_gather_base", [input_tensor], [output_tensor])
        index = self._same_device([input_tensor, output_tensor])
        self._ensure(index)
        staged_in, staged_out = self._allgather_flat(index, output_tensor, input_tensor)
        self._record(index, input_tensor, staged_in)
        self._finish(index, (output_tensor, staged_out))
        return self._result(index, [output_tensor])

    @_loud
    def allgather(
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions = AllgatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.allgather(output_tensors, input_tensors, opts)
        if output_tensors[0]:
            self._fr_op(
                "all_gather", input_tensors, [flight_recorder.flat(output_tensors[0])]
            )
        self._one_per_rank(input_tensors)
        input = input_tensors[0]
        outputs = output_tensors[0]
        if len(outputs) != self.size():
            raise ValueError("all_gather needs world_size output tensors")
        if any(o.numel() != input.numel() or o.dtype != input.dtype for o in outputs):
            raise NotImplementedError(
                "all_gather with uneven or differently typed tensors is not supported"
            )
        index = self._same_device([input, *outputs])
        comm = self._ensure(index)
        with device_module.stream(comm):
            flat = torch.empty(
                self.size() * input.numel(), dtype=input.dtype, device=input.device
            )
        staged_in, _ = self._allgather_flat(index, flat, input)
        n = input.numel()
        self._record(index, input, staged_in)
        # Each output is staged behind a slice of `flat`, which the gather
        # fills: the copy-back and the fence on flat's storage go through
        # _finish so a coalescing block defers them past its group_end.
        self._finish(
            index,
            *[
                (out, flat[r * n : (r + 1) * n].view(out.shape))
                for r, out in enumerate(outputs)
            ],
        )
        return self._result(index, outputs)

    @_loud
    def allgather_into_tensor_coalesced(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: AllgatherOptions = AllgatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.allgather_into_tensor_coalesced(
                output_tensors, input_tensors, opts
            )
        self._fr_op("all_gather_into_tensor_coalesced", input_tensors, output_tensors)
        if len(output_tensors) != len(input_tensors):
            raise ValueError("coalesced all_gather needs as many outputs as inputs")
        index = self._same_device([*input_tensors, *output_tensors])
        self._ensure(index)
        with self._group():
            pairs = [
                self._allgather_flat(index, o, i)
                for o, i in zip(output_tensors, input_tensors)
            ]
        self._record(index, *input_tensors, *[si for si, _ in pairs])
        self._finish(index, *[(o, so) for o, (_, so) in zip(output_tensors, pairs)])
        return self._result(index, output_tensors)

    def _reduce_scatter_flat(
        self,
        index: int,
        output: torch.Tensor,
        input: torch.Tensor,
        op: ReduceOp | ReduceOp.RedOpType,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input.numel() != output.numel() * self.size():
            raise ValueError(
                "reduce_scatter input must hold world_size chunks of the output"
            )
        if output.dtype != input.dtype:
            raise ValueError("reduce_scatter output dtype must match the input")
        dtype, nccl_op = _nccl_dtype(input.dtype), self._op(input.dtype, op)
        staged_in = self._stage_in(index, input)
        staged_out = self._stage_out(index, output)
        self._core.check(
            self._core.reduce_scatter(
                self._handle,
                index,
                staged_in.data_ptr(),
                staged_out.data_ptr(),
                staged_out.numel(),
                dtype,
                nccl_op,
                self._fr_tag(),
            ),
            "ncclReduceScatter",
        )
        return staged_in, staged_out

    @_loud
    def _reduce_scatter_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work:
        if self._is_cpu(input_tensor):
            return self._gloo._reduce_scatter_base(output_tensor, input_tensor, opts)
        self._fr_op("_reduce_scatter_base", [input_tensor], [output_tensor])
        index = self._same_device([input_tensor, output_tensor])
        self._ensure(index)
        staged_in, staged_out = self._reduce_scatter_flat(
            index, output_tensor, input_tensor, opts.reduceOp
        )
        self._record(index, input_tensor, staged_in)
        self._finish(index, (output_tensor, staged_out))
        return self._result(index, [output_tensor])

    @_loud
    def reduce_scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.reduce_scatter(output_tensors, input_tensors, opts)
        if input_tensors[0]:
            self._fr_op(
                "reduce_scatter",
                [flight_recorder.flat(input_tensors[0])],
                output_tensors,
            )
        self._one_per_rank(output_tensors)
        output = output_tensors[0]
        inputs = input_tensors[0]
        if len(inputs) != self.size():
            raise ValueError("reduce_scatter needs world_size input tensors")
        if any(i.numel() != output.numel() or i.dtype != output.dtype for i in inputs):
            raise ValueError(
                "reduce_scatter inputs must match the output's size and dtype"
            )
        index = self._same_device([output, *inputs])
        comm = self._ensure(index)
        comm.wait_stream(torch.accelerator.current_stream(index))
        with device_module.stream(comm):
            flat = torch.cat([t.reshape(-1) for t in inputs])
        _, staged_out = self._reduce_scatter_flat(index, output, flat, opts.reduceOp)
        self._record(index, flat, *inputs)
        self._finish(index, (output, staged_out))
        return self._result(index, [output])

    @_loud
    def reduce_scatter_tensor_coalesced(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: ReduceScatterOptions = ReduceScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.reduce_scatter_tensor_coalesced(
                output_tensors, input_tensors, opts
            )
        self._fr_op("reduce_scatter_tensor_coalesced", input_tensors, output_tensors)
        if len(output_tensors) != len(input_tensors):
            raise ValueError("coalesced reduce_scatter needs as many outputs as inputs")
        index = self._same_device([*input_tensors, *output_tensors])
        self._ensure(index)
        with self._group():
            pairs = [
                self._reduce_scatter_flat(index, o, i, opts.reduceOp)
                for o, i in zip(output_tensors, input_tensors)
            ]
        self._record(index, *input_tensors, *[si for si, _ in pairs])
        self._finish(index, *[(o, so) for o, (_, so) in zip(output_tensors, pairs)])
        return self._result(index, output_tensors)

    def _splits(self, tensor: torch.Tensor, sizes: list[int], what: str) -> list[int]:
        """Element counts per peer of an all-to-all buffer (rows of dim 0)."""
        world = self.size()
        rows = tensor.shape[0] if tensor.dim() > 0 else 1
        row = tensor.numel() // rows if rows else 0
        if not sizes:
            if rows % world:
                raise ValueError(
                    f"all_to_all {what} of {rows} rows does not split evenly across {world} ranks"
                )
            return [(rows // world) * row] * world
        if len(sizes) != world or any(s < 0 for s in sizes) or sum(sizes) != rows:
            raise ValueError(
                f"all_to_all {what} split sizes must be {world} nonnegative sizes summing to {rows} rows"
            )
        return [s * row for s in sizes]

    def _sendrecv_chunks(
        self,
        index: int,
        output: torch.Tensor,
        input: torch.Tensor,
        out_sizes: list[int],
        in_sizes: list[int],
    ):
        if output.dtype != input.dtype:
            raise ValueError("all_to_all output dtype must match the input")
        dtype = _nccl_dtype(input.dtype)
        itemsize = input.element_size()
        with self._group():
            in_off = out_off = 0
            for peer in range(self.size()):
                n_in, n_out = in_sizes[peer], out_sizes[peer]
                if n_in:
                    self._core.check(
                        self._core.send(
                            self._handle,
                            index,
                            input.data_ptr() + in_off * itemsize,
                            n_in,
                            dtype,
                            peer,
                            self._fr_tag(),
                        ),
                        "ncclSend",
                    )
                if n_out:
                    self._core.check(
                        self._core.recv(
                            self._handle,
                            index,
                            output.data_ptr() + out_off * itemsize,
                            n_out,
                            dtype,
                            peer,
                            self._fr_tag(),
                        ),
                        "ncclRecv",
                    )
                in_off += n_in
                out_off += n_out

    @_loud
    def alltoall_base(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        output_split_sizes: list[int],
        input_split_sizes: list[int],
        opts: AllToAllOptions = AllToAllOptions(),
    ) -> Work:
        if self._is_cpu(input_tensor):
            return self._gloo.alltoall_base(
                output_tensor, input_tensor, output_split_sizes, input_split_sizes, opts
            )
        self._fr_op("all_to_all", [input_tensor], [output_tensor])
        index = self._same_device([input_tensor, output_tensor])
        self._ensure(index)
        in_sizes = self._splits(input_tensor, list(input_split_sizes), "input")
        out_sizes = self._splits(output_tensor, list(output_split_sizes), "output")
        staged_in = self._stage_in(index, input_tensor)
        staged_out = self._stage_out(index, output_tensor)
        self._sendrecv_chunks(index, staged_out, staged_in, out_sizes, in_sizes)
        self._record(index, input_tensor, staged_in)
        self._finish(index, (output_tensor, staged_out))
        return self._result(index, [output_tensor])

    @_loud
    def alltoall(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[torch.Tensor],
        opts: AllToAllOptions = AllToAllOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.alltoall(output_tensors, input_tensors, opts)
        self._fr_op("all_to_all", input_tensors, output_tensors)
        if len(output_tensors) != self.size() or len(input_tensors) != self.size():
            raise ValueError("all_to_all needs world_size input and output tensors")
        index = self._same_device([*input_tensors, *output_tensors])
        self._ensure(index)
        staged_in = [self._stage_in(index, t) for t in input_tensors]
        staged_out = [self._stage_out(index, t) for t in output_tensors]
        with self._group():
            for peer, (i, o) in enumerate(zip(staged_in, staged_out)):
                if i.numel():
                    self._core.check(
                        self._core.send(
                            self._handle,
                            index,
                            i.data_ptr(),
                            i.numel(),
                            _nccl_dtype(i.dtype),
                            peer,
                            self._fr_tag(),
                        ),
                        "ncclSend",
                    )
                if o.numel():
                    self._core.check(
                        self._core.recv(
                            self._handle,
                            index,
                            o.data_ptr(),
                            o.numel(),
                            _nccl_dtype(o.dtype),
                            peer,
                            self._fr_tag(),
                        ),
                        "ncclRecv",
                    )
        self._record(index, *input_tensors, *staged_in)
        self._finish(index, *zip(output_tensors, staged_out))
        return self._result(index, output_tensors)

    @_loud
    def gather(
        self,
        output_tensors: list[list[torch.Tensor]],
        input_tensors: list[torch.Tensor],
        opts: GatherOptions = GatherOptions(),
    ) -> Work:
        if self._is_cpu(input_tensors[0]):
            return self._gloo.gather(output_tensors, input_tensors, opts)
        self._fr_op(
            "gather", input_tensors, output_tensors[0] if output_tensors else []
        )
        self._one_per_rank(input_tensors)
        input = input_tensors[0]
        root = self.rank() == opts.rootRank
        outputs = output_tensors[0] if root else []
        if root and (
            len(outputs) != self.size()
            or any(
                o.numel() != input.numel() or o.dtype != input.dtype for o in outputs
            )
        ):
            raise ValueError("gather needs world_size outputs matching the input")
        index = self._same_device([input, *outputs])
        comm = self._ensure(index)
        dtype = _nccl_dtype(input.dtype)
        staged_in = self._stage_in(index, input)
        staged_out = [self._stage_out(index, t) for t in outputs]
        comm.wait_stream(torch.accelerator.current_stream(index))
        if nccl.uses_mojoccl():
            self._gather_by_all_gather(index, staged_in, staged_out)
            self._finish(index, *zip(outputs, staged_out))
            self._record(index, input, staged_in)
            return self._result(index, outputs if root else input_tensors)
        with self._group():
            if root:
                for peer, o in enumerate(staged_out):
                    if peer != self.rank():
                        self._core.check(
                            self._core.recv(
                                self._handle,
                                index,
                                o.data_ptr(),
                                o.numel(),
                                dtype,
                                peer,
                                self._fr_tag(),
                            ),
                            "ncclRecv",
                        )
            else:
                self._core.check(
                    self._core.send(
                        self._handle,
                        index,
                        staged_in.data_ptr(),
                        staged_in.numel(),
                        dtype,
                        opts.rootRank,
                        self._fr_tag(),
                    ),
                    "ncclSend",
                )
        if root:
            with device_module.stream(comm):
                staged_out[self.rank()].copy_(staged_in)
            self._finish(index, *zip(outputs, staged_out))
        self._record(index, input, staged_in)
        return self._result(index, outputs if root else input_tensors)

    # MojoCCL has no point-to-point operations: gather and scatter (what
    # DTensor's distribute_tensor uses to shard a module) go through the
    # collectives it has, moving world_size times the data of the root's
    # sends -- fine for setup traffic, which is what they carry.

    def _gather_by_all_gather(
        self, index: int, input: torch.Tensor, outputs: list[torch.Tensor]
    ):
        with device_module.stream(self._ready[index]):
            gathered = torch.empty(
                self.size() * input.numel(), dtype=input.dtype, device=input.device
            )
        self._core.check(
            self._core.allgather(
                self._handle,
                index,
                input.data_ptr(),
                gathered.data_ptr(),
                input.numel(),
                _nccl_dtype(input.dtype),
                self._fr_tag(),
            ),
            "ncclAllGather",
        )
        with device_module.stream(self._ready[index]):
            for o, chunk in zip(outputs, gathered.view(self.size(), -1)):
                o.copy_(chunk.view_as(o))

    def _scatter_by_broadcast(
        self,
        index: int,
        output: torch.Tensor,
        inputs: list[torch.Tensor],
        root_rank: int,
    ):
        with device_module.stream(self._ready[index]):
            if inputs:
                full = torch.cat([i.reshape(-1) for i in inputs])
            else:
                full = torch.empty(
                    self.size() * output.numel(),
                    dtype=output.dtype,
                    device=output.device,
                )
        self._core.check(
            self._core.broadcast(
                self._handle,
                index,
                full.data_ptr(),
                full.numel(),
                _nccl_dtype(output.dtype),
                root_rank,
                self._fr_tag(),
            ),
            "ncclBroadcast",
        )
        with device_module.stream(self._ready[index]):
            output.copy_(full.view(self.size(), -1)[self.rank()].view_as(output))

    @_loud
    def scatter(
        self,
        output_tensors: list[torch.Tensor],
        input_tensors: list[list[torch.Tensor]],
        opts: ScatterOptions = ScatterOptions(),
    ) -> Work:
        if self._is_cpu(output_tensors[0]):
            return self._gloo.scatter(output_tensors, input_tensors, opts)
        self._fr_op(
            "scatter", input_tensors[0] if input_tensors else [], output_tensors
        )
        self._one_per_rank(output_tensors)
        output = output_tensors[0]
        root = self.rank() == opts.rootRank
        inputs = input_tensors[0] if root else []
        if root and (
            len(inputs) != self.size()
            or any(
                i.numel() != output.numel() or i.dtype != output.dtype for i in inputs
            )
        ):
            raise ValueError("scatter needs world_size inputs matching the output")
        index = self._same_device([output, *inputs])
        comm = self._ensure(index)
        dtype = _nccl_dtype(output.dtype)
        staged_out = self._stage_out(index, output)
        staged_in = [self._stage_in(index, t) for t in inputs]
        comm.wait_stream(torch.accelerator.current_stream(index))
        if nccl.uses_mojoccl():
            self._scatter_by_broadcast(index, staged_out, staged_in, opts.rootRank)
            self._record(index, *inputs, *staged_in)
            self._finish(index, (output, staged_out))
            return self._result(index, output_tensors)
        with self._group():
            if root:
                for peer, i in enumerate(staged_in):
                    if peer != self.rank():
                        self._core.check(
                            self._core.send(
                                self._handle,
                                index,
                                i.data_ptr(),
                                i.numel(),
                                dtype,
                                peer,
                                self._fr_tag(),
                            ),
                            "ncclSend",
                        )
            else:
                self._core.check(
                    self._core.recv(
                        self._handle,
                        index,
                        staged_out.data_ptr(),
                        staged_out.numel(),
                        dtype,
                        opts.rootRank,
                        self._fr_tag(),
                    ),
                    "ncclRecv",
                )
        if root:
            with device_module.stream(comm):
                staged_out.copy_(staged_in[self.rank()])
            self._record(index, *inputs, *staged_in)
        self._finish(index, (output, staged_out))
        return self._result(index, output_tensors)

    @_loud
    def send(self, tensors: list[torch.Tensor], dst_rank: int, tag: int) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.send(tensors, dst_rank, tag)
        self._fr_op(f"send {self.rank()}->{dst_rank}", tensors, tensors, p2p=True)
        index = self._same_device(tensors)
        self._ensure(index)
        staged = [self._stage_in(index, t) for t in tensors]
        with self._group():
            for s in staged:
                self._core.check(
                    self._core.send(
                        self._handle,
                        index,
                        s.data_ptr(),
                        s.numel(),
                        _nccl_dtype(s.dtype),
                        dst_rank,
                        self._fr_tag(),
                    ),
                    "ncclSend",
                )
        self._record(index, *tensors, *staged)
        return self._result(index, tensors)

    @_loud
    def recv(self, tensors: list[torch.Tensor], src_rank: int, tag: int) -> Work:
        if self._is_cpu(tensors[0]):
            return self._gloo.recv(tensors, src_rank, tag)
        self._fr_op(f"recv {self.rank()}<-{src_rank}", tensors, tensors, p2p=True)
        index = self._same_device(tensors)
        self._ensure(index)
        staged = [self._stage_out(index, t) for t in tensors]
        with self._group():
            for s in staged:
                self._core.check(
                    self._core.recv(
                        self._handle,
                        index,
                        s.data_ptr(),
                        s.numel(),
                        _nccl_dtype(s.dtype),
                        src_rank,
                        self._fr_tag(),
                    ),
                    "ncclRecv",
                )
        self._finish(index, *zip(tensors, staged))
        return self._result(index, tensors)

    # torch 2.13 renamed the tensor-shaped collectives on ProcessGroup; the
    # old names stay for 2.12 and older. Both spellings resolve to one body.
    all_gather_single = _allgather_base
    all_gather_single_coalesced = allgather_into_tensor_coalesced
    reduce_scatter_single = _reduce_scatter_base
    reduce_scatter_single_coalesced = reduce_scatter_tensor_coalesced
    all_to_all_single = alltoall_base

    @_loud
    def barrier(self, opts: BarrierOptions = BarrierOptions()) -> Work:
        """Every rank finishes its device work (all streams, comm included),
        then a host rendezvous over gloo."""
        for index in self._ready:
            torch.accelerator.synchronize(index)
            self._core.check(
                self._core.synchronize_comm(self._handle, index),
                "comm stream synchronize",
            )
        return self._gloo.barrier(opts)


def create_mojo_process_group(
    store: Store, rank: int, world_size: int, timeout: datetime.timedelta
) -> MojoProcessGroup:
    """The factory `torch.distributed.Backend.register_backend` stores."""
    return MojoProcessGroup(store, rank, world_size, timeout)


_HSA_LEVEL = "ROCR_VISIBLE_DEVICES"
_RUNTIME_LEVEL = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")


def _visible_entries(var: str) -> list[str] | None:
    """The comma list in `var`, or None when it is unset or empty."""
    visible = os.environ.get(var)
    if visible is None:
        return None
    entries = [entry.strip() for entry in visible.split(",") if entry.strip()]
    return entries or None


def _keep_local_rank_entry(var: str, entries: list[str], rank_index: int):
    if len(entries) <= 1:
        return  # already pinned, e.g. one srun task per GPU
    if rank_index >= len(entries):
        raise RuntimeError(
            f"LOCAL_RANK={rank_index} but {var}={os.environ[var]!r} lists only "
            f"{len(entries)} devices"
        )
    os.environ[var] = entries[rank_index]


def use_local_rank_gpu():
    """Pin this torchrun worker to its GPU via the vendor's visibility variable.

    Call as early as possible — before any mojo tensor is created and before
    MAX enumerates devices (enumeration is cached per process). With exactly
    one visible GPU per rank, ``mojo``/``mojo:0`` is always the right device,
    the phantom ``privateuseone:0`` TensorImpl index is always truthful, and
    each process binds a single CUDA context / HIP device.

    SLURM (and other launchers) often pre-set the visibility variable to the
    whole allocation — ``CUDA_VISIBLE_DEVICES=0,...,7`` on NVIDIA,
    ``ROCR_VISIBLE_DEVICES=0,...,3`` on AMD; in that case each rank keeps
    only its LOCAL_RANK-th entry. A single already-pinned entry is left
    alone (one srun task per GPU pins that way). When no variable is set at
    all, both the CUDA and HIP ones are set to LOCAL_RANK, whichever runtime
    turns out to be present.

    Only one level is ever narrowed. When ``ROCR_VISIBLE_DEVICES`` is
    present it is the level that gets the rank's entry, and any runtime-level
    list (SLURM's gres plugin exports ``CUDA_VISIBLE_DEVICES`` next to it by
    default) is rewritten to ``"0"``, the only index that exists in a
    one-entry HSA set — narrowing both levels by rank would compose to no
    visible GPU at all for every rank but 0.
    """
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    try:
        rank_index = int(local_rank)
    except ValueError:
        raise RuntimeError(f"LOCAL_RANK={local_rank!r} is not an integer") from None
    hsa = _visible_entries(_HSA_LEVEL)
    if hsa is not None:
        _keep_local_rank_entry(_HSA_LEVEL, hsa, rank_index)
        for var in _RUNTIME_LEVEL:
            if _visible_entries(var) is not None:
                os.environ[var] = "0"
        return
    pinned = False
    for var in _RUNTIME_LEVEL:
        entries = _visible_entries(var)
        if entries is None:
            continue
        _keep_local_rank_entry(var, entries, rank_index)
        pinned = True
    if not pinned:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank_index)
        os.environ["HIP_VISIBLE_DEVICES"] = str(rank_index)
