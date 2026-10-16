"""Store operations bounded by a deadline, for the threads that must not
hang on a c10d store: the collective watchdog's desync reporter and the
flight recorder's monitor."""

from __future__ import annotations

import contextlib
import queue
import sys
import threading
from collections.abc import Callable
from typing import TypeVar, cast

from torch._C._distributed_c10d import Store

_T = TypeVar("_T")


_Job = tuple[Callable[[Store], object], int | None, list[object], threading.Event]


class StoreWorker:
    """Store operations on a thread of their own, each bounded by a deadline
    the caller enforces: a store client cannot be bounded from outside (its
    `check` blocks in a socket read that ignores `set_timeout`, and `clone`
    connects with the original timeout), so a job that overruns is
    abandoned -- its thread exits once that operation returns -- and the
    next job starts a fresh thread and client. A job tagged with an index
    generation older than `latest` is dropped unrun: the index has moved
    on. `quiet`: no line on stderr when a job overruns or raises."""

    def __init__(
        self,
        store: Store,
        clone: bool = True,
        quiet: bool = False,
        name: str = "mojo-pg-desync-store",
    ):
        self._store = store
        self._clone = clone
        self._quiet = quiet
        self._name = name
        self._jobs: queue.SimpleQueue[_Job | None] | None = None
        self.latest = 0

    def _serve(self, jobs: queue.SimpleQueue[_Job | None]):
        client = None
        while (item := jobs.get()) is not None:
            job, gen, box, done = item
            try:
                if gen is not None and gen < self.latest:
                    box.append(None)
                else:
                    if client is None:
                        client = (
                            reporter_store(self._store) if self._clone else self._store
                        )
                    box.append(job(client))
            except Exception as error:  # reported by the caller
                box.append(error)
            done.set()

    def run(
        self, job: Callable[[Store], _T], timeout: float, gen: int | None = None
    ) -> _T | None:
        """`job(client)`'s result, or None when it raised, overran or was
        dropped as stale."""
        if self._jobs is None:
            self._jobs = queue.SimpleQueue()
            threading.Thread(
                target=self._serve, args=(self._jobs,), name=self._name, daemon=True
            ).start()
        box: list[object] = []
        done = threading.Event()
        self._jobs.put((job, gen, box, done))
        if not done.wait(max(timeout, 0.1)):
            self.close()  # stuck: that thread exits once its op returns
            if not self._quiet:
                sys.stderr.write(
                    "mojo watchdog: the c10d store did not answer in time\n"
                )
            return None
        if isinstance(box[0], Exception):
            if not self._quiet:
                sys.stderr.write(f"mojo watchdog: c10d store error: {box[0]!r}\n")
            return None
        return cast(_T, box[0])

    def close(self):
        if self._jobs is not None:
            self._jobs.put(None)
            self._jobs = None


def reporter_store(store: Store) -> Store:
    """A client of its own when the store can make one (`clone()`, torch >=
    2.6): a TCPStore client holds its lock for a whole blocking op, so
    sharing the main thread's would stall the report behind it."""
    clone = getattr(store, "clone", None)
    if clone is not None:
        with contextlib.suppress(Exception):
            return clone()
    return store
