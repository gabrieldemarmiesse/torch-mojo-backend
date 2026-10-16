"""The flight recorder's dump format and coordination, without a GPU: a
Recorder over a fake native snapshot, held against torch's own offline
analyzer (torch.distributed.flight_recorder). The two-rank GPU runs are in
test_distributed.py (`test_two_rank_flight_recorder*`)."""

import ctypes
import json
import os
import pickle
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from torch._C import _distributed_c10d as c10d_c

from torch_mojo_backend import monkeypatching
from torch_mojo_backend.distributed import flight_recorder as fr
from torch_mojo_backend.distributed.store_worker import StoreWorker

ALLREDUCE, SEND, RECV, ALLGATHER, SENDRECV = 1, 6, 7, 4, 8
FLOAT = 7  # ncclFloat32


def _word(rid: int, op: int, cseq: int, pseq: int = 0, **kw: int) -> list[int]:
    w = [0] * fr.F_WORDS
    w[fr.F_ID] = rid + 1
    w[fr.F_PG] = kw.get("pg", 0)
    w[fr.F_CSEQ], w[fr.F_PSEQ], w[fr.F_OPID] = cseq, pseq, rid + 1
    w[fr.F_OP], w[fr.F_COUNT], w[fr.F_DTYPE] = op, kw.get("count", 1024), FLOAT
    w[fr.F_ROOT] = kw.get("root", -1)
    w[fr.F_FLAGS] = kw.get("flags", 0)
    w[fr.F_CREATED] = 1_000 + rid
    w[fr.F_TAG] = kw.get("tag", 0)
    w[fr.F_COMM] = kw.get("comm", 1)
    if kw.get("done", True):
        w[fr.F_DONE] = 2_000 + rid
    return w


class _Fake:
    """A Recorder over `rows` instead of the Mojo ring."""

    def __init__(self, rows: list[list[int]], world: int = 2, rank: int = 0):
        self.rows = rows
        self.rec = fr.Recorder(8, self.snapshot, "nccl")
        self.last = 0
        self.rec._groups[0] = fr._Group(
            "0", "default_pg", list(range(world)), rank, 600_000
        )

    def snapshot(self, buf: int, cap: int) -> int:
        out = (ctypes.c_int64 * (cap * fr.F_WORDS)).from_address(buf)
        for i, row in enumerate(self.rows[:cap]):
            out[i * fr.F_WORDS : (i + 1) * fr.F_WORDS] = row
        return min(len(self.rows), cap)

    def begin(self, *args: Any) -> int:
        self.last = self.rec.begin(*args)
        return self.last

    def tag(self) -> int:
        """The record id the last `begin` tagged."""
        return self.last & fr.TAG_MASK


def _analyze(tmp_path: Path, dumps: dict[int, dict[str, Any]]) -> str:
    for rank, dump in dumps.items():
        (tmp_path / f"trace_{rank}").write_bytes(pickle.dumps(dump, protocol=2))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.flight_recorder.fr_trace",
            str(tmp_path),
            "-p",
            "trace_",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    return output


def test_buffer_size_aliases(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TORCH_FR_BUFFER_SIZE", raising=False)
    monkeypatch.delenv("TORCH_NCCL_TRACE_BUFFER_SIZE", raising=False)
    assert fr.buffer_size() == 2000
    monkeypatch.setenv("TORCH_NCCL_TRACE_BUFFER_SIZE", "7")
    assert fr.buffer_size() == 7
    monkeypatch.setenv("TORCH_FR_BUFFER_SIZE", "0")  # the first name wins
    assert fr.buffer_size() == 0


def test_dump_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TORCH_FR_DUMP_TEMP_FILE", raising=False)
    monkeypatch.delenv("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", "/x")
    assert fr.dump_prefix() == "/x/torch/comm_lib_trace_rank_"
    monkeypatch.setenv("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", "/y/t_")
    assert fr.dump_prefix() == "/y/t_"


def test_eager_record_and_native_entry():
    """An entry with a tag takes the eager call's name, shapes and stack; one
    without (a compiled graph's) the native op, its count and dtype."""
    fake = _Fake([])
    x = torch.ones(2, 3)
    fake.begin("all_reduce", [x], [x], 0)
    tag = fake.tag()
    fake.rows = [_word(0, ALLREDUCE, 1, tag=tag), _word(1, ALLGATHER, 2, count=5)]
    e0, e1 = fake.rec.dump()["entries"]
    assert e0["profiling_name"] == "nccl:all_reduce"
    assert e0["input_sizes"] == [[2, 3]] and e0["input_dtypes"] == ["Float"]
    assert [f["name"] for f in e0["frames"][:2]] == [
        "begin",  # _Fake's
        "test_eager_record_and_native_entry",
    ]
    assert e0["time_discovered_completed_ns"] is not None
    assert e1["profiling_name"] == "nccl:_all_gather_base"
    assert e1["input_sizes"] == [[5]] and e1["output_sizes"] == [[10]]
    assert e1["frames"] == [] and e1["state"] == "completed"


def test_tags_carry_the_record_and_the_structure():
    """`begin` hands back the tag word the pg.mojo entries take: a fresh
    record id and the caller's structure flags, whatever thread asks."""
    fake = _Fake([])
    x = torch.ones(1)
    first = fake.begin("send 0->1", [x], [x], fr.TAG_MEMBER | fr.TAG_P2P)
    tags: list[int] = []
    thread = threading.Thread(target=lambda: tags.append(fake.begin("b", [x], [x], 0)))
    thread.start()
    thread.join()
    assert first & ~fr.TAG_MASK == fr.TAG_MEMBER | fr.TAG_P2P
    assert tags[0] & fr.TAG_MASK == (first & fr.TAG_MASK) + 1


def test_coalesced_members_stay_scheduled():
    """A coalescing block's operations are never stamped, as
    ProcessGroupNCCL's; its end entry is."""
    p2p = fr.FL_P2P
    fake = _Fake(
        [
            _word(0, SEND, 0, 1, flags=p2p | fr.FL_MEMBER, root=1, done=False),
            _word(1, RECV, 0, 1, flags=p2p | fr.FL_MEMBER, root=1, done=False),
            _word(2, SENDRECV, 0, 1, flags=p2p | fr.FL_END),
        ]
    )
    entries = fake.rec.dump()["entries"]
    assert [e["state"] for e in entries] == ["scheduled", "scheduled", "completed"]
    assert entries[2]["profiling_name"] == "nccl:coalesced"
    active = fake.rec.dump(only_active=True)["entries"]
    assert [e["record_id"] for e in active] == [0, 1]


def test_untracked_entries_complete_with_a_later_one():
    """An entry the watchdog dropped has no stamp of its own: a later
    collective on its communicator completing means it completed (one
    stream); one on another communicator of the group (another device)
    proves nothing."""
    untracked = fr.FL_UNTRACKED
    rows = [
        _word(0, ALLREDUCE, 1, flags=untracked, done=False),
        _word(1, ALLREDUCE, 2, flags=untracked, done=False, comm=2),
        _word(2, ALLREDUCE, 3),
        _word(3, ALLREDUCE, 4, flags=untracked, done=False),
    ]
    states = [e["state"] for e in _Fake(rows).rec.dump()["entries"]]
    assert states == ["completed", "scheduled", "completed", "scheduled"]


def test_untagged_point_to_point_is_named_with_ranks(tmp_path: Path):
    """A send or recv whose record was lost is named from the ring (peer)
    and the group (own rank), so the analyzer can parse it. (Alone: after a
    collective of the same seq number fr_trace would fold it into a
    coalesced group, on NCCL's dumps too.)"""
    p2p = fr.FL_P2P
    dumps = {
        0: _Fake([_word(0, SEND, 0, 1, flags=p2p, root=1)], rank=0).rec.dump(),
        1: _Fake([_word(0, RECV, 0, 1, flags=p2p, root=0)], rank=1).rec.dump(),
    }
    assert dumps[0]["entries"][0]["profiling_name"] == "nccl:send 0->1"
    assert dumps[1]["entries"][0]["profiling_name"] == "nccl:recv 1<-0"
    output = _analyze(tmp_path, dumps)
    assert "MISMATCH" not in output and "Not all ranks" not in output, output


def test_entries_of_an_unknown_group_are_left_out():
    fake = _Fake([_word(0, ALLREDUCE, 1), _word(1, ALLREDUCE, 1, pg=7)])
    dump = fake.rec.dump()
    assert [e["pg_id"] for e in dump["entries"]] == [0]
    assert list(dump["pg_config"]) == ["0"]


def test_json_dump():
    fake = _Fake([_word(0, ALLREDUCE, 1, done=False)])
    out = json.loads(fr.as_json(fake.rec.dump(include_stack_traces=False)))
    assert out["version"] == "2.10" and out["nccl_comm_state"] == {}
    (e,) = out["entries"]
    assert "frames" not in e and e["time_discovered_completed_ns"] == 0
    assert e["process_group"] == ["0", "default_pg"]
    assert out["pg_status"]["0"]["last_enqueued_collective"] == "1"
    assert "entries" not in json.loads(fr.as_json(fr.empty_dump()))


def test_torch_analyzer_names_the_missing_rank(tmp_path: Path):
    """Two ranks' dumps where rank 1 never issued all_reduce #2: torch's
    analyzer reads our pickles and reports exactly that."""
    dumps = {
        0: _Fake([_word(0, ALLREDUCE, 1), _word(1, ALLREDUCE, 2, done=False)]),
        1: _Fake([_word(0, ALLREDUCE, 1)], rank=1),
    }
    output = _analyze(tmp_path, {r: f.rec.dump() for r, f in dumps.items()})
    assert "Not all ranks joining collective, sequence number: 2" in output, output
    assert "missing ranks: {1}" in output, output


def _block(
    rank: int, ops: list[tuple[str, int, list[torch.Tensor]]], done: bool
) -> dict[str, Any]:
    """A coalescing block as the ring holds it: member entries, then the
    end entry (`done`: stamped)."""
    fake = _Fake([], rank=rank)
    p2p = any(op in (SEND, RECV) for _, op, _ in ops)
    cseq, pseq = (0, 1) if p2p else (1, 0)
    kind = fr.FL_P2P if p2p else 0
    rows = []
    for i, (name, op, tensors) in enumerate(ops):
        fake.begin(name, tensors, tensors, 0)
        rows.append(
            _word(
                i, op, cseq, pseq, flags=fr.FL_MEMBER | kind, tag=fake.tag(), done=False
            )
        )
    fake.begin("coalesced", [], [], 0)
    end = _word(
        len(ops),
        SENDRECV,
        cseq,
        pseq,
        flags=fr.FL_END | kind,
        tag=fake.tag(),
        done=done,
    )
    fake.rows = [*rows, end]
    return fake.rec.dump()


def test_torch_analyzer_pairs_the_sends_with_the_recvs(tmp_path: Path):
    x = torch.ones(4)
    dumps = {
        0: _block(0, [("send 0->1", SEND, [x])], True),
        1: _block(1, [("recv 1<-0", RECV, [x])], True),
    }
    output = _analyze(tmp_path, dumps)
    assert "MISMATCH" not in output and "Not all ranks" not in output, output


def test_torch_analyzer_on_a_block_retired_on_one_rank_only(tmp_path: Path):
    """A collective coalescing block completed on rank 0, still in flight on
    rank 1: its operations are scheduled on both (as ProcessGroupNCCL's), so
    the analyzer finds no state mismatch in them. (It does call the block
    "not joined" by rank 1, completed or not, as on NCCL's dumps: its
    non-p2p coalesced matching does not pair members across ranks.)"""
    x = torch.ones(4)
    ops = [("all_reduce", ALLREDUCE, [x]), ("all_reduce", ALLREDUCE, [x])]
    dumps = {0: _block(0, ops, True), 1: _block(1, ops, False)}
    output = _analyze(tmp_path, dumps)
    assert "has errors" not in output, output


def test_list_collectives_record_the_stacked_buffer(tmp_path: Path):
    """all_gather / reduce_scatter of lists record what ProcessGroupNCCL
    stacks them into, [world, *shape], so the analyzer's numel checks hold."""
    x = torch.ones(4, 2)
    dumps = {}
    for rank in (0, 1):
        fake = _Fake([], rank=rank)
        outs = [torch.empty_like(x), torch.empty_like(x)]
        fake.begin("all_gather", [x], [fr.flat(outs)], 0)
        gather = _word(0, ALLGATHER, 1, tag=fake.tag())
        fake.begin("reduce_scatter", [fr.flat(outs)], [x], 0)
        scatter = _word(1, 5, 2, tag=fake.tag())
        fake.rows = [gather, scatter]
        dumps[rank] = fake.rec.dump()
    e0, e1 = dumps[0]["entries"]
    assert e0["output_sizes"] == [[2, 4, 2]] and e1["input_sizes"] == [[2, 4, 2]]
    output = _analyze(tmp_path, dumps)
    assert "has errors" not in output and "MISMATCH" not in output, output


def _nccl_dump() -> dict[str, Any]:
    """What ProcessGroupNCCL's recorder holds for one all_reduce."""
    entry = _Fake([_word(0, ALLREDUCE, 1)]).rec.dump()["entries"][0]
    return {
        "version": "2.10",
        "pg_config": {"5": {"name": "5", "desc": "nccl_pg", "ranks": "[0, 1]"}},
        "comm_lib_version": "2.27.3",
        "pg_status": {"0": {"last_enqueued_collective": 1}},
        "entries": [{**entry, "process_group": ("5", "nccl_pg")}],
    }


def test_dump_follows_the_mojo_groups(monkeypatch: pytest.MonkeyPatch):
    """`_dump_nccl_trace` is torch's own while no mojo group is alive -- none
    yet, or all destroyed -- and whenever ProcessGroupNCCL recorded
    anything (returned untouched: one dump cannot hold two numbered
    streams); this backend's otherwise."""
    nccl = pickle.dumps(_nccl_dump())
    empty = pickle.dumps(fr.empty_dump())
    answer = [empty]
    monkeypatch.setattr(c10d_c, "_dump_nccl_trace", lambda *a: answer[0], raising=False)
    monkeypatch.setattr(
        c10d_c,
        "_dump_nccl_trace_json",
        lambda *a: fr.as_json(pickle.loads(answer[0])),
        raising=False,
    )
    monkeypatch.setattr(monkeypatching, "_NCCL_TRACE_ORIGINALS", [])
    monkeypatch.setattr(fr, "_RECORDER", [])
    monkeypatching.serve_nccl_trace_dumps_from_the_mojo_backend()
    trace = getattr(c10d_c, "_dump_nccl_trace")  # noqa: B009 -- CUDA builds only in the stub
    trace_json = getattr(c10d_c, "_dump_nccl_trace_json")  # noqa: B009
    assert trace() == empty
    fake = _Fake([_word(0, ALLREDUCE, 1)])
    fake.rec._live[0] = lambda: None  # ty: ignore[invalid-assignment] -- a group known to be alive
    fr._RECORDER.append(fake.rec)
    assert pickle.loads(trace())["pg_config"] == {
        "0": {"name": "0", "desc": "default_pg", "ranks": "[0, 1]"}
    }
    assert json.loads(trace_json())["entries"][0]["process_group"] == [
        "0",
        "default_pg",
    ]
    answer[0] = nccl
    assert trace() == nccl  # ProcessGroupNCCL recorded: its dump, untouched
    assert json.loads(trace_json())["pg_config"] == _nccl_dump()["pg_config"]
    answer[0] = empty
    fake.rec.retire(0)  # destroy_process_group
    assert not fr.active() and trace() == empty


def test_a_name_belongs_to_its_newest_group():
    """A group re-created under a destroyed one's name: the dump describes
    the name by the new group, and leaves the old one's entries out."""
    fake = _Fake([_word(0, ALLREDUCE, 1, pg=0), _word(1, ALLREDUCE, 1, pg=1)])
    fake.rec._groups[0] = fr._Group("0", "default_pg", [0, 1, 2], 0, 1)
    fake.rec._groups[1] = fr._Group("0", "default_pg", [0, 1], 0, 1)
    dump = fake.rec.dump()
    assert [e["pg_id"] for e in dump["entries"]] == [1]
    assert dump["pg_config"]["0"]["ranks"] == "[0, 1]"


def test_peer_ranks_are_those_of_the_live_groups():
    """The ranks a firing rank waits for: in a job whose default group is
    gloo, ranks without a mojo group never dump."""
    fake = _Fake([])
    fake.rec._groups[0] = fr._Group("3", "sub", [2, 3], 0, 1)
    fake.rec._groups[1] = fr._Group("4", "gone", [5], 0, 1)
    fake.rec._live[0] = lambda: None  # ty: ignore[invalid-assignment] -- alive
    assert fake.rec.peer_ranks() == {2, 3}


def test_concurrent_dumps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Several threads dumping at once (two groups' reporters, the monitor)
    each publish a whole file; no temporary file is left."""
    fake = _Fake([_word(0, ALLREDUCE, 1)])
    monkeypatch.setattr(fr, "_RECORDER", [fake.rec])
    monkeypatch.setenv("TORCH_FR_DUMP_TEMP_FILE", str(tmp_path / "trace_"))
    errors: list[BaseException] = []

    def dump():
        try:
            fr.write_dump(3, True)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=dump) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert os.listdir(tmp_path) == ["trace_3"]
    assert len(pickle.loads((tmp_path / "trace_3").read_bytes())["entries"]) == 1


def _poller(store: Any) -> fr._Poller:
    poller = fr._Poller()
    poller.attach(1, store, 0)
    return poller


def test_generations_are_done_pending_or_to_serve(monkeypatch: pytest.MonkeyPatch):
    """Overlapping dumps that both fail leave both generations to serve (a
    single high-water mark lost one); a failure backs retries off."""
    state = fr._StoreState(0, False)
    assert state.claim(1) == (1,)
    assert state.claim(2) == (2,)  # a concurrent claim: only the new one
    state.finish((1,), False)
    state.finish((2,), False)
    assert state.claim(2) == ()  # backing off
    state.dump_retry_at = 0.0
    assert state.claim(2) == (1, 2)
    state.finish((1, 2), True)
    assert state.base == 2 and not state.pending and state.claim(2) == ()
    assert state.claim(4) == (3, 4)
    state.finish((4,), True)
    assert state.base == 2 and state.done == {4}  # 3 still pending
    state.finish((3,), True)
    assert state.base == 4 and not state.done


def test_store_state_follows_the_watched_stores():
    """A second store attached while the first is watched becomes the one
    polled; each keeps its own generations, and a store whose groups are
    gone loses its state (a late answer for it is dropped)."""
    a, b = c10d_c.HashStore(), c10d_c.HashStore()
    fr.signal_dump(a, 3)  # an earlier job's signal on a: its baseline
    poller = _poller(a)
    poller.attach(2, b, 0)
    key_a, key_b = fr.store_key(a), fr.store_key(b)
    assert next(reversed(poller.groups.values()))[0] == key_b
    state_a = poller.state(key_a)
    assert state_a is not None and state_a.base == 1
    assert poller.state(key_b) is not state_a
    poller.detach(1)
    assert poller.state(key_a) is None
    assert fr.claim(a, 2) == (2,)  # no state: every generation is new


def test_acknowledgements_per_generation():
    store = c10d_c.HashStore()
    first = fr.signal_dump(store, 0)
    second = fr.signal_dump(store, 1)
    fr.ack_dump(store, 0, range(first, second + 1))
    assert fr.wait_for_dumps(store, [0, 1], second, 0.0) == [1]
    fr.ack_dump(store, 1, [second])
    assert fr.wait_for_dumps(store, [0, 1], second, 1e18) == []
    assert fr.wait_for_dumps(store, [0, 1], first, 0.0) == [1]
    assert fr.signaller(store) == 1
    assert fr.signaller(c10d_c.HashStore()) == -1  # no key: no blocking get


def test_poll_acknowledges_only_a_written_dump(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A dump that fails is not acknowledged and is retried after a
    backoff; once written it is."""
    store = c10d_c.HashStore()
    fake = _Fake([_word(0, ALLREDUCE, 1)])
    monkeypatch.setattr(fr, "_RECORDER", [fake.rec])
    poller = _poller(store)
    key = fr.store_key(store)
    worker = StoreWorker(store, clone=False, quiet=True)
    gen = fr.signal_dump(store, 1)
    blocked = tmp_path / "file"
    blocked.write_text("")  # a file where the dump directory should be
    monkeypatch.setenv("TORCH_FR_DUMP_TEMP_FILE", str(blocked / "trace_"))
    poller.poll(worker, key, 0)
    assert not store.check([f"{fr.ACK_KEY}/{gen}/0"])
    monkeypatch.setenv("TORCH_FR_DUMP_TEMP_FILE", str(tmp_path / "trace_"))
    poller.poll(worker, key, 0)
    assert not (tmp_path / "trace_0").exists()  # backing off
    state = poller.state(key)
    assert state is not None
    state.dump_retry_at = 0.0
    poller.poll(worker, key, 0)
    assert store.check([f"{fr.ACK_KEY}/{gen}/0"])
    assert (tmp_path / "trace_0").exists()
    worker.close()


def test_poll_answers_a_nccl_dump_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """ProcessGroupNCCL's broadcastDumpSignal sets torch's key and no
    generation: its first appearance dumps once, retried if that failed."""
    store = c10d_c.HashStore()
    fake = _Fake([_word(0, ALLREDUCE, 1)])
    monkeypatch.setattr(fr, "_RECORDER", [fake.rec])
    blocked = tmp_path / "file"
    blocked.write_text("")
    monkeypatch.setenv("TORCH_FR_DUMP_TEMP_FILE", str(blocked / "trace_"))
    poller = _poller(store)
    key = fr.store_key(store)
    worker = StoreWorker(store, clone=False, quiet=True)
    poller.poll(worker, key, 0)
    store.set(fr.DUMP_KEY, struct.pack("=i", 1))  # ty: ignore[invalid-argument-type] -- Store.set takes bytes; the stub says str
    poller.poll(worker, key, 0)  # fails
    monkeypatch.setenv("TORCH_FR_DUMP_TEMP_FILE", str(tmp_path / "trace_"))
    state = poller.state(key)
    assert state is not None and state.torch == "unseen"
    state.dump_retry_at = 0.0
    poller.poll(worker, key, 0)
    assert (tmp_path / "trace_0").exists() and state.torch == "seen"
    (tmp_path / "trace_0").unlink()
    poller.poll(worker, key, 0)
    assert not (tmp_path / "trace_0").exists()
    worker.close()


def test_store_worker_keeps_one_abandoned_job():
    """A store that never answers: one job is abandoned, and no new thread
    or client is started until it returns."""
    release = threading.Event()
    calls: list[int] = []

    def stuck(_: object) -> int:
        calls.append(1)
        release.wait()
        return 1

    worker = StoreWorker(c10d_c.HashStore(), clone=False, quiet=True, max_abandoned=1)
    assert worker.run(stuck, 0.1) is None
    assert worker.run(stuck, 0.1) is None and calls == [1]  # not started
    release.set()
    deadline = time.monotonic() + 5
    while worker.stuck and time.monotonic() < deadline:
        time.sleep(0.01)
    assert worker.run(lambda _: 2, 1.0) == 2
    worker.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_fork_child_gets_fresh_locks_and_no_monitor():
    """A lock another parent thread held at fork stays held in the child:
    the child's hook replaces them and drops the monitor."""
    held = threading.Event()
    release = threading.Event()

    def hold():
        with fr._WRITE_LOCK:
            held.set()
            release.wait()

    thread = threading.Thread(target=hold)
    thread.start()
    held.wait()
    fr._POLLER.append(fr._Poller())
    try:
        pid = os.fork()
        if pid == 0:
            ok = fr._WRITE_LOCK.acquire(timeout=2) and not fr._POLLER
            os._exit(0 if ok else 1)
        _, status = os.waitpid(pid, 0)
    finally:
        release.set()
        thread.join()
        fr._POLLER.clear()
    assert os.waitstatus_to_exitcode(status) == 0


def test_mixed_process_dumps_analyze_per_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Two ranks running a NCCL group and a mojo group, NCCL's ring wrapped
    (record ids 1500-1501) and a mojo all_reduce hung on rank 0: torch's
    dump comes back untouched and the analyzer reads it; this backend's
    dump names the hang. A merged stream would have lost the mojo entries
    to the analyzer's alignment on the first record id."""
    nccl_entry = {**_nccl_dump()["entries"][0], "record_id": 1500}
    second = {**nccl_entry, "record_id": 1501, "collective_seq_id": 2}
    nccl = {**_nccl_dump(), "entries": [nccl_entry, second]}
    monkeypatch.setattr(
        c10d_c, "_dump_nccl_trace", lambda *a: pickle.dumps(nccl), raising=False
    )
    monkeypatch.setattr(monkeypatching, "_NCCL_TRACE_ORIGINALS", [])
    monkeypatching.serve_nccl_trace_dumps_from_the_mojo_backend()
    trace = getattr(c10d_c, "_dump_nccl_trace")  # noqa: B009 -- CUDA builds only in the stub
    ours: dict[int, dict[str, Any]] = {}
    theirs: dict[int, dict[str, Any]] = {}
    for rank in (0, 1):
        rows = [_word(0, ALLREDUCE, 1)]
        if rank == 0:
            rows.append(_word(1, ALLREDUCE, 2, done=False))
        fake = _Fake(rows, rank=rank)
        fake.rec._live[0] = lambda: None  # ty: ignore[invalid-assignment] -- alive
        monkeypatch.setattr(fr, "_RECORDER", [fake.rec])
        theirs[rank] = pickle.loads(trace())
        ours[rank] = pickle.loads(fr.dump_nccl_trace())
    assert theirs[0] == nccl
    (tmp_path / "nccl").mkdir()
    (tmp_path / "mojo").mkdir()
    output = _analyze(tmp_path / "nccl", theirs)
    assert "has errors" not in output and "Not all ranks" not in output, output
    output = _analyze(tmp_path / "mojo", ours)
    assert "Not all ranks joining collective, sequence number: 2" in output, output
