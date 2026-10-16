"""The flight recorder's dump format and coordination, without a GPU: a
Recorder over a fake native snapshot, held against torch's own offline
analyzer (torch.distributed.flight_recorder). The two-rank GPU runs are in
test_distributed.py (`test_two_rank_flight_recorder*`)."""

import ctypes
import gc
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
        self.header = (ctypes.c_int64 * (fr.H_TAG + 2 * fr.TAG_SLOTS))()
        self.rows = rows
        self.rec = fr.Recorder(8, ctypes.addressof(self.header), self.snapshot, "nccl")
        self.rec._groups[0] = fr._Group(
            "0", "default_pg", list(range(world)), rank, 600_000
        )

    def snapshot(self, buf: int, cap: int) -> int:
        out = (ctypes.c_int64 * (cap * fr.F_WORDS)).from_address(buf)
        for i, row in enumerate(self.rows[:cap]):
            out[i * fr.F_WORDS : (i + 1) * fr.F_WORDS] = row
        return min(len(self.rows), cap)

    def tag(self) -> int:
        """The tag this thread's last `begin` wrote."""
        word = self.rec._cell()
        assert word is not None
        return word.value & fr.TAG_MASK


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
    fake.rec.begin("all_reduce", [x], [x], 0)
    tag = fake.tag()
    fake.rows = [_word(0, ALLREDUCE, 1, tag=tag), _word(1, ALLGATHER, 2, count=5)]
    fake.rec.clear()
    assert fake.tag() == 0
    e0, e1 = fake.rec.dump()["entries"]
    assert e0["profiling_name"] == "nccl:all_reduce"
    assert e0["input_sizes"] == [[2, 3]] and e0["input_dtypes"] == ["Float"]
    assert e0["frames"][0]["name"] == "test_eager_record_and_native_entry"
    assert e0["time_discovered_completed_ns"] is not None
    assert e1["profiling_name"] == "nccl:_all_gather_base"
    assert e1["input_sizes"] == [[5]] and e1["output_sizes"] == [[10]]
    assert e1["frames"] == [] and e1["state"] == "completed"


def test_tags_are_per_thread():
    """Every thread tags through a header slot of its own (owner: its
    ident), so another thread's call never touches this thread's tag."""
    fake = _Fake([])
    x = torch.ones(1)
    fake.rec.begin("all_reduce", [x], [x], 0)
    mine = fake.tag()
    owners: list[int] = []

    def run():
        fake.rec.begin("broadcast", [x], [x], fr.TAG_P2P)
        fake.rec.clear()
        owners.extend(fake.header[fr.H_TAG + 2 * i] for i in range(fr.TAG_SLOTS))

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert fake.tag() == mine
    assert threading.get_ident() in owners and thread.ident in owners


def test_tag_slots_run_out_and_come_back():
    """All TAG_SLOTS taken by live threads: one more thread records without
    a tag and touches no slot; the slots come back as their threads end."""
    fake = _Fake([])
    x = torch.ones(1)
    started = threading.Barrier(fr.TAG_SLOTS + 1)
    release = threading.Event()

    def hold():
        fake.rec.begin("all_reduce", [x], [x], 0)
        started.wait()
        release.wait()

    holders = [threading.Thread(target=hold) for _ in range(fr.TAG_SLOTS)]
    for thread in holders:
        thread.start()
    started.wait()
    before = list(fake.header)
    extra: list[object] = []

    def overflow():
        fake.rec.begin("broadcast", [x], [x], 0)
        extra.append(fake.rec._cell())

    thread = threading.Thread(target=overflow)
    thread.start()
    thread.join()
    assert extra == [None] and list(fake.header) == before
    release.set()
    for thread in holders:
        thread.join()
    deadline = time.monotonic() + 5  # a dead thread's locals go soon after
    while len(fake.rec._free_slots) < fr.TAG_SLOTS and time.monotonic() < deadline:
        gc.collect()
        time.sleep(0.01)
    assert sorted(fake.rec._free_slots) == list(range(fr.TAG_SLOTS))
    assert not any(fake.header[fr.H_TAG : fr.H_TAG + 2 * fr.TAG_SLOTS])


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
        fake.rec.begin(name, tensors, tensors, 0)
        rows.append(
            _word(
                i, op, cseq, pseq, flags=fr.FL_MEMBER | kind, tag=fake.tag(), done=False
            )
        )
    fake.rec.begin("coalesced", [], [], 0)
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
    the analyzer finds no state mismatch in them."""
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
        fake.rec.begin("all_gather", [x], [fr.flat(outs)], 0)
        gather = _word(0, ALLGATHER, 1, tag=fake.tag())
        fake.rec.begin("reduce_scatter", [fr.flat(outs)], [x], 0)
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
    yet, or all destroyed -- and both recorders' entries, merged, while one
    is: a ProcessGroupNCCL in the same process keeps its dumps."""
    nccl = _nccl_dump()
    monkeypatch.setattr(
        c10d_c, "_dump_nccl_trace", lambda *a: pickle.dumps(nccl), raising=False
    )
    monkeypatch.setattr(
        c10d_c,
        "_dump_nccl_trace_json",
        lambda *a: json.dumps(nccl).encode(),
        raising=False,
    )
    monkeypatch.setattr(monkeypatching, "_NCCL_TRACE_ORIGINALS", [])
    monkeypatch.setattr(fr, "_RECORDER", [])
    monkeypatching.serve_nccl_trace_dumps_from_the_mojo_backend()
    trace = getattr(c10d_c, "_dump_nccl_trace")  # noqa: B009 -- CUDA builds only in the stub
    trace_json = getattr(c10d_c, "_dump_nccl_trace_json")  # noqa: B009
    assert pickle.loads(trace()) == nccl
    fake = _Fake([_word(0, ALLREDUCE, 1)])
    fake.rec._live[0] = lambda: None  # ty: ignore[invalid-assignment] -- a group known to be alive
    fr._RECORDER.append(fake.rec)
    merged = pickle.loads(trace())
    assert [e["process_group"][0] for e in merged["entries"]] == ["5", "0"]
    assert [e["pg_id"] for e in merged["entries"]] == [0, 1]  # ours shifted
    assert sorted(merged["pg_config"]) == ["0", "5"]
    assert sorted(merged["pg_status"]) == ["0", "1"]
    as_json = json.loads(trace_json())
    assert [e["process_group"][0] for e in as_json["entries"]] == ["5", "0"]
    fake.rec.retire(0)  # destroy_process_group
    assert not fr.active() and pickle.loads(trace()) == nccl


def test_merge_without_nccl_entries_is_ours():
    ours = _Fake([_word(0, ALLREDUCE, 1)]).rec.dump()
    assert fr.merge(fr.empty_dump(), ours) == ours


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


def test_generations_belong_to_an_epoch():
    """Every signal is a new generation; a process claims each once, and a
    claim of an earlier epoch (an RPC on a store since replaced) gets
    nothing."""
    old = c10d_c.HashStore()
    fr.signal_dump(old, 3)  # an earlier job's signal: the baseline
    poller = _poller(old)
    first = fr.signal_dump(old, 0)
    epoch = poller.epoch
    assert poller.claim(first, epoch) == range(first, first + 1)
    assert not poller.claim(first, epoch)
    poller.detach(1)
    new = c10d_c.HashStore()
    poller.attach(2, new, 0)
    assert not poller.claim(first + 5, epoch)  # the old store's late answer
    assert poller.claim(fr.signal_dump(new, 1), poller.epoch) == range(1, 2)


def test_acknowledgements_per_generation():
    store = c10d_c.HashStore()
    first = fr.signal_dump(store, 0)
    second = fr.signal_dump(store, 1)
    fr.ack_dump(store, 0, range(first, second + 1))
    assert fr.wait_for_dumps(store, [0, 1], second, 0.0) == [1]
    fr.ack_dump(store, 1, range(second, second + 1))
    assert fr.wait_for_dumps(store, [0, 1], second, 1e18) == []
    assert fr.wait_for_dumps(store, [0, 1], first, 0.0) == [1]
    assert fr.signaller(store) == 1
    assert fr.signaller(c10d_c.HashStore()) == -1  # no key: no blocking get


def test_poll_acknowledges_only_a_written_dump(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A dump that fails is not acknowledged and is retried at the next
    poll; once written it is."""
    store = c10d_c.HashStore()
    fake = _Fake([_word(0, ALLREDUCE, 1)])
    monkeypatch.setattr(fr, "_RECORDER", [fake.rec])
    poller = _poller(store)
    worker = StoreWorker(store, clone=False, quiet=True)
    gen = fr.signal_dump(store, 1)
    blocked = tmp_path / "file"
    blocked.write_text("")  # a file where the dump directory should be
    monkeypatch.setenv("TORCH_FR_DUMP_TEMP_FILE", str(blocked / "trace_"))
    poller._poll(worker, 0, poller.epoch)
    assert not store.check([f"{fr.ACK_KEY}/{gen}/0"])
    monkeypatch.setenv("TORCH_FR_DUMP_TEMP_FILE", str(tmp_path / "trace_"))
    poller._poll(worker, 0, poller.epoch)
    assert store.check([f"{fr.ACK_KEY}/{gen}/0"])
    assert (tmp_path / "trace_0").exists()
    worker.close()


def test_poll_answers_a_nccl_dump_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """ProcessGroupNCCL's broadcastDumpSignal sets torch's key and no
    generation: its first appearance dumps once."""
    store = c10d_c.HashStore()
    fake = _Fake([_word(0, ALLREDUCE, 1)])
    monkeypatch.setattr(fr, "_RECORDER", [fake.rec])
    monkeypatch.setenv("TORCH_FR_DUMP_TEMP_FILE", str(tmp_path / "trace_"))
    poller = _poller(store)
    worker = StoreWorker(store, clone=False, quiet=True)
    poller._poll(worker, 0, poller.epoch)
    assert not (tmp_path / "trace_0").exists()
    store.set(fr.DUMP_KEY, struct.pack("=i", 1))  # ty: ignore[invalid-argument-type] -- Store.set takes bytes; the stub says str
    poller._poll(worker, 0, poller.epoch)
    assert (tmp_path / "trace_0").exists()
    (tmp_path / "trace_0").unlink()
    poller._poll(worker, 0, poller.epoch)
    assert not (tmp_path / "trace_0").exists()
    worker.close()
