"""The flight recorder's dump format and coordination, without a GPU: a
Recorder over a fake native snapshot, held against torch's own offline
analyzer (torch.distributed.flight_recorder). The two-rank GPU runs are in
test_distributed.py (`test_two_rank_flight_recorder*`)."""

import ctypes
import json
import os
import pickle
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
import torch
from torch._C import _distributed_c10d as c10d_c

from torch_mojo_backend import monkeypatching
from torch_mojo_backend.distributed import flight_recorder as fr

ALLREDUCE, SEND, RECV, ALLGATHER, SENDRECV = 1, 6, 7, 4, 8
FLOAT = 7  # ncclFloat32


def _stamp(rid1: int, us: int) -> int:
    return ((rid1 & fr.DONE_ID_MASK) << fr.DONE_TIME_BITS) | us


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
    if kw.get("done", True):
        w[fr.F_DONE] = _stamp(rid + 1, 2 + rid)
    return w


class _Fake:
    """A Recorder over `rows` instead of the Mojo ring."""

    def __init__(self, rows: list[list[int]], world: int = 2, rank: int = 0):
        self.header = (ctypes.c_int64 * (fr.H_TAG + 64))()
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
        return self.rec._cell()[0].value & fr.TAG_MASK


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
    assert fake.rec._cell()[0].value == 0
    e0, e1 = fake.rec.dump()["entries"]
    assert e0["profiling_name"] == "nccl:all_reduce"
    assert e0["input_sizes"] == [[2, 3]] and e0["input_dtypes"] == ["Float"]
    assert e0["frames"][0]["name"] == "test_eager_record_and_native_entry"
    assert e0["time_discovered_completed_ns"] is not None
    assert e1["profiling_name"] == "nccl:_all_gather_base"
    assert e1["input_sizes"] == [[5]] and e1["output_sizes"] == [[10]]
    assert e1["frames"] == [] and e1["state"] == "completed"


def test_stamp_of_another_id_is_pending():
    """A stamp naming the slot's previous id (a late stamp of a reused
    slot) leaves the entry pending."""
    row = _word(5, ALLREDUCE, 1, done=False)
    row[fr.F_DONE] = _stamp(5, 9)  # entry 4's stamp, the slot is entry 5's
    (e,) = _Fake([row]).rec.dump()["entries"]
    assert e["state"] == "scheduled" and e["time_discovered_completed_ns"] is None


def test_tags_are_per_thread():
    """Another thread's tag never takes this thread's slot unless their
    hashes share it; a shared slot carries the writer's thread bits, so the
    other thread reads no tag rather than a wrong one."""
    fake = _Fake([])
    x = torch.ones(1)
    fake.rec.begin("all_reduce", [x], [x], 0)
    mine = fake.rec._cell()
    other: list[tuple[ctypes.c_int64, int]] = []

    def run():
        fake.rec.begin("broadcast", [x], [x], 0)
        other.append(fake.rec._cell())

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    word, bits = mine
    if ctypes.addressof(word) == ctypes.addressof(other[0][0]):
        assert word.value & ~((1 << fr.TAG_THREAD_SHIFT) - 1) == other[0][1]
    else:
        assert word.value == 1 | bits  # still this thread's first record


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
    collective of its group completing means it completed (one stream)."""
    rows = [
        _word(0, ALLREDUCE, 1, flags=fr.FL_UNTRACKED, done=False),
        _word(1, ALLREDUCE, 2),
        _word(2, ALLREDUCE, 3, flags=fr.FL_UNTRACKED, done=False),
    ]
    states = [e["state"] for e in _Fake(rows).rec.dump()["entries"]]
    assert states == ["completed", "completed", "scheduled"]


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


def test_dump_falls_back_to_torch_without_a_mojo_group(monkeypatch: pytest.MonkeyPatch):
    """`_dump_nccl_trace` is torch's own until a mojo group records: a CUDA
    user's ProcessGroupNCCL keeps its dumps."""
    monkeypatch.setattr(
        c10d_c, "_dump_nccl_trace", lambda *a: b"torch's", raising=False
    )
    monkeypatch.setattr(
        c10d_c, "_dump_nccl_trace_json", lambda *a: b"torch's json", raising=False
    )
    monkeypatch.setattr(monkeypatching, "_NCCL_TRACE_ORIGINALS", [])
    monkeypatch.setattr(fr, "_RECORDER", [])
    monkeypatching.serve_nccl_trace_dumps_from_the_mojo_backend()
    trace = getattr(c10d_c, "_dump_nccl_trace")  # noqa: B009 -- CUDA builds only in the stub
    trace_json = getattr(c10d_c, "_dump_nccl_trace_json")  # noqa: B009
    assert trace() == b"torch's" and trace_json() == b"torch's json"
    fake = _Fake([_word(0, ALLREDUCE, 1)])
    fake.rec._live[0] = lambda: None  # ty: ignore[invalid-assignment] -- a group known to have lived
    fr._RECORDER.append(fake.rec)
    assert pickle.loads(trace())["entries"][0]["profiling_name"] == "nccl:all_reduce"
    assert json.loads(trace_json())["entries"][0]["record_id"] == 0


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


def test_signal_generations(monkeypatch: pytest.MonkeyPatch):
    """Every signal is a new generation, acknowledged per generation; a
    process serves each generation once."""
    store = c10d_c.HashStore()
    monkeypatch.setattr(fr, "_SERVED", [fr.current_generation(store)])
    first = fr.signal_dump(store, 0)
    assert fr.claim(first) == range(1, 2) and fr.claim(first) == range(2, 2)
    second = fr.signal_dump(store, 1)
    assert second == first + 1
    fr.ack_dump(store, 0, range(first, second + 1))
    assert fr.wait_for_dumps(store, 2, second, 0.0) == [1]
    fr.ack_dump(store, 1, range(second, second + 1))
    assert fr.wait_for_dumps(store, 2, second, 1e18) == []
    assert fr.wait_for_dumps(store, 2, first, 0.0) == [1]  # not acked for #1
