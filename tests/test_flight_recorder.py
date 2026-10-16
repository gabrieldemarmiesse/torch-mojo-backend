"""The flight recorder's dump format, without a GPU: a Recorder over a fake
native snapshot, held against torch's own offline analyzer
(torch.distributed.flight_recorder). The two-rank GPU runs are in
test_distributed.py (`test_two_rank_flight_recorder*`)."""

import ctypes
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from torch_mojo_backend.distributed import flight_recorder as fr

ALLREDUCE, SEND, RECV, ALLGATHER = 1, 6, 7, 4
FLOAT = 7  # ncclFloat32


def _word(rid: int, op: int, cseq: int, pseq: int = 0, **kw: int) -> list[int]:
    w = [0] * fr.F_WORDS
    w[fr.F_ID] = rid + 1
    w[fr.F_CSEQ], w[fr.F_PSEQ], w[fr.F_OPID] = cseq, pseq, rid + 1
    w[fr.F_OP], w[fr.F_COUNT], w[fr.F_DTYPE] = op, kw.get("count", 1024), FLOAT
    w[fr.F_FLAGS] = kw.get("flags", 0)
    w[fr.F_CREATED] = 1_000 + rid
    w[fr.F_TAG] = kw.get("tag", 0)
    if kw.get("done", True):
        w[fr.F_DONE], w[fr.F_DONE_NS] = rid + 1, 2_000 + rid
    return w


class _Fake:
    """A Recorder over `rows` instead of the Mojo ring."""

    def __init__(self, rows: list[list[int]], world: int = 2):
        self.header = (ctypes.c_int64 * 3)()
        self.rows = rows
        self.rec = fr.Recorder(
            len(rows) or 1, ctypes.addressof(self.header), self.snapshot, "nccl"
        )
        self.rec._groups[0] = fr._Group("0", "default_pg", list(range(world)), 600_000)

    def snapshot(self, buf: int, cap: int) -> int:
        out = (ctypes.c_int64 * (cap * fr.F_WORDS)).from_address(buf)
        for i, row in enumerate(self.rows[:cap]):
            out[i * fr.F_WORDS : (i + 1) * fr.F_WORDS] = row
        return min(len(self.rows), cap)


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
    tag = fake.header[fr.H_TAG] & fr.TAG_MASK
    assert fake.header[fr.H_TAG] >> fr.TAG_THREAD_SHIFT  # the thread's bits
    fake.rows = [_word(0, ALLREDUCE, 1, tag=tag), _word(1, ALLGATHER, 2, count=5)]
    fake.rec.size = 2
    fake.rec.clear()
    assert fake.header[fr.H_TAG] == 0
    e0, e1 = fake.rec.dump()["entries"]
    assert e0["profiling_name"] == "nccl:all_reduce"
    assert e0["input_sizes"] == [[2, 3]] and e0["input_dtypes"] == ["Float"]
    assert e0["frames"][0]["name"] == "test_eager_record_and_native_entry"
    assert e1["profiling_name"] == "nccl:_all_gather_base"
    assert e1["input_sizes"] == [[5]] and e1["output_sizes"] == [[10]]
    assert e1["frames"] == [] and e1["state"] == "completed"


def test_coalesced_members_take_the_end_state():
    """A coalescing block's operations have no completion of their own: they
    complete with the block's end entry."""
    p2p = fr.FL_P2P
    fake = _Fake(
        [
            _word(0, SEND, 0, 1, flags=p2p | fr.FL_MEMBER, done=False),
            _word(1, RECV, 0, 1, flags=p2p | fr.FL_MEMBER, done=False),
            _word(2, 8, 0, 1, flags=p2p | fr.FL_END),
            _word(3, SEND, 0, 2, flags=p2p | fr.FL_MEMBER, done=False),
            _word(4, 8, 0, 2, flags=p2p | fr.FL_END, done=False),
        ]
    )
    states = [e["state"] for e in fake.rec.dump()["entries"]]
    assert states == ["completed"] * 3 + ["scheduled"] * 2
    active = fake.rec.dump(only_active=True)["entries"]
    assert [e["record_id"] for e in active] == [3, 4]


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
    ranks = {
        0: [_word(0, ALLREDUCE, 1), _word(1, ALLREDUCE, 2, done=False)],
        1: [_word(0, ALLREDUCE, 1)],
    }
    for rank, rows in ranks.items():
        dump = _Fake(rows).rec.dump()
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
    assert "Not all ranks joining collective" in output, output


def _p2p_dump(rank: int) -> dict[str, Any]:
    """batch_isend_irecv from rank 0 to rank 1 as the ring holds it: a member
    entry and the block's end entry."""
    fake = _Fake([])
    fake.rec.size = 2  # records are kept for twice the ring's size
    x = torch.ones(4)
    name, op = ("send 0->1", SEND) if rank == 0 else ("recv 1<-0", RECV)
    fake.rec.begin(name, [x], [x], 0)
    tag = fake.header[fr.H_TAG] & fr.TAG_MASK
    member = _word(0, op, 0, 1, flags=fr.FL_P2P | fr.FL_MEMBER, tag=tag, done=False)
    fake.rec.begin("coalesced", [], [], 0)
    tag = fake.header[fr.H_TAG] & fr.TAG_MASK
    end = _word(1, 8, 0, 1, flags=fr.FL_P2P | fr.FL_END, tag=tag)
    fake.rows = [member, end]
    return fake.rec.dump()


def test_torch_analyzer_pairs_the_sends_with_the_recvs(tmp_path: Path):
    for rank in (0, 1):
        (tmp_path / f"trace_{rank}").write_bytes(
            pickle.dumps(_p2p_dump(rank), protocol=2)
        )
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
    assert "MISMATCH" not in output and "Not all ranks" not in output, output
