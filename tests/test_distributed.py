"""Tests for the mojo distributed backend (torch_mojo_backend/distributed).

Single-process pieces (registration, dtype maps, CPU/gloo delegation, library
discovery) run anywhere. Real multi-rank NCCL/RCCL coverage launches torchrun
subprocesses and needs multiple GPUs, so those tests skip on smaller machines;
they are also exercised on the cluster by the SLURM jobs in
demo_scripts/nanogpt_ddp and tests/multinode/.
"""

import datetime
import os
import re
import pickle
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist

from torch_mojo_backend import get_accelerators, register_mojo_devices
from torch_mojo_backend.distributed import nccl, use_local_rank_gpu
from torch_mojo_backend.distributed.process_group import (
    _nccl_dtype,
    _nccl_red_op,
    _report_desync,
    _set_index,
    _StoreWorker,
)
from torch_mojo_backend.mojo_device import hip_peer
from torch_mojo_backend.torch_compile_backend import utils

_WORKER = Path(__file__).parent / "ddp_worker.py"


def _gpu_count() -> int:
    return len(list(get_accelerators()))


def test_backend_name_registered():
    register_mojo_devices()
    assert "mojo" in dist.Backend.backend_list
    assert dist.Backend.default_device_backend_map.get("mojo") == "mojo"


def test_nccl_dtype_and_op_maps():
    assert _nccl_dtype(torch.bfloat16) == nccl.NCCL_BFLOAT16 == 9
    assert _nccl_dtype(torch.float32) == nccl.NCCL_FLOAT32 == 7
    assert _nccl_dtype(torch.int64) == nccl.NCCL_INT64 == 4
    assert _nccl_dtype(torch.bool) == nccl.NCCL_UINT8 == 1
    with pytest.raises(TypeError):
        _nccl_dtype(torch.complex64)
    assert _nccl_red_op(dist.ReduceOp.SUM) == nccl.NCCL_SUM
    assert _nccl_red_op(dist.ReduceOp.AVG) == nccl.NCCL_AVG


@pytest.mark.parametrize(
    "preset,local_rank,expected",
    [
        # NVIDIA, SLURM-style whole-allocation list: keep the LOCAL_RANK-th.
        ({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}, 2, {"CUDA_VISIBLE_DEVICES": "2"}),
        # AMD under SLURM (--gpus-per-task=N sets the HSA-level variable):
        # narrow it and leave HIP_VISIBLE_DEVICES alone, since that one
        # indexes INTO the ROCR-visible set and narrowing both hides all.
        (
            {"ROCR_VISIBLE_DEVICES": "0,1,2,3"},
            3,
            {"ROCR_VISIBLE_DEVICES": "3", "HIP_VISIBLE_DEVICES": None},
        ),
        # One srun task per GPU: SLURM already pinned a single device.
        (
            {"ROCR_VISIBLE_DEVICES": "0"},
            0,
            {"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": None},
        ),
        # AMD with only the runtime-level list set.
        ({"HIP_VISIBLE_DEVICES": "0,1"}, 1, {"HIP_VISIBLE_DEVICES": "1"}),
        # SLURM's gres plugin exports CUDA_VISIBLE_DEVICES next to the ROCR
        # list by default, and the HIP runtime reads it as its fallback: the
        # HSA level takes the rank's entry, the runtime level becomes the
        # only index left in a one-entry set. Narrowing both by rank would
        # hide every GPU from every rank but 0.
        (
            {"ROCR_VISIBLE_DEVICES": "0,1,2,3", "CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            2,
            {"ROCR_VISIBLE_DEVICES": "2", "CUDA_VISIBLE_DEVICES": "0"},
        ),
        (
            {"ROCR_VISIBLE_DEVICES": "0,1,2,3", "HIP_VISIBLE_DEVICES": "0,1,2,3"},
            3,
            {"ROCR_VISIBLE_DEVICES": "3", "HIP_VISIBLE_DEVICES": "0"},
        ),
        # One srun task per GPU with the runtime level already at 0.
        (
            {"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"},
            0,
            {"ROCR_VISIBLE_DEVICES": "0", "HIP_VISIBLE_DEVICES": "0"},
        ),
        # An empty value is "unset", and entries may carry spaces.
        (
            {"CUDA_VISIBLE_DEVICES": ""},
            1,
            {"CUDA_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "1"},
        ),
        ({"CUDA_VISIBLE_DEVICES": "0, 1, 2, 3"}, 2, {"CUDA_VISIBLE_DEVICES": "2"}),
        # Nothing set: pin both vendors' variables, whichever runtime is there.
        ({}, 1, {"CUDA_VISIBLE_DEVICES": "1", "HIP_VISIBLE_DEVICES": "1"}),
        # A hand-pinned single entry is respected, and nothing else is set.
        (
            {"CUDA_VISIBLE_DEVICES": "5"},
            1,
            {"CUDA_VISIBLE_DEVICES": "5", "HIP_VISIBLE_DEVICES": None},
        ),
    ],
)
def test_use_local_rank_gpu_pins_one_device_per_rank(
    monkeypatch,
    preset: dict[str, str],
    local_rank: int,
    expected: dict[str, str | None],
):
    # A private copy of the environment: the function under test ADDS
    # variables, and monkeypatch.delenv on an absent name records nothing to
    # undo, so a leaked HIP_VISIBLE_DEVICES=1 would narrow MAX's device count
    # for the rest of the session (it skipped the multi-GPU tests once).
    env = {k: v for k, v in os.environ.items() if not k.endswith("_VISIBLE_DEVICES")}
    env.update(preset)
    env["LOCAL_RANK"] = str(local_rank)
    monkeypatch.setattr(os, "environ", env)
    use_local_rank_gpu()
    for var, value in expected.items():
        assert env.get(var) == value, var


def test_use_local_rank_gpu_rejects_a_rank_beyond_the_visible_list(monkeypatch):
    env = dict(os.environ, ROCR_VISIBLE_DEVICES="0,1", LOCAL_RANK="2")
    monkeypatch.setattr(os, "environ", env)
    with pytest.raises(RuntimeError, match="lists only 2 devices"):
        use_local_rank_gpu()


def test_use_local_rank_gpu_rejects_a_non_integer_local_rank(monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ, LOCAL_RANK="zero"))
    with pytest.raises(RuntimeError, match="not an integer"):
        use_local_rank_gpu()


def test_use_local_rank_gpu_is_a_no_op_outside_torchrun(monkeypatch):
    env = {k: v for k, v in os.environ.items() if k != "LOCAL_RANK"}
    env["ROCR_VISIBLE_DEVICES"] = "0,1,2,3"
    monkeypatch.setattr(os, "environ", env)
    use_local_rank_gpu()
    assert env["ROCR_VISIBLE_DEVICES"] == "0,1,2,3"


def test_uses_mojoccl_reads_the_env_var(monkeypatch):
    monkeypatch.delenv("TORCH_MOJO_BACKEND_CCL", raising=False)
    assert nccl.uses_mojoccl() is False
    monkeypatch.setenv("TORCH_MOJO_BACKEND_CCL", "mojo")
    assert nccl.uses_mojoccl() is True
    monkeypatch.setenv("TORCH_MOJO_BACKEND_CCL", "MoJo")  # case-insensitive
    assert nccl.uses_mojoccl() is True
    monkeypatch.setenv("TORCH_MOJO_BACKEND_CCL", "vendor")
    assert nccl.uses_mojoccl() is False


def test_vendor_name_reflects_the_max_accelerator_api(monkeypatch):
    monkeypatch.setattr(
        utils, "get_accelerators", lambda: [SimpleNamespace(api="cuda")]
    )
    assert nccl.vendor_name() == "nccl"
    monkeypatch.setattr(utils, "get_accelerators", lambda: [SimpleNamespace(api="hip")])
    assert nccl.vendor_name() == "rccl"
    monkeypatch.setattr(utils, "get_accelerators", lambda: [])
    assert nccl.vendor_name() == "nccl"


def test_library_path_builds_mojoccl_when_selected(monkeypatch):
    monkeypatch.setenv("TORCH_MOJO_BACKEND_CCL", "mojo")
    monkeypatch.setattr(
        "torch_mojo_backend.distributed.mojoccl_build.ensure_built",
        lambda: "/fake/libmojoccl.so",
    )
    assert nccl.library_path() == "/fake/libmojoccl.so"


def test_library_path_picks_the_vendor_candidate_that_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("TORCH_MOJO_BACKEND_CCL", raising=False)
    fake = tmp_path / "libnccl.so.2"
    fake.write_bytes(b"")
    monkeypatch.setattr(nccl, "vendor_name", lambda: "nccl")
    monkeypatch.setattr(
        nccl,
        "_candidate_libnccl_paths",
        lambda: ["/nonexistent/libnccl.so.2", str(fake)],
    )
    assert nccl.library_path() == str(fake)


def test_library_path_raises_a_clear_error_when_nothing_is_found(monkeypatch):
    monkeypatch.delenv("TORCH_MOJO_BACKEND_CCL", raising=False)
    monkeypatch.setattr(nccl, "vendor_name", lambda: "nccl")
    monkeypatch.setattr(
        nccl, "_candidate_libnccl_paths", lambda: ["/nonexistent/libnccl.so.2"]
    )
    with pytest.raises(RuntimeError, match="no NCCL/RCCL library found"):
        nccl.library_path()


_GPU_PROBE = """
import os
import torch

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.distributed import nccl
from torch_mojo_backend.mojo_device import hip_peer

register_mojo_devices()
on_mojo = torch.zeros(8, device="mojo:0")
assert on_mojo.device.type == "mojo"

vendor = nccl.vendor_name()
print("vendor", vendor)
if not nccl.uses_mojoccl():
    path = nccl.library_path()
    assert os.path.exists(path), path
    print("library", path)
if vendor == "rccl":
    torch.accelerator.synchronize()
    assert hip_peer.available()
    assert hip_peer.runtime_dir() is not None  # MAX already mapped libamdhip64
    assert hip_peer.device_ordinal(on_mojo.data_ptr()) is not None
    assert hip_peer.device_ordinal(0) is None
    assert hip_peer.device_ordinal(torch.zeros(8).data_ptr()) is None
    print("hip_peer ok")
"""


def _run_gpu_probe() -> str:
    """The vendor-library and hip_peer checks, in a process of their own.

    Not in the pytest process on purpose: the first mojo tensor makes MAX
    reserve most of the device's memory for its pool, and on an APU
    (MI300A: 128 GB shared with the host) a torchrun rank pinned to that
    same GPU later in the session then finds too little left for RCCL's
    P2P buffers (`Failed to CUDA calloc`, seen as ncclRecv "unhandled cuda
    error"). A subprocess gives it all back when it exits.
    """
    result = subprocess.run(
        [sys.executable, "-c", _GPU_PROBE],
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=900,
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"gpu probe failed (rc={result.returncode})\n"
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )
    return result.stdout


@pytest.mark.gpu
def test_collective_library_matches_the_gpu_vendor():
    """NCCL on NVIDIA, RCCL on AMD — one binding, the vendor picks the .so."""
    if not any(d.api in ("cuda", "hip") for d in get_accelerators()):
        pytest.skip("NCCL/RCCL requires a CUDA or HIP GPU")
    out = _run_gpu_probe()
    vendor = out.split("vendor ", 1)[1].split()[0]
    assert vendor in ("nccl", "rccl"), out
    assert "library " in out, out


@pytest.mark.gpu
def test_hip_pointer_ordinal_identifies_the_owning_gpu():
    """hip_peer reads device identity off the POINTER, like cuda_peer does:
    the ordinal RCCL binds a communicator to is a fact about the allocation,
    not an assumption that MAX and HIP enumerate alike."""
    if not any(d.api == "hip" for d in get_accelerators()):
        pytest.skip("needs an AMD GPU")
    out = _run_gpu_probe()
    if "vendor rccl" not in out:
        pytest.skip("needs an AMD GPU")
    assert "hip_peer ok" in out, out


@pytest.mark.parametrize(
    "cuda_version,hip_version,apis,amd_present,expect",
    [
        ("13.0", None, ["hip", "cpu"], True, "CUDA build"),  # CUDA wheel on AMD
        (None, "6.4.4", ["hip", "cpu"], True, "ROCm build"),  # two HIP runtimes
        (None, None, ["hip", "cpu"], True, None),  # the CPU wheel: what AMD wants
        ("13.0", None, ["cuda", "cpu"], False, None),  # CUDA wheel on NVIDIA
        ("13.0", None, ["hip", "cpu"], False, None),  # no /dev/kfd: never enumerates
    ],
)
def test_gpu_torch_on_hip_hint(
    monkeypatch,
    cuda_version: str | None,
    hip_version: str | None,
    apis: list[str],
    amd_present: bool,
    expect: str | None,
):
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(torch.version, "hip", hip_version)
    monkeypatch.setattr(hip_peer, "_amd_gpu_present", lambda: amd_present)
    enumerated: list[bool] = []

    def fake_accelerators() -> list[SimpleNamespace]:
        enumerated.append(True)
        return [SimpleNamespace(api=api) for api in apis]

    monkeypatch.setattr(utils, "get_accelerators", fake_accelerators)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        hip_peer.warn_if_gpu_torch_on_hip()
    hints = [str(w.message) for w in caught if "torch-mojo-backend" in str(w.message)]
    if expect is None:
        assert hints == []
    else:
        assert len(hints) == 1 and expect in hints[0], hints
        assert "download.pytorch.org/whl/cpu" in hints[0]
    # Registration must not touch MAX's device enumeration unless an AMD GPU
    # is present and torch is a GPU build.
    assert bool(enumerated) == (
        amd_present and (cuda_version or hip_version) is not None
    )


def test_gpu_torch_on_hip_hint_never_breaks_registration(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(hip_peer, "_amd_gpu_present", lambda: True)

    def broken() -> list[object]:
        raise RuntimeError("enumeration failed")

    monkeypatch.setattr(utils, "get_accelerators", broken)
    hip_peer.warn_if_gpu_torch_on_hip()  # swallowed: a hint, not a gate


def _watchdog_line(
    seq: int, op: str, count: int, root: int = -1, cpos: int | None = None
) -> str:
    """watchdog.mojo's line; the fingerprint stands in for the arguments.
    `cpos` (the collective position) defaults to `seq` for collectives and
    to 0, uncounted, for point-to-point."""
    if cpos is None:
        cpos = 0 if op in ("send", "recv") else seq
    return (
        f"id=1 tid=0 seq={seq} cpos={cpos} done={seq - 1} last={seq} op={op}"
        f" count={count} dtype=float32 root={root} redop=-1 nops=1"
        f" fp={hash((op, count, root))} device=0 pending_ms=10000 where=device"
    )


def _publish(store: dist.Store, rank: int, line: str):
    """What another rank's reporter leaves in the store for an incident."""
    store.set(f"mojo_watchdog/{rank}/1", line)
    store.set(f"mojo_watchdog/{rank}", f"1 mojo_watchdog/{rank}/1")


def _report(line: str, store: dist.Store, world: int):
    _report_desync(line, _StoreWorker(store, clone=False), 0, world, 0.3)


def test_desync_report_names_the_diverging_ranks(capsys):
    """The report the watchdog's Python thread prints from the store: rank 1
    is in a broadcast with another root at the same position, rank 2 is
    behind, rank 3 never answers (rank 5's incident is over: an empty
    index), rank 4 waits where this rank does, rank 6 too although a
    send before it gave it another local sequence number."""
    store = dist.HashStore()
    _publish(store, 1, _watchdog_line(2, "broadcast", 4, root=1))
    _publish(store, 2, _watchdog_line(1, "broadcast", 4, root=0))
    _publish(store, 4, _watchdog_line(2, "broadcast", 4, root=0))
    store.set("mojo_watchdog/5", "2")
    _publish(store, 6, _watchdog_line(3, "broadcast", 4, root=0, cpos=2))
    _report(_watchdog_line(2, "broadcast", 4, root=0), store, 7)
    err = capsys.readouterr().err
    assert (
        "[rank 0] desync report for #2 (collective #2) (broadcast 4 x float32"
        " root 0)" in err
    )
    assert (
        "[rank 0]   rank 1: waiting in collective #2 broadcast 4 x float32 root 1"
        "  <-- the same position, different arguments" in err
    )
    assert (
        "[rank 0]   rank 2: waiting in collective #1 broadcast 4 x float32 root 0"
        "  <-- behind" in err
    )
    assert "[rank 0]   waiting in this same collective: ranks 0, 4, 6\n" in err
    assert "[rank 0]   no report: ranks 3, 5." in err
    assert "Inconclusive" not in err


def test_desync_report_with_every_rank_in_the_same_collective_is_inconclusive(capsys):
    store = dist.HashStore()
    _publish(store, 1, _watchdog_line(5, "allreduce", 8))
    _report(_watchdog_line(5, "allreduce", 8), store, 2)
    err = capsys.readouterr().err
    assert "[rank 0]   waiting in this same collective: ranks 0, 1\n" in err
    assert "Inconclusive: every rank that reported is waiting where this one" in err


def test_desync_report_does_not_compare_point_to_point(capsys):
    """A send and its matching recv differ in op and peer by construction:
    a P2P wait is listed, never called a desync."""
    store = dist.HashStore()
    _publish(store, 1, _watchdog_line(3, "recv", 8, root=0))
    _report(_watchdog_line(4, "send", 8, root=1), store, 2)
    err = capsys.readouterr().err
    assert "rank 1: waiting in #3 recv 8 x float32 peer 0 (not compared" in err
    assert "<--" not in err
    assert "Inconclusive" in err


def test_desync_index_never_goes_back_a_generation():
    """An index update that lands late (its store worker was abandoned)
    does not roll a newer index back."""
    store = dist.HashStore()
    assert _set_index(store, 3, 5, ["mojo_watchdog/3/9"])
    assert _set_index(store, 3, 4, ["mojo_watchdog/3/1"])  # stale: ignored
    assert bytes(store.get("mojo_watchdog/3")).decode() == "5 mojo_watchdog/3/9"
    assert _set_index(store, 3, 6, [])
    assert bytes(store.get("mojo_watchdog/3")).decode() == "6"


def test_desync_report_does_not_compare_different_groupings(capsys):
    """Two allreduces grouped on this rank, issued one by one on rank 1: the
    same calls (overlapping positions), fingerprints different by
    construction -- not compared, and not called a desync."""
    store = dist.HashStore()
    _publish(store, 1, _watchdog_line(2, "allreduce", 8))
    grouped = _watchdog_line(2, "allreduce", 16).replace("nops=1", "nops=2")
    _report(grouped, store, 2)
    err = capsys.readouterr().err
    assert "rank 1: waiting in collective #2 allreduce 8 x float32" in err
    assert "(not compared: grouped differently)" in err
    assert "<--" not in err
    assert "Inconclusive" in err


def test_desync_report_outranks_an_earlier_incarnation(capsys):
    """A group re-created under the same name on the same store: its first
    report replaces the index the previous incarnation left behind."""
    store = dist.HashStore()
    store.set("mojo_watchdog/0", "7 mojo_watchdog/0/1")  # the old group's
    _publish(store, 1, _watchdog_line(2, "allreduce", 8))
    _report_desync(
        _watchdog_line(2, "allreduce", 8),
        _StoreWorker(store, clone=False),
        0,
        2,
        0.3,
        gen=time.time_ns(),
    )
    index = bytes(store.get("mojo_watchdog/0")).decode().split()
    assert int(index[0]) > 7 and index[1:] == ["mojo_watchdog/0/0/1"]
    assert "waiting in this same collective: ranks 0, 1" in capsys.readouterr().err


def test_desync_report_survives_a_store_that_does_not_answer(capsys):
    """A store stuck in a blocking call: the report still comes out, with
    the store's silence said, within the window plus a margin."""

    class _Stuck:
        def set(self, *args: object):
            time.sleep(3600)

    start = time.monotonic()
    _report_desync(
        _watchdog_line(2, "allreduce", 8),
        _StoreWorker(cast(dist.Store, _Stuck()), clone=False),
        0,
        2,
        0.3,
    )
    err = capsys.readouterr().err
    assert time.monotonic() - start < 5
    assert "the c10d store did not answer in time" in err
    assert "[rank 0] desync report for #2" in err


def test_cpu_collectives_through_gloo_delegation():
    """World-size-1 init with backend="mojo": CPU tensors ride the private gloo."""
    register_mojo_devices()
    store = dist.TCPStore("127.0.0.1", 29517, 1, is_master=True)
    dist.init_process_group(
        backend="mojo",
        store=store,
        rank=0,
        world_size=1,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        t = torch.arange(4.0)
        dist.all_reduce(t)
        assert t.tolist() == [0.0, 1.0, 2.0, 3.0]
        objs = [None]
        dist.all_gather_object(objs, {"hello": "world"})
        assert objs[0] == {"hello": "world"}
        dist.barrier()
        assert dist.get_backend() == "mojo"
    finally:
        dist.destroy_process_group()


def _run_torchrun(
    nproc: int,
    mode: str,
    extra_env: dict[str, str] | None = None,
    worker: Path = _WORKER,
    expect_success: bool = True,
    timeout: float = 900,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # Flight recorder dumps of a timeout go here, not to ~/.cache/torch.
    env.setdefault(
        "TORCH_FR_DUMP_TEMP_FILE",
        os.path.join(tempfile.mkdtemp(prefix="mojo-fr-"), "trace_"),
    )
    # The worker pins per-rank visibility from LOCAL_RANK, slicing whichever
    # vendor list the launcher left (SLURM sets ROCR_VISIBLE_DEVICES on AMD).
    env.update(extra_env or {})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            str(worker),
            mode,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=Path(__file__).parent.parent,
    )
    if expect_success and result.returncode != 0:
        raise AssertionError(
            f"torchrun {mode} failed (rc={result.returncode})\n"
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )
    return result


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.parametrize(
    "mode", ["collectives", "ddp_parity", "stream_ordering", "stress", "abort"]
)
@pytest.mark.gpu
def test_two_rank_nccl(mode: str, ccl: str):
    """`ccl="mojo"` runs the same workers against mojoccl
    (torch_mojo_backend/mojo/tmb/ccl), the in-repo NCCL-API library,
    instead of vendor NCCL/RCCL — same process_group.py, same ddp_worker.py,
    only the loaded .so differs (nccl.py's `library_path()`). ddp_worker.py
    itself skips the collectives mojoccl does not implement yet (Reduce,
    Send/Recv, AllToAll, Gather/Scatter) when this is set.

    `mode="stress"` hammers every collective back to back at mixed sizes; the
    mojoccl-only collectives inside it are skipped the same way `collectives`
    skips them, so it runs (and asserts something) under both `ccl` values.

    `mode="abort"` exercises `MojoProcessGroup.abort()`/`shutdown()` and needs
    2+ ranks; abort() deliberately drops the aborted communicator rather than
    wedging the group, so a collective issued right after it transparently
    rebuilds one, and it behaves the same under both libraries.
    """
    if _gpu_count() < 2:
        pytest.skip("needs at least 2 GPUs")
    extra_env = {"TORCH_MOJO_BACKEND_CCL": ccl}
    _run_torchrun(2, mode, extra_env)


def _run_desync(mode: str, ccl: str, extra_env: dict[str, str] | None = None):
    if _gpu_count() < 2:
        pytest.skip("needs at least 2 GPUs")
    start = time.monotonic()
    result = _run_torchrun(
        2,
        mode,
        {"TORCH_MOJO_BACKEND_CCL": ccl, **(extra_env or {})},
        expect_success=False,
        timeout=600,
    )
    elapsed = time.monotonic() - start
    output = result.stdout + result.stderr
    context = f"rc={result.returncode}, {elapsed:.0f} s\n{output[-8000:]}"
    assert "UNEXPECTED" not in output, context
    return result.returncode, output, context


def _assert_aborted(rc: int, output: str, context: str, seq: int, what: str):
    """Rank 0 reported collective `seq` within the timeout and exited."""
    assert rc != 0, context
    assert (
        f"[rank 0] Watchdog caught collective operation timeout: collective #{seq}"
        f" ({what}) on device 0 has not completed" in output
    ), context
    assert f"[rank 0] desync report for #{seq} (collective #{seq})" in output, context
    assert "[rank 0]   waiting in this same collective: ranks 0\n" in output, context
    assert "[rank 0]   no report: ranks 1." in output, context
    assert "Aborting the communicators and exiting with status 1" in output, context
    pending = re.search(r"has not completed (\d+\.\d) s after it was issued", output)
    assert pending and 10.0 <= float(pending.group(1)) < 15.0, context


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_desync_watchdog(ccl: str):
    """A collective one rank never matches fails the run instead of hanging.

    ddp_worker.py's `desync` mode: on a group with a 10 s timeout, rank 0
    issues an all_reduce (#2) that rank 1 never issues. Rank 0's watchdog
    must name the collective, print the desync report (rank 1 never answers:
    it is not in a collective), and exit non-zero; torchrun then stops rank 1.
    """
    rc, output, context = _run_desync("desync", ccl)
    _assert_aborted(rc, output, context, 2, "allreduce 1024 x float32 sum")
    assert "#1 allreduce 1024 x float32 sum  completed" in output, context


def _fr_trace(directory: Path) -> tuple[int, str]:
    """torch's offline analyzer over the dumps in `directory`."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.flight_recorder.fr_trace",
            str(directory),
            "-p",
            "trace_",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    return result.returncode, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("ccl", "buffer"),
    [("vendor", "2000"), ("mojo", "2000"), ("vendor", "8"), ("vendor", "0")],
)
@pytest.mark.gpu
def test_two_rank_flight_recorder(ccl: str, buffer: str, tmp_path: Path):
    """ddp_worker.py's `flight_recorder` mode checks `_dump_nccl_trace`
    in-process (names, seq ids, shapes, states, the ring keeping the newest
    TORCH_FR_BUFFER_SIZE, nothing at 0); each rank then dumps through the
    debug pipe, and torch's analyzer reads the two dumps without an error."""
    if _gpu_count() < 2:
        pytest.skip("needs at least 2 GPUs")
    env = {
        "TORCH_MOJO_BACKEND_CCL": ccl,
        "TORCH_FR_BUFFER_SIZE": buffer,
        "TORCH_FR_DUMP_TEMP_FILE": str(tmp_path / "trace_"),
    }
    if buffer != "0":
        env["TORCH_NCCL_DEBUG_INFO_PIPE_FILE"] = str(tmp_path / "pipe_")
    _run_torchrun(2, "flight_recorder", env)
    if buffer == "0":
        assert not list(tmp_path.glob("trace_*"))
        return
    for f in tmp_path.glob("pipe_*"):
        f.unlink()  # the analyzer reads every file of the directory
    rc, output = _fr_trace(tmp_path)
    assert rc == 0, output
    for problem in ("has errors", "Not all ranks", "MISMATCH"):
        assert problem not in output, output


def _fr_dumps(directory: Path) -> dict[int, dict[str, Any]]:
    return {
        int(f.name[len("trace_") :]): pickle.loads(f.read_bytes())
        for f in directory.glob("trace_*")
    }


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_flight_recorder_desync(ccl: str, tmp_path: Path):
    """The `desync` hang dumps every rank's flight recorder before the abort:
    rank 0 on its timeout, rank 1 on rank 0's store signal. torch's analyzer
    then names collective #2 of the short group as the one rank 1 never
    joined, and rank 0 printed that collective's Python stack."""
    rc, output, context = _run_desync(
        "desync", ccl, {"TORCH_FR_DUMP_TEMP_FILE": str(tmp_path / "trace_")}
    )
    _assert_aborted(rc, output, context, 2, "allreduce 1024 x float32 sum")
    assert "[rank 0] flight recorder: collective timeout, dumped to" in output, context
    assert "[rank 1] flight recorder: rank 0 signalled a dump, dumped to" in output, (
        context
    )
    assert re.search(
        r"\[rank 0\]   File .*ddp_worker.py\", line \d+, in run_desync", output
    ), context
    dumps = _fr_dumps(tmp_path)
    assert sorted(dumps) == [0, 1], context
    hung = [e for e in dumps[0]["entries"] if e["state"] != "completed"]
    assert [(e["process_group"][0], e["collective_seq_id"]) for e in hung] == [
        ("1", 2)
    ], dumps[0]
    rc, analysis = _fr_trace(tmp_path)
    assert rc == 0, analysis
    assert "Not all ranks joining collective, sequence number: 2" in analysis, analysis
    assert "missing ranks: {1}" in analysis, analysis
    assert "run_desync" in analysis, analysis  # the collective's stack


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_desync_watchdog_after_a_burst(ccl: str):
    """3000 collectives in flight at once overflow the watchdog's 1024-slot
    ring; the unmatched one issued right behind them is still the one
    reported, with its exact sequence number."""
    rc, output, context = _run_desync("desync_burst", ccl)
    _assert_aborted(rc, output, context, 3002, "allreduce 4 x float32 sum")


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_desync_watchdog_destroy(ccl: str):
    """Destroying a group whose collective hangs is not a hang either: the
    destroy waits for the collective, the watchdog reports it and exits."""
    rc, output, context = _run_desync("desync_destroy", ccl)
    _assert_aborted(rc, output, context, 2, "allreduce 1024 x float32 sum")


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_desync_watchdog_report_mode(ccl: str):
    """TORCH_MOJO_BACKEND_PG_WATCHDOG=report: three late collectives (#2 late
    on rank 0, #3 on rank 1, #4 on rank 0) are each reported by the rank
    left waiting, which keeps running; the run then finishes cleanly. A
    record of an earlier incident is gone from the store by the next one,
    so every report names the late rank as silent, never as behind."""
    rc, output, context = _run_desync(
        "desync_report_twice", ccl, {"TORCH_MOJO_BACKEND_PG_WATCHDOG": "report"}
    )
    assert rc == 0, context
    assert output.count("report_twice finished") == 2, context
    for waiting, late, seq in ((1, 0, 2), (0, 1, 3), (1, 0, 4)):
        assert (
            f"[rank {waiting}] Watchdog caught collective operation timeout:"
            f" collective #{seq} " in output
        ), context
        report = re.search(
            rf"\[rank {waiting}\] desync report for #{seq} .*?"
            rf"\[rank {waiting}\]   no report: ranks {late}\.",
            output,
            re.S,
        )
        assert report and "<-- behind" not in report.group(0), context
    assert "Aborting the communicators" not in output, context


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_desync_watchdog_recovers_after_overflow(ccl: str):
    """A full queue (lowered to 256 in flight) drops the burst's newest
    collectives with a warning; once the burst drained, tracking resumes and
    the unmatched collective after it is reported."""
    rc, output, context = _run_desync("desync_overflow", ccl)
    _assert_aborted(rc, output, context, 4002, "allreduce 4 x float32 sum")
    assert "the newest ones are not watched until some complete" in output, context
    assert "watching new ones again" in output, context


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["desync_at_exit", "desync_ctrl_c"])
def test_two_rank_desync_watchdog_at_interpreter_exit(mode: str, ccl: str):
    """An unmatched collective in flight when the script ends (sys.exit,
    Ctrl-C), without destroy_process_group: reported after a short grace,
    not after the timeout, and the run fails."""
    rc, output, context = _run_desync(mode, ccl)
    assert rc != 0, context
    found = re.search(
        r"\[rank 0\] Watchdog: the process is exiting with collective #2 \(allreduce"
        r" 1024 x float32 sum\) on device 0 still in flight, (\d+\.\d) s after",
        output,
    )
    assert found and float(found.group(1)) < 5.0, context
    assert "[rank 0] desync report for #2 (collective #2)" in output, context
    assert "Aborting the communicators and exiting with status 1" in output, context


@pytest.mark.gpu
def test_two_rank_desync_watchdog_exit_with_a_call_blocked_on_the_host():
    """The main thread exits while a daemon thread is blocked on the host in
    the library (vendor NCCL's lazy p2p connect): the exit does not swallow
    the report. The blocked thread holds the backend mutex, so the exiting
    thread may itself block freeing a tensor before the exit hook runs; the
    watchdog's timeout then reports it, so either report is the right one."""
    rc, output, context = _run_desync("desync_exit_host_blocked", "vendor")
    assert rc != 0, context
    assert (
        "the ncclGroupEnd call has been blocked on the host inside the"
        " collectives library" in output
    ), context
    assert "Aborting the communicators and exiting with status 1" in output, context


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_watchdog_off_destroy_completes_collectives(ccl: str):
    """With the watchdog off, destroying a group still lets its collectives
    in flight finish (rank 1 issues its side 3 s late) instead of cancelling
    them with the local abort."""
    rc, output, context = _run_desync(
        "watchdog_off_destroy", ccl, {"TORCH_MOJO_BACKEND_PG_WATCHDOG": "off"}
    )
    assert rc == 0, context
    assert "[rank 0] off destroy correct: True" in output, context
    assert "[rank 1] off destroy correct: True" in output, context


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_watchdog_slow_peer_at_destroy(ccl: str):
    """Rank 0 reaches destroy_process_group 15 s after rank 1 (a final
    checkpoint, say), past the 10 s timeout: a healthy run, which must not
    fail -- the destroy does not wait for its peers."""
    rc, output, context = _run_desync("watchdog_slow_destroy", ccl)
    assert rc == 0, context
    assert output.count("slow_destroy finished") == 2, context
    assert "Watchdog" not in output, context


@pytest.mark.gpu
def test_two_rank_desync_watchdog_host_blocked():
    """A call that blocks on the host instead of the GPU: vendor NCCL's first
    point-to-point call to a peer connects the pair inside ncclGroupEnd and
    waits there for the peer, which never comes. The in-call mark reports
    it and the process exits (mojoccl has no point-to-point)."""
    rc, output, context = _run_desync("desync_host_blocked", "vendor")
    assert rc != 0, context
    assert (
        "[rank 0] Watchdog caught collective operation timeout: the ncclGroupEnd"
        " call has been blocked on the host inside the collectives library for"
        in output
    ), context
    assert "Aborting the communicators and exiting with status 1" in output, context


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_watchdog_never_blocks_the_issuer(ccl: str):
    """900 async all_reduces issued on rank 0 before the store signal that
    lets rank 1 issue its own: the issuing thread must return from every
    call before the peer can progress, so the watchdog must not wait for
    the GPU. (Past ~1024 both libraries block the host themselves -- vendor
    NCCL's work FIFO, mojoccl's queue -- and the watchdog then reports the
    blocked call.)"""
    rc, output, context = _run_desync("watchdog_async_burst", ccl)
    assert rc == 0, context
    assert "[rank 0] async burst correct: True" in output, context
    assert "[rank 1] async burst correct: True" in output, context
    assert "Watchdog" not in output, context


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_watchdog_clean_exit_without_destroy(ccl: str):
    """Collectives in flight at interpreter exit, no destroy_process_group:
    the exit hook stops the watchdog before teardown, so nothing is
    reported and the run exits 0."""
    rc, output, context = _run_desync("watchdog_clean_exit", ccl)
    assert rc == 0, context
    assert output.count("exiting without destroy_process_group") == 2, context
    assert "Watchdog" not in output and "watchdog:" not in output, context


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_watchdog_fault_then_destroy_report_mode(ccl: str):
    """A driver error on a communicator (injected on every rank) is reported
    in report mode; the group's destroy then aborts that communicator
    instead of destroying it on the word of a queue nobody watches any
    more."""
    rc, output, context = _run_desync(
        "watchdog_fault_destroy", ccl, {"TORCH_MOJO_BACKEND_PG_WATCHDOG": "report"}
    )
    assert rc == 0, context
    assert output.count("fault_destroy finished") == 2, context
    assert (
        "[rank 0] Watchdog: an event record on device 0 failed with driver error -2"
        in output
    ), context
    assert (
        "[rank 0] watchdog: collectives of this process group are unaccounted for;"
        " aborting its communicators without waiting for them" in output
    ), context


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.parametrize(
    "mode", ["parity", "reduce_scatter", "fsdp_collectives_stress"]
)
def test_two_rank_fsdp2(mode: str, ccl: str):
    if _gpu_count() < 2:
        pytest.skip("needs at least 2 GPUs")
    extra_env = {"TORCH_MOJO_BACKEND_CCL": ccl}
    _run_torchrun(2, mode, extra_env, Path(__file__).parent / "fsdp_worker.py")


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.parametrize(
    "mode",
    ["functional", "interleaved", "tensor_parallel", "unsupported", "flight_recorder"],
)
def test_two_rank_compiled_collectives(mode: str, ccl: str):
    """Collectives inside graphs compiled with `mojo_backend`
    (torch_compile_backend/collectives.py), compared with eager mode."""
    if _gpu_count() < 2:
        pytest.skip("needs at least 2 GPUs")
    extra_env = {"TORCH_MOJO_BACKEND_CCL": ccl}
    _run_torchrun(2, mode, extra_env, Path(__file__).parent / "compile_c10d_worker.py")


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.gpu
def test_two_rank_desync_watchdog_compiled_graph(ccl: str):
    """A collective issued from a graph compiled with `mojo_backend` that
    rank 1 never issues is watched like an eager one: reported, and the run
    exits."""
    if _gpu_count() < 2:
        pytest.skip("needs at least 2 GPUs")
    result = _run_torchrun(
        2,
        "desync_compiled",
        {"TORCH_MOJO_BACKEND_CCL": ccl},
        Path(__file__).parent / "compile_c10d_worker.py",
        expect_success=False,
        timeout=600,
    )
    output = result.stdout + result.stderr
    context = f"rc={result.returncode}\n{output[-8000:]}"
    assert "UNEXPECTED" not in output, context
    _assert_aborted(
        result.returncode, output, context, 2, "allreduce 1024 x float32 sum"
    )


def test_two_rank_reduce_scatter_chunked():
    """mojoccl's reduce-scatter over a region too small to hold the message.

    One launch per call is the normal case (the push slots may use the whole
    staging arena, 512 MiB at the default region), so the chunk loop —
    offsets into the input, a `generation` per chunk — is only reached by a
    message larger than that. A 1 MiB region puts the 700k-element cases of
    `check_reduce_scatter` over it.
    """
    if _gpu_count() < 2:
        pytest.skip("needs at least 2 GPUs")
    extra_env = {"TORCH_MOJO_BACKEND_CCL": "mojo", "MOJOCCL_REGION_MB": "1"}
    _run_torchrun(
        2, "reduce_scatter", extra_env, Path(__file__).parent / "fsdp_worker.py"
    )
