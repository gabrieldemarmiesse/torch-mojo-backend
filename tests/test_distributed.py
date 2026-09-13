"""Tests for the mojo distributed backend (torch_mojo_backend/distributed).

Single-process pieces (registration, dtype maps, CPU/gloo delegation, library
discovery) run anywhere. Real multi-rank NCCL/RCCL coverage launches torchrun
subprocesses and needs multiple GPUs, so those tests skip on smaller machines;
they are also exercised on the cluster by the SLURM jobs in
demo_scripts/nanogpt_ddp and tests/multinode/.
"""

import datetime
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from torch_mojo_backend import get_accelerators, register_mojo_devices
from torch_mojo_backend.distributed import nccl, use_local_rank_gpu
from torch_mojo_backend.distributed.process_group import _nccl_dtype, _nccl_red_op
from torch_mojo_backend.mojo_device import hip_peer
from torch_mojo_backend.torch_compile_backend import utils

_WORKER = Path(__file__).parent / "ddp_worker.py"


def _gpu_count() -> int:
    return len(list(get_accelerators())) - 1  # get_accelerators appends the CPU


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


def test_collective_library_matches_the_gpu_vendor():
    """NCCL on NVIDIA, RCCL on AMD — one binding, the vendor picks the .so."""
    if _gpu_count() < 1:
        pytest.skip("needs a GPU")
    out = _run_gpu_probe()
    vendor = out.split("vendor ", 1)[1].split()[0]
    assert vendor in ("nccl", "rccl"), out
    assert "library " in out, out


def test_hip_pointer_ordinal_identifies_the_owning_gpu():
    """hip_peer reads device identity off the POINTER, like cuda_peer does:
    the ordinal RCCL binds a communicator to is a fact about the allocation,
    not an assumption that MAX and HIP enumerate alike."""
    if _gpu_count() < 1:
        pytest.skip("needs a GPU")
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


def _run_torchrun(nproc: int, mode: str, extra_env: dict[str, str] | None = None):
    env = dict(os.environ)
    # The worker pins per-rank visibility from LOCAL_RANK, slicing whichever
    # vendor list the launcher left (SLURM sets ROCR_VISIBLE_DEVICES on AMD).
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.update(extra_env or {})
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={nproc}",
            str(_WORKER),
            mode,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        cwd=Path(__file__).parent.parent,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"torchrun {mode} failed (rc={result.returncode})\n"
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )


@pytest.mark.parametrize("ccl", ["vendor", "mojo"])
@pytest.mark.parametrize(
    "mode", ["collectives", "ddp_parity", "stream_ordering", "stress", "abort"]
)
def test_two_rank_nccl(mode: str, ccl: str):
    """`ccl="mojo"` runs the same workers against mojoccl
    (torch_mojo_backend/distributed/mojoccl), the in-repo NCCL-API library,
    instead of vendor NCCL/RCCL — same process_group.py, same ddp_worker.py,
    only the loaded .so differs (nccl.py's `library_path()`). ddp_worker.py
    itself skips the collectives mojoccl does not implement yet (Reduce,
    ReduceScatter, Send/Recv, AllToAll, Gather/Scatter) when this is set.

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
    extra_env = {"TORCH_MOJO_BACKEND_CCL": "mojo"} if ccl == "mojo" else None
    _run_torchrun(2, mode, extra_env)
