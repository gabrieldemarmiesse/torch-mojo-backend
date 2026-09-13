"""Default MAX to the CUDA 12.8 ptxas from the nvidia-cuda-nvcc-cu12 wheel.

MAX bundles the newest CUDA's ptxas, and the cubins it assembles need a
driver at least as new (r580 for CUDA 13): on an older driver MAX refuses to
create a device at all. The wheel pinned in pyproject.toml ships ptxas 12.8,
whose cubins every r570+ driver loads, so it is the default for every
``mojo build`` subprocess and for MAX's in-process compiler. The package
imports this module first, before ``max`` loads, so the setting is in place
before anything reads it. An explicit ``MODULAR_NVPTX_COMPILER_PATH`` always
wins.

``apply_triton_default`` extends the same choice to Triton, for the code
paths that run Triton kernels on the mojo device.
"""

import importlib.util
import os
from pathlib import Path

ENV_VAR = "MODULAR_NVPTX_COMPILER_PATH"
TRITON_ENV_VAR = "TRITON_PTXAS_PATH"


def wheel_ptxas() -> Path | None:
    """The ptxas shipped by nvidia-cuda-nvcc-cu12, or None when not installed."""
    try:
        spec = importlib.util.find_spec("nvidia.cuda_nvcc")
    except ModuleNotFoundError:  # no `nvidia` namespace package at all (macOS)
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "bin" / "ptxas"
        if candidate.is_file():
            return candidate
    return None


def apply_default():
    """Point MAX at the wheel's ptxas unless the user chose one."""
    if os.environ.get(ENV_VAR):
        return
    ptxas = wheel_ptxas()
    if ptxas is not None:
        os.environ[ENV_VAR] = str(ptxas)


def torch_wheel_ptxas() -> Path | None:
    """The ptxas inside the installed torch wheel, which is what
    ``torch/_inductor/runtime/compile_tasks.py``'s ``_set_triton_ptxas_path``
    puts in ``TRITON_PTXAS_PATH`` -- at *import* of that module, so it is
    usually already set by the time anything of ours runs."""
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "bin" / "ptxas"
        if candidate.is_file():
            return candidate
    return None


def apply_triton_default():
    """Assemble Triton's kernels with the same ptxas as the device's own.

    Torch's choice tracks its wheel's CUDA rather than the driver: the cu130
    wheel on an r570 driver assembles cubins with ELF ABI version 8, and
    loading one fails with "device kernel image is invalid". MAX's ptxas is
    already pinned to something this driver loads, and a Triton kernel
    launched on the mojo device has exactly the same constraint, so it gets
    the same assembler -- but only over torch's default, recognized by its
    path. A ``TRITON_PTXAS_PATH`` pointing anywhere else is the user's and
    stands.
    """
    ptxas = os.environ.get(ENV_VAR)
    current = os.environ.get(TRITON_ENV_VAR)
    if not ptxas:
        return
    if current and Path(current) != torch_wheel_ptxas():
        return
    os.environ[TRITON_ENV_VAR] = ptxas


apply_default()
