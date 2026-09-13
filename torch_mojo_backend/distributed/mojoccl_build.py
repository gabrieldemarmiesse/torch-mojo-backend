"""Builds libmojoccl.so (the in-repo Mojo collectives, NCCL C ABI) on first
use through the native backend's cached library builder."""

from pathlib import Path

from torch_mojo_backend import native

_ENTRY = Path(__file__).parent / "mojoccl" / "mojoccl.mojo"


def ensure_built() -> str:
    """Build (if needed) and return the path to libmojoccl.so."""
    return str(native.build_library(_ENTRY))
