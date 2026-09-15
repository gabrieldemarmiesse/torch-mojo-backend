"""Builds libmojoccl.so (the in-repo Mojo collectives, NCCL C ABI) on first
use through the native backend's cached library builder."""

import os
from pathlib import Path

from torch_mojo_backend import native

_ENTRY = Path(__file__).parent / "mojoccl" / "mojoccl.mojo"


def build_defines() -> dict[str, str]:
    """`MOJOCCL_BUILD_DEFINES="ccl_fused_unroll=4,ccl_fused_threads=256"`:
    `-D` defines for the build, so a `get_defined_int` tuning constant can be
    re-fitted in one job without editing source. Part of the cache key, so
    variants coexist. Every rank must use the same value (the fused kernel's
    geometry is part of the wire layout)."""
    raw = os.environ.get("MOJOCCL_BUILD_DEFINES", "").strip()
    if not raw:
        return {}
    out = {}
    for item in raw.split(","):
        k, _, v = item.strip().partition("=")
        if not k or not v:
            raise ValueError(f"MOJOCCL_BUILD_DEFINES: expected k=v, got {item!r}")
        out[k] = v
    return out


def ensure_built() -> str:
    """Build (if needed) and return the path to libmojoccl.so."""
    return str(native.build_library(_ENTRY, defines=build_defines()))
