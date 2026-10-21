import os

POSITIVE_VALUES = ("1", "true", "yes")

# Module constants: the env-var registry test reads names from these.
PROFILE = "TORCH_MOJO_BACKEND_PROFILE"
LEGACY_PROFILE = "PYTORCH_MOJO_BACKEND_PROFILE"
VERBOSE = "TORCH_MOJO_BACKEND_VERBOSE"
LEGACY_VERBOSE = "PYTORCH_MOJO_BACKEND_VERBOSE"
DEBUG_GRAPH = "TORCH_MOJO_BACKEND_DEBUG_GRAPH"
LEGACY_DEBUG_GRAPH = "PYTORCH_MOJO_BACKEND_DEBUG_GRAPH"


def _enabled(name: str, legacy_name: str) -> bool:
    """`name` (or its legacy PYTORCH_-prefixed alias) is set to a truthy value."""
    return (
        os.environ.get(name, "0").lower() in POSITIVE_VALUES
        or os.environ.get(legacy_name, "0").lower() in POSITIVE_VALUES
    )


def profiling_enabled() -> bool:
    return _enabled(PROFILE, LEGACY_PROFILE)


def verbose_enabled() -> bool:
    return _enabled(VERBOSE, LEGACY_VERBOSE)


def debug_graph() -> bool:
    return _enabled(DEBUG_GRAPH, LEGACY_DEBUG_GRAPH)


def compile_native_kernels_enabled() -> bool:
    """Whether the torch.compile backend calls this repository's eager kernels
    (`tmb/graph/gemm.mojo`, `nn.mojo`) for the ops that have one -- the GEMMs, softmax,
    layer norm, embedding -- instead of composing MAX's own ops.
    `TORCH_MOJO_BACKEND_COMPILE_NATIVE_KERNELS=0` keeps MAX's kernels.
    """
    x = os.environ.get("TORCH_MOJO_BACKEND_COMPILE_NATIVE_KERNELS", "1").lower()
    return x not in ("0", "false", "no")
