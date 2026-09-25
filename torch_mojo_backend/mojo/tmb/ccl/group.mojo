# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/group.cc

from tmb.ccl.nccl import NCCL_SUCCESS


# ---------------------------------------------------------------------------
# Group semantics: every op here already executes on the same stream, in
# issue order, so aggregating a group buys nothing -- immediate execution is
# correct, per the design brief.
# ---------------------------------------------------------------------------


def ncclGroupStart() -> Int32:
    return NCCL_SUCCESS


def ncclGroupEnd() -> Int32:
    return NCCL_SUCCESS
