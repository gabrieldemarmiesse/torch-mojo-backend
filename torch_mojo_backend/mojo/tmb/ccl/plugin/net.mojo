# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/plugin/net.cc

from std.os import getenv

from tmb.ccl.env_vars import MOJOCCL_NET
from tmb.ccl.transport.net_ib.init import verbs_available
from tmb.ccl.transport.net_ofi import fabric_available


# Which transport `ib_setup` opened. Runtime rather than comptime: one
# build of libmojoccl.so has to run on an InfiniBand cluster and on a
# Slingshot one, and `MOJOCCL_NET` has to be able to force either.
comptime NET_VERBS = 0
comptime NET_FABRIC = 1


def _select_backend() raises -> Int:
    """Which transport to open: `MOJOCCL_NET` wins, else what is present.

    "Present" means the library opens AND has something usable behind it --
    a login node with a Mellanox card and a compute node with four Slingshot
    NICs and no /dev/infiniband both give an unambiguous answer, and a
    machine with neither gets the same error message this library has always
    given. Verbs is tried first because it is the measured path.
    """
    var want = getenv(MOJOCCL_NET, "")
    if want == "verbs":
        return NET_VERBS
    if want == "fabric":
        return NET_FABRIC
    if want.byte_length() > 0:
        raise Error(
            "mojoccl: MOJOCCL_NET must be `verbs` or `fabric`, got " + want
        )
    if verbs_available():
        return NET_VERBS
    if fabric_available():
        return NET_FABRIC
    raise Error(
        "mojoccl: no ACTIVE InfiniBand port found (libibverbs) and no"
        " libfabric provider offering FI_RMA|FI_MSG|FI_HMEM on an FI_EP_RDM"
        " endpoint; a multi-node communicator needs one of the two."
        " MOJOCCL_NET=verbs|fabric forces a choice, MOJOCCL_LIBFABRIC points"
        " at a libfabric.so.1 that is not on the loader path"
    )
