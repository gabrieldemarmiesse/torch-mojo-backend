# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/include/transport.h

from std.utils import StaticTuple

from tmb.ccl.include.device import MAX_WORLD


struct NvlsRegion(Movable):
    """One rank's slice of a node-wide multicast allocation.

    `mc` is the multicast VA -- only `multimem.*` may touch it. `uc` is a plain
    mapping of the same physical bytes and is what every unicast kernel, the
    staging copies and the flag spin use; it is also what `regions[local_rank]`
    holds, so device/symmetric/ never learns that the region changed.
    """

    var mc: Int
    var uc: Int
    var size: Int
    var granularity: Int
    var mc_handle: UInt64
    var mem_handle: UInt64
    var peer_va: StaticTuple[Int, MAX_WORLD]
    var peer_handle: StaticTuple[UInt64, MAX_WORLD]

    def __init__(out self):
        self.mc = 0
        self.uc = 0
        self.size = 0
        self.granularity = 0
        self.mc_handle = 0
        self.mem_handle = 0
        self.peer_va = StaticTuple[Int, MAX_WORLD](fill=0)
        self.peer_handle = StaticTuple[UInt64, MAX_WORLD](fill=0)
