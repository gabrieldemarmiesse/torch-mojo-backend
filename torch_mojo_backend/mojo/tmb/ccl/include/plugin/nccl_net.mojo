# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/include/plugin/nccl_net.h


# Largest number of NODES one communicator's inter-node transport addresses.
# Part of the wire layout (the verbs bootstrap blob carries one queue-pair
# number per node), so both transports and the engine share it.
comptime MAX_NODES = 16


# ===-------------------------------------------------------------------=== #
# What the progress engine sees of a completion
# ===-------------------------------------------------------------------=== #

# An inbound message carrying an immediate (a peer's shard, or a credit).
comptime NC_RECV = 0
# One of THIS rank's data writes finished: the NIC is done reading the source
# buffer, which is what lets the consumer kernel overwrite it.
comptime NC_SEND = 1
# The flush read came back (see `transport/net.mojo`'s header).
comptime NC_FLUSH = 2
# Something the engine does not account for -- a libfabric notification-send
# completion, say. Counted as progress and dropped.
comptime NC_OTHER = 3


struct NetCompletion(Copyable, ImplicitlyCopyable, Movable):
    """One completion, in the shape the engine understands, whichever
    transport produced it.

    The two transports fill it from very different sources -- ibverbs from a
    `struct ibv_wc` (peer from the completion's QP number, immediate from
    `imm_data`), libfabric from a `struct fi_cq_data_entry` (peer from the
    node index packed into the 64-bit remote CQ data, since the cxi provider
    does not implement FI_SOURCE) -- but the engine only ever reads these
    five fields.
    """

    var kind: Int
    var peer: Int  # index into `IbState.peers`, or -1 if it is not a peer's
    var imm: UInt32  # NC_RECV: the 32-bit immediate (credit bit + sequence)
    var wr_id: Int  # NC_SEND: the exchange number the write belonged to
    var status: Int  # 0 on success, else a transport-specific error code

    def __init__(out self):
        self.kind = NC_OTHER
        self.peer = -1
        self.imm = 0
        self.wr_id = 0
        self.status = 0
