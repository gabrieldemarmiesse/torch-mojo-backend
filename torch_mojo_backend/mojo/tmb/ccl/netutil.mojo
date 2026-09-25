# Raw-memory plumbing shared by the two inter-node transports
# (`ibverbs.mojo` and `libfabric.mojo`) and by the engine that sits on them
# (`internode.mojo`).
#
# Both libraries have the same shape of problem: `std.ffi` still has no
# C-struct ABI (MOCO-3692), so every C struct either library takes or returns
# is built and read in a raw byte buffer at hand-verified offsets, and both
# their data paths are `static inline` headers that dispatch through a
# function-pointer table hanging off the object (`ibv_context->ops` for
# verbs, `fid_ep->rma->writemsg` and friends for libfabric). The loads,
# stores and the "call this address as a C function" trick are identical, so
# they live here once.

from std.ffi import external_call
from std.memory.alloc import unsafe_alloc

comptime P8 = Pointer[UInt8, MutAnyOrigin]

# Largest number of NODES one communicator's inter-node transport addresses.
# Part of the wire layout (the verbs bootstrap blob carries one queue-pair
# number per node), so both transports and the engine share it.
comptime MAX_NODES = 16


@always_inline
def alloc_bytes(n: Int) -> P8:
    var p = P8(unsafe_from_address=Int(unsafe_alloc[UInt8](n)))
    for i in range(n):
        p[unsafe_offset=i] = 0
    return p


@always_inline
def free_bytes(p: P8):
    """Release a buffer from `alloc_bytes`."""
    p.unsafe_free()


@always_inline
def st8(p: P8, off: Int, v: UInt8):
    p[unsafe_offset=off] = v


@always_inline
def st16(p: P8, off: Int, v: UInt16):
    p.unsafe_bitcast[UInt16]()[unsafe_offset=off // 2] = v


@always_inline
def st32(p: P8, off: Int, v: Int32):
    p.unsafe_bitcast[Int32]()[unsafe_offset=off // 4] = v


@always_inline
def stu32(p: P8, off: Int, v: UInt32):
    p.unsafe_bitcast[UInt32]()[unsafe_offset=off // 4] = v


@always_inline
def st64(p: P8, off: Int, v: Int):
    p.unsafe_bitcast[Int64]()[unsafe_offset=off // 8] = Int64(v)


@always_inline
def stu64(p: P8, off: Int, v: UInt64):
    p.unsafe_bitcast[UInt64]()[unsafe_offset=off // 8] = v


@always_inline
def ld8(p: P8, off: Int) -> Int:
    return Int(p[unsafe_offset=off])


@always_inline
def ld16(p: P8, off: Int) -> Int:
    return Int(p.unsafe_bitcast[UInt16]()[unsafe_offset=off // 2])


@always_inline
def ld32(p: P8, off: Int) -> Int:
    return Int(p.unsafe_bitcast[Int32]()[unsafe_offset=off // 4])


@always_inline
def ldu32(p: P8, off: Int) -> UInt32:
    return p.unsafe_bitcast[UInt32]()[unsafe_offset=off // 4]


@always_inline
def ld64(p: P8, off: Int) -> Int:
    return Int(p.unsafe_bitcast[Int64]()[unsafe_offset=off // 8])


@always_inline
def ldu64(p: P8, off: Int) -> UInt64:
    return p.unsafe_bitcast[UInt64]()[unsafe_offset=off // 8]


@always_inline
def as_fn[F: TrivialRegisterPassable](addr: Int) -> F:
    """A callable from a raw function address.

    Same shape as `std.ffi._get_dylib_function`'s cache hit: bitcast a
    pointer TO the address variable, then load. A `Pointer.unsafe_bitcast`
    straight to a function type is rejected (function types are not
    `AnyType`), and the type must be `thin` -- a plain `def(...)` type is a
    closure trait, not a function pointer.
    """
    var a = addr
    return Pointer(to=a).unsafe_bitcast[F]()[]


def c_string(var s: String) -> P8:
    """A NUL-terminated copy of `s` in freshly allocated memory.

    The buffer outlives every call it is passed to (it is never freed); the
    few dozen bytes per communicator this costs are not worth a lifetime.
    """
    var b = s.as_bytes()
    var p = alloc_bytes(len(b) + 1)
    for i in range(len(b)):
        p[unsafe_offset=i] = b[i]
    return p


def read_c_string(addr: Int, cap: Int) -> String:
    """The NUL-terminated C string at `addr` (empty if the pointer is null),
    reading at most `cap` bytes."""
    if addr == 0:
        return String("")
    var p = P8(unsafe_from_address=addr)
    var s = String("")
    var i = 0
    while i < cap and p[unsafe_offset=i] != 0:
        s += chr(Int(p[unsafe_offset=i]))
        i += 1
    return s^


# ===-------------------------------------------------------------------=== #
# What the progress engine sees of a completion
# ===-------------------------------------------------------------------=== #

# An inbound message carrying an immediate (a peer's shard, or a credit).
comptime NC_RECV = 0
# One of THIS rank's data writes finished: the NIC is done reading the source
# buffer, which is what lets the consumer kernel overwrite it.
comptime NC_SEND = 1
# The flush read came back (see `internode.mojo`'s header).
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


# ===-------------------------------------------------------------------=== #
# Picking the NIC nearest this rank's GPU
# ===-------------------------------------------------------------------=== #


def realpath(path: String) -> String:
    """realpath(3): the ABSOLUTE /sys/devices path behind a sysfs symlink.

    `readlink` would return the stored relative target (`../../..0000:18:00.0`),
    and two of those share no comparable prefix -- the whole point here is to
    compare where a GPU and a NIC sit in one PCI tree.
    """
    var buf = alloc_bytes(4096)
    var rc = Int(external_call["realpath", Int64](c_string(String(path)), buf))
    if rc == 0:
        return String("")
    return read_c_string(Int(buf), 4095)


def common_prefix(a: String, b: String) -> Int:
    var ab = a.as_bytes()
    var bb = b.as_bytes()
    var n = min(len(ab), len(bb))
    var i = 0
    while i < n and ab[i] == bb[i]:
        i += 1
    return i


def pci_pick(
    device_paths: List[String], gpu_bdf: String, local_rank: Int
) raises -> Int:
    """Index of the NIC this rank should use, by PCI proximity to its GPU.

    Both the GPU and the NIC are PCI devices, so the longer the shared prefix
    of their /sys/devices paths, the fewer switch hops between them -- the
    cheap version of what NCCL's topology detection measures. Ranks that tie
    (the common case: one PCI switch serving a pair of GPUs) fall back to
    `local_rank % n` over the tied set, which on these nodes already hands
    every rank its own NIC. `device_paths[i]` is the sysfs `device` symlink
    of NIC i (`/sys/class/infiniband/<hca>/device`, `/sys/class/cxi/<cxiN>/device`).
    """
    if len(device_paths) == 0:
        raise Error("mojoccl: no network device to choose from")
    if len(device_paths) == 1 or gpu_bdf.byte_length() == 0:
        return local_rank % len(device_paths)
    var gpu_path = realpath("/sys/bus/pci/devices/" + gpu_bdf)
    if gpu_path.byte_length() == 0:
        return local_rank % len(device_paths)
    var best_score = -1
    var ties = 0
    for i in range(len(device_paths)):
        var score = common_prefix(gpu_path, realpath(device_paths[i]))
        if score > best_score:
            best_score = score
            ties = 1
        elif score == best_score:
            ties += 1
    var chosen = List[Int]()
    for i in range(len(device_paths)):
        if common_prefix(gpu_path, realpath(device_paths[i])) == best_score:
            chosen.append(i)
    if len(chosen) == 0:
        return local_rank % len(device_paths)
    return chosen[local_rank % len(chosen)]
