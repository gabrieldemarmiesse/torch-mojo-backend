# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/misc/utils.cc
#
# Raw-memory plumbing shared by the two inter-node transports
# (`transport/net_ib/` and `transport/net_ofi.mojo`) and by the engine that sits on them
# (`transport/net.mojo`).
#
# Both libraries have the same shape of problem: `std.ffi` still has no
# C-struct ABI (MOCO-3692), so every C struct either library takes or returns
# is built and read in a raw byte buffer at hand-verified offsets, and both
# their data paths are `static inline` headers that dispatch through a
# function-pointer table hanging off the object (`ibv_context->ops` for
# verbs, `fid_ep->rma->writemsg` and friends for libfabric). The loads,
# stores and the "call this address as a C function" trick are identical, so
# they live here once.

from std.memory.alloc import unsafe_alloc
from std.ffi import external_call


@always_inline
def _alloc[T: AnyType](n: Int) -> Pointer[T, MutAnyOrigin]:
    """`unsafe_alloc` with the origin the FFI signatures here declare.

    Every buffer this module allocates is small, lives for the duration of
    one `ncclCommInitRank`, and is handed to libc by address; rebinding to
    `MutAnyOrigin` once here keeps the call sites free of casts.
    """
    return Pointer[T, MutAnyOrigin](unsafe_from_address=Int(unsafe_alloc[T](n)))


# ===-------------------------------------------------------------------=== #
# Host identity
# ===-------------------------------------------------------------------=== #


def host_hash() -> UInt64:
    """A 64-bit id of this physical host, equal on every rank of a node and
    different across nodes. /etc/machine-id first (stable, always present on
    these images), boot_id next, hostname last."""
    var text = String("")
    for p in ["/etc/machine-id", "/proc/sys/kernel/random/boot_id"]:
        try:
            with open(String(p), "r") as f:
                text = String(f.read().strip())
            if text.byte_length() > 0:
                break
        except:
            continue
    if text.byte_length() == 0:
        var buf = _alloc[UInt8](256)
        for i in range(256):
            buf[unsafe_offset=i] = 0
        _ = external_call["gethostname", Int32](buf, UInt64(255))
        var n = 0
        while n < 255 and buf[unsafe_offset=n] != 0:
            n += 1
        text = String(capacity_bytes=n)
        for i in range(n):
            text += chr(Int(buf[unsafe_offset=i]))
    # FNV-1a: any stable 64-bit mix is fine, this one needs no table.
    var h: UInt64 = 0xCBF29CE484222325
    for byte in text.as_bytes():
        h = (h ^ UInt64(byte)) * 0x100000001B3
    return h


@always_inline
def _any(p: Pointer[UInt8, MutUntrackedOrigin]) -> Pointer[UInt8, MutAnyOrigin]:
    """Rebinds an `unsafe_alloc`-returned pointer's origin to `MutAnyOrigin`,
    matching what misc/cudawrap.mojo/bootstrap.mojo (and the exported ABI functions
    they share signatures with) declare."""
    return Pointer[UInt8, MutAnyOrigin](unsafe_from_address=Int(p))


comptime P8 = Pointer[UInt8, MutAnyOrigin]


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
