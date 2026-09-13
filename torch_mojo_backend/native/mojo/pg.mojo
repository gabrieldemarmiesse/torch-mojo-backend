"""Process-group core: NCCL / RCCL / mojoccl collectives on a per-device comm
stream.

torch's `MojoProcessGroup` (distributed/process_group.py) is a thin adapter:
it resolves data pointers, stages non-contiguous tensors with torch ops on
the comm stream and wraps a device-typed torch Future whose completion events
are recorded on the comm stream, so consumers on any stream wait exactly as
with ProcessGroupNCCL. Communicators, the comm stream and the library calls
live here. The three libraries share the NCCL C ABI; the 128-byte unique id
is passed the way the x86-64 SysV ABI lays out a by-value struct: 16 words
after the register arguments (see mojoccl.mojo's ncclCommInitRank; Python
refuses other architectures).

The entries are reached from Python outside the boxed adapter, so each one
holds the backend mutex over its whole body itself (`with Locked():`).
"""
from std.collections import Dict
from std.ffi import OwnedDLHandle, c_char, external_call
from std.memory.alloc import unsafe_alloc

from device import _add_stream, current_stream, dev, set_error, stream_ctx

comptime UID_BYTES = 128


struct Locked:
    """Holds the backend mutex (shim tmb_lock) for a `with` block.

    Only `with Locked():` takes the lock: constructing one and binding it to
    a local does not. Mojo destroys a value right after its LAST USE, so a
    `var _lock = Locked()` that nothing touches again is destroyed on the
    spot -- a lock-in-`__init__` / unlock-in-`__deinit__` guard would have
    unlocked before the body it was meant to protect ever ran (verified: an
    unused guard printed its destructor message before the body's).
    `__enter__` / `__exit__` are scoped to the block instead, and run on the
    exceptional exit too.
    """

    def __init__(out self):
        pass

    def __enter__(mut self):
        external_call["tmb_lock", NoneType]()

    def __exit__(mut self):
        external_call["tmb_unlock", NoneType]()


@fieldwise_init
struct Comm(Copyable, Movable):
    var handle: Int64
    var device: Int
    var stream: Int  # comm stream id on the device
    var raw: Int  # its vendor stream handle (what the library enqueues on)


comptime CollFn = def(Int, Int, Int, Int32, Int32, Int64, Int64) thin abi(
    "C"
) -> Int32
comptime ReduceFn = def(
    Int, Int, Int, Int32, Int32, Int32, Int64, Int64
) thin abi("C") -> Int32
comptime GatherFn = def(Int, Int, Int, Int32, Int64, Int64) thin abi(
    "C"
) -> Int32
comptime P2pFn = def(Int, Int, Int32, Int32, Int64, Int64) thin abi(
    "C"
) -> Int32
comptime GroupFn = def() thin abi("C") -> Int32
comptime CommFn = def(Int64) thin abi("C") -> Int32


def _symbol(lib: OwnedDLHandle, name: String) raises -> Int:
    var s = lib.get_symbol[NoneType](name)
    if not s:
        raise Error("collectives library lacks ", name)
    return Int(s.value())


@always_inline
def _fn[F: TrivialRegisterPassable](addr: Int) -> F:
    var a = addr
    return Pointer(to=a).unsafe_bitcast[F]()[]


struct PG(Movable):
    var lib: OwnedDLHandle
    var rank: Int
    var world: Int
    var comms: Dict[Int, Comm]
    # entry points resolved once (a dlsym per collective was measurable)
    var f_allreduce: Int
    var f_broadcast: Int
    var f_reduce: Int
    var f_allgather: Int
    var f_reduce_scatter: Int
    var f_send: Int
    var f_recv: Int
    var f_group_start: Int
    var f_group_end: Int

    def __init__(out self, path: String, rank: Int, world: Int) raises:
        self.lib = OwnedDLHandle(path)
        self.rank = rank
        self.world = world
        self.comms = Dict[Int, Comm]()
        self.f_allreduce = _symbol(self.lib, "ncclAllReduce")
        self.f_broadcast = _symbol(self.lib, "ncclBroadcast")
        self.f_reduce = _symbol(self.lib, "ncclReduce")
        self.f_allgather = _symbol(self.lib, "ncclAllGather")
        self.f_reduce_scatter = _symbol(self.lib, "ncclReduceScatter")
        self.f_send = _symbol(self.lib, "ncclSend")
        self.f_recv = _symbol(self.lib, "ncclRecv")
        self.f_group_start = _symbol(self.lib, "ncclGroupStart")
        self.f_group_end = _symbol(self.lib, "ncclGroupEnd")

    def symbol(self, name: String) raises -> Int:
        return _symbol(self.lib, name)

    def error_string(self, rc: Int32) -> String:
        try:
            var p = self.lib.get_function[Pointer[UInt8, MutUntrackedOrigin]](
                "ncclGetErrorString"
            )(rc)
            return String(unsafe_from_utf8_ptr=p)
        except e:
            return "error " + String(rc)

    def check(self, rc: Int32, what: StaticString) raises:
        if rc != 0:
            raise Error(what, ": ", self.error_string(rc))

    def comm(self, device: Int) raises -> Comm:
        var c = self.comms.find(device)
        if not c:
            raise Error(
                "no communicator on mojo:",
                device,
                " (init_device first, or it was aborted)",
            )
        return c.value().copy()

    def sync_in(self, c: Comm) raises:
        """The comm stream waits for everything the caller's current stream has
        enqueued so far (inputs are ready, outputs are no longer being read)."""
        var cur = current_stream(c.device)
        if cur != c.stream:
            stream_ctx(c.device, c.stream).enqueue_wait_for(
                stream_ctx(c.device, cur)
            )


comptime PGP = Pointer[PG, MutUntrackedOrigin]


@always_inline
def _pg(p: Int) -> PGP:
    return PGP(unsafe_from_address=p)


def tmb_pg_create(
    path: Pointer[c_char, MutUntrackedOrigin], rank: Int32, world: Int32
) abi("C") -> Int:
    with Locked():
        try:
            var box = unsafe_alloc[PG](1)
            box.unsafe_write(
                PG(
                    String(unsafe_from_utf8_ptr=path.unsafe_bitcast[UInt8]()),
                    Int(rank),
                    Int(world),
                )
            )
            return Int(box)
        except e:
            set_error(String(e))
            return 0


def tmb_pg_destroy(p: Int) abi("C"):
    if p == 0:
        return
    with Locked():
        var box = _pg(p)
        try:
            var f = _fn[CommFn](box[].symbol("ncclCommDestroy"))
            for item in box[].comms.items():
                _ = f(item.value.handle)
        except e:
            set_error(String(e))
        var moved = box.unsafe_take_pointee()
        box.unsafe_free()
        _ = moved^


def tmb_pg_version(p: Int) abi("C") -> Int32:
    with Locked():
        try:
            var v: Int32 = 0
            _pg(p)[].check(
                _pg(p)[].lib.get_function[Int32]("ncclGetVersion")(
                    Pointer(to=v)
                ),
                "ncclGetVersion",
            )
            return v
        except e:
            set_error(String(e))
            return -1


def tmb_pg_unique_id(
    p: Int, buf: Pointer[UInt8, MutUntrackedOrigin]
) abi("C") -> Int32:
    with Locked():
        try:
            _pg(p)[].check(
                _pg(p)[].lib.get_function[Int32]("ncclGetUniqueId")(buf),
                "ncclGetUniqueId",
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_init_device(
    p: Int, device: Int32, uid: Pointer[UInt8, MutUntrackedOrigin]
) abi("C") -> Int32:
    """Create the communicator of this rank on mojo:`device` (one per device)
    and its dedicated comm stream. Collective across ranks."""
    with Locked():
        try:
            var pg = _pg(p)
            if pg[].comms.find(Int(device)):
                raise Error("mojo:", device, " already has a communicator")
            var d = dev(Int(device))
            if d[].is_cpu:
                raise Error(
                    "the mojo process group needs an accelerator device"
                )
            var sid = _add_stream(d, 0)
            var raw = d[].raw[sid]
            if raw == 0:
                raise Error(
                    "no vendor stream handle for the comm stream (NCCL needs"
                    " one)"
                )
            var w = uid.unsafe_bitcast[UInt64]()
            var handle: Int64 = 0
            var rc: Int32
            with d[].ctx.push_context():  # ncclCommInitRank binds the communicator to the current device
                rc = pg[].lib.get_function[Int32]("ncclCommInitRank")(
                    Pointer(to=handle),
                    Int32(pg[].world),
                    Int32(pg[].rank),
                    Int64(0),
                    Int64(0),
                    Int64(0),
                    w[unsafe_offset=0],
                    w[unsafe_offset=1],
                    w[unsafe_offset=2],
                    w[unsafe_offset=3],
                    w[unsafe_offset=4],
                    w[unsafe_offset=5],
                    w[unsafe_offset=6],
                    w[unsafe_offset=7],
                    w[unsafe_offset=8],
                    w[unsafe_offset=9],
                    w[unsafe_offset=10],
                    w[unsafe_offset=11],
                    w[unsafe_offset=12],
                    w[unsafe_offset=13],
                    w[unsafe_offset=14],
                    w[unsafe_offset=15],
                )
            pg[].check(rc, "ncclCommInitRank")
            pg[].comms[Int(device)] = Comm(handle, Int(device), sid, raw)
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_comm_stream(p: Int, device: Int32) abi("C") -> Int64:
    with Locked():
        try:
            return Int64(_pg(p)[].comm(Int(device)).stream)
        except e:
            set_error(String(e))
            return -1


def tmb_pg_allreduce(
    p: Int, device: Int32, ptr: Int, count: Int, dtype: Int32, op: Int32
) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            pg[].check(
                _fn[CollFn](pg[].f_allreduce)(
                    ptr, ptr, count, dtype, op, c.handle, Int64(c.raw)
                ),
                "ncclAllReduce",
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_broadcast(
    p: Int, device: Int32, ptr: Int, count: Int, dtype: Int32, root: Int32
) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            pg[].check(
                _fn[CollFn](pg[].f_broadcast)(
                    ptr, ptr, count, dtype, root, c.handle, Int64(c.raw)
                ),
                "ncclBroadcast",
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_reduce(
    p: Int,
    device: Int32,
    ptr: Int,
    count: Int,
    dtype: Int32,
    op: Int32,
    root: Int32,
) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            pg[].check(
                _fn[ReduceFn](pg[].f_reduce)(
                    ptr, ptr, count, dtype, op, root, c.handle, Int64(c.raw)
                ),
                "ncclReduce",
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_allgather(
    p: Int, device: Int32, send: Int, recv: Int, count: Int, dtype: Int32
) abi("C") -> Int32:
    """recv[world * count] <- every rank's send[count]."""
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            pg[].check(
                _fn[GatherFn](pg[].f_allgather)(
                    send, recv, count, dtype, c.handle, Int64(c.raw)
                ),
                "ncclAllGather",
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_reduce_scatter(
    p: Int,
    device: Int32,
    send: Int,
    recv: Int,
    count: Int,
    dtype: Int32,
    op: Int32,
) abi("C") -> Int32:
    """recv[count] <- reduce of send[world * count] chunk `rank`."""
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            pg[].check(
                _fn[CollFn](pg[].f_reduce_scatter)(
                    send, recv, count, dtype, op, c.handle, Int64(c.raw)
                ),
                "ncclReduceScatter",
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_send(
    p: Int, device: Int32, ptr: Int, count: Int, dtype: Int32, peer: Int32
) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            pg[].check(
                _fn[P2pFn](pg[].f_send)(
                    ptr, count, dtype, peer, c.handle, Int64(c.raw)
                ),
                "ncclSend",
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_recv(
    p: Int, device: Int32, ptr: Int, count: Int, dtype: Int32, peer: Int32
) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            pg[].sync_in(c)
            pg[].check(
                _fn[P2pFn](pg[].f_recv)(
                    ptr, count, dtype, peer, c.handle, Int64(c.raw)
                ),
                "ncclRecv",
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_group_start(p: Int) abi("C") -> Int32:
    with Locked():
        try:
            _pg(p)[].check(
                _fn[GroupFn](_pg(p)[].f_group_start)(), "ncclGroupStart"
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_group_end(p: Int) abi("C") -> Int32:
    with Locked():
        try:
            _pg(p)[].check(_fn[GroupFn](_pg(p)[].f_group_end)(), "ncclGroupEnd")
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_async_error(p: Int, device: Int32) abi("C") -> Int32:
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            var err: Int32 = 0
            pg[].check(
                pg[].lib.get_function[Int32]("ncclCommGetAsyncError")(
                    c.handle, Pointer(to=err)
                ),
                "ncclCommGetAsyncError",
            )
            return err
        except e:
            set_error(String(e))
            return -1


def tmb_pg_abort(p: Int, device: Int32) abi("C") -> Int32:
    """Abort and forget the device's communicator (vendor NCCL destroys it)."""
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            _ = pg[].comms.pop(Int(device))
            pg[].check(
                _fn[CommFn](pg[].symbol("ncclCommAbort"))(c.handle),
                "ncclCommAbort",
            )
            return 0
        except e:
            set_error(String(e))
            return 1


def tmb_pg_synchronize_comm(p: Int, device: Int32) abi("C") -> Int32:
    """Host-wait for every collective issued so far on the device's comm stream.
    """
    with Locked():
        try:
            var pg = _pg(p)
            var c = pg[].comm(Int(device))
            stream_ctx(c.device, c.stream).synchronize()
            return 0
        except e:
            set_error(String(e))
            return 1


comptime PG_VTABLE_SLOTS = 18


def pg_vtable() -> Pointer[Int, MutUntrackedOrigin]:
    """Function addresses for the Python adapter, in this fixed order:
    0 create, 1 destroy, 2 version, 3 unique_id, 4 init_device, 5 comm_stream,
    6 allreduce, 7 broadcast, 8 reduce, 9 allgather, 10 reduce_scatter,
    11 send, 12 recv, 13 group_start, 14 group_end, 15 async_error, 16 abort,
    17 synchronize_comm."""
    var t = unsafe_alloc[Int](PG_VTABLE_SLOTS)
    var f0: def(Pointer[c_char, MutUntrackedOrigin], Int32, Int32) thin abi(
        "C"
    ) -> Int = tmb_pg_create
    var f1: def(Int) thin abi("C") -> None = tmb_pg_destroy
    var f2: def(Int) thin abi("C") -> Int32 = tmb_pg_version
    var f3: def(Int, Pointer[UInt8, MutUntrackedOrigin]) thin abi(
        "C"
    ) -> Int32 = tmb_pg_unique_id
    var f4: def(Int, Int32, Pointer[UInt8, MutUntrackedOrigin]) thin abi(
        "C"
    ) -> Int32 = tmb_pg_init_device
    var f5: def(Int, Int32) thin abi("C") -> Int64 = tmb_pg_comm_stream
    var f6: def(Int, Int32, Int, Int, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_allreduce
    var f7: def(Int, Int32, Int, Int, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_broadcast
    var f8: def(Int, Int32, Int, Int, Int32, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_reduce
    var f9: def(Int, Int32, Int, Int, Int, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_allgather
    var f10: def(Int, Int32, Int, Int, Int, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_reduce_scatter
    var f11: def(Int, Int32, Int, Int, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_send
    var f12: def(Int, Int32, Int, Int, Int32, Int32) thin abi(
        "C"
    ) -> Int32 = tmb_pg_recv
    var f13: def(Int) thin abi("C") -> Int32 = tmb_pg_group_start
    var f14: def(Int) thin abi("C") -> Int32 = tmb_pg_group_end
    var f15: def(Int, Int32) thin abi("C") -> Int32 = tmb_pg_async_error
    var f16: def(Int, Int32) thin abi("C") -> Int32 = tmb_pg_abort
    var f17: def(Int, Int32) thin abi("C") -> Int32 = tmb_pg_synchronize_comm
    t[unsafe_offset=0] = Pointer(to=f0).unsafe_bitcast[Int]()[]
    t[unsafe_offset=1] = Pointer(to=f1).unsafe_bitcast[Int]()[]
    t[unsafe_offset=2] = Pointer(to=f2).unsafe_bitcast[Int]()[]
    t[unsafe_offset=3] = Pointer(to=f3).unsafe_bitcast[Int]()[]
    t[unsafe_offset=4] = Pointer(to=f4).unsafe_bitcast[Int]()[]
    t[unsafe_offset=5] = Pointer(to=f5).unsafe_bitcast[Int]()[]
    t[unsafe_offset=6] = Pointer(to=f6).unsafe_bitcast[Int]()[]
    t[unsafe_offset=7] = Pointer(to=f7).unsafe_bitcast[Int]()[]
    t[unsafe_offset=8] = Pointer(to=f8).unsafe_bitcast[Int]()[]
    t[unsafe_offset=9] = Pointer(to=f9).unsafe_bitcast[Int]()[]
    t[unsafe_offset=10] = Pointer(to=f10).unsafe_bitcast[Int]()[]
    t[unsafe_offset=11] = Pointer(to=f11).unsafe_bitcast[Int]()[]
    t[unsafe_offset=12] = Pointer(to=f12).unsafe_bitcast[Int]()[]
    t[unsafe_offset=13] = Pointer(to=f13).unsafe_bitcast[Int]()[]
    t[unsafe_offset=14] = Pointer(to=f14).unsafe_bitcast[Int]()[]
    t[unsafe_offset=15] = Pointer(to=f15).unsafe_bitcast[Int]()[]
    t[unsafe_offset=16] = Pointer(to=f16).unsafe_bitcast[Int]()[]
    t[unsafe_offset=17] = Pointer(to=f17).unsafe_bitcast[Int]()[]
    return t
