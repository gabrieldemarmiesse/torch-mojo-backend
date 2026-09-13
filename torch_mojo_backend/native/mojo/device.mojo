"""Devices, streams, events and memory of the native backend.

One `Dev` per mojo index: the accelerators MAX enumerates, then the MAX CPU
device last (so `mojo:N` is the CPU on an N-GPU box, as before). A stream is a
MAX stream of the device's base context; kernels get the context *view* bound
to it (`DeviceContext.select_stream`), so real multi-stream execution costs
nothing at launch time. Everything here runs under the shim's mutex.
"""
from std.ffi import _get_global_or_null, c_char, external_call
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.atomic.atomic import Atomic
from std.time import perf_counter_ns

from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceEvent,
    DeviceStream,
    HostBuffer,
)

from vendor import Vendor, raw_stream

comptime BufP = Pointer[Buf, MutUntrackedOrigin]
comptime EvP = Pointer[Ev, MutUntrackedOrigin]
comptime StagingP = Pointer[Staging, MutUntrackedOrigin]
comptime U8P = Pointer[UInt8, MutUntrackedOrigin]
comptime POOL_STREAMS = 4


def _warn(what: StaticString, e: Error):
    print("torch-mojo-backend: ", what, ": ", String(e))


struct Dev(Movable):
    var ctx: DeviceContext
    var views: List[
        DeviceContext
    ]  # views[s] submits to stream s; views[0] is ctx
    var streams: List[DeviceStream]  # created streams, kept alive
    var raw: List[Int]  # vendor stream handle per stream (0 when unknown)
    var api: String
    var is_cpu: Bool
    var pool: List[Int]
    var pool_next: Int
    var staging: List[Int]  # pending pageable-H2D staging boxes (addresses)

    def __init__(out self, var ctx: DeviceContext, is_cpu: Bool) raises:
        self.api = ctx.api()
        self.is_cpu = is_cpu
        self.views = List[DeviceContext]()
        self.views.append(ctx)
        self.streams = List[DeviceStream]()
        self.raw = List[Int]()
        self.raw.append(
            0
        )  # filled once the vendor driver exists (init_backend)
        self.pool = List[Int]()
        self.pool_next = 0
        self.staging = List[Int]()
        self.ctx = ctx^

    def view(self, s: Int) raises -> DeviceContext:
        if s < 0 or s >= len(self.views):
            raise Error("invalid stream id ", s, " on mojo device ", self.api)
        return self.views[s]


@fieldwise_init
struct Backend(Movable):
    var devices: List[Dev]
    var vendor: Optional[Vendor]
    var n_accel: Int


comptime BACKEND_GLOBAL = "TMB_NATIVE_BACKEND"


@always_inline
def be() -> Pointer[Backend, MutUntrackedOrigin]:
    var p = _get_global_or_null(BACKEND_GLOBAL)
    return p.value().unsafe_bitcast[Backend]()


def dev(i: Int) raises -> Pointer[Dev, MutUntrackedOrigin]:
    ref devs = be()[].devices
    if i < 0 or i >= len(devs):
        raise Error("invalid mojo device index ", i)
    return Pointer(to=devs[i]).unsafe_origin_cast[MutUntrackedOrigin]()


def stream_ctx(device: Int, stream: Int) raises -> DeviceContext:
    return dev(device)[].view(stream)


def _probe_api() -> Tuple[String, Int]:
    """Which accelerator api MAX drives here and how many devices it has,
    asked at run time: this library is built once for every accelerator
    (the compile-time default api would tie it to the build machine's)."""
    var apis = List[String]()
    apis.append("cuda")
    apis.append("hip")
    apis.append("metal")
    for api in apis:
        var n = 0
        try:
            n = DeviceContext.number_of_devices(api=api)
        except e:
            n = 0  # MAX has no support for that api on this machine
        if n > 0:
            return (api, n)
    return (String("cpu"), 0)


def init_backend() raises -> Int:
    """Enumerate devices once; returns the mojo device count (GPUs + CPU)."""
    if _get_global_or_null(BACKEND_GLOBAL):
        return len(be()[].devices)
    var probed = _probe_api()
    var api = probed[0]
    var n = probed[1]
    var vendor: Optional[Vendor] = None
    if api == "cuda" or api == "hip":
        try:
            vendor = Vendor(api)
        except e:
            _warn(
                "vendor driver unavailable, events fall back to host timing", e
            )
    var devs = List[Dev]()
    for i in range(n):
        devs.append(Dev(DeviceContext(i, api=api), False))
    devs.append(Dev(DeviceContext(api="cpu"), True))
    var box = unsafe_alloc[Backend](1)
    box.unsafe_write(Backend(devs^, vendor^, n))
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(BACKEND_GLOBAL), box.unsafe_bitcast[NoneType]()
    )
    if be()[].vendor:
        for i in range(n):
            var d = dev(i)
            try:
                d[].raw[0] = raw_stream(be()[].vendor.value(), d[].ctx)
            except e:
                _warn("no raw stream handle", e)
    return n + 1


# --- memory -------------------------------------------------------------------


@fieldwise_init
struct Buf(Movable):
    var buf: DeviceBuffer[DType.uint8]
    var device: Int
    var stream: Int  # owner stream: where MAX orders the release
    var users: List[Int]  # other streams that used the buffer (record_stream)


def _create(
    ctx: DeviceContext, nbytes: Int
) raises -> DeviceBuffer[DType.uint8]:
    return ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))


def _create_retry(
    d: Pointer[Dev, MutUntrackedOrigin], ctx: DeviceContext, nbytes: Int
) raises -> DeviceBuffer[DType.uint8]:
    """MAX's allocator can fail transiently on a full arena (modular#6801):
    drain the device once and retry before reporting out-of-memory."""
    try:
        return _create(ctx, nbytes)
    except e:
        for i in range(len(d[].views)):
            d[].views[i].synchronize()
        return _create(ctx, nbytes)


def h_alloc(
    nbytes: Int,
    device: Int32,
    stream: Int64,
    data: Pointer[Int, MutUntrackedOrigin],
) abi("C") -> Int:
    try:
        var d = dev(Int(device))
        var ctx = d[].view(Int(stream))
        var buf = _create_retry(d, ctx, nbytes)
        data[] = Int(buf.unsafe_ptr())
        var box = unsafe_alloc[Buf](1)
        box.unsafe_write(Buf(buf^, Int(device), Int(stream), List[Int]()))
        return Int(box)
    except e:
        set_error(String(e))
        return 0


def h_free(handle: Int) abi("C"):
    if handle == 0:
        return
    var box = BufP(unsafe_from_address=handle)
    # MAX releases the block stream-ordered on the OWNER stream only. Every
    # other stream that used it (torch.Tensor.record_stream) gets fenced now,
    # at release time: the owner waits for all of that stream's work so far.
    if len(box[].users) > 0:
        try:
            var d = dev(box[].device)
            var owner = d[].view(box[].stream)
            for i in range(len(box[].users)):
                owner.enqueue_wait_for(d[].view(box[].users[i]))
        except e:
            # Ordering could not be established: leak the block rather than
            # let MAX reuse memory another stream may still be reading.
            set_error(String(e))
            return
    var moved = box.unsafe_take_pointee()
    box.unsafe_free()
    _ = moved^


def h_copy_data(
    dst: Int, src: Int, nbytes: Int, device: Int32, stream: Int64
) abi("C"):
    try:
        copy_d2d(stream_ctx(Int(device), Int(stream)), dst, src, nbytes)
    except e:
        set_error(String(e))


def h_device_of_ptr(ptr: Int) abi("C") -> Int32:
    return -1


def h_record_stream(handle: Int, device: Int32, stream: Int64) abi("C"):
    """torch.Tensor.record_stream: remember that `stream` uses the buffer; the
    fence happens when the buffer is released (h_free), covering every use
    enqueued on that stream up to then, as CUDA's caching allocator does."""
    if handle == 0:
        return
    var box = BufP(unsafe_from_address=handle)
    if Int(device) != box[].device:
        set_error("record_stream: stream of another device")
        return
    if Int(stream) == box[].stream:
        return
    for i in range(len(box[].users)):
        if box[].users[i] == Int(stream):
            return
    box[].users.append(Int(stream))


def wrap_raw(
    ctx: DeviceContext, addr: Int, nbytes: Int
) -> DeviceBuffer[DType.uint8]:
    return DeviceBuffer[DType.uint8](
        ctx, U8P(unsafe_from_address=addr), nbytes, owning=False
    )


def copy_d2d(ctx: DeviceContext, dst: Int, src: Int, nbytes: Int) raises:
    """Stream-ordered device copy. The CPU device runs copies on a worker pool
    that is not ordered with kernel execution, so there it completes first."""
    if nbytes == 0:
        return
    var d = wrap_raw(ctx, dst, nbytes)
    var s = wrap_raw(ctx, src, nbytes)
    d.enqueue_copy_from(s)
    if ctx.api() == "cpu":
        ctx.synchronize()


def copy_to_host(
    ctx: DeviceContext, dev_ptr: Int, host_ptr: Int, nbytes: Int
) raises:
    """Blocking D2H."""
    if nbytes == 0:
        return
    var s = wrap_raw(ctx, dev_ptr, nbytes)
    s.enqueue_copy_to(U8P(unsafe_from_address=host_ptr))
    ctx.synchronize()


@fieldwise_init
struct Staging(Movable):
    var buf: HostBuffer[DType.uint8]
    var done: Atomic[DType.int32]  # set from the driver's callback thread


def _staging_done(p: Pointer[NoneType, MutAnyOrigin]):
    p.unsafe_bitcast[Staging]()[].done.store(1)


def _drain_staging(d: Pointer[Dev, MutUntrackedOrigin]):
    var keep = List[Int]()
    for i in range(len(d[].staging)):
        var box = StagingP(unsafe_from_address=d[].staging[i])
        if box[].done.load() != 0:
            var moved = box.unsafe_take_pointee()
            box.unsafe_free()
            _ = moved^
        else:
            keep.append(d[].staging[i])
    d[].staging = keep^


def copy_from_host(
    device: Int, ctx: DeviceContext, dev_ptr: Int, host_ptr: Int, nbytes: Int
) raises:
    """H2D from pageable host memory: on GPUs, stage through a MAX pinned
    buffer (a synchronous memcpy, then an asynchronous DMA) and release the
    staging buffer once a host callback reports the copy done. The CPU
    device copies synchronously."""
    if nbytes == 0:
        return
    var dst = wrap_raw(ctx, dev_ptr, nbytes)
    if ctx.api() == "cpu":
        dst.enqueue_copy_from(U8P(unsafe_from_address=host_ptr))
        ctx.synchronize()
        return
    var d = dev(device)
    _drain_staging(d)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    unsafe_memcpy(
        dest=host.unsafe_ptr(),
        src=U8P(unsafe_from_address=host_ptr),
        count=nbytes,
    )
    dst.enqueue_copy_from(host)
    var box = unsafe_alloc[Staging](1)
    box.unsafe_write(Staging(host^, Atomic[DType.int32](0)))
    ctx.stream().enqueue_host_func(
        _staging_done,
        box.unsafe_bitcast[NoneType]().unsafe_origin_cast[MutAnyOrigin](),
    )
    d[].staging.append(Int(box))


def read_bytes_sync(
    ctx: DeviceContext, dev_ptr: Int, dst_addr: Int, nbytes: Int
) raises:
    """Blocking readback of a few bytes (the .item() primitive)."""
    var s = wrap_raw(ctx, dev_ptr, nbytes)
    s.enqueue_copy_to(U8P(unsafe_from_address=dst_addr))
    ctx.synchronize()


def memset_bytes(
    ctx: DeviceContext, dev_ptr: Int, value: UInt8, nbytes: Int
) raises:
    if nbytes == 0:
        return
    var d = wrap_raw(ctx, dev_ptr, nbytes)
    ctx.enqueue_memset(d, value)


def memset_typed[
    dt: DType
](ctx: DeviceContext, dev_ptr: Int, value: Scalar[dt], count: Int) raises:
    if count == 0:
        return
    var d = DeviceBuffer[dt](
        ctx,
        Pointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=dev_ptr),
        count,
        owning=False,
    )
    ctx.enqueue_memset(d, value)


# --- devices and streams -----------------------------------------------------


def h_device_count() abi("C") -> Int32:
    return Int32(len(be()[].devices))


def h_synchronize_device(device: Int32) abi("C"):
    try:
        var d = dev(Int(device))
        for i in range(len(d[].views)):
            d[].views[i].synchronize()
        _drain_staging(d)
    except e:
        set_error(String(e))


def _add_stream(
    d: Pointer[Dev, MutUntrackedOrigin], priority: Int
) raises -> Int:
    var s = d[].ctx.create_stream(priority=priority)
    var idx = d[].ctx.num_streams() - 1
    var view = d[].ctx.select_stream(idx)
    var r = 0
    if be()[].vendor:
        try:
            r = raw_stream(be()[].vendor.value(), view)
        except e:
            _warn("no raw stream handle", e)
    d[].streams.append(s^)
    d[].views.append(view^)
    d[].raw.append(r)
    return idx


def h_new_stream(device: Int32, priority: Int32) abi("C") -> Int64:
    try:
        var d = dev(Int(device))
        if d[].is_cpu:
            return 0
        return Int64(_add_stream(d, Int(priority)))
    except e:
        set_error(String(e))
        return 0


def h_stream_from_pool(device: Int32, high_priority: Int32) abi("C") -> Int64:
    try:
        var d = dev(Int(device))
        if d[].is_cpu:
            return 0
        if len(d[].pool) < POOL_STREAMS:
            d[].pool.append(_add_stream(d, 0))
            return Int64(d[].pool[len(d[].pool) - 1])
        var s = d[].pool[d[].pool_next]
        d[].pool_next = (d[].pool_next + 1) % POOL_STREAMS
        return Int64(s)
    except e:
        set_error(String(e))
        return 0


def h_synchronize_stream(device: Int32, stream: Int64) abi("C"):
    try:
        stream_ctx(Int(device), Int(stream)).synchronize()
    except e:
        set_error(String(e))


def h_query_stream(device: Int32, stream: Int64) abi("C") -> Int32:
    try:
        var d = dev(Int(device))
        var r = d[].raw[Int(stream)] if Int(stream) < len(d[].raw) else 0
        if r != 0 and be()[].vendor:
            return 1 if be()[].vendor.value().stream_query(r) else 0
        d[].view(Int(stream)).synchronize()
        return 1
    except e:
        set_error(String(e))
        return 1


def h_stream_native_handle(device: Int32, stream: Int64) abi("C") -> Int:
    try:
        var d = dev(Int(device))
        if Int(stream) < len(d[].raw):
            return d[].raw[Int(stream)]
    except e:
        set_error(String(e))
    return 0


# --- events ---------------------------------------------------------------------


@fieldwise_init
struct Ev(Movable):
    var device: Int
    var timing: Bool
    var recorded: Bool
    var max_ev: DeviceEvent
    var raw: Int
    var host_ns: Int


def h_event_create(device: Int32, enable_timing: Int32) abi("C") -> Int:
    try:
        var d = dev(Int(device))
        var raw = 0
        if be()[].vendor and d[].raw[0] != 0:
            raw = (
                be()[].vendor.value().event_create(d[].ctx, enable_timing != 0)
            )
        var box = unsafe_alloc[Ev](1)
        box.unsafe_write(
            Ev(
                Int(device),
                enable_timing != 0,
                False,
                d[].ctx.create_event(),
                raw,
                0,
            )
        )
        return Int(box)
    except e:
        set_error(String(e))
        return 0


def h_event_destroy(ev: Int, device: Int32) abi("C"):
    if ev == 0:
        return
    var box = EvP(unsafe_from_address=ev)
    if box[].raw != 0 and be()[].vendor:
        try:
            be()[].vendor.value().event_destroy(box[].raw)
        except e:
            _warn("event destroy", e)
    var moved = box.unsafe_take_pointee()
    box.unsafe_free()
    _ = moved^


def h_event_record(ev: Int, device: Int32, stream: Int64) abi("C"):
    try:
        var box = EvP(unsafe_from_address=ev)
        var d = dev(Int(device))
        var ctx = d[].view(Int(stream))
        if box[].raw != 0:
            be()[].vendor.value().event_record(box[].raw, d[].raw[Int(stream)])
        ctx.stream().record_event(
            box[].max_ev
        )  # after the vendor event: waiting on it covers both
        box[].host_ns = perf_counter_ns()
        box[].recorded = True
    except e:
        set_error(String(e))


def h_event_block(ev: Int, device: Int32, stream: Int64) abi("C"):
    try:
        var box = EvP(unsafe_from_address=ev)
        if not box[].recorded:
            return
        stream_ctx(Int(device), Int(stream)).stream().enqueue_wait_for(
            box[].max_ev
        )
    except e:
        set_error(String(e))


def h_event_query(ev: Int) abi("C") -> Int32:
    try:
        var box = EvP(unsafe_from_address=ev)
        if not box[].recorded:
            return 1
        if box[].raw != 0:
            return 1 if be()[].vendor.value().event_query(box[].raw) else 0
        box[].max_ev.synchronize()
        return 1
    except e:
        set_error(String(e))
        return 1


def h_event_synchronize(ev: Int) abi("C"):
    try:
        var box = EvP(unsafe_from_address=ev)
        if box[].recorded:
            box[].max_ev.synchronize()
            if box[].raw != 0:
                be()[].vendor.value().event_synchronize(box[].raw)
    except e:
        set_error(String(e))


def h_event_elapsed_ms(start: Int, end: Int) abi("C") -> Float64:
    try:
        var a = EvP(unsafe_from_address=start)
        var b = EvP(unsafe_from_address=end)
        if not a[].timing or not b[].timing:
            raise Error(
                "elapsed_time needs events created with enable_timing=True"
            )
        if a[].raw != 0 and b[].raw != 0:
            return be()[].vendor.value().event_elapsed_ms(a[].raw, b[].raw)
        if not dev(a[].device)[].is_cpu:
            raise Error(
                "elapsed_time: device timing needs the vendor driver (not"
                " available on this device)"
            )
        a[].max_ev.synchronize()
        b[].max_ev.synchronize()
        return Float64(b[].host_ns - a[].host_ns) / 1.0e6
    except e:
        set_error(String(e))
        return 0.0


# --- the table --------------------------------------------------------------


def set_error(msg: String):
    var tmp = String(msg)
    external_call["tmb_set_error", NoneType](
        tmp.as_c_string_slice().unsafe_ptr()
    )


def hooks_table() -> Pointer[Int, MutUntrackedOrigin]:
    """The TmbBackendHooks struct (tmb.h): a u32 size then 23 function pointers.
    """
    comptime N = 24
    var t = unsafe_alloc[Int](N)
    for i in range(N):
        t[unsafe_offset=i] = 0
    t[unsafe_offset=0] = N * 8
    var f_alloc: def(
        Int, Int32, Int64, Pointer[Int, MutUntrackedOrigin]
    ) thin abi("C") -> Int = h_alloc
    var f_free: def(Int) thin abi("C") -> None = h_free
    var f_copy: def(Int, Int, Int, Int32, Int64) thin abi(
        "C"
    ) -> None = h_copy_data
    var f_devof: def(Int) thin abi("C") -> Int32 = h_device_of_ptr
    var f_rec: def(Int, Int32, Int64) thin abi("C") -> None = h_record_stream
    var f_count: def() thin abi("C") -> Int32 = h_device_count
    var f_syncdev: def(Int32) thin abi("C") -> None = h_synchronize_device
    var f_newstream: def(Int32, Int32) thin abi("C") -> Int64 = h_new_stream
    var f_pool: def(Int32, Int32) thin abi("C") -> Int64 = h_stream_from_pool
    var f_syncstream: def(Int32, Int64) thin abi(
        "C"
    ) -> None = h_synchronize_stream
    var f_qstream: def(Int32, Int64) thin abi("C") -> Int32 = h_query_stream
    var f_native: def(Int32, Int64) thin abi(
        "C"
    ) -> Int = h_stream_native_handle
    var f_evc: def(Int32, Int32) thin abi("C") -> Int = h_event_create
    var f_evd: def(Int, Int32) thin abi("C") -> None = h_event_destroy
    var f_evr: def(Int, Int32, Int64) thin abi("C") -> None = h_event_record
    var f_evb: def(Int, Int32, Int64) thin abi("C") -> None = h_event_block
    var f_evq: def(Int) thin abi("C") -> Int32 = h_event_query
    var f_evs: def(Int) thin abi("C") -> None = h_event_synchronize
    var f_eve: def(Int, Int) thin abi("C") -> Float64 = h_event_elapsed_ms
    t[unsafe_offset=1] = Pointer(to=f_alloc).unsafe_bitcast[Int]()[]
    t[unsafe_offset=2] = Pointer(to=f_free).unsafe_bitcast[Int]()[]
    t[unsafe_offset=3] = Pointer(to=f_copy).unsafe_bitcast[Int]()[]
    t[unsafe_offset=4] = Pointer(to=f_devof).unsafe_bitcast[Int]()[]
    t[unsafe_offset=5] = Pointer(to=f_rec).unsafe_bitcast[Int]()[]
    t[unsafe_offset=6] = Pointer(to=f_count).unsafe_bitcast[Int]()[]
    t[unsafe_offset=7] = Pointer(to=f_syncdev).unsafe_bitcast[Int]()[]
    t[unsafe_offset=8] = Pointer(to=f_newstream).unsafe_bitcast[Int]()[]
    t[unsafe_offset=9] = Pointer(to=f_pool).unsafe_bitcast[Int]()[]
    t[unsafe_offset=10] = Pointer(to=f_syncstream).unsafe_bitcast[Int]()[]
    t[unsafe_offset=11] = Pointer(to=f_qstream).unsafe_bitcast[Int]()[]
    t[unsafe_offset=12] = Pointer(to=f_native).unsafe_bitcast[Int]()[]
    t[unsafe_offset=13] = Pointer(to=f_evc).unsafe_bitcast[Int]()[]
    t[unsafe_offset=14] = Pointer(to=f_evd).unsafe_bitcast[Int]()[]
    t[unsafe_offset=15] = Pointer(to=f_evr).unsafe_bitcast[Int]()[]
    t[unsafe_offset=16] = Pointer(to=f_evb).unsafe_bitcast[Int]()[]
    t[unsafe_offset=17] = Pointer(to=f_evq).unsafe_bitcast[Int]()[]
    t[unsafe_offset=18] = Pointer(to=f_evs).unsafe_bitcast[Int]()[]
    t[unsafe_offset=19] = Pointer(to=f_eve).unsafe_bitcast[Int]()[]
    # 20..22: prof_mark / prof_range_push / prof_range_pop stay NULL for now
    return t


# --- what ops need ------------------------------------------------------------


def current_device() -> Int:
    return Int(external_call["tmb_current_device", Int32]())


def current_stream(device: Int) -> Int:
    return Int(external_call["tmb_current_stream", Int64](Int32(device)))


def ctx_for(device: Int) raises -> DeviceContext:
    """The context view an op launches on: the device's current stream."""
    return stream_ctx(device, current_stream(device))


def ctx_ptr(ctx: DeviceContext) -> Int:
    """The C++ DeviceContext handle kernels rebuild a `DeviceContext` from
    (TensorSpec.ctx_ptr / the trailing ctx slot): the same pointer
    `Device._device_context_ptr()` handed the old Python path."""
    return Int(ctx._handle.value())


def record_stream(handle: Int, device: Int, stream: Int) raises:
    """Op-side twin of h_record_stream."""
    var box = BufP(unsafe_from_address=handle)
    if device != box[].device:
        raise Error("record_stream: stream of another device")
    if stream == box[].stream:
        return
    for i in range(len(box[].users)):
        if box[].users[i] == stream:
            return
    box[].users.append(stream)
