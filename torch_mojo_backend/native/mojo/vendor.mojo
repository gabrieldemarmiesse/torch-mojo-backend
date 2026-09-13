"""Vendor driver calls on MAX's raw streams.

MAX's DeviceEvent can be recorded, waited on and synchronized but not queried
or timed, so the two things torch.Event needs beyond ordering come from the
CUDA / HIP driver on the CUstream / hipStream_t behind each MAX stream. Metal
and the CPU device have no driver here; device.mojo answers those from host
clocks instead.
"""
from std.ffi import OwnedDLHandle, external_call

from max.gpu.host import DeviceContext

comptime NOT_READY = Int32(600)  # CUDA_ERROR_NOT_READY == hipErrorNotReady
comptime DISABLE_TIMING = UInt32(
    2
)  # CU_EVENT_DISABLE_TIMING == hipEventDisableTiming


@fieldwise_init
struct DriverNames(Copyable, Movable):
    """The vendor driver library and its entry points for one device api,
    chosen at run time from `DeviceContext.api()` so that one build of this
    library serves NVIDIA and AMD machines (and Metal, which has no driver
    here: device.mojo answers those from host clocks)."""

    var lib: String
    var event_create: String
    var event_record: String
    var event_query: String
    var event_sync: String
    var event_elapsed: String
    var event_destroy: String
    var stream_wait_event: String
    var stream_query: String
    var raw_stream: String


def driver_names(api: String) raises -> DriverNames:
    if api == "cuda":
        return DriverNames(
            "libcuda.so.1",
            "cuEventCreate",
            "cuEventRecord",
            "cuEventQuery",
            "cuEventSynchronize",
            "cuEventElapsedTime",
            "cuEventDestroy_v2",
            "cuStreamWaitEvent",
            "cuStreamQuery",
            "AsyncRT_DeviceStream_cuda_stream",
        )
    if api == "hip":
        return DriverNames(
            "libamdhip64.so",
            "hipEventCreateWithFlags",
            "hipEventRecord",
            "hipEventQuery",
            "hipEventSynchronize",
            "hipEventElapsedTime",
            "hipEventDestroy",
            "hipStreamWaitEvent",
            "hipStreamQuery",
            "AsyncRT_DeviceStream_hip_stream",
        )
    raise Error("no vendor driver for the ", api, " device api")


struct Vendor(Movable):
    var lib: OwnedDLHandle
    var asyncrt: OwnedDLHandle  # the process: MAX's AsyncRT is already loaded
    var names: DriverNames

    def __init__(out self, api: String) raises:
        self.names = driver_names(api)
        self.lib = OwnedDLHandle(self.names.lib)
        self.asyncrt = OwnedDLHandle()

    def _check(self, rc: Int32, what: String) raises:
        if rc != 0:
            raise Error(what, " failed with driver error ", rc)

    def event_create(self, ctx: DeviceContext, timing: Bool) raises -> Int:
        var ev: Int = 0
        with ctx.push_context():  # cuEventCreate needs the device's context current
            self._check(
                self.lib.get_function[Int32](self.names.event_create)(
                    Pointer(to=ev), UInt32(0) if timing else DISABLE_TIMING
                ),
                self.names.event_create,
            )
        return ev

    def event_record(self, ev: Int, raw_stream: Int) raises:
        self._check(
            self.lib.get_function[Int32](self.names.event_record)(
                ev, raw_stream
            ),
            self.names.event_record,
        )

    def event_query(self, ev: Int) raises -> Bool:
        var rc = self.lib.get_function[Int32](self.names.event_query)(ev)
        if rc == 0:
            return True
        if rc == NOT_READY:
            return False
        self._check(rc, self.names.event_query)
        return True

    def event_synchronize(self, ev: Int) raises:
        self._check(
            self.lib.get_function[Int32](self.names.event_sync)(ev),
            self.names.event_sync,
        )

    def event_elapsed_ms(self, start: Int, end: Int) raises -> Float64:
        var ms: Float32 = 0
        self._check(
            self.lib.get_function[Int32](self.names.event_elapsed)(
                Pointer(to=ms), start, end
            ),
            self.names.event_elapsed,
        )
        return Float64(ms)

    def event_destroy(self, ev: Int) raises:
        _ = self.lib.get_function[Int32](self.names.event_destroy)(ev)

    def stream_wait_event(self, raw_stream: Int, ev: Int) raises:
        self._check(
            self.lib.get_function[Int32](self.names.stream_wait_event)(
                raw_stream, ev, UInt32(0)
            ),
            self.names.stream_wait_event,
        )

    def stream_query(self, raw_stream: Int) raises -> Bool:
        var rc = self.lib.get_function[Int32](self.names.stream_query)(
            raw_stream
        )
        if rc == 0:
            return True
        if rc == NOT_READY:
            return False
        self._check(rc, self.names.stream_query)
        return True


def raw_stream(vendor: Vendor, ctx: DeviceContext) raises -> Int:
    """The CUstream / hipStream_t behind a context view's stream (an AsyncRT
    entry of MAX's own runtime library, resolved by name at run time)."""
    var raw: Int = 0
    var err = vendor.asyncrt.get_function[OpaquePointer[MutUntrackedOrigin]](
        vendor.names.raw_stream
    )(Pointer(to=raw), ctx.stream()._handle)
    if Int(err) != 0:
        raise Error("raw stream handle unavailable")
    return raw
