# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/misc/strongstream.cc

from max.gpu.host import DeviceContext
from std.ffi import OwnedDLHandle

from tmb.ccl.misc.cudawrap import AMD, _check, open_driver


comptime FN_LAUNCH_HOST_FUNC = (
    "hipLaunchHostFunc" if AMD else "cuLaunchHostFunc"
)


comptime FN_STREAM_QUERY = "hipStreamQuery" if AMD else "cuStreamQuery"


struct CompletionEvent(Movable):
    """Owned ordering event; MAX's DeviceEvent has no nonblocking query."""

    var lib: OwnedDLHandle
    var handle: Int

    def __init__(out self, ctx: DeviceContext) raises:
        self.lib = open_driver()
        self.handle = 0
        comptime name = "hipEventCreateWithFlags" if AMD else "cuEventCreate"
        with ctx.push_context():
            _check(
                self.lib.get_function[Int32](name)(
                    Pointer(to=self.handle), UInt32(2)  # DISABLE_TIMING
                ),
                name,
            )

    def __deinit__(deinit self):
        try:
            self.release()
        except e:
            print("mojoccl: completion event cleanup failed:", e)

    def release(mut self) raises:
        if self.handle != 0:
            comptime name = "hipEventDestroy" if AMD else "cuEventDestroy_v2"
            _check(self.lib.get_function[Int32](name)(self.handle), name)
            self.handle = 0

    def record(self, stream: Int64) raises:
        comptime name = "hipEventRecord" if AMD else "cuEventRecord"
        _check(self.lib.get_function[Int32](name)(self.handle, stream), name)

    def wait_on(self, stream: Int64) raises:
        comptime name = "hipStreamWaitEvent" if AMD else "cuStreamWaitEvent"
        _check(
            self.lib.get_function[Int32](name)(stream, self.handle, UInt32(0)),
            name,
        )

    def synchronize(self) raises:
        comptime name = "hipEventSynchronize" if AMD else "cuEventSynchronize"
        _check(self.lib.get_function[Int32](name)(self.handle), name)

    def query(self) raises -> Bool:
        comptime name = "hipEventQuery" if AMD else "cuEventQuery"
        var rc = self.lib.get_function[Int32](name)(self.handle)
        if rc == 600:  # CUDA_ERROR_NOT_READY == hipErrorNotReady
            return False
        _check(rc, name)
        return True

    def done(self) -> Bool:
        try:
            return self.query()
        except:
            return False


def launch_host_func(
    lib: OwnedDLHandle, stream: Int, func_addr: Int, user_data: Int
) raises:
    """cuLaunchHostFunc / hipLaunchHostFunc on a RAW stream handle.

    The ordering primitive the inter-node hop stands on: a host function
    enqueued on the caller's stream runs after everything enqueued before it
    has completed, and everything enqueued after it waits for it to return.
    That is what lets `transport/net.mojo` post its RDMA writes knowing the
    reduce-scatter's output is final, and lets the add kernel start knowing
    the peers' shards have landed.

    Contract (identical in both vendors' docs): the callback must not call
    into the driver. This library's callback only touches libibverbs and its
    own host memory, never cu*/hip*.
    """
    _check(
        lib.get_function[Int32](FN_LAUNCH_HOST_FUNC)(
            stream, func_addr, user_data
        ),
        FN_LAUNCH_HOST_FUNC,
    )


def stream_done(lib: OwnedDLHandle, stream: Int) -> Bool:
    """`cuStreamQuery`/`hipStreamQuery` on a RAW stream handle: has everything
    enqueued on it completed?

    `ncclCommAbort` polls this instead of synchronizing. A synchronize is
    exactly the unbounded wait abort must not do, and the whole point of the
    abort word is that the spin kernels are already leaving; a poll turns
    "quiesced" into something abort can put a deadline on. Anything but
    success (`CUDA_ERROR_NOT_READY`, or a sticky error from a launch that
    faulted) reads as not-done and the caller's deadline ends the wait.
    """
    try:
        return lib.get_function[Int32](FN_STREAM_QUERY)(stream) == 0
    except:
        return False
