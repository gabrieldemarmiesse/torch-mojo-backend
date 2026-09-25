"""Read-only lifecycle assertions for stream_order_probe.py."""

from tmb.ccl.include.comm import _comm_ptr


@export
def mojoccl_test_resources_released(comm: Int64) abi("C") -> Int32:
    ref state = _comm_ptr(comm)[]
    return Int32(
        state.released
        and state.ib == 0
        and state.abort_host == 0
        and state.abort_dev == 0
        and state.order_event.handle == 0
    )
