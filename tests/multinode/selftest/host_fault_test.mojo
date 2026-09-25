"""Host-only fault publication checks; no DeviceContext or GPU allocation."""
from std.testing import assert_equal
from tmb.ccl.include.comm import _latch_host_fault_record
from tmb.ccl.include.device import (
    STATUS_FAULT_WORD,
    STATUS_HOST_FAULT_WORD,
    ERR_HOST_LAUNCH,
)


def main() raises:
    var storage = Array[UInt64, 16](uninitialized=True)
    var page = Pointer(to=storage).unsafe_bitcast[UInt64]()
    for device_first in range(2):
        for i in range(16):
            page[unsafe_offset=i] = 0
        # Details may precede the device's final code publication.
        for i in range(1, 7):
            page[unsafe_offset=STATUS_FAULT_WORD + i] = UInt64(100 + i)
        page[unsafe_offset=STATUS_FAULT_WORD] = UInt64(device_first * 7)
        _latch_host_fault_record(Int(page), ERR_HOST_LAUNCH, 123)
        var host = page[unsafe_offset=STATUS_HOST_FAULT_WORD]
        assert_equal(host >> 63, UInt64(device_first))
        assert_equal((host >> 32) & 0x7FFF_FFFF, UInt64(ERR_HOST_LAUNCH))
        assert_equal(host & 0xFFFF_FFFF, UInt64(123))
        assert_equal(
            page[unsafe_offset=STATUS_FAULT_WORD], UInt64(device_first * 7)
        )
        for i in range(1, 7):
            assert_equal(
                page[unsafe_offset=STATUS_FAULT_WORD + i], UInt64(100 + i)
            )
        # A late device publication and a second host error cannot change
        # either the selected source or the original host details.
        page[unsafe_offset=STATUS_FAULT_WORD] = 7
        _latch_host_fault_record(Int(page), ERR_HOST_LAUNCH, 456)
        assert_equal(page[unsafe_offset=STATUS_HOST_FAULT_WORD], host)
    _ = storage
    print("host_fault PASS: disjoint records, stable first-observed selection")
