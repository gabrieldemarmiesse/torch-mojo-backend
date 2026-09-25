# Host-only assertions of the production dispatch values. Cross-compile for
# gfx942, sm_90a and sm_80, then run on any CPU; no DeviceContext is opened.
from std.sys.info import _accelerator_arch
from std.sys import argv
from std.testing import assert_equal

from tmb.ccl.proxy import _proxy_idle_ns
from tmb.ccl.device.symmetric.all_reduce_gin import (
    FUSED_THREADS,
    FUSED_UNROLL,
    FUSED_PUSH_UNROLL,
    FUSED_CTAS_PER_SM,
    fused_block_cap,
    fused_big_block_cap,
    fused_big_bytes,
)
from tmb.ccl.init import _region_cap_bytes, _pipe_split_unit, _fused_enabled
from tmb.ccl.transport.nvls import (
    _nvls_min_bytes,
    _nvls_recommended_granularity,
)
from tmb.ccl.os.linux_ipcsocket import _socket_dir


def main() raises:
    # Expected values come from the runner, independently of the production
    # architecture gate. This catches host/device target spelling mistakes.
    var args = argv()
    assert_equal(len(args), 4)
    assert_equal(fused_block_cap(), Int(String(args[1])))
    assert_equal(fused_big_block_cap(), Int(String(args[2])))
    assert_equal(_region_cap_bytes(), Int(String(args[3])) * 1024 * 1024)
    assert_equal(fused_big_bytes(), 128 * 1024 * 1024)
    assert_equal(FUSED_THREADS, 512)
    assert_equal(FUSED_UNROLL, 4)
    assert_equal(FUSED_PUSH_UNROLL, 4)
    assert_equal(FUSED_CTAS_PER_SM, 1)
    assert_equal(_pipe_split_unit(), 640_000)
    assert_equal(_fused_enabled(), True)
    assert_equal(_proxy_idle_ns(), 20_000)
    assert_equal(_nvls_min_bytes(), 48 * 1024 * 1024)
    assert_equal(_nvls_recommended_granularity(), False)
    assert_equal(_socket_dir(), String("/tmp"))
    print(
        "defaults PASS",
        _accelerator_arch(),
        "caps",
        fused_block_cap(),
        fused_big_block_cap(),
        "region_bytes",
        _region_cap_bytes(),
    )
