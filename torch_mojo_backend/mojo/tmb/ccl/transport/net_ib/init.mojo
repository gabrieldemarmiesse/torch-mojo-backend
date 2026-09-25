# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/transport/net_ib/init.cc

from std.os import getenv

from tmb.ccl.env_vars import MOJOCCL_IB_HCA
from tmb.ccl.include.ibvcore import (
    DEV_NAME,
    IBV_LINK_LAYER_INFINIBAND,
    IBV_PORT_ACTIVE,
    PA_ACTIVE_MTU,
    PA_LID,
    PA_LINK_LAYER,
    PA_STATE,
    SZ_PORT_ATTR,
)
from tmb.ccl.misc.ibvwrap import Ibv
from tmb.ccl.misc.utils import P8, alloc_bytes, ld16, ld32, ld64, ld8


# ===-------------------------------------------------------------------=== #
# Device selection
# ===-------------------------------------------------------------------=== #


struct IbPort(Copyable, Movable):
    """One usable (device, port): ACTIVE and InfiniBand link layer."""

    var name: String
    var ctx: Int
    var port: Int
    var lid: Int
    var mtu: Int
    var gid_index: Int
    var subnet_prefix: UInt64

    def __init__(
        out self,
        var name: String,
        ctx: Int,
        port: Int,
        lid: Int,
        mtu: Int,
        gid_index: Int,
        subnet_prefix: UInt64,
    ):
        self.name = name^
        self.ctx = ctx
        self.port = port
        self.lid = lid
        self.mtu = mtu
        self.gid_index = gid_index
        self.subnet_prefix = subnet_prefix


def _device_name(dev: Int) -> String:
    var dp = P8(unsafe_from_address=dev)
    var s = String("")
    var k = 0
    while k < 64 and dp[unsafe_offset=DEV_NAME + k] != 0:
        s += chr(Int(dp[unsafe_offset=DEV_NAME + k]))
        k += 1
    return s^


def list_ib_ports(ibv: Ibv, want: String) raises -> List[IbPort]:
    """Every ACTIVE InfiniBand port, in device order.

    RoCE (link_layer 2) and DOWN ports are skipped -- these nodes carry 10
    IB HCAs plus 2 RoCE ports and only the former are wanted. `want`, if
    non-empty, keeps only that device name (MOJOCCL_IB_HCA).
    Contexts of devices with no accepted port are closed again.
    """
    var out = List[IbPort]()
    var nbuf = alloc_bytes(8)
    var lst = ibv.get_device_list(nbuf)
    if lst == 0:
        return out^
    var n = ld32(nbuf, 0)
    var lp = P8(unsafe_from_address=lst)
    for i in range(n):
        var dev = ld64(lp, i * 8)
        var name = _device_name(dev)
        if want.byte_length() > 0 and name != want:
            continue
        var ctx = ibv.open_device(dev)
        if ctx == 0:
            continue
        var kept = False
        var pa = alloc_bytes(SZ_PORT_ATTR)
        # phys_port_cnt lives in ibv_device_attr; every HCA here is
        # single-port and NCCL itself iterates 1..phys_port_cnt, so probe
        # ports 1 and 2 and let query_port reject what is not there.
        for port in range(1, 3):
            for j in range(SZ_PORT_ATTR):
                pa[unsafe_offset=j] = 0
            if ibv.query_port(ctx, port, pa) != 0:
                continue
            if Int32(ld32(pa, PA_STATE)) != IBV_PORT_ACTIVE:
                continue
            if ld8(pa, PA_LINK_LAYER) != IBV_LINK_LAYER_INFINIBAND:
                continue
            var gid = alloc_bytes(16)
            var prefix: UInt64 = 0
            if ibv.query_gid(ctx, port, 0, gid) == 0:
                prefix = gid.unsafe_bitcast[UInt64]()[unsafe_offset=0]
            out.append(
                IbPort(
                    String(name),
                    ctx,
                    port,
                    ld16(pa, PA_LID),
                    ld32(pa, PA_ACTIVE_MTU),
                    0,
                    prefix,
                )
            )
            kept = True
        if not kept:
            ibv.close_device(ctx)
    ibv.free_device_list(lst)
    return out^


def verbs_available() -> Bool:
    """True if libibverbs opens and lists at least one usable port. Used by
    the backend auto-selection in `transport/net.mojo`; every context it opens is
    closed again before it returns."""
    try:
        var ibv = Ibv()
        var ports = list_ib_ports(ibv, getenv(MOJOCCL_IB_HCA, ""))
        var n = len(ports)
        for i in range(n):
            ibv.close_device(ports[i].ctx)
        return n > 0
    except:
        return False
