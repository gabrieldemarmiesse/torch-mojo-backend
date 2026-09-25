# Rewrite of: https://github.com/NVIDIA/nccl/blob/master/src/transport/multicast.cc

from std.sys import has_amd_gpu_accelerator
from std.ffi import OwnedDLHandle
from std.memory.alloc import unsafe_alloc

from tmb.ccl.include.transport import NvlsRegion
from tmb.ccl.os.linux_ipcsocket import scm_recv, scm_send, socket_path


# CUmemAllocationHandleType
comptime CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1


comptime MSG_KIND_MC = 0
"""Payload word 0 of the datagram carrying the multicast object's fd."""
comptime MSG_KIND_UC = 1
"""...and of the one carrying a peer's own memory handle; word 1 is that
peer's local rank."""

comptime NVLS_AVAILABLE = not has_amd_gpu_accelerator()
"""Every entry point below is CUDA-only; HIP has no multicast equivalent and
RCCL none either, so an AMD build never reaches them."""


def _cu(lib: OwnedDLHandle, rc: Int32, what: String) raises:
    if rc != 0:
        raise Error("mojoccl: " + what + " failed, driver rc=" + String(rc))


def _mc_prop(world: Int, size: Int) -> Pointer[UInt64, MutUntrackedOrigin]:
    """CUmulticastObjectProp{numDevices, size, handleTypes, flags}, 32 B."""
    var p = unsafe_alloc[UInt64](4)
    p[unsafe_offset=0] = UInt64(UInt32(world))  # numDevices +0, pad +4
    p[unsafe_offset=1] = UInt64(size)  # size +8
    p[unsafe_offset=2] = UInt64(
        CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
    )  # handleTypes +16
    p[unsafe_offset=3] = 0  # flags +24
    return p


def nvls_create_and_share(
    lib: OwnedDLHandle,
    libc: OwnedDLHandle,
    dir: String,
    magic: UInt64,
    local_rank: Int,
    local_world: Int,
    ordinal: Int,
    size: Int,
    gran: Int,
    sock: Int,
    timeout_s: Float64,
    mut region: NvlsRegion,
) raises:
    """Steps 1-3 of the bring-up: the multicast object exists on every rank and
    every device has joined it. The caller barriers after this, then calls
    `nvls_bind_and_map` -- the split is where the "every device in the team
    before any memory is bound" rule lives.
    """
    comptime if not NVLS_AVAILABLE:
        raise Error("mojoccl: NVLS is CUDA-only")
    region.size = size
    region.granularity = gran
    var mch: UInt64 = 0
    if local_rank == 0:
        _cu(
            lib,
            lib.get_function[Int32]("cuMulticastCreate")(
                Pointer(to=mch), _mc_prop(local_world, size)
            ),
            "cuMulticastCreate",
        )
        # Owned by the region from this line on, so a failure in the export
        # or the fd hand-off below is released by `nvls_teardown`.
        region.mc_handle = mch
        var fd: Int32 = -1
        _cu(
            lib,
            lib.get_function[Int32]("cuMemExportToShareableHandle")(
                Pointer(to=fd),
                mch,
                Int32(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
                UInt64(0),
            ),
            "cuMemExportToShareableHandle(multicast)",
        )
        try:
            for r in range(1, local_world):
                scm_send(
                    libc,
                    socket_path(dir, magic, r),
                    Int(fd),
                    MSG_KIND_MC,
                    0,
                    timeout_s,
                )
        except e:
            _ = libc.get_function[Int32]("close")(fd)
            raise e
        _ = libc.get_function[Int32]("close")(fd)
    else:
        var got = scm_recv(libc, sock)
        if got[1] != MSG_KIND_MC:
            _ = libc.get_function[Int32]("close")(Int32(got[0]))
            raise Error(
                "mojoccl: expected the multicast fd, got kind " + String(got[1])
            )
        var rc = lib.get_function[Int32]("cuMemImportFromShareableHandle")(
            Pointer(to=mch),
            got[0],
            Int32(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR),
        )
        _ = libc.get_function[Int32]("close")(Int32(got[0]))
        _cu(lib, rc, "cuMemImportFromShareableHandle(multicast)")
        region.mc_handle = mch
    _cu(
        lib,
        lib.get_function[Int32]("cuMulticastAddDevice")(mch, Int32(ordinal)),
        "cuMulticastAddDevice",
    )
