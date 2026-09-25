"""Entry point of the native backend: `tmb_native_init` registers the device
hooks with the C++ shim and every aten op name with torch's dispatcher.

This library holds the runtime — devices, streams, events, memory, the
loader, the record ABI — and every op body, registered eagerly through the
`register_<group>` list of each group file (registry.mojo). Kernels are not
in it: each (op, dtype) specialization is built by the loader at first use,
so the library carries no device code and ships prebuilt.

Build: `mojo build tmb/backend/entry.mojo --emit shared-lib -I torch_mojo_backend/mojo`
(native/__init__.py does it, cached like every other on-demand build).
"""
from std.ffi import c_char, external_call

from tmb.backend.abi import set_shim_error
from tmb.backend.device import hooks_table, init_backend
from tmb.backend.kernel_call import init_loader
from tmb.ops.attention import register_attention
from tmb.ops.binary import register_binary
from tmb.ops.compare import register_compare
from tmb.ops.composed import register_composed
from tmb.ops.core import register_core
from tmb.ops.data_movement import register_data_movement
from tmb.ops.deform_conv import register_deform_conv
from tmb.ops.factories import register_factories
from tmb.ops.foreach import register_foreach
from tmb.ops.matmul import register_matmul
from tmb.ops.nn import register_nn
from tmb.ops.nms import register_nms
from tmb.ops.random import register_random
from tmb.ops.reductions import register_reductions
from tmb.ops.roi import register_roi
from tmb.ops.unary import register_unary
from tmb.backend.pg import pg_vtable
from tmb.backend.registry import Lib, RegisterFn, Site


def _group[reg: RegisterFn](lib: Int) raises:
    """Register one file's ops."""
    reg(Site(lib))


def _register_ops(lib: Int) raises:
    _group[register_core](lib)
    _group[register_unary](lib)
    # after every group it composes from
    _group[register_composed](lib)
    _group[register_binary](lib)
    _group[register_compare](lib)
    _group[register_data_movement](lib)
    _group[register_factories](lib)
    _group[register_random](lib)
    _group[register_reductions](lib)
    _group[register_matmul](lib)
    _group[register_nn](lib)
    _group[register_attention](lib)
    _group[register_foreach](lib)


def _register_detection(lib: Int) raises:
    _group[register_deform_conv](lib)
    _group[register_nms](lib)
    _group[register_roi](lib)


@export
def tmb_native_init(
    mojo_root: Pointer[c_char, MutUntrackedOrigin],
    cache_dir: Pointer[c_char, MutUntrackedOrigin],
    mojo_exe: Pointer[c_char, MutUntrackedOrigin],
    toolchain: Pointer[c_char, MutUntrackedOrigin],
    trace: Int32,
) abi("C") -> Int32:
    """Returns the mojo device count, or -1 with the error in tmb_get_error."""
    try:
        var n = init_backend()
        init_loader(
            String(unsafe_from_utf8_ptr=mojo_root.unsafe_bitcast[UInt8]()),
            String(unsafe_from_utf8_ptr=cache_dir.unsafe_bitcast[UInt8]()),
            String(unsafe_from_utf8_ptr=mojo_exe.unsafe_bitcast[UInt8]()),
            String(unsafe_from_utf8_ptr=toolchain.unsafe_bitcast[UInt8]()),
            trace != 0,
        )
        var table = hooks_table()
        if external_call["tmb_backend_register", Int32](table) != 0:
            raise Error("tmb_backend_register failed")
        var ns = String("aten")
        var key = String("PrivateUse1")
        var lib = external_call["tmb_library_new", Lib](
            ns.as_c_string_span().ptr(),
            key.as_c_string_span().ptr(),
        )
        if Int(lib) == 0:
            raise Error("tmb_library_new failed")
        _register_ops(Int(lib))
        ns = String("torchvision")
        var detection_lib = external_call["tmb_library_new", Lib](
            ns.as_c_string_span().ptr(),
            key.as_c_string_span().ptr(),
        )
        if Int(detection_lib) == 0:
            raise Error("tmb_library_new(torchvision) failed")
        _register_detection(Int(detection_lib))
        return Int32(n)
    except e:
        set_shim_error(String(e))
        return -1


@export
def tmb_pg_vtable() abi("C") -> Int:
    """Addresses of the process-group entries (pg.mojo pg_vtable order):
    functions of an imported module are not exported from the library, so
    Python takes them from this table."""
    return Int(pg_vtable())
