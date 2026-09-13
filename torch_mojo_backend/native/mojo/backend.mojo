"""Entry point of the native backend: `tmb_native_init` registers the device
hooks with the C++ shim and every aten op name with torch's dispatcher.

This library is the runtime only — devices, streams, events, memory, the
loader, the record ABI and the registration list below. No op body is
compiled into it: each is built alone, from its own `ops_<group>.mojo`, at
its first call (registry.mojo). The group files are still imported here, for
their `register_<group>` lists; those lists reference the op functions only
inside the branch the `TMB_OP` define drops, so the bodies stay out.

Build: `mojo build backend.mojo --emit shared-lib -I native/mojo -I eager_kernels`
(native/__init__.py does it, cached like every other on-demand build).
"""
from std.ffi import c_char, external_call

from abi import set_shim_error
from device import hooks_table, init_backend
from kernels import init_loader
from ops_attention import register_attention
from ops_binary import register_binary
from ops_compare import register_compare
from ops_composed import register_composed
from ops_core import register_core
from ops_data_movement import register_data_movement
from ops_factories import register_factories
from ops_foreach import register_foreach
from ops_matmul import register_matmul
from ops_nn import register_nn
from ops_reductions import register_reductions
from ops_unary import register_unary
from pg import Locked, pg_vtable
from registry import Lib, RegisterFn, Site


def _group[
    reg: RegisterFn
](lib: Int, group: StaticString, prebuild: Bool) raises:
    """Register one file's ops (or, with `prebuild`, build their extensions
    right away); `group` names the file the extensions are built from."""
    var unused = 0
    reg(
        Site(
            lib,
            group,
            Pointer(to=unused).unsafe_origin_cast[MutUntrackedOrigin](),
            prebuild,
        )
    )


def _register_ops(lib: Int, prebuild: Bool = False) raises:
    _group[register_core](lib, "ops_core", prebuild)
    _group[register_unary](lib, "ops_unary", prebuild)
    # after every group it composes from
    _group[register_composed](lib, "ops_composed", prebuild)
    _group[register_binary](lib, "ops_binary", prebuild)
    _group[register_compare](lib, "ops_compare", prebuild)
    _group[register_data_movement](lib, "ops_data_movement", prebuild)
    _group[register_factories](lib, "ops_factories", prebuild)
    _group[register_reductions](lib, "ops_reductions", prebuild)
    _group[register_matmul](lib, "ops_matmul", prebuild)
    _group[register_nn](lib, "ops_nn", prebuild)
    _group[register_attention](lib, "ops_attention", prebuild)
    _group[register_foreach](lib, "ops_foreach", prebuild)


@export
def tmb_native_init(
    kernels_dir: Pointer[c_char, MutUntrackedOrigin],
    mojo_dir: Pointer[c_char, MutUntrackedOrigin],
    cache_dir: Pointer[c_char, MutUntrackedOrigin],
    mojo_exe: Pointer[c_char, MutUntrackedOrigin],
    toolchain: Pointer[c_char, MutUntrackedOrigin],
    trace: Int32,
) abi("C") -> Int32:
    """Returns the mojo device count, or -1 with the error in tmb_get_error."""
    try:
        var n = init_backend()
        init_loader(
            String(unsafe_from_utf8_ptr=kernels_dir.unsafe_bitcast[UInt8]()),
            String(unsafe_from_utf8_ptr=mojo_dir.unsafe_bitcast[UInt8]()),
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
            ns.as_c_string_slice().unsafe_ptr(),
            key.as_c_string_slice().unsafe_ptr(),
        )
        if Int(lib) == 0:
            raise Error("tmb_library_new failed")
        _register_ops(Int(lib))
        return Int32(n)
    except e:
        set_shim_error(String(e))
        return -1


@export
def tmb_prebuild_ops() abi("C") -> Int32:
    """Build every op extension now, instead of one per first call. Nothing
    needs it at runtime; it exists so a test suite or a CI image pays the
    compilations up front (and outside any GPU lock) rather than inside the
    first call of each op."""
    try:
        with Locked():  # the loader's tables are shared with the lazy first calls
            _register_ops(0, prebuild=True)
        return 0
    except e:
        set_shim_error(String(e))
        return 1


@export
def tmb_pg_vtable() abi("C") -> Int:
    """Addresses of the process-group entries (pg.mojo pg_vtable order):
    functions of an imported module are not exported from the library, so
    Python takes them from this table."""
    return Int(pg_vtable())
