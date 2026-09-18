"""Registration of the aten ops.

Every op body is compiled into the backend library and registered with
torch's dispatcher at `tmb_native_init`, one `impl[op, name](site)` line per
aten name inside each group's `register_<group>`. One build holds the
runtime and all the ops (about 4.6 MB, ~50 s cold / ~10 s warm), against a
0.4 MB runtime plus one 6-7 s `mojo build` per op at its first call; the
library still carries no device code, so it ships prebuilt (see
tests/test_backend_has_no_device_code.py).
"""
from std.ffi import external_call

from abi import OpFn, op_address

comptime Lib = OpaquePointer[MutUntrackedOrigin]


@fieldwise_init
struct Site(Copyable, Movable):
    """Where the registration sites of one group file register."""

    var lib: Int  # torch's Library (a TmbLibrary handle)


comptime RegisterFn = def(Site) raises thin -> None


def impl[op: OpFn, name: StaticString](site: Site) raises:
    """Register a kernel in site's library namespace. External namespaces
    use qualified names such as `torchvision::nms`."""
    var s = String(name)
    var rc = external_call["tmb_library_impl", Int32](
        site.lib,
        s.as_c_string_slice().unsafe_ptr(),
        op_address[op](),
        0,
    )
    if rc != 0:
        raise Error("registering ", name, " failed")
