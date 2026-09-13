"""Registration of the aten ops, and the entry of one op's extension.

An op body is never compiled into the backend library: it is compiled on its
first call, alone, into an extension built from its own `ops_<group>.mojo`
with `-D TMB_OP=<aten name>`. The same `impl[op, name](site)` call site does
both jobs, selected by that define:

* **no define** (the backend library) — register `name` with torch's
  dispatcher behind the lazy trampoline below. `op` appears only in the branch
  the compiler drops, so no op body is elaborated.
* **`TMB_OP=<name>`** (that op's extension) — hand back the address of this
  op's entry. Every other call site of the group compiles to nothing, which is
  what keeps an extension to one op instead of a whole group.

The backend's registration list therefore stays exactly where it was, one
`impl[...]` line per aten name inside each group's `register_<group>`.
"""
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc
from std.sys.defines import get_defined_string

from abi import OpFn, op_address, set_shim_error
from kernels import loader

comptime Lib = OpaquePointer[MutUntrackedOrigin]

comptime TARGET_OP = get_defined_string["TMB_OP", ""]()


@fieldwise_init
struct Site(Copyable, Movable):
    """What the registration sites of one group file do in this build."""

    var lib: Int  # backend library: torch's Library (a TmbLibrary handle)
    var group: StaticString  # backend library: file the extension builds from
    var addr: Pointer[Int, MutUntrackedOrigin]  # extension: resolved entry
    var prebuild: Bool  # backend library: build now instead of registering


comptime RegisterFn = def(Site) raises thin -> None


@fieldwise_init
struct LazyOp(Movable):
    """What the trampoline needs to build an op's extension. One per
    registered aten name, alive for the process."""

    var group: StaticString
    var name: StaticString


def _resolve(
    ctx: Int, slot: Pointer[Int, MutUntrackedOrigin]
) abi("C") -> Int32:
    """tmb.h TmbResolveFn: build/load this op's extension and report the
    address of its entry, which the shim then calls directly."""
    try:
        var op = Pointer[LazyOp, MutUntrackedOrigin](unsafe_from_address=ctx)
        slot[] = loader()[].op_entry(String(op[].group), String(op[].name))
        return 0
    except e:
        set_shim_error(String(e))
        return 1


def _register_lazy(
    lib: Int, group: StaticString, name: StaticString, prebuild: Bool
) raises:
    """One call per registration site, so the backend library holds a name
    and a call rather than a copy of this per op."""
    if prebuild:
        _ = loader()[].op_entry(String(group), String(name))
        return
    var box = unsafe_alloc[LazyOp](1)
    box.unsafe_write(LazyOp(group, name))
    var f: def(Int, Pointer[Int, MutUntrackedOrigin]) thin abi(
        "C"
    ) -> Int32 = _resolve
    var s = String(name)
    var rc = external_call["tmb_library_impl_lazy", Int32](
        lib,
        s.as_c_string_slice().unsafe_ptr(),
        Pointer(to=f).unsafe_bitcast[Int]()[],
        box.unsafe_bitcast[NoneType](),
    )
    if rc != 0:
        raise Error("registering aten::", name, " failed")


def impl[op: OpFn, name: StaticString](site: Site) raises:
    """Register `op` as the PrivateUse1 kernel of aten::<name> ("add.Tensor",
    "view", "fill_.Scalar", ...)."""
    comptime if TARGET_OP == "":
        _register_lazy(site.lib, site.group, name, site.prebuild)
    elif TARGET_OP == name:
        site.addr[] = op_address[op]()


def op_address_of[reg: RegisterFn]() -> Int:
    """The entry of the one op this extension was built for (0 = none, which
    means the TMB_OP define names no op of this group)."""
    comptime if TARGET_OP == "":
        return 0
    else:
        var addr = 0
        try:
            reg(
                Site(
                    0,
                    "",
                    Pointer(to=addr).unsafe_origin_cast[MutUntrackedOrigin](),
                    False,
                )
            )
        except e:
            set_shim_error(String(e))
            return 0
        return addr
