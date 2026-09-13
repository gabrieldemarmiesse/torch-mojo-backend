"""The process-wide kernel loader and the spec-op calling convention.

`spec_call(family, op, dtypes, specs...)` is what an op uses to run one
kernel: it derives the specialization defines the way the Python loader did
(OP, DTYPE_ARG_i, DTYPE_OUT, flags), builds the argument slots and calls the
family's C entry through the loader.
"""
from std.ffi import _get_global_or_null, external_call
from std.memory.alloc import unsafe_alloc

from abi import dtype_code
from loader import Loader, call_family
from op_utils import Arg, Argv, TensorSpec, _f64_slot

comptime LOADER_GLOBAL = "TMB_NATIVE_LOADER"


def init_loader(
    kernels_dir: String,
    mojo_dir: String,
    cache_dir: String,
    mojo_exe: String,
    toolchain: String,
    trace: Bool,
):
    if _get_global_or_null(LOADER_GLOBAL):
        return
    var box = unsafe_alloc[Loader](1)
    box.unsafe_write(
        Loader(kernels_dir, mojo_dir, cache_dir, mojo_exe, toolchain, trace)
    )
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(LOADER_GLOBAL), box.unsafe_bitcast[NoneType]()
    )


@always_inline
def loader() -> Pointer[Loader, MutUntrackedOrigin]:
    return _get_global_or_null(LOADER_GLOBAL).value().unsafe_bitcast[Loader]()


def dtype_name(dt: DType) -> String:
    """The `max.dtype.DType.name` spelling the gates compare against."""
    if dt == DType.float32:
        return "float32"
    if dt == DType.bfloat16:
        return "bfloat16"
    if dt == DType.float16:
        return "float16"
    if dt == DType.float64:
        return "float64"
    if dt == DType.int64:
        return "int64"
    if dt == DType.int32:
        return "int32"
    if dt == DType.int16:
        return "int16"
    if dt == DType.int8:
        return "int8"
    if dt == DType.uint8:
        return "uint8"
    if dt == DType.uint16:
        return "uint16"
    if dt == DType.uint32:
        return "uint32"
    if dt == DType.uint64:
        return "uint64"
    if dt == DType.bool:
        return "bool"
    return String(dt)


@always_inline
def _mix(mut h: UInt64, x: UInt64):
    h ^= x
    h *= 1099511628211


@always_inline
def _mix_bytes(mut h: UInt64, s: String):
    for b in s.as_bytes():
        _mix(h, UInt64(b))


struct Defines(Movable):
    """The -D set of one specialization, in the canonical (sorted) order the
    cache key uses. `key` is a running hash of the same information so the
    loader's hot path never builds a string: the strings are only produced
    on a miss (`sorted()`)."""

    var items: List[String]
    var key: UInt64

    def __init__(out self, op: String):
        self.items = List[String]()
        self.items.append("OP=" + op)
        self.key = 14695981039346656037
        _mix_bytes(self.key, op)

    def arg(mut self, i: Int, dt: DType):
        self.items.append("DTYPE_ARG_" + String(i) + "=" + dtype_name(dt))
        _mix(self.key, UInt64(0x1000 + i))
        _mix(self.key, UInt64(dtype_code(dt)))

    def out(mut self, dt: DType):
        self.items.append("DTYPE_OUT=" + dtype_name(dt))
        _mix(self.key, UInt64(0x2000))
        _mix(self.key, UInt64(dtype_code(dt)))

    def out_i(mut self, i: Int, dt: DType):
        self.items.append("DTYPE_OUT_" + String(i) + "=" + dtype_name(dt))
        _mix(self.key, UInt64(0x3000 + i))
        _mix(self.key, UInt64(dtype_code(dt)))

    def flag(mut self, name: String, value: Int):
        self.items.append(name + "=" + String(value))
        _mix(self.key, UInt64(0x4000))
        _mix_bytes(self.key, name)
        _mix(self.key, UInt64(value))

    def sorted(self) -> List[String]:
        var out = self.items.copy()
        sort(out)
        return out^


comptime MAX_CALL_SPECS = 16


struct KernelCall(Movable):
    """One kernel invocation: the specialization defines plus the argument
    slots, and the storage behind them (TensorSpecs, tuples) that must outlive
    the call. Mojo destroys a local right after its last use, so a spec whose
    address was taken and then handed over as an Int would be gone by the
    time the kernel reads it; owning everything here and calling `run()` on
    the struct keeps every address valid for exactly the call.
    """

    var family: String
    var defines: Defines
    var specs: List[TensorSpec]
    var tuples: List[List[Int]]
    var slots: List[Int]

    def __init__(out self, family: String, op: String):
        self.family = family
        self.defines = Defines(op)
        self.specs = List[TensorSpec](
            capacity=MAX_CALL_SPECS
        )  # never reallocates: addresses stay put
        self.tuples = List[List[Int]]()
        self.slots = List[Int]()

    def arg_dtype(mut self, i: Int, dt: DType):
        self.defines.arg(i, dt)

    def out_dtype(mut self, dt: DType):
        self.defines.out(dt)

    def out_dtype_i(mut self, i: Int, dt: DType):
        self.defines.out_i(i, dt)

    def flag(mut self, name: String, value: Int):
        self.defines.flag(name, value)

    def spec(mut self, var s: TensorSpec) raises:
        if len(self.specs) >= MAX_CALL_SPECS:
            raise Error("too many spec arguments in one kernel call")
        self.specs.append(s^)
        self.slots.append(Int(Pointer(to=self.specs[len(self.specs) - 1])))

    def int(mut self, v: Int):
        self.slots.append(v)

    def f64(mut self, v: Float64):
        self.slots.append(_f64_slot(v))

    def tuple(mut self, values: List[Int]):
        """A `[len, e0, e1, ...]` tuple slot (the kernels read it with _raw_tuple_*).
        """
        var t = List[Int](capacity=len(values) + 1)
        t.append(len(values))
        for v in values:
            t.append(v)
        self.tuples.append(t^)
        self.slots.append(Int(self.tuples[len(self.tuples) - 1].unsafe_ptr()))

    def run(self) raises:
        var key = self.defines.key
        _mix_bytes(key, self.family)
        var l = loader()
        if l[].fast.find(key):
            call_family(
                l[],
                self.family,
                key,
                List[String](),
                Argv(unsafe_from_address=Int(self.slots.unsafe_ptr())),
                len(self.slots),
            )
        else:
            call_family(
                l[],
                self.family,
                key,
                self.defines.sorted(),
                Argv(unsafe_from_address=Int(self.slots.unsafe_ptr())),
                len(self.slots),
            )
