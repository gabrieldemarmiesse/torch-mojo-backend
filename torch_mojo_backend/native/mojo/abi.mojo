"""Records and tensor views shared by every op of the native backend.

The C++ shim (native/csrc/tmb.h) hands each op a stack of `Value` records,
one per schema argument, and takes the results back in the same form.
Tensors travel as `at::Tensor*` handles; what an op needs about one is read
once into a `T` view through the shim's C getters.
"""
from std.ffi import c_char, external_call
from std.sys._libc import free as libc_free
from std.memory import bitcast
from std.memory.alloc import unsafe_alloc
from std.utils import IndexList

from op_utils import MAX_RANK, TensorSpec

# --- record tags (tmb.h TmbTag) --------------------------------------------
comptime TAG_NONE = 0
comptime TAG_TENSOR = 1
comptime TAG_TENSOR_REF = 2
comptime TAG_INT = 3
comptime TAG_DOUBLE = 4
comptime TAG_BOOL = 5
comptime TAG_COMPLEX = 6
comptime TAG_INT_LIST = 7
comptime TAG_DOUBLE_LIST = 8
comptime TAG_BOOL_LIST = 9
comptime TAG_TENSOR_LIST = 10
comptime TAG_OPT_TENSOR_LIST = 11
comptime TAG_DTYPE = 12
comptime TAG_LAYOUT = 13
comptime TAG_DEVICE = 14
comptime TAG_MEMORY_FORMAT = 15
comptime TAG_STRING = 16
comptime TAG_GENERATOR = 17
comptime TAG_SCALAR_INT = 18
comptime TAG_SCALAR_DOUBLE = 19
comptime TAG_SCALAR_BOOL = 20
comptime TAG_STREAM = 21

# --- torch ScalarType values (c10/core/ScalarType.h) ------------------------
comptime ST_UINT8 = Int32(0)
comptime ST_INT8 = Int32(1)
comptime ST_INT16 = Int32(2)
comptime ST_INT32 = Int32(3)
comptime ST_INT64 = Int32(4)
comptime ST_FLOAT16 = Int32(5)
comptime ST_FLOAT32 = Int32(6)
comptime ST_FLOAT64 = Int32(7)
comptime ST_COMPLEX32 = Int32(8)
comptime ST_COMPLEX64 = Int32(9)
comptime ST_COMPLEX128 = Int32(10)
comptime ST_BOOL = Int32(11)
comptime ST_BFLOAT16 = Int32(15)
comptime ST_UINT16 = Int32(27)
comptime ST_UINT32 = Int32(28)
comptime ST_UINT64 = Int32(29)

comptime DEVICE_TYPE_CPU = 0
comptime DEVICE_TYPE_PRIVATEUSE1 = 20

comptime MEMORY_FORMAT_CONTIGUOUS = 0
comptime MEMORY_FORMAT_PRESERVE = 1
comptime MEMORY_FORMAT_CHANNELS_LAST = 2
comptime MEMORY_FORMAT_CHANNELS_LAST_3D = 3


def max_dtype(stype: Int32) raises -> DType:
    if stype == ST_FLOAT32:
        return DType.float32
    if stype == ST_BFLOAT16:
        return DType.bfloat16
    if stype == ST_FLOAT16:
        return DType.float16
    if stype == ST_INT64:
        return DType.int64
    if stype == ST_BOOL:
        return DType.bool
    if stype == ST_FLOAT64:
        return DType.float64
    if stype == ST_INT32:
        return DType.int32
    if stype == ST_UINT8:
        return DType.uint8
    if stype == ST_INT8:
        return DType.int8
    if stype == ST_INT16:
        return DType.int16
    if stype == ST_UINT16:
        return DType.uint16
    if stype == ST_UINT32:
        return DType.uint32
    if stype == ST_UINT64:
        return DType.uint64
    unsupported(
        "dtype (torch ScalarType "
        + String(stype)
        + ") is not supported on the mojo device"
    )
    return DType.float32


def torch_dtype(dt: DType) raises -> Int32:
    if dt == DType.float32:
        return ST_FLOAT32
    if dt == DType.bfloat16:
        return ST_BFLOAT16
    if dt == DType.float16:
        return ST_FLOAT16
    if dt == DType.int64:
        return ST_INT64
    if dt == DType.bool:
        return ST_BOOL
    if dt == DType.float64:
        return ST_FLOAT64
    if dt == DType.int32:
        return ST_INT32
    if dt == DType.uint8:
        return ST_UINT8
    if dt == DType.int8:
        return ST_INT8
    if dt == DType.int16:
        return ST_INT16
    if dt == DType.uint16:
        return ST_UINT16
    if dt == DType.uint32:
        return ST_UINT32
    if dt == DType.uint64:
        return ST_UINT64
    raise Error("no torch dtype for ", dt)


def dtype_code(dt: DType) -> Int:
    """The numeric value of `max.dtype.DType` (what kernels decode with
    `_raw_dtype_int` / `DType._from_ui8`)."""
    if dt == DType.float32:
        return 81
    if dt == DType.bfloat16:
        return 80
    if dt == DType.float16:
        return 79
    if dt == DType.float64:
        return 82
    if dt == DType.bool:
        return 1
    if dt == DType.uint8:
        return 134
    if dt == DType.int8:
        return 135
    if dt == DType.uint16:
        return 136
    if dt == DType.int16:
        return 137
    if dt == DType.uint32:
        return 138
    if dt == DType.int32:
        return 139
    if dt == DType.uint64:
        return 140
    return 141  # int64


def dtype_itemsize(dt: DType) -> Int:
    if dt == DType.float32 or dt == DType.int32 or dt == DType.uint32:
        return 4
    if (
        dt == DType.bfloat16
        or dt == DType.float16
        or dt == DType.int16
        or dt == DType.uint16
    ):
        return 2
    if dt == DType.float64 or dt == DType.int64 or dt == DType.uint64:
        return 8
    return 1


def itemsize_of(stype: Int32) raises -> Int:
    return dtype_itemsize(max_dtype(stype))


def is_floating(stype: Int32) -> Bool:
    return (
        stype == ST_FLOAT32
        or stype == ST_BFLOAT16
        or stype == ST_FLOAT16
        or stype == ST_FLOAT64
    )


def dtype_name(stype: Int32) -> String:
    """What `caffe2::TypeMeta` streams into a torch error message.

    It is the demangled C++ type behind the ScalarType, not the python
    `torch.float32` spelling, so an `out=` message raised here reads exactly
    like CPU torch's. The integer spellings are gcc's; clang writes `short`
    where gcc writes `short int`.
    """
    if stype == ST_FLOAT32:
        return String("float")
    if stype == ST_FLOAT64:
        return String("double")
    if stype == ST_FLOAT16:
        return String("c10::Half")
    if stype == ST_BFLOAT16:
        return String("c10::BFloat16")
    if stype == ST_BOOL:
        return String("bool")
    if stype == ST_INT8:
        return String("signed char")
    if stype == ST_UINT8:
        return String("unsigned char")
    if stype == ST_INT16:
        return String("short int")
    if stype == ST_UINT16:
        return String("short unsigned int")
    if stype == ST_INT32:
        return String("int")
    if stype == ST_UINT32:
        return String("unsigned int")
    if stype == ST_INT64:
        return String("long int")
    if stype == ST_UINT64:
        return String("long unsigned int")
    return String("ScalarType ") + String(stype)


# --- errors -------------------------------------------------------------------
comptime UNSUPPORTED_PREFIX = "[unsupported] "


def unsupported(msg: String) raises:
    """Decline the call: reported to torch as NotImplementedError (rc 2)."""
    raise Error(UNSUPPORTED_PREFIX + msg)


def shim_error() -> String:
    var p = external_call[
        "tmb_get_error", Pointer[c_char, MutUntrackedOrigin]
    ]()
    if Int(p) == 0:
        return String("")
    return String(unsafe_from_utf8_ptr=p.unsafe_bitcast[UInt8]())


def check(rc: Int32, what: StaticString) raises:
    if rc != 0:
        raise Error(what, ": ", shim_error())


def set_shim_error(msg: String):
    var tmp = String(msg)
    external_call["tmb_set_error", NoneType](
        tmp.as_c_string_slice().unsafe_ptr()
    )


# --- records ------------------------------------------------------------------
@fieldwise_init
struct Value(Copyable, Movable):
    var tag: Int32
    var len: Int32
    var a: Int64
    var b: Int64


comptime Values = Pointer[Value, MutUntrackedOrigin]


@always_inline
def f64_bits(v: Float64) -> Int64:
    return bitcast[DType.int64](v)


@always_inline
def bits_f64(b: Int64) -> Float64:
    return bitcast[DType.float64](b)


def v_is_none(v: Value) -> Bool:
    return v.tag == TAG_NONE


def v_int(v: Value) raises -> Int:
    if (
        v.tag == TAG_INT
        or v.tag == TAG_SCALAR_INT
        or v.tag == TAG_BOOL
        or v.tag == TAG_SCALAR_BOOL
    ):
        return Int(v.a)
    if v.tag == TAG_DOUBLE or v.tag == TAG_SCALAR_DOUBLE:
        return Int(bits_f64(v.a))
    raise Error("expected an int argument, got record tag ", v.tag)


def v_int_or(v: Value, default: Int) raises -> Int:
    if v.tag == TAG_NONE:
        return default
    return v_int(v)


def v_f64(v: Value) raises -> Float64:
    if v.tag == TAG_DOUBLE or v.tag == TAG_SCALAR_DOUBLE:
        return bits_f64(v.a)
    if (
        v.tag == TAG_INT
        or v.tag == TAG_SCALAR_INT
        or v.tag == TAG_BOOL
        or v.tag == TAG_SCALAR_BOOL
    ):
        return Float64(v.a)
    raise Error("expected a number argument, got record tag ", v.tag)


def v_f64_or(v: Value, default: Float64) raises -> Float64:
    if v.tag == TAG_NONE:
        return default
    return v_f64(v)


def v_bool(v: Value) raises -> Bool:
    if (
        v.tag == TAG_BOOL
        or v.tag == TAG_SCALAR_BOOL
        or v.tag == TAG_INT
        or v.tag == TAG_SCALAR_INT
    ):
        return v.a != 0
    if v.tag == TAG_DOUBLE or v.tag == TAG_SCALAR_DOUBLE:
        return bits_f64(v.a) != 0.0
    raise Error("expected a bool argument, got record tag ", v.tag)


def v_bool_or(v: Value, default: Bool) raises -> Bool:
    if v.tag == TAG_NONE:
        return default
    return v_bool(v)


def v_scalar_is_integral(v: Value) -> Bool:
    """True for a Scalar carrying an int or bool (torch's `isIntegral(true)`).
    """
    return (
        v.tag == TAG_SCALAR_INT
        or v.tag == TAG_SCALAR_BOOL
        or v.tag == TAG_INT
        or v.tag == TAG_BOOL
    )


def v_scalar_is_bool(v: Value) -> Bool:
    return v.tag == TAG_SCALAR_BOOL or v.tag == TAG_BOOL


def v_dtype_or(v: Value, default: Int32) raises -> Int32:
    if v.tag == TAG_NONE:
        return default
    if v.tag != TAG_DTYPE and v.tag != TAG_INT:
        raise Error("expected a dtype argument, got record tag ", v.tag)
    return Int32(v.a)


def v_device_index(v: Value) -> Int:
    """The mojo index of a Device? argument: -1 when None or unset."""
    if v.tag != TAG_DEVICE:
        return -1
    return Int(v.b)


def v_device_type(v: Value) -> Int:
    if v.tag != TAG_DEVICE:
        return -1
    return Int(v.a)


def v_memory_format_or(v: Value, default: Int) raises -> Int:
    if v.tag == TAG_NONE:
        return default
    return Int(v.a)


def v_stream(v: Value) raises -> Tuple[Int, Int]:
    """A torch.Stream argument: (device index, stream id)."""
    if v.tag != TAG_STREAM:
        raise Error("expected a Stream argument, got record tag ", v.tag)
    return (Int(v.a), Int(v.b))


def v_generator(v: Value) -> Int:
    """`at::Generator*` or 0 for None."""
    if v.tag != TAG_GENERATOR:
        return 0
    return Int(v.a)


def v_string(v: Value) raises -> String:
    if v.tag != TAG_STRING:
        raise Error("expected a string argument, got record tag ", v.tag)
    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
    return String(
        StringSlice(
            from_utf8=Span[UInt8, MutUntrackedOrigin](
                unsafe_ptr=p, length=Int(v.len)
            )
        )
    )


struct IntList(Copyable, Movable, Sized):
    """A borrowed `int[]` argument (the arena lives for the call)."""

    var addr: Int
    var n: Int

    def __init__(out self, v: Value) raises:
        if v.tag == TAG_NONE:
            self.addr = 0
            self.n = 0
            return
        if v.tag != TAG_INT_LIST:
            raise Error("expected an int[] argument, got record tag ", v.tag)
        self.addr = Int(v.a)
        self.n = Int(v.len)

    @always_inline
    def __len__(self) -> Int:
        return self.n

    @always_inline
    def __getitem__(self, i: Int) -> Int:
        return Int(
            Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=self.addr)[
                unsafe_offset=i
            ]
        )

    def to_list(self) -> List[Int]:
        var out = List[Int](capacity=self.n)
        for i in range(self.n):
            out.append(self[i])
        return out^


def v_int_list_is_none(v: Value) -> Bool:
    return v.tag == TAG_NONE


struct DoubleList(Copyable, Movable, Sized):
    var addr: Int
    var n: Int

    def __init__(out self, v: Value) raises:
        if v.tag == TAG_NONE:
            self.addr = 0
            self.n = 0
            return
        if v.tag != TAG_DOUBLE_LIST:
            raise Error("expected a float[] argument, got record tag ", v.tag)
        self.addr = Int(v.a)
        self.n = Int(v.len)

    def __len__(self) -> Int:
        return self.n

    def __getitem__(self, i: Int) -> Float64:
        return Pointer[Float64, MutUntrackedOrigin](
            unsafe_from_address=self.addr
        )[unsafe_offset=i]


# --- tensor views -------------------------------------------------------------


struct T(Copyable, Movable):
    """What an op needs about one tensor, read once from its `at::Tensor*`.

    Shapes and strides use the TensorSpec convention: rank <= MAX_RANK,
    leading-padded (1 / 0). `device` is the mojo index, -1 for a tensor that
    lives elsewhere (a CPU tensor handed to `_copy_from`).
    """

    var h: Int
    var ptr: Int
    var rank: Int
    var shape: IndexList[MAX_RANK]
    var strides: IndexList[MAX_RANK]
    var offset: Int
    var numel: Int
    var stype: Int32
    var dtype: DType
    var itemsize: Int
    var contig: Bool
    var device: Int
    var device_type: Int  # torch DeviceType (DEVICE_TYPE_PRIVATEUSE1, DEVICE_TYPE_CPU, ...)

    def __init__(out self, h: Int) raises:
        self.h = h
        self.ptr = external_call["tmb_tensor_data_ptr", Int](h)
        self.rank = Int(external_call["tmb_tensor_dim", Int64](h))
        if self.rank > MAX_RANK:
            raise Error(
                "tensor rank ",
                self.rank,
                " exceeds the mojo device limit of ",
                MAX_RANK,
            )
        self.shape = IndexList[MAX_RANK](1)
        self.strides = IndexList[MAX_RANK](0)
        if self.rank > 0:
            var sizes = Pointer[Int64, MutUntrackedOrigin](
                unsafe_from_address=external_call["tmb_tensor_sizes", Int](h)
            )
            var strides = Pointer[Int64, MutUntrackedOrigin](
                unsafe_from_address=external_call["tmb_tensor_strides", Int](h)
            )
            var pad = MAX_RANK - self.rank
            for i in range(self.rank):
                self.shape[pad + i] = Int(sizes[unsafe_offset=i])
                self.strides[pad + i] = Int(strides[unsafe_offset=i])
        self.offset = Int(external_call["tmb_tensor_storage_offset", Int64](h))
        self.numel = Int(external_call["tmb_tensor_numel", Int64](h))
        self.stype = external_call["tmb_tensor_dtype", Int32](h)
        self.dtype = max_dtype(self.stype)
        self.itemsize = dtype_itemsize(self.dtype)
        self.contig = external_call["tmb_tensor_is_contiguous", Int32](h) != 0
        self.device = Int(external_call["tmb_tensor_device_index", Int32](h))
        self.device_type = Int(
            external_call["tmb_tensor_device_type", Int32](h)
        )

    @always_inline
    def dim(self, i: Int) -> Int:
        """Logical size of dim i (negative i counts from the end)."""
        var k = i + self.rank if i < 0 else i
        return self.shape[MAX_RANK - self.rank + k]

    @always_inline
    def stride(self, i: Int) -> Int:
        var k = i + self.rank if i < 0 else i
        return self.strides[MAX_RANK - self.rank + k]

    def spec(self, ctx_ptr: Int) -> TensorSpec:
        return TensorSpec(
            self.ptr,
            self.rank,
            self.shape,
            self.strides,
            self.offset,
            self.dtype,
            self.itemsize,
            self.numel,
            self.contig,
            ctx_ptr,
        )

    def on_mojo(self) -> Bool:
        return self.device_type == DEVICE_TYPE_PRIVATEUSE1

    def on_cpu(self) -> Bool:
        return self.device_type == DEVICE_TYPE_CPU

    def storage_ctx(self) -> Int:
        """The allocation handle behind the storage (0 if not ours)."""
        return external_call["tmb_tensor_storage_ctx", Int](self.h)

    def requires_grad(self) -> Bool:
        return external_call["tmb_tensor_requires_grad", Int32](self.h) != 0

    def bump_version(self):
        """Autograd's version counter. The dispatcher's ADInplaceOrView kernel
        bumps a `Tensor(a!)` argument itself but not the members of a
        `Tensor(a!)[]`, so ops mutating tensor lists call this per tensor."""
        external_call["tmb_tensor_bump_version", NoneType](self.h)

    def storage_ptr(self) -> Int:
        return external_call["tmb_tensor_storage_data_ptr", Int](self.h)

    def storage_nbytes(self) -> Int:
        return Int(external_call["tmb_tensor_storage_nbytes", Int64](self.h))

    def same_shape(self, other: T) -> Bool:
        if self.rank != other.rank:
            return False
        for i in range(self.rank):
            if self.dim(i) != other.dim(i):
                return False
        return True

    def logical_shape(self) -> List[Int]:
        var out = List[Int](capacity=self.rank)
        for i in range(self.rank):
            out.append(self.dim(i))
        return out^


def v_tensor(v: Value) raises -> T:
    if v.tag != TAG_TENSOR and v.tag != TAG_TENSOR_REF:
        raise Error("expected a Tensor argument, got record tag ", v.tag)
    return T(Int(v.a))


def v_opt_tensor(v: Value) raises -> Optional[T]:
    if v.tag == TAG_NONE:
        return None
    return v_tensor(v)


def v_tensor_list(v: Value) raises -> List[T]:
    """`Tensor[]` / `Tensor?[]` (None entries are skipped, see v_opt_tensor_list).
    """
    var out = List[T]()
    if v.tag == TAG_NONE:
        return out^
    if v.tag != TAG_TENSOR_LIST and v.tag != TAG_OPT_TENSOR_LIST:
        raise Error("expected a Tensor[] argument, got record tag ", v.tag)
    var ptrs = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
    out.reserve(Int(v.len))
    for i in range(Int(v.len)):
        var h = ptrs[unsafe_offset=i]
        if h != 0:
            out.append(T(h))
    return out^


def v_opt_tensor_list_present(v: Value) -> List[Bool]:
    var out = List[Bool]()
    if v.tag != TAG_OPT_TENSOR_LIST and v.tag != TAG_TENSOR_LIST:
        return out^
    var ptrs = Pointer[Int, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
    for i in range(Int(v.len)):
        out.append(ptrs[unsafe_offset=i] != 0)
    return out^


# --- results ------------------------------------------------------------------


def ret_tensor(rets: Values, i: Int, t: T):
    """Hand an owned handle (from new_tensor/view) back as result i."""
    rets[unsafe_offset=i] = Value(TAG_TENSOR, 0, Int64(t.h), 0)


def ret_owned(rets: Values, i: Int, mut o: Owned):
    rets[unsafe_offset=i] = Value(TAG_TENSOR, 0, Int64(o.take().h), 0)


def ret_ref(rets: Values, i: Int, t: T):
    """Return an input tensor itself (in-place ops, `out=` variants)."""
    rets[unsafe_offset=i] = Value(TAG_TENSOR_REF, 0, Int64(t.h), 0)


def ret_int(rets: Values, i: Int, x: Int):
    rets[unsafe_offset=i] = Value(TAG_INT, 0, Int64(x), 0)


def ret_bool(rets: Values, i: Int, x: Bool):
    rets[unsafe_offset=i] = Value(TAG_BOOL, 0, Int64(1) if x else Int64(0), 0)


def ret_f64(rets: Values, i: Int, x: Float64):
    rets[unsafe_offset=i] = Value(TAG_DOUBLE, 0, f64_bits(x), 0)


def ret_scalar_int(rets: Values, i: Int, x: Int):
    rets[unsafe_offset=i] = Value(TAG_SCALAR_INT, 0, Int64(x), 0)


def ret_scalar_f64(rets: Values, i: Int, x: Float64):
    rets[unsafe_offset=i] = Value(TAG_SCALAR_DOUBLE, 0, f64_bits(x), 0)


def ret_scalar_bool(rets: Values, i: Int, x: Bool):
    rets[unsafe_offset=i] = Value(
        TAG_SCALAR_BOOL, 0, Int64(1) if x else Int64(0), 0
    )


def ret_tensor_list(rets: Values, i: Int, ts: List[T]):
    """Owned handles; the adapter frees the malloc'd array."""
    var n = max(len(ts), 1)
    var arr = external_call["malloc", Pointer[Int, MutUntrackedOrigin]](n * 8)
    for k in range(len(ts)):
        arr[unsafe_offset=k] = ts[k].h
    rets[unsafe_offset=i] = Value(
        TAG_TENSOR_LIST, Int32(len(ts)), Int64(Int(arr)), 0
    )


# --- tensor creation (through the shim's allocator, no dispatcher) ----------


def contiguous_strides(
    shape: IndexList[MAX_RANK], rank: Int
) -> IndexList[MAX_RANK]:
    var strides = IndexList[MAX_RANK](0)
    var acc = 1
    for k in range(rank):
        var i = MAX_RANK - 1 - k
        strides[i] = acc
        acc *= shape[i]
    return strides


def strides_equal(
    a: IndexList[MAX_RANK], b: IndexList[MAX_RANK], rank: Int
) -> Bool:
    var pad = MAX_RANK - rank
    for i in range(rank):
        if a[pad + i] != b[pad + i]:
            return False
    return True


def memory_format_name(mf: Int) -> String:
    """`c10::MemoryFormat` as torch spells it in an error message."""
    if mf == MEMORY_FORMAT_CONTIGUOUS:
        return "Contiguous"
    if mf == MEMORY_FORMAT_PRESERVE:
        return "Preserve"
    if mf == MEMORY_FORMAT_CHANNELS_LAST:
        return "ChannelsLast"
    if mf == MEMORY_FORMAT_CHANNELS_LAST_3D:
        return "ChannelsLast3d"
    return String(mf)


def _channels_last_strides(
    shape: IndexList[MAX_RANK], rank: Int
) -> IndexList[MAX_RANK]:
    """NHWC (rank 4) / NDHWC (rank 5): channel innermost, then the spatial
    dims right to left, batch outermost."""
    var strides = IndexList[MAX_RANK](0)
    var pad = MAX_RANK - rank
    var acc = 1
    strides[pad + 1] = acc
    acc *= shape[pad + 1]
    for d in range(rank - 1, 1, -1):
        strides[pad + d] = acc
        acc *= shape[pad + d]
    strides[pad] = acc
    return strides


def strides_for_memory_format(
    shape: IndexList[MAX_RANK], rank: Int, mf: Int
) raises -> IndexList[MAX_RANK]:
    """torch's `TensorImpl::empty_tensor_restride`: the strides a MemoryFormat
    asks a fresh allocation for, rank checks and messages included.

    `Preserve` has no answer here -- it is a property of an INPUT tensor, and
    torch rejects it in this position too."""
    if mf == MEMORY_FORMAT_CONTIGUOUS:
        return contiguous_strides(shape, rank)
    if mf == MEMORY_FORMAT_CHANNELS_LAST:
        if rank != 4:
            raise Error("required rank 4 tensor to use channels_last format")
        return _channels_last_strides(shape, rank)
    if mf == MEMORY_FORMAT_CHANNELS_LAST_3D:
        if rank != 5:
            raise Error("required rank 5 tensor to use channels_last_3d format")
        return _channels_last_strides(shape, rank)
    raise Error("unsupported memory format ", memory_format_name(mf))


def is_dense(
    shape: IndexList[MAX_RANK], strides: IndexList[MAX_RANK], rank: Int
) -> Bool:
    """c10's `_compute_non_overlapping_and_dense`: the elements cover
    `numel * itemsize` bytes from the first one, in SOME permutation of the
    dims (contiguous and channels-last both qualify)."""
    if rank == 0:
        return True
    var pad = MAX_RANK - rank
    if rank == 1:
        return shape[pad] < 2 or strides[pad] == 1
    # Dims sorted by stride ascending, the size-1 (and size-0) ones last:
    # only those carry a meaningful stride for the density chain below.
    var perm = IndexList[MAX_RANK](0)
    for i in range(rank):
        perm[i] = i
    for i in range(1, rank):
        var key = perm[i]
        var j = i - 1
        while j >= 0 and _stride_order_less(shape, strides, pad, key, perm[j]):
            perm[j + 1] = perm[j]
            j -= 1
        perm[j + 1] = key
    var require = 1
    for i in range(rank):
        var d = perm[i]
        if shape[pad + d] < 2:
            return True
        if strides[pad + d] != require:
            return False
        require *= shape[pad + d]
    return True


def _stride_order_less(
    shape: IndexList[MAX_RANK],
    strides: IndexList[MAX_RANK],
    pad: Int,
    a: Int,
    b: Int,
) -> Bool:
    if shape[pad + a] < 2:
        return False
    if shape[pad + b] < 2:
        return True
    return strides[pad + a] < strides[pad + b]


def dense_strides_like(
    shape: IndexList[MAX_RANK], strides: IndexList[MAX_RANK], rank: Int
) -> IndexList[MAX_RANK]:
    """torch's `infer_dense_strides`: a DENSE layout that keeps the input's
    dim order -- what a `preserve_format` request falls back to when the input
    is not itself dense, and what makes `x.t()[:, ::2].clone()` come back
    transposed rather than row-major."""
    if rank <= 1:
        return contiguous_strides(shape, rank)
    var pad = MAX_RANK - rank
    # Insertion sort by stride, ambiguous comparisons (a zero stride, equal
    # strides with equal sizes) leaving the order alone, exactly as
    # TensorIterator propagates strides.
    var perm = IndexList[MAX_RANK](0)
    for i in range(rank):
        perm[i] = rank - 1 - i
    for i in range(1, rank):
        var dim1 = i
        var dim0 = i - 1
        while dim0 >= 0:
            var cmp = _should_swap(shape, strides, pad, perm[dim0], perm[dim1])
            if cmp > 0:
                var tmp = perm[dim0]
                perm[dim0] = perm[dim1]
                perm[dim1] = tmp
                dim1 = dim0
            elif cmp < 0:
                break
            dim0 -= 1
    var out = IndexList[MAX_RANK](0)
    var acc = 1
    for i in range(rank):
        var d = perm[i]
        out[pad + d] = acc
        acc *= shape[pad + d]
    return out


def _should_swap(
    shape: IndexList[MAX_RANK],
    strides: IndexList[MAX_RANK],
    pad: Int,
    d0: Int,
    d1: Int,
) -> Int:
    var s0 = strides[pad + d0]
    var s1 = strides[pad + d1]
    if s0 == 0 or s1 == 0:
        return 0
    if s0 < s1:
        return -1
    if s0 > s1:
        return 1
    if shape[pad + d0] > shape[pad + d1]:
        return 1
    return 0


def new_strided(
    shape: IndexList[MAX_RANK],
    strides: IndexList[MAX_RANK],
    rank: Int,
    stype: Int32,
    device: Int,
) raises -> T:
    var sizes = InlineArray[Int64, MAX_RANK](fill=0)
    var strd = InlineArray[Int64, MAX_RANK](fill=0)
    var pad = MAX_RANK - rank
    for i in range(rank):
        sizes[i] = Int64(shape[pad + i])
        strd[i] = Int64(strides[pad + i])
    var h: Int = 0
    check(
        external_call["tmb_empty_strided", Int32](
            Int64(rank),
            sizes.unsafe_ptr(),
            strd.unsafe_ptr(),
            stype,
            Int32(device),
            Pointer(to=h),
        ),
        "tmb_empty_strided",
    )
    return T(h)


def new_tensor(
    shape: IndexList[MAX_RANK], rank: Int, stype: Int32, device: Int
) raises -> T:
    return new_strided(
        shape, contiguous_strides(shape, rank), rank, stype, device
    )


def new_like(t: T) raises -> T:
    """Contiguous, same logical shape / dtype / device."""
    return new_tensor(t.shape, t.rank, t.stype, t.device)


def new_like_dtype(t: T, stype: Int32) raises -> T:
    return new_tensor(t.shape, t.rank, stype, t.device)


def new_scalar(stype: Int32, device: Int) raises -> T:
    return new_tensor(IndexList[MAX_RANK](1), 0, stype, device)


def view_strided(
    base: T,
    shape: IndexList[MAX_RANK],
    strides: IndexList[MAX_RANK],
    rank: Int,
    offset: Int,
) raises -> T:
    """A zero-copy view over base's storage (an owned handle to return)."""
    var sizes = InlineArray[Int64, MAX_RANK](fill=0)
    var strd = InlineArray[Int64, MAX_RANK](fill=0)
    var pad = MAX_RANK - rank
    for i in range(rank):
        sizes[i] = Int64(shape[pad + i])
        strd[i] = Int64(strides[pad + i])
    var h: Int = 0
    check(
        external_call["tmb_as_strided", Int32](
            base.h,
            Int64(rank),
            sizes.unsafe_ptr(),
            strd.unsafe_ptr(),
            Int64(offset),
            Pointer(to=h),
        ),
        "tmb_as_strided",
    )
    return T(h)


def set_sizes_strides(
    t: T,
    shape: IndexList[MAX_RANK],
    strides: IndexList[MAX_RANK],
    rank: Int,
    offset: Int,
) raises:
    var sizes = InlineArray[Int64, MAX_RANK](fill=0)
    var strd = InlineArray[Int64, MAX_RANK](fill=0)
    var pad = MAX_RANK - rank
    for i in range(rank):
        sizes[i] = Int64(shape[pad + i])
        strd[i] = Int64(strides[pad + i])
    check(
        external_call["tmb_tensor_set_sizes_strides", Int32](
            t.h,
            Int64(rank),
            sizes.unsafe_ptr(),
            strd.unsafe_ptr(),
            Int64(offset),
        ),
        "tmb_tensor_set_sizes_strides",
    )


struct Owned(Movable):
    """An output tensor until it is handed to torch: released on every path
    that does not `take()` it (a declined route, a failed kernel build)."""

    var t: T
    var live: Bool

    def __init__(out self, var t: T):
        self.t = t^
        self.live = True

    def take(mut self) -> T:
        self.live = False
        return self.t.copy()

    def __deinit__(deinit self):
        if self.live:
            release(self.t.h)


def own(var t: T) -> Owned:
    return Owned(t^)


def own_if_new(var result: T, original: T) -> Owned:
    """`Owned` over `result`, live only when it is a fresh allocation
    distinct from `original`.

    `ops_common.contiguous` / `cast_to` return their input unchanged when it
    already has the requested layout and dtype, so a caller that wrapped the
    result in `own()` unconditionally would release a handle it only
    borrowed (an op's `self`). This releases it exactly when it was
    allocated, on the raising paths too."""
    var fresh = result.h != original.h
    var o = Owned(result^)
    o.live = fresh
    return o^


def retain(t: T) -> Int:
    return external_call["tmb_tensor_retain", Int](t.h)


def release(h: Int):
    external_call["tmb_tensor_release", NoneType](h)


def cpu_empty(shape: IndexList[MAX_RANK], rank: Int, stype: Int32) raises -> T:
    var sizes = InlineArray[Int64, MAX_RANK](fill=0)
    var pad = MAX_RANK - rank
    for i in range(rank):
        sizes[i] = Int64(shape[pad + i])
    var h: Int = 0
    check(
        external_call["tmb_cpu_empty", Int32](
            Int64(rank), sizes.unsafe_ptr(), stype, Pointer(to=h)
        ),
        "tmb_cpu_empty",
    )
    return T(h)


def default_dtype() -> Int32:
    return external_call["tmb_default_dtype", Int32]()


# --- op entry -----------------------------------------------------------------
comptime OpFn = def(Values, Int, Values, Int) raises thin -> None


def op_entry[
    op: OpFn
](
    ctx: Int,
    name: Pointer[c_char, MutUntrackedOrigin],
    overload: Pointer[c_char, MutUntrackedOrigin],
    args: Values,
    n_args: Int32,
    rets: Values,
    n_rets: Int32,
) abi("C") -> Int32:
    """The boxed-kernel signature (tmb.h TmbKernelFn) around one Mojo op.
    rc 1 = RuntimeError, 2 = NotImplementedError (a declined call)."""
    try:
        op(args, Int(n_args), rets, Int(n_rets))
        return 0
    except e:
        var msg = String(e)
        if msg.startswith(UNSUPPORTED_PREFIX):
            set_shim_error(
                String(msg[byte = UNSUPPORTED_PREFIX.byte_length() :])
            )
            return 2
        set_shim_error(msg)
        return 1


def op_address[op: OpFn]() -> Int:
    var f: def(
        Int,
        Pointer[c_char, MutUntrackedOrigin],
        Pointer[c_char, MutUntrackedOrigin],
        Values,
        Int32,
        Values,
        Int32,
    ) thin abi("C") -> Int32 = op_entry[op]
    return Pointer(to=f).unsafe_bitcast[Int]()[]


# --- calling other aten ops (composites, CPU-side work) ------------------------


def tensor_arg(t: T) -> Value:
    return Value(TAG_TENSOR, 0, Int64(t.h), 0)


def bool_arg(b: Bool) -> Value:
    return Value(TAG_BOOL, 0, Int64(1) if b else Int64(0), 0)


def int_arg(x: Int) -> Value:
    return Value(TAG_INT, 0, Int64(x), 0)


def none_arg() -> Value:
    return Value(TAG_NONE, 0, 0, 0)


struct Results(Movable):
    """The result records of a `call_op`, owning everything the shim
    allocated in them: every handle the caller does not `take_tensor` is
    released when this dies, on the raising paths too.

    `tmb_call_op` hands back a FRESH `at::Tensor*` for every tensor result --
    including the one an in-place op returns, which is just another reference
    to an argument -- so a discarded result record is a leaked handle.
    """

    var records: List[Value]

    def __init__(out self, var records: List[Value]):
        self.records = records^

    def __len__(self) -> Int:
        return len(self.records)

    def __getitem__(self, i: Int) -> Value:
        return self.records[i].copy()

    def take_tensor(mut self, i: Int) raises -> T:
        """Result `i` as an owned tensor handle; this container forgets it."""
        var r = self.records[i].copy()
        if r.tag != TAG_TENSOR and r.tag != TAG_TENSOR_REF:
            raise Error(
                "result ", i, " is not a Tensor (record tag ", r.tag, ")"
            )
        self.records[i] = none_arg()
        return T(Int(r.a))

    def __deinit__(deinit self):
        for r in self.records:
            if r.tag == TAG_TENSOR:
                release(Int(r.a))
            elif r.tag == TAG_TENSOR_LIST:
                var ptrs = Pointer[Int, MutUntrackedOrigin](
                    unsafe_from_address=Int(r.a)
                )
                for i in range(Int(r.len)):
                    release(ptrs[unsafe_offset=i])
                _free(Int(r.a))
            elif r.tag == TAG_INT_LIST:
                _free(Int(r.a))


def _free(addr: Int):
    libc_free(
        Pointer[NoneType, MutUntrackedOrigin](
            unsafe_from_address=addr
        ).unsafe_origin_cast[MutAnyOrigin]()
    )


def call_op_raw(
    op: String,
    overload: String,
    args: Values,
    n_args: Int,
    rets: Values,
    n_rets: Int,
) raises:
    """Run any aten op through torch's dispatcher (shim `tmb_call_op`) over
    caller-owned record arrays. THE implementation: `call_op` below is the
    List-based front end, and nothing else calls `tmb_call_op` directly.

    What an op uses to reach a neighbouring op's kernel or ATen's own
    composite: the records are the same ones a kernel gets, tensor arguments
    are borrowed and tensor results come back as owned handles. Dispatch is on
    the arguments, so an op must never call *itself* this way. A declining
    kernel comes back as `unsupported` (rc 2) and keeps that prefix, so a
    caller with another route can tell it apart from a real failure.
    """
    var o = String(op)
    var ov = String(overload)
    var rc = external_call["tmb_call_op", Int32](
        o.as_c_string_slice().unsafe_ptr(),
        ov.as_c_string_slice().unsafe_ptr(),
        args,
        Int32(n_args),
        rets,
        Int32(n_rets),
    )
    if rc == 2:
        unsupported(shim_error())
    if rc != 0:
        raise Error(op, ": ", shim_error())


def call_op(
    op: String, overload: String, var args: List[Value], n_rets: Int
) raises -> Results:
    """`call_op_raw` over a `List[Value]` of arguments in schema order (exact
    arity: the dispatcher checks it against the op's schema). `op` must be
    namespace-qualified (`"aten::normal_"`), `overload` is `""` for the
    default one. The results own their handles -- see `Results`."""
    var rets = List[Value](capacity=max(n_rets, 1))
    for _ in range(n_rets):
        rets.append(none_arg())
    var n_args = len(args)
    call_op_raw(
        op,
        overload,
        args.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        n_args,
        rets.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](),
        n_rets,
    )
    _ = args^  # the records are read by the call; keep them alive across it
    return Results(rets^)
