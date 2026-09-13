"""aten ops: matmul group — mm, bmm, addmm, linear, linear_backward, addr and
the convolution forward.

The route cascade is the old fast path's (aten_fast.py), unchanged:

    gemm16 (bf16/f16 tensor cores, CUDA sm_90a)
      -> tf32 (fp32 tensor cores, CUDA sm_90a, opt-in)
      -> the generic matmul_ops spec kernels (MatmulSpec / MatmulBiasSpec /
         BmmSpec), which own every other target, dtype and layout.

Each step is host metadata only: a route whose operands do not fit returns
None and the next one runs; when the last one declines, the op raises
NotImplementedError through `unsupported`, exactly where the old path
returned NOT_HANDLED.
"""
from std.ffi import _get_global_or_null, external_call
from std.memory.alloc import unsafe_alloc
from std.os.path import exists
from std.utils import IndexList

from max.gpu.host import DeviceAttribute

from abi import (
    IntList,
    T,
    TAG_BOOL_LIST,
    TAG_NONE,
    TAG_SCALAR_INT,
    TAG_TENSOR,
    UNSUPPORTED_PREFIX,
    Value,
    Values,
    contiguous_strides,
    dtype_code,
    new_tensor,
    own,
    release,
    ret_owned,
    ret_ref,
    ret_tensor,
    unsupported,
    v_bool,
    v_f64,
    v_int,
    v_scalar_is_bool,
    v_tensor,
    view_strided,
)
from device import ctx_for, ctx_ptr
from kernels import KernelCall, loader
from op_utils import MAX_RANK
from ops_common import (
    call_op_raw,
    check_out,
    contiguous,
    copy_strided_into,
    fill_value,
    resize_out,
)
from registry import Site, impl, op_address_of


# --- small shape helpers ------------------------------------------------------


def _index_list(dims: List[Int]) raises -> IndexList[MAX_RANK]:
    if len(dims) > MAX_RANK:
        raise Error("tensor rank ", len(dims), " exceeds the mojo device limit")
    var shape = IndexList[MAX_RANK](1)
    var pad = MAX_RANK - len(dims)
    for i in range(len(dims)):
        shape[pad + i] = dims[i]
    return shape


def _new(dims: List[Int], stype: Int32, device: Int) raises -> T:
    """A fresh contiguous tensor of this logical shape (uninitialized)."""
    return new_tensor(_index_list(dims), len(dims), stype, device)


def _zeros(dims: List[Int], stype: Int32, device: Int) raises -> T:
    var t = own(_new(dims, stype, device))
    fill_value(t.t, 0.0)
    return t.take()


def _empty_result(stype: Int32, device: Int) raises -> T:
    """Placeholder for an output computed only when requested."""
    return _new([0], stype, device)


def _ret_undefined(rets: Values, i: Int):
    """An output the caller did not ask for: a None record, which the shim
    hands back as an undefined Tensor exactly like ATen's own backward
    kernels (the generated autograd node never reads it)."""
    rets[unsafe_offset=i] = Value(TAG_NONE, 0, 0, 0)


def _view(t: T, dims: List[Int]) raises -> T:
    """A contiguous reshape view of a contiguous `t` (an owned handle)."""
    var shape = _index_list(dims)
    return view_strided(
        t, shape, contiguous_strides(shape, len(dims)), len(dims), t.offset
    )


def _prod(dims: List[Int]) -> Int:
    var p = 1
    for d in dims:
        p *= d
    return p


def _leading_dims(t: T) -> List[Int]:
    """t's logical shape without its last dimension."""
    var out = List[Int]()
    for i in range(t.rank - 1):
        out.append(t.dim(i))
    return out^


def _is_float(dt: DType) -> Bool:
    """The dtypes every matmul-family kernel in this repo covers."""
    return dt == DType.float32 or dt == DType.bfloat16 or dt == DType.float16


struct Tmp(Movable):
    """A contiguous form of a tensor, released only when it is a fresh copy
    (`contiguous` hands back the input's own handle when it already is)."""

    var t: T
    var fresh: Bool

    def __init__(out self, src: T) raises:
        self.t = contiguous(src)
        self.fresh = self.t.h != src.h

    def __deinit__(deinit self):
        if self.fresh:
            release(self.t.h)


# --- route gates (cached: these sit in front of every matmul) -----------------

comptime MATMUL_CACHE = "TMB_MATMUL_CACHE"
comptime SLOT_GEMM16 = 0
comptime SLOT_TF32 = 1
comptime SLOT_ARCH0 = 2
comptime MAX_CACHED_DEVICES = 32
comptime CACHE_SLOTS = SLOT_ARCH0 + MAX_CACHED_DEVICES


def _cache() -> Pointer[Int, MutUntrackedOrigin]:
    """Process-global verdict cache: bridge availability (stat calls) and the
    per-device architecture query, each answered once. Both sit in front of
    every matmul, where re-answering them costs more than everything else the
    host does."""
    var p = _get_global_or_null(MATMUL_CACHE)
    if p:
        return p.value().unsafe_bitcast[Int]()
    var box = unsafe_alloc[Int](CACHE_SLOTS)
    for i in range(CACHE_SLOTS):
        box[unsafe_offset=i] = -1
    external_call["KGEN_CompilerRT_InsertGlobal", NoneType](
        StringSlice(MATMUL_CACHE), box.unsafe_bitcast[NoneType]()
    )
    return box


def _bridge_available(
    slot: Int, family: String, files: List[String]
) raises -> Bool:
    """Whether an optional bridge and every source it imports are present.

    The native loader builds a family from its entry file, so "available" is
    "the files exist": a partial checkout must not pay for a predictably
    failing compile in front of an ordinary eager matmul.
    """
    var c = _cache()
    if c[unsafe_offset=slot] < 0:
        var dir = loader()[].kernels_dir + "/" + family + "/"
        var ok = True
        for f in files:
            if not exists(dir + f):
                ok = False
        c[unsafe_offset=slot] = 1 if ok else 0
    return c[unsafe_offset=slot] == 1


def _gemm16_available() raises -> Bool:
    return _bridge_available(
        SLOT_GEMM16,
        "gemm16_matmul_ops",
        [
            "gemm16_matmul_ops.mojo",
            "gemm16_v3_kernels.mojo",
            "gemm16_tn_v4_kernels.mojo",
            "gemm16_kernels.mojo",
        ],
    )


def _tf32_available() raises -> Bool:
    return _bridge_available(
        SLOT_TF32,
        "tf32_matmul_ops",
        ["tf32_matmul_ops.mojo", "tf32_gemm_kernels.mojo"],
    )


def _sm90_cuda(device: Int) raises -> Bool:
    """The H100-class CUDA target both tensor-core bridges are written for.

    The old path compared `Device.architecture_name == "sm_90a"`; compute
    capability 9.0 is the same set of parts and is a runtime query, so a
    backend build cached from another box cannot claim this one's GPU.
    """
    if device < 0 or device >= MAX_CACHED_DEVICES:
        return False
    var c = _cache()
    var slot = SLOT_ARCH0 + device
    if c[unsafe_offset=slot] < 0:
        var ok = False
        var ctx = ctx_for(device)
        if ctx.api() == "cuda":
            var major = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MAJOR
            )
            var minor = ctx.get_attribute(
                DeviceAttribute.COMPUTE_CAPABILITY_MINOR
            )
            ok = major == 9 and minor == 0
        _ = ctx
        c[unsafe_offset=slot] = 1 if ok else 0
    return c[unsafe_offset=slot] == 1


def _tf32_enabled() -> Bool:
    """Whether the TF32 bridge may run: a numerics decision (TF32 drops
    mantissa bits), taken from `torch.get_float32_matmul_precision()` exactly
    as the old path did: any setting but "highest" (torch's default) allows
    it."""
    return external_call["tmb_float32_matmul_precision", Int32]() != 0


# --- GEMM operands ------------------------------------------------------------


@fieldwise_init
struct Mat(Copyable, ImplicitlyCopyable, Movable):
    """One dense 2-D GEMM operand: its logical (rows, cols) plus the physical
    transpose flag its strides encode. `t == 1` means the buffer holds the
    transposed matrix, which every bridge reads for free."""

    var ptr: Int
    var rows: Int
    var cols: Int
    var t: Int


@fieldwise_init
struct Mat3(Copyable, ImplicitlyCopyable, Movable):
    """One batched GEMM operand: per-matrix layout plus the runtime batch
    stride in elements (0 = a broadcast matrix shared by every item)."""

    var ptr: Int
    var batch: Int
    var rows: Int
    var cols: Int
    var t: Int
    var bstride: Int


def _dense_2d(t: T) -> Optional[Mat]:
    """An exact dense 2-D layout, row-major or transposed, else None."""
    if t.rank != 2:
        return None
    var rows = t.dim(0)
    var cols = t.dim(1)
    if t.stride(0) == cols and t.stride(1) == 1:
        return Mat(t.ptr, rows, cols, 0)
    if t.stride(0) == 1 and t.stride(1) == rows:
        return Mat(t.ptr, rows, cols, 1)
    return None


def _flat_2d(t: T) -> Optional[Mat]:
    """The matrix a rank >= 2 projection input already is: a contiguous
    higher-rank operand is the same row-major matrix once its leading
    dimensions are flattened, so only metadata changes."""
    if t.rank == 2:
        return _dense_2d(t)
    if t.rank < 2 or not t.contig:
        return None
    var k = t.dim(t.rank - 1)
    var m = t.numel // k if k > 0 else 0
    return Mat(t.ptr, m, k, 0)


def _batched_3d(t: T) -> Optional[Mat3]:
    """Dense matrices separated by a non-overlapping batch stride, or a
    broadcast operand (batch stride 0 — one matrix shared by every item, what
    `expand()` produces). Padding between matrices is fine; any other
    overlapping batch stride is not."""
    if t.rank != 3:
        return None
    var batch = t.dim(0)
    var rows = t.dim(1)
    var cols = t.dim(2)
    var bs = t.stride(0)
    var rs = t.stride(1)
    var cs = t.stride(2)
    var flag: Int
    if cs == 1 and (rows == 1 or rs == cols):
        flag = 0
    elif rs == 1 and (cols == 1 or cs == rows):
        flag = 1
    else:
        return None
    if batch <= 0 or rows <= 0 or cols <= 0:
        return None
    if bs != 0 and bs < rows * cols:
        return None
    return Mat3(t.ptr, batch, rows, cols, flag, bs)


# --- the two tensor-core bridges ---------------------------------------------


def _bias_fits(bias: T, n: Int, stype: Int32, device: Int) -> Bool:
    return (
        bias.device == device
        and bias.stype == stype
        and bias.rank == 1
        and bias.dim(0) == n
        and bias.contig
    )


def _gemm_bridge(
    family: StaticString,
    op: StaticString,
    a: Mat,
    b: Mat,
    transpose_b: Bool,
    bias: Optional[T],
    dt: DType,
    stype: Int32,
    device: Int,
    out_dims: List[Int],
) raises -> Optional[T]:
    """One dense 2-D GEMM through gemm16 / tf32: C = op(A) @ op(B) [+ bias].

    Both bridges take this ABI verbatim (11 slots); only the family, the OP
    name and the dtype they accept differ.
    """
    var m = a.rows
    var k = a.cols
    var n = b.rows if transpose_b else b.cols
    var rhs_k = b.cols if transpose_b else b.rows
    if m <= 0 or n <= 0 or k <= 0 or rhs_k != k:
        return None
    var has_bias = Bool(bias)
    if has_bias and not _bias_fits(bias.value(), n, stype, device):
        return None
    var dims = List[Int]()
    if len(out_dims) == 0:
        dims.append(m)
        dims.append(n)
    else:
        dims = out_dims.copy()
        if dims[len(dims) - 1] != n or _prod(dims) != m * n:
            return None
    # TT (both flags set) is launched directly rather than operand-swapped
    # into NN: the TT routes compute C straight into this contiguous row-major
    # buffer, so `.stride()` matches what CUDA torch returns for the same call
    # (a swapped NN kernel would hand back a column-major C).
    var out = own(_new(dims, stype, device))
    var ctx = ctx_for(device)
    var call = KernelCall(String(family), String(op))
    call.arg_dtype(0, dt)
    call.arg_dtype(1, dt)
    if has_bias:
        call.arg_dtype(2, dt)
    call.out_dtype(dt)
    call.flag("HAS_BIAS", 1 if has_bias else 0)
    call.flag("TRANSPOSE_B", 1 if transpose_b else 0)
    call.int(out.t.ptr)
    call.int(a.ptr)
    call.int(b.ptr)
    call.int(bias.value().ptr if has_bias else out.t.ptr)
    call.int(m)
    call.int(n)
    call.int(k)
    call.int(a.t)
    call.int(b.t ^ (1 if transpose_b else 0))
    call.int(1 if has_bias else 0)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx
    return out.take()


def _bmm_bridge(
    family: StaticString,
    op: StaticString,
    a: Mat3,
    b: Mat3,
    transpose_b: Bool,
    dt: DType,
    stype: Int32,
    device: Int,
) raises -> Optional[T]:
    """One batched GEMM through gemm16 / tf32 (13 slots, no bias)."""
    var batch = a.batch
    var m = a.rows
    var k = a.cols
    var n = b.rows if transpose_b else b.cols
    var kb = b.cols if transpose_b else b.rows
    if b.batch != batch or kb != k:
        return None
    var out = own(_new([batch, m, n], stype, device))
    var ctx = ctx_for(device)
    var call = KernelCall(String(family), String(op))
    call.arg_dtype(0, dt)
    call.arg_dtype(1, dt)
    call.out_dtype(dt)
    call.flag("TRANSPOSE_B", 1 if transpose_b else 0)
    call.int(out.t.ptr)
    call.int(a.ptr)
    call.int(b.ptr)
    call.int(batch)
    call.int(m)
    call.int(n)
    call.int(k)
    call.int(m * n)
    call.int(a.bstride)
    call.int(b.bstride)
    call.int(a.t)
    call.int(b.t ^ (1 if transpose_b else 0))
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx
    return out.take()


def _gemm16_gate(a: T, b: T) raises -> Bool:
    return (
        (a.dtype == DType.bfloat16 or a.dtype == DType.float16)
        and a.stype == b.stype
        and a.on_mojo()
        and b.on_mojo()
        and a.device == b.device
        and _sm90_cuda(a.device)
        and _gemm16_available()
    )


def _tf32_gate(a: T, b: T) raises -> Bool:
    return (
        a.dtype == DType.float32
        and a.stype == b.stype
        and a.on_mojo()
        and b.on_mojo()
        and a.device == b.device
        and _tf32_enabled()
        and _sm90_cuda(a.device)
        and _tf32_available()
    )


def _try_gemm16_mm(
    a: T, b: T, bias: Optional[T], transpose_b: Bool, out_dims: List[Int]
) raises -> Optional[T]:
    if not _gemm16_gate(a, b):
        return None
    var am = _dense_2d(a)
    var bm = _dense_2d(b)
    if not am or not bm:
        return None
    return _gemm_bridge(
        "gemm16_matmul_ops",
        "Gemm16",
        am.value(),
        bm.value(),
        transpose_b,
        bias,
        a.dtype,
        a.stype,
        a.device,
        out_dims,
    )


def _try_tf32_mm(
    a: T, b: T, bias: Optional[T], transpose_b: Bool, out_dims: List[Int]
) raises -> Optional[T]:
    if not _tf32_gate(a, b):
        return None
    var am = _dense_2d(a)
    var bm = _dense_2d(b)
    if not am or not bm:
        return None
    return _gemm_bridge(
        "tf32_matmul_ops",
        "Tf32GemmF32",
        am.value(),
        bm.value(),
        transpose_b,
        bias,
        a.dtype,
        a.stype,
        a.device,
        out_dims,
    )


def _try_gemm16_bmm(a: T, b: T, transpose_b: Bool) raises -> Optional[T]:
    if not _gemm16_gate(a, b):
        return None
    var am = _batched_3d(a)
    var bm = _batched_3d(b)
    if not am or not bm:
        return None
    return _bmm_bridge(
        "gemm16_matmul_ops",
        "Bmm16",
        am.value(),
        bm.value(),
        transpose_b,
        a.dtype,
        a.stype,
        a.device,
    )


def _try_tf32_bmm(a: T, b: T, transpose_b: Bool) raises -> Optional[T]:
    if not _tf32_gate(a, b):
        return None
    var am = _batched_3d(a)
    var bm = _batched_3d(b)
    if not am or not bm:
        return None
    return _bmm_bridge(
        "tf32_matmul_ops",
        "Tf32BmmF32",
        am.value(),
        bm.value(),
        transpose_b,
        a.dtype,
        a.stype,
        a.device,
    )


def _alignment_favors_split(a: Mat, b: Mat, transpose_b: Bool) -> Bool:
    """Could gemm16 possibly route this shape through a v3/v4 tensor-core
    route (aligned or split-K), regardless of bias?

    Every such route needs m, n and k to each be a multiple of at least 64 —
    the smallest tile any of them uses. When that fails, gemm16 lands on the
    accepted mma.sync kernel either way, so splitting the bias off only pays
    for a second launch: measured, that took an already-good 357x789x333 from
    sub-1.0x to 1.25-1.65x stock. A conservative yes costs at most that one
    launch and never affects correctness.
    """
    var m = a.rows
    var k = a.cols
    var rhs_k = b.cols if transpose_b else b.rows
    var n = b.rows if transpose_b else b.cols
    if rhs_k != k:
        return False
    return m % 64 == 0 and n % 64 == 0 and k % 64 == 0


def _try_gemm16_linear(a: T, w: T, bias: Optional[T]) raises -> Optional[T]:
    """A dense rank >= 2 16-bit projection without copies.

    Every gemm16 tensor-core route declines outright when a bias is present,
    so a fused-bias call would silently fall back to the far slower accepted
    mma.sync kernel (measured 3.6-7.4x stock on deep-K shapes, versus ~1.3x
    for the identical unbiased mm). When the shape could reach a fast route at
    all (`_alignment_favors_split`), compute the bias-free mm and add the bias
    afterwards with the ordinary broadcasting add instead.
    """
    if not _gemm16_gate(a, w):
        return None
    var am = _flat_2d(a)
    var wm = _dense_2d(w)
    if not am or not wm:
        return None
    var dims = _leading_dims(a)
    dims.append(w.dim(0))
    if not bias:
        return _gemm_bridge(
            "gemm16_matmul_ops",
            "Gemm16",
            am.value(),
            wm.value(),
            True,
            None,
            a.dtype,
            a.stype,
            a.device,
            dims,
        )
    if _alignment_favors_split(am.value(), wm.value(), True):
        var mm_out = _gemm_bridge(
            "gemm16_matmul_ops",
            "Gemm16",
            am.value(),
            wm.value(),
            True,
            None,
            a.dtype,
            a.stype,
            a.device,
            dims,
        )
        if mm_out:
            var plain = own(mm_out.value().copy())
            var biased = _try_add(plain.t, bias.value())
            if biased:
                return biased.value().copy()
            # The add declined this bias: drop the unbiased product and fall
            # through to the fused kernel rather than return a biasless one.
            _ = plain^
    # Either the shape can never reach a fast route regardless of bias, or the
    # fast add declined for this bias: the bias-fused kernel is at worst
    # identical, and never drops the bias silently.
    return _gemm_bridge(
        "gemm16_matmul_ops",
        "Gemm16",
        am.value(),
        wm.value(),
        True,
        bias,
        a.dtype,
        a.stype,
        a.device,
        dims,
    )


def _try_tf32_linear(a: T, w: T, bias: Optional[T]) raises -> Optional[T]:
    """A dense rank >= 2 fp32 projection through TF32 without copies."""
    if not _tf32_gate(a, w):
        return None
    var am = _flat_2d(a)
    var wm = _dense_2d(w)
    if not am or not wm:
        return None
    var dims = _leading_dims(a)
    dims.append(w.dim(0))
    return _gemm_bridge(
        "tf32_matmul_ops",
        "Tf32GemmF32",
        am.value(),
        wm.value(),
        True,
        bias,
        a.dtype,
        a.stype,
        a.device,
        dims,
    )


# --- the generic spec kernels -------------------------------------------------


def _declined(e: Error) -> Bool:
    """Whether an error is a route declining its operands rather than a real
    failure: only the explicit `[unsupported]` status counts.

    Everything else -- an allocator refusing device memory, ptxas failing to
    assemble, a driver launch error -- is re-raised with its own message. The
    host gates in `_spec_matmul` already restate every check the spec entry
    makes, so a decline reaching here at all would be a gate this file is
    missing rather than a route to retry."""
    return String(e).startswith(UNSUPPORTED_PREFIX)


def _spec_matmul(
    op: StaticString, a: T, b: T, bias: Optional[T], transpose_b: Int
) raises -> Optional[T]:
    """MatmulSpec / MatmulBiasSpec / BmmSpec over the operands as they lie.

    Strided operands are passed straight through: the kernel reads the strides
    off the TensorSpec and picks a copy-free route where the target has one
    (the gfx942 TN MFMA route and the sm_90 strict-fp32 TN route both read a
    transposed weight-gradient A in place) and scratch-copies otherwise, so
    materializing here would only hide those routes behind a transpose the
    kernel does not need.
    """
    if a.stype != b.stype or not _is_float(a.dtype):
        return None
    if not a.on_mojo() or not b.on_mojo() or a.device != b.device:
        return None
    var dims = List[Int]()
    if op == "BmmSpec":
        if a.rank != 3 or b.rank != 3:
            return None
        var batch = a.dim(0)
        var m = a.dim(1)
        var k = a.dim(2)
        if b.dim(0) != batch:
            return None
        var n = b.dim(1) if transpose_b != 0 else b.dim(2)
        var kb = b.dim(2) if transpose_b != 0 else b.dim(1)
        if kb != k or batch == 0 or m == 0 or n == 0 or k == 0:
            return None
        dims.append(batch)
        dims.append(m)
        dims.append(n)
    else:
        if a.rank < 2 or b.rank != 2:
            return None
        var k = a.dim(a.rank - 1)
        var n = b.dim(0) if transpose_b != 0 else b.dim(1)
        var kb = b.dim(1) if transpose_b != 0 else b.dim(0)
        if kb != k or k == 0 or n == 0:
            return None
        for i in range(a.rank):
            if a.dim(i) == 0:
                return None
        if op == "MatmulBiasSpec":
            if not bias:
                return None
            if (
                bias.value().stype != a.stype
                or bias.value().rank != 1
                or bias.value().dim(0) != n
            ):
                return None
            # The kernel reads the bias as a bare pointer on A's stream.
            if not bias.value().on_mojo() or bias.value().device != a.device:
                raise Error("expected every operand on the same mojo device")
        dims = _leading_dims(a)
        dims.append(n)
    var out = own(_new(dims, a.stype, a.device))
    var ctx = ctx_for(a.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("matmul_ops", String(op))
    call.arg_dtype(0, a.dtype)
    call.arg_dtype(1, b.dtype)
    if op == "MatmulBiasSpec":
        call.arg_dtype(2, bias.value().dtype)
    call.out_dtype(a.dtype)
    call.flag("TRANSPOSE_B", 1 if transpose_b != 0 else 0)
    call.spec(a.spec(cp))
    call.spec(b.spec(cp))
    if op == "MatmulBiasSpec":
        call.spec(bias.value().spec(cp))
    call.int(transpose_b)
    call.spec(out.t.spec(cp))
    try:
        call.run()
    except e:
        if _declined(e):
            return None
        raise e^
    _ = ctx
    return out.take()


# --- neighbouring ops reached through the dispatcher --------------------------


def _try_add(a: T, b: T) raises -> Optional[T]:
    """`a + b` through aten::add.Tensor — the same broadcasting elementwise
    add every other caller of that op gets. None when it declines."""
    var args = InlineArray[Value, 3](fill=Value(TAG_NONE, 0, 0, 0))
    args[0] = Value(TAG_TENSOR, 0, Int64(a.h), 0)
    args[1] = Value(TAG_TENSOR, 0, Int64(b.h), 0)
    args[2] = Value(TAG_SCALAR_INT, 0, 1, 0)
    var rets = InlineArray[Value, 1](fill=Value(TAG_NONE, 0, 0, 0))
    try:
        call_op_raw(
            "aten::add",
            "Tensor",
            Values(unsafe_from_address=Int(args.unsafe_ptr())),
            3,
            Values(unsafe_from_address=Int(rets.unsafe_ptr())),
            1,
        )
    except e:
        _ = args
        if String(e).startswith(UNSUPPORTED_PREFIX):
            return None
        raise e^
    var out = T(Int(rets[0].a))
    _ = args
    _ = rets
    return out^


def _call_1(
    op: StaticString, overload: StaticString, a: Value, b: Value
) raises -> T:
    """A two-argument aten op through the dispatcher, one Tensor result."""
    var args = InlineArray[Value, 2](fill=Value(TAG_NONE, 0, 0, 0))
    args[0] = a.copy()
    args[1] = b.copy()
    var rets = InlineArray[Value, 1](fill=Value(TAG_NONE, 0, 0, 0))
    call_op_raw(
        String(op),
        String(overload),
        Values(unsafe_from_address=Int(args.unsafe_ptr())),
        2,
        Values(unsafe_from_address=Int(rets.unsafe_ptr())),
        1,
    )
    var out = T(Int(rets[0].a))
    _ = args
    _ = rets
    return out^


def _tensor_arg(t: T) -> Value:
    return Value(TAG_TENSOR, 0, Int64(t.h), 0)


def _add_or_raise(a: T, b: T) raises -> T:
    var out = _try_add(a, b)
    if not out:
        unsupported("aten::add.Tensor declined the operands of aten::addr")
    return out.value().copy()


def _opt_tensor_arg(v: Value) raises -> Optional[T]:
    """A `Tensor?` argument that may arrive as an *undefined* at::Tensor.

    torch's C++ composites hand an absent optional tensor over as
    `std::optional<Tensor>` holding an undefined Tensor (this is what
    `F.conv2d(x, w)` does to convolution's bias), and the shim's record
    conversion boxes that as a Tensor rather than as None. Every accessor but
    `numel()` throws on an undefined tensor, so probe that first: it reads 0,
    and a genuinely length-0 bias contributes nothing either, so both answer
    "no bias". The proper fix is one line in the shim's `to_record` (map an
    undefined tensor to TMB_NONE) and would make this a plain v_opt_tensor.
    """
    if v.tag == TAG_NONE:
        return None
    var h = Int(v.a)
    if external_call["tmb_tensor_numel", Int64](h) == 0:
        return None
    return T(h)


# --- aten::mm / aten::bmm -----------------------------------------------------


def _mm_route(a: T, b: T) raises -> Optional[T]:
    var g = _try_gemm16_mm(a, b, None, False, List[Int]())
    if g:
        return g.value().copy()
    var t = _try_tf32_mm(a, b, None, False, List[Int]())
    if t:
        return t.value().copy()
    return _spec_matmul("MatmulSpec", a, b, None, 0)


def _store_out(rets: Values, dest: T, var result: T) raises:
    """Finish an `out=` variant: move a freshly computed result into the
    caller's tensor, resizing it the way torch's own out= kernels do, and
    return that tensor. TorchInductor reaches every extern kernel through
    these overloads (`extern_kernels.mm(a, b, out=buf)`).

    Never a cast: `check_out` has already required `dest` to hold the
    result's dtype, so a mismatch here would be a bug in the route, and
    `copy_strided_into` raises on one rather than truncating.
    """
    var held = own(result^)
    var dst = dest.copy()
    if not dst.same_shape(held.t):
        # Only a MISMATCHING out= is resized: a resize re-lays the tensor out
        # contiguously, so an already-correct out keeps its own strides.
        resize_out(dst, held.t.shape, held.t.rank)
    copy_strided_into(dst, held.t)
    _ = held^  # alive past the launch: reading `.t` copies a non-owning view
    ret_ref(rets, 0, dst)


# aten::mm(Tensor self, Tensor mat2) -> Tensor
def op_mm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var out = _mm_route(a, b)
    if not out:
        unsupported("aten::mm with these operands")
    ret_tensor(rets, 0, out.value())


# aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
def op_mm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var dest = v_tensor(args[unsafe_offset=2])
    check_out(dest, a)  # TORCH_META_FUNC(mm) sets the output from `self`
    var out = _mm_route(a, b)
    if not out:
        unsupported("aten::mm.out with these operands")
    _store_out(rets, dest, out.value().copy())


def _bmm_route(a: T, b: T) raises -> Optional[T]:
    var g = _try_gemm16_bmm(a, b, False)
    if g:
        return g.value().copy()
    var t = _try_tf32_bmm(a, b, False)
    if t:
        return t.value().copy()
    return _spec_matmul("BmmSpec", a, b, None, 0)


# aten::bmm(Tensor self, Tensor mat2) -> Tensor
def op_bmm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var out = _bmm_route(a, b)
    if not out:
        unsupported("aten::bmm with these operands")
    ret_tensor(rets, 0, out.value())


# aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
def op_bmm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var dest = v_tensor(args[unsafe_offset=2])
    check_out(dest, b)  # common_checks_baddbmm_bmm uses `batch2.options()`
    var out = _bmm_route(a, b)
    if not out:
        unsupported("aten::bmm.out with these operands")
    _store_out(rets, dest, out.value().copy())


# --- aten::addmm --------------------------------------------------------------


def _addmm_route(bias: T, mat1: T, mat2: T) raises -> Optional[T]:
    var opt_bias = Optional[T](bias.copy())
    # See _try_gemm16_linear: every gemm16 tensor-core route declines outright
    # when a bias is present, so compute the bias-free mm and add separately
    # whenever the shape could plausibly reach one.
    var am = _dense_2d(mat1)
    var bm = _dense_2d(mat2)
    if am and bm and _alignment_favors_split(am.value(), bm.value(), False):
        var mm_out = _try_gemm16_mm(mat1, mat2, None, False, List[Int]())
        if mm_out:
            var plain = own(mm_out.value().copy())
            var biased = _try_add(plain.t, bias)
            if biased:
                return biased.value().copy()
            # The add declined this bias: drop the unbiased product and fall
            # through to the fused kernel rather than return a biasless one.
            _ = plain^
    var g = _try_gemm16_mm(mat1, mat2, opt_bias, False, List[Int]())
    if g:
        return g.value().copy()
    var t = _try_tf32_mm(mat1, mat2, opt_bias, False, List[Int]())
    if t:
        return t.value().copy()
    return _spec_matmul("MatmulBiasSpec", mat1, mat2, opt_bias, 0)


def _addmm_unit_scaling(beta: Value, alpha: Value) raises:
    if v_f64(beta) != 1.0 or v_f64(alpha) != 1.0:
        # beta/alpha scaling is not implemented by the fast path.
        unsupported("aten::addmm with beta != 1 or alpha != 1")


# aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1,
#             Scalar alpha=1) -> Tensor
def op_addmm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    _addmm_unit_scaling(args[unsafe_offset=3], args[unsafe_offset=4])
    var out = _addmm_route(
        v_tensor(args[unsafe_offset=0]),
        v_tensor(args[unsafe_offset=1]),
        v_tensor(args[unsafe_offset=2]),
    )
    if not out:
        unsupported("aten::addmm with these operands")
    ret_tensor(rets, 0, out.value())


# aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1,
#                 Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
def op_addmm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var mat1 = v_tensor(args[unsafe_offset=1])
    var dest = v_tensor(args[unsafe_offset=5])
    # ADDMM_META sets the output from `mat1.options()`, and the meta function
    # runs before the kernel -- so the out= contract outranks the decline.
    check_out(dest, mat1)
    _addmm_unit_scaling(args[unsafe_offset=3], args[unsafe_offset=4])
    var out = _addmm_route(
        v_tensor(args[unsafe_offset=0]),
        mat1,
        v_tensor(args[unsafe_offset=2]),
    )
    if not out:
        unsupported("aten::addmm.out with these operands")
    _store_out(rets, dest, out.value().copy())


# --- aten::linear -------------------------------------------------------------


def _linear_route(a: T, w: T, bias: Optional[T]) raises -> Optional[T]:
    """input @ weight.T [+ bias]. The GEMM kernels read B transposed for free,
    so the weight is never materialized in transposed layout."""
    var g = _try_gemm16_linear(a, w, bias)
    if g:
        return g.value().copy()
    var t = _try_tf32_linear(a, w, bias)
    if t:
        return t.value().copy()
    if bias:
        return _spec_matmul("MatmulBiasSpec", a, w, bias, 1)
    return _spec_matmul("MatmulSpec", a, w, None, 1)


def _linear_vector(a: T, w: T, bias: Optional[T]) raises -> Optional[T]:
    """The rank-1 input the spec ABI (rank >= 2) cannot take, reusing the
    rank-2 implementation. Inspected only after the ordinary routes decline,
    so strict FP32 reaches its SIMT path without touching this."""
    if a.rank != 1 or w.rank != 2 or w.dim(1) != a.dim(0):
        return None
    if w.stype != a.stype or w.device != a.device:
        return None
    if bias:
        if (
            bias.value().rank != 1
            or bias.value().dim(0) != w.dim(0)
            or bias.value().stype != a.stype
            or bias.value().device != a.device
        ):
            return None
    var out_features = w.dim(0)
    if out_features == 0:
        return _new([0], a.stype, a.device)
    if a.dim(0) == 0:
        if bias:
            var clone = own(_new([out_features], a.stype, a.device))
            copy_strided_into(clone.t, bias.value())
            return clone.take()
        return _zeros([out_features], a.stype, a.device)
    var vec = Tmp(a)
    var matrix = own(_view(vec.t, [1, a.dim(0)]))
    var res = _linear_route(matrix.t, w, bias)
    _ = vec^
    if not res:
        return None
    var flat = own(res.value().copy())
    var shaped = _view(flat.t, [out_features])
    _ = flat^  # `_view` reads `flat`'s handle: it must outlive the call
    return shaped^


# aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
def op_linear(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var w = v_tensor(args[unsafe_offset=1])
    var bias = _opt_tensor_arg(args[unsafe_offset=2])
    var out = _linear_route(a, w, bias)
    if out:
        ret_tensor(rets, 0, out.value())
        return
    var vec = _linear_vector(a, w, bias)
    if not vec:
        unsupported("aten::linear with these operands")
    ret_tensor(rets, 0, vec.value())


# --- aten::linear_backward ----------------------------------------------------


def _bool_list(v: Value) raises -> List[Bool]:
    if v.tag != TAG_BOOL_LIST:
        raise Error("expected a bool[] argument, got record tag ", v.tag)
    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
    var out = List[Bool](capacity=Int(v.len))
    for i in range(Int(v.len)):
        out.append(p[unsafe_offset=i] != 0)
    return out^


def _sum_rows(a: T) raises -> T:
    """`a.sum(dim=0)` for a contiguous (rows, cols) matrix.

    reduction_ops reads a leading reduce interval where it lies (outer 1,
    reduce rows, inner cols), so this needs no transposed materialization.
    """
    var out = own(_new([a.dim(1)], a.stype, a.device))
    var ctx = ctx_for(a.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("reduction_ops", "SumSpec")
    call.arg_dtype(0, a.dtype)
    call.out_dtype(a.dtype)
    call.spec(a.spec(cp))
    call.tuple([0])
    call.int(0)
    call.spec(out.t.spec(cp))
    call.run()
    _ = ctx
    return out.take()


def _transpose_2d(t: T) raises -> T:
    """A zero-copy (cols, rows) view: the weight gradient's left operand."""
    var shape = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](0)
    shape[MAX_RANK - 2] = t.dim(1)
    shape[MAX_RANK - 1] = t.dim(0)
    strides[MAX_RANK - 2] = t.stride(1)
    strides[MAX_RANK - 1] = t.stride(0)
    return view_strided(t, shape, strides, 2, t.offset)


# aten::linear_backward(Tensor self, Tensor grad_output, Tensor weight,
#                       bool[3] output_mask) -> (Tensor, Tensor, Tensor)
def op_linear_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var input = v_tensor(args[unsafe_offset=0])
    var grad = v_tensor(args[unsafe_offset=1])
    var w = v_tensor(args[unsafe_offset=2])
    var mask = _bool_list(args[unsafe_offset=3])
    if (
        len(mask) != 3
        or input.rank < 1
        or w.rank != 2
        or not _is_float(input.dtype)
        or grad.stype != input.stype
        or w.stype != input.stype
        or grad.device != input.device
        or w.device != input.device
        or input.dim(input.rank - 1) != w.dim(1)
    ):
        unsupported("aten::linear_backward with these operands")
    var expected = _leading_dims(input)
    expected.append(w.dim(0))
    if grad.rank != len(expected):
        unsupported("aten::linear_backward: grad_output has the wrong rank")
    for i in range(grad.rank):
        if grad.dim(i) != expected[i]:
            unsupported(
                "aten::linear_backward: grad_output has the wrong shape"
            )

    var stype = input.stype
    var device = input.device
    if not mask[0] and not mask[1] and not mask[2]:
        for i in range(3):
            _ret_undefined(rets, i)
        return

    var rows = _prod(_leading_dims(input)) if input.rank > 1 else 1
    var in_features = input.dim(input.rank - 1)
    var out_features = w.dim(0)
    # PyTorch's registered Meta/MPS contract defines both parameter outputs
    # whenever either one is requested: an unrequested bias result is only an
    # allocation, but requesting the bias also requires the weight GEMM.
    var need_params = mask[1] or mask[2]

    if rows == 0 or out_features == 0:
        if mask[0]:
            ret_tensor(rets, 0, _zeros(input.logical_shape(), stype, device))
        else:
            _ret_undefined(rets, 0)
        if need_params:
            ret_tensor(rets, 1, _zeros(w.logical_shape(), stype, device))
            if mask[2]:
                ret_tensor(rets, 2, _zeros([out_features], stype, device))
            else:
                ret_tensor(rets, 2, _new([out_features], stype, device))
        else:
            _ret_undefined(rets, 1)
            _ret_undefined(rets, 2)
        return

    var grad_c = Tmp(grad)
    var grad_matrix = own(_view(grad_c.t, [rows, out_features]))

    var grad_input = own(_empty_result(stype, device))
    if mask[0]:
        if in_features == 0:
            grad_input = own(_new(input.logical_shape(), stype, device))
        else:
            var dx = _mm_route(grad_matrix.t, w)
            if not dx:
                unsupported("aten::linear_backward: no mm route for dgrad")
            var flat = own(dx.value().copy())
            grad_input = own(_view(flat.t, input.logical_shape()))
            _ = flat^  # `_view` reads `flat`'s handle (see above)

    var grad_weight = own(_empty_result(stype, device))
    if need_params:
        if in_features == 0:
            grad_weight = own(_new(w.logical_shape(), stype, device))
        else:
            var input_c = Tmp(input)
            var input_matrix = own(_view(input_c.t, [rows, in_features]))
            var gt = own(_transpose_2d(grad_matrix.t))
            var dw = _mm_route(gt.t, input_matrix.t)
            _ = input_c^
            if not dw:
                unsupported("aten::linear_backward: no mm route for wgrad")
            grad_weight = own(dw.value().copy())

    var grad_bias = own(_empty_result(stype, device))
    if need_params:
        if mask[2]:
            grad_bias = own(_sum_rows(grad_matrix.t))
        else:
            grad_bias = own(_new([out_features], stype, device))

    _ = grad_c^
    if mask[0]:
        ret_owned(rets, 0, grad_input)
    else:
        _ret_undefined(rets, 0)
    if need_params:
        ret_owned(rets, 1, grad_weight)
        ret_owned(rets, 2, grad_bias)
    else:
        _ret_undefined(rets, 1)
        _ret_undefined(rets, 2)


# --- aten::addr ---------------------------------------------------------------


def _addr_fast(
    self: T, vec1: T, vec2: T, beta: Float64, alpha: Float64
) raises -> Optional[T]:
    """beta*self + alpha*outer(vec1, vec2) in one launch that reproduces CPU's
    own addr_kernel op order and per-op rounding exactly (see `_addr_bcast` in
    logic_ops.mojo for why the order, not a wider accumulator, is what fp16
    and bf16 need). Declines whenever `self` is not standard-broadcastable to
    (len(vec1), len(vec2)) or the dtypes do not already match; the caller then
    runs ATen's own composite, so declining never removes support."""
    if self.device != vec1.device or self.device != vec2.device:
        return None
    var dt = vec1.dtype
    if self.stype != vec1.stype or vec2.stype != vec1.stype:
        return None
    if not _is_float(dt):
        return None
    if vec1.rank != 1 or vec2.rank != 1 or self.rank > 2:
        return None
    var n = vec1.dim(0)
    var m = vec2.dim(0)
    # Right-align `self` against (n, m), the same rule as every other
    # broadcast op here — but NOT against vec1/vec2, which torch's own addr
    # places at dim 0 / dim 1 respectively regardless of rank.
    var a0 = 1
    var a1 = 1
    var as0 = 0
    var as1 = 0
    if self.rank == 2:
        a0 = self.dim(0)
        a1 = self.dim(1)
        as0 = self.stride(0)
        as1 = self.stride(1)
    elif self.rank == 1:
        a1 = self.dim(0)
        as1 = self.stride(0)
    if (a0 != 1 and a0 != n) or (a1 != 1 and a1 != m):
        return None
    if a0 == 1:
        as0 = 0
    if a1 == 1:
        as1 = 0
    var out = own(_new([n, m], self.stype, self.device))
    if out.t.numel > 0:
        var ctx = ctx_for(self.device)
        var call = KernelCall("logic_ops", "AddrBcast")
        call.arg_dtype(0, self.dtype)
        call.arg_dtype(1, vec1.dtype)
        call.arg_dtype(2, vec2.dtype)
        call.out_dtype(out.t.dtype)
        call.int(out.t.ptr)
        call.int(self.ptr)
        call.int(vec1.ptr)
        call.int(vec2.ptr)
        call.tuple([n, m, as0, as1, vec1.stride(0), vec2.stride(0)])
        call.f64(beta)
        call.f64(alpha)
        call.int(dtype_code(dt))
        call.int(ctx_ptr(ctx))
        call.run()
        _ = ctx
    return out.take()


def _addr_composite(
    self: T, vec1: T, vec2: T, beta: Value, alpha: Value
) raises -> T:
    """ATen's own `math_addr`, branch for branch.

    The old registration redispatched a declined call to the
    CompositeExplicitAutograd kernel; `tmb_call_op` dispatches on the operands
    and would land back in this op, so the composition is spelled out over the
    ops it is made of. beta == 0 ignores `self` entirely, so nans and infs in
    it do not propagate.
    """
    var beta_v = v_f64(beta)
    var alpha_v = v_f64(alpha)
    var outer = own(
        _call_1("aten::outer", "", _tensor_arg(vec1), _tensor_arg(vec2))
    )
    # Every Owned below stays alive past the call that reads its `.t`: Mojo
    # destroys a value at its last use, which would otherwise be the argument
    # read, before the dispatcher runs.
    if beta_v == 0.0:
        if alpha_v == 1.0:
            return outer.take()
        var r = _call_1("aten::mul", "Scalar", _tensor_arg(outer.t), alpha)
        _ = outer^
        return r^
    if alpha_v == 1.0:
        if beta_v == 1.0:
            var r = _add_or_raise(self, outer.t)
            _ = outer^
            return r^
        var lhs = own(_call_1("aten::mul", "Scalar", _tensor_arg(self), beta))
        var r = _add_or_raise(lhs.t, outer.t)
        _ = lhs^
        _ = outer^
        return r^
    var scaled = own(
        _call_1("aten::mul", "Scalar", _tensor_arg(outer.t), alpha)
    )
    _ = outer^
    if beta_v == 1.0:
        var r = _add_or_raise(self, scaled.t)
        _ = scaled^
        return r^
    var lhs = own(_call_1("aten::mul", "Scalar", _tensor_arg(self), beta))
    var r = _add_or_raise(lhs.t, scaled.t)
    _ = lhs^
    _ = scaled^
    return r^


# aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1,
#            Scalar alpha=1) -> Tensor
def op_addr(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var vec1 = v_tensor(args[unsafe_offset=1])
    var vec2 = v_tensor(args[unsafe_offset=2])
    var beta = args[unsafe_offset=3].copy()
    var alpha = args[unsafe_offset=4].copy()
    if not v_scalar_is_bool(beta) and not v_scalar_is_bool(alpha):
        var fast = _addr_fast(self, vec1, vec2, v_f64(beta), v_f64(alpha))
        if fast:
            ret_tensor(rets, 0, fast.value())
            return
    ret_tensor(rets, 0, _addr_composite(self, vec1, vec2, beta, alpha))


# --- aten::convolution --------------------------------------------------------


def _pair(xs: IntList) raises -> List[Int]:
    """int[1] or int[2] as (h, w); empty when it is neither."""
    var out = List[Int]()
    if len(xs) == 1:
        out.append(xs[0])
        out.append(xs[0])
    elif len(xs) == 2:
        out.append(xs[0])
        out.append(xs[1])
    return out^


def _conv_forward(
    input: T,
    weight: T,
    bias: Optional[T],
    stride: IntList,
    padding: IntList,
    dilation: IntList,
    transposed: Bool,
    groups: Int,
) raises -> Optional[T]:
    """Batched im2col + the pure-Mojo GEMM, with torch's (K, C, R, S) weight
    used as-is and an NCHW output — no layout permutes, no cuDNN. Grouped
    convolutions slice the channel-major im2col rows and the weights per group
    with element offsets."""
    if transposed or groups < 1:
        return None
    if input.stype != weight.stype or not _is_float(input.dtype):
        return None
    if input.device != weight.device:
        return None
    if input.rank != 4 or weight.rank != 4:
        return None
    var strides = _pair(stride)
    var pads = _pair(padding)
    var dils = _pair(dilation)
    if len(strides) != 2 or len(pads) != 2 or len(dils) != 2:
        return None
    var n = input.dim(0)
    var c = input.dim(1)
    var in_h = input.dim(2)
    var in_w = input.dim(3)
    if n == 0 or c == 0 or in_h == 0 or in_w == 0:
        return None
    var out_c = weight.dim(0)
    var c_per_group = weight.dim(1)
    var kh = weight.dim(2)
    var kw = weight.dim(3)
    var sh = strides[0]
    var sw = strides[1]
    var ph = pads[0]
    var pw = pads[1]
    var dh = dils[0]
    var dw = dils[1]
    if sh <= 0 or sw <= 0 or dh <= 0 or dw <= 0:
        return None
    var out_h = (in_h + 2 * ph - (dh * (kh - 1) + 1)) // sh + 1
    var out_w = (in_w + 2 * pw - (dw * (kw - 1) + 1)) // sw + 1
    if c_per_group * groups != c or out_h <= 0 or out_w <= 0:
        return None
    if out_c % groups != 0:
        return None
    if bias:
        if (
            bias.value().stype != input.stype
            or bias.value().rank != 1
            or bias.value().dim(0) != out_c
            or bias.value().device != input.device
        ):
            return None

    var a = Tmp(input)
    var w = Tmp(weight)
    var device = input.device
    var ctx = ctx_for(device)
    var cp = ctx_ptr(ctx)
    var cols = out_h * out_w
    var ckk = c * kh * kw
    var col = own(_new([0], input.stype, device))
    var col_ptr = a.t.ptr
    var one_by_one = (
        kh == 1
        and kw == 1
        and sh == 1
        and sw == 1
        and ph == 0
        and pw == 0
        and dh == 1
        and dw == 1
    )
    if not one_by_one:
        # Anything but a 1x1 stride-1 conv builds the patch matrix; for that
        # one case the NCHW input already is the col matrix.
        col = own(_new([n, ckk, cols], input.stype, device))
        var im = KernelCall("conv_ops", "Im2col")
        im.arg_dtype(0, input.dtype)
        im.out_dtype(input.dtype)
        im.int(col.t.ptr)
        im.int(a.t.ptr)
        im.tuple(
            [in_h, in_w, out_h, out_w, kh, kw, sh, sw, ph, pw, dh, dw, c, n]
        )
        im.int(dtype_code(input.dtype))
        im.int(cp)
        im.run()
        col_ptr = col.t.ptr

    var out = own(_new([n, out_c, out_h, out_w], input.stype, device))
    if groups == 1:
        var mm = KernelCall("matmul_ops", "Bmm")
        mm.arg_dtype(0, weight.dtype)
        mm.arg_dtype(1, input.dtype)
        mm.out_dtype(input.dtype)
        # Same define shape as every other Bmm site: the shared-A broadcast is
        # RUNTIME data (the trailing 1 in the params tuple, matmul_ops._bmm_go)
        # so naming it here would only fork this call site onto a second .so
        # of identical code.
        mm.flag("TRANSPOSE_B", 0)
        mm.int(out.t.ptr)
        mm.int(w.t.ptr)
        mm.int(col_ptr)
        mm.tuple([n, out_c, cols, ckk, 0, 1])
        mm.int(dtype_code(input.dtype))
        mm.int(cp)
        mm.run()
    else:
        # Channel-major im2col rows make each group a contiguous
        # (crs_g, cols) slice; one offset GEMM per (sample, group).
        var crs_g = c_per_group * kh * kw
        var oc_g = out_c // groups
        for s in range(n):
            for g in range(groups):
                var mm = KernelCall("matmul_ops", "Matmul")
                mm.arg_dtype(0, weight.dtype)
                mm.arg_dtype(1, input.dtype)
                mm.out_dtype(input.dtype)
                mm.flag("TRANSPOSE_B", 0)
                mm.int(out.t.ptr)
                mm.int(w.t.ptr)
                mm.int(col_ptr)
                mm.tuple(
                    [
                        oc_g,
                        cols,
                        crs_g,
                        0,
                        (s * out_c + g * oc_g) * cols,
                        g * oc_g * crs_g,
                        (s * c + g * c_per_group) * kh * kw * cols,
                    ]
                )
                mm.int(dtype_code(input.dtype))
                mm.int(cp)
                mm.run()
    if bias:
        var bt = Tmp(bias.value())
        var ba = KernelCall("conv_ops", "BiasAddChan")
        ba.arg_dtype(0, input.dtype)
        ba.arg_dtype(1, bt.t.dtype)
        ba.out_dtype(input.dtype)
        ba.int(out.t.ptr)
        ba.int(bt.t.ptr)
        ba.tuple([cols, out_c, n * out_c * cols])
        ba.int(dtype_code(input.dtype))
        ba.int(cp)
        ba.run()
        _ = bt^
    _ = ctx
    _ = a^
    _ = w^
    _ = col^
    return out.take()


# aten::convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride,
#   SymInt[] padding, SymInt[] dilation, bool transposed,
#   SymInt[] output_padding, SymInt groups) -> Tensor
def op_convolution(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var input = v_tensor(args[unsafe_offset=0])
    var weight = v_tensor(args[unsafe_offset=1])
    var bias = _opt_tensor_arg(args[unsafe_offset=2])
    var out = _conv_forward(
        input,
        weight,
        bias,
        IntList(args[unsafe_offset=3]),
        IntList(args[unsafe_offset=4]),
        IntList(args[unsafe_offset=5]),
        v_bool(args[unsafe_offset=6]),
        v_int(args[unsafe_offset=8]),
    )
    if not out:
        unsupported("aten::convolution with these operands")
    ret_tensor(rets, 0, out.value())


def register_matmul(site: Site) raises:
    impl[op_addmm, "addmm"](site)
    impl[op_addmm_out, "addmm.out"](site)
    impl[op_addr, "addr"](site)
    impl[op_bmm, "bmm"](site)
    impl[op_bmm_out, "bmm.out"](site)
    impl[op_convolution, "convolution"](site)
    impl[op_linear, "linear"](site)
    impl[op_linear_backward, "linear_backward"](site)
    impl[op_mm, "mm"](site)
    impl[op_mm_out, "mm.out"](site)


@export
def tmb_op_address() abi("C") -> Int:
    """Entry of this file's one-op extension: the address of the op the
    TMB_OP define selected (registry.mojo)."""
    return op_address_of[register_matmul]()
