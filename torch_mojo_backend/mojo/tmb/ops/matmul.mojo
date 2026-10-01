"""ATen ops: matmul group — mm, bmm, addmm, addmv, baddbmm, addbmm (with
alpha / beta, their `.dtype` overloads, _addmm_activation), _int_mm,
_weight_int8pack_mm, linear, linear_backward, addr and the convolution
forward and backward.

The route cascade is the old fast path's (aten_fast.py), unchanged:

    gemm16 (bf16/f16 tensor cores, CUDA sm_90a)
      -> tf32 (fp32 tensor cores, CUDA sm_90a, opt-in: the NT layout runs
         gemm16's WGMMA kernels at float32, the rest the SM80-class family)
      -> the generic matmul spec kernels (MatmulSpec / MatmulBiasSpec /
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

from tmb.backend.abi import (
    IntList,
    Owned,
    ST_BFLOAT16,
    ST_FLOAT16,
    ST_FLOAT32,
    ST_FLOAT64,
    ST_INT32,
    ST_INT8,
    ST_UINT8,
    T,
    TAG_DTYPE,
    TAG_BOOL_LIST,
    TAG_DOUBLE,
    TAG_NONE,
    TAG_SCALAR_BOOL,
    TAG_SCALAR_DOUBLE,
    TAG_SCALAR_INT,
    TAG_TENSOR,
    UNSUPPORTED_PREFIX,
    Value,
    Values,
    bits_f64,
    call_op,
    contiguous_strides,
    dtype_code,
    dtype_name,
    max_dtype,
    new_tensor,
    own,
    own_if_new,
    release,
    ret_owned,
    ret_ref,
    ret_tensor,
    tensor_arg,
    unsupported,
    v_bool,
    v_dtype_or,
    v_f64,
    v_int,
    v_scalar_is_bool,
    v_scalar_is_integral,
    v_tensor,
    view_strided,
)
from tmb.backend.device import ctx_for, ctx_ptr, dev
from tmb.backend.kernel_call import KernelCall, loader
from tmb.kernels.common.op_utils import MAX_RANK
from tmb.ops.binary import Res, _b_tside
from tmb.ops.common import (
    assert_no_internal_overlap,
    call_op_raw,
    can_cast,
    cast_into,
    cast_to,
    check_out,
    check_out_as,
    contiguous,
    copy_strided_into,
    device_str,
    fill_value,
    is_float_stype,
    is_int_stype,
    promote_types,
    resize_out,
    result_type,
    same_view,
    scalar_to_float,
    scalar_to_int,
    shares_storage,
)
from tmb.ops.data_movement import _scalar_type_name
from tmb.ops.pointwise import _none_side, _p, _pw_run
from tmb.ops.unary import _direct_unary_out, _gelu_spec, _unary_out
from tmb.backend.registry import Site, impl


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
        var dir = loader()[].family_dir(family) + "/"
        var ok = True
        for f in files:
            if not exists(dir + f):
                ok = False
        c[unsafe_offset=slot] = 1 if ok else 0
    return c[unsafe_offset=slot] == 1


def _gemm16_available() raises -> Bool:
    return _bridge_available(
        SLOT_GEMM16,
        "gemm16_matmul",
        [
            "entry.mojo",
            "gemm16_v3_kernels.mojo",
            "gemm16_tn_v4_kernels.mojo",
            "gemm16_kernels.mojo",
            "gemm16_candidate_dispatch.mojo",
            "gemm16_rolling_kernels.mojo",
            "gemm16_nt_bias_kernels.mojo",
            "gemm16_sched_pool.mojo",
            # The float32 (TF32) ladder falls back to this kernel in-family.
            "../tf32_matmul/tf32_gemm_kernels.mojo",
        ],
    )


def _tf32_available() raises -> Bool:
    return _bridge_available(
        SLOT_TF32,
        "tf32_matmul",
        ["entry.mojo", "tf32_gemm_kernels.mojo"],
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


comptime GEMM16_FAMILY = "gemm16_matmul"


def _gemm16_tuning(mut call: KernelCall):
    """The build settings the gemm16 candidate kernels were measured with.

    They are `-D` defines of the kernel family, not of this extension, so
    they have to travel on the call that builds it; passing them through
    `KernelCall.flag` also puts them in the specialization key, so changing
    one here compiles (and caches) its own library rather than silently
    reusing the last one. `get_defined_bool` reads 1 as true.

    All three were fitted on an H100 PCIe, the card the candidates were
    measured on: PAIR_CAST selects the paired bf16 conversion of the rolling
    NN epilogue, TUNE_NT_ROLLING the fused NT kernel's rolling stage/parity
    counters, and TUNE_NT_RASTER its raster height (upstream's default is
    16). No matrix dimension is ever a compile-time constant.
    """
    call.flag("PAIR_CAST", 1)
    call.flag("TUNE_NT_ROLLING", 1)
    call.flag("TUNE_NT_RASTER", 8)


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
    if family == GEMM16_FAMILY:
        _gemm16_tuning(call)
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


def _nt_bias_regime(m: Int, n: Int, k: Int) -> Bool:
    """The cheap metadata half of the fused NT-bias gate.

    A copy of the shape predicate `try_enqueue_candidate_nt_bias` applies
    (gemm16_candidate_dispatch.mojo), restated here for one reason: without
    it every biased projection on the device would allocate an output and
    cross into the kernel family only to be declined. The kernel-side helper
    stays the authoritative gate -- it alone checks dtype, architecture,
    pointer alignment, TMA extents and launch resources -- so this one is
    allowed to be looser, never tighter.
    """
    if m < 4096 or n < 1024 or k < 1024:
        return False
    # m needs only m % 8 == 0: the kernel's A/C TMA descriptors carry M as
    # the row-major operand's outer (non-innermost) extent with K (already
    # % 64 == 0, i.e. a row stride of >=128 bytes) as the inner dimension, so
    # no descriptor stride keys off M -- the 192x192 rolling route clips a
    # ragged M edge via TMA the same way it already clips ragged N. Kept at
    # m % 8 (not fully unaligned) because that is what was measured: the
    # ragged 6600x4800x1600 shape reaches the fused route at 148 us here,
    # vs 547 us when it fell through to the unfused (matmul + broadcast
    # add) path.
    if m % 8 != 0 or n % 64 != 0 or k % 64 != 0:
        return False
    # The residue-64 regime the candidate was fitted for; 128-aligned N and K
    # keep the routes they had.
    if n % 128 != 64 and k % 128 != 64:
        return False
    # max(n, k) <= 8 * min(n, k), without the overflow: a vocabulary
    # projection is far outside the measured aspect band.
    return 1 + (max(n, k) - 1) // 8 <= min(n, k)


def _try_gemm16_nt_bias_fused(
    a: Mat,
    b: Mat,
    transpose_b: Bool,
    bias: T,
    dt: DType,
    stype: Int32,
    device: Int,
    out_dims: List[Int],
) raises -> Optional[T]:
    """`C = A @ B.T + bias` in ONE launch, or None when nothing was launched.

    `Gemm16` always hands back an output tensor, so a kernel that declines
    inside it is indistinguishable from one that ran: this OP adds a host
    status slot that says which happened. On None the caller must fall back
    to its ORIGINAL unbiased GEMM plus broadcasting add -- never to the old
    bias-enabled ladder, which is the slow accepted mma.sync kernel.

    The caller has already checked the operands belong to gemm16 (dtype,
    device, sm_90a, family present); everything specific to the fused
    kernel is checked here and, finally, inside the kernel helper itself.
    """
    var m = a.rows
    var k = a.cols
    var n = b.rows if transpose_b else b.cols
    var rhs_k = b.cols if transpose_b else b.rows
    if m <= 0 or n <= 0 or k <= 0 or rhs_k != k:
        return None
    # The kernel reads A as (M, K) row-major and B as (N, K) row-major: the
    # physical NT pair, whatever combination of view and requested transpose
    # produced it. A transposed weight view makes an aten::linear physically
    # NN, and that is a different kernel's call.
    if a.t != 0 or (b.t ^ (1 if transpose_b else 0)) != 1:
        return None
    # The candidate is bf16 only; float16 keeps every route it had.
    if dt != DType.bfloat16:
        return None
    if not _bias_fits(bias, n, stype, device):
        return None
    if not _nt_bias_regime(m, n, k):
        return None
    var dims = List[Int]()
    if len(out_dims) == 0:
        dims.append(m)
        dims.append(n)
    else:
        dims = out_dims.copy()
        if dims[len(dims) - 1] != n or _prod(dims) != m * n:
            return None
    var out = own(_new(dims, stype, device))
    var ctx = ctx_for(device)
    # The kernel family writes its verdict here. It is a host Int64 whose
    # address travels as a slot, so it -- like `ctx` and `out` -- must still
    # be alive when `run()` returns: reading it below is what keeps it so.
    var accepted = Int64(0)
    var call = KernelCall(GEMM16_FAMILY, "Gemm16NTBiasTry")
    call.arg_dtype(0, dt)
    call.arg_dtype(1, dt)
    call.arg_dtype(2, dt)
    call.out_dtype(dt)
    _gemm16_tuning(call)
    call.int(out.t.ptr)
    call.int(a.ptr)
    call.int(b.ptr)
    call.int(bias.ptr)
    call.int(m)
    call.int(n)
    call.int(k)
    call.int(0)  # transpose_a: physical NT, checked above
    call.int(1)  # transpose_b
    call.int(1)  # has_bias
    call.int(ctx_ptr(ctx))
    call.int(Int(Pointer(to=accepted)))
    call.run()
    _ = ctx
    if accepted == 0:
        # Nothing was enqueued: drop the unused output and let the caller's
        # original route run.
        return None
    _ = accepted
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
        "gemm16_matmul",
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


def _tf32_wgmma_nt(a: Mat, b: Mat, transpose_b: Bool) raises -> Bool:
    """Whether a bias-free fp32 GEMM (TF32 already allowed) should go to
    gemm16's float32 build, whose NT routes are Hopper WGMMA + TMA kernels,
    rather than to the SM80-class tf32_matmul family.

    Only the physical NT pair -- A (m, k) row-major, B stored (n, k) -- was
    ported to a 4-byte operand. The rest is the TMA/epilogue hardware rule:
    both row pitches k * 4 bytes a multiple of 16 (k % 4 == 0), an even n for
    the epilogue's two-element stores, and 16-byte-aligned bases (an offset
    view's need not be). m is free, and so is k % BK: TMA zero-fills a
    partial trailing tile. Everything the gemm16 ladder still declines on
    the device (the output pointer, grid limits) runs the same SM80-class
    kernel inside that family, so this gate may be a guess, never a trap.
    """
    var m = a.rows
    var k = a.cols
    var n = b.rows if transpose_b else b.cols
    var rhs_k = b.cols if transpose_b else b.rows
    return (
        a.t == 0
        and (b.t ^ (1 if transpose_b else 0)) == 1
        and m > 0
        and n > 0
        and k > 0
        and rhs_k == k
        and k % 4 == 0
        and n % 2 == 0
        and a.ptr % 16 == 0
        and b.ptr % 16 == 0
        and _gemm16_available()
    )


def _tf32_bridge(
    a: T,
    am: Mat,
    bm: Mat,
    transpose_b: Bool,
    bias: Optional[T],
    out_dims: List[Int],
) raises -> Optional[T]:
    """One TF32 GEMM: the WGMMA NT route when `_tf32_wgmma_nt` takes it (with
    a bias added afterwards: those kernels have no bias epilogue), the
    SM80-class family otherwise."""
    if _tf32_wgmma_nt(am, bm, transpose_b):
        var mm_out = _gemm_bridge(
            GEMM16_FAMILY,
            "Gemm16",
            am,
            bm,
            transpose_b,
            None,
            a.dtype,
            a.stype,
            a.device,
            out_dims,
        )
        if not bias:
            return mm_out^
        if mm_out:
            var biased = _add_bias(mm_out.value().copy(), bias.value())
            if biased:
                return biased^
    return _gemm_bridge(
        "tf32_matmul",
        "Tf32GemmF32",
        am,
        bm,
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
    return _tf32_bridge(a, am.value(), bm.value(), transpose_b, bias, out_dims)


def _try_gemm16_bmm(a: T, b: T, transpose_b: Bool) raises -> Optional[T]:
    if not _gemm16_gate(a, b):
        return None
    var am = _batched_3d(a)
    var bm = _batched_3d(b)
    if not am or not bm:
        return None
    return _bmm_bridge(
        "gemm16_matmul",
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
        "tf32_matmul",
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

    ONE gemm16 route computes a bias: the fused NT kernel, over the narrow
    bf16 shape regime it was measured on (`_try_gemm16_nt_bias_fused`). It is
    tried first and, when it takes the call, its single launch IS the result
    -- no second add ever runs on that output.

    Every other tensor-core route declines outright when a bias is present,
    so a fused-bias call through them would silently fall back to the far
    slower accepted mma.sync kernel (measured 3.6-7.4x stock on deep-K
    shapes, versus ~1.3x for the identical unbiased mm). When the shape could
    reach a fast route at all (`_alignment_favors_split`), compute the
    bias-free mm and add the bias afterwards with the ordinary broadcasting
    add instead.
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
            "gemm16_matmul",
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
    var fused = _try_gemm16_nt_bias_fused(
        am.value(),
        wm.value(),
        True,
        bias.value(),
        a.dtype,
        a.stype,
        a.device,
        dims,
    )
    if fused:
        return fused.value().copy()
    if _alignment_favors_split(am.value(), wm.value(), True):
        var mm_out = _gemm_bridge(
            "gemm16_matmul",
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
            var biased = _add_bias(mm_out.value().copy(), bias.value())
            if biased:
                return biased^
    # Either the shape can never reach a fast route regardless of bias, or the
    # fast add declined for this bias: the bias-fused kernel is at worst
    # identical, and never drops the bias silently.
    return _gemm_bridge(
        "gemm16_matmul",
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
    return _tf32_bridge(a, am.value(), wm.value(), True, bias, dims)


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
    if a.stype != b.stype or (
        not _is_float(a.dtype)
        and not (a.dtype == DType.float64 and op == "MatmulSpec")
    ):
        return None
    if not a.on_mojo() or not b.on_mojo() or a.device != b.device:
        return None
    if a.dtype == DType.float64 and dev(a.device)[].api not in ("cuda", "hip"):
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
    var call = KernelCall("matmul", String(op))
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


def _bias_add_dtype_ok(dt: DType) -> Bool:
    """logic SPEC_BCAST_DTYPES, minus bool: what its broadcast add takes."""
    return (
        dt == DType.float32
        or dt == DType.bfloat16
        or dt == DType.float16
        or dt == DType.float64
        or dt == DType.int8
        or dt == DType.int16
        or dt == DType.int32
        or dt == DType.int64
        or dt == DType.uint8
    )


def _try_bias_inplace(dst: T, bias: T) raises -> Bool:
    """`dst += bias` into the product this op just computed, rather than
    `aten::add` through the dispatcher: one boxed call, one output buffer and
    one copy less per biased GEMM (192 of them per GPT-2 XL step).

    The kernel is the broadcast add `aten::add.Tensor` would itself have
    launched for this pair, and it writes element i from the element it read
    for i, so a destination that is also the left operand is exact. `dst` is
    this op's own fresh allocation; `bias` is the only aliasing to rule out.
    """
    if (
        not dst.on_mojo()
        or not bias.on_mojo()
        or dst.device != bias.device
        or dst.stype != bias.stype
        or not dst.contig
        or not bias.contig
        or not _bias_add_dtype_ok(dst.dtype)
        or bias.rank > dst.rank
        or dst.rank > 4
    ):
        return False
    for i in range(MAX_RANK):
        if bias.shape[i] != dst.shape[i] and bias.shape[i] != 1:
            return False
    # Contiguous, so `numel * itemsize` is the exact byte span of each.
    if not (
        bias.ptr >= dst.ptr + dst.numel * dst.itemsize
        or dst.ptr >= bias.ptr + bias.numel * bias.itemsize
    ):
        return False
    if dst.numel == 0:
        return True
    var ctx = ctx_for(dst.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("logic", "AddSpec")
    call.arg_dtype(0, dst.dtype)
    call.arg_dtype(1, bias.dtype)
    call.out_dtype(dst.dtype)
    call.spec(dst.spec(cp))
    call.spec(bias.spec(cp))
    call.spec(dst.spec(cp))
    try:
        call.run()
    except e:
        if _declined(e):
            return False
        raise e^
    _ = ctx
    return True


def _add_bias(var product: T, bias: T) raises -> Optional[T]:
    """`product + bias` for a GEMM whose route has no bias epilogue: in place
    into the fresh product when it can be, else through aten::add. None when
    both decline -- the unbiased product is dropped, and the caller falls
    through to a bias-fused route rather than return a biasless result."""
    var plain = own(product^)
    if _try_bias_inplace(plain.t, bias):
        return plain.take()
    var biased = _try_add(plain.t, bias)
    _ = plain^
    return biased^


# --- neighbouring ops reached through the dispatcher --------------------------


def _try_add(a: T, b: T) raises -> Optional[T]:
    """`a + b` through aten::add.Tensor — the same broadcasting elementwise
    add every other caller of that op gets. None when it declines."""
    var args = Array[Value, 3](fill=Value(TAG_NONE, 0, 0, 0))
    args[0] = Value(TAG_TENSOR, 0, Int64(a.h), 0)
    args[1] = Value(TAG_TENSOR, 0, Int64(b.h), 0)
    args[2] = Value(TAG_SCALAR_INT, 0, 1, 0)
    var rets = Array[Value, 1](fill=Value(TAG_NONE, 0, 0, 0))
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
    var args = Array[Value, 2](fill=Value(TAG_NONE, 0, 0, 0))
    args[0] = a.copy()
    args[1] = b.copy()
    var rets = Array[Value, 1](fill=Value(TAG_NONE, 0, 0, 0))
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


def _addr_integral_scalar(v: Value, name: StaticString) raises:
    if v.tag == TAG_SCALAR_DOUBLE or v.tag == TAG_DOUBLE:
        raise Error(
            "For integral input tensors, argument ",
            name,
            " must not be a floating point number.",
        )


def _bool_scalar(b: Bool) -> Value:
    return Value(TAG_SCALAR_BOOL, 0, Int64(1) if b else Int64(0), 0)


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


def _bmm_route(a: T, b: T) raises -> Optional[T]:
    var g = _try_gemm16_bmm(a, b, False)
    if g:
        return g.value().copy()
    var t = _try_tf32_bmm(a, b, False)
    if t:
        return t.value().copy()
    return _spec_matmul("BmmSpec", a, b, None, 0)


def _addmm_route(bias: T, mat1: T, mat2: T) raises -> Optional[T]:
    var opt_bias = Optional[T](bias.copy())
    var am = _dense_2d(mat1)
    var bm = _dense_2d(mat2)
    # The one gemm16 route that computes a bias itself: a `mat2` stored
    # transposed is the physical NT pair the fused kernel takes, and the
    # usual `input @ weight.T` addmm is exactly that view. On acceptance its
    # single launch is the whole result.
    if am and bm and _gemm16_gate(mat1, mat2):
        var fused = _try_gemm16_nt_bias_fused(
            am.value(),
            bm.value(),
            False,
            bias,
            mat1.dtype,
            mat1.stype,
            mat1.device,
            List[Int](),
        )
        if fused:
            return fused.value().copy()
    # See _try_gemm16_linear: every other gemm16 tensor-core route declines
    # outright when a bias is present, so compute the bias-free mm and add
    # separately whenever the shape could plausibly reach one.
    if am and bm and _alignment_favors_split(am.value(), bm.value(), False):
        var mm_out = _try_gemm16_mm(mat1, mat2, None, False, List[Int]())
        if mm_out:
            var biased = _add_bias(mm_out.value().copy(), bias)
            if biased:
                return biased^
    var g = _try_gemm16_mm(mat1, mat2, opt_bias, False, List[Int]())
    if g:
        return g.value().copy()
    var t = _try_tf32_mm(mat1, mat2, opt_bias, False, List[Int]())
    if t:
        return t.value().copy()
    return _spec_matmul("MatmulBiasSpec", mat1, mat2, opt_bias, 0)


# --- the BLAS family: alpha / beta, out_dtype, empty and integer operands -----
#
# mm, bmm, addmm, addmv, baddbmm, addbmm and their `.out` / in-place /
# `.dtype` overloads are one computation:
#
#     out = beta * self + alpha * (A @ B)          (beta == 0: self unread)
#
# run as the GEMM routes above (`_product`) followed by one fused pointwise
# epilogue (`blas_scale` / `blas_axpby` in tmb/kernels/pointwise), the
# cuBLAS `alpha * acc + beta * C` of aten/src/ATen/native/cuda/Blas.cpp in
# the compute type -- float for float and the half types, double for double,
# scalar_t for integers -- rounded once. A float16 / bfloat16 call that
# needs an epilogue therefore computes its product in float32 (from exact
# float32 copies of the operands), so the result rounds where cuBLAS rounds
# instead of once more after the product. A bare product (mm, bmm, alpha 1
# with beta 0) runs the half-precision routes directly.
#
# Integer operands (CPU torch's mm / bmm / addmm on integers, and
# aten::_int_mm's int8 -> int32) run the tiled kernel with an integer
# accumulator (`IntMmSpec`); stock CUDA has no integer GEMM but these.


def _dims_str(dims: List[Int]) -> String:
    var s = String("[")
    for i in range(len(dims)):
        if i:
            s += ", "
        s += String(dims[i])
    return s + "]"


def _matrix_str(t: T) -> String:
    return String(t.dim(0), "x", t.dim(1))


def _check_same_device(ts: List[T]) raises:
    """checkAllSameGPU: every operand on one device, and that one ours."""
    for i in range(1, len(ts)):
        if (
            ts[i].device_type != ts[0].device_type
            or ts[i].device != ts[0].device
        ):
            raise Error(
                (
                    "Expected all tensors to be on the same device, but found"
                    " at least two devices, "
                ),
                device_str(ts[0]),
                " and ",
                device_str(ts[i]),
                "!",
            )
    if not ts[0].on_mojo():
        unsupported("a matmul whose operands are not on the mojo device")


def _check_expand(self: T, dims: List[Int], fname: StaticString) raises:
    """`expand_size(self, dims, fn)`: self broadcasts to `dims` (it may only
    add leading dimensions and stretch size-1 ones)."""
    var r = len(dims)
    if self.rank > r:
        raise Error(
            "expand(",
            fname,
            "): the number of sizes provided (",
            r,
            (
                ") must be greater or equal to the number of dimensions in the"
                " tensor ("
            ),
            self.rank,
            ")",
        )
    for i in range(self.rank):
        var d = r - self.rank + i
        var s = self.dim(i)
        if s != 1 and s != dims[d]:
            raise Error(
                "The expanded size of the tensor (",
                dims[d],
                ") must match the existing size (",
                s,
                ") at non-singleton dimension ",
                d,
                ".  Target sizes: ",
                _dims_str(dims),
                ".  Tensor sizes: ",
                _dims_str(self.logical_shape()),
            )


def _check_inplace(self: T, dims: List[Int]) raises:
    """A structured in-place op's output is `self`, never resized."""
    var same = self.rank == len(dims)
    if same:
        for i in range(self.rank):
            if self.dim(i) != dims[i]:
                same = False
    if not same:
        raise Error(
            "Bad in-place call: input tensor size ",
            _dims_str(self.logical_shape()),
            " and output tensor size ",
            _dims_str(dims),
            " should match",
        )


@fieldwise_init
struct _Coef(Copyable, ImplicitlyCopyable, Movable):
    """alpha or beta as the kernel applies it: `Scalar::to<opmath_t>()` (a
    float, or double for double operands) or `to<scalar_t>()` for integer
    operands."""

    var f: Float64
    var i: Int
    var integral: Bool

    def zero(self) -> Bool:
        return self.i == 0 if self.integral else self.f == 0.0

    def one(self) -> Bool:
        return self.i == 1 if self.integral else self.f == 1.0

    def param(self) -> Float64:
        """The pointwise slot: an integer travels as its int64 bits."""
        return bits_f64(Int64(self.i)) if self.integral else self.f


def _unit(st: Int32, value: Int) -> _Coef:
    return _Coef(Float64(value), value, is_int_stype(st))


def _coef(v: Value, st: Int32) raises -> _Coef:
    """`Scalar::to<opmath_t>()` for operands of dtype `st` (range-checked:
    a finite value beyond float's range raises as c10's checked_convert)."""
    if is_int_stype(st):
        return _Coef(0.0, scalar_to_int(v, st), True)
    if st == ST_FLOAT64:
        return _Coef(v_f64(v), 0, False)
    return _Coef(scalar_to_float(v, ST_FLOAT32), 0, False)


def _coef_scalar_t(v: Value, st: Int32) raises -> _Coef:
    """`Scalar::to<scalar_t>()`: addmv's gemv takes alpha and beta in the
    tensor's own dtype, so a half call rounds them to half first."""
    if is_int_stype(st) or st == ST_FLOAT64 or st == ST_FLOAT32:
        return _coef(v, st)
    var f = scalar_to_float(v, st)
    if st == ST_FLOAT16:
        f = f.cast[DType.float16]().cast[DType.float64]()
    else:
        f = f.cast[DType.bfloat16]().cast[DType.float64]()
    return _Coef(f, 0, False)


def _round_to(c: _Coef, st: Int32) -> _Coef:
    """`scalar_tensor(beta, self.scalar_type())`: beta stored in self's
    dtype (the k == 0 shortcut of addmm / addmv multiplies by it)."""
    if c.integral or st == ST_FLOAT64 or st == ST_FLOAT32:
        return c
    if st == ST_FLOAT16:
        return _Coef(c.f.cast[DType.float16]().cast[DType.float64](), 0, False)
    return _Coef(c.f.cast[DType.bfloat16]().cast[DType.float64](), 0, False)


def _mul_scalar(v: Value, st: Int32) raises -> _Coef:
    """beta as `result.mul_(beta)` applies it: a wrapped double (unchecked)
    on a floating result; on an integer one a floating Scalar promotes the
    product to the default float, which the in-place result cannot hold."""
    if is_int_stype(st):
        if not v_scalar_is_integral(v):
            raise Error(
                "result type Float can't be cast to the desired output type ",
                _scalar_type_name(max_dtype(st)),
            )
        return _Coef(0.0, v_int(v), True)
    return _Coef(v_f64(v), 0, False)


def _raw_zero(v: Value) raises -> Bool:
    """`scalar.toComplexDouble() == 0`: no range check, nothing converted."""
    return v_f64(v) == 0.0


def _coefs(
    alpha_v: Value,
    beta_v: Value,
    st: Int32,
    self_st: Int32,
    numel: Int,
    k: Int,
    scalar_t: Bool = False,
) raises -> Tuple[_Coef, _Coef]:
    """(alpha, beta) converted only where CUDA converts them: nothing for an
    empty result; for an empty reduction (k == 0) alpha is never read and
    beta is `scalar_tensor(beta, self.scalar_type())` (range-checked in
    self's dtype); otherwise both in opmath (or scalar_t: addmv's gemv)."""
    var zero = _unit(st, 0)
    if numel == 0:
        return (zero, zero)
    if k == 0:
        if _raw_zero(beta_v):
            return (zero, zero)
        if is_int_stype(self_st):
            return (zero, _Coef(0.0, scalar_to_int(beta_v, self_st), True))
        var f = scalar_to_float(beta_v, self_st)
        return (zero, _round_to(_Coef(f, 0, False), self_st))
    if scalar_t:
        return (_coef_scalar_t(alpha_v, st), _coef_scalar_t(beta_v, st))
    return (_coef(alpha_v, st), _coef(beta_v, st))


def _int_product(a: T, b: T, out_stype: Int32) raises -> T:
    """`a @ b` on integer operands: IntMmSpec over dense copies."""
    var ca = Tmp(a)
    var cb = Tmp(b)
    var dims = _leading_dims(a)
    dims.append(b.dim(b.rank - 1))
    var out = own(_new(dims, out_stype, a.device))
    var ctx = ctx_for(a.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("matmul", "IntMmSpec")
    call.arg_dtype(0, a.dtype)
    call.out_dtype(out.t.dtype)
    call.spec(ca.t.spec(cp))
    call.spec(cb.t.spec(cp))
    call.spec(out.t.spec(cp))
    call.run()
    _ = ctx
    _ = ca^
    _ = cb^
    return out.take()


def _product_dims(a: T, b: T) -> List[Int]:
    var dims = _leading_dims(a)
    dims.append(b.dim(b.rank - 1))
    return dims^


def _product(a: T, b: T, out_stype: Int32) raises -> T:
    """`a @ b` for rank-2 or batched rank-3 operands of one dtype on one
    device, as dtype `out_stype`: a's own, or float32 for float16 /
    bfloat16 operands (computed from exact float32 copies). Empty results
    and an empty reduction (k == 0: zeros) never reach a kernel."""
    var dims = _leading_dims(a)
    dims.append(b.dim(b.rank - 1))
    if _prod(dims) == 0:
        return _new(dims, out_stype, a.device)
    if a.dim(a.rank - 1) == 0:
        return _zeros(dims, out_stype, a.device)
    if is_int_stype(a.stype):
        return _int_product(a, b, out_stype)
    if out_stype != a.stype:
        var a32 = own_if_new(cast_to(a, out_stype), a)
        var b32 = own_if_new(cast_to(b, out_stype), b)
        var wide = _product(a32.t, b32.t, out_stype)
        _ = a32^
        _ = b32^
        return wide^
    var r = _bmm_route(a, b) if a.rank == 3 else _mm_route(a, b)
    if not r:
        unsupported(
            "a matmul of " + dtype_name(a.stype) + " operands on this device"
        )
    return r.value().copy()


def _blas(
    addend: Optional[T],
    a: T,
    b: T,
    alpha: _Coef,
    beta: _Coef,
    out_stype: Int32,
    dims: List[Int],
    dst: Optional[T],
    gemv: Bool = False,
) raises -> Res:
    """`beta * addend + alpha * (a @ b)` of shape `dims` (the product's own
    shape, or addmv's `[m]` view of its `[m, 1]`), stored as `out_stype`:
    into `dst` when the epilogue can write it directly, else fresh. The
    caller has validated everything; `addend` broadcasts to `dims`."""
    var k = a.dim(a.rank - 1)
    var use_addend = Bool(addend) and not beta.zero()
    var compute = out_stype
    if is_float_stype(out_stype) and out_stype != ST_FLOAT64:
        compute = ST_FLOAT32
    if not use_addend and (alpha.one() or k == 0):
        compute = out_stype  # the bare product: no epilogue
    # alpha == 0, as stock CUDA behaves (measured, torch 2.14 / H100): the
    # float32 / float64 GEMMs and addmv's gemv never read A or B (a NaN
    # there does not propagate, the result is `beta * addend`); the float16
    # / bfloat16 GEMMs (cublasGemmEx, float compute) still form the product,
    # so `0 * NaN` (or `0 * inf`) is NaN.
    var half = a.stype == ST_FLOAT16 or a.stype == ST_BFLOAT16
    var skip = alpha.zero() and (gemv or not half)
    var product = own(
        _zeros(_product_dims(a, b), compute, a.device) if skip else _product(
            a, b, compute
        )
    )
    if not _has_dims(product.t, dims):
        var shaped = _view(product.t, dims)
        product = own(shaped^)  # the view keeps the storage alive
    if compute == out_stype and not use_addend and (alpha.one() or k == 0):
        return Res(product.take(), True)
    # k == 0: the product is exactly zero, and alpha (inf or NaN included)
    # never touches it: CUDA's shortcut is `beta * self` (or zeros).
    var al = alpha
    if k == 0:
        al = _unit(out_stype, 0)
    var params = _p(al.param(), beta.param())
    var none = _none_side()
    var a_side = _b_tside(product.t)
    var b_side = _b_tside(addend.value()) if use_addend else none.copy()
    var res: Res
    if use_addend:
        res = _pw_run(
            "blas_axpby",
            2,
            a_side,
            b_side,
            none,
            compute,
            out_stype,
            params,
            dst,
        )
    else:
        res = _pw_run(
            "blas_scale",
            1,
            a_side,
            b_side,
            none,
            compute,
            out_stype,
            params,
            dst,
        )
    _ = product^  # the epilogue reads it: alive past the launch
    return res^


def _has_dims(t: T, dims: List[Int]) -> Bool:
    if t.rank != len(dims):
        return False
    for i in range(t.rank):
        if t.dim(i) != dims[i]:
            return False
    return True


comptime ACT_NONE = 0
comptime ACT_RELU = 1
comptime ACT_GELU = 2


def _activate(t: T, activation: Int) raises:
    """`_addmm_activation`'s epilogue as CUDA's non-Lt path runs it: relu_
    or gelu_(approximate="tanh") over the finished (rounded) result."""
    if activation == ACT_NONE or t.numel == 0:
        return
    var dst = t.copy()
    if activation == ACT_RELU:
        _direct_unary_out("ReluSpec", t, dst)
    else:
        _unary_out("elementwise", _gelu_spec("tanh"), t, dst, t.dtype)


def _hand_back(
    rets: Values,
    dest: Optional[T],
    var res: Res,
    dims: List[Int],
    activation: Int,
) raises:
    """Return a `_blas` result: as a fresh tensor, or in the caller's
    `out=` / in-place `self`, resized there like `resize_output` when it is
    not already `dims` (never an in-place self: its shape was checked)."""
    if not dest:
        var fresh = own(res.t.copy())
        _activate(fresh.t, activation)
        ret_owned(rets, 0, fresh)
        return
    var d = dest.value().copy()
    if res.owned:
        var held = own(res.t.copy())
        if not _has_dims(d, dims):
            resize_out(d, _index_list(dims), len(dims))
        copy_strided_into(d, held.t)
        _ = held^  # alive past the copy's launch
    _activate(d, activation)
    ret_ref(rets, 0, d)


def _blas_out(
    rets: Values,
    addend: Optional[T],
    a: T,
    b: T,
    alpha: _Coef,
    beta: _Coef,
    out_stype: Int32,
    dims: List[Int],
    dest: Optional[T],
    activation: Int = ACT_NONE,
    gemv: Bool = False,
) raises:
    """`_blas` into a fresh result or the caller's `dest`.

    The epilogue writes `dest` directly when it already has the result's
    shape and does not partially overlap the addend (the GEMM operands are
    consumed into the product before the epilogue runs, so their aliasing
    `dest` is harmless). A `dest` to be resized is resized first only when
    it shares no storage with an input -- a resize may move the storage an
    input still views -- else the result is computed fresh and copied in."""
    if not dest:
        _hand_back(
            rets,
            None,
            _blas(addend, a, b, alpha, beta, out_stype, dims, None, gemv),
            dims,
            activation,
        )
        return
    var d = dest.value().copy()
    if not _has_dims(d, dims):
        var shared = shares_storage(d, a) or shares_storage(d, b)
        if addend:
            shared = shared or shares_storage(d, addend.value())
        if not shared:
            resize_out(d, _index_list(dims), len(dims))
    _dest_overlap(Optional[T](d.copy()), dims)
    var direct = _has_dims(d, dims)
    if direct and addend:
        var s = addend.value().copy()
        if shares_storage(d, s) and not same_view(d, s):
            direct = False
    var target = Optional[T](d.copy()) if direct else Optional[T]()
    _hand_back(
        rets,
        Optional[T](d.copy()),
        _blas(addend, a, b, alpha, beta, out_stype, dims, target, gemv),
        dims,
        activation,
    )


def _check_dest(dest: T, stype: Int32, like: T) raises:
    """An `out=`: the result's dtype and device (its self-aliasing is
    checked by `_dest_overlap` once any resize has happened)."""
    check_out_as(dest, stype, like)


def _dest_overlap(dest: Optional[T], dims: List[Int]) raises:
    """`assert_no_internal_overlap` of the tensor written, as the structured
    kernels run it: after the meta's resize, so only a destination that
    already has the result's shape (and will be written as it lies) can
    alias itself. An in-place self always has it."""
    if dest and _has_dims(dest.value(), dims):
        assert_no_internal_overlap(dest.value())


def _float_out_dtype_ok(in_st: Int32, out_dtype: Int32) -> Bool:
    """The `.dtype` overloads: the input dtype itself, or float32 out of
    float16 / bfloat16 inputs."""
    return out_dtype == in_st or (
        out_dtype == ST_FLOAT32
        and (in_st == ST_FLOAT16 or in_st == ST_BFLOAT16)
    )


# --- aten::mm -----------------------------------------------------------------


def _mm_checks(a: T, b: T) raises:
    """TORCH_META_FUNC(mm), then addmm_out_cuda_impl's dtype check."""
    if a.rank != 2:
        raise Error("self must be a matrix")
    if b.rank != 2:
        raise Error("mat2 must be a matrix")
    if a.dim(1) != b.dim(0):
        raise Error(
            "mat1 and mat2 shapes cannot be multiplied (",
            _matrix_str(a),
            " and ",
            _matrix_str(b),
            ")",
        )
    if a.stype != b.stype:
        raise Error(
            "expected mat1 and mat2 to have the same dtype, but got: ",
            dtype_name(a.stype),
            " != ",
            dtype_name(b.stype),
        )
    _check_same_device([a.copy(), b.copy()])


# aten::mm(Tensor self, Tensor mat2) -> Tensor
def op_mm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    _mm_checks(a, b)
    _blas_out(
        rets,
        None,
        a,
        b,
        _unit(a.stype, 1),
        _unit(a.stype, 0),
        a.stype,
        [a.dim(0), b.dim(1)],
        None,
    )


# aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
def op_mm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var dest = v_tensor(args[unsafe_offset=2])
    _mm_checks(a, b)
    _check_dest(dest, a.stype, a)  # TORCH_META_FUNC(mm): `self.options()`
    _blas_out(
        rets,
        None,
        a,
        b,
        _unit(a.stype, 1),
        _unit(a.stype, 0),
        a.stype,
        [a.dim(0), b.dim(1)],
        Optional[T](dest.copy()),
    )


def _mm_dtype_checks(a: T, b: T, out_dtype: Int32) raises:
    """_mm_dtype_out_cuda (cuda/Blas.cpp), less its `out` checks."""
    if a.rank != 2:
        raise Error("self must be a matrix, got ", a.rank, "-D tensor")
    if b.rank != 2:
        raise Error("mat2 must be a matrix, got ", b.rank, "-D tensor")
    if a.dim(1) != b.dim(0):
        raise Error(
            "mat1 and mat2 shapes cannot be multiplied (",
            _matrix_str(a),
            " and ",
            _matrix_str(b),
            ")",
        )
    if a.stype != b.stype:
        raise Error("input dtypes must be the same")
    if not _float_out_dtype_ok(a.stype, out_dtype):
        raise Error(
            "out_dtype must be the same as input dtype or fp32 for fp16/bf16"
            " inputs"
        )


def _same_shape_out(dest: T, a: T, b: T) raises:
    """addmm_out_cuda_impl when `result.is_same(self)` (the `.dtype_out`
    overloads pass their `out` as self): never resized."""
    if dest.rank != 2:
        raise Error("tensors must be 2-D")
    if dest.dim(0) != a.dim(0):
        raise Error("self dim 0 must match mat1 dim 0")
    if dest.dim(1) != b.dim(1):
        raise Error("self dim 1 must match mat2 dim 1")


# aten::mm.dtype(Tensor self, Tensor mat2, ScalarType out_dtype) -> Tensor
def op_mm_dtype(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var out_dtype = v_dtype_or(args[unsafe_offset=2], a.stype)
    _mm_dtype_checks(a, b, out_dtype)
    _check_same_device([a.copy(), b.copy()])
    _blas_out(
        rets,
        None,
        a,
        b,
        _unit(a.stype, 1),
        _unit(a.stype, 0),
        out_dtype,
        [a.dim(0), b.dim(1)],
        None,
    )


# aten::mm.dtype_out(Tensor self, Tensor mat2, ScalarType out_dtype, *,
#                    Tensor(a!) out) -> Tensor(a!)
def op_mm_dtype_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var out_dtype = v_dtype_or(args[unsafe_offset=2], a.stype)
    var dest = v_tensor(args[unsafe_offset=3])
    _mm_dtype_checks(a, b, out_dtype)
    if dest.stype != out_dtype:
        raise Error(
            "out_dtype must be the same as the dtype of the provided out tensor"
        )
    _same_shape_out(dest, a, b)
    _check_same_device([dest.copy(), a.copy(), b.copy()])
    _blas_out(
        rets,
        None,
        a,
        b,
        _unit(a.stype, 1),
        _unit(a.stype, 0),
        out_dtype,
        [a.dim(0), b.dim(1)],
        Optional[T](dest.copy()),
    )


# --- aten::bmm ----------------------------------------------------------------


def _bmm_dims(b1: T, b2: T) raises -> List[Int]:
    """common_checks_baddbmm_bmm (LinearAlgebra.cpp): the shapes."""
    if b1.rank != 3:
        raise Error("batch1 must be a 3D tensor")
    if b2.rank != 3:
        raise Error("batch2 must be a 3D tensor")
    var bs = b1.dim(0)
    var k = b1.dim(2)
    if b2.dim(0) != bs or b2.dim(1) != k:
        raise Error(
            "Expected size for first two dimensions of batch2 tensor to be: [",
            bs,
            ", ",
            k,
            "] but got: [",
            b2.dim(0),
            ", ",
            b2.dim(1),
            "].",
        )
    return [bs, b1.dim(1), b2.dim(2)]


def _bmm_same_dtype(b1: T, b2: T) raises:
    """baddbmm_out_cuda_impl reads batch2 as batch1's scalar_t."""
    if b1.stype != b2.stype:
        raise Error(
            "expected scalar type ",
            _scalar_type_name(b1.dtype),
            " but found ",
            _scalar_type_name(b2.dtype),
        )


# aten::bmm(Tensor self, Tensor mat2) -> Tensor
def op_bmm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var dims = _bmm_dims(a, b)
    _bmm_same_dtype(a, b)
    _check_same_device([a.copy(), b.copy()])
    _blas_out(
        rets,
        None,
        a,
        b,
        _unit(a.stype, 1),
        _unit(a.stype, 0),
        a.stype,
        dims,
        None,
    )


# aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
def op_bmm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var dest = v_tensor(args[unsafe_offset=2])
    var dims = _bmm_dims(a, b)
    _check_dest(dest, b.stype, b)  # common_checks: `batch2.options()`
    _bmm_same_dtype(a, b)
    _check_same_device([dest.copy(), a.copy(), b.copy()])
    _blas_out(
        rets,
        None,
        a,
        b,
        _unit(a.stype, 1),
        _unit(a.stype, 0),
        a.stype,
        dims,
        Optional[T](dest.copy()),
    )


def _bmm_dtype_checks(b1: T, b2: T, out_dtype: Int32) raises -> List[Int]:
    """baddbmm_bmm_out_dtype_checks (cuda/Blas.cpp)."""
    var dims = _bmm_dims(b1, b2)
    if b1.stype != b2.stype:
        raise Error("batch1 and batch2 must have the same dtype")
    if not _float_out_dtype_ok(b1.stype, out_dtype):
        raise Error(
            "out_dtype must be the same as input dtype or fp32 for fp16/bf16"
            " inputs"
        )
    return dims^


# aten::bmm.dtype(Tensor self, Tensor mat2, ScalarType out_dtype) -> Tensor
def op_bmm_dtype(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var out_dtype = v_dtype_or(args[unsafe_offset=2], a.stype)
    var dims = _bmm_dtype_checks(a, b, out_dtype)
    _check_same_device([a.copy(), b.copy()])
    _blas_out(
        rets,
        None,
        a,
        b,
        _unit(a.stype, 1),
        _unit(a.stype, 0),
        out_dtype,
        dims,
        None,
    )


# aten::bmm.dtype_out(Tensor self, Tensor mat2, ScalarType out_dtype, *,
#                     Tensor(a!) out) -> Tensor(a!)
def op_bmm_dtype_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var out_dtype = v_dtype_or(args[unsafe_offset=2], a.stype)
    var dest = v_tensor(args[unsafe_offset=3])
    var dims = _bmm_dtype_checks(a, b, out_dtype)
    if dest.stype != out_dtype:
        raise Error(
            "out_dtype must be the same as the dtype of the provided out tensor"
        )
    _check_same_device([dest.copy(), a.copy(), b.copy()])
    _blas_out(
        rets,
        None,
        a,
        b,
        _unit(a.stype, 1),
        _unit(a.stype, 0),
        out_dtype,
        dims,
        Optional[T](dest.copy()),
    )


# --- aten::addmm / aten::_addmm_activation ------------------------------------


def _addmm_meta(self: T, mat1: T, mat2: T, dtype_overload: Bool) raises:
    """ADDMM_META (LinearAlgebra.cpp); the `.dtype` overloads instead let
    self be the out dtype (checked by their caller)."""
    if not dtype_overload and self.stype != mat2.stype:
        raise Error(
            "self and mat2 must have the same dtype, but got ",
            _scalar_type_name(self.dtype),
            " and ",
            _scalar_type_name(mat2.dtype),
        )
    if mat1.rank != 2:
        raise Error("mat1 must be a matrix, got ", mat1.rank, "-D tensor")
    if mat2.rank != 2:
        raise Error("mat2 must be a matrix, got ", mat2.rank, "-D tensor")
    if mat1.dim(1) != mat2.dim(0):
        raise Error(
            "mat1 and mat2 shapes cannot be multiplied (",
            _matrix_str(mat1),
            " and ",
            _matrix_str(mat2),
            ")",
        )
    if mat1.stype != mat2.stype:
        raise Error(
            "mat1 and mat2 must have the same dtype, but got ",
            _scalar_type_name(mat1.dtype),
            " and ",
            _scalar_type_name(mat2.dtype),
        )


def _addmm_run(
    rets: Values,
    self: T,
    mat1: T,
    mat2: T,
    beta_v: Value,
    alpha_v: Value,
    out_stype: Int32,
    dest: Optional[T],
    activation: Int,
) raises:
    """addmm_out_cuda_impl after the meta checks: `self` broadcast to the
    result (checked only when beta != 0: CUDA never reads it otherwise),
    the k == 0 shortcut `beta * self` with beta in self's dtype, and the
    bias-fused GEMM routes for the common unit-scaled call."""
    var ts = List[T]()
    if dest:
        ts.append(dest.value().copy())
    ts.append(self.copy())
    ts.append(mat1.copy())
    ts.append(mat2.copy())
    _check_same_device(ts)
    var m = mat1.dim(0)
    var k = mat1.dim(1)
    var n = mat2.dim(1)
    var dims: List[Int] = [m, n]
    if not _raw_zero(beta_v):
        _check_expand(self, dims, "addmm")
    var coefs = _coefs(alpha_v, beta_v, mat1.stype, self.stype, m * n, k)
    var alpha = coefs[0]
    var beta = coefs[1]
    var act = activation
    if k == 0:
        # CUDA's k == 0 shortcut returns `beta * self` before the epilogue
        # that would apply _addmm_activation's relu / gelu.
        act = ACT_NONE
    _dest_overlap(dest, dims)
    # The bias-fused GEMM routes take a bias vector (`self` of rank 1); any
    # other self is the cuBLAS epilogue's `C`, added in the compute type
    # before the one rounding (a half product past the dtype's range can
    # come back into it), which those routes' separate add would not do.
    if (
        alpha.one()
        and beta.one()
        and self.rank == 1
        and out_stype == mat1.stype
        and self.stype == mat1.stype
        and _is_float(mat1.dtype)
        and m > 0
        and n > 0
        and k > 0
    ):
        var fused = _addmm_route(self, mat1, mat2)
        if fused:
            _hand_back(rets, dest, Res(fused.value().copy(), True), dims, act)
            return
    _blas_out(
        rets,
        Optional[T](self.copy()),
        mat1,
        mat2,
        alpha,
        beta,
        out_stype,
        dims,
        dest,
        act,
    )


# aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1,
#             Scalar alpha=1) -> Tensor
def op_addmm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var mat1 = v_tensor(args[unsafe_offset=1])
    var mat2 = v_tensor(args[unsafe_offset=2])
    _addmm_meta(self, mat1, mat2, False)
    _addmm_run(
        rets,
        self,
        mat1,
        mat2,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        mat1.stype,
        None,
        ACT_NONE,
    )


# aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1,
#                 Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
def op_addmm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var mat1 = v_tensor(args[unsafe_offset=1])
    var mat2 = v_tensor(args[unsafe_offset=2])
    var dest = v_tensor(args[unsafe_offset=5])
    _addmm_meta(self, mat1, mat2, False)
    # ADDMM_META sets the output from `mat1.options()`.
    _check_dest(dest, mat1.stype, mat1)
    _addmm_run(
        rets,
        self,
        mat1,
        mat2,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        mat1.stype,
        Optional[T](dest.copy()),
        ACT_NONE,
    )


# aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1,
#              Scalar alpha=1) -> Tensor(a!)
def op_addmm_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var mat1 = v_tensor(args[unsafe_offset=1])
    var mat2 = v_tensor(args[unsafe_offset=2])
    _addmm_meta(self, mat1, mat2, False)
    _check_inplace(self, [mat1.dim(0), mat2.dim(1)])
    _addmm_run(
        rets,
        self,
        mat1,
        mat2,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        mat1.stype,
        Optional[T](self.copy()),
        ACT_NONE,
    )


def _addmm_dtype_checks(self: T, mat1: T, mat2: T, out_dtype: Int32) raises:
    """_addmm_dtype_out_cuda (cuda/Blas.cpp), less its `out` check."""
    _addmm_meta(self, mat1, mat2, True)
    if not _float_out_dtype_ok(mat1.stype, out_dtype):
        raise Error(
            "out_dtype must be the same as input dtype or fp32 for fp16/bf16"
            " inputs"
        )


def _addmm_dtype_self(self: T, mat1: T, out_dtype: Int32) raises:
    if self.stype != out_dtype and self.stype != mat1.stype:
        raise Error("self dtype must match either out_dtype or mat1 dtype")


# aten::addmm.dtype(Tensor self, Tensor mat1, Tensor mat2,
#                   ScalarType out_dtype, *, Scalar beta=1, Scalar alpha=1)
#                   -> Tensor
def op_addmm_dtype(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var mat1 = v_tensor(args[unsafe_offset=1])
    var mat2 = v_tensor(args[unsafe_offset=2])
    var out_dtype = v_dtype_or(args[unsafe_offset=3], mat1.stype)
    _addmm_dtype_checks(self, mat1, mat2, out_dtype)
    _addmm_dtype_self(self, mat1, out_dtype)
    _addmm_run(
        rets,
        self,
        mat1,
        mat2,
        args[unsafe_offset=4],
        args[unsafe_offset=5],
        out_dtype,
        None,
        ACT_NONE,
    )


# aten::addmm.dtype_out(Tensor self, Tensor mat1, Tensor mat2,
#                       ScalarType out_dtype, *, Scalar beta=1,
#                       Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
def op_addmm_dtype_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var mat1 = v_tensor(args[unsafe_offset=1])
    var mat2 = v_tensor(args[unsafe_offset=2])
    var out_dtype = v_dtype_or(args[unsafe_offset=3], mat1.stype)
    var dest = v_tensor(args[unsafe_offset=6])
    _addmm_dtype_checks(self, mat1, mat2, out_dtype)
    if dest.stype != out_dtype:
        raise Error(
            "out_dtype must be the same as the dtype of the provided out tensor"
        )
    _addmm_dtype_self(self, mat1, out_dtype)
    _addmm_run(
        rets,
        self,
        mat1,
        mat2,
        args[unsafe_offset=4],
        args[unsafe_offset=5],
        out_dtype,
        Optional[T](dest.copy()),
        ACT_NONE,
    )


# aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *,
#                         Scalar beta=1, Scalar alpha=1, bool use_gelu=False)
#                         -> Tensor
def op_addmm_activation(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var mat1 = v_tensor(args[unsafe_offset=1])
    var mat2 = v_tensor(args[unsafe_offset=2])
    _addmm_meta(self, mat1, mat2, False)
    _addmm_run(
        rets,
        self,
        mat1,
        mat2,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        mat1.stype,
        None,
        ACT_GELU if v_bool(args[unsafe_offset=5]) else ACT_RELU,
    )


# aten::_addmm_activation.out(Tensor self, Tensor mat1, Tensor mat2, *,
#                             Scalar beta=1, Scalar alpha=1,
#                             bool use_gelu=False, Tensor(a!) out)
#                             -> Tensor(a!)
def op_addmm_activation_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var mat1 = v_tensor(args[unsafe_offset=1])
    var mat2 = v_tensor(args[unsafe_offset=2])
    var dest = v_tensor(args[unsafe_offset=6])
    _addmm_meta(self, mat1, mat2, False)
    _check_dest(dest, mat1.stype, mat1)
    _addmm_run(
        rets,
        self,
        mat1,
        mat2,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        mat1.stype,
        Optional[T](dest.copy()),
        ACT_GELU if v_bool(args[unsafe_offset=5]) else ACT_RELU,
    )


# --- aten::baddbmm ------------------------------------------------------------


def _baddbmm_run(
    rets: Values,
    self: T,
    b1: T,
    b2: T,
    beta_v: Value,
    alpha_v: Value,
    out_stype: Int32,
    dims: List[Int],
    dest: Optional[T],
) raises:
    """baddbmm_out_cuda_impl: alpha and beta as opmath (the k == 0 shortcut
    multiplies by the unrounded beta, unlike addmm's)."""
    _bmm_same_dtype(b1, b2)
    var ts = List[T]()
    if dest:
        ts.append(dest.value().copy())
    ts.append(self.copy())
    ts.append(b1.copy())
    ts.append(b2.copy())
    _check_same_device(ts)
    var zero = _unit(b1.stype, 0)
    var alpha = zero
    var beta = zero
    var k = b1.dim(2)
    if _prod(dims) > 0:
        if k > 0:
            alpha = _coef(alpha_v, b1.stype)
            beta = _coef(beta_v, b1.stype)
        elif not _raw_zero(beta_v):
            beta = _mul_scalar(beta_v, b1.stype)
    _blas_out(
        rets,
        Optional[T](self.copy()),
        b1,
        b2,
        alpha,
        beta,
        out_stype,
        dims,
        dest,
    )


def _baddbmm_meta(self: T, b1: T, b2: T) raises -> List[Int]:
    """TORCH_META_FUNC(baddbmm): self broadcasts to the result whatever
    beta is, and must have batch1's dtype."""
    if b1.rank == 3 and b2.rank == 3:
        _check_expand(self, [b1.dim(0), b1.dim(1), b2.dim(2)], "baddbmm")
    if self.stype != b1.stype:
        raise Error(
            "Input dtypes must be the same, got: input ",
            dtype_name(self.stype),
            ", batch1: ",
            dtype_name(b1.stype),
            ", batch2: ",
            dtype_name(b2.stype),
        )
    return _bmm_dims(b1, b2)


# aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1,
#               Scalar alpha=1) -> Tensor
def op_baddbmm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var b1 = v_tensor(args[unsafe_offset=1])
    var b2 = v_tensor(args[unsafe_offset=2])
    var dims = _baddbmm_meta(self, b1, b2)
    _baddbmm_run(
        rets,
        self,
        b1,
        b2,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        b2.stype,
        dims,
        None,
    )


# aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *,
#                   Scalar beta=1, Scalar alpha=1, Tensor(a!) out)
#                   -> Tensor(a!)
def op_baddbmm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var b1 = v_tensor(args[unsafe_offset=1])
    var b2 = v_tensor(args[unsafe_offset=2])
    var dest = v_tensor(args[unsafe_offset=5])
    var dims = _baddbmm_meta(self, b1, b2)
    _check_dest(dest, b2.stype, b2)
    _baddbmm_run(
        rets,
        self,
        b1,
        b2,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        b2.stype,
        dims,
        Optional[T](dest.copy()),
    )


# aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *,
#                Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
def op_baddbmm_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var b1 = v_tensor(args[unsafe_offset=1])
    var b2 = v_tensor(args[unsafe_offset=2])
    var dims = _baddbmm_meta(self, b1, b2)
    if not _has_dims(self, dims):
        raise Error(
            "Expected an output tensor with shape ",
            _dims_str(dims),
            " but got shape ",
            _dims_str(self.logical_shape()),
        )
    _baddbmm_run(
        rets,
        self,
        b1,
        b2,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        b2.stype,
        dims,
        Optional[T](self.copy()),
    )


def _baddbmm_dtype_out_checks(
    self: T, b1: T, b2: T, out_dtype: Int32, out_rank: Int, out_dims: List[Int]
) raises -> List[Int]:
    """baddbmm_bmm_out_dtype_checks with `out` as its self_baddbmm, then
    `out.copy_(self)`: self broadcasts to the result whatever beta is."""
    var dims = _bmm_dtype_checks(b1, b2, out_dtype)
    if out_rank != 3:
        raise Error("self must be a 3D tensor")
    if out_dims != dims:
        raise Error("self must have the same shape as the output")
    _check_expand(self, dims, "copy_")
    return dims^


# aten::baddbmm.dtype(Tensor self, Tensor batch1, Tensor batch2,
#                     ScalarType out_dtype, *, Scalar beta=1, Scalar alpha=1)
#                     -> Tensor
def op_baddbmm_dtype(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var b1 = v_tensor(args[unsafe_offset=1])
    var b2 = v_tensor(args[unsafe_offset=2])
    var out_dtype = v_dtype_or(args[unsafe_offset=3], b1.stype)
    if self.stype != out_dtype and self.stype != b1.stype:
        raise Error("self dtype must match either out_dtype or batch1 dtype")
    var dims = _bmm_dims(b1, b2)
    _ = _baddbmm_dtype_out_checks(self, b1, b2, out_dtype, 3, dims)
    _baddbmm_run(
        rets,
        self,
        b1,
        b2,
        args[unsafe_offset=4],
        args[unsafe_offset=5],
        out_dtype,
        dims,
        None,
    )


# aten::baddbmm.dtype_out(Tensor self, Tensor batch1, Tensor batch2,
#                         ScalarType out_dtype, *, Scalar beta=1,
#                         Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
def op_baddbmm_dtype_out(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var b1 = v_tensor(args[unsafe_offset=1])
    var b2 = v_tensor(args[unsafe_offset=2])
    var out_dtype = v_dtype_or(args[unsafe_offset=3], b1.stype)
    var dest = v_tensor(args[unsafe_offset=6])
    var dims = _baddbmm_dtype_out_checks(
        self, b1, b2, out_dtype, dest.rank, dest.logical_shape()
    )
    if dest.stype != out_dtype:
        raise Error(
            "out_dtype must be the same as the dtype of the provided out tensor"
        )
    _baddbmm_run(
        rets,
        self,
        b1,
        b2,
        args[unsafe_offset=4],
        args[unsafe_offset=5],
        out_dtype,
        dims,
        Optional[T](dest.copy()),
    )


# --- aten::addbmm -------------------------------------------------------------


def _addbmm_checks(self: T, b1: T, b2: T) raises -> List[Int]:
    """addbmm_impl_ (LinearAlgebra.cpp) and the addmm_ it runs per batch."""
    if b1.rank != 3:
        raise Error("batch1 must be a 3D tensor")
    if b2.rank != 3:
        raise Error("batch2 must be a 3D tensor")
    if b1.dim(0) != b2.dim(0):
        raise Error(
            "batch1 and batch2 must have same number of batches, got ",
            b1.dim(0),
            " and ",
            b2.dim(0),
        )
    if b1.dim(2) != b2.dim(1):
        raise Error(
            "Incompatible matrix sizes for bmm (",
            b1.dim(1),
            "x",
            b1.dim(2),
            " and ",
            b2.dim(1),
            "x",
            b2.dim(2),
            ")",
        )
    var dims: List[Int] = [b1.dim(1), b2.dim(2)]
    _check_expand(self, dims, "addbmm_out")
    if self.stype != b2.stype:
        raise Error(
            "self and mat2 must have the same dtype, but got ",
            _scalar_type_name(self.dtype),
            " and ",
            _scalar_type_name(b2.dtype),
        )
    if b1.stype != b2.stype:
        raise Error(
            "mat1 and mat2 must have the same dtype, but got ",
            _scalar_type_name(b1.dtype),
            " and ",
            _scalar_type_name(b2.dtype),
        )
    return dims^


def _batch_matrix(t: T, i: Int) raises -> T:
    """`t[i]` of a rank-3 tensor: a rank-2 view (an owned handle)."""
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = t.dim(1)
    shape[MAX_RANK - 1] = t.dim(2)
    var strides = IndexList[MAX_RANK](0)
    strides[MAX_RANK - 2] = t.stride(1)
    strides[MAX_RANK - 1] = t.stride(2)
    return view_strided(t, shape, strides, 2, t.offset + i * t.stride(0))


def _addbmm_run(
    rets: Values,
    self: T,
    b1: T,
    b2: T,
    beta_v: Value,
    alpha_v: Value,
    dims: List[Int],
    dest: Optional[T],
) raises:
    """addbmm_impl_ (LinearAlgebra.cpp, CUDA's too): one addmm_ per batch
    into the result, beta on the first and 1 after, so a float result is
    rounded once per batch exactly where torch rounds it. Integer sums are
    exact in any order: one GEMM over the batches laid side by side along
    k, `[m, B*k] @ [B*k, n]` -- as is the batch-free `beta * self`."""
    var ts = List[T]()
    if dest:
        ts.append(dest.value().copy())
    ts.append(self.copy())
    ts.append(b1.copy())
    ts.append(b2.copy())
    _check_same_device(ts)
    var nb = b1.dim(0)
    var m = b1.dim(1)
    var k = b1.dim(2)
    var n = b2.dim(2)
    _dest_overlap(dest, dims)
    var alpha = _unit(b1.stype, 0)
    var beta = alpha
    if nb == 0:
        if m * n > 0 and not _raw_zero(beta_v):
            beta = _mul_scalar(beta_v, self.stype)
    else:
        # The first batch's addmm_ (its k == 0 shortcut included).
        var coefs = _coefs(alpha_v, beta_v, b1.stype, self.stype, m * n, k)
        alpha = coefs[0]
        beta = coefs[1]
    if nb > 0 and not is_int_stype(b1.stype):
        var a0 = own(_batch_matrix(b1, 0))
        var c0 = own(_batch_matrix(b2, 0))
        var first = _blas(
            Optional[T](self.copy()),
            a0.t,
            c0.t,
            alpha,
            beta,
            b1.stype,
            dims,
            None,
        )
        _ = a0^
        _ = c0^
        var acc = own(first.t.copy())
        if k > 0:
            for i in range(1, nb):
                var ai = own(_batch_matrix(b1, i))
                var ci = own(_batch_matrix(b2, i))
                _ = _blas(
                    Optional[T](acc.t.copy()),
                    ai.t,
                    ci.t,
                    alpha,
                    _unit(b1.stype, 1),
                    b1.stype,
                    dims,
                    Optional[T](acc.t.copy()),
                )
                _ = ai^
                _ = ci^
        _hand_back(rets, dest, Res(acc.take(), True), dims, ACT_NONE)
        return
    # batch1 as [m, B, k] (a permuted view), made dense, then [m, B*k].
    var pshape = IndexList[MAX_RANK](1)
    pshape[MAX_RANK - 3] = m
    pshape[MAX_RANK - 2] = nb
    pshape[MAX_RANK - 1] = k
    var pstrides = IndexList[MAX_RANK](0)
    pstrides[MAX_RANK - 3] = b1.stride(1)
    pstrides[MAX_RANK - 2] = b1.stride(0)
    pstrides[MAX_RANK - 1] = b1.stride(2)
    var perm = own(view_strided(b1, pshape, pstrides, 3, b1.offset))
    var a_dense = own_if_new(contiguous(perm.t), perm.t)
    var a2 = own(_view(a_dense.t, [m, nb * k]))
    var b_dense = own_if_new(contiguous(b2), b2)
    var b2v = own(_view(b_dense.t, [nb * k, n]))
    _blas_out(
        rets,
        Optional[T](self.copy()),
        a2.t,
        b2v.t,
        alpha,
        beta,
        b1.stype,
        dims,
        dest,
    )
    _ = a2^
    _ = b2v^
    _ = a_dense^
    _ = b_dense^
    _ = perm^


# aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1,
#              Scalar alpha=1) -> Tensor
def op_addbmm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var b1 = v_tensor(args[unsafe_offset=1])
    var b2 = v_tensor(args[unsafe_offset=2])
    var dims = _addbmm_checks(self, b1, b2)
    _addbmm_run(
        rets,
        self,
        b1,
        b2,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        dims,
        None,
    )


# aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *,
#                  Scalar beta=1, Scalar alpha=1, Tensor(a!) out)
#                  -> Tensor(a!)
def op_addbmm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var b1 = v_tensor(args[unsafe_offset=1])
    var b2 = v_tensor(args[unsafe_offset=2])
    var dest = v_tensor(args[unsafe_offset=5])
    var dims = _addbmm_checks(self, b1, b2)
    _check_dest(dest, self.stype, self)  # `result.resize_as_(self)`
    _addbmm_run(
        rets,
        self,
        b1,
        b2,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        dims,
        Optional[T](dest.copy()),
    )


# aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *,
#               Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
def op_addbmm_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var b1 = v_tensor(args[unsafe_offset=1])
    var b2 = v_tensor(args[unsafe_offset=2])
    # Composite in ATen (`addbmm_out(self, ..., self)`): a broadcastable
    # self is computed fresh, then resized to the result and written; a
    # self already of the result's shape is accumulated in place.
    var dims = _addbmm_checks(self, b1, b2)
    _addbmm_run(
        rets,
        self,
        b1,
        b2,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        dims,
        Optional[T](self.copy()),
    )


# --- aten::addmv --------------------------------------------------------------


def _addmv_meta(self: T, mat: T, vec: T) raises:
    """TORCH_META_FUNC(addmv) (Blas.cpp)."""
    if not (mat.rank == 2 and vec.rank == 1 and self.rank <= 1):
        raise Error(
            "vector + matrix @ vector expected, got ",
            self.rank,
            ", ",
            mat.rank,
            ", ",
            vec.rank,
        )
    if mat.dim(1) != vec.dim(0) or (
        mat.dim(0) != self.numel and self.numel != 1
    ):
        raise Error(
            "size mismatch, got input (",
            self.dim(0) if self.rank == 1 else self.numel,
            "), mat (",
            _matrix_str(mat),
            "), vec (",
            vec.dim(0),
            ")",
        )
    if self.stype != mat.stype or mat.stype != vec.stype:
        raise Error(
            "addmv input tensors must have the same dtype, but got ",
            _scalar_type_name(self.dtype),
            ", ",
            _scalar_type_name(mat.dtype),
            ", and ",
            _scalar_type_name(vec.dtype),
        )


def _addmv_run(
    rets: Values,
    self: T,
    mat: T,
    vec: T,
    beta_v: Value,
    alpha_v: Value,
    dest: Optional[T],
) raises:
    """addmv_out_cuda: `mat @ vec` as the GEMM `[m, k] @ [k, 1]` (vec's
    own strides, viewed), alpha and beta in scalar_t like its gemv."""
    var ts = List[T]()
    if dest:
        ts.append(dest.value().copy())
    ts.append(self.copy())
    ts.append(mat.copy())
    ts.append(vec.copy())
    _check_same_device(ts)
    var k = vec.dim(0)
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 2] = k
    var strides = IndexList[MAX_RANK](0)
    strides[MAX_RANK - 2] = vec.stride(0)
    strides[MAX_RANK - 1] = 1
    var col = own(view_strided(vec, shape, strides, 2, vec.offset))
    var coefs = _coefs(
        alpha_v, beta_v, mat.stype, self.stype, mat.dim(0), mat.numel, True
    )
    _blas_out(
        rets,
        Optional[T](self.copy()),
        mat,
        col.t,
        coefs[0],
        coefs[1],
        vec.stype,
        [mat.dim(0)],
        dest,
        gemv=True,
    )
    _ = col^


# aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1,
#             Scalar alpha=1) -> Tensor
def op_addmv(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var mat = v_tensor(args[unsafe_offset=1])
    var vec = v_tensor(args[unsafe_offset=2])
    _addmv_meta(self, mat, vec)
    _addmv_run(
        rets,
        self,
        mat,
        vec,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        None,
    )


# aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1,
#                 Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
def op_addmv_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var mat = v_tensor(args[unsafe_offset=1])
    var vec = v_tensor(args[unsafe_offset=2])
    var dest = v_tensor(args[unsafe_offset=5])
    _addmv_meta(self, mat, vec)
    _check_dest(dest, vec.stype, vec)  # the meta's `vec.options()`
    _addmv_run(
        rets,
        self,
        mat,
        vec,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        Optional[T](dest.copy()),
    )


# aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1,
#              Scalar alpha=1) -> Tensor(a!)
def op_addmv_(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var self = v_tensor(args[unsafe_offset=0])
    var mat = v_tensor(args[unsafe_offset=1])
    var vec = v_tensor(args[unsafe_offset=2])
    _addmv_meta(self, mat, vec)
    _check_inplace(self, [mat.dim(0)])
    _addmv_run(
        rets,
        self,
        mat,
        vec,
        args[unsafe_offset=3],
        args[unsafe_offset=4],
        Optional[T](self.copy()),
    )


# --- aten::_int_mm / aten::_weight_int8pack_mm --------------------------------


def _expect_dtype(t: T, stype: Int32) raises:
    """`Tensor::data_ptr<scalar_t>()`'s check."""
    if t.stype != stype:
        raise Error(
            "expected scalar type ",
            _scalar_type_name(max_dtype(stype)),
            " but found ",
            _scalar_type_name(t.dtype),
        )


def _int_mm_checks(a: T, b: T) raises:
    """_int_mm_out_cuda (cuda/Blas.cpp): cuBLASLt's int8 GEMM limits."""
    if a.rank != 2:
        raise Error("Expected self to be of dimension 2 but got ", a.rank)
    if b.rank != 2:
        raise Error("Expected mat2 to be of dimension 2 but got ", b.rank)
    if a.dim(0) <= 16:
        raise Error(
            "self.size(0) needs to be greater than 16, but got ", a.dim(0)
        )
    if a.dim(1) <= 0 or a.dim(1) % 8 != 0:
        raise Error(
            (
                "self.size(1) needs to be greater than 0 and a multiple of 8,"
                " but got "
            ),
            a.dim(1),
        )
    if a.dim(1) != b.dim(0):
        raise Error(
            "self.size(1) needs to match mat2.size(0) but got ",
            a.dim(1),
            " and ",
            b.dim(0),
        )
    if b.dim(1) <= 0 or b.dim(1) % 8 != 0:
        raise Error(
            (
                "mat2.size(1) needs to be greater than 0 and a multiple of 8,"
                " but got "
            ),
            b.dim(1),
        )
    _expect_dtype(a, ST_INT8)
    _expect_dtype(b, ST_INT8)


# aten::_int_mm(Tensor self, Tensor mat2) -> Tensor
def op_int_mm(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    _int_mm_checks(a, b)
    _check_same_device([a.copy(), b.copy()])
    var out = own(_int_product(a, b, ST_INT32))
    ret_owned(rets, 0, out)


# aten::_int_mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out)
#                   -> Tensor(a!)
def op_int_mm_out(args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    var a = v_tensor(args[unsafe_offset=0])
    var b = v_tensor(args[unsafe_offset=1])
    var dest = v_tensor(args[unsafe_offset=2])
    _int_mm_checks(a, b)
    if dest.stype != ST_INT32:
        raise Error(
            "Expected result dtype to be of type kInt but got ",
            dtype_name(dest.stype),
        )
    if dest.rank != 2:
        raise Error("Expected result to be of dimension 2 but got ", dest.rank)
    if dest.dim(0) != a.dim(0):
        raise Error(
            "Expected result.size(0) to be ",
            a.dim(0),
            " but got ",
            dest.dim(0),
        )
    if dest.dim(1) != b.dim(1):
        raise Error(
            "Expected result.size(1) to be ",
            b.dim(1),
            " but got ",
            dest.dim(1),
        )
    if not dest.contig:
        raise Error("Expected result to be contiguous.")
    _check_same_device([dest.copy(), a.copy(), b.copy()])
    var out = own(_int_product(a, b, ST_INT32))
    copy_strided_into(dest, out.t)
    _ = out^
    ret_ref(rets, 0, dest)


# aten::_weight_int8pack_mm(Tensor self, Tensor mat2, Tensor scales)
#                           -> Tensor
def op_weight_int8pack_mm(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    """_weight_int8pack_mm_cuda (cuda/int8mm.cu): `x @ w.T` in float32 --
    x and the int8 weight widened exactly, a strict-fp32 dot per output --
    times the float32 per-row scale, then cast to x's dtype."""
    var x = v_tensor(args[unsafe_offset=0])
    var w = v_tensor(args[unsafe_offset=1])
    var scales = v_tensor(args[unsafe_offset=2])
    if x.rank != 2:
        raise Error("x must be 2D")
    if w.rank != 2:
        raise Error("w must be 2D")
    if scales.rank != 1:
        raise Error("scale must be 1D")
    if x.dim(1) != w.dim(1):
        raise Error("K dimension mismatch: x.size(1) != w.size(1)")
    if w.dim(0) != scales.dim(0):
        raise Error("Output dim mismatch: w.size(0) != scale.size(0)")
    if not is_float_stype(x.stype):
        raise Error("expected x to be f32/f16/bf16, got ", dtype_name(x.stype))
    _expect_dtype(w, ST_INT8)
    if not is_float_stype(scales.stype):
        raise Error(
            "expected scales to be floating point, got ",
            dtype_name(scales.stype),
        )
    _check_same_device([x.copy(), w.copy(), scales.copy()])
    var m = x.dim(0)
    var n = w.dim(0)
    var k = x.dim(1)
    if m == 0 or n == 0 or k == 0:
        var r = own(_zeros([m, n], x.stype, x.device))
        ret_owned(rets, 0, r)
        return
    var x32 = own_if_new(cast_to(x, ST_FLOAT32), x)
    var none = _none_side()
    var w32 = own(
        _pw_run(
            "widen",
            1,
            _b_tside(w),
            none,
            none.copy(),
            ST_INT8,
            ST_FLOAT32,
            _p(),
            None,
        ).t.copy()
    )
    var p = _spec_matmul("MatmulSpec", x32.t, w32.t, None, 1)
    _ = x32^
    _ = w32^
    if not p:
        unsupported("aten::_weight_int8pack_mm on this device")
    var prod = own(p.value().copy())
    var res = _pw_run(
        "masked_scale",
        2,
        _b_tside(prod.t),
        _b_tside(scales),
        none,
        ST_FLOAT32,
        x.stype,
        _p(1.0),
        None,
    )
    _ = prod^
    ret_tensor(rets, 0, res.t)


# --- aten::_convert_weight_to_int4pack / aten::_weight_int4pack_mm ------------
#
# The packed weight is opaque: only _weight_int4pack_mm reads it. Its layout
# here is this backend's own -- the `[n, k / 2]` uint8 input's bytes as they
# are (two int4 values per byte, the even k in the high nibble), rows padded
# with zeros to a multiple of 8 -- stored in the int32 tensor of the shape
# torch's meta function gives every device, `[ceil(n / 8), k / (innerKTiles
# * 16), 32, innerKTiles / 2]`, so torch.compile's fake tensors agree. The
# matmul dequantizes the weight as tinygemm does (cuda/int4mm.cu: `(q - 8) *
# scale + zero` per group, rounded once to the activation dtype) and runs
# the GEMM routes on it.


def _torch_check(ok: Bool, cond: StaticString) raises:
    """A bare TORCH_CHECK(cond)'s message."""
    if not ok:
        raise Error(
            "Expected ",
            cond,
            (
                " to be true, but got false.  (Could this error message be"
                " improved?  If so, please report an enhancement request to"
                " PyTorch.)"
            ),
        )


def _as_bytes(t: T) raises -> T:
    """`t.view(torch.uint8)` of a contiguous tensor, flattened to 1-D (an
    owned handle)."""
    var r = call_op(
        "aten::view",
        "dtype",
        [tensor_arg(t), Value(TAG_DTYPE, 0, Int64(ST_UINT8), 0)],
        1,
    )
    var bytes = own(r.take_tensor(0))
    var flat = _view(bytes.t, [bytes.t.numel])
    _ = bytes^
    return flat^


# aten::_convert_weight_to_int4pack(Tensor self, int innerKTiles) -> Tensor
def op_convert_weight_to_int4pack(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var w = v_tensor(args[unsafe_offset=0])
    var inner_k_tiles = v_int(args[unsafe_offset=1])
    _torch_check(w.rank == 2, "in.dim() == 2")
    _torch_check(w.stype == ST_UINT8, "in.dtype() == at::kByte")
    _torch_check(w.contig, "in.is_contiguous()")
    _torch_check(
        inner_k_tiles == 2 or inner_k_tiles == 4 or inner_k_tiles == 8,
        "innerKTiles == 2 || innerKTiles == 4 || innerKTiles == 8",
    )
    var n = w.dim(0)
    var half_k = w.dim(1)
    _torch_check(
        (half_k * 2) % (inner_k_tiles * 16) == 0,
        "isEvenDivisor(in.size(1) * 2, innerKTiles * kKTileSize)",
    )
    if not w.on_mojo():
        unsupported("_convert_weight_to_int4pack of a tensor off the device")
    var n_tiles = (n + 7) // 8
    var out = own(
        _zeros(
            [
                n_tiles,
                half_k * 2 // (inner_k_tiles * 16),
                32,
                inner_k_tiles // 2,
            ],
            ST_INT32,
            w.device,
        )
    )
    if n > 0 and half_k > 0:
        var bytes = own(_as_bytes(out.t))
        var rows = own(_view(bytes.t, [n_tiles * 8, half_k]))
        var dst = own(
            view_strided(rows.t, w.shape, w.strides, 2, rows.t.offset)
        )
        copy_strided_into(dst.t, w)
        _ = dst^
        _ = rows^
        _ = bytes^
    ret_owned(rets, 0, out)


# aten::_weight_int4pack_mm(Tensor self, Tensor mat2, int qGroupSize,
#                           Tensor qScaleAndZeros) -> Tensor
def op_weight_int4pack_mm(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var x = v_tensor(args[unsafe_offset=0])
    var packed = v_tensor(args[unsafe_offset=1])
    var group = v_int(args[unsafe_offset=2])
    var qsz = v_tensor(args[unsafe_offset=3])
    _check_same_device([x.copy(), packed.copy(), qsz.copy()])
    _torch_check(packed.rank == 4, "B.dim() == 4")
    var inner_k_tiles = packed.dim(3) * 2
    _torch_check(
        inner_k_tiles == 2 or inner_k_tiles == 4 or inner_k_tiles == 8,
        "B_innerKTiles == 2 || B_innerKTiles == 4 || B_innerKTiles == 8",
    )
    if not is_float_stype(x.stype) or x.stype == ST_FLOAT64:
        raise Error("expected x to be f32/f16/bf16, got ", dtype_name(x.stype))
    _torch_check(x.rank == 2, "A.dim() == 2")
    _torch_check(packed.stype == ST_INT32, "B.dtype() == at::kInt")
    _torch_check(packed.contig, "B.is_contiguous()")
    var m = x.dim(0)
    var k = x.dim(1)
    var n = packed.dim(0) * 8
    _torch_check(
        packed.dim(1) * inner_k_tiles * 16 == k,
        "B.size(1) == k / (B_innerKTiles * kKTileSize)",
    )
    _torch_check(packed.dim(2) == 32, "B.size(2) == 32")
    _torch_check(
        group == 32 or group == 64 or group == 128 or group == 256,
        (
            "qGroupSize == 32 || qGroupSize == 64 || qGroupSize == 128 ||"
            " qGroupSize == 256"
        ),
    )
    _torch_check(qsz.rank == 3, "qScaleAndZeros.dim() == 3")
    _torch_check(
        k >= group and k % group == 0,
        (
            "kTiles * kKTileSize >= qGroupSize && isEvenDivisor(kTiles *"
            " kKTileSize, qGroupSize)"
        ),
    )
    _torch_check(
        qsz.dim(0) == k // group, "qScaleAndZeros.size(0) == k / qGroupSize"
    )
    _torch_check(qsz.dim(1) == n, "qScaleAndZeros.size(1) == n")
    _torch_check(qsz.dim(2) == 2, "qScaleAndZeros.size(2) == 2")
    if qsz.stype != x.stype:
        raise Error(
            "expected qScaleAndZeros to have dtype ",
            dtype_name(x.stype),
            ", got ",
            dtype_name(qsz.stype),
        )
    if m == 0 or n == 0 or k == 0:
        var r = own(_zeros([m, n], x.stype, x.device))
        ret_owned(rets, 0, r)
        return
    var groups = k // group
    # The packed bytes, each read twice: [n, k / 2, 2] with a zero stride.
    var bytes = own(_as_bytes(packed))
    var shape = IndexList[MAX_RANK](1)
    shape[MAX_RANK - 3] = n
    shape[MAX_RANK - 2] = k // 2
    shape[MAX_RANK - 1] = 2
    var strides = IndexList[MAX_RANK](0)
    strides[MAX_RANK - 3] = k // 2
    strides[MAX_RANK - 2] = 1
    var twice = own(view_strided(bytes.t, shape, strides, 3, bytes.t.offset))
    var parity = own(_zeros([2], ST_UINT8, x.device))  # [0, 1]
    var odd = own(_view(parity.t, [1]))
    var odd_one = own(
        view_strided(
            parity.t, odd.t.shape, odd.t.strides, 1, parity.t.offset + 1
        )
    )
    fill_value(odd_one.t, 1.0)
    _ = odd_one^
    _ = odd^
    var none = _none_side()
    var q = own(
        _pw_run(
            "int4_nibble",
            2,
            _b_tside(twice.t),
            _b_tside(parity.t),
            none,
            ST_UINT8,
            ST_UINT8,
            _p(),
            None,
        ).t.copy()
    )
    _ = twice^
    _ = bytes^
    _ = parity^
    var qg = own(_view(q.t, [n, groups, group]))
    # scale / zero of (row, group) at qScaleAndZeros[group, row, 0 / 1].
    var gshape = IndexList[MAX_RANK](1)
    gshape[MAX_RANK - 3] = n
    gshape[MAX_RANK - 2] = groups
    gshape[MAX_RANK - 1] = group
    var gstrides = IndexList[MAX_RANK](0)
    gstrides[MAX_RANK - 3] = qsz.stride(1)
    gstrides[MAX_RANK - 2] = qsz.stride(0)
    var scale = own(view_strided(qsz, gshape, gstrides, 3, qsz.offset))
    var zero = own(
        view_strided(qsz, gshape, gstrides, 3, qsz.offset + qsz.stride(2))
    )
    var w = own(
        _pw_run(
            "int4_dequant",
            3,
            _b_tside(qg.t),
            _b_tside(scale.t),
            _b_tside(zero.t),
            ST_FLOAT32,
            x.stype,
            _p(),
            None,
        ).t.copy()
    )
    _ = qg^
    _ = q^
    _ = scale^
    _ = zero^
    # x @ w.T: the transposed view of the dequantized [n, k] weight.
    var wshape = IndexList[MAX_RANK](1)
    wshape[MAX_RANK - 2] = k
    wshape[MAX_RANK - 1] = n
    var wstrides = IndexList[MAX_RANK](0)
    wstrides[MAX_RANK - 2] = 1
    wstrides[MAX_RANK - 1] = k
    var wt = own(view_strided(w.t, wshape, wstrides, 2, w.t.offset))
    var out = own(_product(x, wt.t, x.stype))
    _ = wt^
    _ = w^
    ret_owned(rets, 0, out)


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
    """ATen's `linear` (Linear.cpp), branch for branch: a 2-D input with a
    bias is `addmm(bias, input, weight.t())`; a bias that is a contiguous
    vector (after squeezing) over a contiguous input is the same addmm on
    the flattened input; everything else is `matmul(input, weight.t())`
    followed by `output.add_(bias)`. The bias-fused GEMM routes serve the
    addmm branches whose bias is a vector, and the bias-free product."""
    var a = v_tensor(args[unsafe_offset=0])
    var w = v_tensor(args[unsafe_offset=1])
    var bias = _opt_tensor_arg(args[unsafe_offset=2])
    if a.rank == 0 or w.rank == 0:
        raise Error(
            "both arguments to linear need to be at least 1D, but they are ",
            a.rank,
            "D and ",
            w.rank,
            "D",
        )
    var addmm_branch = False
    if bias:
        var b = bias.value().copy()
        var squeezed = 0
        for i in range(b.rank):
            if b.dim(i) != 1:
                squeezed += 1
        var fusable = (b.rank == 1 or squeezed == 1) and b.contig
        addmm_branch = a.rank == 2 or (fusable and a.contig)
    if not bias or (addmm_branch and bias.value().rank == 1):
        var out = _linear_route(a, w, bias)
        if out:
            ret_tensor(rets, 0, out.value())
            return
        var vec = _linear_vector(a, w, bias)
        if vec:
            ret_tensor(rets, 0, vec.value())
            return
    _linear_general(rets, a, w, bias, addmm_branch)


def _linear_general(
    rets: Values, a: T, w: T, bias: Optional[T], addmm_branch: Bool
) raises:
    """The branches the fused routes decline, through the BLAS family's
    GEMM + epilogue on the input flattened to `[rows, k]`, viewed back."""
    if w.rank != 2:
        unsupported("aten::linear with a weight of rank " + String(w.rank))
    var k = a.dim(a.rank - 1)
    var n = w.dim(0)
    var lead = _leading_dims(a)
    var rows = _prod(lead)
    if w.dim(1) != k:
        raise Error(
            "mat1 and mat2 shapes cannot be multiplied (",
            rows,
            "x",
            k,
            " and ",
            w.dim(1),
            "x",
            n,
            ")",
        )
    if a.stype != w.stype:
        raise Error(
            "mat1 and mat2 must have the same dtype, but got ",
            _scalar_type_name(a.dtype),
            " and ",
            _scalar_type_name(w.dtype),
        )
    var ts: List[T] = [a.copy(), w.copy()]
    var dims: List[Int] = [rows, n]
    var addend = Optional[T]()
    if addmm_branch:
        # addmm's ADDMM_META and expand_size, on the flattened input
        var b = bias.value().copy()
        if b.stype != w.stype:
            raise Error(
                "self and mat2 must have the same dtype, but got ",
                _scalar_type_name(b.dtype),
                " and ",
                _scalar_type_name(w.dtype),
            )
        _check_expand(b, dims, "addmm")
        ts.append(b.copy())
        addend = b^
    _check_same_device(ts)
    var dense = own_if_new(contiguous(a), a)
    var a2 = own(_view(dense.t, [rows, k]))
    var wshape = IndexList[MAX_RANK](1)
    wshape[MAX_RANK - 2] = k
    wshape[MAX_RANK - 1] = n
    var wstrides = IndexList[MAX_RANK](0)
    wstrides[MAX_RANK - 2] = w.stride(1)
    wstrides[MAX_RANK - 1] = w.stride(0)
    var wt = own(view_strided(w, wshape, wstrides, 2, w.offset))
    var one = _unit(a.stype, 1)
    var res = _blas(addend, a2.t, wt.t, one, one, a.stype, dims, None)
    _ = a2^
    _ = wt^
    _ = dense^
    var flat = own(res.t.copy())
    var shape = lead.copy()
    shape.append(n)
    var shaped = own(_view(flat.t, shape))
    _ = flat^
    if bias and not addmm_branch:
        # `output.add_(bias)`: in place, so the output keeps its shape (a
        # bias that would enlarge it raises) and dtype -- the sum is formed
        # in `result_type(output, bias)` (a 0-d bias does not promote a
        # dimensioned output) and cast back.
        var bias_t = bias.value().copy()
        var common = result_type(shaped.t, bias_t)
        if not can_cast(common, shaped.t.stype):
            raise Error(
                "result type ",
                _scalar_type_name(max_dtype(common)),
                " can't be cast to the desired output type ",
                _scalar_type_name(shaped.t.dtype),
            )
        var acc = own_if_new(cast_to(shaped.t, common), shaped.t)
        # TensorIterator computes in `common`, the operands cast to it.
        var bias_c = own_if_new(cast_to(bias_t, common), bias_t)
        var added = call_op(
            "aten::add_",
            "Tensor",
            [
                tensor_arg(acc.t),
                tensor_arg(bias_c.t),
                Value(TAG_SCALAR_INT, 0, 1, 0),
            ],
            1,
        )
        _ = added^
        _ = bias_c^
        if acc.t.h != shaped.t.h:
            cast_into(shaped.t, acc.t)
        _ = acc^
    ret_owned(rets, 0, shaped)


# --- aten::linear_backward ----------------------------------------------------


def _bool_list(v: Value) raises -> List[Bool]:
    if v.tag != TAG_BOOL_LIST:
        raise Error("expected a bool[] argument, got record tag ", v.tag)
    var p = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=Int(v.a))
    var out = List[Bool](capacity=Int(v.len))
    for i in range(Int(v.len)):
        out.append(p[unsafe_offset=i] != 0)
    return out^


def _sum_dims(a: T, dims: List[Int], out_len: Int) raises -> T:
    """`a.sum(dims)` of a contiguous matrix into a fresh 1-D tensor of the
    one dim left: `[0]` (a leading reduce interval) or `[1]` (trailing), both
    of which the reduction kernels read in place."""
    var out = own(_new([out_len], a.stype, a.device))
    var ctx = ctx_for(a.device)
    var cp = ctx_ptr(ctx)
    var call = KernelCall("reduction", "SumSpec")
    call.arg_dtype(0, a.dtype)
    call.out_dtype(a.dtype)
    call.spec(a.spec(cp))
    call.tuple(dims)
    call.int(0)
    call.spec(out.t.spec(cp))
    call.run()
    _ = ctx
    return out.take()


def _sum_rows(a: T) raises -> T:
    """`a.sum(dim=0)` for a contiguous (rows, cols) matrix (outer 1, reduce
    rows, inner cols: no transposed materialization)."""
    return _sum_dims(a, [0], a.dim(1))


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
    logic.mojo for why the order, not a wider accumulator, is what fp16
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
        var call = KernelCall("logic", "AddrBcast")
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
    if (
        self.dtype == DType.bool
        and vec1.dtype == DType.bool
        and vec2.dtype == DType.bool
    ):
        # A bool addr is `(beta and self) or (alpha and vec1 and vec2)`
        # (addr_kernel's bool branch): beta and alpha taken as bools, so the
        # composite's products stay bool instead of promoting to int64.
        # check_addr_scalar: a bool result is integral, so no float scalar.
        _addr_integral_scalar(beta, "beta")
        _addr_integral_scalar(alpha, "alpha")
        beta = _bool_scalar(v_f64(beta) != 0.0)
        alpha = _bool_scalar(v_f64(alpha) != 0.0)
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


@fieldwise_init
struct ConvGeom(Copyable, ImplicitlyCopyable, Movable):
    """The geometry of one non-transposed convolution, as the 2-D path sees
    it: a rank-3 (conv1d) operand has in_h = kh = out_h = 1 and stride 1,
    padding 0, dilation 1 on that unit H axis."""

    var conv1d: Bool
    var n: Int
    var c: Int
    var in_h: Int
    var in_w: Int
    var out_c: Int
    var c_per_group: Int
    var groups: Int
    var kh: Int
    var kw: Int
    var sh: Int
    var sw: Int
    var ph: Int
    var pw: Int
    var dh: Int
    var dw: Int
    var out_h: Int
    var out_w: Int

    def cols(self) -> Int:
        """Output pixels per sample: the im2col matrix's column count."""
        return self.out_h * self.out_w

    def ckk(self) -> Int:
        """Patch rows over every group: C * KH * KW."""
        return self.c * self.kh * self.kw

    def crs_g(self) -> Int:
        """Patch rows of one group (the weight's row length)."""
        return self.c_per_group * self.kh * self.kw

    def oc_g(self) -> Int:
        return self.out_c // self.groups

    def one_by_one(self) -> Bool:
        """A 1x1 stride-1 unpadded conv: the NCHW input already is the
        (C, H*W) patch matrix of each sample."""
        return (
            self.kh == 1
            and self.kw == 1
            and self.sh == 1
            and self.sw == 1
            and self.ph == 0
            and self.pw == 0
            and self.dh == 1
            and self.dw == 1
        )

    def output_shape(self) -> List[Int]:
        var out = List[Int]()
        out.append(self.n)
        out.append(self.out_c)
        if not self.conv1d:
            out.append(self.out_h)
        out.append(self.out_w)
        return out^

    def patch_params(self) -> List[Int]:
        """The conv family's im2col / col2im params tuple."""
        return [
            self.in_h,
            self.in_w,
            self.out_h,
            self.out_w,
            self.kh,
            self.kw,
            self.sh,
            self.sw,
            self.ph,
            self.pw,
            self.dh,
            self.dw,
            self.c,
            self.n,
        ]


def _conv_geometry(
    input: T,
    weight: T,
    stride: IntList,
    padding: IntList,
    dilation: IntList,
    transposed: Bool,
    groups: Int,
) raises -> Optional[ConvGeom]:
    """What the im2col + GEMM path supports, or None: a non-transposed rank-3
    or rank-4 float convolution of non-empty operands on one device."""
    if transposed or groups < 1:
        return None
    if input.stype != weight.stype or not _is_float(input.dtype):
        return None
    if input.device != weight.device:
        return None
    # Rank 3 (conv1d) is the 2-D path with a unit H axis: a contiguous
    # (N, C, L) input / (K, C, S) weight already has the memory layout of
    # (N, C, 1, L) / (K, C, 1, S), so it runs with in_h = kh = 1 and the one
    # stride/padding/dilation on W, then returns (N, K, out_w).
    var conv1d = input.rank == 3
    if input.rank != weight.rank or (input.rank != 4 and not conv1d):
        return None
    var strides = _pair(stride)
    var pads = _pair(padding)
    var dils = _pair(dilation)
    if conv1d:
        if len(stride) != 1 or len(padding) != 1 or len(dilation) != 1:
            return None
        # H is the unit axis: kernel 1, stride 1, no padding, no dilation.
        strides[0] = 1
        pads[0] = 0
        dils[0] = 1
    if len(strides) != 2 or len(pads) != 2 or len(dils) != 2:
        return None
    var n = input.dim(0)
    var c = input.dim(1)
    var in_h = 1 if conv1d else input.dim(2)
    var in_w = input.dim(input.rank - 1)
    if n == 0 or c == 0 or in_h == 0 or in_w == 0:
        return None
    var out_c = weight.dim(0)
    var c_per_group = weight.dim(1)
    var kh = 1 if conv1d else weight.dim(2)
    var kw = weight.dim(weight.rank - 1)
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
    return ConvGeom(
        conv1d,
        n,
        c,
        in_h,
        in_w,
        out_c,
        c_per_group,
        groups,
        kh,
        kw,
        sh,
        sw,
        ph,
        pw,
        dh,
        dw,
        out_h,
        out_w,
    )


def _conv_patches(
    op: StaticString,
    dst_ptr: Int,
    src_ptr: Int,
    g: ConvGeom,
    dtype: DType,
    cp: Int,
) raises:
    """One conv-family patch kernel: Im2col / Im2colPatchMajor (image ->
    columns) or Col2im (patch-major columns -> image)."""
    var call = KernelCall("conv", String(op))
    call.arg_dtype(0, dtype)
    call.out_dtype(dtype)
    call.int(dst_ptr)
    call.int(src_ptr)
    call.tuple(g.patch_params())
    call.int(dtype_code(dtype))
    call.int(cp)
    call.run()


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
    var geom = _conv_geometry(
        input, weight, stride, padding, dilation, transposed, groups
    )
    if not geom:
        return None
    var g = geom.value()
    if bias:
        if (
            bias.value().stype != input.stype
            or bias.value().rank != 1
            or bias.value().dim(0) != g.out_c
            or bias.value().device != input.device
        ):
            return None

    var n = g.n
    var c = g.c
    var out_c = g.out_c
    var c_per_group = g.c_per_group
    var a = Tmp(input)
    var w = Tmp(weight)
    var device = input.device
    var ctx = ctx_for(device)
    var cp = ctx_ptr(ctx)
    var cols = g.cols()
    var ckk = g.ckk()
    var col = own(_new([0], input.stype, device))
    var col_ptr = a.t.ptr
    if not g.one_by_one():
        # Anything but a 1x1 stride-1 conv builds the patch matrix; for that
        # one case the NCHW input already is the col matrix.
        col = own(_new([n, ckk, cols], input.stype, device))
        _conv_patches("Im2col", col.t.ptr, a.t.ptr, g, input.dtype, cp)
        col_ptr = col.t.ptr

    var out = own(_new(g.output_shape(), input.stype, device))
    if groups == 1:
        var mm = KernelCall("matmul", "Bmm")
        mm.arg_dtype(0, weight.dtype)
        mm.arg_dtype(1, input.dtype)
        mm.out_dtype(input.dtype)
        # Same define shape as every other Bmm site: the shared-A broadcast is
        # RUNTIME data (the trailing 1 in the params tuple, matmul._bmm_go)
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
        var crs_g = g.crs_g()
        var oc_g = g.oc_g()
        var kk = g.kh * g.kw
        for s in range(n):
            for gi in range(groups):
                var mm = KernelCall("matmul", "Matmul")
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
                        (s * out_c + gi * oc_g) * cols,
                        gi * oc_g * crs_g,
                        (s * c + gi * c_per_group) * kk * cols,
                    ]
                )
                mm.int(dtype_code(input.dtype))
                mm.int(cp)
                mm.run()
    if bias:
        var bt = Tmp(bias.value())
        var ba = KernelCall("conv", "BiasAddChan")
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


# --- aten::convolution_backward -----------------------------------------------
#
# The backward folds the batch into the GEMMs' shared dimension. With the
# columns laid out patch-major -- (C*KH*KW, N*OH*OW), one row per filter tap
# holding every sample's output pixels -- and grad_output as (K, N*OH*OW):
#
#   grad_weight = grad_out @ col^T          (a linear: K x C*KH*KW)
#   grad_input  = col2im(weight^T @ grad_out)
#   grad_bias   = grad_out summed over batch and space
#
# so each gradient is ONE GEMM per group through the ordinary ladders
# (`_linear_route` / `_mm_route`: gemm16, tf32, then the spec kernels), with
# no per-sample partials to reduce afterwards. The only data movement beyond
# the forward's is one permuting copy of grad_output (skipped when N == 1)
# and col2im, im2col's adjoint.


def _grad_output_patch_major(grad: T, g: ConvGeom) raises -> Owned:
    """grad_output as a contiguous (K, N, OH*OW) buffer: its sample axis moved
    inside the channel axis, so it reads as the (K, N*OH*OW) GEMM operand.
    For N == 1 that already is NCHW's layout, and a contiguous grad_output is
    used where it lies."""
    if g.n == 1:
        return own_if_new(contiguous(grad), grad)
    var out = own(_new([g.out_c, g.n, g.cols()], grad.stype, grad.device))
    # A (N, K, [OH,] OW) view of that buffer, strided so the copy lands each
    # element at its patch-major place, whatever grad_output's own layout.
    var dims = g.output_shape()
    var shape = _index_list(dims)
    var strides = IndexList[MAX_RANK](0)
    var pad = MAX_RANK - len(dims)
    strides[pad] = g.cols()
    strides[pad + 1] = g.n * g.cols()
    if not g.conv1d:
        strides[pad + 2] = g.out_w
    strides[MAX_RANK - 1] = 1
    var permuted = own(
        view_strided(out.t, shape, strides, len(dims), out.t.offset)
    )
    copy_strided_into(permuted.t, grad)
    _ = permuted^
    return out^


def _rows(t: T, first: Int, count: Int, width: Int) raises -> T:
    """Rows [first, first + count) of a contiguous (*, width) buffer, as a
    (count, width) view (an owned handle)."""
    var shape = IndexList[MAX_RANK](1)
    var strides = IndexList[MAX_RANK](0)
    shape[MAX_RANK - 2] = count
    shape[MAX_RANK - 1] = width
    strides[MAX_RANK - 2] = width
    strides[MAX_RANK - 1] = 1
    return view_strided(t, shape, strides, 2, t.offset + first * width)


def _store_rows(dst: T, first: Int, var block: T) raises:
    """Copy a fresh (count, width) GEMM result into rows [first, ...) of the
    contiguous `dst`, then release it."""
    var held = own(block^)
    var slot = own(_rows(dst, first, held.t.dim(0), held.t.dim(1)))
    copy_strided_into(slot.t, held.t)
    _ = slot^
    _ = held^


def _conv_grad_weight(
    input: T, go: T, g: ConvGeom, cp: Int, weight_shape: List[Int]
) raises -> T:
    """grad_out (K, N*cols) @ col (C*KH*KW, N*cols)^T, one GEMM per group."""
    var rows = g.n * g.cols()
    var x = Tmp(input)
    var cols_t: Owned
    if g.n == 1 and g.one_by_one():
        # One sample of a 1x1 stride-1 conv: NCHW is the (C, H*W) matrix.
        cols_t = own(_view(x.t, [g.ckk(), rows]))
    else:
        cols_t = own(_new([g.ckk(), rows], input.stype, input.device))
        _conv_patches(
            "Im2colPatchMajor", cols_t.t.ptr, x.t.ptr, g, input.dtype, cp
        )
    var oc_g = g.oc_g()
    var crs_g = g.crs_g()
    var out = own(_empty_result(input.stype, input.device))
    if g.groups > 1:
        out = own(_new(weight_shape, input.stype, input.device))
    for gi in range(g.groups):
        var a = own(_rows(go, gi * oc_g, oc_g, rows))
        var b = own(_rows(cols_t.t, gi * crs_g, crs_g, rows))
        var dw = _linear_route(a.t, b.t, None)
        if not dw:
            unsupported("aten::convolution_backward: no GEMM route for wgrad")
        if g.groups == 1:
            var flat = own(dw.value().copy())
            out = own(_view(flat.t, weight_shape))
            _ = flat^
        else:
            _store_rows(out.t, gi * oc_g, dw.value().copy())
        _ = a^
        _ = b^
    _ = cols_t^
    _ = x^
    return out.take()


def _conv_grad_input(
    weight: T, go: T, g: ConvGeom, cp: Int, input_shape: List[Int]
) raises -> T:
    """col2im(weight^T (C*KH*KW, K) @ grad_out (K, N*cols)), one GEMM per
    group into the patch-major column buffer."""
    var rows = g.n * g.cols()
    var w = Tmp(weight)
    var oc_g = g.oc_g()
    var crs_g = g.crs_g()
    var dcol = own(_new([0], weight.stype, weight.device))
    if g.groups > 1:
        dcol = own(_new([g.ckk(), rows], weight.stype, weight.device))
    for gi in range(g.groups):
        # weight^T of group gi: its (oc_g, crs_g) rows read transposed.
        var wt_shape = IndexList[MAX_RANK](1)
        var wt_strides = IndexList[MAX_RANK](0)
        wt_shape[MAX_RANK - 2] = crs_g
        wt_shape[MAX_RANK - 1] = oc_g
        wt_strides[MAX_RANK - 2] = 1
        wt_strides[MAX_RANK - 1] = crs_g
        var wt = own(
            view_strided(
                w.t, wt_shape, wt_strides, 2, w.t.offset + gi * oc_g * crs_g
            )
        )
        var b = own(_rows(go, gi * oc_g, oc_g, rows))
        var part = _mm_route(wt.t, b.t)
        if not part:
            unsupported("aten::convolution_backward: no GEMM route for dgrad")
        if g.groups == 1:
            dcol = own(part.value().copy())
        else:
            _store_rows(dcol.t, gi * crs_g, part.value().copy())
        _ = wt^
        _ = b^
    _ = w^
    if g.n == 1 and g.one_by_one():
        # col2im is the identity: (C, H*W) columns already are NCHW.
        var dx = own(_view(dcol.t, input_shape))
        _ = dcol^
        return dx.take()
    var dx = own(_new(input_shape, weight.stype, weight.device))
    _conv_patches("Col2im", dx.t.ptr, dcol.t.ptr, g, weight.dtype, cp)
    _ = dcol^
    return dx.take()


# aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight,
#   SymInt[]? bias_sizes, SymInt[] stride, SymInt[] padding, SymInt[] dilation,
#   bool transposed, SymInt[] output_padding, SymInt groups,
#   bool[3] output_mask) -> (Tensor, Tensor, Tensor)
def op_convolution_backward(
    args: Values, n_args: Int, rets: Values, n_rets: Int
) raises:
    var grad = v_tensor(args[unsafe_offset=0])
    var input = v_tensor(args[unsafe_offset=1])
    var weight = v_tensor(args[unsafe_offset=2])
    var mask = _bool_list(args[unsafe_offset=10])
    # The forward's own gate: whatever `aten::convolution` ran here (a
    # non-transposed rank-3/4 float conv) has a backward, and a transposed
    # conv never gets this far because its forward declines too.
    var geom = _conv_geometry(
        input,
        weight,
        IntList(args[unsafe_offset=4]),
        IntList(args[unsafe_offset=5]),
        IntList(args[unsafe_offset=6]),
        v_bool(args[unsafe_offset=7]),
        v_int(args[unsafe_offset=9]),
    )
    if len(mask) != 3 or not geom:
        unsupported("aten::convolution_backward with these operands")
    if not mask[0] and not mask[1] and not mask[2]:
        for i in range(3):
            _ret_undefined(rets, i)
        return
    var g = geom.value()
    var expected = g.output_shape()
    if (
        grad.stype != input.stype
        or grad.device != input.device
        or not grad.on_mojo()
        or grad.rank != len(expected)
    ):
        unsupported("aten::convolution_backward: grad_output does not match")
    for i in range(grad.rank):
        if grad.dim(i) != expected[i]:
            unsupported(
                "aten::convolution_backward: grad_output has the wrong shape"
            )
    if g.out_c == 0:
        unsupported("aten::convolution_backward with no output channels")

    var ctx = ctx_for(input.device)
    var cp = ctx_ptr(ctx)
    # All three gradients read grad_output as (K, N*cols); the bias gradient
    # is then a trailing-axis sum the reduction kernels take in place.
    var go3 = _grad_output_patch_major(grad, g)
    var go = own(_view(go3.t, [g.out_c, g.n * g.cols()]))
    _ = go3^  # `go` holds its own reference to the storage

    var grad_input = own(_empty_result(input.stype, input.device))
    if mask[0]:
        grad_input = own(
            _conv_grad_input(weight, go.t, g, cp, input.logical_shape())
        )
    var grad_weight = own(_empty_result(input.stype, input.device))
    if mask[1]:
        grad_weight = own(
            _conv_grad_weight(input, go.t, g, cp, weight.logical_shape())
        )
    var grad_bias = own(_empty_result(input.stype, input.device))
    if mask[2]:
        grad_bias = own(_sum_dims(go.t, [1], g.out_c))
    _ = go^
    _ = ctx

    if mask[0]:
        ret_owned(rets, 0, grad_input)
    else:
        _ret_undefined(rets, 0)
    if mask[1]:
        ret_owned(rets, 1, grad_weight)
    else:
        _ret_undefined(rets, 1)
    if mask[2]:
        ret_owned(rets, 2, grad_bias)
    else:
        _ret_undefined(rets, 2)


def register_matmul(site: Site) raises:
    impl[op_addbmm, "addbmm"](site)
    impl[op_addbmm_out, "addbmm.out"](site)
    impl[op_addbmm_, "addbmm_"](site)
    impl[op_addmm, "addmm"](site)
    impl[op_addmm_out, "addmm.out"](site)
    impl[op_addmm_, "addmm_"](site)
    impl[op_addmm_dtype, "addmm.dtype"](site)
    impl[op_addmm_dtype_out, "addmm.dtype_out"](site)
    impl[op_addmm_activation, "_addmm_activation"](site)
    impl[op_addmm_activation_out, "_addmm_activation.out"](site)
    impl[op_addmv, "addmv"](site)
    impl[op_addmv_out, "addmv.out"](site)
    impl[op_addmv_, "addmv_"](site)
    impl[op_addr, "addr"](site)
    impl[op_baddbmm, "baddbmm"](site)
    impl[op_baddbmm_out, "baddbmm.out"](site)
    impl[op_baddbmm_, "baddbmm_"](site)
    impl[op_baddbmm_dtype, "baddbmm.dtype"](site)
    impl[op_baddbmm_dtype_out, "baddbmm.dtype_out"](site)
    impl[op_bmm, "bmm"](site)
    impl[op_bmm_out, "bmm.out"](site)
    impl[op_bmm_dtype, "bmm.dtype"](site)
    impl[op_bmm_dtype_out, "bmm.dtype_out"](site)
    impl[op_convert_weight_to_int4pack, "_convert_weight_to_int4pack"](site)
    impl[op_int_mm, "_int_mm"](site)
    impl[op_int_mm_out, "_int_mm.out"](site)
    impl[op_weight_int4pack_mm, "_weight_int4pack_mm"](site)
    impl[op_weight_int8pack_mm, "_weight_int8pack_mm"](site)
    impl[op_convolution, "convolution"](site)
    impl[op_convolution_backward, "convolution_backward"](site)
    impl[op_linear, "linear"](site)
    impl[op_linear_backward, "linear_backward"](site)
    impl[op_mm, "mm"](site)
    impl[op_mm_out, "mm.out"](site)
    impl[op_mm_dtype, "mm.dtype"](site)
    impl[op_mm_dtype_out, "mm.dtype_out"](site)
