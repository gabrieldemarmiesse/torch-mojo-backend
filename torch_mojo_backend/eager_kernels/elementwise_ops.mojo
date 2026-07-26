# ===----------------------------------------------------------------------=== #
# Fast eager-mode elementwise kernels for mojo_device.
#
# This module is imported from Python through `mojo.importer`: the first
# import runs `mojo build --emit shared-lib` and caches the resulting
# CPython extension under `__mojocache__/` next to this file (content
# addressed, so editing this file triggers exactly one recompile).
#
# The design mirrors `max._interpreter_ops.elementwise_binary_ops` (the MO
# interpreter's own op bindings): each Python-visible function receives raw
# tensor-data pointers (plain ints, storage offset already applied) plus
# explicit numel/dtype ints and the device's DeviceContext pointer — there
# are no `max.driver.Buffer` objects and no attribute access at all —
# dispatches on dtype at *runtime* (all dtype specializations are compiled
# into this one extension), and enqueues the kernel on MAX's own device
# context — so ordering with regular MAX driver operations (copies, other
# kernels) comes for free.
#
# Every kernel here works on *contiguous* buffers with fully dynamic sizes:
# one compiled extension serves every shape with zero recompilation.
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from std.gpu.host import DeviceContext
from std.math import (
    acos,
    atanh,
    ceil,
    ceildiv,
    cos,
    cosh,
    erf,
    exp,
    floor,
    log,
    log1p,
    pow,
    sin,
    sinh,
    sqrt,
    tanh,
)
from std.python import PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder
from std.sys.info import (
    has_accelerator,
    has_apple_gpu_accelerator,
    is_apple_gpu,
    simd_width_of,
    size_of,
)
from std.utils.index import IndexList
from std.utils.coord import Coord
from std.utils.numerics import isnan

from std.algorithm.functional import elementwise

from foreach_clip_contract import (
    FOREACH_CHUNK_ELEMENTS,
    FOREACH_DESC_CAP,
    FOREACH_THREADS,
    ForeachDesc,
    empty_foreach_desc,
)
from foreach_clip_kernels import _chunk_bounds
from op_utils import (
    FLOAT_DTYPES,
    GS_THREADS,
    MAX_RANK,
    _enqueue_cached,
    _gs_blocks,
    _make_ptr,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_int,
    _raw_tuple_len,
    _scratch_contig,
    _spec_ptr,
    _spec_result,
    _spec_unsupported,
)

# ---------------------------------------------------------------------------
# Raw-pointer calling convention: every Python-visible kernel below receives
# tensor operands as a single int (the `._ptr` address, storage offset
# already applied), unpacked with `_raw_int` and turned into a typed
# pointer with `_make_ptr[dt]`; numel and dtype are explicit int args
# (`_raw_int` / `_raw_dtype_int`); `ctx_ptr` (int) is always last
# (`_raw_ctx`). The dispatchers register as METH_FASTCALL functions
# (`def_py_c_function`), skipping the owning PythonObject wrappers of the
# `def_function` path entirely.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Binary elementwise kernels
# ---------------------------------------------------------------------------

comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_DIV = 3
comptime OP_MAX = 4
comptime OP_MIN = 5


def _bin_contig_kernel4[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lhs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    n4: Int,
):
    comptime vec_align = 4 * size_of[dtype]()
    var c = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while c < n4:
        var i = c * 4
        var a = lhs_ptr.load[width=4, alignment=vec_align](i)
        var b = rhs_ptr.load[width=4, alignment=vec_align](i)
        comptime if op_code == OP_ADD:
            out_ptr.store[width=4, alignment=vec_align](i, a + b)
        comptime if op_code == OP_SUB:
            out_ptr.store[width=4, alignment=vec_align](i, a - b)
        comptime if op_code == OP_MUL:
            out_ptr.store[width=4, alignment=vec_align](i, a * b)
        comptime if op_code == OP_DIV:
            comptime if dtype.is_floating_point():
                out_ptr.store[width=4, alignment=vec_align](i, a / b)
        comptime if op_code == OP_MAX:
            out_ptr.store[width=4, alignment=vec_align](i, max(a, b))
        comptime if op_code == OP_MIN:
            out_ptr.store[width=4, alignment=vec_align](i, min(a, b))
        c += gstride


def _bin_contig_kernel[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lhs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        var a = lhs_ptr[i]
        var b = rhs_ptr[i]
        comptime if op_code == OP_ADD:
            out_ptr[i] = a + b
        comptime if op_code == OP_SUB:
            out_ptr[i] = a - b
        comptime if op_code == OP_MUL:
            out_ptr[i] = a * b
        comptime if op_code == OP_DIV:
            comptime if dtype.is_floating_point():
                out_ptr[i] = a / b
        comptime if op_code == OP_MAX:
            out_ptr[i] = max(a, b)
        comptime if op_code == OP_MIN:
            out_ptr[i] = min(a, b)
        i += gstride


@always_inline
def _bin_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    lhs_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    rhs_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    """out = op(lhs, rhs) over `size` contiguous elements."""

    comptime if op_code == OP_DIV and not dtype.is_floating_point():
        raise Error("integer/bool div is not supported in the fast path")
    else:
        if ctx.api() == "cpu":

            @always_inline
            @parameter
            @__copy_capture(out_ptr, lhs_ptr, rhs_ptr)
            def func[width: Int, alignment: Int = 1](idx: Coord):
                var i = Int(idx[0].value())
                var a = lhs_ptr.load[width=width](i)
                var b = rhs_ptr.load[width=width](i)
                comptime if op_code == OP_ADD:
                    out_ptr.store[width=width](i, a + b)
                comptime if op_code == OP_SUB:
                    out_ptr.store[width=width](i, a - b)
                comptime if op_code == OP_MUL:
                    out_ptr.store[width=width](i, a * b)
                comptime if op_code == OP_DIV:
                    out_ptr.store[width=width](i, a / b)
                comptime if op_code == OP_MAX:
                    out_ptr.store[width=width](i, max(a, b))
                comptime if op_code == OP_MIN:
                    out_ptr.store[width=width](i, min(a, b))

            elementwise[func, simd_width=simd_width_of[dtype]()](
                Coord(size), ctx
            )
        else:
            comptime if has_accelerator():
                comptime if dtype != DType.float64:
                    if size % 4 == 0:
                        var n4 = size // 4
                        _enqueue_cached[_bin_contig_kernel4[dtype, op_code]](
                            ctx,
                            String(t"ew_bin4_{op_code}_{dtype}"),
                            _gs_blocks(n4),
                            1,
                            1,
                            GS_THREADS,
                            out_ptr.as_unsafe_any_origin(),
                            lhs_ptr.as_unsafe_any_origin().as_immutable(),
                            rhs_ptr.as_unsafe_any_origin().as_immutable(),
                            n4,
                        )
                        return
                    _enqueue_cached[_bin_contig_kernel[dtype, op_code]](
                        ctx,
                        String(t"ew_bin_{op_code}_{dtype}"),
                        _gs_blocks(size),
                        1,
                        1,
                        GS_THREADS,
                        out_ptr.as_unsafe_any_origin(),
                        lhs_ptr.as_unsafe_any_origin().as_immutable(),
                        rhs_ptr.as_unsafe_any_origin().as_immutable(),
                        size,
                    )
                else:
                    raise Error("float64 is not supported on GPU")
            else:
                raise Error("no GPU accelerator available at compile time")


def _bin_go[
    op_code: Int
](
    out_ptr: PyObjectPtr,
    lhs_ptr: PyObjectPtr,
    rhs_ptr: PyObjectPtr,
    numel: PyObjectPtr,
    dtype_val: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var lhs_addr = _raw_int(lhs_ptr)
    var rhs_addr = _raw_int(rhs_ptr)
    var size = _raw_int(numel)
    var dtype = _raw_dtype_int(dtype_val)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.float64,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
    ]:
        if dtype == dt:
            _bin_elementwise[dt, op_code](
                _make_ptr[dt](out_addr),
                _make_ptr[dt](lhs_addr),
                _make_ptr[dt](rhs_addr),
                size,
                ctx,
            )
            handled = True
    if not handled:
        abort(
            String("unsupported dtype for fast binary elementwise op: ", dtype)
        )


# ---------------------------------------------------------------------------
# Unary elementwise kernels
#
# Opcodes fall in three buckets:
#   * RELU / ABS / NEG / SIGN work on integer *and* float dtypes and compute
#     directly in the tensor dtype (no float round-trip).
#   * every other opcode is float-only (`_float_unary` below): half-precision
#     inputs are promoted to float32, computed, and cast back — matching
#     torch's numerics and keeping the polynomial math accurate.
# Two of the composed ops deserve a note: `tan` and `asinh` are built from
# sin/cos and log/sqrt rather than the std.math primitives, because those
# lower to libm (`_call_libm`) which `comptime assert`s CPU-only and would
# refuse to compile for the GPU target.
# ---------------------------------------------------------------------------

comptime UOP_RELU = 0
comptime UOP_EXP = 1
comptime UOP_TANH = 2
comptime UOP_ABS = 3
comptime UOP_NEG = 4
comptime UOP_SIGN = 5
comptime UOP_CEIL = 6
comptime UOP_FLOOR = 7
comptime UOP_ACOS = 8
comptime UOP_ASINH = 9
comptime UOP_ATANH = 10
comptime UOP_COS = 11
comptime UOP_COSH = 12
comptime UOP_ERF = 13
comptime UOP_LOG = 14
comptime UOP_LOG1P = 15
comptime UOP_RECIPROCAL = 16
comptime UOP_RSQRT = 17
comptime UOP_SIGMOID = 18
comptime UOP_SILU = 19
comptime UOP_SIN = 20
comptime UOP_SINH = 21
comptime UOP_SQRT = 22
comptime UOP_TAN = 23
comptime UOP_GELU_NONE = 24
comptime UOP_GELU_TANH = 25


@always_inline
def _float_unary[
    dtype: DType, width: Int, op_code: Int
](a: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_floating_point():
    """The float-only unary math, evaluated in `dtype` (float32 or float64).

    Only instantiated for float32/float64 (half inputs are promoted before
    the call), so every std.math call below sees a supported dtype.
    """
    var res = a
    comptime if op_code == UOP_EXP:
        res = exp(a)
    comptime if op_code == UOP_TANH:
        res = tanh(a)
    comptime if op_code == UOP_CEIL:
        res = ceil(a)
    comptime if op_code == UOP_FLOOR:
        res = floor(a)
    comptime if op_code == UOP_ACOS:
        res = acos(a)
    comptime if op_code == UOP_ASINH:
        # asinh(x) = log(x + sqrt(x^2 + 1)); std.math.asinh is libm/CPU-only.
        res = log(a + sqrt(a * a + 1))
    comptime if op_code == UOP_ATANH:
        res = atanh(a)
    comptime if op_code == UOP_COS:
        res = cos(a)
    comptime if op_code == UOP_COSH:
        res = cosh(a)
    comptime if op_code == UOP_ERF:
        res = erf(a)
    comptime if op_code == UOP_LOG:
        res = log(a)
    comptime if op_code == UOP_LOG1P:
        comptime if is_apple_gpu():
            # Mojo's log1p currently upcasts to float64, which Metal rejects.
            # Use the compensated float32 algorithm from PyTorch's Metal
            # support so small nonzero inputs do not collapse to zero.
            var xp1 = 1 + a
            var rc = log(xp1)
            var corrected = rc * (a / (xp1 - 1))
            rc = (a.gt(-0.5) & a.lt(0.5)).select(corrected, rc)
            res = xp1.eq(1).select(a, rc)
        else:
            res = log1p(a)
    comptime if op_code == UOP_RECIPROCAL:
        res = 1 / a
    comptime if op_code == UOP_RSQRT:
        res = 1 / sqrt(a)
    comptime if op_code == UOP_SIGMOID:
        res = 1 / (1 + exp(-a))
    comptime if op_code == UOP_SILU:
        res = a / (1 + exp(-a))
    comptime if op_code == UOP_SIN:
        res = sin(a)
    comptime if op_code == UOP_SINH:
        res = sinh(a)
    comptime if op_code == UOP_SQRT:
        res = sqrt(a)
    comptime if op_code == UOP_TAN:
        # tan(x) = sin(x)/cos(x); std.math.tan is libm/CPU-only.
        res = sin(a) / cos(a)
    comptime if op_code == UOP_GELU_NONE:
        # 0.5 * x * (1 + erf(x / sqrt(2)))
        comptime inv_sqrt2 = 0.70710678118654752440
        res = 0.5 * a * (1 + erf(a * inv_sqrt2))
    comptime if op_code == UOP_GELU_TANH:
        # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        comptime sqrt_2_over_pi = 0.79788456080286535588
        var inner = sqrt_2_over_pi * (a + 0.044715 * a * a * a)
        res = 0.5 * a * (1 + tanh(inner))
    return res


def _unary_contig_kernel[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    size: Int,
):
    comptime is_direct = (
        op_code == UOP_RELU
        or op_code == UOP_ABS
        or op_code == UOP_NEG
        or op_code == UOP_SIGN
    )
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < size:
        var a = in_ptr[i]
        comptime if op_code == UOP_RELU:
            out_ptr[i] = max(a, Scalar[dtype](0))
        comptime if op_code == UOP_ABS:
            out_ptr[i] = abs(a)
        comptime if op_code == UOP_NEG:
            # `-a` (pop.neg) wraps for unsigned/overflow exactly like torch.
            out_ptr[i] = -a
        comptime if op_code == UOP_SIGN:
            var zero = Scalar[dtype](0)
            var pos = a.gt(zero).cast[dtype]()
            var neg = a.lt(zero).cast[dtype]()
            # NaN compares false on both sides -> 0, matching torch.
            out_ptr[i] = pos - neg
        comptime if not is_direct:
            comptime if dtype == DType.float16 or dtype == DType.bfloat16:
                var af = a.cast[DType.float32]()
                out_ptr[i] = _float_unary[DType.float32, 1, op_code](af).cast[
                    dtype
                ]()
            elif dtype.is_floating_point():
                out_ptr[i] = _float_unary[dtype, 1, op_code](a)
        i += gstride


@always_inline
def _unary_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    comptime is_direct = (
        op_code == UOP_RELU
        or op_code == UOP_ABS
        or op_code == UOP_NEG
        or op_code == UOP_SIGN
    )
    comptime if not is_direct and not dtype.is_floating_point():
        # Transcendentals / ceil / floor / gelu require a float dtype; the
        # Python side already gates on this, so this only ever fires as a
        # defensive guard (and keeps the float math out of int instantiations).
        raise Error("this unary op requires a floating point dtype")
    else:
        if ctx.api() == "cpu":

            @always_inline
            @parameter
            @__copy_capture(out_ptr, in_ptr)
            def func[width: Int, alignment: Int = 1](idx: Coord):
                var i = Int(idx[0].value())
                var a = in_ptr.load[width=width](i)
                comptime if op_code == UOP_RELU:
                    out_ptr.store[width=width](i, max(a, SIMD[dtype, width](0)))
                comptime if op_code == UOP_ABS:
                    out_ptr.store[width=width](i, abs(a))
                comptime if op_code == UOP_NEG:
                    # `-a` (pop.neg) wraps for unsigned/overflow like torch.
                    out_ptr.store[width=width](i, -a)
                comptime if op_code == UOP_SIGN:
                    var zero = SIMD[dtype, width](0)
                    var pos = a.gt(zero).cast[dtype]()
                    var neg = a.lt(zero).cast[dtype]()
                    # NaN compares false on both sides -> 0, matching torch.
                    out_ptr.store[width=width](i, pos - neg)
                comptime if not is_direct:
                    comptime if (
                        dtype == DType.float16 or dtype == DType.bfloat16
                    ):
                        var af = a.cast[DType.float32]()
                        out_ptr.store[width=width](
                            i, _float_unary[op_code=op_code](af).cast[dtype]()
                        )
                    elif dtype.is_floating_point():
                        out_ptr.store[width=width](
                            i, _float_unary[op_code=op_code](a)
                        )

            elementwise[func, simd_width=simd_width_of[dtype]()](
                Coord(size), ctx
            )
        else:
            comptime if has_accelerator():
                comptime if dtype != DType.float64:
                    _enqueue_cached[_unary_contig_kernel[dtype, op_code]](
                        ctx,
                        String(t"ew_unary_{op_code}_{dtype}"),
                        _gs_blocks(size),
                        1,
                        1,
                        GS_THREADS,
                        out_ptr.as_unsafe_any_origin(),
                        in_ptr.as_unsafe_any_origin().as_immutable(),
                        size,
                    )
                else:
                    raise Error("float64 is not supported on GPU")
            else:
                raise Error("no GPU accelerator available at compile time")


comptime BUOP_ISNAN = 0
comptime BUOP_LOGICAL_NOT = 1


@always_inline
def _unary_bool[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[DType.bool], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    size: Int,
    ctx: DeviceContext,
) raises:
    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var a = in_ptr.load[width=width](i)
        comptime if op_code == BUOP_ISNAN:
            # `numerics.isnan` is bit-based (llvm.is.fpclass), so it survives
            # the fast-math flags that would fold `a != a` to False; it also
            # returns all-False for integer dtypes.
            out_ptr.store[width=width](i, isnan(a))
        comptime if op_code == BUOP_LOGICAL_NOT:
            out_ptr.store[width=width](i, a.eq(SIMD[dtype, width](0)))

    if ctx.api() == "cpu":
        elementwise[func, simd_width=simd_width_of[dtype]()](Coord(size), ctx)
    else:
        comptime if has_accelerator():
            comptime if (
                dtype == DType.float64 and has_apple_gpu_accelerator()
            ):
                raise Error("float64 is not supported on Apple GPU")
            else:
                elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


comptime SOP_ADD = 0
comptime SOP_MUL = 1
comptime SOP_POW = 2


@always_inline
def _scalar_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    scalar: Float32,
    size: Int,
    ctx: DeviceContext,
) raises:
    comptime if not dtype.is_floating_point():
        raise Error("scalar elementwise ops require a floating point dtype")
    else:

        @always_inline
        @parameter
        @__copy_capture(out_ptr, in_ptr, scalar)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            var a = in_ptr.load[width=width](i).cast[DType.float32]()
            var s = SIMD[DType.float32, width](scalar)
            comptime if op_code == SOP_ADD:
                out_ptr.store[width=width](i, (a + s).cast[dtype]())
            comptime if op_code == SOP_MUL:
                out_ptr.store[width=width](i, (a * s).cast[dtype]())
            comptime if op_code == SOP_POW:
                out_ptr.store[width=width](i, pow(a, s).cast[dtype]())

        if ctx.api() == "cpu":
            elementwise[func, simd_width=simd_width_of[dtype]()](
                Coord(size), ctx
            )
        else:
            comptime if has_accelerator():
                elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
            else:
                raise Error("no GPU accelerator available at compile time")


comptime IOP_ADD = 0
comptime IOP_MUL = 1


@always_inline
def _int_scalar_elementwise[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    in_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    scalar: Int,
    size: Int,
    ctx: DeviceContext,
) raises:
    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, scalar)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var a = in_ptr.load[width=width](i)
        comptime if op_code == IOP_ADD:
            out_ptr.store[width=width](i, a + SIMD[dtype, width](scalar))
        comptime if op_code == IOP_MUL:
            out_ptr.store[width=width](i, a * SIMD[dtype, width](scalar))

    if ctx.api() == "cpu":
        elementwise[func, simd_width=simd_width_of[dtype]()](Coord(size), ctx)
    else:
        comptime if has_accelerator():
            elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


@always_inline
def _fill[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    value: Float64,
    size: Int,
    ctx: DeviceContext,
) raises:
    var scalar = value.cast[dtype]()

    @always_inline
    @parameter
    @__copy_capture(out_ptr, scalar)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        out_ptr.store[width=width](i, SIMD[dtype, width](scalar))

    if ctx.api() == "cpu":
        elementwise[func, simd_width=simd_width_of[dtype]()](Coord(size), ctx)
    else:
        comptime if has_accelerator():
            comptime if (
                dtype == DType.float64 and has_apple_gpu_accelerator()
            ):
                raise Error("float64 is not supported on Apple GPU")
            else:
                elementwise[func, simd_width=1, target="gpu"](Coord(size), ctx)
        else:
            raise Error("no GPU accelerator available at compile time")


@always_inline
def _arange[
    dtype: DType
](
    out_ptr: UnsafePointer[Scalar[dtype], MutUntrackedOrigin],
    start: Float64,
    step: Float64,
    size: Int,
    ctx: DeviceContext,
) raises:
    # Match PyTorch's accumulator types: f32 accumulates in f64 on CPU and
    # f32 on GPU; half/bfloat16 use f32; integral outputs use int64. Metal
    # must never see the f64 closure or even capture a Float64 value.
    comptime if dtype == DType.float32:
        if ctx.api() == "cpu":

            @always_inline
            @parameter
            @__copy_capture(out_ptr, start, step)
            def cpu_f32[width: Int, alignment: Int = 1](idx: Coord):
                var i = Int(idx[0].value())
                out_ptr[i] = (start + Float64(i) * step).cast[dtype]()

            elementwise[cpu_f32, simd_width=1](Coord(size), ctx)
        else:
            var start_f32 = start.cast[DType.float32]()
            var step_f32 = step.cast[DType.float32]()

            @always_inline
            @parameter
            @__copy_capture(out_ptr, start_f32, step_f32)
            def gpu_f32[width: Int, alignment: Int = 1](idx: Coord):
                var i = Int(idx[0].value())
                out_ptr[i] = (
                    start_f32 + Scalar[DType.float32](i) * step_f32
                ).cast[dtype]()

            comptime if has_accelerator():
                elementwise[gpu_f32, simd_width=1, target="gpu"](
                    Coord(size), ctx
                )
            else:
                raise Error("no GPU accelerator available at compile time")
    elif dtype == DType.float16 or dtype == DType.bfloat16:
        var start_f32 = start.cast[DType.float32]()
        var step_f32 = step.cast[DType.float32]()

        @always_inline
        @parameter
        @__copy_capture(out_ptr, start_f32, step_f32)
        def lowp[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            out_ptr[i] = (start_f32 + Scalar[DType.float32](i) * step_f32).cast[
                dtype
            ]()

        if ctx.api() == "cpu":
            elementwise[lowp, simd_width=1](Coord(size), ctx)
        else:
            comptime if has_accelerator():
                elementwise[lowp, simd_width=1, target="gpu"](Coord(size), ctx)
            else:
                raise Error("no GPU accelerator available at compile time")
    elif dtype.is_integral():
        var start_i64 = start.cast[DType.int64]()
        var step_i64 = step.cast[DType.int64]()

        @always_inline
        @parameter
        @__copy_capture(out_ptr, start_i64, step_i64)
        def integral[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            out_ptr[i] = (start_i64 + Scalar[DType.int64](i) * step_i64).cast[
                dtype
            ]()

        if ctx.api() == "cpu":
            elementwise[integral, simd_width=1](Coord(size), ctx)
        else:
            comptime if has_accelerator():
                elementwise[integral, simd_width=1, target="gpu"](
                    Coord(size), ctx
                )
            else:
                raise Error("no GPU accelerator available at compile time")
    else:

        @always_inline
        @parameter
        @__copy_capture(out_ptr, start, step)
        def f64[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            out_ptr[i] = (start + Float64(i) * step).cast[dtype]()

        if ctx.api() == "cpu":
            elementwise[f64, simd_width=1](Coord(size), ctx)
        else:
            comptime if has_apple_gpu_accelerator():
                raise Error("float64 is not supported on Apple GPU")
            elif has_accelerator():
                elementwise[f64, simd_width=1, target="gpu"](Coord(size), ctx)
            else:
                raise Error("no GPU accelerator available at compile time")


def _arange_go(
    out_ptr: PyObjectPtr,
    start: PyObjectPtr,
    step: PyObjectPtr,
    numel: PyObjectPtr,
    dtype_val: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var out_addr = _raw_int(out_ptr)
    var start_val = _raw_f64(start)
    var step_val = _raw_f64(step)
    var size = _raw_int(numel)
    var dtype = _raw_dtype_int(dtype_val)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.float64,
        DType.int64,
        DType.int32,
        DType.int16,
        DType.int8,
        DType.uint8,
    ]:
        if dtype == dt:
            _arange[dt](_make_ptr[dt](out_addr), start_val, step_val, size, ctx)
            handled = True
    if not handled:
        abort(String("unsupported dtype for fast arange: ", dtype))


# ---------------------------------------------------------------------------
# METH_FASTCALL wrappers: raw CPython argument unpacking (no owning
# PythonObject per argument). Argument types are guaranteed by the internal
# Python callers in aten_fast.py; errors cannot cross the C ABI, and the
# only raise sites are unsupported-dtype guards already gated upstream.
# ---------------------------------------------------------------------------


def _bin_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _bin_go[op_code](args[0], args[1], args[2], args[3], args[4], args[5])
    except:
        pass
    return _raw_ret_none()


def _arange_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _arange_go(args[0], args[1], args[2], args[3], args[4], args[5])
    except:
        pass
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# TensorSpec entries (docs/tensor_spec_design.md): the whole op prologue —
# input checks, output alloc, kernel launch — in one boundary call over
# cached TensorSpecs, reusing the contiguous kernels above. Failed checks
# raise a real NotImplementedError into Python ("take the classic path");
# nothing is swallowed on spec paths.
# ---------------------------------------------------------------------------

# Dtypes the unary spec entries dispatch on for the "direct" (in-dtype) ops
# and the bool-output ops; the transcendental ops gate down to FLOAT_DTYPES.
# float64 works on the CPU device (the kernels comptime-refuse it on GPU).
comptime SPEC_UNARY_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
    DType.uint8,
]


def _unary_spec_go[op_code: Int](a_o: PyObjectPtr) raises -> PyObjectPtr:
    ref a = _spec_ptr(a_o)[]

    comptime is_direct = (
        op_code == UOP_RELU
        or op_code == UOP_ABS
        or op_code == UOP_NEG
        or op_code == UOP_SIGN
    )
    var supported = False
    comptime if is_direct:
        comptime for dt in SPEC_UNARY_DTYPES:
            if a.dtype == dt:
                supported = True
    else:
        comptime for dt in FLOAT_DTYPES:
            if a.dtype == dt:
                supported = True
    if not supported:
        raise Error("mojo spec unary: unsupported dtype ", a.dtype)

    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if a.numel > 0:
        if a.contig:
            comptime for dt in SPEC_UNARY_DTYPES:
                if a.dtype == dt:
                    _unary_elementwise[dt, op_code](
                        _make_ptr[dt](addr), _make_ptr[dt](a.ptr), a.numel, ctx
                    )
        else:
            # Mojo-side temporary (design doc §4.7): materialize the strided
            # input into a scratch buffer inside the call — Python never
            # mints a wrapper for it.
            var tmp = _scratch_contig(a, ctx)
            var tmp_addr = Int(tmp.unsafe_ptr())
            comptime for dt in SPEC_UNARY_DTYPES:
                if a.dtype == dt:
                    _unary_elementwise[dt, op_code](
                        _make_ptr[dt](addr),
                        _make_ptr[dt](tmp_addr),
                        a.numel,
                        ctx,
                    )
            _ = tmp^
    return _spec_result(
        buf^,
        addr,
        nbytes,
        a.rank,
        a.shape,
        a.dtype,
        a.itemsize,
        a.numel,
        a.ctx_ptr,
    )


def _unary_spec_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        return _unary_spec_go[op_code](args[0])
    except e:
        return _spec_unsupported(e)


def _unary_bool_spec_go[op_code: Int](a_o: PyObjectPtr) raises -> PyObjectPtr:
    ref a = _spec_ptr(a_o)[]
    # bool inputs are read through their uint8 storage (bit-compatible).
    var kdtype = a.dtype
    if a.dtype == DType.bool:
        kdtype = DType.uint8
    var supported = False
    comptime for dt in SPEC_UNARY_DTYPES:
        if kdtype == dt:
            supported = True
    if not supported:
        raise Error("mojo spec unary bool: unsupported dtype ", a.dtype)

    var ctx = a.ctx()
    var nbytes = a.numel  # bool output, itemsize 1
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if a.numel > 0:
        if a.contig:
            comptime for dt in SPEC_UNARY_DTYPES:
                if kdtype == dt:
                    _unary_bool[dt, op_code](
                        _make_ptr[DType.bool](addr),
                        _make_ptr[dt](a.ptr),
                        a.numel,
                        ctx,
                    )
        else:
            # Mojo-side temporary; see _unary_spec_go.
            var tmp = _scratch_contig(a, ctx)
            var tmp_addr = Int(tmp.unsafe_ptr())
            comptime for dt in SPEC_UNARY_DTYPES:
                if kdtype == dt:
                    _unary_bool[dt, op_code](
                        _make_ptr[DType.bool](addr),
                        _make_ptr[dt](tmp_addr),
                        a.numel,
                        ctx,
                    )
            _ = tmp^
    return _spec_result(
        buf^, addr, nbytes, a.rank, a.shape, DType.bool, 1, a.numel, a.ctx_ptr
    )


def _unary_bool_spec_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        return _unary_bool_spec_go[op_code](args[0])
    except e:
        return _spec_unsupported(e)


def _scalar_spec_go[
    op_code: Int
](a_o: PyObjectPtr, scalar_o: PyObjectPtr) raises -> PyObjectPtr:
    ref a = _spec_ptr(a_o)[]
    var supported = False
    comptime for dt in FLOAT_DTYPES:
        if a.dtype == dt:
            supported = True
    if not supported:
        raise Error("mojo spec scalar: unsupported dtype ", a.dtype)

    var scalar = Float32(_raw_f64(scalar_o))
    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if a.numel > 0:
        if a.contig:
            comptime for dt in FLOAT_DTYPES:
                if a.dtype == dt:
                    _scalar_elementwise[dt, op_code](
                        _make_ptr[dt](addr),
                        _make_ptr[dt](a.ptr),
                        scalar,
                        a.numel,
                        ctx,
                    )
        else:
            # Mojo-side temporary; see _unary_spec_go.
            var tmp = _scratch_contig(a, ctx)
            var tmp_addr = Int(tmp.unsafe_ptr())
            comptime for dt in FLOAT_DTYPES:
                if a.dtype == dt:
                    _scalar_elementwise[dt, op_code](
                        _make_ptr[dt](addr),
                        _make_ptr[dt](tmp_addr),
                        scalar,
                        a.numel,
                        ctx,
                    )
            _ = tmp^
    return _spec_result(
        buf^,
        addr,
        nbytes,
        a.rank,
        a.shape,
        a.dtype,
        a.itemsize,
        a.numel,
        a.ctx_ptr,
    )


def _scalar_spec_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        return _scalar_spec_go[op_code](args[0], args[1])
    except e:
        return _spec_unsupported(e)


def _scalar_inplace_go[
    op_code: Int
](a_o: PyObjectPtr, scalar_o: PyObjectPtr) raises -> PyObjectPtr:
    """`a op= scalar` for a contiguous float tensor, in place.

    The functional spec above allocates an output buffer, and the ATen in-place
    wrapper then copies it back over `a` -- an allocation and a
    device-to-device copy per call. That is invisible next to a real tensor but
    dominates a one-element tensor: nanoGPT's fused AdamW bumps 75 scalar step
    counters per step through `_foreach_add_.Scalar`, which ATen decomposes into
    75 `add_.Scalar`, and the copies alone cost ~376 us/step of GPU time.
    """
    ref a = _spec_ptr(a_o)[]
    var supported = False
    comptime for dt in FLOAT_DTYPES:
        if a.dtype == dt:
            supported = True
    if not supported:
        raise Error("mojo spec scalar inplace: unsupported dtype ", a.dtype)
    if not a.contig:
        raise Error("mojo spec scalar inplace: input is not contiguous")

    var scalar = Float32(_raw_f64(scalar_o))
    if a.numel > 0:
        var ctx = a.ctx()
        comptime for dt in FLOAT_DTYPES:
            if a.dtype == dt:
                _scalar_elementwise[dt, op_code](
                    _make_ptr[dt](a.ptr),
                    _make_ptr[dt](a.ptr),
                    scalar,
                    a.numel,
                    ctx,
                )
    return _raw_ret_none()


# Ints per tensor in the `ForeachAddScalar` metadata tuple: (address, numel).
comptime _FOREACH_ADD_FIELDS = 2


@__name(t"foreach_add_scalar_{dtype}_v1")
def _foreach_add_scalar_kernel[
    dtype: DType
](
    descs: InlineArray[ForeachDesc, FOREACH_DESC_CAP],
    desc_count: Int,
    scalar: Float32,
):
    """`t += scalar`, in place, for a whole list of tensors in one launch.

    One block per fixed-size chunk across the CONCATENATION of the list, so a
    list of 75 one-element tensors and a list of one 75-million-element tensor
    both land on a grid that describes the work rather than the list: the
    descriptor's `chunk_end` is a running prefix sum of chunk counts, and
    `_chunk_bounds` walks it to turn a flat block index back into (tensor,
    range).  The arithmetic is the same widen-add-narrow that
    `_scalar_elementwise[dtype, SOP_ADD]` does one tensor at a time, so the
    result is bit-identical to the `add_.Scalar` path this replaces.
    """
    var chunk = Int(block_idx.x)
    var desc, begin, end = _chunk_bounds(descs, desc_count, chunk)
    var values = _make_ptr[dtype](desc.tensor_addr)
    var index = begin + Int(thread_idx.x)
    while index < end:
        values[index] = (
            values.load[width=1](index).cast[DType.float32]() + scalar
        ).cast[dtype]()[0]
        index += FOREACH_THREADS


def _foreach_add_scalar_go(
    metadata_o: PyObjectPtr,
    scalar_o: PyObjectPtr,
    dtype_o: PyObjectPtr,
    ctx_o: PyObjectPtr,
) raises -> PyObjectPtr:
    """`aten::_foreach_add_.Scalar` for one contiguous float dtype.

    Without a PrivateUse1 entry ATen runs the CompositeExplicitAutograd
    fallback, which is a Python-level loop of `add_.Scalar`: nanoGPT's fused
    AdamW bumps 75 one-element step counters per step and pays 75 launches for
    75 additions.  The whole list is one launch here.
    """
    var dtype = _raw_dtype_int(dtype_o)
    var supported = False
    comptime for dt in FLOAT_DTYPES:
        if dtype == dt:
            supported = True
    if not supported:
        raise Error("mojo foreach add scalar: unsupported dtype ", dtype)
    var fields = _raw_tuple_len(metadata_o)
    if fields % _FOREACH_ADD_FIELDS != 0:
        raise Error("mojo foreach add scalar: malformed metadata tuple")
    var record_count = fields // _FOREACH_ADD_FIELDS
    # ATen's mutable-TensorList contract is all-or-nothing, so every record is
    # validated before the first launch rather than as it is consumed.
    for record in range(record_count):
        var base = record * _FOREACH_ADD_FIELDS
        if _raw_tuple_int(metadata_o, base + 1) < 0:
            raise Error("mojo foreach add scalar: negative numel")
        if (
            _raw_tuple_int(metadata_o, base + 1) > 0
            and _raw_tuple_int(metadata_o, base) == 0
        ):
            raise Error("mojo foreach add scalar: null pointer")
    var scalar = Float32(_raw_f64(scalar_o))
    var ctx = _raw_ctx(ctx_o)

    var record = 0
    while record < record_count:
        # `FOREACH_DESC_CAP` bounds ONE launch argument, not the list: a longer
        # list is several launches of the same compiled kernel.
        var descs = InlineArray[ForeachDesc, FOREACH_DESC_CAP](
            fill=empty_foreach_desc()
        )
        var desc_count = 0
        var total_chunks = 0
        while record < record_count and desc_count < FOREACH_DESC_CAP:
            var base = record * _FOREACH_ADD_FIELDS
            var numel = _raw_tuple_int(metadata_o, base + 1)
            if numel > 0:
                total_chunks += ceildiv(numel, FOREACH_CHUNK_ELEMENTS)
            descs[desc_count] = ForeachDesc(
                _raw_tuple_int(metadata_o, base), 0, numel, total_chunks
            )
            record += 1
            desc_count += 1
        if total_chunks > 0:
            comptime for dt in FLOAT_DTYPES:
                if dtype == dt:
                    _enqueue_cached[_foreach_add_scalar_kernel[dt]](
                        ctx,
                        String(t"foreach_add_scalar_{dt}_v1"),
                        total_chunks,
                        1,
                        1,
                        FOREACH_THREADS,
                        descs,
                        desc_count,
                        scalar,
                    )
    return _raw_ret_none()


def _foreach_add_scalar_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        if nargs != 4:
            raise Error("ForeachAddScalar expects exactly four arguments")
        return _foreach_add_scalar_go(args[0], args[1], args[2], args[3])
    except e:
        return _spec_unsupported(e)


def _scalar_inplace_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        return _scalar_inplace_go[op_code](args[0], args[1])
    except e:
        return _spec_unsupported(e)


def _int_scalar_spec_go[
    op_code: Int
](a_o: PyObjectPtr, scalar_o: PyObjectPtr) raises -> PyObjectPtr:
    ref a = _spec_ptr(a_o)[]
    var supported = False
    comptime for dt in [DType.int32, DType.int64]:
        if a.dtype == dt:
            supported = True
    if not supported:
        raise Error("mojo spec int scalar: unsupported dtype ", a.dtype)

    var scalar = _raw_int(scalar_o)
    var ctx = a.ctx()
    var nbytes = a.numel * a.itemsize
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if a.numel > 0:
        if a.contig:
            comptime for dt in [DType.int32, DType.int64]:
                if a.dtype == dt:
                    _int_scalar_elementwise[dt, op_code](
                        _make_ptr[dt](addr),
                        _make_ptr[dt](a.ptr),
                        scalar,
                        a.numel,
                        ctx,
                    )
        else:
            # Mojo-side temporary; see _unary_spec_go.
            var tmp = _scratch_contig(a, ctx)
            var tmp_addr = Int(tmp.unsafe_ptr())
            comptime for dt in [DType.int32, DType.int64]:
                if a.dtype == dt:
                    _int_scalar_elementwise[dt, op_code](
                        _make_ptr[dt](addr),
                        _make_ptr[dt](tmp_addr),
                        scalar,
                        a.numel,
                        ctx,
                    )
            _ = tmp^
    return _spec_result(
        buf^,
        addr,
        nbytes,
        a.rank,
        a.shape,
        a.dtype,
        a.itemsize,
        a.numel,
        a.ctx_ptr,
    )


def _int_scalar_spec_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        return _int_scalar_spec_go[op_code](args[0], args[1])
    except e:
        return _spec_unsupported(e)


comptime SPEC_FILL_DTYPES = [
    DType.float32,
    DType.float16,
    DType.bfloat16,
    DType.float64,
    DType.int64,
    DType.int32,
    DType.int16,
    DType.int8,
    DType.uint8,
    DType.bool,
]


def _fill_spec_go(
    shape_t: PyObjectPtr,
    rank_o: PyObjectPtr,
    numel_o: PyObjectPtr,
    value_o: PyObjectPtr,
    dtype_o: PyObjectPtr,
    ctx_o: PyObjectPtr,
) raises -> PyObjectPtr:
    """Filled-tensor construction: alloc + fill in one boundary call (the
    classic path is a Python-side _alloc plus a Fill call)."""
    var rank = _raw_int(rank_o)
    var numel = _raw_int(numel_o)
    var value = _raw_f64(value_o)
    var dtype = _raw_dtype_int(dtype_o)
    var shape = IndexList[MAX_RANK](1)
    for i in range(MAX_RANK):
        shape[i] = _raw_tuple_int(shape_t, i)

    var supported = False
    var itemsize = 0
    comptime for dt in SPEC_FILL_DTYPES:
        if dtype == dt:
            supported = True
            itemsize = size_of[dt]()
    if not supported:
        raise Error("mojo spec fill: unsupported dtype ", dtype)
    # bool fill must store exactly 0/1.
    if dtype == DType.bool:
        value = Float64(1) if value != 0 else Float64(0)

    var ctx = _raw_ctx(ctx_o)
    var nbytes = numel * itemsize
    var buf = ctx.enqueue_create_buffer[DType.uint8](max(nbytes, 1))
    var addr = Int(buf.unsafe_ptr())
    if numel > 0:
        comptime for dt in SPEC_FILL_DTYPES:
            if dtype == dt:
                _fill[dt](_make_ptr[dt](addr), value, numel, ctx)
    return _spec_result(
        buf^, addr, nbytes, rank, shape, dtype, itemsize, numel, _raw_int(ctx_o)
    )


def _fill_spec_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        return _fill_spec_go(
            args[0], args[1], args[2], args[3], args[4], args[5]
        )
    except e:
        return _spec_unsupported(e)


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_elementwise_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("elementwise_ops")
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_RELU],
            "ReluSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); relu",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_EXP],
            "ExpSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); exp",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_TANH],
            "TanhSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); tanh",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_ABS],
            "AbsSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); abs",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_NEG],
            "NegSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); neg",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SIGN],
            "SignSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); sign",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_CEIL],
            "CeilSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); ceil",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_FLOOR],
            "FloorSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); floor",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_ACOS],
            "AcosSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); acos",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_ASINH],
            "AsinhSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); asinh",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_ATANH],
            "AtanhSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); atanh",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_COS],
            "CosSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); cos",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_COSH],
            "CoshSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); cosh",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_ERF],
            "ErfSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); erf",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_LOG],
            "LogSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); log",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_LOG1P],
            "Log1pSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); log1p",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_RECIPROCAL],
            "ReciprocalSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); reciprocal",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_RSQRT],
            "RsqrtSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); rsqrt",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SIGMOID],
            "SigmoidSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); sigmoid",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SILU],
            "SiluSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); silu",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SIN],
            "SinSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); sin",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SINH],
            "SinhSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); sinh",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_SQRT],
            "SqrtSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); sqrt",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_TAN],
            "TanSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); tan",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_GELU_NONE],
            "GeluNoneSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); gelunone",
        )
        b.def_py_c_function(
            _unary_spec_dispatcher[UOP_GELU_TANH],
            "GeluTanhSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); gelutanh",
        )
        b.def_py_c_function(
            _unary_bool_spec_dispatcher[BUOP_ISNAN],
            "IsNanSpec",
            docstring="(a_spec) -> (holder, spec, shape, ptr); isnan -> bool",
        )
        b.def_py_c_function(
            _unary_bool_spec_dispatcher[BUOP_LOGICAL_NOT],
            "LogicalNotSpec",
            docstring=(
                "(a_spec) -> (holder, spec, shape, ptr); logicalnot -> bool"
            ),
        )
        b.def_py_c_function(
            _scalar_spec_dispatcher[SOP_ADD],
            "AddScalarSpec",
            docstring="(a_spec, scalar) -> (holder, spec, shape, ptr); float",
        )
        b.def_py_c_function(
            _scalar_spec_dispatcher[SOP_MUL],
            "MulScalarSpec",
            docstring="(a_spec, scalar) -> (holder, spec, shape, ptr); float",
        )
        b.def_py_c_function(
            _scalar_spec_dispatcher[SOP_POW],
            "PowScalarSpec",
            docstring="(a_spec, scalar) -> (holder, spec, shape, ptr); float",
        )
        b.def_py_c_function(
            _scalar_inplace_dispatcher[SOP_ADD],
            "AddScalarInplace",
            docstring="(a_spec, scalar) -> None; a += scalar, contiguous float",
        )
        b.def_py_c_function(
            _scalar_inplace_dispatcher[SOP_MUL],
            "MulScalarInplace",
            docstring="(a_spec, scalar) -> None; a *= scalar, contiguous float",
        )
        b.def_py_c_function(
            _foreach_add_scalar_dispatcher,
            "ForeachAddScalar",
            docstring=(
                "((addr, numel) * n, scalar, dtype, ctx) -> None; "
                "aten::_foreach_add_.Scalar over one contiguous float dtype"
            ),
        )
        b.def_py_c_function(
            _int_scalar_spec_dispatcher[IOP_ADD],
            "AddScalarIntSpec",
            docstring="(a_spec, scalar) -> (holder, spec, shape, ptr); int",
        )
        b.def_py_c_function(
            _int_scalar_spec_dispatcher[IOP_MUL],
            "MulScalarIntSpec",
            docstring="(a_spec, scalar) -> (holder, spec, shape, ptr); int",
        )
        b.def_py_c_function(
            _fill_spec_dispatcher,
            "FillSpec",
            docstring=(
                "(shape8, rank, numel, value, dtype, ctx) -> result group"
            ),
        )
        b.def_py_c_function(
            _bin_dispatcher[OP_ADD],
            "Add",
            docstring="out = lhs + rhs (contiguous, dtype dispatch)",
        )
        b.def_py_c_function(
            _arange_dispatcher,
            "Arange",
            docstring="out[i] = start + i * step (contiguous, int/float)",
        )
        return b.finalize()
    except e:
        abort(t"failed to create elementwise_ops python module: {e}")
