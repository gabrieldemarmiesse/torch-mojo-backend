# ===----------------------------------------------------------------------=== #
# Fast eager-mode logic kernels for mojo_device: broadcast-strided binary
# arithmetic, comparisons (bool output), bitwise ops, bitwise not, and isin.
#
# These cover the small bookkeeping tensors that drive generation loops
# (stopping criteria, attention-mask prep, position ids), where the operands
# frequently broadcast. Each binary kernel takes the contiguous output's
# dims padded to rank 4 plus per-operand strides in elements (0 for
# broadcast dims), computed on the Python side.
#
# Raw-pointer calling convention (mirrors elementwise_ops.mojo /
# tensor_holder.mojo): every Python-visible function receives plain ints —
# each tensor operand as one address int (storage offset already applied),
# dtype as one `max.dtype.DType.value` int per operand role, sizes/counts as
# ints, the dims+strides bundle as a tuple of ints (read with
# `_raw_tuple_int`) — plus the device's DeviceContext pointer as the last
# int. No `max.driver.Buffer` object crosses this boundary anymore.
# Dispatchers are registered as METH_FASTCALL functions
# (`def_py_c_function`) reading raw CPython arguments directly (see
# op_utils), and work is enqueued on MAX's own device queue (fire and
# forget, no sync).
# ===----------------------------------------------------------------------=== #

from std.os import abort
from std.gpu import block_dim, block_idx, grid_dim, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv, pow
from std.python import PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder
from std.sys.info import has_accelerator, has_apple_gpu_accelerator, size_of
from std.utils.coord import Coord

from max.algorithm import elementwise

from std.utils import IndexList

from op_utils import (
    GS_THREADS,
    MAX_RANK,
    _bw_flat_blocks,
    _enqueue_cached,
    _gs_blocks,
    _make_ptr,
    _parallel_for,
    _raw_ctx,
    _raw_dtype_int,
    _raw_f64,
    _raw_int,
    _raw_ret_none,
    _raw_tuple_int,
    _spec_dispatcher3,
    _spec_ptr,
    _spec_unsupported,
)

from variant_gates import (
    _dtype_arg_abi_on,
    _dtype_arg_on,
    _op_on,
    _register_call,
)


# ---------------------------------------------------------------------------
# Broadcast-strided binary elementwise: out is contiguous with dims
# d0..d3; each operand is indexed with its own strides (0 on broadcast
# dims). Bitwise ops only instantiate for integer dtypes (the Python side
# routes bool through uint8 views), div only for floats.
# ---------------------------------------------------------------------------

comptime BOP_ADD = 0
comptime BOP_SUB = 1
comptime BOP_MUL = 2
comptime BOP_DIV = 3
comptime BOP_MAX = 4
comptime BOP_MIN = 5
comptime BOP_AND = 6
comptime BOP_OR = 7
comptime BOP_XOR = 8
comptime BOP_REMAINDER = 9
comptime BOP_FLOORDIV = 10
comptime BOP_POW = 11

comptime _ADD_F32_BF16_BLOCK = 256
comptime _ADD_F32_BF16_VEC = 4


@__name("torch_mojo_add_f32_bf16_vec4")
def _add_f32_bf16_contig_kernel(
    output_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input_f32: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    input_bf16: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    elements_arg: Int64,
    vec_count_arg: Int64,
):
    """One-launch contiguous mixed add with in-register BF16 widening."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var elements = Int(elements_arg)
    var vec_count = Int(vec_count_arg)
    var gid = Int(block_idx.x) * _ADD_F32_BF16_BLOCK + Int(thread_idx.x)
    if gid < vec_count:
        var base = gid * _ADD_F32_BF16_VEC
        var lhs = input_f32.load[width=_ADD_F32_BF16_VEC, alignment=16](base)
        var rhs = input_bf16.load[width=_ADD_F32_BF16_VEC, alignment=8](
            base
        ).cast[DType.float32]()
        output_f32.store[width=_ADD_F32_BF16_VEC, alignment=16](base, lhs + rhs)

    # The vector body leaves at most three elements.  With an unaligned base,
    # vec_count is zero and this same launch covers the full input scalarly.
    var index = vec_count * _ADD_F32_BF16_VEC + gid
    var stride = Int(grid_dim.x) * _ADD_F32_BF16_BLOCK
    while index < elements:
        output_f32[index] = (
            input_f32[index] + input_bf16[index].cast[DType.float32]()
        )
        index += stride


@always_inline
def _add_f32_bf16_contig(
    output_f32: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    input_f32: UnsafePointer[Scalar[DType.float32], ImmutAnyOrigin],
    input_bf16: UnsafePointer[Scalar[DType.bfloat16], ImmutAnyOrigin],
    elements: Int,
    ctx: DeviceContext,
) raises:
    if elements <= 0:
        return

    var aligned = (Int(output_f32) | Int(input_f32) | Int(input_bf16)) % 16 == 0
    var vec_count = elements // _ADD_F32_BF16_VEC if aligned else 0
    var work_items = vec_count if vec_count > 0 else elements
    var grid = ceildiv(work_items, _ADD_F32_BF16_BLOCK)
    comptime if has_accelerator():
        _enqueue_cached[_add_f32_bf16_contig_kernel](
            ctx,
            "add_f32_bf16_vec4",
            grid,
            1,
            1,
            _ADD_F32_BF16_BLOCK,
            output_f32,
            input_f32,
            input_bf16,
            Int64(elements),
            Int64(vec_count),
        )
    else:
        raise Error("mixed FP32/BF16 add requires an accelerator build")


@always_inline
def _bin_bcast_body[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    a: Scalar[dtype],
    b: Scalar[dtype],
    i: Int,
):
    """One output element of the broadcast binary op — shared verbatim by
    the CPU closure and the cached GPU kernel."""
    comptime if op_code == BOP_ADD:
        out_ptr[i] = a + b
    comptime if op_code == BOP_SUB:
        out_ptr[i] = a - b
    comptime if op_code == BOP_MUL:
        out_ptr[i] = a * b
    comptime if op_code == BOP_DIV:
        comptime if dtype.is_floating_point():
            out_ptr[i] = a / b
    comptime if op_code == BOP_MAX:
        out_ptr[i] = max(a, b)
    comptime if op_code == BOP_MIN:
        out_ptr[i] = min(a, b)
    comptime if op_code == BOP_AND:
        comptime if not dtype.is_floating_point():
            out_ptr[i] = a & b
    comptime if op_code == BOP_OR:
        comptime if not dtype.is_floating_point():
            out_ptr[i] = a | b
    comptime if op_code == BOP_XOR:
        comptime if not dtype.is_floating_point():
            out_ptr[i] = a ^ b
    comptime if op_code == BOP_REMAINDER:
        # Mojo's `%` follows the divisor's sign (Python/torch
        # semantics) for both signed integers and floats.
        out_ptr[i] = a % b
    comptime if op_code == BOP_FLOORDIV:
        # `//` = floor(a / b), matching torch.floor_divide for
        # both float and integer dtypes.
        out_ptr[i] = a // b
    comptime if op_code == BOP_POW:
        # Float only (gated at the launcher); accumulate halves in
        # float32 to match torch's numerics.
        comptime if dtype == DType.float16 or dtype == DType.bfloat16:
            out_ptr[i] = pow(
                a.cast[DType.float32](), b.cast[DType.float32]()
            ).cast[dtype]()
        elif dtype.is_floating_point():
            out_ptr[i] = pow(a, b)


def _bin_bcast_kernel[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    l_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    r_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d1_arg: Int64,
    d2_arg: Int64,
    d3_arg: Int64,
    ls0_arg: Int64,
    ls1_arg: Int64,
    ls2_arg: Int64,
    ls3_arg: Int64,
    rs0_arg: Int64,
    rs1_arg: Int64,
    rs2_arg: Int64,
    rs3_arg: Int64,
    total_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var d1 = Int(d1_arg)
    var d2 = Int(d2_arg)
    var d3 = Int(d3_arg)
    var ls0 = Int(ls0_arg)
    var ls1 = Int(ls1_arg)
    var ls2 = Int(ls2_arg)
    var ls3 = Int(ls3_arg)
    var rs0 = Int(rs0_arg)
    var rs1 = Int(rs1_arg)
    var rs2 = Int(rs2_arg)
    var rs3 = Int(rs3_arg)
    var total = Int(total_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        var a = l_ptr[i0 * ls0 + i1 * ls1 + i2 * ls2 + i3 * ls3]
        var b = r_ptr[i0 * rs0 + i1 * rs1 + i2 * rs2 + i3 * rs3]
        _bin_bcast_body[dtype, op_code](out_ptr, a, b, i)
        i += gstride


@always_inline
def _bin_vec_op[
    dtype: DType, op_code: Int, width: Int
](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """SIMD counterpart of `_bin_bcast_body` (non-comparison ops only).

    Runtime-unreachable (dtype, op) combos are pre-gated by the launcher;
    they still instantiate, so every branch must compile — hence the
    trailing pass-through return."""
    comptime if op_code == BOP_ADD:
        return a + b
    comptime if op_code == BOP_SUB:
        return a - b
    comptime if op_code == BOP_MUL:
        return a * b
    comptime if op_code == BOP_DIV:
        comptime if dtype.is_floating_point():
            return a / b
    comptime if op_code == BOP_MAX:
        return max(a, b)
    comptime if op_code == BOP_MIN:
        return min(a, b)
    comptime if op_code == BOP_AND:
        comptime if not dtype.is_floating_point():
            return a & b
    comptime if op_code == BOP_OR:
        comptime if not dtype.is_floating_point():
            return a | b
    comptime if op_code == BOP_XOR:
        comptime if not dtype.is_floating_point():
            return a ^ b
    comptime if op_code == BOP_REMAINDER:
        return a % b
    comptime if op_code == BOP_FLOORDIV:
        return a // b
    comptime if op_code == BOP_POW:
        comptime if dtype == DType.float16 or dtype == DType.bfloat16:
            return pow(a.cast[DType.float32](), b.cast[DType.float32]()).cast[
                dtype
            ]()
        elif dtype.is_floating_point():
            return pow(a, b)
    return a


def _bin_flat_vec_kernel[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    l_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    r_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    total_arg: Int64,
):
    """16-byte-vectorized binary op for the no-broadcast, all-contiguous
    case (both operands and the output flat over the same extent). The
    launcher guarantees 16B base alignment."""
    comptime VW = 16 // size_of[dtype]()
    comptime vec_align = VW * size_of[dtype]()
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var total = Int(total_arg)
    var tid = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    var nvec = total // VW
    var c = tid
    while c < nvec:
        var i = c * VW
        var a = l_ptr.load[width=VW, alignment=vec_align](i)
        var b = r_ptr.load[width=VW, alignment=vec_align](i)
        out_ptr.store[width=VW, alignment=vec_align](
            i, _bin_vec_op[dtype, op_code, VW](a, b)
        )
        c += gstride
    var tail = total - nvec * VW
    if tid < tail:
        var i = nvec * VW + tid
        out_ptr[i] = _bin_vec_op[dtype, op_code, 1](l_ptr[i], r_ptr[i])[0]


def _bin_rowvec_kernel[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    l_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    r_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d1_arg: Int64,
    d2_arg: Int64,
    d3_arg: Int64,
    ls0_arg: Int64,
    ls1_arg: Int64,
    ls2_arg: Int64,
    rs0_arg: Int64,
    rs1_arg: Int64,
    rs2_arg: Int64,
    rows_arg: Int64,
):
    """Vectorized along a stride-1 innermost dim; outer dims may be strided
    or broadcast (stride 0). One block per row (grid-stride over rows); the
    per-row index math runs once per row instead of once per element. The
    launcher guarantees d3 % VW == 0 and 16B alignment of every row base."""
    comptime VW = 16 // size_of[dtype]()
    comptime vec_align = VW * size_of[dtype]()
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var d1 = Int(d1_arg)
    var d2 = Int(d2_arg)
    var d3 = Int(d3_arg)
    var ls0 = Int(ls0_arg)
    var ls1 = Int(ls1_arg)
    var ls2 = Int(ls2_arg)
    var rs0 = Int(rs0_arg)
    var rs1 = Int(rs1_arg)
    var rs2 = Int(rs2_arg)
    var rows = Int(rows_arg)
    var row = Int(block_idx.x)
    while row < rows:
        var i2 = row % d2
        var rest = row // d2
        var i1 = rest % d1
        var i0 = rest // d1
        var lbase = i0 * ls0 + i1 * ls1 + i2 * ls2
        var rbase = i0 * rs0 + i1 * rs1 + i2 * rs2
        var obase = row * d3
        var j = Int(thread_idx.x) * VW
        var step = Int(block_dim.x) * VW
        while j < d3:
            var a = l_ptr.load[width=VW, alignment=vec_align](lbase + j)
            var b = r_ptr.load[width=VW, alignment=vec_align](rbase + j)
            out_ptr.store[width=VW, alignment=vec_align](
                obase + j, _bin_vec_op[dtype, op_code, VW](a, b)
            )
            j += step
        row += Int(grid_dim.x)


@always_inline
def _bin_bcast[
    dtype: DType, op_code: Int
](
    out_addr: Int,
    l_addr: Int,
    r_addr: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    ls0: Int,
    ls1: Int,
    ls2: Int,
    ls3: Int,
    rs0: Int,
    rs1: Int,
    rs2: Int,
    rs3: Int,
    total: Int,
    ctx: DeviceContext,
) raises:
    comptime if (
        op_code == BOP_DIV or op_code == BOP_POW
    ) and not dtype.is_floating_point():
        raise Error("integer/bool div/pow is not supported in the fast path")
    else:
        comptime if (
            op_code == BOP_AND or op_code == BOP_OR or op_code == BOP_XOR
        ) and dtype.is_floating_point():
            raise Error("bitwise ops require an integer dtype")
        else:
            var out_ptr = _make_ptr[dtype](out_addr)
            var l_ptr = _make_ptr[dtype](l_addr)
            var r_ptr = _make_ptr[dtype](r_addr)

            if ctx.api() == "cpu":

                @always_inline
                @parameter
                @__copy_capture(out_ptr, l_ptr, r_ptr)
                def func[width: Int, alignment: Int = 1](idx: Coord):
                    var i = Int(idx[0].value())
                    var i3 = i % d3
                    var rest = i // d3
                    var i2 = rest % d2
                    rest = rest // d2
                    var i1 = rest % d1
                    var i0 = rest // d1
                    var a = l_ptr[i0 * ls0 + i1 * ls1 + i2 * ls2 + i3 * ls3]
                    var b = r_ptr[i0 * rs0 + i1 * rs1 + i2 * rs2 + i3 * rs3]
                    _bin_bcast_body[dtype, op_code](
                        out_ptr.as_unsafe_any_origin(), a, b, i
                    )

                elementwise[func, simd_width=1](Coord(total), ctx)
            else:
                comptime if (
                    dtype == DType.float64 and has_apple_gpu_accelerator()
                ):
                    raise Error("float64 is not supported on Apple GPU")
                else:
                    comptime if has_accelerator():
                        # Tiered dispatch: the generic scalar kernel pays
                        # three integer divisions and strided scalar loads
                        # per element, which is division-bound at large
                        # sizes. Prefer 16-byte vector kernels whenever the
                        # layout allows them.
                        comptime itemsize = size_of[dtype]()
                        comptime VW = 16 // itemsize
                        var d0 = total // max(1, d1 * d2 * d3)
                        var cont3 = d3
                        var cont2 = d2 * d3
                        var cont1 = d1 * d2 * d3
                        var aligned16 = (
                            out_addr % 16 == 0
                            and l_addr % 16 == 0
                            and r_addr % 16 == 0
                        )
                        var l_flat = (
                            (ls3 == 1 or d3 == 1)
                            and (ls2 == cont3 or d2 == 1)
                            and (ls1 == cont2 or d1 == 1)
                            and (ls0 == cont1 or d0 == 1)
                        )
                        var r_flat = (
                            (rs3 == 1 or d3 == 1)
                            and (rs2 == cont3 or d2 == 1)
                            and (rs1 == cont2 or d1 == 1)
                            and (rs0 == cont1 or d0 == 1)
                        )
                        var rows_aligned = (
                            ls3 == 1
                            and rs3 == 1
                            and d3 % VW == 0
                            and d3 >= VW
                            and (ls0 * itemsize) % 16 == 0
                            and (ls1 * itemsize) % 16 == 0
                            and (ls2 * itemsize) % 16 == 0
                            and (rs0 * itemsize) % 16 == 0
                            and (rs1 * itemsize) % 16 == 0
                            and (rs2 * itemsize) % 16 == 0
                        )
                        if aligned16 and l_flat and r_flat:
                            _enqueue_cached[
                                _bin_flat_vec_kernel[dtype, op_code]
                            ](
                                ctx,
                                String(t"lg_bcast_fv_{op_code}_{dtype}"),
                                _bw_flat_blocks(
                                    max(1, total // VW), 3 * total * itemsize
                                ),
                                1,
                                1,
                                GS_THREADS,
                                out_ptr.as_unsafe_any_origin(),
                                l_ptr.as_unsafe_any_origin().as_immutable(),
                                r_ptr.as_unsafe_any_origin().as_immutable(),
                                Int64(total),
                            )
                        elif aligned16 and rows_aligned:
                            var rows = total // d3
                            _enqueue_cached[_bin_rowvec_kernel[dtype, op_code]](
                                ctx,
                                String(t"lg_bcast_rv_{op_code}_{dtype}"),
                                max(1, min(rows, 65535)),
                                1,
                                1,
                                GS_THREADS,
                                out_ptr.as_unsafe_any_origin(),
                                l_ptr.as_unsafe_any_origin().as_immutable(),
                                r_ptr.as_unsafe_any_origin().as_immutable(),
                                Int64(d1),
                                Int64(d2),
                                Int64(d3),
                                Int64(ls0),
                                Int64(ls1),
                                Int64(ls2),
                                Int64(rs0),
                                Int64(rs1),
                                Int64(rs2),
                                Int64(rows),
                            )
                        else:
                            _enqueue_cached[_bin_bcast_kernel[dtype, op_code]](
                                ctx,
                                String(t"lg_bcast_{op_code}_{dtype}"),
                                _gs_blocks(total),
                                1,
                                1,
                                GS_THREADS,
                                out_ptr.as_unsafe_any_origin(),
                                l_ptr.as_unsafe_any_origin().as_immutable(),
                                r_ptr.as_unsafe_any_origin().as_immutable(),
                                Int64(d1),
                                Int64(d2),
                                Int64(d3),
                                Int64(ls0),
                                Int64(ls1),
                                Int64(ls2),
                                Int64(ls3),
                                Int64(rs0),
                                Int64(rs1),
                                Int64(rs2),
                                Int64(rs3),
                                Int64(total),
                            )
                    else:
                        raise Error(
                            "no GPU accelerator available at compile time"
                        )


# ---------------------------------------------------------------------------
# Broadcast-strided comparisons: same indexing, bool output.
# ---------------------------------------------------------------------------

comptime COP_EQ = 0
comptime COP_NE = 1
comptime COP_LT = 2
comptime COP_LE = 3
comptime COP_GT = 4
comptime COP_GE = 5
comptime COP_LAND = 6
comptime COP_LXOR = 7


@always_inline
def _cmp_bcast_body[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    a: Scalar[dtype],
    b: Scalar[dtype],
    i: Int,
):
    """One output element of the broadcast comparison — shared verbatim by
    the CPU closure and the cached GPU kernel."""
    comptime if op_code == COP_EQ:
        out_ptr[i] = Scalar[DType.bool](a == b)
    comptime if op_code == COP_NE:
        out_ptr[i] = Scalar[DType.bool](a != b)
    comptime if op_code == COP_LT:
        out_ptr[i] = Scalar[DType.bool](a < b)
    comptime if op_code == COP_LE:
        out_ptr[i] = Scalar[DType.bool](a <= b)
    comptime if op_code == COP_GT:
        out_ptr[i] = Scalar[DType.bool](a > b)
    comptime if op_code == COP_GE:
        out_ptr[i] = Scalar[DType.bool](a >= b)
    comptime if op_code == COP_LAND or op_code == COP_LXOR:
        # Logical ops test each operand for nonzero-ness, then combine.
        # Output is bool regardless of the (arbitrary) input dtype.
        var la = a != Scalar[dtype](0)
        var lb = b != Scalar[dtype](0)
        comptime if op_code == COP_LAND:
            out_ptr[i] = la & lb
        comptime if op_code == COP_LXOR:
            out_ptr[i] = la ^ lb


def _cmp_bcast_kernel[
    dtype: DType, op_code: Int
](
    out_ptr: UnsafePointer[Scalar[DType.bool], MutAnyOrigin],
    l_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    r_ptr: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    d1_arg: Int64,
    d2_arg: Int64,
    d3_arg: Int64,
    ls0_arg: Int64,
    ls1_arg: Int64,
    ls2_arg: Int64,
    ls3_arg: Int64,
    rs0_arg: Int64,
    rs1_arg: Int64,
    rs2_arg: Int64,
    rs3_arg: Int64,
    total_arg: Int64,
):
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var d1 = Int(d1_arg)
    var d2 = Int(d2_arg)
    var d3 = Int(d3_arg)
    var ls0 = Int(ls0_arg)
    var ls1 = Int(ls1_arg)
    var ls2 = Int(ls2_arg)
    var ls3 = Int(ls3_arg)
    var rs0 = Int(rs0_arg)
    var rs1 = Int(rs1_arg)
    var rs2 = Int(rs2_arg)
    var rs3 = Int(rs3_arg)
    var total = Int(total_arg)
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    var gstride = Int(grid_dim.x) * Int(block_dim.x)
    while i < total:
        var i3 = i % d3
        var rest = i // d3
        var i2 = rest % d2
        rest = rest // d2
        var i1 = rest % d1
        var i0 = rest // d1
        var a = l_ptr[i0 * ls0 + i1 * ls1 + i2 * ls2 + i3 * ls3]
        var b = r_ptr[i0 * rs0 + i1 * rs1 + i2 * rs2 + i3 * rs3]
        _cmp_bcast_body[dtype, op_code](out_ptr, a, b, i)
        i += gstride


@always_inline
def _cmp_bcast[
    dtype: DType, op_code: Int
](
    out_addr: Int,
    l_addr: Int,
    r_addr: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    ls0: Int,
    ls1: Int,
    ls2: Int,
    ls3: Int,
    rs0: Int,
    rs1: Int,
    rs2: Int,
    rs3: Int,
    total: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[DType.bool](out_addr)
    var l_ptr = _make_ptr[dtype](l_addr)
    var r_ptr = _make_ptr[dtype](r_addr)

    if ctx.api() == "cpu":

        @always_inline
        @parameter
        @__copy_capture(out_ptr, l_ptr, r_ptr)
        def func[width: Int, alignment: Int = 1](idx: Coord):
            var i = Int(idx[0].value())
            var i3 = i % d3
            var rest = i // d3
            var i2 = rest % d2
            rest = rest // d2
            var i1 = rest % d1
            var i0 = rest // d1
            var a = l_ptr[i0 * ls0 + i1 * ls1 + i2 * ls2 + i3 * ls3]
            var b = r_ptr[i0 * rs0 + i1 * rs1 + i2 * rs2 + i3 * rs3]
            _cmp_bcast_body[dtype, op_code](
                out_ptr.as_unsafe_any_origin(), a, b, i
            )

        elementwise[func, simd_width=1](Coord(total), ctx)
    else:
        comptime if dtype == DType.float64 and has_apple_gpu_accelerator():
            raise Error("float64 is not supported on Apple GPU")
        else:
            comptime if has_accelerator():
                _enqueue_cached[_cmp_bcast_kernel[dtype, op_code]](
                    ctx,
                    String(t"lg_cmp_{op_code}_{dtype}"),
                    _gs_blocks(total),
                    1,
                    1,
                    GS_THREADS,
                    out_ptr.as_unsafe_any_origin(),
                    l_ptr.as_unsafe_any_origin().as_immutable(),
                    r_ptr.as_unsafe_any_origin().as_immutable(),
                    Int64(d1),
                    Int64(d2),
                    Int64(d3),
                    Int64(ls0),
                    Int64(ls1),
                    Int64(ls2),
                    Int64(ls3),
                    Int64(rs0),
                    Int64(rs1),
                    Int64(rs2),
                    Int64(rs3),
                    Int64(total),
                )
            else:
                raise Error("no GPU accelerator available at compile time")


# ---------------------------------------------------------------------------
# Runtime dtype dispatch shared by both broadcast kernel families. The
# strides/dims arrive as one raw CPython 12-tuple (d0..d3, ls0..ls3,
# rs0..rs3), read element-by-element with `_raw_tuple_int` (borrowed
# references, no refcount traffic).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Bitwise not over a contiguous span. `~` on bool is logical not, on
# integers the usual complement — matching torch.
# ---------------------------------------------------------------------------


@always_inline
def _bitwise_not[
    dtype: DType
](out_addr: Int, in_addr: Int, size: Int, ctx: DeviceContext) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        out_ptr[i] = ~in_ptr[i]

    _parallel_for[func](size, ctx)


def _bitwise_not_go(
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    numel: PyObjectPtr,
    dtype: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var dtype_val = _raw_dtype_int(dtype)
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var size = _raw_int(numel)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in [
        DType.bool,
        DType.uint8,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
    ]:
        comptime if _dtype_arg_on[0, dt]():
            if dtype_val == dt:
                _bitwise_not[dt](out_addr, in_addr, size, ctx)
                handled = True
    if not handled:
        raise Error(
            "unsupported dtype for fast bitwise not: " + String(dtype_val)
        )


def _bitwise_not_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _bitwise_not_go(args[0], args[1], args[2], args[3], args[4])
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# isin: out[i] = (x[i] in test[0..n_test)) ^ invert. Integer dtypes only
# (float equality-by-value is gated out on the Python side). The inner loop
# over test elements is sequential — n_test is tiny (eos token lists).
# ---------------------------------------------------------------------------


@always_inline
def _isin[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    test_addr: Int,
    size: Int,
    n_test: Int,
    invert: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[DType.bool](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var test_ptr = _make_ptr[dtype](test_addr)

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, test_ptr)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var found = False
        for j in range(n_test):
            if in_ptr[i] == test_ptr[j]:
                found = True
                break
        if invert != 0:
            found = not found
        out_ptr[i] = found

    _parallel_for[func](size, ctx)


def _isin_go(
    out_ptr: PyObjectPtr,
    el_ptr: PyObjectPtr,
    te_ptr: PyObjectPtr,
    el_numel: PyObjectPtr,
    te_numel: PyObjectPtr,
    invert: PyObjectPtr,
    dtype: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var dtype_val = _raw_dtype_int(dtype)
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(el_ptr)
    var test_addr = _raw_int(te_ptr)
    var size = _raw_int(el_numel)
    var n_test_val = _raw_int(te_numel)
    var invert_val = _raw_int(invert)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in [DType.int64, DType.int32]:
        comptime if _dtype_arg_on[0, dt]():
            if dtype_val == dt:
                _isin[dt](
                    out_addr,
                    in_addr,
                    test_addr,
                    size,
                    n_test_val,
                    invert_val,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast isin: " + String(dtype_val))


def _isin_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _isin_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# clamp(self, min?, max?): out = min(max(x, lo), hi), each bound optional.
# Bounds arrive as Float64 and are cast to the tensor dtype (exact for the
# small integer bounds torch passes). Contiguous unary; the Python side
# materializes strided inputs first.
# ---------------------------------------------------------------------------


@always_inline
def _clamp_scalar[
    dtype: DType
](
    out_addr: Int,
    in_addr: Int,
    lo: Float64,
    hi: Float64,
    has_min: Int,
    has_max: Int,
    size: Int,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var in_ptr = _make_ptr[dtype](in_addr)
    var lo_s = lo.cast[dtype]()
    var hi_s = hi.cast[dtype]()

    @always_inline
    @parameter
    @__copy_capture(out_ptr, in_ptr, lo_s, hi_s, has_min, has_max)
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i = Int(idx[0].value())
        var v = in_ptr[i]
        if has_min != 0:
            v = max(v, lo_s)
        if has_max != 0:
            v = min(v, hi_s)
        out_ptr[i] = v

    _parallel_for[func](size, ctx)


def _clamp_scalar_go(
    out_ptr: PyObjectPtr,
    in_ptr: PyObjectPtr,
    lo: PyObjectPtr,
    hi: PyObjectPtr,
    has_min: PyObjectPtr,
    has_max: PyObjectPtr,
    numel: PyObjectPtr,
    dtype: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var dtype_val = _raw_dtype_int(dtype)
    var out_addr = _raw_int(out_ptr)
    var in_addr = _raw_int(in_ptr)
    var lo_v = _raw_f64(lo)
    var hi_v = _raw_f64(hi)
    var has_min_v = _raw_int(has_min)
    var has_max_v = _raw_int(has_max)
    var size = _raw_int(numel)
    var ctx = _raw_ctx(ctx_ptr)

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
    ]:
        comptime if _dtype_arg_on[0, dt]():
            if dtype_val == dt:
                _clamp_scalar[dt](
                    out_addr,
                    in_addr,
                    lo_v,
                    hi_v,
                    has_min_v,
                    has_max_v,
                    size,
                    ctx,
                )
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast clamp: " + String(dtype_val))


def _clamp_scalar_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _clamp_scalar_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# Ternary broadcast: addcmul = self + value * (t1 * t2)
#                     addcdiv = self + value * (t1 / t2)
# out is contiguous with dims d0..d3; self/t1/t2 are indexed with their own
# strides (0 on broadcast dims). Half precision accumulates in float32.
# addcdiv is float-only (torch errors for integer inputs); addcmul also
# handles integer dtypes.
# ---------------------------------------------------------------------------

comptime TOP_ADDCMUL = 0
comptime TOP_ADDCDIV = 1


@always_inline
def _ternary_bcast[
    dtype: DType, op_code: Int
](
    out_addr: Int,
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    d1: Int,
    d2: Int,
    d3: Int,
    as0: Int,
    as1: Int,
    as2: Int,
    as3: Int,
    bs0: Int,
    bs1: Int,
    bs2: Int,
    bs3: Int,
    cs0: Int,
    cs1: Int,
    cs2: Int,
    cs3: Int,
    value: Float64,
    total: Int,
    ctx: DeviceContext,
) raises:
    comptime if op_code == TOP_ADDCDIV and not dtype.is_floating_point():
        raise Error("integer addcdiv is not supported in the fast path")
    else:
        var out_ptr = _make_ptr[dtype](out_addr)
        var a_ptr = _make_ptr[dtype](a_addr)
        var b_ptr = _make_ptr[dtype](b_addr)
        var c_ptr = _make_ptr[dtype](c_addr)

        # The Float64 `value` scalar must be narrowed to its GPU-side type
        # *before* it is captured by the closure below. Metal has zero
        # support for the `double` type -- not just arithmetic on it, but
        # even a single convert-from-f64 instruction inside a GPU-compiled
        # kernel trips "Metal-unsupported instructions", regardless of the
        # destination dtype. A comptime if/else can't merge into one `var`
        # of two different static types afterwards, so the two cases get
        # their own (otherwise identical) closure, each capturing an
        # already-narrowed scalar -- mirrors tensor_holder.mojo's
        # _strided_fill, which never leaks f64 into GPU codegen.
        comptime if dtype == DType.float16 or dtype == DType.bfloat16:
            var value_f32 = value.cast[DType.float32]()

            @always_inline
            @parameter
            @__copy_capture(out_ptr, a_ptr, b_ptr, c_ptr, value_f32)
            def func[width: Int, alignment: Int = 1](idx: Coord):
                var i = Int(idx[0].value())
                var i3 = i % d3
                var rest = i // d3
                var i2 = rest % d2
                rest = rest // d2
                var i1 = rest % d1
                var i0 = rest // d1
                var a = a_ptr[i0 * as0 + i1 * as1 + i2 * as2 + i3 * as3]
                var b = b_ptr[i0 * bs0 + i1 * bs1 + i2 * bs2 + i3 * bs3]
                var c = c_ptr[i0 * cs0 + i1 * cs1 + i2 * cs2 + i3 * cs3]
                var af = a.cast[DType.float32]()
                var bf = b.cast[DType.float32]()
                var cf = c.cast[DType.float32]()
                comptime if op_code == TOP_ADDCMUL:
                    out_ptr[i] = (af + value_f32 * (bf * cf)).cast[dtype]()
                else:
                    out_ptr[i] = (af + value_f32 * (bf / cf)).cast[dtype]()

            _parallel_for[func](total, ctx)
        else:
            var value_dt = value.cast[dtype]()

            @always_inline
            @parameter
            @__copy_capture(out_ptr, a_ptr, b_ptr, c_ptr, value_dt)
            def func2[width: Int, alignment: Int = 1](idx: Coord):
                var i = Int(idx[0].value())
                var i3 = i % d3
                var rest = i // d3
                var i2 = rest % d2
                rest = rest // d2
                var i1 = rest % d1
                var i0 = rest // d1
                var a = a_ptr[i0 * as0 + i1 * as1 + i2 * as2 + i3 * as3]
                var b = b_ptr[i0 * bs0 + i1 * bs1 + i2 * bs2 + i3 * bs3]
                var c = c_ptr[i0 * cs0 + i1 * cs1 + i2 * cs2 + i3 * cs3]
                comptime if dtype.is_floating_point():
                    comptime if op_code == TOP_ADDCMUL:
                        out_ptr[i] = a + value_dt * (b * c)
                    else:
                        out_ptr[i] = a + value_dt * (b / c)
                else:
                    # Integer addcmul: value is an exact integer scalar.
                    out_ptr[i] = a + value_dt * (b * c)

            _parallel_for[func2](total, ctx)


def _ternary_bcast_go[
    op_code: Int
](
    out_ptr: PyObjectPtr,
    a_ptr: PyObjectPtr,
    b_ptr: PyObjectPtr,
    c_ptr: PyObjectPtr,
    params: PyObjectPtr,  # (d0..d3, as0..as3, bs0..bs3, cs0..cs3)
    value: PyObjectPtr,
    dtype: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var dtype_val = _raw_dtype_int(dtype)
    var out_addr = _raw_int(out_ptr)
    var a_addr = _raw_int(a_ptr)
    var b_addr = _raw_int(b_ptr)
    var c_addr = _raw_int(c_ptr)
    var d0 = _raw_tuple_int(params, 0)
    var d1 = _raw_tuple_int(params, 1)
    var d2 = _raw_tuple_int(params, 2)
    var d3 = _raw_tuple_int(params, 3)
    var as0 = _raw_tuple_int(params, 4)
    var as1 = _raw_tuple_int(params, 5)
    var as2 = _raw_tuple_int(params, 6)
    var as3 = _raw_tuple_int(params, 7)
    var bs0 = _raw_tuple_int(params, 8)
    var bs1 = _raw_tuple_int(params, 9)
    var bs2 = _raw_tuple_int(params, 10)
    var bs3 = _raw_tuple_int(params, 11)
    var cs0 = _raw_tuple_int(params, 12)
    var cs1 = _raw_tuple_int(params, 13)
    var cs2 = _raw_tuple_int(params, 14)
    var cs3 = _raw_tuple_int(params, 15)
    var value_v = _raw_f64(value)
    var total = d0 * d1 * d2 * d3
    var ctx = _raw_ctx(ctx_ptr)

    @always_inline
    @parameter
    def run[dt: DType]() raises:
        _ternary_bcast[dt, op_code](
            out_addr,
            a_addr,
            b_addr,
            c_addr,
            d1,
            d2,
            d3,
            as0,
            as1,
            as2,
            as3,
            bs0,
            bs1,
            bs2,
            bs3,
            cs0,
            cs1,
            cs2,
            cs3,
            value_v,
            total,
            ctx,
        )

    var handled = False
    comptime for dt in [
        DType.float32,
        DType.float16,
        DType.bfloat16,
        DType.int8,
        DType.int16,
        DType.int32,
        DType.int64,
        DType.uint8,
    ]:
        comptime if _dtype_arg_on[0, dt]():
            if dtype_val == dt:
                run[dt]()
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast addc* op: " + String(dtype_val))


def _ternary_bcast_dispatcher[
    op_code: Int
](
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _ternary_bcast_go[op_code](
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# addr: out = beta*self + alpha*outer(vec1, vec2); self: (n, m) (any
# strides), vec1: (n,), vec2: (m,). A dedicated 2-D kernel rather than a new
# `_ternary_bcast` op_code: PyTorch's own vec1.reshape({n, 1}) broadcast
# (build_addr_iter in aten/src/ATen/native/LinearAlgebra.cpp) places vec1's
# only axis at dim 0, which plain right-aligned broadcasting of a 1-D
# operand against a 2-D one would instead place at dim 1 -- so the caller
# always passes each operand's *own* row/column stride below and this
# kernel indexes them by (i, j) directly, no shared broadcast-dims plumbing
# needed.
#
# This exists because, absent a native addr kernel, PyTorch's
# CompositeExplicitAutograd fallback for backends like this one
# (`math_addr`) composes outer/scale/add as three separately-dtype-rounded
# ops in a different multiplication order than CPU/CUDA's native addr_stub
# kernel -- enough rounding-order drift on top of an already coarse dtype
# to fail OpInfo conformance for fp16 (up to 18% of sampled elements; see
# conformance/test_opinfo.py::test_matches_cpu_addr_mojo_float16).
#
# The fix is NOT a higher-precision (float32) accumulator: CPU's own
# addr_kernel (aten/src/ATen/native/cpu/LinearAlgebraKernel.cpp) computes
# `beta*self + alpha*vec1*vec2` directly in the tensor's own dtype, and for
# fp16/bf16 that still rounds after every individual +/- (Half/BFloat16's
# operator overloads convert to float, do ONE op, and round back), so the
# reference this compares against has four separate low-precision roundings
# baked in, in a specific order: t1 = beta*self, t2 = alpha*vec1, t3 =
# t2*vec2, result = t1+t3. A single-final-rounding fp32 accumulation is
# *more* accurate but that makes it a worse match for this specific
# imperfectly-rounded reference (confirmed empirically: it still failed
# ~8% of elements at fp16). Reproducing CPU's exact op order and rounding
# granularity below matches it exactly.
#
# Hard-won, separate from the above: getting the arithmetic right was not
# enough on its own. With that arithmetic launched through `_parallel_for`
# (== `elementwise[func, simd_width=1]` on CPU), a couple of percent of
# fp16/bf16 elements still came out wrong -- reproducible, deterministic,
# and unaffected by which equivalent arithmetic formula or branch style
# (`if`/`else` statement vs. the branch-free `... if ... else ...`
# expression below) was used. Replacing `_parallel_for` with a plain
# sequential loop over the *same* closure on CPU (below) made it exact
# (0 mismatches over 100k+ randomized elements across both dtypes and two
# shape/beta/alpha combinations). Root cause not traced further than that;
# treat it as a MAX/Mojo `elementwise` CPU-backend issue specific to this
# closure's shape (six captured values incl. a Bool, three independently
# strided pointer reads) rather than a correctness property of the
# arithmetic. GPU still goes through `elementwise[..., target="gpu"]` via
# `_parallel_for`, same as every other kernel in this file, and was not
# rewritten to match: on an actual H100
# (test_matches_cpu_addr_mojo_float16/bfloat16), it landed exactly one
# element out of 50 just outside tolerance (down from up to 18% before
# this fix), where the arithmetic above reproduces CPU exactly to the bit
# on every sample tried. A hand-written grid-stride GPU kernel (bypassing
# `elementwise` entirely, mirroring `_bin_bcast_kernel` above) was tried
# and closed the CPU path back down to the pre-workaround failure rate
# when the same restructuring was applied there, so it was not safe to
# ship blind without more GPU time to verify; left as the next step.
# ---------------------------------------------------------------------------


@always_inline
def _addr_bcast[
    dtype: DType
](
    out_addr: Int,
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    n: Int,
    m: Int,
    as0: Int,
    as1: Int,
    bs0: Int,
    cs0: Int,
    beta: Float64,
    alpha: Float64,
    ctx: DeviceContext,
) raises:
    var out_ptr = _make_ptr[dtype](out_addr)
    var a_ptr = _make_ptr[dtype](a_addr)
    var b_ptr = _make_ptr[dtype](b_addr)
    var c_ptr = _make_ptr[dtype](c_addr)
    var total = n * m
    var beta_dt = beta.cast[dtype]()
    var alpha_dt = alpha.cast[dtype]()
    var beta_is_zero = beta_dt == 0
    # Every multiply below runs in float32 (never a native Scalar[dtype]
    # arithmetic op for fp16/bf16, only load/round-trip conversions): this
    # matches how CPU's own Half/BFloat16 operator overloads are
    # implemented (promote to float, do ONE op, round back), and sidesteps
    # it entirely for float32 dtype (where these casts are no-ops).
    var beta_f32 = beta_dt.cast[DType.float32]()
    var alpha_f32 = alpha_dt.cast[DType.float32]()

    @always_inline
    @parameter
    @__copy_capture(
        out_ptr, a_ptr, b_ptr, c_ptr, beta_f32, alpha_f32, beta_is_zero
    )
    def func[width: Int, alignment: Int = 1](idx: Coord):
        var i_flat = Int(idx[0].value())
        var i = i_flat // m
        var j = i_flat % m
        var bv = b_ptr[i * bs0]
        var cv = c_ptr[j * cs0]
        # Multiply in float32, then explicitly round down to `dtype` and
        # back up before the next op -- one rounding per op, matching
        # CPU's Half/BFloat16 operator overloads (convert to float, do ONE
        # op, round back) exactly. Skipping the round-trip and chaining
        # float32 multiplies with a single final rounding is *more*
        # accurate but drifts from this specific imperfectly-rounded
        # reference (see the module comment above).
        var t2 = (
            (alpha_f32 * bv.cast[DType.float32]())
            .cast[dtype]()
            .cast[DType.float32]()
        )
        var t3 = (
            (t2 * cv.cast[DType.float32]()).cast[dtype]().cast[DType.float32]()
        )
        # `self` is masked to 0 rather than branched around: beta==0 must
        # not propagate nan/inf from `self` (matches CPU), and a select on
        # an already-loaded value keeps this branch-free per element.
        var av = Scalar[dtype](0) if beta_is_zero else a_ptr[i * as0 + j * as1]
        var t1 = (
            (beta_f32 * av.cast[DType.float32]())
            .cast[dtype]()
            .cast[DType.float32]()
        )
        out_ptr[i_flat] = (t1 + t3).cast[dtype]()

    # Not `_parallel_for` on CPU -- see the module comment above.
    if ctx.api() == "cpu":
        for i in range(total):
            func[1](Coord(i))
    else:
        _parallel_for[func](total, ctx)


def _addr_bcast_go(
    out_ptr: PyObjectPtr,
    a_ptr: PyObjectPtr,
    b_ptr: PyObjectPtr,
    c_ptr: PyObjectPtr,
    params: PyObjectPtr,  # (n, m, as0, as1, bs0, cs0)
    beta: PyObjectPtr,
    alpha: PyObjectPtr,
    dtype: PyObjectPtr,
    ctx_ptr: PyObjectPtr,
) raises:
    var dtype_val = _raw_dtype_int(dtype)
    var out_addr = _raw_int(out_ptr)
    var a_addr = _raw_int(a_ptr)
    var b_addr = _raw_int(b_ptr)
    var c_addr = _raw_int(c_ptr)
    var n = _raw_tuple_int(params, 0)
    var m = _raw_tuple_int(params, 1)
    var as0 = _raw_tuple_int(params, 2)
    var as1 = _raw_tuple_int(params, 3)
    var bs0 = _raw_tuple_int(params, 4)
    var cs0 = _raw_tuple_int(params, 5)
    var beta_v = _raw_f64(beta)
    var alpha_v = _raw_f64(alpha)
    var ctx = _raw_ctx(ctx_ptr)

    @always_inline
    @parameter
    def run[dt: DType]() raises:
        _addr_bcast[dt](
            out_addr,
            a_addr,
            b_addr,
            c_addr,
            n,
            m,
            as0,
            as1,
            bs0,
            cs0,
            beta_v,
            alpha_v,
            ctx,
        )

    var handled = False
    comptime for dt in [DType.float32, DType.float16, DType.bfloat16]:
        comptime if _dtype_arg_on[0, dt]():
            if dtype_val == dt:
                run[dt]()
                handled = True
    if not handled:
        raise Error("unsupported dtype for fast addr op: " + String(dtype_val))


def _addr_bcast_dispatcher(
    py_self: PyObjectPtr,
    args_safe: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    var args = UnsafePointer(args_safe)
    try:
        _addr_bcast_go(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
        )
    except e:
        return _spec_unsupported(e)
    return _raw_ret_none()


# ---------------------------------------------------------------------------
# TensorSpec entries (docs/tensor_spec_design.md): the whole binary-broadcast
# op prologue — input checks, broadcast layout, output alloc, kernel launch —
# in one boundary call over cached TensorSpecs, reusing `_bin_bcast` /
# `_cmp_bcast` above. Failed checks raise a real NotImplementedError into
# Python ("take the classic path"); nothing is swallowed on spec paths.
# ---------------------------------------------------------------------------

comptime SPEC_BCAST_DTYPES = [
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


def _add_f32_bf16_spec_into_go(
    a_o: PyObjectPtr, b_o: PyObjectPtr, out_o: PyObjectPtr
) raises:
    """Contiguous FP32 + BF16 -> caller-allocated FP32."""
    ref a = _spec_ptr(a_o)[]
    ref b = _spec_ptr(b_o)[]
    ref out = _spec_ptr(out_o)[]

    if a.ctx_ptr != b.ctx_ptr or out.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec add f32 bf16 into: device mismatch")
    if not (
        (a.dtype == DType.float32 and b.dtype == DType.bfloat16)
        or (a.dtype == DType.bfloat16 and b.dtype == DType.float32)
    ):
        raise Error(
            "mojo spec add f32 bf16 into: expected one FP32 and one BF16"
        )
    if not (a.contig and b.contig and out.contig):
        raise Error("mojo spec add f32 bf16 into: tensors must be contiguous")
    if a.rank != b.rank or a.numel != b.numel or out.numel != a.numel:
        raise Error("mojo spec add f32 bf16 into: shapes differ")
    for i in range(MAX_RANK):
        if a.shape[i] != b.shape[i]:
            raise Error("mojo spec add f32 bf16 into: shapes differ")
    if out.dtype != DType.float32:
        raise Error("mojo spec add f32 bf16 into: output must be FP32")

    var ctx = a.ctx()
    if ctx.api() == "cpu":
        raise Error("mojo spec add f32 bf16 into: accelerator context required")

    var fp32_addr = a.ptr if a.dtype == DType.float32 else b.ptr
    var bf16_addr = a.ptr if a.dtype == DType.bfloat16 else b.ptr
    if a.numel > 0:
        _add_f32_bf16_contig(
            _make_ptr[DType.float32](out.ptr).as_unsafe_any_origin(),
            _make_ptr[DType.float32](fp32_addr)
            .as_unsafe_any_origin()
            .as_immutable(),
            _make_ptr[DType.bfloat16](bf16_addr)
            .as_unsafe_any_origin()
            .as_immutable(),
            a.numel,
            ctx,
        )


def _binary_spec_into_go[
    op_code: Int, is_cmp: Bool
](a_o: PyObjectPtr, b_o: PyObjectPtr, out_o: PyObjectPtr) raises:
    """Launch into a caller-allocated contiguous output. Python owns
    allocation and shape math, so the call returns nothing and can hold a
    FIFO slot while its unit builds."""
    ref a = _spec_ptr(a_o)[]
    ref b = _spec_ptr(b_o)[]
    ref out = _spec_ptr(out_o)[]

    if a.dtype != b.dtype:
        raise Error("mojo spec binary into: operand dtypes differ")
    if a.ctx_ptr != b.ctx_ptr or out.ctx_ptr != a.ctx_ptr:
        raise Error("mojo spec binary into: operands on different devices")

    var kdtype = a.dtype
    if a.dtype == DType.bool:
        comptime if (
            is_cmp
            or op_code == BOP_MUL
            or op_code == BOP_AND
            or op_code == BOP_OR
            or op_code == BOP_XOR
        ):
            kdtype = DType.uint8
        else:
            raise Error("mojo spec binary into: bool operands not supported")

    comptime if not is_cmp:
        comptime if op_code == BOP_DIV or op_code == BOP_POW:
            if not kdtype.is_floating_point():
                raise Error(
                    "mojo spec binary into: div/pow requires a float dtype"
                )
        comptime if (
            op_code == BOP_AND or op_code == BOP_OR or op_code == BOP_XOR
        ):
            if kdtype.is_floating_point():
                raise Error(
                    "mojo spec binary into: bitwise requires an int dtype"
                )

    var supported = False
    comptime for dt in SPEC_BCAST_DTYPES:
        comptime if _dtype_arg_abi_on[0, dt]():
            if kdtype == dt:
                supported = True
    if not supported:
        raise Error("mojo spec binary into: unsupported dtype ", a.dtype)

    var out_dtype = a.dtype
    comptime if is_cmp:
        out_dtype = DType.bool
    if out.dtype != out_dtype:
        raise Error("mojo spec binary into: output dtype mismatch")

    var d = IndexList[4](1)
    var ls = IndexList[4](0)
    var rs = IndexList[4](0)
    if a.rank > 4 or b.rank > 4:
        if a.rank != b.rank:
            raise Error("mojo spec binary into: rank > 4 needs equal shapes")
        for i in range(MAX_RANK):
            if a.shape[i] != b.shape[i]:
                raise Error(
                    "mojo spec binary into: rank > 4 needs equal shapes"
                )
        if not (a.contig and b.contig):
            raise Error(
                "mojo spec binary into: rank > 4 needs contiguous operands"
            )
        d[3] = a.numel
        ls[3] = 1
        rs[3] = 1
    else:
        for k in range(4):
            var i = MAX_RANK - 4 + k
            var sa = a.shape[i]
            var sb = b.shape[i]
            var s: Int
            if sa == sb:
                s = sa
            elif sa == 1:
                s = sb
            elif sb == 1:
                s = sa
            else:
                raise Error("mojo spec binary into: shapes do not broadcast")
            d[k] = s
            ls[k] = a.strides[i] if sa != 1 else 0
            rs[k] = b.strides[i] if sb != 1 else 0
    var numel = d[0] * d[1] * d[2] * d[3]
    if out.numel != numel or not out.contig:
        raise Error("mojo spec binary into: output buffer mismatch")

    var ctx = a.ctx()
    var addr = out.ptr
    if numel > 0:
        comptime for dt in SPEC_BCAST_DTYPES:
            comptime if _dtype_arg_abi_on[0, dt]():
                if kdtype == dt:
                    comptime if is_cmp:
                        _cmp_bcast[dt, op_code](
                            addr,
                            a.ptr,
                            b.ptr,
                            d[1],
                            d[2],
                            d[3],
                            ls[0],
                            ls[1],
                            ls[2],
                            ls[3],
                            rs[0],
                            rs[1],
                            rs[2],
                            rs[3],
                            numel,
                            ctx,
                        )
                    else:
                        _bin_bcast[dt, op_code](
                            addr,
                            a.ptr,
                            b.ptr,
                            d[1],
                            d[2],
                            d[3],
                            ls[0],
                            ls[1],
                            ls[2],
                            ls[3],
                            rs[0],
                            rs[1],
                            rs[2],
                            rs[3],
                            numel,
                            ctx,
                        )


# ---------------------------------------------------------------------------
# Python module definition
# ---------------------------------------------------------------------------


@export
def PyInit_logic_ops() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("logic_ops")
        comptime if _op_on["AddF32Bf16Spec"]():
            _register_call(
                b,
                _spec_dispatcher3[_add_f32_bf16_spec_into_go, "AddF32Bf16Spec"],
                docstring=(
                    "(a_spec, b_spec, out_spec); contiguous FP32 + BF16 -> FP32"
                ),
            )
        comptime if _op_on["AddSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_ADD, False], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); + ",
            )
        comptime if _op_on["SubSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_SUB, False], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); - ",
            )
        comptime if _op_on["MulSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_MUL, False], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); * ",
            )
        comptime if _op_on["DivSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_DIV, False], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); / float",
            )
        comptime if _op_on["MaximumSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_MAX, False], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); max",
            )
        comptime if _op_on["MinimumSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_MIN, False], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); min",
            )
        comptime if _op_on["PowSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_POW, False], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); ** float",
            )
        comptime if _op_on["RemainderSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_REMAINDER, False],
                    "a binary spec op",
                ],
                docstring="(a_spec, b_spec, out_spec); %",
            )
        comptime if _op_on["FloorDivSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_FLOORDIV, False],
                    "a binary spec op",
                ],
                docstring="(a_spec, b_spec, out_spec); //",
            )
        comptime if _op_on["BitwiseAndSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_AND, False], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); & int/bool",
            )
        comptime if _op_on["BitwiseOrSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_OR, False], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); | int/bool",
            )
        comptime if _op_on["BitwiseXorSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[BOP_XOR, False], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); ^ int/bool",
            )
        comptime if _op_on["EqSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[COP_EQ, True], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); == -> bool",
            )
        comptime if _op_on["NeSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[COP_NE, True], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); != -> bool",
            )
        comptime if _op_on["LtSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[COP_LT, True], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); < -> bool",
            )
        comptime if _op_on["LeSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[COP_LE, True], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); <= -> bool",
            )
        comptime if _op_on["GtSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[COP_GT, True], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); > -> bool",
            )
        comptime if _op_on["GeSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[COP_GE, True], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); >= -> bool",
            )
        comptime if _op_on["LogicalAndSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[COP_LAND, True], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); and -> bool",
            )
        comptime if _op_on["LogicalXorSpec"]():
            _register_call(
                b,
                _spec_dispatcher3[
                    _binary_spec_into_go[COP_LXOR, True], "a binary spec op"
                ],
                docstring="(a_spec, b_spec, out_spec); xor -> bool",
            )
        comptime if _op_on["BitwiseNot"]():
            _register_call(
                b,
                _bitwise_not_dispatcher,
                docstring="out = ~x (bool/int, contiguous)",
            )
        comptime if _op_on["IsIn"]():
            _register_call(
                b,
                _isin_dispatcher,
                docstring="out[i] = x[i] in test (int dtypes) ^ invert",
            )
        comptime if _op_on["ClampScalar"]():
            _register_call(
                b,
                _clamp_scalar_dispatcher,
                docstring="out = min(max(x, lo), hi) with optional bounds",
            )
        comptime if _op_on["AddcmulBcast"]():
            _register_call(
                b,
                _ternary_bcast_dispatcher[TOP_ADDCMUL],
                docstring="out = self + value * (t1 * t2) (broadcast strides)",
            )
        comptime if _op_on["AddcdivBcast"]():
            _register_call(
                b,
                _ternary_bcast_dispatcher[TOP_ADDCDIV],
                docstring="out = self + value * (t1 / t2) (broadcast strides)",
            )
        comptime if _op_on["AddrBcast"]():
            _register_call(
                b,
                _addr_bcast_dispatcher,
                docstring=(
                    "out = beta*self + alpha*(vec1 outer vec2); self:(n,m)"
                    " vec1:(n,) vec2:(m,)"
                ),
            )
        return b.finalize()
    except e:
        abort(t"failed to create logic_ops python module: {e}")
