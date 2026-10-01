"""ATen ops: resample group (see agents_docs/native_backend.md).

Reflection / replication padding (1-d, 2-d, 3-d) and nearest, nearest-exact,
linear, bilinear, bicubic, trilinear and antialiased bilinear upsampling,
each with its `.out` overload, its backward and the backward's
`.grad_input` overload -- the ops behind `F.pad(mode="reflect" |
"replicate")` and `F.interpolate`. The `.vec`
overloads are CompositeImplicitAutograd over these and need nothing here.

One kernel family, `resample` (tmb/kernels/resample/entry.mojo), serves all
of them: every op views its operands as contiguous (planes, D, H, W) with the
missing spatial extents set to 1, and the MODE / RANK / REFLECT defines pick
the formula. Shape checks and error messages follow the structured meta
functions (UpSample.h, Upsample*.cpp, ReflectionPad.cpp,
ReplicationPadding.cpp, Padding.h at v2.14.0); the scales follow the CUDA
host code (compute_scales_value, area_pixel_compute_scale), rounded to the
kernel's accumulation type here so the device reads them exactly.

Outputs are contiguous: CUDA keeps a channels-last input's memory format for
upsampling, the values are the same.
"""
from std.utils import IndexList

from tmb.backend.abi import (
    ST_UINT8,
    T,
    Value,
    Values,
    IntList,
    is_floating,
    new_tensor,
    own,
    own_if_new,
    ret_owned,
    ret_ref,
    unsupported,
    v_bool,
    v_f64,
    v_is_none,
    v_tensor,
)
from tmb.backend.device import ctx_for, ctx_ptr
from tmb.backend.kernel_call import KernelCall
from tmb.kernels.common.op_utils import MAX_RANK, _f64_slot
from tmb.ops.common import (
    OVERLAP_FULL,
    OVERLAP_PARTIAL,
    assert_no_internal_overlap,
    assert_no_partial_overlap,
    overlap_status,
    check_out,
    fill_value,
    same_view,
    shares_storage,
    contiguous,
    copy_strided_into,
    resize_out,
)
from tmb.backend.registry import Site, impl

comptime NEAREST = 0
comptime NEAREST_EXACT = 1
comptime LINEAR = 2
comptime CUBIC = 3
comptime BILINEAR_AA = 4


def _sizes_str(t: T) -> String:
    """`IntArrayRef` as torch prints it: [2, 3, 4]."""
    var s = String("[")
    for i in range(t.rank):
        if i:
            s += ", "
        s += String(t.dim(i))
    return s + "]"


def _list_str(l: List[Int]) -> String:
    var s = String("[")
    for i in range(len(l)):
        if i:
            s += ", "
        s += String(l[i])
    return s + "]"


def _pick(i: Int, a: StaticString, b: StaticString, c: StaticString) -> String:
    """The i-th of three names (spatial axes D, H, W)."""
    if i == 0:
        return String(a)
    if i == 1:
        return String(b)
    return String(c)


def _shape(dims: List[Int]) -> IndexList[MAX_RANK]:
    var shape = IndexList[MAX_RANK](1)
    for i in range(len(dims)):
        shape[MAX_RANK - len(dims) + i] = dims[i]
    return shape


def _require_mojo(t: T, what: String) raises:
    if not t.on_mojo():
        unsupported(what + ": operand is not on the mojo device")


comptime DIRECT = 0  # the kernel writes the caller's tensor
comptime COPY_BACK = 1  # into a fresh tensor, then copied into the caller's
comptime RESIZE_COPY = 2  # fresh tensor, then the caller's is resized + copied
comptime NO_OP = 3  # the caller's tensor already holds the result


def _dest(
    args: Values,
    i: Int,
    like: T,
    dims: List[Int],
    inputs: List[T],
    identity: Bool,
    internal_check: Bool = True,
) raises -> Tuple[T, Int]:
    """The tensor an `out=` / `grad_input=` overload writes, and how the
    result reaches the caller's tensor (DIRECT / COPY_BACK / RESIZE_COPY /
    NO_OP).

    `identity`: the op copies its input unchanged (CUDA's `output.copy_(input)`
    shortcut), so an `out` that IS that input view is a no-op, as on CUDA.
    torch's upsample and pad kernels run no overlap check against their
    input, so an `out` overlapping it does not raise: the result is computed
    into a fresh tensor and copied back (the input is read before anything
    is written), except upsample's same-size shortcut, CUDA's
    `output.copy_(input)`, which refuses a partial overlap like copy_.
    `internal_check`: upsample's out still refuses internal overlap (an
    expanded out), pad's does not. An `out` that must be resized
    while it shares storage with an input is resized only after the kernel
    ran into a fresh tensor: the resize may reallocate the storage the
    input's pointer still addresses."""
    var dst = v_tensor(args[unsafe_offset=i])
    check_out(dst, like)
    var shape = _shape(dims)
    var matches = dst.rank == len(dims)
    if matches:
        for k in range(len(dims)):
            if dst.dim(k) != dims[k]:
                matches = False
                break
    if matches and identity and same_view(dst, inputs[0]):
        return (dst^, NO_OP)
    var overlaps = False
    for k in range(len(inputs)):
        var status = overlap_status(dst, inputs[k])
        if status == OVERLAP_FULL or status == OVERLAP_PARTIAL:
            overlaps = True
        if identity and internal_check:
            # Upsample's same-size shortcut is `output.copy_(input)`, whose
            # TensorIterator refuses a partial overlap.
            assert_no_partial_overlap(dst, inputs[k])
    if not matches:
        for k in range(len(inputs)):
            if shares_storage(dst, inputs[k]):
                return (
                    new_tensor(shape, len(dims), dst.stype, dst.device),
                    RESIZE_COPY,
                )
        resize_out(dst, shape, len(dims))
        return (dst^, DIRECT)
    if internal_check:
        assert_no_internal_overlap(dst)
    if dst.contig and not overlaps:
        return (dst^, DIRECT)
    return (new_tensor(shape, len(dims), dst.stype, dst.device), COPY_BACK)


def _finish[
    OUT: Bool
](
    rets: Values,
    var res: T,
    how: Int,
    args: Values,
    out_index: Int,
    dims: List[Int],
) raises:
    """Hand `res` back: owned for the functional overload, else copied into
    the caller's tensor (resized first when `how` says so) and returned by
    reference."""
    comptime if OUT:
        var dst = v_tensor(args[unsafe_offset=out_index])
        if how == COPY_BACK or how == RESIZE_COPY:
            var tmp = own(res^)
            if how == RESIZE_COPY:
                resize_out(dst, _shape(dims), len(dims))
            copy_strided_into(dst, tmp.t)
            _ = tmp^
        ret_ref(rets, 0, dst)
    else:
        var o = own(res^)
        ret_owned(rets, 0, o)


# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------


def _up_name[MODE: Int, RANK: Int]() -> String:
    comptime if MODE == NEAREST:
        return String("upsample_nearest", RANK, "d")
    elif MODE == NEAREST_EXACT:
        return String("_upsample_nearest_exact", RANK, "d")
    elif MODE == CUBIC:
        return String("upsample_bicubic2d")
    elif MODE == BILINEAR_AA:
        return String("_upsample_bilinear2d_aa")
    else:
        comptime if RANK == 1:
            return String("upsample_linear1d")
        elif RANK == 2:
            return String("upsample_bilinear2d")
        else:
            return String("upsample_trilinear3d")


def _up_check[
    RANK: Int
](input_size: List[Int], osize: IntList) raises -> List[Int]:
    """upsample_{1,2,3}d_common_check: the full output size."""
    if len(osize) != RANK:
        raise Error(
            "It is expected output_size equals to ",
            RANK,
            ", but got size ",
            len(osize),
        )
    if len(input_size) != RANK + 2:
        raise Error(
            "It is expected input_size equals to ",
            RANK + 2,
            ", but got size ",
            len(input_size),
        )
    var positive = True
    for k in range(RANK):
        if input_size[2 + k] <= 0 or osize[k] <= 0:
            positive = False
    if not positive:
        var ins = String()
        var outs = String()
        for k in range(RANK):
            var label = _pick(3 - RANK + k, "D", "H", "W")
            if k:
                ins += ", "
                outs += ", "
            ins += String(label, ": ", input_size[2 + k])
            outs += String(label, ": ", osize[k])
        comptime if RANK == 1:
            raise Error(
                (
                    "Input and output sizes should be greater than 0, but got"
                    " input ("
                ),
                ins,
                ") and output (",
                outs,
                ")",
            )
        else:
            raise Error(
                (
                    "Input and output sizes should be greater than 0, but got"
                    " input ("
                ),
                ins,
                ") output (",
                outs,
                ")",
            )
    var full = List[Int]()
    full.append(input_size[0])
    full.append(input_size[1])
    for k in range(RANK):
        full.append(osize[k])
    return full^


def _opt_scale(v: Value) raises -> Float64:
    """An optional `float? scales_*` argument; 0 stands for None."""
    if v_is_none(v):
        return 0.0
    return v_f64(v)


def _kernel_scale[
    MODE: Int, BACKWARD: Bool
](
    in_size: Int, out_size: Int, align: Bool, scale: Float64, f64: Bool
) -> Float64:
    """The per-axis ratio the kernel reads, rounded to the type the CUDA
    kernel holds it in (float for nearest; accscalar_t otherwise)."""
    comptime if MODE <= NEAREST_EXACT:
        # compute_scales_value<float> / compute_scales_value_backwards<float>
        comptime if BACKWARD:
            if scale > 0.0:
                return Float64(Float32(scale))
            return Float64(Float32(out_size) / Float32(in_size))
        else:
            if scale > 0.0:
                return Float64(Float32(1.0 / scale))
            return Float64(Float32(in_size) / Float32(out_size))
    else:
        # area_pixel_compute_scale<accscalar_t>
        if align:
            if out_size <= 1:
                return 0.0
            if f64:
                return Float64(in_size - 1) / Float64(out_size - 1)
            return Float64(Float32(in_size - 1) / Float32(out_size - 1))
        if scale > 0.0:
            return 1.0 / scale if f64 else Float64(Float32(1.0 / scale))
        if f64:
            return Float64(in_size) / Float64(out_size)
        return Float64(Float32(in_size) / Float32(out_size))


def _up_dtype_ok[MODE: Int](t: T) -> Bool:
    if is_floating(t.stype):
        return True
    comptime if MODE <= NEAREST_EXACT:
        return t.stype == ST_UINT8
    else:
        return False


def _run_upsample[
    MODE: Int, RANK: Int, BACKWARD: Bool
](
    dst: T,
    src: T,
    in_dims: List[Int],
    out_dims: List[Int],
    align: Bool,
    scales: List[Float64],
) raises:
    """Launch Upsample / UpsampleBackward: `src` and `dst` contiguous, the
    dims full (N, C, spatial...) sizes of the forward input and output."""
    var geom = List[Int]()
    var sc = List[Int]()
    for k in range(3 - RANK):
        _ = k
        geom.append(1)
    for k in range(RANK):
        geom.append(in_dims[2 + k])
    for k in range(3 - RANK):
        _ = k
        geom.append(1)
    for k in range(RANK):
        geom.append(out_dims[2 + k])
    var f64 = src.dtype == DType.float64
    for k in range(3 - RANK):
        _ = k
        sc.append(_f64_slot(1.0))
    for k in range(RANK):
        sc.append(
            _f64_slot(
                _kernel_scale[MODE, BACKWARD](
                    in_dims[2 + k], out_dims[2 + k], align, scales[k], f64
                )
            )
        )
    var ctx = ctx_for(src.device)
    var call = KernelCall(
        "resample", "UpsampleBackward" if BACKWARD else "Upsample"
    )
    call.arg_dtype(0, src.dtype)
    call.flag("MODE", MODE)
    call.flag("RANK", RANK)
    call.int(dst.ptr)
    call.int(src.ptr)
    call.int(in_dims[0] * in_dims[1])
    call.tuple(geom)
    call.tuple(sc)
    call.int(1 if align else 0)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


def _is_identity[
    MODE: Int, RANK: Int, BACKWARD: Bool
](in_dims: List[Int], out_dims: List[Int], scales: List[Float64]) -> Bool:
    """Whether CUDA copies the input unchanged, per kernel: the host-side
    `copy_` of upsample_nearest2d / _upsample_nearest_exact2d /
    upsample_bilinear2d (forward and backward), the in-kernel "just copy" of
    the linear 1-d, trilinear and bicubic kernels, and the antialiased
    backward's. The nearest 1-d / 3-d kernels have no shortcut, but at an
    unchanged size with a unit scale their source index is the identity.
    The antialiased forward has none."""
    for k in range(RANK):
        if in_dims[2 + k] != out_dims[2 + k]:
            return False
    comptime if MODE <= NEAREST_EXACT and RANK != 2:
        for k in range(RANK):
            var n = in_dims[2 + k]
            if (
                _kernel_scale[MODE, BACKWARD](n, n, False, scales[k], False)
                != 1.0
            ):
                return False
        return True
    elif MODE == BILINEAR_AA:
        return BACKWARD
    else:
        return True


def op_upsample[
    MODE: Int, RANK: Int, OUT: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """aten::upsample_nearest{1,2,3}d, _upsample_nearest_exact{1,2,3}d,
    upsample_linear1d, upsample_bilinear2d, upsample_bicubic2d,
    upsample_trilinear3d and their `.out`:
    (Tensor self, SymInt[R] output_size, [bool align_corners,]
    float? scales..., [*, Tensor(a!) out])."""
    comptime name = _up_name[MODE, RANK]()
    comptime interp = MODE >= LINEAR
    comptime first_scale = 3 if interp else 2
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a, name)
    var osize = IntList(args[unsafe_offset=1])
    var align = v_bool(args[unsafe_offset=2]) if interp else False
    var scales = List[Float64]()
    for k in range(RANK):
        scales.append(_opt_scale(args[unsafe_offset=first_scale + k]))
    var in_dims = List[Int]()
    for k in range(a.rank):
        in_dims.append(a.dim(k))
    var out_dims = _up_check[RANK](in_dims, osize)
    if a.numel == 0 and in_dims[1] == 0:
        raise Error(
            "Non-empty ",
            RANK + 2,
            "D data tensor expected but got a tensor with sizes ",
            _sizes_str(a),
        )
    if not _up_dtype_ok[MODE](a):
        unsupported(String(name, ": dtype ", a.dtype))
    var identity = _is_identity[MODE, RANK, False](in_dims, out_dims, scales)
    var dst: T
    var how = DIRECT
    comptime if OUT:
        var d = _dest(
            args, first_scale + RANK, a, out_dims, [a.copy()], identity
        )
        dst = d[0].copy()
        how = d[1]
    else:
        dst = new_tensor(_shape(out_dims), RANK + 2, a.stype, a.device)
    if dst.numel > 0 and how != NO_OP:
        var src = own_if_new(contiguous(a), a)
        if identity:
            copy_strided_into(dst, src.t)
        else:
            _run_upsample[MODE, RANK, False](
                dst, src.t, in_dims, out_dims, align, scales
            )
        _ = src^
    _finish[OUT](rets, dst^, how, args, first_scale + RANK, out_dims)


def op_upsample_backward[
    MODE: Int, RANK: Int, OUT: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """The `_backward` / `_backward.grad_input` overloads: (Tensor
    grad_output, SymInt[R] output_size, SymInt[R+2] input_size,
    [bool align_corners,] float? scales..., [*, Tensor(a!) grad_input])."""
    comptime name = _up_name[MODE, RANK]() + "_backward"
    comptime interp = MODE >= LINEAR
    comptime first_scale = 4 if interp else 3
    var g = v_tensor(args[unsafe_offset=0])
    _require_mojo(g, name)
    var osize = IntList(args[unsafe_offset=1])
    var isize = IntList(args[unsafe_offset=2])
    var align = v_bool(args[unsafe_offset=3]) if interp else False
    var scales = List[Float64]()
    for k in range(RANK):
        scales.append(_opt_scale(args[unsafe_offset=first_scale + k]))
    var in_dims = isize.to_list()
    var out_dims = _up_check[RANK](in_dims, osize)
    if g.rank != RANK + 2:
        raise Error(
            "Expected grad_output to be a tensor of dimension ",
            RANK + 2,
            " but got: dimension ",
            g.rank,
        )
    for k in range(RANK + 2):
        if g.dim(k) != out_dims[k]:
            raise Error(
                "Expected grad_output to have the same shape as output;",
                " output.size(",
                k,
                ") = ",
                out_dims[k],
                " but got grad_output.size(",
                k,
                ") = ",
                g.dim(k),
            )
    if not _up_dtype_ok[MODE](g):
        unsupported(String(name, ": dtype ", g.dtype))
    var identity = _is_identity[MODE, RANK, True](in_dims, out_dims, scales)
    var dst: T
    var how = DIRECT
    comptime if OUT:
        var d = _dest(
            args, first_scale + RANK, g, in_dims, [g.copy()], identity
        )
        dst = d[0].copy()
        how = d[1]
    else:
        dst = new_tensor(_shape(in_dims), RANK + 2, g.stype, g.device)
    comptime if MODE >= LINEAR:
        # These backwards zero grad_input before copying grad_output into
        # it, so a grad_input that IS grad_output comes back zeroed.
        # Only bilinear2d copies from grad_output itself; the linear 1-d,
        # trilinear, bicubic and antialiased backwards read a
        # `.contiguous()` of it, which for a strided grad_output is a copy
        # made before the zeroing -- the values survive there.
        if how == NO_OP and dst.numel > 0:
            if (MODE == LINEAR and RANK == 2) or g.contig:
                fill_value(dst, 0.0)
    if dst.numel > 0 and how != NO_OP:
        var src = own_if_new(contiguous(g), g)
        if identity:
            copy_strided_into(dst, src.t)
        else:
            _run_upsample[MODE, RANK, True](
                dst, src.t, in_dims, out_dims, align, scales
            )
        _ = src^
    _finish[OUT](rets, dst^, how, args, first_scale + RANK, in_dims)


# ---------------------------------------------------------------------------
# Reflection / replication padding
# ---------------------------------------------------------------------------


def _pad_name[REFLECT: Bool, RANK: Int]() -> String:
    comptime if REFLECT:
        return String("reflection_pad", RANK, "d")
    else:
        return String("replication_pad", RANK, "d")


def _reflect_check[RANK: Int](t: T, padding: IntList) raises:
    """Reflection's `pad < input size` check, per padded dim (the meta
    functions of the forward and of the backward)."""
    var lead = t.rank - RANK
    for k in range(RANK):
        # padding pairs run from the last dim: (left, right, top, ...)
        var axis = RANK - 1 - k
        var n = t.dim(lead + axis)
        var lo = padding[2 * k]
        var hi = padding[2 * k + 1]
        if lo >= n or hi >= n:
            var prefix = String()
            comptime if RANK != 2:
                prefix = String(
                    "Argument ",
                    _pick(3 - RANK + axis, "#8", "#6", "#4"),
                    ": ",
                )
            raise Error(
                prefix,
                (
                    "Padding size should be less than the corresponding"
                    " input dimension, but got: padding ("
                ),
                lo,
                ", ",
                hi,
                ") at dimension ",
                lead + axis,
                " of input ",
                _sizes_str(t),
            )


def _pad_geom[
    REFLECT: Bool, RANK: Int
](t: T, padding: IntList) raises -> List[Int]:
    """The meta checks of reflection_pad{1,2,3}d / replication_pad{1,2,3}d
    (Padding.h's check_valid_input, then the per-op size checks): the full
    output size."""
    if len(padding) != 2 * RANK:
        comptime if REFLECT:
            raise Error(
                "padding size is expected to be ",
                2 * RANK,
                ", but got: ",
                len(padding),
            )
        else:
            raise Error("padding size is expected to be ", 2 * RANK)
    var batch_mode = t.rank == RANK + 2
    var valid = batch_mode or t.rank == RANK + 1
    if valid:
        for k in range(1 if batch_mode else 0, t.rank):
            if t.dim(k) == 0:
                valid = False
    if not valid:
        raise Error(
            "Expected ",
            RANK + 1,
            "D or ",
            RANK + 2,
            (
                "D (batch mode) tensor with possibly 0 batch size and other"
                " non-zero dimensions for input, but got: "
            ),
            _sizes_str(t),
        )
    var lead = t.rank - RANK
    comptime if REFLECT:
        _reflect_check[RANK](t, padding)
    var dims = List[Int]()
    for k in range(lead):
        dims.append(t.dim(k))
    var any_pos = False
    var all_pos = True
    for axis in range(RANK):
        var k = RANK - 1 - axis
        var o = t.dim(lead + axis) + padding[2 * k] + padding[2 * k + 1]
        dims.append(o)
        if o >= 1:
            any_pos = True
        else:
            all_pos = False
    # Reflection 2-d/3-d only require one positive extent (CUDA's `||`).
    var ok = all_pos
    comptime if REFLECT and RANK >= 2:
        ok = any_pos
    if not ok:
        var ins = String()
        var outs = String()
        for axis in range(RANK):
            var label = _pick(3 - RANK + axis, "D", "H", "W")
            if axis:
                ins += ", "
                outs += " "
            ins += String(label, ": ", t.dim(lead + axis))
            outs += String(label, ": ", dims[lead + axis])
        comptime if RANK == 1:
            raise Error(
                "input (", ins, ") is too small. Calculated output ", outs
            )
        elif REFLECT:
            raise Error(
                "input (", ins, ") is too small. Calculated output ", outs
            )
        else:
            raise Error(
                "Calculated output ",
                outs,
                " must be >= 1 in every dimension (input ",
                ins,
                ")",
            )
    for k in range(len(dims)):
        if dims[k] < 0:
            raise Error(
                "Trying to create tensor with negative dimension ",
                dims[k],
                ": ",
                _list_str(dims),
            )
    return dims^


def _run_pad[
    REFLECT: Bool, RANK: Int, BACKWARD: Bool
](
    dst: T, src: T, in_dims: List[Int], out_dims: List[Int], padding: IntList
) raises:
    var lead = len(in_dims) - RANK
    var batch = 1
    for k in range(lead):
        batch *= in_dims[k]
    var geom = List[Int]()
    for k in range(3 - RANK):
        _ = k
        geom.append(1)
    for axis in range(RANK):
        geom.append(in_dims[lead + axis])
    for k in range(3 - RANK):
        _ = k
        geom.append(0)
    for axis in range(RANK):
        geom.append(padding[2 * (RANK - 1 - axis)])
    for k in range(3 - RANK):
        _ = k
        geom.append(1)
    for axis in range(RANK):
        geom.append(out_dims[lead + axis])
    var ctx = ctx_for(src.device)
    var call = KernelCall("resample", "PadBackward" if BACKWARD else "Pad")
    call.arg_dtype(0, src.dtype)
    call.flag("REFLECT", 1 if REFLECT else 0)
    call.int(dst.ptr)
    call.int(src.ptr)
    call.int(batch)
    call.tuple(geom)
    call.int(ctx_ptr(ctx))
    call.run()
    _ = ctx


def op_pad[
    REFLECT: Bool, RANK: Int, OUT: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """aten::reflection_pad{1,2,3}d / replication_pad{1,2,3}d and `.out`:
    (Tensor self, SymInt[2R] padding, [*, Tensor(a!) out])."""
    comptime name = _pad_name[REFLECT, RANK]()
    var a = v_tensor(args[unsafe_offset=0])
    _require_mojo(a, name)
    var padding = IntList(args[unsafe_offset=1])
    var out_dims = _pad_geom[REFLECT, RANK](a, padding)
    if a.dtype == DType.bool:
        unsupported(String(name, ": bool"))
    var in_dims = List[Int]()
    for k in range(a.rank):
        in_dims.append(a.dim(k))
    # All-zero padding copies the input unchanged.
    var identity = True
    for k in range(len(padding)):
        if padding[k] != 0:
            identity = False
    var dst: T
    var how = DIRECT
    comptime if OUT:
        var d = _dest(args, 2, a, out_dims, [a.copy()], identity, False)
        dst = d[0].copy()
        how = d[1]
    else:
        dst = new_tensor(_shape(out_dims), a.rank, a.stype, a.device)
    if dst.numel > 0 and how != NO_OP:
        var src = own_if_new(contiguous(a), a)
        _run_pad[REFLECT, RANK, False](dst, src.t, in_dims, out_dims, padding)
        _ = src^
    _finish[OUT](rets, dst^, how, args, 2, out_dims)


def op_pad_backward[
    REFLECT: Bool, RANK: Int, OUT: Bool
](args: Values, n_args: Int, rets: Values, n_rets: Int) raises:
    """The `_backward` / `_backward.grad_input` overloads: (Tensor
    grad_output, Tensor self, SymInt[2R] padding, [*, Tensor(a!)
    grad_input])."""
    comptime name = _pad_name[REFLECT, RANK]() + "_backward"
    var g = v_tensor(args[unsafe_offset=0])
    var a = v_tensor(args[unsafe_offset=1])
    _require_mojo(g, name)
    var padding = IntList(args[unsafe_offset=2])
    if len(padding) != 2 * RANK:
        raise Error("padding size is expected to be ", 2 * RANK)
    if a.rank != RANK + 1 and a.rank != RANK + 2:
        unsupported(String(name, ": input of rank ", a.rank))
    comptime if REFLECT:
        _reflect_check[RANK](a, padding)
    var lead = a.rank - RANK
    if g.rank != a.rank:
        raise Error(
            "grad_output rank unexpected. Expected: ",
            a.rank,
            ", Got: ",
            g.rank,
        )
    for k in range(lead):
        if g.dim(k) != a.dim(k):
            raise Error(
                "gradOutput channel unexpected. Expected: ",
                a.dim(k),
                ", Got: ",
                g.dim(k),
            )
    var in_dims = List[Int]()
    var out_dims = List[Int]()
    for k in range(lead):
        in_dims.append(a.dim(k))
        out_dims.append(a.dim(k))
    for axis in range(RANK):
        var k = RANK - 1 - axis
        var n = a.dim(lead + axis)
        in_dims.append(n)
        var o = n + padding[2 * k] + padding[2 * k + 1]
        if o != g.dim(lead + axis):
            raise Error(
                "grad_output ",
                _pick(3 - RANK + axis, "depth", "height", "width"),
                " unexpected. Expected: ",
                o,
                ", Got: ",
                g.dim(lead + axis),
            )
        out_dims.append(o)
    if not is_floating(g.stype):
        unsupported(String(name, ": dtype ", g.dtype))
    if a.stype != g.stype:
        unsupported(String(name, ": grad_output and self dtypes differ"))
    var dst: T
    var how = DIRECT
    comptime if OUT:
        # `self` is only read for its shape, so it may alias grad_input.
        var d = _dest(args, 3, g, in_dims, [g.copy()], False, False)
        dst = d[0].copy()
        how = d[1]
    else:
        dst = new_tensor(_shape(in_dims), a.rank, g.stype, g.device)
    if dst.numel > 0:
        var src = own_if_new(contiguous(g), g)
        _run_pad[REFLECT, RANK, True](dst, src.t, in_dims, out_dims, padding)
        _ = src^
    _finish[OUT](rets, dst^, how, args, 3, in_dims)


def register_resample(site: Site) raises:
    impl[op_upsample[NEAREST, 1, False], "upsample_nearest1d"](site)
    impl[op_upsample[NEAREST, 1, True], "upsample_nearest1d.out"](site)
    impl[
        op_upsample_backward[NEAREST, 1, False], "upsample_nearest1d_backward"
    ](site)
    impl[
        op_upsample_backward[NEAREST, 1, True],
        "upsample_nearest1d_backward.grad_input",
    ](site)
    impl[op_upsample[NEAREST, 2, False], "upsample_nearest2d"](site)
    impl[op_upsample[NEAREST, 2, True], "upsample_nearest2d.out"](site)
    impl[
        op_upsample_backward[NEAREST, 2, False], "upsample_nearest2d_backward"
    ](site)
    impl[
        op_upsample_backward[NEAREST, 2, True],
        "upsample_nearest2d_backward.grad_input",
    ](site)
    impl[op_upsample[NEAREST, 3, False], "upsample_nearest3d"](site)
    impl[op_upsample[NEAREST, 3, True], "upsample_nearest3d.out"](site)
    impl[
        op_upsample_backward[NEAREST, 3, False], "upsample_nearest3d_backward"
    ](site)
    impl[
        op_upsample_backward[NEAREST, 3, True],
        "upsample_nearest3d_backward.grad_input",
    ](site)
    impl[op_upsample[NEAREST_EXACT, 1, False], "_upsample_nearest_exact1d"](
        site
    )
    impl[op_upsample[NEAREST_EXACT, 1, True], "_upsample_nearest_exact1d.out"](
        site
    )
    impl[
        op_upsample_backward[NEAREST_EXACT, 1, False],
        "_upsample_nearest_exact1d_backward",
    ](site)
    impl[
        op_upsample_backward[NEAREST_EXACT, 1, True],
        "_upsample_nearest_exact1d_backward.grad_input",
    ](site)
    impl[op_upsample[NEAREST_EXACT, 2, False], "_upsample_nearest_exact2d"](
        site
    )
    impl[op_upsample[NEAREST_EXACT, 2, True], "_upsample_nearest_exact2d.out"](
        site
    )
    impl[
        op_upsample_backward[NEAREST_EXACT, 2, False],
        "_upsample_nearest_exact2d_backward",
    ](site)
    impl[
        op_upsample_backward[NEAREST_EXACT, 2, True],
        "_upsample_nearest_exact2d_backward.grad_input",
    ](site)
    impl[op_upsample[NEAREST_EXACT, 3, False], "_upsample_nearest_exact3d"](
        site
    )
    impl[op_upsample[NEAREST_EXACT, 3, True], "_upsample_nearest_exact3d.out"](
        site
    )
    impl[
        op_upsample_backward[NEAREST_EXACT, 3, False],
        "_upsample_nearest_exact3d_backward",
    ](site)
    impl[
        op_upsample_backward[NEAREST_EXACT, 3, True],
        "_upsample_nearest_exact3d_backward.grad_input",
    ](site)
    impl[op_upsample[LINEAR, 1, False], "upsample_linear1d"](site)
    impl[op_upsample[LINEAR, 1, True], "upsample_linear1d.out"](site)
    impl[op_upsample_backward[LINEAR, 1, False], "upsample_linear1d_backward"](
        site
    )
    impl[
        op_upsample_backward[LINEAR, 1, True],
        "upsample_linear1d_backward.grad_input",
    ](site)
    impl[op_upsample[LINEAR, 2, False], "upsample_bilinear2d"](site)
    impl[op_upsample[LINEAR, 2, True], "upsample_bilinear2d.out"](site)
    impl[
        op_upsample_backward[LINEAR, 2, False], "upsample_bilinear2d_backward"
    ](site)
    impl[
        op_upsample_backward[LINEAR, 2, True],
        "upsample_bilinear2d_backward.grad_input",
    ](site)
    impl[op_upsample[LINEAR, 3, False], "upsample_trilinear3d"](site)
    impl[op_upsample[LINEAR, 3, True], "upsample_trilinear3d.out"](site)
    impl[
        op_upsample_backward[LINEAR, 3, False], "upsample_trilinear3d_backward"
    ](site)
    impl[
        op_upsample_backward[LINEAR, 3, True],
        "upsample_trilinear3d_backward.grad_input",
    ](site)
    impl[op_upsample[CUBIC, 2, False], "upsample_bicubic2d"](site)
    impl[op_upsample[CUBIC, 2, True], "upsample_bicubic2d.out"](site)
    impl[op_upsample_backward[CUBIC, 2, False], "upsample_bicubic2d_backward"](
        site
    )
    impl[
        op_upsample_backward[CUBIC, 2, True],
        "upsample_bicubic2d_backward.grad_input",
    ](site)
    impl[op_upsample[BILINEAR_AA, 2, False], "_upsample_bilinear2d_aa"](site)
    impl[op_upsample[BILINEAR_AA, 2, True], "_upsample_bilinear2d_aa.out"](site)
    impl[
        op_upsample_backward[BILINEAR_AA, 2, False],
        "_upsample_bilinear2d_aa_backward",
    ](site)
    impl[
        op_upsample_backward[BILINEAR_AA, 2, True],
        "_upsample_bilinear2d_aa_backward.grad_input",
    ](site)
    impl[op_pad[True, 1, False], "reflection_pad1d"](site)
    impl[op_pad[True, 1, True], "reflection_pad1d.out"](site)
    impl[op_pad_backward[True, 1, False], "reflection_pad1d_backward"](site)
    impl[
        op_pad_backward[True, 1, True], "reflection_pad1d_backward.grad_input"
    ](site)
    impl[op_pad[True, 2, False], "reflection_pad2d"](site)
    impl[op_pad[True, 2, True], "reflection_pad2d.out"](site)
    impl[op_pad_backward[True, 2, False], "reflection_pad2d_backward"](site)
    impl[
        op_pad_backward[True, 2, True], "reflection_pad2d_backward.grad_input"
    ](site)
    impl[op_pad[True, 3, False], "reflection_pad3d"](site)
    impl[op_pad[True, 3, True], "reflection_pad3d.out"](site)
    impl[op_pad_backward[True, 3, False], "reflection_pad3d_backward"](site)
    impl[
        op_pad_backward[True, 3, True], "reflection_pad3d_backward.grad_input"
    ](site)
    impl[op_pad[False, 1, False], "replication_pad1d"](site)
    impl[op_pad[False, 1, True], "replication_pad1d.out"](site)
    impl[op_pad_backward[False, 1, False], "replication_pad1d_backward"](site)
    impl[
        op_pad_backward[False, 1, True], "replication_pad1d_backward.grad_input"
    ](site)
    impl[op_pad[False, 2, False], "replication_pad2d"](site)
    impl[op_pad[False, 2, True], "replication_pad2d.out"](site)
    impl[op_pad_backward[False, 2, False], "replication_pad2d_backward"](site)
    impl[
        op_pad_backward[False, 2, True], "replication_pad2d_backward.grad_input"
    ](site)
    impl[op_pad[False, 3, False], "replication_pad3d"](site)
    impl[op_pad[False, 3, True], "replication_pad3d.out"](site)
    impl[op_pad_backward[False, 3, False], "replication_pad3d_backward"](site)
    impl[
        op_pad_backward[False, 3, True], "replication_pad3d_backward.grad_input"
    ](site)
