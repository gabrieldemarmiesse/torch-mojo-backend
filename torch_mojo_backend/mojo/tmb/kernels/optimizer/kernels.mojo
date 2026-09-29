"""Runtime-dynamic, multi-tensor fused optimizer kernels: Adam, AdamW, SGD and
Adagrad (`aten::_fused_{adam,adamw,sgd,adagrad}_`).

One body per algorithm, comptime-parametrized by the parameter dtype and the
optimizer-state dtype (the same except for Adam/AdamW's mixed-precision mode,
float32 params and grads with bfloat16 states). The element math is ATen's
CUDA functors, operation for operation, with their rounding points:

  * Adam / AdamW: `adam_math` in aten/src/ATen/native/cuda/fused_adam_utils.cuh
    (the two nested fmas of the moment updates included), bias corrections
    from `powf` / `pow` of the float step (`FusedAdamMathFunctor`, or in
    double for the mixed-precision `FusedAdamMathFunctorMP`);
  * SGD: `sgd_math` in FusedSgdKernel.cu;
  * Adagrad: `adagrad_math` in fused_adagrad_utils.cuh, whose learning-rate
    arithmetic is in double.

Half and bfloat16 values are widened to float32 (`at::opmath_type`), float64
stays double, and every state is rounded to its own dtype on store while the
float value keeps flowing through the rest of the update, exactly as the
functor's registers do. Apple GPUs have no float64, so the double parts (host
hyperparameters, Adagrad's lr arithmetic, the mixed-precision bias
corrections) run in float32 there.

Two launch shapes share that body. Everywhere but Apple a descriptor batch is
passed by value and each block owns one chunk of the concatenated list (the
chunk size filling the device, `foreach_ew_chunk_elements`), mapped back to
its tensor through the descriptors' `chunk_end` prefix sums. Metal only
translates pointer-typed kernel *arguments* into GPU addresses, so on Apple
each tensor is one launch with real pointer arguments. Tensor sizes and
addresses are runtime data and never compilation keys.
"""

from std.collections import Array
from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import fma, min
from std.sys.info import has_apple_gpu_accelerator, size_of

from tmb.kernels.common.op_utils import _enqueue_cached, ieee_sqrt
from tmb.kernels.common.pow_math import pow_c99
from tmb.kernels.optimizer.contract import (
    FUSED_OPT_DESC_CAP,
    FUSED_OPT_THREADS,
    FusedOptDesc,
)


comptime FUSED_ADAM = 0
comptime FUSED_ADAMW = 1
comptime FUSED_SGD = 2
comptime FUSED_ADAGRAD = 3

# Bits of the runtime `flags` argument.
comptime FO_AMSGRAD = 1
comptime FO_MAXIMIZE = 2
comptime FO_NESTEROV = 4
comptime FO_FIRST_STEP = 8
comptime FO_MOMENTUM = 16  # SGD: a momentum buffer list was given

# Host hyperparameters cross the launch ABI in the precision ATen's functors
# receive them (double), except on Apple GPUs, which have no float64.
comptime FO_HYPER = (
    DType.float32 if has_apple_gpu_accelerator() else DType.float64
)

comptime _VEC = 4


@always_inline
def fused_opt_label[algo: Int]() -> StaticString:
    """Algorithm fragment of the kernel name a profiler prints."""
    comptime if algo == FUSED_ADAM:
        return "adam"
    elif algo == FUSED_ADAMW:
        return "adamw"
    elif algo == FUSED_SGD:
        return "sgd"
    elif algo == FUSED_ADAGRAD:
        return "adagrad"
    else:
        comptime assert False, "unknown fused optimizer"


@always_inline
def _opmath[dtype: DType]() -> DType:
    """`at::opmath_type`: float for the half types, the type itself else."""
    return DType.float64 if dtype == DType.float64 else DType.float32


@always_inline
def _wide[pdt: DType]() -> DType:
    """Where ATen computes in double: double, or float32 on Apple GPUs."""
    return DType.float64 if pdt == DType.float64 else FO_HYPER


@always_inline
def _ld[
    dtype: DType, ot: DType, width: Int
](ptr: Pointer[Scalar[dtype], MutAnyOrigin], index: Int) -> SIMD[ot, width]:
    return ptr.unsafe_load[width=width, alignment=size_of[dtype]()](index).cast[
        ot
    ]()


@always_inline
def _st[
    dtype: DType, ot: DType, width: Int
](
    ptr: Pointer[Scalar[dtype], MutAnyOrigin],
    index: Int,
    value: SIMD[ot, width],
):
    ptr.unsafe_store[width=width, alignment=size_of[dtype]()](
        index, value.cast[dtype]()
    )


struct _Consts[ot: DType, aw: DType](TrivialRegisterPassable):
    """One block's constants, laid out per algorithm by `_fused_opt_range`:

    Adam/AdamW: k0 lr, k1 beta1, k2 beta2, k3 weight_decay, k4 eps,
                k5 bias_correction1, k6 bias_correction2_sqrt
    SGD:        k0 lr, k1 weight_decay, k2 momentum, k3 dampening
    Adagrad:    w0 corrected lr, w1 weight_decay, w2 eps (ATen's doubles)
    """

    var k0: Scalar[Self.ot]
    var k1: Scalar[Self.ot]
    var k2: Scalar[Self.ot]
    var k3: Scalar[Self.ot]
    var k4: Scalar[Self.ot]
    var k5: Scalar[Self.ot]
    var k6: Scalar[Self.ot]
    var w0: Scalar[Self.aw]
    var w1: Scalar[Self.aw]
    var w2: Scalar[Self.aw]
    var grad_scale: Scalar[Self.ot]

    def __init__(out self):
        self.k0 = 0
        self.k1 = 0
        self.k2 = 0
        self.k3 = 0
        self.k4 = 0
        self.k5 = 0
        self.k6 = 0
        self.w0 = 0
        self.w1 = 0
        self.w2 = 0
        self.grad_scale = 1


@always_inline
def _fused_opt_elements[
    pdt: DType, sdt: DType, algo: Int, width: Int
](
    params: Pointer[Scalar[pdt], MutAnyOrigin],
    grads: Pointer[Scalar[pdt], MutAnyOrigin],
    state0: Pointer[Scalar[sdt], MutAnyOrigin],
    state1: Pointer[Scalar[sdt], MutAnyOrigin],
    state2: Pointer[Scalar[sdt], MutAnyOrigin],
    index: Int,
    c: _Consts[_opmath[pdt](), _wide[pdt]()],
    has_grad_scale: Bool,
    flags: Int,
):
    """`width` elements at `index`: the single definition of each
    algorithm's element math (vector body and scalar tail both call it)."""
    comptime ot = _opmath[pdt]()
    comptime aw = _wide[pdt]()
    comptime V = SIMD[ot, width]
    var p = _ld[pdt, ot, width](params, index)
    var g = _ld[pdt, ot, width](grads, index)
    var maximize = (flags & FO_MAXIMIZE) != 0

    # Every operand is loaded before anything is stored, and the stores run
    # in the functor's order (param, grad, then the states in list order;
    # the mixed-precision Adam functor stores grad last): when two lists
    # alias, the one CUDA writes last wins here too.
    comptime if algo == FUSED_ADAM or algo == FUSED_ADAMW:
        var amsgrad = (flags & FO_AMSGRAD) != 0
        var m = _ld[sdt, ot, width](state0, index)
        var v = _ld[sdt, ot, width](state1, index)
        var mx = V(0)
        if amsgrad:
            mx = _ld[sdt, ot, width](state2, index)
        if has_grad_scale:
            g = g / c.grad_scale
        var grad_to_store = g
        if maximize:
            g = -g
        if c.k3 != 0:
            comptime if algo == FUSED_ADAM:
                g += p * c.k3
            else:
                p -= c.k0 * c.k3 * p
        m = fma(V(c.k1), m, fma(V(-c.k1), g, g))
        var gg = g * g
        v = fma(V(c.k2), v, fma(V(-c.k2), gg, gg))
        var step_size = c.k0 / c.k5
        var denom: V
        if amsgrad:
            # std::max(a, b) is `a < b ? b : a`: a NaN `a` stays.
            mx = mx.lt(v).select(v, mx)
            denom = ieee_sqrt(mx) / c.k6 + c.k4
        else:
            denom = ieee_sqrt(v) / c.k6 + c.k4
        p -= step_size * m / denom
        _st[pdt, ot, width](params, index, p)
        comptime if sdt == pdt:
            if has_grad_scale:
                _st[pdt, ot, width](grads, index, grad_to_store)
        _st[sdt, ot, width](state0, index, m)
        _st[sdt, ot, width](state1, index, v)
        if amsgrad:
            _st[sdt, ot, width](state2, index, mx)
        comptime if sdt != pdt:
            if has_grad_scale:
                _st[pdt, ot, width](grads, index, grad_to_store)
    elif algo == FUSED_SGD:
        var has_momentum = (flags & FO_MOMENTUM) != 0
        var first_step = (flags & FO_FIRST_STEP) != 0
        var buf = V(0)
        if has_momentum and not first_step:
            buf = _ld[sdt, ot, width](state0, index)
        if has_grad_scale:
            g = g / c.grad_scale
        var grad_to_store = g
        if maximize:
            g = -g
        if c.k1 != 0:
            g += c.k1 * p
        if has_momentum:
            if first_step:
                buf = g
            else:
                buf = c.k2 * buf + (1 - c.k3) * g
            if (flags & FO_NESTEROV) != 0:
                g = g + c.k2 * buf
            else:
                g = buf
        p -= c.k0 * g
        _st[pdt, ot, width](params, index, p)
        if has_grad_scale:
            _st[pdt, ot, width](grads, index, grad_to_store)
        if has_momentum:
            _st[sdt, ot, width](state0, index, buf)
    elif algo == FUSED_ADAGRAD:
        var state_sum = _ld[sdt, ot, width](state0, index)
        if has_grad_scale:
            g = (g.cast[aw]() / c.grad_scale.cast[aw]()).cast[ot]()
        var grad_to_store = g
        if maximize:
            g = -g
        if c.w1 != 0:
            g = (g.cast[aw]() + p.cast[aw]() * c.w1).cast[ot]()
        state_sum += g * g
        p = (
            p.cast[aw]()
            - c.w0 * g.cast[aw]() / (ieee_sqrt(state_sum).cast[aw]() + c.w2)
        ).cast[ot]()
        _st[pdt, ot, width](params, index, p)
        if has_grad_scale:
            _st[pdt, ot, width](grads, index, grad_to_store)
        _st[sdt, ot, width](state0, index, state_sum)
    else:
        comptime assert False, "no element math for this fused optimizer"


@always_inline
def _fused_opt_range[
    pdt: DType, sdt: DType, algo: Int
](
    params: Pointer[Scalar[pdt], MutAnyOrigin],
    grads: Pointer[Scalar[pdt], MutAnyOrigin],
    state0: Pointer[Scalar[sdt], MutAnyOrigin],
    state1: Pointer[Scalar[sdt], MutAnyOrigin],
    state2: Pointer[Scalar[sdt], MutAnyOrigin],
    step_ptr: Pointer[Float32, MutAnyOrigin],
    lr_ptr: Pointer[Float32, MutAnyOrigin],
    grad_scale_ptr: Pointer[Float32, MutAnyOrigin],
    has_lr_ptr: Bool,
    has_grad_scale: Bool,
    begin: Int,
    end: Int,
    h0: Scalar[FO_HYPER],
    h1: Scalar[FO_HYPER],
    h2: Scalar[FO_HYPER],
    h3: Scalar[FO_HYPER],
    h4: Scalar[FO_HYPER],
    flags: Int,
):
    """One block's element range [begin, end) of one tensor.

    Host hyperparameters `h*`, per algorithm:
      Adam/AdamW: lr, beta1, beta2, weight_decay, eps
      SGD:        lr, weight_decay, momentum, dampening
      Adagrad:    lr, lr_decay, weight_decay, eps
    """
    comptime ot = _opmath[pdt]()
    comptime aw = _wide[pdt]()
    var c = _Consts[ot, aw]()

    comptime if algo == FUSED_ADAM or algo == FUSED_ADAMW:
        # Bias corrections in opmath (`FusedAdamMathFunctor`), or in double
        # for the float32-param / bfloat16-state mode
        # (`FusedAdamMathFunctorMP`), then handed to adam_math as opmath.
        comptime bt = FO_HYPER if sdt != pdt else ot
        var step = step_ptr[unsafe_offset=0].cast[bt]()
        var bias1 = Scalar[bt](1) - pow_c99[bt](h1.cast[bt](), step)
        var bias2 = Scalar[bt](1) - pow_c99[bt](h2.cast[bt](), step)
        if has_lr_ptr:
            c.k0 = lr_ptr[unsafe_offset=0].cast[ot]()
        else:
            c.k0 = h0.cast[ot]()
        c.k1 = h1.cast[ot]()
        c.k2 = h2.cast[ot]()
        c.k3 = h3.cast[ot]()
        c.k4 = h4.cast[ot]()
        c.k5 = bias1.cast[ot]()
        c.k6 = ieee_sqrt(bias2).cast[ot]()
    elif algo == FUSED_SGD:
        if has_lr_ptr:
            c.k0 = lr_ptr[unsafe_offset=0].cast[ot]()
        else:
            c.k0 = h0.cast[ot]()
        c.k1 = h1.cast[ot]()
        c.k2 = h2.cast[ot]()
        c.k3 = h3.cast[ot]()
    elif algo == FUSED_ADAGRAD:
        # corrected_lr = lr / (1 + (step - 1) * lr_decay): `step - 1` in
        # float, the rest in double.
        var lr: Scalar[aw]
        if has_lr_ptr:
            lr = lr_ptr[unsafe_offset=0].cast[aw]()
        else:
            lr = h0.cast[aw]()
        var step = step_ptr[unsafe_offset=0]
        c.w0 = lr / (1 + (step - 1).cast[aw]() * h1.cast[aw]())
        c.w1 = h2.cast[aw]()
        c.w2 = h3.cast[aw]()
    if has_grad_scale:
        c.grad_scale = grad_scale_ptr[unsafe_offset=0].cast[ot]()

    var lane = Int(thread_idx.x)
    var index = begin + lane * _VEC
    while index + _VEC <= end:
        _fused_opt_elements[pdt, sdt, algo, _VEC](
            params,
            grads,
            state0,
            state1,
            state2,
            index,
            c,
            has_grad_scale,
            flags,
        )
        index += FUSED_OPT_THREADS * _VEC
    # Only a tensor's last chunk can have a scalar tail; it starts at the first
    # element past the vector region, so stores stay disjoint.
    index = begin + ((end - begin) // _VEC) * _VEC + lane
    while index < end:
        _fused_opt_elements[pdt, sdt, algo, 1](
            params,
            grads,
            state0,
            state1,
            state2,
            index,
            c,
            has_grad_scale,
            flags,
        )
        index += FUSED_OPT_THREADS


@always_inline
def _any_ptr[dtype: DType](addr: Int) -> Pointer[Scalar[dtype], MutAnyOrigin]:
    return Pointer[Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=addr
    ).as_unsafe_any_origin()


@__name(
    t"fused_{fused_opt_label[algo]()}_desc_{pdt}_{sdt}_t{FUSED_OPT_THREADS}"
)
def _fused_opt_desc_kernel[
    pdt: DType, sdt: DType, algo: Int
](
    descs: Array[FusedOptDesc, FUSED_OPT_DESC_CAP],
    desc_count_arg: Int64,
    chunk_elements_arg: Int64,
    lr_addr_arg: Int64,
    grad_scale_addr_arg: Int64,
    found_inf_addr_arg: Int64,
    h0: Scalar[FO_HYPER],
    h1: Scalar[FO_HYPER],
    h2: Scalar[FO_HYPER],
    h3: Scalar[FO_HYPER],
    h4: Scalar[FO_HYPER],
    flags_arg: Int64,
):
    """One block per chunk of the concatenation of the list."""
    # Int is not device-passable (host/device width mismatch); scalars cross
    # the launch ABI as Int64 and index math stays in Int.
    var found_inf_addr = Int(found_inf_addr_arg)
    # found_inf gates every write, the unscaled-gradient store included.
    if found_inf_addr != 0:
        if _any_ptr[DType.float32](found_inf_addr)[unsafe_offset=0] == 1.0:
            return
    var desc_count = Int(desc_count_arg)
    var chunk_elements = Int(chunk_elements_arg)
    var chunk = Int(block_idx.x)
    var desc_index = 0
    while desc_index + 1 < desc_count and chunk >= descs[desc_index].chunk_end:
        desc_index += 1
    var desc = descs[desc_index]
    var first_chunk = 0
    if desc_index != 0:
        first_chunk = descs[desc_index - 1].chunk_end
    var begin = (chunk - first_chunk) * chunk_elements
    var end = min(begin + chunk_elements, desc.numel)
    var lr_addr = Int(lr_addr_arg)
    var grad_scale_addr = Int(grad_scale_addr_arg)
    _fused_opt_range[pdt, sdt, algo](
        _any_ptr[pdt](desc.param_addr),
        _any_ptr[pdt](desc.grad_addr),
        _any_ptr[sdt](desc.state0_addr),
        _any_ptr[sdt](desc.state1_addr),
        _any_ptr[sdt](desc.state2_addr),
        _any_ptr[DType.float32](desc.step_addr),
        _any_ptr[DType.float32](lr_addr),
        _any_ptr[DType.float32](grad_scale_addr),
        lr_addr != 0,
        grad_scale_addr != 0,
        begin,
        end,
        h0,
        h1,
        h2,
        h3,
        h4,
        Int(flags_arg),
    )


@__name(t"fused_{fused_opt_label[algo]()}_tensor_apple_{pdt}_{sdt}")
def _fused_opt_apple_kernel[
    pdt: DType, sdt: DType, algo: Int
](
    params: Pointer[Scalar[pdt], MutAnyOrigin],
    grads: Pointer[Scalar[pdt], MutAnyOrigin],
    state0: Pointer[Scalar[sdt], MutAnyOrigin],
    state1: Pointer[Scalar[sdt], MutAnyOrigin],
    state2: Pointer[Scalar[sdt], MutAnyOrigin],
    step_ptr: Pointer[Float32, MutAnyOrigin],
    lr_ptr: Pointer[Float32, MutAnyOrigin],
    grad_scale_ptr: Pointer[Float32, MutAnyOrigin],
    found_inf_ptr: Pointer[Float32, MutAnyOrigin],
    numel_arg: Int64,
    chunk_elements_arg: Int64,
    has_lr_arg: Int64,
    has_grad_scale_arg: Int64,
    has_found_inf_arg: Int64,
    h0: Scalar[FO_HYPER],
    h1: Scalar[FO_HYPER],
    h2: Scalar[FO_HYPER],
    h3: Scalar[FO_HYPER],
    h4: Scalar[FO_HYPER],
    flags_arg: Int64,
):
    """Apple/Metal: one tensor per launch, every pointer a real argument
    (an optional one is a valid dummy plus a has_* flag, never null)."""
    if has_found_inf_arg != 0:
        if found_inf_ptr[unsafe_offset=0] == 1.0:
            return
    var chunk_elements = Int(chunk_elements_arg)
    var begin = Int(block_idx.x) * chunk_elements
    var end = min(begin + chunk_elements, Int(numel_arg))
    _fused_opt_range[pdt, sdt, algo](
        params,
        grads,
        state0,
        state1,
        state2,
        step_ptr,
        lr_ptr,
        grad_scale_ptr,
        has_lr_arg != 0,
        has_grad_scale_arg != 0,
        begin,
        end,
        h0,
        h1,
        h2,
        h3,
        h4,
        Int(flags_arg),
    )


@always_inline
def _or_dummy(addr: Int, dummy: Int) -> Int:
    return addr if addr != 0 else dummy


def enqueue_fused_optimizer[
    pdt: DType, sdt: DType, algo: Int
](
    descs: Array[FusedOptDesc, FUSED_OPT_DESC_CAP],
    desc_count: Int,
    total_chunks: Int,
    chunk_elements: Int,
    lr_addr: Int,
    grad_scale_addr: Int,
    found_inf_addr: Int,
    hyper: Array[Float64, 5],
    flags: Int,
    ctx: DeviceContext,
) raises:
    if desc_count <= 0 or total_chunks <= 0:
        return
    # Rounded to the kernel's hyperparameter type here, on the host: no
    # float64 value may reach a Metal kernel.
    var h0 = Scalar[FO_HYPER](hyper[0])
    var h1 = Scalar[FO_HYPER](hyper[1])
    var h2 = Scalar[FO_HYPER](hyper[2])
    var h3 = Scalar[FO_HYPER](hyper[3])
    var h4 = Scalar[FO_HYPER](hyper[4])
    comptime if has_apple_gpu_accelerator():
        comptime if pdt == DType.float64 or sdt == DType.float64:
            raise Error("fused optimizers: Apple GPUs have no float64")
        else:
            var first_chunk = 0
            for desc_index in range(desc_count):
                var desc = descs[desc_index]
                var chunk_count = desc.chunk_end - first_chunk
                first_chunk = desc.chunk_end
                if chunk_count <= 0:
                    continue
                var dummy = desc.param_addr
                _enqueue_cached[_fused_opt_apple_kernel[pdt, sdt, algo]](
                    ctx,
                    chunk_count,
                    1,
                    1,
                    FUSED_OPT_THREADS,
                    _any_ptr[pdt](desc.param_addr),
                    _any_ptr[pdt](desc.grad_addr),
                    _any_ptr[sdt](_or_dummy(desc.state0_addr, dummy)),
                    _any_ptr[sdt](_or_dummy(desc.state1_addr, dummy)),
                    _any_ptr[sdt](_or_dummy(desc.state2_addr, dummy)),
                    _any_ptr[DType.float32](_or_dummy(desc.step_addr, dummy)),
                    _any_ptr[DType.float32](_or_dummy(lr_addr, dummy)),
                    _any_ptr[DType.float32](_or_dummy(grad_scale_addr, dummy)),
                    _any_ptr[DType.float32](_or_dummy(found_inf_addr, dummy)),
                    Int64(desc.numel),
                    Int64(chunk_elements),
                    Int64(1 if lr_addr != 0 else 0),
                    Int64(1 if grad_scale_addr != 0 else 0),
                    Int64(1 if found_inf_addr != 0 else 0),
                    h0,
                    h1,
                    h2,
                    h3,
                    h4,
                    Int64(flags),
                )
    else:
        _enqueue_cached[_fused_opt_desc_kernel[pdt, sdt, algo]](
            ctx,
            total_chunks,
            1,
            1,
            FUSED_OPT_THREADS,
            descs,
            Int64(desc_count),
            Int64(chunk_elements),
            Int64(lr_addr),
            Int64(grad_scale_addr),
            Int64(found_inf_addr),
            h0,
            h1,
            h2,
            h3,
            h4,
            Int64(flags),
        )
