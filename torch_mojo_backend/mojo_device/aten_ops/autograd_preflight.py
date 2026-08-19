"""Ops that preflight their native autograd node from the forward.

Each op here has a fast forward but records a backward node the autograd
engine may not be able to run. A raise from inside the engine aborts the
process on this backend instead of propagating (the engine restores streams
through PyTorch's noexcept Python device guard, which std::terminates when the
Python round-trip fails during unwind), so the refusal has to happen in the
forward, where it still has a traceback naming the op. Each function's
docstring spells out its own case.
"""

from collections.abc import Callable, Sequence

import torch

from torch_mojo_backend.mojo_device.torch_mojo_tensor import TorchMojoTensor

from .support import _eager_impl, _fast, _refuse_unsupported_backward, _unsupported

# Most of these ops have no autograd escape other than turning grad off: their
# parameters legitimately require grad during training, so unlike batch norm
# (whose `training=False` forward records a node nobody runs) there is no mode
# that keeps training working. `.eval()` specifically does NOT help — it leaves
# grad mode on and the parameters still require grad — and saying otherwise has
# sent users chasing a workaround that cannot exist.
_GRAD_OFF_ONLY = (
    "The forward itself is supported: run it under torch.no_grad() or "
    "torch.inference_mode(). Module .eval() alone does not help here, because "
    "it leaves grad mode enabled and the parameters still require grad; "
    "training through this op needs the backward kernel itself."
)


def _preflight_unsupported_backward(
    op_name: str,
    fast_name: str,
    backward_op: str,
    grad_operands: Sequence[int],
    workaround: str = _GRAD_OFF_ONLY,
) -> Callable[..., TorchMojoTensor | tuple[TorchMojoTensor, ...]]:
    """`aten_fast.<fast_name>`, fronted by the forward-time autograd refusal.

    One entry per op whose whole native backward is missing, which is the
    common case and needs no reasoning beyond "which operands' gradients would
    require it": `grad_operands` are positions in the ATen schema, so an op
    whose backward only needs the missing kernel for one operand (scatter's
    `src`, say) refuses only that. Ops needing more than a position list get a
    hand-written function further down.
    """
    dispatch = _eager_impl(fast_name, op_name)

    def preflighted(
        *args: object, **kwargs: object
    ) -> TorchMojoTensor | tuple[TorchMojoTensor, ...]:
        operands = [
            args[index]
            for index in grad_operands
            if index < len(args) and isinstance(args[index], torch.Tensor)
        ]
        if len(operands) < len(grad_operands):
            # The PrivateUse1 boxed kernel passes every positional schema
            # argument positionally, so this is unreachable today. If some call
            # shape ever moves one into kwargs, checking every keyword tensor
            # over-refuses at worst; skipping the check aborts the process.
            operands += [
                value for value in kwargs.values() if isinstance(value, torch.Tensor)
            ]
        _refuse_unsupported_backward(op_name, backward_op, tuple(operands), workaround)
        return dispatch(*args, **kwargs)

    preflighted.__name__ = "mojo_device_" + op_name.removeprefix("aten::").replace(
        ".", "_"
    )
    preflighted.__qualname__ = preflighted.__name__
    return preflighted


# ---------------------------------------------------------------------------
# Ops whose entire native backward is unsupported: one table entry each.
# Alphabetical by aten name. `backward_op` names what the engine would actually
# have failed on, which for a few of these is an ordinary forward op the
# generated backward composes (cumsum needs `flip`, scatter needs `gather`,
# advanced indexing needs `_index_put_impl_`) rather than a `*_backward` op.
# ---------------------------------------------------------------------------

mojo_device__adaptive_avg_pool2d = _preflight_unsupported_backward(
    "aten::_adaptive_avg_pool2d",
    "fast_aten__adaptive_avg_pool2d",
    "aten::_adaptive_avg_pool2d_backward",
    grad_operands=(0,),
)

mojo_device__scaled_dot_product_efficient_attention = _preflight_unsupported_backward(
    "aten::_scaled_dot_product_efficient_attention",
    "fast_aten__scaled_dot_product_efficient_attention",
    "aten::_scaled_dot_product_efficient_attention_backward",
    grad_operands=(0, 1, 2, 3),
)

mojo_device__softmax = _preflight_unsupported_backward(
    "aten::_softmax",
    "fast_aten__softmax",
    "aten::_softmax_backward_data",
    grad_operands=(0,),
)

mojo_device_avg_pool2d = _preflight_unsupported_backward(
    "aten::avg_pool2d",
    "fast_aten_avg_pool2d",
    "aten::avg_pool2d_backward",
    grad_operands=(0,),
)

mojo_device_cumsum = _preflight_unsupported_backward(
    "aten::cumsum", "fast_aten_cumsum", "aten::flip", grad_operands=(0,)
)

mojo_device_index = _preflight_unsupported_backward(
    "aten::index.Tensor",
    "fast_aten_index",
    "aten::_index_put_impl_",
    grad_operands=(0,),
)

mojo_device_max_pool2d_with_indices = _preflight_unsupported_backward(
    "aten::max_pool2d_with_indices",
    "fast_aten_max_pool2d_with_indices",
    "aten::max_pool2d_with_indices_backward",
    grad_operands=(0,),
)

mojo_device_relu = _preflight_unsupported_backward(
    "aten::relu", "fast_aten_relu", "aten::threshold_backward", grad_operands=(0,)
)

# Only the `src` gradient goes through `gather`; the `self` gradient is a
# scatter of zeros, which runs, so a self-only call must not be refused.
mojo_device_scatter_src = _preflight_unsupported_backward(
    "aten::scatter.src", "fast_aten_scatter_src", "aten::gather", grad_operands=(3,)
)

mojo_device_sigmoid = _preflight_unsupported_backward(
    "aten::sigmoid", "fast_aten_sigmoid", "aten::sigmoid_backward", grad_operands=(0,)
)

mojo_device_tanh = _preflight_unsupported_backward(
    "aten::tanh", "fast_aten_tanh", "aten::tanh_backward", grad_operands=(0,)
)

mojo_device_upsample_bilinear2d = _preflight_unsupported_backward(
    "aten::upsample_bilinear2d",
    "fast_aten_upsample_bilinear2d",
    "aten::upsample_bilinear2d_backward",
    grad_operands=(0,),
)

# Nearest-neighbor upsample has no interpolation weights, but its backward is
# still a scatter-add: each input-gradient pixel sums the contributions of
# every output pixel that nearest-mapped to it. No such reduction kernel
# exists here yet, so refuse the same way bilinear's backward is refused.
mojo_device_upsample_nearest2d = _preflight_unsupported_backward(
    "aten::upsample_nearest2d",
    "fast_aten_upsample_nearest2d",
    "aten::upsample_nearest2d_backward",
    grad_operands=(0,),
)


# ---------------------------------------------------------------------------
# Ops whose preflight needs op-specific reasoning. Alphabetical.
# ---------------------------------------------------------------------------


def mojo_device_convolution(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride: Sequence[int],
    padding: Sequence[int],
    dilation: Sequence[int],
    transposed: bool,
    output_padding: Sequence[int],
    groups: int,
) -> TorchMojoTensor:
    """Convolution with a forward-time preflight of its native autograd node.

    The forward exists here; `aten::convolution_backward` does not (it needs
    the im2col-transpose GEMM pair). Recording ConvolutionBackward0 anyway
    would move the failure into the autograd engine, where an exception aborts
    the process instead of raising — the unwind hazard
    `_refuse_unsupported_backward` documents — so a call whose gradient the
    engine would come back for is refused here, from the forward, with a
    traceback that names the op.

    Unlike batch norm this has no training-mode escape: a conv weight normally
    requires grad, so in practice this refuses every conv in a training model,
    and the honest message is that the backward kernel is missing rather than
    that some flag is set wrong. Inference is unaffected under
    `torch.no_grad()` / `torch.inference_mode()`.
    """
    _refuse_unsupported_backward(
        "aten::convolution",
        "aten::convolution_backward",
        (input, weight, bias),
        _GRAD_OFF_ONLY,
    )
    aten_fast = _fast()
    result = aten_fast.fast_aten_convolution(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
    )
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::convolution", (input, weight, bias))
    return result


def mojo_device_embedding(
    weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False
):
    """Native-autograd embedding with a forward-time autograd-mode preflight.

    An exception raised while the autograd engine runs a backward node aborts
    the process on this backend: the engine's stream guard restores streams
    through PyTorch's noexcept Python device guard, which std::terminates
    when the Python round-trip fails during unwind. Unsupported autograd
    modes must therefore be rejected before EmbeddingBackward0 is recorded,
    not when the engine reaches the sparse or dense backward.
    """
    aten_fast = _fast()
    if torch.is_grad_enabled() and weight.requires_grad:
        if sparse or scale_grad_by_freq:
            mode = "sparse=True" if sparse else "scale_grad_by_freq=True"
            raise NotImplementedError(
                f"Mojo eager embedding autograd does not yet support {mode}"
            )
        # The recorded backward's atomic accumulation is nondeterministic;
        # alerting there is too late for the same unwind-abort reason.
        aten_fast._alert_not_deterministic("embedding_dense_backward on Mojo")
    result = aten_fast.fast_aten_embedding(
        weight, indices, padding_idx, scale_grad_by_freq, sparse
    )
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::embedding", (weight, indices))
    return result


def _linear_backward_unsupported_reason(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    output_mask: tuple[bool, bool, bool],
) -> str | None:
    """Why `aten::linear_backward` will decline these operands, or None.

    The verdict has to come from the kernel layer rather than from a copy of
    its conditions here, because those conditions move: the fp32 weight-
    gradient GEMM only recently learned to materialize its transposed operand,
    and a stale copy would reject calls that now run. So we ask `aten_fast`
    for a predicate, and until it grows one (see the follow-ups) we fall back
    to reading its own dtype constant and applying only the *necessary*
    conditions from `fast_aten_linear_backward`'s entry gate.

    That fallback is deliberately incomplete: it names regimes the kernel
    provably cannot take, and stays silent about the GEMM-path questions it
    cannot answer without launching one. Under-reporting leaves the pre-
    existing abort in place for regimes it misses; over-reporting would break
    working models, so every uncertainty resolves to None.
    """
    aten_fast = _fast()
    predicate = getattr(aten_fast, "fast_aten_linear_backward_supported", None)
    if predicate is not None:
        try:
            if predicate(input, weight, output_mask):
                return None
        except Exception:
            return None
        return (
            "the eager kernel layer reports no path for this dtype and layout "
            "combination"
        )

    # `fast_aten_linear_backward` tests every operand dtype against this tuple
    # before it does anything else, so reading the tuple (rather than
    # restating its contents) keeps the check current if the kernels widen.
    supported = getattr(aten_fast, "_FLOAT_DTYPES", None)
    if supported is None:
        return None
    try:
        if input._dtype not in supported:
            covered = ", ".join(
                str(dtype).removeprefix("DType.") for dtype in supported
            )
            return (
                f"linear_backward covers {covered} only, and these operands "
                f"are {input.dtype}"
            )
        if weight._dtype != input._dtype or (
            bias is not None and bias._dtype != input._dtype
        ):
            return (
                "linear_backward requires input, weight and bias to share one "
                f"dtype, and these are {input.dtype}/{weight.dtype}"
                + ("" if bias is None else f"/{bias.dtype}")
            )
    except AttributeError:
        # A non-mojo operand: the forward below reports that far better.
        return None
    return None


def mojo_device_linear(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> TorchMojoTensor:
    """Linear with a forward-time preflight of its native autograd node.

    Registering `aten::linear` keeps `nn.Linear` from decomposing to addmm, so
    its backward is the fused `aten::linear_backward` node. That node runs
    inside the autograd engine, where an exception aborts the process on this
    backend: the engine's stream guard restores streams through PyTorch's
    noexcept Python device guard, which std::terminates when the Python
    round-trip fails during unwind — "terminate called without an active
    exception", no traceback, no frame naming the op. A backward that cannot
    run has to be reported before LinearBackward is recorded, exactly as
    `mojo_device_embedding` above rejects its unsupported autograd modes.

    `requires_grad` is the engine's `task_should_compute_output` mask for an
    ordinary `loss.backward()`; a partial `torch.autograd.grad(inputs=...)`
    can narrow it further, so the preflight is asked about the widest mask the
    engine could request.
    """
    if torch.is_grad_enabled() and (
        input.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    ):
        reason = _linear_backward_unsupported_reason(
            input,
            weight,
            bias,
            (
                bool(input.requires_grad),
                bool(weight.requires_grad),
                bias is not None and bool(bias.requires_grad),
            ),
        )
        if reason is not None:
            raise NotImplementedError(
                "aten::linear would record an autograd node "
                "(aten::linear_backward) that mojo eager mode cannot run: "
                f"{reason} (input {tuple(input.shape)} {input.dtype}, weight "
                f"{tuple(weight.shape)} {weight.dtype}, device "
                f"{input.device}). linear_backward forms the weight gradient "
                "as mm(grad_output.transpose(0, 1), input) — a GEMM whose "
                "left operand is a non-contiguous transposed view — so not "
                "every dtype and layout has a path. Workarounds: run the "
                "layer in bfloat16, float16 or float32; for float32, "
                'torch.set_float32_matmul_precision("high") additionally '
                "opens the TF32 GEMM, which takes arbitrary 2-D layouts. "
                "Raised from the forward on purpose: raised from the backward "
                "node instead, this aborts the process without a traceback."
            )
    aten_fast = _fast()
    result = aten_fast.fast_aten_linear(input, weight, bias)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::linear", (input, weight, bias))
    return result


def mojo_device_native_batch_norm(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    running_mean: torch.Tensor | None,
    running_var: torch.Tensor | None,
    training: bool,
    momentum: float,
    eps: float,
) -> tuple[TorchMojoTensor, TorchMojoTensor, TorchMojoTensor]:
    """Batch norm with a forward-time preflight of its native autograd node.

    The training forward exists here; `aten::native_batch_norm_backward` does
    not (docs/optimization_backlog.md N2). Letting the forward record
    NativeBatchNormBackward anyway would move the failure into the autograd
    engine, where an exception aborts the process on this backend rather than
    raising — the same unwind hazard `mojo_device_linear` above documents — so
    a training call that would need a gradient is refused here, from the
    forward, with a traceback that names the op.

    Inference needs no preflight: `training=False` records a backward this
    backend never has to run.
    """
    if (
        training
        and torch.is_grad_enabled()
        and (
            input.requires_grad
            or (weight is not None and weight.requires_grad)
            or (bias is not None and bias.requires_grad)
        )
    ):
        raise NotImplementedError(
            "aten::native_batch_norm (training=True) would record an autograd "
            "node (aten::native_batch_norm_backward) that mojo eager mode "
            f"does not implement (input {tuple(input.shape)} {input.dtype}, "
            f"device {input.device}). The forward itself is supported: run it "
            "under torch.no_grad(), or put the module in eval() mode. Raised "
            "from the forward on purpose: raised from the backward node "
            "instead, this aborts the process without a traceback."
        )
    aten_fast = _fast()
    result = aten_fast.fast_aten_native_batch_norm(
        input, weight, bias, running_mean, running_var, training, momentum, eps
    )
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported(
            "aten::native_batch_norm", (input, weight, bias, running_mean, running_var)
        )
    return result


def mojo_device_native_group_norm(
    input: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    N: int,
    C: int,
    HxW: int,
    group: int,
    eps: float,
) -> tuple[TorchMojoTensor, TorchMojoTensor, TorchMojoTensor]:
    """Group norm with a forward-time preflight of its native autograd node.

    The forward exists here; `aten::native_group_norm_backward` does not (it
    needs the per-group reduction statistics). Recording
    NativeGroupNormBackward0 anyway would move the failure into the autograd
    engine, where an exception aborts the process instead of raising, so the
    call is refused from the forward as `mojo_device_native_batch_norm` above
    refuses its training case.

    Group norm has no `training` flag to key on, so unlike batch norm every
    grad-requiring call is refused; inference is unaffected under
    `torch.no_grad()` / `torch.inference_mode()`.
    """
    _refuse_unsupported_backward(
        "aten::native_group_norm",
        "aten::native_group_norm_backward",
        (input, weight, bias),
        _GRAD_OFF_ONLY,
    )
    aten_fast = _fast()
    result = aten_fast.fast_aten_native_group_norm(
        input, weight, bias, N, C, HxW, group, eps
    )
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::native_group_norm", (input, weight, bias))
    return result


def mojo_device_softmax(
    self: torch.Tensor, dim: int, dtype: torch.dtype | None = None
) -> TorchMojoTensor:
    """`aten::softmax.int`, refused when a gradient would silently go missing.

    This one fails differently from its neighbours, and worse. `softmax.int` is
    CompositeImplicitAutograd upstream, so registering a PrivateUse1 kernel for
    it (worth it: the kernel is a fused softmax) takes it out of reach of the
    decomposition autograd would otherwise differentiate. Autograd then falls
    back to `autograd_not_implemented_fallback`, which does not abort and does
    not raise — it warns and produces **no gradient at all**, leaving
    `x.grad is None` after `backward()` and training silently stuck.

    `aten::_softmax_backward_data` has no kernel here either, so there is no
    gradient to be had by any route; refusing beats returning a model that
    trains on zeros. Raised from the forward, same as the aborting ops around
    it, so the traceback names the op.
    """
    if torch.is_grad_enabled() and self.requires_grad:
        raise NotImplementedError(
            "aten::softmax.int has no gradient in mojo eager mode, and unlike "
            "the ops around it would not have failed: autograd cannot see "
            "through the fused kernel registered for this op, so backward() "
            "would leave .grad as None and train on nothing "
            f"(input {tuple(self.shape)} {self.dtype} on {self.device}). "
            "aten::_softmax_backward_data has no kernel here either, so no "
            "route produces one. " + _GRAD_OFF_ONLY
        )
    aten_fast = _fast()
    result = aten_fast.fast_aten_softmax(self, dim, dtype)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::softmax.int", (self,))
    return result
