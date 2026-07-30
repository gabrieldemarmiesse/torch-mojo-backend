"""AutogradPrivateUse1 plumbing for eager Mojo operations.

``TorchMojoTensor`` is a true wrapper subclass, so PyTorch's generated autograd
can be used whenever it exposes a suitable native backward ATen operation. The
custom Functions left here cover fused formulas or unsupported native routes;
their sidecar metadata also validates saved-tensor hook results and restores
values that hooks deliberately unpack onto the host.
"""

import math
from typing import TypeVar

import torch

from .torch_mojo_tensor import TorchMojoTensor

# _require_handled passes its argument through unchanged, including the
# symbolic stand-ins host-contract tests route through it.
_ResultT = TypeVar("_ResultT")

_registered = False


def _fast():
    from torch_mojo_backend.eager_kernels import aten_fast

    return aten_fast


def _require_handled(result: _ResultT | None, operation: str) -> _ResultT:
    aten_fast = _fast()
    if result is None or result is aten_fast.NOT_HANDLED:
        raise NotImplementedError(
            f"{operation} is not supported by Mojo eager autograd for these inputs"
        )
    return result


def _contiguous_view(tensor: TorchMojoTensor, shape) -> TorchMojoTensor:
    tensor = tensor._contig()
    return _require_handled(_fast().fast_aten_view(tensor, tuple(shape)), "view")


class _SavedMojoPayload:
    """Metadata used to validate or restore a PyTorch-unpacked saved variable.

    The ordinary wrapper-subclass path returns a fresh, complete
    ``TorchMojoTensor`` with the shared version counter and allocation payload.
    The recorded fields validate that object without retaining a Tensor
    directly on ``ctx``.

    Ordinarily the payload must retain the Mojo holder because TensorImpl does
    not own that out-of-tree allocation. With active saved-tensor hooks, the
    hook's packed object owns the saved value instead. In that case retaining
    the original holder here would defeat CPU/offload hooks, so ``holder`` is
    deliberately ``None`` and restore must use the unpack hook's value.
    """

    __slots__ = (
        "holder",
        "ptr",
        "shape",
        "strides",
        "offset",
        "dtype",
        "itemsize",
        "numel",
        "device",
        "torch_device",
        "is_contiguous",
    )

    def __init__(self, tensor: TorchMojoTensor):
        self.holder = None if _saved_tensor_hooks_active() else tensor._holder
        self.ptr = tensor._ptr
        self.shape = tensor._shape
        self.strides = tensor._strides
        self.offset = tensor._offset
        self.dtype = tensor._dtype
        self.itemsize = tensor._itemsize
        self.numel = tensor._numel
        self.device = tensor._device
        self.torch_device = tensor._torch_device
        self.is_contiguous = tensor._is_contiguous

    def restore(self, tensor: torch.Tensor) -> TorchMojoTensor:
        if isinstance(tensor, TorchMojoTensor) and hasattr(tensor, "_holder"):
            return self._validate_hook_result(tensor)

        # A saved-tensor hook may unpack to host memory. Honor the unpacked
        # value by moving it back rather than silently reading the original
        # Mojo allocation retained for the ordinary SavedVariable path.
        if tensor.device.type != "mojo":
            try:
                restored = tensor.to(self.torch_device)
            except Exception as exc:
                raise RuntimeError(
                    "saved-tensor hook result could not be restored to the "
                    f"original Mojo device {self.torch_device}"
                ) from exc
            return self._validate_hook_result(restored)

        if self.holder is None:
            raise RuntimeError(
                "saved-tensor hook returned an unusable Mojo tensor without "
                "a TorchMojoTensor allocation holder; its unpack hook must "
                "return a complete TorchMojoTensor or a host tensor"
            )

        # Compatibility path for a PyTorch version that unpacks a plain
        # PrivateUse1 tensor despite the wrapper subclass. SavedVariable has
        # already checked the original shared version counter at this point;
        # construct a valid wrapper instead of attempting an invalid class swap.
        return TorchMojoTensor._make(
            self.holder,
            self.ptr,
            self.shape,
            self.strides,
            self.offset,
            self.dtype,
            self.device,
            contiguous=self.is_contiguous,
        )

    def _validate_hook_result(self, tensor: torch.Tensor) -> TorchMojoTensor:
        required = (
            "_holder",
            "_ptr",
            "_shape",
            "_strides",
            "_offset",
            "_dtype",
            "_itemsize",
            "_numel",
            "_device",
            "_torch_device",
            "_is_contiguous",
        )
        if not isinstance(tensor, TorchMojoTensor) or any(
            not hasattr(tensor, field) for field in required
        ):
            raise RuntimeError(
                "saved-tensor hook did not restore a complete TorchMojoTensor"
            )
        if (
            tuple(tensor._shape) != tuple(self.shape)
            or tensor._dtype != self.dtype
            or tensor._device != self.device
        ):
            raise RuntimeError(
                "saved-tensor hook restored incompatible Mojo tensor metadata: "
                f"expected shape={tuple(self.shape)}, dtype={self.dtype}, "
                f"device={self.torch_device}"
            )
        return tensor


def _saved_tensor_hooks_active() -> bool:
    """Best-effort active-hook detection across supported PyTorch versions.

    PyTorch does not expose this as public API. Newer versions accept a bool
    argument on ``_top_saved_tensors_default_hooks``; older variants accepted
    no argument. If neither private form is available, retaining the holder is
    the conservative compatibility fallback: correctness is preserved, only
    hook-driven allocation offload is less effective on that PyTorch version.
    """
    top_hooks = getattr(
        getattr(torch._C, "_autograd", None), "_top_saved_tensors_default_hooks", None
    )
    if top_hooks is None:
        return False
    try:
        return top_hooks(False) is not None
    except TypeError:
        try:
            return top_hooks() is not None
        except Exception:
            return False
    except Exception:
        return False


def _restore_saved_mojo_tensors(ctx):
    saved_tensors = ctx.saved_tensors
    payloads = ctx.saved_payloads
    if len(saved_tensors) != len(payloads):
        raise RuntimeError("Mojo autograd saved-tensor payload count mismatch")
    return tuple(
        payload.restore(tensor)
        for tensor, payload in zip(saved_tensors, payloads, strict=True)
    )


class _FusedFlashAttentionAutograd(torch.autograd.Function):
    """gfx942 fused flash attention, forward and backward in one kernel each.

    The generic node below saves the whole probability matrix; this one saves Q,
    K, V, the output and the per-row log-sum-exp, and the backward recomputes
    the scores.  At nanoGPT 124M / batch 48 / block 1024 that is 6.100 ms/step
    forward against the decomposition's 29.854, and 28.896 against 46.153.
    """

    @staticmethod
    def forward(
        ctx, query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    ):
        aten_fast = _fast()
        result = aten_fast.fast_fused_flash_attention_forward(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
        if result is aten_fast.NOT_HANDLED:
            raise NotImplementedError(
                "fused gfx942 flash attention declined these inputs"
            )
        output, lse, q_used, k_used, v_used = result
        # Save what the KERNEL read, not the caller's views: a non-contiguous
        # input was copied on the way in, and the backward must recompute the
        # scores from the same bytes the forward saw.
        saved = (q_used, k_used, v_used, output, lse)
        ctx.save_for_backward(*saved)
        ctx.saved_payloads = tuple(_SavedMojoPayload(tensor) for tensor in saved)
        # Same introspection surface as the decomposition's node, so saved-tensor
        # hook tests and debugging tools can name what this one holds. The set
        # differs on purpose: O(n*d) inputs plus the log-sum-exp, rather than the
        # O(n^2) probability matrix.
        ctx.saved_names = ("query", "key", "value", "output", "logsumexp")
        ctx.is_causal = bool(is_causal)
        ctx.scale = scale
        ctx.needed_input_gradients = tuple(
            bool(ctx.needs_input_grad[index]) for index in range(3)
        )
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return (None,) * 8
        aten_fast = _fast()
        q, k, v, output, lse = _restore_saved_mojo_tensors(ctx)
        result = aten_fast.fast_fused_flash_attention_backward(
            grad_output, q, k, v, output, lse, ctx.is_causal, ctx.scale
        )
        if result is aten_fast.NOT_HANDLED:
            raise RuntimeError(
                "fused gfx942 flash attention backward declined inputs its "
                "own forward accepted"
            )
        grad_query, grad_key, grad_value = result
        need_query, need_key, need_value = ctx.needed_input_gradients
        return (
            grad_query if need_query else None,
            grad_key if need_key else None,
            grad_value if need_value else None,
            None,
            None,
            None,
            None,
            None,
        )


# Eligible FA4 calls are routed through PyTorch's native lower flash pair below.
# This custom node remains only for the generic math/dropout implementation,
# whose fused intermediate-saving backward has no native ATen schema.
class _ScaledDotProductAttentionAutograd(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    ):
        if attn_mask is not None or enable_gqa:
            raise NotImplementedError(
                "Mojo eager SDPA autograd currently supports no attention mask, "
                "and enable_gqa=False"
            )

        aten_fast = _fast()
        result = aten_fast._sdpa_math_forward_with_dropout(
            query, key, value, is_causal, scale, dropout_p
        )
        if result is aten_fast.NOT_HANDLED:
            raise NotImplementedError(
                "aten::scaled_dot_product_attention is not supported by Mojo "
                "eager autograd for these inputs"
            )
        output, probabilities, dropout_mask = result
        need_query, need_key, need_value = (
            bool(ctx.needs_input_grad[index]) for index in range(3)
        )
        saved = []
        saved_names = []

        def save(name, tensor):
            saved_names.append(name)
            saved.append(tensor)

        # dQ consumes K and V; dK consumes Q and V; dV consumes only P_drop.
        # Every requested input gradient needs the pre-dropout probabilities.
        if need_key:
            save("query", query)
        if need_query:
            save("key", key)
        if need_query or need_key:
            save("value", value)
        save("probabilities", probabilities)
        if dropout_mask is not None:
            save("dropout_mask", dropout_mask)
        ctx.save_for_backward(*saved)
        ctx.saved_payloads = tuple(_SavedMojoPayload(tensor) for tensor in saved)
        ctx.saved_names = tuple(saved_names)
        ctx.needed_input_gradients = (need_query, need_key, need_value)
        ctx.query_shape = tuple(query._shape)
        ctx.key_shape = tuple(key._shape)
        ctx.value_shape = tuple(value._shape)
        ctx.scale = (
            float(scale) if scale is not None else 1.0 / math.sqrt(query._shape[-1])
        )
        ctx.is_causal = bool(is_causal)
        ctx.has_dropout = dropout_mask is not None
        ctx.is_causal = bool(is_causal)
        ctx.dropout_scale = (
            (0.0 if float(dropout_p) == 1.0 else 1.0 / (1.0 - float(dropout_p)))
            if ctx.has_dropout
            else 1.0
        )
        ctx.set_materialize_grads(False)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        aten_fast = _fast()
        restored = _restore_saved_mojo_tensors(ctx)
        saved = dict(zip(ctx.saved_names, restored, strict=True))
        del restored
        need_query, need_key, need_value = ctx.needed_input_gradients
        probabilities = saved.pop("probabilities")
        dropout_mask = saved.pop("dropout_mask", None)
        batch, heads, query_length, head_dim = ctx.query_shape
        key_length = ctx.key_shape[2]
        batch_heads = batch * heads

        p3 = _contiguous_view(probabilities, (batch_heads, query_length, key_length))
        grad3 = _contiguous_view(grad_output, (batch_heads, query_length, head_dim))
        del probabilities

        mask3 = None
        if ctx.has_dropout:
            mask3 = _contiguous_view(
                dropout_mask, (batch_heads, query_length, key_length)
            )
            del dropout_mask

        grad_query3 = None
        grad_key3 = None
        grad_value3 = None

        # The two contraction-side causal regimes skip contraction indices that
        # multiply exactly zero, so they are exact whatever else runs; nothing
        # here depends on the fused softmax backward being selected. The
        # output-side regime (which would leave dP's masked half unwritten) is
        # deliberately not used: it is dispatch-bound at these shapes and
        # measures slower than the dense GEMM. See optimization_journal.md,
        # diagnostic experiment AA.
        causal_bmm = bool(getattr(ctx, "is_causal", False))

        # Fused Apple-GPU route: at most five launches, no permute copies and
        # no dropout-backward pass (causal structure exploited end to end).
        # Unsupported inputs fall through to the exact composed sequence.
        fused = aten_fast.NOT_HANDLED
        if need_query or need_key or need_value:
            q3 = (
                _contiguous_view(saved["query"], (batch_heads, query_length, head_dim))
                if "query" in saved
                else None
            )
            k3 = (
                _contiguous_view(saved["key"], (batch_heads, key_length, head_dim))
                if "key" in saved
                else None
            )
            v3 = (
                _contiguous_view(saved["value"], (batch_heads, key_length, head_dim))
                if "value" in saved
                else None
            )
            fused = aten_fast.fast_sdpa_backward(
                p3,
                grad3,
                mask3,
                q3,
                k3,
                v3,
                need_query,
                need_key,
                need_value,
                getattr(ctx, "is_causal", False),
                ctx.dropout_scale,
                ctx.scale,
            )
            del q3, k3, v3
        if fused is not aten_fast.NOT_HANDLED:
            grad_query3, grad_key3, grad_value3 = fused
            saved.clear()
        elif need_value:
            effective_p3 = p3
            if ctx.has_dropout:
                effective_p3 = _require_handled(
                    aten_fast.fast_aten_native_dropout_backward(
                        p3, mask3, ctx.dropout_scale
                    ),
                    "SDPA reconstruct dropped probabilities",
                )

            # dV = P_drop^T @ grad = (grad^T @ P_drop)^T. Materializing
            # grad^T costs O(L*E), rather than materializing P_drop^T at
            # O(L*S). The final transpose copy is the required contiguous dV.
            grad_t = _require_handled(
                aten_fast.fast_aten_transpose(grad3, 1, 2),
                "SDPA transpose output gradient",
            )
            grad_t = _contiguous_view(grad_t, (batch_heads, head_dim, query_length))
            grad_value_t = None
            if causal_bmm:
                # P is exactly zero above the diagonal, so output column block c
                # only needs contraction indices from c's first column up.
                grad_value_t = aten_fast._try_sdpa_causal_bmm(
                    grad_t, effective_p3, False, aten_fast.SDPA_CAUSAL_B_COLS
                )
            if grad_value_t is None:
                grad_value_t = _require_handled(
                    aten_fast.fast_aten_bmm(grad_t, effective_p3), "SDPA value gradient"
                )
            grad_value_view = _require_handled(
                aten_fast.fast_aten_transpose(grad_value_t, 1, 2),
                "SDPA transpose value gradient",
            )
            grad_value3 = _contiguous_view(
                grad_value_view, (batch_heads, key_length, head_dim)
            )
            del grad_t, grad_value_t, grad_value_view
            if effective_p3 is not p3:
                del effective_p3

        if fused is aten_fast.NOT_HANDLED and (need_query or need_key):
            v = saved.pop("value")._contig()
            v3 = _contiguous_view(v, (batch_heads, key_length, head_dim))
            # dP_drop = grad @ V^T. BmmSpec's logical transpose flag keeps V
            # dense and avoids materializing its (E,S) transpose.
            grad_effective_probabilities = _require_handled(
                aten_fast._fast_aten_bmm_transpose_b(grad3, v3),
                "SDPA probability gradient",
            )
            del v, v3
            grad_scores = aten_fast.fast_sdpa_dropout_softmax_backward(
                p3,
                grad_effective_probabilities,
                mask3 if ctx.has_dropout else None,
                ctx.dropout_scale,
                ctx.scale,
                ctx.is_causal,
                query_length,
            )
            if grad_scores is aten_fast.NOT_HANDLED:
                # Keep the pre-fusion composition as the exact compatibility
                # route for non-FP32, non-GPU, or otherwise unsupported eager
                # inputs.  Dropout stays before P multiplication and the row
                # reduction, matching the fused arithmetic contract.
                effective_p3 = (
                    _require_handled(
                        aten_fast.fast_aten_native_dropout_backward(
                            grad_effective_probabilities, mask3, ctx.dropout_scale
                        ),
                        "SDPA dropout gradient",
                    )
                    if ctx.has_dropout
                    else grad_effective_probabilities
                )
                if effective_p3 is not grad_effective_probabilities:
                    del grad_effective_probabilities

                weighted = _require_handled(
                    aten_fast.fast_aten_mul(effective_p3, p3),
                    "SDPA softmax weighted gradient",
                )
                row_sum = _require_handled(
                    aten_fast.fast_aten_sum(weighted, dim=[2], keepdim=True),
                    "SDPA softmax gradient reduction",
                )
                del weighted
                centered = _require_handled(
                    aten_fast.fast_aten_sub(effective_p3, row_sum),
                    "SDPA softmax centered gradient",
                )
                del effective_p3, row_sum
                unscaled_grad_scores = _require_handled(
                    aten_fast.fast_aten_mul(centered, p3), "SDPA softmax gradient"
                )
                del centered
                grad_scores = _require_handled(
                    aten_fast.fast_aten_mul(unscaled_grad_scores, ctx.scale),
                    "SDPA scale gradient",
                )
                del unscaled_grad_scores
            else:
                del grad_effective_probabilities

            if need_query:
                k = saved.pop("key")._contig()
                k3 = _contiguous_view(k, (batch_heads, key_length, head_dim))
                grad_query3 = None
                if causal_bmm:
                    # dScores is exactly zero above the diagonal.
                    grad_query3 = aten_fast._try_sdpa_causal_bmm(
                        grad_scores, k3, False, aten_fast.SDPA_CAUSAL_A_ROWS
                    )
                if grad_query3 is None:
                    grad_query3 = _require_handled(
                        aten_fast.fast_aten_bmm(grad_scores, k3), "SDPA query gradient"
                    )
                del k, k3

            if need_key:
                q = saved.pop("query")._contig()
                q3 = _contiguous_view(q, (batch_heads, query_length, head_dim))
                # dK = dScores^T @ Q = (Q^T @ dScores)^T. Only Q^T's
                # O(L*E) storage is materialized; the final copy makes dK
                # contiguous in its required (S,E) layout.
                q_t = _require_handled(
                    aten_fast.fast_aten_transpose(q3, 1, 2), "SDPA transpose query"
                )
                q_t = _contiguous_view(q_t, (batch_heads, head_dim, query_length))
                grad_key_t = None
                if causal_bmm:
                    grad_key_t = aten_fast._try_sdpa_causal_bmm(
                        q_t, grad_scores, False, aten_fast.SDPA_CAUSAL_B_COLS
                    )
                if grad_key_t is None:
                    grad_key_t = _require_handled(
                        aten_fast.fast_aten_bmm(q_t, grad_scores), "SDPA key gradient"
                    )
                grad_key_view = _require_handled(
                    aten_fast.fast_aten_transpose(grad_key_t, 1, 2),
                    "SDPA transpose key gradient",
                )
                grad_key3 = _contiguous_view(
                    grad_key_view, (batch_heads, key_length, head_dim)
                )
                del q, q3, q_t, grad_key_t, grad_key_view
            del grad_scores

        del grad3, p3, mask3
        if saved:
            raise RuntimeError(f"unused Mojo SDPA saved tensors: {tuple(saved)}")

        grad_query = (
            _contiguous_view(grad_query3, ctx.query_shape) if need_query else None
        )
        grad_key = _contiguous_view(grad_key3, ctx.key_shape) if need_key else None
        grad_value = (
            _contiguous_view(grad_value3, ctx.value_shape) if need_value else None
        )
        return grad_query, grad_key, grad_value, None, None, None, None, None


def _scaled_dot_product_attention_autograd(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    *,
    scale=None,
    enable_gqa=False,
):
    needs_backward = torch.is_grad_enabled() and (
        query.requires_grad or key.requires_grad or value.requires_grad
    )
    aten_fast = _fast()
    # The eligible FA4 regime already implements PyTorch's lower flash forward
    # and backward pair. Dispatch through it so generated autograd owns the
    # saves, version checks, and backward call; retain the custom Function only
    # for the generic math/dropout fallback, which has no native backward op.
    if (
        needs_backward
        and aten_fast._fa4_bf16_d64_causal_inputs(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
        is not None
    ):
        return torch.ops.aten._scaled_dot_product_flash_attention.default(
            query, key, value, dropout_p, is_causal, False, scale=scale
        )[0]
    if not needs_backward:
        return _require_handled(
            aten_fast.fast_aten_scaled_dot_product_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
            ),
            "aten::scaled_dot_product_attention",
        )
    if (
        aten_fast._fused_fa_inputs(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
        is not None
    ):
        return _FusedFlashAttentionAutograd.apply(
            query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
        )
    return _ScaledDotProductAttentionAutograd.apply(
        query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa
    )


def register_autograd_ops() -> None:
    """Install concrete AutogradPrivateUse1 kernels once."""
    global _registered
    if _registered:
        return

    torch.library.impl("aten::scaled_dot_product_attention", "AutogradPrivateUse1")(
        _scaled_dot_product_attention_autograd
    )
    _registered = True
