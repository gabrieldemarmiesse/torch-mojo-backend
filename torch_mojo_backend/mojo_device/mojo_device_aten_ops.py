"""ATen op registrations for mojo_device eager mode.

Every op is either bound to its fast implementation in
`eager_kernels/aten_fast.py` (Mojo kernels over raw pointers) or raises
`NotImplementedError` with an actionable message. The old graph fallback
(per-op MAX graph build + interpret, ~2.2 ms/op) is gone — see
docs/strided_owning_tensors_design.md.
"""

import functools
import math
from collections.abc import Callable

import torch
from max.experimental.torch.torch import torch_dtype_to_max

import torch_mojo_backend.is_running_tests
from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
    TorchMojoTensor,
    _copy_strided_into,
    _record_h2d_source,
    _resize_payload,
    find_equivalent_max_device,
)

# Global registry for functions to register
_aten_ops_registry: list[tuple[str, Callable]] = []

# Under tests, each registered op's dispatcher is wrapped with a call
# counter so `CallChecker` can assert the backend's impl for a given op ran
# — uniformly for fast, custom, and out-variant registrations (see
# torch_mojo_backend/testing.py). Keyed by the aten op name.
EAGER_CALL_COUNTERS: dict[str, Callable] = {}

# Unsupported foreach regimes must reach ATen's exact sequential semantics
# without redispatching to this PrivateUse1 registration again.
_COMPOSITE_EXPLICIT_AUTOGRAD = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeExplicitAutograd
)


def register_aten_op(op_name: str):
    """Decorator to mark a function for aten op registration.

    Args:
        op_name: The aten operation name (e.g., "aten::add.Tensor")
    """

    def decorator(func: Callable) -> Callable:
        if torch_mojo_backend.is_running_tests.IS_RUNNING_TESTS:

            @functools.wraps(func)
            def counted(*args, **kwargs):
                counted.call_count += 1
                return func(*args, **kwargs)

            counted.call_count = 0
            EAGER_CALL_COUNTERS[op_name] = counted
            _aten_ops_registry.append((op_name, counted))
            return counted
        _aten_ops_registry.append((op_name, func))
        return func

    return decorator


_aten_fast_module = None


def _fast():
    """The aten_fast module.

    Imported lazily: the first import triggers the (cached) Mojo kernel
    compilation, which pure torch.compile workloads should never pay for.
    """
    global _aten_fast_module
    if _aten_fast_module is None:
        from torch_mojo_backend.eager_kernels import aten_fast

        _aten_fast_module = aten_fast
    return _aten_fast_module


def _describe_args(args, kwargs) -> str:
    descs = []
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, TorchMojoTensor):
            descs.append(f"{tuple(a._shape)}:{a._dtype}")
        elif isinstance(a, torch.Tensor):
            descs.append(f"{tuple(a.shape)}:{a.dtype}:{a.device}")
    return ", ".join(descs) or "none"


def _unsupported(op_name: str, args=(), kwargs=None) -> NotImplementedError:
    return NotImplementedError(
        f"{op_name} is not supported by mojo eager mode for these inputs "
        f"(tensor args: {_describe_args(args, kwargs or {})}). The graph "
        "fallback was removed; add a fast kernel in "
        "torch_mojo_backend/eager_kernels/ or open an issue."
    )


def _eager_impl(fast_name: str, op_name: str) -> Callable:
    """Bind an op to its aten_fast implementation; raise on NOT_HANDLED."""
    fast_fn: Callable | None = None
    not_handled = None

    def dispatcher(*args, **kwargs):
        nonlocal fast_fn, not_handled
        if fast_fn is None:
            aten_fast = _fast()
            fast_fn = getattr(aten_fast, fast_name)
            not_handled = aten_fast.NOT_HANDLED
        result = fast_fn(*args, **kwargs)
        if result is not_handled:
            raise _unsupported(op_name, args, kwargs)
        return result

    return dispatcher


def _not_implemented(op_name: str) -> Callable:
    """Explicit raiser for ops that used to run through the graph fallback
    and have no fast implementation yet. Registered (instead of left
    unregistered) so users get an actionable message, and so the remaining
    surface is greppable."""

    def raiser(*args, **kwargs):
        raise _unsupported(op_name, args, kwargs)

    return raiser


def _register_fast(op_name: str, fast_name: str):
    register_aten_op(op_name)(_eager_impl(fast_name, op_name))


def _register_missing(op_name: str):
    register_aten_op(op_name)(_not_implemented(op_name))


def _copy_into_tensor(dst: TorchMojoTensor, src: TorchMojoTensor) -> None:
    """dst[...] = src[...] with dtype cast + broadcast, any strides."""
    aten_fast = _fast()
    if src._dtype != dst._dtype:
        src = aten_fast._cast_tensor(src, dst._dtype)
    if tuple(src._shape) != tuple(dst._shape):
        expanded = aten_fast.fast_aten_expand(src, dst._shape)
        if expanded is aten_fast.NOT_HANDLED:
            raise _unsupported("aten::copy_ (broadcast)", (dst, src))
        src = expanded
    aten_fast._copy_into(dst, src)


def _out_variant(op_name: str, fast_name: str, *, dtype_policy: str = "safe_cast"):
    """Wrap a functional fast implementation as an out= variant: compute,
    then copy into `out` (strided-safe)."""

    def dispatcher(*args, out: TorchMojoTensor, **kwargs):
        if not isinstance(out, TorchMojoTensor):
            raise RuntimeError(f"{op_name}: expected out to be a mojo tensor")

        # Reject a cross-device destination before launching the functional
        # composition.  Fast implementations already require all tensor
        # operands to share a device, so the first mojo operand identifies
        # the only valid output context.
        input_device = next(
            (arg._device for arg in args if isinstance(arg, TorchMojoTensor)), None
        )
        if input_device is not None and out._device != input_device:
            raise RuntimeError(
                f"{op_name}: expected out and input tensors to be on the same device"
            )

        aten_fast = _fast()
        result = getattr(aten_fast, fast_name)(*args, **kwargs)
        if result is aten_fast.NOT_HANDLED:
            raise _unsupported(op_name, args, kwargs)

        if out._device != result._device:
            raise RuntimeError(
                f"{op_name}: expected out and result tensors to be on the same device"
            )
        result_dtype = max_dtype_to_torch_dtype(result._dtype)
        out_dtype = max_dtype_to_torch_dtype(out._dtype)
        if dtype_policy == "exact":
            valid_dtype = result_dtype == out_dtype
        elif dtype_policy == "bool_or_uint8":
            valid_dtype = out_dtype in (torch.bool, torch.uint8)
        elif dtype_policy == "safe_cast":
            valid_dtype = torch.can_cast(result_dtype, out_dtype)
        else:
            raise AssertionError(f"unknown out dtype policy: {dtype_policy}")
        if not valid_dtype:
            raise RuntimeError(
                f"result type {result_dtype} can't be cast to the desired "
                f"output type {out_dtype}"
            )

        if tuple(result._shape) == tuple(out._shape):
            _copy_into_tensor(out, result)
        else:
            _resize_payload(out, result._shape)
            _copy_into_tensor(out, result)
        return out

    return dispatcher


# ----------------------------------------------------------------------------------
# Data transfer: H2D / D2H / D2D and dtype/device copies
# ----------------------------------------------------------------------------------


@register_aten_op("aten::_copy_from")
def mojo_device__copy_from(self, dest, non_blocking: bool = False):
    src_is_mojo = isinstance(self, TorchMojoTensor)
    dest_is_mojo = isinstance(dest, TorchMojoTensor)

    if src_is_mojo and dest_is_mojo:
        if self._device != dest._device:
            # Cross mojo-device: bounce through the host.
            bounced = TorchMojoTensor._from_cpu(
                self._to_cpu_tensor(), dest._device, non_blocking=non_blocking
            )
            _copy_into_tensor(dest, bounced)
            return dest
        _copy_into_tensor(dest, self)
        return dest

    if src_is_mojo and not dest_is_mojo:
        # D2H; dest is a CPU torch tensor (copy_ handles cast/layout).
        dest.copy_(self._to_cpu_tensor())
        return dest

    if not src_is_mojo and dest_is_mojo:
        # H2D. Resolve dtype/broadcast on the host, then one upload.
        cpu = self.detach()
        torch_dtype = max_dtype_to_torch_dtype(dest._dtype)
        if cpu.dtype != torch_dtype:
            cpu = cpu.to(torch_dtype)
        if tuple(cpu.shape) != tuple(dest._shape):
            cpu = cpu.broadcast_to(dest._shape)
        cpu = cpu.contiguous()
        if dest._is_contiguous:
            if dest._numel > 0:
                from torch_mojo_backend import eager_kernels

                transfer_owner = eager_kernels.tensor_holder.copy_from_host(
                    eager_kernels._ctx_ptr(dest._device),
                    dest._ptr,
                    cpu.data_ptr(),
                    dest._numel * dest._itemsize,
                )
                _record_h2d_source(dest._device, transfer_owner, non_blocking)
        else:
            staged = TorchMojoTensor._from_cpu(
                cpu, dest._device, non_blocking=non_blocking
            )
            _copy_strided_into(dest, staged)
        return dest

    raise RuntimeError(
        f"invalid _copy_from configuration: {type(self)} -> {type(dest)}"
    )


def max_dtype_to_torch_dtype(dtype):
    from max.experimental.torch import max_dtype_to_torch

    return max_dtype_to_torch(dtype)


@register_aten_op("aten::_to_copy")
def mojo_device__to_copy(
    tensor,
    *,
    dtype: torch.dtype | None = None,
    layout: torch.layout | None = None,
    device: torch.device | None = None,
    pin_memory: bool | None = None,
    non_blocking: bool = False,
    memory_format: torch.memory_format | None = None,
):
    aten_fast = _fast()
    if not isinstance(tensor, TorchMojoTensor):
        # CPU tensor moving onto mojo_device (optionally casting on host).
        t = tensor.detach()
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return TorchMojoTensor._from_cpu(
            t, find_equivalent_max_device(device), non_blocking=non_blocking
        )

    result = tensor
    if dtype is not None:
        max_dtype = torch_dtype_to_max(dtype)
        if max_dtype != result._dtype:
            if (
                result._dtype in aten_fast._CAST_DTYPES
                and max_dtype in aten_fast._CAST_DTYPES
            ):
                result = aten_fast._cast_tensor(result, max_dtype)
            else:
                # Exotic dtype pair: cast on the host.
                cpu = result._to_cpu_tensor().to(dtype)
                if device is not None and device.type == "cpu":
                    return cpu
                target = (
                    find_equivalent_max_device(device)
                    if device is not None
                    else result._device
                )
                return TorchMojoTensor._from_cpu(cpu, target, non_blocking=non_blocking)
    if device is not None and device.type == "cpu":
        return result._to_cpu_tensor(non_blocking=non_blocking)
    if device is not None:
        target = find_equivalent_max_device(device)
        if target != result._device:
            return TorchMojoTensor._from_cpu(
                result._to_cpu_tensor(), target, non_blocking=non_blocking
            )
    if result is tensor:
        # _to_copy always returns a fresh tensor.
        result = tensor._materialize_contiguous()
    return result


# ----------------------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------------------


@register_aten_op("aten::empty.memory_format")
def mojo_device_empty_memory_format(
    size, *, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None
) -> TorchMojoTensor:
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return TorchMojoTensor._alloc(
        tuple(size), torch_dtype_to_max(dtype), find_equivalent_max_device(device)
    )


@register_aten_op("aten::empty_strided.memory_format")
@register_aten_op("aten::empty_strided")
def empty_strided(
    size, stride, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMojoTensor:
    # The requested strides are ignored: allocation is always contiguous
    # (matching the previous behavior; our metadata is self-consistent).
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return TorchMojoTensor._alloc(
        tuple(size), torch_dtype_to_max(dtype), find_equivalent_max_device(device)
    )


@register_aten_op("aten::empty_permuted")
def mojo_device_empty_permuted(
    size, physical_layout, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMojoTensor:
    # Uninitialized memory: a contiguous allocation of `size` is valid.
    dtype = torch.get_default_dtype() if dtype is None else dtype
    return TorchMojoTensor._alloc(
        tuple(size), torch_dtype_to_max(dtype), find_equivalent_max_device(device)
    )


@register_aten_op("aten::empty_like")
def mojo_device_empty_like(
    self: TorchMojoTensor,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    mojo_device = self._device if device is None else find_equivalent_max_device(device)
    return TorchMojoTensor._alloc(self._shape, max_dtype, mojo_device)


def _new_factory_device(self: TorchMojoTensor, device):
    """Target MAX device for a `new_*` factory. torch passes `self`'s device
    (whose torch-side index is the phantom 0) when the caller doesn't
    override it, so default to `self`'s real MAX device; only an explicit
    CPU request is honored differently."""
    if device is None:
        return self._device
    torch_dev = torch.device(device) if not isinstance(device, torch.device) else device
    if torch_dev.type == "cpu":
        return find_equivalent_max_device(torch_dev)
    return self._device


@register_aten_op("aten::new_empty")
def mojo_device_new_empty(
    self: TorchMojoTensor,
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    return TorchMojoTensor._alloc(
        tuple(size), max_dtype, _new_factory_device(self, device)
    )


@register_aten_op("aten::new_zeros")
def mojo_device_new_zeros(
    self: TorchMojoTensor,
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    result = _fast().fast_filled(size, 0, max_dtype, _new_factory_device(self, device))
    if result is None:
        raise _unsupported("aten::new_zeros", (self,))
    return result


@register_aten_op("aten::new_ones")
def mojo_device_new_ones(
    self: TorchMojoTensor,
    size,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    result = _fast().fast_filled(size, 1, max_dtype, _new_factory_device(self, device))
    if result is None:
        raise _unsupported("aten::new_ones", (self,))
    return result


@register_aten_op("aten::new_full")
def mojo_device_new_full(
    self: TorchMojoTensor,
    size,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    result = _fast().fast_filled(
        size, fill_value, max_dtype, _new_factory_device(self, device)
    )
    if result is None:
        raise _unsupported("aten::new_full", (self,))
    return result


def _fast_filled_tensor(size, value, dtype, device):
    """Filled-tensor factory (alloc + Fill), or raises."""
    try:
        max_dtype = torch_dtype_to_max(dtype)
    except (KeyError, ValueError):
        raise _unsupported("aten::full (dtype)", (dtype,)) from None
    result = _fast().fast_filled(
        size, value, max_dtype, find_equivalent_max_device(device)
    )
    if result is None:
        raise _unsupported("aten::full", (size, value, dtype))
    return result


@register_aten_op("aten::full")
def mojo_device_full(
    size, fill_value, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMojoTensor:
    if dtype is not None:
        resolved = dtype
    elif isinstance(fill_value, bool):
        resolved = torch.bool
    elif isinstance(fill_value, int):
        resolved = torch.int64
    else:
        resolved = torch.get_default_dtype()
    return _fast_filled_tensor(size, fill_value, resolved, device)


@register_aten_op("aten::full_like")
def mojo_device_full_like(
    self: TorchMojoTensor,
    fill_value,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
) -> TorchMojoTensor:
    max_dtype = self._dtype if dtype is None else torch_dtype_to_max(dtype)
    mojo_device = self._device if device is None else find_equivalent_max_device(device)
    result = _fast().fast_filled(self._shape, fill_value, max_dtype, mojo_device)
    if result is None:
        raise _unsupported("aten::full_like", (self, fill_value))
    return result


@register_aten_op("aten::ones")
def mojo_device_ones(
    size, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMojoTensor:
    resolved = torch.get_default_dtype() if dtype is None else dtype
    return _fast_filled_tensor(size, 1, resolved, device)


@register_aten_op("aten::ones_like")
def mojo_device_ones_like(
    self: TorchMojoTensor,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
) -> TorchMojoTensor:
    return mojo_device_full_like(self, 1, dtype=dtype, device=device)


@register_aten_op("aten::zeros")
def mojo_device_zeros(
    size, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMojoTensor:
    resolved = torch.get_default_dtype() if dtype is None else dtype
    return _fast_filled_tensor(size, 0, resolved, device)


@register_aten_op("aten::zeros_like")
def mojo_device_zeros_like(
    self: TorchMojoTensor,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    memory_format=None,
) -> TorchMojoTensor:
    return mojo_device_full_like(self, 0, dtype=dtype, device=device)


@register_aten_op("aten::scalar_tensor")
def mojo_device_scalar_tensor(
    s, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMojoTensor:
    resolved = torch.float32 if dtype is None else dtype
    return _fast_filled_tensor((), s, resolved, device)


def _host_arange_tensor(start, end, step, dtype: torch.dtype | None) -> torch.Tensor:
    """torch.arange built on the host (exact torch semantics)."""
    return torch.arange(start, end, step, dtype=dtype)


def _device_arange(start, end, step, dtype: torch.dtype | None, device):
    """torch.arange computed by a device kernel, or None to use the host
    path. HF generation loops call torch.arange(..., device=...) every
    step; the host path costs a blocking H2D copy (full queue drain) per
    call, so the common numeric cases run on device instead."""
    for v in (start, end, step):
        # bool is an int subclass; torch treats it as 0/1 here.
        if not isinstance(v, int | float):
            return None
        if isinstance(v, float) and not math.isfinite(v):
            return None
        if abs(v) > _MAX_EXACT_F64_INT:
            # Python inputs cross the kernel boundary as Float64.
            return None
    if step == 0:
        return None  # host path raises torch's own error
    if dtype is None:
        all_int = all(isinstance(v, int) for v in (start, end, step))
        dtype = torch.int64 if all_int else torch.get_default_dtype()
    try:
        max_dtype = torch_dtype_to_max(dtype)
    except (KeyError, ValueError):
        return None
    if isinstance(start, int) and isinstance(end, int) and isinstance(step, int):
        numel = max(0, -(-(end - start) // step))
    else:
        numel = max(0, math.ceil((float(end) - float(start)) / float(step)))
    return _fast().fast_arange(
        numel, start, step, max_dtype, find_equivalent_max_device(device)
    )


# fast_arange receives start/step as Float64; its accumulator follows the
# output dtype and device, matching PyTorch's range kernels.
_MAX_EXACT_F64_INT = 2**53


@register_aten_op("aten::arange")
def mojo_device_arange(
    start, end=None, step=1, *, dtype=None, layout=None, device=None, pin_memory=None
) -> TorchMojoTensor:
    if end is None:
        start, end = 0, start
    result = _device_arange(start, end, step, dtype, device)
    if result is not None:
        return result
    # Build on the host with exact torch semantics, then one H2D copy.
    cpu = _host_arange_tensor(start, end, step, dtype)
    return TorchMojoTensor._from_cpu(cpu, find_equivalent_max_device(device))


@register_aten_op("aten::arange.start_out")
def mojo_device_arange_start_out(start, end, step=1, *, out) -> TorchMojoTensor:
    # torch.arange(start, end, step, device=...) dispatches to the out
    # variant with a pre-allocated `out` of the right size and dtype.
    torch_dtype = max_dtype_to_torch_dtype(out._dtype)
    staged = _device_arange(start, end, step, torch_dtype, out.device)
    if staged is None:
        cpu = _host_arange_tensor(start, end, step, torch_dtype)
        staged = TorchMojoTensor._from_cpu(cpu, out._device)
    if tuple(staged._shape) == tuple(out._shape):
        _copy_into_tensor(out, staged)
    else:
        _resize_payload(out, staged._shape)
        _copy_into_tensor(out, staged)
    return out


@register_aten_op("aten::embedding")
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


@register_aten_op("aten::linear")
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


@register_aten_op("aten::native_batch_norm")
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


@register_aten_op("aten::normal_")
def mojo_device_normal_(
    self: TorchMojoTensor, mean: float = 0.0, std: float = 1.0, generator=None
) -> TorchMojoTensor:
    if generator is not None:
        raise _unsupported("aten::normal_ (generator)", (self,))
    cpu = torch.empty(self._shape, dtype=max_dtype_to_torch_dtype(self._dtype)).normal_(
        mean, std
    )
    staged = TorchMojoTensor._from_cpu(cpu, self._device)
    _copy_into_tensor(self, staged)
    return self


# ----------------------------------------------------------------------------------
# In-place ops with custom plumbing
# ----------------------------------------------------------------------------------


@register_aten_op("aten::add_.Tensor")
def mojo_device_add_(
    self: TorchMojoTensor, other, alpha: float = 1.0
) -> TorchMojoTensor:
    result = _fast().fast_aten_add_(self, other, alpha)
    if result is None:
        raise _unsupported("aten::add_.Tensor", (self, other))
    return result


@register_aten_op("aten::fill_.Scalar")
def mojo_device_fill__scalar(self: TorchMojoTensor, value) -> TorchMojoTensor:
    result = _fast().fast_aten_fill__scalar(self, value)
    if result is None:
        raise _unsupported("aten::fill_.Scalar", (self, value))
    return result


@register_aten_op("aten::masked_fill_.Scalar")
@register_aten_op("aten::masked_fill_.Tensor")
def mojo_device_masked_fill_(
    self: TorchMojoTensor, mask: TorchMojoTensor, value
) -> TorchMojoTensor:
    result = _fast().fast_aten_masked_fill_(self, mask, value)
    if result is None:
        raise _unsupported("aten::masked_fill_", (self, mask, value))
    return result


@register_aten_op("aten::mul_.Tensor")
def mojo_device_mul_(self: TorchMojoTensor, other) -> TorchMojoTensor:
    result = _fast().fast_aten_mul_(self, other)
    if result is None:
        raise _unsupported("aten::mul_.Tensor", (self, other))
    return result


@register_aten_op("aten::relu_")
def mojo_device_relu_(self: TorchMojoTensor) -> TorchMojoTensor:
    aten_fast = _fast()
    result = aten_fast.fast_aten_relu(self)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::relu_", (self,))
    _copy_into_tensor(self, result)
    return self


@register_aten_op("aten::zero_")
def mojo_device_zero_(self: TorchMojoTensor) -> TorchMojoTensor:
    return mojo_device_fill__scalar(self, 0)


# ----------------------------------------------------------------------------------
# Out-variants
# ----------------------------------------------------------------------------------

register_aten_op("aten::addcdiv.out")(
    _out_variant("aten::addcdiv.out", "fast_aten_addcdiv")
)
register_aten_op("aten::addcmul.out")(
    _out_variant("aten::addcmul.out", "fast_aten_addcmul")
)
register_aten_op("aten::div.out")(_out_variant("aten::div.out", "fast_aten_div"))
register_aten_op("aten::lerp.Scalar_out")(
    _out_variant("aten::lerp.Scalar_out", "fast_aten_lerp")
)
register_aten_op("aten::linalg_vector_norm.out")(
    _out_variant(
        "aten::linalg_vector_norm.out",
        "fast_aten_linalg_vector_norm",
        dtype_policy="exact",
    )
)
register_aten_op("aten::mul.out")(_out_variant("aten::mul.out", "fast_aten_mul"))
register_aten_op("aten::mean.out")(_out_variant("aten::mean.out", "fast_aten_mean"))
register_aten_op("aten::sub.out")(_out_variant("aten::sub.out", "fast_aten_sub"))
register_aten_op("aten::any.out")(
    _out_variant("aten::any.out", "fast_aten_any", dtype_policy="bool_or_uint8")
)
register_aten_op("aten::isin.Tensor_Tensor_out")(
    _out_variant("aten::isin.Tensor_Tensor_out", "fast_aten_isin", dtype_policy="exact")
)


# ----------------------------------------------------------------------------------
# min along one dim: functional (values, indices) + out= variant.
# ----------------------------------------------------------------------------------


@register_aten_op("aten::min.dim")
def mojo_device_min_dim(
    input: TorchMojoTensor, dim: int, keepdim: bool = False
) -> tuple[TorchMojoTensor, TorchMojoTensor]:
    """Functional torch.min(x, dim): (values, indices). Registered so torch
    doesn't synthesize it from the out= variant (which would allocate the
    outputs on the phantom index-0 device)."""
    aten_fast = _fast()
    result = aten_fast.fast_aten_min_dim(input, dim, keepdim)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::min.dim", (input,))
    return result


@register_aten_op("aten::min.dim_min")
def mojo_device_min_dim_min(
    input: TorchMojoTensor,
    dim: int,
    keepdim: bool = False,
    min: TorchMojoTensor | None = None,
    min_indices: TorchMojoTensor | None = None,
) -> tuple[TorchMojoTensor, TorchMojoTensor]:
    """Out-variant of torch.min along a dim: writes values into `min` and
    int64 indices into `min_indices` (resizing via payload rebind when the
    pre-allocated shapes don't match, like the other out-variants)."""
    aten_fast = _fast()
    result = aten_fast.fast_aten_min_dim(input, dim, keepdim)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported("aten::min.dim_min", (input,))
    values, indices = result
    for dst, src in ((min, values), (min_indices, indices)):
        if dst is None:
            continue
        if tuple(dst._shape) == tuple(src._shape):
            _copy_into_tensor(dst, src)
        else:
            _resize_payload(dst, src._shape)
            _copy_into_tensor(dst, src)
    return (min, min_indices)


# ----------------------------------------------------------------------------------
# Fast-implemented ops (alphabetical).
# ----------------------------------------------------------------------------------

_register_fast("aten::_adaptive_avg_pool2d", "fast_aten__adaptive_avg_pool2d")


def _register_foreach_inplace(op_name: str, fast_name: str) -> None:
    """Register a mutable ()-returning foreach op: batched fast path, with
    ATen's exact sequential semantics as the fallback (redispatched below
    this PrivateUse1 registration, like `_foreach_mul_.Tensor`)."""
    packet_name, _, overload_name = op_name.removeprefix("aten::").partition(".")
    aten_op = getattr(getattr(torch.ops.aten, packet_name), overload_name or "default")

    @register_aten_op(op_name)
    def dispatcher(self, *args, **kwargs):
        aten_fast = _fast()
        result = getattr(aten_fast, fast_name)(self, *args, **kwargs)
        if result is aten_fast.NOT_HANDLED:
            result = aten_op.redispatch(
                _COMPOSITE_EXPLICIT_AUTOGRAD, self, *args, **kwargs
            )
            # This explicit redispatch runs below ADInplaceOrView, so the
            # TensorList version update is manual on both paths (mutable
            # TensorList schemas returning () get no automatic bump).
            torch.autograd.graph.increment_version(self)
            return result
        torch.autograd.graph.increment_version(self)
        return None


_register_foreach_inplace(
    "aten::_foreach_add_.Scalar", "fast_aten__foreach_add__scalar"
)
_register_foreach_inplace(
    "aten::_foreach_addcdiv_.ScalarList", "fast_aten__foreach_addcdiv__scalarlist"
)
_register_foreach_inplace(
    "aten::_foreach_addcmul_.Scalar", "fast_aten__foreach_addcmul__scalar"
)
_register_foreach_inplace(
    "aten::_foreach_div_.ScalarList", "fast_aten__foreach_div__scalarlist"
)
_register_foreach_inplace(
    "aten::_foreach_lerp_.Scalar", "fast_aten__foreach_lerp__scalar"
)
_register_foreach_inplace(
    "aten::_foreach_mul_.Scalar", "fast_aten__foreach_mul__scalar"
)


@register_aten_op("aten::_foreach_mul_.Tensor")
def mojo_device__foreach_mul__tensor(self, other):
    aten_fast = _fast()
    result = aten_fast.fast_aten__foreach_mul__tensor(self, other)
    if result is aten_fast.NOT_HANDLED:
        result = torch.ops.aten._foreach_mul_.Tensor.redispatch(
            _COMPOSITE_EXPLICIT_AUTOGRAD, self, other
        )
        # This explicit redispatch runs below ADInplaceOrView. A true wrapper
        # subclass therefore needs the same manual TensorList version update
        # as the direct Mojo kernel path.
        torch.autograd.graph.increment_version(self)
        return result

    # Mutable TensorList schemas returning () do not receive an automatic
    # version bump. Match CUDA, including empty and duplicate list entries.
    torch.autograd.graph.increment_version(self)
    return None


@register_aten_op("aten::_foreach_norm.Scalar")
def mojo_device__foreach_norm_scalar(self, ord=2, dtype=None):
    aten_fast = _fast()
    result = aten_fast.fast_aten__foreach_norm(self, ord, dtype=dtype)
    if result is aten_fast.NOT_HANDLED:
        result = aten_fast.foreach_norm_sequential_fallback(self, ord, dtype=dtype)
        if result is aten_fast.NOT_HANDLED:
            return torch.ops.aten._foreach_norm.Scalar.redispatch(
                _COMPOSITE_EXPLICIT_AUTOGRAD, self, ord, dtype=dtype
            )
    return result


@register_aten_op("aten::_foreach_sqrt")
def mojo_device__foreach_sqrt(self):
    aten_fast = _fast()
    result = aten_fast.fast_aten__foreach_sqrt(self)
    if result is aten_fast.NOT_HANDLED:
        return torch.ops.aten._foreach_sqrt.default.redispatch(
            _COMPOSITE_EXPLICIT_AUTOGRAD, self
        )
    return result


_register_fast("aten::_fused_adamw_", "fast_aten__fused_adamw")
_register_fast("aten::_fused_adamw_.tensor_lr", "fast_aten__fused_adamw")
_register_fast("aten::_local_scalar_dense", "fast_aten__local_scalar_dense")
_register_fast("aten::_log_softmax", "fast_aten__log_softmax")
_register_fast(
    "aten::_log_softmax_backward_data", "fast_aten__log_softmax_backward_data"
)
_register_fast(
    "aten::_native_batch_norm_legit_no_training",
    "fast_aten__native_batch_norm_legit_no_training",
)
_register_fast(
    "aten::_scaled_dot_product_attention_math",
    "fast_aten__scaled_dot_product_attention_math",
)
_register_fast(
    "aten::_scaled_dot_product_efficient_attention",
    "fast_aten__scaled_dot_product_efficient_attention",
)
_register_fast(
    "aten::_scaled_dot_product_flash_attention",
    "fast_aten__scaled_dot_product_flash_attention",
)
_register_fast(
    "aten::_scaled_dot_product_flash_attention_backward",
    "fast_aten__scaled_dot_product_flash_attention_backward",
)
_register_fast("aten::_softmax", "fast_aten__softmax")
_register_fast("aten::_unsafe_view", "fast_aten__unsafe_view")
_register_fast("aten::abs", "fast_aten_abs")
_register_fast("aten::acos", "fast_aten_acos")
_register_fast("aten::add.Tensor", "fast_aten_add")
_register_fast("aten::addcdiv", "fast_aten_addcdiv")
_register_fast("aten::addcmul", "fast_aten_addcmul")
_register_fast("aten::addmm", "fast_aten_addmm")
_register_fast("aten::alias", "fast_aten_alias")
_register_fast("aten::all", "fast_aten_all")
_register_fast("aten::all.dim", "fast_aten_all")
_register_fast("aten::all.dims", "fast_aten_all")
_register_fast("aten::amax", "fast_aten_amax")
_register_fast("aten::amin", "fast_aten_amin")
_register_fast("aten::any", "fast_aten_any")
_register_fast("aten::any.dim", "fast_aten_any")
_register_fast("aten::any.dims", "fast_aten_any")
_register_fast("aten::argmax", "fast_aten_argmax")
_register_fast("aten::argmin", "fast_aten_argmin")
_register_fast("aten::asinh", "fast_aten_asinh")
_register_fast("aten::atanh", "fast_aten_atanh")
_register_fast("aten::avg_pool2d", "fast_aten_avg_pool2d")
_register_fast("aten::bitwise_and.Scalar", "fast_aten_bitwise_and")
_register_fast("aten::bitwise_and.Tensor", "fast_aten_bitwise_and")
_register_fast("aten::bitwise_not", "fast_aten_bitwise_not")
_register_fast("aten::bitwise_or.Scalar", "fast_aten_bitwise_or")
_register_fast("aten::bitwise_or.Tensor", "fast_aten_bitwise_or")
_register_fast("aten::bitwise_xor.Scalar", "fast_aten_bitwise_xor")
_register_fast("aten::bitwise_xor.Tensor", "fast_aten_bitwise_xor")
_register_fast("aten::bmm", "fast_aten_bmm")
_register_fast("aten::cat", "fast_aten_cat")
_register_fast("aten::ceil", "fast_aten_ceil")
_register_fast("aten::clamp", "fast_aten_clamp")
_register_fast("aten::clone", "fast_aten_clone")
_register_fast("aten::convolution", "fast_aten_convolution")
_register_fast("aten::cos", "fast_aten_cos")
_register_fast("aten::cosh", "fast_aten_cosh")
_register_fast("aten::cumsum", "fast_aten_cumsum")
_register_fast("aten::detach", "fast_aten_detach")
_register_fast("aten::div.Tensor", "fast_aten_div")
_register_fast("aten::embedding_dense_backward", "fast_aten_embedding_dense_backward")
_register_fast("aten::eq", "fast_aten_eq")
_register_fast("aten::eq.Scalar", "fast_aten_eq")
_register_fast("aten::eq.Tensor", "fast_aten_eq")
_register_fast("aten::erf", "fast_aten_erf")
_register_fast("aten::exp", "fast_aten_exp")
_register_fast("aten::expand", "fast_aten_expand")
_register_fast("aten::fill.Scalar", "fast_aten_fill_scalar")
_register_fast("aten::floor", "fast_aten_floor")
_register_fast("aten::floor_divide", "fast_aten_floor_divide")
_register_fast("aten::floor_divide.Scalar", "fast_aten_floor_divide")
_register_fast("aten::floordiv", "fast_aten_floor_divide")
_register_fast("aten::ge", "fast_aten_ge")
_register_fast("aten::ge.Scalar", "fast_aten_ge")
_register_fast("aten::ge.Tensor", "fast_aten_ge")
_register_fast("aten::gelu", "fast_aten_gelu")
_register_fast("aten::gelu_backward", "fast_aten_gelu_backward")
_register_fast("aten::gt", "fast_aten_gt")
_register_fast("aten::gt.Scalar", "fast_aten_gt")
_register_fast("aten::gt.Tensor", "fast_aten_gt")
_register_fast("aten::index.Tensor", "fast_aten_index")
_register_fast("aten::isin.Tensor_Tensor", "fast_aten_isin")
_register_fast("aten::isnan", "fast_aten_isnan")
_register_fast("aten::le", "fast_aten_le")
_register_fast("aten::le.Scalar", "fast_aten_le")
_register_fast("aten::le.Tensor", "fast_aten_le")
_register_fast("aten::lerp.Scalar", "fast_aten_lerp")
# aten::linear is registered above as mojo_device_linear: its native autograd
# node needs a forward-time preflight, because a NOT_HANDLED reached from
# inside the engine aborts the process instead of raising.
_register_fast("aten::linear_backward", "fast_aten_linear_backward")
_register_fast("aten::log", "fast_aten_log")
_register_fast("aten::log1p", "fast_aten_log1p")
_register_fast("aten::logical_and", "fast_aten_logical_and")
_register_fast("aten::logical_not", "fast_aten_logical_not")
_register_fast("aten::logical_xor", "fast_aten_logical_xor")
_register_fast("aten::lt", "fast_aten_lt")
_register_fast("aten::lt.Scalar", "fast_aten_lt")
_register_fast("aten::lt.Tensor", "fast_aten_lt")
_register_fast("aten::masked_fill.Scalar", "fast_aten_masked_fill")
_register_fast("aten::masked_fill.Tensor", "fast_aten_masked_fill")
_register_fast("aten::max", "fast_aten_max")
_register_fast("aten::max_pool2d_with_indices", "fast_aten_max_pool2d_with_indices")
_register_fast("aten::maximum", "fast_aten_maximum")
_register_fast("aten::mean", "fast_aten_mean")
# Registering the base name only covers the default overload; mean.dim would
# otherwise get decomposed by PyTorch into a chain of sum/div/... ops.
_register_fast("aten::mean.dim", "fast_aten_mean")
_register_fast("aten::min", "fast_aten_min")
_register_fast("aten::minimum", "fast_aten_minimum")
_register_fast("aten::mm", "fast_aten_mm")
_register_fast("aten::mul.Tensor", "fast_aten_mul")
# aten::native_batch_norm is registered above as mojo_device_native_batch_norm:
# its training form records a native autograd node whose backward this backend
# does not implement, and a raise from inside the engine aborts the process
# instead of raising, so the refusal has to happen in the forward.
_register_fast("aten::native_dropout", "fast_aten_native_dropout")
_register_fast("aten::native_dropout_backward", "fast_aten_native_dropout_backward")
_register_fast("aten::native_group_norm", "fast_aten_native_group_norm")
_register_fast("aten::native_layer_norm", "fast_aten_native_layer_norm")
_register_fast(
    "aten::native_layer_norm_backward", "fast_aten_native_layer_norm_backward"
)
_register_fast("aten::ne", "fast_aten_ne")
_register_fast("aten::ne.Scalar", "fast_aten_ne")
_register_fast("aten::ne.Tensor", "fast_aten_ne")
_register_fast("aten::neg", "fast_aten_neg")
_register_fast(
    "aten::nll_loss_backward.grad_input", "fast_aten_nll_loss_backward_grad_input"
)
_register_fast("aten::nll_loss_forward.output", "fast_aten_nll_loss_forward_output")
_register_fast("aten::nonzero", "fast_aten_nonzero")
_register_fast("aten::permute", "fast_aten_permute")
_register_fast("aten::pow.Tensor_Scalar", "fast_aten_pow")
_register_fast("aten::pow.Tensor_Tensor", "fast_aten_pow_tensor_tensor")
_register_fast("aten::reciprocal", "fast_aten_reciprocal")
_register_fast("aten::relu", "fast_aten_relu")
_register_fast("aten::remainder.Scalar", "fast_aten_remainder")
_register_fast("aten::remainder.Scalar_Tensor", "fast_aten_remainder")
_register_fast("aten::remainder.Tensor", "fast_aten_remainder")
_register_fast("aten::repeat", "fast_aten_repeat")
_register_fast("aten::rsqrt", "fast_aten_rsqrt")
_register_fast(
    "aten::scaled_dot_product_attention", "fast_aten_scaled_dot_product_attention"
)
_register_fast("aten::scatter.src", "fast_aten_scatter_src")
_register_fast("aten::scatter.value", "fast_aten_scatter_value")
_register_fast("aten::select.int", "fast_aten_select")
_register_fast("aten::select_scatter", "fast_aten_select_scatter")
_register_fast("aten::sigmoid", "fast_aten_sigmoid")
_register_fast("aten::sign", "fast_aten_sign")
_register_fast("aten::silu", "fast_aten_silu")
_register_fast("aten::sin", "fast_aten_sin")
_register_fast("aten::sinh", "fast_aten_sinh")
_register_fast("aten::slice.Tensor", "fast_aten_slice")
_register_fast("aten::softmax.int", "fast_aten_softmax")
_register_fast("aten::split.Tensor", "fast_aten_split")
_register_fast("aten::split_with_sizes", "fast_aten_split_with_sizes")
_register_fast("aten::sqrt", "fast_aten_sqrt")
_register_fast("aten::squeeze.dim", "fast_aten_squeeze_dim")
_register_fast("aten::stack", "fast_aten_stack")
_register_fast("aten::sub.Tensor", "fast_aten_sub")
_register_fast("aten::sum.dim_IntList", "fast_aten_sum")
_register_fast("aten::t", "fast_aten_t")
_register_fast("aten::tan", "fast_aten_tan")
_register_fast("aten::tanh", "fast_aten_tanh")
_register_fast("aten::transpose.int", "fast_aten_transpose")
_register_fast("aten::tril", "fast_aten_tril")
_register_fast("aten::triu", "fast_aten_triu")
_register_fast("aten::unbind.int", "fast_aten_unbind")
_register_fast("aten::unsqueeze", "fast_aten_unsqueeze")
_register_fast("aten::upsample_bilinear2d", "fast_aten_upsample_bilinear2d")
_register_fast("aten::var.correction", "fast_aten_var")
_register_fast("aten::view", "fast_aten_view")
_register_fast("aten::where.self", "fast_aten_where")


# ----------------------------------------------------------------------------------
# Ops with no fast implementation yet: explicit raisers (previously served
# by the graph fallback). Training-only backward passes; implement in
# eager_kernels and move up if an inference workload needs them.
# ----------------------------------------------------------------------------------

_register_missing("aten::_adaptive_avg_pool2d_backward")
