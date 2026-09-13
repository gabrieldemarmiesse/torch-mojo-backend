"""Run CUDA-only code -- a package with its own compiled kernels -- on mojo
tensors.

The mojo device owns its memory and its streams, and a package written
against `torch.cuda` (causal-conv1d, mamba-ssm, apex, ...) will not look at
either: its C++ asks `x.is_cuda()`, takes `at::cuda::getCurrentCUDAStream()`
and launches there. Triton needs none of this because it launches through its
own driver and only *asks* torch which stream is current
(`triton_driver.py`); a compiled extension instead calls into libtorch_cuda,
so it needs a CUDA build of torch, and it needs to be handed CUDA tensors and
a CUDA stream.

Both are aliases rather than copies:

* **Memory.** MAX allocates through the CUDA driver on the device's *primary*
  context -- the same context `torch.cuda` uses -- so one device pointer is
  valid in both worlds and a tensor can be retagged rather than copied.
  `as_cuda` / `as_mojo` do that by taking torch's own DLPack export and
  rewriting the device code in the capsule (`dlpack.retag_capsule`), which
  keeps shape, strides, dtype and the addressed elements exactly and leaves
  the source tensor pinned by the capsule's deleter. Nothing is allocated and
  nothing is copied; `t.data_ptr()` is equal on both sides.
* **Ordering.** `on_mojo_stream()` makes torch.cuda's current stream an
  `ExternalStream` over the mojo current stream's vendor handle, so the
  package's launches queue behind our kernels and ours behind its, with no
  synchronization.

**An alias is memory and nothing else.** It carries no stream handoff and no
allocator stream tracking, so the ordering above is the *only* thing that
makes it safe, and the two halves are not optional extras of one another:

* torch's `__dlpack__(stream=...)` negotiation, which would have the producer
  record an event for the consumer's stream, is bypassed -- the capsule goes
  out raw -- so an alias used on any stream but the one its memory was
  produced on races that producer.
* `Tensor.record_stream` cannot fix that either. A mojo alias of CUDA memory
  carries a `from_blob` deleter, which mojo's `recordDataPtrOnStream` ignores
  and torch 2.11's CUDA caching allocator ignores for foreign deleters, so
  recording the alias on another stream does *not* stop the original
  allocation being recycled while that stream still reads it.

So: use one stream for both sides -- `on_mojo_stream()`, or `call_cuda` which
enters it for you -- and keep the source tensor alive for as long as any
stream still uses the alias. `as_cuda` / `as_mojo` enforce the first half by
refusing outside an `on_mojo_stream()` block; `unordered=True` is the opt-out
for an alias that is only inspected (shape, dtype, `data_ptr`) or is ordered
some other way, and it is a promise, not a fix.

`call_cuda(fn, *args)` is the two together: convert, call, convert back, all
under the stream. `enable_cuda_fallback()` is the same conversion installed as
a dispatcher fallback, so an op with a CUDA kernel and no Mojo op runs this
way -- with the two exceptions `_convolution_backward_overrideable` describes.

Gradients do not cross an alias (DLPack carries no autograd history), so
`call_cuda` is a leaf: wrap a package's forward and backward entry points in
`CudaAutogradFunction` to get a differentiable mojo-level op.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING

import torch
import torch.utils.dlpack

from torch_mojo_backend.mojo_device.dlpack import retag_capsule
from torch_mojo_backend.native import device_module

if TYPE_CHECKING:
    from torch.autograd.function import FunctionCtx

# DLPack device codes (ATen/dlpack.h): kDLCUDA=2, kDLROCM=10, kDLExtDev=12.
_KDL_CUDA = 2
_KDL_ROCM = 10
_KDL_EXT_DEV = 12

_MOJO = "mojo"


def _vendor_dlpack_code() -> int:
    """ROCm builds of torch spell their device "cuda" but export kDLROCM."""
    return _KDL_ROCM if torch.version.hip is not None else _KDL_CUDA


def is_available() -> bool:
    """Whether this process can run the CUDA (or ROCm) leg at all: a vendor
    build of torch whose driver actually initializes."""
    return torch.cuda.is_available()


# Which mojo device `on_mojo_stream` currently has torch.cuda pointed at, per
# thread: the one thing that orders an alias with our kernels, so the one
# thing `as_cuda` / `as_mojo` demand.
_ordering = threading.local()


def _require_ordering(index: int, unordered: bool, what: str):
    """An alias carries no stream handoff of its own (see the module
    docstring), so the default is to insist on the context that gives it one.
    """
    if unordered:
        return
    active = getattr(_ordering, "index", None)
    if active is None:
        raise RuntimeError(
            f"{what}() outside on_mojo_stream(): an alias is only memory, it "
            "carries no stream handoff, so a launch on torch.cuda's own stream "
            "races our kernels over the same allocation. Wrap the use in "
            "`with cuda_interop.on_mojo_stream(index):`, or call `call_cuda`, "
            "which does. Pass unordered=True for an alias that is only "
            "inspected or is ordered some other way."
        )
    if active != index:
        raise RuntimeError(
            f"{what}() for mojo:{index} inside on_mojo_stream({active}): the "
            "installed CUDA stream is another device's, so it orders nothing "
            "here. One `on_mojo_stream` block per device."
        )


def _gpu_index(device: torch.device) -> int:
    return device.index if device.index is not None else 0


def as_cuda(t: torch.Tensor, unordered: bool = False) -> torch.Tensor:
    """A `cuda` tensor aliasing a mojo tensor's memory. Zero copy.

    Same shape, strides, dtype and addressed elements; `storage_offset()` is
    not carried over, because torch's DLPack export normalizes it into the
    pointer (`data` is `data_ptr()` and `byte_offset` is 0), so a view's alias
    starts at the view's first element with offset 0. The mojo device index is
    the CUDA ordinal, and that is the alias's device, full stop -- MAX
    enumerates accelerators in vendor order, as `triton_driver` relies on too.

    The returned tensor holds the mojo one alive through the DLPack deleter,
    so the alias may outlive every other reference to the source. It is a leaf
    with `requires_grad=False`: see `CudaAutogradFunction` for gradients.

    The alias carries no ordering and no allocator stream tracking, so this
    refuses outside an `on_mojo_stream()` block for the same device unless
    `unordered=True` says the caller orders it some other way (the module
    docstring has the whole contract).
    """
    if t.device.type != _MOJO:
        raise ValueError(f"expected a mojo tensor, got {t.device}")
    index = _gpu_index(t.device)
    if index >= device_module.device_count() - 1:
        raise ValueError(f"mojo:{index} is the MAX CPU device, which has no CUDA alias")
    if index >= torch.cuda.device_count():
        raise ValueError(
            f"mojo:{index} has no CUDA counterpart: torch.cuda sees "
            f"{torch.cuda.device_count()} device(s)"
        )
    _require_ordering(index, unordered, "as_cuda")
    capsule = torch.utils.dlpack.to_dlpack(t.detach())
    return torch.from_dlpack(retag_capsule(capsule, _vendor_dlpack_code(), index))


def as_mojo(t: torch.Tensor, unordered: bool = False) -> torch.Tensor:
    """A `mojo` tensor aliasing a CUDA tensor's memory. Zero copy.

    The reverse of `as_cuda`, with the same contract in both directions: the
    mojo index is the tensor's CUDA ordinal and cannot be chosen (relabelling
    a pointer as another device would move nothing and alias memory the other
    GPU cannot address), `storage_offset()` is normalized into the pointer,
    and the alias is refused outside an `on_mojo_stream()` block for that
    device unless `unordered=True`.

    This is the direction a package's *outputs* take: the memory then belongs
    to torch's CUDA caching allocator rather than to MAX, and stays alive
    because the capsule's deleter pins the CUDA tensor -- which is the only
    thing keeping it alive, since the allocator does not track the mojo
    streams that read it.
    """
    if t.device.type != "cuda":
        raise ValueError(f"expected a cuda tensor, got {t.device}")
    index = _gpu_index(t.device)
    if index >= device_module.device_count() - 1:
        raise ValueError(
            f"{t.device} has no mojo counterpart: MAX sees "
            f"{device_module.device_count() - 1} GPU(s)"
        )
    _require_ordering(index, unordered, "as_mojo")
    capsule = torch.utils.dlpack.to_dlpack(t.detach())
    return torch.from_dlpack(retag_capsule(capsule, _KDL_EXT_DEV, index))


# The traversals below are hand-rolled rather than `tree_map`: they run once
# per converted call (and once per op under the fallback), and pytree costs
# ~3 us per container -- more than a conversion itself. Only the exact
# built-in container types are descended into, so a tuple subclass such as
# `torch.Size` is passed through whole, which is what an op wants anyway.


def _mojo_index(x: object) -> int | None:
    """The device of the first mojo tensor in `x`, converting nothing.

    `on_mojo_stream` has to be entered *before* the conversion -- an alias
    outside it is refused -- and the conversion is what would otherwise have
    discovered the device, so this one cheap pass runs first. It stops at the
    first hit, which is the first argument in nearly every call."""
    if isinstance(x, torch.Tensor):
        return _gpu_index(x.device) if x.device.type == _MOJO else None
    if type(x) is list or type(x) is tuple:
        return _first_mojo_index(x)
    if type(x) is dict:
        return _first_mojo_index(x.values())
    return None


def _first_mojo_index(values: Iterable[object]) -> int | None:
    for v in values:
        index = _mojo_index(v)
        if index is not None:
            return index
    return None


def _to_cuda(x: object, originals: dict[int, torch.Tensor]) -> object:
    if isinstance(x, torch.Tensor):
        if x.device.type != _MOJO:
            return x
        alias = as_cuda(x)
        originals[id(alias)] = x
        return alias
    if type(x) is list:
        return [_to_cuda(v, originals) for v in x]
    if type(x) is tuple:
        return tuple(_to_cuda(v, originals) for v in x)
    if type(x) is dict:
        return {k: _to_cuda(v, originals) for k, v in x.items()}
    if isinstance(x, torch.device) and x.type == _MOJO:
        return torch.device("cuda", _gpu_index(x))
    return x


def _to_mojo(x: object, originals: dict[int, torch.Tensor]) -> object:
    if isinstance(x, torch.Tensor):
        if x.device.type != "cuda":
            return x
        original = originals.get(id(x))
        # not `or`: a multi-element tensor has no truth value
        return as_mojo(x) if original is None else original
    if type(x) is list:
        return [_to_mojo(v, originals) for v in x]
    if type(x) is tuple:
        return tuple(_to_mojo(v, originals) for v in x)
    if type(x) is dict:
        return {k: _to_mojo(v, originals) for k, v in x.items()}
    return x


def _convert_args(
    args: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[tuple[object, ...], dict[str, object], dict[int, torch.Tensor]]:
    """Everything to CUDA in one pass, remembering which mojo tensor each
    alias came from.

    That second half is what makes in-place and `out=` ops behave: those
    return an argument, and torch's contract is that they return *that
    object*. An alias built fresh from the result would share the memory but
    not the identity, so `torch.add(a, b, out=c)` would stop satisfying
    `result is c` and an autograd in-place check would see a different tensor.
    """
    originals: dict[int, torch.Tensor] = {}
    cuda_args = tuple(_to_cuda(a, originals) for a in args)
    cuda_kwargs = (
        {k: _to_cuda(v, originals) for k, v in kwargs.items()} if kwargs else kwargs
    )
    return cuda_args, cuda_kwargs, originals


# One ExternalStream per mojo stream: constructing one, and asking the shim for
# its vendor handle, cost more than everything else `on_mojo_stream` does, and
# a mojo stream keeps its handle for life.
_external_streams: dict[tuple[int, int], torch.cuda.Stream] = {}


def _external_stream(index: int) -> torch.cuda.Stream:
    stream = device_module.current_stream(index)
    cached = _external_streams.get((index, stream.stream_id))
    if cached is None:
        handle = device_module.stream_native_handle(stream)
        if handle == 0:
            raise RuntimeError(
                f"mojo:{index} has no vendor stream handle (the MAX CPU device "
                "cannot back a CUDA ExternalStream)"
            )
        cached = torch.cuda.ExternalStream(handle, device=index)
        _external_streams[(index, stream.stream_id)] = cached
    return cached


@contextlib.contextmanager
def on_mojo_stream(
    device: int | str | torch.device | None = None,
) -> Iterator[torch.cuda.Stream]:
    """Make torch.cuda's current stream the mojo current stream.

    A compiled extension launches on `at::cuda::getCurrentCUDAStream()`, so
    this is what orders its kernels with ours -- both sides then enqueue on
    one vendor stream and neither has to synchronize. The previous CUDA
    stream (and device) are restored on exit; the mojo current stream is not
    touched.

    It is also what makes an alias legal: `as_cuda` / `as_mojo` refuse outside
    this block, since an alias carries no ordering of its own.
    """
    index = device_module._index(device)
    external = _external_stream(index)
    previous = getattr(_ordering, "index", None)
    _ordering.index = index
    try:
        with torch.cuda.stream(external):
            yield external
    finally:
        _ordering.index = previous


def call_cuda(fn: Callable[..., object], *args: object, **kwargs: object) -> object:
    """Run a CUDA-only callable on mojo tensors.

    The mojo current stream is installed as torch.cuda's for the whole thing;
    inside it every mojo tensor in `args`/`kwargs` becomes a CUDA alias, every
    `torch.device("mojo", i)` becomes `cuda:i`, and every CUDA tensor coming
    back becomes a mojo alias. Tensors the callable mutates in place need no
    conversion back: an alias shares the memory, so the mojo tensor already
    holds the result.

    The device is the first mojo tensor's, and one call is one device: a
    second device's tensor has no ordering under that stream, and `as_cuda`
    says so rather than aliasing it.

    This is a leaf with respect to autograd (see the module docstring).
    """
    index = _first_mojo_index(args)
    if index is None:
        index = _first_mojo_index(kwargs.values())
    if index is None:
        raise ValueError("call_cuda needs at least one mojo tensor among its arguments")
    with on_mojo_stream(index):
        cuda_args, cuda_kwargs, originals = _convert_args(args, kwargs)
        out = fn(*cuda_args, **cuda_kwargs)
        return _to_mojo(out, originals)


class CudaAutogradFunction(torch.autograd.Function):
    """Differentiable `call_cuda`, for a package that exposes its forward and
    backward entry points separately (causal-conv1d's
    `causal_conv1d_fwd_function` / `causal_conv1d_bwd_function`, and most
    kernel packages, because their own `autograd.Function` is written that
    way).

    `torch.autograd.grad` over the CUDA aliases would be the general answer,
    but it is not available here: registering a PrivateUse1 backend makes it
    torch's one accelerator, and from then on the engine's
    `TORCH_INTERNAL_ASSERT(opt_ready_stream && opt_parent_stream)` fires for
    any backward over CUDA tensors (see `require_cuda_autograd` in
    tests/conftest.py). Driving the two halves by hand keeps the autograd
    graph entirely on mojo tensors, where it works.

    `backward` returns one gradient per forward input, so `bwd` is called as
    ``bwd(grad_out, *saved)`` and must return that tuple (None for inputs
    that take no gradient).
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        fwd: Callable[..., torch.Tensor],
        bwd: Callable[..., tuple[torch.Tensor | None, ...]],
        *args: torch.Tensor,
    ) -> torch.Tensor:
        out = call_cuda(fwd, *args)
        if not isinstance(out, torch.Tensor):
            raise TypeError("the forward callable must return one tensor")
        ctx.bwd = bwd  # ty: ignore[unresolved-attribute] -- FunctionCtx takes arbitrary attributes
        ctx.save_for_backward(*args)
        return out

    @staticmethod
    def backward(
        ctx: FunctionCtx, grad_out: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        grads = call_cuda(
            ctx.bwd,  # ty: ignore[unresolved-attribute] -- set in forward
            grad_out.contiguous(),
            *ctx.saved_tensors,  # ty: ignore[unresolved-attribute] -- FunctionCtx stub lacks it
        )
        if not isinstance(grads, tuple):
            raise TypeError("the backward callable must return a tuple of gradients")
        return (None, None, *grads)


def cuda_autograd(
    fwd: Callable[..., torch.Tensor],
    bwd: Callable[..., tuple[torch.Tensor | None, ...]],
) -> Callable[..., torch.Tensor]:
    """`fwd`/`bwd`, a package's two CUDA entry points, as one differentiable
    function of mojo tensors."""

    @functools.wraps(fwd)
    def call(*args: torch.Tensor) -> torch.Tensor:
        return CudaAutogradFunction.apply(fwd, bwd, *args)

    return call


# ---------------------------------------------------------------------------
# The generic fallback


# Every Library whose registrations must stay: a `torch.library.Library` has a
# weakref finalizer that calls `m.reset()`, so dropping one takes its
# registrations with it.
_fallback_libs: list[torch.library.Library] = []
_fallback_counts: dict[str, int] = {}


def _cuda_fallback(
    op: torch._ops.OpOverload, *args: object, **kwargs: object
) -> object:
    """Boxed fallback for the mojo dispatch key: run the op's CUDA kernel on
    CUDA aliases of the mojo arguments.

    The dispatcher only reaches a fallback for an op that has no kernel
    registered at this key, so every op the native backend implements is
    untouched -- this is exactly the complement of its registration list.
    """
    _fallback_counts[str(op)] = _fallback_counts.get(str(op), 0) + 1
    index = _first_mojo_index(args)
    if index is None:
        index = _first_mojo_index(kwargs.values())
    if index is None:
        # Nothing to alias; redispatching would land back here forever.
        raise NotImplementedError(
            f"{op} reached the mojo CUDA fallback with no mojo tensor to convert"
        )
    with on_mojo_stream(index):
        cuda_args, cuda_kwargs, originals = _convert_args(args, kwargs)
        out = op(*cuda_args, **cuda_kwargs)
        return _to_mojo(out, originals)


def _convolution_backward_overrideable(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    transposed: bool,
    output_padding: list[int],
    groups: int,
    output_mask: list[bool],
) -> tuple[torch.Tensor | None, ...]:
    """`aten::convolution_backward` never reaches the fallback.

    It is CompositeExplicitAutograd and branches on the backend itself,
    sending anything that is not CPU / CUDA / MKLDNN to this stub -- which
    has a CompositeExplicitAutograd kernel of its own that only raises "use
    TORCH_LIBRARY_IMPL to override this function". A fallback fires where
    *no* kernel is registered, so it never sees either name. Registering the
    stub by hand is that TORCH_LIBRARY_IMPL, and it is what makes a
    convolution trainable on the mojo device while `convolution_backward`
    has no Mojo op.
    """
    bias_sizes = [weight.shape[1] * groups if transposed else weight.shape[0]]
    with on_mojo_stream(_gpu_index(input.device)):
        grads = torch.ops.aten.convolution_backward(
            as_cuda(grad_output),
            as_cuda(input),
            as_cuda(weight),
            bias_sizes,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups,
            output_mask,
        )
        return tuple(None if g is None else as_mojo(g) for g in grads)


# Ops that need a registration of their own rather than the fallback, because
# ATen already put a kernel at the mojo key for them (see the docstring above).
_EXPLICIT_ROUTES = {
    "aten::convolution_backward_overrideable": _convolution_backward_overrideable
}


def _install(lib: torch.library.Library, aten: torch.library.Library):
    lib.fallback(_cuda_fallback, "PrivateUse1")
    for name, fn in _EXPLICIT_ROUTES.items():
        aten.impl(name, fn, "PrivateUse1", allow_override=True)


def enable_cuda_fallback():
    """Route every op the mojo device does not implement through CUDA, for
    the rest of the process. Needs a CUDA build of torch; idempotent."""
    if _fallback_libs:
        return
    if not is_available():
        raise RuntimeError("the CUDA fallback needs a CUDA build of torch")
    lib = torch.library.Library("_", "IMPL")  # noqa: TOR901 -- a process-lifetime registration, by design
    aten = torch.library.Library("aten", "IMPL")  # noqa: TOR901 -- idem
    _install(lib, aten)
    _fallback_libs.extend((lib, aten))


@contextlib.contextmanager
def cuda_fallback() -> Iterator[None]:
    """The same fallback, only for the duration of a block.

    A dispatcher registration lives as long as the `Library` object that
    carries it, so scoping the libraries scopes the fallback. Use this rather
    than `enable_cuda_fallback` wherever silently routing an unimplemented op
    to CUDA would hide a missing mojo kernel from the code that follows.
    """
    if not is_available():
        raise RuntimeError("the CUDA fallback needs a CUDA build of torch")
    with (
        torch.library._scoped_library("_", "IMPL") as lib,
        torch.library._scoped_library("aten", "IMPL") as aten,
    ):
        _install(lib, aten)
        yield


def fallback_counts() -> dict[str, int]:
    """How many times each op went through the CUDA fallback (for tests and
    for finding out what a model is missing)."""
    return dict(_fallback_counts)
