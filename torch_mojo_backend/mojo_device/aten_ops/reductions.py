"""Per-dim value+index ops: min.dim, sort and topk.

All of them return a `(values, indices)` pair, which `_register_out`
cannot serve (it wraps single-tensor functionals), so each functional
and each `out=` overload gets a small hand-written binding here.
"""

from torch_mojo_backend.mojo_device.torch_mojo_tensor import (
    TorchMojoTensor,
    _resize_payload,
)

from .support import _copy_into_tensor, _fast, _unsupported


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


def _write_pair(
    result: tuple[TorchMojoTensor, TorchMojoTensor],
    values: TorchMojoTensor | None,
    indices: TorchMojoTensor | None,
) -> tuple[TorchMojoTensor | None, TorchMojoTensor | None]:
    """Copy a computed (values, indices) pair into caller-owned outputs.

    Pre-allocated destinations of the wrong shape are rebound the way every
    other out-variant here does it, rather than refused.
    """
    for dst, src in zip((values, indices), result):
        if dst is None:
            continue
        if tuple(dst._shape) != tuple(src._shape):
            _resize_payload(dst, src._shape)
        _copy_into_tensor(dst, src)
    return (values, indices)


def _sort_result(
    op_name: str, input: TorchMojoTensor, dim: int, descending: bool
) -> tuple[TorchMojoTensor, TorchMojoTensor]:
    aten_fast = _fast()
    result = aten_fast.fast_aten_sort(input, dim, descending)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported(op_name, (input,))
    return result


def _topk_result(
    op_name: str, input: TorchMojoTensor, k: int, dim: int, largest: bool, sorted: bool
) -> tuple[TorchMojoTensor, TorchMojoTensor]:
    aten_fast = _fast()
    result = aten_fast.fast_aten_topk(input, k, dim, largest, sorted)
    if result is aten_fast.NOT_HANDLED:
        raise _unsupported(op_name, (input,))
    return result


def mojo_device_sort(
    input: TorchMojoTensor, dim: int = -1, descending: bool = False
) -> tuple[TorchMojoTensor, TorchMojoTensor]:
    """Functional torch.sort: (values, indices) along `dim`."""
    return _sort_result("aten::sort", input, dim, descending)


def mojo_device_sort_stable(
    input: TorchMojoTensor,
    stable: bool | None = None,
    dim: int = -1,
    descending: bool = False,
) -> tuple[TorchMojoTensor, TorchMojoTensor]:
    """torch.sort(stable=...): the kernel's order is total, so `stable` is
    already what it produces and needs no branch."""
    del stable
    return _sort_result("aten::sort.stable", input, dim, descending)


def mojo_device_sort_values(
    input: TorchMojoTensor,
    dim: int = -1,
    descending: bool = False,
    values: TorchMojoTensor | None = None,
    indices: TorchMojoTensor | None = None,
) -> tuple[TorchMojoTensor | None, TorchMojoTensor | None]:
    """Out-variant of torch.sort."""
    return _write_pair(
        _sort_result("aten::sort.values", input, dim, descending), values, indices
    )


def mojo_device_sort_values_stable(
    input: TorchMojoTensor,
    stable: bool | None = None,
    dim: int = -1,
    descending: bool = False,
    values: TorchMojoTensor | None = None,
    indices: TorchMojoTensor | None = None,
) -> tuple[TorchMojoTensor | None, TorchMojoTensor | None]:
    """Out-variant of torch.sort(stable=...) -- the overload torch routes
    every `sort`, `argsort` and `msort` call through."""
    del stable
    return _write_pair(
        _sort_result("aten::sort.values_stable", input, dim, descending),
        values,
        indices,
    )


def mojo_device_topk(
    input: TorchMojoTensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
) -> tuple[TorchMojoTensor, TorchMojoTensor]:
    """Functional torch.topk: (values, indices) along `dim`."""
    return _topk_result("aten::topk", input, k, dim, largest, sorted)


def mojo_device_topk_values(
    input: TorchMojoTensor,
    k: int,
    dim: int = -1,
    largest: bool = True,
    sorted: bool = True,
    values: TorchMojoTensor | None = None,
    indices: TorchMojoTensor | None = None,
) -> tuple[TorchMojoTensor | None, TorchMojoTensor | None]:
    """Out-variant of torch.topk -- the overload torch routes every
    `torch.topk` call through."""
    return _write_pair(
        _topk_result("aten::topk.values", input, k, dim, largest, sorted),
        values,
        indices,
    )
