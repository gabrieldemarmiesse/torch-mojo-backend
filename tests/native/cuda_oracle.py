"""Live stock-CUDA oracle: run the same code on torch's `cuda` device and on
the mojo device of the same GPU, in one process, and compare bit for bit.

    def test_asinh(cuda_oracle):
        x = torch.randn(1000)                        # CPU inputs, copied to both
        cuda_oracle.assert_bitwise_equal(torch.asinh, x)
        cuda_oracle.assert_bitwise_equal(torch.add, x, x, kwargs={"alpha": 2})
        cuda_oracle.assert_bitwise_equal_by_device(  # fn builds on `device`
            lambda device: torch.rand(9, device=device), seed=1
        )

The `cuda_oracle` fixture (tests/native/conftest.py) skips without a CUDA
build of torch whose driver initializes, or off NVIDIA; on an NVIDIA CI
runner it fails instead. agents_docs/cuda_oracle.md has the recipes.
"""

from __future__ import annotations

import math
import os
import struct
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch.utils._pytree import tree_flatten, tree_map

from torch_mojo_backend import get_accelerators
from torch_mojo_backend.native import device_module

# "exact": a NaN must have CUDA's sign and payload; "any": any NaN matches any NaN.
NanPolicy = Literal["exact", "any"]
# The bits of a tensor element of each width.
_INT_OF_WIDTH = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
_MAX_REPORTED = 8
_FLOAT8 = {
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
    torch.float8_e8m0fnu,
}


class Raised:
    """What a side did instead of returning: its exception."""

    def __init__(self, error: BaseException):
        self.error = error

    def summary(self, check_message: bool) -> str:
        if not check_message:
            return type(self.error).__name__
        return f"{type(self.error).__name__}: {error_message(self.error)}"


def error_message(error: BaseException) -> str:
    """The first line of an error, with device names as DEV."""
    first = str(error).split("\n")[0].split(" [aten::")[0]
    for name in ("cuda:0", "mojo:0", "cuda", "mojo"):
        first = first.replace(name, "DEV")
    return first


def _bits(t: torch.Tensor) -> torch.Tensor:
    """A CPU integer tensor of `t`'s bit patterns (a complex `t` as its
    real view: [..., 2])."""
    t = t.detach().resolve_conj().resolve_neg().cpu().contiguous()
    if t.is_complex():
        t = torch.view_as_real(t)
    if t.dtype == torch.bool:
        return t.view(torch.uint8)
    return t.view(_INT_OF_WIDTH[t.element_size()])


def _hex(bits: int, width: int) -> str:
    return f"0x{bits & ((1 << (8 * width)) - 1):0{2 * width}x}"


def _element(t: torch.Tensor, index: tuple[int, ...]) -> str:
    """Element `index` of `t` (a complex `t` indexed through its real view)
    as `0xBITS (value)`."""
    host = t.detach().resolve_conj().resolve_neg().cpu()
    if host.is_complex():
        host = torch.view_as_real(host)
    width = host.element_size()
    bits = int(_bits(host)[index])
    value = host[index]
    shown = value.item() if value.dtype not in _FLOAT8 else value.float().item()
    return f"{_hex(bits, width)} ({shown!r})"


def tensor_mismatch(
    ours: torch.Tensor,
    cuda: torch.Tensor,
    *,
    nan: NanPolicy = "exact",
    signed_zero: bool = True,
    check_stride: bool = False,
    inputs: Sequence[torch.Tensor] = (),
) -> str | None:
    """Why `ours` (the mojo device's tensor) is not bit-identical to `cuda`
    (stock CUDA's), or None when it is. `nan` and `signed_zero` relax the
    float comparison; `inputs` of the output's shape are shown at each
    mismatching index."""
    if ours.dtype != cuda.dtype:
        return f"dtype: ours {ours.dtype}, CUDA {cuda.dtype}"
    if ours.shape != cuda.shape:
        return f"shape: ours {tuple(ours.shape)}, CUDA {tuple(cuda.shape)}"
    if check_stride and ours.stride() != cuda.stride():
        return f"stride: ours {ours.stride()}, CUDA {cuda.stride()}"
    a, b = _bits(ours), _bits(cuda)
    same = a == b
    if (ours.is_floating_point() or ours.is_complex()) and (
        nan == "any" or not signed_zero
    ):

        def real(t: torch.Tensor) -> torch.Tensor:
            t = t.detach().resolve_conj().resolve_neg().cpu()
            return (torch.view_as_real(t) if t.is_complex() else t).to(torch.float64)

        x, y = real(ours), real(cuda)
        if nan == "any":
            same |= x.isnan() & y.isnan()
        if not signed_zero:
            same |= (x == 0) & (y == 0)
    if bool(same.all()):
        return None
    bad = (~same).nonzero()
    complex_parts = ours.is_complex()
    lines = [f"{len(bad)} of {same.numel()} elements differ (dtype {ours.dtype}):"]
    for row in bad[:_MAX_REPORTED].tolist():
        index = tuple(row)
        where = list(index[:-1] if complex_parts else index)
        part = (".real", ".imag")[index[-1]] if complex_parts else ""
        line = f"  [{', '.join(map(str, where))}]{part}: ours {_element(ours, index)}"
        line += f", CUDA {_element(cuda, index)}"
        for k, t in enumerate(inputs):
            if t.shape == ours.shape and t.is_complex() == complex_parts:
                line += f"; input {k} {_element(t, index)}"
        lines.append(line)
    return "\n".join(lines)


def _scalar_bits(v: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def _raised_mismatch(ours: object, cuda: object, check_message: bool) -> str | None:
    a = ours.summary(check_message) if isinstance(ours, Raised) else None
    b = cuda.summary(check_message) if isinstance(cuda, Raised) else None
    if a == b:
        return None
    return (
        f"ours {a or f'returned {_describe(ours)}'},"
        f" CUDA {b or f'returned {_describe(cuda)}'}"
    )


def _leaf_mismatch(
    ours: object,
    cuda: object,
    nan: NanPolicy,
    signed_zero: bool,
    check_stride: bool,
    check_message: bool,
    inputs: Sequence[torch.Tensor],
) -> str | None:
    if isinstance(ours, Raised) or isinstance(cuda, Raised):
        return _raised_mismatch(ours, cuda, check_message)
    if isinstance(ours, torch.Tensor) and isinstance(cuda, torch.Tensor):
        return tensor_mismatch(
            ours,
            cuda,
            nan=nan,
            signed_zero=signed_zero,
            check_stride=check_stride,
            inputs=inputs,
        )
    if type(ours) is not type(cuda):
        return f"ours {ours!r} ({type(ours).__name__}), CUDA {cuda!r}"
    if isinstance(ours, (float, complex)) and isinstance(cuda, (float, complex)):
        pairs = [(complex(ours).real, complex(cuda).real)]
        pairs.append((complex(ours).imag, complex(cuda).imag))
        for x, y in pairs:
            if _scalar_bits(x) == _scalar_bits(y):
                continue
            if nan == "any" and math.isnan(x) and math.isnan(y):
                continue
            if not signed_zero and x == 0 and y == 0:
                continue
            return f"ours {ours!r}, CUDA {cuda!r}"
        return None
    return None if ours == cuda else f"ours {ours!r}, CUDA {cuda!r}"


def _describe(value: object) -> str:
    """`value` with each tensor as its dtype and shape."""
    return repr(
        tree_map(
            lambda v: (
                f"<{v.dtype} {tuple(v.shape)}>" if isinstance(v, torch.Tensor) else v
            ),
            value,
        )
    )


def assert_bits_equal(
    ours: object,
    cuda: object,
    *,
    nan: NanPolicy = "exact",
    signed_zero: bool = True,
    check_stride: bool = False,
    check_message: bool = False,
    inputs: Sequence[torch.Tensor] = (),
    what: str = "",
):
    """Assert that `ours` (results on the mojo device) is bit-identical to
    `cuda` (stock CUDA's): the same nesting of tuples / lists / dicts, every
    tensor the same dtype, shape and bits, every scalar the same value (a
    float by its bits). An exception on one side matches the same exception
    type on the other (and its first line too with `check_message`)."""
    prefix = f"{what}: " if what else ""
    ours_leaves, ours_spec = tree_flatten(ours)
    cuda_leaves, cuda_spec = tree_flatten(cuda)
    if ours_spec != cuda_spec:
        if any(isinstance(v, Raised) for v in ours_leaves + cuda_leaves):
            raise AssertionError(
                f"{prefix}{_raised_mismatch(ours, cuda, check_message)}"
            )
        raise AssertionError(f"{prefix}structure: ours {ours_spec}, CUDA {cuda_spec}")
    for k, (a, b) in enumerate(zip(ours_leaves, cuda_leaves, strict=True)):
        why = _leaf_mismatch(
            a, b, nan, signed_zero, check_stride, check_message, inputs
        )
        if why is not None:
            leaf = f"output {k}: " if len(ours_leaves) > 1 else ""
            raise AssertionError(f"{prefix}{leaf}{why}")


@dataclass(frozen=True)
class CudaOracle:
    """One GPU seen as both `cuda:<i>` (stock torch) and `mojo:<i>`."""

    mojo: str
    cuda: str

    @property
    def index(self) -> int:
        return int(self.mojo.rsplit(":", 1)[-1])

    def rng_state(self, device: str) -> torch.Tensor:
        if device == self.cuda:
            return torch.cuda.get_rng_state(self.index).clone()
        return device_module.get_rng_state(self.index).clone()

    def run_by_device(
        self, fn: Callable[[str], object], *, seed: int | None = None
    ) -> tuple[object, object]:
        """`fn(device)` on the mojo device, then on CUDA, each from
        `torch.manual_seed(seed)` when given: (ours, cuda), an exception
        caught as a side's result."""
        results = []
        for device in (self.mojo, self.cuda):
            if seed is not None:
                torch.manual_seed(seed)
            results.append(_call(fn, device))
        return results[0], results[1]

    def assert_bitwise_equal_by_device(
        self,
        fn: Callable[[str], object],
        *,
        seed: int | None = None,
        nan: NanPolicy = "exact",
        signed_zero: bool = True,
        check_stride: bool = False,
        check_message: bool = False,
        inputs: Sequence[torch.Tensor] = (),
        what: str = "",
    ):
        """`fn(device)` must give the same bits on both devices. With `seed`,
        each side starts from `torch.manual_seed(seed)` and must also leave
        the device generator in the same state, whether `fn` returned or
        raised."""

        def with_state(device: str) -> object:
            out = _call(fn, device)
            return (out, self.rng_state(device)) if seed is not None else out

        ours, cuda = self.run_by_device(with_state, seed=seed)
        assert_bits_equal(
            ours,
            cuda,
            nan=nan,
            signed_zero=signed_zero,
            check_stride=check_stride,
            check_message=check_message,
            inputs=inputs,
            what=what,
        )

    def assert_bitwise_equal(
        self,
        fn: Callable[..., object],
        *args: object,
        kwargs: dict[str, object] | None = None,
        seed: int | None = None,
        nan: NanPolicy = "exact",
        signed_zero: bool = True,
        check_stride: bool = False,
        check_message: bool = False,
        what: str = "",
    ):
        """`fn(*args, **kwargs)` must give the same bits on both devices.
        Every tensor in `args` / `kwargs` (nested too) is copied to the
        device first, one copy per distinct tensor object (passing `x` twice
        aliases the two; a view arrives as a dense copy that aliases nothing,
        so build strided or overlapping operands inside `fn`). The inputs are
        compared after the call too, raised or not (after a raise, a tensor
        resized by the call by its shape only), which covers in-place ops and
        `out=`."""
        flat_inputs = [
            t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)
        ]

        def call(device: str) -> object:
            copies: dict[int, torch.Tensor] = {}

            def copy(v: object) -> object:
                if not isinstance(v, torch.Tensor):
                    return v
                if id(v) not in copies:
                    copies[id(v)] = v.to(device, copy=True)
                return copies[id(v)]

            moved = tree_map(copy, (args, kwargs or {}))
            before = {id(t): (t.shape, t.data_ptr()) for t in copies.values()}
            out = _call(lambda _: fn(*moved[0], **moved[1]), device)
            if isinstance(out, Raised):
                # an out= resized before the raise holds uninitialized
                # memory: its shape only
                moved = tree_map(
                    lambda v: (
                        tuple(v.shape)
                        if isinstance(v, torch.Tensor)
                        and before[id(v)] != (v.shape, v.data_ptr())
                        else v
                    ),
                    moved,
                )
            return {"result": out, "inputs_after": moved}

        self.assert_bitwise_equal_by_device(
            call,
            seed=seed,
            nan=nan,
            signed_zero=signed_zero,
            check_stride=check_stride,
            check_message=check_message,
            inputs=flat_inputs,
            what=what,
        )


def _call(fn: Callable[[str], object], device: str) -> object:
    try:
        return fn(device)
    except Exception as error:  # noqa: BLE001 -- compared, not raised
        return Raised(error)


def cuda_oracle_for(mojo_gpu: str) -> CudaOracle:
    """The oracle for `mojo_gpu`, or a skip: off NVIDIA, or without a CUDA
    build of torch whose driver initializes -- except on a CI runner with an
    NVIDIA GPU, where a skip would leave the parity tests unchecked, so it
    fails instead."""
    index = int(mojo_gpu.rsplit(":", 1)[-1])
    if list(get_accelerators())[index].api != "cuda":
        pytest.skip("the CUDA oracle needs an NVIDIA GPU")
    if torch.version.cuda is None or not torch.cuda.is_available():
        if os.environ.get("CI"):
            pytest.fail("stock CUDA torch expected on this NVIDIA runner")
        pytest.skip(
            "needs a CUDA build of torch whose driver initializes"
            " (agents_docs/cuda_oracle.md)"
        )
    return CudaOracle(mojo=mojo_gpu, cuda=f"cuda:{index}")
