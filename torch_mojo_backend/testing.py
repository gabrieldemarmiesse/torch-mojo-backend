import contextlib
import inspect
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import cast

import torch

from torch_mojo_backend import mojo_backend, native
from torch_mojo_backend.types import CountedCallable


@contextlib.contextmanager
def _xfail_if_unsupported(device: str) -> Iterator[None]:
    """xfail (rather than fail) when the mojo eager backend raises
    NotImplementedError for an input its fast kernels don't cover.

    Killing the graph fallback (docs/strided_owning_tensors_design.md) turned
    "unsupported input" from a slow fallback into a clear raise; this makes the
    existing suite record those as expected-unsupported instead of hard
    failures, without editing individual tests or masking real errors.

    Two spellings are recognized, one per generation of the backend. The old
    Python eager path said "not supported by mojo eager mode"; the native
    backend's `unsupported()` comes through the C++ shim as
    `<why> [aten::<op>.<overload>]` (`raise_from_kernel` in
    native/csrc/shim_dispatch.cpp) and never names the device, so that
    bracketed suffix is what identifies it. Any other NotImplementedError is
    re-raised.
    """
    try:
        yield
    except NotImplementedError as exc:
        declined = "mojo" in str(exc) or "[aten::" in str(exc)
        if str(device).startswith("mojo") and declined:
            import pytest  # noqa: PLC0415 -- pytest is a dev dependency; this module imports without it

            pytest.xfail(f"unsupported on mojo eager: {exc}")
        raise


# Composites the mojo device deliberately registers no kernel for, mapped to
# the native ops ATen's decomposition of them actually calls. Keyed by the
# `aten_functions` twin's name, because that is what a test registers.
#
#   *_like and fill.Scalar are CompositeExplicitAutograd upstream, so they
#   reach the device as `empty.memory_format` (+ `fill_.Scalar` when they
#   write a value) -- see the module docstring of native/mojo/ops_factories.mojo.
#
#   scaled_dot_product_attention and _scaled_dot_product_attention_math are
#   CompositeImplicitAutograd. Registering either would take it out of reach
#   of the decomposition autograd differentiates and silently drop the
#   gradient, so native/mojo/ops_attention.mojo registers only the lower ops:
#   a route with no fused kernel (a mask, the CPU device, an unsupported
#   shape) runs ATen's own math composition, two batched matmuls around one
#   softmax.
_COMPOSITE_NATIVE_OPS: dict[str, tuple[str, ...]] = {
    "aten_empty_like": ("aten::empty.memory_format", "aten::empty_strided"),
    "aten_fill_scalar": ("aten::fill_.Scalar",),
    "aten_ones_like": ("aten::empty.memory_format", "aten::fill_.Scalar"),
    "aten_scaled_dot_product_attention": ("aten::bmm", "aten::_softmax"),
    "aten__scaled_dot_product_attention_math": ("aten::bmm", "aten::_softmax"),
}


class CallChecker:
    """Asserts that at least one of the registered implementations ran.

    Ops have two implementations: the graph one in `aten_functions` (used by
    the torch.compile backend) and the native one on the mojo device. A test
    registers the `aten_functions` twin; `register` also accepts the native
    op(s) of the same name, counted by the C++ shim per boxed-kernel call
    (`native.op_counts`), so the same test passes whether the op routed to
    the graph path (compile) or the native path (eager).

    An op the mojo device leaves to ATen's decomposition has no native op of
    its own name: for those, `_COMPOSITE_NATIVE_OPS` names the ops the
    decomposition calls, and running any of them counts as running the
    composite natively.
    """

    def __init__(self):
        self._functions_to_check: tuple[CountedCallable, ...] | None = None
        self._counts_before_starting_to_check: list[int] | None = None
        self._native_names: list[str] = []
        self._native_before: dict[str, int] = {}

    @staticmethod
    def _native_candidates(func: Callable[..., object]) -> list[str]:
        """Op-name patterns of an aten_functions twin: `aten_mean_out` ->
        aten::mean_out, aten::mean.out and every aten::mean_out.* overload,
        plus, for a composite, the ops its decomposition calls."""
        name = getattr(func, "__name__", "")
        if not name.startswith("aten_"):
            return []
        base = name[len("aten_") :]
        candidates = [f"aten::{base}", f"aten::{base}."]
        if "_" in base:
            head, tail = base.rsplit("_", 1)
            candidates.append(f"aten::{head}.{tail}")
        if "scaled_dot_product" in base:
            candidates.append("scaled_dot_product")
        candidates.extend(_COMPOSITE_NATIVE_OPS.get(name, ()))
        return candidates

    @staticmethod
    def _matches(pattern: str, op_name: str) -> bool:
        # Case-folded: a twin's name is all lowercase, so the overload it
        # yields is too (`aten_fill__scalar` -> `aten::fill_.scalar`), while
        # ATen capitalizes type-named overloads (`aten::fill_.Scalar`). No
        # two aten ops differ only in case.
        pattern, op_name = pattern.lower(), op_name.lower()
        if pattern == "scaled_dot_product":
            return pattern in op_name
        if pattern.endswith("."):
            return op_name.startswith(pattern)
        return op_name == pattern

    def register(self, *funcs: Callable[..., object] | str):
        """Register the implementations expected to run.

        A callable is an `aten_functions` twin, typed `Callable` (each
        caller's own precise signature, e.g. `aten_functions.aten_min`) and
        not `CountedCallable`: under tests `map_to` always wraps them with a
        `call_count` attribute, but that fact is deliberately hidden from
        their static type (see `aten_functions.map_to`). Cast here, at the
        one place that relies on it.

        A string is a native op name (`"aten::addr"`), for an op with no
        `aten_functions` twin because the graph backend leaves it to ATen's
        decomposition: only the mojo device's own kernel can satisfy it.
        """
        expanded: list[CountedCallable] = []
        self._native_names = []
        for func in funcs:
            if isinstance(func, str):
                if func not in self._native_names:
                    self._native_names.append(func)
                continue
            counted_func = cast(CountedCallable, func)
            if counted_func not in expanded:
                expanded.append(counted_func)
            for pattern in self._native_candidates(func):
                if pattern not in self._native_names:
                    self._native_names.append(pattern)
        self._functions_to_check = tuple(expanded)
        self._counts_before_starting_to_check = [
            f.call_count for f in self._functions_to_check
        ]
        if native.is_registered():
            native.op_counting(True)
            self._native_before = native.op_counts()
        else:
            self._native_before = {}

    def _native_called(self) -> bool:
        if not native.is_registered() or not self._native_names:
            return False
        now = native.op_counts()
        for op_name, count in now.items():
            if count > self._native_before.get(op_name, 0) and any(
                self._matches(p, op_name) for p in self._native_names
            ):
                return True
        return False

    def check_was_called(self):
        if (
            self._functions_to_check is None
            or self._counts_before_starting_to_check is None
        ):
            raise ValueError(
                "No function to check was set, call call_checker.register first"
            )
        if not self._functions_to_check and not self._native_names:
            raise ValueError("call_checker.register was called with nothing to check")
        graph_called = any(
            func.call_count > count_before
            for func, count_before in zip(
                self._functions_to_check, self._counts_before_starting_to_check
            )
        )
        if not graph_called and not self._native_called():
            names = ", ".join(
                [f.__name__ for f in self._functions_to_check] + self._native_names
            )
            raise AssertionError(
                f"Expected one of [{names}] (or the native mojo op of the same "
                "name) to be called at least once in the test, but none was"
            )


def _as_tensor_list(
    outputs: torch.Tensor | Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    return [outputs] if isinstance(outputs, torch.Tensor) else list(outputs)


def check_functions_are_equivalent(
    fn: Callable[..., torch.Tensor | Sequence[torch.Tensor]],
    device: str | None,
    inputs: list[torch.Tensor],
    fn_compiled: Callable[..., torch.Tensor | Sequence[torch.Tensor]] | None = None,
    rtol: float | None = None,
    atol: float | None = None,
):
    fn_compiled = fn_compiled or torch.compile(backend=mojo_backend)(fn)
    if device is not None:
        inputs = [input_tensor.to(device) for input_tensor in inputs]

    # We use the compiled first because compiled never changes
    # the input tensors, while the original function might.
    output_compiled = fn_compiled(*inputs)
    output_original = fn(*inputs)

    assert type(output_original) is type(output_compiled)

    for i, (original, compiled) in enumerate(
        zip(_as_tensor_list(output_original), _as_tensor_list(output_compiled))
    ):
        assert original.shape == compiled.shape, f"Issue with output {i}"
        assert original.device == compiled.device, f"Issue with output {i}"
        assert original.dtype == compiled.dtype, f"Issue with output {i}"
        torch.testing.assert_close(original, compiled, rtol=rtol, atol=atol)


# torch.testing.assert_close's default (rtol, atol) per floating dtype.
_DEFAULT_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float16: (1e-3, 1e-5),
    torch.bfloat16: (1.6e-2, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
}


def assert_close_fp64_anchored(
    actual: torch.Tensor,
    expected: torch.Tensor,
    reference: torch.Tensor,
    *,
    slack: float = 2.0,
):
    """`actual` must sit as close to the float64 `reference` as `expected`
    (torch's own result in the working dtype) does, within a factor `slack`.

    For fp32 reductions torch's default bar (rtol 1.3e-6, atol 1e-5) is
    tighter than torch's own rounding error against the exact answer, so
    two correct implementations that merely sum in a different order fail
    it on some elements, depending on the CPU's SIMD width. This bar asks
    the question the default one means to: is the kernel as accurate as
    torch's? The default tolerance stays as a floor, so where torch is exact
    the check is the ordinary one.
    """
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert reference.dtype == torch.float64, reference.dtype
    rtol, atol = _DEFAULT_TOLERANCES[expected.dtype]
    actual, expected = actual.cpu(), expected.cpu()
    if not torch.isfinite(reference).all():
        torch.testing.assert_close(
            actual, expected, rtol=rtol, atol=atol, equal_nan=True
        )
        return
    error = (actual.double() - reference).abs()
    torch_error = (expected.double() - reference).abs()
    allowed = torch.clamp(atol + rtol * reference.abs(), min=slack * torch_error.max())
    bad = error > allowed
    if bad.any():
        worst = int(error.flatten().argmax())
        raise AssertionError(
            f"{int(bad.sum())} / {bad.numel()} elements further from the float64 "
            f"reference than {slack}x torch's own worst error "
            f"({float(torch_error.max()):.3g}) and outside rtol={rtol}, atol={atol}. "
            f"Worst: actual {float(actual.flatten()[worst])!r}, torch "
            f"{float(expected.flatten()[worst])!r}, reference "
            f"{float(reference.flatten()[worst])!r} at flat index {worst}."
        )


@dataclass
class Conf:
    device: str
    compile: bool

    def __str__(self) -> str:
        if self.compile:
            word = "compiled"
        else:
            word = "eager"
        return f"{self.device}, {word}"


def to_device(tensors: list[torch.Tensor], device: str) -> list[torch.Tensor]:
    return [torch.clone(tensor).to(device) for tensor in tensors]


def check_outputs(
    fn: Callable[..., torch.Tensor | Sequence[torch.Tensor]],
    conf: Conf,
    inputs: list[torch.Tensor],
    *,
    rtol: float | None = None,
    atol: float | None = None,
):
    # We compare to eager cpu execution
    # We first check if the function has a device argument
    has_device_arg = "device" in inspect.signature(fn).parameters
    inputs_cpu = to_device(inputs, "cpu")
    if has_device_arg:
        outputs_eager_cpu = fn(*inputs_cpu, device="cpu")
    else:
        outputs_eager_cpu = fn(*inputs_cpu)

    if conf.compile:
        fn_to_run = torch.compile(fn, backend=mojo_backend)
    else:
        fn_to_run = fn

    inputs_on_device = to_device(inputs, conf.device)
    with _xfail_if_unsupported(conf.device):
        if has_device_arg:
            outputs_conf = fn_to_run(*inputs_on_device, device=conf.device)
        else:
            outputs_conf = fn_to_run(*inputs_on_device)

    for i, (output_eager_cpu, output_conf) in enumerate(
        zip(_as_tensor_list(outputs_eager_cpu), _as_tensor_list(outputs_conf))
    ):
        expected_device = torch.device(conf.device)
        if not (output_conf.device == expected_device):
            raise AssertionError(
                f"Issue with output {i}, expected device {repr(expected_device)} but got {repr(output_conf.device)}"
            )
        assert output_eager_cpu.shape == output_conf.shape, f"Issue with output {i}"
        assert output_eager_cpu.dtype == output_conf.dtype, f"Issue with output {i}"
        # move to cpu for comparison
        output_conf_cpu = output_conf.to("cpu")
        torch.testing.assert_close(
            output_eager_cpu, output_conf_cpu, rtol=rtol, atol=atol
        )
