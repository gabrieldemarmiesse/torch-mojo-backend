"""torch.compile with tensors on the mojo (mojo_device) eager device.

The mojo_backend path compiles the traced graph to a single MAX graph and
feeds/adopts mojo tensor memory zero-copy via DLPack. The stock dynamo
backends ("eager", "aot_eager") execute the traced graph through the
eager mojo kernels instead.
"""

import ctypes
import gc
from collections.abc import Callable

import max.driver
import pytest
import torch

from tests.test_compiler import check_saved_input_survives_storage_regrow
from torch_mojo_backend import aten_functions, mojo_backend, register_mojo_devices
from torch_mojo_backend.native import device_module
from torch_mojo_backend.torch_compile_backend import compiler

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_max_device():
    register_mojo_devices()


def assert_close_cpu(out, ref, rtol=1e-4, atol=1e-4):
    assert out.device.type == "mojo"
    assert out.device.type == "mojo"
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out.cpu(), ref, rtol=rtol, atol=atol)


def test_compile_elementwise(mojo_device):
    def fn(x, y):
        return torch.relu(x * y + 1.0) - x

    x = torch.randn(4, 8).to(mojo_device)
    y = torch.randn(4, 8).to(mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x, y)
    assert_close_cpu(out, fn(x.cpu(), y.cpu()))


def test_compile_matmul(mojo_device):
    def fn(x, y):
        return torch.relu(x @ y + 1.0)

    x = torch.randn(4, 8).to(mojo_device)
    y = torch.randn(8, 16).to(mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x, y)
    # Loose tolerance: MAX uses tf32-style matmul on GPU.
    assert_close_cpu(out, fn(x.cpu(), y.cpu()), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_compile_dtypes(mojo_device, dtype):
    def fn(x, y):
        return x + y * 2

    if dtype.is_floating_point:
        x = torch.randn(4, 8, dtype=dtype).to(mojo_device)
        y = torch.randn(4, 8, dtype=dtype).to(mojo_device)
    else:
        x = torch.arange(32, dtype=dtype).reshape(4, 8).to(mojo_device)
        y = torch.arange(32, dtype=dtype).reshape(4, 8).to(mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x, y)
    assert_close_cpu(out, fn(x.cpu(), y.cpu()), rtol=1e-2, atol=1e-2)


def test_compile_non_contiguous_input(mojo_device):
    def fn(x):
        return x + 1.0

    x = torch.randn(4, 8).to(mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x.t())
    assert_close_cpu(out, fn(x.cpu().t()))


def test_compile_multiple_outputs(mojo_device):
    def fn(x):
        a = x + 1.0
        return a, None, a.t(), x.sum()

    x = torch.randn(4, 8).to(mojo_device)
    outs = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x)
    refs = fn(x.cpu())
    assert outs[1] is None
    for out, ref in zip([outs[0], outs[2], outs[3]], [refs[0], refs[2], refs[3]]):
        assert_close_cpu(out, ref)


def test_compile_output_feeds_eager_ops(mojo_device):
    """Compiled outputs adopt MAX buffers zero-copy; eager kernels must be
    able to consume them directly."""

    def fn(x):
        return x * 2.0

    x = torch.randn(4, 8).to(mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x)
    eager_result = (out + 1.0).sum()
    assert_close_cpu(eager_result, (x.cpu() * 2.0 + 1.0).sum())


def sleep_on_stream(max_device, stream, microseconds):
    """Stall a raw CUDA/HIP stream on the host for `microseconds`.

    Everything launched on a stream after a host function waits for it to
    return, so work queued behind this sleep cannot finish before it ends,
    however fast the GPU is -- a slow kernel without the compute. libc's
    usleep is the host function: its one integer argument travels in the same
    register as the void* user data on x86-64 and aarch64, and it never takes
    the GIL, so a host thread blocked on the stream cannot deadlock it.
    """
    driver = {
        "cuda": ("libcuda.so.1", "cuLaunchHostFunc"),
        "hip": ("libamdhip64.so", "hipLaunchHostFunc"),
    }.get(max_device.api)
    if driver is None:
        pytest.skip(f"no host callbacks on the {max_device.api} MAX backend")
    launch_host_func = getattr(ctypes.CDLL(driver[0]), driver[1])
    launch_host_func.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    launch_host_func.restype = ctypes.c_int
    usleep = ctypes.cast(ctypes.CDLL(None).usleep, ctypes.c_void_p)
    assert launch_host_func(stream, usleep, microseconds) == 0


def test_compile_output_ordered_after_graph(mojo_gpu):
    """An eager op reading a compiled output runs after the graph wrote it.

    The graph runs on MAX's own stream, not on the mojo stream, and the
    compiled call returns before its kernels finish. Instead of a slow graph,
    a 0.5 s sleep is launched on MAX's stream in front of it. An eager read
    not ordered behind the graph sees the buffer unwritten.
    """
    device = torch.device(mojo_gpu)
    max_device = compiler._max_device_for_mojo(device)

    def fn(x):
        return x * 2.0 + 1.0

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    # Compile the graph and the eager add/copy kernels up front: an inline
    # kernel build during the sleep would delay the eager read past it.
    (compiled(torch.zeros(8, 8, device=device)) + 0.0).cpu()

    x = torch.arange(64, dtype=torch.float32).reshape(8, 8) + 1000.0
    x_mojo = x.to(device)
    torch.accelerator.synchronize(device)
    max_device.default_queue.synchronize()

    sleep_on_stream(max_device, max_device.default_queue.native_stream_handle, 500_000)
    out = compiled(x_mojo)
    result = (out + 0.0).cpu()
    torch.testing.assert_close(result, fn(x))


def test_compile_input_ordered_after_eager_op(mojo_gpu):
    """A compiled graph reads its input after the eager op that wrote it.

    The eager op producing the input is queued on the mojo stream behind a
    0.5 s sleep, and the compiled call follows right away. Importing the input
    must make MAX's stream wait for the mojo stream, or the graph reads the
    buffer before the eager op has written it.
    """
    device = torch.device(mojo_gpu)
    max_device = compiler._max_device_for_mojo(device)

    def fn(x):
        return x * 2.0 + 1.0

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    # Compile the graph and the eager mul kernel up front, on values unlike
    # the real input: an inline kernel build would launch the mul after the
    # sleep, and memory recycled from this call must not already hold the
    # right input.
    compiled(torch.zeros(8, 8, device=device) * 3.0).cpu()

    src = torch.arange(64, dtype=torch.float32).reshape(8, 8) + 1000.0
    src_mojo = src.to(device)
    torch.accelerator.synchronize(device)
    max_device.default_queue.synchronize()

    mojo_stream = torch.accelerator.current_stream(device)
    sleep_on_stream(
        max_device, device_module.stream_native_handle(mojo_stream), 500_000
    )
    x_mojo = src_mojo * 3.0  # launched behind the sleep
    result = compiled(x_mojo).cpu()
    torch.testing.assert_close(result, fn(src * 3.0))


def test_compile_nn_module(mojo_device):
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    )
    x = torch.randn(3, 8)
    # No autograd graph: `register_mojo_devices` turns on
    # `set_swap_module_params_on_conversion` (needed to preserve tied
    # weights), and swapping a parameter that a live backward graph still
    # references is rejected by torch itself -- identically for `.to("cuda")`.
    with torch.no_grad():
        ref = model(x)
    model_dev = model.to(mojo_device)
    compiled = torch.compile(model_dev, backend=mojo_backend, fullgraph=True)
    out = compiled(x.to(mojo_device))
    assert_close_cpu(out, ref.detach(), rtol=1e-2, atol=1e-2)


def test_compile_dynamic_shapes(mojo_device):
    def fn(x):
        return torch.relu(x) * 2.0

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    # The second call (new shape) triggers a recompile with dynamic dims.
    for n in (4, 5, 6):
        x = torch.randn(n, 3).to(mojo_device)
        assert_close_cpu(compiled(x), fn(x.cpu()))


def test_compile_shape_int_output(mojo_device):
    def fn(x):
        return x + 1.0, x.shape[0] * 2

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True, dynamic=True)
    x = torch.randn(7, 3).to(mojo_device)
    out, dim = compiled(x)
    assert dim == 14
    assert_close_cpu(out, x.cpu() + 1.0)


def test_compile_input_mutated_between_calls(mojo_device):
    """The cross-call buffer cache aliases input memory: in-place updates
    between calls (optimizer-step pattern) must be visible to the graph."""

    def fn(x, w):
        return x @ w

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    x = torch.randn(2, 3).to(mojo_device)
    w = torch.randn(3, 4).to(mojo_device)
    torch.testing.assert_close(
        compiled(x, w).cpu(), x.cpu() @ w.cpu(), rtol=1e-2, atol=1e-3
    )
    with torch.no_grad():
        w += 1.0
    torch.testing.assert_close(
        compiled(x, w).cpu(), x.cpu() @ w.cpu(), rtol=1e-2, atol=1e-3
    )


def test_compile_lifted_constant(mojo_device):
    """A tensor constant created inside the compiled function: dynamo lifts
    the real mojo tensor via aten::lift_fresh, which FakeTensorMode must
    accept and the graph factory must bake in as a MAX constant."""

    def fn(x):
        return x + torch.tensor([1.0, 2.0, 3.0], device=x.device)

    x = torch.randn(2, 3).to(mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x)
    assert_close_cpu(out, x.cpu() + torch.tensor([1.0, 2.0, 3.0]))


def test_compile_symint_arithmetic(mojo_device):
    """Symbolic dims used as scalars in tensor arithmetic (dynamic shapes)."""

    def fn(x):
        return torch.relu(x) + x.shape[0]

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    for n in (4, 5, 6):
        x = torch.randn(n, 3).to(mojo_device)
        assert_close_cpu(compiled(x), fn(x.cpu()))


def _ar(n: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.arange(n, dtype=dtype)


# name: (fn, per-call args with CPU tensors, mark dim 0 of the first tensor
# dynamic, compiles: the first value's static graph, then one for the rest)
SCALAR_ARG_CASES: dict[
    str, tuple[Callable[..., torch.Tensor], list[tuple[object, ...]], bool, int]
] = {
    "int": (lambda x, s: x * s, [(_ar(4), 2), (_ar(4), 3), (_ar(4), -5)], False, 2),
    "int_int32_tensor": (
        lambda x, s: x * s + s,
        [(_ar(4, torch.int32), 2), (_ar(4, torch.int32), 3), (_ar(4, torch.int32), -7)],
        False,
        2,
    ),
    "float": (
        lambda x, s: x * s,
        [(_ar(4), 0.5), (_ar(4), 0.25), (_ar(4), -1.5)],
        False,
        2,
    ),
    "several": (
        lambda x, a, b, c: x * a + b - c,
        [(_ar(4), 2, 3, 0.5), (_ar(4), 4, 5, 0.25), (_ar(4), -1, 7, 2.0)],
        False,
        2,
    ),
    # dynamo specializes the bool: one more graph for False
    "int_and_bool": (
        lambda x, s, b: x * s if b else x - s,
        [(_ar(4), 2, True), (_ar(4), 3, True), (_ar(4), 4, False), (_ar(4), 5, False)],
        False,
        3,
    ),
    "symint_expression": (
        lambda x, s: x * (s * 2 + 1),
        [(_ar(4), 2), (_ar(4), 3), (_ar(4), -3)],
        False,
        2,
    ),
    "symint_mod": (
        lambda x, s: x * (s % 3),
        [(_ar(4), 2), (_ar(4), 4), (_ar(4), -5), (_ar(4), -7)],
        False,
        2,
    ),
    # the guard `b == 2 * a` makes b an expression of a's symbol
    "guard_replaced_symint": (
        lambda x, a, b: x * a + b if b == 2 * a else x - a,
        [(_ar(4), 1, 2), (_ar(4), 2, 4), (_ar(4), 3, 6)],
        False,
        2,
    ),
    "int64_extremes": (
        lambda x, s: x + s,
        [
            (_ar(3, torch.int64) * 0, 2),
            (_ar(3, torch.int64) * 0, 3),
            (_ar(3, torch.int64) * 0, -(2**63)),
            (_ar(3, torch.int64) * 0, 2**63 - 1),
        ],
        False,
        2,
    ),
    # a SymInt promotes like a Python int: bool -> int64, int8/uint8/fp16 stay
    "bool_tensor": (
        lambda x, s: x * s,
        [(_ar(4) % 2 == 0, 2), (_ar(4) % 2 == 0, 3), (_ar(4) % 2 == 0, -4)],
        False,
        2,
    ),
    "int8_tensor": (
        lambda x, s: x * s,
        [(_ar(4, torch.int8), 2), (_ar(4, torch.int8), 3), (_ar(4, torch.int8), -4)],
        False,
        2,
    ),
    "uint8_tensor": (
        lambda x, s: x * s,
        [(_ar(4, torch.uint8), 2), (_ar(4, torch.uint8), 3), (_ar(4, torch.uint8), 5)],
        False,
        2,
    ),
    "fp16_tensor": (
        lambda x, s: x * s,
        [
            (_ar(4, torch.float16), 2),
            (_ar(4, torch.float16), 3),
            (_ar(4, torch.float16), -4),
        ],
        False,
        2,
    ),
    "view": (
        lambda x, s: x.view(s, -1),
        [(_ar(12), 2), (_ar(12), 3), (_ar(12), 4)],
        False,
        2,
    ),
    "view_mul": (
        lambda x, s: x.view(s, -1) * 2,
        [(_ar(12), 2), (_ar(12), 3), (_ar(12), 4)],
        False,
        2,
    ),
    "ones": (
        lambda x, s: x.sum() + torch.ones(s, device=x.device),
        [(_ar(4), 2), (_ar(4), 3), (_ar(4), 5)],
        False,
        2,
    ),
    "slice_end": (
        lambda x, s: x[:s],
        [(_ar(4), 2), (_ar(4), 3), (_ar(4), 4)],
        False,
        2,
    ),
    "slice_end_negative": (
        lambda x, s: x[:s],
        [(_ar(5), -1), (_ar(5), -2), (_ar(5), -3)],
        False,
        2,
    ),
    "slice_end_past_size": (
        lambda x, s: x[:s] * 1,
        [(_ar(4), 10), (_ar(4), 11), (_ar(4), 12)],
        False,
        2,
    ),
    "slice_start_past_size": (
        lambda x, s: x[s:] * 1,
        [(_ar(4), -10), (_ar(4), -11), (_ar(4), -12)],
        False,
        2,
    ),
    # a returned view of an input: AOT autograd rebuilds it from the input
    # with the output's own offset, so the output must be the real view
    "slice_start": (
        lambda x, s: x[s:],
        [(_ar(5), 1), (_ar(5), 2), (_ar(5), 3)],
        False,
        2,
    ),
    "slice_start_negative": (
        lambda x, s: x[s:],
        [(_ar(5), -1), (_ar(5), -2), (_ar(5), -3)],
        False,
        2,
    ),
    "slice_start_mul": (
        lambda x, s: x[s:] * 1,
        [(_ar(5), 1), (_ar(5), 2), (_ar(5), 3)],
        False,
        2,
    ),
    "slice_start_negative_mul": (
        lambda x, s: x[s:] * 1,
        [(_ar(5), -1), (_ar(5), -2), (_ar(5), -3)],
        False,
        2,
    ),
    "slice_columns": (
        lambda x, s: x[:, s:],
        [(_ar(18).view(3, 6), 1), (_ar(18).view(3, 6), 2), (_ar(18).view(3, 6), 3)],
        False,
        2,
    ),
    "narrow": (
        lambda x, s: x.narrow(0, s, 2),
        [(_ar(5), 1), (_ar(5), 2), (_ar(5), 3)],
        False,
        2,
    ),
    "narrow_mul": (
        lambda x, s: x.narrow(0, s, 2) * 1,
        [(_ar(5), 1), (_ar(5), 2), (_ar(5), 3)],
        False,
        2,
    ),
    "mark_dynamic": (
        lambda x, s: x * s + x.shape[0],
        [(_ar(5), 2), (_ar(6), 3), (_ar(7), -4)],
        True,
        2,
    ),
    "mark_dynamic_slice": (
        lambda x, s: x[:s] * s,
        [(_ar(5), 2), (_ar(6), 3), (_ar(7), 4)],
        True,
        2,
    ),
    "mark_dynamic_slice_start": (
        lambda x, s: x[s:],
        [(_ar(5), 1), (_ar(6), 2), (_ar(7), 3)],
        True,
        2,
    ),
    "mark_dynamic_half": (
        lambda x: x[x.shape[0] // 2 :],
        [(_ar(5),), (_ar(6),), (_ar(7),)],
        True,
        1,
    ),
    "mark_dynamic_half_mul": (
        lambda x: x[x.shape[0] // 2 :] * 1,
        [(_ar(5),), (_ar(6),), (_ar(7),)],
        True,
        1,
    ),
    "mark_dynamic_narrow": (
        lambda x: x.narrow(0, x.shape[0] - 2, 2),
        [(_ar(5),), (_ar(6),), (_ar(7),)],
        True,
        1,
    ),
}


@pytest.mark.parametrize("case", SCALAR_ARG_CASES)
def test_compile_scalar_args_change_between_calls(mojo_device, case):
    """#664: once a python scalar argument changes, dynamo passes an int as a
    SymInt graph input and a float as a 0-d CPU tensor. Each must give the
    right result, and later values must reuse that graph."""
    fn, calls, dynamic, expected_compiles = SCALAR_ARG_CASES[case]
    compiles = []

    def counting_backend(gm, example_inputs):
        compiles.append(gm)
        return mojo_backend(gm, example_inputs)

    compiled = torch.compile(fn, backend=counting_backend, fullgraph=True)
    for args in calls:
        device_args = [
            a.to(mojo_device) if isinstance(a, torch.Tensor) else a for a in args
        ]
        if dynamic:
            torch._dynamo.mark_dynamic(device_args[0], 0)
        out = compiled(*device_args)
        ref = fn(*args)
        assert out.dtype == ref.dtype
        torch.testing.assert_close(out.cpu(), ref, rtol=0, atol=0)
    assert len(compiles) == expected_compiles


@pytest.mark.parametrize("op", ["mul", "add"])
def test_compile_cpu_0d_tensor_does_not_promote(mojo_device, op):
    """A 0-d CPU tensor joins a mojo tensor's op as torch allows, and only
    promotes across categories: fp32 times a float64 0-d tensor is fp32."""

    def fn(x, t):
        return getattr(torch, op)(x, t)

    x = torch.arange(4.0)
    t = torch.tensor(0.5, dtype=torch.float64)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x.to(mojo_device), t)
    assert_close_cpu(out, fn(x, t))


def _add_then_view(x: torch.Tensor, s: int) -> torch.Tensor:
    x.add_(1)
    return x[s:]


def _view_then_mul(x: torch.Tensor, s: int) -> torch.Tensor:
    y = x[s:]
    x.mul_(2)
    return y


@pytest.mark.parametrize("fn", [_add_then_view, _view_then_mul])
def test_compile_view_of_mutated_input(mojo_device, fn):
    """A returned view of an input the graph mutates: AOT autograd copies the
    new value into the input, then rebuilds the view from the input with the
    output's own metadata, so the output must be the real view of it."""
    compiles = []

    def counting_backend(gm, example_inputs):
        compiles.append(gm)
        return mojo_backend(gm, example_inputs)

    compiled = torch.compile(fn, backend=counting_backend, fullgraph=True)
    for s in (1, 2, 3):
        x = torch.arange(5.0)
        x_dev = x.to(mojo_device)
        out = compiled(x_dev, s)
        expected = fn(x, s)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(x_dev.cpu(), x, rtol=0, atol=0)
        assert out.untyped_storage().data_ptr() == x_dev.untyped_storage().data_ptr()
    assert len(compiles) == 2


def test_compile_half_times_wide_cpu_scalar_uses_opmath(mojo_device):
    """fp16 times a 0-d fp32 CPU tensor is fp16, computed in fp32 as CUDA's
    opmath kernels do: 2**-10 * 2**16 is 64, not 2**-10 * fp16(inf)."""
    x = torch.full((4,), 2**-10, dtype=torch.float16)
    t = torch.tensor(2.0**16)
    out = torch.compile(lambda x, t: x * t, backend=mojo_backend, fullgraph=True)(
        x.to(mojo_device), t
    )
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.cpu(), torch.full((4,), 64.0, dtype=torch.float16))


@pytest.mark.parametrize("op", ["mul", "add", "sub"])
def test_compile_bf16_with_changing_float_rounds_once(mojo_device, op):
    """A changing float reaches the graph as a 0-d fp32 tensor; the bf16
    result is computed in fp32 and rounded once, as CUDA's opmath kernels
    do (stock CUDA matches bit-exactly; CPU eager, which rounds the scalar
    to bf16 first for add and sub, does not)."""
    fn = getattr(torch, op)
    x = torch.linspace(-3, 3, 64).to(torch.bfloat16)
    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    for s in (0.5, 0.3, 0.7, 1.1):
        expected = fn(x.float(), torch.tensor(s, dtype=torch.float32)).bfloat16()
        out = compiled(x.to(mojo_device), s)
        assert out.dtype == torch.bfloat16
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


_HALF = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", _HALF, ids=["fp16", "bf16"])
@pytest.mark.parametrize("scalar_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("op", ["mul", "add"])
def test_compile_half_with_0d_device_tensor_rounds_it_first(
    mojo_device, dtype, scalar_dtype, op
):
    """A 0-d *device* tensor is an ordinary operand on CUDA: rounded to the
    result dtype first, unlike a CPU scalar (stock CUDA matches bit-exactly)."""
    fn = getattr(torch, op)
    a = (torch.arange(1, 257, dtype=torch.float64) * 0.0137 + 0.3).to(dtype)
    t = torch.tensor(0.3, dtype=scalar_dtype)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(
        a.to(mojo_device), t.to(mojo_device)
    )
    expected = fn(a.float(), t.to(dtype).float()).to(dtype)
    assert out.dtype == dtype
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_compile_foreach_half_times_cpu_scalar_uses_opmath(mojo_device):
    """A CPU scalar inside a foreach list computes in opmath too: 64, not inf."""

    def fn(x, s):
        return torch._foreach_mul([x], [s])[0]

    x = torch.full((4,), 2**-10, dtype=torch.float16)
    s = torch.tensor(2.0**16)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x.to(mojo_device), s)
    assert out.dtype == torch.float16
    torch.testing.assert_close(out.cpu(), torch.full((4,), 64.0, dtype=torch.float16))


def test_compile_cpu_scalar_kwarg_uses_opmath(mojo_device):
    """A CPU scalar passed by keyword (`mul.Tensor(x, other=t)`) computes in
    opmath too. Core-ATen traces pass it positionally, so the node is built
    by hand."""
    x = torch.full((4,), 2**-10, dtype=torch.float16).to(mojo_device)
    t = torch.tensor(2.0**16)
    graph = torch.fx.Graph()
    x_node = graph.placeholder("x")
    x_node.meta["val"] = x
    t_node = graph.placeholder("t")
    t_node.meta["val"] = t
    out = graph.call_function(torch.ops.aten.mul.Tensor, (x_node,), {"other": t_node})
    out.meta["val"] = torch.empty_like(x)
    graph.output((out,))
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    (result,) = compiler.BaseMaxCompiler(gm, [x, t])(x, t)
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float16
    torch.testing.assert_close(
        result.cpu(), torch.full((4,), 64.0, dtype=torch.float16)
    )


@pytest.mark.parametrize("dtype", _HALF, ids=["fp16", "bf16"])
@pytest.mark.parametrize("op", ["add", "sub"])
@pytest.mark.parametrize("alpha", [0.3, -0.01])
def test_compile_half_add_alpha_rounds_once(mojo_device, dtype, op, alpha):
    """`a + alpha * b` on half tensors (SGD's `p.add_(g, alpha=-lr)`) is
    computed in float32 and rounded once, as CUDA's add kernel (stock CUDA
    matches bit-exactly)."""
    fn = getattr(torch, op)
    a = (torch.arange(1, 257, dtype=torch.float64) * 0.0137 + 0.3).to(dtype)
    b = (torch.arange(256, dtype=torch.float64) * 0.071 - 4.1).to(dtype)
    out = torch.compile(
        lambda a, b: fn(a, b, alpha=alpha), backend=mojo_backend, fullgraph=True
    )(a.to(mojo_device), b.to(mojo_device))
    # one float32 rounding of `a + alpha * b` (a fused multiply-add), then one
    # to the half dtype
    alpha32 = torch.tensor(alpha if op == "add" else -alpha).double()
    expected = (a.double() + alpha32 * b.double()).float().to(dtype)
    assert out.dtype == dtype
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def _mutate_then_slice(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x.add_(1)
    return (x[1:] * w).sum()


def _mutate_then_transpose(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x.add_(1)
    return (x.t() * w).sum()


def _mutate_then_flatten(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x.mul_(2)
    return (x.view(-1) * w).sum()


@pytest.mark.parametrize(
    ("fn", "x_shape", "w_shape"),
    [
        (_mutate_then_slice, (5,), (4,)),
        (_mutate_then_transpose, (2, 3), (3, 2)),
        (_mutate_then_flatten, (2, 3), (6,)),
    ],
    ids=["slice", "transpose", "flatten"],
)
@pytest.mark.parametrize("dynamic", [False, True])
def test_compile_training_saves_view_of_mutated_input(
    mojo_device, fn, x_shape, w_shape, dynamic
):
    """A training graph that mutates a buffer and saves a view of its new
    value for backward: the saved tensor stays the graph's own result, which
    AOT autograd's copy into the buffer must not touch (eager saves the view
    after the in-place op)."""
    torch.manual_seed(0)
    x = torch.randn(x_shape)
    w = torch.randn(w_shape)
    x_dev = x.to(mojo_device)
    w_dev = w.to(mojo_device).requires_grad_()
    w.requires_grad_()
    compiled = torch.compile(fn, backend=mojo_backend, dynamic=dynamic)
    compiled(x_dev, w_dev).backward()
    fn(x, w).backward()
    assert w_dev.grad is not None and w.grad is not None
    torch.testing.assert_close(w_dev.grad.cpu(), w.grad)
    torch.testing.assert_close(x_dev.cpu(), x)


def test_compile_misaligned_input(mojo_device):
    """An input viewed at an odd offset (here 4 bytes in) is copied to an
    aligned buffer: MAX's vector loads fault on it otherwise."""

    def fn(x, y):
        return x * 2 + y

    base = torch.arange(9.0)
    y = torch.arange(8.0)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(
        base.to(mojo_device)[1:], y.to(mojo_device)
    )
    assert_close_cpu(out, fn(base[1:], y))


def test_compile_backward_of_expression_sized_input(mojo_device):
    """The backward of `cat([a, b])` gets a tangent sized `2*s0 + 1`: no
    valid MAX dim name, so a fresh one rebound to the expression."""

    def fn(a, b):
        return (torch.cat([a, b]).unsqueeze(0) * 2).square().sum()

    compiled = torch.compile(fn, backend=mojo_backend)
    for n in (3, 4, 5):
        a, b = torch.randn(n), torch.randn(n + 1)
        a_dev = a.to(mojo_device).requires_grad_()
        b_dev = b.to(mojo_device).requires_grad_()
        compiled(a_dev, b_dev).backward()
        a.requires_grad_()
        b.requires_grad_()
        fn(a, b).backward()
        assert a_dev.grad is not None and b_dev.grad is not None
        torch.testing.assert_close(a_dev.grad.cpu(), a.grad)
        torch.testing.assert_close(b_dev.grad.cpu(), b.grad)


def test_compile_prims_convert_element_type(mojo_device, call_checker):
    call_checker.register(aten_functions.prims_convert_element_type)

    def fn(x):
        return torch.ops.prims.convert_element_type(x, torch.float16)

    x = torch.randn(3, 4)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x.to(mojo_device))
    assert_close_cpu(out, fn(x))


def test_compile_factory_function(mojo_device):
    def fn(x, device):
        return x + torch.ones(4, 8, device=device)

    x = torch.randn(4, 8).to(mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x, mojo_device)
    assert_close_cpu(out, x.cpu() + torch.ones(4, 8))


def test_compile_device_attribute(mojo_device):
    """Reading `x.device` in compiled code (the ubiquitous
    `torch.arange(T, device=idx.device)` pattern) traces through the plain
    `mojo`-device tensor's `device` property without a graph break."""

    def fn(x):
        return x + torch.ones(4, 8, device=x.device)

    x = torch.randn(4, 8).to(mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x)
    assert_close_cpu(out, x.cpu() + torch.ones(4, 8))


def test_compile_backward(mojo_device):
    def fn(x, w):
        return ((x @ w).relu() ** 2).sum()

    x = torch.randn(4, 8).to(mojo_device)
    w = torch.randn(8, 3).to(mojo_device).requires_grad_()
    loss = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x, w)
    loss.backward()
    assert w.grad is not None
    assert w.grad.device.type == "mojo"

    x_cpu = x.cpu().detach()
    w_cpu = w.cpu().detach().requires_grad_(True)
    fn(x_cpu, w_cpu).backward()
    assert w_cpu.grad is not None
    torch.testing.assert_close(w.grad.cpu(), w_cpu.grad, rtol=2e-2, atol=2e-3)


@pytest.mark.parametrize("dynamic", [False, True])
def test_compile_saved_input_survives_storage_regrow(mojo_device, dynamic):
    check_saved_input_survives_storage_regrow(mojo_device, dynamic)


def test_compile_softmax_backward_produces_a_gradient(mojo_device):
    """torch.softmax's gradient must not silently vanish under torch.compile.

    It once did (#406): the old eager layer registered aten::softmax.int with
    no autograd kernel, so AOTAutograd built no backward node and x.grad
    stayed None without an error.
    """

    def fn(x):
        return torch.softmax(x, -1).square().sum()

    x = torch.randn(2, 5, device=mojo_device, requires_grad=True)
    torch.compile(fn, backend=mojo_backend, fullgraph=True)(x).backward()
    assert x.grad is not None

    x_cpu = x.detach().cpu().requires_grad_(True)
    fn(x_cpu).backward()
    torch.testing.assert_close(x.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)


def test_compile_recompiles_for_cpu_inputs(mojo_device):
    """The same compiled function serves mojo and cpu inputs (device guard)."""

    def fn(x):
        return x * 3.0

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    x = torch.randn(4, 8)
    out_mojo = compiled(x.to(mojo_device))
    out_cpu = compiled(x)
    assert out_mojo.device.type == "mojo"
    assert out_cpu.device.type != "mojo"
    assert out_cpu.device.type == "cpu"
    torch.testing.assert_close(out_mojo.cpu(), out_cpu)


def test_compile_eager_backend(mojo_device):
    """Dynamo-only backend: the traced graph runs through the eager kernels."""

    def fn(x):
        return x * 3.0 - 1.0

    x = torch.randn(4, 8).to(mojo_device)
    out = torch.compile(fn, backend="eager", fullgraph=True)(x)
    assert_close_cpu(out, fn(x.cpu()))


def test_compile_aot_eager_backend(mojo_device):
    """aot_eager runs the functionalized aten graph through the eager kernels."""

    def fn(x, y):
        return torch.relu(x @ y + 1.0)

    x = torch.randn(4, 8).to(mojo_device)
    y = torch.randn(8, 16).to(mojo_device)
    out = torch.compile(fn, backend="aot_eager", fullgraph=True)(x, y)
    assert_close_cpu(out, fn(x.cpu(), y.cpu()), rtol=1e-2, atol=1e-2)


class _MiniAttentionBlock(torch.nn.Module):
    """Attention + MLP block exercising embeddings, layernorm, views,
    transposes, masked softmax and matmuls in one compiled graph."""

    mask: torch.Tensor

    def __init__(self, vocab=64, n_embd=32, n_head=4, block_size=16):
        super().__init__()
        self.tok = torch.nn.Embedding(vocab, n_embd)
        self.pos = torch.nn.Embedding(block_size, n_embd)
        self.ln = torch.nn.LayerNorm(n_embd)
        self.c_attn = torch.nn.Linear(n_embd, 3 * n_embd)
        self.head = torch.nn.Linear(n_embd, vocab, bias=False)
        self.n_head = n_head
        self.n_embd = n_embd
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok(idx) + self.pos(pos.unsqueeze(0))
        h = self.ln(x)
        q, k, v = self.c_attn(h).split(self.n_embd, dim=2)
        C = self.n_embd
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (C // self.n_head) ** 0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.head(x + y)


def test_compile_attention_block(mojo_device):
    torch.manual_seed(0)
    model = _MiniAttentionBlock()
    model.eval()
    idx = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        ref = model(idx)
    model_dev = model.to(mojo_device)
    compiled = torch.compile(model_dev, backend=mojo_backend, fullgraph=True)
    with torch.no_grad():
        out = compiled(idx.to(mojo_device))
    assert_close_cpu(out, ref, rtol=2e-2, atol=2e-3)


def test_dlpack_export_keeps_memory_alive(mojo_device):
    """The DLPack capsule must pin the mojo allocation for the consumer.

    Uses `compiler.fast_from_dlpack` -- the actual zero-copy path this
    backend feeds mojo-device inputs into MAX with. The plain single-arg
    `max.driver.Buffer.from_dlpack(x)` is not usable here even after
    `monkeypatching.fix_privateuse1_dlpack_device_type`: MAX's own DLPack
    importer only recognizes the real vendor device codes (CPU/CUDA/ROCm/
    Metal), and every mojo tensor's capsule is tagged DLPack's kDLExtDev
    (ATen's code for any PrivateUse1 backend, renamed or not -- correct for
    torch-to-torch or torch-to-numpy DLPack, but MAX raises "unsupported
    device type in dlpack implementation" on it regardless of whether the
    mojo device happens to be GPU- or CPU-backed). `fast_from_dlpack`
    instead hands MAX the real device explicitly, exactly as compiler.py's
    own input-tensor exchange does.
    """
    x = torch.arange(100, dtype=torch.float32).to(mojo_device)
    expected = x.cpu()
    buffer = compiler.fast_from_dlpack(x)
    del x
    gc.collect()
    # Churn some allocations to surface use-after-free if the pin is broken.
    for _ in range(4):
        _ = torch.randn(100).to(mojo_device)
    # Use the production consumer too. MAX 26.5's raw imported Metal
    # Buffer.to(CPU) rejects a live external buffer; the native copy path
    # can read it, and still requires this Buffer to keep the producer alive.
    roundtrip = compiler._mojo_tensor_from_buffer(buffer).cpu()
    torch.testing.assert_close(roundtrip, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.int64])
def test_max_buffer_copy_preserves_alias(mojo_gpu: str, dtype: torch.dtype):
    """Imported storage must support copies in both directions without detaching."""
    source = torch.arange(359, dtype=dtype)
    buffer = max.driver.Buffer.from_dlpack(source).to(
        compiler._max_device_for_mojo(torch.device(mojo_gpu))
    )
    imported = compiler._mojo_tensor_from_buffer(buffer)
    assert imported.data_ptr() == buffer._data_ptr()
    torch.testing.assert_close(imported.clone().cpu(), source, rtol=0, atol=0)
    replacement = source + 1
    imported.copy_(replacement.to(mojo_gpu))
    device_module.synchronize(mojo_gpu)
    torch.testing.assert_close(
        torch.from_dlpack(buffer.to(max.driver.CPU())), replacement, rtol=0, atol=0
    )
