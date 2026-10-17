"""Stream orchestration must survive compilation, including repeated calls."""

import max.driver
import pytest
import torch

from tests.conftest import require_cuda_autograd
from tests.native.conftest import skip_if_metal
from torch_mojo_backend import mojo_backend
from torch_mojo_backend.torch_compile_backend import compiler


def current_stream(device):
    # Registering PrivateUse1 makes it torch.accelerator's default, even
    # though CUDA tensors and CUDA streams remain available in the process.
    if device.type == "cuda":
        return torch.cuda.current_stream(device)
    return torch.accelerator.current_stream(device)


def stream_context(stream, device):
    return torch.cuda.stream(stream) if device.startswith("cuda") else stream


@pytest.fixture(params=["cuda", "mojo"])
def stream_device(request):
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("requires CUDA torch")
        return "cuda:0"
    device = request.getfixturevalue("mojo_gpu")
    skip_if_metal(device, "Metal has no independent streams/events")
    return device


@pytest.mark.parametrize("fullgraph", [False, True])
def test_compile_wait_for_producer(stream_device, fullgraph):
    producer = torch.Stream(device=stream_device)

    def fn(x):
        current_stream(x.device).wait_stream(producer)
        return x + 1

    x = torch.ones(32, device=stream_device)
    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=fullgraph)
    compiled(x).cpu()
    for value in (7, 11):
        producer.wait_stream(current_stream(x.device))
        with stream_context(producer, stream_device):
            # Enough queued work to expose a missing device-side wait without
            # relying on a CUDA-only sleep or a wall-clock assertion.
            work = torch.ones(1024, 1024, device=stream_device)
            for _ in range(32):
                work = work @ work
            x.fill_(value)
        result = compiled(x)
        torch.testing.assert_close(result.cpu(), torch.full((32,), value + 1.0))
        producer.synchronize()


@pytest.mark.parametrize("create_inside", [False, True])
def test_compile_two_streams(stream_device, create_inside):
    first = torch.Stream(device=stream_device)
    second = torch.Stream(device=stream_device)

    def fn(x):
        a = torch.Stream(device=x.device) if create_inside else first
        b = torch.Stream(device=x.device) if create_inside else second
        current = current_stream(x.device)
        a.wait_stream(current)
        b.wait_stream(current)
        with stream_context(a, stream_device):
            y = x.sin()
        with stream_context(b, stream_device):
            z = x.cos()
        current.wait_stream(a)
        current.wait_stream(b)
        return y + z

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    for _ in range(2):
        x = torch.randn(32, device=stream_device)
        torch.testing.assert_close(compiled(x).cpu(), fn(x).cpu())


@pytest.mark.parametrize("event_api", ["stream", "event"])
def test_compile_event_handoff(stream_device, event_api):
    first = torch.Stream(device=stream_device)
    second = torch.Stream(device=stream_device)
    event = torch.Event(device=stream_device)

    def fn(x):
        current = current_stream(x.device)
        first.wait_stream(current)
        with stream_context(first, stream_device):
            y = x + 3
            if event_api == "event":
                event.record()
                recorded = event
            else:
                recorded = first.record_event()
        with stream_context(second, stream_device):
            if event_api == "event":
                recorded.wait()
            else:
                second.wait_event(recorded)
            z = y * 2
        current.wait_stream(second)
        return z

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    for _ in range(2):
        x = torch.randn(32, device=stream_device)
        torch.testing.assert_close(compiled(x).cpu(), (x.cpu() + 3) * 2)


def test_compile_independent_regions_launch_on_distinct_max_streams(stream_device):
    first = torch.Stream(device=stream_device)
    second = torch.Stream(device=stream_device)

    def fn(x):
        current = current_stream(x.device)
        first.wait_stream(current)
        second.wait_stream(current)
        with stream_context(first, stream_device):
            a = x.sin()
        with stream_context(second, stream_device):
            b = x.cos()
        current.wait_stream(first)
        current.wait_stream(second)
        return a + b

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    x = torch.randn(1024, device=stream_device)
    compiled(x).cpu()
    max.driver.begin_launch_trace()
    try:
        result = compiled(x)
    finally:
        launches = max.driver.take_launch_trace()
    torch.testing.assert_close(result.cpu(), x.cpu().sin() + x.cpu().cos())
    sine_streams = {
        entry.stream_index for entry in launches if "elementwise_sin" in entry.name
    }
    cosine_streams = {
        entry.stream_index for entry in launches if "elementwise_cos" in entry.name
    }
    assert sine_streams and cosine_streams, [entry.name for entry in launches]
    assert sine_streams.isdisjoint(cosine_streams)


def test_compile_nested_streams_and_backward(stream_device):
    require_cuda_autograd(torch.device(stream_device).type)
    first = torch.Stream(device=stream_device)
    second = torch.Stream(device=stream_device)

    def fn(x):
        current = current_stream(x.device)
        first.wait_stream(current)
        with stream_context(first, stream_device):
            a = x * 2
            second.wait_stream(first)
            with stream_context(second, stream_device):
                b = a * x
            first.wait_stream(second)
            c = b + x
        current.wait_stream(first)
        return c

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    for _ in range(2):
        x = torch.randn(32, device=stream_device, requires_grad=True)
        out = compiled(x)
        out.sum().backward()
        torch.testing.assert_close(
            out.detach().cpu(), 2 * x.detach().cpu() ** 2 + x.detach().cpu()
        )
        assert x.grad is not None
        torch.testing.assert_close(x.grad.cpu(), 4 * x.detach().cpu() + 1)


def test_compile_stream_regions_backward(stream_device):
    require_cuda_autograd(torch.device(stream_device).type)
    first = torch.Stream(device=stream_device)
    second = torch.Stream(device=stream_device)

    def fn(x):
        current = current_stream(x.device)
        first.wait_stream(current)
        second.wait_stream(current)
        with stream_context(first, stream_device):
            a = x.sin()
        with stream_context(second, stream_device):
            b = x.cos()
        current.wait_stream(first)
        current.wait_stream(second)
        return a + b

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True, dynamic=True)
    for rows in (3, 7):
        x = torch.randn(rows, 4, device=stream_device, requires_grad=True)
        compiled(x).sum().backward()
        assert x.grad is not None
        host = x.detach().cpu()
        torch.testing.assert_close(x.grad.cpu(), host.cos() - host.sin())


def test_compile_backward_hooks_observe_forward_streams(stream_device):
    require_cuda_autograd(torch.device(stream_device).type)
    first = torch.Stream(device=stream_device)
    second = torch.Stream(device=stream_device)

    def fn(x, y):
        current = current_stream(x.device)
        first.wait_stream(current)
        second.wait_stream(current)
        with stream_context(first, stream_device):
            a = x.sin()
        with stream_context(second, stream_device):
            b = y.cos()
        current.wait_stream(first)
        current.wait_stream(second)
        return a + b

    def first_hook(grad):
        return grad * (
            2 if current_stream(grad.device).stream_id == first.stream_id else 3
        )

    def second_hook(grad):
        return grad * (
            2 if current_stream(grad.device).stream_id == second.stream_id else 3
        )

    x = torch.ones(32, device=stream_device, requires_grad=True)
    y = torch.ones(32, device=stream_device, requires_grad=True)
    x.register_hook(first_hook)
    y.register_hook(second_hook)
    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    compiled(x, y).sum().backward()
    assert x.grad is not None and y.grad is not None
    torch.testing.assert_close(x.grad.cpu(), 2 * x.detach().cpu().cos())
    torch.testing.assert_close(y.grad.cpu(), -2 * y.detach().cpu().sin())


def test_compile_mutation_on_stream(stream_device):
    side = torch.Stream(device=stream_device)

    def fn(x):
        current = current_stream(x.device)
        side.wait_stream(current)
        with stream_context(side, stream_device):
            x.add_(3)
        current.wait_stream(side)

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    x = torch.zeros(32, device=stream_device)
    for expected in (3, 6):
        assert compiled(x) is None
        torch.testing.assert_close(x.cpu(), torch.full((32,), float(expected)))


def test_compile_called_on_different_streams(stream_device):
    # No stream node inside the graph: runtime handoffs must still refresh
    # when the same tensor object and compiled function change streams.
    compiled = torch.compile(lambda x: x + 1, backend=mojo_backend, fullgraph=True)
    x = torch.zeros(32, device=stream_device)
    compiled(x).cpu()
    current = current_stream(x.device)
    for value in (4, 9):
        side = torch.Stream(device=stream_device)
        side.wait_stream(current)
        with stream_context(side, stream_device):
            x.fill_(value)
            result = compiled(x)
        current.wait_stream(side)
        torch.testing.assert_close(result.cpu(), torch.full((32,), value + 1.0))


def test_compile_factory_inside_stream(stream_device):
    side = torch.Stream(device=stream_device)

    def fn():
        with stream_context(side, stream_device):
            x = torch.ones(128, device=stream_device) * 7
        current_stream(x.device).wait_stream(side)
        return x + 1

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    for _ in range(2):
        torch.testing.assert_close(compiled().cpu(), torch.full((128,), 8.0))


def test_compile_stream_dynamic_shapes(stream_device):
    side = torch.Stream(device=stream_device)

    def fn(x):
        current = current_stream(x.device)
        side.wait_stream(current)
        with stream_context(side, stream_device):
            y = x.sin().reshape(x.shape[0], -1)
        current.wait_stream(side)
        return y + 1

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True, dynamic=True)
    for rows in (3, 7, 5):
        x = torch.randn(rows, 4, device=stream_device)
        torch.testing.assert_close(compiled(x).cpu(), x.cpu().sin() + 1)


def slow_work(device):
    # Queues enough work on the current stream to expose a missing wait.
    work = torch.ones(1024, 1024, device=device)
    for _ in range(32):
        work = work @ work


def test_compile_stream_regions_with_a_scalar_that_changes(stream_device):
    # Once `scale` changes, dynamo retraces it as a SymInt that no tensor
    # input has as a dimension: the side-stream regions must see its Dim,
    # also as the size of a tensor they create.
    first = torch.Stream(device=stream_device)
    second = torch.Stream(device=stream_device)

    def fn(x, scale):
        current = current_stream(x.device)
        first.wait_stream(current)
        second.wait_stream(current)
        with stream_context(first, stream_device):
            a = x * scale
            c = torch.ones(scale, device=x.device) * 2
        with stream_context(second, stream_device):
            b = x + scale
        current.wait_stream(first)
        current.wait_stream(second)
        return a, b, c

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    x = torch.randn(64, device=stream_device)
    for scale in (2, 3, 5, 7):
        a, b, c = compiled(x, scale)
        torch.testing.assert_close(a.cpu(), x.cpu() * scale)
        torch.testing.assert_close(b.cpu(), x.cpu() + scale)
        torch.testing.assert_close(c.cpu(), torch.full((scale,), 2.0))


def test_compile_misaligned_input_written_on_its_stream(stream_device):
    # `base[1:]` is contiguous but not 16-byte aligned, so the graph reads a
    # staged copy. A multi-context region runs on the ambient stream, which
    # never waited for `first`: the copy must still follow the write there.
    first = torch.Stream(device=stream_device)
    second = torch.Stream(device=stream_device)

    def fn(v, w):
        current = current_stream(v.device)
        second.wait_stream(current)
        with stream_context(first, stream_device):
            a = v * 2
        with stream_context(second, stream_device):
            b = w.cos()
        current.wait_stream(first)
        current.wait_stream(second)
        return a, b

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    base = torch.zeros(4097, device=stream_device)
    view = base[1:]
    w = torch.randn(64, device=stream_device)
    for value in (3, 5, 7):
        first.wait_stream(current_stream(view.device))
        with stream_context(first, stream_device):
            slow_work(stream_device)
            view.fill_(value)
        a, b = compiled(view, w)
        torch.testing.assert_close(a.cpu(), torch.full((4096,), 2.0 * value))
        torch.testing.assert_close(b.cpu(), w.cpu().cos())


def test_compile_mutation_in_multi_context_region(stream_device):
    # AOT autograd writes the mutated input back after the graph, on the
    # ambient stream; a later reader on the mutation's stream must see it.
    first = torch.Stream(device=stream_device)
    second = torch.Stream(device=stream_device)

    def fn(x, w):
        current = current_stream(x.device)
        first.wait_stream(current)
        second.wait_stream(current)
        with stream_context(first, stream_device):
            x.mul_(2)
        with stream_context(second, stream_device):
            b = w.cos()
        current.wait_stream(second)
        return b

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    # Large enough that the copy back is still running when the reader starts.
    x = torch.ones(1 << 26, device=stream_device)
    w = torch.randn(64, device=stream_device)
    for expected in (2.0, 4.0, 8.0):
        b = compiled(x, w)
        with stream_context(first, stream_device):
            tail = x[-4096:].clone()
        current_stream(x.device).wait_stream(first)
        torch.testing.assert_close(tail.cpu(), torch.full((4096,), expected))
        torch.testing.assert_close(b.cpu(), w.cpu().cos())


@pytest.mark.parametrize("explicit_device", [False, True])
def test_cuda_current_stream_after_mojo_registration(mojo_gpu, explicit_device):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA torch alongside mojo")
    side = torch.cuda.Stream()

    def fn(x):
        current = (
            torch.cuda.current_stream(x.device)
            if explicit_device
            else torch.cuda.current_stream()
        )
        side.wait_stream(current)
        with torch.cuda.stream(side):
            y = x + 2
        current.wait_stream(side)
        return y

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    for value in (1.0, 4.0):
        x = torch.full((32,), value, device="cuda")
        torch.testing.assert_close(compiled(x).cpu(), x.cpu() + 2)


@pytest.mark.parametrize("outer_device", ["cuda:0", "mojo:0"])
def test_compile_nested_streams_on_different_devices(mojo_gpu, outer_device):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA torch alongside mojo")
    skip_if_metal(mojo_gpu, "Metal has no independent streams/events")
    inner_device = "mojo:0" if outer_device == "cuda:0" else "cuda:0"
    outer = torch.Stream(device=outer_device)
    inner = torch.Stream(device=inner_device)

    def fn(x):
        with stream_context(outer, outer_device):
            with stream_context(inner, inner_device):
                y = x + 1
        current_stream(x.device).wait_stream(outer)
        return y

    x = torch.ones(32, device=outer_device)
    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    compiled(x).cpu()
    for value in (7, 11):
        outer.wait_stream(current_stream(x.device))
        with stream_context(outer, outer_device):
            work = torch.ones(1024, 1024, device=outer_device)
            for _ in range(32):
                work = work @ work
            x.fill_(value)
        torch.testing.assert_close(compiled(x).cpu(), torch.full((32,), value + 1.0))
        outer.synchronize()


def test_compile_stream_restored_on_runtime_error(stream_device, monkeypatch):
    side = torch.Stream(device=stream_device)
    original = compiler.BaseMaxCompiler.__call__
    fail = False

    def run(self, *args):
        if fail:
            raise RuntimeError("stream region failed")
        return original(self, *args)

    monkeypatch.setattr(compiler.BaseMaxCompiler, "__call__", run)

    def fn(x):
        with stream_context(side, stream_device):
            return x + 1

    x = torch.ones(32, device=stream_device)
    initial = current_stream(x.device)
    side.wait_stream(initial)
    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    result = compiled(x)
    initial.wait_stream(side)
    torch.testing.assert_close(result.cpu(), x.cpu() + 1)
    fail = True
    with pytest.raises(RuntimeError, match="stream region failed"):
        compiled(x)
    assert current_stream(x.device) == initial
