"""torch.compile with tensors on the mojo (mojo_device) eager device.

The mojo_backend path compiles the traced graph to a single MAX graph and
feeds/adopts mojo tensor memory zero-copy via DLPack. The stock dynamo
backends ("eager", "aot_eager") execute the traced graph through the
eager mojo kernels instead.
"""

import gc

import max.driver
import pytest
import torch

from torch_mojo_backend import TorchMojoTensor, mojo_backend, register_mojo_devices

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_max_device():
    register_mojo_devices()


def assert_close_cpu(out, ref, rtol=1e-4, atol=1e-4):
    assert isinstance(out, TorchMojoTensor)
    assert out.device.type == "mojo"
    assert out.dtype == ref.dtype
    torch.testing.assert_close(out.cpu(), ref, rtol=rtol, atol=atol)


def test_compile_elementwise(mojo_device):
    def fn(x, y):
        return torch.relu(x * y + 1.0) - x

    x = torch.randn(4, 8, device=mojo_device)
    y = torch.randn(4, 8, device=mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x, y)
    assert_close_cpu(out, fn(x.cpu(), y.cpu()))


def test_compile_matmul(mojo_device):
    def fn(x, y):
        return torch.relu(x @ y + 1.0)

    x = torch.randn(4, 8, device=mojo_device)
    y = torch.randn(8, 16, device=mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x, y)
    # Loose tolerance: MAX uses tf32-style matmul on GPU.
    assert_close_cpu(out, fn(x.cpu(), y.cpu()), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int32])
def test_compile_dtypes(mojo_device, dtype):
    def fn(x, y):
        return x + y * 2

    if dtype.is_floating_point:
        x = torch.randn(4, 8, device=mojo_device, dtype=dtype)
        y = torch.randn(4, 8, device=mojo_device, dtype=dtype)
    else:
        x = torch.arange(32, device=mojo_device, dtype=dtype).reshape(4, 8)
        y = torch.arange(32, device=mojo_device, dtype=dtype).reshape(4, 8)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x, y)
    assert_close_cpu(out, fn(x.cpu(), y.cpu()), rtol=1e-2, atol=1e-2)


def test_compile_non_contiguous_input(mojo_device):
    def fn(x):
        return x + 1.0

    x = torch.randn(4, 8, device=mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x.t())
    assert_close_cpu(out, fn(x.cpu().t()))


def test_compile_multiple_outputs(mojo_device):
    def fn(x):
        a = x + 1.0
        return a, None, a.t(), x.sum()

    x = torch.randn(4, 8, device=mojo_device)
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

    x = torch.randn(4, 8, device=mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x)
    eager_result = (out + 1.0).sum()
    assert_close_cpu(eager_result, (x.cpu() * 2.0 + 1.0).sum())


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
        x = torch.randn(n, 3, device=mojo_device)
        assert_close_cpu(compiled(x), fn(x.cpu()))


def test_compile_shape_int_output(mojo_device):
    def fn(x):
        return x + 1.0, x.shape[0] * 2

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True, dynamic=True)
    x = torch.randn(7, 3, device=mojo_device)
    out, dim = compiled(x)
    assert dim == 14
    assert_close_cpu(out, x.cpu() + 1.0)


def test_compile_input_mutated_between_calls(mojo_device):
    """The cross-call buffer cache aliases input memory: in-place updates
    between calls (optimizer-step pattern) must be visible to the graph."""

    def fn(x, w):
        return x @ w

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    x = torch.randn(2, 3, device=mojo_device)
    w = torch.randn(3, 4, device=mojo_device)
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

    x = torch.randn(2, 3, device=mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x)
    assert_close_cpu(out, x.cpu() + torch.tensor([1.0, 2.0, 3.0]))


def test_compile_symint_arithmetic(mojo_device):
    """Symbolic dims used as scalars in tensor arithmetic (dynamic shapes)."""

    def fn(x):
        return torch.relu(x) + x.shape[0]

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    for n in (4, 5, 6):
        x = torch.randn(n, 3, device=mojo_device)
        assert_close_cpu(compiled(x), fn(x.cpu()))


def test_compile_factory_function(mojo_device):
    def fn(x, device):
        return x + torch.ones(4, 8, device=device)

    x = torch.randn(4, 8, device=mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x, mojo_device)
    assert_close_cpu(out, x.cpu() + torch.ones(4, 8))


def test_compile_device_attribute(mojo_device):
    """Reading `x.device` in compiled code (the ubiquitous
    `torch.arange(T, device=idx.device)` pattern) traces through
    TorchMojoTensor's `device` property without a graph break."""

    def fn(x):
        return x + torch.ones(4, 8, device=x.device)

    x = torch.randn(4, 8, device=mojo_device)
    out = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x)
    assert_close_cpu(out, x.cpu() + torch.ones(4, 8))


def test_compile_backward(mojo_device):
    def fn(x, w):
        return ((x @ w).relu() ** 2).sum()

    x = torch.randn(4, 8, device=mojo_device)
    w = torch.randn(8, 3, device=mojo_device, requires_grad=True)
    loss = torch.compile(fn, backend=mojo_backend, fullgraph=True)(x, w)
    loss.backward()
    assert isinstance(w.grad, TorchMojoTensor)
    assert w.grad.device.type == "mojo"

    x_cpu = x.cpu().detach()
    w_cpu = w.cpu().detach().requires_grad_(True)
    fn(x_cpu, w_cpu).backward()
    torch.testing.assert_close(w.grad.cpu(), w_cpu.grad, rtol=2e-2, atol=2e-3)


def test_compile_recompiles_for_cpu_inputs(mojo_device):
    """The same compiled function serves mojo and cpu inputs (device guard)."""

    def fn(x):
        return x * 3.0

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    x = torch.randn(4, 8)
    out_mojo = compiled(x.to(mojo_device))
    out_cpu = compiled(x)
    assert isinstance(out_mojo, TorchMojoTensor)
    assert not isinstance(out_cpu, TorchMojoTensor)
    assert out_cpu.device.type == "cpu"
    torch.testing.assert_close(out_mojo.cpu(), out_cpu)


def test_compile_eager_backend(mojo_device):
    """Dynamo-only backend: the traced graph runs through the eager kernels."""

    def fn(x):
        return x * 3.0 - 1.0

    x = torch.randn(4, 8, device=mojo_device)
    out = torch.compile(fn, backend="eager", fullgraph=True)(x)
    assert_close_cpu(out, fn(x.cpu()))


def test_compile_aot_eager_backend(mojo_device):
    """aot_eager runs the functionalized aten graph through the eager kernels."""

    def fn(x, y):
        return torch.relu(x @ y + 1.0)

    x = torch.randn(4, 8, device=mojo_device)
    y = torch.randn(8, 16, device=mojo_device)
    out = torch.compile(fn, backend="aot_eager", fullgraph=True)(x, y)
    assert_close_cpu(out, fn(x.cpu(), y.cpu()), rtol=1e-2, atol=1e-2)


class _MiniAttentionBlock(torch.nn.Module):
    """Attention + MLP block exercising embeddings, layernorm, views,
    transposes, masked softmax and matmuls in one compiled graph."""

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
    """The DLPack capsule must pin the mojo allocation for the consumer."""
    x = torch.arange(100, device=mojo_device, dtype=torch.float32)
    expected = x.cpu()
    buffer = max.driver.Buffer.from_dlpack(x)
    del x
    gc.collect()
    # Churn some allocations to surface use-after-free if the pin is broken.
    for _ in range(4):
        _ = torch.randn(100, device=mojo_device)
    roundtrip = torch.from_dlpack(buffer.to(max.driver.CPU()))
    torch.testing.assert_close(roundtrip, expected)
