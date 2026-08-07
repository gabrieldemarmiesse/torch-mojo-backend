import io
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch._dynamo import mark_dynamic
from torch._dynamo.exc import BackendCompilerFailed
from torch.ops import aten

import torch_mojo_backend
import torch_mojo_backend.torch_compile_backend.compiler
from torch_mojo_backend import (
    MAPPING_TORCH_ATEN_TO_MOJO,
    make_torch_op_from_mojo,
    mojo_backend,
)
from torch_mojo_backend.testing import check_functions_are_equivalent

from .conftest import require_cuda_autograd


# MAX lowers an fp32 matmul to TF32 tensor cores on NVIDIA GPUs while torch
# eager defaults to full fp32, so a graph containing a matmul cannot be
# compared against eager at assert_close's fp32 defaults on GPU
# (`test_compile_matmul` in test_compile_mojo_device.py makes the same
# allowance). Verified exactly: for `x @ w + b` on cuda the backend's output
# is bit-identical to torch's own `allow_tf32=True` result.
#
# The numbers below are the measured tf32-vs-fp32 envelope for these shapes
# over 2000 random draws: max absolute gap 4.8e-3 for one matmul and 1.5e-2
# for the chained pair in `test_get_attr_multiple_parameters`. The relative
# gap is unbounded (outputs cancel to near zero), which is why atol carries
# the tolerance. atol=2e-2 is ~1.3x the measured worst case and still ~50x
# below the O(1) error an actually wrong matmul produces on N(0, 1) data.
# CPU keeps assert_close's exact fp32 defaults.
def matmul_tolerance(device: str) -> dict[str, float]:
    if device == "cpu":
        return {}
    return {"rtol": 1e-2, "atol": 2e-2}


def test_basic_training(device: str):
    require_cuda_autograd(device)

    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

        def forward(self, x):
            return self.linear(x)

    model = MyModel().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def train_step(x, y):
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = F.mse_loss(output, y)
        loss.backward()
        optimizer.step()
        return loss

    a = torch.randn(5, 3).to(device)
    b = torch.randn(5, 2).to(device)

    # We need to reset the parameters before each test
    # to check the model weights afterwards
    model.linear.weight.data.fill_(0.01)
    model.linear.bias.data.fill_(0.01)

    loss_not_compiled = train_step(a, b).cpu().detach().numpy()
    weight_not_compiled = model.linear.weight.data.cpu().numpy()
    bias_not_compiled = model.linear.bias.data.cpu().numpy()

    # Now with the default backed
    model.linear.weight.data.fill_(0.01)
    model.linear.bias.data.fill_(0.01)

    loss_compiled_default = torch.compile()(train_step)(a, b).cpu().detach().numpy()
    weight_compiled_default = model.linear.weight.data.cpu().numpy()
    bias_compiled_default = model.linear.bias.data.cpu().numpy()

    np.testing.assert_allclose(
        loss_not_compiled, loss_compiled_default, rtol=5e-2, atol=5e-3
    )
    np.testing.assert_allclose(
        weight_not_compiled, weight_compiled_default, rtol=5e-2, atol=5e-3
    )
    np.testing.assert_allclose(
        bias_not_compiled, bias_compiled_default, rtol=5e-2, atol=5e-3
    )

    model.linear.weight.data.fill_(0.01)
    model.linear.bias.data.fill_(0.01)

    loss_compiled = (
        torch.compile(backend=mojo_backend)(train_step)(a, b).cpu().detach().numpy()
    )
    weight_compiled = model.linear.weight.data.cpu().numpy()
    bias_compiled = model.linear.bias.data.cpu().numpy()

    np.testing.assert_allclose(loss_not_compiled, loss_compiled, rtol=5e-2, atol=5e-3)
    np.testing.assert_allclose(
        weight_not_compiled, weight_compiled, rtol=5e-2, atol=5e-3
    )
    np.testing.assert_allclose(bias_not_compiled, bias_compiled, rtol=5e-2, atol=5e-3)


def test_get_attr_parameter(device: str):
    """Test get_attr node with parameter access"""
    require_cuda_autograd(device)

    class ParameterModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(3, 4))
            self.bias = torch.nn.Parameter(torch.randn(4))

        def forward(self, x):
            # This will create get_attr nodes for self.weight and self.bias
            return x @ self.weight + self.bias

    module = ParameterModule().to(device)

    x = torch.randn(2, 3)

    # Verify get_attr nodes are in the graph
    # Test with tracing to ensure get_attr nodes are created
    traced = torch.fx.symbolic_trace(module)
    get_attr_nodes = [node for node in traced.graph.nodes if node.op == "get_attr"]
    assert len(get_attr_nodes) >= 2, (
        f"Expected at least 2 get_attr nodes, got {len(get_attr_nodes)}"
    )
    # Should have nodes for weight and bias
    targets = [node.target for node in get_attr_nodes]
    assert "weight" in targets
    assert "bias" in targets

    check_functions_are_equivalent(module, device, [x], **matmul_tolerance(device))


def test_get_attr_nested_parameter(device: str):
    """Test get_attr node with nested module parameter access"""
    require_cuda_autograd(device)

    class NestedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 4)
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, x):
            # This will create get_attr nodes for nested parameters
            return self.linear(x) * self.scale

    module = NestedModule().to(device)

    x = torch.randn(2, 3)

    # Verify get_attr nodes are in the graph
    traced = torch.fx.symbolic_trace(module)
    get_attr_nodes = [node for node in traced.graph.nodes if node.op == "get_attr"]
    # Should have at least the scale parameter as get_attr
    # Linear might be optimized into call_module instead
    targets = [node.target for node in get_attr_nodes]
    assert "scale" in targets

    check_functions_are_equivalent(module, device, [x], **matmul_tolerance(device))


def test_get_attr_buffer(device: str):
    """Test get_attr node with buffer access"""
    require_cuda_autograd(device)

    class ModuleWithBuffer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("running_mean", torch.zeros(4))
            self.weight = torch.nn.Parameter(torch.ones(4))

        def forward(self, x):
            # This will create get_attr nodes for both parameter and buffer
            return (x + self.running_mean) * self.weight

    module = ModuleWithBuffer().to(device)

    x = torch.randn(2, 4)

    # Verify get_attr nodes are in the graph
    traced = torch.fx.symbolic_trace(module)
    get_attr_nodes = [node for node in traced.graph.nodes if node.op == "get_attr"]
    targets = [node.target for node in get_attr_nodes]
    # Should have weight and running_mean
    assert "weight" in targets
    assert "running_mean" in targets

    check_functions_are_equivalent(module, device, [x])


def test_get_attr_multiple_parameters(device: str):
    """Test get_attr nodes with multiple parameters"""
    require_cuda_autograd(device)

    class MultiParamModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight1 = torch.nn.Parameter(torch.randn(3, 4))
            self.weight2 = torch.nn.Parameter(torch.randn(4, 2))
            self.bias1 = torch.nn.Parameter(torch.randn(4))
            self.bias2 = torch.nn.Parameter(torch.randn(2))

        def forward(self, x):
            # Multiple get_attr nodes will be created
            h = x @ self.weight1 + self.bias1
            return h @ self.weight2 + self.bias2

    module = MultiParamModule().to(device)

    x = torch.randn(2, 3)

    check_functions_are_equivalent(module, device, [x], **matmul_tolerance(device))


def test_get_attr_with_arithmetic(device: str):
    """Test get_attr nodes combined with arithmetic operations"""
    require_cuda_autograd(device)

    class ArithmeticModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(3.0))
            self.offset = torch.nn.Parameter(torch.tensor(1.5))

        def forward(self, x, y):
            # get_attr nodes will be used for scale and offset
            return (x * self.scale + self.offset) + y

    module = ArithmeticModule().to(device)

    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    check_functions_are_equivalent(module, device, [x, y])


def test_get_attr_constant_tensor(device: str):
    """Test get_attr node with constant tensor"""

    class ConstantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Register a constant tensor (not a parameter)
            self.register_buffer(
                "constant", torch.tensor([1.0, 2.0, 3.0]), persistent=False
            )

        def forward(self, x):
            # This will create a get_attr node for the constant
            return x + self.constant

    module = ConstantModule().to(device)

    x = torch.randn(2, 3)

    check_functions_are_equivalent(module, device, [x])


def test_get_attr_deeply_nested(device: str):
    """Test get_attr node with deeply nested module hierarchy"""
    require_cuda_autograd(device)

    class InnerModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_weight = torch.nn.Parameter(torch.randn(3, 3))

    class MiddleModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = InnerModule()
            self.middle_bias = torch.nn.Parameter(torch.randn(3))

    class OuterModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.middle = MiddleModule()

        def forward(self, x):
            # This will create get_attr nodes with dotted paths
            return x @ self.middle.inner.inner_weight + self.middle.middle_bias

    module = OuterModule().to(device)

    x = torch.randn(2, 3)

    check_functions_are_equivalent(module, device, [x], **matmul_tolerance(device))


def test_get_attr_mixed_with_functions(device: str):
    """Test get_attr nodes mixed with function calls"""
    require_cuda_autograd(device)

    class MixedModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(3, 4))

        def forward(self, x):
            # Mix get_attr with function calls
            linear_out = x @ self.weight
            return torch.sin(linear_out) + torch.cos(linear_out)

    module = MixedModule().to(device)

    x = torch.randn(2, 3)

    check_functions_are_equivalent(module, device, [x], **matmul_tolerance(device))


def test_get_attr_simple_constant(device: str):
    """Test get_attr with a simple constant parameter"""
    require_cuda_autograd(device)

    class SimpleConstantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Create a simple parameter that will definitely create get_attr
            self.constant = torch.nn.Parameter(torch.tensor([2.0, 3.0, 4.0]))

        def forward(self, x):
            # Simple addition that should create get_attr node
            return x + self.constant

    module = SimpleConstantModule().to(device)

    x = torch.randn(3)

    # Verify get_attr nodes are in the graph
    traced = torch.fx.symbolic_trace(module)
    get_attr_nodes = [node for node in traced.graph.nodes if node.op == "get_attr"]
    assert len(get_attr_nodes) >= 1
    targets = [node.target for node in get_attr_nodes]
    assert "constant" in targets

    check_functions_are_equivalent(module, device, [x])


def test_get_attr_torch_tensor(device: str):
    class SimpleConstantModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.constant = torch.tensor([2.0, 3.0, 4.0]).to(device)

        def forward(self, x):
            # Simple addition that should create get_attr node
            return x + self.constant

    module = SimpleConstantModule().to(device)

    x = torch.randn(3)

    # Verify get_attr nodes are in the graph
    traced = torch.fx.symbolic_trace(module)
    get_attr_nodes = [node for node in traced.graph.nodes if node.op == "get_attr"]
    assert len(get_attr_nodes) >= 1
    targets = [node.target for node in get_attr_nodes]
    assert "constant" in targets

    check_functions_are_equivalent(module, device, [x])


# Graph Break Tests
def test_graph_break_with_print(device: str):
    """Test graph break caused by print statements"""

    def fn_with_print(x):
        a = x + 1
        print(f"Processing tensor with shape: {x.shape}")
        return a * 2

    x = torch.randn(3, 4)
    explanation = torch._dynamo.explain(fn_with_print)(x)
    assert explanation.graph_break_count == 1
    assert explanation.graph_count == 2

    # This should cause a graph break due to print
    with patch("sys.stdout", new_callable=io.StringIO):
        check_functions_are_equivalent(fn_with_print, device, [x])


def test_graph_break_with_item_access(device: str):
    def fn_with_item(x):
        x = x * x
        if x[0, 0] > 0:
            return x * 2
        else:
            return x

    x = torch.randn(2, 3) + 1.0  # Ensure non-zero values
    explanation = torch._dynamo.explain(fn_with_item)(x)
    assert explanation.graph_break_count == 1
    assert explanation.graph_count == 2
    check_functions_are_equivalent(fn_with_item, device, [x])


def test_graph_break_with_python_loop_over_tensor(device: str):
    """Test graph break caused by Python loops over tensor elements"""

    def fn_with_python_loop(x):
        x = x * x
        # Python iteration over tensor shapes causes graph breaks
        result = x
        for i in range(int(x[0, 0])):  # This will cause graph break
            result = result * (i + 1)
        return result

    x = torch.randint(1, 3, (3, 2)).to(torch.float32)
    explanation = torch._dynamo.explain(fn_with_python_loop)(x)
    assert explanation.graph_break_count == 1
    assert explanation.graph_count == 2
    check_functions_are_equivalent(fn_with_python_loop, device, [x])


def test_graph_break_with_python_loop_over_tensor_complexe_dtypes(device: str):
    """Test graph break caused by Python loops over tensor elements"""

    def fn_with_python_loop(x):
        x = x * x
        result = x
        for i in range(int(x[0, 0])):  # This will cause graph break
            result = (result * (i + 1)).to(torch.int32)
        return result

    x = torch.randint(1, 3, (3, 2)).to(torch.int32)
    explanation = torch._dynamo.explain(fn_with_python_loop)(x)
    # In theory there should be only one graph break, but for some reason
    # because of the mojo_device we get one per loop. Worth investigating.
    assert explanation.graph_break_count >= 1
    assert explanation.graph_count >= 2
    check_functions_are_equivalent(fn_with_python_loop, device, [x])


def test_graph_break_with_string_operations(device: str):
    """Test graph break caused by string operations"""

    def fn_with_string_ops(x):
        x = x * 2
        tensor_info = f"Tensor shape: {x}, dtype: {x.dtype}"
        # Just return the tensor since we can't return strings
        return x * (len(tensor_info) % 10)

    x = torch.randn(2, 3)
    explanation = torch._dynamo.explain(fn_with_string_ops)(x)
    assert explanation.graph_break_count == 1
    assert explanation.graph_count == 2
    # This should cause graph breaks due to string operations
    check_functions_are_equivalent(fn_with_string_ops, device, [x])


def test_multiple_graph_breaks_in_sequence(device: str):
    """Test function with multiple operations that cause graph breaks"""

    def fn_with_multiple_breaks(x):
        # First graph break: print
        x = x * x
        print(f"Input shape: {x.shape}")

        x = x + 1

        print(f"Result computed {x.shape}")

        return x * x

    x = torch.randn(2, 3)
    explanation = torch._dynamo.explain(fn_with_multiple_breaks)(x)
    assert explanation.graph_break_count == 2
    assert explanation.graph_count == 3

    with patch("sys.stdout", new_callable=io.StringIO):
        check_functions_are_equivalent(fn_with_multiple_breaks, device, [x])


def test_no_graph_breaks_with_supported_operations(device: str):
    def well_supported_fn(x, y):
        # Only use operations that should be well supported
        z = x + y
        z = torch.sin(z)
        z = torch.cos(z)
        z = z * 2
        z = torch.abs(z)
        return z

    x = torch.randn(3, 4)
    y = torch.randn(3, 4)
    explanation = torch._dynamo.explain(well_supported_fn)(x, y)
    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1
    check_functions_are_equivalent(well_supported_fn, device, [x, y])


class mojo_backendCallCount:
    def __init__(self, compiler):
        self.call_count = 0
        self.compiler = compiler

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        return self.compiler(*args, **kwargs)


def test_dynamic_shapes(device: str):
    """Testing the behavior with mark_dynamic()."""

    def fn(x, y):
        return x + y

    counter = mojo_backendCallCount(mojo_backend)
    fn_compiled = torch.compile(backend=counter)(fn)

    a = torch.randn(20, 2).to(device)
    b = torch.randn(2).to(device)

    mark_dynamic(a, 0)

    check_functions_are_equivalent(fn, None, [a, b], fn_compiled)

    for i in range(5, 15):
        a = torch.randn(i, 2).to(device)
        b = torch.randn(2).to(device)
        mark_dynamic(a, 0)

        check_functions_are_equivalent(fn, None, [a, b], fn_compiled)
        # Ensure only one instance of the mojo_backend is created
    assert counter.call_count == 1


def test_recompilation(device: str):
    """Testing the behavior without mark_dynamic()."""

    def fn(x, y):
        return x + y

    counter = mojo_backendCallCount(mojo_backend)
    fn_compiled = torch.compile(backend=counter)(fn)

    a = torch.randn(20, 2).to(device)
    b = torch.randn(2).to(device)

    check_functions_are_equivalent(fn, None, [a, b], fn_compiled)

    a = torch.randn(10, 2).to(device)
    b = torch.randn(2).to(device)

    check_functions_are_equivalent(fn, None, [a, b], fn_compiled)
    # Ensure a second instance of the mojo_backend is created
    assert counter.call_count == 2

    # TODO: Make it work if called with more shapes (dynamo doesn't recompile)


def test_error_message_exception_in_op(monkeypatch):
    def not_working_add(x, y):
        raise RuntimeError("Ho no crash!")

    monkeypatch.setitem(MAPPING_TORCH_ATEN_TO_MOJO, aten.add, not_working_add)

    def fn(x, y):
        return x + y

    with pytest.raises(RuntimeError) as exc_info:
        torch.compile(backend=mojo_backend)(fn)(torch.randn(2, 3), torch.randn(2, 3))

    assert "return x + y" in str(exc_info.value)
    assert "Ho no crash!" in str(exc_info.value)
    assert "torch._ops.aten.aten::add" in str(exc_info.value)
    assert "https://github.com/gabrieldemarmiesse/torch-mojo-backend/issues" in str(
        exc_info.value
    )
    assert "not_working_add" in str(exc_info.value)


def test_error_message_op_not_supported(monkeypatch):
    monkeypatch.delitem(MAPPING_TORCH_ATEN_TO_MOJO, aten.add)

    def fn(x, y):
        return x + y

    with pytest.raises(BackendCompilerFailed) as exc_info:
        torch.compile(backend=mojo_backend)(fn)(torch.randn(2, 3), torch.randn(2, 3))

    assert "return x + y" in str(exc_info.value)
    assert "torch._ops.aten.aten::add" in str(exc_info.value)
    assert "https://github.com/gabrieldemarmiesse/torch-mojo-backend/issues" in str(
        exc_info.value
    )
    assert "is not supported" in str(exc_info.value)


def test_bug_keyerror_input(device: str):
    """Test a specific bug where KeyError occurs in input handling"""

    def fn(x):
        y = torch.arange(0, x.shape[1], 1, dtype=x.dtype, device=x.device)
        z = y[None, :]
        return x + z

    # Create inputs
    x = torch.randn(2, 5)

    mark_dynamic(x, 1)

    check_functions_are_equivalent(fn, device, [x])


def test_symint_scalar_arithmetic(device: str):
    """A symbolic dim used as a scalar in tensor arithmetic (dynamic shapes)."""

    def fn(x):
        return torch.relu(x) + x.shape[0]

    fn_compiled = torch.compile(backend=mojo_backend)(fn)
    for n in (4, 5, 6):
        x = torch.randn(n, 3)
        check_functions_are_equivalent(fn, device, [x], fn_compiled=fn_compiled)


def test_sdpa_with_attention_mask(device: str):
    """An explicit float mask makes cuda pick the mem-efficient sdpa overload."""

    def fn(q, k, v, mask):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    mask = torch.zeros(1, 4, 8, 8).masked_fill(
        torch.rand(1, 4, 8, 8) > 0.7, float("-inf")
    )

    check_functions_are_equivalent(fn, device, [q, k, v, mask], rtol=1e-2, atol=1e-3)


def test_sdpa_decode_gpt2_mask(device: str):
    """GPT-2 decode shape exercises the fused Mojo graph custom op on GPU."""

    def fn(q, k, v, mask):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.125)

    q = torch.randn(4, 12, 1, 64)
    k = torch.randn(4, 12, 128, 64)
    v = torch.randn(4, 12, 128, 64)
    mask = torch.zeros(4, 1, 1, 128)
    mask[..., -7:] = float("-inf")
    check_functions_are_equivalent(fn, device, [q, k, v, mask], rtol=1e-2, atol=1e-3)


def test_constant_pad_nd(device: str):
    def fn(x):
        return F.pad(x, (1, 2, 0, 3), mode="constant", value=1.5)

    x = torch.randn(2, 3, 4)
    check_functions_are_equivalent(fn, device, [x])


def test_lifted_tensor_constant(device: str):
    """A tensor constant created inside the compiled function (dynamo lifts
    it as a graph get_attr + lift_fresh_copy)."""

    def fn(x):
        return x + torch.tensor([1.0, 2.0, 3.0], device=x.device)

    x = torch.randn(2, 3)
    check_functions_are_equivalent(fn, device, [x])


def test_compiled_input_buffer_cache(device: str):
    """Compiled graph inputs are converted to MAX buffers once and cached
    by tensor identity. In-place updates alias the same memory and must be
    visible; swapping the storage (`t.data = ...`) must invalidate."""

    def fn(x, w):
        return x @ w

    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=True)
    x = torch.randn(2, 3, device=device)
    w = torch.randn(3, 4, device=device)
    torch.testing.assert_close(
        compiled(x, w).cpu(), x.cpu() @ w.cpu(), rtol=1e-2, atol=1e-3
    )

    # Same tensor objects, mutated in place: the cached buffer aliases them.
    with torch.no_grad():
        w += 1.0
    torch.testing.assert_close(
        compiled(x, w).cpu(), x.cpu() @ w.cpu(), rtol=1e-2, atol=1e-3
    )

    # Storage swap: the data pointer guard must drop the cached buffer.
    w.data = torch.randn(3, 4, device=device)
    torch.testing.assert_close(
        compiled(x, w).cpu(), x.cpu() @ w.cpu(), rtol=1e-2, atol=1e-3
    )


def test_bitwise_and_0d_operand(device: str):
    def fn(a, b):
        return a & b

    a = torch.tensor(True)
    b = torch.tensor([True, False, True])
    check_functions_are_equivalent(fn, device, [a, b])


def test_scalar_as_input():
    def fn(x):
        y = torch.arange(0, x[0], 1, dtype=x.dtype, device=x.device)
        z = y[None, :]
        return x + z

    # Create inputs
    x = torch.randint(1, 10, (1,), dtype=torch.int32, device="cpu")

    mark_dynamic(x, 1)

    check_functions_are_equivalent(fn, None, [x])


def test_decomposition_overload(monkeypatch):
    """We verify that we skip decomposition for ops that are in the decomposition table,
    and that we registered as an OpOverload (here `aten.t.default`).
    """

    def fn(x):
        x = x * 2
        return x.t() * 2

    # grab the input of init_compiler
    input_gm = None
    init_compiler = (
        torch_mojo_backend.torch_compile_backend.compiler.BaseMaxCompiler.__init__
    )

    def fake_init_compiler(self, gm, *args, **kwargs):
        nonlocal input_gm
        input_gm = gm
        return init_compiler(self, gm, *args, **kwargs)

    monkeypatch.setattr(
        torch_mojo_backend.torch_compile_backend.compiler.BaseMaxCompiler,
        "__init__",
        fake_init_compiler,
    )

    a = torch.compile(backend=mojo_backend)(fn)
    a(torch.randn(2, 3))

    # it's normally decomposed. We check that it's not the case since we
    # implemented it ourselves.
    assert aten.t.default in [node.target for node in input_gm.graph.nodes]


def test_decomposition_overload_packet(monkeypatch):
    """We verify that we skip decomposition for ops that are in the decomposition table,
    and that we registered as an OpOverloadPacket (here `aten.transpose`).
    """

    def fn(x):
        x = x * 2
        return torch.transpose(x, 0, 1) * 2

    # grab the input of init_compiler
    input_gm = None
    init_compiler = (
        torch_mojo_backend.torch_compile_backend.compiler.BaseMaxCompiler.__init__
    )

    def fake_init_compiler(self, gm, *args, **kwargs):
        nonlocal input_gm
        input_gm = gm
        return init_compiler(self, gm, *args, **kwargs)

    monkeypatch.setattr(
        torch_mojo_backend.torch_compile_backend.compiler.BaseMaxCompiler,
        "__init__",
        fake_init_compiler,
    )

    a = torch.compile(backend=mojo_backend)(fn)
    a(torch.randn(2, 3))

    # it's normally decomposed. We check that it's not the case since we
    # implemented it ourselves.
    assert aten.transpose.int in [node.target for node in input_gm.graph.nodes]


def allocate_outputs_grayscale(pic: torch.Tensor) -> torch.Tensor:
    return pic.new_empty(pic.shape[:-1], dtype=torch.float32)


def grayscale_eager(pic: torch.Tensor):
    pic = pic.to(dtype=torch.float32)
    r = pic[:, :, 0]
    g = pic[:, :, 1]
    b = pic[:, :, 2]
    return torch.clamp((0.21 * r + 0.71 * g + 0.07 * b), max=255)


def test_mojo_custom_op(device: str):
    img = torch.randn(224, 224, 3, device=device).to(dtype=torch.uint8)

    my_torch_grayscale = make_torch_op_from_mojo(
        Path(__file__).parent / "dummy_mojo_kernels",
        "grayscale",
        allocate_outputs_grayscale,
    )

    x = my_torch_grayscale(img)
    y = grayscale_eager(img)
    torch.testing.assert_close(x, y)
    check_functions_are_equivalent(
        grayscale_eager, None, [img], fn_compiled=my_torch_grayscale
    )

    def more_complexe_graph(x: torch.Tensor):
        x = x + 8
        y = my_torch_grayscale(x)
        y = y - 16
        return y

    def more_complexe_graph_eager(x: torch.Tensor):
        x = x + 8
        y = grayscale_eager(x)
        y = y - 16
        return y

    x = more_complexe_graph(img)
    y = more_complexe_graph_eager(img)

    complexe_graph_compiled = torch.compile(backend=mojo_backend, fullgraph=True)(
        more_complexe_graph
    )
    z = complexe_graph_compiled(img)
    torch.testing.assert_close(x, y)
    torch.testing.assert_close(x, z)

    explanation = torch._dynamo.explain(more_complexe_graph)(img)
    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1

    check_functions_are_equivalent(
        more_complexe_graph_eager, None, [img], fn_compiled=complexe_graph_compiled
    )


def allocate_outputs_grayscale_multi(
    pic: torch.Tensor, noise: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(pic.new_empty(noise.shape, dtype=torch.float32) for _ in range(2))


def grayscale_multi_eager(
    pic: torch.Tensor, noise: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pic = pic.to(dtype=torch.float32)
    r = pic[:, :, 0] + noise
    g = pic[:, :, 1] + noise
    b = pic[:, :, 2] + noise

    return (torch.clamp((0.21 * r + 0.71 * g + 0.07 * b), max=255), r)


def test_mojo_custom_op_multi(device: str):
    img = torch.randn(224, 224, 3, device=device).to(dtype=torch.uint8)
    noise = torch.randint(0, 2, (224, 224), device=device).to(dtype=torch.uint8)

    my_torch_grayscale_multi = make_torch_op_from_mojo(
        Path(__file__).parent / "dummy_mojo_kernels",
        "grayscale_multi",
        allocate_outputs_grayscale_multi,
    )
    x1, x2 = my_torch_grayscale_multi(img, noise)
    y1, y2 = grayscale_multi_eager(img, noise)
    torch.testing.assert_close(x1, y1)
    torch.testing.assert_close(x2, y2)
    check_functions_are_equivalent(
        grayscale_multi_eager, None, [img, noise], fn_compiled=my_torch_grayscale_multi
    )

    def more_complexe_graph(
        x: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + 8
        y1, y2 = my_torch_grayscale_multi(x, noise)
        y1 = y1 - 16
        y2 = y2 + 3
        return y1, y2

    def more_complexe_graph_eager(
        x: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + 8
        y1, y2 = grayscale_multi_eager(x, noise)
        y1 = y1 - 16
        y2 = y2 + 3
        return y1, y2

    x1, x2 = more_complexe_graph(img, noise)
    y1, y2 = more_complexe_graph_eager(img, noise)

    complexe_graph_compiled = torch.compile(backend=mojo_backend, fullgraph=True)(
        more_complexe_graph
    )
    z1, z2 = complexe_graph_compiled(img, noise)
    torch.testing.assert_close(x1, y1)
    torch.testing.assert_close(x1, z1)
    torch.testing.assert_close(x2, y2)
    torch.testing.assert_close(x2, z2)

    explanation = torch._dynamo.explain(more_complexe_graph)(img, noise)
    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1

    check_functions_are_equivalent(
        more_complexe_graph_eager,
        None,
        [img, noise],
        fn_compiled=complexe_graph_compiled,
    )


def test_mojo_custom_op_multi_dynamic_dims(device: str):
    img = torch.randn(224, 224, 3, device=device).to(dtype=torch.uint8)
    mark_dynamic(img, 0)
    mark_dynamic(img, 1)
    noise = torch.randint(0, 2, (224, 224), device=device).to(dtype=torch.uint8)
    mark_dynamic(noise, 0)
    mark_dynamic(noise, 1)
    my_torch_grayscale_multi = make_torch_op_from_mojo(
        Path(__file__).parent / "dummy_mojo_kernels",
        "grayscale_multi",
        allocate_outputs_grayscale_multi,
    )

    x1, x2 = my_torch_grayscale_multi(img, noise)
    y1, y2 = grayscale_multi_eager(img, noise)
    torch.testing.assert_close(x1, y1)
    torch.testing.assert_close(x2, y2)
    check_functions_are_equivalent(
        grayscale_multi_eager, None, [img, noise], fn_compiled=my_torch_grayscale_multi
    )

    def more_complexe_graph(
        x: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + 8
        y1, y2 = my_torch_grayscale_multi(x, noise)
        y1 = y1 - 16
        y2 = y2 + 3
        return y1, y2

    def more_complexe_graph_eager(
        x: torch.Tensor, noise: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + 8
        y1, y2 = grayscale_multi_eager(x, noise)
        y1 = y1 - 16
        y2 = y2 + 3
        return y1, y2

    x1, x2 = more_complexe_graph(img, noise)
    y1, y2 = more_complexe_graph_eager(img, noise)

    complexe_graph_compiled = torch.compile(backend=mojo_backend, fullgraph=True)(
        more_complexe_graph
    )
    z1, z2 = complexe_graph_compiled(img, noise)
    torch.testing.assert_close(x1, y1)
    torch.testing.assert_close(x1, z1)
    torch.testing.assert_close(x2, y2)
    torch.testing.assert_close(x2, z2)

    explanation = torch._dynamo.explain(more_complexe_graph)(img, noise)
    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1

    check_functions_are_equivalent(
        more_complexe_graph_eager,
        None,
        [img, noise],
        fn_compiled=complexe_graph_compiled,
    )
