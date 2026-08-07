"""
Comprehensive unit tests for MNIST SimpleNet model.

This module provides thorough testing of the MNIST model implementation,
including forward/backward passes, compilation, training steps, and
integration with the Mojo backend.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_mojo_backend import mojo_backend
from torch_mojo_backend.testing import check_functions_are_equivalent

from ..conftest import require_cuda_autograd

# TF32 keeps 10 explicit mantissa bits, so its unit roundoff is 2**-11.
_TF32_UNIT_ROUNDOFF = 2.0**-11

# Covers the maximum over the output elements being compared: the largest of 80
# draws from a Gaussian sits near sqrt(2 * ln(80)) = 3.0 sigma, and this test
# feeds unseeded inputs, so take a bit over twice that.
_TF32_MAX_ELEMENT_FACTOR = 8.0


def _tf32_chain_atol(model: nn.Module) -> float:
    """Absolute error budget for ``model``'s logits on an NVIDIA GPU.

    MAX lowers a float32 matmul to ``rmo.matmul``, whose NVIDIA kernel rounds
    both operands to TF32 and accumulates in fp32; ``max.graph.ops.matmul``
    exposes no precision argument, so there is nothing to switch off. torch
    eager meanwhile keeps ``torch.get_float32_matmul_precision() == "highest"``,
    i.e. IEEE fp32, so on this device the two are running genuinely different
    arithmetic and are *expected* to disagree. Measured on an H100: the compiled
    ``fc2`` (K=128) and ``fc3`` (K=64) outputs are bit-identical -- 0 ulp on
    every element -- to torch's own once ``torch.backends.cuda.matmul.allow_tf32``
    is turned on, which is what says the lowering is right and only the
    arithmetic is coarser. ``fc1`` (K=784) spans several k-tiles so its
    partial-sum order differs, but it still lands ~150x closer to torch's TF32
    result than TF32 is to IEEE.

    For one dot product ``y = sum_k w_k a_k``, rounding both operands gives
    ``dy = sum_k w_k a_k (d_k + e_k)`` with ``|d|, |e| <= u = 2**-11``,
    independent round-to-nearest errors of variance ``u**2 / 3`` each, so
    ``sd(dy) = u * sqrt(2/3) * ||w (*) a||_2``. The terms carry mixed signs, so
    ``y`` is itself a random walk of that same norm, ``sd(y) = ||w (*) a||_2``.
    One GEMM therefore perturbs its output by ``u * sqrt(2/3)`` *relative to
    that output's own scale*, independently of K.

    ReLU is 1-Lipschitz and every layer here is contractive (gain
    ``sqrt(sum w**2) = sqrt(1/3) < 1``), so the L per-layer injections add in
    quadrature at worst: ``sqrt(L) * u * sqrt(2/3) * sd(logits)``. The logit
    scale follows from ``nn.Linear``'s default init ``W ~ U(+-1/sqrt(K))`` of
    variance ``1/(3K)``: each layer multiplies the activation variance by
    ``K * 1/(3K) = 1/3`` and each ReLU halves it, so from ``x ~ N(0, 1)`` the
    chain runs ``1 -> 1/3 -> 1/6 -> 1/18 -> 1/36 -> 1/108``.

    That yields 5.3e-4 for SimpleNet. Over 150 random draws the worst error
    actually observed was 2.2e-4, so the budget sits 2.4x clear; emulating TF32
    by masking the low 13 mantissa bits in torch reproduces 2.0e-4, confirming
    operand rounding accounts for essentially all of the gap. It stays tight
    enough to fail on a bug: replaying those draws through a forward pass
    missing fc1's bias overshoots the budget by 27x, missing fc3's bias by 235x,
    missing the second ReLU by 649x, and silently downgrading the GEMMs from
    TF32 to bfloat16 by 2.8x.
    """
    linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
    activation_var = 1.0  # the inputs are N(0, 1)
    for position, _ in enumerate(linears):
        activation_var /= 3.0  # K * Var(U(+-1/sqrt(K))) = K * 1/(3K)
        if position < len(linears) - 1:
            activation_var /= 2.0  # ReLU zeroes half the units
    per_gemm_relative_sd = _TF32_UNIT_ROUNDOFF * math.sqrt(2.0 / 3.0)
    return (
        _TF32_MAX_ELEMENT_FACTOR
        * math.sqrt(len(linears))
        * per_gemm_relative_sd
        * math.sqrt(activation_var)
    )


class SimpleNet(nn.Module):
    """Simple feedforward neural network for MNIST classification."""

    def __init__(self):
        super().__init__()
        # Input: 28x28 = 784 pixels
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes (digits 0-9)

    def forward(self, x):
        # Flatten the input
        x = x.view(-1, 784)

        # Hidden layers with RELU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer (no activation, will use CrossEntropyLoss)
        x = self.fc3(x)
        return x


class TestMNISTForwardPass:
    """Tests for MNIST model forward pass."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32, 64])
    def test_forward_output_shape(self, batch_size: int, device: str):
        """Test forward pass produces correct output shape for various batch sizes."""
        model = SimpleNet().to(device)
        x = torch.randn(batch_size, 1, 28, 28).to(device)

        output = model(x)

        assert output.shape == (batch_size, 10), (
            f"Expected shape ({batch_size}, 10), got {output.shape}"
        )
        assert output.device.type == torch.device(device).type

    def test_forward_output_type(self, device: str):
        """Test that forward pass returns correct tensor type."""
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)

        output = model(x)

        assert isinstance(output, torch.Tensor)
        assert output.dtype == torch.float32

    def test_forward_deterministic(self, device: str):
        """Test that forward pass is deterministic for same input."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(8, 1, 28, 28).to(device)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        torch.testing.assert_close(output1, output2)

    def test_forward_different_inputs_different_outputs(self, device: str):
        """Test that different inputs produce different outputs."""
        model = SimpleNet().to(device)
        model.eval()

        x1 = torch.randn(8, 1, 28, 28).to(device)
        x2 = torch.randn(8, 1, 28, 28).to(device)

        with torch.no_grad():
            output1 = model(x1)
            output2 = model(x2)

        # Outputs should be different
        assert not torch.allclose(output1, output2)

    def test_forward_with_zero_input(self, device: str):
        """Test forward pass with zero input."""
        model = SimpleNet().to(device)
        x = torch.zeros(8, 1, 28, 28).to(device)

        output = model(x)

        assert output.shape == (8, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_forward_with_ones_input(self, device: str):
        """Test forward pass with ones input."""
        model = SimpleNet().to(device)
        x = torch.ones(8, 1, 28, 28).to(device)

        output = model(x)

        assert output.shape == (8, 10)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestMNISTBackwardPass:
    """Tests for MNIST model backward pass and gradients."""

    def test_backward_computes_gradients(self, device: str):
        """Test that backward pass computes gradients."""
        require_cuda_autograd(device)
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (8,)).to(device)

        criterion = nn.CrossEntropyLoss()
        model.train()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Verify gradients are computed for all parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Gradient not computed for {name}"
            assert not torch.all(param.grad == 0), f"All gradients are zero for {name}"

    def test_gradient_flow_through_layers(self, device: str):
        """Test that gradients flow through all layers."""
        require_cuda_autograd(device)
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (8,)).to(device)

        criterion = nn.CrossEntropyLoss()
        model.train()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Check gradients exist for each layer
        assert model.fc1.weight.grad is not None
        assert model.fc1.bias.grad is not None
        assert model.fc2.weight.grad is not None
        assert model.fc2.bias.grad is not None
        assert model.fc3.weight.grad is not None
        assert model.fc3.bias.grad is not None

    def test_zero_grad_clears_gradients(self, device: str):
        """Test that optimizer.zero_grad() clears gradients."""
        require_cuda_autograd(device)
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (8,)).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # First forward-backward pass
        output = model(x)
        loss = criterion(output, y)
        loss.backward()

        # Clear gradients
        optimizer.zero_grad()

        # Verify all gradients are zero or None
        for param in model.parameters():
            if param.grad is not None:
                assert torch.all(param.grad == 0), "Gradients were not zeroed"


class TestMNISTCompilation:
    """Tests for MNIST model compilation with the Mojo backend."""

    def test_model_compiles_successfully(self, device: str):
        """Test that SimpleNet compiles successfully with mojo_backend."""
        require_cuda_autograd(device)
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)

        check_functions_are_equivalent(model, device, [x], atol=1e-3, rtol=1e-3)

    def test_compiled_no_graph_breaks(self, device: str):
        """Test that SimpleNet compiles without graph breaks (fullgraph=True)."""
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)

        explanation = torch._dynamo.explain(model)(x)

        assert explanation.graph_break_count == 0, (
            f"Expected 0 graph breaks, got {explanation.graph_break_count}"
        )
        assert explanation.graph_count == 1, (
            f"Expected 1 graph, got {explanation.graph_count}"
        )

    def test_compiled_model_forward(self, device: str):
        """Test that compiled model produces same output as eager."""
        model = SimpleNet().to(device)
        model.eval()
        x = torch.randn(8, 1, 28, 28).to(device)

        # Eager execution
        with torch.no_grad():
            output_eager = model(x)

        # Compiled execution
        compiled_model = torch.compile(model, backend=mojo_backend, fullgraph=True)
        with torch.no_grad():
            output_compiled = compiled_model(x)

        # Verify outputs match. On NVIDIA the backend's GEMMs run on TF32 tensor
        # cores while torch eager stays on IEEE fp32, so the absolute budget is
        # widened to the size that difference can reach; see _tf32_chain_atol.
        # rtol is left alone -- cancellation leaves the relative error of a
        # near-zero logit unbounded (2.2e-2 was seen on one element), so it is
        # atol that has to carry this, not rtol.
        atol = _tf32_chain_atol(model) if device.startswith("cuda") else 1e-4
        torch.testing.assert_close(output_eager, output_compiled, rtol=1e-4, atol=atol)

    def test_compiled_model_multiple_calls(self, device: str):
        """Test that compiled model can be called multiple times."""
        require_cuda_autograd(device)
        model = SimpleNet().to(device)
        compiled_model = torch.compile(model, backend=mojo_backend, fullgraph=True)

        x1 = torch.randn(8, 1, 28, 28).to(device)
        x2 = torch.randn(4, 1, 28, 28).to(device)

        # First call
        output1 = compiled_model(x1)
        assert output1.shape == (8, 10)

        # Second call with different batch size
        output2 = compiled_model(x2)
        assert output2.shape == (4, 10)


class TestMNISTTraining:
    """Tests for MNIST model training functionality."""

    def test_optimizer_updates_weights(self, device: str):
        """Test that SGD optimizer updates model weights."""
        require_cuda_autograd(device)
        model = SimpleNet().to(device)
        x = torch.randn(8, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (8,)).to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Store initial weights
        initial_fc1_weight = model.fc1.weight.data.clone()
        initial_fc2_weight = model.fc2.weight.data.clone()
        initial_fc3_weight = model.fc3.weight.data.clone()

        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        # Verify weights changed
        assert not torch.equal(initial_fc1_weight, model.fc1.weight.data), (
            "FC1 weights were not updated"
        )
        assert not torch.equal(initial_fc2_weight, model.fc2.weight.data), (
            "FC2 weights were not updated"
        )
        assert not torch.equal(initial_fc3_weight, model.fc3.weight.data), (
            "FC3 weights were not updated"
        )

    def test_loss_decreases_with_training(self, device: str):
        """Test that loss decreases over multiple training steps."""
        require_cuda_autograd(device)
        model = SimpleNet().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.CrossEntropyLoss()

        # Fixed input/output for reproducibility
        torch.manual_seed(42)
        x = torch.randn(32, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (32,)).to(device)

        losses = []
        for _ in range(5):
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

    def test_model_overfits_single_batch(self, device: str):
        """Test that model can overfit a single batch (sanity check)."""
        require_cuda_autograd(device)
        model = SimpleNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Single batch
        torch.manual_seed(42)
        x = torch.randn(16, 1, 28, 28).to(device)
        y = torch.randint(0, 10, (16,)).to(device)

        # Train for many iterations
        for _ in range(100):
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

        # Check final accuracy is high
        model.eval()
        with torch.no_grad():
            output = model(x)
            predictions = output.argmax(dim=1)
            accuracy = (predictions == y).float().mean()

        # Should achieve high accuracy on single batch
        assert accuracy > 0.8, (
            f"Failed to overfit single batch: accuracy={accuracy:.2f}"
        )


class TestMNISTEvaluation:
    """Tests for MNIST model evaluation functionality."""

    def test_eval_mode_no_grad(self, device: str):
        """Test evaluation mode with torch.no_grad() context."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(16, 1, 28, 28).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (16, 10)
        assert output.requires_grad is False

    def test_eval_mode_consistent_predictions(self, device: str):
        """Test that predictions are consistent in eval mode."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(16, 1, 28, 28).to(device)

        with torch.no_grad():
            output1 = model(x)
            output2 = model(x)

        torch.testing.assert_close(output1, output2)

    def test_predictions_in_valid_range(self, device: str):
        """Test that model predictions are valid logits."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(16, 1, 28, 28).to(device)

        with torch.no_grad():
            output = model(x)

        # Check for NaN or Inf
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

        # Check that each sample has 10 class scores
        assert output.shape[1] == 10


class TestMNISTInputVariations:
    """Tests for MNIST model with various input variations."""

    def test_normalized_input(self, device: str):
        """Test with normalized MNIST input (mean=0.1307, std=0.3081)."""
        model = SimpleNet().to(device)

        # Simulate normalized MNIST data
        x = torch.randn(8, 1, 28, 28).to(device) * 0.3081 + 0.1307

        output = model(x)
        assert output.shape == (8, 10)

    def test_unnormalized_input(self, device: str):
        """Test with unnormalized input [0, 1] range."""
        model = SimpleNet().to(device)

        # Simulate unnormalized MNIST data [0, 1]
        x = torch.rand(8, 1, 28, 28).to(device)

        output = model(x)
        assert output.shape == (8, 10)

    def test_negative_input(self, device: str):
        """Test model handles negative input values."""
        model = SimpleNet().to(device)

        x = torch.randn(8, 1, 28, 28).to(device) - 2.0

        output = model(x)
        assert output.shape == (8, 10)
        assert not torch.isnan(output).any()

    def test_large_input_values(self, device: str):
        """Test model handles large input values."""
        model = SimpleNet().to(device)

        x = torch.randn(8, 1, 28, 28).to(device) * 10.0

        output = model(x)
        assert output.shape == (8, 10)
        assert not torch.isnan(output).any()


class TestMNISTEdgeCases:
    """Tests for MNIST model edge cases."""

    def test_single_sample_inference(self, device: str):
        """Test inference on a single sample."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(1, 1, 28, 28).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 10)

    def test_large_batch_inference(self, device: str):
        """Test inference on a large batch."""
        model = SimpleNet().to(device)
        model.eval()

        x = torch.randn(128, 1, 28, 28).to(device)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (128, 10)

    def test_model_state_dict_save_load(self, device: str):
        """Test saving and loading model state dict."""
        model1 = SimpleNet().to(device)
        model2 = SimpleNet().to(device)

        # Set random weights for model1
        torch.manual_seed(42)
        for param in model1.parameters():
            param.data.normal_()

        # Save and load state dict
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)

        # Verify weights are identical
        for (name1, param1), (name2, param2) in zip(
            model1.named_parameters(), model2.named_parameters()
        ):
            assert name1 == name2
            torch.testing.assert_close(param1, param2)

    def test_model_dtype_consistency(self, device: str):
        """Test that model maintains dtype consistency."""
        model = SimpleNet().to(device)

        # All parameters should be float32
        for param in model.parameters():
            assert param.dtype == torch.float32
