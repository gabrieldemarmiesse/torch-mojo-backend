"""Unit tests for basic native mojo device functionality.

The device is a real PrivateUse1 backend (see docs/native_backend.md):
tensors are plain, unwrapped `torch.Tensor`s over MAX device memory, with no
`TorchMojoTensor` wrapper, no `_holder`/`_ptr`/`_mojo_strides` payload
attributes, and no Python-level transfer bookkeeping (`_PENDING_H2D`,
dlpack pending exports, `cuda_peer`) to introspect -- all of that lived in
the old eager Python path and is gone. Tests that used to poke at those
internals are rewritten here at the public torch API level, or dropped with
a one-line reason (see the docstring of each removed test's replacement, and
the migration report).
"""

import io
import time

import pytest
import torch
from torch.optim.optimizer import _default_to_fused_or_foreach

from torch_mojo_backend import get_accelerators, mojo_backend, register_mojo_devices
from torch_mojo_backend.native import device_module

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture(autouse=True)
def setup_max_device():
    """Setup mojo_device for all tests"""
    register_mojo_devices()


def test_mojo_is_the_default_torch_accelerator():
    assert torch.accelerator.current_accelerator(check_available=True) == torch.device(
        "mojo"
    )


def test_torch_accelerator_synchronize_dispatches_and_validates_device():
    original_device = device_module.current_device()
    try:
        torch.accelerator.synchronize()
        torch.accelerator.synchronize("mojo")
        torch.accelerator.synchronize(0)
        if torch.accelerator.device_count() > 1:
            device_module.set_device(1)
            torch.accelerator.synchronize()
    finally:
        device_module.set_device(original_device)

    with pytest.raises(ValueError, match="doesn't match the current accelerator mojo"):
        torch.accelerator.synchronize("cpu")


def test_tensor_to_max_device(mojo_device):
    """Test converting regular tensor to mojo_device"""
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    mojo_tensor = cpu_tensor.to(mojo_device)
    assert mojo_tensor.device.type == "mojo"
    assert mojo_tensor.shape == (3,)
    assert mojo_tensor.dtype == torch.float32


def test_max_tensor_to_cpu(mojo_device):
    """Test converting MaxTensor back to CPU"""
    cpu_tensor = torch.tensor([1.0, 2.0, 3.0])
    mojo_tensor = cpu_tensor.to(mojo_device)
    result = mojo_tensor.to("cpu")
    assert isinstance(result, torch.Tensor)
    torch.testing.assert_close(result, cpu_tensor)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::arange.start_out")
def test_factory_arange(mojo_device):
    """Test torch.arange with mojo_device"""
    tensor = torch.arange(5, device=mojo_device)
    assert tensor.device.type == "mojo"
    assert tensor.shape == (5,)
    cpu_result = tensor.to("cpu")
    expected = torch.arange(5)
    torch.testing.assert_close(cpu_result, expected)


@pytest.mark.xfail(
    strict=False, reason="op not ported yet: aten::uniform_ (torch.rand's RNG kernel)"
)
def test_factory_rand(mojo_device):
    """Test torch.rand with mojo_device"""
    tensor = torch.rand(3, 4, device=mojo_device)
    assert tensor.device.type == "mojo"
    assert tensor.shape == (3, 4)
    cpu_result = tensor.to("cpu")
    assert torch.all(cpu_result >= 0)
    assert torch.all(cpu_result <= 1)


def test_factory_empty(mojo_device):
    """Test torch.empty with mojo_device"""
    tensor = torch.empty(2, 3, device=mojo_device)
    assert tensor.device.type == "mojo"
    assert tensor.shape == (2, 3)


def test_device_string_variations():
    """Test different mojo device string formats"""
    t1 = torch.tensor([1.0]).to("mojo")
    assert t1.device.type == "mojo"
    t2 = torch.tensor([1.0]).to("mojo:0")
    assert t2.device.type == "mojo"


def test_indexless_mojo_device_uses_and_restores_current_device():
    """An indexless mojo target follows the current device (see also the
    more focused version of this test in test_mojo_device_runtime.py)."""
    if device_module.device_count() < 2:
        pytest.skip("requires two Mojo devices, including the MAX CPU device")

    original_index = device_module.current_device()
    alternate_index = (original_index + 1) % device_module.device_count()
    try:
        device_module.set_device(alternate_index)
        empty_tensor = torch.empty(1, device="mojo")
        assert empty_tensor.device.type == "mojo"
        assert empty_tensor.device.index == alternate_index
    finally:
        device_module.set_device(original_index)
    assert device_module.current_device() == original_index


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::abs.out (torch's tensor-repr formatter "
    "needs it, along with isfinite/masked_select, to choose a print style)",
)
def test_tensor_properties(mojo_device):
    """A native mojo tensor is a plain torch.Tensor: shape/dtype/device and
    the standard torch repr all just work, with no custom wrapper needed."""
    original = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    mojo_tensor = original.to(mojo_device)

    assert mojo_tensor.shape == (2, 2)
    assert mojo_tensor.dtype == torch.float64
    assert mojo_tensor.device == torch.device(mojo_device)

    repr_str = repr(mojo_tensor)
    assert mojo_device in repr_str
    assert "float64" in repr_str


def test_round_trip_conversion(mojo_device):
    """Test CPU -> mojo_device -> CPU round trip"""
    original = torch.tensor([1.0, 2.0, 3.0, 4.0])
    mojo_tensor = original.to(mojo_device)
    result = mojo_tensor.to("cpu")
    torch.testing.assert_close(result, original)


def test_non_blocking_cpu_to_mojo_transfers(mojo_device):
    """Both to() and copy_() honor PyTorch's non_blocking transfer API."""
    source = torch.arange(4096, dtype=torch.float32)

    via_to = source.to(mojo_device, non_blocking=True)
    via_copy = torch.empty_like(via_to)
    via_copy.copy_(source, non_blocking=True)

    torch.accelerator.synchronize(mojo_device)
    torch.testing.assert_close(via_to.cpu(), source)
    torch.testing.assert_close(via_copy.cpu(), source)


def test_non_blocking_cpu_source_lifetime(mojo_device):
    """An async upload remains valid after its temporary CPU source dies.

    The old test also asserted `uploaded._device not in _PENDING_H2D`
    directly; that bookkeeping is now internal to the C++ shim
    (`native/csrc/shim_runtime.cpp`), so only the observable correctness
    contract -- the value survives the source's destruction -- is left to
    test here.
    """
    source = torch.arange(1 << 20, dtype=torch.int32)
    expected = source.clone()
    uploaded = source.to(mojo_device, non_blocking=True)
    assert uploaded.device.type == "mojo"

    del source
    # Encourage the CPU allocator to reuse the released storage while the H2D
    # operation may still be queued.
    for _ in range(8):
        torch.empty_like(expected).fill_(-1)

    torch.accelerator.synchronize(mojo_device)
    torch.testing.assert_close(uploaded.cpu(), expected)


def test_non_blocking_h2d_does_not_drain_prior_gpu_work(mojo_gpu: str):
    """An async upload returns without waiting for older default-stream work."""
    a = torch.full((4096, 4096), 1.0, device=mojo_gpu)
    b = torch.full((4096, 4096), 2.0, device=mojo_gpu)
    torch.accelerator.synchronize(mojo_gpu)

    # Establish a conservative duration for the work placed ahead of H2D:
    # enough launches that the device time dwarfs the enqueue cost.
    def burst():
        out = a * b
        for _ in range(31):
            out = out * b
        return out

    # The first staged upload of a process pays the pinned-buffer setup
    # (hundreds of ms); the steady state is what the check is about.
    _ = torch.arange(16).to(mojo_gpu, non_blocking=True)
    _ = burst()
    torch.accelerator.synchronize(mojo_gpu)
    started = time.perf_counter()
    _ = burst()
    torch.accelerator.synchronize(mojo_gpu)
    mul_seconds = time.perf_counter() - started

    delayed = burst()
    started = time.perf_counter()
    uploaded = torch.arange(4096).to(mojo_gpu, non_blocking=True)
    upload_return_seconds = time.perf_counter() - started

    assert upload_return_seconds < mul_seconds * 0.5
    torch.accelerator.synchronize(mojo_gpu)
    torch.testing.assert_close(uploaded.cpu(), torch.arange(4096))
    assert delayed.shape == (4096, 4096)


@pytest.mark.xfail(
    strict=False,
    reason="native D2H (`_copy_from` mojo->cpu) currently completes "
    "synchronously regardless of non_blocking=True: measured ~2.5-4.5ms "
    "return time for a 4MB download independent of how much prior GPU work "
    "was queued (no async/pinned-staging D2H path yet, unlike H2D)",
)
def test_non_blocking_mojo_to_cpu_does_not_drain_prior_gpu_work(mojo_gpu: str):
    """Async D2H returns without draining queued kernels.

    The "prior work" queue is many elementwise multiplies rather than a
    matmul (`aten::mm` isn't ported yet): a single large elementwise op is
    fast enough on modern hardware that its GPU time can be within noise of
    the async-download's own Python-side dispatch cost, so this queues many
    of them and uses a generous margin to stay robust on a busy shared GPU.
    """
    a = torch.full((4096, 4096), 1.0, device=mojo_gpu)
    b = torch.full((4096, 4096), 2.0, device=mojo_gpu)
    expected = torch.arange(1 << 20, dtype=torch.float32)
    source = expected.to(mojo_gpu)

    # Warm the async-download path before timing.
    _ = source.to("cpu", non_blocking=True)
    torch.accelerator.synchronize(mojo_gpu)

    queue_repeats = 200
    started = time.perf_counter()
    for _ in range(queue_repeats):
        _ = a * b
    torch.accelerator.synchronize(mojo_gpu)
    queued_seconds = time.perf_counter() - started

    delayed = [a * b for _ in range(queue_repeats)]
    started = time.perf_counter()
    downloaded = source.to("cpu", non_blocking=True)
    download_return_seconds = time.perf_counter() - started

    assert download_return_seconds < queued_seconds * 0.5
    torch.accelerator.synchronize(mojo_gpu)
    torch.testing.assert_close(downloaded, expected)
    assert all(result.shape == (4096, 4096) for result in delayed)


def test_non_blocking_strided_d2h_survives_source_destruction(mojo_gpu: str):
    """A strided D2H download outlives its source tensor and storage."""
    expected = torch.arange(1 << 20, dtype=torch.int32)
    storage = torch.stack((expected, -expected), dim=1).to(mojo_gpu)
    source = storage[:, 0]
    downloaded = source.to("cpu", non_blocking=True)

    del source, storage
    torch.accelerator.synchronize(mojo_gpu)
    torch.testing.assert_close(downloaded, expected)


def test_non_blocking_d2h_survives_destination_destruction(mojo_gpu: str):
    """Dropping the CPU alias early must not crash or corrupt later transfers
    (the in-flight download's pinned host buffer is kept alive internally
    until the copy actually completes)."""
    a = torch.full((4096, 4096), 1.0, device=mojo_gpu)
    b = torch.full((4096, 4096), 2.0, device=mojo_gpu)
    source = torch.arange(1 << 20, dtype=torch.float32).to(mojo_gpu)
    torch.accelerator.synchronize(mojo_gpu)

    delayed = [a * b for _ in range(8)]
    downloaded = source.to("cpu", non_blocking=True)
    del downloaded

    torch.accelerator.synchronize(mojo_gpu)
    assert all(result.shape == (4096, 4096) for result in delayed)
    # A later, ordinary transfer must still be correct (no corruption left
    # behind by the dropped in-flight one).
    again = source.to("cpu")
    torch.testing.assert_close(again, source.cpu())


def test_same_device_d2d_does_not_drain_prior_gpu_work(mojo_gpu: str):
    """Contiguous and strided D2D copies stay queued on the device stream."""
    a = torch.full((4096, 4096), 1.0, device=mojo_gpu)
    b = torch.full((4096, 4096), 2.0, device=mojo_gpu)
    elements = 1 << 20
    expected = torch.arange(elements, dtype=torch.float32)

    contiguous_source = expected.to(mojo_gpu)
    contiguous_destination = torch.empty_like(contiguous_source)

    strided_source_storage = torch.stack((expected, -expected), dim=1).to(mojo_gpu)
    strided_source = strided_source_storage[:, 0]
    strided_destination_storage = torch.empty_like(strided_source_storage)
    strided_destination = strided_destination_storage[:, 1]

    # Warm every copy path before measuring Python return latency.
    contiguous_destination.copy_(contiguous_source)
    strided_destination.copy_(strided_source)
    repeats = 200
    for _ in range(repeats):
        _ = a * b
    torch.accelerator.synchronize(mojo_gpu)

    started = time.perf_counter()
    for _ in range(repeats):
        _ = a * b
    torch.accelerator.synchronize(mojo_gpu)
    mul_seconds = time.perf_counter() - started

    for layout, destination, source in (
        ("contiguous", contiguous_destination, contiguous_source),
        ("strided", strided_destination, strided_source),
    ):
        delayed = [a * b for _ in range(repeats)]
        started = time.perf_counter()
        destination.copy_(source)
        copy_return_seconds = time.perf_counter() - started

        assert copy_return_seconds < mul_seconds * 0.5, layout
        torch.accelerator.synchronize(mojo_gpu)
        torch.testing.assert_close(destination.cpu(), expected)
        assert all(result.shape == (4096, 4096) for result in delayed)


def test_dtype_preservation(mojo_device):
    """Test that dtypes are preserved during conversion"""
    for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
        original = torch.tensor([1, 2, 3], dtype=dtype)
        mojo_tensor = original.to(mojo_device)
        result = mojo_tensor.to("cpu")
        assert result.dtype == dtype
        torch.testing.assert_close(result, original)


def test_multiple_conversions():
    """Test multiple to() calls don't cause issues"""
    tensor = torch.tensor([1.0, 2.0])
    max1 = tensor.to("mojo")
    max2 = max1.to("mojo")  # Should return same tensor
    cpu1 = max2.to("cpu")
    cpu2 = cpu1.to("cpu")  # Should work normally
    torch.testing.assert_close(cpu2, tensor)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::sub.Tensor")
def test_multiple_conversions_arithmetic():
    """Operate on the round-tripped tensors: a same-device sub then square."""
    tensor = torch.tensor([1.0, 2.0])
    max1 = tensor.to("mojo")
    max2 = max1.to("mojo")
    diff = max1 - max2
    squared = diff * diff
    summed = torch.sum(squared)
    assert summed.to("cpu").item() == 0


def test_module_to_mojo_preserves_tied_parameters(mojo_device):
    """See the fuller version of this test in test_model_state_conversion.py
    (module conversion owns the tied-weight contract there); kept here too
    since it also exercises plain `.to(device)` on an `nn.Module`."""

    class TiedWeights(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 8)
            self.projection = torch.nn.Linear(8, 16, bias=False)
            self.projection.weight = self.embedding.weight

    module = TiedWeights()
    module.to(mojo_device)

    assert module.embedding.weight is module.projection.weight
    assert module.embedding.weight.device.type == "mojo"
    assert len(list(module.parameters())) == 1


def test_mojo_parameters_enable_foreach_optimizer_selection(mojo_device):
    parameter = torch.nn.Parameter(torch.ones(8)).to(mojo_device)
    fused, foreach = _default_to_fused_or_foreach([parameter], differentiable=False)
    assert not fused
    assert foreach


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::mul.out (AdamW's foreach/single-tensor "
    "step needs the out= arithmetic ops)",
)
@pytest.mark.parametrize("foreach", [None, True, False])
def test_mojo_adamw_step_matches_cpu(mojo_gpu_available, foreach):
    """The optimizer path used by nanoGPT must update parameters and moments."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    mojo_gpu = "mojo:0"
    initial = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)
    cpu_parameter = torch.nn.Parameter(initial.clone())
    mojo_parameter = torch.nn.Parameter(initial.to(mojo_gpu))
    cpu_optimizer = torch.optim.AdamW(
        [cpu_parameter],
        lr=0.025,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=foreach,
    )
    mojo_optimizer = torch.optim.AdamW(
        [mojo_parameter],
        lr=0.025,
        betas=(0.8, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        foreach=foreach,
    )

    for grad in (
        torch.tensor([0.25, -0.5, 0.75, -1.0]),
        torch.tensor([-0.125, 0.25, -0.375, 0.5]),
    ):
        cpu_parameter.grad = grad.clone()
        mojo_parameter.grad = grad.to(mojo_gpu)
        cpu_optimizer.step()
        mojo_optimizer.step()

    torch.accelerator.synchronize(mojo_gpu)
    torch.testing.assert_close(mojo_parameter.cpu(), cpu_parameter)
    cpu_state = cpu_optimizer.state[cpu_parameter]
    mojo_state = mojo_optimizer.state[mojo_parameter]
    for name in ("exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(mojo_state[name].cpu(), cpu_state[name])
    assert mojo_state["step"].item() == cpu_state["step"].item() == 2


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::set_.source_Storage (torch.load's "
    "tensor rebuild path needs it before map_location moves the value to cpu)",
)
def test_mojo_checkpoint_resumes_through_portable_cpu_state(mojo_gpu_available):
    """The nanoGPT resume path loads CPU state, then moves it normally."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    mojo_gpu = "mojo:0"
    model = torch.nn.Linear(3, 2).to(mojo_gpu)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=None)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    checkpoint_bytes = io.BytesIO()
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
        checkpoint_bytes,
    )
    checkpoint_bytes.seek(0)
    checkpoint = torch.load(checkpoint_bytes, map_location="cpu")
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["model"].values())

    resumed_model = torch.nn.Linear(3, 2)
    resumed_model.load_state_dict(checkpoint["model"])
    resumed_model.to(mojo_gpu)
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(), lr=0.01, foreach=None
    )
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    for state in resumed_optimizer.state.values():
        assert state["step"].device.type == "cpu"
        assert state["exp_avg"].device == torch.device(mojo_gpu)
        assert state["exp_avg_sq"].device == torch.device(mojo_gpu)

    for parameter in resumed_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    resumed_optimizer.step()
    torch.accelerator.synchronize(mojo_gpu)
    assert all(
        torch.isfinite(parameter.cpu()).all()
        for parameter in resumed_model.parameters()
    )


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::linalg_vector_norm.out (foreach L2-norm "
    "clip needs the reduction)",
)
@pytest.mark.parametrize("foreach", [None, True, False])
def test_mojo_clip_grad_norm_matches_cpu(mojo_gpu_available, foreach):
    """nanoGPT's FP32 gradient clipping uses the foreach L2-norm path."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    mojo_gpu = "mojo:0"
    cpu_parameters = [
        torch.nn.Parameter(torch.zeros(3)),
        torch.nn.Parameter(torch.zeros(2, 2)),
    ]
    mojo_parameters = [
        torch.nn.Parameter(parameter.detach().to(mojo_gpu))
        for parameter in cpu_parameters
    ]
    gradients = (
        torch.tensor([3.0, 4.0, -2.0]),
        torch.tensor([[1.0, -2.0], [2.0, -1.0]]),
    )
    for cpu_parameter, mojo_parameter, gradient in zip(
        cpu_parameters, mojo_parameters, gradients, strict=True
    ):
        cpu_parameter.grad = gradient.clone()
        mojo_parameter.grad = gradient.to(mojo_gpu)

    expected_norm = torch.nn.utils.clip_grad_norm_(
        cpu_parameters, 1.25, foreach=foreach
    )
    actual_norm = torch.nn.utils.clip_grad_norm_(mojo_parameters, 1.25, foreach=foreach)
    torch.accelerator.synchronize(mojo_gpu)

    torch.testing.assert_close(actual_norm.cpu(), expected_norm)
    for actual, expected in zip(mojo_parameters, cpu_parameters, strict=True):
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad.cpu(), expected.grad)


def test_device_ordering_gpu_first_cpu_last():
    """The mojo device convention: index 0 is the first GPU (when present),
    the highest index is the MAX CPU device. `get_accelerators()` (MAX's own
    accelerator list, unrelated to the old TorchMojoTensor/torch_mojo_tensor
    machinery) carries the `.label` used to check this without any private
    import."""
    accelerators = list(get_accelerators())
    assert len(accelerators) > 0
    assert accelerators[-1].label == "cpu"
    assert device_module.cpu() == torch.device(f"mojo:{len(accelerators) - 1}")

    gpu_labels = [a.label for a in accelerators if a.label == "gpu"]
    if gpu_labels:
        assert accelerators[0].label == "gpu"
        t_gpu = torch.tensor([1.0]).to("mojo")
        assert t_gpu.device.type == "mojo" and t_gpu.device.index == 0
    cpu_index = len(accelerators) - 1
    t_cpu = torch.tensor([1.0]).to(f"mojo:{cpu_index}")
    assert t_cpu.device.type == "mojo"


# Original tests from the existing file
def function_equivalent_on_both_devices(
    func, device, *args, rtol=1e-4, atol=1e-4, **kwargs
):
    # This helper checks forward values only. Keeping the first forward's
    # autograd graph alive while the same closure-owned module moves back to
    # CPU adds a legitimate TensorImpl reference, which PyTorch's required
    # swap-on-conversion path rejects. Avoid manufacturing that unrelated
    # lifetime condition in forward-equivalence tests.
    with torch.no_grad():
        out1 = func(*args, device=device, **kwargs)
        out2 = func(*args, device="cpu", **kwargs)
    if isinstance(out1, list | tuple):
        assert type(out1) is type(out2)
    else:
        assert isinstance(out1, torch.Tensor)
        assert isinstance(out2, torch.Tensor)
        out1 = [out1]
        out2 = [out2]

    out1 = [o.to("cpu") for o in out1]

    for i, (o1, o2) in enumerate(zip(out1, out2)):
        assert o1.device == o2.device, f"Issue with output {i}"
        assert o1.shape == o2.shape, f"Issue with output {i}"
        assert o1.dtype == o2.dtype, f"Issue with output {i}"
        assert torch.allclose(o1, o2, rtol=rtol, atol=atol), f"Issue with output {i}"


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::sqrt.out")
def test_mojo_device_basic(mojo_device):
    def do_sqrt(device):
        a = torch.arange(4, device=device, dtype=torch.float32)
        return torch.sqrt(a)

    function_equivalent_on_both_devices(do_sqrt, mojo_device)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::sqrt.out")
def test_mojo_device_basic_arange_sqrt(mojo_device):
    a = torch.arange(4, device=mojo_device, dtype=torch.float32)
    sqrt_result = torch.sqrt(a)
    result_cpu = sqrt_result.to("cpu")
    assert torch.allclose(
        result_cpu, torch.tensor([0.0, 1.0, 1.4142, 1.7320]), atol=1e-4
    )
    b = torch.arange(4, device=mojo_device, dtype=torch.float32)
    chained = sqrt_result + b
    chained_cpu = chained.to("cpu")
    assert torch.allclose(
        chained_cpu, torch.tensor([0.0, 2.0, 3.4142, 4.7320]), atol=1e-4
    )


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::arange.start_out")
def test_device_creation(mojo_device):
    torch_device = torch.device(mojo_device)
    arr = torch.arange(4, device=torch_device, dtype=torch.float32)
    arr_cpu = arr.to("cpu")
    assert torch.allclose(arr_cpu, torch.tensor([0.0, 1.0, 2.0, 3.0]), atol=1e-4)


def test_device_basic_full(mojo_device):
    def do_full(device):
        a = torch.full((2, 3), 7.0, device=device, dtype=torch.float32)
        return a

    function_equivalent_on_both_devices(do_full, mojo_device)


@pytest.mark.xfail(
    strict=False,
    reason="op not ported yet: aten::normal_ (randn inputs) and conv2d itself",
)
def test_convolution_2d(mojo_device):
    input_tensor_cpu = torch.randn(1, 3, 32, 32, device="cpu")
    weight_cpu = torch.randn(6, 3, 5, 5, device="cpu")
    bias_cpu = torch.randn(6, device="cpu")

    def do_convolution(device):
        input_tensor = input_tensor_cpu.to(device)
        weight = weight_cpu.to(device)
        bias = bias_cpu.to(device)
        return torch.nn.functional.conv2d(
            input_tensor, weight, bias=bias, stride=1, padding=2
        )

    function_equivalent_on_both_devices(do_convolution, mojo_device)


def test_simple_module(mojo_device):
    """No forward pass here (see test_custom_module for that): just that
    `.to(device)` on a module carries its weight over correctly."""
    linear = torch.nn.Linear(4, 8)

    def run_module(device):
        my_linear = linear.to(device)
        return my_linear.weight

    function_equivalent_on_both_devices(run_module, mojo_device)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::addmm.out")
def test_custom_module(mojo_device):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 8)

        def forward(self, x):
            return self.linear(x)

    module = MyModule()
    input_tensor = torch.randn(2, 4)

    def run_module(device):
        in_device_module = module.to(device)
        in_device_input_tensor = input_tensor.to(device)
        return in_device_module(in_device_input_tensor)

    function_equivalent_on_both_devices(run_module, mojo_device, rtol=1e-3, atol=1e-3)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::addmm.out")
def test_custom_module_with_seqential(mojo_device):
    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 8)

        def forward(self, x):
            return self.linear(x)

    module = torch.nn.Sequential(MyModule())
    input_tensor = torch.randn(2, 4)

    def run_module(device):
        in_device_module = module.to(device)
        in_device_input_tensor = input_tensor.to(device)
        return in_device_module(in_device_input_tensor)

    function_equivalent_on_both_devices(run_module, mojo_device, rtol=1e-3, atol=1e-3)


def test_compile_with_max_device(mojo_device):
    """The torch.compile(backend=mojo_backend) graph path is independent of
    the native eager device: it lowers straight to a MAX graph rather than
    dispatching aten ops one at a time, so it is unaffected by which native
    ops have landed yet. Its outputs come back as mojo tensors through
    `mojo_device/dlpack.py`'s kDLExtDev capsule, not through any wrapper."""

    @torch.compile(backend=mojo_backend)
    def do_sqrt(device):
        a = torch.arange(4, device=device, dtype=torch.float32)
        return torch.sqrt(a)

    function_equivalent_on_both_devices(do_sqrt, mojo_device)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a second backend to copy from"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_to_mojo_from_another_backend(mojo_gpu, dtype):
    """A tensor on another accelerator must transfer, not fault.

    Regression test for a real segfault: `_to_copy`/`_copy_from` used to
    branch on "is it our own wrapper tensor", not "is it on the host", so
    with a GPU torch installed they also received a CUDA source and
    dereferenced its pointer as HOST memory. Found by the OpInfo `to`
    samples, which include a cross-device target tensor.
    """
    src = torch.randn(3, 5, device="cuda", dtype=dtype)
    moved = src.to(mojo_gpu)
    assert moved.device.type == "mojo"
    assert moved.dtype == dtype
    assert torch.equal(moved.cpu(), src.cpu())

    cast = src.to(mojo_gpu, dtype=torch.float32)
    assert cast.dtype == torch.float32
    assert torch.allclose(cast.cpu(), src.float().cpu())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a second backend to copy from"
)
def test_copy_into_mojo_from_another_backend(mojo_gpu):
    """The same trap on the `copy_` path, including the broadcasting arm."""
    dest = torch.zeros(3, 5, device=mojo_gpu)
    src = torch.randn(3, 5, device="cuda")
    dest.copy_(src)
    assert torch.equal(dest.cpu(), src.cpu())

    wide = torch.zeros(4, 3, device=mojo_gpu)
    row = torch.arange(3, device="cuda", dtype=torch.float32)
    wide.copy_(row)
    assert torch.equal(wide.cpu(), row.cpu().expand(4, 3).contiguous())


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a second backend to copy from"
)
def test_same_gpu_transfer_skips_the_host(mojo_gpu):
    """The transfer must not go through host memory when both live on one GPU.

    Asserted by timing rather than by values, because a host bounce is
    CORRECT -- just two PCIe crossings instead of one HBM copy.
    """
    big = torch.randn(1 << 26, device="cuda")  # 256 MB
    big.to(mojo_gpu)
    torch.accelerator.synchronize(mojo_gpu)

    start = time.perf_counter()
    moved = big.to(mojo_gpu)
    torch.accelerator.synchronize(mojo_gpu)
    on_device = time.perf_counter() - start

    start = time.perf_counter()
    bounced = big.cpu().to(mojo_gpu)
    torch.accelerator.synchronize(mojo_gpu)
    through_host = time.perf_counter() - start

    assert torch.equal(moved.cpu(), bounced.cpu())
    assert on_device * 10 < through_host, (
        f"expected the on-device route; {on_device * 1e3:.2f}ms on device vs "
        f"{through_host * 1e3:.2f}ms through the host"
    )


@pytest.mark.xfail(
    strict=False, reason="op not ported yet: aten::linalg_vector_norm.out"
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_vector_norm_with_an_accumulation_dtype(mojo_gpu_available, dtype):
    """FSDP1's clip_grad_norm_ asks for the norm in float32 explicitly."""
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    cpu = torch.randn(4096, dtype=dtype)
    expected = torch.linalg.vector_norm(cpu, 2.0, dtype=torch.float32)
    got = torch.linalg.vector_norm(cpu.to("mojo:0"), 2.0, dtype=torch.float32)
    assert got.dtype == torch.float32
    torch.testing.assert_close(got.cpu(), expected, rtol=1e-5, atol=1e-4)


def test_data_assignment_moves_the_payload(mojo_gpu_available):
    """``x.data = y`` has to move the allocation, not just TensorImpl metadata.

    FSDP1 swaps a flat parameter between its sharded and unsharded buffers
    with exactly this assignment.
    """
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    destination = torch.zeros(8, device="mojo:0")
    source = torch.full((8,), 100.0, device="mojo:0")
    destination.data = source
    assert destination.cpu().tolist() == source.cpu().tolist()

    # A strided view at a non-zero offset must survive intact: FSDP hands
    # every original parameter a slice of the flat parameter this way.
    base = torch.arange(16, dtype=torch.float32).to("mojo:0")
    view = base[4:12].reshape(2, 4)
    holder = torch.zeros(1, device="mojo:0")
    holder.data = view
    assert tuple(holder.shape) == (2, 4)
    assert holder.stride() == view.stride()
    assert holder.cpu().tolist() == [[4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]


def test_as_strided_zero_copy_view(mojo_gpu_available):
    """A strided view at an offset, and the allocation-bounds refusal.

    as_strided is the one view op whose arguments are unconstrained by the
    input's own shape, so the backend must bound-check against the real
    allocation rather than trust the caller (DDP's Reducer builds gradient
    bucket views with it).
    """
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    cpu = torch.arange(24, dtype=torch.float32)
    dev = cpu.to("mojo:0")
    view = torch.as_strided(dev, (3, 4), (8, 2), 1)
    assert torch.equal(view.cpu(), torch.as_strided(cpu, (3, 4), (8, 2), 1))
    # Zero-copy: writes through the base are visible in the view.
    dev.fill_(1.0)
    torch.testing.assert_close(
        view.cpu(), torch.as_strided(torch.ones(24), (3, 4), (8, 2), 1)
    )
    # A layout that would reach past the allocation must be refused: a real
    # error (TORCH_CHECK in the C++ shim), not silently accepted.
    with pytest.raises(RuntimeError, match="sizes/strides reach"):
        torch.as_strided(dev, (5, 4), (8, 2), 1)


@pytest.mark.xfail(strict=False, reason="op not ported yet: aten::set_.source_Tensor")
def test_set_source_tensor_adopts_the_allocation(mojo_gpu_available):
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    destination = torch.zeros(8, device="mojo:0")
    source = torch.full((4,), 50.0, device="mojo:0")
    returned = destination.set_(source)  # ty: ignore[invalid-argument-type] -- torch's stub lacks the Tensor overload
    assert returned is destination
    assert tuple(destination.shape) == (4,)
    assert destination.cpu().tolist() == [50.0, 50.0, 50.0, 50.0]
    # Sharing the allocation, not a copy of it.
    source.fill_(1.0)
    assert destination.cpu().tolist() == [1.0, 1.0, 1.0, 1.0]
