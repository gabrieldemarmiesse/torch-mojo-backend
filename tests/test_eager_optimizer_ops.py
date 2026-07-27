"""Direct eager ATen contracts used by AdamW and gradient clipping."""

import pytest
import torch

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.testing import CallChecker

pytestmark = pytest.mark.xdist_group(name="group1")


@pytest.fixture
def mojo_gpu(mojo_gpu_available: bool) -> str:
    if not mojo_gpu_available:
        pytest.skip("requires a MAX GPU")
    register_mojo_devices()
    return "mojo:0"


def _watch_eager_op(call_checker: CallChecker, op_name: str) -> None:
    """Require the exact PrivateUse1 registration, not a decomposition twin."""
    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    call_checker.register(EAGER_CALL_COUNTERS[op_name])


def _fused_adamw_case(device: str, *, amsgrad: bool, maximize: bool):
    """Create nonuniform, nonzero AdamW state without sharing storage."""
    shapes = ((), (0,), (7,), (17, 65))

    def values(shape, *, scale, offset):
        numel = torch.empty(shape).numel()
        return (
            torch.arange(numel, dtype=torch.float32)
            .mul(scale)
            .add(offset)
            .reshape(shape)
        )

    parameters = [
        values(shape, scale=0.003, offset=-0.75 + index * 0.1)
        for index, shape in enumerate(shapes)
    ]
    gradients = [
        values(shape, scale=-0.0007, offset=0.3 - index * 0.02)
        for index, shape in enumerate(shapes)
    ]
    exp_avgs = [
        values(shape, scale=0.0002, offset=-0.08 + index * 0.01)
        for index, shape in enumerate(shapes)
    ]
    exp_avg_sqs = [
        values(shape, scale=0.00001, offset=0.01 + index * 0.001)
        for index, shape in enumerate(shapes)
    ]
    max_exp_avg_sqs = [value.mul(1.25) for value in exp_avg_sqs] if amsgrad else []
    state_steps = [
        torch.tensor(float(step), dtype=torch.float32) for step in (3, 5, 11, 17)
    ]
    groups = (
        parameters,
        gradients,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    )
    return tuple([[value.to(device) for value in group] for group in groups])


@pytest.mark.parametrize(
    ("amsgrad", "maximize"), [(False, False), (False, True), (True, False)]
)
def test_fused_adamw_direct_matches_cpu_for_runtime_shapes(
    mojo_gpu: str, call_checker: CallChecker, amsgrad: bool, maximize: bool
):
    """Exercise the exact mutable TensorList op used by fused AdamW."""
    _watch_eager_op(call_checker, "aten::_fused_adamw_")
    cpu_groups = _fused_adamw_case("cpu", amsgrad=amsgrad, maximize=maximize)
    mojo_groups = _fused_adamw_case(mojo_gpu, amsgrad=amsgrad, maximize=maximize)
    kwargs = {
        "lr": 0.025,
        "beta1": 0.8,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "eps": 1e-8,
        "amsgrad": amsgrad,
        "maximize": maximize,
    }
    aliases = [
        [(tensor._holder, tensor._ptr) for tensor in group]
        for group in mojo_groups[: 5 if amsgrad else 4]
    ]
    original_grads = [tensor.cpu().clone() for tensor in mojo_groups[1]]
    original_steps = [tensor.cpu().clone() for tensor in mojo_groups[5]]

    assert torch.ops.aten._fused_adamw_.default(*cpu_groups, **kwargs) is None
    assert torch.ops.aten._fused_adamw_.default(*mojo_groups, **kwargs) is None

    for cpu_group, mojo_group in zip(cpu_groups, mojo_groups, strict=True):
        for expected, actual in zip(cpu_group, mojo_group, strict=True):
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    for group, group_aliases in zip(
        mojo_groups[: 5 if amsgrad else 4], aliases, strict=True
    ):
        for tensor, (holder, ptr) in zip(group, group_aliases, strict=True):
            assert tensor._holder is holder
            assert tensor._ptr == ptr
    for actual, expected in zip(mojo_groups[1], original_grads, strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    for actual, expected in zip(mojo_groups[5], original_steps, strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_fused_adamw_tensor_lr_overload_matches_cpu(
    mojo_gpu: str, call_checker: CallChecker
):
    _watch_eager_op(call_checker, "aten::_fused_adamw_.tensor_lr")
    cpu_groups = _fused_adamw_case("cpu", amsgrad=False, maximize=False)
    mojo_groups = _fused_adamw_case(mojo_gpu, amsgrad=False, maximize=False)
    cpu_lr = torch.tensor(0.0125, dtype=torch.float32)
    mojo_lr = cpu_lr.to(mojo_gpu)
    kwargs = {
        "beta1": 0.8,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
    }

    assert (
        torch.ops.aten._fused_adamw_.tensor_lr(*cpu_groups, lr=cpu_lr, **kwargs) is None
    )
    assert (
        torch.ops.aten._fused_adamw_.tensor_lr(*mojo_groups, lr=mojo_lr, **kwargs)
        is None
    )
    for cpu_group, mojo_group in zip(cpu_groups, mojo_groups, strict=True):
        for expected, actual in zip(cpu_group, mojo_group, strict=True):
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(mojo_lr.cpu(), cpu_lr, rtol=0, atol=0)


def test_fused_adamw_zero_decay_preserves_nonfinite_parameter(
    mojo_gpu: str, call_checker: CallChecker
):
    """Zero decay must not evaluate ``0 * inf`` and turn infinity into NaN."""
    _watch_eager_op(call_checker, "aten::_fused_adamw_")

    def groups(device):
        return (
            [torch.tensor([float("inf")], dtype=torch.float32, device=device)],
            [torch.tensor([0.25], dtype=torch.float32, device=device)],
            [torch.tensor([0.0], dtype=torch.float32, device=device)],
            [torch.tensor([1.0], dtype=torch.float32, device=device)],
            [],
            [torch.tensor(1.0, dtype=torch.float32, device=device)],
        )

    cpu_groups = groups("cpu")
    mojo_groups = groups(mojo_gpu)
    kwargs = {
        "lr": 0.01,
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 0.0,
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
    }

    torch.ops.aten._fused_adamw_.default(*cpu_groups, **kwargs)
    torch.ops.aten._fused_adamw_.default(*mojo_groups, **kwargs)

    assert torch.isinf(cpu_groups[0][0]).all()
    assert torch.isinf(mojo_groups[0][0].cpu()).all()
    for cpu_group, mojo_group in zip(cpu_groups[1:], mojo_groups[1:], strict=True):
        for expected, actual in zip(cpu_group, mojo_group, strict=True):
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)


def test_fused_adamw_accepts_contiguous_singleton_stride_variants(
    mojo_gpu: str, call_checker: CallChecker
):
    """Size-one dimensions can have arbitrary strides and remain contiguous."""
    _watch_eager_op(call_checker, "aten::_fused_adamw_")

    def groups(device):
        def singleton_view(values):
            return torch.tensor([values], dtype=torch.float32, device=device).transpose(
                0, 1
            )

        return (
            [torch.tensor([[1.0], [-2.0]], dtype=torch.float32, device=device)],
            [singleton_view([0.25, -0.5])],
            [singleton_view([0.0, 0.1])],
            [singleton_view([1.0, 1.5])],
            [],
            [torch.tensor(3.0, dtype=torch.float32, device=device)],
        )

    cpu_groups = groups("cpu")
    mojo_groups = groups(mojo_gpu)
    parameter, gradient = mojo_groups[0][0], mojo_groups[1][0]
    assert parameter._strides == (1, 1)
    assert gradient._strides == (1, 2)
    assert parameter._is_contiguous and gradient._is_contiguous
    kwargs = {
        "lr": 0.01,
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
    }

    torch.ops.aten._fused_adamw_.default(*cpu_groups, **kwargs)
    torch.ops.aten._fused_adamw_.default(*mojo_groups, **kwargs)

    for cpu_group, mojo_group in zip(cpu_groups, mojo_groups, strict=True):
        for expected, actual in zip(cpu_group, mojo_group, strict=True):
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)


def test_fused_adamw_batches_more_than_descriptor_capacity(
    mojo_gpu: str, call_checker: CallChecker
):
    """Cross the 32-descriptor launch boundary with interspersed empty tensors."""
    _watch_eager_op(call_checker, "aten::_fused_adamw_")
    empty_indices = {0, 31, 32, 36}
    sizes = [0 if index in empty_indices else index % 9 + 1 for index in range(37)]

    def values(size, *, scale, offset):
        return torch.arange(size, dtype=torch.float32).mul(scale).add(offset)

    cpu_groups = (
        [
            values(size, scale=0.01, offset=index * 0.1)
            for index, size in enumerate(sizes)
        ],
        [values(size, scale=-0.02, offset=0.25) for size in sizes],
        [values(size, scale=0.001, offset=-0.05) for size in sizes],
        [values(size, scale=0.0001, offset=0.01) for size in sizes],
        [],
        [torch.tensor(float(index % 7 + 1)) for index in range(len(sizes))],
    )
    mojo_groups = tuple(
        [[tensor.to(mojo_gpu) for tensor in group] for group in cpu_groups]
    )
    kwargs = {
        "lr": 0.01,
        "beta1": 0.9,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "eps": 1e-8,
        "amsgrad": False,
        "maximize": False,
    }

    torch.ops.aten._fused_adamw_.default(*cpu_groups, **kwargs)
    torch.ops.aten._fused_adamw_.default(*mojo_groups, **kwargs)

    for cpu_group, mojo_group in zip(cpu_groups, mojo_groups, strict=True):
        for expected, actual in zip(cpu_group, mojo_group, strict=True):
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize("found_inf_value", [0.0, 1.0])
def test_fused_adamw_grad_scale_and_found_inf_match_cpu(
    mojo_gpu: str, call_checker: CallChecker, found_inf_value: float
):
    _watch_eager_op(call_checker, "aten::_fused_adamw_")
    cpu_groups = _fused_adamw_case("cpu", amsgrad=True, maximize=False)
    mojo_groups = _fused_adamw_case(mojo_gpu, amsgrad=True, maximize=False)
    original_mojo = [
        [tensor.cpu().clone() for tensor in group] for group in mojo_groups
    ]
    cpu_grad_scale = torch.tensor(2.0, dtype=torch.float32)
    mojo_grad_scale = cpu_grad_scale.to(mojo_gpu)
    cpu_found_inf = torch.tensor(found_inf_value, dtype=torch.float32)
    mojo_found_inf = cpu_found_inf.to(mojo_gpu)
    kwargs = {
        "lr": 0.025,
        "beta1": 0.8,
        "beta2": 0.95,
        "weight_decay": 0.1,
        "eps": 1e-8,
        "amsgrad": True,
        "maximize": False,
    }

    torch.ops.aten._fused_adamw_.default(
        *cpu_groups, grad_scale=cpu_grad_scale, found_inf=cpu_found_inf, **kwargs
    )
    torch.ops.aten._fused_adamw_.default(
        *mojo_groups, grad_scale=mojo_grad_scale, found_inf=mojo_found_inf, **kwargs
    )

    for cpu_group, mojo_group in zip(cpu_groups, mojo_groups, strict=True):
        for expected, actual in zip(cpu_group, mojo_group, strict=True):
            torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    if found_inf_value == 0.0:
        for original, actual in zip(original_mojo[1], mojo_groups[1], strict=True):
            torch.testing.assert_close(actual.cpu(), original / 2.0, rtol=0, atol=0)
    else:
        for original_group, actual_group in zip(
            original_mojo, mojo_groups, strict=True
        ):
            for original, actual in zip(original_group, actual_group, strict=True):
                torch.testing.assert_close(actual.cpu(), original, rtol=0, atol=0)
    torch.testing.assert_close(mojo_grad_scale.cpu(), cpu_grad_scale, rtol=0, atol=0)
    torch.testing.assert_close(mojo_found_inf.cpu(), cpu_found_inf, rtol=0, atol=0)


def test_fused_adamw_validates_every_tensor_before_write(
    mojo_gpu: str, call_checker: CallChecker
):
    _watch_eager_op(call_checker, "aten::_fused_adamw_")
    parameters = [
        torch.tensor([1.0, -2.0], device=mojo_gpu),
        torch.tensor([3.0, -4.0, 5.0], device=mojo_gpu),
    ]
    grads = [torch.ones_like(parameter) for parameter in parameters]
    exp_avgs = [torch.zeros_like(parameter) for parameter in parameters]
    exp_avg_sqs = [torch.ones_like(parameter) for parameter in parameters]
    # Only the later entry is malformed; a partial-launch implementation would
    # have already updated the valid first entry before discovering this.
    exp_avg_sqs[1] = torch.ones(4, device=mojo_gpu)
    steps = [torch.tensor(1.0, device=mojo_gpu) for _ in parameters]
    mutable_groups = (parameters, grads, exp_avgs, exp_avg_sqs)
    snapshots = [[tensor.cpu().clone() for tensor in group] for group in mutable_groups]

    with pytest.raises(RuntimeError, match="same dtype, device, shape"):
        torch.ops.aten._fused_adamw_.default(
            parameters,
            grads,
            exp_avgs,
            exp_avg_sqs,
            [],
            steps,
            lr=0.01,
            beta1=0.9,
            beta2=0.95,
            weight_decay=0.1,
            eps=1e-8,
            amsgrad=False,
            maximize=False,
        )
    for snapshot_group, actual_group in zip(snapshots, mutable_groups, strict=True):
        for snapshot, actual in zip(snapshot_group, actual_group, strict=True):
            torch.testing.assert_close(actual.cpu(), snapshot, rtol=0, atol=0)


def test_sub_out_supports_optimizer_step_rollback(
    mojo_gpu: str, call_checker: CallChecker
):
    _watch_eager_op(call_checker, "aten::sub.out")
    step = torch.tensor(4.0, device=mojo_gpu)
    found_inf = torch.tensor(1.0, device=mojo_gpu)
    holder, ptr = step._holder, step._ptr

    returned = torch.ops.aten.sub.out(step, found_inf, out=step)

    assert returned is step
    assert step._holder is holder
    assert step._ptr == ptr
    torch.testing.assert_close(step.cpu(), torch.tensor(3.0), rtol=0, atol=0)


def test_fused_adamw_optimizer_two_groups_matches_cpu(
    mojo_gpu: str, call_checker: CallChecker
):
    """Cover PyTorch's state setup, foreach step increment, and fused route."""
    _watch_eager_op(call_checker, "aten::_fused_adamw_")
    initial = [
        torch.linspace(-1.0, 1.0, 31, dtype=torch.float32),
        torch.linspace(0.5, -0.75, 35, dtype=torch.float32).reshape(5, 7),
    ]
    cpu_parameters = [torch.nn.Parameter(value.clone()) for value in initial]
    mojo_parameters = [torch.nn.Parameter(value.to(mojo_gpu)) for value in initial]
    cpu_optimizer = torch.optim.AdamW(
        [
            {"params": [cpu_parameters[0]], "weight_decay": 0.1},
            {"params": [cpu_parameters[1]], "weight_decay": 0.0},
        ],
        lr=0.0125,
        betas=(0.8, 0.95),
        eps=1e-8,
        fused=True,
    )
    mojo_optimizer = torch.optim.AdamW(
        [
            {"params": [mojo_parameters[0]], "weight_decay": 0.1},
            {"params": [mojo_parameters[1]], "weight_decay": 0.0},
        ],
        lr=0.0125,
        betas=(0.8, 0.95),
        eps=1e-8,
        fused=True,
    )

    from torch_mojo_backend.mojo_device.mojo_device_aten_ops import EAGER_CALL_COUNTERS

    fused_counter = EAGER_CALL_COUNTERS["aten::_fused_adamw_"]
    calls_before = fused_counter.call_count
    for step in range(2):
        for index, (cpu_parameter, mojo_parameter) in enumerate(
            zip(cpu_parameters, mojo_parameters, strict=True)
        ):
            gradient = torch.linspace(
                -0.2 + step * 0.03,
                0.35 - index * 0.02,
                cpu_parameter.numel(),
                dtype=torch.float32,
            ).reshape(cpu_parameter.shape)
            cpu_parameter.grad = gradient.clone()
            mojo_parameter.grad = gradient.to(mojo_gpu)
        cpu_optimizer.step()
        mojo_optimizer.step()

    assert fused_counter.call_count - calls_before == 4
    assert all(group["fused"] is True for group in mojo_optimizer.param_groups)

    for cpu_parameter, mojo_parameter in zip(
        cpu_parameters, mojo_parameters, strict=True
    ):
        torch.testing.assert_close(
            mojo_parameter.cpu(), cpu_parameter, rtol=2e-6, atol=2e-7
        )
        cpu_state = cpu_optimizer.state[cpu_parameter]
        mojo_state = mojo_optimizer.state[mojo_parameter]
        for name in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(
                mojo_state[name].cpu(), cpu_state[name], rtol=2e-6, atol=2e-7
            )
        assert mojo_state["step"].device == torch.device(mojo_gpu)
        assert mojo_state["step"].dtype == torch.float32
        assert mojo_state["step"].item() == 2


def test_fused_adamw_optimizer_found_inf_rolls_back_step(
    mojo_gpu: str, call_checker: CallChecker
):
    """PyTorch, not the fused ATen op, owns preincrement and rollback."""
    _watch_eager_op(call_checker, "aten::_fused_adamw_")
    initial = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    gradient = torch.tensor([0.2, -0.4, 0.6], dtype=torch.float32)
    cpu_parameter = torch.nn.Parameter(initial.clone())
    mojo_parameter = torch.nn.Parameter(initial.to(mojo_gpu))
    cpu_optimizer = torch.optim.AdamW([cpu_parameter], lr=0.01, fused=True)
    mojo_optimizer = torch.optim.AdamW([mojo_parameter], lr=0.01, fused=True)
    cpu_parameter.grad = gradient.clone()
    mojo_parameter.grad = gradient.to(mojo_gpu)
    cpu_optimizer.grad_scale = torch.tensor(2.0)
    cpu_optimizer.found_inf = torch.tensor(1.0)
    mojo_optimizer.grad_scale = torch.tensor(2.0, device=mojo_gpu)
    mojo_optimizer.found_inf = torch.tensor(1.0, device=mojo_gpu)

    cpu_optimizer.step()
    mojo_optimizer.step()
    torch.testing.assert_close(mojo_parameter.cpu(), initial, rtol=0, atol=0)
    mojo_state = mojo_optimizer.state[mojo_parameter]
    assert mojo_state["step"].device == torch.device(mojo_gpu)
    assert mojo_state["step"].dtype == torch.float32
    assert mojo_state["step"].item() == 0

    cpu_optimizer.found_inf.fill_(0.0)
    mojo_optimizer.found_inf.fill_(0.0)
    cpu_optimizer.step()
    mojo_optimizer.step()
    torch.testing.assert_close(
        mojo_parameter.cpu(), cpu_parameter, rtol=2e-6, atol=2e-7
    )
    for name in ("exp_avg", "exp_avg_sq", "step"):
        torch.testing.assert_close(
            mojo_state[name].cpu(),
            cpu_optimizer.state[cpu_parameter][name],
            rtol=2e-6,
            atol=2e-7,
        )
    assert mojo_state["step"].item() == 1


def _foreach_lists(device: str) -> list[list[torch.Tensor]]:
    """Deterministic nonuniform FP32 lists for the batched foreach ops.

    Shapes cover empties, multi-dim tensors, and one tensor crossing the
    65_536-element chunk boundary. The third list (divisors for addcdiv)
    stays bounded away from zero.
    """
    shapes = ((7,), (17, 65), (0,), (5, 3, 2), (65_539,))

    def values(shape, *, scale: float, offset: float) -> torch.Tensor:
        numel = torch.empty(shape).numel()
        return (
            torch.arange(numel, dtype=torch.float32)
            .mul(scale)
            .add(offset)
            .reshape(shape)
        )

    mutated = [
        values(shape, scale=0.003, offset=-0.4 + index * 0.1)
        for index, shape in enumerate(shapes)
    ]
    operands = [
        values(shape, scale=-0.0007, offset=0.3 - index * 0.02)
        for index, shape in enumerate(shapes)
    ]
    divisors = [
        values(shape, scale=0.0002, offset=-0.08 + index * 0.01).abs().add(0.5)
        for index, shape in enumerate(shapes)
    ]
    return [
        [tensor.to(device) for tensor in group]
        for group in (mutated, operands, divisors)
    ]


_FOREACH_BATCHED_SCALARS = [0.5, -1.5, 2.0, 0.25, -0.125]
_FOREACH_BATCHED_OPS = [
    ("aten::_foreach_add_.Scalar", lambda a, b, c: torch._foreach_add_(a, 1e-3)),
    (
        "aten::_foreach_addcdiv_.ScalarList",
        lambda a, b, c: torch._foreach_addcdiv_(a, b, c, _FOREACH_BATCHED_SCALARS),
    ),
    (
        "aten::_foreach_addcmul_.Scalar",
        lambda a, b, c: torch._foreach_addcmul_(a, b, c, value=0.01),
    ),
    (
        "aten::_foreach_div_.ScalarList",
        lambda a, b, c: torch._foreach_div_(a, _FOREACH_BATCHED_SCALARS),
    ),
    ("aten::_foreach_lerp_.Scalar", lambda a, b, c: torch._foreach_lerp_(a, b, 0.1)),
    ("aten::_foreach_mul_.Scalar", lambda a, b, c: torch._foreach_mul_(a, 0.998)),
]


@pytest.mark.parametrize(("op_name", "apply"), _FOREACH_BATCHED_OPS)
def test_batched_foreach_elementwise_matches_cpu(
    mojo_gpu: str, call_checker: CallChecker, op_name: str, apply
):
    """The exact batched in-place registration runs and matches CPU."""
    _watch_eager_op(call_checker, op_name)
    cpu_lists = _foreach_lists("cpu")
    mojo_lists = _foreach_lists(mojo_gpu)
    versions_before = [tensor._version for tensor in mojo_lists[0]]
    apply(*cpu_lists)
    apply(*mojo_lists)
    for expected, actual in zip(cpu_lists[0], mojo_lists[0], strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-6, atol=2e-7)
    for tensor, version in zip(mojo_lists[0], versions_before, strict=True):
        assert tensor._version > version


def test_batched_foreach_sqrt_matches_cpu(mojo_gpu: str, call_checker: CallChecker):
    _watch_eager_op(call_checker, "aten::_foreach_sqrt")
    cpu_inputs = [tensor.abs() for tensor in _foreach_lists("cpu")[0]]
    mojo_inputs = [tensor.abs() for tensor in _foreach_lists(mojo_gpu)[0]]
    expected = torch._foreach_sqrt(cpu_inputs)
    actual = torch._foreach_sqrt(mojo_inputs)
    for expected_out, actual_out in zip(expected, actual, strict=True):
        torch.testing.assert_close(actual_out.cpu(), expected_out, rtol=0, atol=0)
    for original, actual_out in zip(mojo_inputs, actual, strict=True):
        if original.numel() > 0:
            assert actual_out._ptr != original._ptr


def test_batched_foreach_falls_back_for_non_f32(
    mojo_gpu: str, call_checker: CallChecker
):
    """Unsupported regimes reach ATen's sequential semantics unchanged."""
    _watch_eager_op(call_checker, "aten::_foreach_mul_.Scalar")
    cpu_tensors = [torch.arange(5), torch.arange(3)]
    mojo_tensors = [tensor.to(mojo_gpu) for tensor in cpu_tensors]
    torch._foreach_mul_(cpu_tensors, 3)
    torch._foreach_mul_(mojo_tensors, 3)
    for expected, actual in zip(cpu_tensors, mojo_tensors, strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_lerp_scalar_broadcast_uses_narrowed_fp32_branch(
    mojo_gpu: str, call_checker: CallChecker
):
    _watch_eager_op(call_checker, "aten::lerp.Scalar")

    # This Python double rounds to exactly 0.5f. ATen narrows the scalar before
    # selecting its stable formula, so this also exercises the >= 0.5 branch.
    weight = 0.5 - 2**-30
    start = torch.tensor(
        [[[-1.0687099695205688, -2.0, 3.0]], [[4.0, -5.0, 6.0]]], dtype=torch.float32
    )
    end = torch.tensor(
        [[[2.028475284576416, 8.0, -3.0]] for _ in range(4)], dtype=torch.float32
    ).transpose(0, 1)
    assert start.shape == (2, 1, 3)
    assert end.shape == (1, 4, 3)

    expected = torch.ops.aten.lerp.Scalar(start, end, weight)
    actual = torch.ops.aten.lerp.Scalar(start.to(mojo_gpu), end.to(mojo_gpu), weight)

    assert actual.shape == expected.shape == (2, 4, 3)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def test_lerp_scalar_out_supports_strided_self_alias(
    mojo_gpu: str, call_checker: CallChecker
):
    _watch_eager_op(call_checker, "aten::lerp.Scalar_out")

    weight = 0.5 - 2**-30
    base = torch.arange(24, dtype=torch.float32).reshape(3, 8) - 7.0
    end = torch.tensor([[2.0, -3.0, 5.0, -7.0]], dtype=torch.float32)
    expected_base = base.clone()
    expected_base[:, 1::2] = torch.lerp(base[:, 1::2], end, weight)

    device_base = base.to(mojo_gpu)
    device_self = device_base[:, 1::2]
    device_end = end.to(mojo_gpu)
    assert not device_self._is_contiguous
    assert not device_self.is_contiguous()
    holder, ptr = device_self._holder, device_self._ptr

    returned = torch.ops.aten.lerp.Scalar_out(
        device_self, device_end, weight, out=device_self
    )

    assert returned is device_self
    assert device_self._holder is holder
    assert device_self._ptr == ptr
    torch.testing.assert_close(device_base.cpu(), expected_base, rtol=0, atol=0)
    torch.testing.assert_close(device_end.cpu(), end, rtol=0, atol=0)


def test_lerp_scalar_rejects_fp32_weight_overflow(
    mojo_gpu: str, call_checker: CallChecker
):
    _watch_eager_op(call_checker, "aten::lerp.Scalar")

    start = torch.tensor([1.0, -2.0], dtype=torch.float32)
    end = torch.tensor([3.0, 4.0], dtype=torch.float32)
    with pytest.raises(RuntimeError, match="cannot be converted.*float.*overflow"):
        torch.ops.aten.lerp.Scalar(start, end, 1e40)
    with pytest.raises(RuntimeError, match="cannot be converted.*float.*overflow"):
        torch.ops.aten.lerp.Scalar(start.to(mojo_gpu), end.to(mojo_gpu), 1e40)


@pytest.mark.parametrize("operation", ["lerp", "vector_norm"])
def test_optimizer_out_ops_reject_invalid_integer_output(
    mojo_gpu: str, call_checker: CallChecker, operation: str
):
    if operation == "lerp":
        op_name = "aten::lerp.Scalar_out"
        lhs = torch.tensor([1.0, -2.0], device=mojo_gpu)
        rhs = torch.tensor([3.0, 4.0], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.lerp.Scalar_out(lhs, rhs, 0.25, out=out)

    else:
        op_name = "aten::linalg_vector_norm.out"
        input = torch.tensor([3.0, 4.0], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.linalg_vector_norm.out(
                input, 2, None, False, dtype=None, out=out
            )

    _watch_eager_op(call_checker, op_name)
    out = torch.empty((), dtype=torch.int64, device=mojo_gpu)
    if operation == "lerp":
        out = torch.empty(2, dtype=torch.int64, device=mojo_gpu)

    with pytest.raises(RuntimeError):
        invoke(out)


@pytest.mark.parametrize("operation", ["lerp", "vector_norm"])
def test_optimizer_out_ops_resize_public_shape_metadata(
    mojo_gpu: str, call_checker: CallChecker, operation: str
):
    if operation == "lerp":
        op_name = "aten::lerp.Scalar_out"
        lhs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        rhs = lhs + 6.0
        expected = torch.lerp(lhs, rhs, 0.25)
        device_lhs = lhs.to(mojo_gpu)
        device_rhs = rhs.to(mojo_gpu)

        def invoke(out):
            return torch.ops.aten.lerp.Scalar_out(device_lhs, device_rhs, 0.25, out=out)

    else:
        op_name = "aten::linalg_vector_norm.out"
        input = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        expected = torch.linalg.vector_norm(input, dim=1)
        device_input = input.to(mojo_gpu)

        def invoke(out):
            return torch.ops.aten.linalg_vector_norm.out(
                device_input, 2, [1], False, dtype=None, out=out
            )

    _watch_eager_op(call_checker, op_name)
    out = torch.empty(0, dtype=torch.float32, device=mojo_gpu)
    marker = object()
    out.user_marker = marker
    version = out._version
    returned = invoke(out)

    assert returned is out
    assert out.user_marker is marker
    assert out._version == version + 1
    assert tuple(out._shape) == tuple(expected.shape)
    torch.testing.assert_close(out.cpu(), expected)
    # Python methods and operators that read TensorImpl directly must agree
    # with the eager payload after the identity-preserving resize.
    assert tuple(out.shape) == tuple(expected.shape)
    assert out.numel() == expected.numel()
    assert torch.numel(out) == expected.numel()
    assert out.ndimension() == expected.ndimension()
    assert out.flatten().shape == expected.flatten().shape


@pytest.mark.parametrize("case", ["strided", "empty"])
def test_linalg_vector_norm_out_strided_and_empty(
    mojo_gpu: str, call_checker: CallChecker, case: str
):
    _watch_eager_op(call_checker, "aten::linalg_vector_norm.out")

    if case == "strided":
        base = torch.linspace(-3.0, 4.0, 35).reshape(5, 7)
        input = base.t()
        device_input = base.to(mojo_gpu).t()
        dim = [1]
        keepdim = True
        assert not device_input._is_contiguous
    else:
        input = torch.empty((2, 0), dtype=torch.float32)
        device_input = input.to(mojo_gpu)
        dim = [1]
        keepdim = False

    expected = torch.linalg.vector_norm(input, 2, dim, keepdim)
    out = torch.empty_like(expected, device=mojo_gpu)
    holder, ptr = out._holder, out._ptr
    returned = torch.ops.aten.linalg_vector_norm.out(
        device_input, 2, dim, keepdim, dtype=None, out=out
    )

    assert returned is out
    assert out._holder is holder
    assert out._ptr == ptr
    torch.testing.assert_close(out.cpu(), expected)


def test_mul_tensor_inplace_preserves_strided_alias(
    mojo_gpu: str, call_checker: CallChecker
):
    _watch_eager_op(call_checker, "aten::mul_.Tensor")

    base = torch.arange(12, dtype=torch.float32).to(mojo_gpu)
    view = base[::2]
    observer = base.view(3, 4)
    coefficient = torch.tensor(0.25, dtype=torch.float32).to(mojo_gpu)
    expected = torch.arange(12, dtype=torch.float32)
    expected[::2] *= 0.25
    holder, ptr = view._holder, view._ptr
    version = view._version

    returned = torch.ops.aten.mul_.Tensor(view, coefficient)

    assert returned is view
    assert view._holder is holder
    assert view._ptr == ptr
    assert view._version == version + 1
    torch.testing.assert_close(observer.cpu().reshape(-1), expected)


@pytest.mark.parametrize("storage_offset", [0, 2])
def test_mul_out_resize_preserves_existing_storage_alias(
    mojo_gpu: str, call_checker: CallChecker, storage_offset: int
):
    _watch_eager_op(call_checker, "aten::mul.out")

    base = torch.arange(8, dtype=torch.float32).to(mojo_gpu)
    out = base[storage_offset:storage_offset]
    lhs = torch.tensor([2.0, 3.0], device=mojo_gpu)
    rhs = torch.tensor([5.0, 7.0], device=mojo_gpu)
    holder = base._holder

    returned = torch.ops.aten.mul.out(lhs, rhs, out=out)

    assert returned is out
    assert out._holder is holder is base._holder
    assert out.shape == (2,)
    assert torch.numel(out) == 2
    torch.testing.assert_close(out.cpu(), torch.tensor([10.0, 21.0]))
    expected_base = torch.arange(8, dtype=torch.float32)
    expected_base[storage_offset : storage_offset + 2] = torch.tensor([10.0, 21.0])
    torch.testing.assert_close(base.cpu(), expected_base)


@pytest.mark.parametrize(
    ("op_name", "valid_dtype", "invalid_dtype"),
    [
        ("aten::linalg_vector_norm.out", torch.float32, torch.float16),
        ("aten::any.out", torch.uint8, torch.int64),
        ("aten::isin.Tensor_Tensor_out", torch.bool, torch.int64),
    ],
)
def test_out_variants_enforce_operation_specific_dtype_contracts(
    mojo_gpu: str,
    call_checker: CallChecker,
    op_name: str,
    valid_dtype: torch.dtype,
    invalid_dtype: torch.dtype,
):
    _watch_eager_op(call_checker, op_name)

    if op_name == "aten::linalg_vector_norm.out":
        input = torch.tensor([3.0, 4.0], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.linalg_vector_norm.out(
                input, 2, None, False, dtype=None, out=out
            )

        expected = torch.tensor(5.0, dtype=valid_dtype)
    elif op_name == "aten::any.out":
        input = torch.tensor([[0, 2, 0], [0, 0, 0]], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.any.out(input, 1, False, out=out)

        expected = torch.tensor([1, 0], dtype=valid_dtype)
    else:
        input = torch.tensor([1, 2, 3], device=mojo_gpu)
        test_elements = torch.tensor([2, 4], device=mojo_gpu)

        def invoke(out):
            return torch.ops.aten.isin.Tensor_Tensor_out(
                input, test_elements, assume_unique=False, invert=False, out=out
            )

        expected = torch.tensor([False, True, False], dtype=valid_dtype)

    valid_out = torch.empty(0, dtype=valid_dtype, device=mojo_gpu)
    assert invoke(valid_out) is valid_out
    assert valid_out.dtype == valid_dtype
    torch.testing.assert_close(valid_out.cpu(), expected)

    invalid_out = torch.empty(0, dtype=invalid_dtype, device=mojo_gpu)
    with pytest.raises(RuntimeError):
        invoke(invalid_out)
