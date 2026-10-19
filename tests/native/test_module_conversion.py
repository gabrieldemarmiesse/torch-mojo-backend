"""`nn.Module.to()` / `.half()` / `.cuda()` with the mojo device registered.

`Module._apply` keeps a Parameter object and moves it in place
(`param.data = moved`) when `torch._has_compatible_shallow_copy_type` says
so; the mojo device answers that through its own kernel, as CUDA does
through TensorImpl's dense-backend list. Registration must not touch the
process-wide `torch.__future__` conversion flags, which would change the
behaviour of every CPU and CUDA model in the process too.
"""

import weakref

import pytest
import torch
from torch import nn


@pytest.fixture
def cuda() -> str:
    if not torch.cuda.is_available():
        pytest.skip("needs a working CUDA build of torch")
    return "cuda:0"


def _mlp() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3))


def _move_keeps_identity(module: nn.Module, move) -> list[nn.Parameter]:
    """Apply `move`; assert every Parameter object, and a weakref to it,
    survives (what CPU <-> CUDA does). Returns the parameters."""
    before = list(module.parameters())
    refs = [weakref.ref(p) for p in before]
    move(module)
    after = list(module.parameters())
    assert len(after) == len(before)
    for p, q, r in zip(before, after, refs):
        assert p is q
        assert r() is p
    return after


def test_registration_leaves_torch_future_flags_alone():
    assert not torch.__future__.get_swap_module_params_on_conversion()
    assert not torch.__future__.get_overwrite_module_params_on_conversion()


# The two cases a process-wide swap flag broke for ordinary models: a
# weakref to a parameter (an EMA helper) and a live autograd graph that saved
# one both raised "Couldn't swap Linear.weight".
def _with_weakrefs(module: nn.Module) -> list[weakref.ref[nn.Parameter]]:
    return [weakref.ref(p) for p in module.parameters()]


def _with_live_graph(module: nn.Module) -> list[torch.Tensor]:
    x = torch.randn(2, 4, requires_grad=True, device=next(module.parameters()).device)
    return [(module(x) ** 2).sum()]


@pytest.mark.parametrize("holder", [_with_weakrefs, _with_live_graph])
def test_cpu_model_converts_in_place(holder):
    m = _mlp()
    keep = holder(m)
    _move_keeps_identity(m, lambda mod: mod.to(torch.float64))
    assert all(p.dtype == torch.float64 for p in m.parameters())
    del keep


# Needs no mojo device, but a CUDA one only the GPU runners have.
@pytest.mark.gpu
@pytest.mark.parametrize("holder", [_with_weakrefs, _with_live_graph])
def test_cuda_model_converts_in_place(cuda, holder):
    m = _mlp().to(cuda)
    keep = holder(m)
    _move_keeps_identity(m, lambda mod: mod.half())
    _move_keeps_identity(m, lambda mod: mod.cpu())
    assert all(p.device.type == "cpu" for p in m.parameters())
    del keep


@pytest.mark.parametrize("holder", [_with_weakrefs, _with_live_graph])
def test_mojo_model_round_trip_keeps_parameters(mojo_gpu, holder):
    m = _mlp()
    ref = [p.detach().clone() for p in m.parameters()]
    keep = holder(m)
    params = _move_keeps_identity(m, lambda mod: mod.to(mojo_gpu))
    assert all(p.device == torch.device(mojo_gpu) for p in params)
    keep += holder(m)
    _move_keeps_identity(m, lambda mod: mod.to("cpu"))
    for p, r in zip(m.parameters(), ref):
        assert p.device.type == "cpu"
        torch.testing.assert_close(p.detach(), r, rtol=0, atol=0)
    del keep


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_mojo_model_dtype_cast_in_place(mojo_gpu, dtype):
    m = _mlp()
    ref = [p.detach().to(dtype) for p in m.parameters()]
    m.to(mojo_gpu)
    params = _move_keeps_identity(m, lambda mod: mod.to(dtype))
    for p, r in zip(params, ref):
        assert p.dtype == dtype and p.device.type == "mojo"
        assert p.requires_grad
        torch.testing.assert_close(p.detach().cpu(), r, rtol=0, atol=0)
    params = _move_keeps_identity(m, lambda mod: mod.float())
    assert all(p.dtype == torch.float32 for p in params)


def test_mojo_tied_weights_survive_moves_and_casts(mojo_gpu):
    class Tied(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.projection = nn.Linear(8, 16, bias=False)
            self.projection.weight = self.embedding.weight

    m = Tied()
    for move in (
        lambda mod: mod.to(mojo_gpu),
        lambda mod: mod.half(),
        lambda mod: mod.float(),
        lambda mod: mod.to("cpu"),
        lambda mod: mod.to(mojo_gpu),
    ):
        move(m)
        assert m.embedding.weight is m.projection.weight
        assert len(list(m.parameters())) == 1
    assert m.embedding.weight.device.type == "mojo"
    tokens = torch.tensor([[1, 5, 7]], device=mojo_gpu)
    m.projection(m.embedding(tokens)).sum().backward()
    assert m.embedding.weight.grad is not None
    assert m.embedding.weight.grad.device.type == "mojo"


def test_mojo_move_keeps_grads_hooks_and_optimizer(mojo_gpu):
    m = _mlp()
    ref = _mlp()
    x = torch.randn(5, 4)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.1, momentum=0.9)

    # A first step on CPU leaves a .grad on every parameter.
    for mod, o in ((m, opt), (ref, ref_opt)):
        mod(x).square().sum().backward()
        o.step()

    calls = []
    first = m[0]
    assert isinstance(first, nn.Linear)
    first.weight.register_hook(lambda g: calls.append(g.device.type))
    params = _move_keeps_identity(m, lambda mod: mod.to(mojo_gpu))
    for p in params:
        assert p.grad is not None and p.grad.device.type == "mojo"
    assert opt.param_groups[0]["params"] == params

    # The momentum buffers are keyed by the same Parameter objects.
    for mojo_state in opt.state.values():
        mojo_state["momentum_buffer"] = mojo_state["momentum_buffer"].to(mojo_gpu)
    for mod, o, inp in ((m, opt, x.to(mojo_gpu)), (ref, ref_opt, x)):
        o.zero_grad()
        mod(inp).square().sum().backward()
        o.step()
    assert calls == ["mojo"]
    for p, r in zip(m.parameters(), ref.parameters()):
        torch.testing.assert_close(p.detach().cpu(), r.detach(), rtol=1e-5, atol=1e-5)


def test_mojo_load_state_dict_copies_into_parameters(mojo_gpu):
    src = _mlp()
    m = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 3)).to(mojo_gpu)
    params = list(m.parameters())
    m.load_state_dict(src.state_dict())
    for p, q, s in zip(params, m.parameters(), src.parameters()):
        assert p is q and p.device.type == "mojo"
        torch.testing.assert_close(p.detach().cpu(), s.detach(), rtol=0, atol=0)


def test_set_data_across_cpu_and_mojo(mojo_gpu):
    """`.data =` goes through the dispatched op (pytorch#196975 suspected
    otherwise), in both directions."""
    p = nn.Parameter(torch.zeros(3))
    p.data = torch.ones(3, device=mojo_gpu)
    assert p.device.type == "mojo" and isinstance(p, nn.Parameter)
    torch.testing.assert_close(p.detach().cpu(), torch.ones(3))
    p.data = torch.full((3,), 2.0)
    assert p.device.type == "cpu"
    torch.testing.assert_close(p.detach(), torch.full((3,), 2.0))


def test_has_compatible_shallow_copy_type_matches_dense_backends(mojo_gpu):
    """Mojo answers like a dense backend; Meta and sparse arguments keep
    torch's own answer (False against any dense tensor, as against CUDA)."""
    compatible = torch._has_compatible_shallow_copy_type
    cpu = torch.zeros(2)
    dev = torch.zeros(2, device=mojo_gpu)
    assert compatible(cpu, dev) and compatible(dev, cpu)
    assert compatible(dev, dev)
    # More dims than a mojo tensor may have: only the device type is read.
    assert compatible(torch.zeros((1,) * 12), dev)
    assert compatible(nn.Parameter(cpu), dev)
    meta = torch.zeros(2, device="meta")
    assert not compatible(meta, dev) and not compatible(dev, meta)
    sparse = torch.zeros(2, 2).to_sparse()
    assert not compatible(sparse, dev) and not compatible(dev, sparse)


def test_meta_to_empty_on_mojo(mojo_gpu):
    """`to_empty` from Meta replaces the Parameters, as for CUDA."""
    with torch.device("meta"):
        m = _mlp()
    before = list(m.parameters())
    m.to_empty(device=mojo_gpu)
    for p, old in zip(m.parameters(), before):
        assert isinstance(p, nn.Parameter) and p.device.type == "mojo"
        assert p is not old


def test_mojo_and_cuda_interplay(mojo_gpu, cuda):
    m = _mlp()
    ref = [p.detach().clone() for p in m.parameters()]
    keep = _with_weakrefs(m)
    _move_keeps_identity(m, lambda mod: mod.to(mojo_gpu))
    _move_keeps_identity(m, lambda mod: mod.to(cuda))
    assert all(p.device.type == "cuda" for p in m.parameters())
    _move_keeps_identity(m, lambda mod: mod.to(mojo_gpu))
    for p, r in zip(m.parameters(), ref):
        assert p.device.type == "mojo"
        torch.testing.assert_close(p.detach().cpu(), r, rtol=0, atol=0)
    del keep
