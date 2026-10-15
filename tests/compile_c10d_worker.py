"""torchrun worker for tests/test_distributed.py: collectives inside graphs
compiled with `mojo_backend` (not a pytest module).

Every check compiles a function that issues collectives the way user code
does on CUDA -- functional collectives, `dist.*` calls dynamo rewrites,
DTensor tensor parallelism -- and compares it with the same function run
eagerly on the mojo device. Exits non-zero on failure.
"""

# ruff: noqa: E402 -- use_local_rank_gpu() must run before torch/MAX initialize
import datetime
import os
import sys
from collections.abc import Callable

from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from torch_mojo_backend import mojo_backend, register_mojo_devices

_MOJO_CCL = os.environ.get("TORCH_MOJO_BACKEND_CCL") == "mojo"


def _check(failures: list[str], name: str, ok: bool, detail: str = ""):
    rank = dist.get_rank()
    print(f"[rank {rank}] {'OK  ' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        failures.append(name)


def _compare(
    failures: list[str],
    name: str,
    fn: Callable[..., object],
    *inputs: torch.Tensor,
    exact: bool = False,
    fullgraph: bool = True,
):
    torch._dynamo.reset()
    expected = fn(*[x.clone() for x in inputs])
    compiled = torch.compile(fn, backend=mojo_backend, fullgraph=fullgraph)
    for attempt in range(3):  # a reused graph must give the same answer
        got = compiled(*[x.clone() for x in inputs])
        exp_list = expected if isinstance(expected, list | tuple) else [expected]
        got_list = got if isinstance(got, list | tuple) else [got]
        ok = len(exp_list) == len(got_list)
        detail = ""
        for e, g in zip(exp_list, got_list):
            assert isinstance(e, torch.Tensor) and isinstance(g, torch.Tensor)
            if e.shape != g.shape or e.dtype != g.dtype or g.device != e.device:
                ok, detail = False, f"{g.shape}/{g.dtype}/{g.device} vs {e.shape}"
                break
            e_cpu, g_cpu = e.cpu().float(), g.cpu().float()
            close = (
                torch.equal(e_cpu, g_cpu)
                if exact
                else torch.allclose(e_cpu, g_cpu, rtol=1e-5, atol=1e-5)
            )
            if not close:
                ok = False
                detail = f"max abs diff {(e_cpu - g_cpu).abs().max().item()}"
                break
        _check(failures, f"{name}[call {attempt}]", ok, detail)


def run_functional(failures: list[str]):
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.group.WORLD
    dev = torch.accelerator.current_accelerator()
    assert dev is not None
    g = torch.Generator().manual_seed(1234 + rank)

    def rand(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(*shape, generator=g).to(dtype).to(dev)

    def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
        return funcol.all_reduce(x * 2, "sum", group) + 1

    def all_reduce_avg(x: torch.Tensor) -> torch.Tensor:
        return funcol.all_reduce(x, "avg", group).sin()

    def all_reduce_max(x: torch.Tensor) -> torch.Tensor:
        return funcol.all_reduce(x, "max", group)

    def dist_all_reduce_in_place(x: torch.Tensor) -> torch.Tensor:
        y = x * 3
        dist.all_reduce(y)
        return y - 1

    def all_gather_dim0(x: torch.Tensor) -> torch.Tensor:
        return funcol.all_gather_tensor(x + 1, 0, group) * 2

    def all_gather_dim1(x: torch.Tensor) -> torch.Tensor:
        # dynamo graph-breaks in the chunk/cat funcol adds for gather_dim != 0
        # (torch._utils._maybe_view_chunk_cat is skipped), on CUDA too
        return funcol.all_gather_tensor(x, 1, group)

    def reduce_scatter(x: torch.Tensor) -> torch.Tensor:
        return funcol.reduce_scatter_tensor(x.cos(), "sum", 0, group)

    def broadcast(x: torch.Tensor) -> torch.Tensor:
        return funcol.broadcast(x * (rank + 1), world - 1, group) + 0

    def coalesced(a: torch.Tensor, b: torch.Tensor) -> list[torch.Tensor]:
        ra, rb = funcol.all_reduce_coalesced([a, b * 2], "sum", group)
        return [ra + 1, rb]

    def chain(x: torch.Tensor) -> torch.Tensor:
        gathered = funcol.all_gather_tensor(x, 0, group)
        scattered = funcol.reduce_scatter_tensor(gathered * 3, "avg", 0, group)
        return funcol.all_reduce(scattered, "sum", group) + gathered.sum()

    def overlapped(x: torch.Tensor, w: torch.Tensor) -> list[torch.Tensor]:
        # Two collectives in flight over independent compute: the temporary
        # each one reads (x * 2, x + 1) dies at its start in the graph, so a
        # missing keepalive lets MAX reuse it for the matmul's output.
        a = funcol.all_reduce(x * 2, "sum", group)
        b = funcol.all_gather_tensor(x + 1, 0, group)
        h = (w @ w).relu() @ w
        return [a + h.sum(), b, h]

    _compare(failures, "all_reduce_sum", all_reduce_sum, rand(357, 789))
    _compare(failures, "all_reduce_avg", all_reduce_avg, rand(1000))
    if not _MOJO_CCL:  # mojoccl reduces SUM and AVG only
        _compare(failures, "all_reduce_max", all_reduce_max, rand(64, 3), exact=True)
    _compare(
        failures, "all_reduce_bf16", all_reduce_sum, rand(512, 64, dtype=torch.bfloat16)
    )
    _compare(failures, "dist_all_reduce_in_place", dist_all_reduce_in_place, rand(33))
    _compare(failures, "all_gather_dim0", all_gather_dim0, rand(7, 5), exact=True)
    _compare(
        failures,
        "all_gather_dim1",
        all_gather_dim1,
        rand(6, 4),
        exact=True,
        fullgraph=False,
    )
    _compare(failures, "reduce_scatter", reduce_scatter, rand(world * 129, 3))
    _compare(failures, "broadcast", broadcast, rand(4, 4), exact=True)
    _compare(failures, "coalesced", coalesced, rand(10), rand(3, 3))
    _compare(failures, "chain", chain, rand(world * 8, 16))
    _compare(failures, "overlapped", overlapped, rand(4096, 1024), rand(1024, 1024))


def run_interleaved(failures: list[str]):
    """Eager and compiled collectives on one communicator, alternating: both
    reach the comm stream in program order, and every rank in the same one."""
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.accelerator.current_accelerator()
    assert dev is not None

    # The size changes every call, so dynamo recompiles with a symbolic one:
    # the all-gather's output is then `s0 * world`, and the reduce-scatter
    # of `z` (`s1` rows) gives `s1 // world`, a shape MAX cannot load as a
    # buffer (collectives.py's copying variant).
    @torch.compile(backend=mojo_backend, fullgraph=True)
    def step(
        x: torch.Tensor, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = funcol.all_reduce(x, "sum", dist.group.WORLD)
        gathered = funcol.all_gather_tensor(y * 2, 0, dist.group.WORLD)
        scattered = funcol.reduce_scatter_tensor(gathered, "sum", 0, dist.group.WORLD)
        return (
            gathered,
            scattered,
            funcol.reduce_scatter_tensor(z, "sum", 0, dist.group.WORLD),
        )

    total = sum(r + 1 for r in range(world))
    ok = True
    for i in range(20):
        n = 1 + (i * 7919) % 50000
        eager = torch.full((n,), float(rank + 1), device=dev)
        dist.all_reduce(eager)
        gathered_c, scattered_c, z_c = step(
            torch.full((n,), float(rank + 1), device=dev),
            torch.arange(world * n, dtype=torch.float32, device=dev),
        )
        gathered = [torch.empty(n, device=dev) for _ in range(world)]
        dist.all_gather(gathered, eager)
        ok &= bool((eager.cpu() == total).all())
        ok &= gathered_c.shape == (world * n,) and scattered_c.shape == (n,)
        ok &= bool((gathered_c.cpu() == 2 * total).all())
        ok &= bool((scattered_c.cpu() == 2 * total * world).all())
        ok &= all(bool((t.cpu() == total).all()) for t in gathered)
        z_expected = world * torch.arange(rank * n, (rank + 1) * n, dtype=torch.float32)
        ok &= z_c.shape == (n,) and torch.equal(z_c.cpu(), z_expected)
    _check(failures, "interleaved_eager_and_compiled", ok)


class _MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up = torch.nn.Linear(64, 256)
        self.down = torch.nn.Linear(256, 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.gelu(self.up(x)))


def run_tensor_parallel(failures: list[str]):
    """The standard Megatron MLP split: the compiled module's forward and
    backward graphs carry the all-reduces DTensor inserts."""
    dev = torch.accelerator.current_accelerator()
    assert dev is not None
    mesh = init_device_mesh(dev.type, (dist.get_world_size(),))
    plan = {"up": ColwiseParallel(), "down": RowwiseParallel()}

    def build() -> _MLP:
        torch.manual_seed(0)
        model = _MLP().to(dev)
        return parallelize_module(model, mesh, plan)

    torch.manual_seed(1)
    x = torch.randn(8, 16, 64).to(dev)
    eager, compiled = build(), build()
    compiled_fn = torch.compile(compiled, backend=mojo_backend)
    for step in range(2):
        out_e = eager(x)
        out_e.square().mean().backward()
        out_c = compiled_fn(x)
        out_c.square().mean().backward()
        ok = torch.allclose(out_e.cpu(), out_c.cpu(), rtol=1e-4, atol=1e-5)
        for (name, pe), pc in zip(eager.named_parameters(), compiled.parameters()):
            ge, gc = pe.grad, pc.grad
            assert ge is not None and gc is not None
            ok &= torch.allclose(
                ge.full_tensor().cpu(), gc.full_tensor().cpu(), rtol=1e-4, atol=1e-5
            )
        _check(failures, f"tensor_parallel_mlp[step {step}]", ok)


def run_unsupported(failures: list[str]):
    dev = torch.accelerator.current_accelerator()
    assert dev is not None

    def all_to_all(x: torch.Tensor) -> torch.Tensor:
        return funcol.all_to_all_single(x, None, None, dist.group.WORLD)

    try:
        torch.compile(all_to_all, backend=mojo_backend, fullgraph=True)(
            torch.ones(dist.get_world_size() * 2, device=dev)
        )
        _check(failures, "all_to_all_refused", False, "compiled without an error")
    except Exception as e:  # dynamo wraps the backend's error
        _check(
            failures, "all_to_all_refused", "all_to_all_single" in str(e), str(e)[:200]
        )


def main():
    mode = sys.argv[1]
    register_mojo_devices()
    dist.init_process_group(backend="mojo", timeout=datetime.timedelta(seconds=300))
    failures: list[str] = []
    if mode == "functional":
        run_functional(failures)
    elif mode == "interleaved":
        run_interleaved(failures)
    elif mode == "tensor_parallel":
        run_tensor_parallel(failures)
    elif mode == "unsupported":
        run_unsupported(failures)
    else:
        raise ValueError(f"unknown mode {mode}")
    dist.destroy_process_group()
    if failures:
        print(f"[rank {os.environ['RANK']}] FAILURES: {failures}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
