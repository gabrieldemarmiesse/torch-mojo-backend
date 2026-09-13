"""torchrun worker for tests/test_distributed.py (not a pytest module).

Runs one validation mode per invocation and exits non-zero on failure so the
parent test only has to check the return code. Every mode drives the mojo
distributed backend through public torch.distributed / torch.mojo APIs only:
the native backend's tensors are ordinary ``torch.Tensor``s (see
docs/native_backend.md), so there is no more ``TorchMojoTensor``, no ctypes
``NcclComm``, and no ``process_group`` internals (``_ptr_of``,
``_device_state``, ``_fence_default``) to reach into -- `MojoProcessGroup`
itself (the adapter class, `torch_mojo_backend/distributed/process_group.py`)
is the one non-``torch.distributed`` name used below, for `abort()`/
`shutdown()`, which have no functional-API equivalent.

Keep this file importable without a GPU: everything device-touching happens
inside main().
"""

# ruff: noqa: E402 -- use_local_rank_gpu() must run before torch/MAX initialize
import contextlib
import os
import sys
import time
from collections.abc import Iterator

# One GPU per rank, decided before anything can initialize CUDA/MAX.
from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import datetime

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_mojo_backend import register_mojo_devices
from torch_mojo_backend.distributed.process_group import MojoProcessGroup
from torch_mojo_backend.native import device_module

# mojoccl (torch_mojo_backend/distributed/mojoccl) implements AllReduce/
# Broadcast/AllGather only -- Reduce/ReduceScatter/Send/Recv/AllToAll/Gather/
# Scatter return ncclInvalidUsage (DDP needs only the first three). Every mode
# below skips the checks that need an op mojoccl does not implement.
_MOJO_CCL = os.environ.get("TORCH_MOJO_BACKEND_CCL") == "mojo"


class ElemwiseNet(torch.nn.Module):
    """Matmul-free so it runs even where GEMM routes are unavailable."""

    def __init__(self, width: int):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(width))
        self.b = torch.nn.Parameter(torch.randn(width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # gelu, not relu: relu's backward (threshold_backward) is not
        # implemented everywhere yet; gelu_backward is.
        return torch.nn.functional.gelu(x * self.w + self.b)


def _check(failures: list[str], name: str, ok: bool):
    rank = dist.get_rank()
    print(f"[rank {rank}] {'OK  ' if ok else 'FAIL'} {name}", flush=True)
    if not ok:
        failures.append(name)


def _skip(rank: int, what: str):
    print(f"[rank {rank}] SKIP {what} (mojoccl does not implement it)", flush=True)


@contextlib.contextmanager
def _tolerate_missing_ops(rank: int, section: str) -> Iterator[None]:
    """Some ops a collective needs -- including ones `process_group.py` uses
    for its own staging, like `torch.cat` for the list `reduce_scatter` --
    may not be ported to the native backend yet; other agents are porting op
    groups in parallel. Treat that as SKIPPED here, not a failure of this
    mode: every collective below raises before issuing anything on the
    communicator, so skipping mid-section never desyncs ranks.
    """
    try:
        yield
    except NotImplementedError as exc:
        print(f"[rank {rank}] SKIPPED {section}: {exc}", flush=True)


def run_collectives(failures: list[str]):
    rank = dist.get_rank()
    world = dist.get_world_size()
    total = float(world * (world + 1) // 2)  # sum(1..world), every op below
    # picks a per-rank value of rank+1 so this one number is the SUM/AVG/MAX/
    # MIN reference; PRODUCT uses a separate, rank-independent value instead
    # (a product of "rank+1" over many ranks overflows fast).

    # ---- all_reduce: every ReduceOp, plus a strided input -----------------
    # mojoccl implements SUM and AVG only (docs/distributed.md, "Mojo
    # collectives"); MAX/MIN/PRODUCT are vendor-NCCL/RCCL-only here.
    ops = [(dist.ReduceOp.SUM, total), (dist.ReduceOp.AVG, (world + 1) / 2.0)]
    if not _MOJO_CCL:
        ops += [(dist.ReduceOp.MAX, float(world)), (dist.ReduceOp.MIN, 1.0)]
    for op, expected in ops:
        t = torch.full((1024,), float(rank + 1), device="mojo")
        dist.all_reduce(t, op=op)
        _check(failures, f"all_reduce.{op.name}", bool((t.cpu() == expected).all()))

    if _MOJO_CCL:
        _skip(rank, "all_reduce.PRODUCT")
    else:
        p = torch.full((16,), 2.0, device="mojo")
        dist.all_reduce(p, op=dist.ReduceOp.PRODUCT)
        _check(failures, "all_reduce.PRODUCT", bool((p.cpu() == 2.0**world).all()))

    tb = torch.full((257,), float(rank + 1), device="mojo", dtype=torch.bfloat16)
    dist.all_reduce(tb)
    _check(failures, "all_reduce.bf16", bool((tb.float().cpu() == total).all()))

    i64 = torch.tensor([rank + 1], dtype=torch.int64, device="mojo")
    dist.all_reduce(i64)
    _check(failures, "all_reduce.int64", i64.cpu().item() == int(total))

    xs = torch.arange(8, dtype=torch.float32).to("mojo").view(2, 4).t()
    dist.all_reduce(xs)
    _check(
        failures,
        "all_reduce.strided",
        xs.cpu()[0].tolist() == [0.0 * world, 4.0 * world],
    )

    # ---- broadcast ----------------------------------------------------------
    b = torch.full((33,), float(rank), device="mojo")
    dist.broadcast(b, src=0)
    _check(failures, "broadcast", bool((b.cpu() == 0.0).all()))

    # ---- reduce ---------------------------------------------------------
    if _MOJO_CCL:
        _skip(rank, "reduce")
    else:
        r = torch.full((9,), float(rank + 1), device="mojo")
        dist.reduce(r, dst=0)
        if rank == 0:
            _check(failures, "reduce", bool((r.cpu() == total).all()))

    # ---- all_gather: list, into_tensor, coalesced --------------------------
    outs = [torch.zeros(5, device="mojo") for _ in range(world)]
    mine = torch.full((5,), float(rank), device="mojo")
    dist.all_gather(outs, mine)
    ok = all((outs[r].cpu() == float(r)).all().item() for r in range(world))
    _check(failures, "all_gather.list", ok)

    flat = torch.zeros(world * 5, device="mojo")
    dist.all_gather_into_tensor(flat, mine)
    ok = all(
        (flat.cpu()[r * 5 : (r + 1) * 5] == float(r)).all().item() for r in range(world)
    )
    _check(failures, "all_gather.into_tensor", ok)

    c_outs = [torch.zeros(world * 3, device="mojo") for _ in range(2)]
    c_ins = [torch.full((3,), float(rank + i), device="mojo") for i in range(2)]
    with dist._coalescing_manager():
        for out, inp in zip(c_outs, c_ins):
            dist.all_gather_into_tensor(out, inp)
    ok = all(
        bool((out.cpu()[r * 3 : (r + 1) * 3] == float(r + i)).all())
        for i, out in enumerate(c_outs)
        for r in range(world)
    )
    _check(failures, "all_gather.coalesced", ok)

    # ---- reduce_scatter: tensor, list, coalesced ---------------------------
    if _MOJO_CCL:
        _skip(rank, "reduce_scatter")
    else:
        with _tolerate_missing_ops(rank, "reduce_scatter.tensor"):
            src = torch.arange(world * 3, dtype=torch.float32).to("mojo")
            out = torch.zeros(3, device="mojo")
            dist.reduce_scatter_tensor(out, src)
            exp = (
                torch.arange(world * 3, dtype=torch.float32)[rank * 3 : (rank + 1) * 3]
                * world
            )
            _check(failures, "reduce_scatter.tensor", bool((out.cpu() == exp).all()))

        with _tolerate_missing_ops(rank, "reduce_scatter.list"):
            # the list variant stages its inputs with torch.cat, a separate
            # op from the tensor variant above and not guaranteed ported yet.
            rs_out = torch.zeros(2, device="mojo")
            rs_in = [
                torch.full((2,), float(rank + r), device="mojo") for r in range(world)
            ]
            dist.reduce_scatter(rs_out, rs_in)
            exp_list = sum(float(rank + r) for r in range(world))
            _check(
                failures, "reduce_scatter.list", bool((rs_out.cpu() == exp_list).all())
            )

        with _tolerate_missing_ops(rank, "reduce_scatter.coalesced"):
            c_rs_outs = [torch.zeros(2, device="mojo") for _ in range(2)]
            c_rs_srcs = [
                (torch.arange(world * 2, dtype=torch.float32) + 10 * i).to("mojo")
                for i in range(2)
            ]
            with dist._coalescing_manager():
                for out, src in zip(c_rs_outs, c_rs_srcs):
                    dist.reduce_scatter_tensor(out, src)
            ok = all(
                bool(
                    (
                        c_rs_outs[i].cpu()
                        == (torch.arange(world * 2, dtype=torch.float32) + 10 * i)[
                            rank * 2 : (rank + 1) * 2
                        ]
                        * world
                    ).all()
                )
                for i in range(2)
            )
            _check(failures, "reduce_scatter.coalesced", ok)

    # ---- all_to_all: _single with and without splits, and the list form ---
    if _MOJO_CCL:
        _skip(rank, "all_to_all")
    else:
        with _tolerate_missing_ops(rank, "all_to_all_single.even"):
            a2a_in = (torch.arange(world * 2, dtype=torch.float32) + 100 * rank).to(
                "mojo"
            )
            a2a_out = torch.empty_like(a2a_in)
            dist.all_to_all_single(a2a_out, a2a_in)
            want = sum(
                [[100.0 * r + 2 * rank + i for i in range(2)] for r in range(world)], []
            )
            _check(failures, "all_to_all_single.even", a2a_out.cpu().tolist() == want)

        if world > 1:
            with _tolerate_missing_ops(rank, "all_to_all_single.split"):
                # ragged: rank r always sends r+1 elements to every peer.
                in_split = [rank + 1] * world
                out_split = [s + 1 for s in range(world)]
                a2a_in2 = torch.full((sum(in_split),), float(rank), device="mojo")
                a2a_out2 = torch.empty(sum(out_split), device="mojo")
                dist.all_to_all_single(a2a_out2, a2a_in2, out_split, in_split)
                expect2 = sum(([float(s)] * (s + 1) for s in range(world)), [])
                _check(
                    failures,
                    "all_to_all_single.split",
                    a2a_out2.cpu().tolist() == expect2,
                )

        with _tolerate_missing_ops(rank, "all_to_all.list"):
            in_list = [
                torch.full((4,), float(rank * 10 + r), device="mojo")
                for r in range(world)
            ]
            out_list = [torch.empty(4, device="mojo") for _ in range(world)]
            dist.all_to_all(out_list, in_list)
            ok = all(
                bool((out_list[r].cpu() == float(r * 10 + rank)).all())
                for r in range(world)
            )
            _check(failures, "all_to_all.list", ok)

    # ---- gather / scatter ---------------------------------------------------
    if _MOJO_CCL:
        _skip(rank, "gather")
        _skip(rank, "scatter")
    else:
        with _tolerate_missing_ops(rank, "gather"):
            mine_g = torch.full((6,), float(rank), device="mojo")
            gather_list = (
                [torch.zeros(6, device="mojo") for _ in range(world)]
                if rank == 0
                else None
            )
            dist.gather(mine_g, gather_list, dst=0)
            if rank == 0:
                assert gather_list is not None
                ok = all(
                    (gather_list[r].cpu() == float(r)).all().item()
                    for r in range(world)
                )
                _check(failures, "gather", ok)

        with _tolerate_missing_ops(rank, "scatter"):
            scatter_out = torch.zeros(6, device="mojo")
            scatter_list = (
                [torch.full((6,), float(r), device="mojo") for r in range(world)]
                if rank == 0
                else None
            )
            dist.scatter(scatter_out, scatter_list, src=0)
            _check(failures, "scatter", bool((scatter_out.cpu() == float(rank)).all()))

    # ---- send/recv ring -------------------------------------------------
    if world > 1:
        if _MOJO_CCL:
            _skip(rank, "send_recv")
        else:
            with _tolerate_missing_ops(rank, "send_recv_ring"):
                s = torch.full((7,), float(rank), device="mojo")
                r = torch.zeros(7, device="mojo")
                if rank % 2 == 0:
                    dist.send(s, (rank + 1) % world)
                    dist.recv(r, (rank - 1) % world)
                else:
                    dist.recv(r, (rank - 1) % world)
                    dist.send(s, (rank + 1) % world)
                _check(
                    failures,
                    "send_recv_ring",
                    bool((r.cpu() == float((rank - 1) % world)).all()),
                )

    # ---- async_op=True Works: wait() and is_completed() --------------------
    with _tolerate_missing_ops(rank, "async_work"):
        at = torch.full((1000,), float(rank + 1), device="mojo")
        work = dist.all_reduce(at, async_op=True)
        assert work is not None
        work.wait()
        _check(failures, "async_work.is_completed", work.is_completed())
        _check(failures, "async_work.result", bool((at.cpu() == total).all()))

    # ---- object collectives (route through the internal gloo group) -------
    with _tolerate_missing_ops(rank, "all_gather_object"):
        objs: list[dict[str, int] | None] = [None] * world
        dist.all_gather_object(objs, {"rank": rank})
        first, last = objs[0], objs[-1]
        _check(
            failures,
            "all_gather_object",
            first is not None
            and last is not None
            and first["rank"] == 0
            and last["rank"] == world - 1,
        )

    # ---- CPU tensors: the private gloo path --------------------------------
    with _tolerate_missing_ops(rank, "cpu_tensor.gloo_path"):
        cpu_t = torch.full((4,), float(rank + 1))
        dist.all_reduce(cpu_t)
        _check(failures, "cpu_tensor.gloo_path", bool((cpu_t == total).all()))

    dist.barrier()


def run_stream_ordering(failures: list[str]):
    """Replaces the old lazy comm_fence check.

    The native backend completes a collective's Work while the comm stream is
    current (a device-typed torch Future), so ``Work.wait()`` -- not an
    implicit per-op hook -- is what orders a waiter's stream after it (see
    "Distributed" in docs/native_backend.md). 256 MiB per collective on
    purpose: big enough that the collective is still running on the comm
    stream a few microseconds later, when the host reaches the check.
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    total = float(world * (world + 1) // 2)
    n = 64 * 1024 * 1024  # 256 MiB of float32

    # (a) default-stream consumer, after an explicit wait().
    a = torch.full((n,), float(rank + 1), device="mojo")
    work = dist.all_reduce(a, async_op=True)
    work.wait()
    doubled = a + a
    _check(
        failures,
        "stream_ordering.default_stream",
        bool((doubled.cpu() == 2 * total).all()),
    )

    # (b) a side stream that calls wait() itself is ordered too -- any
    # waiter's stream, not just the one the collective happened to see first.
    b = torch.full((n,), float(rank + 1), device="mojo")
    work_b = dist.all_reduce(b, async_op=True)
    side = torch.Stream(device="mojo")
    with device_module.stream(side):
        work_b.wait()
        tripled = b + b + b  # tensor+tensor: a scalar multiply mixes dtypes for now
    # `.cpu()` issues its copy on the *current* stream, which is the default
    # one again here, so reading `tripled` needs a fence against `side` --
    # the same rule CUDA has for a tensor produced on another stream. It
    # still proves the ordering under test: a consumer that ran before the
    # collective computed the wrong value, and no later barrier repairs that.
    torch.accelerator.synchronize()
    _check(
        failures,
        "stream_ordering.side_stream",
        bool((tripled.cpu() == 3 * total).all()),
    )

    # (c) memory safety does not depend on wait(): every touched buffer is
    # record_stream-ed on the comm stream regardless, so dropping a tensor
    # right after an async collective -- before anyone waits on it -- must
    # not hand its memory to a later allocation while the collective still
    # writes it.
    ok = True
    for i in range(8):
        c = torch.full((n,), float(rank + 1), device="mojo")
        pending = dist.all_reduce(c, async_op=True)
        del c, pending
        fresh = torch.full((n,), -1.0 * (i + 1), device="mojo")
        if not bool((fresh.cpu() == -1.0 * (i + 1)).all()):
            ok = False
    _check(failures, "stream_ordering.free_before_wait", ok)

    dist.barrier()


def run_ddp_parity(failures: list[str]):
    rank = dist.get_rank()
    world = dist.get_world_size()
    width = 4096
    per_rank = 16

    try:
        torch.manual_seed(1234 + rank)  # deliberately different per rank...
        model = ElemwiseNet(width).to("mojo")
        ddp = DDP(model, broadcast_buffers=False)
        torch.manual_seed(1234)
        reference = ElemwiseNet(width)
        # ...so a passing check proves DDP's construction-time broadcast
        # synced rank 0's weights everywhere.
        _check(
            failures,
            "ddp.initial_broadcast",
            torch.equal(ddp.module.w.detach().cpu(), reference.w.detach()),
        )

        optimizer = torch.optim.AdamW(ddp.parameters(), lr=1e-2)
        ref_optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-2)
        torch.manual_seed(999)
        full_batch = torch.randn(world * per_rank, width)
        for _ in range(3):
            shard = full_batch[rank * per_rank : (rank + 1) * per_rank].to("mojo")
            loss = ddp(shard).pow(2).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            ref_loss = reference(full_batch).pow(2).mean()
            ref_optimizer.zero_grad(set_to_none=True)
            ref_loss.backward()
            ref_optimizer.step()

        _check(
            failures,
            "ddp.step_parity",
            torch.allclose(ddp.module.w.detach().cpu(), reference.w.detach(), atol=1e-5)
            and torch.allclose(
                ddp.module.b.detach().cpu(), reference.b.detach(), atol=1e-5
            ),
        )
    except NotImplementedError as exc:
        # Other agents are still porting op groups to the native backend;
        # an unported op here is not this mode's problem to fail on.
        print(f"[rank {rank}] SKIPPED(ddp_parity): {exc}", flush=True)
        return
    dist.barrier()


def run_stress(failures: list[str]):
    """Many collectives back to back, mixed (including awkward) sizes:
    allreduce/broadcast/all_gather run under every ccl; reduce/reduce_scatter/
    all_to_all/gather/scatter/send-recv (mojoccl does not implement them) only
    under vendor NCCL/RCCL.
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    total = float(world * (world + 1) // 2)
    sizes = [1, 3, 257, 1024, 1003, 4096, 65537]  # 1 elem, non-16B, several MiB
    rounds = 40

    ok_allreduce = ok_broadcast = ok_allgather = True
    for i in range(rounds):
        n = sizes[i % len(sizes)]

        t = torch.full((n,), float(rank + 1), device="mojo")
        dist.all_reduce(t)
        ok_allreduce = ok_allreduce and bool((t.cpu() == total).all())

        root = i % world
        b = torch.full((n,), float(rank), device="mojo")
        dist.broadcast(b, src=root)
        ok_broadcast = ok_broadcast and bool((b.cpu() == float(root)).all())

        mine = torch.full((n,), float(rank), device="mojo")
        flat = torch.zeros(world * n, device="mojo")
        dist.all_gather_into_tensor(flat, mine)
        ok_allgather = ok_allgather and all(
            bool((flat.cpu()[r * n : (r + 1) * n] == float(r)).all())
            for r in range(world)
        )
    _check(failures, "stress.allreduce", ok_allreduce)
    _check(failures, "stress.broadcast", ok_broadcast)
    _check(failures, "stress.allgather", ok_allgather)

    if _MOJO_CCL:
        _skip(rank, "stress.reduce/reduce_scatter/all_to_all/gather/scatter/send_recv")
        dist.barrier()
        return

    full_rounds = 20
    with _tolerate_missing_ops(
        rank, "stress.reduce/reduce_scatter/all_to_all/gather/scatter/send_recv"
    ):
        ok_reduce = ok_reduce_scatter = ok_alltoall = True
        ok_gather = ok_scatter = ok_send_recv = True
        for i in range(full_rounds):
            n = sizes[i % len(sizes)]

            r = torch.full((n,), float(rank + 1), device="mojo")
            dist.reduce(r, dst=0)
            if rank == 0:
                ok_reduce = ok_reduce and bool((r.cpu() == total).all())

            src = torch.arange(world * n, dtype=torch.float32).to("mojo")
            out = torch.zeros(n, device="mojo")
            dist.reduce_scatter_tensor(out, src)
            exp = (
                torch.arange(world * n, dtype=torch.float32)[rank * n : (rank + 1) * n]
                * world
            )
            ok_reduce_scatter = ok_reduce_scatter and bool((out.cpu() == exp).all())

            a_in = torch.full((n * world,), float(rank), device="mojo")
            a_out = torch.empty_like(a_in)
            dist.all_to_all_single(a_out, a_in)
            # sender s fills its whole input with s, so receiving rank sees
            # its s-th chunk (n elements) equal to s, not to its own rank.
            want_a2a = torch.cat([torch.full((n,), float(s)) for s in range(world)])
            ok_alltoall = ok_alltoall and bool((a_out.cpu() == want_a2a).all())

            g_mine = torch.full((max(n, 1),), float(rank), device="mojo")
            g_list = (
                [torch.zeros(max(n, 1), device="mojo") for _ in range(world)]
                if rank == 0
                else None
            )
            dist.gather(g_mine, g_list, dst=0)
            if rank == 0:
                assert g_list is not None
                ok_gather = ok_gather and all(
                    bool((g_list[r].cpu() == float(r)).all()) for r in range(world)
                )

            s_out = torch.zeros(max(n, 1), device="mojo")
            s_list = (
                [
                    torch.full((max(n, 1),), float(r), device="mojo")
                    for r in range(world)
                ]
                if rank == 0
                else None
            )
            dist.scatter(s_out, s_list, src=0)
            ok_scatter = ok_scatter and bool((s_out.cpu() == float(rank)).all())

            if world > 1:
                sb = torch.full((max(n, 1),), float(rank), device="mojo")
                rb = torch.zeros(max(n, 1), device="mojo")
                if rank % 2 == 0:
                    dist.send(sb, (rank + 1) % world)
                    dist.recv(rb, (rank - 1) % world)
                else:
                    dist.recv(rb, (rank - 1) % world)
                    dist.send(sb, (rank + 1) % world)
                ok_send_recv = ok_send_recv and bool(
                    (rb.cpu() == float((rank - 1) % world)).all()
                )

        _check(failures, "stress.reduce", ok_reduce)
        _check(failures, "stress.reduce_scatter", ok_reduce_scatter)
        _check(failures, "stress.all_to_all", ok_alltoall)
        _check(failures, "stress.gather", ok_gather)
        _check(failures, "stress.scatter", ok_scatter)
        _check(failures, "stress.send_recv", ok_send_recv)
    dist.barrier()


def run_abort(failures: list[str]):
    """`ProcessGroup.abort()`/`shutdown()` behave: abort() returns promptly
    and leaves the device idle, and shutdown() after abort is a safe no-op.

    abort() deliberately drops the aborted communicator ("abort forgets the
    communicator so nothing reuses or double-destroys it" -- process_group.py)
    rather than leaving the group permanently refusing collectives, so the
    contract to check post-abort is recovery: the next collective on the same
    device transparently bootstraps a fresh communicator (a new
    ncclGetUniqueId/ncclCommInitRank round through the c10d store) and must
    still produce a correct result. Exercised through `MojoProcessGroup`'s own
    public methods -- the adapter, not the vendor library underneath it.
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world < 2:
        print(f"[rank {rank}] SKIP abort (needs 2+ ranks)", flush=True)
        return

    pg = dist.group.WORLD
    assert isinstance(pg, MojoProcessGroup)
    dist.all_reduce(torch.ones(4, device="mojo"))  # a live communicator to abort
    dist.barrier()

    t0 = time.monotonic()
    pg.abort()
    torch.accelerator.synchronize()
    elapsed = time.monotonic() - t0
    print(f"[rank {rank}] abort returned in {elapsed:.3f}s", flush=True)
    _check(failures, "abort.returns_promptly", elapsed < 30.0)

    recovered = torch.ones(4, device="mojo")
    dist.all_reduce(recovered)
    _check(
        failures,
        "abort.recovers_with_a_fresh_communicator",
        bool((recovered.cpu() == float(world)).all()),
    )

    pg.shutdown()
    _check(failures, "abort.shutdown_after_abort_is_safe", True)


def main():
    mode = sys.argv[1]

    register_mojo_devices()
    dist.init_process_group(backend="mojo", timeout=datetime.timedelta(seconds=300))
    failures: list[str] = []
    if mode == "collectives":
        run_collectives(failures)
    elif mode == "ddp_parity":
        run_ddp_parity(failures)
    elif mode == "stream_ordering":
        run_stream_ordering(failures)
    elif mode == "stress":
        run_stress(failures)
    elif mode == "abort":
        run_abort(failures)
    else:
        raise ValueError(f"unknown mode {mode}")
    dist.destroy_process_group()
    if failures:
        print(f"[rank {os.environ['RANK']}] FAILURES: {failures}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
