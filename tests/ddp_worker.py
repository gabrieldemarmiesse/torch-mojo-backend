"""torchrun worker for tests/test_distributed.py (not a pytest module).

Runs one validation mode per invocation and exits non-zero on failure so the
parent test only has to check the return code. Every mode drives the mojo
distributed backend through public torch.distributed / torch.mojo APIs only:
the native backend's tensors are ordinary ``torch.Tensor``s (see
agents_docs/native_backend.md), so there is no more ``TorchMojoTensor``, no ctypes
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
import datetime
import json
import os
import pickle
import signal
import sys
import threading
import time
from collections.abc import Iterator
from typing import cast

# One GPU per rank, decided before anything can initialize CUDA/MAX.
from torch_mojo_backend.distributed import use_local_rank_gpu

use_local_rank_gpu()

import torch
import torch.distributed as dist
from torch._C import _distributed_c10d as c10d_c
from torch.nn.parallel import DistributedDataParallel as DDP

from torch_mojo_backend import monkeypatching, register_mojo_devices
from torch_mojo_backend.distributed import flight_recorder
from torch_mojo_backend.distributed.process_group import MojoProcessGroup
from torch_mojo_backend.native import device_module

# mojoccl (torch_mojo_backend/mojo/tmb/ccl) implements AllReduce/
# Broadcast/AllGather/ReduceScatter -- Reduce/Send/Recv/AllToAll return
# ncclInvalidUsage (DDP needs only the first three; gather and scatter are
# built from all-gather and broadcast). Every mode below skips the checks
# that need an op mojoccl does not implement.
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
    # mojoccl implements SUM and AVG only (agents_docs/distributed.md, "Mojo
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

    # AVG scales each contribution before summing (NCCL's PreMulSum), so the
    # average of values whose SUM overflows is still finite.
    hot = torch.full((1024,), 3.0e38, device="mojo")
    dist.all_reduce(hot, op=dist.ReduceOp.AVG)
    _check(failures, "all_reduce.AVG.no_overflow", bool((hot.cpu() == 3.0e38).all()))

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
        rs_in = [torch.full((2,), float(rank + r), device="mojo") for r in range(world)]
        dist.reduce_scatter(rs_out, rs_in)
        exp_list = sum(float(rank + r) for r in range(world))
        _check(failures, "reduce_scatter.list", bool((rs_out.cpu() == exp_list).all()))

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
    with _tolerate_missing_ops(rank, "gather"):
        mine_g = torch.full((6,), float(rank), device="mojo")
        gather_list = (
            [torch.zeros(6, device="mojo") for _ in range(world)] if rank == 0 else None
        )
        dist.gather(mine_g, gather_list, dst=0)
        if rank == 0:
            assert gather_list is not None
            ok = all(
                (gather_list[r].cpu() == float(r)).all().item() for r in range(world)
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
        work.wait()  # orders the current stream, does not block the host
        torch.accelerator.synchronize()
        _check(failures, "async_work.is_completed", work.is_completed())
        _check(failures, "async_work.result", bool((at.cpu() == total).all()))
        ft = torch.full((1000,), float(rank + 1), device="mojo")
        later = torch.zeros(8, device="mojo")
        work = dist.all_reduce(ft, async_op=True)
        assert work is not None
        dist.all_reduce(later)  # queued behind: the Future must not depend on it
        value = work.get_future().wait()[0]
        _check(failures, "async_work.future", bool((value.cpu() == total).all()))
        try:
            work.wait(datetime.timedelta(seconds=1))
            _check(failures, "async_work.finite_timeout_rejected", False)
        except RuntimeError:
            _check(failures, "async_work.finite_timeout_rejected", True)

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
    "Distributed" in agents_docs/native_backend.md). 256 MiB per collective on
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
    allreduce/broadcast/all_gather/reduce_scatter run under every ccl;
    reduce/all_to_all/gather/scatter/send-recv (mojoccl does not implement them) only
    under vendor NCCL/RCCL.
    """
    rank = dist.get_rank()
    world = dist.get_world_size()
    total = float(world * (world + 1) // 2)
    sizes = [1, 3, 257, 1024, 1003, 4096, 65537]  # 1 elem, non-16B, several MiB
    rounds = 40

    ok_allreduce = ok_broadcast = ok_allgather = ok_reduce_scatter = True
    ok_tiny_int64 = True
    for i in range(rounds):
        n = sizes[i % len(sizes)]

        t = torch.full((n,), float(rank + 1), device="mojo")
        dist.all_reduce(t)
        ok_allreduce = ok_allreduce and bool((t.cpu() == total).all())

        # The historical AMD ordering flake involved exactly one int64.
        # Change its value each generation so stale flags/payloads cannot pass.
        tiny = torch.full((1,), rank + i, dtype=torch.int64, device="mojo")
        dist.all_reduce(tiny)
        ok_tiny_int64 = ok_tiny_int64 and tiny.cpu().item() == (
            world * i + world * (world - 1) // 2
        )

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
        # Changing values distinguish the current generation from stale
        # staging/inbox data; an odd input offset exercises scalar tails.
        base = torch.arange(world * n + 1, dtype=torch.float32) % 17
        src = (base + rank + i).to("mojo")[1:]
        out = torch.empty(n, device="mojo")
        op = dist.ReduceOp.SUM if i % 2 else dist.ReduceOp.AVG
        dist.reduce_scatter_tensor(out, src, op=op)
        expected = (base[1:].reshape(world, n)[rank] + i) * world + world * (
            world - 1
        ) // 2
        if op == dist.ReduceOp.AVG:
            expected /= world
        # AVG scales each contribution before summing (PreMulSum): exact only
        # when 1/world is, i.e. for a power-of-two world.
        exact = op == dist.ReduceOp.SUM or world & (world - 1) == 0
        got = out.cpu()
        ok_reduce_scatter = ok_reduce_scatter and (
            bool((got == expected).all())
            if exact
            else torch.allclose(got, expected, rtol=1e-6, atol=1e-6)
        )
    _check(failures, "stress.allreduce", ok_allreduce)
    _check(failures, "stress.allreduce_tiny_int64", ok_tiny_int64)
    _check(failures, "stress.broadcast", ok_broadcast)
    _check(failures, "stress.allgather", ok_allgather)
    _check(failures, "stress.reduce_scatter", ok_reduce_scatter)

    if _MOJO_CCL:
        _skip(rank, "stress.reduce/all_to_all/gather/scatter/send_recv")
        dist.barrier()
        return

    full_rounds = 20
    with _tolerate_missing_ops(
        rank, "stress.reduce/all_to_all/gather/scatter/send_recv"
    ):
        ok_reduce = ok_alltoall = True
        ok_gather = ok_scatter = ok_send_recv = True
        for i in range(full_rounds):
            n = sizes[i % len(sizes)]

            r = torch.full((n,), float(rank + 1), device="mojo")
            dist.reduce(r, dst=0)
            if rank == 0:
                ok_reduce = ok_reduce and bool((r.cpu() == total).all())

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


def _fr_tag_isolation(t: torch.Tensor):
    """An all_reduce whose tag another thread writes between the call's
    `_fr_op` and its native entry keeps its own record: every thread tags
    through a header slot of its own."""
    group = cast(MojoProcessGroup, dist.group.WORLD)
    fr = group._fr
    assert fr is not None
    group._fr_op("all_reduce", [t], [t])
    thread = threading.Thread(target=lambda: fr.begin("broadcast", [t], [t], 0))
    thread.start()
    thread.join()
    index = t.device.index or 0
    group._core.check(
        group._core.allreduce(
            group._handle,
            index,
            t.data_ptr(),
            t.numel(),
            7,
            0,  # float32, sum
        ),
        "ncclAllReduce",
    )
    fr.clear()


def _fr_wrap_stress(t: torch.Tensor, size: int) -> bool:
    """Dumps taken on another thread while 300 collectives lap a small ring:
    each holds at most `size` entries, with consecutive record ids."""
    stop = threading.Event()
    bad: list[str] = []
    dumps = [0]

    def snapshot():
        while not stop.is_set():
            entries = pickle.loads(c10d_c._dump_nccl_trace(True, False))["entries"]  # ty: ignore[unresolved-attribute] -- stubbed for CUDA builds only; register_mojo_devices() installs it
            ids = [e["record_id"] for e in entries]
            first = ids[0] if ids else 0
            if len(ids) > size or ids != list(range(first, first + len(ids))):
                bad.append(str(ids))
            dumps[0] += 1
            # A Python thread that never lets go of the GIL starves the
            # issuing one (each boxed op it runs takes the GIL back for its
            # refcounts), whatever it does: pace it like a real dumper.
            time.sleep(0.001)

    thread = threading.Thread(target=snapshot)
    thread.start()
    for _ in range(300):
        dist.all_reduce(t)
    torch.accelerator.synchronize()
    stop.set()
    thread.join()
    print(f"[rank {dist.get_rank()}] {dumps[0]} dumps during the laps", flush=True)
    return not bad and dumps[0] > 0


def run_flight_recorder(failures: list[str]):
    """The flight recorder seen through torch's own entry points: every
    collective of this run in `_dump_nccl_trace` with ProcessGroupNCCL's
    names, seq ids and shapes (list collectives as the stacked buffer),
    completed (a coalescing block's operations stay scheduled, as
    ProcessGroupNCCL's do); per-thread tags; the JSON variant; the ring
    keeping only the newest TORCH_FR_BUFFER_SIZE; a dump through the debug
    pipe."""
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = torch.accelerator.current_accelerator()
    assert dev is not None
    size = int(os.environ.get("TORCH_FR_BUFFER_SIZE", "2000"))
    t = torch.ones(4, 8, device=dev)
    dist.all_reduce(t)
    dist.broadcast(t, 0)
    gathered = torch.empty(4 * world, 8, device=dev)
    dist.all_gather_into_tensor(gathered, t)
    scattered = torch.empty(4 // world, 8, device=dev)
    dist.reduce_scatter_tensor(scattered, t)
    dist.all_gather([torch.empty_like(t) for _ in range(world)], t)
    dist.reduce_scatter(torch.empty_like(t), [t.clone() for _ in range(world)])
    expected = [
        "all_reduce",
        "broadcast",
        "_all_gather_base",
        "_reduce_scatter_base",
        "all_gather",
        "reduce_scatter",
    ]
    members = []
    if not _MOJO_CCL:
        # One way, even ranks to the next: fr_trace pairs a coalesced
        # block's ops starting from its first rank's and looks a recv's
        # peer up by its own rank, so it only matches blocks where that
        # rank sends (on NCCL's dumps too).
        if rank % 2 == 0:
            op = dist.P2POp(dist.isend, t, rank + 1)
            members = [f"send {rank}->{rank + 1}"]
        else:
            op = dist.P2POp(dist.irecv, torch.empty_like(t), rank - 1)
            members = [f"recv {rank}<-{rank - 1}"]
        for work in dist.batch_isend_irecv([op]):
            work.wait()
        expected += [*members, "coalesced"]
    for _ in range(12):
        dist.all_reduce(t)
    expected += ["all_reduce"] * 12
    if size:
        _fr_tag_isolation(t)
    expected += ["all_reduce"]
    torch.accelerator.synchronize()
    time.sleep(0.5)  # the watchdog retires (stamps) them within a poll
    dump = pickle.loads(c10d_c._dump_nccl_trace())  # ty: ignore[unresolved-attribute] -- stubbed for CUDA builds only; register_mojo_devices() installs it
    entries = dump["entries"]
    names = [e["profiling_name"].split(":", 1)[1] for e in entries]
    if size == 0:
        _check(failures, "fr.buffer_size_0_records_nothing", entries == [])
        _check(
            failures,
            "fr.buffer_size_0_starts_no_monitor",
            not any(th.name == "mojo-pg-fr-monitor" for th in threading.enumerate()),
        )
        return
    _check(failures, "fr.names", names == expected[-size:])
    ids = [e["record_id"] for e in entries]
    _check(
        failures,
        "fr.ring_keeps_the_newest",
        ids == list(range(len(expected) - len(ids), len(expected))),
    )
    _check(
        failures,
        "fr.states",
        all(
            e["state"] == ("scheduled" if n in members else "completed")
            for e, n in zip(entries, names)
        ),
    )
    coll = [e["collective_seq_id"] for e in entries if not e["is_p2p"]]
    _check(failures, "fr.collective_seq_ids", coll == sorted(coll))
    _check(
        failures,
        "fr.frames_start_in_torch_distributed",
        all(
            e["frames"] and "torch/distributed" in e["frames"][0]["filename"]
            for e in entries[:-1]
            if not e["profiling_name"].endswith("coalesced")
        ),
    )
    _check(
        failures,
        "fr.tags_per_thread",
        any(f["name"] == "_fr_tag_isolation" for f in entries[-1]["frames"]),
    )
    group = cast(MojoProcessGroup, dist.group.WORLD)
    _check(failures, "fr.native_selftest", group._core.fr_selftest() == 0)
    if size >= len(expected):
        sizes = [(e["input_sizes"], e["output_sizes"]) for e in entries[:6]]
        _check(
            failures,
            "fr.shapes",
            sizes[2] == ([[4, 8]], [[4 * world, 8]])
            and sizes[4] == ([[4, 8]], [[world, 4, 8]])
            and sizes[5] == ([[world, 4, 8]], [[4, 8]])
            and entries[0]["input_dtypes"] == ["Float"],
        )
    as_json = json.loads(c10d_c._dump_nccl_trace_json())  # ty: ignore[unresolved-attribute] -- stubbed for CUDA builds only; register_mojo_devices() installs it
    _check(
        failures,
        "fr.json",
        len(as_json["entries"]) == len(entries)
        and "frames" not in as_json["entries"][0],
    )
    _check(
        failures,
        "fr.pg_config",
        dump["pg_config"]["0"]
        == {"name": "0", "desc": "default_pg", "ranks": str(list(range(world)))},
    )
    stem = os.environ.get("TORCH_NCCL_DEBUG_INFO_PIPE_FILE")
    if stem:
        path = os.environ["TORCH_FR_DUMP_TEMP_FILE"] + str(rank)
        _check(failures, "fr.pipe_exists", os.path.exists(f"{stem}{rank}.pipe"))
        with open(f"{stem}{rank}.pipe", "w") as pipe:
            pipe.write("dump")
        deadline = time.monotonic() + 10
        while not os.path.exists(path) and time.monotonic() < deadline:
            time.sleep(0.1)
        _check(failures, "fr.pipe_dumps", os.path.exists(path))
        dist.barrier()  # both dumps written before torchrun reaps a rank
    if size <= 16:
        _check(failures, "fr.snapshots_while_the_ring_wraps", _fr_wrap_stress(t, size))


def run_fr_subgroup():
    """A gloo default group and a mojo group of global rank 1 alone (its
    rank 0): the pipe, the dump file and the group's ranks use the global
    rank."""
    dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=300))
    rank = dist.get_rank()
    group = dist.new_group([1], backend="mojo")
    failures: list[str] = []
    if rank == 1:
        t = torch.ones(4, device="mojo")
        dist.all_reduce(t, group=group)
        torch.accelerator.synchronize()
        stem = os.environ["TORCH_NCCL_DEBUG_INFO_PIPE_FILE"]
        path = os.environ["TORCH_FR_DUMP_TEMP_FILE"] + "1"
        deadline = time.monotonic() + 10
        _check(failures, "fr.subgroup_pipe_exists", os.path.exists(f"{stem}1.pipe"))
        with open(f"{stem}1.pipe", "w") as pipe:
            pipe.write("dump")
        while not os.path.exists(path) and time.monotonic() < deadline:
            time.sleep(0.1)
        _check(
            failures, "fr.subgroup_dumps_under_the_global_rank", os.path.exists(path)
        )
        if os.path.exists(path):
            with open(path, "rb") as f:
                dump = pickle.load(f)
            _check(
                failures,
                "fr.subgroup_ranks",
                [g["ranks"] for g in dump["pg_config"].values()] == ["[1]"],
            )
    dist.barrier()
    dist.destroy_process_group()
    if failures:
        print(f"[rank {rank}] FAILURES: {failures}", flush=True)
        raise SystemExit(1)
    raise SystemExit(0)


# The timeout of the process group the `desync*` modes hang: what its
# watchdog waits. The default group keeps 300 s, so rendezvous, communicator
# creation and first-use kernel builds (a cold cache) never count against it.
DESYNC_TIMEOUT_S = 10


def _desync_group() -> tuple[int, dist.ProcessGroup, torch.Tensor]:
    """A group with the short timeout, its collective #1 done on every rank
    (after a warm-up on the default group)."""
    rank = dist.get_rank()
    t = torch.ones(1024, device="mojo")
    dist.all_reduce(t)
    torch.accelerator.synchronize()
    group = dist.new_group(
        list(range(dist.get_world_size())),
        timeout=datetime.timedelta(seconds=DESYNC_TIMEOUT_S),
    )
    assert group is not None
    dist.all_reduce(t, group=group)  # #1 of the short group, on every rank
    torch.accelerator.synchronize()
    print(f"[rank {rank}] all_reduce #1 done", flush=True)
    return rank, group, t


def _outlive_the_watchdog(rank: int):
    """Rank 0's abort releases its wait and the process exits right behind
    it; rank 1 is stopped by torchrun. Getting past here is the failure."""
    time.sleep(DESYNC_TIMEOUT_S if rank == 0 else 10 * DESYNC_TIMEOUT_S)
    print(f"[rank {rank}] UNEXPECTED: still alive after the timeout", flush=True)
    raise SystemExit(3)


def run_desync():
    """Rank 0 issues an all_reduce rank 1 never issues; rank 1 stops issuing.
    The watchdog must report collective #2 on rank 0 within the timeout and
    exit the process non-zero, which makes torchrun stop rank 1."""
    rank, group, t = _desync_group()
    if rank == 0:
        dist.all_reduce(t, group=group)  # #2: nobody else issues it
        torch.accelerator.synchronize()
    _outlive_the_watchdog(rank)


def run_desync_burst():
    """More collectives in flight than the watchdog's ring holds, then the
    unmatched one: it must still be the one reported (#BURST + 2), not a
    dropped commit. The matmuls hold the comm stream back so the burst
    really is in flight at once."""
    rank, group, t = _desync_group()
    x = torch.randn(4096, 4096, device="mojo")
    for _ in range(100):
        x = x @ x
    small = torch.ones(1, device="mojo")
    for _ in range(DESYNC_BURST):
        dist.all_reduce(small, group=group)
    if rank == 0:
        dist.all_reduce(torch.ones(4, device="mojo"), group=group)  # unmatched
    torch.accelerator.synchronize()
    _outlive_the_watchdog(rank)


DESYNC_BURST = 3000
OVERFLOW_BURST = 4000
OVERFLOW_LIMIT = 256  # in flight: well under the libraries' own ~1024


def run_desync_overflow():
    """The watchdog's queue full (lowered to OVERFLOW_LIMIT by a test hook):
    the burst's newest collectives go unwatched, and once it drained
    tracking resumes -- the unmatched collective issued afterwards is
    reported with its exact sequence number (#OVERFLOW_BURST + 2)."""
    rank, group, t = _desync_group()
    assert isinstance(group, MojoProcessGroup)
    group._test_watchdog_limit(0, OVERFLOW_LIMIT)
    x = torch.randn(4096, 4096, device="mojo")
    for _ in range(200):
        x = x @ x
    small = torch.ones(1, device="mojo")
    for _ in range(OVERFLOW_BURST):
        dist.all_reduce(small, group=group)
    torch.accelerator.synchronize()
    print(f"[rank {rank}] overflow burst done", flush=True)
    if rank == 0:
        dist.all_reduce(torch.ones(4, device="mojo"), group=group)  # unmatched
        torch.accelerator.synchronize()
    _outlive_the_watchdog(rank)


def run_desync_at_exit():
    """The script ends with an unmatched collective in flight and no
    destroy_process_group: the exit hook gives it a short grace, then
    reports it and the process exits 1 -- within seconds, not the timeout."""
    rank, group, t = _desync_group()
    if rank == 0:
        dist.all_reduce(t, group=group, async_op=True)  # #2, unmatched
        print("[rank 0] leaving with a collective in flight", flush=True)
        raise SystemExit(0)
    _outlive_the_watchdog(rank)


def run_desync_ctrl_c():
    """Ctrl-C on a job hung in a collective: same as an exit, within
    seconds."""
    rank, group, t = _desync_group()
    if rank == 0:
        dist.all_reduce(t, group=group)  # #2, unmatched
        os.kill(os.getpid(), signal.SIGINT)
        time.sleep(DESYNC_TIMEOUT_S)  # the KeyboardInterrupt lands here
    _outlive_the_watchdog(rank)


def run_desync_exit_host_blocked():
    """The main thread exits while a daemon thread is blocked on the host in
    its first point-to-point call (vendor NCCL connects the pair inside
    ncclGroupEnd, waiting for the peer): reported at exit, not swallowed."""
    rank, group, t = _desync_group()
    if rank == 0:
        threading.Thread(
            target=lambda: dist.irecv(
                torch.zeros(4, device="mojo"), src=1, group=group
            ),
            daemon=True,
        ).start()
        time.sleep(1)
        print("[rank 0] leaving with a call blocked on the host", flush=True)
        raise SystemExit(0)
    _outlive_the_watchdog(rank)


def run_watchdog_off_destroy():
    """TORCH_MOJO_BACKEND_PG_WATCHDOG=off: destroying a group with async
    collectives still in flight -- rank 1 issues its side 3 s later --
    completes them rather than cancelling them."""
    rank, group, t = _desync_group()
    bufs = [torch.full((1 << 16,), float(i), device="mojo") for i in range(20)]
    if rank == 1:
        time.sleep(3)
    for b in bufs:
        dist.all_reduce(b, group=group, async_op=True)
    dist.destroy_process_group(group)
    ok = all(bool((b.cpu() == 2.0 * i).all()) for i, b in enumerate(bufs))
    print(f"[rank {rank}] off destroy correct: {ok}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0)


def run_watchdog_slow_destroy():
    """A healthy run whose ranks reach destroy_process_group far apart (rank
    0 still writing its checkpoint, longer than the timeout): the destroy
    does not wait for peers, so nothing times out and the run exits 0."""
    rank, group, t = _desync_group()
    if rank == 0:
        time.sleep(DESYNC_TIMEOUT_S + 5)
    dist.destroy_process_group(group)
    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] slow_destroy finished", flush=True)
    raise SystemExit(0)


def run_desync_report_twice():
    """TORCH_MOJO_BACKEND_PG_WATCHDOG=report, three late collectives in turn:
    each is reported, completes after all, and its rank's record leaves the
    store, so the next report (rank 1 reports twice) sees only fresh ones."""
    rank, group, t = _desync_group()
    late = DESYNC_TIMEOUT_S + 5
    for late_rank in (0, 1, 0):  # #2, #3, #4: this rank issues it late
        if rank == late_rank:
            time.sleep(late)
        dist.all_reduce(t, group=group)
        torch.accelerator.synchronize()
        print(f"[rank {rank}] late all_reduce done", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] report_twice finished", flush=True)
    raise SystemExit(0)


def run_desync_destroy():
    """Destroying a group whose collective hangs: the destroy waits for it,
    so the watchdog reports it and exits instead of the destroy hanging."""
    rank, group, t = _desync_group()
    if rank == 0:
        dist.all_reduce(t, group=group)  # #2: nobody else issues it
        dist.destroy_process_group(group)
    _outlive_the_watchdog(rank)


# Under the libraries' own limit: vendor NCCL's work FIFO and mojoccl's
# queue both block the host near 1024 launches waiting for a peer.
ASYNC_BURST = 900


def run_watchdog_async_burst():
    """Rank 0 issues ASYNC_BURST async all_reduces, then tells rank 1
    through the store to issue its matching ones: valid async work in which
    the issuing thread must return from every call before the peer can make
    progress. The watchdog must neither block the issuer nor report
    anything."""
    rank, group, _ = _desync_group()
    store = dist.distributed_c10d._get_default_store()
    bufs = [torch.full((4,), float(i), device="mojo") for i in range(ASYNC_BURST)]
    if rank == 0:
        for b in bufs:
            dist.all_reduce(b, group=group, async_op=True)
        store.set("watchdog_async_go", "1")
    else:
        store.wait(["watchdog_async_go"])
        for b in bufs:
            dist.all_reduce(b, group=group, async_op=True)
    torch.accelerator.synchronize()
    ok = all(bool((b.cpu() == 2.0 * i).all()) for i, b in enumerate(bufs))
    print(f"[rank {rank}] async burst correct: {ok}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    raise SystemExit(0)


def run_desync_host_blocked():
    """Rank 0's first point-to-point call to rank 1 while rank 1 stopped
    issuing: NCCL connects the pair inside ncclGroupEnd, which blocks on
    the host. The in-call mark must report it and the process exit."""
    rank, group, _ = _desync_group()
    if rank == 0:
        dist.irecv(torch.zeros(4, device="mojo"), src=1, group=group)
    _outlive_the_watchdog(rank)


def run_watchdog_clean_exit():
    """Collectives still in flight when the interpreter exits without
    destroy_process_group: no report, exit status 0."""
    rank, group, t = _desync_group()
    big = torch.ones(1 << 20, device="mojo")
    for _ in range(50):
        dist.all_reduce(big, group=group, async_op=True)
        dist.all_reduce(t, async_op=True)
    print(f"[rank {rank}] exiting without destroy_process_group", flush=True)
    raise SystemExit(0)


def run_watchdog_fault_destroy():
    """TORCH_MOJO_BACKEND_PG_WATCHDOG=report: a driver error on rank 0's
    communicator is reported, and destroying the group afterwards aborts
    that communicator instead of trusting its (now unwatched) queue."""
    rank, group, t = _desync_group()
    assert isinstance(group, MojoProcessGroup)
    # On every rank: an abort facing a destroy is not something vendor NCCL
    # promises to finish (its destroy may wait for the peer's).
    group._test_watchdog_fault(0)
    time.sleep(1)
    dist.destroy_process_group(group)
    dist.barrier()
    dist.destroy_process_group()
    print(f"[rank {rank}] fault_destroy finished", flush=True)
    raise SystemExit(0)


DESYNC_MODES = {
    "desync": run_desync,
    "desync_burst": run_desync_burst,
    "desync_report_twice": run_desync_report_twice,
    "desync_destroy": run_desync_destroy,
    "desync_host_blocked": run_desync_host_blocked,
    "desync_overflow": run_desync_overflow,
    "desync_at_exit": run_desync_at_exit,
    "desync_ctrl_c": run_desync_ctrl_c,
    "desync_exit_host_blocked": run_desync_exit_host_blocked,
    "watchdog_off_destroy": run_watchdog_off_destroy,
    "watchdog_slow_destroy": run_watchdog_slow_destroy,
    "watchdog_async_burst": run_watchdog_async_burst,
    "watchdog_clean_exit": run_watchdog_clean_exit,
    "watchdog_fault_destroy": run_watchdog_fault_destroy,
}


def main():
    mode = sys.argv[1]

    register_mojo_devices()
    if mode == "fr_subgroup":
        run_fr_subgroup()
    dist.init_process_group(backend="mojo", timeout=datetime.timedelta(seconds=300))
    if mode in DESYNC_MODES:
        DESYNC_MODES[mode]()
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
    elif mode == "flight_recorder":
        run_flight_recorder(failures)
    else:
        raise ValueError(f"unknown mode {mode}")
    dist.destroy_process_group()
    if mode == "flight_recorder" and int(os.environ.get("TORCH_FR_BUFFER_SIZE", "1")):
        # Every mojo group destroyed: torch's own `_dump_nccl_trace` again.
        configs = pickle.loads(c10d_c._dump_nccl_trace())["pg_config"]  # ty: ignore[unresolved-attribute] -- stubbed for CUDA builds only; register_mojo_devices() installs it
        torch_has_one = monkeypatching._NCCL_TRACE_ORIGINALS[0] is not None
        ok = not flight_recorder.active() and not (torch_has_one and "0" in configs)
        rank = os.environ["RANK"]  # no default group to ask any more
        print(f"[rank {rank}] {'OK  ' if ok else 'FAIL'} fr.torch_dump_after_destroy")
        if not ok:
            failures.append("fr.torch_dump_after_destroy")
    if failures:
        print(f"[rank {os.environ['RANK']}] FAILURES: {failures}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
