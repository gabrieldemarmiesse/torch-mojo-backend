# Upstream issue drafts

Issues to open against `modular/modular` and `pytorch/pytorch`, drafted while
implementing single-process multi-GPU data parallelism for the mojo eager
device (branch `multi-gpu-m0`, see `docs/multi_gpu_training_plan.md`).

Context common to all of them: torch-mojo-backend is a PyTorch PrivateUse1
backend (pure Python + JIT-compiled Mojo extensions, no C++ extension, no
NCCL/cuBLAS) that runs standard `DistributedDataParallel` in one process with
thread-per-rank ranks, gradient allreduce on MAX's pure-Mojo P2P comm kernels
(`max/kernels/src/comm/`), and NCCL-style comm/compute overlap on side
streams.

Versions everything was verified against:

- modular at the commit pinning nightly `max==26.5.0.dev2026061806`
  (`12beae6f14`, "[Release] Pin lockfiles to Mojo 1.0.0b3.dev2026061806...")
- pytorch `v2.11.0` (`70d99e998b4`), installed `torch 2.11.0+cu130`
- 8x H100 SXM5 (NVLink/NVSwitch), driver 570.211.01

## modular/modular

| file | title | blocks |
|---|---|---|
| [modular-1](modular-1-stream-launch-closure-captures.md) | Closure captures don't survive `compile_function` + `DeviceStream.enqueue_function` under `--emit shared-lib` | priority comm streams (M3) |
| [modular-2](modular-2-collectives-on-nondefault-stream.md) | No way to run `comm` collectives on a non-default / priority stream | comm/compute overlap quality |
| [modular-3](modular-3-explicit-rank-for-collectives.md) | `allreduce`/`reducescatter`/`broadcast` derive the rank from `ctx.id()` — device subsets can't participate | collectives on arbitrary GPU subsets |
| [modular-4](modular-4-2stage-allreduce-signal-lifetime.md) | 2-stage allreduce has no end barrier — signal-payload lifetime contract is undocumented | safe signal-buffer reuse/free |

## pytorch/pytorch

| file | title | blocks |
|---|---|---|
| [pytorch-1](pytorch-1-privateuse1-guard-device-count.md) | Python-backend PrivateUse1 DeviceGuard hardcodes `deviceCount() == 1` — backward on `privateuseone:i` (i>=1) hits an INTERNAL ASSERT | per-device autograd queues for Python backends |
| [pytorch-2](pytorch-2-pinned-memory-hook-python-backend.md) | DDP `find_unused_parameters=True` requires a pinned-memory allocator Python backends cannot register | DDP feature coverage |
| [pytorch-3](pytorch-3-futurewrappingwork-error-propagation.md) | Errors don't propagate through `_create_work_from_future` Works (`wait()` never rethrows; `set_exception` is wrapper-local) | clean async-PG error reporting |
