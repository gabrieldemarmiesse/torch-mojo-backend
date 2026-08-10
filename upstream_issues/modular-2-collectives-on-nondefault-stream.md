# [kernels][comm] No supported way to run the `comm` collectives on a non-default (priority) stream

## Summary

The public collective entry points (`comm.allreduce.allreduce`,
`comm.allgather.allgather`, `comm.reducescatter.reducescatter`,
`comm.broadcast.broadcast`) take a `ctx: DeviceContext` and enqueue their
kernels via `ctx.enqueue_function[...]`, which always targets the context's
default stream (`AsyncRT_DeviceContext_enqueueFunctionDirect`;
`mojo/stdlib/std/gpu/host/device_context.mojo:4710-4770`). There is no
parameter to launch on a caller-provided `DeviceStream`, and no
set-current-stream API on `DeviceContext`.

For NCCL-style comm/compute overlap in eager mode this is the missing piece:
the collective must run on a persistent raised-priority side stream, ordered
against the default stream with events, so that compute enqueued after the
collective does not queue behind it — the pattern
`max/kernels/src/shmem/shmem_context.mojo:560-606` already implements for
SHMEM kernels (`_priority_stream = ctx.create_stream(priority=
ctx.stream_priority_range().greatest)` + begin/end events).

## What we tried

1. **Direct kernel launches on a `DeviceStream`** (replicating
   `_allreduce_p2p`'s dispatch): blocked by the closure-capture lowering
   failure in `--emit shared-lib` builds — see the companion issue
   "[Mojo][GPU] Closure captures fail to lower when a kernel is compiled with
   `compile_function` and launched via `DeviceStream.enqueue_function`". The
   collectives' `output_lambda` epilogue is a capturing closure, so only the
   fused `ctx.enqueue_function[kernel]` path works today.

2. **A second owning `DeviceContext(device_id=i)` per GPU as the "comm
   stream"** — this is our current workaround, and it works: we verified
   (via `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_CONTEXT)`) that MAX
   device contexts for one device share the CUDA primary context, so
   buffers, peer access and events interop across context instances, each
   context has its own default stream, and `ctx.id()` still returns the
   device id (which the kernels use as the rank). But:
   - that stream cannot be given a priority
     (`AsyncRT_DeviceContext_create` takes only `(api, id)`,
     `device_context.mojo:4774-4823`), and without priority a full-grid
     collective time-slices against fat compute kernels instead of
     preempting them;
   - a second context duplicates AsyncRT bookkeeping (its own buffer cache
     and compiled-function cache).

## Ask (any one of these solves it)

- The collectives accept an optional launch target (a `DeviceStream`, or a
  `_FunctionEnqueuer`) defaulting to today's behavior; or
- `AsyncRT_DeviceContext_create` / `DeviceContext.__init__` grows a stream
  `priority` for the context's default stream; or
- fix the closure-capture + `DeviceStream.enqueue_function` path (companion
  issue), after which callers can replicate the dispatch themselves.

Related smaller gaps noticed while building this (can file separately if
preferred):

- Python `max.driver.DeviceStream.__init__(device)` exposes no `priority`
  (Mojo-only feature today).
- Mojo `DeviceEvent` has `synchronize()` but no non-blocking query
  (`AsyncRT_DeviceEvent_*` exposes no query/poll), so host-side event
  reaping needs a dedicated blocking thread.

## Environment

modular @ `12beae6f14` (nightly 26.5.0.dev2026061806), 8x H100 NVLink.
Consumer: github.com/gabrieldemarmiesse/torch-mojo-backend, single-process
multi-GPU DDP on the `comm` kernels (measured ~320-330 GB/s busbw at 256 MiB
— NCCL-class; the remaining gap to NCCL is overlap scheduling).
