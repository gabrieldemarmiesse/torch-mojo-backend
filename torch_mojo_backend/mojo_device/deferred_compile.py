"""Deferred op execution while kernel variants compile in the background.

The mojo device is already asynchronous at the GPU level: the host enqueues
kernels and runs ahead. This layer applies the same idea to kernel
*compilation*. When an op needs a kernel that is still compiling
(`KernelPending` from the eager_kernels loader), the op is not executed —
its output metadata is inferred on the meta device, real storage is
allocated, and a closure is appended to a strict-FIFO launch queue. The main
thread keeps going. A single launcher thread drains the queue in order,
waiting on compile jobs as needed (the build pool works in parallel), and
copies each op's real result into the pre-allocated placeholder so earlier
handed-out tensors are backed by the right bytes.

Deferred mode is an episode, not a permanent state: it turns on at the
first `KernelPending` and off once the queue is empty and no build is in
flight; outside an episode dispatch takes the exact synchronous path it
does today. Sync points — reading values, crossing devices — drain the
queue first, preserving eager semantics. Launcher-side errors are held and
re-raised at the next drain, CUDA-style.
"""

import threading
from collections import deque

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from torch_mojo_backend.eager_kernels import KernelPending, _dispatch_scope

_TLS = threading.local()  # .launcher: running inside the launcher thread


class _Undeferrable(Exception):
    """This op's semantics require immediate execution (metadata effects)."""


class _Runtime:
    """Cooperative single-threaded queue: the MAIN thread is the only one
    that ever touches the device. Compiles run in background subprocesses;
    deferred ops wait in a FIFO and are replayed by the main thread itself —
    opportunistically at each dispatch entry (`pump`), exhaustively at sync
    points (`drain`). Replaying from a second thread is NOT correct: MAX
    device work enqueued from another thread is unordered with respect to
    the main thread's (verified empirically — deterministic stale reads)."""

    def __init__(self) -> None:
        self.queue: deque = deque()
        self.active = False
        self.error: BaseException | None = None

    def start_episode(self) -> None:
        self.active = True

    def submit(self, item: tuple) -> None:
        self.queue.append(item)
        self.active = True  # a queued item always (re)opens the episode

    def pump(self) -> None:
        """Execute queue items whose kernels are ready; stop at the first
        head-of-line item still waiting on a compile. Main thread only:
        autograd's engine thread also dispatches (and may defer), but device
        work from two live threads is unordered on MAX, so replays stay on
        the thread whose stream owns the episode."""
        if in_replay() or threading.current_thread() is not threading.main_thread():
            return
        while self.queue and self.error is None:
            item = self.queue.popleft()
            try:
                _execute(item, blocking=False)
            except KernelPending:
                self.queue.appendleft(item)  # head still compiling
                return
            except BaseException as exc:
                self.error = exc
        if not self.queue:
            self.active = False

    def drain(self) -> None:
        """Execute every queued op (waiting out compiles); re-raise errors.
        No-op during a replay: the replay is itself queue consumption."""
        if in_replay():
            return
        while self.queue and self.error is None:
            item = self.queue.popleft()
            try:
                _execute(item, blocking=True)
            except BaseException as exc:
                self.error = exc
        self.active = False
        error, self.error = self.error, None
        if error is not None:
            raise error

    def drain_until_untainted(self, tensors: list) -> None:
        """Execute the FIFO prefix until none of `tensors` has a pending
        write — the minimal wait for an op that must run immediately but
        only depends on part of the queue. Discovery continues afterward
        with the rest of the queue (and its compiles) still in flight."""
        if in_replay():
            return
        while (
            self.queue and self.error is None and any(_is_tainted(t) for t in tensors)
        ):
            item = self.queue.popleft()
            try:
                _execute(item, blocking=True)
            except BaseException as exc:
                self.error = exc
        if self.error is not None:
            error, self.error = self.error, None
            raise error


_RT = _Runtime()
_VERIFY: list = []  # (op name, placeholder, cpu snapshot) when TMB_VERIFY_FILL
# Pending-write counters keyed by STORAGE (the allocation holder shared by
# every view of a buffer): a storage is "tainted" while a queued op will
# still write it — either the producer of an unfilled placeholder or a
# queued value-mutation. Ops whose inputs are all untainted can run
# immediately (fresh output, no ordering hazard on a single thread/queue);
# ops touching tainted storages must defer behind the queued writers.
# Keying on the holder rather than the tensor makes views alias correctly.
_PENDING: dict[int, int] = {}


def _holder_key(t: object) -> int | None:
    holder = getattr(t, "_holder", None)
    return None if holder is None else id(holder)


def _is_tainted(t: object) -> bool:
    key = _holder_key(t)
    return key is not None and _PENDING.get(key, 0) > 0


def _taint(tensors: list) -> list[int]:
    keys = []
    for t in tensors:
        key = _holder_key(t)
        if key is not None:
            _PENDING[key] = _PENDING.get(key, 0) + 1
            keys.append(key)
    return keys


def _untaint(keys: list[int]) -> None:
    for key in keys:
        left = _PENDING.get(key, 0) - 1
        if left > 0:
            _PENDING[key] = left
        else:
            _PENDING.pop(key, None)


def _written_tensors(func, args, kwargs) -> list:
    """The tensors this op's schema declares it writes (in-place/out=)."""
    written = []
    schema_args = func._schema.arguments
    for i, arg in enumerate(schema_args):
        if arg.alias_info is None or not arg.alias_info.is_write:
            continue
        value = args[i] if i < len(args) else kwargs.get(arg.name)
        if isinstance(value, torch.Tensor):
            written.append(value)
        elif isinstance(value, (list, tuple)):
            written.extend(v for v in value if isinstance(v, torch.Tensor))
    return written


# Debug bisection: TMB_DEFER_ONLY="aten::mm,aten::add" limits which ops may
# defer (everything else drains + runs direct). Unset = defer everything.
import os as _os

_DEFER_ONLY = (
    frozenset(_os.environ["TMB_DEFER_ONLY"].split(","))
    if "TMB_DEFER_ONLY" in _os.environ
    else None
)

# Ops whose results the host is about to look at (or that cross devices):
# they drain the queue and run synchronously.
_SYNC_OPS = {
    "aten::_local_scalar_dense",
    "aten::equal",
    "aten::allclose",
    "aten::_to_copy",
    "aten::copy_",
    "aten::item",
    "aten::cpu",
    "aten::nonzero",
    "aten::masked_select",
}
# Pure-metadata ops: no kernel, execute immediately even mid-episode (their
# result is a view of storage that already exists).
_VIEW_OPS = {
    "aten::view",
    "aten::_unsafe_view",
    "aten::reshape",
    "aten::as_strided",
    "aten::slice.Tensor",
    "aten::select.int",
    "aten::transpose.int",
    "aten::permute",
    "aten::t",
    "aten::expand",
    "aten::squeeze",
    "aten::squeeze.dim",
    "aten::unsqueeze",
    "aten::detach",
    "aten::alias",
    "aten::split.Tensor",
    "aten::split_with_sizes",
    "aten::unbind.int",
}


def in_replay() -> bool:
    return getattr(_TLS, "replaying", False)


class _replay_scope:
    def __enter__(self) -> None:
        _TLS.replaying = True

    def __exit__(self, *exc: object) -> None:
        _TLS.replaying = False


# MAX's DeviceContext is not documented thread-safe: while an episode is
# active the launcher thread and the main thread both enqueue device work
# (kernels vs allocations/transfers), so serialize every device-touching
# call between them.
_DEVICE_LOCK = threading.RLock()


def _direct(func, args, kwargs):
    """The ordinary synchronous path (what dispatch did before this layer)."""
    with _DEVICE_LOCK:
        with _dispatch_scope():
            with torch._C._DisableTorchDispatch():
                return func(*args, **kwargs)


def _direct_blocking(func, args, kwargs):
    """Synchronous path that waits out in-flight compiles instead of
    propagating KernelPending to the caller."""
    while True:
        try:
            return _direct(func, args, kwargs)
        except KernelPending as pending:
            pending.job.wait()


def _run_after_deps(func, args, kwargs, flat_args):
    """Run an op immediately and exactly, first replaying just the FIFO
    prefix that produces its pending inputs (and written targets)."""
    deps = [a for a in flat_args if isinstance(a, torch.Tensor)]
    deps += _written_tensors(func, args, kwargs)
    _RT.drain_until_untainted(deps)
    return _direct_blocking(func, args, kwargs)


def _trace(msg: str) -> None:
    if _os.environ.get("TMB_TRACE"):
        import sys as _sys
        import time as _time

        print(f"[TRACE] t={_time.monotonic():.2f} {msg}", file=_sys.stderr, flush=True)


def _nan_count(t: object) -> str:
    if not isinstance(t, torch.Tensor) or not t.dtype.is_floating_point:
        return "-"
    with torch._C._DisableTorchDispatch():
        return str(int(torch.isnan(t.cpu()).sum()))


def _execute(item: tuple, blocking: bool = True) -> None:
    """Run one deferred op for real (main thread), then back-fill the
    placeholder outputs. Non-blocking mode propagates KernelPending so the
    pump can stop at a head-of-line item whose kernel is still compiling."""
    func, args, kwargs, placeholders, out_spec, taint_keys = item
    with _replay_scope():
        while True:
            try:
                real = _direct(func, args, kwargs)
                break
            except KernelPending as pending:
                if not blocking and not pending.job.done.is_set():
                    raise
                pending.job.wait()
    if _os.environ.get("TMB_TRACE"):
        flat_in, _ = tree_flatten((args, kwargs))
        ins = ",".join(_nan_count(a) for a in flat_in if isinstance(a, torch.Tensor))
        real_flat_t, _ = tree_flatten(real)
        outs = ",".join(
            _nan_count(a) for a in real_flat_t if isinstance(a, torch.Tensor)
        )
        _trace(f"replay {func._schema.name} in-nans=[{ins}] out-nans=[{outs}]")
    real_flat, _ = tree_flatten(real)
    ph_flat, _ = tree_flatten(tree_unflatten(placeholders, out_spec))
    for ph, value in zip(ph_flat, real_flat, strict=True):
        if isinstance(ph, torch.Tensor) and ph is not value:
            with _DEVICE_LOCK, torch._C._DisableTorchDispatch():
                torch.ops.aten.copy_(ph, value)
        if isinstance(ph, torch.Tensor):
            if _os.environ.get("TMB_VERIFY_FILL"):
                with _DEVICE_LOCK, torch._C._DisableTorchDispatch():
                    snap = value.cpu().float()
                    diff = (ph.cpu().float() - snap).abs().max().item()
                _VERIFY.append((func._schema.name, ph, snap))
                import sys as _sys

                extra = ""
                if func._schema.name == "aten::add" and len(args) >= 2:
                    with _DEVICE_LOCK, torch._C._DisableTorchDispatch():
                        cpu_ref = args[0].cpu().float() + args[1].cpu().float()
                    extra = f" cpu-ref-diff={(snap - cpu_ref).abs().max().item()}"
                if func._schema.name == "aten::linear":
                    with _DEVICE_LOCK, torch._C._DisableTorchDispatch():
                        bias = (
                            args[2].cpu().float()
                            if len(args) > 2 and args[2] is not None
                            else None
                        )
                        cpu_ref = torch.nn.functional.linear(
                            args[0].cpu().float(), args[1].cpu().float(), bias
                        )
                    extra = f" cpu-ref-diff={(snap - cpu_ref).abs().max().item()}"
                print(
                    f"[FILL] {func._schema.name} diff={diff}{extra}",
                    file=_sys.stderr,
                    flush=True,
                )
    _untaint(taint_keys)
    if _os.environ.get("TMB_SYNC_AFTER_REPLAY"):
        from . import torch_mojo_device_module as _dm

        _dm.synchronize()


def _meta_mirror(args, kwargs):
    """Map every mojo tensor arg to a meta tensor (identity-preserving)."""
    flat, spec = tree_flatten((args, kwargs))
    seen: dict[int, torch.Tensor] = {}
    out = []
    for leaf in flat:
        if isinstance(leaf, torch.Tensor):
            mirror = seen.get(id(leaf))
            if mirror is None:
                mirror = torch.empty_strided(
                    leaf.shape, leaf.stride(), dtype=leaf.dtype, device="meta"
                )
                seen[id(leaf)] = mirror
            out.append(mirror)
        else:
            out.append(leaf)
    meta_args, meta_kwargs = tree_unflatten(out, spec)
    return meta_args, meta_kwargs, seen


def _defer(func, args, kwargs):
    """Meta-infer outputs, allocate placeholders, queue the real execution.

    Value mutations (in-place/out= that keep their target's metadata) are
    deferrable: FIFO orders the write after every queued reader, and the
    taint on the written storage queues every later toucher behind it.
    Metadata-changing mutations (resize/reallocation) are not — their
    effects are observable synchronously — and raise _Undeferrable."""
    written = _written_tensors(func, args, kwargs)
    meta_args, meta_kwargs, mirrors = _meta_mirror(args, kwargs)
    meta_out = func(*meta_args, **meta_kwargs)

    for w in written:
        mirror = mirrors[id(w)]
        if tuple(mirror.shape) != tuple(w.shape) or mirror.stride() != w.stride():
            # The op changed its written argument's metadata (e.g. the
            # arange.start_out resize): that effect is observable NOW, so
            # this op cannot be deferred.
            raise _Undeferrable(func._schema.name)

    # Map aliasing outputs (in-place/out= schemas) back to the real inputs.
    storage_to_real = {}
    flat_in, _ = tree_flatten((args, kwargs))
    for leaf in flat_in:
        if isinstance(leaf, torch.Tensor):
            mirror = mirrors[id(leaf)]
            storage_to_real[mirror.untyped_storage()._cdata] = leaf

    device = None
    for leaf in flat_in:
        if isinstance(leaf, torch.Tensor):
            device = leaf.device
            break

    meta_flat, out_spec = tree_flatten(meta_out)
    placeholders = []
    for m in meta_flat:
        if not isinstance(m, torch.Tensor):
            placeholders.append(m)
            continue
        aliased = storage_to_real.get(m.untyped_storage()._cdata)
        if aliased is not None and (
            tuple(m.shape) == tuple(aliased.shape) and m.stride() == aliased.stride()
        ):
            placeholders.append(aliased)
            continue
        if aliased is not None:
            # An aliasing output with different metadata is a VIEW of a
            # pending tensor. Materializing it as a copy broke numerics in
            # ways not yet understood (deterministic 3.4-level divergence,
            # layout-independent) — so it stays undeferrable: the caller
            # waits for just the producing FIFO prefix (_run_after_deps).
            raise _Undeferrable(func._schema.name)
        with _DEVICE_LOCK, torch._C._DisableTorchDispatch():
            # Contiguous regardless of the meta stride: the backend's fast
            # paths produce contiguous outputs, and downstream kernel-tier
            # selection (hence accumulation order, hence bitwise results)
            # must match what a non-deferred run would see.
            ph = torch.empty(tuple(m.shape), dtype=m.dtype, device=device)
        placeholders.append(ph)

    # Pending writes: fresh placeholders AND the schema's written targets
    # (which may not appear among the outputs at all, e.g. foreach/fused
    # optimizer ops returning ()).
    taint_keys = _taint(
        [ph for ph in placeholders if isinstance(ph, torch.Tensor)] + written
    )
    _RT.submit((func, args, dict(kwargs), list(placeholders), out_spec, taint_keys))
    return tree_unflatten(placeholders, out_spec)


def dispatch(func, args, kwargs):
    """Entry point called from TorchMojoTensor.__torch_dispatch__."""
    name = func._schema.name
    overload = (
        f"{name}.{func._schema.overload_name}" if func._schema.overload_name else name
    )

    if _RT.error is not None and not in_replay():
        _RT.drain()  # re-raises the held replay error

    episode = _RT.active and not in_replay()
    if episode:
        _RT.pump()  # main thread advances the queue as it goes
        flat_args, _ = tree_flatten((args, kwargs))
        tainted = any(_is_tainted(a) for a in flat_args)
        mutates = any(
            arg.alias_info is not None and arg.alias_info.is_write
            for arg in func._schema.arguments
        )
        sync = name in _SYNC_OPS or overload in _SYNC_OPS
        if (
            sync
            and not _os.environ.get("TMB_CAST_SYNC")
            and name in ("aten::_to_copy", "aten::copy_")
        ):
            # Same-device copies/casts (autocast!) are ordinary data ops —
            # only actual device crossings behave as host-visible syncs.
            devices = {a.device.type for a in flat_args if isinstance(a, torch.Tensor)}
            target = kwargs.get("device")
            if target is not None:
                devices.add(torch.device(target).type)
            sync = len(devices) > 1
        if sync:
            # Host reads / device crossings: wait, but only for the FIFO
            # prefix this op actually depends on.
            if tainted:
                _trace(f"sync-drain {name}")
                _RT.drain_until_untainted(
                    [a for a in flat_args if isinstance(a, torch.Tensor)]
                )
            return _direct_blocking(func, args, kwargs)
        if not tainted and not mutates:
            # All inputs are real (or already filled): safe to run now —
            # single thread, single queue, fresh output. This covers views
            # of real tensors, factories, and any op off the pending chain.
            # A kernel miss defers rather than blocks: that is the point.
            _trace(f"untainted-direct {name}")
            try:
                return _direct(func, args, kwargs)
            except KernelPending:
                try:
                    return _defer(func, args, kwargs)
                except Exception:
                    return _run_after_deps(func, args, kwargs, flat_args)
        if _DEFER_ONLY is not None and name not in _DEFER_ONLY:
            _RT.drain()
            return _direct_blocking(func, args, kwargs)
        _trace(f"defer {name}")
        # Tainted functional ops AND mutations both queue: FIFO places a
        # mutation after every queued reader of its target, and tainting
        # the target makes every later toucher queue behind the mutation.
        try:
            return _defer(func, args, kwargs)
        except Exception:
            # No meta support, metadata-changing mutations (resize / out=
            # reallocation), alias-metadata effects, or other inference
            # issues: run this op exactly, waiting only for the FIFO prefix
            # it depends on — discovery continues past it.
            _trace(f"fallback-wait {name}")
            return _run_after_deps(func, args, kwargs, flat_args)

    try:
        return _direct(func, args, kwargs)
    except KernelPending as pending:
        if (
            in_replay()
            or _os.environ.get("TMB_NO_TRIGGER_DEFER")
            or (_DEFER_ONLY is not None and name not in _DEFER_ONLY)
        ):
            pending.job.wait()
            return _direct_blocking(func, args, kwargs)
        _RT.start_episode()
        try:
            return _defer(func, args, kwargs)
        except Exception:
            _RT.drain()
            return _direct_blocking(func, args, kwargs)


def drain() -> None:
    """Public: wait for all deferred work (used by device synchronize)."""
    _RT.drain()
