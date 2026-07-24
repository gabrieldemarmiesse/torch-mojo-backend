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

# Debug: force every data op through the defer machinery even when its
# kernel is warm, and never pump — the queue only empties at sync points.
# Reproduces the maximal run-ahead regime without a cold cache. Read
# dynamically so a harness can toggle it between passes.
def _force_defer() -> bool:
    return bool(_os.environ.get("TMB_FORCE_DEFER"))


def _is_view_op(func) -> bool:
    """Functional op whose outputs alias an input (the view family)."""
    return any(
        r.alias_info is not None and not r.alias_info.is_write
        for r in func._schema.returns
    )


def _packed_bhtd_strides(shape: tuple) -> tuple | None:
    if len(shape) != 4:
        return None
    b, h, t, d = shape
    return (h * t * d, d, h * d, 1)


# Ops whose BACKEND output layout deviates from the meta impl's: the fa4
# attention kernels emit the packed (B,T,H,D)-memory layout for their
# 4-dim outputs (so the model's transpose(1,2)+view merge is free), while
# upstream meta says contiguous. Placeholder layouts must match the
# backend or downstream graph structure changes (a .contiguous() clone
# appears, autograd layout contracts copy, the fa4 backward misreads).
# Any new deviation shows up as a fill-time "stride-mismatch" trace.
_LAYOUT_OVERRIDES = {
    "aten::_scaled_dot_product_flash_attention": _packed_bhtd_strides,
    "aten::_scaled_dot_product_flash_attention_backward": _packed_bhtd_strides,
}


# Debug: harness-settable hook called with (schema_name, tensors) at every
# placeholder fill during replay. None in production.
_FILL_HOOK = None

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


# The thread whose device work was enqueued last. MAX orders device work
# only WITHIN an enqueuing thread: work submitted from a second thread is
# unordered with respect to the first's (verified: bit-stable stale reads).
# A full device synchronize at each enqueuing-thread switch restores a
# total order — switches are rare (main <-> autograd engine, per backward).
_DEVICE_THREAD: list = [None]


def _order_device_thread() -> None:
    me = threading.current_thread()
    if _DEVICE_THREAD[0] is not None and _DEVICE_THREAD[0] is not me:
        from . import torch_mojo_device_module as _dm

        _trace("thread-switch device synchronize")
        _dm.synchronize()
    _DEVICE_THREAD[0] = me


def _direct(func, args, kwargs):
    """The ordinary synchronous path (what dispatch did before this layer)."""
    with _DEVICE_LOCK:
        _order_device_thread()
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

        thread = (
            "main"
            if threading.current_thread() is threading.main_thread()
            else threading.current_thread().name
        )
        print(
            f"[TRACE] t={_time.monotonic():.2f} [{thread}] {msg}",
            file=_sys.stderr,
            flush=True,
        )


def _nan_count(t: object) -> str:
    if not isinstance(t, torch.Tensor) or not t.dtype.is_floating_point:
        return "-"
    with torch._C._DisableTorchDispatch():
        return str(int(torch.isnan(t.cpu()).sum()))


def _execute(item: tuple, blocking: bool = True) -> None:
    """Run one deferred op for real (main thread), then back-fill the
    placeholder outputs. Non-blocking mode propagates KernelPending so the
    pump can stop at a head-of-line item whose kernel is still compiling."""
    func, args, kwargs, placeholders, out_spec, taint_keys, written = item
    depth = getattr(_TLS, "exec_depth", 0)
    _TLS.exec_depth = depth + 1
    _trace(f"exec-enter d={depth} {func._schema.name}")
    try:
        # no_grad: autograd bookkeeping for this op already happened at
        # dispatch time (above the python key); the replay re-runs only the
        # kernel, possibly outside the caller's original grad context (an
        # optimizer's no_grad step replayed at a later drain would
        # otherwise trip the in-place-on-leaf check). Version counters of
        # the written targets are preserved for the same reason: versions
        # are metadata, and metadata follows DISPATCH order — _defer bumped
        # them when the mutation was queued, so a bump here (mid-drain,
        # possibly after a later op already saved the tensor for backward)
        # would trip the saved-variable version check.
        with _replay_scope(), torch.no_grad():
            with torch.autograd._unsafe_preserve_version_counter(tuple(written)):
                while True:
                    try:
                        real = _direct(func, args, kwargs)
                        break
                    except KernelPending as pending:
                        if not blocking and not pending.job.done.is_set():
                            raise
                        pending.job.wait()
    finally:
        _TLS.exec_depth = depth
    # NOTE: no device reads (.cpu()/.item()) in replay instrumentation —
    # a device read here re-enters dispatch as a sync op and recursively
    # drains the rest of the queue BEFORE this item's fills below have
    # run (proven: reversed replay prints + garbage reads, trace-only).
    _trace(f"replay d={depth} {func._schema.name}")
    real_flat, _ = tree_flatten(real)
    ph_flat, _ = tree_flatten(tree_unflatten(placeholders, out_spec))
    for ph, value in zip(ph_flat, real_flat, strict=True):
        if isinstance(ph, torch.Tensor) and ph is not value:
            if (
                getattr(ph, "_holder", None) is not None
                and ph._holder is getattr(value, "_holder", None)
                and ph._offset == value._offset
                and ph._strides == value._strides
            ):
                continue  # alias output: same bytes, nothing to fill
            if ph.stride() != value.stride():
                # The backend impl disagrees with the meta impl about the
                # output layout: downstream consumers already saw the meta
                # layout, a steady run would have shown them the backend's.
                _trace(
                    f"stride-mismatch {func._schema.name}: "
                    f"ph{tuple(ph.stride())} real{tuple(value.stride())} "
                    f"shape{tuple(ph.shape)}"
                )
            with _DEVICE_LOCK, torch._C._DisableTorchDispatch():
                _order_device_thread()
                # A fill is not a semantic mutation — it completes the op
                # that created `ph`. Autograd may already track views of the
                # placeholder (a deferred split/view of a pending tensor),
                # and a version bump here would trip its multi-view
                # modified-inplace check at backward-graph construction.
                with torch.autograd._unsafe_preserve_version_counter(ph):
                    torch.ops.aten.copy_(ph, value)
    _trace(f"filled d={depth} {func._schema.name}")
    _untaint(taint_keys)
    # Hook AFTER untaint: a device read of the filled tensor from the hook
    # re-enters dispatch as a sync op, and while still tainted that would
    # recursively drain the rest of the queue mid-fill (the heisen-bug).
    if _FILL_HOOK is not None:
        _FILL_HOOK(
            func._schema.name,
            [p for p in ph_flat if isinstance(p, torch.Tensor)],
            (args, kwargs),
        )
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
    fresh = []
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
            # pending tensor. Do NOT synthesize the alias from meta
            # metadata: the backend's view impls sometimes materialize (and
            # always own the output layout), and downstream kernel-tier
            # selection must see exactly the layouts a steady run produces
            # (meta-synthesized aliases measurably shift numerics).
            # dispatch() routes view ops to _defer_view instead; reaching
            # this branch means that path failed, so wait for the producer.
            raise _Undeferrable(func._schema.name)
        override = _LAYOUT_OVERRIDES.get(func._schema.name)
        strides = (override(tuple(m.shape)) if override else None) or tuple(m.stride())
        with _DEVICE_LOCK, torch._C._DisableTorchDispatch():
            # META strides (with per-op backend overrides), not
            # forced-contiguous: downstream consumers (and autograd's layout
            # contract) react to the output layout NOW, so it must be
            # exactly what a steady run would see — a wrong layout inserts
            # clones/copies and changes the backward graph structurally.
            # Fill-time verifies the backend agreed (stride-mismatch trace).
            ph = torch.empty_strided(
                tuple(m.shape), strides, dtype=m.dtype, device=device
            )
        placeholders.append(ph)
        fresh.append(ph)

    if not fresh and not written:
        # Every output is an alias of existing storage and nothing is
        # mutated: the op is fully realized as metadata, nothing to replay.
        return tree_unflatten(placeholders, out_spec)

    # Pending writes: fresh placeholders AND the schema's written targets
    # (which may not appear among the outputs at all, e.g. foreach/fused
    # optimizer ops returning ()).
    taint_keys = _taint(fresh + written)
    _RT.submit(
        (func, args, dict(kwargs), list(placeholders), out_spec, taint_keys, written)
    )
    # The mutation's semantic effect on version counters belongs to NOW
    # (dispatch order), not to the replay: later ops may save the target
    # for backward, and their saved-version snapshot must postdate this
    # write. The replay suppresses its own bump (_execute).
    for w in written:
        torch._C._increment_version(w)
    return tree_unflatten(placeholders, out_spec)


def _defer_view(func, args, kwargs):
    """Deferred execution for view-family ops on pending tensors.

    Run the BACKEND impl immediately: pure-view impls compute metadata only
    (shape/stride/ptr — pending bytes are irrelevant), and the result is a
    true alias whose layout is exactly what a steady run hands downstream —
    the backend, not this layer, owns every materialize-vs-view and layout
    decision, which kernel-tier selection (hence bitwise results) depends
    on. If the backend chose to MATERIALIZE (an output on fresh storage: it
    read bytes, garbage while the input is pending), keep that output as a
    placeholder and queue a re-execution to refill it once the producer has
    replayed; true-alias outputs in the same item are skipped at fill time
    (same holder/offset/strides)."""
    result = _direct(func, args, kwargs)
    flat_in, _ = tree_flatten((args, kwargs))
    in_keys = {_holder_key(a) for a in flat_in if isinstance(a, torch.Tensor)}
    out_flat, out_spec = tree_flatten(result)
    materialized = [
        t
        for t in out_flat
        if isinstance(t, torch.Tensor)
        and _holder_key(t) is not None
        and _holder_key(t) not in in_keys
    ]
    if materialized:
        taint_keys = _taint(materialized)
        _RT.submit(
            (func, args, dict(kwargs), list(out_flat), out_spec, taint_keys, [])
        )
    return result


def dispatch(func, args, kwargs):
    """Entry point called from TorchMojoTensor.__torch_dispatch__."""
    name = func._schema.name
    overload = (
        f"{name}.{func._schema.overload_name}" if func._schema.overload_name else name
    )

    if _RT.error is not None and not in_replay():
        _RT.drain()  # re-raises the held replay error

    force = _force_defer()
    episode = (_RT.active or force) and not in_replay()
    if episode:
        if not force:
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
        if not tainted and not mutates and not force:
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
        if not mutates and _is_view_op(func):
            _trace(f"defer-view {name}")
            try:
                return _defer_view(func, args, kwargs)
            except KernelPending:
                pass  # cold materializing impl: fall through to generic defer
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


def wait_for(tensors: list) -> None:
    """Public: wait until none of `tensors` has a pending deferred write.

    For device code that reads tensor payloads OUTSIDE __torch_dispatch__
    (e.g. the AutogradPrivateUse1 sdpa impl): such readers bypass the
    deferred layer entirely, so they must wait out queued producers before
    touching bytes. Views of pending tensors are only safe to hand out
    because every in-dispatch consumer replays the producing FIFO prefix
    first — this is the same barrier for out-of-dispatch consumers."""
    live = [t for t in tensors if isinstance(t, torch.Tensor)]
    if in_replay() or not any(_is_tainted(t) for t in live):
        return
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError(
            "deferred-compile: payload read from a non-main thread while a "
            "producer is still queued; replays are main-thread-only"
        )
    _trace("wait-for out-of-dispatch reader")
    _RT.drain_until_untainted(live)
