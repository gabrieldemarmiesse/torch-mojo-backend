# Errors don't propagate through `_create_work_from_future` Works: `Work.wait()` never rethrows, and `Future.set_exception` doesn't survive re-wrapping

## 🐛 Describe the bug

Custom Python ProcessGroups are told to build Work objects with
`torch._C._distributed_c10d._create_work_from_future` (this is what
`test/distributed/test_c10d_pypg.py` and
`torch/testing/_internal/distributed/multi_threaded_pg.py` do). For an
**async** PG that completes the future later from a worker thread, error
propagation has two holes at v2.11.0:

### 1. `FutureWrappingWork::wait()` swallows errors

`torch/csrc/distributed/c10d/Work.cpp:184-191`:

```cpp
bool wait(std::chrono::milliseconds timeout) override {
  TORCH_CHECK(timeout == kNoTimeout, ...);
  _fut->wait();          // ivalue::Future::wait — does NOT rethrow
  return true;
}
```

`ivalue::Future::wait()` only blocks until completion; the rethrowing
variant is `waitAndThrow()` (`aten/src/ATen/core/ivalue_inl.h:898-914`).
So after `fut.set_exception(exc)`, `work.wait()` returns `True` as if the
collective succeeded. Every other `Work` subclass (e.g. ProcessGroupGloo,
ProcessGroupNCCL) rethrows from `wait()` — code written against the
documented c10d contract ("`wait()` ... throws on error") silently loses
errors when the PG is future-backed. Fix: call `_fut->waitAndThrow()`.

### 2. `torch.futures.Future.set_exception` is wrapper-local

`torch/futures/__init__.py:255-283` implements `set_exception` as:

```python
super()._set_unwrap_func(raise_error)
self.set_result(result)          # the exception object becomes the VALUE
```

The unwrap hook lives on the Python `PythonFutureWrapper` instance only. The
underlying `ivalue::Future` completes **successfully** with the exception
object as its value. Any consumer that reaches the future through a
different wrapper never sees an error:

```python
fut = torch.futures.Future()
work = torch._C._distributed_c10d._create_work_from_future(fut)
fut.set_exception(ValueError("boom"))

fut.wait()                    # raises ValueError  (original wrapper)
work.wait()                   # returns True       (hole 1)
work.get_future().wait()      # returns the ValueError INSTANCE — no raise
```

In DDP's default comm-hook path the C++ side holds the ivalue future
(`bucket.future_work`) and does `wait()` + `value()` + tensor extraction
(`reducer.cpp:1728-1731`): the "error" surfaces only as a confusing
extraction failure on the exception object rather than the original
exception.

Fix direction: `set_exception` should complete the underlying ivalue future
with `setError` (an eptr built from the Python exception) instead of the
unwrap-func trick, so the error is visible through every wrapper and through
C++ (`Future::value()` / `waitAndThrow` already rethrow `eptr_`). If that's
too breaking, exposing an explicit `Future._set_error` for c10d use and
having `_create_work_from_future` docs point at it would already help.

## Why it matters

Async ProcessGroups cannot raise inside DDP's backward hooks (an exception
inside an autograd node is fatal on some backends and bad everywhere), so
the *only* clean error channel is the future — the "poisoned future"
pattern. Today that pattern half-works: original-wrapper waiters see the
error, `Work.wait()` callers and C++ consumers don't.

## Versions

torch 2.11.0. Hit by github.com/gabrieldemarmiesse/torch-mojo-backend
(async ProcessGroup for a PrivateUse1 backend; we currently document that
errors surface via the future's value path, not `Work.wait()`).
