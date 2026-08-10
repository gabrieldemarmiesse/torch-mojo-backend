# DDP `find_unused_parameters=True` requires a pinned-memory allocator that Python-only PrivateUse1 backends cannot register

## 🐛 Describe the bug

With parameters on a PrivateUse1 device, the DDP Reducer's unused-parameter
machinery copies `local_used_map_` to the device through a **pinned**
staging tensor (`torch/csrc/distributed/c10d/reducer.cpp:737-777` at
v2.11.0):

```cpp
if (local_used_map_dev_.is_cuda() || local_used_map_dev_.is_privateuseone()) {
  auto local_used_map_tmp = at::native::empty_like(
      local_used_map_, ..., /*pinned_memory=*/true);
  TORCH_INTERNAL_ASSERT(local_used_map_tmp.is_pinned());
  ...
```

Pinned allocation for PrivateUse1 goes through
`PrivateUse1HooksInterface::getPinnedMemoryAllocator()`, whose default
implementation throws
(`aten/src/ATen/detail/PrivateUse1HooksInterface.h:56-58`,
`FAIL_PRIVATEUSE1HOOKS_FUNC`), and the Python-backend registration path
(`_setup_privateuseone_for_python_backend` → `_DummyPrivateUse1Hook`,
`torch/utils/backend_registration.py`) provides no way to override it from
Python — the hook trampoline only forwards `is_available` /
`has_primary_context` / `is_built`.

Net effect: `DistributedDataParallel(model, find_unused_parameters=True)`
(and `static_graph=True`, which runs the same path on its first iteration)
throws for every Python-only PrivateUse1 backend, even when the backend has
a perfectly good pinned-memory story of its own (ours allocates pinned host
buffers through its device runtime for all H2D/D2H staging).

## Proposed fix (either)

1. Extend the Python hook trampoline so a backend can provide
   `getPinnedMemoryAllocator` (returning an allocator that calls back into
   the backend module, mirroring how the aten ops themselves are Python);
   or
2. make the Reducer fall back to a plain (non-pinned) staging copy when the
   backend's hooks don't provide a pinned allocator — the pinned copy is an
   optimization, not a correctness requirement, and
   `is_privateuseone()` backends without C++ hooks are exactly the case
   where it cannot exist today.

Option 2 is a small, contained change in
`Reducer::all_reduce_local_used_map`.

## Versions

torch 2.11.0. Hit by github.com/gabrieldemarmiesse/torch-mojo-backend
(Python-only PrivateUse1 backend running thread-per-rank DDP); we currently
document `find_unused_parameters=False` as a hard requirement.
