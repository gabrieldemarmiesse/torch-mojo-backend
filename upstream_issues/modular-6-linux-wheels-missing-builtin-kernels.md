# [MAX][packaging] Linux nightly 26.5.0.dev2026072306 cannot compile any graph: built-in kernel packages missing (`MAXG_addKernelPackage: failed to import kernels from ''`)

## Summary

With the 2026-07-23 Linux nightlies installed from
`https://whl.modular.com/nightly/simple/` (`max==26.5.0.dev2026072306` and
friends), **every** `max.engine` graph compilation fails — no custom
extensions involved:

```
error: failed to resolve built-in kernel package paths
ValueError: MAXG_addKernelPackage: failed to import kernels from ''
```

(the empty path in the second message is the unresolved built-in package
path; with custom extensions registered, the same failure reports the
custom package's path instead, which is misleading).

## Reproduction (stock venv, no user kernels)

```python
import numpy as np
from max import engine
from max.driver import CPU
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, ops

with Graph(
    "add", input_types=(TensorType(DType.float32, ["n"], device=DeviceRef.CPU()),)
) as g:
    (x,) = g.inputs
    g.output(ops.add(x, x))

session = engine.InferenceSession(devices=[CPU()])
model = session.load(g)  # ValueError: MAXG_addKernelPackage: ... from ''
```

Fails identically on a GPU-less host and on H100 nodes, CPU-only or
multi-device sessions.

## Why: the wheels dropped the built-in kernel packages

The previous nightly line shipped `max-mojo-mogg-libs` (a dependency of
`max-core`), whose wheel provided

```
modular/lib/mojo/builtin_kernels.mojoc
modular/lib/mojo/builtin_primitives.mojoc
```

`dev2026072306` removed `max-mojo-mogg-libs` from `max-core`'s
dependencies and publishes no wheel of it for that version — and no other
package in the resolved set ships those two files. The installed
`modular/lib/mojo/` contains the stdlib and library packages
(`std.mojoc`, `nn.mojoc`, `linalg.mojoc`, ...) but no
`builtin_kernels.mojoc` / `builtin_primitives.mojoc`, and
`libmax.so`'s built-in package resolution comes up empty.

Copying the two `.mojoc` files from the `dev2026061806`
`max-mojo-mogg-libs` wheel into `modular/lib/mojo/` does NOT fix it (the
resolution still reports an empty path — presumably a manifest, not a
directory scan), so there is no user-side workaround short of pinning the
whole toolchain back.

## Impact

- Everything eager works (`mojo build --emit shared-lib` extensions,
  drivers, collectives).
- Everything graph-mode is dead: for torch-mojo-backend that is the
  whole `torch.compile(backend=mojo_backend)` path — ~50 test failures
  in `tests/test_aten_functions.py` plus the compile test files, all
  with the `MAXG_addKernelPackage` error, on an otherwise green suite.
  Identical before and after our own changes; first seen when the pin
  moved from dev2026061806 (where graph mode worked) to dev2026072306.

## Environment

Linux x86_64 (Ubuntu 22.04 kernel 5.15), CPython 3.12, driver 570.211.01
(CUDA 12.8) and GPU-less hosts alike; `uv` -resolved environment from the
nightly index.
