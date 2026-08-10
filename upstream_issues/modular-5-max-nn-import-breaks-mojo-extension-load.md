# [MAX][Python] `import max.nn` before dlopening a mojo-built CPython extension makes every later extension load segfault (nightly 26.5.0.dev2026072306)

## Summary

With the 2026-07-23 nightlies (`max==26.5.0.dev2026072306`,
`mojo-compiler==1.0.0b3.dev2026072306`), importing `max.nn` in a Python
process **before** any `mojo build --emit shared-lib` CPython extension has
been loaded puts the process in a state where every subsequent extension
load crashes with SIGSEGV inside the extension's `PyInit_*`. The reverse
order works, and loading any one mojo extension first "pins" the process so
that later extension loads survive an intervening `import max.nn`.

The previously pinned nightlies (`dev2026061806`, which still shipped
`max-mojo-mogg-libs` as a `max-core` dependency) did not have this
behavior. `dev2026072306` dropped `max-mojo-mogg-libs`.

## Reproduction

Any mojo Python extension reproduces it, e.g.:

```mojo
# hello_ext.mojo
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from std.os import abort


def ping() -> PythonObject:
    return PythonObject(42)


@export
def PyInit_hello_ext() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("hello_ext")
        m.def_function[ping]("ping")
        return m.finalize()
    except e:
        abort(t"failed: {e}")
```

```
mojo build hello_ext.mojo --emit shared-lib -o hello_ext.so
```

```python
import max.nn  # <-- the poison; max.graph / max.engine / max.driver are all fine

import importlib.machinery, importlib.util
loader = importlib.machinery.ExtensionFileLoader("hello_ext", "hello_ext.so")
spec = importlib.util.spec_from_file_location("hello_ext", "hello_ext.so", loader=loader)
m = importlib.util.module_from_spec(spec)
loader.exec_module(m)  # SIGSEGV
```

Swap the first two blocks (load the extension, then `import max.nn`) and
everything works — including *further* extension loads after the
`max.nn` import.

## Crash site

lldb backtrace (CPython 3.12, Linux x86_64; identical on a GPU-less host
and on an H100 node):

```
* thread #1, stop reason = signal SIGSEGV: address not mapped to object (fault address=0x158)
  * frame #0: python`PyModule_GetNameObject.cold + 59
    frame #1: python`PyModule_AddFunctions + 16
    frame #2: tensor_holder.so`PyInit_tensor_holder + 129
    frame #3: python`_PyImport_LoadDynamicModuleWithSpec + 383
```

I.e. inside `PythonModuleBuilder`, the module creation fails (returns
NULL) and the bindings pass the NULL module straight to
`PyModule_AddFunctions`, which dereferences it. Two bugs compound:
whatever `import max.nn` changes that makes `PyModule_Create` fail in a
later extension, and the missing NULL check in the bindings that turns a
failed module creation into a segfault instead of an `ImportError`.

## What we ruled out

- Not CUDA/driver related: reproduces identically on a host with no GPU.
- Not torch related: the two-line repro above has no torch import.
- Not fixed by `sys.setdlopenflags(RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND)`
  around the extension load.
- Not fixed by pre-loading the runtime libraries the extension links
  (`libKGENCompilerRTShared.so`, `libAsyncRTMojoBindings.so`, ...) with
  `ctypes.CDLL(..., RTLD_GLOBAL)` — the pin only works when a real
  extension's `PyInit` has run before `max.nn` is imported.
- `max.driver`, `max.dtype`, `max.graph`, `max.engine`,
  `max.experimental.functional`, `max.experimental.torch` are all
  harmless; `max.nn` (observed via `max.nn.kernels` and
  `max.nn.attention`) is the trigger.

## Impact on torch-mojo-backend

Our eager mode compiles kernels on demand as mojo CPython extensions, and
`aten_functions.py` imported `max.nn` at module scope — so with the new
nightly, the first eager kernel load of every fresh process segfaulted.

Workaround shipped on our side (see `preload_tensor_holder_if_cached` in
`eager_kernels/__init__.py`): load one mojo extension (`tensor_holder`)
at package-import time whenever its cached .so exists, and route the only
`max.nn` import in the package through a lazy accessor that forces
`tensor_holder` resident first. This holds unless user code imports
`max.nn` itself before `torch_mojo_backend` on a cold kernel cache.
