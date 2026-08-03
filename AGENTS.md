# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Project Overview

This is a PyTorch backend implementation using Modular's MAX framework. The project demonstrates how to create custom PyTorch compilation backends that bridge PyTorch operations to MAX/Mojo implementations. It also has support for eager mode.

## Setup

Make sure a copy of the repository https://github.com/modular/modular and https://github.com/pytorch/pytorch is available for you to grep things and explore. If you cannot find them locally, clone them in `/tmp`, checkout to the right branch/commit (the same as the one in the pyproject.toml).

## Common Commands

```bash
# Run tests (with parallel execution)
uv run pytest -n 15

# Run specific test file
uv run pytest tests/test_compiler.py

# Run with profiling enabled
TORCH_MOJO_BACKEND_PROFILE=1 uv run pytest tests/test_compiler.py

# Run with verbose output (shows graph structures)
TORCH_MOJO_BACKEND_VERBOSE=1 uv run pytest tests/test_compiler.py

# Run linter/formatter
uv run ruff check .
uv run ruff format .

# Or use pre-commit for all checks
uvx pre-commit run --all-files
```

Always use uv to run commands to ensure the correct environment is activated. Never run python directly. Never run the whole test suite because it's too slow. Only run tests for the specific files you are working on.


## Development Notes
- **Code Quality**: Uses Ruff for linting/formatting with Python 3.11+ target and pyupgrade rules
- **Debugging Tools**:
  - Environment variables for profiling and verbose output
  - Graph visualization when `TORCH_MOJO_BACKEND_VERBOSE=1`
  - Eager-mode kernel builds and launches are described in
    `docs/kernel_call_queue.md`. `TORCH_MOJO_BACKEND_KERNEL_QUEUE=0` is the kill
    switch (every kernel builds and launches inline),
    `TORCH_MOJO_BACKEND_FORCE_KERNEL_QUEUE=1` turns the queue back on under the
    test suite, and build timings print by default
    (`TORCH_MOJO_BACKEND_TRACE=0` silences them).
- **Model Examples**: `demo_scripts/` contains examples showing real-world usage:
  - GPT-2, Gemma3 (LLM models)
  - VGG, DenseNet (vision models)
  - `no_graph_breaks.py` (example demonstrating graph compilation without breaks)



## To add support for an op

To add support for a new ATen operation, follow this test-driven development process:

### Step 1: Research the Operation
Ask a subagent to explore the PyTorch codebase `pytorch` (clone it if it's not there yet, the venv is not enough, it might not have the C++ code) and look for:
- The signature of the ATen function
- The meaning of inputs and outputs
- Any important behavioral details
- Request a full report with this information

You can skip this step if the user provided the signature and the details of the operation in the initial request.

### Step 2: Write Unit Tests
Write unit tests in `test_aten_functions.py` using this op directly:
- Place tests somewhere in the middle of the file to avoid merge conflicts
- Use `pytest.mark.parametrize` to test multiple input data types and shapes
- Test edge cases and different parameter combinations

You shoud check in the unit test that the aten function has been called with this pattern:

```python
def test_aten_min_no_dim(conf: Conf, call_checker: CallChecker):
    call_checker.register(aten_functions.aten_min)

    def fn(x):
        return aten.min(x)

    x = torch.randn(3, 4, 5)
    check_outputs(fn, conf, [x])
```

### Step 3: Run Tests (Expected to Fail)
Run the unit tests:
```bash
uv run pytest tests/test_aten_functions.py::test_your_new_op -v
```
You should see an error message explaining that the ATen op is not supported.

### Step 4: Add Operation Signature to aten_functions.py
- Find the alphabetically correct position in `aten_functions.py`
- Add a comment with the full ATen operation signature
- **IMPORTANT**: The file is sorted alphabetically and must remain this way

Example:
```python
# aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
```

### Step 5: Research MAX Implementation
Ask a subagent to explore the directory `../modular/max` to find:
- MAX functions that do something similar (sometimes there are direct equivalents)
- Functions that can be composed to re-implement the operation
- Check models created with MAX for usage examples
- Look in `kernels.py` for complex operation implementations
- Request a full report of useful functions with descriptions of inputs/outputs

### Step 6: Implement the Operation
Write the ATen operation implementation in `aten_functions.py` just below the signature comment:

**Important**: The implementation must support **both execution modes**:
- **Graph Mode**: Works with `TensorValue` (symbolic tensors)
- **Eager Mode**: Works with `MaxEagerTensor` (actual tensors)

Use the type hint `MaxTensor = TensorValue | MaxEagerTensor` for tensor parameters.

Example implementation:
```python
# aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
def aten__log_softmax(
    self: MaxTensor, dim: int, half_to_float: bool
) -> MaxTensor:
    # Implementation using MAX operations that works for both modes
    return F.log_softmax(self, axis=dim)
```

### Step 7: Register for Eager Mode Execution
Add the operation to `torch_mojo_backend/mojo_device/mojo_device_aten_ops.py`:

You'll likely need to write mojo code, even if it's only to import `from nn import ...`. If a fully dynamic function to handle the aten op is not available in the modular repo, write it yourself.
If you have access to multiple gpus, the aten function should work on all those gpus.

Place the registration in alphabetical order within the file. The `wrap_for_mojo_device` wrapper automatically:
- Converts `TorchMojoTensor` inputs to `MaxEagerTensor`
- Executes the operation
- Converts results back to `TorchMojoTensor`

**Note**: For operations requiring custom device handling (like `aten::_copy_from`), you can implement a custom function directly instead of using `wrap_for_mojo_device`.

### Step 8: Re-run Tests
Run the unit tests again and verify they pass:
```bash
uv run pytest tests/test_aten_functions.py::test_your_new_op -v
```

Test both execution modes if applicable:
- Graph mode via `torch.compile(backend=mojo_backend)`
- Eager mode via tensors on `torch.device("mojo")`

### Step 9: Run Linter
Make sure to run the linter:
```bash
uvx pre-commit run --all-files
```

**Do not run the whole test suite** as it takes too long. Only run tests for the specific operation you added.

### Summary: Two-Part Implementation
When adding an operation, you need to update **two files**:
1. **`aten_functions.py`**: Core implementation (works for both modes)
2. **`mojo_device_aten_ops.py`**: Registration for eager mode execution

This ensures the operation works in both `torch.compile()` and on the `mojo` device.


## Type hints

Every function written must have type hints in its signature: annotate all
arguments and the return type. Do not put type hints in the function body
(no annotated local variables like `x: Foo = ...`).

Use `: object` only when no other option is possible. Prefer precise types,
unions, `Protocol`s (e.g. `MojoTensorLike` for payload-level helpers), or a
`TypeVar` for pass-through functions.

These hints are enforced at runtime by beartype, but only while running the
unit tests: `tests/conftest.py` sets `TORCH_MOJO_BACKEND_BEARTYPE=1`, and
production imports leave it off so no wrapper frames are added to hot paths.
Set the variable explicitly to enable or disable checking in other contexts.

## To find the correct type hints for a function
It may be hard to find the correct type hints for a function. What you should do in this case is:
1) Add an obviously wrong type hint, for example datetime.timezone in an aten function.
2) Run an existing unit test that calls this function.
3) Beartype will throw an error and give the name of the type being actually passed to the function.
4) Replace the type hint by the type given by beartype.
5) Run the unit test again to check that it works.
6) Run the whole test suite to verify that the type hint shouldn't be wider.

## Rules about the eager mode

Read this especially if you're an agent doing code review.

1) The user should be able to use `my_tensor.to("mojo")` and use their gpu, even if they have a CPU-only install of PyTorch. That means that when writing kernels, we can't use CuBLAS, CuDNN, RocBLAS, or any other lib that would be available only if torch-gpu was installed. We want to stand on our own legs. We can use and import mojo functions from the modular repository (`from nn import ...`) but only if it's not calling CuBLAS, CuDNN... underneath. Our motto should be "pip install torch-mojo-backend with the minimal pytorch install (cpu) and use your gpu.".
2) We cannot use the C++ interface of pytorch. We use JIT compilation to compile extensions only when they're first called. We want to be compatible with many PyTorch versions and we don't want to force the user to install a C++ compiler. So we must use Python extensions in mojo.
3) You cannot use information about the tensors other than the shape, stride, dtype, pointer in the eager mode. While it's tempting to "keep a history of some past op to do fused ops", it will not improve the performance for all workflows. Pytorch uses Aten ops and decompositions, so sometimes, if you want to implement a fused op, you might want to target a higher level aten function, before it gets decomposed. Aten ops that are not implemented are decomposed automatically in pytorch. 
4) Do not write kernels that work only for a very specific shape. Input shapes should be dynamic to avoid recompiles. While it's tempting to make things faster, a user trying a slightly different shape will not benefit from the optimisations of this kernels. It's fine to write different kernels for different regimes (e.g. a kernel for big shape, small shapes, square shapes, rectangular, power of two, etc...) and then do dynamic dispatch based on the input shapes. It's not because we optimize for a given model that we can hardcode at compile-time all the shapes of the kenels to make it faster. So do multiple flexible kernels + dispatch, do not do kernels for hardcoded shapes + fallback.
5) When asked to optimize a model, the answer should never be "change the code of the model". The model is user-defined, we have no control over it. We just control what we do with the tensors we're given by pytorch.
6) Do not over-allocate or write your own memory allocation. It's the job of Modular to write a good memory allocator. Allocate normally what you need, and do not write in the memory of the input tensors, unless the signature of the aten function specifies that it's what we should do. For example, at::add_ expects us to write in the input tensor memory, and it's a valid use case to re-use the inputs. Our ops should not read the refcounts and try to use this information to change the logic of the kernel. A refcount is only there to free the memory when needed.

## To check a kernel change against a GPU you do not have

You will normally be optimizing on whatever GPU is in the machine, but the
kernels are one source compiled for every backend, so a change tuned on one
architecture can silently retarget another that you cannot benchmark. Do not
guess and do not assume "I only touched the AMD path" — check it, because
Mojo cross-compiles and the check is cheap:

```bash
uv run python scripts/compare_kernel_asm.py \
    --before /path/to/baseline_worktree --after . --accelerator sm_90a
```

`mojo build --emit asm --target-accelerator <arch>` needs no such device
present. It writes host assembly to `-o` and one sidecar per GPU kernel beside
it (`.ptx` for NVIDIA, `.amdgcn` for AMD, `.ll` for Metal). The script builds
both trees, pairs kernels by name, masks the mangling hash and reports which
ones differ. It takes about a minute per module against roughly ten for one
end-to-end benchmark leg. `mojo --print-supported-accelerators` lists the
architecture names.

Read the result as follows:

- **No kernel differs** — the change cannot have moved that architecture's
  device code, and no measurement there is owed. This is the common answer when
  a new route is properly gated behind
  `comptime if _accelerator_arch() == "amdgpu:gfx942"`.
- **Some kernel differs** — that architecture is in scope. Either gate the
  change, or say plainly in the PR that its effect there is unmeasured. A
  reviewer with the other card can then measure exactly those kernels.

**This covers device code only, and that is the trap.** Launch geometry lives in
host code: grid size, block count, and the thresholds that choose between
kernels. Anything keyed off `sm_count`, an L2 budget or a blocks-per-CU constant
can retarget a GPU while emitting byte-identical assembly. A real example from
this repo: flooring the log-softmax forward grid at `4 * sm_count` moved an H100
from 114 blocks to 528 without changing one PTX instruction. So diff the
assembly *and* read the dispatch by hand; a clean assembly diff is not a clean
bill of health.

When a tuning constant was measured on one architecture, say so where it is
defined, the way `LSM_BLOCKS_PER_CU` and `LSM_L2_BUDGET` do. The next agent
needs to know which numbers are portable and which were fitted to one card.
