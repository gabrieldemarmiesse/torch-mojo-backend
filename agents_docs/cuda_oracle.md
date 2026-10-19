# The live CUDA oracle

Tests that hold the mojo device to stock CUDA bit for bit run the same code on
torch's `cuda` device and on the mojo device of the same GPU, in one process,
and compare the results (`tests/native/cuda_oracle.py`). There are no recorded
CUDA outputs to refresh: the reference is whatever the installed CUDA torch
computes on the GPU at hand.

## Writing a test

```python
from tests.native.cuda_oracle import CudaOracle

def test_asinh_matches_cuda(cuda_oracle: CudaOracle):
    x = torch.randn(4096)  # CPU inputs: copied to each device with .to()
    cuda_oracle.assert_bitwise_equal(torch.asinh, x)
    cuda_oracle.assert_bitwise_equal(torch.add, x, x, kwargs={"alpha": 2})
    # build on the device yourself (strided views, factories, RNG):
    cuda_oracle.assert_bitwise_equal_by_device(
        lambda d: torch.rand(9, device=d)[::2], seed=1
    )
```

- `assert_bitwise_equal(fn, *args, kwargs=None, ...)` copies every tensor of
  `args`/`kwargs` to each device, calls `fn`, and compares the result **and the
  inputs after the call** (so in-place ops and `out=` are covered). `.to()`
  makes views dense: build strided operands inside `fn`.
- `assert_bitwise_equal_by_device(fn, ...)` calls `fn(device)` on both.
- `run_by_device(fn, seed=None)` returns `(ours, cuda)` for a custom
  comparison; `assert_bits_equal(ours, cuda, ...)` and `tensor_mismatch(...)`
  compare results computed by other routes (`torch.compile`, `_foreach_*`).
- Options (keyword-only, same on all of them):
  - `nan="exact"` (default: sign and payload must match) or `"any"`.
  - `signed_zero=True` (default: -0.0 != +0.0) or `False`.
  - `check_stride=False` (default) or `True`.
  - `check_message=False`: when CUDA raises, the mojo device must raise the
    same exception type; `True` also compares the first line, with device
    names folded to `DEV`.
  - `seed=`: `torch.manual_seed(seed)` before each side, and the device
    generator's state after the call must match too.
- Outputs can be tensors of any dtype (bool, ints incl. unsigned, fp16, bf16,
  float8, complex by parts) or scalars, nested in tuples/lists/dicts. A
  failure prints the mismatch count and the first indices with both bit
  patterns, both values, and each input of the same shape at that index.

The `cuda_oracle` fixture skips off NVIDIA and without a CUDA torch whose
driver initializes; on a CI runner with an NVIDIA GPU it fails instead, so the
parity tests cannot silently stop running. It is a GPU fixture (root
`conftest.py`), so its tests land in CI's mojo-device GPU job.

## What does not work with both devices in one process

Registering the mojo device as torch's accelerator breaks CUDA **autograd**
(see `ci.md`): call backward ATen ops directly on the `cuda` side, never
`.backward()`. Everything the converted tests use -- eager ops, `out=`,
in-place ops, the CUDA generator, error paths -- works. The two devices
allocate from separate pools on the one GPU, so a case needs its memory twice.

## Running the oracle tests

In CI nothing special is needed: the GPU jobs use the locked torch wheel
(cu130) on driver 580, and `uv run pytest tests/ -m "gpu and not cuda and not
cpu_torch"` includes them.

Locally, the locked cu130 wheel needs an r580+ driver. On a datacenter GPU with
an older driver (the Kyutai H100s run 570), CUDA's forward-compatibility
libraries make it work without root. Fetch the package that matches CI's
driver once:

```bash
mkdir -p ~/ddp_work/cuda_oracle && cd ~/ddp_work/cuda_oracle
curl -sO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-13-0_580.126.20-1ubuntu1_amd64.deb
dpkg-deb -x cuda-compat-13-0_580.126.20-1ubuntu1_amd64.deb compat_x
```

then run the tests with those libraries first on the library path (MAX picks
them up too, and the mojo device works unchanged):

```bash
srun -p kyutai_shell -c 10 --mem=30G --gres=gpu:1 bash -c '
  export LD_LIBRARY_PATH=$HOME/ddp_work/cuda_oracle/compat_x/usr/local/cuda-13.0/compat
  uv run --no-sync pytest tests/native/test_cuda_oracle.py tests/native/test_random_parity.py \
    tests/native/test_unary.py -k "cuda" tests/native/test_binary.py tests/native/test_float8.py'
```

Consumer GPUs have no forward compatibility: use a driver >= 580, or a venv
with a cu12 torch wheel of the same version (`uv pip install --no-deps
torch==<ver>+cu128 --index https://download.pytorch.org/whl/cu128`). The
oracle is then CUDA 12's libdevice, and the `asinh` cases skip because the
mojo port follows CUDA 13's.
