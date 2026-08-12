# OpInfo conformance run — findings

First full sweep of PyTorch's `op_db` against the mojo device.
NVIDIA H100 PCIe, torch 2.11.0+cu130, `conformance/test_opinfo.py`.
**Nothing here has been fixed. This file reports.**

## Totals

2786 nodes collected, 2601 executed before the process died (see below).

| outcome | nodes |
|---|---|
| passed | 818 |
| skipped — operator not implemented, or OpInfo gave no samples for that dtype | 1560 |
| failed | 220 |
| xfail / error | 2 |

## 1. Segfault: any CUDA tensor moved to the mojo device

**This killed the run at 93%**, on `test_matches_cpu_to_mojo_bfloat16`.

```python
torch.randn(4, device="cuda").to("mojo")     # Fatal Python error: Segmentation fault
```

Isolated, one process per case:

| case | result |
|---|---|
| `torch.randn(4, device='cuda').to('mojo')` | **segfault** |
| `torch.randn(4, device='cuda', dtype=torch.float64).to('mojo')` | **segfault** |
| `torch.randn(4, dtype=torch.float64).to('mojo')` | fine |

So the trigger is the **source device**, not the dtype. The faulting frame is
`torch_mojo_tensor.py:_from_cpu`, reached from
`mojo_device_aten_ops.py:mojo_device__to_copy` — a CUDA source is routed into a
path whose name says it assumes host memory, so a device pointer is read as a
host pointer.

It reached us through OpInfo because the `to` samples deliberately include a
cross-device target tensor; no test in `tests/` does. A user with both backends
installed writing `x.cuda().to("mojo")` takes down the process with no
diagnostic.

## 2. We do not raise where PyTorch requires an error — 113 operators

`test_errors_match` feeds each operator the inputs OpInfo declares must raise,
and asserts the exception type PyTorch specifies. 113 operators accept at least
one malformed call instead of rejecting it, among them `cat`, `dot`, `bmm`,
`amax`, `amin`, `aminmax`, `diagonal`, `bucketize`, `diff`, `cov`,
`as_strided_scatter`, `diag_embed`, `dsplit`, `dstack`, `_chunk_cat`,
`bernoulli`, `__rsub__`.

Silently accepting a malformed call is worse than not implementing the
operator: the caller gets a wrong answer rather than an error.

## 3. Value mismatches vs CPU — 80 nodes across 33 operators

At OpInfo's own tolerances (the operator's `precisionOverride` where it
declares one, `assert_close` dtype defaults otherwise). Two clusters:

* **`fft_*` — 30 nodes, 10 operators** (`fft_fft2`, `fft_fftn`, `fft_hfft`,
  `fft_hfft2`, `fft_hfftn`, `fft_ifft2`, `fft_ifftn`, `fft_irfft`,
  `fft_irfft2`, `fft_irfftn`). These fail rather than skip, i.e. they do not
  raise `NotImplementedError` — they decompose and return something wrong.
* **the rest — 50 nodes**, including `bmm` at float32 and `tensor_split` on
  every dtype.

Also failing on every dtype: `cdouble`, `cfloat`, `chalf` — complex dtype
conversions, which produce a wrong answer instead of declining.

## 4. Not a backend bug: 27 nodes

`empty`, `empty_like`, `empty_permuted`, `empty_strided`, `new_empty`,
`new_empty_strided` return **uninitialised memory**, so comparing values
against a CPU run is meaningless. These are a limitation of this harness, not
findings; they need an allowlist that asserts shape/dtype/device only.

## Reproducing

```bash
uv run pytest conformance/                       # everything (~40 min)
uv run pytest conformance/ -k "test_errors_match"  # just the error-input conformance
uv run pytest conformance/ -k "bmm"                # one operator
```

The suite skips an unimplemented operator with its reason rather than failing
it, so the failure list stays the list of things that are *wrong* rather than
merely *absent*.
