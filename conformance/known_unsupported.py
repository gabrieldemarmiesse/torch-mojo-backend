"""Conformance cases this backend is known not to pass -- declared, never inferred.

`test_opinfo.py` asks OpInfo what an operator must do and compares the mojo
device against the CPU.  A case we cannot run AT ALL still has to be told
apart from a case we run WRONGLY, and this file is where that distinction is
declared: one entry per (test, operator, dtype) node, i.e. per pytest node the
suite instantiates.

Why a declaration and not the exception the call raised.  An exception cannot
say why it was raised.  The harness used to recognise "we simply have not
implemented this" from the exception itself -- `isinstance(exc,
NotImplementedError)`, or its message containing "not implemented" / "no fast
implementation" -- and skipped the node.  That let any kernel raising
NotImplementedError for a BUG reason (a mis-gated dtype, a dispatch typo, an
unreachable branch) turn its own failure into a skip, erasing exactly the
distinction the suite exists to draw.  An implementation must not be able to
excuse itself.  Only this file excuses a case, and only a case someone listed
on purpose.

Semantics, implemented in `test_opinfo.py` and pinned by
`test_known_unsupported.py`:

* A declared node is still RUN, and it must still fail: it reports as xfail.
  When it starts passing, the suite FAILS and names the entry to delete, so
  the list cannot rot into a permanent mute.  Adding support for an operator
  is not finished until its entries here are gone.
* An undeclared node that raises NotImplementedError FAILS, like any other
  wrong answer.

An entry is a NODE, not an operator: one node runs every sample input OpInfo
generates for that dtype, so an operator whose common cases work but which
declines one exotic sample -- an empty tensor, a bool operand, a
zero-dimensional operand -- is listed here too.  The entry says "not all of
this node can pass", never "none of this operator works", and deleting it
requires the whole node to pass.

The two tokens of an entry are the two tokens of the node id, so a failing
node names its own entry:

    test_matches_cpu_addbmm_mojo_float16
                     ^^^^^^         ^^^^^^^
                     operator       dtype
    -> _MATCHES_CPU["addbmm"] contains "float16"

Support differs between accelerators -- a Mojo kernel gated on one
architecture, a dtype one card does not have -- so the tables cannot be
global and must not be copied per GPU either.  `_MATCHES_CPU` and
`_ERRORS_MATCH` are the base tables, measured on `BASE_ACCELERATOR`; every
other accelerator records only its DELTA against them in
`_ACCELERATOR_DELTAS`, keyed by the architecture name of the accelerator the
mojo device runs kernels on ("sm_90a", "amdgpu:gfx942", ...) -- the same
string the eager kernels themselves gate architecture-specific routes on.

Regenerating either the base tables or a delta:

    uv run python conformance/regenerate_known_unsupported.py --help
"""

from __future__ import annotations

import functools

import torch

from torch_mojo_backend import get_accelerators

# The accelerator the base tables below were measured on.  Another
# accelerator is expressed as a delta against them, not as a second copy.
BASE_ACCELERATOR = "sm_90a"

# --- BEGIN GENERATED test_matches_cpu (regenerate_known_unsupported.py) ---
_MATCHES_CPU: dict[str, tuple[str, ...]] = {
    "__getitem__": ("float32", "bfloat16", "float16", "int64", "bool", "complex64"),
    "__radd__": ("complex64",),
    "__rdiv__": ("complex64",),
    "__rmatmul__": ("complex64",),
    "__rmul__": ("complex64",),
    "__rpow__": ("complex64",),
    "__rsub__": ("complex64",),
    "_batch_norm_with_update": ("bfloat16", "float16"),
    # Matches CUDA, not CPU: SoftMax.cu rounds `grad * output` to the half
    # dtype before its float reduction; CPU's kernel keeps it in float.
    "_softmax_backward_data": ("bfloat16", "float16"),
    "_unsafe_masked_index": (
        "float32",
        "bfloat16",
        "float16",
        "int64",
        "bool",
        "complex64",
    ),
    "_unsafe_masked_index_put_accumulate": (
        "float32",
        "bfloat16",
        "float16",
        "int64",
        "bool",
        "complex64",
    ),
    "abs": ("complex64",),
    "acos": ("complex64",),
    "acosh": ("complex64",),
    "add": ("complex64",),
    "addbmm": ("complex64",),
    "addcdiv": ("complex64",),
    "addcmul": ("complex64",),
    "addmm": ("complex64",),
    "addmm_decomposed": ("complex64",),
    "addmv": ("complex64",),
    "addr": ("complex64",),
    "allclose": ("complex64",),
    "amax": ("bool",),
    "amin": ("bool",),
    "angle": ("complex64",),
    "argsort": ("bfloat16", "float16", "bool"),
    "asin": ("complex64",),
    "asinh": ("complex64",),
    "atan": ("complex64",),
    "atanh": ("complex64",),
    "baddbmm": ("complex64",),
    "bmm": ("complex64",),
    "cat": ("float32", "bfloat16", "float16", "int64", "bool"),
    "chalf": ("float32", "bfloat16", "float16", "int64", "bool", "complex64"),
    "cholesky": ("complex64",),
    "cholesky_inverse": ("complex64",),
    "cholesky_solve": ("complex64",),
    "complex": ("float32", "float16"),
    "corrcoef": ("complex64",),
    "cos": ("complex64",),
    "cosh": ("complex64",),
    "cov": ("complex64",),
    "cross": ("complex64",),
    "cumprod": ("complex64",),
    "cumsum": ("bfloat16", "float16", "complex64"),
    # Matches CUDA, not CPU: CUDA's half cumsum rounds every addition (scan_dim<scalar_t>); bit-equal to stock CUDA on every sample.
    "cumulative_trapezoid": ("float16", "complex64"),
    "diff": ("complex64",),
    "dist": ("complex64",),
    "div_no_rounding_mode": ("complex64",),
    "dot": ("complex64",),
    "einsum": ("complex64",),
    "exp": ("complex64",),
    "exp2": ("complex64",),
    "expm1": ("complex64",),
    # float32 FFT rounding on a near-zero output of the 3-D sample: 1.4e-7 of
    # the largest output from float64 (cuFFT 8.6e-8, CPU 8.9e-8); stock CUDA
    # fails the complex64 node as well.
    "fft_hfft2": ("int64", "complex64"),
    "float_power": ("complex64",),
    "geqrf": ("complex64",),
    "gradient": ("complex64",),
    "histc": ("bfloat16", "float16"),
    "index_fill": ("complex64",),
    "index_put": ("float32", "bfloat16", "float16", "int64", "bool"),
    "inner": ("complex64",),
    "isclose": ("complex64",),
    "isfinite": ("complex64",),
    "isinf": ("complex64",),
    "isnan": ("complex64",),
    "istft": ("complex64",),
    "kron": ("complex64",),
    "kthvalue": ("bfloat16", "int64"),
    "ldexp": ("complex64",),
    "lerp": ("complex64",),
    "linalg_cholesky": ("complex64",),
    "linalg_cholesky_ex": ("complex64",),
    "linalg_cond": ("complex64",),
    "linalg_cross": ("complex64",),
    "linalg_det": ("complex64",),
    "linalg_eig": ("float32", "complex64"),
    "linalg_eigh": ("complex64",),
    "linalg_eigvals": ("float32", "complex64"),
    "linalg_eigvalsh": ("complex64",),
    "linalg_householder_product": ("complex64",),
    "linalg_inv": ("complex64",),
    "linalg_inv_ex": ("complex64",),
    "linalg_ldl_factor": ("complex64",),
    "linalg_ldl_factor_ex": ("complex64",),
    "linalg_ldl_solve": ("complex64",),
    "linalg_lstsq": ("float32", "complex64"),
    "linalg_lstsq_grad_oriented": ("float32", "complex64"),
    "linalg_lu_solve": ("complex64",),
    "linalg_matrix_norm": ("complex64",),
    "linalg_matrix_power": ("complex64",),
    "linalg_matrix_rank": ("complex64",),
    "linalg_matrix_rank_hermitian": ("complex64",),
    "linalg_multi_dot": ("complex64",),
    "linalg_norm": ("complex64",),
    "linalg_norm_subgradients_at_zero": ("complex64",),
    "linalg_pinv": ("complex64",),
    "linalg_pinv_hermitian": ("complex64",),
    "linalg_qr": ("complex64",),
    "linalg_slogdet": ("complex64",),
    "linalg_solve": ("complex64",),
    "linalg_solve_ex": ("complex64",),
    "linalg_solve_triangular": ("complex64",),
    "linalg_svd": ("complex64",),
    "linalg_svdvals": ("complex64",),
    "linalg_tensorinv": ("complex64",),
    "linalg_tensorsolve": ("float32", "complex64"),
    "linalg_vander": ("complex64",),
    "linalg_vecdot": ("complex64",),
    "linalg_vector_norm": ("complex64",),
    "log": ("complex64",),
    "log10": ("complex64",),
    "log1p": ("complex64",),
    "log2": ("complex64",),
    "logaddexp": ("complex64",),
    "logcumsumexp": ("float16", "complex64"),
    "logdet": ("complex64",),
    "logsumexp": ("complex64",),
    "lu": ("complex64",),
    "lu_solve": ("complex64",),
    "lu_unpack": ("complex64",),
    "masked_cumprod": ("complex64",),
    "masked_cumsum": ("bfloat16", "float16", "complex64"),
    "masked_logsumexp": ("complex64",),
    "masked_mean": ("complex64",),
    "masked_normalize": ("complex64",),
    "masked_prod": ("complex64",),
    "masked_scatter": ("complex64",),
    "masked_std": ("complex64",),
    "masked_sum": ("complex64",),
    "masked_var": ("complex64",),
    "matmul": ("complex64",),
    "matrix_exp": ("bfloat16", "float16", "complex64"),
    "mean": ("complex64",),
    "mm": ("complex64",),
    "mode": ("bfloat16", "int64", "bool"),
    "mul": ("complex64",),
    "mv": ("complex64",),
    "nanmean": ("float32", "bfloat16", "float16"),
    "nanquantile": ("float32",),
    # CUDA's dropout_backward dispatches on floating grads only ("masked_scale"
    # not implemented for 'Long'); CPU computes grad * mask * scale for any.
    "native_dropout_backward": ("int64", "bool"),
    "native_layer_norm": ("float32", "bfloat16", "float16"),
    "neg": ("complex64",),
    "nn_functional_alpha_dropout": ("float32", "bfloat16", "float16"),
    "nn_functional_avg_pool1d": ("int64",),
    "nn_functional_avg_pool2d": ("int64",),
    "nn_functional_avg_pool3d": ("int64",),
    "nn_functional_channel_shuffle": ("complex64",),
    "nn_functional_conv1d": ("int64", "complex64"),
    "nn_functional_conv2d": ("int64", "complex64"),
    "nn_functional_conv_transpose1d": ("int64", "complex64"),
    "nn_functional_conv_transpose2d": ("int64", "complex64"),
    "nn_functional_conv_transpose3d": ("int64", "complex64"),
    "nn_functional_cosine_embedding_loss": ("bool",),
    "nn_functional_cross_entropy": ("bfloat16",),
    "nn_functional_embedding_bag": ("bfloat16", "float16"),
    "nn_functional_group_norm": ("float32", "bfloat16", "float16"),
    "nn_functional_huber_loss": ("float16",),
    "nn_functional_l1_loss": ("complex64",),
    "nn_functional_layer_norm": ("float32", "bfloat16", "float16"),
    "nn_functional_linear": ("complex64",),
    "nn_functional_local_response_norm": ("int64",),
    "nn_functional_max_pool2d": ("int64",),
    "nn_functional_max_pool3d": ("int64",),
    "nn_functional_nll_loss": ("float16",),
    "nn_functional_normalize": ("complex64",),
    "nn_functional_pad_reflect": ("complex64",),
    "nn_functional_pad_replicate": ("complex64",),
    "nn_functional_pad_replicate_negative": ("complex64",),
    "nn_functional_pairwise_distance": ("complex64",),
    "nn_functional_poisson_nll_loss": ("int64",),
    "nn_functional_rms_norm": ("complex64",),
    "nn_functional_silu_complex": ("complex64",),
    "nn_functional_softmin_with_dtype": ("complex64",),
    "nn_functional_softsign": ("complex64",),
    "nn_functional_tanhshrink": ("complex64",),
    "nn_functional_triplet_margin_loss": ("complex64",),
    "nn_functional_triplet_margin_with_distance_loss": ("complex64",),
    "nn_functional_unfold": ("complex64",),
    "norm": ("complex64",),
    "norm_fro": ("complex64",),
    "norm_inf": ("complex64",),
    "norm_nuc": ("complex64",),
    "ormqr": ("complex64",),
    "outer": ("complex64",),
    "pinverse": ("complex64",),
    "polar": ("float32",),
    "pow": ("complex64",),
    "prod": ("bool", "complex64"),
    "put": ("complex64",),
    "qr": ("complex64",),
    "quantile": ("float32",),
    "reciprocal": ("complex64",),
    "renorm": ("complex64",),
    "rsqrt": ("complex64",),
    "rsub": ("complex64",),
    "scatter": ("complex64",),
    "scatter_add": ("complex64",),
    "sgn": ("complex64",),
    "sigmoid": ("complex64",),
    "sin": ("complex64",),
    "sinc": ("complex64",),
    "sinh": ("complex64",),
    "sort": ("bfloat16", "float16", "bool"),
    "sparse_sampled_addmm": ("float32", "complex64"),
    "sqrt": ("complex64",),
    "square": ("complex64",),
    "std": ("complex64",),
    "std_mean": ("complex64",),
    "std_mean_unbiased": ("complex64",),
    "std_unbiased": ("complex64",),
    "stft": ("complex64",),
    "sub": ("complex64",),
    "sum": ("complex64",),
    "sum_to_size": ("complex64",),
    "svd": ("complex64",),
    "take": ("complex64",),
    "tan": ("complex64",),
    "tanh": ("complex64",),
    "tensordot": ("complex64",),
    "to_sparse": ("float32", "bfloat16", "float16", "int64", "bool", "complex64"),
    "topk": ("bfloat16", "int64"),
    "trace": ("complex64",),
    "trapezoid": ("complex64",),
    "trapz": ("complex64",),
    "triangular_solve": ("complex64",),
    "true_divide": ("complex64",),
    "unfold": ("complex64",),
    "unfold_copy": ("complex64",),
    "var": ("complex64",),
    "var_mean": ("complex64",),
    "var_mean_unbiased": ("complex64",),
    "var_unbiased": ("complex64",),
    "vdot": ("complex64",),
    "view_as_complex": ("float16",),
}
# --- END GENERATED test_matches_cpu ---

# --- BEGIN GENERATED test_errors_match (regenerate_known_unsupported.py) ---
_ERRORS_MATCH: dict[str, tuple[str, ...]] = {
    "amax": ("float32",),
    "amin": ("float32",),
    "aminmax": ("float32",),
    "cat": ("float32",),
    "clamp_max": ("float32",),
    "clamp_min": ("float32",),
    "complex": ("float32",),
    "dstack": ("float32",),
    "hstack": ("float32",),
    "masked_select": ("float32",),
    "mean": ("float32",),
    "median": ("float32",),
    "native_layer_norm": ("float32",),
    "neg": ("float32",),
    "nn_functional_gelu": ("float32",),
    "roll": ("float32",),
    "vstack": ("float32",),
}
# --- END GENERATED test_errors_match ---

_BASE: dict[str, dict[str, tuple[str, ...]]] = {
    "test_matches_cpu": _MATCHES_CPU,
    "test_errors_match": _ERRORS_MATCH,
}

# Where each test's table lives, for the message that tells an author which
# entry to delete.
TABLE_NAMES: dict[str, str] = {
    "test_matches_cpu": "_MATCHES_CPU",
    "test_errors_match": "_ERRORS_MATCH",
}

# accelerator architecture name -> test -> operator -> dtypes, where an
# operator listed here REPLACES its base line entirely: the dtypes given are
# the ones that cannot pass on THIS accelerator, and the empty tuple means
# every dtype of that operator passes here.  One rule, so an operator's state
# on an accelerator is one line, e.g.
#
#     "amdgpu:gfx942": {
#         # The tensor-core route this needs is gated on sm_90a.
#         "test_matches_cpu": {"nn_functional_scaled_dot_product_attention": ()},
#     },
#
# "cpu" is a machine with no accelerator at all -- CI's ubuntu-latest
# runners. Kernels still run there (MAX has a CPU target), so this is a real
# measurement, not a stub: some ops raise on this path where they would not
# on a GPU (declared below under "test_matches_cpu"), and some
# error_inputs_func cases that provoke a GPU-only failure mode on
# BASE_ACCELERATOR instead run to completion here, so their base declaration
# is overridden to the empty tuple (declared below under "test_errors_match").
# Measured with `regenerate_known_unsupported.py -n 15 --records ...`; see
# that script's docstring for why `--write` refuses off BASE_ACCELERATOR and
# a delta has to be hand-derived from the printed diff instead.
#
# Not declared here: test_matches_cpu_stack_mojo_int64 segfaults (SIGSEGV,
# exit 139) on this path, reproducibly and standalone. A declared entry
# cannot excuse it -- a declared case still calls the operator and only
# excuses an EXCEPTION it raises; a segfault never raises one, it takes the
# whole process down before any Python `except` runs. That is a real bug in
# the stack/cat path for int64 with no accelerator present, out of scope for
# this table, and it will keep failing (or crashing whatever process runs
# it, CI shard included) until fixed at the kernel level.
_ACCELERATOR_DELTAS: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    # float_power promotes even integer/half inputs to float64 (ATen
    # native/Pow.cpp); Metal has no float64 device arithmetic or storage.
    "4-metal4": {
        "test_matches_cpu": {
            "float_power": (
                "float32",
                "bfloat16",
                "float16",
                "int64",
                "bool",
                "complex64",
            ),
            # torch's composites build a float64 0-d tolerance tensor the
            # float32 singular values are compared against.
            "linalg_matrix_rank": ("float32", "complex64"),
            "linalg_matrix_rank_hermitian": ("float32", "complex64"),
            "linalg_pinv": ("float32", "complex64"),
            "linalg_pinv_hermitian": ("float32", "complex64"),
            "pinverse": ("float32", "complex64"),
            "linalg_tensorsolve": ("complex64",),
            # Metal's float32 GEMM: 2.6x CPU's error from float64 on one
            # element of a 5x5 exponential with entries near 2000.
            "matrix_exp": ("float32", "bfloat16", "float16", "complex64"),
            # dtype=torch.float64 in every sample: no float64 on Metal.
            "log_softmax_with_dtype": (
                "float32",
                "bfloat16",
                "float16",
                "int64",
                "bool",
                "complex64",
            ),
            "nn_functional_softmin_with_dtype": (
                "float32",
                "bfloat16",
                "float16",
                "int64",
                "complex64",
            ),
            # float32 FFT rounding (see the base complex64 entry), 1.6e-5
            # relative on one near-zero output of the 3-D sample.
            "fft_hfft2": ("float32", "int64", "complex64"),
            # a float64 operand on the way (no float64 on Metal)
            "nn_functional_l1_loss": ("complex64",),
            "softmax_with_dtype": (
                "float32",
                "bfloat16",
                "float16",
                "int64",
                "bool",
                "complex64",
            ),
        },
        # error inputs built with a float64 tensor (uniform_ of float64):
        # no float64 on Metal. Base-table passes since complex arrived.
        "test_errors_match": {
            "complex": ("float32",),
            "nn_functional_conv1d": ("float32",),
            "nn_functional_conv2d": ("float32",),
            "nn_functional_conv3d": ("float32",),
        },
    },
    "cpu": {
        "test_matches_cpu": {
            "bmm": ("float32", "int64"),
            "cumulative_trapezoid": ("float32", "bfloat16", "float16", "int64"),
            "log_softmax": ("bfloat16", "float16"),
            # The float64 row-softmax kernel is GPU-only (not measured here:
            # it raises without an accelerator).
            "log_softmax_with_dtype": (
                "float32",
                "bfloat16",
                "float16",
                "int64",
                "bool",
            ),
            "masked_log_softmax": ("bfloat16", "float16"),
            "native_dropout_backward": (
                "float32",
                "bfloat16",
                "float16",
                "int64",
                "bool",
            ),
            "nn_functional_batch_norm": ("bfloat16", "float16"),
            "nn_functional_conv2d": ("bfloat16", "float16", "int64"),
            "nn_functional_instance_norm": ("bfloat16", "float16"),
            "nn_functional_softmin_with_dtype": (
                "float32",
                "bfloat16",
                "float16",
                "int64",
            ),
            "softmax_with_dtype": ("float32", "bfloat16", "float16", "int64", "bool"),
        },
        "test_errors_match": {},
    },
    "gfx942": {"test_matches_cpu": {}, "test_errors_match": {}},
}


def dtype_token(dtype: torch.dtype) -> str:
    """The dtype as it appears in a test node id: `torch.float16` -> "float16"."""
    return str(dtype).removeprefix("torch.")


@functools.cache
def accelerator_key() -> str:
    """Architecture name of the accelerator the mojo device runs kernels on.

    "sm_90a", "gfx942" (MI300A), ... -- MAX's `architecture_name` of the
    device, so a table keyed by it splits exactly where support can differ.
    "cpu" when the machine has no accelerator at all.

    Cached: every node asks, and the answer cannot change inside one process.
    """
    for device in get_accelerators():
        if device.label != "cpu":
            return str(device.architecture_name)
    return "cpu"


def declared(test: str, accelerator: str | None = None) -> dict[str, frozenset[str]]:
    """Operator -> dtype tokens declared unable to pass `test` on `accelerator`.

    The base table with this accelerator's delta applied.  Operators the
    delta maps to no dtypes at all are dropped: they are the "supported here"
    markers, and support is the absence of an entry.
    """
    if accelerator is None:
        accelerator = accelerator_key()
    table = dict(_BASE[test])
    table.update(_ACCELERATOR_DELTAS.get(accelerator, {}).get(test, {}))
    return {op: frozenset(dtypes) for op, dtypes in table.items() if dtypes}


def declared_unsupported(
    test: str, op_name: str, dtype: torch.dtype, accelerator: str | None = None
) -> str | None:
    """Why this node is declared unable to pass, or None if nothing declares it.

    `op_name` is `OpInfo.formatted_name`, the operator token of the node id.
    """
    if accelerator is None:
        accelerator = accelerator_key()
    token = dtype_token(dtype)
    if token not in declared(test, accelerator).get(op_name, frozenset()):
        return None
    return (
        f"declared unsupported on {accelerator}: "
        f"conformance/known_unsupported.py {TABLE_NAMES[test]}[{op_name!r}] "
        f"lists {token!r}"
    )
