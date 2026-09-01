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
    "__getitem__": ("float32", "bfloat16", "float16", "int64", "bool"),
    "__radd__": ("bool",),
    "__rdiv__": ("int64", "bool"),
    "__rmatmul__": ("float32", "bfloat16", "float16", "int64"),
    "__rpow__": ("int64",),
    "__rsub__": ("float32", "bfloat16", "float16", "int64"),
    "_batch_norm_with_update": ("float32", "bfloat16", "float16"),
    "_segment_reduce_lengths": ("float32", "bfloat16", "float16"),
    "_segment_reduce_offsets": ("float32", "bfloat16", "float16"),
    "_softmax_backward_data": ("float32", "bfloat16", "float16"),
    "_unsafe_masked_index": ("float32", "bfloat16", "float16", "int64", "bool"),
    "_unsafe_masked_index_put_accumulate": (
        "float32",
        "bfloat16",
        "float16",
        "int64",
        "bool",
    ),
    "_upsample_bilinear2d_aa": ("float32",),
    "acos": ("int64", "bool"),
    "acosh": ("float32", "bfloat16", "float16", "int64", "bool"),
    "add": ("bool",),
    "addbmm": ("float32", "bfloat16", "float16", "int64"),
    "addmm": ("float32", "bfloat16", "float16", "int64"),
    "addmm_decomposed": ("float32", "bfloat16", "float16", "int64"),
    "addmv": ("float32", "bfloat16", "float16", "int64"),
    "addr": ("bool",),
    "all": ("float32", "bfloat16", "float16", "int64", "bool"),
    "allclose": ("float32", "bfloat16", "float16"),
    "amax": ("float32", "bfloat16", "float16", "int64", "bool"),
    "amin": ("float32", "bfloat16", "float16", "int64", "bool"),
    "aminmax": ("float32", "bfloat16", "float16", "int64", "bool"),
    "angle": ("float32", "bfloat16", "float16", "int64", "bool"),
    "any": ("float32", "bfloat16", "float16", "int64", "bool"),
    "argmax": ("float32", "bfloat16", "float16", "int64"),
    "argmin": ("float32", "bfloat16", "float16", "int64"),
    "argsort": ("bfloat16", "float16", "bool"),
    "as_strided": ("float32", "bfloat16", "float16", "int64", "bool"),
    "as_strided_copy": ("float32", "bfloat16", "float16", "int64", "bool"),
    "as_strided_scatter": ("float32", "bfloat16", "float16", "int64", "bool"),
    "asin": ("float32", "bfloat16", "float16", "int64", "bool"),
    "asinh": ("int64", "bool"),
    "atan": ("float32", "bfloat16", "float16", "int64", "bool"),
    "atan2": ("float32", "bfloat16", "float16", "int64", "bool"),
    "atanh": ("int64", "bool"),
    "baddbmm": ("float32", "bfloat16", "float16", "int64"),
    "bincount": ("int64",),
    "bmm": ("int64",),
    "cat": ("float32", "bfloat16", "float16", "int64", "bool"),
    "cdist": ("float32",),
    "cdouble": ("float32", "bfloat16", "float16", "int64", "bool"),
    "cfloat": ("float32", "bfloat16", "float16", "int64", "bool"),
    "chalf": ("float32", "bfloat16", "float16", "int64", "bool"),
    "cholesky": ("float32",),
    "cholesky_inverse": ("float32",),
    "cholesky_solve": ("float32",),
    "clamp": ("float32", "bfloat16", "float16", "int64"),
    "clamp_max": ("float32", "bfloat16", "float16", "int64", "bool"),
    "clamp_min": ("float32", "bfloat16", "float16", "int64", "bool"),
    "combinations": ("float32", "bfloat16", "float16", "int64", "bool"),
    "complex": ("float32", "float16"),
    "copysign": ("float32", "bfloat16", "float16", "int64", "bool"),
    "corrcoef": ("float32", "bfloat16", "float16", "int64"),
    "cos": ("int64", "bool"),
    "cosh": ("int64", "bool"),
    "count_nonzero": ("float32", "bfloat16", "float16", "int64", "bool"),
    "cov": ("float32", "bfloat16", "float16", "int64"),
    "cross": ("float32", "bfloat16", "float16", "int64"),
    "cummax": ("float32", "bfloat16", "float16", "int64", "bool"),
    "cummin": ("float32", "bfloat16", "float16", "int64", "bool"),
    "cumprod": ("float32", "bfloat16", "float16", "int64"),
    "cumsum": ("float32", "bfloat16", "float16", "int64"),
    "cumulative_trapezoid": ("float32", "bfloat16", "float16", "int64"),
    "deg2rad": ("int64", "bool"),
    "diag": ("float32", "bfloat16", "float16", "int64", "bool"),
    "diag_embed": ("float32", "bfloat16", "float16", "int64", "bool"),
    "diagflat": ("float32", "bfloat16", "float16", "int64", "bool"),
    "diagonal": ("float32", "bfloat16", "float16", "int64", "bool"),
    "diagonal_copy": ("float32", "bfloat16", "float16", "int64", "bool"),
    "diagonal_scatter": ("float32", "bfloat16", "float16", "int64", "bool"),
    "digamma": ("float32", "bfloat16", "float16", "int64", "bool"),
    "dist": ("float32", "bfloat16", "float16"),
    "div_no_rounding_mode": ("bool",),
    "dot": ("float32", "bfloat16", "float16", "int64"),
    "einsum": ("float32", "bfloat16", "float16", "int64"),
    "equal": ("float32", "bfloat16", "float16", "int64", "bool"),
    "erf": ("int64", "bool"),
    "erfc": ("float32", "bfloat16", "float16", "int64", "bool"),
    "erfinv": ("float32", "bfloat16", "float16", "int64", "bool"),
    "exp": ("int64", "bool"),
    "exp2": ("float32", "bfloat16", "float16", "int64", "bool"),
    "expm1": ("float32", "bfloat16", "float16", "int64", "bool"),
    "fft_fft": ("float32", "int64", "bool"),
    "fft_fft2": ("float32", "int64", "bool"),
    "fft_fftn": ("float32", "int64", "bool"),
    "fft_fftshift": ("float32", "bfloat16", "float16", "int64", "bool"),
    "fft_hfft": ("float32", "int64", "bool"),
    "fft_hfft2": ("float32", "int64", "bool"),
    "fft_hfftn": ("float32", "int64", "bool"),
    "fft_ifft": ("float32", "int64", "bool"),
    "fft_ifft2": ("float32", "int64", "bool"),
    "fft_ifftn": ("float32", "int64", "bool"),
    "fft_ifftshift": ("float32", "bfloat16", "float16", "int64", "bool"),
    "fft_ihfft": ("float32", "int64", "bool"),
    "fft_ihfft2": ("float32", "int64", "bool"),
    "fft_ihfftn": ("float32", "int64", "bool"),
    "fft_irfft": ("float32", "int64", "bool"),
    "fft_irfft2": ("float32", "int64", "bool"),
    "fft_irfftn": ("float32", "int64", "bool"),
    "fft_rfft": ("float32", "int64", "bool"),
    "fft_rfft2": ("float32", "int64", "bool"),
    "fft_rfftn": ("float32", "int64", "bool"),
    "flip": ("float32", "bfloat16", "float16", "int64", "bool"),
    "fliplr": ("float32", "bfloat16", "float16", "int64", "bool"),
    "flipud": ("float32", "bfloat16", "float16", "int64", "bool"),
    "float_power": ("float32", "bfloat16", "float16", "int64", "bool"),
    "fmax": ("float32", "bfloat16", "float16", "int64", "bool"),
    "fmin": ("float32", "bfloat16", "float16", "int64", "bool"),
    "fmod": ("float32", "bfloat16", "float16", "int64"),
    "frac": ("float32", "bfloat16", "float16"),
    "frexp": ("float32", "bfloat16", "float16"),
    "gather": ("float32", "bfloat16", "float16", "int64", "bool"),
    "gcd": ("int64",),
    "geqrf": ("float32",),
    "gradient": ("int64",),
    "grid_sampler_2d": ("float32", "bfloat16", "float16"),
    "grid_sampler_3d": ("float32", "bfloat16", "float16"),
    "hash_tensor": ("float32", "bfloat16", "float16", "int64", "bool"),
    "heaviside": ("float32", "bfloat16", "float16", "int64", "bool"),
    "histc": ("float32", "bfloat16", "float16"),
    "histogram": ("float32",),
    "histogramdd": ("float32",),
    "hypot": ("float32", "bfloat16", "float16"),
    "i0": ("float32", "bfloat16", "float16", "int64", "bool"),
    "igamma": ("float32", "bfloat16", "float16"),
    "igammac": ("float32", "bfloat16", "float16"),
    "index_add": ("float32", "bfloat16", "float16", "int64", "bool"),
    "index_copy": ("float32", "bfloat16", "float16", "int64", "bool"),
    "index_fill": ("float32", "bfloat16", "float16", "int64", "bool"),
    "index_put": ("float32", "bfloat16", "float16", "int64", "bool"),
    "index_reduce_amax": ("float32", "bfloat16", "float16", "int64"),
    "index_reduce_amin": ("float32", "bfloat16", "float16", "int64"),
    "index_reduce_mean": ("float32", "bfloat16", "float16", "int64"),
    "index_reduce_prod": ("float32", "bfloat16", "float16", "int64"),
    "index_select": ("float32", "bfloat16", "float16", "int64", "bool"),
    "inner": ("float32", "bfloat16", "float16", "int64"),
    "isclose": ("float32", "bfloat16", "float16", "int64", "bool"),
    "isin": ("float32", "bfloat16", "float16"),
    "isneginf": ("float32", "bfloat16", "float16", "int64", "bool"),
    "isposinf": ("float32", "bfloat16", "float16", "int64", "bool"),
    # float32/float16: OpInfo includes a 0-d input, which the reused
    # sort/topk kernel this op composes on top of doesn't accept.
    # bfloat16/int64: real, not-implementable-away tie-index disagreement
    # against ATen's own introselect tie order on inputs with duplicate
    # values (bfloat16's narrow range and int64's small-integer OpInfo
    # samples both hit real ties) -- same accepted-disagreement class as
    # "sort"/"topk" below, whose own kernel this op reuses.
    "kthvalue": ("float32", "float16", "bfloat16", "int64"),
    "lcm": ("int64",),
    "ldexp": ("float32", "int64", "bool"),
    "lerp": ("float32", "bfloat16", "float16"),
    "lgamma": ("float32", "bfloat16", "float16", "int64", "bool"),
    "linalg_cholesky": ("float32",),
    "linalg_cholesky_ex": ("float32",),
    "linalg_cond": ("float32",),
    "linalg_cross": ("float32", "bfloat16", "float16", "int64"),
    "linalg_det": ("float32",),
    "linalg_diagonal": ("float32", "bfloat16", "float16", "int64", "bool"),
    "linalg_eig": ("float32",),
    "linalg_eigh": ("float32",),
    "linalg_eigvals": ("float32",),
    "linalg_eigvalsh": ("float32",),
    "linalg_householder_product": ("float32",),
    "linalg_inv": ("float32",),
    "linalg_inv_ex": ("float32",),
    "linalg_ldl_factor": ("float32",),
    "linalg_ldl_factor_ex": ("float32",),
    "linalg_ldl_solve": ("float32",),
    "linalg_lstsq": ("float32",),
    "linalg_lstsq_grad_oriented": ("float32",),
    "linalg_lu_solve": ("float32",),
    "linalg_matrix_norm": ("float32", "bfloat16", "float16"),
    "linalg_matrix_power": ("float32",),
    "linalg_matrix_rank": ("float32",),
    "linalg_matrix_rank_hermitian": ("float32",),
    "linalg_multi_dot": ("float32", "bfloat16", "float16", "int64"),
    "linalg_norm": ("float32", "bfloat16", "float16"),
    "linalg_norm_subgradients_at_zero": ("float32", "bfloat16", "float16"),
    "linalg_pinv": ("float32",),
    "linalg_pinv_hermitian": ("float32",),
    "linalg_qr": ("float32",),
    "linalg_slogdet": ("float32",),
    "linalg_solve": ("float32",),
    "linalg_solve_ex": ("float32",),
    "linalg_solve_triangular": ("float32",),
    "linalg_svd": ("float32",),
    "linalg_svdvals": ("float32",),
    "linalg_tensorinv": ("float32",),
    "linalg_tensorsolve": ("float32",),
    "linalg_vander": ("float32", "int64"),
    "linalg_vecdot": ("float32", "bfloat16", "float16"),
    "linalg_vector_norm": ("float32", "bfloat16", "float16"),
    "log": ("int64", "bool"),
    "log10": ("float32", "bfloat16", "float16", "int64", "bool"),
    "log1p": ("int64", "bool"),
    "log2": ("float32", "bfloat16", "float16", "int64", "bool"),
    "log_softmax": ("float32",),
    "log_softmax_with_dtype": ("float32", "bfloat16", "float16", "int64", "bool"),
    "logaddexp": ("float32", "bfloat16", "float16"),
    "logaddexp2": ("float32", "bfloat16", "float16"),
    "logcumsumexp": ("float32", "bfloat16", "float16"),
    "logdet": ("float32",),
    "logical_or": ("float32", "bfloat16", "float16", "int64", "bool"),
    "logit": ("float32", "bfloat16", "float16", "int64", "bool"),
    "logsumexp": ("float32", "bfloat16", "float16", "int64", "bool"),
    "lu": ("float32",),
    "lu_solve": ("float32",),
    "lu_unpack": ("float32",),
    "masked_amax": ("float32", "bfloat16", "float16", "int64"),
    "masked_amin": ("float32", "bfloat16", "float16", "int64"),
    "masked_argmax": ("float32", "bfloat16", "float16", "int64"),
    "masked_argmin": ("float32", "bfloat16", "float16", "int64"),
    "masked_cumprod": ("float32", "bfloat16", "float16", "int64"),
    "masked_cumsum": ("float32", "bfloat16", "float16", "int64"),
    "masked_fill": ("float32", "bfloat16", "float16", "int64", "bool"),
    "masked_log_softmax": ("float32",),
    "masked_logaddexp": ("float32", "bfloat16", "float16"),
    "masked_logsumexp": ("float32", "bfloat16", "float16", "int64"),
    "masked_mean": ("float32", "bfloat16", "float16"),
    "masked_median": ("float32", "bfloat16", "float16"),
    "masked_norm": ("float32", "bfloat16", "float16"),
    "masked_normalize": ("float32", "bfloat16", "float16"),
    "masked_prod": ("float32", "bfloat16", "float16", "int64", "bool"),
    "masked_scatter": ("float32", "bfloat16", "float16", "int64", "bool"),
    "masked_select": ("float32", "bfloat16", "float16", "int64", "bool"),
    "masked_softmax": ("float32", "bfloat16", "float16"),
    "masked_softmin": ("float32", "bfloat16", "float16"),
    "masked_std": ("float32", "bfloat16", "float16", "int64"),
    "masked_sum": ("float32", "bfloat16", "float16", "int64", "bool"),
    "masked_var": ("float32", "bfloat16", "float16", "int64"),
    "matmul": ("float32", "bfloat16", "float16", "int64"),
    "matrix_exp": ("float32", "bfloat16", "float16"),
    "max_binary": ("bool",),
    "max_pool2d_with_indices_backward": ("float32", "bfloat16", "float16"),
    "max_reduction_no_dim": ("float32", "bfloat16", "float16", "int64", "bool"),
    "max_reduction_with_dim": ("float32", "bfloat16", "float16", "int64", "bool"),
    "maximum": ("bool",),
    "mean": ("float32", "bfloat16", "float16"),
    "median": ("float32", "bfloat16", "float16", "int64"),
    "min_binary": ("bool",),
    "min_reduction_no_dim": ("float32", "bfloat16", "float16", "int64", "bool"),
    "min_reduction_with_dim": ("float32", "bfloat16", "float16", "int64", "bool"),
    "minimum": ("bool",),
    "mm": ("float32", "bfloat16", "float16", "int64"),
    "mode": ("float32", "bfloat16", "float16", "int64", "bool"),
    "mv": ("float32", "bfloat16", "float16", "int64"),
    "mvlgamma_mvlgamma_p_1": ("float32", "bfloat16", "float16", "int64"),
    "mvlgamma_mvlgamma_p_3": ("float32", "bfloat16", "float16", "int64"),
    "mvlgamma_mvlgamma_p_5": ("float32", "bfloat16", "float16", "int64"),
    "nan_to_num": ("float32", "bfloat16", "float16", "int64", "bool"),
    "nanmean": ("float32", "bfloat16", "float16"),
    "nanmedian": ("float32", "bfloat16", "float16", "int64"),
    "nanquantile": ("float32",),
    "nansum": ("float32", "bfloat16", "float16", "int64", "bool"),
    "native_dropout_backward": ("bfloat16", "float16", "int64", "bool"),
    "native_layer_norm": ("float32", "bfloat16", "float16"),
    "nextafter": ("float32", "bfloat16", "float16"),
    "nn_functional_adaptive_avg_pool1d": ("float32", "bfloat16", "float16"),
    "nn_functional_adaptive_avg_pool2d": ("float32", "bfloat16", "float16"),
    "nn_functional_adaptive_avg_pool3d": ("float32", "bfloat16", "float16"),
    "nn_functional_adaptive_max_pool1d": ("float32", "bfloat16", "float16"),
    "nn_functional_adaptive_max_pool2d": ("float32", "bfloat16", "float16"),
    "nn_functional_adaptive_max_pool3d": ("float32", "bfloat16", "float16"),
    "nn_functional_alpha_dropout": ("float32", "bfloat16", "float16"),
    "nn_functional_avg_pool1d": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_avg_pool2d": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_avg_pool3d": ("float32", "int64"),
    "nn_functional_bilinear": ("int64",),
    "nn_functional_binary_cross_entropy": ("float32", "bfloat16", "float16"),
    "nn_functional_binary_cross_entropy_with_logits": (
        "float32",
        "bfloat16",
        "float16",
    ),
    "nn_functional_celu": ("float32", "bfloat16", "float16"),
    "nn_functional_channel_shuffle": (
        "float32",
        "bfloat16",
        "float16",
        "int64",
        "bool",
    ),
    "nn_functional_conv1d": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_conv2d": ("int64",),
    "nn_functional_conv_transpose1d": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_conv_transpose2d": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_conv_transpose3d": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_cosine_embedding_loss": (
        "float32",
        "bfloat16",
        "float16",
        "int64",
        "bool",
    ),
    "nn_functional_cosine_similarity": ("float32", "bfloat16", "float16"),
    "nn_functional_cross_entropy": ("float32", "bfloat16", "float16"),
    "nn_functional_ctc_loss": ("float32",),
    "nn_functional_elu": ("float32", "bfloat16", "float16"),
    "nn_functional_embedding_bag": ("float32", "bfloat16", "float16"),
    "nn_functional_gaussian_nll_loss": ("float32", "bfloat16", "float16"),
    "nn_functional_glu": ("float32", "bfloat16", "float16"),
    "nn_functional_grid_sample": ("float32", "bfloat16", "float16"),
    "nn_functional_group_norm": ("float32", "bfloat16", "float16"),
    "nn_functional_hardshrink": ("float32", "bfloat16", "float16"),
    "nn_functional_hardsigmoid": ("float32", "bfloat16", "float16"),
    "nn_functional_hardswish": ("float32", "bfloat16", "float16"),
    "nn_functional_hardtanh": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_hinge_embedding_loss": ("float32", "bfloat16", "float16"),
    "nn_functional_huber_loss": ("float32", "bfloat16", "float16"),
    "nn_functional_interpolate_area": ("float32", "bfloat16", "float16"),
    "nn_functional_interpolate_bicubic": ("float32", "bfloat16", "float16"),
    "nn_functional_interpolate_linear": ("float32", "bfloat16", "float16"),
    "nn_functional_interpolate_nearest": ("float32", "bfloat16", "float16"),
    "nn_functional_interpolate_nearest-exact": ("float32", "bfloat16", "float16"),
    "nn_functional_interpolate_trilinear": ("float32", "bfloat16", "float16"),
    "nn_functional_kl_div": ("float32", "bfloat16", "float16"),
    "nn_functional_l1_loss": ("float32", "bfloat16", "float16"),
    "nn_functional_layer_norm": ("float32", "bfloat16", "float16"),
    "nn_functional_leaky_relu": ("float32", "bfloat16", "float16"),
    "nn_functional_linear": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_local_response_norm": ("int64",),
    "nn_functional_logsigmoid": ("float32", "bfloat16", "float16"),
    "nn_functional_margin_ranking_loss": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_max_pool1d": ("float32", "bfloat16", "float16"),
    "nn_functional_max_pool2d": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_max_pool3d": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_max_unpool1d": ("float32", "bfloat16", "float16"),
    "nn_functional_max_unpool1d_grad": ("float32", "bfloat16", "float16"),
    "nn_functional_max_unpool2d": ("float32", "bfloat16", "float16"),
    "nn_functional_max_unpool2d_grad": ("float32", "bfloat16", "float16"),
    "nn_functional_max_unpool3d": ("float32", "bfloat16", "float16"),
    "nn_functional_max_unpool3d_grad": ("float32", "bfloat16", "float16"),
    "nn_functional_mish": ("float32", "bfloat16", "float16"),
    "nn_functional_mse_loss": ("float32", "bfloat16", "float16"),
    "nn_functional_multi_margin_loss": ("float32",),
    "nn_functional_multilabel_margin_loss": ("float32",),
    "nn_functional_multilabel_soft_margin_loss": ("float32", "bfloat16", "float16"),
    "nn_functional_nll_loss": ("float32", "bfloat16", "float16"),
    "nn_functional_normalize": ("float32", "bfloat16", "float16"),
    "nn_functional_one_hot": ("int64",),
    "nn_functional_pad_reflect": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_pad_replicate": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_pad_replicate_negative": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_pairwise_distance": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_pdist": ("float32",),
    "nn_functional_poisson_nll_loss": ("int64",),
    "nn_functional_prelu": ("float32", "bfloat16", "float16"),
    "nn_functional_relu6": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_selu": ("float32", "bfloat16", "float16"),
    "nn_functional_smooth_l1_loss": ("float32", "bfloat16", "float16"),
    "nn_functional_soft_margin_loss": ("float32", "bfloat16", "float16"),
    "nn_functional_softmin": ("float32", "bfloat16", "float16"),
    "nn_functional_softmin_with_dtype": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_softplus": ("float32", "bfloat16", "float16"),
    "nn_functional_softshrink": ("float32", "bfloat16", "float16"),
    "nn_functional_tanhshrink": ("int64",),
    "nn_functional_threshold": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_triplet_margin_loss": ("float32", "bfloat16", "float16", "int64"),
    "nn_functional_triplet_margin_with_distance_loss": (
        "float32",
        "bfloat16",
        "float16",
        "int64",
    ),
    "nn_functional_unfold": ("float32", "bfloat16", "float16", "bool"),
    "nn_functional_upsample_nearest": ("float32", "bfloat16", "float16"),
    "norm": ("float32", "bfloat16", "float16"),
    "norm_inf": ("float32", "bfloat16", "float16"),
    "norm_nuc": ("float32",),
    "ormqr": ("float32",),
    "pinverse": ("float32",),
    "polar": ("float32",),
    "polygamma_polygamma_n_0": ("float32", "bfloat16", "float16", "int64", "bool"),
    "polygamma_polygamma_n_1": ("float32", "bfloat16", "int64", "bool"),
    "polygamma_polygamma_n_2": ("float32", "bfloat16", "int64", "bool"),
    "polygamma_polygamma_n_3": ("float32", "bfloat16", "int64", "bool"),
    "polygamma_polygamma_n_4": ("float32", "bfloat16", "int64", "bool"),
    "pow": ("int64",),
    "prod": ("float32", "bfloat16", "float16", "int64", "bool"),
    "put": ("float32", "bfloat16", "float16", "int64", "bool"),
    "qr": ("float32",),
    "quantile": ("float32",),
    "rad2deg": ("int64", "bool"),
    "reciprocal": ("int64", "bool"),
    "renorm": ("float32", "bfloat16", "float16"),
    "repeat_interleave": ("float32", "bfloat16", "float16", "int64", "bool"),
    "resize_": ("float32", "bfloat16", "float16", "int64", "bool"),
    "resize_as_": ("float32", "bfloat16", "float16", "int64", "bool"),
    "roll": ("float32", "bfloat16", "float16", "int64", "bool"),
    "rot90": ("float32", "bfloat16", "float16", "int64", "bool"),
    "round": ("float32", "bfloat16", "float16", "int64"),
    "round_decimals_0": ("float32", "bfloat16", "float16"),
    "round_decimals_3": ("float32", "bfloat16"),
    "round_decimals_neg_3": ("float32", "bfloat16"),
    "rsqrt": ("int64", "bool"),
    "rsub": ("float32", "bfloat16", "float16", "int64"),
    "scatter": ("float32", "bfloat16", "float16", "int64", "bool"),
    "scatter_add": ("float32", "bfloat16", "float16", "int64", "bool"),
    "scatter_reduce_amax": ("float32", "bfloat16", "float16", "int64", "bool"),
    "scatter_reduce_amin": ("float32", "bfloat16", "float16", "int64", "bool"),
    "scatter_reduce_mean": ("float32", "bfloat16", "float16", "int64"),
    "scatter_reduce_prod": ("float32", "bfloat16", "float16", "int64", "bool"),
    "scatter_reduce_sum": ("float32", "bfloat16", "float16", "int64", "bool"),
    "sgn": ("float32", "bfloat16", "float16", "int64", "bool"),
    "sigmoid": ("int64", "bool"),
    "sign": ("bool",),
    "signbit": ("float32", "bfloat16", "float16", "int64", "bool"),
    "sin": ("int64", "bool"),
    "sinc": ("float32", "bfloat16", "float16", "int64", "bool"),
    "sinh": ("int64", "bool"),
    "slice_scatter": ("float32", "bfloat16", "float16", "int64", "bool"),
    "softmax": ("float32", "bfloat16", "float16"),
    "softmax_with_dtype": ("float32", "bfloat16", "float16", "int64", "bool"),
    "sort": ("bfloat16", "float16", "bool"),
    "sparse_sampled_addmm": ("float32",),
    "special_airy_ai": ("float32", "int64", "bool"),
    "special_bessel_j0": ("float32", "int64", "bool"),
    "special_bessel_j1": ("float32", "int64", "bool"),
    "special_bessel_y0": ("float32", "int64", "bool"),
    "special_bessel_y1": ("float32", "int64", "bool"),
    "special_entr": ("float32", "bfloat16", "float16", "int64", "bool"),
    "special_erfcx": ("float32", "int64", "bool"),
    "special_i0e": ("float32", "bfloat16", "float16", "int64", "bool"),
    "special_i1": ("float32", "bfloat16", "float16", "int64", "bool"),
    "special_i1e": ("float32", "bfloat16", "float16", "int64", "bool"),
    "special_log_ndtr": ("float32", "int64", "bool"),
    "special_modified_bessel_i0": ("float32", "int64", "bool"),
    "special_modified_bessel_i1": ("float32", "int64", "bool"),
    "special_modified_bessel_k0": ("float32", "int64", "bool"),
    "special_modified_bessel_k1": ("float32", "int64", "bool"),
    "special_ndtr": ("int64", "bool"),
    "special_ndtri": ("float32", "int64", "bool"),
    "special_polygamma_special_polygamma_n_0": (
        "float32",
        "bfloat16",
        "float16",
        "int64",
        "bool",
    ),
    "special_scaled_modified_bessel_k0": ("float32", "int64", "bool"),
    "special_scaled_modified_bessel_k1": ("float32", "int64", "bool"),
    "special_spherical_bessel_j0": ("float32", "int64", "bool"),
    "special_xlog1py": ("float32", "bfloat16", "float16", "int64", "bool"),
    "sqrt": ("int64", "bool"),
    "square": ("int64", "bool"),
    "squeeze": ("float32", "bfloat16", "float16", "int64", "bool"),
    "squeeze_copy": ("float32", "bfloat16", "float16", "int64", "bool"),
    "squeeze_multiple": ("float32", "bfloat16", "float16", "int64", "bool"),
    "std": ("float32", "bfloat16", "float16"),
    "std_mean": ("float32", "bfloat16", "float16"),
    "std_mean_unbiased": ("float32", "bfloat16", "float16"),
    "std_unbiased": ("float32", "bfloat16", "float16"),
    "stft": ("float32",),
    "sum": ("float32", "bfloat16", "float16", "int64", "bool"),
    "sum_to_size": ("float32", "bfloat16", "float16", "int64", "bool"),
    "svd": ("float32",),
    "take": ("float32", "bfloat16", "float16", "int64", "bool"),
    "take_along_dim": ("float32", "bfloat16", "float16", "int64", "bool"),
    "tan": ("int64", "bool"),
    "tanh": ("int64", "bool"),
    "tensordot": ("float32", "bfloat16", "float16", "int64"),
    "to_sparse": ("float32", "bfloat16", "float16", "int64", "bool"),
    "topk": ("bfloat16", "int64"),
    "torch_ops_aten__safe_softmax_default": (
        "float32",
        "bfloat16",
        "float16",
        "int64",
        "bool",
    ),
    "trace": ("float32", "int64"),
    "trapezoid": ("int64",),
    "trapz": ("int64",),
    "triangular_solve": ("float32",),
    "true_divide": ("bool",),
    "trunc": ("float32", "bfloat16", "float16", "int64"),
    "unfold": ("float32", "bfloat16", "float16", "int64", "bool"),
    "unfold_copy": ("float32", "bfloat16", "float16", "int64", "bool"),
    "unique_consecutive": ("float32", "bfloat16", "float16", "int64", "bool"),
    "var_mean": ("float32", "bfloat16", "float16"),
    "var_mean_unbiased": ("float32", "bfloat16", "float16"),
    "vdot": ("float32", "bfloat16", "float16", "int64"),
    "view_as_complex": ("float32", "float16"),
    "xlogy": ("float32", "bfloat16", "float16", "int64", "bool"),
}
# --- END GENERATED test_matches_cpu ---

# --- BEGIN GENERATED test_errors_match (regenerate_known_unsupported.py) ---
_ERRORS_MATCH: dict[str, tuple[str, ...]] = {
    "_chunk_cat": ("float32",),
    "amax": ("float32",),
    "amin": ("float32",),
    "aminmax": ("float32",),
    "as_strided_scatter": ("float32",),
    "bernoulli": ("float32",),
    "cat": ("float32",),
    "cauchy": ("float32",),
    "clamp_max": ("float32",),
    "clamp_min": ("float32",),
    "complex": ("float32",),
    "cov": ("float32",),
    "diag_embed": ("float32",),
    "diagonal": ("float32",),
    "diagonal_copy": ("float32",),
    "diff": ("float32",),
    "dot": ("float32",),
    "dsplit": ("float32",),
    "dstack": ("float32",),
    "exponential": ("float32",),
    "eye": ("float32",),
    "fft_fft": ("float32",),
    "fft_fft2": ("float32",),
    "fft_fftn": ("float32",),
    "fft_hfft": ("float32",),
    "fft_hfft2": ("float32",),
    "fft_hfftn": ("float32",),
    "fft_ifft": ("float32",),
    "fft_ifft2": ("float32",),
    "fft_ifftn": ("float32",),
    "fft_ihfft": ("float32",),
    "fft_ihfft2": ("float32",),
    "fft_ihfftn": ("float32",),
    "fft_irfft": ("float32",),
    "fft_irfft2": ("float32",),
    "fft_irfftn": ("float32",),
    "fft_rfft": ("float32",),
    "fft_rfft2": ("float32",),
    "fft_rfftn": ("float32",),
    "fliplr": ("float32",),
    "flipud": ("float32",),
    "gather": ("float32",),
    "geometric": ("float32",),
    "gradient": ("float32",),
    "histogramdd": ("float32",),
    "hsplit": ("float32",),
    "hstack": ("float32",),
    "index_select": ("float32",),
    "isclose": ("float32",),
    "item": ("float32",),
    # A 0-d input + an out-of-range k + out= together: this backend's
    # kthvalue.values raises the matching RuntimeError, but something in
    # how OpInfo's error_inputs invokes the out= overload for this specific
    # sample never lets it propagate as a raise the harness observes --
    # confirmed the same call raises correctly called directly
    # (torch.kthvalue(x, k, out=(values, indices))). Every other kthvalue
    # error-input sample (dim out of range, empty reduction dim, k out of
    # range on an ordinary tensor) passes.
    "kthvalue": ("float32",),
    "linalg_cross": ("float32",),
    "linalg_diagonal": ("float32",),
    "linalg_lstsq": ("float32",),
    "linalg_lstsq_grad_oriented": ("float32",),
    "log_normal": ("float32",),
    "logcumsumexp": ("float32",),
    "masked_fill": ("float32",),
    "masked_scatter": ("float32",),
    "masked_select": ("float32",),
    "mean": ("float32",),
    "min_binary": ("float32",),
    "movedim": ("float32",),
    "multinomial": ("float32",),
    "narrow": ("float32",),
    "narrow_copy": ("float32",),
    "native_layer_norm": ("float32",),
    "neg": ("float32",),
    "nn_functional_adaptive_avg_pool1d": ("float32",),
    "nn_functional_adaptive_avg_pool2d": ("float32",),
    "nn_functional_adaptive_avg_pool3d": ("float32",),
    "nn_functional_adaptive_max_pool1d": ("float32",),
    "nn_functional_adaptive_max_pool2d": ("float32",),
    "nn_functional_adaptive_max_pool3d": ("float32",),
    "nn_functional_conv1d": ("float32",),
    "nn_functional_conv2d": ("float32",),
    "nn_functional_conv3d": ("float32",),
    "nn_functional_embedding": ("float32",),
    "nn_functional_gaussian_nll_loss": ("float32",),
    "nn_functional_gelu": ("float32",),
    "nn_functional_group_norm": ("float32",),
    "nn_functional_hardtanh": ("float32",),
    "nn_functional_hinge_embedding_loss": ("float32",),
    "nn_functional_huber_loss": ("float32",),
    "nn_functional_l1_loss": ("float32",),
    "nn_functional_margin_ranking_loss": ("float32",),
    "nn_functional_max_pool1d": ("float32",),
    "nn_functional_max_pool2d": ("float32",),
    "nn_functional_max_pool3d": ("float32",),
    "nn_functional_max_unpool1d": ("float32",),
    "nn_functional_max_unpool2d": ("float32",),
    "nn_functional_max_unpool3d": ("float32",),
    "nn_functional_multi_margin_loss": ("float32",),
    "nn_functional_multilabel_margin_loss": ("float32",),
    "nn_functional_poisson_nll_loss": ("float32",),
    "nn_functional_prelu": ("float32",),
    "nn_functional_rms_norm": ("float32",),
    "nn_functional_rrelu": ("float32",),
    "nn_functional_soft_margin_loss": ("float32",),
    "nn_functional_softshrink": ("float32",),
    "nn_functional_triplet_margin_loss": ("float32",),
    "nn_functional_triplet_margin_with_distance_loss": ("float32",),
    "ormqr": ("float32",),
    "reshape": ("float32",),
    "reshape_as": ("float32",),
    "roll": ("float32",),
    "rot90": ("float32",),
    "scatter": ("float32",),
    "scatter_add": ("float32",),
    "sum_to_size": ("float32",),
    "t": ("float32",),
    "t_copy": ("float32",),
    "take": ("float32",),
    "trace": ("float32",),
    "tril": ("float32",),
    "triu": ("float32",),
    "unbind": ("float32",),
    "unbind_copy": ("float32",),
    "uniform": ("float32",),
    "vdot": ("float32",),
    "view": ("float32",),
    "view_as": ("float32",),
    "view_copy": ("float32",),
    "vsplit": ("float32",),
    "vstack": ("float32",),
    "where": ("float32",),
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
    "cpu": {
        "test_matches_cpu": {
            "__rpow__": ("float32", "int64"),
            "bmm": ("float32", "int64"),
            "floor_divide": ("bfloat16",),
            "log_softmax": ("float32", "bfloat16", "float16"),
            "masked_log_softmax": ("float32", "bfloat16", "float16"),
            "native_dropout_backward": (
                "float32",
                "bfloat16",
                "float16",
                "int64",
                "bool",
            ),
            "nn_functional_batch_norm": ("float32", "bfloat16", "float16"),
            "nn_functional_conv2d": ("bfloat16", "float16", "int64"),
            "nn_functional_instance_norm": ("float32", "bfloat16", "float16"),
            "pow": ("float32", "int64"),
        },
        "test_errors_match": {
            "_chunk_cat": (),
            "diag_embed": (),
            "diagonal": (),
            "diagonal_copy": (),
            "diff": (),
            "dsplit": (),
            "fft_fft": (),
            "fft_fft2": (),
            "fft_fftn": (),
            "fft_hfft2": (),
            "fft_hfftn": (),
            "fft_ifft": (),
            "fft_ifft2": (),
            "fft_ifftn": (),
            "fft_ihfft": (),
            "fft_ihfft2": (),
            "fft_ihfftn": (),
            "fft_irfft2": (),
            "fft_irfftn": (),
            "fft_rfft": (),
            "fft_rfft2": (),
            "fft_rfftn": (),
            "fliplr": (),
            "flipud": (),
            "hsplit": (),
            "isclose": (),
            "item": (),
            "linalg_cross": (),
            "linalg_diagonal": (),
            "movedim": (),
            "narrow_copy": (),
            "nn_functional_adaptive_avg_pool1d": (),
            "nn_functional_adaptive_avg_pool2d": (),
            "nn_functional_adaptive_avg_pool3d": (),
            "nn_functional_adaptive_max_pool1d": (),
            "nn_functional_group_norm": (),
            "nn_functional_hardtanh": (),
            "nn_functional_hinge_embedding_loss": (),
            "nn_functional_l1_loss": (),
            "nn_functional_margin_ranking_loss": (),
            "nn_functional_prelu": (),
            "nn_functional_rms_norm": (),
            "nn_functional_rrelu": (),
            "nn_functional_softshrink": (),
            "rot90": (),
            "scatter_add": (),
            "sum_to_size": (),
            "uniform": (),
            "view_copy": (),
            "vsplit": (),
        },
    }
}


def dtype_token(dtype: torch.dtype) -> str:
    """The dtype as it appears in a test node id: `torch.float16` -> "float16"."""
    return str(dtype).removeprefix("torch.")


@functools.cache
def accelerator_key() -> str:
    """Architecture name of the accelerator the mojo device runs kernels on.

    "sm_90a", "amdgpu:gfx942", ... -- the identity the kernels themselves
    branch on (`tensor._device.architecture_name`), so a table keyed by it
    splits exactly where support can differ.  "cpu" when the machine has no
    accelerator at all.

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
