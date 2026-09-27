import struct
from typing import Literal

import torch
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.torch.torch import torch_dtype_to_max
from max.graph import Dim, TensorType

from torch_mojo_backend.torch_compile_backend import compiler
from torch_mojo_backend.types import MaxTensor, Scalar


def _scalar_to_tensor(input: MaxTensor, other: Scalar) -> MaxTensor:
    # `Scalar` also covers a symbolic Dim for ops that legitimately take one;
    # the bitwise ops that call this never do (ATen's Scalar there is a
    # genuine number), and F.constant only accepts a real number.
    if isinstance(other, Dim):
        raise TypeError(f"bitwise scalar ops expect a number, got a Dim: {other!r}")
    return F.broadcast_to(
        F.constant(other, dtype=input.dtype, device=input.device), input.shape
    )


# The custom ops whose Mojo math is a float32 port: `is_scalar_special` in
# tmb/kernels/common/unary_math.mojo (scalar ports of the CUDA routines,
# evaluated in float32 whatever the input dtype) and the pointwise kinds
# whose eager op declines float64 (`f64_ok=False` in tmb/ops/pointwise.mojo).
# A float64 graph must not reach them: its values would be float32 ones.
# tests/test_float32_math_kinds.py keeps this list in sync with the Mojo.
FLOAT32_MATH_OPS = frozenset(
    {
        "elementwise_airy_ai",
        "elementwise_asin",
        "elementwise_atan",
        "elementwise_bessel_j0",
        "elementwise_bessel_j1",
        "elementwise_bessel_y0",
        "elementwise_bessel_y1",
        "elementwise_digamma",
        "elementwise_entr",
        "elementwise_erfc",
        "elementwise_erfcx",
        "elementwise_erfinv",
        "elementwise_exp2",
        "elementwise_expm1",
        "elementwise_i0",
        "elementwise_i0e",
        "elementwise_i1",
        "elementwise_i1e",
        "elementwise_lgamma",
        "elementwise_log10",
        "elementwise_log_ndtr",
        "elementwise_logit",
        "elementwise_modified_bessel_i0",
        "elementwise_modified_bessel_i1",
        "elementwise_modified_bessel_k0",
        "elementwise_modified_bessel_k1",
        "elementwise_ndtri",
        "elementwise_scaled_modified_bessel_k0",
        "elementwise_scaled_modified_bessel_k1",
        "elementwise_sinc",
        "elementwise_spherical_bessel_j0",
        "elementwise_trigamma",
        "pointwise_igamma",
        "pointwise_igammac",
    }
)


def _refuse_float32_math(name: str, input: MaxTensor):
    """Refuse a float64 computation of a float32-only custom op (an integer
    input counts when the default dtype, which it promotes to, is float64),
    rather than return float32-accurate values in a float64 tensor."""
    if name not in FLOAT32_MATH_OPS:
        return
    promoted_f64 = (
        not input.dtype.is_float() and torch.get_default_dtype() == torch.float64
    )
    if input.dtype == DType.float64 or promoted_f64:
        raise NotImplementedError(
            f"{name} computes in float32 on the mojo backend: float64 inputs "
            "are not supported (they would silently lose precision)"
        )


def _same_type_binary(name: str, input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """An ElementwiseBinaryOp over two operands of one dtype and shape."""
    _refuse_float32_math(name, input)
    return F.custom(
        name=name,
        device=input.device,
        values=[input, other],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=compiler.kernel_extension_paths(),
    )[0]


def bitwise_and(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_and operation.
    """
    return _same_type_binary("bitwise_and", input, other)


def bitwise_and_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_and_scalar operation.
    """
    return bitwise_and(input, _scalar_to_tensor(input, other))


def bitwise_not(input: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_not operation.
    """

    return F.custom(
        name="bitwise_not",
        device=input.device,
        values=[input],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=compiler.kernel_extension_paths(),
    )[0]


def bitwise_or(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_or operation.
    """
    return _same_type_binary("bitwise_or", input, other)


def bitwise_or_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_or_scalar operation.
    """
    return bitwise_or(input, _scalar_to_tensor(input, other))


def bitwise_xor(input: MaxTensor, other: MaxTensor) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_xor operation.
    """
    return _same_type_binary("bitwise_xor", input, other)


def bitwise_xor_scalar(input: MaxTensor, other: Scalar) -> MaxTensor:
    """
    Custom Mojo kernel for bitwise_xor_scalar operation.
    """
    return bitwise_xor(input, _scalar_to_tensor(input, other))


def div(
    input: MaxTensor, other: MaxTensor, kind: Literal["true", "floor", "trunc"]
) -> MaxTensor:
    """torch.div's three modes (`tmb/kernels/common/div_math.mojo`). The
    operands share a dtype and a shape: promotion, broadcasting and any float32
    widening happen before the call."""
    return _same_type_binary(f"div_{kind}", input, other)


def elementwise(
    input: MaxTensor,
    kind: Literal[
        "abs",
        "acos",
        "acosh",
        "angle",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "ceil",
        "cos",
        "cosh",
        "erf",
        "erfc",
        "erfinv",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "gelu_none",
        "gelu_tanh",
        "isinf",
        "isnan",
        "logical_not",
        "log",
        "log10",
        "log1p",
        "log2",
        "neg",
        "reciprocal",
        "relu",
        "rsqrt",
        "sigmoid",
        "sign",
        "signbit",
        "silu",
        "sin",
        "sinc",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    ],
) -> MaxTensor:
    """Call shared unary math through MAX's fusible Mojo registrations."""
    _refuse_float32_math(f"elementwise_{kind}", input)
    if (
        kind
        not in {
            "abs",
            "ceil",
            "floor",
            "gelu_none",
            "gelu_tanh",
            "isinf",
            "isnan",
            "logical_not",
            "neg",
            "relu",
            "sign",
            "signbit",
            "silu",
        }
        and not input.dtype.is_float()
    ):
        # ATen unary_float_op promotes integer and bool inputs to the default
        # floating dtype, whereas ElementwiseUnaryOp preserves its input dtype.
        input = F.cast(input, dtype=torch_dtype_to_max(torch.get_default_dtype()))
    output_dtype = (
        DType.bool
        if kind in {"isinf", "isnan", "logical_not", "signbit"}
        else input.dtype
    )
    return F.custom(
        name=f"elementwise_{kind}",
        device=input.device,
        values=[input],
        out_types=[
            TensorType(dtype=output_dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=compiler.kernel_extension_paths(),
    )[0]


def pointwise_binary(
    input: MaxTensor,
    other: MaxTensor,
    kind: Literal[
        "atan2",
        "copysign",
        "fmax",
        "fmin",
        "fmod",
        "gcd",
        "heaviside",
        "hypot",
        "lcm",
        "logaddexp",
        "logaddexp2",
        "lshift",
        "nextafter",
        "rshift",
        "xlog1py",
        "xlogy",
    ],
) -> MaxTensor:
    """The mojo device's pointwise math (`tmb/kernels/common/pointwise_math`)
    as a fusible binary custom op. The operands share a dtype and a shape:
    promotion and broadcasting happen before the call."""
    return _same_type_binary(f"pointwise_{kind}", input, other)


def gelu_backward(
    grad_output: MaxTensor, input: MaxTensor, *, approximate: str = "none"
) -> MaxTensor:
    """
    Custom Mojo kernel for gelu_backward operation.
    """
    kernel_name = "gelu_backward_tanh" if approximate == "tanh" else "gelu_backward"
    return F.custom(
        name=kernel_name,
        device=input.device,
        values=[grad_output, input],
        out_types=[
            TensorType(dtype=input.dtype, shape=input.shape, device=input.device)
        ],
        custom_extensions=compiler.kernel_extension_paths(),
    )[0]


# --- The eager kernels as graph custom ops (tmb/graph/gemm.mojo, nn.mojo) ----
#
# `torch.compile(backend=mojo_backend)` would otherwise run MAX's own matmul /
# softmax / layer-norm / embedding kernels where the same model in eager mode
# runs this repository's. These wrappers call the eager routes through the
# `@compiler.register` structs of `tmb/graph/`, so both modes run one
# set of kernels with one set of numerics. Every operand is a dense row-major
# 2-D (or 3-D, for bmm) tensor: the aten_functions callers flatten leading
# dimensions and see through transposes; MAX materializes anything else.


def native_gemm(
    a: MaxTensor, b: MaxTensor, bias: MaxTensor | None, *, transpose_b: bool, tf32: bool
) -> MaxTensor:
    """`a @ op(b) [+ bias]` through the eager GEMM ladder (tmb/ops/matmul.mojo's
    `_mm_route` / `_addmm_route`): `a` is (m, k); `b` is (k, n), or (n, k)
    with `transpose_b` -- the stored weight of a Linear, read as `W.T` for
    free; `bias` is one row of n. `tf32` is torch's
    `get_float32_matmul_precision() != "highest"`, the only knob that picks
    the TF32 tensor-core bridge for float32 on an H100."""
    n = b.shape[0] if transpose_b else b.shape[1]
    out_type = TensorType(dtype=a.dtype, shape=[a.shape[0], n], device=a.device)
    return F.custom(
        name="native_gemm" if bias is None else "native_gemm_bias",
        device=a.device,
        values=[a, b] if bias is None else [a, b, bias],
        out_types=[out_type],
        parameters={"transpose_b": transpose_b, "tf32": tf32},
        custom_extensions=compiler.kernel_extension_paths(),
    )[0]


def native_bmm(
    a: MaxTensor, b: MaxTensor, *, transpose_b: bool, tf32: bool
) -> MaxTensor:
    """`a[i] @ op(b[i])` over a dense batch (tmb/ops/matmul.mojo's `_bmm_route`):
    `a` is (batch, m, k); `b` is (batch, k, n), or (batch, n, k) with
    `transpose_b`."""
    n = b.shape[1] if transpose_b else b.shape[2]
    out_type = TensorType(
        dtype=a.dtype, shape=[a.shape[0], a.shape[1], n], device=a.device
    )
    return F.custom(
        name="native_bmm",
        device=a.device,
        values=[a, b],
        out_types=[out_type],
        parameters={"transpose_b": transpose_b, "tf32": tf32},
        custom_extensions=compiler.kernel_extension_paths(),
    )[0]


def native_softmax_rows(x: MaxTensor) -> MaxTensor:
    """Softmax over the trailing dim of a (rows, cols) operand: the eager
    `SoftmaxSpec` kernel (tmb/kernels/nn's `_softmax_rows`)."""
    return F.custom(
        name="native_softmax_rows",
        device=x.device,
        values=[x],
        out_types=[TensorType(dtype=x.dtype, shape=x.shape, device=x.device)],
        custom_extensions=compiler.kernel_extension_paths(),
    )[0]


def native_layer_norm(
    x: MaxTensor, weight: MaxTensor, bias: MaxTensor, eps: float
) -> tuple[MaxTensor, MaxTensor, MaxTensor]:
    """Layer norm over the trailing dim of a (rows, cols) operand with a
    (cols,) weight and bias: the eager `LayerNormForward` kernel. Returns
    `(out, mean, rstd)`, the two statistics float32 of shape (rows,), as
    ATen's accelerator kernel does whatever the input dtype."""
    rows = x.shape[0]
    stats = TensorType(dtype=DType.float32, shape=[rows], device=x.device)
    out, mean, rstd = F.custom(
        name="native_layer_norm",
        device=x.device,
        values=[x, weight, bias],
        out_types=[
            TensorType(dtype=x.dtype, shape=x.shape, device=x.device),
            stats,
            stats,
        ],
        # ATen's `float eps` travels as the bit pattern of its float64: a
        # custom-op parameter is a bool, an int, a str or a dtype.
        parameters={"eps_bits": struct.unpack("<q", struct.pack("<d", eps))[0]},
        custom_extensions=compiler.kernel_extension_paths(),
    )
    return out, mean, rstd


def native_embedding(weight: MaxTensor, indices: MaxTensor) -> MaxTensor:
    """One row of the (num_rows, row_len) table per index of a flat int32 /
    int64 `indices`: the eager `Gather0` kernel (tmb/kernels/nn's `_gather0`)."""
    out_type = TensorType(
        dtype=weight.dtype,
        shape=[indices.shape[0], weight.shape[1]],
        device=weight.device,
    )
    return F.custom(
        name="native_embedding",
        device=weight.device,
        values=[weight, indices],
        out_types=[out_type],
        custom_extensions=compiler.kernel_extension_paths(),
    )[0]
