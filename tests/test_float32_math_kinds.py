"""The torch.compile custom ops whose Mojo math only exists in float32 refuse
float64 inputs instead of returning float32-accurate values in a float64
tensor (`custom_mojo_ops.FLOAT32_MATH_OPS`).

The list is duplicated from the Mojo sources, so this also checks it against
them: `is_scalar_special` in tmb/kernels/common/unary_math.mojo and the
pointwise kinds whose eager op declines float64 in tmb/ops/pointwise.mojo.
"""

import re
from pathlib import Path

import pytest
import torch

from torch_mojo_backend import custom_mojo_ops, mojo_backend
from torch_mojo_backend.aten_functions import MAPPING_TORCH_ATEN_TO_MOJO

MOJO = Path(custom_mojo_ops.__file__).parent / "mojo" / "tmb"


def _function_body(source: str, signature: str) -> str:
    return source.split(signature, 1)[1].split("\n\n\n", 1)[0]


def _scalar_special_kinds() -> set[str]:
    source = (MOJO / "kernels" / "common" / "unary_math.mojo").read_text()
    kinds = set()
    # `has_f64_port`'s kinds are listed apart from `is_scalar_special`'s,
    # which includes them.
    for signature in ("def is_scalar_special[", "def has_f64_port["):
        kinds |= set(
            re.findall(r'kind == "([a-z0-9_]+)"', _function_body(source, signature))
        )
    return kinds


def _pointwise_float32_kinds() -> set[str]:
    source = (MOJO / "ops" / "pointwise.mojo").read_text()
    kinds = set(
        re.findall(r'_pw_math\(\s*"([a-z0-9_]+)",\s*P_[A-Z_]+,\s*False', source)
    )
    # igamma / igammac compute float64 on NVIDIA devices only: the host
    # policy names them in `_pw_igamma_f64`, and elsewhere they stay float.
    body = _function_body(source, "def _pw_igamma_f64(")
    return kinds | set(re.findall(r'kind == "([a-z0-9_]+)"', body))


def test_float32_math_ops_match_the_mojo_sources():
    # The graph's polygamma (tmb/graph/elementwise.mojo) runs
    # `unary_math.elementwise_polygamma`, a float32 evaluation.
    expected = (
        {f"elementwise_{k}" for k in _scalar_special_kinds()}
        | {f"pointwise_{k}" for k in _pointwise_float32_kinds()}
        | {"polygamma"}
    )
    assert custom_mojo_ops.FLOAT32_MATH_OPS == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_float32_math_refuses_float64(dtype):
    """asin through the graph: float32 computes, float64 (and an integer
    input under a float64 default dtype) raises rather than lose precision."""
    if torch.ops.aten.asin not in MAPPING_TORCH_ATEN_TO_MOJO:
        pytest.skip("aten.asin has no torch.compile twin yet")
    x = torch.tensor([0.25, 0.5, 1.0]).to(dtype)
    compiled = torch.compile(torch.asin, backend=mojo_backend)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64 if dtype == torch.int64 else previous)
    try:
        if dtype == torch.float32:
            torch.testing.assert_close(compiled(x), torch.asin(x))
        else:
            with pytest.raises(Exception, match="float64 inputs are not supported"):
                compiled(x)
    finally:
        torch.set_default_dtype(previous)
        torch.compiler.reset()
