"""Conformance suite: PyTorch's own OpInfo database, run against the mojo device.

Kept out of `tests/` deliberately.  `tests/` is the suite a contributor runs
per change; this one walks ~700 operators with PyTorch's full sample-input and
error-input corpus, so it is slow by construction and is meant to be run
deliberately, the way `benchmarks/` is.

The harness is PyTorch's, not ours: `torch/testing/_internal` ships inside the
wheel and `common_device_type.py` already knows how to target an out-of-tree
backend through `PrivateUse1TestBase`, which reads
`torch._C._get_privateuse1_backend_name()` — "mojo" for us.  Selecting it needs
`PYTORCH_TESTING_DEVICE_FOR_CUSTOM=privateuse1` set BEFORE
`common_device_type` is imported, which is why it happens here rather than in
the test module.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_TESTING_DEVICE_FOR_CUSTOM", "privateuse1")
# The internal harness reads this to decide whether it may leave the process
# in a dirty state; we run under pytest, not its own runner.
os.environ.setdefault("PYTORCH_TEST_WITH_SLOW", "0")

import pytest  # noqa: E402
import torch  # noqa: E402

from torch_mojo_backend import register_mojo_devices  # noqa: E402

register_mojo_devices()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--op",
        action="store",
        default=None,
        help="Only run OpInfo entries whose name contains this substring.",
    )
    parser.addoption(
        "--reference-inputs",
        action="store_true",
        default=False,
        help=(
            "Use OpInfo's reference_inputs_func (a superset of sample inputs, "
            "with the awkward shapes and layouts) instead of sample_inputs_func."
        ),
    )


@pytest.fixture(scope="session")
def mojo_available() -> bool:
    return bool(getattr(torch, "mojo", None) and torch.mojo.is_available())
