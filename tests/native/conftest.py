import pytest

from torch_mojo_backend import register_mojo_devices


@pytest.fixture(autouse=True, scope="session")
def _mojo_devices_registered():
    """Every native test runs on the registered mojo device (a test that
    only uses the `mojo_device` / `mojo_gpu` fixtures of tests/conftest.py
    gets it from there as well; the call is idempotent)."""
    register_mojo_devices()
