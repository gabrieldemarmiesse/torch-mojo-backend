"""`register_mojo_devices()` after a failed attempt: torch's own registrations
(rename, device module, generated methods) are not repeatable, everything
after them is, so a build that did not go through must leave the next call
able to finish the job instead of returning early as if it had."""

import os
import subprocess
import sys
from pathlib import Path

_WORKTREE = Path(__file__).resolve().parents[2]

_SCRIPT = """
import torch
from torch_mojo_backend import native
from torch_mojo_backend.mojo_device import register

real = native.register
def failing():
    raise RuntimeError("simulated build failure")
native.register = failing
try:
    register.register_mojo_devices()
except RuntimeError as exc:
    assert "simulated" in str(exc), exc
else:
    raise AssertionError("the failure did not propagate")
assert not register._registered
native.register = real
register.register_mojo_devices()
assert register._registered
print("retry ok", torch.mojo.device_count())
"""


def test_a_failed_registration_can_be_retried():
    env = {"PYTHONPATH": str(_WORKTREE)}
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "retry ok" in proc.stdout
