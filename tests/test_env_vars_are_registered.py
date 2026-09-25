"""Every environment variable this project reads must be listed in
``torch_mojo_backend/env_vars.py`` (see its module docstring).

That list is what ``register_mojo_devices()`` holds the user's environment up
against, so a variable missing from it costs the user the typo check on it:
they export a name with a letter wrong, nothing reads it, nothing complains,
and the default it was meant to override stays in force. Hence a test rather
than a convention.

Three checks, over every tracked ``.py`` / ``.mojo`` / ``.c`` / ``.cpp`` /
``.h`` file in the repository:

* any token spelled like one of ours -- ``TORCH_MOJO_BACKEND_*``,
  ``PYTORCH_MOJO_BACKEND_*``, ``MOJOCCL_*`` -- anywhere in a source file,
  literal or comment, is a name in our namespace and must be registered. This
  is the check that reaches the Mojo side: the two ``env_vars.mojo`` files
  build separately from Python and cannot import the table, so this is what
  keeps all three in step.
* any name handed to ``os.environ`` / ``getenv`` **inside the shipped
  package** must be registered too, as one of ours or as somebody else's
  (``ROCM_PATH``, ``CXX``, ...) that we happen to read.
* the reverse: a registered name that no longer reaches the environment
  anywhere in the repository is a knob the table promises and the code
  dropped.

Names reached through a constant -- ``getenv(TMPDIR)``,
``os.environ.get(_NCCL_LIB_ENV)`` -- are resolved through the module-level
string constants of every scanned file, so moving a name off the call site
does not move it out of the check. That is not a stylistic nicety: both
``env_vars.mojo`` files exist precisely to hold such constants.

Scope is deliberately asymmetric. Our own namespace is enforced everywhere,
because a user can export any of those names. Foreign names are enforced only
in the shipped package: the ad hoc knobs of a multinode probe script
(``BUCKETS``, ``STEPS``, ``N``) are not things a user sets on us, and
registering them would turn a user-facing table into a junk drawer. Shell
scripts and CI workflows are not scanned at all -- they set variables for our
processes rather than read any, and their own locals share the prefix.
"""

import os
import re
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest

from torch_mojo_backend import env_vars

REPO = Path(__file__).resolve().parent.parent
PACKAGE = REPO / "torch_mojo_backend"
# The table names every variable by construction, so it can neither offend the
# checks below nor stand in for a real use in the reverse one.
REGISTRY = PACKAGE / "env_vars.py"
# This file spells deliberate misspellings.
THIS_FILE = Path(__file__).resolve()
SCANNED_SUFFIXES = (".py", ".mojo", ".c", ".cpp", ".h")

# Longest prefix first: `TORCH_MOJO_BACKEND` would otherwise swallow the tail
# of a `PYTORCH_` name. The trailing `[A-Z0-9]` keeps a bare prefix constant
# (`"MOJOCCL_"`) from reading as a variable name.
OURS = re.compile(
    r"\b(?:PYTORCH_MOJO_BACKEND|TORCH_MOJO_BACKEND|MOJOCCL)_[A-Z0-9_]*[A-Z0-9]\b"
)

# A module-level `NAME = "VALUE"` (Python) or `comptime NAME = "VALUE"` (Mojo)
# whose value could be an environment variable name. Anything with a dot or a
# dash in it -- `FABRIC_SONAME = "libfabric.so.1"` -- is not one, and `\w+`
# leaves it out.
CONSTANT = re.compile(
    r"""^(?:comptime\s+)?(\w+)\s*(?::[^=\n]+)?=\s*["'](\w+)["']\s*$""", re.M
)

# A name handed to something that names a variable to the environment, quoted
# or through a constant. Reads and writes both count: `os.environ["CUDA_VISIBLE
# _DEVICES"] = ...` is as much a reason to register the name as reading it.
# `os.` is not required, so `from os import environ` and Mojo's and C's bare
# `getenv` are all caught.
_ARGUMENT = r"""(?:["'](\w+)["']|(\w+))"""
TOUCHES = tuple(
    re.compile(pattern)
    for pattern in (
        rf"""environ\.(?:get|pop|setdefault)\(\s*{_ARGUMENT}""",
        rf"""environ\[\s*{_ARGUMENT}\s*\]""",
        rf"""getenv\s*\(\s*{_ARGUMENT}""",
        r"""["'](\w+)["']\s+(?:not\s+)?in\s+\w*\.?environ\b""",
    )
)


def _tracked_sources(root: Path) -> list[Path]:
    """Every scannable file git knows about under `root`, so neither the venv
    nor an untracked scratch clone joins in."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", str(root)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    paths = [REPO / name for name in out.split("\0") if name]
    return sorted(
        p
        for p in paths
        if p.suffix in SCANNED_SUFFIXES and p.resolve() != THIS_FILE and p.is_file()
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _constants(paths: Iterable[Path]) -> dict[str, str]:
    """Every module-level string constant in `paths`, as one map.

    One map rather than one per file because that is how the code is written:
    `getenv(MOJOCCL_REGION_MB)` in tmb/ccl/init.mojo reads a constant declared in
    env_vars.mojo, and `os.environ.get(_NCCL_LIB_ENV)` reads one declared at
    the top of its own.
    """
    found = {}
    for path in paths:
        found.update(CONSTANT.findall(_read(path)))
    return found


def _names_in_namespace(path: Path) -> set[str]:
    return set(OURS.findall(_read(path)))


def _env_names_touched(path: Path, constants: Mapping[str, str]) -> set[str]:
    """Variable names `path` hands to the environment, constants resolved.

    An argument that is neither a literal nor a known constant -- a parameter,
    an element of a tuple -- cannot be resolved statically and is dropped.
    """
    text = _read(path)
    names = set()
    for pattern in TOUCHES:
        for match in pattern.finditer(text):
            literal = match.group(1) or ""
            symbol = (match.group(2) or "") if pattern.groups > 1 else ""
            if literal:
                names.add(literal)
            elif symbol in constants:
                names.add(constants[symbol])
    return names


def _offenders(
    found: Iterable[tuple[str, Path]], known: frozenset[str] | set[str]
) -> dict[str, list[str]]:
    offenders: dict[str, list[str]] = {}
    for name, path in found:
        if name not in known:
            offenders.setdefault(name, []).append(str(path.relative_to(REPO)))
    return offenders


def _fix_it(offenders: Mapping[str, list[str]], kind: str) -> str:
    lines = [
        f"{name} is {kind} but is not registered (used in "
        + ", ".join(sorted(set(where)))
        + ")"
        for name, where in sorted(offenders.items())
    ]
    return (
        "\n".join(lines)
        + "\n\nAdd each one to OWN_ENV_VARS (ours) or FOREIGN_ENV_VARS "
        "(torch's, MAX's, the vendor runtime's, the OS's) in "
        "torch_mojo_backend/env_vars.py, with a line saying what setting it "
        "does. That table is what register_mojo_devices() checks the user's "
        "environment against, so a name missing from it is a knob the user "
        "gets no typo warning for."
    )


def test_the_scanner_recognizes_the_registered_names():
    """A broken regex must fail loudly rather than pass everything."""
    sources = _tracked_sources(REPO)
    constants = _constants(sources)
    assert len(sources) > 100
    assert "TORCH_MOJO_BACKEND_VERBOSE" in _names_in_namespace(PACKAGE / "flags.py")
    assert "MOJOCCL_REGION_MB" in _names_in_namespace(
        PACKAGE / "mojo" / "tmb" / "ccl" / "env_vars.mojo"
    )
    assert "TORCH_MOJO_BACKEND_TESTING" in _env_names_touched(
        PACKAGE / "is_running_tests.py", constants
    )
    assert "ROCM_PATH" in _env_names_touched(
        PACKAGE / "mojo_device" / "hip_peer.py", constants
    )


def test_the_scanner_resolves_names_reached_through_a_constant():
    """The two env_vars.mojo files exist to hold such constants, so a scanner
    that only saw literals would stop seeing the Mojo side entirely."""
    constants = _constants(_tracked_sources(REPO))
    # Mojo, through `comptime TMPDIR = "TMPDIR"` in a different file.
    assert "TMPDIR" in _env_names_touched(
        PACKAGE / "mojo/tmb/backend/loader.mojo", constants
    )
    assert "TORCH_MOJO_BACKEND_TEST_PEER_COPY" in _env_names_touched(
        PACKAGE / "mojo/tmb/backend/device.mojo", constants
    )
    assert "MOJOCCL_REGION_MB" in _env_names_touched(
        PACKAGE / "mojo/tmb/ccl/init.mojo", constants
    )
    # Python, through a constant at the top of its own module.
    assert "TORCH_MOJO_BACKEND_NCCL_LIB" in _env_names_touched(
        PACKAGE / "distributed/nccl.py", constants
    )
    assert "MODULAR_NVPTX_COMPILER_PATH" in _env_names_touched(
        PACKAGE / "_ptxas.py", constants
    )


def test_every_name_in_our_namespace_is_registered():
    found = [
        (name, path)
        for path in _tracked_sources(REPO)
        for name in _names_in_namespace(path)
    ]
    offenders = _offenders(found, set(env_vars.OWN_ENV_VARS))
    assert not offenders, _fix_it(offenders, "spelled like one of ours")


def test_every_environment_name_in_the_package_is_registered():
    constants = _constants(_tracked_sources(REPO))
    found = [
        (name, path)
        for path in _tracked_sources(PACKAGE)
        if path != REGISTRY
        for name in _env_names_touched(path, constants)
    ]
    offenders = _offenders(found, env_vars.known_env_vars())
    assert not offenders, _fix_it(offenders, "named to the environment")


def test_no_registered_name_has_gone_stale():
    """A name nothing reaches the environment with any more is a knob the
    table promises and the code dropped.

    Liveness is the resolved name set, not a word search: a name that survives
    only in a docstring is exactly the dead entry this is looking for. A
    constant declared for the purpose counts as live -- that is what an
    `env_vars.mojo` declaration is -- so the check is about reachability, not
    about where the string happens to sit.
    """
    sources = [p for p in _tracked_sources(REPO) if p != REGISTRY]
    constants = _constants(sources)
    live = set(constants.values())
    for path in sources:
        live |= _env_names_touched(path, constants)
    stale = sorted(env_vars.known_env_vars() - live)
    assert not stale, (
        "registered in torch_mojo_backend/env_vars.py but nothing reaches the "
        "environment with it: " + ", ".join(stale) + ". Drop the entry, or "
        "spell the name where it is read."
    )


def test_a_misspelling_is_reported_with_a_suggestion():
    bad = "TORCH_MOJO_BACKEND_VERBOZE"
    assert env_vars.unknown_env_vars({bad: "1"}) == [
        (bad, "TORCH_MOJO_BACKEND_VERBOSE")
    ]
    with pytest.warns(env_vars.UnknownEnvVarWarning, match="Did you mean"):
        env_vars.warn_about_unknown_env_vars({bad: "1"})


def test_a_name_resembling_nothing_is_reported_without_one():
    unknown = env_vars.unknown_env_vars({"MOJOCCL_QQQQQQQQ": "1"})
    assert unknown == [("MOJOCCL_QQQQQQQQ", None)]
    with pytest.warns(env_vars.UnknownEnvVarWarning, match="no such environment"):
        env_vars.warn_about_unknown_env_vars({"MOJOCCL_QQQQQQQQ": "1"})


@pytest.mark.parametrize(
    "name",
    [
        "MOJOCCL_FUSED",
        "MOJOCCL_FUSED_BLOCKS",
        "MOJOCCL_FUSED_BIG_BLOCKS",
        "MOJOCCL_FUSED_BIG_MB",
        "MOJOCCL_IB_PROXY",
        "MOJOCCL_IB_PROXY_CPU",
        "MOJOCCL_IB_PROXY_IDLE_US",
        "MOJOCCL_NVLS_GRANULARITY",
        "MOJOCCL_NVLS_MIN_MB",
        "MOJOCCL_PIPE_SPLIT_UNIT",
        "MOJOCCL_FABRIC_HMEM",
        "MOJOCCL_FABRIC_FLUSH",
        "MOJOCCL_FABRIC_SETUP_RETRY_S",
        "MOJOCCL_SOCKET_DIR",
        "MOJOCCL_BUILD_DEFINES",
    ],
)
def test_removed_mojoccl_controls_are_unknown(name: str):
    """Code defaults must not leave silent, apparently supported overrides."""
    assert name not in env_vars.known_env_vars()
    assert list(dict(env_vars.unknown_env_vars({name: "1"}))) == [name]
    with pytest.warns(env_vars.UnknownEnvVarWarning, match=name + " is set"):
        env_vars.warn_about_unknown_env_vars({name: "1"})


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"TORCH_MOJO_BACKEND_VERBOSE": "1", "MOJOCCL_REGION_MB": "64"},
        # Not ours to diagnose: neither prefix, however misspelled.
        {"ROCM_PATH": "/opt/rocm", "PYTORCH_MOJO_BACKEND_VERBOZE": "1", "PATH": "/bin"},
    ],
)
def test_nothing_else_draws_a_warning(environment, recwarn):
    assert env_vars.unknown_env_vars(environment) == []
    env_vars.warn_about_unknown_env_vars(environment)
    assert [w for w in recwarn if issubclass(w.category, UserWarning)] == []


def test_the_real_environment_is_checked_at_registration():
    """The default argument reads `os.environ`, which is the whole point."""
    name = "TORCH_MOJO_BACKEND_NOT_A_REAL_KNOB"
    os.environ[name] = "1"
    try:
        assert name in dict(env_vars.unknown_env_vars())
    finally:
        del os.environ[name]
