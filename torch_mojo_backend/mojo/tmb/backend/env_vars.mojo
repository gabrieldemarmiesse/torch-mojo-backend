# Every environment variable the Mojo base library reads, in one place.
#
# One constant per variable, named exactly like the variable, so a `getenv`
# call site still reads as the name a user would export and a grep for that
# name finds the definition and every use at once. Nothing else in this
# library spells an environment variable as a literal.
#
# `torch_mojo_backend/env_vars.py` is the project-wide union -- these, the
# mojoccl ones (`tmb/ccl/env_vars.mojo`) and the Python ones --
# and is what `register_mojo_devices()` checks the user's environment
# against. The three lists build separately and cannot import one another;
# `tests/test_env_vars_are_registered.py` scans the sources and fails if one
# of them grows a name the Python union does not carry.

# `1` fails every Mojo build on a compiler warning. Off by default -- a user
# on a newer toolchain that warns about something new must not lose their
# device over it -- and on under pytest.
comptime TORCH_MOJO_BACKEND_WERROR = "TORCH_MOJO_BACKEND_WERROR"

# Test hook: the device-to-device copy route to force ("host", "trace", ...)
# instead of letting the backend pick.
comptime TORCH_MOJO_BACKEND_TEST_PEER_COPY = "TORCH_MOJO_BACKEND_TEST_PEER_COPY"

# Test hook: a file descriptor the peer-copy path blocks on, so a test can
# hold a copy open and observe the state around it.
comptime TORCH_MOJO_BACKEND_TEST_PEER_GATE_FD = (
    "TORCH_MOJO_BACKEND_TEST_PEER_GATE_FD"
)

# Not ours: the OS scratch directory, where the loader stages the
# intermediate files of a build before installing them into the cache.
comptime TMPDIR = "TMPDIR"

# The Mojo import path the toolchain reads (torch_mojo_backend/
# _mojo_import_path.py extends it with the live source root); a snapshot
# build drops that root from it (loader.mojo `_build`).
comptime MODULAR_MOJO_MAX_IMPORT_PATH = "MODULAR_MOJO_MAX_IMPORT_PATH"
