"""Probe for tests/native/test_loader.py: the build key follows the sources
a build actually compiles, not the per-process memo.

argv[1] is a private copy of torch_mojo_backend/mojo. The probe memoizes the
`scan` family's key, edits a source of its closure, then asks for the
snapshot a build would compile: its hash must be new, its copy must hold the
edit, and the memo must now be that hash."""
from std.os import listdir
from std.pathlib import Path
from std.sys import argv
from std.testing import assert_equal, assert_not_equal, assert_true

from tmb.backend.loader import Loader


def main() raises:
    var root = String(argv()[1])
    var loader = Loader(root, root + "/cache", "mojo", "probe", False)
    var before = loader.source_hash("scan")
    assert_equal(loader.source_hash("scan"), before)  # memoized
    var edited = root + "/tmb/kernels/scan/entry.mojo"
    Path(edited).write_text(
        Path(edited).read_text() + "\n# edited mid-process\n"
    )
    # The memo is stale on purpose (the hot path never re-reads sources)...
    assert_equal(loader.source_hash("scan"), before)
    # ...but a build snapshots what it compiles and keys it by that.
    var snap = loader.snapshot("scan")
    assert_not_equal(snap[1], before)
    assert_true(
        Path(snap[0] + "/tmb/kernels/scan/entry.mojo")
        .read_text()
        .endswith("# edited mid-process\n")
    )
    assert_equal(loader.source_hash("scan"), snap[1])
    # A file the closure imports is in the snapshot too.
    assert_true(len(listdir(snap[0] + "/tmb/kernels/common")) > 0)
    print("OK")
