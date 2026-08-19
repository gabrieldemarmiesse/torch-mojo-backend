"""Regression test for scripts/compare_kernel_asm.py's collect() multiset fix.

Pure Python, no GPU and no `mojo build` -- synthesizes the `.ptx` sidecar
files a real cross-compile would leave in `--work-dir` and exercises
`collect()` plus the multiset-compare semantics `main()` builds on top of it
directly, in isolation.

The bug: a hash-stripped kernel name is not unique within one build -- several
comptime instantiations of one generically-named function (this repo's bmm
kernel under its `col_a`/`kmaj_b` layout parameters is a real example) can
reduce to the same stripped name. The old `collect()` returned a plain
`dict[str, str]`, so `kernels[stripped_name] = ...` silently kept only the
LAST sidecar `sorted(out_dir.iterdir())` produced for that name and dropped
the rest -- and "before"/"after" need not have kept the same one, since which
survives depends on mangling-hash sort order, not on anything about the code.
Comparing survivor-vs-survivor then reports unrelated instantiations as
"changed" even when the true set of kernel bodies under that name is
identical on both sides.
"""

from __future__ import annotations

from pathlib import Path

from scripts.compare_kernel_asm import HASH_RE, collect


def _write_ptx(directory: Path, stem: str, hexhash: str, body: str) -> None:
    (directory / f"{stem}_{hexhash}.ptx").write_text(body)


def test_collect_keeps_every_instantiation_under_one_stripped_name(
    tmp_path: Path,
) -> None:
    """Three distinct kernel bodies sharing a stripped name must all survive."""
    _write_ptx(tmp_path, "mod_bmm_nt_batched_persistent", "aaaaaaaa", "BODY_A")
    _write_ptx(tmp_path, "mod_bmm_nt_batched_persistent", "bbbbbbbb", "BODY_B")
    _write_ptx(tmp_path, "mod_bmm_nt_batched_persistent", "cccccccc", "BODY_C")

    kernels = collect(tmp_path)

    assert list(kernels.keys()) == ["mod_bmm_nt_batched_persistent"]
    bodies = kernels["mod_bmm_nt_batched_persistent"]
    assert sorted(bodies) == ["BODY_A", "BODY_B", "BODY_C"]


def test_collect_masks_the_hash_inside_the_body_too(tmp_path: Path) -> None:
    _write_ptx(tmp_path, "mod_kernel", "12345678", "call @mod_kernel_12345678")
    body = collect(tmp_path)["mod_kernel"][0]
    assert "12345678" not in body
    assert body == "call @mod_kernel_HASH"


def test_multiset_compare_is_order_independent(tmp_path: Path) -> None:
    """Same set of bodies under a name, different on-disk sort order: equal.

    This is the actual property the fix restores. "before" discovers its two
    instantiations in one hash order, "after" in the reverse order (a
    completely unrelated, cosmetic mangling-hash change is enough to do this
    in a real build) -- the SET of kernel bodies for that name is unchanged
    either way, and a multiset compare (sort before comparing) must say so.
    """
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    # "before": alphabetically-first hash carries BODY_ONE, second BODY_TWO.
    _write_ptx(before_dir, "mod_gemm_persistent", "10000000", "BODY_ONE")
    _write_ptx(before_dir, "mod_gemm_persistent", "20000000", "BODY_TWO")
    # "after": same two bodies, opposite hash-to-body assignment -- as if
    # only the mangling hash moved, which this whole script exists to treat
    # as a non-change.
    _write_ptx(after_dir, "mod_gemm_persistent", "10000000", "BODY_TWO")
    _write_ptx(after_dir, "mod_gemm_persistent", "20000000", "BODY_ONE")

    before = collect(before_dir)
    after = collect(after_dir)

    assert sorted(before["mod_gemm_persistent"]) == sorted(after["mod_gemm_persistent"])


def test_naive_single_value_dict_would_falsely_report_a_change(tmp_path: Path) -> None:
    """RED/GREEN: reproduce the exact bug the multiset return type fixes.

    Same two `before`/`after` directories as the order-independence test
    above (a genuinely UNCHANGED pair of kernel bodies, only the mangling
    hash moved). The naive collection this test inlines -- `dict[str, str]`,
    last write wins, exactly what `collect()` used to do -- picks whichever
    body `sorted(out_dir.iterdir())` visits last for that name. That pick is
    independent on each side, so it can (and here, does) differ even though
    nothing about the kernel changed: this is the false "changed" report
    `collect()`'s list-per-name return type exists to prevent.
    """
    before_dir, after_dir = tmp_path / "before", tmp_path / "after"
    before_dir.mkdir()
    after_dir.mkdir()
    _write_ptx(before_dir, "mod_gemm_persistent", "10000000", "BODY_ONE")
    _write_ptx(before_dir, "mod_gemm_persistent", "20000000", "BODY_TWO")
    _write_ptx(after_dir, "mod_gemm_persistent", "10000000", "BODY_TWO")
    _write_ptx(after_dir, "mod_gemm_persistent", "20000000", "BODY_ONE")

    def naive_collect(out_dir: Path) -> dict[str, str]:
        # This is collect()'s pre-fix body, inlined so the test does not
        # depend on the buggy implementation still existing in the tree.
        kernels: dict[str, str] = {}
        for path in sorted(out_dir.iterdir()):
            if path.suffix == ".ptx":
                kernels[HASH_RE.sub("", path.stem)] = HASH_RE.sub(
                    "_HASH", path.read_text()
                )
        return kernels

    naive_before = naive_collect(before_dir)
    naive_after = naive_collect(after_dir)

    # The bug: two directories holding an identical SET of kernel bodies
    # compare unequal under the naive single-value dict.
    assert naive_before["mod_gemm_persistent"] != naive_after["mod_gemm_persistent"]
    # The fix: the real collect() correctly calls this unchanged.
    before = collect(before_dir)
    after = collect(after_dir)
    assert sorted(before["mod_gemm_persistent"]) == sorted(after["mod_gemm_persistent"])
