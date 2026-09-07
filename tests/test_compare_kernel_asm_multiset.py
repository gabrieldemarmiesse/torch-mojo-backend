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

import pytest

from scripts.compare_kernel_asm import (
    HASH_RE,
    KernelDelta,
    collect,
    diff_names,
    print_kernel_delta,
    residual_bodies,
)


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


# ---------------------------------------------------------------------------
# Aggregation/reporting path: residual_bodies(), diff_names(), and
# print_kernel_delta(). collect() returning every instantiation under a name
# (tested above) is necessary but not sufficient -- main() still has to turn
# "before"/"after" instantiation lists into correct changed/added/removed
# counts and a correct --show-diff pairing. A positional zip of the two
# SORTED lists (an earlier, wrong version of this) pairs whichever bodies
# land at the same sorted index once a multiplicity differs, which is wrong
# whenever a duplicate-name group genuinely changes rather than merely being
# unchanged-but-reordered. These tests cover that path directly.
# ---------------------------------------------------------------------------


def test_residual_bodies_one_replacement_inside_a_group() -> None:
    """before=[A,B,C], after=[B,C,Z]: B and C cancel, only A/Z are residual.

    A positional zip of the sorted lists would pair (A,B) (B,C) (C,Z) and
    report three changes for what is really one replacement.
    """
    residual_before, residual_after = residual_bodies(["A", "B", "C"], ["B", "C", "Z"])
    assert residual_before == ["A"]
    assert residual_after == ["Z"]


def test_residual_bodies_multiplicity_growth() -> None:
    """before=[A,B], after=[A,B,C]: nothing changed, C is purely new."""
    residual_before, residual_after = residual_bodies(["A", "B"], ["A", "B", "C"])
    assert residual_before == []
    assert residual_after == ["C"]


def test_residual_bodies_multiplicity_shrink() -> None:
    """before=[A,B,C], after=[A,B]: nothing changed, C is purely gone."""
    residual_before, residual_after = residual_bodies(["A", "B", "C"], ["A", "B"])
    assert residual_before == ["C"]
    assert residual_after == []


def test_residual_bodies_duplicate_bodies_cancel_by_count() -> None:
    """Two copies of A on each side fully cancel; only the extra copy of A
    on the "after" side, and B's replacement by C, are residual."""
    residual_before, residual_after = residual_bodies(
        ["A", "A", "B"], ["A", "A", "A", "C"]
    )
    assert residual_before == ["B"]
    assert residual_after == ["A", "C"]


def test_diff_names_reports_added_not_changed_for_a_grown_group() -> None:
    """Codex's exact regression case: before=[A,B], after=[A,B,C] must report
    0 changed / +1 new for this name, not 3 changed / +0 new."""
    before = {"mod_kernel": ["A", "B"]}
    after = {"mod_kernel": ["A", "B", "C"]}
    (delta,) = diff_names(before, after)
    assert delta == KernelDelta(
        "mod_kernel",
        changed=0,
        added=1,
        removed=0,
        residual_before=[],
        residual_after=["C"],
    )


def test_diff_names_reports_one_change_for_one_replacement() -> None:
    before = {"mod_kernel": ["A", "B", "C"]}
    after = {"mod_kernel": ["B", "C", "Z"]}
    (delta,) = diff_names(before, after)
    assert delta.changed == 1
    assert delta.added == 0
    assert delta.removed == 0
    assert delta.residual_before == ["A"]
    assert delta.residual_after == ["Z"]


def test_diff_names_reports_removed_for_a_shrunk_group() -> None:
    before = {"mod_kernel": ["A", "B", "C"]}
    after = {"mod_kernel": ["A", "B"]}
    (delta,) = diff_names(before, after)
    assert delta.changed == 0
    assert delta.added == 0
    assert delta.removed == 1


def test_diff_names_skips_a_name_whose_multiset_is_unchanged() -> None:
    """Reordered-only (mangling hash moved, bodies identical): no delta at
    all, matching the fix's original purpose (order-independence)."""
    before = {"mod_kernel": ["A", "B"]}
    after = {"mod_kernel": ["B", "A"]}
    assert diff_names(before, after) == []


def test_diff_names_ignores_names_present_on_only_one_side() -> None:
    """Whole-name add/remove is the caller's fast path (module-level added/
    removed in main()), not diff_names()'s job."""
    before = {"only_before": ["A"]}
    after = {"only_after": ["Z"]}
    assert diff_names(before, after) == []


def test_show_diff_pairs_residuals_not_shifted_sorted_positions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The --show-diff detail for a replacement must diff the actual
    replaced pair (A -> Z), not a positional artifact of sorting the whole
    group (which would show unrelated bodies B and C as "changed")."""
    before = {"mod_kernel": ["A", "B", "C"]}
    after = {"mod_kernel": ["B", "C", "Z"]}
    (delta,) = diff_names(before, after)

    print_kernel_delta("mod/variant/mod_kernel", delta)
    out = capsys.readouterr().out

    assert "before/mod/variant/mod_kernel[0]" in out
    assert "after/mod/variant/mod_kernel[0]" in out
    assert "-A" in out
    assert "+Z" in out
    # B and C canceled and must not appear as changed lines in the diff body.
    assert "-B" not in out
    assert "-C" not in out
    assert "+B" not in out
    assert "+C" not in out


def test_show_diff_header_reports_counts_for_a_grown_group(
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = {"mod_kernel": ["A", "B"]}
    after = {"mod_kernel": ["A", "B", "C"]}
    (delta,) = diff_names(before, after)

    print_kernel_delta("mod/variant/mod_kernel", delta)
    out = capsys.readouterr().out

    assert "0 replaced, 1 added, 0 removed" in out
    # Nothing to pair, so no unified-diff body -- just the header.
    assert "@@" not in out
