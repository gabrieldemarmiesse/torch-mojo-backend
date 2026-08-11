"""The single git-tracked baseline file and its merge discipline.

The file is benchmarks/baselines.html, and it is BOTH the data and the
viewer for it: the ratios live in a

    <script type="application/json" id="baselines"> ... </script>

block at the end of the document, and the markup above them renders that
block as a collapsible tree.  Open it in a browser — off disk, no server,
no network, nothing to install — and the numbers are there.  One file to
open, to link, to attach to a PR, to hand to someone who does not have
the repo.  (github.com shows .html as source rather than rendering it, so
from GitHub either download the file or route it through a raw-HTML
viewer such as htmlpreview.github.io.)

Splitting the two would cost more than it saves: a JSON beside an HTML is
a pair that gets separated the moment anyone copies one of them, and the
viewer then needs a fetch, which needs a server, because file:// cannot
read a sibling file.  Everything in this module is therefore written to
touch ONLY the data block: `split_document` cuts the file into
(viewer, block, viewer) and every write splices a new block between two
byte-identical halves.  The viewer is hand-written source like any other
file in the repo; the block is machine-written and must not be
hand-edited.

Layout of the data block (format 5, the axis tree):

    {
      "format": 5,
      "configs": {
        "NVIDIA H100 PCIe | cuda | kineto-device-time@1395MHz | torch 2.11.0+cu130": {
          "ops": {
            "mm": {
              "dtypes": {
                "bf16": {
                  "shapes": {
                    "S1_4096x4096x4096": {
                      "layouts": {"NN": 5.158, "NT": 3.933, ...}
                    }
                  }
                }
              }
            }
          }
        }
      }
    }

The tree path is hardware -> aten op -> dtype -> shape -> layout -> ratio.
A leaf is addressed by a BenchKey (op, dtype, shape, layout); pytest
derives it from the test's own axes (see bench_lib/check.py), so the
failing test and its baseline entry still name the same kernel regime.

Op above dtype, not the other way round: work is scoped by op (an
engagement optimizes one kernel family across its dtypes), so that order
puts everything one change can move under a single branch, and reading
one op's dtypes side by side is what tells you whether a regression is
dtype-specific.

Why this shape:

* ONE file for every hardware configuration, so H100 / MI300X / Apple /
  CPU baselines are reviewed side by side.
* The leaf value is the RATIO ours/stock (device time), never an absolute
  time: ratios are hardware-normalized and survive clock drift.  ALL the
  context a ratio needs — which hardware, against which stock backend,
  timed how, on which torch build — lives in the config key.  The torch
  version is part of the KEY, not a field: the ratio's denominator IS
  stock PyTorch, so a torch upgrade changes the meaning of every ratio.
  Keying on it makes an upgraded box an unseen config ("never measured":
  pass and record), exactly like new hardware, instead of silently
  re-stamping numbers measured under the old build.  No timestamps, no
  commit stamps.
* MEASUREMENTS ONLY: the block stores one ratio per leaf and nothing
  derived from them.  The per-branch min/median/max are what you actually
  read this tree for, but they are not stored — the viewer around the
  block, and `report.py`, compute them from the leaves while parsing,
  which costs microseconds.  Storing them cost more than that: roughly
  four derived lines beside every measured one, all of which churn in the
  diff whenever any leaf below them moves, and any of which can go stale
  under a hand edit or a bad merge (a whole validate/repair path existed
  for exactly that).  Derived data belongs in the reader.
* At each level the children live under a NAMED container key ("ops" /
  "dtypes" / "shapes" / "layouts"), so a leaf is always a bare number inside
  "layouts", every other value is an object, and the two can never be
  confused.
* Serialization is canonical (sorted keys, indent 2, one entry per line).
  Because every writer re-emits the same canonical form and the block
  holds nothing derived, a partial update rewrites every untouched line
  byte-for-byte — only the measured leaves move — each hardware config
  occupies one contiguous sorted region, and two machines updating
  different configs touch disjoint lines that git merges cleanly.  This is
  why the data is embedded as pretty-printed JSON rather than minified
  into one line: a one-line blob would make every run a whole-file diff
  and every concurrent update a conflict.

Writes go through merge_write(): read the file fresh under an exclusive
flock, update ONLY the measured leaves, splice the block back between the
untouched halves of the document, write atomically (tmp + rename).  A run
that measured three relu cases can therefore never clobber the GEMM
numbers it did not run, nor leaves written by a concurrent run, nor a
viewer improvement someone committed meanwhile.

To read the numbers: open benchmarks/baselines.html in a browser for the
collapsible tree with per-branch min/median/max, or
`uv run python benchmarks/report.py --worst 20` for the terminal.  The
page also takes another baselines file as a query parameter —
`baselines.html?url=<url>` — so one machine's file can render another
machine's numbers (a raw file URL, a CI artifact, another branch) without
a checkout.

Superseded torch versions: after a fleet-wide torch upgrade the old
config block keeps protecting nothing (no machine runs under that key any
more).  Prune it manually in a reviewed commit — delete the old
"... | torch <old>" block once every machine of that hardware has
reseeded under the new version.  Keep both blocks only while machines of
the same hardware genuinely run different torch builds.

Known git-merge limitation (deliberate, documented rather than solved):
two machines each adding a BRAND-NEW block at the same tree level
concurrently can conflict, because both insert lines into the same gap of
a sorted object (updating existing, distinct blocks merges cleanly —
those edits touch disjoint regions).  JSON's no-trailing-comma rule makes
this insertion-point conflict unavoidable in a single file.  To resolve:
keep BOTH added blocks (union them, fixing the comma on the last block),
then re-canonicalize so the next writer diffs cleanly:

    uv run python -c "import sys; sys.path.insert(0, 'benchmarks'); \\
        from bench_lib import baselines; baselines.rebuild()"

rebuild() only ever rewrites the data block, so it is also safe to run
after a merge that touched the viewer.  It is the migration path for an
older block too: it drops the derived "aggregate" objects format 3
carried, transposes the dtype/op nesting formats 3 and 4 used, and
rewrites canonically.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import typing
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

FORMAT_VERSION = 5

BASELINES_PATH = Path(__file__).resolve().parent.parent / "baselines.html"

# The data block inside that file.  Everything outside it is the viewer and
# is never touched by a write; everything inside it is machine-written and
# must never be hand-edited.
BLOCK_OPEN = '<script type="application/json" id="baselines">'
BLOCK_CLOSE = "</script>"

# Format 3 stored a derived min/median/max object beside every container,
# under this key.  Later formats store measurements only; rebuild() drops them.
LEGACY_AGGREGATE_KEY = "aggregate"

# Used only when the file does not exist yet (a scratch path, or a deleted
# baselines.html): the data has to land somewhere, but this process has no
# business inventing the viewer.
MINIMAL_DOCUMENT = f"""<!doctype html>
<meta charset="utf-8">
<title>benchmark baselines (data only)</title>
<p>Baseline ratios without the explorer that renders them: this file was
created from scratch by bench_lib/baselines.py. Copy everything above the
data block of benchmarks/baselines.html into here to get the viewer back.</p>
{BLOCK_OPEN}
{{"configs": {{}}, "format": {FORMAT_VERSION}}}
{BLOCK_CLOSE}
"""


class BenchKey(typing.NamedTuple):
    """Address of one ratio leaf inside a hardware config's axis tree.

    Axis order follows the tree: op -> dtype -> shape -> layout, so
    str(key) is the tree path and every reader can print, match and sort
    on the same string.  The op token is the aten base name with the
    overload variant folded in ("mm", "add.Tensor"); ops without a layout
    notion use the fixed sentinel "contig" so every path has the same
    rank.

    Shape tokens name a regime rather than only its numbers:

    * A leading letter tags the whole shape and is SEPARATED from the
      dims by an underscore — "C_16777216" (contiguous vector),
      "S_4096x4096" (standard/square), "A_357x789" (awkward), "L_4x…"
      (list of 4), "P_2x…" (pieces), "R_1000x64" (row table), and the
      numbered GEMM regimes "S7_768x50304x49152".
    * When the letters instead label individual dimensions they stay
      attached to their own number: "N8xC64x112x112" (NCHW),
      "B8H12S1024D64" (batch/heads/seq/head-dim), "V50304xD768_T49152".
    * Any extra axis the op needs is folded on as a trailing "_suffix"
      ("_all" full reduction, "_d0" reduced dim, "_o7" output size), the
      baseline path having no separate axis to put it on.

    Renaming a shape token renames a baseline key: rename the recorded
    entries in the same commit or the ratios orphan and re-measure as new.
    """

    op: str
    dtype: str
    shape: str
    layout: str

    def __str__(self) -> str:
        return f"{self.op}/{self.dtype}/{self.shape}/{self.layout}"


def _check_ratio(value: float, where: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{where}: a baseline ratio must be a finite positive number "
            f"(ours/stock device time), got {value!r}"
        )


class ShapeBlock(BaseModel):
    """One shape's ratio leaves, keyed by layout token."""

    model_config = ConfigDict(extra="forbid")

    layouts: dict[str, float] = Field(default_factory=dict)

    @field_validator("layouts")
    @classmethod
    def _check_layouts(cls, value: dict[str, float]) -> dict[str, float]:
        for layout, ratio in value.items():
            if not layout:
                raise ValueError("layout tokens must be non-empty strings")
            _check_ratio(ratio, f"layouts[{layout!r}]")
        return value


class DtypeBlock(BaseModel):
    """One dtype's shapes."""

    model_config = ConfigDict(extra="forbid")

    shapes: dict[str, ShapeBlock] = Field(default_factory=dict)


class OpBlock(BaseModel):
    """One aten op's dtypes."""

    model_config = ConfigDict(extra="forbid")

    dtypes: dict[str, DtypeBlock] = Field(default_factory=dict)


class ConfigBlock(BaseModel):
    """One hardware configuration's axis tree: op -> dtype -> shape -> layout."""

    model_config = ConfigDict(extra="forbid")

    ops: dict[str, OpBlock] = Field(default_factory=dict)

    def leaves(self) -> dict[BenchKey, float]:
        """Every ratio leaf under this config, addressed by BenchKey."""
        out: dict[BenchKey, float] = {}
        for op, op_block in self.ops.items():
            for dtype, dtype_block in op_block.dtypes.items():
                for shape, shape_block in dtype_block.shapes.items():
                    for layout, ratio in shape_block.layouts.items():
                        out[BenchKey(op, dtype, shape, layout)] = ratio
        return out


class BaselinesFile(BaseModel):
    """The full contents of the data block. See the module docstring."""

    model_config = ConfigDict(extra="forbid")

    format: int
    configs: dict[str, ConfigBlock] = Field(default_factory=dict)

    @field_validator("format")
    @classmethod
    def _check_format(cls, value: int) -> int:
        if value != FORMAT_VERSION:
            hint = {
                2: " (format 2 was the flat node-id layout; migrate it with "
                "the axis-tree migration before loading)",
                3: " (format 3 nested dtype above op and stored a derived "
                "min/median/max beside every container; run "
                "bench_lib.baselines.rebuild() to migrate it)",
                4: " (format 4 nested dtype above op; run "
                "bench_lib.baselines.rebuild() to transpose it)",
            }.get(value, "")
            raise ValueError(
                f"unknown baseline format {value!r}, expected {FORMAT_VERSION!r}{hint}"
            )
        return value


def prune_empty(data: BaselinesFile) -> BaselinesFile:
    """Drop every block that holds no ratio leaf.  Mutates and returns `data`.

    The suite only ever creates a block on its way to writing a leaf into
    it, so an empty one is debris from a hand edit or a merge.  Writers
    call this through canonical_dump.
    """
    for config in data.configs.values():
        for op_block in config.ops.values():
            for dtype_block in op_block.dtypes.values():
                for shape in [
                    s for s, b in dtype_block.shapes.items() if not b.layouts
                ]:
                    del dtype_block.shapes[shape]
            for dtype in [d for d, b in op_block.dtypes.items() if not b.shapes]:
                del op_block.dtypes[dtype]
        for op in [o for o, b in config.ops.items() if not b.dtypes]:
            del config.ops[op]
    for config_key in [k for k, c in data.configs.items() if not c.ops]:
        del data.configs[config_key]
    return data


def empty() -> BaselinesFile:
    return BaselinesFile(format=FORMAT_VERSION, configs={})


def _parse(raw: object, path: Path) -> BaselinesFile:
    try:
        return BaselinesFile.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"{path}: malformed baseline file:\n{exc}") from exc


def split_document(text: str, path: Path = BASELINES_PATH) -> tuple[str, str, str]:
    """Cut the document into (before, data block, after).

    The two outer parts are the viewer; a write splices a new block between
    them and leaves them byte-for-byte alone.
    """
    start = text.find(BLOCK_OPEN)
    if start < 0:
        raise ValueError(
            f"{path}: no {BLOCK_OPEN} block — this file should carry the "
            "baseline ratios inside it (it is both the data and the viewer); "
            "restore it from git rather than recreating it by hand"
        )
    body = start + len(BLOCK_OPEN)
    end = text.find(BLOCK_CLOSE, body)
    if end < 0:
        raise ValueError(f"{path}: the {BLOCK_OPEN} block is never closed")
    return text[:body], text[body:end], text[end:]


def load(path: Path = BASELINES_PATH) -> BaselinesFile:
    if not path.exists():
        return empty()
    _, block, _ = split_document(path.read_text(), path)
    try:
        raw = json.loads(block)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path}: the data block is not valid JSON ({exc}); it is "
            "machine-written — run bench_lib.baselines.rebuild() if a merge "
            "left it broken"
        ) from exc
    return _parse(raw, path)


def lookup(data: BaselinesFile, hw_key: str, bench_key: BenchKey) -> float | None:
    config = data.configs.get(hw_key)
    if config is None:
        return None
    op_block = config.ops.get(bench_key.op)
    if op_block is None:
        return None
    dtype_block = op_block.dtypes.get(bench_key.dtype)
    if dtype_block is None:
        return None
    shape_block = dtype_block.shapes.get(bench_key.shape)
    if shape_block is None:
        return None
    return shape_block.layouts.get(bench_key.layout)


def canonical_dump(data: BaselinesFile) -> str:
    """The one serialized form of the data block: pruned, sorted, indented.

    "<" is escaped: JSON allows it only inside strings, and inside a
    <script> element a literal "</script>" in the data would end the block
    early.  No baseline key contains one today; the escape means none ever
    can break the document.
    """
    pruned = prune_empty(data.model_copy(deep=True))
    text = json.dumps(pruned.model_dump(), indent=2, sort_keys=True)
    return "\n" + text.replace("<", "\\u003c") + "\n"


def write(data: BaselinesFile, path: Path = BASELINES_PATH) -> None:
    """Splice `data` into the file's data block, atomically.

    The viewer around the block survives byte-for-byte.
    """
    text = path.read_text() if path.exists() else MINIMAL_DOCUMENT
    before, _, after = split_document(text, path)
    tmp_path = path.with_suffix(".html.tmp")
    tmp_path.write_text(before + canonical_dump(data) + after)
    os.replace(tmp_path, path)


def merge_write(
    hw_key: str, entries: dict[BenchKey, float], path: Path = BASELINES_PATH
) -> None:
    """Merge `entries` into the file.

    Every line not in `entries` survives byte-for-byte: only the measured
    leaves change.
    """
    if not entries:
        return
    lock_path = path.with_suffix(".html.lock")
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            data = load(path)  # fresh read under the lock, not a cached copy
            config = data.configs.setdefault(hw_key, ConfigBlock())
            for key, ratio in entries.items():
                op_block = config.ops.setdefault(key.op, OpBlock())
                dtype_block = op_block.dtypes.setdefault(key.dtype, DtypeBlock())
                shape_block = dtype_block.shapes.setdefault(key.shape, ShapeBlock())
                shape_block.layouts[key.layout] = round(float(ratio), 3)
            write(data, path)
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


def _without_legacy_aggregates(node: dict[str, object]) -> dict[str, object]:
    """A format-3 tree minus its derived "aggregate" objects, recursively."""
    return {
        key: _without_legacy_aggregates(child) if isinstance(child, dict) else child
        for key, child in node.items()
        if key != LEGACY_AGGREGATE_KEY
    }


def _transposed(configs: dict[str, dict]) -> dict[str, dict]:
    """Formats 3 and 4 nested dtype above op; format 5 nests op above dtype.

    Rebuilds each config as ops -> dtypes -> shapes, carrying the shape
    blocks across whole: no ratio is read or rewritten, so the migration
    cannot alter a measurement.
    """
    out = {}
    for config_key, config in configs.items():
        ops = {}
        for dtype, dtype_block in config.get("dtypes", {}).items():
            for op, op_block in dtype_block.get("ops", {}).items():
                ops.setdefault(op, {})[dtype] = {"shapes": op_block.get("shapes", {})}
        out[config_key] = {
            "ops": {op: {"dtypes": dtypes} for op, dtypes in ops.items()}
        }
    return out


def rebuild(path: Path = BASELINES_PATH) -> None:
    """Re-canonicalize the data block: prune empty blocks, sort, rewrite.

    The escape hatch for the two documented manual situations — resolving
    a git merge of two added blocks, and migrating a block written by an
    older format (3 also stored aggregates; 3 and 4 both nested dtype
    above op).  The viewer is left alone.
    """
    _, block, _ = split_document(path.read_text(), path)
    raw = json.loads(block)
    if isinstance(raw, dict) and raw.get("format") in (3, 4):
        stripped = _without_legacy_aggregates(raw)
        raw = {
            "configs": _transposed(stripped.get("configs", {})),
            "format": FORMAT_VERSION,
        }
    write(_parse(raw, path), path)
