"""The single git-tracked baseline file and its merge discipline.

Layout of benchmarks/baselines.json (format 3, the axis tree):

    {
      "format": 3,
      "configs": {
        "NVIDIA H100 PCIe | cuda | kineto-device-time@1395MHz | torch 2.11.0+cu130": {
          "aggregate": {"max": 39.4, "median": 3.71, "min": 0.788},
          "dtypes": {
            "bf16": {
              "aggregate": {...},
              "ops": {
                "mm": {
                  "aggregate": {...},
                  "shapes": {
                    "S1_4096x4096x4096": {
                      "aggregate": {...},
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

The tree path is hardware -> dtype -> op -> shape -> layout -> ratio.  A
leaf is addressed by a BenchKey (dtype, op, shape, layout); pytest derives
it from the test's own axes (see bench_lib/check.py), so the failing test
and its baseline entry still name the same kernel regime.

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
* Every category level (config, dtype, op, shape) carries an "aggregate"
  object with the min/median/max of ALL ratio leaves beneath it.
  Placement rule: at each level the children live under a NAMED container
  key ("dtypes" / "ops" / "shapes" / "layouts") and the aggregate sits
  BESIDE that container under the reserved key "aggregate" — never inside
  it.  A leaf is always a bare number inside "layouts"; an aggregate is
  always an object outside every container; the two can never be
  confused, and no dtype/op/shape/layout name can collide with the
  reserved key.  Aggregates are DERIVED data: every writer recomputes
  them from the file's own leaves (canonical_dump does it
  unconditionally), so the file is internally consistent no matter which
  subset of benchmarks ran, and load() refuses a file whose stored
  aggregates disagree with its leaves.
* Serialization is canonical (sorted keys, indent 2, one entry per line,
  trailing newline).  Because every writer re-emits the same canonical
  form, a partial update rewrites untouched LEAVES byte-for-byte (only
  the measured leaves and their ancestors' recomputed aggregates move),
  each hardware config occupies one contiguous sorted block, and two
  machines updating different configs touch disjoint regions that git
  merges cleanly — apart from aggregates, whose churn is confined to a
  handful of short reserved-key lines per updated level.

Writes go through merge_write(): read the file fresh under an exclusive
flock, update ONLY the measured leaves, recompute every aggregate from
the resulting leaves, write atomically (tmp + rename).  A run that
measured three relu cases can therefore never clobber the GEMM numbers it
did not run, nor leaves written by a concurrent run.

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
then re-canonicalize so the next writer diffs cleanly and every aggregate
matches the merged leaves:

    uv run python -c "import sys; sys.path.insert(0, 'benchmarks'); \\
        from bench_lib import baselines; baselines.rebuild()"

rebuild() is also the repair path when load() rejects hand-edited
aggregates: it re-reads the file without the consistency check, recomputes
every aggregate from the leaves, and rewrites canonically.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import statistics
import typing
from collections.abc import Iterator
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

FORMAT_VERSION = 3

BASELINES_PATH = Path(__file__).resolve().parent.parent / "baselines.json"

# Reserved sibling key holding min/median/max at every category level.
AGGREGATE_KEY = "aggregate"


class BenchKey(typing.NamedTuple):
    """Address of one ratio leaf inside a hardware config's axis tree.

    Axis order follows the tree: dtype -> op -> shape -> layout.  The op
    token is the aten base name with the overload variant folded in
    ("mm", "add.Tensor"); any extra axis an op needs is folded into the
    shape token; ops without a layout notion use the fixed sentinel
    "contig" so every path has the same rank.
    """

    dtype: str
    op: str
    shape: str
    layout: str

    def __str__(self) -> str:
        return f"{self.dtype}/{self.op}/{self.shape}/{self.layout}"


def _check_ratio(value: float, where: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{where}: a baseline ratio must be a finite positive number "
            f"(ours/stock device time), got {value!r}"
        )


class Aggregate(BaseModel):
    """min/median/max of every ratio leaf below one category level.

    Derived data: recomputed from the leaves on every write, and checked
    against them on load.  Lives under the reserved sibling key
    "aggregate", outside the child container, so it can never be walked
    into or mistaken for a leaf.
    """

    model_config = ConfigDict(extra="forbid")

    min: float
    median: float
    max: float

    @model_validator(mode="after")
    def _check_ordering(self, info: ValidationInfo) -> Aggregate:
        if isinstance(info.context, dict) and info.context.get("skip_aggregates"):
            # rebuild() is about to recompute this aggregate anyway; refusing
            # to load a broken one here would make repair impossible.
            return self
        for name in ("min", "median", "max"):
            _check_ratio(getattr(self, name), f"aggregate.{name}")
        if not self.min <= self.median <= self.max:
            raise ValueError(
                f"aggregate is not ordered (min {self.min} <= median "
                f"{self.median} <= max {self.max} must hold); it was "
                "hand-edited or corrupted — run bench_lib.baselines.rebuild() "
                "to recompute every aggregate from the leaves"
            )
        return self


def _placeholder_aggregate() -> Aggregate:
    # Never serialized: canonical_dump recomputes every aggregate from the
    # leaves before emitting.  Exists so in-memory blocks can be built
    # leaf-first by merge_write.
    return Aggregate(min=1.0, median=1.0, max=1.0)


class ShapeBlock(BaseModel):
    """One shape's ratio leaves, keyed by layout token."""

    model_config = ConfigDict(extra="forbid")

    aggregate: Aggregate = Field(default_factory=_placeholder_aggregate)
    layouts: dict[str, float] = Field(default_factory=dict)

    @field_validator("layouts")
    @classmethod
    def _check_layouts(cls, value: dict[str, float]) -> dict[str, float]:
        for layout, ratio in value.items():
            if not layout:
                raise ValueError("layout tokens must be non-empty strings")
            _check_ratio(ratio, f"layouts[{layout!r}]")
        return value


class OpBlock(BaseModel):
    """One aten op's shapes."""

    model_config = ConfigDict(extra="forbid")

    aggregate: Aggregate = Field(default_factory=_placeholder_aggregate)
    shapes: dict[str, ShapeBlock] = Field(default_factory=dict)


class DtypeBlock(BaseModel):
    """One dtype's ops."""

    model_config = ConfigDict(extra="forbid")

    aggregate: Aggregate = Field(default_factory=_placeholder_aggregate)
    ops: dict[str, OpBlock] = Field(default_factory=dict)


class ConfigBlock(BaseModel):
    """One hardware configuration's axis tree: dtype -> op -> shape -> layout."""

    model_config = ConfigDict(extra="forbid")

    aggregate: Aggregate = Field(default_factory=_placeholder_aggregate)
    dtypes: dict[str, DtypeBlock] = Field(default_factory=dict)

    def leaves(self) -> dict[BenchKey, float]:
        """Every ratio leaf under this config, addressed by BenchKey."""
        out: dict[BenchKey, float] = {}
        for dtype, dtype_block in self.dtypes.items():
            for op, op_block in dtype_block.ops.items():
                for shape, shape_block in op_block.shapes.items():
                    for layout, ratio in shape_block.layouts.items():
                        out[BenchKey(dtype, op, shape, layout)] = ratio
        return out


class BaselinesFile(BaseModel):
    """The full contents of benchmarks/baselines.json. See the module docstring."""

    model_config = ConfigDict(extra="forbid")

    format: int
    configs: dict[str, ConfigBlock] = Field(default_factory=dict)

    @field_validator("format")
    @classmethod
    def _check_format(cls, value: int) -> int:
        if value != FORMAT_VERSION:
            raise ValueError(
                f"unknown baseline format {value!r}, expected {FORMAT_VERSION!r}"
                + (
                    " (format 2 was the flat node-id layout; migrate it with "
                    "the axis-tree migration before loading)"
                    if value == 2
                    else ""
                )
            )
        return value

    @model_validator(mode="after")
    def _check_aggregates(self, info: ValidationInfo) -> BaselinesFile:
        # Stored aggregates must equal what the leaves imply; otherwise a
        # hand edit went stale.  Skipped via context by rebuild(), whose
        # whole point is repairing exactly this.
        if isinstance(info.context, dict) and info.context.get("skip_aggregates"):
            return self
        recomputed = recompute_aggregates(self.model_copy(deep=True))
        for config_key, config in self.configs.items():
            recomputed_config = recomputed.configs.get(config_key)
            if recomputed_config is None:
                raise ValueError(
                    f"configs[{config_key!r}] contains no ratio leaves at all "
                    "(empty blocks are never written by the suite); delete the "
                    "block or run bench_lib.baselines.rebuild() to prune it"
                )
            try:
                pairs = list(_aggregate_pairs(config, recomputed_config))
            except KeyError as missing:
                raise ValueError(
                    f"configs[{config_key!r}] contains an empty block at or "
                    f"under key {missing.args[0]!r} (no ratio leaves; the "
                    "suite never writes those); delete it or run "
                    "bench_lib.baselines.rebuild() to prune it"
                ) from missing
            for path, stored, expected in pairs:
                if stored != expected:
                    raise ValueError(
                        f"stale aggregate at configs[{config_key!r}]{path}: "
                        f"stored {stored.model_dump()} but the leaves imply "
                        f"{expected.model_dump()}; aggregates are derived "
                        "data — run bench_lib.baselines.rebuild() to "
                        "recompute them from the leaves"
                    )
        return self


def _aggregate_pairs(
    stored: ConfigBlock, expected: ConfigBlock
) -> Iterator[tuple[str, Aggregate, Aggregate]]:
    yield "", stored.aggregate, expected.aggregate
    for dtype, stored_dt in stored.dtypes.items():
        expected_dt = expected.dtypes[dtype]
        yield f".dtypes[{dtype!r}]", stored_dt.aggregate, expected_dt.aggregate
        for op, stored_op in stored_dt.ops.items():
            expected_op = expected_dt.ops[op]
            yield (
                f".dtypes[{dtype!r}].ops[{op!r}]",
                stored_op.aggregate,
                expected_op.aggregate,
            )
            for shape, stored_shape in stored_op.shapes.items():
                yield (
                    f".dtypes[{dtype!r}].ops[{op!r}].shapes[{shape!r}]",
                    stored_shape.aggregate,
                    expected_op.shapes[shape].aggregate,
                )


def _aggregate_of(values: list[float]) -> Aggregate:
    return Aggregate(
        min=round(min(values), 3),
        median=round(statistics.median(values), 3),
        max=round(max(values), 3),
    )


def recompute_aggregates(data: BaselinesFile) -> BaselinesFile:
    """Recompute every aggregate from the leaves, pruning empty blocks.

    Mutates and returns `data`.  This is the ONLY producer of aggregate
    values: writers call it (via canonical_dump) so the stored min/median/
    max can never drift from the leaves they summarize.
    """
    for config in data.configs.values():
        config_leaves: list[float] = []
        for dtype_block in config.dtypes.values():
            dtype_leaves: list[float] = []
            for op_block in dtype_block.ops.values():
                op_leaves: list[float] = []
                for shape, shape_block in list(op_block.shapes.items()):
                    if not shape_block.layouts:
                        del op_block.shapes[shape]
                        continue
                    values = list(shape_block.layouts.values())
                    shape_block.aggregate = _aggregate_of(values)
                    op_leaves.extend(values)
                if op_leaves:
                    op_block.aggregate = _aggregate_of(op_leaves)
                    dtype_leaves.extend(op_leaves)
            for op in [o for o, b in dtype_block.ops.items() if not b.shapes]:
                del dtype_block.ops[op]
            if dtype_leaves:
                dtype_block.aggregate = _aggregate_of(dtype_leaves)
                config_leaves.extend(dtype_leaves)
        for dtype in [d for d, b in config.dtypes.items() if not b.ops]:
            del config.dtypes[dtype]
        if config_leaves:
            config.aggregate = _aggregate_of(config_leaves)
    for config_key in [k for k, c in data.configs.items() if not c.dtypes]:
        del data.configs[config_key]
    return data


def empty() -> BaselinesFile:
    return BaselinesFile(format=FORMAT_VERSION, configs={})


def load(path: Path = BASELINES_PATH, check_aggregates: bool = True) -> BaselinesFile:
    if not path.exists():
        return empty()
    with path.open() as fh:
        raw = json.load(fh)
    try:
        return BaselinesFile.model_validate(
            raw, context={"skip_aggregates": not check_aggregates}
        )
    except ValidationError as exc:
        raise ValueError(f"{path}: malformed baseline file:\n{exc}") from exc


def lookup(data: BaselinesFile, hw_key: str, bench_key: BenchKey) -> float | None:
    config = data.configs.get(hw_key)
    if config is None:
        return None
    dtype_block = config.dtypes.get(bench_key.dtype)
    if dtype_block is None:
        return None
    op_block = dtype_block.ops.get(bench_key.op)
    if op_block is None:
        return None
    shape_block = op_block.shapes.get(bench_key.shape)
    if shape_block is None:
        return None
    return shape_block.layouts.get(bench_key.layout)


def canonical_dump(data: BaselinesFile) -> str:
    """The one serialized form: aggregates freshly recomputed, keys sorted."""
    consistent = recompute_aggregates(data.model_copy(deep=True))
    return json.dumps(consistent.model_dump(), indent=2, sort_keys=True) + "\n"


def merge_write(
    hw_key: str, entries: dict[BenchKey, float], path: Path = BASELINES_PATH
) -> None:
    """Merge `entries` into the file.

    Every leaf not in `entries` survives byte-for-byte; only the measured
    leaves change, plus their ancestors' aggregates, which are recomputed
    from the file's own leaves so the tree stays internally consistent.
    """
    if not entries:
        return
    lock_path = path.with_suffix(".json.lock")
    with lock_path.open("w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            data = load(path)  # fresh read under the lock, not a cached copy
            config = data.configs.setdefault(hw_key, ConfigBlock())
            for key, ratio in entries.items():
                dtype_block = config.dtypes.setdefault(key.dtype, DtypeBlock())
                op_block = dtype_block.ops.setdefault(key.op, OpBlock())
                shape_block = op_block.shapes.setdefault(key.shape, ShapeBlock())
                shape_block.layouts[key.layout] = round(float(ratio), 3)
            tmp_path = path.with_suffix(".json.tmp")
            tmp_path.write_text(canonical_dump(data))
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


def rebuild(path: Path = BASELINES_PATH) -> None:
    """Repair/re-canonicalize: recompute every aggregate and rewrite.

    The escape hatch for the two documented manual situations — resolving
    a git merge of two added blocks, and a hand edit that left aggregates
    stale (which load() refuses).
    """
    path.write_text(canonical_dump(load(path, check_aggregates=False)))
