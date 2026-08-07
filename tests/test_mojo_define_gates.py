"""The specialization key must name exactly the defines the sources read.

A `.so` is identified by its source hash plus the compiler defines that
select its generated code. The loader derives that second half by scanning
the Mojo sources for the gates they call (`eager_kernels._live_defines`),
which makes two silent failures possible:

- **dead key bloat** — Python emits a define no `comptime` gate reads, so
  two calls that differ only in it fork a second, byte-identical build. The
  loader drops such defines, but only as long as its parser understands
  every gate form in the tree: one unparseable file and it conservatively
  keeps everything again.
- **a missing define** — a source gates on a define Python never sends, so
  the gate is permanently false and the operation quietly disappears from
  the build (the .so then has no `call` at all, or worse, the wrong one).

These tests close both directions statically, on any host, with no compiler
and no device.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from torch_mojo_backend import eager_kernels

_KERNEL_DIR = Path(eager_kernels.__file__).parent
_BACKEND_DIR = _KERNEL_DIR.parent
_ATEN_FAST = _KERNEL_DIR / "aten_fast.py"
_ENTRY_SOURCES = sorted(_KERNEL_DIR.glob("*_ops/*_ops.mojo"))
# `DTYPE_OUT` and `DTYPE_OUT_0` are two spellings of one gate (variant_gates
# accepts either for output 0), so compare them as one name.
_OUTPUT_ZERO = {"DTYPE_OUT", "DTYPE_OUT_0"}
_DEFINE_NAME_RE = re.compile(r"[A-Z][A-Z0-9_]*")


def _canonical(names: frozenset[str] | set[str]) -> set[str]:
    return {"DTYPE_OUT" if name in _OUTPUT_ZERO else name for name in names}


@dataclass
class _Emitted:
    """The define names one Mojo source can receive from Python."""

    names: set[str] = field(default_factory=set)
    any_arg_index: bool = False  # a call site builds arg_dtypes dynamically
    any_out_index: bool = False


def _tuple_length(node: ast.expr | None) -> int | None:
    """Upper bound on a tuple expression's length, or None when dynamic."""
    if isinstance(node, ast.Tuple):
        if any(isinstance(element, ast.Starred) for element in node.elts):
            return None
        return len(node.elts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _tuple_length(node.left)
        right = _tuple_length(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.IfExp):  # `(x,) if cond else ()`
        body = _tuple_length(node.body)
        orelse = _tuple_length(node.orelse)
        return None if body is None or orelse is None else max(body, orelse)
    return None


def _extension_sources(tree: ast.Module) -> dict[str, Path]:
    """Imported descriptor alias -> the Mojo source it drives."""
    sources: dict[str, Path] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        package = node.module.rsplit(".", 1)[-1]
        source = Path(package) / f"{package}.mojo"
        if not (_KERNEL_DIR / source).is_file():
            continue
        for alias in node.names:
            sources[alias.asname or alias.name] = source
    return sources


def _literal_define_names(node: ast.AST) -> tuple[set[str], set[str]]:
    """Define names spelled literally under `node`, and f-string prefixes."""
    names: set[str] = set()
    prefixes: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.JoinedStr):
            head = sub.values[0]
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                prefixes.add(head.value)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            if _DEFINE_NAME_RE.fullmatch(sub.value):
                names.add(sub.value)
    return names - prefixes, prefixes


def _class_mojo_file(
    node: ast.ClassDef, sources: dict[str, Path]
) -> tuple[Path | None, bool]:
    """The source a descriptor class drives, and whether it names one at all
    (a class that names one this scan cannot follow must be reported, not
    quietly skipped)."""
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target, value = statement.target.id, statement.value
        elif isinstance(statement, ast.Assign) and isinstance(
            statement.targets[0], ast.Name
        ):
            target, value = statement.targets[0].id, statement.value
        else:
            continue
        if target != "MOJO_FILE":
            continue
        if (
            isinstance(value, ast.Attribute)
            and value.attr == "MOJO_FILE"
            and isinstance(value.value, ast.Name)
        ):
            return sources.get(value.value.id), True
        return None, True
    return None, False


def _emitted_defines() -> tuple[dict[Path, _Emitted], list[str]]:
    """Every define name `aten_fast` can send, per Mojo source.

    Two emitters: the `_call_mojo` sites (op name, one `DTYPE_ARG_i` per
    entry of `arg_dtypes`, the output dtypes, and the literal `flags`), and
    the `MojoExtension` descriptor classes, whose `make_defines` spells its
    names literally. Anything that cannot be read statically is reported so
    the caller fails loudly instead of silently checking less.
    """
    tree = ast.parse(_ATEN_FAST.read_text())
    sources = _extension_sources(tree)
    emitted: dict[Path, _Emitted] = {}
    unresolved: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_call_mojo":
            continue
        target = node.args[0]
        source = sources.get(target.id) if isinstance(target, ast.Name) else None
        if source is None:
            unresolved.append(f"_call_mojo extension at line {node.lineno}")
            continue
        entry = emitted.setdefault(source, _Emitted())
        entry.names.add("OP")
        keywords = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        if "arg_dtypes" not in keywords:
            unresolved.append(f"_call_mojo arg_dtypes at line {node.lineno}")
            continue
        count = _tuple_length(keywords["arg_dtypes"])
        if count is None:
            entry.any_arg_index = True
        else:
            entry.names |= {f"DTYPE_ARG_{index}" for index in range(count)}
        outputs = keywords.get("output_dtypes")
        if outputs is not None:
            count = _tuple_length(outputs)
            if count is None:
                entry.any_out_index = True
            elif count == 1:
                entry.names.add("DTYPE_OUT")
            else:
                entry.names |= {f"DTYPE_OUT_{index}" for index in range(count)}
        flags = keywords.get("flags")
        if isinstance(flags, ast.Dict):
            for key in flags.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    entry.names.add(key.value)
                else:
                    unresolved.append(f"flag name at line {node.lineno}")
        elif flags is not None and not (
            isinstance(flags, ast.Constant) and flags.value is None
        ):
            unresolved.append(f"_call_mojo flags at line {node.lineno}")

    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    for name, node in classes.items():
        source, declared = _class_mojo_file(node, sources)
        if source is None:
            if declared:
                unresolved.append(f"{name}.MOJO_FILE")
            continue  # not a descriptor for one of this package's sources
        entry = emitted.setdefault(source, _Emitted())
        # make_defines may be inherited; follow the first declared base.
        current: str | None = name
        while current in classes:
            owner = classes[current]
            body = next(
                (
                    function
                    for function in owner.body
                    if isinstance(function, ast.FunctionDef)
                    and function.name == "make_defines"
                ),
                None,
            )
            if body is not None:
                names, prefixes = _literal_define_names(body)
                entry.names |= names
                entry.any_arg_index |= any(
                    prefix.startswith("DTYPE_ARG_") for prefix in prefixes
                )
                entry.any_out_index |= any(
                    prefix.startswith("DTYPE_OUT") for prefix in prefixes
                )
                break
            bases = [base.id for base in owner.bases if isinstance(base, ast.Name)]
            current = bases[0] if bases else None
        else:
            unresolved.append(f"{name}.make_defines")
    return emitted, unresolved


def test_the_gate_parser_understands_every_mojo_source() -> None:
    """No source may defeat the scan: one that does silently restores the
    old behaviour (every emitted define back in the key, one build each)."""
    sources = sorted(_BACKEND_DIR.rglob("*.mojo"))
    assert sources, f"no .mojo sources found under {_BACKEND_DIR}"
    unparseable = [
        str(path.relative_to(_BACKEND_DIR))
        for path in sources
        if eager_kernels._scan_define_names((path,)) is None
    ]
    assert not unparseable, (
        "the loader cannot tell which defines these sources read, so every "
        "specialization of them keeps its dead defines and compiles again. "
        "Teach _scan_define_names the new gate form:\n  " + "\n  ".join(unparseable)
    )


def test_every_entry_module_is_gated_on_its_operation() -> None:
    """One `call` per .so is what the OP gate buys; a module that stopped
    reading OP would compile every one of its operations into every
    variant."""
    assert _ENTRY_SOURCES
    for source in _ENTRY_SOURCES:
        names = eager_kernels._define_names_read_by(source.resolve())
        assert names is not None, source
        assert "OP" in names, source


def test_an_unreadable_gate_form_keeps_every_define(tmp_path: Path) -> None:
    """The conservative fallback still exists: when the scan cannot prove
    which defines a source reads, none of them may be dropped."""
    computed = tmp_path / "computed_ops.mojo"
    computed.write_text(
        "from variant_gates import _op_on\n"
        "\n"
        "def PyInit_computed_ops():\n"
        "    comptime if _op_on[selected_name]():\n"
        "        pass\n"
    )
    defines = eager_kernels.normalize_defines({"OP": "AddSpec", "INPLACE": True})

    assert eager_kernels._define_names_read_by(computed) is None
    assert eager_kernels._live_defines(computed, defines) == defines


def test_unread_defines_never_reach_the_specialization_key(tmp_path: Path) -> None:
    gated = tmp_path / "gated_ops.mojo"
    gated.write_text(
        "from variant_gates import _op_on, _dtype_arg_on\n"
        "\n"
        "def PyInit_gated_ops():\n"
        '    comptime if _op_on["AddSpec"]() and _dtype_arg_on[0, DType.float32]():\n'
        "        pass\n"
    )
    defines = eager_kernels.normalize_defines(
        {
            "OP": "AddSpec",
            "DTYPE_ARG_0": "float32",
            "DTYPE_ARG_1": "float32",  # no gate reads argument 1
            "INPLACE": True,  # nor this flag
        }
    )

    assert eager_kernels._live_defines(gated, defines) == (
        ("DTYPE_ARG_0", "float32"),
        ("OP", "AddSpec"),
    )


def test_python_sends_exactly_the_defines_each_module_reads() -> None:
    """Bidirectional: nothing a module gates on is left unsent, and nothing
    sent but ungated survives into its build key."""
    emitted, unresolved = _emitted_defines()
    assert not unresolved, (
        "this test can no longer see which defines Python sends; extend the "
        "scan rather than losing the check:\n  " + "\n  ".join(unresolved)
    )
    assert len(emitted) == len(_ENTRY_SOURCES), sorted(
        {source.parent.name for source in _ENTRY_SOURCES}
        - {source.parent.name for source in emitted}
    )

    for source, entry in sorted(emitted.items()):
        resolved = (_KERNEL_DIR / source).resolve()
        read = eager_kernels._define_names_read_by(resolved)
        assert read is not None, source
        sent = _canonical(entry.names)
        missing = {
            name
            for name in _canonical(read) - sent
            if not (entry.any_arg_index and name.startswith("DTYPE_ARG_"))
            and not (entry.any_out_index and name.startswith("DTYPE_OUT"))
        }
        assert not missing, (
            f"{source} gates on {sorted(missing)}, which no Python call site "
            "sends: those gates are permanently off"
        )
        key = eager_kernels._live_defines(
            resolved, eager_kernels.normalize_defines(dict.fromkeys(entry.names, "x"))
        )
        assert {name for name, _ in key} == read & entry.names, (
            f"{source}: the build key does not match the gates it reads"
        )


def test_dtype_supported_call_sites_count_as_argument_gate_reads(
    tmp_path: Path,
) -> None:
    """`_dtype_supported[DTYPES]` reads DTYPE_ARG_0 (and `..., N]` reads
    DTYPE_ARG_N) through the gate library, which the scanner skips — so its
    call sites must keep those defines in the key."""
    source = tmp_path / "helper_ops.mojo"
    source.write_text(
        "from variant_gates import _op_on, _dtype_supported\n"
        "\n"
        "def PyInit_helper_ops():\n"
        '    comptime if _op_on["AddSpec"]():\n'
        "        if not _dtype_supported[SPEC_DTYPES](a.dtype):\n"
        "            raise Error('unsupported')\n"
        "        if not _dtype_supported[SPEC_DTYPES, 1](b.dtype):\n"
        "            raise Error('unsupported')\n"
    )

    names = eager_kernels._scan_define_names((source,))
    assert names is not None
    assert {"OP", "DTYPE_ARG_0", "DTYPE_ARG_1"} <= names


def test_an_aliased_or_computed_dtype_supported_keeps_every_define(
    tmp_path: Path,
) -> None:
    """The conservative fallback covers the new helper too: an aliased
    mention or a non-literal index defeats the scan and keeps all defines."""
    aliased = tmp_path / "aliased_ops.mojo"
    aliased.write_text(
        "from variant_gates import _dtype_supported\n"
        "\n"
        "def PyInit_aliased_ops():\n"
        "    check = _dtype_supported\n"
    )
    computed_index = tmp_path / "computed_index_ops.mojo"
    computed_index.write_text(
        "from variant_gates import _dtype_supported\n"
        "\n"
        "def PyInit_computed_index_ops():\n"
        "    if not _dtype_supported[SPEC_DTYPES, which](a.dtype):\n"
        "        pass\n"
    )

    assert eager_kernels._scan_define_names((aliased,)) is None
    assert eager_kernels._scan_define_names((computed_index,)) is None
