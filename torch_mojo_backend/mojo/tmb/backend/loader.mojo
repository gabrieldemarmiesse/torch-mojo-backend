"""On-demand builds, from Mojo, of everything that is not the runtime.

Kernel families (torch_mojo_backend/mojo/tmb/kernels/<family>/entry.mojo)
export `tmb_call` and compile one (OP, dtypes, flags) specialization per
build, selected with -D defines. A build hashes the entry file's import
closure, looks the .so up in the cache, otherwise runs `mojo build` in a
subprocess (under a cross-process flock, atomic rename), dlopens it and
keeps the entry pointer. Every build records its compiler warnings beside the
.so, which TORCH_MOJO_BACKEND_WERROR=1 checks, fresh build or cache hit
(agents_docs/mojo_extensions.md, "Warnings records").
"""
from std.builtin.sort import sort
from std.collections import Dict
from std.ffi import OwnedDLHandle, external_call, get_errno
from std.os import getenv, makedirs
from std.os.path import exists, isdir, isfile
from std.pathlib import Path
from std.subprocess import run as run_command
from std.sys.info import CompilationTarget
from std.time import perf_counter_ns

from tmb.backend.env_vars import (
    MODULAR_MOJO_MAX_IMPORT_PATH,
    TMPDIR,
    TORCH_MOJO_BACKEND_WERROR,
)
from tmb.kernels.common.op_utils import Argv


comptime CACHE_ABI = "native-v2"

# Appended to every eager kernel family build below (`entry()`), never to the
# torch.compile graph package (`native.build_graph_package()`'s `mojo
# precompile` passes no `-D` at all -- see `native/__init__.py`).
# `tmb.kernels.common.gpu_elementwise.elementwise` reads it with
# `is_defined["TMB_EAGER_ELEMENTWISE"]()` to take the fast NVIDIA launcher
# only on this path: several eager kernels it calls (`_bias_add_row` in
# matmul/entry.mojo, `_gather0`/`op_utils._parallel_for`, ...) are also
# imported by tmb/graph's custom ops for the torch.compile backend, and that
# backend must keep MAX's own `elementwise` unchanged.
comptime _EAGER_ELEMENTWISE_DEFINE = "TMB_EAGER_ELEMENTWISE=1"

# Every Mojo source lives under one root, torch_mojo_backend/mojo, which is
# the one `-I` of every build, as one top-level package: `from
# tmb.<pkg>.<module> import ...`. Kernel families are
# `tmb/kernels/<family>/entry.mojo`.
comptime PACKAGE = "tmb"
# Written last into a source snapshot (`Loader.snapshot`): its presence says
# the snapshot is complete.
comptime SNAPSHOT_MARKER = ".snapshot-complete"
# How long a snapshot install waits for another process installing the same
# one (a copy of a few dozen source files: milliseconds in practice).
comptime SNAPSHOT_LOCK_TIMEOUT_MS = 120_000
# A snapshot temp dir older than this is an interrupted install's leftover.
comptime SNAPSHOT_TMP_MAX_AGE_MIN = 60
# errno of a busy LOCK_NB flock (EWOULDBLOCK == EAGAIN).
comptime _EWOULDBLOCK = 35 if CompilationTarget.is_macos() else 11


def _open_lock_cloexec(path: String) -> Int32:
    """A read-only, close-on-exec fd on `path`, created if missing (-1 on
    failure): the build subprocesses must not inherit the lock."""
    var p = String(path)
    comptime O_CREAT = 0x200 if CompilationTarget.is_macos() else 0o100
    comptime O_CLOEXEC = 0x1000000 if CompilationTarget.is_macos() else 0x80000
    return external_call["open", Int32, num_fixed_args=2](
        p.as_c_string_span().ptr(),
        Int32(O_CREAT | O_CLOEXEC),
        Int32(0o644),
    )  # O_RDONLY: flock needs no write access


comptime KERNELS = "tmb/kernels"

# `<so>.warnings`: the warning diagnostics of the build that made `<so>`,
# empty when there were none (native/__init__.py's `warnings_record`).
comptime WARNINGS_SUFFIX = ".warnings"


def _werror() -> Bool:
    return getenv(TORCH_MOJO_BACKEND_WERROR) == "1"


def _severity_prefix(text: String) -> String:
    """The diagnostic severity `text` starts with (`warning: ...`), or ""."""
    for sev in ["warning", "error", "note", "remark"]:
        if text.startswith(String(sev) + ": "):
            return String(sev)
    return String()


def _is_location(prefix: String) -> Bool:
    """`prefix` ends with `:<digits>:<digits>` (a `file:line:col`)."""
    var b = prefix.as_bytes()
    var i = len(b)
    for _ in range(2):
        var digits = 0
        while i > 0:
            var c = Int(b[i - 1])
            if c < ord("0") or c > ord("9"):
                break
            i -= 1
            digits += 1
        if digits == 0 or i == 0 or Int(b[i - 1]) != ord(":"):
            return False
        i -= 1
    return True


def _diagnostic_header(line: String) -> Tuple[String, Bool]:
    """(severity, has a `file:line:col`) of one of mojo's diagnostics, ""
    for anything else; native/__init__.py's `_diagnostic_header` is the
    same rule."""
    var at = line.find(": ")
    while at >= 0:
        var sev = _severity_prefix(String(line[byte = at + 2 :]))
        if sev != "":
            var prefix = String(line[byte=:at])
            if _is_location(prefix):
                return (sev, True)
            if prefix == "mojo" or prefix.endswith("/mojo"):
                return (sev, False)
        at = line.find(": ", at + 1)
    return (String(), False)


def warning_diagnostics(output: String) -> String:
    """The warnings in a compiler's output, each with the `Included from`
    lines above it, its source line, caret and attached notes, and nothing
    else (the Crashpad banner, errors, traces); native/__init__.py's
    `warning_diagnostics` is the same filter."""
    var kept = String()
    var included = String()  # `Included from` lines, kept with what follows
    var keeping = False
    var context = 0  # excerpt lines still expected: the source line, the caret
    for l in output.splitlines():
        var line = String(l)
        if line.startswith("Included from ") or line.startswith(
            "Imported from "
        ):
            included += line + "\n"
            context = 0
            continue
        var header = _diagnostic_header(line)
        if header[0] != "":
            keeping = header[0] == "warning" or (
                header[0] == "note" and keeping
            )
            context = 2 if keeping and header[1] else 0
            if keeping:
                kept += included
        elif keeping and context == 2:
            context = 1
        elif keeping and context == 1 and line.lstrip().startswith("^"):
            context = 0
        else:
            keeping = False
            context = 0
        included = String()
        if keeping:
            kept += line + "\n"
    return kept


def _defines_text(defines: List[String]) -> String:
    var out = String()
    for d in defines:
        out += " -D " + d
    return out


def _recorded(so: String) -> Optional[String]:
    """The warnings record of `so`, None when it cannot be read."""
    try:
        return Path(so + WARNINGS_SUFFIX).read_text()
    except:
        return None


def _werror_check(
    so: String, label: String, defines: List[String], diagnostics: String
) raises:
    """Under WERROR: fail if the build of `so` warned, fresh or cached
    (native/__init__.py's `werror_failure` writes the same)."""
    var record = so + WARNINGS_SUFFIX
    if diagnostics != "":
        raise Error(
            "mojo build of ",
            label,
            _defines_text(defines),
            (
                " failed:\nTORCH_MOJO_BACKEND_WERROR=1 makes these compiler"
                " warnings errors (recorded by the build in "
            ),
            record,
            "):\n",
            diagnostics,
        )


def _fnv1a(mut h: UInt64, bytes: Span[UInt8, _]):
    for i in range(len(bytes)):
        h ^= UInt64(bytes[i])
        h *= 1099511628211


def _hex(h: UInt64) -> String:
    return String(hex(h)[byte=2:])


def _slug(defines: List[String]) -> String:
    var h: UInt64 = 14695981039346656037
    for d in defines:
        _fnv1a(h, d.as_bytes())
        _fnv1a(h, "\n".as_bytes())
    return String(_hex(h)[byte=:12])


def _local_dir(prefix: String) raises -> String:
    """A per-user directory on node-local disk, created if missing.

    Expanded here rather than left as `${TMPDIR:-/tmp}` in the build command:
    the command quotes its paths, so the shell took that spelling literally
    and made a directory of that name inside the caller's working
    directory."""
    var tmp = getenv(TMPDIR)
    if tmp == "":
        tmp = String("/tmp")
    var d = tmp + "/" + prefix + String(external_call["getuid", UInt32]())
    if not isdir(d):
        makedirs(d, exist_ok=True)
    return d


def _compiler_env() raises -> String:
    """Environment every `mojo build` subprocess runs with; an explicit value
    wins. Why MODULAR_HOME and MODULAR_CACHE_DIR must be node-local:
    native/__init__.py's `compiler_env`, the same thing on the Python side."""
    # PYTHONEXECUTABLE/PYTHONHOME: the MAX runtime exports the interpreter it
    # found on PATH into this process's environment (invisible to os.environ,
    # inherited by children); with a venv that is not on PATH, the `mojo`
    # launcher script then starts /usr/bin/python3 with the venv's prefix and
    # dies with "Could not find platform independent libraries".
    return (
        'unset PYTHONEXECUTABLE PYTHONHOME; MODULAR_HOME="${MODULAR_HOME:-'
        + _local_dir("modular-home-")
        + '}" MODULAR_CACHE_DIR="${MODULAR_CACHE_DIR:-'
        + _local_dir("modular-cache-")
        + '}"'
    )


struct Family(Movable):
    var lib: OwnedDLHandle
    var entry: Int  # address of tmb_call

    def __init__(out self, path: String) raises:
        self.lib = OwnedDLHandle(path)
        var sym = self.lib.get_symbol[NoneType]("tmb_call")
        if not sym:
            raise Error("tmb_call not exported by ", path)
        self.entry = Int(sym.value())


struct Loader(Movable):
    var root: String  # torch_mojo_backend/mojo: the one -I of every build
    var cache_dir: String  # native/__init__.py's _CACHE_DIR (~/.cache/torch-mojo-backend/native)
    var mojo_exe: String
    var toolchain: String  # versions of mojo/max/python, from the Python side
    var trace: Bool
    var families: Dict[String, Family]  # "<family>.<slug>" -> loaded build
    var source_hashes: Dict[String, String]  # family -> closure hash
    var fast: Dict[
        UInt64, Int
    ]  # (family, defines) hash -> entry address (hot path)

    def __init__(
        out self,
        root: String,
        cache_dir: String,
        mojo_exe: String,
        toolchain: String,
        trace: Bool,
    ):
        self.root = root
        self.cache_dir = cache_dir
        self.mojo_exe = mojo_exe
        self.toolchain = toolchain
        self.trace = trace
        self.families = Dict[String, Family]()
        self.source_hashes = Dict[String, String]()
        self.fast = Dict[UInt64, Int]()

    def family_dir(self, family: String) -> String:
        """The package of one kernel family, `<root>/tmb/kernels/<family>`;
        its `entry.mojo` exports `tmb_call`."""
        return self.root + "/" + KERNELS + "/" + family

    def _module_file(self, dotted: String, importer_dir: String) -> String:
        """The source file an import names, or "" when it is not ours.

        `tmb.a.b` is `<root>/tmb/a/b.mojo` (`<root>/tmb/a/b/__init__.mojo`
        for a package); `.b` is `<importer_dir>/b.mojo` -- the graph package,
        which MAX compiles without any -I, is the one place relative imports
        remain. Everything else (std, max, nn, layout, ...) is the
        toolchain's, keyed by its version rather than hashed here."""
        var path: String
        if dotted.startswith("."):
            path = (
                importer_dir + "/" + String(dotted[byte=1:]).replace(".", "/")
            )
        elif dotted == PACKAGE or dotted.startswith(PACKAGE + "."):
            path = self.root + "/" + dotted.replace(".", "/")
        else:
            return String()
        if exists(path + ".mojo"):
            return path + ".mojo"
        if exists(path + "/__init__.mojo"):
            return path + "/__init__.mojo"
        return String()

    def _closure_texts(
        self, entry: String
    ) raises -> Tuple[List[String], Dict[String, String]]:
        """Every .mojo file `entry` reaches through `from X import` /
        `import X` (sorted: the sources one build compiles in, so touching
        any of them invalidates it), with the text each was read as. Each
        file is read ONCE, so the import walk, the hash and a snapshot all
        see the same bytes. native/__init__.py's `mojo_import_closure` is
        the same walk for the Python-driven builds."""
        var files = List[String]()
        var texts = Dict[String, String]()
        var todo = List[String]()
        todo.append(entry)
        while len(todo) > 0:
            var f = todo.pop()
            if f in texts:
                continue
            var text = Path(f).read_text()
            texts[f] = text
            files.append(f)
            var here = String(f[byte = : f.rfind("/")])
            for line in text.splitlines():
                var s = String(line)
                var name: String
                if s.startswith("from "):
                    var rest = String(s[byte=5:])
                    var sp = rest.find(" ")
                    name = String(rest[byte=:sp]) if sp > 0 else rest
                elif s.startswith("import "):
                    name = String(String(s[byte=7:]).strip())
                    var cut = name.find(" ")
                    if cut > 0:
                        var head = String(name[byte=:cut])
                        name = head^
                else:
                    continue
                var cand = self._module_file(name, here)
                if cand != "" and cand not in texts:
                    todo.append(cand)
        sort(files)
        return (files^, texts^)

    def _hash_texts(
        self, files: List[String], texts: Dict[String, String]
    ) raises -> String:
        """Cache key of one build: every source it compiles in, named
        relative to the root, plus the toolchain."""
        var h: UInt64 = 14695981039346656037
        _fnv1a(h, CACHE_ABI.as_bytes())
        _fnv1a(h, self.toolchain.as_bytes())
        for f in files:
            var rel = String(f[byte = self.root.byte_length() :])
            _fnv1a(h, rel.as_bytes())
            _fnv1a(h, texts[f].as_bytes())
        return _hex(h)

    def _hash_closure(mut self, key: String, entry: String) raises -> String:
        """The memoized cache key (the hot path's: hashed once per process).
        A build never trusts it -- `snapshot` re-reads the sources it
        compiles and keys the build by what it actually read."""
        if key in self.source_hashes:
            return self.source_hashes[key]
        var ft = self._closure_texts(entry)
        var out = self._hash_texts(ft[0], ft[1])
        self.source_hashes[key] = out
        return out

    def snapshot(mut self, family: String) raises -> Tuple[String, String]:
        """(snapshot root, hash) of the family's sources as they are NOW.

        The closure is read once and written to a node-local directory named
        by its hash, and the build compiles THAT copy: a source edited while
        the process runs can then never be compiled under the hash of its
        previous contents (the memoized key of `source_hash`), and two
        processes building the same hash share one stable path (the
        compiler's module cache keys on the source path). The memo is
        refreshed to what was read, so this process's later lookups use the
        key of the code it actually built."""
        var entry = self.family_dir(family) + "/entry.mojo"
        var ft = self._closure_texts(entry)
        var h = self._hash_texts(ft[0], ft[1])
        self.source_hashes["family:" + family] = h
        var base = _local_dir("torch-mojo-backend-src-")
        var dest = base + "/" + family + "-" + h
        if self._snapshot_complete(dest, ft[0]):
            return (dest, h)
        # Install (or repair) under a per-snapshot lock, re-checking inside
        # it: two processes that both found the snapshot missing or thinned
        # by TMPDIR aging then take turns, and the second sees the first's
        # complete copy instead of deleting it. The copy is written to a
        # unique temp dir, completion marker last, then renamed into place,
        # so nobody ever reads a half-written snapshot. A filesystem that
        # refuses the lock outright (not merely busy) gets the same install
        # without it.
        var fd = _open_lock_cloexec(dest + ".lock")
        if fd < 0:
            return self._install_snapshot(dest, h, ft[0], ft[1], False)
        var waited_ms = 0
        while (
            external_call["flock", Int32](fd, Int32(2 | 4)) != 0
        ):  # LOCK_EX | LOCK_NB
            if get_errno().value != Int32(_EWOULDBLOCK):
                _ = external_call["close", Int32](fd)
                return self._install_snapshot(dest, h, ft[0], ft[1], False)
            if waited_ms >= SNAPSHOT_LOCK_TIMEOUT_MS:
                _ = external_call["close", Int32](fd)
                raise Error(
                    "timed out waiting for the source snapshot lock ",
                    dest + ".lock",
                )
            _ = external_call["usleep", Int32](UInt32(10_000))
            waited_ms += 10
        try:
            if not self._snapshot_complete(dest, ft[0]):
                return self._install_snapshot(dest, h, ft[0], ft[1], True)
        finally:
            _ = external_call["flock", Int32](fd, Int32(8))  # LOCK_UN
            _ = external_call["close", Int32](fd)
        return (dest, h)

    def _install_snapshot(
        self,
        dest: String,
        h: String,
        files: List[String],
        texts: Dict[String, String],
        locked: Bool,
    ) raises -> Tuple[String, String]:
        """Write the closure to a unique temp dir (marker last) and rename it
        to `dest`; returns the root to build from. `locked`: we hold the
        snapshot lock, so an incomplete `dest` is ours to replace."""
        # Temp dirs of an interrupted install: old enough that no installer
        # can still be writing them, and named (`.tmp<pid>-<ns>`) after a
        # process that is gone -- a live one may be building from its copy
        # (the fallback below), however old. The directory is per-user, so
        # `kill -0` can probe every pid in it.
        var slash = dest.rfind("/")
        _ = run_command(
            "find '"
            + String(dest[byte=:slash])
            + "' -maxdepth 1 -name '"
            + String(dest[byte = slash + 1 :])
            + ".tmp*' -mmin +"
            + String(SNAPSHOT_TMP_MAX_AGE_MIN)
            + " 2>/dev/null | while read -r d; do p=${d##*.tmp}; p=${p%%-*};"
            + ' kill -0 "$p" 2>/dev/null || rm -rf "$d"; done; true'
        )
        var uniq = (
            String(external_call["getpid", Int32]())
            + "-"
            + String(perf_counter_ns())
        )
        var tmp = dest + ".tmp" + uniq
        for f in files:
            var rel = String(f[byte = self.root.byte_length() :])
            var target = tmp + rel
            makedirs(String(target[byte = : target.rfind("/")]), exist_ok=True)
            Path(target).write_text(texts[f])
        Path(tmp + "/" + SNAPSHOT_MARKER).write_text(h)
        if locked and isdir(dest):
            # Incomplete, and nobody else can be installing it (we hold the
            # lock): move it aside before deleting, so the rename below never
            # targets a non-empty directory.
            var stale = dest + ".stale" + uniq
            var live = String(dest)
            _ = external_call["rename", Int32](
                live.as_c_string_span().ptr(), stale.as_c_string_span().ptr()
            )
            _ = run_command("rm -rf '" + stale + "'")
        var d = String(dest)
        var r = external_call["rename", Int32](
            tmp.as_c_string_span().ptr(), d.as_c_string_span().ptr()
        )
        if r == 0:
            return (dest, h)
        if self._snapshot_complete(dest, files):
            # Lock-free: another process installed it first.
            _ = run_command("rm -rf '" + tmp + "'")
            return (dest, h)
        # Build from our own complete copy rather than fail.
        return (tmp, h)

    def _snapshot_complete(self, dest: String, files: List[String]) -> Bool:
        """A snapshot is reusable only if its marker and every source of the
        closure are present (node-local TMPDIR may age files out)."""
        if not exists(dest + "/" + SNAPSHOT_MARKER):
            return False
        for f in files:
            if not exists(dest + String(f[byte = self.root.byte_length() :])):
                return False
        return True

    def source_hash(mut self, family: String) raises -> String:
        return self._hash_closure(
            "family:" + family, self.family_dir(family) + "/entry.mojo"
        )

    def _build(
        mut self,
        label: String,
        src: String,
        root: String,
        defines: List[String],
        out_path: String,
    ) raises:
        """One `mojo build` into `out_path`. `label` only names the build in
        traces and scratch paths."""
        var tmp = out_path + ".tmp" + String(perf_counter_ns())
        # The compiler writes to local scratch (the cache may be on NFS,
        # where its intermediate archive went missing under load); the
        # finished library is then moved next to its final name.
        var scratch = _local_dir("torch-mojo-backend-")
        var local = (
            scratch
            + "/"
            + label.replace(" ", "_")
            + "."
            + String(perf_counter_ns())
            + ".so"
        )
        # MODULAR_HOME, MODULAR_CACHE_DIR: the compiler's own caches go to
        # local scratch too (native/__init__.py compiler_env explains why)
        var cmd = _compiler_env()
        var import_path = getenv(MODULAR_MOJO_MAX_IMPORT_PATH)
        if root != self.root and import_path != "":
            # The import path names the live source root too, and its entries
            # outrank -I: a snapshot build drops every entry holding a `tmb`
            # package, or each `tmb` module would resolve twice (ambiguous
            # import) -- or worse, to the live file.
            var kept = List[String]()
            for e in import_path.split(","):
                var entry = String(e)
                if entry != "" and not isdir(entry + "/" + KERNELS):
                    kept.append(entry)
            cmd += (
                " " + MODULAR_MOJO_MAX_IMPORT_PATH + "='" + ",".join(kept) + "'"
            )
        cmd += (
            " '"
            + self.mojo_exe
            + "' build '"
            + src
            + "' --emit shared-lib -I '"
            + root
            + "'"
        )
        comptime if CompilationTarget.is_macos():
            # Kernel families call the shim and the base library, resolved
            # at dlopen; ld64 wants to be told so.
            cmd += " -Xlinker -undefined -Xlinker dynamic_lookup"
        for d in defines:
            cmd += " -D '" + d + "'"
        # Never --Werror: WERROR is checked against the recorded warnings.
        cmd += (
            " -o '"
            + local
            + "' 2>&1 && mv -f '"
            + local
            + "' '"
            + tmp
            + "' 2>&1; echo __TMB_RC=$?"
        )
        var t0 = perf_counter_ns()
        var output = run_command(cmd)
        var ms = (perf_counter_ns() - t0) // 1_000_000
        var marker = output.rfind("__TMB_RC=")
        var rc = -1
        if marker >= 0:
            rc = Int(atol(String(String(output[byte = marker + 9 :]).strip())))
        var log = String(output[byte=:marker]) if marker > 0 else output
        if rc != 0:
            if exists(tmp):
                _ = external_call["unlink", Int32](tmp.as_c_string_span().ptr())
            # The assembler is chosen for us (torch_mojo_backend/_ptxas.py),
            # and when it is the wrong one for this driver or this GPU that
            # shows up here as a ptxas line in someone else's log. Say where
            # the answer is; Mojo cannot work it out, Python can.
            var hint = String()
            if "ptxas" in log or "MODULAR_NVPTX_COMPILER_PATH" in log:
                hint = String(
                    "\n\n`torch-mojo-backend ptxas` shows which assembler was"
                    " chosen for this driver and GPU, and what to install if"
                    " none fits."
                )
            raise Error(
                "mojo build of ",
                label,
                _defines_text(defines),
                " failed (rc ",
                rc,
                ", ",
                ms,
                " ms):\n",
                log,
                hint,
            )
        # The .so, then its warnings record. A .so already there (a
        # record-less one, under WERROR; lock-free, a concurrent builder's)
        # is kept, as native/__init__.py's `_install_mojo_build` explains.
        var record = out_path + WARNINGS_SUFFIX
        if exists(out_path):
            _ = external_call["unlink", Int32](tmp.as_c_string_span().ptr())
        else:
            _ = external_call["unlink", Int32](
                record.as_c_string_span().ptr()
            )  # never ahead of its .so
            var dst = String(out_path)
            var r = external_call["rename", Int32](
                tmp.as_c_string_span().ptr(), dst.as_c_string_span().ptr()
            )
            if r != 0:
                _ = external_call["unlink", Int32](tmp.as_c_string_span().ptr())
                if not exists(out_path):
                    raise Error("could not install ", out_path)
        var record_tmp = record + ".tmp" + String(perf_counter_ns())
        Path(record_tmp).write_text(warning_diagnostics(log))
        if (
            external_call["rename", Int32](
                record_tmp.as_c_string_span().ptr(),
                record.as_c_string_span().ptr(),
            )
            != 0
        ):
            _ = external_call["unlink", Int32](
                record_tmp.as_c_string_span().ptr()
            )
            raise Error("could not install ", record)
        if self.trace:
            print(
                "[TRACE] built ",
                label,
                " ",
                " ".join(defines),
                " in ",
                Float64(ms) / 1000.0,
                "s",
            )

    def _usable(self, so: String) -> Bool:
        """A cache entry to load as is: present and, under WERROR, with its
        warnings record (a record-less one is unknown, so it is rebuilt
        once). Without WERROR the record is never looked at."""
        return exists(so) and (not _werror() or isfile(so + WARNINGS_SUFFIX))

    def _ensure_built(
        mut self,
        key: String,
        so: String,
        label: String,
        src: String,
        root: String,
        defines: List[String],
    ) raises:
        """Build `so` unless it is already in the cache, once per box: the
        flock makes concurrent processes wait instead of building twice."""
        if self._usable(so):
            return
        if not isdir(self.cache_dir):
            makedirs(self.cache_dir, exist_ok=True)
        var lock_path = self.cache_dir + "/." + key + ".lock"
        var fd = external_call["creat", Int32](
            lock_path.as_c_string_span().ptr(), Int32(0o644)
        )
        if fd >= 0:
            _ = external_call["flock", Int32](
                fd, Int32(2)
            )  # LOCK_EX (best effort: NFS may refuse)

        try:
            if not self._usable(so):
                self._build(label, src, root, defines, so)
        finally:
            if fd >= 0:
                _ = external_call["flock", Int32](fd, Int32(8))  # LOCK_UN
                _ = external_call["close", Int32](fd)

    def entry(
        mut self, family: String, var defines: List[String]
    ) raises -> Int:
        """Address of the family's `tmb_call` for this specialization.

        Appends `_EAGER_ELEMENTWISE_DEFINE`: every eager build goes through
        here, so this is the one place that marks the whole family build,
        rather than plumbing it through each op's `KernelCall`.
        """
        defines.append(_EAGER_ELEMENTWISE_DEFINE)
        var key = family + "." + _slug(defines)
        if key in self.families:
            return self.families[key].entry
        try:
            var so = (
                self.cache_dir
                + "/"
                + key
                + ".hash-"
                + self.source_hash(family)
                + ".so"
            )
            var checked = not _werror()
            # Twice: a record that cannot be read (removed by another
            # process after `_usable`, or unreadable: removed here) is a miss.
            for _ in range(2):
                if not self._usable(so):
                    # A miss compiles a snapshot of the sources and keys the
                    # build by THAT snapshot's hash, never by the memo.
                    var snap = self.snapshot(family)
                    so = self.cache_dir + "/" + key + ".hash-" + snap[1] + ".so"
                    self._ensure_built(
                        key,
                        so,
                        family,
                        snap[0] + "/" + KERNELS + "/" + family + "/entry.mojo",
                        snap[0],
                        defines,
                    )
                if checked:
                    break
                var diagnostics = _recorded(so)
                if diagnostics:
                    _werror_check(so, family, defines, diagnostics.value())
                    checked = True
                    break
                var record = so + WARNINGS_SUFFIX
                _ = external_call["unlink", Int32](
                    record.as_c_string_span().ptr()
                )
            if not checked:
                raise Error("cannot read ", so + WARNINGS_SUFFIX)
            var fam = Family(so)
            var addr = fam.entry
            self.families[key] = fam^
            return addr
        except e:
            raise e^


comptime ERR_CAP = 4096
comptime FamilyFn = def(
    Argv, Int, Pointer[UInt8, MutUntrackedOrigin], Int
) thin abi("C") -> Int32


def invoke_family(entry: Int, argv: Argv, argc: Int) raises:
    """Call a family's C entry with the argument slots. A non-zero return
    carries the kernel's own message (declined input, bad geometry, ...).

    The buffer is left uninitialized but for its first byte: zeroing all 4 KiB
    on a path that runs ~10k times per training step cost more than every
    error message ever read from it.
    """
    var err = Array[UInt8, ERR_CAP](uninitialized=True)
    err[0] = 0
    var f = Pointer(to=entry).unsafe_bitcast[FamilyFn]()[]
    var rc = f(
        argv,
        argc,
        Pointer[UInt8, MutUntrackedOrigin](
            unsafe_from_address=Int(err.unsafe_ptr())
        ),
        ERR_CAP,
    )
    if rc != 0:
        raise Error(String(unsafe_from_utf8_ptr=err.unsafe_ptr()))
