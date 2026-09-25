"""On-demand builds, from Mojo, of everything that is not the runtime.

Kernel families (torch_mojo_backend/mojo/tmb/kernels/<family>/entry.mojo)
export `tmb_call` and compile one (OP, dtypes, flags) specialization per
build, selected with -D defines. A build hashes the entry file's import
closure, looks the .so up in the cache, otherwise runs `mojo build` in a
subprocess (under a cross-process flock, atomic rename), dlopens it and
keeps the entry pointer.
"""
from std.builtin.sort import sort
from std.collections import Dict
from std.ffi import OwnedDLHandle, external_call
from std.os import getenv, makedirs
from std.os.path import exists, isdir
from std.pathlib import Path
from std.subprocess import run as run_command
from std.sys.info import CompilationTarget
from std.time import perf_counter_ns

from tmb.backend.env_vars import TMPDIR, TORCH_MOJO_BACKEND_WERROR
from tmb.kernels.common.op_utils import Argv


comptime CACHE_ABI = "native-v2"

# Every Mojo source lives under one root, torch_mojo_backend/mojo, which is
# the one `-I` of every build, as one top-level package: `from
# tmb.<pkg>.<module> import ...`. Kernel families are
# `tmb/kernels/<family>/entry.mojo`.
comptime PACKAGE = "tmb"
comptime KERNELS = "tmb/kernels"


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
    wins. Why MODULAR_HOME must be node-local: native/__init__.py's
    `compiler_env`, the same thing on the Python side."""
    # PYTHONEXECUTABLE/PYTHONHOME: the MAX runtime exports the interpreter it
    # found on PATH into this process's environment (invisible to os.environ,
    # inherited by children); with a venv that is not on PATH, the `mojo`
    # launcher script then starts /usr/bin/python3 with the venv's prefix and
    # dies with "Could not find platform independent libraries".
    return (
        'unset PYTHONEXECUTABLE PYTHONHOME; MODULAR_HOME="${MODULAR_HOME:-'
        + _local_dir("modular-home-")
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

    def _closure(self, entry: String) raises -> List[String]:
        """Every .mojo file `entry` reaches through `from X import` /
        `import X`, in a deterministic order: the sources one build compiles
        in, so touching any of them invalidates it. native/__init__.py's
        `mojo_import_closure` is the same walk for the Python-driven
        builds."""
        var files = List[String]()
        var seen = Dict[String, Bool]()
        var todo = List[String]()
        todo.append(entry)
        while len(todo) > 0:
            var f = todo.pop()
            if f in seen:
                continue
            seen[f] = True
            files.append(f)
            var here = String(f[byte = : f.rfind("/")])
            var text = Path(f).read_text()
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
                if cand != "" and cand not in seen:
                    todo.append(cand)
        sort(files)
        return files^

    def _hash_closure(mut self, key: String, entry: String) raises -> String:
        """Cache key of one build: every source it compiles in, named
        relative to the root, plus the toolchain -- so touching any of them
        invalidates it."""
        if key in self.source_hashes:
            return self.source_hashes[key]
        var h: UInt64 = 14695981039346656037
        _fnv1a(h, CACHE_ABI.as_bytes())
        _fnv1a(h, self.toolchain.as_bytes())
        for f in self._closure(entry):
            var rel = String(f[byte = self.root.byte_length() :])
            _fnv1a(h, rel.as_bytes())
            var bytes = Path(f).read_bytes()
            _fnv1a(h, Span(bytes))
        var out = _hex(h)
        self.source_hashes[key] = out
        return out

    def source_hash(mut self, family: String) raises -> String:
        return self._hash_closure(
            "family:" + family, self.family_dir(family) + "/entry.mojo"
        )

    def _build(
        mut self,
        label: String,
        src: String,
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
        # MODULAR_HOME: the compiler's own cache goes to local scratch too
        # (native/__init__.py compiler_env explains why)
        var cmd = (
            _compiler_env()
            + " '"
            + self.mojo_exe
            + "' build '"
            + src
            + "' --emit shared-lib -I '"
            + self.root
            + "'"
        )
        comptime if CompilationTarget.is_macos():
            # Kernel families call the shim and the base library, resolved
            # at dlopen; ld64 wants to be told so.
            cmd += " -Xlinker -undefined -Xlinker dynamic_lookup"
        for d in defines:
            cmd += " -D '" + d + "'"
        # TORCH_MOJO_BACKEND_WERROR=1: warnings fail the build (off by
        # default, on under pytest); native/__init__.py's
        # mojo_diagnostic_flags is the same switch for the Python-driven
        # builds.
        if getenv(TORCH_MOJO_BACKEND_WERROR) == "1":
            cmd += " --Werror"
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
        if rc != 0:
            if exists(tmp):
                _ = external_call["unlink", Int32](tmp.as_c_string_span().ptr())
            var defs = String()
            for d in defines:
                defs += " -D " + d
            var log = String(output[byte=:marker]) if marker > 0 else output
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
                defs,
                " failed (rc ",
                rc,
                ", ",
                ms,
                " ms):\n",
                log,
                hint,
            )
        if exists(
            out_path
        ):  # another process installed the same build first: use theirs
            _ = external_call["unlink", Int32](tmp.as_c_string_span().ptr())
        else:
            var dst = String(out_path)
            var r = external_call["rename", Int32](
                tmp.as_c_string_span().ptr(),
                dst.as_c_string_span().ptr(),
            )
            if r != 0 and not exists(out_path):
                raise Error("could not install ", out_path)
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

    def _ensure_built(
        mut self,
        key: String,
        so: String,
        label: String,
        src: String,
        defines: List[String],
    ) raises:
        """Build `so` unless it is already in the cache, once per box: the
        flock makes concurrent processes wait instead of building twice."""
        if exists(so):
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
            if not exists(so):
                self._build(label, src, defines, so)
        finally:
            if fd >= 0:
                _ = external_call["flock", Int32](fd, Int32(8))  # LOCK_UN
                _ = external_call["close", Int32](fd)

    def entry(mut self, family: String, defines: List[String]) raises -> Int:
        """Address of the family's `tmb_call` for this specialization."""
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
            self._ensure_built(
                key,
                so,
                family,
                self.family_dir(family) + "/entry.mojo",
                defines,
            )
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
