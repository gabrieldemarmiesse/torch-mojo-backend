"""On-demand builds, from Mojo, of everything that is not the runtime.

Two kinds of library, built the same way: hash the entry file's import
closure, look the .so up in the cache, otherwise run `mojo build` in a
subprocess (under a cross-process flock, atomic rename), dlopen it and keep
the entry pointer.

* **kernel families** (torch_mojo_backend/eager_kernels/<family>/<family>.mojo)
  export `tmb_call` and compile one (OP, dtypes, flags) specialization per
  build, selected with -D defines.
* **op extensions** (native/mojo/ops_<group>.mojo, `-D TMB_OP=<aten name>`)
  export `tmb_op_address` and hold the body of exactly one aten op, so the
  backend library itself carries no op code — see registry.mojo.
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

from op_utils import Argv


comptime CACHE_ABI = "native-v1"

# Package directories beside eager_kernels that also hold kernel families
# (see `Loader.family_dir`).
comptime SIBLING_ROOTS = ["eager_flash_attention"]


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
    var tmp = getenv("TMPDIR")
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


comptime AddressFn = def() thin abi("C") -> Int


struct OpExt(Movable):
    """One aten op's extension: `tmb_op_address` reports the address of the
    boxed entry of the op the build selected."""

    var lib: OwnedDLHandle
    var entry: Int

    def __init__(out self, path: String) raises:
        self.lib = OwnedDLHandle(path)
        var sym = self.lib.get_symbol[NoneType]("tmb_op_address")
        if not sym:
            raise Error("tmb_op_address not exported by ", path)
        var addr = Int(sym.value())
        var f = Pointer(to=addr).unsafe_bitcast[AddressFn]()[]
        self.entry = f()
        if self.entry == 0:
            raise Error("no op compiled into ", path)


struct Loader(Movable):
    var kernels_dir: String  # torch_mojo_backend/eager_kernels
    var mojo_dir: String  # torch_mojo_backend/native/mojo
    var cache_dir: String  # <kernels_dir>/__mojocache__/native
    var mojo_exe: String
    var toolchain: String  # versions of mojo/max/python, from the Python side
    var trace: Bool
    var families: Dict[String, Family]  # "<family>.<slug>" -> loaded build
    var ops: Dict[String, OpExt]  # "<group>/<aten name>" -> loaded build
    var source_hashes: Dict[String, String]  # family -> closure hash
    var fast: Dict[
        UInt64, Int
    ]  # (family, defines) hash -> entry address (hot path)

    def __init__(
        out self,
        kernels_dir: String,
        mojo_dir: String,
        cache_dir: String,
        mojo_exe: String,
        toolchain: String,
        trace: Bool,
    ):
        self.kernels_dir = kernels_dir
        self.mojo_dir = mojo_dir
        self.cache_dir = cache_dir
        self.mojo_exe = mojo_exe
        self.toolchain = toolchain
        self.trace = trace
        self.families = Dict[String, Family]()
        self.ops = Dict[String, OpExt]()
        self.source_hashes = Dict[String, String]()
        self.fast = Dict[UInt64, Int]()

    def family_dir(self, family: String) raises -> String:
        """The directory holding `<family>.mojo`.

        Normally `<kernels_dir>/<family>`. The vendored FlashAttention-4
        package is the one family that lives in its own directory beside
        eager_kernels (so a FA4 change does not rehash every ordinary
        family), so a family whose entry file is not under the kernels root
        is looked up in the sibling roots below."""
        var here = self.kernels_dir + "/" + family
        if exists(here + "/" + family + ".mojo"):
            return here
        comptime for root in SIBLING_ROOTS:
            var d = self.kernels_dir + "/../" + String(root)
            if exists(d + "/" + family + ".mojo"):
                return d
        return here

    def _closure(self, entry: String, own_dir: String) raises -> List[String]:
        """Every .mojo file `entry` reaches through `from X import` /
        `import X` (resolved in `own_dir`, then the package root), plus every
        op_utils/*.mojo, in a deterministic order."""
        var fam_dir = own_dir
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
            var text = Path(f).read_text()
            for line in text.splitlines():
                var s = String(line)
                var name = String()
                if s.startswith("from "):
                    var rest = String(s[byte=5:])
                    var sp = rest.find(" ")
                    name = String(rest[byte=:sp]) if sp > 0 else rest
                elif s.startswith("import "):
                    name = String(String(s[byte=7:]).strip())
                else:
                    continue
                var dot = name.find(".")
                if dot > 0:
                    var short = String(name[byte=:dot])
                    name = short^
                if (
                    name == ""
                    or name == "std"
                    or name == "max"
                    or name == "nn"
                    or name == "linalg"
                    or name == "layout"
                ):
                    continue
                var cand = fam_dir + "/" + name + ".mojo"
                if not exists(cand):
                    cand = self.kernels_dir + "/" + name + ".mojo"
                if exists(cand) and cand not in seen:
                    todo.append(cand)
        var op_utils = self.kernels_dir + "/op_utils"
        if isdir(op_utils):
            var names = List[String]()
            for p in Path(op_utils).listdir():
                var ps = String(p)
                if ps.endswith(".mojo"):
                    names.append(op_utils + "/" + ps)
            sort(names)
            for n in names:
                if n not in seen:
                    files.append(n)
        sort(files)
        return files^

    def _hash_closure(
        mut self, key: String, entry: String, own_dir: String, strip: String
    ) raises -> String:
        """Cache key of one build: every source it compiles in, named
        relative to `strip`, plus the toolchain — so touching any of them
        invalidates it."""
        if key in self.source_hashes:
            return self.source_hashes[key]
        var h: UInt64 = 14695981039346656037
        _fnv1a(h, CACHE_ABI.as_bytes())
        _fnv1a(h, self.toolchain.as_bytes())
        for f in self._closure(entry, own_dir):
            var rel = String(f[byte = strip.byte_length() :])
            _fnv1a(h, rel.as_bytes())
            var bytes = Path(f).read_bytes()
            _fnv1a(h, Span(bytes))
        var out = _hex(h)
        self.source_hashes[key] = out
        return out

    def source_hash(mut self, family: String) raises -> String:
        var fam_dir = self.family_dir(family)
        return self._hash_closure(
            "family:" + family,
            fam_dir + "/" + family + ".mojo",
            fam_dir,
            String(self.kernels_dir),
        )

    def op_source_hash(mut self, group: String) raises -> String:
        """Hashes abi/device/kernels/ops_common too (the group file imports
        them), so a runtime change invalidates every op extension."""
        return self._hash_closure(
            "op:" + group,
            self.mojo_dir + "/" + group + ".mojo",
            String(self.mojo_dir),
            String(self.mojo_dir),
        )

    def _build(
        mut self,
        label: String,
        src: String,
        own_dir: String,
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
            + own_dir
            + "' -I '"
            + self.kernels_dir
            + "'"
        )
        comptime if CompilationTarget.is_macos():
            # Op extensions call the shim and the base library, resolved at
            # dlopen; ld64 wants to be told so.
            cmd += " -Xlinker -undefined -Xlinker dynamic_lookup"
        for d in defines:
            cmd += " -D '" + d + "'"
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
                _ = external_call["unlink", Int32](
                    tmp.as_c_string_slice().unsafe_ptr()
                )
            raise Error(
                "mojo build of ",
                label,
                " failed (rc ",
                rc,
                ", ",
                ms,
                " ms):\n",
                String(output[byte=:marker]) if marker > 0 else output,
            )
        if exists(
            out_path
        ):  # another process installed the same build first: use theirs
            _ = external_call["unlink", Int32](
                tmp.as_c_string_slice().unsafe_ptr()
            )
        else:
            var dst = String(out_path)
            var r = external_call["rename", Int32](
                tmp.as_c_string_slice().unsafe_ptr(),
                dst.as_c_string_slice().unsafe_ptr(),
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
        own_dir: String,
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
            lock_path.as_c_string_slice().unsafe_ptr(), Int32(0o644)
        )
        if fd >= 0:
            _ = external_call["flock", Int32](
                fd, Int32(2)
            )  # LOCK_EX (best effort: NFS may refuse)

        try:
            if not exists(so):
                self._build(label, src, own_dir, defines, so)
        finally:
            if fd >= 0:
                _ = external_call["flock", Int32](fd, Int32(8))  # LOCK_UN
                _ = external_call["close", Int32](fd)

    def op_entry(mut self, group: String, name: String) raises -> Int:
        """Address of the boxed entry of aten::<name>, compiling its
        extension from native/mojo/<group>.mojo on the first call.

        A failure is not remembered: the next call retries, so a compiler
        that failed on a full disk or a killed subprocess is not fatal for
        the rest of the process."""
        var key = group + "/" + name
        if key in self.ops:
            return self.ops[key].entry
        var so = (
            self.cache_dir
            + "/tmbop."
            + group
            + "."
            + name
            + ".hash-"
            + self.op_source_hash(group)
            + ".so"
        )
        var defines = List[String]()
        defines.append("TMB_OP=" + name)
        self._ensure_built(
            "tmbop." + group + "." + name,
            so,
            group + " " + name,
            self.mojo_dir + "/" + group + ".mojo",
            String(self.mojo_dir),
            defines,
        )
        var ext = OpExt(so)
        var addr = ext.entry
        self.ops[key] = ext^
        return addr

    def entry(mut self, family: String, defines: List[String]) raises -> Int:
        """Address of the family's `tmb_call` for this specialization."""
        var key = family + "." + _slug(defines)
        if key in self.families:
            return self.families[key].entry
        try:
            var fam_dir = self.family_dir(family)
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
                fam_dir + "/" + family + ".mojo",
                fam_dir,
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


def call_family(
    mut loader: Loader,
    family: String,
    key: UInt64,
    defines: List[String],
    argv: Argv,
    argc: Int,
) raises:
    """Run one kernel: build/load the specialization on first use, then call
    its C entry with the argument slots. `key` hashes (family, defines) so a
    warm call is one dictionary probe; `defines` is only read on a miss.
    A non-zero return carries the kernel's own message (declined input, bad
    geometry, ...)."""
    var entry: Int
    var hit = loader.fast.find(key)
    if hit:
        entry = hit.value()
    else:
        entry = loader.entry(family, defines)
        loader.fast[key] = entry
    var err = InlineArray[UInt8, ERR_CAP](fill=0)
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
