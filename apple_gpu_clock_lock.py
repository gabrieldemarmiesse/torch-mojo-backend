"""Apple GPU clock lock for torch-mojo-backend kernel benchmarks.

Adapted from causal-conv1d-mojo scripts/_apple_gpu_clock_lock.py.

macOS has no public API/CLI to pin the GPU's DVFS clock (unlike
``nvidia-smi --lock-gpu-clocks``/``rocm-smi --setperflevel``). Instruments
does have an internal "Induced GPU Performance State" knob, normally only
settable by hand in its GUI. This module reproduces that by binary-patching
a copy of Instruments' ``Metal System Trace.tracetemplate`` (an
NSKeyedArchiver binary plist) and handing the copy to
``xctrace record --template <path>``.

Must patch the raw binary plist directly: ``plutil -convert xml1`` first
loses the info we need (XML has no native UID type, so cross-references
turn into plain dicts), and even loading the binary plist with
``plistlib`` then re-serializing via ``dump(fmt=FMT_BINARY)`` produces a
file ``xctrace export`` rejects with "Document Missing Template Error" —
so ``_RawBPlist`` below patches the target reference in place instead,
leaving the rest of the ~600 KB file byte-identical to Apple's original.

The ``gpuperformancestate`` enum is undocumented; empirically:
0=Automatic, 1=Minimum, 2=Medium, 3=Maximum (verified via
``gpu-performance-state-intervals`` in a recorded trace).

This is a private, undocumented format with no cross-version guarantee.
``locked_template_path`` returns ``None`` on any failure so callers can
fall back; ``master_bench.py`` instead treats failure as fatal (see
``_lock_metal`` there).
"""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
from pathlib import Path

_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"))
_CACHE_DIR = _CACHE_HOME / "torch_mojo_backend" / "xctrace_templates"

STATE_VALUES = {"Automatic": 0, "Minimum": 1, "Medium": 2, "Maximum": 3}
_TARGET_KEY = "gpuperformancestate"


# ---------------------------------------------------------------------------
# Minimal raw bplist reader — just enough to walk an NSKeyedArchiver object
# graph and locate byte offsets, not a general-purpose plist library.
# ---------------------------------------------------------------------------


class _RawBPlist:
    def __init__(self, data: bytes):
        if data[:8] != b"bplist00":
            raise ValueError("not a binary plist")
        self.data = data
        trailer = data[-32:]
        (
            _,
            self.sort_version,
            self.offset_int_size,
            self.object_ref_size,
            self.num_objects,
            self.top_object,
            self.offset_table_offset,
        ) = struct.unpack(">5sBBBQQQ", trailer)
        self.offsets = [
            self._read_uint(
                self.offset_table_offset + i * self.offset_int_size,
                self.offset_int_size,
            )
            for i in range(self.num_objects)
        ]

    def _read_uint(self, offset: int, size: int) -> int:
        return int.from_bytes(self.data[offset : offset + size], "big")

    def _read_len(self, offset: int, nibble: int) -> tuple[int, int]:
        """Returns (count, header_len) for a marker whose low nibble encodes size."""
        if nibble != 0xF:
            return nibble, 1
        int_marker = self.data[offset + 1]
        int_nbytes = 1 << (int_marker & 0x0F)
        count = self._read_uint(offset + 2, int_nbytes)
        return count, 2 + int_nbytes

    def object_info(self, index: int) -> dict:
        offset = self.offsets[index]
        marker = self.data[offset]
        kind = marker & 0xF0
        nibble = marker & 0x0F
        if kind == 0x00:
            if marker == 0x08:
                return {"type": "bool", "value": False}
            if marker == 0x09:
                return {"type": "bool", "value": True}
            return {"type": "null", "value": None}
        if kind == 0x10:  # int
            nbytes = 1 << nibble
            val = int.from_bytes(
                self.data[offset + 1 : offset + 1 + nbytes], "big", signed=(nbytes >= 8)
            )
            return {"type": "int", "value": val}
        if kind == 0x20:  # real (unused by our search, but must not crash on it)
            return {"type": "real", "value": None}
        if kind == 0x30:  # date (ditto)
            return {"type": "date", "value": None}
        if kind == 0x40:  # data (ditto)
            return {"type": "data", "value": None}
        if kind == 0x50:  # ASCII string
            count, hlen = self._read_len(offset, nibble)
            s = self.data[offset + hlen : offset + hlen + count].decode("ascii")
            return {"type": "string", "value": s}
        if kind == 0x60:  # UTF-16 string
            count, hlen = self._read_len(offset, nibble)
            s = self.data[offset + hlen : offset + hlen + count * 2].decode("utf-16-be")
            return {"type": "string", "value": s}
        if kind == 0x80:  # UID (NSKeyedArchiver cross-reference)
            nbytes = nibble + 1
            val = self._read_uint(offset + 1, nbytes)
            return {"type": "uid", "value": val}
        if kind == 0xA0:  # array
            count, hlen = self._read_len(offset, nibble)
            refs_off = offset + hlen
            refs = [
                self._read_uint(
                    refs_off + i * self.object_ref_size, self.object_ref_size
                )
                for i in range(count)
            ]
            return {
                "type": "array",
                "refs_offset": refs_off,
                "count": count,
                "refs": refs,
            }
        if kind == 0xD0:  # dict
            count, hlen = self._read_len(offset, nibble)
            keys_off = offset + hlen
            vals_off = keys_off + count * self.object_ref_size
            key_refs = [
                self._read_uint(
                    keys_off + i * self.object_ref_size, self.object_ref_size
                )
                for i in range(count)
            ]
            val_refs = [
                self._read_uint(
                    vals_off + i * self.object_ref_size, self.object_ref_size
                )
                for i in range(count)
            ]
            return {"type": "dict", "key_refs": key_refs, "val_refs": val_refs}
        raise ValueError(f"unhandled bplist marker 0x{marker:02x} at object {index}")

    def resolve(self, raw_ref: int, objects_arr: dict) -> dict:
        """Dereference one NSKeyedArchiver ``uid`` indirection hop, if present."""
        info = self.object_info(raw_ref)
        if info["type"] == "uid":
            return self.object_info(objects_arr["refs"][info["value"]])
        return info


# ---------------------------------------------------------------------------
# Patch logic.
# ---------------------------------------------------------------------------


def _instruments_app() -> Path | None:
    dev_dir = subprocess.run(
        ["xcode-select", "-p"], capture_output=True, text=True
    ).stdout.strip()
    if not dev_dir:
        return None
    # <Xcode.app>/Contents/Developer -> <Xcode.app>/Contents/Applications/Instruments.app
    app = Path(dev_dir).parent / "Applications" / "Instruments.app"
    return app if app.exists() else None


def _find_source_template() -> Path | None:
    app = _instruments_app()
    if app is None:
        return None
    matches = list(app.rglob("Metal System Trace.tracetemplate"))
    return matches[0] if matches else None


def _find_dict_with_key(bp: _RawBPlist, objects_arr: dict, key_name: str):
    """Search every logical $objects entry for a dict whose NS.keys array
    (each element one uid-hop away) contains `key_name`. Returns
    (ns_keys_info, ns_objects_info, slot) or None.
    """
    for raw in objects_arr["refs"]:
        info = bp.object_info(raw)
        if info["type"] != "dict":
            continue
        names = [bp.object_info(k).get("value") for k in info["key_refs"]]
        if "NS.keys" not in names or "NS.objects" not in names:
            continue
        ns_keys = bp.object_info(info["val_refs"][names.index("NS.keys")])
        if ns_keys["type"] != "array":
            continue
        key_values = [bp.resolve(r, objects_arr).get("value") for r in ns_keys["refs"]]
        if key_name not in key_values:
            continue
        ns_objects = bp.object_info(info["val_refs"][names.index("NS.objects")])
        return ns_keys, ns_objects, key_values.index(key_name)
    return None


def _find_uid_wrapper_for_int(
    bp: _RawBPlist, objects_arr: dict, target_int: int
) -> int | None:
    logical_idx = None
    for k, raw in enumerate(objects_arr["refs"]):
        info = bp.object_info(raw)
        if info["type"] == "int" and info["value"] == target_int:
            logical_idx = k
            break
    if logical_idx is None:
        return None
    for i in range(bp.num_objects):
        info = bp.object_info(i)
        if info["type"] == "uid" and info["value"] == logical_idx:
            return i
    return None


def _patch(source: bytes, target_int: int) -> bytes:
    data = bytearray(source)
    bp = _RawBPlist(bytes(data))
    top = bp.object_info(bp.top_object)
    top_keys = [bp.object_info(k).get("value") for k in top["key_refs"]]
    objects_arr = bp.object_info(top["val_refs"][top_keys.index("$objects")])

    found = _find_dict_with_key(bp, objects_arr, _TARGET_KEY)
    if found is None:
        raise ValueError(f"{_TARGET_KEY!r} not found in template's object graph")
    _, ns_objects, slot = found

    new_wrapper = _find_uid_wrapper_for_int(bp, objects_arr, target_int)
    if new_wrapper is None:
        raise ValueError(f"no existing object holds int {target_int} to repoint at")

    slot_off = ns_objects["refs_offset"] + slot * bp.object_ref_size
    data[slot_off : slot_off + bp.object_ref_size] = new_wrapper.to_bytes(
        bp.object_ref_size, "big"
    )

    # Sanity check: re-parse the patched bytes and confirm the new value reads back.
    check = _RawBPlist(bytes(data))
    reparsed = _find_dict_with_key(check, objects_arr, _TARGET_KEY)
    if reparsed is None:
        raise ValueError("post-patch re-parse lost the target key")
    _, check_ns_objects, check_slot = reparsed
    got = check.resolve(check_ns_objects["refs"][check_slot], objects_arr).get("value")
    if got != target_int:
        raise ValueError(
            f"post-patch verification mismatch: read back {got}, expected {target_int}"
        )

    return bytes(data)


def _locked_template_path_or_raise(state: str) -> Path:
    """Does the real work; raises with a specific reason on any failure.
    See `locked_template_path` for the exception-swallowing public wrapper.
    """
    target_int = STATE_VALUES.get(state)
    if target_int is None:
        raise ValueError(
            f"unknown state {state!r}; choose one of {sorted(STATE_VALUES)}"
        )

    source = _find_source_template()
    if source is None:
        raise FileNotFoundError(
            "Metal System Trace.tracetemplate not found under Instruments.app "
            "(no Xcode install, or `xcode-select -p` points elsewhere)"
        )

    source_bytes = source.read_bytes()
    key_material = source_bytes + Path(__file__).read_bytes() + state.encode()
    digest = hashlib.sha256(key_material).hexdigest()[:16]
    cached = _CACHE_DIR / f"metal-system-trace.{state.lower()}.{digest}.tracetemplate"
    if cached.exists():
        return cached

    patched = _patch(source_bytes, target_int)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = cached.with_suffix(".tmp")
    tmp.write_bytes(patched)
    tmp.replace(cached)
    # Clear stale entries for this state (different Xcode/script version).
    for old in _CACHE_DIR.glob(f"metal-system-trace.{state.lower()}.*.tracetemplate"):
        if old != cached:
            old.unlink(missing_ok=True)
    return cached


def locked_template_path(state: str = "Maximum") -> Path | None:
    """Path to an Instruments template that forces the GPU's Induced
    Performance State to `state` for the whole recording, or None if the
    patch isn't available on this machine (missing Xcode, unexpected
    template layout, etc — caller must fall back to unlocked).

    Cached under `$XDG_CACHE_HOME/torch_mojo_backend/xctrace_templates/`,
    content-addressed on the source template + this script + the target
    state, so an Xcode update or a fix to this patcher invalidates
    automatically.
    """
    try:
        return _locked_template_path_or_raise(state)
    except Exception:
        return None


if __name__ == "__main__":
    import sys

    state = sys.argv[1] if len(sys.argv) > 1 else "Maximum"
    try:
        path = _locked_template_path_or_raise(state)
    except Exception as e:
        print(f"could not build a {state!r}-locked template: {e!r}", file=sys.stderr)
        raise SystemExit(1) from None
    print(path)
