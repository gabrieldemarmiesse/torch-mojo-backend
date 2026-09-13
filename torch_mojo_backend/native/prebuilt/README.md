# Prebuilt native libraries

The wheel ships the two fixed libraries of the native device here, so a fresh
install does not compile them on its first `register_mojo_devices()`:

* `libtmb_shim-torch<major.minor>-<platform>-<machine>-cxx11abi<0|1>.so|.dylib`
  — the C++ shim, one file per torch series, platform and libstdc++ ABI flag;
* `libtmb_backend-max<version>-<platform>-<machine>.so|.dylib`
  — the Mojo base library, one file per MAX version and platform (it holds no
  device code and makes no compile-time accelerator choice).

`manifest.json` says what each file was built from. `torch_mojo_backend.native`
uses a file only when its manifest entry matches the running environment
*and* the hash of the sources shipped beside it; otherwise it compiles, as it
always did. `TORCH_MOJO_BACKEND_PREBUILT=0` ignores this directory entirely.

Everything but this README is a build artefact: git ignores it, and
`scripts/build_prebuilt.py` (or the `wheel.yml` workflow) writes it. See
`docs/native_backend.md`, "Prebuilt libraries and the wheel".
