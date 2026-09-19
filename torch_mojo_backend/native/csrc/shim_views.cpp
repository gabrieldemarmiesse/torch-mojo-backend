// ATen's own view kernels, registered for this device.
//
// `view`, `_reshape_alias` and `as_strided` produce a tensor that shares its
// input's storage and differs only in sizes, strides and storage offset.
// There is no device code in them: ATen's kernels are what CPU, CUDA, MPS and
// XPU all register (native_functions.yaml), and they are what this file
// registers for PrivateUse1 -- unboxed, so a call skips the IValue boxing a
// boxed-only kernel forces on every argument (a `SymInt[]` becomes a
// heap-allocated c10::List per call) and the Mojo round trip behind it.
// Measured on an H100: 0.4 us per call.
//
// No op logic lives here. The bodies are ATen's, reached through the per-op
// native headers, and the only code is the SymInt-to-int conversion this
// boundary owes them.
#include "tmb_internal.h"

#include <ATen/core/Tensor.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <c10/core/SymIntArrayRef.h>
#include <torch/library.h>

#include <optional>
#include <string>

namespace {

// Every SymInt is GUARDED here (`guard_int`, `C10_AS_INTARRAYREF_SLOW_ALLOC`),
// never merely expected: a backed symbolic size specializes on the way in, as
// the boxed adapter these kernels replaced did and as torch's codegen does for
// a scalar SymInt argument. `C10_AS_INTARRAYREF_SLOW` throws on any symbolic
// element instead. The DimVector holds five dimensions inline, so a guarded
// size costs no allocation for the ranks that occur.
at::Tensor view_pu1(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW_ALLOC(size));
}

at::Tensor reshape_alias_pu1(const at::Tensor& self, c10::SymIntArrayRef size,
                             c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(self, C10_AS_INTARRAYREF_SLOW_ALLOC(size),
                                    C10_AS_INTARRAYREF_SLOW_ALLOC(stride));
}

at::Tensor as_strided_pu1(const at::Tensor& self, c10::SymIntArrayRef size,
                          c10::SymIntArrayRef stride,
                          std::optional<c10::SymInt> storage_offset) {
  return at::native::as_strided_tensorimpl(
      self, C10_AS_INTARRAYREF_SLOW_ALLOC(size),
      C10_AS_INTARRAYREF_SLOW_ALLOC(stride),
      storage_offset.has_value()
          ? std::optional<int64_t>(storage_offset->guard_int(__FILE__, __LINE__))
          : std::nullopt);
}

}  // namespace

extern "C" int32_t tmb_library_impl_aten_view(TmbLibrary lib, const char* name) {
  try {
    auto& m = *reinterpret_cast<torch::Library*>(lib);
    const std::string which(name ? name : "");
    if (which == "view") {
      m.impl("view", TORCH_FN(view_pu1));
    } else if (which == "_reshape_alias") {
      m.impl("_reshape_alias", TORCH_FN(reshape_alias_pu1));
    } else if (which == "as_strided") {
      m.impl("as_strided", TORCH_FN(as_strided_pu1));
    } else {
      tmb_set_error(("mojo backend: no ATen view kernel named " + which).c_str());
      return 1;
    }
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}
