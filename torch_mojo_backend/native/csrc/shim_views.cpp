// ATen's own view kernels, registered for this device.
//
// `view`, `_unsafe_view`, `_reshape_alias` and `as_strided` produce a tensor that shares its
// input's storage and differs only in sizes, strides and storage offset;
// `view_as_real` / `view_as_complex` also reinterpret its dtype (a complex
// element as two real ones), which only a C++ TensorImpl can express.
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
#include <ATen/native/MathBitsFallback.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_as_complex_native.h>
#include <ATen/ops/view_as_real_native.h>
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
  tmb_count_op_call("aten::view");
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW_ALLOC(size));
}

at::Tensor unsafe_view_pu1(const at::Tensor& self, c10::SymIntArrayRef size) {
  tmb_count_op_call("aten::_unsafe_view");
  return at::native::_unsafe_view(self, C10_AS_INTARRAYREF_SLOW_ALLOC(size));
}

at::Tensor reshape_alias_pu1(const at::Tensor& self, c10::SymIntArrayRef size,
                             c10::SymIntArrayRef stride) {
  tmb_count_op_call("aten::_reshape_alias");
  return at::native::_reshape_alias(self, C10_AS_INTARRAYREF_SLOW_ALLOC(size),
                                    C10_AS_INTARRAYREF_SLOW_ALLOC(stride));
}

at::Tensor as_strided_pu1(const at::Tensor& self, c10::SymIntArrayRef size,
                          c10::SymIntArrayRef stride,
                          std::optional<c10::SymInt> storage_offset) {
  tmb_count_op_call("aten::as_strided");
  return at::native::as_strided_tensorimpl(
      self, C10_AS_INTARRAYREF_SLOW_ALLOC(size),
      C10_AS_INTARRAYREF_SLOW_ALLOC(stride),
      storage_offset.has_value()
          ? std::optional<int64_t>(storage_offset->guard_int(__FILE__, __LINE__))
          : std::nullopt);
}

at::Tensor view_as_real_pu1(const at::Tensor& self) {
  tmb_count_op_call("aten::view_as_real");
  return at::native::view_as_real(self);
}

at::Tensor view_as_complex_pu1(const at::Tensor& self) {
  tmb_count_op_call("aten::view_as_complex");
  return at::native::view_as_complex(self);
}

}  // namespace

extern "C" int32_t tmb_library_impl_aten_view(TmbLibrary lib, const char* name) {
  try {
    auto& m = *reinterpret_cast<torch::Library*>(lib);
    const std::string which(name ? name : "");
    if (which == "view") {
      m.impl("view", TORCH_FN(view_pu1));
    } else if (which == "_unsafe_view") {
      m.impl("_unsafe_view", TORCH_FN(unsafe_view_pu1));
    } else if (which == "_reshape_alias") {
      m.impl("_reshape_alias", TORCH_FN(reshape_alias_pu1));
    } else if (which == "as_strided") {
      m.impl("as_strided", TORCH_FN(as_strided_pu1));
    } else if (which == "view_as_real") {
      m.impl("view_as_real", TORCH_FN(view_as_real_pu1));
    } else if (which == "view_as_complex") {
      m.impl("view_as_complex", TORCH_FN(view_as_complex_pu1));
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

// `_copy_from` writes into its `dst` argument, but its schema does not mark
// it mutable, so the Conjugate and Negative keys' boxed fallback hands the
// backend kernel a bit-resolved CLONE of a conj / neg destination
// (`x.conj().copy_(y)`, `x.conj().imag.copy_(y)`, and the write-back of every
// in-place op on such a view) and the write is lost. For a call with a mojo
// tensor, skip the fallback instead: the Mojo `_copy_from` resolves the bits
// on both sides itself (tmb/ops/core.mojo, `_copy_bits`), as `copy_`, on
// the fallback's fallthrough list, already sees them. Any other call
// (another out-of-tree backend's `_copy_from` in the same process) runs
// torch's own fallback, unchanged: this registration is per op and
// process-global, so it must not alter theirs.
namespace {

template <c10::DispatchKey Key>
struct BitFallbackForOthers : at::native::MathOpFallback {
  BitFallbackForOthers()
      : MathOpFallback(Key, Key == c10::DispatchKey::Conjugate ? "conjugate" : "negation") {}
  bool is_bit_set(const at::Tensor& tensor) override {
    return Key == c10::DispatchKey::Conjugate ? tensor.is_conj() : tensor.is_neg();
  }
};

template <c10::DispatchKey Key>
void copy_from_math_bit(const c10::OperatorHandle& op, c10::DispatchKeySet ks,
                        torch::jit::Stack* stack) {
  const auto n = op.schema().arguments().size();
  bool ours = false;
  for (size_t i = stack->size() - n; i < stack->size(); ++i) {
    const auto& v = (*stack)[i];
    if (v.isTensor() && v.toTensor().defined() &&
        v.toTensor().device().type() == c10::DeviceType::PrivateUse1) {
      ours = true;
    }
  }
  if (ours) {
    op.redispatchBoxed(ks & c10::DispatchKeySet(c10::DispatchKeySet::FULL_AFTER, Key), stack);
    return;
  }
  BitFallbackForOthers<Key> fallback;
  fallback.fallback_impl(op, ks, stack);
}

}  // namespace

TORCH_LIBRARY_IMPL(aten, Conjugate, m) {
  m.impl("_copy_from", torch::CppFunction::makeFromBoxedFunction<
                           &copy_from_math_bit<c10::DispatchKey::Conjugate>>());
}

TORCH_LIBRARY_IMPL(aten, Negative, m) {
  m.impl("_copy_from", torch::CppFunction::makeFromBoxedFunction<
                           &copy_from_math_bit<c10::DispatchKey::Negative>>());
}
