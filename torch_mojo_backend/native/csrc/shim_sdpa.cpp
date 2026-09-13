// F.scaled_dot_product_attention picks its backend through the
// _fused_sdp_choice_stub DispatchStub, not through the aten::_fused_sdp_choice
// op, so a PrivateUse1 kernel for the op alone never runs: the composite
// always chose math. This registers the stub for PrivateUse1 and forwards to
// the op (which dispatches to the Mojo registration), falling back to math
// when no Mojo kernel is registered for it.
#include <ATen/SDPBackend.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/ops/_fused_sdp_choice.h>

namespace at::native {
namespace {

int64_t tmb_sdp_choice(const Tensor& query, const Tensor& key, const Tensor& value,
                       const std::optional<Tensor>& attn_mask, double dropout_p, bool is_causal,
                       std::optional<double> scale, bool enable_gqa) {
  static const bool registered = c10::Dispatcher::singleton()
      .findSchemaOrThrow("aten::_fused_sdp_choice", "")
      .hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1);
  if (!registered) return static_cast<int64_t>(at::SDPBackend::math);
  return at::_fused_sdp_choice(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa);
}

}  // namespace

REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &tmb_sdp_choice)

}  // namespace at::native
