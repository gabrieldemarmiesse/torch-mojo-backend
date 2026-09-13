// AutocastPrivateUse1 as ONE boxed fallback with a policy table, instead of
// torch's per-op unboxed WrapFunction templates (which take tens of seconds to
// compile). The policies and the op lists are torch's own CUDA ones.
#include "tmb_internal.h"

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DispatchKeySet.h>
#include <torch/library.h>

#include <mutex>
#include <string>
#include <unordered_map>

// ATen/autocast_mode.h pulls in ATen/ATen.h (seconds of parsing); the two
// exported functions this file needs are declared here instead, and the
// per-op policy lists come from a table the Python builder generates by
// reading that header's AT_FORALL_* macros (tmb_autocast_policies.inc).
namespace at::autocast {
TORCH_API Tensor cached_cast(at::ScalarType to_type, const Tensor& arg, c10::DeviceType device_type);
TORCH_API at::ScalarType get_autocast_dtype(at::DeviceType device_type);
}  // namespace at::autocast

namespace {

enum Policy : int32_t {
  NONE = 0,
  LOWER_PRECISION_FP = 1,
  FP32 = 2,
  FP32_SET_OPT_DTYPE = 3,
  PROMOTE = 4,
  BANNED = 5,
  FP32_APPEND_DTYPE = 6,
};

std::mutex g_policy_mutex;
std::unordered_map<std::string, int32_t> g_policies;
// FP32_APPEND_DTYPE only: the overload of the SAME op that takes the appended
// ScalarType ("aten::norm.Scalar" -> "ScalarOpt_dtype").
std::unordered_map<std::string, std::string> g_append_dtype_targets;
constexpr auto kDevice = c10::DeviceType::PrivateUse1;

// Exact overload keys, like torch's own registrations: "aten::mm" names
// the default overload only, never mm.out.
std::string policy_key(const c10::FunctionSchema& schema) {
  const auto& overload = schema.overload_name();
  return overload.empty() ? schema.name() : schema.name() + "." + overload;
}

int32_t lookup_policy(const std::string& key) {
  std::lock_guard<std::mutex> g(g_policy_mutex);
  auto it = g_policies.find(key);
  return it == g_policies.end() ? NONE : it->second;
}

std::string lookup_append_dtype_target(const std::string& key) {
  std::lock_guard<std::mutex> g(g_policy_mutex);
  auto it = g_append_dtype_targets.find(key);
  return it == g_append_dtype_targets.end() ? std::string() : it->second;
}

at::ScalarType lower_precision_fp() { return at::autocast::get_autocast_dtype(kDevice); }

// at::autocast::is_eligible for the PrivateUse1 device (inline in the header).
bool tensor_eligible(const at::Tensor& t) {
  return t.defined() && t.device().type() == kDevice && t.is_floating_point() && t.scalar_type() != at::kDouble;
}

bool eligible(const c10::IValue& v) { return v.isTensor() && tensor_eligible(v.toTensor()); }

// at::autocast::prioritize: widen toward float32, ignore doubles.
at::ScalarType prioritize(at::ScalarType current, const at::Tensor& next_arg) {
  TORCH_CHECK(current != at::kDouble, "promote type is double in autocast prioritize");
  if (!tensor_eligible(next_arg)) return current;
  const auto next = next_arg.scalar_type();
  if (next == at::kDouble) return current;
  if (current == at::kFloat || next == at::kFloat) return at::kFloat;
  TORCH_CHECK(current == lower_precision_fp() && next == lower_precision_fp(),
              "unexpected floating ScalarType in autocast prioritize");
  return current;
}

// The first Tensor argument decides fp32_set_opt_dtype, as in torch's CUDA kernels.
bool first_arg_eligible(const c10::IValue* args, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (args[i].isTensor()) return eligible(args[i]);
    if (args[i].isTensorList()) {
      auto list = args[i].toTensorList();
      return list.size() > 0 && tensor_eligible(list[0]);
    }
  }
  return false;
}

c10::IValue cast_value(const c10::IValue& v, c10::ScalarType to) {
  if (v.isTensor()) return at::autocast::cached_cast(to, v.toTensor(), kDevice);
  if (v.isTensorList()) {
    auto in = v.toTensorList();
    c10::List<at::Tensor> out;
    out.reserve(in.size());
    for (const at::Tensor t : in) out.push_back(at::autocast::cached_cast(to, t, kDevice));
    return c10::IValue(std::move(out));
  }
  if (v.isOptionalTensorList()) {
    auto in = v.toOptionalTensorList();
    c10::List<std::optional<at::Tensor>> out;
    out.reserve(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
      std::optional<at::Tensor> t = in.get(i);
      out.push_back(t ? std::optional<at::Tensor>(at::autocast::cached_cast(to, *t, kDevice)) : std::nullopt);
    }
    return c10::IValue(std::move(out));
  }
  return v;
}

// at::autocast::type_from_firstarg for CastPolicy::fp32_append_dtype: the
// dtype to append is float32 when the first Tensor argument is eligible, and
// otherwise that tensor's own dtype (so a float64 or an integral input is not
// silently narrowed).
at::ScalarType append_dtype_for(const c10::IValue* args, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (args[i].isTensor()) {
      const auto& t = args[i].toTensor();
      return tensor_eligible(t) ? at::kFloat : t.scalar_type();
    }
  }
  return at::kFloat;
}

void autocast_fallback(const c10::OperatorHandle& op, c10::DispatchKeySet ks, torch::jit::Stack* stack) {
  const auto& schema = op.schema();
  const std::string key = policy_key(schema);
  const int32_t policy = lookup_policy(key);
  c10::impl::ExcludeDispatchKeyGuard no_autocast(c10::DispatchKey::AutocastPrivateUse1);
  if (policy == FP32_APPEND_DTYPE) {
    // AT_FORALL_DIFFERENT_REDISPATCH_SIGNATURE (CUDA's norm overloads):
    // append the result dtype and redispatch to the overload of the same op
    // that takes one, instead of casting the inputs. The target's schema is
    // this one's arguments followed by the ScalarType, so pushing it onto
    // the stack is the whole translation.
    const std::string target = lookup_append_dtype_target(key);
    TORCH_CHECK(!target.empty(), "mojo autocast: no append-dtype redispatch target for ", key);
    const size_t n = schema.arguments().size();
    const at::ScalarType to = append_dtype_for(stack->data() + (stack->size() - n), n);
    auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(schema.name().c_str(), target.c_str());
    stack->push_back(c10::IValue(to));
    handle.callBoxed(*stack);
    return;
  }
  if (policy != NONE) {
    const size_t n = schema.arguments().size();
    c10::IValue* args = stack->data() + (stack->size() - n);
    if (policy == BANNED) {
      TORCH_CHECK(false, schema.name(), " is unsafe to autocast. Run it in float32 outside the autocast region.");
    } else if (policy == LOWER_PRECISION_FP || policy == FP32) {
      const auto to = policy == FP32 ? at::kFloat : lower_precision_fp();
      for (size_t i = 0; i < n; ++i) args[i] = cast_value(args[i], to);
    } else if (policy == PROMOTE) {
      // torch starts from the lower-precision type and widens to float32 when
      // any eligible operand is float32 (never to double).
      auto widest = lower_precision_fp();
      for (size_t i = 0; i < n; ++i) {
        if (args[i].isTensor()) {
          widest = prioritize(widest, args[i].toTensor());
        } else if (args[i].isTensorList()) {
          for (const at::Tensor t : args[i].toTensorList()) widest = prioritize(widest, t);
        }
      }
      for (size_t i = 0; i < n; ++i) args[i] = cast_value(args[i], widest);
    } else if (policy == FP32_SET_OPT_DTYPE) {
      if (first_arg_eligible(args, n)) {
        for (size_t i = 0; i < n; ++i) {
          if (schema.arguments()[i].name() == "dtype" && args[i].isNone()) args[i] = c10::IValue(at::kFloat);
        }
      }
    }
  }
  (void)ks;
  op.callBoxed(stack);  // recomputed key set skips AutocastPrivateUse1 via the TLS exclude above
}

// `redispatch` is the target overload of an FP32_APPEND_DTYPE entry, null for
// every other policy.
struct PolicyEntry { const char* name; int32_t policy; const char* redispatch; };
// Generated by native/__init__.py from ATen/autocast_mode.h of the torch in use.
const PolicyEntry kCudaPolicies[] = {
#include "tmb_autocast_policies.inc"
};

}  // namespace

extern "C" {

// Changes the policy of an op that already has one (the kernel is registered
// at load time for the table's ops only).
int32_t tmb_autocast_policy(const char* qualified_name, int32_t policy) {
  std::lock_guard<std::mutex> g(g_policy_mutex);
  g_policies[qualified_name] = policy;
  return 0;
}

int32_t tmb_autocast_install_cuda_policies(void) {
  std::lock_guard<std::mutex> g(g_policy_mutex);
  for (const auto& e : kCudaPolicies) {
    g_policies[e.name] = e.policy;
    if (e.redispatch) g_append_dtype_targets[e.name] = e.redispatch;
  }
  g_policies["aten::binary_cross_entropy"] = BANNED;
  return 0;
}

}  // extern "C"

// Ops with a policy get the boxed kernel; every other op falls through, so a
// composite without a policy (nll_loss_nd, cross_entropy_loss, ...) still has
// its inner ops autocast. A catch-all fallback would exclude the key before
// redispatching and silently turn autocast off inside every composite.
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  for (const auto& e : kCudaPolicies) {
    m.impl(e.name + 6 /* strip "aten::" */, torch::CppFunction::makeFromBoxedFunction<&autocast_fallback>());
  }
  m.impl("binary_cross_entropy", torch::CppFunction::makeFromBoxedFunction<&autocast_fallback>());
}
