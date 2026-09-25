// Boxed kernel adapter: torch's IValue stack <-> TmbValue records, one boxed
// functor per registered op. This replaces libtorch's StableIValue path
// (torch_library_impl) because that one cannot encode Scalar arguments and
// heap-boxes every optional/list per call.
#include "tmb_internal.h"

#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// Per-op call counters for the test suite (tests/testing.py CallChecker):
// off unless enabled, so the hot path pays one relaxed load.
std::atomic<bool> g_count_calls{false};
std::mutex g_counts_mutex;
std::unordered_map<std::string, int64_t> g_counts;

void count_call(const c10::FunctionSchema& schema) {
  std::lock_guard<std::mutex> g(g_counts_mutex);
  std::string key = schema.name();
  if (!schema.overload_name().empty()) key += "." + schema.overload_name();
  ++g_counts[key];
}

}  // namespace

// The unboxed view kernels (shim_views.cpp) never pass through the boxed
// adapter, so they report themselves here to keep CallChecker's accounting.
void tmb_count_op_call(const char* qualified_name) {
  if (!g_count_calls.load(std::memory_order_relaxed)) return;
  std::lock_guard<std::mutex> g(g_counts_mutex);
  ++g_counts[qualified_name];
}

namespace {

// Backing store for list arguments, built only for a schema that has one
// (Plan::needs_arena). A std::deque allocates on construction, which is why
// one is not used here; inner buffers keep their address when an outer vector
// grows (a moved std::vector keeps its heap block); a generator is boxed so
// that its address survives a reallocation too.
struct Arena {
  std::vector<std::vector<int64_t>> ints;
  std::vector<std::vector<double>> doubles;
  std::vector<std::vector<uint8_t>> bools;
  std::vector<std::vector<at::Tensor>> tensors;
  std::vector<std::vector<const at::Tensor*>> tensor_ptrs;
  std::vector<std::unique_ptr<at::Generator>> generators;
  std::vector<std::unique_ptr<c10::Storage>> storages;
};

inline int64_t double_bits(double d) { int64_t r; std::memcpy(&r, &d, 8); return r; }
inline double bits_double(int64_t b) { double d; std::memcpy(&d, &b, 8); return d; }

void scalar_to_record(const c10::Scalar& s, TmbValue& out) {
  if (s.isBoolean()) { out.tag = TMB_SCALAR_BOOL; out.a = s.toBool(); }
  else if (s.isIntegral(false)) { out.tag = TMB_SCALAR_INT; out.a = s.toLong(); }
  else if (s.isFloatingPoint()) { out.tag = TMB_SCALAR_DOUBLE; out.a = double_bits(s.toDouble()); }
  else { auto c = s.toComplexDouble(); out.tag = TMB_COMPLEX; out.a = double_bits(c.real()); out.b = double_bits(c.imag()); }
}

// How one schema position converts, decided once per schema (Plan below)
// rather than walked out of the TYPES on every call. Optional is unwrapped at
// plan time; a None IValue is checked before the conversion either way.
enum Conv : uint8_t {
  CV_TENSOR, CV_INT, CV_SYMINT, CV_DOUBLE, CV_BOOL, CV_SCALAR, CV_DTYPE,
  CV_LAYOUT, CV_MEMORY_FORMAT, CV_DEVICE, CV_STRING, CV_STREAM,
  // from here on: the Arena kinds, see arena_kind()
  CV_GENERATOR, CV_STORAGE, CV_INT_LIST, CV_DOUBLE_LIST, CV_BOOL_LIST, CV_SCALAR_LIST,
  CV_TENSOR_LIST, CV_OPT_TENSOR_LIST,
  CV_UNSUPPORTED,       // raised at conversion time, in argument order
  CV_UNSUPPORTED_LIST,  // a list whose element type we do not carry
};

// A schema with none of these allocates nothing per call.
inline bool arena_kind(uint8_t c) { return c >= CV_GENERATOR && c <= CV_OPT_TENSOR_LIST; }

uint8_t conv_of(const c10::TypePtr& type) {
  switch (type->kind()) {
    case c10::TypeKind::OptionalType:
      return conv_of(type->castRaw<c10::OptionalType>()->getElementType());
    case c10::TypeKind::TensorType: return CV_TENSOR;
    case c10::TypeKind::IntType: return CV_INT;
    case c10::TypeKind::SymIntType: return CV_SYMINT;
    case c10::TypeKind::FloatType: return CV_DOUBLE;
    case c10::TypeKind::BoolType: return CV_BOOL;
    case c10::TypeKind::NumberType: return CV_SCALAR;
    case c10::TypeKind::ScalarTypeType: return CV_DTYPE;
    case c10::TypeKind::LayoutType: return CV_LAYOUT;
    case c10::TypeKind::MemoryFormatType: return CV_MEMORY_FORMAT;
    case c10::TypeKind::DeviceObjType: return CV_DEVICE;
    case c10::TypeKind::StringType: return CV_STRING;
    case c10::TypeKind::GeneratorType: return CV_GENERATOR;
    case c10::TypeKind::StorageType: return CV_STORAGE;
    case c10::TypeKind::StreamObjType: return CV_STREAM;
    case c10::TypeKind::ListType: {
      const auto inner = type->castRaw<c10::ListType>()->getElementType();
      switch (inner->kind()) {
        case c10::TypeKind::TensorType: return CV_TENSOR_LIST;
        case c10::TypeKind::IntType:
        case c10::TypeKind::SymIntType: return CV_INT_LIST;
        case c10::TypeKind::FloatType: return CV_DOUBLE_LIST;
        case c10::TypeKind::BoolType: return CV_BOOL_LIST;
        case c10::TypeKind::NumberType: return CV_SCALAR_LIST;
        case c10::TypeKind::OptionalType:
          return inner->castRaw<c10::OptionalType>()->getElementType()->kind() ==
                         c10::TypeKind::TensorType
                     ? CV_OPT_TENSOR_LIST
                     : CV_UNSUPPORTED_LIST;
        default: return CV_UNSUPPORTED_LIST;
      }
    }
    default: return CV_UNSUPPORTED;
  }
}

struct RetConv {
  uint8_t conv;
  // A None record for a `Tensor` result is an undefined Tensor (a masked-off
  // gradient, as ATen's own backward kernels return it); `Tensor?` gives None.
  bool none_is_undefined_tensor;
};

inline RetConv ret_conv_of(const c10::TypePtr& type) {
  return {conv_of(type), type->kind() == c10::TypeKind::TensorType};
}

// One schema's conversions, interned at that op's first call.
//
// A plan is identified BY VALUE, not by its schema's address: torch lets a
// schema be deregistered and another registered (a torch.library Library
// destroyed and redefined; a fallback functor serving ops registered later),
// possibly at the same address, and a plan describing the old op would then
// convert the new one's arguments. `matches` compares what the conversions
// are derived from -- the qualified name, owned here, the argument and return
// counts, and the identity of the types -- and holds the types alive, which
// is what makes comparing their addresses mean anything.
//
// The two name strings are owned for the same reason: they reach the kernel
// and the error path, and a deregistered schema's would dangle.
struct Plan {
  std::string name_str;
  std::string overload_str;
  const char* name;  // into name_str: a Plan is immortal and never moves
  const char* overload;
  size_t n_args;
  size_t n_rets;
  bool arena;
  std::vector<uint8_t> args;
  std::vector<RetConv> rets;
  std::vector<c10::TypePtr> types;  // the arguments', then the returns'

  bool matches(const c10::FunctionSchema& s) const {
    const auto& as = s.arguments();
    const auto& rs = s.returns();
    if (as.size() != n_args || rs.size() != n_rets) return false;
    for (size_t i = 0; i < n_args; ++i) {
      if (types[i].get() != as[i].real_type().get()) return false;
    }
    for (size_t i = 0; i < n_rets; ++i) {
      if (types[n_args + i].get() != rs[i].real_type().get()) return false;
    }
    return name_str == s.name() && overload_str == s.overload_name();
  }
};

std::mutex g_plans_mutex;
std::unordered_map<const c10::FunctionSchema*, Plan*> g_plans;
// Plans built since the process started (test support: a warm op must not
// re-intern, which is what makes `matches` a hot-path check and not a cost).
std::atomic<int64_t> g_plan_builds{0};

// Every read and write of the table happens under g_plans_mutex, so a first
// call from the main thread and one from the autograd thread cannot both
// build: the loser finds the winner's plan.
//
// A plan is immortal, and a stale one is replaced rather than freed: another
// functor may still hold it, and will re-intern when its own `matches` fails.
const Plan* plan_for(const c10::FunctionSchema& schema) {
  std::lock_guard<std::mutex> g(g_plans_mutex);
  Plan*& slot = g_plans[&schema];
  if (slot && slot->matches(schema)) return slot;
  auto* p = new Plan();
  p->name_str = schema.name();
  p->overload_str = schema.overload_name();
  p->name = p->name_str.c_str();
  p->overload = p->overload_str.c_str();
  p->n_args = schema.arguments().size();
  p->n_rets = schema.returns().size();
  p->arena = false;
  p->args.reserve(p->n_args);
  p->types.reserve(p->n_args + p->n_rets);
  for (const auto& a : schema.arguments()) {
    p->args.push_back(conv_of(a.real_type()));
    p->arena |= arena_kind(p->args.back());
    p->types.push_back(a.real_type());
  }
  p->rets.reserve(p->n_rets);
  for (const auto& r : schema.returns()) {
    p->rets.push_back(ret_conv_of(r.real_type()));
    p->types.push_back(r.real_type());
  }
  slot = p;
  g_plan_builds.fetch_add(1, std::memory_order_relaxed);
  return p;
}

void to_record(uint8_t conv, const c10::TypePtr& type, const c10::IValue& v, TmbValue& out,
               std::optional<Arena>& arena) {
  out.len = 0; out.a = 0; out.b = 0;
  if (v.isNone()) { out.tag = TMB_NONE; return; }
  switch (conv) {
    case CV_TENSOR:
      // an absent `Tensor?` from a C++ composite is an undefined Tensor, not None
      if (!v.toTensor().defined()) { out.tag = TMB_NONE; return; }
      out.tag = TMB_TENSOR; out.a = reinterpret_cast<int64_t>(&v.toTensor()); return;
    case CV_INT:
      out.tag = TMB_INT; out.a = v.toInt(); return;
    case CV_SYMINT:
      out.tag = TMB_INT; out.a = v.isInt() ? v.toInt() : v.toSymInt().guard_int(__FILE__, __LINE__); return;
    case CV_DOUBLE:
      out.tag = TMB_DOUBLE; out.a = double_bits(v.toDouble()); return;
    case CV_BOOL:
      out.tag = TMB_BOOL; out.a = v.toBool(); return;
    case CV_SCALAR:
      scalar_to_record(v.toScalar(), out); return;
    case CV_DTYPE:
      out.tag = TMB_DTYPE; out.a = static_cast<int64_t>(v.toScalarType()); return;
    case CV_LAYOUT:
      out.tag = TMB_LAYOUT; out.a = static_cast<int64_t>(v.toLayout()); return;
    case CV_MEMORY_FORMAT:
      out.tag = TMB_MEMORY_FORMAT; out.a = static_cast<int64_t>(v.toMemoryFormat()); return;
    case CV_DEVICE: {
      auto d = v.toDevice();
      out.tag = TMB_DEVICE; out.a = static_cast<int64_t>(d.type()); out.b = d.index(); return;
    }
    case CV_STRING: {
      const auto& s = v.toStringRef();
      out.tag = TMB_STRING; out.a = reinterpret_cast<int64_t>(s.data()); out.len = static_cast<int32_t>(s.size()); return;
    }
    case CV_STREAM: {
      auto st = v.toStream();
      out.tag = TMB_STREAM; out.a = st.device_index(); out.b = static_cast<int64_t>(st.id()); return;
    }
    case CV_GENERATOR:
      // boxed: the address must survive the vector growing
      arena->generators.push_back(std::make_unique<at::Generator>(v.toGenerator()));
      out.tag = TMB_GENERATOR; out.a = reinterpret_cast<int64_t>(arena->generators.back().get()); return;
    case CV_STORAGE: {
      // IValue hands a Storage out by value: box it so its address is stable
      arena->storages.push_back(std::make_unique<c10::Storage>(v.toStorage()));
      const c10::Storage& st = *arena->storages.back();
      out.tag = TMB_STORAGE; out.a = reinterpret_cast<int64_t>(&st);
      out.b = static_cast<int64_t>(st.nbytes());
      out.len = st.device().type() == c10::DeviceType::PrivateUse1 ? st.device().index() : -1;
      return;
    }
    case CV_TENSOR_LIST: {
      arena->tensors.push_back(v.toTensorVector());
      auto& ts = arena->tensors.back();
      arena->tensor_ptrs.emplace_back();
      auto& ps = arena->tensor_ptrs.back();
      ps.reserve(ts.size());
      for (auto& t : ts) ps.push_back(&t);
      out.tag = TMB_TENSOR_LIST; out.a = reinterpret_cast<int64_t>(ps.data()); out.len = static_cast<int32_t>(ps.size());
      return;
    }
    case CV_OPT_TENSOR_LIST: {
      auto list = v.toOptionalTensorList();
      arena->tensors.emplace_back();
      auto& ts = arena->tensors.back();
      ts.reserve(list.size());
      std::vector<bool> present;
      present.reserve(list.size());
      for (size_t i = 0; i < list.size(); ++i) {
        std::optional<at::Tensor> e = list.get(i);
        // ATen indexing also represents an omitted axis as an undefined Tensor.
        present.push_back(e.has_value() && e->defined());
        ts.push_back(e.has_value() ? *e : at::Tensor());
      }
      arena->tensor_ptrs.emplace_back();
      auto& ps = arena->tensor_ptrs.back();
      for (size_t i = 0; i < ts.size(); ++i) ps.push_back(present[i] ? &ts[i] : nullptr);
      out.tag = TMB_OPT_TENSOR_LIST; out.a = reinterpret_cast<int64_t>(ps.data()); out.len = static_cast<int32_t>(ps.size());
      return;
    }
    case CV_INT_LIST: {
      if (v.isIntList()) {
        arena->ints.push_back(v.toIntVector());
      } else {
        arena->ints.emplace_back();
        for (auto& s : v.toSymIntVector()) arena->ints.back().push_back(s.guard_int(__FILE__, __LINE__));
      }
      out.tag = TMB_INT_LIST; out.a = reinterpret_cast<int64_t>(arena->ints.back().data()); out.len = static_cast<int32_t>(arena->ints.back().size());
      return;
    }
    case CV_DOUBLE_LIST: {
      arena->doubles.push_back(v.toDoubleVector());
      out.tag = TMB_DOUBLE_LIST; out.a = reinterpret_cast<int64_t>(arena->doubles.back().data()); out.len = static_cast<int32_t>(arena->doubles.back().size());
      return;
    }
    case CV_BOOL_LIST: {
      arena->bools.emplace_back();
      for (bool b : v.toBoolList()) arena->bools.back().push_back(b ? 1 : 0);
      out.tag = TMB_BOOL_LIST; out.a = reinterpret_cast<int64_t>(arena->bools.back().data()); out.len = static_cast<int32_t>(arena->bools.back().size());
      return;
    }
    case CV_SCALAR_LIST: {
      // Scalar[] (the _foreach_*.ScalarList ops). A c10::Scalar is tagged,
      // and the list has no per-element tag to carry, so the LIST takes the
      // tag every element agrees on: all bool -> TMB_BOOL_LIST, else all
      // integral -> TMB_INT_LIST (exact past 2^53, where a double is not),
      // else TMB_DOUBLE_LIST. An empty list is vacuously integral. A mixed
      // list is the only one that rounds, and only its integers.
      const auto& xs = v.toListRef();
      bool all_bool = true, all_int = true;
      for (const auto& e : xs) {
        const auto& s = e.toScalar();
        all_bool &= s.isBoolean();
        all_int &= s.isIntegral(/*includeBool=*/true);
      }
      if (all_bool && !xs.empty()) {
        arena->bools.emplace_back();
        for (const auto& e : xs) arena->bools.back().push_back(e.toScalar().toBool() ? 1 : 0);
        out.tag = TMB_BOOL_LIST; out.a = reinterpret_cast<int64_t>(arena->bools.back().data()); out.len = static_cast<int32_t>(arena->bools.back().size());
        return;
      }
      if (all_int) {
        arena->ints.emplace_back();
        for (const auto& e : xs) arena->ints.back().push_back(e.toScalar().toLong());
        out.tag = TMB_INT_LIST; out.a = reinterpret_cast<int64_t>(arena->ints.back().data()); out.len = static_cast<int32_t>(arena->ints.back().size());
        return;
      }
      arena->doubles.emplace_back();
      for (const auto& e : xs) arena->doubles.back().push_back(e.toScalar().toDouble());
      out.tag = TMB_DOUBLE_LIST; out.a = reinterpret_cast<int64_t>(arena->doubles.back().data()); out.len = static_cast<int32_t>(arena->doubles.back().size());
      return;
    }
    case CV_UNSUPPORTED_LIST:
      TORCH_CHECK(false, "mojo backend: unsupported list argument type ", type->str());
    default:
      TORCH_CHECK(false, "mojo backend: unsupported argument type ", type->str());
  }
}

c10::IValue from_record(RetConv rc, const c10::TypePtr& type, const TmbValue& r) {
  if (r.tag == TMB_NONE) {
    return rc.none_is_undefined_tensor ? c10::IValue(at::Tensor()) : c10::IValue();
  }
  switch (rc.conv) {
    case CV_TENSOR: {
      auto* p = reinterpret_cast<at::Tensor*>(r.a);
      if (r.tag == TMB_TENSOR_REF) return c10::IValue(*p);
      TORCH_CHECK(r.tag == TMB_TENSOR, "mojo backend: kernel returned tag ", r.tag, " for a Tensor result");
      c10::IValue out(std::move(*p));
      delete p;
      return out;
    }
    case CV_INT:
    case CV_SYMINT:
      return c10::IValue(r.a);
    case CV_DOUBLE:
      return c10::IValue(bits_double(r.a));
    case CV_BOOL:
      return c10::IValue(r.a != 0);
    case CV_SCALAR:
      if (r.tag == TMB_SCALAR_BOOL || r.tag == TMB_BOOL) return c10::IValue(c10::Scalar(r.a != 0));
      if (r.tag == TMB_SCALAR_DOUBLE || r.tag == TMB_DOUBLE) return c10::IValue(c10::Scalar(bits_double(r.a)));
      if (r.tag == TMB_COMPLEX) return c10::IValue(c10::Scalar(c10::complex<double>(bits_double(r.a), bits_double(r.b))));
      return c10::IValue(c10::Scalar(r.a));
    case CV_TENSOR_LIST: {
      TORCH_CHECK(r.tag == TMB_TENSOR_LIST, "mojo backend: kernel returned tag ", r.tag, " for a Tensor[] result");
      auto** ps = reinterpret_cast<at::Tensor**>(r.a);
      c10::List<at::Tensor> out;
      out.reserve(r.len);
      for (int32_t i = 0; i < r.len; ++i) { out.push_back(std::move(*ps[i])); delete ps[i]; }
      std::free(ps);
      return c10::IValue(std::move(out));
    }
    case CV_INT_LIST: {
      auto* xs = reinterpret_cast<const int64_t*>(r.a);
      c10::List<int64_t> out;
      for (int32_t i = 0; i < r.len; ++i) out.push_back(xs[i]);
      std::free(const_cast<int64_t*>(xs));
      return c10::IValue(std::move(out));
    }
    case CV_DOUBLE_LIST:
    case CV_BOOL_LIST:
    case CV_SCALAR_LIST:
    case CV_OPT_TENSOR_LIST:
    case CV_UNSUPPORTED_LIST:
      TORCH_CHECK(false, "mojo backend: unsupported list result type ", type->str());
    default:
      TORCH_CHECK(false, "mojo backend: unsupported result type ", type->str());
  }
}

// Reverse direction for tmb_call_op: records -> IValues (inputs are borrowed).
c10::IValue record_to_input(const c10::TypePtr& type, const TmbValue& r) {
  if (r.tag == TMB_NONE) return c10::IValue();
  switch (r.tag) {
    case TMB_TENSOR: case TMB_TENSOR_REF: return c10::IValue(*reinterpret_cast<const at::Tensor*>(r.a));
    case TMB_INT: return c10::IValue(r.a);
    case TMB_DOUBLE: return c10::IValue(bits_double(r.a));
    case TMB_BOOL: return c10::IValue(r.a != 0);
    case TMB_SCALAR_INT: return c10::IValue(c10::Scalar(r.a));
    case TMB_SCALAR_DOUBLE: return c10::IValue(c10::Scalar(bits_double(r.a)));
    case TMB_SCALAR_BOOL: return c10::IValue(c10::Scalar(r.a != 0));
    case TMB_COMPLEX: return c10::IValue(c10::Scalar(c10::complex<double>(bits_double(r.a), bits_double(r.b))));
    case TMB_DTYPE: return c10::IValue(static_cast<c10::ScalarType>(r.a));
    case TMB_LAYOUT: return c10::IValue(static_cast<c10::Layout>(r.a));
    case TMB_MEMORY_FORMAT: return c10::IValue(static_cast<c10::MemoryFormat>(r.a));
    case TMB_DEVICE: return c10::IValue(c10::Device(static_cast<c10::DeviceType>(r.a), static_cast<c10::DeviceIndex>(r.b)));
    case TMB_STRING: return c10::IValue(std::string(reinterpret_cast<const char*>(r.a), r.len));
    case TMB_GENERATOR: return c10::IValue(*reinterpret_cast<const at::Generator*>(r.a));
    case TMB_INT_LIST: {
      auto* xs = reinterpret_cast<const int64_t*>(r.a);
      c10::List<int64_t> l; for (int32_t i = 0; i < r.len; ++i) l.push_back(xs[i]);
      return c10::IValue(std::move(l));
    }
    case TMB_DOUBLE_LIST: {
      auto* xs = reinterpret_cast<const double*>(r.a);
      c10::List<double> l; for (int32_t i = 0; i < r.len; ++i) l.push_back(xs[i]);
      return c10::IValue(std::move(l));
    }
    case TMB_BOOL_LIST: {
      auto* xs = reinterpret_cast<const uint8_t*>(r.a);
      c10::List<bool> l; for (int32_t i = 0; i < r.len; ++i) l.push_back(xs[i] != 0);
      return c10::IValue(std::move(l));
    }
    case TMB_TENSOR_LIST: {
      const at::Tensor* const* ps = reinterpret_cast<const at::Tensor* const*>(r.a);
      c10::List<at::Tensor> l; for (int32_t i = 0; i < r.len; ++i) l.push_back(*ps[i]);
      return c10::IValue(std::move(l));
    }
    case TMB_OPT_TENSOR_LIST: {
      const at::Tensor* const* ps = reinterpret_cast<const at::Tensor* const*>(r.a);
      c10::List<std::optional<at::Tensor>> l;
      for (int32_t i = 0; i < r.len; ++i) l.push_back(ps[i] ? std::optional<at::Tensor>(*ps[i]) : std::nullopt);
      return c10::IValue(std::move(l));
    }
    default:
      TORCH_CHECK(false, "mojo backend: unsupported record tag ", r.tag, " for argument type ", type->str());
  }
}

// Outputs of tmb_call_op: IValues -> owned records.
void result_to_record(const c10::IValue& v, TmbValue& out) {
  out.len = 0; out.a = 0; out.b = 0;
  if (v.isNone()) { out.tag = TMB_NONE; return; }
  if (v.isTensor()) { out.tag = TMB_TENSOR; out.a = reinterpret_cast<int64_t>(new at::Tensor(v.toTensor())); return; }
  if (v.isTensorList()) {
    auto ts = v.toTensorVector();
    auto** ps = static_cast<at::Tensor**>(std::malloc(sizeof(at::Tensor*) * std::max<size_t>(ts.size(), 1)));
    for (size_t i = 0; i < ts.size(); ++i) ps[i] = new at::Tensor(ts[i]);
    out.tag = TMB_TENSOR_LIST; out.a = reinterpret_cast<int64_t>(ps); out.len = static_cast<int32_t>(ts.size()); return;
  }
  if (v.isBool()) { out.tag = TMB_BOOL; out.a = v.toBool(); return; }
  if (v.isInt()) { out.tag = TMB_INT; out.a = v.toInt(); return; }
  if (v.isDouble()) { out.tag = TMB_DOUBLE; out.a = double_bits(v.toDouble()); return; }
  if (v.isScalar()) { scalar_to_record(v.toScalar(), out); return; }
  if (v.isIntList()) {
    auto xs = v.toIntVector();
    auto* p = static_cast<int64_t*>(std::malloc(sizeof(int64_t) * std::max<size_t>(xs.size(), 1)));
    std::memcpy(p, xs.data(), sizeof(int64_t) * xs.size());
    out.tag = TMB_INT_LIST; out.a = reinterpret_cast<int64_t>(p); out.len = static_cast<int32_t>(xs.size()); return;
  }
  TORCH_CHECK(false, "mojo backend: unsupported result of tmb_call_op: ", v.tagKind());
}

// Owned handles a kernel already placed in its result records must not leak
// when the call fails afterwards.
void release_records(TmbValue* rets, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    if (rets[i].tag == TMB_TENSOR) {
      delete reinterpret_cast<at::Tensor*>(rets[i].a);
    } else if (rets[i].tag == TMB_TENSOR_LIST) {
      auto** ps = reinterpret_cast<at::Tensor**>(rets[i].a);
      for (int32_t k = 0; k < rets[i].len; ++k) delete ps[k];
      std::free(ps);
    } else if (rets[i].tag == TMB_INT_LIST) {
      std::free(reinterpret_cast<void*>(rets[i].a));
    }
    rets[i].tag = TMB_NONE;
  }
}

void raise_from_kernel(int32_t rc, const char* op, const char* overload) {
  std::string msg = tmb_thread_error();
  tmb_thread_error().clear();
  if (msg.empty()) msg = "mojo backend kernel failed";
  std::string where = std::string(op) + (overload && *overload ? std::string(".") + overload : std::string());
  if (rc == 2) {
    TORCH_CHECK_NOT_IMPLEMENTED(false, msg, " [", where, "]");
  }
  TORCH_CHECK(false, msg, " [", where, "]");
}

class MojoBoxedKernel final : public c10::OperatorKernel {
 public:
  MojoBoxedKernel(TmbKernelFn fn, void* ctx) : fn_(fn), ctx_(ctx) {}

  void operator()(const c10::OperatorHandle& op, c10::DispatchKeySet /*ks*/, torch::jit::Stack* stack) {
    tmb_check_not_forked();
    const auto& schema = op.schema();
    const Plan* plan = plan_.load(std::memory_order_acquire);
    // Checked by value (see Plan): a schema can be deregistered and another
    // registered at the same address. A fallback (one functor, many ops)
    // fails this check per op and takes the interning path on every call.
    if (C10_UNLIKELY(!plan || !plan->matches(schema))) {
      plan = plan_for(schema);
      plan_.store(plan, std::memory_order_release);
    }
    const size_t n_args = plan->n_args;
    const size_t n_rets = plan->n_rets;
    TmbValue args_buf[16];
    TmbValue rets_buf[8];
    std::vector<TmbValue> args_heap, rets_heap;
    TmbValue* args = args_buf;
    TmbValue* rets = rets_buf;
    if (n_args > 16) { args_heap.resize(n_args); args = args_heap.data(); }
    if (n_rets > 8) { rets_heap.resize(n_rets); rets = rets_heap.data(); }
    std::optional<Arena> arena;
    if (plan->arena) arena.emplace();
    const c10::IValue* ivalues = stack->data() + (stack->size() - n_args);
    for (size_t i = 0; i < n_args; ++i) {
      to_record(plan->args[i], schema.arguments()[i].real_type(), ivalues[i], args[i], arena);
    }
    for (size_t i = 0; i < n_rets; ++i) { rets[i].tag = TMB_NONE; rets[i].len = 0; rets[i].a = 0; rets[i].b = 0; }
    if (g_count_calls.load(std::memory_order_relaxed)) count_call(schema);
    int32_t rc;
    {
      std::lock_guard<std::recursive_mutex> g(tmb_mutex);
      rc = fn_(ctx_, plan->name, plan->overload, args, static_cast<int32_t>(n_args), rets,
               static_cast<int32_t>(n_rets));
    }
    if (rc != 0) {
      release_records(rets, n_rets);
      raise_from_kernel(rc, plan->name, plan->overload);
    }
    // Convert the results BEFORE dropping the inputs: a TENSOR_REF points at
    // a tensor inside the input IValues.
    if (n_rets <= 1) {  // every op but a few: one IValue, constructed once
      if (n_rets == 0) {
        torch::jit::drop(*stack, n_args);
        return;
      }
      c10::IValue out = from_record(plan->rets[0], schema.returns()[0].real_type(), rets[0]);
      torch::jit::drop(*stack, n_args);
      stack->push_back(std::move(out));
      return;
    }
    c10::IValue out_buf[8];
    std::vector<c10::IValue> out_heap;
    c10::IValue* outs = out_buf;
    if (n_rets > 8) { out_heap.resize(n_rets); outs = out_heap.data(); }
    for (size_t i = 0; i < n_rets; ++i) {
      try {
        outs[i] = from_record(plan->rets[i], schema.returns()[i].real_type(), rets[i]);
      } catch (...) {
        release_records(rets + i + 1, n_rets - i - 1);
        throw;
      }
    }
    torch::jit::drop(*stack, n_args);
    for (size_t i = 0; i < n_rets; ++i) stack->push_back(std::move(outs[i]));
  }

 private:
  TmbKernelFn fn_;
  void* ctx_;
  std::atomic<const Plan*> plan_{nullptr};
};

}  // namespace

extern "C" {

int64_t tmb_plan_builds(void) { return g_plan_builds.load(std::memory_order_relaxed); }

void tmb_op_counting(int32_t enabled) { g_count_calls.store(enabled != 0); }
void tmb_op_counts_reset(void) { std::lock_guard<std::mutex> g(g_counts_mutex); g_counts.clear(); }
int64_t tmb_op_count(const char* qualified_name) {
  std::lock_guard<std::mutex> g(g_counts_mutex);
  auto it = g_counts.find(qualified_name);
  return it == g_counts.end() ? 0 : it->second;
}
// "name=count\n" lines; returns the bytes needed (call twice if it exceeds cap).
int64_t tmb_op_counts_dump(char* buf, int64_t cap) {
  std::lock_guard<std::mutex> g(g_counts_mutex);
  std::string out;
  for (const auto& kv : g_counts) out += kv.first + "=" + std::to_string(kv.second) + "\n";
  if (buf && cap > 0) {
    const size_t n = std::min<size_t>(out.size(), static_cast<size_t>(cap - 1));
    std::memcpy(buf, out.data(), n);
    buf[n] = 0;
  }
  return static_cast<int64_t>(out.size() + 1);
}

TmbLibrary tmb_library_new(const char* ns, const char* dispatch_key) {
  try {
    return new torch::Library(torch::Library::Kind::IMPL, std::string(ns),
                              c10::parseDispatchKey(std::string(dispatch_key)), "mojo", 0);
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return nullptr;
  }
}

int32_t tmb_library_impl(TmbLibrary lib, const char* name, TmbKernelFn fn, void* ctx) {
  try {
    reinterpret_cast<torch::Library*>(lib)->impl(
        name, torch::CppFunction::makeFromBoxedFunctor(std::make_unique<MojoBoxedKernel>(fn, ctx)));
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

int32_t tmb_library_fallback(TmbLibrary lib, TmbKernelFn fn, void* ctx) {
  try {
    reinterpret_cast<torch::Library*>(lib)->fallback(
        torch::CppFunction::makeFromBoxedFunctor(std::make_unique<MojoBoxedKernel>(fn, ctx)));
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

int32_t tmb_call_op(const char* op, const char* overload, const TmbValue* args, int32_t n_args,
                    TmbValue* rets, int32_t n_rets) {
  try {
    auto handle = c10::Dispatcher::singleton().findSchemaOrThrow(op, overload ? overload : "");
    const auto& schema = handle.schema();
    TORCH_CHECK(static_cast<size_t>(n_args) == schema.arguments().size(), "tmb_call_op: ", op, " takes ",
                schema.arguments().size(), " arguments, got ", n_args);
    torch::jit::Stack stack;
    stack.reserve(n_args);
    for (int32_t i = 0; i < n_args; ++i) stack.push_back(record_to_input(schema.arguments()[i].real_type(), args[i]));
    handle.callBoxed(stack);
    TORCH_CHECK(static_cast<size_t>(n_rets) == stack.size(), "tmb_call_op: ", op, " returns ", stack.size(),
                " values, caller expected ", n_rets);
    for (int32_t i = 0; i < n_rets; ++i) result_to_record(stack[i], rets[i]);
    return 0;
  } catch (const c10::NotImplementedError& e) {
    tmb_set_error(e.what_without_backtrace());
    return 2;
  } catch (const c10::Error& e) {
    tmb_set_error(e.what_without_backtrace());
    return 1;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

}  // extern "C"
