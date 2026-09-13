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
#include <deque>
#include <mutex>
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

// Backing store for list arguments. Nothing is allocated until a list
// argument appears; inner buffers keep their address when an outer vector
// grows (a moved std::vector keeps its heap block), generators live in a deque.
struct Arena {
  std::vector<std::vector<int64_t>> ints;
  std::vector<std::vector<double>> doubles;
  std::vector<std::vector<uint8_t>> bools;
  std::vector<std::vector<at::Tensor>> tensors;
  std::vector<std::vector<const at::Tensor*>> tensor_ptrs;
  std::deque<at::Generator> generators;
  explicit Arena(size_t) {}
};

inline int64_t double_bits(double d) { int64_t r; std::memcpy(&r, &d, 8); return r; }
inline double bits_double(int64_t b) { double d; std::memcpy(&d, &b, 8); return d; }

void scalar_to_record(const c10::Scalar& s, TmbValue& out) {
  if (s.isBoolean()) { out.tag = TMB_SCALAR_BOOL; out.a = s.toBool(); }
  else if (s.isIntegral(false)) { out.tag = TMB_SCALAR_INT; out.a = s.toLong(); }
  else if (s.isFloatingPoint()) { out.tag = TMB_SCALAR_DOUBLE; out.a = double_bits(s.toDouble()); }
  else { auto c = s.toComplexDouble(); out.tag = TMB_COMPLEX; out.a = double_bits(c.real()); out.b = double_bits(c.imag()); }
}

void to_record(const c10::TypePtr& type, const c10::IValue& v, TmbValue& out, Arena& arena) {
  out.len = 0; out.a = 0; out.b = 0;
  if (v.isNone()) { out.tag = TMB_NONE; return; }
  switch (type->kind()) {
    case c10::TypeKind::OptionalType:
      return to_record(type->castRaw<c10::OptionalType>()->getElementType(), v, out, arena);
    case c10::TypeKind::TensorType:
      // an absent `Tensor?` from a C++ composite is an undefined Tensor, not None
      if (!v.toTensor().defined()) { out.tag = TMB_NONE; return; }
      out.tag = TMB_TENSOR; out.a = reinterpret_cast<int64_t>(&v.toTensor()); return;
    case c10::TypeKind::IntType:
      out.tag = TMB_INT; out.a = v.toInt(); return;
    case c10::TypeKind::SymIntType:
      out.tag = TMB_INT; out.a = v.isInt() ? v.toInt() : v.toSymInt().guard_int(__FILE__, __LINE__); return;
    case c10::TypeKind::FloatType:
      out.tag = TMB_DOUBLE; out.a = double_bits(v.toDouble()); return;
    case c10::TypeKind::BoolType:
      out.tag = TMB_BOOL; out.a = v.toBool(); return;
    case c10::TypeKind::NumberType:
      scalar_to_record(v.toScalar(), out); return;
    case c10::TypeKind::ScalarTypeType:
      out.tag = TMB_DTYPE; out.a = static_cast<int64_t>(v.toScalarType()); return;
    case c10::TypeKind::LayoutType:
      out.tag = TMB_LAYOUT; out.a = static_cast<int64_t>(v.toLayout()); return;
    case c10::TypeKind::MemoryFormatType:
      out.tag = TMB_MEMORY_FORMAT; out.a = static_cast<int64_t>(v.toMemoryFormat()); return;
    case c10::TypeKind::DeviceObjType: {
      auto d = v.toDevice();
      out.tag = TMB_DEVICE; out.a = static_cast<int64_t>(d.type()); out.b = d.index(); return;
    }
    case c10::TypeKind::StringType: {
      const auto& s = v.toStringRef();
      out.tag = TMB_STRING; out.a = reinterpret_cast<int64_t>(s.data()); out.len = static_cast<int32_t>(s.size()); return;
    }
    case c10::TypeKind::GeneratorType:
      arena.generators.push_back(v.toGenerator());
      out.tag = TMB_GENERATOR; out.a = reinterpret_cast<int64_t>(&arena.generators.back()); return;
    case c10::TypeKind::StreamObjType: {
      auto st = v.toStream();
      out.tag = TMB_STREAM; out.a = st.device_index(); out.b = static_cast<int64_t>(st.id()); return;
    }
    case c10::TypeKind::ListType: {
      auto inner = type->castRaw<c10::ListType>()->getElementType();
      auto ik = inner->kind();
      if (ik == c10::TypeKind::TensorType) {
        arena.tensors.push_back(v.toTensorVector());
        auto& ts = arena.tensors.back();
        arena.tensor_ptrs.emplace_back();
        auto& ps = arena.tensor_ptrs.back();
        ps.reserve(ts.size());
        for (auto& t : ts) ps.push_back(&t);
        out.tag = TMB_TENSOR_LIST; out.a = reinterpret_cast<int64_t>(ps.data()); out.len = static_cast<int32_t>(ps.size());
        return;
      }
      if (ik == c10::TypeKind::OptionalType &&
          inner->castRaw<c10::OptionalType>()->getElementType()->kind() == c10::TypeKind::TensorType) {
        auto list = v.toOptionalTensorList();
        arena.tensors.emplace_back();
        auto& ts = arena.tensors.back();
        ts.reserve(list.size());
        std::vector<bool> present;
        present.reserve(list.size());
        for (size_t i = 0; i < list.size(); ++i) {
          std::optional<at::Tensor> e = list.get(i);
          present.push_back(e.has_value());
          ts.push_back(e.has_value() ? *e : at::Tensor());
        }
        arena.tensor_ptrs.emplace_back();
        auto& ps = arena.tensor_ptrs.back();
        for (size_t i = 0; i < ts.size(); ++i) ps.push_back(present[i] ? &ts[i] : nullptr);
        out.tag = TMB_OPT_TENSOR_LIST; out.a = reinterpret_cast<int64_t>(ps.data()); out.len = static_cast<int32_t>(ps.size());
        return;
      }
      if (ik == c10::TypeKind::IntType || ik == c10::TypeKind::SymIntType) {
        if (v.isIntList()) {
          arena.ints.push_back(v.toIntVector());
        } else {
          arena.ints.emplace_back();
          for (auto& s : v.toSymIntVector()) arena.ints.back().push_back(s.guard_int(__FILE__, __LINE__));
        }
        out.tag = TMB_INT_LIST; out.a = reinterpret_cast<int64_t>(arena.ints.back().data()); out.len = static_cast<int32_t>(arena.ints.back().size());
        return;
      }
      if (ik == c10::TypeKind::FloatType) {
        arena.doubles.push_back(v.toDoubleVector());
        out.tag = TMB_DOUBLE_LIST; out.a = reinterpret_cast<int64_t>(arena.doubles.back().data()); out.len = static_cast<int32_t>(arena.doubles.back().size());
        return;
      }
      if (ik == c10::TypeKind::NumberType) {
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
          arena.bools.emplace_back();
          for (const auto& e : xs) arena.bools.back().push_back(e.toScalar().toBool() ? 1 : 0);
          out.tag = TMB_BOOL_LIST; out.a = reinterpret_cast<int64_t>(arena.bools.back().data()); out.len = static_cast<int32_t>(arena.bools.back().size());
          return;
        }
        if (all_int) {
          arena.ints.emplace_back();
          for (const auto& e : xs) arena.ints.back().push_back(e.toScalar().toLong());
          out.tag = TMB_INT_LIST; out.a = reinterpret_cast<int64_t>(arena.ints.back().data()); out.len = static_cast<int32_t>(arena.ints.back().size());
          return;
        }
        arena.doubles.emplace_back();
        for (const auto& e : xs) arena.doubles.back().push_back(e.toScalar().toDouble());
        out.tag = TMB_DOUBLE_LIST; out.a = reinterpret_cast<int64_t>(arena.doubles.back().data()); out.len = static_cast<int32_t>(arena.doubles.back().size());
        return;
      }
      if (ik == c10::TypeKind::BoolType) {
        arena.bools.emplace_back();
        for (bool b : v.toBoolList()) arena.bools.back().push_back(b ? 1 : 0);
        out.tag = TMB_BOOL_LIST; out.a = reinterpret_cast<int64_t>(arena.bools.back().data()); out.len = static_cast<int32_t>(arena.bools.back().size());
        return;
      }
      TORCH_CHECK(false, "mojo backend: unsupported list argument type ", type->str());
    }
    default:
      TORCH_CHECK(false, "mojo backend: unsupported argument type ", type->str());
  }
}

c10::IValue from_record(const c10::TypePtr& type, const TmbValue& r) {
  // a None record for a Tensor result is an undefined Tensor (a masked-off
  // gradient, as ATen's own backward kernels return it); None otherwise
  if (r.tag == TMB_NONE) return type->kind() == c10::TypeKind::TensorType ? c10::IValue(at::Tensor()) : c10::IValue();
  switch (type->kind()) {
    case c10::TypeKind::OptionalType:
      return from_record(type->castRaw<c10::OptionalType>()->getElementType(), r);
    case c10::TypeKind::TensorType: {
      auto* p = reinterpret_cast<at::Tensor*>(r.a);
      if (r.tag == TMB_TENSOR_REF) return c10::IValue(*p);
      TORCH_CHECK(r.tag == TMB_TENSOR, "mojo backend: kernel returned tag ", r.tag, " for a Tensor result");
      c10::IValue out(std::move(*p));
      delete p;
      return out;
    }
    case c10::TypeKind::IntType:
    case c10::TypeKind::SymIntType:
      return c10::IValue(r.a);
    case c10::TypeKind::FloatType:
      return c10::IValue(bits_double(r.a));
    case c10::TypeKind::BoolType:
      return c10::IValue(r.a != 0);
    case c10::TypeKind::NumberType:
      if (r.tag == TMB_SCALAR_BOOL || r.tag == TMB_BOOL) return c10::IValue(c10::Scalar(r.a != 0));
      if (r.tag == TMB_SCALAR_DOUBLE || r.tag == TMB_DOUBLE) return c10::IValue(c10::Scalar(bits_double(r.a)));
      if (r.tag == TMB_COMPLEX) return c10::IValue(c10::Scalar(c10::complex<double>(bits_double(r.a), bits_double(r.b))));
      return c10::IValue(c10::Scalar(r.a));
    case c10::TypeKind::ListType: {
      auto inner = type->castRaw<c10::ListType>()->getElementType();
      if (inner->kind() == c10::TypeKind::TensorType) {
        TORCH_CHECK(r.tag == TMB_TENSOR_LIST, "mojo backend: kernel returned tag ", r.tag, " for a Tensor[] result");
        auto** ps = reinterpret_cast<at::Tensor**>(r.a);
        c10::List<at::Tensor> out;
        out.reserve(r.len);
        for (int32_t i = 0; i < r.len; ++i) { out.push_back(std::move(*ps[i])); delete ps[i]; }
        std::free(ps);
        return c10::IValue(std::move(out));
      }
      if (inner->kind() == c10::TypeKind::IntType || inner->kind() == c10::TypeKind::SymIntType) {
        auto* xs = reinterpret_cast<const int64_t*>(r.a);
        c10::List<int64_t> out;
        for (int32_t i = 0; i < r.len; ++i) out.push_back(xs[i]);
        std::free(const_cast<int64_t*>(xs));
        return c10::IValue(std::move(out));
      }
      TORCH_CHECK(false, "mojo backend: unsupported list result type ", type->str());
    }
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
  // Lazy form: `resolve` compiles and returns the kernel at the first call.
  MojoBoxedKernel(TmbResolveFn resolve, void* resolve_ctx)
      : resolve_(resolve), resolve_ctx_(resolve_ctx) {}

  void operator()(const c10::OperatorHandle& op, c10::DispatchKeySet /*ks*/, torch::jit::Stack* stack) {
    const auto& schema = op.schema();
    const auto& arguments = schema.arguments();
    const auto& returns = schema.returns();
    const size_t n_args = arguments.size();
    const size_t n_rets = returns.size();
    TmbValue args_buf[16];
    TmbValue rets_buf[8];
    std::vector<TmbValue> args_heap, rets_heap;
    TmbValue* args = args_buf;
    TmbValue* rets = rets_buf;
    if (n_args > 16) { args_heap.resize(n_args); args = args_heap.data(); }
    if (n_rets > 8) { rets_heap.resize(n_rets); rets = rets_heap.data(); }
    Arena arena(n_args);
    auto ivalues = torch::jit::last(*stack, n_args);
    for (size_t i = 0; i < n_args; ++i) {
      to_record(arguments[i].real_type(), ivalues[i], args[i], arena);
    }
    for (size_t i = 0; i < n_rets; ++i) { rets[i].tag = TMB_NONE; rets[i].len = 0; rets[i].a = 0; rets[i].b = 0; }
    const char* name = schema.name().c_str();
    const char* overload = schema.overload_name().c_str();
    if (g_count_calls.load(std::memory_order_relaxed)) count_call(schema);
    int32_t rc;
    {
      std::lock_guard<std::recursive_mutex> g(tmb_mutex);
      TmbKernelFn fn = fn_.load(std::memory_order_relaxed);
      if (!fn) {
        TmbKernelFn resolved = nullptr;
        if (resolve_(resolve_ctx_, &resolved) != 0 || !resolved) {
          release_records(rets, n_rets);
          raise_from_kernel(1, name, overload);
        }
        fn_.store(resolved, std::memory_order_release);
        fn = resolved;
      }
      rc = fn(ctx_, name, overload, args, static_cast<int32_t>(n_args), rets, static_cast<int32_t>(n_rets));
    }
    if (rc != 0) {
      release_records(rets, n_rets);
      raise_from_kernel(rc, name, overload);
    }
    // Convert the results BEFORE dropping the inputs: a TENSOR_REF points at
    // a tensor inside the input IValues.
    c10::IValue out_buf[8];
    std::vector<c10::IValue> out_heap;
    c10::IValue* outs = out_buf;
    if (n_rets > 8) { out_heap.resize(n_rets); outs = out_heap.data(); }
    for (size_t i = 0; i < n_rets; ++i) {
      try {
        outs[i] = from_record(returns[i].real_type(), rets[i]);
      } catch (...) {
        release_records(rets + i + 1, n_rets - i - 1);
        throw;
      }
    }
    torch::jit::drop(*stack, n_args);
    for (size_t i = 0; i < n_rets; ++i) stack->push_back(std::move(outs[i]));
  }

 private:
  // Written once, under tmb_mutex, by the first call of a lazily registered op.
  std::atomic<TmbKernelFn> fn_{nullptr};
  void* ctx_{nullptr};
  TmbResolveFn resolve_{nullptr};
  void* resolve_ctx_{nullptr};
};

}  // namespace

extern "C" {

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

int32_t tmb_library_impl_lazy(TmbLibrary lib, const char* name, TmbResolveFn resolve, void* ctx) {
  try {
    reinterpret_cast<torch::Library*>(lib)->impl(
        name, torch::CppFunction::makeFromBoxedFunctor(std::make_unique<MojoBoxedKernel>(resolve, ctx)));
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
