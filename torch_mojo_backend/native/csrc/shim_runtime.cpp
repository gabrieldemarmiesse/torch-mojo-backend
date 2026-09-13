// The c10 objects a PrivateUse1 backend must provide as C++ classes: the
// allocator, the PrivateUse1 hooks, the device guard (devices/streams/events),
// the RNG generator, the profiler stubs. Every method forwards to the function
// table Mojo registered. Also the tensor C API the Mojo ops use.
#include "tmb_internal.h"

#include <ATen/Context.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/Generator.h>
#include <ATen/core/Tensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/GradMode.h>
#include <c10/core/GeneratorImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/csrc/profiler/stubs/base.h>

#include <chrono>
#include <exception>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

std::recursive_mutex tmb_mutex;
TmbBackendHooks tmb_hooks{};
bool tmb_ready = false;

std::string& tmb_thread_error() {
  static thread_local std::string err;
  return err;
}

extern "C" void tmb_set_error(const char* message) {
  tmb_thread_error() = message ? message : "";
}
extern "C" const char* tmb_get_error(void) { return tmb_thread_error().c_str(); }
extern "C" void tmb_lock(void) { tmb_mutex.lock(); }
extern "C" void tmb_unlock(void) { tmb_mutex.unlock(); }

namespace {
thread_local int32_t tls_device = 0;
thread_local std::vector<int64_t> tls_streams;
int64_t& tls_stream_slot(int32_t device) {
  if (static_cast<size_t>(device) >= tls_streams.size()) tls_streams.resize(device + 1, 0);
  return tls_streams[device];
}
}  // namespace
extern "C" int32_t tmb_current_device(void) { return tls_device; }
extern "C" void tmb_set_current_device(int32_t device) { tls_device = device; }
extern "C" int64_t tmb_current_stream(int32_t device) { return device < 0 ? 0 : tls_stream_slot(device); }
extern "C" void tmb_set_current_stream(int32_t device, int64_t stream) { if (device >= 0) tls_stream_slot(device) = stream; }

namespace {

using Lock = std::lock_guard<std::recursive_mutex>;
#define H tmb_hooks
#define REQUIRE_READY() TORCH_CHECK(tmb_ready, "mojo backend: tmb_backend_register() has not run")

// Void hooks report failure through the thread-local message: clear it
// before the call, raise if the hook left one.
struct HookCall {
  HookCall() { tmb_thread_error().clear(); }
  ~HookCall() noexcept(false) {
    if (!tmb_thread_error().empty() && !std::uncaught_exceptions()) {
      std::string msg = tmb_thread_error();
      tmb_thread_error().clear();
      TORCH_CHECK(false, "mojo backend: ", msg);
    }
  }
};
#define HOOK(expr) do { Lock g(tmb_mutex); HookCall hc; expr; } while (0)

// The device tmb_empty_strided asked for, restored on every exit path.
struct AllocDeviceScope {
  int32_t saved;
  explicit AllocDeviceScope(int32_t d);
  ~AllocDeviceScope();
};

// Device index tmb_empty_strided asked for; the allocator has no device
// parameter so the request travels through this thread-local.
thread_local int32_t tls_alloc_device = -1;
AllocDeviceScope::AllocDeviceScope(int32_t d) : saved(tls_alloc_device) { tls_alloc_device = d; }
AllocDeviceScope::~AllocDeviceScope() { tls_alloc_device = saved; }

int32_t alloc_device() {
  if (tls_alloc_device >= 0) return tls_alloc_device;
  return tls_device;
}

// ---- allocator ---------------------------------------------------------------
struct MojoAllocator final : c10::Allocator {
  static void deleter(void* p) {
    if (!p) return;
    Lock g(tmb_mutex);
    H.free(p);
  }
  c10::DataPtr allocate(size_t n) override {
    REQUIRE_READY();
    Lock g(tmb_mutex);
    const int32_t dev = alloc_device();
    void* data = nullptr;
    tmb_thread_error().clear();
    void* handle = n ? H.alloc(n, dev, tls_stream_slot(dev), &data) : nullptr;
    TORCH_CHECK_WITH(OutOfMemoryError, handle || n == 0, "mojo backend: out of memory allocating ", n,
                     " bytes on mojo:", dev, " (", tmb_thread_error(), ")");
    return {data, handle, &deleter, c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(dev))};
  }
  // data != context here, which raw_allocate/raw_deallocate cannot express.
  c10::DeleterFnPtr raw_deleter() const override { return nullptr; }
  void copy_data(void* dst, const void* src, size_t n) const override {
    HOOK(H.copy_data(dst, src, n, tls_device, tls_stream_slot(tls_device)));
  }
};
MojoAllocator g_allocator;

// ---- generator: Philox (seed, offset) ------------------------------------------
struct MojoGeneratorImpl final : c10::GeneratorImpl {
  explicit MojoGeneratorImpl(c10::DeviceIndex index)
      : c10::GeneratorImpl(c10::Device(c10::DeviceType::PrivateUse1, index),
                           c10::DispatchKeySet(c10::DispatchKey::PrivateUse1)) {}
  void set_current_seed(uint64_t seed) override { seed_ = seed; offset_ = 0; }
  void set_offset(uint64_t offset) override { offset_ = offset; }
  uint64_t get_offset() const override { return offset_; }
  uint64_t current_seed() const override { return seed_; }
  uint64_t seed() override {
    auto s = c10::detail::getNonDeterministicRandom(true);
    set_current_seed(s);
    return s;
  }
  // 16 bytes little-endian: seed then offset -- the wire format torch.mojo.get_rng_state() kept.
  void set_state(const c10::TensorImpl& new_state) override {
    TORCH_CHECK(new_state.numel() == 16 && new_state.dtype() == caffe2::TypeMeta::Make<uint8_t>() &&
                    new_state.device().is_cpu() && new_state.is_contiguous(),
                "mojo backend: RNG state must be a contiguous 16-byte uint8 CPU tensor");
    const auto* p = static_cast<const uint8_t*>(new_state.data());
    std::memcpy(&seed_, p, 8);
    std::memcpy(&offset_, p + 8, 8);
  }
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override {
    auto t = at::detail::empty_cpu({16}, c10::ScalarType::Byte);
    auto* p = static_cast<uint8_t*>(t.data_ptr());
    std::memcpy(p, &seed_, 8);
    std::memcpy(p + 8, &offset_, 8);
    return t.getIntrusivePtr();
  }
  MojoGeneratorImpl* clone_impl() const override {
    auto* g = new MojoGeneratorImpl(device_.index());
    g->seed_ = seed_;
    g->offset_ = offset_;
    return g;
  }
  static c10::DeviceType device_type() { return c10::DeviceType::PrivateUse1; }
  uint64_t seed_ = c10::default_rng_seed_val;
  uint64_t offset_ = 0;
};

std::mutex g_generators_mutex;
std::vector<at::Generator> g_default_generators;

MojoGeneratorImpl* mojo_impl(const at::Generator& g) {
  auto* impl = dynamic_cast<MojoGeneratorImpl*>(g.unsafeGetGeneratorImpl());
  TORCH_CHECK(impl, "mojo backend: expected a generator of the mojo device, got ", g.device());
  return impl;
}

at::Generator& default_generator(c10::DeviceIndex index) {
  std::lock_guard<std::mutex> g(g_generators_mutex);
  if (index < 0) index = static_cast<c10::DeviceIndex>(tls_device);
  if (g_default_generators.empty()) {
    const int32_t n = H.device_count();
    g_default_generators.reserve(n);
    for (int32_t i = 0; i < n; ++i) g_default_generators.push_back(at::make_generator<MojoGeneratorImpl>(i));
  }
  TORCH_CHECK(index >= 0 && static_cast<size_t>(index) < g_default_generators.size(),
              "mojo backend: invalid device index ", static_cast<int>(index));
  return g_default_generators[index];
}

// ---- hooks -------------------------------------------------------------------
struct MojoHooks final : at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return tmb_ready && H.device_count() > 0; }
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
  void init() const override {}
  c10::DeviceIndex deviceCount() const override { return tmb_ready ? static_cast<c10::DeviceIndex>(H.device_count()) : 0; }
  void setCurrentDevice(c10::DeviceIndex d) const override { tls_device = d; }
  c10::DeviceIndex getCurrentDevice() const override { return static_cast<c10::DeviceIndex>(tls_device); }
  c10::DeviceIndex exchangeDevice(c10::DeviceIndex d) const override {
    auto old = static_cast<c10::DeviceIndex>(tls_device);
    if (d >= 0) tls_device = d;
    return old;
  }
  c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex d) const override { return d < 0 ? getCurrentDevice() : exchangeDevice(d); }
  bool isPinnedPtr(const void*) const override { return false; }
  c10::Allocator* getPinnedMemoryAllocator() const override { return c10::GetAllocator(c10::kCPU); }
  at::Device getDeviceFromPtr(void* data) const override {
    Lock g(tmb_mutex);
    int32_t d = H.device_of_ptr(data);
    if (d < 0) d = tls_device;
    return {c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(d)};
  }
  const at::Generator& getDefaultGenerator(c10::DeviceIndex index) const override { return default_generator(index); }
  at::Generator getNewGenerator(c10::DeviceIndex index) const override {
    if (index < 0) index = static_cast<c10::DeviceIndex>(tls_device);
    return at::make_generator<MojoGeneratorImpl>(index);
  }
  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t new_bytes) const override {
    Lock g(tmb_mutex);
    auto* impl = storage.unsafeGetStorageImpl();
    const auto dev = impl->device().index();
    c10::DataPtr fresh;
    {
      AllocDeviceScope scope(dev);
      fresh = g_allocator.allocate(new_bytes);
    }
    const size_t keep = std::min(new_bytes, impl->nbytes());
    if (keep && impl->data()) { HookCall hc; H.copy_data(fresh.get(), impl->data(), keep, dev, tls_stream_slot(dev)); }
    impl->set_data_ptr_noswap(std::move(fresh));  // the old block's release is stream-ordered after the copy
    impl->set_nbytes(new_bytes);
  }
};
MojoHooks g_hooks_iface;

// ---- device guard -------------------------------------------------------------
c10::Stream mk_stream(c10::Device d, int64_t id) {
  return c10::Stream(c10::Stream::UNSAFE, d, static_cast<c10::StreamId>(id));
}
c10::DeviceIndex idx_or_current(c10::Device d) {
  return d.index() >= 0 ? d.index() : static_cast<c10::DeviceIndex>(tls_device);
}

struct MojoGuardImpl final : c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return c10::DeviceType::PrivateUse1; }
  c10::Device exchangeDevice(c10::Device d) const override {
    auto old = getDevice();
    if (d.index() >= 0) tls_device = d.index();
    return old;
  }
  c10::Device getDevice() const override {
    return {c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(tls_device)};
  }
  void setDevice(c10::Device d) const override {
    TORCH_CHECK(d.index() < deviceCount(), "mojo backend: invalid device index ", static_cast<int>(d.index()));
    if (d.index() >= 0) tls_device = d.index();
  }
  void uncheckedSetDevice(c10::Device d) const noexcept override {
    if (d.index() >= 0) tls_device = d.index();
  }
  c10::Stream getStream(c10::Device d) const override {
    auto i = idx_or_current(d);
    return mk_stream(c10::Device(c10::DeviceType::PrivateUse1, i), tls_stream_slot(i));
  }
  c10::Stream getDefaultStream(c10::Device d) const override {
    return mk_stream(c10::Device(c10::DeviceType::PrivateUse1, idx_or_current(d)), 0);
  }
  c10::Stream getStreamFromGlobalPool(c10::Device d, bool high) const override {
    auto i = idx_or_current(d);
    int64_t id = 0;
    HOOK(id = H.stream_from_pool(i, high ? 1 : 0));
    return mk_stream(c10::Device(c10::DeviceType::PrivateUse1, i), id);
  }
  c10::Stream getNewStream(c10::Device d, int priority) const override {
    auto i = idx_or_current(d);
    int64_t id = 0;
    HOOK(id = H.new_stream(i, priority));
    return mk_stream(c10::Device(c10::DeviceType::PrivateUse1, i), id);
  }
  c10::Stream exchangeStream(c10::Stream s) const override {
    auto i = s.device_index();
    auto old = mk_stream(s.device(), tls_stream_slot(i));
    tls_stream_slot(i) = s.id();
    return old;
  }
#if TMB_TORCH_VERSION >= 211  // torch.Stream.native_handle exists from 2.11
  void* getStreamNativeHandle(const c10::Stream s) const override {
    Lock g(tmb_mutex);
    return H.stream_native_handle(s.device_index(), s.id());
  }
#endif
  c10::DeviceIndex deviceCount() const noexcept override {
    return tmb_ready ? static_cast<c10::DeviceIndex>(H.device_count()) : 0;
  }
  void destroyEvent(void* ev, const c10::DeviceIndex di) const noexcept override {
    if (!ev || !tmb_ready) return;
    Lock g(tmb_mutex);
    H.event_destroy(ev, di);
  }
  void record(void** event, const c10::Stream& stream, const c10::DeviceIndex di, const c10::EventFlag flag) const override {
    const auto dev = stream.device_index();
    if (!*event) {
      HOOK(*event = H.event_create(dev, flag == c10::EventFlag::BACKEND_DEFAULT ? 1 : 0));
      TORCH_CHECK(*event, "mojo backend: event creation failed");
    }
    HOOK(H.event_record(*event, dev, stream.id()));
    (void)di;
  }
  void block(void* ev, const c10::Stream& s) const override {
    if (!ev) return;
    HOOK(H.event_block(ev, s.device_index(), s.id()));
  }
  bool queryEvent(void* ev) const override {
    if (!ev) return true;
    int32_t r = 1;
    HOOK(r = H.event_query(ev));
    return r != 0;
  }
  void synchronizeEvent(void* ev) const override {
    if (!ev) return;
    HOOK(H.event_synchronize(ev));
  }
  bool queryStream(const c10::Stream& s) const override {
    int32_t r = 1;
    HOOK(r = H.query_stream(s.device_index(), s.id()));
    return r != 0;
  }
  void synchronizeStream(const c10::Stream& s) const override {
    HOOK(H.synchronize_stream(s.device_index(), s.id()));
  }
  void synchronizeDevice(const c10::DeviceIndex di) const override {
    HOOK(H.synchronize_device(di));
  }
  void recordDataPtrOnStream(const c10::DataPtr& p, const c10::Stream& s) const override {
    if (!p.get_context() || p.get_deleter() != &MojoAllocator::deleter) return;  // not ours (e.g. from_blob)
    HOOK(H.record_stream(p.get_context(), s.device_index(), s.id()));
  }
  double elapsedTime(void* e1, void* e2, const c10::DeviceIndex) const override {
    double ms = 0;
    HOOK(ms = H.event_elapsed_ms(e1, e2));
    return ms;
  }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, MojoGuardImpl);

// ---- profiler stubs (torch.autograd.profiler legacy path + record_function marks)
struct MojoProfilerStubs final : torch::profiler::impl::ProfilerStubs {
  void record(c10::DeviceIndex* device, torch::profiler::impl::ProfilerVoidEventStub* event, int64_t* cpu_ns) const override {
    if (cpu_ns) {
      *cpu_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
    }
    int32_t index = tls_device;
    void* ev = nullptr;
    if (tmb_ready) {
      Lock g(tmb_mutex);
      ev = H.event_create(index, 1);
      H.event_record(ev, index, tls_stream_slot(index));
    }
    if (device) *device = static_cast<c10::DeviceIndex>(index);
    if (event) {
      const int32_t dev = index;
      *event = torch::profiler::impl::ProfilerVoidEventStub(ev, [dev](void* p) {
        if (p && tmb_ready) { Lock g(tmb_mutex); H.event_destroy(p, dev); }
      });
    }
  }
  float elapsed(const torch::profiler::impl::ProfilerVoidEventStub* a,
                const torch::profiler::impl::ProfilerVoidEventStub* b) const override {
    if (!a || !b || !*a || !*b || !tmb_ready) return 0.0f;
    Lock g(tmb_mutex);
    H.event_synchronize(b->get());
    return static_cast<float>(H.event_elapsed_ms(a->get(), b->get()) * 1000.0);
  }
  void mark(const char* name) const override { if (H.prof_mark) { Lock g(tmb_mutex); H.prof_mark(name); } }
  void rangePush(const char* name) const override { if (H.prof_range_push) { Lock g(tmb_mutex); H.prof_range_push(name); } }
  void rangePop() const override { if (H.prof_range_pop) { Lock g(tmb_mutex); H.prof_range_pop(); } }
  bool enabled() const override { return tmb_ready; }
  void onEachDevice(std::function<void(int)> op) const override {
    const int32_t n = tmb_ready ? H.device_count() : 0;
    for (int32_t i = 0; i < n; ++i) op(i);
  }
  void synchronize() const override {
    if (!tmb_ready) return;
    Lock g(tmb_mutex);
    const int32_t n = H.device_count();
    for (int32_t i = 0; i < n; ++i) H.synchronize_device(i);
  }
};
MojoProfilerStubs g_profiler_stubs;

inline at::Tensor& T(TmbTensor t) { return *reinterpret_cast<at::Tensor*>(t); }

}  // namespace

extern "C" {

int32_t tmb_backend_register(const TmbBackendHooks* hooks) {
  try {
    TORCH_CHECK(hooks && hooks->size >= sizeof(TmbBackendHooks), "mojo backend: hook table too small");
    TORCH_CHECK(!tmb_ready, "mojo backend: already registered");
    std::memcpy(&tmb_hooks, hooks, sizeof(TmbBackendHooks));
    c10::SetAllocator(c10::DeviceType::PrivateUse1, &g_allocator);
    at::RegisterPrivateUse1HooksInterface(&g_hooks_iface);
    torch::profiler::impl::registerPrivateUse1Methods(&g_profiler_stubs);
    tmb_ready = true;
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

void* tmb_tensor_data_ptr(TmbTensor t) { return T(t).data_ptr(); }
int64_t tmb_tensor_dim(TmbTensor t) { return T(t).dim(); }
const int64_t* tmb_tensor_sizes(TmbTensor t) { return T(t).sizes().data(); }
const int64_t* tmb_tensor_strides(TmbTensor t) { return T(t).strides().data(); }
int64_t tmb_tensor_storage_offset(TmbTensor t) { return T(t).storage_offset(); }
int64_t tmb_tensor_numel(TmbTensor t) { return T(t).numel(); }
int32_t tmb_tensor_dtype(TmbTensor t) { return static_cast<int32_t>(T(t).scalar_type()); }
int32_t tmb_tensor_device_index(TmbTensor t) {
  const auto& x = T(t);
  return x.device().type() == c10::DeviceType::PrivateUse1 ? x.device().index() : -1;
}
int32_t tmb_tensor_device_type(TmbTensor t) { return static_cast<int32_t>(T(t).device().type()); }
int32_t tmb_tensor_is_privateuse1(TmbTensor t) { return T(t).device().type() == c10::DeviceType::PrivateUse1; }
void* tmb_tensor_storage_data_ptr(TmbTensor t) { return T(t).storage().mutable_data(); }
void* tmb_tensor_storage_ctx(TmbTensor t) {
  const auto& dp = T(t).storage().data_ptr();
  return dp.get_deleter() == &MojoAllocator::deleter ? dp.get_context() : nullptr;
}
int64_t tmb_tensor_storage_nbytes(TmbTensor t) { return static_cast<int64_t>(T(t).storage().nbytes()); }
int32_t tmb_tensor_is_contiguous(TmbTensor t) { return T(t).is_contiguous(); }
int32_t tmb_tensor_requires_grad(TmbTensor t) { return T(t).requires_grad(); }
void tmb_tensor_bump_version(TmbTensor t) { T(t).unsafeGetTensorImpl()->bump_version(); }
void* tmb_stream_native_handle(int32_t device, int64_t stream) {
  Lock g(tmb_mutex);
  return H.stream_native_handle(device, stream);
}
int32_t tmb_float32_matmul_precision(void) {
  return static_cast<int32_t>(at::globalContext().float32MatmulPrecision());
}
int32_t tmb_grad_enabled(void) { return c10::GradMode::is_enabled() ? 1 : 0; }
TmbTensor tmb_tensor_retain(TmbTensor t) { return new at::Tensor(T(t)); }
void tmb_tensor_release(TmbTensor t) { delete reinterpret_cast<at::Tensor*>(t); }

int32_t tmb_empty_strided(int64_t ndim, const int64_t* sizes, const int64_t* strides, int32_t dtype,
                          int32_t device, TmbTensor* ret) {
  try {
    AllocDeviceScope scope(device);
    auto t = at::detail::empty_strided_generic(c10::IntArrayRef(sizes, ndim), c10::IntArrayRef(strides, ndim),
                                               &g_allocator, c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                               static_cast<c10::ScalarType>(dtype));
    *ret = new at::Tensor(std::move(t));
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

// The view must stay inside its storage (at::native::checkInBoundsForStorage).
void check_in_bounds(const c10::Storage& storage, int64_t ndim, const int64_t* sizes, const int64_t* strides,
                     int64_t storage_offset, size_t itemsize) {
  TORCH_CHECK(storage_offset >= 0, "negative storage offset ", storage_offset);
  int64_t last = 0;  // offset of the last element, in elements
  for (int64_t i = 0; i < ndim; ++i) {
    TORCH_CHECK(sizes[i] >= 0, "negative size ", sizes[i]);
    if (sizes[i] == 0) return;
    if (strides[i] > 0) last += (sizes[i] - 1) * strides[i];
    else TORCH_CHECK(strides[i] == 0 || (sizes[i] - 1) * strides[i] + storage_offset >= 0, "negative stride reaches before the storage");
  }
  const int64_t needed = (storage_offset + last + 1) * static_cast<int64_t>(itemsize);
  TORCH_CHECK(needed <= static_cast<int64_t>(storage.nbytes()), "setStorage: sizes/strides reach ", needed,
              " bytes but the storage has ", storage.nbytes());
}

int32_t tmb_as_strided(TmbTensor base, int64_t ndim, const int64_t* sizes, const int64_t* strides,
                       int64_t storage_offset, TmbTensor* ret) {
  try {
    const at::Tensor& b = T(base);
    check_in_bounds(b.storage(), ndim, sizes, strides, storage_offset, b.itemsize());
    auto t = at::detail::make_tensor<c10::TensorImpl>(c10::TensorImpl::VIEW, c10::Storage(b.storage()),
                                                      b.key_set(), b.dtype());
    t.unsafeGetTensorImpl()->set_sizes_and_strides(c10::IntArrayRef(sizes, ndim), c10::IntArrayRef(strides, ndim),
                                                   storage_offset);
    *ret = new at::Tensor(std::move(t));
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

int32_t tmb_tensor_set_sizes_strides(TmbTensor t, int64_t ndim, const int64_t* sizes, const int64_t* strides,
                                     int64_t storage_offset) {
  try {
    check_in_bounds(T(t).storage(), ndim, sizes, strides, storage_offset, T(t).itemsize());
    T(t).unsafeGetTensorImpl()->set_sizes_and_strides(c10::IntArrayRef(sizes, ndim), c10::IntArrayRef(strides, ndim),
                                                      storage_offset);
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

int32_t tmb_tensor_set_storage(TmbTensor t, TmbTensor source) {
  try {
    T(t).unsafeGetTensorImpl()->set_storage_keep_dtype(T(source).storage());
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

int32_t tmb_storage_resize(TmbTensor t, int64_t nbytes) {
  try {
    g_hooks_iface.resizePrivateUse1Bytes(T(t).storage(), static_cast<size_t>(nbytes));
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

int32_t tmb_cpu_empty(int64_t ndim, const int64_t* sizes, int32_t dtype, TmbTensor* ret) {
  try {
    *ret = new at::Tensor(at::detail::empty_cpu(c10::IntArrayRef(sizes, ndim), static_cast<c10::ScalarType>(dtype)));
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

int32_t tmb_philox_reserve(TmbGenerator gen, int32_t device, uint64_t increment, uint64_t* seed, uint64_t* offset) {
  try {
    at::Generator g = gen ? *reinterpret_cast<at::Generator*>(gen) : default_generator(static_cast<c10::DeviceIndex>(device));
    std::lock_guard<std::mutex> lock(g.mutex());
    auto* impl = mojo_impl(g);
    *seed = impl->seed_;
    *offset = impl->offset_;
    TORCH_CHECK(increment <= UINT64_MAX - impl->offset_, "mojo backend: Philox counter reservation would wrap");
    impl->offset_ += increment;  // the kernels' own unit (same contract as the old _reserve_philox_state)
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

int32_t tmb_rng_manual_seed(int32_t device, uint64_t seed) {
  try {
    const int32_t n = H.device_count();
    for (int32_t i = 0; i < n; ++i) {
      if (device >= 0 && i != device) continue;
      auto& g = default_generator(static_cast<c10::DeviceIndex>(i));
      std::lock_guard<std::mutex> lock(g.mutex());
      g.set_current_seed(seed);
    }
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

int32_t tmb_rng_get_state(int32_t device, uint8_t* out16) {
  try {
    auto g = default_generator(static_cast<c10::DeviceIndex>(device));
    std::lock_guard<std::mutex> lock(g.mutex());
    auto* impl = mojo_impl(g);
    std::memcpy(out16, &impl->seed_, 8);
    std::memcpy(out16 + 8, &impl->offset_, 8);
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

int32_t tmb_rng_set_state(int32_t device, const uint8_t* in16) {
  try {
    auto g = default_generator(static_cast<c10::DeviceIndex>(device));
    std::lock_guard<std::mutex> lock(g.mutex());
    auto* impl = mojo_impl(g);
    std::memcpy(&impl->seed_, in16, 8);
    std::memcpy(&impl->offset_, in16 + 8, 8);
    return 0;
  } catch (const std::exception& e) {
    tmb_set_error(e.what());
    return 1;
  }
}

int32_t tmb_default_dtype(void) { return static_cast<int32_t>(c10::typeMetaToScalarType(c10::get_default_dtype())); }

}  // extern "C"
