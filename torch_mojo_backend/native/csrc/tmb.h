// C ABI between libtorch's C++ extension points and the Mojo backend.
//
// Everything a PrivateUse1 backend must hand torch as a C++ *object* (boxed
// kernels, allocator, hooks, device guard, generator, profiler stubs, autocast
// policy) is implemented in the three shim_*.cpp files and forwards to the
// function pointers Mojo fills in through tmb_backend_register(). Everything
// else -- the ops themselves -- is Mojo calling the tmb_* functions below.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
// only the tmb_* entries are exported: the shim builds with -fvisibility=hidden
#pragma GCC visibility push(default)
extern "C" {
#endif

// ---- values on the boxed-kernel stack ---------------------------------------
// One record per schema argument / return. Inputs are borrowed for the call;
// tensor outputs are owned handles the adapter takes over (see TMB_TENSOR).
enum TmbTag : int32_t {
  TMB_NONE = 0,            // optional argument left as None
  TMB_TENSOR = 1,          // a = at::Tensor* (input: borrowed; return: owned, adapter deletes)
  TMB_TENSOR_REF = 2,      // return only: a = at::Tensor* of an input to hand back (in-place ops)
  TMB_INT = 3,             // a = int64 (also SymInt, and Scalar-typed args holding an int)
  TMB_DOUBLE = 4,          // a = double bits
  TMB_BOOL = 5,            // a = 0/1
  TMB_COMPLEX = 6,         // a = re bits, b = im bits (Scalar only)
  TMB_INT_LIST = 7,        // a = const int64_t*, len = count
  TMB_DOUBLE_LIST = 8,     // a = const double*, len = count
  TMB_BOOL_LIST = 9,       // a = const uint8_t*, len = count
  TMB_TENSOR_LIST = 10,    // a = at::Tensor* const* (input: borrowed; return: owned handles, array malloc'd by Mojo), len
  TMB_OPT_TENSOR_LIST = 11,// a = at::Tensor* const* with NULL for None entries, len
  TMB_DTYPE = 12,          // a = c10::ScalarType
  TMB_LAYOUT = 13,         // a = c10::Layout
  TMB_DEVICE = 14,         // a = device type, b = device index (-1 = unset)
  TMB_MEMORY_FORMAT = 15,  // a = c10::MemoryFormat
  TMB_STRING = 16,         // a = const char*, len = byte length
  TMB_GENERATOR = 17,      // a = at::Generator* (borrowed)
  TMB_SCALAR_INT = 18,     // Scalar-typed argument: a = int64
  TMB_SCALAR_DOUBLE = 19,  // Scalar-typed argument: a = double bits
  TMB_SCALAR_BOOL = 20,    // Scalar-typed argument: a = 0/1
  TMB_STREAM = 21,         // torch.Stream argument: a = device index, b = stream id
};

typedef struct TmbValue {
  int32_t tag;
  int32_t len;
  int64_t a;
  int64_t b;
} TmbValue;

typedef void* TmbTensor;     // at::Tensor*
typedef void* TmbGenerator;  // at::Generator*
typedef void* TmbLibrary;    // torch::Library*
typedef void* TmbEvent;      // backend event object (opaque to torch)

// Return 0 on success, 1 for a RuntimeError, 2 for NotImplementedError; the
// message comes from the last tmb_set_error() on this thread.
typedef int32_t (*TmbKernelFn)(void* ctx, const char* op, const char* overload,
                               const TmbValue* args, int32_t n_args,
                               TmbValue* rets, int32_t n_rets);

// ---- the function table Mojo hands to the shim -------------------------------
// The current device and the current stream per device are thread-local state
// the shim itself keeps (tmb_current_device / tmb_current_stream); Mojo owns
// the device contexts, streams, events and memory.
typedef struct TmbBackendHooks {
  uint32_t size;  // sizeof(TmbBackendHooks), for forward compatibility
  // memory: allocate on `stream` of `device`; returns an opaque handle (freed
  // with free), data pointer through *data. NULL on out-of-memory.
  void* (*alloc)(size_t nbytes, int32_t device, int64_t stream, void** data);
  void (*free)(void* handle);
  void (*copy_data)(void* dst, const void* src, size_t nbytes, int32_t device, int64_t stream);
  int32_t (*device_of_ptr)(const void* ptr);  // -1 if unknown
  void (*record_stream)(void* handle, int32_t device, int64_t stream);
  // devices
  int32_t (*device_count)(void);
  void (*synchronize_device)(int32_t device);
  // streams (id 0 is the default stream of every device)
  int64_t (*new_stream)(int32_t device, int32_t priority);
  int64_t (*stream_from_pool)(int32_t device, int32_t high_priority);
  void (*synchronize_stream)(int32_t device, int64_t stream);
  int32_t (*query_stream)(int32_t device, int64_t stream);
  void* (*stream_native_handle)(int32_t device, int64_t stream);
  // events
  TmbEvent (*event_create)(int32_t device, int32_t enable_timing);
  void (*event_destroy)(TmbEvent ev, int32_t device);
  void (*event_record)(TmbEvent ev, int32_t device, int64_t stream);
  void (*event_block)(TmbEvent ev, int32_t device, int64_t stream);  // stream waits on ev
  int32_t (*event_query)(TmbEvent ev);
  void (*event_synchronize)(TmbEvent ev);
  double (*event_elapsed_ms)(TmbEvent start, TmbEvent end);
  // profiler ranges (may be NULL)
  void (*prof_mark)(const char* name);
  void (*prof_range_push)(const char* name);
  void (*prof_range_pop)(void);
  // Memory stats: 4 slots (current, peak, allocated, freed) for each of
  // allocated_bytes.all, allocation.all, requested_bytes.all, reserved_bytes.all,
  // then num_device_alloc, num_device_free, num_alloc_retries, num_ooms.
  // No other DeviceStats concepts exist in Mojo. n bounds every write; a
  // mismatched table returns an error instead of writing past the buffer.
  void (*mem_stats)(int32_t device, int64_t* out, int32_t n);
  void (*mem_reset_peak)(int32_t device);
  void (*mem_reset_accumulated)(int32_t device);
  void (*empty_cache)(void);
  int32_t (*mem_get_info)(int32_t device, size_t* free, size_t* total);
  // Properties: major, minor, total_memory, multi_processor_count,
  // max_threads_per_multi_processor, warp_size, regs_per_multiprocessor,
  // max_threads_per_block, regs_per_block, shared_memory_per_block,
  // shared_memory_per_block_optin, shared_memory_per_multiprocessor,
  // max_blocks_per_multi_processor, clock_rate (kHz), max_grid_dim_x.
  // Unknown numbers are -1. text holds name\0api\0arch_name\0 (UTF-8).
  int32_t (*device_props)(int32_t device, int64_t* out, int32_t n,
                          char* text, int32_t text_cap);
  // pinned (page-locked) host memory: torch's getPinnedMemoryAllocator /
  // isPinnedPtr. `data` receives the host pointer; the return value is the
  // opaque handle to pass to host_free. NULL on failure.
  void* (*host_alloc)(size_t nbytes, int32_t device, void** data);
  void (*host_free)(void* handle);
  int32_t (*is_pinned_ptr)(const void* ptr);  // 1 if page-locked by Mojo or CUDA
} TmbBackendHooks;

enum { TMB_MEM_STATS_SLOTS = 20, TMB_DEVICE_PROPS_SLOTS = 15 };

int32_t tmb_device_properties(int32_t device, int64_t* out, int32_t n,
                              char* text, int32_t text_cap);

// ---- registration ------------------------------------------------------------
int32_t tmb_backend_register(const TmbBackendHooks* hooks);       // once per process
TmbLibrary tmb_library_new(const char* ns, const char* dispatch_key);
int32_t tmb_library_impl(TmbLibrary lib, const char* name, TmbKernelFn fn, void* ctx);
int32_t tmb_library_fallback(TmbLibrary lib, TmbKernelFn fn, void* ctx);
// Registers ATen's own kernel for one of the metadata-only view ops
// ("view", "_reshape_alias", "as_strided"), unboxed: shim_views.cpp.
int32_t tmb_library_impl_aten_view(TmbLibrary lib, const char* name);
// autocast policy for AutocastPrivateUse1: 1 lower_precision_fp, 2 fp32,
// 3 fp32_set_opt_dtype, 4 promote. Names are "aten::op" or "aten::op.overload".
int32_t tmb_autocast_policy(const char* qualified_name, int32_t policy);
int32_t tmb_autocast_install_cuda_policies(void);  // torch's own CUDA lists, verbatim
void tmb_set_error(const char* message);  // thread-local, read by the adapter on failure
const char* tmb_get_error(void);
// the backend mutex, for callers that reach Mojo outside the boxed adapter (process group)
void tmb_lock(void);
void tmb_unlock(void);
// 1 in a child forked after registration: the runtime is unusable there
// (torch.mojo._is_in_bad_fork; device use raises)
int32_t tmb_is_in_bad_fork(void);
// thread-local current device / stream, as torch's device guard sees them
int32_t tmb_current_device(void);
void tmb_set_current_device(int32_t device);
int64_t tmb_current_stream(int32_t device);
void tmb_set_current_stream(int32_t device, int64_t stream);

// ---- tensors (handles are at::Tensor*) ---------------------------------------
// Everything abi.mojo's `T` view reads about a tensor, in ONE call: `out` is
// TMB_TENSOR_INFO_SLOTS int64 slots indexed by the enum below. The sizes and
// strides slots are pointers into the tensor's own metadata, valid while it is
// alive and its shape unchanged -- which is the whole conversion, so the
// per-field getters below stay for the occasional single read.
enum TmbTensorInfoSlot : int32_t {
  TMB_INFO_DATA_PTR = 0,
  TMB_INFO_DIM = 1,
  TMB_INFO_SIZES = 2,
  TMB_INFO_STRIDES = 3,
  TMB_INFO_STORAGE_OFFSET = 4,
  TMB_INFO_NUMEL = 5,
  TMB_INFO_DTYPE = 6,
  TMB_INFO_CONTIGUOUS = 7,
  TMB_INFO_DEVICE_INDEX = 8,  // -1 when not on PrivateUse1
  TMB_INFO_DEVICE_TYPE = 9,
  TMB_TENSOR_INFO_SLOTS = 10,
};
void tmb_tensor_info(TmbTensor t, int64_t* out);
void* tmb_tensor_data_ptr(TmbTensor t);
int64_t tmb_tensor_dim(TmbTensor t);
const int64_t* tmb_tensor_sizes(TmbTensor t);
const int64_t* tmb_tensor_strides(TmbTensor t);
int64_t tmb_tensor_storage_offset(TmbTensor t);
int64_t tmb_tensor_numel(TmbTensor t);
int32_t tmb_tensor_dtype(TmbTensor t);
int32_t tmb_tensor_device_index(TmbTensor t);  // -1 when not on PrivateUse1
int32_t tmb_tensor_device_type(TmbTensor t);   // c10::DeviceType
int32_t tmb_tensor_is_privateuse1(TmbTensor t);
void* tmb_tensor_storage_data_ptr(TmbTensor t);
void* tmb_tensor_storage_ctx(TmbTensor t);  // the allocation handle Mojo returned from alloc (NULL if not ours)
int64_t tmb_tensor_storage_nbytes(TmbTensor t);
int32_t tmb_tensor_is_contiguous(TmbTensor t);
int32_t tmb_tensor_is_neg(TmbTensor t);
int32_t tmb_tensor_requires_grad(TmbTensor t);
void tmb_tensor_bump_version(TmbTensor t);
int32_t tmb_float32_matmul_precision(void);  // torch.get_float32_matmul_precision: 0 highest, 1 high, 2 medium
int32_t tmb_grad_enabled(void);
// Whether CUDA considers `ptr` page-locked. Upstream's pin_memory=True factory
// prefers CUDA's allocator while is_pinned() prefers PrivateUse1, so a pointer
// we did not allocate may still be genuinely pinned.
int32_t tmb_cuda_is_pinned_ptr(const void* ptr);
void* tmb_stream_native_handle(int32_t device, int64_t stream);  // the vendor (CUDA/HIP) stream of a mojo stream  // at::GradMode::is_enabled(): whether autograd records this call
TmbTensor tmb_tensor_retain(TmbTensor t);   // new owned handle to the same tensor
void tmb_tensor_release(TmbTensor t);
// Allocation through the registered allocator, no dispatcher round trip.
// `info` (optional) receives the new tensor's slots, so the caller does not
// read back what the creating call already knows.
int32_t tmb_empty_strided(int64_t ndim, const int64_t* sizes, const int64_t* strides,
                          int32_t dtype, int32_t device, TmbTensor* ret, int64_t* info);
// zero-copy view over base's storage
int32_t tmb_as_strided(TmbTensor base, int64_t ndim, const int64_t* sizes,
                       const int64_t* strides, int64_t storage_offset, TmbTensor* ret,
                       int64_t* info);
// in-place metadata changes (set_/resize_/as_strided_)
int32_t tmb_tensor_set_sizes_strides(TmbTensor t, int64_t ndim, const int64_t* sizes,
                                     const int64_t* strides, int64_t storage_offset);
int32_t tmb_tensor_set_storage(TmbTensor t, TmbTensor source);
int32_t tmb_storage_resize(TmbTensor t, int64_t nbytes);
// CPU tensors (for host round trips such as .item() and _local_scalar_dense)
int32_t tmb_cpu_empty(int64_t ndim, const int64_t* sizes, int32_t dtype, TmbTensor* ret);
int32_t tmb_cpu_empty_pinned(int64_t ndim, const int64_t* sizes, int32_t dtype, int32_t device, TmbTensor* ret);
// Philox state lives in the C++ generators (one default per device, plus any
// torch.Generator(device="mojo")). reserve() hands back the (seed, offset) before
// bumping the offset by `increment`; the 16-byte state is (seed, offset) little-endian.
int32_t tmb_philox_reserve(TmbGenerator gen, int32_t device, uint64_t increment,
                           uint64_t* seed, uint64_t* offset);
int32_t tmb_rng_manual_seed(int32_t device, uint64_t seed);  // device -1: all devices
int32_t tmb_rng_get_state(int32_t device, uint8_t* out16);
int32_t tmb_rng_set_state(int32_t device, const uint8_t* in16);
int32_t tmb_default_dtype(void);
int32_t tmb_alert_not_deterministic(const char* caller);
// test support: per-op call counters ("aten::add.Tensor"), off by default
void tmb_op_counting(int32_t enabled);
void tmb_op_counts_reset(void);
int64_t tmb_op_count(const char* qualified_name);
int64_t tmb_op_counts_dump(char* buf, int64_t cap);
// call any aten op through the dispatcher (composites, CPU fallbacks); rets as in kernels
int32_t tmb_call_op(const char* op, const char* overload, const TmbValue* args,
                    int32_t n_args, TmbValue* rets, int32_t n_rets);

#ifdef __cplusplus
}
#pragma GCC visibility pop
#endif
