#pragma once
#include "tmb.h"
#include <mutex>
#include <string>

// MAX's DeviceContext is not documented thread-safe and torch runs forward
// (main thread) and backward (autograd engine) concurrently: one recursive
// mutex around every call into Mojo. Recursive because kernels call back into
// the shim (allocation, tmb_call_op) while holding it.
extern std::recursive_mutex tmb_mutex;
extern TmbBackendHooks tmb_hooks;
extern bool tmb_ready;
std::string& tmb_thread_error();
