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
// The MAX runtime is not fork-safe: its worker threads and device contexts do
// not exist in a forked child, and a device call there waits forever on a
// thread that is gone. tmb_backend_register() installs a pthread_atfork child
// handler that sets this flag, and every entry that reaches the runtime --
// allocation, the boxed kernel -- refuses with a message that names the fix
// (the 'spawn' start method), as CUDA does. agents_docs/native_backend.md, "Fork".
extern bool tmb_in_bad_fork;
void tmb_check_not_forked();
std::string& tmb_thread_error();
void tmb_count_op_call(const char* qualified_name);  // shim_dispatch.cpp
