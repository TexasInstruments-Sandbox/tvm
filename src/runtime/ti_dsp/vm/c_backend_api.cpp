/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file vm/c_backend_api.cpp
 * \brief TVM C Backend API implementation for DSP Runtime using C++14
 *
 * This is a C++ implementation using:
 * - FixedVector for the symbol registry (no heap allocation)
 * - TypedHandle for type-safe FFI object access
 *
 * The C API (extern "C") is preserved for compatibility.
 */

/* C headers with C++ templates in them - include outside extern "C" */
#include "../core/config.h"
#include "../platform/dsp_platform.h"
#include "../platform/dsp_memory.h"
#include "../ffi/ffi_types.h"
#include "../registry/registry.h"
#include "vm_builtins.h"

#include "../cpp/fixed_vector.h"
#include "../cpp/typed_handle.h"

#include <cstring>

/*
 * =============================================================================
 * SYMBOL REGISTRY (using FixedVector)
 * =============================================================================
 */

namespace {

/* Maximum number of registered symbols */
constexpr size_t kMaxSymbols = 128;

/* Symbol registry entry */
struct SymbolEntry {
  const char* name;
  void* ptr;
};

/* Static symbol table using FixedVector */
tvm::dsp::FixedVector<SymbolEntry, kMaxSymbols> g_symbol_registry;

/* Initialization flag for VM builtins */
bool g_builtins_initialized = false;

/* Thread-local error storage (on DSP, just static since single-threaded) */
TVMFFIObjectHandle g_last_error = nullptr;

}  // namespace

/*
 * =============================================================================
 * C API IMPLEMENTATION
 * =============================================================================
 */

extern "C" {

/* Forward declarations */
int TVMBackendGetFuncFromGlobalRegistry(const char* func_name, TVMFFIObjectHandle* out);

/* ---------------------------------------------------------------------------
 * Workspace Memory Management
 *
 * These functions are called by TVM-generated kernels for temporary workspace.
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Allocate temporal workspace memory
 *
 * Called by TVM-generated code for temporary buffers during computation.
 * Must be aligned to kTempAllocaAlignment (typically 64 bytes).
 *
 * \param device_type Device type (ignored on DSP - single device)
 * \param device_id Device ID (ignored on DSP - single device)
 * \param nbytes Number of bytes to allocate
 * \param dtype_code_hint Type code hint (ignored)
 * \param dtype_bits_hint Type bits hint (ignored)
 * \return Pointer to allocated memory, or NULL on failure
 */
void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes,
                               int dtype_code_hint, int dtype_bits_hint) {
  (void)device_type;
  (void)device_id;
  (void)dtype_code_hint;
  (void)dtype_bits_hint;

  /* Allocate from DDR only.  L2 SRAM is managed by the inline bump
   * allocator (tvm_l2_alloc/tvm_l2_reset) emitted in generated code
   * for buffers with global.l2sram scope.  Workspace allocations here
   * are always DDR-backed temporaries. */
  return tvm_dsp_alloc(static_cast<size_t>(nbytes), TVM_DSP_CACHE_LINE_SIZE, TVM_DSP_MEM_MAIN);
}

/*!
 * \brief Free temporal workspace memory
 *
 * \param device_type Device type (ignored on DSP)
 * \param device_id Device ID (ignored on DSP)
 * \param ptr Pointer to free
 * \return 0 on success, -1 on failure
 */
int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  (void)device_type;
  (void)device_id;

  if (ptr != nullptr) {
    tvm_dsp_free(ptr);
  }
  return 0;
}

/* ---------------------------------------------------------------------------
 * Function Registry
 *
 * TVM-generated code calls these to look up functions by name.
 * The actual registry is implemented in registry/registry.cpp.
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Get function from module environment
 *
 * For DSP, modules don't have separate environments, so this just
 * looks up in the global registry.
 *
 * \param mod_node Module handle (ignored on DSP)
 * \param func_name Function name to look up
 * \param out Output function handle
 * \return 0 on success, -1 if not found
 */
int TVMBackendGetFuncFromEnv(void* mod_node, const char* func_name,
                             TVMFFIObjectHandle* out) {
  (void)mod_node;
  return TVMBackendGetFuncFromGlobalRegistry(func_name, out);
}

/*!
 * \brief Get function from global registry
 *
 * \param func_name Function name to look up
 * \param out Output function handle
 * \return 0 on success, -1 if not found
 */
int TVMBackendGetFuncFromGlobalRegistry(const char* func_name,
                                        TVMFFIObjectHandle* out) {
  if (func_name == nullptr || out == nullptr) {
    return -1;
  }

  /* Lazy initialization of VM builtins */
  if (!g_builtins_initialized) {
    g_builtins_initialized = true;
    TVMDSPRegisterVMBuiltins();
  }

  /* Look up in centralized registry */
  *out = TVMRegistryLookup(func_name);
  if (*out != nullptr) {
    return 0;
  }

  /* Not found */
  return -1;
}

/* ---------------------------------------------------------------------------
 * System Library Symbol Registration
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Register a system-wide library symbol
 *
 * Called during static initialization to register exported symbols.
 *
 * \param name Symbol name
 * \param ptr Symbol address
 * \return 0 on success, -1 on failure
 */
int TVMBackendRegisterSystemLibSymbol(const char* name, void* ptr) {
  if (name == nullptr || g_symbol_registry.full()) {
    return -1;
  }

  SymbolEntry entry;
  entry.name = name;
  entry.ptr = ptr;

  if (!g_symbol_registry.push_back(entry)) {
    return -1;
  }

  return 0;
}

/*!
 * \brief Look up a system symbol by name
 *
 * \param name Symbol name
 * \return Symbol address, or NULL if not found
 */
void* TVMDSPGetSymbol(const char* name) {
  if (name == nullptr) {
    return nullptr;
  }

  for (const auto& entry : g_symbol_registry) {
    if (entry.name != nullptr && std::strcmp(entry.name, name) == 0) {
      return entry.ptr;
    }
  }
  return nullptr;
}

/* ---------------------------------------------------------------------------
 * Parallel Execution (Single-threaded stubs for DSP)
 *
 * On single-core DSP, we execute sequentially.
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Launch parallel jobs (single-threaded on DSP)
 *
 * On single-core DSP, this simply executes the lambda sequentially
 * with task_id from 0 to num_task-1.
 *
 * \param flambda Function to execute for each task
 * \param cdata Closure data passed to flambda
 * \param num_task Number of tasks (0 = 1 task on DSP)
 * \return 0 on success, -1 on failure
 */
int TVMBackendParallelLaunch(int (*flambda)(int task_id, void* penv, void* cdata),
                             void* cdata, int num_task) {
  /* Single-threaded parallel group environment */
  struct ParallelEnv {
    void* sync_handle;
    int32_t num_task;
  };

  ParallelEnv penv;
  penv.sync_handle = nullptr;

  /* On single-core DSP, use 1 task if num_task is 0 */
  int actual_tasks = (num_task == 0) ? 1 : num_task;
  penv.num_task = actual_tasks;

  /* Execute tasks sequentially */
  for (int task_id = 0; task_id < actual_tasks; task_id++) {
    int ret = flambda(task_id, &penv, cdata);
    if (ret != 0) {
      return ret;
    }
  }

  return 0;
}

/*!
 * \brief BSP barrier between parallel threads (no-op on single-core DSP)
 *
 * \param task_id Current task ID
 * \param penv Parallel environment
 * \return 0 (always succeeds on single core)
 */
int TVMBackendParallelBarrier(int task_id, void* penv) {
  (void)task_id;
  (void)penv;
  /* No-op on single-core DSP */
  return 0;
}

/* ---------------------------------------------------------------------------
 * Static Initialization
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Run a function once (for static initialization)
 *
 * \param handle Pointer to static handle (initially NULL)
 * \param f Function to run once
 * \param cdata Closure data for function
 * \param nbytes Size of closure data (unused)
 * \return 0 on success, -1 on failure
 */
int TVMBackendRunOnce(void** handle, int (*f)(void*), void* cdata, int nbytes) {
  (void)nbytes;

  if (*handle == nullptr) {
    int ret = f(cdata);
    if (ret != 0) {
      return ret;
    }
    /* Mark as initialized with non-NULL value */
    *handle = reinterpret_cast<void*>(1);
  }
  return 0;
}

/* ---------------------------------------------------------------------------
 * Error Handling
 * ---------------------------------------------------------------------------*/

/*!
 * \brief Set raised error from C string
 *
 * Called by TVM-generated code when an error occurs.
 *
 * \param kind Error kind (e.g., "RuntimeError")
 * \param message Error message
 */
void TVMFFIErrorSetRaisedFromCStr(const char* kind, const char* message) {
  /* On DSP, just log the error - we don't create full error objects */
  (void)kind;
  tvm_dsp_log("ERROR [%s]: %s\n", kind ? kind : "Unknown", message ? message : "");
  /* For now, we don't create a proper error object */
  g_last_error = nullptr;
}

/*!
 * \brief Set raised error object
 *
 * \param error Error object handle
 */
void TVMFFIErrorSetRaised(TVMFFIObjectHandle error) {
  g_last_error = error;
}

/*!
 * \brief Move the last error from environment
 *
 * \param result Output error handle
 */
void TVMFFIErrorMoveFromRaised(TVMFFIObjectHandle* result) {
  *result = g_last_error;
  g_last_error = nullptr;
}

/* Note: TVMFFIObjectFree is provided by ffi_types.c */

}  /* extern "C" */
