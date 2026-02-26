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
 * \file registry/registry.cpp
 * \brief Function registry implementation using C++14 utilities
 *
 * This is a C++ implementation of the function registry that uses the
 * C++14 migration utilities (FixedVector, TypedHandle, Span) while
 * maintaining the C API interface for compatibility with TVM-generated code.
 *
 * =============================================================================
 * ARCHITECTURE
 * =============================================================================
 *
 * The registry has two main data structures:
 *
 * 1. PackedFunc Pool (g_packed_func_pool)
 *    - Static pool of TVMDSPPackedFunc objects
 *    - Allocates PackedFunc wrappers for raw function pointers
 *    - Uses FixedVector for safe capacity management
 *
 * 2. Registry (g_registry)
 *    - Maps function names to PackedFunc objects
 *    - Uses FixedVector for safe capacity management
 *    - Linear search (fast enough for small registries)
 *
 * The C API functions (TVMFFIFunctionCall, TVMBackendAnyList*, etc.) are
 * implemented as extern "C" wrappers around the internal C++ implementation.
 *
 * =============================================================================
 * C++ UTILITIES USED
 * =============================================================================
 *
 * - FixedVector<T, N>: Replaces raw arrays + count variables
 * - TypedHandle<T>: Type-safe handle for PackedFunc objects
 * - Span<T>: Safe view over argument arrays
 *
 * =============================================================================
 */

#include "registry.h"

extern "C" {
#include "../platform/dsp_platform.h"
}

#include "../cpp/fixed_vector.h"
#include "../cpp/typed_handle.h"
#include "../cpp/span.h"

#include <cstring>

namespace {

/*
 * =============================================================================
 * INTERNAL TYPES
 * =============================================================================
 */

/*!
 * \brief Registry entry mapping name to PackedFunc
 */
struct RegistryEntry {
  const char* name;
  TVMDSPPackedFunc* func;

  RegistryEntry() : name(nullptr), func(nullptr) {}
  RegistryEntry(const char* n, TVMDSPPackedFunc* f) : name(n), func(f) {}
};

/*
 * =============================================================================
 * GLOBAL STATE
 * =============================================================================
 *
 * These are the module-level data structures. Using FixedVector ensures:
 * - No buffer overflows
 * - Clear capacity limits
 * - Safe iteration
 */

/*! \brief Pool of PackedFunc objects */
static tvm::dsp::FixedVector<TVMDSPPackedFunc, TVM_DSP_MAX_PACKED_FUNCS>
    g_packed_func_pool;

/*! \brief Registry mapping names to functions */
static tvm::dsp::FixedVector<RegistryEntry, TVM_DSP_MAX_PACKED_FUNCS> g_registry;

/*! \brief Initialization flag */
static bool g_registry_initialized = false;

/*
 * =============================================================================
 * INTERNAL HELPER FUNCTIONS
 * =============================================================================
 */

/*!
 * \brief Find a registry entry by name
 *
 * \param name Function name to search for
 * \return Pointer to entry if found, nullptr otherwise
 */
static RegistryEntry* FindEntryByName(const char* name) {
  for (auto& entry : g_registry) {
    if (entry.name != nullptr && std::strcmp(entry.name, name) == 0) {
      return &entry;
    }
  }
  return nullptr;
}

/*!
 * \brief Create a TypedHandle from a raw FFI object handle
 *
 * This provides type-safe access to PackedFunc objects.
 */
static tvm::dsp::TypedHandle<TVMDSPPackedFunc> AsPackedFunc(
    TVMFFIObjectHandle handle) {
  return tvm::dsp::TypedHandle<TVMDSPPackedFunc>::FromRaw(handle);
}

}  // anonymous namespace

/*
 * =============================================================================
 * C API IMPLEMENTATION
 * =============================================================================
 *
 * These extern "C" functions implement the public C API defined in registry.h.
 * They wrap the internal C++ implementation.
 */

extern "C" {

/* ---------------------------------------------------------------------------
 * Registry Initialization
 * ---------------------------------------------------------------------------*/

void TVMRegistryInit(void) {
  if (g_registry_initialized) {
    return;
  }
  g_registry_initialized = true;
  g_packed_func_pool.clear();
  g_registry.clear();
}

/* ---------------------------------------------------------------------------
 * PackedFunc Pool Management
 * ---------------------------------------------------------------------------*/

TVMDSPPackedFunc* TVMRegistryAllocPackedFunc(TVMDSPPackedFuncRaw func,
                                              const char* name) {
  if (g_packed_func_pool.full()) {
    tvm_dsp_log("ERROR: PackedFunc pool exhausted (capacity=%zu)\n",
                g_packed_func_pool.capacity());
    return nullptr;
  }

  /* Create new PackedFunc in pool */
  TVMDSPPackedFunc pfunc;
  pfunc.type_index = TVM_DSP_PACKED_FUNC_TYPE_INDEX;
  pfunc.ref_counter = 1;  /* Static objects have permanent ref */
  pfunc.deleter = nullptr;  /* Static objects are not freed */
  pfunc.func = func;
  pfunc.name = name;

  if (!g_packed_func_pool.push_back(pfunc)) {
    tvm_dsp_log("ERROR: Failed to add PackedFunc to pool\n");
    return nullptr;
  }

  /* Return pointer to the entry in the pool */
  return &g_packed_func_pool[g_packed_func_pool.size() - 1];
}

TVMDSPPackedFunc* TVMDSPPackedFuncCreate(TVMDSPPackedFuncRaw func,
                                         const char* name) {
  return TVMRegistryAllocPackedFunc(func, name);
}

/* ---------------------------------------------------------------------------
 * Function Registration
 * ---------------------------------------------------------------------------*/

int TVMRegistryRegister(const char* name, TVMDSPPackedFuncRaw func) {
  if (name == nullptr || func == nullptr) {
    return -1;
  }

  /* Initialize if needed */
  if (!g_registry_initialized) {
    TVMRegistryInit();
  }

  /* Check if already registered (update existing) */
  RegistryEntry* existing = FindEntryByName(name);
  if (existing != nullptr) {
    /* Update existing entry's function pointer */
    existing->func->func = func;
    return 0;
  }

  /* Check capacity */
  if (g_registry.full()) {
    tvm_dsp_log("ERROR: Function registry full (capacity=%zu)\n",
                g_registry.capacity());
    return -1;
  }

  /* Allocate PackedFunc from pool */
  TVMDSPPackedFunc* pfunc = TVMRegistryAllocPackedFunc(func, name);
  if (pfunc == nullptr) {
    return -1;
  }

  /* Add to registry */
  if (!g_registry.push_back(RegistryEntry(name, pfunc))) {
    tvm_dsp_log("ERROR: Failed to add entry to registry\n");
    return -1;
  }

  return 0;
}

TVMFFIObjectHandle TVMRegistryLookup(const char* name) {
  if (name == nullptr) {
    return nullptr;
  }

  RegistryEntry* entry = FindEntryByName(name);
  if (entry != nullptr) {
    return static_cast<TVMFFIObjectHandle>(entry->func);
  }

  return nullptr;
}

/* ---------------------------------------------------------------------------
 * FFI Function Call
 * ---------------------------------------------------------------------------*/

int TVMFFIFunctionCall(TVMFFIObjectHandle func, TVMFFIAny* args,
                       int32_t num_args, TVMFFIAny* result) {
  /* Use TypedHandle for type-safe access */
  auto pfunc = AsPackedFunc(func);

  if (pfunc.IsNull()) {
    tvm_dsp_log("ERROR: TVMFFIFunctionCall with NULL function\n");
    return -1;
  }

  /* Verify it's a PackedFunc */
  if (pfunc->type_index != TVM_DSP_PACKED_FUNC_TYPE_INDEX) {
    tvm_dsp_log(
        "ERROR: TVMFFIFunctionCall with non-function object (type=%d)\n",
        pfunc->type_index);
    return -1;
  }

  if (pfunc->func == nullptr) {
    tvm_dsp_log("ERROR: TVMFFIFunctionCall with NULL function pointer\n");
    return -1;
  }

  /* Ensure result starts as None */
  if (result != nullptr) {
    result->type_index = kTVMFFINone;
    result->small_len = 0;
    result->v_int64 = 0;
  }

  /* Call the packed function */
  return pfunc->func(args, num_args, result);
}

/* ---------------------------------------------------------------------------
 * AnyList Helper Functions
 *
 * These use Span<TVMFFIAny> internally for bounds checking when available,
 * but the C API doesn't provide size information, so we trust the caller.
 * ---------------------------------------------------------------------------*/

int TVMBackendAnyListSetPackedArg(void* anylist, int index, TVMFFIAny* args,
                                  int arg_offset) {
  if (anylist == nullptr || args == nullptr) {
    return -1;
  }

  TVMFFIAny* src_list = static_cast<TVMFFIAny*>(anylist);
  TVMFFIAny* src = &src_list[index];
  TVMFFIAny* dst = &args[arg_offset];

  /* Copy the value */
  dst->type_index = src->type_index;
  dst->small_len = src->small_len;
  dst->v_int64 = src->v_int64;

  /*
   * Note: We intentionally do NOT increment ref_counter here.
   * The register file owns the reference, and we're just borrowing it
   * for the duration of the packed function call. The args array is
   * temporary and will be discarded after the call.
   */

  return 0;
}

int TVMBackendAnyListMoveFromPackedReturn(void* anylist, int index,
                                           TVMFFIAny* args, int ret_offset) {
  if (anylist == nullptr || args == nullptr) {
    return -1;
  }

  TVMFFIAny* list = static_cast<TVMFFIAny*>(anylist);
  TVMFFIAny* src = &args[ret_offset];

  /* Release old value if it was an object (using centralized helper) */
  TVMFFIAnyDecRef(&list[index]);

  /* Move the value (no ref count change - it's a move, not copy) */
  list[index].type_index = src->type_index;
  list[index].small_len = src->small_len;
  list[index].v_int64 = src->v_int64;

  /* Clear source to indicate move */
  src->type_index = kTVMFFINone;
  src->small_len = 0;
  src->v_int64 = 0;

  return 0;
}

int TVMBackendAnyListResetItem(void* anylist, int index) {
  if (anylist == nullptr) {
    return -1;
  }

  TVMFFIAny* list = static_cast<TVMFFIAny*>(anylist);

  /* Release object if present and set to None (using centralized helper) */
  TVMFFIAnyClear(&list[index]);

  return 0;
}

}  /* extern "C" */
