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
 * \file codegen_c_static_templates.h
 * \brief String templates for C static code generation
 *
 * This file contains string literal templates used by CodeGenCStatic
 * for emitting header declarations, helper functions, and other
 * boilerplate code. Separating these templates from the main code
 * generator improves readability and maintainability.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_C_STATIC_TEMPLATES_H_
#define TVM_TARGET_SOURCE_CODEGEN_C_STATIC_TEMPLATES_H_

namespace tvm {
namespace codegen {
namespace templates {

// ============================================================================
// DSP Target Templates
// ============================================================================

/*!
 * \brief DSP target header includes and declarations
 *
 * Includes lightweight DSP runtime headers (no C++ exceptions/RTTI)
 * and declares backend API functions for workspace management.
 */
constexpr const char* kDSPHeaders = R"(
// TI DSP target configuration
#ifdef __TI_COMPILER_VERSION__
  #if defined(__C7000__)
    #include <c7x.h>
  #else
    #include <c6x.h>
  #endif
#endif

// Dynamic export attribute (TI compiler only, for DLOAD visibility)
#ifdef __TI_COMPILER_VERSION__
  #define TVM_DSP_EXPORT __declspec(dllexport)
#else
  #define TVM_DSP_EXPORT
#endif

// TVM DSP Runtime headers (lightweight, no C++ exceptions/RTTI)
#include "ffi/ffi_types.h"
#include "container/ndarray.h"
#include "vm/vm_builtins.h"
#include "vm/storage.h"
#include "registry/registry.h"
#include "platform/cycle_counter.h"
#include "cpp/vm_array.h"
#include "cpp/vm_builtins.h"
#include "dma/tvm_dsp_dma.h"
#include "kernels/tvm_int8_residual_add.h"
#include "kernels/tidl_activation_wrappers.h"
#include "kernels/tidl_avgpool_wrappers.h"
#include "kernels/tidl_norm_wrappers.h"
#include "kernels/c7x_pool_relu_wrappers.h"
#include "mmalib/tidl_maxpool_wrapper.h"
#include "kernels/tvm_dequantize_vecmatmul.h"
#include "kernels/tvm_sdpa_decode.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>

typedef unsigned long ulong;

using std::max;
using std::min;
using std::fmax;
using std::fmin;
using std::fabs;

// TVM Backend API declarations
extern "C" void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t nbytes,
                                          int dtype_code_hint, int dtype_bits_hint);
extern "C" int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr);

// C API for constants - returns raw TVMFFIAny array (no std::vector)
extern "C" TVMFFIAny* TVMDSPConstantsGet(int* count);

// ---------------------------------------------------------------
// L2 SRAM bump allocator for DMA tiling
//
// Base address and pool size come from firmware getter functions
// resolved by DLOAD at load time.  The allocator itself is
// inlined -- only tvm_l2_reset() calls through the GOT (once
// per kernel).  Falls back to DDR if L2 is exhausted.
// ---------------------------------------------------------------
extern "C" uint8_t* tvm_dsp_get_l2_base(void);
extern "C" uint32_t tvm_dsp_get_l2_size(void);

static uint8_t* tvm_l2_ptr;
static uint32_t tvm_l2_avail;

static inline void* tvm_l2_alloc(uint32_t nbytes) {
    nbytes = (nbytes + 127u) & ~127u;  // 128-byte align
    if (nbytes <= tvm_l2_avail) {
        uint8_t* p = tvm_l2_ptr;
        tvm_l2_ptr += nbytes;
        tvm_l2_avail -= nbytes;
        return p;
    }
    // L2 exhausted -- fall back to DDR
    return TVMBackendAllocWorkspace(1, 0, (uint64_t)nbytes, 2, 32);
}

static inline void tvm_l2_reset(void) {
#if defined(__C7000__) && !defined(C7X_HOST_EMULATION)
    /* Hardcode L2 base on real C7x hardware to bypass PLT call.
     * C7X_HOST_EMULATION is defined by the c7x_host build system to
     * distinguish host emulation (GCC + TI Host Emu headers) from
     * actual C7x hardware (TI cl7x compiler). */
    tvm_l2_ptr = (uint8_t*)0x7E000000;
    tvm_l2_avail = 0x140000;
#else
    tvm_l2_ptr = tvm_dsp_get_l2_base();
    tvm_l2_avail = tvm_dsp_get_l2_size();
#endif
}

)";

/*!
 * \brief DSP helper functions for FFI value manipulation
 *
 * These inline functions provide a clean interface for setting
 * TVMFFIAny values and unwrapping ObjectRef arguments.
 */
constexpr const char* kDSPHelperFunctions = R"(
// Helper function for unwrapping ObjectRef arguments to raw pointers
// TVMFFIObject header size depends on architecture (includes deleter pointer union)
inline void* UnwrapObjectRefArg(const TVMFFIAny& arg) {
  // Use actual sizeof(TVMFFIObject) to handle platform differences correctly
  constexpr size_t kObjectRefHeaderSize = sizeof(TVMFFIObject);
  if (arg.type_index == kTVMFFITensor) {
    return reinterpret_cast<void*>(
        reinterpret_cast<char*>(arg.v_ptr) + kObjectRefHeaderSize);
  }
  return arg.v_ptr;
}

// Helper functions for setting TVMFFIAny values
inline void SetFFIAnyInt(TVMFFIAny* any, int64_t value) {
  any->type_index = kTVMFFIInt;
  any->v_int64 = value;
}

inline void SetFFIAnyFloat(TVMFFIAny* any, double value) {
  any->type_index = kTVMFFIFloat;
  any->v_float64 = value;
}

inline void SetFFIAnyPtr(TVMFFIAny* any, void* value) {
  any->type_index = kTVMFFIOpaquePtr;
  any->v_int64 = 0;
  any->v_ptr = value;
}

inline void SetFFIAnyNone(TVMFFIAny* any) {
  any->type_index = kTVMFFINone;
  any->v_int64 = 0;
}

)";

/*!
 * \brief DSP FFI dispatch declarations (used when C++ API is disabled)
 *
 * These declarations are only needed when using FFI dispatch mode.
 * When C++ API mode is enabled, these are omitted from generated code.
 */
constexpr const char* kDSPFFIDeclarations = R"(
// FFI function registry lookup (only needed for indirect VM builtin calls)
extern "C" int TVMBackendGetFuncFromGlobalRegistry(const char* name, void** out);

// TVM_FFI_SAFE_CALL macros for generated code (no-op on DSP, no exceptions)
#ifndef TVM_FFI_SAFE_CALL_BEGIN
#define TVM_FFI_SAFE_CALL_BEGIN() do {
#endif
#ifndef TVM_FFI_SAFE_CALL_END
#define TVM_FFI_SAFE_CALL_END() return 0; } while(0)
#endif

)";

/*!
 * \brief DSP FFI helper functions (used when C++ API is disabled)
 *
 * These functions implement the AnyList operations for FFI dispatch mode.
 * They handle reference counting and value movement between FFI arrays.
 */
constexpr const char* kDSPFFIHelpers = R"(
// DSP-compatible AnyList operations (simplified, no tvm::ffi::Any)
// Note: TVMFFIAnyIsObject is provided by ffi_types.h

inline int TVMBackendAnyListSetPackedArg(void* anylist, int index, TVMFFIAny* args, int arg_offset) {
  TVM_FFI_SAFE_CALL_BEGIN();
  TVMFFIAny* list = static_cast<TVMFFIAny*>(anylist);
  args[arg_offset] = list[index];
  TVM_FFI_SAFE_CALL_END();
}

inline int TVMBackendAnyListResetItem(void* anylist, int index) {
  TVM_FFI_SAFE_CALL_BEGIN();
  TVMFFIAny* list = static_cast<TVMFFIAny*>(anylist);
  // Release old value if it was an object
  if (TVMFFIAnyIsObject(&list[index]) && list[index].v_obj != nullptr) {
    TVMFFIObject* obj = static_cast<TVMFFIObject*>(list[index].v_obj);
    if (obj->ref_counter > 0) {
      obj->ref_counter--;
      if (obj->ref_counter == 0 && obj->deleter != nullptr) {
        obj->deleter(obj);
      }
    }
  }
  TVMFFIAnySetNone(&list[index]);
  TVM_FFI_SAFE_CALL_END();
}

inline int TVMBackendAnyListMoveFromPackedReturn(void* anylist, int index, TVMFFIAny* args,
                                                 int ret_offset) {
  TVM_FFI_SAFE_CALL_BEGIN();
  TVMFFIAny* list = static_cast<TVMFFIAny*>(anylist);
  // First, release old value if it was an object
  if (TVMFFIAnyIsObject(&list[index]) && list[index].v_obj != nullptr) {
    TVMFFIObject* obj = static_cast<TVMFFIObject*>(list[index].v_obj);
    if (obj->ref_counter > 0) {
      obj->ref_counter--;
      if (obj->ref_counter == 0 && obj->deleter != nullptr) {
        obj->deleter(obj);
      }
    }
  }
  // Move the value (no ref count change - it's a move, not copy)
  list[index] = args[ret_offset];
  TVMFFIAnySetNone(&args[ret_offset]);  // Clear source after move
  TVM_FFI_SAFE_CALL_END();
}

)";

/*!
 * \brief DSP layer profiling infrastructure
 *
 * When profile-layers is enabled, this code provides per-layer
 * cycle counting and result printing functionality.
 */
constexpr const char* kDSPProfilingExterns = R"(
// Layer profiling extern declarations (shared across compilation units)
#include <inttypes.h>
#define TVM_PROFILE_MAX_LAYERS 1024
extern uint64_t _tvm_layer_cycles[TVM_PROFILE_MAX_LAYERS];
extern const char* _tvm_layer_names[TVM_PROFILE_MAX_LAYERS];
extern int _tvm_layer_count;

)";

constexpr const char* kDSPProfilingDefinitions = R"(
// Layer profiling definitions (main file only)
uint64_t _tvm_layer_cycles[TVM_PROFILE_MAX_LAYERS];
const char* _tvm_layer_names[TVM_PROFILE_MAX_LAYERS];
int _tvm_layer_count = 0;

// Print layer profiling results (exported for firmware to call after cycle recording)
extern "C" TVM_DSP_EXPORT void TVMPrintLayerProfile(void) {
  printf("\n===== TVM Layer Profile =====\n");
  uint64_t total_cycles = 0;
  for (int i = 0; i < _tvm_layer_count; i++) {
    printf("[%3d] %-40s %" PRIu64 " cycles\n", i, _tvm_layer_names[i],
           _tvm_layer_cycles[i]);
    total_cycles += _tvm_layer_cycles[i];
  }
  printf("-----------------------------\n");
  printf("Total: %" PRIu64 " cycles (%d layers)\n", total_cycles, _tvm_layer_count);
  printf("=============================\n");
}

)";

/*!
 * \brief DSP debug allocation tracing infrastructure
 *
 * When debug-alloc is enabled, this code provides diagnostic tracing
 * for every AllocStorage and TVMBackendAllocWorkspace call.  Helps
 * identify OOM failures by printing request sizes, pool free space,
 * and result pointers.
 */
constexpr const char* kDSPDebugAllocInfrastructure = R"(
// Debug allocation tracing infrastructure
#include <inttypes.h>
static int _tvm_alloc_storage_count = 0;

// tvm_dsp_get_free_memory() is declared via dsp_platform.h (included through
// registry.h -> packed_func.h -> object.h -> dsp_platform.h)

static inline void _tvm_debug_alloc_storage_pre(int idx, int64_t size) {
  size_t l2_free = tvm_dsp_get_free_memory(TVM_DSP_MEM_FAST);
  size_t main_free = tvm_dsp_get_free_memory(TVM_DSP_MEM_MAIN);
  printf("[alloc-storage #%d] request %" PRId64 " bytes  "
         "(L2 free: %u, Main free: %u)\n",
         idx, size, (unsigned)l2_free, (unsigned)main_free);
}

static inline void _tvm_debug_alloc_storage_post(int idx, void* result) {
  if (result) {
    printf("[alloc-storage #%d] -> %p OK\n", idx, result);
  } else {
    size_t l2_free = tvm_dsp_get_free_memory(TVM_DSP_MEM_FAST);
    size_t main_free = tvm_dsp_get_free_memory(TVM_DSP_MEM_MAIN);
    printf("[alloc-storage #%d] -> NULL FAILED!  "
           "(L2 free: %u, Main free: %u)\n",
           idx, (unsigned)l2_free, (unsigned)main_free);
  }
}

// Note: Workspace calls (TVMBackendAllocWorkspace/Free) are not wrapped here.
// They go through tvm_dsp_alloc() which has unconditional OOM logging, so
// workspace failures will still be reported via the runtime layer.

// Print allocation summary (exported for firmware to call post-inference)
extern "C" TVM_DSP_EXPORT void TVMPrintAllocSummary(void) {
  printf("\n===== TVM Alloc Debug Summary =====\n");
  printf("Total AllocStorage calls: %d\n", _tvm_alloc_storage_count);
  size_t l2_free = tvm_dsp_get_free_memory(TVM_DSP_MEM_FAST);
  size_t main_free = tvm_dsp_get_free_memory(TVM_DSP_MEM_MAIN);
  printf("Final pool state: L2 free=%u, Main free=%u\n",
         (unsigned)l2_free, (unsigned)main_free);
  printf("===================================\n");
}

)";

// ============================================================================
// Standard TVM Target Templates
// ============================================================================

/*!
 * \brief Standard TVM target header includes and declarations
 *
 * Includes full TVM runtime headers with C++ exception/RTTI support.
 */
constexpr const char* kStandardTVMHeaders = R"(
// Custom backend for C Static code generation

typedef unsigned long ulong;
#include <tvm/runtime/logging.h>
#include <tvm/runtime/vm/executable.h>
#include <tvm/runtime/vm/vm.h>
#include <tvm/runtime/tensor.h>
#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/memory/memory_manager.h>
#include <vector>
#include <cstdint>
#include <cmath>

using tvm::runtime::ObjectRef;
using tvm::runtime::Tensor;
using tvm::runtime::memory::AllocatorType;
using tvm::ffi::String;
using tvm::ffi::Array;
using std::max;
using std::min;
using std::fmax;
using std::fmin;
using std::fabs;

extern std::vector<tvm::ffi::Any> TVMGetConstants();

// Helper function for unwrapping ObjectRef arguments to raw pointers
// ObjectRef wraps pointer with TVMFFIObject header (ref count + type info + deleter)
// Use kTVMFFITensor from tvm/ffi/c_api.h (available via c_backend_api.h)
inline void* UnwrapObjectRefArg(const TVMFFIAny& arg) {
  constexpr size_t kObjectRefHeaderSize = sizeof(TVMFFIObject);
  if (arg.type_index == kTVMFFITensor) {
    return reinterpret_cast<void*>(
        reinterpret_cast<char*>(arg.v_ptr) + kObjectRefHeaderSize);
  }
  return arg.v_ptr;
}

// Helper functions for setting TVMFFIAny values
// These combine type_index and value assignment into a single call
// Note: We use distinct names rather than overloading to avoid ambiguity
// with implicit conversions (e.g., long might match both int64_t and double)
inline void SetFFIAnyInt(TVMFFIAny* any, int64_t value) {
  any->type_index = kTVMFFIInt;
  any->v_int64 = value;
}

inline void SetFFIAnyFloat(TVMFFIAny* any, double value) {
  any->type_index = kTVMFFIFloat;
  any->v_float64 = value;
}

inline void SetFFIAnyPtr(TVMFFIAny* any, void* value) {
  any->type_index = kTVMFFIOpaquePtr;
  any->v_int64 = 0; // Clear padding
  any->v_ptr = value;
}

inline void SetFFIAnyNone(TVMFFIAny* any) {
  any->type_index = kTVMFFINone;
  any->v_int64 = 0;
}

)";

/*!
 * \brief Standard TVM FFI helper functions (used when C++ API is disabled)
 *
 * These functions use tvm::ffi::Any and full TVM runtime features.
 */
constexpr const char* kStandardTVMFFIHelpers = R"(
inline int TVMBackendAnyListSetPackedArg(void* anylist, int index, TVMFFIAny* args, int arg_offset) {
  using namespace tvm::runtime;
  TVM_FFI_SAFE_CALL_BEGIN();
  auto* list = static_cast<TVMFFIAny*>(anylist);
  args[arg_offset] = list[index];
  TVM_FFI_SAFE_CALL_END();
}

inline int TVMBackendAnyListResetItem(void* anylist, int index) {
  using namespace tvm::runtime;
  TVM_FFI_SAFE_CALL_BEGIN();
  auto* list = static_cast<tvm::ffi::Any*>(anylist);
  list[index] = nullptr;
  TVM_FFI_SAFE_CALL_END();
}

inline int TVMBackendAnyListMoveFromPackedReturn(void* anylist, int index, TVMFFIAny* args,
                                          int ret_offset) {
  using namespace tvm::runtime;
  TVM_FFI_SAFE_CALL_BEGIN();
  auto* list = static_cast<tvm::ffi::Any*>(anylist);
  list[index] = tvm::ffi::details::AnyUnsafe::MoveTVMFFIAnyToAny(&args[ret_offset]);
  TVM_FFI_SAFE_CALL_END();
}

)";

}  // namespace templates
}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_C_STATIC_TEMPLATES_H_
