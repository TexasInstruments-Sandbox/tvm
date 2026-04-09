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
 * \file codegen_c_static_dsp.cc
 * \brief Implementation of TI DSP-specific code generation extensions.
 */
#include "codegen_c_static_dsp.h"

#include <tvm/tir/expr.h>

#include <algorithm>

#include "codegen_c_static_templates.h"

namespace tvm {
namespace codegen {

void DSPCodeGenExtension::EmitLoopPragmas(const PrimExpr& extent, tir::ForKind loop_kind,
                                          std::ostream& os,
                                          const std::function<void()>& print_indent) {
  // Collect pragmas to emit, then wrap in #ifdef only if non-empty.
  bool is_dynamic = extent.as<tir::IntImmNode>() == nullptr;
  bool is_unrolled = (loop_kind == tir::ForKind::kUnrolled) &&
                     (extent.as<tir::IntImmNode>() != nullptr);
  if (!is_dynamic && !is_unrolled) return;

  print_indent();
  os << "#ifdef __TI_COMPILER_VERSION__\n";

  // MUST_ITERATE pragma - provides trip count hints for software pipelining.
  // Only emit for dynamic bounds; the compiler already knows constant bounds.
  if (is_dynamic) {
    print_indent();
    os << "#pragma MUST_ITERATE(1, , 1)\n";
  }

  // UNROLL pragma for explicitly unrolled loops with a known bound.
  if (is_unrolled) {
    int64_t factor = std::min(extent.as<tir::IntImmNode>()->value, static_cast<int64_t>(8));
    print_indent();
    os << "#pragma UNROLL(" << factor << ")\n";
  }

  print_indent();
  os << "#endif  // __TI_COMPILER_VERSION__\n";
}

void DSPCodeGenExtension::EmitHeaders(bool use_cpp_api, bool profile_layers, bool debug_alloc,
                                      std::ostream& os) {
  // TI DSP target: Use lightweight DSP runtime headers directly
  os << templates::kDSPHeaders;
  os << templates::kDSPHelperFunctions;

  // Only emit FFI-specific declarations when C++ API is disabled
  if (!use_cpp_api) {
    os << templates::kDSPFFIDeclarations;
  }

  // Only emit FFI helper functions when C++ API is disabled
  if (!use_cpp_api) {
    os << templates::kDSPFFIHelpers;
  }

  // Add profiling infrastructure when profile-layers is enabled
  if (profile_layers) {
    os << templates::kDSPProfilingInfrastructure;
  }

  // Add debug allocation tracing infrastructure when debug-alloc is enabled
  if (debug_alloc) {
    os << templates::kDSPDebugAllocInfrastructure;
  }
}

void DSPCodeGenExtension::EmitProfilingInit(std::ostream& os,
                                            const std::function<void()>& print_indent) {
  print_indent();
  os << "// Initialize cycle counter for layer profiling\n";
  print_indent();
  os << "TVMDSPCycleCounter_init();\n";
  print_indent();
  os << "_tvm_layer_count = 0;\n\n";
}

void DSPCodeGenExtension::EmitLayerProfilingStart(int layer_idx, const std::string& layer_name,
                                                  std::ostream& os,
                                                  const std::function<void()>& print_indent) {
  print_indent();
  os << "// Layer " << layer_idx << ": " << layer_name << "\n";
  print_indent();
  os << "_tvm_layer_names[" << layer_idx << "] = \"" << layer_name << "\";\n";
  print_indent();
  os << "uint64_t _layer_start_" << layer_idx << " = TVMDSPCycleCounter_getCount64();\n";
}

void DSPCodeGenExtension::EmitLayerProfilingEnd(int layer_idx, std::ostream& os,
                                                const std::function<void()>& print_indent) {
  print_indent();
  os << "_tvm_layer_cycles[" << layer_idx << "] = TVMDSPCycleCounter_elapsed("
     << "_layer_start_" << layer_idx << ", TVMDSPCycleCounter_getCount64());\n";
  print_indent();
  os << "_tvm_layer_count = " << (layer_idx + 1) << ";\n";
}

void DSPCodeGenExtension::EmitDebugAllocInit(std::ostream& os,
                                             const std::function<void()>& print_indent) {
  print_indent();
  os << "// Reset debug-alloc counters\n";
  print_indent();
  os << "_tvm_alloc_storage_count = 0;\n\n";
}

void DSPCodeGenExtension::EmitDebugAllocStoragePre(int alloc_idx, const std::string& size_expr,
                                                   std::ostream& os,
                                                   const std::function<void()>& print_indent) {
  print_indent();
  os << "_tvm_debug_alloc_storage_pre(" << alloc_idx << ", (int64_t)(" << size_expr << "));\n";
}

void DSPCodeGenExtension::EmitDebugAllocStoragePost(int alloc_idx, const std::string& result_expr,
                                                    std::ostream& os,
                                                    const std::function<void()>& print_indent) {
  print_indent();
  os << "_tvm_debug_alloc_storage_post(" << alloc_idx << ", (void*)(" << result_expr << "));\n";
  print_indent();
  os << "_tvm_alloc_storage_count = " << (alloc_idx + 1) << ";\n";
}

}  // namespace codegen
}  // namespace tvm
