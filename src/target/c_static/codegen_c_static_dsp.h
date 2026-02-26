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
 * \file codegen_c_static_dsp.h
 * \brief TI DSP-specific code generation extensions.
 *
 * This module provides helper functions for generating code specific to
 * TI C66x/C7x DSP targets. It handles:
 * - TI compiler pragmas for loop optimization (MUST_ITERATE, UNROLL)
 * - DSP-specific header emission
 * - Layer-level cycle profiling infrastructure
 *
 * Design: Stateless helper class with static methods. State (layer counts,
 * profiled layer names) remains in CodeGenCStatic's DSPConfig struct.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_CSTATIC_DSP_H_
#define TVM_TARGET_SOURCE_CODEGEN_CSTATIC_DSP_H_

#include <tvm/ir/expr.h>
#include <tvm/tir/stmt.h>

#include <cstdint>
#include <functional>
#include <ostream>
#include <string>

namespace tvm {
namespace codegen {

/*!
 * \brief TI DSP-specific code generation extensions.
 *
 * Provides static helper methods for generating DSP-optimized code.
 * All methods are stateless - any state tracking is done by the caller.
 */
class DSPCodeGenExtension {
 public:
  /*!
   * \brief Emit TI compiler loop optimization pragmas.
   *
   * Generates #pragma directives for the TI compiler to enable
   * software pipelining and loop unrolling optimizations:
   * - MUST_ITERATE: Provides trip count hints
   * - UNROLL: Requests loop unrolling for small loops
   *
   * Pragmas are wrapped in __TI_COMPILER_VERSION__ guards for portability.
   *
   * \param extent Loop extent expression (for trip count)
   * \param loop_kind TIR loop kind (kSerial, kUnrolled, etc.)
   * \param os Output stream
   * \param print_indent Function to emit current indentation
   */
  static void EmitLoopPragmas(const PrimExpr& extent, tir::ForKind loop_kind,
                              std::ostream& os,
                              const std::function<void()>& print_indent);

  /*!
   * \brief Emit DSP-specific headers and declarations.
   *
   * Emits the appropriate headers based on configuration:
   * - DSP runtime headers (tvm_dsp_runtime.h, etc.)
   * - Helper functions (SetFFIAny*, etc.)
   * - FFI declarations (when C++ API disabled)
   * - Profiling infrastructure (when profile_layers enabled)
   *
   * \param use_cpp_api Whether C++ API mode is enabled
   * \param profile_layers Whether layer profiling is enabled
   * \param os Output stream for declarations
   */
  static void EmitHeaders(bool use_cpp_api, bool profile_layers, bool debug_alloc,
                          std::ostream& os);

  /*!
   * \brief Emit cycle counter initialization for profiling.
   *
   * Called at the start of the main function when profiling is enabled.
   * Initializes the cycle counter and layer count.
   *
   * \param os Output stream
   * \param print_indent Function to emit current indentation
   */
  static void EmitProfilingInit(std::ostream& os,
                                const std::function<void()>& print_indent);

  /*!
   * \brief Emit layer profiling start code.
   *
   * Called before a packed function call to record the start cycle count.
   *
   * \param layer_idx Index of this layer (0-based)
   * \param layer_name Name of the layer/function being profiled
   * \param os Output stream
   * \param print_indent Function to emit current indentation
   */
  static void EmitLayerProfilingStart(int layer_idx, const std::string& layer_name,
                                      std::ostream& os,
                                      const std::function<void()>& print_indent);

  /*!
   * \brief Emit layer profiling end code.
   *
   * Called after a packed function call to record elapsed cycles.
   *
   * \param layer_idx Index of this layer (0-based)
   * \param os Output stream
   * \param print_indent Function to emit current indentation
   */
  static void EmitLayerProfilingEnd(int layer_idx, std::ostream& os,
                                    const std::function<void()>& print_indent);

  /*!
   * \brief Emit debug-alloc counter reset at start of main function.
   *
   * \param os Output stream
   * \param print_indent Function to emit current indentation
   */
  static void EmitDebugAllocInit(std::ostream& os,
                                 const std::function<void()>& print_indent);

  /*!
   * \brief Emit pre-allocation debug tracing for AllocStorage.
   *
   * \param alloc_idx Allocation index (0-based)
   * \param size_expr C expression string for the allocation size
   * \param os Output stream
   * \param print_indent Function to emit current indentation
   */
  static void EmitDebugAllocStoragePre(int alloc_idx, const std::string& size_expr,
                                       std::ostream& os,
                                       const std::function<void()>& print_indent);

  /*!
   * \brief Emit post-allocation debug tracing for AllocStorage.
   *
   * \param alloc_idx Allocation index (0-based)
   * \param result_expr C expression string for the result pointer
   * \param os Output stream
   * \param print_indent Function to emit current indentation
   */
  static void EmitDebugAllocStoragePost(int alloc_idx, const std::string& result_expr,
                                        std::ostream& os,
                                        const std::function<void()>& print_indent);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_CSTATIC_DSP_H_
