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
 * \file codegen_c_static_wrapper.h
 * \brief Wrapper function generator for C static code generation.
 *
 * This module generates C/C++ wrapper functions that provide convenient
 * interfaces to TIR-compiled functions. Two types of wrappers are supported:
 *
 * 1. Standard wrappers (C++ with TVM runtime):
 *    - Use std::vector for register files
 *    - Throw exceptions on error
 *    - Integrate with TVM VirtualMachine
 *
 * 2. DSP wrappers (C-compatible for embedded):
 *    - Use static arrays (no heap allocation)
 *    - Return error codes instead of exceptions
 *    - Compatible with TI C66x/C7x DSP targets
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_CSTATIC_WRAPPER_H_
#define TVM_TARGET_SOURCE_CODEGEN_CSTATIC_WRAPPER_H_

#include <cstdint>
#include <map>
#include <ostream>
#include <string>
#include <unordered_map>

namespace tvm {
namespace codegen {

/*!
 * \brief Generates wrapper functions for TIR-compiled functions.
 *
 * Wrapper functions provide a clean C/C++ API for calling __vmtir__* functions,
 * handling argument marshalling, VM initialization, and error conversion.
 */
class WrapperGenerator {
 public:
  // TIR calling convention constants
  static constexpr int kTIRArgsSize = 4;
  static constexpr int kTIRVMContextIndex = 0;
  static constexpr int kTIRRegFileIndex = 1;
  static constexpr int kTIRConstantsIndex = 2;
  static constexpr int kTIRReservedIndex = 3;

  // Naming convention constants
  static constexpr const char* kVMTIRPrefix = "__vmtir__";
  static constexpr const char* kWrapperPrefix = "cg_";

  /*!
   * \brief Code generation information for a TIR function.
   *
   * This struct captures the metadata needed to generate wrapper functions.
   */
  struct FunctionInfo {
    /*! \brief Maximum register index used in this function */
    int64_t max_register_index = -1;
    /*! \brief Number of function arguments/parameters */
    int64_t num_args = 0;
    /*! \brief Whether function returns tuple (Array<NDArray>) vs single NDArray */
    bool returns_tuple = false;
    /*! \brief Whether function was marked as private (skip wrapper generation) */
    bool was_private = false;
  };

  /*!
   * \brief Emit standard C++ wrapper functions.
   *
   * Generates wrapper functions that use TVM runtime (std::vector, exceptions).
   * For each public TIR function, generates:
   * 1. Primary interface with explicit NDArray parameters
   * 2. Convenience overload taking Array<NDArray>
   *
   * \param functions Map of TIR function name to info
   * \param vm_builtins VM builtins used (for initialization check)
   * \param os Output stream for generated code
   */
  static void EmitStandardWrappers(
      const std::unordered_map<std::string, FunctionInfo>& functions,
      const std::map<std::string, std::string>& vm_builtins,
      std::ostream& os);

  /*!
   * \brief Emit DSP-specific wrapper functions.
   *
   * Generates C-compatible wrapper functions suitable for embedded DSP targets.
   * Features:
   * - Static register file allocation (no heap)
   * - Error codes instead of exceptions
   * - extern "C" linkage for C compatibility
   *
   * \param functions Map of TIR function name to info
   * \param vm_builtins VM builtins used (for initialization code)
   * \param os Output stream for generated code
   */
  static void EmitDSPWrappers(
      const std::unordered_map<std::string, FunctionInfo>& functions,
      const std::map<std::string, std::string>& vm_builtins,
      bool tidl_runtime,
      std::ostream& os);

 private:
  /*!
   * \brief Extract wrapper function name from TIR function name.
   *
   * Transforms "__vmtir__main" to "cg_main" (or "cg_main_dsp" for DSP).
   *
   * \param tir_func_name TIR function name (e.g., "__vmtir__main")
   * \param dsp_suffix Whether to add "_dsp" suffix
   * \return Wrapper function name (e.g., "cg_main")
   */
  static std::string GetWrapperName(const std::string& tir_func_name, bool dsp_suffix = false);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_CSTATIC_WRAPPER_H_
