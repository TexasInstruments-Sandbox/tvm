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
 * \file codegen_c_static.h
 * \brief Generate C code to build a static binary for Relax VM execution.
 *
 * This is an adaptation derived from CodeGenC (the generic C backend) and
 * CodeGenCHost (the C backend for a host CPU).
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_CSTATIC_H_
#define TVM_TARGET_SOURCE_CODEGEN_CSTATIC_H_

#include <set>
#include <string>
#include <vector>
#include <algorithm>

#include "../source/codegen_c.h"
#include "codegen_c_static_ffi_pattern.h"
#include "tvm/target/codegen.h"
#include "tvm/tir/expr.h"
#include "tvm/ffi/c_api.h"  // For TVMFFITypeIndex enum constants

namespace tvm {
namespace codegen {

using namespace tir;

/*!
 * \brief Check if a function name is a VM builtin.
 *
 * VM builtins are functions whose names start with "vm.builtin." prefix.
 * These are handled specially by the code generator.
 *
 * \param func_name The function name to check.
 * \return true if the function name starts with "vm.builtin."
 */
inline bool IsVMBuiltin(const std::string& func_name) {
  static constexpr const char* kVMBuiltinPrefix = "vm.builtin.";
  return func_name.find(kVMBuiltinPrefix) == 0;
}

// Static C Backend - enables generation of static binaries for Relax VM execution
class CodeGenCStatic final : public CodeGenC {
 public:
  CodeGenCStatic();
  void Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
            const std::string& target_str, const std::unordered_set<std::string>& devices,
            bool profile_layers = false, bool skip_runtime_checks = false,
            bool use_cpp_api = false, bool debug_alloc = false);

  void AddFunction(const GlobalVar& gvar, const PrimFunc& f) override;
  void AddFunction(const GlobalVar& gvar, const PrimFunc& f, bool emit_fwd_func_decl);
  void InitFuncState(const PrimFunc& f) override;
  void PreFunctionBody(const PrimFunc& f) override;
  void DeclarePackedCalls(const PrimFunc& f);

  void PrintType(DataType t, std::ostream& os) final;  // NOLINT(*)

    /*!
   * Print Type representation of type type.
   * \param type The type representation.
   * \param os The stream to print the ctype into
   */
  void PrintType(const Type& type, std::ostream& os) override;  // NOLINT(*)

  void PrintFuncPrefix(std::ostream& os) final;        // NOLINT(*)
  void PrintTrailer();

  // expression visitors
  void VisitExpr_(const CallNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const BufferLoadNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const BroadcastNode* op, std::ostream& os) override;  // NOLINT(*)
  void VisitExpr_(const DivNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const MaxNode* op, std::ostream& os) override;       // NOLINT(*)
  void VisitExpr_(const MinNode* op, std::ostream& os) override;       // NOLINT(*)


  // statment vistors
  void VisitStmt_(const LetStmtNode* op) override;
  void VisitStmt_(const BufferStoreNode* op) override;
  void VisitStmt_(const AllocateNode* op) override;
  void VisitStmt_(const AssertStmtNode* op) override;
  void VisitStmt_(const SeqStmtNode* op) override;
  void VisitStmt_(const EvaluateNode* op) override;
  void VisitStmt_(const ForNode* op) override;

  Array<String> GetFunctionNames() const { return function_names_; }

  void DumpCGFunctionInfo() const;
  void EmitWrapperFunctions();

    /*!
   * \brief Print external function call.
   * \param ret_type The return type.
   * \param global_symbol The symbolc of the target function.
   * \param args The arguments to the function.
   * \param skip_first_arg Whether to skip the first arguments.
   * \param os The output stream.
   */
  void PrintCallExtern(Type ret_type, String global_symbol, const Array<PrimExpr>& args,
                               bool skip_first_arg, std::ostream& os) override; // NOLINT(*)

 private:
  /*!
   * \brief Error Handling Policy for CodeGenCStatic
   *
   * This code generator employs a multi-layered error handling strategy:
   *
   * 1. **Code Generator Internal Errors** (CodeGenCStatic methods):
   *    - Use LOG(FATAL) for unrecoverable internal errors (invariant violations)
   *    - Use LOG(WARNING)/DLOG(WARNING) for recoverable issues with fallback behavior
   *    - Use ICHECK for precondition validation
   *
   * 2. **Generated TIR Functions** (__vmtir__* functions):
   *    - Return int error codes following TVM convention:
   *      * 0: Success
   *      * -1: Generic error (runtime failure, null pointer, allocation failure)
   *    - No exceptions - pure C calling convention
   *
   * 3. **Generated Wrapper Functions** (cg_* functions):
   *    - Throw std::runtime_error for user-facing errors
   *    - Provide descriptive error messages with context
   *    - Convert TIR function return codes to exceptions
   *    - Validate arguments and throw std::invalid_argument when appropriate
   *
   * This layered approach ensures:
   * - Code generator failures are caught during compilation
   * - TIR functions can be called from C code without exception handling
   * - Wrapper functions provide idiomatic C++ error reporting
   */

  // Constants for code generation patterns
  static constexpr const char* kVMBuiltinPrefix = "vm.builtin.";
  static constexpr const char* kVMTIRPrefix = "__vmtir__";
  static constexpr const char* kWrapperPrefix = "cg_";
  static constexpr const char* kRegisterFileIdentifier = "r";
  static constexpr const char* kPackedFuncSuffix = "_packed";

  // Target string parsing constants
  static constexpr const char* kMcpuAttr = "mcpu=";
  static constexpr size_t kMcpuAttrLen = 5;  // strlen("mcpu=")
  static constexpr const char* kDeviceAttr = "device=";
  static constexpr size_t kDeviceAttrLen = 7;  // strlen("device=")
  
  // TIR calling convention constants
  static constexpr int kTIRArgsSize = 4;
  static constexpr int kTIRVMContextIndex = 0;
  static constexpr int kTIRRegFileIndex = 1;
  static constexpr int kTIRConstantsIndex = 2;
  static constexpr int kTIRReservedIndex = 3;
  
  // TIR function return codes (matches TVM convention)
  static constexpr int kTIRSuccess = 0;
  static constexpr int kTIRError = -1;
  
  // Register index tracking constants
  static constexpr size_t kMinArgsForRegisterTracking = 2;
  static constexpr size_t kRegisterFileArgIndex = 0;
  static constexpr size_t kRegisterIndexArgIndex = 1;

  /* \brief Internal structure to store information about function calls */
  struct FunctionInfo {
    /* \brief function name */
    std::string func_name;
    /* number of arguments required by the function */
    int64_t num_args;
    /* \brief name of resource_handle to pass */
    std::string resource_handle_name;
  };

  /* \brief Code generation information for a specific function */
  struct CGFunctionInfo {
    /* \brief Maximum register index used in this function */
    int64_t max_register_index = -1;
    /* \brief Number of function arguments/parameters */
    int64_t num_args = 0;
    /* \brief Whether function returns tuple (Array<NDArray>) vs single NDArray */
    bool returns_tuple = false;
    /* \brief Number of outputs (1 for single output, N for tuple with N elements) */
    int64_t num_outputs = 1;
    /* \brief Total number of function parameters */
    uint64_t total_params = 0;
    /* \brief Whether function was marked as private */
    bool was_private = false;
  };
  std::string module_name_;
  /* \brief mapping global packed func to the unique name */
  std::unordered_map<std::string, std::string> declared_globals_;
  /* \brief names of the functions declared in this module */
  Array<String> function_names_;
  /* \brief current function being processed */
  std::string current_function_name_;
  /* \brief mapping function names to codegen information */
  std::unordered_map<std::string, CGFunctionInfo> function_info_map_;
  /*! \brief whether to emit asserts in the resulting C code */
  bool emit_asserts_;
  /*! \brief whether to emit forwared function declarations in the resulting C code */
  bool emit_fwd_func_decl_;

  std::string stack_name_;
  size_t stack_size_{0};  // Size of FFI argument stack (from tvm_stack_alloca)

  /* \brief VM builtins used in the module (func_name -> packed_func_var_name) */
  std::map<std::string, std::string> vm_builtins_used_;

  /*! \brief TI DSP target configuration (C66x, C7x) and profiling state */
  struct DSPConfig {
    bool enabled = false;           // Whether targeting TI DSP (C66x or C7x)
    std::string mcpu;               // Target CPU (e.g., "c66", "c7x")
    std::string device_name;        // Device identifier for CCXML generation
    bool profile_layers = false;    // Enable per-layer cycle profiling
    bool debug_alloc = false;       // Enable diagnostic allocation tracing
    int layer_call_index = 0;       // Counter for profiled layer calls
    int alloc_storage_index = 0;    // Counter for traced AllocStorage calls
    std::vector<std::string> profiled_layer_names;  // Names of profiled layers
  };
  DSPConfig dsp_;

  /*! \brief Check if targeting C7x DSP (for intrinsic emission) */
  inline bool IsC7xTarget() const {
    return dsp_.enabled && dsp_.mcpu.find("c7") == 0;
  }

  // Skip runtime checks optimization (skip_runtime_checks target attribute)
  // When enabled, generates no-op stubs for check_tensor_info and match_shape
  bool skip_runtime_checks_{false};

  // C++ API mode (use-cpp-api target attribute)
  // When enabled, uses C++ API with AnyArray wrappers and direct VM calls instead of FFI
  bool use_cpp_api_{false};

  // Track whether AnyArray wrapper declarations have been emitted for current function
  // Reset in InitFuncState(), set when first clean VM builtin call is emitted
  bool anyarray_decls_emitted_{false};

  // Argument tracking for direct VM calls (legacy, used by pending_packed_args_)
  std::vector<PackedArgInfo> pending_packed_args_;
  std::string pending_result_array_;  // destination array for result
  int pending_result_index_{-1};      // destination index for result

  /*!
   * \brief Emit direct call for a VM builtin
   * \param pattern FFI call pattern to replace
   * \return true if direct call was emitted, false to fall back to FFI
   */
  bool EmitDirectVMBuiltinCall(const FFICallPattern& pattern);

  /*!
   * \brief Try to emit merged struct set statements for FFI type+value pairs
   * \param seq Statement sequence
   * \param index Current index in the sequence
   * \param[out] next_index Updated index if statements were merged
   * \return true if merged emission was performed
   *
   * Detects patterns like:
   *   tvm_struct_set(buf, idx, kTypeIndex, INT_VALUE)
   *   tvm_struct_set(buf, idx, kUnionValue, value)
   * And merges them into:
   *   SetFFIAnyInt(&buf[idx], value)
   */
  bool TryEmitMergedStructSet(const Array<Stmt>& seq, size_t index, size_t* next_index);

  /*!
   * \brief Emit direct VM builtin call using AnyArray API
   * \param pattern FFI call pattern with argument source info
   * \return true if direct call was emitted
   *
   * Generates cleaner code like:
   *   _r.SetNDArray(3, AllocTensor(_r.GetStorage(2), 0, _c.GetShape(5), _c.GetDType(6)));
   * Instead of the verbose stack_ffi_any-based approach.
   */
  bool EmitDirectVMBuiltinCallClean(const FFICallPattern& pattern);

  void PrintCallPacked(const CallNode* op);
  std::string GetPackedName(const CallNode* op);
  void PrintGetFuncFromBackend(const std::string& func_name, const std::string& packed_func_name);
  void EmitVMBuiltinInitFunction();
  void PrintStorageScope(const std::string& scope, std::ostream& os) override;  // NOLINT(*)
  void UpdateMaxRegisterIndex(const Array<PrimExpr>& args);

  /*!
   * \brief Print ternary conditional operator implementing binary `op`
   * Forces the operands to be in SSA form.
   * \param binary_op binary operator being expressed
   * \param compare_op string representation of comparison operator
   * \param output_stream stream reference to print into
   */
  template <typename BinaryOpNode>
  inline void PrintTernaryCondExpr(const BinaryOpNode* binary_op, const char* compare_op,
                                   std::ostream& output_stream);  // NOLINT(*)
  /*! \brief restrict keyword */
  std::string restrict_keyword_{"__restrict__"};
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_CSTATIC_H_
