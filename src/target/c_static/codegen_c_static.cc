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
 * \file codegen_c_static.cc
 * \brief C Static Code Generator for TVM Relax VM
 *
 * This code generator produces standalone C/C++ code for executing TVM Relax
 * models compiled to the VM representation. It is derived from CodeGenC
 * (the generic C backend) and CodeGenCHost (the C backend for a host CPU).
 *
 * Key features:
 * - Generates self-contained C/C++ code with minimal runtime dependencies
 * - Supports VM builtins (alloc_storage, alloc_tensor, etc.)
 * - Creates wrapper functions for easy integration with C++ applications
 * - Uses tvm::ffi::Any for polymorphic value handling
 * - Manages register files for intermediate computation results
 * - Handles both embedded and binary-serialized model parameters
 *
 * Architecture:
 * - TIR functions are generated as __vmtir__<name> with C calling convention
 * - Wrapper functions (cg_<name>) provide convenient C++ interfaces
 * - VM builtins are initialized once at first wrapper invocation
 * - Constants/parameters are loaded via TVMGetConstants()
 *
 * Usage flow:
 * 1. TVM compiles Relax model to TIR with VM operations
 * 2. This codegen translates TIR to C/C++ code
 * 3. Generated code is compiled into a shared library or executable
 * 4. User calls wrapper functions (e.g., cg_main, cg_forward) with inputs
 */
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/module.h>
#include <tvm/target/codegen.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/ir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/runtime/logging.h>
#include <tvm/arith/analyzer.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <map>

#include "../../arith/pattern_match.h"
#include "../../support/str_escape.h"
#include "../build_common.h"
#include "../source/codegen_c.h"
#include "../source/codegen_params.h"
#include "codegen_c_static.h"
#include "codegen_c_static_dsp.h"
#include "codegen_c_static_templates.h"
#include "codegen_c_static_wrapper.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Constructor for CodeGenCStatic
 *
 * Initializes the code generator with a unique module name for the FFI library context.
 */
CodeGenCStatic::CodeGenCStatic() {
  module_name_ = name_supply_->FreshName("__tvm_ffi_library_ctx");
}

/*!
 * \brief Initialize the code generator with configuration options
 * \param output_ssa Whether to output SSA form (static single assignment)
 * \param emit_asserts Whether to emit assertion checks in generated code
 * \param emit_fwd_func_decl Whether to emit forward function declarations
 * \param target_str Target string specifying compilation target
 * \param devices Set of device types that may be used
 *
 * Adapted from CodeGenC. Sets up the code generation context including:
 * - Assertion and SSA output configuration
 * - Declaration tracking for global variables
 * - Include headers and namespace setup
 * - Helper function declarations for FFI type conversions
 */
void CodeGenCStatic::Init(bool output_ssa, bool emit_asserts, bool emit_fwd_func_decl,
                      const std::string& target_str,
                      const std::unordered_set<std::string>& devices,
                      bool profile_layers, bool skip_runtime_checks, bool use_cpp_api,
                      bool debug_alloc) {
  emit_asserts_ = emit_asserts;
  emit_fwd_func_decl_ = emit_fwd_func_decl;
  skip_runtime_checks_ = skip_runtime_checks;
  use_cpp_api_ = use_cpp_api;
  declared_globals_.clear();

  // Extract TI DSP target configuration from target string
  dsp_.enabled = (target_str.find("mcpu=c66") != std::string::npos ||
                  target_str.find("mcpu=c7") != std::string::npos);

  // Extract mcpu value if present
  size_t mcpu_pos = target_str.find(kMcpuAttr);
  if (mcpu_pos != std::string::npos) {
    size_t start = mcpu_pos + kMcpuAttrLen;
    size_t end = target_str.find_first_of(" -", start);
    dsp_.mcpu = target_str.substr(start, (end != std::string::npos) ? end - start : std::string::npos);
  }

  // Extract device name if present
  size_t device_pos = target_str.find(kDeviceAttr);
  if (device_pos != std::string::npos) {
    size_t start = device_pos + kDeviceAttrLen;
    size_t end = target_str.find_first_of(" -", start);
    dsp_.device_name = target_str.substr(start, (end != std::string::npos) ? end - start : std::string::npos);
  }

  // Set profile-layers option (enables per-layer cycle counting)
  dsp_.profile_layers = profile_layers;
  // Set debug-alloc option (enables allocation tracing)
  dsp_.debug_alloc = debug_alloc;

  decl_stream << "// tvm target: " << target_str << "\n";

  // Emit target-specific headers and declarations
  if (dsp_.enabled) {
    // TI DSP target: Use DSPCodeGenExtension for headers
    DSPCodeGenExtension::EmitHeaders(use_cpp_api_, dsp_.profile_layers, dsp_.debug_alloc,
                                     decl_stream);
  } else {
    // Standard TVM target: Use full TVM runtime headers
    decl_stream << templates::kStandardTVMHeaders;

    // Only emit FFI helper functions when C++ API is disabled
    if (!use_cpp_api_) {
      decl_stream << templates::kStandardTVMFFIHelpers;
    }
  }
  CodeGenC::Init(output_ssa);
}

void CodeGenCStatic::PrintTrailer() {
}

void CodeGenCStatic::AddFunction(const GlobalVar& gvar, const PrimFunc& func) {
  return AddFunction(gvar, func, /*emit_fwd_func_decl=*/false);
}

void CodeGenCStatic::AddFunction(const GlobalVar& gvar, const PrimFunc& func,
                             bool emit_fwd_func_decl) {
  auto global_symbol = func->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  ICHECK(global_symbol.has_value())
      << "CodeGenCStatic: Expect PrimFunc to have the global_symbol attribute";
  function_names_.push_back(global_symbol.value());

  // Track current function being processed
  current_function_name_ = global_symbol.value();
  // Initialize CGFunctionInfo if not exists (try_emplace avoids double lookup)
  auto [it, inserted] = function_info_map_.try_emplace(current_function_name_);
  if (inserted) {
    // Use tir.num_input attribute if available, otherwise set to -1
    ffi::Optional<IntImm> num_input_attr = func->GetAttr<IntImm>("tir.num_input");
    it->second.num_args = num_input_attr.has_value() ? num_input_attr.value()->value : -1;
    // Use tir.returns_tuple attribute if available, otherwise default to false
    ffi::Optional<Bool> returns_tuple_attr = func->GetAttr<Bool>("tir.returns_tuple");
    it->second.returns_tuple =
        returns_tuple_attr.has_value() ? returns_tuple_attr.value()->value : false;

    // Read number of outputs for multi-output functions
    if (it->second.returns_tuple) {
      ffi::Optional<IntImm> num_outputs_attr = func->GetAttr<IntImm>("tir.num_outputs");
      if (num_outputs_attr.has_value()) {
        it->second.num_outputs = num_outputs_attr.value()->value;
        // Check DSP output limit (max 8 outputs supported by DSP runtime)
        constexpr int64_t kDSPMaxOutputs = 8;
        if (dsp_.enabled && it->second.num_outputs > kDSPMaxOutputs) {
          LOG(FATAL) << "Function '" << current_function_name_ << "' returns "
                     << it->second.num_outputs << " outputs, but DSP runtime (mcpu="
                     << dsp_.mcpu << ") supports maximum " << kDSPMaxOutputs
                     << " outputs.";
        }
      }
    }

    it->second.total_params = func->params.size();
    it->second.was_private = func->GetAttr<Bool>("was_private").has_value();
  }

  emit_fwd_func_decl_ = emit_fwd_func_decl;
  CodeGenC::AddFunction(gvar, func);

  if (func->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
    ICHECK(global_symbol.has_value())
        << "CodeGenCHost: The entry func must have the global_symbol attribute, "
        << "but function " << gvar << " only has attributes " << func->attrs;

    function_names_.push_back(ffi::symbol::tvm_ffi_main);
    stream << "// CodegenC: NOTE: Auto-generated entry function\n";
    PrintFuncPrefix(stream);
    PrintType(func->ret_type, stream);
    stream << " " << tvm::ffi::symbol::tvm_ffi_main
           << "(void* self_handle, void* args, int num_args, void* result) {\n";
    stream << "  return " << global_symbol.value()
           << "(self_handle, args, num_args, result);\n";
    stream << "}\n";
  }
}

/*!
 * \brief Pre-analysis visitor to collect buffer variable dtypes.
 *
 * This visitor walks the function body to find all BufferLoad and BufferStore
 * operations, recording the mapping from buffer data variables to their dtypes.
 * This information is used to generate typed restrict pointer declarations
 * for TI DSP targets.
 */
class BufferTypeCollector : public tir::StmtExprVisitor {
 public:
  void VisitExpr_(const tir::BufferLoadNode* op) final {
    RegisterBufferType(op->buffer);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const tir::BufferStoreNode* op) final {
    RegisterBufferType(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  const std::unordered_map<const tir::VarNode*, DataType>& GetBufferTypes() const {
    return buffer_types_;
  }

 private:
  void RegisterBufferType(const tir::Buffer& buffer) {
    const tir::VarNode* var = buffer->data.get();
    auto it = buffer_types_.find(var);
    if (it == buffer_types_.end()) {
      buffer_types_[var] = buffer->dtype;
    }
  }

  std::unordered_map<const tir::VarNode*, DataType> buffer_types_;
};

void CodeGenCStatic::InitFuncState(const PrimFunc& f) {
   CodeGenC::InitFuncState(f);
   this->stack_name_.clear();
   this->stack_size_ = 0;
   this->anyarray_decls_emitted_ = false;

   // For TI DSP targets, collect buffer dtypes from the function body
   // This enables typed restrict pointer declarations for better optimization
   if (dsp_.enabled) {
     BufferTypeCollector collector;
     collector(f->body);
     for (const auto& kv : collector.GetBufferTypes()) {
       // Register the buffer variable's dtype
       // This will be used when generating LetStmt for data pointer extraction
       handle_data_type_[kv.first] = kv.second;
     }
   }
}

// Override to add DSP-specific cycle counter initialization for profiling
void CodeGenCStatic::PreFunctionBody(const PrimFunc& f) {
  auto global_symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol);
  std::string func_name = static_cast<std::string>(global_symbol.value());

  // Emit cycle counter initialization for main function when profiling is enabled
  if (dsp_.profile_layers && dsp_.enabled && func_name.find("__vmtir__main") != std::string::npos) {
    DSPCodeGenExtension::EmitProfilingInit(this->stream, [this]() { this->PrintIndent(); });
  }

  // Emit debug-alloc counter reset for main function
  if (dsp_.debug_alloc && dsp_.enabled && func_name.find("__vmtir__main") != std::string::npos) {
    DSPCodeGenExtension::EmitDebugAllocInit(this->stream, [this]() { this->PrintIndent(); });
    dsp_.alloc_storage_index = 0;
  }

  // Only emit pre-body code for device kernel functions, not host functions
  // Note: Cannot use kTarget attribute to distinguish because it becomes defined
  // after SplitHostDevice pass. Using name pattern match instead.
  if (func_name.find("_kernel") == std::string::npos)
    return;
}


// verbatim from CodegenCHost
void CodeGenCStatic::PrintFuncPrefix(std::ostream& os) {  // NOLINT(*)
  // Generate extern "C" linkage for TIR functions
  // Note: We always generate C++ code, so no #ifdef __cplusplus guard needed
  os << "extern \"C\"\n";
}

// verbatim from CodegenCHost
void CodeGenCStatic::PrintType(DataType t, std::ostream& os) {  // NOLINT(*)
  int lanes = t.lanes();
  if (t.is_handle()) {
    ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
      case 8:
        os << "char";
        break;
      case 16:
        os << "short";
        break;
      case 32:
        os << "int";
        break;
      case 64:
        os << "long";
        break;
      case 1:
        os << "int";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  LOG(FATAL) << "Cannot convert type " << t << " to C type";
}

// Simplified broadcast handling - relies on C++ compiler vector extensions
void CodeGenCStatic::VisitExpr_(const BroadcastNode* op, std::ostream& os) {  // NOLINT(*)
  std::string v = PrintExpr(op->value);
  os << "((";
  PrintType(op->dtype, os);
  os << ")(";
  os << v;
  os << "))";
}

// C7x intrinsic: float division → __recip, with rsqrt pattern detection
void CodeGenCStatic::VisitExpr_(const DivNode* op, std::ostream& os) {  // NOLINT(*)
  if (IsC7xTarget() && op->dtype.is_float() && op->dtype.lanes() == 1) {
    // Detect 1.0f / sqrtf(x) pattern → __recip_sqrt(x)
    auto* float_imm = op->a.as<FloatImmNode>();
    if (float_imm && float_imm->value == 1.0) {
      auto* call = op->b.as<CallNode>();
      if (call && (call->op.same_as(builtin_call_pure_extern_) ||
                   call->op.same_as(builtin_call_extern_))) {
        auto* func_name = call->args[0].as<StringImmNode>();
        if (func_name && func_name->value == "sqrtf" && call->args.size() >= 2) {
          os << "__recip_sqrt((";
          PrintExpr(call->args[1], os);
          os << "))";
          return;
        }
      }
    }
    // General float division: (a) * __recip((b))
    os << "(";
    PrintExpr(op->a, os);
    os << ") * __recip((";
    PrintExpr(op->b, os);
    os << "))";
    return;
  }
  CodeGenC::VisitExpr_(op, os);
}

// Handle special floating-point constants (INFINITY, NAN) for C code generation
void CodeGenCStatic::VisitExpr_(const FloatImmNode* op, std::ostream& os) {  // NOLINT(*)
  switch (op->dtype.bits()) {
    case 64:
    case 32: {
      if (std::isinf(op->value)) {
        if (op->value > 0) {
          os << "INFINITY";
        } else {
          os << "(-INFINITY)";
        }
      } else if (std::isnan(op->value)) {
        os << "NAN";
      } else {
        std::ostringstream temp;
        temp << std::scientific << op->value;
        if (op->dtype.bits() == 32) temp << 'f';
        MarkConst(temp.str());
        os << temp.str();
      }
      break;
    }
    case 16: {
      if (std::isinf(op->value)) {
        os << '(';
        PrintType(op->dtype, os);
        os << ')';
        if (op->value > 0) {
          os << "INFINITY";
        } else {
          os << "(-INFINITY)";
        }
      } else if (std::isnan(op->value)) {
        os << '(';
        PrintType(op->dtype, os);
        os << ')' << "NAN";
      } else {
        os << '(';
        PrintType(op->dtype, os);
        os << ')' << std::scientific << op->value << 'f';
      }
      break;
    }
    default:
      LOG(FATAL) << "Bad bit-width for float: " << op->dtype << "\n";
  }
}

// C7x intrinsic: float max -> __max
void CodeGenCStatic::VisitExpr_(const MaxNode* op, std::ostream& os) {  // NOLINT(*)
  if (IsC7xTarget() && op->dtype.is_float() && op->dtype.lanes() == 1) {
    os << "__max((";
    PrintExpr(op->a, os);
    os << "), (";
    PrintExpr(op->b, os);
    os << "))";
    return;
  }
  // Use fmax for float types (C99 standard), fall back to base for integer
  if (op->dtype.is_float() && op->dtype.lanes() == 1) {
    os << "fmax(";
    PrintExpr(op->a, os);
    os << ", ";
    PrintExpr(op->b, os);
    os << ")";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

// C7x intrinsic: float min -> __min
void CodeGenCStatic::VisitExpr_(const MinNode* op, std::ostream& os) {  // NOLINT(*)
  if (IsC7xTarget() && op->dtype.is_float() && op->dtype.lanes() == 1) {
    os << "__min((";
    PrintExpr(op->a, os);
    os << "), (";
    PrintExpr(op->b, os);
    os << "))";
    return;
  }
  // Use fmin for float types (C99 standard), fall back to base for integer
  if (op->dtype.is_float() && op->dtype.lanes() == 1) {
    os << "fmin(";
    PrintExpr(op->a, os);
    os << ", ";
    PrintExpr(op->b, os);
    os << ")";
  } else {
    CodeGenC::VisitExpr_(op, os);
  }
}

/*!
 * \brief Generate code to retrieve a packed function from the global registry
 * \param func_name Name of the function to retrieve (e.g., "vm.builtin.alloc_storage")
 * \param packed_func_name Variable name for the cached packed function pointer
 *
 * This function handles two cases:
 * 1. VM builtins (vm.builtin.*): Tracked for batch initialization, no NULL check emitted
 *    since they are initialized once in InitVMBuiltins() at program startup.
 * 2. Other packed functions: Emits a NULL check and registry lookup pattern inline.
 *
 * The NULL check pattern for non-builtins:
 *   if (func_packed == NULL) {
 *     if (TVMBackendGetFuncFromGlobalRegistry("func_name", &func_packed) != 0) {
 *       return -1;
 *     }
 *   }
 */
void CodeGenCStatic::PrintGetFuncFromBackend(const std::string& func_name,
                                           const std::string& packed_func_name) {
  // Track VM builtins for initialization and skip NULL check since they're initialized at startup
  if (IsVMBuiltin(func_name)) {
    vm_builtins_used_[func_name] = packed_func_name;
    // Skip emitting NULL check - builtins are initialized in InitVMBuiltins()
    return;
  }

  // For non-VM builtins, emit the NULL check pattern
  this->PrintIndent();
  this->stream << "if (" << packed_func_name << " == NULL) {\n";
  {
    ScopeGuard packed_func_if_scope(this);
    this->PrintIndent();
    this->stream << "if (TVMBackendGetFuncFromGlobalRegistry(" << "\"" << func_name << "\""
                 << ", &" << packed_func_name << ") != 0) {\n";
    {
      ScopeGuard get_func_env_scope(this);
      this->PrintIndent();
      this->stream << "return -1;\n";
    }
    this->PrintIndent();
    this->stream << "}\n";
  }
  this->PrintIndent();
  this->stream << "}\n";
}

/*!
 * \brief Emit a static initialization function for VM builtin functions
 *
 * Generates an InitVMBuiltins() function that initializes all VM builtin packed
 * functions (vm.builtin.alloc_storage, vm.builtin.alloc_tensor, etc.) in one place.
 *
 * The function uses a static boolean flag to ensure initialization happens only once,
 * even if called multiple times from different wrapper functions. This is important
 * because multiple entry points (e.g., cg_main, cg_forward) may exist in the generated
 * code, and each needs VM builtins initialized but should share the same function
 * pointers.
 *
 * Generated code structure:
 *   static int InitVMBuiltins() {
 *     static bool initialized = false;
 *     if (initialized) return 0;  // Early exit if already initialized
 *     // Initialize each VM builtin
 *     if (TVMBackendGetFuncFromGlobalRegistry("vm.builtin.X", &vm_builtin_X_packed) != 0)
 *       return -1;
 *     ...
 *     initialized = true;
 *     return 0;
 *   }
 *
 * \note This function is called from EmitWrapperFunctions() before the main TIR
 *       function definitions.
 */
void CodeGenCStatic::EmitVMBuiltinInitFunction() {
  if (vm_builtins_used_.empty()) {
    return;
  }

  // InitVMBuiltins is static - called internally by wrapper functions
  this->stream << "static int InitVMBuiltins() {\n";
  this->stream << "  static bool initialized = false;\n";
  this->stream << "  if (initialized) {\n";
  this->stream << "    return 0;\n";
  this->stream << "  }\n";

  for (const auto& pair : vm_builtins_used_) {
    const std::string& func_name = pair.first;
    const std::string& packed_func_name = pair.second;

    // When skip_runtime_checks_ is enabled, skip registering validation functions
    // (their calls are already elided in PrintCallPacked)
    if (skip_runtime_checks_ &&
        (func_name == "vm.builtin.check_tensor_info" ||
         func_name == "vm.builtin.match_shape")) {
      continue;  // Skip - not used
    }

    // Look up from registry
    this->stream << "  if (TVMBackendGetFuncFromGlobalRegistry(\"" << func_name
                 << "\", &" << packed_func_name << ") != 0) {\n";
    this->stream << "    return -1;\n";
    this->stream << "  }\n";
  }
  this->stream << "  initialized = true;\n";
  this->stream << "  return 0;\n";
  this->stream << "}\n";
}

void CodeGenCStatic::PrintCallPacked(const CallNode* op) {
  const StringImmNode* func_name = op->args[0].as<StringImmNode>();
  ICHECK(func_name != nullptr)
      << "tvm_call_[c]packed_lowered expects first argument as function name";

  // When skip_runtime_checks_ is enabled, skip validation-only builtins
  // These validate tensor shape/type info at runtime. For static shapes,
  // TVM's type inference system already validates shape consistency at
  // compile time - the runtime checks are redundant safety nets.
  // Skipping them saves ~40-50k cycles per inference on C66x
  // Check this BEFORE calling GetPackedName to avoid declaring unused variables
  if (skip_runtime_checks_ &&
      (func_name->value == "vm.builtin.check_tensor_info" ||
       func_name->value == "vm.builtin.match_shape")) {
    this->PrintIndent();
    this->stream << "// [Compile-time validated] " << func_name->value
                 << " - shapes verified by TVM type inference\n";
    return;
  }

  // Optimize vm.builtin.null_value: skip the FFI call entirely.
  //
  // Background: null_value takes 0 args and returns None (used for killing VM
  // registers to enable reference counting). The TIR lowering pass
  // (lower_tvm_builtin.cc:MakeCallPackedGeneric) always initializes the return
  // slot to None before every packed call as a safety measure. Analysis of
  // generated code confirms that 100% of null_value FFI calls are preceded by
  // SetFFIAnyNone on the return slot.
  //
  // Before optimization:
  //   SetFFIAnyNone(&((stack_ffi_any)[0]));           // TIR-generated init
  //   if (TVMFFIFunctionCall(vm_builtin_null_value_packed, stack_ffi_any, 0,
  //                          &tvm_stack[0]) != 0) {
  //     return -1;
  //   }                                               // Redundant! Just returns None
  //   TVMBackendAnyListMoveFromPackedReturn(r, 5, stack_ffi_any, 0);
  //
  // After optimization:
  //   SetFFIAnyNone(&((stack_ffi_any)[0]));           // TIR-generated init (kept)
  //   // [Optimized] null_value skipped - return slot already None
  //   TVMBackendAnyListMoveFromPackedReturn(r, 5, stack_ffi_any, 0);
  //
  // This saves ~100 cycles of FFI dispatch overhead per call. With ~227 calls
  // in typical models (CLISTA-DoA), this saves ~23,000 cycles (~4% of total).
  if (func_name->value == "vm.builtin.null_value") {
    this->PrintIndent();
    this->stream << "// [Optimized] null_value skipped - return slot already None\n";
    return;
  }

  // Direct VM calls optimization annotation
  // When use_cpp_api_ is enabled, annotate calls that could benefit from
  // direct C++ API calls instead of FFI dispatch. The runtime API is available
  // in cpp/vm_array.h and cpp/vm_builtins.h.
  //
  // Full implementation requires intercepting TVMBackendAnyListSetPackedArg
  // statements to track argument sources, which requires buffering statements
  // or a TIR transformation pass.
  //
  // Target pattern transformation:
  //   Before (FFI): TVMFFIFunctionCall(vm_builtin_alloc_tensor_packed, ...)
  //   After (Direct): r.SetNDArray(idx, vm::AllocTensor(storage, offset, shape, dtype))
  if (use_cpp_api_ && std::string(func_name->value).find("vm.builtin.") == 0) {
    this->PrintIndent();
    this->stream << "// [Direct API available] " << func_name->value
                 << " - see cpp/vm_builtins.h\n";
  }

  int64_t begin = op->args[2].as<IntImmNode>()->value;
  int64_t end = op->args[3].as<IntImmNode>()->value;
  int64_t num_args = end - begin;
  ICHECK_GE(num_args, 0) << "packed call argument range is invalid (end < begin)";

  std::string packed_func_name;
  bool is_cpacked = op->op.same_as(builtin::tvm_call_cpacked_lowered());
  if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
    packed_func_name = GetPackedName(op);
    this->PrintGetFuncFromBackend(func_name->value, packed_func_name);
  } else {
    // directly use the original symbol with __tvm_ffi_ prefix (0.23.0 MakePackedAPI convention)
    ICHECK(is_cpacked)
        << "expected tvm_call_cpacked_lowered but got unknown packed call builtin";
    packed_func_name = std::string(ffi::symbol::tvm_ffi_symbol_prefix) + func_name->value;
  }

  std::string args_stack = PrintExpr(op->args[1]);

  // Layer profiling: emit cycle measurement for cpacked calls (actual compute kernels)
  // Skip vm.builtin calls as they are just memory management operations
  bool profile_this_call = dsp_.profile_layers && is_cpacked && dsp_.enabled;
  int layer_idx = -1;
  if (profile_this_call) {
    layer_idx = dsp_.layer_call_index++;
    dsp_.profiled_layer_names.push_back(packed_func_name);
    DSPCodeGenExtension::EmitLayerProfilingStart(
        layer_idx, packed_func_name, this->stream, [this]() { this->PrintIndent(); });
  }

  this->PrintIndent();
  if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
    this->stream << "if (TVMFFIFunctionCall(" << packed_func_name << ", ";
  } else {
    this->stream << "if (" << packed_func_name << "(NULL, ";
  }
  this->stream <<  args_stack << ", " << num_args << ", "
               << "&" << this->stack_name_ << "[" << num_args << "]" << ") != 0) {\n";
  {
    ScopeGuard func_call_scope(this);
    this->PrintIndent();
    this->stream << "return -1;\n";
  }
  this->PrintIndent();
  this->stream << "}\n";

  // Layer profiling: record elapsed cycles after the call
  if (profile_this_call) {
    DSPCodeGenExtension::EmitLayerProfilingEnd(
        layer_idx, this->stream, [this]() { this->PrintIndent(); });
  }
}


// Override to track register file usage for AnyList operations
void CodeGenCStatic::PrintCallExtern(Type ret_type, ffi::String global_symbol,
                                 const ffi::Array<PrimExpr>& args,
                                 bool skip_first_arg, std::ostream& os) {  // NOLINT(*)
  if (global_symbol == "TVMBackendAnyListSetPackedArg" ||
      global_symbol == "TVMBackendAnyListResetItem" ||
      global_symbol == "TVMBackendAnyListMoveFromPackedReturn") {
    UpdateMaxRegisterIndex(args);
  }
  CodeGenC::PrintCallExtern(ret_type, global_symbol, args, skip_first_arg, os);
}


/*!
 * \brief Get or create a unique variable name for a packed function
 * \param op Call node containing the function name as first argument
 * \return Unique variable name for the packed function pointer
 *
 * This function manages the global static variables that cache packed function pointers.
 * Each packed function (e.g., "vm.builtin.alloc_storage") gets a static void* variable
 * (e.g., "vm_builtin_alloc_storage_packed") that is initialized to NULL and populated
 * on first use.
 *
 * Key behaviors:
 * - Ensures each packed function has exactly one global variable (deduplication)
 * - Generates unique names to avoid collisions using name_supply_
 * - Declares the variable in decl_stream for the header section
 * - Tracks VM builtins in vm_builtins_used_ for batch initialization
 *
 * \note Adapted from CodeGenCHost
 */
std::string CodeGenCStatic::GetPackedName(const CallNode* op) {
  const StringImmNode* s = op->args[0].as<StringImmNode>();
  ICHECK(s != nullptr) << "tvm_call_packed_lowered expects first argument as function name";
  std::string func_name = s->value;
  std::string packed_func_name = func_name + kPackedFuncSuffix;
  std::string unique_name;

  // Check if we've already declared this packed function
  auto it = declared_globals_.find(packed_func_name);
  if (it != declared_globals_.end()) {
    unique_name = it->second;
  } else {
    // First time seeing this function - create a unique variable name
    unique_name = name_supply_->FreshName(packed_func_name);
    declared_globals_[packed_func_name] = unique_name;
    decl_stream << "static void* " << unique_name << " = NULL;\n";

    // Track VM builtins for initialization
    if (IsVMBuiltin(func_name)) {
      vm_builtins_used_[func_name] = unique_name;
    }
  }
  return unique_name;
}

// from CodeGenCHost
void CodeGenCStatic::VisitExpr_(const CallNode* op, std::ostream& os) {  // NOLINT(*)
  if (op->op.same_as(builtin::tvm_stack_alloca())) {
    this->stack_name_ = name_supply_->FreshName("tvm_stack");
    const std::string& type = op->args[0].as<StringImmNode>()->value;
    const IntImmNode* num = op->args[1].as<IntImmNode>();
    ICHECK(num != nullptr) << "tvm_stack_alloca size argument must be an integer immediate";
    static_assert(alignof(TVMFFIAny) % alignof(DLTensor) == 0, "invariant");
    size_t unit = sizeof(TVMFFIAny);
    size_t size = 0;
    if (type == "shape") {
      size = (num->value * sizeof(ffi::Shape::index_type) + unit - 1) / unit;
    } else if (type == "tvm_ffi_any") {
      size = (num->value * sizeof(TVMFFIAny) + unit - 1) / unit;
    } else if (type == "array") {
      size = (num->value * sizeof(DLTensor) + unit - 1) / unit;
    } else {
      LOG(FATAL) << "Unknown stack alloca type " << type;
    }
    this->stack_size_ = size;  // Track stack size for AnyArray wrapper
    this->PrintIndent();
    // Zero-initialize when using AnyArray wrappers to avoid DecRefOld on garbage
    if (use_cpp_api_) {
      this->stream << "TVMFFIAny " << this->stack_name_ << "[" << size << "] = {};\n";
    } else {
      this->stream << "TVMFFIAny " << this->stack_name_ << "[" << size << "];\n";
    }
    os << this->stack_name_;
  } else if (op->op.same_as(builtin::tvm_call_packed_lowered())) {
    this->PrintCallPacked(op);
  } else if (op->op.same_as(builtin::tvm_call_cpacked_lowered())) {
    this->PrintCallPacked(op);
  } else if (op->op.same_as(builtin::tvm_throw_last_error())) {
    this->PrintIndent();
    this->stream << "return -1;\n";
  } else if (op->op.same_as(builtin::ret())) {
    os << "return ";
    PrintExpr(op->args[0], os);
  } else {
    // C7x intrinsic replacements for scalar math functions
    if (IsC7xTarget() &&
        (op->op.same_as(builtin_call_pure_extern_) || op->op.same_as(builtin_call_extern_))) {
      auto* func_name = op->args[0].as<StringImmNode>();
      if (func_name) {
        // sqrtf(x) → ((x) != 0.0f ? (x) * __recip_sqrt((x)) : 0.0f)
        // Zero guard: __recip_sqrt(0) is undefined (inf), and 0 * inf = NaN
        if (func_name->value == "sqrtf" && op->args.size() >= 2) {
          std::ostringstream temp_x;
          VisitExpr(op->args[1], temp_x);
          std::string x = SSAGetID(temp_x.str(), op->args[1].dtype());
          os << "((" << x << ") != 0.0f ? (" << x << ") * __recip_sqrt((" << x << ")) : 0.0f)";
          return;
        }
        // fabsf(x) → __abs((x))
        if (func_name->value == "fabsf" && op->args.size() >= 2) {
          os << "__abs((";
          PrintExpr(op->args[1], os);
          os << "))";
          return;
        }
      }
    }
    CodeGenC::VisitExpr_(op, os);
  }
}

// Override for precise type handling in stack allocation and ObjectRef unwrapping
void CodeGenCStatic::VisitStmt_(const LetStmtNode* op) {
  // When skip_runtime_checks is enabled, skip emitting type_index variables
  // that are generated by MakePackedAPI for runtime type checking. These variables
  // are unused in c_static because UnwrapObjectRefArg handles type checking at runtime.
  // This eliminates "variable was declared but never referenced" compiler warnings.
  // Note: We only skip type_index variables as other variables (shape, dev_id) may be
  // used elsewhere in the generated code.
  if (skip_runtime_checks_) {
    std::string var_name = op->var->name_hint;
    // Check if this is a type_index variable from MakePackedAPI
    // Pattern: <param>.type_index or <param>_type_index
    if (var_name.find("type_index") != std::string::npos) {
      // Skip the LetStmt declaration, just emit the body
      PrintStmt(op->body);
      return;
    }
  }

  // Check for tvm_stack_alloca and emit with precise type
  if (auto* call = op->value.as<tir::CallNode>()) {
    if (call->op.same_as(builtin::tvm_stack_alloca()) && op->var.dtype().is_handle()) {
      const std::string& type = call->args[0].as<StringImmNode>()->value;
      std::string value = PrintExpr(op->value);
      PrintIndent();
      if (type == "tvm_ffi_any") {
        // Use precise type TVMFFIAny* instead of void*
        this->stream << "TVMFFIAny* " << AllocVarID(op->var.get()) << " = " << value << ";\n";
      } else {
        // For other types (shape, array), use void*
        this->stream << "void* " << AllocVarID(op->var.get()) << " = " << value << ";\n";
      }
      PrintStmt(op->body);
      return;
    }
  }

  // Check if this is an ObjectRef unwrapping pattern
  // Pattern: void* var = (var_type_index == kTVMFFITensor) ? (cast)(ptr + offset) : ptr
  if (auto* select = op->value.as<tir::SelectNode>()) {
    if (auto* eq = select->condition.as<tir::EQNode>()) {
      if (auto* int_imm = eq->b.as<IntImmNode>()) {
        if (int_imm->value == kTVMFFITensor && op->var.dtype().is_handle()) {
          // Look at false_value to find base expression
          PrimExpr base_expr = select->false_value;
          // Strip any casts
          while (auto* cast = base_expr.as<tir::CastNode>()) {
            base_expr = cast->value;
          }
          // Check if it's a tvm_struct_get call
          if (auto* call = base_expr.as<tir::CallNode>()) {
            if (call->op.same_as(tir::builtin::tvm_struct_get())) {
              // Generate: void* var = UnwrapObjectRefArg(((TVMFFIAny*)buffer)[index]);
              PrintIndent();
              PrintType(op->var.dtype(), this->stream);
              this->stream << ' ' << AllocVarID(op->var.get()) << " = UnwrapObjectRefArg(";
              this->stream << "((TVMFFIAny*)";
              PrintExpr(call->args[0], this->stream);
              this->stream << ")[";
              PrintExpr(call->args[1], this->stream);
              this->stream << "]);\n";
              PrintStmt(op->body);
              return;
            }
          }
        }
      }
    }
  }

  // For TI DSP targets, add restrict qualifier to typed pointer declarations
  // This helps the compiler with alias analysis for better optimization
  if (dsp_.enabled && op->var.dtype() == DataType::Handle() &&
      handle_data_type_.count(op->var.get())) {
    std::string value = PrintExpr(op->value);
    PrintIndent();
    PrintType(handle_data_type_.at(op->var.get()), this->stream);
    // Add __restrict__ qualifier for TI DSP targets
    this->stream << "* " << restrict_keyword_ << " " << AllocVarID(op->var.get()) << " = (";
    PrintType(handle_data_type_.at(op->var.get()), this->stream);
    this->stream << "*)" << value << ";\n";
    PrintStmt(op->body);
    return;
  }

  CodeGenC::VisitStmt_(op);
}

// Delegate to base class for standard allocation.
// Note: global.l2sram buffers from DMA tiling are merged by StorageRewrite
// into standard workspace allocations (TVMBackendAllocWorkspace).  The DSP
// runtime's allocator attempts L2 SRAM first for all workspace requests.
void CodeGenCStatic::VisitStmt_(const AllocateNode* op) {
  CodeGenC::VisitStmt_(op);
}

// Conditional assertion emission based on emit_asserts_ flag
void CodeGenCStatic::VisitStmt_(const AssertStmtNode* op) {  // NOLINT(*)
  if (emit_asserts_) {
    std::string cond = PrintExpr(op->condition);
    PrintIndent();
    stream << "if (!(" << cond << ")) {\n";
    {
      ScopeGuard assert_if_scope(this);
      PrintIndent();
      stream << "TVMAPISetLastError(\"" << op->message.as<StringImmNode>()->value << "\");\n";
      PrintIndent();
      stream << "return -1;\n";
    }
    PrintIndent();
    stream << "}\n";
  }
  this->PrintStmt(op->body);
}

// Helper: extract struct_set info from a statement for TryEmitMergedStructSet.
namespace {
struct StructSetInfo {
  const tir::CallNode* call = nullptr;
  PrimExpr buffer;
  PrimExpr index;
  int64_t kind = -1;
  PrimExpr value;
  bool IsValid() const { return call != nullptr; }
  bool TargetsSameElement(const StructSetInfo& other) const {
    if (!IsValid() || !other.IsValid()) return false;
    return tvm::StructuralEqual()(buffer, other.buffer) &&
           tvm::StructuralEqual()(index, other.index);
  }
};

StructSetInfo ExtractStructSetInfo(const Stmt& stmt) {
  StructSetInfo info;
  if (auto* eval = stmt.as<EvaluateNode>()) {
    if (auto* call = eval->value.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_struct_set()) && call->args.size() == 4) {
        if (auto* kind_imm = call->args[2].as<IntImmNode>()) {
          info.call = call;
          info.buffer = call->args[0];
          info.index = call->args[1];
          info.kind = kind_imm->value;
          info.value = call->args[3];
        }
      }
    }
  }
  return info;
}
}  // namespace


bool CodeGenCStatic::TryEmitMergedStructSet(const ffi::Array<Stmt>& seq, size_t index,
                                            size_t* next_index) {
  ICHECK(next_index != nullptr) << "TryEmitMergedStructSet: next_index output parameter is null";

  // Try to detect type_index + value pair
  auto info1 = ExtractStructSetInfo(seq[index]);
  if (!info1.IsValid() || info1.kind != builtin::kTVMFFIAnyTypeIndex) {
    return false;
  }

  if (index + 1 >= seq.size()) {
    return false;
  }

  auto info2 = ExtractStructSetInfo(seq[index + 1]);

  // Check if next statement sets the value for the same element
  if (!info2.IsValid() || info2.kind != builtin::kTVMFFIAnyUnionValue ||
      !info1.TargetsSameElement(info2)) {
    return false;
  }

  // We have a pair! Check the type and generate merged call
  auto* type_imm = info1.value.as<IntImmNode>();
  if (!type_imm) {
    return false;
  }

  int type_index = type_imm->value;
  std::string helper_func;

  // Determine which helper function to use based on type
  if (type_index == kTVMFFINone && is_zero(info2.value)) {
    // Special case: SetFFIAnyNone with C++ API AnyArray wrapper
    if (use_cpp_api_ && anyarray_decls_emitted_) {
      // Use direct AST inspection instead of PrintExpr to avoid SSA side effects
      std::string buffer_name;
      if (auto* var = info1.buffer.as<VarNode>()) {
        buffer_name = var->name_hint;
      }
      if (buffer_name == "stack_ffi_any" && info1.index.as<IntImmNode>()) {
        int idx = info1.index.as<IntImmNode>()->value;
        this->PrintIndent();
        this->stream << "_stack.SetNone(" << idx << ");\n";
        *next_index = index + 2;  // Skip both statements
        return true;
      }
    }
    // Fall back to SetFFIAnyNone helper
    helper_func = "SetFFIAnyNone";
    this->PrintIndent();
    this->stream << helper_func << "(&((";
    PrintExpr(info1.buffer, this->stream);
    this->stream << ")[";
    PrintExpr(info1.index, this->stream);
    this->stream << "]));\n";
    *next_index = index + 2;
    return true;
  } else if (type_index == kTVMFFIInt) {
    helper_func = "SetFFIAnyInt";
  } else if (type_index == kTVMFFIFloat) {
    helper_func = "SetFFIAnyFloat";
  } else if (type_index == kTVMFFIOpaquePtr) {
    helper_func = "SetFFIAnyPtr";
  } else {
    // Unknown type index, don't merge
    return false;
  }

  // Generate the helper function call
  this->PrintIndent();
  this->stream << helper_func << "(&((";
  PrintExpr(info1.buffer, this->stream);
  this->stream << ")[";
  PrintExpr(info1.index, this->stream);
  this->stream << "]), ";
  PrintExpr(info2.value, this->stream);
  this->stream << ");\n";
  *next_index = index + 2;  // Skip both statements
  return true;
}

bool CodeGenCStatic::EmitAnylistVMBuiltinCall(const CallNode* call) {
  // call->args layout:
  //   [0] = list_handle (Var "r" or "c")
  //   [1] = list_index  (IntImm — result slot)
  //   [2] = func_name   (StringImm, e.g. "vm.builtin.alloc_tensor")
  //   [3..N] = actual args (anylist_getitem or IntImm)
  FFICallPattern pattern;

  auto* name_node = call->args[2].as<StringImmNode>();
  if (!name_node) return false;
  pattern.builtin_name = name_node->value;

  auto* list_var = call->args[0].as<VarNode>();
  if (!list_var) return false;
  pattern.result_array = list_var->name_hint;

  auto* slot_imm = call->args[1].as<IntImmNode>();
  if (!slot_imm) return false;
  pattern.result_slot = static_cast<int>(slot_imm->value);

  size_t num_actual = call->args.size() - 3;
  pattern.num_args = static_cast<int64_t>(num_actual);

  for (size_t i = 3; i < call->args.size(); ++i) {
    PackedArgInfo info;
    if (auto* get = call->args[i].as<CallNode>();
        get && get->op.same_as(builtin::anylist_getitem())) {
      info.type = PackedArgInfo::kArray;
      if (auto* arr_var = get->args[0].as<VarNode>()) {
        info.source_array = arr_var->name_hint;
      }
      if (auto* idx = get->args[1].as<IntImmNode>()) {
        info.source_index = static_cast<int>(idx->value);
      }
    } else if (auto* imm = call->args[i].as<IntImmNode>()) {
      info.type = PackedArgInfo::kLiteral;
      info.literal_value = imm->value;
    }
    pattern.args.push_back(info);
  }

  pattern.valid = true;
  return EmitDirectVMBuiltinCallClean(pattern);
}

/*!
 * \brief Emit direct VM builtin call using clean AnyArray API
 *
 * Generates efficient code that directly accesses source arrays instead of
 * going through stack_ffi_any intermediate buffer.
 */
bool CodeGenCStatic::EmitDirectVMBuiltinCallClean(const FFICallPattern& pattern) {
  if (!pattern.valid) return false;

  // Update max_register_index if we're writing to register file
  // This is critical for correct reg_file sizing in wrapper functions
  if (pattern.result_array == kRegisterFileIdentifier && pattern.result_slot >= 0) {
    auto func_it = function_info_map_.find(current_function_name_);
    if (func_it != function_info_map_.end()) {
      func_it->second.max_register_index = std::max(
          func_it->second.max_register_index, static_cast<int64_t>(pattern.result_slot));
    }
  }

  // Emit AnyArray wrapper declarations on first use
  if (!anyarray_decls_emitted_) {
    this->PrintIndent();
    this->stream << "// AnyArray wrappers for clean VM builtin access\n";
    this->PrintIndent();
    this->stream << "using tvm::dsp::vm::AnyArray;\n";
    this->PrintIndent();
    this->stream << "namespace vm = tvm::dsp::vm;\n";
    this->PrintIndent();
    this->stream << "AnyArray _r(r);\n";
    this->PrintIndent();
    this->stream << "AnyArray _c(c);\n";
    // Add stack wrapper for TIR kernel call marshaling
    if (!this->stack_name_.empty() && this->stack_size_ > 0) {
      this->PrintIndent();
      this->stream << "AnyArray _stack(stack_ffi_any);\n";
    }
    this->stream << "\n";

    anyarray_decls_emitted_ = true;
  }

  const std::string& name = pattern.builtin_name;

  // Helper to generate source access expression from PackedArgInfo
  auto GetSourceAccess = [&](int arg_idx, const std::string& accessor) -> std::string {
    if (arg_idx < 0 || arg_idx >= static_cast<int>(pattern.args.size())) {
      return "/* invalid arg */";
    }
    const PackedArgInfo& arg = pattern.args[arg_idx];
    std::ostringstream os;
    if (arg.type == PackedArgInfo::kArray) {
      // Direct access: _r.GetXxx(idx) or _c.GetXxx(idx)
      os << "_" << arg.source_array << "." << accessor << "(" << arg.source_index << ")";
    } else if (arg.type == PackedArgInfo::kLiteral) {
      os << arg.literal_value;
    } else {
      os << "/* none */";
    }
    return os.str();
  };

  // vm.builtin.alloc_storage: (ctx, size_shape, device_type, device_id, dtype) -> storage
  // Args: [0]=ctx, [1]=size_shape, [2]=device_type, [3]=device_id, [4]=dtype
  if (name == "vm.builtin.alloc_storage") {
    if (pattern.num_args >= 4 && pattern.args.size() >= 5) {
      std::string size_expr = GetSourceAccess(1, "GetShape") + "->data[0]";
      std::string dtype_expr = GetSourceAccess(4, "GetDType");
      if (dsp_.debug_alloc) {
        int idx = dsp_.alloc_storage_index++;
        DSPCodeGenExtension::EmitDebugAllocStoragePre(
            idx, size_expr, this->stream, [this]() { this->PrintIndent(); });
      }
      this->PrintIndent();
      this->stream << "// [Direct] " << name << "\n";
      this->PrintIndent();
      this->stream << "_" << pattern.result_array << ".SetStorage(" << pattern.result_slot
                   << ", tvm::dsp::vm::AllocStorage("
                   << size_expr << ", " << dtype_expr << "));\n";
      if (dsp_.debug_alloc) {
        int idx = dsp_.alloc_storage_index - 1;
        std::string result_expr = "_" + pattern.result_array + ".GetStorage(" +
                                  std::to_string(pattern.result_slot) + ")";
        DSPCodeGenExtension::EmitDebugAllocStoragePost(
            idx, result_expr, this->stream, [this]() { this->PrintIndent(); });
      }
      return true;
    }
    return false;
  }

  // vm.builtin.alloc_tensor: (storage, offset, shape, dtype) -> ndarray
  // Args: [0]=storage, [1]=offset, [2]=shape, [3]=dtype
  if (name == "vm.builtin.alloc_tensor") {
    if (pattern.num_args >= 4 && pattern.args.size() >= 4) {
      this->PrintIndent();
      this->stream << "// [Direct] " << name << "\n";
      this->PrintIndent();
      this->stream << "_" << pattern.result_array << ".SetNDArray(" << pattern.result_slot
                   << ", tvm::dsp::vm::AllocTensor("
                   << GetSourceAccess(0, "GetStorage") << ", "
                   << GetSourceAccess(1, "GetInt") << ", "
                   << GetSourceAccess(2, "GetShape") << ", "
                   << GetSourceAccess(3, "GetDType") << "));\n";
      return true;
    }
    return false;
  }

  // vm.builtin.reshape: (tensor, shape) -> ndarray
  // Args: [0]=tensor, [1]=shape
  if (name == "vm.builtin.reshape") {
    if (pattern.num_args >= 2 && pattern.args.size() >= 2) {
      this->PrintIndent();
      this->stream << "// [Direct] " << name << "\n";
      this->PrintIndent();
      this->stream << "_" << pattern.result_array << ".SetNDArray(" << pattern.result_slot
                   << ", tvm::dsp::vm::Reshape("
                   << GetSourceAccess(0, "GetNDArray") << ", "
                   << GetSourceAccess(1, "GetShape") << "));\n";
      return true;
    }
    return false;
  }

  // vm.builtin.make_tuple: single element case (optimization)
  if (name == "vm.builtin.make_tuple" && pattern.num_args == 1) {
    if (pattern.args.size() >= 1) {
      this->PrintIndent();
      this->stream << "// [Direct] " << name << " (single element)\n";
      this->PrintIndent();
      this->stream << "{\n";
      {
        ScopeGuard scope(this);

        // Get source, IncRef, and set
        this->PrintIndent();
        this->stream << "TVMDSPNDArray* _val = " << GetSourceAccess(0, "GetNDArray") << ";\n";
        this->PrintIndent();
        this->stream << "if (_val) _val->ref_counter++;\n";
        this->PrintIndent();
        this->stream << "_" << pattern.result_array << ".SetNDArray("
                     << pattern.result_slot << ", _val);\n";
      }
      this->PrintIndent();
      this->stream << "}\n";
      return true;
    }
    return false;
  }

  // vm.builtin.make_tuple: multi-element case
  if (name == "vm.builtin.make_tuple" && pattern.num_args > 1) {
    if (pattern.args.size() >= static_cast<size_t>(pattern.num_args)) {
      this->PrintIndent();
      this->stream << "// [Direct] " << name << " (" << pattern.num_args << " elements)\n";
      this->PrintIndent();
      this->stream << "{\n";
      {
        ScopeGuard scope(this);

        // Create temporary array for arguments
        this->PrintIndent();
        this->stream << "TVMFFIAny _tuple_args[" << pattern.num_args << "];\n";

        // Copy each argument to temp array (with IncRef for objects)
        for (int i = 0; i < pattern.num_args; i++) {
          const PackedArgInfo& arg = pattern.args[i];
          if (arg.type == PackedArgInfo::kArray) {
            this->PrintIndent();
            this->stream << "_tuple_args[" << i << "] = _" << arg.source_array
                         << ".GetAny(" << arg.source_index << ");\n";
            // IncRef for object types
            this->PrintIndent();
            this->stream << "if (_tuple_args[" << i << "].type_index >= kTVMFFIStaticObjectBegin && "
                         << "_tuple_args[" << i << "].v_obj != nullptr) "
                         << "_tuple_args[" << i << "].v_obj->ref_counter++;\n";
          }
        }

        // Call MakeTupleArray and set result
        this->PrintIndent();
        this->stream << "_" << pattern.result_array << ".SetArray("
                     << pattern.result_slot << ", vm::MakeTupleArray(_tuple_args, "
                     << pattern.num_args << "));\n";
      }
      this->PrintIndent();
      this->stream << "}\n";
      return true;
    }
    return false;
  }

  // vm.builtin.copy: (value) -> value (with IncRef)
  // Uses generic SetFrom to handle any object type (NDArray, Array/tuple, etc.)
  if (name == "vm.builtin.copy") {
    if (pattern.num_args >= 1 && pattern.args.size() >= 1) {
      const PackedArgInfo& arg = pattern.args[0];
      if (arg.type == PackedArgInfo::kArray) {
        this->PrintIndent();
        this->stream << "// [Direct] " << name << " (generic)\n";
        this->PrintIndent();
        this->stream << "_" << pattern.result_array << ".SetFrom(_" << arg.source_array
                     << ", " << pattern.result_slot << ", " << arg.source_index << ");\n";
        return true;
      }
    }
    return false;
  }

  // vm.builtin.null_value: () -> none
  // Sets the destination register to kTVMFFINone
  if (name == "vm.builtin.null_value") {
    this->PrintIndent();
    this->stream << "// [Direct] " << name << "\n";
    this->PrintIndent();
    this->stream << "_" << pattern.result_array << ".SetNone(" << pattern.result_slot << ");\n";
    return true;
  }

  // vm.builtin.check_tensor_info: runtime assertion, skip when checks are disabled
  if (name == "vm.builtin.check_tensor_info") {
    this->PrintIndent();
    this->stream << "// [Skipped] " << name << "\n";
    return true;
  }

  // vm.builtin.match_shape: runtime assertion, skip when checks are disabled
  if (name == "vm.builtin.match_shape") {
    this->PrintIndent();
    this->stream << "// [Skipped] " << name << "\n";
    return true;
  }

  return false;
}

// Override SeqStmt to handle compact anylist intrinsics and merge struct_set pairs
void CodeGenCStatic::VisitStmt_(const SeqStmtNode* op) {
  for (size_t i = 0; i < op->seq.size(); ++i) {
    // Preserved anylist intrinsics (compact form, use-cpp-api path).
    // When LowerTVMBuiltin skips anylist expansion, the codegen receives
    // anylist_setitem_call_{c,}packed directly and converts them to C++ API
    // calls without the round-trip through struct_set + call_packed_lowered.
    if (use_cpp_api_ && dsp_.enabled) {
      auto* eval = op->seq[i].as<EvaluateNode>();
      if (eval) {
        auto* call = eval->value.as<CallNode>();
        if (call && (call->op.same_as(builtin::anylist_setitem_call_packed()) ||
                     call->op.same_as(builtin::anylist_setitem_call_cpacked()))) {
          if (EmitAnylistVMBuiltinCall(call)) continue;
        }
      }
    }

    // Try to merge struct set type_index + value pairs
    size_t next_index;
    if (TryEmitMergedStructSet(op->seq, i, &next_index)) {
      i = next_index - 1;  // Loop will increment i
      continue;
    }

    // Default: print statement normally
    this->PrintStmt(op->seq[i]);
  }
}

// Override for C++ API mode AnyArray wrappers and struct_set type handling
void CodeGenCStatic::VisitStmt_(const EvaluateNode* op) {
  if (is_const_int(op->value)) return;

  const CallNode* call = op->value.as<CallNode>();

  // Preserved anylist intrinsics (compact form, use-cpp-api path)
  if (use_cpp_api_ && dsp_.enabled && call &&
      (call->op.same_as(builtin::anylist_setitem_call_packed()) ||
       call->op.same_as(builtin::anylist_setitem_call_cpacked()))) {
    if (EmitAnylistVMBuiltinCall(call)) return;
  }

  // Clean TIR kernel call setup when C++ API is enabled
  // Emit AnyArray declarations on first use (after r, c variables are defined)
  if (use_cpp_api_ && call) {
    // TVMBackendAnyListSetPackedArg(src_array, src_idx, stack, stack_idx)
    // -> _stack.SetFrom(_src_array, stack_idx, src_idx)
    static const Op& set_packed_arg_op = Op::Get("tir.TVMBackendAnyListSetPackedArg");
    if (call->op.same_as(set_packed_arg_op) && call->args.size() >= 4) {
      // Use direct AST inspection instead of PrintExpr to avoid SSA side effects
      std::string src_array;
      if (auto* var = call->args[0].as<VarNode>()) {
        src_array = var->name_hint;
      } else {
        // Fall back to base class for non-variable source
        CodeGenC::VisitStmt_(op);
        return;
      }

      auto* src_idx = call->args[1].as<IntImmNode>();
      auto* stack_idx = call->args[3].as<IntImmNode>();

      if (src_idx && stack_idx) {
        // Emit AnyArray declarations on first use
        if (!anyarray_decls_emitted_) {
          this->PrintIndent();
          this->stream << "// AnyArray wrappers for clean VM builtin access\n";
          this->PrintIndent();
          this->stream << "using tvm::dsp::vm::AnyArray;\n";
          this->PrintIndent();
          this->stream << "namespace vm = tvm::dsp::vm;\n";
          this->PrintIndent();
          this->stream << "AnyArray _r(r);\n";
          this->PrintIndent();
          this->stream << "AnyArray _c(c);\n";
          if (!this->stack_name_.empty() && this->stack_size_ > 0) {
            this->PrintIndent();
            this->stream << "AnyArray _stack(" << this->stack_name_ << ");\n";
          }
          this->stream << "\n";
          anyarray_decls_emitted_ = true;
        }

        // Map source array name to wrapper name
        std::string src_wrapper = (src_array == kRegisterFileIdentifier) ? "_r" : "_c";
        this->PrintIndent();
        // Use SetFromUnchecked for stack - slots are zero-initialized or cleared by MoveFrom
        this->stream << "_stack.SetFromUnchecked(" << src_wrapper << ", "
                     << stack_idx->value << ", " << src_idx->value << ");\n";
        return;
      }
    }

    // TVMBackendAnyListMoveFromPackedReturn(dst_array, dst_idx, stack, stack_idx)
    // -> _dst_array.MoveFrom(_stack, dst_idx, stack_idx)
    static const Op& move_from_return_op = Op::Get("tir.TVMBackendAnyListMoveFromPackedReturn");
    if (call->op.same_as(move_from_return_op) && call->args.size() >= 4 && anyarray_decls_emitted_) {
      // Use direct AST inspection instead of PrintExpr to avoid SSA side effects
      std::string dst_array;
      if (auto* var = call->args[0].as<VarNode>()) {
        dst_array = var->name_hint;
      } else {
        // Fall back to base class for non-variable destination
        CodeGenC::VisitStmt_(op);
        return;
      }

      auto* dst_idx = call->args[1].as<IntImmNode>();
      auto* stack_idx = call->args[3].as<IntImmNode>();

      if (dst_idx && stack_idx) {
        // Update max_register_index if writing to register file
        if (dst_array == kRegisterFileIdentifier) {
          auto func_it = function_info_map_.find(current_function_name_);
          if (func_it != function_info_map_.end()) {
            func_it->second.max_register_index = std::max(
                func_it->second.max_register_index, dst_idx->value);
          }
        }
        // Map destination array name to wrapper name
        std::string dst_wrapper = (dst_array == kRegisterFileIdentifier) ? "_r" : "_c";
        this->PrintIndent();
        this->stream << dst_wrapper << ".MoveFrom(_stack, "
                     << dst_idx->value << ", " << stack_idx->value << ");\n";
        return;
      }
    }
  }

  if (call && call->op.same_as(builtin::tvm_struct_set())) {
    ICHECK_EQ(call->args.size(), 4)
        << "tvm_struct_set expects 4 arguments: (handle, index, kind, value)";
    int kind = call->args[2].as<IntImmNode>()->value;

    // Special handling for type_index assignments - use enum constants instead of integers
    if (kind == builtin::kTVMFFIAnyTypeIndex) {
      if (auto* int_imm = call->args[3].as<IntImmNode>()) {
        // Map integer type index to enum constant name
        std::string enum_name;
        switch (int_imm->value) {
          case kTVMFFINone: enum_name = "kTVMFFINone"; break;
          case kTVMFFIInt: enum_name = "kTVMFFIInt"; break;
          case kTVMFFIBool: enum_name = "kTVMFFIBool"; break;
          case kTVMFFIFloat: enum_name = "kTVMFFIFloat"; break;
          case kTVMFFIOpaquePtr: enum_name = "kTVMFFIOpaquePtr"; break;
          case kTVMFFITensor: enum_name = "kTVMFFITensor"; break;
          case kTVMFFIModule: enum_name = "kTVMFFIModule"; break;
          default:
            // For unknown type indices, fall through to base class handler
            CodeGenC::VisitStmt_(op);
            return;
        }

        // Generate: ref = enum_name;
        DataType store_dtype = call->args[3].dtype();
        std::string ref = GetStructRef(store_dtype, call->args[0], call->args[1], kind);
        this->PrintIndent();
        this->stream << ref << " = " << enum_name << ";\n";
        return;
      }
      // If not an IntImmNode (e.g., a variable), fall through to base class handler
    }
  }

  // Fall back to base class for all other cases
  CodeGenC::VisitStmt_(op);
}

void CodeGenCStatic::VisitStmt_(const ForNode* op) {
  // Emit TI-specific pragmas before the loop
  if (dsp_.enabled) {
    DSPCodeGenExtension::EmitLoopPragmas(op->extent, op->kind, stream,
                                         [this]() { PrintIndent(); });
  }

  // Call base class implementation
  CodeGenC::VisitStmt_(op);
}

// No-op: CodeGenCStatic handles multiple storage scopes without restrictions
void CodeGenCStatic::PrintStorageScope(const std::string& scope, std::ostream& os) {  // NOLINT(*)
}


template <typename BinaryOpNode>
inline void CodeGenCStatic::PrintTernaryCondExpr(const BinaryOpNode* binary_op,
                                                 const char* compare_op,
                                                 std::ostream& output_stream) {  // NOLINT(*)
  std::ostringstream temp_a;
  VisitExpr(binary_op->a, temp_a);
  std::string a_id = SSAGetID(temp_a.str(), binary_op->a.dtype());
  std::ostringstream temp_b;
  VisitExpr(binary_op->b, temp_b);
  std::string b_id = SSAGetID(temp_b.str(), binary_op->b.dtype());

  output_stream << "((" << a_id << ") " << compare_op << " (" << b_id << ") "
                << "? (" << a_id << ") : (" << b_id << "))";
}

// Override for flat memory buffer loads with type casting support
void CodeGenCStatic::VisitExpr_(const BufferLoadNode* op, std::ostream& os) {  // NOLINT(*)
  ICHECK_EQ(op->indices.size(), 1) << "Load from non-flat memory not supported.";

  DataType value_dtype = op->dtype;
  PrimExpr index = op->indices[0];
  Var buffer_var = op->buffer->data;
  DataType element_dtype = op->buffer->dtype;

  int lanes = op->dtype.lanes();
  if (value_dtype.lanes() == element_dtype.lanes()) {
    std::string ref = GetBufferRef(op->dtype, op->buffer.get(), index);
    HandleVolatileLoads(ref, op, os);
  } else {
    // Check for ramp pattern (base, base+1, base+2, ...) to enable vector loads
    // Simplified from original: Trust that ramp patterns are properly aligned
    // instead of performing modular arithmetic analysis
    arith::PVar<PrimExpr> base;
    if (arith::ramp(base, 1, op->dtype.lanes()).Match(index)) {
      std::string ref = GetVecLoad(op->dtype, op->buffer.get(), base.Eval());
      HandleVolatileLoads(ref, op, os);
    } else {
      std::ostringstream svalue_expr;
      std::string sindex = SSAGetID(PrintExpr(index), index.dtype());
      std::string vid = GetVarID(buffer_var.get());
      DataType elem_type = op->dtype.element_of();
      for (int i = 0; i < lanes; ++i) {
        std::ostringstream value_temp;
        if (!HandleTypeMatch(buffer_var.get(), elem_type)) {
          value_temp << "((";
          if (buffer_var.get()->dtype.is_handle()) {
            auto it = alloc_storage_scope_.find(buffer_var.get());
            if (it != alloc_storage_scope_.end()) {
              PrintStorageScope(it->second, value_temp);
            }
          }
          PrintType(elem_type, value_temp);
          value_temp << "*)" << vid << ')';
        } else {
          value_temp << vid;
        }
        value_temp << '[';
        PrintVecElemLoad(sindex, index.dtype(), i, value_temp);
        value_temp << ']';
        PrintVecElemLoadExpr(op->dtype, i, value_temp.str(), svalue_expr);
      }
      os << svalue_expr.str();
    }
  }
}

// Delegate to base class for buffer store operations
void CodeGenCStatic::VisitStmt_(const BufferStoreNode* op) {
  CodeGenC::VisitStmt_(op);
}

void CodeGenCStatic::PrintType(const Type& type, std::ostream& os) {  // NOLINT(*)
  if (auto* ptr = type.as<PrimTypeNode>()) {
    return PrintType(ptr->dtype, os);
  } else if (auto* ptr = type.as<PointerTypeNode>()) {
    PrintType(ptr->element_type, os);
    os << '*';
  } else if (IsVoidType(type)) {
    os << "void";
  } else {
    LOG(FATAL) << "Type " << type << " does not have a corresponding C Type";
  }
}

/*!
 * \brief Debug utility to dump code generation function information
 *
 * Outputs detailed information about all functions being processed by the code generator.
 * This is useful for debugging wrapper generation, register allocation, and understanding
 * the function metadata collected during code generation.
 *
 * For each function, outputs:
 * - Function name (e.g., "__vmtir__main")
 * - num_args: Number of input arguments
 * - returns_tuple: Whether the function returns a tuple (Array<NDArray>) or single value
 * - total_params: Total parameter count including context pointers
 * - was_private: Whether the function is private (internal implementation detail)
 * - max_register_index: Highest register index accessed (for register file sizing)
 *
 * \note This is a development/debugging tool and output goes to stdout
 */
void CodeGenCStatic::DumpCGFunctionInfo() const {
  std::cout << "\n=== CGFunctionInfo Summary ===" << std::endl;

  for (const auto& [func_name, func_info] : function_info_map_) {
    std::cout << "Function: " << func_name
              << " | num_args: " << func_info.num_args
              << " | returns_tuple: " << (func_info.returns_tuple ? "true" : "false")
              << " | total_params: " << func_info.total_params
              << " | was_private: " << (func_info.was_private ? "true" : "false")
              << " | max_register_index: " << func_info.max_register_index;

    // Query VMFuncInfo for non-private functions
    if (!func_info.was_private) {
      // Remove __vmtir__ prefix if it exists
      std::string vm_func_name = func_name;
      const size_t prefix_len = std::strlen(kVMTIRPrefix);
      if (vm_func_name.substr(0, prefix_len) == kVMTIRPrefix) {
        vm_func_name = vm_func_name.substr(prefix_len);
      }
    }

    std::cout << std::endl;
  }
  std::cout << "==============================\n" << std::endl;
}

/*!
 * \brief Track maximum register index used by the current function
 * \param args Arguments to a TVMBackendAnyList* function call
 *
 * This function analyzes calls to TVMBackendAnyListSetPackedArg and similar functions
 * to determine the highest register index accessed by the current function. This is
 * critical for allocating the register file with the correct size in wrapper functions.
 *
 * The register file is a vector of tvm::ffi::Any that holds:
 * - Input arguments (at indices 0, 1, 2, ...)
 * - Intermediate computation results
 * - Output values (at indices following inputs)
 *
 * Argument structure for TVMBackendAnyList* functions:
 * - args[0]: Register file identifier ('r' for register file)
 * - args[1]: Register index (integer) - the slot being accessed
 * - args[2]: TVMValue* stack_value
 * - args[3]: int* stack_tcode
 * - args[4+]: Additional parameters...
 *
 * The maximum register index + 1 determines the register file allocation size:
 *   std::vector<tvm::ffi::Any> reg_file(max_register_index + 1);
 *
 * \note Only tracks accesses to the 'r' register file (not other identifiers)
 */
void CodeGenCStatic::UpdateMaxRegisterIndex(const ffi::Array<PrimExpr>& args) {
  // Validate minimum argument count
  if (args.size() < kMinArgsForRegisterTracking) {
    DLOG(WARNING) << "UpdateMaxRegisterIndex: insufficient arguments (got "
                  << args.size() << ", need at least " << kMinArgsForRegisterTracking << ")";
    return;
  }

  // Validate register index is an integer immediate
  const IntImmNode* register_index = args[kRegisterIndexArgIndex].as<IntImmNode>();
  if (register_index == nullptr) {
    DLOG(WARNING) << "UpdateMaxRegisterIndex: register index at position "
                  << kRegisterIndexArgIndex << " is not an integer immediate";
    return;
  }

  // Validate register index is non-negative
  if (register_index->value < 0) {
    LOG(WARNING) << "UpdateMaxRegisterIndex: negative register index "
                 << register_index->value << " encountered";
    return;
  }

  // Extract register file identifier
  std::ostringstream first_arg_stream;
  VisitExpr(args[kRegisterFileArgIndex], first_arg_stream);

  // Track register usage for functions if first argument is 'r' (return register file)
  if (first_arg_stream.str() == kRegisterFileIdentifier) {
    auto it = function_info_map_.find(current_function_name_);
    if (it != function_info_map_.end()) {
      it->second.max_register_index = std::max(it->second.max_register_index, register_index->value);
    } else {
      DLOG(WARNING) << "UpdateMaxRegisterIndex: current function '"
                    << current_function_name_ << "' not found in function_info_map_";
    }
  }
}

/*!
 * \brief Generate C++ wrapper functions for all public TIR functions
 *
 * This function creates convenient C++ wrapper functions that provide easy-to-use
 * interfaces to the low-level TIR functions. The wrappers handle:
 * - VM initialization and memory allocator setup
 * - VM builtin function initialization (via InitVMBuiltins)
 * - Register file allocation for intermediate values
 * - Argument marshalling to the TIR calling convention
 * - Return value extraction from the register file
 * - Error handling for TIR function failures
 *
 * For each public TIR function "__vmtir__<name>", two wrapper functions are generated:
 * 1. Primary interface: cg_<name>(NDArray& arg0, NDArray& arg1, ...)
 *    - Takes explicit NDArray parameters
 *    - Returns NDArray or Array<NDArray> for tuple results
 *    - Uses std::move for efficient return value transfer
 *
 * 2. Convenience interface: cg_<name>(const Array<NDArray>& args)
 *    - Takes a dynamic array of arguments
 *    - Validates argument count at runtime
 *    - Delegates to the primary interface
 *
 * The wrapper structure:
 *   NDArray cg_forward(NDArray& arg0) {
 *     // Setup constants and register file
 *     // Initialize VM and allocators
 *     // Initialize VM builtins (if needed)
 *     // Marshal arguments
 *     // Call __vmtir__forward and check return code
 *     // Extract result from register file
 *     return std::move(result);
 *   }
 *
 * \note Private functions (those not exported from the module) are skipped.
 */
void CodeGenCStatic::EmitWrapperFunctions() {
  // Emit VM builtin initialization function first
  EmitVMBuiltinInitFunction();

  // Convert CGFunctionInfo to WrapperGenerator::FunctionInfo
  std::unordered_map<std::string, WrapperGenerator::FunctionInfo> wrapper_funcs;
  for (const auto& [name, info] : function_info_map_) {
    WrapperGenerator::FunctionInfo wf;
    wf.max_register_index = info.max_register_index;
    wf.num_args = info.num_args;
    wf.returns_tuple = info.returns_tuple;
    wf.was_private = info.was_private;
    wrapper_funcs[name] = wf;
  }

  // For TI DSP targets, generate C-compatible wrapper functions
  // These use static arrays instead of std::vector and return error codes instead of exceptions
  if (dsp_.enabled) {
    WrapperGenerator::EmitDSPWrappers(wrapper_funcs, vm_builtins_used_, stream);
    return;
  }

  // Generate standard C++ wrapper functions
  WrapperGenerator::EmitStandardWrappers(wrapper_funcs, vm_builtins_used_, stream);
}

// Modeled after BuildCHost() in codegen_c_host.cc
ffi::Module BuildCStatic(IRModule mod, Target target) {
  bool output_ssa = false;
  bool emit_asserts = false;
  bool emit_fwd_func_decl = true;

  std::unordered_set<std::string> devices;
  if (mod->GetAttr<ffi::Map<GlobalVar, ffi::String>>("device_contexts") != nullptr) {
    ffi::Map<GlobalVar, ffi::String> device_contexts =
        mod->GetAttr<ffi::Map<GlobalVar, ffi::String>>("device_contexts").value();
    for (auto const& context : device_contexts) {
      devices.insert(context.second.data());
    }
  }

  CodeGenCStatic cg;
  bool profile_layers = target->GetAttr<Integer>("profile-layers").value_or(0)->value != 0;
  bool skip_runtime_checks = target->GetAttr<Integer>("skip-runtime-checks").value_or(0)->value != 0;
  bool use_cpp_api = target->GetAttr<Integer>("use-cpp-api").value_or(0)->value != 0;
  bool debug_alloc = target->GetAttr<Integer>("debug-alloc").value_or(0)->value != 0;
  cg.Init(output_ssa, emit_asserts, emit_fwd_func_decl, target->str(), devices, profile_layers,
          skip_runtime_checks, use_cpp_api, debug_alloc);
  cg.SetConstantsByteAlignment(target->GetAttr<Integer>("constants-byte-alignment").value_or(16));

  auto is_aot_executor_fn = [](const PrimFunc& func) -> bool {
    return func->GetAttr<Bool>("runner_function", Bool(false)).value();
  };

  std::vector<std::pair<GlobalVar, PrimFunc>> funcs;
  for (auto [gvar, base_func] : mod->functions) {
    ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodegenCHost: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    funcs.push_back({gvar, prim_func});
  }

  // Sort functions
  auto sort_key = [&is_aot_executor_fn](const auto& func_pair) {
    return std::tuple{is_aot_executor_fn(func_pair.second), func_pair.first->name_hint};
  };
  std::sort(funcs.begin(), funcs.end(), [&sort_key](const auto& a, const auto& b) {
    return sort_key(a) < sort_key(b);
  });

  // Declare all functions first.  This ensures that all functions,
  // including the __tvm_main__ used in AOT, have access to forward
  // declarations of other functions in the IRModule.
  for (const auto& [gvar, prim_func] : funcs) {
    cg.DeclareFunction(gvar, prim_func);
  }

  // Codegen all functions.  Passing emit_fwd_func_decl=true adds a
  // forward declaration for any `builtin::call_extern`, based on the
  // arguments provided to it.
  for (const auto& [gvar, prim_func] : funcs) {
    cg.AddFunction(gvar, prim_func, emit_fwd_func_decl);
  }

  // Print collected function information
  //cg.DumpCGFunctionInfo();

  // Emit wrapper functions for non-private functions
  cg.EmitWrapperFunctions();

  cg.PrintTrailer();

  // Note: System library mode is not supported for CStatic backend
  // CStatic generates C++ wrapper code that requires C++ runtime

  std::string code = cg.Finish();
  return CSourceModuleCreate(code, "c", cg.GetFunctionNames());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.c_static", BuildCStatic);
}
}  // namespace codegen
}  // namespace tvm
