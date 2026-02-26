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
 * \file codegen_c_static_ffi_pattern.cc
 * \brief Implementation of FFI call pattern analysis for C static code generation.
 */
#include "codegen_c_static_ffi_pattern.h"

#include <tvm/ffi/c_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt.h>

namespace tvm {
namespace codegen {

using namespace tir;

// Helper: Check if function name is a VM builtin
static bool IsVMBuiltin(const std::string& func_name) {
  static constexpr const char* kVMBuiltinPrefix = "vm.builtin.";
  return func_name.find(kVMBuiltinPrefix) == 0;
}

bool StructSetInfo::TargetsSameElement(const StructSetInfo& other) const {
  if (!IsValid() || !other.IsValid()) return false;
  return tvm::StructuralEqual()(buffer, other.buffer) &&
         tvm::StructuralEqual()(index, other.index);
}

StructSetInfo FFIPatternAnalyzer::ExtractStructSetInfo(const Stmt& stmt) {
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

bool FFIPatternAnalyzer::IsArgSetupStatement(const Stmt& stmt, PackedArgInfo* info) {
  auto* eval = stmt.as<EvaluateNode>();
  if (!eval) return false;

  auto* call = eval->value.as<CallNode>();
  if (!call) return false;

  // Check for TIR op: tir.TVMBackendAnyListSetPackedArg
  static const Op& set_packed_arg_op = Op::Get("tir.TVMBackendAnyListSetPackedArg");
  if (call->op.same_as(set_packed_arg_op)) {
    if (call->args.size() >= 4) {
      // args[0] = list_handle (source array identifier, e.g., "r" or "c")
      // args[1] = list_index (source index)
      // args[2] = stack
      // args[3] = stack_offset
      if (auto* var = call->args[0].as<VarNode>()) {
        info->source_array = var->name_hint;
      } else {
        return false;
      }

      if (auto* idx = call->args[1].as<IntImmNode>()) {
        info->source_index = static_cast<int>(idx->value);
      } else {
        return false;
      }

      if (auto* stack_idx = call->args[3].as<IntImmNode>()) {
        info->stack_index = static_cast<int>(stack_idx->value);
      } else {
        return false;
      }

      info->type = PackedArgInfo::kArray;
      return true;
    }
  }

  // Check for external call (SetFFIAny*, etc.)
  bool is_extern_call = call->op.same_as(builtin::call_extern()) ||
                        call->op.same_as(builtin::call_pure_extern());
  if (!is_extern_call) return false;
  if (call->args.empty()) return false;

  auto* name_node = call->args[0].as<StringImmNode>();
  if (!name_node) return false;

  std::string func_name = name_node->value;

  // SetFFIAnyInt(&stack[idx], value)
  if (func_name == "SetFFIAnyInt" && call->args.size() >= 3) {
    if (auto* val = call->args[2].as<IntImmNode>()) {
      info->type = PackedArgInfo::kLiteral;
      info->literal_value = val->value;
      info->stack_index = -1;
      return true;
    }
  }

  // SetFFIAnyNone(&stack[idx])
  if (func_name == "SetFFIAnyNone") {
    info->type = PackedArgInfo::kNone;
    info->stack_index = -1;
    return true;
  }

  return false;
}

bool FFIPatternAnalyzer::IsMoveFromPackedReturn(const Stmt& stmt, std::string* dest_array,
                                                 int* dest_index) {
  auto* eval = stmt.as<EvaluateNode>();
  if (!eval) return false;

  auto* call = eval->value.as<CallNode>();
  if (!call) return false;

  // Check for TIR op: tir.TVMBackendAnyListMoveFromPackedReturn
  static const Op& move_from_return_op = Op::Get("tir.TVMBackendAnyListMoveFromPackedReturn");
  if (call->op.same_as(move_from_return_op)) {
    if (call->args.size() >= 2) {
      if (auto* var = call->args[0].as<VarNode>()) {
        *dest_array = var->name_hint;
      } else {
        return false;
      }

      if (auto* idx = call->args[1].as<IntImmNode>()) {
        *dest_index = static_cast<int>(idx->value);
        return true;
      }
    }
  }

  return false;
}

void FFIPatternAnalyzer::ExtractArgSourcesForPattern(const ffi::Array<Stmt>& seq, size_t call_idx,
                                                      FFICallPattern* pattern) {
  // Resize args vector to hold all arguments
  pattern->args.resize(pattern->num_args);

  // Initialize all args as unknown
  for (auto& arg : pattern->args) {
    arg.type = PackedArgInfo::kNone;
    arg.source_array = "";
    arg.source_index = -1;
    arg.stack_index = -1;
    arg.literal_value = 0;
  }

  // Scan backwards from call_idx to find argument setup statements
  for (size_t i = call_idx; i > 0; --i) {
    const Stmt& stmt = seq[i - 1];
    auto* eval = stmt.as<EvaluateNode>();
    if (!eval) break;

    auto* call = eval->value.as<CallNode>();
    if (!call) break;

    // Check for tir.TVMBackendAnyListSetPackedArg op
    static const Op& set_packed_arg_op = Op::Get("tir.TVMBackendAnyListSetPackedArg");
    if (call->op.same_as(set_packed_arg_op) && call->args.size() >= 4) {
      std::string source_array;
      if (auto* var = call->args[0].as<VarNode>()) {
        source_array = var->name_hint;
      } else {
        continue;
      }

      auto* src_idx = call->args[1].as<IntImmNode>();
      auto* stack_idx = call->args[3].as<IntImmNode>();

      if (src_idx && stack_idx) {
        int stack_offset = static_cast<int>(stack_idx->value);
        if (stack_offset >= 0 && stack_offset < static_cast<int>(pattern->args.size())) {
          pattern->args[stack_offset].type = PackedArgInfo::kArray;
          pattern->args[stack_offset].source_array = source_array;
          pattern->args[stack_offset].source_index = static_cast<int>(src_idx->value);
          pattern->args[stack_offset].stack_index = stack_offset;
        }
      }
      pattern->first_arg_index = i - 1;
      continue;
    }

    // Check for struct_set (SetFFIAny* lowered form)
    if (!call->op.same_as(builtin::tvm_struct_set())) {
      // Check for call_extern
      bool is_extern = call->op.same_as(builtin::call_extern()) ||
                       call->op.same_as(builtin::call_pure_extern());
      if (!is_extern || call->args.empty()) break;

      auto* name_node = call->args[0].as<StringImmNode>();
      if (!name_node) break;

      std::string func_name = name_node->value;
      bool is_setffi = func_name == "SetFFIAnyInt" || func_name == "SetFFIAnyNone" ||
                       func_name == "SetFFIAnyFloat" || func_name == "SetFFIAnyPtr";
      if (is_setffi) {
        pattern->first_arg_index = i - 1;
        continue;
      }
      break;
    }

    {
      auto info = ExtractStructSetInfo(stmt);
      if (!info.IsValid()) {
        // Check for call_extern
        bool is_extern = call->op.same_as(builtin::call_extern()) ||
                         call->op.same_as(builtin::call_pure_extern());
        if (!is_extern || call->args.empty()) break;

        auto* name_node = call->args[0].as<StringImmNode>();
        if (!name_node) break;

        std::string func_name = name_node->value;
        bool is_setffi = func_name == "SetFFIAnyInt" || func_name == "SetFFIAnyNone" ||
                         func_name == "SetFFIAnyFloat" || func_name == "SetFFIAnyPtr";
        if (is_setffi) {
          pattern->first_arg_index = i - 1;
          continue;
        }
        break;
      }

      // Handle union_value statements by looking back for matching type_index
      if (info.kind == builtin::kTVMFFIAnyUnionValue && i >= 2) {
        auto type_info = ExtractStructSetInfo(seq[i - 2]);
        bool has_matching_type = type_info.IsValid() &&
                                 type_info.kind == builtin::kTVMFFIAnyTypeIndex &&
                                 type_info.TargetsSameElement(info);
        if (has_matching_type) {
          // Extract the stack offset (argument index) and type from the pair
          auto* idx_imm = info.index.as<IntImmNode>();
          auto* type_imm = type_info.value.as<IntImmNode>();
          if (idx_imm && type_imm) {
            int stack_offset = static_cast<int>(idx_imm->value);
            int type_index = type_imm->value;
            bool valid_offset = stack_offset >= 0 &&
                                stack_offset < static_cast<int>(pattern->args.size());
            if (valid_offset) {
              if (type_index == kTVMFFIInt) {
                // Extract literal integer value
                if (auto* val_imm = info.value.as<IntImmNode>()) {
                  pattern->args[stack_offset].type = PackedArgInfo::kLiteral;
                  pattern->args[stack_offset].literal_value = val_imm->value;
                  pattern->args[stack_offset].stack_index = stack_offset;
                }
              } else if (type_index == kTVMFFINone) {
                // SetFFIAnyNone - result slot placeholder
                pattern->args[stack_offset].type = PackedArgInfo::kNone;
                pattern->args[stack_offset].stack_index = stack_offset;
              }
            }
          }
          pattern->first_arg_index = i - 2;
          continue;
        }
      }

      // Handle type_index statements (paired with following union_value)
      if (info.kind == builtin::kTVMFFIAnyTypeIndex) {
        pattern->first_arg_index = i - 1;
        continue;
      }
    }

    break;
  }
}

FFICallPattern FFIPatternAnalyzer::DetectFFICallPattern(const ffi::Array<Stmt>& seq, size_t call_idx) {
  FFICallPattern pattern;
  pattern.valid = false;

  if (call_idx >= seq.size()) return pattern;

  auto* eval = seq[call_idx].as<EvaluateNode>();
  if (!eval) return pattern;

  auto* call = eval->value.as<CallNode>();
  if (!call) return pattern;

  if (!call->op.same_as(builtin::tvm_call_packed_lowered())) return pattern;

  auto* func_name_node = call->args[0].as<StringImmNode>();
  if (!func_name_node) return pattern;

  std::string func_name = func_name_node->value;
  if (!IsVMBuiltin(func_name)) return pattern;

  // Skip already-optimized builtins
  if (func_name == "vm.builtin.null_value" ||
      func_name == "vm.builtin.check_tensor_info" ||
      func_name == "vm.builtin.match_shape") {
    return pattern;
  }

  // Get number of arguments
  int64_t begin = call->args[2].as<IntImmNode>()->value;
  int64_t end = call->args[3].as<IntImmNode>()->value;
  pattern.num_args = end - begin;

  pattern.builtin_name = func_name;
  pattern.call_index = call_idx;

  // Look backwards to find argument setup statements
  std::vector<PackedArgInfo> args(pattern.num_args);
  size_t first_arg_idx = call_idx;

  for (size_t i = call_idx; i > 0; --i) {
    PackedArgInfo arg_info;
    bool is_arg_setup = IsArgSetupStatement(seq[i - 1], &arg_info);
    if (is_arg_setup) {
      first_arg_idx = i - 1;
      if (arg_info.type == PackedArgInfo::kArray && arg_info.stack_index >= 0 &&
          arg_info.stack_index < static_cast<int>(pattern.num_args)) {
        args[arg_info.stack_index] = arg_info;
      }
    } else {
      break;
    }
  }

  pattern.first_arg_index = first_arg_idx;
  pattern.args = args;
  pattern.args.resize(pattern.num_args);

  // Look forward for MoveFromPackedReturn
  if (call_idx + 1 < seq.size()) {
    std::string dest_array;
    int dest_index;
    bool found_move = IsMoveFromPackedReturn(seq[call_idx + 1], &dest_array, &dest_index);
    if (found_move) {
      pattern.result_array = dest_array;
      pattern.result_slot = dest_index;
      pattern.result_index = call_idx + 1;
      pattern.valid = true;
    }
  }

  return pattern;
}

std::set<size_t> FFIPatternAnalyzer::PreScanFFIPatterns(const ffi::Array<Stmt>& seq) {
  std::set<size_t> skip_indices;

  for (size_t i = 0; i < seq.size(); ++i) {
    auto* eval = seq[i].as<EvaluateNode>();
    if (!eval) continue;

    auto* call = eval->value.as<CallNode>();
    if (!call || !call->op.same_as(builtin::tvm_call_packed_lowered())) continue;

    auto* func_name_node = call->args[0].as<StringImmNode>();
    if (!func_name_node) continue;

    std::string func_name = func_name_node->value;
    if (!IsVMBuiltin(func_name)) continue;

    // Skip already-optimized builtins
    if (func_name == "vm.builtin.null_value" ||
        func_name == "vm.builtin.check_tensor_info" ||
        func_name == "vm.builtin.match_shape") {
      continue;
    }

    FFICallPattern pattern = DetectFFICallPattern(seq, i);
    if (pattern.valid) {
      bool supported = (func_name == "vm.builtin.alloc_storage" ||
                        func_name == "vm.builtin.alloc_tensor" ||
                        func_name == "vm.builtin.reshape" ||
                        func_name == "vm.builtin.make_tuple" ||
                        func_name == "vm.builtin.copy");

      if (supported) {
        // Note: Currently disabled - detection needs improvement
        // for (size_t j = pattern.first_arg_index; j < pattern.call_index; ++j) {
        //   skip_indices.insert(j);
        // }
      }
    }
  }

  return skip_indices;
}

void FFIPatternAnalyzer::ScanVMBuiltinPatterns(const ffi::Array<Stmt>& seq,
                                                std::map<size_t, FFICallPattern>* patterns,
                                                std::set<size_t>* skip_indices) {
  ICHECK(patterns != nullptr && skip_indices != nullptr)
      << "ScanVMBuiltinPatterns: output parameters must not be null";

  for (size_t i = 0; i < seq.size(); ++i) {
    auto* eval = seq[i].as<EvaluateNode>();
    if (!eval) continue;

    auto* call = eval->value.as<CallNode>();
    if (!call || !call->op.same_as(builtin::tvm_call_packed_lowered())) continue;

    auto* func_name_node = call->args[0].as<StringImmNode>();
    if (!func_name_node) continue;

    std::string func_name = func_name_node->value;
    if (!IsVMBuiltin(func_name)) continue;

    // Skip builtins that have dedicated optimizations elsewhere
    if (func_name == "vm.builtin.null_value" ||
        func_name == "vm.builtin.check_tensor_info" ||
        func_name == "vm.builtin.match_shape") {
      continue;
    }

    FFICallPattern pattern = DetectFFICallPattern(seq, i);
    if (!pattern.valid) continue;

    ExtractArgSourcesForPattern(seq, i, &pattern);

    // Check if we have enough info for clean codegen
    bool can_emit_clean = true;
    for (size_t arg_idx = 0; arg_idx < pattern.args.size(); ++arg_idx) {
      if (pattern.args[arg_idx].type == PackedArgInfo::kArray &&
          pattern.args[arg_idx].source_array.empty()) {
        can_emit_clean = false;
        break;
      }
    }

    if (can_emit_clean) {
      (*patterns)[i] = pattern;
      for (size_t j = pattern.first_arg_index; j < i; ++j) {
        skip_indices->insert(j);
      }
    }
  }
}

}  // namespace codegen
}  // namespace tvm
