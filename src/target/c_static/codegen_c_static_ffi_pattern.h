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
 * \file codegen_c_static_ffi_pattern.h
 * \brief FFI call pattern detection and analysis for C static code generation.
 *
 * This module provides pure analysis functions for detecting FFI call patterns
 * in TIR statement sequences. The analysis is side-effect free and can be used
 * independently from code emission.
 *
 * FFI call patterns consist of:
 * 1. SetPackedArg/SetFFIAny* statements (argument setup)
 * 2. call_packed_lowered statement (FFI dispatch)
 * 3. MoveFromPackedReturn statement (result extraction)
 *
 * Design: Stateless analyzer class with static methods. All functions work
 * directly with TIR AST nodes without modifying code generation state.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_CSTATIC_FFI_PATTERN_H_
#define TVM_TARGET_SOURCE_CODEGEN_CSTATIC_FFI_PATTERN_H_

#include <tvm/ir/op.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace tvm {
namespace codegen {

/*!
 * \brief Information about a single argument in an FFI call
 *
 * Tracks where an argument comes from (register array, constant array, or literal)
 * and its position in both the source and FFI stack.
 */
struct PackedArgInfo {
  std::string source_array;  // "r" (register) or "c" (constant)
  int source_index = 0;      // index in source array
  int stack_index = 0;       // index in FFI stack
  enum ArgType { kArray, kLiteral, kNone } type = kNone;
  int64_t literal_value = 0;  // for literal values
};

/*!
 * \brief Information about an FFI call pattern in a statement sequence
 *
 * An FFI call pattern consists of:
 * 1. SetPackedArg/SetFFIAny* statements (argument setup)
 * 2. call_packed_lowered statement (FFI dispatch)
 * 3. MoveFromPackedReturn statement (result extraction)
 */
struct FFICallPattern {
  bool valid = false;                     // Whether pattern was detected
  std::string builtin_name;               // VM builtin name (e.g., "vm.builtin.alloc_tensor")
  size_t call_index = 0;                  // Index of call_packed_lowered in sequence
  size_t result_index = 0;                // Index of MoveFromPackedReturn in sequence
  size_t first_arg_index = 0;             // Index of first argument setup statement
  int64_t num_args = 0;                   // Number of arguments to the builtin
  std::string result_array;               // Destination array for result ("r" or "c")
  int result_slot = -1;                   // Destination index for result
  std::vector<PackedArgInfo> args;        // Argument information
};

/*!
 * \brief Information extracted from a tvm_struct_set call
 *
 * Used for detecting SetFFIAny* patterns that are lowered to struct_set pairs.
 */
struct StructSetInfo {
  const tir::CallNode* call = nullptr;
  PrimExpr buffer;
  PrimExpr index;
  int64_t kind = -1;
  PrimExpr value;

  bool IsValid() const { return call != nullptr; }

  bool TargetsSameElement(const StructSetInfo& other) const;
};

/*!
 * \brief Pure analysis functions for FFI call pattern detection
 *
 * All methods are static and side-effect free. They work directly with
 * TIR AST nodes without modifying any code generation state.
 */
class FFIPatternAnalyzer {
 public:
  /*!
   * \brief Check if statement is an argument setup for FFI call
   *
   * Detects:
   * - TVMBackendAnyListSetPackedArg(array, idx, stack, stack_idx)
   * - SetFFIAnyInt(&stack[idx], value)
   * - SetFFIAnyNone(&stack[idx])
   *
   * \param stmt Statement to check
   * \param[out] info Extracted argument info if applicable
   * \return true if statement is an argument setup call
   */
  static bool IsArgSetupStatement(const tir::Stmt& stmt, PackedArgInfo* info);

  /*!
   * \brief Check if statement is MoveFromPackedReturn
   *
   * \param stmt Statement to check
   * \param[out] dest_array Destination array identifier ("r" or "c")
   * \param[out] dest_index Destination index
   * \return true if statement is MoveFromPackedReturn
   */
  static bool IsMoveFromPackedReturn(const tir::Stmt& stmt, std::string* dest_array,
                                     int* dest_index);

  /*!
   * \brief Detect FFI call pattern starting at given index
   *
   * \param seq Statement sequence
   * \param call_idx Index of potential call_packed_lowered statement
   * \return FFI call pattern info (valid=false if not a pattern)
   */
  static FFICallPattern DetectFFICallPattern(const Array<tir::Stmt>& seq, size_t call_idx);

  /*!
   * \brief Scan statement sequence for VM builtin call patterns
   *
   * \param seq Statement sequence to scan
   * \param[out] patterns Map from statement index to detected FFI call pattern
   * \param[out] skip_indices Set of statement indices to skip (arg setup)
   *
   * Identifies VM builtin calls that can be emitted using direct calls
   * and marks their argument setup statements for skipping.
   */
  static void ScanVMBuiltinPatterns(const Array<tir::Stmt>& seq,
                                    std::map<size_t, FFICallPattern>* patterns,
                                    std::set<size_t>* skip_indices);

  /*!
   * \brief Extract argument sources from statements preceding a VM builtin call
   *
   * \param seq Statement sequence
   * \param call_idx Index of the call_packed_lowered statement
   * \param[out] pattern Pattern to populate with argument sources
   *
   * Analyzes SetPackedArg and SetFFIAny* statements before the call to determine
   * where each argument comes from (register array, constant array, or literal).
   */
  static void ExtractArgSourcesForPattern(const Array<tir::Stmt>& seq, size_t call_idx,
                                          FFICallPattern* pattern);

  /*!
   * \brief Pre-scan sequence for FFI patterns and return indices to skip
   *
   * \param seq Statement sequence
   * \return Set of statement indices that should be skipped (arg setup statements)
   *
   * This enables cleaner code generation by identifying which SetPackedArg/SetFFIAny*
   * statements are part of VM builtin calls that will be replaced with direct calls.
   */
  static std::set<size_t> PreScanFFIPatterns(const Array<tir::Stmt>& seq);

  /*!
   * \brief Extract StructSetInfo from an Evaluate statement
   *
   * \param stmt Statement to analyze
   * \return Extracted struct set info (IsValid() false if not a struct_set)
   */
  static StructSetInfo ExtractStructSetInfo(const tir::Stmt& stmt);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_CSTATIC_FFI_PATTERN_H_
