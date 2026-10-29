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
 * \file hoist_reduction_init.cc
 * \brief Hoist reduction initialization outside of reduction loops.
 *
 * This pass transforms patterns like:
 *
 *   for (spatial_var) {
 *     for (reduce_var1) {
 *       for (reduce_var2) {
 *         if ((reduce_var1 == 0) && (reduce_var2 == 0)) {
 *           out[spatial_var] = 0.0f;
 *         }
 *         out[spatial_var] += ...;
 *       }
 *     }
 *   }
 *
 * Into:
 *
 *   for (spatial_var) {
 *     out[spatial_var] = 0.0f;
 *     for (reduce_var1) {
 *       for (reduce_var2) {
 *         out[spatial_var] += ...;
 *       }
 *     }
 *   }
 *
 * This transformation enables software pipelining on DSPs like TI C66x,
 * where conditional statements inside loops prevent optimal scheduling.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

/*!
 * \brief Information about a detected reduction init pattern.
 */
struct ReductionInitInfo {
  /*! \brief The initialization statement to hoist */
  Stmt init_stmt;
  /*! \brief Variables that appear in the condition (loop vars == 0) */
  std::unordered_set<const VarNode*> condition_vars;
  /*! \brief Whether this is a valid pattern */
  bool valid{false};
};

/*!
 * \brief Detects reduction init conditions of the form (var1 == 0) && (var2 == 0) && ...
 *
 * Recursively decomposes And expressions and checks each term for var == 0 pattern.
 */
class ReductionConditionDetector : public tvm::ExprVisitor {
 public:
  using tvm::ExprVisitor::Visit;
  using tvm::ExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const prim::AndNode* op) final {
    if (auto interrupt = Visit(ffi::AnyView(op->a))) return interrupt;
    return Visit(ffi::AnyView(op->b));
  }

  ffi::Optional<VisitInterrupt> Visit_(const prim::EQNode* op) final {
    // Check for pattern: var == 0
    if (const auto* var = op->a.as<VarNode>()) {
      if (const auto* imm = op->b.as<IntImmNode>()) {
        if (imm->value == 0) {
          vars_.insert(var);
          return std::nullopt;
        }
      }
    }
    // Also check reversed: 0 == var
    if (const auto* var = op->b.as<VarNode>()) {
      if (const auto* imm = op->a.as<IntImmNode>()) {
        if (imm->value == 0) {
          vars_.insert(var);
          return std::nullopt;
        }
      }
    }
    // Not a valid pattern
    is_valid_ = false;
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) final {
    // A bare variable in the condition is not our pattern
    is_valid_ = false;
    return std::nullopt;
  }

  /*!
   * \brief Analyze a condition expression
   * \param cond The condition to analyze
   * \return Pair of (set of variables in var == 0 pattern, is_valid flag)
   */
  static std::pair<std::unordered_set<const VarNode*>, bool> Analyze(const PrimExpr& cond) {
    auto detector = ffi::make_object<ReductionConditionDetector>();
    detector->Visit(ffi::AnyView(cond));
    return std::make_pair(detector->vars_, detector->is_valid_);
  }

 private:
  std::unordered_set<const VarNode*> vars_;
  bool is_valid_{true};
};

/*!
 * \brief Hoists reduction initialization statements outside of reduction loops.
 *
 * The pass works bottom-up: when visiting a For loop, it first processes
 * the body, then checks if the processed body contains an IfThenElse
 * with a reduction init pattern where this loop's variable appears.
 *
 * If the current loop variable is the last one in the condition, the
 * init statement is hoisted to before this loop. Otherwise, the condition
 * is simplified and passed up.
 */
class ReductionInitHoister : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    // First, process the body recursively. Expressions are never rewritten by this pass.
    Stmt new_body = Mutate(op->body, InplaceMode::kDisallow).ValueOrUnchanged(op->body);

    // Check if the body is a SeqStmt containing an IfThenElse init pattern
    ReductionInitInfo info = DetectInitPattern(new_body, op->loop_var.get());

    if (info.valid) {
      // This loop's variable was in the condition
      // Remove the init from the body - only keep the compute
      Stmt body_without_init = RemoveInitFromBody(new_body);
      Stmt new_for = For(op->loop_var, op->min, op->extent, op->kind, body_without_init,
                         op->thread_binding, op->annotations, std::nullopt, op->span);

      if (info.condition_vars.empty()) {
        // All condition variables have been processed - hoist the init completely
        return SeqStmt({info.init_stmt, new_for});
      } else {
        // There are still more reduction variables in outer loops
        // Hoist outside this loop but keep the conditional guard
        // This allows the parent loop to detect and continue hoisting
        PrimExpr new_cond = RebuildCondition(info.condition_vars);
        Stmt new_if = IfThenElse(new_cond, info.init_stmt);
        return SeqStmt({new_if, new_for});
      }
    }

    // No pattern found, return with updated body
    if (!new_body.same_as(op->body)) {
      return For(op->loop_var, op->min, op->extent, op->kind, new_body, op->thread_binding,
                 op->annotations, std::nullopt, op->span);
    }
    return ffi::Unchanged();
  }

 private:
  /*!
   * \brief Detect reduction init pattern in a statement.
   * \param stmt The statement to check (usually a SeqStmt)
   * \param loop_var The current loop variable to check against
   * \return Information about detected pattern
   */
  ReductionInitInfo DetectInitPattern(const Stmt& stmt, const VarNode* loop_var) {
    ReductionInitInfo info;

    // Handle SeqStmt - look for IfThenElse as first element
    if (const auto* seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() >= 1) {
        if (const auto* if_stmt = seq->seq[0].as<IfThenElseNode>()) {
          // Check if this is a reduction init pattern
          if (!if_stmt->else_case.defined()) {
            std::pair<std::unordered_set<const VarNode*>, bool> result =
                ReductionConditionDetector::Analyze(if_stmt->condition);
            if (result.second && !result.first.empty()) {
              // Check if the current loop variable is in the condition
              if (result.first.count(loop_var) > 0) {
                info.init_stmt = if_stmt->then_case;
                info.condition_vars = result.first;
                info.condition_vars.erase(loop_var);
                info.valid = true;
              }
            }
          }
        }
      }
    }

    // Also handle direct IfThenElse (not in SeqStmt)
    if (const auto* if_stmt = stmt.as<IfThenElseNode>()) {
      if (!if_stmt->else_case.defined()) {
        std::pair<std::unordered_set<const VarNode*>, bool> result =
            ReductionConditionDetector::Analyze(if_stmt->condition);
        if (result.second && !result.first.empty()) {
          if (result.first.count(loop_var) > 0) {
            info.init_stmt = if_stmt->then_case;
            info.condition_vars = result.first;
            info.condition_vars.erase(loop_var);
            info.valid = true;
          }
        }
      }
    }

    return info;
  }

  /*!
   * \brief Remove the IfThenElse init from a statement body.
   */
  Stmt RemoveInitFromBody(const Stmt& stmt) {
    if (const auto* seq = stmt.as<SeqStmtNode>()) {
      if (seq->seq.size() >= 1) {
        if (seq->seq[0].as<IfThenElseNode>()) {
          // Remove the first element (the IfThenElse)
          if (seq->seq.size() == 2) {
            return seq->seq[1];
          } else {
            ffi::Array<Stmt> new_seq;
            for (size_t i = 1; i < seq->seq.size(); ++i) {
              new_seq.push_back(seq->seq[i]);
            }
            return SeqStmt(new_seq);
          }
        }
      }
    }
    // For direct IfThenElse, we need special handling - return empty
    // This shouldn't happen in practice as there's always a compute after init
    return stmt;
  }

  /*!
   * \brief Insert init statement at the beginning of a body.
   */
  Stmt InsertInitInBody(const Stmt& stmt, const Stmt& init) {
    if (const auto* seq = stmt.as<SeqStmtNode>()) {
      ffi::Array<Stmt> new_seq;
      new_seq.push_back(init);
      for (const auto& s : seq->seq) {
        new_seq.push_back(s);
      }
      return SeqStmt(new_seq);
    }
    return SeqStmt({init, stmt});
  }

  /*!
   * \brief Rebuild condition from remaining variables.
   */
  PrimExpr RebuildCondition(const std::unordered_set<const VarNode*>& vars) {
    TVM_FFI_ICHECK(!vars.empty());
    std::vector<const VarNode*> sorted_vars(vars.begin(), vars.end());
    PrimExpr first_var = ffi::GetRef<Var>(sorted_vars[0]).as_or_throw<PrimExpr>();
    PrimExpr cond = equal(first_var, prim::MakeConst(first_var.ty(), 0));
    for (size_t i = 1; i < sorted_vars.size(); ++i) {
      PrimExpr v = ffi::GetRef<Var>(sorted_vars[i]).as_or_throw<PrimExpr>();
      cond = logical_and(cond, equal(v, prim::MakeConst(v.ty(), 0)));
    }
    return cond;
  }
};

PrimFunc HoistReductionInit(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  auto hoister = ffi::make_object<ReductionInitHoister>();
  fptr->body = hoister->Mutate(fptr->body, InplaceMode::kDisallow).ValueOrUnchanged(fptr->body);
  return func;
}

namespace transform {

Pass HoistReductionInit() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return HoistReductionInit(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.HoistReductionInit", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.HoistReductionInit", HoistReductionInit);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
