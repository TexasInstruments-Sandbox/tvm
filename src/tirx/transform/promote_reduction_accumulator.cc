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
 * \file promote_reduction_accumulator.cc
 * \brief Promote reduction accumulators from memory to registers.
 *
 * This pass transforms patterns like:
 *
 *   for (reduce_var) {
 *     buf[inv_idx] = buf[inv_idx] + expr;
 *   }
 *
 * Into:
 *
 *   acc = buf[inv_idx];
 *   for (reduce_var) {
 *     acc = acc + expr;
 *   }
 *   buf[inv_idx] = acc;
 *
 * Where inv_idx is loop-invariant with respect to reduce_var.
 *
 * This transformation eliminates the memory read-modify-write dependency
 * in the inner loop, enabling better software pipelining on DSPs like
 * TI C66x where the loop-carried dependency through memory limits the
 * initiation interval (ii).
 *
 * Before: ii=9 cycles (memory dependency)
 * After:  ii=1-2 cycles (register dependency)
 */
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tirx {

/*!
 * \brief Collects all variables used in an expression.
 */
class VarCollector : public tvm::ExprVisitor {
 public:
  using tvm::ExprVisitor::Visit;
  using tvm::ExprVisitor::Visit_;

  ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) final {
    vars_.insert(op);
    return std::nullopt;
  }

  static std::unordered_set<const VarNode*> Collect(const PrimExpr& expr) {
    auto collector = ffi::make_object<VarCollector>();
    collector->Visit(ffi::AnyView(expr));
    return collector->vars_;
  }

 private:
  std::unordered_set<const VarNode*> vars_;
};

/*!
 * \brief Structural equality of two index lists.
 */
inline bool IndicesMatch(const ffi::Array<PrimExpr>& a, const ffi::Array<PrimExpr>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!ffi::StructuralEqual()(a[i], b[i])) return false;
  }
  return true;
}

/*!
 * \brief Information about a detected reduction accumulation pattern.
 */
struct ReductionAccumInfo {
  /*! \brief The buffer being accumulated to */
  BufferVar buffer;
  /*! \brief The indices (must be loop-invariant) */
  ffi::Array<PrimExpr> indices;
  /*! \brief The expression being added (without the self-reference) */
  PrimExpr add_expr;
  /*! \brief Whether pattern is buf[idx] = buf[idx] + expr (true) or expr + buf[idx] (false) */
  bool self_on_left{true};
  /*! \brief Whether this is a valid pattern */
  bool valid{false};
};

/*!
 * \brief Detects reduction accumulation patterns in a BufferStore.
 *
 * Looks for patterns like:
 *   buf[idx] = buf[idx] + expr
 *   buf[idx] = expr + buf[idx]
 */
class ReductionPatternDetector {
 public:
  ReductionPatternDetector(const BufferVar& store_buffer, const ffi::Array<PrimExpr>& store_indices)
      : store_buffer_(store_buffer), store_indices_(store_indices) {}

  /*!
   * \brief Analyze a store value expression for reduction pattern.
   * \param value The value being stored
   * \return Information about detected pattern
   */
  ReductionAccumInfo Analyze(const PrimExpr& value) {
    ReductionAccumInfo info;

    // Check if value is an Add expression
    const auto* add = value.as<prim::AddNode>();
    if (!add) {
      return info;
    }

    // Check if either operand is a BufferLoad from the same buffer/indices
    if (MatchesSelfLoad(add->a)) {
      info.buffer = store_buffer_;
      info.indices = store_indices_;
      info.add_expr = add->b;
      info.self_on_left = true;
      info.valid = true;
    } else if (MatchesSelfLoad(add->b)) {
      info.buffer = store_buffer_;
      info.indices = store_indices_;
      info.add_expr = add->a;
      info.self_on_left = false;
      info.valid = true;
    }

    return info;
  }

 private:
  /*!
   * \brief Check if an expression is a BufferLoad from the store buffer at same indices.
   */
  bool MatchesSelfLoad(const PrimExpr& expr) {
    const auto* load = expr.as<TensorLoadNode>();
    if (!load) {
      return false;
    }

    // Check same buffer
    if (!load->source.same_as(store_buffer_)) {
      return false;
    }

    // Check same indices
    return IndicesMatch(load->indices, store_indices_);
  }

  const BufferVar& store_buffer_;
  const ffi::Array<PrimExpr>& store_indices_;
};

/*!
 * \brief Checks if buffer indices are invariant with respect to a loop variable.
 */
class LoopInvariantChecker {
 public:
  /*!
   * \brief Check if all indices are invariant with respect to the given loop variable.
   */
  static bool AreIndicesInvariant(const ffi::Array<PrimExpr>& indices, const VarNode* loop_var) {
    std::unordered_set<const VarNode*> index_vars;
    for (const auto& idx : indices) {
      auto vars = VarCollector::Collect(idx);
      index_vars.insert(vars.begin(), vars.end());
    }
    return index_vars.count(loop_var) == 0;
  }
};

/*!
 * \brief Promotes reduction accumulators to a 1-element local buffer.
 *
 * This allocates a 1-element buffer for the accumulator, which the
 * C backend lowers to a local variable. Combined with compiler optimization,
 * this should be kept in a register.
 */
class ReductionAccumulatorPromoter : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    // First recursively visit to handle nested loops
    Stmt new_body = Mutate(op->body, InplaceMode::kDisallow).ValueOrUnchanged(op->body);

    // Check if the (possibly transformed) body contains a reduction pattern
    ReductionAccumInfo info = DetectReductionPattern(new_body, op->loop_var.get());

    if (info.valid) {
      return TransformLoop(op, info, new_body);
    }

    if (!new_body.same_as(op->body)) {
      return For(op->loop_var, op->min, op->extent, op->kind, new_body, op->thread_binding,
                 op->annotations, std::nullopt, op->span);
    }
    return ffi::Unchanged();
  }

 private:
  int accum_counter_{0};

  ReductionAccumInfo DetectReductionPattern(const Stmt& body, const VarNode* loop_var) {
    ReductionAccumInfo info;

    // Find BufferStore in the body
    const BufferStoreNode* store = FindBufferStore(body);
    if (!store) {
      return info;
    }

    // Check if indices are loop-invariant
    if (!LoopInvariantChecker::AreIndicesInvariant(store->indices, loop_var)) {
      return info;
    }

    // Check for reduction pattern
    ReductionPatternDetector detector(store->buffer, store->indices);
    return detector.Analyze(store->value);
  }

  const BufferStoreNode* FindBufferStore(const Stmt& stmt) {
    if (const auto* store = stmt.as<BufferStoreNode>()) {
      return store;
    }
    if (const auto* seq = stmt.as<SeqStmtNode>()) {
      for (const auto& s : seq->seq) {
        if (auto* result = FindBufferStore(s)) {
          return result;
        }
      }
    }
    if (const auto* ite = stmt.as<IfThenElseNode>()) {
      if (auto* result = FindBufferStore(ite->then_case)) {
        return result;
      }
    }
    // The old LetStmt (a scoped binding with a body) no longer exists: Bind is a flat statement
    // that scopes over the rest of its enclosing sequence, which FindBufferStore already walks.
    return nullptr;
  }

  Stmt TransformLoop(const ForNode* op, const ReductionAccumInfo& info, const Stmt& body) {
    // Create accumulator buffer (1-element) with "local" storage scope
    // This tells the C backend to emit a stack-allocated variable instead of
    // using TVMBackendAllocWorkspace, which enables register allocation.
    std::string buf_name =
        std::string(info.buffer.name()) + "_acc_" + std::to_string(accum_counter_++);
    PrimType elem_dtype = info.buffer->dtype;
    BufferVar accum_buf(buf_name,
                        BufferType("local", elem_dtype, {IntImm::Int32(1)}, {}, PrimExpr(), 0, 0));

    // Create transformed body that uses accum_buf instead of info.buffer at info.indices
    Stmt new_body = ReplaceAccumulator(body, info, accum_buf);

    // Create the loop with transformed body
    Stmt new_loop = For(op->loop_var, op->min, op->extent, op->kind, new_body,
                        op->thread_binding, op->annotations, std::nullopt, op->span);

    // Load initial value and store final value
    ffi::Array<PrimExpr> zero_index{IntImm::Int32(0)};
    PrimExpr init_val = BufferLoad(info.buffer, info.indices);
    Stmt init_store = BufferStore(accum_buf, init_val, zero_index);
    Stmt final_load_store = BufferStore(info.buffer, BufferLoad(accum_buf, zero_index), info.indices);

    // The "disable_lower_builtin" annotation prevents LowerTVMBuiltin from
    // converting this to TVMBackendAllocWorkspace, keeping it as a stack variable
    // that the C compiler can optimize into a register.
    ffi::Map<ffi::String, ffi::Any> annotations;
    annotations.Set(transform::kDisableLowerTVMBuiltin, IntImm::Bool(true));
    return SeqStmt::Flatten(AllocBuffer(accum_buf, annotations), init_store, new_loop,
                            final_load_store);
  }

  Stmt ReplaceAccumulator(const Stmt& stmt, const ReductionAccumInfo& info,
                          const BufferVar& accum_buf) {
    class Replacer : public StmtExprMutator {
     public:
      using StmtExprMutator::Mutate;
      using StmtExprMutator::Mutate_;

      Replacer(const BufferVar& target_buf, const ffi::Array<PrimExpr>& target_indices,
               const BufferVar& accum_buf)
          : target_buf_(target_buf), target_indices_(target_indices), accum_buf_(accum_buf) {}

      UnchangedOr<PrimExpr> Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) final {
        if (op->source.same_as(target_buf_) && IndicesMatch(op->indices, target_indices_)) {
          return BufferLoad(accum_buf_, {IntImm::Int32(0)});
        }
        return StmtExprMutator::Mutate_(op, inplace_mode);
      }

      UnchangedOr<Stmt> Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) final {
        if (op->buffer.same_as(target_buf_) && IndicesMatch(op->indices, target_indices_)) {
          PrimExpr new_value = Mutate(op->value, InplaceMode::kDisallow).ValueOrUnchanged(op->value)
                                   .template as_or_throw<PrimExpr>();
          return BufferStore(accum_buf_, new_value, {IntImm::Int32(0)});
        }
        return StmtExprMutator::Mutate_(op, inplace_mode);
      }

     private:
      BufferVar target_buf_;
      ffi::Array<PrimExpr> target_indices_;
      BufferVar accum_buf_;
    };

    auto replacer = ffi::make_object<Replacer>(info.buffer, info.indices, accum_buf);
    return replacer->Mutate(stmt, InplaceMode::kDisallow).ValueOrUnchanged(stmt);
  }
};

PrimFunc PromoteReductionAccumulator(PrimFunc func) {
  auto fptr = func.CopyOnWrite();
  auto promoter = ffi::make_object<ReductionAccumulatorPromoter>();
  fptr->body = promoter->Mutate(fptr->body, InplaceMode::kDisallow).ValueOrUnchanged(fptr->body);
  return func;
}

namespace transform {

Pass PromoteReductionAccumulator() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return PromoteReductionAccumulator(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.PromoteReductionAccumulator", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.PromoteReductionAccumulator", PromoteReductionAccumulator);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
