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
 * \file simplify_power_two.cc
 * \brief Replace pow(x, 2.0) with x * x for better performance.
 *
 * This pass replaces calls to the power function with exponent 2
 * with a simple multiplication. This is beneficial because:
 * 1. powf() is a general-purpose function with significant overhead
 * 2. x * x is a single multiply instruction
 * 3. On DSPs like TI C66x, this enables better instruction scheduling
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tirx {

/*!
 * \brief Replaces pow(x, 2) with x * x in expressions.
 */
class PowerTwoSimplifier : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  UnchangedOr<Expr> Mutate_(const CallNode* op, InplaceMode inplace_mode) final {
    // First visit children.
    auto result = StmtExprMutator::Mutate_(op, inplace_mode);
    if (!result.IsUnchanged()) {
      op = ffi::AnyView(result).as<CallNode>();
    }

    // Check if this is a pow call.
    static const Op& pow_op = Op::Get("tirx.pow");
    if (op != nullptr && op->op.same_as(pow_op) && op->args.size() == 2) {
      PrimExpr base = op->args[0].as_or_throw<PrimExpr>();
      PrimExpr exponent = op->args[1].as_or_throw<PrimExpr>();

      // Check if exponent is constant 2.0 (or 2).
      if (const FloatImmNode* float_imm = exponent.as<FloatImmNode>()) {
        if (float_imm->value == 2.0) {
          // Replace pow(x, 2.0) with x * x.
          ++num_replacements_;
          return mul(base, base, op->span);
        }
      } else if (const IntImmNode* int_imm = exponent.as<IntImmNode>()) {
        if (int_imm->value == 2) {
          // Replace pow(x, 2) with x * x.
          ++num_replacements_;
          return mul(base, base, op->span);
        }
      }
    }

    return result;
  }

  int num_replacements() const { return num_replacements_; }

 private:
  int num_replacements_{0};
};

PrimFunc SimplifyPowerTwo(PrimFunc func) {
  auto simplifier = ffi::make_object<PowerTwoSimplifier>();
  auto fptr = func.CopyOnWrite();
  fptr->body = simplifier->Mutate(fptr->body, InplaceMode::kAllow).ValueOrUnchanged(fptr->body);
  return func;
}

namespace transform {

Pass SimplifyPowerTwo() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return SimplifyPowerTwo(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.SimplifyPowerTwo", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.SimplifyPowerTwo", SimplifyPowerTwo);
}

}  // namespace transform

}  // namespace tirx
}  // namespace tvm
