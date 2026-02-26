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
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tir {

/*!
 * \brief Replaces pow(x, 2) with x * x in expressions.
 */
class PowerTwoSimplifier : public StmtExprMutator {
 public:
  PrimExpr VisitExpr_(const CallNode* op) override {
    // First visit children
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    const CallNode* call = expr.as<CallNode>();
    if (call == nullptr) {
      return expr;
    }

    // Check if this is a pow call
    static const Op& pow_op = Op::Get("tir.pow");
    if (call->op.same_as(pow_op) && call->args.size() == 2) {
      PrimExpr base = call->args[0];
      PrimExpr exponent = call->args[1];

      // Check if exponent is constant 2.0 (or 2)
      if (const FloatImmNode* float_imm = exponent.as<FloatImmNode>()) {
        if (float_imm->value == 2.0) {
          // Replace pow(x, 2.0) with x * x
          ++num_replacements_;
          return mul(base, base, call->span);
        }
      } else if (const IntImmNode* int_imm = exponent.as<IntImmNode>()) {
        if (int_imm->value == 2) {
          // Replace pow(x, 2) with x * x
          ++num_replacements_;
          return mul(base, base, call->span);
        }
      }
    }

    return expr;
  }

  int num_replacements() const { return num_replacements_; }

 private:
  int num_replacements_{0};
};

PrimFunc SimplifyPowerTwo(PrimFunc func) {
  PowerTwoSimplifier simplifier;
  auto fptr = func.CopyOnWrite();
  fptr->body = simplifier(std::move(fptr->body));
  return func;
}

namespace transform {

Pass SimplifyPowerTwo() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return SimplifyPowerTwo(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.SimplifyPowerTwo", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.transform.SimplifyPowerTwo", SimplifyPowerTwo);
}

}  // namespace transform

}  // namespace tir
}  // namespace tvm
