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
 * \file add_noalias.cc
 * \brief Add tir.noalias attribute to all PrimFuncs.
 *
 * This pass adds the "tir.noalias" attribute to all PrimFuncs, which
 * enables the code generator to emit `restrict` qualifiers on pointer
 * parameters. The restrict keyword informs the compiler that pointers
 * do not alias, enabling more aggressive optimizations.
 *
 * For TI C66x DSPs, the restrict keyword is critical for enabling
 * software pipelining and SIMD vectorization, as the compiler can
 * better schedule loads and stores when it knows pointers don't alias.
 */
#include <tvm/tir/function.h>
#include <tvm/tir/transform.h>
#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tir {

PrimFunc AddNoAlias(PrimFunc func) {
  // Check if noalias is already set
  if (func->HasNonzeroAttr(tir::attr::kNoAlias)) {
    return func;
  }

  // Add noalias attribute
  return WithAttr(std::move(func), tir::attr::kNoAlias, Integer(1));
}

namespace transform {

Pass AddNoAlias() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    return AddNoAlias(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.AddNoAlias", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.transform.AddNoAlias", AddNoAlias);
}

}  // namespace transform

}  // namespace tir
}  // namespace tvm
