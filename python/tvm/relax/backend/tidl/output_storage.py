# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Grow the storage backing a TIDL subgraph's output tensors.

TIDL writes more bytes than the logical output tensor holds: it appends
``outPadCh`` padding channels plus one spare, each ``outChannelPitch`` elements.
The memory planner sizes the output from shape x dtype alone, so ``algProcess``
runs off the end of the allocation and corrupts whatever follows -- measured as
8000 bytes of overrun on MobileNetV2 (4000-byte tensor, 12000 required) and
2048 on a small conv.

The padding lives *after* the real channels, so growing the allocation is
sufficient on its own: the tensor's shape, dtype and data pointer are all
unchanged and no copy is introduced.

This mirrors what the Relay backend does for the same hardware -- see
``OverrideTidlOutputStorageSizes`` in ``graph_executor_codegen.cc`` on the
``outbuff_initialization_TIDL`` branch of the Relay tree, which overrides the
planner's ``storage_size`` from the same ``params_1.bin`` descriptor.

Must run after ``StaticPlanBlockMemory`` (which introduces
``relax.memory.alloc_storage``) and before ``LowerAllocTensor``.  It is a no-op
unless the module carries the ``tidl.output_storage_bytes`` attribute, so the
shared CPU pipeline is unaffected for non-TIDL builds.
"""

import logging
import os
from typing import Dict, List

import tvm
from tvm import relax

logger = logging.getLogger(__name__)

MODULE_ATTR = "tidl.output_storage_bytes"

_ALLOC_STORAGE = tvm.ir.Op.get("relax.memory.alloc_storage")
_ALLOC_TENSOR = tvm.ir.Op.get("relax.memory.alloc_tensor")
_BUILTIN_ALLOC_TENSOR = tvm.ir.Op.get("relax.builtin.alloc_tensor")


def _extra_bytes() -> int:
    """Debug knob: extra bytes to add on top of the descriptor's requirement.

    For bisecting how much slack a new TIDL version needs, without rebuilding.
    """
    try:
        return max(0, int(os.environ.get("TIDL_OUTPUT_STORAGE_EXTRA_BYTES", "0")))
    except ValueError:
        return 0


def _as_requirements(attr) -> Dict[str, List[int]]:
    """Convert the module attribute into {subgraph_func_name: [bytes per slot]}."""
    extra = _extra_bytes()
    out: Dict[str, List[int]] = {}
    for name, sizes in attr.items():
        out[str(name)] = [int(s) + extra if int(s) > 0 else 0 for s in sizes]
    return out


@tvm.ir.transform.module_pass(opt_level=0, name="PadTIDLOutputStorage")
class PadTIDLOutputStorage:
    """Raise ``alloc_storage`` sizes feeding TIDL subgraph outputs."""

    def transform_module(self, mod: tvm.IRModule, _ctx) -> tvm.IRModule:
        attr = mod.attrs.get(MODULE_ATTR) if mod.attrs is not None else None
        if attr is None:
            return mod
        required = _as_requirements(attr)
        if not required:
            return mod

        # storage var name_hint -> required bytes, collected across the module.
        wanted: Dict[str, int] = {}
        # var name_hint of a builtin.alloc_tensor output -> required bytes.  The
        # planner leaves the function's returned tensor as builtin.alloc_tensor,
        # which has no size argument, so those are rewritten into an oversized
        # memory.alloc_storage plus a correctly-shaped memory.alloc_tensor view.
        split: Dict[str, int] = {}
        # tensor var name_hint -> storage var name_hint
        tensor_to_storage: Dict[str, str] = {}
        builtin_allocs: set = set()

        for _gv, func in mod.functions_items():
            if not isinstance(func, relax.Function):
                continue
            # First pass: remember which storage each alloc_tensor draws from.
            for binding in _iter_bindings(func):
                value = binding.value
                if not isinstance(value, relax.Call):
                    continue
                if value.op == _ALLOC_TENSOR and isinstance(value.args[0], relax.Var):
                    tensor_to_storage[binding.var.name] = value.args[0].name
                elif value.op == _BUILTIN_ALLOC_TENSOR:
                    builtin_allocs.add(binding.var.name)
            # Second pass: for each TIDL call, map its output args to storages.
            for binding in _iter_bindings(func):
                value = binding.value
                if not isinstance(value, relax.Call):
                    continue
                if not isinstance(value.op, relax.expr.GlobalVar):
                    continue
                sizes = required.get(value.op.name_hint)
                if not sizes:
                    continue
                # CallTIRRewrite appends the output tensors to the arg list, so
                # the outputs are the trailing len(sizes) arguments.
                out_args = list(value.args)[-len(sizes) :]
                for slot, (arg, nbytes) in enumerate(zip(out_args, sizes)):
                    if not isinstance(arg, relax.Var):
                        continue
                    storage = tensor_to_storage.get(arg.name)
                    if storage is not None:
                        wanted[storage] = max(wanted.get(storage, 0), nbytes)
                    elif arg.name in builtin_allocs:
                        split[arg.name] = max(split.get(arg.name, 0), nbytes)
                    else:
                        logger.warning(
                            "PadTIDLOutputStorage: %s output %d (%s) comes from neither "
                            "memory.alloc_tensor nor builtin.alloc_tensor; leaving its "
                            "allocation alone",
                            value.op.name_hint,
                            slot,
                            arg.name,
                        )

        if not wanted and not split:
            return mod

        grown: Dict[str, tuple] = {}
        grower = _StorageGrower(mod, wanted, grown, split)
        for gv, func in mod.functions_items():
            if isinstance(func, relax.Function):
                grower.builder_.update_func(gv, grower.visit_expr(func))
        new_mod = grower.builder_.get()

        for name, (old, new) in sorted(grown.items()):
            logger.info(
                "PadTIDLOutputStorage: %s grown %d -> %d bytes for TIDL output padding",
                name,
                old,
                new,
            )
        return new_mod


def _iter_bindings(func: relax.Function):
    """Yield every VarBinding in a function body, dataflow blocks included."""
    if not isinstance(func.body, relax.SeqExpr):
        return
    for block in func.body.blocks:
        for binding in block.bindings:
            if isinstance(binding, relax.VarBinding):
                yield binding


@relax.expr_functor.mutator
class _StorageGrower(relax.PyExprMutator):
    """Rewrite the size argument of the named alloc_storage calls."""

    def __init__(
        self,
        mod: tvm.IRModule,
        wanted: Dict[str, int],
        grown: Dict[str, tuple],
        split: Dict[str, int],
    ):
        super().__init__(mod)
        self._wanted = wanted
        self._grown = grown
        self._split = split
        self._current: str = ""

    def visit_var_binding_(self, binding: relax.VarBinding) -> None:
        self._current = binding.var.name
        super().visit_var_binding_(binding)
        self._current = ""

    def visit_call_(self, call: relax.Call) -> relax.Expr:  # noqa: D102
        call = super().visit_call_(call)
        if call.op == _BUILTIN_ALLOC_TENSOR:
            return self._split_builtin_alloc(call)
        if call.op != _ALLOC_STORAGE:
            return call
        nbytes = self._wanted.get(self._current)
        if nbytes is None:
            return call
        size_arg = call.args[0]
        # StaticPlanBlockMemory emits the byte count as a 1-D ShapeExpr.
        if not (isinstance(size_arg, relax.ShapeExpr) and len(size_arg.values) == 1):
            logger.warning(
                "PadTIDLOutputStorage: %s has an unexpected alloc_storage size "
                "argument (%s); leaving it alone",
                self._current,
                type(size_arg).__name__,
            )
            return call
        current = size_arg.values[0]
        if not isinstance(current, tvm.tirx.IntImm):
            logger.warning(
                "PadTIDLOutputStorage: %s has a symbolic storage size; leaving it alone",
                self._current,
            )
            return call
        if int(current) >= nbytes:
            return call
        self._grown[self._current] = (int(current), nbytes)
        new_size = relax.ShapeExpr([tvm.tirx.IntImm(current.dtype, nbytes)])
        new_args = [new_size] + list(call.args)[1:]
        return relax.Call(call.op, new_args, call.attrs, call.ty_args)

    def _split_builtin_alloc(self, call: relax.Call) -> relax.Expr:
        """Turn ``builtin.alloc_tensor`` into oversized storage + a tensor view.

        ``builtin.alloc_tensor`` derives its size from the shape alone, so there
        is no size argument to raise.  Emitting the storage explicitly lets the
        allocation be larger than the tensor while the tensor keeps its real
        shape, dtype and (offset 0) data pointer -- no copy, and the consumer
        sees exactly the tensor it expects.
        """
        nbytes = self._split.get(self._current)
        if nbytes is None:
            return call
        shape, dtype = call.args[0], call.args[1]
        if not isinstance(shape, relax.ShapeExpr):
            logger.warning(
                "PadTIDLOutputStorage: %s has a non-static shape; leaving its "
                "allocation alone",
                self._current,
            )
            return call

        # Never allocate less than the tensor itself needs.  Unlike the
        # alloc_storage path there is no existing size to compare against here,
        # so the floor has to be recomputed from shape x dtype: a descriptor
        # reporting a smaller requirement must not shrink the allocation.
        logical = 1
        for dim in shape.values:
            if not isinstance(dim, tvm.tirx.IntImm):
                logger.warning(
                    "PadTIDLOutputStorage: %s has a symbolic dimension; leaving "
                    "its allocation alone",
                    self._current,
                )
                return call
            logical *= int(dim)
        logical *= (tvm.DataType(dtype.value).bits * tvm.DataType(dtype.value).lanes) // 8
        if nbytes <= logical:
            return call
        # builtin.alloc_tensor(shape, dtype, runtime_device_index, storage_scope)
        storage_scope = call.args[3] if len(call.args) > 3 else relax.StringImm("global")
        vdevice = call.args[2] if len(call.args) > 2 else relax.prim_value(0)
        storage = self.builder_.emit(
            relax.op.memory.alloc_storage(
                relax.ShapeExpr([tvm.tirx.IntImm("int64", nbytes)]),
                vdevice,
                storage_scope,
                dtype,
            ),
            name_hint=f"{self._current}_tidl_storage",
        )
        self._grown[self._current] = (0, nbytes)
        return relax.op.memory.alloc_tensor(storage, relax.prim_value(0), shape, dtype)
