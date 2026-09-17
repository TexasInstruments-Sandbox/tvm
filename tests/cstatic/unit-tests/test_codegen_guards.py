#!/usr/bin/env python
"""Codegen hardening tests for c_static_lib (P1-14 / P1-15).

- ``vm.builtin.read_if_cond`` must be emitted with null/type guards rather
  than an unchecked ``((DLTensor*)...)->data[0]`` read.
- A preserved anylist VM builtin with no compact-form handler must fail with
  an actionable codegen error instead of a cryptic base-class FATAL.
"""

import os
import tarfile
import tempfile

import pytest
import tvm
from tvm import tir
from tvm.script import ir as I
from tvm.script import relax as R


@I.ir_module
class IfModule:
    @R.function
    def main(
        x: R.Tensor((4,), "float32"),
        cond: R.Tensor((), "bool"),
    ) -> R.Tensor((4,), "float32"):
        R.func_attr({"num_input": 2})
        if cond:
            out = R.add(x, R.const(1.0, "float32"))
        else:
            out = R.subtract(x, R.const(1.0, "float32"))
        return out


def _build_and_get_source(mod, target_str):
    """Compile through c_static_lib and return the generated lib0.c source."""
    target = tvm.target.Target(target_str)
    with tvm.transform.PassContext(opt_level=0):
        ex = tvm.relax.build(mod, target=target, exec_mode="compiled")

    with tempfile.TemporaryDirectory() as td:
        tar_path = os.path.join(td, "model.tar")
        ex.export_library(tar_path, target=target)
        with tarfile.open(tar_path) as tf:
            tf.extractall(td)
        lib0_path = os.path.join(td, "lib0.c")
        if not os.path.exists(lib0_path):
            return ""
        with open(lib0_path) as f:
            return f.read()


def test_read_if_cond_is_guarded():
    """The direct read_if_cond path must null/type-check before dereferencing."""
    source = _build_and_get_source(IfModule, "c_static_lib -use-cpp-api=1")

    assert source, "codegen produced empty output"
    assert "vm.builtin.read_if_cond" in source

    # Guarded emission checks the argument is a tensor with a data pointer.
    assert "__cond_arg->type_index != kTVMFFITensor" in source
    assert "__cond_tensor->data == NULL" in source

    # The pre-fix form dereferenced the argument slot inline without a check.
    assert "UnwrapObjectRefArg(stack_ffi_any" not in source


def test_unhandled_anylist_builtin_clear_error():
    """Unhandled compact-form builtins must raise an actionable error."""
    r = tir.Var("r", "handle")
    body = tir.Evaluate(
        tir.op.anylist_setitem_call_packed(
            r, 0, "vm.builtin.tuple_getitem", tir.IntImm("int32", 0)
        )
    )
    func = tir.PrimFunc([r], body).with_attr("global_symbol", "main")
    mod = tvm.IRModule({"main": func})

    target = tvm.target.Target("c_static_lib -use-cpp-api=1")
    with pytest.raises(tvm.error.TVMError) as exc_info:
        tvm.target.codegen.build_module(mod, target)

    msg = str(exc_info.value)
    assert "vm.builtin.tuple_getitem" in msg
    assert "no compact-form handler" in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
