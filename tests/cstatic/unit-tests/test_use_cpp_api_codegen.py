#!/usr/bin/env python
"""Test that use-cpp-api=1 codegen works for non-DSP c_static targets.

Verifies that the anylist compact-form intrinsics preserved by
LowerTVMBuiltin (when use-cpp-api is set) are correctly handled by
the c_static codegen regardless of whether a DSP mcpu is specified.

This is a codegen-only test: it checks the generated C++ source but
does not compile or run it (the AnyArray API requires the DSP runtime
which is not available in host-only builds).
"""

import os
import tarfile
import tempfile

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T


@I.ir_module
class MatmulModule:
    @T.prim_func(private=True)  # pyright: ignore
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:  # type: ignore
        A = T.match_buffer(x, (16, 16))  # type: ignore
        B = T.match_buffer(y, (16, 16))  # type: ignore
        C = T.match_buffer(z, (16, 16))  # type: ignore
        for i, j, k in T.grid(16, 16, 16):  # type: ignore
            with T.block("matmul"):
                vi = T.axis.S(16, i)  # type: ignore
                vj = T.axis.S(16, j)  # type: ignore
                vk = T.axis.R(16, k)  # type: ignore
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @R.function
    def main(
        x: R.Tensor((16, 16), "float32"),  # type: ignore
        w: R.Tensor((16, 16), "float32"),
    ) -> R.Tensor((16, 16), "float32"):  # type: ignore
        R.func_attr({"num_input": 1})  # type: ignore
        gv0 = R.call_tir(  # type: ignore
            MatmulModule.tir_matmul, (x, w), R.Tensor((16, 16), dtype="float32")
        )
        return gv0


def _build_and_get_source(mod, target_str):
    """Build a module through c_static and return the generated lib0.c source.

    Uses exec_mode="compiled" so that Relax VM operations (alloc_storage,
    alloc_tensor, etc.) are compiled to TIR as anylist_setitem_call_packed
    intrinsics, rather than being emitted as bytecode.
    """
    target = tvm.target.Target(target_str)
    with tvm.transform.PassContext(opt_level=0):
        ex = relax.build(mod, target=target, exec_mode="compiled")

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


def _prepare_model():
    """Bind a random weight to produce a single-input model."""
    w_np = np.random.rand(16, 16).astype(np.float32)
    params = {"w": tvm.runtime.tensor(w_np)}
    return relax.transform.BindParams("main", params)(MatmulModule)


def test_use_cpp_api_codegen_no_dsp():
    """use-cpp-api=1 without DSP mcpu must produce AnyArray-based code.

    LowerTVMBuiltin preserves anylist intrinsics when use-cpp-api is
    set.  The c_static codegen must handle these regardless of whether
    dsp_.enabled is true.  Before the fix, the codegen guarded this
    path on dsp_.enabled, causing the preserved intrinsics to fall
    through unhandled for non-DSP targets.
    """
    mod = _prepare_model()
    source = _build_and_get_source(mod, "c_static -use-cpp-api=1")

    assert source, "codegen produced empty output"
    assert "AnyArray" in source, (
        "Expected AnyArray wrappers in use-cpp-api=1 output; "
        "anylist intrinsics may not be handled for non-DSP targets"
    )


def test_use_cpp_api_disabled_no_anyarray():
    """use-cpp-api=0 must NOT produce AnyArray code (expanded path)."""
    mod = _prepare_model()
    source = _build_and_get_source(mod, "c_static -use-cpp-api=0")

    assert source, "codegen produced empty output"
    assert "AnyArray" not in source, (
        "AnyArray should not appear when use-cpp-api is disabled"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
