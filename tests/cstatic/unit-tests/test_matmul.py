#!/usr/bin/env python

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm_utils import compile_and_run_on_target


@I.ir_module
class InputModule:
    @T.prim_func(private=True)  # pyright: ignore
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:  # type: ignore
        A = T.match_buffer(x, (16, 16))  # type: ignore
        B = T.match_buffer(y, (16, 16))  # type: ignore
        C = T.match_buffer(z, (16, 16))  # type: ignore
        for i0, j, k0, i1, k1 in T.grid(4, 16, 4, 4, 4):  # type: ignore
            with T.block("matmul"):
                vi = T.axis.S(16, i0 * 4 + i1)  # type: ignore
                vj = T.axis.S(16, j)  # type: ignore
                vk = T.axis.R(16, k0 * 4 + k1)  # type: ignore
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

    @R.function
    def main(
        x: R.Tensor((16, 16), "float32"),  # type: ignore
        w: R.Tensor((16, 16), "float32"),
    ) -> R.Tensor(
        (16, 16),
        "float32",  # type: ignore
    ):
        R.func_attr({"num_input": 1})  # type: ignore
        gv0 = R.call_tir(InputModule.tir_matmul, (x, w), R.Tensor((16, 16), dtype="float32"))  # type: ignore
        return gv0


def create_matmul_model():
    """Create and prepare matmul model for testing."""
    # Create weight parameter
    w_np = np.random.rand(16, 16).astype(np.float32)
    w_tvm = tvm.runtime.tensor(w_np)

    # Convert weight parameter from input to embedded constant
    params_dict = {"w": w_tvm}
    mod = relax.transform.BindParams("main", params_dict)(InputModule)

    return mod


@pytest.mark.parametrize(
    "target_c_static", ["c_static"]
)
def test_matmul_comparison(target_c_static):
    """Test matmul model comparing llvm vs c_static targets."""
    mod = create_matmul_model()
    input_data = np.random.rand(16, 16).astype(np.float32)

    # Get results from both targets
    llvm_result = compile_and_run_on_target(target_string="llvm", mod=mod, input=input_data)
    c_static_result = compile_and_run_on_target(
        target_string=target_c_static, mod=mod, input=input_data
    )

    # Compare results
    assert np.allclose(llvm_result, c_static_result, rtol=1e-3, atol=1e-5), (
        f"Results differ for {target_c_static}. Max difference: {np.max(np.abs(llvm_result - c_static_result))}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
