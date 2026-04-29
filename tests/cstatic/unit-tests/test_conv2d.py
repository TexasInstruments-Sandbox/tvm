#!/usr/bin/env python

import numpy as np
import pytest
import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm_utils import compile_and_run_on_target


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False
        )

    def main(self, x):
        x = self.conv1(x)
        return x


def create_conv2d_model():
    """Create and prepare conv2d model for testing."""
    mod, param_spec = MLPModel().export_tvm(  # type: ignore
        spec={"main": {"x": nn.spec.Tensor((1, 1, 32, 32), "float32")}}
    )

    # Initialize the weights for conv2d
    device = tvm.cpu()
    params = [np.random.rand(*param.shape).astype("float32") for _, param in param_spec]
    params = [tvm.runtime.tensor(param, device=device) for param in params]

    # Convert parameters from inputs to embedded constants
    func_params_dict = dict(zip(mod["main"].params[1:], params))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)

    return mod


@pytest.mark.parametrize(
    "target_c_static", ["c_static"]
)
def test_conv2d_comparison(target_c_static):
    """Test conv2d model comparing llvm vs c_static targets."""
    mod = create_conv2d_model()
    input_data = np.full((1, 1, 32, 32), 42.0, dtype="float32")

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
