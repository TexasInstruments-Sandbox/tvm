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
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


def create_mlp_model():
    """Create and prepare MLP model for testing."""
    mod, param_spec = MLPModel().export_tvm(  # type: ignore
        {"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
    )

    # Initialize the weights for MLP
    device = tvm.cpu()
    params = [np.random.rand(*param.shape).astype("float32") for _, param in param_spec]
    params = [tvm.runtime.tensor(param, device=device) for param in params]

    # Convert parameters from inputs to embedded constants
    func_params_dict = dict(zip(mod["forward"].params[1:], params))
    mod = relax.transform.BindParams(func_name="forward", params=func_params_dict)(mod)

    return mod


# Parameters too large for source approach
@pytest.mark.parametrize(
    "target_c_static",
    [
        "c_static",
    ],
)
def test_mlp_comparison(target_c_static):
    """Test MLP model comparing llvm vs c_static targets."""
    mod = create_mlp_model()
    input_data = np.random.rand(1, 784).astype("float32")

    # Get results from both targets
    llvm_result = compile_and_run_on_target(
        target_string="llvm", mod=mod, input=input_data, entry_func_name="forward"
    )
    c_static_result = compile_and_run_on_target(
        target_string=target_c_static, mod=mod, input=input_data, entry_func_name="forward"
    )

    # Handle multi-output case: extract first output if model returns multiple outputs
    if isinstance(llvm_result, list):
        llvm_result = llvm_result[0]
    if isinstance(c_static_result, list):
        c_static_result = c_static_result[0]

    # Compare results
    assert np.allclose(llvm_result, c_static_result, rtol=1e-3, atol=1e-5), (
        f"Results differ for {target_c_static}. Max difference: {np.max(np.abs(llvm_result - c_static_result))}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
