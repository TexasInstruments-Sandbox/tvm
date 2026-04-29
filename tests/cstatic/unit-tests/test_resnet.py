#!/usr/bin/env python

import numpy as np
import pytest
import torch
import tvm
from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18
from tvm.relax.frontend.torch import from_exported_program
from tvm_utils import process_relax, compile_and_run_on_target


def torch_to_relax(torch_model, example_input) -> tvm.IRModule:
    """Convert a PyTorch model to a Relax IRModule."""
    with torch.no_grad():
        exported_program = export(torch_model, example_input)
        mod = from_exported_program(exported_program, keep_params_as_input=True)
    return mod


def create_resnet_model():
    """Create and prepare ResNet-18 model for testing."""
    # Initialize torch model with pre-trained weights
    torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

    # Create example input for torch.export
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

    # Convert to Relax IRModule and process
    mod = torch_to_relax(torch_model, example_args)
    mod = process_relax(mod)
    return mod


# Parameters are too large to use the C source approach
@pytest.mark.parametrize("target_c_static", ["c_static"])
def test_resnet_comparison(target_c_static):
    """Test ResNet-18 model comparing llvm vs c_static targets."""
    mod = create_resnet_model()
    input_data = np.random.rand(1, 3, 224, 224).astype("float32")

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
