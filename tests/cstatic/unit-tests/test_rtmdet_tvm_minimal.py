"""Pytest tests for RTMDet TVM compilation with flattened output structure

This test suite validates TVM's ability to compile and execute multi-output object
detection models using the C Static backend. It uses a mock RTMDet architecture that
returns 6 tensors (3 classification + 3 bounding box predictions across FPN levels).

Test Coverage:
    - Output structure validation (tuple of 6 tensors with correct shapes)
    - TVM IR export from PyTorch via torch.export
    - Multi-output tuple detection and handling
    - Full C Static backend compilation and execution
    - Output shape preservation through the compilation pipeline

Architecture:
    MockRTMDetWrapper simulates RTMDet's FPN output structure:
    - Input: [1, 3, 640, 640] RGB image
    - Output: Flat tuple of 6 tensors
        * cls_0, cls_1, cls_2: Classification scores at 3 scales
        * bbox_0, bbox_1, bbox_2: Bounding box predictions at 3 scales

Usage:
    # Run all tests
    pytest test_rtmdet_tvm_minimal.py -v

    # Run specific test class
    pytest test_rtmdet_tvm_minimal.py::TestTVMCompilation -v

    # Run single test
    pytest test_rtmdet_tvm_minimal.py::TestTVMCompilation::test_full_compilation_with_c_static -v

Context:
    This test infrastructure was developed to enable TVM compilation for RTMDet object
    detection models, validating that the C Static backend can handle complex multi-output
    architectures common in modern detection models (YOLO, RT-DETR, etc.).

See Also:
    - od_rtmdet.py: Full RTMDet implementation with TVM support
    - od_rtmdet_pure.py: Pure PyTorch RTMDet with TVM compilation
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

from tvm_utils import compile_and_run_on_target, model_returns_tuple, process_relax


class MockRTMDetWrapper(nn.Module):
    """Mock RTMDet that returns flat tuple of 6 tensors"""

    def __init__(self):
        super().__init__()
        # Simple conv layers to simulate backbone/neck/head
        self.conv1 = nn.Conv2d(3, 80, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Returns flat tuple of 6 tensors matching RTMDet output structure

        For 640x640 input, returns:
        - 3 classification tensors: [1,80,80,80], [1,80,40,40], [1,80,20,20]
        - 3 bbox tensors: [1,4,80,80], [1,4,40,40], [1,4,20,20]
        """
        # Simulate FPN levels with different resolutions
        cls_0 = torch.nn.functional.interpolate(self.conv1(x), size=(80, 80))
        cls_1 = torch.nn.functional.interpolate(self.conv1(x), size=(40, 40))
        cls_2 = torch.nn.functional.interpolate(self.conv1(x), size=(20, 20))

        bbox_0 = torch.nn.functional.interpolate(self.conv2(x), size=(80, 80))
        bbox_1 = torch.nn.functional.interpolate(self.conv2(x), size=(40, 40))
        bbox_2 = torch.nn.functional.interpolate(self.conv2(x), size=(20, 20))

        # Return flat tuple of 6 tensors
        return (cls_0, cls_1, cls_2, bbox_0, bbox_1, bbox_2)


class TestRTMDetWrapper:
    """Tests for RTMDet wrapper output structure"""

    def test_output_structure(self):
        """Test that the wrapper returns the expected flat tuple structure"""
        model = MockRTMDetWrapper()
        model.eval()

        # Create test input
        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            outputs = model(x)

        # Verify it's a tuple with 6 elements
        assert isinstance(outputs, tuple), f"Expected tuple, got {type(outputs)}"
        assert len(outputs) == 6, f"Expected 6 outputs, got {len(outputs)}"

    def test_output_shapes(self):
        """Test that all output shapes are correct"""
        model = MockRTMDetWrapper()
        model.eval()

        # Create test input
        x = torch.randn(1, 3, 640, 640)

        with torch.no_grad():
            outputs = model(x)

        # Verify shapes
        expected_shapes = [
            (1, 80, 80, 80),  # cls_0
            (1, 80, 40, 40),  # cls_1
            (1, 80, 20, 20),  # cls_2
            (1, 4, 80, 80),  # bbox_0
            (1, 4, 40, 40),  # bbox_1
            (1, 4, 20, 20),  # bbox_2
        ]

        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            actual_shape = tuple(output.shape)
            assert actual_shape == expected_shape, (
                f"Output {i}: expected shape {expected_shape}, got {actual_shape}"
            )


class TestTVMExport:
    """Tests for TVM export functionality"""

    def test_export_to_ir(self):
        """Test that the model can be exported to TVM IR"""
        model = MockRTMDetWrapper()
        model.eval()

        # Create example input for export
        example_input = (torch.randn(1, 3, 640, 640),)

        # Export to TVM
        with torch.no_grad():
            exported_program = export(model, example_input, strict=False)
            mod = from_exported_program(exported_program, keep_params_as_input=True)

        # Check that main function exists
        assert "main" in mod, "Main function not found in module"

    def test_export_returns_main_function(self):
        """Test that exported module has main function with proper structure"""
        model = MockRTMDetWrapper()
        model.eval()

        # Create example input for export
        example_input = (torch.randn(1, 3, 640, 640),)

        # Export to TVM
        with torch.no_grad():
            exported_program = export(model, example_input, strict=False)
            mod = from_exported_program(exported_program, keep_params_as_input=True)

        # Check return structure
        main_func = mod["main"]
        ret_info = main_func.ret_struct_info
        assert ret_info is not None, "Return structure info should not be None"

    def test_model_returns_tuple(self):
        """Test that exported model is detected as returning a tuple"""
        model = MockRTMDetWrapper()
        model.eval()

        # Create example input for export
        example_input = (torch.randn(1, 3, 640, 640),)

        # Export to TVM
        with torch.no_grad():
            exported_program = export(model, example_input, strict=False)
            mod = from_exported_program(exported_program, keep_params_as_input=True)

        # The return should be detected as a tuple
        returns_tuple = model_returns_tuple(mod, "main")
        assert returns_tuple, "Expected model to return tuple"


class TestTVMCompilation:
    """Tests for full TVM compilation"""

    def test_full_compilation_with_c_static(self):
        """Test full TVM compilation with C Static backend"""
        model = MockRTMDetWrapper()
        model.eval()

        # Export to TVM
        example_input = (torch.randn(1, 3, 640, 640),)
        with torch.no_grad():
            exported_program = export(model, example_input, strict=False)
            mod = from_exported_program(exported_program, keep_params_as_input=True)

        # Process the module (detach and bind parameters)
        mod = process_relax(mod)

        # Create test input
        test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

        # Compile and run on C Static target
        outputs = compile_and_run_on_target(
            target_string="c_static",
            mod=mod,
            input=test_input,
            verbose_output=False,
        )

        # Verify outputs is a list
        assert isinstance(outputs, list), f"Expected list, got {type(outputs)}"

    def test_compilation_output_count(self):
        """Test that compilation produces correct number of outputs"""
        model = MockRTMDetWrapper()
        model.eval()

        # Export to TVM
        example_input = (torch.randn(1, 3, 640, 640),)
        with torch.no_grad():
            exported_program = export(model, example_input, strict=False)
            mod = from_exported_program(exported_program, keep_params_as_input=True)

        # Process the module (detach and bind parameters)
        mod = process_relax(mod)

        # Create test input
        test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

        # Compile and run on C Static target
        outputs = compile_and_run_on_target(
            target_string="c_static",
            mod=mod,
            input=test_input,
            verbose_output=False,
        )

        assert len(outputs) == 6, f"Expected 6 outputs, got {len(outputs)}"

    def test_compilation_output_shapes(self):
        """Test that compilation produces correct output shapes"""
        model = MockRTMDetWrapper()
        model.eval()

        # Export to TVM
        example_input = (torch.randn(1, 3, 640, 640),)
        with torch.no_grad():
            exported_program = export(model, example_input, strict=False)
            mod = from_exported_program(exported_program, keep_params_as_input=True)

        # Process the module (detach and bind parameters)
        mod = process_relax(mod)

        # Create test input
        test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

        # Compile and run on C Static target
        outputs = compile_and_run_on_target(
            target_string="c_static",
            mod=mod,
            input=test_input,
            verbose_output=False,
        )

        # Verify output shapes
        expected_shapes = [
            (1, 80, 80, 80),  # cls_0
            (1, 80, 40, 40),  # cls_1
            (1, 80, 20, 20),  # cls_2
            (1, 4, 80, 80),  # bbox_0
            (1, 4, 40, 40),  # bbox_1
            (1, 4, 20, 20),  # bbox_2
        ]

        for i, (output, expected_shape) in enumerate(zip(outputs, expected_shapes)):
            actual_shape = tuple(output.shape)
            assert actual_shape == expected_shape, (
                f"Output {i}: expected shape {expected_shape}, got {actual_shape}"
            )
