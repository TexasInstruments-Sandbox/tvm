"""
Model creation utilities for DSP tests.

This module provides functions to create TVM Relax models from PyTorch for DSP testing.
Each model creation function returns a tuple of (mod, torch_model, example_input) to
enable both TVM compilation and PyTorch reference inference.

Usage:
    from model_utils import create_conv2d_model

    mod, torch_model, example_input = create_conv2d_model()
    # mod: TVM IRModule with parameters bound
    # torch_model: PyTorch model for reference inference
    # example_input: numpy array for test input
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.export import export

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

_THIS_DIR = Path(__file__).parent


def torch_to_relax_with_params(
    torch_model: nn.Module,
    example_input: tuple,
) -> tvm.IRModule:
    """
    Convert PyTorch model to TVM Relax IRModule with parameters bound.

    This function:
    1. Exports the PyTorch model using torch.export
    2. Converts to TVM Relax IR
    3. Detaches and binds parameters as constants

    Args:
        torch_model: PyTorch model in eval mode
        example_input: Tuple of example input tensors for tracing

    Returns:
        TVM IRModule with parameters bound as constants
    """
    torch_model.eval()

    with torch.no_grad():
        exported_program = export(torch_model, example_input)
        mod = from_exported_program(exported_program, keep_params_as_input=True)

    # Detach and bind parameters
    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)  # pyright: ignore[reportArgumentType]

    return mod


# -----------------------------------------------------------------------------
# Conv2D Model
# -----------------------------------------------------------------------------


class Conv2DModel(nn.Module):
    """Simple Conv2D model for testing."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        input_size: int = 32,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias=False,
        )
        self.input_size = input_size
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def create_conv2d_model(
    in_channels: int = 1,
    out_channels: int = 1,
    kernel_size: int = 3,
    input_size: int = 32,
    seed: int = 42,
) -> tuple:
    """
    Create Conv2D model for DSP testing.

    Args:
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 1)
        kernel_size: Convolution kernel size (default: 3)
        input_size: Spatial dimension of input (default: 32)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (tvm_mod, torch_model, input_data):
        - tvm_mod: TVM IRModule with parameters bound
        - torch_model: PyTorch model for reference inference
        - input_data: numpy array for test input [1, in_channels, input_size, input_size]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    torch_model = Conv2DModel(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        input_size=input_size,
    )
    torch_model.eval()

    # Create example input for tracing
    example_input = (torch.randn(1, in_channels, input_size, input_size),)

    # Convert to TVM
    tvm_mod = torch_to_relax_with_params(torch_model, example_input)

    # Create deterministic test input
    # Using sequential values for reproducibility across platforms
    total_elements = in_channels * input_size * input_size
    input_data = np.array([(i + 1) * 0.1 for i in range(total_elements)], dtype=np.float32)
    input_data = input_data.reshape(1, in_channels, input_size, input_size)

    return tvm_mod, torch_model, input_data


# -----------------------------------------------------------------------------
# MLP Model
# -----------------------------------------------------------------------------


class MLPModel(nn.Module):
    """Simple MLP model for testing."""

    def __init__(
        self,
        input_size: int = 784,
        hidden_size: int = 128,
        output_size: int = 10,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_mlp_model(
    input_size: int = 784,
    hidden_size: int = 128,
    output_size: int = 10,
    seed: int = 42,
) -> tuple:
    """
    Create MLP model for DSP testing.

    Args:
        input_size: Input feature dimension (default: 784)
        hidden_size: Hidden layer size (default: 128)
        output_size: Output dimension (default: 10)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (tvm_mod, torch_model, input_data):
        - tvm_mod: TVM IRModule with parameters bound
        - torch_model: PyTorch model for reference inference
        - input_data: numpy array for test input [1, input_size]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    torch_model = MLPModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
    )
    torch_model.eval()

    # Create example input for tracing
    example_input = (torch.randn(1, input_size),)

    # Convert to TVM
    tvm_mod = torch_to_relax_with_params(torch_model, example_input)

    # Create deterministic test input
    input_data = np.array([(i + 1) * 0.01 for i in range(input_size)], dtype=np.float32)
    input_data = input_data.reshape(1, input_size)

    return tvm_mod, torch_model, input_data


# -----------------------------------------------------------------------------
# Matmul Model
# -----------------------------------------------------------------------------


class MatmulModel(nn.Module):
    """Simple matrix multiplication model for testing."""

    def __init__(self, m: int = 64, k: int = 64, n: int = 64):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(k, n))
        self.m = m
        self.k = k
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weight)


def create_matmul_model(
    m: int = 64,
    k: int = 64,
    n: int = 64,
    seed: int = 42,
) -> tuple:
    """
    Create Matmul model for DSP testing.

    Args:
        m: First dimension of input matrix (default: 64)
        k: Second dimension of input / first dimension of weight (default: 64)
        n: Second dimension of weight / output (default: 64)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (tvm_mod, torch_model, input_data):
        - tvm_mod: TVM IRModule with parameters bound
        - torch_model: PyTorch model for reference inference
        - input_data: numpy array for test input [m, k]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    torch_model = MatmulModel(m=m, k=k, n=n)
    torch_model.eval()

    # Create example input for tracing
    example_input = (torch.randn(m, k),)

    # Convert to TVM
    tvm_mod = torch_to_relax_with_params(torch_model, example_input)

    # Create deterministic test input
    input_data = np.array([(i + 1) * 0.01 for i in range(m * k)], dtype=np.float32)
    input_data = input_data.reshape(m, k)

    return tvm_mod, torch_model, input_data


# -----------------------------------------------------------------------------
# CLISTA-DoA Model (Convolutional LISTA for Direction of Arrival)
# -----------------------------------------------------------------------------


def create_clista_model(
    num_iterations: int = 8,
    num_atoms: int = 64,
    num_antennas: int = 16,
    seed: int = 42,
) -> tuple:
    """
    Create CLISTA-DoA model for DSP testing.

    CLISTA-DoA is a convolutional learned ISTA model for radar direction of arrival
    estimation. It processes complex I/Q radar signals and produces a sparse
    angular spectrum representation.

    Args:
        num_iterations: Number of ISTA unrolling iterations (default: 8)
        num_atoms: Number of dictionary atoms / angular bins (default: 64)
        num_antennas: Number of virtual antennas (default: 16)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (tvm_mod, torch_model, input_data):
        - tvm_mod: TVM IRModule with parameters bound
        - torch_model: PyTorch model for reference inference
        - input_data: numpy array for test input [1, 2, num_antennas]
    """
    from models.clista import CLISTA_DoA

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    torch_model = CLISTA_DoA(
        num_iterations=num_iterations,
        num_atoms=num_atoms,
        num_antennas=num_antennas,
    )
    torch_model.eval()

    # Create example input for tracing: [batch, 2 (I/Q), num_antennas]
    example_input = (torch.randn(1, 2, num_antennas, dtype=torch.float32),)

    # Convert to TVM
    tvm_mod = torch_to_relax_with_params(torch_model, example_input)

    # Create deterministic test input
    # Using sequential values for reproducibility across platforms
    total_elements = 2 * num_antennas
    input_data = np.array([(i + 1) * 1.0 for i in range(total_elements)], dtype=np.float32)
    input_data = input_data.reshape(1, 2, num_antennas)

    return tvm_mod, torch_model, input_data


# -----------------------------------------------------------------------------
# LeNet-5 Model
# -----------------------------------------------------------------------------


class LeNet5Model(nn.Module):
    """
    LeNet-5 style model for MNIST digit classification.

    Configurable channel counts to allow smaller models for DSP.

    Default (small) architecture for DSP:
    - Conv1: 1 -> 4 channels, 5x5 kernel
    - Pool1: 2x2 avg pool
    - Conv2: 4 -> 8 channels, 5x5 kernel
    - Pool2: 2x2 avg pool
    - FC1: 8*4*4=128 -> 32
    - FC2: 32 -> 10

    Memory estimate for small config (~5KB weights):
    - Conv1: 4*1*5*5 = 100 params
    - Conv2: 8*4*5*5 = 800 params
    - FC1: 128*32 = 4,096 params
    - FC2: 32*10 = 320 params
    - Total: ~5.3K params = ~21KB (float32)
    """

    def __init__(self, conv1_out: int = 4, conv2_out: int = 8, fc1_out: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # After conv+pool on 28x28: 28->24->12->8->4, so 4x4 spatial
        self.fc1 = nn.Linear(conv2_out * 4 * 4, fc1_out)
        self.fc2 = nn.Linear(fc1_out, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_lenet_model(
    conv1_out: int = 4,
    conv2_out: int = 8,
    fc1_out: int = 32,
    seed: int = 42,
) -> tuple:
    """
    Create LeNet-5 style model for DSP testing.

    LeNet-5 is a classic CNN for MNIST digit classification.
    Default uses smaller channel counts to fit in C66x L2 memory.

    Args:
        conv1_out: Output channels for first conv layer (default: 4)
        conv2_out: Output channels for second conv layer (default: 8)
        fc1_out: Output size for first FC layer (default: 32)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (tvm_mod, torch_model, input_data):
        - tvm_mod: TVM IRModule with parameters bound
        - torch_model: PyTorch model for reference inference
        - input_data: numpy array for test input [1, 1, 28, 28]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model with configurable sizes
    torch_model = LeNet5Model(
        conv1_out=conv1_out,
        conv2_out=conv2_out,
        fc1_out=fc1_out,
    )
    torch_model.eval()

    # Create example input for tracing: [batch, channels, height, width]
    # MNIST input size: 28x28 grayscale
    example_input = (torch.randn(1, 1, 28, 28),)

    # Convert to TVM
    tvm_mod = torch_to_relax_with_params(torch_model, example_input)

    # Create deterministic test input
    # Using scaled sequential values for reproducibility
    total_elements = 1 * 28 * 28
    input_data = np.array([(i + 1) * 0.01 for i in range(total_elements)], dtype=np.float32)
    input_data = input_data.reshape(1, 1, 28, 28)

    return tvm_mod, torch_model, input_data


# -----------------------------------------------------------------------------
# Conv2D Stack Model (ResNet-like conv2d layers without skip connections)
# -----------------------------------------------------------------------------


class Conv2DStackModel(nn.Module):
    """
    Sequential stack of conv2d + batch_norm + relu layers.

    Isolates the most expensive conv2d configurations found in ResNet18
    without skip connections, making it faster to compile and debug.

    Architecture:
    - conv1: 3->64, 3x3, stride=1, pad=1  (56x56, like ResNet layer1 entry)
    - conv2: 64->64, 3x3, stride=1, pad=1 (56x56, like ResNet layer1 block)
    - conv3: 64->128, 3x3, stride=2, pad=1 (56->28, like ResNet layer2 downsample)
    - conv4: 128->128, 3x3, stride=1, pad=1 (28x28, like ResNet layer2 block)

    Input:  [1, 3, 56, 56]  (~37 KB)
    Output: [1, 128, 28, 28] (~392 KB)
    Params: ~125K (~490 KB weights)
    """

    def __init__(self):
        super().__init__()
        # Layer 1: 3->64, stride=1 (56x56 -> 56x56)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Layer 2: 64->64, stride=1 (56x56 -> 56x56)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Layer 3: 64->128, stride=2 (56x56 -> 28x28)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)

        # Layer 4: 128->128, stride=1 (28x28 -> 28x28)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        return x


def create_conv2d_stack_model(seed: int = 42) -> tuple:
    """
    Create Conv2D stack model for DSP testing.

    4-layer sequential conv2d + batch_norm + relu stack that isolates
    the most expensive conv2d configurations from ResNet18 without
    skip connections.

    Args:
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (tvm_mod, torch_model, input_data):
        - tvm_mod: TVM IRModule with parameters bound
        - torch_model: PyTorch model for reference inference
        - input_data: numpy array for test input [1, 3, 56, 56]
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create model
    torch_model = Conv2DStackModel()
    torch_model.eval()

    # Create example input for tracing: [batch, channels, height, width]
    example_input = (torch.randn(1, 3, 56, 56),)

    # Convert to TVM
    tvm_mod = torch_to_relax_with_params(torch_model, example_input)

    # Create deterministic test input
    # Using scaled sequential values for reproducibility
    in_channels, h, w = 3, 56, 56
    total_elements = in_channels * h * w
    input_data = np.array([(i + 1) * 0.001 for i in range(total_elements)], dtype=np.float32)
    input_data = input_data.reshape(1, in_channels, h, w)

    return tvm_mod, torch_model, input_data


def create_quantized_conv2d_stack_model(seed: int = 42) -> tuple:
    """
    Create INT8 quantized Conv2D stack model for DSP testing.

    Uses PT2E static quantization with XNNPACKQuantizer to produce
    a QDQ graph with per-tensor quantization over the same 4-layer
    conv2d + batch_norm + relu stack as create_conv2d_stack_model().

    Args:
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (tvm_mod, quantized_gm, input_data):
        - tvm_mod: TVM IRModule with parameters bound
        - quantized_gm: PyTorch quantized GraphModule for reference
        - input_data: numpy array for test input [1, 3, 56, 56]
    """
    import warnings

    # torch.ao.quantization is deprecated in torch 2.10 in favor of
    # torchao, but torchao 0.16 doesn't ship XNNPACKQuantizer yet.
    # Suppress the warnings until we can migrate.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        from torch.ao.quantization.quantize_pt2e import (
            convert_pt2e,
            prepare_pt2e,
        )
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create float model
    torch_model = Conv2DStackModel()
    torch_model.eval()

    example_args = (torch.randn(1, 3, 56, 56, dtype=torch.float32),)

    # Step 1: Capture the float model (prepare_pt2e needs a GraphModule)
    with torch.no_grad():
        exported_program = export(torch_model, example_args)
    model_gm = exported_program.module()

    # Step 2: PT2E quantization
    quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
    prepared = prepare_pt2e(model_gm, quantizer)

    # Calibrate with random inputs
    with torch.no_grad():
        for _ in range(10):
            prepared(torch.randn(1, 3, 56, 56, dtype=torch.float32))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="erase_node")
        quantized_gm = convert_pt2e(prepared)

    # Step 3: Re-export the quantized model and import to TVM
    with torch.no_grad():
        exported_program_q = export(quantized_gm, example_args)
        mod = from_exported_program(exported_program_q, keep_params_as_input=True)

    # Detach and bind parameters
    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)  # pyright: ignore[reportArgumentType]

    # Create deterministic test input
    in_channels, h, w = 3, 56, 56
    total_elements = in_channels * h * w
    input_data = np.array([(i + 1) * 0.001 for i in range(total_elements)], dtype=np.float32)
    input_data = input_data.reshape(1, in_channels, h, w)

    return mod, quantized_gm, input_data
