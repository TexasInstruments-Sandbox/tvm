"""
Quantized model creation utilities for DSP tests.

Each function creates a quantized TorchVision model using PT2E static
quantization (C7xMMAQuantizer, symmetric INT8, QDQ graph) and returns
a tuple of (tvm_mod, quantized_gm, input_data).
"""

import numpy as np
import torch
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from tvm import relax
from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program


def _pt2e_quantize(torch_model, example_args):
    """Shared PT2E quantization pipeline.

    Returns (tvm_mod, quantized_gm) with parameters bound.
    """
    with torch.no_grad():
        exported_program = export(torch_model, example_args)
    model_gm = exported_program.module()

    quantizer = C7xMMAQuantizer(dtype="int8", symmetric_activations=True)
    prepared = prepare_pt2e(model_gm, quantizer)

    with torch.no_grad():
        for _ in range(10):
            prepared(torch.randn_like(example_args[0]))

    quantized_gm = convert_pt2e(prepared)

    with torch.no_grad():
        exported_program_q = export(quantized_gm, example_args)
        mod = from_exported_program(exported_program_q, keep_params_as_input=True)

    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)

    return mod, quantized_gm


def create_quantized_resnet_model(seed: int = 42) -> tuple:
    """Create INT8 quantized ResNet-18 model. Input: [1, 3, 224, 224]."""
    from torchvision.models.resnet import ResNet18_Weights, resnet18

    np.random.seed(seed)
    torch_model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
    mod, quantized_gm = _pt2e_quantize(torch_model, example_args)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    return mod, quantized_gm, input_data


def create_quantized_googlenet_model(seed: int = 42) -> tuple:
    """Create INT8 quantized GoogLeNet model. Input: [1, 3, 224, 224]."""
    from torchvision.models.googlenet import GoogLeNet_Weights, googlenet

    np.random.seed(seed)
    torch_model = googlenet(weights=GoogLeNet_Weights.DEFAULT).eval()
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
    mod, quantized_gm = _pt2e_quantize(torch_model, example_args)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    return mod, quantized_gm, input_data


def create_quantized_inception_v3_model(seed: int = 42) -> tuple:
    """Create INT8 quantized InceptionV3 model. Input: [1, 3, 299, 299]."""
    from torchvision.models.inception import Inception_V3_Weights, inception_v3

    np.random.seed(seed)
    torch_model = inception_v3(weights=Inception_V3_Weights.DEFAULT).eval()
    example_args = (torch.randn(1, 3, 299, 299, dtype=torch.float32),)
    mod, quantized_gm = _pt2e_quantize(torch_model, example_args)
    input_data = np.random.rand(1, 3, 299, 299).astype(np.float32)
    return mod, quantized_gm, input_data


def create_quantized_mobilenet_v2_model(seed: int = 42) -> tuple:
    """Create INT8 quantized MobileNetV2 model. Input: [1, 3, 224, 224]."""
    from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, mobilenet_v2

    np.random.seed(seed)
    torch_model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval()
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
    mod, quantized_gm = _pt2e_quantize(torch_model, example_args)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    return mod, quantized_gm, input_data


def create_quantized_mobilenet_v3_model(seed: int = 42) -> tuple:
    """Create INT8 quantized MobileNetV3 Large model. Input: [1, 3, 224, 224]."""
    from torchvision.models.mobilenetv3 import (
        MobileNet_V3_Large_Weights,
        mobilenet_v3_large,
    )

    np.random.seed(seed)
    torch_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).eval()
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
    mod, quantized_gm = _pt2e_quantize(torch_model, example_args)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    return mod, quantized_gm, input_data


def create_quantized_resnext101_model(seed: int = 42) -> tuple:
    """Create INT8 quantized ResNeXt-101 (32x8D) model. Input: [1, 3, 224, 224]."""
    from torchvision.models.resnet import ResNeXt101_32X8D_Weights, resnext101_32x8d

    np.random.seed(seed)
    torch_model = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.DEFAULT).eval()
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
    mod, quantized_gm = _pt2e_quantize(torch_model, example_args)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    return mod, quantized_gm, input_data


def create_quantized_shufflenet_v2_model(seed: int = 42) -> tuple:
    """Create INT8 quantized ShuffleNetV2 (x0.5) model. Input: [1, 3, 224, 224]."""
    from torchvision.models.shufflenetv2 import (
        ShuffleNet_V2_X0_5_Weights,
        shufflenet_v2_x0_5,
    )

    np.random.seed(seed)
    torch_model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT).eval()
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
    mod, quantized_gm = _pt2e_quantize(torch_model, example_args)
    input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    return mod, quantized_gm, input_data
