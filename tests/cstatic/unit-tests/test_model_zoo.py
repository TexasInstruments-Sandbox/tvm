"""
Model zoo tests for the c_static backend.

Parametrized tests covering 111 models across four categories:
- TorchVision Classification (80 models)
- TorchVision Object Detection (5 single-stage models)
- TorchVision Semantic Segmentation (6 models)
- YOLO Object Detection (20 models: v5, v8, v11)

Each test compiles for both LLVM (reference) and c_static, then asserts
numerical agreement (rtol=1e-3, atol=1e-5).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.export import export

import tvm
from tvm.relax.frontend.torch import from_exported_program
from tvm_utils import compile_and_run_on_target, process_relax

# Add parent dir so we can import from standalone scripts
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================================
# Model Lists
# ============================================================================

CLASSIFICATION_MODELS = [
    "alexnet",
    "convnext_base", "convnext_large", "convnext_small", "convnext_tiny",
    "densenet121", "densenet161", "densenet169", "densenet201",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "efficientnet_v2_l", "efficientnet_v2_m", "efficientnet_v2_s",
    "googlenet", "inception_v3",
    "maxvit_t",
    "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3",
    "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
    "regnet_x_16gf", "regnet_x_1_6gf", "regnet_x_32gf", "regnet_x_3_2gf",
    "regnet_x_400mf", "regnet_x_800mf", "regnet_x_8gf",
    "regnet_y_128gf", "regnet_y_16gf", "regnet_y_1_6gf", "regnet_y_32gf",
    "regnet_y_3_2gf", "regnet_y_400mf", "regnet_y_800mf", "regnet_y_8gf",
    "resnet101", "resnet152", "resnet18", "resnet34", "resnet50",
    "resnext101_32x8d", "resnext101_64x4d", "resnext50_32x4d",
    "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
    "squeezenet1_0", "squeezenet1_1",
    "swin_b", "swin_s", "swin_t", "swin_v2_b", "swin_v2_s", "swin_v2_t",
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
    "vit_b_16", "vit_b_32", "vit_h_14", "vit_l_16", "vit_l_32",
    "wide_resnet101_2", "wide_resnet50_2",
]

DETECTION_MODELS = [
    "fcos_resnet50_fpn",
    "retinanet_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "ssd300_vgg16",
    "ssdlite320_mobilenet_v3_large",
]

SEGMENTATION_MODELS = [
    "fcn_resnet50",
    "fcn_resnet101",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenet_v3_large",
    "lraspp_mobilenet_v3_large",
]

YOLO_MODELS = [
    "yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x",
    "yolov5n6", "yolov5s6", "yolov5m6", "yolov5l6", "yolov5x6",
    "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
    "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
]

# ============================================================================
# Helpers
# ============================================================================


def _torch_to_relax(model, example_input):
    with torch.no_grad():
        ep = export(model, example_input, strict=False)
        mod = from_exported_program(ep, keep_params_as_input=True)
    return mod


def _to_list(result):
    """Normalize output to a list of arrays for uniform comparison."""
    if isinstance(result, list):
        return result
    return [result]


def _compare_outputs(llvm_result, cstatic_result, model_name):
    a_list = _to_list(llvm_result)
    b_list = _to_list(cstatic_result)
    assert len(a_list) == len(b_list), (
        f"{model_name}: output count mismatch ({len(a_list)} vs {len(b_list)})"
    )
    for i, (a, b) in enumerate(zip(a_list, b_list)):
        assert np.allclose(a, b, rtol=1e-3, atol=1e-5), (
            f"{model_name} output[{i}]: max diff {np.max(np.abs(a - b))}"
        )


# ============================================================================
# Classification Tests
# ============================================================================


@pytest.mark.model_zoo
@pytest.mark.parametrize("model_name", CLASSIFICATION_MODELS)
def test_classification(model_name):
    """Test TorchVision classification model: LLVM vs c_static."""
    import torchvision.models as models

    torch_model = getattr(models, model_name)(weights="DEFAULT").eval()

    input_shape = (1, 3, 224, 224)
    if model_name == "inception_v3":
        input_shape = (1, 3, 299, 299)
    elif model_name == "vit_h_14":
        input_shape = (1, 3, 518, 518)

    example_input = (torch.randn(*input_shape, dtype=torch.float32),)
    mod = _torch_to_relax(torch_model, example_input)
    mod = process_relax(mod)

    input_data = np.random.rand(*input_shape).astype(np.float32)
    llvm_result = compile_and_run_on_target("llvm", mod, input_data)
    cstatic_result = compile_and_run_on_target("c_static", mod, input_data)
    _compare_outputs(llvm_result, cstatic_result, model_name)


# ============================================================================
# Detection Tests
# ============================================================================


@pytest.mark.model_zoo
@pytest.mark.parametrize("model_name", DETECTION_MODELS)
def test_detection(model_name):
    """Test TorchVision single-stage detection model: LLVM vs c_static."""
    from od_torchvision import prepare_model_for_tvm
    import torchvision.models.detection as det_models

    torch_model = getattr(det_models, model_name)(weights="DEFAULT").eval()

    input_shape = (1, 3, 320, 320)
    if "ssd300" in model_name:
        input_shape = (1, 3, 300, 300)

    result = prepare_model_for_tvm(torch_model, model_name, input_shape=input_shape)
    assert result is not None, f"{model_name}: prepare_model_for_tvm returned None"
    mod, _components = result

    input_data = np.random.rand(*input_shape).astype(np.float32)
    llvm_result = compile_and_run_on_target("llvm", mod, input_data)
    cstatic_result = compile_and_run_on_target("c_static", mod, input_data)
    _compare_outputs(llvm_result, cstatic_result, model_name)


# ============================================================================
# Segmentation Tests
# ============================================================================


@pytest.mark.model_zoo
@pytest.mark.parametrize("model_name", SEGMENTATION_MODELS)
def test_segmentation(model_name):
    """Test TorchVision segmentation model: LLVM vs c_static."""
    import torchvision.models.segmentation as seg_models

    torch_model = getattr(seg_models, model_name)(weights="DEFAULT").eval()

    input_shape = (1, 3, 224, 224)
    example_input = (torch.randn(*input_shape, dtype=torch.float32),)
    mod = _torch_to_relax(torch_model, example_input)
    mod = process_relax(mod)

    input_data = np.random.rand(*input_shape).astype(np.float32)
    llvm_result = compile_and_run_on_target("llvm", mod, input_data)
    cstatic_result = compile_and_run_on_target("c_static", mod, input_data)
    _compare_outputs(llvm_result, cstatic_result, model_name)


# ============================================================================
# YOLO Tests
# ============================================================================


@pytest.mark.model_zoo
@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_yolo(model_name):
    """Test YOLO detection model: LLVM vs c_static."""
    from od_yolo import load_yolo_model, prepare_model_for_tvm, detect_yolo_version

    version = detect_yolo_version(model_name)
    model, _ = load_yolo_model(model_name)

    input_shape = (1, 3, 640, 640)
    if model_name.endswith("6"):
        input_shape = (1, 3, 1280, 1280)

    mod = prepare_model_for_tvm(model, input_shape=input_shape, version=version)

    input_data = np.random.rand(*input_shape).astype(np.float32)
    llvm_result = compile_and_run_on_target("llvm", mod, input_data)
    cstatic_result = compile_and_run_on_target("c_static", mod, input_data)
    _compare_outputs(llvm_result, cstatic_result, model_name)
