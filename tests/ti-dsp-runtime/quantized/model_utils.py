"""
Quantized model creation utilities for DSP tests.

Each function creates a quantized TorchVision model using PT2E static
quantization (C7xMMAQuantizer, symmetric INT8, QDQ graph) and returns
a tuple of (tvm_mod, quantized_gm, input_data).
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from tvm import relax
from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program

# The yolov5 torch.hub code checks its requirements through ultralytics, which
# by default pip-installs anything it finds missing: it upgraded the pinned
# ultralytics==8.4.14 (see setup_test_env.sh) mid-run, breaking every YOLO
# model loaded after it. ultralytics reads this once, at first import.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

_THIS_DIR = Path(__file__).parent
_TEST_IMAGES_DIR = _THIS_DIR.parent.parent / "cstatic" / "test_images"
_CALIB_IMAGES_DIR = _THIS_DIR / "calib_images"
# Input for the classifier tests. It is not in calib_images/, so the DSP is
# checked on an image the activation ranges were not fitted to.
_EVAL_IMAGE = _TEST_IMAGES_DIR / "dog.jpg"


def _pt2e_quantize(torch_model, example_args, calibration_data=None, strict=True):
    """Shared PT2E quantization pipeline.

    calibration_data: optional iterable of calibration tensors. Defaults to
    random noise, whose activation ranges don't match real inputs; every
    model in this file passes real images instead.
    strict: passed through to torch.export.export. Models with dynamic
    control flow (e.g. raw ultralytics/yolov5 modules) require False.

    Returns (tvm_mod, quantized_gm) with parameters bound.
    """
    with torch.no_grad():
        exported_program = export(torch_model, example_args, strict=strict)
    model_gm = exported_program.module()

    quantizer = C7xMMAQuantizer(dtype="int8", symmetric_activations=True)
    prepared = prepare_pt2e(model_gm, quantizer)

    with torch.no_grad():
        if calibration_data is not None:
            for sample in calibration_data:
                prepared(sample)
        else:
            for _ in range(10):
                prepared(torch.randn_like(example_args[0]))

    quantized_gm = convert_pt2e(prepared)

    with torch.no_grad():
        exported_program_q = export(quantized_gm, example_args, strict=strict)
        mod = from_exported_program(exported_program_q, keep_params_as_input=True)

    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(func_name="main", params=func_params_dict)(mod)

    return mod, quantized_gm


def _quantize_classifier(build_model, weights) -> tuple:
    """Quantize a pretrained TorchVision classifier, calibrated on real images.

    Calibrates on every image in calib_images/ and returns _EVAL_IMAGE as
    the input, both preprocessed with the weights' own transform (resize,
    center crop, ImageNet normalization) so the recorded activation ranges
    match what the model sees in use. Returns (tvm_mod, quantized_gm,
    input_data).
    """
    from PIL import Image  # noqa: PLC0415

    transform = weights.transforms()

    def load(path):
        return transform(Image.open(path).convert("RGB")).unsqueeze(0)

    calibration_data = [load(p) for p in sorted(_CALIB_IMAGES_DIR.glob("*.jpg"))]
    if not calibration_data:
        raise FileNotFoundError(f"no calibration images in {_CALIB_IMAGES_DIR}")
    input_tensor = load(_EVAL_IMAGE)

    torch_model = build_model(weights=weights).eval()
    mod, quantized_gm = _pt2e_quantize(
        torch_model, (input_tensor,), calibration_data=calibration_data
    )
    return mod, quantized_gm, input_tensor.numpy()


def create_quantized_resnet_model() -> tuple:
    """Create INT8 quantized ResNet-18 model. Input: [1, 3, 224, 224]."""
    from torchvision.models.resnet import ResNet18_Weights, resnet18

    return _quantize_classifier(resnet18, ResNet18_Weights.DEFAULT)


def create_quantized_googlenet_model() -> tuple:
    """Create INT8 quantized GoogLeNet model. Input: [1, 3, 224, 224]."""
    from torchvision.models.googlenet import GoogLeNet_Weights, googlenet

    return _quantize_classifier(googlenet, GoogLeNet_Weights.DEFAULT)


def create_quantized_inception_v3_model() -> tuple:
    """Create INT8 quantized InceptionV3 model. Input: [1, 3, 299, 299]."""
    from torchvision.models.inception import Inception_V3_Weights, inception_v3

    return _quantize_classifier(inception_v3, Inception_V3_Weights.DEFAULT)


def create_quantized_mobilenet_v2_model() -> tuple:
    """Create INT8 quantized MobileNetV2 model. Input: [1, 3, 224, 224]."""
    from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, mobilenet_v2

    return _quantize_classifier(mobilenet_v2, MobileNet_V2_Weights.DEFAULT)


def create_quantized_mobilenet_v3_model() -> tuple:
    """Create INT8 quantized MobileNetV3 Large model. Input: [1, 3, 224, 224]."""
    from torchvision.models.mobilenetv3 import (
        MobileNet_V3_Large_Weights,
        mobilenet_v3_large,
    )

    return _quantize_classifier(mobilenet_v3_large, MobileNet_V3_Large_Weights.DEFAULT)


def create_quantized_resnext101_model() -> tuple:
    """Create INT8 quantized ResNeXt-101 (32x8D) model. Input: [1, 3, 224, 224]."""
    from torchvision.models.resnet import ResNeXt101_32X8D_Weights, resnext101_32x8d

    return _quantize_classifier(resnext101_32x8d, ResNeXt101_32X8D_Weights.DEFAULT)


def create_quantized_shufflenet_v2_model() -> tuple:
    """Create INT8 quantized ShuffleNetV2 (x0.5) model. Input: [1, 3, 224, 224]."""
    from torchvision.models.shufflenetv2 import (
        ShuffleNet_V2_X0_5_Weights,
        shufflenet_v2_x0_5,
    )

    return _quantize_classifier(shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights.DEFAULT)


# -----------------------------------------------------------------------------
# YOLO object detection (MMALIB, not TIDL)
# -----------------------------------------------------------------------------

YOLO_INPUT_SHAPE = (1, 3, 320, 320)


class YOLOWrapper(nn.Module):
    """Extract the core YOLO model for torch.export compatibility."""

    def __init__(self, yolo_model, version: str = "v5"):
        super().__init__()
        self.version = version
        if hasattr(yolo_model, "model"):
            self.model = yolo_model.model
        else:
            self.model = yolo_model
        # yolo26 ("v26") runs its default NMS-free "one2one" deploy head
        # (end2end=True), returning already-selected [1, 300, 6] detections
        # (x1,y1,x2,y2,confidence,class_idx) — its actual production
        # inference path, not the auxiliary "one2many" training branch.
        # This requires three fixes elsewhere: _div/_rsub dtype handling and
        # _index_tensor's ndim computation in base_fx_graph_translator.py
        # (the postprocess's `ori_index[arange, index // nc]` needs both),
        # and C7xMMAQuantizer's _TRANSPARENT_OPS skipping any flatten that
        # feeds a topk (see c7x_mma_quantizer.py) — without it, the
        # score-ranking flatten->topk gets quantized with a scale calibrated
        # for tiny sigmoid probabilities, corrupting which detections topk
        # selects.
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        if isinstance(output, (list, tuple)):
            return output[0]
        return output


def _load_yolov5(model_name: str):
    """Load YOLOv5 model from local .pt file (preferred) or torch.hub."""
    # Prefer cached hub repo (source="local") to avoid network access
    # behind corporate proxies.  Falls back to github download.
    hub_dir = Path(torch.hub.get_dir()) / "ultralytics_yolov5_master"
    local_pt = _THIS_DIR.parent / f"{model_name}.pt"
    # Weights cached by torch.hub at TORCH_HOME/hub/checkpoints/
    cached_pt = Path(torch.hub.get_dir()) / "checkpoints" / f"{model_name}.pt"
    hub_cached = hub_dir.is_dir()
    # Find a .pt file: local dir > hub checkpoints cache
    pt_file = local_pt if local_pt.exists() else (cached_pt if cached_pt.exists() else None)
    if pt_file and hub_cached:
        model = torch.hub.load(
            str(hub_dir),
            "custom",
            path=str(pt_file),
            source="local",
            verbose=False,
        )
    elif pt_file:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=str(pt_file), verbose=False)
    else:
        import pytest  # noqa: PLC0415

        pytest.skip(f"{model_name} requires pre-cached weights at {cached_pt} or {local_pt}")
    model.eval()
    return model


def _load_yolov8(model_name: str):
    """Load a YOLOv8/YOLO26 model via the ultralytics package.

    The ultralytics.YOLO() constructor loads any model architecture the
    installed ultralytics package recognizes by checkpoint name, so this
    loader is not v8-specific — it's reused as-is for yolo26.
    """
    from ultralytics import YOLO  # noqa: PLC0415

    model = YOLO(f"{model_name}.pt")
    model.model.eval()
    return model


def _load_yolo_calibration_frames(size: int = 320) -> list:
    """Load real test images for PT2E calibration, resized to (size, size).

    Random-noise calibration (used by the other models in this file) causes
    poorly-scaled activations at the YOLO DFL head's softmax, which can
    underflow to a NaN (the same failure documented for TIDL calibration in
    tidl-tests/test_yolo_dsp.py). Real images give much better INT8 scale
    estimates.  Returns a list of float32 [1, 3, size, size] tensors in
    [0, 1]; falls back to a single random frame if no images are found.
    """
    from PIL import Image  # noqa: PLC0415

    frames = []
    for p in sorted(_TEST_IMAGES_DIR.glob("*.jpg")):
        img = Image.open(p).convert("RGB").resize((size, size))
        arr = np.array(img).astype(np.float32) / 255.0  # HWC [0,1]
        chw = arr.transpose(2, 0, 1)  # CHW
        frames.append(torch.from_numpy(chw).unsqueeze(0))
    if not frames:
        frames.append(torch.rand(1, 3, size, size, dtype=torch.float32))
    return frames


def create_quantized_yolo_model(model_name: str, version: str, seed: int = 42) -> tuple:
    """Create INT8 quantized YOLO model. Input: [1, 3, 320, 320].

    model_name: e.g. "yolov5n", "yolov5s", "yolov8n", "yolov8s", "yolo26n".
    version: "v5" (torch.hub loader) or "v8"/"v26" (ultralytics package
    loader — identical loading code). "v26" runs yolo26's real one2one
    deploy head; see YOLOWrapper's docstring for what that requires.
    """
    raw_model = _load_yolov5(model_name) if version == "v5" else _load_yolov8(model_name)
    wrapped = YOLOWrapper(raw_model, version=version).eval()

    example_args = (torch.randn(*YOLO_INPUT_SHAPE, dtype=torch.float32),)
    calibration_data = _load_yolo_calibration_frames(size=YOLO_INPUT_SHAPE[-1])
    mod, quantized_gm = _pt2e_quantize(
        wrapped, example_args, calibration_data=calibration_data, strict=False
    )

    np.random.seed(seed)
    input_data = np.random.rand(*YOLO_INPUT_SHAPE).astype(np.float32)
    return mod, quantized_gm, input_data
