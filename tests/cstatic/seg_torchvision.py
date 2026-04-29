#!/usr/bin/env python
"""TorchVision Semantic Segmentation Model Tester

This script provides comprehensive testing and validation for TorchVision semantic segmentation
models, with support for TVM compilation and comparison between PyTorch and TVM C Static backends.

Features:
    - Automatic discovery of all semantic segmentation models in TorchVision
    - Automatic extraction of preprocessing transforms from model weights
    - PyTorch inference (default)
    - TVM C Static compilation and inference (--tvm)
    - Side-by-side comparison of PyTorch vs TVM results (--compare)
    - Batch testing of multiple models with filtering and limits
    - Segmentation visualization with PASCAL VOC colormap and legend
    - Metrics: pixel accuracy, mean IoU, class-wise IoU
    - CSV logging for batch test results

Usage Examples:
    # Test single model with PyTorch
    python seg_torchvision.py --model fcn_resnet50

    # Test with TVM C Static compilation
    python seg_torchvision.py --model deeplabv3_resnet50 --tvm

    # Compare PyTorch vs TVM C Static
    python seg_torchvision.py --model lraspp_mobilenet_v3_large --compare

    # Test multiple models in parallel with filtering
    python seg_torchvision.py --test-all --parallel --filter deeplabv3 --max-models 5

    # Batch testing with logging
    python seg_torchvision.py --test-all --parallel --workers 8 --log-file results.csv

    # Verbose mode for detailed information
    python seg_torchvision.py --model fcn_resnet50 --verbose

    # Save visualizations for inspection
    python seg_torchvision.py --model fcn_resnet50 --compare --save-vis

    # Test with custom overlay transparency
    python seg_torchvision.py --model deeplabv3_resnet50 --compare --alpha 0.7 --save-vis

Command-Line Options:
    --model MODEL
        Name of the TorchVision segmentation model to test
        Default: fcn_resnet50
        Available: fcn_resnet50, fcn_resnet101, deeplabv3_resnet50, deeplabv3_resnet101,
                  deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large

    --image IMAGE
        Path or URL to test image
        Default: test_images/bird_0.jpg
        Examples: 'test_images/bird_0.jpg', 'https://example.com/image.jpg'

    --tvm
        Run TVM C Static compilation and inference (implies --compare)
        Default: False (PyTorch only)

    --compare
        Compare PyTorch vs TVM results side-by-side
        Default: False
        Note: Automatically sets --tvm=True

    --test-all
        Test all available segmentation models (use with --filter, --max-models)
        Default: False (test single model specified by --model)

    --filter PATTERN [PATTERN ...]
        Filter models by name patterns (case-insensitive)
        Examples: --filter fcn (tests fcn_resnet50, fcn_resnet101)
                 --filter deeplabv3 fcn (tests DeepLabV3 and FCN variants)

    --max-models N
        Maximum number of models to test when using --test-all
        Default: All available models
        Examples: --max-models 2 (test only 2 models)

    --parallel
        Run tests in parallel using multiple workers
        Default: False (sequential testing)
        Note: Only effective with --test-all

    --workers N
        Number of parallel workers for --parallel mode
        Default: Number of CPU cores on system
        Examples: --workers 4 (use 4 processes)

    --log-file PATH
        CSV file to append test results (creates file if not exists)
        Default: None (no logging)
        Columns: timestamp, model_name, success, pytorch_classes, tvm_classes,
                pixel_accuracy, mean_iou, error_message

    --save-vis
        Save segmentation visualizations as JPG files
        Default: False (no visualization saved)
        Output: segmentation_outputs/{model_name}_{backend}.jpg

    --alpha FLOAT
        Overlay transparency for segmentation visualization (0.0 to 1.0)
        Default: 0.6
        0.0 = fully transparent (original image), 1.0 = fully opaque (segmentation)

    --verbose, -v
        Enable verbose output with DEBUG level logging
        Default: False (INFO level)
        Shows: Model loading details, weight discovery, preprocessing info

    --quiet, -q
        Enable quiet mode with WARNING level logging
        Default: False (INFO level)
        Shows: Only errors and warnings

Supported Models:
    - FCN (fcn_resnet50, fcn_resnet101)
    - DeepLabV3 (deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large)
    - LRASPP (lraspp_mobilenet_v3_large)

All models:
    - Trained on PASCAL VOC dataset with 21 classes (20 objects + background)
    - Output format: OrderedDict with 'out' (main) and 'aux' (auxiliary) keys in PyTorch
    - TVM converts dict outputs to tuple: (main_output, aux_output)
    - Output resolution matches input resolution (fully convolutional architecture)
    - Input images automatically resized to 224 pixels (largest dimension) for TVM

Key Implementation Details:

1. Model Wrapping:
   - PyTorch outputs: OrderedDict with 'out' and 'aux' keys
   - TVM Relax conversion flattens dict to tuple: (main_output, aux_output)
   - Main output contains per-pixel class logits for segmentation

2. Post-Processing:
   - Apply argmax to logits to get class predictions per pixel
   - Compute pixel accuracy and mean IoU
   - No NMS needed (unlike detection)

3. Preprocessing:
   - All models use standard ImageNet normalization
   - mean = [0.485, 0.456, 0.406]
   - std = [0.229, 0.224, 0.225]
   - Models are fully convolutional - accept arbitrary input sizes

4. Visualization:
   - PASCAL VOC colormap (21 colors for 21 classes)
   - Overlay segmentation on original image with transparency
   - Display legend showing detected classes
   - Save as JPG with high quality

Expected Results:
    - All models should achieve high numerical agreement (>99% pixel accuracy)
    - Mean IoU should be high for matching outputs
    - Visualizations show semantic segmentation with proper coloring
"""

import argparse
import csv
import logging
import os
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models.segmentation as segmentation_models
import torchvision.transforms as transforms
import tvm
from PIL import Image, ImageDraw, ImageFont
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

from tvm_utils import compile_and_run_on_target, process_relax

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_IMAGE_URL = "test_images/bird_0.jpg"
DEFAULT_COCO_MEAN = [0.485, 0.456, 0.406]
DEFAULT_COCO_STD = [0.229, 0.224, 0.225]
DEFAULT_INPUT_SIZE = 512
C_STATIC_TARGET = "c_static"
LLVM_TARGET = "llvm"

# Comparison tolerances
RTOL_COMPARISON = 1e-3
ATOL_COMPARISON = 1e-5

# Thread lock for CSV file writing
_CSV_WRITE_LOCK = threading.Lock()

# PASCAL VOC 2012 class names and colors
PASCAL_VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# PASCAL VOC color palette (RGB)
PASCAL_VOC_COLORS = [
    (0, 0, 0),  # background - black
    (128, 0, 0),  # aeroplane - dark red
    (0, 128, 0),  # bicycle - dark green
    (128, 128, 0),  # bird - olive
    (0, 0, 128),  # boat - dark blue
    (128, 0, 128),  # bottle - purple
    (0, 128, 128),  # bus - teal
    (128, 128, 128),  # car - gray
    (64, 0, 0),  # cat - dark red
    (192, 0, 0),  # chair - red
    (64, 128, 0),  # cow - olive green
    (192, 128, 0),  # diningtable - orange
    (64, 0, 128),  # dog - purple
    (192, 0, 128),  # horse - magenta
    (64, 128, 128),  # motorbike - dark cyan
    (192, 128, 128),  # person - light gray
    (0, 64, 0),  # pottedplant - dark green
    (128, 64, 0),  # sheep - brown
    (0, 192, 0),  # sofa - lime
    (128, 192, 0),  # train - yellow green
    (0, 64, 128),  # tvmonitor - steel blue
]


# Data classes
@dataclass
class SegmentationResult:
    """Results from a segmentation inference"""

    segmentation_map: np.ndarray  # (H, W) with class indices
    class_logits: np.ndarray  # (C, H, W) raw logits
    classes_present: List[int]  # Unique class IDs
    pixel_counts: Dict[int, int]  # Pixels per class


@dataclass
class ModelTestResult:
    """Results from testing a single segmentation model"""

    model_name: str
    success: bool = True
    pytorch_classes: Optional[List[int]] = None
    tvm_classes: Optional[List[int]] = None
    pixel_accuracy: Optional[float] = None
    mean_iou: Optional[float] = None
    compile_time: Optional[float] = None
    inference_time: Optional[float] = None
    error_message: Optional[str] = None
    tvm_compile_success: Optional[bool] = None
    tvm_inference_success: Optional[bool] = None


@dataclass
class ComparisonResult:
    """Comparison between PyTorch and TVM segmentation"""

    model_name: str
    classes_match: bool
    pixel_accuracy: float
    mean_iou: float
    max_logit_diff: float
    error: bool = False
    error_message: Optional[str] = None


# Helper functions
def setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging based on verbosity flags.

    Sets up the logging level for the entire application based on user preferences.

    Args:
        verbose: If True, enable DEBUG level logging with detailed information
        quiet: If True, enable WARNING level logging (errors and warnings only)

    Note:
        If both verbose and quiet are False, INFO level is used (default).
        If both are True, quiet takes precedence.
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(level=level, format="%(message)s")


def get_segmentation_architecture(model_name: str) -> str:
    """Determine the segmentation architecture type from model name.

    Args:
        model_name: Name of the segmentation model (e.g., 'fcn_resnet50')

    Returns:
        Architecture type: 'fcn', 'deeplabv3', 'lraspp', or 'unknown'

    Example:
        >>> get_segmentation_architecture('fcn_resnet50')
        'fcn'
        >>> get_segmentation_architecture('deeplabv3_resnet101')
        'deeplabv3'
    """
    model_lower = model_name.lower()
    if "fcn" in model_lower:
        return "fcn"
    elif "deeplabv3" in model_lower:
        return "deeplabv3"
    elif "lraspp" in model_lower:
        return "lraspp"
    return "unknown"


def load_model_with_preprocessing(
    model_name: str = "fcn_resnet50", weight_name: Optional[str] = None
) -> Tuple[nn.Module, Callable, Optional[Any]]:
    """Load a TorchVision segmentation model with its preprocessing transforms.

    This function automatically discovers and loads the appropriate model weights
    and preprocessing transforms from TorchVision's model zoo.

    Args:
        model_name: Name of the TorchVision segmentation model (e.g., 'fcn_resnet50',
                   'deeplabv3_resnet101', 'lraspp_mobilenet_v3_large')
        weight_name: Specific weight variant to use (e.g., 'COCO_WITH_VOC_LABELS_V1'),
                    or None to use the DEFAULT weights

    Returns:
        A tuple containing:
        - model: The loaded PyTorch model in eval mode
        - preprocessing: Callable transform pipeline for image preprocessing
        - weight: The weight enum object used, or None if using defaults

    Note:
        The function automatically falls back to default weights if the specified
        weight_name cannot be found or if weight discovery fails.
    """
    logger.debug(f"Loading model: {model_name}")

    # Get the model function from segmentation module
    model_func = getattr(segmentation_models, model_name)

    weight: Optional[Any] = None

    # Try to find and use weights
    try:
        model_name_lower = model_name.lower()
        weights_enum = None

        # Search for matching weights class
        for attr_name in dir(segmentation_models):
            if attr_name.endswith("_Weights") and model_name_lower == attr_name.lower().replace(
                "_weights", ""
            ):
                weights_enum = getattr(segmentation_models, attr_name)
                logger.debug(f"  Found weights class: {attr_name}")
                break

        if weights_enum:
            # Use specific weight or default
            if weight_name:
                weight = getattr(weights_enum, weight_name)
            else:
                weight = getattr(weights_enum, "DEFAULT", list(weights_enum)[0])

            assert weight is not None
            logger.debug(f"  Weights: {weight.name}")
            model = model_func(weights=weight)
            preprocessing = get_preprocessing_from_weight(weight, model_name)
        else:
            # Fallback if weights not found
            logger.debug("  Weights class not found, using default")
            model = model_func(weights="DEFAULT")
            preprocessing = get_default_preprocessing()

    except (AttributeError, ValueError) as e:
        logger.debug(f"  Error loading weights: {e}, using default")
        model = model_func(weights="DEFAULT")
        preprocessing = get_default_preprocessing()

    # Set to evaluation mode
    model.eval()

    return model, preprocessing, weight


def get_preprocessing_from_weight(weight: Any, model_name: str) -> Callable:
    """Build preprocessing transforms with resize for efficient TVM compilation.

    Note: We always build custom preprocessing with explicit resize to 224x224
    rather than using weight.transforms(). This ensures:
    1. Manageable model size for TVM C Static compilation
    2. Reasonable compilation times
    3. Consistent input dimensions across all models

    Args:
        weight: TorchVision weight enum instance containing metadata for normalization
        model_name: Name of the model (for logging purposes)

    Returns:
        A torchvision.transforms.Compose object containing:
        - Resize to 224 (largest dimension)
        - ToTensor conversion
        - Normalization with mean and std from weight metadata or ImageNet defaults
    """
    # Always build custom preprocessing with resize for TVM (not using weight.transforms())
    # This ensures manageable model size for compilation
    transforms_list = [
        transforms.Resize(224),  # Resize to small size for fast TVM compilation
        transforms.ToTensor(),
    ]

    # Try to build from metadata
    if hasattr(weight, "meta") and weight.meta:
        meta = weight.meta
        if "mean" in meta and "std" in meta:
            transforms_list.append(transforms.Normalize(mean=meta["mean"], std=meta["std"]))
        else:
            transforms_list.append(
                transforms.Normalize(mean=DEFAULT_COCO_MEAN, std=DEFAULT_COCO_STD)
            )
    else:
        transforms_list.append(transforms.Normalize(mean=DEFAULT_COCO_MEAN, std=DEFAULT_COCO_STD))

    return transforms.Compose(transforms_list)


def get_default_preprocessing() -> transforms.Compose:
    """Get default ImageNet preprocessing transforms.

    Returns:
        Transform pipeline with ToTensor and ImageNet normalization
        (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_COCO_MEAN, std=DEFAULT_COCO_STD),
        ]
    )


def load_image(image_path_or_url: str) -> Optional[Image.Image]:
    """Load an image from a local file path or URL.

    Supports both local file paths and HTTP/HTTPS URLs. Images are automatically
    converted to RGB format for consistency.

    Args:
        image_path_or_url: Local file path (e.g., 'test_images/bird.jpg') or
                          URL (e.g., 'https://example.com/image.jpg')

    Returns:
        PIL Image in RGB format, or None if loading failed

    Example:
        >>> image = load_image('test_images/bird_0.jpg')
        >>> image = load_image('https://example.com/sample.jpg')
    """
    try:
        if image_path_or_url.startswith(("http://", "https://")):
            # Load from URL
            from io import BytesIO

            import requests

            response = requests.get(image_path_or_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            logger.debug(f"Loaded image from URL: {image_path_or_url}")
        else:
            # Load from file
            image = Image.open(image_path_or_url).convert("RGB")
            logger.debug(f"Loaded image from file: {image_path_or_url}")

        return image
    except Exception as e:
        logger.error(f"Failed to load image {image_path_or_url}: {e}")
        return None


# Segmentation processing functions
def extract_segmentation_output(result: Union[Tuple[Any, ...], Dict[str, Any], Any]) -> np.ndarray:
    """Extract main segmentation logits from model output.

    Handles the different output formats between PyTorch and TVM:
    - PyTorch: OrderedDict with 'out' (main) and 'aux' (auxiliary) keys
    - TVM: Tuple or list with (main_output, aux_output)

    Args:
        result: Model output in one of the following formats:
               - Dict with 'out' key (PyTorch)
               - Tuple/list with 2 elements (TVM)
               - Direct tensor/array (fallback)

    Returns:
        Main segmentation logits as numpy array with shape (num_classes, H, W)
        or (batch, num_classes, H, W) if batch dimension is present

    Note:
        The auxiliary output is ignored; we only extract the main segmentation output.
    """
    if isinstance(result, (tuple, list)) and len(result) >= 1:
        # TVM Relax output: (main_output, aux_output) or [main_output, aux_output]
        seg_map = result[0]
    elif isinstance(result, dict) and "out" in result:
        # PyTorch FCN/DeepLabV3 output: {'out': ..., 'aux': ...}
        seg_map = result["out"]
    else:
        seg_map = result

    # Convert to numpy if it's a tensor
    if hasattr(seg_map, "numpy"):
        return seg_map.numpy()  # type: ignore[return-value]
    elif isinstance(seg_map, np.ndarray):
        return seg_map
    else:
        # Assume it's already an array-like object
        return np.array(seg_map)


def logits_to_segmentation_mask(logits: np.ndarray) -> np.ndarray:
    """Convert segmentation logits to class predictions via argmax.

    Performs argmax operation across the class dimension to determine the predicted
    class for each pixel. Automatically removes batch dimension if present.

    Args:
        logits: Raw logits with shape (batch, num_classes, H, W) or (num_classes, H, W).
               Logits are typically raw model outputs before softmax (range: -inf to +inf).

    Returns:
        Segmentation mask with shape (H, W) where each element is a class index (0-90)
        representing the predicted PASCAL VOC class for that pixel.

    Example:
        >>> logits.shape  # (1, 21, 224, 224)
        >>> mask = logits_to_segmentation_mask(logits)
        >>> mask.shape  # (224, 224)
        >>> mask[0, 0]  # class index 0-20
    """
    if logits.ndim == 4:
        logits = logits[0]  # Remove batch dimension

    # Argmax across class dimension
    seg_mask = np.argmax(logits, axis=0)  # (H, W)
    return seg_mask


def compute_segmentation_metrics(pred_mask: np.ndarray, ref_mask: np.ndarray) -> Dict[str, float]:
    """Compute segmentation metrics

    Args:
        pred_mask: Predicted segmentation mask (H, W)
        ref_mask: Reference segmentation mask (H, W)

    Returns:
        Dictionary with 'pixel_accuracy' and 'mean_iou' keys
    """
    if pred_mask.shape != ref_mask.shape:
        return {"pixel_accuracy": 0.0, "mean_iou": 0.0}

    # Pixel accuracy
    correct = np.sum(pred_mask == ref_mask)
    total = pred_mask.size
    pixel_accuracy = float(correct) / total if total > 0 else 0.0

    # Mean IoU
    num_classes = max(pred_mask.max(), ref_mask.max()) + 1
    iou_per_class = []

    for class_id in range(num_classes):
        pred_mask_c = pred_mask == class_id
        ref_mask_c = ref_mask == class_id

        intersection = np.sum(pred_mask_c & ref_mask_c)
        union = np.sum(pred_mask_c | ref_mask_c)

        if union > 0:
            iou = intersection / union
            iou_per_class.append(iou)

    mean_iou = float(np.mean(iou_per_class)) if iou_per_class else 0.0

    return {"pixel_accuracy": pixel_accuracy, "mean_iou": mean_iou}


def compare_segmentation_results(
    pytorch_result: Union[Tuple[Any, ...], Dict[str, Any], Any],
    tvm_result: Union[Tuple[Any, ...], Dict[str, Any], Any],
    rtol: float = RTOL_COMPARISON,
    atol: float = ATOL_COMPARISON,
) -> ComparisonResult:
    """Compare PyTorch and TVM segmentation results

    Args:
        pytorch_result: PyTorch output
        tvm_result: TVM output
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        ComparisonResult object
    """
    try:
        # Extract logits
        pytorch_logits = extract_segmentation_output(pytorch_result)
        tvm_logits = extract_segmentation_output(tvm_result)

        # Convert to masks
        pytorch_mask = logits_to_segmentation_mask(pytorch_logits)
        tvm_mask = logits_to_segmentation_mask(tvm_logits)

        # Check classes
        pytorch_classes = sorted(np.unique(pytorch_mask).tolist())
        tvm_classes = sorted(np.unique(tvm_mask).tolist())
        classes_match = pytorch_classes == tvm_classes

        # Compute metrics
        metrics = compute_segmentation_metrics(tvm_mask, pytorch_mask)

        # Max logit difference
        max_logit_diff = float(np.max(np.abs(pytorch_logits - tvm_logits)))

        return ComparisonResult(
            model_name="",
            classes_match=classes_match,
            pixel_accuracy=metrics["pixel_accuracy"],
            mean_iou=metrics["mean_iou"],
            max_logit_diff=max_logit_diff,
        )

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        return ComparisonResult(
            model_name="",
            classes_match=False,
            pixel_accuracy=0.0,
            mean_iou=0.0,
            max_logit_diff=float("inf"),
            error=True,
            error_message=str(e),
        )


# Visualization functions
def create_segmentation_colormap(seg_mask: np.ndarray) -> np.ndarray:
    """Convert segmentation mask to RGB colormap

    Args:
        seg_mask: Segmentation mask (H, W) with class indices

    Returns:
        RGB colormap (H, W, 3)
    """
    height, width = seg_mask.shape
    colormap = np.zeros((height, width, 3), dtype=np.uint8)

    for class_idx in range(len(PASCAL_VOC_COLORS)):
        mask = seg_mask == class_idx
        colormap[mask] = PASCAL_VOC_COLORS[class_idx]

    return colormap


def overlay_segmentation_on_image(
    original_image: Image.Image, seg_mask: np.ndarray, alpha: float = 0.6
) -> Image.Image:
    """Overlay segmentation mask on original image

    Args:
        original_image: Original PIL image
        seg_mask: Segmentation mask (H, W)
        alpha: Transparency factor

    Returns:
        PIL Image with overlay
    """
    colormap = create_segmentation_colormap(seg_mask)
    orig_array = np.array(original_image)

    # Don't overlay background
    background_mask = seg_mask == 0
    non_bg_mask = ~background_mask

    blended = orig_array.copy()
    blended[non_bg_mask] = (
        alpha * colormap[non_bg_mask] + (1 - alpha) * orig_array[non_bg_mask]
    ).astype(np.uint8)

    return Image.fromarray(blended)


def save_segmentation_visualization(
    original_image: Image.Image,
    seg_mask: np.ndarray,
    output_path: str,
    title: str = "Segmentation Result",
) -> None:
    """Save segmentation visualization with legend

    Args:
        original_image: Original PIL image
        seg_mask: Segmentation mask
        output_path: Output file path
        title: Title for visualization
    """
    try:
        overlay_image = overlay_segmentation_on_image(original_image, seg_mask)
        unique_classes = np.unique(seg_mask)

        img_width, img_height = overlay_image.size
        legend_width = 200
        total_width = img_width + legend_width

        combined = Image.new("RGB", (total_width, img_height), color="white")
        combined.paste(overlay_image, (0, 0))

        draw = ImageDraw.Draw(combined)
        font = ImageFont.load_default()  # type: ignore

        draw.text((img_width + 10, 10), title, fill="black", font=font)

        y_offset = 40
        for class_idx in sorted(unique_classes):
            if class_idx < len(PASCAL_VOC_CLASSES):
                color = PASCAL_VOC_COLORS[class_idx]
                class_name = PASCAL_VOC_CLASSES[class_idx]

                draw.rectangle(
                    [img_width + 10, y_offset, img_width + 25, y_offset + 15],
                    fill=color,
                    outline="black",
                )
                draw.text((img_width + 30, y_offset), class_name, fill="black", font=font)
                y_offset += 20

        combined.save(output_path, "JPEG", quality=95)
        logger.info(f"Saved visualization to: {output_path}")

    except Exception as e:
        logger.warning(f"Could not save visualization: {e}")


# TVM functions
def torch_to_relax(torch_model: nn.Module, example_input: Tuple[torch.Tensor, ...]) -> tvm.IRModule:
    """Convert PyTorch model to TVM Relax IR

    Args:
        torch_model: PyTorch model
        example_input: Example input tuple

    Returns:
        TVM Relax IRModule
    """
    with torch.no_grad():
        torch_model.eval()
        exported_program = export(torch_model, example_input)  # type: ignore
        mod = from_exported_program(exported_program, keep_params_as_input=True)  # type: ignore
    return mod


def prepare_model_for_tvm(model: nn.Module, image_tensor: torch.Tensor) -> tvm.IRModule:
    """Prepare segmentation model for TVM compilation

    Args:
        model: PyTorch segmentation model
        image_tensor: Example input tensor

    Returns:
        Optimized TVM Relax IRModule
    """
    logger.debug("Converting PyTorch to Relax...")
    example_input = (image_tensor,)
    mod = torch_to_relax(model, example_input)

    logger.debug("Applying Relax optimizations...")
    mod = process_relax(mod)  # type: ignore

    return mod


def run_inference_tvm(
    relax_module: tvm.IRModule,
    image_tensor: torch.Tensor,
    target: str = C_STATIC_TARGET,
) -> Any:
    """Compile and run TVM inference

    Args:
        relax_module: TVM Relax IRModule
        image_tensor: Input tensor
        target: TVM target string

    Returns:
        Tuple or list of output arrays (main_output, aux_output)
    """
    input_data = image_tensor.numpy()
    logger.debug(f"Compiling for target: {target}")

    result = compile_and_run_on_target(  # type: ignore
        target_string=target, mod=relax_module, input=input_data
    )

    return result


# Main functions
def run_pytorch_inference(model: nn.Module, image_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Run PyTorch inference

    Args:
        model: PyTorch segmentation model
        image_tensor: Input tensor (1, 3, H, W)

    Returns:
        Dictionary with 'out' and 'aux' keys
    """
    with torch.no_grad():
        result = model(image_tensor)
    return result


def print_comparison_table(results: List[ComparisonResult]) -> None:
    """Print comparison table"""
    print("\nComparison Table: PyTorch vs TVM C Static")
    print("-" * 100)
    print(
        f"{'Model':<30} {'Classes Match':<15} {'Pixel Acc':<15} {'Mean IoU':<15} {'Max Logit Diff':<15}"
    )
    print("-" * 100)

    for result in results:
        classes_str = "✓" if result.classes_match else "✗"
        print(
            f"{result.model_name:<30} {classes_str:<15} {result.pixel_accuracy:<15.3f} "
            f"{result.mean_iou:<15.3f} {result.max_logit_diff:<15.6f}"
        )

    print("-" * 100)


def get_all_segmentation_models() -> List[str]:
    """Get all available segmentation models in TorchVision

    Returns:
        List of model names
    """
    models = []
    for attr_name in dir(segmentation_models):
        if not attr_name.startswith("_"):
            attr = getattr(segmentation_models, attr_name)
            # Check if it's a callable model function
            if callable(attr) and attr_name[0].islower():
                # Exclude utility functions
                if not any(
                    exclude in attr_name.lower()
                    for exclude in ["resnet", "mobilenet", "weights", "deeplabv3_mobilenet"]
                    if exclude != attr_name
                ):
                    # Try to instantiate to verify it's a model
                    try:
                        # Check if it has a weights enum
                        weights_class_name = f"{attr_name.upper()}_Weights"
                        if hasattr(segmentation_models, weights_class_name):
                            models.append(attr_name)
                    except Exception:
                        pass

    # Manually add known models to ensure coverage
    known_models = [
        "fcn_resnet50",
        "fcn_resnet101",
        "deeplabv3_resnet50",
        "deeplabv3_resnet101",
        "deeplabv3_mobilenet_v3_large",
        "lraspp_mobilenet_v3_large",
    ]

    for model in known_models:
        if model not in models:
            models.append(model)

    return sorted(models)


def test_single_model(model_name: str, image_path: str, compare: bool = False) -> ModelTestResult:
    """Test a single segmentation model

    Args:
        model_name: Name of model to test
        image_path: Path to test image
        compare: Whether to run comparison

    Returns:
        ModelTestResult object
    """
    result = ModelTestResult(model_name=model_name)

    try:
        # Load model
        model, preprocessing, weight = load_model_with_preprocessing(model_name)

        # Load image
        image = load_image(image_path)
        if image is None:
            result.success = False
            result.error_message = "Failed to load image"
            return result

        # Preprocess
        image_tensor = preprocessing(image).unsqueeze(0)

        # PyTorch inference
        pytorch_result = run_pytorch_inference(model, image_tensor)
        pytorch_logits = extract_segmentation_output(pytorch_result)
        pytorch_mask = logits_to_segmentation_mask(pytorch_logits)
        result.pytorch_classes = sorted(np.unique(pytorch_mask).tolist())

        if compare:
            # TVM inference
            relax_mod = prepare_model_for_tvm(model, image_tensor)
            tvm_result = run_inference_tvm(relax_mod, image_tensor, C_STATIC_TARGET)
            result.tvm_inference_success = True
            result.tvm_compile_success = True

            tvm_logits = extract_segmentation_output(tvm_result)
            tvm_mask = logits_to_segmentation_mask(tvm_logits)
            result.tvm_classes = sorted(np.unique(tvm_mask).tolist())

            # Compare
            comparison = compare_segmentation_results(pytorch_result, tvm_result)
            result.pixel_accuracy = comparison.pixel_accuracy
            result.mean_iou = comparison.mean_iou

    except Exception as e:
        result.success = False
        result.error_message = str(e)
        logger.error(f"Error testing {model_name}: {e}")

    return result


def test_multiple_models(args: argparse.Namespace) -> None:
    """Test multiple segmentation models

    Args:
        args: Parsed command-line arguments
    """
    all_models = get_all_segmentation_models()

    # Filter models
    if args.filter:
        filtered = []
        for model in all_models:
            if any(pattern.lower() in model.lower() for pattern in args.filter):
                filtered.append(model)
        all_models = filtered

    # Limit models
    if args.max_models:
        all_models = all_models[: args.max_models]

    logger.info(f"Testing {len(all_models)} segmentation models")

    if args.parallel:
        test_multiple_models_parallel(args, all_models)
    else:
        test_multiple_models_sequential(args, all_models)


def test_multiple_models_sequential(args: argparse.Namespace, models: List[str]) -> None:
    """Test models sequentially

    Args:
        args: Parsed command-line arguments
        models: List of model names to test
    """
    results = []

    for i, model_name in enumerate(models, 1):
        logger.info(f"[{i}/{len(models)}] Testing {model_name}...")

        result = test_single_model(model_name, args.image, compare=args.compare)
        results.append(result)

        if result.success:
            status = "✓"
        else:
            status = "✗"

        logger.info(f"  {status} {model_name}")

        if args.log_file:
            append_result_to_log(args.log_file, result)


def test_multiple_models_parallel(args: argparse.Namespace, models: List[str]) -> None:
    """Test models in parallel

    Args:
        args: Parsed command-line arguments
        models: List of model names to test
    """
    results = []
    num_workers = args.workers or os.cpu_count() or 4

    logger.info(f"Testing {len(models)} models in parallel with {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(test_single_model, model, args.image, args.compare): model
            for model in models
        }

        for i, future in enumerate(as_completed(futures), 1):
            model_name = futures[future]
            try:
                result = future.result()
                results.append(result)

                status = "✓" if result.success else "✗"
                logger.info(f"[{i}/{len(models)}] {status} {model_name}")

                if args.log_file:
                    append_result_to_log(args.log_file, result)

            except Exception as e:
                logger.error(f"[{i}/{len(models)}] ✗ {model_name} - {e}")

    # Print summary
    successful = sum(1 for r in results if r.success)
    logger.info(f"\nSummary: {successful}/{len(results)} models tested successfully")


def append_result_to_log(log_file: str, result: ModelTestResult) -> None:
    """Append test result to CSV log

    Args:
        log_file: Path to CSV log file
        result: ModelTestResult object
    """
    with _CSV_WRITE_LOCK:
        try:
            file_path = Path(log_file)
            file_exists = file_path.exists()

            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if not file_exists:
                    header = [
                        "timestamp",
                        "model_name",
                        "success",
                        "pytorch_classes",
                        "tvm_classes",
                        "pixel_accuracy",
                        "mean_iou",
                        "error_message",
                    ]
                    writer.writerow(header)

                row = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    result.model_name,
                    "Yes" if result.success else "No",
                    str(result.pytorch_classes) if result.pytorch_classes else "N/A",
                    str(result.tvm_classes) if result.tvm_classes else "N/A",
                    f"{result.pixel_accuracy:.4f}" if result.pixel_accuracy else "N/A",
                    f"{result.mean_iou:.4f}" if result.mean_iou else "N/A",
                    result.error_message or "",
                ]
                writer.writerow(row)

        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")


def main():
    """Main function for single model testing"""
    parser = argparse.ArgumentParser(description="TorchVision Semantic Segmentation Model Tester")
    parser.add_argument(
        "--model", default="fcn_resnet50", help="Model name to test (default: fcn_resnet50)"
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE_URL, help="Path or URL to test image")
    parser.add_argument("--tvm", action="store_true", help="Run TVM compilation and inference")
    parser.add_argument(
        "--compare", action="store_true", help="Compare PyTorch vs TVM (implies --tvm)"
    )
    parser.add_argument(
        "--test-all", action="store_true", help="Test all available segmentation models"
    )
    parser.add_argument(
        "--filter",
        nargs="+",
        help="Filter models by name patterns (e.g., --filter fcn deeplabv3)",
    )
    parser.add_argument("--max-models", type=int, help="Maximum number of models to test")
    parser.add_argument(
        "--parallel", action="store_true", help="Run tests in parallel (only with --test-all)"
    )
    parser.add_argument("--workers", type=int, help="Number of parallel workers")
    parser.add_argument("--log-file", help="CSV log file for results")
    parser.add_argument("--save-vis", action="store_true", help="Save segmentation visualizations")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay transparency (0-1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    setup_logging(args.verbose, args.quiet)

    if args.compare:
        args.tvm = True

    if args.test_all:
        test_multiple_models(args)
    else:
        # Single model test
        logger.info(f"Loading model: {args.model}")
        model, preprocessing, weight = load_model_with_preprocessing(args.model)

        # Load image
        image = load_image(args.image)
        if image is None:
            logger.error("Failed to load image")
            return

        # Preprocess
        image_tensor = preprocessing(image).unsqueeze(0)
        logger.info(f"Image tensor shape: {image_tensor.shape}")

        # PyTorch inference
        logger.info("Running PyTorch inference...")
        pytorch_result = run_pytorch_inference(model, image_tensor)
        pytorch_logits = extract_segmentation_output(pytorch_result)
        pytorch_mask = logits_to_segmentation_mask(pytorch_logits)
        pytorch_classes = sorted(np.unique(pytorch_mask).tolist())

        logger.info(f"PyTorch - Classes: {pytorch_classes}")
        logger.info(f"PyTorch - Segmentation shape: {pytorch_mask.shape}")

        # TVM inference
        tvm_mask: Optional[np.ndarray] = None
        if args.tvm:
            logger.info("Preparing model for TVM...")
            relax_mod = prepare_model_for_tvm(model, image_tensor)

            logger.info("Running TVM inference...")
            tvm_result = run_inference_tvm(relax_mod, image_tensor, C_STATIC_TARGET)
            tvm_logits = extract_segmentation_output(tvm_result)
            tvm_mask = logits_to_segmentation_mask(tvm_logits)
            tvm_classes = sorted(np.unique(tvm_mask).tolist())

            logger.info(f"TVM - Classes: {tvm_classes}")
            logger.info(f"TVM - Segmentation shape: {tvm_mask.shape}")

            if args.compare:
                logger.info("Comparing results...")
                comparison = compare_segmentation_results(pytorch_result, tvm_result)
                logger.info(f"Classes match: {comparison.classes_match}")
                logger.info(f"Pixel accuracy: {comparison.pixel_accuracy:.4f}")
                logger.info(f"Mean IoU: {comparison.mean_iou:.4f}")
                logger.info(f"Max logit diff: {comparison.max_logit_diff:.6f}")

        # Visualizations
        if args.save_vis:
            output_dir = os.path.join(os.path.dirname(__file__), "segmentation_outputs")
            os.makedirs(output_dir, exist_ok=True)

            output_path = os.path.join(output_dir, f"{args.model}_pytorch.jpg")
            save_segmentation_visualization(
                image, pytorch_mask, output_path, f"PyTorch - {args.model}"
            )

            if args.tvm and tvm_mask is not None:
                output_path = os.path.join(output_dir, f"{args.model}_tvm.jpg")
                save_segmentation_visualization(image, tvm_mask, output_path, f"TVM - {args.model}")


if __name__ == "__main__":
    main()
