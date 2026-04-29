#!/usr/bin/env python
"""TorchVision Object Detection Model Tester

This script provides comprehensive testing and validation for TorchVision object detection models,
with support for TVM compilation and comparison between PyTorch and TVM C Static backends.

Features:
    - Automatic discovery of all COCO object detection models in TorchVision
    - Automatic extraction of preprocessing transforms from model weights
    - PyTorch inference (default)
    - TVM C Static compilation and inference (--tvm)
    - Side-by-side comparison of PyTorch vs TVM results (--compare)
    - Batch testing of multiple models with filtering and limits
    - Configuration file support for reusable test setups
    - Comprehensive logging with adjustable verbosity levels
    - Bounding box visualization and detection metrics

Usage Examples:
    # Test single model with PyTorch
    python od_torchvision.py --model fasterrcnn_resnet50_fpn

    # Test with TVM C Static compilation
    python od_torchvision.py --model fasterrcnn_mobilenet_v3_large_fpn --tvm

    # Compare PyTorch vs TVM C Static
    python od_torchvision.py --model retinanet_resnet50_fpn --compare

    # Test multiple models with filtering
    python od_torchvision.py --test-all --filter fasterrcnn retinanet --max-models 5

    # Test multiple models in parallel (much faster)
    python od_torchvision.py --test-all --parallel

    # Parallel testing with custom worker count and logging
    python od_torchvision.py --test-all --parallel --workers 8 --log-file results.csv

    # Parallel comparison with filtering and logging
    python od_torchvision.py --test-all --compare --parallel --filter fasterrcnn --log-file results.csv

    # Verbose mode shows detailed information
    python od_torchvision.py --model fasterrcnn_resnet50_fpn --verbose

    # Quiet mode suppresses most output
    python od_torchvision.py --model fasterrcnn_resnet50_fpn --quiet

    # Use configuration file
    python od_torchvision.py --config my_config.yaml

Command-Line Options:
    --model MODEL              Model name to test (default: fasterrcnn_resnet50_fpn)
    --weight WEIGHT            Weight name to use (default: None, uses DEFAULT)
    --image IMAGE              Path or URL to test image
    --test-all                 Test all available object detection models
    --max-models N             Maximum number of models to test with --test-all
    --filter PATTERN [...]     Filter models by name patterns
    --parallel                 Run tests in parallel (only with --test-all)
    --workers N                Number of parallel workers (default: CPU count)
    --log-file PATH            CSV log file for appending results (with --test-all)
    --tvm                      Use TVM compilation with C Static target
    --compare                  Compare PyTorch vs TVM results (implies --tvm)
    --score-threshold FLOAT    Minimum confidence score for detections (default: 0.5)
    --verbose, -v              Enable verbose output with detailed logging
    --quiet, -q                Enable quiet mode (minimal output)
    --config FILE              Load options from YAML configuration file

Logging Levels:
    The script uses Python's logging framework with three levels:

    - WARNING (--quiet): Only errors and warnings
    - INFO (default): Standard progress and results
    - DEBUG (--verbose): Detailed information including:
        * Model weight details
        * Image dimensions
        * TVM compilation progress
        * Detailed comparison metrics
        * Successful test details in batch mode

    Note: The verbose parameter is being phased out in favor of logging levels.
    Functions should use logger.info(), logger.debug(), etc. instead of
    checking verbose flags.

Architecture:
    The script follows a modular design with clear separation of concerns:

    1. Model Loading & Preprocessing:
       - load_model_with_preprocessing()
       - get_preprocessing_from_weight()
       - get_default_coco_preprocessing()
       - get_ssd_preprocessing_for_tvm() - Special preprocessing for SSD models

    2. Inference Backends:
       - run_inference() - PyTorch backend
       - prepare_model_for_tvm() - TVM compilation
       - run_inference_tvm() - TVM execution

    3. Detection & Visualization:
       - display_detections() - Show detected objects
       - filter_detections_by_score() - Filter low-confidence detections
       - compute_detection_metrics() - Calculate mAP, IoU, etc.

    4. Comparison & Analysis:
       - compare_inference_results()
       - Helper functions for table and summary output

    5. Orchestration:
       - main() - Single model testing
       - test_multiple_models() - Batch testing

TVM Support for Single-Stage Detectors (SSD/SSDLite):
    TVM C Static compilation is supported for single-stage detection models including:
    - SSD300_VGG16, SSD512_VGG16
    - SSDLite320_MobileNet_V3_Large

    Two-stage detectors (Faster R-CNN, Mask R-CNN, etc.) are not supported due to
    complex RPN + ROI head pipelines that cannot be exported via torch.export.

    Key Implementation Details:

    1. Model Wrapping (DetectionModelWrapper):
       - Extracts backbone + head (bypassing internal RCNNTransform)
       - Returns tuple: (bbox_regression, cls_logits)
       - Enables torch.export for TVM compilation

    2. Anchor Extraction (extract_detection_components):
       - Pre-generates anchors for the target image size
       - Extracts box_coder weights for delta decoding
       - Must match image size used during inference

    3. Preprocessing - Critical for Correct Results:
       SSD variants use DIFFERENT normalization parameters:

       a) SSD300/SSD512 (VGG backbone):
          - mean = [0.48235, 0.45882, 0.40784]
          - std = [1/255, 1/255, 1/255] ≈ [0.00392, 0.00392, 0.00392]
          - Input range: [0, 255] (ToTensor → scale to 255 → normalize)
          - Target size: 300×300 or 512×512

       b) SSDLite320 (MobileNet backbone):
          - mean = [0.5, 0.5, 0.5]
          - std = [0.5, 0.5, 0.5]
          - Input range: [0, 1] (ToTensor → normalize directly)
          - Target size: 320×320

       The get_ssd_preprocessing_for_tvm() function auto-detects which strategy
       to use based on std values (< 0.01 → uses 255 scale, else uses [0,1]).

       IMPORTANT: Using wrong normalization causes incorrect detections!
       Example: SSDLite with [0,255] scaling → detects banana instead of bird.

    4. Post-Processing (apply_detection_postprocessing):
       - Decodes box deltas using pre-generated anchors and box_coder weights
       - Applies softmax to classification logits
       - For SSD: Takes max score across NON-BACKGROUND classes only (indices 1-90)
       - Filters by score threshold (model default: 0.01 for SSD, 0.001 for SSDLite)
       - Applies per-class NMS (IoU threshold: 0.45 for SSD, 0.55 for SSDLite)

    5. Score Threshold Differences:
       - PyTorch full model: Uses internal threshold (0.01 for SSD, 0.001 for SSDLite)
       - TVM inference: Default 0.5 for consistency, can be adjusted via --score-threshold
       - Lower threshold (e.g., 0.01) produces more detections but may include false positives

Expected Results:
    - SSD300_VGG16: Bird detection ~99.65% confidence
    - SSDLite320_MobileNet_V3_Large: Bird detection ~95.68% confidence
    - Compilation time: 3-5 minutes per model (first run)
    - Inference: Fast execution via C Static binary

TVM Support for Anchor-Free Detectors (FCOS):
    TVM C Static compilation is also supported for anchor-free detectors:
    - FCOS_ResNet50_FPN

    FCOS (Fully Convolutional One-Stage Object Detection) is an anchor-free
    detector that predicts bounding boxes directly from feature map grid points.

    Key Implementation Details:

    1. Architecture Differences from SSD:
       - Uses FPN (Feature Pyramid Network) with 5 levels
       - Stride values: [8, 16, 32, 64, 128]
       - No predefined anchors; uses grid point centers
       - Predicts distances (left, top, right, bottom) instead of deltas
       - Includes centerness prediction to improve box quality

    2. Model Wrapping (DetectionModelWrapper):
       - Returns 3-tuple: (bbox_regression, cls_logits, bbox_ctrness)
       - bbox_regression contains distance predictions (l, t, r, b)
       - bbox_ctrness is used to weight final confidence scores

    3. Grid Point Generation (extract_detection_components):
       - Generates single point per spatial location (e.g., 13,343 points for 800×800)
       - Extracts stride information for each feature level
       - Strides used to scale distance predictions to image coordinates

    4. Preprocessing:
       - Uses standard COCO preprocessing (same as RetinaNet)
       - mean = [0.485, 0.456, 0.406]
       - std = [0.229, 0.224, 0.225]
       - Target size: Maintains aspect ratio with max_size=1333

    5. Box Decoding (decode_boxes_fcos):
       - Converts distance predictions to absolute coordinates:
         x1 = grid_x - left * stride
         y1 = grid_y - top * stride
         x2 = grid_x + right * stride
         y2 = grid_y + bottom * stride

    6. Post-Processing (apply_detection_postprocessing):
       - Applies sigmoid to classification logits (no background class)
       - Applies sigmoid to centerness predictions
       - Multiplies classification scores by centerness: score = cls_score * ctrness
       - Default thresholds: score_thresh=0.2, nms_thresh=0.6
       - Applies per-class NMS

Expected Results:
    - FCOS_ResNet50_FPN: Typical detection confidence ~30-70%
    - Centerness weighting improves localization quality
    - Compilation time: Similar to SSD (~3-5 minutes)
    - Inference: Fast execution via C Static binary

TVM Support for RetinaNet:
    TVM C Static compilation is also supported for RetinaNet detectors:
    - RetinaNet_ResNet50_FPN
    - RetinaNet_ResNet50_FPN_V2

    RetinaNet is an anchor-based detector using FPN and focal loss for handling
    class imbalance.

    Key Implementation Details:

    1. Architecture:
       - Uses FPN with 5 feature levels
       - Multiple anchors per location (9 anchors with 3 scales × 3 aspect ratios)
       - Typical anchor count: ~34,974 for standard input sizes
       - Sigmoid activation per class (independent binary classifiers)

    2. Model Wrapping (DetectionModelWrapper):
       - Returns 2-tuple: (bbox_regression, cls_logits)
       - Similar to SSD but uses sigmoid instead of softmax

    3. Preprocessing:
       - Uses standard COCO preprocessing (same as FCOS)
       - mean = [0.485, 0.456, 0.406]
       - std = [0.229, 0.224, 0.225]
       - Target size: Maintains aspect ratio with max_size=1333

    4. Post-Processing (apply_detection_postprocessing):
       - Applies sigmoid to classification logits
       - No background class (unlike SSD)
       - IMPORTANT: Do NOT add 1 to label indices - argmax already gives
         correct COCO labels (0-89 corresponding to COCO classes)
       - Default thresholds: score_thresh=0.05, nms_thresh=0.5

Expected Results:
    - RetinaNet_ResNet50_FPN: Typical detection confidence ~90-95%
    - RetinaNet_ResNet50_FPN_V2: Similar performance with improved training
    - Compilation time: Similar to other models (~3-5 minutes)
    - Inference: Fast execution via C Static binary

Critical Implementation Notes - Coordinate Spaces and Label Indexing:

    1. Coordinate Space Handling for SSD Models:
       Problem: PyTorch models have internal GeneralizedRCNNTransform that resizes
       images during inference. For example, a 500×368 input image is resized to
       300×300 for SSD300. PyTorch outputs boxes in ORIGINAL coordinates (500×368),
       but TVM outputs boxes in PREPROCESSED coordinates (300×300) where anchors
       were generated.

       Solution: In compare_inference_results(), TVM boxes are scaled from
       preprocessed coordinates to original coordinates using scale factors:
         scale_x = original_width / preprocessed_width
         scale_y = original_height / preprocessed_height
         scaled_box = [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]

       This scaling is applied automatically for SSD/SSDLite models and ensures
       box coordinate matching between PyTorch and TVM outputs.

       Impact: Without this fix, box IoU was 0.000 (complete mismatch). After fix,
       IoU improved to 0.98+ for SSD models.

    2. Label Indexing for Anchor-Free Detectors (FCOS, RetinaNet):
       Problem: FCOS and RetinaNet don't have background classes. Their classification
       outputs have 90 classes corresponding to COCO categories. The argmax operation
       returns indices 0-89, which map directly to COCO labels. However, earlier
       implementations incorrectly added 1 to these indices, shifting all labels
       (e.g., "bird" index 15 became "cat" index 16).

       Solution: In apply_detection_postprocessing(), removed the incorrect
       "labels = labels + 1" for FCOS and RetinaNet. Added explicit comments:
         "Do NOT add 1 to labels - argmax already gives correct COCO indices"

       Contrast with SSD: SSD DOES have a background class at index 0, so we slice
       it out (class_probs[:, 1:]), take argmax (returning 0-89), then add 1 back
       to get labels 1-90. This is correct for SSD but was incorrectly applied to
       FCOS/RetinaNet.

       Impact: Without this fix, label mismatch rate was 40% (RetinaNet models
       detected wrong objects). After fix, all models achieve 100% label match.

    3. Target Size Auto-Detection:
       Different SSD variants use different target sizes (SSD300: 300×300,
       SSDLite320: 320×320). The get_model_target_size() function extracts this
       from model.transform.max_size or model.transform.min_size, ensuring
       preprocessing uses the correct target size for each model variant.

    Test Results After Fixes:
       - Box match rate: 100% (5/5 models)
       - Average IoU: 0.839 across all single-stage detectors
       - Label accuracy: 100% for all models

Dependencies:
    - torch, torchvision: Model definitions and weights
    - tvm: Compiler framework for ML models
    - PIL: Image loading and drawing
    - numpy: Numerical operations
    - requests: Fetching COCO labels
    - pyyaml (optional): Configuration file support
"""

import argparse
import csv
import logging
import threading
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.models.detection as detection_models
import torchvision.transforms as transforms
import tvm
from PIL import Image
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

from tvm_utils import process_relax

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_IMAGE_URL = "test_images/bird_0.jpg"
COCO_LABELS_URL = (
    "https://raw.githubusercontent.com/amikelive/coco-labels/master/coco-labels-2014_2017.txt"
)
COCO_NUM_CLASSES = 91  # COCO has 80 labeled classes but uses IDs 1-90 (some missing)
DEFAULT_COCO_MEAN = [0.485, 0.456, 0.406]
DEFAULT_COCO_STD = [0.229, 0.224, 0.225]
DEFAULT_MIN_SIZE = 800
DEFAULT_MAX_SIZE = 1333
DEFAULT_SCORE_THRESHOLD = 0.5
DEFAULT_INPUT_SHAPE = (1, 3, 800, 800)
C_STATIC_TARGET = "c_static"
LLVM_TARGET = "llvm"

# Comparison tolerances
RTOL_COMPARISON = 1e-3
ATOL_COMPARISON = 1e-5
IOU_THRESHOLD = 0.5  # For box matching in comparisons
NMS_IOU_THRESHOLD = 0.5  # NMS IoU threshold for post-processing

# Model type classification for export compatibility
SINGLE_STAGE_PATTERNS = ["ssd", "ssdlite", "retinanet", "fcos"]
TWO_STAGE_PATTERNS = ["fasterrcnn", "maskrcnn", "keypointrcnn"]

# Detection architecture types
ANCHOR_BASED_MODELS = ["ssd", "ssdlite", "retinanet"]
ANCHOR_FREE_MODELS = ["fcos"]

# Cache for COCO labels
_COCO_LABELS_CACHE: Optional[Dict[int, str]] = None

# Thread lock for CSV file writing
_CSV_WRITE_LOCK = threading.Lock()


# Data classes for structured data
@dataclass
class Detection:
    """Single object detection"""

    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    label: int
    label_name: str
    score: float


@dataclass
class ModelTestResult:
    """Results from testing a single object detection model"""

    model: Optional[nn.Module]
    preprocessing: Optional[Callable]
    image: Optional[Image.Image]
    detections: Optional[List[Detection]]  # Changed from predictions to detections
    success: bool = True  # True if test succeeded, False if failed
    error_message: Optional[str] = None  # Error message if success=False
    comparison: Optional[Dict[str, Any]] = None
    tvm_compile_success: Optional[bool] = None  # True if TVM compilation succeeded
    tvm_inference_success: Optional[bool] = None  # True if TVM inference succeeded


@dataclass
class ComparisonResult:
    """Results from comparing PyTorch vs TVM detection outputs"""

    model_name: str
    num_detections_match: Optional[bool]
    boxes_match: Optional[bool]  # Changed from top1_match
    mean_iou: Optional[float]  # Changed from top5_match
    error: bool = False
    error_message: Optional[str] = None


@dataclass
class DetectionModelComponents:
    """Components extracted from a detection model for TVM inference + post-processing"""

    anchors: torch.Tensor  # Pre-generated anchors [num_anchors, 4]
    num_classes: int  # Number of object classes (e.g., 91 for COCO)
    architecture: str  # 'ssd', 'retinanet', or 'fcos'
    image_size: Tuple[int, int]  # (height, width) used for anchor generation
    box_coder_params: Optional[Dict[str, Any]] = None  # Parameters for box decoding


# Model classification functions
def is_single_stage_detector(model_name: str) -> bool:
    """Check if model is a single-stage detector (exportable via torch.export)"""
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in SINGLE_STAGE_PATTERNS)


def is_two_stage_detector(model_name: str) -> bool:
    """Check if model is a two-stage detector (not exportable)"""
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in TWO_STAGE_PATTERNS)


def is_anchor_based(model_name: str) -> bool:
    """Check if model uses anchor-based detection"""
    model_lower = model_name.lower()
    return any(pattern in model_lower for pattern in ANCHOR_BASED_MODELS)


def get_detection_architecture(model_name: str) -> Optional[str]:
    """Get detection architecture type: 'ssd', 'retinanet', 'fcos', or None"""
    model_lower = model_name.lower()
    if "ssd" in model_lower:
        return "ssd"
    elif "retinanet" in model_lower:
        return "retinanet"
    elif "fcos" in model_lower:
        return "fcos"
    return None


def setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging based on verbosity flags

    Args:
        verbose: Enable debug-level logging
        quiet: Enable only warning-level logging
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Configure root logger
    logging.basicConfig(level=level, format="%(message)s")

    # Also configure tvm_utils logger to match the same level
    tvm_utils_logger = logging.getLogger("tvm_utils")
    tvm_utils_logger.setLevel(level)


def append_result_to_log(
    log_file: str,
    model_name: str,
    num_detections: Optional[int] = None,
    top_detection: Optional[str] = None,
    confidence: Optional[float] = None,
    num_detections_match: Optional[bool] = None,
    boxes_match: Optional[bool] = None,
    mean_iou: Optional[float] = None,
    tvm_compile_success: Optional[bool] = None,
    tvm_inference_success: Optional[bool] = None,
    error: bool = False,
    error_message: Optional[str] = None,
) -> None:
    """Append a detection test result to the CSV log file (thread-safe)

    Args:
        log_file: Path to the CSV log file
        model_name: Name of the model tested
        num_detections: Number of detections found
        top_detection: Highest confidence detection label
        confidence: Highest confidence score (0-100)
        num_detections_match: Whether detection counts match (PyTorch vs TVM)
        boxes_match: Whether bounding boxes match (PyTorch vs TVM)
        mean_iou: Mean IoU for matched detections
        tvm_compile_success: Whether TVM compilation succeeded
        tvm_inference_success: Whether TVM inference succeeded
        error: Whether an error occurred
        error_message: Error message if error is True
    """

    # Helper to convert bool to Yes/No/N/A
    def bool_to_str(value: Optional[bool]) -> str:
        if value is None:
            return "N/A"
        return "Yes" if value else "No"

    # Prepare row data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        timestamp,
        model_name,
        str(num_detections) if num_detections is not None else "N/A",
        top_detection or "N/A",
        f"{confidence:.2f}" if confidence is not None else "N/A",
        bool_to_str(num_detections_match),
        bool_to_str(boxes_match),
        f"{mean_iou:.3f}" if mean_iou is not None else "N/A",
        bool_to_str(tvm_compile_success),
        bool_to_str(tvm_inference_success),
        "Yes" if error else "No",
        error_message or "",
    ]

    # Thread-safe file writing
    with _CSV_WRITE_LOCK:
        try:
            file_path = Path(log_file)
            file_exists = file_path.exists()

            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Write header if file is new
                if not file_exists:
                    header = [
                        "timestamp",
                        "model_name",
                        "num_detections",
                        "top_detection",
                        "confidence",
                        "num_detections_match",
                        "boxes_match",
                        "mean_iou",
                        "tvm_compile",
                        "tvm_inference",
                        "error",
                        "error_message",
                    ]
                    writer.writerow(header)
                    logger.debug(f"Created new log file: {log_file}")

                # Write result row
                writer.writerow(row)

        except Exception as e:
            logger.error(f"Failed to write to log file {log_file}: {e}")


def load_model_with_preprocessing(
    model_name: str = "fasterrcnn_resnet50_fpn", weight_name: Optional[str] = None
) -> Tuple[nn.Module, Callable, Optional[Any]]:
    """Load detection model and automatically determine preprocessing transforms

    Args:
        model_name: Name of the TorchVision detection model to load
        weight_name: Specific weight name to use, or None for default

    Returns:
        Tuple of (model, preprocessing_transforms, weight_enum)
    """
    logger.debug(f"Loading model: {model_name}")

    # Get the model function from detection module
    model_func = getattr(detection_models, model_name)

    weight: Optional[Any] = None

    # Try to find the weights enum by searching for a matching class
    try:
        # Look for weights class that matches the model name
        # Detection models have weights like FasterRCNN_ResNet50_FPN_Weights
        model_name_lower = model_name.lower()
        weights_enum = None

        # Search for matching weights class in detection_models
        for attr_name in dir(detection_models):
            if attr_name.endswith("_Weights") and model_name_lower == attr_name.lower().replace(
                "_weights", ""
            ):
                weights_enum = getattr(detection_models, attr_name)
                logger.debug(f"  Found weights class: {attr_name}")
                break

        if weights_enum:
            # Use specific weight or default
            if weight_name:
                weight = getattr(weights_enum, weight_name)
            else:
                # Get default weight
                weight = getattr(weights_enum, "DEFAULT", list(weights_enum)[0])

            assert weight is not None
            logger.debug(f"  Weights: {weight.name}")

            # Load model with specific weights
            model = model_func(weights=weight)

            # Get preprocessing transforms
            preprocessing = get_preprocessing_from_weight(weight, model_name)
        else:
            # Fallback if weights class not found
            logger.debug(f"  Weights class not found for {model_name}, using default")
            model = model_func(weights="DEFAULT")
            preprocessing = get_default_coco_preprocessing()

    except (AttributeError, ValueError) as e:
        # Fallback for models without weights enum
        logger.debug(f"  Error loading weights: {e}, using default")
        model = model_func(weights="DEFAULT")
        preprocessing = get_default_coco_preprocessing()

    # Set to evaluation mode
    model.eval()

    return model, preprocessing, weight


def get_preprocessing_from_weight(weight: Any, model_name: str) -> Callable:
    """Extract preprocessing transforms from weight metadata for detection models

    Args:
        weight: TorchVision weight enum instance
        model_name: Name of the model (for logging)

    Returns:
        Callable preprocessing transforms (either transforms.Compose or detection transform)
    """
    # First, try to get transforms directly from weight
    if hasattr(weight, "transforms") and callable(weight.transforms):
        try:
            transforms_obj = weight.transforms()
            logger.debug("  Using transforms from weight object")
            return transforms_obj  # type: ignore[return-value]
        except Exception as e:
            logger.warning(f"  Could not get transforms from weight object: {e}")

    # Second, try to build from metadata
    # For object detection, we typically only do ToTensor() and Normalize()
    # Detection models handle resizing internally
    if hasattr(weight, "meta") and weight.meta:
        meta = weight.meta
        transforms_list = []

        # Convert to tensor
        transforms_list.append(transforms.ToTensor())

        # Normalization
        if "mean" in meta and "std" in meta:
            mean = meta["mean"]
            std = meta["std"]
            transforms_list.append(transforms.Normalize(mean=mean, std=std))
        else:
            transforms_list.append(
                transforms.Normalize(mean=DEFAULT_COCO_MEAN, std=DEFAULT_COCO_STD)
            )

        preprocessing = transforms.Compose(transforms_list)
        return preprocessing

    # Fallback to default
    return get_default_coco_preprocessing()


def get_default_coco_preprocessing() -> transforms.Compose:
    """Default COCO preprocessing as fallback

    Returns:
        Standard COCO preprocessing transforms for object detection
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_COCO_MEAN, std=DEFAULT_COCO_STD),
        ]
    )


def get_model_target_size(model: nn.Module) -> Optional[int]:
    """Extract target image size from model's transform

    Args:
        model: Detection model with transform attribute

    Returns:
        Target size (e.g., 300 for SSD300, 320 for SSDLite) or None if not found
    """
    if hasattr(model, "transform"):
        transform = model.transform  # type: ignore[attr-defined]
        if hasattr(transform, "max_size"):
            return transform.max_size  # type: ignore[return-value,attr-defined]
        elif hasattr(transform, "min_size") and transform.min_size:  # type: ignore[attr-defined]
            # min_size is typically a tuple like (300,)
            min_size = transform.min_size  # type: ignore[attr-defined]
            if isinstance(min_size, (tuple, list)) and len(min_size) > 0:
                return min_size[0]  # type: ignore[return-value]
    return None


def get_ssd_preprocessing_for_tvm(model: nn.Module, target_size: Optional[int] = None) -> Callable:
    """Get preprocessing transform for SSD models compatible with TVM wrapped model

    SSD models have an internal GeneralizedRCNNTransform that applies normalization.
    When we wrap the model for TVM, we bypass the internal transform, so we need
    to apply the normalization ourselves.

    Different SSD variants use different normalization:
    - SSD300/512 (VGG): std ≈ 1/255 → expects [0, 255] range input
    - SSDLite (MobileNet): std = 0.5 → expects [0, 1] range input

    Args:
        model: SSD model instance
        target_size: Target image size (300 for SSD300, 320 for SSDLite, etc.)
                     If None, will be auto-detected from model.transform

    Returns:
        Preprocessing transform that resizes, converts to tensor, and normalizes
    """
    # Auto-detect target size if not provided
    if target_size is None:
        target_size = get_model_target_size(model)
        if target_size is None:
            logger.warning("Could not detect target size from model, defaulting to 300")
            target_size = 300
    # Get normalization params from model.transform
    if hasattr(model, "transform"):
        mean = model.transform.image_mean  # type: ignore[attr-defined]
        std = model.transform.image_std  # type: ignore[attr-defined]

        # Detect which normalization strategy to use based on std values
        # If std is very small (around 1/255 = 0.00392), it expects [0, 255] input
        # If std is larger (0.5), it expects [0, 1] input
        uses_255_scale = all(s < 0.01 for s in std)  # type: ignore[union-attr]

        if uses_255_scale:
            # SSD300/512 style: scale to [0, 255] before normalizing
            return transforms.Compose(
                [
                    transforms.Resize((target_size, target_size)),
                    transforms.ToTensor(),  # Converts to [0, 1]
                    transforms.Lambda(lambda x: x * 255.0),  # Scale to [0, 255]
                    transforms.Normalize(
                        mean=[m * 255.0 for m in mean],  # type: ignore[misc]
                        std=[s * 255.0 for s in std],  # type: ignore[misc]
                    ),
                ]
            )
        else:
            # SSDLite style: keep [0, 1] range, normalize directly
            return transforms.Compose(
                [
                    transforms.Resize((target_size, target_size)),
                    transforms.ToTensor(),  # Converts to [0, 1]
                    transforms.Normalize(mean=mean, std=std),  # type: ignore[arg-type]
                ]
            )
    else:
        # Fallback to standard preprocessing
        return transforms.Compose(
            [
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=DEFAULT_COCO_MEAN, std=DEFAULT_COCO_STD),
            ]
        )


def load_image(image_path_or_url: str) -> Optional[Image.Image]:
    """Load image from file path or URL

    Args:
        image_path_or_url: Path to local file or HTTP(S) URL

    Returns:
        PIL Image if successful, None if loading fails
    """
    logger.debug(f"Loading image: {image_path_or_url}")

    try:
        if image_path_or_url.startswith(("http://", "https://")):
            # Load from URL
            response = requests.get(image_path_or_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            # Load from local file
            image = Image.open(image_path_or_url).convert("RGB")

        logger.debug(f"  Size: {image.size}")
        return image

    except (requests.RequestException, OSError) as e:
        logger.error(f"Error loading image: {e}")
        return None


def load_coco_labels(force_reload: bool = False) -> Dict[int, str]:
    """Load COCO class labels (cached after first call)

    Uses the COCO category mapping from TorchVision weights metadata,
    which provides the correct category IDs (0-90 with gaps).

    Args:
        force_reload: If True, bypass cache and reload

    Returns:
        Dictionary mapping class IDs to class names
    """
    global _COCO_LABELS_CACHE

    if _COCO_LABELS_CACHE is not None and not force_reload:
        return _COCO_LABELS_CACHE

    try:
        # Use TorchVision's COCO categories from model weights
        # This ensures correct mapping of category IDs (which have gaps)
        weights = detection_models.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        if hasattr(weights, "meta") and "categories" in weights.meta:
            categories = weights.meta["categories"]
            # Categories is a list where index = category ID
            label_dict = dict(enumerate(categories))
            logger.debug(f"Loaded {len(label_dict)} COCO categories from TorchVision weights")
        else:
            # Fallback: create dummy labels
            logger.warning("Could not load categories from weights, using dummy labels")
            label_dict = {i: f"class_{i}" for i in range(COCO_NUM_CLASSES)}

        _COCO_LABELS_CACHE = label_dict
        return label_dict

    except Exception as e:
        logger.warning(f"Could not load COCO labels: {e}")
        # Fallback: create dummy labels
        return {i: f"class_{i}" for i in range(COCO_NUM_CLASSES)}


def run_inference(
    model: nn.Module, image_tensor: torch.Tensor, score_threshold: float = DEFAULT_SCORE_THRESHOLD
) -> Dict[str, torch.Tensor]:
    """Run inference on the preprocessed image for object detection

    Args:
        model: PyTorch detection model in eval mode
        image_tensor: Preprocessed image tensor with shape (C, H, W)
        score_threshold: Minimum confidence score for detections

    Returns:
        Dictionary with 'boxes', 'labels', 'scores' tensors
    """
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    batch_tensor = image_tensor.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(batch_tensor)

    # Detection models return list of dicts, one per image
    detections = outputs[0]

    # Filter by score threshold
    keep = detections["scores"] >= score_threshold
    filtered_detections = {
        "boxes": detections["boxes"][keep],
        "labels": detections["labels"][keep],
        "scores": detections["scores"][keep],
    }

    return filtered_detections


def display_detections(
    detections: Dict[str, torch.Tensor], labels: Dict[int, str], max_display: int = 10
) -> List[Detection]:
    """Display detected objects

    Args:
        detections: Dictionary with 'boxes', 'labels', 'scores' tensors
        labels: Dictionary mapping class IDs to class names
        max_display: Maximum number of detections to display

    Returns:
        List of Detection objects
    """
    boxes = detections["boxes"]
    det_labels = detections["labels"]
    scores = detections["scores"]

    num_detections = len(boxes)
    logger.info(f"\nDetected {num_detections} objects:")
    logger.info("-" * 80)

    results = []
    for i in range(min(num_detections, max_display)):
        box = boxes[i].tolist()
        label_id = int(det_labels[i].item())
        label_name = labels.get(label_id, f"class_{label_id}")
        score = scores[i].item()

        logger.info(
            f"{i + 1:2d}. {label_name:20s} {score * 100:6.2f}% "
            f"Box: [{box[0]:6.1f}, {box[1]:6.1f}, {box[2]:6.1f}, {box[3]:6.1f}]"
        )

        det = Detection(
            box=(box[0], box[1], box[2], box[3]), label=label_id, label_name=label_name, score=score
        )
        results.append(det)

    if num_detections > max_display:
        logger.info(f"... and {num_detections - max_display} more detections")

    return results


def compute_iou(
    box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]
) -> float:
    """Compute Intersection over Union (IoU) between two boxes

    Args:
        box1: First box as (x1, y1, x2, y2)
        box2: Second box as (x1, y1, x2, y2)

    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Compute intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Compute union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def print_preprocessing_details(preprocessing: Callable, weight: Optional[Any] = None) -> None:
    """Print detailed information about preprocessing transforms

    Args:
        preprocessing: The preprocessing transforms
        weight: Optional weight enum with metadata
    """
    logger.debug("\nPreprocessing Pipeline:")

    if weight and hasattr(weight, "meta") and weight.meta:
        meta = weight.meta
        if "min_size" in meta or "max_size" in meta:
            min_size = meta.get("min_size", "N/A")
            max_size = meta.get("max_size", "N/A")
            logger.debug(f"  Min size: {min_size}, Max size: {max_size}")
        if "mean" in meta and "std" in meta:
            logger.debug(f"  Normalize: mean={meta['mean']}, std={meta['std']}")
    else:
        logger.debug("  Using default COCO preprocessing")


class DetectionModelWrapper(nn.Module):
    """Wrapper to export detection models with torch.export

    Extracts backbone + head outputs for TVM compilation.
    Post-processing (box decoding, NMS) happens in Python after TVM inference.

    Based on ObjectDetectionHeadSSDExample from ExecutorTch examples.

    Returns:
        For SSD/RetinaNet: Tuple of (bbox_regression, cls_logits)
        For FCOS: Tuple of (bbox_regression, cls_logits, bbox_ctrness)

        - bbox_regression: Box deltas [batch, num_anchors, 4] or distances for FCOS
        - cls_logits: Class logits [batch, num_anchors, num_classes]
        - bbox_ctrness: Centerness scores for FCOS [batch, num_anchors, 1]
    """

    def __init__(self, model: nn.Module, architecture: str) -> None:
        super().__init__()
        self.backbone: nn.Module = model.backbone  # type: ignore[assignment]
        self.head: nn.Module = model.head  # type: ignore[assignment]
        self.architecture = architecture

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Run backbone
        features = self.backbone(x)

        # Convert dict/OrderedDict to list
        if isinstance(features, dict):
            features_list = list(features.values())
        else:
            features_list = features

        # Handle dimension mismatches (3D -> 4D) from model_aot_compilation pattern
        processed_features = [
            feat.unsqueeze(0) if feat.ndim == 3 else feat for feat in features_list
        ]

        # Run head
        outputs = self.head(processed_features)

        # Extract and return bbox and cls outputs
        bbox_regression = outputs["bbox_regression"]
        cls_logits = outputs["cls_logits"]

        # FCOS also has centerness prediction
        if self.architecture == "fcos" and "bbox_ctrness" in outputs:
            bbox_ctrness = outputs["bbox_ctrness"]
            return bbox_regression, cls_logits, bbox_ctrness

        # Return as tuple for torch.export
        return bbox_regression, cls_logits


def extract_detection_components(
    model: nn.Module, model_name: str, image_size: Tuple[int, int] = (800, 800)
) -> Optional[DetectionModelComponents]:
    """Extract anchors and metadata from a TorchVision detection model

    Args:
        model: TorchVision detection model
        model_name: Model name (for architecture detection)
        image_size: Image size (H, W) for anchor generation

    Returns:
        DetectionModelComponents with anchors and metadata, or None if unsupported
    """
    architecture = get_detection_architecture(model_name)

    if architecture is None:
        logger.error(f"Unknown detection architecture for {model_name}")
        return None

    # Special handling for FCOS (anchor-free detector)
    if architecture == "fcos":
        try:
            # FCOS generates grid points at each spatial location across FPN levels
            # Strides: [8, 16, 32, 64, 128] for 5 FPN levels
            from torchvision.models.detection.image_list import ImageList

            with torch.no_grad():
                dummy_input = torch.randn(1, 3, *image_size)
                features = model.backbone(dummy_input)  # type: ignore[attr-defined]

                if isinstance(features, dict):
                    features_list = list(features.values())
                else:
                    features_list = features

                # Generate grid points for each feature level
                # FCOS anchor generator creates one anchor per spatial location
                image_sizes = [image_size]
                image_list = ImageList(dummy_input, image_sizes)

                # Get anchor points (grid centers)
                anchor_generator = model.anchor_generator
                grid_points_list = anchor_generator(image_list, features_list)  # type: ignore[misc]

                # Concatenate all grid points
                grid_points = torch.cat(grid_points_list, dim=0)

                # Store grid points as "anchors" for FCOS
                # Grid points have shape (num_points, 4) but for FCOS we only need (x, y)
                # The x1, y1, x2, y2 format becomes center_x, center_y, center_x, center_y

                logger.debug(f"  Generated {len(grid_points)} grid points for FCOS")

                # Extract strides for each feature level
                # FCOS uses these to scale the predicted distances
                strides = []
                for feat in features_list:
                    h, w = feat.shape[-2:]
                    stride_h = image_size[0] / h
                    stride_w = image_size[1] / w
                    stride = (stride_h + stride_w) / 2  # Average stride
                    strides.extend([stride] * (h * w))

                strides_tensor = torch.tensor(strides, dtype=torch.float32)

                # Store strides in box_coder_params for FCOS
                box_coder_params = {"strides": strides_tensor}

            return DetectionModelComponents(
                anchors=grid_points,  # Grid points for FCOS
                num_classes=COCO_NUM_CLASSES,
                architecture=architecture,
                image_size=image_size,
                box_coder_params=box_coder_params,
            )

        except Exception as e:
            logger.error(f"Failed to generate FCOS grid points: {e}")
            logger.debug("Traceback:", exc_info=True)
            return None

    # Check if model has anchor generator (for anchor-based models)
    if not hasattr(model, "anchor_generator"):
        logger.error(f"Model {model_name} missing anchor_generator")
        return None

    try:
        # Generate anchors for the specified image size
        from torchvision.models.detection.image_list import ImageList

        anchor_generator = model.anchor_generator

        # For SSD/RetinaNet, anchors are generated based on feature map sizes
        # We need to do a dummy forward pass to get feature map sizes
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, *image_size)
            features = model.backbone(dummy_input)  # type: ignore[attr-defined]

            if isinstance(features, dict):
                features_list = list(features.values())
            else:
                features_list = features

            # Create ImageList for anchor generation
            image_sizes = [image_size]
            image_list = ImageList(dummy_input, image_sizes)

            # Generate anchors using the image list and feature maps
            anchors_list = anchor_generator(image_list, features_list)  # type: ignore[misc]

            # Concatenate anchors from all feature levels (anchors_list is already per-image)
            anchors = torch.cat(anchors_list, dim=0)

        logger.debug(f"  Extracted {len(anchors)} anchors for {architecture}")

        # Extract box coder parameters if available
        box_coder_params = None
        if hasattr(model, "box_coder"):
            box_coder = model.box_coder
            if hasattr(box_coder, "weights"):
                box_coder_params = {"weights": box_coder.weights}  # type: ignore[attr-defined]

        # Get number of classes
        num_classes = COCO_NUM_CLASSES
        if hasattr(model, "num_classes"):
            num_classes = int(model.num_classes)  # type: ignore[arg-type, attr-defined]
        elif hasattr(model.head, "num_classes"):  # type: ignore[attr-defined]
            num_classes = int(model.head.num_classes)  # type: ignore[arg-type, attr-defined]

        return DetectionModelComponents(
            anchors=anchors,
            num_classes=num_classes,
            architecture=architecture,
            image_size=image_size,
            box_coder_params=box_coder_params,
        )

    except Exception as e:
        logger.error(f"Failed to extract components from {model_name}: {e}")
        logger.debug("Traceback:", exc_info=True)
        return None


def decode_boxes(
    box_deltas: torch.Tensor,
    anchors: torch.Tensor,
    weights: Optional[Tuple[float, float, float, float]] = None,
) -> torch.Tensor:
    """Decode box deltas into absolute coordinates using anchors

    Implements the standard box decoding used by TorchVision detection models.

    Args:
        box_deltas: Predicted box deltas [num_boxes, 4] in (dx, dy, dw, dh) format
        anchors: Anchor boxes [num_boxes, 4] in (x1, y1, x2, y2) format
        weights: Box coder weights for delta scaling (default: 1.0, 1.0, 1.0, 1.0)

    Returns:
        Decoded boxes [num_boxes, 4] in (x1, y1, x2, y2) format
    """
    if weights is None:
        weights = (1.0, 1.0, 1.0, 1.0)

    wx, wy, ww, wh = weights

    # Convert anchors from (x1, y1, x2, y2) to (cx, cy, w, h)
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
    anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights

    # Extract deltas
    dx = box_deltas[:, 0] / wx
    dy = box_deltas[:, 1] / wy
    dw = box_deltas[:, 2] / ww
    dh = box_deltas[:, 3] / wh

    # Apply deltas to anchors
    pred_ctr_x = dx * anchor_widths + anchor_ctr_x
    pred_ctr_y = dy * anchor_heights + anchor_ctr_y
    pred_w = torch.exp(dw) * anchor_widths
    pred_h = torch.exp(dh) * anchor_heights

    # Convert back to (x1, y1, x2, y2)
    pred_boxes = torch.zeros_like(box_deltas)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w  # x2
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h  # y2

    return pred_boxes


def decode_boxes_fcos(
    box_distances: torch.Tensor,
    grid_points: torch.Tensor,
    strides: torch.Tensor,
) -> torch.Tensor:
    """Decode FCOS distance predictions into absolute box coordinates

    FCOS predicts distances (left, top, right, bottom) from grid center points.
    This function converts those distances to absolute box coordinates following
    PyTorch's BoxLinearCoder with normalize_by_size=True.

    Args:
        box_distances: Predicted distances [num_points, 4] in (l, t, r, b) format
                       These are NORMALIZED by anchor box size in PyTorch
        grid_points: Grid center points [num_points, 4] in (x1, y1, x2, y2) format
                     These are anchor boxes (e.g., [-4, -4, 4, 4] for 8x8 anchor)
        strides: Stride multipliers [num_points] for each point (not used, kept for compatibility)

    Returns:
        Decoded boxes [num_points, 4] in (x1, y1, x2, y2) format
    """
    # Extract grid center coordinates
    # For FCOS anchors: center = 0.5 * (x1 + x2)
    grid_x = 0.5 * (grid_points[:, 0] + grid_points[:, 2])
    grid_y = 0.5 * (grid_points[:, 1] + grid_points[:, 3])

    # Get anchor box sizes for normalization
    # FCOS uses anchor sizes that match the stride at each level
    # (8x8 for stride 8, 16x16 for stride 16, etc.)
    anchor_w = grid_points[:, 2] - grid_points[:, 0]
    anchor_h = grid_points[:, 3] - grid_points[:, 1]

    # Unnormalize distances by anchor size (PyTorch BoxLinearCoder behavior)
    # The predicted distances are relative to anchor size
    left = box_distances[:, 0] * anchor_w
    top = box_distances[:, 1] * anchor_h
    right = box_distances[:, 2] * anchor_w
    bottom = box_distances[:, 3] * anchor_h

    # Decode to absolute coordinates
    # x1 = center_x - left distance
    # y1 = center_y - top distance
    # x2 = center_x + right distance
    # y2 = center_y + bottom distance
    pred_boxes = torch.zeros_like(box_distances)
    pred_boxes[:, 0] = grid_x - left  # x1
    pred_boxes[:, 1] = grid_y - top  # y1
    pred_boxes[:, 2] = grid_x + right  # x2
    pred_boxes[:, 3] = grid_y + bottom  # y2

    return pred_boxes


def apply_detection_postprocessing(
    bbox_deltas: torch.Tensor,
    cls_logits: torch.Tensor,
    components: DetectionModelComponents,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    bbox_ctrness: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Apply post-processing to detection model outputs (adapted from od_yolo.py)

    Pipeline:
    1. Decode boxes: deltas + anchors → absolute coordinates (or distances for FCOS)
    2. Apply softmax/sigmoid to class logits → probabilities
    3. For FCOS: multiply by centerness scores
    4. Filter by score threshold
    5. Apply per-class NMS
    6. Return standard detection dict

    Args:
        bbox_deltas: Box deltas from TVM [batch, num_anchors, 4] or [num_anchors, 4]
                     For FCOS: distances (l, t, r, b)
        cls_logits: Class logits from TVM [batch, num_anchors, num_classes]
        components: DetectionModelComponents with anchors/grid points and metadata
        score_threshold: Minimum confidence score
        iou_threshold: NMS IoU threshold
        bbox_ctrness: Centerness scores for FCOS [batch, num_anchors, 1] (optional)

    Returns:
        Dictionary with 'boxes', 'labels', 'scores' tensors
    """
    from torchvision.ops import batched_nms

    logger.debug(f"Post-processing {components.architecture} outputs")
    logger.debug(f"  bbox_deltas shape: {bbox_deltas.shape}")
    logger.debug(f"  cls_logits shape: {cls_logits.shape}")

    # Remove batch dimension if present
    if bbox_deltas.ndim == 3:
        bbox_deltas = bbox_deltas[0]
    if cls_logits.ndim == 3:
        cls_logits = cls_logits[0]

    # Decode boxes from deltas + anchors
    if components.architecture in ["ssd", "retinanet"]:
        # Anchor-based: decode using anchors
        weights = None
        if components.box_coder_params and "weights" in components.box_coder_params:
            weights = tuple(components.box_coder_params["weights"])

        decoded_boxes = decode_boxes(bbox_deltas, components.anchors, weights)
        logger.debug(f"  Decoded {len(decoded_boxes)} boxes from anchors")

    elif components.architecture == "fcos":
        # FCOS: boxes are distances (l, t, r, b) from grid points
        if components.box_coder_params is None or "strides" not in components.box_coder_params:
            raise ValueError("FCOS requires strides in box_coder_params")

        strides = components.box_coder_params["strides"]
        decoded_boxes = decode_boxes_fcos(bbox_deltas, components.anchors, strides)
        logger.debug(f"  Decoded {len(decoded_boxes)} boxes from FCOS grid points")
    else:
        raise ValueError(f"Unknown architecture: {components.architecture}")

    # Apply softmax to class logits (SSD uses softmax, RetinaNet uses sigmoid)
    if components.architecture == "ssd":
        # SSD: softmax over classes (multiclass with background at index 0)
        logger.debug(
            f"  Before softmax - cls_logits shape: {cls_logits.shape}, range: [{cls_logits.min():.4f}, {cls_logits.max():.4f}]"
        )
        class_probs = torch.softmax(cls_logits, dim=-1)
        logger.debug(
            f"  After softmax - class_probs shape: {class_probs.shape}, range: [{class_probs.min():.4f}, {class_probs.max():.4f}]"
        )

        # For SSD, we take the max score across NON-BACKGROUND classes only (indices 1-90)
        # Background class (index 0) is implicitly handled: if max_non_bg_score is low, it's background
        non_bg_probs = class_probs[:, 1:]  # Remove background class
        scores, labels = non_bg_probs.max(dim=1)
        labels = labels + 1  # Shift back because we removed background (indices now 1-90)

        logger.debug(f"  Non-background scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        logger.debug(f"  Labels (after bg removal): min={labels.min()}, max={labels.max()}")

    elif components.architecture == "retinanet":
        # RetinaNet: sigmoid (independent binary classifiers per class)
        # RetinaNet doesn't have a background class
        class_probs = torch.sigmoid(cls_logits)
        logger.debug(f"  Class probs shape: {class_probs.shape}")
        logger.debug(f"  Class probs range: [{class_probs.min():.4f}, {class_probs.max():.4f}]")

        # Get best class and score for each box
        scores, labels = class_probs.max(dim=1)
        # Note: Do NOT add 1 to labels - RetinaNet argmax already gives correct COCO indices (no background class)

    elif components.architecture == "fcos":
        # FCOS: sigmoid for classification + centerness
        # FCOS has 91 classes but the model outputs class indices that exclude background
        # Labels are already in the correct range (1-90) after argmax
        class_probs = torch.sigmoid(cls_logits)
        logger.debug(f"  Class probs shape: {class_probs.shape}")
        logger.debug(f"  Class probs range: [{class_probs.min():.4f}, {class_probs.max():.4f}]")

        # Get best class and score for each location
        scores, labels = class_probs.max(dim=1)
        # Note: Do NOT add 1 to labels - FCOS argmax already gives correct indices (background excluded)

        # Multiply scores by centerness for FCOS
        if bbox_ctrness is not None:
            # Remove batch dimension from centerness if present
            if bbox_ctrness.ndim == 3:
                bbox_ctrness = bbox_ctrness[0]

            # Centerness is (num_points, 1), squeeze to (num_points,)
            ctrness_scores = torch.sigmoid(bbox_ctrness.squeeze(-1))
            logger.debug(
                f"  Centerness range: [{ctrness_scores.min():.4f}, {ctrness_scores.max():.4f}]"
            )

            # FCOS final score = classification score * centerness
            scores = scores * ctrness_scores
            logger.debug(f"  Scores after centerness: [{scores.min():.4f}, {scores.max():.4f}]")
        else:
            logger.warning("  FCOS: bbox_ctrness not provided, using classification scores only")

    else:
        class_probs = torch.softmax(cls_logits, dim=-1)
        scores, labels = class_probs.max(dim=1)

    if len(scores) > 0:
        logger.debug(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        logger.debug(
            f"  Detections above threshold {score_threshold}: {(scores >= score_threshold).sum()}"
        )
    else:
        logger.debug("  No scores to evaluate (empty tensor)")

    # Filter by score threshold
    mask = scores >= score_threshold
    filtered_boxes = decoded_boxes[mask]
    filtered_scores = scores[mask]
    filtered_labels = labels[mask]

    if len(filtered_boxes) == 0:
        logger.debug("  No detections above confidence threshold")
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,), dtype=torch.float32),
        }

    # Apply per-class NMS using batched_nms
    keep_indices = batched_nms(
        filtered_boxes,
        filtered_scores,
        filtered_labels,  # Labels serve as batch indices for per-class NMS
        iou_threshold,
    )

    logger.debug(f"  After NMS: {len(keep_indices)} detections")

    return {
        "boxes": filtered_boxes[keep_indices],
        "labels": filtered_labels[keep_indices],
        "scores": filtered_scores[keep_indices],
    }


def torch_to_relax(torch_model: nn.Module, example_input: Tuple[torch.Tensor, ...]) -> tvm.IRModule:
    """Convert a PyTorch model to a Relax IRModule with flexible input handling

    Adapted from model_aot_compilation pattern with try/except for batch dimensions.

    Args:
        torch_model: PyTorch model to convert
        example_input: Example input tuple for torch.export

    Returns:
        TVM Relax IRModule
    """
    with torch.no_grad():
        logger.debug("  PyTorch export...")

        # Try export with provided input
        try:
            exported_program = export(torch_model, example_input, strict=False)
            logger.debug(f"    Export succeeded with input shape: {example_input[0].shape}")
        except Exception as e_unbatched:
            logger.debug(f"    Export failed with shape {example_input[0].shape}: {e_unbatched}")

            # Try with batched input if unbatched failed (pattern from model_aot_compilation)
            if example_input[0].ndim == 3:  # (C, H, W)
                logger.debug("    Retrying with batched input (add batch dimension)...")
                batched_input = (example_input[0].unsqueeze(0),)  # (1, C, H, W)
                try:
                    exported_program = export(torch_model, batched_input, strict=False)
                    logger.debug(
                        f"    Export succeeded with batched shape: {batched_input[0].shape}"
                    )
                except Exception as e_batched:
                    logger.error("    Export failed with both unbatched and batched inputs")
                    logger.debug(f"      Unbatched error: {e_unbatched}")
                    logger.debug(f"      Batched error: {e_batched}")
                    raise e_batched from e_unbatched
            else:
                raise

        logger.debug("  TVM Relax IR import...")
        mod = from_exported_program(exported_program, keep_params_as_input=True)
    return mod


def prepare_model_for_tvm(
    torch_model: nn.Module,
    model_name: str,
    input_shape: Tuple[int, ...] = DEFAULT_INPUT_SHAPE,
) -> Optional[Tuple[tvm.IRModule, DetectionModelComponents]]:
    """Prepare a detection model for TVM compilation with component extraction

    Args:
        torch_model: PyTorch detection model in eval mode
        model_name: Model name (for architecture detection and filtering)
        input_shape: Input tensor shape (batch, channels, height, width)

    Returns:
        Tuple of (TVM IRModule, DetectionModelComponents) or None if unsupported
    """
    logger.debug(f"  Preparing {model_name} for TVM...")

    # Filter out two-stage detectors early
    if is_two_stage_detector(model_name):
        logger.warning(
            f"Model {model_name} is a two-stage detector (Faster R-CNN / Mask R-CNN / Keypoint R-CNN). "
            "Two-stage detectors have complex RPN + ROI head pipelines that cannot be exported via torch.export. "
            "Only single-stage detectors (SSD, RetinaNet, FCOS) are supported."
        )
        return None

    # Check if it's a supported single-stage detector
    if not is_single_stage_detector(model_name):
        logger.warning(
            f"Model {model_name} is not recognized as a supported detection model. "
            f"Supported patterns: {SINGLE_STAGE_PATTERNS}"
        )
        return None

    architecture = get_detection_architecture(model_name)
    if architecture is None:
        logger.error(f"Could not determine architecture for {model_name}")
        return None

    # Extract detection components (anchors, metadata) BEFORE wrapping
    image_size = (input_shape[2], input_shape[3])  # (H, W)
    components = extract_detection_components(torch_model, model_name, image_size)

    if components is None:
        logger.error(f"Failed to extract detection components from {model_name}")
        return None

    # Wrap model to export backbone + head with tuple output
    logger.debug(f"  Wrapping model with DetectionModelWrapper (architecture={architecture})")
    wrapped_model = DetectionModelWrapper(torch_model, architecture)

    # Create example input
    example_input = (torch.randn(*input_shape, dtype=torch.float32),)

    try:
        # Convert to Relax IRModule (with flexible input handling)
        mod = torch_to_relax(wrapped_model, example_input)

        # Process the module (detach and bind parameters)
        mod = process_relax(mod)

        logger.debug("  TVM IR conversion complete")
        logger.debug(
            f"  Components: {len(components.anchors)} anchors, {components.num_classes} classes"
        )

        return mod, components

    except Exception as e:
        logger.error(f"  Failed to convert {model_name} to TVM IR: {e}")
        logger.debug("  Traceback:", exc_info=True)
        return None


def run_inference_tvm(
    mod: tvm.IRModule,
    components: DetectionModelComponents,
    image_tensor: torch.Tensor,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    compare_llvm: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run inference using TVM with C Static target for detection models

    Pipeline (similar to od_yolo.py):
    1. Compile and run TVM inference → raw outputs (bbox_deltas, cls_logits)
    2. Apply post-processing (box decoding + NMS)
    3. Return standard detection dict

    Args:
        mod: TVM IRModule (wrapped detection model)
        components: DetectionModelComponents with anchors and metadata
        image_tensor: Input image as PyTorch tensor (3D or 4D)
        score_threshold: Minimum confidence score for detections
        iou_threshold: NMS IoU threshold
        compare_llvm: If True, also compile for LLVM and compare (debugging)

    Returns:
        Dict with 'boxes', 'labels', 'scores' tensors

    Raises:
        RuntimeError: If TVM compilation or execution fails
    """
    from tvm_utils import compile_and_run_on_target

    logger.debug(f"  Compiling and running {components.architecture} model with TVM C Static...")

    # Add batch dimension if needed
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Verify input shape matches expected size
    expected_h, expected_w = components.image_size
    if image_tensor.shape[2:] != (expected_h, expected_w):
        logger.warning(
            f"Input size {image_tensor.shape[2:]} doesn't match expected {components.image_size}. "
            "This may cause anchor mismatch!"
        )

    try:
        # Compile and run on C Static target
        tvm_output = compile_and_run_on_target(
            target_string=C_STATIC_TARGET,
            mod=mod,
            input=image_tensor.numpy(),
            verbose_output=False,
        )

        # TVM returns tuple of outputs:
        # - SSD/RetinaNet: (bbox_deltas, cls_logits)
        # - FCOS: (bbox_distances, cls_logits, bbox_ctrness)
        bbox_ctrness = None

        if isinstance(tvm_output, (list, tuple)):
            if len(tvm_output) == 2:
                bbox_deltas_np, cls_logits_np = tvm_output
            elif len(tvm_output) == 3:
                # FCOS has centerness as 3rd output
                bbox_deltas_np, cls_logits_np, bbox_ctrness_np = tvm_output
                bbox_ctrness = torch.from_numpy(bbox_ctrness_np)
                logger.debug(f"  TVM bbox_ctrness shape: {bbox_ctrness.shape}")
            else:
                raise ValueError(f"Expected 2 or 3 outputs from TVM, got {len(tvm_output)}")
        else:
            raise ValueError("Expected tuple output from TVM")

        # Convert to torch tensors
        bbox_deltas = torch.from_numpy(bbox_deltas_np)
        cls_logits = torch.from_numpy(cls_logits_np)

        logger.debug(f"  TVM bbox_deltas shape: {bbox_deltas.shape}")
        logger.debug(f"  TVM cls_logits shape: {cls_logits.shape}")
        logger.debug(f"  TVM bbox_deltas range: [{bbox_deltas.min():.4f}, {bbox_deltas.max():.4f}]")
        logger.debug(f"  TVM cls_logits range: [{cls_logits.min():.4f}, {cls_logits.max():.4f}]")

        # Apply post-processing (box decoding + NMS)
        detections = apply_detection_postprocessing(
            bbox_deltas=bbox_deltas,
            cls_logits=cls_logits,
            components=components,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            bbox_ctrness=bbox_ctrness,
        )

        logger.debug(f"  Final detections: {len(detections['boxes'])}")

        return detections

    except Exception as e:
        logger.error(f"TVM inference failed: {e}")
        logger.debug("Traceback:", exc_info=True)
        raise RuntimeError(f"TVM inference failed: {e}") from e


def compare_inference_results(
    pytorch_detections: Dict[str, torch.Tensor],
    tvm_detections: Dict[str, torch.Tensor],
    labels: Dict[int, str],
    model_name: Optional[str] = None,
    original_image_size: Optional[Tuple[int, int]] = None,
    preprocessed_image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Compare PyTorch and TVM detection inference results

    Args:
        pytorch_detections: PyTorch detection outputs with 'boxes', 'labels', 'scores'
        tvm_detections: TVM detection outputs with 'boxes', 'labels', 'scores'
        labels: Dictionary mapping class IDs to class names
        model_name: Model name for architecture detection
        original_image_size: Original image size (H, W) - used for scaling
        preprocessed_image_size: Preprocessed image size (H, W) - used for scaling

    Returns:
        Dictionary with comparison metrics
    """
    pytorch_boxes = pytorch_detections["boxes"].detach().numpy()
    pytorch_labels = pytorch_detections["labels"].detach().numpy()
    pytorch_scores = pytorch_detections["scores"].detach().numpy()

    tvm_boxes = tvm_detections["boxes"].detach().numpy()
    tvm_labels = tvm_detections["labels"].detach().numpy()
    tvm_scores = tvm_detections["scores"].detach().numpy()

    # Scale TVM boxes if using SSD preprocessing (different coordinate spaces)
    if (
        model_name
        and original_image_size
        and preprocessed_image_size
        and is_single_stage_detector(model_name)
    ):
        arch = get_detection_architecture(model_name)
        if arch in ["ssd", "ssdlite"]:
            # TVM boxes are in preprocessed image coordinates, PyTorch in original
            # Scale TVM boxes to original image space for comparison
            orig_h, orig_w = original_image_size
            prep_h, prep_w = preprocessed_image_size
            scale_x = orig_w / prep_w
            scale_y = orig_h / prep_h

            if len(tvm_boxes) > 0:
                tvm_boxes[:, [0, 2]] *= scale_x  # Scale x coordinates
                tvm_boxes[:, [1, 3]] *= scale_y  # Scale y coordinates
                logger.debug(
                    f"  Scaled TVM boxes from {preprocessed_image_size} to {original_image_size} "
                    f"(scale: {scale_x:.3f}, {scale_y:.3f})"
                )

    num_pytorch = len(pytorch_boxes)
    num_tvm = len(tvm_boxes)

    logger.info("\nDetection Comparison:")
    logger.info(f"  PyTorch detections: {num_pytorch}")
    logger.info(f"  TVM detections:     {num_tvm}")

    # Try to match detections using IoU and label matching
    matched_pairs = []
    unmatched_pytorch = list(range(num_pytorch))
    unmatched_tvm = list(range(num_tvm))

    # Greedy matching based on IoU
    for i in range(num_pytorch):
        best_iou = 0.0
        best_j = -1

        for j in unmatched_tvm:
            # Only consider if labels match
            if pytorch_labels[i] == tvm_labels[j]:
                iou = compute_iou(tuple(pytorch_boxes[i]), tuple(tvm_boxes[j]))
                if iou > best_iou and iou >= IOU_THRESHOLD:
                    best_iou = iou
                    best_j = j

        if best_j >= 0:
            matched_pairs.append((i, best_j, best_iou))
            unmatched_tvm.remove(best_j)

    # Remove matched indices from unmatched_pytorch
    for i, _, _ in matched_pairs:
        if i in unmatched_pytorch:
            unmatched_pytorch.remove(i)

    num_matched = len(matched_pairs)
    mean_iou = np.mean([iou for _, _, iou in matched_pairs]) if matched_pairs else 0.0

    # Compute score differences for matched detections
    score_diffs = []
    for i, j, _iou in matched_pairs:
        score_diff = abs(pytorch_scores[i] - tvm_scores[j])
        score_diffs.append(score_diff)

    mean_score_diff = np.mean(score_diffs) if score_diffs else 0.0

    comparison = {
        "num_pytorch_detections": num_pytorch,
        "num_tvm_detections": num_tvm,
        "num_matched": num_matched,
        "num_unmatched_pytorch": len(unmatched_pytorch),
        "num_unmatched_tvm": len(unmatched_tvm),
        "mean_iou": mean_iou,
        "mean_score_diff": mean_score_diff,
        "num_detections_match": (num_pytorch == num_tvm),
        "boxes_match": (num_matched == num_pytorch and num_matched == num_tvm),
    }

    logger.info(f"  Matched pairs:      {num_matched}")
    logger.info(f"  Mean IoU:           {mean_iou:.3f}")
    logger.info(f"  Mean score diff:    {mean_score_diff:.3f}")

    # Show unmatched detections
    if unmatched_pytorch:
        logger.debug(f"\n  Unmatched PyTorch detections ({len(unmatched_pytorch)}):")
        for i in unmatched_pytorch[:5]:  # Show first 5
            label_name = labels.get(int(pytorch_labels[i]), f"class_{pytorch_labels[i]}")
            logger.debug(f"    {label_name} ({pytorch_scores[i]:.2f}): {pytorch_boxes[i]}")

    if unmatched_tvm:
        logger.debug(f"\n  Unmatched TVM detections ({len(unmatched_tvm)}):")
        for j in unmatched_tvm[:5]:  # Show first 5
            label_name = labels.get(int(tvm_labels[j]), f"class_{tvm_labels[j]}")
            logger.debug(f"    {label_name} ({tvm_scores[j]:.2f}): {tvm_boxes[j]}")

    return comparison


# Helper functions for main()
def _run_pytorch_inference(
    model: nn.Module, image_tensor: torch.Tensor, score_threshold: float = DEFAULT_SCORE_THRESHOLD
) -> Dict[str, torch.Tensor]:
    """Run PyTorch inference for object detection"""
    logger.info("\nRunning PyTorch inference...")
    return run_inference(model, image_tensor, score_threshold)


def _run_tvm_inference(
    model: nn.Module,
    model_name: str,
    image_tensor: torch.Tensor,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    input_shape: Optional[Tuple[int, ...]] = None,
    compare_llvm: bool = False,
) -> Tuple[Dict[str, torch.Tensor], bool, bool]:
    """Run TVM inference with error handling and status tracking for detection models

    Args:
        model: PyTorch detection model to run
        model_name: Model name (for architecture detection)
        image_tensor: Preprocessed input tensor (3D or 4D)
        score_threshold: Minimum confidence score for detections
        iou_threshold: NMS IoU threshold
        input_shape: Shape for TVM compilation (if None, inferred from image_tensor)
        compare_llvm: Whether to also run LLVM target for comparison

    Returns:
        Tuple of (detections_dict, compile_success, inference_success)

    Raises:
        Exception: If compilation or inference fails (with status info attached)
    """
    compile_success = False
    inference_success = False

    # Add batch dimension if needed for TVM (C, H, W) -> (1, C, H, W)
    if image_tensor.ndim == 3:
        image_tensor_batched = image_tensor.unsqueeze(0)
    else:
        image_tensor_batched = image_tensor

    # If input_shape not provided, infer from image_tensor
    if input_shape is None:
        input_shape = tuple(image_tensor_batched.shape)

    try:
        # prepare_model_for_tvm now returns (mod, components) tuple
        result = prepare_model_for_tvm(model, model_name, input_shape=input_shape)

        if result is None:
            # Model not supported (two-stage or unknown)
            exc = Exception(
                f"Model {model_name} is not supported for TVM export. "
                "Only single-stage detectors (SSD, RetinaNet, FCOS) are supported."
            )
            exc.tvm_compile_success = False  # type: ignore
            exc.tvm_inference_success = False  # type: ignore
            raise exc

        tvm_mod, components = result
        compile_success = True

    except Exception as e:
        logger.error(f"TVM compilation failed: {e}")
        exc = Exception(f"TVM compilation failed: {e}")
        exc.tvm_compile_success = compile_success  # type: ignore
        exc.tvm_inference_success = inference_success  # type: ignore
        raise exc from e

    try:
        # Run TVM inference with post-processing
        detections = run_inference_tvm(
            mod=tvm_mod,
            components=components,
            image_tensor=image_tensor,  # Pass original (may be 3D or 4D)
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            compare_llvm=compare_llvm,
        )
        inference_success = True
        return detections, compile_success, inference_success

    except Exception as e:
        logger.error(f"TVM inference failed: {e}")
        exc = Exception(f"TVM inference failed: {e}")
        exc.tvm_compile_success = compile_success  # type: ignore
        exc.tvm_inference_success = inference_success  # type: ignore
        raise exc from e


def _run_comparison(
    model: nn.Module,
    model_name: str,
    image_tensor: torch.Tensor,
    labels: Dict[int, str],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = NMS_IOU_THRESHOLD,
    pytorch_image_tensor: Optional[torch.Tensor] = None,
    original_image_size: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], bool, bool]:
    """Run both PyTorch and TVM inference and compare detection results

    Args:
        model: PyTorch model
        model_name: Model name for architecture detection
        image_tensor: Preprocessed image for TVM (may use SSD-specific preprocessing)
        labels: COCO label mapping
        score_threshold: Detection confidence threshold
        iou_threshold: NMS IoU threshold for TVM
        pytorch_image_tensor: Optional separate preprocessing for PyTorch. If None, uses image_tensor
        original_image_size: Original image size (H, W) before preprocessing

    Returns:
        Tuple of (detections, comparison, compile_success, inference_success)
    """
    # Use separate preprocessing for PyTorch if provided (for SSD models)
    pytorch_input = pytorch_image_tensor if pytorch_image_tensor is not None else image_tensor

    pytorch_detections = _run_pytorch_inference(model, pytorch_input, score_threshold)
    tvm_detections, compile_success, inference_success = _run_tvm_inference(
        model, model_name, image_tensor, score_threshold, iou_threshold, compare_llvm=False
    )

    # Get preprocessed image size from TVM input
    preprocessed_image_size = (image_tensor.shape[-2], image_tensor.shape[-1])  # (H, W)

    comparison = compare_inference_results(
        pytorch_detections,
        tvm_detections,
        labels,
        model_name=model_name,
        original_image_size=original_image_size,
        preprocessed_image_size=preprocessed_image_size,
    )

    return tvm_detections, comparison, compile_success, inference_success


def _print_comparison_table(comparison_results: List[Dict[str, Any]]) -> None:
    """Print comparison table for multiple object detection models with TVM status"""
    logger.info("\nComparison Table: PyTorch vs TVM C Static")
    logger.info(f"{'-' * 105}")
    logger.info(
        f"{'Model':<30s} {'Det Match':<12s} {'Box Match':<12s} {'Mean IoU':<12s} {'TVM Compile':<15s} {'TVM Inference':<15s}"
    )
    logger.info(f"{'-' * 105}")

    for comp in comparison_results:
        model_name = comp["model"][:28]

        if comp["error"]:
            det_str = "ERROR"
            box_str = "ERROR"
            iou_str = "N/A"
        else:
            det_str = "✓" if comp.get("num_detections_match") else "✗"
            box_str = "✓" if comp.get("boxes_match") else "✗"
            mean_iou = comp.get("mean_iou")
            iou_str = f"{mean_iou:.3f}" if mean_iou is not None else "N/A"

        # TVM status columns
        compile_status = comp.get("tvm_compile_success")
        inference_status = comp.get("tvm_inference_success")

        if compile_status is None:
            compile_str = "N/A"
        else:
            compile_str = "✓" if compile_status else "✗"

        if inference_status is None:
            inference_str = "N/A"
        else:
            inference_str = "✓" if inference_status else "✗"

        logger.info(
            f"{model_name:<30s} {det_str:<12s} {box_str:<12s} {iou_str:<12s} {compile_str:<15s} {inference_str:<15s}"
        )


def _print_comparison_summary(comparison_results: List[Dict[str, Any]]) -> None:
    """Print comparison summary statistics for object detection including TVM status"""
    valid_comps = [c for c in comparison_results if not c["error"]]
    if not valid_comps:
        return

    det_match_failures = sum(1 for c in valid_comps if not c.get("num_detections_match", False))
    box_match_failures = sum(1 for c in valid_comps if not c.get("boxes_match", False))
    total = len(valid_comps)

    # Calculate average IoU
    ious = [c.get("mean_iou", 0.0) for c in valid_comps if c.get("mean_iou") is not None]
    avg_iou = np.mean(ious) if ious else 0.0

    # TVM status statistics
    compile_successes = sum(1 for c in valid_comps if c.get("tvm_compile_success") is True)
    inference_successes = sum(1 for c in valid_comps if c.get("tvm_inference_success") is True)

    logger.info("\nComparison Summary:")
    logger.info(f"  Models tested: {total}")
    logger.info(
        f"  Detection count match rate: {(total - det_match_failures) / total * 100:.1f}% ({total - det_match_failures}/{total})"
    )
    logger.info(
        f"  Box match rate: {(total - box_match_failures) / total * 100:.1f}% ({total - box_match_failures}/{total})"
    )
    logger.info(f"  Average mean IoU: {avg_iou:.3f}")
    logger.info(
        f"  TVM compile success rate: {compile_successes / total * 100:.1f}% ({compile_successes}/{total})"
    )
    logger.info(
        f"  TVM inference success rate: {inference_successes / total * 100:.1f}% ({inference_successes}/{total})"
    )


def main(
    model_name: str = "fasterrcnn_resnet50_fpn",
    weight_name: Optional[str] = None,
    image_url: Optional[str] = None,
    use_tvm: bool = False,
    compare: bool = False,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = NMS_IOU_THRESHOLD,
) -> ModelTestResult:
    """Main function to run the complete object detection pipeline

    Args:
        model_name: Name of the TorchVision detection model
        weight_name: Specific weight name or None for default
        image_url: URL or path to image, or None for default
        use_tvm: Use TVM C Static compilation
        compare: Compare PyTorch vs TVM C Static
        score_threshold: Minimum confidence score for detections
        iou_threshold: NMS IoU threshold (only for TVM)

    Returns:
        ModelTestResult with success=True if successful, success=False if failed
    """
    # Load model and automatically determine preprocessing
    model, preprocessing, weight = load_model_with_preprocessing(model_name, weight_name)

    # Print preprocessing details
    print_preprocessing_details(preprocessing, weight)

    # Load image
    image_url = image_url or DEFAULT_IMAGE_URL
    image = load_image(image_url)
    if image is None:
        return ModelTestResult(
            model=model,
            preprocessing=preprocessing,
            image=None,
            detections=None,
            success=False,
            error_message="Failed to load image",
        )

    # Capture original image size (H, W) for box scaling in comparison
    original_image_size = (image.size[1], image.size[0])  # PIL: (W, H) -> (H, W)

    # Preprocess image
    # For SSD models, we may need different preprocessing for PyTorch vs TVM
    pytorch_image_tensor = None

    if (use_tvm or compare) and is_single_stage_detector(model_name):
        arch = get_detection_architecture(model_name)
        if arch in ["ssd", "ssdlite"]:
            # Use SSD-specific preprocessing for TVM (bypass model's internal transform)
            image_tensor = get_ssd_preprocessing_for_tvm(model)(image)
            logger.debug("  Using SSD-specific preprocessing for TVM")

            # For compare mode, PyTorch needs only ToTensor() because model.forward()
            # applies its own internal transform (resize + normalize). The boxes will be
            # in different coordinate spaces (original vs preprocessed size), which will be
            # handled by scaling during comparison.
            if compare:
                pytorch_image_tensor = transforms.ToTensor()(image)
                logger.debug(
                    "  PyTorch will apply internal transform (boxes will be in original image coordinates)"
                )
        else:
            image_tensor = preprocessing(image)
    else:
        image_tensor = preprocessing(image)

    assert isinstance(image_tensor, torch.Tensor)

    # Load COCO labels
    labels = load_coco_labels()

    # Run inference (TVM or PyTorch or both for comparison)
    comparison_data = None
    tvm_compile_success = None
    tvm_inference_success = None

    try:
        if compare:
            (
                detections_dict,
                comparison_data,
                tvm_compile_success,
                tvm_inference_success,
            ) = _run_comparison(
                model,
                model_name,
                image_tensor,
                labels,
                score_threshold,
                iou_threshold,
                pytorch_image_tensor=pytorch_image_tensor,
                original_image_size=original_image_size,
            )
        elif use_tvm:
            detections_dict, tvm_compile_success, tvm_inference_success = _run_tvm_inference(
                model, model_name, image_tensor, score_threshold, iou_threshold
            )
        else:
            detections_dict = _run_pytorch_inference(model, image_tensor, score_threshold)
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        logger.debug("Traceback:", exc_info=True)

        # Extract TVM status from exception if available
        if use_tvm or compare:
            if hasattr(e, "tvm_compile_success"):
                tvm_compile_success = e.tvm_compile_success  # type: ignore
            if hasattr(e, "tvm_inference_success"):
                tvm_inference_success = e.tvm_inference_success  # type: ignore

        # Determine specific error message based on TVM status
        error_message = str(e)
        if tvm_compile_success is False:
            error_message = "TVM compilation failed"
        elif tvm_compile_success is True and tvm_inference_success is False:
            error_message = "TVM inference failed"

        return ModelTestResult(
            model=model,
            preprocessing=preprocessing,
            image=image,
            detections=None,
            success=False,
            error_message=error_message,
            tvm_compile_success=tvm_compile_success,
            tvm_inference_success=tvm_inference_success,
        )

    # Display detection results
    detections = display_detections(detections_dict, labels, max_display=10)

    # Additional analysis
    logger.info("\nDetection Statistics:")
    logger.info(f"  Total detections: {len(detections)}")
    if detections:
        logger.info(f"  Highest confidence: {detections[0].score * 100:.2f}%")
        logger.info(f"  Lowest confidence: {detections[-1].score * 100:.2f}%")

    return ModelTestResult(
        model=model,
        preprocessing=preprocessing,
        image=image,
        detections=detections,
        comparison=comparison_data,
        tvm_compile_success=tvm_compile_success,
        tvm_inference_success=tvm_inference_success,
    )


def get_all_detection_models(tvm_only=False):
    """Programmatically discover all object detection models in TorchVision

    Args:
        tvm_only: If True, only return models supported by TVM (single-stage detectors).
                  If False, return all detection models.

    Returns:
        List of tuples (model_name, default_weight_name)
    """
    import inspect
    import typing

    detection_model_list = []

    # Common detection model prefixes/patterns in TorchVision
    detection_patterns = [
        "fasterrcnn",
        "retinanet",
        "fcos",
        "ssd",
        "ssdlite",
        "maskrcnn",
        "keypointrcnn",
    ]

    # Get all attributes from torchvision.models.detection
    for name in dir(detection_models):
        # Skip private attributes and non-callables
        if name.startswith("_"):
            continue

        # Filter for detection models by name pattern
        if not any(pattern in name.lower() for pattern in detection_patterns):
            continue

        attr = getattr(detection_models, name)

        # Check if it's a callable (function) and not a Weights enum
        if not callable(attr) or name.endswith("_Weights"):
            continue

        # Get the function signature and check for weights parameter
        try:
            sig = inspect.signature(attr)

            # Check if function has a 'weights' parameter with type annotation
            if "weights" not in sig.parameters:
                continue

            weights_param = sig.parameters["weights"]

            # Extract the weight enum class from the type annotation
            if weights_param.annotation == inspect.Parameter.empty:
                continue

            # Handle Optional[WeightsEnum] annotations
            weights_enum = None
            annotation = weights_param.annotation

            # Check if it's Optional[...]
            if hasattr(typing, "get_args") and hasattr(typing, "get_origin"):
                origin = typing.get_origin(annotation)
                if origin is typing.Union:
                    args = typing.get_args(annotation)
                    # Find the non-None type in Optional (Union[X, None])
                    for arg in args:
                        if (
                            arg is not type(None)
                            and hasattr(arg, "__name__")
                            and arg.__name__.endswith("_Weights")
                        ):
                            weights_enum = arg
                            break

            if weights_enum is None:
                continue

            # Try to get the default weight
            if hasattr(weights_enum, "DEFAULT"):
                default_weight = weights_enum.DEFAULT
            else:
                # Get the first available weight
                available_weights = list(weights_enum)
                if available_weights:
                    default_weight = available_weights[0]
                else:
                    continue

            # For detection models, check if it's a COCO-trained model
            if hasattr(default_weight, "meta") and default_weight.meta:
                meta = default_weight.meta

                # COCO detection models typically have 91 classes (or 80 labeled)
                if "categories" in meta:
                    num_classes = len(meta["categories"])
                    # COCO has 80-91 classes depending on representation
                    if 70 <= num_classes <= 100:
                        detection_model_list.append((name, default_weight.name))
                elif "num_classes" in meta:
                    num_classes = meta["num_classes"]
                    if 70 <= num_classes <= 100:
                        detection_model_list.append((name, default_weight.name))
                else:
                    # If no class info, assume it's a detection model based on name
                    detection_model_list.append((name, default_weight.name))
            else:
                # If no meta, assume it's a detection model based on name
                detection_model_list.append((name, default_weight.name))

        except Exception as e:
            # Skip models that fail inspection
            logger.debug(f"  Skipping {name}: {e}")
            continue

    # Filter for TVM-compatible models if requested
    if tvm_only:
        supported_models = []
        unsupported_models = []
        for model_name, weight_name in detection_model_list:
            if is_single_stage_detector(model_name):
                supported_models.append((model_name, weight_name))
            else:
                unsupported_models.append(model_name)

        if unsupported_models:
            logger.info(
                f"Excluding {len(unsupported_models)} unsupported models for TVM: "
                f"{', '.join(unsupported_models[:5])}"
                + (
                    f" and {len(unsupported_models) - 5} more"
                    if len(unsupported_models) > 5
                    else ""
                )
            )
        return supported_models

    return detection_model_list


def test_multiple_models(
    image_url=None,
    max_models=None,
    model_filter=None,
    use_tvm=False,
    compare=False,
    log_file=None,
    score_threshold=DEFAULT_SCORE_THRESHOLD,
    iou_threshold=NMS_IOU_THRESHOLD,
):
    """Test the automatic preprocessing with all available object detection models

    Args:
        image_url: URL or path to test image. If None, uses default image.
        max_models: Maximum number of models to test. If None, tests all models.
        model_filter: Optional list of model name substrings to filter by (e.g., ['fasterrcnn', 'retinanet'])
        use_tvm: Use TVM compilation with C Static target.
        compare: Compare PyTorch and TVM C Static results.
        log_file: Optional path to CSV log file for appending results.
        score_threshold: Minimum confidence score for detections.
        iou_threshold: NMS IoU threshold for TVM inference.
    """
    logger.debug("\nDiscovering TorchVision object detection models...")

    # Automatically discover all detection models
    # Filter for TVM-compatible models if using TVM
    all_models = get_all_detection_models(tvm_only=(use_tvm or compare))

    # Apply filter if specified
    if model_filter:
        filtered_models = []
        for model_name, weight_name in all_models:
            if any(filter_str in model_name.lower() for filter_str in model_filter):
                filtered_models.append((model_name, weight_name))
        all_models = filtered_models
        logger.debug(f"Filtered to {len(all_models)} models matching {model_filter}")

    # Limit number of models if specified
    if max_models:
        all_models = all_models[:max_models]
        logger.debug(f"Limited to first {max_models} models")

    logger.info(f"\nTesting {len(all_models)} models...\n")

    # Use default test image if not provided
    if image_url is None:
        image_url = DEFAULT_IMAGE_URL

    # Track results
    successful_tests = []
    failed_tests = []
    comparison_results = []  # For compare mode

    for i, (model_name, weight_name) in enumerate(all_models, 1):
        try:
            logger.info(f"\n[{i}/{len(all_models)}] Testing {model_name}...")

            result = main(
                model_name, weight_name, image_url, use_tvm, compare, score_threshold, iou_threshold
            )

            # Check if test failed
            if not result.success:
                failed_tests.append(
                    {
                        "model": model_name,
                        "weight": weight_name,
                        "error": result.error_message or "Unknown error",
                    }
                )
                if compare:
                    comparison_results.append(
                        {
                            "model": model_name,
                            "num_detections_match": None,
                            "boxes_match": None,
                            "mean_iou": None,
                            "tvm_compile_success": result.tvm_compile_success,
                            "tvm_inference_success": result.tvm_inference_success,
                            "error": True,
                        }
                    )

                # Append to log file
                if log_file:
                    append_result_to_log(
                        log_file=log_file,
                        model_name=model_name,
                        error=True,
                        error_message=result.error_message or "Unknown error",
                        tvm_compile_success=result.tvm_compile_success,
                        tvm_inference_success=result.tvm_inference_success,
                    )
                continue

            # Process result (ModelTestResult)
            if compare and result.comparison:
                # Store comparison results with TVM status
                comparison_results.append(
                    {
                        "model": model_name,
                        "num_detections_match": result.comparison.get("num_detections_match"),
                        "boxes_match": result.comparison.get("boxes_match"),
                        "mean_iou": result.comparison.get("mean_iou"),
                        "tvm_compile_success": result.tvm_compile_success,
                        "tvm_inference_success": result.tvm_inference_success,
                        "error": False,
                    }
                )

            # Success path - detections should be populated
            assert result.detections is not None, (
                "Detections should not be None for successful results"
            )
            top_detection = result.detections[0]
            logger.info(
                f"  Result: {len(result.detections)} detections, top: {top_detection.label_name} ({top_detection.score * 100:.1f}%)"
            )

            successful_tests.append(
                {
                    "model": model_name,
                    "weight": weight_name,
                    "num_detections": len(result.detections),
                    "top_detection": top_detection.label_name,
                    "confidence": top_detection.score * 100,
                }
            )

            # Append to log file
            if log_file:
                append_result_to_log(
                    log_file=log_file,
                    model_name=model_name,
                    num_detections=len(result.detections),
                    top_detection=top_detection.label_name,
                    confidence=top_detection.score * 100,
                    num_detections_match=result.comparison.get("num_detections_match")
                    if result.comparison
                    else None,
                    boxes_match=result.comparison.get("boxes_match") if result.comparison else None,
                    mean_iou=result.comparison.get("mean_iou") if result.comparison else None,
                    tvm_compile_success=result.tvm_compile_success,
                    tvm_inference_success=result.tvm_inference_success,
                    error=False,
                )

        except Exception as e:
            logger.error(f"  Failed: {e}")
            failed_tests.append({"model": model_name, "weight": weight_name, "error": str(e)})
            if compare:
                comparison_results.append(
                    {
                        "model": model_name,
                        "num_detections_match": None,
                        "boxes_match": None,
                        "mean_iou": None,
                        "tvm_compile_success": None,
                        "tvm_inference_success": None,
                        "error": True,
                    }
                )

            # Append to log file
            if log_file:
                append_result_to_log(
                    log_file=log_file, model_name=model_name, error=True, error_message=str(e)
                )

    # Print comparison table if in compare mode
    if compare and comparison_results:
        _print_comparison_table(comparison_results)
        _print_comparison_summary(comparison_results)

    # Print summary
    logger.info("\nSummary")
    logger.info(f"{'-' * 60}")
    logger.info(
        f"Total: {len(all_models)} | Successful: {len(successful_tests)} | Failed: {len(failed_tests)}"
    )

    # Detailed successful tests at DEBUG level
    if successful_tests:
        logger.debug(f"\n{'-' * 60}")
        logger.debug("Successful Tests:")
        logger.debug(f"{'-' * 60}")
        for test in successful_tests:
            det = test.get("top_detection", "N/A")[:35]
            num_dets = test.get("num_detections", 0)
            logger.debug(
                f"  ✓ {test['model']:25s} -> {num_dets} dets, top: {det:35s} ({test['confidence']:5.1f}%)"
            )

    # Failed tests always shown at INFO level
    if failed_tests:
        logger.info(f"\n{'-' * 60}")
        logger.info("Failed Tests:")
        logger.info(f"{'-' * 60}")
        for test in failed_tests:
            error = str(test["error"])[:50]
            logger.info(f"  ✗ {test['model']:25s} -> {error}")

    return successful_tests, failed_tests


def test_multiple_models_parallel(
    image_url=None,
    max_models=None,
    model_filter=None,
    use_tvm=False,
    compare=False,
    max_workers=None,
    log_file=None,
    score_threshold=DEFAULT_SCORE_THRESHOLD,
    iou_threshold=NMS_IOU_THRESHOLD,
):
    """Test multiple object detection models in parallel using concurrent.futures

    Args:
        image_url: URL or path to test image. If None, uses default image.
        max_models: Maximum number of models to test. If None, tests all models.
        model_filter: Optional list of model name substrings to filter by
        use_tvm: Use TVM compilation with C Static target.
        compare: Compare PyTorch and TVM C Static results.
        max_workers: Maximum number of parallel workers. If None, uses CPU count.
        log_file: Optional path to CSV log file for appending results.
        score_threshold: Minimum confidence score for detections.
        iou_threshold: NMS IoU threshold for TVM inference.

    Returns:
        Tuple of (successful_tests, failed_tests)
    """
    import os

    logger.debug("\nDiscovering TorchVision object detection models...")

    # Automatically discover all detection models
    # Filter for TVM-compatible models if using TVM
    all_models = get_all_detection_models(tvm_only=(use_tvm or compare))

    # Apply filter if specified
    if model_filter:
        filtered_models = []
        for model_name, weight_name in all_models:
            if any(filter_str in model_name.lower() for filter_str in model_filter):
                filtered_models.append((model_name, weight_name))
        all_models = filtered_models
        logger.debug(f"Filtered to {len(all_models)} models matching {model_filter}")

    # Limit number of models if specified
    if max_models:
        all_models = all_models[:max_models]
        logger.debug(f"Limited to first {max_models} models")

    # Use default test image if not provided
    if image_url is None:
        image_url = DEFAULT_IMAGE_URL

    # Determine number of workers
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(all_models))

    logger.info(f"\nTesting {len(all_models)} models in parallel with {max_workers} workers...\n")

    # Track results
    successful_tests = []
    failed_tests = []
    comparison_results = []

    # Submit all tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create futures mapping
        future_to_model = {
            executor.submit(
                main,
                model_name,
                weight_name,
                image_url,
                use_tvm,
                compare,
                score_threshold,
                iou_threshold,
            ): (
                model_name,
                weight_name,
            )
            for model_name, weight_name in all_models
        }

        # Process completed tasks with progress tracking
        completed = 0
        for future in as_completed(future_to_model):
            model_name, weight_name = future_to_model[future]
            completed += 1

            try:
                result = future.result()

                if not result.success:
                    # Test failed
                    failed_tests.append(
                        {
                            "model": model_name,
                            "weight": weight_name,
                            "error": result.error_message or "Unknown error",
                            "tvm_compile_success": result.tvm_compile_success,
                            "tvm_inference_success": result.tvm_inference_success,
                        }
                    )
                    logger.warning(
                        f"[{completed}/{len(all_models)}] ✗ {model_name} - {result.error_message}"
                    )

                    if compare:
                        comparison_results.append(
                            {
                                "model": model_name,
                                "num_detections_match": None,
                                "boxes_match": None,
                                "mean_iou": None,
                                "tvm_compile_success": result.tvm_compile_success,
                                "tvm_inference_success": result.tvm_inference_success,
                                "error": True,
                            }
                        )

                    # Append to log file
                    if log_file:
                        append_result_to_log(
                            log_file=log_file,
                            model_name=model_name,
                            error=True,
                            error_message=result.error_message or "Unknown error",
                            tvm_compile_success=result.tvm_compile_success,
                            tvm_inference_success=result.tvm_inference_success,
                        )
                else:
                    # Success - detections should be populated
                    assert result.detections is not None, (
                        "Detections should not be None for successful results"
                    )
                    top_det = result.detections[0]
                    successful_tests.append(
                        {
                            "model": model_name,
                            "weight": weight_name,
                            "num_detections": len(result.detections),
                            "top_detection": top_det.label_name,
                            "confidence": top_det.score * 100,
                            "tvm_compile_success": result.tvm_compile_success,
                            "tvm_inference_success": result.tvm_inference_success,
                        }
                    )

                    # Format status indicators
                    status_str = ""
                    if result.tvm_compile_success is not None:
                        compile_mark = "✓" if result.tvm_compile_success else "✗"
                        infer_mark = "✓" if result.tvm_inference_success else "✗"
                        status_str = f" [Compile:{compile_mark} Infer:{infer_mark}]"

                    logger.info(
                        f"[{completed}/{len(all_models)}] ✓ {model_name} - "
                        f"{len(result.detections)} dets, {top_det.label_name[:30]} ({top_det.score * 100:.1f}%){status_str}"
                    )

                    if compare and result.comparison:
                        comparison_results.append(
                            {
                                "model": model_name,
                                "num_detections_match": result.comparison.get(
                                    "num_detections_match"
                                ),
                                "boxes_match": result.comparison.get("boxes_match"),
                                "mean_iou": result.comparison.get("mean_iou"),
                                "tvm_compile_success": result.tvm_compile_success,
                                "tvm_inference_success": result.tvm_inference_success,
                                "error": False,
                            }
                        )

                    # Append to log file
                    if log_file:
                        append_result_to_log(
                            log_file=log_file,
                            model_name=model_name,
                            num_detections=len(result.detections),
                            top_detection=top_det.label_name,
                            confidence=top_det.score * 100,
                            num_detections_match=result.comparison.get("num_detections_match")
                            if result.comparison
                            else None,
                            boxes_match=result.comparison.get("boxes_match")
                            if result.comparison
                            else None,
                            mean_iou=result.comparison.get("mean_iou")
                            if result.comparison
                            else None,
                            tvm_compile_success=result.tvm_compile_success,
                            tvm_inference_success=result.tvm_inference_success,
                            error=False,
                        )

            except BrokenExecutor as e:
                # Worker process crashed - log it and stop gracefully
                error_msg = "Worker process crashed (likely OOM, segfault, or killed)"
                logger.error(f"[{completed}/{len(all_models)}] ✗ {model_name} - {error_msg}")
                logger.error(f"  Reason: {e}")
                logger.warning(
                    f"  ProcessPoolExecutor is broken. Stopping remaining {len(all_models) - completed} models."
                )

                failed_tests.append(
                    {
                        "model": model_name,
                        "weight": weight_name,
                        "error": error_msg,
                        "tvm_compile_success": None,
                        "tvm_inference_success": None,
                    }
                )

                if compare:
                    comparison_results.append(
                        {
                            "model": model_name,
                            "num_detections_match": None,
                            "boxes_match": None,
                            "mean_iou": None,
                            "tvm_compile_success": None,
                            "tvm_inference_success": None,
                            "error": True,
                        }
                    )

                # Append to log file
                if log_file:
                    append_result_to_log(
                        log_file=log_file,
                        model_name=model_name,
                        error=True,
                        error_message=f"{error_msg}: {e}",
                    )

                # Break out of the loop since executor is broken
                break

            except Exception as e:
                # Other exceptions during processing (not a broken executor)
                error_msg = str(e)
                exception_type = type(e).__name__

                # Detect if this might be a process crash even if not caught as BrokenExecutor
                if "process pool" in error_msg.lower() or "abruptly" in error_msg.lower():
                    error_msg = f"Worker process crashed: {exception_type}: {error_msg}"
                    logger.error(f"[{completed}/{len(all_models)}] ✗ {model_name} - Worker crashed")
                else:
                    logger.error(
                        f"[{completed}/{len(all_models)}] ✗ {model_name} - {exception_type}: {error_msg}"
                    )

                failed_tests.append(
                    {
                        "model": model_name,
                        "weight": weight_name,
                        "error": error_msg,
                        "tvm_compile_success": None,
                        "tvm_inference_success": None,
                    }
                )

                if compare:
                    comparison_results.append(
                        {
                            "model": model_name,
                            "num_detections_match": None,
                            "boxes_match": None,
                            "mean_iou": None,
                            "tvm_compile_success": None,
                            "tvm_inference_success": None,
                            "error": True,
                        }
                    )

                # Append to log file
                if log_file:
                    append_result_to_log(
                        log_file=log_file,
                        model_name=model_name,
                        error=True,
                        error_message=f"{exception_type}: {error_msg}",
                    )

    # Print comparison table if in compare mode
    if compare and comparison_results:
        _print_comparison_table(comparison_results)
        _print_comparison_summary(comparison_results)

    # Print summary
    logger.info("\nSummary")
    logger.info(f"{'-' * 60}")
    logger.info(
        f"Total: {len(all_models)} | Successful: {len(successful_tests)} | Failed: {len(failed_tests)}"
    )

    # Detailed successful tests at DEBUG level
    if successful_tests:
        logger.debug(f"\n{'-' * 60}")
        logger.debug("Successful Tests:")
        logger.debug(f"{'-' * 60}")
        for test in successful_tests:
            det = test.get("top_detection", "N/A")[:35]
            num_dets = test.get("num_detections", 0)
            logger.debug(
                f"  ✓ {test['model']:25s} -> {num_dets} dets, top: {det:35s} ({test['confidence']:5.1f}%)"
            )

    # Failed tests always shown at INFO level
    if failed_tests:
        logger.info(f"\n{'-' * 60}")
        logger.info("Failed Tests:")
        logger.info(f"{'-' * 60}")
        for test in failed_tests:
            error = str(test["error"])[:50]
            logger.info(f"  ✗ {test['model']:25s} -> {error}")

    return successful_tests, failed_tests


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file

    Args:
        config_path: Path to YAML config file. If None, returns defaults.

    Returns:
        Dictionary of configuration values
    """
    defaults = {
        "model": "fasterrcnn_resnet50_fpn",
        "image": DEFAULT_IMAGE_URL,
        "verbose": False,
        "use_tvm": False,
        "compare": False,
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
    }

    if config_path and Path(config_path).exists():
        import yaml  # Import here to satisfy type checker

        with open(config_path) as f:
            config = yaml.safe_load(f)
            defaults.update(config)

    return defaults


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchVision Object Detection Model Tester")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument(
        "--model",
        type=str,
        default="fasterrcnn_resnet50_fpn",
        help="Model name to test (default: fasterrcnn_resnet50_fpn)",
    )
    parser.add_argument(
        "--weight", type=str, default=None, help="Weight name to use (default: None, uses DEFAULT)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE_URL,
        help="Path or URL to test image",
    )
    parser.add_argument(
        "--test-all", action="store_true", help="Test all available object detection models"
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Maximum number of models to test when using --test-all",
    )
    parser.add_argument(
        "--filter",
        type=str,
        nargs="+",
        default=None,
        help="Filter models by name (e.g., --filter resnet efficientnet)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (default for single model tests)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Enable quiet mode (minimal output)"
    )
    parser.add_argument(
        "--tvm",
        action="store_true",
        help="Use TVM compilation with C Static target (compares LLVM vs C Static)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both PyTorch and TVM, then compare results (implies --tvm)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (only with --test-all, much faster)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count, only with --parallel)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to CSV log file for appending results (only with --test-all)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f"Minimum confidence score for detections (default: {DEFAULT_SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=NMS_IOU_THRESHOLD,
        help=f"NMS IoU threshold for TVM inference (default: {NMS_IOU_THRESHOLD})",
    )

    args = parser.parse_args()

    # Load config file if specified
    if args.config:
        config = load_config(args.config)
        # Command-line args override config file
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)

    # Setup logging based on verbosity
    setup_logging(args.verbose, args.quiet)

    # If compare is set, ensure tvm is also set
    if args.compare:
        args.tvm = True

    if args.test_all:
        # Test multiple models (parallel or sequential)
        if args.parallel:
            test_multiple_models_parallel(
                image_url=args.image,
                max_models=args.max_models,
                model_filter=args.filter,
                use_tvm=args.tvm,
                compare=args.compare,
                max_workers=args.workers,
                log_file=args.log_file,
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold,
            )
        else:
            test_multiple_models(
                image_url=args.image,
                max_models=args.max_models,
                model_filter=args.filter,
                use_tvm=args.tvm,
                compare=args.compare,
                log_file=args.log_file,
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold,
            )
    else:
        # Test single model
        result = main(
            model_name=args.model,
            weight_name=args.weight,
            image_url=args.image,
            use_tvm=args.tvm,
            compare=args.compare,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
        )
