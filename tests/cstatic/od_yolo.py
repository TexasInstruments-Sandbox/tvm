#!/usr/bin/env python
"""YOLO Object Detection Model Tester (YOLOv5, YOLOv8, and YOLOv11)

This script provides comprehensive testing and validation for YOLO object detection models
supporting YOLOv5 (via torch.hub), YOLOv8, and YOLOv11 (via ultralytics package).

Features:
    - Automatic loading of YOLOv5, YOLOv8, and YOLOv11 models
    - PyTorch inference (default)
    - TVM C Static compilation and inference (--tvm)
    - Side-by-side comparison of PyTorch vs TVM results (--compare)
    - Batch testing of multiple YOLO variants
    - Comprehensive logging with adjustable verbosity levels
    - Bounding box visualization and detection metrics

Usage Examples:
    # Test YOLOv5
    python od_yolo.py --model yolov5s

    # Test YOLOv8
    python od_yolo.py --model yolov8n

    # Test YOLOv11
    python od_yolo.py --model yolo11n

    # Test with TVM compilation (YOLOv5 only)
    python od_yolo.py --model yolov5s --tvm

    # Compare PyTorch vs TVM
    python od_yolo.py --model yolov5s --compare

    # Test with custom confidence threshold
    python od_yolo.py --model yolov5m --score-threshold 0.3

    # Test multiple models
    python od_yolo.py --test-all

    # Verbose mode
    python od_yolo.py --model yolo11s --verbose

Command-Line Options:
    --model MODEL              YOLO model variant (default: yolov5s)
    --image IMAGE              Path or URL to test image
    --test-all                 Test all available YOLO models
    --max-models N             Maximum number of models to test with --test-all
    --filter PATTERN [...]     Filter models by name patterns
    --parallel                 Run tests in parallel (only with --test-all)
    --workers N                Number of parallel workers (default: CPU count)
    --log-file PATH            CSV log file for appending results (with --test-all)
    --tvm                      Use TVM compilation with C Static target
    --compare                  Compare PyTorch vs TVM results
    --score-threshold FLOAT    Minimum confidence score for detections (default: 0.25)
    --verbose, -v              Enable verbose output with detailed logging
    --quiet, -q                Enable quiet mode (minimal output)

Supported Models:
    YOLOv5 (via torch.hub):
        - yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
        - yolov5n6, yolov5s6, yolov5m6, yolov5l6, yolov5x6 (P6 1280px)

    YOLOv8 (via ultralytics):
        - yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

    YOLOv11 (via ultralytics):
        - yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

Note on TVM Support:
    TVM compilation extracts the core YOLO model without pre/post-processing wrappers.
    The compiled model produces raw detection outputs (no NMS) which are post-processed
    in Python. Currently, only YOLOv5 successfully compiles with TVM C Static backend.
"""

import argparse
import csv
import logging
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
import torch.nn as nn
import tvm
from PIL import Image
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

from tvm_utils import compile_and_run_on_target, process_relax

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_IMAGE_URL = "test_images/bird_0.jpg"
DEFAULT_SCORE_THRESHOLD = 0.25  # YOLO default confidence threshold
DEFAULT_IOU_THRESHOLD = 0.45  # YOLO default NMS IoU threshold
IOU_THRESHOLD = 0.5  # For box matching in comparisons
DEFAULT_INPUT_SHAPE = (1, 3, 640, 640)  # YOLOv5 default input size
C_STATIC_TARGET = "c_static"
LLVM_TARGET = "llvm"

# Thread lock for CSV file writing
_CSV_WRITE_LOCK = threading.Lock()

# Available YOLO models
YOLOV5_MODELS = [
    "yolov5n",  # Nano
    "yolov5s",  # Small
    "yolov5m",  # Medium
    "yolov5l",  # Large
    "yolov5x",  # Extra Large
    "yolov5n6",  # Nano P6 (1280px)
    "yolov5s6",  # Small P6
    "yolov5m6",  # Medium P6
    "yolov5l6",  # Large P6
    "yolov5x6",  # Extra Large P6
]

YOLOV8_MODELS = [
    "yolov8n",  # Nano
    "yolov8s",  # Small
    "yolov8m",  # Medium
    "yolov8l",  # Large
    "yolov8x",  # Extra Large
]

YOLOV11_MODELS = [
    "yolo11n",  # Nano
    "yolo11s",  # Small
    "yolo11m",  # Medium
    "yolo11l",  # Large
    "yolo11x",  # Extra Large
]

YOLO_MODELS = YOLOV5_MODELS + YOLOV8_MODELS + YOLOV11_MODELS


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
    image: Optional[Image.Image]
    detections: Optional[List[Detection]]
    success: bool = True
    error_message: Optional[str] = None
    comparison: Optional[Dict[str, Any]] = None
    tvm_compile_success: Optional[bool] = None
    tvm_inference_success: Optional[bool] = None


@dataclass
class ComparisonResult:
    """Results from comparing PyTorch vs TVM detection outputs"""

    model_name: str
    num_detections_match: Optional[bool]
    boxes_match: Optional[bool]
    mean_iou: Optional[float]
    error: bool = False
    error_message: Optional[str] = None


def setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure logging based on verbosity flags"""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(level=level, format="%(message)s")


def append_result_to_log(
    log_file: str,
    model_name: str,
    num_detections: Optional[int] = None,
    top_detection: Optional[str] = None,
    confidence: Optional[float] = None,
    error: bool = False,
    error_message: Optional[str] = None,
) -> None:
    """Append a detection test result to the CSV log file (thread-safe)"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = [
        timestamp,
        model_name,
        str(num_detections) if num_detections is not None else "N/A",
        top_detection or "N/A",
        f"{confidence:.2f}" if confidence is not None else "N/A",
        "Yes" if error else "No",
        error_message or "",
    ]

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
                        "num_detections",
                        "top_detection",
                        "confidence",
                        "error",
                        "error_message",
                    ]
                    writer.writerow(header)
                    logger.debug(f"Created new log file: {log_file}")

                writer.writerow(row)

        except Exception as e:
            logger.error(f"Failed to write to log file {log_file}: {e}")


def detect_yolo_version(model_name: str) -> str:
    """Detect YOLO version from model name

    Args:
        model_name: Model name (e.g., 'yolov5s', 'yolov8n', 'yolo11n')

    Returns:
        Version string: 'v5', 'v8', or 'v11'
    """
    if model_name.startswith("yolov5"):
        return "v5"
    elif model_name.startswith("yolov8") or model_name.startswith("yolo8"):
        return "v8"
    elif model_name.startswith("yolo11"):
        return "v11"
    else:
        # Try to infer from name pattern
        if "v5" in model_name or model_name.endswith("5"):
            return "v5"
        elif "v8" in model_name or model_name.endswith("8"):
            return "v8"
        elif "v11" in model_name or model_name.endswith("11"):
            return "v11"
        else:
            logger.warning(f"Cannot detect YOLO version from '{model_name}', assuming v5")
            return "v5"


def load_yolo_model(model_name: str = "yolov5s") -> Tuple[Any, str]:
    """Load YOLO model (YOLOv5, YOLOv8, or YOLOv11)

    Args:
        model_name: YOLO model variant (e.g., 'yolov5s', 'yolov8n', 'yolo11n')

    Returns:
        Tuple of (model, version) where version is 'v5', 'v8', or 'v11'
    """
    version = detect_yolo_version(model_name)

    if version == "v5":
        model = load_yolov5_model(model_name)
    elif version == "v8":
        model = load_yolov8_model(model_name)
    else:  # v11
        model = load_yolov11_model(model_name)

    return model, version


def load_yolov8_model(model_name: str = "yolov8n") -> Any:
    """Load YOLOv8 model from ultralytics package

    Args:
        model_name: YOLOv8 model variant (e.g., 'yolov8n', 'yolov8s', 'yolov8m')

    Returns:
        Loaded YOLOv8 model in eval mode (with YOLO wrapper)
    """
    logger.debug(f"Loading YOLOv8 model: {model_name}")

    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]

        # Load model from ultralytics package
        model = YOLO(f"{model_name}.pt")
        model.model.eval()  # type: ignore[attr-defined]

        logger.debug("  Model loaded successfully")
        logger.debug(f"  Classes: {len(model.names)}")

        return model

    except ImportError as e:
        logger.error(f"ultralytics package not installed: {e}")
        logger.error("Install with: pip install ultralytics")
        raise
    except Exception as e:
        logger.error(f"Failed to load YOLOv8 model '{model_name}': {e}")
        raise


def load_yolov5_model(model_name: str = "yolov5s") -> Any:
    """Load YOLOv5 model from torch.hub

    Args:
        model_name: YOLOv5 model variant (e.g., 'yolov5s', 'yolov5m', 'yolov5l')

    Returns:
        Loaded YOLOv5 model in eval mode (with AutoShape wrapper)
    """
    logger.debug(f"Loading YOLOv5 model: {model_name}")

    try:
        # Load model from ultralytics/yolov5 via torch.hub
        # Type is Any because torch.hub.load returns a dynamic AutoShape wrapper
        model = torch.hub.load("ultralytics/yolov5", model_name, pretrained=True)
        model.eval()  # type: ignore[attr-defined]

        logger.debug("  Model loaded successfully")
        logger.debug(f"  Classes: {len(model.names)}")  # type: ignore[attr-defined]

        return model

    except Exception as e:
        logger.error(f"Failed to load YOLOv5 model '{model_name}': {e}")
        raise


def load_yolov11_model(model_name: str = "yolo11n") -> Any:
    """Load YOLOv11 model from ultralytics package

    Args:
        model_name: YOLOv11 model variant (e.g., 'yolo11n', 'yolo11s', 'yolo11m')

    Returns:
        Loaded YOLOv11 model in eval mode (with YOLO wrapper)
    """
    logger.debug(f"Loading YOLOv11 model: {model_name}")

    try:
        from ultralytics import YOLO  # type: ignore[attr-defined]

        # Load model from ultralytics package
        # Model file will be downloaded if not present
        model = YOLO(f"{model_name}.pt")
        model.model.eval()  # type: ignore[attr-defined]

        logger.debug("  Model loaded successfully")
        logger.debug(f"  Classes: {len(model.names)}")

        return model

    except ImportError as e:
        logger.error(f"ultralytics package not installed: {e}")
        logger.error("Install with: pip install ultralytics")
        raise
    except Exception as e:
        logger.error(f"Failed to load YOLOv11 model '{model_name}': {e}")
        raise


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
            response = requests.get(image_path_or_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")

        logger.debug(f"  Size: {image.size}")
        return image

    except (requests.RequestException, OSError) as e:
        logger.error(f"Error loading image: {e}")
        return None


def run_inference(
    model: Any,
    image: Image.Image,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    version: str = "v5",
) -> Dict[str, torch.Tensor]:
    """Run YOLO inference on image (supports v5 and v11)

    Args:
        model: YOLO model (YOLOv5 with AutoShape or YOLOv11 with YOLO wrapper)
        image: PIL Image
        score_threshold: Minimum confidence score for detections
        iou_threshold: NMS IoU threshold
        version: YOLO version ('v5' or 'v11')

    Returns:
        Dictionary with 'boxes', 'labels', 'scores' tensors
    """
    # Set model thresholds
    model.conf = score_threshold  # type: ignore[attr-defined]
    model.iou = iou_threshold  # type: ignore[attr-defined]

    # Run inference
    with torch.no_grad():
        if version == "v5":
            # YOLOv5 AutoShape doesn't support verbose parameter
            results = model(image)
        else:
            # YOLOv8/v11 support verbose parameter
            results = model(image, verbose=False)

    # Extract detections based on version
    if version == "v5":
        # YOLOv5: results.xyxy[0] contains [x1, y1, x2, y2, confidence, class]
        detections_tensor = results.xyxy[0]  # Tensor of shape (N, 6)
    else:  # v11
        # YOLOv11: results is a list, first element has .boxes attribute
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            if hasattr(result, "boxes") and len(result.boxes) > 0:
                # boxes.data contains [x1, y1, x2, y2, confidence, class]
                detections_tensor = result.boxes.data
            else:
                detections_tensor = torch.zeros((0, 6), dtype=torch.float32)
        else:
            detections_tensor = torch.zeros((0, 6), dtype=torch.float32)

    if len(detections_tensor) == 0:
        # No detections
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,), dtype=torch.float32),
        }

    # Parse detections (same format for both versions)
    boxes = detections_tensor[:, :4]  # x1, y1, x2, y2
    scores = detections_tensor[:, 4]  # confidence
    labels = detections_tensor[:, 5].long()  # class id

    return {
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
    }


def display_detections(
    detections: Dict[str, torch.Tensor],
    model: Any,
    max_display: int = 10,
) -> List[Detection]:
    """Display detected objects

    Args:
        detections: Dictionary with 'boxes', 'labels', 'scores' tensors
        model: YOLOv5 model (contains class names)
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
        label_name = str(model.names[label_id])  # type: ignore[attr-defined]
        score = scores[i].item()

        logger.info(
            f"{i + 1:2d}. {label_name:20s} {score * 100:6.2f}% "
            f"Box: [{box[0]:6.1f}, {box[1]:6.1f}, {box[2]:6.1f}, {box[3]:6.1f}]"
        )

        det = Detection(
            box=(box[0], box[1], box[2], box[3]),
            label=label_id,
            label_name=label_name,
            score=score,
        )
        results.append(det)

    if num_detections > max_display:
        logger.info(f"... and {num_detections - max_display} more detections")

    return results


def compute_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
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


class YOLOWrapper(nn.Module):
    """Wrapper to make YOLO models (v5 and v11) compatible with torch.export

    Both YOLOv5 and YOLOv11 include preprocessing/postprocessing wrappers that have
    dynamic operations. This wrapper extracts the core model for static graph export.

    YOLOv5: Extracts model from AutoShape wrapper
    YOLOv11: Extracts model from YOLO predictor wrapper
    """

    def __init__(self, yolo_model, version: str = "v5"):
        super().__init__()
        self.version = version

        # Extract the core model without wrappers
        if hasattr(yolo_model, "model"):
            self.model = yolo_model.model
        else:
            self.model = yolo_model

        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns raw model output

        Args:
            x: Input tensor of shape [batch, 3, height, width]

        Returns:
            Raw output tensor from YOLO model (before NMS)
            YOLOv5: [batch, num_predictions, 85]  - [x,y,w,h, objectness, classes...]
            YOLOv11: [batch, 84, num_predictions] - [x,y,w,h, classes...]
        """
        output = self.model(x)

        # Both v5 and v11 may return tuples
        # Return the first output (the predictions)
        if isinstance(output, (list, tuple)):
            return output[0]
        return output


# Alias for backward compatibility
YOLOv5Wrapper = YOLOWrapper


def torch_to_relax(torch_model: nn.Module, example_input: Tuple[torch.Tensor, ...]) -> tvm.IRModule:
    """Convert a PyTorch model to a Relax IRModule

    Args:
        torch_model: PyTorch model to convert
        example_input: Example input tuple for torch.export

    Returns:
        TVM Relax IRModule
    """
    with torch.no_grad():
        # Use strict=False for better compatibility with YOLOv8/v11
        # This allows some dynamic operations that can be resolved at compile time
        exported_program = export(torch_model, example_input, strict=False)
        mod = from_exported_program(exported_program, keep_params_as_input=True)
    return mod


def prepare_model_for_tvm(
    torch_model: nn.Module, input_shape: Tuple[int, ...] = DEFAULT_INPUT_SHAPE, version: str = "v5"
) -> tvm.IRModule:
    """Prepare a YOLO model (v5 or v11) for TVM compilation

    Args:
        torch_model: YOLO model in eval mode
        input_shape: Shape of input tensor (batch, channels, height, width)
        version: YOLO version ('v5' or 'v11')

    Returns:
        Processed TVM IRModule ready for compilation

    Raises:
        ValueError: If there are issues with model conversion
    """
    logger.debug(f"  Converting YOLO{version} to TVM Relax IR...")

    # Wrap the model for export compatibility
    wrapped_model = YOLOWrapper(torch_model, version=version)

    # Create example input for torch.export
    example_input = (torch.randn(*input_shape, dtype=torch.float32),)

    # Convert to Relax IRModule
    mod = torch_to_relax(wrapped_model, example_input)

    # Process the module (detach and bind parameters)
    mod = process_relax(mod)

    logger.debug("  TVM IR conversion complete")

    return mod


def apply_nms_postprocessing(
    raw_output: torch.Tensor,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    version: str = "v5",
) -> Dict[str, torch.Tensor]:
    """Apply NMS post-processing to raw YOLO output

    Args:
        raw_output: Raw tensor output from TVM inference
                   YOLOv5:  [batch, 25200, 85]  - [x,y,w,h, objectness, classes...]
                   YOLOv11: [batch, 84, 8400]   - [x,y,w,h, classes...] (transposed)
        score_threshold: Minimum confidence score
        iou_threshold: NMS IoU threshold
        version: YOLO version ('v5' or 'v11')

    Returns:
        Dictionary with 'boxes', 'labels', 'scores' tensors
    """
    from torchvision.ops import nms

    logger.debug(f"Raw TVM output shape: {raw_output.shape}")
    logger.debug(f"Raw output min: {raw_output.min():.4f}, max: {raw_output.max():.4f}")

    # Handle version-specific format differences
    if version in ["v8", "v11"]:
        # YOLOv8/v11: [batch, 84, 8400] -> transpose to [batch, 8400, 84]
        raw_output = raw_output.permute(0, 2, 1)

    # Remove batch dimension
    if raw_output.ndim == 3:
        raw_output = raw_output[0]

    # Extract components (version-specific)
    boxes_xywh = raw_output[:, :4]  # [num_predictions, 4]

    if version == "v5":
        # YOLOv5: [x, y, w, h, objectness, class_0...class_79]
        objectness = raw_output[:, 4]
        class_scores = raw_output[:, 5:]
        class_conf, class_pred = class_scores.max(1)
        conf = objectness * class_conf  # Combined confidence
    elif version in ["v8", "v11"]:
        # YOLOv8/v11: [x, y, w, h, class_0...class_79] (no separate objectness)
        # Both v8 and v11 use the same output format
        class_scores = raw_output[:, 4:]

        # Check if class scores are raw logits (values > 1.0) or already probabilities
        # PyTorch models may export with sigmoid already applied
        if class_scores.max() > 1.0:
            # Raw logits - apply sigmoid to convert to probabilities
            logger.debug("Applying sigmoid to class scores (detected raw logits)")
            class_scores = torch.sigmoid(class_scores)

        conf, class_pred = class_scores.max(1)  # Direct confidence
    else:
        raise ValueError(f"Unsupported YOLO version: {version}")

    # Convert from xywh (center format) to xyxy (corner format)
    x_center, y_center, width, height = boxes_xywh.unbind(1)
    boxes_xyxy = torch.stack(
        [
            x_center - width / 2,  # x1
            y_center - height / 2,  # y1
            x_center + width / 2,  # x2
            y_center + height / 2,  # y2
        ],
        dim=1,
    )

    logger.debug(f"Confidence range: [{conf.min():.4f}, {conf.max():.4f}]")
    logger.debug(f"Detections above threshold {score_threshold}: {(conf >= score_threshold).sum()}")

    # Filter by confidence threshold
    mask = conf >= score_threshold
    boxes_filtered = boxes_xyxy[mask]
    scores_filtered = conf[mask]
    labels_filtered = class_pred[mask]

    if len(boxes_filtered) == 0:
        logger.debug("No detections above confidence threshold")
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,), dtype=torch.float32),
        }

    # Apply NMS
    keep_indices = nms(boxes_filtered, scores_filtered, iou_threshold)

    logger.debug(f"After NMS: {len(keep_indices)} detections")

    return {
        "boxes": boxes_filtered[keep_indices],
        "labels": labels_filtered[keep_indices],
        "scores": scores_filtered[keep_indices],
    }


def run_inference_tvm(
    mod: tvm.IRModule,
    image_tensor: torch.Tensor,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    compare_llvm: bool = True,
    original_image_size: Optional[Tuple[int, int]] = None,
    version: str = "v5",
) -> Dict[str, torch.Tensor]:
    """Run inference using TVM with C Static target for YOLO models

    Args:
        mod: TVM IRModule to execute
        image_tensor: Input image as PyTorch tensor (3D or 4D)
        score_threshold: Minimum confidence score for detections
        iou_threshold: NMS IoU threshold
        compare_llvm: If True, also compile for LLVM and compare results
        original_image_size: Original image size (height, width) for coordinate scaling
        version: YOLO version ('v5' or 'v11')

    Returns:
        Dict with 'boxes', 'labels', 'scores' tensors (in original image coordinates)

    Raises:
        RuntimeError: If TVM compilation or execution fails
    """
    logger.debug("  Compiling with TVM C Static backend...")

    # Add batch dimension if needed
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Ensure correct shape (640x640 for YOLO)
    if image_tensor.shape[2:] != (640, 640):
        import torchvision.transforms.functional as F

        image_tensor = F.resize(image_tensor, [640, 640])

    try:
        # Compile and run on C Static target
        tvm_output = compile_and_run_on_target(
            target_string=C_STATIC_TARGET,
            mod=mod,
            input=image_tensor.numpy(),
            verbose_output=False,
        )

        # Convert output back to torch tensor
        if isinstance(tvm_output, (list, tuple)):
            raw_output = torch.from_numpy(tvm_output[0])
        else:
            raw_output = torch.from_numpy(tvm_output)

        logger.debug(f"  TVM output shape: {raw_output.shape}")

        # Apply NMS post-processing (version-specific)
        detections = apply_nms_postprocessing(
            raw_output, score_threshold, iou_threshold, version=version
        )

        # Scale coordinates back to original image size if provided
        if original_image_size is not None and len(detections["boxes"]) > 0:
            orig_h, orig_w = original_image_size
            logger.debug(f"  Scaling coordinates from 640x640 to {orig_w}x{orig_h}")
            detections["boxes"][:, [0, 2]] *= orig_w / 640  # Scale x coordinates
            detections["boxes"][:, [1, 3]] *= orig_h / 640  # Scale y coordinates

        return detections

    except Exception as e:
        logger.error(f"TVM inference failed: {e}")
        raise RuntimeError(f"TVM inference failed: {e}") from e


def compare_detection_results(
    pytorch_detections: Dict[str, torch.Tensor],
    tvm_detections: Dict[str, torch.Tensor],
    model: Any,
) -> Dict[str, Any]:
    """Compare PyTorch and TVM detection inference results

    Args:
        pytorch_detections: PyTorch detection outputs with 'boxes', 'labels', 'scores'
        tvm_detections: TVM detection outputs with 'boxes', 'labels', 'scores'
        model: YOLOv5 model (for class names)

    Returns:
        Dictionary with comparison metrics
    """
    pytorch_boxes = pytorch_detections["boxes"].detach().cpu().numpy()
    pytorch_labels = pytorch_detections["labels"].detach().cpu().numpy()
    pytorch_scores = pytorch_detections["scores"].detach().cpu().numpy()

    tvm_boxes = tvm_detections["boxes"].detach().cpu().numpy()
    tvm_labels = tvm_detections["labels"].detach().cpu().numpy()
    tvm_scores = tvm_detections["scores"].detach().cpu().numpy()

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
        for i in unmatched_pytorch[:5]:
            label_name = str(model.names[int(pytorch_labels[i])])  # type: ignore[attr-defined]
            logger.debug(f"    {label_name} ({pytorch_scores[i]:.2f}): {pytorch_boxes[i]}")

    if unmatched_tvm:
        logger.debug(f"\n  Unmatched TVM detections ({len(unmatched_tvm)}):")
        for j in unmatched_tvm[:5]:
            label_name = str(model.names[int(tvm_labels[j])])  # type: ignore[attr-defined]
            logger.debug(f"    {label_name} ({tvm_scores[j]:.2f}): {tvm_boxes[j]}")

    return comparison


def main(
    model_name: str = "yolov5s",
    image_url: Optional[str] = None,
    use_tvm: bool = False,
    compare: bool = False,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> ModelTestResult:
    """Main function to run YOLO object detection pipeline (v5 or v11)

    Args:
        model_name: YOLO model variant name (e.g., 'yolov5s', 'yolo11n')
        image_url: URL or path to image, or None for default
        use_tvm: Use TVM C Static compilation
        compare: Compare PyTorch vs TVM results
        score_threshold: Minimum confidence score for detections
        iou_threshold: NMS IoU threshold

    Returns:
        ModelTestResult with success=True if successful, success=False if failed
    """
    tvm_compile_success = None
    tvm_inference_success = None
    comparison_data = None

    try:
        # Load model (auto-detect version)
        model, version = load_yolo_model(model_name)
        logger.info(f"Loaded YOLO{version} model: {model_name}")

        # Load image
        image_url = image_url or DEFAULT_IMAGE_URL
        image = load_image(image_url)
        if image is None:
            return ModelTestResult(
                model=model,
                image=None,
                detections=None,
                success=False,
                error_message="Failed to load image",
            )

        # Preprocess image for inference
        import torchvision.transforms as transforms

        # Store original image size for coordinate scaling
        original_image_size = (image.height, image.width)

        preprocess = transforms.Compose(
            [
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ]
        )
        image_tensor: torch.Tensor = preprocess(image)  # type: ignore[assignment]

        # Run inference based on mode
        if compare:
            # Compare mode: run both PyTorch and TVM
            logger.info("\nRunning PyTorch inference...")
            pytorch_detections = run_inference(
                model, image, score_threshold, iou_threshold, version=version
            )

            logger.info("\nCompiling model with TVM...")
            try:
                tvm_mod = prepare_model_for_tvm(model, DEFAULT_INPUT_SHAPE, version=version)
                tvm_compile_success = True

                logger.info("Running TVM inference...")
                tvm_detections = run_inference_tvm(
                    tvm_mod,
                    image_tensor,
                    score_threshold,
                    iou_threshold,
                    compare_llvm=False,
                    original_image_size=original_image_size,
                    version=version,
                )
                tvm_inference_success = True

                # Compare results
                comparison_data = compare_detection_results(
                    pytorch_detections, tvm_detections, model
                )

                detections_dict = pytorch_detections

            except Exception as e:
                logger.error(f"TVM compilation/inference failed: {e}")
                logger.debug("Traceback:", exc_info=True)
                if tvm_compile_success is None:
                    tvm_compile_success = False
                elif tvm_inference_success is None:
                    tvm_inference_success = False

                # Fall back to PyTorch results
                detections_dict = pytorch_detections

        elif use_tvm:
            # TVM only mode
            logger.info("\nCompiling model with TVM...")
            try:
                tvm_mod = prepare_model_for_tvm(model, DEFAULT_INPUT_SHAPE, version=version)
                tvm_compile_success = True

                logger.info("Running TVM inference...")
                detections_dict = run_inference_tvm(
                    tvm_mod,
                    image_tensor,
                    score_threshold,
                    iou_threshold,
                    compare_llvm=True,
                    original_image_size=original_image_size,
                    version=version,
                )
                tvm_inference_success = True

            except Exception as e:
                logger.error(f"TVM compilation/inference failed: {e}")
                logger.debug("Traceback:", exc_info=True)
                if tvm_compile_success is None:
                    tvm_compile_success = False
                else:
                    tvm_inference_success = False
                raise

        else:
            # PyTorch only mode
            logger.info(f"\nRunning YOLO{version} inference...")
            detections_dict = run_inference(
                model, image, score_threshold, iou_threshold, version=version
            )

        # Display results
        detections = display_detections(detections_dict, model, max_display=10)

        # Statistics
        logger.info("\nDetection Statistics:")
        logger.info(f"  Total detections: {len(detections)}")
        if detections:
            logger.info(f"  Highest confidence: {detections[0].score * 100:.2f}%")
            logger.info(f"  Lowest confidence: {detections[-1].score * 100:.2f}%")

        return ModelTestResult(
            model=model,
            image=image,
            detections=detections,
            success=True,
            comparison=comparison_data,
            tvm_compile_success=tvm_compile_success,
            tvm_inference_success=tvm_inference_success,
        )

    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        logger.debug("Traceback:", exc_info=True)

        return ModelTestResult(
            model=None,
            image=None,
            detections=None,
            success=False,
            error_message=str(e),
            tvm_compile_success=tvm_compile_success,
            tvm_inference_success=tvm_inference_success,
        )


def get_all_yolo_models() -> List[str]:
    """Get list of all available YOLOv5 models

    Returns:
        List of YOLOv5 model names
    """
    return YOLO_MODELS.copy()


def test_multiple_models(
    image_url=None,
    max_models=None,
    model_filter=None,
    log_file=None,
    score_threshold=DEFAULT_SCORE_THRESHOLD,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
):
    """Test multiple YOLOv5 models

    Args:
        image_url: URL or path to test image
        max_models: Maximum number of models to test
        model_filter: List of model name substrings to filter by
        log_file: Optional path to CSV log file
        score_threshold: Minimum confidence score for detections
        iou_threshold: NMS IoU threshold
    """
    logger.debug("\nGetting YOLOv5 model list...")

    all_models = get_all_yolo_models()

    # Apply filter
    if model_filter:
        filtered_models = []
        for model_name in all_models:
            if any(filter_str in model_name.lower() for filter_str in model_filter):
                filtered_models.append(model_name)
        all_models = filtered_models
        logger.debug(f"Filtered to {len(all_models)} models matching {model_filter}")

    # Limit number
    if max_models:
        all_models = all_models[:max_models]
        logger.debug(f"Limited to first {max_models} models")

    logger.info(f"\nTesting {len(all_models)} models...\n")

    # Use default image if not provided
    if image_url is None:
        image_url = DEFAULT_IMAGE_URL

    # Track results
    successful_tests = []
    failed_tests = []

    for i, model_name in enumerate(all_models, 1):
        try:
            logger.info(f"\n[{i}/{len(all_models)}] Testing {model_name}...")

            result = main(model_name, image_url, False, False, score_threshold, iou_threshold)

            if not result.success:
                failed_tests.append(
                    {
                        "model": model_name,
                        "error": result.error_message or "Unknown error",
                    }
                )

                if log_file:
                    append_result_to_log(
                        log_file=log_file,
                        model_name=model_name,
                        error=True,
                        error_message=result.error_message or "Unknown error",
                    )
                continue

            # Success
            assert result.detections is not None
            top_detection = result.detections[0] if result.detections else None

            if top_detection:
                logger.info(
                    f"  Result: {len(result.detections)} detections, "
                    f"top: {top_detection.label_name} ({top_detection.score * 100:.1f}%)"
                )

                successful_tests.append(
                    {
                        "model": model_name,
                        "num_detections": len(result.detections),
                        "top_detection": top_detection.label_name,
                        "confidence": top_detection.score * 100,
                    }
                )

                if log_file:
                    append_result_to_log(
                        log_file=log_file,
                        model_name=model_name,
                        num_detections=len(result.detections),
                        top_detection=top_detection.label_name,
                        confidence=top_detection.score * 100,
                        error=False,
                    )
            else:
                logger.info("  Result: 0 detections")
                successful_tests.append(
                    {
                        "model": model_name,
                        "num_detections": 0,
                        "top_detection": "None",
                        "confidence": 0.0,
                    }
                )

        except Exception as e:
            logger.error(f"  Failed: {e}")
            failed_tests.append({"model": model_name, "error": str(e)})

            if log_file:
                append_result_to_log(
                    log_file=log_file,
                    model_name=model_name,
                    error=True,
                    error_message=str(e),
                )

    # Print summary
    logger.info("\nSummary")
    logger.info(f"{'-' * 60}")
    logger.info(
        f"Total: {len(all_models)} | Successful: {len(successful_tests)} | Failed: {len(failed_tests)}"
    )

    if successful_tests:
        logger.debug(f"\n{'-' * 60}")
        logger.debug("Successful Tests:")
        logger.debug(f"{'-' * 60}")
        for test in successful_tests:
            det = test.get("top_detection", "N/A")[:35]
            num_dets = test.get("num_detections", 0)
            logger.debug(
                f"  ✓ {test['model']:15s} -> {num_dets} dets, "
                f"top: {det:35s} ({test['confidence']:5.1f}%)"
            )

    if failed_tests:
        logger.info(f"\n{'-' * 60}")
        logger.info("Failed Tests:")
        logger.info(f"{'-' * 60}")
        for test in failed_tests:
            error = str(test["error"])[:50]
            logger.info(f"  ✗ {test['model']:15s} -> {error}")

    return successful_tests, failed_tests


def test_multiple_models_parallel(
    image_url=None,
    max_models=None,
    model_filter=None,
    max_workers=None,
    log_file=None,
    score_threshold=DEFAULT_SCORE_THRESHOLD,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
):
    """Test multiple YOLOv5 models in parallel

    Args:
        image_url: URL or path to test image
        max_models: Maximum number of models to test
        model_filter: List of model name substrings to filter by
        max_workers: Maximum number of parallel workers
        log_file: Optional path to CSV log file
        score_threshold: Minimum confidence score
        iou_threshold: NMS IoU threshold

    Returns:
        Tuple of (successful_tests, failed_tests)
    """
    import os

    logger.debug("\nGetting YOLOv5 model list...")

    all_models = get_all_yolo_models()

    # Apply filter
    if model_filter:
        filtered_models = []
        for model_name in all_models:
            if any(filter_str in model_name.lower() for filter_str in model_filter):
                filtered_models.append(model_name)
        all_models = filtered_models
        logger.debug(f"Filtered to {len(all_models)} models matching {model_filter}")

    # Limit number
    if max_models:
        all_models = all_models[:max_models]
        logger.debug(f"Limited to first {max_models} models")

    # Use default image
    if image_url is None:
        image_url = DEFAULT_IMAGE_URL

    # Determine workers
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, len(all_models))

    logger.info(f"\nTesting {len(all_models)} models in parallel with {max_workers} workers...\n")

    # Track results
    successful_tests = []
    failed_tests = []

    # Submit all tasks
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_model = {
            executor.submit(
                main, model_name, image_url, False, False, score_threshold, iou_threshold
            ): model_name
            for model_name in all_models
        }

        # Process completed tasks
        completed = 0
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            completed += 1

            try:
                result = future.result()

                if not result.success:
                    failed_tests.append(
                        {
                            "model": model_name,
                            "error": result.error_message or "Unknown error",
                        }
                    )
                    logger.warning(
                        f"[{completed}/{len(all_models)}] ✗ {model_name} - {result.error_message}"
                    )

                    if log_file:
                        append_result_to_log(
                            log_file=log_file,
                            model_name=model_name,
                            error=True,
                            error_message=result.error_message or "Unknown error",
                        )
                else:
                    # Success
                    assert result.detections is not None
                    if result.detections:
                        top_det = result.detections[0]
                        successful_tests.append(
                            {
                                "model": model_name,
                                "num_detections": len(result.detections),
                                "top_detection": top_det.label_name,
                                "confidence": top_det.score * 100,
                            }
                        )

                        logger.info(
                            f"[{completed}/{len(all_models)}] ✓ {model_name} - "
                            f"{len(result.detections)} dets, {top_det.label_name[:30]} "
                            f"({top_det.score * 100:.1f}%)"
                        )

                        if log_file:
                            append_result_to_log(
                                log_file=log_file,
                                model_name=model_name,
                                num_detections=len(result.detections),
                                top_detection=top_det.label_name,
                                confidence=top_det.score * 100,
                                error=False,
                            )

            except Exception as e:
                error_msg = str(e)
                exception_type = type(e).__name__
                logger.error(
                    f"[{completed}/{len(all_models)}] ✗ {model_name} - {exception_type}: {error_msg}"
                )

                failed_tests.append({"model": model_name, "error": error_msg})

                if log_file:
                    append_result_to_log(
                        log_file=log_file,
                        model_name=model_name,
                        error=True,
                        error_message=f"{exception_type}: {error_msg}",
                    )

    # Print summary
    logger.info("\nSummary")
    logger.info(f"{'-' * 60}")
    logger.info(
        f"Total: {len(all_models)} | Successful: {len(successful_tests)} | Failed: {len(failed_tests)}"
    )

    if successful_tests:
        logger.debug(f"\n{'-' * 60}")
        logger.debug("Successful Tests:")
        logger.debug(f"{'-' * 60}")
        for test in successful_tests:
            det = test.get("top_detection", "N/A")[:35]
            num_dets = test.get("num_detections", 0)
            logger.debug(
                f"  ✓ {test['model']:15s} -> {num_dets} dets, "
                f"top: {det:35s} ({test['confidence']:5.1f}%)"
            )

    if failed_tests:
        logger.info(f"\n{'-' * 60}")
        logger.info("Failed Tests:")
        logger.info(f"{'-' * 60}")
        for test in failed_tests:
            error = str(test["error"])[:50]
            logger.info(f"  ✗ {test['model']:15s} -> {error}")

    return successful_tests, failed_tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv5 Object Detection Model Tester")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5s",
        help="YOLOv5 model variant (default: yolov5s)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE_URL,
        help="Path or URL to test image",
    )
    parser.add_argument(
        "--test-all",
        action="store_true",
        help="Test all available YOLOv5 models",
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
        help="Filter models by name (e.g., --filter yolov5s yolov5m)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Enable quiet mode (minimal output)",
    )
    parser.add_argument(
        "--tvm",
        action="store_true",
        help="Use TVM compilation with C Static target",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare PyTorch vs TVM results (implies --tvm)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (only with --test-all)",
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
        default=DEFAULT_IOU_THRESHOLD,
        help=f"NMS IoU threshold (default: {DEFAULT_IOU_THRESHOLD})",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose, args.quiet)

    # If compare is set, ensure tvm is also set
    if args.compare:
        args.tvm = True

    if args.test_all:
        # Test multiple models
        if args.parallel:
            test_multiple_models_parallel(
                image_url=args.image,
                max_models=args.max_models,
                model_filter=args.filter,
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
                log_file=args.log_file,
                score_threshold=args.score_threshold,
                iou_threshold=args.iou_threshold,
            )
    else:
        # Test single model
        result = main(
            model_name=args.model,
            image_url=args.image,
            use_tvm=args.tvm,
            compare=args.compare,
            score_threshold=args.score_threshold,
            iou_threshold=args.iou_threshold,
        )
