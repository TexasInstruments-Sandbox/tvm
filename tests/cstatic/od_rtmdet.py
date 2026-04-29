#!/usr/bin/env python
"""RTMDet Object Detection Model Tester

This script provides comprehensive testing and validation for RTMDet object detection models
from the MMDetection framework.

Features:
    - Automatic loading of RTMDet models (tiny, s, m, l, x variants)
    - PyTorch inference (default)
    - TVM C Static compilation and inference (--tvm)
    - Side-by-side comparison of PyTorch vs TVM results (--compare)
    - Batch testing of multiple RTMDet variants
    - Comprehensive logging with adjustable verbosity levels
    - Bounding box visualization and detection metrics

Usage Examples:
    # Test RTMDet-s
    python od_rtmdet.py --model rtmdet_s

    # Test RTMDet-tiny
    python od_rtmdet.py --model rtmdet_tiny

    # Test with TVM compilation
    python od_rtmdet.py --model rtmdet_s --tvm

    # Compare PyTorch vs TVM
    python od_rtmdet.py --model rtmdet_s --compare

    # Test with custom confidence threshold
    python od_rtmdet.py --model rtmdet_m --score-threshold 0.3

    # Test multiple models
    python od_rtmdet.py --test-all

    # Verbose mode
    python od_rtmdet.py --model rtmdet_s --verbose

Command-Line Options:
    --model MODEL              RTMDet model variant (default: rtmdet_s)
    --image IMAGE              Path or URL to test image
    --test-all                 Test all available RTMDet models
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
    RTMDet (via MMDetection):
        - rtmdet_tiny (5.0M params, 640x640)
        - rtmdet_s (8.9M params, 640x640)
        - rtmdet_m (24.7M params, 640x640)
        - rtmdet_l (52.3M params, 640x640)
        - rtmdet_x (94.9M params, 640x640)

Note on TVM Support:
    TVM compilation extracts the RTMDet model's backbone, neck, and detection head.
    The compiled model produces raw detection outputs (bbox predictions and classification
    scores per FPN level) which are post-processed in Python with NMS.
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
DEFAULT_SCORE_THRESHOLD = 0.25  # Default confidence threshold
DEFAULT_IOU_THRESHOLD = 0.45  # Default NMS IoU threshold
IOU_THRESHOLD = 0.5  # For box matching in comparisons
DEFAULT_INPUT_SHAPE = (1, 3, 640, 640)  # RTMDet default input size
C_STATIC_TARGET = "c_static"
LLVM_TARGET = "llvm"

# Thread lock for CSV file writing
_CSV_WRITE_LOCK = threading.Lock()

# Available RTMDet models
RTMDET_MODELS = [
    "rtmdet_tiny",  # Tiny (5.0M params, 640x640)
    "rtmdet_s",  # Small (8.9M params, 640x640)
    "rtmdet_m",  # Medium (24.7M params, 640x640)
    "rtmdet_l",  # Large (52.3M params, 640x640)
    "rtmdet_x",  # Extra Large (94.9M params, 640x640)
]

# RTMDet config files (from MMDetection model zoo)
RTMDET_CONFIGS = {
    "rtmdet_tiny": "rtmdet_tiny_8xb32-300e_coco.py",
    "rtmdet_s": "rtmdet_s_8xb32-300e_coco.py",
    "rtmdet_m": "rtmdet_m_8xb32-300e_coco.py",
    "rtmdet_l": "rtmdet_l_8xb32-300e_coco.py",
    "rtmdet_x": "rtmdet_x_8xb32-300e_coco.py",
}

# RTMDet checkpoint URLs (from MMDetection model zoo)
RTMDET_CHECKPOINTS = {
    "rtmdet_tiny": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    "rtmdet_s": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
    "rtmdet_m": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
    "rtmdet_l": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
    "rtmdet_x": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_x_8xb32-300e_coco/rtmdet_x_8xb32-300e_coco_20220715_230555-cc79b9ae.pth",
}


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


def load_rtmdet_model(model_name: str = "rtmdet_s") -> Any:
    """Load RTMDet model from MMDetection

    Args:
        model_name: RTMDet model variant (e.g., 'rtmdet_tiny', 'rtmdet_s', 'rtmdet_m')

    Returns:
        Loaded RTMDet model with config in eval mode
    """
    logger.debug(f"Loading RTMDet model: {model_name}")

    try:
        from mmdet.apis import init_detector

        # Validate model name
        if model_name not in RTMDET_CONFIGS:
            raise ValueError(
                f"Unknown RTMDet model: {model_name}. "
                f"Available models: {list(RTMDET_CONFIGS.keys())}"
            )

        # Get config and checkpoint
        config_name = RTMDET_CONFIGS[model_name]
        checkpoint_url = RTMDET_CHECKPOINTS[model_name]

        # Try to use MMDetection's built-in config
        # This will use mim to get the config from the model zoo
        try:
            # First try: use config name from model zoo
            config_file = f"configs/rtmdet/{config_name}"
            logger.debug(f"  Config: {config_file}")
            logger.debug(f"  Checkpoint: {checkpoint_url}")

            # init_detector handles config loading and checkpoint download
            model = init_detector(config_file, checkpoint_url, device="cpu")
            model.eval()

            logger.debug("  Model loaded successfully")
            logger.debug(f"  Classes: {len(model.dataset_meta.get('classes', []))}")  # type: ignore[union-attr]

            return model

        except Exception as e:
            logger.debug(f"  Failed to load from model zoo: {e}")
            logger.debug("  Attempting direct config URL...")

            # Fallback: use direct URLs for both config and checkpoint
            config_url = f"https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs/rtmdet/{config_name}"
            logger.debug(f"  Config URL: {config_url}")

            model = init_detector(config_url, checkpoint_url, device="cpu")
            model.eval()

            logger.debug("  Model loaded successfully from URLs")
            logger.debug(f"  Classes: {len(model.dataset_meta.get('classes', []))}")  # type: ignore[union-attr]

            return model

    except ImportError as e:
        logger.error(f"MMDetection not installed: {e}")
        logger.error("Install with: pip install mmdet mmcv mmengine")
        raise
    except Exception as e:
        logger.error(f"Failed to load RTMDet model '{model_name}': {e}")
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
) -> Dict[str, torch.Tensor]:
    """Run RTMDet inference on image

    Args:
        model: RTMDet model from MMDetection
        image: PIL Image
        score_threshold: Minimum confidence score for detections
        iou_threshold: NMS IoU threshold

    Returns:
        Dictionary with 'boxes', 'labels', 'scores' tensors
    """
    import numpy as np
    from mmdet.apis import inference_detector

    # Convert PIL image to format expected by MMDetection
    # MMDetection expects BGR format (OpenCV convention)
    image_np = np.array(image)
    if image_np.shape[-1] == 3:
        # Convert RGB to BGR
        image_np = image_np[:, :, ::-1]

    # Run inference using MMDetection API
    # inference_detector returns DetDataSample with predictions
    result = inference_detector(model, image_np)

    # Extract predictions from result
    # RTMDet result structure: result.pred_instances contains bboxes, labels, scores
    pred_instances = result.pred_instances  # type: ignore[attr-defined]

    # Filter by score threshold
    scores = pred_instances.scores.cpu()  # type: ignore[attr-defined]
    mask = scores >= score_threshold

    if mask.sum() == 0:
        # No detections above threshold
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,), dtype=torch.float32),
        }

    # Get filtered detections
    boxes = pred_instances.bboxes.cpu()[mask]  # type: ignore[attr-defined]  # [x1, y1, x2, y2]
    labels = pred_instances.labels.cpu()[mask]  # type: ignore[attr-defined]
    scores = scores[mask]

    # Apply additional NMS if needed (MMDetection already applies NMS, but we can be stricter)
    from torchvision.ops import nms

    keep_indices = nms(boxes, scores, iou_threshold)

    return {
        "boxes": boxes[keep_indices],
        "labels": labels[keep_indices],
        "scores": scores[keep_indices],
    }


def display_detections(
    detections: Dict[str, torch.Tensor],
    model: Any,
    max_display: int = 10,
) -> List[Detection]:
    """Display detected objects

    Args:
        detections: Dictionary with 'boxes', 'labels', 'scores' tensors
        model: RTMDet model (contains class names in dataset_meta)
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

    # Get class names from model (RTMDet stores them in dataset_meta)
    class_names = model.dataset_meta.get("classes", [])

    results = []
    for i in range(min(num_detections, max_display)):
        box = boxes[i].tolist()
        label_id = int(det_labels[i].item())

        # Get label name from class names
        if label_id < len(class_names):
            label_name = class_names[label_id]
        else:
            label_name = f"class_{label_id}"

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


class RTMDetWrapper(nn.Module):
    """Wrapper to make RTMDet compatible with torch.export

    RTMDet from MMDetection includes data preprocessing and post-processing logic
    that contains dynamic operations. This wrapper extracts the core detection model
    (backbone + neck + bbox_head) for static graph export to TVM.
    """

    def __init__(self, rtmdet_model):
        super().__init__()

        # Extract core components from RTMDet
        self.backbone = rtmdet_model.backbone
        self.neck = rtmdet_model.neck
        self.bbox_head = rtmdet_model.bbox_head

        # Set to eval mode
        self.backbone.eval()
        self.neck.eval()
        self.bbox_head.eval()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass that returns raw RTMDet outputs as a FLAT tuple

        Args:
            x: Input tensor of shape [batch, 3, height, width]
               Expected to be normalized with ImageNet mean/std

        Returns:
            Flat tuple of 6 tensors (for TVM compatibility):
            - cls_scores[0]: Classification scores, FPN level 0 (stride 8)
              Shape: [batch, num_classes, 80, 80] for 640x640 input
            - cls_scores[1]: Classification scores, FPN level 1 (stride 16)
              Shape: [batch, num_classes, 40, 40] for 640x640 input
            - cls_scores[2]: Classification scores, FPN level 2 (stride 32)
              Shape: [batch, num_classes, 20, 20] for 640x640 input
            - bbox_preds[0]: Bbox distance predictions, FPN level 0
              Shape: [batch, 4, 80, 80] for 640x640 input
            - bbox_preds[1]: Bbox distance predictions, FPN level 1
              Shape: [batch, 4, 40, 40] for 640x640 input
            - bbox_preds[2]: Bbox distance predictions, FPN level 2
              Shape: [batch, 4, 20, 20] for 640x640 input

        Note:
            The nested tuple structure (cls_scores, bbox_preds) is flattened
            to avoid complexity in TVM C Static compilation. Post-processing
            code will reconstruct the nested structure.
        """
        # Backbone forward (extract multi-scale features)
        x = self.backbone(x)

        # Neck forward (FPN processing)
        x = self.neck(x)

        # Detection head forward (classification + bbox regression)
        # bbox_head returns (cls_scores, bbox_preds) for each FPN level
        cls_scores, bbox_preds = self.bbox_head(x)

        # FLATTEN the nested structure for TVM compatibility
        # Return 6 tensors in a flat tuple instead of nested (tuple, tuple)
        return (
            cls_scores[0],
            cls_scores[1],
            cls_scores[2],
            bbox_preds[0],
            bbox_preds[1],
            bbox_preds[2],
        )


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
    torch_model: nn.Module, input_shape: Tuple[int, ...] = DEFAULT_INPUT_SHAPE
) -> tvm.IRModule:
    """Prepare an RTMDet model for TVM compilation

    Args:
        torch_model: RTMDet model in eval mode
        input_shape: Shape of input tensor (batch, channels, height, width)

    Returns:
        Processed TVM IRModule ready for compilation

    Raises:
        ValueError: If there are issues with model conversion
    """
    logger.debug("  Converting RTMDet to TVM Relax IR...")

    # Wrap the model for export compatibility
    wrapped_model = RTMDetWrapper(torch_model)

    # Create example input for torch.export
    # RTMDet expects ImageNet-normalized input
    example_input = (torch.randn(*input_shape, dtype=torch.float32),)

    # Convert to Relax IRModule
    mod = torch_to_relax(wrapped_model, example_input)

    # Process the module (detach and bind parameters)
    mod = process_relax(mod)

    logger.debug("  TVM IR conversion complete")

    return mod


def apply_rtmdet_postprocessing(
    cls_scores: Tuple[torch.Tensor, ...],
    bbox_preds: Tuple[torch.Tensor, ...],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    input_size: Tuple[int, int] = (640, 640),
) -> Dict[str, torch.Tensor]:
    """Apply NMS post-processing to raw RTMDet output

    Args:
        cls_scores: Tuple of classification score tensors per FPN level
                   Each tensor shape: [batch, num_classes, H, W]
        bbox_preds: Tuple of bbox prediction tensors per FPN level
                   Each tensor shape: [batch, 4, H, W] (distance format: l, t, r, b)
        score_threshold: Minimum confidence score
        iou_threshold: NMS IoU threshold
        input_size: Input image size (height, width)

    Returns:
        Dictionary with 'boxes', 'labels', 'scores' tensors
    """
    from torchvision.ops import nms

    logger.debug(f"RTMDet output: {len(cls_scores)} FPN levels")

    all_boxes = []
    all_scores = []
    all_labels = []

    # RTMDet FPN strides (typical for 640x640 input)
    strides = [8, 16, 32]

    # Process each FPN level
    for level_idx, (cls_score, bbox_pred) in enumerate(zip(cls_scores, bbox_preds)):
        # Remove batch dimension
        if cls_score.dim() == 4:
            cls_score = cls_score[0]  # [num_classes, H, W]
            bbox_pred = bbox_pred[0]  # [4, H, W]

        num_classes, h, w = cls_score.shape
        stride = strides[level_idx] if level_idx < len(strides) else 8 * (2**level_idx)

        logger.debug(
            f"  Level {level_idx}: cls_score {cls_score.shape}, bbox_pred {bbox_pred.shape}, stride {stride}"
        )

        # Reshape to [H*W, num_classes] and [H*W, 4]
        cls_score = cls_score.permute(1, 2, 0).reshape(-1, num_classes)  # [H*W, num_classes]
        bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)  # [H*W, 4]

        # Apply sigmoid to classification scores
        cls_score = torch.sigmoid(cls_score)

        # Get max scores and labels
        max_scores, labels = cls_score.max(dim=1)

        # Filter by score threshold
        mask = max_scores >= score_threshold
        if mask.sum() == 0:
            continue

        filtered_scores = max_scores[mask]
        filtered_labels = labels[mask]
        filtered_bbox_pred = bbox_pred[mask]

        # Generate anchor points (grid centers)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing="ij",
        )
        grid_points = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)  # [H*W, 2]
        grid_points = grid_points[mask]

        # Convert distance predictions to boxes
        # RTMDet predicts distances: [left, top, right, bottom]
        # Need to convert to [x1, y1, x2, y2] in pixel coordinates
        grid_points_scaled = grid_points * stride

        x1 = grid_points_scaled[:, 0] - filtered_bbox_pred[:, 0] * stride
        y1 = grid_points_scaled[:, 1] - filtered_bbox_pred[:, 1] * stride
        x2 = grid_points_scaled[:, 0] + filtered_bbox_pred[:, 2] * stride
        y2 = grid_points_scaled[:, 1] + filtered_bbox_pred[:, 3] * stride

        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        all_boxes.append(boxes)
        all_scores.append(filtered_scores)
        all_labels.append(filtered_labels)

    # Combine all FPN levels
    if len(all_boxes) == 0:
        logger.debug("No detections above confidence threshold")
        return {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.int64),
            "scores": torch.zeros((0,), dtype=torch.float32),
        }

    all_boxes = torch.cat(all_boxes, dim=0)
    all_scores = torch.cat(all_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    logger.debug(f"Total candidates before NMS: {len(all_boxes)}")

    # Apply NMS
    keep_indices = nms(all_boxes, all_scores, iou_threshold)

    logger.debug(f"After NMS: {len(keep_indices)} detections")

    return {
        "boxes": all_boxes[keep_indices],
        "labels": all_labels[keep_indices],
        "scores": all_scores[keep_indices],
    }


def run_inference_tvm(
    mod: tvm.IRModule,
    image_tensor: torch.Tensor,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    compare_llvm: bool = False,
    original_image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    """Run inference using TVM with C Static target for RTMDet

    Args:
        mod: TVM IRModule to execute
        image_tensor: Input image as PyTorch tensor (3D or 4D), ImageNet normalized
        score_threshold: Minimum confidence score for detections
        iou_threshold: NMS IoU threshold
        compare_llvm: If True, also compile for LLVM and compare results (not implemented)
        original_image_size: Original image size (height, width) for coordinate scaling

    Returns:
        Dict with 'boxes', 'labels', 'scores' tensors (in original image coordinates)

    Raises:
        RuntimeError: If TVM compilation or execution fails
    """
    logger.debug("  Compiling with TVM C Static backend...")

    # Add batch dimension if needed
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Ensure correct shape (640x640 for RTMDet)
    if image_tensor.shape[2:] != (640, 640):
        import torchvision.transforms.functional as F

        image_tensor = F.resize(image_tensor, [640, 640])

    try:
        # RTMDetWrapper now returns a FLAT tuple of 6 tensors:
        # - cls_scores[0], cls_scores[1], cls_scores[2]  (3 FPN levels)
        # - bbox_preds[0], bbox_preds[1], bbox_preds[2]  (3 FPN levels)
        #
        # For 640x640 input, the shapes are:
        # output_0: [1, 80, 80, 80]  cls_scores level 0 (stride 8)
        # output_1: [1, 80, 40, 40]  cls_scores level 1 (stride 16)
        # output_2: [1, 80, 20, 20]  cls_scores level 2 (stride 32)
        # output_3: [1, 4, 80, 80]   bbox_preds level 0 (stride 8)
        # output_4: [1, 4, 40, 40]   bbox_preds level 1 (stride 16)
        # output_5: [1, 4, 20, 20]   bbox_preds level 2 (stride 32)

        logger.debug("  Running TVM compilation and execution...")

        # The compile_and_run_on_target utility automatically:
        # - Detects tuple returns via model_returns_tuple()
        # - Passes MODEL_RETURNS_TUPLE=ON to CMake
        # - Saves all outputs to NPZ file
        # - Returns list of numpy arrays
        tvm_outputs = compile_and_run_on_target(
            target_string=C_STATIC_TARGET,
            mod=mod,
            input=image_tensor.numpy(),
            verbose_output=False,
        )

        # Validate we got the expected 6 outputs
        if not isinstance(tvm_outputs, list):
            raise RuntimeError(
                f"Expected list of arrays from TVM, got {type(tvm_outputs).__name__}"
            )

        if len(tvm_outputs) != 6:
            raise RuntimeError(
                f"Expected 6 outputs from RTMDet (3 cls_scores + 3 bbox_preds), "
                f"got {len(tvm_outputs)}"
            )

        logger.debug("  Successfully compiled and executed TVM model")
        logger.debug(f"  Received {len(tvm_outputs)} outputs from TVM")

        # Reconstruct nested structure for post-processing
        # Convert numpy arrays back to PyTorch tensors
        cls_scores = tuple(torch.from_numpy(tvm_outputs[i]) for i in range(3))
        bbox_preds = tuple(torch.from_numpy(tvm_outputs[i]) for i in range(3, 6))

        logger.debug(
            f"  Reconstructed structure: "
            f"{len(cls_scores)} cls_scores + {len(bbox_preds)} bbox_preds"
        )

        # Log shapes for debugging
        for i, (cls, bbox) in enumerate(zip(cls_scores, bbox_preds)):
            logger.debug(f"    FPN Level {i}: cls {cls.shape}, bbox {bbox.shape}")

        # Apply post-processing (NMS, etc.) using the same function as PyTorch path
        detections = apply_rtmdet_postprocessing(
            cls_scores,
            bbox_preds,
            score_threshold,
            iou_threshold,
            input_size=(640, 640),
        )

        logger.debug(f"  Post-processing complete: {len(detections['boxes'])} detections")

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
        model: RTMDet model (for class names in dataset_meta)

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

    # Get class names from RTMDet model
    class_names = model.dataset_meta.get("classes", [])

    # Show unmatched detections
    if unmatched_pytorch:
        logger.debug(f"\n  Unmatched PyTorch detections ({len(unmatched_pytorch)}):")
        for i in unmatched_pytorch[:5]:
            label_id = int(pytorch_labels[i])
            label_name = (
                class_names[label_id] if label_id < len(class_names) else f"class_{label_id}"
            )
            logger.debug(f"    {label_name} ({pytorch_scores[i]:.2f}): {pytorch_boxes[i]}")

    if unmatched_tvm:
        logger.debug(f"\n  Unmatched TVM detections ({len(unmatched_tvm)}):")
        for j in unmatched_tvm[:5]:
            label_id = int(tvm_labels[j])
            label_name = (
                class_names[label_id] if label_id < len(class_names) else f"class_{label_id}"
            )
            logger.debug(f"    {label_name} ({tvm_scores[j]:.2f}): {tvm_boxes[j]}")

    return comparison


def main(
    model_name: str = "rtmdet_s",
    image_url: Optional[str] = None,
    use_tvm: bool = False,
    compare: bool = False,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> ModelTestResult:
    """Main function to run RTMDet object detection pipeline

    Args:
        model_name: RTMDet model variant name (e.g., 'rtmdet_s', 'rtmdet_m')
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
        # Load model
        model = load_rtmdet_model(model_name)
        logger.info(f"Loaded RTMDet model: {model_name}")

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

        # Preprocess image for inference (ImageNet normalization)
        import torchvision.transforms as transforms

        normalize = transforms.Normalize(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
        )
        preprocess = transforms.Compose(
            [
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        image_tensor: torch.Tensor = preprocess(image)  # type: ignore[assignment]

        # Run PyTorch inference (always run for baseline / comparison)
        logger.info("\nRunning RTMDet PyTorch inference...")
        pytorch_detections_dict = run_inference(model, image, score_threshold, iou_threshold)

        # If TVM mode is requested, compile and run TVM inference
        tvm_detections_dict = None
        if use_tvm or compare:
            try:
                logger.info("\nPreparing model for TVM compilation...")
                mod = prepare_model_for_tvm(model, DEFAULT_INPUT_SHAPE)

                tvm_compile_success = True
                logger.info("Running TVM inference with C Static backend...")

                tvm_detections_dict = run_inference_tvm(
                    mod,
                    image_tensor,
                    score_threshold,
                    iou_threshold,
                )

                tvm_inference_success = True
                logger.info("✓ TVM inference completed successfully")

                # If compare mode, run comparison
                if compare:
                    logger.info("\nComparing PyTorch vs TVM results...")
                    comparison_data = compare_detection_results(
                        pytorch_detections_dict,
                        tvm_detections_dict,
                        model,
                    )

            except Exception as e:
                logger.error(f"✗ TVM compilation/inference failed: {e}")
                logger.debug("Traceback:", exc_info=True)
                tvm_compile_success = False
                tvm_inference_success = False
                # Continue with PyTorch results

        # Display results (use PyTorch for display)
        detections = display_detections(pytorch_detections_dict, model, max_display=10)

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


def get_all_rtmdet_models() -> List[str]:
    """Get list of all available RTMDet models

    Returns:
        List of RTMDet model names
    """
    return RTMDET_MODELS.copy()


def test_multiple_models(
    image_url=None,
    max_models=None,
    model_filter=None,
    log_file=None,
    score_threshold=DEFAULT_SCORE_THRESHOLD,
    iou_threshold=DEFAULT_IOU_THRESHOLD,
):
    """Test multiple RTMDet models

    Args:
        image_url: URL or path to test image
        max_models: Maximum number of models to test
        model_filter: List of model name substrings to filter by
        log_file: Optional path to CSV log file
        score_threshold: Minimum confidence score for detections
        iou_threshold: NMS IoU threshold
    """
    logger.debug("\nGetting RTMDet model list...")

    all_models = get_all_rtmdet_models()

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
    """Test multiple RTMDet models in parallel

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

    logger.debug("\nGetting RTMDet model list...")

    all_models = get_all_rtmdet_models()

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
    parser = argparse.ArgumentParser(description="RTMDet Object Detection Model Tester")
    parser.add_argument(
        "--model",
        type=str,
        default="rtmdet_s",
        help="RTMDet model variant (default: rtmdet_s)",
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
        help="Test all available RTMDet models",
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
        help="Filter models by name (e.g., --filter rtmdet_s rtmdet_m)",
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
