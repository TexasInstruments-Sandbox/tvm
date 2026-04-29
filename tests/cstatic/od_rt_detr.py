#!/usr/bin/env python
"""RT-DETR Object Detection Model Tester

This script provides comprehensive testing and validation for RT-DETR (Real-Time DEtection TRansformer)
object detection models from Baidu via the ultralytics package.

RT-DETR is a Vision Transformer-based real-time object detector that uses:
    - Transformer architecture with query-based detection (no anchors)
    - Set-based predictions with bipartite matching
    - No NMS required (queries naturally avoid duplicates)
    - Efficient hybrid encoding for real-time performance

Features:
    - Automatic loading of RT-DETR models from Ultralytics
    - PyTorch inference (default)
    - TVM C Static compilation and inference (--tvm)
    - Side-by-side comparison of PyTorch vs TVM results (--compare)
    - Batch testing of multiple RT-DETR variants
    - Comprehensive logging with adjustable verbosity levels
    - Bounding box visualization and detection metrics

Usage Examples:
    # Test RT-DETR-L (default)
    python od_rt_detr.py --model rtdetr-l

    # Test RT-DETR-X (largest model)
    python od_rt_detr.py --model rtdetr-x

    # Test with TVM compilation
    python od_rt_detr.py --model rtdetr-l --tvm

    # Compare PyTorch vs TVM
    python od_rt_detr.py --model rtdetr-l --compare

    # Test with custom confidence threshold
    python od_rt_detr.py --model rtdetr-l --score-threshold 0.3

    # Test multiple models
    python od_rt_detr.py --test-all

    # Verbose mode
    python od_rt_detr.py --model rtdetr-l --verbose

Command-Line Options:
    --model MODEL              RT-DETR model variant (default: rtdetr-l)
    --image IMAGE              Path or URL to test image
    --test-all                 Test all available RT-DETR models (rtdetr-l and rtdetr-x)
    --log-file PATH            CSV log file for appending results (with --test-all)
    --tvm                      Use TVM compilation with C Static target
    --compare                  Compare PyTorch vs TVM results
    --score-threshold FLOAT    Minimum confidence score for detections (default: 0.3)
    --verbose, -v              Enable verbose output with detailed logging
    --quiet, -q                Enable quiet mode (minimal output)

Supported Models:
    RT-DETR (via ultralytics):
        - rtdetr-l: Large model (32M params)
        - rtdetr-x: Extra large model (76M params)

Note on TVM Support:
    WARNING: TVM compilation is currently NOT SUPPORTED for RT-DETR models.

    RT-DETR uses deformable attention in its transformer architecture, which requires
    operations not yet supported by TVM's PyTorch frontend:
      - grid_sampler (deformable attention)
      - max.dim (dimensional reductions)
      - Runtime assertions and boolean operations

    These operations are fundamental to RT-DETR's architecture and cannot be bypassed.
    Use PyTorch inference mode only (default behavior without --tvm flag).

    The --tvm and --compare flags are provided for future compatibility when TVM adds
    support for these transformer operations.
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
DEFAULT_SCORE_THRESHOLD = 0.3  # RT-DETR default confidence threshold
IOU_THRESHOLD = 0.5  # For box matching in comparisons
DEFAULT_INPUT_SHAPE = (1, 3, 640, 640)  # RT-DETR default input size (same as YOLO)
C_STATIC_TARGET = "c_static"
LLVM_TARGET = "llvm"

# Thread lock for CSV file writing
_CSV_WRITE_LOCK = threading.Lock()

# Available RT-DETR models from Ultralytics
# RT-DETR uses a query-based transformer architecture
RTDETR_MODELS = [
    "rtdetr-l",  # Large model (32M params), good balance of speed and accuracy
    "rtdetr-x",  # Extra large model (76M params), highest accuracy
]


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


def load_rtdetr_model(model_name: str = "rtdetr-l") -> Any:
    """Load RT-DETR model from ultralytics package

    Args:
        model_name: RT-DETR model variant (e.g., 'rtdetr-l', 'rtdetr-x')

    Returns:
        Loaded RT-DETR model in eval mode (with RTDETR wrapper)
    """
    logger.debug(f"Loading RT-DETR model: {model_name}")

    try:
        from ultralytics import RTDETR  # type: ignore[attr-defined]

        # Load model from ultralytics package
        # Model file will be downloaded if not present
        model = RTDETR(f"{model_name}.pt")
        model.model.eval()  # type: ignore[attr-defined]

        logger.debug("  Model loaded successfully")
        logger.debug(f"  Classes: {len(model.names)}")
        logger.debug(f"  Architecture: Transformer-based (DETR)")

        return model

    except ImportError as e:
        logger.error(f"ultralytics package not installed: {e}")
        logger.error("Install with: pip install ultralytics")
        raise
    except Exception as e:
        logger.error(f"Failed to load RT-DETR model '{model_name}': {e}")
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
) -> Dict[str, torch.Tensor]:
    """Run RT-DETR inference on image

    Args:
        model: RT-DETR model (with RTDETR wrapper from Ultralytics)
        image: PIL Image
        score_threshold: Minimum confidence score for detections

    Returns:
        Dictionary with 'boxes', 'labels', 'scores' tensors
    """
    # Set model confidence threshold
    # RT-DETR doesn't need IoU threshold since it doesn't use NMS
    model.conf = score_threshold  # type: ignore[attr-defined]

    # Run inference
    with torch.no_grad():
        # RT-DETR from ultralytics supports verbose parameter
        results = model(image, verbose=False)

    # Extract detections from RT-DETR results
    # RT-DETR returns results in same format as YOLO from ultralytics
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

    # Parse detections
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


class RTDETRWrapper(nn.Module):
    """Wrapper to make RT-DETR models compatible with torch.export

    RT-DETR includes preprocessing/postprocessing wrappers that have dynamic operations.
    This wrapper extracts the core model for static graph export.

    RT-DETR uses a transformer-based architecture with query-based detection.
    """

    def __init__(self, rtdetr_model):
        super().__init__()

        # Extract the core model without wrappers
        if hasattr(rtdetr_model, "model"):
            self.model = rtdetr_model.model
        else:
            self.model = rtdetr_model

        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that returns raw model output

        Args:
            x: Input tensor of shape [batch, 3, height, width]

        Returns:
            Raw output tensor from RT-DETR model (before post-processing)
            RT-DETR: [batch, num_queries, num_classes + 4]
            Typically 300 queries, 80 classes (COCO) + 4 box coords
            Output format: [x1, y1, x2, y2, class_0, class_1, ..., class_79]
        """
        output = self.model(x)

        # RT-DETR may return tuples or lists
        # Return the first output (the predictions)
        if isinstance(output, (list, tuple)):
            return output[0]
        return output


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
    """Prepare an RT-DETR model for TVM compilation

    Args:
        torch_model: RT-DETR model in eval mode
        input_shape: Shape of input tensor (batch, channels, height, width)

    Returns:
        Processed TVM IRModule ready for compilation

    Raises:
        ValueError: If there are issues with model conversion
    """
    logger.debug("  Converting RT-DETR to TVM Relax IR...")

    # Wrap the model for export compatibility
    wrapped_model = RTDETRWrapper(torch_model)

    # Create example input for torch.export
    example_input = (torch.randn(*input_shape, dtype=torch.float32),)

    # Convert to Relax IRModule
    mod = torch_to_relax(wrapped_model, example_input)

    # Process the module (detach and bind parameters)
    mod = process_relax(mod)

    logger.debug("  TVM IR conversion complete")

    return mod


def apply_rtdetr_postprocessing(
    raw_output: torch.Tensor,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> Dict[str, torch.Tensor]:
    """Apply post-processing to raw RT-DETR output

    RT-DETR uses query-based detection with bipartite matching, so NO NMS is needed.
    The model naturally produces non-overlapping detections through the transformer queries.

    Args:
        raw_output: Raw tensor output from TVM inference
                   RT-DETR: [batch, num_queries, num_classes + 4]
                   Typically: [1, 300, 84] where 84 = 4 (box) + 80 (COCO classes)
                   Output format: [x1, y1, x2, y2, class_0, class_1, ..., class_79]
        score_threshold: Minimum confidence score

    Returns:
        Dictionary with 'boxes', 'labels', 'scores' tensors
    """
    logger.debug(f"Raw TVM output shape: {raw_output.shape}")
    logger.debug(f"Raw output min: {raw_output.min():.4f}, max: {raw_output.max():.4f}")

    # Remove batch dimension
    if raw_output.ndim == 3:
        raw_output = raw_output[0]  # Now [num_queries, num_classes + 4]

    # Extract box coordinates and class scores
    # RT-DETR format: [x1, y1, x2, y2, class_0, class_1, ..., class_79]
    boxes_xyxy = raw_output[:, :4]  # [num_queries, 4]
    class_scores = raw_output[:, 4:]  # [num_queries, num_classes]

    # Check if class scores need sigmoid activation
    if class_scores.max() > 1.0 or class_scores.min() < 0.0:
        logger.debug("Applying sigmoid to class scores (detected raw logits)")
        class_scores = torch.sigmoid(class_scores)

    # Get max class score and predicted class for each query
    conf, class_pred = class_scores.max(1)

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

    # NO NMS needed for RT-DETR!
    # The transformer architecture with bipartite matching ensures non-duplicate detections
    logger.debug(f"Total detections after filtering: {len(boxes_filtered)}")

    return {
        "boxes": boxes_filtered,
        "labels": labels_filtered,
        "scores": scores_filtered,
    }


def run_inference_tvm(
    mod: tvm.IRModule,
    image_tensor: torch.Tensor,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    original_image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, torch.Tensor]:
    """Run inference using TVM with C Static target for RT-DETR models

    Args:
        mod: TVM IRModule to execute
        image_tensor: Input image as PyTorch tensor (3D or 4D)
        score_threshold: Minimum confidence score for detections
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

    # Ensure correct shape (640x640 for RT-DETR)
    if image_tensor.shape[2:] != (640, 640):
        import torchvision.transforms.functional as F

        image_tensor = F.resize(image_tensor, [640, 640])

    try:
        # RT-DETR output shape
        # [batch, num_queries, num_classes + 4]
        # Typically: [1, 300, 84] where 84 = 4 (box) + 80 (COCO classes)
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

        # Apply post-processing (no NMS needed for RT-DETR)
        detections = apply_rtdetr_postprocessing(raw_output, score_threshold)

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
    model_name: str = "rtdetr-l",
    image_url: Optional[str] = None,
    use_tvm: bool = False,
    compare: bool = False,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> ModelTestResult:
    """Main function to run RT-DETR object detection pipeline

    Args:
        model_name: RT-DETR model variant name (e.g., 'rtdetr-l', 'rtdetr-x')
        image_url: URL or path to image, or None for default
        use_tvm: Use TVM C Static compilation
        compare: Compare PyTorch vs TVM results
        score_threshold: Minimum confidence score for detections

    Returns:
        ModelTestResult with success=True if successful, success=False if failed
    """
    tvm_compile_success = None
    tvm_inference_success = None
    comparison_data = None

    try:
        # Load RT-DETR model
        model = load_rtdetr_model(model_name)
        logger.info(f"Loaded RT-DETR model: {model_name}")

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
            pytorch_detections = run_inference(model, image, score_threshold)

            logger.info("\nCompiling model with TVM...")
            try:
                tvm_mod = prepare_model_for_tvm(model, DEFAULT_INPUT_SHAPE)
                tvm_compile_success = True

                logger.info("Running TVM inference...")
                tvm_detections = run_inference_tvm(
                    tvm_mod,
                    image_tensor,
                    score_threshold,
                    original_image_size=original_image_size,
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
                tvm_mod = prepare_model_for_tvm(model, DEFAULT_INPUT_SHAPE)
                tvm_compile_success = True

                logger.info("Running TVM inference...")
                detections_dict = run_inference_tvm(
                    tvm_mod,
                    image_tensor,
                    score_threshold,
                    original_image_size=original_image_size,
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
            logger.info("\nRunning RT-DETR inference...")
            detections_dict = run_inference(model, image, score_threshold)

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


def get_all_rtdetr_models() -> List[str]:
    """Get list of all available RT-DETR models

    Returns:
        List of RT-DETR model names
    """
    return RTDETR_MODELS.copy()


def test_multiple_models(
    image_url=None,
    max_models=None,
    model_filter=None,
    log_file=None,
    score_threshold=DEFAULT_SCORE_THRESHOLD,
):
    """Test multiple RT-DETR models

    Args:
        image_url: URL or path to test image
        max_models: Maximum number of models to test
        model_filter: List of model name substrings to filter by
        log_file: Optional path to CSV log file
        score_threshold: Minimum confidence score for detections
    """
    logger.debug("\nGetting RT-DETR model list...")

    all_models = get_all_rtdetr_models()

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

            result = main(model_name, image_url, False, False, score_threshold)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT-DETR Object Detection Model Tester")
    parser.add_argument(
        "--model",
        type=str,
        default="rtdetr-l",
        help="RT-DETR model variant (default: rtdetr-l, options: rtdetr-l, rtdetr-x)",
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
        help="Test all available RT-DETR models (rtdetr-l and rtdetr-x)",
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

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose, args.quiet)

    # If compare is set, ensure tvm is also set
    if args.compare:
        args.tvm = True

    if args.test_all:
        # Test all RT-DETR models (only 2 models, so no parallel needed)
        test_multiple_models(
            image_url=args.image,
            max_models=None,
            model_filter=None,
            log_file=args.log_file,
            score_threshold=args.score_threshold,
        )
    else:
        # Test single model
        result = main(
            model_name=args.model,
            image_url=args.image,
            use_tvm=args.tvm,
            compare=args.compare,
            score_threshold=args.score_threshold,
        )
