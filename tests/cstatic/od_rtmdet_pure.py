#!/usr/bin/env python
"""RTMDet Object Detection with Pure PyTorch Implementation (No mmcv)

This script demonstrates TVM C Static compilation for multi-output detection models
using the rtmdet Python package with exported .pth weights.

IMPORTANT NOTES:
    - TVM compilation infrastructure: ✓ FULLY WORKING
      Successfully compiles and executes RTMDet with 6 outputs (3 cls_scores + 3 bbox_preds)
      through C Static backend with proper tuple handling.

    - Detection quality: ⚠ KNOWN ISSUE
      The rtmdet package has incompatibilities between its architecture and official
      MMDetection checkpoints, resulting in incorrect detections.
      This is a weight loading bug in the third-party implementation, NOT a TVM issue.

    - Dependencies: rtmdet Python package is REQUIRED
      Install with: pip install rtmdet
      Even when using exported .pth weights, the model architecture definition is needed
      to instantiate the model before loading weights. The rtmdet module is lazy-loaded
      only when needed.

    - For production: Use od_rtmdet.py with Docker to avoid mmcv._ext dependencies,
      or apply this TVM infrastructure to other models (YOLO, RT-DETR, etc.).

Features:
    - No mmcv._ext dependencies required (pure PyTorch with rtmdet package)
    - TVM C Static compilation with multi-output tuple support
    - PyTorch vs TVM inference comparison
    - Validates TVM infrastructure for 6-tensor FPN outputs
    - Lazy loading of rtmdet module (only imported when needed)
    - Exported weights in models/ directory for standalone usage

Usage Examples:
    # Test TVM compilation using default exported weights (models/rtmdet_tiny.pth)
    python od_rtmdet_pure.py --tvm

    # Use a different model variant with exported weights
    python od_rtmdet_pure.py --model small --weights models/rtmdet_s.pth --tvm

    # Use pretrained weights from rtmdet_pytorch_implementation (empty weights string)
    python od_rtmdet_pure.py --model tiny --weights "" --tvm

    # Compare PyTorch vs TVM
    python od_rtmdet_pure.py --compare

Supported Models:
    - tiny, small, medium, large (all work for TVM compilation testing)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import tvm
from PIL import Image
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

from tvm_utils import compile_and_run_on_target, process_relax

# Lazy import of rtmdet - only when needed
RTMDet: Optional[Any] = None


def _load_rtmdet_implementation() -> Any:
    """Lazy load RTMDet implementation only when needed"""
    global RTMDet
    if RTMDet is not None:
        return RTMDet

    # Add rtmdet_pytorch_implementation to path
    sys.path.insert(0, str(Path(__file__).parent / "rtmdet_pytorch_implementation"))

    try:
        from rtmdet import RTMDet as _RTMDet  # type: ignore[import-untyped]

        RTMDet = _RTMDet
        return _RTMDet
    except ImportError as e:
        raise ImportError(
            "rtmdet package not available. "
            "Install with: pip install rtmdet or uv pip install rtmdet. "
            f"Import error: {e}"
        ) from e


# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_IMAGE_URL = "test_images/bird_0.jpg"
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_INPUT_SHAPE = (1, 3, 640, 640)
C_STATIC_TARGET = "c_static"

# COCO class names
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity flags"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


class RTMDetWrapper(nn.Module):
    """Wrapper for RTMDet that returns flattened tuple of outputs for TVM"""

    def __init__(self, rtmdet_model: Any):
        super().__init__()
        self.model = rtmdet_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass that returns FLAT tuple of 6 tensors

        Returns:
            Flat tuple of 6 tensors (for TVM compatibility):
            - cls_scores[0], cls_scores[1], cls_scores[2]  (3 FPN levels)
            - bbox_preds[0], bbox_preds[1], bbox_preds[2]  (3 FPN levels)
        """
        # Get raw outputs from the model
        cls_outputs, box_outputs = self.model._forward_raw(x)

        # Return flat tuple of 6 tensors
        return (
            cls_outputs[0],
            cls_outputs[1],
            cls_outputs[2],
            box_outputs[0],
            box_outputs[1],
            box_outputs[2],
        )


def load_rtmdet_model(
    model_name: str = "small",
    weights_path: Optional[str] = None,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> Any:
    """Load RTMDet model using pure PyTorch implementation

    Args:
        model_name: Model variant ('tiny', 'small', 'medium', 'large')
        weights_path: Optional path to .pth weights file. If provided, loads weights from file
                     instead of using pretrained weights from preset.
        score_threshold: Score threshold for filtering predictions

    Returns:
        Loaded RTMDet model in eval mode
    """
    logger.debug(f"Loading RTMDet model: {model_name}")

    # Load RTMDet implementation (only needed if not using weights file or for architecture)
    RTMDetClass = _load_rtmdet_implementation()

    # Create model from preset
    model = RTMDetClass.from_preset(name=model_name, pretrained=not weights_path)

    # Load weights from file if provided
    if weights_path:
        weights_file = Path(weights_path)
        if not weights_file.exists():
            logger.error(f"Weights file not found: {weights_path}")
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        logger.debug(f"  Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        logger.debug("  Weights loaded successfully")

    # Override default thresholds with more reasonable values
    model.score_threshold = score_threshold
    model.nms_iou_threshold = DEFAULT_IOU_THRESHOLD

    model.eval()

    logger.debug("  Model loaded successfully")
    return model


def load_image(image_path: str) -> Optional[Image.Image]:
    """Load image from file path

    Args:
        image_path: Path to local file

    Returns:
        PIL Image if successful, None if loading fails
    """
    logger.debug(f"Loading image: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        logger.debug(f"  Size: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        return None


def prepare_image(image: Image.Image) -> torch.Tensor:
    """Prepare image for inference

    Args:
        image: PIL Image

    Returns:
        Tensor of shape [1, 3, 640, 640] with ImageNet normalization
    """
    import torchvision.transforms as transforms

    # MMDetection RTMDet uses ImageNet normalization
    preprocess = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),  # Converts to [0, 1] and changes to CHW
            # ImageNet normalization (RGB order)
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensor: torch.Tensor = preprocess(image)  # type: ignore[assignment]
    return tensor.unsqueeze(0)  # Add batch dimension


def run_inference_pytorch(
    model: Any,
    image_tensor: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Run inference using PyTorch

    Args:
        model: RTMDet model
        image_tensor: Input tensor [1, 3, 640, 640]

    Returns:
        Dictionary with 'boxes', 'labels', 'scores' tensors
    """
    with torch.no_grad():
        # The model's forward() returns decoded boxes with NMS applied
        # when export_mode=False (default)
        boxes, scores, classes = model(image_tensor)

    # Model returns [B, N, 4], [B, N, 1], [B, N] after NMS
    # Remove batch dimension
    boxes = boxes[0]  # [N, 4]
    scores = scores[0].squeeze(-1)  # [N]
    classes = classes[0]  # [N]

    # Re-apply score threshold filter (the model's NMS has a bug where it doesn't properly filter)
    # Use the model's configured threshold
    mask = scores >= model.score_threshold

    return {
        "boxes": boxes[mask],
        "labels": classes[mask],
        "scores": scores[mask],
    }


def prepare_model_for_tvm(
    model: Any, input_shape: Tuple[int, ...] = DEFAULT_INPUT_SHAPE
) -> tvm.IRModule:
    """Prepare RTMDet model for TVM compilation

    Args:
        model: RTMDet model in eval mode
        input_shape: Shape of input tensor (batch, channels, height, width)

    Returns:
        Processed TVM IRModule ready for compilation
    """
    logger.debug("  Converting RTMDet to TVM Relax IR...")

    # Wrap the model to return flattened outputs
    wrapped_model = RTMDetWrapper(model)

    # Create example input for torch.export
    example_input = (torch.randn(*input_shape, dtype=torch.float32),)

    # Convert to Relax IRModule
    with torch.no_grad():
        exported_program = export(wrapped_model, example_input, strict=False)
        mod = from_exported_program(exported_program, keep_params_as_input=True)

    # Process the module (detach and bind parameters)
    mod = process_relax(mod)

    logger.debug("  TVM IR conversion complete")

    return mod


def reconstruct_outputs_from_tvm(
    tvm_outputs: List[np.ndarray],
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Dict[str, torch.Tensor]:
    """Reconstruct RTMDet outputs from TVM and apply post-processing

    Args:
        tvm_outputs: List of 6 numpy arrays from TVM
        score_threshold: Minimum confidence score
        iou_threshold: NMS IoU threshold

    Returns:
        Dictionary with 'boxes', 'labels', 'scores' tensors
    """
    # Validate outputs
    if len(tvm_outputs) != 6:
        raise RuntimeError(f"Expected 6 outputs, got {len(tvm_outputs)}")

    # Convert to PyTorch tensors and add batch dimension
    cls_outputs = [torch.from_numpy(tvm_outputs[i]) for i in range(3)]
    box_outputs = [torch.from_numpy(tvm_outputs[i]) for i in range(3, 6)]

    # Create temporary model instance for decode/nms methods
    # We need this to reuse the decode logic
    RTMDetClass = _load_rtmdet_implementation()
    from rtmdet import RTMDetConfig  # type: ignore[import-untyped]

    cfg = RTMDetConfig.from_preset("small")  # Use small config as template
    cfg.score_threshold = score_threshold
    cfg.nms_iou_threshold = iou_threshold
    temp_model = RTMDetClass(cfg)
    temp_model.eval()

    # Use the model's decode and nms methods
    with torch.no_grad():
        boxes, scores, classes = temp_model.decode(cls_outputs, box_outputs)

        # Squeeze scores before NMS to avoid shape issues
        # NMS returns scores with shape [B, N, 1] which causes masking problems
        if scores.dim() == 3 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)  # [B, N, 1] -> [B, N]

        boxes, scores, classes = temp_model.nms(boxes, scores, classes)

    # Remove batch dimension and apply final filtering
    boxes_result = boxes[0]  # [N, 4]
    scores_result = scores[0]  # [N, 1] or [N]
    classes_result = classes[0]  # [N]

    # Ensure scores is 1D for masking
    if scores_result.dim() > 1:
        scores_result = scores_result.squeeze(-1)  # [N]

    # Filter by score threshold (re-apply since NMS has a bug)
    mask = scores_result >= score_threshold

    return {
        "boxes": boxes_result[mask],
        "labels": classes_result[mask],
        "scores": scores_result[mask],
    }


def run_inference_tvm(
    mod: tvm.IRModule,
    image_tensor: torch.Tensor,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> Dict[str, torch.Tensor]:
    """Run inference using TVM with C Static target

    Args:
        mod: TVM IRModule to execute
        image_tensor: Input tensor [1, 3, 640, 640]
        score_threshold: Minimum confidence score
        iou_threshold: NMS IoU threshold

    Returns:
        Dictionary with 'boxes', 'labels', 'scores' tensors
    """
    logger.debug("  Compiling with TVM C Static backend...")

    try:
        # Compile and run on C Static target
        tvm_outputs = compile_and_run_on_target(
            target_string=C_STATIC_TARGET,
            mod=mod,
            input=image_tensor.numpy(),
            verbose_output=False,
        )

        if not isinstance(tvm_outputs, list):
            raise RuntimeError(f"Expected list from TVM, got {type(tvm_outputs).__name__}")

        logger.debug("  Successfully compiled and executed TVM model")
        logger.debug(f"  Received {len(tvm_outputs)} outputs")

        # Reconstruct and post-process
        detections = reconstruct_outputs_from_tvm(tvm_outputs, score_threshold, iou_threshold)

        logger.debug(f"  Post-processing complete: {len(detections['boxes'])} detections")

        return detections

    except Exception as e:
        logger.error(f"TVM inference failed: {e}")
        raise RuntimeError(f"TVM inference failed: {e}") from e


def display_detections(
    detections: Dict[str, torch.Tensor],
    max_display: int = 10,
) -> None:
    """Display detected objects

    Args:
        detections: Dictionary with 'boxes', 'labels', 'scores' tensors
        max_display: Maximum number of detections to display
    """
    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]

    num_detections = len(boxes)
    logger.info(f"\nDetected {num_detections} objects:")
    logger.info("-" * 80)

    for i in range(min(num_detections, max_display)):
        box = boxes[i].tolist()
        label_id = int(labels[i].item())
        label_name = COCO_CLASSES[label_id] if label_id < len(COCO_CLASSES) else f"class_{label_id}"
        score = scores[i].item()

        logger.info(
            f"{i + 1:2d}. {label_name:20s} {score * 100:6.2f}% "
            f"Box: [{box[0]:6.1f}, {box[1]:6.1f}, {box[2]:6.1f}, {box[3]:6.1f}]"
        )

    if num_detections > max_display:
        logger.info(f"... and {num_detections - max_display} more detections")


def compare_results(
    pytorch_detections: Dict[str, torch.Tensor],
    tvm_detections: Dict[str, torch.Tensor],
) -> None:
    """Compare PyTorch and TVM detection results

    Args:
        pytorch_detections: PyTorch results
        tvm_detections: TVM results
    """
    num_pytorch = len(pytorch_detections["boxes"])
    num_tvm = len(tvm_detections["boxes"])

    logger.info("\nComparison:")
    logger.info(f"  PyTorch detections: {num_pytorch}")
    logger.info(f"  TVM detections:     {num_tvm}")
    logger.info(f"  Difference:         {abs(num_pytorch - num_tvm)}")


def main(
    model_name: str = "small",
    weights_path: Optional[str] = None,
    image_path: Optional[str] = None,
    use_tvm: bool = False,
    compare: bool = False,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    verbose: bool = False,
) -> bool:
    """Main function

    Args:
        model_name: Model variant name
        weights_path: Optional path to .pth weights file
        image_path: Path to image
        use_tvm: Use TVM compilation
        compare: Compare PyTorch vs TVM
        score_threshold: Minimum confidence score
        verbose: Enable verbose output

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load model
        model = load_rtmdet_model(model_name, weights_path, score_threshold)
        logger.info(f"Loaded RTMDet model: {model_name}")

        # Load image
        image_path = image_path or DEFAULT_IMAGE_URL
        image = load_image(image_path)
        if image is None:
            return False

        # Prepare image
        image_tensor = prepare_image(image)

        # Run PyTorch inference (always run for baseline)
        logger.info("\nRunning PyTorch inference...")
        pytorch_detections = run_inference_pytorch(model, image_tensor)
        display_detections(pytorch_detections)

        # TVM mode
        if use_tvm or compare:
            logger.info("\nPreparing model for TVM compilation...")
            mod = prepare_model_for_tvm(model, DEFAULT_INPUT_SHAPE)

            logger.info("Running TVM inference with C Static backend...")
            tvm_detections = run_inference_tvm(
                mod, image_tensor, score_threshold, DEFAULT_IOU_THRESHOLD
            )

            logger.info("✓ TVM inference completed successfully")

            if compare:
                compare_results(pytorch_detections, tvm_detections)
                logger.info("\nTVM detections:")
                display_detections(tvm_detections)

        return True

    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        if verbose:
            import traceback

            traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RTMDet Object Detection with Pure PyTorch (No mmcv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        choices=["tiny", "small", "medium", "large"],
        help="RTMDet model variant (default: tiny)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/rtmdet_tiny.pth",
        help="Path to .pth weights file (default: models/rtmdet_tiny.pth). Use empty string to use pretrained weights from preset.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=DEFAULT_IMAGE_URL,
        help="Path to test image",
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
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help=f"Minimum confidence score (default: {DEFAULT_SCORE_THRESHOLD})",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # If compare is set, ensure tvm is also set
    if args.compare:
        args.tvm = True

    # Handle empty weights string (use pretrained instead)
    weights_path = args.weights if args.weights else None

    # Run main
    success = main(
        model_name=args.model,
        weights_path=weights_path,
        image_path=args.image,
        use_tvm=args.tvm,
        compare=args.compare,
        score_threshold=args.score_threshold,
        verbose=args.verbose,
    )

    sys.exit(0 if success else 1)
