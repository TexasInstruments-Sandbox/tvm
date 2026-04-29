#!/usr/bin/env python
"""TorchVision Classification Model Tester

This script provides comprehensive testing and validation for TorchVision classification models,
with support for TVM compilation and comparison between PyTorch and TVM C Static backends.

Features:
    - Automatic discovery of all ImageNet classification models in TorchVision
    - Automatic extraction of preprocessing transforms from model weights
    - PyTorch inference (default)
    - TVM C Static compilation and inference (--tvm)
    - Side-by-side comparison of PyTorch vs TVM results (--compare)
    - Batch testing of multiple models with filtering and limits
    - Configuration file support for reusable test setups
    - Comprehensive logging with adjustable verbosity levels

Usage Examples:
    # Test single model with PyTorch
    python cl_torchvision.py --model resnet50

    # Test with TVM C Static compilation
    python cl_torchvision.py --model resnet18 --tvm

    # Compare PyTorch vs TVM C Static
    python cl_torchvision.py --model mobilenet_v3_small --compare

    # Test multiple models with filtering
    python cl_torchvision.py --test-all --filter resnet efficientnet --max-models 5

    # Test multiple models in parallel (much faster)
    python cl_torchvision.py --test-all --parallel

    # Parallel testing with custom worker count and logging
    python cl_torchvision.py --test-all --parallel --workers 8 --log-file results.csv

    # Parallel comparison with filtering and logging
    python cl_torchvision.py --test-all --compare --parallel --filter resnet --log-file resnet_results.csv

    # Verbose mode shows detailed information
    python cl_torchvision.py --model resnet18 --verbose

    # Quiet mode suppresses most output
    python cl_torchvision.py --model resnet18 --quiet

    # Use configuration file
    python cl_torchvision.py --config my_config.yaml

Command-Line Options:
    --model MODEL              Model name to test (default: resnet50)
    --weight WEIGHT            Weight name to use (default: None, uses DEFAULT)
    --image IMAGE              Path or URL to test image
    --test-all                 Test all available classification models
    --max-models N             Maximum number of models to test with --test-all
    --filter PATTERN [...]     Filter models by name patterns
    --parallel                 Run tests in parallel (only with --test-all)
    --workers N                Number of parallel workers (default: CPU count)
    --log-file PATH            CSV log file for appending results (with --test-all)
    --tvm                      Use TVM compilation with C Static target
    --compare                  Compare PyTorch vs TVM results (implies --tvm)
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
       - get_default_imagenet_preprocessing()

    2. Inference Backends:
       - run_inference() - PyTorch backend
       - prepare_model_for_tvm() - TVM compilation
       - run_inference_tvm() - TVM execution

    3. Comparison & Analysis:
       - compare_inference_results()
       - Helper functions for table and summary output

    4. Orchestration:
       - main() - Single model testing
       - test_multiple_models() - Batch testing

Dependencies:
    - torch, torchvision: Model definitions and weights
    - tvm: Compiler framework for ML models
    - PIL: Image loading
    - numpy: Numerical operations
    - requests: Fetching ImageNet labels
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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import tvm
from PIL import Image
from torch.export import export
from tvm.relax.frontend.torch import from_exported_program

from tvm_utils import compile_and_run_on_target, process_relax

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_IMAGE_URL = "test_images/YellowLabradorLooking_new.jpg"
IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
IMAGENET_NUM_CLASSES = 1000
DEFAULT_IMAGENET_MEAN = [0.485, 0.456, 0.406]
DEFAULT_IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_RESIZE_SIZE = 256
DEFAULT_CROP_SIZE = 224
DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)
C_STATIC_TARGET = "c_static"
LLVM_TARGET = "llvm"

# Comparison tolerances
RTOL_COMPARISON = 1e-3
ATOL_COMPARISON = 1e-5

# Cache for ImageNet labels
_IMAGENET_LABELS_CACHE: Optional[List[str]] = None

# Thread lock for CSV file writing
_CSV_WRITE_LOCK = threading.Lock()


# Data classes for structured data
@dataclass
class ModelTestResult:
    """Results from testing a single model"""

    model: Optional[nn.Module]
    preprocessing: Optional[Union[transforms.Compose, Callable]]
    image: Optional[Image.Image]
    predictions: Optional[List[Tuple[str, float, int]]]
    success: bool = True  # True if test succeeded, False if failed
    error_message: Optional[str] = None  # Error message if success=False
    comparison: Optional[Dict[str, Any]] = None
    tvm_compile_success: Optional[bool] = None  # True if TVM compilation succeeded
    tvm_inference_success: Optional[bool] = None  # True if TVM inference succeeded


@dataclass
class ComparisonResult:
    """Results from comparing PyTorch vs TVM"""

    model_name: str
    top1_match: Optional[bool]
    top5_match: Optional[bool]
    error: bool = False
    error_message: Optional[str] = None


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

    logging.basicConfig(level=level, format="%(message)s")


def append_result_to_log(
    log_file: str,
    model_name: str,
    top_prediction: Optional[str] = None,
    confidence: Optional[float] = None,
    top1_match: Optional[bool] = None,
    top5_match: Optional[bool] = None,
    tvm_compile_success: Optional[bool] = None,
    tvm_inference_success: Optional[bool] = None,
    error: bool = False,
    error_message: Optional[str] = None,
) -> None:
    """Append a test result to the CSV log file (thread-safe)

    Args:
        log_file: Path to the CSV log file
        model_name: Name of the model tested
        top_prediction: Top-1 prediction label
        confidence: Confidence score (0-100)
        top1_match: Whether top-1 predictions match (PyTorch vs TVM)
        top5_match: Whether top-5 predictions match (PyTorch vs TVM)
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
        top_prediction or "N/A",
        f"{confidence:.2f}" if confidence is not None else "N/A",
        bool_to_str(top1_match),
        bool_to_str(top5_match),
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
                        "top_prediction",
                        "confidence",
                        "top1_match",
                        "top5_match",
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
    model_name: str = "resnet50", weight_name: Optional[str] = None
) -> Tuple[nn.Module, Union[transforms.Compose, Callable], Optional[Any]]:
    """Load model and automatically determine preprocessing transforms

    Args:
        model_name: Name of the TorchVision model to load
        weight_name: Specific weight name to use, or None for default

    Returns:
        Tuple of (model, preprocessing_transforms, weight_enum)
    """
    logger.debug(f"Loading model: {model_name}")

    # Get the model function
    model_func = getattr(models, model_name)

    weight: Optional[Any] = None

    # Try to get weights enum using the official API
    try:
        weights_enum = models.get_model_weights(model_name)

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

    except (AttributeError, ValueError):
        # Fallback for models without weights enum
        logger.debug("  No weights enum found, using default")
        model = model_func(weights="DEFAULT")
        preprocessing = get_default_imagenet_preprocessing()

    # Set to evaluation mode
    model.eval()

    return model, preprocessing, weight


def get_preprocessing_from_weight(
    weight: Any, model_name: str
) -> Union[transforms.Compose, Callable]:
    """Extract preprocessing transforms from weight metadata

    Args:
        weight: TorchVision weight enum instance
        model_name: Name of the model (for logging)

    Returns:
        Callable preprocessing transforms (either transforms.Compose or ImageClassification)
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
    if hasattr(weight, "meta") and weight.meta:
        meta = weight.meta
        transforms_list = []

        # Resize transform
        if "resize_size" in meta:
            resize_size = meta["resize_size"]
            transforms_list.append(transforms.Resize(resize_size))
        elif "min_size" in meta:
            min_size = meta["min_size"]
            transforms_list.append(transforms.Resize(min_size))
        else:
            transforms_list.append(transforms.Resize(DEFAULT_RESIZE_SIZE))

        # Crop transform
        if "crop_size" in meta:
            crop_size = meta["crop_size"]
            transforms_list.append(transforms.CenterCrop(crop_size))
        else:
            transforms_list.append(transforms.CenterCrop(DEFAULT_CROP_SIZE))

        # Convert to tensor
        transforms_list.append(transforms.ToTensor())

        # Normalization
        if "mean" in meta and "std" in meta:
            mean = meta["mean"]
            std = meta["std"]
            transforms_list.append(transforms.Normalize(mean=mean, std=std))
        else:
            transforms_list.append(
                transforms.Normalize(mean=DEFAULT_IMAGENET_MEAN, std=DEFAULT_IMAGENET_STD)
            )

        preprocessing = transforms.Compose(transforms_list)
        return preprocessing

    # Fallback to default
    return get_default_imagenet_preprocessing()


def get_default_imagenet_preprocessing() -> transforms.Compose:
    """Default ImageNet preprocessing as fallback

    Returns:
        Standard ImageNet preprocessing transforms
    """
    return transforms.Compose(
        [
            transforms.Resize(DEFAULT_RESIZE_SIZE),
            transforms.CenterCrop(DEFAULT_CROP_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_IMAGENET_MEAN, std=DEFAULT_IMAGENET_STD),
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


def load_imagenet_labels(force_reload: bool = False) -> List[str]:
    """Load ImageNet class labels (cached after first call)

    Args:
        force_reload: If True, bypass cache and reload from URL

    Returns:
        List of 1000 ImageNet class labels
    """
    global _IMAGENET_LABELS_CACHE

    if _IMAGENET_LABELS_CACHE is not None and not force_reload:
        return _IMAGENET_LABELS_CACHE

    try:
        response = requests.get(IMAGENET_LABELS_URL, timeout=10)
        response.raise_for_status()
        labels = response.text.strip().split("\n")
        _IMAGENET_LABELS_CACHE = labels
        return labels

    except requests.RequestException as e:
        logger.warning(f"Could not load labels: {e}")
        # Fallback: create dummy labels
        return [f"class_{i}" for i in range(IMAGENET_NUM_CLASSES)]


def run_inference(
    model: nn.Module, image_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference on the preprocessed image

    Args:
        model: PyTorch model in eval mode
        image_tensor: Preprocessed image tensor with shape (C, H, W)

    Returns:
        Tuple of (raw_outputs, probabilities)
    """
    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    batch_tensor = image_tensor.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(batch_tensor)

    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    return outputs[0], probabilities


def display_results(
    probabilities: torch.Tensor, labels: List[str], top_k: int = 5
) -> List[Tuple[str, float, int]]:
    """Display top-k predictions

    Args:
        probabilities: Probability tensor for all classes
        labels: List of class labels
        top_k: Number of top predictions to display

    Returns:
        List of tuples (label, probability, class_index)
    """
    logger.info(f"\nTop {top_k} Predictions:")
    logger.info("-" * 50)

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []
    for i in range(top_k):
        idx = int(top_indices[i].item())
        prob = top_probs[i].item()
        label = labels[idx]

        logger.info(f"{i + 1:2d}. {label:30s} {prob * 100:6.2f}%")
        results.append((label, prob, idx))

    return results


def print_preprocessing_details(
    preprocessing: Union[transforms.Compose, Callable], weight: Optional[Any] = None
) -> None:
    """Print detailed information about preprocessing transforms

    Args:
        preprocessing: The preprocessing transforms
        weight: Optional weight enum with metadata
    """
    logger.debug("\nPreprocessing Pipeline:")

    if weight and hasattr(weight, "meta") and weight.meta:
        meta = weight.meta
        if "resize_size" in meta or "crop_size" in meta:
            resize = meta.get("resize_size", meta.get("min_size", "N/A"))
            crop = meta.get("crop_size", "N/A")
            logger.debug(f"  Resize: {resize}, Crop: {crop}")
        if "mean" in meta and "std" in meta:
            logger.debug(f"  Normalize: mean={meta['mean']}, std={meta['std']}")
    else:
        logger.debug("  Using default ImageNet preprocessing")


def torch_to_relax(torch_model: nn.Module, example_input: Tuple[torch.Tensor, ...]) -> tvm.IRModule:
    """Convert a PyTorch model to a Relax IRModule

    Args:
        torch_model: PyTorch model to convert
        example_input: Example input tuple for torch.export

    Returns:
        TVM Relax IRModule
    """
    with torch.no_grad():
        exported_program = export(torch_model, example_input)
        mod = from_exported_program(exported_program, keep_params_as_input=True)
    return mod


def prepare_model_for_tvm(
    torch_model: nn.Module, input_shape: Tuple[int, ...] = DEFAULT_INPUT_SHAPE
) -> tvm.IRModule:
    """Prepare a PyTorch model for TVM compilation

    Args:
        torch_model: PyTorch model in eval mode
        input_shape: Shape of input tensor (batch, channels, height, width)

    Returns:
        Processed TVM IRModule ready for compilation
    """
    logger.debug("  Converting to TVM Relax IR...")

    # Create example input for torch.export
    example_input = (torch.randn(*input_shape, dtype=torch.float32),)

    # Convert to Relax IRModule
    mod = torch_to_relax(torch_model, example_input)

    # Process the module (detach and bind parameters)
    mod = process_relax(mod)

    logger.debug("  TVM IR conversion complete")

    return mod


def run_inference_tvm(
    mod: tvm.IRModule, image_tensor: torch.Tensor, compare_llvm: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run inference using TVM with C Static target (optionally comparing with LLVM)

    Args:
        mod: TVM IRModule to execute
        image_tensor: Input image as PyTorch tensor (3D or 4D)
        compare_llvm: If True, also compile for LLVM and compare results

    Returns:
        Tuple of (raw_outputs, probabilities) as PyTorch tensors

    Raises:
        RuntimeError: If TVM compilation or execution fails
    """
    logger.info("Running TVM inference...")

    # Convert to numpy, add batch dimension if needed
    if image_tensor.ndim == 3:
        input_data = image_tensor.unsqueeze(0).numpy()
    else:
        input_data = image_tensor.numpy()

    llvm_result: Optional[np.ndarray] = None
    if compare_llvm:
        # Compile and run on LLVM target (reference)
        logger.debug("  Compiling for LLVM target...")
        llvm_result_raw = compile_and_run_on_target(
            target_string=LLVM_TARGET, mod=mod, input=input_data
        )
        # Extract first output if multi-output model
        if isinstance(llvm_result_raw, list):
            logger.debug(
                f"  Note: LLVM returned {len(llvm_result_raw)} outputs, using first output"
            )
            llvm_result = llvm_result_raw[0]
        else:
            llvm_result = llvm_result_raw

    # Compile and run on C Static target
    logger.debug("  Compiling for C Static target...")
    c_static_result_raw = compile_and_run_on_target(
        target_string=C_STATIC_TARGET, mod=mod, input=input_data
    )

    # Handle multi-output models: extract primary output (classification logits)
    # Classification models should return single output, but some variants may have auxiliary outputs
    if isinstance(c_static_result_raw, list):
        logger.debug(
            f"  Note: C Static returned {len(c_static_result_raw)} outputs, using first output"
        )
        c_static_result = c_static_result_raw[0]
    else:
        c_static_result = c_static_result_raw

    # Compare LLVM vs C Static if both were run
    if compare_llvm:
        assert llvm_result is not None
        max_diff = np.max(np.abs(llvm_result - c_static_result))
        matches = np.allclose(
            llvm_result, c_static_result, rtol=RTOL_COMPARISON, atol=ATOL_COMPARISON
        )

        logger.debug("  LLVM vs C Static comparison:")
        logger.debug(f"    Max difference: {max_diff:.2e}")
        logger.debug(f"    Results match: {'✓' if matches else '✗'}")

        if not matches:
            logger.warning("  LLVM and C Static results differ significantly!")

    # Convert numpy result back to PyTorch
    # c_static_result is now guaranteed to be a 2D array: (batch_size, num_classes)
    # For batch_size=1, we get the first (and only) batch element
    outputs = torch.from_numpy(c_static_result[0])
    probabilities = torch.nn.functional.softmax(outputs, dim=0)

    logger.debug("  TVM inference complete")

    return outputs, probabilities


def compare_inference_results(
    pytorch_outputs: torch.Tensor,
    tvm_outputs: torch.Tensor,
    pytorch_probs: torch.Tensor,
    tvm_probs: torch.Tensor,
    labels: List[str],
) -> Dict[str, Any]:
    """Compare PyTorch and TVM inference results

    Args:
        pytorch_outputs: Raw PyTorch outputs (logits)
        tvm_outputs: Raw TVM outputs (logits)
        pytorch_probs: PyTorch probabilities after softmax
        tvm_probs: TVM probabilities after softmax
        labels: List of class labels

    Returns:
        Dictionary with comparison metrics
    """
    # Convert to numpy for comparison
    pytorch_logits_np = pytorch_outputs.detach().numpy()
    tvm_logits_np = tvm_outputs.detach().numpy()
    pytorch_probs_np = pytorch_probs.detach().numpy()
    tvm_probs_np = tvm_probs.detach().numpy()

    # Compute differences
    logit_max_diff = np.max(np.abs(pytorch_logits_np - tvm_logits_np))
    logit_mean_diff = np.mean(np.abs(pytorch_logits_np - tvm_logits_np))
    prob_max_diff = np.max(np.abs(pytorch_probs_np - tvm_probs_np))
    prob_mean_diff = np.mean(np.abs(pytorch_probs_np - tvm_probs_np))

    # Compute cosine similarity
    logit_cos_sim = np.dot(pytorch_logits_np, tvm_logits_np) / (
        np.linalg.norm(pytorch_logits_np) * np.linalg.norm(tvm_logits_np)
    )
    prob_cos_sim = np.dot(pytorch_probs_np, tvm_probs_np) / (
        np.linalg.norm(pytorch_probs_np) * np.linalg.norm(tvm_probs_np)
    )

    # Check if results match within tolerance
    logits_match = np.allclose(
        pytorch_logits_np, tvm_logits_np, rtol=RTOL_COMPARISON, atol=ATOL_COMPARISON
    )
    probs_match = np.allclose(
        pytorch_probs_np, tvm_probs_np, rtol=RTOL_COMPARISON, atol=ATOL_COMPARISON
    )

    # Get top-5 predictions for both
    pytorch_top5_indices = np.argsort(pytorch_probs_np)[-5:][::-1]
    tvm_top5_indices = np.argsort(tvm_probs_np)[-5:][::-1]

    # Check if top-1 and top-5 match
    top1_match = pytorch_top5_indices[0] == tvm_top5_indices[0]
    top5_match = set(pytorch_top5_indices) == set(tvm_top5_indices)

    comparison = {
        "logit_max_diff": logit_max_diff,
        "logit_mean_diff": logit_mean_diff,
        "prob_max_diff": prob_max_diff,
        "prob_mean_diff": prob_mean_diff,
        "logit_cosine_similarity": logit_cos_sim,
        "prob_cosine_similarity": prob_cos_sim,
        "logits_match": logits_match,
        "probs_match": probs_match,
        "top1_match": top1_match,
        "top5_match": top5_match,
        "pytorch_top5": [(labels[i], pytorch_probs_np[i]) for i in pytorch_top5_indices],
        "tvm_top5": [(labels[i], tvm_probs_np[i]) for i in tvm_top5_indices],
    }

    # Show details if predictions don't match
    if not top1_match:
        logger.info(
            f"  PyTorch top-1: {labels[pytorch_top5_indices[0]]} ({pytorch_probs_np[pytorch_top5_indices[0]] * 100:.2f}%)"
        )
        logger.info(
            f"  TVM top-1:     {labels[tvm_top5_indices[0]]} ({tvm_probs_np[tvm_top5_indices[0]] * 100:.2f}%)"
        )

    if not top5_match:
        logger.info(f"  PyTorch top-5: {[labels[i] for i in pytorch_top5_indices]}")
        logger.info(f"  TVM top-5:     {[labels[i] for i in tvm_top5_indices]}")

    # Detailed metrics at DEBUG level
    logger.debug("\nDetailed Comparison:")
    logger.debug("  Logits:")
    logger.debug(f"    Max difference:      {logit_max_diff:.2e}")
    logger.debug(f"    Mean difference:     {logit_mean_diff:.2e}")
    logger.debug(f"    Cosine similarity:   {logit_cos_sim:.6f}")
    logger.debug(f"    Match (tol={ATOL_COMPARISON:.0e}):    {'✓' if logits_match else '✗'}")
    logger.debug("  Probabilities:")
    logger.debug(f"    Max difference:      {prob_max_diff:.2e}")
    logger.debug(f"    Mean difference:     {prob_mean_diff:.2e}")
    logger.debug(f"    Cosine similarity:   {prob_cos_sim:.6f}")
    logger.debug(f"    Match (tol={ATOL_COMPARISON:.0e}):    {'✓' if probs_match else '✗'}")

    return comparison


# Helper functions for main()
def _run_pytorch_inference(
    model: nn.Module, image_tensor: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run PyTorch inference"""
    logger.info("\nRunning PyTorch inference...")
    return run_inference(model, image_tensor)


def _run_tvm_inference(
    model: nn.Module,
    image_tensor: torch.Tensor,
    input_shape: Optional[Tuple[int, ...]] = None,
    compare_llvm: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, bool, bool]:
    """Run TVM inference with error handling and status tracking

    Args:
        model: PyTorch model to run
        image_tensor: Preprocessed input tensor (3D or 4D)
        input_shape: Shape for TVM compilation. If None, inferred from image_tensor.
                     This ensures the compiled model matches the actual input size.
        compare_llvm: Whether to also run LLVM target for comparison

    Returns:
        Tuple of (outputs, probabilities, compile_success, inference_success)

    Raises:
        Exception: If compilation or inference fails (with status info attached)
    """
    compile_success = False
    inference_success = False

    # Add batch dimension if needed for TVM (C, H, W) -> (1, C, H, W)
    if image_tensor.ndim == 3:
        original_shape = tuple(image_tensor.shape)
        image_tensor = image_tensor.unsqueeze(0)
        logger.debug(
            f"  Adding batch dimension for TVM: {original_shape} -> {tuple(image_tensor.shape)}"
        )

    # If input_shape not provided, infer from image_tensor
    if input_shape is None:
        input_shape = tuple(image_tensor.shape)

    try:
        tvm_mod = prepare_model_for_tvm(model, input_shape=input_shape)
        compile_success = True
    except Exception as e:
        logger.error(f"TVM compilation failed: {e}")
        # Attach status info to exception
        exc = Exception(f"TVM compilation failed: {e}")
        exc.tvm_compile_success = compile_success  # type: ignore
        exc.tvm_inference_success = inference_success  # type: ignore
        raise exc from e

    try:
        outputs, probs = run_inference_tvm(tvm_mod, image_tensor, compare_llvm=compare_llvm)
        inference_success = True
        return outputs, probs, compile_success, inference_success
    except Exception as e:
        logger.error(f"TVM inference failed: {e}")
        # Attach status info to exception
        exc = Exception(f"TVM inference failed: {e}")
        exc.tvm_compile_success = compile_success  # type: ignore
        exc.tvm_inference_success = inference_success  # type: ignore
        raise exc from e


def _run_comparison(
    model: nn.Module, image_tensor: torch.Tensor, labels: List[str]
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], bool, bool]:
    """Run both PyTorch and TVM inference and compare

    Returns:
        Tuple of (outputs, probabilities, comparison, compile_success, inference_success)
    """
    pytorch_outputs, pytorch_probs = _run_pytorch_inference(model, image_tensor)
    tvm_outputs, tvm_probs, compile_success, inference_success = _run_tvm_inference(
        model, image_tensor, compare_llvm=False
    )

    comparison = compare_inference_results(
        pytorch_outputs, tvm_outputs, pytorch_probs, tvm_probs, labels
    )

    return pytorch_outputs, pytorch_probs, comparison, compile_success, inference_success


def _print_comparison_table(comparison_results: List[Dict[str, Any]]) -> None:
    """Print comparison table for multiple models with TVM status"""
    logger.info("\nComparison Table: PyTorch vs TVM C Static")
    logger.info(f"{'-' * 95}")
    logger.info(
        f"{'Model':<30s} {'Top-1':<10s} {'Top-5':<10s} {'TVM Compile':<15s} {'TVM Inference':<15s}"
    )
    logger.info(f"{'-' * 95}")

    for comp in comparison_results:
        model_name = comp["model"][:28]
        top1_str = "ERROR" if comp["error"] else ("✓" if comp["top1_match"] else "✗")
        top5_str = "ERROR" if comp["error"] else ("✓" if comp["top5_match"] else "✗")

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
            f"{model_name:<30s} {top1_str:<10s} {top5_str:<10s} {compile_str:<15s} {inference_str:<15s}"
        )


def _print_comparison_summary(comparison_results: List[Dict[str, Any]]) -> None:
    """Print comparison summary statistics including TVM status"""
    valid_comps = [c for c in comparison_results if not c["error"]]
    if not valid_comps:
        return

    top1_failures = sum(1 for c in valid_comps if not c["top1_match"])
    top5_failures = sum(1 for c in valid_comps if not c["top5_match"])
    total = len(valid_comps)

    # TVM status statistics
    compile_successes = sum(1 for c in valid_comps if c.get("tvm_compile_success") is True)
    inference_successes = sum(1 for c in valid_comps if c.get("tvm_inference_success") is True)

    logger.info("\nComparison Summary:")
    logger.info(f"  Models tested: {total}")
    logger.info(
        f"  Top-1 match rate: {(total - top1_failures) / total * 100:.1f}% ({total - top1_failures}/{total})"
    )
    logger.info(
        f"  Top-5 match rate: {(total - top5_failures) / total * 100:.1f}% ({total - top5_failures}/{total})"
    )
    logger.info(
        f"  TVM compile success rate: {compile_successes / total * 100:.1f}% ({compile_successes}/{total})"
    )
    logger.info(
        f"  TVM inference success rate: {inference_successes / total * 100:.1f}% ({inference_successes}/{total})"
    )


def main(
    model_name: str = "resnet50",
    weight_name: Optional[str] = None,
    image_url: Optional[str] = None,
    use_tvm: bool = False,
    compare: bool = False,
) -> ModelTestResult:
    """Main function to run the complete pipeline with automatic preprocessing

    Args:
        model_name: Name of the TorchVision model
        weight_name: Specific weight name or None for default
        image_url: URL or path to image, or None for default
        use_tvm: Use TVM C Static compilation
        compare: Compare PyTorch vs TVM C Static

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
            predictions=None,
            success=False,
            error_message="Failed to load image",
        )

    # Preprocess image
    image_tensor = preprocessing(image)
    assert isinstance(image_tensor, torch.Tensor)

    # Load labels
    labels = load_imagenet_labels()

    # Run inference (TVM or PyTorch or both for comparison)
    comparison_data = None
    tvm_compile_success = None
    tvm_inference_success = None

    try:
        if compare:
            (
                raw_outputs,
                probabilities,
                comparison_data,
                tvm_compile_success,
                tvm_inference_success,
            ) = _run_comparison(model, image_tensor, labels)
        elif use_tvm:
            raw_outputs, probabilities, tvm_compile_success, tvm_inference_success = (
                _run_tvm_inference(model, image_tensor)
            )
        else:
            raw_outputs, probabilities = _run_pytorch_inference(model, image_tensor)
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
            predictions=None,
            success=False,
            error_message=error_message,
            tvm_compile_success=tvm_compile_success,
            tvm_inference_success=tvm_inference_success,
        )

    # Display results
    results = display_results(probabilities, labels, top_k=5)

    # Additional analysis
    logger.info("\nModel Statistics:")
    logger.info(f"  Confidence: {torch.max(probabilities).item() * 100:.2f}%")
    logger.info(
        f"  Entropy: {-torch.sum(probabilities * torch.log(probabilities + 1e-8)).item():.3f}"
    )

    return ModelTestResult(
        model=model,
        preprocessing=preprocessing,
        image=image,
        predictions=results,
        comparison=comparison_data,
        tvm_compile_success=tvm_compile_success,
        tvm_inference_success=tvm_inference_success,
    )


def get_all_classification_models():
    """Programmatically discover all classification models in TorchVision

    Returns:
        List of tuples (model_name, default_weight_name)
    """
    import inspect
    import typing

    classification_models = []

    # Get all attributes from torchvision.models
    for name in dir(models):
        # Skip private attributes and non-callables
        if name.startswith("_"):
            continue

        attr = getattr(models, name)

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

            # Check if this is an ImageNet classification model (1000 classes)
            if hasattr(default_weight, "meta") and default_weight.meta:
                meta = default_weight.meta

                # ImageNet models typically have 1000 classes
                if "categories" in meta:
                    num_classes = len(meta["categories"])
                    if num_classes == 1000:
                        classification_models.append((name, default_weight.name))
                elif "num_classes" in meta and meta["num_classes"] == 1000:
                    classification_models.append((name, default_weight.name))

        except Exception as e:
            # Skip models that fail inspection
            logger.debug(f"  Skipping {name}: {e}")
            continue

    return classification_models


def test_multiple_models(
    image_url=None, max_models=None, model_filter=None, use_tvm=False, compare=False, log_file=None
):
    """Test the automatic preprocessing with all available classification models

    Args:
        image_url: URL or path to test image. If None, uses default cat image.
        max_models: Maximum number of models to test. If None, tests all models.
        model_filter: Optional list of model name substrings to filter by (e.g., ['resnet', 'efficientnet'])
        use_tvm: Use TVM compilation with C Static target.
        compare: Compare PyTorch and TVM C Static results.
        log_file: Optional path to CSV log file for appending results.
    """
    logger.debug("\nDiscovering TorchVision classification models...")

    # Automatically discover all classification models
    all_models = get_all_classification_models()

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
        image_url = "test_images/YellowLabradorLooking_new.jpg"

    # Track results
    successful_tests = []
    failed_tests = []
    comparison_results = []  # For compare mode

    for i, (model_name, weight_name) in enumerate(all_models, 1):
        try:
            logger.info(f"\n[{i}/{len(all_models)}] Testing {model_name}...")

            result = main(model_name, weight_name, image_url, use_tvm, compare)

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
                            "top1_match": None,
                            "top5_match": None,
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
                        "top1_match": result.comparison["top1_match"],
                        "top5_match": result.comparison["top5_match"],
                        "tvm_compile_success": result.tvm_compile_success,
                        "tvm_inference_success": result.tvm_inference_success,
                        "error": False,
                    }
                )

            # Success path - predictions should be populated
            assert result.predictions is not None, (
                "Predictions should not be None for successful results"
            )
            top_prediction = result.predictions[0]
            logger.info(f"  Result: {top_prediction[0][:40]} ({top_prediction[1] * 100:.1f}%)")

            successful_tests.append(
                {
                    "model": model_name,
                    "weight": weight_name,
                    "top_prediction": top_prediction[0],
                    "confidence": top_prediction[1] * 100,
                }
            )

            # Append to log file
            if log_file:
                append_result_to_log(
                    log_file=log_file,
                    model_name=model_name,
                    top_prediction=top_prediction[0],
                    confidence=top_prediction[1] * 100,
                    top1_match=result.comparison.get("top1_match") if result.comparison else None,
                    top5_match=result.comparison.get("top5_match") if result.comparison else None,
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
                        "top1_match": None,
                        "top5_match": None,
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
            pred = test["top_prediction"][:35]
            logger.debug(f"  ✓ {test['model']:25s} -> {pred:35s} ({test['confidence']:5.1f}%)")

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
):
    """Test multiple models in parallel using concurrent.futures

    Args:
        image_url: URL or path to test image. If None, uses default cat image.
        max_models: Maximum number of models to test. If None, tests all models.
        model_filter: Optional list of model name substrings to filter by
        use_tvm: Use TVM compilation with C Static target.
        compare: Compare PyTorch and TVM C Static results.
        max_workers: Maximum number of parallel workers. If None, uses CPU count.
        log_file: Optional path to CSV log file for appending results.

    Returns:
        Tuple of (successful_tests, failed_tests)
    """
    import os

    logger.debug("\nDiscovering TorchVision classification models...")

    # Automatically discover all classification models
    all_models = get_all_classification_models()

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
        image_url = "test_images/YellowLabradorLooking_new.jpg"

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
            executor.submit(main, model_name, weight_name, image_url, use_tvm, compare): (
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
                                "top1_match": None,
                                "top5_match": None,
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
                    # Success - predictions should be populated
                    assert result.predictions is not None, (
                        "Predictions should not be None for successful results"
                    )
                    top_pred = result.predictions[0]
                    successful_tests.append(
                        {
                            "model": model_name,
                            "weight": weight_name,
                            "top_prediction": top_pred[0],
                            "confidence": top_pred[1] * 100,
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
                        f"{top_pred[0][:30]} ({top_pred[1] * 100:.1f}%){status_str}"
                    )

                    if compare and result.comparison:
                        comparison_results.append(
                            {
                                "model": model_name,
                                "top1_match": result.comparison["top1_match"],
                                "top5_match": result.comparison["top5_match"],
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
                            top_prediction=top_pred[0],
                            confidence=top_pred[1] * 100,
                            top1_match=result.comparison.get("top1_match")
                            if result.comparison
                            else None,
                            top5_match=result.comparison.get("top5_match")
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
                            "top1_match": None,
                            "top5_match": None,
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
                            "top1_match": None,
                            "top5_match": None,
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
            pred = test["top_prediction"][:35]
            logger.debug(f"  ✓ {test['model']:25s} -> {pred:35s} ({test['confidence']:5.1f}%)")

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
        "model": "resnet50",
        "image": "test_images/YellowLabradorLooking_new.jpg",
        "verbose": False,
        "use_tvm": False,
        "compare": False,
    }

    if config_path and Path(config_path).exists():
        import yaml  # Import here to satisfy type checker

        with open(config_path) as f:
            config = yaml.safe_load(f)
            defaults.update(config)

    return defaults


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchVision Classification Model Tester")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument(
        "--model", type=str, default="resnet50", help="Model name to test (default: resnet50)"
    )
    parser.add_argument(
        "--weight", type=str, default=None, help="Weight name to use (default: None, uses DEFAULT)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="test_images/YellowLabradorLooking_new.jpg",
        help="Path or URL to test image",
    )
    parser.add_argument(
        "--test-all", action="store_true", help="Test all available classification models"
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
            )
        else:
            test_multiple_models(
                image_url=args.image,
                max_models=args.max_models,
                model_filter=args.filter,
                use_tvm=args.tvm,
                compare=args.compare,
                log_file=args.log_file,
            )
    else:
        # Test single model
        result = main(
            model_name=args.model,
            weight_name=args.weight,
            image_url=args.image,
            use_tvm=args.tvm,
            compare=args.compare,
        )
