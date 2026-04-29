#!/usr/bin/env python
"""
Test suite for FCN ResNet-50 semantic segmentation model on TVM targets.

This module tests the Fully Convolutional Network (FCN) with ResNet-50 backbone
for semantic segmentation tasks. The tests compare execution between LLVM and CStatic targets.

FCN Model Architecture:
- Uses ResNet-50 as the backbone for feature extraction
- Applies dilated convolutions to maintain spatial resolution
- Includes upsampling layers to restore input resolution
- Trained on PASCAL VOC dataset with 21 classes (20 object classes + background)

PyTorch vs TVM Output Format:
- PyTorch FCN returns: OrderedDict with keys 'out' (main) and 'aux' (auxiliary)
- TVM Relax converts this to: Tuple(main_output, aux_output)
- Both outputs have shape (batch_size, num_classes, height, width)
- Output resolution matches input resolution due to upsampling

Output Content Details:
- 'out' (main): Final segmentation logits from the FCN head classifier
  * Used for inference and final predictions
  * Contains per-pixel class scores for all 21 PASCAL VOC classes
  * Values are raw logits (pre-softmax) ranging typically from -10 to +10
- 'aux' (auxiliary): Intermediate segmentation logits from ResNet layer3
  * Used during training for auxiliary loss to improve gradient flow
  * Contains the same 21-class predictions but from an earlier network layer
  * Helps train deeper layers by providing additional supervision signal
  * Often ignored during inference, but included for model compatibility

Key Features Tested:
- Dynamic input shapes (224x224, 320x240, 480x640)
- Cross-target consistency (LLVM vs CStatic)
- Proper output shape validation
- Numerical accuracy comparison
- Segmentation visualization with PASCAL VOC color mapping

Expected Warnings:
- "SystemLib symbol add1 get overriden": This is a benign TVM warning that occurs when
  multiple models are compiled in the same Python session. TVM generates internal
  function names like 'add1', 'conv2d', etc. that can conflict across different model
  compilations. TVM automatically handles these conflicts by overriding the symbol
  addresses. This does not affect functionality or accuracy - it's just TVM's way of
  managing its internal function registry. Common when testing multiple input sizes
  or running multiple segmentation tests sequentially.
"""

import os
from typing import Any, Dict, Tuple, Union

import numpy as np
import pytest
import torch
import torchvision.transforms as T
import tvm
from PIL import Image, ImageDraw, ImageFont
from torch.export import export
from torchvision.models.segmentation import FCN_ResNet50_Weights, fcn_resnet50
from tvm.relax.frontend.torch import from_exported_program

from tvm_utils import compile_and_run_on_target, process_relax  # type: ignore

# PASCAL VOC 2012 class names and colors for visualization
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

# Color palette for PASCAL VOC classes (RGB values)
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


def load_test_image(size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and preprocess the test image for segmentation.

    Uses bird_0.jpg which contains a bird (PASCAL VOC class 'bird').
    The image will be resized to the specified dimensions and normalized
    using ImageNet preprocessing standards.
    """
    # Get the path to the bird test image
    test_image_path = os.path.join(os.path.dirname(__file__), "test_images", "bird_0.jpg")

    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test image not found: {test_image_path}")

    # Load image and resize to specified size
    image = Image.open(test_image_path).convert("RGB")

    # Convert to tensor and normalize (standard torchvision preprocessing)
    # torchvision.transforms.Resize expects (height, width) which matches our size parameter
    transform = T.Compose(
        [
            T.Resize(size),  # size is already (height, width) as expected by torchvision
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_tensor = transform(image)  # type: ignore

    # Add batch dimension and return as numpy array
    return image_tensor.unsqueeze(0).numpy()  # type: ignore


def torch_to_relax(
    torch_model: torch.nn.Module, example_input: Tuple[torch.Tensor, ...]
) -> tvm.IRModule:
    """
    Convert a PyTorch model to a Relax IRModule.

    This function performs the PyTorch to TVM conversion in two steps:
    1. torch.export: Converts PyTorch model to FX graph representation
    2. from_exported_program: Converts FX graph to TVM Relax IR

    Important: torch.export flattens dictionary outputs into separate tensors,
    so FCN's {'out': tensor1, 'aux': tensor2} becomes tuple(tensor1, tensor2)
    """
    with torch.no_grad():
        # Set model to eval mode to disable dropout, batch norm updates, etc.
        torch_model.eval()
        # Export PyTorch model to FX graph format - this flattens dict outputs
        exported_program = export(torch_model, example_input)  # type: ignore
        # Convert FX graph to TVM Relax IR with parameters as inputs
        mod = from_exported_program(exported_program, keep_params_as_input=True)  # type: ignore
    return mod


def create_segmentation_model(input_size: Tuple[int, int] = (224, 224)) -> tvm.IRModule:
    """
    Create and prepare FCN segmentation model for testing.

    FCN Architecture Details:
    - Backbone: ResNet-50 pre-trained on ImageNet
    - Head: Fully convolutional layers for dense prediction
    - Classes: 21 (PASCAL VOC: 20 objects + background)
    - Output: Same resolution as input due to upsampling layers

    Model Outputs (PyTorch):
    - 'out': Main segmentation output for inference
    - 'aux': Auxiliary output from intermediate layer (used during training)

    After TVM conversion, these become tuple elements: (main_output, aux_output)
    """
    # Initialize FCN ResNet-50 with ImageNet pre-trained backbone + PASCAL VOC segmentation head
    torch_model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT).eval()  # type: ignore

    # Create example input for torch.export
    # Shape: (batch_size=1, channels=3, height, width)
    example_input = (torch.randn(1, 3, input_size[0], input_size[1], dtype=torch.float32),)

    # Convert PyTorch model to TVM Relax IR and apply optimization passes
    mod = torch_to_relax(torch_model, example_input)
    mod = process_relax(mod)  # type: ignore  # Apply TVM optimization passes
    return mod


def extract_segmentation_output(result: Union[Tuple[Any, ...], Dict[str, Any], Any]) -> np.ndarray:
    """
    Extract and format segmentation output from FCN result.

    Args:
        result: Tuple containing (main_output, aux_output) from TVM Relax conversion,
                or dictionary containing 'out' key with segmentation map, or direct tensor

    Returns:
        Main segmentation map as numpy array
    """
    if isinstance(result, tuple) and len(result) >= 1:
        # TVM Relax converts FCN dict outputs to tuple: (main_output, aux_output)
        # We want the main output (first element)
        seg_map = result[0]
    elif isinstance(result, dict) and "out" in result:
        # Original PyTorch FCN returns dict with 'out' key containing the main segmentation output
        seg_map = result["out"]
    else:
        # Direct tensor output
        seg_map = result

    if hasattr(seg_map, "numpy"):
        return seg_map.numpy()  # type: ignore
    else:
        return seg_map  # type: ignore


def compare_segmentation_results(
    llvm_result: Union[Tuple[Any, ...], Dict[str, Any], Any],
    c_static_result: Union[Tuple[Any, ...], Dict[str, Any], Any],
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Compare segmentation results between two targets.

    For segmentation models, we compare the main output tensors which contain
    per-pixel class logits. The comparison uses numpy.allclose with tolerances
    appropriate for floating-point numerical differences between targets.

    Args:
        llvm_result: Output from LLVM target execution
        c_static_result: Output from C_Static target execution
        rtol: Relative tolerance for numpy.allclose
        atol: Absolute tolerance for numpy.allclose

    Returns:
        bool: True if results match within tolerance, False otherwise
    """
    # Extract main segmentation outputs (first element of tuple for TVM results)
    llvm_seg = extract_segmentation_output(llvm_result)
    c_static_seg = extract_segmentation_output(c_static_result)

    # Verify shapes match before comparing values
    if llvm_seg.shape != c_static_seg.shape:
        print(f"Shape mismatch: LLVM {llvm_seg.shape} vs C_Static {c_static_seg.shape}")
        return False

    # Compare segmentation logits elementwise with tolerance for numerical precision
    try:
        return np.allclose(llvm_seg, c_static_seg, rtol=rtol, atol=atol)
    except Exception as e:
        print(f"Comparison failed: {e}")
        return False


def load_original_test_image(size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Load the original test image without preprocessing for visualization.

    Args:
        size: Target size for resizing

    Returns:
        PIL Image in RGB format
    """
    # Use the same bird image as load_test_image
    test_image_path = os.path.join(os.path.dirname(__file__), "test_images", "bird_0.jpg")

    if not os.path.exists(test_image_path):
        raise FileNotFoundError(f"Test image not found: {test_image_path}")

    # Load and resize image without normalization
    image = Image.open(test_image_path).convert("RGB")
    # PIL resize expects (width, height) but size is (height, width)
    # Convert (height, width) to (width, height) for PIL
    pil_size = (size[1], size[0])  # (width, height)
    image = image.resize(pil_size, Image.Resampling.LANCZOS)  # type: ignore
    return image


def logits_to_segmentation_mask(logits: np.ndarray) -> np.ndarray:
    """
    Convert segmentation logits to class predictions.

    Args:
        logits: Segmentation logits with shape (batch_size, num_classes, height, width)

    Returns:
        Segmentation mask with shape (height, width) containing class indices
    """
    # Remove batch dimension and get argmax across class dimension
    if logits.ndim == 4:
        logits = logits[0]  # Remove batch dimension: (num_classes, height, width)

    # Get class with highest probability for each pixel
    seg_mask = np.argmax(logits, axis=0)  # Shape: (height, width)
    return seg_mask


def create_segmentation_colormap(seg_mask: np.ndarray) -> np.ndarray:
    """
    Convert segmentation mask to RGB colormap using PASCAL VOC colors.

    Args:
        seg_mask: Segmentation mask with shape (height, width) containing class indices

    Returns:
        RGB colormap with shape (height, width, 3)
    """
    height, width = seg_mask.shape
    colormap = np.zeros((height, width, 3), dtype=np.uint8)

    # Map each class index to its corresponding color
    for class_idx in range(len(PASCAL_VOC_COLORS)):
        mask = seg_mask == class_idx
        colormap[mask] = PASCAL_VOC_COLORS[class_idx]

    return colormap


def overlay_segmentation_on_image(
    original_image: Image.Image, seg_mask: np.ndarray, alpha: float = 0.6
) -> Image.Image:
    """
    Overlay segmentation mask on original image with transparency.

    Args:
        original_image: Original PIL image
        seg_mask: Segmentation mask with shape (height, width)
        alpha: Transparency factor (0.0 = fully transparent, 1.0 = fully opaque)

    Returns:
        PIL Image with segmentation overlay
    """
    # Create colormap from segmentation mask
    colormap = create_segmentation_colormap(seg_mask)

    # Convert original image to numpy array
    orig_array = np.array(original_image)

    # Blend original image with colormap
    # Don't overlay background (class 0)
    background_mask = seg_mask == 0

    # Start with original image
    blended = orig_array.copy()

    # Overlay segmentation colors where not background
    non_bg_mask = ~background_mask
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
    """
    Save segmentation visualization as JPG with legend.

    Args:
        original_image: Original PIL image
        seg_mask: Segmentation mask with shape (height, width)
        output_path: Path to save the visualization
        title: Title for the visualization
    """
    # Create overlay image
    overlay_image = overlay_segmentation_on_image(original_image, seg_mask)

    # Get unique classes present in the segmentation
    unique_classes = np.unique(seg_mask)

    # Create a combined visualization with legend
    img_width, img_height = overlay_image.size
    legend_width = 200
    total_width = img_width + legend_width

    # Create new image for combined visualization
    combined = Image.new("RGB", (total_width, img_height), color="white")

    # Paste the overlay image
    combined.paste(overlay_image, (0, 0))

    # Draw legend
    draw = ImageDraw.Draw(combined)

    # Try to load a font, fallback to default if not available
    font = ImageFont.load_default()  # type: ignore

    # Draw title
    draw.text((img_width + 10, 10), title, fill="black", font=font)

    # Draw legend for present classes
    y_offset = 40
    for class_idx in sorted(unique_classes):
        if class_idx < len(PASCAL_VOC_CLASSES):
            color = PASCAL_VOC_COLORS[class_idx]
            class_name = PASCAL_VOC_CLASSES[class_idx]

            # Draw color square
            draw.rectangle(
                [img_width + 10, y_offset, img_width + 25, y_offset + 15],
                fill=color,
                outline="black",
            )

            # Draw class name
            draw.text((img_width + 30, y_offset), class_name, fill="black", font=font)

            y_offset += 20

    # Save as JPG
    combined.save(output_path, "JPEG", quality=95)
    print(f"Segmentation visualization saved to: {output_path}")


def generate_segmentation_visualization(
    result: Union[Tuple[Any, ...], Dict[str, Any], Any],
    input_size: Tuple[int, int],
    target_name: str,
    test_name: str = "segmentation",
) -> None:
    """
    Generate and save segmentation visualization for a target result.

    Args:
        result: Segmentation result from model execution
        input_size: Input image dimensions (height, width)
        target_name: Name of the target (e.g., "llvm", "c_static")
        test_name: Name of the test for file naming
    """
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(__file__), "segmentation_outputs")
        os.makedirs(output_dir, exist_ok=True)

        # Load original image and generate visualization
        original_image = load_original_test_image(size=input_size)
        seg_output = extract_segmentation_output(result)
        seg_mask = logits_to_segmentation_mask(seg_output)

        # Create filename based on test, target, and size
        output_filename = f"{test_name}_{target_name}_{input_size[0]}x{input_size[1]}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Generate title for visualization
        title = f"FCN Segmentation - {target_name.upper()} ({input_size[0]}x{input_size[1]})"

        save_segmentation_visualization(original_image, seg_mask, output_path, title)

    except Exception as e:
        print(f"Warning: Could not generate {target_name} visualization: {e}")


# Test multiple input sizes to verify dynamic shape handling
@pytest.mark.slow
@pytest.mark.parametrize(
    "input_size,target_c_static",
    [
        ((224, 224), "c_static"),
        ((320, 240), "c_static"),
    ],
)
def test_segmentation_dynamic_shapes(input_size: Tuple[int, int], target_c_static: str) -> None:
    """
    Test FCN segmentation model with dynamic shapes comparing LLVM vs CStatic targets.

    This test verifies that the FCN model produces consistent results across different
    input resolutions and compilation targets. FCN is fully convolutional, so it can
    handle arbitrary input sizes and will output segmentation maps at the same resolution.

    Test Process:
    1. Create FCN model compiled for specific input size
    2. Load and preprocess test image to target size
    3. Execute on both LLVM (reference) and CStatic (target) backends
    4. Compare numerical outputs for consistency
    """
    print(f"Testing with input size: {input_size}")

    # Create FCN model optimized for this specific input resolution
    mod = create_segmentation_model(input_size=input_size)
    input_data = load_test_image(size=input_size)

    # Execute model on LLVM backend (reference implementation)
    llvm_result = compile_and_run_on_target(  # type: ignore
        target_string="llvm", mod=mod, input=input_data
    )

    # Execute model on CStatic backend (target implementation)
    c_static_result = compile_and_run_on_target(  # type: ignore
        target_string=target_c_static, mod=mod, input=input_data
    )

    # Compare segmentation outputs between targets
    # Only the first output needs to be compared. The second output is an aux output.
    results_match = compare_segmentation_results(llvm_result[0], c_static_result[0])

    # Generate visualizations for both targets
    generate_segmentation_visualization(llvm_result[0], input_size, "llvm", "dynamic_shapes")
    generate_segmentation_visualization(
        c_static_result[0], input_size, "c_static", "dynamic_shapes"
    )

    if not results_match:
        # Extract segmentation maps for detailed error reporting
        llvm_seg = extract_segmentation_output(llvm_result[0])
        c_static_seg = extract_segmentation_output(c_static_result[0])
        max_diff = (
            np.max(np.abs(llvm_seg - c_static_seg))
            if llvm_seg.shape == c_static_seg.shape
            else float("inf")
        )

        raise AssertionError(
            f"Results differ for {target_c_static} with input size {input_size}. "
            f"LLVM shape: {llvm_seg.shape}, C_Static shape: {c_static_seg.shape}, "
            f"Max difference: {max_diff}"
        )

    print(f"✓ Dynamic shape test passed for input size {input_size}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
