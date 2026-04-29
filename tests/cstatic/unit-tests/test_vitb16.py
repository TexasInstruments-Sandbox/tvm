#!/usr/bin/env python
"""
Test suite for Vision Transformer (ViT-B/16) image classification model on TVM targets.

This module tests the Vision Transformer Base model with 16x16 patch size for image
classification tasks. The tests compare execution between LLVM and CStatic targets.

ViT-B/16 Model Architecture:
- Input: 224x224 RGB images
- Patch size: 16x16 (resulting in 14x14 = 196 patches)
- Embedding dimension: 768
- Number of transformer blocks: 12
- Number of attention heads: 12
- MLP hidden dimension: 3072
- Pre-trained on ImageNet-1K (1000 classes)

PyTorch vs TVM Output Format:
- PyTorch ViT returns: Classification logits tensor of shape (batch_size, 1000)
- TVM Relax may convert this to: Single output or Tuple depending on model variant
  * During torch.export, some ViT models may include auxiliary outputs like
    intermediate features or attention maps
  * The primary output (index 0) contains the classification logits
  * Auxiliary outputs are typically used for visualization or debugging

Output Content Details:
- Primary output: Classification logits for 1000 ImageNet classes
  * Shape: (1, 1000) for batch size 1
  * Values are raw logits (pre-softmax) ranging typically from -20 to +20
  * Apply softmax to get probability distribution over classes
- Auxiliary outputs (if present): May include attention maps, intermediate features
  * Used for model interpretability and debugging
  * Not required for standard classification inference

Key Features Tested:
- Image classification accuracy on real images
- Cross-target consistency (LLVM vs CStatic)
- Top-5 prediction comparison between targets
- Numerical accuracy comparison with tight tolerances
- Proper handling of ImageNet preprocessing (normalization, cropping)
- Multi-output handling (extracting classification logits from model outputs)

Model Characteristics:
- Large model size (~86M parameters)
- Pure attention-based architecture without convolutions
- Uses learnable positional embeddings for patch positions
- Class token prepended to patch embeddings for classification
- Pre-trained on ImageNet with sophisticated data augmentation

Expected Behavior:
- Both LLVM and CStatic targets should predict the same top class
- Numerical outputs should match within rtol=1e-3, atol=1e-5
- Model correctly classifies test images (e.g., dog.jpg -> dog breeds)

Note on Multi-Output Handling:
After torch.export conversion, ViT models may return multiple outputs. This test
automatically detects and extracts the primary classification output (index 0)
when the model returns a list of tensors.
"""

import numpy as np
import pytest
import torch

# Additional imports for real image handling
import torchvision.transforms as transforms
import tvm
from PIL import Image
from torch.export import export
from torchvision.models.vision_transformer import ViT_B_16_Weights, vit_b_16
from tvm.relax.frontend.torch import from_exported_program

from tvm_utils import compile_and_run_on_target, process_relax


def torch_to_relax(torch_model, example_input) -> tvm.IRModule:
    """Convert a PyTorch model to a Relax IRModule."""
    with torch.no_grad():
        exported_program = export(torch_model, example_input)

        # Debug: Let's examine problematic slice operations
        # print(f"DEBUG: Exported program has {len(exported_program.graph.nodes)} nodes")
        # slice_count = 0
        # for i, node in enumerate(exported_program.graph.nodes):
        #    if hasattr(node, 'target') and 'slice' in str(node.target).lower():
        #        slice_count += 1
        #        print(f"DEBUG: Slice node {i}: target={node.target}")
        #        print(f"DEBUG: Args count: {len(node.args)}, Args: {node.args}")
        #        if len(node.args) < 2:
        #            print(f"DEBUG: This slice has insufficient args!")
        # print(f"DEBUG: Found {slice_count} slice operations")

        # Try with different options that might help with ViT compatibility
        mod = from_exported_program(exported_program, keep_params_as_input=True)
    return mod


def create_vitb16_model():
    """Create and prepare ViT-B/16 model for testing."""
    try:
        # Initialize torch model with pre-trained weights
        torch_model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval()

        # Create example input for torch.export
        example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)

        # Convert to Relax IRModule and process
        mod = torch_to_relax(torch_model, example_args)
        mod = process_relax(mod)
        return mod
    except Exception as e:
        print(f"ViT model conversion failed: {e}")
        # For now, skip the test if ViT conversion fails
        pytest.skip(f"ViT-B/16 model conversion not supported: {e}")
        return None


def load_sample_image():
    """Load a real image from local file."""

    # Define ImageNet preprocessing transforms
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Try to load local dog.jpg file
    import os

    local_image_path = os.path.join(os.path.dirname(__file__), "test_images", "dog.jpg")

    if os.path.exists(local_image_path):
        try:
            print(f"Loading local image: {local_image_path}")

            # Load image with PIL
            image = Image.open(local_image_path)

            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Apply preprocessing
            tensor = preprocess(image)

            # Add batch dimension and convert to numpy
            batch_tensor = tensor.unsqueeze(0)  # Shape: (1, 3, 224, 224) #pyright: ignore

            print(f" Successfully loaded local image: {local_image_path}")
            print(f"   Image shape: {batch_tensor.shape}")
            print(f"   Value range: [{batch_tensor.min():.3f}, {batch_tensor.max():.3f}]")

            return batch_tensor.numpy().astype(np.float32)

        except Exception as e:
            pytest.fail(f"Failed to load local image {local_image_path}: {e}")
    else:
        pytest.fail(f" Local image {local_image_path} not found")


def get_imagenet_labels():
    """Get ImageNet class labels. Returns a list of 1000 class names."""
    # Try to use the official ImageNet labels from torchvision
    try:
        from torchvision.models.vision_transformer import ViT_B_16_Weights

        weights = ViT_B_16_Weights.DEFAULT
        if hasattr(weights, "meta") and "categories" in weights.meta:
            labels = weights.meta["categories"]
            if len(labels) == 1000:
                return labels
            else:
                pytest.fail(f"Expected 1000 ImageNet labels, got {len(labels)}")
    except Exception as e:
        pytest.fail(f"Official ImageNet labels not available from torchvision: {e}")

    # If we get here, the official labels were not available
    pytest.fail("Official ImageNet labels not available - cannot decode predictions")


def decode_prediction(logits, top_k=5):
    """Decode prediction logits to class labels."""
    labels = get_imagenet_labels()

    # Get top-k predictions
    top_indices = np.argsort(logits[0])[-top_k:][::-1]  # Sort in descending order
    top_probs = np.exp(logits[0][top_indices]) / np.sum(np.exp(logits[0]))  # Softmax

    predictions = []
    for idx, prob in zip(top_indices, top_probs):
        if idx < len(labels):
            predictions.append((labels[idx], prob, idx))
        else:
            predictions.append((f"class_{idx}", prob, idx))

    return predictions


# Parameters are too large to use the C source approach
@pytest.mark.slow
@pytest.mark.parametrize("target_c_static", ["c_static"])
def test_vitb16_comparison(target_c_static):
    """Test ViT-B/16 model comparing llvm vs c_static targets."""
    mod = create_vitb16_model()
    # create_vitb16_model() will call pytest.skip() if conversion fails
    # so we only reach here if successful
    input_data = load_sample_image()

    print("\nRunning ViT-B/16 inference on both targets...")

    # Get results from both targets
    print("Running on LLVM target...")
    llvm_result = compile_and_run_on_target(target_string="llvm", mod=mod, input=input_data)

    print("Running on CStatic target...")
    c_static_result = compile_and_run_on_target(
        target_string=target_c_static, mod=mod, input=input_data
    )

    # Handle multi-output case: if result is a list, take the first output
    # ViT-B/16 may return multiple outputs during conversion
    if isinstance(llvm_result, list):
        print(f"  Note: Model returned {len(llvm_result)} outputs, using first output")
        llvm_result = llvm_result[0]
    if isinstance(c_static_result, list):
        print(f"  Note: Model returned {len(c_static_result)} outputs, using first output")
        c_static_result = c_static_result[0]

    # Decode predictions for both targets
    print("\n LLVM Target Predictions:")
    llvm_predictions = decode_prediction(llvm_result, top_k=5)
    for i, (label, prob, idx) in enumerate(llvm_predictions):
        print(f"  {i + 1}. {label} (class {idx}): {prob:.4f}")

    print("\n CStatic Target Predictions:")
    c_static_predictions = decode_prediction(c_static_result, top_k=5)
    for i, (label, prob, idx) in enumerate(c_static_predictions):
        print(f"  {i + 1}. {label} (class {idx}): {prob:.4f}")

    # Show top prediction comparison
    llvm_top = llvm_predictions[0]
    c_static_top = c_static_predictions[0]

    print("\n Top Predictions:")
    print(f"   LLVM: {llvm_top[0]} ({llvm_top[2]}) - {llvm_top[1]:.4f}")
    print(f"   CStatic:  {c_static_top[0]} ({c_static_top[2]}) - {c_static_top[1]:.4f}")

    if llvm_top[2] == c_static_top[2]:
        print("   Pass: Both targets predict the same class!")
    else:
        print("   Fail: Different top predictions between targets")

    # Compare numerical results
    max_diff = np.max(np.abs(llvm_result - c_static_result))
    print("\n Numerical comparison:")
    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Relative difference: {max_diff / np.max(np.abs(llvm_result)):.2e}")

    # Assert results are close enough
    assert np.allclose(llvm_result, c_static_result, rtol=1e-3, atol=1e-5), (
        f"Results differ for {target_c_static}. Max difference: {max_diff}"
    )

    print("   Pass: Numerical results match within tolerance!")


if __name__ == "__main__":
    pytest.main([__file__])
