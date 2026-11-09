"""Named PT2E quantization recipes, shared by quantized/ and pt2e-tests/.

A recipe says how a model gets quantized: the quantizer settings and what it
calibrates on. Naming it lets an accuracy result state what produced it.

Recipes calibrate only on images in this repository. Datasets used to measure
accuracy stay out of them.
"""

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.fx import GraphModule
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

from tvm.relax.frontend.torch import C7xMMAQuantizer

logger = logging.getLogger(__name__)

CALIB_IMAGES_DIR = Path(__file__).parent.parent / "quantized" / "calib_images"

# Where a recipe's calibration samples come from.
CALIB_IMAGES = "calib_images"  # every image in CALIB_IMAGES_DIR, once
TEST_INPUT = "test_input"  # the model's test input, once


@dataclass(frozen=True)
class QuantRecipe:
    """How to quantize a model.

    Attributes:
        name: identifies the recipe in logs and reports.
        calibration: CALIB_IMAGES or TEST_INPUT.
        dtype: "int8" or "int16".
        symmetric_activations: passed to C7xMMAQuantizer.
    """

    name: str
    calibration: str
    dtype: str = "int8"
    symmetric_activations: bool = True

    def quantizer(self) -> C7xMMAQuantizer:
        return C7xMMAQuantizer(dtype=self.dtype, symmetric_activations=self.symmetric_activations)

    def calibration_samples(
        self, test_input: tuple, load_image: Callable[[Path], torch.Tensor] | None = None
    ) -> list[tuple]:
        """Return the samples to calibrate on, each a tuple of model arguments.

        Args:
            test_input: the model's example arguments.
            load_image: reads an image path into a model input with the
                model's own preprocessing. Required for CALIB_IMAGES.
        """
        if self.calibration == TEST_INPUT:
            samples = [test_input]
        elif self.calibration == CALIB_IMAGES:
            if load_image is None:
                raise ValueError(f"recipe {self.name} calibrates on images: pass load_image")
            paths = sorted(CALIB_IMAGES_DIR.glob("*.jpg"))
            if not paths:
                raise FileNotFoundError(f"no calibration images in {CALIB_IMAGES_DIR}")
            samples = [(load_image(path),) for path in paths]
        else:
            raise ValueError(f"recipe {self.name}: unknown calibration {self.calibration!r}")
        logger.info("recipe %s: %d calibration samples", self.name, len(samples))
        return samples


# The per-model tests in quantized/ (test_quantized_<model>.py, YOLO).
BASELINE_CALIB_IMAGES = QuantRecipe("baseline-calib-images", CALIB_IMAGES)

# The torchvision sweep and the pt2e MobileNetV2 test: one sample, and the test
# input is the calibration input.
BASELINE_TEST_IMAGE = QuantRecipe("baseline-test-image", TEST_INPUT)


def export_module(model: nn.Module, example_inputs: tuple, strict: bool = True) -> GraphModule:
    """Export a float model (eval mode set here) to the GraphModule prepare_pt2e expects.

    strict: passed through to torch.export.export. Models with dynamic control
    flow (e.g. raw ultralytics/yolov5 modules) require False.
    """
    model.eval()
    with torch.no_grad():
        # export() returns an ExportedProgram; .module() unwraps it to a GraphModule.
        return torch.export.export(model, example_inputs, strict=strict).module()


def calibrate_and_convert(
    model_gm: GraphModule, quantizer: C7xMMAQuantizer, samples: list[tuple]
) -> GraphModule:
    """Prepare → calibrate → convert an exported model. Returns the Q/DQ GraphModule."""
    # prepare_pt2e calls quantizer.annotate() to read the quantization specs, then
    # inserts MinMaxObserver / PerChannelMinMaxObserver modules at each annotated
    # tensor edge.  Running inputs through the prepared model fills in the observed
    # min/max ranges that will become the Q/DQ scale factors.
    prepared = prepare_pt2e(model_gm, quantizer)
    with torch.no_grad():
        for sample in samples:
            prepared(*sample)

    with warnings.catch_warnings():
        # convert_pt2e removes observer nodes from the graph; torch emits a
        # harmless UserWarning("erase_node") for each one.
        warnings.filterwarnings("ignore", message="erase_node")
        # convert_pt2e replaces observer modules with explicit quantize/dequantize
        # (Q/DQ) nodes, producing a graph of the form q → dq → op → q → dq → ...
        return convert_pt2e(prepared)


def quantize_with_recipe(
    model: nn.Module,
    example_inputs: tuple,
    recipe: QuantRecipe,
    load_image: Callable[[Path], torch.Tensor] | None = None,
    strict: bool = True,
) -> GraphModule:
    """Export, calibrate and convert a float model. Returns the Q/DQ GraphModule.

    The tests that compile for the DSP and the accuracy harness both quantize
    through this, so accuracy is measured on the graph the DSP tests run.
    """
    return calibrate_and_convert(
        export_module(model, example_inputs, strict),
        recipe.quantizer(),
        recipe.calibration_samples(example_inputs, load_image),
    )
