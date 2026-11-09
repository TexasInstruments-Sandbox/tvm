"""Unit tests for quant_recipe.

Pure-Python tests — no DSP hardware or TI toolchain required.
"""

import pytest
import torch
import torch.nn as nn
from pt2e_utils import e2e_quantize_and_import
from quant_recipe import (
    BASELINE_CALIB_IMAGES,
    BASELINE_TEST_IMAGE,
    CALIB_IMAGES_DIR,
    TEST_INPUT,
    QuantRecipe,
    calibrate_and_convert,
)

from tvm.relax.frontend.torch import C7xMMAQuantizer


class _ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


def _input_scale(samples: list[tuple]) -> float:
    """The scale of the input's quantize node after calibrating on samples."""
    exported = torch.export.export(_ConvModel().eval(), samples[0]).module()
    quantized = calibrate_and_convert(exported, C7xMMAQuantizer("int8"), samples)
    for node in quantized.graph.nodes:
        if node.op == "call_function" and "quantize_per_tensor" in str(node.target):
            scale = node.args[1]
            assert isinstance(scale, float)
            return scale
    raise AssertionError("no quantize node in the converted graph")


@pytest.mark.quick
def test_baseline_calib_images_loads_each_image_once_in_order():
    loaded = []

    def load_image(path):
        loaded.append(path)
        return torch.zeros(1)

    samples = BASELINE_CALIB_IMAGES.calibration_samples((torch.ones(1),), load_image)

    # The set is part of the baseline's definition: changing it moves the
    # activation ranges of every model that uses this recipe.
    assert len(samples) == 12
    assert loaded == sorted(CALIB_IMAGES_DIR.glob("*.jpg"))


@pytest.mark.quick
def test_baseline_test_image_calibrates_on_the_test_input():
    test_input = (torch.ones(1),)
    samples = BASELINE_TEST_IMAGE.calibration_samples(test_input)
    assert len(samples) == 1
    assert samples[0] is test_input


@pytest.mark.quick
def test_e2e_rejects_dtype_with_recipe():
    # A recipe sets its own dtype, so an int16 test passing both would run on int8.
    with pytest.raises(ValueError, match="not both"):
        e2e_quantize_and_import(
            _ConvModel(),
            (torch.randn(1, 3, 8, 8),),
            dtype="int16",
            recipe=BASELINE_TEST_IMAGE,
        )


@pytest.mark.quick
def test_calib_images_recipe_requires_load_image():
    with pytest.raises(ValueError, match="load_image"):
        BASELINE_CALIB_IMAGES.calibration_samples((torch.ones(1),))


@pytest.mark.quick
def test_unknown_calibration_source_is_rejected():
    with pytest.raises(ValueError, match="unknown calibration"):
        QuantRecipe("bad", "no-such-source").calibration_samples((torch.ones(1),))


@pytest.mark.quick
@pytest.mark.parametrize(("dtype", "symmetric"), [("int8", True), ("int8", False), ("int16", True)])
def test_recipe_configures_the_quantizer(dtype, symmetric):
    recipe = QuantRecipe("x", TEST_INPUT, dtype=dtype, symmetric_activations=symmetric)
    quantizer = recipe.quantizer()
    assert quantizer.dtype == dtype
    assert quantizer.symmetric_activations == symmetric


@pytest.mark.quick
def test_calibration_observes_every_sample():
    small = (torch.full((1, 3, 8, 8), 0.5),)
    large = (torch.full((1, 3, 8, 8), 2.0),)

    # The wide range wins whichever position it's in, so neither the first nor
    # the last sample alone sets the scale.
    assert _input_scale([small, large]) == _input_scale([large])
    assert _input_scale([large, small]) == _input_scale([large])
    assert _input_scale([small]) != _input_scale([large])
