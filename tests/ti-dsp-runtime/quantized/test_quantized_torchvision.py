#!/usr/bin/env python
"""
Quantized TorchVision classification models — MMALIB DSP test.

Phase 1 of porting tests/cstatic/cl_torchvision.py to quantized/: get every
TorchVision ImageNet classification model compiling and running correctly
with -mmalib=1, via PT2E quantization (C7xMMAQuantizer). This is a
correctness sweep, not an offload-maximization pass -- some models will
have ops that fall back to the scalar (non-MMA) path, which is fine here.

Reuses infrastructure directly rather than duplicating it:
  - cl_torchvision.py's load_model_with_preprocessing/load_image/
    get_all_classification_models (dynamic model discovery + correct
    per-model preprocessing)
  - pt2e-tests/pt2e_utils.py's e2e_quantize_and_import/run_and_check
    (quantize -> import -> compile with MMALIB -> run -> assert a
    principled +/-1 LSB int8 tolerance), the same mechanism
    pt2e-tests/test_mobilenet_v2_pt2e_dsp.py already uses for mobilenet_v2.

4 models are excluded: their int8 weight size alone exceeds the 256 MiB
AM67A DLOAD DDR heap (DDR_C7X_1_LOCAL_HEAP). 7 more are excluded for a
different reason: runtime DDR pool exhaustion at inference time (not
weight size). See _EXCLUDED below and quantized/README.md.

Usage:
    pytest test_quantized_torchvision.py -v --dsp-mode=c7x_host --mmalib
    pytest test_quantized_torchvision.py -v --dsp-mode=c7x_host --mmalib -k resnet50
    pytest test_quantized_torchvision.py -v --dsp-mode=c7x_dload --mmalib
    python test_quantized_torchvision.py --model resnet50 --dsp-mode c7x_host --mmalib
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

_QUANTIZED_DIR = Path(__file__).parent
_CSTATIC_DIR = _QUANTIZED_DIR.parent.parent / "cstatic"
sys.path.insert(0, str(_CSTATIC_DIR))
from cl_torchvision import (  # noqa: E402
    get_all_classification_models,
    load_image,
    load_model_with_preprocessing,
)

_PT2E_TESTS_DIR = _QUANTIZED_DIR.parent / "pt2e-tests"
sys.path.insert(0, str(_PT2E_TESTS_DIR))
from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402
from pt2e_utils import e2e_quantize_and_import, run_and_check  # noqa: E402

pytestmark = [pytest.mark.c7x_only]

logger = logging.getLogger(__name__)

_TEST_IMAGE = _CSTATIC_DIR / "test_images" / "dog.jpg"

# Exceed the 256 MiB AM67A DLOAD DDR heap (DDR_C7X_1_LOCAL_HEAP) at pure
# int8 weight size alone, before any runtime/workspace overhead:
#   regnet_y_128gf 615MB, vit_h_14 603MB, vit_l_32 292MB, vit_l_16 290MB.
_EXCLUDED_WEIGHT_SIZE = {"regnet_y_128gf", "vit_h_14", "vit_l_32", "vit_l_16"}

# Not a weight-size problem -- these compile and run most of the way
# through, then exhaust the 352 MiB TVM DDR pool (workspace + weights +
# DLOAD segments, shared) at a late layer. Measured shortfall at the
# first failing allocation: convnext_large 2.8 KB, efficientnet_b6
# 2.74 MB, efficientnet_b7 43.6 KB, swin_b ~355 KB, swin_v2_b ~1.07 MB,
# swin_v2_s ~54.6 KB, swin_v2_t ~376.6 KB. Not limited to one
# architecture family -- the swin_* models are transformers, not CNNs.
# See quantized/README.md.
_EXCLUDED_DDR_OOM = {
    "convnext_large",
    "efficientnet_b6",
    "efficientnet_b7",
    "swin_b",
    "swin_v2_b",
    "swin_v2_s",
    "swin_v2_t",
}

_EXCLUDED = _EXCLUDED_WEIGHT_SIZE | _EXCLUDED_DDR_OOM

TORCHVISION_MODELS = sorted(
    name for name, _ in get_all_classification_models() if name not in _EXCLUDED
)


def _run_test(model_name: str, dsp_mode: str, mmalib: bool, record_cycles) -> None:
    model, preproc, _ = load_model_with_preprocessing(model_name)
    img = load_image(str(_TEST_IMAGE))
    example = (preproc(img).unsqueeze(0),)

    mod, input_np, ref_np = e2e_quantize_and_import(
        model, example, dtype="int8", n_calibration_batches=10
    )

    if mmalib:
        # run_and_check's default max_diff=2 (+/-1 LSB) is calibrated for
        # single-op unit tests (test_c7x_mma_quantizer_e2e_dsp.py) where
        # there's exactly one quantize/dequantize hop. Whole classification
        # models compound int8 rounding error across many sequential
        # layers -- mobilenet_v2's own end-to-end test bypasses this
        # default entirely and hand-checks max_diff<=20.0 for the same
        # reason. 25.0 matches the atol quantized/test_quantized_*.py
        # already uses for --mmalib on these same architectures.
        run_and_check(
            mod,
            input_np,
            ref_np,
            dsp_mode,
            record_cycles,
            cycles_key=f"{model_name}_pt2e_i8",
            max_diff=25,
        )
    else:
        # Generic (non-MMALIB) int8 codegen path: validates the
        # C7xMMAQuantizer + TVM import pipeline on its own. Top-1 argmax
        # match, not per-element -- mirrors mobilenet_v2's Part 1 rationale
        # in test_mobilenet_v2_pt2e_dsp.py.
        results = compile_and_run_dsp(
            mod=mod,
            input_data=input_np,
            target_string=get_target_string(dsp_mode, use_cpp_api=True),
            execution_mode=dsp_mode,
        )
        result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
        dsp_out = results[result_key].reshape(ref_np.shape).astype(np.float32)
        assert np.argmax(dsp_out) == np.argmax(ref_np), (
            f"{model_name}: no-MMALIB top-1 mismatch: "
            f"DSP={int(np.argmax(dsp_out))}, ref={int(np.argmax(ref_np))}"
        )


@pytest.mark.parametrize("model_name", TORCHVISION_MODELS, ids=TORCHVISION_MODELS)
def test_quantized_torchvision_dsp(model_name, dsp_mode, mmalib, record_cycles):
    """Test a quantized TorchVision classification model on DSP."""
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")
    _run_test(model_name, dsp_mode, mmalib, record_cycles)


def main():
    parser = argparse.ArgumentParser(description="Quantized TorchVision DSP Test")
    parser.add_argument("--model", required=True, choices=TORCHVISION_MODELS)
    parser.add_argument("--dsp-mode", required=True, choices=["c7x_host", "c7x_dload"])
    parser.add_argument("--mmalib", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

    def _record_cycles(name, cycles):
        if cycles:
            print(f"  cycles: {cycles:,} ({cycles / 1e6:.2f} ms @ 1 GHz)")

    print(f"Quantized {args.model} DSP Test (mode: {args.dsp_mode}, mmalib={args.mmalib})")
    _run_test(args.model, args.dsp_mode, args.mmalib, _record_cycles)
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
