#!/usr/bin/env python
"""
QuartzNet 15x5 (English) end-to-end DSP test.

Compiles QuartzNet's encoder and CTC head (see quartznet_c7x.py) for the C7x
DSP, float32 or quantized with C7xMMAQuantizer and run through MMALIB, and
compares its log-probabilities on one LibriSpeech test-clean utterance with
PyTorch's: the float model for float32, the Q/DQ graph (fake quant) for int8.

float32 is held to the PyTorch output closely, and must produce the same
transcript. int8 can't be: the DSP's integer kernels don't requantize
exactly as the Q/DQ graph's float scales do (MMALIB, for one, takes a
per-channel 7-bit multiplier and a shift), and through 94 MMALIB convs and
77 depthwise convs the differences change the most likely character on a
few percent of frames. Its gates are that frame-level agreement and the
character error rate between the DSP's and the reference's transcripts.

Usage:
    pytest test_quartznet_e2e.py --dsp-mode=c7x_host -v -s
    pytest test_quartznet_e2e.py --dsp-mode=c7x_dload --board beagley-ai -v -s
    python test_quartznet_e2e.py --dsp-mode c7x_host --dtype int8
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))
sys.path.insert(0, str(_THIS_DIR))

from dsp_utils import add_board_arg, compile_and_run_dsp, get_target_string  # noqa: E402
from quartznet_c7x import (  # noqa: E402
    DEFAULT_T,
    calibration_inputs,
    error_rate,
    eval_utterance,
    export_and_bind,
    greedy_decode,
    load_model,
    log_mel,
    quantize,
)

logger = logging.getLogger(__name__)

# Measured, identical on c7x_host and BeagleY-AI: float32 log-probs within 1.7e-4
# of PyTorch's; int8 matches the Q/DQ graph's most likely character on 95.1% of
# frames, with the same transcript. The frame gate allows twice the measured
# disagreement; the transcript gate, about 3 characters of this utterance's 150.
_FLOAT_ATOL = 0.01
_INT8_MIN_FRAME_AGREEMENT = 0.90
_INT8_MAX_CER_VS_REF = 0.02

# The float32 model takes about 140 s on the board.
_TIMEOUT_MS = 300000


def _run_quartznet_dsp_test(dsp_mode: str, dtype: str, T: int = DEFAULT_T) -> dict:
    """Compile QuartzNet, run one utterance on the DSP, compare with PyTorch."""
    model, frontend = load_model()
    utterance = eval_utterance(T)
    features = log_mel(utterance["audio"], frontend, T)
    x = torch.from_numpy(features)

    target = get_target_string(dsp_mode, use_cpp_api=True)
    if dtype == "float32":
        mod, reference = export_and_bind(model, T), model
    else:
        mod, reference = quantize(model, calibration_inputs(frontend, T), dtype)
        target += " -mmalib=1"
    with torch.no_grad():
        ref = reference(x).numpy()
        float_ref = model(x).numpy()

    dsp_results = compile_and_run_dsp(
        mod=mod,
        input_data=features,
        target_string=target,
        execution_mode=dsp_mode,
        timeout_ms=_TIMEOUT_MS,
    )
    if f"{dsp_mode}_error" in dsp_results:
        raise AssertionError(f"{dsp_mode} execution error: {dsp_results[f'{dsp_mode}_error']}")
    dsp = dsp_results[f"{dsp_mode}_result"].reshape(ref.shape)

    truth = utterance["transcript"]
    text = {"dsp": greedy_decode(dsp), "ref": greedy_decode(ref), "float": greedy_decode(float_ref)}
    return {
        "id": utterance["id"],
        "truth": truth,
        "text": text,
        "max_diff": float(np.abs(dsp - ref).max()),
        "frame_agreement": float((dsp.argmax(-1) == ref.argmax(-1)).mean()),
        "cer_vs_ref": error_rate(text["ref"], text["dsp"], words=False),
        "cer": {k: error_rate(truth, t, words=False) for k, t in text.items()},
        "wer": {k: error_rate(truth, t, words=True) for k, t in text.items()},
        "cycles": dsp_results.get(f"{dsp_mode}_cycles", 0),
    }


def _report(results: dict, dsp_mode: str, dtype: str) -> None:
    print(f"\nQuartzNet {dtype} on {dsp_mode}, LibriSpeech {results['id']}")
    print(f"  truth: {results['truth']}")
    for k, t in results["text"].items():
        print(f"  {k:5s}: {t}  (CER {results['cer'][k]:.3f}, WER {results['wer'][k]:.3f})")
    print(
        f"  DSP vs reference: max |log-prob diff| {results['max_diff']:.4g}, "
        f"frame argmax agreement {results['frame_agreement']:.4f}, "
        f"transcript CER {results['cer_vs_ref']:.4f}"
    )
    if results["cycles"]:
        print(f"  cycles: {results['cycles']:,} ({results['cycles'] / 1e6:.1f} ms @ 1 GHz)")


def _assert_passed(results: dict, dsp_mode: str, dtype: str) -> None:
    _report(results, dsp_mode, dtype)
    if dtype == "float32":
        assert results["max_diff"] <= _FLOAT_ATOL, (
            f"{dsp_mode} log-probs differ from PyTorch by {results['max_diff']:.4g} > {_FLOAT_ATOL}"
        )
        assert results["text"]["dsp"] == results["text"]["ref"], "transcripts differ"
    else:
        assert results["frame_agreement"] >= _INT8_MIN_FRAME_AGREEMENT, (
            f"{dsp_mode} matches the Q/DQ reference's most likely character on "
            f"{results['frame_agreement']:.4f} of frames < {_INT8_MIN_FRAME_AGREEMENT}"
        )
        assert results["cer_vs_ref"] <= _INT8_MAX_CER_VS_REF, (
            f"{dsp_mode} transcript CER against the Q/DQ reference's "
            f"{results['cer_vs_ref']:.4f} > {_INT8_MAX_CER_VS_REF}"
        )


@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_quartznet_c7x_host(dsp_mode, dtype):
    """QuartzNet on C7x host emulation (no hardware required)."""
    if dsp_mode not in (None, "c7x_host"):
        pytest.skip(f"test_quartznet_c7x_host requires --dsp-mode=c7x_host (got {dsp_mode})")
    _assert_passed(_run_quartznet_dsp_test("c7x_host", dtype), "c7x_host", dtype)


@pytest.mark.c7x_only
@pytest.mark.parametrize("dtype", ["float32", "int8"])
def test_quartznet_c7x_dload(dsp_mode, dtype, record_cycles):
    """QuartzNet deployed via DLOAD to real AM67A hardware."""
    if dsp_mode != "c7x_dload":
        pytest.skip("test_quartznet_c7x_dload requires --dsp-mode=c7x_dload")
    results = _run_quartznet_dsp_test("c7x_dload", dtype)
    _assert_passed(results, "c7x_dload", dtype)
    if results["cycles"]:
        record_cycles(f"quartznet_{dtype}_c7x_dload", results["cycles"])


def main():
    parser = argparse.ArgumentParser(description="QuartzNet DSP Test")
    parser.add_argument("--dsp-mode", required=True, choices=["c7x_host", "c7x_dload"])
    parser.add_argument("--dtype", choices=["float32", "int8", "int16"], default="float32")
    parser.add_argument("--frames", type=int, default=DEFAULT_T)
    add_board_arg(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )
    results = _run_quartznet_dsp_test(args.dsp_mode, args.dtype, T=args.frames)
    try:
        _assert_passed(results, args.dsp_mode, args.dtype)
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    print("\nTEST PASSED")


if __name__ == "__main__":
    main()
