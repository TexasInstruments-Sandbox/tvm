#!/usr/bin/env python
"""
GTCRN end-to-end DSP test.

Compiles GTCRN (a tiny speech-enhancement model, see gtcrn_c7x.py) for the
C7x DSP and compares its output against a PyTorch (host CPU) reference on a
single fixed-shape [1, 257, T, 2] STFT chunk.

Correctness is gated on audio-domain SNR (ISTFT both outputs, compare
waveforms), not raw spectrogram-domain max-diff. Measured directly: raw
complex-spectrogram diff between C7x and PyTorch swings widely across
different inputs (0.27 to 0.76 max abs, seed-dependent, presumably from the
63-step internal GRU recurrence compounding small per-step precision
differences between the C7x vector ISA's transcendental (sigmoid/tanh)
implementations and PyTorch's CPU reference) — raw diff is not a stable
correctness signal for this model. Audio-domain SNR from the same runs was
consistently 39.8-48.0dB (near-imperceptible), so that's the actual
invariant worth testing. The _MIN_SNR_DB threshold below (20dB) is a
generous margin under that observed range: a real bug (as opposed to benign
precision noise) is expected to produce a qualitatively worse, much lower
SNR, not a borderline one.

Usage:
    pytest test_gtcrn_e2e.py --dsp-mode=c7x_host -v
    pytest test_gtcrn_e2e.py --dsp-mode=c7x_dload -v
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
from gtcrn_c7x import (  # noqa: E402
    DEFAULT_T,
    FS,
    chunk_samples,
    export_and_bind,
    istft_chunk,
    load_model,
    stft_chunk,
)

logger = logging.getLogger(__name__)

_MIN_SNR_DB = 20.0  # see module docstring


def _synthesize_chunk(T: int, seed: int) -> np.ndarray:
    """A fixed-seed sine-tone + white-noise waveform, chunk_samples(T) long.

    Used instead of raw torch.randn as the model input: random values fed
    directly as if they were a spectrogram lack any of the magnitude-decay
    or temporal/spectral coherence a real STFT has, which turned out to make
    the raw-diff instability *worse*, not just irrelevant. Passing a
    synthesized *waveform* through the real STFT keeps the input a valid
    (if not realistic) spectrogram.
    """
    rng = np.random.default_rng(seed)
    n = chunk_samples(T)
    t = np.arange(n) / FS
    tone = 0.3 * np.sin(2 * np.pi * 220.0 * t)
    noise = 0.05 * rng.standard_normal(n)
    return (tone + noise).astype(np.float32)


def _run_gtcrn_dsp_test(dsp_mode: str, T: int = DEFAULT_T, seed: int = 0) -> dict:
    """Compile GTCRN, run one chunk on the DSP, and compare against PyTorch."""
    model = load_model()

    inp = stft_chunk(_synthesize_chunk(T, seed))
    x = torch.from_numpy(inp)
    with torch.no_grad():
        torch_result = model(x).numpy()

    mod = export_and_bind(model, T=T)
    target_string = get_target_string(dsp_mode)

    dsp_results = compile_and_run_dsp(
        mod=mod,
        input_data=inp,
        target_string=target_string,
        execution_mode=dsp_mode,
    )

    result_key = f"{dsp_mode}_result"
    error_key = f"{dsp_mode}_error"
    if error_key in dsp_results:
        raise AssertionError(f"{dsp_mode} execution error: {dsp_results[error_key]}")
    dsp_result = dsp_results[result_key]

    raw_max_diff = float(np.abs(dsp_result - torch_result).max())
    audio_torch = istft_chunk(torch_result)
    audio_dsp = istft_chunk(dsp_result)
    err = audio_torch - audio_dsp
    snr_db = 10 * np.log10(np.mean(audio_torch**2) / np.mean(err**2))

    # Only c7x_dload reports real hardware cycles (from the DSP's TSC
    # counter); c7x_host is a host-emulation binary with no such counter.
    cycles = dsp_results.get(f"{dsp_mode}_cycles", 0)

    return {
        "dsp_results": dsp_results,
        "raw_max_diff": raw_max_diff,
        "snr_db": float(snr_db),
        "cycles": cycles,
    }


def _assert_passed(results: dict, dsp_mode: str) -> None:
    print(f"\n{dsp_mode} vs PyTorch: raw_max_diff={results['raw_max_diff']:.4f}  audio_SNR={results['snr_db']:.1f}dB")
    if results["cycles"]:
        print(f"{dsp_mode} cycles: {results['cycles']:,} ({results['cycles'] / 1e6:.2f} ms @ 1 GHz)")
    assert results["snr_db"] >= _MIN_SNR_DB, (
        f"{dsp_mode} audio-domain SNR too low: {results['snr_db']:.1f}dB < {_MIN_SNR_DB}dB "
        f"(raw spectrogram max diff was {results['raw_max_diff']:.4f})"
    )


@pytest.mark.quick
def test_gtcrn_c7x_host(dsp_mode):
    """GTCRN on C7x host emulation (fast, no hardware required)."""
    if dsp_mode not in (None, "c7x_host"):
        pytest.skip(f"test_gtcrn_c7x_host requires --dsp-mode=c7x_host (got {dsp_mode})")
    results = _run_gtcrn_dsp_test("c7x_host")
    _assert_passed(results, "c7x_host")


@pytest.mark.c7x_only
def test_gtcrn_c7x_dload(dsp_mode, record_cycles):
    """GTCRN deployed via DLOAD to real AM67A hardware."""
    if dsp_mode != "c7x_dload":
        pytest.skip("test_gtcrn_c7x_dload requires --dsp-mode=c7x_dload")
    results = _run_gtcrn_dsp_test("c7x_dload")
    _assert_passed(results, "c7x_dload")
    if results["cycles"]:
        record_cycles("gtcrn_c7x_dload", results["cycles"])


# -----------------------------------------------------------------------------
# Standalone Script Mode
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GTCRN DSP Test")
    parser.add_argument("--dsp-mode", required=True, choices=["c7x_host", "c7x_dload"])
    parser.add_argument("--frames", type=int, default=DEFAULT_T)
    add_board_arg(parser)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s"
    )

    print(f"\n{'=' * 60}\nGTCRN DSP Test\n  Mode: {args.dsp_mode}\n  Frames: {args.frames}\n{'=' * 60}\n")

    results = _run_gtcrn_dsp_test(args.dsp_mode, T=args.frames)

    print("\n" + "=" * 60 + "\nResults Summary\n" + "=" * 60)
    passed = results["snr_db"] >= _MIN_SNR_DB
    print(f"\n{args.dsp_mode} vs PyTorch:")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print(f"  Raw spectrogram max diff: {results['raw_max_diff']:.4f}")
    print(f"  Audio-domain SNR: {results['snr_db']:.1f}dB (threshold: {_MIN_SNR_DB}dB)")
    if results["cycles"]:
        print(f"  Cycles: {results['cycles']:,} ({results['cycles'] / 1e6:.2f} ms @ 1 GHz)")

    print("\n" + "=" * 60)
    print("TEST PASSED" if passed else "TEST FAILED")
    print("=" * 60)
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
