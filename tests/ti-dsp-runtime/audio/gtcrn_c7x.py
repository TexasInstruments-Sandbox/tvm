#!/usr/bin/env python
"""
GTCRN model prep/compile CLI for the C7x DSP pipeline.

GTCRN (github.com/Xiaobin-Rong/gtcrn, MIT license) is a tiny
speech-enhancement model (23.67K params) operating on a fixed-shape
[1, 257, T, 2] STFT chunk (257 = n_fft/2+1 for n_fft=512, T = frame count,
last dim = [real, imag]). The model source and trained checkpoint are
vendored under 3rdparty/gtcrn/ (TVM's convention for third-party code; see
README.md for exact provenance) rather than cloned/downloaded at runtime --
both are tiny (13KB source, 566KB checkpoint) and this avoids any
network/credential dependency in CI. Per TI's
bitbucket fork's ti/export_onnx_fixed.py docstring, a fixed (non-dynamic)
time axis is required for TVM import - this module goes straight from the
PyTorch source to Relax via torch.export instead of through an intermediate
ONNX file, using the same fixed-shape convention.

Usage:
    python gtcrn_c7x.py --dsp-mode c7x_host
    python gtcrn_c7x.py --dsp-mode c7x_dload
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.export import export

import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

import dsp_utils  # noqa: E402

logger = logging.getLogger(__name__)

# STFT convention per the TI fork's ti/infer_onnx_fixed.py: centered STFT
# (librosa default), sqrt-Hann window, n_fft=512, hop=256.
FS = 16000
N_FFT = 512
HOP = 256
WIN = 512
DEFAULT_T = 63  # matches onnx-models/gtcrn/gtcrn_dns3.onnx's fixed frame count


_3RDPARTY_DIR = _THIS_DIR / "3rdparty" / "gtcrn"


def load_model(checkpoint: str = "model_trained_on_dns3.tar") -> nn.Module:
    """Load GTCRN with the given checkpoint, unmodified from the upstream source.

    Earlier revisions of this module patched two upstream ops (nn.Unfold-based
    SFE, and the decoder's dilated ConvTranspose2d) to work around gaps in
    TVM's torch->Relax->c_static pipeline. Both gaps are now fixed directly in
    TVM (conv2d_transpose dilation support in topi/nn/conv2d_transpose.py, and
    the index_tensor mixed basic+advanced indexing fix in
    base_fx_graph_translator.py's _index_tensor), so no model-side patching is
    needed anymore.
    """
    sys.path.insert(0, str(_3RDPARTY_DIR))
    import gtcrn  # noqa: E402  # pyright: ignore[reportMissingImports]

    model = gtcrn.GTCRN().eval()
    ckpt_path = _3RDPARTY_DIR / "checkpoints" / checkpoint
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model


def export_and_bind(model: nn.Module, T: int = DEFAULT_T) -> tvm.IRModule:
    """torch.export the model with a fixed [1, 257, T, 2] input, import into
    Relax with parameters bound as constants (the same pattern used
    throughout this test suite, e.g. SmolLM/smollm_c7x.py).

    PyTorch's default decomposition unrolls aten.gru.input into ~1500
    primitive ops per GRU instance *before* TVM's frontend ever sees a GRU
    node (GTCRN has 14 GRU instances at T=63 steps each) -- this is the
    actual source of the C7x compile-time blowup, not TVM's own GRU
    converter. Building a decomp_table with the gru-related keys removed and
    passing it to run_decompositions() explicitly keeps aten.gru.input
    opaque, so TVM's from_exported_program (with run_ep_decomposition=False)
    hands it to _gru/_gru_cell_unroll, which lowers it via topi.nn.gru's
    genuine TIR loop instead of unrolling.
    """
    example_input = (torch.randn(1, 257, T, 2),)
    with torch.no_grad():
        exported_program = export(model, example_input)
        decomp_table = torch.export.default_decompositions()
        for op in list(decomp_table):
            if "gru" in str(op).lower():
                del decomp_table[op]
        exported_program = exported_program.run_decompositions(decomp_table=decomp_table)
        mod = from_exported_program(
            exported_program, keep_params_as_input=True, run_ep_decomposition=False
        )
    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    mod = relax.transform.BindParams(  # pyright: ignore[reportArgumentType]
        func_name="main", params=func_params_dict
    )(mod)
    return mod


def chunk_samples(T: int = DEFAULT_T) -> int:
    """Raw audio sample count that produces exactly T centered-STFT frames."""
    return (T - 1) * HOP


def stft_chunk(waveform: np.ndarray) -> np.ndarray:
    """Centered STFT + sqrt-Hann window -> the model's [1, 257, T, 2] input layout."""
    import librosa

    window = (np.hanning(WIN) ** 0.5).astype(np.float32)
    spec = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP, win_length=WIN, window=window)
    return np.stack([spec.real, spec.imag], axis=-1)[np.newaxis].astype(np.float32)


def istft_chunk(model_output: np.ndarray) -> np.ndarray:
    """Inverse of stft_chunk: the model's [1, 257, T, 2] output -> a waveform chunk."""
    import librosa

    window = (np.hanning(WIN) ** 0.5).astype(np.float32)
    spec = model_output[0, ..., 0] + 1j * model_output[0, ..., 1]
    return librosa.istft(spec, hop_length=HOP, win_length=WIN, window=window)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsp-mode", default="c7x_host", choices=["c7x_host", "c7x_dload"],
        help="DSP execution mode",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_T, help="Fixed STFT frame count")
    parser.add_argument("--checkpoint", default="model_trained_on_dns3.tar")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print(f"Loading GTCRN ({args.checkpoint}) from {_3RDPARTY_DIR} ...")
    model = load_model(args.checkpoint)

    print(f"torch.export with T={args.frames} frames ...")
    mod = export_and_bind(model, T=args.frames)

    print("Compiling for c_static -mcpu=c7x ...")
    generated_dir = dsp_utils.compile_for_dsp(mod, "c_static -mcpu=c7x")
    print(f"Generated: {generated_dir}")


if __name__ == "__main__":
    main()
