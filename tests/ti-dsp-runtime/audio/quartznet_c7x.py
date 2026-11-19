#!/usr/bin/env python
"""
QuartzNet 15x5 (English) model prep/compile helpers for the C7x DSP pipeline.

NVIDIA's NeMo checkpoint stt_en_quartznet15x5 (character CTC ASR, 18.9M
parameters, 28 characters plus the CTC blank) rebuilt in plain PyTorch, so
neither NeMo nor its dependencies are needed. The module tree mirrors NeMo's
ConvASREncoder/ConvASRDecoder, so the checkpoint's state dict loads strictly.

Only the encoder and the CTC head (NeMo's ConvASRDecoder, including its
log_softmax) run on the DSP, on a fixed [1, 64, T] log-mel input, as in
NeMo's own export. The mel front end runs on the host (log_mel), with the
window and filterbank taken from the checkpoint.

Each Conv1d is a Conv2d over a [1, C, 1, T] tensor here, with the same
weights: C7xMMAQuantizer annotates conv2d, and the MMALIB fusion passes
match conv2d. Masking (NeMo's conv_mask) is left out, as NeMo's export does;
every frame of the fixed-length input counts as valid.

The checkpoint (71 MB, NGC Terms of Use) and LibriSpeech test-clean
(CC-BY-4.0, 350 MB) are downloaded on first use and cached; see README.md.

Usage:
    python quartznet_c7x.py --dsp-mode c7x_host
    python quartznet_c7x.py --dsp-mode c7x_host --quantize int8
"""

import argparse
import io
import logging
import sys
import tarfile
import warnings
from functools import cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.export import export

import tvm
from tvm import relax
from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program

_THIS_DIR = Path(__file__).parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

import dsp_utils  # noqa: E402
from quant_recipe import calibrate_and_convert, export_module  # noqa: E402

logger = logging.getLogger(__name__)

_NEMO_NAME = "stt_en_quartznet15x5.nemo"
_NEMO_URL = (
    "https://api.ngc.nvidia.com/v2/models/nvidia/nemo/stt_en_quartznet15x5/"
    f"versions/1.0.0rc1/files/{_NEMO_NAME}"
)
_NEMO_SHA256 = "1b9b7b87a9277e6fef164d8f99d1226f0511af154423bbf919b920421ac9602f"

# LibriSpeech test-clean (2620 utterances, 40 speakers), pinned to the dataset
# repo's current commit. The model was trained on LibriSpeech's training
# speakers, none of whom are in test-clean. Every fifth test-clean speaker (by
# id) is for calibration and the rest for evaluation, so no evaluation clip,
# or voice, is calibrated on.
_LIBRISPEECH_REPO = "openslr/librispeech_asr"
_LIBRISPEECH_REVISION = "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1"
_LIBRISPEECH_FILE = "all/test.clean/0000.parquet"

# The checkpoint's decoder vocabulary (load_model checks it against the
# checkpoint's config); the CTC blank is the extra last class.
VOCAB = [" ", *"abcdefghijklmnopqrstuvwxyz", "'"]
BLANK = len(VOCAB)

# NeMo's preprocessor settings for this checkpoint (model_config.yaml) and
# FilterbankFeatures' defaults.
FS = 16000
N_FFT = 512
HOP = 160  # window_stride 0.01 s
WIN = 320  # window_size 0.02 s
N_MELS = 64
_PREEMPH = 0.97
_LOG_GUARD = 2.0**-24
_NORM_EPS = 1e-5

# (filters, repeat, kernel, stride, dilation, residual, separable) for each of
# the encoder's 18 blocks, from model_config.yaml.
_ENCODER_BLOCKS = (
    [(256, 1, 33, 2, 1, False, True)]
    + [(256, 5, 33, 1, 1, True, True)] * 3
    + [(256, 5, 39, 1, 1, True, True)] * 3
    + [(512, 5, 51, 1, 1, True, True)] * 3
    + [(512, 5, 63, 1, 1, True, True)] * 3
    + [(512, 5, 75, 1, 1, True, True)] * 3
    + [(512, 1, 87, 1, 2, False, True)]
    + [(1024, 1, 1, 1, 1, False, False)]
)

# Default input length in feature frames (10.23 s). Must be even (the first
# block has stride 2).
DEFAULT_T = 1024


class _Conv(nn.Module):
    """NeMo's MaskedConv1d without the mask, as a Conv2d over [N, C, 1, T]."""

    def __init__(self, cin, cout, kernel, stride=1, dilation=1, groups=1, bias=False):
        super().__init__()
        padding = dilation * (kernel - 1) // 2  # NeMo's get_same_padding
        self.conv = nn.Conv2d(
            cin,
            cout,
            (1, kernel),
            stride=(1, stride),
            padding=(0, padding),
            dilation=(1, dilation),
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


def _bn(channels):
    return nn.BatchNorm2d(channels, eps=1e-3, momentum=0.1)  # NeMo's JasperBlock BN


class _JasperBlock(nn.Module):
    """NeMo's JasperBlock (residual_mode "add", ReLU, eval-mode dropout).

    mconv keeps NeMo's layer indices: each repeat is [depthwise, pointwise,
    BN] (or [conv, BN] when not separable), followed by [ReLU, dropout] on all
    but the last repeat.
    """

    def __init__(self, cin, cout, repeat, kernel, stride, dilation, residual, separable):
        super().__init__()
        layers: list[nn.Module] = []
        c = cin
        for i in range(repeat):
            if separable:
                layers += [
                    _Conv(c, c, kernel, stride, dilation, groups=c),
                    _Conv(c, cout, 1),
                ]
            else:
                layers.append(_Conv(c, cout, kernel, stride, dilation))
            layers.append(_bn(cout))
            if i < repeat - 1:
                layers += [nn.ReLU(), nn.Identity()]
            c = cout
        self.mconv = nn.ModuleList(layers)
        self.res = (
            nn.ModuleList([nn.ModuleList([_Conv(cin, cout, 1), _bn(cout)])]) if residual else None
        )

    def forward(self, x):
        out = x
        for layer in self.mconv:
            out = layer(out)
        if self.res is not None:
            res = x
            for layer in self.res[0].children():
                res = layer(res)
            out = out + res
        return F.relu(out)


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        cin = N_MELS
        for filters, repeat, kernel, stride, dilation, residual, separable in _ENCODER_BLOCKS:
            blocks.append(
                _JasperBlock(cin, filters, repeat, kernel, stride, dilation, residual, separable)
            )
            cin = filters
        self.encoder = nn.Sequential(*blocks)

    def forward(self, x):
        return self.encoder(x)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_layers = nn.Sequential(nn.Conv2d(1024, BLANK + 1, 1, bias=True))

    def forward(self, x):
        return self.decoder_layers(x)


class QuartzNet(nn.Module):
    """Encoder + CTC head: [1, 64, T] log-mel features -> [1, T/2, 29] log-probs."""

    def __init__(self):
        super().__init__()
        self.encoder = _Encoder()
        self.decoder = _Decoder()

    def forward(self, features):
        x = self.encoder(features.unsqueeze(2))
        logits = self.decoder(x).squeeze(2).transpose(1, 2)
        return F.log_softmax(logits, dim=-1)


def _cached_download(url: str, sha256: str, name: str) -> Path:
    path = Path(torch.hub.get_dir()) / "checkpoints" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading %s to %s", url, path)
        torch.hub.download_url_to_file(url, str(path), hash_prefix=sha256)
    return path


def _checkpoint() -> tuple[dict, dict]:
    """(state dict, model_config.yaml) from the cached .nemo archive."""
    nemo = _cached_download(_NEMO_URL, _NEMO_SHA256, _NEMO_NAME)
    with tarfile.open(nemo) as tar:
        weights = tar.extractfile("./model_weights.ckpt")
        config = tar.extractfile("./model_config.yaml")
        assert weights is not None and config is not None
        state = torch.load(io.BytesIO(weights.read()), map_location="cpu", weights_only=True)
        return state, yaml.safe_load(config)


def load_model() -> tuple[QuartzNet, dict]:
    """Return (the eval-mode model, the preprocessor's window and filterbank)."""
    state, config = _checkpoint()
    vocab = config["decoder"]["vocabulary"]
    assert vocab == VOCAB, f"checkpoint vocabulary {vocab} != VOCAB"
    model = QuartzNet()
    weights = {}
    for key, value in state.items():
        if key.startswith("preprocessor."):
            continue
        # Conv1d weight [C_out, C_in/g, K] -> Conv2d weight [C_out, C_in/g, 1, K].
        if value.dim() == 3:
            value = value.unsqueeze(2)
        weights[key] = value
    model.load_state_dict(weights, strict=True)
    frontend = {
        "window": state["preprocessor.featurizer.window"],
        "fb": state["preprocessor.featurizer.fb"],
    }
    return model.eval(), frontend


def log_mel(waveform: np.ndarray, frontend: dict, T: int = DEFAULT_T) -> np.ndarray:
    """NeMo's FilterbankFeatures (eval mode) -> the model's [1, 64, T] input.

    Pre-emphasis, |STFT|^2 (centered, zero padded), the checkpoint's mel
    filterbank, log, then per-feature normalization over the utterance's
    frames; frames past the utterance are zero (NeMo's pad_value). A waveform
    longer than T frames is cut to fit.
    """
    max_samples = (T - 1) * HOP
    x = torch.from_numpy(np.asarray(waveform, dtype=np.float32)[:max_samples])
    n_frames = x.shape[0] // HOP + 1
    x = torch.cat([x[:1], x[1:] - _PREEMPH * x[:-1]])
    spec = torch.stft(
        x,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=WIN,
        window=frontend["window"],
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    power = torch.view_as_real(spec).pow(2).sum(-1)
    mel = torch.log(torch.matmul(frontend["fb"][0], power) + _LOG_GUARD)[:, :n_frames]
    mean = mel.mean(dim=1, keepdim=True)
    std = mel.std(dim=1, keepdim=True) + _NORM_EPS
    out = torch.zeros(N_MELS, T)
    out[:, :n_frames] = (mel - mean) / std
    return out.unsqueeze(0).numpy()


def greedy_decode(log_probs: np.ndarray) -> str:
    """CTC greedy decoding of [1, T', 29] log-probs."""
    ids = log_probs[0].argmax(axis=-1)
    chars = []
    prev = BLANK
    for i in ids:
        if i != prev and i != BLANK:
            chars.append(VOCAB[i])
        prev = i
    return "".join(chars).strip()


def _edit_distance(ref: list, hyp: list) -> int:
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1]


def error_rate(ref: str, hyp: str, words: bool) -> float:
    """WER (words=True) or CER of hyp against ref."""
    r, h = (ref.split(), hyp.split()) if words else (list(ref), list(hyp))
    return _edit_distance(r, h) / max(len(r), 1)


@cache
def _librispeech_rows() -> list[dict]:
    """test-clean's rows ({"id", "speaker_id", "text", "audio", "frames"}), by id."""
    import pyarrow.parquet as pq
    import soundfile as sf
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        _LIBRISPEECH_REPO,
        _LIBRISPEECH_FILE,
        repo_type="dataset",
        revision=_LIBRISPEECH_REVISION,
    )
    rows = pq.read_table(path, columns=["id", "speaker_id", "text", "audio"]).to_pylist()
    for row in rows:
        info = sf.info(io.BytesIO(row["audio"]["bytes"]))  # the FLAC header only
        assert info.samplerate == FS, f"{row['id']}: {info.samplerate} Hz"
        row["frames"] = info.frames // HOP + 1  # log_mel's frame count
    return sorted(rows, key=lambda r: r["id"])


def _speaker_rows(speakers: str) -> list[dict]:
    """Rows of the "calibration" speakers (every fifth by id) or the
    "evaluation" speakers (the others)."""
    rows = _librispeech_rows()
    calibration = set(sorted({r["speaker_id"] for r in rows})[::5])
    want = speakers == "calibration"
    return [r for r in rows if (r["speaker_id"] in calibration) == want]


def _decode(row: dict) -> dict:
    import soundfile as sf

    audio, _ = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    transcript = row["text"].lower()  # as the model spells
    return {"id": row["id"], "transcript": transcript, "audio": audio, "frames": row["frames"]}


def librispeech_utterances(
    speakers: str, count: int | None = None, max_frames: int | None = None
) -> list[dict]:
    """LibriSpeech test-clean utterances of the "calibration" or "evaluation"
    speakers, sorted by id.

    Each is {"id", "transcript", "audio": float32 16 kHz waveform, "frames"}.
    max_frames keeps the utterances whose features fit in that many frames.
    With count, every n-th utterance, count of them: sorting by id groups the
    utterances by speaker, so the picks spread over the speakers.
    """
    rows = _speaker_rows(speakers)
    if max_frames is not None:
        rows = [r for r in rows if r["frames"] <= max_frames]
    if count is not None:
        rows = rows[:: max(len(rows) // count, 1)][:count]
    return [_decode(r) for r in rows]


def calibration_inputs(frontend: dict, T: int = DEFAULT_T, count: int = 16) -> list[np.ndarray]:
    """Log-mel inputs for count calibration-speaker utterances, each cut or
    zero-padded to T frames."""
    return [log_mel(u["audio"], frontend, T) for u in librispeech_utterances("calibration", count)]


def eval_utterance(T: int = DEFAULT_T) -> dict:
    """The longest evaluation-speaker utterance that fits in T frames whole
    (the first by id among equals), so the input is mostly speech."""
    rows = [r for r in _speaker_rows("evaluation") if r["frames"] <= T]
    return _decode(min(rows, key=lambda r: (-r["frames"], r["id"])))


def _bind_params(mod: tvm.IRModule) -> tvm.IRModule:
    mod, params = relax.frontend.detach_params(mod)
    func_params_dict = dict(zip(mod["main"].params[1:], params["main"]))
    return relax.transform.BindParams(
        func_name="main",
        params=func_params_dict,  # pyright: ignore[reportArgumentType]
    )(mod)


def export_and_bind(model: nn.Module, T: int = DEFAULT_T) -> tvm.IRModule:
    """torch.export the float model with a fixed [1, 64, T] input, import into
    Relax with the parameters bound as constants."""
    with torch.no_grad():
        exported_program = export(model, (torch.zeros(1, N_MELS, T),))
        mod = from_exported_program(exported_program, keep_params_as_input=True)
    return _bind_params(mod)


def quantize(
    model: nn.Module, calib: list[np.ndarray], dtype: str = "int8"
) -> tuple[tvm.IRModule, torch.fx.GraphModule]:
    """PT2E-quantize with C7xMMAQuantizer, calibrated on calib.

    Activation ranges come from the histogram observer. With min/max ranges a
    few extreme activations set the scales: at T=1024, 16 calibration
    utterances gave 16% WER on 50 evaluation utterances against 3.8% for
    float32, more calibration data made it worse, and the result moved by a
    point with float rounding alone (thread count). With the histogram
    observer the same set scores 3.5-3.8%.

    Returns (the Relax module with parameters bound, the Q/DQ GraphModule
    that serves as the PyTorch reference).
    """
    example = (torch.from_numpy(calib[0]),)
    samples = [(torch.from_numpy(c),) for c in calib]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="torchao")
        quantized_gm = calibrate_and_convert(
            export_module(model, example),
            C7xMMAQuantizer(dtype, activation_observer="histogram"),
            samples,
        )
    with torch.no_grad():
        exported_program = export(quantized_gm, example)
        mod = from_exported_program(exported_program, keep_params_as_input=True)
    return _bind_params(mod), quantized_gm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dsp-mode",
        default="c7x_host",
        choices=["c7x_host", "c7x_dload"],
        help="DSP execution mode",
    )
    parser.add_argument("--frames", type=int, default=DEFAULT_T, help="Fixed feature frame count")
    parser.add_argument("--quantize", choices=["int8", "int16"], default=None)
    dsp_utils.add_board_arg(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    model, frontend = load_model()
    if args.quantize:
        calib = calibration_inputs(frontend, args.frames)
        mod, _ = quantize(model, calib, args.quantize)
        target = dsp_utils.get_target_string(args.dsp_mode, use_cpp_api=True) + " -mmalib=1"
    else:
        mod = export_and_bind(model, args.frames)
        target = dsp_utils.get_target_string(args.dsp_mode, use_cpp_api=True)

    print(f"Compiling for {target} ...")
    generated_dir = dsp_utils.compile_for_dsp(mod, target)
    print(f"Generated: {generated_dir}")


if __name__ == "__main__":
    main()
