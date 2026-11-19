# Audio models on C7x DSP

- [GTCRN](#gtcrn-speech-enhancement): speech enhancement, float32.
- [QuartzNet 15x5](#quartznet-15x5-english-speech-recognition): English speech
  recognition, float32 and int8 (C7xMMAQuantizer + MMALIB).

## GTCRN speech enhancement

GTCRN (23.67K params, 33 MMACs) is a tiny speech-enhancement model operating
on a fixed-shape `[1, 257, T, 2]` STFT chunk (257 = n_fft/2+1 for n_fft=512,
T = frame count, last dim = `[real, imag]`). This directory compiles it for
TI's C7x DSP and validates it against a PyTorch (host CPU) reference, in
both `c7x_host` (x86 emulation, no hardware) and `c7x_dload` (DLOAD deploy
to an AM67A board) modes.

### Status

| Mode | Result | Audio-domain SNR | Cycles |
|------|--------|-------------------|--------|
| `c7x_host` | PASS | 45.1 dB | n/a (host emulation, no hardware counter) |
| `c7x_dload` | PASS (AM67A hardware) | 45.1 dB | 537,984,515 (≈538 ms @ 1 GHz) |

The `c7x_dload` cycle count is for one `T=63` chunk (≈0.99 s of 16 kHz
audio), a real-time factor of ≈0.54.

### Model source (vendored)

`3rdparty/gtcrn/` (TVM's own convention for vendored third-party code, e.g.
the top-level `3rdparty/` directory) contains files copied from TI's
internal fork of the
public [`Xiaobin-Rong/gtcrn`](https://github.com/Xiaobin-Rong/gtcrn) repo
(`ssh://git@bitbucket.itg.ti.com/audioai-algo/gtcrn.git`, commit
`57d878eda21d53c43db7bfed562c85c58c8a8c89`, obtained 2026-07-21):

- `gtcrn.py` — the model source (13 KB), unmodified.
- `checkpoints/model_trained_on_dns3.tar` — the trained checkpoint (566 KB).
  It matches the model behind the TI-internal
  `onnx-models/gtcrn/gtcrn_dns3.onnx` export to within float32 noise
  (~2.6e-6, compared in onnxruntime on CPU).
- `LICENSE` — the upstream repo's MIT license, included per its terms.

Both files are tiny and change rarely, so they're vendored rather than
cloned or downloaded at test time: no network or credential dependency in
CI. This matches the convention elsewhere in `tests/ti-dsp-runtime/` of
committing small model weights (e.g. `yolov5n.pt`/`yolov8n.pt` at the repo
root); models too large to vendor, such as SmolLM, download on demand.

The time axis is fixed (non-dynamic) because TVM can't import the symbolic
dimension that `torch.onnx.export`'s `dynamic_axes` produces; TI's fork's
`ti/export_onnx_fixed.py` (not vendored) uses the same convention.
`gtcrn_c7x.py` goes straight from the PyTorch source to Relax via
`torch.export` with a concrete `T`, without an intermediate ONNX file.

### STFT convention

Centered STFT (`librosa.stft`'s `center=True` default) with a **sqrt-Hann**
window, `n_fft=512`, `hop=256`, matching that fork's `ti/infer_onnx_fixed.py`
reference script (PyTorch vs. onnxruntime on the same input: ~2.6e-6 max
diff). For `T=63` frames, the chunk length is `(T-1)*hop = 15872` samples
(`gtcrn_c7x.chunk_samples`).

### Files

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest fixtures for this directory (`dsp_mode`, `record_cycles`, etc.); `--board` comes from `tests/ti-dsp-runtime/conftest.py` |
| `gtcrn_c7x.py` | Model loading, `torch.export`→Relax conversion, and STFT/ISTFT helpers |
| `test_gtcrn_e2e.py` | The `c7x_host`/`c7x_dload` pytest tests, plus a standalone-script mode |
| `3rdparty/gtcrn/` | Vendored upstream model source + checkpoint + license (see [Model source](#model-source-vendored)) |

### Quick start

```bash
export TI_CGT_C7000_PATH=<path to TI C7000 CGT>

# c7x_host (no hardware) -- ~15 s
pytest test_gtcrn_e2e.py --dsp-mode=c7x_host -v -s

# c7x_dload (board) -- ~5 min, most of it cl7x cross-compile
pytest test_gtcrn_e2e.py --dsp-mode=c7x_dload --board beagley-ai -v -s

# Standalone script mode (same logic, no pytest)
python test_gtcrn_e2e.py --dsp-mode c7x_host
```

### TVM support GTCRN depends on

1. **`conv2d_transpose` dilation** — GTCRN's decoder uses dilated depthwise
   `ConvTranspose2d`. TOPI's `conv2d_transpose` legalization handles dilation
   with the same "effective kernel size" approach as ordinary `conv2d`.
2. **`aten.index.Tensor` mixed indexing** — `nn.Unfold` decomposes (via
   `torch.export`) into an advanced-indexing call with a `None` prefix mixed
   with real indices of different rank. The torch frontend's
   `_index_tensor` handles it by permuting the indexed axes to the front,
   calling `index_tensor`, and permuting back, matching NumPy's placement
   rules.
3. **GRUs as loops** — GTCRN has 14 `nn.GRU` instances of 63 timesteps each.
   `gtcrn_c7x.py` passes `run_decompositions` a decomposition table without
   the GRU entries, so `aten.gru.input` reaches the frontend whole, and the
   frontend lowers it to `topi.nn.gru` (`te.extern` around a hand-written
   `tir.ir_builder` loop): one `for` loop in the generated C instead of one
   block of code per timestep. `te.scan`, which `topi.nn.lstm` uses, isn't an
   option: `te.create_prim_func`, which `block_builder.call_te` goes through,
   rejects `te.ScanOp`.

### Correctness methodology: audio-domain SNR, not raw spectrogram diff

`test_gtcrn_e2e.py` gates correctness on **audio-domain SNR** (ISTFT both
the DSP and PyTorch outputs, compare waveforms), not raw complex-spectrogram
max-diff. The raw spectrogram difference between C7x and PyTorch varies
widely with the input (0.27 to 0.76 max abs), presumably from the 63-step
GRU recurrence compounding small differences between the C7x vector ISA's
transcendental (sigmoid/tanh) implementations and PyTorch's. Audio-domain
SNR over the same runs is 39.8–48.0 dB (near-imperceptible) regardless, so
it's the stable invariant to test. The `_MIN_SNR_DB` threshold (20 dB)
leaves a wide margin under that range: a real bug is expected to give a much
lower SNR, not a borderline one.

### Performance

| Stage | Time | Applies to |
|-------|------|------------|
| TVM codegen (`relax.build`, Relax→TIR→C) | under 15 s | Both modes |
| `g++` host-emulation build | seconds | `c7x_host` only |
| `cl7x -O3` cross-compile | ~4.6 min | `c7x_dload` only |

The 14 GRUs lower to 4 kernel functions (GRUs of the same shape share one),
each a `for` loop, and the generated dispatcher (`__tvm_ffi___vmtir__main` in
`lib0.c`) is ~148 KB / ~3,245 lines. A full `test_gtcrn_c7x_dload` run (TVM
codegen, cmake build, board deploy and run) takes ~4:48.

Inference: **537,984,515 cycles (≈538 ms @ 1 GHz)** per `T=63` (~0.99 s
audio) chunk on AM67A hardware, a real-time factor of ≈0.54.

### Known limitations

- The compiled model has a **static** input shape (`T=63` by default) — a
  different chunk duration requires recompiling with a different `T` passed
  to `export_and_bind`.
- `test_gtcrn_c7x_dload` needs a reachable board, named with `--board`, and
  is marked `@pytest.mark.c7x_only`; it skips when `--dsp-mode=c7x_dload`
  isn't passed.

## QuartzNet 15x5 English speech recognition

QuartzNet 15x5 is NVIDIA NeMo's `stt_en_quartznet15x5`: character-level CTC
speech recognition for English, 18.9M parameters, 28 characters (space, a-z,
apostrophe) plus the CTC blank. Its encoder is 18 blocks of time-channel
separable 1D convolutions (depthwise kernels 33 to 87 frames, then pointwise),
BatchNorm and ReLU, with residual adds in 15 of them. This directory runs the
encoder and the CTC head (a 1x1 conv and log_softmax) on the DSP, on a fixed
`[1, 64, T]` log-mel input, as NeMo's own export does. The mel front end runs
on the host. At the default `T=1024` (10.23 s of audio) a run is about 9.7
GMAC, about 0.95 GMAC per second of audio.

### Status

Input `T=1024` (10.23 s of audio), LibriSpeech utterance `7729-102255-0024`;
the reference is PyTorch float32 for float32 and the Q/DQ graph for int8.

| Mode | dtype | Max \|log-prob diff\| | Frame argmax agreement | Transcript CER vs reference | Run time |
|------|-------|----------------------|-----------------------|-----------------------------|----------|
| `c7x_host` | float32 | 1.7e-4 | 1.0000 | 0 | ~20 s emulated |
| `c7x_host` | int8 | 9.5 | 0.9863 | 0.0068 | ~5 min emulated |
| `c7x_dload` (BeagleY-AI) | float32 | 8.7e-5 | 1.0000 | 0 | 140,707,178,327 cycles (≈141 s @ 1 GHz) |
| `c7x_dload` (BeagleY-AI) | int8 | 9.5 | 0.9863 | 0.0068 | 312,298,643 cycles (≈0.31 s @ 1 GHz) |

int8 runs at a real-time factor of about 0.03 on the board (DSP compute
only, at 1 GHz); float32 at about 13.8, so float32 is a correctness check,
not a deployment option. Each test (one variant) takes about 3 minutes on the
board, most of it compile time and the float32 run. int8 needs firmware
service version 5.1.0 or later, which exports `c7x_int8_dwconv1d`.

### int8 layer breakdown on the board

One steady-state inference (`--profile`, second of two runs) on BeagleY-AI,
312 M cycles over 268 layers:

| Kernel | Layers | Cycles | Share |
|--------|--------|--------|-------|
| Depthwise conv (`c7x_int8_dwconv1d`) | 77 | 246,620,751 | **79.6%** |
| Pointwise conv and CTC head (MMALIB) | 94 | 29,668,099 | 9.6% |
| ReLU (`c7x_int8_relu`) | 78 | 17,657,659 | 5.7% |
| log_softmax | 1 | 4,826,645 | 1.6% |
| Residual add | 15 | 4,180,962 | 1.3% |
| Requantize | 1 | 3,951,415 | 1.3% |
| Input quantize | 1 | 2,700,241 | 0.9% |
| Output dequantize | 1 | 392,910 | 0.1% |

The depthwise convs take 80% of the run with about 9% of the MACs; the
pointwise convs, with about 89% of the MACs, run on MMALIB in 10% of it. By
shape (C channels, 512 output frames):

| C | Kernel | Stride | Dilation | Layers | Cycles per layer | Total | MACs per cycle |
|---|--------|--------|----------|--------|------------------|-------|----------------|
| 64 | 1x33 | 2 | 1 | 1 | 1.53 M | 1.5 M | 0.7 |
| 256 | 1x33 | 1 | 1 | 15 | 1.59 M | 23.8 M | 2.7 |
| 256 | 1x39 | 1 | 1 | 15 | 1.72 M | 25.8 M | 3.0 |
| 256 | 1x51 | 1 | 1 | 1 | 1.94 M | 1.9 M | 3.4 |
| 512 | 1x51 | 1 | 1 | 14 | 3.89 M | 54.5 M | 3.4 |
| 512 | 1x63 | 1 | 1 | 15 | 4.25 M | 63.7 M | 3.9 |
| 512 | 1x75 | 1 | 1 | 15 | 4.70 M | 70.5 M | 4.2 |
| 512 | 1x87 | 1 | 2 | 1 | 4.93 M | 4.9 M | 4.6 |

MMALIB's depthwise kernel takes 3x3, 5x5 and 7x7 only, so these run on
`c7x_int8_dwconv1d` (`src/runtime/ti_dsp/kernels/c7x_dwconv1d.cpp`, through
`FuseQDQToC7xDwConv1d`). Its inner loop does about 14 MACs a cycle; it
averages 3 to 4.6 because each block of 32 outputs also pays the tap loop's
pipeline start-up and the requantize, which is why longer kernels get closer.
The ReLUs follow MMALIB convs and could be fused into them, and log_softmax
isn't needed for greedy decoding; together they are 7% of the run.

### Accuracy: float32 vs int8

Transcripts of 100 utterances of the evaluation speakers (every n-th by id
among those up to 20.47 s, whole, at `T=2048`: 31 speakers, 747 s, 2002
words) from PyTorch: the float32 model and the Q/DQ graphs `C7xMMAQuantizer`
produces, calibrated as `quantize` does (below).

| Model | WER | CER | CER vs float32 transcript |
|-------|-----|-----|---------------------------|
| float32 | 4.00% | 1.18% | — |
| int8 | 4.55% | 1.34% | 0.30% |
| int16 | 4.05% | 1.17% | 0.05% |

On the board, over 50 evaluation utterances that fit `T=1024` whole (629
words), int8 on the DSP scores WER 3.50% and CER 1.12% against float32's
3.82% and 1.24% in PyTorch, and its transcripts differ from float32's by
0.24% CER. Most likely class per frame agrees with float32's on 98.4% of
frames, and with the Q/DQ graph's on 98.7%.

On the DSP, for the test utterance (truth: *the free state men clung to their
prairie towns and prairie ravines with all the obstinacy and courage of true
defenders of their homes and firesides*):

| | Transcript | CER | WER |
|-|------------|-----|-----|
| float32, DSP and PyTorch | the free state men clung to their prairie towns and prairie avins with all the obstinacy and courage of true defenders of their homes and firesides | 0.013 | 0.038 |
| int8, DSP | the free state men clung to their prairie towns and prairie avins with all the obstinacy and courage of true defenders of their homes and firesides | 0.013 | 0.038 |
| int8, Q/DQ graph | the free state men clung to their prairie towns and prairie avines with all the obstinacy and courage of true defenders of their homes and firesides | 0.007 | 0.038 |

### int8 quantization recipe

`quantize` calibrates `C7xMMAQuantizer("int8", activation_observer="histogram")`
on 16 utterances. The quantizer's default min/max observer doesn't work for
this model: a few extreme activations set the ranges, so more calibration
data makes it worse, and its results move with float rounding alone.
PyTorch, Q/DQ graph:

| Calibration | WER, 300 utterances at `T=2048` (float32 4.28%) | WER, 50 utterances at `T=1024`, 12 / 1 threads (float32 3.82%) |
|-------------|------------------------------------------|-------------------------------------------------|
| 16 utterances, min/max | 5.10% | 16.38% / 15.42% |
| 64 utterances, min/max | 12.34% | |
| 256 utterances, min/max | 59.73% | |
| 515 utterances, min/max | 18.38% | |
| **16 utterances, histogram** | **4.47%** | **3.50% / 3.66%** |
| 64 utterances, histogram | 4.56% | 3.82% / 3.50% |

With min/max ranges, turning off `depthwise_block_percentile` or
per-channel depthwise inputs (`depthwise_input_headroom=None`) didn't fix it
(5.55% and 9.69% at 64 utterances, 300-utterance set).

MMALIB requantizes each conv's output channel with an integer multiplier and
a shift. The conv and depthwise conv wrappers take multipliers up to 255
(`_MMALIB_CONV_MAX_SCALE`, half the worst-case rounding error of the 127
limit MMALIB's FC kernel needs). On the board, 50 utterances at `T=1024`:

| MMALIB conv multiplier | DSP WER | DSP CER | DSP transcript CER vs float32 |
|------------------------|---------|---------|-------------------------------|
| up to 127 | 3.97% | 1.15% | 0.48% |
| up to 255 | 3.50% | 1.12% | 0.24% |

### Model source (downloaded, not vendored)

The checkpoint is NGC's `stt_en_quartznet15x5` 1.0.0rc1 `.nemo` archive (71
MB, under the NGC Terms of Use, which rules out committing it). It is
downloaded on first use, checked against its SHA-256 and cached in
`$TORCH_HOME/hub/checkpoints/`. `model_weights.ckpt` is read with
`torch.load(weights_only=True)`, and the vocabulary in `model_config.yaml` is
checked against the one `quartznet_c7x.py` decodes with.

`quartznet_c7x.py` rebuilds the model in plain PyTorch: neither NeMo nor its
dependencies are needed. The module tree copies NeMo's `ConvASREncoder` and
`ConvASRDecoder`, so the state dict loads with `strict=True`. Against NeMo's
own ONNX export of the same checkpoint
([OpenVoiceOS/stt_en_quartznet15x5_onnx](https://huggingface.co/OpenVoiceOS/stt_en_quartznet15x5_onnx),
commit `7a98393`) in onnxruntime, on three LibriSpeech test-clean utterances,
the port's log-probs agree within 8.3e-4 and on the most likely character on
every frame. That comparison isn't part of the test.

Two departures from NeMo's code, neither of which changes the arithmetic:

- Every Conv1d is a Conv2d over a `[1, C, 1, T]` tensor, with the weights
  unsqueezed. `C7xMMAQuantizer` annotates conv2d and the MMALIB fusion passes
  match conv2d, not conv1d.
- NeMo's `conv_mask` (zeroing frames past each utterance's length before every
  conv) is left out, as NeMo's export leaves it out. Frames past the utterance
  are zeros in the input and count as valid.

The front end (`log_mel`) is NeMo's `FilterbankFeatures` in eval mode:
pre-emphasis 0.97, centered 512-point STFT with a 320-sample Hann window and
160-sample hop, power spectrum, the checkpoint's own 64-band mel filterbank,
log, and per-feature normalization over the utterance's frames.

### Audio: LibriSpeech test-clean

The test downloads LibriSpeech test-clean
([openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr),
CC-BY-4.0, one 350 MB parquet file of FLAC audio) at a pinned revision through
`huggingface_hub`, into the Hugging Face cache, and reads it with `pyarrow` and
`soundfile` (`setup_test_env.sh` installs both). The model was trained on
LibriSpeech's training speakers, none of whom are in test-clean.

test-clean's 40 speakers are split by id: every fifth speaker (8 speakers, 515
utterances) is for calibration and the other 32 (2105 utterances) for
evaluation, so no evaluation clip, or voice, is calibrated on.

- Calibration: 16 utterances of the calibration speakers, every n-th by id so
  they spread over the speakers, each cut or zero-padded to T frames (see
  [int8 quantization recipe](#int8-quantization-recipe)).
- Evaluation: the longest utterance of the evaluation speakers that fits in T
  frames whole, `7729-102255-0024` (10.23 s, all 1024 frames).

### Files

| File | Purpose |
|------|---------|
| `quartznet_c7x.py` | Model, checkpoint download and loading, log-mel front end, LibriSpeech loading, CTC greedy decoding, CER/WER, float export and `C7xMMAQuantizer` quantization |
| `test_quartznet_e2e.py` | `c7x_host` / `c7x_dload` tests, each for float32 and int8, plus a standalone-script mode (also takes `--dtype int16`) |

### Quick start

```bash
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_6.2.1.LTS

# c7x_host, float32 and int8
pytest test_quartznet_e2e.py --dsp-mode=c7x_host -v -s

# c7x_dload (BeagleY-AI)
pytest test_quartznet_e2e.py --dsp-mode=c7x_dload --board beagley-ai -v -s

# One variant, without pytest
python test_quartznet_e2e.py --dsp-mode c7x_host --dtype int8
```

### Correctness gates

- **float32**: log-probs within 0.01 of PyTorch's, and the same transcript.
- **int8**: compared with the Q/DQ graph `convert_pt2e` returns (fake quant),
  the same reference the quantized classifier tests use, and gated on the
  transcript: its character error rate against the reference's at most 0.02
  (about 3 characters of the utterance's 150; measured 0.0068, one
  character). Frame agreement is reported, not gated.

  The DSP's integer kernels don't requantize exactly as the graph's float
  scales do (MMALIB, for one, takes a per-channel 8-bit multiplier and a
  shift), so the DSP and the graph are separate approximations of float32.
  On the test utterance (512 output frames, on the board):

  | Most likely class agrees | Frames |
  |--------------------------|--------|
  | DSP int8 vs Q/DQ int8 | 98.6% |
  | DSP int8 vs float32 | 97.5% |
  | Q/DQ int8 vs float32 | 97.7% |

  All 7 frames where the DSP and the graph differ are frames where one emits a
  character and the other the CTC blank: where the character is emitted moves
  by a frame. On the 236 frames where both emit a character, they agree.
  Per-frame cosine similarity of the log-probs doesn't separate this from
  noise, and the mean total-variation distance of the frame distributions only
  counts the same flipped frames, so neither makes a better gate. The
  arithmetic itself is checked bit for bit by the kernels' unit tests, and the
  int8 output is identical on `c7x_host` and the board.

### Known limitations

- Static input shape: `T` is fixed at compile time (`--frames` /
  `export_and_bind(model, T)`); a longer utterance is cut.
- No masking: frames past an utterance's end are zeros, not masked out, which
  can change the last characters slightly compared with NeMo's own inference.
- The first run downloads about 420 MB (checkpoint and LibriSpeech test-clean).
- int8 on `c7x_host` is slow (about 5 minutes at `T=1024`): host emulation
  runs the MMA bit-accurately.
