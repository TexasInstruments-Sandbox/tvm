# GTCRN on C7x DSP

GTCRN (23.67K params, 33 MMACs) is a tiny speech-enhancement model operating
on a fixed-shape `[1, 257, T, 2]` STFT chunk (257 = n_fft/2+1 for n_fft=512,
T = frame count, last dim = `[real, imag]`). This directory compiles it for
TI's C7x DSP and validates it against a PyTorch (host CPU) reference, in
both `c7x_host` (x86 emulation, no hardware) and `c7x_dload` (real DLOAD
deploy to an AM67A board) modes.

## Status

| Mode | Result | Audio-domain SNR | Cycles |
|------|--------|-------------------|--------|
| `c7x_host` | PASS | 45.1 dB | n/a (host emulation, no hardware counter) |
| `c7x_dload` | PASS (real AM67A hardware) | 45.1 dB | 537,984,515 (≈538 ms @ 1 GHz) |

The `c7x_dload` cycle count is for one `T=63` chunk (≈0.99 s of 16 kHz
audio), giving a real-time factor of ≈0.54 (≈538 ms of DSP compute for
≈992 ms of audio) — comfortably faster than real-time on the actual
hardware. See [Performance](#performance) below for the compile-time
breakdown, including the GRU-loop fix that cut `cl7x` cross-compile time
by ~4.7x with no change to correctness or these cycle numbers.

## Model source (vendored)

`3rdparty/gtcrn/` (TVM's own convention for vendored third-party code, e.g.
the top-level `3rdparty/` directory) contains two files copied from TI's
internal fork of the
public [`Xiaobin-Rong/gtcrn`](https://github.com/Xiaobin-Rong/gtcrn) repo
(`ssh://git@bitbucket.itg.ti.com/audioai-algo/gtcrn.git`, commit
`57d878eda21d53c43db7bfed562c85c58c8a8c89`, obtained 2026-07-21):

- `gtcrn.py` — the model source (13 KB), unmodified.
- `checkpoints/model_trained_on_dns3.tar` — the trained checkpoint (566 KB).
  Confirmed via a separate onnxruntime-CPU comparison to match the model
  behind the TI-internal `onnx-models/gtcrn/gtcrn_dns3.onnx` export to
  within float32 noise (~2.6e-6).
- `LICENSE` — the upstream repo's MIT license, included per its terms.

Both files are tiny and change rarely, so they're vendored directly rather
than cloned or downloaded at test time — this avoids any network/credential
dependency in CI (no SSH key provisioning needed for `bitbucket.itg.ti.com`,
no risk of the upstream repo being reorganized or made unreachable). This
matches the existing convention elsewhere in `tests/ti-dsp-runtime/` of
committing model weights directly (e.g. the `yolov5n.pt`/`yolov8n.pt` files
at the repo root) rather than SmolLM's download-on-demand approach, which
exists specifically because SmolLM's weights (333 MB - 621 MB) are too large
to vendor.

TI's fork's `ti/export_onnx_fixed.py` (not vendored — only used as a design
reference, not imported at runtime) documents *why* a fixed (non-dynamic)
time axis is required for TVM import: `torch.onnx.export`'s `dynamic_axes`
produces a symbolic dim that TVM can't import; a concrete `T` bakes every
dimension to a fixed integer instead. `gtcrn_c7x.py` follows the same
fixed-shape convention, but goes straight from the PyTorch source to Relax
via `torch.export` rather than through an intermediate ONNX file.

## STFT convention

Centered STFT (`librosa.stft`'s `center=True` default) with a **sqrt-Hann**
window, `n_fft=512`, `hop=256` — matching that fork's `ti/infer_onnx_fixed.py`
reference script exactly (confirmed by feeding it the same input and
comparing PyTorch vs. onnxruntime output: ~2.6e-6 max diff). For `T=63`
frames, the required chunk length is `(T-1)*hop = 15872` samples
(`gtcrn_c7x.chunk_samples`).

## Files

| File | Purpose |
|------|---------|
| `conftest.py` | Pytest fixtures (`dsp_mode`, `record_cycles`, etc.), following the standard per-directory pattern used elsewhere in `tests/ti-dsp-runtime/` (no shared root conftest exists) |
| `gtcrn_c7x.py` | Model loading, `torch.export`→Relax conversion, and STFT/ISTFT helpers |
| `test_gtcrn_e2e.py` | The `c7x_host`/`c7x_dload` pytest tests, plus a standalone-script mode |
| `3rdparty/gtcrn/` | Vendored upstream model source + checkpoint + license (see [Model source](#model-source-vendored)) |

## Quick start

```bash
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS

# c7x_host (fast, no hardware) -- ~15 s
pytest test_gtcrn_e2e.py --dsp-mode=c7x_host -v -s

# c7x_dload (real AM67A hardware) -- ~5 min, dominated by cl7x cross-compile
pytest test_gtcrn_e2e.py --dsp-mode=c7x_dload -v -s

# Standalone script mode (same logic, no pytest)
python test_gtcrn_e2e.py --dsp-mode c7x_host
```

## Compiling GTCRN required three TVM fixes

1. **`conv2d_transpose` dilation** — GTCRN's decoder uses dilated depthwise
   `ConvTranspose2d`, which TOPI's legalization used to reject outright
   (`topi/nn/conv2d_transpose.py` never threaded a `dilation` parameter
   through its padding/indexing math). Fixed by mirroring the "effective
   kernel size" pattern already used by ordinary `conv2d`.
2. **`aten.index.Tensor` mixed indexing** — `nn.Unfold` decomposes (via
   `torch.export`) into an advanced-indexing call with a `None`-prefix mixed
   with real indices of different rank, which the torch frontend's
   `_index_tensor` converter mishandled (its general-case fallback assumed a
   different broadcast/placement rule than PyTorch actually uses). Fixed
   with a permute-to-prefix / `index_tensor` / permute-back approach that
   matches NumPy's actual placement semantics.
3. **GRU recurrence unrolling into thousands of ops** — GTCRN's 14 `nn.GRU`
   instances (63 timesteps each) were unrolled into per-timestep Relax ops
   twice over: once by PyTorch's own default decomposition (before this
   frontend ever saw a GRU node), and again by this frontend's own
   `_gru`/`_gru_cell_unroll` converter. Both compounded into a single
   ~33,000-line generated dispatcher function, which is what made `cl7x -O3`
   take ~21-22 minutes. Fixed by (a) `gtcrn_c7x.py` passing a custom
   `decomp_table` to keep `aten.gru.input` opaque instead of letting PyTorch
   decompose it, and (b) a new `topi.nn.gru` (`te.extern` + a hand-written
   `tir.ir_builder` loop) that `_gru_cell_unroll` now calls into instead of
   unrolling — this produces a genuine `for` loop in the generated C instead
   of one call per timestep. `te.scan` (the mechanism `topi.nn.lstm` uses for
   the same problem) doesn't work here: `te.create_prim_func`, which
   `block_builder.call_te` goes through, explicitly rejects `te.ScanOp`. See
   [Performance](#performance) for the measured effect.

## Correctness methodology: audio-domain SNR, not raw spectrogram diff

`test_gtcrn_e2e.py` gates correctness on **audio-domain SNR** (ISTFT both
the DSP and PyTorch outputs, compare waveforms), not raw complex-spectrogram
max-diff. This was a deliberate finding, not a convenience: raw spectrogram
diff between C7x and PyTorch swings widely across different inputs (0.27 to
0.76 max abs observed, input-dependent) — presumably from GTCRN's internal
63-step GRU recurrence compounding small per-step precision differences
between the C7x vector ISA's transcendental (sigmoid/tanh) implementations
and PyTorch's CPU reference. Audio-domain SNR from the same runs was
consistently 39.8–48.0 dB (near-imperceptible) regardless of that raw-diff
swing, so it's the actual stable invariant worth testing. The `_MIN_SNR_DB`
threshold (20 dB) is a generous margin under that observed range: a real bug
is expected to produce a qualitatively worse, much lower SNR, not a
borderline one.

## Performance

Compile time has two very different components depending on mode:

| Stage | Time (before GRU-loop fix) | Time (after) | Applies to |
|-------|---------------------------|---------------|------------|
| TVM codegen (`relax.build`, Relax→TIR→C) | ~156 s | well under 15 s | Both modes |
| `g++` host-emulation build | seconds | seconds | `c7x_host` only |
| `cl7x -O3` native cross-compile | ~21–22 min | **~4.6 min** | `c7x_dload` only |

Before the fix, GTCRN's 14 GRU instances (63 steps each) were unrolled into
per-timestep Relax/TIR ops, producing a single ~1.64 MB / ~32,900-line
generated dispatcher function (`__tvm_ffi___vmtir__main` in `lib0.c`) — this
is what made `cl7x -O3` (a single-translation-unit compile of that one giant
function) take ~21-22 minutes. After the fix, the same 14 GRU instances
lower to 4 deduplicated kernel functions (same-shaped GRUs share one
generated kernel, called from multiple sites) each containing a genuine
`for` loop, and the dispatcher shrinks to ~148 KB / ~3,245 lines (the
dispatcher function itself: ~2,878 lines). Total `test_gtcrn_c7x_dload` wall
time (TVM codegen + cmake build + board deploy/run) dropped from ~24 min to
**~4:48**.

This is a one-time cost per model/shape, not a per-inference cost, and (as
measured below) doesn't affect correctness or runtime performance.

Inference itself (the number that matters for real-time deployment) is fast:
**537,984,515 cycles (≈538 ms @ 1 GHz)** per `T=63` (~0.99 s audio) chunk on
real AM67A hardware — real-time factor ≈0.54, statistically unchanged from
before the GRU-loop fix (≈542 ms) since it only changes *compile* time, not
the arithmetic the DSP actually executes.

## Known limitations

- The compiled model has a **static** input shape (`T=63` by default) — a
  different chunk duration requires recompiling with a different `T` passed
  to `export_and_bind`.
- `test_gtcrn_c7x_dload` requires a reachable AM67A board
  (`root@am67a` via SSH) and is marked `@pytest.mark.c7x_only`; it skips
  cleanly when `--dsp-mode=c7x_dload` isn't passed.
