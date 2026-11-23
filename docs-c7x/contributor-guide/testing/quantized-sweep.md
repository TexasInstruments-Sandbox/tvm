# Quantized Model Sweep

End-to-end tests for INT8-quantized TorchVision and YOLO models (PT2E
`C7xMMAQuantizer`) on the TVM `c_static_lib` backend, with and without MMALIB
offload, on C7x host emulation and real hardware (AM67A and BeagleY-AI).
Located at `tests/ti-dsp-runtime/quantized/`.

## Running

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=<path to TI C7000 CGT>

# One model, host emulation
pytest --rootdir=. quantized/test_quantized_resnet.py -v --dsp-mode=c7x_host --mmalib

# One model, AM67A hardware
pytest --rootdir=. quantized/test_quantized_resnet.py -v --dsp-mode=c7x_dload --board j722s-evm --mmalib

# One model, BeagleY-AI hardware
pytest --rootdir=. quantized/test_quantized_yolo.py \
    -v --dsp-mode=c7x_dload --board beagley-ai --mmalib -k yolo26n

# Full TorchVision classification sweep, one model
pytest --rootdir=. "quantized/test_quantized_torchvision.py::test_quantized_torchvision_dsp[resnet50]" \
    -v --dsp-mode=c7x_dload --board j722s-evm --mmalib

# Standalone script
python quantized/test_quantized_resnet.py --dsp-mode c7x_host --mmalib
```

`c7x_dload` tests talk to real DSP hardware and require `--board
<j722s-evm|beagley-ai>` (no default -- omitting it is an error, since the
codegen target and the SSH deploy host both depend on it): run them one at
a time, in the foreground, never in the background or concurrently (single
DSP core; conflicts hang the firmware and require a board reboot/power
cycle). BeagleY-AI's firmware has no TIDL kernels linked, so its `c_static_lib`
target string needs `-tidl-kernels=0`; `get_target_string()` in
`dsp-cpp/dsp_utils.py` adds this automatically whenever `--board
beagley-ai` is passed.

## Test files

| File | Model(s) | Status |
|------|----------|--------|
| `test_quantized_resnet.py` | ResNet-18 | PASS |
| `test_quantized_resnext101.py` | ResNeXt-101 (32x8d) | PASS |
| `test_quantized_googlenet.py` | GoogLeNet | PASS |
| `test_quantized_inception_v3.py` | InceptionV3 | PASS |
| `test_quantized_mobilenet_v2.py` | MobileNetV2 | PASS |
| `test_quantized_mobilenet_v3.py` | MobileNetV3-Large | PASS |
| `test_quantized_shufflenet_v2.py` | ShuffleNetV2 (x0.5) | PASS |
| `test_quantized_yolo.py` | YOLOv5n/s, YOLOv8n/s, YOLO26n (object detection) | PASS, all 5 (see below) |
| `test_quantized_torchvision.py` | All 80 TorchVision ImageNet classifiers, via `cl_torchvision.py`'s dynamic loader | see sweep below |

The first 7 use `model_utils.py`'s per-model `create_quantized_*_model`
functions (hardcoded torchvision import, PT2E via `_quantize_classifier`).
They calibrate on the 12 images in `quantized/calib_images/` and run the
DSP on `tests/cstatic/test_images/dog.jpg`, which is not in that set; both
go through the weights' own preprocessing. `test_quantized_torchvision.py`
instead cross-imports `tests/cstatic/cl_torchvision.py` (model loading +
correct per-model preprocessing) and `pt2e-tests/pt2e_utils.py`
(`e2e_quantize_and_import`) directly, so it covers
whatever TorchVision model `cl_torchvision.py` can load without needing a
dedicated function per model.

All of these run with MMALIB offload. Only `test_quantized_resnet.py` also
runs without `--mmalib`, as the one model-level check of the generic int8
path; the other classifier tests skip without it (`MMALIB_ONLY_REASON` in
`model_utils.py`), as YOLO already did. BeagleY-AI and J722S firmware always
link MMALIB, the generic int8 conv path is covered more cheaply by the
`dsp-tests` conv tests, and compiling it for a whole network is slow.

The 7 classifier tests compare the DSP output with the fake-quant
reference using `atol` = twice the largest difference measured on
`c7x_host` and BeagleY-AI, rounded up to a multiple of 0.25 (`_ATOL` in
each file, `rtol=0`), and also require the same top-1 class. Host and
board outputs are bit-identical for these models, so one value covers
both. Re-measure and update `_ATOL` when a change legitimately moves the
numbers (calibration set, quantizer, kernel rounding).

## `test_quantized_yolo.py` status

All 5 models pass on `c7x_host` and BeagleY-AI hardware (`--board
beagley-ai`); v5n/s and v8n/s were also previously verified on AM67A.
They calibrate on the 12 images in `quantized/calib_images/` and run on
`tests/cstatic/test_images/car.jpg` (a car and people, not in that set),
all resized to 320x320 without letterboxing.

Two comparison strategies, by output shape -- follow whichever matches
for any future detection model:

- **YOLOv5n/s, YOLOv8n/s**: raw per-anchor tensor (`[1, 4+nc,
  num_anchors]`, no NMS), mixing small-magnitude box regression with
  bounded-range class scores in the same tensor -- an element-wise bound
  doesn't work here, so pass/fail uses cosine similarity against the
  PyTorch fake-quantized reference (`>= 0.998`; the four models measure
  0.9993-0.9996).
- **YOLO26n**: runs the real NMS-free "one2one" head, which does an
  internal `topk` and returns already-*selected* detections
  (`[1, 300, 6]`). A near-tied class score can legitimately make the DSP
  and the fake-quant reference pick a different anchor at the selection
  boundary -- both correct, but a different row at that index -- so this
  one uses greedy IoU+class matching (`_match_fraction`) against the
  reference set instead (`>= 0.60`; it measures 0.82).

Getting YOLO26n running required a DSP `topk` kernel
(`src/runtime/ti_dsp/kernels/c7x_topk.cpp`, `relax.topk` has no other
DSP-compilable lowering) plus PT2E frontend/quantizer fixes for models
with `topk`-heavy postprocessing -- see git history around the YOLO26
bring-up if a future model hits the same class of issue. Bringing this
up on BeagleY-AI also caught a model-agnostic MMALIB bug: PT2E's
`reshape(Constant)` conv bias wasn't being resolved, so every
MMALIB-offloaded conv in this whole suite was silently running with
zero bias -- fixed via `_resolve_constant_tensor` in
`ti_mmalib_legalize.py`.

## `test_quantized_torchvision.py` sweep status

80 candidate TorchVision classification models. 13 are excluded outright
(never run) -- see the corresponding `_EXCLUDED_*` set in
`test_quantized_torchvision.py` for the exact model list:

| Reason | Count | Set |
|---|---|---|
| Int8 weight size alone exceeds the 256 MiB DLOAD DDR heap | 4 | `_EXCLUDED_WEIGHT_SIZE` |
| Runtime DDR pool exhaustion at a late layer (not weight size) | 8 | `_EXCLUDED_DDR_OOM` |
| TVM pass bug hit after quantization, not root-caused | 1 | `_EXCLUDED_QUANT_BUG` |

All of the remaining 67 pass on `c7x_dload` with MMALIB on BeagleY-AI (full
sweep 2026-11-23, firmware service 5.2.0, firmware and modules built with cl7x
6.2.1.LTS at commit 4ea4a66fb6, 8-bit MMALIB conv requant multipliers). Their
accuracy and cycles are in [Results on BeagleY-AI](#results-on-beagley-ai).

Each model must give the reference's top-1 class, or a class the reference
scores within `_TOP1_TIE` (0.25) of it, and, with MMALIB, stay within
`_MAX_ABS_DIFF` (8.5) of its logits. The tie allowance is for references that
can't separate two classes themselves: on `dog.jpg` the references of
efficientnet_v2_s, _m and _l score classes 258 and 259 equal and vgg13_bn's
0.087 apart, and with the 8-bit multiplier efficientnet_v2_s and vgg13_bn pick
the other class at logit cosines of 0.9975 and 0.9996. The logits are compared as floats. The
bound is set like the per-model tests' `_ATOL`: twice the largest difference of
a passing model outside `_KNOWN_FAILURES` on `c7x_host` and BeagleY-AI, rounded
up to a multiple of 0.25.
It catches a miscompile that leaves top-1 intact; it doesn't measure precision.
For that, each run adds or replaces the model's row in
`results/sweep_metrics.csv` (under `$WORKSPACE`, default the working directory,
beside `cycles.csv`):

| Column | Meaning |
|---|---|
| `top1_match`, `ref_top1`, `dsp_top1` | Top-1 class of the reference and the DSP |
| `ref_margin` | The reference's top-1 logit minus its second: a small margin marks a near-tie |
| `dsp_top1_gap` | The reference's top-1 logit minus its logit for the DSP's top-1 (0 when they match); at most `_TOP1_TIE` to pass |
| `max_abs_diff`, `cosine` | Of the DSP and reference logits |
| `cycles` | DSP cycles, on hardware only |

### Known failures

`_KNOWN_FAILURES` lists the models that fail the check, or pass it only because
the reference's top-1 margin exceeds a known DSP error, the modes they do so in,
and what was measured. In those modes a failure is reported as xfail with that
reason; the model still runs and records its metrics, and a pass is a pass. Remove
an entry once its model's logits match the reference.

There are none at present. `alexnet` was listed until 2026-11-18: on firmware
built with cl7x 5.0.1, MMALIB's `matrixMatrixMultiplyBias` returned mostly
zeros for FC1 (9216x4096) unless another `matmulBias` call ran just before it
(top-1 331 vs 258, cosine 0.75; `c7x_host` matched). It doesn't reproduce on
firmware built with 6.2.1.LTS, which doesn't show that the cause is gone; if
`alexnet` fails on the board again, check FC1 first.

### Notes on specific cases

- **`_EXCLUDED_DDR_OOM` (8 models)** all fail identically
  (`c7x: INFER failed: status=-11` / `Function call failed`) -- despite
  looking segfault-like, this is genuine DDR pool (`DDR_C7X_1_LOCAL_HEAP`,
  352 MiB) exhaustion at a late layer, confirmed via the allocator in
  `platform/common/memory_pool.c` (not a leak). Fixing needs an MMU/heap
  extension (the 16-region-descriptor cap is already full) and carries
  real DSP-hang risk from a bad region change; excluded rather than
  chased further. If any of these are ever re-enabled, note that
  `swin_v2_*` additionally needs `ConstReachability`
  (`ti_c7x_const_reachability.py`) for an unrelated compile-time
  segfault in constant-folding, already fixed but worth knowing about.
- **MMALIB conv2d/depthwise geometry limits**: four real MMALIB
  hardware kernel limits (asymmetric stride+padding, unpadded "VALID"
  stride-1 convs, regular convs at stride above 2, and an
  MMA-panel-width/odd-feature-map limit on depthwise convs) are caught
  at compile time and declined to the scalar codegen path instead of
  aborting the DSP or, at stride 4 and above, computing wrong values
  for ConvNeXt's, Swin's and ViT's patch convs -- see
  `_check_conv2d_mmalib_constraints` and `_check_conv2d_row_kernel_geometry`
  (`ti_mmalib_legalize.py`) and `_check_dwconv2d_geometry`
  (`ti_mmalib_qdq_dwconv.py`) for the exact rules, and
  `unit-tests/test_mmalib_qdq_geometry_decline.py` and
  `mmalib-tests/test_mmalib_conv2d_stride_dsp.py` for regression coverage.
- **Native cl7x compile time** for `swin_s`/`swin_t`/`vit_b_16`/
  `vit_b_32` is slow -- up to ~12 min for `vit_b_16`, genuine `cg7x`
  codegen compute (not a hang) -- give these extra timeout headroom
  rather than treating "no output for N minutes" as stuck.

## Shared infrastructure

| Function | Location | Purpose |
|---|---|---|
| `QuantRecipe`, `quantize_with_recipe` | `tests/ti-dsp-runtime/dsp-cpp/quant_recipe.py` | Named quantizer settings and calibration source, and the shared export → `prepare_pt2e` → calibrate → `convert_pt2e` step. `BASELINE_CALIB_IMAGES` (the 12 images in `calib_images/`) is used by the 7 per-model files and the YOLO test; `BASELINE_TEST_IMAGE` (the test input) by `test_quantized_torchvision.py` and the pt2e MobileNetV2 test. Recipes calibrate only on images in the repo; datasets used to measure accuracy stay out of them |
| `_pt2e_quantize` | `model_utils.py` | Export → calibrate with a `QuantRecipe` → import into TVM; used by the 7 per-model files and the YOLO test |
| `_quantize_classifier` | `model_utils.py` | Calibrates a TorchVision classifier on `calib_images/` and returns the held-out `dog.jpg` as its input; used by the 7 per-model files |
| `load_model_with_preprocessing`, `load_image`, `get_all_classification_models` | `tests/cstatic/cl_torchvision.py` | Dynamic model loading with correct per-model preprocessing; cross-imported by `test_quantized_torchvision.py` |
| `e2e_quantize_and_import` | `tests/ti-dsp-runtime/pt2e-tests/pt2e_utils.py` | Full quantize→import pipeline; cross-imported by `test_quantized_torchvision.py` |

## Measuring accuracy

The tests above compare the DSP with PyTorch on one image, which says nothing
about accuracy. `tests/ti-dsp-runtime/accuracy/evaluate.py` measures what
quantization costs: it runs the float model and the Q/DQ graph from
`convert_pt2e` on the same images, in PyTorch only, with no compile or board.

```bash
cd tests/ti-dsp-runtime
C7X_DATASETS_DIR=~/datasets python accuracy/evaluate.py resnet18 --n 5000 --out report.json
```

It needs ImageNetV2 MatchedFrequency under `C7X_DATASETS_DIR`, in the layout
described in `accuracy/eval_datasets.py`. Nothing downloads it, and no data or
file list belongs in this repository. Its images have the same terms of use as
ImageNet's.

- `top1_delta` is quantized minus float on the same images, and
  `top1_delta_stderr` is its standard error. Read the delta rather than the
  absolute accuracy, which isn't comparable with published ImageNet numbers.
- `top1_tied_quant` is how often the largest quantized logits tie. Ties break by
  lowest class index, as `np.argmax` does.
- `--recipe` picks a named `QuantRecipe`. `--calibrate-on-dataset N` calibrates
  on N held-out dataset images instead, as an analysis, not a recipe.

`accuracy/evaluate_board.py` measures the same on the board. It quantizes as
`evaluate.py` does, compiles as the sweep does, and runs the model on the first
`--n` images of the same split in one `C7xVirtualMachine` session, so the module
loads once:

```bash
cd tests/ti-dsp-runtime
C7X_DATASETS_DIR=~/datasets python accuracy/evaluate_board.py resnet18 \
    --board beagley-ai --n 200 --work /tmp/acc --out board.json
```

The report's `quant` and `board` hold `evaluate.py`'s metrics for the Q/DQ graph
and for the board, each against float, and `board_vs_quant` how closely the
board follows the Q/DQ graph. `--stage prepare` compiles and saves the inputs and
host logits under `--work`, and `--stage run` uses the board, so one model can
compile while the board runs another. Never run two `--stage run` at once on
one board.

Per-layer error and MMALIB coverage aren't measured yet. Its unit tests
(`accuracy/test_accuracy_harness.py`) need no dataset and aren't in a CI stage yet.

## Results on BeagleY-AI

Measured 2026-11-22 and 23 with firmware and modules built at commit 4ea4a66fb6
(cl7x 6.2.1.LTS, MMALIB, `--board beagley-ai`). Each model is quantized with
the `baseline-calib-images` recipe, as in the per-model tests, and measured on
ImageNetV2 MatchedFrequency. Read the changes against float, not the absolute
accuracies (see [Measuring accuracy](#measuring-accuracy)).

| Column | Meaning |
|---|---|
| Mcycles | Median DSP cycles per inference over the 200 board images, from `evaluate_board.py`. A single sweep run's `cycles.csv` agrees within 1% for the median model and differs by up to 12% for a few small ones, such as regnet_x_3_2gf and mnasnet0_75 |
| Float top-1 % | The float model in PyTorch on 5,000 images, or 2,000 where Δtop-1 is marked † |
| Δtop-1, Δtop-5 | Fake-quant minus float on the same images, in percentage points, from `evaluate.py`. ± is the standard error of Δtop-1 |
| Board Δtop-1 | The board minus float on the first 200 of those images, from `evaluate_board.py`, and in parentheses fake-quant on the same 200. One image is 0.5 pp and the standard error about 1 pp, so compare the board with the value in parentheses, not with the Δtop-1 column |
| Board = fake-quant top-1 % | How often the board and the Q/DQ graph pick the same top-1 class on those 200 images |
| `dog.jpg` cosine | The sweep's check: the cosine of the DSP and fake-quant logits on `dog.jpg`. The sweep calibrates on `dog.jpg` itself, not on `calib_images/` |

Top-1 is the measure to read. Top-5 changes by no more than top-1 for 45 of
the 67 models and by more than 0.5 pp beyond it for 4 (densenet121,
regnet_y_1_6gf, vit_b_16 and wide_resnet50_2), so it rarely changes the
picture.

| Model | Mcycles | Float top-1 % | Δtop-1 pp | Δtop-5 pp | Board Δtop-1 pp (fake-quant) | Board = fake-quant top-1 % | `dog.jpg` cosine |
|---|--:|--:|--:|--:|--:|--:|--:|
| alexnet | 369 | 42.8 | −0.4 ± 0.2 | +0.1 | −0.5 (−1.0) | 94.5 | 0.9999 |
| convnext_small | 14,872 | 73.3 | −2.1 ± 0.5 † | −0.9 | −3.0 (−2.5) | 98.0 | 0.9972 |
| convnext_tiny | 8,174 | 70.7 | −0.5 ± 0.3 | −0.4 | +0.5 (+0.5) | 96.0 | 0.9941 |
| densenet121 | 1,715 | 60.6 | −0.3 ± 0.3 | −0.9 | +0.5 (+0.0) | 96.0 | 0.9983 |
| densenet161 | 3,592 | 64.2 | −0.4 ± 0.3 | −0.2 | −0.5 (+0.5) | 96.5 | 0.9977 |
| densenet169 | 2,187 | 62.2 | −0.5 ± 0.3 | −0.8 | −1.0 (−1.5) | 92.5 | 0.9979 |
| densenet201 | 3,210 | 63.9 | −0.9 ± 0.3 | −0.8 | −3.5 (−2.0) | 93.0 | 0.9984 |
| efficientnet_b0 | 204 | 64.7 | −3.0 ± 0.4 | −2.0 | −1.0 (−3.0) | 92.0 | 0.9977 |
| efficientnet_b1 | 315 | 67.4 | −3.6 ± 0.4 | −2.8 | −4.5 (−4.0) | 86.0 | 0.9940 |
| efficientnet_b2 | 435 | 68.3 | −2.1 ± 0.3 | −1.1 | −3.5 (−3.5) | 97.0 | 0.9984 |
| efficientnet_b3 | 751 | 70.2 | −1.4 ± 0.3 | −1.2 | +0.0 (−1.0) | 97.0 | 0.9993 |
| efficientnet_b4 | 1,445 | 72.6 | −3.4 ± 0.4 | −1.4 | −5.0 (−4.5) | 96.5 | 0.9969 |
| efficientnet_b5 | 2,306 | 72.5 | −4.4 ± 0.6 † | −2.9 | −3.0 (−3.5) | 97.5 | 0.9994 |
| efficientnet_v2_l | 6,435 | 75.5 | −3.8 ± 0.6 † | −2.5 | −3.5 (−3.5) | 99.5 | 0.9987 |
| efficientnet_v2_m | 2,241 | 74.0 | −1.7 ± 0.5 † | −1.3 | −3.0 (−2.0) | 97.0 | 0.9977 |
| efficientnet_v2_s | 779 | 73.0 | −1.9 ± 0.5 † | −0.6 | −2.5 (−2.0) | 95.5 | 0.9981 |
| googlenet | 174 | 56.7 | −0.3 ± 0.2 | −0.5 | −0.5 (−0.5) | 93.5 | 0.9983 |
| inception_v3 | 2,203 | 64.8 | +0.0 ± 0.3 | −0.1 | −1.5 (−0.5) | 95.5 | 0.9969 |
| mnasnet0_5 | 37 | 53.1 | −1.6 ± 0.4 | −1.8 | +0.5 (−2.0) | 89.0 | 0.9893 |
| mnasnet0_75 | 54 | 57.5 | −0.4 ± 0.3 | −0.9 | +0.5 (−0.5) | 92.5 | 0.9950 |
| mnasnet1_0 | 193 | 59.8 | −1.2 ± 0.3 | −0.7 | +0.5 (+1.0) | 94.0 | 0.9986 |
| mnasnet1_3 | 92 | 63.6 | −1.1 ± 0.3 | −0.5 | −1.0 (−2.0) | 94.0 | 0.9961 |
| mobilenet_v2 | 50 | 59.0 | −0.4 ± 0.3 | −0.1 | −0.5 (−1.5) | 95.5 | 0.9989 |
| mobilenet_v3_large | 61 | 60.8 | −3.8 ± 0.4 | −2.6 | −3.0 (−2.5) | 93.5 | 0.9927 |
| mobilenet_v3_small | 52 | 53.8 | −6.1 ± 0.5 | −5.4 | −5.5 (−8.5) | 79.5 | 0.9956 |
| regnet_x_16gf | 274 | 71.9 | +0.1 ± 0.3 † | +0.1 | +1.5 (+0.0) | 96.5 | 0.9956 |
| regnet_x_1_6gf | 81 | 66.8 | −0.0 ± 0.3 | −0.4 | −2.5 (−1.5) | 92.5 | 0.9952 |
| regnet_x_32gf | 509 | 71.7 | −0.5 ± 0.4 † | +0.2 | −0.5 (+0.0) | 98.0 | 0.9986 |
| regnet_x_3_2gf | 64 | 68.7 | −0.5 ± 0.3 | −0.1 | +0.0 (−0.5) | 93.0 | 0.9923 |
| regnet_x_400mf | 17 | 61.3 | −1.0 ± 0.4 | −1.2 | −3.5 (−3.0) | 93.5 | 0.9903 |
| regnet_x_800mf | 24 | 64.6 | −0.1 ± 0.3 | −0.1 | −0.5 (+0.5) | 90.0 | 0.9927 |
| regnet_x_8gf | 124 | 70.0 | −0.1 ± 0.4 † | −0.2 | +1.5 (+1.0) | 95.0 | 0.9949 |
| regnet_y_16gf | 517 | 71.5 | −4.4 ± 0.8 † | −3.2 | −8.0 (−8.0) | 88.0 | 0.9968 |
| regnet_y_1_6gf | 173 | 68.7 | +0.1 ± 0.3 | −0.5 | +2.0 (+1.5) | 93.0 | 0.9939 |
| regnet_y_32gf | 959 | 72.2 | +0.1 ± 0.4 † | −0.0 | −1.5 (−1.0) | 95.5 | 0.9983 |
| regnet_y_3_2gf | 193 | 70.5 | −1.5 ± 0.3 | −0.4 | +0.0 (+0.5) | 94.5 | 0.9966 |
| regnet_y_400mf | 79 | 62.7 | −1.6 ± 0.3 | −1.2 | −3.0 (−2.5) | 86.5 | 0.9939 |
| regnet_y_800mf | 79 | 66.0 | −1.0 ± 0.3 | −0.8 | +0.0 (+2.0) | 90.5 | 0.9907 |
| regnet_y_8gf | 353 | 72.5 | −1.3 ± 0.5 † | −0.4 | −3.0 (−3.5) | 94.0 | 0.9942 |
| resnet101 | 97 | 69.3 | −0.4 ± 0.3 | −0.5 | +2.0 (+0.0) | 94.5 | 0.9971 |
| resnet152 | 131 | 71.5 | −0.5 ± 0.4 † | −0.3 | −2.0 (−1.0) | 94.5 | 0.9934 |
| resnet18 | 18 | 55.6 | +0.0 ± 0.2 | +0.0 | −0.5 (−0.5) | 95.5 | 0.9993 |
| resnet34 | 28 | 59.8 | −0.5 ± 0.3 | +0.1 | +0.0 (−2.0) | 94.0 | 0.9988 |
| resnet50 | 66 | 69.1 | −0.7 ± 0.3 | −0.5 | +0.5 (+0.5) | 95.0 | 0.9966 |
| resnext101_32x8d | 233 | 72.0 | +0.3 ± 0.3 † | +0.3 | +1.0 (+1.0) | 96.0 | 0.9956 |
| resnext101_64x4d | 222 | 72.4 | −0.3 ± 0.4 † | −0.1 | +3.0 (+1.5) | 97.0 | 0.9968 |
| resnext50_32x4d | 91 | 68.4 | −0.2 ± 0.3 | −0.2 | −0.5 (+0.0) | 95.5 | 0.9975 |
| shufflenet_v2_x0_5 | 37 | 46.8 | −1.1 ± 0.3 | −0.8 | −1.0 (−2.0) | 82.5 | 0.9938 |
| shufflenet_v2_x1_0 | 155 | 55.0 | −0.8 ± 0.3 | −0.6 | −1.0 (−0.5) | 93.0 | 0.9952 |
| shufflenet_v2_x1_5 | 139 | 58.9 | −1.2 ± 0.4 | −0.8 | −2.5 (−3.0) | 95.0 | 0.9954 |
| shufflenet_v2_x2_0 | 236 | 63.1 | −1.4 ± 0.4 | −0.7 | −2.0 (−1.0) | 92.5 | 0.9948 |
| squeezenet1_0 | 80 | 43.8 | −0.6 ± 0.3 | −0.4 | −2.0 (−1.5) | 91.5 | 0.9999 |
| squeezenet1_1 | 50 | 44.5 | −0.9 ± 0.3 | +0.2 | −2.0 (−2.0) | 93.0 | 0.9999 |
| swin_s | 13,966 | 70.6 | −1.4 ± 0.5 † | −1.7 | −3.5 (−2.5) | 93.0 | 0.9841 |
| swin_t | 7,643 | 68.8 | −2.2 ± 0.4 | −1.7 | −5.0 (−4.5) | 90.5 | 0.9741 |
| vgg11 | 180 | 54.6 | −0.1 ± 0.2 | −0.1 | −1.5 (+0.0) | 96.0 | 0.9999 |
| vgg11_bn | 180 | 56.0 | −0.6 ± 0.3 | −0.5 | +1.0 (+1.0) | 92.5 | 0.9997 |
| vgg13 | 248 | 56.0 | +0.1 ± 0.3 † | +0.3 | −0.5 (+0.5) | 95.5 | 0.9999 |
| vgg13_bn | 248 | 57.6 | +0.1 ± 0.3 † | −0.4 | +2.5 (+1.0) | 96.0 | 0.9998 |
| vgg16 | 312 | 57.7 | −0.5 ± 0.3 † | +0.0 | −1.0 (−1.5) | 98.0 | 0.9998 |
| vgg16_bn | 312 | 58.7 | −0.4 ± 0.3 † | +0.1 | −1.0 (−0.5) | 96.5 | 0.9997 |
| vgg19 | 373 | 58.4 | +0.2 ± 0.3 † | +0.2 | −0.5 (−0.5) | 97.5 | 0.9999 |
| vgg19_bn | 372 | 61.4 | −0.9 ± 0.3 † | −0.3 | −0.5 (−0.5) | 99.0 | 0.9997 |
| vit_b_16 | 14,806 | 69.0 | −1.9 ± 0.6 † | −2.9 | +0.5 (−2.0) | 92.0 | 0.9886 |
| vit_b_32 | 3,345 | 62.6 | −0.6 ± 0.3 | −0.1 | −0.5 (−1.0) | 89.0 | 0.9831 |
| wide_resnet101_2 | 246 | 71.5 | −1.5 ± 0.5 † | −0.7 | −1.5 (−0.5) | 96.5 | 0.9976 |
| wide_resnet50_2 | 182 | 69.3 | −0.5 ± 0.5 † | −1.1 | −1.5 (−0.5) | 95.5 | 0.9973 |

Of the 67, 40 lose at most 1 pp of top-1, and 7 lose 3 pp or more:
mobilenet_v3_small (6.1), efficientnet_b5 and regnet_y_16gf (4.4),
mobilenet_v3_large and efficientnet_v2_l (3.8), efficientnet_b1 (3.6) and
efficientnet_b4 (3.4). Those losses are in the fake-quant graph on the host, so
they come from quantization, not from the board. On the same 200 images the
board's change is within 3 pp of fake-quant's for every model and within 1.5 pp
for all but 7. Its top-1 agrees with fake-quant's on 79.5-99.5% of the images,
94.5% for the median model; the lowest are mobilenet_v3_small (79.5%),
shufflenet_v2_x0_5 (82.5%) and efficientnet_b1 (86.0%).

mnasnet1_0 takes twice mnasnet1_3's cycles, though it is the smaller model.
That hasn't been explained.

The per-model tests ran in the same queue. Their cycles are single runs from
`cycles.csv`, beside the sweep's single run of the same TorchVision model; the
classifiers' accuracy is in the table above. YOLO accuracy isn't measured.

| Test | Mcycles | Same model in the sweep |
|---|--:|--:|
| `test_quantized_resnet.py` | 18 | 18 |
| `test_quantized_resnext101.py` | 233 | 233 |
| `test_quantized_googlenet.py` | 173 | 175 |
| `test_quantized_inception_v3.py` | 2,196 | 2,198 |
| `test_quantized_mobilenet_v2.py` | 52 | 50 |
| `test_quantized_mobilenet_v3.py` | 59 | 59 |
| `test_quantized_shufflenet_v2.py` | 37 | 37 |
| `test_quantized_yolo.py[yolov5n]` | 324 | — |
| `test_quantized_yolo.py[yolov5s]` | 457 | — |
| `test_quantized_yolo.py[yolov8n]` | 259 | — |
| `test_quantized_yolo.py[yolov8s]` | 452 | — |
| `test_quantized_yolo.py[yolo26n]` | 279 | — |

## Prerequisites

- TVM built with the `c_static_lib` backend (`TVM_HOME` set, `PYTHONPATH`
  includes `python/`)
- `TI_CGT_C7000_PATH` for DSP tests
- For `c7x_dload`: firmware deployed on the target board (`deploy-c7x.sh
  --board <j722s-evm|beagley-ai>`), and the matching `pytest --board
  <j722s-evm|beagley-ai>` -- both required, no default
- `--mmalib` fixture/flag (from `conftest.py`) selects the MMALIB target;
  omitting it runs the generic (non-MMALIB) int8 codegen path instead
