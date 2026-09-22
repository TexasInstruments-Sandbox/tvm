# Quantized Model Sweep

End-to-end tests for INT8-quantized TorchVision and YOLO models (PT2E
`C7xMMAQuantizer`) on the TVM `c_static_lib` backend, with and without MMALIB
offload, on C7x host emulation and real hardware (AM67A and BeagleY-AI).
Located at `tests/ti-dsp-runtime/quantized/`.

## Running

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS

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
functions (hardcoded torchvision import, synthetic random input, PT2E via
`_pt2e_quantize`). `test_quantized_torchvision.py` instead cross-imports
`tests/cstatic/cl_torchvision.py` (model loading + correct per-model
preprocessing) and `pt2e-tests/pt2e_utils.py` (`e2e_quantize_and_import` /
`run_and_check`) directly, so it covers whatever TorchVision model
`cl_torchvision.py` can load without needing a dedicated function per model.

## `test_quantized_yolo.py` status

All 5 models pass on `c7x_host` and BeagleY-AI hardware (`--board
beagley-ai`); v5n/s and v8n/s were also previously verified on AM67A.

Two comparison strategies, by output shape -- follow whichever matches
for any future detection model:

- **YOLOv5n/s, YOLOv8n/s**: raw per-anchor tensor (`[1, 4+nc,
  num_anchors]`, no NMS), mixing small-magnitude box regression with
  bounded-range class scores in the same tensor -- an element-wise bound
  doesn't work here, so pass/fail uses cosine similarity against the
  PyTorch fake-quantized reference.
- **YOLO26n**: runs the real NMS-free "one2one" head, which does an
  internal `topk` and returns already-*selected* detections
  (`[1, 300, 6]`). A near-tied class score can legitimately make the DSP
  and the fake-quant reference pick a different anchor at the selection
  boundary -- both correct, but a different row at that index -- so this
  one uses greedy IoU+class matching (`_match_fraction`) against the
  reference set instead.

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

80 candidate TorchVision classification models. 14 are excluded outright
(never run) -- see the corresponding `_EXCLUDED_*` set in
`test_quantized_torchvision.py` for the exact model list:

| Reason | Count | Set |
|---|---|---|
| Int8 weight size alone exceeds the 256 MiB DLOAD DDR heap | 4 | `_EXCLUDED_WEIGHT_SIZE` |
| Runtime DDR pool exhaustion at a late layer (not weight size) | 8 | `_EXCLUDED_DDR_OOM` |
| TVM pass bug hit after quantization, not root-caused | 1 | `_EXCLUDED_QUANT_BUG` |
| Genuine MMALIB misclassification | 1 | `_EXCLUDED_MISCLASSIFY` |

Of the remaining 66, all pass via `c7x_dload --mmalib`: 65 on the
elementwise `max_diff<=25` bound, 1 (`squeezenet1_1`) via a top-1
classification match instead (see below).

### Notes on specific cases

- **`squeezenet1_1`** exceeds `max_diff<=25` (27) but still classifies
  correctly, so it's checked via top-1 match instead (`_TOP1_MATCH_ONLY`
  in `test_quantized_torchvision.py`) rather than loosening the bound
  for the whole sweep. `squeezenet1_0` (excluded) is not benign -- it
  misclassifies outright. Root cause for both: MMALIB's closed-source
  requantization kernel has no rounding-mode field, and SqueezeNet is
  the only model here with no BatchNorm to reset the resulting per-layer
  bias each block -- not fixable in our own code.
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
- **MMALIB conv2d/depthwise geometry limits**: three real MMALIB
  hardware kernel limits (asymmetric stride+padding, unpadded "VALID"
  stride-1 convs, and an MMA-panel-width/odd-feature-map limit on
  depthwise convs) are caught at compile time and declined to the
  scalar codegen path instead of aborting the DSP -- see
  `_check_conv2d_row_kernel_geometry` (`ti_mmalib_legalize.py`) and
  `_check_dwconv2d_geometry` (`ti_mmalib_qdq_dwconv.py`) for the exact
  rules, and `unit-tests/test_mmalib_qdq_geometry_decline.py` for
  regression coverage.
- **Native cl7x compile time** for `swin_s`/`swin_t`/`vit_b_16`/
  `vit_b_32` is slow -- up to ~12 min for `vit_b_16`, genuine `cg7x`
  codegen compute (not a hang) -- give these extra timeout headroom
  rather than treating "no output for N minutes" as stuck.

## Shared infrastructure

| Function | Location | Purpose |
|---|---|---|
| `_pt2e_quantize` | `model_utils.py` | Export → `prepare_pt2e` → calibrate (random noise or real images) → `convert_pt2e`; used by the 7 per-model files |
| `load_model_with_preprocessing`, `load_image`, `get_all_classification_models` | `tests/cstatic/cl_torchvision.py` | Dynamic model loading with correct per-model preprocessing; cross-imported by `test_quantized_torchvision.py` |
| `e2e_quantize_and_import`, `run_and_check` | `tests/ti-dsp-runtime/pt2e-tests/pt2e_utils.py` | Full quantize→import pipeline and MMALIB compile+run+assert; cross-imported by `test_quantized_torchvision.py` |

`run_and_check`'s default tolerance (`max_diff=2`, ±1 LSB) is calibrated
for single-op unit tests (see [PT2E Quantizer Suite](pt2e-suite.md)) —
whole models compound int8 rounding error across many layers, so
`test_quantized_torchvision.py` passes `max_diff=25` explicitly instead
of relying on the default.

## Prerequisites

- TVM built with the `c_static_lib` backend (`TVM_HOME` set, `PYTHONPATH`
  includes `python/`)
- `TI_CGT_C7000_PATH` for DSP tests
- For `c7x_dload`: firmware deployed on the target board (`deploy-c7x.sh
  --board <j722s-evm|beagley-ai>`), and the matching `pytest --board
  <j722s-evm|beagley-ai>` -- both required, no default
- `--mmalib` fixture/flag (from `conftest.py`) selects the MMALIB target;
  omitting it runs the generic (non-MMALIB) int8 codegen path instead
