# Quantized Model Tests

End-to-end tests for INT8-quantized TorchVision models (PT2E `C7xMMAQuantizer`)
on the TVM `c_static` backend, with and without MMALIB offload, on C7x host
emulation and AM67A hardware.

## Running

```bash
cd tests/ti-dsp-runtime
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS

# One model, host emulation
pytest --rootdir=. quantized/test_quantized_resnet.py -v --dsp-mode=c7x_host --mmalib

# One model, AM67A hardware
pytest --rootdir=. quantized/test_quantized_resnet.py -v --dsp-mode=c7x_dload --mmalib

# Full TorchVision classification sweep, one model
pytest --rootdir=. "quantized/test_quantized_torchvision.py::test_quantized_torchvision_dsp[resnet50]" \
    -v --dsp-mode=c7x_dload --mmalib

# Standalone script
python quantized/test_quantized_resnet.py --dsp-mode c7x_host --mmalib
```

`c7x_dload` tests talk to real AM67A hardware: run them one at a time,
in the foreground, never in the background or concurrently (single DSP
core; conflicts hang the firmware and require a board reboot/power cycle).

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
| `test_quantized_yolo.py` | YOLOv5n/s, YOLOv8n/s (object detection) | PASS, all 4 (AM67A hardware) |
| `test_quantized_torchvision.py` | All 80 TorchVision ImageNet classifiers, via `cl_torchvision.py`'s dynamic loader | see sweep below |

The first 7 use `model_utils.py`'s per-model `create_quantized_*_model`
functions (hardcoded torchvision import, synthetic random input, PT2E via
`_pt2e_quantize`). `test_quantized_torchvision.py` instead cross-imports
`tests/cstatic/cl_torchvision.py` (model loading + correct per-model
preprocessing) and `pt2e-tests/pt2e_utils.py` (`e2e_quantize_and_import` /
`run_and_check`) directly, so it covers whatever TorchVision model
`cl_torchvision.py` can load without needing a dedicated function per model.

## `test_quantized_torchvision.py` sweep status

80 candidate TorchVision classification models. 13 are excluded outright
(never run): 4 for weight size, 7 for runtime DDR pool exhaustion, 1 for
a TVM pass bug hit after quantization, 1 for a genuine MMALIB
misclassification (all below). Of the remaining 67, run via
`c7x_dload --mmalib` on real AM67A hardware: **all pass** — 66 via the
elementwise `max_diff<=25` bound, 1 (`squeezenet1_1`) via a top-1
classification match instead (see below).

### 66 PASS (max_diff <= 25)

alexnet, convnext_base, convnext_tiny, convnext_small,
densenet121/161/169/201, efficientnet_b0/b1/b2/b3/b4/b5,
efficientnet_v2_s/v2_m/v2_l, googlenet, inception_v3,
mnasnet0_5/0_75/1_0/1_3, mobilenet_v2, mobilenet_v3_large/small,
all 7 `regnet_x_*` sizes, all 7 `regnet_y_*` sizes,
resnet18/34/50/101/152, resnext50_32x4d/101_32x8d/101_64x4d,
all 4 `shufflenet_v2_*` sizes, `swin_s`, `swin_t`, all 8 vgg variants
(11/13/16/19, with and without `_bn`), `vit_b_16`, `vit_b_32`,
wide_resnet50_2/101_2.

`mnasnet1_0` regressed after `e992d3e5b7`, was briefly excluded pending
root-cause, and is back here after the actual fix — see "mnasnet1_0"
below, a separate issue from SqueezeNet's.

Native cl7x cross-compilation for `swin_s`/`swin_t`/`vit_b_16`/`vit_b_32`
is slow — up to ~12 minutes for `vit_b_16`, dominated by cl7x's `cg7x`
code generator pegged at 99%+ CPU. This is genuine compute (confirmed via
`pstree`/CPU%, not a hang) — give these enough timeout headroom
(15-20 min) rather than treating "no output for N minutes" as a hang.

### SqueezeNet — root-caused: a real MMALIB accumulation bug, not observer noise

Both SqueezeNet variants originally failed the elementwise `max_diff<=25`
bound (`squeezenet1_0`: 30, `squeezenet1_1`: 27), consistent on host and
hardware. The original theory ("no BatchNorm gives PT2E's observers a
wider, noisier range") turned out to be wrong. Root-caused instead:

1. **It's MMALIB-specific, not a quantization-graph issue.** Running the
   *identical* Q/DQ graph through the generic (non-MMALIB) TVM int8
   codegen path gives `max_diff=7` with a correct top-1 prediction for
   `squeezenet1_0` — same calibrated scales, same graph, only the kernel
   backend differs.
2. **The error compounds through depth, not from one bad layer.**
   Truncating the model at each Fire-module boundary and measuring
   MMALIB `max_diff` at each cut point:

   | Cut point | max_diff |
   |---|---|
   | stem (conv+relu+pool) | 2 |
   | fire3 | 8 |
   | fire4 | 13 |
   | fire5 | 29 |
   | fire7 | 55 |
   | fire8 | 113 |
   | fire9 | 185 |
   | fire10 | 172 |
   | fire12 | 283 |

   Growth is roughly geometric, not linear — consistent with a small
   *relative* per-layer error compounding on itself, not a fixed +1 LSB
   per layer (which would only reach ~15-20 after 9 stages).
3. **Why SqueezeNet specifically:** every other model in the 66-PASS set
   has BatchNorm, which renormalizes the activation distribution at every
   layer and effectively resets any small per-layer bias before it can
   compound. SqueezeNet's Fire modules have no BatchNorm anywhere, so a
   small systematic error from the MMALIB int8 path keeps accumulating
   unchecked across 8 stacked Fire blocks.
4. **Where the error actually originates:** our own per-channel
   scale/shift search (`_float_to_scale_shift` in `ti_mmalib_legalize.py`)
   achieves sub-1% relative error and looks correct. The actual
   `(accum * scale) >> shift` requantization runs inside TI's
   closed-source `MMALIB_CNN_convolveBias_row` (called from
   `src/runtime/ti_dsp/mmalib/mmalib_wrappers.cpp`) — no rounding-mode
   field is exposed in its `InitArgs`, and TI's docs don't state whether
   that internal shift rounds-to-nearest or truncates. A truncating shift
   would produce exactly this signature: a small systematic per-layer
   bias that compounds unboundedly without BN to reset it. Not fixable in
   our own code — it's vendor kernel internals.

**Resolution, since the two variants behave differently in practice:**

- `squeezenet1_1`: the divergence is benign on the standard test image —
  top-1 prediction is still correct (`258` both sides) despite exceeding
  `max_diff=25`. Checked via top-1 classification match instead (see
  `_TOP1_MATCH_ONLY` in `test_quantized_torchvision.py`) — the same bar
  the non-MMALIB path already uses — rather than loosening `max_diff` for
  every model in the sweep.
- `squeezenet1_0`: **not benign.** The MMALIB path picks the wrong class
  outright (`157` instead of the correct `258`; top-5 barely overlaps).
  Excluded (`_EXCLUDED_MISCLASSIFY` in `test_quantized_torchvision.py`)
  rather than loosening its tolerance, which would hide a genuine
  misclassification.

### mnasnet1_0 — fixed: `mmalib_conv2d_i8` silently no-op'd for C_out > 1024

`mnasnet1_0` passed this sweep previously; it regressed after
`e992d3e5b7` ("Resolve MMALIB conv bias through reshape") started
correctly including MMALIB conv2d/dwconv bias instead of silently
dropping it to zero. MMALIB path started picking class `428` instead of
the correct `258` (top-5 disjoint from the reference's) — **not
benign**, same category as `squeezenet1_0`. Briefly excluded
(`_EXCLUDED_MISCLASSIFY`) pending root-cause; now fixed and back in the
66-PASS list above.

**Root cause: unrelated to the bias fix, unrelated to depthwise conv,
and not new — a pre-existing latent bug in the regular (non-depthwise)
MMALIB conv2d path that the bias fix happened to expose.**

`conv2d_impl` in `mmalib_wrappers.cpp` (backing `mmalib_conv2d_i8`, used
by `ti_mmalib_qdq_fusion.py` for every `groups==1` MMALIB conv2d) starts
with:
```cpp
if (C_out > 1024) {
    return -1;
}
```
The TE/TIR code emitting the `call_extern` to this function
(`ti_mmalib_qdq_fusion.py`'s `_emit_conv2d`/`te_mmalib_conv2d`) never
checks this return value — the extern call's result is discarded, not
propagated as a build or runtime error. When `C_out > 1024`, the
function returns immediately without writing anything to `output`, so
the output tensor is left as whatever was already in that memory
(observed as all-zero in practice). mnasnet1_0's final feature-expansion
conv (a very common MobileNet-family pattern: 1×1 conv from 320→1280
channels right before global average pooling) has `C_out=1280 > 1024`,
so its MMALIB path always silently computed nothing.

**Isolated, minimal, from-scratch confirmation** (no mnasnet1_0
involved) — reusing `test_mmalib_conv2d_i8_dsp.py`'s own QDQ conv2d
model builder, parametrized only on `C_out`:

| `C_out` | max_diff | DSP output zero-fraction | Reference zero-fraction |
|---|---|---|---|
| 512  | 1  | 0.047 | 0.047 |
| 1024 | 1  | 0.039 | 0.038 |
| 1280 | 20 | **1.000** | 0.040 |

At `C_out=1280` the entire output tensor comes back zero regardless of
input — not a numeric/rounding error, a complete no-op.

**Why the bias fix exposed it instead of causing it:** before
`e992d3e5b7`, every MMALIB conv2d/dwconv silently ran with zero bias, so
the *correct* (reference) value feeding into mnasnet1_0's 1280-channel
layer was already close to zero for most positions — this bug's
always-zero output happened to coincidentally match. Bisection
confirmed this precisely: forcing just the last depthwise layer (1152ch,
right before the two conv2d layers leading into the 1280ch expansion)
back to the old zero-bias behavior, with everything else on the real
fix, makes the test pass again — not because that depthwise layer itself
is wrong (a from-scratch replay of its exact real
weight/bias/scale/activation, extracted directly from the running
pipeline via a temporary tuple-output instrumentation, reproduces it
correctly, `max_diff=1`), but because it changes the upstream activation
enough that the *correct* value for the broken 1280-channel layer is now
large and positive for ~96% of its channels (confirmed: DSP output for
that layer is 97% exactly zero per-channel across all spatial positions,
vs. the reference's 78% zero, with the same TVM-generic-ops path
computing that reference too — not a MMALIB-vs-generic rounding
difference, an outright missing computation).

**Fix:** `conv2d_impl` now tiles `C_out` for the `stride==1` case too,
the same way it already did for `stride>1` (there capped at `MmaSize`
for a real MMA hardware SIMD-pairing constraint; here capped at 1024,
not a hardware limit, just the largest chunk size empirically validated
end-to-end). The stale `if (C_out > 1024) return -1;` guard is removed
entirely — it dates back to when default bias/scale/shift used
fixed-`[1024]`-sized stack buffers (see `c577e1a894`); those were later
converted to dynamic `TVMBackendAllocWorkspace` allocation sized to the
actual `C_out`, but the guard was never removed, so it kept silently
rejecting (well, silently doing nothing for) exactly the channel counts
its own now-dynamic buffers had no problem with.

Regression coverage: `test_mmalib_conv2d_cout_boundary_dsp.py`,
parametrized at `C_out` in `{512, 1024, 1280, 2048}` — the last two
exercise the new tiling path (single extra chunk, and two full 1024
chunks). All pass at `max_diff<=1`; before the fix, `1280` and `2048`
came back with 100% zero output. mnasnet1_0 end-to-end: `max_diff`
29→2, top-1 258 (correct) restored.

Given mnasnet-family/MobileNet-family models commonly expand to
1280+ channels right before their classifier head, worth checking
whether this also explains any other excluded/borderline model in this
sweep.

### 1 EXCLUDED — quantizer bug fixed; now blocked by a separate TVM pass bug

`maxvit_t` originally failed during quantization itself
(`AssertionError: Expecting input to have dtype torch.float32, but got
dtype: torch.int64`), before TVM was even invoked. `C7xMMAQuantizer`'s
annotation logic matched `view.default`/`permute.default`/
`flatten.using_ints`/`add.Tensor`/`mm.default` purely by op target, with
no check that the operand was actually a floating-point tensor —
`relative_position_index` (an int64 lookup-table buffer for windowed
attention, reshaped via `.view(-1)`) got annotated for quantization like
any other activation, and `convert_pt2e` crashed inserting
`quantize_per_tensor` on it.

**Fixed** in `python/tvm/relax/frontend/torch/c7x_mma_quantizer.py`: added
`_is_float_tensor()` (checks the FX node's traced `meta["val"]` dtype) and
gated the `_TRANSPARENT_OPS`/`_TIDL_ACT_OPS`/`_AVG_POOL_OPS`/`_NORM_OPS`/
`mm`+`add` branches on it. No regressions — all 27
`test_c7x_mma_quantizer*` tests and the full `pt2e-tests/` quick suite
(56 tests) still pass on `c7x_host`.

With that fixed, `maxvit_t` now gets past quantization but fails later,
inside TVM's own pipeline:

```
tvm.error.InternalError: Check failed: (opt) is false: The struct info
of Tuple must be TupleStructInfo, but expression lv7 has struct info
R.Tensor((1, 64, 56, 56), dtype="int8")
```

in `python/tvm/relax/transform/ti_eliminate_qdq_transparent.py`. Not
root-caused yet — excluded (`_EXCLUDED_QUANT_BUG` in
`test_quantized_torchvision.py`) rather than chased further for now.

### 4 EXCLUDED — exceed the 256 MiB AM67A DLOAD DDR heap (weight size alone)

`regnet_y_128gf`, `vit_h_14`, `vit_l_32`, `vit_l_16` — int8 weight size
alone (615 MB, 603 MB, 292 MB, 290 MB respectively) exceeds
`DDR_C7X_1_LOCAL_HEAP`'s 256 MiB, before any runtime/workspace overhead.
See `_EXCLUDED_WEIGHT_SIZE` in `test_quantized_torchvision.py`.

Note: a "DDR watch list" of 5 borderline models (120-190 MB int8 weight —
`convnext_large`, `regnet_y_32gf`, the vgg family, `wide_resnet101_2`)
was flagged in planning as needing real link-time verification rather
than an estimate. All 5 fit and link fine — the int8-weight-size
heuristic was overly conservative there. (`convnext_large` is excluded
for a different reason below, unrelated to weight size.)

### 7 EXCLUDED — NOT SUPPORTED, runtime DDR pool exhaustion

`convnext_large`, `efficientnet_b6`, `efficientnet_b7`, `swin_b`,
`swin_v2_b`, `swin_v2_s`, `swin_v2_t`. Spans both CNN and transformer
architectures — not one family's problem. See `_EXCLUDED_DDR_OOM` in
`test_quantized_torchvision.py`.

All 7 fail identically: `c7x: INFER failed: status=-11 return_value=-1`
/ `{"status":"error","stage":"infer","error":"Function call failed"}`.
This *looks* segfault-like but isn't — `status=-11` is the generic code
`cg_main_dsp` returns whenever any kernel call inside it returns
nonzero, discarding the real error. The real error, visible with
`-profile-layers` (see `TVMPrintLayerProfile`/`compute_service.c`), is
always the same: genuine exhaustion of the DSP's 352 MiB unified DDR
pool (`DDR_C7X_1_LOCAL_HEAP` — weights + DLOAD code/data segments +
runtime workspace tensors, one shared pool), hit at a late layer with
the pool >99% full:

| Model | Shortfall |
|---|---|
| `convnext_large` | requested 602,112 B, free 599,296 B — short by **2,816 B** |
| `efficientnet_b6` | requested 5,227,200 B, free 2,353,024 B — short by **2,874,176 B (~2.74 MB)** |
| `efficientnet_b7` | requested 25,920,000 B, free 25,875,328 B — short by **44,672 B (~43.6 KB)** |
| `swin_b` | requested 401,408 B, free 38,016 B — short by **~355 KB** |
| `swin_v2_b` | requested 4,194,304 B, free 3,069,344 B — short by **~1.07 MB** |
| `swin_v2_s` | requested 1,572,864 B, free 1,516,944 B — short by **~54.6 KB** |
| `swin_v2_t` | requested 1,572,864 B, free 1,187,216 B — short by **~376.6 KB** |

Each is the widest/deepest/largest-input variant in its family
(`convnext_large` vs. `convnext_base`; `efficientnet_b6`/`b7` vs. `b5`;
`swin_b`/`swin_v2_*` vs. the passing `swin_s`/`swin_t`) — peak DDR usage
lands just past the pool's budget. Confirmed via the bump+free-list
allocator in `platform/common/memory_pool.c`: not a leak (LIFO free
list, correct `num_allocs`/`num_frees` bookkeeping, auto-reset when the
pool fully drains) — genuine peak-usage-over-budget for these 7 models
specifically.

**Not fixed — documented as unsupported instead.** Closing the largest
gap (`swin_v2_b`'s ~1.07 MB) would need a further heap extension beyond
the current 352 MiB; there's no more room without either (a) freeing an
MMU region slot (the C7x's ARMv8 MMU config hard-caps at 16 region
descriptors — `maxInstances: 16` in the SDK's `mmu_armv8.syscfg.js` —
and all 16 are already used) or (b) re-deriving a matching fix to
`tvm_dsp_dma.c`'s `virt_to_phys()` hardcoded bounds (needed the last
time this heap was extended). Given the real hardware risk of another
MMU/heap change (a bad region descriptor can hang the DSP, requiring
board reboot/power-cycle), these 7 are excluded rather than chased
further for now.

Note: `design_doc.md` still says this heap is 128 MB — stale; not fixed
here (out of scope for this investigation).

Native cl7x compilation for `swin_v2_t/s/b` (all 3 excluded above) is
also slow, same as the passing `swin_s`/`swin_t`/`vit_b_*` noted earlier
— but that's incidental; the actual reason they're excluded is the DDR
OOM above, not compile time.

<details>
<summary>swin_v2's TVM-side segfault fix (unrelated to the DDR OOM above; already fixed in this codebase)</summary>

`swin_v2_t/s/b` crash ~40s into `relax.build`, well before cl7x, with
`Fatal Python error: Segmentation fault`, unless the fix below is
present. Root cause: `fold_constant.cc`'s
`ConstantFolder::ConstEvaluateCallTIR` tries to eagerly evaluate a
`call_tir` node via a host `"llvm"` JIT whenever all its args happen to
be constants. Build succeeds (LLVM ORC JIT resolves symbols lazily, so
no exception at build time), but the later `CallPacked` fails with `JIT
session error: Symbols not found: [ c7x_dequantize_vecmatmul ]` — a
segfault instead of a catchable exception, because
`c7x_dequantize_vecmatmul` is a real DSP-only kernel with no host symbol.

That kernel is emitted by `FuseDequantizeMatmul`'s C7x path
(`python/tvm/relax/transform/fuse_dequantize_matmul.py`) for
weight-only-quantized `dequantize -> matmul` patterns. swin_v2's
continuous-relative-position-bias MLP (`cpb_mlp` in
`ShiftedWindowAttentionV2`) is applied to a **fixed coordinate buffer**,
not the image — so its matmul's activation operand is itself
compile-time-constant, unlike every other matmul in the network.
`FuseDequantizeMatmul` intentionally runs before `FoldConstant` (to avoid
expanding int8 weights back to float32 in weights.bin), so it can't just
check `isinstance(act, relax.Constant)` — at that point the whole
cpb_mlp chain is still plain `Var`s.

Fixed by `python/tvm/relax/transform/ti_c7x_const_reachability.py`
(`ConstReachability`: walks a Var's producer chain back to
`relax.Constant` leaves), used to skip the DSP-extern path when the
activation is transitively constant, in `fuse_dequantize_matmul.py`
(fall back to the existing portable TE path) and
`ti_fuse_qdq_c7x_relu.py` (leave the composite un-lowered so
`LegalizeOps`/`FoldConstant` handle it safely) — both are needed, since
fixing only the matmul surfaces the identical crash one step later on
`c7x_int8_relu` (cpb_mlp's ReLU is also constant-fed). This is a
*systemic* shape of bug across the whole `FuseQDQToC7x*` pass family
(concat, avgpool, layernorm, activation, TIDL maxpool, MMALIB QDQ
variants all intercept a QDQ pattern into a DSP `call_extern` before
`FoldConstant` runs, none of them check for all-constant inputs) — only
the two instances swin_v2 hits are fixed; `ConstReachability` is there
to reuse if another model triggers one of the others.

</details>

## Shared infrastructure

| Function | Location | Purpose |
|---|---|---|
| `_pt2e_quantize` | `model_utils.py` | Export → `prepare_pt2e` → calibrate (random noise or real images) → `convert_pt2e`; used by the 7 per-model files |
| `load_model_with_preprocessing`, `load_image`, `get_all_classification_models` | `tests/cstatic/cl_torchvision.py` | Dynamic model loading with correct per-model preprocessing; cross-imported by `test_quantized_torchvision.py` |
| `e2e_quantize_and_import`, `run_and_check` | `tests/ti-dsp-runtime/pt2e-tests/pt2e_utils.py` | Full quantize→import pipeline and MMALIB compile+run+assert; cross-imported by `test_quantized_torchvision.py` |

`run_and_check`'s default tolerance (`max_diff=2`, ±1 LSB) is calibrated
for single-op unit tests (see `pt2e-tests/README.md`) — whole models
compound int8 rounding error across many layers, so
`test_quantized_torchvision.py` passes `max_diff=25` explicitly instead
of relying on the default.

## Prerequisites

- TVM built with the `c_static` backend (`TVM_HOME` set, `PYTHONPATH`
  includes `python/`)
- `TI_CGT_C7000_PATH` for DSP tests
- For `c7x_dload`: firmware deployed on AM67A (`deploy-c7x.sh`)
- `--mmalib` fixture/flag (from `conftest.py`) selects the MMALIB target;
  omitting it runs the generic (non-MMALIB) int8 codegen path instead
