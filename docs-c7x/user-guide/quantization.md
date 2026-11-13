# Quantization

## Why Quantize for the C7x NPU

The C7x's MMA coprocessor -- accessed through MMALIB -- only executes
fixed-point kernels: int8 or int16. It has no float32 path. A `conv2d` or
`matmul` left in float32 still runs, but on the C7x's scalar pipeline
instead of the MMA coprocessor, and the difference is not small: a single
64ch 56x56 int8 conv2d layer takes ~45M cycles as scalar C7x code versus
~1.67M cycles through MMALIB (27x), dropping further to ~477K cycles (96x)
once the data is staged into L2 SRAM. ResNet-18 end to end sees a 47x
speedup this way. See [MMALIB Integration](../contributor-guide/backend/mmalib-integration.md#performance-am67a-c7x-1-ghz)
for the full per-model numbers.

Quantizing a model to int8/int16 is what makes its ops eligible for that
MMALIB offload in the first place -- it is a prerequisite for the speedup,
not an optional accuracy/size trade-off layered on top of it.

## How Models Are Quantized Here

Every quantized model in this repo (and both runnable examples) uses
PyTorch's PT2E static quantization workflow with `C7xMMAQuantizer`, a TVM-
supplied quantizer that annotates the exported graph with the quantization
scheme the c_static_lib/MMALIB backend expects:

```python
import torch
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program

exported = torch.export.export(model, example_args)
quantizer = C7xMMAQuantizer(dtype="int8")   # or "int16"
prepared = prepare_pt2e(exported.module(), quantizer)

for sample in calibration_data:             # real inputs -- see Calibration below
    prepared(sample)

quantized_gm = convert_pt2e(prepared)
mod = from_exported_program(
    torch.export.export(quantized_gm, example_args),
    keep_params_as_input=True,
)
```

!!! warning "`C7xMMAQuantizer` is a proof of concept, not production-ready"
    `C7xMMAQuantizer` is a proof of concept created to test TVM's compilation
    of quantized models (the PT2E-to-MMALIB path). It is not production-ready.

No manual per-op scale/zero-point computation is needed -- this is the
standard PT2E `prepare` / calibrate / `convert` sequence; `C7xMMAQuantizer`
only decides *which* ops get annotated and with what quantization scheme, so
that the resulting graph is one the backend can offload to MMALIB.

`mod` is a regular Relax `IRModule` from here on -- compile it for `c_static_lib
-mcpu=c7x -mmalib=1` the same way as any other model; see
[Compilation](compilation.md) for that step.

## Calibration: Use Real Data

`prepared(sample)` runs each calibration sample through the model to record
the activation ranges each Q/DQ node's scale/zero-point is computed from.

Calibrate on real, representative data, preprocessed the same way as the
inputs the model will see: the scales come from the ranges observed here,
and random noise produces different ranges from real images. The
per-model tests in `quantized/` calibrate on `quantized/calib_images/` and
run the DSP on an image outside that set.

Make inputs NCHW-contiguous (`.contiguous()`) for calibration and for any
PyTorch reference you compare the DSP with. TorchVision's transforms return a
channels-last view, and the quantized graph's output depends on memory layout
even though the float model's doesn't: a tiny float difference flips a rounding
decision, and the change spreads through later layers. The DSP is always given
NCHW data.

## How Activation Ranges Are Chosen

Most activation tensors get one scale (per tensor). How calibration turns the values
it sees into that range is the main accuracy lever, and you don't have to choose it
per model: the quantizer recognises a few kinds of tensor from the graph's structure
and ranges them differently. Everything else uses the observer.

`C7xMMAQuantizer(activation_observer=...)` picks the observer:

- `"minmax"` (default) uses the extremes. One outlier widens the range for every
  value, so on models with outliers, adding calibration data can reduce accuracy.
- `"histogram"` picks the range with the least total squared error over the values
  seen. It saturates an outlier once the precision it costs the other values
  outweighs clipping it, so a tensor with few values, such as the output of a
  global pool, still behaves much like min/max. It calibrates more slowly.

These tensors are ranged by rule instead, because a few channels or values set their
range and leave the rest few codes. All but the clamp rule apply to int8 only; int16
has the resolution to spare:

| Tensor | Range | Turn off with |
|---|---|---|
| A conv or linear output read only by ReLU6 (or another `hardtanh`) | the clamp's range: the clamp discards anything beyond it | (symmetric activations only; always on) |
| The inputs and outputs of SiLU, hardswish and hardsigmoid, and a sigmoid's input | 0.1% of values clipped from each tail | `smooth_activation_percentile=None` |
| In depthwise (inverted-residual) blocks: a conv output read only by ReLUs next to a depthwise conv, and a squeeze-excite output (a SiLU or hardswish output times its gate) | 0.01% clipped from each tail | `depthwise_block_percentile=None` |
| A depthwise conv's input and, when it comes through a SiLU, hardswish or ReLU, the conv output before it | one scale per channel: the channel's calibrated peak times 2, never coarser than the per-tensor scale; any other reader (a residual add) still reads it per tensor | `depthwise_input_headroom=None` (symmetric activations only) |

A network with none of these tensors, such as a ResNet, quantizes with the observer
alone. The rules are kept to these tensors on purpose: clipping a ResNet's ReLU
tensors the same way costs it accuracy. Per-channel scales work on a depthwise
conv's input because it keeps its channels apart, so each channel's scale folds into
its own requantize; a depthwise block's input channels can differ by over 100x, more
than one scale can serve.

Measured effect, quantized minus float top-1 in points (calibrated on the repo's 12
images, 5,000 ImageNetV2 images):

| Model | min/max only | Defaults |
|---|---|---|
| ResNet-18 | -0.2 | -0.2 |
| MobileNetV2 (ReLU6) | -1.1 | -1.1 |
| ShuffleNetV2 x0.5 | -1.6 | -1.2 |
| MNASNet 0.5 | -8.5 | -1.9 |
| MobileNetV3-Large | -12.7 | -3.9 |
| EfficientNet-B0 | -41.7 | -3.4 |
| MobileNetV3-Small | -52.9 | -6.0 |

MobileNetV2's clamp rule is on in both columns.

## Choosing int8 vs int16

`C7xMMAQuantizer(dtype=...)` picks the quantization width for both weights
and activations:

| | int8 (default) | int16 |
|---|---|---|
| Weights | int8, per-channel scale | int16, per-channel scale |
| Activations | int8; symmetric or asymmetric (`symmetric_activations=`) | int16, always symmetric (asymmetric int16 has no lowering path -- forced on with a warning if requested otherwise) |
| When to use | Default -- widest MMALIB op coverage, smallest weights/activations | A layer's accuracy suffers under int8's narrower range |

Both dtypes require symmetric *weight* quantization (`w_zp=0`) regardless
of the activation setting. See [MMALIB Integration -- Supported
Operations](../contributor-guide/backend/mmalib-integration.md#supported-operations)
for the exact shape/alignment constraints MMALIB places on each op per
dtype.

## Checking Accuracy on Your Model

Measure what quantization costs before you deploy. `convert_pt2e`'s output
(`quantized_gm` above) is the reference the DSP reproduces: the repo's DSP tests
compare the board against it. So comparing it with the float model in PyTorch, on
held-out data and with your task's own metric, predicts the DSP's accuracy without
compiling anything:

```python
with torch.no_grad():
    for x, target in held_out_data:            # not the calibration samples
        x = x.contiguous()
        float_out, quant_out = model(x), quantized_gm(x)
        # score both against target with your metric; also compare them directly,
        # e.g. top-1 agreement or the cosine of the outputs
```

Report the difference between the two, not the quantized score alone. For
ImageNet-style classifiers, `tests/ti-dsp-runtime/accuracy/evaluate.py` does this
on ImageNetV2; see [Quantized Model Sweep -- Measuring accuracy](../contributor-guide/testing/quantized-sweep.md#measuring-accuracy).

## If Accuracy Is Poor

In order:

1. Check the calibration data: real inputs, preprocessed exactly as at inference,
   NCHW-contiguous (see [Calibration](#calibration-use-real-data)).
2. Try `activation_observer="histogram"`. It was never clearly better than the
   default on the models measured here, but it can be on a model with outliers in
   tensors the rules don't cover.
3. If your model isn't one of the families above, measure with each range rule
   turned off as well (`smooth_activation_percentile=None`,
   `depthwise_block_percentile=None`, `depthwise_input_headroom=None`). The rules
   were measured on those families.
4. Use `dtype="int16"`. It costs speed and narrows MMALIB coverage (see the
   table above), and is the fallback when int8 can't represent the model.

There's no per-layer report yet that names the tensors costing the most, and no way
to put only some layers in int16.

## Confirming It Actually Offloaded

Quantizing a model doesn't by itself guarantee any given op reaches
MMALIB -- an op the backend doesn't recognize as fusable still gets
quantized (paying the int8/int16 cast overhead) but then falls back to a
plain scalar loop instead of the MMA coprocessor, which is strictly worse
than leaving it float. Two example-script flags make this visible without
guesswork:

- `--visualize` generates an HTML graph of which ops went to MMALIB versus
  scalar C7x code, with no board required.
- `--profile-layers` adds per-layer DSP cycle counts, which `--visualize`
  can overlay on that same graph.

See [Examples -- Visualizing the MMALIB
offload](examples.md#visualizing-the-mmalib-offload) for the exact
commands.

## Examples in This Repo

- **YOLO26 object detection** (Python API) -- int8 PT2E, calibrated on real
  JPEGs. See [Examples: YOLO26 & ResNet-18](examples.md).
- **ResNet-18 classification** (C++ API) -- int8 PT2E, calibrated on the
  same real image used for inference. See [Examples: YOLO26 &
  ResNet-18](examples.md).

## See Also

- [Compilation](compilation.md) for the target string, the public
  `relax.build` API, and the end-to-end build/deploy flow.
- [MMALIB Integration](../contributor-guide/backend/mmalib-integration.md)
  for the compiler pass pipeline, per-op dtype/shape constraints, and full
  performance tables.
- [Python / C++ API Reference](python-api.md) for running the compiled,
  quantized model on the board.
