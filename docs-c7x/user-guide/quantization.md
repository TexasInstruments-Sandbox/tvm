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
scheme the c_static/MMALIB backend expects:

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

No manual per-op scale/zero-point computation is needed -- this is the
standard PT2E `prepare` / calibrate / `convert` sequence; `C7xMMAQuantizer`
only decides *which* ops get annotated and with what quantization scheme, so
that the resulting graph is one the backend can offload to MMALIB.

`mod` is a regular Relax `IRModule` from here on -- compile it for `c_static
-mcpu=c7x -mmalib=1` the same way as any other model; see
[Compilation](compilation.md) for that step.

## Calibration: Use Real Data

`prepared(sample)` runs each calibration sample through the model to record
the activation ranges each Q/DQ node's scale/zero-point is computed from.

If you only need to verify the DSP and CPU agree on some input, random
noise calibration is fine (this is what most of `quantized/model_utils.py`
does for its own test coverage). For anything where the actual prediction
matters, calibrate on real, representative data.

## Choosing int8 vs int16

`C7xMMAQuantizer(dtype=...)` picks the quantization width for both weights
and activations:

| | int8 (default) | int16 |
|---|---|---|
| Weights | int8, per-channel scale | int8, per-channel scale |
| Activations | int8; symmetric or asymmetric (`symmetric_activations=`) | int16, always symmetric (asymmetric int16 has no lowering path -- forced on with a warning if requested otherwise) |
| When to use | Default -- widest MMALIB op coverage, smallest weights/activations | A layer's accuracy suffers under int8's narrower range |

Both dtypes require symmetric *weight* quantization (`w_zp=0`) regardless
of the activation setting. See [MMALIB Integration -- Supported
Operations](../contributor-guide/backend/mmalib-integration.md#supported-operations)
for the exact shape/alignment constraints MMALIB places on each op per
dtype.

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
