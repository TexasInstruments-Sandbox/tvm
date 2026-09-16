---
name: model-workflow
description: "End-to-end workflow for compiling a new model to C7x DSP. Use when: adding a new model (CNN, transformer, LLM), choosing offload strategy (pure c_static_lib vs TIDL vs MMALIB), selecting quantization approach (float32, int8 QDQ, int16, weight-only), exporting from PyTorch (torch.export, PT2E), running first inference on AM67A, or diagnosing model-level failures. NOT for individual pass implementation or kernel writing."
---

# New Model Workflow

End-to-end recipe for compiling a PyTorch model to C7x DSP.

## Decision Tree

### 1. Choose Offload Strategy

| Strategy | Best for | Tradeoffs |
|----------|----------|-----------|
| **Pure c_static_lib** | Small models, custom ops, prototyping | Full flexibility, scalar C7x loops (slow for large matmul/conv) |
| **MMALIB** (`-mmalib=1`) | Models with many conv2d/matmul ops needing fine-grained control | 27-96x speedup per layer, supports int8 + int16, requires aligned dimensions |
| **TIDL** | Standard CNN topologies (ResNet, MobileNet, YOLO) needing max throughput | Highest peak perf for supported ops, but less flexible, requires calibration data |

Decision criteria:
- **All ops supported by TIDL?** → Use TIDL (highest throughput)
- **Mix of TIDL-supported and custom ops?** → TIDL offload (supported subgraphs) + c_static_lib (remainder)
- **LLM / transformer?** → MMALIB int16 for linear layers + c_static_lib for attention/norms
- **Need int8 CNN with MMA?** → MMALIB QDQ (conv2d, dwconv, FC)
- **Prototyping / small model?** → Pure c_static_lib

**TIDL and MMALIB are mutually exclusive** — both use the same MMA hardware.

### 2. Choose Quantization

| Approach | Precision | Calibration needed | Use case |
|----------|-----------|-------------------|----------|
| Float32 | Full | No | Baseline, debugging |
| Weight-only INT8 | Mixed (fp32 act × int8 weights) | No | LLM decode (SmolLM current) |
| PT2E W8A8 (int8) via `C7xMMAQuantizer("int8")` | Full int8 | Yes (100+ samples) | CNN (ResNet, MobileNet) |
| PT2E W16A16 (int16) via `C7xMMAQuantizer("int16")` | Full int16 | Yes (symmetric only; d_zp=0) | CNN with higher precision requirement |
| Int16 LLM (`LegalizeMLPToMMALIBInt16`) | int16 accum | No (dynamic per-tensor at runtime) | LLM linear layers; recommended for SmolLM |

**PT2E int16 notes:** `from_exported_program` emits `int8` zero_point constants
(required by TVM's `relax.dequantize`; safe since symmetric zp is always 0).
Int16 depthwise is 3×3 only (MMALIB-882). Use `C7xMMAQuantizer("int16")` then
the standard MMALIB pipeline — `FuseMMALIBQDQConv2dI16/DwConv2dI16/FCI16/FuseInt16ResidualAdd`
fire automatically.

### 3. Choose Test Mode

Start with `c7x_host` (fast iteration, no hardware needed), then validate on `c7x_dload` (real hardware).

## Step-by-Step Recipe

### CNN (Classification/Detection)

```bash
# 1. Export from PyTorch + quantize (PT2E)
#    See existing tests: test_resnet_dsp.py, test_classification_dsp.py,
#    quantized/test_quantized_resnet.py

# 2. Compile for host emulation
pytest --rootdir=. dsp-tests/test_my_model_dsp.py -v --dsp-mode=c7x_host

# 3. Validate on hardware
pytest --rootdir=. dsp-tests/test_my_model_dsp.py -v --dsp-mode=c7x_dload

# 4. With MMALIB (optional)
pytest --rootdir=. dsp-tests/test_my_model_dsp.py -v \
    --dsp-mode=c7x_dload --mmalib --profile
```

### LLM / Transformer

```bash
# 1. Export with QuantizedLinear (weight-only int8)
# 2. Compile: python smollm_c7x.py compile-chat --quantize -o /tmp/model
# 3. Test:   python smollm_c7x.py test --quantize --dsp-mode c7x_host
# 4. Deploy: python smollm_c7x.py deploy --artifacts /tmp/model
# 5. Board:  ssh root@am67a python3 /opt/smollm/smollm_board.py --model-dir /opt/smollm
```

### TIDL Offload

```bash
# 1. Partition + import (requires tidl_model_import_relax.so)
# 2. See tidl-tests/test_tidl_import_e2e.py for the pattern
# 3. Compile with TIDL artifacts embedded in DLOAD ELF
```

## Common Pitfalls by Model Category

### CNN
- **Dimension alignment**: none required — MMALIB conv2d handles arbitrary C_out via a partial-chunk loop in the kernel (the old C_out-divisible-by-32 gate was removed; see `quantized_model_optimization.md` Step 6)
- **NHWC vs NCHW**: `-mmalib=1` forces NCHW; without it, NHWC is used for DMA tiling
- **Multi-output (detection)**: Use `InferMulti()` (capped at `Model::kMaxOutputs` = 128)

### Transformer / LLM
- **Static shapes**: c_static_lib requires fixed tensor shapes → two-model design (prefill + decode)
- **KV cache**: Must be explicit inputs/outputs (not mutable state)
- **Int8 MMALIB fails at depth**: Requantization error compounds across 30+ layers; use int16
- **Vocab size**: lm_head (576→49152) dominates compute; quantize it for decode speedup

### General
- **Weight size**: ResNet-18 = ~47 MB ELF; larger models may exceed the 256 MB DLOAD DDR heap (`DDR_C7X_1_LOCAL_HEAP`, raised from 128 MB to fit SmolLM's weights)
- **L2 budget**: 1.25 MB on J722S (`l2-sram-size` target attr); `ScheduleC7xDMATiling` tiles conv2d/matmul PrimFuncs whose double-buffered working set doesn't fit
- **fp_reassoc**: TI compiler may reorder float ops; use `--fp_reassoc=off` if accuracy matters

## Verification Checklist

1. Host emulation passes (`c7x_host`) — fast iteration
2. Numerical match vs PyTorch reference (rtol=1e-4 for float32, looser for quantized)
3. Hardware passes (`c7x_dload`) — catches alignment/DMA issues
4. Cycle count is reasonable (compare to similar models in `results/cycles.csv`)
5. Profile layers (`--profile`) — identify bottleneck ops for optimization

## Related Skills

- `relax-c7x:cstatic` — Pipeline passes and target options
- `relax-c7x:mmalib-offload` — MMALIB pattern matching and constraints
- `relax-c7x:tidl-offload` — TIDL partitioning and import pipeline
- `relax-c7x:testing` — Test authoring patterns and fixtures
- `relax-c7x:build` — Environment setup and build commands
