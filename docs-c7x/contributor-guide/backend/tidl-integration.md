# TIDL Integration

Offloading Relax subgraphs to TI Deep Learning (TIDL) on the C7x MMA
accelerator, via the `TIDLOffloadCompiler` pipeline in
`python/tvm/relax/backend/tidl/`. Non-offloaded ops remain in TVM and
execute as generated C code on the C7x scalar pipeline.

> **J722S only.** TIDL subgraph offload requires firmware built with
> `--tidl ON` (the `j722s-evm`/AM67A default, which also forces
> `--mmalib ON`). BeagleY-AI firmware is always built `--tidl OFF
> --mmalib ON` — no TIDL offload path exists on that board. See
> [MMALIB Integration](mmalib-integration.md#firmware-decoupling-mmalib-from-tidl)
> for the firmware-side `USE_TIDL_RUNTIME`/`USE_TI_MMALIB` split, and pass
> `-tidl-kernels=0` when compiling for BeagleY-AI regardless of whether a
> given model uses TIDL subgraph offload at all (it also controls whether
> `max_pool2d` targets the TIDL-backed kernel).

Target string: `c_static_lib -mcpu=c7x` (TIDL artifacts are embedded and
`-tidl-runtime=1` is applied automatically by `TIDLOffloadCompiler.build()`
whenever a model actually produces TIDL subgraphs).

## Files

| File | Purpose |
|------|---------|
| `python/tvm/relax/backend/tidl/tidl.py` | `TIDLOffloadCompiler`: partition, import, lower, bridge generation, build |
| `python/tvm/relax/backend/tidl/patterns.py` | TIDL `FusionPattern` definitions (74 patterns) and Python-side constraint checks |
| `python/tvm/relax/backend/tidl/README.md` | Full pattern/constraint reference and pipeline architecture (source of truth for this page) |
| `src/runtime/ti_dsp/tidl/tidl_api.{c,h}` | DLOAD-module-side IALG lifecycle (`init/process/free_tidl_subgraph`) |
| `src/runtime/ti_dsp/tidl/tidl_api_mem.{c,h}` | `appMemAlloc`/`appMemFree` routed to the TVM DDR heap |
| `src/runtime/ti_dsp/tidl/ti_mem_manager.{c,h}` | Bump allocator for L1/L2/L3 SRAM pools used by TIDL |
| `src/runtime/ti_dsp/tidl/tidl_host_stubs.c` | x86 stubs for firmware-provided symbols, `c7x_host` builds only |
| `src/runtime/ti_dsp/tidl/README.md` | Runtime-side IALG lifecycle, UDMA sharing, memory pools, known issues |
| `tests/ti-dsp-runtime/tidl-tests/` | Partition, codegen, import, and hardware test suites |

## Pipeline

```
Relax IR (conv2d, relu, pool, softmax, ...)
    |
    v
partition()        FuseOpsByPattern (TIDL patterns) -> MergeCompositeFunctions
    |               into Codegen="tidl" subgraph functions
    v
tidl_import()       Loads tidl_model_import_relax.so; per subgraph:
    |               ImportInit -> ImportNode -> Link -> Optimize -> PostProcess
    |               Produces net.bin + params_1.bin artifacts on disk
    v
lower_tidl()        Replace Codegen="tidl" functions with TIR PrimFunc stubs;
    |               each stub emits call_extern("tidl_subgraph_N_process", ...)
    v
generate_bridge()   Generate tidl_bridge.c/h resolving the extern calls
  + relax.build()    c_static_lib codegen emits lib0.c + weights.bin
    v
_build_dynmod()     TI C7x cross-compile; embeds TIDL artifacts as .rodata;
    |               links tidl_api.c (IALG wrapper) + bridge
    v
lib0.out  --DLOAD-->  AM67A (J722S): TIDL int8 on MMA + TVM float32 on scalar C7x
```

`TIDLOffloadCompiler.build(mod, params=params)` runs the whole pipeline in
one call; `build(exec_mode="c7x_host")` builds an x86-64 host-emulation
executable instead of a DLOAD module, using the PC TIDL reference
libraries, for pipeline validation without hardware. Individual stages
are available for step-by-step debugging — see the README linked above
for both usage styles.

Batch normalization is not partitioned as its own pattern: `prepare()`
runs `FoldBatchnormToConv2D` + `FoldConstant` first, algebraically folding
inference-mode BN parameters into the preceding conv2d's weight/bias so
the existing `conv2d_bias_relu`-style patterns match directly.

## Supported Operations

74 patterns across these categories (exact op list and per-pattern
constraints in `python/tvm/relax/backend/tidl/README.md`):

| Category | Ops |
|----------|-----|
| Convolution | `conv2d` (plain, +bias, +relu, +clip, and combinations — 6 patterns); `conv2d_transpose` (plain and +bias — 2 patterns) |
| Pooling | `max_pool2d`, `avg_pool2d` |
| Reduction | `mean` (spatial axes only), `sum` (single axis), `reduce_max`/`reduce_min` (axis=HEIGHT only), `argmax`/`argmin` (axis=channel, `keepdims=True` only) |
| Activations | `relu`, `sigmoid`, `tanh`, `clip` (incl. relu6), `leakyrelu`, `prelu`, `elu`, `hard_sigmoid`, `hard_swish`, `mish` |
| Element-wise | `add`, `multiply`, `subtract`, `divide`, `maximum`, `minimum` (4-D inputs only) |
| Linear | `matmul`, `matmul` + bias (FC) |
| Attention | `softmax` |
| Shape/layout | `reshape`, `flatten`, `squeeze`, `expand_dims`, `strided_slice`, `permute_dims`, `concat` |
| Normalization | `layer_norm`, `instance_norm` |
| Data type/padding | `astype` (cast), `nn.pad` (constant zero-padding) |
| Advanced | `image.resize2d`, `take`, `topk`, `split`, `nn.pixel_shuffle` (depth-to-space), `broadcast_to` (expand), `scatter_elements`, `scatter_nd`, `image.grid_sample` |
| Math/unary | `abs`, `sqrt`, `power`, `exp`, `log`, `erf`, `floor`, `negative`, `sin`/`cos`/`tan`, `sinh`/`cosh`, `asin`/`acos`/`atan`/`asinh` |
| Quantization (stubs) | `quantize`, `dequantize` |

Ops not in this list stay on the TVM scalar path automatically — nothing
needs to be excluded explicitly.

## Configuration

`TIDLOffloadCompiler(config={...})` accepts:

| Key | Default | Description |
|-----|---------|--------------|
| `artifacts_dir` | `/tmp/tidl_artifacts` | Output directory for TIDL binaries |
| `tidl_tools_path` | auto-detect from `C7X_MMA_TIDL_PATH` env | Path to the TIDL import tool/config |
| `calibration_inputs` | **required** | List of per-frame numpy arrays (one `(1,C,H,W)` array per calibration image). Random calibration data is rejected: TIDL derives int8 quantization scales from the statistics of these inputs, so unrepresentative data produces wrong scales and wrong classification results downstream |
| `num_calibration_frames` | 1 | Number of calibration iterations |
| `skip_failing_subgraphs` | `False` | Fall back to TVM instead of raising when a subgraph fails TIDL's own network-compiler/post-processing step |
| `max_subgraphs` | `None` (unlimited) | Cap the number of subgraphs offloaded to TIDL, keeping the top-N by estimated FLOPs |

Calibration uses real intermediate activations per subgraph, not just at
the model's inputs: the compiler augments the partitioned module to also
emit each TIDL subgraph's *inputs* as extra outputs, runs that module on
CPU with the calibration frames, and feeds the collected activations into
TIDL's own calibration step for that specific subgraph. This matters for
subgraphs that aren't the first one in the model — their inputs are
post-activation feature maps, not raw pixel values, so calibrating them
with the wrong distribution produces wrong int8 scales.

## Firmware Integration

TIDL's algorithm libraries are linked into the **firmware**, not the
model module (same split as MMALIB's L2 DMA prefetch — see
[MMALIB Integration](mmalib-integration.md)). The firmware exports the
shared resources a loaded module needs via the DLOAD symbol table
(IALG function table, DDR heap allocation, the UDMA driver handle, L1/L2
SRAM pools, cache writeback). TIDL and TVM's own DMA tiling share one
UDMA driver instance, initialized once at firmware boot.

Each TIDL subgraph goes through the standard IALG lifecycle at runtime
(alloc → init → activate → process → deactivate, freed at module unload)
— see `src/runtime/ti_dsp/tidl/README.md` for the exact call sequence.
`tidl_bridge_cleanup()` releases every TIDL instance's handle, DMA
channels, and memory records before `dyn_loader_unload()` frees the
module's ELF segments, for the same reason MMALIB/TIDL's other cleanup
ordering matters — see
[C Static Lib Backend — Module Unload Lifecycle](../firmware/design-deep-dive.md#dynamic-module-loading-dload).

## Known Limitations

- **`argmax`/`argmin` reducing to a single channel**: `axis=1` (channel)
  with `keepdims=True` on an input where that leaves exactly one output
  channel hits a TIDL library limitation (a "zero channels processed per
  call" internal error) for some input configurations. Prefer reducing
  over a spatial axis instead, or treat the case as `xfail` if it's
  intrinsic to the model.
- **Softmax-heavy heads may diverge on hardware despite passing host
  emulation.** A model whose TIDL-offloaded output feeds directly into a
  TVM-side `exp()`-based op (e.g. a detection head's distribution-focal-loss
  softmax) can produce `NaN` on real hardware (`c7x_dload`) while passing
  on the PC AVX reference path (`c7x_host`) with identical calibration.
  The C7x MMA and the PC AVX reference round int8 dequantization slightly
  differently; if that shift pushes values far enough negative, `exp()`
  underflows to exactly `0.0` for every element of a reduction, producing
  `0/0 = NaN`. This is a hardware/reference rounding gap, not a
  calibration bug — per-subgraph calibration (above) is necessary but not
  sufficient to fix it. If a model hits this, keep the affected head off
  the TIDL path (either leave it on the TVM scalar path, or use
  [MMALIB](mmalib-integration.md) + `C7xMMAQuantizer` for that portion of
  the model instead, which bakes quantization scales into the compiled
  binary and applies them identically on host and hardware).
- **Element-wise ops require rank exactly 4.** A sub-4D use (e.g. an FC
  bias add with shape `(1, N)`) is rejected by the pattern's constraint
  check rather than being offloaded and crashing at runtime.
- **`topk` always returns both values and indices.** Request
  `ret_type="both"` and extract index `[0]` for values-only usage.

## Testing

```bash
cd tests/ti-dsp-runtime

# Partition + codegen — no .so, no hardware
pytest tidl-tests/test_tidl_partition.py tidl-tests/test_tidl_codegen.py -v

# Per-layer partition tests — no .so, no hardware (~0.3s)
pytest tidl-tests/test_tidl_layer_offload.py -k TestLayerPartition -v

# Import pipeline tests (needs tidl_model_import_relax.so)
pytest tidl-tests/test_tidl_relax_import.py -v

# Per-layer hardware tests on AM67A (all supported layer types)
TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS \
  pytest tidl-tests/test_tidl_layer_offload.py -k TestLayerHardware -v

# Full model hardware tests
pytest tidl-tests/test_tidl_resnet_e2e.py -v -s
pytest tidl-tests/test_tidl_mv2_e2e.py -v -s
pytest tidl-tests/test_yolo_dsp.py -v -s
```

Requirements vary by test tier: pure partition/codegen tests need only
TVM; import and hardware tests additionally need
`tidl_model_import_relax.so` (built from a separate TIDL source tree),
`TI_CGT_C7000_PATH`, and, for hardware tests, a connected AM67A/J722S
board. See `tests/ti-dsp-runtime/tidl-tests/README.md` for the full test
list and per-file requirements.

## Related Documentation

- [MMALIB Integration](mmalib-integration.md) — the other C7x MMA offload
  path; firmware `--tidl`/`--mmalib` linkage, board differences
- [C Static Lib Backend](c-static-lib.md) — target configuration, DLOAD
  deployment
- [Firmware Design Deep-Dive](../firmware/design-deep-dive.md) — DLOAD
  module unload ordering, symbol export table
