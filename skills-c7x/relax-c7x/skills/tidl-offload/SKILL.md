---
name: tidl-offload
description: "TIDL subgraph offloading for TVM Relax c_static_lib backend. Use when working on: TIDL partitioning (FuseOpsByPattern + MergeCompositeFunctions), tidl_import() FFI pipeline, LowerTIDLToTIR, bridge generation (tidl_bridge.c), TIDL composite patterns (74 patterns: conv2d/conv2d_transpose variants, pooling, batch_norm, activations, elementwise, reductions, transformer ops, quantize/dequantize, and more), tidl_model_import_relax.so, TIDL device config, IALG lifecycle (init/process/free), UDMA handle management, or TIDL hardware resource allocation (L1/L2/L3/DDR pools, DRU channels). NOT for MMALIB direct integration (see mmalib-offload) or general pipeline configuration (see cstatic)."
---

# TIDL Subgraph Offloading

Offload supported subgraphs from TVM/Relax models to TI's TIDL accelerator on the C7x MMA. Non-TIDL ops remain in TVM and execute as generated C code on the C7x scalar pipeline.

## When to Use TIDL (vs MMALIB)

Use TIDL for standard CNN topologies where entire subgraphs can be offloaded for maximum throughput. TIDL handles multi-op fusion internally (conv+bn+relu as single accelerated kernel). Requires `tidl_model_import_relax.so` and calibration data. **TIDL and MMALIB are mutually exclusive** — both target the same MMA hardware; do not use `-mmalib=1` with TIDL partitioning.

## Pipeline

```
Relax IR
  → Partition (FuseOpsByPattern + MergeCompositeFunctions, Codegen="tidl")
  → TIDL Import (tidl_import() via Relax FFI → net.bin + params_1.bin)
  → Lower (LowerTIDLToTIR: replace TIDL funcs with call_extern stubs)
  → c_static_lib Codegen (emit lib0.c + weights.bin; sets `-tidl-runtime=1` target attr)
  → IO Meta (write_io_meta() → tvm_dsp_io_meta.bin, declared input_buf/
    output_buf capacity for the per-buffer-dmabuf c7x_dload protocol)
  → Bridge Generation (tidl_bridge.c: eager tidl_bridge_init_all() + per-subgraph process() + tidl_bridge_cleanup())
  → C7x DLOAD Build (lib0.out with embedded TIDL artifacts + IO-meta blob)
  → Deploy to AM67A via c7x_compute
```

**The IO Meta step is easy to silently skip** if a caller re-implements
this pipeline by hand instead of going through `TIDLOffloadCompiler.build()`/
`codegen_and_build()` (see "API surface" below) — a module built without
`tvm_dsp_io_meta.bin` reports zero declared IO capacity to `c7x_compute`,
which since the per-buffer-dmabuf change means a zero-capacity `input_buf`
and `-EFBIG`/`DSPOutOfMemoryError` at the *first* c7x_dload inference, not
at load or compile time. `write_io_meta()` logs nothing on success or
skip in `tidl.py` itself (unlike the equivalent path in `dsp_utils.py`,
which does) -- if a hand-rolled build fails this way, check whether
`tvm_dsp_io_meta.bin` exists next to the generated `lib0.c`/`weights.bin`
before looking anywhere else. See `docs-c7x/contributor-guide/firmware/dmabuf-design.md` for the
protocol's design (the DSP-side wire format, capacity policy, and why a
missing blob means zero capacity rather than a build failure).

### API surface: `compile()` vs `build()` vs `codegen_and_build()`

`TIDLOffloadCompiler.compile(mod, params)` runs only partition→import→lower
and returns `(lowered, artifacts)` — this is the expensive step (TIDL
import/calibration). `build(mod, params, ..., exec_mode)` is a convenience
wrapper: `compile()` once, then hand off to `codegen_and_build(lowered,
artifacts, ..., exec_mode)`, which covers everything from c_static_lib codegen
through the native build (the last five stages of the pipeline above).

Call `compile()` once yourself and then `codegen_and_build()` twice
(`exec_mode="c7x_host"` and `exec_mode="c7x_dload"`) when you need both
artifacts from the *same* compiled module — this is what `test_yolo_tidl`
does, so the expensive TIDL import doesn't run twice per model. Calling
`build()` twice instead would re-run `compile()` each time.

## Two-Level Gating

1. **TVM-side (Python):** pattern matching identifies offloadable ops via `FuseOpsByPattern` with 74 composite patterns. Constraint callbacks reject ops exceeding TIDL hardware limits.

2. **TIDL-side (C++ .so):** `tidl_model_import_relax.so` checks each operator against device-specific constraints and compiles to optimized MMA code.

## 74 Composite Patterns

`python/tvm/relax/backend/tidl/patterns.py`'s `get_tidl_patterns()` returns 74
patterns (verify with `len(get_tidl_patterns())`; this has grown well past
earlier fixed counts as ops were added — don't hardcode the number in new
docs). Grouped by category:
- Composite activations matched first (priority over standalone conv2d+relu): `tidl.hard_swish`, `tidl.hard_sigmoid`, `tidl.mish`, `tidl.elu`
- `tidl.nn.conv2d` variants (plain, +bias, +relu, +bias+relu, +bias+clip, +clip) and `tidl.nn.conv2d_transpose` variants
- Pooling (`tidl.nn.max_pool2d`, `tidl.nn.avg_pool2d`), `tidl.nn.batch_norm`, `tidl.mean`
- Standalone activations: `tidl.nn.relu`, `tidl.sigmoid`, `tidl.tanh`, `tidl.clip`, `tidl.nn.leakyrelu`, `tidl.nn.prelu`
- Elementwise: `tidl.add`, `tidl.multiply`, `tidl.divide`, `tidl.subtract`, `tidl.maximum`, `tidl.minimum`
- Transformer/matmul ops: `tidl.matmul`, `tidl.matmul_bias`, `tidl.permute_dims`, `tidl.nn.softmax`, `tidl.nn.layer_norm`, `tidl.nn.instance_norm`
- Shape/tensor ops: `tidl.reshape`, `tidl.flatten`, `tidl.squeeze`, `tidl.expand_dims`, `tidl.strided_slice`, `tidl.cast`, `tidl.nn.pad`, `tidl.concat`, `tidl.split`, `tidl.take`, `tidl.topk`, `tidl.expand`
- Reductions: `tidl.sum`, `tidl.reduce_max`, `tidl.reduce_min`, `tidl.argmax`, `tidl.argmin`
- Math unary: `tidl.abs/sqrt/exp/log/erf/floor/negative/sin/cos/tan/.../power`
- `tidl.quantize`, `tidl.dequantize`
- Misc: `tidl.nn.depth_to_space`, `tidl.image.resize2d`, `tidl.image.grid_sample`, `tidl.scatter_elements`, `tidl.scatter_nd`

## TIDL Import Pipeline

`TIDLOffloadCompiler.tidl_import()`, once per subgraph:
1. Load `tidl_model_import_relax.so` — path is derived from `tidl_tools_path`'s
   parent dir (the `tidl_relax_so_path` config key has been removed)
2. `TIDL_relaxInit()` — device config + artifacts dir (once, before the
   per-subgraph loop begins)
3. `TIDL_relaxImportInit()` — once per subgraph, before the per-composite loop
4. For each composite: lift constants, construct synthetic `relax.Call`, import node, link edges
5. `TIDL_relaxOptimizeNet()` — run network compiler
6. `TIDL_relaxPostProcessNet()` — write net.bin + params_1.bin

Config: `artifacts_dir`, `tidl_tools_path`, `num_calibration_frames`,
`skip_failing_subgraphs`, `max_subgraphs`, and **`calibration_inputs`**
(required — a list of per-frame `(1,C,H,W)` numpy arrays; `tidl_import()`
raises `ValueError` if omitted, since random calibration data produced
miscalibrated INT8 scales and caused a YOLOv8 DFL NaN on `c7x_dload`).

## Bridge Generation

`generate_bridge()` produces `tidl_bridge.c`. Init is **eager**, not lazy:
`tidl_bridge_init_all()` initializes every subgraph instance up front via
`init_tidl_subgraph()`, and is called once from `cg_main_dsp` when the
codegen target carries `-tidl-runtime=1` (added so all subgraphs are ready
before the first inference request — needed for YOLO offloading). Each
`tidl_subgraph_N_process()` then just:
1. Wrap `void*` in DLTensor structs
2. Flush input cache (`TVM_cacheWbInvRegion`)
3. Call `process_tidl_subgraph(instance, in_tensors, out_tensors)`
4. Invalidate output cache

`tidl_bridge_cleanup()` frees all persistent TIDL handles via
`free_tidl_subgraph()` at module teardown.

## IALG Lifecycle (tidl_api.c)

```
init_tidl_subgraph():
  init_mem_regions(L1/L2/L3) → TIDL_createParamsInit(cp) →
  algNumAlloc → algAlloc → alloc_mem_records → algInit →
  init_inbufs/init_outbufs

process_tidl_subgraph():
  algActivate → connect_input_output_tensors → algProcess →
  disconnect_input_output_tensors → algDeactivate
```

## Hardware Resources (J722S)

| Pool | Size | Source |
|------|------|--------|
| L1 DARAM | firmware-provided | C7x L1 SRAM |
| L2 SRAM | firmware-provided | C7x L2 SRAM |
| L3 (aux L2) | 240 KB (`MSMCSIZE_KB=240`) | Auxiliary L2 |
| DDR | 128 MB | `tvm_dsp_alloc` heap |

DMA: up to 14 DRU channels (via DmaUtilsAutoInc3d). J722S has 16 total.

Device config: `DEVICE_NAME=4` = TIDL_AM62A (C7504 with MMA2, single-core).

## Known Gotchas

1. `appMemAlloc` must use `tvm_dsp_alloc` (128 MB DDR), not RTS heap (128 KB)
2. `traceWriteLevel` must be 0 unless `TIDLWriteBinToFile` callback provided
3. IOBufDesc uses `TIDL_IO_MAX_NUM_CORES=4` (from source header), not SOC-dependent value
4. DLOAD modules must call `appUdmaGetObj()`, never `getUDMADrvObjPtr()`
5. Import `.so` resolves paths relative to CWD
6. **`parse_conv2d_impl()` must set `actParams.actType` for relu/clip variants.**
   The shared implementation handles all 6 conv2d patterns; each relu/clip
   specialization must set `layer.actParams.actType = TIDL_RelU` (or
   `TIDL_Clip` + extract min/max from the inner `relax.clip` call args) after
   calling `parse_conv2d_impl()`. Missing this causes `TIDL_floatSat()` to use
   `TIDL_NoAct` saturation `[-FLT_MAX, FLT_MAX]` in the calibration forward
   pass — intermediate activations grow unboundedly, producing wildly wrong
   INT8 scale factors for all downstream layers. Symptom: calibration stats
   max > 10^6 and max\_diff ~10^7 vs PyTorch. Fixed in `tidl_parse_relax_conv.cpp`
   commit `0a60ecf`; see `docs/dsp/tidl_resnet18_inaccuracy.md`.
7. **`constTensorsDims` must be explicitly set in matmul parsers.**
   `tidl_addConstDataLayers` reads `allowlistingMetaData.constTensorsDims[0]`
   to set `TIDL_ConstDataLayer.outData[0].dimValues`. The optimizer can clear
   this field; if absent the ConstDataLayer gets the model input shape instead
   of the weight shape, and TIDL reads from the wrong params offset. Symptom:
   InnerProduct (FC) output is ~10^7× the correct value. For `tidl.matmul_bias`
   also set `constTensorsDims[1]` for the bias (the bias-size constraint checker
   reads it). Fixed in `tidl_parse_relax_matmul.cpp` commit `312d39b`.

## Testing

Tests at `tests/ti-dsp-runtime/tidl-tests/` and `unit-tests/`:

| File | Tests | Scope | Requirements |
|------|-------|-------|--------------|
| `test_tidl_partition.py` | 17 | Pattern matching + constraint callbacks | None |
| `test_tidl_codegen.py` | 12 | Lowering, TIR stubs, bridge codegen | None |
| `test_tidl_layer_offload.py` | 65 | Per-layer offload unit tests | None |
| `test_tidl_new_ops.py` | 4 | Transformer op patterns (softmax, layernorm) | None |
| `test_tidl_relax_import.py` | 10 | FFI load, init, import pipeline | `tidl_model_import_relax.so` |
| `test_tidl_e2e.py` | 2 | Stub bridge pipeline on c7x_host | TI_CGT_C7000_PATH |
| `test_tidl_import_e2e.py` | 2 | Full TIDL on AM67A hardware | AM67A + .so |
| `test_tidl_resnet_e2e.py` | 3 | ResNet-18 TIDL e2e (fully INT8, all layers on MMA) | AM67A + .so |
| `test_tidl_mv2_e2e.py` | 2 | MV2 TIDL e2e with FC on MMA; calibration health check | AM67A + .so |
| `test_yolo_dsp.py` | 2 | YOLOv5/v8 c_static_lib + TIDL offload (exercises eager bridge init) | AM67A + .so for TIDL variant |
| `unit-tests/test_tidl_large_weight_calib.py` | 2 | Calibration overflow isolation (large-weight conv) | `.so` only |

Markers: `quick` (no hardware, no .so), `core` (dependency-free + c7x_host stub).

```bash
cd tests/ti-dsp-runtime

# Quick tests (partition + codegen + layer offload, no deps, ~90s)
pytest tidl-tests/ -m quick -v

# Core gate (adds c7x_host stub e2e)
pytest tidl-tests/ -m core -v

# Import tests (requires tidl_model_import_relax.so)
pytest tidl-tests/test_tidl_relax_import.py -v

# Full hardware e2e (AM67A)
pytest tidl-tests/test_tidl_import_e2e.py -v
pytest tidl-tests/test_tidl_resnet_e2e.py -v
```

TIDL tests do NOT use `--dsp-mode` — they manage their own build/execution flows.

For general test debugging (DSP_KEEP_TEMP, failure modes), see `relax-c7x:testing`.

## Key Files

| File | Purpose |
|------|---------|
| `python/tvm/relax/backend/tidl/patterns.py` | 74 composite patterns (`get_tidl_patterns()`) + constraints |
| `python/tvm/relax/backend/tidl/tidl.py` | TIDLOffloadCompiler: partition, import, lower, bridge, codegen_and_build |
| `python/tvm/contrib/c7x/io_meta.py` | `write_io_meta()`/`compute_io_meta()` — the `tvm_dsp_io_meta.bin` blob |
| `src/runtime/ti_dsp/tidl/tidl_api.c` | IALG lifecycle on-device |
| `src/runtime/ti_dsp/tidl/tidl_api_mem.c` | appMemAlloc/Free wrappers |
| `tests/ti-dsp-runtime/tidl-tests/` | All TIDL tests |
| `docs/dsp/tidl-subgraph-offloading.md` | Full design document |

## Building TIDL Import Library

```bash
cd ~/ml/c7x-mma-tidl
bash build_j722s.sh          # incremental
bash build_j722s.sh clean    # clean + full rebuild
```

## Related Skills

- `relax-c7x:mmalib-offload` — Alternative offload strategy (mutually exclusive)
- `relax-c7x:cstatic` — Pipeline pass order and code generation
- `relax-c7x:firmware` — Firmware symbol exports for TIDL API
- `relax-c7x:model-workflow` — Choosing TIDL vs MMALIB for a new model
