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
  → Output Dtype Fixup (_fix_output_dtype_mismatches(): compares each
    subgraph's actual output element type/scale/zero-point, read from
    params_1.bin, against what its TIR stub declares; inserts a dequant/
    cast in @main at the call_tir site on mismatch — see README.md's
    "Pipeline" section for the exact formula and known pitfalls)
  → c_static_lib Codegen (emit lib0.c + weights.bin; sets `-tidl-runtime=1` target attr;
    relax.build()'s finalize passes run PadTIDLOutputStorage here too, growing an
    output's allocated storage to cover TIDL's write-past-the-tensor padding channels)
  → IO Meta (write_io_meta() → tvm_dsp_io_meta.bin, declared input_buf/
    output_buf capacity for the per-buffer-dmabuf c7x_dload protocol)
  → Bridge Generation (tidl_bridge.c: eager tidl_bridge_init_all() + per-subgraph process() + tidl_bridge_cleanup())
  → C7x DLOAD Build (lib0.out with embedded TIDL artifacts + IO-meta blob)
  → Deploy to AM67A via c7x_compute
```

Both the Output Dtype Fixup and PadTIDLOutputStorage steps are no-ops when
there is nothing for them to do (no detected mismatch; no module attribute
set, respectively) — they only touch the generated IR when a subgraph
output actually needs it.

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
before looking anywhere else. See `docs-c7x/contributor-guide/firmware/architecture.md` ("Per-Buffer dmabufs") for the
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

## Who decides what can be offloaded

**Exactly one place: `TIDL_relaxAllowNode` in `tidl_model_import_relax.so`.**
`patterns.py` matches *structure only* (which Relax ops, in which arrangement);
every pattern's check is `_gate.tidl_gate(name)` (`_gate.py`), which rebuilds the
composite function `FuseOpsByPattern` would create for the match, wraps it in the
same synthetic call `tidl_import` uses, and asks the library. There is no
Python-side constraint logic: dtype, rank, shape, attribute, constant-vs-variable
and hardware-limit rules all live in the C++ Relax parsers
(`tidlParseRelax/`, chiefly `tidl_parse_relax_declines.cpp`) and TIDL's shared
`tidlAllowlistingConstraints/` headers. **To change what is offloaded, change
the C++ and rebuild the `.so` (`c7x-mma-tidl/build_j722s.sh`); do not add a
`_check_*` to `patterns.py`.**

Consequences:

- **Partitioning needs the `.so`.** Without it `partition()` raises a clear
  error (fail closed). Structure-only tests pass
  `config["allow_fn"] = lambda call: 1`.
- **Rejection reasons** are available: `_gate.recording()` yields a `GateResult`
  per pattern-check call (`.verdict`, `.reason`, the synthesised `.composite`);
  `TIDL_relaxGetLastRejectReason` returns the last one, including
  `[constraint: <TIDL catalogue text>]` when TIDL's own constraint failed.
- **`TIDL_relaxDescribeNode(call)`** returns the parsed layer as JSON (metadata
  dims/dtypes, and layer params such as axis/stride/pad/slice points). Use it to
  test that an *accepted* op was parsed correctly: a wrong axis is accepted and
  only visible there.
- **Name lists:** the gate stamps `tidl_node_name` =
  `<pattern>:<root var name_hint in the input IR>` so `deny_list:layer_name`
  works. Names are not guaranteed unique (Relax reuses `lv`).
- `partition()` runs `CanonicalizeBindings` first so a weight bound as
  `w = R.const(...)` is inlined into the call; otherwise fusion makes it a
  composite *parameter* and TIDL declines the (then variable) weight.

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
- Misc: `tidl.nn.depth_to_space`, `tidl.image.resize2d`, `tidl.scatter_elements`, `tidl.scatter_nd`, `tidl.image.grid_sample` (constant grid only: the importer transposes Relax's `[N,2,H,W]` grid to TIDL's `[N,H,W,2]`; a variable grid is declined)

## TIDL Import Pipeline

`TIDLOffloadCompiler.tidl_import()`, once per subgraph:
1. Load `tidl_model_import_relax.so` — `TIDL_RELAX_SO_PATH`, the wheel's bundled copy,
   then `tidl_tools_path`'s parent dir / `C7X_MMA_TIDL_PATH`. `tidl_tools_path` itself
   resolves `config` → `$C7X_MMA_TIDL_PATH/tidl_tools` → bundled `tvm/data/ti_dsp/tidl/tidl_tools`
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

Output sizing reads `params_1.bin` via `io_buf_desc.py`, using the wheel's
`sTIDL_IOBufDesc_t.layout.json` (generated from `itidl_io.h` by `build_wheel.sh`) or, in a source
tree, `itidl_io.h` under `C7X_MMA_TIDL_PATH`. Missing layout / descriptor or a size mismatch is a
compile error (it used to warn and skip output padding).

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

| Pool | Region | Size |
|------|--------|------|
| L1 DARAM | L2 AUX as L1, `0x7F03C000` | 16 KB |
| L2 SRAM | L2 AUX, `0x7F000000` | 224 KB (`L2MEMSIZE_KB` in `j722s_config.cfg`) |
| L3 | L2 MAIN, `0x7E000000` | 1.25 MB |
| DDR | `tvm_dsp_alloc` heap | 352 MB |

This is the vision_apps layout (`app_cfg_c7x_1.h`); with L2/L3 the other way
round a large YOLO subgraph was wrong on the device. L2 AUX is also the runtime's
fast pool (storage <= 32 KB), so the firmware turns that pool off while a TIDL
module is loaded (`TVMDSPStorageSetFastPoolEnabled`): a small TVM tensor live
across a TIDL call would otherwise be overwritten.

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
   commit `0a60ecf`.
7. **`constTensorsDims` must be explicitly set in matmul parsers.**
   `tidl_addConstDataLayers` reads `allowlistingMetaData.constTensorsDims[0]`
   to set `TIDL_ConstDataLayer.outData[0].dimValues`. The optimizer can clear
   this field; if absent the ConstDataLayer gets the model input shape instead
   of the weight shape, and TIDL reads from the wrong params offset. Symptom:
   InnerProduct (FC) output is ~10^7× the correct value. For `tidl.matmul_bias`
   also set `constTensorsDims[1]` for the bias (the bias-size constraint checker
   reads it). Fixed in `tidl_parse_relax_matmul.cpp` commit `312d39b`.
8. **Eligibility rules carried over from the Python checks are the stricter
   bound, on purpose.** The ported rules keep the former Python limits (data
   types float32/int8/int16/uint8 only; elementwise ops exactly 4-D; most other
   ops rank <= 6) even where TIDL's shared constraints are looser, so moving
   them never widened what gets offloaded. They are correctness/safety gates,
   not just missing optimizations (strided_slice single-axis/unit-stride, pad
   constant-mode/4-D/no-N-C-padding, split <= 16 outputs, take non-constant
   data, resize method, expand_dims output rank): don't loosen one without
   re-reading why it exists, and give any new accept path a numeric test
   (`test_tidl_declines_hw.py`).
9. **A parser that accepts by omission is the real bug class.** Three found by
   the decline table: `pad` read `pad_value` from a constant operand (it is an
   attribute), so any value imported as 0; `resize2d` left the nearest-neighbour
   default for an unknown method (cubic imported as nearest); the softmax
   parser counted dims by indexing `varTensorsDims[0]` up to 6 (past the end of
   a 4-D shape) so a non-negative axis was normalised with the wrong rank.
   Add a case to `test_tidl_declines.py` for every decline, and an accepting
   case with `TIDL_relaxDescribeNode` params for every field a decline protects.
    Also found this way: non-global `avg_pool2d` could never parse; the deconv
    parser never set `padW/padH` (TIDL's 4x4 rule reads them), and TIDL deconv
    needs out == in * stride; `take` copied int32 constant indices through
    `NDArrtoFloat`, so only index 0 was right (copy integer constants raw).
10. **The Relax `.so` does not compile `tidlParseTVM/`.** Anything shared with
    the Relay parsers must live in a file both targets build
    (`tidl_import_common_model_check.cpp`); an `extern` to a Relay-only symbol
    links fine and aborts the process at first call (lazy binding). Also, the
    makefile does not track header dependencies for `tidlParseRelax/*.cpp`:
    after editing `tidl_parse_relax.h` run `touch tidlParseRelax/*.cpp` before
    `build_j722s.sh`, or stale objects keep the old macro/layout.

## Testing

Tests at `tests/ti-dsp-runtime/tidl-tests/` and `unit-tests/`:

| File | Tests | Scope | Requirements |
|------|-------|-------|--------------|
| `test_tidl_partition.py` | 17 | Pattern matching + eligibility through the gate | `.so` |
| `test_tidl_codegen.py` | 12 | Lowering, TIR stubs, bridge codegen | None |
| `test_tidl_layer_offload.py` | 65 | Per-layer offload unit tests | `.so` |
| `test_tidl_declines.py` | 152 | Accept/reject table per eligibility rule; `DescribeNode` parse checks | `.so` |
| `test_tidl_constraint_coverage.py` | 5 | TIDL constraint inventory vs. tests; ratchet | `.so` |
| `test_tidl_allownode.py` | 63 | Gate seam: composite fidelity, driver declines, name lists, fail-closed | `.so` |
| `test_tidl_partition_snapshot.py` | 58 | Partition of 54 layer models + ResNet-18/MV2/YOLOv5n/v8n vs. committed snapshots | `.so` |
| `test_tidl_declines_hw.py` | 24 | Accepted configs vs. a TVM CPU reference on AM67A | AM67A + `.so` |
| `test_tidl_new_ops.py` | 4 | Transformer op patterns (softmax, layernorm) | None |
| `test_tidl_relax_import.py` | 10 | FFI load, init, import pipeline | `tidl_model_import_relax.so` |
| `test_tidl_e2e.py` | 2 | Stub bridge pipeline on c7x_host | TI_CGT_C7000_PATH |
| `test_tidl_import_e2e.py` | 2 | Full TIDL on AM67A hardware | AM67A + .so |
| `test_tidl_resnet_e2e.py` | 3 | ResNet-18 TIDL e2e (fully INT8, all layers on MMA) | AM67A + .so |
| `test_tidl_mv2_e2e.py` | 2 | MV2 TIDL e2e with FC on MMA; calibration health check | AM67A + .so |
| `test_yolo_dsp.py` | 2 | YOLOv5/v8 c_static_lib + TIDL offload (exercises eager bridge init) | AM67A + .so for TIDL variant |
| `test_tidl_subgraph_io.py` | 3 | Multi-output buffers, rank-3 concat of reshapes, small tensor live across a TIDL call | `.so`; the last one only bites on AM67A |
| `unit-tests/test_tidl_large_weight_calib.py` | 2 | Calibration overflow isolation (large-weight conv) | `.so` only |

Markers: `quick` (no hardware), `core` (+ c7x_host stub). Most `tidl-tests` need `tidl_model_import_relax.so` to partition (`conftest.py` skips them when it is not built); `C7X_MMA_TIDL_PATH` defaults to `~/ml/c7x-mma-tidl`.

```bash
cd tests/ti-dsp-runtime

# Quick tests (partition + codegen + layer offload + decline table, ~2 min; needs the .so)
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
| `python/tvm/relax/backend/tidl/patterns.py` | composite patterns (`get_tidl_patterns()`), structure only |
| `python/tvm/relax/backend/tidl/_gate.py` | the eligibility gate: asks `TIDL_relaxAllowNode` about each match |
| `c7x-mma-tidl/.../tidlParseRelax/tidl_parse_relax_declines.cpp` | the eligibility rules (one per composite) |
| `python/tvm/relax/backend/tidl/tidl.py` | TIDLOffloadCompiler: partition, import, lower, bridge, codegen_and_build |
| `python/tvm/contrib/c7x/io_meta.py` | `write_io_meta()`/`compute_io_meta()` — the `tvm_dsp_io_meta.bin` blob |
| `src/runtime/ti_dsp/tidl/tidl_api.c` | IALG lifecycle on-device |
| `src/runtime/ti_dsp/tidl/tidl_api_mem.c` | appMemAlloc/Free wrappers |
| `tests/ti-dsp-runtime/tidl-tests/` | All TIDL tests |
| `docs-c7x/contributor-guide/backend/tidl-integration.md` | Full design document |

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
