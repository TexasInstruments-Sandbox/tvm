# TIDL Integration

Offloading Relax subgraphs to TI Deep Learning (TIDL) on the C7x MMA
accelerator, via the `TIDLOffloadCompiler` pipeline in
`python/tvm/relax/backend/tidl/`. Non-offloaded ops remain in TVM and
execute as generated C code on the C7x scalar pipeline.

> **J722S and AM62D only.** TIDL subgraph offload requires firmware built
> with `--tidl ON` (the `j722s-evm`/AM67A default, which also forces
> `--mmalib ON`, as does `am62d`, see below). BeagleY-AI firmware is always
> built `--tidl OFF --mmalib ON` — no TIDL offload path exists on that board. See
> [MMALIB Integration](mmalib-integration.md#firmware-decoupling-mmalib-from-tidl)
> for the firmware-side `USE_TIDL_RUNTIME`/`USE_TI_MMALIB` split, and pass
> `-tidl-kernels=0` when compiling for BeagleY-AI regardless of whether a
> given model uses TIDL subgraph offload at all (it also controls whether
> `max_pool2d` targets the TIDL-backed kernel).

Target string: `c_static_lib -mcpu=c7x` (TIDL artifacts are embedded and
`-tidl-runtime=1` is applied automatically by `TIDLOffloadCompiler.build()`
whenever a model actually produces TIDL subgraphs).

## AM62D

On `--board am62d` (C7504) the same flow applies with AM62A-flavoured TIDL
libraries and the C7504 MMALIB:

- Build the importer and TIDL libs with `c7x-mma-tidl/build_am62d.sh`
  (`TARGET_SOC=AM62A`), then build the firmware with
  `firmware/c7x/dsp/build.sh --board am62d` (`--tidl ON` is the default; it
  needs `C7X_MMA_TIDL_PATH`).
- `TIDLOffloadCompiler` takes the board from `config["board"]` (`j722s-evm`,
  `beagley-ai` or `am62d`; default `j722s-evm`) and maps it to a device
  (`j722s` or `am62d`). The `c_static_lib` target only knows the device: the
  default target gets `-device=<device>`, an explicit target's `-device=` wins,
  and `-device=am62d` defaults `l2-sram-size` to 240 KB. The AM62D toolchain
  and PC libs follow from the device. The dynmod build needs no SDK: the TIDL
  IALG wrapper is in the firmware (see [Firmware-side wrapper](#firmware-side-wrapper)).
- TIDL's "L2" is the 240 KB L2 AUX bank and its "L3" the 1 MB L2 MAIN bank,
  the same assignment as on J722S (see [On-chip memory](#on-chip-memory)).
- Verified on an AM62D EVM: MobileNetV2 (`test_tidl_mv2_e2e.py -k correctness
  --board am62d --dsp-mode=c7x_dload`). On hardware the top two classes swap,
  which the test records as an xfail near-tie (`mean_diff` still checked).
  Other tidl-tests have not been run on AM62D yet.
- Use per-board work directories; a CMake cache reused across boards keeps the
  other board's SDK path and flags.

## Files

| File | Purpose |
|------|---------|
| `python/tvm/relax/backend/tidl/tidl.py` | `TIDLOffloadCompiler`: partition, import, lower, bridge generation, build |
| `python/tvm/relax/backend/tidl/patterns.py` | TIDL `FusionPattern` definitions (74 patterns); structure only, no constraint logic |
| `python/tvm/relax/backend/tidl/_gate.py` | The eligibility gate: rebuilds the composite for each match and asks `TIDL_relaxAllowNode` in `tidl_model_import_relax.so` |
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
_fix_output_dtype_  Compares each subgraph's actual output element type/
  mismatches()      scale/zero-point (read from params_1.bin) against what
    |               its TIR stub declares. On mismatch, re-declares the
    |               stub's output buffer and the call_tir result with
    |               TIDL's real dtype (so the storage matches the bytes
    |               TIDL writes, e.g. 1 byte per argmax index) and inserts
    |               a dequant/cast in @main. No-op if nothing mismatches.
    v
generate_bridge()   Generate tidl_bridge.c/h resolving the extern calls
  + relax.build()    c_static_lib codegen emits lib0.c + weights.bin.
    |               relax.build()'s finalize passes also run
    |               PadTIDLOutputStorage here, growing an output's
    |               allocated storage to cover TIDL's write-past-the-
    |               tensor padding channels (no-op for non-TIDL builds).
    v
_build_dynmod()     TI C7x cross-compile; embeds TIDL artifacts as .rodata;
    |               links the bridge, which imports init/process/
    |               free_tidl_subgraph from the firmware
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

## How offload eligibility is decided

Eligibility is decided in exactly one place, `TIDL_relaxAllowNode` in
`tidl_model_import_relax.so`. The Python patterns match structure only; every
pattern's check (`_gate.tidl_gate(name)`) rebuilds the composite function that
`FuseOpsByPattern` would create, wraps it in the synthetic call the importer
uses, and asks the library. The rules themselves are in the C++ Relax parsers
(`tidlParseRelax/`, mostly `tidl_parse_relax_declines.cpp`) plus TIDL's shared
`tidlAllowlistingConstraints/`. Consequences for contributors:

- To change what is offloaded, change the C++ and rebuild the `.so`
  (`c7x-mma-tidl/build_j722s.sh`). Do not add Python constraint logic.
- `partition()` needs the `.so` and fails with a clear error without it.
  Tests that only check pattern structure set
  `config["allow_fn"] = lambda call: 1`.
- Why an op stayed on TVM: `_gate.recording()` returns a `GateResult` per
  match with `.verdict` and `.reason` (the library's message, plus
  `[constraint: ...]` when a TIDL constraint fired).
- `TIDL_relaxDescribeNode(call)` returns what the parser populated (dims,
  dtypes, axis, strides, pad, slice points) for testing accepted ops.
- Deny lists by layer name use `<pattern>:<root variable name in the input IR>`.

Rules are tested by a decline table (`test_tidl_declines.py`), a constraint
coverage gate against TIDL's own constraint headers
(`test_tidl_constraint_coverage.py`), partition snapshots
(`test_tidl_partition_snapshot.py`), and numeric hardware checks of accepted
configurations (`test_tidl_declines_hw.py`).

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
| Advanced | `image.resize2d`, `take`, `topk`, `split`, `nn.pixel_shuffle` (depth-to-space), `broadcast_to` (expand), `scatter_elements`, `scatter_nd` |
| Math/unary | `abs`, `sqrt`, `power`, `exp`, `log`, `erf`, `floor`, `negative`, `sin`/`cos`/`tan`, `sinh`/`cosh`, `asin`/`acos`/`atan`/`asinh` |
| Quantization (stubs) | `quantize`, `dequantize` |

`image.grid_sample` is offloaded only with a constant grid: the importer transposes the
0.27 grid layout `[N,2,H,W]` into TIDL's `[N,H,W,2]`. A variable grid runs on the TVM lowering.

Ops not in this list stay on the TVM scalar path automatically — nothing
needs to be excluded explicitly.

## Configuration

`TIDLOffloadCompiler(config={...})` accepts:

| Key | Default | Description |
|-----|---------|--------------|
| `artifacts_dir` | per-process temp dir, removed at exit | Output directory for TIDL binaries; set it to keep them |
| `tidl_tools_path` | `$C7X_MMA_TIDL_PATH/tidl_tools`, else the copy bundled in the wheel | Path to the TIDL import tools and `device_config.cfg` |
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

### On-chip memory

`tidl_support_init()` (`firmware/c7x/dsp/src/tidl_support.c`) hands TIDL's
IALG allocator three regions, following the vision_apps layout
(`app_cfg_c7x_1.h`) on both SoCs:

| TIDL region | Bank | J722S | AM62D |
|-------------|------|-------|-------|
| L1 | L2 AUX as L1, `0x7F03C000` | 16 KB | 16 KB |
| L2 | L2 AUX, `0x7F000000` | 224 KB (`L2MEMSIZE_KB` in `j722s_config.cfg`) | 240 KB |
| L3 | L2 MAIN, `0x7E000000` | 1.25 MB | 1 MB |

Both banks are shared with TVM. That is safe for the generated-code L2
scratch in MAIN (`tvm_l2_alloc`), which is reset at the start of every
kernel and never lives across a TIDL call. It is not safe for the runtime's
fast pool in AUX, where storage of 32 KB or less normally goes: storage
outlives a single op, so a small tensor produced before a TIDL subgraph and
read after it would be overwritten by TIDL's scratch. When the firmware loads
a module that exports `tidl_bridge_cleanup`, it calls
`TVMDSPStorageSetFastPoolEnabled(0)` so all of that module's storage goes to
DDR, and turns the fast pool back on at unload. Modules without TIDL keep
using it.

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
- **Host and hardware are expected to match.** A large gap between
  `c7x_host` and `c7x_dload`, or a subgraph that is wrong on both, has so far
  always been an import or memory bug rather than int8 rounding (missing
  multi-output buffers, rank < 4 tensors laid out differently at the
  subgraph boundary, TVM tensors sharing a bank with TIDL scratch). Check
  `numOutputBuf` in the subgraph's `params_1.bin` against its output count
  and run the subgraph alone before suspecting calibration;
  `tidl-tests/test_tidl_subgraph_io.py` covers the known cases.
- **Element-wise ops require rank exactly 4.** A sub-4D use (e.g. an FC
  bias add with shape `(1, N)`) is rejected by the pattern's constraint
  check rather than being offloaded and crashing at runtime.
- **`topk` always returns both values and indices.** Request
  `ret_type="both"` and extract index `[0]` for values-only usage.
- **The dtype gate is one blanket allowlist, not per-op.** Every pattern's
  constraint check accepts only {float32, int8, int16, uint8} dtypes, with
  no op-specific exceptions. A dtype that some reference TIDL importer
  permits for a specific op (e.g. int32 as `take`'s data input, or
  int32/int64 for an elementwise `mod`) is still declined here, since the
  allowlist doesn't vary per op. An op whose rank/shape cannot be
  statically determined is declined too, rather than optimistically
  accepted — this trades offload coverage on dynamic-shape models for not
  risking a mis-parse.
- **`strided_slice` only supports a single axis with unit stride.**
  Multi-axis `axes`/`begin`/`end`/`strides`, or any stride other than 1,
  is declined rather than silently executed with the wrong bounds.
- **`nn.pad` only supports constant-mode, 4-D, H/W-only padding.** Any
  non-`"constant"` `pad_mode`, a non-4-D input, or non-zero N/C (batch/
  channel) padding is declined.
- **`split` is capped at 16 output tensors.** TIDL's internal slice-points
  buffer is fixed-size; a larger split is declined rather than overflowing
  it.
- **`take` declines when `data` (not just `indices`) is a constant.** The
  common inverse case — a constant embedding table gathered by variable
  token ids — offloads fine; a fully-constant gather does not.

## Testing

```bash
cd tests/ti-dsp-runtime

# Partition + codegen — needs the .so (eligibility), no hardware
pytest tidl-tests/test_tidl_partition.py tidl-tests/test_tidl_codegen.py -v

# Per-layer partition tests — needs the .so, no hardware
pytest tidl-tests/test_tidl_layer_offload.py -k TestLayerPartition -v

# Eligibility rules: decline table, constraint coverage, partition snapshots
pytest tidl-tests/test_tidl_declines.py tidl-tests/test_tidl_constraint_coverage.py \
  tidl-tests/test_tidl_partition_snapshot.py tidl-tests/test_tidl_allownode.py -v

# Numeric check of accepted configs on AM67A (serial; one board)
pytest tidl-tests/test_tidl_declines_hw.py --dsp-mode=c7x_dload --board j722s-evm -v

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

## Firmware-side wrapper

The IALG wrapper `src/runtime/ti_dsp/tidl/tidl_api.c` (with `tidl_api_mem.c`
and `ti_mem_manager.c`) is compiled into the `c7x_compute` firmware when it is
built `--tidl ON`, and exported through the DLOAD symbol table as
`init_tidl_subgraph`, `process_tidl_subgraph` and `free_tidl_subgraph`. A
TIDL `lib0.out` holds only the generated bridge, the embedded `tidl_net_N` /
`tidl_io_N` blobs and `tidl_bridge_cleanup`, so building it needs neither
`itidl_ti.h` (c7x-mma-tidl) nor `ivision.h` (PSDK).

Consequences:

- TIDL modules need firmware service version 4.2.0 or later. On older
  firmware DLOAD fails to resolve `init_tidl_subgraph`.
- The module-to-firmware contract is the three signatures in `tidl_api.h` plus
  the `tvm_tidl_rt_info` layout.
- Changing `tidl_api.c` (tracing, `flowCtrl`, `TIDL_COPY_NETWORK_BUF`) on
  hardware needs a firmware rebuild, `deploy-c7x.sh` and a reboot, not a model
  recompile. `c7x_host` still compiles `tidl_api.c` per build.
- The wrapper's pool state (`init_mem_regions`, scratch pointers) lives for the
  firmware's lifetime, shared by every loaded module, as the physical L1/L2/L3
  regions already were.

## Wheel-only compiles

A wheel built with `--tidl ON` bundles what the import and compile steps used
to read from a c7x-mma-tidl tree:

| Bundled under `tvm/data/ti_dsp/tidl/` | Replaces |
|---|---|
| `tidl_model_import_relax.so`, `tidl_model_custom_import.so` | `ti_dl/utils/tidlModelImport/out/` |
| `tidl_tools/` (`PC_dsp_test_dl_algo.out`, `ti_cnnperfsim.out`, `device_config.cfg`) | `$C7X_MMA_TIDL_PATH/tidl_tools` |
| `sTIDL_IOBufDesc_t.layout.json` | parsing `itidl_io.h` to size TIDL outputs |

`build_wheel.sh` takes all of them from the same c7x-mma-tidl tree, so the
layout table matches the `.so` that writes `params_1.bin`. Explicit settings
still win: `config["tidl_tools_path"]` and `C7X_MMA_TIDL_PATH` are checked
before the bundled tools, and an explicit `header_path` before the bundled
table. A missing layout, a missing `params_1.bin` or a layout/blob size
mismatch fails the compile; earlier versions only logged a warning and shipped
the model without TIDL output padding.
