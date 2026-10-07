# TIDL Offloading Tests

Tests for TIDL subgraph offloading in the TVM/Relax c_static_lib backend.

## Tests

| Test | Count | What it tests | Requirements |
|------|-------|---------------|--------------|
| `test_tidl_partition.py` | 17 | Pattern matching, partitioning, eligibility through the gate | `tidl_model_import_relax.so` |
| `test_tidl_codegen.py` | 12 | Lowering pass, TIR stubs, c_static_lib codegen, bridge generation (single + multi-subgraph, stub + real) | `.so` (partitioning) |
| `test_tidl_relax_import.py` | 18 | FFI load, init, AllowNode, tidl_import() pipeline | `tidl_model_import_relax.so` + c7x-mma-tidl tree |
| `test_tidl_layer_offload.py` | 103 | Per-layer offload validation: pattern matching (Level 1, 61 tests) and hardware inference (Level 4, 42 tests) covering all supported layer types | Level 1: `.so` (partitioning); Level 4: + `TI_CGT_C7000_PATH` + AM67A |
| `test_tidl_new_ops.py` | 4 | Newer composite ops (softmax, multiply, permute_dims, concat) through full build + AM67A pipeline | `.so` + `TI_CGT_C7000_PATH` + AM67A |
| `test_tidl_allownode.py` | 63 | The eligibility gate: composite fidelity (synthesised == fused), `TIDL_relaxDescribeNode`, driver declines, name lists, fail-closed without the `.so` | `.so` |
| `test_tidl_declines.py` | 253 | Accept/reject table for every eligibility rule and every reachable TIDL constraint (including `tensor_bits` and `c7x_firmware_version` gates), with parsed-layer (`DescribeNode`) checks on accepted cases | `.so` |
| `test_tidl_constraint_coverage.py` | 5 | Inventory of TIDL's own allowlisting constraints vs. the decline table (covered / subsumed / unreachable-with-reason); fails on any constraint left without one | `.so` + c7x-mma-tidl tree |
| `test_tidl_partition_snapshot.py` | 58 | Partition of 54 layer models + ResNet-18, MobileNetV2, YOLOv5n, YOLOv8n vs. `snapshots/partition_snapshots.json` | `.so` |
| `test_tidl_declines_hw.py` | 29 | Accepted configurations run on hardware and compared with a TVM CPU reference | `.so` + AM67A (`--dsp-mode=c7x_dload --board j722s-evm`) |
| `test_tidl_e2e.py` | 2 | Full pipeline on c7x_host: stub bridge (no TIDL libs) and real bridge (TIDL PC/AVX reference path) | `TI_CGT_C7000_PATH`; real-bridge test also needs `.so` + PC TIDL algo libs |
| `test_tidl_import_e2e.py` | 2 | `compiler.build()` one-call pipeline -> deploy -> run on AM67A (single-subgraph ConvReluSoftmax + multi-subgraph 2-conv model) | `.so` + `TI_CGT_C7000_PATH` + AM67A |
| `test_tidl_resnet_e2e.py` | 3 | ResNet-18 TIDL build pipeline validation, hardware correctness, cycle comparison | `.so` + `TI_CGT_C7000_PATH` (build test); + AM67A (hardware tests) |
| `test_tidl_mv2_e2e.py` | 2 | MobileNetV2 TIDL build + hardware correctness -- calibration infrastructure check against ResNet-18's Bug 4 | `.so` + `TI_CGT_C7000_PATH` (build test); + AM67A (correctness test) |
| `test_yolo_dsp.py` | 6 | YOLOv5/YOLOv8 (n, s) c_static_lib DSP and TIDL offloading tests | `TI_CGT_C7000_PATH`; TIDL variants need `.so` + AM67A |
| `diag_tidl_levels.py` | -- | Standalone multi-level TIDL init debug script | `TI_CGT_C7000_PATH` + artifacts + AM67A |

```bash
# Run partition + codegen tests (need the .so, no hardware):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_partition.py \
       tests/ti-dsp-runtime/tidl-tests/test_tidl_codegen.py -v

# Per-layer partition tests — need the .so, no hardware:
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_layer_offload.py \
       -k TestLayerPartition -v

# Eligibility rules: decline table, constraint coverage, partition snapshots, gate seam:
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_declines.py \
       tests/ti-dsp-runtime/tidl-tests/test_tidl_constraint_coverage.py \
       tests/ti-dsp-runtime/tidl-tests/test_tidl_partition_snapshot.py \
       tests/ti-dsp-runtime/tidl-tests/test_tidl_allownode.py -v

# Numeric check of accepted configs on AM67A (serial; one board):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_declines_hw.py \
       --dsp-mode=c7x_dload --board j722s-evm -v

# Import tests (needs tidl_model_import_relax.so):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_relax_import.py -v

# Per-layer hardware tests on AM67A (~10min, all layer types):
TI_CGT_C7000_PATH=~/ti/.../ti-cgt-c7000_5.0.1.LTS \
  pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_layer_offload.py \
         -k TestLayerHardware -v

# Hardware e2e tests (needs .so + C7x compiler + AM67A):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_import_e2e.py -v -s
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_resnet_e2e.py -v -s

# ResNet-18 build-only test (no AM67A needed, validates full pipeline):
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_resnet_e2e.py::TestTIDLResNetE2E::test_tidl_resnet18_build -v

# Generate TIDL offloading visualization:
python tests/ti-dsp-runtime/tidl-tests/test_tidl_resnet_e2e.py \
    --visualize resnet18_tidl.html

# Standalone TIDL diagnostic (levels 0/1/1b/2/3):
python tests/ti-dsp-runtime/tidl-tests/diag_tidl_levels.py 3
```

## Library-gated tests

Offload eligibility is decided by `TIDL_relaxAllowNode` in
`tidl_model_import_relax.so`, so partitioning needs it: build it with
`c7x-mma-tidl/build_j722s.sh` (CI must do this before `tidl-tests`). **The
`quick` and `core` tiers are therefore no longer `.so`-free.**

`conftest.py` defaults `C7X_MMA_TIDL_PATH` to `~/ml/c7x-mma-tidl` and skips the
library-dependent tests, rather than failing, when the `.so` is absent. Two
mechanisms do that:

- modules listed in `_NEEDS_SO_MODULES` (`conftest.py`) are skipped at
  collection — these are the ones that partition without gating themselves;
- modules with their own `.so` skipif or autouse skip fixture
  (`test_tidl_allownode.py`, `test_tidl_declines*.py`,
  `test_tidl_partition_snapshot.py`, `test_tidl_relax_import.py`,
  `test_tidl_new_ops.py`, `test_tidl_import_e2e.py`,
  `test_tidl_output_storage.py`) are not listed.

A test that only checks pattern *structure* can pass
`allow_fn=lambda call: 1` in the compiler config and come off that list.

Other files in this directory used by the eligibility tests:

- `partition_snapshot.py`, `diag_tidl_partition_snapshot.py`,
  `snapshots/` - partition snapshots. Any difference must be classified; see
  `snapshots/README.md`.
- `constraint_coverage.py`, `diag_tidl_constraint_coverage.py`,
  `constraint_coverage.json` - TIDL's constraint inventory and its test map.
- `diag_tidl_gate_spike.py` - gate cost and composite-fidelity on real models.

## Per-layer Tests (`test_tidl_layer_offload.py`)

Isolated validation for each supported layer type, organized into two
test classes.  For constraint details (axis restrictions, calibration
behavior for math ops, etc.) see
`python/tvm/relax/backend/tidl/README.md`.

### Level 1 — Partition (`TestLayerPartition`, 61 tests)

Pattern-matching tests.  Partitioning asks `tidl_model_import_relax.so`
which ops may be offloaded, so the `.so` is required; no hardware or TI
compiler.  Runs in seconds.

Each test builds a minimal Relax IRModule with the target op, calls
`partition_for_tidl()`, and asserts the expected composite name appears.

Includes **constraint rejection** tests:
- `test_reduce_multi_axis_rejected` — multi-axis reduction rejected
- `test_divide_rejects_rank2` — sub-4D element-wise ops rejected

### Level 4 — Hardware (`TestLayerHardware`, 42 tests)

End-to-end tests running the full TIDL pipeline on AM67A:
`partition → tidl_import → lower → codegen → bridge → build → run_dsp_dload`

Requirements: `tidl_model_import_relax.so`, `TI_CGT_C7000_PATH`, AM67A.

| Test group | Tests |
|---|---|
| Activations | sigmoid, tanh, clip, leakyrelu, elu, hard_sigmoid, hard_swish, mish |
| Element-wise | subtract, maximum, minimum |
| Reductions | sum, reduce_max, argmax, argmin |
| Advanced | resize2d, strided_slice, strided_slice_to_end, global_avg_pool (via mean), permute_dims, concat, topk, split, depth_to_space, conv2d_transpose |
| Math/unary | abs, sqrt, exp, log, erf, floor, negative, sin, cos, tan, sinh, cosh, asin, acos, atan, asinh, power |

All hardware tests assert `np.isfinite(output).all()`.  Math/unary ops
with unbounded output range (`exp`, `sinh`, `cosh`, `power`) may produce
all-zero output with random calibration data — this validates the
import/codegen pipeline without depending on calibration quality.

## TIDL Artifacts

TIDL subgraphs need compiled binary artifacts (`net.bin` +
`params_1.bin`).  Artifacts are generated at runtime by
`tidl_import()` (the Relax FFI import path) using
`tidl_model_import_relax.so`.

### Artifact naming

TIDL's default naming convention (used by both the standalone import
tool and the Relax FFI):

| File | Description |
|------|-------------|
| `subgraph{id}_net.bin` | Network topology + weights (~1.3 MB with MMA code) |
| `subgraph{id}_params_1.bin` | I/O buffer descriptors (~378 KB) |

### Version compatibility

The import `.so`, calibration tool, network compiler, and firmware
algo libs must all be built from the same `c7x-mma-tidl` source tree
with `TARGET_SOC=J722S` to ensure struct layout and version
consistency.

### Device config (J722S/AM67A)

The device config is at
`c7x-mma-tidl/ti_dl/test/testvecs/config/import/device_config.cfg`
(symlinked to `device_configs/j722s_config.cfg`):

```
L2MEMSIZE_KB    = 224
MSMCSIZE_KB     = 240
DEVICE_NAME     = 4
```

`MSMCSIZE_KB` controls the L3 scratch buffer size baked into the
network binary.  J722S has no MSMC SRAM; we map 240 KB of auxiliary
L2 as the L3 pool.  Setting this to 2048 (the default) causes TIDL
to request 2 MB of SRAM that does not exist.

### Known issues

- **IOBufDesc header mismatch**: The c7x-mma-tidl source header
  (`itidl_io.h`) uses `TIDL_IO_MAX_NUM_CORES=4` for the IOBufDesc
  struct layout.  The PSDK header uses SOC-dependent
  `TIDL_MAX_NUM_CORES` (2 for J722S).  The DLOAD module
  (`CMakeLists.txt`) must include the c7x-mma-tidl source header,
  not the PSDK header, to match the import tool's artifact layout.

- **CWD sensitivity**: The import `.so` uses hardcoded relative paths
  (`../../test/`, `../../utils/perfsim/`).  The `tidl_import()` FFI
  handles this by setting the correct CWD.

- **72 MB artifacts**: If `ti_cnnperfsim.out` is not built, artifacts
  are ~72 MB (generic code) instead of ~1.3 MB (MMA optimized).
  `build_j722s.sh` (see "Building TIDL Dependencies" below) builds it
  as part of the normal flow.

- **device_config.cfg symlink**: Must exist at
  `ti_dl/test/testvecs/config/import/device_config.cfg` pointing to
  `device_configs/j722s_config.cfg`.  Always build via `build_j722s.sh`
  (see "Building TIDL Dependencies" below), which creates this
  automatically.

## Diagnostic Script

`diag_tidl_levels.py` is a standalone script for debugging TIDL init
on AM67A hardware.  It provides multiple diagnostic levels:

| Level | What it does |
|-------|-------------|
| `0` | Stub bridge only -- no TIDL calls, verifies module load/run |
| `1` | Calls `algNumAlloc` + dumps IALG function table addresses |
| `1b` | Calls `algAlloc` directly with TIDL headers and NULL callbacks |
| `2` | Full `init_tidl_subgraph` via `tidl_api.c` |
| `3` | Full `process_tidl_subgraph` (real int8 inference on MMA) |

```bash
export TI_CGT_C7000_PATH=~/ti/.../ti-cgt-c7000_5.0.1.LTS
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Start with level 0 and work up
python tests/ti-dsp-runtime/tidl-tests/diag_tidl_levels.py 0
python tests/ti-dsp-runtime/tidl-tests/diag_tidl_levels.py 1
python tests/ti-dsp-runtime/tidl-tests/diag_tidl_levels.py 1b
python tests/ti-dsp-runtime/tidl-tests/diag_tidl_levels.py 2
```

If the DSP hangs, read the remoteproc trace buffer from A53:
```bash
ssh root@am67a cat /sys/kernel/debug/remoteproc/remoteproc0/trace0
```

## Building TIDL Dependencies from Source

c7x-mma-tidl is a separate internal repository (Bitbucket) with its
own single build script, `build_j722s.sh` -- the same one the Jenkins
pipeline runs (see `Jenkinsfile`, stage "TIDL Build (c7x-mma-tidl)").

### Prerequisites

```bash
export PSDK_INSTALL_PATH=/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS   # optional, see below
export C7X_MMA_TIDL_PATH=~/ml/c7x-mma-tidl                    # where to clone it
export TVM_HOME=~/ml/tvm                                      # this checkout
```

### Clone and build

```bash
git clone --branch tvm-relax --recurse-submodules --shallow-submodules \
    ssh://git@bitbucket.itg.ti.com/mctools/c7x-mma-tidl.git \
    $C7X_MMA_TIDL_PATH
cd $C7X_MMA_TIDL_PATH
bash build_j722s.sh clean    # clean + full rebuild; omit "clean" for incremental
```

This builds all six components (PC algo libs, DSP algo libs,
calibration tool, network compiler, import tools, and
`tidl_model_import_relax.so`) and creates the `device_config.cfg`
symlink automatically for J722S. `RELAX_TVM_HOME` (defaults to
`TVM_HOME`) controls which TVM fork the Relax `.so` is compiled
against.

### Wire up firmware

The firmware build picks up `C7X_MMA_TIDL_PATH` directly -- no manual
symlinking needed (see `USE_TIDL_RUNTIME` in
`src/runtime/ti_dsp/firmware/c7x/dsp/CMakeLists.txt`). With the env
vars above still exported:

```bash
cd src/runtime/ti_dsp/firmware/c7x/dsp
./build.sh --board j722s-evm
```

### Verify

```bash
# Run partition + codegen + import tests
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_partition.py \
       tests/ti-dsp-runtime/tidl-tests/test_tidl_codegen.py \
       tests/ti-dsp-runtime/tidl-tests/test_tidl_relax_import.py -v

# Run hardware e2e test (needs AM67A with c7x_compute firmware)
pytest tests/ti-dsp-runtime/tidl-tests/test_tidl_import_e2e.py -v -s
```
