# AGENTS.md

Guidance for AI coding agents working in this repository. Read this before
making changes.

## What this repo is

This is a **fork of Apache TVM 0.23.0** that adds a compiler backend and
runtime for Texas Instruments' **C7™ NPU** (a floating-point vector DSP +
deep-learning accelerator) on the **AM67A / J722S** SoC, with Arm cores. It
also retains the older **C66x** (AWRL6844) DSP path.

- The C7x-specific work is documented in **`docs-c7x/`** (MkDocs, published at
  <https://TexasInstruments-Sandbox.github.io/tvm/>). Treat `docs-c7x/` as the
  source of truth for anything TI/C7x/DSP specific. Upstream TVM docs live in
  `docs/`.
- **Status:** not production-ready — an active work-in-progress fork with
  incomplete operator coverage.
- **License:** Apache-2.0 (`LICENSE`). Export/third-party manifest:
  `TI_TVM_for_C7x_MMA_0.23.0_manifest.html`.

Start with these two docs before touching C7x code:

- `docs-c7x/contributor-guide/architecture-overview.md`
- `docs-c7x/user-guide/getting-started.md`

## Repository map (C7x-specific)

Compiler (TVM Python + C++):

- `src/target/c_static_lib/` — the `c_static_lib` backend: Relax VM → C/C++
  codegen. `codegen_c_static_lib_dsp.{h,cc}` holds TI-specific pragmas
  (`MUST_ITERATE`, `UNROLL`), per-layer profiling, and C7x vector-type emission.
- `python/tvm/relax/transform/` — Relax/TIR passes:
  - `schedule_c7x_dma.py` — H-tiling DMA scheduler (`cache_read` →
    `global.l2sram`, software-pipeline annotations, async prefetch).
  - `ti_mmalib_*.py` — MMALIB QDQ pattern fusion (conv2d/depthwise/FC/
    residual-add, int8+int16) and L2 DMA injection. **All pass instantiation is
    centralized in `get_mmalib_qdq_passes()` in `ti_mmalib_passes.py`.**
- `python/tvm/relax/backend/cpu_generic/pipeline.py` — wires the target-attr-
  dependent pass sequence. `get_default_pipeline(target)` is the entry point.

Runtime / firmware:

- `src/runtime/ti_dsp/` — lightweight C++14 DSP runtime: `model.h` API, static
  memory pools, zero-copy NDArrays. Build scripts: `build_runtime.sh`,
  `build_all.sh`, `validate_all.sh`.
- `src/runtime/ti_dsp/firmware/c7x/` — FreeRTOS compute service (RPMessage IPC,
  DLOAD dynamic module loader, UDMA/DRU DMA, shared-memory printf).
- `src/runtime/ti_dsp/firmware/c7x/arm/` — Arm-side `libc7x_arm_runtime.so`
  backing the `C7xVirtualMachine` (Python) / `c7x::Module` (C++) inference API.
- `src/runtime/ti_dsp/mmalib/` — C wrappers for 8 MMALIB kernels.
- `src/runtime/ti_dsp/dynmod/` — CMake build for relocatable C7x ELF modules.

Tests:

- `tests/ti-dsp-runtime/` — all DSP/MMALIB/quantization suites. Subdirs:
  `dsp-tests/` (model tests), `mmalib-tests/`, `pt2e-tests/`, `quantized/`,
  `unit-tests/`, `dynamic-tests/`, `wheel-tests/`, `SmolLM/`, `audio/`,
  `tidl-tests/`, `examples/`, and shared `dsp-cpp/` build infrastructure
  (`dsp_utils.py` is the canonical compile/build/run helper).

## Build

Two toolchains are involved: the C7x **host emulation** path (fast, no board)
and the **cross-compile** path (produces `lib0.out` for the board). The TI C7000
compiler is required for both `c7x_host` and `c7x` (host emu library ships with
it).

Required env vars (all paths absolute):

```bash
export TVM_HOME=$(pwd)                      # repo root
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS   # required for c7x
export MCU_PLUS_SDK_PATH=...                # required for c7x cross-compile
```

### Canonical full build (Docker)

The self-contained path uses `docker/Dockerfile.ci_c7x` (BeagleY-AI only) plus
`docker/bash.sh`, which bind-mounts this repo and runs as the host user so all
output lands in this checkout:

```bash
docker build -t tvm.ci_c7x:latest --build-arg BASE_IMAGE=ubuntu:24.04 \
  -f docker/Dockerfile.ci_c7x docker/

# Build TVM core + DSP runtime + firmware + ARM client + wheels
docker/bash.sh tvm.ci_c7x -- \
  bash src/runtime/ti_dsp/build_all.sh --board beagley-ai --wheels
```

`build_all.sh` runs: `patches/apply.sh` → TVM core (into `build-ci-c7x/`, not
`build/`) → `build_runtime.sh c7x_host` → `build_runtime.sh c7x` → firmware +
ARM client → optional wheels.

### Native build

```bash
# TVM core (uses build/ by default)
mkdir -p build && cp cmake/config.cmake build/
cmake -G Ninja -S . -B build && ninja -C build

# DSP runtime variants
cd src/runtime/ti_dsp
bash build_runtime.sh c7x_host   # host emulation (needs TI_CGT_C7000_PATH)
bash build_runtime.sh c7x        # C7x cross-compile (needs TI_CGT_C7000_PATH + MCU_PLUS_SDK_PATH)
bash build_runtime.sh c66x_host  # C66x host emulation (no TI compiler)
bash build_runtime.sh c66x       # C66x cross-compile (TI_CGT_C6000_PATH + MMWAVE_SDK_PATH)
bash build_runtime.sh all        # c66x + c7x + c7x_host
```

### Board / TIDL-MMALIB convention

`--board` is **required** (no default) and must be one of `j722s-evm` or
`beagley-ai`. Board-specific linkage:

- `beagley-ai` → `--tidl OFF --mmalib ON` (no TIDL subgraph offload on this
  board).
- `j722s-evm` → firmware default (`--tidl ON`, which forces `--mmalib ON`).

`build_all.sh` and the wheel build apply this convention automatically.

## Target strings

The backend target is `c_static_lib`. Key target attributes (full table in
`docs-c7x/user-guide/compilation.md`):

```python
target = tvm.target.Target("c_static_lib -mcpu=c7x -mmalib=1")
```

- `mcpu` — `c66x` or `c7x` (also accepts `arm`-prefixed / `generic`).
- `mmalib` — route eligible conv2d/matmul ops to MMALIB (requires `mcpu=c7x`).
- `use-cpp-api` (default `true`) — direct C++ VM calls. **Required, not just
  faster, for `c7x_dload`.**
- `profile-layers` — per-layer DSP cycle profiling via the shared-memory trace
  buffer.
- `tidl-kernels` (default `true`) — **must be `-tidl-kernels=0`** on BeagleY-AI,
  whose firmware links `--tidl OFF`; otherwise codegen emits calls to symbols
  the firmware doesn't export (fails only at DLOAD load time, not compile time).
- `tidl-runtime`, `skip-runtime-checks` (default `true`), `debug-alloc`.

## Testing

All DSP tests live under `tests/ti-dsp-runtime/` and **require `--dsp-mode`**
(no default):

| Mode | Meaning |
|------|---------|
| `c66x_host` / `c7x_host` | host emulation, no board |
| `c66x` | C66x hardware via JTAG/CCS |
| `c7x_dload` | AM67A/BeagleY-AI hardware via `c7x_compute` |

Tier markers on `dsp-tests/` (subdirs like `mmalib-tests/` use the same markers):

- `quick` — PR gate (~20 s host).
- `core` — post-merge (superset of `quick`, except `test_mmalib_oc_tile_consistency.py` which is `quick`-only).
- no marker — nightly/full.

Common commands (run from repo root):

```bash
cd $TVM_HOME

# PR gate, C7x host emulation
pytest tests/ti-dsp-runtime/dsp-tests/ -m quick --dsp-mode=c7x_host -v

# Post-merge, C7x host emulation
pytest tests/ti-dsp-runtime/dsp-tests/ -m core --dsp-mode=c7x_host -v

# MMALIB kernel unit tests, host emulation
pytest --rootdir=tests/ti-dsp-runtime tests/ti-dsp-runtime/mmalib-tests/ -m quick --dsp-mode=c7x_host -v

# Quantized ResNet-18 with MMALIB on real hardware
pytest --rootdir=tests/ti-dsp-runtime tests/ti-dsp-runtime/dsp-tests/test_quantized_resnet_dsp.py \
    -v --dsp-mode=c7x_dload --use-cpp-api --mmalib --profile
```

Also:

- `pytest --rootdir=tests/ti-dsp-runtime ...` is the convention for the
  ti-dsp-runtime suites (their `conftest.py` supplies DSP fixtures).
- `DSP_KEEP_TEMP=1` preserves the generated C/build artifacts under
  `/tmp/dsp_test_<name>_<timestamp>/` — useful when debugging codegen.
- The root `conftest.py` handles sharding for the *upstream* `tests/` suite
  only; ti-dsp-runtime tests use their own `conftest.py`.

### Critical: never parallelize `c7x_dload`

**`c7x_dload` tests must never run in parallel or in the background.** The
AM67A has a single DSP core; concurrent sessions cause DMA-BUF exhaustion and
firmware hangs. The same rule applies to running the native and Docker Jenkins
pipelines concurrently against the same physical board.

## Architecture

Pipeline (float32 or PT2E-quantized PyTorch model):

```
torch.export → from_exported_program → Relax IRModule
  → relax.build(..., target="c_static_lib -mcpu=c7x [-mmalib=1]")
  → export_library → lib0.c / devc.c / weights.bin
  → native build:
      c7x_dload: cl7x + lnk7x --dynamic=lib → lib0.out (relocatable C7x ELF)
      c7x_host: g++ + TI Host Emu lib → cg_dsp (x86 executable)
```

- With `-mmalib=1`, the pipeline skips `ConvertLayoutNHWC` (MMALIB kernels want
  NCHW), fuses eligible QDQ ops into `call_extern("mmalib_*")`, and folds quant
  scale/shift/bias at compile time.
- Reuse the harness helpers in `tests/ti-dsp-runtime/dsp-cpp/dsp_utils.py`
  (`get_target_string`, `compile_for_dsp`, `build_dsp_dynmod`,
  `build_dsp_c7x_host`, `run_dsp_host`) instead of re-implementing the
  compile→build sequence — they already encode BeagleY-AI's `-tidl-kernels=0`,
  the two-stage DLOAD link, and weight embedding.

Runtime deployment flow: compile on the dev host → `scp lib0.out` to the board
→ load via DLOAD → infer via `C7xVirtualMachine` (Python) / `c7x::Module`
(C++), which talk to the `c7x_compute` firmware service over board-local rpmsg
IPC. **Those APIs run on the board, not the dev host.**

## Conventions and gotchas

- **Read `docs-c7x/` first.** Every subsystem has a dedicated doc under
  `docs-c7x/contributor-guide/{backend,dsp-runtime,firmware,testing}/` and
  `docs-c7x/user-guide/`. Update the relevant doc when you change behavior.
- **MMALIB pass ordering matters.** MMALIB QDQ fusion passes must run *before*
  `FuseQDQToInt8Conv2D` and `EliminateQDQRoundTrip` (those destroy the QDQ
  pattern MMALIB needs). Add passes via `get_mmalib_qdq_passes()` in
  `ti_mmalib_passes.py`; never hand-edit `pipeline.py`.
- **DSP runtime constraints.** The runtime is C++14, no exceptions/RTTI, no
  `malloc()` in the hot path — error handling is a `ModelError` enum, memory is
  pre-allocated static pools (bump-pointer + free-list).
- **Three distinct memory maps** exist (standard runtime, deployed firmware,
  standalone JTAG harness). Don't cross-check pool sizes across them — see
  `docs-c7x/contributor-guide/dsp-runtime/memory-map.md`.
- **Lifetime rules** for the board inference APIs (zero-copy output/pre-staged
  inputs are views into shared DDR) are a common bug source — see
  `docs-c7x/user-guide/python-api.md`.
- **Board hostname convention:** `beagley-ai` maps to SSH host `beagley-ai`;
  anything else maps to `am67a`. Add an SSH-config alias if your board differs.
- Keep C++/Python formatting consistent with the existing tree (`.clang-format`,
  `ruff`/lint via the upstream tooling). Don't reformat unrelated files.
