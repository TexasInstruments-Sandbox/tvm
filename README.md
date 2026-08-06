# TVM for TI C7x DSP

This is a fork of [Apache TVM](https://github.com/apache/tvm) that adds
a compiler backend and runtime for Texas Instruments' C7x DSP -- a
floating-point vector DSP core that combines traditional DSP
capability, vector processing, and a deep learning accelerator, paired
with Arm cores in TI's AM67A/J722S SoCs. Covers the full pipeline from
Relax graph-level IR through C code generation, a minimal embedded
runtime, remoteproc firmware for the AM67A, and comprehensive
pytest-based test infrastructure.

## License

TVM is licensed under the [Apache-2.0](LICENSE) license. For the full
open-source license manifest and export classification for this
release, see
[TI_TVM_for_C7x_MMA_0.23.0_manifest.html](TI_TVM_for_C7x_MMA_0.23.0_manifest.html).

## Supported Targets

| Target | Device | DSP |
|--------|--------|-----|
| `c_static -mcpu=c7x` | J722S / AM67A | C7x |

Two boards are supported via `--board`/`--ddr` on the runtime and
firmware build scripts (see the usage header in
[`build_runtime.sh`](src/runtime/ti_dsp/build_runtime.sh)); `--board` is
required for any hardware build -- there is no default:

- **[AM67A EVM](https://www.ti.com/tool/J722SXH01EVM)** (`j722s-evm`)
  -- TI's evaluation module, orderable directly from TI.
- **[BeagleY-AI](https://www.beagleboard.org/boards/beagley-ai)**
  (`beagley-ai`) -- an open-hardware single-board computer built around
  the same J722S SoC.

Board choice has no effect on model compilation -- `relax.build(mod,
target="c_static -mcpu=c7x")` produces the same `lib0.c`/`weights.bin`
either way. `--board`/`--ddr` only matter when building the runtime
library and firmware (they select the shared-DMA carveout physical
address and SDK version); the runtime and firmware builds must use the
same `--board`/`--ddr`, or DMA addressing silently corrupts at runtime.

Also supports host emulation (x86 GCC) for development and CI without
hardware.

## Architecture

```
                +---------------------------------------+
                |         float32 PyTorch Model         |
                |  (torchvision, Hugging Face, custom)  |
                +---------------------------------------+
                                    |
                +---------------------------------------+
                |     PT2E Quantization  [optional]     |
                |   (C7xMMAQuantizer: prepare_pt2e ->   |
                |      calibrate -> convert_pt2e)       |
                +---------------------------------------+
                                    |
                +---------------------------------------+
                | torch.export + from_exported_program  |
                +---------------------------------------+
                                    |
                +---------------------------------------+
                |      Relax Model (PyTorch, ONNX)      |
                +---------------------------------------+
                                    |
                 +------------------+------------------+
                 |                                     |
+---------------------------------+   +---------------------------------+
|          ConvertLayout          |   |        MMALIB QDQ Fusion        |
| (NHWC; skipped when -mmalib=1)  |   |      (conv2d/depthwise/FC/      |
|                                 |   |     residual-add -> MMALIB      |
|         ScheduleC7xDMA          |   |     call_extern, -mmalib=1)     |
|    (DMA tiling, L2 prefetch)    |   |                                 |
+---------------------------------+   +---------------------------------+
                 |                                     |
                 +------------------+------------------+
                                    |
                +---------------------------------------+
                |            CodeGenCStatic             |
                |           (C/C++ emission)            |
                +---------------------------------------+
                                    |
                +---------------------------------------+
                |            TI DSP Runtime             |
                |   (model.h API, static pools; calls   |
                | into MMALIB wrappers when -mmalib=1)  |
                +---------------------------------------+
                                    |
                +---------------------------------------+
                |             C7x Hardware              |
                |      (Dynamic load, MMA               |
                |        coprocessor via MMALIB)        |
                +---------------------------------------+
```

**Note:** the `c_static` backend generates self-contained C/C++ code
that compiles with any toolchain.  The "static" means no shared library
dependencies at runtime -- it does NOT mean static shapes.

**Quantization is optional:** an unquantized float32 model goes through
`torch.export`/`from_exported_program` unchanged (see
[Compile and Run a Model](#compile-and-run-a-model) below); PT2E
quantization via `C7xMMAQuantizer` is what produces the QDQ-annotated
graph that feeds the MMALIB QDQ Fusion branch. See
[MMALIB Offloading](#mmalib-offloading) below for the full quantization
and fusion pipeline.

### Deployment Flow

Compilation happens on an x86 dev host; the resulting module is copied
to the AM67A board and loaded into a long-running firmware process via
DLOAD (TI's dynamic loader), avoiding a firmware rebuild or DSP
restart per model:

```
 Dev Host (x86 Linux)                    AM67A / J722S Board
 ─────────────────────                   ─────────────────────
 relax.build(mod, target="c_static
             -mcpu=c7x")
        |
        v
 lib0.c / lib1.c / weights.bin
        |
 cl7x + lnk7x --dynamic=lib
        v
 lib0.out (relocatable C7x ELF)  --scp-->  c7x_compute load lib0.out
                                                  |
                                            DLOAD: parse ELF, resolve
                                            61 symbols, relocate into DDR
                                                  v
                                            c7x_compute infer <handle>
                                            (cg_main_dsp() on device)
                                                  v
                                            c7x_compute unload <handle>
```

See [C7x Hardware Deployment](#c7x-hardware-deployment) below for the
per-step commands.

## Components

### Compiler (TVM Python + C++)

| Component | Location | Description |
|-----------|----------|-------------|
| C Static Backend | `src/target/c_static/` | C/C++ code generator for Relax VM; emits wrapper functions, weight serialisation, register file management. [README](src/target/c_static/README.md) |
| DSP Code Extensions | `src/target/c_static/codegen_c_static_dsp.{h,cc}` | TI-specific: compiler pragmas (`MUST_ITERATE`, `UNROLL`), per-layer cycle profiling, C7x vector type emission |
| C7x DMA Scheduler | `python/tvm/relax/transform/schedule_c7x_dma.py` | TIR pass: H-tiling with `cache_read` into `global.l2sram`, software pipeline annotations, async DMA prefetch |
| MMALIB Passes | `python/tvm/relax/transform/ti_mmalib_*.py` | QDQ pattern fusion (conv2d/depthwise/FC/residual-add, int8+int16) and L2 DMA injection for direct MMA coprocessor offload via `-mmalib=1`. See [MMALIB Offloading](#mmalib-offloading) below. |

### Runtime

| Component | Location | Description |
|-----------|----------|-------------|
| DSP Runtime Library | `src/runtime/ti_dsp/` | C++14 Model API with static memory pools, zero-copy NDArrays, cross-platform (host/C7x). [README](src/runtime/ti_dsp/README.md) |
| C7x Firmware | `src/runtime/ti_dsp/firmware/c7x/` | FreeRTOS compute service: RPMessage (TI's Arm↔DSP IPC framework), DLOAD dynamic module loader, UDMA (TI's Navigator Subsystem DMA engine, via the C7x's local DRU) transfers, shared-memory printf. [README](src/runtime/ti_dsp/firmware/c7x/README.md) · [Design](src/runtime/ti_dsp/firmware/c7x/design_doc.md) |
| C7x Arm Runtime | `src/runtime/ti_dsp/firmware/c7x/arm/` | Arm-side shared library (`libc7x_arm_runtime.so`) and C++ `c7x::Module` API; Python `C7xVirtualMachine` wrapper. [README](src/runtime/ti_dsp/firmware/c7x/arm/README.md) |
| MMALIB Wrappers | `src/runtime/ti_dsp/mmalib/` | C wrappers for 8 MMALIB kernels (conv2d/depthwise-conv2d/matmul/matmul_bias x int8/int16), linked into the C7x firmware and exported via DLOAD. [README](src/runtime/ti_dsp/mmalib/README.md) |
| DLOAD Infrastructure | `src/runtime/ti_dsp/dynmod/` | CMake build for C7x relocatable ELF modules (linker scripts, symbol exports) |
| Deployment Scripts | `src/runtime/ti_dsp/scripts/` | `run_on_c75x.sh` (J722S) DSP debug/load script via JTAG. [README](src/runtime/ti_dsp/scripts/README.md) |

### Tests

All DSP/MMALIB/quantized-model test suites live under
`tests/ti-dsp-runtime/`, covering the C7x target across host emulation
and AM67A hardware. See the
[README](tests/ti-dsp-runtime/README.md) for the full directory
breakdown, test tiers (`quick`/`core`/nightly), and Jenkins commands.

## Quick Start

### Docker (self-contained build environment)

`docker/Dockerfile.ci_c7x` bakes in everything needed to build for
BeagleY-AI -- the TI CGT C7000 compiler, TI SysConfig, PSDK RTOS, LLVM,
the aarch64 cross-compiler, and `uv` -- so you can skip installing any
of the Prerequisites below on the host:

```bash
# Build the image (behind a corporate proxy: pass it through as shown;
# otherwise drop the --build-arg lines)
docker build -t tvm.ci_c7x:latest \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  -f docker/Dockerfile.ci_c7x docker/

# Build TVM + DSP runtime + firmware + ARM client for BeagleY-AI
docker/bash.sh tvm.ci_c7x -- \
    bash src/runtime/ti_dsp/build_all.sh --board beagley-ai

# ...and the x86/arm64 packaging wheels too
docker/bash.sh tvm.ci_c7x -- \
    bash src/runtime/ti_dsp/build_all.sh --board beagley-ai --wheels
```

`docker/bash.sh` bind-mounts this repo into the container and runs as
your host user, so build output lands in the same working tree you
already have checked out, owned by you -- nothing is copied into or
built inside the image itself. Scoped to BeagleY-AI only today.

Hardware validation -- deploy firmware and run the quantized-model test
suite on a real BeagleY-AI board -- works the same way, via
`src/runtime/ti_dsp/validate_all.sh`. It needs two things beyond a
plain build: SSH access to the board, and the proxy passed again as
`--env` (the `--build-arg`s above only reach `docker build`; this
script's `uv pip install` runs at container *runtime*, fetching from
PyPI/download.pytorch.org):

```bash
docker/bash.sh --net=host \
    -v ~/.ssh:$(pwd)/.ssh:ro \
    -v ~/.cache/torch:$(pwd)/.cache/torch:ro \
    --env http_proxy=$http_proxy --env https_proxy=$https_proxy \
    tvm.ci_c7x -- \
    bash src/runtime/ti_dsp/validate_all.sh --board beagley-ai
```

`docker/bash.sh` sets the container's `$HOME` to wherever the repo gets
mounted, so the `-v ~/.ssh:...` mount above has to land at that same
path -- `$(pwd)/.ssh` when running this by hand, or `/workspace/.ssh`
under Jenkins (see `tests/ti-dsp-runtime/Jenkinsfile.docker`). It's
just your existing SSH config/key already trusted by the board, not a
new credential -- but `--board beagley-ai` always connects to the
literal hostname `beagley-ai` (never an IP), so your `~/.ssh/config`
needs a `Host beagley-ai` entry (or equivalent DNS/hosts-file entry)
pointing at wherever the board actually is; this is the same
board-name-to-host convention `deploy-c7x.sh` and the pytest suite
already use, not something specific to Docker. `~/.cache/torch` is the
pre-cached torchvision weights the test suite needs -- mount it in if
your build environment can't reach download.pytorch.org directly.

`tests/ti-dsp-runtime/Jenkinsfile.docker` wires this whole flow --
image build, `build_all.sh`, `validate_all.sh` -- into a Jenkins
pipeline, so the node itself only needs Docker, none of the
Prerequisites below. It's a separate, manual-trigger-only job from
`tests/ti-dsp-runtime/Jenkinsfile`'s native (non-Docker) nightly
pipeline -- the two must never run concurrently against the same
physical board.

### Prerequisites

```bash
# TVM build
mkdir -p build && cp cmake/config.cmake build/
# Enable in build/config.cmake: BUILD_STATIC_RUNTIME=ON
cd build && cmake -G Ninja .. && ninja && cd ..

# Python
export TVM_HOME=$(pwd)
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
uv pip install -e python/

# TI C7x compiler (required for all DSP tests)
# Download from https://www.ti.com/tool/C7000-CGT
export TI_CGT_C7000_PATH=<path to>/ti-cgt-c7000_5.0.1.LTS
```

### Build DSP Runtime

```bash
cd src/runtime/ti_dsp
bash build_runtime.sh c7x_host   # Host emulation (no hardware needed)
bash build_runtime.sh c7x        # C7x cross-compilation (J722S/AM67A)
```

### Run Tests

```bash
cd tests/ti-dsp-runtime

# Quick smoke test — host emulation, no hardware (~20s)
pytest --rootdir=. dsp-tests/ -m quick --dsp-mode=c7x_host -v

# Quick smoke test — AM67A hardware (~5min)
pytest --rootdir=. dsp-tests/ -m quick --dsp-mode=c7x_dload -v

# Full regression — host emulation (73 tests, ~5min)
pytest --rootdir=. dsp-tests/ --dsp-mode=c7x_host -v

# Full regression — AM67A hardware (73 tests, ~2-3h)
# NEVER run in background or concurrent sessions (single DSP core)
pytest --rootdir=. dsp-tests/ --dsp-mode=c7x_dload -v
```

### Compile and Run a Model

```python
import torch
import tvm
from tvm import relax
from tvm.relax.frontend.torch import from_exported_program

# 1. Export PyTorch model
model = torch.hub.load("pytorch/vision", "resnet18", weights="DEFAULT")
ep = torch.export.export(model.eval(), (torch.randn(1, 3, 224, 224),))

# 2. Import to Relax
mod = from_exported_program(ep, keep_params_as_input=True)

# 3. Compile for C7x DSP
target = "c_static -mcpu=c7x"
with tvm.transform.PassContext(opt_level=3):
    lib = relax.build(mod, target=target)

# 4. Export C code + weights
lib.export_library("/tmp/model/lib0.c")
# Compile with TI cl7x, link with DSP runtime, deploy via DLOAD
```

See `tests/ti-dsp-runtime/dsp-cpp/dsp_utils.py` for the full build and
deploy pipeline used by all pytest tests.

## C7x Hardware Deployment

The J722S/AM67A deployment uses [remoteproc](https://docs.kernel.org/staging/remoteproc.html)
(the Linux kernel framework for booting and controlling co-processor
firmware) together with DLOAD, a custom runtime ELF loader that lets
the firmware load and run new compiled models without a firmware
rebuild. The workflow is:

1. **Build firmware** (once): `src/runtime/ti_dsp/firmware/c7x/dsp/build.sh`
   -- requires TI's MCU+ SDK (part of the Processor SDK RTOS for J722S)
   and MMALIB, installed separately from TI and pointed to via
   `PSDK_INSTALL_PATH`/`MCU_PLUS_SDK_PATH` and `MMALIB_PATH`. Neither SDK
   is bundled with this repo.
2. **Deploy firmware**: `./deploy-c7x.sh dsp/build/c7x_compute.out`
3. **Build host CLI**: `src/runtime/ti_dsp/firmware/c7x/arm/build.sh deploy`
4. **Compile model**: TVM generates `lib0.c` + `weights.bin`
5. **Build DLOAD module**: CMake with dynmod linker scripts produces `lib0.out`
6. **Run on DSP** — CLI or Python/C++ API:

```bash
# CLI
c7x_compute run lib0.out --input in.bin --output out.bin
```

```python
# Python (VirtualMachine-compatible)
from tvm.contrib.c7x import C7xVirtualMachine
vm = C7xVirtualMachine("lib0.out")
out = vm["main"](tvm.nd.array(data))
```

```cpp
// C++
auto vm = c7x::Module::Load("lib0.out");
auto out = vm.Run(&input_dl_tensor);
```

See the [Arm Runtime README](src/runtime/ti_dsp/firmware/c7x/arm/README.md)
for the full Python/C++ API reference, zero-copy input/output modes, and
design rationale.

See [firmware README](src/runtime/ti_dsp/firmware/c7x/README.md) for
deployment, troubleshooting, and recovery procedures.

## MMALIB Offloading

For ops in MMALIB's supported set, `-mmalib=1` routes them directly to
TI's MMALIB library, which programs the C7x MMA coprocessor via a
`call_extern` from the single generated `lib0.c` to an MMALIB wrapper,
with quantization scale/shift/bias folded in at compile time:

```
target = "c_static -mcpu=c7x -mmalib=1"
```

A single 64ch 56×56 int8 conv2d layer takes ~45M cycles as scalar C7x
code; the same layer takes ~1.67M cycles via the MMA coprocessor (27x),
dropping to ~477K cycles (96x) when input data is staged into L2 SRAM
via DMA before the MMA call.

Int8/int16 QDQ patterns are produced via PyTorch PT2E quantization with
`C7xMMAQuantizer`, then matched and fused by dedicated passes for
conv2d, depthwise conv2d, matmul/FC, and residual add. Non-eligible ops
fall through to scalar loop codegen:

```python
from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program

quantizer = C7xMMAQuantizer(dtype="int8")   # or "int16"
prepared = prepare_pt2e(model, quantizer)
# ... calibrate ...
quantized = convert_pt2e(prepared)
mod = from_exported_program(torch.export.export(quantized, example_inputs),
                             keep_params_as_input=True)
```

```bash
# Quick MMALIB kernel unit tests, host emulation
pytest --rootdir=tests/ti-dsp-runtime tests/ti-dsp-runtime/mmalib-tests/ -m quick --dsp-mode=c7x_host -v

# Full quantized model with MMALIB (AM67A hardware)
pytest --rootdir=tests/ti-dsp-runtime tests/ti-dsp-runtime/dsp-tests/test_quantized_resnet_dsp.py \
    -v --dsp-mode=c7x_dload --use-cpp-api --mmalib --profile
```

See the [MMALIB README](src/runtime/ti_dsp/mmalib/README.md) for the
full QDQ fusion pipeline, supported-op constraints, per-model
performance, and the firmware/codegen build flags that scope
BeagleY-AI to MMALIB alone.

## Environment Variables

| Variable | Required for | Default |
|----------|-------------|---------|
| `TI_CGT_C7000_PATH` | All C7x tests | none (tests fail without it) -- install [C7000-CGT](https://www.ti.com/tool/C7000-CGT) |
| `TVM_HOME` | Python imports | none |
| `PYTHONPATH` | TVM Python module | none |
| `DSP_KEEP_TEMP` | Debug: preserve build artifacts | unset (cleanup on) |

## Documentation

Per-component READMEs, referenced throughout this document:

- [C Static Backend](src/target/c_static/README.md)
- [DSP Runtime Library](src/runtime/ti_dsp/README.md)
- [C7x Firmware](src/runtime/ti_dsp/firmware/c7x/README.md) ([Design](src/runtime/ti_dsp/firmware/c7x/design_doc.md))
- [C7x Arm Runtime](src/runtime/ti_dsp/firmware/c7x/arm/README.md)
- [MMALIB Offloading](src/runtime/ti_dsp/mmalib/README.md)
- [Deployment Scripts](src/runtime/ti_dsp/scripts/README.md)
- [Test Suite](tests/ti-dsp-runtime/README.md)

---

## About Apache TVM (upstream)

This repository is forked from [Apache TVM](https://github.com/apache/tvm).
The sections below describe upstream Apache TVM in general -- they are
not specific to this fork's C7x DSP backend. In particular, the
"Getting Started" link below is upstream's generic tutorial, not this
fork's Quick Start (see above for that).

<img src=https://raw.githubusercontent.com/apache/tvm-site/main/images/logo/tvm-logo-small.png width=128/> Open Machine Learning Compiler Framework
==============================================
[Documentation](https://tvm.apache.org/docs) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

Apache TVM is an open machine learning compilation framework,
following the following principles:

- Python-first development that enables quick customization of machine learning compiler pipelines.
- Universal deployment to bring models into minimum deployable modules.

### Getting Started

Check out the [TVM Documentation](https://tvm.apache.org/docs/) site for installation instructions, tutorials, examples, and more.
The [Getting Started with TVM](https://tvm.apache.org/docs/get_started/overview.html) tutorial is a great
place to start.

### Contribute to TVM

TVM adopts the Apache committer model. We aim to create an open-source project maintained and owned by the community.
Check out the [Contributor Guide](https://tvm.apache.org/docs/contribute/).

### History and Acknowledgement

TVM started as a research project for deep learning compilation.
The first version of the project benefited a lot from the following projects:

- [Halide](https://github.com/halide/Halide): Part of TVM's TIR and arithmetic simplification module
 originates from Halide. We also learned and adapted some parts of the lowering pipeline from Halide.
- [Loopy](https://github.com/inducer/loopy): use of integer set analysis and its loop transformation primitives.
- [Theano](https://github.com/Theano/Theano): the design inspiration of symbolic scan operator for recurrence.

Since then, the project has gone through several rounds of redesigns.
The current design is also drastically different from the initial design, following the
development trend of the ML compiler community.

The most recent version focuses on a cross-level design with TensorIR as the tensor-level representation
and Relax as the graph-level representation and Python-first transformations.
The project's current design goal is to make the ML compiler accessible by enabling most
transformations to be customizable in Python and bringing a cross-level representation that can jointly
optimize computational graphs, tensor programs, and libraries. The project is also a foundation
infra for building Python-first vertical compilers for domains, such as LLMs.
