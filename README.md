# TVM for TI C7™ NPU

This is a fork of [Apache TVM](https://github.com/apache/tvm) that adds
a compiler backend and runtime for Texas Instruments' C7™ NPU -- a
floating-point vector DSP core that combines traditional DSP
capability, vector processing, and a deep learning accelerator, paired
with Arm cores in TI's AM67A/J722S SoCs. Covers the full pipeline from
Relax graph-level IR through C code generation, a minimal embedded
runtime, remoteproc firmware for the AM67A, and comprehensive
pytest-based test infrastructure.

**New here? Jump to [Quick Start](#quick-start).**

## Contents

- [License](#license)
- [Supported Targets](#supported-targets)
- [Architecture](#architecture)
- [Components](#components)
- [Quick Start](#quick-start)
- [MMALIB Offloading](#mmalib-offloading)
- [Documentation](#documentation)
- [About Apache TVM (upstream)](#about-apache-tvm-upstream)

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
restart per model. On the board, application code drives inference
through the Python (`C7xVirtualMachine`) or C++ (`c7x::Module`) API --
both talk to the `c7x_compute` firmware service over the board's local
rpmsg IPC channel, so these calls must run on the board itself, not
the dev host:

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
 lib0.out (relocatable C7x ELF)  --scp-->   Python: C7xVirtualMachine("lib0.out")
                                            C++:    c7x::Module::Load("lib0.out")
                                                  |
                                            DLOAD: parse ELF, resolve
                                            61 symbols, relocate into DDR
                                                  v
                                            Python: vm["main"](inp)
                                                    (or vm.run_nocopy(inp))
                                            C++:    vm.Run(&input)
                                            (cg_main_dsp() on device)
                                                  v
                                            Python: vm.close() (or `with` exit)
                                            C++:    vm.Close() (or destructor)
```

Both APIs are a thin, IPC-backed wrapper around the same load/infer/
unload operations the `c7x_compute` CLI binary exposes -- that CLI is
meant for manual testing and board health checks (`c7x_compute ping`/
`status`), not application use. See the
[C7x Inference API reference](python/tvm/contrib/c7x/README.md) for
the full API (zero-copy inputs/outputs, cycle counts, lifetime rules),
and
[`tests/ti-dsp-runtime/examples/`](tests/ti-dsp-runtime/examples/README.md)
for full compile -> deploy -> run walkthroughs of both.

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
| C7x Arm Runtime | `src/runtime/ti_dsp/firmware/c7x/arm/` | Arm-side shared library (`libc7x_arm_runtime.so`) backing the C++ `c7x::Module` / Python `C7xVirtualMachine` inference API. [API reference](python/tvm/contrib/c7x/README.md) · [build/deploy](src/runtime/ti_dsp/firmware/c7x/arm/README.md) |
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

### Docker (recommended -- self-contained build environment)

`docker/Dockerfile.ci_c7x` bakes in everything needed to build for
BeagleY-AI -- the TI CGT C7000 compiler, TI SysConfig, PSDK RTOS, LLVM,
the aarch64 cross-compiler, and `uv` -- so none of that needs installing
on the host directly. This is the fastest path from a clean checkout to
a validated board; three commands:

```bash
# 1. Build the image (behind a corporate proxy: pass it through as
#    shown; otherwise drop the --build-arg lines)
docker build -t tvm.ci_c7x:latest \
  --build-arg http_proxy=$http_proxy \
  --build-arg https_proxy=$https_proxy \
  -f docker/Dockerfile.ci_c7x docker/

# 2. Build TVM + DSP runtime + firmware + ARM client + packaging
#    wheels for BeagleY-AI
docker/bash.sh tvm.ci_c7x -- \
    bash src/runtime/ti_dsp/build_all.sh --board beagley-ai --wheels

# 3. Deploy firmware and hardware-validate on a real BeagleY-AI board.
#    Installs and tests against the wheel built in step 2 (not a
#    PYTHONPATH into this checkout) inside a .venv-ci-c7x venv at the
#    repo root. Needs SSH access to the board, cached torchvision
#    weights, and (behind a proxy) --env http_proxy/https_proxy again --
#    this script's pip installs happen at container *runtime*, unlike
#    step 1's --build-arg.
docker/bash.sh --net=host \
    -v ~/.ssh:$(pwd)/.ssh:ro \
    -v ~/.cache/torch:$(pwd)/.cache/torch:ro \
    --env http_proxy=$http_proxy --env https_proxy=$https_proxy \
    tvm.ci_c7x -- \
    bash src/runtime/ti_dsp/validate_all.sh --board beagley-ai
```

`docker/bash.sh` bind-mounts this repo into the container and runs as
your host user, so all output -- builds, wheels, the `.venv-ci-c7x` test
venv -- lands in this same working tree, owned by you; nothing is
copied into or built inside the image itself. Scoped to BeagleY-AI only
today.

See [`docker/README_c7x.md`](docker/README_c7x.md) for the SSH/torch
cache mount details, why `PYTHONPATH` gets explicitly unset before
testing, and how this wires into Jenkins.

### Compile and Run a Model

For full runnable examples of both offload APIs -- YOLO26 object
detection via the Python API (with optional MMALIB offload visualization
and per-layer cycle profiling), and ResNet-18 classification via the C++
API -- quantizing, compiling, deploying, and running end-to-end on a real
BeagleY-AI/AM67A board, see
[`tests/ti-dsp-runtime/examples/README.md`](tests/ti-dsp-runtime/examples/README.md).
See `tests/ti-dsp-runtime/dsp-cpp/dsp_utils.py` for the lower-level build
and deploy pipeline used by all pytest tests.

See the [Arm Runtime README](src/runtime/ti_dsp/firmware/c7x/arm/README.md)
for building/deploying `libc7x_arm_runtime.so` and its internal design.

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

## Documentation

Per-component READMEs, referenced throughout this document:

- [C Static Backend](src/target/c_static/README.md)
- [DSP Runtime Library](src/runtime/ti_dsp/README.md)
- [C7x Firmware](src/runtime/ti_dsp/firmware/c7x/README.md) ([Design](src/runtime/ti_dsp/firmware/c7x/design_doc.md))
- [C7x Inference API](python/tvm/contrib/c7x/README.md)
- [C7x Arm Runtime](src/runtime/ti_dsp/firmware/c7x/arm/README.md)
- [Examples](tests/ti-dsp-runtime/examples/README.md)
- [MMALIB Offloading](src/runtime/ti_dsp/mmalib/README.md)
- [Deployment Scripts](src/runtime/ti_dsp/scripts/README.md)
- [Test Suite](tests/ti-dsp-runtime/README.md)

---

## About Apache TVM (upstream)

This repository is forked from [Apache TVM](https://github.com/apache/tvm).
The sections below describe upstream Apache TVM in general -- they are
not specific to this fork's C7™ NPU backend. In particular, the
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
