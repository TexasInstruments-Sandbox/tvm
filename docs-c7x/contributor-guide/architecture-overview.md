# Architecture Overview

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
                +---------------------------------------+
```

**Note:** the `c_static` backend generates self-contained C/C++ code
that compiles with any toolchain. The "static" means no shared library
dependencies at runtime -- it does NOT mean static shapes.

**Quantization is optional:** an unquantized float32 model goes through
`torch.export`/`from_exported_program` unchanged (see
[Compile and Run a Model](../user-guide/getting-started.md#compile-and-run-a-model)
in Getting Started); PT2E quantization via `C7xMMAQuantizer` is what
produces the QDQ-annotated graph that feeds the MMALIB QDQ Fusion
branch. See [MMALIB Integration](backend/mmalib-integration.md) for
the full quantization and fusion pipeline.

## Deployment Flow

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
[Python / C++ API Reference](../user-guide/python-api.md) for the full API
(zero-copy inputs/outputs, cycle counts, lifetime rules), and
[Examples: YOLO26 & ResNet-18](../user-guide/examples.md) for full
compile -> deploy -> run walkthroughs of both.

## Components

### Compiler (TVM Python + C++)

| Component | Location | Description |
|-----------|----------|-------------|
| C Static Backend | `src/target/c_static/` | C/C++ code generator for Relax VM; emits wrapper functions, weight serialisation, register file management. [Docs](backend/c-static.md) |
| DSP Code Extensions | `src/target/c_static/codegen_c_static_dsp.{h,cc}` | TI-specific: compiler pragmas (`MUST_ITERATE`, `UNROLL`), per-layer cycle profiling, C7x vector type emission |
| C7x DMA Scheduler | `python/tvm/relax/transform/schedule_c7x_dma.py` | TIR pass: H-tiling with `cache_read` into `global.l2sram`, software pipeline annotations, async DMA prefetch |
| MMALIB Passes | `python/tvm/relax/transform/ti_mmalib_*.py` | QDQ pattern fusion (conv2d/depthwise/FC/residual-add, int8+int16) and L2 DMA injection for direct MMA coprocessor offload via `-mmalib=1`. See [MMALIB Integration](backend/mmalib-integration.md). |

### Runtime

| Component | Location | Description |
|-----------|----------|-------------|
| DSP Runtime Library | `src/runtime/ti_dsp/` | C++14 Model API with static memory pools, zero-copy NDArrays, cross-platform (host/C7x). [Docs](dsp-runtime/internals.md) |
| C7x Firmware | `src/runtime/ti_dsp/firmware/c7x/` | FreeRTOS compute service: RPMessage (TI's Arm↔DSP IPC framework), DLOAD dynamic module loader, UDMA (TI's Navigator Subsystem DMA engine, via the C7x's local DRU) transfers, shared-memory printf. [Docs](firmware/architecture.md) · [Design](firmware/design-deep-dive.md) |
| C7x Arm Runtime | `src/runtime/ti_dsp/firmware/c7x/arm/` | Arm-side shared library (`libc7x_arm_runtime.so`) backing the C++ `c7x::Module` / Python `C7xVirtualMachine` inference API. [API reference](../user-guide/python-api.md) · [build/deploy](../user-guide/deploying-firmware.md) |
| MMALIB Wrappers | `src/runtime/ti_dsp/mmalib/` | C wrappers for 8 MMALIB kernels (conv2d/depthwise-conv2d/matmul/matmul_bias x int8/int16), linked into the C7x firmware and exported via DLOAD. [Docs](backend/mmalib-integration.md) |
| DLOAD Infrastructure | `src/runtime/ti_dsp/dynmod/` | CMake build for C7x relocatable ELF modules (linker scripts, symbol exports) |
| Deployment Scripts | `src/runtime/ti_dsp/scripts/` | `run_on_c75x.sh` (J722S) DSP debug/load script via JTAG. [Docs](hardware-debug.md) |

### Tests

All DSP/MMALIB/quantized-model test suites live under
`tests/ti-dsp-runtime/`, covering the C7x target across host emulation
and AM67A hardware. See [Testing Overview](testing/overview.md) for the
full directory breakdown, test tiers (`quick`/`core`/nightly), and
Jenkins commands.
