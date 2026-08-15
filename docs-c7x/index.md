# TVM for TI C7™ NPU

This is a fork of [Apache TVM](https://github.com/apache/tvm) 0.23.0
that adds a compiler backend and runtime for Texas Instruments' C7™ NPU
-- a floating-point vector DSP core that combines traditional DSP
capability, vector processing, and a deep learning accelerator, paired
with Arm cores in TI's AM67A/J722S SoCs. It covers the full pipeline
from Relax graph-level IR through C code generation, a minimal
embedded runtime, remoteproc firmware for the AM67A, and comprehensive
pytest-based test infrastructure.

**New here? Jump to [Getting Started](user-guide/getting-started.md).**

!!! warning "Project status"
    TVM/Relax for the C7™ NPU is **not production-ready** -- it is an
    active work-in-progress fork of Apache TVM. Expect incomplete
    operator coverage, and unpolished edges.

!!! note "Scope"
    - This project targets only **one of the two C7™ DSP cores**
      present on the AM67A SoC (TI's
      [AM67A datasheet](https://www.ti.com/product/AM67A).
    - **TIDL subgraph offload is not enabled on BeagleY-AI.** That
      board uses MMALIB direct offload only (`--tidl OFF --mmalib ON`).

## License

TVM is licensed under the Apache-2.0 license (`LICENSE` in the source
tree). For the full open-source license manifest and export
classification for this release, see
`TI_TVM_for_C7x_MMA_0.23.0_manifest.html` in the repository root.

## Supported Targets

| Target | Device | DSP |
|--------|--------|-----|
| `c_static -mcpu=c7x` | J722S / AM67A | C7™ DSP |

Two boards are supported via `--board`/`--ddr` on the runtime and
firmware build scripts (see the usage header in
`src/runtime/ti_dsp/build_runtime.sh`); `--board` is required for any
hardware build -- there is no default:

- **[AM67A EVM](https://www.ti.com/tool/J722SXH01EVM)** (`j722s-evm`)
  -- TI's evaluation module, orderable directly from TI.
- **[BeagleY-AI](https://www.beagleboard.org/boards/beagley-ai)**
  (`beagley-ai`) -- an open-hardware single-board computer built around
  the same J722S SoC.

Also supports host emulation (x86 GCC) for development and CI without
hardware.

## Documentation

- **User Guide**: [Getting Started](user-guide/getting-started.md),
  [Deploying Firmware](user-guide/deploying-firmware.md),
  [Verifying Your Deployment](user-guide/verifying-deployment.md),
  [Examples](user-guide/examples.md),
  [Python / C++ API Reference](user-guide/python-api.md)
- **Contributor Guide**: [Architecture Overview](contributor-guide/architecture-overview.md)
  and the Backend/DSP Runtime/Firmware/Testing sections in the nav

## Important Notice and Disclaimer

TI PROVIDES TECHNICAL AND RELIABILITY DATA (INCLUDING DATASHEETS), DESIGN
RESOURCES (INCLUDING REFERENCE DESIGNS), APPLICATION OR OTHER DESIGN
ADVICE, WEB TOOLS, SAFETY INFORMATION, AND OTHER RESOURCES "AS IS" AND
WITH ALL FAULTS, AND DISCLAIMS ALL WARRANTIES, EXPRESS AND IMPLIED,
INCLUDING WITHOUT LIMITATION ANY IMPLIED WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE OR NON-INFRINGEMENT OF THIRD PARTY
INTELLECTUAL PROPERTY RIGHTS.

These resources are intended for skilled developers designing with TI
products. You are solely responsible for (1) selecting the appropriate
TI products for your application, (2) designing, validating and testing
your application, and (3) ensuring your application meets applicable
standards, and any other safety, security, or other requirements. These
resources are subject to change without notice. TI grants you permission
to use these resources only for development of an application that uses
the TI products described in the resource. Other reproduction and
display of these resources is prohibited. No license is granted to any
other TI intellectual property right or to any third party intellectual
property right. TI disclaims responsibility for, and you will fully
indemnify TI and its representatives against, any claims, damages,
costs, losses, and liabilities arising out of your use of these
resources.

TI's products are provided subject to TI's
[Terms of Sale](https://www.ti.com/legal/termsofsale.html) or other
applicable terms available either on ti.com or provided in conjunction
with such TI products. TI's provision of these resources does not
expand or otherwise alter TI's applicable warranties or warranty
disclaimers for TI products.

Mailing Address: Texas Instruments, Post Office Box 655303, Dallas,
Texas 75265
