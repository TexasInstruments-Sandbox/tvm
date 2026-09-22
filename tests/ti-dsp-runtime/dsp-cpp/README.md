# TVM DSP Integration

Infrastructure for running TVM-compiled models on TI DSP hardware (C66x
and C7x) using the minimal DSP runtime, including the standalone JTAG
test harness and its J722S memory/MMU configuration.

See [DSP C++ Harness Build Reference](../../../docs-c7x/contributor-guide/testing/verifying-deployment.md#dsp-c-harness-build-reference)
in the docs site for build/usage instructions, and the standalone JTAG
harness's reference MMU implementation (`j722s/mmu.c`) for the memory
layout and MMU configuration.
