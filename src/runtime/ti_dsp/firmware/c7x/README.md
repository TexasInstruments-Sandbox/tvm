# C7x Firmware

Host-DSP compute service for TI J722S/AM67A that enables Linux
applications to offload data processing and ML inference to the C7x
DSP via RPMessage IPC and shared DDR memory, including a dynamic
module loader (DLOAD) for loading TVM-compiled C7x ELF modules at
runtime.

See [Deploying Firmware](../../../../../docs-c7x/user-guide/deploying-firmware.md)
in the docs site for build/deploy/usage/troubleshooting, and
[Firmware Architecture](../../../../../docs-c7x/contributor-guide/firmware/architecture.md)
for the internal design (see also
[Firmware Design Deep-Dive](../../../../../docs-c7x/contributor-guide/firmware/design-deep-dive.md)
for the full protocol/memory-layout/DLOAD reference).
