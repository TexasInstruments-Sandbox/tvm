# C7x Memory Map Reference

This describes the memory layout used by the standalone JTAG test harness
(see [DSP C++ Harness](../testing/dsp-cpp-harness.md)), defined in
`tests/ti-dsp-runtime/dsp-cpp/j722s/linker_c7x.cmd`.

**This is a separate, independent memory map from the firmware's unified
DDR pool** (see [Firmware Design Deep-Dive](../firmware/design-deep-dive.md))
-- the two build targets are not related, and their DDR heap sizes
genuinely differ (128 MB here vs. 352 MiB in the firmware). Don't cross-check
numbers between the two; each is correct for its own linker script.

## J722S C7x Memory Layout

The J722S C75 DSP has 2MB of L2 SRAM local to the core plus access to DDR memory.
The linker command file (`j722s/linker_c7x.cmd`) defines the memory layout for
standalone JTAG execution.

### Code-in-DDR Mode (Default)

The default configuration places application code in DDR (cached via MMU) to
maximize L2 SRAM for the TVM data heap. Only boot code and MMU init remain in L2.

### L2 SRAM Regions (2MB at 0x7E000000)

| Region | Address Range | Size | Purpose |
|--------|---------------|------|---------|
| L2_VECS | 0x7E000000 - 0x7E003FFF | 16KB | Interrupt/exception vectors |
| L2_SECVECS | 0x7E004000 - 0x7E007FFF | 16KB | Secure vectors |
| L2_BOOT | 0x7E008000 - 0x7E008FFF | 4KB | Boot code (`_c_int00_secure`) |
| L2_INIT | 0x7E009000 - 0x7E018FFF | 64KB | Pre-MMU init code (`.text:l2_init`) |
| L2_DATA | 0x7E019000 - 0x7E038FFF | 128KB | Data sections (`.data`, `.bss`, `.cio`) |
| L2_STACK | 0x7E039000 - 0x7E068FFF | **192KB** | Stack (`.stack`) - expanded |
| L2_SCRATCH | 0x7E069000 - 0x7E1FFFFF | **1.59MB** | **TVM L2 pool** (`.tvm_l2_heap`) |
| L2_AUX | 0x7F000000 - 0x7F03BFFF | 240KB | Auxiliary storage (`.l2aux`) |

**Note:** The standard malloc heap (`.sysmem`) has been moved to DDR to maximize L2 for
the TVM runtime. This allows a larger stack (192KB vs 128KB) and keeps all of L2_SCRATCH
available for TVM tensor allocations.

**Key difference from code-in-L2 mode**: Application code (`.text`) goes to DDR,
freeing ~1.3MB additional L2 space for the TVM heap.

### DDR Regions

| Region | Address Range | Size | Purpose |
|--------|---------------|------|---------|
| DDR_C7X_BOOT | 0xAD200000 | 1KB | DDR boot code |
| DDR_C7X_VECS | 0xAD400000 | 16KB | DDR vectors |
| DDR_C7X_SECVECS | 0xAD600000 | 16KB | DDR secure vectors |
| DDR_C7X_CODE | 0xAD604000 - 0xAD803FFF | 2MB | Application code (cached via MMU) |
| DDR_SYSMEM | 0xAD804000 - 0xAD843FFF | 256KB | Standard malloc heap (`.sysmem`) |
| DDR_C7X_MAIN | 0xAD844000 - 0xB0FFFFFF | ~55.7MB | **Model weights** (`.rodata.weights`) |
| DDR_C7X_EXTENDED | 0x108000000 - 0x10FFFFFFF | 128MB | **TVM DDR heap** (runtime tensors) |

**Note:** DDR_C7X_EXTENDED uses high DDR above 4GB (virtual addresses 0x108000000+
map to physical 0x888000000+ per TI SDK memory map). This requires 8GB LPDDR4 and
is verified by the TI RTOS SDK `app_mem_map.h`.

### TVM Runtime Memory Pools

The TVM DSP runtime uses two memory pools configured via linker symbols:

| Pool | Region | Address Range | Size | Usage |
|------|--------|---------------|------|-------|
| L2 (Fast) | L2_SCRATCH | 0x7E069000 - 0x7E1FFFFF | **1.59MB** | Intermediate tensors, frequently accessed data |
| DDR (Main) | DDR_C7X_EXTENDED | 0x108000000 - 0x110000000 | **128MB** | Runtime tensor allocations, large outputs |

**Model weights** are placed in DDR_C7X_MAIN (~55.7MB) via the `.rodata.weights` section,
separate from the runtime heap.

**Linker symbols** (read by TVM runtime at initialization):
```
__TVM_DSP_L2_HEAP_START  = 0x7E069000
__TVM_DSP_L2_HEAP_END    = 0x7E200000
__TVM_DSP_DDR_HEAP_START = 0x108000000
__TVM_DSP_DDR_HEAP_END   = 0x110000000
```

### Important: Memory Pool Separation

The TVM heaps **must not overlap** with `.sysmem` (standard malloc heap used by
`printf`, `fopen`, etc.). Earlier versions had overlap issues causing memory
corruption when printf's internal buffers overwrote TVM tensor data.

**Current layout (correct, code-in-DDR mode):**
- `.sysmem` (malloc) → DDR_SYSMEM (0xAD804000, 256KB) - in low DDR
- `.stack` → L2_STACK (0x7E039000, 192KB) - expanded
- `.rodata.weights` → DDR_C7X_MAIN (0xAD844000, ~55.7MB) - model weights
- TVM L2 pool → L2_SCRATCH (0x7E069000, 1.59MB)
- TVM DDR pool → DDR_C7X_EXTENDED (0x108000000, 128MB) - in high DDR (>4GB)

**Why use extended DDR for TVM heap:** The J722S has 8GB LPDDR4 with high DDR
addresses (0x108000000+) mapped via MMU. Using extended DDR for the TVM runtime
heap frees DDR_C7X_MAIN for model weights (up to ~55.7MB), enabling larger models.

To customize the TVM pool locations, modify the linker symbols in `linker_c7x.cmd`.
The runtime reads these symbols at initialization, so no library rebuild is needed.

### Section Placement Summary

| Section | Region | Description |
|---------|--------|--------------|
| `.vecs` | L2_VECS | Interrupt vector table |
| `.text` | DDR_C7X_CODE | Application code (cached via MMU) |
| `.text:l2_init` | L2_INIT | Pre-MMU init code (runs before DDR cached) |
| `.const` | DDR_C7X_CODE | Read-only constants (small) |
| `.rodata.weights` | DDR_C7X_MAIN | **Model weights** (up to ~55.7MB) |
| `.data` | L2_DATA | Initialized global data |
| `.bss` | L2_DATA | Zero-initialized data |
| `.cio` | L2_DATA | Console I/O buffer (printf) |
| `.stack` | L2_STACK | Program stack (192KB) |
| `.sysmem` | DDR_SYSMEM | Standard heap (malloc/free) - 256KB in DDR |
| `.tvm_l2_heap` | L2_SCRATCH | TVM fast memory pool (1.59MB) |
| `.tvm_ddr_heap` | DDR_C7X_EXTENDED | TVM main memory pool (128MB in high DDR) |
| `.fardata` | DDR_C7X_MAIN | Large data arrays |

## C7x MMU Configuration

The C7x DSP on J722S requires MMU (Memory Management Unit) configuration for
cached and executable DDR access. The application is responsible for
initializing the MMU before calling any TVM runtime functions.

### Why MMU is Needed

Without proper MMU configuration, DDR memory accesses are:
- Uncached (slow, every access goes to external memory)
- Potentially non-executable (can't run code from DDR)

The MMU enables:
- Cached DDR access for model weights and tensors
- Executable DDR regions for code-in-DDR mode (maximizes L2 for data)
- Proper memory attributes for L2 SRAM and peripheral regions

### MMU Register Configuration

The MMU uses direct ECR (Extended Control Register) access, adapted from
TI's edgeai-tidl-kernels approach:

| Register | ECR | Value | Description |
|----------|-----|-------|-------------|
| SCR | ECR784 | 0x80000000000000C1 | System Control: MMU + caches enabled |
| TCR0 | ECR785 | 0x0000000000002A21 | Translation Control: 4KB granule, 2GB space |
| TBR0 | ECR787 | (page table addr) | Translation Base: points to level 1 table |
| MAR | ECR789 | 0x3D3D3D2915032A00 | Memory Attributes: cacheable/device types |

### MAIR (Memory Attribute Indirection Register)

The MAR value packs 8 memory attribute configurations (MAIR0-7):

| Index | Value | Description |
|-------|-------|-------------|
| MAIR0 | 0x00 | Device-nGnRnE (strongly ordered device memory) |
| MAIR1 | 0x2A | Write-Through No-Allocate |
| MAIR2 | 0x03 | Device-nGnRE |
| MAIR3 | 0x15 | Write-Through Allocate |
| MAIR4 | 0x29 | Non-cacheable |
| MAIR5 | 0x3D | Write-Back Read-Allocate Write-Allocate (normal cached) |
| MAIR6 | 0x3D | Write-Back Read-Allocate Write-Allocate |
| MAIR7 | 0x3D | Write-Back Read-Allocate Write-Allocate |

### Page Table Structure

The J722S implementation uses ARMv8-style 2-level page tables with 1GB L1 blocks
and 2MB L2 blocks:

```
Level 1 (512 entries)         Level 2 (512 entries)
┌─────────────────────┐       ┌─────────────────────┐
│ [0] 0x00-0x3F: Dev  │ Block │                     │
│ [1] 0x40-0x7F: Tbl  │──────>│ MSMC (0x70): Cached │
│ [2] 0x80-0xBF: DDR  │ Block │ L2   (0x7E): Cached │
│ [3] 0xC0-0xFF: DDR  │ Block │ L2AUX(0x7F): Cached │
│ [4] 1.0-1.3G: DDR   │ Block │                     │
│ [5] 1.4-1.7G: DDR   │ Block └─────────────────────┘
└─────────────────────┘
```

### Memory Regions

| Region | Address Range | Size | Attributes |
|--------|---------------|------|------------|
| Peripherals | 0x00000000-0x3FFFFFFF | 1GB | Device, Non-Shareable |
| Secure Proxy | 0x48000000-0x4FFFFFFF | 128MB | Device (for DMSC comm) |
| MSMC | 0x70000000-0x703FFFFF | 4MB | Cached, Outer Shareable |
| L2 SRAM | 0x7E000000-0x7E1FFFFF | 2MB | Cached, Non-Shareable |
| L2 AUX | 0x7F000000-0x7F03FFFF | 256KB | Cached, Non-Shareable |
| DDR | 0x80000000-0xFFFFFFFF | 2GB | Cached, Outer Shareable |
| DDR (ext) | 0x100000000-0x17FFFFFFF | 2GB | Cached, Outer Shareable (TVM heap at 0x108000000) |

### Block Descriptor Attributes

| Attribute | Bits | Values |
|-----------|------|--------|
| Type | [1:0] | 0b01 = Block descriptor |
| AttrIndx | [4:2] | MAIR index (0-7) |
| NS | [5] | Non-Secure |
| AP | [7:6] | Access permissions (0b00 = RW) |
| SH | [9:8] | Shareability (0=NSH, 2=OSH, 3=ISH) |
| AF | [10] | Access Flag (must be 1) |
| Address | [47:21] | 2MB-aligned physical address (L2) |

### Shareability Considerations

- **Non-Shareable (NSH)**: Use for L2 SRAM which is local to each C75 core
- **Outer Shareable (OSH)**: Use for DDR and MSMC for multi-core coherency

### MMU Source Files

The MMU implementation is in `j722s/`:

| File | Description |
|------|--------------|
| `mmu.c` | MMU initialization with detailed comments |
| `c75_asm.S` | Assembly functions for ECR register access |
| `boot_c75.c` | Boot code that calls `MmuP_init()` before `main()` |
| `linker_c7x.cmd` | Linker script with page table sections |

### Security Mode

The C7x runs in CXM=3 (RootSupervisor) mode when loaded via JTAG. This mode:
- Has full access to MMU configuration registers
- Can modify page tables and enable/disable MMU
- Requires direct ECR register access (SDK MmuP functions may not work)

The runtime detects the security mode and reports it during initialization:
```
C7x security mode: CXM=3 (RootSupervisor)
```
