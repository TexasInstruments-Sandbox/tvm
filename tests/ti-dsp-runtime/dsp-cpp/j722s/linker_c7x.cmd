/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * linker_c7x.cmd
 *
 * Linker command file for J722S C75 DSP - TVM Model Execution
 *
 * Memory Layout (standalone JTAG mode):
 *   L2SRAM:     2MB at 0x7E000000 (local to C75x_0)
 *   DDR (<4GB): ~58MB at 0xAD604000 (code + weights)
 *   DDR (>4GB): 128MB at 0x108000000 (TVM runtime heap)
 *
 * For TVM model inference:
 *   - Code goes in DDR_C7X_CODE (2MB, cached via MMU)
 *   - Model weights (.rodata.weights) go in DDR_C7X_MAIN (~55.7MB)
 *   - TVM runtime heap in extended DDR (128MB at 0x108000000)
 *   - L2 pool (~1.59MB) for frequently accessed tensors
 *
 * TVM heap bounds are derived from linker region definitions.
 */

--rom_model
-heap  0x40000                  /* 256KB standard heap in DDR (for malloc) */
-stack 0x30000                  /* 192KB stack (combined L2_STACK + old L2_HEAP) */
--args 0x1000
--diag_suppress=10068
--cinit_compression=off
-e _c_int00_secure

/*
 * Memory Region Definitions (used by both MEMORY directive and symbols)
 *
 * CODE IN DDR MODE: Most code runs from DDR (cached via MMU).
 * Only boot and MMU init code stays in L2. This maximizes L2 for TVM heap.
 *
 * L2 Layout (code in DDR):
 *   0x7E000000 - Vectors (16KB)
 *   0x7E004000 - Secure vectors (16KB)
 *   0x7E008000 - Boot code (4KB)
 *   0x7E009000 - L2 init code (64KB) - MMU init, must run before DDR cached
 *   0x7E019000 - Data/BSS (128KB)
 *   0x7E039000 - Stack (192KB) - combined with old heap space
 *   0x7E069000 - TVM L2 pool (~1.59MB)
 *   0x7E200000 - End of L2
 *
 * malloc heap (.sysmem) moved to DDR (256KB) - TVM uses its own pools
 */
#define L2_SCRATCH_BASE     0x7E069000
#define L2_SCRATCH_SIZE     0x00197000    /* ~1.59MB (expanded from 268KB) */

/* DDR code region - 2MB for application code (runs after MMU enabled) */
#define DDR_C7X_CODE_BASE   0xAD604000
#define DDR_C7X_CODE_SIZE   0x00200000    /* 2MB for application code */

/* DDR sysmem region - for C malloc heap (256KB) */
#define DDR_SYSMEM_BASE     (DDR_C7X_CODE_BASE + DDR_C7X_CODE_SIZE)  /* 0xAD804000 */
#define DDR_SYSMEM_SIZE     0x00040000    /* 256KB */

/* DDR main region - for model weights (.rodata.weights) */
#define DDR_C7X_MAIN_BASE   (DDR_SYSMEM_BASE + DDR_SYSMEM_SIZE)      /* 0xAD844000 */
#define DDR_C7X_MAIN_SIZE   0x037BC000    /* ~55.7MB (56MB - 256KB sysmem) */

/* Extended DDR region - for TVM runtime heap (above 4GB, 128MB) */
/* Virtual 0x108000000+ maps to physical 0x888000000+ per TI SDK memory map */
#define DDR_C7X_EXTENDED_BASE  0x108000000
#define DDR_C7X_EXTENDED_SIZE  0x08000000  /* 128MB */

/*
 * Memory Regions
 */
MEMORY
{
    /* ================================================================= */
    /* L2SRAM - Local to C7x core (2MB total)                            */
    /* CODE IN DDR MODE: Most code in DDR, only boot/MMU init in L2      */
    /* ================================================================= */
    L2_VECS     (RX):    org = 0x7E000000, len = 0x00004000    /* 16KB - Vectors */
    L2_SECVECS  (RX):    org = 0x7E004000, len = 0x00004000    /* 16KB - Secure vectors */
    L2_BOOT     (RX):    org = 0x7E008000, len = 0x00001000    /* 4KB - Boot code */
    L2_INIT     (RX):    org = 0x7E009000, len = 0x00010000    /* 64KB - Pre-MMU init code */
    L2_DATA     (RW):    org = 0x7E019000, len = 0x00020000    /* 128KB - Data/BSS */
    L2_STACK    (RW):    org = 0x7E039000, len = 0x00030000    /* 192KB - Stack (expanded) */
    L2_SCRATCH  (RW):    org = L2_SCRATCH_BASE, len = L2_SCRATCH_SIZE  /* ~1.59MB - TVM L2 pool */

    /* L2SRAM AUX (240KB) - separate address range */
    L2_AUX      (RWIX):  org = 0x7F000000, len = 0x0003C000    /* 240KB */

    /* ================================================================= */
    /* DDR Memory Regions (requires bootloader/JTAG init)                */
    /* Based on vision_apps memory map for C7x_1                         */
    /* ================================================================= */

    /* C7x_1 DDR code region */
    DDR_C7X_BOOT    (RWIX):  org = 0xAD200000, len = 0x00000400    /* 1KB - Boot */
    DDR_C7X_VECS    (RWIX):  org = 0xAD400000, len = 0x00004000    /* 16KB - DDR vectors */
    DDR_C7X_SECVECS (RWIX):  org = 0xAD600000, len = 0x00004000    /* 16KB - Secure vectors */

    /* C7x_1 DDR code region (2MB) for application code with MMU */
    DDR_C7X_CODE    (RWIX):  org = DDR_C7X_CODE_BASE, len = DDR_C7X_CODE_SIZE

    /* C7x_1 DDR sysmem region (256KB) for C malloc heap */
    DDR_SYSMEM      (RW):    org = DDR_SYSMEM_BASE, len = DDR_SYSMEM_SIZE

    /* C7x_1 DDR main region (~55.7MB) for TVM runtime (heap/weights) */
    DDR_C7X_MAIN    (RWIX):  org = DDR_C7X_MAIN_BASE, len = DDR_C7X_MAIN_SIZE

    /* Extended DDR for TVM runtime heap (above 4GB)
     * Combines DDR_C7X_1_LOCAL_HEAP + DDR_C7X_1_SCRATCH from TI SDK memory map.
     * Virtual: 0x108000000-0x110000000 -> Physical: 0x888000000-0x890000000
     * This frees DDR_C7X_MAIN for model weights (.rodata.weights)
     */
    DDR_C7X_EXTENDED  (RWIX): org = DDR_C7X_EXTENDED_BASE, len = DDR_C7X_EXTENDED_SIZE
}

/*
 * Section Placement
 */
SECTIONS
{
    /* ================================================================= */
    /* Vector Tables                                                     */
    /* ================================================================= */
    .vecs               >   L2_VECS         ALIGN(0x1000)
    .secure_vecs        >   L2_SECVECS      ALIGN(0x1000)

    /* ================================================================= */
    /* Boot and Code                                                     */
    /* ================================================================= */
    /* Boot code must be in L2 (runs before any init) */
    .text:_c_int00_secure > L2_BOOT         ALIGN(0x200)

    /* Pre-MMU init code stays in L2 (runs before DDR is cached) */
    /* This includes MMU init functions marked with .text:l2_init */
    .text:l2_init       >   L2_INIT         ALIGN(0x100)

    /* Application code runs from DDR (cached via MMU) */
    /* This maximizes L2 space for TVM data heap */
    .text               >   DDR_C7X_CODE    ALIGN(0x100)
    .const              >   DDR_C7X_CODE

    /* ================================================================= */
    /* Data Sections                                                     */
    /* ================================================================= */
    .data               >   L2_DATA
    .data:exception_info >  L2_DATA ALIGN(8)  /* Exception diagnostic info */
    .cinit              >   L2_DATA
    .init_array         >   L2_DATA
    .switch             >   L2_DATA
    .cio                >   L2_DATA
    .args               >   L2_DATA

    /* BSS - Zero initialized data */
    .bss: RUN_START(__BSS_START), RUN_END(__BSS_END) > L2_DATA

    /* ================================================================= */
    /* MMU Page Tables - Must be 4KB aligned                             */
    /* These are pre-computed tables for TVM DSP MMU setup               */
    /* ================================================================= */
    GROUP:              >   L2_DATA ALIGN(4096)
    {
        /* TVM DSP Runtime page tables (from c7x_mmu_tables.c) */
        .data:pte:pte_lvl0
        .data:pte:pte_lvl1
        .data:pte:pte_lvl2_40000000
        .data:pte:pte_lvl2_80000000
        /* SDK MMU tables (if any) */
        .data.Mmu_level1Table   : type=NOINIT
        .data.Mmu_level2Table   : type=NOINIT
    }

    /* ================================================================= */
    /* Stack and Heap                                                    */
    /* ================================================================= */
    .stack              >   L2_STACK
    .sysmem             >   DDR_SYSMEM      /* malloc heap in DDR (256KB) */

    /* ================================================================= */
    /* TVM DSP Runtime Memory Pools                                      */
    /* ================================================================= */
    /* L2 heap - for frequently accessed tensors (~1.59MB)               */
    /* .sysmem (malloc) moved to DDR, so no overlap concerns             */
    .tvm_l2_heap (NOLOAD) : {} > L2_SCRATCH
        RUN_START(__TVM_DSP_L2_HEAP_START)

    /* DDR heap - for runtime tensor allocations (128MB in extended DDR) */
    /* Uses combined DDR_C7X_1_LOCAL_HEAP + SCRATCH from TI SDK map      */
    /* Virtual 0x108000000+ maps to physical 0x888000000+ (high DDR)     */
    .tvm_ddr_heap (NOLOAD) : {} > DDR_C7X_EXTENDED
        RUN_START(__TVM_DSP_DDR_HEAP_START)

    /* ================================================================= */
    /* Model Weights -> DDR_C7X_MAIN (~55.7MB available)                 */
    /* Embedded weights use .rodata.weights section for easy placement   */
    /* ================================================================= */
    .rodata.weights     >   DDR_C7X_MAIN    ALIGN(64)

    /* ================================================================= */
    /* Large Data (other tensors) -> DDR                                 */
    /* Use: #pragma DATA_SECTION(var, ".fardata")                        */
    /* ================================================================= */
    .fardata            >   DDR_C7X_MAIN    ALIGN(64)
    .far                >   DDR_C7X_MAIN    ALIGN(64)

    /* DDR heap sections (NOLOAD - memory allocated at runtime) */
    /* These sections are unused but kept for SDK compatibility */
    .bss:ddr_local_mem          (NOLOAD) : {} > DDR_C7X_EXTENDED
    .bss:ddr_scratch_mem        (NOLOAD) : {} > DDR_C7X_EXTENDED

    /* ================================================================= */
    /* L2 AUX for fast temporary storage                                 */
    /* Use: #pragma DATA_SECTION(var, ".l2aux")                          */
    /* ================================================================= */
    .bss:l2mem      (NOLOAD)(NOINIT) : {} > L2_AUX
}

/*
 * Exported Symbols for TVM DSP Runtime
 *
 * These are derived from the #define macros at the top of this file.
 * __TVM_DSP_L2_HEAP_START and __TVM_DSP_DDR_HEAP_START are set via
 * RUN_START() in the SECTIONS above.
 */

/* L2 heap end - derived from L2_SCRATCH region */
__TVM_DSP_L2_HEAP_END   = L2_SCRATCH_BASE + L2_SCRATCH_SIZE;
__TVM_DSP_L2_HEAP_SIZE  = L2_SCRATCH_SIZE;

/* DDR heap - derived from DDR_C7X_EXTENDED region (128MB in high DDR) */
__TVM_DSP_DDR_HEAP_END   = DDR_C7X_EXTENDED_BASE + DDR_C7X_EXTENDED_SIZE;
__TVM_DSP_DDR_HEAP_SIZE  = DDR_C7X_EXTENDED_SIZE;
