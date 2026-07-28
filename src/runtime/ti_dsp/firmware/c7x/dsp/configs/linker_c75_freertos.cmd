/*
 * Linker command file for hello_world_rproc - Vision Apps Compatible
 *
 * This linker file matches the vision_apps firmware layout exactly.
 * It uses the same memory regions and section placements to ensure
 * MMU configuration compatibility.
 *
 * Key differences from standalone:
 * - Entry point is _c_int00_secure (SDK standard)
 * - L2SRAM regions are NOINIT/NOLOAD (remoteproc cannot load into L2)
 * - Resource table at 0xAD100000 for Linux remoteproc
 * - IPC trace buffer for debug output
 *
 * Memory map based on:
 * - vision_apps/platform/j722s/rtos/c7x_1/linker_mem_map.cmd
 * - vision_apps/platform/j722s/rtos/c7x_1/j722s_linker_freertos.cmd
 */

--ram_model
-heap  0x20000
-stack 0x100000
--args 0x1000
--diag_suppress=10068 /* "no matching section" */
--cinit_compression=off

/* Standard SDK entry point - MMU enabled during C runtime init */
-e _c_int00_secure

/* Retain symbols needed by Linux remoteproc */
--retain=gRPMessage_linuxResourceTable
--retain=gDebugMemLog

/*
 * =============================================================================
 * MEMORY REGIONS - Exact copy from vision_apps linker_mem_map.cmd
 * =============================================================================
 */
MEMORY
{
    /* L2 SRAM for C7x_1 [ size 1.25 MB ] - Runtime only, cannot be loaded
     * J722S TRM: 1.25MB L2 SRAM with ECC protection. */
    L2RAM_C7x_1_MAIN         ( RWIX ) : ORIGIN = 0x7E000000 , LENGTH = 0x00140000
    /* L2 for C7x_1 [ size 240.00 KB ] - Runtime only
     * Split: TVM L2 scratch pool gets the whole 240KB region.
     * The .bss:l2mem section (vision_apps placeholder) is unused. */
    L2RAM_C7x_1_AUX          ( RWIX ) : ORIGIN = 0x7F000000 , LENGTH = 0x0003C000
    /* L1 for C7x_1 [ size 16.00 KB ] - Runtime only */
    L2RAM_C7x_1_AUX_AS_L1    ( RWIX ) : ORIGIN = 0x7F03C000 , LENGTH = 0x00004000

    /*
     * Memory for IPC Vring's. MUST be non-cached or cache-coherent
     * IMPORTANT: Must be within the Linux carveout (0xAD000000-0xB0FFFFFF)
     * Placed at end of carveout to avoid conflicts with code/data
     * Size reduced to 8MB to fit within carveout
     */
    IPC_VRING_MEM                     : ORIGIN = 0xAF800000 , LENGTH = 0x00800000
    /* Memory for remote core logging [ size 256.00 KB ] */
    APP_LOG_MEM                       : ORIGIN = 0xA7000000 , LENGTH = 0x00040000
    /* Memory for TI OpenVX shared memory. MUST be non-cached or cache-coherent [ size 63.75 MB ] */
    TIOVX_OBJ_DESC_MEM                : ORIGIN = 0xA7040000 , LENGTH = 0x03FC0000
    /* Memory for remote core file operations [ size 4.00 MB ] */
    APP_FILEIO_MEM                    : ORIGIN = 0xAB000000 , LENGTH = 0x00400000

    /* DDR for C7x_1 for Linux IPC [ size 1024.00 KB ] */
    DDR_C7x_1_IPC            ( RWIX ) : ORIGIN = 0xAD000000 , LENGTH = 0x00100000
    /* DDR for C7x_1 for Linux resource table [ size 1024 B ] */
    DDR_C7x_1_RESOURCE_TABLE ( RWIX ) : ORIGIN = 0xAD100000 , LENGTH = 0x00000400
    /* DDR for C7x_1 for Linux IPC trace [ size 1023.00 KB ] */
    DDR_C7x_1_IPC_TRACE      ( RWIX ) : ORIGIN = 0xAD100400 , LENGTH = 0x000FFC00
    /* DDR for C7x_1 for boot section [ size 1024 B ] */
    DDR_C7x_1_BOOT           ( RWIX ) : ORIGIN = 0xAD200000 , LENGTH = 0x00000400
    /* DDR for C7x_1 for vecs section [ size 16.00 KB ] */
    DDR_C7x_1_VECS           ( RWIX ) : ORIGIN = 0xAD400000 , LENGTH = 0x00004000
    /* DDR for C7x_1 for secure vecs section [ size 16.00 KB ] */
    DDR_C7x_1_SECURE_VECS    ( RWIX ) : ORIGIN = 0xAD600000 , LENGTH = 0x00004000
    /* DDR for C7x_1 for code/data [ size ~34MB - reduced to make room for IPC_VRING at 0xAF800000 ] */
    DDR_C7x_1                ( RWIX ) : ORIGIN = 0xAD604000 , LENGTH = 0x021FC000

    /* Memory for shared memory buffers in DDR [ size 512.00 MB ] */
    DDR_SHARED_MEM                    : ORIGIN = 0xC0000000 , LENGTH = 0x20000000

    /* DDR for c7x_1 for local heap + scratch [ size 352.00 MB ]
     * Merged LOCAL_HEAP (64 MB) + SCRATCH (64 MB) + c7x_2 regions (128 MB)
     * + the vision_apps non-cacheable heap/scratch range (96 MB, at
     * 0x102000000-0x108000000 -- unused here: this firmware never places
     * anything in .bss:ddr_non_cache_mem/.bss:ddr_scratch_non_cache_mem,
     * unlike vision_apps' app_init.c) into a single contiguous cacheable
     * region for TVM runtime + DLOAD allocations.  c7x_2 regions are
     * unused on J722S (single C7x core).
     * MMU Region 13 in c75ss0.syscfg maps 0x102000000-0x118000000 (352 MB)
     * exactly -- this region now spans the same range. */
    DDR_C7X_1_LOCAL_HEAP     ( RWIX ) : ORIGIN = 0x102000000 , LENGTH = 0x16000000
}

/*
 * =============================================================================
 * SECTIONS - Based on vision_apps j722s_linker_freertos.cmd
 * =============================================================================
 */
SECTIONS
{
    /*
     * Boot code - SDK's _c_int00_secure entry point
     * Must be 2MB aligned for MMU
     */
    boot:
    {
        boot.*<boot.oe71>(.text)
    } load > DDR_C7x_1_BOOT ALIGN(0x200000)

    .text:_c_int00_secure > DDR_C7x_1_BOOT ALIGN(0x200000)

    /* Vector tables */
    .vecs           >   DDR_C7x_1_VECS ALIGN(0x400000)
    .secure_vecs    >   DDR_C7x_1_SECURE_VECS ALIGN(0x100000)

    /* Code section */
    .text           >   DDR_C7x_1 ALIGN(0x200000)

    /* Zero-initialized data (BSS) */
    .bss            >   DDR_C7x_1
    RUN_START(__BSS_START)
    RUN_END(__BSS_END)

    /* Initialized data */
    .data           >   DDR_C7x_1

    /* C runtime initialization */
    .cinit          >   DDR_C7x_1
    .init_array     >   DDR_C7x_1

    /* Stack - 128KB aligned for nested interrupts */
    .stack          >   DDR_C7x_1 ALIGN(0x20000)

    /* Arguments */
    .args           >   DDR_C7x_1

    /* CIO for printf */
    .cio            >   DDR_C7x_1

    /* Constants */
    .const          >   DDR_C7x_1
    .switch         >   DDR_C7x_1

    /* Heap for malloc/FreeRTOS */
    .sysmem         >   DDR_C7x_1

    /* FreeRTOS task stacks */
    .bss:taskStackSection > DDR_C7x_1

    /*
     * DDR heap sections - for TVM/TIDL large allocations
     * These use virtual addresses that MMU translates to physical DDR
     *
     * Note: DDR_C7X_1_LOCAL_HEAP is shared between .bss:ddr_local_mem
     * (vision_apps, unused) and .bss:tvm_ddr_heap (TVM runtime).
     * The TI linker packs them sequentially; since the vision_apps
     * section is empty, TVM gets the full region.
     */
    .bss:ddr_local_mem          (NOLOAD) : {} > DDR_C7X_1_LOCAL_HEAP

    /*
     * TVM DSP Runtime Memory Pools
     *
     * L2 scratch pool: Carved from L2 AUX SRAM for fast TVM allocations
     * DDR heap: Uses the cacheable local heap region for TVM + DLOAD
     *
     * These symbols are read by the TVM runtime's c7x_platform.c
     * during tvm_dsp_platform_init().
     */
    /*
     * TVM DSP Runtime Memory Pools
     *
     * RUN_START/RUN_END produce linker symbols from the actual
     * section placement — no hardcoded addresses needed.
     *
     * L2 scratch: L2RAM_C7x_1_AUX (128 KB) — fast temporary tensors
     * DDR heap:   DDR_C7X_1_LOCAL_HEAP (352 MB) — large allocations,
     *             DLOAD code segments, model constants
     */
    .bss:tvm_l2_heap        (NOLOAD)(NOINIT) : { . = . + 0x20000; } > L2RAM_C7x_1_AUX
        RUN_START(__TVM_DSP_L2_HEAP_START)
        RUN_END(__TVM_DSP_L2_HEAP_END)

    .bss:tvm_ddr_heap        (NOLOAD) : { . = . + 0x16000000; } > DDR_C7X_1_LOCAL_HEAP
        RUN_START(__TVM_DSP_DDR_HEAP_START)
        RUN_END(__TVM_DSP_DDR_HEAP_END)

    /* Vision apps shared memory sections (optional - for future use) */
    .bss:app_log_mem        (NOLOAD) : {} > APP_LOG_MEM
    .bss:app_fileio_mem     (NOLOAD) : {} > APP_FILEIO_MEM
    .bss:tiovx_obj_desc_mem (NOLOAD) : {} > TIOVX_OBJ_DESC_MEM

    /* IPC VRing memory - shared with Linux for RPMessage */
    /* NOTE: Section name uses dot not colon to match SysConfig generated code */
    .bss.ipc_vring_mem      (NOLOAD) : {} > IPC_VRING_MEM

    /*
     * L2SRAM sections - Runtime only (NOLOAD/NOINIT)
     * IMPORTANT: Remoteproc CANNOT load into L2SRAM!
     * These are only available after DSP starts.
     */
    /*
     * L2/L3 SRAM sections - Runtime only (NOLOAD/NOINIT)
     * Empty vision_apps placeholders; not used by TVM firmware.
     * L2RAM_C7x_1_MAIN (1.25 MB) is accessed directly by the
     * bump allocator at runtime (see c7x_platform.c).
     */
    .bss:l1mem              (NOLOAD)(NOINIT) : {} > L2RAM_C7x_1_AUX_AS_L1
    .bss:l2mem              (NOLOAD)(NOINIT) : {} > L2RAM_C7x_1_AUX

    /* IPC data buffer */
    ipc_data_buffer         >   DDR_C7x_1

    /* Trace buffer - align for DMA */
    .tracebuf               : {} align(1024) > DDR_C7x_1

    /*
     * Resource table - Linux remoteproc parses this to find:
     * - VRINGs for IPC
     * - Trace buffer location
     * - Carveout memories
     */
    .resource_table:
    {
        __RESOURCE_TABLE = .;
        *(.resource_table*)
    } > DDR_C7x_1_RESOURCE_TABLE

    /* Debug trace buffer - visible via /sys/kernel/debug/remoteproc/.../trace0 */
    .bss.debug_mem_trace_buf    > DDR_C7x_1_IPC_TRACE

    /*
     * MMU page tables - SDK allocates these during MmuP_init
     * Must be in DDR (not L2) for remoteproc compatibility
     */
    GROUP:              >  DDR_C7x_1
    {
        .data.Mmu_tableArray          : type=NOINIT
        .data.Mmu_tableArraySlot      : type=NOINIT
        .data.Mmu_level1Table         : type=NOINIT
        .data.gMmu_tableArray_NS      : type=NOINIT
        .data.Mmu_tableArraySlot_NS   : type=NOINIT
        .data.Mmu_level1Table_NS      : type=NOINIT
    }
}

