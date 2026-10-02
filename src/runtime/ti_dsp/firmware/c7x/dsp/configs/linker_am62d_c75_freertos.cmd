/* AM62D (C7504) linker command file for the c7x_compute firmware.
 *
 * Derived from freertos_sdk_am62dx_12_00_00_22/examples/tisp/
 * fft2d_dsp_offload/fft2d_linux_dsp_offload/am62dx-evm/c75ss0-0_freertos/
 * ti-c7000/linker.cmd.  Exports the same __TVM_DSP_{L2,DDR}_HEAP_* symbols as
 * linker_c75_freertos.cmd (j722s): the runtime uses two pools (L2 + DDR) on
 * every device.  The MMU map lives in am62d_c75ss0.syscfg.
 */

--ram_model
-heap  0xF0000
-stack 0x20000
--args 0x1000
--diag_suppress=10068 /* to suppress no matching section error */
--cinit_compression=off
-e _c_int00_secure

#define DDR0_ALLOCATED_START  0x80000000
#define C7X_ALLOCATED_START DDR0_ALLOCATED_START + 0x19800000

#define C7X_RESOURCE_TABLE_BASE (C7X_ALLOCATED_START + 0x00100000)
#define C7X_BOOT_BASE           (C7X_ALLOCATED_START + 0x00200000)
#define C7X_VECTOR_BASE         (C7X_ALLOCATED_START + 0x00400000)
#define C7X_DDR_SPACE_BASE      (C7X_ALLOCATED_START + 0x00410000)

MEMORY
{
    L2SRAM (RWX):  org = 0x7E000000,          len = 0x00100000
    DDR0_RESERVED: org = 0x80000000,          len = 0x19800000   /*  Reserved for A53 OS */
    C7X_IPC_D:     org = C7X_ALLOCATED_START, len = 0x00100000   /*  1MB DDR */
    C7X_BOOT_D:    org = C7X_BOOT_BASE,       len = 0x400        /*  1024B DDR */
    C7X_VECS_D:    org = C7X_VECTOR_BASE,     len = 0x4000       /*  16KB DDR */
    C7X_CIO_MEM:   org = C7X_DDR_SPACE_BASE,  len = 0x1000       /*  4KB DDR */
    C7X_DDR_SPACE: org = C7X_DDR_SPACE_BASE+0x1000, len = 0x00BF0000-0x1000  /*  11.9MB - 4KB DDR  */
    /* For resource table */
    DDR_IPC_RESOURCE_TABLE_LINUX:      org = C7X_RESOURCE_TABLE_BASE, len = 0x400    /*  1024B DDR */
    /* IPC trace buffer */
    DDR_IPC_TRACE_LINUX: org = 0x99900400, len = 0xFFC00      /* 1023KB DDR */
    DDR_LOG_SHM_MEM             : ORIGIN = 0xA1000000, LENGTH = 0x40000
    /* Shared memory for RTOS NORTOS IPC */
    DDR_IPC_VRING_RTOS: org = 0xA0400000, len = 0x300000  /* 3MB DDR */

    /* Added for TVM */
    /* L2 for C7x_1 [ size 240.00 KB ] - Runtime only
     * Split: TVM L2 scratch pool gets the whole 240KB region.
     * The .bss:l2mem section (vision_apps placeholder) is unused. */
    L2RAM_C7x_1_AUX          ( RWIX ) : ORIGIN = 0x7F000000 , LENGTH = 0x0003C000
    /* L1 for C7x_1 [ size 16.00 KB ] - Runtime only */
    L2RAM_C7x_1_AUX_AS_L1    ( RWIX ) : ORIGIN = 0x7F03C000 , LENGTH = 0x00004000

    /* Memory for shared memory buffers in DDR [ size 172.00 MB ] */
    DDR_SHARED_MEM                    : ORIGIN = 0xA3000000 , LENGTH = 0x0AC00000

    /* Non-cacheable DDR regions: mapped by am62d_c75ss0.syscfg but not used
     * by the two-pool TVM runtime (reserved for DMA-buffer pools). */
    /* DDR for c7x_1 for non cacheable local heap [ size 16.00 MB ] */
    DDR_C7X_1_LOCAL_HEAP_NON_CACHEABLE ( RWIX ) : ORIGIN = 0xB0000000 , LENGTH = 0x01000000
    /* DDR for c7x_1 for non cacheable scratch Memory [ size 16.00 MB ] */
    DDR_C7X_1_SCRATCH_NON_CACHEABLE   ( RWIX ) : ORIGIN = 0xB1000000 , LENGTH = 0x01000000
    /* DDR for c7x_1 cacheable heaps - 112 MB each (total 224 MB)
     * MMU region "DDR_C7X_1_LOCAL_HEAP" in am62d_c75ss0.syscfg maps
     * 0xB2000000-0xC0000000 (224 MB).  TVM uses the first (persistent) one
     * as its DDR pool; the scratch heap is currently unused. */
    DDR_C7X_1_PERSISTENT_HEAP ( RWIX ) : ORIGIN = 0xB2000000 , LENGTH = 0x07000000
    DDR_C7X_1_SCRATCH_HEAP    ( RWIX ) : ORIGIN = 0xB9000000 , LENGTH = 0x07000000
}

SECTIONS
{
    boot:
    {
      boot.*<boot.oe71>(.text)
    } load > C7X_BOOT_D ALIGN(0x200000)
    .vecs       >       C7X_VECS_D ALIGN(0x400000)
    .secure_vecs    >   C7X_DDR_SPACE ALIGN(0x200000)
    .text:_c_int00_secure > C7X_DDR_SPACE ALIGN(0x200000)
    .text       >       C7X_DDR_SPACE ALIGN(0x200000)

    .bss        >       C7X_DDR_SPACE  /* Zero-initialized data */
    RUN_START(__BSS_START)
    RUN_END(__BSS_END)

    .data       >       C7X_DDR_SPACE  /* Initialized data */

    .cinit      >       C7X_DDR_SPACE  /* could be part of const */
    .init_array >       C7X_DDR_SPACE  /* C++ initializations */
    .stack      >       C7X_DDR_SPACE ALIGN(0x2000)
    .args       >       C7X_DDR_SPACE
    .cio        >       C7X_CIO_MEM
    .const      >       C7X_DDR_SPACE
    .switch     >       C7X_DDR_SPACE /* For exception handling. */
    .sysmem     >       C7X_DDR_SPACE /* heap */

    .staticData > C7X_DDR_SPACE

    GROUP:              >  C7X_DDR_SPACE
    {
        .data.Mmu_tableArray          : type=NOINIT
        .data.Mmu_tableArraySlot      : type=NOINIT
        .data.Mmu_level1Table         : type=NOINIT
        .data.gMmu_tableArray_NS       : type=NOINIT
        .data.Mmu_tableArraySlot_NS   : type=NOINIT
        .data.Mmu_level1Table_NS      : type=NOINIT
    }

    .benchmark_buffer:     > C7X_DDR_SPACE ALIGN (32)

    .ddrData    > C7X_DDR_SPACE
    .l2sramData > L2SRAM

    /* This is the resource table used by linux to know where the IPC "VRINGs" are located */
    .resource_table: { __RESOURCE_TABLE = .;} > DDR_IPC_RESOURCE_TABLE_LINUX
    /* This IPC log can be viewed via ROV in CCS and when linux is enabled, this log can also be viewed via linux debugfs */
    .bss.debug_mem_trace_buf    : {} palign(128)    > DDR_IPC_TRACE_LINUX
    /* this is used when Debug log's to shared memory is enabled, else this is not used */
    .bss.log_shared_mem  (NOLOAD) : {} > DDR_LOG_SHM_MEM
    /* this is used only when IPC RPMessage is enabled */
    .bss.ipc_vring_mem   (NOLOAD) : {} > DDR_IPC_VRING_RTOS

    /* Added for TVM */
    /*
     * TVM DSP Runtime Memory Pools
     *
     * RUN_START/RUN_END produce linker symbols from the actual section
     * placement — no hardcoded addresses needed.
     *
     * Memory hierarchy (matches vision_apps AM62A convention):
     *   L3 scratch: L2SRAM / L2RAM_C7x_1_MAIN @ 0x7E000000, 1 MB
     *               (extended L2 used as L3; not used by TVM pools)
     *   L2 scratch: L2RAM_C7x_1_AUX @ 0x7F000000, 240 KB
     *   L1 scratch: L2RAM_C7x_1_AUX_AS_L1 @ 0x7F03C000, 16 KB
     *   TVM fast pool: all 240 KB of L2RAM_C7x_1_AUX (via .bss:tvm_l2_heap);
     *               also the L2 base/size used by generated code
     *   TVM DDR pool:  DDR_C7X_1_PERSISTENT_HEAP (112 MB, cacheable)
     */
    .bss:tvm_l2_heap        (NOLOAD)(NOINIT) : { . = . + 0x3C000; } > L2RAM_C7x_1_AUX
        RUN_START(__TVM_DSP_L2_HEAP_START)
        RUN_END(__TVM_DSP_L2_HEAP_END)

    .bss:tvm_ddr_heap       (NOLOAD)(NOINIT) : { . = . + 0x07000000; } > DDR_C7X_1_PERSISTENT_HEAP
        RUN_START(__TVM_DSP_DDR_HEAP_START)
        RUN_END(__TVM_DSP_DDR_HEAP_END)

    /*
     * L2/L3 SRAM scratch sections — Runtime only (NOLOAD/NOINIT)
     * IMPORTANT: Remoteproc CANNOT load into L2SRAM — these regions
     * are only accessible after the DSP starts executing.
     *
     * Naming follows the vision_apps AM62A convention:
     *   .bss:l3mem → L2SRAM (L2 MAIN, 1 MB) — extended L2 used as L3
     *   .bss:l2mem → L2RAM_C7x_1_AUX (240 KB) — L2 AUX as L2 scratch
     *   .bss:l1mem → L2RAM_C7x_1_AUX_AS_L1 (16 KB) — top of AUX as L1
     *
     * g_l3_mem_addr, g_l2_mem_addr, g_l1_mem_addr in tidl_support.c
     * point directly to these regions at runtime (no static arrays needed).
     */
    .bss:l3mem              (NOLOAD)(NOINIT) : {} > L2SRAM
    .bss:l2mem              (NOLOAD)(NOINIT) : {} > L2RAM_C7x_1_AUX
    .bss:l1mem              (NOLOAD)(NOINIT) : {} > L2RAM_C7x_1_AUX_AS_L1


}
