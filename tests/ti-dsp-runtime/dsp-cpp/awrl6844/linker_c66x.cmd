/* Linker command file for AWRL6844 C66x DSP - CLISTA-DoA Radar Model
 *
 * Adapted from hw_awrl6844_c66x/linker.cmd for TVM C static model execution.
 * Memory layout optimized for CLISTA-DoA neural network inference.
 */

/* Stack used by code running within main() in NORTOS mode */
--stack_size=65536  /* 64KB stack */

/* Heap size for malloc() - small since we use HeapP for L3 allocation.
 * This heap is only for CIO and standard library needs. */
--heap_size=65536  /* 64KB - placed in DSS_L3 */

--retain=_vectors

SECTIONS
{
    /* Hard addresses force vectors to be allocated at start of L2 */
    .text:vectors: {. = align(1024); } > 0x00800000

    /* Code sections - place in L2 for fast execution */
    .text:      {} > DSS_L2
    .const:     {} > DSS_L2
    .cinit:     {} > DSS_L2
    .switch:    {} > DSS_L2

    /* Data sections - place in L2 */
    .data:      {} > DSS_L2
    .bss:       {} > DSS_L2
    .stack:     {} > DSS_L2
    .sysmem:    {} > DSS_L3  /* Standard heap (malloc) in L3 to free L2 for code */
    .cio:       {} > DSS_L2

    /* Far data sections */
    .fardata:   {} > DSS_L2
    .far:       {} > DSS_L2
    .neardata:  {} > DSS_L2

    /* Grouped together to avoid STATIC_BASE relative relocation errors */
    GROUP {
        .rodata:    {}
    } > DSS_L2

    /* C++ support sections */
    GROUP {
        .c6xabi.exidx:  {} palign(8)   /* C++ exception handling */
        .init_array:    {} palign(8)   /* Function pointers called before main */
        .fini_array:    {} palign(8)   /* Function pointers called after main */
    } > DSS_L2

    /* Large data buffers can be placed in L3 by assigning section name .bss.dss_l3 */
    .bss.dss_l3 {} > DSS_L3

    /* Dedicated test heap regions - isolated to avoid SDK relocation issues */
    .bss.test_heap_l2 {} > TEST_L2_HEAP
    .bss.test_heap_l3 {} > TEST_L3_HEAP

    /* User shared memory (not typically used by layer tests) */
    .bss.user_shared_mem (NOLOAD) : {} > USER_SHM_MEM

    /* Debug log shared memory */
    .bss.log_shared_mem  (NOLOAD) : {} > LOG_SHM_MEM

    /* IPC shared memory (not typically used by layer tests) */
    .ipc_sh_mem {} > IPC_SH_MEM
}

MEMORY
{
    /* Memory map for AWRL6844 C66x DSP
     * 16 bytes reserved at end of each section for CRC
     */

    /* DSS L2 SRAM: 320KB for code/data, 64KB reserved for TEST_L2_HEAP at end */
    DSS_L2:        origin = 0x00800000, length = (0x50000 - 0x10)

    /* IPC Shared Memory: 1KB */
    IPC_SH_MEM:    origin = 0x88000000, length = (0x00000400 - 0x10)

    /* DSS L3 RAM: ~300KB for general use (sysmem, misc L3 data)
     * Remaining space used for TEST_L3_HEAP
     */
    DSS_L3:        origin = 0x88000400, length = (0x00050000 - 0x10)

    /* Test L3 Heap: 1MB dedicated region for large test allocations */
    TEST_L3_HEAP:  origin = 0x88050400, length = (0x00100000 - 0x10)

    /* Test L2 Heap: 64KB carved from end of L2 for fast test allocations
     * Placed at end of L2 to avoid interfering with SDK near-data
     */
    TEST_L2_HEAP:  origin = 0x00850000, length = (0x00010000 - 0x10)

    /* Shared memories accessible by multiple cores
     * IMPORTANT: On C66x, ensure these regions are mapped as non-cacheable
     * in MAR (Memory Attribute Register) bits
     */
    USER_SHM_MEM:  origin = 0xC02E8000, length = (0x00004000 - 0x10)  /* 16KB */
    LOG_SHM_MEM:   origin = 0xC02EC000, length = (0x00004000 - 0x10)  /* 16KB */
}
