/*
 * C75 DSP MMU Module
 *
 * Provides MMU initialization and memory mapping.
 * Based on TI MCU+ SDK MmuP_c75.c and edgeai-tidl-kernels enable_cache_mmu.c
 *
 * This version uses 2-level page tables with 2MB blocks for finer control
 * over memory attributes, particularly shareability for multi-core coherency.
 *
 * Memory Map for J722S C75:
 *   Peripherals: 0x00000000 - 0x3FFFFFFF (Device memory)
 *   MSMC:        0x70000000 - 0x703FFFFF (4MB total, Outer Shareable)
 *                - 0x70000000-0x7001FFFF: Reserved for Linux/ATF (128KB)
 *                - 0x70020000-0x703E7FFF: C7x_1 usable region (3.78MB)
 *                - 0x703F0000-0x703FFFFF: Reserved for DMSC IPC (64KB)
 *   L2SRAM:      0x7E000000 - 0x7E1FFFFF (2MB, Non-Shareable)
 *   L2AUX:       0x7F000000 - 0x7F03FFFF (256KB, Non-Shareable)
 *   DDR:         0x80000000 - 0xFFFFFFFF (Outer Shareable)
 *   DDR (>4GB):  0x100000000+ (Outer Shareable, for extended heaps)
 */

#include <stdint.h>
#include <stdbool.h>
#include <c7x.h>

/*
 * MMU Register Values (from edgeai-tidl-kernels reference)
 */
#define MMU_TCR0_VALUE      0x0000000000002a21ULL
#define MMU_SCR_VALUE       0x80000000000000c1ULL
#define MMU_MAR_VALUE       0x3D3D3D2915032A00ULL

/*
 * MAIR Index definitions (from packed MAR value)
 *   MAIR0 = 0x00 - Device-nGnRnE
 *   MAIR1 = 0x2A - Write-Through No-Allocate
 *   MAIR2 = 0x03 - Device-nGnRE
 *   MAIR3 = 0x15 - Write-Through Allocate
 *   MAIR4 = 0x29 - Non-cacheable
 *   MAIR5 = 0x3D - Write-Back Read-Allocate Write-Allocate (normal cached)
 *   MAIR6 = 0x3D - Write-Back Read-Allocate Write-Allocate
 *   MAIR7 = 0x3D - Write-Back Read-Allocate Write-Allocate
 */
#define MAIR_DEVICE         0   /* Device-nGnRnE */
#define MAIR_NC             4   /* Non-cacheable */
#define MAIR_CACHED         5   /* Write-Back RA/WA */

/* MMU table configuration */
#define MMU_L1_TABLE_LEN    512     /* Level 1: 512 entries, each covers 1GB */
#define MMU_L2_TABLE_LEN    512     /* Level 2: 512 entries, each covers 2MB */

/* Descriptor types */
#define MMU_DESC_INVALID    0ULL
#define MMU_DESC_BLOCK      1ULL    /* Block descriptor (L1=1GB, L2=2MB) */
#define MMU_DESC_TABLE      3ULL    /* Table descriptor (points to next level) */

/* Block descriptor attribute bits */
#define MMU_ATTR_AF         (1ULL << 10)    /* Access Flag */
#define MMU_ATTR_SH_NSH     (0ULL << 8)     /* Non-Shareable */
#define MMU_ATTR_SH_OSH     (2ULL << 8)     /* Outer Shareable */
#define MMU_ATTR_SH_ISH     (3ULL << 8)     /* Inner Shareable */
#define MMU_ATTR_AP_RW      (0ULL << 6)     /* Read-Write access */
#define MMU_ATTR_NS         (1ULL << 5)     /* Non-Secure */

/* Address masks */
#define MMU_L1_BLOCK_MASK   0x0000FFFFC0000000ULL   /* 1GB aligned */
#define MMU_L2_BLOCK_MASK   0x0000FFFFFFE00000ULL   /* 2MB aligned */

/* MMU table arrays - 4KB aligned */
#pragma DATA_SECTION(gMmu_level1Table, ".data.Mmu_level1Table")
#pragma DATA_ALIGN(gMmu_level1Table, 4096)
static uint64_t gMmu_level1Table[MMU_L1_TABLE_LEN];

/* Level 2 table for 0x40000000-0x7FFFFFFF (covers MSMC and L2SRAM) */
#pragma DATA_SECTION(gMmu_level2Table_40, ".data.Mmu_level2Table")
#pragma DATA_ALIGN(gMmu_level2Table_40, 4096)
static uint64_t gMmu_level2Table_40[MMU_L2_TABLE_LEN];

/* Assembly helper function declarations (implemented in c75_asm.S) */
extern void MmuP_setMAR(uint64_t mar);
extern void MmuP_setTCR(uint64_t tcr, bool s);
extern void Mmu_init(uint64_t *table, bool s);
extern void MmuP_setSCR(uint64_t scr);
extern void MmuP_tlbInvAll(void);

/*
 * Create a Level 1 block descriptor (1GB mapping)
 */
static uint64_t MmuP_makeL1BlockDesc(uint64_t paddr, uint8_t attrIdx, uint64_t shareability)
{
    uint64_t desc;
    desc = MMU_DESC_BLOCK;
    desc |= ((uint64_t)(attrIdx & 0x7) << 2);
    desc |= MMU_ATTR_NS;
    desc |= MMU_ATTR_AP_RW;
    desc |= shareability;
    desc |= MMU_ATTR_AF;
    desc |= (paddr & MMU_L1_BLOCK_MASK);
    return desc;
}

/*
 * Create a Level 2 block descriptor (2MB mapping)
 */
static uint64_t MmuP_makeL2BlockDesc(uint64_t paddr, uint8_t attrIdx, uint64_t shareability)
{
    uint64_t desc;
    desc = MMU_DESC_BLOCK;
    desc |= ((uint64_t)(attrIdx & 0x7) << 2);
    desc |= MMU_ATTR_NS;
    desc |= MMU_ATTR_AP_RW;
    desc |= shareability;
    desc |= MMU_ATTR_AF;
    desc |= (paddr & MMU_L2_BLOCK_MASK);
    return desc;
}

/*
 * Create a table descriptor (points to next level)
 */
static uint64_t MmuP_makeTableDesc(uint64_t *tableAddr)
{
    return ((uint64_t)tableAddr & ~0xFFFULL) | MMU_DESC_TABLE;
}

/*
 * Set up Level 2 table for 0x40000000-0x7FFFFFFF region
 *
 * This region contains:
 *   0x48000000 - 0x4DFFFFFF: Peripheral region (includes Secure Proxy) - Device
 *   0x70000000 - 0x703FFFFF: MSMC (4MB) - Outer Shareable for multi-core
 *   0x7E000000 - 0x7E1FFFFF: L2SRAM MAIN (2MB) - Non-Shareable (local)
 *   0x7F000000 - 0x7F03FFFF: L2SRAM AUX + L1 (256KB) - Non-Shareable (local)
 *   Everything else: Invalid
 *
 * Secure Proxy addresses (required for Sciclient/DMSC communication):
 *   0x48250000 - SEC_PROXY MMRS (config registers)
 *   0x4a400000 - SEC_PROXY SCFG (secure config)
 *   0x4a600000 - SEC_PROXY RT (runtime)
 *   0x4d000000 - SEC_PROXY TARGET_DATA (message data)
 *
 * NOTE: Must be in L2 (called from MmuP_setConfig before DDR accessible).
 */
#pragma CODE_SECTION(MmuP_setupLevel2Table_40, ".text:l2_init")
static void MmuP_setupLevel2Table_40(void)
{
    uint32_t i;
    uint64_t addr;

    /* Initialize all entries as invalid */
    for (i = 0; i < MMU_L2_TABLE_LEN; i++) {
        gMmu_level2Table_40[i] = MMU_DESC_INVALID;
    }

    /*
     * Secure Proxy region: 0x48000000 - 0x4FFFFFFF
     * Required for Sciclient to communicate with DMSC.
     *
     * SEC_PROXY MMRS:        0x48250000 (in 2MB block starting 0x48200000)
     * SEC_PROXY SCFG:        0x4a400000 (2MB block)
     * SEC_PROXY RT:          0x4a600000 (2MB block)
     * SEC_PROXY TARGET_DATA: 0x4d000000 (2MB block)
     *
     * Map as Device memory for peripheral access.
     * Index calculation: (addr - 0x40000000) / 2MB
     */

    /* 0x48000000 - 0x49FFFFFF: 16MB for peripheral region including SEC_PROXY_MMRS */
    addr = 0x48000000ULL;
    for (i = 0; i < 8; i++) {  /* 8 x 2MB = 16MB */
        uint32_t idx = (uint32_t)((addr - 0x40000000ULL) >> 21);
        gMmu_level2Table_40[idx] = MmuP_makeL2BlockDesc(addr, MAIR_DEVICE, MMU_ATTR_SH_NSH);
        addr += 0x200000;  /* 2MB */
    }

    /* 0x4a000000 - 0x4BFFFFFF: 32MB for SEC_PROXY SCFG and RT */
    addr = 0x4a000000ULL;
    for (i = 0; i < 16; i++) {  /* 16 x 2MB = 32MB */
        uint32_t idx = (uint32_t)((addr - 0x40000000ULL) >> 21);
        gMmu_level2Table_40[idx] = MmuP_makeL2BlockDesc(addr, MAIR_DEVICE, MMU_ATTR_SH_NSH);
        addr += 0x200000;  /* 2MB */
    }

    /* 0x4c000000 - 0x4FFFFFFF: 64MB for SEC_PROXY TARGET_DATA */
    addr = 0x4c000000ULL;
    for (i = 0; i < 32; i++) {  /* 32 x 2MB = 64MB */
        uint32_t idx = (uint32_t)((addr - 0x40000000ULL) >> 21);
        gMmu_level2Table_40[idx] = MmuP_makeL2BlockDesc(addr, MAIR_DEVICE, MMU_ATTR_SH_NSH);
        addr += 0x200000;  /* 2MB */
    }

    /*
     * MSMC: 0x70000000 - 0x703FFFFF (4MB = 2 x 2MB blocks)
     * Index calculation: (0x70000000 - 0x40000000) / 2MB = 0x30000000 / 0x200000 = 384
     *
     * IMPORTANT: MSMC is partitioned by the system:
     *   0x70000000 - 0x7001FFFF: Reserved for Linux/ATF (128KB)
     *   0x70020000 - 0x703E7FFF: C7x_1 allocation (3.78MB)
     *   0x703F0000 - 0x703FFFFF: Reserved for DMSC IPC (64KB)
     *
     * The linker command file restricts usage to the C7x region.
     * MMU L2 blocks are 2MB minimum, so we map the full 4MB.
     * Hardware firewalls (TIFS) may still block access to reserved regions.
     *
     * MSMC is shared memory - use Outer Shareable for cache coherency.
     */
    addr = 0x70000000ULL;
    for (i = 0; i < 2; i++) {  /* 2 x 2MB = 4MB */
        uint32_t idx = (uint32_t)((addr - 0x40000000ULL) >> 21);  /* 2MB = 2^21 */
        gMmu_level2Table_40[idx] = MmuP_makeL2BlockDesc(addr, MAIR_CACHED, MMU_ATTR_SH_OSH);
        addr += 0x200000;  /* 2MB */
    }

    /*
     * L2SRAM MAIN: 0x7E000000 - 0x7E1FFFFF (2MB = 1 x 2MB block)
     * Index calculation: (0x7E000000 - 0x40000000) / 2MB = 0x3E000000 / 0x200000 = 496
     *
     * L2SRAM is local to each C75 core - use Non-Shareable for better performance
     */
    addr = 0x7E000000ULL;
    {
        uint32_t idx = (uint32_t)((addr - 0x40000000ULL) >> 21);
        gMmu_level2Table_40[idx] = MmuP_makeL2BlockDesc(addr, MAIR_CACHED, MMU_ATTR_SH_NSH);
    }

    /*
     * L2SRAM AUX + L1: 0x7F000000 - 0x7F03FFFF (256KB, within 2MB block)
     * Index calculation: (0x7F000000 - 0x40000000) / 2MB = 0x3F000000 / 0x200000 = 504
     *
     * Also local memory - Non-Shareable
     */
    addr = 0x7F000000ULL;
    {
        uint32_t idx = (uint32_t)((addr - 0x40000000ULL) >> 21);
        gMmu_level2Table_40[idx] = MmuP_makeL2BlockDesc(addr, MAIR_CACHED, MMU_ATTR_SH_NSH);
    }
}

/*
 * Set up default MMU configuration for J722S C75
 *
 * Memory regions with appropriate shareability:
 *   0x00000000 - 0x3FFFFFFF: Device memory (peripherals) - Non-Shareable
 *   0x40000000 - 0x7FFFFFFF: Uses Level 2 table for fine control
 *     - MSMC (0x70000000): Outer Shareable
 *     - L2SRAM (0x7E000000): Non-Shareable
 *     - L2AUX (0x7F000000): Non-Shareable
 *   0x80000000 - 0xFFFFFFFF: DDR - Outer Shareable
 *   0x100000000 - 0x11FFFFFFF: Extended DDR - Outer Shareable (256MB x 4 = 1GB)
 *
 * NOTE: Must be in L2 (called from MmuP_init before DDR accessible).
 */
#pragma CODE_SECTION(MmuP_setConfig, ".text:l2_init")
static void MmuP_setConfig(void)
{
    uint32_t i;

    /* Clear all Level 1 entries */
    for (i = 0; i < MMU_L1_TABLE_LEN; i++) {
        gMmu_level1Table[i] = MMU_DESC_INVALID;
    }

    /* 0x00000000 - 0x3FFFFFFF: Device memory (peripherals) */
    gMmu_level1Table[0] = MmuP_makeL1BlockDesc(0x00000000ULL, MAIR_DEVICE, MMU_ATTR_SH_NSH);

    /* 0x40000000 - 0x7FFFFFFF: Table descriptor -> Level 2 for fine control */
    MmuP_setupLevel2Table_40();
    gMmu_level1Table[1] = MmuP_makeTableDesc(gMmu_level2Table_40);

    /* 0x80000000 - 0xBFFFFFFF: DDR (Outer Shareable for multi-core access) */
    gMmu_level1Table[2] = MmuP_makeL1BlockDesc(0x80000000ULL, MAIR_CACHED, MMU_ATTR_SH_OSH);

    /* 0xC0000000 - 0xFFFFFFFF: More DDR (Outer Shareable) */
    gMmu_level1Table[3] = MmuP_makeL1BlockDesc(0xC0000000ULL, MAIR_CACHED, MMU_ATTR_SH_OSH);

    /*
     * Extended DDR above 4GB (for local heaps and scratch memory)
     * 0x100000000 - 0x13FFFFFFF: 1GB DDR (Outer Shareable)
     * 0x140000000 - 0x17FFFFFFF: 1GB DDR (Outer Shareable)
     * etc.
     */
    gMmu_level1Table[4] = MmuP_makeL1BlockDesc(0x100000000ULL, MAIR_CACHED, MMU_ATTR_SH_OSH);
    gMmu_level1Table[5] = MmuP_makeL1BlockDesc(0x140000000ULL, MAIR_CACHED, MMU_ATTR_SH_OSH);
}

/*
 * MmuP_init - Initialize the MMU
 *
 * Sets up 2-level page tables with proper shareability for multi-core operation.
 *
 * NOTE: Must be in L2 to run before DDR is accessible via MMU.
 */
#pragma CODE_SECTION(MmuP_init, ".text:l2_init")
void MmuP_init(void)
{
    /* Set up memory mappings */
    MmuP_setConfig();

    /*
     * Configure MMU registers in order:
     * 1. TCR0 - Table Control Register
     * 2. TBR0 - Table Base Register (page table pointer)
     * 3. MAR  - Memory Attribute Register (packed MAIR0-7)
     * 4. SCR  - System Control Register (enables MMU - MUST BE LAST)
     */
    MmuP_setTCR(MMU_TCR0_VALUE, false);
    Mmu_init(gMmu_level1Table, false);
    MmuP_setMAR(MMU_MAR_VALUE);
    MmuP_setSCR(MMU_SCR_VALUE);
}
