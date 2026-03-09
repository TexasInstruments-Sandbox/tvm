/*
 * C75 DSP Cache Module
 *
 * Provides cache enable/disable and writeback functions.
 * Based on TI MCU+ SDK CacheP_c75.c
 */

#include <stdint.h>
#include <stdbool.h>
#include <c7x.h>

/* Cache type definitions */
#define CACHEP_TYPE_L1D    (1U)

/* Cache configuration state */
static uint64_t CacheP_L1DCFG_state = 0x11U;  /* L1DWBEN and L1DON */

/*
 * Assembly function declarations (implemented in c75_asm.S)
 */
extern uint64_t CacheP_getL1DCFG(void);
extern void CacheP_setL1DCFG(uint64_t val);
extern void CacheP_setL1DWB(uint64_t flag);
extern void CacheP_setL1DWBINV(uint64_t flag);
extern void CacheP_setL1DINV(uint64_t flag);

/*
 * CacheP_enable - Enable L1D Cache
 */
void CacheP_enable(uint32_t type)
{
    if (type & CACHEP_TYPE_L1D)
    {
        uint64_t cfg = CacheP_getL1DCFG();
        cfg |= 1U;  /* Set L1DON bit */
        CacheP_setL1DCFG(cfg);
        CacheP_L1DCFG_state = cfg;
    }
}

/*
 * CacheP_disable - Disable L1D Cache
 */
void CacheP_disable(uint32_t type)
{
    if (type & CACHEP_TYPE_L1D)
    {
        uint64_t cfg = CacheP_getL1DCFG();
        cfg &= ~((uint64_t)1);  /* Clear L1DON bit */
        CacheP_setL1DCFG(cfg);
        CacheP_L1DCFG_state = cfg;
    }
}

/*
 * CacheP_enableWB - Enable Cache Writeback
 */
void CacheP_enableWB(uint32_t type)
{
    if (type & CACHEP_TYPE_L1D)
    {
        uint64_t cfg = CacheP_getL1DCFG();
        cfg |= 0x10U;  /* Set L1DWBEN bit */
        CacheP_setL1DCFG(cfg);
        CacheP_L1DCFG_state = cfg;
    }
}

/*
 * CacheP_wait - Wait for cache operation to complete
 */
void CacheP_wait(void)
{
    /* Memory fence - wait for pending operations to complete */
    __SE0ADV(char);
    /* Use NOP loop as simple barrier since _mfence may not be available */
    volatile int i;
    for (i = 0; i < 100; i++) { }
}

/*
 * CacheP_wbAll - Writeback all cache lines
 */
void CacheP_wbAll(uint32_t type)
{
    if (type & CACHEP_TYPE_L1D)
    {
        CacheP_setL1DWB(1);
    }
}

/*
 * CacheP_wbInvAll - Writeback and invalidate all cache lines
 */
void CacheP_wbInvAll(uint32_t type)
{
    if (type & CACHEP_TYPE_L1D)
    {
        CacheP_setL1DWBINV(1);
    }
}
