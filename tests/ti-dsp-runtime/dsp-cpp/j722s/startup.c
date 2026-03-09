/*
 * C75 DSP Startup Code
 *
 * Initializes cache, interrupts, and exception handling.
 * Based on TI MCU+ SDK Startup.c
 */

#include <stdint.h>
#include <stdbool.h>
#include <c7x.h>

/* Cache type definitions */
#define CACHEP_TYPE_L1D    (1U)

/* External declarations */
extern void CacheP_enable(uint32_t type);
extern void CacheP_enableWB(uint32_t type);
extern void CacheP_wbInvAll(uint32_t type);
extern void Hwi_Module_startup(void);
extern void Exception_Module_startup(void);
extern char _stack[];

/*
 * Hwi_initStackMin - Initialize interrupt stack marker
 *
 * Writes a marker byte at the base of the interrupt stack
 * for stack overflow detection.
 */
static void Hwi_initStackMin(void)
{
    /* Write marker at stack base for overflow detection */
    volatile uint8_t *stackBase = (volatile uint8_t *)_stack;
    *stackBase = 0xbe;
}

/*
 * c7x_startup_init - C7x Startup Initialization
 *
 * Called from _system_post_cinit() after C initialization.
 * Sets up hardware for normal operation.
 */
void c7x_startup_init(void)
{
    /*
     * Invalidate L1D cache before enabling.
     * When JTAG loads a new program, it writes directly to DDR bypassing cache.
     * Any stale cache lines from the previous program must be invalidated to
     * ensure we read fresh data from DDR.
     */
    CacheP_wbInvAll(CACHEP_TYPE_L1D);

    /* Initialize stack marker */
    Hwi_initStackMin();

    /* Enable L1D cache with writeback */
    CacheP_enable(CACHEP_TYPE_L1D);
    CacheP_enableWB(CACHEP_TYPE_L1D);

    /* Initialize interrupt module */
    Hwi_Module_startup();

    /* Initialize exception handling */
    Exception_Module_startup();
}
