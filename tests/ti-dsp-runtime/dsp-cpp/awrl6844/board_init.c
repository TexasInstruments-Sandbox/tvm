/*
 * board_init.c
 *
 * AWRL6844 board initialization wrapper for layer tests
 */

#include "board_init.h"
#include "syscfg/ti_drivers_config.h"
#include "syscfg/ti_board_config.h"
#include "syscfg/ti_dpl_config.h"
#include "syscfg/ti_drivers_open_close.h"
#include "syscfg/ti_board_open_close.h"

#include <kernel/dpl/CacheP.h>
#include <stdio.h>

int awrl6844_board_init(void)
{
    /* Initialize system-level resources
     * - Device Power/Clock Configuration
     * - Pinmux Configuration
     * - DPL (Driver Porting Layer) initialization
     * - Driver initialization (UART, EDMA, etc.)
     */
    System_init();

    /* Initialize board-specific resources */
    Board_init();

    /* Open configured drivers (UART, EDMA)
     * SDK functions return void - assume success
     */
    Drivers_open();

    /* Open board-specific drivers */
    Board_driversOpen();

    /* Configure DSS_L3 (0x88000000) as non-cacheable.
     * This is required for:
     * 1. HeapP - writes free-list metadata directly to backing memory
     * 2. DMA buffers - DMA bypasses cache, so caching causes coherency issues
     * MAR value 0 = not cacheable (PC=0), not prefetchable (PFX=0) */
    CacheP_setMar((void*)0x88000000, 0x01000000, 0);

    printf("AWRL6844 board initialization complete\n");
    return 0;
}

void awrl6844_board_deinit(void)
{
    /* Close drivers in reverse order of initialization */
    Board_driversClose();
    Drivers_close();
    Board_deinit();
    System_deinit();

    printf("AWRL6844 board deinitialization complete\n");
}
