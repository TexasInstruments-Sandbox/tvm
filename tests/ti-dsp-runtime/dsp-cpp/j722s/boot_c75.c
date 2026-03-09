/*
 * C75 DSP Boot Code
 *
 * This is the entry point for the C75 DSP. It sets up the stack,
 * initializes the MMU, and calls main().
 *
 * Based on TI MCU+ SDK boot_c75.c
 */

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

/* External symbols from linker command file */
extern uint32_t __BSS_START;
extern uint32_t __BSS_END;
extern char __TI_STACK_END[];

/* Stack pointer register */
volatile uint64_t __SP;

/* Forward declarations */
extern void MmuP_init(void);  /* From local mmu.c (in L2 via CODE_SECTION) */
extern int main(void);
extern void c7x_startup_init(void);

/* C runtime auto-initialization - processes .cinit section (for --rom_model) */
extern void __TI_auto_init(void);

/*
 * _c_int00_secure - C Environment Entry Point
 *
 * This is the entry point called by the reset vector.
 * It performs the complete boot sequence:
 * 1. Set up stack pointer
 * 2. Initialize MMU (before accessing DDR)
 * 3. Initialize BSS to zero (_system_pre_init)
 * 4. Process .cinit and call _system_post_cinit (__TI_auto_init)
 *    - Initializes global variables from .cinit (--rom_model)
 *    - Calls _system_post_cinit() which initializes cache/interrupts
 * 5. Call main()
 */
#pragma CODE_SECTION(_c_int00_secure, ".text:_c_int00_secure")
void _c_int00_secure(void)
{
    /* Set up stack pointer - align to 8-byte boundary */
    __SP = (((uint64_t)&__TI_STACK_END) - 16) & ~0b111;

    /* Initialize MMU - uses local mmu.c (in L2 via CODE_SECTION) */
    MmuP_init();

    /* Initialize BSS section to zero */
    _system_pre_init();

    /*
     * Process .cinit section - initializes global variables (for --rom_model)
     * NOTE: __TI_auto_init() calls _system_post_cinit() internally, which
     * triggers c7x_startup_init() to initialize cache/interrupts/exceptions.
     */
    __TI_auto_init();

    /* Call main */
    int ret = main();

    /* Properly terminate - flushes I/O buffers and calls atexit handlers */
    exit(ret);
}

/*
 * _system_pre_init - Pre-initialization Hook
 *
 * Called before C/C++ auto-initialization.
 * Initializes BSS section to zero.
 *
 * Returns 1 to allow C/C++ auto-initialization to proceed.
 *
 * NOTE: Placed in L2 to run immediately after MMU init.
 */
#pragma CODE_SECTION(_system_pre_init, ".text:l2_init")
int _system_pre_init(void)
{
    /* Initialize .bss to zero */
    uint32_t bss_size = ((uintptr_t)&__BSS_END - (uintptr_t)&__BSS_START);
    memset((void*)&__BSS_START, 0x00, bss_size);
    return 1;
}

/*
 * _system_post_cinit - Post-initialization Hook
 *
 * Called after C/C++ auto-initialization but before main().
 * Initializes cache, interrupts, and exception handling.
 *
 * NOTE: Placed in L2 to run immediately after MMU init.
 */
#pragma CODE_SECTION(_system_post_cinit, ".text:l2_init")
void _system_post_cinit(void)
{
    c7x_startup_init();
}
