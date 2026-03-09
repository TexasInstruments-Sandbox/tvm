/*
 * C75 DSP Exception Module
 *
 * Provides exception initialization and handling.
 * Based on TI MCU+ SDK Exception.c
 *
 * This version captures diagnostic info to L2 memory before halting,
 * which helps debug page faults and other exceptions.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <c7x.h>

/* Exception context buffer size */
#define EXCEPTION_CONTEXT_BUF_SIZE  0x1000

/*
 * Exception diagnostic structure - stored in L2 for debugger visibility.
 * Place in a dedicated section to ensure it's in L2 SRAM.
 */
#pragma DATA_SECTION(g_exception_info, ".data:exception_info")
typedef struct {
    uint32_t magic;          /* 0xDEADFAUL when exception occurred */
    uint32_t exception_type; /* 1=internal, 2=page_fault, 3=nme */
    uint64_t ierr;           /* Internal Exception Report Register */
    uint64_t nrp;            /* Next Return Pointer (faulting PC) */
    uint64_t ntsr;           /* Next Task State Register */
    uint64_t ecr784_scr;     /* System Control Register (MMU status) */
    uint64_t tsc;            /* Timestamp when exception occurred */
    uint32_t handler_called; /* Incremented each time handler runs */
    uint32_t reserved;
} ExceptionInfo;

static ExceptionInfo g_exception_info = {0};

/* Module state */
static char *Exception_excPtr;

/* External declarations */
extern char* Hwi_getIsrStackAddress(void);

/* Assembly helper function declarations (implemented in c75_asm.asm) */
extern uint64_t Exception_getIERR(void);
extern void Exception_clearIERR(void);
extern uint64_t Exception_getTSC(void);   /* Timestamp Counter */
extern uint64_t Exception_getSCR(void);   /* System Control Register (MMU status) */

/*
 * Exception_Module_startup - Initialize exception handling
 *
 * Sets up exception stack pointer.
 */
void Exception_Module_startup(void)
{
    /* Use ISR stack for exception handling */
    Exception_excPtr = Hwi_getIsrStackAddress();

    /* Clear exception info */
    g_exception_info.magic = 0;
    g_exception_info.handler_called = 0;
}

/*
 * Exception_pageFaultHandler - Page fault exception handler
 *
 * Called from vectors.asm when a page fault occurs.
 * Captures diagnostic info to L2 and halts.
 *
 * NOTE: This function MUST be in L2 to run during a page fault!
 * If this function is in DDR and DDR caused the fault, we'll double fault.
 */
#pragma CODE_SECTION(Exception_pageFaultHandler, ".text:l2_init")
void Exception_pageFaultHandler(void)
{
    /* Capture diagnostic info using assembly helpers (all in L2) */
    g_exception_info.magic = 0xDEADFA01;  /* Page fault marker */
    g_exception_info.exception_type = 2;
    g_exception_info.ierr = Exception_getIERR();
    g_exception_info.nrp = 0;  /* NRP requires special handling - TODO */
    g_exception_info.ntsr = 0; /* NTSR requires special handling - TODO */
    g_exception_info.tsc = Exception_getTSC();
    g_exception_info.ecr784_scr = Exception_getSCR();
    g_exception_info.handler_called++;

    /* Clear IERR to prevent cascading */
    Exception_clearIERR();

    /*
     * Try to print diagnostic info.
     * This may fail if CIO/printf relies on DDR, but worth trying.
     * The info is also saved in g_exception_info for debugger inspection.
     */
    printf("\n!!! PAGE FAULT !!!\n");
    printf("  IERR:           0x%016llx\n", (unsigned long long)g_exception_info.ierr);
    printf("  SCR (MMU):      0x%016llx\n", (unsigned long long)g_exception_info.ecr784_scr);
    printf("  Handler count:  %u\n", g_exception_info.handler_called);
    printf("Halting...\n");

    /* Halt - use idle instruction which is debugger-friendly */
    for (;;) {
        /* idle to allow debugger to connect */
    }
}

/*
 * Exception_internalHandler - Internal exception handler
 *
 * Handles internal exceptions (illegal instruction, etc.)
 */
#pragma CODE_SECTION(Exception_internalHandler, ".text:l2_init")
void Exception_internalHandler(void)
{
    g_exception_info.magic = 0xDEADFA02;  /* Internal exception marker */
    g_exception_info.exception_type = 1;
    g_exception_info.ierr = Exception_getIERR();
    g_exception_info.nrp = 0;
    g_exception_info.ntsr = 0;
    g_exception_info.tsc = Exception_getTSC();
    g_exception_info.handler_called++;

    Exception_clearIERR();

    printf("\n!!! INTERNAL EXCEPTION !!!\n");
    printf("  IERR: 0x%016llx\n", (unsigned long long)g_exception_info.ierr);
    printf("Halting...\n");

    for (;;) {
        /* idle to allow debugger to connect */
    }
}

/*
 * Exception_handler - Generic exception handler (legacy API)
 */
void Exception_handler(bool abortFlag, int vectorType)
{
    g_exception_info.magic = 0xDEADFA00;
    g_exception_info.exception_type = (uint32_t)vectorType;
    g_exception_info.ierr = Exception_getIERR();
    g_exception_info.handler_called++;

    /* Spin forever - debugger can inspect state */
    volatile int spin = 1;
    while (spin) {
        /* Set breakpoint here to catch exceptions */
    }
}
