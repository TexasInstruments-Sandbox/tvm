/*
 * C75 DSP Hardware Interrupt Module
 *
 * Provides interrupt initialization and handling.
 * Based on TI MCU+ SDK HwiP_c75.c
 *
 * This is a simplified version for standalone operation.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <c7x.h>

/* Number of interrupt events */
#define HWI_NUM_INTERRUPTS  64

/* ECSP size for interrupt context (8KB per nesting level, 8 levels) */
#define HWI_ECSP_SIZE       0x10000

/* TSR CXM modes */
#define HWI_TSR_CXM_SECURE_SUPERVISOR    5

/* External symbols */
extern char _stack[];
extern char Hwi_vectorsBase[];

/* Assembly helper function declarations (implemented in c75_asm.S) */
extern uint32_t Hwi_getCXM(void);
extern uint32_t Hwi_disable(void);
extern uint32_t Hwi_enable(void);
extern void Hwi_restore(uint32_t key);
extern void Hwi_setCOP(int cop);
extern void Hwi_setESTP(uint64_t val);
extern void Hwi_setECSP(uint64_t val);
extern void Hwi_setTCSP(uint64_t val);

/* Interrupt dispatch table */
typedef void (*Hwi_FuncPtr)(uint32_t arg);

static Hwi_FuncPtr Hwi_dispatchTable[HWI_NUM_INTERRUPTS];
static int32_t Hwi_intEvents[HWI_NUM_INTERRUPTS];

/* Module state */
static char *Hwi_isrStack;
static char *Hwi_taskSP;

/*
 * Get ISR stack address
 */
char* Hwi_getIsrStackAddress(void)
{
    extern uint8_t __TI_STACK_SIZE;
    uint64_t isrStack;

    isrStack = (uint64_t)_stack;
    isrStack += (uint64_t)_symval(&__TI_STACK_SIZE);
    isrStack -= 0x1;
    isrStack &= ~0x7;  /* Align to 8 bytes */

    return (char *)isrStack;
}

/*
 * Hwi_Module_startup - Initialize interrupt module
 *
 * Sets up vector table, event context stack pointers, and
 * initializes the interrupt dispatch table.
 */
void Hwi_Module_startup(void)
{
    int i;

    /* Initialize vector table pointer (ESTP) */
    uint64_t estp = (uint64_t)Hwi_vectorsBase;
    Hwi_setESTP(estp);

    /* Initialize ISR stack pointer */
    Hwi_isrStack = Hwi_getIsrStackAddress() - 16;

    /* Initialize event context stack pointers */
    uint64_t ecsp = (uint64_t)_stack;
    Hwi_setECSP(ecsp);

    uint64_t tcsp = ecsp + HWI_ECSP_SIZE;
    Hwi_setTCSP(tcsp);

    /* Signal we're on ISR stack */
    Hwi_taskSP = (char *)-1;

    /* Initialize dispatch table */
    for (i = 0; i < HWI_NUM_INTERRUPTS; i++) {
        Hwi_dispatchTable[i] = NULL;
        Hwi_intEvents[i] = -1;
    }

    /* Clear all pending events using intrinsic */
    __set_indexed(__EFCLR, 0, 0xFFFFFFFFFFFFFFFFULL);

    /* Set co-processor control */
    Hwi_setCOP(0xff);
}

/*
 * Hwi_dispatchC - C dispatcher for interrupts
 *
 * Called from assembly dispatcher with interrupt number.
 */
void Hwi_dispatchC(int intNum)
{
    Hwi_FuncPtr fxn;

    if (intNum >= 0 && intNum < HWI_NUM_INTERRUPTS) {
        fxn = Hwi_dispatchTable[intNum];
        if (fxn != NULL) {
            fxn((uint32_t)intNum);
        }
    }

    Hwi_setCOP(0xff);
}

/*
 * HwiP_disableInt - Disable specific interrupt
 */
void HwiP_disableInt(uint32_t intNum)
{
    if (intNum < HWI_NUM_INTERRUPTS) {
        uint64_t mask = 1ULL << intNum;
        /* Use EECLR to disable event */
        __set_indexed(__EECLR, 0, mask);
    }
}

/*
 * HwiP_enableInt - Enable specific interrupt
 */
void HwiP_enableInt(uint32_t intNum)
{
    if (intNum < HWI_NUM_INTERRUPTS) {
        uint64_t mask = 1ULL << intNum;
        /* Use EESET to enable event */
        __set_indexed(__EESET, 0, mask);
    }
}
