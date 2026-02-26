/*
 * C7x Compute Service - DSP Main Entry Point
 *
 * FreeRTOS-based firmware that provides compute services to Linux host.
 */

#include <stdio.h>
#include <string.h>
#include <kernel/dpl/DebugP.h>
#include <kernel/dpl/ClockP.h>
#include <kernel/dpl/HwiP.h>
#include <kernel/dpl/MmuP_armv8.h>
#include <drivers/ipc_notify.h>
#include <drivers/ipc_rpmsg.h>
#include <FreeRTOS.h>
#include <task.h>

#include "ti_drivers_config.h"
#include "ti_drivers_open_close.h"
#include "ti_board_open_close.h"

#include "compute_service.h"
#include "c7x_compute_protocol.h"
#include "dma/tvm_dsp_dma.h"

/*
 * =============================================================================
 * IPC Shutdown Handling
 * =============================================================================
 */

/* Stored by ISR callback, used by main task to send ACK after cleanup */
static volatile uint16_t gShutdownRemoteCoreId = 0;

/**
 * IPC mailbox callback for shutdown notification from Linux.
 */
static void ipc_rp_mbox_callback(uint16_t remoteCoreId, uint16_t localClientId,
                                  uint32_t msgValue, void *args)
{
    if (msgValue == IPC_NOTIFY_RP_MBOX_SHUTDOWN) {
        DebugP_log("[IPC] Received SHUTDOWN request from Linux\r\n");

        /* Store remote core ID for sending ACK later (from task context) */
        gShutdownRemoteCoreId = remoteCoreId;

        /* Signal service loop to stop (unblocks RPMessage_recv) */
        compute_service_stop();

        /* NOTE: ACK is sent after service loop exits and cleanup completes,
         * in c7x_compute_main(). Sending it here (ISR context) would cause
         * the kernel to proceed with reset before the DSP is ready. */
    }
}

/*
 * =============================================================================
 * Main Application
 * =============================================================================
 */

void c7x_compute_main(void *args)
{
    int32_t status;
    uint32_t mmuEnabled;

    DebugP_log("\r\n");
    DebugP_log("===========================================\r\n");
    DebugP_log("  C7x Compute Service v%d.%d.%d\r\n",
               C7X_VERSION_MAJOR(C7X_SERVICE_VERSION),
               C7X_VERSION_MINOR(C7X_SERVICE_VERSION),
               C7X_VERSION_PATCH(C7X_SERVICE_VERSION));
    DebugP_log("===========================================\r\n");
    DebugP_log("\r\n");

    /* Verify MMU is enabled */
    mmuEnabled = MmuP_isEnabled();
    DebugP_log("[INIT] MMU enabled = %u\r\n", mmuEnabled);
    if (mmuEnabled) {
        DebugP_log("[INIT] Cache coherent access to shared memory enabled\r\n");
    } else {
        DebugP_log("[INIT] WARNING: MMU disabled, cache coherency may not work\r\n");
    }

    /* Log shared buffer configuration */
    DebugP_log("[INIT] Shared buffer: 0x%08llx - 0x%08llx (%u MB)\r\n",
               C7X_SHARED_BASE, C7X_SHARED_BASE + C7X_SHARED_SIZE,
               (uint32_t)(C7X_SHARED_SIZE / (1024 * 1024)));
    DebugP_log("[INIT] Input buffer:  0x%08llx (%u MB)\r\n",
               C7X_INPUT_BUFFER_ADDR, (uint32_t)(C7X_INPUT_BUFFER_SIZE / (1024 * 1024)));
    DebugP_log("[INIT] Output buffer: 0x%08llx (%u MB)\r\n",
               C7X_OUTPUT_BUFFER_ADDR, (uint32_t)(C7X_OUTPUT_BUFFER_SIZE / (1024 * 1024)));

    /* Wait for Linux to initialize virtio vrings (polls resource table status) */
    DebugP_log("[IPC] Waiting for Linux to be ready...\r\n");
    status = RPMessage_waitForLinuxReady(SystemP_WAIT_FOREVER);
    if (status == SystemP_SUCCESS) {
        DebugP_log("[IPC] Linux is ready\r\n");
    } else {
        DebugP_log("[IPC] Linux ready wait failed: %d\r\n", status);
    }

    /* Register shutdown callback */
    status = IpcNotify_registerClient(IPC_NOTIFY_CLIENT_ID_RP_MBOX,
                                       ipc_rp_mbox_callback, NULL);
    if (status != SystemP_SUCCESS) {
        DebugP_log("[IPC] Failed to register RP_MBOX callback: %d\r\n", status);
    }
    DebugP_log("[IPC] Registered shutdown callback\r\n");

    /* Initialize TVM DSP platform (memory pools, cycle counter) */
    {
        extern int tvm_dsp_platform_init(void);
        int tvm_status = tvm_dsp_platform_init();
        if (tvm_status == 0) {
            DebugP_log("[INIT] TVM DSP platform initialized\r\n");
        } else {
            DebugP_log("[INIT] WARNING: TVM DSP platform init failed: %d\r\n", tvm_status);
        }
    }

    /* DMA subsystem (EDMA via DRU direct TR mode) and CLEC event routing
     * are initialized per-module in compute_service.c (on load) and
     * torn down on unload.  This ensures UDMA/DRU resources are released
     * before the host's DMA-BUF cleanup runs between invocations. */

    /* Initialize compute service */
    status = compute_service_init();
    if (status != SystemP_SUCCESS) {
        DebugP_log("[INIT] Failed to initialize compute service: %d\r\n", status);
        goto cleanup;
    }

    DebugP_log("[INIT] Compute service ready, entering message loop...\r\n");
    DebugP_log("\r\n");

    /* Run service message loop (blocks until shutdown) */
    compute_service_run();

cleanup:
    DebugP_log("[SHUTDOWN] Cleaning up...\r\n");

    /* Deinitialize DMA subsystem */
    tvm_dsp_dma_deinit();

    /* Send shutdown ACK to Linux BEFORE tearing down IPC resources.
     * RPMessage_destruct in compute_service_deinit() would disrupt the
     * virtio transport needed for the mailbox ACK to reach the kernel. */
    if (gShutdownRemoteCoreId != 0) {
        DebugP_log("[SHUTDOWN] Sending SHUTDOWN_ACK to core %u\r\n",
                   gShutdownRemoteCoreId);
        IpcNotify_sendMsg(gShutdownRemoteCoreId,
                          IPC_NOTIFY_CLIENT_ID_RP_MBOX,
                          IPC_NOTIFY_RP_MBOX_SHUTDOWN_ACK, 1);
    }

    /* Deinitialize compute service (destroys RPMessage endpoint) */
    compute_service_deinit();

    /* Close drivers and board */
    Board_driversClose();
    Drivers_close();

    DebugP_log("[SHUTDOWN] Entering IDLE state for remoteproc stop\r\n");

    /* Disable interrupts and halt */
    HwiP_disable();
    __asm(" IDLE");
}

/*
 * =============================================================================
 * FreeRTOS Entry Point
 * =============================================================================
 */

#define MAIN_TASK_STACK_SIZE    (16 * 1024)  /* in StackType_t units (8 bytes each) = 128KB */
#define MAIN_TASK_PRIORITY      (configMAX_PRIORITIES - 1)

static StaticTask_t gMainTaskObj;
static StackType_t  gMainTaskStack[MAIN_TASK_STACK_SIZE] __attribute__((aligned(0x2000)));

int main(void)
{
    /* Initialize system (DPL, drivers, IPC) */
    System_init();
    Board_init();

    /* Create main task */
    xTaskCreateStatic(c7x_compute_main,
                      "c7x_compute_main",
                      MAIN_TASK_STACK_SIZE,
                      NULL,
                      MAIN_TASK_PRIORITY,
                      gMainTaskStack,
                      &gMainTaskObj);

    /* Start FreeRTOS scheduler */
    vTaskStartScheduler();

    /* Should never reach here */
    return 0;
}
