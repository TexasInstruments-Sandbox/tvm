/*
 * C7x Compute Service - DSP Service Header
 *
 * RPMessage-based compute service for host applications.
 */

#ifndef COMPUTE_SERVICE_H
#define COMPUTE_SERVICE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize the compute service.
 *
 * Creates the RPMessage endpoint. Must be called after System_init()
 * and IPC initialization. Call compute_service_run() after this to
 * start processing messages.
 *
 * @return 0 on success, negative error code on failure
 */
int32_t compute_service_init(void);

/**
 * Run the compute service message loop.
 *
 * Blocks and processes RPMessage requests until the service is stopped
 * (e.g. via shutdown callback). Runs in the caller's task context.
 */
void compute_service_run(void);

/**
 * Signal the service loop to stop.
 *
 * Safe to call from ISR context (e.g. shutdown callback).
 * The loop will exit after the current RPMessage_recv timeout.
 */
void compute_service_stop(void);

/**
 * Deinitialize the compute service.
 *
 * Releases resources. The service loop must have exited first.
 */
void compute_service_deinit(void);

/**
 * Get service statistics.
 *
 * @param jobs_completed Output: number of successfully completed jobs
 * @param jobs_failed    Output: number of failed jobs
 * @param uptime_ms      Output: service uptime in milliseconds
 */
void compute_service_get_stats(uint32_t *jobs_completed,
                                uint32_t *jobs_failed,
                                uint32_t *uptime_ms);

#ifdef __cplusplus
}
#endif

#endif /* COMPUTE_SERVICE_H */
