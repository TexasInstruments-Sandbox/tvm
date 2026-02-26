/*
 * Shared Memory Printf for DSP
 *
 * Redirects printf/stdout to a shared memory buffer that the ARM
 * host reads after inference. Uses TI RTS add_device() to register
 * a custom I/O device, then freopen() to redirect stdout.
 *
 * The printf output accumulates in memory during inference with no
 * IPC overhead. After inference, a single CacheP_wb() makes the
 * data visible to the ARM host via DMA_BUF_IOCTL_SYNC.
 */

#ifndef SHM_PRINTF_H
#define SHM_PRINTF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize shared memory printf device.
 *
 * Registers an "shmout" device via add_device(), then redirects
 * stdout to it with freopen(). Must be called once at startup,
 * after the shared memory region is mapped.
 *
 * @param buf_addr  Start of the printf buffer in shared memory.
 * @param buf_size  Total size of the printf buffer (e.g. 64 KB).
 */
void shm_printf_init(void *buf_addr, uint32_t buf_size);

/**
 * Reset the printf buffer write index to zero.
 * Call before each inference to start with a clean buffer.
 */
void shm_printf_reset(void);

/**
 * Flush the printf buffer to DDR via cache writeback.
 *
 * @return Number of bytes written since last reset.
 */
uint32_t shm_printf_finish(void);

/**
 * Printf directly into the shared memory buffer.
 *
 * This is the fast-path target for the DLOAD printf symbol alias.
 * Bypasses FILE* machinery for lower overhead.
 *
 * @return Number of characters written (excluding NUL), or -1 on error.
 */
int shm_printf(const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* SHM_PRINTF_H */
