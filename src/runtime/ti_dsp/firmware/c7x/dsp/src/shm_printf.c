/*
 * Shared Memory Printf for DSP
 *
 * Implements a custom I/O device via TI RTS add_device() that writes
 * printf output to a shared memory buffer. The ARM host reads this
 * buffer after inference completes.
 *
 * Buffer layout at the start of the printf region:
 *   [0..3]   magic    = 0x50524E54 ("PRNT")
 *   [4..7]   wr_index = bytes written so far
 *   [8..11]  buf_size = usable text area size (total - 16)
 *   [12..15] reserved
 *   [16..]   text data
 *
 * Two output paths:
 *   1. shm_printf() - direct fast path for DLOAD symbol alias
 *   2. SHM_write()  - add_device driver for fprintf(stdout,...) etc.
 */

#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include <kernel/dpl/CacheP.h>
#include <kernel/dpl/DebugP.h>

#include "shm_printf.h"

/* TI RTS file I/O interface */
#include <file.h>

/*
 * =============================================================================
 * Buffer Header
 * =============================================================================
 */

#define SHM_PRINTF_MAGIC    0x50524E54U  /* "PRNT" */
#define SHM_PRINTF_HDR_SIZE 16U

struct shm_printf_hdr {
    uint32_t magic;
    uint32_t wr_index;
    uint32_t buf_size;
    uint32_t reserved;
};

/*
 * =============================================================================
 * Module State
 * =============================================================================
 */

static struct shm_printf_hdr *g_hdr = NULL;
static char                  *g_buf = NULL;
static uint32_t               g_buf_size = 0;

/*
 * =============================================================================
 * Internal: write data to the SHM buffer
 * =============================================================================
 */

static void shm_buf_write(const char *data, uint32_t len)
{
    if (g_hdr == NULL || len == 0)
        return;

    uint32_t avail = g_buf_size - g_hdr->wr_index;
    if (len > avail)
        len = avail;  /* Silent truncation on overflow */

    if (len > 0) {
        memcpy(g_buf + g_hdr->wr_index, data, len);
        g_hdr->wr_index += len;
    }
}

/*
 * =============================================================================
 * add_device() Driver Callbacks
 * =============================================================================
 */

static int SHM_open(const char *path, unsigned flags, int llv_fd)
{
    (void)path;
    (void)flags;
    /* Return the low-level fd assigned by the RTS */
    return llv_fd;
}

static int SHM_close(int dev_fd)
{
    (void)dev_fd;
    return 0;
}

static int SHM_read(int dev_fd, char *buf, unsigned count)
{
    (void)dev_fd;
    (void)buf;
    (void)count;
    return 0;  /* Read not supported */
}

static int SHM_write(int dev_fd, const char *buf, unsigned count)
{
    (void)dev_fd;
    shm_buf_write(buf, (uint32_t)count);
    return (int)count;
}

static off_t SHM_lseek(int dev_fd, off_t offset, int origin)
{
    (void)dev_fd;
    (void)offset;
    (void)origin;
    return 0;
}

static int SHM_unlink(const char *path)
{
    (void)path;
    return 0;
}

static int SHM_rename(const char *old_name, const char *new_name)
{
    (void)old_name;
    (void)new_name;
    return 0;
}

/*
 * =============================================================================
 * Public API
 * =============================================================================
 */

void shm_printf_init(void *buf_addr, uint32_t buf_size)
{
    int ret;

    if (buf_addr == NULL || buf_size <= SHM_PRINTF_HDR_SIZE) {
        DebugP_log("[SHM_PRINTF] Invalid buffer: addr=%p size=%u\r\n",
                   buf_addr, buf_size);
        return;
    }

    /* Set up buffer pointers */
    g_hdr = (struct shm_printf_hdr *)buf_addr;
    g_buf = (char *)buf_addr + SHM_PRINTF_HDR_SIZE;
    g_buf_size = buf_size - SHM_PRINTF_HDR_SIZE;

    /* Initialize header */
    g_hdr->magic = SHM_PRINTF_MAGIC;
    g_hdr->wr_index = 0;
    g_hdr->buf_size = g_buf_size;
    g_hdr->reserved = 0;

    /* Register the shmout device with TI RTS */
    ret = add_device("shmout", _MSA,
                     SHM_open, SHM_close, SHM_read, SHM_write,
                     SHM_lseek, SHM_unlink, SHM_rename);
    if (ret != 0) {
        DebugP_log("[SHM_PRINTF] add_device failed: %d\r\n", ret);
        return;
    }

    /* Redirect stdout to the shmout device */
    if (freopen("shmout:stdout", "w", stdout) == NULL) {
        DebugP_log("[SHM_PRINTF] freopen failed\r\n");
        return;
    }

    /* Line-buffered so each printf line appears in the buffer */
    setvbuf(stdout, NULL, _IOLBF, 0);

    DebugP_log("[SHM_PRINTF] Initialized: buf=%p size=%u text=%u\r\n",
               buf_addr, buf_size, g_buf_size);
}

void shm_printf_reset(void)
{
    if (g_hdr != NULL) {
        g_hdr->wr_index = 0;
    }
}

uint32_t shm_printf_finish(void)
{
    uint32_t written;

    if (g_hdr == NULL)
        return 0;

    /* Flush any buffered stdout data through the device driver */
    fflush(stdout);

    written = g_hdr->wr_index;

    if (written > 0) {
        /* Writeback header + text data so ARM can read it */
        uint32_t wb_size = SHM_PRINTF_HDR_SIZE + written;
        CacheP_wb((void *)g_hdr, wb_size, CacheP_TYPE_ALL);
    }

    return written;
}

int shm_printf(const char *fmt, ...)
{
    char tmp[512];
    va_list args;
    int len;

    if (g_hdr == NULL)
        return -1;

    va_start(args, fmt);
    len = vsnprintf(tmp, sizeof(tmp), fmt, args);
    va_end(args);

    if (len > 0) {
        uint32_t n = (uint32_t)len;
        if (n > sizeof(tmp) - 1)
            n = sizeof(tmp) - 1;
        shm_buf_write(tmp, n);
    }

    return len;
}
