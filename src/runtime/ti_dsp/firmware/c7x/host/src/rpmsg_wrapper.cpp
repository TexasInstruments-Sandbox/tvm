/*
 * C7x Compute Service - RPMessage Wrapper Implementation
 *
 * Uses Linux rpmsg_char interface for communication with DSP.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <fcntl.h>
#include <climits>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/select.h>
#include <linux/rpmsg.h>

#include "rpmsg_wrapper.h"
#include "raii.h"

/* Maximum path length for device files */
#define MAX_PATH_LEN 256

/* RPMSG control device path format */
#define RPMSG_CTRL_DEV_FMT "/dev/rpmsg_ctrl%d"

/* RPMSG endpoint device path format */
#define RPMSG_ENDPT_DEV_FMT "/dev/rpmsg%d"

/*
 * Find the rpmsg_ctrl device for a given DSP by matching its device tree
 * address in sysfs. The rpmsg_ctrl index can change across reboots or
 * remoteproc stop/start cycles, but the device address (e.g. "7e000000.dsp")
 * is stable hardware identity from the device tree.
 */
static int find_rpmsg_ctrl(const char *device_addr)
{
    char sysfs_path[MAX_PATH_LEN];
    char resolved[PATH_MAX];
    char dev_path[MAX_PATH_LEN];

    for (int ctrl_id = 0; ctrl_id < 10; ctrl_id++) {
        snprintf(sysfs_path, sizeof(sysfs_path),
                 "/sys/class/rpmsg/rpmsg_ctrl%d/device", ctrl_id);

        /* Resolve the full sysfs path (follows symlinks) */
        if (realpath(sysfs_path, resolved) == nullptr)
            continue;

        /* Check if the device address appears in the resolved path */
        if (strstr(resolved, device_addr) != nullptr) {
            snprintf(dev_path, sizeof(dev_path), RPMSG_CTRL_DEV_FMT, ctrl_id);
            int fd = open(dev_path, O_RDWR);
            if (fd >= 0) {
                fprintf(stderr, "rpmsg: Using rpmsg_ctrl%d (%s)\n",
                        ctrl_id, device_addr);
                return fd;
            }
        }
    }

    return -ENODEV;
}

/*
 * Find the highest-numbered /dev/rpmsgN that currently exists.
 * Returns -1 if none exist.
 */
static int find_max_rpmsg_index(void)
{
    char path[MAX_PATH_LEN];
    int hi = -1;

    /* Binary-search-style: probe at increasing powers of 2, then narrow. */
    int upper = 1;
    while (upper < 100000) {
        snprintf(path, sizeof(path), RPMSG_ENDPT_DEV_FMT, upper);
        if (access(path, F_OK) != 0)
            break;
        upper *= 2;
    }

    /* Linear scan from upper/2 .. upper to find exact max */
    int lo = (upper > 1) ? upper / 2 : 0;
    for (int i = lo; i <= upper; i++) {
        snprintf(path, sizeof(path), RPMSG_ENDPT_DEV_FMT, i);
        if (access(path, F_OK) == 0)
            hi = i;
    }

    /* Also check below lo (there may be gaps) — only needed on the
     * very first call when lo=0 and the scan above already covered it. */
    if (lo > 0) {
        for (int i = 0; i < lo; i++) {
            snprintf(path, sizeof(path), RPMSG_ENDPT_DEV_FMT, i);
            if (access(path, F_OK) == 0 && i > hi)
                hi = i;
        }
    }

    return hi;
}

int rpmsg_open(const char *device_addr, int remote_endpt, const char *service_name)
{
    char path[MAX_PATH_LEN];
    struct rpmsg_endpoint_info ept_info;

    /*
     * Record the highest /dev/rpmsgN index before creating the endpoint.
     * The TI rpmsg_char driver does not remove device files when the fd
     * is closed, so indices grow monotonically.  The new endpoint will
     * get an index > max_before.
     */
    int max_before = find_max_rpmsg_index();

    /* Find and open rpmsg_ctrl device by matching device tree address */
    UniqueFd ctrl_fd(find_rpmsg_ctrl(device_addr));
    if (!ctrl_fd) {
        fprintf(stderr, "rpmsg: Failed to find rpmsg_ctrl for device %s\n", device_addr);
        return ctrl_fd.get();
    }

    /* Create endpoint */
    memset(&ept_info, 0, sizeof(ept_info));
    if (service_name) {
        strncpy(ept_info.name, service_name, sizeof(ept_info.name) - 1);
    } else {
        snprintf(ept_info.name, sizeof(ept_info.name), "rpmsg-client-%d", getpid());
    }
    ept_info.src = RPMSG_ADDR_ANY;  /* Let kernel assign local endpoint */
    ept_info.dst = remote_endpt;

    int ret = ioctl(ctrl_fd.get(), RPMSG_CREATE_EPT_IOCTL, &ept_info);
    if (ret < 0) {
        fprintf(stderr, "rpmsg: Failed to create endpoint: %s\n", strerror(errno));
        return -errno;
    }

    /* ctrl_fd closed automatically by UniqueFd destructor */

    /*
     * Find the newly created endpoint device.
     * The new device will have an index > max_before since the kernel
     * assigns monotonically increasing minor numbers.
     */
    usleep(100000); /* 100ms for device node to appear */

    int start = (max_before >= 0) ? max_before + 1 : 0;
    int scan_limit = start + 16;  /* New device should be right after */

    for (int i = start; i < scan_limit; i++) {
        snprintf(path, sizeof(path), RPMSG_ENDPT_DEV_FMT, i);
        int endpt_fd = open(path, O_RDWR);
        if (endpt_fd >= 0) {
            fprintf(stderr, "rpmsg: Opened endpoint device %s\n", path);
            return endpt_fd;
        }
    }

    fprintf(stderr, "rpmsg: Failed to find endpoint device (scanned %d..%d)\n",
            start, scan_limit - 1);
    return -ENODEV;
}

int rpmsg_send(int fd, const void *data, size_t len)
{
    ssize_t ret = write(fd, data, len);
    if (ret < 0) {
        return -errno;
    }
    return static_cast<int>(ret);
}

int rpmsg_recv(int fd, void *data, size_t max_len, int timeout_ms)
{
    fd_set rfds;
    struct timeval tv;
    struct timeval *ptv = nullptr;

    /* Set up timeout */
    if (timeout_ms >= 0) {
        tv.tv_sec = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;
        ptv = &tv;
    }

    /* Wait for data */
    FD_ZERO(&rfds);
    FD_SET(fd, &rfds);

    ssize_t ret = select(fd + 1, &rfds, nullptr, nullptr, ptv);
    if (ret < 0) {
        return -errno;
    }
    if (ret == 0) {
        return -ETIMEDOUT;
    }

    /* Read data */
    ret = read(fd, data, max_len);
    if (ret < 0) {
        return -errno;
    }

    return static_cast<int>(ret);
}

void rpmsg_close(int fd)
{
    if (fd >= 0) {
        /* The endpoint is destroyed when the device is closed */
        close(fd);
    }
}

int rpmsg_get_local_endpt(int fd)
{
    /* This would require reading from sysfs or using an ioctl */
    /* For now, return -1 to indicate not implemented */
    (void)fd;
    return -ENOSYS;
}
