/*
 * C7x Compute Service - RPMessage Wrapper Header
 *
 * Thin wrapper around Linux rpmsg_char interface.
 */

#ifndef RPMSG_WRAPPER_H
#define RPMSG_WRAPPER_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Open an RPMessage connection to a remote endpoint.
 *
 * @param device_addr   Device tree address (e.g., "7e000000.dsp")
 * @param remote_endpt  Remote endpoint number
 * @param service_name  Service name for announcement (can be NULL)
 *
 * @return File descriptor on success, negative error code on failure
 */
int rpmsg_open(const char *device_addr, int remote_endpt, const char *service_name);

/**
 * Send data to the remote endpoint.
 *
 * @param fd    File descriptor from rpmsg_open
 * @param data  Data to send
 * @param len   Length of data in bytes
 *
 * @return Number of bytes sent on success, negative error code on failure
 */
int rpmsg_send(int fd, const void *data, size_t len);

/**
 * Receive data from the remote endpoint.
 *
 * @param fd        File descriptor from rpmsg_open
 * @param data      Buffer to receive data
 * @param max_len   Maximum bytes to receive
 * @param timeout_ms  Timeout in milliseconds (-1 for infinite)
 *
 * @return Number of bytes received on success, negative error code on failure
 */
int rpmsg_recv(int fd, void *data, size_t max_len, int timeout_ms);

/**
 * Close the RPMessage connection.
 *
 * @param fd  File descriptor from rpmsg_open
 */
void rpmsg_close(int fd);

/**
 * Get the local endpoint number.
 *
 * @param fd  File descriptor from rpmsg_open
 *
 * @return Local endpoint number, or negative error code
 */
int rpmsg_get_local_endpt(int fd);

#ifdef __cplusplus
}
#endif

#endif /* RPMSG_WRAPPER_H */
