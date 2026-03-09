/*
 * board_init.h
 *
 * AWRL6844 board initialization wrapper for layer tests
 * Provides unified init/deinit interface for test execution
 */

#ifndef AWRL6844_BOARD_INIT_H_
#define AWRL6844_BOARD_INIT_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize AWRL6844 board resources for test execution
 *
 * Performs complete system initialization sequence:
 * 1. System_init() - DPL, PowerClock, Pinmux, Drivers
 * 2. Board_init() - Board-specific initialization
 * 3. Drivers_open() - Open UART, EDMA drivers
 * 4. Board_driversOpen() - Open board-specific drivers
 *
 * Call this once at the start of main() before running tests.
 *
 * @return 0 on success, negative error code on failure
 */
int awrl6844_board_init(void);

/**
 * @brief Deinitialize AWRL6844 board resources
 *
 * Performs cleanup sequence:
 * 1. Board_driversClose() - Close board-specific drivers
 * 2. Drivers_close() - Close UART, EDMA drivers
 * 3. Board_deinit() - Board-specific cleanup
 * 4. System_deinit() - System-level cleanup
 *
 * Call this at the end of main() after all tests complete.
 */
void awrl6844_board_deinit(void);

#ifdef __cplusplus
}
#endif

#endif /* AWRL6844_BOARD_INIT_H_ */
