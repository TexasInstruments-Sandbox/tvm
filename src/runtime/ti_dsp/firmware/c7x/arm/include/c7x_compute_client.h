/*
 * C7x Compute Service - Host Client Library Header
 *
 * Provides easy-to-use API for communicating with C7x DSP.
 */

#ifndef C7X_COMPUTE_CLIENT_H
#define C7X_COMPUTE_CLIENT_H

#include <stdint.h>
#include <stddef.h>
#include "c7x_compute_protocol.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque client handle */
typedef struct c7x_client c7x_client_t;

/* Service status information */
typedef struct {
    uint32_t version;       /* Service version (0xMMmmpp) */
    uint32_t uptime_ms;     /* Uptime in milliseconds */
    uint32_t jobs_completed;/* Total successful jobs */
    uint32_t jobs_failed;   /* Total failed jobs */
} c7x_status_t;

/**
 * Open a connection to the C7x compute service.
 *
 * @return Client handle on success, NULL on failure
 */
c7x_client_t *c7x_client_open(void);

/**
 * Close the connection to the C7x compute service.
 *
 * @param client  Client handle from c7x_client_open
 */
void c7x_client_close(c7x_client_t *client);

/**
 * Test connectivity with the DSP (PING).
 *
 * @param client     Client handle
 * @param version    Output: service version (can be NULL)
 * @param uptime_ms  Output: service uptime (can be NULL)
 *
 * @return 0 on success, negative error code on failure
 */
int c7x_client_ping(c7x_client_t *client, uint32_t *version, uint32_t *uptime_ms);

/**
 * Get service status from the DSP.
 *
 * @param client  Client handle
 * @param status  Output: status information
 *
 * @return 0 on success, negative error code on failure
 */
int c7x_client_get_status(c7x_client_t *client, c7x_status_t *status);

/**
 * Get pointer to the input buffer in shared memory.
 *
 * @param client  Client handle
 * @param size    Output: buffer size in bytes
 *
 * @return Pointer to input buffer, or NULL on failure
 */
void *c7x_client_get_input_buffer(c7x_client_t *client, size_t *size);

/**
 * Get pointer to the output buffer in shared memory.
 *
 * @param client  Client handle
 * @param size    Output: buffer size in bytes
 *
 * @return Pointer to output buffer, or NULL on failure
 */
void *c7x_client_get_output_buffer(c7x_client_t *client, size_t *size);

/**
 * Get the staging buffer offset where input tensor data should start.
 *
 * After c7x_client_dyn_load(), the loaded ELF's in-place rodata segments
 * occupy [0, offset).  Input tensor data must be placed at or after this
 * offset to avoid overwriting the loaded module.  This is the value to use
 * as the base for CreateInput() allocations.
 *
 * @param client  Client handle
 *
 * @return Byte offset into the staging buffer (0 if client is NULL)
 */
size_t c7x_client_get_input_data_offset(c7x_client_t *client);

/*
 * =============================================================================
 * Dynamic Loading API
 * =============================================================================
 */

/**
 * Load a TVM model weights file onto the DSP.
 *
 * Reads the weights file, stages it in the shared input buffer,
 * and sends a MODEL_LOAD command. The DSP parses the weights
 * into TVM constants.
 *
 * @param client        Client handle
 * @param weights_file  Path to weights.bin file
 * @param model_id_out  Output: model ID for use with infer/unload
 *
 * @return 0 on success, negative error code on failure
 */
int c7x_client_model_load(c7x_client_t *client, const char *weights_file,
                          uint32_t *model_id_out);

/**
 * Unload a previously loaded model.
 *
 * @param client    Client handle
 * @param model_id  Model ID from c7x_client_model_load
 *
 * @return 0 on success, negative error code on failure
 */
int c7x_client_model_unload(c7x_client_t *client, uint32_t model_id);

/**
 * Dynamically load an ELF shared object onto the DSP.
 *
 * Reads the .out file, stages it in the shared input buffer,
 * and sends a DYN_LOAD command. The DSP loads segments, resolves
 * symbols against the firmware, and returns a module handle.
 *
 * @param client      Client handle
 * @param elf_file    Path to .out ELF file
 * @param handle_out  Output: module handle for use with infer/unload
 *
 * @return 0 on success, negative error code on failure
 */
int c7x_client_dyn_load(c7x_client_t *client, const char *elf_file,
                        uint32_t *handle_out);

/**
 * Unload a previously loaded dynamic module.
 *
 * @param client  Client handle
 * @param handle  Module handle from c7x_client_dyn_load
 *
 * @return 0 on success, negative error code on failure
 */
int c7x_client_dyn_unload(c7x_client_t *client, uint32_t handle);

/**
 * Tensor descriptor for host-side inference API.
 */
typedef struct {
    void    *data;          /* Pointer to tensor data (host memory) */
    size_t   data_size;     /* Size of data in bytes */
    int32_t  ndim;          /* Number of dimensions */
    int32_t  dtype_code;    /* DLDataType code (0=Int, 1=UInt, 2=Float) */
    int32_t  dtype_bits;    /* Bits per element */
    int64_t  shape[C7X_TENSOR_MAX_NDIM]; /* Dimension sizes */
} c7x_tensor_desc_t;

/**
 * Run inference on the DSP.
 *
 * Stages input tensor data in the shared input buffer, sends the
 * INFER command with tensor metadata, and reads back the output.
 *
 * @param client         Client handle
 * @param module_handle  Module handle from c7x_client_dyn_load
 * @param model_id       Model ID from c7x_client_model_load
 * @param inputs         Array of input tensor descriptors
 * @param num_inputs     Number of input tensors
 * @param outputs        Output: array of output tensor descriptors
 * @param num_outputs    Output: number of output tensors
 * @param cycles         Output: DSP cycles consumed, 64-bit (can be NULL)
 *
 * @return 0 on success, negative error code on failure
 */
int c7x_client_infer(c7x_client_t *client,
                     uint32_t module_handle,
                     uint32_t model_id,
                     const c7x_tensor_desc_t *inputs, int num_inputs,
                     c7x_tensor_desc_t *outputs, int *num_outputs,
                     uint64_t *cycles);

/**
 * Run inference with repeat count (for profiling).
 *
 * Same as c7x_client_infer but sets the repeat count in the INFER
 * request flags.  The firmware loops cg_main_dsp() ``repeat`` times,
 * recording per-iteration cycles.  Use repeat=2 to separate one-time
 * init cost from steady-state inference.
 *
 * The response ``cycles`` reports the LAST iteration (steady-state).
 * DSP printf output contains all iterations' layer profiles.
 *
 * @param repeat  Number of inference iterations (0 or 1 = run once)
 */
int c7x_client_infer_repeat(c7x_client_t *client,
                            uint32_t module_handle,
                            uint32_t model_id,
                            const c7x_tensor_desc_t *inputs, int num_inputs,
                            c7x_tensor_desc_t *outputs, int *num_outputs,
                            uint64_t *cycles,
                            uint32_t repeat);

/**
 * Get error message for a status code.
 *
 * @param status  Status code from c7x functions
 *
 * @return Human-readable error message
 */
const char *c7x_strerror(int status);

#ifdef __cplusplus
}
#endif

#endif /* C7X_COMPUTE_CLIENT_H */
