/*
 * C7x Compute Service - TVM Model Manager
 *
 * Manages loading/unloading of TVM model weights (constants).
 * Uses the TVM DSP runtime's constants parser to process weights.bin
 * data into TVMFFIAny constant arrays that cg_main_dsp can use.
 */

#ifndef TVM_MODEL_H
#define TVM_MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque forward declaration - actual type is TVMFFIAny from TVM runtime */
struct TVMFFIAny;

/**
 * Initialize the TVM model manager.
 *
 * Must be called after tvm_dsp_platform_init().
 *
 * @return 0 on success, negative error code on failure
 */
int32_t tvm_model_init(void);

/**
 * Load model weights from memory.
 *
 * Parses weights.bin data into the TVM runtime's constants array.
 * The weights data must be accessible at the given address (typically
 * the shared input buffer).
 *
 * @param weights_addr  Address of weights.bin data
 * @param weights_size  Size of weights data in bytes
 * @param model_id_out  Output: model ID for future reference
 *
 * @return 0 on success, negative error code on failure
 */
int32_t tvm_model_load_weights(uint64_t weights_addr, uint32_t weights_size,
                               uint32_t *model_id_out);

/**
 * Get the constants array for a loaded model.
 *
 * @param model_id        Model ID from tvm_model_load_weights()
 * @param constants_out   Output: pointer to TVMFFIAny constants array
 * @param num_constants_out Output: number of constants
 *
 * @return 0 on success, negative error code on failure
 */
int32_t tvm_model_get_constants(uint32_t model_id,
                                struct TVMFFIAny **constants_out,
                                int *num_constants_out);

/**
 * Unload a previously loaded model.
 *
 * @param model_id  Model ID from tvm_model_load_weights()
 *
 * @return 0 on success, negative error code on failure
 */
int32_t tvm_model_unload(uint32_t model_id);

/**
 * Deinitialize the TVM model manager.
 */
void tvm_model_deinit(void);

#ifdef __cplusplus
}
#endif

#endif /* TVM_MODEL_H */
