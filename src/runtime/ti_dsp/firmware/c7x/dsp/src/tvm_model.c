/*
 * C7x Compute Service - TVM Model Manager
 *
 * Manages TVM model weights/constants lifecycle. Uses the TVM DSP
 * runtime's constants parsing API (TVMDSPSetWeightsData, TVMDSPLoadConstants)
 * to convert weights.bin into TVMFFIAny constant arrays.
 *
 * Current limitation: only one model can be loaded at a time because
 * the TVM runtime uses global state for constants. A future version
 * could support multiple models by saving/restoring constants arrays.
 */

#include <stdint.h>
#include <string.h>

#include <kernel/dpl/DebugP.h>
#include <kernel/dpl/CacheP.h>

#include "tvm_model.h"

/* TVM DSP Runtime - Constants C API */
extern void  TVMDSPSetWeightsData(const void *data, size_t size);
extern int   TVMDSPLoadConstants(void);
extern void *TVMDSPGetConstant(int index);
extern void *TVMDSPGetAllConstants(int *count);
extern int   TVMDSPConstantsLoaded(void);

/*
 * =============================================================================
 * Model State
 * =============================================================================
 */

#define MAX_MODELS  4

typedef struct {
    uint32_t    id;
    int         num_constants;
    void       *constants;      /* TVMFFIAny* from TVMDSPGetAllConstants */
    int         active;
} model_entry_t;

static model_entry_t g_models[MAX_MODELS];
static uint32_t g_next_model_id = 1;
static int g_initialized = 0;

/*
 * =============================================================================
 * Internal Helpers
 * =============================================================================
 */

static model_entry_t *find_model_by_id(uint32_t model_id)
{
    int slot;
    for (slot = 0; slot < MAX_MODELS; slot++) {
        if (g_models[slot].active && g_models[slot].id == model_id)
            return &g_models[slot];
    }
    return NULL;
}

/*
 * =============================================================================
 * Public API
 * =============================================================================
 */

int32_t tvm_model_init(void)
{
    if (g_initialized) return 0;

    memset(g_models, 0, sizeof(g_models));
    g_next_model_id = 1;
    g_initialized = 1;

    DebugP_log("[TVM_MODEL] Model manager initialized\r\n");
    return 0;
}

int32_t tvm_model_load_weights(uint64_t weights_addr, uint32_t weights_size,
                               uint32_t *model_id_out)
{
    int slot;
    int num_constants;

    if (!g_initialized) return -1;
    if (model_id_out == NULL) return -2;

    /* Find free slot */
    for (slot = 0; slot < MAX_MODELS; slot++) {
        if (!g_models[slot].active) break;
    }
    if (slot >= MAX_MODELS) {
        DebugP_log("[TVM_MODEL] No free model slots\r\n");
        return -3;
    }

    /* Invalidate cache on weights data to ensure we read from DDR */
    CacheP_inv((void *)(uintptr_t)weights_addr, weights_size, CacheP_TYPE_ALL);

    DebugP_log("[TVM_MODEL] Loading weights: addr=0x%08llx size=%u\r\n",
               weights_addr, weights_size);

    /* Set weights data source for TVM runtime */
    TVMDSPSetWeightsData((const void *)(uintptr_t)weights_addr, weights_size);

    /* Parse constants */
    num_constants = TVMDSPLoadConstants();
    if (num_constants < 0) {
        DebugP_log("[TVM_MODEL] Failed to parse constants: %d\r\n", num_constants);
        return -4;
    }

    /* Get the parsed constants array */
    int count = 0;
    void *constants = TVMDSPGetAllConstants(&count);
    if (constants == NULL || count != num_constants) {
        DebugP_log("[TVM_MODEL] Failed to get constants array\r\n");
        return -5;
    }

    /* Store model entry */
    g_models[slot].id = g_next_model_id++;
    g_models[slot].num_constants = num_constants;
    g_models[slot].constants = constants;
    g_models[slot].active = 1;

    *model_id_out = g_models[slot].id;

    DebugP_log("[TVM_MODEL] Loaded model_id=%u, %d constants\r\n",
               *model_id_out, num_constants);
    return 0;
}

int32_t tvm_model_get_constants(uint32_t model_id,
                                struct TVMFFIAny **constants_out,
                                int *num_constants_out)
{
    model_entry_t *entry;

    if (!g_initialized) return -1;
    if (constants_out == NULL || num_constants_out == NULL) return -2;

    entry = find_model_by_id(model_id);
    if (entry == NULL) {
        DebugP_log("[TVM_MODEL] Model not found: id=%u\r\n", model_id);
        return -3;
    }

    *constants_out = (struct TVMFFIAny *)entry->constants;
    *num_constants_out = entry->num_constants;
    return 0;
}

int32_t tvm_model_unload(uint32_t model_id)
{
    model_entry_t *entry;

    if (!g_initialized) return -1;

    entry = find_model_by_id(model_id);
    if (entry == NULL) {
        DebugP_log("[TVM_MODEL] Model not found: id=%u\r\n", model_id);
        return -2;
    }

    entry->active = 0;
    entry->constants = NULL;
    entry->num_constants = 0;
    DebugP_log("[TVM_MODEL] Unloaded model_id=%u\r\n", model_id);
    return 0;
}

void tvm_model_deinit(void)
{
    int slot;

    for (slot = 0; slot < MAX_MODELS; slot++) {
        if (g_models[slot].active) {
            tvm_model_unload(g_models[slot].id);
        }
    }

    g_initialized = 0;
    DebugP_log("[TVM_MODEL] Model manager deinitialized\r\n");
}
