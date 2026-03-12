/*
 * C7x Compute Service - DSP Service Implementation
 *
 * Handles RPMessage communication with Linux host and dispatches
 * compute operations on shared memory buffers.
 */

#include <stdint.h>
#include <string.h>
#include <kernel/dpl/DebugP.h>
#include <kernel/dpl/ClockP.h>
#include <kernel/dpl/CycleCounterP.h>
#include <c7x.h>  /* __TSC (64-bit cycle counter) */
#include <kernel/dpl/CacheP.h>
#include <drivers/ipc_rpmsg.h>

#include "compute_service.h"
#include "c7x_compute_protocol.h"
#include "dyn_loader.h"
#include "tvm_model.h"
#include "dma/tvm_dsp_dma.h"

#include <kernel/nortos/dpl/c75/csl_clec.h>
#include <drivers/hw_include/cslr_soc.h>
#include "shm_printf.h"

/* TVM DSP Runtime types */
#include "ffi_types.h"
#include "ndarray.h"

/* TVM runtime cleanup functions (defined in TVM DSP runtime library) */
extern int TVMDSPRegFileCleanup(void);
extern void TVMDSPConstantsCleanup(void);
extern void tvm_dsp_reset_pools(void);

/* cg_main_dsp function pointer type */
typedef int (*cg_main_dsp_fn)(TVMFFIAny *inputs, int num_inputs,
                              TVMFFIAny *constants, TVMFFIAny *output);

/* Track the currently loaded module handle and entry point */
static uint32_t g_loaded_module_handle = 0;
static cg_main_dsp_fn g_cg_main_dsp = NULL;
static uint32_t g_embedded_model_id = 0;

/*
 * =============================================================================
 * Service State
 * =============================================================================
 */

/* RPMessage endpoint */
static RPMessage_Object gRpmsgObj;
static uint16_t         gLocalEndpoint;

/* Service statistics */
static uint32_t gJobsCompleted = 0;
static uint32_t gJobsFailed = 0;
static uint64_t gStartTimeUs = 0;

/* Message buffers */
static uint8_t gRecvBuf[C7X_MAX_MSG_SIZE] __attribute__((aligned(16)));
static uint8_t gSendBuf[C7X_MAX_MSG_SIZE] __attribute__((aligned(16)));

/* Service running flag */
static volatile uint32_t gServiceRunning = 0;

/*
 * =============================================================================
 * Response Helper
 * =============================================================================
 */

static void send_response(uint32_t msg_type, uint32_t seq,
                           uint32_t resp_len,
                           uint16_t srcCore, uint32_t srcEndpt)
{
    struct c7x_msg_hdr *hdr = (struct c7x_msg_hdr *)gSendBuf;
    int32_t status;

    hdr->type = msg_type;
    hdr->seq = seq;
    hdr->len = resp_len;

    status = RPMessage_send(hdr, resp_len,
                            srcCore, srcEndpt,
                            RPMessage_getLocalEndPt(&gRpmsgObj),
                            SystemP_WAIT_FOREVER);
    if (status != SystemP_SUCCESS) {
        DebugP_log("[COMPUTE] Failed to send response type=0x%x: %d\r\n",
                   msg_type, status);
    }
}

/*
 * =============================================================================
 * Message Handlers
 * =============================================================================
 */

static void handle_ping(struct c7x_msg_ping *req,
                        uint16_t srcCore, uint32_t srcEndpt)
{
    struct c7x_msg_ping_resp *resp = (struct c7x_msg_ping_resp *)gSendBuf;

    DebugP_log("[COMPUTE] PING from core %u endpoint %u\r\n", srcCore, srcEndpt);

    /* Build response */
    resp->hdr.status = C7X_STATUS_SUCCESS;
    resp->version = C7X_SERVICE_VERSION;
    resp->uptime_ms = (uint32_t)((ClockP_getTimeUsec() - gStartTimeUs) / 1000);

    send_response(C7X_MSG_PING_RESP, req->hdr.seq,
                  sizeof(*resp), srcCore, srcEndpt);
}

static void handle_get_status(struct c7x_msg_get_status *req,
                              uint16_t srcCore, uint32_t srcEndpt)
{
    struct c7x_msg_status_resp *resp = (struct c7x_msg_status_resp *)gSendBuf;

    DebugP_log("[COMPUTE] STATUS request\r\n");

    /* Build response */
    resp->hdr.status = C7X_STATUS_SUCCESS;
    resp->version = C7X_SERVICE_VERSION;
    resp->uptime_ms = (uint32_t)((ClockP_getTimeUsec() - gStartTimeUs) / 1000);
    resp->jobs_completed = gJobsCompleted;
    resp->jobs_failed = gJobsFailed;

    send_response(C7X_MSG_STATUS_RESP, req->hdr.seq,
                  sizeof(*resp), srcCore, srcEndpt);
}

/*
 * =============================================================================
 * Dynamic Loading & TVM Inference Handlers
 * =============================================================================
 */

static void handle_dyn_load(struct c7x_msg_dyn_load *req,
                            uint16_t srcCore, uint32_t srcEndpt)
{
    struct c7x_msg_dyn_load_resp *resp = (struct c7x_msg_dyn_load_resp *)gSendBuf;
    int32_t status;
    uint32_t handle = 0;
    uint64_t sym_addr = 0;

    DebugP_log("[COMPUTE] DYN_LOAD elf_size=%u\r\n", req->elf_size);

    /* Validate ELF size */
    if (req->elf_size == 0 || req->elf_size > C7X_INPUT_BUFFER_SIZE) {
        DebugP_log("[COMPUTE] Invalid ELF size\r\n");
        resp->hdr.status = C7X_STATUS_ERR_SIZE;
        goto done;
    }

    /* Load the ELF from input buffer */
    status = dyn_loader_load(C7X_INPUT_BUFFER_ADDR, req->elf_size, &handle);
    if (status != 0) {
        DebugP_log("[COMPUTE] dyn_loader_load failed: %d\r\n", status);
        resp->hdr.status = C7X_STATUS_ERR_LOAD;
        goto done;
    }

    /* Look up cg_main_dsp entry point */
    status = dyn_loader_query_symbol(handle, "cg_main_dsp", &sym_addr);
    if (status == 0 && sym_addr != 0) {
        g_cg_main_dsp = (cg_main_dsp_fn)(uintptr_t)sym_addr;
        DebugP_log("[COMPUTE] Found cg_main_dsp at 0x%08llx\r\n", sym_addr);
    } else {
        DebugP_log("[COMPUTE] cg_main_dsp not found (will look up at infer time)\r\n");
        g_cg_main_dsp = NULL;
    }

    g_loaded_module_handle = handle;

    /* Check for embedded weights in loaded module */
    {
        uint64_t ws_addr = 0, wz_addr = 0;
        if (dyn_loader_query_symbol(handle, "_binary_weights_bin_start", &ws_addr) == 0 &&
            dyn_loader_query_symbol(handle, "_binary_weights_bin_size", &wz_addr) == 0 &&
            ws_addr != 0 && wz_addr != 0) {
            /* _binary_weights_bin_size is a .word label — dereference to get value */
            uint32_t wsize = *(const uint32_t *)(uintptr_t)wz_addr;
            DebugP_log("[COMPUTE] Embedded weights: addr=0x%08llx size=%u\r\n",
                       ws_addr, wsize);
            uint32_t mid = 0;
            if (tvm_model_load_weights(ws_addr, wsize, &mid) == 0) {
                g_embedded_model_id = mid;
                DebugP_log("[COMPUTE] Embedded weights loaded: model_id=%u\r\n", mid);
            } else {
                DebugP_log("[COMPUTE] Embedded weights parse failed\r\n");
            }
        }
    }

    resp->hdr.status = C7X_STATUS_SUCCESS;
    resp->module_handle = handle;
    /* text_size and data_size are set by DLIF callbacks, we report 0 for now */
    resp->text_size = 0;
    resp->data_size = 0;

    DebugP_log("[COMPUTE] Module loaded: handle=%u\r\n", handle);

done:
    send_response(C7X_MSG_DYN_LOAD_RESP, req->hdr.seq,
                  sizeof(*resp), srcCore, srcEndpt);
}

static void handle_dyn_unload(struct c7x_msg_dyn_unload *req,
                              uint16_t srcCore, uint32_t srcEndpt)
{
    struct c7x_msg_dyn_unload_resp *resp = (struct c7x_msg_dyn_unload_resp *)gSendBuf;
    int32_t status;

    DebugP_log("[COMPUTE] DYN_UNLOAD handle=%u\r\n", req->module_handle);

    /*
     * Clean up TVM runtime state BEFORE unloading the module.
     * The loaded module contains static data (e.g. the register file)
     * that must be accessed before dyn_loader_unload() frees the
     * module's memory segments.
     */
    if (req->module_handle == g_loaded_module_handle) {
        /* Free TIDL subgraph instances so TIDL can release DMA
         * channels, IALG memory, and MMA state before the module's
         * code/data sections are freed by dyn_loader_unload(). */
        {
            uint64_t cleanup_addr = 0;
            if (dyn_loader_query_symbol(req->module_handle,
                    "tidl_bridge_cleanup", &cleanup_addr) == 0
                    && cleanup_addr != 0) {
                DebugP_log("[COMPUTE] Calling tidl_bridge_cleanup\r\n");
                ((void (*)(void))(uintptr_t)cleanup_addr)();
                DebugP_log("[COMPUTE] tidl_bridge_cleanup done\r\n");
            }
        }

        /* Free storage/NDArray objects from the last inference.
         * These are heap-allocated objects referenced by the static
         * register file inside the loaded module. */
        TVMDSPRegFileCleanup();

        /* Free the model slot so it can be reused */
        if (g_embedded_model_id != 0) {
            tvm_model_unload(g_embedded_model_id);
        }

        /* Free constants memory pools */
        TVMDSPConstantsCleanup();
    }

    status = dyn_loader_unload(req->module_handle);
    if (status != 0) {
        resp->hdr.status = C7X_STATUS_ERR_HANDLE;
    } else {
        resp->hdr.status = C7X_STATUS_SUCCESS;
        if (req->module_handle == g_loaded_module_handle) {
            g_loaded_module_handle = 0;
            g_cg_main_dsp = NULL;
            g_embedded_model_id = 0;
        }
        /* UDMA driver is initialized once at boot and shared between
         * TVM DMA tiling and TIDL.  Do NOT call tvm_dsp_dma_deinit()
         * here — TIDL's algFree releases its DMA channels from the
         * shared driver, and calling Udma_deinit after that crashes
         * on stale channel handles. */

        /* Reset memory pools to eliminate fragmentation from the
         * load/unload cycle.  All pool memory has been freed by
         * the cleanup steps above + dyn_loader_unload. */
        tvm_dsp_reset_pools();
    }

    send_response(C7X_MSG_DYN_UNLOAD_RESP, req->hdr.seq,
                  sizeof(*resp), srcCore, srcEndpt);
}

static void handle_model_load(struct c7x_msg_model_load *req,
                              uint16_t srcCore, uint32_t srcEndpt)
{
    struct c7x_msg_model_load_resp *resp = (struct c7x_msg_model_load_resp *)gSendBuf;
    int32_t status;
    uint32_t model_id = 0;
    TVMFFIAny *constants = NULL;
    int num_constants = 0;

    DebugP_log("[COMPUTE] MODEL_LOAD weights_size=%u\r\n", req->weights_size);

    if (req->weights_size == 0 || req->weights_size > C7X_INPUT_BUFFER_SIZE) {
        DebugP_log("[COMPUTE] Invalid weights size\r\n");
        resp->hdr.status = C7X_STATUS_ERR_SIZE;
        resp->model_id = 0;
        resp->num_constants = 0;
        goto done;
    }

    /* Invalidate cache on weights data to ensure we read fresh data from DDR.
     * The ARM host wrote weights to the shared buffer and flushed its cache,
     * but the DSP D-cache may still hold stale data from a previous operation
     * (e.g. ELF data from a prior DLOAD load at the same address). */
    CacheP_inv((void *)(uintptr_t)C7X_INPUT_BUFFER_ADDR,
               req->weights_size, CacheP_TYPE_ALL);

    status = tvm_model_load_weights(C7X_INPUT_BUFFER_ADDR, req->weights_size,
                                    &model_id);
    if (status != 0) {
        DebugP_log("[COMPUTE] tvm_model_load_weights failed: %d\r\n", status);
        resp->hdr.status = C7X_STATUS_ERR_WEIGHTS;
        resp->model_id = 0;
        resp->num_constants = 0;
        goto done;
    }

    /* Get constants count */
    tvm_model_get_constants(model_id, &constants, &num_constants);

    resp->hdr.status = C7X_STATUS_SUCCESS;
    resp->model_id = model_id;
    resp->num_constants = (uint32_t)num_constants;

    DebugP_log("[COMPUTE] Model loaded: id=%u, %d constants\r\n",
               model_id, num_constants);

done:
    send_response(C7X_MSG_MODEL_LOAD_RESP, req->hdr.seq,
                  sizeof(*resp), srcCore, srcEndpt);
}

static void handle_model_unload(struct c7x_msg_model_unload *req,
                                uint16_t srcCore, uint32_t srcEndpt)
{
    struct c7x_msg_model_unload_resp *resp = (struct c7x_msg_model_unload_resp *)gSendBuf;
    int32_t status;

    DebugP_log("[COMPUTE] MODEL_UNLOAD id=%u\r\n", req->model_id);

    status = tvm_model_unload(req->model_id);
    resp->hdr.status = (status == 0) ? C7X_STATUS_SUCCESS : C7X_STATUS_ERR_HANDLE;

    send_response(C7X_MSG_MODEL_UNLOAD_RESP, req->hdr.seq,
                  sizeof(*resp), srcCore, srcEndpt);
}

#define MAX_INFER_INPUTS 4

/**
 * build_input_ndarrays - Validate, cache-invalidate and construct NDArrays
 *                        from INFER request tensor descriptors.
 *
 * Returns C7X_STATUS_SUCCESS or C7X_STATUS_ERR_TENSOR on bad address.
 */
static int32_t build_input_ndarrays(
    struct c7x_msg_infer *req,
    TVMDSPNDArray *ndarrays,
    int64_t shapes[][C7X_TENSOR_MAX_NDIM],
    TVMFFIAny *anys)
{
    uint32_t i;

    /* Validate and cache-invalidate input tensor data regions */
    for (i = 0; i < req->num_inputs; i++) {
        struct c7x_tensor_desc *td = &req->inputs[i];

        if (td->data_addr != 0 && td->data_size != 0) {
            if (!C7X_IS_VALID_INPUT_ADDR(td->data_addr, td->data_size)) {
                DebugP_log("[COMPUTE] Input %u addr 0x%llx+0x%llx outside "
                           "shared buffer\r\n",
                           i, (unsigned long long)td->data_addr,
                           (unsigned long long)td->data_size);
                return C7X_STATUS_ERR_TENSOR;
            }
            uint32_t cache_size = (td->data_size > 0xFFFFFFFFU)
                                  ? 0xFFFFFFFFU : (uint32_t)td->data_size;
            CacheP_inv((void *)(uintptr_t)td->data_addr,
                       cache_size, CacheP_TYPE_ALL);
        }
    }

    /* Build NDArray descriptors */
    memset(ndarrays, 0, MAX_INFER_INPUTS * sizeof(TVMDSPNDArray));
    memset(anys, 0, MAX_INFER_INPUTS * sizeof(TVMFFIAny));

    for (i = 0; i < req->num_inputs; i++) {
        struct c7x_tensor_desc *td = &req->inputs[i];

        int32_t ndim = td->ndim;
        if (ndim < 0) ndim = 0;
        if (ndim > C7X_TENSOR_MAX_NDIM) ndim = C7X_TENSOR_MAX_NDIM;

        int j;
        for (j = 0; j < ndim; j++) {
            shapes[i][j] = td->shape[j];
        }

        ndarrays[i].type_index = kTVMFFITensor;
        ndarrays[i].ref_counter = 1;
        ndarrays[i].deleter = NULL;
        ndarrays[i].data = (void *)(uintptr_t)td->data_addr;
        ndarrays[i].device.device_type = 0;
        ndarrays[i].device.device_id = 0;
        ndarrays[i].ndim = ndim;
        ndarrays[i].dtype.code = (uint8_t)td->dtype_code;
        ndarrays[i].dtype.bits = (uint8_t)td->dtype_bits;
        ndarrays[i].dtype.lanes = 1;
        ndarrays[i].shape = shapes[i];
        ndarrays[i].strides = NULL;
        ndarrays[i].byte_offset = 0;

        anys[i].type_index = kTVMFFITensor;
        anys[i].zero_padding = 0;
        anys[i].v_ptr = &ndarrays[i];
    }

    return C7X_STATUS_SUCCESS;
}

/**
 * extract_infer_output - Extract output tensor metadata from TVMFFIAny,
 *                        cache writeback and copy to shared output buffer.
 *
 * Fills resp->num_outputs and resp->outputs[0].
 */
static void extract_infer_output(TVMFFIAny *output_any,
                                 struct c7x_msg_infer_resp *resp)
{
    uint32_t i;

    resp->num_outputs = 0;
    if (output_any->type_index != kTVMFFITensor || output_any->v_ptr == NULL)
        return;

    TVMDSPNDArray *out_nd = (TVMDSPNDArray *)output_any->v_ptr;
    struct c7x_tensor_desc *out_td = &resp->outputs[0];

    out_td->data_addr = (uint64_t)(uintptr_t)out_nd->data;
    int32_t out_ndim = out_nd->ndim;
    if (out_ndim < 0) out_ndim = 0;
    if (out_ndim > C7X_TENSOR_MAX_NDIM) out_ndim = C7X_TENSOR_MAX_NDIM;
    out_td->ndim = out_ndim;
    out_td->dtype_code = out_nd->dtype.code;
    out_td->dtype_bits = out_nd->dtype.bits;

    int64_t total_elements = 1;
    for (i = 0; i < (uint32_t)out_ndim; i++) {
        out_td->shape[i] = out_nd->shape[i];
        total_elements *= out_nd->shape[i];
    }
    out_td->data_size = (uint64_t)total_elements * (out_nd->dtype.bits / 8);

    /* Cache writeback output data so host can read it */
    if (out_nd->data != NULL && out_td->data_size > 0) {
        uint32_t wb_size = (out_td->data_size > 0xFFFFFFFFU)
                           ? 0xFFFFFFFFU : (uint32_t)out_td->data_size;
        CacheP_wb(out_nd->data, wb_size, CacheP_TYPE_ALL);
    }

    /* Copy output data to shared output buffer if it's not already there */
    if ((uint64_t)(uintptr_t)out_nd->data < C7X_OUTPUT_BUFFER_ADDR ||
        (uint64_t)(uintptr_t)out_nd->data >= C7X_OUTPUT_BUFFER_ADDR + C7X_OUTPUT_BUFFER_SIZE) {
        if (out_td->data_size <= C7X_OUTPUT_BUFFER_SIZE) {
            memcpy((void *)(uintptr_t)C7X_OUTPUT_BUFFER_ADDR,
                   out_nd->data, (size_t)out_td->data_size);
            uint32_t wb_size2 = (out_td->data_size > 0xFFFFFFFFU)
                                ? 0xFFFFFFFFU : (uint32_t)out_td->data_size;
            CacheP_wb((void *)(uintptr_t)C7X_OUTPUT_BUFFER_ADDR,
                     wb_size2, CacheP_TYPE_ALL);
            out_td->data_addr = C7X_OUTPUT_BUFFER_ADDR;
        }
    }

    resp->num_outputs = 1;
}

static void handle_infer(struct c7x_msg_infer *req, uint16_t recvMsgSize,
                         uint16_t srcCore, uint32_t srcEndpt)
{
    struct c7x_msg_infer_resp *resp = (struct c7x_msg_infer_resp *)gSendBuf;
    int32_t status;
    uint64_t start_cycles, end_cycles;
    uint64_t sym_addr = 0;
    TVMFFIAny *constants = NULL;
    int num_constants = 0;
    int ret;
    TVMDSPNDArray input_ndarrays[MAX_INFER_INPUTS];
    int64_t input_shapes[MAX_INFER_INPUTS][C7X_TENSOR_MAX_NDIM];
    TVMFFIAny input_anys[MAX_INFER_INPUTS];
    TVMFFIAny output_any;

    DebugP_log("[COMPUTE] INFER module=%u model=%u inputs=%u\r\n",
               req->module_handle, req->model_id, req->num_inputs);

    /* A. Validate handles (model_id=0 allowed for testing without weights) */
    if (req->module_handle == 0) {
        DebugP_log("[COMPUTE] Invalid module handle\r\n");
        resp->hdr.status = C7X_STATUS_ERR_HANDLE;
        goto done;
    }

    /* Validate num_inputs against actual received message size */
    if (req->num_inputs > MAX_INFER_INPUTS) {
        DebugP_log("[COMPUTE] Too many inputs: %u\r\n", req->num_inputs);
        resp->hdr.status = C7X_STATUS_ERR_TENSOR;
        goto done;
    }
    {
        uint32_t required_size = (uint32_t)sizeof(struct c7x_msg_infer);
        if (req->num_inputs > 1) {
            required_size += (req->num_inputs - 1) * (uint32_t)sizeof(struct c7x_tensor_desc);
        }
        if ((uint32_t)recvMsgSize < required_size) {
            DebugP_log("[COMPUTE] INFER message too small for %u inputs: "
                       "got %u, need %u\r\n",
                       req->num_inputs, (uint32_t)recvMsgSize, required_size);
            resp->hdr.status = C7X_STATUS_ERR_TENSOR;
            goto done;
        }
    }

    /* B. Resolve entry point */
    if (g_cg_main_dsp == NULL) {
        status = dyn_loader_query_symbol(req->module_handle, "cg_main_dsp", &sym_addr);
        if (status != 0 || sym_addr == 0) {
            DebugP_log("[COMPUTE] cg_main_dsp symbol not found\r\n");
            resp->hdr.status = C7X_STATUS_ERR_SYMBOL;
            goto done;
        }
        g_cg_main_dsp = (cg_main_dsp_fn)(uintptr_t)sym_addr;
        DebugP_log("[COMPUTE] cg_main_dsp resolved at %p\r\n",
                   (void *)(uintptr_t)sym_addr);
    }

    /* C. Resolve constants */
    {
        uint32_t eff_model_id = req->model_id;
        if (eff_model_id == 0 && g_embedded_model_id != 0) {
            eff_model_id = g_embedded_model_id;
        }
        if (eff_model_id != 0) {
            status = tvm_model_get_constants(eff_model_id, &constants, &num_constants);
            if (status != 0) {
                DebugP_log("[COMPUTE] Failed to get constants\r\n");
                resp->hdr.status = C7X_STATUS_ERR_HANDLE;
                goto done;
            }
        }
    }

    /* D+E. Cache invalidation + NDArray construction */
    status = build_input_ndarrays(req, input_ndarrays, input_shapes, input_anys);
    if (status != C7X_STATUS_SUCCESS) {
        resp->hdr.status = status;
        goto done;
    }

    /* F. TVM execution — with optional repeat loop for profiling.
     * flags bits[15:0] = repeat count; 0 or 1 = run once. */
    {
        uint32_t repeat = req->flags & 0xFFFF;
        if (repeat < 1) repeat = 1;

        /* Look up layer-profile printer once (may not exist) */
        void (*print_profile)(void) = NULL;
        {
            uint64_t profile_fn_addr = 0;
            if (g_loaded_module_handle != 0 &&
                dyn_loader_query_symbol(g_loaded_module_handle,
                    "TVMPrintLayerProfile", &profile_fn_addr) == 0) {
                print_profile = (void (*)(void))(uintptr_t)profile_fn_addr;
            }
        }

        /* Reset printf buffer once before all iterations */
        shm_printf_reset();

        uint32_t iter;
        for (iter = 0; iter < repeat; iter++) {
            memset(&output_any, 0, sizeof(output_any));
            output_any.type_index = kTVMFFINone;

            CycleCounterP_reset();
            start_cycles = __TSC;

            DebugP_log("[COMPUTE] >>> calling cg_main_dsp at %p (iter %u/%u)\r\n",
                       (void *)g_cg_main_dsp, iter + 1, repeat);

            ret = g_cg_main_dsp(input_anys, (int)req->num_inputs,
                                 constants, &output_any);

            end_cycles = __TSC;

            uint64_t iter_cycles = end_cycles - start_cycles;

            DebugP_log("[COMPUTE] cg_main_dsp returned %d, %llu cycles\r\n",
                       ret, (unsigned long long)iter_cycles);

            /* Per-iteration header (only when profiling with repeat>1) */
            if (repeat > 1) {
                printf("\n[Iteration %u/%u] %llu cycles (%.2f ms)\n",
                       iter + 1, repeat,
                       (unsigned long long)iter_cycles,
                       iter_cycles / 1e6);
            }

            /* Print layer profile AFTER cycle recording so printf
             * overhead is not counted in inference cycles. */
            if (print_profile) {
                print_profile();
            }

            /* Keep last iteration's cycles for the response */
            resp->return_value = ret;
            resp->cycles = iter_cycles;

            if (ret != 0) break;
        }
    }

    if (ret != 0) {
        resp->hdr.status = C7X_STATUS_ERR_CALL;
        resp->num_outputs = 0;
        gJobsFailed++;
        goto done;
    }

    /* G. Output extraction + staging */
    extract_infer_output(&output_any, resp);

    /* Flush printf buffer and report size to host */
    resp->printf_size = shm_printf_finish();

    resp->hdr.status = C7X_STATUS_SUCCESS;
    gJobsCompleted++;

done:
    /* Response size depends on number of outputs */
    resp->hdr.len = (uint32_t)(sizeof(struct c7x_msg_infer_resp) +
                    (resp->num_outputs > 1 ? (resp->num_outputs - 1) * sizeof(struct c7x_tensor_desc) : 0));

    send_response(C7X_MSG_INFER_RESP, req->hdr.seq,
                  resp->hdr.len, srcCore, srcEndpt);
}

/*
 * =============================================================================
 * Service Loop (runs in caller's task context)
 * =============================================================================
 */

void compute_service_run(void)
{
    int32_t status;
    uint16_t recvMsgSize;
    uint16_t srcCore;
    uint32_t srcEndpt;
    struct c7x_msg_hdr *hdr;

    DebugP_log("[COMPUTE] Service loop started, endpoint %u\r\n", gLocalEndpoint);

    while (gServiceRunning) {
        /* Wait for message (timeout allows checking gServiceRunning flag) */
        recvMsgSize = sizeof(gRecvBuf);
        status = RPMessage_recv(&gRpmsgObj,
                                gRecvBuf, &recvMsgSize,
                                &srcCore, &srcEndpt,
                                ClockP_usecToTicks(30000000)); /* 30s timeout */

        if (status == SystemP_TIMEOUT) {
            /* Periodic status while waiting */
            DebugP_log("[COMPUTE] Waiting for messages... (jobs: %u ok, %u fail)\r\n",
                       gJobsCompleted, gJobsFailed);
            continue;
        }
        if (status != SystemP_SUCCESS) {
            DebugP_log("[COMPUTE] RPMessage_recv failed: %d\r\n", status);
            continue;
        }

        if (recvMsgSize < sizeof(struct c7x_msg_hdr)) {
            DebugP_log("[COMPUTE] Message too small: %u bytes\r\n", recvMsgSize);
            continue;
        }

        /* Parse header and dispatch */
        hdr = (struct c7x_msg_hdr *)gRecvBuf;

        switch (hdr->type) {
        case C7X_MSG_PING:
            handle_ping((struct c7x_msg_ping *)hdr, srcCore, srcEndpt);
            break;

        case C7X_MSG_GET_STATUS:
            handle_get_status((struct c7x_msg_get_status *)hdr, srcCore, srcEndpt);
            break;

        case C7X_MSG_DYN_LOAD:
            if (recvMsgSize >= sizeof(struct c7x_msg_dyn_load)) {
                handle_dyn_load((struct c7x_msg_dyn_load *)hdr, srcCore, srcEndpt);
            } else {
                DebugP_log("[COMPUTE] DYN_LOAD message too small\r\n");
                goto send_error;
            }
            break;

        case C7X_MSG_DYN_UNLOAD:
            if (recvMsgSize >= sizeof(struct c7x_msg_dyn_unload)) {
                handle_dyn_unload((struct c7x_msg_dyn_unload *)hdr, srcCore, srcEndpt);
            } else {
                DebugP_log("[COMPUTE] DYN_UNLOAD message too small\r\n");
                goto send_error;
            }
            break;

        case C7X_MSG_MODEL_LOAD:
            if (recvMsgSize >= sizeof(struct c7x_msg_model_load)) {
                handle_model_load((struct c7x_msg_model_load *)hdr, srcCore, srcEndpt);
            } else {
                DebugP_log("[COMPUTE] MODEL_LOAD message too small\r\n");
                goto send_error;
            }
            break;

        case C7X_MSG_MODEL_UNLOAD:
            if (recvMsgSize >= sizeof(struct c7x_msg_model_unload)) {
                handle_model_unload((struct c7x_msg_model_unload *)hdr, srcCore, srcEndpt);
            } else {
                DebugP_log("[COMPUTE] MODEL_UNLOAD message too small\r\n");
                goto send_error;
            }
            break;

        case C7X_MSG_INFER:
            if (recvMsgSize >= sizeof(struct c7x_msg_infer)) {
                handle_infer((struct c7x_msg_infer *)hdr, recvMsgSize,
                             srcCore, srcEndpt);
            } else {
                DebugP_log("[COMPUTE] INFER message too small\r\n");
                goto send_error;
            }
            break;

        default:
            DebugP_log("[COMPUTE] Unknown message type: 0x%04x\r\n", hdr->type);
            goto send_error;
        }
        continue;

    send_error:
        /* Always send an error response so the host does not hang */
        {
            struct c7x_msg_hdr *err_resp = (struct c7x_msg_hdr *)gSendBuf;
            err_resp->type = hdr->type | 0x1000; /* response bit */
            err_resp->status = C7X_STATUS_ERR_INVALID;
            err_resp->seq = hdr->seq;
            err_resp->len = (uint32_t)sizeof(struct c7x_msg_hdr);
            RPMessage_send(err_resp, err_resp->len,
                           srcCore, srcEndpt,
                           RPMessage_getLocalEndPt(&gRpmsgObj),
                           SystemP_WAIT_FOREVER);
        }
    }

    DebugP_log("[COMPUTE] Service loop exiting\r\n");
}

/*
 * =============================================================================
 * Public API
 * =============================================================================
 */

int32_t compute_service_init(void)
{
    int32_t status;
    RPMessage_CreateParams rpmsgParams;

    DebugP_log("[COMPUTE] Initializing compute service...\r\n");

    /* Record start time */
    gStartTimeUs = ClockP_getTimeUsec();

    /* Create RPMessage endpoint */
    RPMessage_CreateParams_init(&rpmsgParams);
    rpmsgParams.localEndPt = C7X_SERVICE_ENDPOINT;

    status = RPMessage_construct(&gRpmsgObj, &rpmsgParams);
    if (status != SystemP_SUCCESS) {
        DebugP_log("[COMPUTE] Failed to create RPMessage endpoint: %d\r\n", status);
        return status;
    }

    gLocalEndpoint = RPMessage_getLocalEndPt(&gRpmsgObj);
    DebugP_log("[COMPUTE] RPMessage endpoint created: %u\r\n", gLocalEndpoint);

    /* Announce service to Linux - creates rpmsg channel for rpmsg_char access */
    status = RPMessage_announce(CSL_CORE_ID_A53SS0_0, gLocalEndpoint, "rpmsg_chrdev");
    if (status == SystemP_SUCCESS) {
        DebugP_log("[COMPUTE] Announced rpmsg_chrdev on endpoint %u\r\n", gLocalEndpoint);
    } else {
        DebugP_log("[COMPUTE] WARNING: Failed to announce rpmsg_chrdev: %d\r\n", status);
    }

    /* Initialize dynamic loader */
    status = dyn_loader_init();
    if (status != 0) {
        DebugP_log("[COMPUTE] WARNING: Dynamic loader init failed: %d\r\n", status);
        /* Continue anyway - dynamic loading won't work but ping/process will */
    }

    /* Initialize TVM model manager */
    status = tvm_model_init();
    if (status != 0) {
        DebugP_log("[COMPUTE] WARNING: TVM model manager init failed: %d\r\n", status);
    }

    /* Initialize CLEC event routing for DRU (events 128-143 -> C7x 32-47)
     * and the shared UDMA driver.  Done once at boot; the driver is shared
     * between TVM DMA tiling and TIDL's MMA/DMA operations. */
    {
        CSL_ClecEventConfig cfgClec;
        CSL_CLEC_EVTRegs *clecBaseAddr = (CSL_CLEC_EVTRegs *)CSL_C7X256V0_CLEC_BASE;
        uint32_t i;
        for (i = 128; i < 144; i++) {
            cfgClec.secureClaimEnable = FALSE;
            cfgClec.evtSendEnable = TRUE;
            cfgClec.rtMap = CSL_CLEC_RTMAP_CPU_ALL;
            cfgClec.extEvtNum = 0;
            cfgClec.c7xEvtNum = (i - 128) + 32;
            CSL_clecConfigEvent(clecBaseAddr, i, &cfgClec);
        }
        tvm_dsp_dma_init(1);
    }

    /* Initialize shared memory printf device */
    shm_printf_init((void *)(uintptr_t)C7X_PRINTF_BUF_ADDR,
                    (uint32_t)C7X_PRINTF_BUF_SIZE);

    /* Mark service as running - service loop runs in caller's task context */
    gServiceRunning = 1;

    DebugP_log("[COMPUTE] Service initialized successfully\r\n");
    return SystemP_SUCCESS;
}

void compute_service_stop(void)
{
    gServiceRunning = 0;
    /* Unblock RPMessage_recv so the service loop exits immediately
     * instead of waiting up to 30s for the next timeout. */
    RPMessage_unblock(&gRpmsgObj);
}

void compute_service_deinit(void)
{
    DebugP_log("[COMPUTE] Deinitializing compute service...\r\n");

    gServiceRunning = 0;

    tvm_model_deinit();
    dyn_loader_deinit();

    RPMessage_destruct(&gRpmsgObj);

    DebugP_log("[COMPUTE] Service deinitialized\r\n");
}

void compute_service_get_stats(uint32_t *jobs_completed,
                                uint32_t *jobs_failed,
                                uint32_t *uptime_ms)
{
    if (jobs_completed) {
        *jobs_completed = gJobsCompleted;
    }
    if (jobs_failed) {
        *jobs_failed = gJobsFailed;
    }
    if (uptime_ms) {
        *uptime_ms = (uint32_t)((ClockP_getTimeUsec() - gStartTimeUs) / 1000);
    }
}
