/*
 * C7x Compute Service - Shared Protocol Definitions
 *
 * This header is shared between ARM host and C7x DSP.
 * It defines the message protocol for host-DSP communication.
 */

#ifndef C7X_COMPUTE_PROTOCOL_H
#define C7X_COMPUTE_PROTOCOL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =============================================================================
 * Service Configuration
 * =============================================================================
 */

#define C7X_SERVICE_NAME        "c7x-compute"
#define C7X_SERVICE_ENDPOINT    20

/* Maximum message size (must fit in RPMessage buffer) */
#define C7X_MAX_MSG_SIZE        512

/*
 * =============================================================================
 * Shared Memory Configuration
 * =============================================================================
 */

/*
 * Shared DDR region for data transfer.
 *
 * Uses the vision_apps_shared-memories DMA heap carveout:
 *   Physical:  0x900000000 (512 MB, dma-heap-carveout in device tree)
 *   DSP vaddr: 0xC0000000  (MMU region: 0xC0000000 -> 0x900000000, 512 MB)
 *   Host:      allocated via /dev/dma_heap/carveout_vision_apps_shared-memories
 */
#define C7X_SHARED_BASE         0xC0000000ULL   /* DSP virtual address */
#define C7X_SHARED_PHYS_BASE    0x900000000ULL  /* Physical address (host DMA heap) */
#define C7X_SHARED_SIZE         0x20000000ULL   /* 512 MB total — full DMA heap carveout */

/* Staging buffer: first 504 MB of the shared DDR carveout.
 * Used for host-to-DSP data transfer: ELF modules (DLOAD), weights
 * (MODEL_LOAD), and inference input tensors (INFER).
 *
 * The address is the DSP virtual address from the static MMU mapping:
 *   Physical 0x900000000 -> DSP virtual 0xC0000000
 * The host side uses mmap'd userspace pointers (client->staging_buf). */
#define C7X_STAGING_ADDR   0xC0000000ULL
#define C7X_STAGING_SIZE   0x1F800000ULL   /* 504 MB */

/* Result buffer: last 8 MB (DSP-to-host: inference output + printf) */
#define C7X_RESULT_ADDR  0xDF800000ULL
#define C7X_RESULT_SIZE  0x00800000ULL   /* 8 MB */

/* Printf buffer: last 64 KB of result buffer */
#define C7X_PRINTF_BUF_SIZE     0x00010000ULL   /* 64 KB */
#define C7X_PRINTF_BUF_ADDR     (C7X_RESULT_ADDR + \
        C7X_RESULT_SIZE - C7X_PRINTF_BUF_SIZE)

/*
 * =============================================================================
 * Message Types
 * =============================================================================
 */

/* Request messages (Host → DSP) */
#define C7X_MSG_PING            0x0001  /* Connectivity test */
#define C7X_MSG_GET_STATUS      0x0003  /* Get service status */

/* Response messages (DSP → Host) */
#define C7X_MSG_PING_RESP       0x1001
#define C7X_MSG_STATUS_RESP     0x1003

/* Dynamic loading (Host → DSP) */
#define C7X_MSG_DYN_LOAD        0x0010  /* Load ELF shared object */
#define C7X_MSG_DYN_LOAD_RESP   0x1010
#define C7X_MSG_DYN_UNLOAD      0x0012  /* Unload shared object */
#define C7X_MSG_DYN_UNLOAD_RESP 0x1012

/* TVM model operations (Host → DSP) */
#define C7X_MSG_MODEL_LOAD      0x0020  /* Load weights/constants */
#define C7X_MSG_MODEL_LOAD_RESP 0x1020
#define C7X_MSG_INFER           0x0021  /* Run inference */
#define C7X_MSG_INFER_RESP      0x1021
#define C7X_MSG_MODEL_UNLOAD      0x0022  /* Unload model weights */
#define C7X_MSG_MODEL_UNLOAD_RESP 0x1022

/*
 * =============================================================================
 * Status Codes
 * =============================================================================
 */

#define C7X_STATUS_SUCCESS      0
#define C7X_STATUS_ERR_GENERIC  -1
#define C7X_STATUS_ERR_INVALID  -2      /* Invalid parameter */
#define C7X_STATUS_ERR_NOMEM    -3      /* Out of memory */
#define C7X_STATUS_ERR_BUSY     -4      /* Service busy */
#define C7X_STATUS_ERR_TIMEOUT  -5      /* Operation timeout */
#define C7X_STATUS_ERR_ADDR     -6      /* Invalid address */
#define C7X_STATUS_ERR_SIZE     -7      /* Invalid size */
#define C7X_STATUS_ERR_OP       -8      /* Unknown operation */
#define C7X_STATUS_ERR_LOAD     -9      /* Dynamic load failed */
#define C7X_STATUS_ERR_SYMBOL   -10     /* Symbol lookup failed */
#define C7X_STATUS_ERR_CALL     -11     /* Function call failed */
#define C7X_STATUS_ERR_HANDLE   -12     /* Invalid handle/id */
#define C7X_STATUS_ERR_WEIGHTS  -13     /* Weights parsing failed */
#define C7X_STATUS_ERR_TENSOR   -14     /* Tensor construction failed */

/*
 * =============================================================================
 * Message Structures
 * =============================================================================
 */

/*
 * Common message header (16 bytes)
 * All messages start with this header.
 */
struct c7x_msg_hdr {
    uint32_t type;              /* Message type (C7X_MSG_*) */
    uint32_t seq;               /* Sequence number for correlation */
    uint32_t len;               /* Total message length including header */
    int32_t  status;            /* Response status (0 = success) */
} __attribute__((packed));

/*
 * PING request (16 bytes)
 * Simple connectivity test.
 */
struct c7x_msg_ping {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_PING */
} __attribute__((packed));

/*
 * PING response (24 bytes)
 */
struct c7x_msg_ping_resp {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_PING_RESP */
    uint32_t version;           /* Service version */
    uint32_t uptime_ms;         /* Uptime in milliseconds */
} __attribute__((packed));

/*
 * STATUS request (16 bytes)
 */
struct c7x_msg_get_status {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_GET_STATUS */
} __attribute__((packed));

/*
 * STATUS response (32 bytes)
 */
struct c7x_msg_status_resp {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_STATUS_RESP */
    uint32_t version;           /* Service version */
    uint32_t uptime_ms;         /* Uptime in milliseconds */
    uint32_t jobs_completed;    /* Total jobs processed */
    uint32_t jobs_failed;       /* Total jobs failed */
} __attribute__((packed));

/*
 * =============================================================================
 * Dynamic Loading Message Structures
 * =============================================================================
 */

/*
 * DYN_LOAD request (32 bytes)
 * Load an ELF shared object pre-staged at input buffer.
 */
struct c7x_msg_dyn_load {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_DYN_LOAD */
    uint32_t elf_size;          /* Size of ELF data in input buffer */
    uint32_t flags;             /* Reserved flags */
    uint32_t reserved[2];
} __attribute__((packed));

/*
 * DYN_LOAD response (32 bytes)
 */
struct c7x_msg_dyn_load_resp {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_DYN_LOAD_RESP */
    uint32_t module_handle;     /* Handle for loaded module */
    uint32_t text_size;         /* Size of code segments */
    uint32_t data_size;         /* Size of data segments */
    uint32_t reserved;
} __attribute__((packed));

/*
 * DYN_UNLOAD request (20 bytes)
 */
struct c7x_msg_dyn_unload {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_DYN_UNLOAD */
    uint32_t module_handle;     /* Handle from DYN_LOAD_RESP */
} __attribute__((packed));

/*
 * DYN_UNLOAD response (16 bytes)
 */
struct c7x_msg_dyn_unload_resp {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_DYN_UNLOAD_RESP */
} __attribute__((packed));

/*
 * =============================================================================
 * TVM Model Message Structures
 * =============================================================================
 */

/*
 * MODEL_LOAD request (32 bytes)
 * Load weights.bin pre-staged at input buffer.
 */
struct c7x_msg_model_load {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_MODEL_LOAD */
    uint32_t weights_size;      /* Size of weights data in input buffer */
    uint32_t flags;             /* Reserved flags */
    uint32_t reserved[2];
} __attribute__((packed));

/*
 * MODEL_LOAD response (32 bytes)
 */
struct c7x_msg_model_load_resp {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_MODEL_LOAD_RESP */
    uint32_t model_id;          /* Model ID for future reference */
    uint32_t num_constants;     /* Number of parsed constants */
    uint32_t reserved[2];
} __attribute__((packed));

/*
 * MODEL_UNLOAD request (20 bytes)
 */
struct c7x_msg_model_unload {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_MODEL_UNLOAD */
    uint32_t model_id;          /* Model ID from MODEL_LOAD_RESP */
} __attribute__((packed));

/*
 * MODEL_UNLOAD response (16 bytes)
 */
struct c7x_msg_model_unload_resp {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_MODEL_UNLOAD_RESP */
} __attribute__((packed));

/*
 * Tensor descriptor for INFER messages (80 bytes)
 */
#define C7X_TENSOR_MAX_NDIM     6

struct c7x_tensor_desc {
    uint64_t data_addr;         /* Physical address in shared DDR */
    uint64_t data_size;         /* Size in bytes */
    int32_t  ndim;              /* Number of dimensions */
    int32_t  dtype_code;        /* DLDataType code (kDLFloat=2, kDLInt=0, etc.) */
    int32_t  dtype_bits;        /* Bits per element */
    int32_t  reserved;
    int64_t  shape[C7X_TENSOR_MAX_NDIM]; /* Up to 6 dimensions */
} __attribute__((packed));

/*
 * INFER request (variable size, fits in C7X_MAX_MSG_SIZE for 1 input)
 *
 * flags field:
 *   bits [15:0] = repeat count.  The firmware loops cg_main_dsp() this
 *                 many times, recording per-iteration cycles and printing
 *                 layer profiles.  0 or 1 = run once (backward compatible).
 *                 Use repeat=2 to separate one-time init from steady-state.
 *   bits [31:16] = reserved (must be 0).
 */
struct c7x_msg_infer {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_INFER */
    uint32_t module_handle;     /* Loaded module handle */
    uint32_t model_id;          /* Loaded weights model ID */
    uint32_t num_inputs;        /* Number of input tensors */
    uint32_t flags;             /* See above: bits[15:0]=repeat count */
    struct c7x_tensor_desc inputs[1]; /* Variable-length array */
} __attribute__((packed));

/*
 * INFER response (variable size)
 */
struct c7x_msg_infer_resp {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_INFER_RESP */
    int32_t  return_value;      /* Return value from cg_main_dsp */
    uint64_t cycles;            /* DSP cycles consumed (64-bit TSC) */
    uint32_t num_outputs;       /* Number of output tensors */
    uint32_t printf_size;       /* Bytes of printf data in SHM buffer */
    struct c7x_tensor_desc outputs[1]; /* Variable-length array */
} __attribute__((packed));

/*
 * Union of all message types for buffer allocation
 */
union c7x_msg {
    struct c7x_msg_hdr              hdr;
    struct c7x_msg_ping             ping;
    struct c7x_msg_ping_resp        ping_resp;
    struct c7x_msg_get_status       get_status;
    struct c7x_msg_status_resp      status_resp;
    struct c7x_msg_dyn_load         dyn_load;
    struct c7x_msg_dyn_load_resp    dyn_load_resp;
    struct c7x_msg_dyn_unload       dyn_unload;
    struct c7x_msg_dyn_unload_resp  dyn_unload_resp;
    struct c7x_msg_model_load       model_load;
    struct c7x_msg_model_load_resp  model_load_resp;
    struct c7x_msg_model_unload     model_unload;
    struct c7x_msg_model_unload_resp model_unload_resp;
    struct c7x_msg_infer            infer;
    struct c7x_msg_infer_resp       infer_resp;
    uint8_t                         raw[C7X_MAX_MSG_SIZE];
};

/*
 * =============================================================================
 * Helper Macros
 * =============================================================================
 */

/* Check if address+size is within shared buffer region (overflow-safe) */
#define C7X_IS_VALID_STAGING_ADDR(addr, size) \
    ((size) <= C7X_STAGING_SIZE && \
     (addr) >= C7X_STAGING_ADDR && \
     ((addr) - C7X_STAGING_ADDR) <= (C7X_STAGING_SIZE - (size)))

#define C7X_IS_VALID_RESULT_ADDR(addr, size) \
    ((size) <= C7X_RESULT_SIZE && \
     (addr) >= C7X_RESULT_ADDR && \
     ((addr) - C7X_RESULT_ADDR) <= (C7X_RESULT_SIZE - (size)))

/* Service version: major.minor.patch encoded as 0xMMmmpp */
#define C7X_SERVICE_VERSION     0x020000  /* v2.0.0 */

/* Extract version components */
#define C7X_VERSION_MAJOR(v) (((v) >> 16) & 0xFF)
#define C7X_VERSION_MINOR(v) (((v) >>  8) & 0xFF)
#define C7X_VERSION_PATCH(v) ( (v)        & 0xFF)

#ifdef __cplusplus
}
#endif

#endif /* C7X_COMPUTE_PROTOCOL_H */
