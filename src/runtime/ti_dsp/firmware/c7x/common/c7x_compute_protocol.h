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
/* Physical address (host DMA heap).  Overridable via -D so a single
 * board/ddr choice (see cmake/boards.cmake) can retarget it: 8gb boards
 * keep 0x900000000, 4gb boards (e.g. BeagleY-AI) use 0x8a0000000. */
#ifndef C7X_SHARED_PHYS_BASE
#define C7X_SHARED_PHYS_BASE    0x900000000ULL  /* Physical address (host DMA heap) */
#endif
#define C7X_SHARED_SIZE         0x20000000ULL   /* 512 MB total — full DMA heap carveout */

/* Staging region: first 468 MB of the shared DDR carveout.
 * Used for host-to-DSP data transfer: ELF modules (DLOAD) and weights
 * (MODEL_LOAD) only -- inference input tensors go to the separate,
 * dynamically-sized input_buf dmabuf instead (client->input_buf).
 *
 * The address is the DSP virtual address from the static MMU mapping:
 *   Physical 0x900000000 -> DSP virtual 0xC0000000
 * The host allocates staging+KV as one dmabuf first so first-fit packs it
 * at the pool base and this address stays valid (client->staging). */
#define C7X_STAGING_ADDR   0xC0000000ULL
#define C7X_STAGING_SIZE   0x1D400000ULL   /* 468 MB */

/* KV cache fixed region: 12 MB right after staging.
 * Persists across inferences — not touched by watermark restore.
 * Used when C7X_INFER_FLAG_KV_RESIDENT is set: the DSP copies KV output
 * tensors here after cg_main_dsp() and reads them back as inputs on the
 * next inference without host involvement.
 *
 * Reserved by the host allocating one dmabuf covering
 * [C7X_STAGING_ADDR, C7X_KV_ADDR + C7X_KV_SIZE) first, so first-fit packs
 * it at the pool's low end and this fixed address stays valid without the
 * host needing to steer the allocator (gen_pool has no such API). */
#define C7X_KV_ADDR      0xDD400000ULL   /* staging + 468 MB */
#define C7X_KV_SIZE      0x00C00000ULL   /* 12 MB */
#define C7X_KV_NUM_TENSORS   60
#define C7X_KV_TENSOR_SIZE   196608      /* 1*3*256*64*sizeof(float) */

/* Inference input/output buffers (input_buf/output_buf) are separate,
 * per-session dmabufs sized from tvm_dsp_io_meta and communicated per-INFER
 * via c7x_msg_infer's input_dsp_addr/input_size/result_dsp_addr/result_size
 * -- there is no fixed address/size for either anymore. The DSP tracks the
 * current output_buf as g_result_dsp_addr/g_result_size (compute_service.c);
 * input_buf needs no such global since every validation site already has
 * the request in hand.
 *
 * Printf buffer: 64 KB, communicated once via C7X_MSG_SET_PRINTF_BUF right
 * after c7x_client_open(). Before that first rebind, shm_printf_init() at
 * boot targets a scratch address at the top of the whole shared carveout
 * (nothing else claims that address until the host starts allocating). */
#define C7X_PRINTF_BUF_SIZE       0x00010000ULL   /* 64 KB */
#define C7X_PRINTF_BUF_BOOT_ADDR  (C7X_SHARED_BASE + C7X_SHARED_SIZE - C7X_PRINTF_BUF_SIZE)

/* struct shm_printf_hdr's on-wire size (shm_printf.c): magic/wr_index/
 * buf_size/reserved, 4 uint32_t. Shared here so the host's printf read
 * (c7x_compute_client.cpp) doesn't duplicate this as a magic number. */
#define C7X_SHM_PRINTF_HDR_SIZE  16U

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
#define C7X_MSG_INFER           0x0021  /* Run inference (up to 4 inline inputs) */
#define C7X_MSG_INFER_RESP      0x1021
#define C7X_MSG_MODEL_UNLOAD      0x0022  /* Unload model weights */
#define C7X_MSG_MODEL_UNLOAD_RESP 0x1022
/* INFER_LARGE: run inference with many inputs.  Tensor descriptors are
 * stored in the staging buffer (descs_addr/descs_size) instead of inline
 * in the IPC message, avoiding the 512-byte rpmsg size limit.  The response
 * message type is the same C7X_MSG_INFER_RESP. */
#define C7X_MSG_INFER_LARGE     0x0023

/* Sent once right after c7x_client_open(), before any DYN_LOAD -- rebinds
 * DSP printf from its boot-time scratch address to the host's actual
 * printf_buf dmabuf (see shm_printf_rebind()). */
#define C7X_MSG_SET_PRINTF_BUF      0x0030
#define C7X_MSG_SET_PRINTF_BUF_RESP 0x1030

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
 * SET_PRINTF_BUF request (32 bytes)
 * Rebind DSP printf from its boot-time scratch address (shm_printf_init(),
 * C7X_PRINTF_BUF_BOOT_ADDR) to the host's actual printf_buf dmabuf. Sent
 * once, right after c7x_client_open(), before any DYN_LOAD.
 */
struct c7x_msg_set_printf_buf {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_SET_PRINTF_BUF */
    uint64_t printf_dsp_addr;
    uint32_t printf_size;
    uint32_t reserved;
} __attribute__((packed));

/*
 * SET_PRINTF_BUF response (16 bytes)
 */
struct c7x_msg_set_printf_buf_resp {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_SET_PRINTF_BUF_RESP */
} __attribute__((packed));

/*
 * =============================================================================
 * Dynamic Loading Message Structures
 * =============================================================================
 */

/*
 * Compile-time IO metadata, emitted by the Python export path
 * (tvm.contrib.c7x.io_meta) and embedded in lib0.out next to weights.bin.
 * Looked up in handle_dyn_load() via
 * dyn_loader_query_symbol(handle, "_binary_tvm_dsp_io_meta_start", ...),
 * the same mechanism already used for _binary_weights_bin_start. Declares
 * the input_buf/output_buf capacity to report in DYN_LOAD_RESP; a missing
 * symbol means "no metadata" (io_* fields all zero), not an error -- older
 * modules and ones with symbolic entry shapes take this path.
 */
#define TVM_DSP_IO_META_MAGIC             0x4D4F4954U /* "TIOM", little-endian */
#define TVM_DSP_IO_META_VERSION           1U
#define TVM_DSP_IO_META_FLAG_SIZES_EXACT  (1U << 0)

struct tvm_dsp_io_meta {
    uint32_t magic;         /* TVM_DSP_IO_META_MAGIC */
    uint32_t version;       /* TVM_DSP_IO_META_VERSION */
    uint32_t num_inputs;
    uint32_t num_outputs;
    uint64_t input_bytes;   /* input_buf capacity, incl. descriptor region */
    uint64_t output_bytes;  /* output_buf capacity */
    uint32_t flags;         /* bit0 = sizes exact; clear = upper bound only */
    uint32_t reserved;
} __attribute__((packed));

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
 * DYN_LOAD response (68 bytes)
 */
struct c7x_msg_dyn_load_resp {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_DYN_LOAD_RESP */
    uint32_t module_handle;     /* Handle for loaded module */
    uint32_t text_size;         /* Size of code segments */
    uint32_t data_size;         /* Size of data segments */
    uint32_t oom_requested;     /* status==ERR_NOMEM: bytes requested by the
                                 * failing MAIN-pool alloc. 0 otherwise. */
    uint32_t oom_free;          /* status==ERR_NOMEM: pool free bytes at the
                                 * moment of failure. 0 otherwise. */
    uint32_t oom_total;         /* status==ERR_NOMEM: total pool bytes.
                                 * 0 otherwise. */
    uint64_t io_input_bytes;    /* From tvm_dsp_io_meta; 0 if absent */
    uint64_t io_output_bytes;   /* From tvm_dsp_io_meta; 0 if absent */
    uint32_t io_num_inputs;
    uint32_t io_num_outputs;
    uint32_t io_flags;
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
 * MODEL_LOAD response (36 bytes)
 */
struct c7x_msg_model_load_resp {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_MODEL_LOAD_RESP */
    uint32_t model_id;          /* Model ID for future reference */
    uint32_t num_constants;     /* Number of parsed constants */
    uint32_t oom_requested;     /* status==ERR_NOMEM: bytes requested by the
                                 * failing MAIN-pool alloc. 0 otherwise. */
    uint32_t oom_free;          /* status==ERR_NOMEM: pool free bytes at the
                                 * moment of failure. 0 otherwise. */
    uint32_t oom_total;         /* status==ERR_NOMEM: total pool bytes.
                                 * 0 otherwise. */
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
 *   bits [15:0]  = repeat count.  The firmware loops cg_main_dsp() this
 *                  many times, recording per-iteration cycles and printing
 *                  layer profiles.  0 or 1 = run once (backward compatible).
 *                  Use repeat=2 to separate one-time init from steady-state.
 *   bit  [16]    = C7X_INFER_FLAG_KV_RESIDENT: DSP copies KV outputs to
 *                  fixed region and returns only logits.
 *   bits [31:17] = reserved (must be 0).
 */
struct c7x_msg_infer {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_INFER */
    uint32_t module_handle;     /* Loaded module handle */
    uint32_t model_id;          /* Loaded weights model ID */
    uint32_t num_inputs;        /* Number of input tensors */
    uint32_t flags;             /* See above: bits[15:0]=repeat count */
    uint64_t input_dsp_addr;    /* input_buf base, for C7X_IS_VALID_INPUT_ADDR */
    uint64_t input_size;        /* input_buf declared capacity */
    uint64_t result_dsp_addr;   /* output_buf base -- mandatory (D7) */
    uint64_t result_size;       /* output_buf declared capacity -- mandatory */
    struct c7x_tensor_desc inputs[1]; /* Variable-length array */
} __attribute__((packed));

/*
 * INFER_LARGE request (80 bytes).
 *
 * Used when the number of input tensors is too large to fit the tensor
 * descriptors inline in the 512-byte rpmsg buffer.  The host writes the
 * full c7x_tensor_desc array into the staging buffer at descs_addr, then
 * sends this compact message.  The DSP reads the descriptors from staging
 * DDR before processing.
 *
 * Input tensor data is placed in staging DDR immediately after the
 * descriptors array (i.e. at descs_addr + descs_size, aligned to 64 bytes).
 * The descs[i].data_addr fields are already set to the correct DDR addresses
 * by the host before sending this message.
 */
struct c7x_msg_infer_large {
    struct c7x_msg_hdr hdr;         /* type = C7X_MSG_INFER_LARGE */
    uint32_t module_handle;         /* Loaded module handle */
    uint32_t model_id;              /* Loaded weights model ID */
    uint32_t num_inputs;            /* Number of input tensors */
    uint32_t flags;                 /* Same flags as c7x_msg_infer */
    uint64_t descs_addr;            /* Staging DDR address of descriptor array */
    uint32_t descs_size;            /* Size in bytes of descriptor array */
    uint32_t reserved;
    uint64_t input_dsp_addr;        /* Same meaning as c7x_msg_infer's */
    uint64_t input_size;
    uint64_t result_dsp_addr;
    uint64_t result_size;
} __attribute__((packed));

/* INFER flags */
#define C7X_INFER_FLAG_KV_RESIDENT  (1U << 16)

/*
 * INFER response (variable size)
 */
/*
 * INFER response.  Two layouts depending on output count:
 *
 * Small (fits in 512-byte rpmsg): tensor descriptors are inline in
 * outputs[].  descs_addr == 0.
 *
 * Large (too many outputs for 512-byte rpmsg, e.g. KV cache models with
 * 61 outputs): the full c7x_tensor_desc array is written into the result
 * buffer at descs_addr.  outputs[0] is unused; host reads descriptors
 * from the result buffer DDR address instead.
 */
struct c7x_msg_infer_resp {
    struct c7x_msg_hdr hdr;     /* type = C7X_MSG_INFER_RESP */
    int32_t  return_value;      /* Return value from cg_main_dsp */
    uint64_t cycles;            /* DSP cycles consumed (64-bit TSC) */
    uint32_t num_outputs;       /* Number of output tensors */
    uint32_t printf_size;       /* Bytes of printf data in SHM buffer */
    uint64_t descs_addr;        /* Non-zero: descriptor array is in result
                                 * buffer at this DSP address (DDR), not
                                 * inline in outputs[]. */
    uint32_t descs_size;        /* Byte size of the out-of-band desc array */
    uint32_t oom_requested;     /* status==ERR_NOMEM: bytes requested by the
                                 * failing MAIN-pool alloc. 0 otherwise. */
    uint32_t oom_free;          /* status==ERR_NOMEM: pool free bytes at the
                                 * moment of failure. 0 otherwise. */
    uint32_t oom_total;         /* status==ERR_NOMEM: total pool bytes.
                                 * 0 otherwise. */
    uint64_t result_required;   /* status==ERR_SIZE: output_buf capacity
                                 * that would have fit the tensor that
                                 * overflowed it. 0 otherwise. */
    struct c7x_tensor_desc outputs[1]; /* Inline (descs_addr == 0) */
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
    struct c7x_msg_infer_large      infer_large;
    struct c7x_msg_infer_resp       infer_resp;
    uint8_t                         raw[C7X_MAX_MSG_SIZE];
};

/*
 * =============================================================================
 * Helper Macros
 * =============================================================================
 */

/* Does [addr, addr+size) fall entirely within [base, base+bound)?
 * (overflow-safe). input_buf/output_buf have no fixed base/size of their
 * own anymore -- each message carries its sender's actual buffer base+size
 * (D12), so validation takes them as explicit parameters rather than
 * reading fixed C7X_* constants. */
#define C7X_IS_VALID_RANGE(addr, size, base, bound) \
    ((size) <= (bound) && \
     (addr) >= (base) && \
     ((addr) - (base)) <= ((bound) - (size)))

#define C7X_IS_VALID_KV_ADDR(addr, size) \
    C7X_IS_VALID_RANGE(addr, size, C7X_KV_ADDR, C7X_KV_SIZE)

/* input_base/input_bound: the sender's input_dsp_addr/input_size. */
#define C7X_IS_VALID_INPUT_ADDR(addr, size, input_base, input_bound) \
    (C7X_IS_VALID_RANGE(addr, size, input_base, input_bound) || \
     C7X_IS_VALID_KV_ADDR(addr, size))

/* Service version: major.minor.patch encoded as 0xMMmmpp.
 *
 * Bump the major on any incompatible wire change -- a struct changing size,
 * or a message becoming mandatory.  v3.0.0 covers the per-buffer dmabuf
 * protocol: c7x_msg_infer, c7x_msg_infer_large, c7x_msg_dyn_load_resp and
 * c7x_msg_infer_resp all grew, and SET_PRINTF_BUF became mandatory at open.
 * A mismatched pair otherwise fails deep inside open or infer with an error
 * that reads as a bug rather than as a stale firmware image. */
#define C7X_SERVICE_VERSION     0x030000  /* v3.0.0 */

/* Extract version components */
#define C7X_VERSION_MAJOR(v) (((v) >> 16) & 0xFF)
#define C7X_VERSION_MINOR(v) (((v) >>  8) & 0xFF)
#define C7X_VERSION_PATCH(v) ( (v)        & 0xFF)

#ifdef __cplusplus
}
#endif

#endif /* C7X_COMPUTE_PROTOCOL_H */
