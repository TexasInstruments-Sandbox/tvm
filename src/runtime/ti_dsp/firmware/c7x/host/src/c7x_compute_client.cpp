/*
 * C7x Compute Service - Host Client Library Implementation
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <memory>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/dma-buf.h>
#include <linux/dma-heap.h>

#include "c7x_compute_client.h"
#include "c7x_compute_protocol.h"
#include "rpmsg_wrapper.h"
#include "raii.h"

/*
 * =============================================================================
 * DMA-buf to physical address conversion (TI remoteproc extension)
 * See: ti-processor-sdk-rtos/app_utils/utils/mem/include/linux/dma_buf_phys.h
 * =============================================================================
 */

struct dma_buf_phys_data {
    __u32 fd;
    __u64 phys;
};

#define RPROC_MAGIC             0xB7
#define RPROC_IOC_DMA_BUF_ATTACH _IOWR(RPROC_MAGIC, 0, struct dma_buf_phys_data)

/* DMA heap device for the shared memory carveout */
#define DMA_HEAP_DEVICE  "/dev/dma_heap/carveout_vision_apps_shared-memories"

/* Device tree address for C7x DSP (stable across reboots) */
#define C7X_DEVICE_ADDR  "7e000000.dsp"

/*
 * Find the remoteproc index for a given DSP by matching the device tree
 * address in sysfs.  Reused from c7x_compute_cli.cpp pattern.
 */
static int find_remoteproc_index(const char *device_addr)
{
    char path[256], link[512];
    for (int i = 0; i < 16; i++) {
        snprintf(path, sizeof(path),
                 "/sys/class/remoteproc/remoteproc%d/device", i);
        ssize_t len = readlink(path, link, sizeof(link) - 1);
        if (len < 0) continue;
        link[len] = '\0';
        if (strstr(link, device_addr))
            return i;
    }
    return -1;
}

/*
 * =============================================================================
 * Client State
 * =============================================================================
 */

struct c7x_client {
    UniqueFd rpmsg_fd;          /* RPMessage file descriptor */
    UniqueFd dma_heap_fd;       /* DMA heap device fd */
    UniqueFd dma_buf_fd;        /* dmabuf fd for shared memory allocation */
    UniqueFd rproc_fd;          /* /dev/remoteproc0 fd -- must stay open to keep
                                   the dmabuf device attachment alive for DSP */
    MmapRegion shared_map;      /* mmap'd shared buffer (input + output) */
    void *input_buf = nullptr;  /* = shared_map.get() */
    void *output_buf = nullptr; /* = shared_map.get() + C7X_INPUT_BUFFER_SIZE */
    uint64_t phys_addr = 0;     /* physical address of shared buffer */
    uint32_t seq = 0;           /* Message sequence number */
    size_t input_data_offset = 0; /* Offset in input buffer for tensor data.
                                     Set to elf_size after DYN_LOAD so that
                                     in-place rodata segments in the ELF are
                                     not overwritten by input tensor staging. */
};

/* C7x core 0 device tree address (stable across reboots/stop-start cycles) */
#define C7X_DEVICE_ADDR "7e000000.dsp"

/* Response timeout in milliseconds (300s for large models like ResNet-18) */
#define RESPONSE_TIMEOUT_MS  300000

/*
 * =============================================================================
 * Internal Helpers
 * =============================================================================
 */

/**
 * Flush ARM cache for the shared DMA buffer so the DSP sees fresh data.
 * Must be called after writing to input_buf and before sending RPMsg.
 */
static void sync_input_to_device(c7x_client_t *client)
{
    struct dma_buf_sync sync = {};
    sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    ioctl(client->dma_buf_fd.get(), DMA_BUF_IOCTL_SYNC, &sync);
}

/**
 * Invalidate ARM cache for the shared DMA buffer so the host sees DSP writes.
 * Must be called after receiving RPMsg response and before reading output_buf.
 */
static void sync_output_from_device(c7x_client_t *client)
{
    struct dma_buf_sync sync = {};
    sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    ioctl(client->dma_buf_fd.get(), DMA_BUF_IOCTL_SYNC, &sync);
}

static int send_and_recv(c7x_client_t *client,
                         void *req, size_t req_len,
                         void *resp, size_t resp_max_len)
{
    int ret = rpmsg_send(client->rpmsg_fd.get(), req, req_len);
    if (ret < 0) {
        fprintf(stderr, "c7x: Failed to send message: %s\n", strerror(-ret));
        return ret;
    }

    ret = rpmsg_recv(client->rpmsg_fd.get(), resp, resp_max_len, RESPONSE_TIMEOUT_MS);
    if (ret < 0) {
        if (ret == -ETIMEDOUT) {
            fprintf(stderr, "c7x: Response timeout\n");
        } else {
            fprintf(stderr, "c7x: Failed to receive response: %s\n", strerror(-ret));
        }
        return ret;
    }

    return ret;
}

/*
 * =============================================================================
 * Public API
 * =============================================================================
 */

c7x_client_t *c7x_client_open(void)
{
    struct dma_heap_allocation_data heap_data;
    struct dma_buf_phys_data phys_data;

    auto client = std::make_unique<c7x_client>();

    /* Open RPMessage connection */
    client->rpmsg_fd = UniqueFd(rpmsg_open(C7X_DEVICE_ADDR, C7X_SERVICE_ENDPOINT,
                                           C7X_SERVICE_NAME));
    if (!client->rpmsg_fd) {
        fprintf(stderr, "c7x: Failed to open RPMessage: %d\n", client->rpmsg_fd.get());
        return nullptr;
    }

    /* Open DMA heap for shared memory allocation */
    client->dma_heap_fd = UniqueFd(open(DMA_HEAP_DEVICE, O_RDONLY | O_CLOEXEC));
    if (!client->dma_heap_fd) {
        fprintf(stderr, "c7x: Failed to open DMA heap %s: %s\n",
                DMA_HEAP_DEVICE, strerror(errno));
        return nullptr;
    }

    /* Allocate shared buffer from DMA heap (input + output) */
    memset(&heap_data, 0, sizeof(heap_data));
    heap_data.len = C7X_SHARED_SIZE;
    heap_data.fd_flags = O_CLOEXEC | O_RDWR;
    heap_data.heap_flags = 0;

    if (ioctl(client->dma_heap_fd.get(), DMA_HEAP_IOCTL_ALLOC, &heap_data) < 0) {
        fprintf(stderr, "c7x: DMA heap alloc failed (%zu bytes): %s\n",
                static_cast<size_t>(C7X_SHARED_SIZE), strerror(errno));
        return nullptr;
    }
    client->dma_buf_fd = UniqueFd(static_cast<int>(heap_data.fd));

    /* Map the dmabuf into user-space */
    void *mapped = mmap(nullptr, C7X_SHARED_SIZE,
                        PROT_READ | PROT_WRITE, MAP_SHARED,
                        client->dma_buf_fd.get(), 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "c7x: Failed to mmap dmabuf: %s\n", strerror(errno));
        return nullptr;
    }
    client->shared_map = MmapRegion(mapped, C7X_SHARED_SIZE);

    client->input_buf  = client->shared_map.get();
    client->output_buf = static_cast<uint8_t *>(client->shared_map.get())
                         + C7X_INPUT_BUFFER_SIZE;

    /* Get physical address via remoteproc driver.
     * rproc_fd must stay open -- RPROC_IOC_DMA_BUF_ATTACH creates a device
     * attachment that maps the dmabuf into the DSP's address space.  Closing
     * rproc_fd would unmap it, making the shared memory invisible to the DSP. */
    {
        int idx = find_remoteproc_index(C7X_DEVICE_ADDR);
        if (idx < 0) {
            fprintf(stderr, "c7x: Failed to find remoteproc for %s\n", C7X_DEVICE_ADDR);
            return nullptr;
        }
        char rproc_path[64];
        snprintf(rproc_path, sizeof(rproc_path), "/dev/remoteproc%d", idx);
        client->rproc_fd = UniqueFd(open(rproc_path, O_RDONLY | O_CLOEXEC));
    }
    if (!client->rproc_fd) {
        fprintf(stderr, "c7x: Failed to open remoteproc for %s\n", C7X_DEVICE_ADDR);
        return nullptr;
    }

    memset(&phys_data, 0, sizeof(phys_data));
    phys_data.fd = static_cast<__u32>(client->dma_buf_fd.get());
    if (ioctl(client->rproc_fd.get(), RPROC_IOC_DMA_BUF_ATTACH, &phys_data) < 0) {
        fprintf(stderr, "c7x: Failed to get physical address: %s\n", strerror(errno));
        return nullptr;
    }
    client->phys_addr = static_cast<uint64_t>(phys_data.phys);

    /* Verify the physical address matches the expected DMA heap region */
    if (client->phys_addr != C7X_SHARED_PHYS_BASE) {
        fprintf(stderr, "c7x: WARNING: DMA heap allocated at phys 0x%llx, "
                "expected 0x%llx\n",
                static_cast<unsigned long long>(client->phys_addr),
                static_cast<unsigned long long>(C7X_SHARED_PHYS_BASE));
    }

    printf("c7x: Connected to compute service\n");
    printf("c7x: Input buffer:  %p (phys 0x%llx)\n",
           client->input_buf,
           static_cast<unsigned long long>(client->phys_addr));
    printf("c7x: Output buffer: %p (phys 0x%llx)\n",
           client->output_buf,
           static_cast<unsigned long long>(client->phys_addr + C7X_INPUT_BUFFER_SIZE));

    return client.release();
}

void c7x_client_close(c7x_client_t *client)
{
    if (!client) return;
    printf("c7x: Disconnected from compute service\n");
    delete client;
}

int c7x_client_ping(c7x_client_t *client, uint32_t *version, uint32_t *uptime_ms)
{
    struct c7x_msg_ping req;
    struct c7x_msg_ping_resp resp;

    if (!client) return -EINVAL;

    /* Build request */
    memset(&req, 0, sizeof(req));
    req.hdr.type = C7X_MSG_PING;
    req.hdr.seq = ++client->seq;
    req.hdr.len = sizeof(req);

    /* Send and receive */
    int ret = send_and_recv(client, &req, sizeof(req), &resp, sizeof(resp));
    if (ret < 0) {
        return ret;
    }

    /* Verify response */
    if (resp.hdr.type != C7X_MSG_PING_RESP) {
        fprintf(stderr, "c7x: Unexpected response type: 0x%04x\n", resp.hdr.type);
        return -EPROTO;
    }
    if (resp.hdr.seq != req.hdr.seq) {
        fprintf(stderr, "c7x: Sequence mismatch: expected %u, got %u\n",
                req.hdr.seq, resp.hdr.seq);
        return -EPROTO;
    }

    /* Return results */
    if (version) *version = resp.version;
    if (uptime_ms) *uptime_ms = resp.uptime_ms;

    return 0;
}

int c7x_client_get_status(c7x_client_t *client, c7x_status_t *status)
{
    struct c7x_msg_get_status req;
    struct c7x_msg_status_resp resp;

    if (!client || !status) return -EINVAL;

    /* Build request */
    memset(&req, 0, sizeof(req));
    req.hdr.type = C7X_MSG_GET_STATUS;
    req.hdr.seq = ++client->seq;
    req.hdr.len = sizeof(req);

    /* Send and receive */
    int ret = send_and_recv(client, &req, sizeof(req), &resp, sizeof(resp));
    if (ret < 0) {
        return ret;
    }

    /* Verify response */
    if (resp.hdr.type != C7X_MSG_STATUS_RESP) {
        return -EPROTO;
    }

    /* Copy results */
    status->version = resp.version;
    status->uptime_ms = resp.uptime_ms;
    status->jobs_completed = resp.jobs_completed;
    status->jobs_failed = resp.jobs_failed;

    return 0;
}

void *c7x_client_get_input_buffer(c7x_client_t *client, size_t *size)
{
    if (!client) return nullptr;
    if (size) *size = C7X_INPUT_BUFFER_SIZE;
    return client->input_buf;
}

void *c7x_client_get_output_buffer(c7x_client_t *client, size_t *size)
{
    if (!client) return nullptr;
    if (size) *size = C7X_OUTPUT_BUFFER_SIZE;
    return client->output_buf;
}

/*
 * =============================================================================
 * Dynamic Loading & TVM Inference API
 * =============================================================================
 */

/**
 * Helper: stage a file into the shared input buffer.
 */
static int stage_file(c7x_client_t *client, const char *file_path, size_t *size_out)
{
    UniqueFile f(fopen(file_path, "rb"));
    if (!f) {
        fprintf(stderr, "c7x: Failed to open %s: %s\n", file_path, strerror(errno));
        return -errno;
    }

    fseek(f.get(), 0, SEEK_END);
    size_t file_size = ftell(f.get());
    fseek(f.get(), 0, SEEK_SET);

    if (file_size > C7X_INPUT_BUFFER_SIZE) {
        fprintf(stderr, "c7x: File too large: %zu bytes (max %llu)\n",
                file_size, static_cast<unsigned long long>(C7X_INPUT_BUFFER_SIZE));
        return -EFBIG;
    }

    if (fread(client->input_buf, 1, file_size, f.get()) != file_size) {
        fprintf(stderr, "c7x: Failed to read %s\n", file_path);
        return -EIO;
    }

    sync_input_to_device(client);
    *size_out = file_size;
    return 0;
}

int c7x_client_model_load(c7x_client_t *client, const char *weights_file,
                          uint32_t *model_id_out)
{
    struct c7x_msg_model_load req;
    struct c7x_msg_model_load_resp resp;
    size_t file_size;

    if (!client || !weights_file || !model_id_out) return -EINVAL;

    /* Stage weights.bin in shared input buffer */
    int ret = stage_file(client, weights_file, &file_size);
    if (ret < 0) return ret;

    /* Build request */
    memset(&req, 0, sizeof(req));
    req.hdr.type = C7X_MSG_MODEL_LOAD;
    req.hdr.seq = ++client->seq;
    req.hdr.len = sizeof(req);
    req.weights_size = static_cast<uint32_t>(file_size);

    /* Send and receive */
    ret = send_and_recv(client, &req, sizeof(req), &resp, sizeof(resp));
    if (ret < 0) return ret;

    if (resp.hdr.type != C7X_MSG_MODEL_LOAD_RESP) return -EPROTO;
    if (resp.hdr.status != C7X_STATUS_SUCCESS) {
        fprintf(stderr, "c7x: MODEL_LOAD failed: status=%d\n", resp.hdr.status);
        return resp.hdr.status;
    }

    *model_id_out = resp.model_id;
    printf("c7x: Loaded model_id=%u, %u constants\n",
           resp.model_id, resp.num_constants);

    return 0;
}

int c7x_client_model_unload(c7x_client_t *client, uint32_t model_id)
{
    struct c7x_msg_model_unload req;
    struct c7x_msg_model_unload_resp resp;

    if (!client) return -EINVAL;

    memset(&req, 0, sizeof(req));
    req.hdr.type = C7X_MSG_MODEL_UNLOAD;
    req.hdr.seq = ++client->seq;
    req.hdr.len = sizeof(req);
    req.model_id = model_id;

    int ret = send_and_recv(client, &req, sizeof(req), &resp, sizeof(resp));
    if (ret < 0) return ret;

    if (resp.hdr.type != C7X_MSG_MODEL_UNLOAD_RESP) return -EPROTO;
    if (resp.hdr.status != C7X_STATUS_SUCCESS) {
        fprintf(stderr, "c7x: MODEL_UNLOAD failed: status=%d\n", resp.hdr.status);
        return resp.hdr.status;
    }

    printf("c7x: Unloaded model_id=%u\n", model_id);
    return 0;
}

int c7x_client_dyn_load(c7x_client_t *client, const char *elf_file,
                        uint32_t *handle_out)
{
    struct c7x_msg_dyn_load req;
    struct c7x_msg_dyn_load_resp resp;
    size_t file_size;

    if (!client || !elf_file || !handle_out) return -EINVAL;

    /* Stage ELF in shared input buffer */
    int ret = stage_file(client, elf_file, &file_size);
    if (ret < 0) return ret;

    /* Build request */
    memset(&req, 0, sizeof(req));
    req.hdr.type = C7X_MSG_DYN_LOAD;
    req.hdr.seq = ++client->seq;
    req.hdr.len = sizeof(req);
    req.elf_size = static_cast<uint32_t>(file_size);

    /* Send and receive */
    ret = send_and_recv(client, &req, sizeof(req), &resp, sizeof(resp));
    if (ret < 0) return ret;

    if (resp.hdr.type != C7X_MSG_DYN_LOAD_RESP) return -EPROTO;
    if (resp.hdr.status != C7X_STATUS_SUCCESS) {
        fprintf(stderr, "c7x: DYN_LOAD failed: status=%d\n", resp.hdr.status);
        return resp.hdr.status;
    }

    *handle_out = resp.module_handle;
    /* DLOAD maps rodata segments in-place in the input buffer.
     * Stage input tensors after the ELF to avoid overwriting them. */
    client->input_data_offset = file_size;
    printf("c7x: Loaded module handle=%u (text=%u data=%u "
           "input_offset=%zu)\n",
           resp.module_handle, resp.text_size, resp.data_size,
           client->input_data_offset);

    return 0;
}

int c7x_client_dyn_unload(c7x_client_t *client, uint32_t handle)
{
    struct c7x_msg_dyn_unload req;
    struct c7x_msg_dyn_unload_resp resp;

    if (!client) return -EINVAL;

    memset(&req, 0, sizeof(req));
    req.hdr.type = C7X_MSG_DYN_UNLOAD;
    req.hdr.seq = ++client->seq;
    req.hdr.len = sizeof(req);
    req.module_handle = handle;

    int ret = send_and_recv(client, &req, sizeof(req), &resp, sizeof(resp));
    if (ret < 0) return ret;

    if (resp.hdr.type != C7X_MSG_DYN_UNLOAD_RESP) return -EPROTO;
    if (resp.hdr.status != C7X_STATUS_SUCCESS) {
        fprintf(stderr, "c7x: DYN_UNLOAD failed: status=%d\n", resp.hdr.status);
        return resp.hdr.status;
    }

    /* In-place rodata region is no longer needed -- reset offset so
     * the next load cycle can use the full input buffer. */
    client->input_data_offset = 0;
    printf("c7x: Unloaded module handle=%u\n", handle);
    return 0;
}

/**
 * Internal helper: run INFER with an explicit repeat count in flags.
 */
static int c7x_client_infer_impl(c7x_client_t *client,
                                  uint32_t module_handle,
                                  uint32_t model_id,
                                  const c7x_tensor_desc_t *inputs, int num_inputs,
                                  c7x_tensor_desc_t *outputs, int *num_outputs,
                                  uint64_t *cycles,
                                  uint32_t repeat)
{
    /* INFER message - sized for up to 4 inputs */
    uint8_t req_buf[512];
    uint8_t resp_buf[512];
    auto *req = reinterpret_cast<struct c7x_msg_infer *>(req_buf);
    auto *resp = reinterpret_cast<struct c7x_msg_infer_resp *>(resp_buf);
    size_t data_offset;

    if (!client || !inputs || num_inputs < 1 || !outputs || !num_outputs)
        return -EINVAL;

    /* Stage input tensor data AFTER the ELF region in the input buffer.
     * DLOAD maps rodata segments in-place from the ELF, so we must not
     * overwrite that region with input tensors. */
    data_offset = client->input_data_offset;
    for (int i = 0; i < num_inputs; i++) {
        if (data_offset + inputs[i].data_size > C7X_INPUT_BUFFER_SIZE) {
            fprintf(stderr, "c7x: Input data exceeds buffer size\n");
            return -EFBIG;
        }
        if (inputs[i].data && inputs[i].data_size > 0) {
            memcpy(static_cast<uint8_t *>(client->input_buf) + data_offset,
                   inputs[i].data, inputs[i].data_size);
        }
        data_offset += inputs[i].data_size;
    }
    sync_input_to_device(client);

    /* Build request */
    memset(req_buf, 0, sizeof(req_buf));
    req->hdr.type = C7X_MSG_INFER;
    req->hdr.seq = ++client->seq;
    req->module_handle = module_handle;
    req->model_id = model_id;
    req->num_inputs = static_cast<uint32_t>(num_inputs);
    req->flags = (repeat > 1) ? (repeat & 0xFFFF) : 0;  /* 0 = run once (backward compat) */

    /* Fill tensor descriptors.  DSP addresses start after the ELF
     * region to match the host-side staging offset (rodata segments
     * mapped in-place by DLOAD must not be overwritten). */
    uint64_t cur_addr = C7X_INPUT_BUFFER_ADDR
                      + client->input_data_offset;
    for (int i = 0; i < num_inputs; i++) {
        req->inputs[i].data_addr = cur_addr;
        req->inputs[i].data_size = inputs[i].data_size;
        req->inputs[i].ndim = inputs[i].ndim;
        req->inputs[i].dtype_code = inputs[i].dtype_code;
        req->inputs[i].dtype_bits = inputs[i].dtype_bits;
        for (int j = 0; j < inputs[i].ndim && j < C7X_TENSOR_MAX_NDIM; j++) {
            req->inputs[i].shape[j] = inputs[i].shape[j];
        }
        cur_addr += inputs[i].data_size;
    }

    size_t req_size = sizeof(struct c7x_msg_hdr) + 4 * sizeof(uint32_t) +
                      num_inputs * sizeof(struct c7x_tensor_desc);
    req->hdr.len = static_cast<uint32_t>(req_size);

    /* Send and receive */
    int ret = send_and_recv(client, req, req_size, resp, sizeof(resp_buf));
    if (ret < 0) return ret;

    if (resp->hdr.type != C7X_MSG_INFER_RESP) return -EPROTO;
    if (resp->hdr.status != C7X_STATUS_SUCCESS) {
        fprintf(stderr, "c7x: INFER failed: status=%d return_value=%d\n",
                resp->hdr.status, resp->return_value);
        return resp->hdr.status;
    }

    /* Sync output buffer from DSP before reading */
    sync_output_from_device(client);

    /* Extract output tensor metadata */
    *num_outputs = static_cast<int>(resp->num_outputs);
    for (int i = 0; i < static_cast<int>(resp->num_outputs)
                    && i < C7X_TENSOR_MAX_NDIM; i++) {
        struct c7x_tensor_desc *out_td = &resp->outputs[i];
        outputs[i].data = client->output_buf; /* points to mmap'd output buffer */
        outputs[i].data_size = static_cast<size_t>(out_td->data_size);
        outputs[i].ndim = out_td->ndim;
        outputs[i].dtype_code = out_td->dtype_code;
        outputs[i].dtype_bits = out_td->dtype_bits;
        for (int j = 0; j < out_td->ndim && j < C7X_TENSOR_MAX_NDIM; j++) {
            outputs[i].shape[j] = out_td->shape[j];
        }
    }

    /* Print DSP printf output to stderr (profile text, layer traces).
     * Using stderr so it doesn't interfere with JSON on stdout. */
    if (resp->printf_size > 0 &&
        resp->printf_size <= C7X_PRINTF_BUF_SIZE) {
        size_t printf_offset = C7X_OUTPUT_BUFFER_SIZE
                             - C7X_PRINTF_BUF_SIZE + 16;
        const char *pdata = static_cast<const char *>(client->output_buf)
                            + printf_offset;
        fwrite(pdata, 1, resp->printf_size, stderr);
        fflush(stderr);
    }

    if (cycles) *cycles = resp->cycles;

    return 0;
}

int c7x_client_infer(c7x_client_t *client,
                     uint32_t module_handle,
                     uint32_t model_id,
                     const c7x_tensor_desc_t *inputs, int num_inputs,
                     c7x_tensor_desc_t *outputs, int *num_outputs,
                     uint64_t *cycles)
{
    return c7x_client_infer_impl(client, module_handle, model_id,
                                  inputs, num_inputs, outputs, num_outputs,
                                  cycles, /*repeat=*/1);
}

int c7x_client_infer_repeat(c7x_client_t *client,
                            uint32_t module_handle,
                            uint32_t model_id,
                            const c7x_tensor_desc_t *inputs, int num_inputs,
                            c7x_tensor_desc_t *outputs, int *num_outputs,
                            uint64_t *cycles,
                            uint32_t repeat)
{
    return c7x_client_infer_impl(client, module_handle, model_id,
                                  inputs, num_inputs, outputs, num_outputs,
                                  cycles, repeat);
}

const char *c7x_strerror(int status)
{
    switch (status) {
    case C7X_STATUS_SUCCESS:        return "Success";
    case C7X_STATUS_ERR_GENERIC:    return "Generic error";
    case C7X_STATUS_ERR_INVALID:    return "Invalid parameter";
    case C7X_STATUS_ERR_NOMEM:      return "Out of memory";
    case C7X_STATUS_ERR_BUSY:       return "Service busy";
    case C7X_STATUS_ERR_TIMEOUT:    return "Operation timeout";
    case C7X_STATUS_ERR_ADDR:       return "Invalid address";
    case C7X_STATUS_ERR_SIZE:       return "Invalid size";
    case C7X_STATUS_ERR_OP:         return "Unknown operation";
    case C7X_STATUS_ERR_LOAD:       return "Dynamic load failed";
    case C7X_STATUS_ERR_SYMBOL:     return "Symbol lookup failed";
    case C7X_STATUS_ERR_CALL:       return "Function call failed";
    case C7X_STATUS_ERR_HANDLE:     return "Invalid handle/id";
    case C7X_STATUS_ERR_WEIGHTS:    return "Weights parsing failed";
    case C7X_STATUS_ERR_TENSOR:     return "Tensor construction failed";
    case -EINVAL:                   return "Invalid argument";
    case -ETIMEDOUT:                return "Timeout";
    case -EPROTO:                   return "Protocol error";
    case -ENODEV:                   return "Device not found";
    case -EFBIG:                    return "File too large";
    default:                        return "Unknown error";
    }
}
