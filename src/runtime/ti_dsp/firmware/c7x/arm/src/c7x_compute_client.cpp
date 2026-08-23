/*
 * C7x Compute Service - Host Client Library Implementation
 */

#include <cstddef>
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
 *
 * Upstream <linux/remoteproc_cdev.h> on this toolchain predates this ioctl
 * (it only has RPROC_SET/GET_SHUTDOWN_ON_RELEASE); the real struct is TI's
 * downstream addition, not in mainline. Layout verified against
 * ti-processor-sdk-linux-adas-j722s-evm-11_02_01_03/board-support/
 * ti-linux-kernel-6.12.57+git-ti/include/uapi/linux/remoteproc_cdev.h, whose
 * struct rproc_dma_buf_attach_data is byte-identical to this one -- field
 * renamed to `da` to match that name (it holds the device address, not a
 * raw physical address, despite the historical `dma_buf_phys_data` name).
 * =============================================================================
 */

struct dma_buf_phys_data {
    __u32 fd;
    __u64 da;
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
 * DmaBuf: one per-purpose dmabuf (D1) -- fd, mmap, and the DSP address it
 * translates to, each independently DMA_BUF_IOCTL_SYNC'able (D11: the ioctl
 * has no offset/length, so per-buffer granularity is the only lever).
 * =============================================================================
 */

struct DmaBuf {
    UniqueFd fd;                /* dmabuf fd */
    UniqueFd rproc_fd;          /* own /dev/remoteprocN attachment (D10) --
                                    must stay open, same reason as the single
                                    rproc_fd the client used to hold */
    MmapRegion map;             /* mmap'd region */
    uint64_t phys_addr = 0;
    uint64_t dsp_addr = 0;      /* phys_addr translated via the linear
                                    carveout mapping (F1) */
    size_t size = 0;

    void *ptr() const { return map.get(); }
};

/*
 * =============================================================================
 * Client State
 * =============================================================================
 */

struct c7x_client {
    UniqueFd rpmsg_fd;          /* RPMessage file descriptor */
    UniqueFd dma_heap_fd;       /* DMA heap device fd, shared by every alloc_dmabuf() call */
    uint32_t seq = 0;           /* Message sequence number */

    /* Allocated at open(): ELF+weights (480 MB, fixed at pool base so
     * C7X_STAGING_ADDR/C7X_KV_ADDR stay valid -- D6, G2 minimal scope) and
     * the 64 KB printf buffer. */
    DmaBuf staging;
    DmaBuf printf_buf;

    /* Allocated/grown in c7x_client_dyn_load() from tvm_dsp_io_meta, or via
     * c7x_client_reserve_io() (D2, D4). Persist across load/unload; only
     * grow, never shrink or move once io_capacity_locked (W4). */
    DmaBuf input_buf;
    DmaBuf output_buf;
    bool io_capacity_locked = false;

    size_t input_data_offset = 0; /* Size of the descriptor region at the
                                     front of input_buf (D9); tensor data
                                     starts right after it. Set after each
                                     DYN_LOAD from the declared num_inputs. */
    /* Byte counts from the most recent ERR_NOMEM response (infer or load --
     * only one call is ever in flight on a client, so one shared set of
     * fields is enough). Zeroed on every call, populated only when the
     * response actually carried nonzero OOM byte counts. */
    uint32_t last_oom_requested = 0;
    uint32_t last_oom_free = 0;
    uint32_t last_oom_total = 0;
    /* Declared input_buf/output_buf capacity from the most recent DYN_LOAD's
     * tvm_dsp_io_meta (0 if the loaded module had none). */
    uint64_t io_input_bytes = 0;
    uint64_t io_output_bytes = 0;
    uint32_t io_num_inputs = 0;
    uint32_t io_num_outputs = 0;
    uint32_t io_flags = 0;
};

/* Allocate `size` bytes from the shared carveout heap, mmap it, and attach
 * it to the DSP via its own rproc_fd (D10). `client` supplies dma_heap_fd
 * (opened once, shared across every dmabuf) and the device address needed
 * to resolve /dev/remoteprocN. */
static bool alloc_dmabuf(c7x_client_t *client, size_t size, DmaBuf *out)
{
    struct dma_heap_allocation_data heap_data;
    memset(&heap_data, 0, sizeof(heap_data));
    heap_data.len = size;
    heap_data.fd_flags = O_CLOEXEC | O_RDWR;
    heap_data.heap_flags = 0;

    if (ioctl(client->dma_heap_fd.get(), DMA_HEAP_IOCTL_ALLOC, &heap_data) < 0) {
        fprintf(stderr, "c7x: DMA heap alloc failed (%zu bytes): %s\n",
                size, strerror(errno));
        return false;
    }
    DmaBuf buf;
    buf.fd = UniqueFd(static_cast<int>(heap_data.fd));

    void *mapped = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED,
                        buf.fd.get(), 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "c7x: Failed to mmap dmabuf (%zu bytes): %s\n",
                size, strerror(errno));
        return false;
    }
    buf.map = MmapRegion(mapped, size);
    buf.size = size;

    int idx = find_remoteproc_index(C7X_DEVICE_ADDR);
    if (idx < 0) {
        fprintf(stderr, "c7x: Failed to find remoteproc for %s\n", C7X_DEVICE_ADDR);
        return false;
    }
    char rproc_path[64];
    snprintf(rproc_path, sizeof(rproc_path), "/dev/remoteproc%d", idx);
    buf.rproc_fd = UniqueFd(open(rproc_path, O_RDONLY | O_CLOEXEC));
    if (!buf.rproc_fd) {
        fprintf(stderr, "c7x: Failed to open remoteproc for %s\n", C7X_DEVICE_ADDR);
        return false;
    }

    struct dma_buf_phys_data phys_data;
    memset(&phys_data, 0, sizeof(phys_data));
    phys_data.fd = static_cast<__u32>(buf.fd.get());
    if (ioctl(buf.rproc_fd.get(), RPROC_IOC_DMA_BUF_ATTACH, &phys_data) < 0) {
        fprintf(stderr, "c7x: Failed to get physical address: %s\n", strerror(errno));
        return false;
    }
    buf.phys_addr = static_cast<uint64_t>(phys_data.da);

    if (buf.phys_addr < C7X_SHARED_PHYS_BASE ||
        buf.phys_addr >= C7X_SHARED_PHYS_BASE + C7X_SHARED_SIZE) {
        fprintf(stderr, "c7x: WARNING: dmabuf allocated at phys 0x%llx, "
                "outside expected range [0x%llx, 0x%llx)\n",
                static_cast<unsigned long long>(buf.phys_addr),
                static_cast<unsigned long long>(C7X_SHARED_PHYS_BASE),
                static_cast<unsigned long long>(C7X_SHARED_PHYS_BASE + C7X_SHARED_SIZE));
    }
    buf.dsp_addr = buf.phys_addr - C7X_SHARED_PHYS_BASE + C7X_SHARED_BASE;

    *out = std::move(buf);
    return true;
}

/* Grow input_buf/output_buf to at least (in_bytes, out_bytes) if not
 * already that large -- the only reallocation in the design (D2), safe
 * whenever called before any pointer into the buffer being grown has been
 * handed out (c7x_client_dyn_load(), or c7x_client_reserve_io() before the
 * first CreateInput()/INFER -- see io_capacity_locked). */
static bool grow_buf_if_needed(c7x_client_t *client, DmaBuf *buf, uint64_t needed_bytes,
                               const char *name)
{
    if (needed_bytes <= buf->size) return true;

    /* Release the old buffer before allocating the replacement.  Only ~32 MB
     * of the carveout sits above the 480 MB staging reservation, so holding
     * both live across the alloc double-books it: growing a 12 MB buffer to
     * 16 MB would transiently need 28 MB of the 32 MB and fail at roughly
     * half the real headroom, reported as a bare "DMA heap alloc failed".
     * Releasing first is safe because io_capacity_locked guarantees no
     * pointer into this buffer is outstanding; on failure the buffer is left
     * empty, which is the same state as never-allocated and can be retried
     * at a smaller capacity, rather than a half-resized one. */
    *buf = DmaBuf{};

    DmaBuf fresh;
    if (!alloc_dmabuf(client, static_cast<size_t>(needed_bytes), &fresh)) {
        fprintf(stderr, "c7x: Failed to size %s to %llu bytes\n",
                name, static_cast<unsigned long long>(needed_bytes));
        return false;
    }
    *buf = std::move(fresh);
    return true;
}

static bool ensure_io_capacity(c7x_client_t *client, uint64_t in_bytes, uint64_t out_bytes)
{
    return grow_buf_if_needed(client, &client->input_buf, in_bytes, "input_buf") &&
           grow_buf_if_needed(client, &client->output_buf, out_bytes, "output_buf");
}

/* C7x core 0 device tree address (stable across reboots/stop-start cycles) */
#define C7X_DEVICE_ADDR "7e000000.dsp"

/* Response timeout in milliseconds (600s to accommodate profiled models with many layers) */
#define RESPONSE_TIMEOUT_MS  600000

/*
 * =============================================================================
 * Internal Helpers
 * =============================================================================
 */

/**
 * Flush ARM cache for a dmabuf so the DSP sees fresh data.
 * Must be called after writing to the buffer and before sending RPMsg.
 */
static void sync_input_to_device(DmaBuf &buf)
{
    struct dma_buf_sync sync = {};
    sync.flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_WRITE;
    ioctl(buf.fd.get(), DMA_BUF_IOCTL_SYNC, &sync);
}

/**
 * Invalidate ARM cache for a dmabuf so the host sees DSP writes.
 * Must be called after receiving RPMsg response and before reading it.
 */
static void sync_output_from_device(DmaBuf &buf)
{
    struct dma_buf_sync sync = {};
    sync.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ;
    ioctl(buf.fd.get(), DMA_BUF_IOCTL_SYNC, &sync);
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
    auto client = std::make_unique<c7x_client>();

    /* Open RPMessage connection */
    client->rpmsg_fd = UniqueFd(rpmsg_open(C7X_DEVICE_ADDR, C7X_SERVICE_ENDPOINT,
                                           C7X_SERVICE_NAME));
    if (!client->rpmsg_fd) {
        fprintf(stderr, "c7x: Failed to open RPMessage: %d\n", client->rpmsg_fd.get());
        return nullptr;
    }

    /* Handshake before allocating anything: compare the firmware's protocol
     * major against ours.  The four wire structs this protocol uses changed
     * size in v3, and SET_PRINTF_BUF became mandatory below, so a mismatched
     * pair otherwise fails much deeper -- a new host against old firmware
     * reports a printf rebind failure, an old host against new firmware gets
     * "INFER message too small" in a log only the DSP sees.  Minor and patch
     * differences are wire-compatible by convention and pass silently. */
    {
        uint32_t fw_version = 0;
        int ret = c7x_client_ping(client.get(), &fw_version, nullptr);
        if (ret < 0) {
            fprintf(stderr, "c7x: Failed to reach the compute service (%s)\n",
                    strerror(-ret));
            return nullptr;
        }
        if (C7X_VERSION_MAJOR(fw_version) !=
            C7X_VERSION_MAJOR(C7X_SERVICE_VERSION)) {
            fprintf(stderr, "c7x: Protocol mismatch: firmware is v%u.%u.%u, "
                    "this client speaks v%u.%u.%u -- reflash c7x_compute.out "
                    "from this build (see deploy-c7x.sh)\n",
                    C7X_VERSION_MAJOR(fw_version), C7X_VERSION_MINOR(fw_version),
                    C7X_VERSION_PATCH(fw_version),
                    C7X_VERSION_MAJOR(C7X_SERVICE_VERSION),
                    C7X_VERSION_MINOR(C7X_SERVICE_VERSION),
                    C7X_VERSION_PATCH(C7X_SERVICE_VERSION));
            return nullptr;
        }
    }

    /* Open DMA heap for shared memory allocation (shared by every
     * alloc_dmabuf() call below). */
    client->dma_heap_fd = UniqueFd(open(DMA_HEAP_DEVICE, O_RDONLY | O_CLOEXEC));
    if (!client->dma_heap_fd) {
        fprintf(stderr, "c7x: Failed to open DMA heap %s: %s\n",
                DMA_HEAP_DEVICE, strerror(errno));
        return nullptr;
    }

    /* Allocate the staging dmabuf FIRST, covering both the ELF/weights
     * staging region and the KV region (C7X_STAGING_SIZE + C7X_KV_SIZE =
     * 480 MB). The carveout heap is gen_pool_first_fit (F2) -- allocating
     * this one first guarantees it lands at the pool's low end, so its
     * dsp_addr equals the fixed C7X_STAGING_ADDR and the DSP-side KV logic
     * (still addressed at the fixed C7X_KV_ADDR) keeps working unchanged
     * (D6, G2 minimal scope). Smaller buffers allocated afterward (printf_buf
     * below, input_buf/output_buf later) pack in above this range. */
    constexpr size_t kStagingSize =
        static_cast<size_t>(C7X_STAGING_SIZE + C7X_KV_SIZE);
    if (!alloc_dmabuf(client.get(), kStagingSize, &client->staging)) {
        fprintf(stderr, "c7x: Failed to allocate staging buffer\n");
        return nullptr;
    }
    if (client->staging.dsp_addr != C7X_STAGING_ADDR) {
        fprintf(stderr, "c7x: WARNING: staging buffer at dsp_addr 0x%llx, "
                "expected 0x%llx -- DYN_LOAD/KV addressing will be wrong\n",
                static_cast<unsigned long long>(client->staging.dsp_addr),
                static_cast<unsigned long long>(C7X_STAGING_ADDR));
    }

    if (!alloc_dmabuf(client.get(), C7X_PRINTF_BUF_SIZE, &client->printf_buf)) {
        fprintf(stderr, "c7x: Failed to allocate printf buffer\n");
        return nullptr;
    }

    printf("c7x: Connected to compute service\n");
    printf("c7x: Staging buffer: %p (phys 0x%llx, dsp 0x%llx)\n",
           client->staging.ptr(),
           static_cast<unsigned long long>(client->staging.phys_addr),
           static_cast<unsigned long long>(client->staging.dsp_addr));
    printf("c7x: Printf buffer:  %p (phys 0x%llx, dsp 0x%llx)\n",
           client->printf_buf.ptr(),
           static_cast<unsigned long long>(client->printf_buf.phys_addr),
           static_cast<unsigned long long>(client->printf_buf.dsp_addr));

    /* Rebind DSP printf from its boot-time scratch address to this
     * client's actual printf_buf (D8). Sent before any DYN_LOAD. */
    {
        struct c7x_msg_set_printf_buf req;
        struct c7x_msg_set_printf_buf_resp resp;
        memset(&req, 0, sizeof(req));
        req.hdr.type = C7X_MSG_SET_PRINTF_BUF;
        req.hdr.seq = ++client->seq;
        req.hdr.len = sizeof(req);
        req.printf_dsp_addr = client->printf_buf.dsp_addr;
        req.printf_size = static_cast<uint32_t>(client->printf_buf.size);

        int ret = send_and_recv(client.get(), &req, sizeof(req), &resp, sizeof(resp));
        if (ret < 0 || resp.hdr.type != C7X_MSG_SET_PRINTF_BUF_RESP ||
            resp.hdr.status != C7X_STATUS_SUCCESS) {
            fprintf(stderr, "c7x: Failed to rebind printf buffer (ret=%d, status=%d)\n",
                    ret, ret < 0 ? 0 : resp.hdr.status);
            return nullptr;
        }
    }

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

int c7x_client_get_last_oom(c7x_client_t *client, uint32_t *requested,
                            uint32_t *free_bytes, uint32_t *total)
{
    if (!client) return 0;
    if (client->last_oom_requested == 0 && client->last_oom_free == 0 &&
        client->last_oom_total == 0) {
        return 0;
    }
    if (requested) *requested = client->last_oom_requested;
    if (free_bytes) *free_bytes = client->last_oom_free;
    if (total) *total = client->last_oom_total;
    return 1;
}

/*
 * =============================================================================
 * Dynamic Loading & TVM Inference API
 * =============================================================================
 */

/**
 * Helper: stage a file into the shared staging buffer.
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

    if (file_size > C7X_STAGING_SIZE) {
        fprintf(stderr, "c7x: File too large: %zu bytes (max %llu)\n",
                file_size, static_cast<unsigned long long>(C7X_STAGING_SIZE));
        return -EFBIG;
    }

    if (fread(client->staging.ptr(), 1, file_size, f.get()) != file_size) {
        fprintf(stderr, "c7x: Failed to read %s\n", file_path);
        return -EIO;
    }

    sync_input_to_device(client->staging);
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

    /* Stage weights.bin in shared staging buffer */
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
    if (resp.hdr.status == C7X_STATUS_ERR_NOMEM) {
        client->last_oom_requested = resp.oom_requested;
        client->last_oom_free = resp.oom_free;
        client->last_oom_total = resp.oom_total;
    } else {
        client->last_oom_requested = client->last_oom_free = client->last_oom_total = 0;
    }
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

    /* Stage ELF in the staging buffer (ELF+weights, still fixed-address --
     * unaffected by input_buf/output_buf sizing below). */
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
    if (resp.hdr.status == C7X_STATUS_ERR_NOMEM) {
        client->last_oom_requested = resp.oom_requested;
        client->last_oom_free = resp.oom_free;
        client->last_oom_total = resp.oom_total;
    } else {
        client->last_oom_requested = client->last_oom_free = client->last_oom_total = 0;
    }
    if (resp.hdr.status != C7X_STATUS_SUCCESS) {
        fprintf(stderr, "c7x: DYN_LOAD failed: status=%d\n", resp.hdr.status);
        return resp.hdr.status;
    }

    *handle_out = resp.module_handle;
    client->io_input_bytes = resp.io_input_bytes;
    client->io_output_bytes = resp.io_output_bytes;
    client->io_num_inputs = resp.io_num_inputs;
    client->io_num_outputs = resp.io_num_outputs;
    client->io_flags = resp.io_flags;

    /* Descriptor region at the front of input_buf (D9). Sized to the
     * declared input count; when no io_meta was found (io_num_inputs == 0),
     * fall back to the inline-INFER threshold (today's models, per F5, are
     * always either well under this or have a table) rather than the
     * INFER_LARGE-scale 128-input ceiling: the region is reserved
     * unconditionally regardless of message type, so a large blind guess
     * would make reserve_io() unusable for exactly the small, no-table
     * models the fallback exists for. A no-table module that genuinely
     * needs INFER_LARGE isn't supported -- the descs_size check below
     * fails it clearly rather than silently guessing wrong. */
    constexpr uint32_t kFallbackDescRegionInputs = 4;
    uint32_t num_inputs_for_descs =
        client->io_num_inputs > 0 ? client->io_num_inputs : kFallbackDescRegionInputs;
    size_t descs_bytes = static_cast<size_t>(num_inputs_for_descs) *
                         sizeof(struct c7x_tensor_desc);
    client->input_data_offset = (descs_bytes + 63) & ~static_cast<size_t>(63);

    /* Clear the lock before growing, not after: no pointer for this module
     * has been handed out yet, and a failed growth below must still leave
     * c7x_client_reserve_io() usable -- unlocking only on the success path
     * would make -ENOMEM permanent, with close() the sole way out. */
    client->io_capacity_locked = false;

    /* Grow input_buf/output_buf if this module needs more than whatever a
     * previous load already sized them to (D2's only reallocation path).
     *
     * A declaration that can't be met warns rather than failing the load.
     * tvm_dsp_io_meta is derived from the entry signature, which is an upper
     * bound over calling conventions, not a per-call figure: a caller that
     * sets C7X_INFER_FLAG_KV_RESIDENT has its KV outputs diverted to
     * C7X_KV_ADDR and never puts them in output_buf, so it can need far less
     * than the signature implies (SmolLM prefill: 12.6 MB against a declared
     * 24.4 MB, on a carveout with only ~32 MB above the staging reservation).
     * Refusing the load would make such a module unloadable for a need it
     * does not have, and the caller has no way to say so first -- capacity
     * can only be declared against an already-loaded module.
     *
     * Whatever partial sizing succeeded is kept, and the caller states its
     * real need via c7x_client_reserve_io() (still usable: the lock was
     * cleared just above). Nothing is silently absorbed -- an
     * under-reservation still fails loudly at the first inference: -EFBIG
     * host-side for input, C7X_STATUS_ERR_SIZE + result_required for
     * output. */
    if (!ensure_io_capacity(client, resp.io_input_bytes, resp.io_output_bytes)) {
        fprintf(stderr, "c7x: WARNING: declared IO capacity (input=%llu "
                "output=%llu bytes) exceeds what the carveout can provide "
                "(allocated input_buf=%zu output_buf=%zu) -- call "
                "c7x_client_reserve_io() with this session's real need "
                "before the first CreateInput()/INFER\n",
                static_cast<unsigned long long>(resp.io_input_bytes),
                static_cast<unsigned long long>(resp.io_output_bytes),
                client->input_buf.size, client->output_buf.size);
    }

    printf("c7x: Loaded module handle=%u (text=%u data=%u)\n",
           resp.module_handle, resp.text_size, resp.data_size);
    if (resp.io_input_bytes != 0 || resp.io_output_bytes != 0) {
        printf("c7x: IO meta: input=%llu output=%llu inputs=%u outputs=%u "
               "flags=0x%x (input_buf=%zu output_buf=%zu bytes, "
               "descs=%zu bytes)\n",
               static_cast<unsigned long long>(resp.io_input_bytes),
               static_cast<unsigned long long>(resp.io_output_bytes),
               resp.io_num_inputs, resp.io_num_outputs, resp.io_flags,
               client->input_buf.size, client->output_buf.size,
               client->input_data_offset);
    } else {
        printf("c7x: IO meta: none (module built without it; "
               "call c7x_client_reserve_io() before the first "
               "CreateInput()/INFER)\n");
    }

    return 0;
}


int c7x_client_reserve_io(c7x_client_t *client, uint64_t in_bytes, uint64_t out_bytes)
{
    if (!client) return -EINVAL;
    if (client->io_capacity_locked) {
        fprintf(stderr, "c7x: reserve_io() rejected: input_buf/output_buf "
                "already in use by this load (CreateInput()/INFER already "
                "called) -- reserve before the first one\n");
        return -EBUSY;
    }
    if (!ensure_io_capacity(client, in_bytes, out_bytes)) return -ENOMEM;
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

    /* No module loaded, no known descriptor-region layout until the next
     * DYN_LOAD recomputes it -- avoid exposing the previous module's stale
     * value via c7x_client_get_input_data_offset() in between. */
    client->input_data_offset = 0;
    printf("c7x: Unloaded module handle=%u\n", handle);
    return 0;
}

/**
 * Internal helper: run INFER with explicit flags.
 * flags bits[15:0] = repeat count, bits[31:16] = feature flags.
 */
static int c7x_client_infer_impl(c7x_client_t *client,
                                  uint32_t module_handle,
                                  uint32_t model_id,
                                  const c7x_tensor_desc_t *inputs, int num_inputs,
                                  c7x_tensor_desc_t *outputs, int *num_outputs,
                                  uint64_t *cycles,
                                  uint32_t flags)
{
    /* INFER message - sized for up to 4 inputs */
    uint8_t req_buf[512];
    uint8_t resp_buf[512];
    auto *req = reinterpret_cast<struct c7x_msg_infer *>(req_buf);
    auto *resp = reinterpret_cast<struct c7x_msg_infer_resp *>(resp_buf);
    size_t data_offset;

    if (!client || !inputs || num_inputs < 1 || !outputs || !num_outputs)
        return -EINVAL;

    /* Stage input tensor data AFTER the descriptor region at the front of
     * input_buf (D9) -- input_data_offset is that region's size.
     *
     * Zero-copy path: if inputs[i].data already falls within input_buf's
     * range, skip the memcpy and compute the DSP address directly from the
     * pointer offset.  This is used by c7x::Module::CreateInput(). */
    data_offset = client->input_data_offset;

    const uint8_t *input_base = static_cast<const uint8_t *>(client->input_buf.ptr());
    const size_t   input_size = client->input_buf.size;

    /* Temporary descriptor array (stack, max 128 inputs). */
    struct c7x_tensor_desc desc_arr[128];
    if (num_inputs > 128) {
        fprintf(stderr, "c7x: Too many inputs: %d (max 128)\n", num_inputs);
        return -EINVAL;
    }

    /* Pass 1: stage non-pre-staged inputs and record DSP data_addr for all.
     * KV-resident inputs have data=NULL and a non-zero data_size — their
     * DSP address is derived from the fixed KV region layout (unchanged in
     * minimal scope, D6). */
    uint64_t data_addrs[128] = {};
    int kv_idx = 0;
    for (int i = 0; i < num_inputs; i++) {
        const uint8_t *input_ptr = static_cast<const uint8_t *>(inputs[i].data);

        if (input_ptr == nullptr && inputs[i].data_size > 0) {
            /* KV-resident: data lives at fixed DSP address */
            data_addrs[i] = C7X_KV_ADDR +
                            static_cast<uint64_t>(kv_idx) * C7X_KV_TENSOR_SIZE;
            kv_idx++;
            continue;
        }

        bool prestaged = (input_ptr != nullptr &&
                          input_ptr >= input_base &&
                          input_ptr + inputs[i].data_size <= input_base + input_size);
        if (prestaged) {
            /* Already in input_buf — derive DSP virtual address directly. */
            data_addrs[i] = client->input_buf.dsp_addr +
                            static_cast<uint64_t>(input_ptr - input_base);
        } else {
            /* Normal path: copy to input_buf at the current data_offset.
             * A zero-size input_buf (no io_meta, reserve_io() never called)
             * fails here rather than dereferencing a null input_base --
             * D4: exceeding declared capacity is always an error, never
             * absorbed. */
            if (data_offset + inputs[i].data_size > input_size) {
                fprintf(stderr, "c7x: Input data exceeds input_buf capacity "
                        "(need %zu at offset %zu, capacity %zu -- see "
                        "c7x_client_reserve_io())\n",
                        static_cast<size_t>(inputs[i].data_size), data_offset,
                        input_size);
                return -EFBIG;
            }
            if (input_ptr != nullptr && inputs[i].data_size > 0) {
                memcpy(const_cast<uint8_t *>(input_base) + data_offset,
                       input_ptr, inputs[i].data_size);
            }
            data_addrs[i] = client->input_buf.dsp_addr + static_cast<uint64_t>(data_offset);
            data_offset += inputs[i].data_size;
        }
    }

    /* Pass 2: build protocol descriptor array using the resolved addresses. */
    for (int i = 0; i < num_inputs; i++) {
        memset(&desc_arr[i], 0, sizeof(desc_arr[i]));
        desc_arr[i].data_addr  = data_addrs[i];
        desc_arr[i].data_size  = inputs[i].data_size;
        desc_arr[i].ndim       = inputs[i].ndim;
        desc_arr[i].dtype_code = inputs[i].dtype_code;
        desc_arr[i].dtype_bits = inputs[i].dtype_bits;
        for (int j = 0; j < inputs[i].ndim && j < C7X_TENSOR_MAX_NDIM; j++)
            desc_arr[i].shape[j] = inputs[i].shape[j];
    }

    /* Choose message type based on whether the inline form fits in the
     * 512-byte rpmsg buffer.  Inline INFER supports a handful of tensors;
     * for larger models (e.g. KV cache with 60+ tensors) use INFER_LARGE
     * which stages the descriptor array at the front of input_buf. */
    const size_t INLINE_HDR  = offsetof(struct c7x_msg_infer, inputs);
    const size_t inline_size = INLINE_HDR + num_inputs * sizeof(struct c7x_tensor_desc);

    size_t req_size;
    int ret;

    if (inline_size <= C7X_MAX_MSG_SIZE) {
        /* Standard INFER: descriptors inline in IPC message. Input tensor
         * data was already staged above; flush before sending. */
        sync_input_to_device(client->input_buf);

        memset(req_buf, 0, sizeof(req_buf));
        req->hdr.type = C7X_MSG_INFER;
        req->hdr.seq = ++client->seq;
        req->module_handle = module_handle;
        req->model_id = model_id;
        req->num_inputs = static_cast<uint32_t>(num_inputs);
        req->flags = flags;
        req->input_dsp_addr = client->input_buf.dsp_addr;
        req->input_size = static_cast<uint64_t>(client->input_buf.size);
        req->result_dsp_addr = client->output_buf.dsp_addr;
        req->result_size = static_cast<uint64_t>(client->output_buf.size);
        for (int i = 0; i < num_inputs; i++)
            req->inputs[i] = desc_arr[i];
        req_size = inline_size;
        req->hdr.len = static_cast<uint32_t>(req_size);
        ret = send_and_recv(client, req, req_size, resp, sizeof(resp_buf));
    } else {
        /* INFER_LARGE: descriptor array goes at the front of input_buf
         * (D9) -- tensor data above was already staged right after it, at
         * input_data_offset. */
        size_t descs_size = static_cast<size_t>(num_inputs) *
                            sizeof(struct c7x_tensor_desc);
        if (descs_size > client->input_data_offset) {
            fprintf(stderr, "c7x: Descriptor array (%zu bytes) exceeds the "
                    "reserved descriptor region (%zu bytes)\n",
                    descs_size, client->input_data_offset);
            return -EFBIG;
        }
        memcpy(client->input_buf.ptr(), desc_arr, descs_size);
        sync_input_to_device(client->input_buf);  /* covers both descriptors and tensor data */

        /* Build compact INFER_LARGE message (fits easily in 512 bytes) */
        uint8_t large_buf[sizeof(struct c7x_msg_infer_large)];
        memset(large_buf, 0, sizeof(large_buf));
        auto *lreq = reinterpret_cast<struct c7x_msg_infer_large *>(large_buf);
        lreq->hdr.type     = C7X_MSG_INFER_LARGE;
        lreq->hdr.seq      = ++client->seq;
        lreq->hdr.len      = sizeof(large_buf);
        lreq->module_handle = module_handle;
        lreq->model_id     = model_id;
        lreq->num_inputs   = static_cast<uint32_t>(num_inputs);
        lreq->flags        = flags;
        lreq->descs_addr   = client->input_buf.dsp_addr;
        lreq->descs_size   = static_cast<uint32_t>(descs_size);
        lreq->input_dsp_addr = client->input_buf.dsp_addr;
        lreq->input_size     = static_cast<uint64_t>(client->input_buf.size);
        lreq->result_dsp_addr = client->output_buf.dsp_addr;
        lreq->result_size     = static_cast<uint64_t>(client->output_buf.size);

        ret = send_and_recv(client, lreq, sizeof(large_buf),
                            resp, sizeof(resp_buf));
    }

    if (ret < 0) return ret;

    if (resp->hdr.type != C7X_MSG_INFER_RESP) return -EPROTO;

    /* Sync output_buf from DSP before reading it -- must happen even when
     * inference failed, not just on success (an out-of-band descs_addr from
     * a KV-resident call can still be present either way). */
    sync_output_from_device(client->output_buf);

    /* Print DSP printf output to stderr (profile text, layer traces).
     * Using stderr so it doesn't interfere with JSON on stdout. Done
     * before the status check: profile_layers records the name of the
     * in-flight call before invoking it, so on failure this is often the
     * only way to see which kernel call actually failed, instead of just
     * a generic -1.
     *
     * printf_size is delivered inline in resp (already valid at this
     * point), so printf_buf itself only needs syncing when there's
     * actually something to read -- skips a DMA_BUF_IOCTL_SYNC on every
     * normal (non-profiling) inference.
     *
     * The bound is the text area, not the whole mapping: the read starts past
     * the header, so accepting printf_size == printf_buf.size would run the
     * last C7X_SHM_PRINTF_HDR_SIZE bytes off the end of the 64 KB mmap. */
    const size_t printf_text_cap =
        client->printf_buf.size > C7X_SHM_PRINTF_HDR_SIZE
            ? client->printf_buf.size - C7X_SHM_PRINTF_HDR_SIZE
            : 0;
    if (resp->printf_size > 0 && resp->printf_size <= printf_text_cap) {
        sync_output_from_device(client->printf_buf);
        const char *pdata = static_cast<const char *>(client->printf_buf.ptr())
                            + C7X_SHM_PRINTF_HDR_SIZE;
        fwrite(pdata, 1, resp->printf_size, stderr);
        fflush(stderr);
    }

    if (resp->hdr.status == C7X_STATUS_ERR_NOMEM) {
        client->last_oom_requested = resp->oom_requested;
        client->last_oom_free = resp->oom_free;
        client->last_oom_total = resp->oom_total;
    } else {
        client->last_oom_requested = client->last_oom_free = client->last_oom_total = 0;
    }

    if (resp->hdr.status != C7X_STATUS_SUCCESS) {
        fprintf(stderr, "c7x: INFER failed: status=%d return_value=%d",
                resp->hdr.status, resp->return_value);
        if (resp->hdr.status == C7X_STATUS_ERR_SIZE) {
            fprintf(stderr, " (output_buf needs >= %llu bytes; "
                    "see c7x_client_reserve_io())",
                    static_cast<unsigned long long>(resp->result_required));
        }
        fprintf(stderr, "\n");
        return resp->hdr.status;
    }

    /* Extract output tensor metadata.
     * For small output counts: descriptors are inline in resp->outputs[].
     * For large output counts (KV cache): descriptors are in output_buf
     * at resp->descs_addr (out-of-band), already sync'd above. Every
     * address is range-checked against output_buf.size before use --
     * unlike the old fixed 32 MB result buffer, output_buf is now sized
     * to the declared capacity, so a firmware/protocol bug here would
     * otherwise read out of bounds instead of failing loudly. */
    const struct c7x_tensor_desc *td_base;
    if (resp->descs_addr != 0) {
        if (!C7X_IS_VALID_RANGE(resp->descs_addr, resp->descs_size,
                                client->output_buf.dsp_addr, client->output_buf.size)) {
            fprintf(stderr, "c7x: INFER response descs_addr out of bounds\n");
            return -EPROTO;
        }
        uint64_t descs_offset = resp->descs_addr - client->output_buf.dsp_addr;
        td_base = reinterpret_cast<const struct c7x_tensor_desc *>(
            static_cast<const uint8_t *>(client->output_buf.ptr()) + descs_offset);
    } else {
        td_base = &resp->outputs[0];
    }

    *num_outputs = static_cast<int>(resp->num_outputs);
    for (int i = 0; i < static_cast<int>(resp->num_outputs); i++) {
        const struct c7x_tensor_desc *out_td = &td_base[i];
        if (!C7X_IS_VALID_RANGE(out_td->data_addr, out_td->data_size,
                                client->output_buf.dsp_addr, client->output_buf.size)) {
            fprintf(stderr, "c7x: INFER response output %d data_addr out of bounds\n", i);
            return -EPROTO;
        }
        /* data_addr is a DSP virtual address in output_buf; convert to a
         * host userspace pointer via output_buf's mmap offset. */
        uint64_t data_offset_out = out_td->data_addr - client->output_buf.dsp_addr;
        outputs[i].data = static_cast<uint8_t *>(client->output_buf.ptr())
                          + data_offset_out;
        outputs[i].data_size = static_cast<size_t>(out_td->data_size);
        outputs[i].ndim = out_td->ndim;
        outputs[i].dtype_code = out_td->dtype_code;
        outputs[i].dtype_bits = out_td->dtype_bits;
        for (int j = 0; j < out_td->ndim && j < C7X_TENSOR_MAX_NDIM; j++) {
            outputs[i].shape[j] = out_td->shape[j];
        }
    }

    if (cycles) *cycles = resp->cycles;

    /* Lock capacity now, not at function entry: outputs[] above points into
     * output_buf, and that pointer must survive until the caller is done
     * with it, so no c7x_client_reserve_io() resize is safe after this
     * point. A failed attempt (-EFBIG below, or any early return above)
     * never reaches here, so a too-small reservation can still be grown
     * and retried on the same client (D2's "no corruption" retry path). */
    client->io_capacity_locked = true;

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
                                  cycles, /*flags=*/0);
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
                                  cycles, /*flags=*/(repeat & 0xFFFF));
}

int c7x_client_infer_flags(c7x_client_t *client,
                           uint32_t module_handle,
                           uint32_t model_id,
                           const c7x_tensor_desc_t *inputs, int num_inputs,
                           c7x_tensor_desc_t *outputs, int *num_outputs,
                           uint64_t *cycles,
                           uint32_t flags)
{
    return c7x_client_infer_impl(client, module_handle, model_id,
                                  inputs, num_inputs, outputs, num_outputs,
                                  cycles, flags);
}

size_t c7x_client_input_capacity(c7x_client_t *client)
{
    return client ? client->input_buf.size : 0;
}

/* Locks capacity only when a pointer is actually handed back.  Locking on
 * entry would also lock the "nothing to hand out" case -- a module with no
 * io_meta has a zero-size input_buf, so the caller gets nullptr and is told
 * to call c7x_client_reserve_io(), which would then be rejected -EBUSY for
 * the life of the client. */
void *c7x_client_get_input_buffer(c7x_client_t *client, size_t *size)
{
    if (!client) { if (size) *size = 0; return nullptr; }
    if (size) *size = client->input_buf.size;
    void *ptr = client->input_buf.ptr();
    if (ptr) client->io_capacity_locked = true;
    return ptr;
}

size_t c7x_client_get_input_data_offset(c7x_client_t *client)
{
    return client ? client->input_data_offset : 0;
}

void c7x_client_get_io_meta(c7x_client_t *client,
                            uint64_t *input_bytes, uint64_t *output_bytes,
                            uint32_t *num_inputs, uint32_t *num_outputs,
                            uint32_t *flags)
{
    if (input_bytes)  *input_bytes  = client ? client->io_input_bytes  : 0;
    if (output_bytes) *output_bytes = client ? client->io_output_bytes : 0;
    if (num_inputs)   *num_inputs   = client ? client->io_num_inputs   : 0;
    if (num_outputs)  *num_outputs  = client ? client->io_num_outputs  : 0;
    if (flags)        *flags        = client ? client->io_flags       : 0;
}

/* Same "lock only on a real hand-out" rule as c7x_client_get_input_buffer(). */
void *c7x_client_get_output_buffer(c7x_client_t *client, size_t *size)
{
    if (!client) { if (size) *size = 0; return nullptr; }
    if (size) *size = client->output_buf.size;
    void *ptr = client->output_buf.ptr();
    if (ptr) client->io_capacity_locked = true;
    return ptr;
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
    case -EBUSY:                    return "input_buf/output_buf already in use";
    default:                        return "Unknown error";
    }
}
