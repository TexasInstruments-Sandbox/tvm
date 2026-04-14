/*
 * C7x Arm Runtime — implementation of c7x::Module.
 *
 * Wraps c7x_compute_client IPC calls and exposes a DLPack-based inference API
 * that mirrors the TVM C++ Module interface.
 *
 * Zero-copy strategy:
 *   - Outputs: c7x_client_infer() already returns pointers into the mmap'd
 *     result DDR buffer.  OutputTensor.dl.data wraps these directly — no copy.
 *   - Inputs: if the user's DLTensor.data falls within [staging_buf,
 *     staging_buf + C7X_STAGING_SIZE), the staging memcpy is skipped.
 *     Use CreateInput() to obtain a DLTensor pre-allocated in staging DDR.
 */

#include "c7x_runtime.h"
#include "c7x_compute_client.h"
#include "c7x_compute_protocol.h"

#include <cstdio>
#include <cstring>
#include <cerrno>
#include <stdexcept>
#include <string>
#include <vector>

namespace c7x {

/* =========================================================================
 * Impl — private implementation (pimpl idiom)
 * ========================================================================= */

struct Module::Impl {
    c7x_client_t *client       = nullptr;
    uint32_t      handle       = 0;        /* DYN_LOAD module handle */
    uint32_t      model_id     = 0;        /* 0 = embedded weights */

    /* Staging buffer pre-allocation state (for CreateInput). */
    size_t        staging_alloc_offset = 0; /* next free offset in staging_buf
                                               (past ELF region, set after load) */

    /* Per-call output scratch (reused across Run() calls). */
    c7x_tensor_desc_t out_descs[kMaxOutputs];
    int               num_outputs = 0;

    /* Input DLTensors allocated via CreateInput (one per slot). */
    static const int kMaxInputSlots = kMaxInputs;
    DLTensor input_tensors[kMaxInputSlots];
    int64_t  input_shapes[kMaxInputSlots][6];
    int      num_inputs_alloc = 0;

    ~Impl() { close(); }

    void close() {
        if (handle && client) {
            c7x_client_dyn_unload(client, handle);
            handle = 0;
        }
        if (client) {
            c7x_client_close(client);
            client = nullptr;
        }
        /* Clear pre-staged input tensors so their data pointers no longer
         * point into the now-unmapped staging buffer. */
        num_inputs_alloc = 0;
        staging_alloc_offset = 0;
    }
};

/* =========================================================================
 * Internal helpers
 * ========================================================================= */

/*
 * Compute element size in bytes from DLDataType.
 */
static size_t dtype_itemsize(DLDataType dtype)
{
    return (static_cast<size_t>(dtype.bits) * dtype.lanes + 7) / 8;
}

/*
 * Compute total byte size of a DLTensor.
 */
static size_t tensor_nbytes(const DLTensor* t)
{
    size_t n = dtype_itemsize(t->dtype);
    for (int i = 0; i < t->ndim; i++) n *= static_cast<size_t>(t->shape[i]);
    return n;
}

/*
 * Build a c7x_tensor_desc_t from a DLTensor (for the client API).
 * data is set to the DLTensor's data pointer (host address).
 */
static c7x_tensor_desc_t dl_to_c7x_desc(const DLTensor* t)
{
    c7x_tensor_desc_t d;
    memset(&d, 0, sizeof(d));
    d.data      = static_cast<uint8_t*>(t->data) + t->byte_offset;
    d.data_size = tensor_nbytes(t);
    d.ndim      = t->ndim;
    d.dtype_code = static_cast<int32_t>(t->dtype.code);
    d.dtype_bits = static_cast<int32_t>(t->dtype.bits);
    for (int i = 0; i < t->ndim && i < C7X_TENSOR_MAX_NDIM; i++)
        d.shape[i] = t->shape[i];
    return d;
}

/*
 * Wrap an output c7x_tensor_desc_t as an OutputTensor.
 * out.dl.data points directly into result_buf — zero-copy.
 */
static OutputTensor c7x_desc_to_output(const c7x_tensor_desc_t& d)
{
    OutputTensor out;
    memset(&out, 0, sizeof(out));
    out.data_size = d.data_size;

    out.dl.data        = d.data;  /* pointer into mmap'd result DDR */
    out.dl.device      = {kDLCPU, 0};
    out.dl.ndim        = d.ndim;
    out.dl.dtype.code  = static_cast<uint8_t>(d.dtype_code);
    out.dl.dtype.bits  = static_cast<uint8_t>(d.dtype_bits);
    out.dl.dtype.lanes = 1;
    out.dl.shape       = out._shape;
    out.dl.strides     = nullptr;
    out.dl.byte_offset = 0;
    for (int i = 0; i < d.ndim && i < C7X_TENSOR_MAX_NDIM; i++)
        out._shape[i] = d.shape[i];
    return out;
}

/* =========================================================================
 * Module::Load
 * ========================================================================= */

Module Module::Load(const std::string& lib0_path)
{
    auto *impl = new Impl();

    impl->client = c7x_client_open();
    if (!impl->client) {
        delete impl;
        throw std::runtime_error(
            "c7x::Module::Load: c7x_client_open() failed — "
            "is the c7x_compute firmware running?");
    }

    int rc = c7x_client_dyn_load(impl->client, lib0_path.c_str(), &impl->handle);
    if (rc != 0) {
        c7x_client_close(impl->client);
        impl->client = nullptr;
        delete impl;
        throw std::runtime_error(
            std::string("c7x::Module::Load: c7x_client_dyn_load(") +
            lib0_path + ") failed: " + c7x_strerror(rc));
    }

    /* Record the staging offset after the ELF so CreateInput allocates past it.
     * c7x_client_get_input_data_offset() returns the byte offset that the
     * client already computed during DYN_LOAD (= actual ELF file size). */
    impl->staging_alloc_offset = c7x_client_get_input_data_offset(impl->client);

    return Module(impl);
}

/* =========================================================================
 * Lifecycle
 * ========================================================================= */

Module::~Module() { delete impl_; }

Module::Module(Module&& o) noexcept : impl_(o.impl_) { o.impl_ = nullptr; }

Module& Module::operator=(Module&& o) noexcept {
    if (this != &o) {
        delete impl_;
        impl_ = o.impl_;
        o.impl_ = nullptr;
    }
    return *this;
}

void Module::Close() {
    if (impl_) {
        impl_->close();
        impl_->handle = 0;
        impl_->client = nullptr;
    }
}

/* =========================================================================
 * Function::operator() — core inference dispatch
 * ========================================================================= */

int Module::Function::operator()(const DLTensor* const* inputs, int num_inputs,
                                  OutputTensor* outputs, int* num_outputs) const
{
    if (!mod || !mod->impl_) return -EINVAL;
    if (!inputs || num_inputs < 0 || num_inputs > kMaxInputs) return -EINVAL;
    if (!outputs || !num_outputs) return -EINVAL;

    Impl* impl = mod->impl_;

    /* Build c7x_tensor_desc_t array from DLTensor inputs.
     * Pre-staged detection: if input data is already in the staging buffer,
     * c7x_client_infer_impl skips the memcpy for that tensor. */
    c7x_tensor_desc_t in_descs[kMaxInputs];
    for (int i = 0; i < num_inputs; i++)
        in_descs[i] = dl_to_c7x_desc(inputs[i]);

    uint64_t cycles = 0;
    int rc = c7x_client_infer(impl->client, impl->handle, impl->model_id,
                               in_descs, num_inputs,
                               impl->out_descs, &impl->num_outputs,
                               &cycles);
    if (rc != 0) {
        fprintf(stderr, "c7x::Function: c7x_client_infer failed: %s\n",
                c7x_strerror(rc));
        return rc;
    }

    *num_outputs = impl->num_outputs;
    for (int i = 0; i < impl->num_outputs; i++)
        outputs[i] = c7x_desc_to_output(impl->out_descs[i]);

    return 0;
}

/* =========================================================================
 * Module::operator[] and Run convenience wrappers
 * ========================================================================= */

Module::Function Module::operator[](const std::string& /*name*/)
{
    /* All function names route to the single cg_main_dsp entry point. */
    return Function{this};
}

OutputTensor Module::Run(const DLTensor* input)
{
    OutputTensor outputs[kMaxOutputs];
    int num_outputs = 0;
    const DLTensor* inputs[] = {input};
    auto fn = (*this)["main"];
    int rc = fn(inputs, 1, outputs, &num_outputs);
    if (rc != 0)
        throw std::runtime_error(std::string("c7x::Module::Run failed: ") +
                                 c7x_strerror(rc));
    return outputs[0];
}

std::vector<OutputTensor> Module::Run(const std::vector<const DLTensor*>& inputs)
{
    OutputTensor outputs[kMaxOutputs];
    int num_outputs = 0;
    auto fn = (*this)["main"];
    int rc = fn(inputs.data(), static_cast<int>(inputs.size()),
                outputs, &num_outputs);
    if (rc != 0)
        throw std::runtime_error(std::string("c7x::Module::Run failed: ") +
                                 c7x_strerror(rc));
    return std::vector<OutputTensor>(outputs, outputs + num_outputs);
}

/* =========================================================================
 * CreateInput — pre-allocate input tensor in staging DDR
 * ========================================================================= */

DLTensor* Module::CreateInput(const int64_t* shape, int ndim, DLDataType dtype)
{
    if (!impl_ || !impl_->client) return nullptr;
    if (impl_->num_inputs_alloc >= Impl::kMaxInputSlots) return nullptr;

    /* Compute byte size of the tensor */
    size_t nbytes = dtype_itemsize(dtype);
    for (int i = 0; i < ndim; i++) nbytes *= static_cast<size_t>(shape[i]);

    /* Get staging buffer base from client */
    size_t staging_size = 0;
    void *staging_base = c7x_client_get_input_buffer(impl_->client, &staging_size);
    if (!staging_base) return nullptr;

    /* Check space */
    if (impl_->staging_alloc_offset + nbytes > staging_size) {
        fprintf(stderr, "c7x::CreateInput: staging buffer full "
                "(need %zu, offset %zu, size %zu)\n",
                nbytes, impl_->staging_alloc_offset, staging_size);
        return nullptr;
    }

    int slot = impl_->num_inputs_alloc++;
    DLTensor* t = &impl_->input_tensors[slot];
    int64_t* shp = impl_->input_shapes[slot];

    /* Fill DLTensor with pointer into staging DDR */
    memset(t, 0, sizeof(*t));
    t->data        = static_cast<uint8_t*>(staging_base) + impl_->staging_alloc_offset;
    t->device      = {kDLCPU, 0};
    t->ndim        = ndim;
    t->dtype       = dtype;
    t->shape       = shp;
    t->strides     = nullptr;
    t->byte_offset = 0;
    for (int i = 0; i < ndim && i < 6; i++) shp[i] = shape[i];

    impl_->staging_alloc_offset += nbytes;
    /* Align to 64 bytes for subsequent allocations */
    impl_->staging_alloc_offset = (impl_->staging_alloc_offset + 63) & ~63UL;

    return t;
}

/* =========================================================================
 * StagingBuffer
 * ========================================================================= */

void* Module::StagingBuffer(size_t* size_out) const
{
    if (!impl_ || !impl_->client) {
        if (size_out) *size_out = 0;
        return nullptr;
    }
    return c7x_client_get_input_buffer(impl_->client, size_out);
}

} /* namespace c7x */
