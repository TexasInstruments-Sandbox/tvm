/*
 * C7x Arm Runtime — C++ inference API for TVM c_static modules on AM67A.
 *
 * Provides a VirtualMachine-compatible interface that routes inference to the
 * C7x DSP via the c7x_compute IPC service.  Only depends on DLPack (no TVM
 * runtime headers required) so it can be used from any C++ application.
 *
 * Usage (mirrors TVM C++ Module API):
 *
 *   // Standard — copy-based inputs, zero-copy outputs:
 *   auto vm = c7x::Module::Load("/models/resnet18.out");
 *   auto out = vm.Run(&input_dl_tensor);
 *   // out.dl.data → pointer into result DDR, valid until next Run() / Close()
 *
 *   // Zero-copy inputs — pre-allocate in staging DDR:
 *   DLTensor *inp = vm.CreateInput(shape, ndim, dtype);
 *   memcpy(inp->data, my_data, nbytes);   // write directly to staging DDR
 *   auto out = vm.Run(inp);               // no staging memcpy
 *
 * Build: included in libc7x_arm_runtime.so (aarch64).
 *
 * Full API reference (this C++ API and its Python mirror, documented
 * together since one is a direct translation of the other):
 * python/tvm/contrib/c7x/README.md
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <vector>
#include <stdexcept>

/* DLPack — standard tensor descriptor (no TVM dependency) */
#include <dlpack/dlpack.h>

namespace c7x {

/* Maximum number of output tensors from a single inference.
 * Sized to handle KV-cache models with many outputs (e.g. SmolLM: 61). */
static const int kMaxOutputs = 64;

/* Maximum number of input tensors per inference. */
static const int kMaxInputs  = 128;

/*
 * OutputTensor — thin DLTensor wrapper for inference results.
 *
 * dl.data points directly into the mmap'd result DDR buffer — no copy.
 * Valid until the next Module::Run() call or Module::Close().
 *
 * If you need to retain the data beyond the next inference, copy it:
 *   memcpy(my_buf, out.dl.data, out.dl.data_size);
 */
struct OutputTensor {
    DLTensor dl;          /* Standard DLTensor; data points into result_buf */
    int64_t  _shape[6];   /* Shape storage (dl.shape → this array) */
    size_t   data_size;   /* Byte size of the output (convenience) */
};

/*
 * Module — loaded C7x inference module.
 *
 * Not copyable; movable.  Create with Module::Load().
 */
class Module {
public:
    /*
     * Load a TVM c_static lib0.out for C7x inference.
     *
     * Opens a connection to the c7x_compute IPC service and DYN_LOADs the
     * module.  Connection and module handle are released on Close() or ~Module().
     *
     * Throws std::runtime_error on failure (DSP not reachable, file not found,
     * load error, etc.).
     */
    static Module Load(const std::string& lib0_path);

    ~Module();
    Module(Module&&) noexcept;
    Module& operator=(Module&&) noexcept;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;

    /* -----------------------------------------------------------------------
     * Standard inference — copy-based inputs, zero-copy outputs
     * ----------------------------------------------------------------------- */

    /*
     * Callable returned by operator[].  Takes DLTensor input arrays and
     * fills an OutputTensor array.  All function names map to cg_main_dsp.
     *
     * Returns 0 on success, negative errno on failure.
     */
    struct Function {
        int operator()(const DLTensor* const* inputs,  int num_inputs,
                       OutputTensor*         outputs,  int* num_outputs) const;
        Module* mod;  /* back-pointer (non-owning) */
    };

    /* Get a callable for a compiled function.  All names route to cg_main_dsp. */
    Function operator[](const std::string& name);

    /* Convenience: single-input → single-output (throws on error). */
    OutputTensor Run(const DLTensor* input);

    /* Convenience: multi-input → vector of zero-copy outputs (throws on error). */
    std::vector<OutputTensor> Run(const std::vector<const DLTensor*>& inputs);

    /* -----------------------------------------------------------------------
     * Zero-copy input path — pre-allocate tensor IN staging DDR
     * ----------------------------------------------------------------------- */

    /*
     * Allocate an input tensor directly in the staging buffer.
     *
     * Returns a DLTensor with data pointing into the mmap'd staging_buf.
     * User writes input data there; the next Run() call skips the memcpy for
     * this tensor (pre-staged detection based on pointer range).
     *
     * Inputs are allocated sequentially from the staging buffer start (after
     * the loaded ELF region).  A subsequent call with different shape/dtype
     * advances the offset.  All allocations are valid until Close().
     *
     * Returns nullptr on failure (buffer full).
     */
    DLTensor* CreateInput(const int64_t* shape, int ndim, DLDataType dtype);

    /*
     * Returns a pointer to the raw staging buffer and its size.
     * For advanced usage; prefer CreateInput() for per-tensor allocation.
     */
    void* StagingBuffer(size_t* size_out = nullptr) const;

    /* Unload module and close IPC connection. Idempotent. */
    void Close();

private:
    struct Impl;
    Impl* impl_ = nullptr;
    explicit Module(Impl* impl) : impl_(impl) {}
};

} /* namespace c7x */
