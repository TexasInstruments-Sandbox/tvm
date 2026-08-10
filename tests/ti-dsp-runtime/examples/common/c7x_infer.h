/*
 * c7x_infer.h -- Shared inference helpers for C++ examples.
 *
 * Thin convenience layer over c7x::Module (c7x_runtime.h) shared by every
 * standalone board-side example under tests/ti-dsp-runtime/examples/: hides
 * DLTensor boilerplate, and provides raw-flat-tensor-file input and CLI
 * dtype parsing (dtype table matches arm/test/test_c7x_runtime.cpp for
 * consistency). Not part of the shipped Arm Runtime API -- a real embedded
 * application has live sensor data, not files to read, so this stays
 * example-scoped.
 *
 * Header-only, like c7x_runtime.h itself: no separate .cc, no build-system
 * integration required -- #include this and c7x_runtime.h, then link
 * against libc7x_arm_runtime.so.
 *
 * What's deliberately NOT here: any pre/post-processing (argmax, box
 * decode, label lookup, ...). That's task-specific and belongs in each
 * example's own "main application" .cpp -- this header is only the part
 * that's identical across all of them.
 */
#pragma once

#include "c7x_runtime.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace c7x_examples {

/* ---------------------------------------------------------------------
 * dtype helpers
 * --------------------------------------------------------------------- */

/* DLDataType's (code, bits) plus the byte size, for sizing raw buffers.
 * Codes follow DLPack: 0=Int, 1=UInt, 2=Float. */
struct DTypeDesc {
    uint8_t code;
    uint8_t bits;
    size_t bytes;
};

inline DTypeDesc ParseDType(const std::string &name) {
    if (name == "float32") return {2, 32, 4};
    if (name == "float16") return {2, 16, 2};
    if (name == "int32") return {0, 32, 4};
    if (name == "int8") return {0, 8, 1};
    if (name == "uint8") return {1, 8, 1};
    throw std::runtime_error("Unknown dtype '" + name + "'");
}

inline size_t ShapeNumel(const std::vector<int64_t> &shape) {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
}

/* ---------------------------------------------------------------------
 * Raw flat tensor file input -- no header, matches test_c7x_runtime.cpp
 * and wheel-tests/test_inference_wheel.py's existing convention for
 * cross-language tensor exchange.
 * --------------------------------------------------------------------- */

inline std::vector<uint8_t> ReadRawTensor(const std::string &path) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open " + path + ": " + strerror(errno));
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        throw std::runtime_error("Cannot determine size of " + path);
    }
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    size_t n = fread(buf.data(), 1, buf.size(), f);
    fclose(f);
    if (n != buf.size()) throw std::runtime_error("Short read from " + path);
    return buf;
}

/* ---------------------------------------------------------------------
 * InferenceSession -- hides DLTensor field-filling boilerplate around
 * c7x::Module. This is the "common inference code": load once, run a
 * raw buffer through, get a result view back.
 * --------------------------------------------------------------------- */

class InferenceSession {
 public:
    explicit InferenceSession(const std::string &lib0_path)
        : module_(c7x::Module::Load(lib0_path)) {}

    /* Run one input tensor built from a raw buffer + shape/dtype. Checks
     * the buffer is at least as large as shape/dtype claim -- without
     * this, a truncated or mis-sized input.bin would silently hand the
     * DSP an out-of-bounds read via the DLTensor below (matches
     * test_c7x_runtime.cpp's own input_buf.size() < input_nbytes check).
     *
     * The returned OutputTensor is a view into the DSP result buffer --
     * valid until the next Run() call or Close() (see c7x_runtime.h's own
     * lifetime rules). Note: unlike Python's C7xVirtualMachine.last_cycles,
     * the C++ API has no on-device cycle-count accessor today -- any
     * timing an application built on this header wants is wall-clock only.
     */
    c7x::OutputTensor Run(const std::vector<uint8_t> &data, const std::vector<int64_t> &shape,
                          DTypeDesc dtype) {
        size_t expected = ShapeNumel(shape) * dtype.bytes;
        if (data.size() < expected) {
            throw std::runtime_error("Input buffer is " + std::to_string(data.size()) +
                                     " bytes, expected >= " + std::to_string(expected));
        }
        DLTensor dl;
        memset(&dl, 0, sizeof(dl));
        dl.data = const_cast<void *>(static_cast<const void *>(data.data()));
        dl.device = {kDLCPU, 0};
        dl.ndim = static_cast<int32_t>(shape.size());
        dl.dtype = {dtype.code, dtype.bits, /*lanes=*/1};
        dl.shape = const_cast<int64_t *>(shape.data());
        dl.strides = nullptr; /* dense row-major */
        dl.byte_offset = 0;
        return module_.Run(&dl);
    }

    void Close() { module_.Close(); }

 private:
    c7x::Module module_;
};

} /* namespace c7x_examples */
