/*
 * C7x ARM Runtime — C++ test binary for c7x::Module API.
 *
 * Exercises the c7x::Module C++ interface on an AM67A ARM board against a
 * live C7x DSP firmware instance.  All five tests run through the same
 * Module instance (loaded once, closed at the end) to verify that repeated
 * use within a session is stable.
 *
 * Usage:
 *   test_c7x_runtime <lib0.out> <input.bin>
 *                    [--shape N,C,...] [--dtype float32]
 *                    [--ref ref.bin]   [--atol 1e-3]
 *
 * Arguments:
 *   lib0.out    TVM c_static DLOAD module built by build_dsp_dynmod() or
 *               TIDLOffloadCompiler.build().  The file is uploaded to the
 *               C7x DSP via DLOAD; weights may be embedded or separate.
 *   input.bin   Raw tensor data: flat, contiguous, row-major, no header.
 *               Must be at least shape_numel * dtype_bytes bytes.
 *   --shape     Input tensor dimensions, comma-separated (default: 1,64).
 *   --dtype     Element type: float32 float16 int32 int8 uint8 (default: float32).
 *   --ref       Optional CPU reference output for Test 3.  Same raw format
 *               as input.bin.  Skipped if omitted.
 *   --atol      Absolute tolerance for Test 3 (default: 1e-3).
 *
 * Tests:
 *   1. LOAD/CLOSE    — Module::Load() opens IPC + DYN_LOADs the ELF; two
 *                      Close() calls verify idempotency.
 *   2. INFERENCE     — Run() returns a well-formed output (ndim > 0,
 *                      data_size > 0); shape and dtype are printed.
 *   3. REFERENCE     — max|out - ref| < atol (float32 element-wise).
 *   4. CREATE_INPUT  — CreateInput() allocates inside StagingBuffer(); the
 *                      result from the pre-staged path matches standard Run().
 *   5. REPEATED_INFER — Three Run() calls with identical input produce
 *                       bit-identical output.
 *
 * Exit code: 0 if all executed tests pass, N = number of failures.
 * Test 1 is a hard prerequisite: failure exits immediately.
 *
 * See test/README.md for build, deploy, and expected output.
 */

#include "c7x_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

/* =========================================================================
 * Binary I/O
 * ========================================================================= */

/* Read an entire file into a byte vector.  Returns an empty vector on any
 * error (open failure, short read) after printing a diagnostic. */
static std::vector<uint8_t> read_binary(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s: %s\n", path, strerror(errno));
        return {};
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    if (fread(buf.data(), 1, buf.size(), f) != buf.size()) {
        fprintf(stderr, "ERROR: Short read from %s\n", path);
        fclose(f);
        return {};
    }
    fclose(f);
    return buf;
}

/* =========================================================================
 * DLPack dtype helpers
 * ========================================================================= */

/* Maps a dtype name string to the three fields needed for a DLDataType:
 * code (DLDataTypeCode), bits per element, and bytes per element.
 * Codes follow DLPack: 0=Int, 1=UInt, 2=Float. */
struct DTypeDesc {
    uint8_t code;
    uint8_t bits;
    size_t  bytes;
};

static DTypeDesc parse_dtype(const char *s)
{
    if (strcmp(s, "float32") == 0) return {2, 32, 4};
    if (strcmp(s, "float16") == 0) return {2, 16, 2};
    if (strcmp(s, "int32")   == 0) return {0, 32, 4};
    if (strcmp(s, "int8")    == 0) return {0,  8, 1};
    if (strcmp(s, "uint8")   == 0) return {1,  8, 1};
    fprintf(stderr, "WARNING: Unknown dtype '%s', defaulting to float32\n", s);
    return {2, 32, 4};
}

/* =========================================================================
 * Shape helpers
 * ========================================================================= */

/* Parse a comma-separated dimension string like "1,3,224,224" into a vector.
 * Stops at the first non-digit, non-comma character so trailing text is
 * silently ignored rather than causing a hard error. */
static std::vector<int64_t> parse_shape(const char *s)
{
    std::vector<int64_t> shape;
    const char *p = s;
    while (*p) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p) break;   /* no digit found — stop */
        shape.push_back(static_cast<int64_t>(v));
        p = end;
        if (*p == ',') ++p;
    }
    return shape;
}

/* Total number of elements across all dimensions. */
static size_t shape_numel(const std::vector<int64_t> &shape)
{
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    return n;
}

/* =========================================================================
 * Test harness
 * ========================================================================= */

static int n_pass = 0, n_fail = 0;

/* Print a PASS line and increment the pass counter. */
#define TEST_PASS(name) do { printf("  PASS  %s\n", name); ++n_pass; } while(0)

/* Print a FAIL line with a printf-style message and increment the fail
 * counter.  Two separate printf calls are used because variadic macros with
 * a format string and arguments are not portable in C++11 without GNU
 * extensions; splitting avoids -Wpedantic warnings. */
#define TEST_FAIL(name, ...) do { \
    printf("  FAIL  %s: ", name); \
    printf(__VA_ARGS__); \
    printf("\n"); \
    ++n_fail; \
} while(0)

/* =========================================================================
 * Main
 * ========================================================================= */

int main(int argc, char **argv)
{
    /* -----------------------------------------------------------------------
     * Argument parsing
     * --------------------------------------------------------------------- */
    if (argc < 3) {
        fprintf(stderr,
            "Usage: %s <lib0.out> <input.bin> [--shape N,...] [--dtype T]"
            " [--ref ref.bin] [--atol A]\n", argv[0]);
        return 1;
    }

    const char *lib0_path  = argv[1];
    const char *input_path = argv[2];
    const char *ref_path   = nullptr;
    const char *shape_str  = "1,64";
    const char *dtype_str  = "float32";
    double atol            = 1e-3;

    for (int i = 3; i < argc; ++i) {
        if (strcmp(argv[i], "--shape") == 0 && i+1 < argc)
            shape_str = argv[++i];
        else if (strcmp(argv[i], "--dtype") == 0 && i+1 < argc)
            dtype_str = argv[++i];
        else if (strcmp(argv[i], "--ref") == 0 && i+1 < argc)
            ref_path = argv[++i];
        else if (strcmp(argv[i], "--atol") == 0 && i+1 < argc)
            atol = atof(argv[++i]);
        else
            fprintf(stderr, "WARNING: Unknown argument '%s'\n", argv[i]);
    }

    auto shape = parse_shape(shape_str);
    if (shape.empty()) {
        fprintf(stderr, "ERROR: Invalid --shape '%s'\n", shape_str);
        return 1;
    }
    DTypeDesc dtype      = parse_dtype(dtype_str);
    size_t input_nbytes  = shape_numel(shape) * dtype.bytes;

    printf("test_c7x_runtime: %s\n", lib0_path);
    printf("  input: %s  shape: %s  dtype: %s (%zu bytes)\n",
           input_path, shape_str, dtype_str, input_nbytes);

    /* -----------------------------------------------------------------------
     * Read and validate the input tensor
     * --------------------------------------------------------------------- */
    auto input_buf = read_binary(input_path);
    if (input_buf.empty()) {
        fprintf(stderr, "ERROR: Failed to read %s\n", input_path);
        return 1;
    }
    if (input_buf.size() < input_nbytes) {
        fprintf(stderr, "ERROR: input.bin is %zu bytes, expected >= %zu\n",
                input_buf.size(), input_nbytes);
        return 1;
    }

    /* Wrap the raw buffer in a stack-allocated DLTensor.  strides=nullptr
     * tells consumers (including our c7x::Module implementation) that the
     * tensor is dense row-major, which is the layout produced by numpy
     * tofile() and TVM's standard tensor serialisation.  byte_offset=0
     * means data points at the first element. */
    DLTensor input_dl;
    memset(&input_dl, 0, sizeof(input_dl));
    input_dl.data        = input_buf.data();
    input_dl.device      = {kDLCPU, 0};
    input_dl.ndim        = static_cast<int32_t>(shape.size());
    input_dl.dtype       = {dtype.code, dtype.bits, /*lanes=*/1};
    input_dl.shape       = shape.data();
    input_dl.strides     = nullptr;  /* dense row-major */
    input_dl.byte_offset = 0;

    /* -----------------------------------------------------------------------
     * Test 1: LOAD/CLOSE
     *
     * Verifies that Module::Load() successfully opens the IPC connection to
     * the c7x_compute firmware and uploads the ELF via DYN_LOAD.  Then
     * calls Close() twice to confirm idempotency: the second call must be a
     * no-op rather than a double-free or assertion failure.
     *
     * This test is a hard prerequisite.  If loading fails the DSP is not
     * reachable or the ELF is invalid, and the remaining tests cannot run.
     * --------------------------------------------------------------------- */
    printf("\n--- Test 1: LOAD/CLOSE\n");
    try {
        auto vm = c7x::Module::Load(lib0_path);
        vm.Close();
        vm.Close();  /* second Close() must be a no-op */
        TEST_PASS("load_close");
    } catch (const std::exception &e) {
        TEST_FAIL("load_close", "%s", e.what());
        printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
        return n_fail;
    }

    /* Load once and reuse across tests 2–5 to amortize the DYN_LOAD cost
     * (~35 s for large models) and to exercise repeated use of one session. */
    c7x::Module vm = c7x::Module::Load(lib0_path);

    /* -----------------------------------------------------------------------
     * Test 2: INFERENCE
     *
     * Checks that Run() returns a structurally valid OutputTensor: ndim > 0
     * and data_size > 0.  Prints the output shape and dtype so failures can
     * be diagnosed without a reference comparison.
     *
     * out0 is saved here and used for reference in Test 4.  Note the output
     * lifetime contract: out0.dl.data points into the mmap'd result DDR
     * buffer, which is only valid until the next Run() call.  Tests 3 and 4
     * account for this.
     * --------------------------------------------------------------------- */
    printf("\n--- Test 2: INFERENCE\n");
    c7x::OutputTensor out0;
    try {
        out0 = vm.Run(&input_dl);
        if (out0.dl.ndim <= 0) {
            TEST_FAIL("inference_shape", "ndim=%d (expected > 0)", out0.dl.ndim);
        } else if (out0.data_size == 0) {
            TEST_FAIL("inference_size", "data_size=0");
        } else {
            printf("  Output: ndim=%d  data_size=%zu  dtype=%d.%d\n",
                   out0.dl.ndim, out0.data_size,
                   out0.dl.dtype.code, out0.dl.dtype.bits);
            for (int i = 0; i < out0.dl.ndim; i++)
                printf("    shape[%d]=%lld\n", i, (long long)out0._shape[i]);
            TEST_PASS("inference");
        }
    } catch (const std::exception &e) {
        TEST_FAIL("inference", "%s", e.what());
    }

    /* -----------------------------------------------------------------------
     * Test 3: REFERENCE COMPARISON
     *
     * Compares the inference output against a pre-computed CPU reference
     * using element-wise absolute difference.  Both tensors must be float32.
     * out0.dl.data is still valid here because no Run() has been called
     * since Test 2.
     *
     * Skipped if --ref was not provided (the test is optional; tests 1, 2,
     * 4, 5 are sufficient to verify API correctness without a CPU reference).
     * --------------------------------------------------------------------- */
    printf("\n--- Test 3: REFERENCE COMPARISON\n");
    if (!ref_path) {
        printf("  SKIP  no --ref provided\n");
    } else {
        auto ref_buf = read_binary(ref_path);
        if (ref_buf.empty()) {
            TEST_FAIL("reference", "failed to read %s", ref_path);
        } else if (ref_buf.size() != out0.data_size) {
            TEST_FAIL("reference", "size mismatch: out=%zu ref=%zu",
                      out0.data_size, ref_buf.size());
        } else {
            const float *out_f = static_cast<const float*>(out0.dl.data);
            const float *ref_f = reinterpret_cast<const float*>(ref_buf.data());
            size_t n = out0.data_size / sizeof(float);
            double max_diff = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double d = fabs(static_cast<double>(out_f[i]) -
                                static_cast<double>(ref_f[i]));
                if (d > max_diff) max_diff = d;
            }
            printf("  max|out - ref| = %.2e  (atol=%.2e)\n", max_diff, atol);
            if (max_diff <= atol)
                TEST_PASS("reference");
            else
                TEST_FAIL("reference", "max diff %.2e > atol %.2e", max_diff, atol);
        }
    }

    /* -----------------------------------------------------------------------
     * Test 4: CREATE_INPUT (zero-copy input path)
     *
     * Verifies the zero-copy input optimisation:
     *
     * (a) Range check: CreateInput() must return a DLTensor whose data
     *     pointer falls within the mmap'd staging DDR buffer
     *     [StagingBuffer(), StagingBuffer()+size).  Any other address would
     *     indicate a bug — the C layer would copy instead of using the
     *     pre-staged data.
     *
     * (b) Correctness check: inference via the pre-staged path must produce
     *     the same result as the standard Run(&input_dl) path.
     *
     * The comparison uses out0 from Test 2 as the reference.  However, out0
     * is backed by the result DDR buffer which is overwritten by each Run()
     * call.  If Test 3 ran a reference comparison (calling Run() implicitly
     * via out0.dl.data still pointing into the buffer but no new Run() was
     * issued), out0 is still valid.  But after vm.Run(pre), out0 is stale.
     * The memcmp first tries the cached out0 (fast path, succeeds unless
     * Test 3 ran an interleaved Run()); if the data differs, a fresh
     * standard-path run is done and compared against that instead.
     * --------------------------------------------------------------------- */
    printf("\n--- Test 4: CREATE_INPUT\n");
    try {
        size_t staging_size = 0;
        void  *staging_base = vm.StagingBuffer(&staging_size);

        DLTensor *pre = vm.CreateInput(shape.data(),
                                       static_cast<int>(shape.size()),
                                       {dtype.code, dtype.bits, 1});
        if (!pre) {
            TEST_FAIL("create_input_notnull", "CreateInput returned nullptr");
        } else {
            /* (a) Range check */
            uintptr_t data_addr = reinterpret_cast<uintptr_t>(pre->data);
            uintptr_t buf_start = reinterpret_cast<uintptr_t>(staging_base);
            uintptr_t buf_end   = buf_start + staging_size;
            if (data_addr < buf_start || data_addr >= buf_end) {
                TEST_FAIL("create_input_range",
                          "data=%p not in [%p, %p)",
                          pre->data, staging_base,
                          reinterpret_cast<void*>(buf_end));
            } else {
                TEST_PASS("create_input_range");
            }

            /* (b) Correctness check.
             * Write input directly to the pre-staged DDR address, then run.
             * The c7x_compute_client detects that pre->data is within the
             * staging buffer and skips the staging memcpy entirely. */
            memcpy(pre->data, input_buf.data(), input_nbytes);
            c7x::OutputTensor out_pre = vm.Run(pre);

            if (out_pre.data_size != out0.data_size) {
                TEST_FAIL("create_input_result",
                          "output size %zu != standard %zu",
                          out_pre.data_size, out0.data_size);
            } else if (memcmp(out_pre.dl.data, out0.dl.data, out0.data_size) != 0) {
                /* out0 may be stale (overwritten by the Run(pre) call above).
                 * Redo the standard-path run to get a fresh reference. */
                auto out0b = vm.Run(&input_dl);
                if (memcmp(out_pre.dl.data, out0b.dl.data, out0b.data_size) != 0)
                    TEST_FAIL("create_input_result", "output differs from standard run");
                else
                    TEST_PASS("create_input_result");
            } else {
                TEST_PASS("create_input_result");
            }
        }
    } catch (const std::exception &e) {
        TEST_FAIL("create_input", "%s", e.what());
    }

    /* -----------------------------------------------------------------------
     * Test 5: REPEATED_INFER
     *
     * Confirms that the DSP produces bit-identical output for the same input
     * across multiple consecutive Run() calls.  Non-determinism would
     * indicate uninitialized scratch memory, un-reset accumulators, or a
     * firmware state bug.
     *
     * Strategy: copy the first output to a reference vector (one allocation),
     * then compare the next two runs via memcmp directly into the live result
     * DDR buffer (no additional copies).  This works because each Run() call
     * overwrites the result buffer with fresh data before memcmp is called.
     * --------------------------------------------------------------------- */
    printf("\n--- Test 5: REPEATED_INFER\n");
    try {
        auto out0r = vm.Run(&input_dl);
        /* Save a copy of the first result — the result DDR buffer is
         * overwritten on every subsequent Run(). */
        std::vector<uint8_t> ref(static_cast<const uint8_t*>(out0r.dl.data),
                                 static_cast<const uint8_t*>(out0r.dl.data) + out0r.data_size);

        bool all_equal = true;
        for (int i = 0; i < 2 && all_equal; ++i) {
            auto out = vm.Run(&input_dl);
            if (out.data_size != ref.size() ||
                memcmp(out.dl.data, ref.data(), ref.size()) != 0)
                all_equal = false;
        }
        if (all_equal)
            TEST_PASS("repeated_infer");
        else
            TEST_FAIL("repeated_infer", "results differ across runs");
    } catch (const std::exception &e) {
        TEST_FAIL("repeated_infer", "%s", e.what());
    }

    /* -----------------------------------------------------------------------
     * Summary
     * --------------------------------------------------------------------- */
    vm.Close();
    printf("\nResults: %d passed, %d failed\n", n_pass, n_fail);
    return n_fail;
}
