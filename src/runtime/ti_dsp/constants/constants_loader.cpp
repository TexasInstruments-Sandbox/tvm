/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file constants/constants_loader.cpp
 * \brief DSP-compatible constants loader for TVM-generated code
 *
 * This provides TVMDSPParseConstants() for DSP targets using the
 * weights.bin parser for zero-copy constant loading.
 *
 * For host emulation, also provides TVMGetConstants() returning
 * std::vector<tvm::ffi::Any> for compatibility with generated code.
 *
 * Build configurations:
 * - TVM_DSP_TARGET_HOST: Host emulation (has std::vector)
 * - TVM_DSP_TARGET_C66X: C66x DSP build (no STL)
 *
 * Weight data source configurations:
 * - TVM_DSP_WEIGHTS_EMBEDDED: Use linker-embedded weights
 * - TVM_DSP_WEIGHTS_PATH: Load from filesystem (host only)
 * - Otherwise: Call TVMDSPSetWeightsData() before parsing
 */

#include <cstdint>
#include <cstdio>

/* C headers - no extern "C" wrapper needed (they have their own guards) */
#include "../platform/dsp_platform.h"
#include "constants.h"
#include "constants_c_api.h"

#ifndef TVM_DSP_TARGET_C66X
/* Host emulation - include C++ wrappers */
#include <vector>
#include "../cpp/any.h"
#endif

/*---------------------------------------------------------------------------
 * Weight Data Symbols
 *
 * These symbols are created by the weight embedding process (objcopy/ld).
 * For host emulation with filesystem loading, use TVMDSPSetWeightsData()
 * to set the weights data before calling TVMDSPParseConstants().
 *---------------------------------------------------------------------------*/

#if defined(TVM_DSP_WEIGHTS_EMBEDDED)
/* Linker-embedded weights (created by objcopy or similar) */
extern "C" const char _binary_weights_bin_start[];
/* Note: Use size variable instead of end pointer due to type mismatch
 * between linker-generated pointer vs assembly label symbols */
extern "C" const unsigned int _binary_weights_bin_size;
#endif

/* Constants loading state */
static bool g_constants_parsed = false;

/*!
 * \brief Check if weights data is available
 *
 * Applications should call TVMDSPSetWeightsData() before TVMDSPParseConstants().
 * This function just verifies that weights data has been set.
 */
static int check_weights_data(void) {
    if (TVMDSPGetWeightsData() == nullptr || TVMDSPGetWeightsSize() == 0) {
        tvm_dsp_log("INFO: No weights data available\n");
        return -1;
    }
    return 0;
}

extern "C" {

/*!
 * \brief Parse constants from weights data (C-compatible API)
 *
 * This is the preferred API for all targets - avoids std::vector return issues.
 * Call this once to parse weights.bin. Subsequent calls return cached count.
 *
 * \return Number of constants parsed, or negative error code
 */
int TVMDSPParseConstants(void) {
    if (g_constants_parsed) {
        int count = 0;
        TVMDSPConstantsGet(&count);
        return count;
    }

    /* Verify weights data is available */
    if (check_weights_data() != 0) {
        tvm_dsp_log("ERROR: No weights data set. Call TVMDSPSetWeightsData() first.\n");
        return -1;
    }

    /* Parse weights using the DSP constants loader */
    int count = TVMDSPLoadConstants();
    if (count < 0) {
        tvm_dsp_log("ERROR: Failed to parse constants: %s\n",
                    TVMDSPConstantsErrorString(count));
        return count;
    }

    tvm_dsp_log("INFO: Parsed %d constants from weights.bin\n", count);
    g_constants_parsed = true;
    return count;
}

/*!
 * \brief Check if constants have been parsed
 * \return Non-zero if constants are ready
 */
int TVMDSPConstantsReady(void) {
    return g_constants_parsed ? 1 : 0;
}

/*!
 * \brief Reset constants loader state (for testing)
 */
void TVMDSPConstantsReset(void) {
    g_constants_parsed = false;
    TVMDSPConstantsCleanup();
}

}  /* extern "C" */

#ifndef TVM_DSP_TARGET_C66X
/*!
 * \brief Get model constants (C++ API, host emulation only)
 *
 * Returns vector of Any objects containing model's constant tensors.
 * Uses the DSP weights.bin parser for zero-copy loading.
 *
 * NOTE: On C66x, use TVMDSPParseConstants() and TVMDSPConstantsGet() instead
 * to avoid std::vector return value issues.
 */
std::vector<tvm::ffi::Any> TVMGetConstants() {
    static std::vector<tvm::ffi::Any> cached_constants;

    if (!cached_constants.empty()) {
        return cached_constants;
    }

    /* Use the C-compatible parser */
    int count = TVMDSPParseConstants();
    if (count < 0) {
        return cached_constants;
    }

    /* Build vector from parsed constants */
    cached_constants.reserve(count);
    for (int i = 0; i < count; i++) {
        TVMFFIAny* c = TVMDSPGetConstant(i);
        if (c) {
            /* Wrap TVMFFIAny in tvm::ffi::Any */
            cached_constants.push_back(tvm::ffi::Any(*c));
        }
    }

    return cached_constants;
}
#endif  /* !TVM_DSP_TARGET_C66X */
