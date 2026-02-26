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
 * \file constants/constants_c_api.c
 * \brief Pure C API implementation for constants loading
 */

#include "constants_c_api.h"
#include "constants.h"
#include "../platform/dsp_platform.h"

/*---------------------------------------------------------------------------
 * Static State
 *---------------------------------------------------------------------------*/

static const void* g_weights_data = NULL;
static size_t g_weights_size = 0;
static int g_constants_loaded = 0;

/*---------------------------------------------------------------------------
 * Weight Data Source
 *---------------------------------------------------------------------------*/

void TVMDSPSetWeightsData(const void* data, size_t size) {
  g_weights_data = data;
  g_weights_size = size;
  g_constants_loaded = 0;  /* Reset loaded flag */
}

const void* TVMDSPGetWeightsData(void) {
  return g_weights_data;
}

size_t TVMDSPGetWeightsSize(void) {
  return g_weights_size;
}

/*---------------------------------------------------------------------------
 * Constants Loading
 *---------------------------------------------------------------------------*/

int TVMDSPLoadConstants(void) {
  int count;

  if (g_constants_loaded) {
    /* Already loaded, return count */
    return TVMDSPConstantsCount();
  }

  /* Check if weights data is available */
  if (g_weights_data == NULL || g_weights_size == 0) {
#if defined(TVM_DSP_WEIGHTS_LINKER_EMBEDDED)
    /* Try to use linker-embedded weights */
    g_weights_data = _binary_weights_bin_start;
    g_weights_size = (size_t)(_binary_weights_bin_end - _binary_weights_bin_start);
    tvm_dsp_log("INFO: Using linker-embedded weights (%zu bytes)\n", g_weights_size);
#else
    tvm_dsp_log("INFO: No weights data available\n");
    TVMDSPConstantsInit();
    g_constants_loaded = 1;
    return 0;
#endif
  }

  /* Initialize and parse */
  TVMDSPConstantsInit();
  count = TVMDSPConstantsParse(g_weights_data, g_weights_size);

  if (count < 0) {
    tvm_dsp_log("ERROR: Failed to parse constants: %s\n",
                TVMDSPConstantsErrorString(count));
    return count;
  }

  g_constants_loaded = 1;
  return count;
}

TVMFFIAny* TVMDSPGetConstant(int index) {
  /* Load if needed */
  if (!g_constants_loaded) {
    if (TVMDSPLoadConstants() < 0) {
      return NULL;
    }
  }

  return TVMDSPConstantGetByIndex(index);
}

TVMFFIAny* TVMDSPGetAllConstants(int* count) {
  /* Load if needed */
  if (!g_constants_loaded) {
    if (TVMDSPLoadConstants() < 0) {
      if (count) *count = 0;
      return NULL;
    }
  }

  return TVMDSPConstantsGet(count);
}

int TVMDSPConstantsLoaded(void) {
  return g_constants_loaded;
}
