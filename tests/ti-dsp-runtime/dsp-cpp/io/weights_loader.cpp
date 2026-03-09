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
 * \file io/weights_loader.cpp
 * \brief Weights loading implementation for DSP applications
 *
 * This file loads weights from either:
 * - TVM_DSP_WEIGHTS_EMBEDDED: Linker-embedded weights
 * - TVM_DSP_WEIGHTS_PATH: Filesystem (host emulation)
 */

#include "weights_loader.h"

#include <cstdio>
#include <cstdlib>

extern "C" {
#include "platform/dsp_platform.h"
}

/*---------------------------------------------------------------------------
 * Weight Data Symbols (for linker-embedded weights)
 *---------------------------------------------------------------------------*/

#if defined(TVM_DSP_WEIGHTS_EMBEDDED)
extern "C" const char _binary_weights_bin_start[];
extern "C" const unsigned int _binary_weights_bin_size;
#endif

/* Static buffer for filesystem-loaded weights */
static char* g_weights_buffer = nullptr;
static size_t g_weights_size = 0;

const void* GetWeightsData(size_t* size) {
#if defined(TVM_DSP_WEIGHTS_EMBEDDED)
  /* Use linker-embedded weights */
  *size = _binary_weights_bin_size;
  tvm_dsp_log("INFO: Using embedded weights (%zu bytes)\n", *size);
  return _binary_weights_bin_start;

#elif defined(TVM_DSP_WEIGHTS_PATH)
  /* Load from filesystem (only load once) */
  if (g_weights_buffer != nullptr) {
    *size = g_weights_size;
    return g_weights_buffer;
  }

  FILE* f = fopen(TVM_DSP_WEIGHTS_PATH, "rb");
  if (!f) {
    tvm_dsp_log("ERROR: Cannot open weights file: %s\n", TVM_DSP_WEIGHTS_PATH);
    *size = 0;
    return nullptr;
  }

  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 0, SEEK_SET);

  g_weights_buffer = static_cast<char*>(malloc(file_size));
  if (!g_weights_buffer) {
    fclose(f);
    tvm_dsp_log("ERROR: Failed to allocate %ld bytes for weights\n", file_size);
    *size = 0;
    return nullptr;
  }

  if (fread(g_weights_buffer, 1, file_size, f) != static_cast<size_t>(file_size)) {
    fclose(f);
    free(g_weights_buffer);
    g_weights_buffer = nullptr;
    tvm_dsp_log("ERROR: Failed to read weights file\n");
    *size = 0;
    return nullptr;
  }
  fclose(f);

  g_weights_size = static_cast<size_t>(file_size);
  tvm_dsp_log("INFO: Loaded weights from %s (%ld bytes)\n", TVM_DSP_WEIGHTS_PATH, file_size);

  *size = g_weights_size;
  return g_weights_buffer;

#else
  /* No weights source configured */
  tvm_dsp_log("ERROR: No weights source configured\n");
  *size = 0;
  return nullptr;
#endif
}

void FreeWeightsBuffer() {
  if (g_weights_buffer != nullptr) {
    free(g_weights_buffer);
    g_weights_buffer = nullptr;
    g_weights_size = 0;
  }
}
