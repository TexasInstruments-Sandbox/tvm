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
 * \file io/tensor_file.cpp
 * \brief Binary tensor file I/O implementation
 */

#include "tensor_file.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Maximum number of tensors in a single file */
#define MAX_TENSORS_PER_FILE 256

OwnedNDArray** ReadTensorsFromFile(const char* filename, int* num_tensors) {
  FILE* f = nullptr;
  OwnedNDArray** tensors = nullptr;
  uint32_t magic, version, count;
  int32_t ndim, dtype_code, dtype_bits;
  int64_t data_size;
  int64_t shape[TENSOR_MAX_NDIM];
  uint32_t i;

  *num_tensors = 0;

  f = fopen(filename, "rb");
  if (!f) {
    printf("ERROR: Failed to open file: %s\n", filename);
    return nullptr;
  }

  /* Read and validate file header */
  if (fread(&magic, sizeof(uint32_t), 1, f) != 1) {
    printf("ERROR: Failed to read magic number\n");
    goto error;
  }
  if (magic != TENSOR_FILE_MAGIC) {
    printf("ERROR: Invalid magic number: 0x%08X (expected 0x%08X)\n",
           magic, TENSOR_FILE_MAGIC);
    goto error;
  }

  if (fread(&version, sizeof(uint32_t), 1, f) != 1) {
    printf("ERROR: Failed to read version\n");
    goto error;
  }
  if (version != TENSOR_FILE_VERSION) {
    printf("ERROR: Unsupported version: %u (expected %u)\n",
           version, TENSOR_FILE_VERSION);
    goto error;
  }

  if (fread(&count, sizeof(uint32_t), 1, f) != 1) {
    printf("ERROR: Failed to read tensor count\n");
    goto error;
  }
  if (count > MAX_TENSORS_PER_FILE) {
    printf("ERROR: Too many tensors: %u (max %d)\n", count, MAX_TENSORS_PER_FILE);
    goto error;
  }

  /* Allocate array for tensor pointers */
  tensors = static_cast<OwnedNDArray**>(malloc(count * sizeof(OwnedNDArray*)));
  if (!tensors) {
    printf("ERROR: Failed to allocate tensor array\n");
    goto error;
  }
  memset(tensors, 0, count * sizeof(OwnedNDArray*));

  /* Read each tensor */
  for (i = 0; i < count; i++) {
    /* Read tensor header */
    if (fread(&ndim, sizeof(int32_t), 1, f) != 1) {
      printf("ERROR: Failed to read ndim for tensor %u\n", i);
      goto error;
    }
    if (ndim < 0 || ndim > TENSOR_MAX_NDIM) {
      printf("ERROR: Invalid ndim: %d (max %d)\n", ndim, TENSOR_MAX_NDIM);
      goto error;
    }

    if (fread(shape, sizeof(int64_t), static_cast<size_t>(ndim), f) !=
        static_cast<size_t>(ndim)) {
      printf("ERROR: Failed to read shape for tensor %u\n", i);
      goto error;
    }

    if (fread(&dtype_code, sizeof(int32_t), 1, f) != 1) {
      printf("ERROR: Failed to read dtype_code for tensor %u\n", i);
      goto error;
    }
    if (fread(&dtype_bits, sizeof(int32_t), 1, f) != 1) {
      printf("ERROR: Failed to read dtype_bits for tensor %u\n", i);
      goto error;
    }
    if (fread(&data_size, sizeof(int64_t), 1, f) != 1) {
      printf("ERROR: Failed to read data_size for tensor %u\n", i);
      goto error;
    }

    /* Allocate OwnedNDArray */
    tensors[i] = static_cast<OwnedNDArray*>(malloc(sizeof(OwnedNDArray)));
    if (!tensors[i]) {
      printf("ERROR: Failed to allocate OwnedNDArray for tensor %u\n", i);
      goto error;
    }

    /* Initialize */
    tensors[i]->ndim = ndim;
    tensors[i]->dtype.code = static_cast<uint8_t>(dtype_code);
    tensors[i]->dtype.bits = static_cast<uint8_t>(dtype_bits);
    tensors[i]->dtype.lanes = 1;

    /* Allocate and copy shape */
    tensors[i]->shape = static_cast<int64_t*>(malloc(ndim * sizeof(int64_t)));
    if (!tensors[i]->shape) {
      printf("ERROR: Failed to allocate shape for tensor %u\n", i);
      goto error;
    }
    memcpy(tensors[i]->shape, shape, ndim * sizeof(int64_t));

    /* Allocate data */
    tensors[i]->data = malloc(static_cast<size_t>(data_size));
    if (!tensors[i]->data) {
      printf("ERROR: Failed to allocate data for tensor %u\n", i);
      goto error;
    }

    /* Read data */
    size_t bytes_read = fread(tensors[i]->data, 1, static_cast<size_t>(data_size), f);
    if (bytes_read != static_cast<size_t>(data_size)) {
      printf("ERROR: Failed to read data for tensor %u (got %zu, expected %lld)\n",
             i, bytes_read, static_cast<long long>(data_size));
      goto error;
    }
  }

  fclose(f);
  *num_tensors = static_cast<int>(count);
  return tensors;

error:
  if (f) fclose(f);
  if (tensors) {
    FreeTensors(tensors, static_cast<int>(count));
  }
  return nullptr;
}

int WriteTensorsToFile(const char* filename,
                       OwnedNDArray** tensors,
                       int num_tensors) {
  FILE* f = nullptr;
  uint32_t magic = TENSOR_FILE_MAGIC;
  uint32_t version = TENSOR_FILE_VERSION;
  uint32_t count = static_cast<uint32_t>(num_tensors);
  int32_t dtype_code, dtype_bits;
  int64_t data_size;

  f = fopen(filename, "wb");
  if (!f) {
    printf("ERROR: Failed to open file for writing: %s\n", filename);
    return -1;
  }

  /* Write file header */
  if (fwrite(&magic, sizeof(uint32_t), 1, f) != 1) goto error;
  if (fwrite(&version, sizeof(uint32_t), 1, f) != 1) goto error;
  if (fwrite(&count, sizeof(uint32_t), 1, f) != 1) goto error;

  /* Write each tensor */
  for (int i = 0; i < num_tensors; i++) {
    OwnedNDArray* arr = tensors[i];
    if (!arr) {
      printf("ERROR: NULL tensor at index %d\n", i);
      goto error;
    }

    /* Write tensor header */
    if (fwrite(&arr->ndim, sizeof(int32_t), 1, f) != 1) goto error;
    if (fwrite(arr->shape, sizeof(int64_t), static_cast<size_t>(arr->ndim), f) !=
        static_cast<size_t>(arr->ndim)) {
      goto error;
    }

    dtype_code = static_cast<int32_t>(arr->dtype.code);
    dtype_bits = static_cast<int32_t>(arr->dtype.bits);
    if (fwrite(&dtype_code, sizeof(int32_t), 1, f) != 1) goto error;
    if (fwrite(&dtype_bits, sizeof(int32_t), 1, f) != 1) goto error;

    /* Calculate and write data size */
    data_size = static_cast<int64_t>(arr->SizeBytes());
    if (fwrite(&data_size, sizeof(int64_t), 1, f) != 1) goto error;

    /* Write data */
    size_t written = fwrite(arr->data, 1, static_cast<size_t>(data_size), f);
    if (written != static_cast<size_t>(data_size)) {
      printf("ERROR: Failed to write data for tensor %d (wrote %zu of %lld)\n",
             i, written, static_cast<long long>(data_size));
      goto error;
    }
  }

  fclose(f);
  return 0;

error:
  if (f) fclose(f);
  return -1;
}

int WriteNDArraysToFile(const char* filename,
                        tvm::dsp::NDArray** tensors,
                        int num_tensors) {
  FILE* f = nullptr;
  uint32_t magic = TENSOR_FILE_MAGIC;
  uint32_t version = TENSOR_FILE_VERSION;
  uint32_t count = static_cast<uint32_t>(num_tensors);
  int32_t dtype_code, dtype_bits;
  int64_t data_size;

  f = fopen(filename, "wb");
  if (!f) {
    printf("ERROR: Failed to open file for writing: %s\n", filename);
    return -1;
  }

  /* Write file header */
  if (fwrite(&magic, sizeof(uint32_t), 1, f) != 1) goto error;
  if (fwrite(&version, sizeof(uint32_t), 1, f) != 1) goto error;
  if (fwrite(&count, sizeof(uint32_t), 1, f) != 1) goto error;

  /* Write each tensor */
  for (int i = 0; i < num_tensors; i++) {
    tvm::dsp::NDArray* arr = tensors[i];
    if (!arr) {
      printf("ERROR: NULL tensor at index %d\n", i);
      goto error;
    }

    /* Write tensor header */
    if (fwrite(&arr->ndim, sizeof(int32_t), 1, f) != 1) goto error;
    if (fwrite(arr->shape, sizeof(int64_t), static_cast<size_t>(arr->ndim), f) !=
        static_cast<size_t>(arr->ndim)) {
      goto error;
    }

    dtype_code = static_cast<int32_t>(arr->dtype.code);
    dtype_bits = static_cast<int32_t>(arr->dtype.bits);
    if (fwrite(&dtype_code, sizeof(int32_t), 1, f) != 1) goto error;
    if (fwrite(&dtype_bits, sizeof(int32_t), 1, f) != 1) goto error;

    /* Calculate and write data size */
    data_size = static_cast<int64_t>(arr->SizeBytes());
    if (fwrite(&data_size, sizeof(int64_t), 1, f) != 1) goto error;

    /* Write data */
    size_t written = fwrite(arr->data, 1, static_cast<size_t>(data_size), f);
    if (written != static_cast<size_t>(data_size)) {
      printf("ERROR: Failed to write data for tensor %d (wrote %zu of %lld)\n",
             i, written, static_cast<long long>(data_size));
      goto error;
    }
  }

  fclose(f);
  return 0;

error:
  if (f) fclose(f);
  return -1;
}

void FreeTensors(OwnedNDArray** tensors, int num_tensors) {
  if (!tensors) return;

  for (int i = 0; i < num_tensors; i++) {
    if (tensors[i]) {
      if (tensors[i]->data) free(tensors[i]->data);
      if (tensors[i]->shape) free(tensors[i]->shape);
      free(tensors[i]);
    }
  }

  free(tensors);
}
