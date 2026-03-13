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
 * \file dsp_cpp/main_dsp.cpp
 * \brief DSP main entry point using C++14 Model API
 */

#include <cstdio>

#include "include/model.h"
#include "io/tensor_file.h"
#include "io/weights_loader.h"

using namespace tvm::dsp;

void print_ndarray_info(const char* name, NDArray* arr) {
  if (!arr || !arr->data) {
    printf("%s: (null)\n", name);
    return;
  }
  printf("%s: shape=[", name);
  for (int i = 0; i < arr->ndim; i++) {
    printf("%lld%s", (long long)arr->shape[i], i < arr->ndim - 1 ? "," : "");
  }
  printf("], dtype=%d.%d\n", arr->dtype.code, arr->dtype.bits);
}

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;

  /* Load weights */
  size_t weights_size;
  const void* weights_data = GetWeightsData(&weights_size);
  if (!weights_data) {
    printf("ERROR: Failed to load weights\n");
    return 1;
  }

  /* Load model */
  Model model;
  ModelError err = model.Load(weights_data, weights_size);
  if (err != ModelError::kSuccess) {
    printf("ERROR: Model load failed (%d)\n", static_cast<int>(err));
    return 1;
  }
  printf("Loaded %d constants\n", model.ConstantCount());

  /* Read input(s) */
  int num_inputs = 0;
  OwnedNDArray** input_tensors = ReadTensorsFromFile("input.bin", &num_inputs);
  if (!input_tensors || num_inputs < 1) {
    printf("ERROR: Failed to read input.bin\n");
    return 1;
  }

  /* Create NDArray views for all inputs */
  NDArray inputs[8];
  for (int i = 0; i < num_inputs && i < 8; i++) {
    inputs[i] = input_tensors[i]->AsView();
    char name[32];
    snprintf(name, sizeof(name), "Input[%d]", i);
    print_ndarray_info(name, &inputs[i]);
  }

  /* Run inference (multi-input, multi-output) */
  NDArray* outputs[8];
  int num_outputs = 0;
  err = model.InferMulti(inputs, num_inputs, outputs, &num_outputs);
  if (err != ModelError::kSuccess) {
    printf("ERROR: Inference failed (%d)\n", static_cast<int>(err));
    FreeTensors(input_tensors, num_inputs);
    return 1;
  }

  printf("Cycles: %llu\n", (unsigned long long)model.LastInferenceCycles());
  printf("Num outputs: %d\n", num_outputs);

  for (int i = 0; i < num_outputs; i++) {
    char name[32];
    snprintf(name, sizeof(name), "Output[%d]", i);
    print_ndarray_info(name, outputs[i]);
  }

  /* Write all outputs */
  if (WriteNDArraysToFile("output.bin", outputs, num_outputs) != 0) {
    printf("WARNING: Failed to write output.bin\n");
  }

  /* Memory stats */
  MemoryStats l2 = model.GetMemoryStats(MemoryPool::kFast);
  MemoryStats l3 = model.GetMemoryStats(MemoryPool::kMain);
  printf("Memory: L2 peak=%u, L3 peak=%u\n",
         (unsigned int)l2.peak_used, (unsigned int)l3.peak_used);

  FreeTensors(input_tensors, num_inputs);
  printf("Done\n");

  return 0;
}
