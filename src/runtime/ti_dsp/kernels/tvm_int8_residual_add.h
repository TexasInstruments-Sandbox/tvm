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

#ifndef TVM_INT8_RESIDUAL_ADD_H_
#define TVM_INT8_RESIDUAL_ADD_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Int8 variant: saturation to [-128, 127]. */
int32_t tvm_int8_residual_add_relu(
    const void* x, const void* skip,
    const void* params, void* output,
    int32_t num_elements, int32_t has_relu);

/* Int16 variant: same params layout, saturation to [-32768, 32767].
 * Used by FuseInt16ResidualAdd for int16 PT2E quantized skip connections. */
int32_t tvm_int16_residual_add_relu(
    const void* x, const void* skip,
    const void* params, void* output,
    int32_t num_elements, int32_t has_relu);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_INT8_RESIDUAL_ADD_H_ */
