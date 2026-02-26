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
 * \file host_platform.h
 * \brief TVM DSP Runtime - Host (PC) Emulation Platform
 *
 * This configuration allows the DSP runtime to be compiled and
 * tested on a standard PC, enabling development and debugging
 * without DSP hardware.
 */
#ifndef TVM_RUNTIME_TI_DSP_PLATFORM_HOST_HOST_PLATFORM_H_
#define TVM_RUNTIME_TI_DSP_PLATFORM_HOST_HOST_PLATFORM_H_

/* Target identification */
#define TVM_DSP_TARGET_NAME "Host"
#define TVM_DSP_DEVICE_NAME "PC Emulation"
#define TVM_DSP_IS_C66X 0
#define TVM_DSP_IS_C7X 0
#define TVM_DSP_IS_HOST 1

/* Emulated memory configuration */
#define TVM_DSP_L2_BASE 0              /* Not used - malloc based */
#define TVM_DSP_L2_SIZE (4 * 1024 * 1024)   /* 4MB emulated "fast" memory */
#define TVM_DSP_L3_BASE 0              /* Not used - malloc based */
#define TVM_DSP_L3_SIZE (64 * 1024 * 1024)  /* 64MB emulated "main" memory */

/* Map generic pools (same as L2/L3 for host) */
#define TVM_DSP_MEM_FAST_BASE TVM_DSP_L2_BASE
#define TVM_DSP_MEM_FAST_SIZE TVM_DSP_L2_SIZE
#define TVM_DSP_MEM_MAIN_BASE TVM_DSP_L3_BASE
#define TVM_DSP_MEM_MAIN_SIZE TVM_DSP_L3_SIZE

/* Emulated clock frequency (for cycle simulation) */
#define TVM_DSP_CLOCK_MHZ 1000  /* Simulate 1GHz for easy ns conversion */

/* Alignment requirements (match DSP for testing consistency) */
#define TVM_DSP_CACHE_LINE_SIZE 64
#define TVM_DSP_SIMD_ALIGN 8
#define TVM_DSP_DEFAULT_ALIGN 64

/* Memory placement (no-op on host) */
#define TVM_DSP_FAST_DATA
#define TVM_DSP_MAIN_DATA
#define TVM_DSP_ALIGNED(n) __attribute__((aligned(n)))
#define TVM_DSP_RESTRICT __restrict__

/* Host has full features but we compile in restricted mode for testing */
#define TVM_DSP_NO_EXCEPTIONS 0
#define TVM_DSP_NO_THREADS 0
#define TVM_DSP_NO_DYNAMIC_LINKING 0

/* No intrinsics on host */
#define TVM_DSP_HAS_INTRINSICS 0

#endif /* TVM_RUNTIME_TI_DSP_PLATFORM_HOST_HOST_PLATFORM_H_ */
