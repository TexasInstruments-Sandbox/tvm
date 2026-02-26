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
 * \file c66x_platform.h
 * \brief TVM DSP Runtime - C66x Platform Configuration
 */
#ifndef TVM_RUNTIME_TI_DSP_PLATFORM_C66X_C66X_PLATFORM_H_
#define TVM_RUNTIME_TI_DSP_PLATFORM_C66X_C66X_PLATFORM_H_

/* Target identification */
#define TVM_DSP_TARGET_NAME "C66x"
#define TVM_DSP_IS_C66X 1
#define TVM_DSP_IS_C7X 0
#define TVM_DSP_IS_HOST 0

/* Include device-specific configuration */
#if defined(TVM_DSP_DEVICE_AWRL6844)
#include "awrl6844/awrl6844_config.h"
#else
/* Default C66x memory configuration (generic) */
#define TVM_DSP_DEVICE_NAME "Generic C66x"
#define TVM_DSP_L2_BASE 0x00800000
#define TVM_DSP_L2_SIZE (256 * 1024) /* 256KB default */
#define TVM_DSP_L3_BASE 0x88000000
#define TVM_DSP_L3_SIZE (512 * 1024) /* 512KB default */
#define TVM_DSP_CLOCK_MHZ 600        /* Default clock */
#endif

/* Map generic pools to C66x memory */
#define TVM_DSP_MEM_FAST_BASE TVM_DSP_L2_BASE
#define TVM_DSP_MEM_FAST_SIZE TVM_DSP_L2_SIZE
#define TVM_DSP_MEM_MAIN_BASE TVM_DSP_L3_BASE
#define TVM_DSP_MEM_MAIN_SIZE TVM_DSP_L3_SIZE

/* Alignment requirements */
#define TVM_DSP_CACHE_LINE_SIZE 64 /* C66x cache line */
#define TVM_DSP_SIMD_ALIGN 8       /* 64-bit SIMD minimum */
#define TVM_DSP_DEFAULT_ALIGN 64   /* Cache line aligned */

/* Memory placement macros for TI compiler */
#ifdef __TI_COMPILER_VERSION__
#define TVM_DSP_FAST_DATA __attribute__((section(".bss.test_heap_l2")))
#define TVM_DSP_MAIN_DATA __attribute__((section(".bss.test_heap_l3")))
#define TVM_DSP_ALIGNED(n) __attribute__((aligned(n)))
#define TVM_DSP_RESTRICT __restrict
#include <c6x.h>
#else
/* For non-TI compilers (e.g., clangd analysis) */
#define TVM_DSP_FAST_DATA
#define TVM_DSP_MAIN_DATA
#define TVM_DSP_ALIGNED(n) __attribute__((aligned(n)))
#define TVM_DSP_RESTRICT __restrict__
#endif

/* Runtime constraints for C66x */
#define TVM_DSP_NO_EXCEPTIONS 1
#define TVM_DSP_NO_THREADS 1
#define TVM_DSP_NO_DYNAMIC_LINKING 1

/* C66x-specific intrinsics availability */
#ifdef __TI_COMPILER_VERSION__
#define TVM_DSP_HAS_INTRINSICS 1
#else
#define TVM_DSP_HAS_INTRINSICS 0
#endif

#endif /* TVM_RUNTIME_TI_DSP_PLATFORM_C66X_C66X_PLATFORM_H_ */
