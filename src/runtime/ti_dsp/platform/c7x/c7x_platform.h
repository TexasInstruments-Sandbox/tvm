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
 * \file c7x_platform.h
 * \brief TVM DSP Runtime - C7x Platform Configuration
 *
 * C7x DSP Architecture Notes:
 * - Part of TI's latest DSP generation (C7000 family)
 * - Vector processing with 512-bit SIMD
 * - Unified memory architecture with L2 SRAM and DDR
 * - ARMv8-style MMU (differs from C66x)
 * - 128-byte cache lines
 */
#ifndef TVM_RUNTIME_TI_DSP_PLATFORM_C7X_C7X_PLATFORM_H_
#define TVM_RUNTIME_TI_DSP_PLATFORM_C7X_C7X_PLATFORM_H_

/* Target identification */
#define TVM_DSP_TARGET_NAME "C7x"
#define TVM_DSP_IS_C66X 0
#define TVM_DSP_IS_C7X 1
#define TVM_DSP_IS_HOST 0

/* Include device-specific configuration */
#if defined(TVM_DSP_DEVICE_J722S)
#include "j722s/j722s_config.h"
#else
/* Default C7x memory configuration (generic) */
#define TVM_DSP_DEVICE_NAME "Generic C7x"
#define TVM_DSP_CLOCK_MHZ 800        /* Default clock */
/* Fallback sizes for host emulation when no device is specified */
#ifndef __TI_COMPILER_VERSION__
#define TVM_DSP_L2_SIZE_FALLBACK (512 * 1024)      /* 512KB default */
#define TVM_DSP_DDR_SIZE_FALLBACK (64 * 1024 * 1024) /* 64MB default */
#endif
#endif

/*
 * Memory pool configuration
 *
 * The actual memory pool addresses and sizes are determined at runtime
 * from linker-defined symbols (__TVM_DSP_L2_HEAP_START, etc.).
 * This allows the memory layout to be configured in the linker command
 * file without recompiling the runtime library.
 *
 * See c7x_platform.c:tvm_dsp_platform_init() for initialization.
 */

/* Alignment requirements */
/* C7x has 128-byte cache lines (larger than C66x's 64-byte) */
#define TVM_DSP_CACHE_LINE_SIZE 128
/* C7x supports 512-bit SIMD = 64 bytes */
#define TVM_DSP_SIMD_ALIGN 64
/* Default alignment to cache line for optimal performance */
#define TVM_DSP_DEFAULT_ALIGN 128

/* Memory placement macros for TI compiler */
#ifdef __TI_COMPILER_VERSION__
#define TVM_DSP_FAST_DATA __attribute__((section(".bss.tvm_l2_heap")))
#define TVM_DSP_MAIN_DATA __attribute__((section(".bss.tvm_ddr_heap")))
#define TVM_DSP_ALIGNED(n) __attribute__((aligned(n)))
#define TVM_DSP_RESTRICT __restrict
#include <c7x.h>
#else
/* For non-TI compilers (e.g., clangd analysis) */
#define TVM_DSP_FAST_DATA
#define TVM_DSP_MAIN_DATA
#define TVM_DSP_ALIGNED(n) __attribute__((aligned(n)))
#define TVM_DSP_RESTRICT __restrict__
#endif

/* Runtime constraints for C7x */
#define TVM_DSP_NO_EXCEPTIONS 1
#define TVM_DSP_NO_THREADS 1
#define TVM_DSP_NO_DYNAMIC_LINKING 1

/* C7x-specific intrinsics availability */
#ifdef __TI_COMPILER_VERSION__
#define TVM_DSP_HAS_INTRINSICS 1
#else
#define TVM_DSP_HAS_INTRINSICS 0
#endif

#endif /* TVM_RUNTIME_TI_DSP_PLATFORM_C7X_C7X_PLATFORM_H_ */
