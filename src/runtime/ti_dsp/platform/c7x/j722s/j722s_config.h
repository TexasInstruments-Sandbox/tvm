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
 * \file j722s_config.h
 * \brief TVM DSP Runtime - J722S Device Configuration
 *
 * TI J722S SoC C75 DSP Core Specifications:
 * - C75 DSP (C7524 core) @ 800 MHz
 * - 2MB L2 SRAM (local DSP memory) at 0x7E000000
 * - DDR access for large data buffers
 *
 * Memory Map (from MCU+ SDK hello_world_standalone linker.cmd):
 *
 *   L2 SRAM:      0x7E000000 - 0x7E1FFFFF (2MB total)
 *     - Vectors:  0x7E000000 - 0x7E003FFF (16KB)
 *     - Code:     0x7E009000 - 0x7E108FFF (1MB)
 *     - Data:     0x7E109000 - 0x7E188FFF (512KB)
 *     - Stack:    0x7E189000 - 0x7E19CFFF (80KB)
 *     - Heap:     0x7E19D000 - 0x7E1BCFFF (128KB) <- TVM L2 pool
 *     - Reserved: 0x7E1BD000 - 0x7E1FFFFF
 *
 *   DDR (Standalone Mode):
 *     - DDR_C7X:  0xAD604000 - 0xAFFFFFFF (~58MB) <- TVM DDR pool
 *
 * Note: The actual usable heap sizes may vary depending on
 * code/data size. These values are based on the standalone
 * example linker configuration.
 */
#ifndef TVM_RUNTIME_TI_DSP_PLATFORM_C7X_J722S_J722S_CONFIG_H_
#define TVM_RUNTIME_TI_DSP_PLATFORM_C7X_J722S_J722S_CONFIG_H_

/* Device identification */
#define TVM_DSP_DEVICE_NAME "J722S_C75"

/* Clock frequency for cycle-to-time conversion */
#define TVM_DSP_CLOCK_MHZ 800

/*
 * Linker-defined symbols for heap regions
 *
 * The heap bounds are defined in the linker command file (linker_c7x.cmd)
 * to allow flexible memory layout without recompiling the runtime library.
 *
 * Required linker symbols:
 *   __TVM_DSP_L2_HEAP_START, __TVM_DSP_L2_HEAP_END
 *   __TVM_DSP_DDR_HEAP_START, __TVM_DSP_DDR_HEAP_END
 *
 * Example linker command file definitions:
 *   __TVM_DSP_L2_HEAP_START = 0x7E1BD000;  // L2_SCRATCH region
 *   __TVM_DSP_L2_HEAP_END   = 0x7E1FFFFF;
 *   __TVM_DSP_DDR_HEAP_START = 0xAD700000;
 *   __TVM_DSP_DDR_HEAP_END   = 0xB0FFFFFF;
 */
#ifdef __TI_COMPILER_VERSION__
extern char __TVM_DSP_L2_HEAP_START[];
extern char __TVM_DSP_L2_HEAP_END[];
extern char __TVM_DSP_DDR_HEAP_START[];
extern char __TVM_DSP_DDR_HEAP_END[];
#endif

/*
 * Non-TI compilers (host emulation) have no linker-provided heap size, so
 * these must come from the build system (see CMakeLists.txt's c7x_host
 * target) instead of a silently-stale default here.
 */
#ifndef __TI_COMPILER_VERSION__
#ifndef TVM_DSP_L2_SIZE_FALLBACK
#error "TVM_DSP_L2_SIZE_FALLBACK must be defined by the build system for host emulation"
#endif
#ifndef TVM_DSP_DDR_SIZE_FALLBACK
#error "TVM_DSP_DDR_SIZE_FALLBACK must be defined by the build system for host emulation"
#endif
#endif

#endif /* TVM_RUNTIME_TI_DSP_PLATFORM_C7X_J722S_J722S_CONFIG_H_ */
