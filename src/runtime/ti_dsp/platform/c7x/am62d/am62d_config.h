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
 * \file am62d_config.h
 * \brief TVM DSP Runtime - AM62D Device Configuration
 *
 * TI AM62D SoC C75 DSP Core Specifications:
 * - C75 DSP (C7504 core) @ 800 MHz
 * - L2 SRAM: MAIN bank (1MB at 0x7E000000) + AUX bank (240KB at 0x7F000000)
 * - DDR carveout identity-mapped (virtual == physical), no MMU translation
 *
 * Memory Map (see firmware/c7x/dsp/configs/linker_am62d_c75_freertos.cmd):
 *
 *   L2 MAIN:      0x7E000000 - 0x7E0FFFFF (1MB)  <- not used by TVM pools
 *   L2 AUX:       0x7F000000 - 0x7F03BFFF (240KB) <- TVM L2 heap (pool and
 *                                                    generated-code L2 base)
 *   DDR:          0xB2000000 - 0xB8FFFFFF (112MB) <- TVM DDR pool
 *
 * Only the L2 and DDR pools are provided here (the same two pools as every
 * other device); the linker exports the same symbols as on J722S.
 */
#ifndef TVM_RUNTIME_TI_DSP_PLATFORM_C7X_AM62D_AM62D_CONFIG_H_
#define TVM_RUNTIME_TI_DSP_PLATFORM_C7X_AM62D_AM62D_CONFIG_H_

/* Device identification */
#define TVM_DSP_DEVICE_NAME "AM62D_C75"

/* Clock frequency for cycle-to-time conversion */
#define TVM_DSP_CLOCK_MHZ 800

/*
 * Linker-defined symbols for heap regions.  Required linker symbols:
 *   __TVM_DSP_L2_HEAP_START, __TVM_DSP_L2_HEAP_END
 *   __TVM_DSP_DDR_HEAP_START, __TVM_DSP_DDR_HEAP_END
 */
#ifdef __TI_COMPILER_VERSION__
extern char __TVM_DSP_L2_HEAP_START[];
extern char __TVM_DSP_L2_HEAP_END[];
extern char __TVM_DSP_DDR_HEAP_START[];
extern char __TVM_DSP_DDR_HEAP_END[];
#endif

/*
 * Non-TI compilers (host emulation) have no linker-provided heap size, so
 * these must come from the build system.
 */
#ifndef __TI_COMPILER_VERSION__
#ifndef TVM_DSP_L2_SIZE_FALLBACK
#error "TVM_DSP_L2_SIZE_FALLBACK must be defined by the build system for host emulation"
#endif
#ifndef TVM_DSP_DDR_SIZE_FALLBACK
#error "TVM_DSP_DDR_SIZE_FALLBACK must be defined by the build system for host emulation"
#endif
#endif

#endif /* TVM_RUNTIME_TI_DSP_PLATFORM_C7X_AM62D_AM62D_CONFIG_H_ */
