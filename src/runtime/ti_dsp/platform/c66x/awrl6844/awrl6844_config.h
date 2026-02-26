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
 * \file awrl6844_config.h
 * \brief TVM DSP Runtime - AWRL6844 Device Configuration
 *
 * TI AWRL6844 mmWave Radar Sensor Specifications:
 * - C66x DSP @ 450 MHz
 * - 384KB L2 SRAM (local DSP memory)
 * - 1.5MB L3 SRAM (shared memory)
 *
 * Memory Map (from MMWAVE-L-SDK-6):
 *   DSS_L2:       0x00800000 - 0x0085FFFF (384KB)
 *   DSS_L3:       0x88000000 - 0x8817FFFF (1.5MB)
 *
 * Note: Some L2/L3 regions may be reserved by SDK or used for
 * IPC/shared memory. The usable regions are defined below.
 */
#ifndef TVM_RUNTIME_TI_DSP_PLATFORM_C66X_AWRL6844_AWRL6844_CONFIG_H_
#define TVM_RUNTIME_TI_DSP_PLATFORM_C66X_AWRL6844_AWRL6844_CONFIG_H_

/* Device identification */
#define TVM_DSP_DEVICE_NAME "AWRL6844"

/* Clock frequency for cycle-to-time conversion */
#define TVM_DSP_CLOCK_MHZ 450

/*
 * L2 SRAM Configuration
 *
 * Total L2: 384KB (0x00800000 - 0x0085FFFF)
 * Reserved for code/stack: ~256KB
 * Available for heap: ~64KB in TEST_L2_HEAP region
 *
 * The TEST_L2_HEAP region is defined in the linker script at:
 *   0x00850000 - 0x0085FFFF (64KB)
 */
#define TVM_DSP_L2_BASE 0x00850000
#define TVM_DSP_L2_SIZE (64 * 1024) /* 64KB usable for runtime heap */

/*
 * L3 SRAM Configuration
 *
 * Total L3: 1.5MB (0x88000000 - 0x8817FFFF)
 * Reserved for IPC/SDK: ~320KB
 * Available for TEST_L3_HEAP: ~1MB
 *
 * The TEST_L3_HEAP region is defined in the linker script at:
 *   0x88050400 - 0x8814FFFF (~1MB)
 */
#define TVM_DSP_L3_BASE 0x88050400
#define TVM_DSP_L3_SIZE (1024 * 1024) /* 1MB usable for runtime heap */

/*
 * Stack and Heap Sizes (matches linker script)
 */
#define TVM_DSP_STACK_SIZE (64 * 1024) /* 64KB stack */
#define TVM_DSP_SYSMEM_SIZE (64 * 1024) /* 64KB standard heap (malloc) */

#endif /* TVM_RUNTIME_TI_DSP_PLATFORM_C66X_AWRL6844_AWRL6844_CONFIG_H_ */
