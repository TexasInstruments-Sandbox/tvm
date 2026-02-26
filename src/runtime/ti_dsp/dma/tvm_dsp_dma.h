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
 * \file tvm_dsp_dma.h
 * \brief TVM DSP Runtime - DMA Transfer Interface
 *
 * Provides 1D DMA copy and wait primitives for moving data between
 * DDR and L2 SRAM on TI C7x DSP.  The API matches the TVM TIR
 * builtin::dma_copy / builtin::dma_wait signatures so that
 * LowerDMAToExtern can directly lower to call_extern of these
 * functions.
 *
 * Target (C7x): asynchronous DMA via TI DmaUtilsAutoInc3d (EDMA/DRU).
 * Host emulation: synchronous memcpy (no overlap, but correct behavior).
 */
#ifndef TVM_RUNTIME_TI_DSP_DMA_TVM_DSP_DMA_H_
#define TVM_RUNTIME_TI_DSP_DMA_TVM_DSP_DMA_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Initialize the DMA subsystem.
 *
 * On C7x hardware, this sets up the standalone UDMA driver and
 * DmaUtilsAutoInc3d with the requested number of channels.
 * On host emulation builds, this is a no-op.
 *
 * Must be called once at startup before any tvm_dsp_dma_copy calls.
 *
 * \param num_channels  Number of DMA channels (queue IDs 0..n-1)
 * \return 0 on success, nonzero on failure
 */
int tvm_dsp_dma_init(int num_channels);

/*!
 * \brief Deinitialize the DMA subsystem.
 *
 * Deconfigures channels, tears down DmaUtilsAutoInc3d and UDMA driver.
 * On host emulation builds, this is a no-op.
 */
void tvm_dsp_dma_deinit(void);

/*!
 * \brief Asynchronous 1D DMA copy (DDR <-> L2).
 *
 * Initiates a DMA transfer of \p size bytes from \p src to \p dst.
 * The transfer is associated with the given \p queue_id for later
 * synchronization via tvm_dsp_dma_wait().
 *
 * On C7x hardware, submits an async EDMA transfer via DRU direct TR.
 * On host emulation, performs synchronous memcpy.
 *
 * \param queue_id      DMA queue/channel identifier (from TIR)
 * \param dst           Destination address
 * \param src           Source address
 * \param size          Number of bytes to transfer
 * \param bypass_cache  If nonzero, bypass cache (reserved)
 * \return 0 on success
 */
int tvm_dsp_dma_copy(int queue_id, void* dst, const void* src,
                     int size, int bypass_cache);

/*!
 * \brief Wait for in-flight DMA transfers to complete.
 *
 * Blocks until the number of in-flight DMA transfers on the given
 * queue is at most \p max_inflight.
 *
 * On C7x hardware, waits for EDMA completion.
 * On host emulation, returns immediately (transfers are synchronous).
 *
 * \param queue_id      DMA queue/channel identifier
 * \param max_inflight  Maximum allowed in-flight transfers
 * \return 0 on success
 */
int tvm_dsp_dma_wait(int queue_id, int max_inflight);

#ifdef __cplusplus
}
#endif

#endif  /* TVM_RUNTIME_TI_DSP_DMA_TVM_DSP_DMA_H_ */
