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
 * \file constants/stream.h
 * \brief Sequential binary stream reader for parsing weights.bin
 *
 * This provides a simple sequential reader for parsing TVM's binary
 * serialization format without any dynamic memory allocation.
 */

#ifndef TVM_RUNTIME_TI_DSP_CONSTANTS_STREAM_H_
#define TVM_RUNTIME_TI_DSP_CONSTANTS_STREAM_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Sequential stream reader for binary data
 *
 * This structure maintains read position over a fixed memory buffer.
 * All reads are sequential - no seeking supported.
 */
typedef struct {
  const uint8_t* data;  /*!< Base pointer to data */
  size_t size;          /*!< Total size of data */
  size_t pos;           /*!< Current read position */
} TVMDSPStream;

/*!
 * \brief Initialize a stream reader
 * \param stream Stream to initialize
 * \param data Pointer to binary data
 * \param size Size of binary data in bytes
 */
void TVMDSPStreamInit(TVMDSPStream* stream, const void* data, size_t size);

/*!
 * \brief Read bytes from stream into buffer
 * \param stream Stream to read from
 * \param buf Buffer to read into
 * \param size Number of bytes to read
 * \return 0 on success, -1 if would read past end
 */
int TVMDSPStreamRead(TVMDSPStream* stream, void* buf, size_t size);

/*!
 * \brief Get pointer to current position (zero-copy read)
 *
 * Returns a pointer to the current position in the stream without
 * copying any data. The stream position is NOT advanced.
 *
 * \param stream Stream to read from
 * \param size Number of bytes caller needs access to
 * \return Pointer to data, or NULL if would read past end
 */
const void* TVMDSPStreamPeek(TVMDSPStream* stream, size_t size);

/*!
 * \brief Skip bytes in stream without reading
 * \param stream Stream to read from
 * \param size Number of bytes to skip
 * \return 0 on success, -1 if would skip past end
 */
int TVMDSPStreamSkip(TVMDSPStream* stream, size_t size);

/*!
 * \brief Get remaining bytes in stream
 * \param stream Stream to query
 * \return Number of bytes remaining
 */
size_t TVMDSPStreamRemaining(const TVMDSPStream* stream);

/*!
 * \brief Check if stream has reached end
 * \param stream Stream to query
 * \return 1 if at end, 0 otherwise
 */
int TVMDSPStreamAtEnd(const TVMDSPStream* stream);

/*!
 * \brief Get current stream position
 * \param stream Stream to query
 * \return Current position in bytes from start
 */
size_t TVMDSPStreamPosition(const TVMDSPStream* stream);

/*!
 * \brief Align stream position to specified boundary
 *
 * Advances the stream position to the next multiple of alignment.
 * If already aligned, position is unchanged.
 *
 * \param stream Stream to align
 * \param alignment Alignment boundary (must be power of 2)
 * \return 0 on success, -1 if would align past end
 */
int TVMDSPStreamAlign(TVMDSPStream* stream, size_t alignment);

/* Convenience macros for reading specific types */

/*!
 * \brief Read a uint64_t from stream
 * \param stream Stream to read from
 * \param val Pointer to value to read into
 * \return 0 on success, -1 on error
 */
static inline int TVMDSPStreamReadU64(TVMDSPStream* stream, uint64_t* val) {
  return TVMDSPStreamRead(stream, val, sizeof(uint64_t));
}

/*!
 * \brief Read an int64_t from stream
 * \param stream Stream to read from
 * \param val Pointer to value to read into
 * \return 0 on success, -1 on error
 */
static inline int TVMDSPStreamReadI64(TVMDSPStream* stream, int64_t* val) {
  return TVMDSPStreamRead(stream, val, sizeof(int64_t));
}

/*!
 * \brief Read a uint32_t from stream
 * \param stream Stream to read from
 * \param val Pointer to value to read into
 * \return 0 on success, -1 on error
 */
static inline int TVMDSPStreamReadU32(TVMDSPStream* stream, uint32_t* val) {
  return TVMDSPStreamRead(stream, val, sizeof(uint32_t));
}

/*!
 * \brief Read an int32_t from stream
 * \param stream Stream to read from
 * \param val Pointer to value to read into
 * \return 0 on success, -1 on error
 */
static inline int TVMDSPStreamReadI32(TVMDSPStream* stream, int32_t* val) {
  return TVMDSPStreamRead(stream, val, sizeof(int32_t));
}

/*!
 * \brief Read a double from stream
 * \param stream Stream to read from
 * \param val Pointer to value to read into
 * \return 0 on success, -1 on error
 */
static inline int TVMDSPStreamReadF64(TVMDSPStream* stream, double* val) {
  return TVMDSPStreamRead(stream, val, sizeof(double));
}

#ifdef __cplusplus
}
#endif

#endif  /* TVM_RUNTIME_TI_DSP_CONSTANTS_STREAM_H_ */
