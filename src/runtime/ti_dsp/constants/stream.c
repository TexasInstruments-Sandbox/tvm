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
 * \file constants/stream.c
 * \brief Sequential binary stream reader implementation
 */

#include "stream.h"
#include <string.h>

void TVMDSPStreamInit(TVMDSPStream* stream, const void* data, size_t size) {
  stream->data = (const uint8_t*)data;
  stream->size = size;
  stream->pos = 0;
}

int TVMDSPStreamRead(TVMDSPStream* stream, void* buf, size_t size) {
  if (stream->pos + size > stream->size) {
    return -1;  /* Would read past end */
  }

  memcpy(buf, stream->data + stream->pos, size);
  stream->pos += size;
  return 0;
}

const void* TVMDSPStreamPeek(TVMDSPStream* stream, size_t size) {
  if (stream->pos + size > stream->size) {
    return NULL;  /* Would read past end */
  }

  return stream->data + stream->pos;
}

int TVMDSPStreamSkip(TVMDSPStream* stream, size_t size) {
  if (stream->pos + size > stream->size) {
    return -1;  /* Would skip past end */
  }

  stream->pos += size;
  return 0;
}

size_t TVMDSPStreamRemaining(const TVMDSPStream* stream) {
  return stream->size - stream->pos;
}

int TVMDSPStreamAtEnd(const TVMDSPStream* stream) {
  return stream->pos >= stream->size;
}

size_t TVMDSPStreamPosition(const TVMDSPStream* stream) {
  return stream->pos;
}

int TVMDSPStreamAlign(TVMDSPStream* stream, size_t alignment) {
  size_t current = stream->pos;
  size_t aligned = (current + alignment - 1) & ~(alignment - 1);
  size_t skip = aligned - current;

  if (skip > 0) {
    if (aligned > stream->size) {
      return -1;  /* Would align past end */
    }
    stream->pos = aligned;
  }

  return 0;
}
