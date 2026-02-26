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
 * \file object.c
 * \brief TVM DSP Runtime - Object allocation implementation
 */

#include "object.h"
#include "../core/logging.h"

/*---------------------------------------------------------------------------
 * Object Allocation
 *---------------------------------------------------------------------------*/

TVMFFIObject* TVMDSPObjectAlloc(size_t size, int32_t type_index,
                                 void (*deleter)(TVMFFIObject*)) {
  /* Allocate from main memory pool with cache-line alignment */
  TVMFFIObject* obj = (TVMFFIObject*)tvm_dsp_alloc(
      size, TVM_DSP_DEFAULT_ALIGN, TVM_DSP_MEM_MAIN);

  if (obj == NULL) {
    TVM_DSP_LOGE("Failed to allocate object of size %zu (type=%d)\n",
                 size, type_index);
    return NULL;
  }

  /* Initialize header */
  obj->type_index = type_index;
  obj->ref_counter = 1;  /* Start with refcount of 1 */
  obj->deleter = deleter;

  TVM_DSP_LOGD("Allocated object at %p (size=%zu, type=%d)\n",
               (void*)obj, size, type_index);

  return obj;
}

void TVMDSPObjectFree(TVMFFIObject* obj) {
  if (obj != NULL) {
    TVM_DSP_LOGD("Freeing object at %p (type=%d)\n",
                 (void*)obj, obj->type_index);
    tvm_dsp_free(obj);
  }
}

/*---------------------------------------------------------------------------
 * Default Deleters
 *---------------------------------------------------------------------------*/

void TVMDSPDefaultObjectDeleter(TVMFFIObject* obj) {
  TVMDSPObjectFree(obj);
}
