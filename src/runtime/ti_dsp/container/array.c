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
 * \file container/array.c
 * \brief Array container implementation for TVM DSP Runtime
 */

#include "array.h"
#include "../platform/dsp_platform.h"

void TVMDSPArrayDeleter(TVMFFIObject* self) {
  TVMDSPArray* arr = (TVMDSPArray*)self;
  if (arr == NULL) {
    return;
  }

  /* Decrement reference count on all contained objects */
  for (int32_t i = 0; i < arr->size; i++) {
    TVMFFIAny* elem = &arr->elements[i];
    if (TVMFFIAnyIsObject(elem) && elem->v_obj != NULL) {
      TVMFFIObject* obj = (TVMFFIObject*)elem->v_obj;
      obj->ref_counter--;
      /* If reference count reaches 0, call the object's deleter */
      if (obj->ref_counter <= 0 && obj->deleter != NULL) {
        obj->deleter(obj);
      }
    }
  }

  /* Free the array struct itself */
  tvm_dsp_free(arr);
}
