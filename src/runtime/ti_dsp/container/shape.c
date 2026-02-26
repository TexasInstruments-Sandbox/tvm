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
 * \file container/shape.c
 * \brief Shape implementation for TVM DSP Runtime
 */

#include "shape.h"
#include "../platform/dsp_platform.h"
#include "../platform/dsp_memory.h"

/*---------------------------------------------------------------------------
 * Shape Deleter
 *---------------------------------------------------------------------------*/
void TVMDSPShapeDeleter(TVMFFIObject* obj) {
  /* Simply free the shape object */
  tvm_dsp_free(obj);
}

/*---------------------------------------------------------------------------
 * Shape Creation
 *---------------------------------------------------------------------------*/

TVMDSPShape* TVMDSPShapeCreate(const int64_t* dims, size_t ndim) {
  TVMDSPShape* shape;
  size_t i;

  /* Validate */
  if (ndim > TVM_DSP_SHAPE_MAX_NDIM) {
    return NULL;
  }

  /* Allocate from L3 (L2 reserved for workspace) */
  shape = (TVMDSPShape*)tvm_dsp_alloc(sizeof(TVMDSPShape),
                                       TVM_DSP_CACHE_LINE_SIZE,
                                       TVM_DSP_MEM_MAIN);
  if (shape == NULL) {
    return NULL;
  }

  /* Initialize object header */
  shape->type_index = kTVMFFIShape;
  shape->ref_counter = 1;
  shape->deleter = TVMDSPShapeDeleter;

  /* Copy dimensions to inline storage */
  for (i = 0; i < ndim; i++) {
    shape->shape_data[i] = dims[i];
  }

  /* Set cell fields to point to inline storage */
  shape->data = shape->shape_data;
  shape->size = ndim;

  return shape;
}

TVMDSPShape* TVMDSPShapeCreateEmpty(void) {
  return TVMDSPShapeCreate(NULL, 0);
}
