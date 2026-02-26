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
 * \file logging.h
 * \brief TVM DSP Runtime - Logging macros
 */
#ifndef TVM_RUNTIME_TI_DSP_CORE_LOGGING_H_
#define TVM_RUNTIME_TI_DSP_CORE_LOGGING_H_

#include "../platform/dsp_platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Log levels */
#define TVM_DSP_LOG_NONE 0
#define TVM_DSP_LOG_ERROR 1
#define TVM_DSP_LOG_WARN 2
#define TVM_DSP_LOG_INFO 3
#define TVM_DSP_LOG_DEBUG 4

/* Default log level */
#ifndef TVM_DSP_LOG_LEVEL
#define TVM_DSP_LOG_LEVEL TVM_DSP_LOG_INFO
#endif

/* Logging macros - compile-time filtering based on TVM_DSP_LOG_LEVEL */

/* Error logging */
#if TVM_DSP_LOG_LEVEL >= TVM_DSP_LOG_ERROR
#define TVM_DSP_LOGE(...) tvm_dsp_log("ERROR: " __VA_ARGS__)
#else
#define TVM_DSP_LOGE(...) ((void)0)
#endif

/* Warning logging */
#if TVM_DSP_LOG_LEVEL >= TVM_DSP_LOG_WARN
#define TVM_DSP_LOGW(...) tvm_dsp_log("WARN: " __VA_ARGS__)
#else
#define TVM_DSP_LOGW(...) ((void)0)
#endif

/* Info logging */
#if TVM_DSP_LOG_LEVEL >= TVM_DSP_LOG_INFO
#define TVM_DSP_LOGI(...) tvm_dsp_log("INFO: " __VA_ARGS__)
#else
#define TVM_DSP_LOGI(...) ((void)0)
#endif

/* Debug logging */
#if TVM_DSP_LOG_LEVEL >= TVM_DSP_LOG_DEBUG
#define TVM_DSP_LOGD(...) tvm_dsp_log("DEBUG: " __VA_ARGS__)
#else
#define TVM_DSP_LOGD(...) ((void)0)
#endif

/* Assertion macro */
#ifdef NDEBUG
#define TVM_DSP_ASSERT(cond) ((void)0)
#else
#define TVM_DSP_ASSERT(cond)                                              \
  do {                                                                    \
    if (!(cond)) {                                                        \
      tvm_dsp_log("ASSERT FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
      while (1) {                                                         \
      }                                                                   \
    }                                                                     \
  } while (0)
#endif

/* Check macro (always enabled) */
#define TVM_DSP_CHECK(cond)                                              \
  do {                                                                   \
    if (!(cond)) {                                                       \
      tvm_dsp_log("CHECK FAILED: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
      return -1;                                                         \
    }                                                                    \
  } while (0)

#ifdef __cplusplus
}
#endif

#endif /* TVM_RUNTIME_TI_DSP_CORE_LOGGING_H_ */
