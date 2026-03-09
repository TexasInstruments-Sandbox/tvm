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
 * \file board_init.h
 * \brief J722S C7x board initialization for TVM DSP runtime
 */

#ifndef TVM_DSP_J722S_BOARD_INIT_H_
#define TVM_DSP_J722S_BOARD_INIT_H_

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Initialize J722S C7x board
 *
 * For standalone operation, this initializes MMU, cache, and
 * interrupt handling. The startup code (boot_c75.c) handles
 * most initialization before main() is called.
 *
 * \return 0 on success, non-zero on failure
 */
int j722s_board_init(void);

/*!
 * \brief Deinitialize J722S C7x board
 *
 * Cleanup resources before exit.
 */
void j722s_board_deinit(void);

#ifdef __cplusplus
}
#endif

#endif /* TVM_DSP_J722S_BOARD_INIT_H_ */
