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
 * \file board_init.c
 * \brief J722S C7x board initialization for TVM DSP runtime
 *
 * This file provides board initialization for standalone C7x operation.
 * Most initialization (MMU, cache, interrupts) is done by the startup
 * code before main() is called.
 */

#include "board_init.h"
#include <stdio.h>

int j722s_board_init(void)
{
    /*
     * For standalone operation, initialization is handled by:
     *   - vectors.asm: Reset vector jumps to _c_int00_secure
     *   - boot_c75.c: Sets up stack, MMU, BSS, and calls startup init
     *   - startup.c: Enables cache and initializes interrupt/exception handling
     *
     * By the time main() is called, the C7x is fully initialized.
     * This function just provides a hook for any additional setup.
     */

    printf("J722S C7x board initialization complete\n");
    return 0;
}

void j722s_board_deinit(void)
{
    printf("J722S C7x board deinitialization complete\n");
}
