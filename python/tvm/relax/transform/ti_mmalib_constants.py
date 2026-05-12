# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.
"""MMALIB hardware constants for TI C7x DSP.

These constants define the MMA (Matrix Multiply Accelerator) alignment
requirements based on the vector width of the target C7x core.

AM67A (J722S) uses the C7504 core with 256-bit (32-byte) vector/MMA width:
  - int8:  32 bytes / 1 = 32 elements per MMA vector
  - int16: 32 bytes / 2 = 16 elements per MMA vector

Dimensions of matrices passed to MMALIB must be multiples of these values
for the MMA hardware to operate without padding.

TODO: derive these from a target attribute (e.g., -vector-width=256)
rather than hardcoding for C7504. Other C7x variants:
  - C7120: 512-bit (64-byte) → int8=64, int16=32
  - C7504: 256-bit (32-byte) → int8=32, int16=16
"""

# C7504 (AM67A/J722S): __C7X_VEC_SIZE_BYTES__ = 32
C7X_VEC_SIZE_BYTES = 32

MMA_SIZE_I8 = C7X_VEC_SIZE_BYTES // 1   # 32 elements
MMA_SIZE_I16 = C7X_VEC_SIZE_BYTES // 2  # 16 elements
