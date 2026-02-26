# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Schedule rules for TI C66x/C7x DSP targets with cache-aware tiling.

This module provides factory functions that create schedule rules optimized
for the TI C66x/C7x DSP memory hierarchy:
    - L1D: 32KB (default)
    - L2:  384KB (default)
    - SIMD: 128-bit vectors

The schedule rules constrain tile sizes to fit working sets in L1D cache
for optimal software pipelining performance.

Example usage:
    import tvm
    from tvm import meta_schedule as ms
    from tvm.meta_schedule.schedule_rule.c66x import get_c66x_schedule_rules

    target = tvm.target.Target("c_static -mcpu=c66x")
    rules = get_c66x_schedule_rules(target)

    # Use rules in tuning
    config = ms.TuneConfig(...)
"""
from typing import List, Optional

from tvm.target import Target

from .auto_inline import AutoInline
from .multi_level_tiling import MultiLevelTiling, ReuseType
from .parallel_vectorize_unroll import ParallelizeVectorizeUnroll
from .schedule_rule import ScheduleRule


def _calculate_max_innermost_factor(
    l1_cache_size: int,
    dtype_bytes: int = 4,
    num_buffers: int = 3,
    headroom: float = 0.75,
) -> int:
    """Calculate maximum innermost tile factor that fits in L1 cache.

    For a 2D tile of size (tile_m, tile_n), we need space for multiple buffers
    (typically A, B, C for matmul). The innermost factor should satisfy:

        tile_m * tile_n * dtype_bytes * num_buffers <= l1_cache_size * headroom

    For balanced 2D tiles (tile_m == tile_n == factor):
        factor <= sqrt(l1_cache_size * headroom / (dtype_bytes * num_buffers))

    Parameters
    ----------
    l1_cache_size : int
        L1 data cache size in bytes.
    dtype_bytes : int
        Size of data type in bytes (default: 4 for float32).
    num_buffers : int
        Number of buffers to fit in cache (default: 3 for A, B, C).
    headroom : float
        Fraction of cache to use (default: 0.75 to leave room for other data).

    Returns
    -------
    int
        Maximum innermost tile factor.
    """
    budget = int(l1_cache_size * headroom / num_buffers / dtype_bytes)
    max_factor = int(budget**0.5)
    # Clamp to reasonable range and round down to power of 2 for alignment
    max_factor = min(max_factor, 64)
    max_factor = max(max_factor, 4)
    # Round down to nearest power of 2
    power = 1
    while power * 2 <= max_factor:
        power *= 2
    return power


def _get_vector_load_lens(vector_width: int, dtype_bytes: int = 4) -> List[int]:
    """Calculate vector load lengths based on SIMD width.

    Parameters
    ----------
    vector_width : int
        Vector register width in bits.
    dtype_bytes : int
        Size of data type in bytes.

    Returns
    -------
    List[int]
        List of supported vector load lengths.
    """
    vector_bytes = vector_width // 8
    elements = vector_bytes // dtype_bytes
    return [elements] if elements >= 1 else [1]


def get_c66x_schedule_rules(
    target: Optional[Target] = None,
    dtype_bytes: int = 4,
) -> List[ScheduleRule]:
    """Get schedule rules optimized for TI C66x/C7x DSP.

    Creates a set of schedule rules with cache-aware tiling constraints
    based on the target's memory hierarchy attributes.

    Parameters
    ----------
    target : Optional[Target]
        The c_static target. If None, uses default C66x cache sizes.
    dtype_bytes : int
        Size of data type in bytes (default: 4 for float32).

    Returns
    -------
    List[ScheduleRule]
        List of schedule rules configured for C66x.

    Example
    -------
    >>> target = tvm.target.Target("c_static -mcpu=c66x")
    >>> rules = get_c66x_schedule_rules(target)
    >>> # rules can be used with meta_schedule.TuneConfig
    """
    # Extract cache parameters from target or use C66x defaults
    if target is not None:
        l1_cache_size = int(target.attrs.get("l1d-cache-size", 32768))
        vector_width = int(target.attrs.get("vector-width", 128))
    else:
        l1_cache_size = 32768  # 32KB L1D
        vector_width = 128  # 128-bit SIMD

    # Calculate constrained tile factor
    max_innermost = _calculate_max_innermost_factor(
        l1_cache_size=l1_cache_size,
        dtype_bytes=dtype_bytes,
    )

    # Calculate vector load lengths
    vector_load_lens = _get_vector_load_lens(vector_width, dtype_bytes)

    return [
        # Auto-inline for element-wise fusion
        AutoInline(
            into_producer=True,
            into_consumer=True,
            inline_const_tensor=True,
            disallow_if_then_else=True,
            require_injective=True,
            require_ordered=True,
        ),
        # Multi-level tiling with L1 cache constraints
        MultiLevelTiling(
            structure="SSRSRS",  # 3-level tiling for CPU
            tile_binds=None,  # No thread binding for single-core DSP
            max_innermost_factor=max_innermost,
            vector_load_lens=vector_load_lens,
            reuse_read=ReuseType(
                req="may",
                levels=[1, 2],
                scope="local",
            ),
            reuse_write=ReuseType(
                req="may",
                levels=[1, 2],
                scope="local",
            ),
        ),
        # Vectorization and unrolling
        ParallelizeVectorizeUnroll(
            max_jobs_per_core=-1,  # No parallelization on single-core DSP
            max_vectorize_extent=vector_load_lens[0],
            unroll_max_steps=[0, 16, 64, 512],
            unroll_explicit=True,
        ),
    ]


def get_c66x_multi_level_tiling(
    target: Optional[Target] = None,
    dtype_bytes: int = 4,
    structure: str = "SSRSRS",
    reuse_read: bool = True,
    reuse_write: bool = True,
) -> MultiLevelTiling:
    """Get a MultiLevelTiling rule configured for C66x.

    This is a convenience function for users who want just the tiling rule
    without the full set of schedule rules.

    Parameters
    ----------
    target : Optional[Target]
        The c_static target. If None, uses default C66x cache sizes.
    dtype_bytes : int
        Size of data type in bytes (default: 4 for float32).
    structure : str
        Tiling structure (default: "SSRSRS" for 3-level CPU tiling).
    reuse_read : bool
        Whether to enable read reuse (default: True).
    reuse_write : bool
        Whether to enable write reuse (default: True).

    Returns
    -------
    MultiLevelTiling
        Configured tiling rule.
    """
    if target is not None:
        l1_cache_size = int(target.attrs.get("l1d-cache-size", 32768))
        vector_width = int(target.attrs.get("vector-width", 128))
    else:
        l1_cache_size = 32768
        vector_width = 128

    max_innermost = _calculate_max_innermost_factor(
        l1_cache_size=l1_cache_size,
        dtype_bytes=dtype_bytes,
    )
    vector_load_lens = _get_vector_load_lens(vector_width, dtype_bytes)

    return MultiLevelTiling(
        structure=structure,
        tile_binds=None,
        max_innermost_factor=max_innermost,
        vector_load_lens=vector_load_lens,
        reuse_read=ReuseType(req="may", levels=[1, 2], scope="local")
        if reuse_read
        else None,
        reuse_write=ReuseType(req="may", levels=[1, 2], scope="local")
        if reuse_write
        else None,
    )
