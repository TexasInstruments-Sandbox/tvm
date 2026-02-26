#!/usr/bin/env python3
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

"""
Binary to TI Assembly Converter

Converts binary files to TI C6000 assembly format for embedding.
Uses .byte directives which assemble much faster than compiling C arrays.

Usage:
    python bin_to_asm.py input.bin output.asm [symbol_prefix]

The generated assembly creates these symbols:
    _binary_<prefix>_start  - Start of binary data
    _binary_<prefix>_end    - End of binary data (one past last byte)
    _binary_<prefix>_size   - Size of binary data as uint32
"""

import sys
import os
from pathlib import Path


def binary_to_ti_asm(input_file: str, output_file: str, symbol_prefix: str = "weights_bin") -> None:
    """Convert binary file to TI C6000 assembly format.

    Args:
        input_file: Path to input binary file
        output_file: Path to output assembly file
        symbol_prefix: Symbol name prefix (default: weights_bin)
    """
    # Read binary data
    with open(input_file, "rb") as f:
        data = f.read()

    file_size = len(data)
    input_name = Path(input_file).name

    # Generate TI assembly
    with open(output_file, "w") as f:
        # Header
        f.write(f"; Auto-generated from {input_name}\n")
        f.write(f"; Size: {file_size} bytes ({file_size // 1024} KB)\n")
        f.write(";\n")
        f.write("; TI C6000 assembly format for binary embedding\n")
        f.write("; Uses .byte directives for fast assembly\n")
        f.write(";\n\n")

        # Section directive - use .rodata.weights for model weights
        # This allows the linker to place weights in a separate memory region
        # (e.g., DDR_C7X_MAIN) from regular .rodata which stays with code
        f.write("    .sect \".rodata.weights\"\n\n")

        # Global symbol declarations
        f.write(f"    .global _binary_{symbol_prefix}_start\n")
        f.write(f"    .global _binary_{symbol_prefix}_end\n")
        f.write(f"    .global _binary_{symbol_prefix}_size\n\n")

        # Align to 8 bytes for efficient DSP memory access
        f.write("    .align 8\n\n")

        # Start symbol and data
        f.write(f"_binary_{symbol_prefix}_start:\n")

        # Write data as .byte directives
        # Group bytes into lines for readability (16 bytes per line)
        bytes_per_line = 16
        for i in range(0, file_size, bytes_per_line):
            chunk = data[i:i + bytes_per_line]
            hex_values = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    .byte {hex_values}\n")

        f.write("\n")

        # End symbol (points one past last byte)
        f.write(f"_binary_{symbol_prefix}_end:\n\n")

        # Align before size word
        f.write("    .align 4\n")

        # Size as a 32-bit word
        f.write(f"_binary_{symbol_prefix}_size:\n")
        f.write(f"    .word {file_size}\n")

        f.write("\n; End of embedded data\n")

    print(f"Generated {output_file}: {file_size} bytes embedded")


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output.asm> [symbol_prefix]")
        print()
        print("Converts binary file to TI C6000 assembly for embedding.")
        print()
        print("Arguments:")
        print("  input.bin     Input binary file")
        print("  output.asm    Output TI assembly file")
        print("  symbol_prefix Symbol name prefix (default: weights_bin)")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    symbol_prefix = sys.argv[3] if len(sys.argv) > 3 else "weights_bin"

    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    binary_to_ti_asm(input_file, output_file, symbol_prefix)


if __name__ == "__main__":
    main()
