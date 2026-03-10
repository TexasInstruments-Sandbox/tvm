"""Embed a binary file as a .dsp_syms_out section in C7x assembly.

Usage: python3 gen_dsp_syms_sect.py <input_binary> <output_asm>

The generated assembly file contains the input binary encoded as .byte
directives in a .dsp_syms_out section. This section is embedded into the
loadable module and extracted by DLOAD at load time to populate the
dependent symbol table.

Based on neo-tvm's gen_c7x_sect.py.
"""

import sys

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input_binary> <output_asm>", file=sys.stderr)
    sys.exit(1)

with open(sys.argv[1], "rb") as fi:
    with open(sys.argv[2], "wt") as fo:
        fo.write('\t.sect ".dsp_syms_out"\n\t.retain')
        nbytes = 0
        byte = fi.read(1)
        while byte:
            val = int.from_bytes(byte, byteorder='little', signed=True)
            if (nbytes % 10) == 0:
                fo.write(f"\n\t.byte {val}")
            else:
                fo.write(f", {int(val)}")
            nbytes += 1
            byte = fi.read(1)
        fo.write("\n")

print(f"Embedded {nbytes} bytes from {sys.argv[1]} into {sys.argv[2]}")
