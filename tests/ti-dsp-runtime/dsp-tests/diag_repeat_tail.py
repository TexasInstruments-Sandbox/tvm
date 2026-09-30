#!/usr/bin/env python
"""Is a repeated-inference fault in the TIDL path or the firmware IO path?

Runs a plain TVM MLP -- no TIDL subgraph at all -- under c7x_compute's
``profile`` verb, which sets repeat=2 so the firmware infers twice and returns
the second result.  Sweeps the output width, since a fault that depends on
output size will only show above some threshold.

This is the control to reach for first when a TIDL model misbehaves on its
second and later inferences.  If this script is clean at the same output size,
the firmware repeat loop, output extraction, staging buffer and DSP allocator
are all exonerated and the fault is in the TIDL path; if it also diverges, it is
not TIDL-specific.

Note ``test_repeated_inference_consistent`` in test_c7x_vm_dsp.py is not a
substitute: its MLP output is 8 floats, far below any size-dependent threshold,
and it issues separate INFER requests rather than repeat=2 within one.

Usage:
    python diag_repeat_tail.py --board j722s-evm
    python diag_repeat_tail.py --board j722s-evm --sizes 1000

Requires TI_CGT_C7000_PATH and a board running c7x_compute firmware.  Hardware
runs must not overlap with a test suite.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_TESTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_TESTS_DIR / "dsp-cpp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsp_utils import (  # noqa: E402
    add_board_arg,
    build_dsp_dynmod,
    compile_for_dsp,
    get_target_string,
    run_dsp_dload,
)
from model_utils import create_mlp_model  # noqa: E402


def _build_mlp(output_size: int):
    """Build a pure-TVM MLP with the requested output width for c7x_dload."""
    tvm_mod, _unused_torch_model, input_data = create_mlp_model(
        input_size=64, hidden_size=32, output_size=output_size
    )
    target = get_target_string("c7x_dload")
    gen_dir = Path(tempfile.mkdtemp(prefix=f"diag_tail_gen_{output_size}_"))
    compile_for_dsp(tvm_mod, target_string=target, output_dir=gen_dir)
    build_dir = Path(tempfile.mkdtemp(prefix=f"diag_tail_build_{output_size}_"))
    weights = gen_dir / "weights.bin"
    module_path = build_dsp_dynmod(
        generated_dir=gen_dir,
        build_dir=build_dir,
        weights_file=weights if weights.exists() else None,
    )
    return module_path, (weights if weights.exists() else module_path), input_data


def _find_shift(clean: np.ndarray, dirty: np.ndarray):
    """Return (shift, prefix_bytes) if dirty[s:] == clean[:N-s], else None.

    A match means the repeated-inference output has a tail that verbatim
    repeats its own head, which is what writing past the allocation looks like.
    """
    n = len(clean)
    for s in range(1, n):
        if np.array_equal(dirty[s:], clean[: n - s]):
            return s, s * 4
    return None


def _probe(output_size: int) -> dict:
    """Build at one output width, run under both verbs, and compare."""
    print("\n" + "=" * 72)
    print(f"OUTPUT SIZE {output_size} floats = {output_size * 4} bytes "
          f"(0x{output_size * 4:x})")
    print("=" * 72)

    module_path, weights, input_data = _build_mlp(output_size)

    plain, _, _ = run_dsp_dload(module_path, weights, [input_data],
                                embedded_weights=True, profile=False)
    prof, _, _ = run_dsp_dload(module_path, weights, [input_data],
                               embedded_weights=True, profile=True)
    plain = np.asarray(plain, dtype=np.float64).reshape(-1)
    prof = np.asarray(prof, dtype=np.float64).reshape(-1)

    ndiff = int((plain != prof).sum())
    row: dict = {"size": output_size, "bytes": output_size * 4, "ndiff": ndiff}
    print(f"run vs profile: {ndiff} of {len(plain)} elements differ")

    if ndiff == 0:
        print("  -> CLEAN: repeat=2 matches single inference at this size")
        row["prefix_bytes"] = None
        return row

    found = _find_shift(plain, prof)
    if found:
        shift, prefix_bytes = found
        print(f"  -> CORRUPT: profile[{shift}:] is a verbatim copy of "
              f"run[:{len(plain) - shift}]")
        print(f"     clean prefix {shift} floats = {prefix_bytes} bytes "
              f"(0x{prefix_bytes:x})")
        print(f"     corrupt tail {len(plain) - shift} floats = "
              f"{(len(plain) - shift) * 4} bytes")
        row["prefix_bytes"] = prefix_bytes
    else:
        first = int(np.flatnonzero(plain != prof)[0])
        print(f"  -> CORRUPT but not a head-copy; first differing index "
              f"{first} (byte {first * 4})")
        print(f"     run[{first}:{first + 4}]     = {plain[first:first + 4]}")
        print(f"     profile[{first}:{first + 4}] = {prof[first:first + 4]}")
        row["prefix_bytes"] = first * 4
    return row


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_board_arg(parser)
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[256, 768, 1000, 1024, 2048],
                        help="output widths in floats "
                             "(default sweeps 1024 B .. 8192 B)")
    args = parser.parse_args()

    if args.board is None:
        parser.error("--board is required (this runs on hardware)")
    if os.environ.get("TI_CGT_C7000_PATH") is None:
        parser.error("TI_CGT_C7000_PATH is not set")

    rows = [_probe(size) for size in args.sizes]

    print("\n" + "=" * 72)
    print("SUMMARY (non-TIDL MLP, c7x_dload)")
    print("=" * 72)
    print(f"{'floats':>8} {'bytes':>8} {'differ':>8}  clean prefix")
    for r in rows:
        prefix = "-" if r["prefix_bytes"] is None else \
            f"{r['prefix_bytes']} bytes (0x{r['prefix_bytes']:x})"
        print(f"{r['size']:8d} {r['bytes']:8d} {r['ndiff']:8d}  {prefix}")

    corrupt = [r for r in rows if r["ndiff"] > 0]
    print()
    if not corrupt:
        print("No non-TIDL corruption at any size tested "
              "=> fault is in the TIDL path, not the firmware IO path.")
    else:
        prefixes = {r["prefix_bytes"] for r in corrupt}
        print(f"Non-TIDL modules DO corrupt ({len(corrupt)}/{len(rows)} sizes) "
              "=> fault is in the firmware/allocator path, not TIDL.")
        if len(prefixes) == 1:
            print(f"Clean prefix is a fixed {prefixes.pop()} bytes across sizes "
                  "=> a specific buffer bound, not a proportional chunked copy.")
        else:
            print(f"Clean prefix varies across sizes: {sorted(prefixes)} "
                  "=> size-dependent, consistent with a chunked copy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
