"""O2 vs O3 cycle breakdown for representative quantized conv2d kernels.

Builds the same INT8 single-conv2d model under two flag sets:

  O2 : -O2 --auto_inline=500          (current release flags)
  O3 : -O3 --opt_for_speed=5 --auto_inline=500

Both builds keep the .asm file (-k flag) for inspection.  TSC reads are
injected into each lib0.c to attribute cycles to four regions:

  pad_setup  -- pad_temp fill + weight copy to L2 + DMA setup
  zero_fill  -- ff_init loop (decompose_reduction init pass)
  reduction  -- ry×rx×rc×ff update loops (main compute)
  post_conv  -- cast, multiply, add, divide, clip (requantize chain)

The test prints a side-by-side comparison of cycle counts and the SW
pipeline info for the ff inner loop from both ASM files, enabling a
direct O2 vs O3 explanation on the minimal example.

Usage:
  cd tests/ti-dsp-runtime
  pytest --rootdir=. dsp-tests/test_conv2d_cycle_breakdown.py \\
    -k 128ch --dsp-mode=c7x_dload -v -s
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

_DSP_CPP_DIR = Path(__file__).parent.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))

from dsp_utils import (  # noqa: E402  # pyright: ignore
    compile_for_dsp,
    build_dsp_dynmod,
    run_dsp_dload,
    get_target_string,
)

pytestmark = [pytest.mark.c7x_only]

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

CASES = [
    # (label,   IC,   OC,  IH, IW)
    ("128ch", 128, 128, 28, 28),
    ("256ch", 256, 256, 14, 14),
    ("64ch",   64,  64, 56, 56),
]

O2_FLAGS = ""                                          # default from toolchain
O3_FLAGS = "-O3 --opt_for_speed=5 --auto_inline=500"  # regression flags

# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


class _SingleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(oc)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


def _make_model(ic, oc, ih, iw):
    """Return (tvm_mod, input_np) for a single INT8 conv2d layer."""
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
    from torch.export import export
    from tvm import relax
    from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program

    torch.manual_seed(42)
    model = _SingleConv(ic, oc).eval()
    example = (torch.randn(1, ic, ih, iw),)

    exported = export(model, example)
    quantizer = C7xMMAQuantizer(dtype="int8", symmetric_activations=True)
    prepared = prepare_pt2e(exported.module(), quantizer)
    with torch.no_grad():
        for _ in range(10):
            prepared(torch.randn(1, ic, ih, iw))
    quantized = convert_pt2e(prepared)

    with torch.no_grad():
        exported_q = export(quantized, example)
    tvm_mod = from_exported_program(exported_q, keep_params_as_input=True)

    tvm_mod, params = relax.frontend.detach_params(tvm_mod)
    func_params = dict(zip(tvm_mod["main"].params[1:], params["main"]))
    tvm_mod = relax.transform.BindParams("main", func_params)(tvm_mod)

    np.random.seed(42)
    input_np = np.random.randn(1, ic, ih, iw).astype(np.float32)
    return tvm_mod, input_np


# ---------------------------------------------------------------------------
# TSC injection  (identical to previous version)
# ---------------------------------------------------------------------------

_TSC_HEADER = r"""
/* ---- TSC cycle-breakdown instrumentation (injected) ---- */
#ifdef __TI_COMPILER_VERSION__
#include <c7x.h>
static inline uint64_t _rd_tsc(void) { return (uint64_t)__TSC; }
#else
static inline uint64_t _rd_tsc(void) { return 0; }
#endif
/* --------------------------------------------------------- */
"""


def _find_function_body(src, func_re_str):
    for m in re.finditer(func_re_str, src, re.MULTILINE):
        pos   = m.start()
        brace = src.find('{', pos)
        semi  = src.find(';', pos)
        if brace != -1 and (semi == -1 or brace < semi):
            depth, end = 0, brace
            for i in range(brace, len(src)):
                if src[i] == '{': depth += 1
                elif src[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            return m.group(1), brace, end
    return None, -1, -1


def _matching_brace_end(text, open_pos):
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == '{': depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def inject_tsc(lib0_c: Path, tag: str) -> bool:
    """Inject TSC reads into the first fused conv2d function in lib0.c."""
    src = lib0_c.read_text()

    fname, body_start, body_end = _find_function_body(
        src, r'^int32_t (__tvm_ffi_fused_conv2d\w+)\s*\(void\*\s*self_handle'
    )
    if fname is None:
        print(f"[TSC] No fused conv2d definition found")
        return False
    print(f"[TSC] Instrumenting: {fname[:72]}")

    body = src[body_start:body_end]

    htile_m = re.search(r'for \(int32_t yy\w*\s*=\s*0', body)
    if not htile_m:
        print("[TSC] Cannot find h-tile loop")
        return False
    p_htile_insert = htile_m.start()

    fill_m = re.search(r'for \([^)]*ff_init[^)]*\)', body[htile_m.start():])
    if not fill_m:
        print("[TSC] Cannot find ff_init loop")
        return False
    p_fill_abs     = htile_m.start() + fill_m.start()
    p_fill_abs_end = htile_m.start() + fill_m.end()
    p_fill_insert  = p_fill_abs

    ff_init_brace = body.find('{', p_fill_abs_end)
    p_fill_end    = _matching_brace_end(body, ff_init_brace)

    ry_m = re.search(r'for \([^)]*\bry\b[^)]*\)', body[p_fill_end:])
    if not ry_m:
        # O3 fuses ry into the flat loop — no separate ry loop
        print("[TSC] No ry loop found (O3 fused?) — measuring total h-tile only")
        p_ry_abs = None
        p_ry_end = p_fill_end
    else:
        p_ry_abs = p_fill_end + ry_m.start()
        ry_brace = body.find('{', p_ry_abs + ry_m.end() - 1)
        p_ry_end = _matching_brace_end(body, ry_brace)

    p_htile_end = _matching_brace_end(body, body.find('{', htile_m.start()))

    t = tag
    statics = (
        f"  static uint64_t _{t}_s=0,_{t}_f=0,_{t}_u=0,_{t}_p=0;\n"
        f"  static uint32_t _{t}_n=0;\n"
        f"  uint64_t _{t}_t0=_rd_tsc();\n"
    )
    after_setup     = f"  uint64_t _{t}_ta=_rd_tsc(); _{t}_s+=_{t}_ta-_{t}_t0;\n"
    before_fill     = f"    uint64_t _{t}_tf0=_rd_tsc();\n"
    after_fill      = f"    uint64_t _{t}_tf1=_rd_tsc(); _{t}_f+=_{t}_tf1-_{t}_tf0;\n"
    after_ry        = f"    uint64_t _{t}_tu1=_rd_tsc(); _{t}_u+=_{t}_tu1-_{t}_tf1;\n"
    after_htile     = f"  uint64_t _{t}_tb=_rd_tsc();\n"
    at_return = (
        f"  _{t}_p+=_rd_tsc()-_{t}_tb; ++_{t}_n;\n"
        f"  if(_{t}_n>0){{\n"
        f"    uint64_t _tot=_{t}_s+_{t}_f+_{t}_u+_{t}_p;\n"
        f'    printf("=== {tag} breakdown (%u calls) ===\\n", _{t}_n);\n'
        f'    printf("  pad_setup  %12llu avg\\n",(unsigned long long)(_{t}_s/_{t}_n));\n'
        f'    printf("  zero_fill  %12llu avg\\n",(unsigned long long)(_{t}_f/_{t}_n));\n'
        f'    printf("  reduction  %12llu avg\\n",(unsigned long long)(_{t}_u/_{t}_n));\n'
        f'    printf("  post_conv  %12llu avg\\n",(unsigned long long)(_{t}_p/_{t}_n));\n'
        f'    printf("  TOTAL      %12llu avg\\n",(unsigned long long)(_tot/_{t}_n));\n'
        f'    printf("=====================================\\n");\n'
        f"  }}\n"
    )

    if p_ry_abs is not None:
        new_body = (
            body[:1] + "\n" + statics
            + body[1:p_htile_insert]
            + after_setup
            + body[p_htile_insert:p_fill_insert]
            + before_fill
            + body[p_fill_insert:p_fill_end]
            + after_fill
            + body[p_fill_end:p_ry_end]
            + after_ry
            + body[p_ry_end:p_htile_end]
            + after_htile
            + body[p_htile_end:]
        )
    else:
        # O3: no separate ry loop — measure the whole h-tile body as "reduction"
        new_body = (
            body[:1] + "\n" + statics
            + body[1:p_htile_insert]
            + after_setup
            + body[p_htile_insert:p_fill_end]
            + after_fill   # zero_fill = ff_init
            + body[p_fill_end:p_htile_end]
            + after_ry     # reduction = rest of h-tile body
            + body[p_htile_end:]
            + after_htile
        )

    new_body = re.sub(r'(\s+)return 0;',
                      lambda m: f"{m.group(1)}{at_return}{m.group(1)}return 0;",
                      new_body, count=1)

    nl = src.find('\n') + 1
    patched = src[:nl] + _TSC_HEADER + src[nl:body_start] + new_body + src[body_end:]
    lib0_c.write_text(patched)
    return True


# ---------------------------------------------------------------------------
# ASM analysis helpers
# ---------------------------------------------------------------------------

def _parse_sw_pipeline_info(asm_text: str) -> list[dict]:
    """Extract SW pipeline blocks from ASM text. Returns list of dicts."""
    results = []
    in_sw = False
    buf = []
    for line in asm_text.splitlines():
        if 'SOFTWARE PIPELINE INFORMATION' in line:
            in_sw = True; buf = [line]
        elif in_sw:
            if line.startswith(';*') or line.strip() == ';*':
                buf.append(line)
            else:
                block = '\n'.join(buf)
                ii_m  = re.search(r'ii\s*=\s*(\d+)\s+Schedule found', block)
                it_m  = re.search(r'Known Minimum Iteration Count\s*:\s*(\d+)', block)
                rb_m  = re.search(r'Partitioned Resource Bound\s*:\s*(\d+)\s*\(post', block)
                cant  = 'Cannot allocate' in block
                disq  = 'Disqualified' in block
                if ii_m:
                    results.append({
                        'ii': int(ii_m.group(1)),
                        'iters': int(it_m.group(1)) if it_m else 0,
                        'rb': int(rb_m.group(1)) if rb_m else 0,
                    })
                elif cant:
                    it_tried = re.findall(r'ii\s*=\s*(\d+)\s+Cannot', block)
                    results.append({'ii': None, 'tried': it_tried, 'iters': 0, 'rb': 0})
                in_sw = False; buf = []
    return results


def _asm_summary(asm_path: Path, label: str) -> dict:
    """Return key metrics from an ASM file for a named conv2d function."""
    if not asm_path.exists():
        return {}
    text = asm_path.read_text(errors='replace')

    # find the first fused conv2d function
    fname_m = re.search(r'\.global\s+\|\|(__tvm_ffi_fused_conv2d\w+)\|\|', text)
    if not fname_m:
        return {}
    fname = fname_m.group(1)

    # extract function body
    start = text.find(f'||{fname}||:')
    if start == -1:
        return {}
    # find next .global to mark end
    next_global = text.find('.global', start + 1)
    func_text = text[start:next_global] if next_global != -1 else text[start:]

    vmpyww   = func_text.count('VMPYWW')
    mpyww    = func_text.count('MPYWW') - vmpyww
    mpysuhw  = func_text.count('MPYSUHW')
    piped    = func_text.count('PIPED LOOP KERNEL')
    fused_36k = '36864' in func_text or '147456' in func_text

    sw_info = _parse_sw_pipeline_info(func_text)
    ok_loops = [p for p in sw_info if p.get('ii') is not None]

    return {
        'label': label,
        'fname': fname[:60],
        'VMPYWW': vmpyww,
        'MPYWW':  mpyww,
        'MPYSUHW': mpysuhw,
        'piped_kernels': piped,
        'fused_loop': fused_36k,
        'sw_pipelines': ok_loops,
    }


def _print_asm_comparison(o2: dict, o3: dict) -> None:
    if not o2 or not o3:
        print("  (ASM files missing — was -k passed?)")
        return
    rows = [
        ('VMPYWW (512-bit vector multiply)',  o2['VMPYWW'],  o3['VMPYWW']),
        ('MPYWW  (scalar 32×32 multiply)',    o2['MPYWW'],   o3['MPYWW']),
        ('MPYSUHW (scalar signed×unsigned)',  o2['MPYSUHW'], o3['MPYSUHW']),
        ('PIPED LOOP KERNELs',               o2['piped_kernels'], o3['piped_kernels']),
        ('Fused ≥36K-iter loop',             o2['fused_loop'], o3['fused_loop']),
    ]
    print(f"\n  {'Metric':<40}  {'O2':>8}  {'O3':>8}")
    print("  " + "-" * 58)
    for name, v2, v3 in rows:
        print(f"  {name:<40}  {str(v2):>8}  {str(v3):>8}")

    print(f"\n  SW pipeline schedule (OK loops only):")
    print(f"    O2: {o2['sw_pipelines'][:6]}")
    print(f"    O3: {o3['sw_pipelines'][:6]}")


# ---------------------------------------------------------------------------
# Build + run helper
# ---------------------------------------------------------------------------

def _build_and_run(label, generated_dir, weights, input_np,
                   build_dir, cflags, run_tsc=True):
    """Build dynmod with given cflags, inject TSC, rebuild, run. Returns dict."""
    # always add -k to keep asm
    all_flags = (cflags + " -k").strip()

    module_path = build_dsp_dynmod(
        generated_dir,
        build_dir=build_dir,
        weights_file=weights,
        lib0_cflags=all_flags,
    )
    # Save clean ASM (before TSC injection overwrites it)
    asm_src  = build_dir / "lib0.asm"
    asm_path = build_dir / "lib0_clean.asm"
    if asm_src.exists():
        asm_path.write_text(asm_src.read_text(errors='replace'))

    # baseline run (no TSC)
    output, _, cycles = run_dsp_dload(
        module_path, weights, [input_np], embedded_weights=True)

    tsc_data = {}
    if run_tsc:
        # Inject TSC into lib0.c, rebuild WITHOUT -k to avoid overwriting clean ASM
        lib0_c = Path(generated_dir) / "lib0.c"
        ok = inject_tsc(lib0_c, tag=label)
        if ok:
            r = subprocess.run(
                ["cmake", "--build", str(build_dir), "--", "-j4"],
                capture_output=True, text=True)
            if r.returncode == 0:
                _, stdout_tsc, _ = run_dsp_dload(
                    module_path, weights, [input_np], embedded_weights=True)
                for region in ("pad_setup", "zero_fill", "reduction",
                               "post_conv", "TOTAL"):
                    m = re.search(rf"{region}\s+(\d+)\s+avg", stdout_tsc)
                    if m:
                        tsc_data[region] = int(m.group(1))
                print(f"    TSC stdout:\n{stdout_tsc}")
            else:
                print(f"    TSC rebuild failed:\n{r.stderr[-500:]}")

    return {
        'cycles': cycles,
        'output': output,
        'asm_path': asm_path,
        'tsc': tsc_data,
    }


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,ic,oc,ih,iw", CASES)
def test_conv2d_o2_vs_o3(label, ic, oc, ih, iw, dsp_mode, tmp_path):
    if dsp_mode != "c7x_dload":
        pytest.skip("hardware required")

    print(f"\n{'='*64}\n  {label}: {ic}→{oc}  {ih}×{iw}\n{'='*64}")

    # Compile once — both builds share the same lib0.c
    print("\n[1/4] Compiling model...")
    tvm_mod, input_np = _make_model(ic, oc, ih, iw)
    target = get_target_string(dsp_mode)
    generated_dir = compile_for_dsp(tvm_mod, target)
    weights = Path(generated_dir) / "weights.bin"

    # save original lib0.c so we can restore it between O2 and O3 builds
    lib0_c = Path(generated_dir) / "lib0.c"
    original_src = lib0_c.read_text()

    # --- O2 ---
    print("\n[2/4] Building and running O2...")
    o2_dir = tmp_path / "build-o2"
    o2 = _build_and_run(f"{label}_o2", generated_dir, weights, input_np,
                        o2_dir, O2_FLAGS)
    print(f"    O2 baseline cycles: {o2['cycles']:,}")

    # restore lib0.c (TSC injection modifies it in-place)
    lib0_c.write_text(original_src)

    # --- O3 ---
    print("\n[3/4] Building and running O3...")
    o3_dir = tmp_path / "build-o3"
    o3 = _build_and_run(f"{label}_o3", generated_dir, weights, input_np,
                        o3_dir, O3_FLAGS)
    print(f"    O3 baseline cycles: {o3['cycles']:,}")

    # --- comparison ---
    print(f"\n[4/4] Analysis\n")

    ratio = o3['cycles'] / o2['cycles'] if o2['cycles'] > 0 else 0
    print(f"  O2 cycles: {o2['cycles']:>14,}   (baseline)")
    print(f"  O3 cycles: {o3['cycles']:>14,}   ({ratio:.2f}× {'slower' if ratio>1 else 'faster'})")

    if o2['tsc'] and o3['tsc']:
        total_o2 = o2['tsc'].get('TOTAL', 1)
        total_o3 = o3['tsc'].get('TOTAL', 1)
        print(f"\n  {'Region':<12}  {'O2 avg cyc':>14}  {'O2 %':>6}  "
              f"{'O3 avg cyc':>14}  {'O3 %':>6}  {'ratio':>7}")
        print("  " + "-" * 70)
        for r in ("pad_setup", "zero_fill", "reduction", "post_conv"):
            v2 = o2['tsc'].get(r, 0)
            v3 = o3['tsc'].get(r, 0)
            p2 = 100 * v2 / total_o2 if total_o2 else 0
            p3 = 100 * v3 / total_o3 if total_o3 else 0
            rx = v3 / v2 if v2 > 0 else 0
            print(f"  {r:<12}  {v2:>14,}  {p2:>5.1f}%  {v3:>14,}  {p3:>5.1f}%  {rx:>6.2f}×")
        print(f"  {'TOTAL':<12}  {total_o2:>14,}           {total_o3:>14,}")

    # ASM comparison
    o2_asm = _asm_summary(o2['asm_path'], 'O2')   # lib0_clean.asm — no TSC
    o3_asm = _asm_summary(o3['asm_path'], 'O3')   # lib0_clean.asm — no TSC
    print(f"\n  --- ASM comparison ---")
    _print_asm_comparison(o2_asm, o3_asm)

    # correctness
    assert np.allclose(o2['output'], o3['output'], atol=2), \
        "O2 and O3 outputs differ beyond tolerance"

    # save result
    out_path = Path(__file__).parent / f"cycle_breakdown_{label}.txt"
    with open(out_path, "w") as f:
        f.write(f"{label}: {ic}→{oc}  {ih}×{iw}\n")
        f.write(f"O2: {o2['cycles']:,} cycles\n")
        f.write(f"O3: {o3['cycles']:,} cycles  ({ratio:.2f}x)\n\n")
        for key, val in [("O2 TSC", o2['tsc']), ("O3 TSC", o3['tsc']),
                         ("O2 ASM", o2_asm), ("O3 ASM", o3_asm)]:
            f.write(f"{key}: {val}\n")
    print(f"\n  Results saved to {out_path.name}")
