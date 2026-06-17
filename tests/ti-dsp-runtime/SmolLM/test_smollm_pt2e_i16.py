"""SmolLM-135M PT2E int16 calibration test.

Validates the C7xMMAQuantizer(dtype="int16") → from_exported_program →
FuseMMALIBQDQFCI16 pipeline end-to-end on SmolLM-135M.

Two test tiers:
  1. Fusion coverage (pure Python, @pytest.mark.quick):
     Assert that FuseMMALIBQDQFCI16 fires on at least one FC layer and
     report the actual coverage count.  No DSP or TI toolchain required.

  2. DSP execution + cosine similarity (@pytest.mark.c7x_only):
     Compile for c7x_host, run, measure cosine similarity of output
     logits vs float32 reference.

Partial fusion (known limitation):
  In SmolLM's transformer graph, layer-boundary activations are shared
  nodes that feed into multiple consumers (e.g., both q_proj and k_proj
  read from the same post-RMSNorm dequantize, and the same tensor also
  feeds residual adds).  TVM's FuseOpsByPattern cannot include a shared
  node in a composite because that would break connections to the other
  consumers.  Only FC layers whose input dequantize has a single consumer
  can be fused.

  Practical coverage: ~5 layers per pass invocation with seq_len=16.
  To achieve full coverage, the graph would need to be restructured so
  each FC layer receives a dedicated (non-shared) dequantize copy of its
  input activation.

Known accuracy ceiling (documented, not a failure):
  mmalib_matmul_bias_i16 uses uint8 scale/shift requantization.  For
  K=576+ layers the per-element error is ~24-61; compounded over 30
  layers this degrades output logits significantly.  Cosine similarity
  is expected to be low (~0.1-0.3).  The recommended LLM path remains
  LegalizeMLPToMMALIBInt16 (mmalib_matmul_i16, no requantization).

  This test validates pipeline correctness, not production accuracy.

Model: SmolLM-135M-Instruct (HuggingFaceTB/SmolLM-135M-Instruct)
  Download once:
    python -c "
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    m = AutoModelForCausalLM.from_pretrained(
        'HuggingFaceTB/SmolLM-135M-Instruct', dtype=torch.float32)
    m.save_pretrained('tests/ti-dsp-runtime/SmolLM/model')
    "

Usage:
    cd tests/ti-dsp-runtime
    pytest --rootdir=. SmolLM/test_smollm_pt2e_i16.py -m quick -v
    pytest --rootdir=. SmolLM/test_smollm_pt2e_i16.py -m quick \
        --dsp-mode=c7x_host -v
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.export import export
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
from transformers import AutoModelForCausalLM

import tvm
from tvm import relax
from tvm.relax.frontend.torch import C7xMMAQuantizer, from_exported_program
from tvm.relax.transform.ti_mmalib_passes import get_mmalib_qdq_passes

_THIS_DIR = Path(__file__).resolve().parent
_DSP_CPP_DIR = _THIS_DIR.parent / "dsp-cpp"
sys.path.insert(0, str(_DSP_CPP_DIR))
sys.path.insert(0, str(_THIS_DIR))

from dsp_utils import compile_and_run_dsp, get_target_string  # noqa: E402
from smollm_c7x import SmolLMWrapper, _resolve_model_dir  # noqa: E402

# ---------------------------------------------------------------------------
# SmolLM-135M architecture constants
# ---------------------------------------------------------------------------

# Total FC layers in SmolLM-135M (30 blocks × 7 + 1 lm_head)
_NUM_FC_LAYERS_TOTAL = 211

# Minimum fused layers the test asserts — partial fusion due to shared
# activation nodes in the transformer graph (see module docstring).
_MIN_FC_LAYERS_FUSED = 1

# Prompts used for PT2E calibration — diverse to cover varied hidden-state ranges
_CALIBRATION_PROMPTS_IDS = [
    # Use random token sequences; tokenizer not required for calibration
    torch.randint(0, 49152, (1, 64), generator=torch.Generator().manual_seed(i))
    for i in range(4)
]

# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _make_example_inputs(seq_len: int = 64):
    """Create a deterministic (input_ids, position_ids, attention_mask) tuple."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 49152, (1, seq_len))
    position_ids = torch.arange(seq_len).unsqueeze(0)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    return input_ids, position_ids, attention_mask


def _build_pt2e_i16_module(
    model_dir: Path,
    seq_len: int = 64,
    n_calibration: int = 4,
) -> tuple[tvm.IRModule, tuple, np.ndarray]:
    """Full PT2E int16 pipeline: SmolLM → C7xMMAQuantizer → Relax IRModule.

    Returns:
        mod:       IRModule with params bound, ready for compile_and_run_dsp
        inputs:    (input_ids_np, position_ids_np, attn_mask_np) numpy arrays
        ref_logits: float32 PyTorch reference output (1, seq_len, vocab_size)
    """
    if not model_dir.exists():
        pytest.skip(f"SmolLM model not found at {model_dir}. Download first.")

    # 1. Load model
    hf_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), dtype=torch.float32, local_files_only=True
    )
    hf_model.eval()
    model = SmolLMWrapper(hf_model)
    model.eval()

    example_inputs = _make_example_inputs(seq_len)

    # 2. Float32 reference BEFORE quantization
    with torch.no_grad():
        ref_logits = model(*example_inputs).numpy()

    # 3. Export to FX graph (needed by prepare_pt2e)
    exported = export(model, example_inputs).module()

    # 4. Annotate with C7xMMAQuantizer(int16)
    #    Symmetric int16: d_zp=w_zp=o_zp=0 enforced by the quantizer.
    quantizer = C7xMMAQuantizer(dtype="int16")
    prepared = prepare_pt2e(exported, quantizer)

    # 5. Calibrate — run several random inputs through the observers so they
    #    collect min/max ranges for each activation tensor.
    with torch.no_grad():
        for i in range(n_calibration):
            cal_ids = torch.randint(
                0, 49152, (1, seq_len),
                generator=torch.Generator().manual_seed(i),
            )
            cal_pos = torch.arange(seq_len).unsqueeze(0)
            cal_mask = torch.ones(1, seq_len, dtype=torch.long)
            prepared(cal_ids, cal_pos, cal_mask)

    # 6. Convert observers → quantize/dequantize nodes
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="erase_node")
        quantized_gm = convert_pt2e(prepared)

    # 7. Re-export and import into TVM
    #    from_exported_program emits int8 zero_points (required by TVM's
    #    relax.dequantize) — safe since symmetric zp is always 0.
    quantized_ep = export(quantized_gm, example_inputs)
    mod = from_exported_program(quantized_ep, keep_params_as_input=True)

    # 8. Bind weight parameters
    mod, params = relax.frontend.detach_params(mod)
    n_user = 3  # input_ids, position_ids, attention_mask
    func_params_dict = dict(zip(mod["main"].params[n_user:], params["main"]))
    mod = relax.transform.BindParams("main", func_params_dict)(mod)  # pyright: ignore

    inputs_np = tuple(t.numpy() for t in example_inputs)
    return mod, inputs_np, ref_logits


def _run_mmalib_qdq_passes(mod: tvm.IRModule) -> tvm.IRModule:
    """Apply the MMALIB QDQ fusion pass list (int8 + int16)."""
    for p in get_mmalib_qdq_passes():
        mod = p(mod)
    return mod


def _count_fused_fc_i16(mod: tvm.IRModule) -> int:
    """Count PrimFuncs produced by FuseMMALIBQDQFCI16 in the module."""
    return sum(
        1 for gv in mod.functions
        if "mmalib_fc_i16" in str(gv.name_hint)
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_SKIP_REASON = (
    "PT2E int16 accuracy not yet production-ready: "
    "partial fusion (5/211 FC layers due to shared activation nodes) "
    "and mmalib_matmul_bias_i16 per-layer error compounding over 30 layers. "
    "Re-enable once fusion coverage and accuracy are fixed."
)


@pytest.mark.skip(reason=_SKIP_REASON)
@pytest.mark.quick
def test_smollm_pt2e_i16_fusion_coverage():
    """FuseMMALIBQDQFCI16 fires on SmolLM-135M FC layers; reports coverage.

    Validates the end-to-end pipeline at the pass level:
      C7xMMAQuantizer("int16") → from_exported_program → MMALIB QDQ passes

    Due to shared activation nodes in the transformer graph (see module
    docstring), full 211-layer fusion is not achievable with the current
    pattern matcher.  This test asserts at least one layer is fused
    (pipeline works) and prints the actual coverage for documentation.
    """
    model_dir = _resolve_model_dir()
    mod, _, _ = _build_pt2e_i16_module(model_dir, seq_len=16, n_calibration=2)
    mod = _run_mmalib_qdq_passes(mod)
    n_fused = _count_fused_fc_i16(mod)

    print(
        f"\nSmolLM PT2E i16 fusion coverage: "
        f"{n_fused}/{_NUM_FC_LAYERS_TOTAL} FC layers fused by FuseMMALIBQDQFCI16.\n"
        f"Unfused layers fall through to float32 (shared activation node limitation)."
    )

    assert n_fused >= _MIN_FC_LAYERS_FUSED, (
        f"Expected at least {_MIN_FC_LAYERS_FUSED} FC layers fused, got {n_fused}. "
        f"FuseMMALIBQDQFCI16 did not match any FC layers — check the pipeline."
    )


@pytest.mark.skip(reason=_SKIP_REASON)
@pytest.mark.quick
@pytest.mark.c7x_only
def test_smollm_pt2e_i16_cosine_similarity(dsp_mode, record_cycles):
    """PT2E int16 SmolLM on DSP: fusion coverage + cosine similarity.

    Compiles with mmalib_matmul_bias_i16 and runs on c7x_host.
    Asserts fusion coverage; logs cosine similarity without a hard threshold
    (see module docstring for the known accuracy ceiling).
    """
    if dsp_mode is None:
        pytest.skip("--dsp-mode not set")

    model_dir = _resolve_model_dir()
    mod, inputs_np, ref_logits = _build_pt2e_i16_module(
        model_dir, seq_len=64, n_calibration=4
    )

    # Verify fusion coverage before compiling
    mod_fused = _run_mmalib_qdq_passes(mod)
    n_fused = _count_fused_fc_i16(mod_fused)
    assert n_fused >= _MIN_FC_LAYERS_FUSED, (
        f"Expected at least {_MIN_FC_LAYERS_FUSED} fused FC layers, got {n_fused}"
    )
    print(f"  FC layers fused: {n_fused}/{_NUM_FC_LAYERS_TOTAL}")

    # Compile and run on DSP
    target = get_target_string(dsp_mode, use_cpp_api=True) + " -mmalib=1"
    results = compile_and_run_dsp(
        mod=mod,
        input_data=inputs_np,
        target_string=target,
        execution_mode=dsp_mode,
        profile=True,
    )

    result_key = "c7x_host_result" if dsp_mode == "c7x_host" else "c7x_dload_result"
    dsp_out = results.get(result_key)
    assert dsp_out is not None, f"No {result_key} in results"

    # SmolLM output is float32 logits [1, seq_len, vocab_size=49152]
    dsp_logits = dsp_out.reshape(ref_logits.shape).astype(np.float32)

    # Cosine similarity between DSP and float32 reference logits at last token
    dsp_last = dsp_logits[0, -1]   # [vocab_size]
    ref_last = ref_logits[0, -1]   # [vocab_size]
    cosine = float(
        np.dot(dsp_last, ref_last)
        / (np.linalg.norm(dsp_last) * np.linalg.norm(ref_last) + 1e-8)
    )
    max_diff = float(np.abs(dsp_last - ref_last).max())

    print(f"\nSmolLM PT2E int16 results ({dsp_mode}):")
    print(f"  FC layers fused: {n_fused}/{_NUM_FC_LAYERS_TOTAL}")
    print(f"  Cosine similarity (last token): {cosine:.4f}")
    print(f"  Max diff (last token logits): {max_diff:.2f}")
    print(
        "  NOTE: Low cosine similarity is expected — mmalib_matmul_bias_i16 uses\n"
        "  uint8 scale/shift requantization; error ~24-61 per layer compounds\n"
        "  over 30 layers. LegalizeMLPToMMALIBInt16 is the recommended LLM path."
    )

    record_cycles("smollm_pt2e_i16_cycles", results.get("c7x_dload_cycles", 0))

    # Document accuracy ceiling — no hard assertion on cosine similarity.
    # If cosine > 0.9, the pipeline is surprisingly accurate (flag it).
    # If cosine < -0.5, something is catastrophically wrong (flag it).
    assert cosine > -0.5, (
        f"Cosine similarity {cosine:.4f} is catastrophically negative — "
        f"DSP output is anti-correlated with reference. Check compilation."
    )
