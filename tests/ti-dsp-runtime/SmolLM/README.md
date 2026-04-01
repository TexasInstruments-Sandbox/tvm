# SmolLM-135M on C7x DSP

See the full design document at `docs/dsp/smol.md` (repo root).

## Status

| Variant | c7x_host | c7x_dload |
|---------|----------|-----------|
| Float32 (~621 MB weights) | PASS (max diff 0.21) | ELF too large |
| INT8 weight-only (~333 MB weights) | PASS (max diff 0.19) | PASS (max diff 0.19, --fp_reassoc=off) |

## Quick Start

```bash
export TI_CGT_C7000_PATH=~/ti/ccs2041/ccs/tools/compiler/ti-cgt-c7000_5.0.1.LTS
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH

# Download weights once (if not already present in model/)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
m = AutoModelForCausalLM.from_pretrained(
    'HuggingFaceTB/SmolLM-135M-Instruct', dtype=torch.float32)
t = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-135M-Instruct')
m.save_pretrained('tests/ti-dsp-runtime/SmolLM/model')
t.save_pretrained('tests/ti-dsp-runtime/SmolLM/model')
"

# c7x_host (host emulation)
python smollm_c7x.py test --quantize

# c7x_dload (AM67A hardware) — --fp-reassoc-off required for correct logits
python smollm_c7x.py compile --quantize --dsp-mode c7x_dload \
                              --fp-reassoc-off -o /tmp/smol_dload
python smollm_c7x.py infer   --artifacts /tmp/smol_dload --dsp-mode c7x_dload

# Interactive chat on AM67A
python smollm_c7x.py compile-chat --quantize --dsp-mode c7x_dload \
                                   -o /tmp/smol_chat
python smollm_c7x.py deploy --artifacts /tmp/smol_chat
ssh root@am67a python3 /opt/smollm/smollm_board.py --model-dir /opt/smollm
```

## Scripts

| Script | Purpose |
|--------|---------|
| `smollm_c7x.py` | Main compile/infer/test/deploy script |
| `smollm_board.py` | ARM-side chat loop with KV cache (runs on AM67A) |
| `test_smollm_layers.py` | Layer-sweep diagnostic (`--fp32` proves quantization is not the cause of divergence) |
| `test_component_divergence.py` | Per-op correctness with random weights on c7x_host and c7x_dload |
