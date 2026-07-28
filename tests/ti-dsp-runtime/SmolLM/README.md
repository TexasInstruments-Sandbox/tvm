# SmolLM-135M on C7x DSP

See the full design documents (TI-internal only — see `README_TI.md`'s
[Documentation](../../../README_TI.md#documentation) section):
- `smollm_overview.md` — standalone inference pipeline and model overview
- `smollm_kv_cache.md` — KV cache chat design, IPC protocol, and session details

## Status

| Variant | c7x_host | c7x_dload |
|---------|----------|-----------|
| Float32 (~621 MB weights) | PASS (max diff 0.21) | ELF too large |
| INT8 weight-only (~333 MB weights) | PASS (max diff 0.19) | PASS (max diff 0.19, --fp_reassoc=off) |
| INT8 KV cache chat | PASS | PASS (~1.8 s/token on AM67A) |

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
                                   --fp-reassoc-off \
                                   --prefill-len 16 --max-cache-len 32 \
                                   -o /tmp/smol_chat
python smollm_c7x.py deploy --artifacts /tmp/smol_chat --target root@am67a:/opt/smollm
ssh root@am67a python3 /opt/smollm/smollm_board.py --model-dir /opt/smollm
```

## Chat on AM67A

`smollm_board.py` runs entirely on the ARM Cortex-A53 (no TVM or PyTorch
required on the board).  It uses a persistent DSP session so the 333 MB
ELF is loaded once (~35 s) and subsequent decode steps each take ~1.8 s.

### Performance

| Phase | Time |
|-------|------|
| Prefill (one-shot load+infer) | ~40 s |
| Session ELF load (one-time) | ~35 s |
| Decode per token | ~1.8 s |
| 15-token response (end-to-end) | ~95 s |

### KV Cache and Context Limit

The models are compiled with **static shapes** — the KV cache size is fixed
at compile time via `--max-cache-len`.  With `prefill_len=16` tokens and
`max_cache_len=32`, the cache holds up to 32 positions total: 16 for the
prompt and 16 for generated tokens.  Requesting more tokens than the
remaining slots prints a clean message and stops generation.

To increase the context window, recompile with a larger cache:

```bash
python smollm_c7x.py compile-chat --quantize --fp-reassoc-off \
    --dsp-mode c7x_dload \
    --prefill-len 16 --max-cache-len 256 \
    -o /tmp/smol_chat_256
```

| `--max-cache-len` | KV cache size | Max new tokens |
|-------------------|--------------|----------------|
| 32 | 1.4 MB | ~16 |
| 256 (default) | 11 MB | ~240 |
| 512 | 22 MB | ~496 |

Formula: `2 × 30 layers × 3 KV heads × max_cache_len × 64 head_dim × 4 B`

### smollm_board.py options

```
python3 smollm_board.py --model-dir /opt/smollm [options]

  --max-tokens N       Maximum new tokens to generate (default: 200)
  --temperature T      Sampling temperature: 0 = greedy (default: 1.0)
  --top-k K            Top-k sampling; 0 = disabled (default: 50)
  --prompt TEXT        Single-shot mode; omit for interactive chat
  --c7x-compute PATH   Path to c7x_compute binary
  --profile            Print per-stage timing breakdown for each decode step
  --work-dir PATH      Tmpfs working directory for DSP I/O (default: /tmp/c7x_smollm)
```

## Scripts

| Script | Purpose |
|--------|---------|
| `smollm_c7x.py` | Main compile/infer/test/deploy script |
| `smollm_board.py` | ARM-side chat loop with KV cache (runs on AM67A) |
| `test_smollm_layers.py` | Layer-sweep diagnostic |
| `test_component_divergence.py` | Per-op correctness on c7x_host and c7x_dload |
