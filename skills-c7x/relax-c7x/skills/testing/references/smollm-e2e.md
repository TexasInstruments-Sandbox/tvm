# SmolLM E2E Test Pipeline

## Commands

```bash
cd tests/ti-dsp-runtime/SmolLM

# Compile (generates DLOAD ELFs + metadata)
python smollm_c7x.py compile --quantize -o /tmp/smol_int8
python smollm_c7x.py compile-chat --quantize -o /tmp/smol_chat  # prefill + decode

# Single-shot test (compile + infer + verify)
python smollm_c7x.py test --quantize --dsp-mode c7x_host
python smollm_c7x.py test --quantize --dsp-mode c7x_dload

# Deploy to board (SCP 4 files, ~642 MB)
python smollm_c7x.py deploy --artifacts /tmp/smol_chat

# Board-side inference (runs on AM67A Cortex-A53, no TVM/torch)
ssh root@am67a python3 /opt/smollm/smollm_board.py --model-dir /opt/smollm
```

## Board-Side Requirements

`smollm_board.py` needs only: `python3`, `numpy`, `tokenizers` (Rust-based).
No TVM, no PyTorch. Tokenizer shipped as `tokenizer.json` (3.4 MB).

## Test Variants

| Test | What it verifies |
|------|-----------------|
| `test --quantize` | INT8 weight-only, single inference, logit match |
| `smollm_w16a16.py test --dsp-mode c7x_dload` | MMALIB int16 offload path (separate script, not a `smollm_c7x.py` flag) |
| `compile-chat` + board inference | Full KV cache chat loop |

## Key Files

| File | Purpose |
|------|---------|
| `tests/ti-dsp-runtime/SmolLM/smollm_c7x.py` | Main CLI: compile/infer/test/compile-chat/deploy |
| `tests/ti-dsp-runtime/SmolLM/smollm_board.py` | Board-side inference (no TVM) |
| `tests/ti-dsp-runtime/SmolLM/smollm_w16a16.py` | MMALIB int16 offload test script (`mmalib_matmul_i16`) |
| `tests/ti-dsp-runtime/SmolLM/model/` | HuggingFace model weights (default lookup order: `$SMOLLM_MODEL_DIR` env var, then `~/.cache/smollm/SmolLM-135M-Instruct`, then this directory) |
