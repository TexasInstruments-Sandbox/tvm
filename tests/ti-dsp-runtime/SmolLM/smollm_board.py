#!/usr/bin/env python3
"""
SmolLM-135M chat — runs on the AM67A ARM side.  No TVM or PyTorch needed.

Requires only python3 + numpy + tokenizers (HuggingFace fast tokenizers).
All inference runs on the C7x DSP via local c7x_compute subprocess calls
(no SSH, no network).  The KV cache lives in ARM RAM as numpy arrays and
is written/read via /tmp (tmpfs) — ~5 ms for 11 MB vs ~500 ms over SSH.

Compile the models on the PC first:
    python smollm_c7x.py compile-chat --quantize --fp-reassoc-off \\
        --max-cache-len 256 -o /tmp/smol_chat

Deploy to board (weights are embedded in the ELF, only 4 files needed):
    python smollm_c7x.py deploy --artifacts /tmp/smol_chat --target root@am67a:/opt/smollm
    # OR manually:
    ssh root@am67a "mkdir -p /opt/smollm"
    scp /tmp/smol_chat/prefill/build-dynmod/lib0.out root@am67a:/opt/smollm/prefill.out
    scp /tmp/smol_chat/decode/build-dynmod/lib0.out  root@am67a:/opt/smollm/decode.out
    scp /tmp/smol_chat/tokenizer.json                root@am67a:/opt/smollm/
    scp /tmp/smol_chat/metadata.json                 root@am67a:/opt/smollm/
    scp tests/ti-dsp-runtime/SmolLM/smollm_board.py  root@am67a:/opt/smollm/

Run on board:
    python3 /opt/smollm/smollm_board.py --model-dir /opt/smollm
    python3 /opt/smollm/smollm_board.py --model-dir /opt/smollm --max-tokens 100
"""

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Tensor I/O helpers (matches dsp_utils.py TVMT format)
# ---------------------------------------------------------------------------

_MAGIC = 0x54564D54  # "TVMT"
_VERSION = 1

_DTYPE_CODE_TO_NUMPY = {
    (2, 32): np.float32,
    (2, 16): np.float16,
    (0, 64): np.int64,
    (0, 32): np.int32,
    (0, 8): np.int8,
    (1, 8): np.uint8,
}
_NUMPY_TO_DTYPE_CODE = {
    np.dtype("float32"): (2, 32),
    np.dtype("float16"): (2, 16),
    np.dtype("int64"): (0, 64),
    np.dtype("int32"): (0, 32),
    np.dtype("int8"): (0, 8),
    np.dtype("uint8"): (1, 8),
}
_DTYPE_STR = {
    np.dtype("float32"): "float32",
    np.dtype("float16"): "float16",
    np.dtype("int64"): "int64",
    np.dtype("int32"): "int32",
    np.dtype("int8"): "int8",
    np.dtype("uint8"): "uint8",
}


def _write_tensors(tensors, path):
    """Write list of numpy arrays to TVMT binary file."""
    with open(path, "wb") as f:
        f.write(struct.pack("<III", _MAGIC, _VERSION, len(tensors)))
        for arr in tensors:
            arr = np.ascontiguousarray(arr)
            code, bits = _NUMPY_TO_DTYPE_CODE[arr.dtype]
            f.write(struct.pack("<i", arr.ndim))
            for d in arr.shape:
                f.write(struct.pack("<q", d))
            f.write(struct.pack("<ii", code, bits))
            data = arr.tobytes()
            f.write(struct.pack("<q", len(data)))
            f.write(data)


def _read_tensors(path):
    """Read list of numpy arrays from TVMT binary file."""
    with open(path, "rb") as f:
        magic, version, n = struct.unpack("<III", f.read(12))
        if magic != _MAGIC:
            raise ValueError(f"Bad magic: 0x{magic:08X}")
        arrays = []
        for _ in range(n):
            (ndim,) = struct.unpack("<i", f.read(4))
            shape = struct.unpack(f"<{ndim}q", f.read(8 * ndim))
            code, bits = struct.unpack("<ii", f.read(8))
            (data_size,) = struct.unpack("<q", f.read(8))
            raw = f.read(data_size)
            dtype = _DTYPE_CODE_TO_NUMPY.get((code, bits))
            if dtype is None:
                raise ValueError(f"Unknown dtype code={code} bits={bits}")
            arrays.append(np.frombuffer(raw, dtype=dtype).reshape(shape).copy())
    return arrays


# ---------------------------------------------------------------------------
# Local c7x_compute runner
# ---------------------------------------------------------------------------


def _run_local(module_path, input_arrays, work_dir, c7x_compute, timeout_s=600):
    """Call c7x_compute locally and return list of output numpy arrays."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    input_bin = work / "input.bin"
    output_bin = work / "output.bin"

    # Write all inputs as raw concatenated bytes (c7x_compute flat format)
    with open(input_bin, "wb") as f:
        for arr in input_arrays:
            f.write(np.ascontiguousarray(arr).tobytes())

    shapes = ";".join(",".join(str(d) for d in arr.shape) for arr in input_arrays)
    dtypes = ";".join(_DTYPE_STR[arr.dtype] for arr in input_arrays)

    cmd = [
        c7x_compute,
        "run",
        "--module",
        str(module_path),
        "--input",
        str(input_bin),
        "--output",
        str(output_bin),
        "--shape",
        shapes,
        "--dtype",
        dtypes,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)

    if result.returncode != 0:
        raise RuntimeError(
            f"c7x_compute failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    # Parse JSON result
    out = result.stdout.strip()
    js_start = out.find("{")
    if js_start < 0:
        raise RuntimeError(f"No JSON in output: {out}")
    info = json.loads(out[js_start:])
    if info.get("status") != "ok":
        raise RuntimeError(f"Inference failed: {info.get('error', 'unknown')}")

    cycles = info.get("cycles", 0)

    # Read all output tensors from the binary file
    raw = output_bin.read_bytes()
    outputs_meta = sorted(info.get("outputs", []), key=lambda m: m["index"])
    if not outputs_meta:
        raise ValueError("No outputs in result JSON")

    _code_bits = {
        (2, 32): np.float32,
        (2, 16): np.float16,
        (0, 64): np.int64,
        (0, 32): np.int32,
        (0, 8): np.int8,
        (1, 8): np.uint8,
    }
    offset = 0
    arrays = []
    for m in outputs_meta:
        dtype = _code_bits[(m["dtype_code"], m["dtype_bits"])]
        size = m["data_size"]
        arr = np.frombuffer(raw[offset : offset + size], dtype=dtype)
        arrays.append(arr.reshape(m["shape"]).copy())
        offset += size

    return arrays, cycles


# ---------------------------------------------------------------------------
# KV cache management
# ---------------------------------------------------------------------------


class KVCache:
    """Manages the 60 KV cache buffers for SmolLM-135M on ARM.

    The model's forward pass has 62 inputs:
        input_ids[1, seq], cache_position[seq],
        key_cache_0..29[1, num_kv_heads, max_len, head_dim],
        val_cache_0..29[1, num_kv_heads, max_len, head_dim]

    And 61 outputs:
        logits[1, seq, vocab],
        key_cache_0..29_updated, val_cache_0..29_updated

    This class stores the 60 buffers as numpy arrays and provides
    methods to pack them into a flat input list and update them from
    the flat output list.
    """

    def __init__(self, num_layers, num_kv_heads, max_cache_len, head_dim):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.max_cache_len = max_cache_len
        self.head_dim = head_dim
        # Each buffer: [1, num_kv_heads, max_cache_len, head_dim]
        shape = (1, num_kv_heads, max_cache_len, head_dim)
        self.keys = [np.zeros(shape, dtype=np.float32) for _ in range(num_layers)]
        self.vals = [np.zeros(shape, dtype=np.float32) for _ in range(num_layers)]

    def reset(self):
        """Zero all cache buffers (start of new conversation)."""
        for k in self.keys:
            k[:] = 0
        for v in self.vals:
            v[:] = 0

    def as_inputs(self):
        """Return ordered list matching model input order: key_0..29, val_0..29."""
        return self.keys + self.vals

    def update_from_outputs(self, outputs):
        """Update cache from scatter_elements output tensors.

        outputs[0] is logits.  outputs[1:] are the updated KV tensors in
        parameter order.  The scatter_elements results are sorted by the
        order of the lifted_tensor_* parameters, which alternate
        key/value across layers as they appear in the FX graph.

        The outputs are interleaved: for each layer, key_i comes before
        val_i in the parameter list (consistent with StaticCache internals).
        Half the outputs are keys, half are values.
        """
        kv_outputs = outputs[1:]
        n_kv = len(kv_outputs)
        if n_kv == 0:
            return
        # KV outputs are sorted by lifted_tensor parameter order.
        # StaticCache registers k and v separately per layer; the order
        # depends on the FX graph.  Handle both n_kv==2*n_layers (split)
        # and n_kv==n_layers cases gracefully.
        n = self.num_layers
        if n_kv == 2 * n:
            # First half = keys, second half = values (sorted by layer)
            for i in range(n):
                np.copyto(self.keys[i], kv_outputs[i])
            for i in range(n):
                np.copyto(self.vals[i], kv_outputs[n + i])
        elif n_kv == n:
            # Only keys or only values returned (unusual)
            for i in range(n):
                np.copyto(self.keys[i], kv_outputs[i])

    @property
    def size_mb(self):
        total = 2 * self.num_layers * self.num_kv_heads * self.max_cache_len * self.head_dim * 4
        return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# SmolLM inference engine
# ---------------------------------------------------------------------------


class SmolLMEngine:
    """SmolLM-135M inference on the AM67A ARM/C7x.

    The engine holds:
    - KV cache as numpy arrays in ARM RAM
    - Paths to prefill.out and decode.out compiled DLOAD modules
    - A tokenizers.Tokenizer loaded from tokenizer.json

    Prompt processing uses the prefill model; token-by-token
    generation uses the decode model.

    Example:
        engine = SmolLMEngine.from_dir("/opt/smollm")
        for token in engine.generate("Tell me a joke"):
            print(token, end="", flush=True)
        print()
    """

    def __init__(
        self,
        prefill_out,
        decode_out,
        tokenizer,
        metadata,
        c7x_compute="/usr/local/bin/c7x_compute",
        work_dir="/tmp/c7x_smollm",
    ):
        self.prefill_out = Path(prefill_out)
        self.decode_out = Path(decode_out)
        self.tokenizer = tokenizer
        self.c7x_compute = c7x_compute
        self.work_dir = work_dir

        self.prefill_len = metadata["prefill_len"]
        self.max_cache_len = metadata["max_cache_len"]
        self.eos_token_id = metadata.get("eos_token_id", 0)
        self.vocab_size = metadata.get("vocab_size", 49152)

        num_layers = metadata.get("num_layers", 30)
        num_kv_heads = metadata.get("num_kv_heads", 3)
        head_dim = metadata.get("head_dim", 64)

        self.cache = KVCache(num_layers, num_kv_heads, self.max_cache_len, head_dim)
        self.cache_pos = 0  # next position to write into the KV cache

    @classmethod
    def from_dir(
        cls,
        model_dir,
        c7x_compute="/usr/local/bin/c7x_compute",
        work_dir="/tmp/c7x_smollm",
    ):
        """Load engine from an artifacts directory produced by compile-chat."""
        from tokenizers import Tokenizer  # noqa: PLC0415

        model_dir = Path(model_dir)
        meta_path = model_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {model_dir}. "
                "Run 'python smollm_c7x.py compile-chat' on the PC first."
            )
        with open(meta_path) as f:
            metadata = json.load(f)

        tok_path = model_dir / "tokenizer.json"
        if not tok_path.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_dir}")
        tokenizer = Tokenizer.from_file(str(tok_path))

        # Find prefill/decode modules
        prefill_out = model_dir / "prefill.out"
        decode_out = model_dir / "decode.out"
        for p in (prefill_out, decode_out):
            if not p.exists():
                raise FileNotFoundError(f"DLOAD module not found: {p}")

        return cls(
            prefill_out,
            decode_out,
            tokenizer,
            metadata,
            c7x_compute=c7x_compute,
            work_dir=work_dir,
        )

    def _run(self, module_path, user_inputs):
        """Run a model; user_inputs are [input_ids, cache_position].
        KV cache tensors are appended automatically from self.cache."""
        all_inputs = user_inputs + self.cache.as_inputs()
        outputs, cycles = _run_local(module_path, all_inputs, self.work_dir, self.c7x_compute)
        return outputs, cycles

    def prefill(self, token_ids):
        """Process a prompt (list of token IDs), populate KV cache.

        The prompt is padded or chunked to fit the fixed prefill_len.
        Returns the logits for the last prompt position as a 1D array.
        """
        # Pad or truncate to prefill_len
        n = len(token_ids)
        if n > self.prefill_len:
            # Chunked prefill: process in prefill_len windows
            # Simple approach: truncate to last prefill_len tokens
            token_ids = token_ids[-self.prefill_len :]
            n = self.prefill_len
            self.cache_pos = 0

        pad_ids = token_ids + [0] * (self.prefill_len - n)
        input_ids = np.array(pad_ids, dtype=np.int64).reshape(1, self.prefill_len)
        cache_pos = np.arange(self.prefill_len, dtype=np.int64)

        self.cache.reset()
        self.cache_pos = 0

        outputs, cycles = self._run(self.prefill_out, [input_ids, cache_pos])
        self.cache.update_from_outputs(outputs)
        self.cache_pos = self.prefill_len

        # Return logits at the last real token position (n-1)
        logits = outputs[0]  # [1, prefill_len, vocab]
        return logits[0, n - 1, :]

    def decode_step(self, token_id):
        """Decode one token, update KV cache, return next-token logits."""
        if self.cache_pos >= self.max_cache_len:
            raise RuntimeError(
                f"KV cache full ({self.max_cache_len} tokens). "
                "Start a new conversation or increase --max-cache-len."
            )
        input_ids = np.array([[token_id]], dtype=np.int64)
        cache_pos = np.array([self.cache_pos], dtype=np.int64)

        outputs, cycles = self._run(self.decode_out, [input_ids, cache_pos])
        self.cache.update_from_outputs(outputs)
        self.cache_pos += 1

        logits = outputs[0]  # [1, 1, vocab]
        return logits[0, 0, :]

    def generate(self, prompt, max_new_tokens=200, temperature=1.0, top_k=0):
        """Generate text from a prompt, yielding decoded text incrementally.

        Args:
            prompt: Input text string.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (1.0 = greedy, <1.0 = sharper).
            top_k: If > 0, restrict sampling to top-k logits.

        Yields:
            Decoded text fragments as they are generated.
        """
        token_ids = self.tokenizer.encode(prompt).ids

        # Prefill
        logits = self.prefill(token_ids)
        next_token = _sample(logits, temperature=temperature, top_k=top_k)

        yield self.tokenizer.decode([next_token])

        for _ in range(max_new_tokens - 1):
            if next_token == self.eos_token_id:
                break
            logits = self.decode_step(next_token)
            next_token = _sample(logits, temperature=temperature, top_k=top_k)
            if next_token == self.eos_token_id:
                break
            yield self.tokenizer.decode([next_token])


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _sample(logits, temperature=1.0, top_k=0):
    """Sample the next token from logits.

    Args:
        logits: 1D numpy array of shape [vocab_size].
        temperature: Divide logits by this value before softmax.
                     temperature=1.0 → standard softmax.
                     temperature → 0 → greedy (argmax).
        top_k: If > 0, zero all but the top-k logits before sampling.

    Returns:
        Sampled token id (int).
    """
    if temperature <= 0 or temperature < 1e-6:
        return int(np.argmax(logits))

    logits = logits.astype(np.float64)
    logits /= temperature

    if top_k > 0:
        threshold = np.partition(logits, -top_k)[-top_k]
        logits[logits < threshold] = -1e10

    # Numerically stable softmax
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# Chat REPL
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = "You are SmolLM, a helpful and honest AI assistant running on a TI AM67A DSP."


def chat(engine, max_tokens=200, temperature=1.0, top_k=0):
    """Interactive chat loop using a simple conversation history."""
    print(
        f"SmolLM-135M  |  max_cache={engine.max_cache_len} tokens  |  "
        f"cache={engine.cache.size_mb:.1f} MB"
    )
    print("Type 'quit' or Ctrl-C to exit, 'reset' to start a new conversation.\n")

    history = _SYSTEM_PROMPT
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user:
            continue
        if user.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if user.lower() == "reset":
            history = _SYSTEM_PROMPT
            engine.cache.reset()
            engine.cache_pos = 0
            print("[Conversation reset]\n")
            continue

        # Build prompt with conversation history
        prompt = f"{history}\n\nUser: {user}\n\nAssistant:"
        print("Assistant: ", end="", flush=True)

        response_tokens = []
        try:
            for fragment in engine.generate(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            ):
                print(fragment, end="", flush=True)
                response_tokens.append(fragment)
        except RuntimeError as exc:
            print(f"\n[Error: {exc}]")

        response = "".join(response_tokens)
        print()

        # Append turn to history for next round
        history = f"{prompt}{response}"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SmolLM-135M chat on AM67A (ARM-local KV cache)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --model-dir /opt/smollm
  %(prog)s --model-dir /opt/smollm --max-tokens 100 --temperature 0.8
  %(prog)s --model-dir /opt/smollm --prompt "What is the capital of France?"
""",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing prefill.out, decode.out, tokenizer.json, metadata.json",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum new tokens to generate per turn (default: 200)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (default: 1.0, use 0 for greedy)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50, use 0 to disable)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt (non-interactive mode)",
    )
    parser.add_argument(
        "--c7x-compute",
        default="/usr/local/bin/c7x_compute",
        help="Path to c7x_compute binary (default: /usr/local/bin/c7x_compute)",
    )
    parser.add_argument(
        "--work-dir",
        default="/tmp/c7x_smollm",
        help="Tmpfs working directory for DSP I/O (default: /tmp/c7x_smollm)",
    )
    args = parser.parse_args()

    # Load engine
    print(f"Loading SmolLM from {args.model_dir} ...")
    engine = SmolLMEngine.from_dir(
        args.model_dir,
        c7x_compute=args.c7x_compute,
        work_dir=args.work_dir,
    )
    print(f"Ready.  KV cache: {engine.cache.size_mb:.1f} MB in ARM RAM\n")

    if args.prompt:
        # Single-shot mode
        for fragment in engine.generate(
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        ):
            print(fragment, end="", flush=True)
        print()
    else:
        # Interactive chat
        chat(engine, max_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)

    return 0


if __name__ == "__main__":
    sys.exit(main())
