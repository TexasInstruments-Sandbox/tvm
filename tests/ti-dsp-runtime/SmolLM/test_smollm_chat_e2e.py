"""End-to-end SmolLM chat test on AM67A hardware.

Compiles the model, deploys to board, runs inference with the standard
test prompt, and verifies both accuracy (output text) and performance
(tok/s threshold).

Usage:
    pytest test_smollm_chat_e2e.py -v --dsp-mode=c7x_dload

Requirements:
    - AM67A board reachable as root@am67a
    - Firmware deployed (c7x_compute.out)
    - TI_CGT_C7000_PATH set
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).parent


def _resolve_model_dir() -> Path:
    """Resolve model directory: env var > cache dir > sibling dir."""
    if env := os.environ.get("SMOLLM_MODEL_DIR"):
        return Path(env)
    cache_dir = Path.home() / ".cache" / "smollm" / "SmolLM-135M-Instruct"
    if cache_dir.exists():
        return cache_dir
    return _THIS_DIR / "model"


_MODEL_DIR = _resolve_model_dir()
_BOARD_TARGET = "root@am67a"
_BOARD_MODEL_DIR = "/opt/smollm"
_TEST_PROMPT = "What is the capital of France?"
_EXPECTED_SUBSTR = "Paris"
_MIN_TOK_PER_SEC = 2.0
_MAX_TOKENS = 50


@pytest.mark.c7x_only
@pytest.mark.smollm
def test_smollm_chat_accuracy_and_performance(dsp_mode, record_cycles):
    """Compile, deploy, run SmolLM chat and verify output + throughput."""
    if dsp_mode != "c7x_dload":
        pytest.skip("SmolLM e2e requires c7x_dload (AM67A hardware)")
    if not _MODEL_DIR.exists() or not (_MODEL_DIR / "config.json").exists():
        pytest.skip(f"SmolLM model weights not found at {_MODEL_DIR}")

    artifacts_dir = Path(tempfile.mkdtemp(prefix="smollm_ci_"))

    # Step 1: Compile
    compile_cmd = [
        "python", str(_THIS_DIR / "smollm_c7x.py"), "compile-chat",
        "--model-dir", str(_MODEL_DIR),
        "--quantize",
        "--dsp-mode", "c7x_dload",
        "--prefill-len", "64",
        "--max-cache-len", "256",
        "-o", str(artifacts_dir),
    ]
    # 700s (was 600s): the FuseQDQToC7xMovement ConstReachability guard
    # (see ti_fuse_qdq_c7x_movement.py) fixed the swin_s/swin_t segfault
    # without needing a second full-graph FoldConstant pass, but compile
    # time on this box still lands close to 600s -- bump for headroom
    # against machine-to-machine variance.
    result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=700)
    assert result.returncode == 0, f"Compile failed:\n{result.stderr[-500:]}"

    # Step 2: Deploy
    deploy_cmd = [
        "python", str(_THIS_DIR / "smollm_c7x.py"), "deploy",
        "--artifacts", str(artifacts_dir),
        "--target", f"{_BOARD_TARGET}:{_BOARD_MODEL_DIR}",
    ]
    result = subprocess.run(deploy_cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"Deploy failed:\n{result.stderr[-500:]}"

    # Step 3: Run inference on board
    board_cmd = (
        f"python3 {_BOARD_MODEL_DIR}/smollm_board.py "
        f"--model-dir {_BOARD_MODEL_DIR} "
        f"--prompt '{_TEST_PROMPT}' "
        f"--max-tokens {_MAX_TOKENS} --temperature 0"
    )
    result = subprocess.run(
        ["ssh", _BOARD_TARGET, board_cmd],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"Board inference failed:\n{result.stderr[-500:]}"

    # Parse output: stdout has the generated text + stats on stderr
    output_text = result.stdout.strip()
    # Stats line is on stderr: [N prompt tok / Xs prefill | M gen tok / Ys decode | Z tok/s]
    stats_line = ""
    for line in result.stderr.split("\n"):
        if "tok/s" in line:
            stats_line = line
            break

    # Step 4: Verify accuracy — output should contain the expected prefix
    # The output includes session startup messages; find the actual generated text
    # Generated text appears after "ready\n" from the decode session
    generated = output_text
    if "ready\n" in generated:
        generated = generated.split("ready\n", 1)[-1].strip()
    elif "ready" in generated:
        # Sometimes "ready" is on same line
        parts = generated.split("ready", 1)
        generated = parts[-1].strip()

    print(f"\n  Generated text: {generated[:200]}")
    assert _EXPECTED_SUBSTR in generated, (
        f"Accuracy check failed: expected '{_EXPECTED_SUBSTR}' in output.\n"
        f"Got: {generated[:200]}"
    )

    # Step 5: Verify performance
    tok_per_sec = 0.0
    if stats_line:
        match = re.search(r"([\d.]+)\s*tok/s", stats_line)
        if match:
            tok_per_sec = float(match.group(1))

    print(f"  Throughput: {tok_per_sec:.2f} tok/s (min: {_MIN_TOK_PER_SEC})")
    print(f"  Stats: {stats_line}")

    if record_cycles and tok_per_sec > 0:
        record_cycles("smollm_chat_tok_per_sec", int(tok_per_sec * 1000))

    assert tok_per_sec >= _MIN_TOK_PER_SEC, (
        f"Performance regression: {tok_per_sec:.2f} tok/s < {_MIN_TOK_PER_SEC} tok/s threshold"
    )

    if not os.environ.get("DSP_KEEP_TEMP"):
        shutil.rmtree(artifacts_dir, ignore_errors=True)
