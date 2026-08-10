#!/usr/bin/env bash
# Python venv + test dependency setup for the TI DSP test suite. Extracted
# from tests/ti-dsp-runtime/Jenkinsfile's "Python Setup" stage so the
# docker-based validate flow (src/runtime/ti_dsp/validate_all.sh) and that
# native stage install the exact same pinned versions -- this project has
# already been bitten once by numpy version drift between two
# separately-maintained install paths (see the tvm-relax-c7x:jenkins-debug
# skill's "100% reproducible on Jenkins but 0% locally" section).
#
# Meant to be `source`d (it needs to leave VIRTUAL_ENV/PATH set in the
# calling shell for a subsequent pytest invocation), but running it
# directly also leaves a usable venv on disk.
#
# Assumes cwd is the repo root -- same assumption build_all.sh and the
# Jenkinsfile's own sh steps make (docker/bash.sh sets --workdir to the
# repo mount point, so this holds whether invoked directly or via
# `docker/bash.sh <image> -- bash tests/ti-dsp-runtime/setup_test_env.sh`).
#
# Usage:
#   source tests/ti-dsp-runtime/setup_test_env.sh
#   VENV_DIR=.venv-ci-c7x source tests/ti-dsp-runtime/setup_test_env.sh
#
# VENV_DIR defaults to .venv -- the same path the Jenkinsfile's Python Setup
# stage uses. The docker validate flow overrides it to .venv-ci-c7x: the
# repo is bind-mounted (not copied) into the container, so a literal .venv
# at the repo root would be the same path a developer's own local dev venv
# lives at, and running this against a personal checkout would reuse or
# clobber it with these CI-pinned versions.
#
# TORCH_HOME defaults to $HOME/.cache/torch, matching the Jenkinsfile's own
# default -- but $HOME inside a docker/bash.sh container is the repo mount
# point, not the host user's real home, so a containerized caller (e.g.
# validate_all.sh) must either export TORCH_HOME itself before sourcing
# this, or bind-mount the host's real torch cache to $HOME/.cache/torch.

set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"

if [ ! -d "${TORCH_HOME}/hub/checkpoints" ]; then
    echo "ERROR: Torchvision model cache not found at ${TORCH_HOME}/hub/checkpoints" >&2
    echo "" >&2
    echo "DSP tests require pre-cached pretrained model weights because this" >&2
    echo "machine cannot download them from download.pytorch.org." >&2
    echo "" >&2
    echo "Fix: on a machine with internet access, run:" >&2
    echo "  python -c \"from torchvision import models; [getattr(models, m)(weights='DEFAULT') for m in ['squeezenet1_1','shufflenet_v2_x1_0','mobilenet_v2','mobilenet_v3_small','mobilenet_v3_large','efficientnet_b0','densenet121','resnet34','resnet18']]\"" >&2
    echo "  scp -r ~/.cache/torch <user>@<host>:${TORCH_HOME}" >&2
    return 1 2>/dev/null || exit 1
fi

test -d "${VENV_DIR}" || uv venv "${VENV_DIR}"
export VIRTUAL_ENV="$(cd "${VENV_DIR}" && pwd)"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

uv pip install "numpy==2.5.1" pytest pytest-isolate scipy psutil ml_dtypes build pandas tqdm seaborn transformers
uv pip install "torch>=2.5,<2.11" "torchvision>=0.20,<0.26" --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install "torchao==0.16.0" --extra-index-url https://download.pytorch.org/whl/cpu
# Pinned: 8.4.115's export path emits a relax.cumsum VMTIRCodeGen can't
# lower for yolov8n/yolov8s/yolo26n (and a bare Relax.Constant-in-primfunc
# for yolov5n) -- 8.4.14 doesn't hit either. Re-verify before bumping.
uv pip install "ultralytics==8.4.14"
uv pip install -e 3rdparty/tvm-ffi
