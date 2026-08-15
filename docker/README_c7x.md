# C7x / BeagleY-AI Docker Build + Hardware Validation

Detailed reference for the Docker flow introduced in
[Getting Started](../docs-c7x/user-guide/getting-started.md#docker-recommended----self-contained-build-environment).
Read that first for the three commands that build the image, build the
stack, and validate on hardware -- this file covers the "why" and the
edge cases behind them.

## What `docker/bash.sh` does

`docker/bash.sh <image> -- <command>` bind-mounts this repo into the
container at the same path and runs `<command>` as your host user (via
`with_the_same_user`) -- nothing is copied into or built inside the
image itself, and build output (object files, wheels, venvs) lands in
this same working tree, owned by you. It also sets the container's
`$HOME` to that mount point, which matters for two things below: SSH
config lookup and where the test venv lands.

`docker/Dockerfile.ci_c7x` is scoped to BeagleY-AI only -- it has no
TIDL / `c7x-mma-tidl` toolchain, so it can't build the am67a/j722s-evm
path.

## `build_all.sh --wheels`

`src/runtime/ti_dsp/build_all.sh --board beagley-ai --wheels` runs the
full chain -- TVM core, DSP runtime (`c7x_host` + `c7x`), firmware
(`--tidl OFF --mmalib ON`, this board's convention), ARM client, and the
x86/arm64 packaging wheels -- as one command. TVM core builds into
`build-ci-c7x/`, not the plain `build/` name used elsewhere in this
repo, so it never collides with a native (non-container) build already
sitting in the same bind-mounted checkout. The wheels themselves just
land as files:

- `packaging/ti_dsp/staging-x86/dist/tvm_ti_c7x_compile-*.whl`
- `packaging/ti_dsp/staging-arm64/dist/tvm_ti_c7x_inference-*.whl`

`--wheels` is optional for a plain build, but required before
`validate_all.sh`, which installs and tests against the x86 wheel
rather than the source tree (see below).

## `validate_all.sh` and the test venv

`src/runtime/ti_dsp/validate_all.sh --board beagley-ai` assumes
`build_all.sh --wheels` already ran against this same checkout -- it
only deploys and tests, it doesn't build. It:

1. Creates/reuses a `.venv-ci-c7x` venv at the repo root (via
   `tests/ti-dsp-runtime/setup_test_env.sh`) and installs pytest/numpy/
   torch/etc. into it. Being bind-mounted, this venv persists across
   container runs -- it isn't rebuilt from scratch every time.
2. Installs the x86 wheel from `packaging/ti_dsp/staging-x86/dist/`
   into that venv with `uv pip install --force-reinstall --no-deps`,
   then explicitly `unset`s `PYTHONPATH`. This second step matters:
   `docker/bash.sh` injects `PYTHONPATH=<repo>/python` into any
   `*ci*`-named image (see `docker/bash.sh`'s "Set TVM import path"
   block -- generic upstream TVM behavior, not specific to this fork),
   and `PYTHONPATH` takes precedence over the venv's site-packages. Left
   alone, `import tvm` would silently resolve to the source tree instead
   of the wheel just installed, defeating the point of testing the
   packaged artifact. Same wheel and same `--force-reinstall --no-deps`
   flags as `tests/ti-dsp-runtime/Jenkinsfile`'s own native (non-Docker)
   "Build & Install Wheels" stage.
3. Reboots the board, then deploys to it. For `--board beagley-ai`
   this scp's the arm64 `tvm_ti_c7x_inference` wheel (built alongside
   the x86 one in step above) to the board, installs it there with
   `python3 -m pip install --force-reinstall --no-deps` (bootstrapping
   `python3-pip` via `apt-get` first if missing -- a stock BeagleY-AI
   image has neither `pip` nor `ensurepip`; the same proxy vars as
   below get forwarded for that `apt-get`), then runs the wheel's
   bundled `python3 -m tvm.data.ti_dsp.deploy` helper. That helper
   copies the wheel's bundled firmware image, ARM CLI binary, and
   runtime library to the same system paths (`/usr/local/bin/c7x_compute`,
   `/usr/local/lib/libc7x_arm_runtime.so`, `/lib/firmware/j722s-c71_0-fw`)
   the old scp-based deploy always used, so nothing downstream needs to
   know which path produced them. j722s-evm/am67a still deploys via
   the original scp flow (`deploy-c7x.sh` + `arm/build.sh deploy`).
   Reboots again so remoteproc autostart picks up the new firmware,
   then health-checks `c7x_compute status` (with stop/start/reboot
   recovery on failure), then runs the quantized-model test suite with
   `--mmalib --isolate` against real `c7x_dload` hardware, writing
   `results/quantized_mmalib_dload.xml`.

`--x86-wheel <path>` / `--arm64-wheel <path>` each accept an exact
`.whl` file or a directory to glob, so a wheel pair you didn't build
in this checkout -- e.g. extracted from a `c7x-build-wheels.yml`
GitHub Actions artifact -- can be deployed and validated the same way,
without running `build_all.sh` first.

## Requirements beyond the plain build

- **SSH**: `--board beagley-ai` always connects to the literal hostname
  `beagley-ai` (never an IP), so `~/.ssh/config` needs a `Host
  beagley-ai` entry (or equivalent DNS/hosts-file entry) pointing at
  wherever the board actually is -- the same board-name-to-host
  convention `deploy-c7x.sh` and the pytest suite use natively, nothing
  Docker-specific. Because `docker/bash.sh` sets the container's `$HOME`
  to the repo mount point, the `-v ~/.ssh:...` mount in the validate
  command has to land at that same path -- `$(pwd)/.ssh` when running
  by hand, or `/workspace/.ssh` under Jenkins (see
  `tests/ti-dsp-runtime/Jenkinsfile.docker`). It's just your existing
  SSH config/key already trusted by the board, not a new credential.
- **Torch cache**: `~/.cache/torch` should hold the pre-cached
  torchvision weights the quantized-model suite needs -- mount it in if
  your build environment can't reach `download.pytorch.org` directly.
- **Proxy, twice (three times for beagley-ai)**: the `--build-arg` on
  `docker build` only reaches that build step. `build_all.sh`/
  `validate_all.sh`'s own `uv pip install` calls run at container
  *runtime* (fetching from PyPI / download.pytorch.org), so the proxy
  has to be passed again as `--env` on the `docker/bash.sh`
  invocations that run them. For `--board beagley-ai`,
  `validate_all.sh` forwards that same container-side `http_proxy`/
  `https_proxy` a third time, over ssh, so the board's own `apt-get
  install python3-pip` (see above) can reach Ubuntu's mirrors -- no
  separate variable to set, it reuses whatever `--env` already put in
  the container.

## Jenkins

`tests/ti-dsp-runtime/Jenkinsfile.docker` wires the whole flow -- image
build, `build_all.sh --wheels`, `validate_all.sh` -- into a
manual-trigger-only Jenkins pipeline, so the node itself only needs
Docker, none of the native Prerequisites. It is a separate pipeline from
`tests/ti-dsp-runtime/Jenkinsfile`'s native nightly one; the two must
never run concurrently against the same physical `beagley-ai` board --
there is no shared lock between them yet, and two sessions touching the
same DSP core at once corrupts state the same way two concurrent
`c7x_dload` sessions from one pipeline would.
