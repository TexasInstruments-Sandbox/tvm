---
name: build
description: "Building TVM and all C7x targets end-to-end. Use when: building TVM core (cmake/ninja), setting up the Python environment (TVM_HOME, PYTHONPATH, uv pip install), building the DSP runtime (build_runtime.sh), building C7x firmware (build.sh, PSDK/MMALIB paths, --board/--ddr, --tidl/--mmalib), deploying firmware (deploy-c7x.sh), building the ARM client (c7x_compute), building DLOAD modules (cl7x/lnk7x), building the TIDL import library (build_j722s.sh), or troubleshooting build failures (missing SDK paths, toolchain issues, PSDK_INSTALL_PATH not set). NOT for code generation logic (see cstatic), operator implementation (see dsp-ops), or test authoring (see testing)."
---

# Building TVM for C7x

End-to-end build covering all targets from TVM core through firmware deployment. This skill overlaps with project CLAUDE.md build instructions; it adds troubleshooting, build order dependencies, and per-target detail not in CLAUDE.md.

## Two boards: `--board`/`--ddr`

`build_runtime.sh`, `firmware/c7x/dsp/build.sh`, and `firmware/c7x/arm/build.sh` all accept:

| Flag | Values | Drives |
|---|---|---|
| `--board` | **required**: `j722s-evm` or `beagley-ai` | SDK root paths/versions, default `--ddr` |
| `--ddr` | `8gb`, `4gb` (per-board default) | shared-DMA carveout physical base |

**`--board` is a required flag, not a defaulted one, on every script that
takes it** (`build_runtime.sh`, both firmware `build.sh` scripts,
`arm/build.sh`, `deploy-c7x.sh`). `j722s-evm` being the *value* most other
defaults key off (SDK paths, `--ddr`) does not mean it's implied when the
flag itself is omitted — a bare invocation exits 1 with `Error: --board
<j722s-evm|beagley-ai> is required` instead of silently assuming
`j722s-evm`. This changed at some point after this skill was first written;
older commands copied from memory or old scrollback that omit `--board`
will fail this way. Never assume older documentation, memory, or habit is
still accurate here — check the script's own `--help`/usage error if unsure.

Everything — SDK paths, MMALIB version, the shared-carveout physical base
(`0x900000000` for 8gb, `0x8a0000000` for 4gb) — is resolved by one CMake
module, `src/runtime/ti_dsp/cmake/boards.cmake`, `include()`d by all three
CMake projects. **`PSDK_INSTALL_PATH` has no default in `boards.cmake` —
it must always be set via env or `-D`, for either board.** `--board` only
selects which SDK-version subdirectory gets appended to it when deriving
`MCU_PLUS_SDK_PATH`/`MMALIB_PATH` (`mcu_plus_sdk_j722s_11_02_01_05` +
`mmalib_11_02_00_11` for beagley-ai vs `mcu_plus_sdk_j722s_11_00_00_12` +
`mmalib_11_02_00_06` for j722s-evm) — set those two env vars directly
instead if your SDK layout doesn't nest both board versions under one
`PSDK_INSTALL_PATH` root. An explicit `MCU_PLUS_SDK_PATH`/`MMALIB_PATH`
always wins over the derived path. On this dev machine both board versions
happen to live side by side under the same root:
`/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06` (j722s-evm) and
`/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_02_01_03` (beagley-ai,
despite the `am67a` dirname) — `find /opt/ti/am67a -maxdepth 1 -iname
'*11_02*'` if that's moved.

Build dirs are keyed on board+ddr (e.g. `build-c7x-beagley-ai-4gb`), so
switching `--ddr` never reuses a stale build; omitting both flags is
byte-identical to the old am67a-only behavior (`build-c7x`, no suffix).

**Both the runtime lib and firmware must be built with the same `--board`/
`--ddr`** — they're linked separately and a mismatch corrupts DMA silently
(no build error). See `relax-c7x:firmware` for the BeagleY-AI-specific
Linux-side DTB overlay requirement (separate from anything CMake controls).

## TIDL/MMALIB linkage: `--tidl`/`--mmalib` (firmware only)

`firmware/c7x/dsp/build.sh` additionally accepts:

| Flag | Values (default) | Effect |
|---|---|---|
| `--tidl` | `ON` (default), `OFF` | Links TIDL algo libs (`tidl_algo.lib`, `tidl_obj_algo.lib`, ...) into the firmware, enabling TIDL-backed kernels (e.g. `c7x_int8_max_pool_tidl`) |
| `--mmalib` | `OFF` (default), `ON` | Links MMALIB direct-integration libs (conv2d/matmul offload kernels). Forced `ON` whenever `--tidl ON` — TIDL's own algo lib has unresolved `MMALIB_CNN_*`/`MMALIB_LINALG_*` symbols at link time |

**Project convention: BeagleY-AI firmware must be built `--tidl OFF
--mmalib ON`.** beagley-ai runs the MMALIB-direct-offload path only, never
TIDL subgraph offload; am67a/j722s-evm is the only board that links TIDL.

```bash
cd src/runtime/ti_dsp/firmware/c7x/dsp
PSDK_INSTALL_PATH=/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_02_01_03 \
TI_SYSCONFIG_ROOT=/opt/ti/sysconfig-1.28 \
./build.sh --board beagley-ai --tidl OFF --mmalib ON
```

Build dir is keyed on `--tidl`/`--mmalib` too, exactly like `--board`/
`--ddr` (never silently reuses a stale build when you flip them): the
default (`--tidl ON`, which forces MMALIB ON) keeps the plain
`build[-board-ddr]/` name; any non-default `--tidl` value appends
`-tidl-<value>-mmalib-<effective-value>`, e.g.
`build-beagley-ai-4gb-tidl-OFF-mmalib-ON/`.

**Codegen must match what the firmware was actually built with.** A
no-TIDL firmware has no `c7x_int8_max_pool_tidl` symbol — `dyn_loader.c`
guards both the `extern` declaration and its `SYM()` export-table entry
behind `#ifdef USE_TIDL_RUNTIME`. The c_static_lib target's `tidl-kernels` attr
defaults to `true` regardless of what the firmware actually links (codegen
cannot detect the firmware's capability), so **every beagley-ai compile
must pass `-tidl-kernels=0` explicitly**, or `FuseQDQToTIDLMaxPool` emits a
call to a symbol that doesn't exist there — this fails only at DLOAD load
time on the board, not at compile time. Verify what a given binary actually
exports with:
```bash
/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS/bin/nm7x c7x_compute.out | grep max_pool
# --tidl OFF build: only c7x_int8_max_pool
# --tidl ON  build: both c7x_int8_max_pool and c7x_int8_max_pool_tidl
```
See `relax-c7x:firmware`'s Symbol Export Table section for the general
export mechanism, and `relax-c7x:mmalib-offload`/`relax-c7x:tidl-offload`
for the codegen side of `tidl-kernels`.

## Prerequisites

| Component | Path / Version |
|-----------|---------------|
| TI C7000 compiler | `TI_CGT_C7000_PATH` (ti-cgt-c7000 v5.0.1+ LTS; board-independent) |
| TI SysConfig | `TI_SYSCONFIG_ROOT` (default `/opt/ti/sysconfig-1.28`; **1.26.0 is too old** — the beagley-ai/11.02 SDK sysconfig run fails outright below 1.26.2) |
| TI MCU+ SDK | `MCU_PLUS_SDK_PATH` — derived from `PSDK_INSTALL_PATH` + per-board subdir: j722s-evm→`mcu_plus_sdk_j722s_11_00_00_12`, beagley-ai→`mcu_plus_sdk_j722s_11_02_01_05` |
| TI Processor SDK RTOS | `PSDK_INSTALL_PATH` — **no default, must always be exported** (see CLAUDE.md); on this dev machine: j722s-evm→`/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06`, beagley-ai→`/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_02_01_03` |
| MMALIB | `MMALIB_PATH` — derived from `PSDK_INSTALL_PATH` + per-board subdir: j722s-evm→`mmalib_11_02_00_06`, beagley-ai→`mmalib_11_02_00_11` |
| c7x-mma-tidl | `C7X_MMA_TIDL_PATH` (for TIDL offload; not board-keyed) |
| CMake | 3.16+ |
| Ninja | any |
| LLVM | 15+ |
| Python | 3.10+ with uv |
| aarch64-linux-gnu-g++ | For ARM client cross-compilation |

(Or skip installing any of the above on the host entirely — see
"Docker Build (BeagleY-AI, self-contained)" below.)

## Docker Build (BeagleY-AI, self-contained)

`docker/Dockerfile.ci_c7x` bakes the entire toolchain needed for a
beagley-ai build into one image, so none of the Prerequisites above need
to be installed on the host: TI CGT C7000 compiler, TI SysConfig 1.28,
PSDK RTOS 11.02.01.03 (MCU+ SDK + MMALIB for beagley-ai), LLVM, the
aarch64 cross-compiler, `uv`, and a venv with `build`/`setuptools` for
wheel packaging. It's scoped to beagley-ai only — no TIDL/`c7x-mma-tidl`
is included, since that board is always `--tidl OFF`.

### Build the image

```bash
docker build -t tvm.ci_c7x:latest \
  --build-arg http_proxy=http://wwwgate.ti.com:80 \
  --build-arg https_proxy=http://wwwgate.ti.com:80 \
  -f docker/Dockerfile.ci_c7x docker/
```

**Invoke `docker build` directly, not `docker/build.sh ci_c7x`** —
`docker/build.sh` has no `--build-arg` passthrough, and this Dockerfile
needs the proxy build-args to reach the internet at all from inside a
`docker build` on the TI network (`archive.ubuntu.com`,
`dr-download.ti.com`, and `astral.sh` are all otherwise unreachable; the
network also blocks Docker Hub outright, which is why the Dockerfile's
own `FROM` already points at
`artifactory.itg.ti.com/docker-public/ubuntu:24.04` instead).

### Run a build against the mounted repo

```bash
docker/bash.sh tvm.ci_c7x -- bash src/runtime/ti_dsp/build_all.sh --board beagley-ai
docker/bash.sh tvm.ci_c7x -- bash src/runtime/ti_dsp/build_all.sh --board beagley-ai --wheels
```

`docker/bash.sh` bind-mounts the live repo into the container at the same
path — it never copies or clones source into the image — and remaps the
container user to the host uid/gid via `docker/with_the_same_user`, so
build output lands owned by the actual host user, not root.

`build_all.sh` runs the full chain — TVM core, DSP runtime (`c7x_host`
+ `c7x --board beagley-ai`), firmware (`--tidl OFF --mmalib ON` per the
project convention), ARM client, and (with `--wheels`) the x86/arm64
packaging wheels — as one command. It builds TVM core into
`build-ci-c7x/`, not the plain `build/` name used elsewhere in this
skill, specifically so it never collides with a native (non-container)
build already sitting in the same bind-mounted repo. `--wheels` passes
`--board "$TVM_BOARD" --ddr ... --tidl OFF --mmalib ON` through to
*both* `build_wheel.sh --target x86` and `--target arm64` calls (fixed
2026-08-10 — it used to only forward `--tidl OFF`, so `build_wheel.sh`
silently defaulted `--board`/`--ddr`/`--mmalib` and looked for firmware/
runtime artifacts in the wrong `build-<board>-<ddr>[-tidl-*-mmalib-*]`
directory, failing with `Required file not found: .../build-tidl-OFF-mmalib-OFF/c7x_compute.out`
for a beagley-ai build). If you see that exact error, check whether
`build_all.sh`'s wheel step is passing `--board`/`--ddr`/`--mmalib`
through — `build_wheel.sh` itself has supported these flags directly
(via `board_build_dir.sh`) since before that fix; only the caller was
missing them.

Board/ddr/tidl/mmalib → build-dir-suffix naming is centralized in
`src/runtime/ti_dsp/board_build_dir.sh` (a `source`d shared function),
used by `build_runtime.sh`, both firmware `build.sh` scripts, and
`build_wheel.sh` — so all four always agree on where a given board's
artifacts live without independently re-deriving the naming scheme.
`build_all.sh` no longer sources it directly — since the wheel-step fix
above, it just forwards `--board`/`--ddr`/`--mmalib`/`--tidl` and lets
each downstream script (`build_wheel.sh` included) resolve its own
suffixed paths; the symlink-bridging it used to do for `build_wheel.sh`
(pre-dating that script's own `--board` support) is gone.

### After building: validate on hardware

`src/runtime/ti_dsp/validate_all.sh --board beagley-ai` is the natural
next step after `build_all.sh --board beagley-ai --wheels` — it installs
the just-built x86 wheel into a `.venv-ci-c7x` venv, deploys firmware +
ARM client, reboots and health-checks the board, runs the full quantized
MMALIB pytest suite against real `c7x_dload` hardware, and finally runs
the two standalone examples under `tests/ti-dsp-runtime/examples/`
(YOLO26 Python API, ResNet-18 C++ API) as an end-to-end smoke test of
the public offload APIs themselves — not just the internal test harness.
Like `build_all.sh`, it's meant to run via `docker/bash.sh` against the
same bind-mounted repo (see README.md's Quick Start), but nothing in it
is docker-specific. `--skip-deploy` reuses already-deployed firmware.
Uses `set -euo pipefail` throughout, so it aborts hard on the first
failing step — see `relax-c7x:testing`'s debugging reference if a step
here fails but the same test passes in your own dev venv.

When validating on a board from inside the container, pass `--net=host`
to `docker/bash.sh` — non-interactive `docker/bash.sh` invocations use
Docker's bridge network, and bridge NAT intermittently drops the `ssh`/
`scp` connections that firmware deploy and the DSP test harness make to
the target (`Connection reset by peer`, rc=255), even though the same
commands from the host shell are stable. Board-side SSH auth also lives
in the container's `~/.ssh`, which is the bind-mounted repo's `.ssh/`
(the container `HOME` is the repo path) — copy the board SSH `config` +
key + `known_hosts` there, and keep `.ssh/` gitignored.

### Docker-specific troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `403 dockerBlock` resolving `docker.io/library/ubuntu` | Docker Hub blocked on the TI network | Already handled — `FROM` points at the Artifactory mirror; apply the same swap for any other Dockerfile hitting this |
| `apt-get update` reports `Could not resolve 'archive.ubuntu.com'` inside a `docker build` | No proxy in the build container (ARGs weren't passed) | Add `--build-arg http_proxy=... --build-arg https_proxy=...` to the `docker build` invocation |
| `usermod: user '<name>' does not exist` from `with_the_same_user` inside `docker/bash.sh` | Base image ships a default `ubuntu` user at uid 1000, colliding with a host user also at uid 1000 | Already handled in the Dockerfile (`userdel` guarded on existence); apply the same fix if a future base image reintroduces this |
| `error: The interpreter at /usr is externally managed` from `uv pip install --system` inside a running container | Debian's python3.12 (a transitive dep of `llvm-18-dev`, not baked in deliberately) is PEP-668-protected | Use a venv, not `--system`/`--break-system-packages` — the image already bakes one at `/opt/venv-build` for wheel packaging |
| `Required file not found: .../tidl_model_import_relax.so` from `build_wheel.sh --target x86` | Default `--tidl ON` requires a TIDL `.so` this beagley-ai-scoped image doesn't have | Pass `--tidl OFF` (done automatically by `build_all.sh --wheels` for `--board beagley-ai`) |
| `ssh`/`scp` to the board fail intermittently with rc=255 / `Connection reset by peer` from inside `docker/bash.sh` | Non-interactive container runs use bridge networking; board traffic through bridge NAT is flaky | Re-run board validation with `docker/bash.sh --net=host ...` |

## 1. TVM Core (C++ + Python)

```bash
cd $TVM_HOME/build
ninja
cd ..
```

First-time setup (if building the c_static_lib runtime and `build/` doesn't exist):
```bash
mkdir build && cd build
cp ../cmake/config.cmake .
# Edit config.cmake: set USE_LLVM, BUILD_STATIC_RUNTIME=ON
cmake -G Ninja ..
ninja
```

## 2. Python Environment

```bash
export TVM_HOME=$(pwd)
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$TVM_HOME/build

# Install TVM Python package (editable)
uv pip install -e python/

# Install tvm-ffi submodule (needed for 0.23+)
cd 3rdparty/tvm-ffi && uv pip install . && cd ../..
```

Key config.cmake settings:
```cmake
set(BUILD_STATIC_RUNTIME ON)   # Required for c_static_lib runtime only, not for the TI DSP Runtime
set(USE_LLVM ON)               # LLVM codegen (for non-DSP targets)
```

## 3. DSP Runtime Library

```bash
cd src/runtime/ti_dsp

# C66x host emulation (g++, no external deps)
bash build_runtime.sh c66x_host

# C7x host emulation (x86 + TI Host Emu headers)
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS
bash build_runtime.sh c7x_host

# C7x cross-compilation (TI CGT + MCU+ SDK; am67a/j722s-evm). --board is
# required even for this default board -- see "Two boards" above.
export PSDK_INSTALL_PATH=/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06
bash build_runtime.sh c7x --board j722s-evm

# C7x cross-compilation for BeagleY-AI (4gb DDR; PSDK_INSTALL_PATH still
# required -- only the SDK-version subdir under it is picked automatically)
export PSDK_INSTALL_PATH=/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_02_01_03
bash build_runtime.sh c7x --board beagley-ai

# C66x cross-compilation
bash build_runtime.sh c66x
```

Output libraries (board+ddr suffix omitted for the default j722s-evm/8gb):
| Target | Output |
|--------|--------|
| c66x_host | `build-c66x-host/libtvm_dsp_runtime_host.a` |
| c7x_host | `build-c7x-host/libtvm_dsp_runtime_c7x_host.a` |
| c7x (j722s-evm) | `build-c7x/libtvm_dsp_runtime_c7x.a` |
| c7x (beagley-ai) | `build-c7x-beagley-ai-4gb/libtvm_dsp_runtime_c7x.a` |
| c66x | `build-c66x/libtvm_dsp_runtime_c66x.a` |

## 4. C7x Firmware

```bash
cd src/runtime/ti_dsp/firmware/c7x/dsp

# am67a/j722s-evm (default) — TIDL ON by default; PSDK_INSTALL_PATH and
# C7X_MMA_TIDL_PATH both required (see Prerequisites -- no default SDK root)
PSDK_INSTALL_PATH=/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06 \
C7X_MMA_TIDL_PATH=$HOME/ml/c7x-mma-tidl ./build.sh

# BeagleY-AI — project convention is --tidl OFF --mmalib ON (see
# "TIDL/MMALIB linkage" above); C7X_MMA_TIDL_PATH not needed since TIDL is off
PSDK_INSTALL_PATH=/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_02_01_03 \
TI_SYSCONFIG_ROOT=/opt/ti/sysconfig-1.28 \
./build.sh --board beagley-ai --tidl OFF --mmalib ON
```

Output: `build/c7x_compute.out` (j722s-evm) or
`build-beagley-ai-4gb-tidl-OFF-mmalib-ON/c7x_compute.out` (beagley-ai, per
the no-TIDL convention — plain `build-beagley-ai-4gb/c7x_compute.out` if
someone builds it with `--tidl ON` instead, which the codegen-side
`-tidl-kernels` attr must then also match).

Rebuild required when:
- DSP runtime symbols change (new exports)
- MMALIB wrapper code changes
- TIDL API changes
- Platform/memory layout changes
- Switching `--board`/`--ddr` (the runtime lib for that combo must exist first — build it via step 3 or this fails with `No rule to make target .../libtvm_dsp_runtime_c7x.a`)
- Switching `--tidl`/`--mmalib` (same staleness guarantee as `--board`/`--ddr` — see "TIDL/MMALIB linkage" above)

## 5. Deploy Firmware

```bash
cd src/runtime/ti_dsp/firmware/c7x
./deploy-c7x.sh --board j722s-evm dsp/build/c7x_compute.out                                     # am67a
./deploy-c7x.sh --board beagley-ai dsp/build-beagley-ai-4gb-tidl-OFF-mmalib-ON/c7x_compute.out   # beagley-ai (no-TIDL convention)
```

Cycle: copy firmware to `/lib/firmware/j722s-c71_0-fw` → **reboot the whole
board** (`ssh root@<host> reboot`) → wait ~30-60s for network, then a few
more seconds for SSH/services → verify with `c7x_compute status` (uptime
resets to seconds, jobs-completed resets to 0 — confirms the new image
actually loaded, not just the old one still running). A plain
`deploy-c7x.sh <firmware>` invocation only copies the file — it deliberately
does *not* try `--stop`/`--start` itself (the kernel returns EBUSY while
virtio vdev negotiation from the board's own autostart is still settling,
with no reliable way to know when that window has passed), so a real reboot
is required to activate a newly-copied image, not just `--restart`.
`--board beagley-ai` defaults the deploy host to `beagley-ai`;
`BOARD_HOSTNAME`/`TARGET` env vars always override.

## 6. ARM Client (c7x_compute CLI)

```bash
export TVM_HOME=/path/to/tvm        # required; cmake configure fails silently-ish without it (see gotcha)
cd src/runtime/ti_dsp/firmware/c7x/arm
./build.sh --board j722s-evm                                     # Cross-compile for aarch64 (am67a)
./build.sh --board beagley-ai                                    # Same, cosmetic board-check define only
BOARD_HOSTNAME=beagley-ai ./build.sh --board beagley-ai deploy   # SCP to that host
```

`arm/build.sh deploy` only reads the `BOARD_HOSTNAME` env var, not `--board`
directly — set it explicitly when deploying to a non-default host.

**Critical gotcha — `deploy` never builds anything.** `build.sh`'s
subcommand dispatch treats `deploy` as fully separate from the (no-
subcommand) build case: it only checks that `${BUILD_DIR}/c7x_compute`
already exists, then `scp`s it — it never invokes `cmake`/`make`, no matter
how stale that file is. Two ways this bites:

1. Running `./build.sh --board j722s-evm deploy` alone, expecting it to
   build-then-deploy in one shot, silently ships whatever binary is already
   sitting in `build/` — which may be days or weeks old.
2. Chaining a plain build immediately before a deploy
   (`./build.sh --board X && ./build.sh --board X deploy`) still ships a
   stale binary if the build step itself failed early (e.g. `TVM_HOME` not
   exported → `CMake Error: TVM_HOME is not set` → "Configuring
   incomplete, errors occurred!") and you don't notice in truncated output,
   because `deploy`'s existence check doesn't care whether the last build
   attempt actually succeeded.

**Always run the two steps separately and verify the build step's own
output** before trusting a subsequent deploy: confirm you see `Building CXX
object` lines and a final `Build complete:` block, not just "Deploying to
\<host\>...". If in doubt, compare `build/CMakeFiles/*/src/*.cpp.o` mtimes
against the corresponding `.cpp` source mtimes — a build that silently
no-op'd leaves the old object files untouched.

## 7. DLOAD Modules (Per-Model ELF)

Built automatically by pytest via `dsp_utils.build_dsp_dynmod()`. Manual build:

```bash
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS

# From a workspace with lib0.c + weights.bin:
cd /tmp/dsp_test_*/
mkdir build-c7x_dload && cd build-c7x_dload
cmake -DCMAKE_TOOLCHAIN_FILE=$TVM_HOME/src/runtime/ti_dsp/cmake/toolchain-j722s-c7x.cmake \
      -DDSP_MODE=c7x_dload ..
cmake --build .
# Output: lib0.out
```

Key link flags: `--dynamic=lib --relocatable --import=<symbols>`

## 8. TIDL Import Library

```bash
cd ~/ml/c7x-mma-tidl
bash build_j722s.sh          # Incremental
bash build_j722s.sh clean    # Clean + full rebuild
```

Output: `tidl_model_import_relax.so` (used by `tidl_import()` FFI)

## Build Order (Full Clean Rebuild)

```
1. Python env (uv pip install)
2. TVM core (ninja)
3. DSP runtime (build_runtime.sh c7x_host, c7x)
4. TIDL import library (build_j722s.sh)  [if using TIDL]
5. Firmware (build.sh)
6. Deploy firmware (deploy-c7x.sh) + reboot the board
7. ARM client build (arm/build.sh --board <X>, no subcommand — verify it
   actually recompiled, see step 6's gotcha) then deploy
   (arm/build.sh --board <X> deploy) as a separate invocation
8. Verify: c7x_compute status (check the reported version/uptime actually
   changed, confirming the new binaries are the ones running)
```

Steps 3-7 only needed when runtime/firmware code changes. For pure Python/codegen changes, only step 1 is needed.

## Environment Variables Summary

```bash
# Required for all C7x work
export TVM_HOME=/path/to/tvm
export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH
export TVM_LIBRARY_PATH=$TVM_HOME/build
export TI_CGT_C7000_PATH=/opt/ti/c7x/ti-cgt-c7000_5.0.1.LTS  # board-independent

# PSDK_INSTALL_PATH has NO default for either board -- always required via
# env/-D for build_runtime.sh c7x and firmware/c7x/dsp/build.sh. --board only
# picks the SDK-version subdir appended to it (see Prerequisites table).
export PSDK_INSTALL_PATH=/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_02_01_03  # beagley-ai
# export PSDK_INSTALL_PATH=/opt/ti/am67a/ti-processor-sdk-rtos-j722s-evm-11_00_00_06  # am67a/j722s-evm
export TI_SYSCONFIG_ROOT=/opt/ti/sysconfig-1.28  # beagley-ai/11.02 SDK needs >=1.26.2

export C7X_MMA_TIDL_PATH=$HOME/ml/c7x-mma-tidl   # not board-keyed; only needed when --tidl ON

# Deploy/test target host (not board-keyed — a hostname/IP, distinct from
# the --board SDK-selection flag); defaults to am67a if unset.
export BOARD_HOSTNAME=beagley-ai
```

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `cl7x: command not found` | TI_CGT_C7000_PATH not set or not in PATH | Export env var |
| `fatal error: c7x.h: No such file` | Wrong MCU_PLUS_SDK_PATH (or wrong `--board`) | Check path exists; confirm `--board` matches intended target |
| `undefined reference to mmalib_*` | Firmware not rebuilt with MMALIB | Rebuild firmware with MMALIB_PATH (or matching `--board`) |
| `ImportError: libtvm.so` | TVM_LIBRARY_PATH not set | `export TVM_LIBRARY_PATH=$TVM_HOME/build` |
| `ModuleNotFoundError: tvm` | PYTHONPATH missing | `export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH` |
| `tvm._ffi not found` | tvm-ffi not installed | `cd 3rdparty/tvm-ffi && uv pip install .` |
| Linker: `unresolved symbol` in DLOAD module | New runtime API not in firmware exports | Add to `dyn_loader.c`, rebuild firmware |
| `ninja: error: build.ninja not found` | Never ran cmake | `cd build && cmake -G Ninja ..` |
| `Error: MCU+ SDK for J722S version 11.02.01 requires at least version 1.26.2 of SysConfig` | Default/old `TI_SYSCONFIG_ROOT` (1.26.0) too old for the beagley-ai/11.02 SDK | `export TI_SYSCONFIG_ROOT=/opt/ti/sysconfig-1.28` (or newer); verified no MMU option-key drift vs 1.26.0 for the j722s-evm path |
| Firmware `dsp/CMakeLists.txt` fails with `No rule to make target .../build-c7x-beagley-ai-4gb/libtvm_dsp_runtime_c7x.a` | Runtime lib for that board+ddr combo not built yet | `build_runtime.sh c7x --board beagley-ai` first |
| `remoteproc: bad phdr da 0x... / Boot failed: -22` when starting the DSP on a fresh board | Linux-side DTB overlay not enabled — not a CMake/build issue at all | See `relax-c7x:firmware` DTB overlay section |
| cmake error mentioning `PSDK_INSTALL_PATH=` empty, or `MCU_PLUS_SDK_PATH not found` | `PSDK_INSTALL_PATH` was never exported — it has no default for either board | Export `PSDK_INSTALL_PATH` (or `MCU_PLUS_SDK_PATH`/`MMALIB_PATH` directly) before `build_runtime.sh c7x` or firmware `build.sh` |
| `USE_TIDL_RUNTIME=ON requires MMALIB ... forcing USE_TI_MMALIB=ON` (cmake STATUS, not an error) | Passed `--tidl ON --mmalib OFF`; TIDL's algo lib needs MMALIB symbols | Informational only — MMALIB gets force-enabled; pass `--mmalib ON` explicitly to avoid the surprise |
| DLOAD load fails on the board with an unresolved-symbol error for `c7x_int8_max_pool_tidl` (no compile-time error) | Firmware was built `--tidl OFF` but the model was compiled without `-tidl-kernels=0` | Add `-tidl-kernels=0` to the c_static_lib target string for any board whose firmware has TIDL off (beagley-ai, by convention) |
| ARM client behavior on the board doesn't match the source you just changed, `c7x_compute status` reports an unexpectedly old version, or a client/firmware protocol-version mismatch appears out of nowhere despite both being "just rebuilt" | `arm/build.sh <board> deploy` shipped a stale `build/c7x_compute` — `deploy` never builds, and an earlier bare build attempt may have failed silently on missing `TVM_HOME` | Run `./build.sh --board <X>` (no subcommand) on its own, confirm `Building CXX object`/`Build complete:` actually appeared, *then* run `./build.sh --board <X> deploy` separately — see the ARM Client section's gotcha |
| `Error: --board <j722s-evm\|beagley-ai> is required` from a command copied from older docs/memory/scrollback that omitted `--board` | `--board` is a required flag on `build_runtime.sh`, both firmware `build.sh` scripts, `arm/build.sh`, and `deploy-c7x.sh` — there is no bare-invocation default | Add `--board j722s-evm` (or `beagley-ai`) explicitly; don't trust a remembered command that lacks it |

## Related Skills

- `relax-c7x:firmware` — Firmware architecture and deploy details
- `relax-c7x:dsp-runtime` — Runtime library internals
- `relax-c7x:testing` — Running tests after build
- `relax-c7x:model-workflow` — End-to-end model compilation recipe
