#!/bin/bash
# Apply local patches to vendored submodules that upstream doesn't carry.
#
# Idempotent: a patch already present in the target's working tree is
# skipped, so this is safe to call unconditionally on every build --
# regardless of whether the submodule was just freshly checked out or
# already patched from a previous run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TVM_FFI_DIR="$SCRIPT_DIR/../3rdparty/tvm-ffi"
PATCH="$SCRIPT_DIR/tvm-ffi-c_static-device.patch"

if ! git -C "$TVM_FFI_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    echo "patches/apply.sh: $TVM_FFI_DIR is not checked out -- run 'git submodule update --init' first" >&2
    exit 1
fi

if git -C "$TVM_FFI_DIR" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "patches/apply.sh: $(basename "$PATCH") already applied, skipping"
elif git -C "$TVM_FFI_DIR" apply --check "$PATCH" 2>/dev/null; then
    git -C "$TVM_FFI_DIR" apply "$PATCH"
    echo "patches/apply.sh: applied $(basename "$PATCH")"
else
    echo "patches/apply.sh: $(basename "$PATCH") does not apply to 3rdparty/tvm-ffi -- submodule may have diverged upstream" >&2
    exit 1
fi
