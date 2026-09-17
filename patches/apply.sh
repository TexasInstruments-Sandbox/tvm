#!/bin/bash
# Apply local patches to vendored submodules that upstream doesn't carry.
#
# Idempotent: a patch already present in the target's working tree is
# skipped, so this is safe to call unconditionally on every build --
# regardless of whether the submodule was just freshly checked out or
# already patched from a previous run.
#
# On a persistent workspace (e.g. Jenkins with CLEAN_WORKSPACE=false, or a
# long-lived local checkout), the submodule's on-disk state can diverge from
# what this patch expects: an older/differently-named version of this same
# patch may still be applied as an *uncommitted* working-tree diff, or the
# submodule may be checked out on a stale commit altogether.  Recover by
# resyncing the submodule to the exact commit this repo's gitlink pins --
# the same step a fresh checkout already gets for free -- then retrying once
# before giving up.  This never discards uncommitted submodule edits itself;
# `git submodule update` will refuse loudly (and this script will abort) if
# such edits conflict with the resync, rather than silently wiping them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/.."
TVM_FFI_DIR="$REPO_DIR/3rdparty/tvm-ffi"
PATCH="$SCRIPT_DIR/tvm-ffi-c_static_lib-device.patch"

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
    echo "patches/apply.sh: stale submodule state detected -- resyncing to the pinned commit, and retrying" >&2
    git -C "$REPO_DIR" submodule update -- 3rdparty/tvm-ffi
    if git -C "$TVM_FFI_DIR" apply --check "$PATCH" 2>/dev/null; then
        git -C "$TVM_FFI_DIR" apply "$PATCH"
        echo "patches/apply.sh: applied $(basename "$PATCH") after resync"
    else
        echo "patches/apply.sh: $(basename "$PATCH") does not apply to 3rdparty/tvm-ffi -- submodule may have diverged upstream" >&2
        exit 1
    fi
fi
