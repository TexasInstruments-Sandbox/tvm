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
# what this patch expects in two independent ways that both look the same
# from the outside ("does not apply ... may have diverged upstream"):
#   1. An older/differently-named version of this same patch is still
#      applied as an *uncommitted* working-tree diff, left over from a build
#      that ran before this patch's content last changed.
#   2. The submodule is checked out on a stale commit altogether (e.g. left
#      over from before this repo moved to the patch-based approach), with a
#      differently-named version of the same change baked directly into
#      that commit's history instead of as a diff.
# Recover by discarding any uncommitted diff and resyncing to the exact
# commit this repo's gitlink pins -- the same two steps a fresh checkout
# already gets for free -- then retrying once before giving up.
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
    echo "patches/apply.sh: stale submodule state detected -- discarding local diff, resyncing to the pinned commit, and retrying" >&2
    git -C "$TVM_FFI_DIR" checkout -- .
    git -C "$REPO_DIR" submodule update -- 3rdparty/tvm-ffi
    if git -C "$TVM_FFI_DIR" apply --check "$PATCH" 2>/dev/null; then
        git -C "$TVM_FFI_DIR" apply "$PATCH"
        echo "patches/apply.sh: applied $(basename "$PATCH") after resync"
    else
        echo "patches/apply.sh: $(basename "$PATCH") does not apply to 3rdparty/tvm-ffi -- submodule may have diverged upstream" >&2
        exit 1
    fi
fi
