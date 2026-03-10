#!/usr/bin/env python
"""Standalone diagnostic script for TIDL init on AM67A.

Usage:
    # Level 0: stub only (no TIDL calls, just load+run)
    python diag_tidl_levels.py 0

    # Level 1: call algNumAlloc only
    python diag_tidl_levels.py 1

    # Level 2: call algAlloc (the step that hangs)
    python diag_tidl_levels.py 2

    # Level 3: full init_tidl_subgraph
    python diag_tidl_levels.py 3

Requires:
    TI_CGT_C7000_PATH env var
    TIDL artifacts in ~/ml/edgeai-tidl-tools/artifacts_11_00/
    AM67A board running c7x_compute firmware
"""

import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

# Setup paths
TVM_HOME = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(TVM_HOME / "python"))
sys.path.insert(0, str(TVM_HOME / "tests" / "ti-dsp-runtime" / "dsp-cpp"))

import numpy as np
import tvm
from tvm import relax
from tvm.relax.backend.tidl import LowerTIDLToTIR, partition_for_tidl
from tvm.relax.frontend import nn

TIDL_ARTIFACTS = os.path.expanduser(
    os.environ.get("TIDL_ARTIFACTS_DIR", "~/ml/c7x-mma-tidl/artifacts/j722s"))

# ---------------------------------------------------------------------------
# Bridge templates for each diagnostic level
# ---------------------------------------------------------------------------

BRIDGE_LEVEL_0 = r"""
/* Level 0: pure stub, no TIDL calls at all */
#include <string.h>
#include <stdint.h>

extern int printf(const char *, ...);

void tidl_subgraph_0_process(void* inp0, void* out0) {
    printf("DIAG L0: stub bridge called, no TIDL\n");
    memset(out0, 0, 1*16*32*32*sizeof(float));
    printf("DIAG L0: done\n");
}
"""

BRIDGE_LEVEL_1 = r"""
/* Level 1: call algNumAlloc only — verifies IALG function table */
#include <string.h>
#include <stdint.h>

extern int printf(const char *, ...);
extern void* appUdmaGetObj(void);
extern void* appMemAlloc(int heap_id, uint32_t size, uint32_t align);
extern int32_t TVM_cacheWbInvRegion(void *addr, uint32_t size);

/* TIDL IALG types (minimal) */
typedef int (*algNumAlloc_fn)(void);

/* TIDL_VISION_FXNS is the IALG function table exported by firmware.
 * Layout: first field is IVISION_Fxns which starts with
 *   { size, ..., IALG_Fxns { algActivate, algAlloc, ... algNumAlloc, ... } }
 * On C7x TIDL, offsets (from neo-tvm tidl_api.c):
 *   ialg is at offset 0x10 in IVISION_Fxns
 *   algNumAlloc is at offset 0x28 in IALG_Fxns
 * We just access it via the TIDL_VISION_FXNS.ialg.algNumAlloc() path,
 * but since we link tidl_api.c, we can call init_tidl_subgraph.
 * For level 1, we just want algNumAlloc — access it via the raw symbol.
 */

/* Import the IALG function table from firmware */
extern char TIDL_VISION_FXNS[];

extern unsigned char _binary_tidl_net_start[];
extern unsigned int  _binary_tidl_net_size;
extern unsigned char _binary_tidl_io_start[];

void tidl_subgraph_0_process(void* inp0, void* out0) {
    printf("DIAG L1: === TIDL Function Table Diagnostic ===\n");
    printf("DIAG L1: TIDL_VISION_FXNS = 0x%llx\n",
           (uint64_t)(void*)TIDL_VISION_FXNS);
    printf("DIAG L1: net_start  = 0x%llx\n", (uint64_t)_binary_tidl_net_start);
    printf("DIAG L1: net_size   = %u\n", (unsigned)_binary_tidl_net_size);
    printf("DIAG L1: io_start   = 0x%llx\n", (uint64_t)_binary_tidl_io_start);

    void *udma = appUdmaGetObj();
    printf("DIAG L1: udma_obj   = 0x%llx\n", (uint64_t)udma);

    void *test = appMemAlloc(0, 1024, 128);
    printf("DIAG L1: appMemAlloc(0,1024,128) = 0x%llx\n", (uint64_t)test);

    int32_t cwb = TVM_cacheWbInvRegion(test, 1024);
    printf("DIAG L1: cacheWbInvRegion = %d\n", cwb);

    /* Read first 16 bytes of network for version sanity */
    if (_binary_tidl_net_size >= 16) {
        uint32_t *n32 = (uint32_t*)_binary_tidl_net_start;
        printf("DIAG L1: net[0..3] = 0x%08x 0x%08x 0x%08x 0x%08x\n",
               n32[0], n32[1], n32[2], n32[3]);
    }

    /* Peek at the IALG function table to see what's there.
     * IVISION_Fxns layout (from ivision.h):
     *   struct { IALG_Fxns ialg; ... }
     * IALG_Fxns layout (from ialg.h, 13 function pointers):
     *   0x00: implementationId (void*)
     *   0x08: algActivate
     *   0x10: algAlloc
     *   0x18: algControl
     *   0x20: algDeactivate
     *   0x28: algFree
     *   0x30: algInit
     *   0x38: algMoved
     *   0x40: algNumAlloc
     *   0x48: algNumRecs  -- (not present in all versions)
     */
    uint64_t *fxns = (uint64_t*)(void*)TIDL_VISION_FXNS;
    printf("DIAG L1: fxns[0] (implId)      = 0x%llx\n", fxns[0]);
    printf("DIAG L1: fxns[1] (algActivate) = 0x%llx\n", fxns[1]);
    printf("DIAG L1: fxns[2] (algAlloc)    = 0x%llx\n", fxns[2]);
    printf("DIAG L1: fxns[3] (algControl)  = 0x%llx\n", fxns[3]);
    printf("DIAG L1: fxns[4] (algDeact)    = 0x%llx\n", fxns[4]);
    printf("DIAG L1: fxns[5] (algFree)     = 0x%llx\n", fxns[5]);
    printf("DIAG L1: fxns[6] (algInit)     = 0x%llx\n", fxns[6]);
    printf("DIAG L1: fxns[7] (algMoved)    = 0x%llx\n", fxns[7]);
    printf("DIAG L1: fxns[8] (algNumAlloc) = 0x%llx\n", fxns[8]);

    /* Call algNumAlloc */
    algNumAlloc_fn numAlloc = (algNumAlloc_fn)(void*)fxns[8];
    if ((uint64_t)numAlloc > 0x1000) {
        printf("DIAG L1: calling algNumAlloc at 0x%llx ...\n",
               (uint64_t)numAlloc);
        int n = numAlloc();
        printf("DIAG L1: algNumAlloc() = %d\n", n);
    } else {
        printf("DIAG L1: algNumAlloc ptr looks invalid, skipping\n");
    }

    printf("DIAG L1: === done ===\n");
    memset(out0, 0, 1*16*32*32*sizeof(float));
}
"""

BRIDGE_LEVEL_1B = r"""
/* Level 1b: call algAlloc directly with NULL callbacks.
 * Uses proper TIDL_CreateParams struct (via TIDL headers) instead of
 * hardcoded offsets.  All callbacks are NULL to isolate the crash.
 * Uses dsp_trace_msg for breadcrumbs in remoteproc trace buffer.
 */
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include "itidl_ti.h"

extern int printf(const char *, ...);
extern void* appUdmaGetObj(void);
extern void* appMemAlloc(int heap_id, uint32_t size, uint32_t align);
extern void dsp_trace_msg(const char *msg);

/* TIDL_VISION_FXNS declared in itidl_ti.h as const IVISION_Fxns */
extern unsigned char _binary_tidl_net_start[];
extern unsigned int  _binary_tidl_net_size;
extern unsigned char _binary_tidl_io_start[];

void tidl_subgraph_0_process(void* inp0, void* out0) {
    dsp_trace_msg("L1b: start");
    printf("DIAG L1b: === Direct algAlloc test (NULL callbacks) ===\n");

    /* Print struct layout for verification */
    printf("DIAG L1b: sizeof(TIDL_CreateParams) = %u\n",
           (unsigned)sizeof(TIDL_CreateParams));
    printf("DIAG L1b: offsetof(net) = %u\n",
           (unsigned)offsetof(TIDL_CreateParams, net));
    printf("DIAG L1b: offsetof(udmaDrvObj) = %u\n",
           (unsigned)offsetof(TIDL_CreateParams, udmaDrvObj));
    printf("DIAG L1b: offsetof(cacheWriteBack) = %u\n",
           (unsigned)offsetof(TIDL_CreateParams, visionParams.cacheWriteBack));
    printf("DIAG L1b: offsetof(TIDLVprintf) = %u\n",
           (unsigned)offsetof(TIDL_CreateParams, TIDLVprintf));
    printf("DIAG L1b: offsetof(flowCtrl) = %u\n",
           (unsigned)offsetof(TIDL_CreateParams, flowCtrl));

    /* Step 1: algNumAlloc via function table */
    IALG_Fxns *ialg = (IALG_Fxns*)&TIDL_VISION_FXNS.ialg;
    dsp_trace_msg("L1b: calling algNumAlloc");
    int numMemRec = ialg->algNumAlloc();
    printf("DIAG L1b: algNumAlloc = %d\n", numMemRec);

    /* Step 2: Set up TIDL_CreateParams using proper struct fields */
    TIDL_CreateParams *cp = (TIDL_CreateParams*)appMemAlloc(0,
        sizeof(TIDL_CreateParams), 128);
    if (!cp) {
        printf("DIAG L1b: createParams alloc failed\n");
        memset(out0, 0, 1*16*32*32*sizeof(float));
        return;
    }

    /* Use TIDL_createParamsInit for proper defaults */
    TIDL_createParamsInit(cp);

    /* Copy network to writable memory (TIDL modifies it) */
    void *netCopy = appMemAlloc(0, _binary_tidl_net_size, 128);
    memcpy(netCopy, _binary_tidl_net_start, _binary_tidl_net_size);

    cp->net        = (sTIDL_Network_t*)netCopy;
    cp->udmaDrvObj = appUdmaGetObj();

    /* All callbacks NULL to isolate the crash */
    cp->visionParams.cacheWriteBack  = NULL;
    cp->TIDLVprintf                  = NULL;
    cp->pFxnLock                     = NULL;
    cp->pFxnUnLock                   = NULL;
    cp->TIDLWriteBinToFile           = NULL;
    cp->TIDLReadBinFromFile          = NULL;
    cp->TIDL_CustomLayerProcess      = NULL;
    cp->tracePtr                     = NULL;
    cp->traceLogLevel                = 0;
    cp->traceWriteLevel              = 0;

    printf("DIAG L1b: cp=%llx net=%llx netVer=0x%x udma=%llx\n",
           (uint64_t)cp, (uint64_t)netCopy,
           cp->net ? cp->net->netVersion : 0,
           (uint64_t)cp->udmaDrvObj);

    /* Step 3: Allocate memRec array */
    IALG_MemRec *memRec = (IALG_MemRec*)appMemAlloc(0,
        numMemRec * sizeof(IALG_MemRec), 128);
    memset(memRec, 0, numMemRec * sizeof(IALG_MemRec));

    /* Step 4: Call algAlloc */
    printf("DIAG L1b: calling algAlloc ...\n");
    dsp_trace_msg("L1b: before algAlloc");

    int status = ialg->algAlloc((IALG_Params*)cp, NULL, memRec);

    dsp_trace_msg("L1b: after algAlloc");
    printf("DIAG L1b: algAlloc returned %d\n", status);

    if (status == 0) {
        int i;
        for (i = 0; i < numMemRec && i < 5; i++) {
            printf("DIAG L1b: memRec[%d] size=%d align=%d space=%d\n",
                   i, memRec[i].size, memRec[i].alignment, memRec[i].space);
        }
    }

    dsp_trace_msg("L1b: done");
    printf("DIAG L1b: === done ===\n");
    memset(out0, 0, 1*16*32*32*sizeof(float));
}
"""

BRIDGE_LEVEL_2 = r"""
/* Level 2: call init_tidl_subgraph — uses dsp_trace_msg for breadcrumbs
 * visible in remoteproc trace buffer even if the DSP hangs.
 *
 * After running, read trace with:
 *   ssh root@am67a cat /sys/kernel/debug/remoteproc/remoteproc0/trace0
 * or:
 *   ./deploy-c7x.sh --trace
 */
#include <string.h>
#include <stdint.h>
#include "tidl_api.h"

extern int printf(const char *, ...);
extern void* appUdmaGetObj(void);
extern int32_t TVM_cacheWbInvRegion(void *addr, uint32_t size);
extern void dsp_trace_msg(const char *msg);

extern unsigned char _binary_tidl_net_start[];
extern unsigned int  _binary_tidl_net_size;
extern unsigned char _binary_tidl_io_start[];

void tidl_subgraph_0_process(void* inp0, void* out0) {
    dsp_trace_msg("DIAG L2: start");
    printf("DIAG L2: === algAlloc diagnostic ===\n");

    void *udma = appUdmaGetObj();
    printf("DIAG L2: udma=0x%llx net=0x%llx size=%u\n",
           (uint64_t)udma, (uint64_t)_binary_tidl_net_start,
           (unsigned)_binary_tidl_net_size);

    /* Flush cache on the network binary before TIDL reads it */
    TVM_cacheWbInvRegion((void*)_binary_tidl_net_start, _binary_tidl_net_size);
    TVM_cacheWbInvRegion((void*)_binary_tidl_io_start, 190000);

    dsp_trace_msg("DIAG L2: calling init_tidl_subgraph");
    printf("DIAG L2: calling init_tidl_subgraph...\n");

    void *inst = init_tidl_subgraph(
        _binary_tidl_net_start, _binary_tidl_net_size,
        _binary_tidl_io_start, udma, 1, 0);

    if (inst) {
        dsp_trace_msg("DIAG L2: SUCCESS");
        printf("DIAG L2: SUCCESS instance=0x%llx\n", (uint64_t)inst);
    } else {
        dsp_trace_msg("DIAG L2: FAILED");
        printf("DIAG L2: FAILED (NULL)\n");
    }

    dsp_trace_msg("DIAG L2: done");
    printf("DIAG L2: === done ===\n");
    memset(out0, 0, 1*16*32*32*sizeof(float));
}
"""

BRIDGE_LEVEL_3 = r"""
/* Level 3: init + algActivate + algDeactivate (tests DMA channel setup).
 * algActivate acquires DMA channels via DmaUtils.  If CLEC or UDMA is
 * misconfigured this will hang.  algDeactivate releases them.
 */
#include <string.h>
#include <stdint.h>
#include "tidl_api.h"
#include "dlpack/dlpack.h"

extern int printf(const char *, ...);
extern void* appUdmaGetObj(void);
extern int32_t TVM_cacheWbInvRegion(void *addr, uint32_t size);
extern void dsp_trace_msg(const char *msg);

extern unsigned char _binary_tidl_net_start[];
extern unsigned int  _binary_tidl_net_size;
extern unsigned char _binary_tidl_io_start[];

static void* tidl_inst = NULL;

void tidl_subgraph_0_process(void* inp0, void* out0) {
    if (tidl_inst == NULL) {
        dsp_trace_msg("L3: init_tidl_subgraph");
        printf("DIAG L3: calling init_tidl_subgraph...\n");
        tidl_inst = init_tidl_subgraph(
            _binary_tidl_net_start, _binary_tidl_net_size,
            _binary_tidl_io_start, appUdmaGetObj(), 1, 0);
        if (!tidl_inst) {
            printf("DIAG L3: init FAILED\n");
            memset(out0, 0, 1*16*32*32*sizeof(float));
            return;
        }
        printf("DIAG L3: init OK instance=0x%llx\n", (uint64_t)tidl_inst);
    }

    /* Test algActivate + algDeactivate (DMA channel acquire/release) */
    dsp_trace_msg("L3: before process_tidl_subgraph");
    printf("DIAG L3: calling process_tidl_subgraph...\n");

    /* Build DLTensor for input */
    DLTensor in_tensor;
    memset(&in_tensor, 0, sizeof(in_tensor));
    in_tensor.data = inp0;
    in_tensor.ndim = 4;
    int64_t in_shape[] = {1, 3, 32, 32};
    in_tensor.shape = in_shape;
    in_tensor.dtype.code = kDLFloat;
    in_tensor.dtype.bits = 32;
    in_tensor.dtype.lanes = 1;

    /* Build DLTensor for output */
    DLTensor out_tensor;
    memset(&out_tensor, 0, sizeof(out_tensor));
    out_tensor.data = out0;
    out_tensor.ndim = 4;
    int64_t out_shape[] = {1, 16, 32, 32};
    out_tensor.shape = out_shape;
    out_tensor.dtype.code = kDLFloat;
    out_tensor.dtype.bits = 32;
    out_tensor.dtype.lanes = 1;

    DLTensor* in_arr[] = { &in_tensor };
    DLTensor* out_arr[] = { &out_tensor };

    /* Flush input data from cache before TIDL reads it via DMA */
    TVM_cacheWbInvRegion(inp0, 1*3*32*32*sizeof(float));

    int32_t status = process_tidl_subgraph(tidl_inst, in_arr, out_arr);

    dsp_trace_msg("L3: after process_tidl_subgraph");
    printf("DIAG L3: process returned %d\n", status);

    /* Invalidate output so CPU sees DMA-written data */
    TVM_cacheWbInvRegion(out0, 1*16*32*32*sizeof(float));

    if (status != 0) {
        printf("DIAG L3: process FAILED, zeroing output\n");
        memset(out0, 0, 1*16*32*32*sizeof(float));
    } else {
        /* Print first few output values */
        float *fout = (float*)out0;
        printf("DIAG L3: out[0..3] = %f %f %f %f\n",
               fout[0], fout[1], fout[2], fout[3]);
    }
    printf("DIAG L3: === done ===\n");
}
"""

BRIDGES = {
    0: BRIDGE_LEVEL_0,
    1: BRIDGE_LEVEL_1,
    "1b": BRIDGE_LEVEL_1B,
    2: BRIDGE_LEVEL_2,
    3: BRIDGE_LEVEL_3,
}

BRIDGE_H = r"""
#ifdef __cplusplus
extern "C" {
#endif
void tidl_subgraph_0_process(void*, void*);
#ifdef __cplusplus
}
#endif
"""


def generate_model_code():
    """Generate TVM c_static code for conv-relu-softmax."""
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2D(3, 16, 3, 1, 1, bias=False)

        def main(self, x):
            x = self.conv1(x)
            x = nn.relu(x)
            x = nn.softmax(x, axis=1)
            return x

    mod, ps = M().export_tvm(
        spec={"main": {"x": nn.spec.Tensor((1, 3, 32, 32), "float32")}}
    )
    np.random.seed(42)
    params = [
        tvm.runtime.tensor(
            np.random.rand(*p.shape).astype("float32"), device=tvm.cpu()
        )
        for _, p in ps
    ]
    mod = relax.transform.BindParams(
        "main", dict(zip(mod["main"].params[1:], params))
    )(mod)
    lowered = LowerTIDLToTIR()(partition_for_tidl(mod))

    target = tvm.target.Target("c_static -mcpu=c7x")
    with tvm.transform.PassContext(opt_level=0):
        ex = relax.build(
            lowered, target=target, exec_mode="compiled", system_lib=True
        )

    gen_dir = Path(tempfile.mkdtemp(prefix="tidl_diag_"))
    tar_path = gen_dir / "model.tar"
    ex.export_library(str(tar_path), target=target)
    with tarfile.open(str(tar_path)) as tf:
        tf.extractall(str(gen_dir))
    tar_path.unlink()
    return gen_dir


def main():
    level_str = sys.argv[1] if len(sys.argv) > 1 else "1"
    level = int(level_str) if level_str.isdigit() else level_str
    print(f"=== TIDL Diagnostic Level {level} ===\n")

    if level not in BRIDGES:
        print(f"Unknown level {level}. Use 0, 1, or 2.")
        sys.exit(1)

    from dsp_utils import build_dsp_dynmod, run_dsp_dload

    # Check prerequisites
    if not os.environ.get("TI_CGT_C7000_PATH"):
        print("ERROR: TI_CGT_C7000_PATH not set")
        sys.exit(1)
    if not os.path.exists(os.path.join(TIDL_ARTIFACTS, "subgraph0_net.bin")):
        print(f"ERROR: TIDL artifacts not found at {TIDL_ARTIFACTS}")
        sys.exit(1)

    # 1. Generate model code
    print("Generating model code...")
    gen_dir = generate_model_code()
    print(f"  Generated: {gen_dir}")

    # 2. Write bridge for selected level
    bridge_c = gen_dir / "tidl_bridge.c"
    bridge_h = gen_dir / "tidl_bridge.h"
    bridge_c.write_text(BRIDGES[level])
    bridge_h.write_text(BRIDGE_H)

    # For level 0, we don't need TIDL at all
    use_tidl = level != 0

    # 3. Build
    print("Building c7x-dynmod...")
    build_dir = Path(tempfile.mkdtemp(prefix="tidl_diag_build_"))
    module_path = build_dsp_dynmod(
        generated_dir=gen_dir,
        build_dir=build_dir,
        weights_file=gen_dir / "weights.bin",
        tidl_bridge=str(bridge_c),
        use_tidl=use_tidl,
        tidl_artifacts_dir=TIDL_ARTIFACTS if use_tidl else None,
    )
    size_mb = module_path.stat().st_size / (1024 * 1024)
    print(f"  Module: {module_path} ({size_mb:.1f} MB)")

    # 4. Run
    print("Running on AM67A...")
    input_data = np.random.randn(1, 3, 32, 32).astype("float32")
    result, stdout = run_dsp_dload(
        module_path,
        gen_dir / "weights.bin",
        [input_data],
        embedded_weights=True,
    )

    # 5. Show output
    print("\n========== DSP OUTPUT ==========")
    print(stdout)
    print("================================")

    if result is not None:
        print(f"\nResult: shape={result.shape}, "
              f"min={result.min():.4f}, max={result.max():.4f}")
    else:
        print("\nNo result returned from DSP")

    # Cleanup
    if not os.environ.get("DSP_KEEP_TEMP"):
        shutil.rmtree(str(gen_dir), ignore_errors=True)
        shutil.rmtree(str(build_dir), ignore_errors=True)


if __name__ == "__main__":
    main()
