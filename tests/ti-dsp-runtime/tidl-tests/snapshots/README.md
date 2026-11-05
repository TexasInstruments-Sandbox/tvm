# Partition snapshots

`partition_snapshots.json` holds, per model, the ordered composites of every
TIDL subgraph (`partition_snapshot.py`). It was first captured while the
Python `_check_*` functions still decided offload, then diffed after those
checks moved into `TIDL_relaxAllowNode`. `test_tidl_partition_snapshot.py`
fails on any unexplained change. Regenerate with
`python diag_tidl_partition_snapshot.py --write [--models NAME ...]` and add
the reason here.

Every difference found by the move, classified as in the Track 1 plan (5.7):

| Entry | Change | Bucket | Reason |
|-------|--------|--------|--------|
| `layer/conv2d_transpose` | output shape 16x16 -> 32x32 | fixture | fixture now uses a 4x4 stride-2 pad-1 deconv with a constant weight: TIDL supports 2x2/3x3/4x4 kernels at stride 2 and upsamples by exactly the stride (the earlier 3x3/pad-1 shape, 31x31, is declined and was wrong on hardware); composites unchanged |
| `model/yolov8n` | one `tidl.nn.softmax` composite removed (the DFL head: axis 1 of a `(1, 16, 4, 2100)` tensor) | 1: old check accepted, TIDL rejects | TIDL constraint "Only softmax along width and height axis is supported"; the op now runs on TVM |

Fixtures changed so the gate judges a configuration TIDL can run (snapshot
composites unchanged): `scatter_elements` (axis 3; TIDL supports only width),
`scatter_nd` (indices' last dim = output_rank - 1), `instance_norm` (constant
gamma/beta: the parser took scale and bias only from constants, so variable ones
were silently dropped; they are now declined). The variable-weight conv/prelu fixtures needed no
change: `partition()` now runs `CanonicalizeBindings`, which inlines
`w = R.const(...)` into the call as `prepare()` already does.

No other entry (54 layer models, ResNet-18, MobileNetV2, YOLOv5n) changed.

## `model/yolov8n` depends on the torch version

The entry is captured with torch 2.10, the version the CI pipelines install
(`torch>=2.5,<2.11`). The YOLOv8n export differs between torch versions:
Ultralytics' C2f blocks call `chunk`, and with torch 2.15 the exported graph
reaches the Relax frontend as 16 `strided_slice` ops plus one `split`, while
with torch 2.10 it arrives as 9 `split` ops and no slices. TIDL offloads both
forms, so the partition is 5 subgraphs either way, but the composite lists
differ (230 composites with `strided_slice`, 222 with `split`). The test fails
under a torch that exports the other form; regenerate under the pinned
version. YOLOv8n on AM67A scores cosine similarity 0.997134 (c7x_host and
c7x_dload) with the torch 2.10 partition.

## Behaviour changes these snapshots do not show

Two changes alter what gets offloaded but leave every snapshot entry identical,
so they are recorded here rather than in the table above.

- **`take` with constant indices** was briefly declined because it returned
  wrong values (cosine 0.35-0.5). The cause was the parser copying int32
  indices through `NDArrtoFloat`, which converts them to float values that the
  gather then read as int32 bit patterns, so only index 0 was right. With the
  indices copied as integers it offloads again, subject to TIDL's own gather
  constraints; `layer/take` uses variable indices, so no entry moved either way.
- **`partition()` now runs `CanonicalizeBindings` on every module**, not just
  ones that skipped `prepare()`. It is what makes a `w = R.const(...)` binding
  inline so `TIDL_relaxFindConstants` sees the weight; for already-prepared
  models it is a no-op, which is why the real-model entries are unchanged.
