# Indexer width fix: deployment and upgrade decision, 2026-09-11

Status: deployed and verified on all four Sparks, 2026-09-11 06:02 UTC.

## Decision

The user authorized autonomous repair and deployment, preferring a version upgrade
if practical. The repair uses a backport to the existing image
`vllm-glm52-b12x:dflash2-port2` (vLLM
`0.23.1rc1.dev190+gab6660699.d20260830`, Torch `2.11.0+cu130`, Triton `3.6.0`).
The build-date suffix is not the upstream revision date.

The newer locally available image
`radixark/vllm-glm53-flash:sm121-v12-dflash2` was inspected in a temporary container
without loading a model. It has DFlash2 and Torch `2.13.0+cu130`, Triton `3.7.1`,
vLLM `0.1.dev20051+g487ecf187`, and shared indexer sizing. However its
`SlidingWindowSpec.max_memory_usage_bytes` still asserts DCP size equals one.
The current service depends on a replicated DFlash sliding-window group alongside
the DCP-sharded target. Its KV coordination, block-table, attention and runner
patches need a port to that runtime, not just a package replacement. A dry-run
against the extracted relevant source applied only 5 of 24 patch files; 19 had
conflicts or files absent from the extracted subset. See
[compatibility audit](../results/indexer-block-table-investigation/upgrade-compatibility.json).
This audit is not a claim that upgrading is impossible. It establishes that a
drop-in image replacement would not preserve the validated serving configuration.

Upstream v0.27.0 first includes the general shared-width repair in
[PR #50302](https://github.com/vllm-project/vllm/pull/50302). It should inform the
eventual runtime port. The separate grouped-spec DCP fix #50823 is not an ancestor
of that tag; merging a fix before a release's publication date does not prove the
release contains it. This deployment does not claim an upstream upgrade.

## Implementation

- Variable decode copies the source width and zeroes destination slack.
- Uniform Triton loads use an explicit source-column bound and `other=0`.
- Complete padding rows are zeroed; insufficient workspace capacity is rejected.
- The output remains a contiguous view of the existing persistent workspace.
- A once-per-process diagnostic proves the live mixed-length branch was reached:
  `DSA indexer mixed-length block-table expansion: source=1408 buffer=1409`.

The final indexer SHA-256 is
`dbe5f8a0dab6e3f5992e8fe2d2263c203b488a787f64381a5e78cd73995491b8`.
The overlay, staged copy, deployment manifest and distributable patch are updated.
The original fork baseline is retained. Each node's old source is backed up as
`~/glm53big/indexer.pre-width-fix-20260911.py`. Before staging, every node matched
the previous committed manifest, and all four launcher hashes matched the local
launcher. The rollout changed the indexer only. The launcher's overlay-derived
slab salt consequently changed: previous NVMe cache entries are not reused under
the new namespace, so the first prefill of an old context is cold.

## Validation before rollout

- Historical investigation: both original source trees reproduced the exact Torch
  failure and the uniform kernel's neighboring-row/guard reads; 324 candidate CPU
  checks passed.
- Final-source regression: 158 CPU cases and 158 cases on Spark `spark-a218` using
  CUDA, with four successful CUDA graph replays on poisoned reusable buffers.
  CUDA contiguous source allocations are exact-sized, without the old repro's
  extra guard row.
- The same CUDA gate under `compute-sanitizer --tool memcheck --error-exitcode 99`
  passed with **ERROR SUMMARY: 0 errors**, including all four graph replays.
- 79 existing decode/block-table/deployment tests passed, plus a new pytest entry
  that executes the final-source regression gate.
- A pre-deployment count-to-100 request completed with all 100 numbers and 299
  generated tokens. Its token IDs and request are saved for comparison after boot.

Evidence is in [results/indexer-block-table-investigation](../results/indexer-block-table-investigation).

## Rollout and final checks

Rollout `indexer-width-fix-20260911` completed at **05:57:14 UTC**. The service
became healthy at 05:56:36 and generated `OK`. Standard rollout warmup exercised
long prefill, odd lengths, and a concurrent batch. Graph capture used 1.49 GiB;
the configured GPU KV capacity remains 198,551 tokens.

The service preserves TP4, DCP2, DFlash2 with seven draft tokens, max model length
180224, batch token budget 2048, max sequences 12, a 6,000,000,000-byte KV pool per
rank, and the 150,000,000,000-byte NVMe tier per rank. Adaptive speculation is off,
lossy verification and LSE folding are disabled, and candidate compaction remains on.

The post-boot count-to-100 control matched the pre-boot text and **all 299 output
token IDs exactly**. The short mixed test ran two concurrent count-to-300 streams
and injected raw-token requests of lengths 5, 3, and 7 while both were active.
Both streams returned every integer in order. All four workers logged the actual
mixed-length expansion at 05:57:58, with `source=1408 buffer=1409`. This establishes
live branch coverage in addition to the exact `[5,8,8]` source-level regression.

The long-context probe primed a 50,038-token chat prompt and reused its aligned
49,920-token prefix. Two concurrent count-to-300 streams completed correctly in
27.0 and 21.6 seconds, while three injected completions used total prompt lengths
49,925 / 49,923 / 49,927. Each injection generated 16 tokens and arrived while
both count streams were still decoding. The one-time branch marker already fired
in the short test, so it does not separately establish the exact scheduled lengths
of the long test; the latter establishes successful concurrent long-context use.

Both live probes ran under the four-node memory guard. The lowest available memory
was 3,512.6 MiB on the leader; swap-out and OOM counters did not increase on any
node, and no guard fired. The final snapshot verified all four containers running,
the exact repaired source SHA on each, no error/traceback lines in their new logs,
and HTTP 200 from `/health`. The existing node-local image IDs differ and are
recorded separately; this deployment did not rebuild or replace those images.

Evidence:

- [Four-node runtime and memory summary](../results/indexer-block-table-investigation/post-deploy-runtime.json)
- [Exact count100 comparison](../results/indexer-block-table-investigation/post-deploy-count100.json)
- [Live branch markers](../results/indexer-block-table-investigation/live-branch-coverage.json)
- [Short mixed probe](../results/indexer-block-table-investigation/live-mixed-short/summary.json)
- [50k mixed probe](../results/indexer-block-table-investigation/live-mixed-50k/summary.json)
- [Rollout console](../results/indexer-block-table-investigation/rollout-console.log)

These are focused regression and deployment gates, not a multi-hour soak or a
claim of bitwise parity for every workload. No artificial crash was attempted on
the old live service; the original failure was reproduced from its actual source.

## Reproduce and operate

```bash
.venv/bin/python bench/repro/indexer_block_table_width.py --baseline-only
.venv/bin/python bench/repro/check_indexer_block_table_fix.py \
  --source overlay/vllm/v1/attention/backends/mla/indexer.py --device cpu
python3 bench/repro/probe_indexer_mixed_decode.py --out /tmp/indexer-live-check
python3 bench/repro/probe_indexer_mixed_decode.py \
  --context-tokens 50000 --out /tmp/indexer-live-50k
```

The live probes generate real requests. Run them sequentially with the existing
`bench/spec_memory.py` guard when repeating the long test. To redeploy this state:

```bash
./rollout_dcp.sh indexer-width-fixed 180224 2048 6000000000 1
```

To roll back this repair, restore the indexer overlay, patch, staged copy, and
manifest from the parent of the repair commit, then run the same rollout command
with a new label. Do not modify a live bind-mounted file in place. The per-node
`indexer.pre-width-fix-20260911.py` backup is also available for recovery. Such a
rollback restores the known width defect, so it is a recovery option for a new
regression, not the recommended serving state.
