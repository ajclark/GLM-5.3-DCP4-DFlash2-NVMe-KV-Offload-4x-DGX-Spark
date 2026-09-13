# vLLM 0.29.0 runtime

The four-node deployment and regression results are recorded in the
[upgrade report](../../docs/VLLM-029-UPGRADE.md).

The release is pinned to upstream commit
`98dff2a81d747d1dba01a47f939f48c3526d4206` and the official CUDA 13.0 ARM64
image digest in `manifest.json`. This is a source port onto the release engine,
with Torch 2.13.0 and FlashInfer 0.6.18. The build verifies the upstream files
before replacing them, then verifies every installed overlay.

The serving configuration retains TP4/DCP2, DFlash2 K=7, a 180224-token window,
12 sequences, 2048 batch tokens, 6 GB KV per rank, and a 150 GB slab per rank.
The switchless RoCE ring uses `ag_rs`; FlashInfer all-reduce is disabled and
query replication is explicitly disabled. Worker-local offload preserves token-level
DCP interleaving; P/D page-transfer alignment is not applied to this connector.
Model weights are unchanged.

## Port scope

| Area | Release integration |
| --- | --- |
| Model, indexer, DFlash2 proposer, scheduler | Use native release implementations, including the shared-width indexer fix and native hybrid load-failure recovery. |
| Cache allocation | Use native `BLHNC` packing to keep mixed MLA/indexer pages and draft groups distinct. Skip the ordinary K/V dtype-ratio alignment for MLA, preserve 64-token draft pages, and normalize wrapped worker specs with the same group-aware rule as scheduler specs. |
| Draft cache | Replicate sliding-window groups in memory accounting, block geometry, offload configuration, MRv2 slot mapping, draft input preparation, and FlashAttention. Full-attention groups retain DCP sharding. |
| Fused indexer cache writes | Use the physical page stride for BLHNC. The native fused writer's contiguous-page assumption otherwise overwrites neighbouring MLA data. A byte-for-byte CUDA oracle checks both intended writes and untouched neighbouring bytes. |
| Sparse MLA on GB10 | Use native FlashInfer SM120 kernels. Filter and compact selected positions per DCP rank, preserve supported page sizes and physical strides, return base-2 LSE, and neutralize empty shards. |
| FlashInfer prefill | Repair flat addressing in the inline-scale cache gather and RoPE loads of FlashInfer 0.6.18's prefill kernels. Decode already honors physical strides. Build the corrected extension for SM121 under a distinct module name to exclude unpatched precompiled binaries. Both dependency source hashes and version are verified. |
| Durable NVMe cache | Port the **deployed** CRC-protected `stage/glm-dcp/multinode.py` to normalized `OffloadingConfig`, `LookupResult`, `OffloadingWorker`, and the new CUDA transfer API. Keep worker-local I/O, failure reporting, epoch checks, and cross-reboot persistence. |
| Cache compatibility | Dedicated `/var/tmp/kvcache-vllm029` root; content identity and a common source-manifest digest name the namespace. Normalize sparse FP8 aliases identically in scheduler and worker. The scheduler attaches to its exact namespace rather than the newest directory. |
| Deployment | Verify all images before downtime, retain original containers and mounts, load workers first, monitor memory, test real generation, and restore exact original containers on failure. |

Historical adaptive-cap, lossy-verification, MTP experiments and the June engine
sources remain in the repository for reproducing their recorded results. They
are not patches to install over this release. The new serving lane uses native
DFlash2 verification. The alternative old `direct` and `tiered` offload modes
are not exposed by this launcher's production slab lane.

The historical `deploy_kvtier.sh`, `deploy_slab*.sh`, concurrency-lane and
NCCL multi-communicator sweep scripts explicitly select the legacy runtime,
whose rollout logs and image they expect. Use the release commands below for
new deployments and durable-cache validation.

## Build and test

From the repository root:

```bash
runtime/vllm029/build.sh
PYTHONPATH=tests .venv/bin/python -m pytest tests/ -q
.venv/bin/python runtime/vllm029/rollout.py <unique-label>
.venv/bin/python runtime/vllm029/validate.py <unique-label>
```

`build.sh` does not stop serving. `rollout.py` checks native cache allocation and
scheduler/worker offload agreement at DCP1/2/4, saves the existing runtime
and a count100 control, then stops it and runs CUDA regressions before loading
the new engine. The gate also parses the actual GLM and DFlash2 checkpoint
configurations and checks the full cache admission calculation. Results go under `results/vllm029-upgrade/<label>/`. Full Docker
inventories are private ignored files because they can contain environment
configuration. The original containers remain stopped under backup names after
a successful upgrade. Restore with:

```bash
.venv/bin/python runtime/vllm029/rollout.py <label> --restore
```

The isolated runtime regression can also run in the built image:

```bash
docker run --rm --gpus all --ipc host \
  -v "$HOME/glm-vllm029-build:/regression:ro" \
  --entrypoint python3 spark-vllm:0.29.0-dcp1 \
  /regression/regression.py --cuda
```

Run CUDA checks with adequate free memory. The rollout runs them while model
serving is stopped. The gate covers the installed backend, padded KV pages,
empty DCP shards, the kernel's log base, CUDA graph replay, native GPU copy
handlers, and a disk reload into a different GPU block. CPU checks cover the
normalized configuration, metadata aggregation, namespaces, CRCs, epochs,
failed stores/loads, and manager recovery. A separate native DeepGEMM gate
checks prefill and paged indexer logits against an independent numeric oracle.
The attention oracle crosses the 64-token decode/prefill dispatch boundary and
checks 16, 32, and 64 query heads. It reproduces the unpatched prefill error with
a causal 172-token batch and verifies the corrected compiled extension.

`validate.py` tests automatic tool selection, tool-result continuation, reasoning,
and retrieval from a prompt longer than 90,112 tokens. It runs CUDA memcheck
while the service is stopped, restarts the exact new containers and requires more than 95% of that prefix to reload from NVMe,
then exercises concurrent mixed decoding at 50k context and verifies count100 again. A failure restores the retained original runtime.

After editing an overlay, refresh `manifest.json` with
`.venv/bin/python runtime/vllm029/sync_overlay.py`, rerun tests, and rebuild.
The source hashes are also the persisted-cache compatibility boundary.

## C1 coding sweep

The [2026-09-11/12 sweep report](../../results/vllm029-upgrade/c1-dflash-coding-20260911/REPORT.md)
contains the completed 16-to-7 coding sweep and restoration evidence.
The [lower-K follow-up](../../results/vllm029-upgrade/c1-dflash-lowk-20260912/REPORT.md)
completed K=6, 5 and 4 before the user requested a return to serving. Their
geometric mean decode rates were 37.10, 35.26 and 33.01 tok/s. The earlier K=7
control was 36.93 tok/s; K=6's roughly 0.5% lead does not establish an advantage.
The fresh K=7 benchmark and K=3, 2 and 1 were canceled. K=7 remains the default.

```bash
.venv/bin/python -u runtime/vllm029/coding_sweep.py <unique-label>
.venv/bin/python -u runtime/vllm029/coding_sweep.py <low-k-label> --ks 6,5,4,3,2,1
.venv/bin/python runtime/vllm029/summarize_coding_sweep.py results/vllm029-upgrade/<unique-label>
```

By default the sweep tests every speculative length from 16 down to the selected 7,
restarting the same release image for each setting. It keeps the other serving
arguments fixed and uses one client, greedy decoding with thinking off, three
coding tasks, one warmup per task and three measured repetitions. Each trial
uses a separate cache directory. Engine metrics confirm the effective draft
length; streamed token IDs provide timing and output hashes. Generated code
is checked against independent cases in a bounded subprocess.

The exact original K=7 containers are retained, restored and measured last.
A failure restores them automatically. If manual recovery is needed, use
`./restore_production.sh <unique-label>`. The report includes per-task results,
acceptance, output-length variation and correctness, with an equal-weight
geometric mean across task throughput averages.
`--ks` selects trial lengths in execution order; the original K=7 control is
always restored and measured last. The second command above tests below the
original sweep's lower boundary, allowing an interior throughput peak to be found.
