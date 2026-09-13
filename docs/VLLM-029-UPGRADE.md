# vLLM 0.29.0 upgrade, 2026-09-11

The four-Spark GLM-5.3 runtime has been ported from
`0.23.1rc1.dev190+gab6660699.d20260830` to
[vLLM 0.29.0](https://github.com/vllm-project/vllm/releases/tag/v0.29.0), the latest
stable release checked for this upgrade, published September 9. The release
commit is `98dff2a81d747d1dba01a47f939f48c3526d4206`.

## Reproducible runtime

The [manifest](../runtime/vllm029/manifest.json) pins the official CUDA 13 ARM64
image by digest and records pristine and shipped SHA-256 hashes for 15 vLLM
files and three FlashInfer files. The image uses Torch 2.13.0 and FlashInfer
0.6.18. Builds verify both dependency versions and source hashes; the repaired
SM121 FlashInfer extension is compiled into the image under a distinct module
name so an upstream precompiled binary cannot bypass the fix.

The selected deployment is `upgrade-029-r8`, image tag
`spark-vllm:0.29.0-dcp1`. The tag suffix denotes the patch revision; the serving
configuration is **TP4/DCP2**, with DFlash2 K=7, a 180224-token window, 12
sequences, 2048 batch tokens, 6 GB GPU KV and a 150 GB NVMe slab per rank.
Weights and rank order are unchanged. The RoCE ring retains NCCL Ring and
`ag_rs`. The new cache root is `/var/tmp/kvcache-vllm029`.

## Required changes

The [runtime scope](../runtime/vllm029/README.md#port-scope) describes the port.
The release supplies the model, indexer, scheduler, DFlash2 proposer and native
cache transfer implementation. The remaining integration handles replicated
draft cache groups alongside DCP-sharded target groups, sparse MLA attention
on GB10, and worker-local durable NVMe storage.

Real checkpoint and CUDA tests exposed several incompatibilities that a version
bump alone would miss:

- Native cache admission initially exceeded the existing 6 GB budget. MLA
  alignment must follow the model rather than the indexer backend, replicated
  draft groups must retain 64-token pages, and this worker-local connector must
  retain token-level DCP interleaving. Correct admission is 5,920,212,480 bytes.
- Replicated draft groups need consistent sizing, slot mapping and attention
  metadata throughout MRv2 and DFlash2; only sharded attention implementations
  need DCP log-sum-exp output. Sparse attention uses base-2 LSE and explicitly
  handles empty shards.
- The fused indexer writer assumed contiguous pages and corrupted neighbouring
  MLA data under native BLHNC packing. It now uses the physical page stride. A
  separate native writer provides a byte-for-byte CUDA oracle, including bytes
  that must remain untouched.
- FlashInfer 0.6.18's sparse SM120 prefill path also assumed contiguous inline
  cache entries. Decode batches of at most 64 tokens worked, while a 172-token
  tool prompt failed. The gather and RoPE addresses now respect physical page
  strides. Causal, nonzero-score attention tests cross the dispatch boundary and
  exercise 16, 32 and 64 query heads.
- Scheduler and worker sparse-FP8 aliases normalized differently, preventing
  durable-cache attachment. Both now derive the same content identity and exact
  namespace. CRCs, epochs, bounded transfer buffers and failed-transfer reporting
  are preserved from the deployed slab implementation.

Historical adaptive speculation, lossy verification and MTP experiments remain
available through `VLLM_RUNTIME=legacy`; they are not enabled in this deployment.

## Regression evidence

Evidence is under
[`results/vllm029-upgrade/upgrade-029-r8/`](../results/vllm029-upgrade/upgrade-029-r8/).
The [validation summary](../results/vllm029-upgrade/upgrade-029-r8/validation-summary.json)
records the completed gates, log hashes, memory observations and benchmark comparison.
Raw machine logs and private Docker inventories remain local and ignored.

- Sandbox: **1066 passed**, with two existing Triton interpreter numeric
  warnings, in 26.44 seconds (`PYTHONPATH=tests .venv/bin/python -m pytest tests/ -q`).
- Installed-image configuration checks: DCP1/2/4 native allocation, scheduler
  and worker block/hash agreement, and admission using the actual GLM and draft
  checkpoint configurations.
- Real CUDA: block tables, long slots, graph replay, sparse attention against
  a dense oracle, strided GPU/NVMe round trips, DeepGEMM indexer logits and the
  fused indexer writer. Both CUDA memcheck suites report **zero errors**.
- Live engine: exact text and all 200 token IDs match the original count100
  control; concurrent mixed decoding, automatic tool calls, tool-result
  continuation and reasoning pass.
- Cold retrieval correctly returns `QUARTZ-7294` from the start of a
  **100,736-token** prompt, exceeding the old 90,112-token draft-cache boundary.
  First-token latency is **180.83 seconds**, with zero prefix-cache hits.

- After stopping and restarting the exact new containers, the same request
  correctly returns the code with **100,608 external prefix hits (99.87%)**.
  First-token latency is **2.70 seconds** and total request time is 3.05 seconds.
  GPU prefix hits are zero, distinguishing durable reload from resident reuse.

- Concurrent mixed decoding at 50k context passes after restart. The final
  count100 response again matches the original text and all 200 token IDs.
  Both rollout and extended validation completed successfully, with zero OOM
  events or memory-guard alarms on all four nodes.

## Performance

Same prompts and serving configuration, two repetitions per engine, measured
immediately before upgrade and after restart validation:

| Decode workload | Original tok/s | 0.29.0 tok/s | Change |
| --- | ---: | ---: | ---: |
| count100 | 54.34 | 55.97 | +3.0% |
| prose | 18.86 | 20.31 | +7.7% |
| code | 46.24 | 42.08 | -9.0% |

Count100 benchmark output hashes match, with 299 generated tokens on both
engines. Prose and code can vary between runs. In this comparison the code
response changes from 189 to 249 tokens; its measured decode slowdown is real
for these responses, but is not an identical-output kernel comparison. Code
verification cycles remain approximately 145 ms on both engines. Two samples
per workload establish a basic regression baseline, not a broad speed claim.
See the [original benchmark](../results/vllm029-upgrade/upgrade-029-r8/baseline-bench/bench.json)
and [release benchmark](../results/vllm029-upgrade/upgrade-029-r8/extended/bench/bench.json).

## Operations and rollback

The root launcher and rollout default to the new release. Build and validate:

```bash
./rollout_dcp.sh <unique-label>
.venv/bin/python runtime/vllm029/validate.py <unique-label>
```

The rollout verifies every staged image before downtime and retains the exact
original containers. The validator restarts the same new containers to test
disk recovery. Both restore the originals on failure. To restore the engine
that preceded this deployment:

```bash
./restore_production.sh upgrade-029-r8
```

Original containers are retained as `vllm_glm53big_pre_upgrade-029-r8`, along
with their original image, mounts and cache. Rollback was exercised during the
port. One old-engine restoration encountered a stale Torch compilation cache;
its directory was preserved under a backup name and regenerated, after which
the original engine passed generation, tool and reasoning checks again.

Each node's standard `~/glm53big/launch-glm53big-dcp.sh` now dispatches to
`~/glm-vllm029-build/launch.sh`. The previous launcher is backed up as
`launch-glm53big-dcp.sh.pre-vllm029`; promotion verified hashes and dry runs
without replacing the validated serving containers. The separate
`launch-glm53big-dflash.sh` production launcher is unchanged.

The production lane exercised here is DCP2 with DFlash2 and slab offload.
DCP1/4 have configuration and CUDA kernel coverage, but were not subjected to
separate full-model deployments in this upgrade. Restart recovery refers to
engine containers; this run does not claim a new full-machine reboot test.
