# Comments posted to vLLM issues (2026-09-05)

- #47812: https://github.com/vllm-project/vllm/issues/47812#issuecomment-5548994135
- #53569: https://github.com/vllm-project/vllm/issues/53569#issuecomment-5548994219
- #52735: https://github.com/vllm-project/vllm/issues/52735#issuecomment-5548994295

Branch with the fix and tests:
https://github.com/vllm-project/vllm/compare/main...ajclark:vllm:fix/invalid-blocks-hybrid-kv-groups

Post each with (after editing to taste):

```
gh issue comment <number> -R vllm-project/vllm --body-file <file with just that comment>
```

---

## 1. On #47812 "[RFC]: Precise group-aware recovery for KV load failures on hybrid models"

(the RFC for exactly this crash; #45474 and #50687 are the bug reports; PRs #45497, #48216, #50388 and #50742 are open against it, so this is a data point and a reference, not another PR)

Another instance of Gap 1, with a fix we run in production, in case it helps whichever open PR lands: GLM-5.3 with a DFlash drafter on 4x DGX Spark under DCP=4, so two KV cache groups with different block sizes (a DCP-sharded MLA group at 256 tokens, a replicated sliding-window group at 64), behind an NVMe offloading tier that reports per-rank load failures. The first failure killed the engine at the single-group unpack.

The change: walk every group with its own manager `block_size`, take the earliest invalid token over all groups, and truncate `num_computed_tokens` to it rounded down to the scheduler block size (the LCM of the group block sizes); eviction stays per group. The LCM rounding covers the 16/8 example above: the 8-token block at 88-95 failing rewinds the request to 80, so the 16-token block spanning 80-95 is recomputed too. It does not do Gap 2's peer marking; rewinding to the earliest token across groups was enough under the recompute policy.

Branch with three CPU tests (same block size in both groups, either group failing; different block sizes, LCM alignment, earliest failure wins) that fail on `main` with the `ValueError` and pass with the change: https://github.com/vllm-project/vllm/compare/main...ajclark:vllm:fix/invalid-blocks-hybrid-kv-groups. Anyone is welcome to lift the different-block-size test. Setup, for context: https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark

---

## 2. On #53569 "[Bug][KV Offload] OffloadingConnector fs tier: multi-group MLA+DSA (DeepSeek-V4) TP=2 lookup fully misses across restart; single-group MHA hits 99.6%"

(a DGX Spark user, multi-node TP, fs tier, hits only for single-group models; someone has taken the issue, so this is corroboration and two concrete things to check)

Same hardware and a close setup (4x DGX Spark, TP4, GLM-5.3 with a DFlash drafter, two KV cache groups, fs-backed offloading, `PYTHONHASHSEED` pinned). Stores that work and lookups that never hit for the multi-group model had two causes for us:

1. The draft group is an eagle group to the connector, and until recently the store path held back each prefill step's trailing block for eagle groups but advanced the store index past it, so one block per step was never stored (339 of 390 per 100K-token prompt on disk). The sliding-window lookup needs a consecutive run and never gets one: queries grow, hits stay zero, store bytes look healthy. A request whose prefix is a GPU cache hit re-presents the whole prompt in one step and fills the holes, so in-process reloads can look fine while reloads after a restart never do. #46972 (merged 2026-07-07) fixed the store index; #52771, open, removes the fallback that treats every group as eagle when none is annotated, which is where DSpark and DFlash models sit.
2. The scheduler and the workers must derive the same cache directory key. Ours differed on `cache_dtype` (`fp8` on the scheduler, resolved `fp8_ds_mla` on workers), so the scheduler indexed a directory nobody wrote to. Worth diffing the two sides' directory names for an MLA model.

We ended up with a worker-executed multi-node tier (each rank writes its shard to its own NVMe, the scheduler indexes rank 0's slot headers) plus a fixed-size slab store: https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark (docs/NVME-DESIGN.md §10 is the store-hole analysis). Happy to compare notes.

---

## 3. On #52735 "[Bug]: OffloadingConnector stores but never serves when MTP/EAGLE speculative decoding is enabled"

(the open PR #52771 removes the "every group is an eagle group" fallback and fixes the trailing-block handling; this is independent confirmation from different hardware)

Independent confirmation from different hardware, in case it helps #52771 along: GLM-5.3 with a DFlash drafter on DGX Spark, 0.23 base, native OffloadingConnector with an fs tier. No group carried `is_eagle_group`, so both were treated as eagle and the store index skipped the held-back trailing block every prefill step: 339 of 390 target blocks per 100K-token prompt on disk, in both groups. A GPU-prefix-hit request refilled the holes in one step, which hid it in-process; a prefix stored once hit only its first 1,280 tokens after a restart. Using the eagle-adjusted count for the store index fixed it end to end (once-stored 100K prefix reloads in 3-8 s across evictions and restarts). Header dump and analysis: https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark/blob/main/docs/NVME-DESIGN.md (§10). +1 on dropping the all-groups fallback.
