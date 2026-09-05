<!-- Posted 2026-09-05 as https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/5. Original draft header:
     Post AFTER the DCP issue, and replace #4 below with its number.
     gh issue create -R tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark \
       --title "NVMe-durable KV cache for the TP4 lane: a 100K-token prefix reloads in 3-8 s instead of a 335 s cold prefill, across evictions and restarts (multi-node slab tier)" \
       --body-file docs/upstream/issue-tonyd2wild-nvme-kv-tier.md
-->

Companion to #4 (DCP4 + DFlash2). Once the KV cache is one copy and the pool is large, the thing that hurts is the cold prefill of a long context after it has been evicted, or after a restart: 335 s for 100K tokens on the 743B. This makes that a disk read.

## What it is

vLLM's own KV offloading assumes every rank shares one host (one shared mmap region, keyed by device index), which is not the case when TP spans four Sparks. So this is a multi-node tier: each rank writes its own KV shard to its own NVMe, into a fixed-size slab ring buffer (150 GB per node by default, so it can never grow past what you gave it), and the scheduler answers lookups from rank 0's slot headers. The headers are the index, so the cache survives an engine restart with no separate state to keep consistent. It sits under vLLM's native offloading connector; nothing about the model path changes.

Same repo as the DCP work, `docs/NVME-DESIGN.md`:
https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark

## Numbers

4x GB10, TP4 + DCP4 + DFlash2 k=7, 307,200 window, 6 GB/rank pool, GPU clocks at 2000 MHz.

| 100K-token prefix | TTFT | tokens served from disk |
|---|---|---|
| cold prefill | 335-342 s | 0 |
| same prefix after 3 x 100K other prompts pushed it out of the GPU pool | **3.4 s** | 99,328 / 99,980 |
| same prefix after a full engine restart | 5.4-8.3 s | 99,328 / 99,980 |

A 4.5 GB cap held exactly on all four nodes with eviction doing its job; at the 150 GB cap the four nodes' slab files stay byte-identical in size. The ~650 tokens not served are the trailing block, which is never offloaded for a draft-model group, plus the new question.

## Two things in the base worth knowing

- Block hashes are chained from a seed the engine draws from `os.urandom` unless `PYTHONHASHSEED` is set. Without pinning it, nothing matches after a restart. The launcher pins it in the container.
- Two bugs on this base bite anyone who uses vLLM's KV offloading with DFlash2 or MTP: the engine scheduler's invalid-block recovery assumes a single KV-cache group and crashes with the drafter's second group, and the offloading connector's store progress skipped one block per prefill step for "eagle" groups (with DFlash2, every group), so a prefix stored once could never be reloaded past its first hole. Both are fixed in the patch set; upstream has since fixed the second on newer code.

## Applies to the Flash recipes as-is

Untested there, but the tier does not care about the model: your 4x Flash README's cold first prefill (about 467 tok/s) would become the same 3-8 s reload for anything that has been seen once. It needs the pinned hash seed, the two fixes above, and a directory on each node's NVMe.

Same offer as #4: a lane PR in your format, or a link. And thanks again.
