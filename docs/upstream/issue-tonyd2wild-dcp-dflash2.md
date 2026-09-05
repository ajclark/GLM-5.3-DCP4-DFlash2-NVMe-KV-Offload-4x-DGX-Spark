<!-- Posted 2026-09-05 as https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark/issues/4. Original draft header:
     Post with:
     gh issue create -R tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark \
       --title "DCP works with DFlash2: DCP4 + DFlash2 k=7 on the TP4 lane, one KV copy across the four Sparks (462k-token pool at 262K, 512K window boots) — patch set, numbers, lane offered" \
       --body-file docs/upstream/issue-tonyd2wild-dcp-dflash2.md
-->

Hey, following up on #2 with the next thing we built on your recipe, and first a genuine thank you: the quantizer, the DFlash2 port and the write-ups are what made any of this possible on four Sparks. Everything below runs on your `vllm-glm52-b12x:dflash2-port2` image, unchanged, with your `~/glm-triton` overlays still in place.

## DCP works with DFlash2

The README says, under the NVFP4 KV + DFlash2 lane:

> **DCP is impossible with DFlash2.** The drafter's `SlidingWindowSpec` layers trip `kv_cache_interface.py:528  assert decode_context_parallel_size == 1, "DCP not support sliding window."` — no image or dtype changes it. DCP requires MTP, which is why the reference 655K lane used MTP k=3. Pick DFlash2 (speed) or DCP (pool).

That assert is real, and you are right that no image or dtype changes it; it took a patch set. But it is a placement rule, not a hardware or model limit, and with the patches you no longer have to pick. We have been serving **TP4 + DCP4 + DFlash2 k=7** on the 743B Int4-Int8Mix since 2026-09-04.

The idea: let the two KV-cache groups be sharded differently. The sparse-MLA target group is split across the four ranks, so the cluster holds one copy of the KV cache instead of four. The drafter's sliding-window group stays replicated on every rank exactly as it is today. One helper decides per group, and the block tables, the hybrid coordinator, the input batch and the drafter's flash-attention path just ask it. The drafter never sees DCP, so acceptance is untouched. The rest of the patch set teaches the sparse indexer and the sparse attention backend to work on a sharded cache, which this June base lacks (upstream vLLM grew that later, on newer code).

Everything is here, with the design, the tests and the raw results:
https://github.com/ajclark/GLM-5.3-DCP4-DFlash2-NVMe-KV-Offload-4x-DGX-Spark

## Numbers

4x GB10, TP=4, DCP=4, DFlash2 k=7, fp8_ds_mla KV, the same switchless 200G ring as #2, stock 2418 MHz clocks. "production" is our DCP=1 lane of your recipe at a 120K window with an 8 GB/rank pool.

| lane | ctx | KV pool (tokens) | count100 tok/s | accepted/cycle | cycle ms |
|---|---|---|---|---|---|
| production, DCP1 + DFlash2 k=7 | 120K | 131,072 | 57.0 | 7.87 | 138 |
| DCP4 + DFlash2 k=7, 8 GB/rank | 131K | **524,288** (4.00x) | 45.4 | 7.87 | 173 |
| DCP4 + DFlash2 k=7, 7 GB/rank | 262K | **462,308** (1.76x, at 2.2x the window) | 45.4 | 7.87 | 173 |
| DCP4 + DFlash2 k=7, 8.2 GB/rank | 512K | 543,668 | | | |

- The 512K lane served a 499,797-token prompt (TTFT 1,934 s) and left rank 0 at about 300 MB free at the low point. It works, with no headroom. We run a 307,200 window with a 6 GB/rank pool (396,715 tokens) to leave room for an NVMe KV tier, which is a separate issue.
- Greedy count100 output is byte-identical between DCP1 and DCP4, and accepted tokens per cycle are the same 7.87, because the drafter is the same drafter.
- The decode cost is a constant +36 ms per verify cycle (about -22%): two ring collectives per layer across 78 layers, paid once per 8-token verify pass. We cannot use a2a on the ring, so DCP runs `ag_rs` only; a switched fabric with a2a should pay less.
- Prefill: 40K prompt 131 s, 92K 316 s, 250K 893 s (about 280 tok/s at long context; the sparse indexer dominates, not the attention kernel).

## What it takes to run

No image rebuild. Sixteen Python files are bind-mounted over the installed vLLM by a launcher derived from yours (`launch-glm53big-dcp.sh`, a lane beside the DFlash2 one), which preflights every file and refuses to start otherwise. `~/glm-triton` stays as it is; eight of your ten overlays are still mounted from there. Rollout, post-boot checks, a long-context probe under a memory guard and a one-command restore to the stock launcher are in the repo, because our first 512K boot found the way to a host swap storm and we wanted the next one to be boring.

## What we are not claiming

- No prose numbers against your C1-C6. Our decode figures are counting prompts plus our own prose and code prompts (production 21.3 / 44.8 tok/s, DCP4 16.1 / 34.5), which do not map to your columns.
- No quality eval beyond determinism checks and spot reads. DCP is the same attention over the same KV, but we ran no suite.
- Single cluster, switchless ring, single run per point, no error bars.
- The patches are for this June base (ab666069). A rebuild on current upstream would get sparse-MLA DCP and the replicated sliding-window groups from upstream itself.

## Your Flash recipes

Untested, but worth saying: GLM-5.3-Flash on your 2x and 4x recipes runs the same b12x sparse-MLA path with DFlash2 k=7, and the drafter's KV group is the same sliding-window shape, so the same placement rule should apply and the patch set should carry over with little change. The pool arithmetic is less dramatic there, since Flash KV is much smaller per token and you already have millions of tokens, but it would still turn four copies into one. If you or anyone wants to try, I am happy to help.

## Offer

If you would like this as a lane in the repo, I can send a PR shaped like your other lanes: a `dcp/` directory with the sixteen overlay files, the launcher, the check scripts and a bench section, or just a link from the README if you would rather keep it out of tree. Either way, thanks again for the great work, and for writing everything down so the rest of us could build on it.
