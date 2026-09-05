# Ideas to close (or beat) the DCP decode cost without an RDMA switch

Written 2026-09-05 06:30 by a Claude architecture sub-agent from the measured
picture in `docs/DESIGN.md` §8 and `results/dcp-profile-comparison.md`
(DCP=1 138.9 ms per cycle at the 2000 MHz lock; DCP=4 with candidate
compaction 155.2 ms; the residual is per-layer ring collectives at their
latency floor). The agent read the overlay code, the fork source and the
ring latency sweep; it did not run anything. Its idea #5 (DCP=2 on adjacent
ring pairs) was measured right after: predicted ~147 ms, measured 143.8 ms
(count100 54.5 tok/s, ~198k KV tokens at the 6 GB pool). Nothing else below
has been tried yet. Treat the gains as estimates with the arithmetic shown.

**Update 2026-09-05 15:30, idea 1 measured and rejected** (`bench/nccl_multicomm.py`,
`results/nccl-multicomm/RESULTS.md`): with raw NCCL communicators in the
serving image, two communicators cut the big collectives by ~10% (query
gather 121 -> 109 us, reduce-scatter 115 -> 105, TP all-reduce 89 -> 84),
four or eight are slower on the query gather (146, 200 us; not launch-bound),
forcing LL is worse alone (232 us) and only recovers to 144 us with four
chunks. The reversed-ring communicator [0,3,2,1] is legal on this topology
but brings nothing. The per-collective floor is the shared host-staged
proxy path, not the protocol, so chunking does not parallelise it. Worth at
most ~2 ms per cycle; not pursued. Idea 9 (pair/cross two-rank exchanges)
was not run and is unlikely to beat this given the same per-communicator
overhead. What remains is removing collectives (ideas 2, 3), glue fusion
(4), adaptive verify length (7), or DCP=2 / the switch.

## What the measured picture says that the docs slightly misread

1. **"Bytes are not the problem" is only half true.** The TP all-reduce is 31 us at 1 KiB and 84 us at 96 KB (8 x 6144 x bf16): 60% of it is size-dependent, at ~2.2 GB/s (the NCCL LL protocol with one CTA and one proxy thread; the "Forced LL" column's slope confirms 0.45 us/KiB moved). The big all-gathers/reduce-scatters run under the Simple protocol whose *per-hop* cost is ~25 us (3 hops = the ~75 us floor you see at either payload) versus LL's ~5 us/hop (31 us / 6 hops). So the lever is protocol and channel parallelism, not payload. The `ctas4` sweep proved nothing: CTAs are capped by channels, and channels were pinned at 1.
2. The true collective penalty of DCP4 is ~21 ms (34.3 vs 13.0), masked by a 9.4 ms attention *win* (2.9 vs 12.3 ms). A uniform 2x collective speed-up therefore helps DCP4 by ~17 ms but DCP1 by only ~6.5 ms.
3. The 4.6 ms in-graph idle is ~15 us per collective launch (260 extra collectives). Every collective removed is worth its duration plus ~15 us.
4. The query-replication estimate (8 ms) ignores that on a 273 GB/s device the replicated projection is streamed: q_b_proj for 64 heads is 33.5 MB int8 (123 us) or 16.8 MB int4 (62 us) versus 8.4 MB (31 us) today, plus a 64-head W_UK^T absorb; net ~4-5 ms, not 8, and it needs ~1.8 GB (q_b_proj int4 1.3 GB + W_UK^T all heads ~0.5 GB), which at a 6 GB pool costs the 307k window.
5. "GPU busy 97%" counts NCCL spin kernels; useful SM time is ~120 ms. The ~35 ms of communication is overlappable in principle, but single-sequence decode is a strict chain: only the K-side/indexer branch runs parallel to the q branch.

## Ranked ideas

| # | Idea | Mechanism | Gain per cycle (DCP4 base 155.2) | Memory / context | Size, layer | Risk | Validation |
|---|---|---|---|---|---|---|---|
| 1 | **Multi-communicator bidirectional LL ring** | Create N (2-4) extra 1-channel NCCL comms via pynccl with permuted comm-rank order (identity, reversed 0-3-2-1, both adjacency-legal), split every decode collective into N chunks issued on N side streams in one group; force LL for these comms only (set `NCCL_PROTO` in the env around their creation; prefill keeps the stock comm). N CTAs, N proxies, both NIC ports, both directions. | AR 96 KB: 31 + 53/N -> ~45-58 us, x167 = 4.3-6.7 ms. q AG under LL at N=4: 442 KB/8.8 GB/s + 15 -> ~60 us, x78 = 3.5 ms. RS similar 3.3 ms. Merges 0.5. **Total 11-14 ms -> ~142 ms**; DCP1 would gain only 4-7. | NCCL buffers ~8 MB/comm. No context change. | ~250 lines: `pynccl.py` (rank permutation at `ncclCommInitRank`, 30 lines), a `MultiRingCommunicator` in a new module, dispatch hook in `cuda_communicator.py`; graph-safe (pynccl uses the current stream). | NCCL may reorder the inter-node ring (check `NCCL_DEBUG=INFO` "Channel 00/01 : 0 3 2 1" before connecting; lazy connect makes a bad ring fail loudly not hang). LL throughput may not scale if the C2C/host buffer is the bottleneck. `NCCL_PROTO` may be cached process-wide (fallback: N=8 so chunks fall under the LL threshold). | Extend `nccl_latency.py`: permuted comms, AR 96 KB, AG 590 KB, RS 524 KB at N=1,2,4. Pass: AR <= 60 us, AG <= 75 us. Then boot: verify pass <= 146 ms, count100 byte-identical. |
| 2 | **Fold the LSE all-gather into the reduce-scatter** | Each rank sends `out_r * exp(lse_r - c)` and `exp(lse_r - c)` (as bf16 hi+lo pair, ~16-bit mantissa) in one RS buffer [T, H, 512+8]; receiver divides. `c` is a per-layer constant calibrated once from the gathered LSE (identical on all ranks); the bf16 exponent range gives +-80 in log space, far beyond LSE spread (~log 2048 + max logit). Empty rows are the natural identity (exp(-inf) = 0). | 78 x (22 + 15) = **2.9 ms** | none | ~150 lines Triton pack/unpack in `mla_attention.py` replacing `cp_lse_ag_out_rs`; calibration buffer. | Precision: 1-2 extra bf16 ulps; overflow if a head's LSE exceeds c + 80 (guard: log per-layer max at calibration). | Boot, compare count100 hash (must match), prose/code acceptance within noise, CPU test of merge vs reference to 1e-2 bf16. |
| 3 | **Fuse the indexer merge gather into the query gather (recompute layers) + overlap the q gather with the K-side on all layers** | On top-k layers pack [q \| candidates] into one AG (278 KB vs two collectives). On every layer issue the q AG on a second DCP comm/stream right after q_b_proj+rope so it hides behind kv_a_proj, norm, rope, cache write and the local indexer logits/top-k. | Fusion: 21 x (92 + 15 - 20) = 1.8 ms. Overlap: 58 x ~40 us + 20 x ~60 = 3.5 ms. Partially exclusive; **~3 ms combined** | none | ~120 lines across `sparse_attn_indexer.py` (`_merge_dcp_topk_global` split into pack/merge), `mla_attention.py`, model overlay forward order; second DCP `GroupCoordinator`. | Two comms over the same links contend; graph fork/join correctness; merge output must be ready before the filter. | Profile: big AG count 99 -> 78; q AG kernels overlapping GEMMs in the trace; pass <= previous minus 2 ms. |
| 4 | **DCP glue fusion** | One Triton kernel for filter + compact + empty-row flag; merge kernel reads the kernel's [1,H,T] LSE directly and writes rescaled output in the RS's [H,T,D] layout (drops 2 masked_fills, transpose, 2 movedim copies); Triton 4-way sorted-list merge replacing torch.topk+gather+where+copy; cache filter/compact across the 3 cached layers (static per-layer schedule). | "other" 5.6 -> ~3.5, top-k 1.3 -> ~0.4, ~600 fewer launches: **2-3 ms** | none | ~250 lines in `sparse_utils.py`, `flashmla_sparse.py`, `sparse_attn_indexer.py`. Pure overlay. | Low; cross-layer cache must key on the recompute schedule, not runtime state (CUDA graphs). | CPU tests (existing harness); trace: "other kernels" count 2543 -> <1500. |
| 5 | **DCP2 on adjacent pairs** (MEASURED: 143.8 ms, 54.5 tok/s) | Legal: 4 % 2 == 0; `all_ranks.reshape(-1, 2)` yields groups (0,1),(2,3) = links 06c4-365c and ddbf-a218. Position p lives on rank p%2 of its pair; both pairs hold a full copy; target block 128, drafter 64. Every DCP collective is a single-hop 2-rank exchange; the two pairs run concurrently on disjoint links. | Per layer ~q 45 + LSE 10 + RS 45 + merge/4 ~ 110 us vs 249: -10.8 ms; attention 1024 cands x 64 padded heads (32 real) ~ +2.9 ms. **~147 ms (range 141-148)**; with #1 applied to the 2-rank comms, ~140. | **Half the context**: 6 GB pool -> ~198k tokens; a 307k request needs 9.4 GB; realistic window 262k at 8 GB with boot-2 headroom. | Launcher only (`DCP_SIZE=2`, `MAXLEN`, `KVBYTES`); b12x head padding 32 -> 64 must be sized from the gathered count (already is). | Memory posture returns to rank-0 ~2 GB after boot; 2-rank NCCL latencies unmeasured. | Microbench 2-rank AG/RS at 131-262 KB (pass <= 40 us), then boot 262144/8e9: cycle <= 148, KV tokens ~2x DCP1. |
| 6 | **Query replication (int4 q_b_proj + int8 W_UK^T, all heads)** | Delete the 78 q gathers; each rank computes 64 absorbed heads locally. | 78 x (105 + 15 - 65 extra streaming) = **~4.3 ms** | ~1.8 GB/rank; at 6 GB pool the 307k window is lost (needs pool ~4.2 GB -> ~277k tokens); prefill +~6%. | Model overlay (`ReplicatedLinear` q_b_proj + loader), `mla_attention.py` pregather branch (skip AG), offline int4 requant script. ~200 lines + tooling. | RTN int4 requant quality; host headroom on rank 0. | Boot with pool 4.5e9; count100 identity is not expected (requant); compare acceptance and pass time. |
| 7 | **Adaptive verify length** | Drafter still drafts 7; verify only K' tokens chosen from a running acceptance estimate (prose ~2.7/8 -> K'=3-4). MoE cost is ~0.9 ms per expert slot (PLAN-40), so 4 tokens saves ~29 ms MoE + ~6 ms of smaller AR/AG/RS payloads. | Prose: 2.2-2.4 accepted / ~122 ms = **18.7 tok/s vs 17.1 (DCP1 19.6)**; code unchanged (keeps K=7). | none | Engine: per-request `num_speculative_tokens` in the proposer/runner (fork has `v1/spec_decode/dynamic`), graph sizes for 4-5 token batches. ~150 lines. | Acceptance estimator lag; extra CUDA graphs. | Bench prose/code: prose tok/s +8-12%, code unchanged. |
| 8 | **Zero-code NCCL probes (gate for #1)** | `NCCL_MAX_NCHANNELS=2` with `NCCL_DEBUG_SUBSYS=GRAPH,INIT` and lazy connect to see whether channel 1 is the reversed ring; `NCCL_PROTO=LL` and `NCCL_LL_BUFFSIZE`/`NCCL_BUFFSIZE` on the microbenchmark only. | 0-8 ms | none | launcher env only | Second ring may pair 0-2 (documented at 4 channels); never test on the serving stack. | `nccl_latency.py` sweep, AG/RS included (the existing table only has AR). |
| 9 | **Hierarchical pair/cross 2-rank comms (fallback to #1)** | AG4 = AG2(pair) then AG2(cross) over comms (0,1),(2,3),(1,2),(3,0), all adjacent; RS reversed. Saves one Simple hop per AG/RS. | (105-75) x 78 + (99-75) x 76 = **~4 ms**; no AR gain | none | ~150 lines, distributed layer | Same NCCL ordering risk as #1 | Same microbench |
| 10 | **Two-block DFlash (K=15) on high-acceptance content** | Chain a second drafter block on the first block's draft; verify 16. | 16 tokens -> ~102 distinct experts vs 57: MoE +45 ms -> ~205 ms cycle; code 6.7 -> ~10.5 accepted = +15-20%; prose a loss. | none | Drafter proposer + graph sizes; quality of chained drafting unknown. | Block-size-8 training; acceptance of block 2 conditioned on block 1. | Offline acceptance measurement first. |
| 11 | **Dual micro-batch overlap (DBO)** | Split the 8 tokens 4+4, overlap one half's collectives with the other's GEMMs. | Hides <= 34 ms but re-streams weights: dense +16, MoE +4 -> **net ~ -10 to -14 ms, parity at best**; a loss at DCP1. | none | Very large (ubatch + DCP + sparse indexer + graphs + spec decode). | High | Not recommended before #1-#4. |
| 12 | **fp8 reduce-scatter payload** | Quantize rescaled partials per token/head. | 524 -> 262 KB under Simple: ~1 ms; moot if #1 lands (LL scales with bytes but N comms absorb it). | none | small | precision | skip |

**Stacked outlook keeping DCP4 (396k tokens):** 155.2 - 12 (#1) - 2.9 (#2) - 3 (#3) - 2.5 (#4) = ~135 ms, below the current DCP1 (138.9) with the full context; an equally tuned DCP1 would sit near 132. #7 adds ~10% on prose independently.

## What the agent would NOT do, and why

- **PP4 instead of TP4.** Same one-copy KV benefit, zero per-layer collectives, but each rank streams 95 GB of weights *sequentially*: ~4x the GEMM time for batch-1 decode.
- **EP or attention-DP.** Dispatch needs all-to-all (impossible) or replicated inputs plus an all-reduce (no fewer collectives), and expert imbalance adds ~30% MoE time; DP attention loses the context gain.
- **Sequence parallelism / RS+AG around norms.** Same collective count at M=8; only adds launches.
- **Fusing RS into the o_proj all-reduce via weight replication.** Needs W_UV and o_proj for all 64 heads: ~8.5 GB/rank.
- **A custom verbs transport.** LL's ~5 us/hop is already near the physics; the win is parallelism across CTAs/proxies/ports, which NCCL gives via extra comms with far less risk and no compiled code.
- **Blind `NCCL_MAX_NCHANNELS>1`, Tree, PAT, LL128, GDR.** Documented to create 0-2 pairings or unsupported on Spark; only the explicit-comm route (#1) controls ring order.
- **Payload shrinking alone (fp8 q, further pre-gather).** The Simple-protocol floor, not bytes, sets the big gathers; #1 attacks the floor.

## Files the ideas touch

- fork `vllm/distributed/device_communicators/pynccl.py` and `cuda_communicator.py` (ideas 1, 9: rank-permuted comms, multi-comm dispatch)
- `overlay/vllm/model_executor/layers/attention/mla_attention.py` (ideas 2, 3, 6)
- `overlay/vllm/model_executor/layers/sparse_attn_indexer.py` (ideas 3, 4)
- `overlay/vllm/v1/attention/backends/mla/flashmla_sparse.py`, `sparse_utils.py` (idea 4)
- `~/spark-cluster-experiments/nccl_latency.py` (the microbenchmark gate for 1, 5, 8, 9; needs AG/RS and 2-rank/permuted-comm variants; never run against a serving stack)
