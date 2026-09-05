# Decode Context Parallelism (DCP) for GLM-5.3 on the four-Spark vLLM fork

Status: design + patch set + CPU tests, 2026-09-04. Nothing has been deployed
to the Sparks. The target configuration is **TP4 + DCP4 + DFlash K=7**: the
production drafter is kept, its sliding-window KV group is replicated across
the DCP ranks while the target's cache is sharded (section 3.2). MTP was
evaluated and dropped by the operator's decision (section 6).

## 1. Goal and the number that matters

Today every TP rank holds a full copy of the GLM-5.3 KV cache: MLA is
multi-query, so the latent KV is identical on all four ranks. Per token, per
rank:

| cache | bytes/token/layer | layers | bytes/token |
|---|---:|---:|---:|
| MLA latent, `fp8_ds_mla` | 656 | 78 | 51,168 |
| DSA indexer k-cache (fp8 + scales) | 132 | 78 | 10,296 |
| total | | | ~61.5 kB |

The selected 8 GB pool therefore holds ~131k tokens, replicated 4x. DCP4 shards
the cache round-robin (global token position `p` lives on rank `p % 4`), so the
same 8 GB per rank holds one copy of a ~525k-token context. One copy of the KV
cache, four times the context budget.

The engine already accounts for this without any change: `FullAttentionSpec.
max_memory_usage_bytes` (which `MLAAttentionSpec` inherits) divides
`max_model_len` by the DCP size, block tables shrink by the same factor, and
the slot-mapping kernel writes `-1` for tokens another rank owns.

What fits, at the fixed 8 GB pool:

| max-model-len | tokens/rank for one full request | pool used | concurrency |
|---:|---:|---:|---:|
| 120,000 (today, DCP1) | 120,000 | 7.4 GB | 1.1x |
| 262,144 | 65,536 | 4.0 GB | 2.0x |
| 400,000 | 100,000 | 6.2 GB | 1.3x |
| ~520,000 | 130,000 | 8.0 GB | 1.0x (ceiling) |

## 2. Where the fork stands relative to upstream

The Spark image is vLLM `0.23.1rc1.dev190+gab6660699`; `ab666069` is upstream
main of 2026-06-19. Upstream added DCP for the sparse (DSA) MLA path *after*
that date:

| upstream | date | what it added | needed here |
|---|---|---|---|
| #46076 | 07-01 | indexer under DCP: per-rank local top-k, cross-rank merge, localized causal bounds | yes, re-implemented without the CuteDSL dependency |
| #46514 | 08-19 | FlashMLA sparse: filter global top-k to local slots, return LSE on the fp8 mixed-batch path, neutralize empty rows | yes, adapted to the sm12x Triton backend |
| #50382 | 08-21 | query replication and a2a as GLM defaults | no: a2a is impossible on this cluster, q-replication is a follow-up |
| #52377 | 08-25 | fix-ups after the MLADCPManager refactor | no: the fork predates that manager |
| #54908 | 09-02 | the fix for issue 54907 | **not applicable**, see below |

### Issue 54907 does not apply to this fork

The reported bug is in `vllm/models/deepseek_v32/common/kernels.py`, the fused
norm/RoPE Triton kernel of the NVIDIA DeepSeek-V3.2 path: it treated a negative
cache slot as an unconditional early return, so under DCP the non-owner ranks
never materialized the normalized/rotated K rows that dense prefill then
consumed.

That whole directory does not exist in this fork. Its K-side normalization and
RoPE are plain torch inside `MultiHeadLatentAttentionWrapper.forward`
(`kv_a_layernorm` then `rotary_emb`), computed unconditionally for every token
on every rank; only the *cache write* is slot-gated, in CUDA kernels that
already skip negative slots correctly. Applying the 54908 patch here would be a
no-op at best.

The fork's real gap is larger than 54907: the sparse indexer and the sparse
attention backend have **no DCP handling at all**, and
`MLAAttention.forward_impl` refuses fp8 KV under DCP outright. Everything below
the attention layer (scheduler, KV manager, block tables, slot mapping,
`dcp_local_seq_lens`, cache-write kernels, memory accounting) is already
DCP-aware at the base commit.

### Two hard constraints from this cluster

- The four Sparks are a **switchless RoCE ring**. Only ring collectives work.
  NCCL all-to-all needs p2p between non-adjacent ranks and dies with
  `ibv_modify_qp 110`. The DCP comm backend must be `ag_rs`; the patch rejects
  `a2a` at startup with that explanation.
- **No compiled changes.** Every change is a Python/Triton overlay file
  bind-mounted over `dist-packages`, exactly like the existing `glm-triton`
  kernel overlays.

## 3. How one layer works under DCP4

All tokens, prefill and decode, go through the same sparse "MQA" path in this
backend, which is why the mixed-batch path is the one that must return an LSE
for every token.

```
                       rank r (r = 0..3), one attention layer
   hidden ──► q (16 local heads) ──► indexer q/k/weights
                                        │
   kv_c_normed, k_pe ──► cache write     │ indexer k-cache write
   (slot -1 => skip)                     │ (slot -1 => skip)
                                        ▼
        (1) local logits over MY 1/4 of the context
        (2) local top-k (CUDA op, unchanged)  -> local positions l
        (3) pack (score, global = 4*l + r); all_gather over DCP   [16 kB/token]
        (4) top-k over the 4*2048 candidates -> 2048 GLOBAL positions
                                        │
   q all_gather over DCP (heads 16 -> 64)                        [73 kB/token]
                                        ▼
        (5) filter global positions: keep p with p%4==r, map to my slot
            (Triton); rows with nothing local -> (0, -inf)
        (6) sparse attention over my slots, all 64 heads -> out, lse
                                        ▼
        (7) all_gather(lse) ; rescale ; reduce_scatter(out over heads)
            -> my 16 heads of the exact softmax        [0.3 + 64 kB/token]
                                        ▼
   v_up_proj (16 heads) ─► o_proj (row-parallel all-reduce, unchanged)
```

**Why step 3-4 is exact, not approximate.** A token in the global top-2048 has
at most 2047 tokens ranking above it globally, hence at most 2047 on its own
rank, so it is always inside its owner's local top-2048. Merging the four local
lists therefore reproduces the global top-k. Every rank runs identical
arithmetic on the identical gathered tensor, so all ranks agree on the selected
set, which they must: a token selected by one rank but not by its owner would
silently drop out of the merge.

**Why step 7 is exact.** Standard log-sum-exp merge. Each rank's partial
softmax over its subset is rescaled by `exp(lse_r - lse_global)` and summed. A
rank owning none of the selected tokens contributes `(out = 0, lse = -inf)`,
the identity element. That case is forced explicitly, because the kernels may
leave NaN in those rows and `0 * NaN = NaN` would survive the merge.

### 3.1 Speculative verification under DCP

Any drafter puts K+1 query tokens per request through the sharded target
each cycle. The pieces of this patch that exist for that:

- The sparse backend keeps its spec-decode reorder threshold at `1 + K` under
  DCP (`supports_dcp_with_varlen`; causality comes from the indexer's
  indices, not kernel metadata), so verify rows stay on the decode path. On
  sm12x the indexer takes its *flatten* path for `next_n = K + 1 > 2`: every
  multi-token request is expanded into per-token rows, each with its own
  global causal bound, and only **then** is each bound sharded onto the rank.
  Sharding the request length first and subtracting offsets in local space
  gives too-short bounds (world 2, rank 1: `[3, 4, 5]` instead of `[4, 4, 5]`),
  which is covered by a test.
- The proposer forwards a **stale** `dcp_local_seq_lens` from the base batch
  while advancing `seq_lens`; the indexer builder never reads that field and
  localizes from `seq_lens` itself (a static test pins this).
- The static decode logits width carries `+ next_n` slack and one block of
  spec-decode overrun; the top-k merge runs before spec-decode unpacking so
  padded rows take part in the collective on every rank.

### 3.2 DFlash under DCP: the drafter group is replicated

The DFlash drafter is six dense sliding-window (2048) GQA layers with 8 KV
heads, sharded by heads across TP4 (2 KV / 8 Q heads per rank), and it forms
a **second KV-cache group**. Two facts decide its treatment:

- Sharding its KV across the DCP group is impossible in principle: vLLM's GQA
  DCP needs every rank of a DCP group to hold the same KV head
  (`tp > total_kv_heads`, `dcp <= tp // kv_heads`), and 8 heads at TP4 fails
  that. Upstream main still rejects sliding-window groups under DCP.
- Its cache is tiny (~6 kB/token/rank, ~37 MB per request) and *already*
  laid out as "every position, own head shard" on every rank.

So the drafter group **ignores DCP entirely**, and the target group shards.
One helper is the single source of truth for that split,
`kv_cache_interface.cp_world_size_for_kv_cache_spec(spec, cp)`: full
attention (including MLA, and uniform groups of it) returns the process CP
size, everything else (sliding window, chunked-local, Mamba) returns 1. Every
consumer asks it:

| consumer | sharded target group | replicated drafter group |
|---|---|---|
| `SingleTypeKVCacheManager.block_size` (coordinator passes per-group dcp) | 64 x 4 = 256 tokens | 64 |
| `resolve_kv_cache_block_sizes` | scheduler aligns on lcm = 256, hashes at gcd = 64 | |
| hybrid coordinator cache-hit lookup | manager block size, `dcp_world_size=4` | manager block size, `dcp_world_size=1` |
| `BlockTable` rows and `_compute_slot_mapping_kernel` | `cdiv(max_len, 256)` rows, `-1` for other ranks' positions | `cdiv(max_len, 64)` rows, every position mapped |
| `SlidingWindowSpec.max_memory_usage_bytes` | (n/a) | global need, not divided by dcp |
| `FlashAttentionMetadataBuilder` | (n/a, sparse MLA backend) | `dcp_world_size = 1`: no local context lengths, no gathered heads |
| `FlashAttentionImpl` | (n/a) | `dcp_world_size = 1`, no LSE, plain forward |
| `check_attention_cp_compatibility` | requires LSE | exempt |

Prefix caching stays coherent because both insertion
(`BlockPool.cache_full_blocks`) and lookup (`find_longest_cache_hit`) compose
four 64-token hashes into one 256-token hash through the same
`BlockHashListWithBlockSize`.

**The DFlash proposer itself needs no change.** The runner hands it the
metadata of its *own* KV group (`if self.drafter.kv_cache_gid ==
kv_cache_gid`), and its fused input kernel derives slot mappings from that
block table with the plain `pos // block_size` formula, which is exactly right
for a CP=1 table. The drafter's builder is created by the proposer with the
per-layer `SlidingWindowSpec`, so it takes the replicated path. The impl keys
its decision on `sliding_window is not None`; the builder cross-checks the
impls of its layers against the spec and raises if a sliding-window layer
were ever folded into a token-sharded group (which disabling the hybrid KV
manager would do), instead of silently attending over a quarter of its
window.

Cost: nothing new. The drafter's memory and its TP all-reduces are today's;
only the target's verify pass pays the DCP collectives.

Upstream validated the target-side pairing (#46514, "FlashMLA sparse: DCP on
the fp8_ds_mla mixed-batch path + MTP"); the replicated-group treatment
follows upstream's own rule for non-full-attention specs
(`dcp_world_size_for_kv_cache_spec`, used there for Mamba).

## 4. What this costs

Per decode step (single sequence), summing over 78 layers and using the
measured NCCL latencies from `nccl-latency-results.md`:

| collective | size | per layer | per step |
|---|---:|---:|---:|
| q all-gather | 73 kB | ~70 us | 5.5 ms |
| output reduce-scatter | 64 kB | ~64 us | 5.0 ms |
| LSE all-gather | 0.3 kB | ~31 us | 2.4 ms |
| indexer candidate all-gather | 16 kB | ~40 us | 3.1 ms |

So roughly **+16 ms per target step** (measured on the cluster: +36 ms per
DFlash cycle, see §7; the verify pass moves 8 tokens, not one). Under DFlash
that is per *cycle*:
the verify step (8 tokens) pays it once and the drafter adds nothing (its
group ignores DCP), so a K=7 cycle goes from the measured 145 ms to ~161 ms,
about 11%, in exchange for 2-4x the context. Against a plain ~70 ms decode
step the same 16 ms would be ~25%, which is one reason speculation stays in
the design. Attention compute per rank rises 4x (64
gathered heads instead of 16) but that term is well under a millisecond, and
KV read traffic per rank *falls* 4x because the non-owned candidates are
masked loads that never touch memory.

Memory, per rank, going from the 120k DCP1 launcher to 262k DCP4:

| item | 120k DCP1 | 262k DCP4 |
|---|---:|---:|
| KV pool | 8.0 GB | 8.0 GB |
| indexer gather workspace (`40 * len * 132 B`) | 0.63 GB | 1.38 GB |
| sparse bf16 prefill workspace (`5 * len * 576 * 2 B`) | 0.69 GB | **0** |

The sparse prefill workspace is dropped entirely under DCP: it is only used by
the separate prefill/decode path, which the patched builder refuses under DCP,
so it is provably unreachable. That is why more than doubling the context is
close to memory-neutral. `--max-num-batched-tokens` also drops from 8192 to
4096, because the gathered query and the fp32 attention accumulator are 4x
larger per token under DCP (8192 x 64 heads x 512 x 4 B is 1 GiB of accumulator
on its own).

## 5. The patch set

Thirteen files, each a complete replacement for the same path under
`/usr/local/lib/python3.12/dist-packages/vllm/`. `patches/*.patch` are diffs
against what the image actually runs (`baseline/`), which for
`flashmla_sparse.py` and `sparse_attn_indexer.py` is the `glm-triton` overlay
and for the others the pristine image file. 5.1-5.5 make the sparse-MLA
target DCP-aware; 5.6 lets the DFlash drafter's group stay replicated.

### 5.1 `v1/attention/backends/mla/sparse_utils.py` (+127 lines)

The existing index-conversion kernel gains three constexprs `DCP_SIZE /
DCP_RANK / DCP_INTERLEAVE`. For a global position `p`: owner is
`(p // I) % N`, local position is `(p // (N*I)) * I + p % I`, non-owned entries
become `-1`. With `N == 1` the arithmetic is bit-identical to the old kernel,
which a test asserts against the untouched baseline. New wrapper
`triton_filter_and_convert_dcp_index` for the DCP path. No compaction pass:
the Triton sparse kernels mask `-1` per cell (verified in
`sm12x_sparse_mla_attn.py`, `cand_valid = (slot >= 0) & ...`).

### 5.2 `v1/attention/backends/mla/flashmla_sparse.py` (+160 lines)

- `can_return_lse_for_decode = True`, so `check_attention_cp_compatibility`
  accepts the backend and `need_to_return_lse_for_decode` turns on under DCP.
- Builder: `supports_dcp_with_varlen` when the interleave is 1, so the
  spec-decode reorder threshold is not clamped back to 1 (causality here comes
  from the indexer's indices, not from kernel metadata). Under DCP it rejects
  `a2a`, non-fp8 caches and the separate prefill/decode path.
- The fp8 head-padding envelope and the tile-scheduler metadata are sized from
  the **gathered** head count (`num_heads * dcp_world_size`), since that is
  what the kernel sees. This is what makes upstream's "envelope guard"
  unnecessary rather than merely satisfied.
- `_forward_fp8_kv_mixed_batch` filters global positions to local slots,
  returns `lse` as `[T, H]` fp32, and forces all-`-1` rows to `(0, -inf)`.
- The bf16 prefill workspace is not reserved under DCP (see section 4), with
  an assert on the path that would have used it.

### 5.3 `model_executor/layers/attention/mla_attention.py` (+23 lines)

The DCP branch asserted `not fp8_attention`. Sparse impls with `fp8_ds_mla`
are now exempt: that cache carries per-128-element scales inline and the kernel
dequantizes internally, and the query stays bf16 (sparse impls do not set
`supports_quant_query_input`), so there is no scale the merge would have to
model. The existing q all-gather and `cp_lse_ag_out_rs` merge are reused
unchanged; a missing LSE now fails with a message naming the backend instead of
a `NoneType` error deep in a Triton kernel.

### 5.4 `model_executor/layers/sparse_attn_indexer.py` (+227 lines)

- New `_merge_dcp_topk_global`: a Triton pack kernel pairs each local
  candidate with its score and its global position, `all_gather` over the DCP
  group, `torch.topk` over the `N*K` candidates, scatter back as global ids
  with `-1` padding. Upstream uses a CuteDSL radix select here; that dependency
  is not on the Spark image and a 8192-wide `torch.topk` is cheap.
- The custom op takes three trailing ints (`dcp_rank`, `dcp_world_size`,
  `cp_kv_cache_interleave_size`), filled in by `SparseAttnIndexer.__init__`.
  The merge runs after each prefill chunk's top-k (with
  `row_starts = chunk.cu_seqlen_ks`) and after the decode top-k, before
  spec-decode unpacking.
- A prefill chunk that is globally non-empty can be locally empty (a short
  prompt's tokens all land on low ranks). Those ranks skip the gather, logits
  and top-k, fill `-1`, and still enter the merge so the collective count
  matches on every rank.

### 5.5 `v1/attention/backends/mla/indexer.py` (+199 lines)

- Builder learns the DCP scalars; refuses interleave > 1 and KV compression
  under DCP (both unvalidated upstream too).
- Prefill chunk metadata: the gather sizes and the per-query row starts become
  this rank's local ones, while `token_to_seq` and the chunk-emptiness decision
  stay global so every rank builds an identical chunk list. The per-query
  causal bound is localized inside the Triton metadata kernel with the same
  formula as `get_dcp_local_seq_lens`.
- Decode: sequence lengths are localized **after** the per-token expansion.
  Localizing first and then subtracting decode offsets happens in local space
  and yields too-short bounds (world 2, rank 1, global per-token bounds
  `[8, 9, 10]` become `[3, 4, 5]` instead of `[4, 4, 5]`), so the first decode
  token would score against too short a KV range. The localizer is shape
  agnostic so the 2-D MTP bounds work too.
- The decode logits width and the top-k early-exit bound shrink by the DCP
  size. Without this, raising `max_model_len` to 262k would make every decode
  step scan a 262k-wide logits row per layer even though each rank stores at
  most 66k tokens. The width stays a static, config-derived int, as CUDA graphs
  require.

Added after boot 2: `get_max_prefill_buffer_size` divides max_model_len by
the DCP size (each rank's K gather is over its local shard), and the builder
hands `split_indexer_prefill_chunks` the localized upper bound
`cdiv(seq_len, dcp)` for both its workspace (N) and logits (M x N) budgets.
Unchanged at DCP 1. Tests: `test_dcp_prefill_workspace.py` (every chunk's
local N fits the locked workspace over random batches; slices per 2048-token
chunk at 500k drop 16 -> 4). Also `do_not_specialize` on the chunk-metadata
kernel's query-slice bounds (§7).

### 5.6 The replicated drafter group (+297 lines over eight files)

| file | change |
|---|---|
| `v1/kv_cache_interface.py` | `cp_world_size_for_kv_cache_spec`, the one decision; `SlidingWindowSpec` no longer asserts `dcp == 1` (its need is global by construction) |
| `v1/core/kv_cache_utils.py` | hybrid block-size resolution scales only sharded groups (scheduler lcm 256, hash gcd 64); the DSA sliding-window paging helper drops its assert |
| `v1/core/kv_cache_coordinator.py` | managers get their group's dcp size; the hybrid coordinator drops the blanket assert, allows full-attention + sliding-window groups, and does all hit-length arithmetic on manager block sizes with per-group dcp |
| `v1/worker/block_table.py` | `BlockTable(cp_world_size=...)`: `1` forces CP-free geometry and slot mapping; `MultiGroupBlockTable(cp_world_sizes=...)` |
| `v1/worker/gpu_input_batch.py` | forwards `cp_world_sizes` |
| `v1/worker/gpu_model_runner.py` | per-group CP size and row count when (re)building the input batch |
| `v1/attention/backends/flash_attn.py` | builder takes the replicated path for such groups and cross-checks its layers' impls; impl with a sliding window forces `dcp_world_size = 1` before the DCP combine is armed |
| `v1/worker/cp_utils.py` | the CP compatibility check exempts replicated impls |

## 6. Speculative decoding: DFlash is the design; MTP was evaluated and dropped

The production launcher runs DFlash K=7, so a DCP configuration without
speculation would trade the metric the cluster is tuned for (C1 coding decode
speed) for context. Section 3.2 shows the drafter can keep its layout under
DCP at no cost, so the design ships DFlash (`launch-glm53big-dcp.sh`, default
`dflash`); `SPEC_MODE=none` exists only to bisect. MTP under DCP was worked
through first (it needs no KV-group changes), measured on paper to land
within noise of DFlash, and then dropped by the operator in favour of the
higher-ceiling drafter. The target-side verify machinery is the same for
both, so nothing MTP-specific remains in the code; the `mtp` lane is refused
by the launcher with a message.

### 6.1 What the data says about MTP vs DFlash on GLM-5.3

Inco's published evaluation of the exact DFlash2 draft in use (from
`PLAN-40-TOKS.md`), accepted length per cycle, 7 draft tokens:

| task | native MTP | DFlash2 |
|---|---:|---:|
| HumanEval | 4.85 | 5.48 |
| MBPP | 4.34 | 4.95 |
| GSM8K | 5.12 | 5.94 |
| MT-Bench | 3.81 | 4.19 |

So DFlash accepts ~10-13% more per cycle, and on this cluster the DFlash draft
measured 4.01 pooled on the Pi coding suite (below its own spec, for reasons
that plan investigates). But MTP drafts with one extra MLA+indexer layer run K
times sequentially, while DFlash drafts all K in one pass of six dense layers,
and MTP's verify batch is smaller (K=4 caps a cycle at 5 tokens; DFlash K=7 at
8). For a batch-1 MoE decode, verify cost grows with verified tokens (each
token activates its own experts, and the step is expert-weight-bandwidth
bound), so a shorter MTP cycle offsets its lower acceptance. Back of the
envelope under DCP4, both including the ~16 ms/step DCP collective cost:

| | cycle | accepted/cycle | tok/s |
|---|---:|---:|---:|
| DFlash K=7 under DCP | ~155 ms (139 today + 16) | ~4.0 (measured today) | ~26 |
| MTP K=4 under DCP | ~130 ms (verify 5 + 4 draft passes + 16) | ~3.6 (vendor -10%, cap 5) | ~28 |

Treat these as +/-30%. They say the two are close, which matches the operator's
recollection, and that the decision has to come from the cluster. They also
say MTP is not a fallback here: it is the cheaper experiment and may win.

Three measured facts from `PLAN-40-TOKS.md` that frame the comparison:

- **The DFlash drafter is cheap; verifying is what costs.** The K=7 cycle floor
  is MoE 64.3 ms (verify, grows ~0.9 ms per expert slot, i.e. with verified
  tokens), dense int8 20.8, all-reduce 13.1 (latency-bound, fixed), sparse MLA
  12.5, **draft 10.9** (one block-parallel pass over 4.3 GB of bf16 weights,
  K-independent), and ~20 ms of CUTLASS/gaps/rejection. The draft is ~7.5% of
  the cycle. (The MoE audit refuted the earlier "seven sequential draft
  passes" model; the "35 ms no-spec step" in `coding-benchmark-results.md`
  came from that refuted model and should not be quoted. A plain step derived
  from the K-independent terms plus one token's MoE and attention is ~65-75
  ms.)
- **DCP's cost is fixed per cycle, so speculation amortizes it.** ~17 ms is
  about 12% of a 145 ms speculative cycle but about 25% of a plain step, which
  is a second reason MTP belongs in the design rather than after it.
- **DFlash's K is not a knob; MTP's is.** The drafter's `block_size: 8` is
  trained in, so DFlash runs at K=7 or degrades; MTP's K can be swept (6.2).

### 6.2 If MTP is ever wanted back

Nothing in the patch prevents it: the MTP head shares the target's KV specs,
so it stays one group, and the verify-side pieces in 3.1 serve it unchanged.
Re-adding the launcher lane is a one-line change; K would be worth sweeping
in {4, 6, 8} since, unlike DFlash's trained-in block size, MTP's K is free.

### 6.3 DFlash under DCP: what was built

Operator's decision (2026-09-04): keep DFlash's decode ceiling and take the
drafter's memory as the price. The price is zero: the drafter's KV is
~6 kB/token/rank (6 layers x 2 KV heads/rank x 128 x bf16, K and V) held for
roughly window + batch tokens, ~37 MB per request at today's TP4 layout, and
the ignore-DCP layout *is* today's layout. Section 3.2 has the mechanics and
5.6 the file list; what follows is the reasoning that shaped them.

Why it cannot work as-is, precisely:

- The drafter's sliding-window cache is a second KV-cache group, and three
  base-commit guards assert `dcp_world_size == 1` for it:
  `HybridKVCacheCoordinator.__init__`, `SlidingWindowManager.__init__`, and
  `SlidingWindowSpec.max_memory_usage_bytes` (plus the fork's own
  `_dsa_sliding_window_max_pages`). Upstream main *still* rejects
  sliding-window groups under DCP today; #40996 only added full-attention plus
  Mamba.
- Sharding the drafter's KV across the DCP group is not an option even in
  principle: vLLM's GQA DCP requires every rank in a DCP group to hold the
  same KV head (`tensor_parallel_size > total_num_kv_heads` and
  `dcp <= tp // kv_heads`), and the drafter has 8 KV heads at TP4.
- The drafter's `ParallelConfig` does not inherit `decode_context_parallel_size`
  (it copies only PP/TP/executor fields), so nothing fails at config time; the
  failures are at KV-config and layer-construction time.

The only viable design is therefore to let the drafter group **ignore DCP
entirely**: every rank stores every position for its own TP shard of heads,
which is exactly today's DFlash layout (2 KV heads per rank, all positions), so
it costs no extra memory and adds no collectives; only the target's verify
pass pays the DCP cost. Upstream's `dcp_world_size_for_kv_cache_spec` already
returns 1 for non-full-attention groups and `MambaManager` already undoes the
block-size scaling, which is the template. What has to change, all Python:

| where | change |
|---|---|
| `v1/core/kv_cache_coordinator.py` | drop the assert; pass `dcp_world_size=1` to sliding-window managers and to their `find_longest_cache_hit` |
| `v1/core/single_type_kv_cache_manager.py` | `SlidingWindowManager` undoes the base class's `block_size *= dcp` (as `MambaManager` does upstream) and drops its assert |
| `v1/kv_cache_interface.py`, `v1/core/kv_cache_utils.py` | drop the two sliding-window DCP asserts; in `resolve_kv_cache_block_sizes` scale only the sharded (full-attention) group's block size |
| `v1/worker/block_table.py` | per-group CP world size: the drafter group's table is sized `cdiv(max_len, block_size)` and its slot mapping runs with `TOTAL_CP_WORLD_SIZE=1` |
| `v1/attention/backends/flash_attn.py`, `v1/worker/cp_utils.py` | a per-layer "replicated under DCP" flag that makes the FA builder/impl take the `dcp_world_size == 1` path and exempts the layer from `check_attention_cp_compatibility` |
| `model_executor/models/qwen3_dflash.py` (overlay) | set that flag on the drafter's attention layers |

It came to 297 lines over eight files, with CPU tests driving the real
patched block tables and slot-mapping kernel under a faked 4-rank DCP group,
the block-size resolution, the compatibility check, and the decision helper.

### 6.4 Other items deliberately left out

- **Query replication.** Removes the 5.5 ms/step q all-gather at the cost of
  replicating `q_b_proj` and `W_UK_T` on every rank (~1.2 GB). The fork has
  neither the config field nor `DCPGroupColumnParallelLinear`; new work, and
  it applies equally under 2a or 2b.
- **Candidate compaction.** Under DCP ~3 in 4 of each row's 2048 candidates
  are `-1`; the decode kernel masks them (no wasted memory traffic) but its
  loop bound is still the full width. Compact to a prefix, pass the count as
  `topk_length`, bound the kernel loop by it. Sub-millisecond by the estimate
  in section 4.
- **LMCache connector under DCP.** Each rank would store and retrieve its own
  shard. Orthogonal to the multi-group work in `lmcache-mg`.
- **`cp_kv_cache_interleave_size > 1`.** Rejected at startup.

## 7. Validation

### Done here (CPU, no GPU required): 91 tests, all passing

`tests/` drives the **real patched kernels** (extracted from the overlay files
by AST so the tests cannot drift from the source) under
`TRITON_INTERPRET=1`, with the DCP collectives emulated by a fake group.

- **Index filter.** Every kept index equals the physical slot the KV writer
  used, derived independently from the untouched slot-mapping kernel's formula;
  across the group each in-range position survives on exactly one rank; at
  `dcp_size=1` the output is bit-identical to the unpatched baseline kernel.
- **Top-k merge.** The merged result reproduces the global top-k for several
  world sizes, interleaves and K; all ranks agree; short contexts that leave
  high ranks empty still merge correctly; padding stays `-1`; the prefill
  `row_starts` offset reads scores from the right column.
- **Attention merge.** Sharded partial softmax plus the LSE merge reproduces
  single-rank attention to 1e-5, including a row owned entirely by one rank
  and the replicated-heads case of a TP=1 MTP layer.
  The patched mixed-batch forward turns an all-`-1` row into `(0, -inf)` even
  when the kernel wrote NaN, and returns `[T, H]` fp32.
- **Prefill bounds.** The localized causal bounds equal a naive count of this
  rank's causal tokens; summed over ranks they equal the global bound; at
  `dcp=1` the kernel output is identical to the baseline kernel's.
- **MTP decode path.** Expand-then-localize equals a naive per-token count
  and the reverse order provably does not; the localizer handles the 1-D
  flatten layout and the 2-D native layout in place, and never mutates the
  shared `seq_lens` on the plain path; the static logits width covers every
  local bound by brute force; the merge precedes spec-decode unpacking; the
  builder never reads the proposer's stale `dcp_local_seq_lens`; the launcher
  defaults to MTP.
- **Replicated drafter group.** The decision helper shards full attention
  and MLA and replicates sliding window and Mamba, rejecting mixed groups;
  block-size resolution yields (256, 64) for the GLM + DFlash layout at DCP4
  and is unchanged at DCP1; the real patched `MultiGroupBlockTable` under a
  faked 4-rank group maps every position for the CP=1 group and exactly this
  rank's quarter for the sharded one, with the right row counts; the
  compatibility check accepts a replicated impl and still rejects a sharded
  one without LSE; static checks that the coordinator does all hit arithmetic
  on manager block sizes with per-group dcp, that the flash-attn override
  precedes the DCP-combine setup, and that the launcher mounts and preflights
  every staged file.
- **Static invariants.** The custom op and its fake have identical signatures,
  the layer passes every argument, no top-level definition was dropped, and no
  attribute in the builder's `__init__` is read before assignment. That last
  test was written after it caught a real bug in this patch set: the DCP guard
  was first placed above the `use_fp8_kv_cache` assignment it reads, which
  would have raised `AttributeError` on the first DCP boot.

Run with `PYTHONPATH=tests .venv/bin/python -m pytest tests/ -q`.

### Done on the cluster (2026-09-04, autonomous overnight run)

Deployed with `rollout_dcp.sh` (stage + SHA verify, flushers, teardown,
worker-first launch, container-death and swap-storm watchdog, real-generation
probe, automatic restore of the production launcher on any failure) and
checked with `post_boot_checks.sh` (log scan, warm-up sweep, determinism
matrix, streaming benchmark with `/metrics` deltas, concurrent mixed batches,
long-context probe, log scan). Raw results are under `results/`.

**Boot 1, `dcp4-dflash-131k`**: TP4 + DCP4 + DFlash K=7, max-model-len
131072, max-num-batched-tokens 2048, KV pool 8 GB per rank (production's
value). HEALTHY after 536 s; workers `Worker_TP{r}_DCP{r}`; startup lines
`GPU KV cache size: 524,288 tokens` and `Maximum concurrency for 131,072
tokens per request: 4.00x` (production: 131k tokens, 1.09x at 120k). Only
pre-existing kernels JIT-compiled during warm-up. Then, against the
production DCP1 baseline measured on the same nodes 40 minutes earlier (same
greedy prompts, thinking off, DFlash K=7 on both, `results/baseline-dcp1-prod`):

| prompt   | decode tok/s | accepted/cycle | cycle ms      | TTFT s      |
|----------|--------------|----------------|---------------|-------------|
| count100 | 57.0 -> 45.4 | 7.87 -> 7.87 (byte-identical text) | 137.6 -> 172.8 | 0.39 -> 0.50 |
| prose    | 21.3 -> 16.1 | 2.94 -> 2.79   | 137.9 -> 173.1 | 0.45 -> 0.62 |
| code     | 44.8 -> 34.5 | 6.29 -> 6.17   | 140.1 -> 178.4 | 0.48 -> 0.62 |

Reading it: acceptance is unchanged (the drafter is untouched; count100 is
byte-identical), and the cost is a constant **+36 ms per verify cycle**, i.e.
-22% single-stream decode. §4 predicted 16 ms from single-token collectives;
under DFlash the verify pass carries 8 tokens, so the gathered-q all-gather and
the LSE/output reduce-scatter move ~0.5 MB per layer each, 78 layers, on a
switchless ring. Production is not run-to-run deterministic at greedy for
prose/code (two reps differ on the same stack), so hash equality is only
meaningful for count100; the other prompts are judged on coherence, acceptance
and rate, all of which held.

- Determinism matrix (512 uncached, 1536 uncached, 1024 cached + 1536): all
  coherent; the cached prompt reused 256-token target blocks and 64-token
  drafter blocks (`prefix_cache_hits_total` advanced).
- Concurrent mixed batches (a decode, two prefills of 3k/6k tokens and a
  second decode in flight, staggered): all four coherent, no inference-time
  JIT, no errors.
- Long context: a 92,366-token prompt (random words tokenize at 1.04 tok/word,
  so the 120k target came out short) answered correctly, TTFT 316 s (~290
  tok/s prefill). KV usage returned to zero afterwards.
- Memory: post-boot MemAvailable 1.77 GB on rank 0 (the API server and the
  scheduler live there: `vllm serve` 0.76 GB RSS, EngineCore 0.63 GB) and
  2.5-3.0 GB on the other ranks; 0.69 GB / 1.5-1.7 GB after the whole check
  sequence. Swap-out over the sequence was under 20 MB per node. No OOM, no
  swap storm.
- Inference-time JITs (all succeeded, all four ranks): `_fp8_mqa_logits_kernel`
  and `_prepare_dflash_inputs_kernel` on first call (pre-existing kernels),
  and `_build_prefill_chunk_metadata_kernel` about 33k tokens into the long
  prompt. Cause: the indexer's 256 MB logits budget starts splitting a chunk's
  query once the *global* sequence length passes 32k at 2048 tokens per chunk,
  and the slice bounds then take on new Triton integer-specialization classes
  (`%16`, `==1`). Fixed for boot 2 with `do_not_specialize` on the two bounds
  (§5.5) plus a 40k-token prompt in the warm-up sweep. Boot 2 confirmed it:
  the 40k prompt crossed the split threshold with no recompile. The same
  kernel compiled once more on boot 2's first *mixed* batch (a decode plus a
  prefill): `uncompressed_seq_lens[num_decodes:]` is then a 4-byte-offset
  view, which is Triton's other specialization axis (16-byte pointer
  alignment). Two variants in total, both pre-existing behaviour in
  production; `do_not_specialize_on_alignment=["uncompressed_seq_lens_ptr"]`
  would make it one, left as a follow-up because it was not boot-tested. (The splitter uses the
  global length even though DCP logits are 4x smaller; harmless, just more
  slices than needed.)

**Boot 2, `dcp4-dflash-262k`**: max-model-len 262144, KV pool 7 GB per rank.
The indexer gather workspace is 40 x max_model_len x 132 B (+0.7 GB at 262k
over 131k) and rank 0 ended boot 1 with 0.69 GB, so the pool was cut 1 GB to
keep that headroom. Results are appended below when the run completes.

Boot 2 came up HEALTHY after 511 s: `GPU KV cache size: 462,308 tokens`,
`Maximum concurrency for 262,144 tokens per request: 1.76x`. The workspace
manager logged the indexer gather buffer resizing 36 MB -> 1321 MB (exactly
40 x 262144 x 132 B; it was 660 MB at 131k), confirming it is the largest
workspace requester and scales with max-model-len. Post-boot MemAvailable:
rank 0 2.10 GB, other ranks 3.3 GB (boot 1: 1.77 / 2.5-3.0 GB), so the 1 GB
pool cut bought about 0.3 GB net on every rank. Check results:

- Warm-up: the 40,031-token prompt (131 s, ~305 tok/s) crossed the query-split
  threshold without a recompile; only the two pre-existing first-call kernels
  JIT-compiled. Memory after warm-up: rank 0 1.06 GB, others 2.1 GB (the CUDA
  caching allocator keeps the prefill transients; a one-time step).
- Determinism matrix: all four coherent (hashes differ from production, as
  they do between production's own runs).
- Bench (mean of two reps): count100 45.5 tok/s at 7.87/cycle, text
  byte-identical to boot 1 and production; prose 15.2 tok/s at 2.64/cycle;
  code 37.0 tok/s at 6.56/cycle. Cycle time 172-176 ms, the
  same as boot 1: the 262k configuration adds nothing to decode.
- Concurrent mixed batches: all four coherent.
- **Long context: a 249,943-token prompt answered correctly, TTFT 893 s
  (~280 tok/s prefill), twice production's 120k ceiling.** No memory alert
  during the prefill (watch threshold: MemAvailable < 500 MiB or a swap-out
  burst); afterwards rank 0 0.74 GB, others 1.84-1.90 GB, swap-out nil.
- Final log scan: no errors, no additional JITs during the 250k prefill.

**Boot 3, `dcp4-dflash-512k`** (the user wants 512k as an option, with
ordinary sessions at 160-260k): max-model-len 524,288, KV pool 8.2 GB. Two
changes made it fit. First, under DCP each rank gathers and scores only its
local quarter of a sequence, so the indexer gather workspace is now sized on
`cdiv(max_model_len, dcp)` and the prefill splitter budgets on the same local
lengths (§5.5; it must, because the workspace is locked after warm-up and a
chunk whose local N exceeded it would try to grow it). The workspace went
from a would-be 2.8 GB to 661 MB, the same as at 131k. Second, the pool:
per-request cost at 131,072 local tokens is ~7.9 GB (60.6 kB per token per
rank, drafter window 2048 x 6 layers adds 13 MB), so 8.0 GB would have been
1.01x; 8.2 GB gives margin. Startup: `GPU KV cache size: 543,668 tokens`,
`Maximum concurrency for 524,288 tokens per request: 1.04x`, workspace
resized 36 -> 661 MB, HEALTHY after 514 s. Post-boot MemAvailable: rank 0
2.41 GB, other ranks 3.6 GB (the best of the three boots; the flushers
actually ran during this load, see the rollout note below). Check results:

- Warm-up (40k prompt 132 s), determinism, bench and concurrent: all
  coherent, numbers identical to boots 1-2 (count100 45.2 tok/s @ 7.87,
  cycle 174 ms).
- **500k probe aborted at 23% by hand.** From about 80k tokens in, every rank
  lost memory in step with the prefill (~17 kB per global token per rank) and
  rank 0 swapped out 1 GB in one 30 s interval, the swap-storm signature.
  The client was killed; the engine dropped the request cleanly (no crash,
  no OOM, 0 running), but MemAvailable did not recover: the growth was
  GPU-side allocator reserve, not request state (worker peak RSS rose only
  0.26 GB over boot 2's). The cluster was rolled back to the boot-2 stack.
- Diagnosis: the local-length logits budget. Boot 2's splitter bounded
  M x N_global x 4 B <= 256 MB, but the logits are only local-sized, so its
  real per-slice transients never exceeded 64 MB. Boot 3 budgeted on
  N_local, so M stayed at 2048 and each slice's transients grew with context
  to 256 MB, a different size every chunk, and the caching allocator's
  reserve grew ~2 GB per rank by 117k tokens (fragmentation on a growing
  size sequence). Boot 4 keeps the local-sized workspace (a fixed 661 MB,
  verified) but budgets logits on the global length again, i.e. boot 2's
  proven transient profile at any max-model-len. `post_boot_checks.sh` now
  runs the long-context probe under an automatic memory guard.

**Boot 4, `dcp4-dflash-512k-b4`**: boot 3's configuration with the splitter's
logits budget back on the global length (local-sized workspace kept). Same
startup facts (543,668 tokens, 1.04x, workspace 661 MB, rank 0 2.37 GB
post-boot); warm-up, determinism, bench and concurrent identical to the
other boots. **The 500k probe completed: 499,797 tokens, TTFT 1934 s (~258
tok/s), correct answer, no guard trip.** Memory trajectory during the
prefill (rank 0 / others): 712 / 2.4 GB at 20 s, 599 / 2.3 GB at 85k
tokens, 404 / 2.2 GB at 245k, then flat between 300 and 500 MB on rank 0
and 1.6-2.0 GB on the others to the end; the control run on the boot-2
stack (250k) drifted 0.2 GB in the same way. So the boot-3 growth was the
local-length budget, and at 512k the remaining problem is only rank 0's
headroom (API server + scheduler), which bottoms at ~300 MB during a full
prefill. 512k is therefore possible but leaves nothing for anything else;
the user chose a 300k-class window with room for the NVMe offload tier
instead. Two levers if 512k is wanted later: cap
`cudagraph_capture_sizes` at the real maximum verify batch (12 seqs x 8
tokens = 96; the default list captures up to 512 and the graphs took 0.79
GiB), and the API-server footprint (0.8 GB RSS on rank 0).

**Boot 5, `dcp4-dflash-300k`** (the NVMe-phase configuration): max-model-len
307,200, KV pool 6 GB, boot-4 overlay. `GPU KV cache size: 396,715 tokens`,
`Maximum concurrency for 307,200 tokens per request: 1.29x`, HEALTHY after
546 s. Post-boot MemAvailable rank 0 4.78 GB, others 6.0 GB; after the
warm-up sweep 3.6 / 4.8 GB. Warm-up, determinism, bench (count100 45.4 tok/s
@ 7.87, byte-identical) and concurrent all clean; the guarded 290k probe was
running with rank 0 at 3.1 GB when it was interrupted by an operator error
(below), and was re-run on the relaunch `dcp4-dflash-300k-r2` together with
the soak (`results/dcp4-dflash-300k-r2/soak.json`).

**Incident 17:24.** A dry run of the launcher (`DRYRUN=1`, added to print the
NVMe-tier docker command) was executed on rank 0 while it was serving; the
launcher's `docker rm -f` step ran before the dry-run gate and removed the
serving container. No node was harmed (ranks 1-3 sat idle in NCCL), but the
API was down for ~12 minutes until `rollout_dcp.sh` relaunched the stack
(`SKIP_PREFLIGHT=1`, the preflight otherwise refuses to start without a
healthy stack). Fixes: the launcher never touches a container in dry-run
mode, and the rule in HANDOVER.md: never run the launcher by hand on a
serving node, only through the rollout.

- **Soak, 2026-09-04 18:00-19:34 on `dcp4-dflash-300k-r2`** (`dcp_soak.py`,
  `results/dcp4-dflash-300k-r2/soak.json`): 12 iterations, each a fresh
  120k-token cold prefill, the same prefix with a new question (GPU prefix
  hit), a concurrent mixed batch and a decode check. Cold TTFT
  402-404 s (~298 tok/s), warm TTFT 1.7-2.3 s with 119,552 of
  ~120k tokens hit (the trailing block and the new question recomputed),
  decode 44.9-45.4 tok/s, every concurrent batch coherent, no memory
  alert from the external watch (available under 400 MiB or swap-out
  bursts). No drift in any metric across 1.4M prefilled tokens. This is the
  baseline the NVMe tier has to beat: a 120k context costs 403 s cold and
  2 s warm; the tier's job is to make the cold case a reload.

**NVMe direct tier, `dcp4-dflash-300k-direct` (2026-09-04 21:07)**, the
phase-2 deliverable (`docs/NVME-DESIGN.md`): 307,200 window, 6 GB pool,
`KVTIER=1 KVTIER_MODE=direct`. Boot HEALTHY, 48 bounce slots (185 MB pinned)
per rank, rank 0 4.57 GB after boot. Post-boot checks clean. Offload probe
(`offload_checks.sh`, 100k-token prefix, `results/dcp4-dflash-300k-direct/`):

| step | TTFT | external hits |
|---|---|---|
| cold prefill | 335 s | 0 |
| same prefix, new question (GPU hit) | 1.88 s | 0 (GPU prefix cache) |
| 3 x 100k other prompts (evict) | 334 s each | 0 |
| **same prefix after eviction (NVMe reload)** | **3.43 s** | **99,328 of 99,975** |

Files after the probe: 8,502 per rank, 7.7 GB per rank, identical counts on
all four nodes; the log scan found no error or I/O failure. The reload is
~100x faster than the cold prefill; the ~650 tokens not served are the
trailing block (never offloaded under the eagle rule) plus the new question.
Memory during the probe stayed above 3 GB on rank 0 and 4.5 GB elsewhere
(guard never tripped). Restart durability: see below.

Note: from 22:00 on 2026-09-04 the GPU SM clocks are locked at 2000 MHz
(1995 MHz effective) at the user's request; all numbers above were taken at
the default 2418 MHz application clock. Measured at the lock (boot
`dcp4-dflash-300k-direct2`): count100 44.4 tok/s at 177 ms per cycle versus
45.2 at 173 ms, about 2%, consistent with a cycle dominated by the DCP
collectives rather than SM throughput.

**Restart durability, `dcp4-dflash-300k-direct2` (22:04-22:43), the stack
left serving.** Same configuration plus `PYTHONHASHSEED=0` in the container
(the fork seeds the block-hash chain from `os.urandom` otherwise, so the
first direct run's files could never match after a restart). Offload probe
again: cold 339 s, GPU hit 1.87 s, evictions 338 s each, **NVMe reload
3.05 s (99,328 of 99,975 tokens)**; 8,502 files / 7.7 GB per rank,
identical on all nodes; clean log scan. Then a full engine relaunch through
the rollout and the same prefix again: **4.61 s, 99,328 of 99,975 tokens
served from NVMe on a fresh engine.** Cold prefill of that context costs
335-339 s; the durable reload costs 3-5 s. The launcher's defaults now equal
this configuration (307200 / 2048 / 6e9 / KVTIER=1 direct) on all four nodes.

**Slab store, `dcp4-dflash-300k-slab2` (2026-09-04 23:47 to 2026-09-05 01:16),
the fixed-size ring buffer that replaces the unbounded per-block files**
(`KVTIER_MODE=slab`, NVME-DESIGN.md §9, `deploy_slab.sh`). Boot A ran with
a 4.5 GB cap (892 target slots per rank, less than four 100k prefixes) so
eviction had to happen: the post-boot checks were clean, the offload probe
reloaded its cold prefix as a miss (339 s, as required), and the two slab
files stayed at 4,497,616,384 bytes on all four ranks. Boot B restarted into
the 150 GB cap: the scheduler indexed 4,460 rows from A's slot headers and
its own in-process reload hit in 3.22 s. Boot C restarted again and reloaded
B's prefix from the rebuilt index in **5.39 s, 99,328 of 99,975 tokens**.
Boot B also exposed a fork bug: prefixes that A had stored hit only 1,280
tokens after the restart, because the connector leaves one block per prefill
step unstored and the probe's warm step had been refilling the holes in
every earlier run. NVME-DESIGN.md §10 has the analysis and the fix (15th
overlay file); its validation, `deploy_slab_fix.sh dcp4-dflash-300k-slab3`,
is recorded below.

**Connector fix, `dcp4-dflash-300k-slab3` (2026-09-05 01:44-02:28), the
stack left serving as `dcp4-dflash-300k-slab3-e`.** Boot D rolled out the
15th overlay (NVME-DESIGN.md §10) onto the existing 150 GB slabs, then the
offload probe ran on fresh prefixes with `--no-warm --seed-base 1000`, so
every prefix was stored exactly once and nothing refilled it:

| step (boot D, fixed connector) | TTFT | external hits |
|---|---|---|
| cold prefill, fresh 100k prefix | 342 s | 0 |
| 3 x 100k other prompts (evict) | 338-339 s each | 0 |
| **same prefix after eviction, stored once, no warm step** | **3.39 s** | **99,328 of 99,980** |

Boot E, a full engine relaunch: the once-stored prefix reloaded in
**8.25 s (99,328 of 99,980)** and the third eviction prompt, stored once and
never reloaded, in **3.85 s (99,328 of 99,938)**. Before the fix the same
shape of request hit 1,280 tokens. On disk (`results/dcp4-dflash-300k-slab3-e/rank0-headers.json`)
boot D's four prompts left 1,556 target rows, 389 per prompt, which is every
block but the trailing one, in 7-8 row batches per step instead of 6-7; the
slabs grew by 1.96 GB per prompt, identically on all ranks
(15,513,038,336 bytes). Post-boot log scans on rank 0 were empty and
MemAvailable stayed above 3.3 GB on rank 0 and 4.7 GB elsewhere. The
launcher's defaults equal this configuration (307200 / 2048 / 6e9 /
`KVTIER=1` slab / 150 GB / `PYTHONHASHSEED=0`) on all four nodes.

## 8. Decode latency under DCP: where the 36 ms went, and what came back (2026-09-05)

The +36 ms per DFlash verify cycle (138 ms at DCP1, 173 ms at DCP4, §7) was
attributed in §4 to four ring collectives per layer using the all-reduce
latency table. A torch-profiler trace of one count100 request on each rank
(`dcp_profile.py`, `analyze_trace.py`, `results/dcp-profile-comparison.md`)
says otherwise. Per 8-token verify pass, median of 13 passes, the same
image and overlays at `DCP_SIZE=1` (82k window) and at DCP4 (307k, slab
tier):

| kernel bucket per verify pass | DCP=1 | DCP=4 | delta |
|---|---:|---:|---:|
| MoE experts + marlin GEMM | 84.7 ms | 85.0 ms | 0 |
| b12x sparse MLA attention kernel | 12.3 ms (78 x 158 us) | 23.8 ms (77 x 309 us) | +11.5 |
| NCCL all-gather (query, LSE, indexer merge) | 0.4 ms | 12.3 ms | +11.9 |
| NCCL reduce-scatter (attention output) | 0 | 7.5 ms | +7.5 |
| NCCL all-reduce (TP) | 13.0 ms | 14.6 ms | +1.6 |
| GPU idle inside the pass | 0.7 ms | 4.5 ms | +3.8 |
| verify pass | 136 ms (129 + the 7 ms drafter graph) | 172.9 ms | +36 |

Two corrections to §4. The collectives cost 19.4 ms, not 33: the big
all-gathers run at ~105 us and the reduce-scatters at ~99 us, so halving
their payload is worth 2-3 ms each. And a third of the penalty was never
communication: under DCP the index filter leaves the other ranks' three
quarters of the 2,048 top-k slots in place as -1, and the b12x kernel masks
them per cell but still walks them, for 64 gathered heads instead of 16.

**Candidate compaction** (`GLM_DCP_COMPACT=1`; `compact_dcp_candidates` in
`sparse_utils.py`, a Triton kernel that moves each token's owned candidates
to the front and returns the count; `flashmla_sparse.py` threads the count
to the kernels; `b12x_sparse_helpers.py`, now a 16th overlay, passes it as
the b12x `topk_length`). Boot `dcp4-dflash-300k-compact`:

| per verify pass | DCP=4 | DCP=4 + compaction |
|---|---:|---:|
| attention/indexer kernels | 24.1 ms | **2.9 ms** |
| collectives | 34.4 ms | 34.3 ms |
| verify pass | 172.9 ms | **150.7 ms** |

| bench (1 rep, thinking off, greedy) | DCP=4 | + compaction |
|---|---:|---:|
| count100 decode tok/s, cycle | 44.6, 175.7 ms | **49.7, 157.7 ms** |
| prose | 14.5 | **17.1** |
| code | 36.1 | **43.2** |
| accepted per cycle (count100 / prose / code) | 7.87 / 2.57 / 6.49 | 7.87 / 2.67 / 6.73 |

count100 text is byte-identical; prose and code hashes differ, as they do
between two reps of the baseline itself (greedy is not run-to-run
deterministic on those prompts). Each rank now walks about a quarter of the
candidates, so DCP4's attention time is below DCP1's 12.3 ms: the sharded
cache parallelises the attention as well as the memory. Net DCP overhead
after compaction: about 13 ms per cycle, all of it collectives.

**Query gather before expansion** (`GLM_DCP_Q_PREGATHER=1`; the ranks
all-gather the 256-wide pre-expansion query instead of the 576-wide absorbed
one and expand all 64 heads locally against a replicated `W_UK^T`, ~740 MB
per rank; `dcp_pregather_expand` in `mla_attention.py`, byte-equal to the
stock path in `tests/test_dcp_q_pregather.py`). Boot
`dcp4-dflash-300k-compact-pregather` on top of compaction: verify pass
150.7 -> 150.1 ms, all-gathers 12.1 -> 12.0 ms, count100 49.7 -> 50.6 tok/s,
prose 17.1 -> 17.3, code 43.2 -> 40.2 (single rep, acceptance 6.73 -> 6.50,
noise), workers ~300 MB less host memory. The big all-gathers run at 88-105
us whether they carry 590 KB or 262 KB: the ring's per-collective latency
floor, not bytes, is what remains. Kept as an off-by-default flag.

What is left of the DCP penalty is ~13 ms per cycle of ring collectives:
78 query all-gathers, 78 LSE all-gathers, 76 output reduce-scatters and
~20 indexer merges, each at its latency floor. Fewer collectives, not
smaller ones, is the next lever: the fork's single all-to-all merge
(`dcp_a2a_lse_reduce`, disabled on the ring by the overlay) replaces two of
the four per layer once a switch is in place, and overlapping the indexer
merge with the query gather on the top-k layers is worth ~2 ms on the ring.
Full query replication would remove the query gather entirely for
~2.6 GB/rank of int8 `q_b_proj`, which the memory budget does not have.

**Same-clock baseline.** The DCP1 production numbers above were taken at
the 2418 MHz application clock, before the 2000 MHz lock. Re-taken at the
lock on 2026-09-05 06:00 (`results/baseline-dcp1-prod-2000mhz`, production
launcher, 2 reps): count100 56.5 tok/s at 138.9 ms (7.87 accepted/cycle,
text identical), prose 19.6 at 139.9 ms, code 48.0 at 140.4 ms. The lock
costs about 1% at DCP1. Against the compaction stack's 50.5 tok/s at 155.2 ms,
the DCP4 penalty at the same clock is 16.3 ms per cycle, 11% on count100
(it was 36 ms and 22% before compaction).

**Serving configuration after this work:** `dcp4-dflash-300k-compact-prod`,
307,200 window, 6 GB/rank pool, slab tier, compaction on by default
(`DCP_COMPACT=1` in the launcher), profiler and pre-gather off.

