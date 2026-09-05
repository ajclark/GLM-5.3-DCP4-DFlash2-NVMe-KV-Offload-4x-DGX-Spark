# Handover: GLM-5.3 DCP, 2026-09-04

What exists, what was verified, what to do next. `docs/DESIGN.md` has the
reasoning; this file is the operational summary.

## State

- **Deployed (2026-09-04).** The DCP lane lives beside production on every
  node: overlays in `~/glm-dcp/` (SHA-verified) and
  `~/glm53big/launch-glm53big-dcp.sh`. The production launcher
  `~/glm53big/launch-glm53big-dflash.sh` was never modified, and
  `~/lmcache-mg/` was never touched.
- Boot 1 (max-model-len 131072, KV 8 GB) validated: 524,288 KV tokens, 4.00x
  concurrency at 131k, acceptance unchanged, decode -22% (+36 ms/cycle),
  mixed batches and a 92k prompt correct, no OOM. Boot 2 (262144, KV 7 GB):
  462,308 KV tokens, 1.76x at 262k, a 249,943-token prompt answered in 893 s,
  same decode as boot 1. 512k (8.2 GB pool) also boots and served a 500k
  prompt, with no headroom left. **Serving now (2026-09-05): the slab-store
  stack, 307,200 window, 6 GB pool, NVMe tier on** (`KVTIER=1`,
  `KVTIER_MODE=slab`, 150 GB per rank, `PYTHONHASHSEED=0`), and the DCP
  launcher's defaults equal it. An evicted or restarted 100k context reloads
  from NVMe in 3-8 s instead of 335 s, now from a single store (the
  connector fix of NVME-DESIGN.md §10, validated 02:28). GPU clocks locked
  at 2000 MHz. Since 2026-09-05 05:00 the decode path also runs with
  candidate compaction (`DCP_COMPACT=1`, launcher default): the DCP verify
  cycle went from 173 to ~151-158 ms, count100 44.6 -> 49.7 tok/s, prose
  14.5 -> 17.1, code 36.1 -> 43.2 (DESIGN.md §8). Serving label:
  `dcp2-dflash-180k-prod2` since 2026-09-05 17:50: after the three-lane
  concurrency sweep the user chose DCP2 as the daily default (within 5-8% of
  DCP1 at every concurrency, twice its KV; DCP4 loses 19% at C=12). Launcher
  and rollout defaults: `DCP_SIZE=2 MAXLEN=180224 KVBYTES=6e9 KVTIER=1`; the
  DCP4 lane is `DCP_SIZE=4 ./rollout_dcp.sh <label> 307200 2048 6000000000 1`.
  pi's glm-5.3 entry on the sandbox is a 155,648 context window with 32,768
  max tokens (270,336 for the DCP4 lane). The rule:
  vLLM rejects a request whose prompt plus max_tokens exceeds the window
  (measured, HTTP 400), pi sends its full maxTokens every time and compacts
  at contextWindow minus a 16,384 reserve, so (contextWindow - 16,384) x 1.05
  tokenizer margin + maxTokens must stay under the window: 299k < 307,200
  here, 155,648 for the 180k lane. (At the old 120k/120k, pi hit the 400
  past 87k tokens and recovered by auto-compacting.) Details in DESIGN.md §7-8,
  NVME-DESIGN.md and `results/`.
- Sixteen patched vLLM files plus one new module in `overlay/` (thirteen for
  DCP, the engine scheduler's invalid-block recovery, the offloading
  connector's store progress, the b12x sparse-attention helper's candidate
  count, and `multinode.py`), staged flat in `stage/glm-dcp/` with checksums
  (seventeen files), plus `launch-glm53big-dcp.sh` (TP4 + DCP4 + **DFlash K=7**
  by default; `MAXLEN`, `MAXBATCHED`, `KVBYTES` env overrides). Five files
  make the sparse-MLA target DCP-aware; eight let the DFlash drafter's
  sliding-window KV group stay replicated across the ranks.
- 156 CPU tests pass: `PYTHONPATH=tests .venv/bin/python -m pytest tests/ -q`.
- Operating scripts: `rollout_dcp.sh <label> [MAXLEN] [MAXBATCHED] [KVBYTES] [KVTIER]`
  (deploy with watchdog and automatic production restore; `SKIP_PREFLIGHT=1`
  when the stack is already down), `post_boot_checks.sh <label> <baseline_dir>
  [longctx_tokens]` (long-context probe under a memory guard),
  `dcp_soak.py` / `run_soak_after_checks.sh` (stability soak: cold and warm
  TTFT, concurrency, decode, memory per iteration), `offload_checks.sh` and
  `deploy_kvtier.sh` (NVMe tier: cold / warm / evict / reload / restart),
  `deploy_slab.sh` (slab store: small cap, real cap, restart) and
  `deploy_slab_fix.sh` (first-time-store proof: `--no-warm --seed-base`
  probe, restart, reload of once-stored prefixes), `dcp_probe.py`,
  `compare_runs.py`, `restore_production.sh` (one command back to
  production). Detached follow-ons (`run_checks_after_boot.sh`,
  `run_deploy_after_soak.sh`) chain these so a session drop does not stop
  a sequence half-way.
- Phase 2, NVMe-durable KV cache: `docs/NVME-DESIGN.md` and
  `overlay/vllm/v1/kv_offload/tiering/multinode.py` (launcher lane `KVTIER=1`;
  `KVTIER_MODE=slab` by default, a fixed-size ring buffer of
  `KVTIER_DISK_BYTES` per rank; `direct` grows per-block files without
  bound; `tiered` keeps a CPU cache tier but caps a reload at the tier's
  size). The fork's own tiering is single-host and cannot be used here. The
  tier lane also mounts `offloading_scheduler.py`, which fixes a fork bug
  that left one block per prefill step unstored (NVME-DESIGN.md §10); a
  probe with a warm step between cold prefill and reload hides that class of
  bug, so use `dcp_probe.py --no-warm --seed-base N` for first-time stores.

## The one-paragraph version

GLM-5.3's KV cache is replicated on all four ranks because MLA is multi-query.
Decode Context Parallelism shards it instead, so the same 8 GB per rank holds
one copy of a ~520k-token context rather than four copies of a 131k one. The
engine's scheduler, KV manager, block tables and memory accounting are already
DCP-aware at the fork's base commit; what was missing is DCP support in the
DSA sparse indexer and the sparse attention backend, a blanket refusal of fp8
KV under DCP in the MLA layer, and any way for the DFlash drafter's
sliding-window KV group to coexist with a sharded target. This patch set adds
the first two, narrows the third, and makes the drafter group replicated
(every rank keeps every position for its own TP shard of heads, which is the
layout it already has) while the target shards.

## Things a reader will want to know that are not obvious from the code

1. **The GitHub issue that started this (#54907) does not apply.** Its bug is
   in `vllm/models/deepseek_v32/common/kernels.py`, a directory this fork does
   not have. The fork computes K-side norm/RoPE in plain torch, unconditionally
   on every rank. Applying #54908 here changes nothing.

2. **`a2a` is not a tuning knob on this cluster, it is impossible.** The Sparks
   are cabled as a switchless ring; NCCL all-to-all pairs non-adjacent ranks
   that have no cable between them and fails with `ibv_modify_qp 110`. Upstream
   made `a2a` the GLM default in #50382; the patch rejects it at startup with
   that explanation so nobody re-enables it from an upstream doc.

3. **DFlash runs under DCP with its cache replicated, and that is the only
   way it can.** Sharding the drafter's cache is impossible in principle:
   vLLM's GQA DCP needs `tp > kv_heads`, and the drafter has 8 KV heads at
   TP4. Upstream main still rejects sliding-window groups under DCP. So one
   helper (`cp_world_size_for_kv_cache_spec`) declares sliding-window groups
   CP-free, and the KV manager, block tables, slot mapping and flash-attn
   builder/impl all follow it; the DFlash proposer itself is untouched
   because the runner already hands it its own group's block table. Memory
   cost: zero (it is today's layout). MTP under DCP was worked through first,
   estimated within noise of DFlash, and dropped by the operator; the launcher
   refuses `mtp` with a message and the verify-side code serves DFlash.

4. **Decode gets slower, but less than it sounds.** Measured +36 ms per
   DFlash cycle (-22% single-stream decode); the estimate was +16 ms of NCCL
   per cycle (four extra collectives per layer across 78 layers, sized from
   the measured `nccl-latency-results.md` table), paid once per DFlash cycle
   on the verify pass: ~145 ms becomes ~161 ms, about 11%. The drafter adds
   nothing. Query replication would recover about a third of that and is
   listed as follow-up work.

5. **Raising max-model-len is nearly memory-neutral only because of one
   deliberate change.** The sparse bf16 prefill workspace grows as
   `5 * max_model_len * 576 * 2 B` (0.69 GB at 120k, 2.3 GB at 400k) but is
   unreachable under DCP, so the patch stops reserving it. Without that,
   262144 would cost about +1.4 GB per rank.

## If you deploy (or redeploy)

`./rollout_dcp.sh <label> [MAXLEN] [MAXBATCHED] [KVBYTES]` does the whole
thing and never leaves the cluster down: it refuses to start unless production
is healthy, stages and SHA-verifies `stage/glm-dcp/` and the launcher, starts
the cache flushers, tears production down, launches ranks 3,2,1,0 with the
DCP launcher, waits for `/health` while watching for container death and
swap-out storms, then proves a real generation (`/health` returns 200 on a
wedged engine). Any failure runs `restore_production`. Then
`./post_boot_checks.sh <label> results/baseline-dcp1-prod [longctx_tokens]`
and `compare_runs.py`. Boot 1 used `131072 2048` (8 GB pool), boot 2
`262144 2048 7000000000`.

What to look for: the startup line `Maximum concurrency for N tokens per
request` (4.00x at 131k with an 8 GB pool), two KV cache groups, no "DCP not
support" assertion, `jit_monitor` warnings in the log scans (a JIT on a
memory-starved node is what wedged the cluster once), and the `mem` lines
between phases: rank 0 carries the API server and scheduler and sits about 1
GB below the other ranks.

Roll back with `./restore_production.sh`; nothing in `~/glm-triton/` or the
image was modified, so the rollback is a relaunch.

## If it goes wrong

- **Before any rollout, re-stage.** `stage/glm-dcp/` is what the nodes get;
  an edit to `overlay/` that is not copied there (and `stage/SHA256SUMS`
  regenerated) deploys the old code. The first slab boot ran the pre-review
  module for exactly this reason.
- **Kill patterns must not match your own shell.** `pkill -f 'foo.sh'` from a
  command line that mentions `foo.sh` kills the shell running it (exit 144)
  and everything after it silently never runs. Use `'[f]oo.sh'`.
- **Every manager entry point runs for every request**, attached or not:
  the connector calls `touch()`/`complete_*()` for prompts shorter than a
  block that never reached `lookup()`. Slab boot A crashed the engine on
  that (IndexError in `touch`); all entry points now no-op until attached.
- **GPU clocks are locked at 2000 MHz** (`sudo nvidia-smi -lgc 2000,2000`, set
  2026-09-04 22:00 at the user's request; reads back as 1995 MHz, the
  nearest supported step). The lock does not survive a reboot or driver
  reload; re-apply it on all four nodes after either. Every benchmark before
  22:00 that night ran at the default 2418 MHz application clock.
- **Never run `launch-glm53big-dcp.sh` by hand on a serving node**, not even
  `DRYRUN=1`: it begins with `docker rm -f vllm_glm53big`. (The dry run is
  now gated before that step, after it took rank 0 down once on 2026-09-04.)
  Deploy and relaunch only through `rollout_dcp.sh`; when the stack is
  already down, `SKIP_PREFLIGHT=1 ./rollout_dcp.sh <label> ...` skips the
  "production must be healthy" preflight.
- **Memory drains during a long prefill on every rank.** That is GPU-side
  allocator growth, not the KV pool (which is fixed) and not request state
  (aborting the request will not give it back). Boot 3 showed it: per-chunk
  transients that scale with context and change size every chunk make the
  caching allocator's reserve balloon. Kill the probe client (`pkill -f
  dcp_probe.py`), which the engine handles cleanly, then relaunch with the
  proven configuration; only a relaunch returns the memory. The splitter's
  logits budget must stay on the *global* sequence length for this reason,
  even though the logits are local-sized (see DESIGN.md §7, boot 3).

- **Hang at startup, all ranks in NCCL.** Ranks disagreed on the number of
  collectives. The prefill chunk list is meant to be identical on every rank;
  check that `build_prefill_chunk_metadata` still returns `None` on the
  *global* length, and that no rank skipped `_merge_dcp_topk_global`.
- **Coherent at DCP1, gibberish at DCP4.** Almost certainly the index mapping.
  `tests/test_dcp_index_filter.py` checks it against the untouched slot-mapping
  kernel's formula; re-run it, then check `cp_kv_cache_interleave_size` really
  is 1 and that block_size is still 64. If `SPEC_MODE=none` is coherent and
  `dflash` is not, the drafter's group is being sharded somewhere: look for
  the flash-attn builder's "its KV cache group is ... but the attention impl
  is ..." error, and check that the startup log reports the sliding-window
  group with block size 64 and the target group with 256.
- **Prefix-cache hits vanish at DCP4.** The target group hashes at 64 tokens
  but caches at 256; both sides compose through `BlockHashListWithBlockSize`.
  Run with `--no-enable-prefix-caching` to confirm the rest is fine, then
  compare `cache_full_blocks` and `find_longest_cache_hit` block sizes.
- **`AttributeError` or `NoneType` in the attention layer.** The patch asserts
  with a message naming the backend when the decode kernel returns no LSE;
  that means `can_return_lse_for_decode` did not take effect, i.e. the
  `flashmla_sparse.py` mount is the `glm-triton` one, not the DCP one.
- **Small numeric drift between ranks that should be identical.** The fp8
  sparse kernel falls back from b12x to the Triton path per rank on an
  exception (`_fp8_flash_mla_kernel`). If that fires on one rank only, the
  LSE merge mixes two numerically different but mathematically equal
  kernels; harmless, but the "B12x GLM sparse MLA failed" warning in that
  rank's log is the tell. Independent review flagged it; nothing to fix.
- **OOM during the profile run.** Lower `--max-num-batched-tokens` first (the
  gathered query and the fp32 accumulator scale with it, 4x under DCP); only
  then reduce `--max-model-len`. Do not touch `--kv-cache-memory-bytes` or
  `--gpu-memory-utilization`.

## Independent reviews

Two separate reviewers traced the patches against the unmodified tree for
wrong output, crashes and distributed hangs.

*Target-side patches (five files):* no blocking bug across nine categories
(collective counts, local/global sizing, decode aliasing, static shapes,
custom op inside CUDA graphs, slot-mapping formula, DCP1 byte-identity,
read-before-assign, LSE numerics). Two low-severity notes: the kernel
fallback drift is in the failure guide above; the other assumed the MTP head
is replicated at `draft_tensor_parallel_size: 1`, which is not the case (that
setting only reaches the separate-draft-model path).

*Replicated-drafter patches (eight files):* no blocking bug across seven
invariants (consistency of the sharded/replicated decision in every consumer,
prefix-cache hash composition on insertion and lookup, the DFlash proposer
receiving the CP=1 block table, no runner assumption of a single CP size,
DCP1 byte-identity, signatures, hangs). Three low-severity items, all fixed:
the compatibility-check exemption now tests `dcp_world_size == 1` rather than
`> 1` (dense MLA impls carry a `-1` placeholder before their first forward);
the input batch is no longer rebuilt once for nothing at DCP1; and the
flash-attn builder now applies the GQA-DCP head rule to token-sharded groups,
so a drafter layer without a sliding window fails at startup instead of
gathering queries across ranks that hold different KV-head shards.

## Follow-up work, in the order I would do it

DFlash under DCP is built and deployed (2026-09-04). `docs/DESIGN.md`
sections 3.2, 5.6, 6 and 7 have the mechanics, the reasoning and the numbers.

1. (Measured 2026-09-05, DESIGN.md §8.) The +36 ms per cycle was a third
   attention kernel walking masked candidates and two thirds collectives.
   Candidate compaction (`GLM_DCP_COMPACT=1`) recovered 22 ms; the query
   gather before expansion (`GLM_DCP_Q_PREGATHER=1`, ~740 MB/rank) is worth
   only what the payload halving buys at ~105 us per all-gather. What is
   left is ~13 ms of ring collectives; the switch's single all-to-all merge
   (`dcp_a2a_lse_reduce`, disabled on the ring in `flashmla_sparse.py`) is
   the next lever, then full query replication if memory ever allows.
2. (Done in boot 3.) The indexer workspace and prefill splitter now budget on
   local lengths under DCP, which is what makes max-model-len 524,288 fit.
3. Long-prompt TTFT: ~290 tok/s at 92k. Profile one 2048-token chunk at 100k+
   context before touching anything; the gather workspace and the sparse
   indexer dominate, not the attention kernel.
4. (Done 2026-09-05.) Candidate compaction with the kernel's `topk_length`
   is in and measured: the attention kernel went from 309 to 37 us per
   layer under DCP4. The profile is in `results/dcp-profile-comparison.md`;
   `dcp_profile.py` / `analyze_trace.py` / `measure_variant.sh` reproduce it
   (launch with `PROFILER_DIR=/kvcache/profiles`, tier on so the traces land
   on the host).
5. Pool vs context: `KVBYTES` trades KV tokens for host headroom (the indexer
   gather workspace is 40 x max_model_len x 132 B). Rank 0 is the constraint
   (API server + scheduler); read its `MemAvailable` after a full check
   sequence before raising either knob.
6. `_build_prefill_chunk_metadata_kernel` still compiles a second variant on
   the first mixed decode+prefill batch (pointer-alignment specialization of
   the `uncompressed_seq_lens[num_decodes:]` view). Add
   `do_not_specialize_on_alignment=["uncompressed_seq_lens_ptr"]` to its
   `@triton.jit` and boot-test it; until then `post_boot_checks.sh`'s
   concurrent phase triggers that compile while someone is watching.
7. MTP can be re-added as a launcher lane in one line if ever wanted; nothing
   in the patch prevents it.
8. Report the offloading-connector store-progress bug (NVME-DESIGN.md §10)
   to the fork; upstream has since moved to chunk-based progress
   (`storable_chunks`), so check whether it still applies there. A related
   refinement: flag only the drafter's KV group as `is_eagle_group` (the fork
   flags none, so every group is treated as eagle) and the target group's
   final block would be stored too, one more 256-token block per hit.
