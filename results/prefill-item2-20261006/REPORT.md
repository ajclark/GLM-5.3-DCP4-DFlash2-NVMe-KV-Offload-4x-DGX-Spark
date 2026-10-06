# Prefill throughput (2026-10-06)

Status: **all kept as launcher defaults (2026-10-06): 1c, E1, E2b, and the NCCL TP all-reduce on both PCIe links (2 channels, 4 NICs; persistent `roce-p0-twin` profiles, verified across a reboot of all four nodes). Uncached prefill 587 / 575 / 567 → 759 / 747 / 740 tok/s at 4K / 32K / 60K (+29–31%), 60K TTFT 108.4 → 83.0 s, decode unchanged. The micro-batch overlap was dropped after analysis; E2 pipelining was built and measured (+0.8%) and is left off.**

Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3 Int4-Int8Mix (Int4
group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

The levers, by the labels used below and in the evidence directories:
- **1c:** prefill-size DCP collectives on RoCEnante, striped over both PCIe links of the DCP port.
- **E1:** the DCP layout copies removed.
- **E2:** RoCE gather staging: E2a a larger kernel grid, E2b an early local copy, and E2 pipelining (chunked staging).
- **E3:** the NCCL TP all-reduce over both PCIe links: 2 channels first, then 4 NICs.

The RoCE network and its addresses are listed in [NETWORK.md](../../docs/NETWORK.md).

## 1c design: prefill DCP collectives on RoCEnante, striped over both PCIe twins

**What changes.**
- Prefill-size DCP collectives move off NCCL onto RoCEnante. These are the query all-gather, the indexer candidate
  merge and the output reduce-scatter; each is at most 36 MiB per rank at a 2048-token chunk.
- RoCEnante stripes every payload over the HCAs it is given. `roceP2p1s0f1` is added next to `rocep1s0f1`.
  - Both reach the DCP partner over the same 200 GbE port, but through different PCIe Gen5 x4 links (domains 0002 and
    0000).
  - Today NCCL carries all of these collectives over a single x4 link, `roceP2p1s0f0`+`roceP2p1s0f1` merged, both on
    domain 0002.
- GIDs (index 3, RoCE v2) give HCA index h the same subnet on both partners:

  | pair | `rocep1s0f1` | `roceP2p1s0f1` |
  |---|---|---|
  | 06c4–365c | 192.168.101.x | 192.168.100.x |
  | ddbf–a218 | 192.168.105.x | 192.168.103.x |

**Switches.** `runtime/vllm029/launch.sh` and `start-glm53.sh`, all default off:

| switch | default | 1c value |
|---|---|---|
| `ROCE_GATHER_MAX` | `4MiB` | `36MiB` |
| `ROCE_DCP_HCAS` | `rocep1s0f1` | `rocep1s0f1,roceP2p1s0f1` |
| `NCCL_CHANNELS`, `NCCL_CTAS` | `1` | (not used by 1c) |

- A DRYRUN on spark-06c4 showed the edited launcher's docker command is byte-identical to the old one with defaults.
  The edited launcher was installed on all four nodes.
- The TP ring keeps its explicit `GLM_ROCE_RING_EXCLUDE=roceP2p1s0f1`, so it is unaffected.

**Exactness: bit-identical.**
- Gathers are copies.
- The two-rank reduce-scatter is one IEEE add per element (`runtime/vllm029/roce/glm_roce/install.py:149-166`), which is
  commutative.

**Memory.**
- Pinned per rank: 6 × 36 MiB = 216 MiB (`roce_layout`: 2 slots × (2 recv + 1 send)).
- Plus 108 MiB of device gather scratch.
- That is about 0.3 GB more than today's 36 MiB per rank. Rank 0 had ~1.5 GB MemAvailable on 10-05.

**Gates before serving.**
- Pair test (`runtime/vllm029/roce/run_dcp_pair_test.sh`, extended with `--prefill-tokens 1024,2048`, `HCAS`,
  `GATHER_MAX`, `TEST_FILE`): byte-equal to NCCL and to the CPU reference at prefill sizes on both pairs.
- With two HCAs, the decode checks too (eager, graph replays, latency at T=8/32), since decode DCP collectives also
  stripe.
- Then the boot: rank-0 MemAvailable ≥ 0.8 GB under load.

**Keep criteria:**
- ≥ +3% on 60K prefill;
- decode within ±2 ms per cycle at fixed K=7 and K=1;
- no RoCE health errors.

## Baseline: production boot `verifycap10-stack`, up 34 h, 2026-10-06 01:43–02:20 UTC

Tool: `runtime/vllm029/prefill_bench.py`, new. Each request gets a unique `cache_salt`, so neither the GPU
prefix cache nor the NVMe tier can serve it (0 hits on every request), and each waits for an idle endpoint (no request
was contaminated). Prompts are corpus text (the token corpus that `runtime/vllm029/kvtier_bench.py` builds; not
published).

**Uncached prefill** (`A-prod-baseline/ttft.json`, 3 runs each):

| prompt | prefill s (median) | tok/s |
|---:|---:|---:|
| 4,096 | 6.98 | 587 |
| 32,768 | 56.97 | 575 |
| 61,440 | 108.43 | 567 |

Run-to-run spread is ≤ ±1.5%.

**Decode, fixed K** (`A-prod-baseline/cycle-k{7,1}.json`, `runtime/vllm029/cycle_bench.py --same-prompt`, control file
set to fixed and restored to `auto` after):

| context | K=7 (ms per cycle) | K=1 (ms per cycle) |
|---:|---:|---:|
| 2K | 119.9 | 73.6 |
| 16K | 110.9 | 74.9 |

**A/A quality envelope** (per-token data not published). Fixed prompts of 8K, 24K, 48K and 98K tokens, each prefilled
fresh twice in the same boot, 256 greedy tokens plus the first token's top-20 logprobs:
- The first token's top-1 matched in every pair. Its logprob moved by 0.006–0.09 nats.
- The rest of the top-20 moved by up to 0.7 / 1.1 / 0.7 / 4.5 nats. At 98K the rank-2 candidate changed.
- Greedy continuations first diverged at token 7 / 2 / 8 / 13. The corpus prompts have near-tie continuations.

So **prefill is not reproducible run to run today** (`VLLM_MARLIN_USE_ATOMIC_ADD=1` atomic split-K, among others). End
to end, a bit-identical change can only be checked against this envelope. Its proof is the pair test's byte
comparisons.

## 1c results (2026-10-06)

**Pair tests** (model stopped; `pair-1hca/`, `pair-2hca/`). Both DCP pairs pass every check, byte-equal to NCCL and to
the CPU reference:
- T=1024 and 2048: query gather 36 MiB, indexer merge 32 MiB, LSE, output reduce-scatter 64 MiB in;
- with 2 HCAs, also the decode checks: eager, 300 graph replays at T=8/32, latency.

Eager ms per collective at T=2048:

| collective | NCCL | RoCE, 1 HCA | RoCE, 2 HCAs |
|---|---:|---:|---:|
| query gather | 5.3–5.6 | 4.9–5.1 | **3.6–3.8** |
| indexer merge | 4.8–5.2 | 4.2–4.6 | **3.2–3.4** |
| output reduce-scatter | 5.1–5.5 | 4.9–5.3 | **3.8–4.0** |

With 2 HCAs, decode collectives also got faster:
- T=8: 22.5–23.2 µs per collective (NCCL 44–45; `results/step3-roce-20261001/REPORT.md` had 25 µs on 1 HCA);
- T=32: 41–44 µs (NCCL 84–85; step 3: 54).

**Prefill** (`B-1c-2hca/ttft.json`; boot with `ROCE_DCP_HCAS=rocep1s0f1,roceP2p1s0f1 ROCE_GATHER_MAX=36MiB`, profiler
armed):

| prompt | baseline | 1c | change |
|---:|---:|---:|---:|
| 4K | 587 tok/s | 648 | **+10.5%** |
| 32K | 575 | 635 | **+10.4%** |
| 60K | 567 (108.4 s) | 629 (97.7 s) | **+11.0%** |

The quality phase prefills ran 8–10% faster too.

**Decode** (`B-1c-2hca/cycle-k{7,1}.json`, same method as the baseline):

| context | K=1, 1c | K=1, baseline | K=7, 1c |
|---:|---:|---:|---:|
| 2K | 74.0 ms | 73.6 | 116.2 ms (runs 101–119) |
| 16K | 74.5 ms | 74.9 | 118.2 ms (runs 115–121) |

- K=7 runs scatter ±8 ms in both boots. Pooled K=7 means are 115.0 ms with 1c and 115.4 ms at baseline.
- **Decode is unchanged.**

**Quality.** A vs 1c falls inside the A/A envelope: first divergences at tokens 2–93, logprob deltas of the same size.
This is as expected for a byte-identical collective on a stack that varies run to run.

**Memory.**
- Rank-0 MemAvailable: 2.9 GB at idle after boot, 0.9 GB during the 98K quality prefill. That is above the 0.8 GB
  floor and about 0.3 GB below the pre-1c state, as designed.
- After the clean restart: 3.2 GB at boot, 2.8 GB after the decode check.

**Decision: kept.**
- `launch.sh` defaults are now `ROCE_DCP_HCAS=rocep1s0f1,roceP2p1s0f1` and `ROCE_GATHER_MAX=36MiB`. A DRYRUN of the
  new defaults is byte-equal to the running 1c boot's command. Installed on all four nodes.
- Previous behaviour: `ROCE_GATHER_MAX=4MiB ROCE_DCP_HCAS=rocep1s0f1`.

## Chunk profile (2026-10-06)

The 4K chunk profile was taken in the 1c boot (`B-1c-2hca/profile-ctx4096/`: per-rank `summary.json` buckets and
profiler tables; the torch traces are not published). Rank-0 buckets for one 2048-token chunk:

| bucket | share |
|---|---:|
| routed MoE | 26.4% |
| NCCL TP all-reduce | 17.6% |
| RoCE DCP collectives | 14.0% |
| sparse MLA | 12.6% |
| dense Marlin | 11.4% |
| copies | 7.8% |
| indexer | 0.6% |

**Profiler settings.** With vLLM's defaults, `/stop_profile` stalled the engine:
- vLLM's stop-time CUDA-time table (`torch_profiler_dump_cuda_time_total`, default true) runs `key_averages` in the
  worker.
- On ranks 1–3 (13.5 MB traces) it took ~4 min. On rank 0 (52 MB trace) it pushed the node into heavy swapping: 13 GB
  of swap, 41 MB available, the worker in state D.
- The 60K/120K profiles were therefore not taken, and the model was restarted with 1c and no profiler.

**Fix:** `launch.sh` now always adds `--profiler-config.torch_profiler_with_stack=false` and
`--profiler-config.torch_profiler_dump_cuda_time_total=false` when `PROFILER_DIR` is set.

## Analysis: micro-batch overlap, and where the serial work is (2026-10-06)

**The micro-batch overlap was dropped.** It would split each 2048-token chunk into two micro-batches and overlap one
half's collectives with the other half's compute. A read-only analysis estimated its first phase at −8% to +3%, for two
reasons:
- **Half-size penalty.** Splitting a 2048-token chunk streams the expert weights twice. The 135-token tail chunk in
  the profile measures the streaming floor at about 5.4–5.8 ms per MoE layer, so the split costs at least
  150–300 ms per chunk.
- **No room to overlap.** Marlin and sparse MLA hold all 48 SMs with 91–101 KB of shared memory each, so NCCL's
  98 KB-smem kernel cannot run beside them.

The analysis ranked three serial-work removals instead:
- **E1:** DCP layout copies, bit-identical, high confidence.
- **E2:** pipelined RoCE staging, bit-identical, medium confidence.
- **E3:** TP all-reduce over both PCIe links, reduction-order-level, medium to medium-low confidence.

Corrections it made to the earlier numbers:
- The 4K profile holds two forwards: the 2,041-token chunk and a 135-token tail. The true chunk is 3,132 ms, with
  33.5% communication.
- Rank 0's 52 MB trace came from shared-memory busy-wait Python events recorded with stack capture on, so profiles
  now run with stack capture off.

## E1: DCP layout copies removed (2026-10-06)

**Design.** Everything sits behind `VLLM_DCP_GLUE=1` (launcher `DCP_GLUE`, default 1 since 06:30 UTC), outside
`manifest.json`, so the slab salt is unchanged. Four layout copies go:

1. **Middle-dim all-gathers** (the DCP query gather [T,16,576] and the indexer candidate merge [T,2048,2]) run as
   last-dim gathers of their 2-D views, `runtime/vllm029/roce/glm_roce/install.py`. RoCEnante writes the concatenated
   layout directly, which equals `torch.cat(dim=1)` by index arithmetic, so the dim-0 gather + movedim + reshape copy is
   gone.
2. **The combine's correction kernel** writes head-major output (`runtime/vllm029/verify_cap_overlay/glm_fast/dcp_glue.py`:
   the stock `_correct_attn_cp_out_kernel` body statement for statement, with separate output strides). The
   reduce-scatter's `movedim(0,1).contiguous()` becomes a no-op on both the RoCE and the NCCL path.
3. **The reduce-scatter add** writes into a strided view of the output layout (`torch.add(out=)`), so the post-copy is
   gone.
4. **The backend's full-output `out.masked_fill_`** for empty local shards is skipped (`verify_cap_overlay` copy of
   `flashinfer_mla_sparse_sm120.py`). The correction kernel already stores `where(factor == 0, 0, out*factor)`, and
   factor is 0 for those rows because their lse is −inf, so junk never reaches the result.

The image build pins sha256 of the stock kernel, `_cp_lse_common` and `cp_lse_ag_out_rs`
(`runtime/vllm029/verify_cap_overlay/glm_fast/install.py`), so upstream drift fails the build. The query `cat` (~0.9%)
was left: it would need changes to the custom `fused_q` kernel.

**Tests.** All passed (`e1-tests/`):
- glm_fast CPU tests: 19/19, including the switch wiring.
- Single-GPU bit-identity (`runtime/vllm029/verify_cap_overlay/glm_fast/gpu_test_dcp_glue.py`, 149 PASS / 0 FAIL on
  spark-a218 and spark-365c):
  - stock (masked_fill + in-place correction + copying reduce-scatter) == glue, byte for byte;
  - T ∈ {2, 8, 32, 96, 1024, 2048}, both lse bases;
  - NaN / inf junk in empty-shard rows;
  - 2-D gather == cat for the real shapes.
- Pair tests, both pairs, image `verifycap11-dcpglue`, `GLUE=1`:
  - 12 eager combine cases, glue == stock == CPU reference;
  - 60 CUDA-graph replays of the glued combine;
  - prefill collectives byte-equal at T=1024/2048;
  - the decode eager, graph and latency checks.
  - Isolated per-collective time at T=2048 dropped from 3.6–3.8 to 3.0–3.4 ms (query gather) and from 3.2–3.4 to
    2.7–2.8 ms (indexer merge).

**Prefill.** Same image, same day, uncached, 3 runs each (`E1-A-glue0/`, `E1-B-glue1/`):

| prompt | glue off | glue on | change | vs 10-05 baseline |
|---:|---:|---:|---:|---:|
| 4K | 650 tok/s | **689** | +6.0% | +17.4% |
| 32K | 637 | **678** | +6.4% | +17.8% |
| 60K | 632 (97.2 s) | **670 (91.7 s)** | +6.0% | **+18.1%** (108.4 s) |

The glue-off arm matches the 1c boot (648 / 635 / 629), so the image change itself is neutral. The 98K quality
prefill took 148.7 s, against 159.5 s with 1c and 176.0 s at baseline.

**Decode.** Fixed K, `E1-B-glue1/cycle-k*.json`: K=1 at 72.1 / 74.0 ms (baseline 73.6 / 74.9). K=7 medians are
114.2 / 111.8 ms (baseline 119.9 / 110.9; ±8 ms run scatter). Unchanged or slightly better.

**Quality.** Against the baseline, results sit inside the A/A envelope: first divergence at tokens 2–96, logprob
deltas no larger than A/A.

**Memory.** Rank 0 had 2.9–3.0 GB at boot and 1.28 GB after the full bench, with swap at 114 MB (unchanged).

**Decision: kept.**
- Launcher defaults are now image `verifycap11-dcpglue-20261006` with `DCP_GLUE=1`; a DRYRUN of the defaults is
  byte-equal to the running boot.
- `start-glm53.sh` defaults to the new image, and its preflight requires `VLLM_DCP_GLUE=1` and the 36 MiB
  gather limit.
- Rollback: `DCP_GLUE=0` (same image), or `DCP_IMAGE=…verifycap10-stack-20261001`.

## E2: RoCE gather staging (2026-10-06)

**Design** (`runtime/vllm029/roce/glm_roce/gather_v2.py`, the adapter builds it for the DCP group when a switch is set):
- A subclass of the vendored runtime plus a copy of its all-gather kernel. Vendored b12x is untouched.
- Only shards of at least 4 MiB (prefill) take the new path; decode gathers keep the stock kernel and grid.
- Levers:
  - **E2a, larger grid** (`GLM_ROCE_LARGE_BLOCKS` 16/32; capped at 32 so all blocks are co-resident and the
    last-block doorbell cannot deadlock).
  - **E2b, early local copy** (`GLM_ROCE_OWNCOPY_EARLY=1`): the local shard is written to its output columns right
    after the doorbell, while the NIC moves the payload, instead of after the flag wait. Same bytes to the same
    addresses.
- The protocol (slots, flags, epoch, counters, proxy) is unchanged.

**Pair sweep, both pairs** (`e2-sweep/`, eager ms per collective at T=2048, all byte-equal to NCCL and the CPU
reference, glue checks included):

| arm | query gather | indexer merge | reduce-scatter |
|---|---:|---:|---:|
| stock (8 blocks) | 3.05 | 2.73 | 3.58 |
| 32 blocks | 3.01 | 2.70 | 3.64 |
| **early copy, 8 blocks** | **2.68** | **2.39** | **3.28** |
| early, 32 blocks | 2.94 | 2.69 | 3.58 |
| early, 16 blocks | 3.01 | 2.59 | 3.52 |

- A larger grid gives nothing, and it hurts when combined with the early copy: the copies are not SM-bound.
- Early copy with 8 blocks saves 0.30–0.37 ms per large collective.
- The gate run (`e2-sweep/gate-b8-e1/`: decode eager, graphs and latency 21–22 µs at T=8; glue combine; prefill
  1024/2048) passed on both pairs.

**Live** (image `verifycap12-e2-20261006`, `ROCE_OWNCOPY_EARLY=1`, `E2b-owncopy-early/`):

| prompt | E1 | E1 + E2b | change | vs 10-05 baseline |
|---:|---:|---:|---:|---:|
| 4K | 689 tok/s | **695** | +0.9% | +18.4% |
| 32K | 678 | **691** | +2.0% | +20.2% |
| 60K | 670 (91.7 s) | **684 (89.8 s)** | +2.1% | **+20.7%** (108.4 s) |

Decode at K=1: 72.8 / 74.0 ms, unchanged.

**Decision: kept.** Launcher defaults are image `verifycap12-e2-20261006` with `ROCE_OWNCOPY_EARLY=1`; the preflight
requires it.

**Below the estimate.** The analysis estimated +4.7–6.5%; the grid and early-copy levers delivered about 2%. Per
36 MiB collective:
- the wire takes ~1.4 ms (the 1- vs 2-HCA difference, 1.3 ms, matches doubling one x4 link);
- ~1.2 ms is still staging before the doorbell plus copy-out of the peer shard after the wait.

Recovering that needs real pipelining: per-chunk doorbells in the kernel, chunked posting in the C proxy and
per-chunk flags. The estimate is about −1 ms per collective, roughly another +5%. It changes the proxy protocol in a
fail-stop runtime (medium risk, 1–2 days), so it was deferred at this point; it was built and measured later (see
"E2 pipelining experiment" below).

## E3: TP all-reduce over both PCIe links (2026-10-06)

**Benchmark** (`runtime/vllm029/roce/bench/nccl_tp_dualring.py`, `run_nccl_tp_dualring.sh`; serving stopped;
`e3-bench/`, whose logs keep the result records, warnings and errors, with the `NCCL INFO` lines dropped):
- Prefill-size TP all-reduce on all four ranks, eager.
- Slowest rank, median of 7×20.
- Error measured against an fp32 reference and the single-ring result.

| config | 12.6 MB | 25.2 MB | bus GB/s (25 MB) |
|---|---:|---:|---:|
| today: 1 ring, 1 channel, `roceP2p1s0f0+roceP2p1s0f1` merged (both on domain 0002) | 1.68 ms | 3.17 ms | 11.9 |
| two opposite rings (forward + reversed communicators), same NICs | 1.58 ms | 3.03 ms | 12.5 |
| 1 ring, **2 channels**, same NICs | 1.48 ms | **2.88 ms** | 13.1 |
| two rings over `roceP2p1s0f0` + `rocep1s0f1` (one NIC per PCIe domain), merged or not | fails | fails | — |

**Why cross-domain fails.** NCCL binds each channel's ring to one (virtual) NIC for both send and receive, and it
will not merge NICs from different PCI devices.
- On domain 0000 only port 1's twin (`rocep1s0f1`) has an IPv4 address, so no domain-0000 NIC reaches both ring
  neighbours. `ibv_modify_qp` times out; NCCL's hint is "NICs are not cross-rail connected".
- The fix is an IPv4 address on `rocep1s0f0` (`enp1s0f0np0`, port 0's twin on domain 0000) on all four nodes, in two
  new /24s for the 06c4–a218 and 365c–ddbf links. NCCL could then merge `rocep1s0f0`+`rocep1s0f1` into a second
  virtual NIC and run 2 channels, one per PCIe link.
- Expected: the 25 MB all-reduce from ~3.2 ms to ~1.6–1.8 ms, about +7–8% prefill.
- That is a host network change, so it was tested separately (part 2 below).

**By-products:**
- A second NCCL communicator costs about 70 MB per rank, not GBs.
- Two opposite rings change about 23–29% of output elements by one bf16 ulp, with the same error against fp32.

**Kept: NCCL 2 channels** (`NCCL_CHANNELS=2 NCCL_CTAS=2`, launcher default; `E3-nccl2ch/`):

| prompt | E2b | + 2 channels | vs 10-05 baseline |
|---:|---:|---:|---:|
| 4K | 695 tok/s | **714** (+2.7%) | +21.6% |
| 32K | 691 | **702** (+1.5%) | +22.0% |
| 60K | 684 (89.8 s) | **697 (88.2 s)** (+1.8%) | **+22.9%** (108.4 s) |

- **Quality** against the baseline is inside the A/A envelope: first divergence at tokens 2–52, or never for one 8K
  pair; logprob deltas no larger than A/A.
- **Decode:** K=1 at 72.4 / 74.1 ms, unchanged. K=7 at 113.9 / 119.1 ms, within its ±8 ms run scatter.
- Previous behaviour: `NCCL_CHANNELS=1 NCCL_CTAS=1`.

## E3, part 2: both PCIe links with temporary addresses (2026-10-06)

**Setup.**
- Temporary, runtime-only addresses were added: `enp1s0f0np0` (`rocep1s0f0`) set unmanaged by NetworkManager, MTU
  9000, with 06c4 `192.168.106.2` ↔ a218 `.106.1` and 365c `192.168.107.1` ↔ ddbf `.107.2`.
- A first try without `managed no` was flushed within a minute by NetworkManager's DHCP retry loop.
- Each node then had the IPv4 RoCE v2 GID at index 3, and both links pinged.

**Benchmark** (`e3-bench/4nic-*`, `stress-*`, `hash-*`). `NCCL_IB_HCA='=roceP2p1s0f0,roceP2p1s0f1,rocep1s0f0,rocep1s0f1'`
merged gives two 400G virtual NICs, `[4] rocep1s0f0+rocep1s0f1` (domain 0000) and `[5] roceP2p1s0f0+roceP2p1s0f1`
(domain 0002), with channel 0 on `[4]` and channel 1 on `[5]`:

| 25.2 MB TP all-reduce | time | bus GB/s |
|---|---:|---:|
| 1 link, 2 channels (production) | 2.88 ms | 13.1 |
| **4 NICs, 2 channels (both links)** | **1.81 ms** | 20.9 |
| 4 NICs, 4 channels | 1.72 ms | 22.0 |

- **Integrity:** 3,000 repeats per size on 4 NICs and on 1 link were all bit-identical to the first result.
- **Numerics:** the 4-NIC output hash equals the 1-link 2-channel hash at both sizes. The four-NIC all-reduce is
  bit-identical to production's, so it cannot change model numerics.
- **Memory:** about 10 MB more per rank (648 vs 638 MB for the first communicator plus context).

**Live** (`E3-4nic-2ch/`, production image with `NCCL_HCAS=…4 NICs…` and `ROCE_TP_EXCLUDE=roceP2p1s0f1,rocep1s0f0`;
the decode ring was unchanged, cw `rocep1s0f1` / ccw `roceP2p1s0f0`, split):

| prompt | 1 link, 2 ch | 4 NICs, 2 ch | change | vs 10-05 baseline |
|---:|---:|---:|---:|---:|
| 4K | 714 tok/s | **750** | +5.0% | +27.7% |
| 32K | 702 | **737** | +5.1% | +28.2% |
| 60K | 697 (88.2 s) | **730 (84.2 s)** | +4.7% | **+28.7%** (108.4 s) |

**Decode:** K=1 at 74.0 / 74.3 ms; K=7 at 117.6 / 119.1 ms (within scatter).

**Quality.** Inside the envelope at 8K / 24K / 48K. At 98K, one of two runs gave a different first-token
distribution:
- token 9611 fell to −2.7 and token 3932, the usual second token, came first;
- 9 earlier 98K runs across all configurations gave 9611 at −0.01 to −0.14.

The outlier is attributed to the stack's existing run-to-run variation (sparse top-2048 selection at 98K on top of
Marlin atomic split-K). The all-reduce is bit-identical to the validated 2-channel config, there were no NCCL or RoCE
warnings, and the stress test was clean. The baseline's single 98K A/A pair was too small to have shown such an event.

**Not kept at this point.** It depended on runtime-only addresses and on the interface staying out of NetworkManager.
A reboot or a NetworkManager restart drops them; mid-serving that would break NCCL's connections on `rocep1s0f0`.
- Production was restored to the persistent defaults: 1 link, 2 channels, E1 + E2b.
- Adopting it needs a persistent per-node configuration of `enp1s0f0np0` (a NetworkManager profile with a static
  IPv4 address and MTU 9000, replacing its DHCP profile), then the `NCCL_HCAS` 4-NIC default and
  `ROCE_TP_EXCLUDE=roceP2p1s0f1,rocep1s0f0`.

## Adopted permanently, with a reboot test (2026-10-06)

- **Persistent profile.** `node/roce-p0-twin.sh apply` (run on every node) created the NetworkManager profile
  `roce-p0-twin`: `enp1s0f0np0`, manual IPv4 (.106.x / .107.x), MTU 9000, IPv6 disabled, autoconnect, persisted as
  `/etc/netplan/90-NM-<uuid>.yaml`. It replaces NetworkManager's in-memory DHCP default ("Wired connection 2").
- **Launcher defaults.**
  - `NCCL_HCAS='=roceP2p1s0f0,roceP2p1s0f1,rocep1s0f0,rocep1s0f1'` and `ROCE_TP_EXCLUDE=roceP2p1s0f1,rocep1s0f0`; the
    DRYRUN is byte-equal to the validated test config.
  - `start-glm53.sh` now refuses to start unless every node has the IPv4 RoCE v2 GID at index 3 on
    `rocep1s0f0`.
- **Reboot test.** All four Sparks were rebooted (18:57 UTC) after a clean stop. They came back within about a minute
  with:
  - the same kernels;
  - `roce-p0-twin` connected and every RoCE address intact;
  - IPv4 GIDs on `rocep1s0f0`;
  - the GPU clock lock active (`gpu-clock-lock.service`, ~1995 MHz).

  `start-glm53.sh` then brought the stack up in 2 min 22 s: virtual NICs [4] and [5] present, the decode ring
  unchanged, DCP on both HCAs, early copy on.
- **Post-reboot performance** (`E3-4nic-postreboot/`):

  | prompt | prefill | vs 10-05 baseline |
  |---:|---:|---:|
  | 4K | 759 tok/s | +29.2% |
  | 32K | 747 | +30.0% |
  | 60K | 740 (83.0 s) | +30.5% |

  Decode K=1: 70.8 / 71.3 ms. Rank 0 had 2.2 GB MemAvailable at boot.

## E2 pipelining experiment (2026-10-06)

**Design** (image `verifycap13-pipe-20261006`, `ROCE_PIPE=1`, default off; `runtime/vllm029/roce/glm_roce/pipe.py`,
`runtime/vllm029/roce/glm_roce/_pipe_proxy.c`):
- A second, eager-only RoCE runtime per DCP pair, used for gathers of at least 4 MiB (prefill). It has its own pinned
  region, queue pairs and proxy thread.
- Decode stays on the vendored runtime, whose gather limit drops to 4 MiB, so pinned memory is unchanged overall.
  Graph capture is refused by the new path.
- The proxy is derived from the vendored one (`pipe_*` symbols). It adds a chunk dimension to the flags and
  per-(slot, chunk) ready words in the control record, and posts each chunk as soon as the kernel's last block
  marks it staged.
- The kernel stages in chunks (up to 8), copies the local shard early, then waits for and copies out each chunk
  while later chunks are on the wire.
- Op-level slots, sequences, the doorbell and the catch-up are the vendored ones, so the slot-reuse argument is
  unchanged.
- Vendored b12x is untouched (`runtime/vllm029/roce/tests/test_vendored_b12x.py` ok).

**Pair tests, both pairs** (`e2p-tests/`):
- **Gate:** all byte-equal.
  - 300 fresh-input stress rounds: 900 pipe ops plus 300 interleaved decode-size ops on the vendored runtime,
    0 mismatches.
  - Prefill collectives byte-equal to NCCL and the CPU reference.
  - Decode eager, graph and latency checks unchanged (21–23 µs at T=8); glue checks pass.
- **Chunk sweep** (ms per collective at T=2048):

  | config | query gather | indexer merge | reduce-scatter | sum |
  |---|---:|---:|---:|---:|
  | E2b | 2.68 | 2.39 | 3.28 | 8.35 |
  | pipe, 8 MiB chunks | 2.63 | 2.02 | 3.27 | 7.92 |
  | pipe, **4 MiB chunks** | **2.04** | **1.99** | **2.79** | **6.82** |
  | pipe, 16 MiB chunks | 2.64 | 2.21 | 3.09 | 7.94 |

  2 MiB equals 4 MiB at these sizes (the 8-chunk cap).

**Live** (4 MiB chunks, against the same-day post-reboot E2b boot; `E2p-pipe-4m*/`):

| prompt | E2b | pipe | change |
|---:|---:|---:|---:|
| 4K | 758.6 tok/s (n=3) | 766.1 (n=6) | +1.0% |
| 32K | 747.3 | 752.4 | +0.7% |
| 60K | 740.0 (83.0 s) | 746.2 (82.3 s) | +0.8% |

- The ranges do not overlap, so the gain is real but about a third of the isolated estimate.
- Likely cause: in the live forward the two DCP partners reach each collective at different times, and the wait for
  the later partner does not shrink with faster transfers.

**Cost.** During prefill, rank 0 runs four threads at 100%: the worker's main loop and three RDMA proxies (vendored
DCP, TP ring, pipe), about 13.5 CPU-min each in the 16 min since boot. The pipe adds one continuously spinning core
per node while traffic flows, on top of the protocol code.

**Status: off** (operator's decision). `ROCE_PIPE` defaults to 0, and production was restarted onto the launcher
defaults: image `verifycap12-e2-20261006`, 4-NIC NCCL, E1 and E2b, no pipe runtime. The code stays in the repo for
reference.
