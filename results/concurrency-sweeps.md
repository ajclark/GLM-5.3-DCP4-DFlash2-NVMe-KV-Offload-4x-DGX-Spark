# Concurrency sweeps by lane (`~/spark-cluster-experiments/concurrency_cycle.py`, count-to-400 prompts, K=7)

cycle_ms = request_decode_time / drafts from the engine metrics; the tool needs no restart within a lane.
GPU clocks locked at 2000 MHz. 2026-09-05, boots `dcp4-dflash-300k-compact-prod5` (16:50), `dcp2-dflash-180k-sweep` (17:10),
the production launcher (17:20). Raw: `results/concurrency-{dcp4-compact,dcp2,dcp1-prod}.json`.

## Aggregate tokens/s

| C | DCP=1 | DCP=2 | DCP=4 | DCP=2 vs 1 | DCP=4 vs 1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 54.0 | 49.9 | 46.9 | -8% | -13% |
| 2 | 82.6 | 77.3 | 71.3 | -6% | -14% |
| 4 | 136.4 | 125.3 | 114.6 | -8% | -16% |
| 8 | 192.1 | 187.7 | 167.2 | -2% | -13% |
| 12 | 245.7 | 232.7 | 198.8 | -5% | -19% |

## Cycle time (ms) and marginal cost per token (ms)

| C | tokens/step | DCP=1 cycle | DCP=2 cycle | DCP=4 cycle | DCP=1 marginal | DCP=2 marginal | DCP=4 marginal |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 141.6 | 148.1 | 157.6 | - | - | - |
| 2 | 16 | 181.8 | 191.9 | 211.0 | 5.03 | 5.48 | 6.68 |
| 4 | 32 | 220.1 | 239.1 | 261.0 | 2.40 | 2.95 | 3.13 |
| 8 | 64 | 304.6 | 317.0 | 353.0 | 2.64 | 2.43 | 2.87 |
| 12 | 96 | 363.9 | 383.2 | 446.6 | 1.85 | 2.07 | 2.93 |

Acceptance is 7.83-7.89 of 8 in every cell. KV tokens at these pools: DCP=1 131k (8 GB), DCP=2 ~198k, DCP=4 396k.

## Reading

- Single stream, the DCP cost is the per-cycle collective floor: +7 ms (DCP=2) and +16 ms (DCP=4) per cycle.
- With concurrency the per-step payloads grow (the query gather carries ~7 MB at 96 tokens), so DCP=4's three-hop
  collectives become bandwidth-bound and its marginal cost per token stays ~2.9 ms while DCP=1 and DCP=2 fall to
  ~1.9-2.1 ms. The DCP=4 penalty therefore widens from 11% at C=1 to 19% at C=12; DCP=2 stays within 5-8% of DCP=1
  at every concurrency while holding twice the KV of DCP=1.
- For a multi-session workload the choice is DCP=2 unless a single session needs more than the 180k window or the
  four-fold pool is needed to keep several very long sessions resident.
- Reference: the user's earlier DCP=1 field-report sweep at 2418 MHz reached 192.6-204 aggregate at C=12 with a
  different tool; this series is the like-for-like one.
