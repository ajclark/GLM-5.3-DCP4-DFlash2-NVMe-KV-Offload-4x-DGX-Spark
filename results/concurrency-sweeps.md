# Concurrency sweeps by lane (`~/spark-cluster-experiments/concurrency_cycle.py`, count-to-400 prompts, K=7)

cycle_ms = request_decode_time / drafts from the engine metrics; the tool needs
no restart. GPU clocks locked at 2000 MHz.

## DCP=4 + candidate compaction, 307k window, 6 GB pool (`dcp4-dflash-300k-compact-prod5`, 2026-09-05 16:50)

| C | tokens/step | cycle ms | marginal ms/token | mean accepted | aggregate tok/s | per-request tok/s |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 157.6 | - | 7.876 | 46.9 | 46.9 |
| 2 | 16 | 211.0 | 6.7 | 7.876 | 71.3 | 35.6 |
| 4 | 32 | 261.0 | 3.1 | 7.876 | 114.6 | 30.1 |
| 8 | 64 | 353.0 | 2.9 | 7.871 | 167.2 | 22.3 |
| 12 | 96 | 446.6 | 2.9 | 7.857 | 198.8 | 17.6 |

Raw: `results/concurrency-dcp4-compact.json`. The ~16 ms of DCP collectives
is a fixed cost per cycle, so it is 0.17 ms per token at C=12; the marginal
~2.9 ms per token is MoE expert streaming.

Reference, DCP=1 production lane at the 2418 MHz clock, the user's field
report sweep (harness-based counting prompts, `context-sweep-results.md`
in spark-cluster-experiments): C1 53.8 tok/s, C12 aggregate 192.6-204 tok/s.

DCP=2 and DCP=1 series with this tool: not yet run (each needs a rollout).
