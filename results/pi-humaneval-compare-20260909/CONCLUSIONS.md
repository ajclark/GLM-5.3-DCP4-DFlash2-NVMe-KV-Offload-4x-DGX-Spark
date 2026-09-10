# Conclusions from the aborted pi / HumanEval comparison

Keep the currently deployed stack. This sample does not establish a throughput
benefit from replacing it with the frozen V5 adaptive stack.

- **C1 was effectively flat, with lower measured rates on adaptive.** Whole-batch
  output throughput was 26.53 tok/s deployed and 26.22 adaptive (−1.15%).
  Server decode throughput per request was 36.54 versus 35.45 tok/s (−2.97%).
  With one small batch per configuration, these differences do not establish a
  statistically reliable slowdown; they provide no evidence of a speedup.
- **The higher-concurrency batch results are mixed and strongly affected by
  output length.** Adaptive's aggregate rate was higher in three of the nine
  matched cells. At C5, it appeared 55.8% faster in aggregate, but generated
  4,597 tokens versus 24,013 on deployed; batch duration was 119.2 versus 970.0
  seconds. Its per-request server decode rate in that cell was actually lower,
  15.51 versus 21.78 tok/s. These metrics describe different aspects of a batch
  whose realized concurrency changes as tasks finish.
- **Adaptation was active and the observed fallback behavior was correct.**
  C1 used K3 for 74, K5 for 186, and K7 for 523 completed verification cycles.
  Across the completed adaptive C1–C9 cells, the trace contains 9,841 verification
  rows, zero shortened cycles when adaptation was ineligible, and zero writer
  errors. Adaptation applies only to a single eligible decode request, so
  higher-concurrency cells primarily test fallback behavior and their C1 tails.
- **There is no quality advantage in this sample.** On the matched C1–C9
  batches, deployed passed 108/108 solves and adaptive 107/108. Adaptive failed
  HumanEval/132 at C2. Across all completed deployed C1–C12 batches, deployed
  passed 143/144, with HumanEval/132 failing at C12. These are repeated solves
  of twelve tasks, not a full HumanEval score.
- **The completed measurements passed their validity checks.** All 108 paired
  initial request hashes and native provider settings match. The 21 completed
  cells have full task sets and no pi/server token-accounting validity flags.
  No memory guard tripped. Whole-stack differences include the adaptive build's
  draft-cache repair and other V5 runtime changes, as well as its controller.

The run was stopped at the user's request during adaptive C10. Its unfinished
batch is preserved and excluded from throughput comparisons. Adaptive C11/C12
were not run. The controller restores the exact original containers and uses
one short native pi generation probe; the optional restored C1/C12 repeatability
batches are omitted. Exact restoration and endpoint health have been verified on all four Sparks.
The native pi probe passed its official test with no token-accounting errors.
See `restoration.json` and `restoration-probe-final/` for the evidence.

The main design limitation is finite twelve-task batches with native pi tools
and high thinking. HumanEval/132 repeatedly produced long reasoning tails;
maximum pi-session concurrency did not hold GPU decode concurrency constant.
This run therefore does not establish sustained C1–C12 capacity or a causal
engine speedup. Sequential lane order, naturally warming prefix caches, the
preserved/retried initial C3 timeout, and the absence of repeated batches add
uncertainty.

A future capacity measurement would need repeated batches that keep each
concurrency level occupied, with an explicit generation budget and identical
pi profiles on both stacks. No further sweep is being run here.

The full tables, metric definitions and source provenance are in [REPORT.md](REPORT.md).
Machine-readable measurements are in [summary.csv](summary.csv), with checks in
[audit.json](audit.json). The [chart](throughput.png) plots the completed cells.
