# C1 DFlash coding sweep

**Stopped at the user's request.** Completed K=6, K=5, K=4. Skipped benchmarks: K=3, K=2, K=1, K=7. The exact original K=7 containers are restored and generation verified.
Restoration evidence is recorded in abort-status.json. No fresh K=7 control was measured in this run.

The earlier high-K sweep measured K=7 at 36.93 decode tokens/second. The best average here, K=6 at 37.10, differs by +0.47% from that earlier control. This comparison spans separate runs; small differences do not demonstrate a performance advantage.

The sweep was configured with order: K=6, K=5, K=4, K=3, K=2, K=1, K=7. K counts speculative tokens, excluding the bonus token.
TP4/DCP2, vLLM 0.29.0, DFlash2, C1, greedy decoding, thinking off. Three coding tasks, one warmup each and 3 measured repetitions each; 1200-token output limit. Other serving arguments are held fixed.

| K | Merge tok/s | LRU tok/s | Toposort tok/s | Geomean tok/s | vs K7 | Tokens/cycle | Cycle ms (est.) | Acceptance | Correct |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 36.81 | 38.55 | 35.99 | 37.10 | not measured | 5.12 | 137.4 | 68.6% | 8/9 |
| 5 | 35.75 | 34.88 | 35.15 | 35.26 | not measured | 4.55 | 129.3 | 71.0% | 9/9 |
| 4 | 32.71 | 33.32 | 32.99 | 33.01 | not measured | 4.10 | 123.6 | 77.4% | 8/9 |

Decode throughput excludes the first streamed token batch and its latency, using token IDs rather than text chunks. The geometric mean gives each coding task equal weight. Tokens/cycle means one bonus token plus mean accepted draft tokens; acceptance is accepted/proposed draft tokens. Engine metrics verify each configured K.

Correctness is checked by executing generated code with independent cases in a bounded subprocess. These checks cover behavior, not a formal complexity proof. Output lengths and token hashes are retained because different greedy responses can affect throughput comparisons. Three repetitions and short prompts do not establish long-context or concurrent performance.

Raw request/response and metric deltas are in each k*/ directory. summary.json contains per-case ranges and output lengths. checked-rows.json contains independent correctness results, including warmups. Restoration is recorded in abort-status.json for an interrupted sweep, or status.json for a completed sweep.

Failed measured outputs remain included in throughput averages. The small sample does not establish a correctness difference between draft lengths.

- K=4, topological_sort, repetition 2: ValueError: Graph contains a cycle; topological sort impossible.
- K=6, topological_sort, repetition 3: ValueError: Graph contains a cycle; no topological order exists.
