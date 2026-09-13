# C1 DFlash coding sweep

**K=7 has the highest measured average on this coding corpus at 36.93 decode tokens/second.** The exact original K=7 containers are restored and serving. Generated code passed 86/90 measured output checks.

K=7 and K=8 differ by only 0.75%. Output lengths and timing vary across repetitions; a small gap should not be treated as a demonstrated performance advantage.

Order: K=16 down to K=7. K counts speculative tokens, excluding the bonus token.
TP4/DCP2, vLLM 0.29.0, DFlash2, C1, greedy decoding, thinking off. Three coding tasks, one warmup each and three measured repetitions each; 1200-token output limit. Other serving arguments are held fixed.

| K | Merge tok/s | LRU tok/s | Toposort tok/s | Geomean tok/s | vs K7 | Tokens/cycle | Cycle ms (est.) | Acceptance | Correct |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 28.96 | 31.01 | 27.30 | 29.05 | -21.3% | 5.95 | 203.2 | 30.9% | 9/9 |
| 15 | 30.72 | 31.17 | 29.20 | 30.35 | -17.8% | 5.92 | 195.5 | 32.8% | 8/9 |
| 14 | 30.95 | 31.63 | 30.21 | 30.93 | -16.2% | 5.89 | 190.4 | 34.9% | 9/9 |
| 13 | 34.58 | 31.95 | 30.49 | 32.30 | -12.5% | 5.90 | 185.2 | 37.7% | 9/9 |
| 12 | 32.56 | 33.81 | 31.23 | 32.51 | -12.0% | 5.82 | 179.3 | 40.2% | 9/9 |
| 11 | 33.05 | 32.61 | 30.75 | 32.12 | -13.0% | 5.76 | 180.3 | 43.2% | 9/9 |
| 10 | 33.53 | 33.67 | 31.34 | 32.83 | -11.1% | 5.71 | 174.3 | 47.1% | 8/9 |
| 9 | 36.00 | 37.21 | 34.65 | 35.94 | -2.7% | 5.80 | 161.0 | 53.3% | 8/9 |
| 8 | 36.33 | 38.18 | 35.49 | 36.65 | -0.7% | 5.65 | 153.5 | 58.1% | 9/9 |
| 7 | 37.07 | 38.15 | 35.60 | 36.93 | +0.0% | 5.39 | 145.6 | 62.7% | 8/9 |

Decode throughput excludes the first streamed token batch and its latency, using token IDs rather than text chunks. The geometric mean gives each coding task equal weight. Tokens/cycle means one bonus token plus mean accepted draft tokens; acceptance is accepted/proposed draft tokens. Engine metrics verify each configured K.

Correctness is checked by executing generated code with independent cases in a bounded subprocess. These checks cover behavior, not a formal complexity proof. Output lengths and token hashes are retained because different greedy responses can affect throughput comparisons. Three repetitions and short prompts do not establish long-context or concurrent performance.

Raw request/response and metric deltas are in each k*/ directory. summary.json contains per-case ranges and output lengths. checked-rows.json contains independent correctness results, including warmups. The original K=7 containers are restored at the end; status.json records their IDs.

The completed run has 30 warmups and 90 measured requests in the requested descending order. Restoration, truncation and memory observations are recorded in verification.json.

Failed measured outputs remain included in throughput averages. The small sample does not establish a correctness difference between draft lengths.

- K=10, topological_sort, repetition 1: ValueError: Graph contains a cycle; topological sort impossible.
- K=15, topological_sort, repetition 2: NameError: name 'heapify' is not defined
- K=7, topological_sort, repetition 3: ValueError: Graph contains a cycle; topological sort impossible.
- K=9, topological_sort, repetition 3: ValueError: Graph contains a cycle; topological sort impossible.
