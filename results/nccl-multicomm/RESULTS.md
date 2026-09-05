# Idea 1 measured: multi-communicator NCCL ring (2026-09-05 15:22-15:30)

Serving stack down; `bench/nccl_multicomm.py` in the serving image with the serving launcher's NCCL
environment, raw NCCL communicators via vLLM's wrapper (rank order chosen per communicator), each
decode-sized collective split into N chunks on N communicators/streams in one NCCL group, timed with
CUDA events around 100 eager iterations, median of 7 repeats, slowest rank. `alt` = every second
communicator uses the reversed ring order [0,3,2,1] (adjacent links only; it worked, zero error).
`ll` = NCCL_PROTO=LL for the whole process. Graph-replay timing was launch-bound in this process
(600-900 us of CPU per replay) and is not reported; the eager CPU issue cost is ~10 us per launch, so
rows with 4+ chunks of the small collectives are launch-bound (see the cpu_issue_us fields in the jsonl).

| eager p50 us | 1 comm | 2 fwd | 2 alt | 4 fwd | 8 fwd | LL 1 | LL 2 | LL 4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| query all-gather (147 KB/rank) | 121 | 109 | 109 | 146 | 200 | 232 | 163 | 144 |
| output reduce-scatter (524 KB in) | 115 | 105 | 109 | 121 | 193 | 202 | 133 | 121 |
| TP all-reduce (96 KB) | 89 | 84 | 80 | 101 | 189 | 88 | 84 | 96 |
| indexer merge all-gather (32 KB/rank) | 64 | 60 | 68 | 80 | 166 | 69 | 66 | 76 |
| LSE all-gather (512 B/rank) | 33 | 42 | 58 | 67 | 165 | 38 | 54 | 100 |

Serving traces for reference: query all-gather ~105 us, reduce-scatter ~99 us, TP all-reduce ~88 us, LSE ~22 us.

## Verdict

- Two communicators buy ~10% on the big collectives (query gather 121 -> 109 us, reduce-scatter 115 -> 105,
  all-reduce 89 -> 84): about 2 ms per verify cycle if it carried into the graph, not the 11-14 ms hoped for.
- Four and eight communicators are slower on the query gather (146, 200 us), which is not launch-bound
  (CPU issue 38-83 us), so the per-hop path does not parallelise across communicators: the host-staged
  proxy path is shared and each extra communicator adds overhead.
- Forcing LL is worse alone (232 us) and only recovers to 144 us with four chunks, still above the default's 121.
- The reversed-ring communicator is legal on this topology (useful for anything bidirectional) but gives no gain.
- NCCL_GRAPH_MIXING_SUPPORT=0 changes nothing for eager launches (126/122/97 us).

Idea 1 is rejected as a major lever. The remaining code-only levers are the ones that remove collectives
(fold the LSE gather into the reduce-scatter; fuse the indexer merge into the query gather) and the glue
fusion; the structural lever is DCP=2 (measured 143.8 ms) or the switch.
