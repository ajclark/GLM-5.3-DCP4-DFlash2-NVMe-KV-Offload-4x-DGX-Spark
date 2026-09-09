# Next phase: single-stream speed

**User direction, September 9, 2026:** prioritize C1 coding and prose speed;
defer power optimization. The completed power results remain historical evidence.
Energy per token is not a candidate-selection requirement in this phase.
Memory-pressure checks, target correctness, serving capacity and exact rollback
remain requirements. Report coding and prose separately, with decode tok/sec,
time to first token, whole-request latency and streaming gaps.

## Priority and comparison order

1. **Built-in MTP K=1/2/3 versus repaired DFlash.** Begin with K=2, then sweep
   its neighbors if functional and execution checks pass. Compare against both
   fixed-seven and adaptive DFlash in the same pinned target runtime. Keep
   TP4/DCP2, 180224 context, twelve configured sequences, the 6 GB/rank KV pool,
   target weights and sampling settings fixed. Use isolated caches for different
   draft configurations. Changing the verification cap does not eliminate the
   complete DFlash proposal pass; MTP changes that cost as well.
2. **Communication and verification costs.** Profile the winning small target
   shapes and revisit ring-compatible NCCL protocol/channel settings. A smaller
   verified batch changes expert work and collective payloads. Change one
   mechanism at a time. RoCEnante's direct-to-every-peer protocol needs a separate
   topology design for our switchless ring; its published numbers do not justify
   simply enabling its environment switch here.
3. **Current-proposal confidence.** The completed score screen found useful
   same-proposal information, while the tested two-step-old predictor lost to
   acceptance history. A worker-side decision must preserve rank agreement and
   in-flight counts, with dispatch/synchronization costs measured explicitly.
4. **Selector improvement or trees.** Collect candidate coverage versus selector
   error only after the cheaper comparisons. Training and branch-aware target
   verification remain separate architectural changes.

Use development prompts for screening, then freeze a candidate and a separate
held-out coding/prose evaluation. Include complete-function checks and actual Pi
tool workflows, not only truncated generation. Preserve output differences and
the existing target-repeatability issue in reports. Use herdr and the 60-second
state/pane watcher for any Pi invocation. Small screens cannot establish broad
gains, and no new MTP Spark throughput result exists yet.

## Why revisit MTP now

[Light Foundry's September 9 post](https://x.com/light_foundry/status/2097524565916414221)
reports 38.8 tok/sec prose after moving from DFlash2 K7 to built-in MTP K2 on
eight Sparks. Its stack also changes TP collectives, so that headline is not
an isolated draft-length effect or a four-Spark prediction. The relevant code
is [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement)
and [b12x](https://github.com/local-inference-lab/b12x), packaged by
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker).

Our [earlier design](DESIGN.md#6-speculative-decoding-dflash-is-the-design-mtp-was-evaluated-and-dropped)
records an operator decision to omit MTP and contains rough cost estimates.
Those estimates are not a fresh paired MTP K2 benchmark. The normal launcher
still refuses its MTP lane. The new speed direction supports reopening the
comparison in an isolated experiment while retaining the selected deployment.

The inspected RoCEnante source at `75ffee6375b0577ce2c8d6931ffacefda3ecbdd6`
connects every peer on every configured HCA; our documented ring lacks those
non-neighbor connections. The vLLM adapter inspected at
`09aa9ecaa6a943de5ab132bd2a78770f0c54b89d` requires its corresponding runtime API.
Its [four-Spark receipts](https://github.com/local-inference-lab/b12x/blob/75ffee6375b0577ce2c8d6931ffacefda3ecbdd6/docs/rocenante.md)
show useful small-collective latency reductions, but final C1 coding throughput
within noise. A ring transport adaptation needs its own correctness and latency
evidence.

## Local MTP preflight: a reproduced compatibility defect

Read-only inspection confirms that the installed checkpoint has 78 target
layers and one next-token prediction layer. The safetensors headers contain
2343 tensors under `model.layers.78`, totaling 10,044,780,288 stored bytes:

| Component | Stored bytes across the checkpoint |
|---|---:|
| Routed-expert tensors | 9,668,931,584 |
| Hidden/embedding projection | 150,994,944 |
| Remaining MTP tensors | 224,853,760 |

These are checkpoint storage sizes, not per-rank resident or peak memory.
TP-sharded expert weights, replicated projections, temporary embedding/head
allocations, workspaces and graph captures must be accounted for separately.
All four original containers were running during inspection, with approximately
4009/5472/5943/6159 MiB available and zero
memory-pressure averages. Those observations do not authorize an unchecked
additional allocation or establish the future MTP boot peak.

The actual pinned `DeepSeekMultiTokenPredictorLayer.forward` returns two tensors:
pre-final-normalization hidden state for logits and normalized hidden state for
the next draft step. Its wrapper forwards that tuple. However, the captured V2
`MTPSpeculator.model_returns_tuple` returns false. Its autoregressive caller then
treats the tuple as a tensor, and the first sampling-index operation fails.

[Seven CPU tests](../tests/test_spec_mtp_contract.py) execute the captured model
and caller methods with tiny deterministic layers. They reproduce the original
failure, check separate logits/recycling normalization across K=1/2/3, and
preserve the original dispatch declaration for other architectures. Source
fixtures and hashes are [pinned here](../tests/fixtures/spec_mtp/manifest.json).
Read-only checks matched all three source hashes on all four running ranks.
The reproduction does not execute full GLM attention or CUDA graphs.

The [isolated overlay](../experiments/mtp/speculator.py) selects tuple handling
for this pinned `DeepSeekMTPModel` contract. It is not mounted by the ordinary
launcher or the current guarded experiment packager. Before a Spark trial,
the packager must verify the captured model/proposer hashes, mount this file
only for the MTP experiment, turn off the DFlash-specific policy, and capture
the actual target/draft graph shapes. Validate the MTP weight-loading peak and
KV geometry before bounded functional requests. This local fix is a prerequisite,
not evidence that the rest of the MTP/DCP path already works.

Validation: `.venv/bin/pytest -q tests/test_spec_mtp_contract.py` — **7 passed**.
No container was stopped, no inference request was issued, and no runtime
setting was changed during this preflight.
