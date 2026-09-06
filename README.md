# GLM-5.3 on 4x DGX Spark: TP4 + DCP4 + DFlash2, with an NVMe-durable KV cache

A patch set for the community DGX Spark vLLM image that keeps **one copy of
the KV cache across the four ranks instead of four**, so the context window
grows with the group instead of being replicated across it, while keeping
DFlash2 speculative decoding. On top of that, a **multi-node NVMe KV tier**
(a fixed-size slab ring buffer on each node's disk) makes long cold prefills
durable across evictions and engine restarts.

Deployed and serving on the author's cluster since 2026-09-04 (GLM-5.3
Int4-Int8Mix, TP4 over a switchless RoCE ring). Measured against the
production DCP1 lane of the same image:

| | production (DCP1) | this work (DCP4 + DFlash2 K=7) |
|---|---|---|
| KV pool at a 131k window, 8 GB/rank | 131k tokens | 524,288 tokens (4.00x) |
| KV pool at a 262k window, 7 GB/rank | n/a | 462,308 tokens (1.76x) |
| largest window booted | 120k | 524,288 (served a 500k-token prompt, no headroom left) |
| decode, count100 greedy (GPU clocks locked at 2000 MHz) | 56.5 tok/s, 139 ms/cycle | 50.5 tok/s, 155 ms/cycle (-11%) with candidate compaction; acceptance unchanged |
| 250k-token prompt | n/a | 893 s |
| 100k prefix, cold prefill | 335 s | 335 s |
| 100k prefix after eviction or engine restart | 335 s (recompute) | **3-8 s from NVMe**, 99.4% of tokens served |

Decode by lane, single stream, greedy, thinking off, GPU clocks locked at
2000 MHz, DFlash2 K=7 (`docs/DESIGN.md` §8; DCP=4 with candidate compaction):

| lane | count100 | prose | code | verify cycle | aggregate at C=12 | KV tokens at 6 GB/rank |
|---|---|---|---|---|---|---|
| DCP=1 (production launcher) | 56.5 tok/s | 19.6 | 48.0 | 139 ms | 245.7 tok/s | 99k |
| DCP=2 (pairs on adjacent ring links) | 54.5 | 18.1 | 41.5 | 144 ms | 232.7 | ~198k |
| DCP=4 (serving) | 50.5 | 17.5 | 38.6 | 155 ms | 198.8 | 396k |

Accepted tokens per cycle are the same across lanes (7.87 of 8 on count100,
~2.7 on prose, ~6.5 on code); count100 output is byte-identical. Prose and
code vary run to run at greedy on every lane, so treat those two columns as
±5%.

Aggregate decode throughput under concurrency (count-to-400 prompts, C
simultaneous streams, the engine's own cycle metrics; `results/concurrency-sweeps.md`):

| C | DCP=1 | DCP=2 | DCP=4 | DCP=2 vs 1 | DCP=4 vs 1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 54.0 tok/s | 49.9 | 46.9 | -8% | -13% |
| 2 | 82.6 | 77.3 | 71.3 | -6% | -14% |
| 4 | 136.4 | 125.3 | 114.6 | -8% | -16% |
| 8 | 192.1 | 187.7 | 167.2 | -2% | -13% |
| 12 | 245.7 | 232.7 | 198.8 | -5% | -19% |

Single-stream, DCP costs a fixed per-cycle collective floor. Under load the
per-step payloads grow (the query gather carries ~7 MB at 96 tokens per
step), so DCP=4's three-hop ring collectives become bandwidth-bound: its
marginal cost stays ~2.9 ms per token while DCP=1 and DCP=2 fall to
1.9-2.1 ms, and its penalty widens from 11% to 19%. DCP=2, whose DCP groups
sit on adjacent ring links, stays within 5-8% of DCP=1 at every concurrency
with twice its KV. For multi-session use DCP=2 is the better lane unless one
session needs more than its 180k window; DCP=4 is the lane for the largest
single contexts.

The serving default is the DCP=2 lane: a 180,224-token window with a
6 GB/rank pool (~198k KV tokens, two copies) and a 150 GB/rank slab store,
within 5-8% of production's decode speed at every concurrency. The DCP=4
lane (307,200 window, 396k tokens, one copy) is one launcher env away
(`DCP_SIZE=4 MAXLEN=307200`) for the largest single contexts; 512k works at
DCP=4 but leaves no host memory for the tier. The DCP cost was +36 ms per verify cycle; a profiler trace showed a third
of it was the sparse attention kernel walking masked candidates, which
compaction removed, leaving ~13 ms of ring collectives (`docs/DESIGN.md`
§8). The drafter is untouched, so acceptance is too.

What is in the patches, briefly: the sparse-MLA indexer and attention
backend learn to work on a sharded KV cache (top-k merged across ranks,
local-length workspaces, chunk metadata that does not recompile per
prompt); the DFlash drafter's sliding-window KV group stays replicated
under DCP through one helper that decides per group; the engine
scheduler's invalid-block recovery and the offloading connector's store
progress get two bug fixes; and a new module, `multinode.py`, implements the
NVMe tier, because vLLM's own tiering assumes every rank shares one host.
`docs/DESIGN.md` is the design and the numbers, `docs/NVME-DESIGN.md` the
tier, `docs/HANDOVER.md` the operating notes. Upstream vLLM has since
gained DCP for sparse MLA on newer code; these patches are for the June
2026 base the Spark images pin.

## Layout

| path | what it is |
|---|---|
| `docs/DESIGN.md` | the design, the cost analysis, and the validation plan |
| `baseline/vllm/…` | sixteen files exactly as the running image has them |
| `overlay/vllm/…` | the same sixteen files patched (thirteen for DCP: target sharded, DFlash drafter replicated, top-k candidates compacted per rank; the engine scheduler's invalid-block recovery; the offloading connector's store progress; the b12x attention helper's candidate-count passthrough) plus the new `v1/kv_offload/tiering/multinode.py` NVMe tier |
| `patches/*.patch` | `baseline` to `overlay` diffs, plus `apply.sh` |
| `stage/glm-dcp/` | the seventeen files flattened for bind-mounting, with `SHA256SUMS` |
| `launch-glm53big-dcp.sh` | TP4 + DCP4 + DFlash launcher, derived from the selected one |
| `tests/` | 156 CPU tests driving the real patched kernels, the NVMe tier and the connector fix |
| `upstream-vllm/` | an upstream clone, used to locate the fork's base commit |

`baseline` is what the image runs: for `flashmla_sparse.py` and
`sparse_attn_indexer.py` that is the `glm-triton` overlay the launcher already
bind-mounts, and for the other eleven the pristine file from the image's
`dist-packages`.

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install pytest torch triton numpy
PYTHONPATH=tests .venv/bin/python -m pytest tests/ -q     # 156 passed
```

They run Triton in interpreter mode on CPU and extract the kernels straight out
of `overlay/` by AST, so they cannot drift from the shipped source. The NVMe
tier tests import the fork's own connector modules from a source tree at
`~/lmcache-mg/spark-src/vllm` (the image's vLLM at commit ab666069); point
`SRC` in `tests/nvme_harness.py` at any checkout of that commit.

## Deploying

The image is not rebuilt: the sixteen files in `stage/glm-dcp/` are
bind-mounted over the installed vLLM by `launch-glm53big-dcp.sh`, which
preflights every file and refuses to start otherwise. `docs/HANDOVER.md` has
the state, the rules learned the hard way, and the recovery paths.

```bash
./rollout_dcp.sh <label> [MAXLEN] [MAXBATCHED] [KVBYTES] [KVTIER]   # stage, verify, launch, watchdog, auto-restore
./post_boot_checks.sh <label> results/baseline-dcp1-prod [longctx_tokens]
./deploy_slab.sh <label>            # NVMe slab tier: small cap, real cap, restart (eviction + durability)
./deploy_slab_fix.sh <label>        # first-time-store proof: no-warm probe, restart, reloads
./restore_production.sh             # back to the production launcher
```

Keep `~/glm-triton/` in place: eight of its ten overlays are still mounted from
there, and the DFlash draft weights are mounted exactly as the production
launcher does. Hostnames, paths and the image tag are the author's; they are
variables at the top of the launcher and the scripts.

## The GitHub issue

The task started from
[vllm-project/vllm#54907](https://github.com/vllm-project/vllm/issues/54907).
That bug is in the fused NVIDIA DeepSeek-V3.2 norm/RoPE kernel, in a directory
this fork does not have, so its fix (#54908) is a no-op here. The fork's actual
gap is that the sparse indexer and the sparse attention backend have no DCP
support at all. `docs/DESIGN.md` section 2 has the details.

## Idle power

`spark-idle.sh --down` powers the ConnectX-7 off on every node with the cables
attached (202 -> 120 W measured for the four idle nodes); `--up` powers it back
on and verifies the ring. One script, mirrored from
[ajclark/dgx-spark-idle-power](https://github.com/ajclark/dgx-spark-idle-power).
Stop the serving stack before `--down`, relaunch after `--up` with
`SKIP_PREFLIGHT=1 ./rollout_dcp.sh <label>`. Background: `docs/CX7-POWER.md`,
`docs/IDLE-POWER.md`.
