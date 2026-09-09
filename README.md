# GLM-5.3 on 4x DGX Spark: TP4 + DCP4 + DFlash2, with an NVMe-durable KV cache

A patch set for [tonyd2wild's GLM-5.3 Int4-Int8Mix TP4 recipe for 4x DGX Spark](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark), whose vLLM image, sparse-MLA kernels and DFlash2 speculative decoding this work builds on, that keeps **one copy of
the KV cache across the four ranks instead of four**, so the context window
grows with the group instead of being replicated across it, while keeping
DFlash2 speculative decoding. On top of that, a **multi-node NVMe KV tier**
(a fixed-size slab ring buffer on each node's disk) makes long cold prefills
durable across evictions, engine restarts, and a full machine reboot.

## Highlights

- **Decode context parallelism for GLM-5.3's sparse MLA, with speculative decoding kept.** One KV cache shared across the four Sparks instead of four copies: 4.00x the KV tokens at the same window (131k → 524k), a 262k window with 462k tokens, a 500k-token prompt served. DFlash2 (K=7) runs alongside it through a replicated drafter group; acceptance is unchanged and greedy output is byte-identical to the stock lane.
- **NVMe-durable KV cache, across a reboot.** A multi-node slab tier on each node's disk: a 100k-token prefix reloads in ~2.5 s instead of a ~217 s recompute, and survives evictions, engine restarts, and a full cold reboot of all four nodes (measured: 9,640 blocks recovered from on-disk headers, 99.7% served). Each slot is self-describing (magic, token-hash key, epoch, length, payload CRC) and sealed header-last, so a torn or reordered write is detected on read and never served; recovery is a header scan, gated on the run config plus a content identity (weight fingerprints, dtype/quantization/RoPE, the overlay digest), since the key names the input tokens, not the KV bytes; toggleable (`persist_across_reboot`, default on). Fixed-size ring buffer, no janitor needed.
- **Idle power, in a sibling repo.** Switching the ConnectX-7 off with the cables attached takes four idle nodes from 202 W to 120 W: [dgx-spark-idle-power](https://github.com/ajclark/dgx-spark-idle-power).

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
| 100k prefix after eviction, engine restart, or **a full reboot** | 335 s (recompute) | **~2.5 s from NVMe**, 99.7% of tokens served |

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

## Adaptive C1 verification experiment (2026-09-09)

An opt-in controller chooses target verification caps 1/3/5/7 while retaining
DFlash2's trained block of eight and seven draft tokens. On the current
TP4/DCP2 lane, the held-out prose benchmark improved 15.8%; the expanded
coding follow-up measured +1.14% with a 95% interval of -1.30% to +3.64%.
NVIDIA-device energy per prose token fell 17.4%. Whole-system energy remains
unmeasured in this experiment, and CX-7 cycling remains paused.

`GLM_SPEC_POLICY` defaults to `off`. The experiment preserves the 180224-token
window, 12 sequences and 6 GB/rank KV allocation. See the
[benchmark report and reproducible commands](results/adaptive-spec/README.md),
[completion audit](results/adaptive-spec/COMPLETION-AUDIT.md), and
[design](docs/ADAPTIVE-SPECULATION-PLAN.md). Broad promotion remains gated.

The [follow-up experiments](docs/ADAPTIVE-SPECULATION-NEXT.md) repair a replicated
draft-cache table defect, validate real Pi text/tool hints through herdr, reject
the tested lagged-confidence predictor, and measure active/idle GPU-clock
tradeoffs. Loaded-idle device power falls from about 32 W across four GPUs to
22.14 W at 600 MHz, with no permanent clock policy installed. Target-output
repeatability remains unresolved after an isolated atomic-reduction control.
The [curated evidence](results/adaptive-next/README.md) preserves all outcomes,
invalid controls, privacy transformations and exact restoration checks.

## Layout

| path | what it is |
|---|---|
| `docs/DESIGN.md` | the design, the cost analysis, and the validation plan |
| `baseline/vllm/…` | sixteen files exactly as the running image has them |
| `overlay/vllm/…` | the same sixteen files patched (thirteen for DCP: target sharded, DFlash drafter replicated, top-k candidates compacted per rank; the engine scheduler's invalid-block recovery; the offloading connector's store progress; the b12x attention helper's candidate-count passthrough) plus the new `v1/kv_offload/tiering/multinode.py` NVMe tier |
| `patches/*.patch` | `baseline` to `overlay` diffs, plus `apply.sh` |
| `stage/glm-dcp/` | deployed sources flattened for bind-mounting, with `SHA256SUMS` |
| `launch-glm53big-dcp.sh` | TP4 + DCP4 + DFlash launcher, derived from the selected one |
| `tests/` | Local tests of real patched kernels, NVMe tier, adaptive verification, API controls and guarded experiments; validation counts are recorded with each experiment |
| `upstream-vllm/` | an upstream clone, used to locate the fork's base commit |

`baseline` is what the image runs: for `flashmla_sparse.py` and
`sparse_attn_indexer.py` that is the `glm-triton` overlay the launcher already
bind-mounts, and for the other eleven the pristine file from the image's
`dist-packages`.

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install pytest torch triton numpy pydantic==2.13.5
PYTHONPATH=tests .venv/bin/python -m pytest tests/ -q
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

## Incident write-up

`docs/INCIDENT-SHIELD-REMOTE.md`: a Bluetooth NVIDIA SHIELD Remote that had once
been paired with two of the Sparks powered them off from another room (its power
key is a HID power key; logind's default for any power key is poweroff). Evidence,
mechanism, the GID-index fragility it exposed on relaunch, and fixes.

## Credits

Everything here starts from
[tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark):
the vLLM image these sixteen files are overlaid on, the sm12x sparse-MLA
kernels, the DFlash2 drafter port and the launcher that made GLM-5.3 run on
four Sparks in the first place. This repo adds decode context parallelism and
the NVMe tier on top of that recipe; the "production" lane in every table above
is that recipe unmodified.
