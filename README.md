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
| decode, count100 greedy | 57.0 tok/s, 138 ms/cycle | 49.7 tok/s, 158 ms/cycle (-13%) with candidate compaction; acceptance unchanged |
| 250k-token prompt | n/a | 893 s |
| 100k prefix, cold prefill | 335 s | 335 s |
| 100k prefix after eviction or engine restart | 335 s (recompute) | **3-8 s from NVMe**, 99.4% of tokens served |

The serving configuration is a 307,200-token window with a 6 GB/rank pool
and a 150 GB/rank slab store; 512k works but leaves no host memory for the
tier. The DCP cost was +36 ms per verify cycle; a profiler trace showed a third
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
| `tests/` | 150 CPU tests driving the real patched kernels, the NVMe tier and the connector fix |
| `upstream-vllm/` | an upstream clone, used to locate the fork's base commit |

`baseline` is what the image runs: for `flashmla_sparse.py` and
`sparse_attn_indexer.py` that is the `glm-triton` overlay the launcher already
bind-mounts, and for the other eleven the pristine file from the image's
`dist-packages`.

## Tests

```bash
python3 -m venv .venv && .venv/bin/pip install pytest torch triton numpy
PYTHONPATH=tests .venv/bin/python -m pytest tests/ -q     # 150 passed
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
