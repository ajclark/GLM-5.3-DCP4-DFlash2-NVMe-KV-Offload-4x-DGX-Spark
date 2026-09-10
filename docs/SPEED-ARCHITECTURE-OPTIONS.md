# GLM-5.3 on four Sparks: architecture options for C1 speed

**2026-09-09.** Written after reading the adaptive-speculation work of 2026-09-08/09
(`ADAPTIVE-SPECULATION-*.md`, `research/*.md`, `results/adaptive-*`,
`results/pi-humaneval-compare-20260909`) and the earlier cycle anatomy
(`~/spark-cluster-experiments/PLAN-40-TOKS.md`, `GLM-NONDISRUPTIVE-MOE-AUDIT.md`,
`DESIGN.md` §8, `IDEAS-DCP-LATENCY.md`). Nothing here was run on the cluster; every
gain below is an estimate with its arithmetic shown, and every external number is
someone else's hardware unless marked otherwise.

## TL;DR

1. Adaptive verification is done and the answer is in: **+16% prose, flat code**
   (180 paired requests; the HumanEval-through-pi comparison was flat at C1).
   The 35-42% long-context gains were an artifact of a real block-table defect
   that is now fixed but not yet deployed. Stop investing in the controller.
2. The cycle is **~46% expert-weight streaming, ~15% dense int8 streaming,
   ~9% TP all-reduce latency, ~6% DCP collectives, ~8% draft, ~16% glue** at
   ~144 ms per DCP2 cycle. Nothing in that list can be made faster by a
   scheduler; each needs a format, kernel, or transport change.
3. Code speed is a **multiplier** problem (4.0-6.1 accepted of 8 on agentic
   traffic vs 7.9 on `count100`). The only lever that moves the code multiplier
   is a draft trained on this traffic. Prose speed is a **cycle-per-accepted-
   token** problem (2.7 of 8): bounded-lossy verification or an MTP lane.
4. Ranked by expected gain per week of work: (A) bounded-lossy verification for
   prose, opt-in per request (+15-30%, the cheapest lever); (B) MTP K=2 lane for
   prose (+25-30% vs fixed K7, but 2.4 GB/rank and a boot per lane; codex's #1,
   prerequisite defect already fixed locally); (C) 3-bit trellis experts, which already run
   the full GLM-5.3 on Sparks under vLLM in the community; (D) dense/indexer
   bytes to 4-bit or NVFP4; (E) ring-native one-shot collectives (the RoCEnante
   idea, adapted to a switchless ring); (F) a workload-trained DFlash2 draft.
5. The RoCEnante-class transport is worth ~8% on this ring (arithmetic in §E),
   which matches codex's "single digits to 10% stretch". It is real but not the
   largest lever, and it needs a forwarding design the published code lacks.

## 1. Where the time goes

Measured cycle, DCP1 at 2418 MHz (`PLAN-40-TOKS.md`, 2026-09-01), K=7 verify of
8 tokens; DCP2 adds ~5 ms of one-hop collectives and the 2000 MHz lock ~1-2%:

| bucket | ms | share | what bounds it |
|---|---:|---:|---|
| routed experts (Marlin W4A16, ~57 distinct of 256 per layer x 75 layers) | 64.3 | 45% | ~15 GB/rank/cycle at the measured 240 GB/s |
| dense int8 (attention projections, shared expert, lm_head; ~4.6 GB/rank) | 20.8 | 15% | same bandwidth, 92% of it |
| TP all-reduce (167 x ~80 us, 96 KB) | 13.1 | 9% | ring latency, 6 hops under NCCL LL |
| sparse MLA + indexer (DCP1) | 12.5 | 9% | mostly per-layer latency |
| DFlash2 draft pass | 10.9 | 8% | 6 dense bf16 layers, replicated per rank |
| bf16 CUTLASS (indexer projections etc., SM80 WMMA kernels) | 8.7 | 6% | ~1.4 GB/rank of bf16 weights + kernel choice |
| gaps, launches, top-k, rejection | 11.7 | 8% | CPU/launch, ~15 us per collective |
| **verify cycle** | **142** | | |

At DCP2 (daily lane): +5.7 ms all-gathers, +3.2 ms reduce-scatters, -6 ms
attention (sharded) net ~+5 ms: **143.8 ms** measured (`DESIGN.md` §8).

The multiplier, accepted tokens per 8-token cycle:

| workload | accepted/cycle | decode tok/s (DCP1 / DCP2) |
|---|---:|---:|
| `count100` (structured) | 7.87 | 56.5 / 54.5 |
| code benchmark prompt | 6.5-6.7 | 48.0 / 41.5 |
| opencode, terse prompt, effort high, T=0 | 6.1 | 41.6 pooled |
| pi coding suite, T=0 | 4.55 | ~31 |
| pi coding suite, T=1.0, thinking max | 4.01 | 27.6 |
| prose | 2.6-2.7 | 19.6 / 18.1 |
| prose, adaptive K3 (codex, +15.8%) | ~2.2 of 4 | ~17.8-18.6 at DCP2 |

`tok/s = accepted / cycle`. Code is bounded by the multiplier (6.1 -> 7.3 is
the whole distance to 50 tok/s); prose is bounded by paying a 144 ms cycle
for 2.7 tokens.

## 2. What the last two days settled (do not redo)

| question | answer | evidence |
|---|---|---|
| Does adaptive K help? | Prose +15.8% (CI 13.2-18.7), code +1.0% (CI -2.1-4.4); HumanEval-via-pi C1 -1.2% (one batch) | `results/adaptive-spec/README.md`, `pi-humaneval-compare-20260909/CONCLUSIONS.md` |
| Long-context acceptance collapse at 100k/170k? | A replicated-draft block-table sizing bug (1408 vs 2816 columns); fixed in `8f684a4`; acceptance at 100k is now 0.65-0.74 first-position, 16-17 tok/s | `research/LONG-CONTEXT-DIAGNOSTIC.md` |
| Selector confidence as a predictor? | Same-proposal scores are informative (Brier 0.12 vs 0.19), but they arrive two proposals late; the lagged predictor adds nothing | `research/CONFIDENCE-FEASIBILITY.md` |
| Pi workload hints? | No measurable gain, one API-boundary bug fixed | `research/PI-HINTS.md` |
| True K=0? | Deferred: no low-acceptance regime left after the cache repair | `research/K0-FEASIBILITY.md` |
| Exact-copy / n-gram reuse? | Sparse opportunities on every screened context | `research/ARCHITECTURE-SCREENS.md` |
| Trees, longer blocks, selector training? | CPU screens only; each has an evidence gate, none met | same |
| Clocks | 1600 MHz: -5% tok/s, -18% J/token; no permanent change | `research/POWER-EXPERIMENT.md` |
| MTP under this fork | The pinned MTP caller mishandles the layer's tuple return; fixed in `experiments/mtp/speculator.py`, 7 tests; not yet booted | `SPEED-NEXT.md` |
| NCCL multi-communicator, LL, reversed ring | <=2 ms/cycle; rejected | `results/nccl-multicomm/RESULTS.md` |

**Deploy the cache repair** regardless of anything else: the running containers
still clamp draft positions beyond 90,111 tokens, so any session past 90k is
decoding at ~6.5 tok/s instead of ~16.

## 3. The levers

Gains are per-lane single-stream estimates on the DCP2 daily lane unless
stated. "Weeks" is engineering time before a Spark trial, not calendar time.

| # | lever | targets | est. gain | cost | risk | first gate |
|---|---|---|---:|---|---|---|
| A | bounded-lossy verification (MARS margin rule, greedy only), opt-in per request | prose | +15-30% prose (MARS: +13/+24/+28% accepted length on 70B/235B/13B; more if our prose has more near-ties) | 1 week (Triton rejection kernel + xargs + tests) | quality, must be gated and opt-in; sampled-mode variant gains ~1% | blind prose A/B + code stays exact |
| B | MTP K=2 lane | prose | +25-30% prose vs fixed K7 (+10-15% vs adaptive); code -30% | codex's overlay + a boot | memory: MTP layer ~2.4 GB/rank | boot, count100, prose/code paired |
| C | 3-bit trellis experts (EXL3 or GLQ) | cycle | -12 to -16 ms (8-11%) | 2-4 weeks (requant 743B, plugin on this fork, DCP compatibility) | quality at 3 bpw; prefill kernel maturity | perplexity/KL vs current int4; then cycle |
| D | dense + indexer bytes: int8 -> 4-bit/NVFP4, bf16 indexer -> int8 | cycle | -8 to -13 ms (6-9%) | 1-2 weeks (targeted requant, b12x NVFP4 dense GEMM or Marlin W4) | quality of attention projections at 4-bit | per-tensor KL; boot |
| E | ring-native one-shot collectives (RoCEnante with 2-hop forwarding) | cycle | -10 to -13 ms (7-9%) | 2-3 weeks (verbs proxy, forwarding, graph-safe flags, DCP AG/RS) | correctness under graphs; NIC contention with NCCL prefill | microbench AR/AG/RS <= 30/15/15 us |
| F | workload-trained DFlash2 draft | code multiplier | +20-30% code (6.1 -> 7.3 accepted) | days + $1-3k GPU rental, plus data pipeline | licence: new checkpoint, not a fine-tune of Inco's | offline accepted length on held-out pi traffic |
| G | prefill path (NVFP4/b12x fused MoE or dequant+GEMM at large M) | TTFT, task wall | 1.5-2.5x uncached prefill; 10-20% of agentic task wall | 1-2 weeks after a prefill profile | none for decode | profile one 2048-token chunk |
| H | small cycle items: drafter fp8, fp8 AR payload, LSE fold, glue fusion, striped links for prefill | cycle | -6 to -10 ms total | days each | low | each in isolation |
| I | 200G switch, or TP8 | cycle | switch: -12 to -18 ms; TP8: -40% cycle | money | switch: PFC/ECN tuning | |

Compounding (independent to first order): C+D+E on the cycle is
144 -> ~105-110 ms (+30-35% at any acceptance); A or B on prose is a separate
multiplier; F is the code multiplier. Code at 6.1 accepted / 108 ms = 56 tok/s,
prose at ~3.3 accepted (A) / 108 ms = 30 tok/s (or ~2.3 of 3 / ~68 ms with B's
MTP lane on the same cycle savings, ~34). Those are the ceilings this memo can
justify without new hardware.

### A. Bounded-lossy verification for prose (opt-in)

The prose drafter is not bad; prose is high-entropy. When the target's top two
tokens are near-equiprobable, rejecting the draft's choice and resampling buys
nothing measurable in quality and costs a whole cycle. The literature converged
on this in 2026:

- **MARS** (Jan 2026, arXiv 2601.15498): training-free; relax rejection only where
  the target's decision margin is small ("rejecting plausible runner-up tokens
  yields negligible information gain while incurring substantial rollback
  cost"); 8B-235B targets, "preserves generation quality" on their benchmarks.
- **Revisiting Lossy Verification** (Jul 2026, arXiv 2607.26627): the survey that
  matters. Block efficiency 5.5 -> 5.6-6.2 for truncation rules (typical
  acceptance, min-p) but 5.5 -> up to 10.5 for collaborative rules; the failure
  mode is *draft overshoot* (accepting where draft q >> target p). Their ablation:
  an overshoot ceiling alone keeps task accuracy at the lossless level, **but
  the quality-preserving variant gains only ~1% block efficiency** (Table 1:
  BE 5.54-5.59 vs ~5.5), while the fast variants collapse MBPP+ from 75 to
  50-66; plain typical acceptance loses 6-11 points on hard math and degrades
  under tree verification. Judge-style rules depend entirely on supervision
  quality (math-heavy judge 83% vs code-heavy 64% on MATH). Net: the sampled-
  request lever is effectively dead; the greedy margin rule is the credible one.
- Judge Decoding (2501.19309), SelfJudge (2510.02329), DIVERSED (2604.07622):
  learned or ensemble judges; more upside, need training data. Not first.

Proposed rule (greedy, which is what pi/opencode run at T=0): accept draft token
d at a position if it is the target's runner-up and `z_argmax - z_d <= M` nats
(optionally `p_target(d) >= p_min`), else the exact rule fires. Sampled requests
keep the lossless rule in v1 (see above). The concrete plan, with the kernel
anchors (acceptance is decided in the fork's V2 `rejection_sampler_utils.py`,
on identical all-gathered logits on every TP rank), tests, gates and sweep, is
in `LOSSY-VERIFICATION-PLAN.md`.

Why it is a good fit here: the verifier already materialises target logits for
all 8 positions; the rejection kernel is a small Triton function in the fork
(`dflash2/speculator.py`, Gumbel-argmax); the change is local, graph-safe, and
adds no bytes. Make it opt-in via `vllm_xargs` so code paths stay exact, and
let pi/opencode set it for prose personas only. Quality gate: codex's blind
prose-opening protocol plus a 30-prompt held-out prose set scored by a judge
model, and the existing complete-function checks to prove code is unchanged.
Expected, from MARS's published +13-28% accepted length: prose acceptance
2.7 -> ~3.1-3.5 of 8, i.e. **prose 18 -> 21-23.5 tok/s** at today's cycle,
with upside only if our prose has more near-tie positions than their chat
benchmarks (baseline 4.1-5.7 of 8 there vs 2.7 here). The plan's promotion
bar is a +15% paired lower bound. The `PLAN-40-TOKS.md` ledger already ranked this "highest
upside per line of code"; the 2026 evidence now says how to do it safely.

### B. MTP K=2 as a prose lane (codex's current #1)

A cycle model for K=2 MTP at DCP2, from the buckets in §1: verify 3 tokens
touches ~22 distinct experts (vs 57) so MoE 64 -> ~25 ms; dense 21; TP AR
~9 (smaller payload); DCP collectives ~9; attention ~8; bf16/glue ~12; two
MTP draft steps (one MoE layer, lm_head shard, two ARs each) ~5 ms: **~90-95 ms**.
MTP's chat/prose acceptance from Inco's table (3.81 of 7 vs DFlash2's 4.19)
gives ~2.3 accepted of 3 at K=2, so **~24-26 tok/s prose vs 18.1 today** and
vs ~18.6 with adaptive DFlash. On code the same model gives ~27 tok/s
(vs 41.5): MTP is a prose lane, not a replacement. Light Foundry's 38.8 prose
on eight Sparks is consistent with this model at half the per-rank bytes.

Costs: the MTP layer is a full MoE layer (10.0 GB stored, ~2.4 GB/rank resident
at TP4 int4) and there is no headroom on rank 0 next to a 6 GB pool plus the
NVMe tier; so either a **prose boot** (MTP, no DFlash, +2.4 GB from the pool)
or both resident with the pool cut to ~3.6 GB (~120k tokens at DCP2). Per-request
drafter routing with both resident is the ideal but is a scheduler/runner change
on top; the boot-per-lane version is a launcher flag once the contract fix works.
Note A and B target the same tokens; A is cheaper and keeps DFlash's code speed.

### C. Three-bit trellis experts

Expert bytes are 45% of the cycle. Int4 -> 3 bpw is -25% of those bytes: MoE
64 -> ~48 ms (**-16 ms, -11%**) with no change to acceptance. This is no longer
exotic on Sparks:

- `drowzeys/keys-GLM-5.3-EXL3`: **the full 743B GLM-5.3 with 3.00 bpw EXL3 routed
  experts, attention in bf16, 308 GB**, served by vLLM on four Sparks at TP4+DCP4
  (Mia AI Lab container; 75 GiB/rank; MTP k=3). Their 11-15 tok/s is with bf16
  attention and MTP, not comparable to ours, but it proves the format loads and
  the DCP path exists in someone's fork.
- `vcruz305/...EXL3-MixedK-DGX-Spark-recipe`: the `vllm-exl3` v0.3.1 plugin has
  "native sm_121 fused MoE and Super Fat GEMM prefill kernels", FULL_DECODE_ONLY
  CUDA graphs validated, TP=1 only documented; prefill 280-360 tok/s (slow).
- `cnygaard/glq`: trellis/lattice 2-8 bpw with fused grouped-trellis MoE decode
  under full CUDA graphs, a vLLM plugin (`--quantization glq`), validated on
  sm_120; B=1 decode within 2% of bf16 on their test; quantizing a 3B model
  takes ~35 min on one GPU (Viterbi encode), so 743B is days of GPU time.
- Turboderp's EXL3 at 2.05 bpw + DFlash2 K7 runs GLM-5.3-Flash on one Spark at
  64 tok/s structured / 25 prose with KLD 0.12 (forum #382140): 2 bpw is too
  lossy for this use, 3 bpw is the interesting point.
- MoQE/MxMoE-style results: expert FFNs tolerate 2-3 bit far better than
  attention; keep attention >= int8 (or NVFP4, see D).

Work: requantise the 256x75 expert tensors (outside the cluster, or on the
Sparks during a maintenance window), integrate the kernel as a plugin on this
fork (the routed path is `MarlinExperts.apply` via `select_wna16_moe_backend`,
a hardcoded priority list with no override; `GLM-NONDISRUPTIVE-MOE-AUDIT.md` §4-5),
keep our DCP overlays. Gate: token-level KL/top-1 agreement against the current
int4 on a fixed corpus (Turboderp publishes 0.12 KLD / 89% top-1 at 2 bpw;
demand < 0.03 / > 96% at 3 bpw), then acceptance must not drop (the draft was
trained on a higher-precision target; a small drop is possible), then the cycle.
Prefill for these kernels is immature: pair with G or accept slower TTFT.

### D. Dense and indexer bytes

**Correction (2026-09-10, `research/DENSE-BYTES-PLAN.md`, from the real
safetensors headers):** only 21 of 78 layers instantiate a DSA indexer
(`index_topk_freq=4`, the others reuse indices), so the bf16 indexer streams
1.64 ms/cycle, not ~5.7; `lm_head` is bf16 (1.98 ms) and was wrongly counted
in the int8 family; the int8 dense family is 19.7 ms; the derived bf16
`W_UK_T`/`W_UV` add 2.4 ms and the bf16 MoE routers 1.0 ms. The
immediately implementable indexer change (21 replicated `wq_b` to W8A16)
saves 0.73 ms; a drafter fp8 recipe that keeps QKV bf16 saves ~2.7 ms.
D-lite is therefore ~3.4 ms (2.3%), below H and B in priority; the 4-bit
dense projections (o_proj, q_b, kv_b/W_UK, shared expert, dense MLPs)
remain the larger part of this lever. Offline tooling exists
(`bench/repack_dense_int8.py`, `bench/dense_bytes_inventory.py`).

The 4.6 GB/rank of int8 dense weights (attention projections, shared expert,
lm_head) streams at 92% of bandwidth: **20.8 ms**. The quant recipe we run
(`Int4-Int8Mix`, tclf90/Tech2wild) is a generic "experts int4, everything else
int8" recipe, not a sensitivity study. Options:

- 4-bit (AWQ W4A16 g128 or NVFP4) for the large, robust projections only:
  `o_proj` (50M/layer), `q_b_proj` (19M), `kv_b_proj`/W_UK (17M), shared expert
  (38M), the three dense-layer MLPs and `lm_head`. That is ~80% of the dense
  bytes: **-8 to -9 ms**. Keep `q_a`/`kv_a` (the latent projections) at int8.
  The howtospark GLM-5.2 recipe did exactly this split (NVFP4 on o/q_b/kv_b
  only) and reports GSM8K-50 0.92 held, dropping to 0.81 only when the shared
  expert was included: use that as the warning.
- Kernels exist today: Marlin W4A16 (already used for experts) or b12x's
  sm121 NVFP4 dense GEMM. NVFP4 also matters for G.
- The **bf16 leftovers**: the compressed-tensors ignore list keeps layer 0, the
  MoE gate, the DSA indexer projections (`wq_b` 2048x4096, `wk`, `weights_proj`)
  and MTP tensors in bf16, run through SM80 WMMA cuBLAS kernels: 8.7 ms. The
  indexer weights are ~1.4 GB/rank/cycle; RTN int8 on them is lossless in
  practice and halves that: **-2.5 to -3 ms**. This is a checkpoint edit plus
  removing three ignore patterns, no new kernel.
- The DFlash2 drafter: 6 bf16 layers replicated on every rank (draft TP1 was
  measured faster than TP4). fp8/int8 weights: **-3 to -5 ms** (`PLAN-40` said
  "<= 4 ms ideal"). Inco's licence is CC BY-NC-ND: a private quantised copy for
  our own serving is not distributed, but do not publish it.

### E. Ring-native one-shot collectives (the RoCEnante idea, adapted)

What RoCEnante actually is (b12x `docs/rocenante.md`, inspected at
`75ffee6`): a **host-proxy libibverbs** runtime, not GPU-initiated RDMA. Pinned
unified-memory receive slots per peer, a C proxy thread that posts data plus a
4-byte sequence flag on reliable QPs striped over the two PCIe functions of
each port, GPU kernels that stage input and spin on peer flags, reduction in
fixed rank order (bit-identical). Only all-reduce and all-gather; sizes
<= 16 MB routed to it, larger stays on NCCL. It works without GDR, which is why
it works on GB10 at all (our NCCL sweep confirmed `GDR 0`, no dma-buf). Their
four-Spark numbers, **through a switch, every peer directly reachable**:

| collective | NCCL | RoCEnante in graph |
|---|---:|---:|
| all-reduce 8 KB | 52.6 us | 16.8 us |
| all-reduce 48 KB (6-token decode) | 65.5 us | 23.6 us |
| all-reduce 256 KB | 174 us | 58 us |
| all-gather 6x38720 | 337 us | 97 us |

End-to-end on GLM-5.3-Flash TP4: per-step decode 62-64 vs 65-66 ms (3-5%)
in the final A/B, +12.7% at c=1 coding in an earlier one, +15-32% at c=16.

On our ring the one-shot pattern does not exist: rank r has no QP to rank r+2.
The adaptation is a **two-hop forwarding proxy**: each rank writes its chunk to
both neighbours; each neighbour's proxy forwards the chunk it received from one
side to the other side; the opposite rank receives one copy from each direction
(or half from each). Latency ~2 x (one-hop write ~10-12 us) + reduce ~5 us:
~30 us for the 96 KB AR vs 78-84 us today; DCP2's one-hop AG/RS become ~12-15 us
vs 36-43 us. Per cycle: 167 x ~50 us + 154 x ~25 us = **-12 ms (-8%)**, plus a
share of the ~15 us launch gap per collective inside graphs (flags, not launches).
That is codex's "single digits to 10% stretch", now with the arithmetic.

Engineering: a new communicator class behind `CudaCommunicator.all_reduce` /
`all_gather` (the same hook RoCEnante's vLLM adapter uses; the fork's
`custom_all_reduce.py` and `symm_mem.py` are single-node IPC paths and do not
apply), a verbs proxy with forwarding and both-direction striping, CUDA-graph
capture of the stage/wait kernels, and a size cap so prefill stays on NCCL.
Also DCP's reduce-scatter (76/pass) needs a reduce-scatter primitive RoCEnante
lacks (or express it as AG + local slice, which doubles bytes but bytes are not
the floor). Risk: two RDMA stacks on the same ports; NCCL prefill traffic
contending with the proxy's flags; correctness of forwarded sequence numbers
under cancellation. Gate first with a standalone microbenchmark: AR 96 KB
<= 30 us, AG 147 KB <= 15 us, RS 524 KB <= 20 us, all bit-identical, all in a
captured graph. If the microbench misses those by 2x, stop.

Alternative with less code: a **200G switch** makes the published RoCEnante
usable as-is, lets the fork's `a2a` DCP merge replace two of the four
per-layer DCP collectives, and unlocks NCCL tree/one-shot. The PLAN-40 note
rejected a 100G MikroTik on prefill-bandwidth grounds; a 200G-port Spectrum
(SN3700-class) does not have that problem but costs real money and needs
lossless PFC/ECN.

### F. A workload-trained DFlash2 draft (the code multiplier)

**Declined by the operator on 2026-09-10; kept for the record.**

This is the only lever that moves code from 6.1 accepted toward 7.3. Nothing
in the last two days changed the 2026-09-01 diagnosis: the shipped Inco draft
reaches 95% of its vendor numbers on the vendor's benchmarks and loses 10%
each to agent framing, real repo content, T=1 sampling, and max thinking. The
per-call attribution says the stream is 60-88% tool-call arguments at 6.1-6.4
accepted, and a single pure edit call hit 7.65 of 8 at 50 tok/s
(`PLAN-OPENCODE-50.md`). A draft trained on our own pi/opencode transcripts
with our own quantised target's hidden states attacks exactly that.

Tooling is now mature: SpecForge v0.3 (sgl-project) trains DFlash/DFlash2/
DSpark/Domino online and offline; Speculators v0.5 (vLLM) trains DFlash and
DFlash2 ("DFlash plus local dynamic convolutions and a candidate selector",
objective marked experimental) with an offline `generate-offline-data` path,
so hidden states can be produced first and training run separately. Meta
shipped a DFlash drafter with **16-token blocks** for Muse Glimmer (Aug 2026)
and TreeFlash (2606.03819) adds an AR-approximation MLP for +12% block
efficiency; both are variants to consider once a training pipeline exists.

Cost: hidden states for ~20M tokens of our traffic need the 743B target
resident somewhere: the cluster itself at 520-640 tok/s prefill (~10 h of
maintenance windows) or an 8xH200 rental at FP8 for a day; then training the
6-layer drafter needs a few GPU-days. Expect $1-3k and a week of pipeline
work. Licence: Inco's draft is CC BY-NC-ND, so this is a **new checkpoint
distilled from our target**, not a fine-tune. Evaluate offline on held-out
pi traffic (accepted length per category) before any Spark time. This is the
one item with a plausible +20-30% on code and it has been sitting in
`PLAN-40-TOKS.md` Phase 3 since 2026-09-01.

### G. Prefill and TTFT

Uncached prefill is 520-640 tok/s (~11 TFLOPS per GB10, roughly 9% of its
bf16 peak). tonyd2wild's GLM-5.3-Flash NVFP4 recipe on four Sparks reports
1,863 tok/s warm at 114k, and the vLLM Spark blog 1.6-1.9k tok/s for a 120B
NVFP4 model on one Spark. Our routed experts go through Marlin, a small-M
decode kernel, at M=2048 per chunk; nobody has profiled a prefill chunk on this
stack. Each pi tool call re-prefills the previous turn and tool output
(0.3-2.5 s, 5-8 calls per task, 15-30% of task wall per `PLAN-40`), and a
fresh 100k repository prompt is 215-220 s.

Work: profile one 2048-token chunk (no restart; arm the profiler and send one
uncached prompt), then either a dequant-to-bf16 + cuBLAS path for M >= 256 in
`MarlinExperts`, or the sm121-native fused b12x MoE, which is "hardware-ideal"
but accepts NVFP4 experts only (`GLM-NONDISRUPTIVE-MOE-AUDIT.md` §4). That
makes an **NVFP4 requant of the experts** a two-for-one with D (same 4.5
bits/weight as int4-g128, plus the fused kernel for prefill and possibly less
decode glue: `moe_align_block_size`, sort, two launches per layer). Also
cheap and prefill-only: the ring pins one QP per connection and reaches
~98 Gbit/s on 200G links; `NCCL_IB_QPS_PER_CONNECTION=2` or a graph file with
both port functions roughly doubles large-message bandwidth (~0.5 s of the
3-4 s chunk is all-reduce).

### H. Small cycle items worth batching into one maintenance window

From `IDEAS-DCP-LATENCY.md` (not yet tried): fold the LSE all-gather into the
reduce-scatter (-2.9 ms), fuse the indexer merge into the query gather and
overlap it (-3 ms), DCP glue fusion (-2 to -3 ms). New: fp8 AR payload
(NCCL LL cost is 0.45 us/KiB, 96 -> 48 KB saves ~20 us x 167 = **-3.5 ms**;
moot if E lands), drafter fp8 (D), indexer int8 (D). Together **-6 to -10 ms**
without new transports or formats. Each is a few hundred lines of overlay.

### I. Money

TP8 (four more Sparks) halves bytes per rank: the cycle drops to ~90 ms and
everything above still applies on top. That is the Light Foundry configuration
(48 tok/s single-stream GLM-5.2 with MTP on eight). A switch is the cheaper
purchase and mostly buys E without the forwarding engineering.

### J. Not architecture, but the largest measured lever

The 2026-09-01 harness study stands: same model, same sampling, pi 3.97
accepted at 27 tok/s and 95 s per task; opencode with a terse prompt, effort
`high`, T=0: 6.07-6.13 accepted, **41.6 tok/s**, 21-28 s per task, all checks
passing. `reasoning_effort` defaults to `max` for any harness that sends
nothing. If pi remains the daily harness, its system prompt and effort setting
are worth more than any single item above and cost nothing.

## 3b. Measured on 2026-09-10 (guarded holds; production restored after each)

| lever | what was booted | result | verdict |
|---|---|---|---|
| A lossy verification | DCP2 daily lane + rejection-kernel overlay, `GLM_SPEC_LOSSY=1` | dev sweep m0.5..m2.5: monotonic, m2.5 +12.6%; **held-out m2.5 +19.7% prose [17.0, 22.2]**, all 15 prompts positive, TTFT/gap unchanged, no next-cycle acceptance penalty; blind 256-token screen 54 ties / 2 / 4 | speed gate passed; quality gates G3/G4/G6 pending; code path exact by construction (opt-in) |
| B MTP K=2 | pool 3.2e9, window 90112, policy off, codex's tuple-contract overlay | cycle 105 ms; prose 20.0 vs 15.0 (screen) and 21.7 vs 18.1 (bench); repo 22.2 vs 19.4; code 24.8 vs 26.3 (screen) and 27.5 vs 40.3 (bench); count100 27.7 vs 54 | prose-session boot option; not a default (3-token cycle ceiling kills structured output) |
| H LSE fold | `GLM_DCP_LSE_FOLD=1` | count100 146.0 vs 144.9 ms, text identical; prefill +0.5 s of torch ops | neutral at decode, harmful at prefill; off; would need a fused pack kernel |
| G prefill profile | `PROFILER_DIR`, one 4158-token uncached prefill | 490 tok/s; NCCL 33% (AR 14%, DCP AG 10%, RS 8%, all ~11-13 GB/s), glue 17% (copies/casts; reduce-scatter's `movedim().contiguous()`; compaction chain with a per-layer `.all()` sync), MoE 18% (~25% MFU), attention 13% | levers: glue overlay ~6-7%, MoE prefill kernels; not chunk size or QPs (below) |
| G prefill chunk 4096 | `--maxbatched 4096` | 40k 80.1 s (500 tok/s), 100k 217.9 s (459) vs 473/460 at 2048 | no gain; 8192 leg cut; 2048 stays |
| G NCCL QPs=2 | `--ib-qps 2` | 40k 79.8 s, 100k 215.1 s; decode identical | no gain; the ring's one-channel/one-CTA limit, not the QP |
| D-lite | offline inventory only (codex) | indexer is 21 layers (0.73 ms if W8A16), drafter fp8 ~2.7 ms; drafter actually runs TP4 | ~2.3% total; deferred |
| F | declined by the operator | — | — |

| B+A stacked | MTP K=2 + lossy (hold 6) | m1.5 0.99, m2.5 1.00, m3.5 1.03 vs the MTP baseline | do not stack; the rule rarely fires on MTP drafts |
| A ceiling | DFlash lossy m2.5/m3.5/m5.0 (hold 7, dev prose) | +10.9% / +15.5% / +21.9% [18.0, 26.6]; relaxed 0.18 -> 0.31/cycle; no next-cycle penalty | ~+30% held-out plausible at m5.0; quality gates first |

Not yet measured: C (3-bit experts), E (ring collectives), the prefill glue
overlay.

## 4. Recommended order

1. **Now, no restart:** deploy the draft-cache repair (already validated in
   isolated boots). Profile one uncached 2048-token prefill chunk. Set pi's
   effort/temperature the way opencode's `glm_best` is set.
2. **Week 1:** A (bounded-lossy verification, opt-in). Local Triton + tests,
   one boot, blind prose A/B and complete-function checks. If prose reaches
   ~25 tok/s with quality held, B becomes optional.
3. **Week 1-2, in parallel, offline:** D's checkpoint work (indexer to int8,
   the big projections to 4-bit) with per-tensor KL against the current
   weights; E's standalone verbs microbenchmark (needs the stack down for an
   hour, same as the multicomm test). Both are go/no-go gates, not deployments.
4. **Week 2-3:** B's boot if A did not cover prose; C's requant if D's KL
   method is trusted and C's kernel plugin loads on this fork in the sandbox.
5. **Start the data pipeline for F now** (transcript capture + hidden-state
   export are cheap); decide on rental vs cluster time once A-E have set the
   new cycle, since F's value is a multiplier on whatever cycle they leave.

What not to do again: adaptive-policy tuning, controller features, NCCL
environment tuning, K sweeps of the trained-in block, copy/n-gram drafters,
and any "small draft model" that traverses the target.

## Sources

Local: `docs/DESIGN.md` §8, `docs/IDEAS-DCP-LATENCY.md`, `docs/SPEED-NEXT.md`,
`docs/ADAPTIVE-SPECULATION-NEXT.md`, `docs/research/*.md`,
`results/adaptive-spec/README.md`, `results/pi-humaneval-compare-20260909/`,
`results/nccl-multicomm/RESULTS.md`, `~/spark-cluster-experiments/PLAN-40-TOKS.md`,
`PLAN-OPENCODE-50.md`, `PHASE1-ACCEPTANCE-RESULTS.md`, `GLM-NONDISRUPTIVE-MOE-AUDIT.md`,
`nccl-latency-results.md`, `~/lmcache-mg/models/glm-5.3/config.json`.

External (inspected 2026-09-09):
- RoCEnante: https://github.com/local-inference-lab/b12x/blob/75ffee6375b0577ce2c8d6931ffacefda3ecbdd6/docs/rocenante.md
- MARS: https://arxiv.org/abs/2601.15498 ; Revisiting Lossy Verification: https://arxiv.org/abs/2607.26627 ;
  Judge Decoding: https://arxiv.org/abs/2501.19309 ; SelfJudge: https://arxiv.org/abs/2510.02329 ;
  DIVERSED: https://arxiv.org/abs/2604.07622
- GLQ: https://github.com/cnygaard/glq ; GLM-5.3 EXL3 3 bpw on four Sparks:
  https://huggingface.co/drowzeys/keys-GLM-5.3-EXL3-Abliterated ; vllm-exl3 recipe:
  https://github.com/vcruz305/DeepSeek-V4-Flash-Vision-EXL3-MixedK-DGX-Spark-recipe ;
  2 bpw Flash on one Spark: https://forums.developer.nvidia.com/t/60-tok-s-glm-5-3-flash-on-a-single-dgx-spark/382140 ;
  2-bit experts + NVFP4 big-3 attention on two Sparks: https://howtospark.com/recipes/glm-5-2-dual-spark-tp2
- SpecForge: https://github.com/sgl-project/SpecForge ; Speculators DFlash training:
  https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/ ;
  Muse Glimmer DFlash2 (16-token blocks): https://huggingface.co/z-lab/Muse-Glimmer-30B-DFlash2 ;
  TreeFlash: https://arxiv.org/abs/2606.03819
- Prefill references: https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark ;
  https://vllm.ai/blog/2026-06-01-vllm-dgx-spark
- Light Foundry eight-Spark MTP: https://x.com/light_foundry/status/2080548348566880326 ;
  https://x.com/light_foundry/status/2097524565916414221
