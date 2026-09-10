# D-lite: checkpoint bytes, loader contracts, and admission gates

2026-09-10. Offline implementation and a **gated future experiment**, not a deployment.
Read alongside [architecture §D](../SPEED-ARCHITECTURE-OPTIONS.md#d-dense-and-indexer-bytes),
[lever status](../SPEED-LEVERS-STATUS.md), and the
[non-disruptive MoE audit](/home/napta2k/spark-cluster-experiments/GLM-NONDISRUPTIVE-MOE-AUDIT.md).
The header inventory corrects several estimates in those documents. The immediately
implementable change is **21 replicated indexer `wq_b` matrices to W8A16**:
175,988,736 fewer streaming bytes/rank/cycle, **0.733 ms at 240 GB/s**. Quantizing
the drafter needs a narrower checkpoint recipe: all-linear online fp8 breaks its
raw context-KV path. MLP + output projections + `fc`, keeping QKV bf16, offers
641,728,512 fewer bytes/rank/pass before scales, **2.674 ms** of bandwidth accounting.
Neither is lossless by construction; neither saving has been measured.

## 1. Evidence and accounting convention

[Saved metadata](dense-bytes-checkpoint-metadata.json) contains the real config,
index/config hashes, every non-expert tensor's name/shape/dtype/offset/bytes/shard,
all shard-header hashes, and routed-expert header groups (layer, suffix, dtype,
shape, count and bytes). It was captured at **2026-09-10 00:44:22 UTC** using
stdlib-only header reads on `spark-06c4.local`, solely within the two authorized
checkpoint directories. Target: 282 shards; draft: one shard, 96 bf16 tensors,
4,918,848,512 bytes. No weight payload, GPU operation, model execution, endpoint
request, or container operation was used. Header hashes establish the checkpoint
snapshot; they are not hashes of unseen payloads.

Local source abbreviations below: `F` = `/home/napta2k/lmcache-mg/spark-src/vllm`,
`M` = `/home/napta2k/lmcache-mg/glm-triton/deepseek_v2.py`, `I` = the neighboring
`sparse_attn_indexer.py`. [Source manifest](dense-bytes-sources.json) pins their
SHA256s and line counts. Both the fork and the mounted-model copies were read.
References are to that snapshot, not a claim about a later running image.

Bytes are decimal bytes, milliseconds = bytes / **240,000,000**. A verify cycle
means one target pass at C1/K7 (eight query positions), not eight reads of every
matrix. TP4 shards column/row linears; DCP2 does not further shard their weights.
Tables count **one full read of each participating weight/scales tensor per
operation**. They exclude activation/KV traffic, tile rereads, cache reuse,
collectives, kernels' scratch and allocator overhead. This is reproducible
weight accounting, not a claim that each byte reaches DRAM once. `weight_shape`
is load metadata, not per-cycle traffic. Norms are included for completeness.
Full tensor rows, including runtime-derived weights, are reproducible locally:

```bash
python3 bench/dense_bytes_inventory.py --out /tmp/dense-bytes-inventory.json
```

### Target TP4 totals

| Family | Stream bytes/rank/verify cycle | ms @ 240 GB/s |
|---|---:|---:|
| Int8 dense projections, including group scales; excluding packed `kv_b` | 4,718,704,640 | 19.661 |
| Derived bf16 `W_UK_T` / `W_UV`, all 78 layers | 572,522,496 | 2.386 |
| bf16 `lm_head` | 475,791,360 | 1.982 |
| Active bf16 indexer, 21 layers | 393,619,968 | 1.640 |
| bf16 MoE router weights | 235,929,600 | 0.983 |
| bf16 layer-0 matrices, excluding indexer and derived `kv_b` | 212,598,784 | 0.886 |
| Other bf16 norms | 2,328,576 | 0.010 |
| Router correction bias, F32 | 76,800 | 0.00032 |
| Embedding table lookup, packed `kv_b` decode read, unused MTP | 0 full-matrix streaming bytes | 0 |

This removes `lm_head` from the memo's int8 family and MTP from its active bf16
bucket. It also separates checkpoint `kv_b` from the **bf16** matrices actually
used in decode. The audit's earlier 0.72 GB dense estimate and “13% of bandwidth”
conclusion do not follow from these headers. Nor can its seven-serial-draft-steps
description be applied to the current block-parallel DFlash2 implementation.

### Unquantized inventory

Names below are suffixes of `model.layers.L.` unless fully qualified; shapes are
checkpoint `[N,K]`. “TP4” means a quarter of matrix bytes, “rep” means all ranks
hold/read the full matrix. Layer 0 is dense; layers 3–77 have MoE routers.

| Name / layers | Shape, dtype | Placement | Bytes/rank/cycle, each tensor |
|---|---|---|---:|
| `self_attn.indexer.wq_b.weight`, L in S | [4096,2048], BF16 | rep | 16,777,216 |
| `self_attn.indexer.wk.weight`, L in S | [128,6144], BF16 | rep, fused at load | 1,572,864 |
| `self_attn.indexer.weights_proj.weight`, L in S | [32,6144], BF16 | rep, fused at load | 393,216 |
| `self_attn.indexer.k_norm.{weight,bias}`, L in S | [128], BF16, each | rep | 256 |
| `mlp.gate.weight`, L=3..77 | [256,6144], BF16 | rep | 3,145,728 |
| `mlp.gate.e_score_correction_bias`, L=3..77 | [256], F32 | rep | 1,024 |
| `self_attn.q_a_proj.weight`, L=0 | [2048,6144], BF16 | rep | 25,165,824 |
| `self_attn.kv_a_proj_with_mqa.weight`, L=0 | [576,6144], BF16 | rep | 7,077,888 |
| `self_attn.q_b_proj.weight`, L=0 | [16384,2048], BF16 | TP4 column | 16,777,216 |
| `self_attn.o_proj.weight`, L=0 | [6144,16384], BF16 | TP4 row | 50,331,648 |
| `mlp.{gate_proj,up_proj}.weight`, L=0, each | [12288,6144], BF16 | TP4 column | 37,748,736 |
| `mlp.down_proj.weight`, L=0 | [6144,12288], BF16 | TP4 row | 37,748,736 |
| `self_attn.kv_b_proj.weight`, L=0 | [28672,512], BF16 | TP4 column | 0 packed/checkpoint read in decode |
| `{input_layernorm,post_attention_layernorm}.weight`, L=0..77, each | [6144], BF16 | rep | 12,288 |
| `self_attn.q_a_layernorm.weight`, L=0..77 | [2048], BF16 | rep | 4,096 |
| `self_attn.kv_a_layernorm.weight`, L=0..77 | [512], BF16 | rep | 1,024 |
| `model.norm.weight` | [6144], BF16 | rep | 12,288 |
| `lm_head.weight` | [154880,6144], BF16 | TP4 vocab | 475,791,360 |
| `model.embed_tokens.weight` | [154880,6144], BF16 | TP4 vocab | lookup only; 475,791,360 resident |

**S = {0,1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74}.**
There are 21 active indexers plus one unused MTP indexer, not 78 active copies.
This follows both the headers and `index_topk_freq=4`, offset=3;
`M:1037–1079` selects which layers instantiate an indexer. The remaining layers
reuse indices (`I`, sparse-indexer forward/top-k path). `M:628–645` constructs
replicated WQ and fused WK/weights projections; `linear.py:290–377` implements
full-size `ReplicatedLinear`. `M:866–884` disables TP for fused q/kv-A;
`M:964–1008` shards q-B/kv-B/o; `M:217–229` shards dense/shared MLPs.
`M:275–280` constructs the router without an fp32 override; the GateLinear
defaults (`F/model_executor/layers/fused_moe/router/gate_linear.py:37–65`)
preserve the BF16 checkpoint/execution dtype despite `router_dtype` in HF config.

MTP has **zero loaded/streamed bytes in this target + separate-DFlash lane**:
`M:1435–1440` skips layer 78. Its BF16 leftovers are
`model.layers.78.eh_proj.weight` [6144,12288] (150,994,944 bytes),
`{enorm,hnorm,input_layernorm,post_attention_layernorm,shared_head.norm}.weight`
[6144] (12,288 each), `mlp.gate.weight` [256,6144] (3,145,728), the same five
indexer tensors as above, and q/kv-A norms [2048]/[512]. Its router bias is
[256] F32. Its other attention/shared/expert projections use per-channel W8:
`weight_packed [N,K/4] I32`, `weight_scale [N,1] BF16`, `weight_shape [2] I64`;
exact N/K and expert multiplicities are in the saved headers. Those are a
different MTP experiment's admission cost, not D-lite savings. Neither
`self_attn.indexers_proj` nor `shared_head.head` has a tensor in this checkpoint;
their ignore patterns match absent modules. The actual `lm_head` remains BF16
because the quantization groups do not target it (`M:1697–1703`).

### Existing W8A16 dense inventory

Each row names `model.layers.L.<module>.{weight_packed,weight_scale,weight_shape}`.
Packed tensors are I32, scales BF16, shapes I64 [2] (16 bytes/module, replicated,
load-only). All active W8 rows are symmetric group-128. Table bytes include
scales; multiply by layer count for the family total.

| Module / layers | Packed shape | Scale shape | TP split | Bytes/rank/cycle per layer |
|---|---|---|---|---:|
| `self_attn.q_a_proj`, 1..77 | [2048,1536] | [2048,48] | rep | 12,779,520 |
| `self_attn.kv_a_proj_with_mqa`, 1..77 | [576,1536] | [576,48] | rep | 3,594,240 |
| `self_attn.q_b_proj`, 1..77 | [16384,512] | [16384,16] | N/4 | 8,519,680 |
| `self_attn.o_proj`, 1..77 | [6144,4096] | [6144,128] | K/4 | 25,559,040 |
| `mlp.shared_experts.{gate_proj,up_proj}`, 3..77, each | [2048,1536] | [2048,48] | N/4 | 3,194,880 |
| `mlp.shared_experts.down_proj`, 3..77 | [6144,512] | [6144,16] | K/4 | 3,194,880 |
| `mlp.{gate_proj,up_proj}`, 1..2, each | [12288,1536] | [12288,48] | N/4 | 19,169,280 |
| `mlp.down_proj`, 1..2 | [6144,3072] | [6144,96] | K/4 | 19,169,280 |
| `self_attn.kv_b_proj`, 1..77 | [28672,128] | [28672,4] | N/4 | **0 in decode**; 3,727,360 resident before shape |

For all L=0..77, `self_attn.mla_attn.W_UK_T [16,192,512] BF16`
(3,145,728 bytes/rank) and `W_UV [16,512,256] BF16` (4,194,304) replace the
packed kv-B decode GEMM. The loader applies the quantized linear to an identity
to recover the weight and creates these bf16 buffers
(`F/model_executor/layers/attention/mla_attention.py:860–888,945–947`). Packed
kv-B remains available for prefill. Reducing its checkpoint precision alone
does **not** halve these decode buffers. Routed int4 experts are excluded from
this dense-only inventory; their selected expert count is request-dependent.

### DFlash2 pass inventory and the TP1 discrepancy

The requested draft TP1 setting is not honored by the pinned **V2** loader.
`F/v1/worker/gpu/spec_decode/dflash/utils.py:25–45` replaces the attention config
only, passes the target `parallel_config` to `get_model`, and shares target
embedding/head at lines 60–76. Qwen draft attention gets its TP size from the
global group. The extracted loader contract test proves TP4 survives a
`draft_tensor_parallel_size=1` config. DCP-replicated draft KV is not TP1
weight replication. A future boot must log the actual qkv/o/fc parameter shapes;
if they differ, regenerate the table before admission. The TP1 column below is
the explicit counterfactual requested for comparison, not the current V2 path.

Checkpoint names use `layers.0..5.`; quantization runtime prefixes use target
offset `model.layers.78..83.` (`qwen3_dflash.py:350–362`). Every listed tensor is
BF16. Bytes below aggregate all six layers (or both codebooks), per draft pass.

| Checkpoint name(s) | Shape per tensor | Actual TP4 bytes/pass | TP1 replicated bytes/pass |
|---|---|---:|---:|
| `layers.L.self_attn.q_proj.weight` | [8192,6144] | 150,994,944 | 603,979,776 |
| `layers.L.self_attn.{k_proj,v_proj}.weight`, combined | [1024,6144] each | 37,748,736 | 150,994,944 |
| `layers.L.self_attn.o_proj.weight` | [6144,8192] | 150,994,944 | 603,979,776 |
| `layers.L.mlp.{gate_proj,up_proj,down_proj}.weight`, combined | [12288,6144], [12288,6144], [6144,12288] | 679,477,248 | 2,717,908,992 |
| `layers.L.{attention_conv,mlp_conv}.kernel_projection.weight`, combined | [1536,6144] each; replicated | 226,492,416 | 226,492,416 |
| `layers.L.{attention_conv,mlp_conv}.base_kernel`, combined | [2,2,6144] each; replicated | 589,824 | 589,824 |
| `fc.weight` | [6144,36864]; replicated | 452,984,832 | 452,984,832 |
| `candidate_selector.hidden_projection.weight` | [256,6144]; replicated | 3,145,728 | 3,145,728 |
| All norms | layer input/post [6144], q/k [128]; hidden/final [6144]; replicated | 175,104 | 175,104 |
| `candidate_selector.{predecessor,successor}_codebook` | [154880,256] each; replicated | gathered rows only | gathered rows only |
| Derived `_fused_kv_weight`, context precompute | TP4 [3072,6144]; TP1 [12288,6144] | 37,748,736 | 150,994,944 |
| Shared target `lm_head.weight`, candidate logits | [154880,6144], vocabulary-sharded with target | 475,791,360 | 1,903,165,440 |
| **One-read matrix total** | | **2,216,143,872** | **6,814,411,776** |
| **ms @ 240 GB/s** | | **9.234** | **28.393** |

Codebooks add 158,597,120 resident bytes/rank, but `_score_edges` gathers candidate
rows; do not charge the full tables per step (`qwen3_dflash2.py:166–184`). The
shared head is read for full-vocabulary candidate selection before taking top16
(`:283–287`), and is shared storage, not a second allocation. Draft embeddings
are also gathered. The six decoder layers alone contain 4,304,096,256 BF16 bytes
at TP1. The observed ~10.9 ms draft pass is consistent with TP4-scale accounting,
not a mandatory 28.4 ms TP1 weight stream. K7 is **one block-parallel pass**, not
seven serial autoregressive passes; context precompute is a separate read of
the fused K/V copy (`qwen3_dflash.py:392–487`).

## 2. Change A: WQ indexer W8A16, with selective ignore subtraction

First candidate: only `model.layers.L.self_attn.indexer.wq_b.weight` for L in S.
Keep WK, weights_proj, norms, routers, the rest of layer 0, MTP and existing int8
groups unchanged. `M:628–634` passes quant_config to ReplicatedLinear WQ. By
contrast `M:635–645` fuses WK/weights into `wk_weights_proj` with
**quant_config=None, disable_tp=True**. Removing ignore patterns cannot quantize
that path. A later WK/weights candidate requires a model construction/mapping
change and separate quality validation; it is excluded from this repack command.

The offline tool emits **checkpoint pack-quantized**, not the post-load Marlin
tile layout. For each selected stem `model.layers.L.self_attn.indexer.wq_b`:

| Tensor suffix | Per-channel | Group-128 | Dtype / convention |
|---|---|---|---|
| `weight_packed` | [4096,512] | [4096,512] | I32, four unsigned-biased 8-bit lanes along K |
| `weight_scale` | [4096,1] | [4096,16] | BF16, checkpoint execution dtype |
| `weight_shape` | [2], values [4096,2048] | same | I64 |

Each row/group uses `s = bf16(max(abs(w))/127)`, with s=1 for zero groups,
`q = clamp(round_to_nearest_even(w / float(s)), -127,127)`, stored as q+128.
Lane K+0 occupies bits 0..7; lane K+3 occupies bits 24..31 of signed int32.
No zero-point or g_idx tensor is emitted. Dequantization is
`(((packed >> (8*lane)) & 255) - 128) * scale`.
The exact consuming allocation/parameter dimensions are
`F/.../compressed_tensors/schemes/compressed_tensors_wNa16.py:86–110,126–207`;
the fork's unpack/dequant is
`F/.../compressed_tensors/compressed_tensors_embedding.py:25–87`.
Tests use the repository `tests/harness.py extract` pattern on a pinned copy of
that actual dequant kernel, executing CPU Triton interpretation, and compare
both schemes and BF16/F16/F32 scales to RTN within one quantization step.

Marlin accepts symmetric `uint8b128`, BF16 activations, group size -1 or 128,
full/partition `(K,N)=(2048,4096)`. Both dimensions already satisfy its tile
requirements; ReplicatedLinear has no TP reduction or partitioned-scale issue.
`F/model_executor/kernels/linear/mixed_precision/marlin.py:34–80` accepts these
shapes and permits padding when no activation order is present; `:119–137`
transposes parameter layout and calls `gptq_marlin_repack`. Supported groups are
`[-1,32,64,128]` (`marlin_utils.py:34`). Hypothetical WK `(6144,128)` is aligned;
standalone weights_proj N=32 or fused N=160 needs N padding to 64 or 192 in
this fork. Padding support does not fix the hard-coded unquantized fused module.
Kernel eligibility is source-verified; SM121 execution remains a future gate.

`config.json` retains compressed-tensors/pack-quantized and inserts a first
`dense_repack_w8a16` config group with exact selected **module** targets,
weights `{num_bits:8,type:int,symmetric:true,strategy:channel,group_size:-1,dynamic:false}`
(or strategy=group, group_size=128). Existing groups keep their order and values.
For every matching ignore regex, the tool prepends
`(?!(?:<escaped selected module names>)(?:$|\.))` to subtract only those modules.
This edits both the broad layer-0 exclusion and the later indexer exclusion;
simply deleting the latter misses layer 0. Every unselected checkpoint module's
ignore decision is checked before output. Fork regex/fused matching and first
matching group semantics: `compressed_tensors/utils.py:49–102,180–194` and
`compressed_tensors.py:309–367,877–892`.

| WQ-only candidate, all 21 layers | Stream bytes/rank/cycle | Saved bytes/cycle | Saved ms |
|---|---:|---:|---:|
| Original BF16 | 352,321,536 | 0 | 0 |
| W8 per-channel | 176,332,800 | 175,988,736 | 0.733 |
| W8 group-128 | 178,913,280 | 173,408,256 | 0.723 |

Add 336 bytes of shape metadata to either new checkpoint total. Per-channel
creates 176,333,136 payload bytes and reduces logical checkpoint bytes by
175,988,400. Existing source shards remain physically intact. Quantizing all
active indexer matrices would have less than 0.83 ms of one-read savings; the
memo's 2.5–3 ms claim is not supported by this checkpoint.

### Indexer quality gate, executable on CPU

Indexer weights affect **which 2048 keys are selected**, including reuse by
later layers. Small projection error can move keys across a discontinuous
selection boundary. RTN error alone is insufficient; “lossless in practice”
is a hypothesis to test. Save paired **final aggregated, masked pre-top-k
scores**, using exactly the same target prefixes, key identities and masks.
Use saved activations to replay WQ + the indexer transforms on CPU, or capture
paired teacher-forced requests in a later guarded experiment, then score locally.
Free-running continuations with different prefixes are not matched samples.
An isolated CPU projection test shares the saved key cache; paired full-model
requests additionally measure drift propagated into each model's own cache.
The score replay must preserve RoPE, query/key quantization, head weights,
masking and head aggregation from `M:692–779` / `I`; raw WQ output is not a
substitute. No activation collector or inference was run for this task.

For **each** of S and each context stratum (8k, 32k, 100k; prose/code separately),
require >=512 held-out query rows with eligible keys >2048. Use stable key-index
ties. Fail if mean KL(softmax(BF16)||softmax(W8)) >1e-3, p95 KL >1e-2, mean top-k
set overlap <99%, or p05 overlap <98%. These are proposed screening thresholds,
not a theorem about task quality. The tool rejects mismatched masks/nonfinite
scores and vacuous short-context overlap; mmap inputs bound RAM by one query row.

```bash
.venv/bin/python bench/dense_indexer_gate.py saved/layer06-32k-code-bf16.npy \
  saved/layer06-32k-code-w8.npy --topk 2048 --min-rows 512 \
  --out results/dense-layer06-32k-code-gate.json
```

**A promotion gate:** all CPU gates pass; target functional-code checks and
deterministic prose constraints do not regress; fixed7 accepted tokens/cycle
is >=99% of the matched control; paired verify-cycle saving has a 95% bootstrap
lower bound >=0.5 ms and end-to-end decode speed's lower bound is positive.
TTFT p95 <=1.05x control. Peak host memory must satisfy §4. Otherwise reject or
try group-128 as a separately named, separately gated candidate. Do not combine
with lossy verification when measuring this change.

## 3. Change B: selective draft fp8; reject the blanket online flag

The configuration plumbing exists: `SpeculativeConfig.quantization` feeds the
draft `ModelConfig` (`F/config/speculative.py:103,713–728`);
`get_draft_quant_config` (`F/model_executor/models/utils.py:737–758`) resolves
the draft's own quantization config. `qwen3_dflash2.py` inherits the six-layer
Qwen DFlash model; `qwen3_dflash.py:330,355,371–377` passes quant_config to its
layers and replicated `fc`. The loader adds `model.` to checkpoint names and
collects them into a dictionary before `AutoWeightsLoader`, then builds fused
context buffers (`:689–721`). Runtime layer prefixes start at target layer 78.

`{"quantization":"fp8"}` in speculative-config with a BF16 checkpoint selects
`Fp8PerTensorOnlineLinearMethod` (`F/.../quantization/fp8.py:176–201`). The online
loader uses meta weights/JIT materialization (`online/fp8.py:61–103`) and
quantizes/transposes the weight to `[K,N]` fp8 at `:153–166`. This generic support
is **not end-to-end DFlash2 support**: `_build_fused_kv_buffers` directly slices
`a.qkv_proj.weight[a.q_size:]` as `[N,K]` and its context path calls raw
`F.linear` with BF16 inputs (`qwen3_dflash.py:392–446,487`). Online transposition
gives the wrong slice/layout; a serialized fp8 QKV weight also fails the raw
BF16 context linear contract. CPU tests expose the layout failure. Reject an
all-linear fp8 flag experiment before any boot.

Viable next recipe: pre-quantize **MLP gate/up/down, o_proj and fc only**;
preserve QKV BF16 and its `_fused_kv_weight` layout/dtype. Conv projections and
the selector already use quant_config=None (`qwen3_dflash2.py:69–76,206–212`);
norms and shared embedding/head stay BF16. A selective serialized fp8 checkpoint
can use `Fp8Config.ignored_layers` to exclude all six runtime fused
`model.layers.78..83.self_attn.qkv_proj` modules (confirm the actual prefix during
the constructor contract test, including any caller prefix). Fused ignore
mapping must match all q/k/v components. `fp8.py:156–174` treats a config with
`quant_method:fp8` as **already serialized**, so adding an HF ignore config to
untouched BF16 weights is not an online-quantization workaround. Either produce
the selective serialized tensors/scales with the pinned FP8 loader's contract,
or add explicit selective online plumbing in a separate future change. This
task's repacker deliberately implements int8 only, not an unvalidated fp8 writer.

At actual TP4, eligible BF16 bytes are `fc 452,984,832 + MLP 679,477,248 +
o_proj 150,994,944 = 1,283,457,024`; fp8 saves **641,728,512 bytes/pass before
scales**, or 2.674 ms @240 GB/s. Scaling the observed 10.9 ms by this fraction
of total draft bytes gives an optimistic ~3.16 ms estimate; budget **2–3 ms**
until profiling, not an unconditional halving. The TP1 counterfactual saves
1,887,436,800 bytes (7.864 ms), but is not the source-derived running layout.
Changing the draft need not change exact verification's target distribution;
it can reduce acceptance enough to make decoding slower, and finite-precision
target/backend effects still require the ordinary quality checks.

**B promotion gate:** selective fp8 loader contract passes (same BF16 QKV/cache
buffers, finite outputs, correct scale dispatch); held-out fixed7 accepted
tokens/cycle >=99% of control; paired draft-pass saving has a 95% bootstrap lower
bound >=2 ms, end-to-end decode speed's lower bound is positive, and code/prose
quality and TTFT gates from A pass. Peak host memory must satisfy §4. Reject
if QKV quantization is needed to reach the threshold; treat that as a separate
model-code project. Evaluate A and B independently before any combined trial.

License: the draft's CC BY-NC-ND terms permit producing/reproducing adapted
material for **noncommercial** purposes but prohibit sharing it; keep quantized
draft weights private, never publish the derivative. “Private” alone does not
authorize commercial serving. See [CC BY-NC-ND 4.0 §2(a)(1)(B)](https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.en).

## 4. Symlink loading, bounded memory and guarded experiment design

**Index-only replacement is not sufficient in this fork.**
`weight_utils.py:582–601` filters shard filenames using the index;
`:948–954` still iterates **every key** in each selected shard. Source shards
containing unchanged tensors also contain the retired BF16 `.weight`, which
would be loaded despite its absence from the new index (`M:1599` looks up the
parameter by name). Tests reproduce that failure condition with the extracted
default iterator. Copying the whole ~380 GB checkpoint is unnecessary.

`bench/dense_indexed_loader.py` supplies an opt-in `dense-indexed` loader. It
reads only indexed names **before** `get_tensor`, preserves prefixes and expert
filtering, and uses the default safetensors loader for unmarked models (including
the original draft). Importing the module does not register or load anything.
For a future isolated experiment, expose `dense_indexed_loader:register` as a
`vllm.general_plugins` entry point via a small experiment-only package on
PYTHONPATH and use `--load-format dense-indexed`; preserve any existing plugins.
The package needs the module plus `.dist-info/entry_points.txt` containing:

```ini
[vllm.general_plugins]
dense_indexed = dense_indexed_loader:register
```

Also supply package METADATA (`Name: dense-indexed`, `Version: 0.1`). Stage and
hash this package through the guarded harness in a later implementation; do
not install it into the base image. Marked checkpoints use
`dense-repack-manifest.json`; unmarked checkpoints retain normal iteration.
Until that lane is implemented, outputs are **offline artifacts, not drop-in
`--load-format auto` checkpoints**. Relative index paths point to absolute
symlinks: at eventual staging, recreate each link to that host's original model
shard; sandbox absolute paths are not portable to a Spark.

### Admission arithmetic

Host peak must be <= the **current checkpoint's measured load peak**, on every
rank, with the same KV pool, context, dtype, compile/cache policy and loader
strategy. Existing MemoryGuard's 768 MiB preflight / 512 MiB hard minimum,
swap/PSI checks, heartbeat/watchdog and exact-container rollback remain mandatory
(`bench/spec_memory.py:28–61,112–130`; `spec_experiment.py:325–416`). A predicted
saving does not waive these gates. Do not reclaim the saving by increasing KV.

* **WQ per-channel:** persistent tensor reduction is 175,988,400 bytes/rank
  before Marlin scratch. A new matrix payload is 8,396,816 bytes versus
  16,777,216 BF16. Post-load Marlin repack may hold input, contiguous transpose
  and output temporarily: reserve two extra 8,388,608-byte packed buffers plus
  scale scratch. Even a conservative **32 MiB incremental reserve** leaves
  >142 MB below baseline. Workspace is `4 * SM_count` bytes per layer
  (`marlin_utils.py:366–374`), not a full-weight buffer. Initialization allocates
  the smaller packed weights from the outset; old BF16 WQ must never be
  materialized by the indexed loader. Unchanged tensors' largest read is no
  larger than baseline. Require the memory trace to confirm this phase model;
  abort a trial with an unaccounted allocation rather than using the residual
  headroom as permission to exceed baseline.
* **Selective serialized draft fp8:** persistent reduction ~641.7 MB/rank before
  scales. The BF16 draft loader currently retains a dictionary of all 4.919 GB
  source tensors until copying completes. Selective FP8 reduces logical source
  payload by 1,887,436,800 bytes (to ~3.031 GB before scales); indexed iteration
  must exclude the retired BF16 payloads here too. Do not hold both versions in
  that dictionary. Largest changed `fc` falls 452,984,832 ->226,492,416 bytes;
  reserving one additional fp8 transpose buffer of 226.5 MB leaves >415 MB of
  persistent headroom before scales. Unchanged QKV context copies and temporary
  shared embedding/head construction are baseline costs, not new savings.
  Admission requires a phase-by-phase loader allocation ledger and CPU tests
  of the selective serialized recipe before a boot. Blanket online FP8 is
  already rejected by its KV contract; it also cannot claim the serialized
  source-payload reduction.

These are incremental bounds, not a newly invented absolute “safe host peak.”
Use matched control/trial memory.jsonl and the same starting host state; compare
per-rank minimum MemAvailable, process/allocator peaks and swap deltas. Exclude
whole-shard eager/prefetch strategies in the candidate. The indexed adapter is
lazy and single-threaded. Never run the repacker on a Spark: its CLI refuses
Spark hostnames/ARM and hides CUDA; tensors/operations explicitly use CPU.
The default 256 MiB input-tensor cap is checked from headers before payload
allocation. For this WQ selection the largest input is 16 MiB, output ~8 MiB,
and RTN scratch is bounded to 32 rows; no checkpoint-sized tensor list, source
shard copy or whole-payload hashing buffer is used.
Wider matrices automatically use fewer rows so each FP32/I32 scratch array is
at most 1 MiB; rows wider than 262,144 elements fail before allocation. Dry-run
also reports a conservative numeric-buffer budget (input + output + 64 bytes
per scratch element), excluding Python/torch and metadata overhead. For WQ it
is about 28 MiB plus those fixed overheads.

### Future harness sequence (not executed here)

1. Freeze control checkpoint/config, source manifest, fixed7 policy, prompt set,
   seed, output budget and actual draft TP shape evidence. Capture a matched
   control memory/load ledger through the existing guarded flow. Prepare the
   CPU gate corpora; absence of a corpus is a failed admission prerequisite.
2. Repack on the sandbox; validate hashes, index iteration and CPU dequant.
   Transfer **only new packed shards/config/index/manifest**, recreate unchanged
   symlinks locally on each host, and leave original model directories intact.
   A selective draft recipe likewise needs an index over new tensors plus
   symlinked unchanged source storage; no 380 GB target copy is involved.
3. The present `validated_lane` whitelist has no model-path/load-format/draft
   quantization override (`bench/spec_experiment.py:67–105`). Claude must add a
   narrowly validated future D-lite lane: immutable candidate model paths,
   plugin hashes, indexed load format and optional selective draft path,
   preserving the original model mounts, command and rollback metadata. This
   plan supplies the required values/contracts, not fictitious runnable harness
   flags. Do not bypass that whitelist with an ad hoc launcher or mutate the
   serving model's config. No protected harness/launcher file was changed here.
4. Use the established prepare/run/HOLD/restore lifecycle, one candidate per
   boot, with the same KV and TP/DCP settings. During that future hold, perform
   count100 and fixed7 smokes, then quality/acceptance and profiling pairs.
   Check loader dispatch and tensor layout before interpreting any timing.
5. Run control/candidate/control/candidate boots with identical warmup and a
   fixed C1 case order; reverse order for the second block (AB/BA). Across
   different model boots this is boot-block pairing, not request-level switching
   in one endpoint. Use development for recipe selection, then freeze it and
   evaluate 15 held-out prose + code/long-context strata, >=3 repeats. Pair by
   prompt/seed/repeat; bootstrap at prompt level (10,000 resamples), not token
   or trace-row level. Record accepted/cycle, per-position acceptance, verify
   and draft family time, decode tok/s, TTFT and every memory gate. Reuse the
   [prose/code tooling](LOSSY-BENCH-TOOLING.md) with variant names identifying
   the **boot model**; send fixed7 lossless requests for both checkpoints.
6. Apply the independently frozen A/B gates above and restore exactly on failure
   or completion. A successful trial authorizes a reviewable result, not a
   production promotion by this offline task. No commits or Spark execution
   were performed here.

## 5. Sandbox commands and checks

`DENSE_MODEL_DIR` must name an already available **sandbox** copy/mount of the
original checkpoint; `DENSE_REPACK_DIR` must name a new sibling output directory.
This task read remote headers only and did not fetch the model payload. Regexes
full-match tensor names; the 0..77 range deliberately excludes MTP layer 78 and
matches exactly the 21 active WQ tensors in the recorded checkpoint.

```bash
export DENSE_MODEL_DIR=/home/napta2k/model-snapshots/GLM-5.3-Int4-Int8Mix
export DENSE_REPACK_DIR=/home/napta2k/model-snapshots/GLM-5.3-indexer-wq-w8-channel

.venv/bin/python bench/repack_dense_int8.py "$DENSE_MODEL_DIR" \
  --tensor-regex 'model\.layers\.(?:[0-9]|[1-6][0-9]|7[0-7])\.self_attn\.indexer\.wq_b\.weight' \
  --scheme per-channel --out "$DENSE_REPACK_DIR" --dry-run

.venv/bin/python bench/repack_dense_int8.py "$DENSE_MODEL_DIR" \
  --tensor-regex 'model\.layers\.(?:[0-9]|[1-6][0-9]|7[0-7])\.self_attn\.indexer\.wq_b\.weight' \
  --scheme per-channel --out "$DENSE_REPACK_DIR"

.venv/bin/pytest -q tests/
python3 -m py_compile bench/repack_dense_int8.py bench/dense_indexed_loader.py \
  bench/dense_bytes_inventory.py bench/dense_indexer_gate.py \
  tests/test_repack_dense_int8.py tests/fixtures/dense_int8/*.py
```

The example snapshot paths are operator-selected destinations, not claims that
the full model exists on this VM. `--dry-run` imports no torch, reads only
headers/config/index, writes nothing, and reports exact selected/new/logical
payload bytes (JSON/header storage excluded). A real run refuses an existing
output, writes one new shard per selected tensor, retains unchanged shards as
symlinks, records source/new hashes, and removes only its own new directory on
failure. Repeatable `--tensor-regex` accepts other explicitly reviewed matrices;
the serializer's shape compatibility alone is not proof of a model's quantized
forward-path compatibility. `--scheme group-128` requires a distinct output and
the same gates. Tiny synthetic checkpoints exercise both complete real writes
and dry runs; no real checkpoint conversion was performed during this task.
