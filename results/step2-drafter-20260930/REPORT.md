# Step 2: drafter diet (2026-09-30 / 10-01)

Overview: `docs/DECODE-SPEEDUPS.md`, step 2.

**Image:** `spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap7-drafter-20260930`. It is step 1 plus the drafter overlay
files `qwen3_dflash.py` and `qwen3_dflash2.py`.

**Launch:**
- `DRAFT_DIR=/var/tmp/models/GLM-5.3-DFlash2-draft-int8`
- `DFLASH_FC_SPLIT=1`
- `DFLASH_HEAD_FP8=1`

Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3 Int4-Int8Mix (Int4
group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

## Design

- **int8 drafter.**
  - `tools/drafter_int8.py` re-encodes the decoder q/k/v/o/gate/up/down, `fc` and the 12 conv `kernel_projection`s
    as compressed-tensors W8A16 int8, group 128. This is the target's dense format and quantizer
    (`quantize_int8`), served by the same Marlin kernel.
  - Size: 4.9 GB → 2.58 GB. Relative weight error per tensor is 0.65–0.73% (`int8-report.json` next to the
    checkpoint).
  - The checkpoint is at `/var/tmp/models/GLM-5.3-DFlash2-draft-int8` on all four Sparks, sha256 `e8e43d8b…`.
  - **Context-KV fix:** `_build_context_kv_buffers` reads the K/V rows before the Marlin repack, so `_dense_rows`
    dequantizes them. `runtime/vllm029/verify_cap_overlay/test_drafter_int8.py` checks it bit-exactly against a
    reference dequantizing decoder.
  - **Conv projections:** `DFlashGroupedConv` now receives the quant config, so a BF16 checkpoint is unaffected.
- **`fc` split** (`VLLM_DFLASH_FC_SPLIT=1`): `fc` is now `ColumnParallelLinear(gather_output=True)` instead of
  `ReplicatedLinear`.
- **FP8 draft head** (`VLLM_DFLASH_HEAD_FP8=1`):
  - The drafter's candidate top-k reads an e4m3 copy of the shared target lm_head (one exponent per 32×32 block,
    Triton GEMM), ported from knapcio's `glm_ds_draft.py` (MIT).
  - The copy is built on the first eager call. The target head, and therefore every committed token, is unchanged.

## Cycle time (live periods, fixed K, short context; `costgrid/periods.json`, `int8only-cost/`)

| | step 1 | int8 drafter only | int8 + fc split + fp8 head | change |
|---|---:|---:|---:|---:|
| n=1 K=1 | 94.3 ms | 90.3 | **89.2** | −5.1 ms |
| n=1 K=3 | 114.2 | — | **108.4** | −5.8 |
| n=1 K=5 | 127.4 | — | **120.7** | −6.7 |
| n=1 K=7 | 142.3 | 138.2 | **135.2** | −7.2 |
| n=2 K=7 | 201.6 | — | 191.7 | −9.9 |
| n=4 K=7 | 284.8 | — | 278.0 | −6.8 |

The int8 drafter alone saves 4.1 ms. The `fc` split and FP8 head add another 1–3 ms.

## Memory (rank 0 MemAvailable, idle after boot)

| step 1 | int8 only | int8 + fc split + fp8 head |
|---:|---:|---:|
| 1,051 MB | 2,022 MB | 1,828 MB |

This fixes step 1's sub-0.8 GB minimum under load.

## Drafter acceptance at fixed K=7 (`c1-fixed7/` vs the stock BF16-drafter probe run, not published)

| set | BF16 tok/cycle | int8 tok/cycle | change |
|---|---:|---:|---:|
| prose (15) | 2.39 | 2.38 | −0.5% |
| code (15) | 4.80 | 4.85 | +1.2% |
| prose+think (4 clean) | 2.19 | 2.30 | +5.2% |
| code+think (5) | 4.37 | 4.18 | −4.3% |

Prose and code acceptance are unchanged. The thinking sets are too small to tell either way.

## C1, verify cap on (`c1-auto-full/`; agent-turn output text is blanked in the published rows)

| set | verify cap v3 (09-30) | step 1 | step 2 |
|---|---:|---:|---:|
| prose (15), mean tok/s | 20.78 | 20.58 | **21.91** (+5.4% vs v3, +6.5% vs step 1) |
| pi agent turns (63), median tok/s | 26.9 | — | 26.7 (paired with v3: +1.8%, faster in 38 of 63) |
| pi agent turns, tokens/cycle | 3.01 | — | 2.89 (−4%) |

The code and thinking sets were cut short (3 code prompts) to make room for the cache-salt test.

**The agent turns gained less than the cycle-time cut predicts (~+5%),** because acceptance on agent turns dipped
about 4%.

**Cache-salt test (`salt/`):** the hypothesis was that the prefixes cached in the GPU/NVMe slab hold draft KV
written by the BF16 drafter. To test it, 8 agent turns were replayed twice in one boot: once as-is, and once with a
fresh `cache_salt` so the int8 drafter computes all of its own context.

| | stale cached context | fresh context (salted) |
|---|---:|---:|
| all 8 turns, tokens/cycle | 2.725 | 2.687 |
| the 4 turns with comparable outputs | | +1.6%, +5.7%, +2.3%, −4.9% |

Stale context does not explain the dip. **Open item:** compare agent acceptance at fixed K=7 between the BF16 and
int8 drafters, and with the `fc` split on and off. This needs extra boots.

## C2–C4 aggregate (`conc/`, same harness as step 1; means of clean rounds)

| n | set | production (base) | step 1 | step 2 | vs step 1 | vs base |
|---|---|---:|---:|---:|---:|---:|
| 2 | prose | 24.1 | 30.0 | 31.3 | +4.6% | **+30.2%** |
| 2 | code | 46.5 | 47.8 | 47.4 | −0.7% | +2.0% |
| 2 | mix | 31.7 | 36.7 | 34.7 (2 rounds) | −5.6% | +9.6% |
| 3 | prose | 28.8 | 37.3 | 39.5 | +6.1% | **+37.1%** |
| 3 | code | 56.2 | 56.2 | 56.3 | +0.2% | +0.3% |
| 3 | mix | 36.6 | 42.3 | 44.3 | +4.6% | **+21.0%** |
| 4 | prose | 33.8 | 44.6 | 46.0 | +3.2% | **+36.1%** |
| 4 | code | 65.1 | 64.9 | 65.1 | +0.2% | −0.1% |
| 4 | mix | 43.3 | 47.9 | 49.6 | +3.7% | **+14.6%** |

An outside user was active during part of this run. Four rounds were flagged as contaminated and excluded; the cells
with 2 rounds are those.

## Decision

**Keep.**
- Cycles are 5–10 ms faster everywhere.
- Prose is +5% at C1 and +3–6% at C2–C4.
- About 0.8 GB per rank is freed.
- Prose and code acceptance are unchanged.

The agent-turn acceptance dip (−4%) stays open; see above.
