# SPDX-License-Identifier: Apache-2.0
"""VLLM_VOCAB_PARALLEL_ARGMAX: greedy target selection without the full-vocab logits gather. Exact.

Stock path (GPUModelRunner.sample, every verify step, every TP rank):
    lm_head shard GEMM [L, 38720] bf16 -> all-gather -> [L, 154880] bf16 (L = n * (K + 1) rows)
    -> RejectionSampler: per-8192-block max/argmax, global argmax, accept/reject, then a
       154-block resample pass for the bonus row
At L = 8 that is 1.9 MB received per rank (184 us NCCL ring in the DCP2 trace) plus ~80 us of
copies and sampler kernels; at L = 32 it is 7.4 MB.

This path (all ranks, same decision):
    the very same lm_head shard GEMM (LogitsProcessor._apply_head)
    -> one Triton kernel: per row, the first maximal shard column, packed with its value into
       one int64 key  [ordered(fp32 value) : 32 | 0xFFFFFFFF - global id : 32]
    -> all-gather of 8 bytes per row (TP group, dim 0)
    -> one Triton kernel per request: max key over the TP ranks (= largest value, lowest global
       id on ties), then the greedy branch of the stock rejection kernel on token ids.

Exactness. The stock greedy selection is "first maximal index of the row": tl.max(...,
return_indices=True) is tie-break-left inside each vocab block and tl.argmax over the block
maxima is tie-break-left too; torch.argmax on the gathered row agrees. Here the local kernel
keeps the first maximum of its contiguous shard, and the max over keys prefers the lowest
global id among equal values, because shards are contiguous id ranges and the low key word
is 0xFFFFFFFF - id. bf16 -> fp32 is exact, so comparing fp32 values matches the stock
comparisons. -0.0 is folded into +0.0 before encoding (the stock comparisons treat them as
equal; their bit patterns would not). +-inf rows behave as in the stock path (an all -inf
row selects id 0).

NaN. The stock result for a row with NaN depends on Triton's reduction tree, and the two stock
paths even disagree (the rejection kernels map a NaN block maximum to -inf, the plain sampler's
torch.argmax over block maxima prefers it). Here NaN is treated as -inf everywhere, so the id
is always in range and identical on every rank. Check mode reports NaN rows separately.

Eligibility (read only from host state that every TP rank holds identically, so all ranks
take the same branch and the collective sequence stays aligned): every request greedy
(temperature 0), no logits processing of any kind (penalties, logit bias, min_tokens,
bad words, thinking budget, min_p/top_k/top_p), no logprobs or logprob token ids, no
grammar bitmask this step, no NaN counting, no trace replay, no sampling-mask output, no
batch-sharded sampling, the stock RejectionSampler (no synthetic rates, no adaptive
verification), a draft-logits cache (if any) at least as wide as the vocab, and an lm_head
whose shards are unpadded contiguous ranges (no bias, scale 1, no soft cap). Anything
else runs the stock path unchanged.

Modes:
    VLLM_VOCAB_PARALLEL_ARGMAX=0      off (default; the hook is not even installed)
    VLLM_VOCAB_PARALLEL_ARGMAX=1      fast path when eligible
    VLLM_VOCAB_PARALLEL_ARGMAX=check  run the fast path AND the stock path on every eligible
                                      step, return the STOCK result, count mismatches on the
                                      device (read every LOG_EVERY steps; TP rank 0 logs)
    VLLM_VOCAB_PARALLEL_ARGMAX_LOG_EVERY=500     stats cadence (eligible + ineligible steps)
    VLLM_VOCAB_PARALLEL_ARGMAX_CHECK_SYNC=1      check mode: compare on the host every step and
                                                 log the first 20 mismatches in full (slow)

Credits: the value/index reduction is vLLM's own LogitsProcessor.get_top_tokens (vllm#34049,
zixi-qi; used upstream for drafts only). The target-side port, its tie-order argument and the
greedy-verify-on-ids idea follow knapcio's overlay/glm_target_argmax.py
(knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 @770d115, Apache-2.0); re-implemented here for MRv2 of
vLLM 0.29 with a single-int64 key, NaN folding and a device-side check mode.
"""
from __future__ import annotations

import hashlib
import inspect
import logging
import os

import numpy as np
import torch

logger = logging.getLogger("vllm.glm_fast")

_OFF = ("", "0", "off", "false", "no")
NO_LOGPROBS = -1
KEY_LOW = 0xFFFFFFFF
KEY_SHIFT = 1 << 32
BLOCK = 4096


def mode() -> str:
    raw = os.environ.get("VLLM_VOCAB_PARALLEL_ARGMAX", "0").strip().lower()
    if raw in _OFF:
        return "0"
    return "check" if raw == "check" else "1"


LOG_EVERY = int(os.environ.get("VLLM_VOCAB_PARALLEL_ARGMAX_LOG_EVERY", "500") or 0)
CHECK_SYNC = os.environ.get("VLLM_VOCAB_PARALLEL_ARGMAX_CHECK_SYNC", "0").strip().lower() not in _OFF


# ---------------------------------------------------------------------------------------------
# Pure tensor reference (CPU-testable; also the CUDA fallback when Triton is unavailable)
# ---------------------------------------------------------------------------------------------
def encode_keys(vals: torch.Tensor, global_idx: torch.Tensor) -> torch.Tensor:
    """fp32 values (NaN already folded) + int64 global ids -> int64 keys, ordered by
    (value, -id). -0.0 is folded into +0.0 first."""
    v = vals.to(torch.float32)
    v = torch.where(v == 0, torch.zeros_like(v), v)
    bits = v.view(torch.int32).to(torch.int64)
    ordered = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    return ordered * KEY_SHIFT + (KEY_LOW - global_idx.to(torch.int64))


def decode_ids(keys: torch.Tensor) -> torch.Tensor:
    return KEY_LOW - (keys & KEY_LOW)


def local_keys_torch(shard_logits: torch.Tensor, vocab_start: int) -> torch.Tensor:
    """[L, Vs] shard logits -> [L] int64 keys of each row's first maximum (NaN as -inf)."""
    x = shard_logits.to(torch.float32)
    x = torch.where(torch.isnan(x), torch.full_like(x, float("-inf")), x)
    idx = x.argmax(dim=-1)  # first maximal index (torch.argmax contract)
    val = x.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    return encode_keys(val, idx + int(vocab_start))


def reduce_keys(gathered: torch.Tensor, tp: int) -> torch.Tensor:
    """[tp * L] rank-major keys (all_gather dim 0) -> [L] int64 global argmax ids."""
    return decode_ids(gathered.view(tp, -1).max(dim=0).values)


def greedy_verify_torch(target: torch.Tensor, draft: torch.Tensor, cu: torch.Tensor,
                        num_reqs: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy branch of the stock rejection sampler, on token ids.

    target [L] int64: argmax of every logits row. draft [L]: input_ids at the logits rows (the
    draft checked at row j is draft[j + 1] of the same request). cu [num_reqs + 1]: cu_num_logits.
    Returns sampled [num_reqs, width] int64 and num_sampled [num_reqs] int32. Columns past
    num_sampled hold the argmax of row min(start + c, end - 1) (the stock tensor leaves them
    uninitialised; nothing reads them)."""
    cu_l = [int(c) for c in cu[: num_reqs + 1].tolist()]
    t = target.tolist()
    d = draft.tolist()
    sampled = torch.empty(num_reqs, width, dtype=torch.int64)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32)
    for r in range(num_reqs):
        s, e = cu_l[r], cu_l[r + 1]
        n = e - s - 1
        acc = 0
        while acc < n and d[s + acc + 1] >= 0 and t[s + acc] == d[s + acc + 1]:
            acc += 1
        for c in range(width):
            if c < acc:
                sampled[r, c] = d[s + c + 1]
            else:
                sampled[r, c] = t[min(s + c, e - 1)]
        num_sampled[r] = acc + 1
    return sampled.to(target.device), num_sampled.to(target.device)


# ---------------------------------------------------------------------------------------------
# Triton kernels (module level, so the JIT resolves the helper as a global)
# ---------------------------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _vp_local_key_kernel(logits_ptr, logits_stride, key_ptr, nan_ptr, V, vocab_start,
                             BLOCK_SIZE: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        best_v = tl.full((), float("-inf"), tl.float32)
        best_i = tl.zeros((), tl.int64)
        any_nan = tl.zeros((), tl.int32)
        for start in range(0, V, BLOCK_SIZE):
            offs = start + tl.arange(0, BLOCK_SIZE)
            x = tl.load(logits_ptr + row * logits_stride + offs, mask=offs < V,
                        other=float("-inf")).to(tl.float32)
            isn = x != x
            any_nan = tl.maximum(any_nan, tl.max(isn.to(tl.int32), axis=0))
            x = tl.where(isn, float("-inf"), x)
            v, i = tl.max(x, axis=0, return_indices=True, return_indices_tie_break_left=True)
            take = v > best_v  # strict: an equal maximum in a later block keeps the earlier id
            best_i = tl.where(take, start + i.to(tl.int64), best_i)
            best_v = tl.where(take, v, best_v)
        best_v = tl.where(best_v == 0.0, 0.0, best_v)  # fold -0.0 into +0.0
        bits = best_v.to(tl.int32, bitcast=True)
        ordered = tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF).to(tl.int64)
        low = tl.full((), 4294967295, tl.int64)
        key = ordered * (low + 1) + (low - (best_i + vocab_start))
        tl.store(key_ptr + row, key)
        tl.store(nan_ptr + row, any_nan)

    @triton.jit
    def _vp_row_argmax(keys_ptr, L, row, TP: tl.constexpr):
        best = tl.load(keys_ptr + row)
        for r in tl.static_range(1, TP):
            best = tl.maximum(best, tl.load(keys_ptr + r * L + row))
        low = tl.full((), 4294967295, tl.int64)
        return low - (best & low)

    @triton.jit
    def _vp_verify_kernel(keys_ptr, L, draft_ptr, cu_ptr, sampled_ptr, sampled_stride,
                          num_sampled_ptr, TP: tl.constexpr, WIDTH: tl.constexpr):
        # Mirrors the greedy branch of _rejection_kernel (+ _resample/_insert_resampled for
        # the bonus row) of the stock rejection sampler, on token ids.
        r = tl.program_id(0)
        start = tl.load(cu_ptr + r).to(tl.int64)
        end = tl.load(cu_ptr + r + 1).to(tl.int64)
        n = end - start - 1
        accepted = tl.zeros((), tl.int64)
        verifying = accepted == 0
        for i in range(0, n):
            if verifying:
                t = _vp_row_argmax(keys_ptr, L, start + i, TP)
                d = tl.load(draft_ptr + start + i + 1).to(tl.int64)
                ok = (t == d) & (d >= 0)
                verifying = ok
                accepted += ok.to(tl.int64)
        for c in tl.static_range(WIDTH):
            row = tl.minimum(start + c, end - 1)
            t = _vp_row_argmax(keys_ptr, L, row, TP)
            if c < accepted:
                t = tl.load(draft_ptr + start + c + 1).to(tl.int64)
            tl.store(sampled_ptr + r * sampled_stride + c, t)
        tl.store(num_sampled_ptr + r, (accepted + 1).to(tl.int32))


def _use_triton(t: torch.Tensor) -> bool:
    return HAS_TRITON and (t.is_cuda or os.environ.get("TRITON_INTERPRET", "0") == "1")


def local_keys(shard_logits: torch.Tensor, vocab_start: int) -> tuple[torch.Tensor, torch.Tensor]:
    """[L, Vs] -> ([L] int64 keys, [L] int32 row-has-NaN flags)."""
    if not _use_triton(shard_logits):
        nan = torch.isnan(shard_logits).any(dim=-1).to(torch.int32)
        return local_keys_torch(shard_logits, vocab_start), nan
    x = shard_logits if shard_logits.stride(-1) == 1 else shard_logits.contiguous()
    L, V = x.shape
    keys = torch.empty(L, dtype=torch.int64, device=x.device)
    nan = torch.empty(L, dtype=torch.int32, device=x.device)
    if L > 0:
        _vp_local_key_kernel[(L,)](x, x.stride(0), keys, nan, V, int(vocab_start), BLOCK_SIZE=BLOCK)
    return keys, nan


def greedy_verify(gathered: torch.Tensor, tp: int, draft: torch.Tensor, cu: torch.Tensor,
                  num_reqs: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    L = gathered.numel() // tp
    if not _use_triton(gathered):
        return greedy_verify_torch(reduce_keys(gathered, tp), draft, cu, num_reqs, width)
    sampled = torch.empty(num_reqs, width, dtype=torch.int64, device=gathered.device)
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=gathered.device)
    if num_reqs > 0:
        _vp_verify_kernel[(num_reqs,)](gathered.contiguous(), L, draft.contiguous(), cu.contiguous(),
                                 sampled, sampled.stride(0), num_sampled, TP=tp, WIDTH=width,
                                 num_warps=1)
    return sampled, num_sampled


def vocab_parallel_keys(shard_logits, vocab_start, tp, all_gather):
    """Local keys -> gathered [tp * L] keys (rank-major). `all_gather(t)` must be the TP group's
    dim-0 all-gather."""
    keys, nan = local_keys(shard_logits, vocab_start)
    gathered = keys if tp == 1 else all_gather(keys)
    return gathered, nan


# ---------------------------------------------------------------------------------------------
# Dispatch (rank-invariant: host request state + static config only)
# ---------------------------------------------------------------------------------------------
class Stats:
    fast = 0
    full = 0
    reasons: dict = {}
    checked_steps = 0
    checked_reqs = 0
    mismatch = 0
    nan_mismatch = 0
    nan_reqs = 0
    dev: dict = {}  # device-side counters (check mode)
    logged_mismatches = 0


def _reason(r: str):
    Stats.reasons[r] = Stats.reasons.get(r, 0) + 1
    return None


def resolve_head(model):
    """(lm_head, logits_processor) of the target, or a string saying why the path is off."""
    m = model
    for _ in range(6):
        if hasattr(m, "lm_head") and hasattr(m, "logits_processor"):
            break
        nxt = getattr(m, "language_model", None)
        if nxt is None and hasattr(m, "get_language_model"):
            try:
                nxt = m.get_language_model()
            except Exception:  # noqa: BLE001
                nxt = None
        if nxt is None or nxt is m:
            return "no lm_head/logits_processor found"
        m = nxt
    else:
        return "no lm_head/logits_processor found"
    lm_head, lp = m.lm_head, m.logits_processor
    try:
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        if type(lp) is not LogitsProcessor:
            return f"logits processor is {type(lp).__name__}"
    except Exception:  # noqa: BLE001 - CPU tests
        if type(lp).__name__ != "LogitsProcessor":
            return f"logits processor is {type(lp).__name__}"
    if getattr(lp, "logits_as_input", False) or lp.soft_cap is not None or lp.scale != 1.0:
        return "logits_as_input / soft_cap / scale != 1"
    if not getattr(lp, "use_all_gather", True):
        return "logits processor gathers to rank 0 only"
    si = getattr(lm_head, "shard_indices", None)
    if si is None or not hasattr(lp, "_apply_head"):
        return "lm_head has no shard_indices"
    if si.num_org_vocab_padding != 0 or si.num_added_elements_padded != 0:
        return "vocab padding or added vocab"
    if int(lm_head.num_embeddings_padded) != int(lp.org_vocab_size):
        return f"gathered width {lm_head.num_embeddings_padded} != vocab {lp.org_vocab_size}"
    if getattr(lm_head, "bias", None) is not None:
        return "lm_head bias"
    return (lm_head, lp)


def plan(runner, input_batch, grammar_output):
    """None -> stock path. Otherwise ("reject", width) or ("sampler", 1)."""
    if grammar_output is not None:
        return _reason("grammar")
    if getattr(runner, "batch_sharder", None) is not None:
        return _reason("batch-sharded sampling")
    num_reqs = int(input_batch.num_reqs)
    if num_reqs <= 0:
        return _reason("empty batch")
    sampler = getattr(runner, "sampler", None)
    if sampler is None or type(sampler).__name__ != "Sampler":
        return _reason("no / custom sampler")
    if getattr(sampler, "compute_nans", False) or getattr(sampler, "trace_replay_state", None) is not None \
            or getattr(sampler, "return_sampling_mask", False):
        return _reason("nan counting / trace replay / sampling mask")
    head = runner.__dict__.get("_glm_vp_head")
    if head is None:
        head = resolve_head(runner.model)
        runner.__dict__["_glm_vp_head"] = head
        if isinstance(head, str):
            logger.warning("glm_fast argmax: fast path off for this runner: %s", head)
        else:
            logger.info("glm_fast argmax: lm_head shard [%d:%d) of %d, tp %d",
                        int(head[0].shard_indices.org_vocab_start_index),
                        int(head[0].shard_indices.org_vocab_end_index),
                        int(head[1].org_vocab_size), int(head[0].tp_size))
    if isinstance(head, str):
        return _reason("head")
    idx_np = np.asarray(input_batch.idx_mapping_np)[:num_reqs]
    if idx_np.size == 0 or (idx_np < 0).any():
        return _reason("masked batch")
    st = sampler.sampling_states
    if not np.all(st.temperature.np[idx_np] == 0.0):
        return _reason("sampled rows")
    if np.any(sampler.needs_logits_processing[idx_np]):
        return _reason("logits processing")
    if st.max_num_logprobs(idx_np) != NO_LOGPROBS:
        return _reason("logprobs")
    if sampler.logprob_token_ids_state.max_num_token_ids(idx_np) > 0:
        return _reason("logprob token ids")
    if int(input_batch.num_draft_tokens) == 0 or runner.rejection_sampler is None:
        if int(input_batch.logits_indices.shape[0]) != num_reqs:
            return _reason("sampler rows != requests")
        return ("sampler", 1)
    rs = runner.rejection_sampler
    if type(rs).__name__ != "RejectionSampler" or rs.synthetic_conditional_rates is not None:
        return _reason("custom / synthetic rejection sampler")
    if getattr(rs, "enable_adaptive_verification", False):
        return _reason("adaptive verification")
    spec = getattr(runner, "speculator", None)
    dl = getattr(spec, "draft_logits", None) if spec is not None else None
    if dl is not None and dl.size(-1) < int(head[1].org_vocab_size):
        return _reason("draft vocab narrower than target")  # stock clamps the vocab to the draft's
    return ("reject", int(rs.num_speculative_steps) + 1)


def fast_sample(runner, hidden_states, input_batch, how):
    """The fast path. Returns (SamplerOutput, num_sampled, num_rejected, nan_flags)."""
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    lm_head, lp = runner.__dict__["_glm_vp_head"]
    h = hidden_states[input_batch.logits_indices]
    shard = lp._apply_head(lm_head, h, None)  # the identical call LogitsProcessor._get_logits makes
    tp = int(lm_head.tp_size)
    gathered, nan = vocab_parallel_keys(
        shard, int(lm_head.shard_indices.org_vocab_start_index), tp,
        lambda t: tensor_model_parallel_all_gather(t, dim=0))
    num_reqs = int(input_batch.num_reqs)
    kind, width = how
    if kind == "sampler":
        sampled = reduce_keys(gathered, tp).view(-1, 1)
        num_sampled = input_batch.seq_lens.new_ones(num_reqs)
    else:
        draft = input_batch.input_ids[input_batch.logits_indices]
        sampled, num_sampled = greedy_verify(gathered, tp, draft,
                                             input_batch.cu_num_logits[: num_reqs + 1], num_reqs, width)
    num_sampled, num_rejected = get_num_sampled_and_rejected(
        num_sampled, input_batch.seq_lens, input_batch.cu_num_logits, input_batch.idx_mapping,
        runner.sampler.req_states.prefill_len.gpu)
    out = SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None, num_nans=None,
                        num_sampled=num_sampled, num_rejected=num_rejected)
    return out, num_sampled, num_rejected, nan


def compare_device(fast, ref, input_batch):
    """Per-request mismatch flags on the device (no host sync): [num_reqs] bool, and per-request
    'some local logits row had NaN' flags."""
    fo, fns, fnr, nan = fast
    ro, rns, rnr = ref
    a, b = fo.sampled_token_ids, ro.sampled_token_ids
    w = min(a.shape[1], b.shape[1])
    ns = rns.to(torch.int64)
    live = torch.arange(w, device=a.device)[None, :] < ns[:, None]
    tok_bad = ((a[:, :w] != b[:, :w]) & live).any(dim=1)
    bad = tok_bad | (fns.to(torch.int64) != ns) | (fnr.to(torch.int64) != rnr.to(torch.int64))
    if a.shape[1] != b.shape[1]:
        bad = torch.ones_like(bad)
    num_reqs = bad.shape[0]
    cu = input_batch.cu_num_logits[: num_reqs + 1].to(torch.int64)
    rows = torch.arange(nan.shape[0], device=nan.device)
    req_of_row = torch.searchsorted(cu[1:].contiguous(), rows, right=True).clamp_(max=num_reqs - 1)
    req_nan = torch.zeros(num_reqs, dtype=torch.int32, device=nan.device)
    req_nan.scatter_reduce_(0, req_of_row, nan.to(torch.int32), reduce="amax")
    return bad, req_nan.bool()


def source_digest(fn) -> str:
    try:
        return hashlib.sha256(inspect.getsource(inspect.unwrap(fn)).encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


# sha256[:16] of GPUModelRunner.sample in verify_cap_overlay/.../model_runner.py as of 2026-10-01.
# A different digest means sample() changed (e.g. step 4); the hook still installs, and check
# mode is what proves the fast path still matches it.
KNOWN_SAMPLE_DIGESTS = {"021198f1cec14aee"}


def install(module) -> None:
    cls = module.GPUModelRunner
    if getattr(cls, "_glm_fast_argmax", False):
        return
    orig_sample = cls.sample
    params = list(inspect.signature(inspect.unwrap(orig_sample)).parameters)
    if params[:4] != ["self", "hidden_states", "input_batch", "grammar_output"]:
        raise RuntimeError(f"glm_fast argmax: GPUModelRunner.sample{params} has an unexpected signature")
    digest = source_digest(orig_sample)
    if digest not in KNOWN_SAMPLE_DIGESTS:
        logger.warning("glm_fast argmax: GPUModelRunner.sample source digest %s is not the reviewed one; "
                       "validate with VLLM_VOCAB_PARALLEL_ARGMAX=check before enabling", digest)

    def _rank0() -> bool:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            return get_tensor_model_parallel_rank() == 0
        except Exception:  # noqa: BLE001
            return True

    def _log_stats(m: str) -> None:
        if not _rank0():
            return
        msg = f"glm_fast argmax: steps fast={Stats.fast} stock={Stats.full} reasons={Stats.reasons}"
        if m == "check":
            if Stats.dev:
                Stats.mismatch = int(Stats.dev["bad"].item())
                Stats.nan_mismatch = int(Stats.dev["nan_bad"].item())
                Stats.nan_reqs = int(Stats.dev["nan_reqs"].item())
            msg += (f" checked_steps={Stats.checked_steps} checked_reqs={Stats.checked_reqs}"
                    f" mismatches={Stats.mismatch - Stats.nan_mismatch} nan_req_mismatches={Stats.nan_mismatch}"
                    f" nan_reqs={Stats.nan_reqs}")
        logger.info(msg)

    def sample(self, hidden_states, input_batch, grammar_output, *args, **kwargs):
        m = mode()
        if m == "0" or args or kwargs:
            return orig_sample(self, hidden_states, input_batch, grammar_output, *args, **kwargs)
        how = plan(self, input_batch, grammar_output)
        if how is None:
            Stats.full += 1
            res = orig_sample(self, hidden_states, input_batch, grammar_output)
        elif m == "check":
            fast = fast_sample(self, hidden_states, input_batch, how)
            res = orig_sample(self, hidden_states, input_batch, grammar_output)
            bad, req_nan = compare_device(fast, res, input_batch)
            dev = Stats.dev
            if not dev:
                z = lambda: torch.zeros((), dtype=torch.int64, device=bad.device)  # noqa: E731
                dev.update(bad=z(), nan_bad=z(), nan_reqs=z())
            dev["bad"] += bad.sum()
            dev["nan_bad"] += (bad & req_nan).sum()
            dev["nan_reqs"] += req_nan.sum()
            Stats.checked_steps += 1
            Stats.checked_reqs += int(input_batch.num_reqs)
            Stats.fast += 1
            if CHECK_SYNC and bool(bad.any()) and Stats.logged_mismatches < 20 and _rank0():
                Stats.logged_mismatches += 1
                logger.warning("glm_fast argmax MISMATCH: fast sampled=%s num_sampled=%s | stock sampled=%s "
                               "num_sampled=%s | nan=%s", fast[0].sampled_token_ids.tolist(), fast[1].tolist(),
                               res[0].sampled_token_ids.tolist(), res[1].tolist(), req_nan.tolist())
        else:
            Stats.fast += 1
            res = fast_sample(self, hidden_states, input_batch, how)[:3]
        n = Stats.fast + Stats.full
        if LOG_EVERY and n % LOG_EVERY == 0:
            _log_stats(m)
        return res

    sample.__wrapped__ = orig_sample
    sample.__glm_fast__ = True
    cls.sample = sample
    cls._glm_fast_argmax = True
    logger.info("glm_fast argmax: GPUModelRunner.sample hooked (mode=%s, sample digest %s)", mode(), digest)
