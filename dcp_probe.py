#!/usr/bin/env python3
"""Probe and benchmark a GLM-5.3 vLLM endpoint: warm-up sweep, determinism
hashes, streaming decode benchmarks with /metrics deltas, long-context probe.

Stdlib only. Usage:
  dcp_probe.py --base http://spark-06c4.local:8000 --label dcp1-prod --out results/x \
      [--phase warmup|determinism|bench|longctx|all] [--longctx-tokens 200000]

Every request is greedy (temperature 0) with thinking off, so two stacks that
compute the same thing produce the same text; hashes are recorded per label so
labels can be diffed afterwards.
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request

WORDS = (
    "system kernel memory cache tensor layer attention token block rank shard "
    "context window decode prefill schedule buffer stream latency bandwidth "
    "matrix vector gradient softmax index merge gather scatter reduce rotate "
    "north river stone amber quiet signal orbit lantern meadow copper harbor "
    "violet thunder canvas marble spiral fossil glacier ember prism velvet"
).split()


def make_prompt_words(n_words: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def metrics(base: str) -> dict:
    txt = urllib.request.urlopen(base + "/metrics", timeout=20).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, val = line.rpartition(" ")
        key = name.split("{", 1)[0]
        try:
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            pass
    return out


def chat(base, messages, max_tokens, timeout, stream=False, thinking=False):
    body = {
        "model": "glm-5.3",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1.0,
        "chat_template_kwargs": {"enable_thinking": thinking},
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        t1 = time.monotonic()
        content = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage", {})
        return {
            "content": content,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "wall_s": t1 - t0,
        }
    # streaming: measure TTFT and pure decode rate
    content = []
    t_first = None
    t_last = None
    n_chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                delta = ch.get("delta", {}).get("content")
                if delta:
                    now = time.monotonic()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    n_chunks += 1
                    content.append(delta)
    t_end = time.monotonic()
    return {
        "content": "".join(content),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "wall_s": t_end - t0,
        "ttft_s": (t_first - t0) if t_first else None,
        "decode_s": (t_last - t_first) if (t_first and t_last) else None,
        "chunks": n_chunks,
    }


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def phase_warmup(base, out):
    """Exercise a spread of prompt lengths so any inference-time JIT happens
    now, on fresh memory, while we watch."""
    rows = []
    # 38500 words (~40k tokens) crosses the 32k global length where the
    # indexer's logits budget starts splitting a chunk's query; the split
    # path has its own kernel specializations.
    for n_words in (1, 4, 13, 70, 250, 700, 1400, 2800, 38500):
        prompt = make_prompt_words(n_words, seed=n_words) + "\nReply with the single word OK."
        r = chat(base, [{"role": "user", "content": prompt}], 24, timeout=1200)
        rows.append({"n_words": n_words, "prompt_tokens": r["prompt_tokens"],
                     "completion_tokens": r["completion_tokens"], "wall_s": round(r["wall_s"], 2),
                     "content": r["content"][:60]})
        print(f"  warmup {n_words:5d} words -> {r['prompt_tokens']} tok, {r['wall_s']:.1f}s: {r['content'][:40]!r}", flush=True)
    json.dump(rows, open(os.path.join(out, "warmup.json"), "w"), indent=1)


def determinism_prompts():
    """The upstream PR's matrix: 512 uncached, 1536 uncached, 1024 cached + 1536."""
    p512 = make_prompt_words(380, seed=512)
    p1536 = make_prompt_words(1150, seed=1536)
    shared = make_prompt_words(770, seed=1024)
    tail = make_prompt_words(1150, seed=2560)
    q = "\n\nSummarize the text above in exactly three sentences, then list the five most frequent words."
    return [
        ("u512", [{"role": "user", "content": p512 + q}]),
        ("u1536", [{"role": "user", "content": p1536 + q}]),
        # First request seeds the prefix cache with `shared`; the third reuses it.
        ("c1024-seed", [{"role": "user", "content": shared + q}]),
        ("c1024+u1536", [{"role": "user", "content": shared + tail + q}]),
    ]


def phase_determinism(base, out, label):
    rows = []
    for name, msgs in determinism_prompts():
        r = chat(base, msgs, 128, timeout=900)
        row = {"name": name, "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
               "sha": sha(r["content"]), "head": r["content"][:120], "wall_s": round(r["wall_s"], 2)}
        rows.append(row)
        print(f"  {name:12s} {r['prompt_tokens']:6d}->{r['completion_tokens']:4d} tok  sha={row['sha']}  {r['wall_s']:.1f}s  {r['content'][:50]!r}", flush=True)
    json.dump(rows, open(os.path.join(out, "determinism.json"), "w"), indent=1)
    return rows


BENCH = [
    ("count100", "Count from 1 to 100, comma-separated, nothing else.", 400),
    ("prose", "Explain, in about 300 words, how a tensor-parallel transformer splits attention heads across GPUs and why the all-reduce after the output projection is needed.", 450),
    ("code", "Write a Python function `merge_intervals(intervals)` that merges overlapping [start, end] intervals and returns them sorted. Include a short docstring and three asserts as tests. Output only code.", 400),
]


def phase_bench(base, out, label, reps=2):
    rows = []
    for name, prompt, max_tokens in BENCH:
        for rep in range(reps):
            m0 = metrics(base)
            r = chat(base, [{"role": "user", "content": prompt}], max_tokens, timeout=900, stream=True)
            m1 = metrics(base)
            d = {k: m1.get(k, 0) - m0.get(k, 0) for k in
                 ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total",
                  "vllm:generation_tokens_total")}
            drafts = d["vllm:spec_decode_num_drafts_total"]
            acc = d["vllm:spec_decode_num_accepted_tokens_total"]
            gen = r["completion_tokens"] or d["vllm:generation_tokens_total"]
            decode_tps = (gen - 1) / r["decode_s"] if r.get("decode_s") else None
            row = {"name": name, "rep": rep, "prompt_tokens": r["prompt_tokens"], "gen_tokens": gen,
                   "ttft_s": round(r["ttft_s"], 3) if r.get("ttft_s") else None,
                   "decode_s": round(r["decode_s"], 3) if r.get("decode_s") else None,
                   "decode_tok_s": round(decode_tps, 2) if decode_tps else None,
                   "wall_tok_s": round(gen / r["wall_s"], 2),
                   "drafts": drafts, "accepted": acc,
                   "accepted_per_cycle": round(1 + acc / drafts, 3) if drafts else None,
                   "cycle_ms": round(1000 * r["decode_s"] / drafts, 1) if (drafts and r.get("decode_s")) else None,
                   "sha": sha(r["content"]), "head": r["content"][:80]}
            rows.append(row)
            print(f"  {name:9s} rep{rep} gen={gen:4d} ttft={row['ttft_s']}s decode={row['decode_tok_s']} tok/s "
                  f"acc/cycle={row['accepted_per_cycle']} cycle={row['cycle_ms']}ms sha={row['sha']}", flush=True)
    json.dump(rows, open(os.path.join(out, "bench.json"), "w"), indent=1)
    return rows


def phase_concurrent(base, out, label):
    """Mixed batches: decodes and prefill chunks sharing a forward step (the
    sharded verify path and the prefill path together). Four requests in
    flight with staggered starts; every one must come back coherent."""
    import threading
    jobs = [
        ("gen-prose", [{"role": "user", "content": BENCH[1][1]}], 350),
        ("pf-3k", [{"role": "user", "content": make_prompt_words(2800, seed=31)
                    + "\n\nSummarize the text above in two sentences."}], 96),
        ("pf-6k", [{"role": "user", "content": make_prompt_words(5600, seed=32)
                    + "\n\nList the ten most frequent words above, comma-separated."}], 96),
        ("gen-code", [{"role": "user", "content": BENCH[2][1]}], 350),
    ]
    results = {}

    def run(name, msgs, max_tokens, delay):
        time.sleep(delay)
        try:
            results[name] = chat(base, msgs, max_tokens, timeout=1800, stream=True)
        except Exception as e:  # noqa: BLE001
            results[name] = {"error": repr(e)}

    m0 = metrics(base)
    t0 = time.monotonic()
    threads = [threading.Thread(target=run, args=(n, m, mt, 2.0 * i)) for i, (n, m, mt) in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    m1 = metrics(base)
    rows = []
    for name, _, _ in jobs:
        r = results.get(name, {"error": "missing"})
        if "error" in r:
            rows.append({"name": name, "error": r["error"]})
            print(f"  {name:9s} ERROR {r['error']}", flush=True)
            continue
        gen = r["completion_tokens"] or 0
        dec = (gen - 1) / r["decode_s"] if r.get("decode_s") else None
        rows.append({"name": name, "prompt_tokens": r["prompt_tokens"], "gen_tokens": gen,
                     "ttft_s": round(r["ttft_s"], 3) if r.get("ttft_s") else None,
                     "decode_tok_s": round(dec, 2) if dec else None,
                     "sha": sha(r["content"]), "head": r["content"][:100]})
        print(f"  {name:9s} {r['prompt_tokens']:6d}->{gen:4d} tok ttft={rows[-1]['ttft_s']}s "
              f"decode={rows[-1]['decode_tok_s']} tok/s  {r['content'][:60]!r}", flush=True)
    drafts = m1.get("vllm:spec_decode_num_drafts_total", 0) - m0.get("vllm:spec_decode_num_drafts_total", 0)
    acc = m1.get("vllm:spec_decode_num_accepted_tokens_total", 0) - m0.get("vllm:spec_decode_num_accepted_tokens_total", 0)
    summary = {"wall_s": round(time.monotonic() - t0, 1), "drafts": drafts, "accepted": acc,
               "accepted_per_cycle": round(1 + acc / drafts, 3) if drafts else None}
    print(f"  concurrent window: {summary}", flush=True)
    json.dump({"rows": rows, "summary": summary}, open(os.path.join(out, "concurrent.json"), "w"), indent=1)
    return rows


def phase_offload(base, out, label, n_tokens=100000, evict_tokens=3, warm=True, seed_base=0):
    """NVMe tier validation (docs/NVME-DESIGN.md section 4):
      cold   - fresh prompt: prefill, then the tier stores it (timed TTFT)
      warm   - same prefix, new question: GPU prefix hit (control)
      evict  - `evict_tokens` other prompts of the same size push it out of
               the GPU pool (and past the small CPU tier)
      reload - same prefix again: must come back from the tier, not prefill
    Node-side file counts are collected by offload_checks.sh around this."""
    rows = {}
    def ask(prefix, q, mt=8):
        m0 = metrics(base)
        r = chat(base, [{"role": "user", "content": prefix + q}], mt, timeout=7200, stream=True)
        m1 = metrics(base)
        return {"prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["ttft_s"] or 0, 2),
                "content": (r["content"] or "")[:40],
                "prefix_hits": m1.get("vllm:prefix_cache_hits_total", 0) - m0.get("vllm:prefix_cache_hits_total", 0),
                "prefix_queries": m1.get("vllm:prefix_cache_queries_total", 0) - m0.get("vllm:prefix_cache_queries_total", 0),
                "ext_hits": m1.get("vllm:external_prefix_cache_hits_total", 0) - m0.get("vllm:external_prefix_cache_hits_total", 0),
                "ext_queries": m1.get("vllm:external_prefix_cache_queries_total", 0) - m0.get("vllm:external_prefix_cache_queries_total", 0)}
    prefix = make_prompt_words(int(n_tokens / 1.04), seed=4242 + seed_base)
    rows["cold"] = ask(prefix, "\n\nReply with the single word OK.")
    print(f"  cold   {rows['cold']}", flush=True)
    time.sleep(20)   # let the tier finish storing (async, after the request)
    if warm:
        # A GPU prefix hit re-presents the whole prompt to the connector in
        # one step; --no-warm keeps the prefix stored exactly once so the
        # reload proves the first-time store alone (see docs/NVME-DESIGN.md).
        rows["warm"] = ask(prefix, "\n\nHow many words are above, roughly? One number.", 16)
        print(f"  warm   {rows['warm']}", flush=True)
    for i in range(evict_tokens):
        other = make_prompt_words(int(n_tokens / 1.04), seed=5000 + seed_base + i)
        r = ask(other, "\n\nReply with the single word OK.")
        rows[f"evict{i}"] = r
        print(f"  evict{i} {r}", flush=True)
    time.sleep(20)
    rows["reload"] = ask(prefix, "\n\nReply with the single word OK.")
    print(f"  reload {rows['reload']}", flush=True)
    json.dump(rows, open(os.path.join(out, "offload.json"), "w"), indent=1)
    return rows


def phase_reload(base, out, label, n_tokens=100000, seed=4242, tag="reload-after-restart"):
    """After an engine restart: re-send the offload phase's cold prefix. With
    the NVMe tier the TTFT must be a reload, not a prefill."""
    prefix = make_prompt_words(int(n_tokens / 1.04), seed=seed)
    m0 = metrics(base)
    r = chat(base, [{"role": "user", "content": prefix + "\n\nReply with the single word OK."}], 8, timeout=7200, stream=True)
    m1 = metrics(base)
    row = {"prompt_tokens": r["prompt_tokens"], "ttft_s": round(r["ttft_s"] or 0, 2), "content": (r["content"] or "")[:40],
           "ext_hits": m1.get("vllm:external_prefix_cache_hits_total", 0) - m0.get("vllm:external_prefix_cache_hits_total", 0),
           "ext_queries": m1.get("vllm:external_prefix_cache_queries_total", 0) - m0.get("vllm:external_prefix_cache_queries_total", 0)}
    print(f"  {tag} {row}", flush=True)
    json.dump(row, open(os.path.join(out, f"{tag}.json"), "w"), indent=1)
    return row


def phase_longctx(base, out, n_tokens):
    """One very long prompt: proves the context budget and measures TTFT."""
    n_words = int(n_tokens / 1.04)  # measured: 88888 words -> 92366 tokens
    prompt = make_prompt_words(n_words, seed=777)
    prompt += "\n\nThe text above is a long random word list. Reply with the single word OK and nothing else."
    m0 = metrics(base)
    r = chat(base, [{"role": "user", "content": prompt}], 16, timeout=7200, stream=True)
    m1 = metrics(base)
    row = {"requested_tokens": n_tokens, "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
           "ttft_s": round(r["ttft_s"], 1) if r.get("ttft_s") else None, "wall_s": round(r["wall_s"], 1),
           "content": r["content"][:60],
           "kv_usage_after": m1.get("vllm:kv_cache_usage_perc")}
    print(f"  longctx {row['prompt_tokens']} tok: ttft={row['ttft_s']}s wall={row['wall_s']}s -> {r['content'][:30]!r}", flush=True)
    json.dump(row, open(os.path.join(out, f"longctx-{n_tokens}.json"), "w"), indent=1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", default="all", choices=["warmup", "determinism", "bench", "concurrent", "longctx", "offload", "reload", "all"])
    ap.add_argument("--longctx-tokens", type=int, default=200000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--offload-tokens", type=int, default=100000)
    ap.add_argument("--no-warm", action="store_true", help="offload phase without the GPU-hit control step")
    ap.add_argument("--seed-base", type=int, default=0, help="offset every offload/reload prompt seed (fresh prefixes)")
    ap.add_argument("--reload-seed", type=int, default=None, help="reload phase: prompt seed (default: the cold prefix)")
    ap.add_argument("--reload-tag", default="reload-after-restart", help="reload phase: result file / log tag")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f"== {a.label} @ {a.base} phase={a.phase}", flush=True)
    if a.phase in ("warmup", "all"):
        phase_warmup(a.base, a.out)
    if a.phase in ("determinism", "all"):
        phase_determinism(a.base, a.out, a.label)
    if a.phase in ("bench", "all"):
        phase_bench(a.base, a.out, a.label, reps=a.reps)
    if a.phase == "concurrent":
        phase_concurrent(a.base, a.out, a.label)
    if a.phase == "reload":
        phase_reload(a.base, a.out, a.label, a.offload_tokens,
                     seed=(a.reload_seed if a.reload_seed is not None else 4242 + a.seed_base), tag=a.reload_tag)
    if a.phase == "offload":
        phase_offload(a.base, a.out, a.label, a.offload_tokens, warm=not a.no_warm, seed_base=a.seed_base)
    if a.phase == "longctx":
        phase_longctx(a.base, a.out, a.longctx_tokens)
    print("== done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
