#!/usr/bin/env python3
"""Prefill benchmark (results/prefill-item2-20261006/REPORT.md; stdlib only, runs from any host that reaches the endpoint).

Every request carries a unique ``cache_salt`` (part of the first block's hash, so neither the GPU
prefix cache nor the NVMe tier can serve it) unless a phase reuses one on purpose. Each request waits
for an idle endpoint, and a request that overlapped other traffic is marked ``contaminated``.

Phases (--phases, comma separated):
  ttft     uncached prompts of --sizes tokens (nonce + corpus text), max_tokens 1: client TTFT and
           server prefill seconds -> prefill tok/s.
  quality  --quality-sizes fixed prompts (same token ids in every boot), each prefilled fresh
           --quality-runs times (distinct salts): 256 greedy tokens as token ids plus the first
           token's top-20 logprobs. ``compare`` judges B against A's A/A envelope.
  profile  for each --profile-ctx: warm P (ctx - 2048 tokens) with salt S, then /start_profile,
           P + a fresh 2048-token tail with salt S (exactly one prefill chunk at full context),
           /stop_profile; fetches each rank's trace from --prof-host-dir and buckets GPU kernels.

usage:
  prefill_bench.py run LABEL [--phases ttft,quality,profile] [--prof-host-dir /var/tmp/models/prof-X]
  prefill_bench.py compare A/quality.json B/quality.json
  prefill_bench.py analyze DIR_WITH_TRACES
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HOSTS = ["spark-06c4", "spark-365c", "spark-ddbf", "spark-a218"]
CORPUS = REPO / "results" / "nvme-kvtier-20260928" / "corpus-tokens.json"
M = re.compile(r'^(vllm:[a-zA-Z_:]+)(\{[^}]*\})?\s+([-+0-9.eE]+)$')
WANT = ("num_requests_running", "num_requests_waiting", "request_success_total",
        "request_prefill_time_seconds_sum", "prompt_tokens_total", "prefix_cache_hits_total",
        "prefix_cache_queries_total", "external_prefix_cache_hits_total")


def post(base, path, body=None, timeout=3600):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(base + path, data=data, method="POST",
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw and raw[:1] in b"{[" else {}


def scrape(base):
    out = {}
    for line in urllib.request.urlopen(base + "/metrics", timeout=30).read().decode().splitlines():
        m = M.match(line)
        if m:
            k = m.group(1).split(":", 1)[1]
            if k in WANT:
                out[k] = out.get(k, 0.0) + float(m.group(3))
    return out


def wait_idle(base, limit=3600):
    t0 = time.time()
    while time.time() - t0 < limit:
        s = scrape(base)
        if s.get("num_requests_running", 0) == 0 and s.get("num_requests_waiting", 0) == 0:
            return s
        time.sleep(1.0)
    raise RuntimeError("endpoint never went idle")


def complete(base, model, ids, max_tokens, salt, logprobs=None):
    """Streamed /v1/completions with server-metric deltas; returns timings, token ids, logprobs."""
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "temperature": 0.0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True},
            "cache_salt": salt, "return_tokens_as_token_ids": True}
    if logprobs:
        body["logprobs"] = logprobs
    a = wait_idle(base)
    req = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(),
                                 {"content-type": "application/json"})
    t0 = time.time(); first = None; toks = []; top = []; usage = None
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                lp = ch.get("logprobs") or {}
                if ch.get("text") or lp.get("tokens"):
                    first = first or time.time()
                toks += lp.get("tokens") or []
                top += lp.get("top_logprobs") or []
    t1 = time.time()
    time.sleep(0.5)
    b = scrape(base)
    d = {k: b.get(k, 0) - a.get(k, 0) for k in WANT}
    return {"prompt_len": len(ids), "ttft_s": (first - t0) if first else None, "wall_s": t1 - t0,
            "prefill_s": d["request_prefill_time_seconds_sum"], "usage": usage,
            "prefix_hit_tokens": d["prefix_cache_hits_total"],
            "external_hit_tokens": d["external_prefix_cache_hits_total"],
            "contaminated": d["request_success_total"] != 1, "tokens": toks, "top_logprobs": top}


class Prompts:
    def __init__(self, base, model, seed):
        self.base, self.model = base, model
        self.corpus = json.loads(CORPUS.read_text())
        self.rng = random.Random(seed)

    def tok(self, s):
        return post(self.base, "/tokenize", {"model": self.model, "prompt": s,
                                             "add_special_tokens": False})["tokens"]

    def body(self, n, rng=None):
        rng = rng or self.rng
        out = []
        while len(out) < n:
            s = rng.randrange(0, len(self.corpus) - 4096)
            out += self.corpus[s:s + rng.randrange(512, 4096)]
        return out[:n]

    def context(self, tag, n, rng=None):
        rng = rng or self.rng
        nn = self.tok(f"### session {tag} {rng.getrandbits(64):016x}\n")
        return nn + self.body(n - len(nn), rng)


def phase_ttft(a, P, log):
    rows = []
    for n in a.sizes:
        for rep in range(a.reps):
            ids = P.context(f"ttft-{n}-{rep}-{uuid.uuid4().hex[:8]}", n)
            r = complete(a.base, a.model, ids, 1, f"ttft-{uuid.uuid4().hex}")
            r["tok_s_server"] = n / r["prefill_s"] if r["prefill_s"] else None
            r["tok_s_client"] = n / r["ttft_s"] if r["ttft_s"] else None
            del r["tokens"], r["top_logprobs"]
            rows.append(r)
            log(f"ttft n={n} rep={rep}: ttft {r['ttft_s']:.2f}s prefill {r['prefill_s']:.2f}s "
                f"-> {r['tok_s_server']:.0f} tok/s (hits {r['prefix_hit_tokens']:.0f}/{r['external_hit_tokens']:.0f})"
                + (" CONTAMINATED" if r["contaminated"] else ""))
    summ = {}
    for n in a.sizes:
        xs = [r for r in rows if r["prompt_len"] == n and not r["contaminated"]]
        summ[n] = {"n": len(xs), "prefill_s_median": statistics.median(r["prefill_s"] for r in xs),
                   "ttft_s_median": statistics.median(r["ttft_s"] for r in xs),
                   "tok_s_median": statistics.median(r["tok_s_server"] for r in xs)} if xs else {"n": 0}
        log(f"ttft summary n={n}: {summ[n]}")
    return {"rows": rows, "summary": summ}


def phase_quality(a, P, log):
    out = []
    for n in a.quality_sizes:
        ids = P.context(f"quality-{n}", n, random.Random(7919 * n))  # same ids in every boot
        for run in range(a.quality_runs):
            r = complete(a.base, a.model, ids, a.quality_tokens, f"q-{a.label}-{run}-{uuid.uuid4().hex}",
                         logprobs=20)
            out.append({"size": n, "run": run, "prompt_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16], "tokens": r["tokens"],
                        "first_top": r["top_logprobs"][0] if r["top_logprobs"] else None,
                        "prefill_s": r["prefill_s"], "hits": r["prefix_hit_tokens"] + r["external_hit_tokens"],
                        "contaminated": r["contaminated"]})
            log(f"quality n={n} run={run}: {len(r['tokens'])} tokens, prefill {r['prefill_s']:.1f}s, "
                f"hits {out[-1]['hits']:.0f}" + (" CONTAMINATED" if r["contaminated"] else ""))
    return out


def first_div(x, y):
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def top_delta(x, y):
    if not x or not y:
        return None
    common = set(x) & set(y)
    return {"max_abs": max((abs(x[k] - y[k]) for k in common), default=None),
            "same_top1": max(x, key=x.get) == max(y, key=y.get), "overlap": len(common)}


def compare(pa, pb):
    A = [r for r in json.load(open(pa)) if not r["contaminated"]]
    B = [r for r in json.load(open(pb)) if not r["contaminated"]]
    res = {}
    for n in sorted({r["size"] for r in A}):
        ra = [r for r in A if r["size"] == n]
        rb = [r for r in B if r["size"] == n]
        aa = [(first_div(x["tokens"], y["tokens"]), top_delta(x["first_top"], y["first_top"]))
              for i, x in enumerate(ra) for y in ra[i + 1:]]
        ab = [(first_div(x["tokens"], y["tokens"]), top_delta(x["first_top"], y["first_top"]))
              for x in ra for y in rb]
        res[n] = {"aa": aa, "ab": ab}
        print(f"n={n}: A/A first divergence {[d for d, _ in aa]} max|dlogp| "
              f"{[t and t['max_abs'] for _, t in aa]}")
        print(f"       A/B first divergence {[d for d, _ in ab]} max|dlogp| "
              f"{[t and t['max_abs'] for _, t in ab]}")
    return res


# ------------------------------------------------------------------------------- profile
def bucket(name):
    n = name.lower()
    if "nccldevkernel_allreduce" in n: return "nccl all-reduce (TP)"
    if "nccldevkernel_allgather" in n: return "nccl all-gather"
    if "nccldevkernel_reducescatter" in n: return "nccl reduce-scatter"
    if "nccl" in n: return "nccl other"
    if "oneshot" in n or "allgather_cute" in n or "roce" in n or "_ring" in n: return "rdma (rocenante/ring)"
    if "marlin_moe" in n or "moe_wna16" in n: return "moe routed (marlin)"
    if "marlin" in n: return "dense marlin"
    if "moe_align" in n or "grouped_topk" in n or "count_and_sort" in n or "moe_sum" in n: return "moe glue"
    if "mqa_logits" in n or "paged_mqa" in n: return "indexer logits"
    if "top_k_per_row" in n or "topk" in n or "radix" in n or "top_k" in n: return "top-k / selection"
    if "sparse" in n or "flash_mla" in n or "mla" in n or "flashinfer" in n: return "sparse mla / attention"
    if "gemm" in n or "cutlass" in n or "cublas" in n or "nvjet" in n or "sm90" in n or "sm100" in n: return "dense gemm (bf16/fp8)"
    if "norm" in n or "rope" in n or "silu" in n or "act_and_mul" in n: return "norm/rope/act"
    if "quant" in n or "fp8" in n: return "quantize"
    if "copy" in n or "elementwise" in n or "cat" in n or "index" in n or "gather" in n or "scatter" in n: return "copies/indexing"
    return "other"


def summarize(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    k = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    if not k:
        return {"error": "no kernel events"}
    t0 = min(e["ts"] for e in k); t1 = max(e["ts"] + e["dur"] for e in k)
    by = collections.defaultdict(lambda: {"ms": 0.0, "count": 0, "durs": []})
    names = collections.defaultdict(lambda: [0.0, 0])
    for e in k:
        b = by[bucket(e["name"])]
        b["ms"] += e["dur"] / 1e3; b["count"] += 1; b["durs"].append(e["dur"])
        names[e["name"][:110]][0] += e["dur"] / 1e3; names[e["name"][:110]][1] += 1
    busy = sum(b["ms"] for b in by.values())
    buckets = {kk: {"ms": round(v["ms"], 1), "count": v["count"], "share_of_wall": round(v["ms"] / ((t1 - t0) / 1e3), 4),
                    "median_us": round(statistics.median(v["durs"]), 1)}
               for kk, v in sorted(by.items(), key=lambda kv: -kv[1]["ms"])}
    top = [{"name": nm, "ms": round(v[0], 2), "count": v[1]} for nm, v in sorted(names.items(), key=lambda kv: -kv[1][0])[:40]]
    return {"wall_ms": round((t1 - t0) / 1e3, 1), "gpu_busy_ms": round(busy, 1), "buckets": buckets, "top": top}


def analyze(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "**", "*.json.gz"), recursive=True)):
        s = summarize(p)
        out[os.path.relpath(p, d)] = s
        print(f"== {os.path.relpath(p, d)}: wall {s.get('wall_ms')} ms busy {s.get('gpu_busy_ms')} ms")
        for b, row in list(s.get("buckets", {}).items())[:14]:
            print(f"   {b:28s} {row['ms']:8.1f} ms {100 * row['share_of_wall']:5.1f}%  n={row['count']:6d} "
                  f"med {row['median_us']:8.1f} us")
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    return out


def phase_profile(a, P, log, outdir):
    res = []
    for ctx in a.profile_ctx:
        tag = f"prof-{ctx}-{uuid.uuid4().hex[:8]}"
        n_p = max(1024, (ctx - 2048) // 1024 * 1024)
        rng = random.Random(tag)
        p_ids = P.context(tag, n_p, rng)
        tail = P.body(2048, random.Random(tag + "-tail"))
        salt = f"{tag}-{uuid.uuid4().hex}"
        w = complete(a.base, a.model, p_ids, 1, salt)
        log(f"profile ctx={ctx}: warm P={n_p} prefill {w['prefill_s']:.1f}s")
        marker = time.time()
        post(a.base, "/start_profile")
        r = complete(a.base, a.model, p_ids + tail, 1, salt)
        post(a.base, "/stop_profile")
        log(f"profile ctx={ctx}: chunk prefill {r['prefill_s']:.2f}s, hits {r['prefix_hit_tokens']:.0f}/{n_p} "
            f"external {r['external_hit_tokens']:.0f}")
        dest = outdir / f"profile-ctx{ctx}"
        dest.mkdir(parents=True, exist_ok=True)
        for _ in range(120):  # traces are written asynchronously after stop
            time.sleep(5)
            got = 0
            for h in HOSTS:
                q = subprocess.run(["ssh", "-o", "BatchMode=yes", f"{h}.local",
                                    f"find {a.prof_host_dir} -name '*.json.gz' -newermt @{int(marker)} | wc -l"],
                                   capture_output=True, text=True)
                got += int((q.stdout.strip() or "0")) > 0
            if got == len(HOSTS):
                break
        time.sleep(10)
        for h in HOSTS:
            subprocess.run(["ssh", "-o", "BatchMode=yes", f"{h}.local",
                            f"cd {a.prof_host_dir} && find . -name '*.json.gz' -newermt @{int(marker)} | "
                            f"tar -cf - -T - "], stdout=open(dest / f"{h}.tar", "wb"), check=False)
            subprocess.run(["tar", "-xf", str(dest / f"{h}.tar"), "-C", str(dest)], check=False)
            os.remove(dest / f"{h}.tar")
        res.append({"ctx": ctx, "p_tokens": n_p, "warm": {k: w[k] for k in ("prefill_s", "ttft_s")},
                    "chunk": {k: r[k] for k in ("prefill_s", "ttft_s", "prefix_hit_tokens", "external_hit_tokens",
                                                "contaminated")}, "summary": analyze(str(dest))})
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("label")
    r.add_argument("--base", default="http://spark-06c4.local:8000")
    r.add_argument("--model", default="glm-5.3")
    r.add_argument("--out", type=Path, default=REPO / "results" / "prefill-item2-20261006")
    r.add_argument("--phases", default="ttft,quality")
    r.add_argument("--sizes", type=lambda s: [int(x) for x in s.split(",")], default=[4096, 32768, 61440])
    r.add_argument("--reps", type=int, default=3)
    r.add_argument("--quality-sizes", type=lambda s: [int(x) for x in s.split(",")], default=[8192, 24576, 49152, 98304])
    r.add_argument("--quality-runs", type=int, default=2)
    r.add_argument("--quality-tokens", type=int, default=256)
    r.add_argument("--profile-ctx", type=lambda s: [int(x) for x in s.split(",")], default=[4096, 61440, 122880])
    r.add_argument("--prof-host-dir", default="")
    r.add_argument("--seed", type=int, default=20261006)
    c = sub.add_parser("compare"); c.add_argument("a"); c.add_argument("b")
    z = sub.add_parser("analyze"); z.add_argument("dir")
    a = ap.parse_args()
    if a.cmd == "compare":
        compare(a.a, a.b); return
    if a.cmd == "analyze":
        analyze(a.dir); return
    outdir = a.out / a.label
    outdir.mkdir(parents=True, exist_ok=True)
    logf = open(outdir / "bench.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True); logf.write(line + "\n"); logf.flush()

    P = Prompts(a.base, a.model, a.seed)
    complete(a.base, a.model, P.context("warmup", 512), 4, f"warm-{uuid.uuid4().hex}")
    phases = a.phases.split(",")
    if "ttft" in phases:
        json.dump(phase_ttft(a, P, log), open(outdir / "ttft.json", "w"), indent=1)
    if "quality" in phases:
        json.dump(phase_quality(a, P, log), open(outdir / "quality.json", "w"))
    if "profile" in phases:
        assert a.prof_host_dir, "--prof-host-dir is required for the profile phase"
        json.dump(phase_profile(a, P, log, outdir), open(outdir / "profile.json", "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
