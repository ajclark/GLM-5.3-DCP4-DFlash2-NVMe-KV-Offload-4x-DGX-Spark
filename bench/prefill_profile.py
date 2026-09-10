#!/usr/bin/env python3
"""Torch-profiler trace of one uncached prefill on the guarded experiment stack.

Requires a boot armed with PROFILER_DIR=/kvcache/profiles (harness `--profiler`).
Steps: warm the request path with a short prompt (JITs outside the trace);
POST /start_profile; one uncached prompt of `--tokens` words with max_tokens 1
(prefill only, chunked at the lane's max-num-batched-tokens); POST
/stop_profile; wait for one trace per rank under the experiment's kvcache dir
on every node; copy them; bucket GPU kernel time for the whole trace (no
verify-pass cutting: this is prefill). Lever G of SPEED-ARCHITECTURE-OPTIONS.md.

Usage: prefill_profile.py --label <experiment label> --out results/<dir> [--tokens 4096]
"""
import argparse
import glob
import gzip
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

HOSTS = ["spark-06c4", "spark-365c", "spark-ddbf", "spark-a218"]
SSH_USER = os.environ.get("SSH_USER") or os.environ.get("USER") or "user"
WORDS = ("river window garden stone paper summer chair path station orange cloud "
         "table copper bridge lantern meadow harbor signal velvet tunnel").split()


def post(base, path):
    req = urllib.request.Request(base + path, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status


def chat(base, prompt, max_tokens):
    body = {"model": "glm-5.3", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return out["usage"], time.monotonic() - t0


def bucket(name):
    n = name.lower()
    if "ncclDevKernel_AllReduce" in name: return "nccl AllReduce (TP)"
    if "ncclDevKernel_AllGather" in name: return "nccl AllGather"
    if "ncclDevKernel_ReduceScatter" in name: return "nccl ReduceScatter"
    if "nccl" in n: return "nccl other"
    if "marlin_moe" in n or "moe_wna16" in n: return "MoE routed (marlin_moe)"
    if "marlin" in n: return "dense marlin GEMM"
    if "moe_align" in n or "grouped_topk" in n or "count_and_sort" in n: return "MoE glue"
    if "sparse_mla" in n or "mqa" in n or "flash" in n or "attention" in n or "prefill" in n: return "attention/indexer kernels"
    if "topk" in n or "radixsort" in n or "top_k" in n: return "top-k"
    if "gemm" in n or "cutlass" in n or "cublas" in n or "wmma" in n or "mm" in n: return "dense GEMM (cutlass/cublas)"
    if "quant" in n or "fp8" in n or "scale" in n: return "quantize/dequantize"
    if "norm" in n or "rope" in n or "silu" in n or "act" in n: return "norm/rope/activation"
    if "copy" in n or "elementwise" in n or "cat" in n or "index" in n or "gather" in n or "scatter" in n: return "copies/indexing"
    return "other kernels"


def summarize(path):
    ev = json.load(gzip.open(path, "rt"))["traceEvents"]
    kernels = [e for e in ev if e.get("cat") == "kernel" and "dur" in e]
    if not kernels:
        return {"error": "no kernel events"}
    t0 = min(e["ts"] for e in kernels)
    t1 = max(e["ts"] + e["dur"] for e in kernels)
    by = {}
    for e in kernels:
        b = bucket(e["name"])
        row = by.setdefault(b, {"ms": 0.0, "count": 0})
        row["ms"] += e["dur"] / 1000.0
        row["count"] += 1
    busy = sum(r["ms"] for r in by.values())
    wall = (t1 - t0) / 1000.0
    top = sorted(((e["name"], e["dur"]) for e in kernels), key=lambda x: -x[1])[:5]
    return {"wall_ms": wall, "gpu_busy_ms": busy, "gpu_idle_ms": wall - busy,
            "buckets": dict(sorted(by.items(), key=lambda kv: -kv[1]["ms"])),
            "longest_kernels_us": [{"name": n[:120], "us": d} for n, d in top]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://spark-06c4.local:8000")
    ap.add_argument("--label", required=True, help="guarded experiment label (kvcache dir on each node)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=4096, help="approximate uncached prompt tokens")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.analyze_only:
        rng = random.Random(int(time.time()))
        # ~1.04 tokens per word for this tokenizer; unique salt defeats prefix caching.
        salt = f"session-{rng.randrange(1 << 40)}"
        warm = f"{salt} warm-up. " + " ".join(rng.choice(WORDS) for _ in range(300)) + " Reply OK."
        usage, dt = chat(a.base, warm, 4)
        print("warm-up", usage, f"{dt:.1f}s", flush=True)
        prompt = f"{salt} data. " + " ".join(rng.choice(WORDS) for _ in range(int(a.tokens / 1.04))) + " Reply OK."
        print("start_profile", post(a.base, "/start_profile"), flush=True)
        usage, dt = chat(a.base, prompt, 1)
        print("stop_profile", post(a.base, "/stop_profile"), flush=True)
        print("prefill", usage, f"{dt:.2f}s", f"{usage['prompt_tokens'] / dt:.0f} tok/s wall (incl. HTTP)", flush=True)
        json.dump({"usage": usage, "seconds": dt, "tokens_target": a.tokens}, open(os.path.join(a.out, "prefill.json"), "w"), indent=1)
        remote = f"glm-spec/{a.label}/kvcache/profiles"
        for _ in range(60):
            done = 0
            for h in HOSTS:
                r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"{SSH_USER}@{h}.local",
                                    f"ls {remote}/*.gz 2>/dev/null | wc -l"], capture_output=True, text=True)
                done += int((r.stdout.strip() or "0").split()[-1]) > 0
            if done == len(HOSTS):
                break
            time.sleep(5)
        for h in HOSTS:
            subprocess.run(["scp", "-q", f"{SSH_USER}@{h}.local:{remote}/*.gz", a.out], check=False)
    summary = {}
    for p in sorted(glob.glob(os.path.join(a.out, "*.gz"))):
        summary[os.path.basename(p)] = summarize(p)
        s = summary[os.path.basename(p)]
        print("==", os.path.basename(p), f"wall {s.get('wall_ms', 0):.0f} ms busy {s.get('gpu_busy_ms', 0):.0f} ms")
        for b, row in list(s.get("buckets", {}).items())[:10]:
            print(f"   {b:34s} {row['ms']:8.1f} ms  {row['count']:6d}")
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
