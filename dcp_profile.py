#!/usr/bin/env python3
"""Take a torch-profiler trace of a DFlash decode on the DCP stack and
summarize where a verify cycle's time goes.

Requires the stack to be launched with PROFILER_DIR=/kvcache/profiles (the
launcher arms vLLM's torch profiler; nothing is recorded until
/start_profile). Steps:
  1. warm-up count100 (JITs, cudagraphs) outside the trace
  2. POST /start_profile, count100 (greedy, thinking off), POST /stop_profile
  3. wait for one trace file per rank on each node, copy them here
  4. summarize: NCCL kernel time by collective name, compute kernel time,
     GPU idle, all normalised per verify cycle

Usage: dcp_profile.py --out results/<label> [--max-tokens 100]
Analysis only: dcp_profile.py --out results/<label> --analyze-only
"""
import argparse
import glob
import gzip
import json
import os
SSH_USER = os.environ.get("SSH_USER") or os.environ.get("USER") or "user"
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dcp_probe as P  # noqa: E402

HOSTS = ["spark-06c4", "spark-365c", "spark-ddbf", "spark-a218"]
PROF_DIR = "/var/tmp/kvcache/profiles"
COUNT_PROMPT = "Count from 1 to 100, separated by spaces. Output only the numbers."


def post(base, path):
    req = urllib.request.Request(base + path, method="POST", data=b"")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status


def ssh(host, cmd):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", f"{SSH_USER}@{host}.local", cmd],
                          capture_output=True, text=True, timeout=120).stdout.strip()


def take_trace(base, out, max_tokens):
    os.makedirs(out, exist_ok=True)
    msgs = [{"role": "user", "content": COUNT_PROMPT}]
    print("warm-up count100 (untraced)", flush=True)
    r = P.chat(base, msgs, max_tokens, timeout=600, stream=True)
    print(f"  {r['completion_tokens']} tokens in {r['wall_s']:.2f} s", flush=True)
    before = {h: ssh(h, f"ls {PROF_DIR} 2>/dev/null | wc -l") for h in HOSTS}
    print("start_profile", post(base, "/start_profile"), flush=True)
    t0 = time.perf_counter()
    r = P.chat(base, msgs, max_tokens, timeout=600, stream=True)
    wall = time.perf_counter() - t0
    print("stop_profile", post(base, "/stop_profile"), flush=True)
    row = {"completion_tokens": r["completion_tokens"], "wall_s": round(wall, 3),
           "ttft_s": round(r["ttft_s"] or 0, 3), "decode_tok_s": r.get("decode_tok_s"),
           "accepted_per_cycle": r.get("accepted_per_cycle"), "cycle_ms": r.get("cycle_ms"),
           "content": (r["content"] or "")[:60]}
    print("traced request:", row, flush=True)
    json.dump(row, open(os.path.join(out, "traced-request.json"), "w"), indent=1)
    # wait for the trace files (written asynchronously after stop)
    for h in HOSTS:
        for _ in range(60):
            n = ssh(h, f"ls {PROF_DIR} 2>/dev/null | wc -l")
            if int(n or 0) > int(before[h] or 0):
                break
            time.sleep(5)
        files = ssh(h, f"ls -t {PROF_DIR}/*rank*.pt.trace.json.gz | head -3")
        print(f"  {h}: {files.replace(chr(10), ' | ')}", flush=True)
        newest = files.split("\n")[0] if files else ""
        if newest:
            subprocess.run(["scp", "-q", f"{SSH_USER}@{h}.local:{newest}",
                            os.path.join(out, f"{h}-{os.path.basename(newest)}")], check=False)
    return row


def load_trace(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        d = json.load(f)
    return d["traceEvents"] if isinstance(d, dict) else d


def summarize(out):
    files = sorted(glob.glob(os.path.join(out, "spark-*json*")))
    if not files:
        print("no trace files in", out)
        return
    for path in files:
        ev = load_trace(path)
        kernels = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
        if not kernels:
            print(path, ": no kernel events"); continue
        t_start = min(e["ts"] for e in kernels); t_end = max(e["ts"] + e["dur"] for e in kernels)
        wall = (t_end - t_start) / 1e3
        by_name = {}
        for e in kernels:
            name = e["name"]
            key = ("nccl:" + name.split("(")[0].replace("ncclDevKernel_", "").replace("ncclKernel_", "")
                   if "nccl" in name.lower() else
                   "memcpy" if e.get("cat") in ("gpu_memcpy", "gpu_memset") else
                   "compute:" + name.split("(")[0].split("<")[0][:60])
            n, tot = by_name.get(key, (0, 0.0))
            by_name[key] = (n + 1, tot + e["dur"])
        nccl_total = sum(t for k, (n, t) in by_name.items() if k.startswith("nccl:"))
        comp_total = sum(t for k, (n, t) in by_name.items() if not k.startswith("nccl:"))
        # busy time on the GPU (union of kernel intervals; streams may overlap)
        ivs = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
        busy = 0.0; cur_s, cur_e = ivs[0]
        for s, e in ivs[1:]:
            if s > cur_e:
                busy += cur_e - cur_s; cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        busy += cur_e - cur_s
        print(f"\n== {os.path.basename(path)}")
        print(f"   window {wall:.1f} ms, GPU busy {busy/1e3:.1f} ms ({100*busy/(t_end-t_start):.0f}%), "
              f"NCCL kernels {nccl_total/1e3:.1f} ms, other kernels {comp_total/1e3:.1f} ms (sums, may overlap)")
        print("   top NCCL kernels:")
        for k, (n, t) in sorted(by_name.items(), key=lambda kv: -kv[1][1]):
            if k.startswith("nccl:"):
                print(f"     {k:52s} n={n:5d} total={t/1e3:8.1f} ms mean={t/n:7.1f} us")
        print("   top other kernels:")
        for k, (n, t) in sorted(by_name.items(), key=lambda kv: -kv[1][1])[:14]:
            if not k.startswith("nccl:"):
                print(f"     {k:52s} n={n:5d} total={t/1e3:8.1f} ms mean={t/n:7.1f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://spark-06c4.local:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    if not a.analyze_only:
        take_trace(a.base, a.out, a.max_tokens)
    summarize(a.out)


if __name__ == "__main__":
    main()
