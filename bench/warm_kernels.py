#!/usr/bin/env python3
"""Startup warm-up: make every request path compile and load its Triton kernels while the
process is fresh (on DGX Spark GB10 / driver 580 a CUDA module load hours into a process can
fail with "operation not permitted" and kill the rank: results/incident-20260907-1418-*).
Covers: a long prefill (DSA indexer prefill kernels, chunk metadata), odd lengths (Triton's
divisibility-by-16 specializations), a concurrent batch (DFlash prepare, batched decode, top-k).
Usage: bench/warm_kernels.py [--base URL] [--long-tokens 12000]   exit 0 only if every request answered."""
import argparse, concurrent.futures as cf, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dcp_probe as P  # noqa: E402

def req(base, prompt, max_tokens, timeout=900):
    t0 = time.perf_counter()
    try:
        r = P.chat(base, [{"role": "user", "content": prompt}], max_tokens, timeout=timeout, stream=True)
        return {"ok": True, "prompt_tokens": r.get("prompt_tokens"), "completion_tokens": r.get("completion_tokens"), "wall_s": round(time.perf_counter() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:160], "wall_s": round(time.perf_counter() - t0, 1)}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", default="http://spark-06c4.local:8000"); ap.add_argument("--long-tokens", type=int, default=12000)
    a = ap.parse_args(); ok = True
    filler = "The quick brown fox jumps over the lazy dog. " * (a.long_tokens // 10)
    steps = [
        ("long-prefill", filler + "\nSummarize the above in one sentence.", 32),
        ("odd-length-1", "Count from 1 to 7, comma separated." + " x" * 15, 24),
        ("odd-length-2", "List three colors." + " y" * 31, 24),
    ]
    for name, prompt, n in steps:
        r = req(a.base, prompt, n); r["step"] = name; print(json.dumps(r), flush=True); ok = ok and r["ok"]
    with cf.ThreadPoolExecutor(6) as ex:
        futs = [ex.submit(req, a.base, f"Write {i+1} short sentences about the sea.", 64) for i in range(6)]
        for i, f in enumerate(futs):
            r = f.result(); r["step"] = f"concurrent-6x-{i}"; print(json.dumps(r), flush=True); ok = ok and r["ok"]
    raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()
