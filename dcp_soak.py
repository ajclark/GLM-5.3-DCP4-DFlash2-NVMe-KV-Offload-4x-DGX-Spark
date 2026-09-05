#!/usr/bin/env python3
"""Stability soak for the DCP stack. Each iteration: a COLD prefill of a fresh
long prompt (timed TTFT), the same prefix with a new question (WARM, prefix
hit; the number NVMe offload must beat when the hit is gone), a concurrent
mixed batch, a decode sanity check, and a memory sample on every node. Stops
early if a node reports MemAvailable < 300 MiB twice in a row or swaps out
more than 512 MiB between samples.

Usage: dcp_soak.py --base URL --label L --out DIR [--minutes 90] [--tokens 120000]
"""
import argparse, json, os, subprocess, time
import dcp_probe as P

HOSTS = ["spark-06c4", "spark-365c", "spark-ddbf", "spark-a218"]


def node_mem(h):
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", f"napta2k@{h}.local",
                              "awk '/MemAvailable/{printf \"%d \", $2/1024}' /proc/meminfo; awk '/^pswpout/{print $2}' /proc/vmstat"],
                             capture_output=True, text=True, timeout=20).stdout.split()
        return int(out[0]), int(out[1])
    except Exception:  # noqa: BLE001
        return None, None


class Guard:
    def __init__(self):
        self.low = {h: 0 for h in HOSTS}; self.prev = {}
    def sample(self):
        rows, trip = {}, None
        for h in HOSTS:
            a, sw = node_mem(h)
            if a is None:
                continue
            d = (sw - self.prev.get(h, sw)) // 256; self.prev[h] = sw
            self.low[h] = self.low[h] + 1 if a < 300 else 0
            rows[h.replace("spark-", "")] = a
            if self.low[h] >= 2 or d > 512:
                trip = f"{h} avail={a}MiB swapout={d}MiB"
        return rows, trip


def timed_stream(base, msgs, max_tokens, timeout=5400):
    r = P.chat(base, msgs, max_tokens, timeout=timeout, stream=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--label", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--minutes", type=int, default=90); ap.add_argument("--tokens", type=int, default=120000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    guard = Guard(); rows = []; t_end = time.monotonic() + 60 * a.minutes; i = 0
    print(f"== soak {a.label} for {a.minutes} min, {a.tokens}-token cold prompts", flush=True)
    while time.monotonic() < t_end:
        i += 1
        prefix = P.make_prompt_words(int(a.tokens / 1.04), seed=10_000 + i)
        row = {"iter": i, "t": time.strftime("%H:%M:%S")}
        # 1. cold prefill
        r = timed_stream(a.base, [{"role": "user", "content": prefix + "\n\nReply with the single word OK."}], 8)
        row.update(cold_tokens=r["prompt_tokens"], cold_ttft_s=round(r["ttft_s"] or 0, 1), cold_ok="OK" in (r["content"] or "").upper())
        mem, trip = guard.sample(); row["mem_after_cold"] = mem
        if trip:
            row["guard"] = trip; rows.append(row); print(f"  iter {i}: GUARD {trip}", flush=True); break
        # 2. warm: same prefix, new question -> prefix-cache hit
        m0 = P.metrics(a.base)
        r = timed_stream(a.base, [{"role": "user", "content": prefix + "\n\nHow many words are in the text above, roughly? Answer with one number."}], 16)
        m1 = P.metrics(a.base)
        row.update(warm_ttft_s=round(r["ttft_s"] or 0, 2), warm_answer=(r["content"] or "")[:30],
                   prefix_hits_delta=m1.get("vllm:prefix_cache_hits_total", 0) - m0.get("vllm:prefix_cache_hits_total", 0),
                   prefix_queries_delta=m1.get("vllm:prefix_cache_queries_total", 0) - m0.get("vllm:prefix_cache_queries_total", 0))
        # 3. concurrent mix
        conc = P.phase_concurrent(a.base, a.out, a.label)
        row["concurrent_ok"] = all("error" not in c for c in conc)
        # 4. decode sanity
        r = timed_stream(a.base, [{"role": "user", "content": P.BENCH[0][1]}], 400)
        gen = r["completion_tokens"] or 0
        row.update(decode_tok_s=round((gen - 1) / r["decode_s"], 1) if r.get("decode_s") else None, decode_sha=P.sha(r["content"]))
        mem, trip = guard.sample(); row["mem_after_iter"] = mem
        rows.append(row)
        json.dump(rows, open(os.path.join(a.out, "soak.json"), "w"), indent=1)
        print(f"  iter {i} {row['t']}: cold {row['cold_tokens']} tok ttft={row['cold_ttft_s']}s ok={row['cold_ok']} | warm ttft={row['warm_ttft_s']}s "
              f"hits={row['prefix_hits_delta']:.0f}/{row['prefix_queries_delta']:.0f} | conc={row['concurrent_ok']} | decode={row['decode_tok_s']} | mem={mem}", flush=True)
        if trip:
            print(f"  iter {i}: GUARD {trip}", flush=True); break
    json.dump(rows, open(os.path.join(a.out, "soak.json"), "w"), indent=1)
    print("== soak done", flush=True)


if __name__ == "__main__":
    main()
