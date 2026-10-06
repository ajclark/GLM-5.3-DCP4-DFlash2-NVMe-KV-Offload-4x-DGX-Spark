#!/usr/bin/env python3
"""C1 decode cycle time on this GPU: random-token prompts of several lengths, fixed-length greedy
generations (ignore_eos), one request at a time. Cycle time = server-side decode time / verify
cycles (spec-decode draft count), cross-checked with client-side streaming timestamps.
usage: cycle_bench.py OUT.json [--lens 2048,8192,...] [--reps 3] [--max-tokens 512] [--same-prompt]
BASE=http://host:8000 selects the endpoint; --same-prompt reuses one prompt per length
(prefix-cached after the first repeat, so long contexts do not pay a cold prefill each time)."""
import argparse, json, os, random, re, statistics, time, urllib.request
BASE = os.environ.get("BASE", "http://127.0.0.1:8000")
M = re.compile(r'^(vllm:[a-z_:]+)(\{[^}]*\})?\s+([-+0-9.eE]+)$')
WANT = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:generation_tokens_total", "vllm:num_requests_running", "vllm:num_requests_waiting",
        "vllm:request_decode_time_seconds_sum", "vllm:request_decode_time_seconds_count",
        "vllm:request_prefill_time_seconds_sum")
def scrape():
    out = {}
    for line in urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode().splitlines():
        m = M.match(line)
        if m and m.group(1) in WANT:
            out[m.group(1).split(":")[1]] = out.get(m.group(1).split(":")[1], 0) + float(m.group(3))
    return out
def idle():
    while True:
        s = scrape()
        if s.get("num_requests_running", 0) == 0 and s.get("num_requests_waiting", 0) == 0:
            return
        time.sleep(0.5)
def one(n_ctx, max_tokens, seed):
    rng = random.Random(seed)
    prompt = [rng.randrange(1000, 150000) for _ in range(n_ctx)]
    body = {"model": "glm-5.3", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "ignore_eos": True,
            "stream": True, "stream_options": {"include_usage": True}}
    idle(); a = scrape()
    req = urllib.request.Request(BASE + "/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time(); times = []; usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"): usage = ev["usage"]
            if ev.get("choices"): times.append(time.time())
    time.sleep(0.5); b = scrape()
    d = {k: b.get(k, 0) - a.get(k, 0) for k in b}
    drafts = d.get("spec_decode_num_drafts_total", 0)
    steps = drafts if drafts else (usage["completion_tokens"] - 1 if usage else 0)
    dec = d.get("request_decode_time_seconds_sum", 0)
    rec = {"ctx": n_ctx, "gen": usage["completion_tokens"] if usage else None, "ttft_s": times[0] - t0 if times else None,
           "prefill_s": d.get("request_prefill_time_seconds_sum"), "decode_s": dec, "steps": steps,
           "accepted": d.get("spec_decode_num_accepted_tokens_total", 0),
           "cycle_ms_server": 1000 * dec / steps if steps else None,
           "cycle_ms_client": 1000 * (times[-1] - times[0]) / max(1, steps - 1) if len(times) > 1 else None}
    rec["tokens_per_step"] = (rec["gen"] - 1) / steps if steps and rec["gen"] else None
    print(json.dumps(rec), flush=True)
    return rec
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--lens", default="2048,8192,16384,32768,65536")
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--same-prompt", action="store_true"); ap.add_argument("--label", default="")
a = ap.parse_args()
one(256, 64, 999)  # warm-up
rows = [one(L, a.max_tokens, 1000 * L + (0 if a.same_prompt else r)) for L in map(int, a.lens.split(",")) for r in range(a.reps)]
summ = {}
for L in sorted({r["ctx"] for r in rows}):
    rs = [r for r in rows if r["ctx"] == L and r["cycle_ms_server"]]
    summ[L] = {"cycle_ms_server_median": statistics.median(r["cycle_ms_server"] for r in rs),
               "cycle_ms_client_median": statistics.median(r["cycle_ms_client"] for r in rs if r["cycle_ms_client"]),
               "tokens_per_step": statistics.median(r["tokens_per_step"] for r in rs), "n": len(rs)}
print(json.dumps(summ, indent=1))
json.dump({"label": a.label, "rows": rows, "summary": summ}, open(a.out, "w"), indent=1)
