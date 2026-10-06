#!/usr/bin/env python3
"""Prefix-cache hit on an identical repeated prompt: GPU and external (NVMe) hit tokens
for the second send, from vLLM's prefix-cache counters. Prompts are salted per run."""
import json, random, sys, time, urllib.request
BASE = "http://spark-06c4.local:8000"
KEYS = ("prefix_cache_queries_total", "prefix_cache_hits_total",
        "external_prefix_cache_queries_total", "external_prefix_cache_hits_total")
def scrape():
    out = {}
    for line in urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode().splitlines():
        for k in KEYS:
            if line.startswith("vllm:" + k + "{"):
                out[k] = float(line.rsplit(" ", 1)[1])
    return out
def send(text):
    body = {"model": "glm-5.3", "messages": [{"role": "user", "content": text}], "max_tokens": 1,
            "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    return r["usage"]["prompt_tokens"], time.time() - t
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november".split()
rows = []
for target_words in [int(x) for x in sys.argv[1].split(",")]:
    rng = random.Random(time.time())
    text = f"[{rng.random():.12f}] " + " ".join(rng.choice(words) for _ in range(target_words)) + "\nReply with OK."
    send(text); time.sleep(1.0)
    a = scrape(); n, dt = send(text); b = scrape()
    d = {k: b[k] - a[k] for k in KEYS}
    row = {"prompt_tokens": n, "gpu_hit": d["prefix_cache_hits_total"], "gpu_query": d["prefix_cache_queries_total"],
           "ext_hit": d["external_prefix_cache_hits_total"], "ext_query": d["external_prefix_cache_queries_total"],
           "second_send_s": round(dt, 3)}
    print(json.dumps(row), flush=True); rows.append(row)
json.dump(rows, open(sys.argv[2], "w"), indent=1)
