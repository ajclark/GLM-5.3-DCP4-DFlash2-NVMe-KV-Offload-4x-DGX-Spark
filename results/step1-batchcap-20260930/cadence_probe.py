#!/usr/bin/env python3
"""Prefill cadence probe: two greedy count-to-N decoders run while a fresh (salted,
uncached) long prompt is prefilled. Reports the decoders' tok/s during the long prompt's
prefill window and the long prompt's TTFT. The cadence is switched through the control
file on rank 0 (prefill_cadence), so all arms run in one boot.

    python3 cadence_probe.py OUT.json --cadences 1,2,4 --words 45000
"""
import argparse
import json
import random
import subprocess
import threading
import time
import urllib.request

BASE = "http://spark-06c4.local:8000"
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november".split()


def stream(body, out):
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    out["t_send"], out["marks"] = time.time(), []
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            ev = json.loads(line[5:])
            u = ev.get("usage")
            if u and ev.get("choices"):
                out["marks"].append((time.time(), u.get("completion_tokens", 0)))
            if u:
                out["usage"] = u


def tokens_between(marks, t0, t1):
    inside = [c for t, c in marks if t0 <= t <= t1]
    before = [c for t, c in marks if t < t0]
    return (inside[-1] - (before[-1] if before else 0)) if inside else 0


def set_cadence(n, base_ctl):
    ctl = dict(base_ctl, prefill_cadence=n)
    subprocess.run(["ssh", "spark-06c4.local", f"echo '{json.dumps(ctl)}' > ~/verify-cap-live/control.json"],
                   check=True)
    time.sleep(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cadences", default="1,2,4")
    ap.add_argument("--words", type=int, default=45000)
    ap.add_argument("--control", default='{"mode": "auto", "batch": true, "policy": "ratio", "costs": "costs-batch.json"}')
    a = ap.parse_args()
    rows = []
    for n in [int(x) for x in a.cadences.split(",")]:
        set_cadence(n, json.loads(a.control))
        decs = [{}, {}]
        bodies = [{"model": "glm-5.3", "messages": [{"role": "user", "content": f"Count from 1 to 600, one number per line. ({i})"}],
                   "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True},
                   "temperature": 0, "max_tokens": 2400, "chat_template_kwargs": {"enable_thinking": False}}
                  for i in range(2)]
        th = [threading.Thread(target=stream, args=(b, o)) for b, o in zip(bodies, decs)]
        for t in th:
            t.start()
        while not all(o.get("marks") for o in decs):
            time.sleep(0.2)
        time.sleep(5)
        rng = random.Random(time.time())
        text = f"[{rng.random():.12f}] " + " ".join(rng.choice(WORDS) for _ in range(a.words)) + "\nReply with OK."
        longo = {}
        lb = {"model": "glm-5.3", "messages": [{"role": "user", "content": text}], "stream": True,
              "stream_options": {"include_usage": True, "continuous_usage_stats": True},
              "temperature": 0, "max_tokens": 4, "chat_template_kwargs": {"enable_thinking": False}}
        stream(lb, longo)
        t0 = longo["t_send"]
        t1 = longo["marks"][0][0] if longo["marks"] else time.time()
        for t in th:
            t.join()
        dec_rates = [tokens_between(o["marks"], t0, t1) / (t1 - t0) for o in decs]
        before = [tokens_between(o["marks"], t0 - 5, t0) / 5 for o in decs]
        row = {"cadence": n, "long_prompt_tokens": longo.get("usage", {}).get("prompt_tokens"),
               "long_ttft_s": round(t1 - t0, 2), "decoder_tok_s_during_prefill": [round(x, 2) for x in dec_rates],
               "decoder_tok_s_before": [round(x, 2) for x in before]}
        print(json.dumps(row), flush=True)
        rows.append(row)
        json.dump(rows, open(a.out, "w"), indent=1)
    set_cadence(1, json.loads(a.control))


if __name__ == "__main__":
    main()
