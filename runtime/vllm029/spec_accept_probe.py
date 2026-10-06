#!/usr/bin/env python3
"""Measure DFlash acceptance on realistic traffic.

Replays real pi agent turns (requests captured through pi itself, so system
prompt, tools and message serialization are exact) plus small prose and code
sets, one request at a time on an otherwise idle endpoint. Per request it
diffs vLLM's global speculative-decoding counters (drafts, draft tokens,
accepted tokens, accepted per position) and checks that no other traffic ran
in the window (generation-token delta must equal the request's own count).

    python3 spec_accept_probe.py --captures DIR --out results/dflash-acceptance-YYYYMMDD
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
import urllib.request
from pathlib import Path

BASE = "http://spark-06c4.local:8000"
_M = re.compile(r'^(vllm:[a-z_:]+)(\{[^}]*\})?\s+([-+0-9.eE]+)$')


def scrape() -> dict:
    out = {}
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            m = _M.match(line)
            if not m:
                continue
            name, labels, val = m.groups()
            pos = re.search(r'position="(\d+)"', labels or "")
            if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total" and pos:
                out[f"pos{pos.group(1)}"] = float(val)
            elif name in ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
                          "vllm:spec_decode_num_accepted_tokens_total", "vllm:generation_tokens_total",
                          "vllm:num_requests_running", "vllm:num_requests_waiting"):
                out[name.split(":")[1]] = float(val)
    return out


def wait_idle(max_wait: float = 900) -> float:
    t0 = time.time()
    while time.time() - t0 < max_wait:
        m = scrape()
        if m.get("num_requests_running", 0) == 0 and m.get("num_requests_waiting", 0) == 0:
            return time.time() - t0
        time.sleep(2)
    return -1


def stream_chat(body: dict, timeout: float = 1800) -> dict:
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t_send = time.time()
    first = last = None
    chars = {"reasoning": 0, "content": 0, "tool_calls": 0}
    usage, err, text = None, None, []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices", []):
                    d = ch.get("delta") or {}
                    got = False
                    for k in ("reasoning", "reasoning_content"):
                        if d.get(k):
                            chars["reasoning"] += len(d[k]); got = True
                    if d.get("content"):
                        chars["content"] += len(d["content"]); text.append(d["content"]); got = True
                    for tc in d.get("tool_calls") or []:
                        chars["tool_calls"] += len(json.dumps(tc.get("function", {}))); got = True
                    if got:
                        now = time.time()
                        first = first or now
                        last = now
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    return {"t_send": t_send, "ttft": (first - t_send) if first else None,
            "decode_s": (last - first) if first and last else None, "chars": chars,
            "usage": usage, "error": err, "text": "".join(text)[:300]}


def measure(body: dict, meta: dict, log) -> dict:
    waited = wait_idle()
    a = scrape()
    r = stream_chat(body)
    time.sleep(0.5)
    b = scrape()
    d = {k: b.get(k, 0) - a.get(k, 0) for k in b if k.startswith(("spec_", "pos", "generation"))}
    comp = (r["usage"] or {}).get("completion_tokens", 0)
    drafts = d.get("spec_decode_num_drafts_total", 0)
    acc = d.get("spec_decode_num_accepted_tokens_total", 0)
    rec = {**meta, "waited_s": waited, "error": r["error"], "ttft": r["ttft"], "decode_s": r["decode_s"],
           "prompt_tokens": (r["usage"] or {}).get("prompt_tokens"), "completion_tokens": comp,
           "chars": r["chars"], "drafts": drafts, "accepted": acc,
           "per_pos": [d.get(f"pos{i}", 0) for i in range(7)],
           "gen_delta": d.get("generation_tokens_total", 0),
           "clean": waited >= 0 and abs(d.get("generation_tokens_total", 0) - comp) <= 2 and not r["error"],
           "text": r["text"]}
    rec["tokens_per_cycle"] = (acc + drafts) / drafts if drafts else None
    rec["decode_tok_s"] = (comp - 1) / r["decode_s"] if r["decode_s"] and comp > 1 else None
    tot = sum(r["chars"].values()) or 1
    rec["mix"] = {k: round(v / tot, 2) for k, v in r["chars"].items()}
    log(f"{meta['set']:6s} {meta.get('id',''):>10s} prompt {rec['prompt_tokens']} gen {comp} "
        f"tok/cycle {rec['tokens_per_cycle'] or 0:.2f} tok/s {rec['decode_tok_s'] or 0:.1f} "
        f"mix {rec['mix']} {'CLEAN' if rec['clean'] else 'CONTAMINATED'} {r['error'] or ''}")
    return rec


PROSE = [
    "Write a 600-word short story about a lighthouse keeper who finds a message in a bottle.",
    "Explain to a curious teenager how vaccines train the immune system. Use plain language and a couple of analogies.",
    "Write a persuasive op-ed arguing that cities should replace parking lots with parks.",
    "Describe a walk through a busy night market in Taipei, focusing on sounds and smells.",
    "Write a heartfelt letter from a grandmother to her grandson who is leaving for university.",
    "Summarize the causes and consequences of the 2008 financial crisis for a general audience.",
    "Write a dialogue between two friends arguing about whether to adopt a dog.",
    "Give a detailed, friendly guide to planning a first solo backpacking trip in Europe.",
    "Write a eulogy for a beloved local bookstore that is closing after 40 years.",
    "Explain the difference between weather and climate, with examples, in about 400 words.",
    "Write a product description and launch announcement for a fictional solar-powered e-bike.",
    "Tell a bedtime story about a shy dragon who is afraid of the dark.",
    "Write a reflective essay on what makes a good mentor.",
    "Describe the history of the printing press and its effect on European society.",
    "Write a humorous complaint letter to a hotel about a very noisy ice machine.",
]
CODE = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H5M' into seconds, with tests.",
    "Implement an LRU cache class in Python without using functools, with get/put in O(1).",
    "Write a bash script that finds the 10 largest files under a directory and prints human-readable sizes.",
    "Write a TypeScript function that debounces another function, with proper generic typing.",
    "Implement Dijkstra's shortest path in Python using heapq, with a small example graph.",
    "Write a Rust function that reads a CSV file and computes the mean of a numeric column, handling errors.",
    "Write a SQL query to find the top 3 customers by total order value per month, and explain it.",
    "Implement a thread-safe bounded queue in Python using threading.Condition.",
    "Write a Go HTTP server with one JSON endpoint that returns the current time and a request counter.",
    "Refactor this into idiomatic Python and add type hints: def f(l):\n  r=[]\n  for i in range(len(l)):\n    if l[i]%2==0: r.append(l[i]*l[i])\n  return r",
    "Write a React component for a searchable, sortable table given an array of objects.",
    "Implement binary search on a sorted list in C, returning the index or -1, with a main() test.",
    "Write a Python asyncio program that fetches 5 URLs concurrently with a timeout and reports status codes.",
    "Write a regex and a Python function to validate and normalize UK postcodes.",
    "Implement a trie in Python supporting insert, search and prefix listing.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--max-context-chars", type=int, default=320_000)
    ap.add_argument("--max-tokens", type=int, default=768)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    logf = open(a.out / "probe.log", "a")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    rows = []
    # --- agent replay: the last `turns` assistant turns whose context fits the cap
    for cap in sorted(a.captures.glob("s*.json")):
        b = json.loads(cap.read_text())
        msgs = b["messages"]
        if msgs and msgs[-1]["role"] == "user" and msgs[-1]["content"] == [{"type": "text", "text": "continue"}]:
            msgs = msgs[:-1]
        cut = [i for i, m in enumerate(msgs) if m["role"] == "assistant"
               and len(json.dumps(msgs[:i])) <= a.max_context_chars]
        cut = cut[-a.turns:]
        log(f"session {cap.stem}: {len(msgs)} msgs, replaying assistant turns at {cut}")
        for i in cut:
            body = {k: v for k, v in b.items() if k not in ("messages", "max_completion_tokens", "store")}
            body.update(messages=msgs[:i], max_completion_tokens=a.max_tokens, stream=True,
                        stream_options={"include_usage": True})
            rows.append(measure(body, {"set": "agent", "id": f"{cap.stem}:{i}"}, log))
            (a.out / "rows.json").write_text(json.dumps(rows, indent=1))
    # --- prose and code, thinking off and on
    for think in (False, True):
        for name, prompts in (("prose", PROSE), ("code", CODE)):
            for j, p in enumerate(prompts if not think else prompts[:5]):
                body = {"model": "glm-5.3", "messages": [{"role": "user", "content": p}], "stream": True,
                        "stream_options": {"include_usage": True}, "temperature": 0, "top_p": 1,
                        "max_completion_tokens": a.max_tokens,
                        "chat_template_kwargs": {"enable_thinking": think, "reasoning_effort": "high"}}
                rows.append(measure(body, {"set": name + ("+think" if think else ""), "id": str(j)}, log))
                (a.out / "rows.json").write_text(json.dumps(rows, indent=1))

    # --- summary
    summary = {}
    for s in sorted({r["set"] for r in rows}):
        rs = [r for r in rows if r["set"] == s and r["clean"] and r["drafts"]]
        if not rs:
            continue
        drafts = sum(r["drafts"] for r in rs)
        acc = sum(r["accepted"] for r in rs)
        pos = [sum(r["per_pos"][k] for r in rs) / drafts for k in range(7)]
        summary[s] = {
            "n": len(rs), "n_contaminated": sum(1 for r in rows if r["set"] == s and not r["clean"]),
            "tokens_per_cycle": (acc + drafts) / drafts,
            "per_request_tokens_per_cycle_median": statistics.median(r["tokens_per_cycle"] for r in rs),
            "acceptance_by_position": [round(x, 3) for x in pos],
            "decode_tok_s_median": statistics.median(r["decode_tok_s"] for r in rs if r["decode_tok_s"]),
            "output_mix": {k: round(sum(r["chars"][k] for r in rs) / max(1, sum(sum(r["chars"].values()) for r in rs)), 2)
                           for k in ("reasoning", "content", "tool_calls")},
        }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    for s, v in summary.items():
        log(f"SUMMARY {s:12s} n={v['n']:3d} tok/cycle {v['tokens_per_cycle']:.2f} "
            f"(median {v['per_request_tokens_per_cycle_median']:.2f}) tok/s {v['decode_tok_s_median']:.1f} "
            f"pos-accept {v['acceptance_by_position']} mix {v['output_mix']}")


if __name__ == "__main__":
    main()
