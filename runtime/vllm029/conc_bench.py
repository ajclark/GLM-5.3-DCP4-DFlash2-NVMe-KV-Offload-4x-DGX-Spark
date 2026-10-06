#!/usr/bin/env python3
"""Aggregate decode throughput with n concurrent greedy streams (C2-C4 companion to
spec_accept_probe.py, whose prose/code prompts and pi agent captures it reuses).

Each round starts n requests together on an idle endpoint and streams them with
continuous usage stats. Aggregate tok/s counts only the window in which all n streams are
decoding (from the last first-token to the first last-token), so prefill and stragglers
are excluded. Tokens per cycle comes from vLLM's spec-decode counters over the round, and
a round is marked contaminated if the endpoint generated tokens for anything else.

    python3 conc_bench.py --out results/<dir> --n 2,3,4 --sets prose,code,agent,mix \\
        --captures CAPTURE_DIR [--rounds 4] [--label x]

CAPTURE_DIR holds captured agent requests (s*.json); ours are private and not published.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path

from spec_accept_probe import BASE, CODE, PROSE, scrape, wait_idle


def stream(body: dict, out: dict) -> None:
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    out.update(t_send=time.time(), marks=[], error=None, usage=None)
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                u = ev.get("usage")
                if u:
                    out["usage"] = u
                    if ev.get("choices"):
                        out["marks"].append((time.time(), u.get("completion_tokens", 0)))
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"


def tokens_at(marks: list, t: float) -> float:
    """Completion tokens streamed by time t (linear between chunks)."""
    if not marks or t <= marks[0][0]:
        return marks[0][1] if marks else 0
    for (t0, c0), (t1, c1) in zip(marks, marks[1:]):
        if t <= t1:
            return c0 + (c1 - c0) * (t - t0) / (t1 - t0) if t1 > t0 else c1
    return marks[-1][1]


def chat_body(prompt: str, max_tokens: int) -> dict:
    return {"model": "glm-5.3", "messages": [{"role": "user", "content": prompt}], "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "temperature": 0, "top_p": 1, "max_completion_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "high"}}


def agent_bodies(captures: Path, max_tokens: int, min_prev_tokens: int, prev_rows: Path | None) -> list:
    """Replayed pi turns (same selection as spec_accept_probe), preferring turns whose
    earlier replay generated at least min_prev_tokens so the streams overlap."""
    long_ids = None
    if prev_rows and prev_rows.exists():
        long_ids = {r["id"] for r in json.loads(prev_rows.read_text())
                    if r.get("set") == "agent" and (r.get("completion_tokens") or 0) >= min_prev_tokens}
    out = []
    for cap in sorted(captures.glob("s*.json")):
        b = json.loads(cap.read_text())
        msgs = b["messages"]
        if msgs and msgs[-1]["role"] == "user" and msgs[-1]["content"] == [{"type": "text", "text": "continue"}]:
            msgs = msgs[:-1]
        cut = [i for i, m in enumerate(msgs) if m["role"] == "assistant"
               and len(json.dumps(msgs[:i])) <= 320_000][-8:]
        for i in cut:
            if long_ids is not None and f"{cap.stem}:{i}" not in long_ids:
                continue
            body = {k: v for k, v in b.items() if k not in ("messages", "max_completion_tokens", "store")}
            body.update(messages=msgs[:i], max_completion_tokens=max_tokens, stream=True,
                        stream_options={"include_usage": True, "continuous_usage_stats": True})
            out.append((f"{cap.stem}:{i}", body))
    return out


def run_round(bodies: list, log) -> dict:
    waited = wait_idle()
    a = scrape()
    outs = [{} for _ in bodies]
    threads = [threading.Thread(target=stream, args=(b, o)) for (_, b), o in zip(bodies, outs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    time.sleep(0.5)
    b = scrape()
    d = {k: b.get(k, 0) - a.get(k, 0) for k in b if k.startswith(("spec_", "generation"))}
    comp = sum((o.get("usage") or {}).get("completion_tokens", 0) for o in outs)
    firsts = [o["marks"][0][0] for o in outs if o.get("marks")]
    lasts = [o["marks"][-1][0] for o in outs if o.get("marks")]
    rec = {"ids": [i for i, _ in bodies], "waited_s": waited,
           "errors": [o.get("error") for o in outs if o.get("error")],
           "completion_tokens": [(o.get("usage") or {}).get("completion_tokens", 0) for o in outs],
           "prompt_tokens": [(o.get("usage") or {}).get("prompt_tokens", 0) for o in outs],
           "drafts": d.get("spec_decode_num_drafts_total", 0),
           "accepted": d.get("spec_decode_num_accepted_tokens_total", 0),
           "gen_delta": d.get("generation_tokens_total", 0)}
    rec["clean"] = (waited >= 0 and not rec["errors"] and len(firsts) == len(bodies)
                    and abs(rec["gen_delta"] - comp) <= 2 * len(bodies))
    rec["tokens_per_cycle"] = (rec["accepted"] + rec["drafts"]) / rec["drafts"] if rec["drafts"] else None
    if len(firsts) == len(bodies):
        t0, t1 = max(firsts), min(lasts)
        rec["overlap_s"] = t1 - t0
        if t1 - t0 > 2.0:
            toks = [tokens_at(o["marks"], t1) - tokens_at(o["marks"], t0) for o in outs]
            rec["aggregate_tok_s"] = sum(toks) / (t1 - t0)
            rec["per_stream_tok_s"] = [x / (t1 - t0) for x in toks]
    log(f"n={len(bodies)} {rec['ids']} agg {rec.get('aggregate_tok_s') or 0:.1f} tok/s over "
        f"{rec.get('overlap_s') or 0:.1f}s, tok/cycle {rec['tokens_per_cycle'] or 0:.2f}, "
        f"gen {rec['completion_tokens']} {'CLEAN' if rec['clean'] else 'CONTAMINATED'} {rec['errors'] or ''}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", default="2,3,4")
    ap.add_argument("--sets", default="prose,code,agent,mix")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--captures", type=Path)
    ap.add_argument("--prev-agent-rows", type=Path,
                    default=Path(__file__).resolve().parents[2] / "results/dflash-acceptance-20260930-verifycap3-agent/rows.json")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    logf = open(a.out / "conc.log", "a")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {a.label} {s}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    pools = {"prose": [(f"prose{i}", chat_body(p, a.max_tokens)) for i, p in enumerate(PROSE)],
             "code": [(f"code{i}", chat_body(p, a.max_tokens)) for i, p in enumerate(CODE)]}
    sets = a.sets.split(",")
    if "agent" in sets or "mix" in sets:
        if a.captures is None:
            raise SystemExit("--captures is required for the agent and mix sets")
        pools["agent"] = agent_bodies(a.captures, a.max_tokens, 400, a.prev_agent_rows)
        log(f"{len(pools['agent'])} agent turns selected")
    rows_path = a.out / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    for n in [int(x) for x in a.n.split(",")]:
        for s in sets:
            if s == "mix":
                cyc = [itertools.cycle(pools[k]) for k in ("prose", "code", "agent")]
                picks = [[next(cyc[(r + j) % 3]) for j in range(n)] for r in range(a.rounds)]
            else:
                pool = pools[s]
                picks = [[pool[(r * n + j) % len(pool)] for j in range(n)] for r in range(a.rounds)]
            for r, bodies in enumerate(picks):
                rec = run_round(bodies, log)
                rec.update(n=n, set=s, round=r, label=a.label)
                rows.append(rec)
                rows_path.write_text(json.dumps(rows, indent=1))
    summary = {}
    for (lab, n, s), grp in itertools.groupby(sorted(rows, key=lambda r: (r["label"], r["n"], r["set"])),
                                              key=lambda r: (r["label"], r["n"], r["set"])):
        g = [r for r in grp if r["clean"] and r.get("aggregate_tok_s")]
        if not g:
            continue
        drafts = sum(r["drafts"] for r in g)
        summary[f"{lab}|n{n}|{s}"] = {
            "rounds": len(g), "aggregate_tok_s_median": statistics.median(r["aggregate_tok_s"] for r in g),
            "aggregate_tok_s_mean": statistics.mean(r["aggregate_tok_s"] for r in g),
            "tokens_per_cycle": (sum(r["accepted"] for r in g) + drafts) / drafts if drafts else None}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    for k, v in summary.items():
        log(f"SUMMARY {k:28s} rounds {v['rounds']} agg median {v['aggregate_tok_s_median']:.1f} "
            f"mean {v['aggregate_tok_s_mean']:.1f} tok/s, tok/cycle {v['tokens_per_cycle'] or 0:.2f}")


if __name__ == "__main__":
    main()
