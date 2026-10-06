#!/usr/bin/env python3
"""Stale-draft-KV check: replay agent turns twice in one boot, once as-is (prefix hits reuse
draft-group KV written by the earlier BF16 drafter) and once with a fresh cache_salt (no
prefix hit: the current drafter computes all of its context KV). Same greedy request
otherwise; per-request spec-decode counters as in spec_accept_probe.

    python3 salt_test.py OUT_DIR
"""
import json
import os
import sys
import time
import uuid
from pathlib import Path

R = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(R / "runtime/vllm029"))
from spec_accept_probe import measure  # noqa: E402

CAP = Path(os.environ["CAP"])  # probe and pi agent-turn captures (not published)
prev = json.loads((R / "results/dflash-acceptance-20260930-verifycap3-agent/rows.json").read_text())
# Turns with a long reply (enough cycles) and a moderate context (salted prefill stays short).
picks = sorted((r for r in prev if r.get("set") == "agent" and r.get("completion_tokens", 0) >= 400 and (r.get("prompt_tokens") or 0) <= 50000),
               key=lambda r: r["prompt_tokens"])[:8]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
logf = open(out / "salt.log", "a")


def log(s):
    line = f"{time.strftime('%H:%M:%S')} {s}"
    print(line, flush=True)
    logf.write(line + "\n")
    logf.flush()


rows = []
for p in picks:
    stem, i = p["id"].split(":")
    i = int(i)
    b = json.loads((CAP / f"{stem}.json").read_text())
    msgs = b["messages"]
    if msgs and msgs[-1]["role"] == "user" and msgs[-1]["content"] == [{"type": "text", "text": "continue"}]:
        msgs = msgs[:-1]
    body = {k: v for k, v in b.items() if k not in ("messages", "max_completion_tokens", "store")}
    body.update(messages=msgs[:i], max_completion_tokens=768, stream=True, stream_options={"include_usage": True})
    for arm in ("cached", "salted"):
        bb = dict(body)
        if arm == "salted":
            bb["cache_salt"] = uuid.uuid4().hex
        rows.append(measure(bb, {"set": arm, "id": p["id"]}, log))
        (out / "rows.json").write_text(json.dumps(rows, indent=1))
for arm in ("cached", "salted"):
    rs = [r for r in rows if r["set"] == arm and r["clean"] and r["drafts"]]
    d = sum(r["drafts"] for r in rs)
    a = sum(r["accepted"] for r in rs)
    log(f"SUMMARY {arm}: n={len(rs)} tok/cycle {(a + d) / d:.3f} "
        f"mean tok/s {sum(r['decode_tok_s'] for r in rs) / len(rs):.1f}")
