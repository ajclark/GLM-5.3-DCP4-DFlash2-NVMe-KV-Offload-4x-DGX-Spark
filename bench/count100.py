#!/usr/bin/env python3
"""count100 decode check: greedy, thinking off, streamed; one JSON line per run.
Usage: bench/count100.py [--base http://spark-06c4.local:8000] [--runs 2] [--tag label]
Fields come from dcp_probe.chat: completion_tokens, wall_s, ttft_s, decode_tok_s, cycle_ms, accepted_per_cycle."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dcp_probe as P  # noqa: E402

PROMPT = "Count from 1 to 100, separated by spaces. Output only the numbers."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://spark-06c4.local:8000")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    ok = True
    for i in range(a.runs):
        t0 = time.perf_counter()
        try:
            r = P.chat(a.base, [{"role": "user", "content": PROMPT}], a.max_tokens, timeout=600, stream=True)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"tag": a.tag, "run": i, "error": str(e)[:200], "wall_s": round(time.perf_counter() - t0, 2)}), flush=True)
            ok = False
            continue
        content = (r.get("content") or "")
        row = {"tag": a.tag, "run": i, "completion_tokens": r.get("completion_tokens"), "wall_s": round(r.get("wall_s") or 0, 2),
               "ttft_s": round(r.get("ttft_s") or 0, 2), "decode_tok_s": r.get("decode_tok_s"), "cycle_ms": r.get("cycle_ms"),
               "accepted_per_cycle": r.get("accepted_per_cycle"), "starts_ok": content.strip().startswith("1 2 3"), "tail": content.strip()[-20:]}
        print(json.dumps(row), flush=True)
        ok = ok and bool(row["starts_ok"])
    raise SystemExit(0 if ok else 1)

if __name__ == "__main__":
    main()
