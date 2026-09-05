#!/usr/bin/env python3
"""Compare two dcp_probe.py result directories: determinism hashes and bench rows.

Usage: compare_runs.py <baseline_dir> <candidate_dir>
"""
import json
import os
import sys


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def main(a, b):
    print(f"baseline={a}\ncandidate={b}\n")
    da, db = load(a, "determinism.json"), load(b, "determinism.json")
    if da and db:
        print("== determinism (greedy, thinking off)")
        same = 0
        for ra, rb in zip(da, db):
            match = ra["sha"] == rb["sha"]
            same += match
            print(f"  {ra['name']:12s} base={ra['sha']} cand={rb['sha']} "
                  f"{'IDENTICAL' if match else 'DIFFERS'}  tokens {ra['completion_tokens']} vs {rb['completion_tokens']}")
            if not match:
                print(f"     base: {ra['head'][:110]!r}")
                print(f"     cand: {rb['head'][:110]!r}")
        print(f"  {same}/{len(da)} identical\n")
    ba, bb = load(a, "bench.json"), load(b, "bench.json")
    if ba and bb:
        print("== bench (per prompt, mean over reps)")
        def agg(rows):
            out = {}
            for r in rows:
                out.setdefault(r["name"], []).append(r)
            return out
        ga, gb = agg(ba), agg(bb)
        def mean(rows, k):
            v = [r[k] for r in rows if r.get(k) is not None]
            return sum(v) / len(v) if v else None
        print(f"  {'prompt':9s} {'decode tok/s':>22s} {'acc/cycle':>18s} {'cycle ms':>18s} {'ttft s':>16s}")
        for name in ga:
            if name not in gb:
                continue
            f = lambda k: (mean(ga[name], k), mean(gb[name], k))
            d, c, cy, t = f("decode_tok_s"), f("accepted_per_cycle"), f("cycle_ms"), f("ttft_s")
            fmt = lambda p, w=1: f"{p[0]:.{w}f} -> {p[1]:.{w}f}" if None not in p else "n/a"
            print(f"  {name:9s} {fmt(d):>22s} {fmt(c,2):>18s} {fmt(cy):>18s} {fmt(t,2):>16s}")
        print()
        # identical greedy text?
        for name in ga:
            if name in gb:
                sa = {r["sha"] for r in ga[name]}
                sb = {r["sha"] for r in gb[name]}
                print(f"  {name:9s} text hashes base={sorted(sa)} cand={sorted(sb)} {'IDENTICAL' if sa == sb else 'DIFFER'}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
