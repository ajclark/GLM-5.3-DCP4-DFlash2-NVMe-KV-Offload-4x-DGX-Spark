#!/usr/bin/env python3
"""Uncached prefill wall time: N fresh random-word prompts of ~T tokens, max_tokens=1,
thinking off. Each prompt carries a unique salt so nothing hits the prefix cache.
Prints per-run seconds and tok/s and writes <out>.json. No profiler, no hold needed;
pair a production run with the same command inside a guarded experiment."""
import argparse, json, random, time, urllib.request

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho "
         "sigma tau upsilon phi chi psi omega river stone cloud forest window garden silver copper "
         "table copper bridge lantern meadow harbor signal velvet tunnel").split()


def chat(base, prompt, max_tokens):
    body = {"model": "glm-5.3", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    return out["usage"], time.monotonic() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://spark-06c4.local:8000")
    ap.add_argument("--tokens", type=int, nargs="+", default=[4096])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(int(time.time() * 1000))
    salt = f"session-{rng.randrange(1 << 40)}"
    usage, dt = chat(a.base, f"{salt} warm-up. " + " ".join(rng.choice(WORDS) for _ in range(300)) + " Reply OK.", 4)
    print("warm-up", usage["prompt_tokens"], f"{dt:.2f}s", flush=True)
    rows = []
    for tokens in a.tokens:
        for i in range(a.repeats):
            prompt = f"{salt}-{tokens}-{i} data. " + " ".join(rng.choice(WORDS) for _ in range(int(tokens / 1.04))) + " Reply OK."
            usage, dt = chat(a.base, prompt, 1)
            rows.append({"target": tokens, "repeat": i, "prompt_tokens": usage["prompt_tokens"], "seconds": dt})
            print(f"prefill target {tokens} repeat {i}: {usage['prompt_tokens']} tokens {dt:.2f}s {usage['prompt_tokens']/dt:.0f} tok/s", flush=True)
    json.dump({"base": a.base, "rows": rows}, open(a.out, "w"), indent=1)
    for tokens in a.tokens:
        sel = [r["seconds"] for r in rows if r["target"] == tokens]
        print(f"target {tokens}: median {sorted(sel)[len(sel)//2]:.2f}s min {min(sel):.2f}s max {max(sel):.2f}s")


if __name__ == "__main__":
    main()
