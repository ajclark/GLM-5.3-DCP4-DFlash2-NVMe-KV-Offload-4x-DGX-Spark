#!/usr/bin/env python3
"""Inject short raw-token requests while two speculative chat streams decode.

The short requests have exactly 5/3/7 input tokens, below the K+1=8 decode
threshold. With --context-tokens, those suffixes follow an aligned shared cached
prefix instead. This exercises mixed metadata without changing speculative settings.
Confirm the indexer mixed-length diagnostic in worker logs after this probe.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import re
import random
import threading
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://spark-06c4.local:8000")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--context-tokens", type=int, default=0,
                    help="Prime a shared context so each short suffix is a ~50k-context decode")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assert args.context_tokens == 0 or 1024 <= args.context_tokens <= 52000
    count_prompt = "Count from 1 to 300, comma-separated, nothing else."
    prefix_tokens = []

    def request(path, body):
        return urllib.request.Request(args.base + path, data=json.dumps(body).encode(),
                                      headers={"Content-Type": "application/json"})

    def post(path, body):
        with urllib.request.urlopen(request(path, body), timeout=600) as response:
            return json.load(response)

    def count_stream(index, started):
        body = {"model": "glm-5.3", "messages": [{"role": "user", "content":
                count_prompt}], "max_tokens": 1000,
                "temperature": 0, "seed": index, "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        begin = time.monotonic()
        content, usage, done = [], {}, False
        with urllib.request.urlopen(request("/v1/chat/completions", body), timeout=180) as response:
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                payload = raw[5:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                event = json.loads(payload)
                if "error" in event:
                    raise RuntimeError(event["error"])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {}).get("content")
                    if delta:
                        started.set()
                        content.append(delta)
        text = "".join(content)
        result = {"stream": index, "text": text, "usage": usage, "done": done,
                  "wall_s": time.monotonic() - begin}
        (args.out / f"stream-{index}.json").write_text(json.dumps(result, indent=2) + "\n")
        assert done, "Stream ended without completion marker"
        assert [int(x) for x in re.findall(r"\d+", text)] == list(range(1, 301)), text
        return result

    tokenized = post("/tokenize", {"model": "glm-5.3", "prompt": "Hello", "add_special_tokens": False})
    token = tokenized["tokens"][-1]
    if args.context_tokens:
        rng = random.Random(20260911)
        words = [rng.choice("river window garden stone paper summer chair path orange cloud table copper".split())
                 for _ in range(args.context_tokens)]
        n = len(words)
        for _ in range(8):
            count_prompt = ("Ignore this arbitrary test data:\n" + " ".join(words[:n]) +
                            "\nEnd of data. Count from 1 to 300, comma-separated, nothing else.")
            tokens = post("/tokenize", {"model": "glm-5.3", "messages":
                          [{"role": "user", "content": count_prompt}],
                          "chat_template_kwargs": {"enable_thinking": False}})
            if abs(tokens["count"] - args.context_tokens) <= 128:
                break
            n = max(1, min(len(words), round(n * (args.context_tokens - 32) / tokens["count"])))
        else:
            raise RuntimeError("Long prompt token calibration failed")
        # Align the reusable prefix to both the target's global DCP block (128)
        # and replicated drafter block (64), leaving at least one uncached token.
        prefix_tokens = tokens["tokens"][:((tokens["count"] - 1) // 128) * 128]
        prime = post("/v1/chat/completions", {"model": "glm-5.3", "messages":
                     [{"role": "user", "content": count_prompt}], "max_tokens": 1,
                     "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        (args.out / "primed-context.json").write_text(json.dumps({"prompt_tokens": tokens["count"],
            "shared_prefix_tokens": len(prefix_tokens), "prompt_sha256": hashlib.sha256(count_prompt.encode()).hexdigest(),
            "response": prime}, indent=2) + "\n")
        print("Primed shared context:", tokens["count"], "tokens", flush=True)
    started = [threading.Event(), threading.Event()]
    injections = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        streams = [pool.submit(count_stream, i, started[i]) for i in range(2)]
        for event in started:
            assert event.wait(60), "Speculative stream did not begin"
        for length in (5, 3, 7):
            pending = sum(not f.done() for f in streams)
            assert pending == 2, "Need both streams still decoding at injection"
            begin = time.monotonic()
            response = post("/v1/completions", {"model": "glm-5.3", "prompt": prefix_tokens + [token] * length,
                                              "max_tokens": 16, "temperature": 0})
            assert response["usage"]["prompt_tokens"] == len(prefix_tokens) + length
            assert response["usage"]["completion_tokens"] > 0
            injections.append({"suffix_tokens": length, "shared_prefix_tokens": len(prefix_tokens),
                               "pending_long_streams": pending,
                               "response": response, "wall_s": time.monotonic() - begin})
        results = [f.result() for f in streams]
    summary = {"ok": True, "injections": injections,
               "streams": [{k: v for k, v in r.items() if k != "text"} for r in results]}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
