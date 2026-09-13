#!/usr/bin/env python3
"""Prove same-model durable KV survives a prepared-weight worker restart."""
import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runtime/vllm029"))
from rollout import request, BASE

OUT = ROOT / "results/nvme-loader"
BODY = {"model": "glm-5.3", "messages": [
    {"role": "system", "content": "Reference words for this loader durability check:\n" +
     ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu\n" * 128)},
    {"role": "user", "content": "Reply with exactly NVME_READY and nothing else."}],
    "max_tokens": 32, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False}, "return_token_ids": True}


def hits():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        lines = r.read().decode().splitlines()
    return {name: sum(float(line.rsplit(" ", 1)[1]) for line in lines
                      if line.startswith("vllm:" + name + "{"))
            for name in ("prefix_cache_hits_total", "external_prefix_cache_hits_total")}


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("mode", choices=("record", "verify"))
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if a.mode == "record":
        request(BODY, "/v1/chat/completions", timeout=180)
    before = hits()
    result = request(BODY, "/v1/chat/completions", timeout=180)
    content = result["choices"][0]["message"]["content"].strip()
    if content != "NVME_READY":
        raise RuntimeError("durability probe generated incorrect text: " + repr(content))
    for _ in range(10):
        after = hits()
        deltas = {name: after[name] - before[name] for name in after}
        cached = sum(deltas.values())
        if cached >= 256:
            break
        time.sleep(1)
    if cached < 256:
        raise RuntimeError("durability probe did not hit a complete target KV block")
    result["cache_metric_deltas"] = deltas
    (a.output or OUT / ("kv-" + a.mode + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"mode": a.mode, "cached_tokens": cached,
                      "prompt_tokens": result["usage"]["prompt_tokens"], "content": content}))
