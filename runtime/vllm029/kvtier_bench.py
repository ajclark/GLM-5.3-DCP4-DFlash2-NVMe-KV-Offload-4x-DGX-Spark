#!/usr/bin/env python3
"""Before/after benchmark for the NVMe KV tier (MultiNodeSlabConnector).

Stdlib only; runs from any host against the live endpoint. Prompts are token-id
lists built from a tokenized text corpus, so context lengths are exact and every
context starts with a unique nonce (no accidental prefix sharing).

Phases (select with --phases, comma separated):
  iso     isolated reload: cold-prefill a context, reset ONLY the GPU prefix cache
          (POST /reset_prefix_cache, needs VLLM_SERVER_DEV_MODE=1; the connector is
          never reset), re-send it: TTFT = NVMe restore. Disk read rate is sampled on
          every Spark while the restore runs.
  agents  5 concurrent simulated agents (shared prefix + unique 50-80k history),
          several rounds of appended turns with think time: TTFT p50/p95, queueing,
          external hit rate, reload GB/s, errors.
  decode  C1 and C4 decode tok/s (streamed), plus C1 decode while an 80k context
          is being restored from NVMe (only when iso is possible).

Never calls /reset_prefix_cache with reset_connector (that bumps the slab epoch =
wipes the whole tier).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HOSTS = ["spark-06c4", "spark-365c", "spark-ddbf", "spark-a218"]
REPO = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------- http
def post(url: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else {}


def get_text(url: str, timeout: float = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def stream_completion(base: str, model: str, prompt_ids: list[int], max_tokens: int,
                      timeout: float = 3600, ignore_eos: bool = True) -> dict:
    """Streamed /v1/completions. Returns send/first/last token wall times and token
    arrival times (one entry per streamed chunk with text)."""
    body = {"model": model, "prompt": prompt_ids, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True, "ignore_eos": ignore_eos,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    t_send = time.time()
    arrivals: list[float] = []
    usage = None
    text_parts: list[str] = []
    err = None
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
                    if ch.get("text"):
                        arrivals.append(time.time())
                        text_parts.append(ch["text"])
    except Exception as e:  # noqa: BLE001 - recorded, not raised
        err = f"{type(e).__name__}: {e}"
    t_end = time.time()
    return {"t_send": t_send, "t_first": arrivals[0] if arrivals else None, "t_end": t_end,
            "ttft": (arrivals[0] - t_send) if arrivals else None, "n_chunks": len(arrivals),
            "arrivals": arrivals, "usage": usage, "error": err, "text": "".join(text_parts),
            "prompt_len": len(prompt_ids)}


# -------------------------------------------------------------------------- metrics
_METRIC_RE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf)$')


def scrape(base: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in get_text(base + "/metrics").splitlines():
        if not line or line.startswith("#"):
            continue
        m = _METRIC_RE.match(line)
        if not m:
            continue
        name, labels, val = m.group(1), m.group(2) or "", m.group(3)
        if not name.startswith("vllm:"):
            continue
        labels = re.sub(r'(engine|model_name)="[^"]*",?', "", labels).replace("{}", "")
        try:
            out[name + labels] = out.get(name + labels, 0.0) + float(val)
        except ValueError:
            pass
    return out


COUNTERS = [
    "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
    "vllm:kv_offload_load_bytes_total", "vllm:kv_offload_load_time_total",
    "vllm:kv_offload_store_bytes_total", "vllm:kv_offload_store_time_total",
    "vllm:kv_offload_lookup_async_delay_seconds_sum", "vllm:kv_offload_lookup_async_delay_seconds_count",
    "vllm:request_queue_time_seconds_sum", "vllm:request_queue_time_seconds_count",
    "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count",
    "vllm:num_preemptions_total",
]


def counter_delta(a: dict, b: dict) -> dict:
    d = {k.replace("vllm:", ""): b.get(k, 0.0) - a.get(k, 0.0) for k in COUNTERS}
    q = d["external_prefix_cache_queries_total"]
    d["external_hit_rate"] = d["external_prefix_cache_hits_total"] / q if q else None
    q = d["prefix_cache_queries_total"]
    d["gpu_hit_rate"] = d["prefix_cache_hits_total"] / q if q else None
    t = d["kv_offload_load_time_total"]
    d["load_GBps_jobtime"] = d["kv_offload_load_bytes_total"] / t / 1e9 if t else None
    t = d["kv_offload_store_time_total"]
    d["store_GBps_jobtime"] = d["kv_offload_store_bytes_total"] / t / 1e9 if t else None
    return d


class MetricsPoller(threading.Thread):
    """1 Hz gauges: running/waiting/kv usage."""

    def __init__(self, base: str):
        super().__init__(daemon=True)
        self.base, self.samples, self._stop = base, [], threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                m = scrape(self.base)
                self.samples.append({
                    "t": time.time(),
                    "running": m.get("vllm:num_requests_running", 0.0),
                    "waiting": m.get("vllm:num_requests_waiting", 0.0),
                    "waiting_capacity": m.get('vllm:num_requests_waiting_by_reason{reason="capacity"}', 0.0),
                    "waiting_deferred": m.get('vllm:num_requests_waiting_by_reason{reason="deferred"}', 0.0),
                    "kv_usage": m.get("vllm:kv_cache_usage_perc", 0.0),
                })
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(1.0)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)


# ------------------------------------------------------------------------ disk rate
DISK_SAMPLER = r'''
import time, sys
dev = sys.argv[1]; period = float(sys.argv[2]); end = time.time() + float(sys.argv[3])
while time.time() < end:
    with open("/proc/diskstats") as f:
        for l in f:
            p = l.split()
            if p[2] == dev:
                print(f"{time.time():.3f} {p[5]} {p[9]}", flush=True)
                break
    time.sleep(period)
'''


class DiskSampler:
    """Samples sectors read/written on each Spark's NVMe (10 Hz) for a bounded time."""

    def __init__(self, hosts=HOSTS, dev="nvme0n1", period=0.1, max_seconds=7200):
        import tempfile
        self.procs = {}
        self.files = {}
        for h in hosts:
            # a file, not a pipe: nothing drains stdout while the benchmark runs
            f = tempfile.TemporaryFile(mode="w+")
            p = subprocess.Popen(
                ["ssh", "-o", "ConnectTimeout=5", f"{h}.local", "python3", "-",
                 dev, str(period), str(max_seconds)],
                stdin=subprocess.PIPE, stdout=f, stderr=subprocess.DEVNULL, text=True)
            p.stdin.write(DISK_SAMPLER)
            p.stdin.close()
            self.procs[h], self.files[h] = p, f
        self.series: dict[str, list[tuple[float, int, int]]] = {}

    def stop(self):
        for h, p in self.procs.items():
            p.terminate()
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
            self.files[h].seek(0)
            out = self.files[h].read()
            rows = []
            for line in out.splitlines():
                try:
                    t, r, w = line.split()
                    rows.append((float(t), int(r), int(w)))
                except ValueError:
                    pass
            self.series[h] = rows
        return self.series

    def window(self, host: str, t0: float, t1: float) -> dict:
        """Read/write stats inside [t0, t1]: total GB, mean GB/s, peak 0.5 s GB/s."""
        rows = [r for r in self.series.get(host, []) if t0 - 0.2 <= r[0] <= t1 + 0.2]
        if len(rows) < 2:
            return {}
        rd = (rows[-1][1] - rows[0][1]) * 512
        wr = (rows[-1][2] - rows[0][2]) * 512
        dt = rows[-1][0] - rows[0][0]
        peak = 0.0
        j = 0
        for i in range(len(rows)):
            while rows[i][0] - rows[j][0] > 0.5:
                j += 1
            if rows[i][0] - rows[j][0] >= 0.25:
                peak = max(peak, (rows[i][1] - rows[j][1]) * 512 / (rows[i][0] - rows[j][0]))
        return {"read_GB": rd / 1e9, "write_GB": wr / 1e9, "seconds": dt,
                "read_GBps_mean": rd / dt / 1e9 if dt else None, "read_GBps_peak_0p5s": peak / 1e9}


# --------------------------------------------------------------------------- corpus
def load_corpus_tokens(base: str, model: str, cache: Path) -> list[int]:
    if cache.exists():
        return json.loads(cache.read_text())
    roots = [REPO / "docs", REPO / "runtime"]
    files = sorted(p for r in roots if r.exists() for p in r.rglob("*")
                   if p.suffix in (".py", ".md") and p.is_file() and p.stat().st_size < 400_000)
    text = "\n".join(p.read_text(errors="replace") for p in files)
    toks: list[int] = []
    step = 200_000
    for i in range(0, len(text), step):
        toks += post(base + "/tokenize", {"model": model, "prompt": text[i:i + step],
                                          "add_special_tokens": False})["tokens"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(toks))
    return toks


class ContextFactory:
    def __init__(self, base, model, corpus: list[int], seed: int):
        self.base, self.model, self.corpus = base, model, corpus
        self.rng = random.Random(seed)

    def nonce(self, tag: str) -> list[int]:
        s = f"### session {tag} {self.rng.getrandbits(64):016x}\n"
        return post(self.base + "/tokenize", {"model": self.model, "prompt": s,
                                              "add_special_tokens": False})["tokens"]

    def body(self, n: int) -> list[int]:
        out: list[int] = []
        while len(out) < n:
            start = self.rng.randrange(0, len(self.corpus) - 4096)
            out += self.corpus[start:start + self.rng.randrange(512, 4096)]
        return out[:n]

    def context(self, tag: str, n: int) -> list[int]:
        nn = self.nonce(tag)
        return nn + self.body(n - len(nn))


# --------------------------------------------------------------------------- phases
def reset_gpu_prefix_cache(base: str) -> bool:
    for _ in range(60):
        try:
            req = urllib.request.Request(base + "/reset_prefix_cache", data=b"", method="POST")
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status == 200:
                    return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
        time.sleep(1.0)
    return False


def wait_store_quiet(base: str, settle: float = 3.0, limit: float = 300.0):
    """Wait until kv_offload store bytes stop growing (stores drained)."""
    t0 = time.time()
    last = scrape(base).get("vllm:kv_offload_store_bytes_total", 0.0)
    quiet_since = time.time()
    while time.time() - t0 < limit:
        time.sleep(0.5)
        cur = scrape(base).get("vllm:kv_offload_store_bytes_total", 0.0)
        if cur != last:
            last, quiet_since = cur, time.time()
        elif time.time() - quiet_since >= settle:
            return time.time() - t0
    return None


def phase_iso(args, cf_, disk: DiskSampler, log) -> dict:
    base, model = args.base, args.model
    res = {"samples": []}
    if not reset_gpu_prefix_cache(base):
        log("iso: /reset_prefix_cache unavailable (VLLM_SERVER_DEV_MODE off) - skipping")
        res["skipped"] = "no dev-mode reset endpoint"
        return res
    for size in args.iso_sizes:
        ctx = cf_.context(f"iso-{size}", size)
        m0 = scrape(base)
        cold = stream_completion(base, model, ctx, 1)
        m1 = scrape(base)
        store_wait = wait_store_quiet(base)
        log(f"iso {size}: cold TTFT {cold['ttft']:.2f}s, store drained after {store_wait}s")
        s = {"size": size, "cold_ttft": cold["ttft"], "cold_error": cold["error"],
             "cold_counters": counter_delta(m0, m1), "store_drain_s": store_wait, "reps": []}
        # GPU-hit control (no reset): same context + fresh 1-token suffix
        ctl = stream_completion(base, model, ctx + cf_.body(16), 1)
        s["gpu_hit_ttft"] = ctl["ttft"]
        for rep in range(args.iso_reps):
            ok = reset_gpu_prefix_cache(base)
            time.sleep(0.5)
            a = scrape(base)
            r = stream_completion(base, model, ctx + cf_.body(16), 1)
            b = scrape(base)
            d = counter_delta(a, b)
            ttft = r["ttft"]
            per_rank_bytes = d["kv_offload_load_bytes_total"] / 4 if d["kv_offload_load_bytes_total"] else 0
            rep_res = {"reset_ok": ok, "ttft": ttft, "error": r["error"],
                       "external_hits": d["external_prefix_cache_hits_total"],
                       "load_bytes_all_ranks": d["kv_offload_load_bytes_total"],
                       "load_jobtime_s": d["kv_offload_load_time_total"],
                       "restore_GBps_per_rank_ttft": (per_rank_bytes / ttft / 1e9) if ttft and per_rank_bytes else None,
                       "_window": (r["t_send"], r["t_first"] or r["t_end"])}
            s["reps"].append(rep_res)
            log(f"iso {size} rep{rep}: TTFT {ttft:.3f}s hits {rep_res['external_hits']:.0f} "
                f"load {per_rank_bytes/1e9:.2f} GB/rank -> {rep_res['restore_GBps_per_rank_ttft'] or 0:.2f} GB/s/rank")
        res["samples"].append(s)
    return res


def phase_verify(args, cf_, log) -> dict:
    """Greedy output from NVMe-restored KV must equal output from computed KV."""
    base, model = args.base, args.model
    out = {"cases": []}
    if not reset_gpu_prefix_cache(base):
        return {"skipped": "no dev-mode reset endpoint"}
    for size in (8000, 60000):
        ctx = cf_.context(f"verify-{size}", size) + post(base + "/tokenize", {
            "model": model, "prompt": "\n\nSummarize the text above in three sentences:\n",
            "add_special_tokens": False})["tokens"]
        reset_gpu_prefix_cache(base)
        cold = stream_completion(base, model, ctx, 64, ignore_eos=True)
        wait_store_quiet(base)
        gpu = stream_completion(base, model, ctx, 64, ignore_eos=True)
        reset_gpu_prefix_cache(base)
        a = scrape(base)
        nvme = stream_completion(base, model, ctx, 64, ignore_eos=True)
        b = scrape(base)
        hits = b.get("vllm:external_prefix_cache_hits_total", 0) - a.get("vllm:external_prefix_cache_hits_total", 0)
        case = {"size": len(ctx), "nvme_hit_tokens": hits, "gpu_equals_cold": gpu["text"] == cold["text"],
                "nvme_equals_cold": nvme["text"] == cold["text"], "cold": cold["text"][:200],
                "nvme": nvme["text"][:200], "errors": [x["error"] for x in (cold, gpu, nvme) if x["error"]]}
        out["cases"].append(case)
        log(f"verify {len(ctx)}: nvme hits {hits:.0f}, gpu==cold {case['gpu_equals_cold']}, "
            f"nvme==cold {case['nvme_equals_cold']}")
    return out


def _complete_logprobs(base, model, ids, n):
    body = {"model": model, "prompt": ids, "max_tokens": n, "temperature": 0.0, "logprobs": 5}
    r = post(base + "/v1/completions", body)
    ch = r["choices"][0]
    return ch["text"], ch["logprobs"]["token_logprobs"], ch["logprobs"]["top_logprobs"]


def phase_needles(args, cf_, log) -> dict:
    """Placement check: codes spread over the whole context must be recalled from
    NVMe-restored KV as well as from computed KV; first-token logprobs compared
    across cold / GPU-hit / NVMe-restored."""
    base, model = args.base, args.model
    if not reset_gpu_prefix_cache(base):
        return {"skipped": "no dev-mode reset endpoint"}
    rng = random.Random(args.seed + 99)
    out = {"cases": []}
    for size in (60000, 150000):
        codes = [f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{rng.randrange(1000, 9999)}"
                 for _ in range(10)]
        tok = lambda t: post(base + "/tokenize", {"model": model, "prompt": t, "add_special_tokens": False})["tokens"]
        seg = size // 10
        ids = cf_.nonce(f"needles-{size}")
        for i, c in enumerate(codes):
            ids += cf_.body(seg - 40) + tok(f"\n\nIMPORTANT: secret code number {i + 1} is {c}.\n\n")
        ids += tok("\n\nQuestion: list secret codes number 1 through 10, in order, one per line, "
                   "formatted as 'N: CODE'.\nAnswer:\n1:")
        reset_gpu_prefix_cache(base)
        cold_text, cold_lp, cold_top = _complete_logprobs(base, model, ids, 120)
        wait_store_quiet(base)
        gpu_text, gpu_lp, _ = _complete_logprobs(base, model, ids, 120)
        reset_gpu_prefix_cache(base)
        a = scrape(base)
        nv_text, nv_lp, nv_top = _complete_logprobs(base, model, ids, 120)
        b = scrape(base)
        hits = b.get("vllm:external_prefix_cache_hits_total", 0) - a.get("vllm:external_prefix_cache_hits_total", 0)
        score = lambda t: sum(1 for c in codes if c in t)
        k = 16
        d_gpu = max(abs(x - y) for x, y in zip(cold_lp[:k], gpu_lp[:k]))
        d_nv = max(abs(x - y) for x, y in zip(cold_lp[:k], nv_lp[:k]))
        case = {"size": len(ids), "nvme_hit_tokens": hits, "codes": codes,
                "recall_cold": score(cold_text), "recall_gpu": score(gpu_text), "recall_nvme": score(nv_text),
                "max_abs_logprob_diff_first16": {"gpu_vs_cold": d_gpu, "nvme_vs_cold": d_nv},
                "cold": cold_text, "nvme": nv_text}
        out["cases"].append(case)
        log(f"needles {len(ids)}: nvme hits {hits:.0f}; recall cold {case['recall_cold']}/10 gpu "
            f"{case['recall_gpu']}/10 nvme {case['recall_nvme']}/10; max |dlogprob| first16 "
            f"gpu {d_gpu:.4f} nvme {d_nv:.4f}")
    return out


def phase_agents(args, cf_, log) -> dict:
    base, model = args.base, args.model
    rng = random.Random(args.seed + 1)
    shared = cf_.context("shared-system", args.shared_prefix)
    agents = []
    for i in range(args.agents):
        hist = rng.randrange(args.hist_min, args.hist_max)
        agents.append({"id": i, "ctx": shared + cf_.context(f"agent{i}", hist), "turns": []})
    poller = MetricsPoller(base)
    poller.start()
    m0 = scrape(base)
    t_start = time.time()
    lock = threading.Lock()

    def run_agent(a):
        for rnd in range(args.rounds):
            if rnd:
                time.sleep(rng.uniform(args.think_min, args.think_max))
                a["ctx"] = a["ctx"] + cf_.body(rng.randrange(args.turn_min, args.turn_max))
            r = stream_completion(base, model, a["ctx"], args.gen_tokens)
            # the model's reply becomes part of the history (as in an agent loop)
            a["ctx"] = a["ctx"] + cf_.body(args.gen_tokens)
            rec = {"round": rnd, "prompt_len": r["prompt_len"], "ttft": r["ttft"],
                   "e2e": r["t_end"] - r["t_send"], "error": r["error"], "t_send": r["t_send"]}
            with lock:
                a["turns"].append(rec)
            log(f"agent{a['id']} round{rnd}: len {rec['prompt_len']} TTFT "
                f"{(rec['ttft'] or float('nan')):.2f}s e2e {rec['e2e']:.1f}s {rec['error'] or ''}")

    with cf.ThreadPoolExecutor(args.agents) as ex:
        list(ex.map(run_agent, agents))
    m1 = scrape(base)
    poller.stop()
    measured = [t for a in agents for t in a["turns"] if t["round"] >= 1]
    ttfts = sorted(t["ttft"] for t in measured if t["ttft"] is not None)

    def pct(xs, p):
        if not xs:
            return None
        k = min(len(xs) - 1, max(0, int(round(p / 100 * (len(xs) - 1)))))
        return xs[k]
    return {
        "wall_s": time.time() - t_start,
        "counters": counter_delta(m0, m1),
        "ttft_warm_rounds": {"n": len(ttfts), "p50": pct(ttfts, 50), "p95": pct(ttfts, 95),
                             "max": ttfts[-1] if ttfts else None,
                             "mean": statistics.mean(ttfts) if ttfts else None},
        "ttft_round0": [t["ttft"] for a in agents for t in a["turns"] if t["round"] == 0],
        "errors": [t["error"] for a in agents for t in a["turns"] if t["error"]],
        "max_waiting": max((s["waiting"] for s in poller.samples), default=None),
        "max_waiting_capacity": max((s["waiting_capacity"] for s in poller.samples), default=None),
        "max_waiting_deferred": max((s["waiting_deferred"] for s in poller.samples), default=None),
        "turns": [{"agent": a["id"], **t} for a in agents for t in a["turns"]],
        "gauges": poller.samples,
    }


def decode_rate(r: dict) -> float | None:
    arr = r["arrivals"]
    if len(arr) < 3:
        return None
    toks = (r["usage"] or {}).get("completion_tokens") or len(arr)
    return (toks - 1) / (arr[-1] - arr[0]) if arr[-1] > arr[0] else None


def phase_decode(args, cf_, disk, log, can_reset: bool) -> dict:
    base, model = args.base, args.model
    prompts = {
        "count": "Count from 1 to 2000, separated by commas:\n1, 2, 3, 4,",
        "prose": "Write a long, detailed essay about the history of the printing press.\n\n",
        "code": "# A complete Python implementation of a red-black tree with insert, delete and search.\n\nclass Node:\n",
    }
    ids = {k: post(base + "/tokenize", {"model": model, "prompt": v})["tokens"] for k, v in prompts.items()}
    out = {"c1": {}, "c4": {}}
    for k, p in ids.items():
        rates = []
        for _ in range(args.decode_reps):
            r = stream_completion(base, model, p, args.decode_tokens)
            rates.append(decode_rate(r))
        out["c1"][k] = rates
        log(f"decode C1 {k}: {[round(x or 0, 1) for x in rates]} tok/s")
    for k, p in ids.items():
        with cf.ThreadPoolExecutor(4) as ex:
            rs = list(ex.map(lambda _: stream_completion(base, model, p, args.decode_tokens), range(4)))
        per = [decode_rate(r) for r in rs]
        t0 = min(r["t_first"] for r in rs if r["t_first"])
        t1 = max(r["t_end"] for r in rs)
        agg = sum((r["usage"] or {}).get("completion_tokens", 0) for r in rs) / (t1 - t0)
        out["c4"][k] = {"per_stream": per, "aggregate": agg}
        log(f"decode C4 {k}: aggregate {agg:.1f} tok/s")
    if can_reset:
        # C1 decode while an 80k context is restored from NVMe
        ctx = cf_.context("decode-under-load", args.iso_sizes[-1] if args.iso_sizes else 80000)
        stream_completion(base, model, ctx, 1)
        wait_store_quiet(base)
        reset_gpu_prefix_cache(base)
        time.sleep(0.5)
        holder: dict = {}
        th = threading.Thread(target=lambda: holder.update(
            r=stream_completion(base, model, ids["count"], args.decode_tokens * 2)))
        th.start()
        time.sleep(8)
        rl = stream_completion(base, model, ctx + cf_.body(16), 1)
        th.join()
        dr = holder["r"]
        w0, w1 = rl["t_send"], (rl["t_first"] or rl["t_end"])
        inside = [t for t in dr["arrivals"] if w0 <= t <= w1]
        before = [t for t in dr["arrivals"] if t < w0]
        rate_in = (len(inside) - 1) / (inside[-1] - inside[0]) if len(inside) > 2 else None
        rate_before = (len(before) - 1) / (before[-1] - before[0]) if len(before) > 2 else None
        out["c1_during_reload"] = {"reload_ttft": rl["ttft"], "chunks_per_s_during": rate_in,
                                   "chunks_per_s_before": rate_before,
                                   "_window": (w0, w1)}
        log(f"decode during reload: reload TTFT {rl['ttft']:.2f}s, stream chunk rate "
            f"{rate_before} -> {rate_in} /s")
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--base", default="http://spark-06c4.local:8000")
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--phases", default="iso,agents,decode")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "nvme-kvtier-20260928")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--iso-sizes", type=lambda s: [int(x) for x in s.split(",")], default=[20000, 80000, 160000])
    ap.add_argument("--iso-reps", type=int, default=3)
    ap.add_argument("--agents", type=int, default=5)
    ap.add_argument("--shared-prefix", type=int, default=12000)
    ap.add_argument("--hist-min", type=int, default=50000)
    ap.add_argument("--hist-max", type=int, default=80000)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--turn-min", type=int, default=1000)
    ap.add_argument("--turn-max", type=int, default=3000)
    ap.add_argument("--think-min", type=float, default=5.0)
    ap.add_argument("--think-max", type=float, default=15.0)
    ap.add_argument("--gen-tokens", type=int, default=200)
    ap.add_argument("--decode-tokens", type=int, default=512)
    ap.add_argument("--decode-reps", type=int, default=3)
    args = ap.parse_args()

    outdir = args.out / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    logf = open(outdir / "bench.log", "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    corpus = load_corpus_tokens(args.base, args.model, args.out / "corpus-tokens.json")
    log(f"corpus: {len(corpus)} tokens; label {args.label}; phases {args.phases}")
    cf_ = ContextFactory(args.base, args.model, corpus, args.seed)
    phases = args.phases.split(",")
    result = {"label": args.label, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "metrics_start": scrape(args.base)}
    disk = DiskSampler()
    time.sleep(1.0)
    try:
        if "needles" in phases:
            result["needles"] = phase_needles(args, cf_, log)
        if "verify" in phases:
            result["verify"] = phase_verify(args, cf_, log)
        if "iso" in phases:
            result["iso"] = phase_iso(args, cf_, disk, log)
        if "agents" in phases:
            result["agents"] = phase_agents(args, cf_, log)
        if "decode" in phases:
            can_reset = "iso" in result and "skipped" not in result["iso"]
            result["decode"] = phase_decode(args, cf_, disk, log, can_reset)
    finally:
        disk.stop()
        result["metrics_end"] = scrape(args.base)
        # attach disk read rates to every timed window
        for s in result.get("iso", {}).get("samples", []):
            for rep in s["reps"]:
                w0, w1 = rep.pop("_window")
                rep["disk"] = {h: disk.window(h, w0, w1) for h in HOSTS}
        dur = result.get("decode", {}).get("c1_during_reload")
        if dur:
            w0, w1 = dur.pop("_window")
            dur["disk"] = {h: disk.window(h, w0, w1) for h in HOSTS}
        if "agents" in result:
            t0 = min(t["t_send"] for t in result["agents"]["turns"])
            result["agents"]["disk"] = {h: disk.window(h, t0, time.time()) for h in HOSTS}
        (outdir / "disk-series.json").write_text(json.dumps(disk.series))
        (outdir / "result.json").write_text(json.dumps(result, indent=1, default=str))
        log(f"wrote {outdir / 'result.json'}")


if __name__ == "__main__":
    sys.exit(main())
