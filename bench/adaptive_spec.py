#!/usr/bin/env python3
"""Bounded, C1, paired-cap experiment with a mandatory four-node memory guard.

Runs in the sandbox. Only lightweight /proc readers run on the nodes. Records
raw token IDs and SSE timings; decode rate excludes the first emitted block.
The caller must have authorization to use the serving endpoint.
"""
import argparse
import hashlib
import json
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from spec_memory import MemoryGuard
from spec_think_utils import completion_accounting

PROMPTS = {
    "prose": "Write a detailed, engaging essay about how cities can make summer heat more bearable. Discuss trees, street design, housing, public transit, and the tradeoffs residents face. Use specific examples and connected prose. Aim for 1500 words.",
    "code": "Write a complete Python module implementing an LRU cache with expiration, a fake clock for testing, thread safety, and unittest tests covering eviction, TTL, replacement and concurrency. Include type annotations and concise docstrings. Return the code only.",
    "repo": "Review this asynchronous scheduler and provide a corrected implementation with tests. Explain race conditions and cancellation behavior.\n\nimport asyncio\nclass Runner:\n    def __init__(self):\n        self.pending = []\n        self.running = False\n    async def submit(self, job):\n        future = asyncio.Future()\n        self.pending.append((job, future))\n        if not self.running:\n            asyncio.create_task(self.run())\n        return await future\n    async def run(self):\n        self.running = True\n        while self.pending:\n            job, future = self.pending.pop(0)\n            future.set_result(await job())\n        self.running = False\n",
}


def request_body(prompt, cap, label, max_tokens):
    return {"model":"glm-5.3", "messages":[{"role":"user", "content":prompt}],
            "temperature":0, "seed":42, "max_tokens":max_tokens,
            "stream":True, "stream_options":{"include_usage":True},
            "return_token_ids":True, "chat_template_kwargs":{"enable_thinking":False},
            "vllm_xargs":{"spec_policy":"fixed", "spec_verify_cap":cap, "spec_label":label}}


def lossy_xargs(name):
    """Parse the bounded rank-two family; never silently disable a bad request."""
    number = r'(?:\d+(?:\.\d+)?|\.\d+)'
    match = re.fullmatch(rf'lossy-(think-)?m({number})(?:-p({number}))?', name)
    if not match:
        raise ValueError('invalid lossy variant')
    margin, min_p = float(match[2]), float(match[3] or 0)
    if not 0 < margin <= 5 or not 0 <= min_p < .5:
        raise ValueError('lossy margin must be in (0,5]; min_p in [0,0.5)')
    fields = dict(spec_lossy_margin=margin, spec_lossy_rank=2, spec_lossy_min_p=min_p)
    if match[1]:
        fields['spec_lossy_scope'] = 'think'
    return fields


def parse_variants(names):
    allowed = {**{f'fixed{k}': ('fixed', k) for k in (1, 3, 5, 7)},
               'adaptive': ('adaptive', 7), 'shadow': ('shadow', 7)}
    variants = []
    for name in names.split(','):
        if name.startswith('lossy-'):
            lossy_xargs(name)
            variants.append((name, 7))
        elif name in allowed:
            variants.append(allowed[name])
        else:
            raise ValueError('invalid variant: ' + name)
    return variants


def select_cases(prompts, cases=None, prose_only=False):
    selected = cases.split(',') if cases else list(prompts)
    if not all(case in prompts for case in selected):
        raise ValueError('unknown corpus case')
    if prose_only:
        selected = [case for case in selected if case.startswith('prose_')]
    if not selected:
        raise ValueError('no selected corpus cases')
    return selected


def variant_body(prompt, mode, cap, label, max_tokens):
    body = request_body(prompt, cap, label, max_tokens)
    body['vllm_xargs']['spec_policy'] = 'fixed' if mode.startswith('lossy-') else mode
    if mode.startswith('lossy-'):
        body['vllm_xargs'].update(lossy_xargs(mode))
    return body


def idle_check(base):
    with urllib.request.urlopen(base+"/metrics", timeout=5) as response:
        metrics = response.read().decode()
    found = set()
    for line in metrics.splitlines():
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{")):
            name = line.split("{",1)[0]
            found.add(name)
            if float(line.rsplit(" ",1)[1]) != 0:
                raise RuntimeError("serving endpoint has other work; experiment refused")
    if len(found) != 2:
        raise RuntimeError("unable to verify endpoint idle state")


def spec_metrics(base):
    with urllib.request.urlopen(base+'/metrics',timeout=5) as response:
        lines = response.read().decode().splitlines()
    return {line.rsplit(' ',1)[0]:float(line.rsplit(' ',1)[1]) for line in lines
            if line.startswith('vllm:spec_decode_num_') and '_created' not in line}


def add_costs(body,costs):
    if 'calibration_points' in costs:
        body['vllm_xargs']['spec_cost_table']=json.dumps(costs,separators=(',',':'))
        return
    body['vllm_xargs'].update(spec_cycle_ms=[costs['cycle_ms'][str(k)] for k in (1,3,5,7)],
                             spec_cost_lane=costs['lane'],spec_cost_context=costs['context_range'])
    if 'conditional_acceptance_prior' in costs:
        body['vllm_xargs']['spec_acceptance_prior']=costs['conditional_acceptance_prior']


def generate(base, body, guard, deadline_seconds=300):
    chunks, errors, response_slot = [], [], []
    start = time.monotonic()
    started_at = time.time()
    def read():
        try:
            req = urllib.request.Request(base+"/v1/chat/completions", json.dumps(body).encode(),
                                         {"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=min(deadline_seconds,600)) as response:
                response_slot.append(response)
                for line in response:
                    if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                        chunks.append({"seconds":time.monotonic()-start, "data":json.loads(line[6:])})
        except urllib.error.HTTPError as exc:
            errors.append(f'HTTP {exc.code}: '+exc.read(4096).decode(errors='replace'))
        except Exception as exc:
            errors.append(str(exc))
    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    try:
        while thread.is_alive():
            guard.check()
            if time.monotonic()-start > deadline_seconds:
                raise RuntimeError("bounded generation deadline exceeded")
            thread.join(timeout=0.2)
    except BaseException:
        # Closing the stream cancels this request at the API server.
        if response_slot:
            try:
                response_slot[0].fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError,OSError):
                pass
            response_slot[0].close()
        raise
    if errors:
        raise RuntimeError(errors[0])
    ids, text, reasoning, token_chunks, usage = [], "", "", [], None
    reasoning_field_present = False
    for row in chunks:
        data = row["data"]
        if "error" in data:
            raise RuntimeError(str(data["error"]))
        if data.get("usage"):
            usage = data["usage"]
        for choice in data.get("choices", []):
            reasoning_field_present |= any(field in choice.get('delta', {}) for field in ('reasoning', 'reasoning_content'))
            new_ids = choice.get("token_ids") or []
            ids.extend(new_ids)
            text += choice.get("delta", {}).get("content") or ""
            reasoning += (choice.get('delta',{}).get('reasoning')
                          or choice.get('delta',{}).get('reasoning_content') or '')
            if new_ids:
                token_chunks.append((row["seconds"], len(new_ids)))
    if not usage or not ids or usage["completion_tokens"] != len(ids):
        raise RuntimeError("missing or inconsistent token IDs/usage")
    elapsed = token_chunks[-1][0] - token_chunks[0][0]
    rate = (len(ids)-token_chunks[0][1])/elapsed if elapsed > 0 else None
    return {"started_at":started_at,"token_ids":ids, "token_sha256":hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "text":text, "reasoning_text":reasoning, "usage":usage, "seconds":time.monotonic()-start,
            "ttft":token_chunks[0][0], "decode_tps":rate, "chunks":chunks,
            "thinking":bool(body.get('chat_template_kwargs', {}).get('enable_thinking')),
            "max_tokens":body.get('max_completion_tokens', body.get('max_tokens')),
            "reasoning_field_present":reasoning_field_present,
            **completion_accounting(text, reasoning, usage,
                                   thinking=bool(body.get('chat_template_kwargs', {}).get('enable_thinking')),
                                   reasoning_field_present=reasoning_field_present)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://spark-06c4.local:8000")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--caps", default="7")
    ap.add_argument("--cases", help="Comma-separated case names; default is all corpus cases")
    ap.add_argument("--prose-only", action="store_true", help="Select only prose_* corpus cases")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--policy", choices=["fixed","adaptive","shadow"],default="fixed")
    ap.add_argument("--costs",type=Path,help="Measured cycle cost table for same-boot request-scoped calibration")
    ap.add_argument("--corpus",type=Path,help="JSON mapping of case names to prompt strings or objects with a prompt field")
    ap.add_argument("--variants",help="Interleaved variants, e.g. fixed7,fixed3,adaptive or fixed7,lossy-m1.0-p0.1")
    ap.add_argument("--integration",action="store_true",help="Run bounded fallback, C1/C2 transition and cancellation probes")
    ap.add_argument("--context-smoke",action="store_true",help="Guarded 32k and 100k cold/warm verification checks")
    ap.add_argument("--code-smoke",action="store_true",help="Generate small complete functions and execute bounded sandbox checks")
    ap.add_argument("--deadline-seconds",type=int,default=300,help="Per-request bound; up to 900 seconds for cold long prompts")
    ap.add_argument("--thinking",action="store_true",help="Enable reasoning for every arm, sharing the --tokens completion budget")
    ap.add_argument("--thinking-repeat",action="append",type=int,default=[],help="Enable thinking for every arm in this zero-based repeat (repeatable), sharing --tokens")
    ap.add_argument("--repo-context",type=int,choices=[4000,32000,100000,170000],help="Token-counted repository reference with code/prose tasks")
    args = ap.parse_args()
    corpus = json.loads(args.corpus.read_text()) if args.corpus else PROMPTS
    prompts = {case: value['prompt'] if isinstance(value, dict) else value
               for case, value in corpus.items()}
    costs = json.loads(args.costs.read_text()) if args.costs else None
    try:
        # Repository cases are selected after the reference is built.
        cases = list(prompts) if args.repo_context else select_cases(prompts, args.cases, args.prose_only)
    except ValueError as exc:
        ap.error(str(exc))
    caps = [int(k) for k in args.caps.split(',')]
    if not set(caps) <= {1,3,5,7} or not 1 <= args.tokens <= 2048 or not 1 <= args.repeats <= 8:
        ap.error("caps must be 1,3,5,7; tokens 1..2048; repeats 1..8")
    if any(not 0 <= repeat < args.repeats for repeat in args.thinking_repeat):
        ap.error('thinking-repeat must identify a requested repeat')
    if not 30 <= args.deadline_seconds <= 900:
        ap.error('deadline must be 30..900 seconds')
    variants = [(args.policy,k) for k in caps]
    if args.variants:
        try:variants = parse_variants(args.variants)
        except ValueError as exc:ap.error(str(exc))
    lossy_study = any(mode.startswith('lossy-') for mode, cap in variants)
    if lossy_study and (len(set(variants)) != len(variants) or
                       any(mode != 'fixed' and not mode.startswith('lossy-') or cap != 7
                           for mode, cap in variants)):
        ap.error('lossy comparisons require distinct lossy variants and fixed7 only')
    if any(mode=='adaptive' for mode,cap in variants) and costs is None:
        ap.error('adaptive comparisons require a measured --costs table')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/"config.json").write_text(json.dumps({**vars(args), "out":str(args.out), "prompts":prompts},indent=2,default=str)+"\n")
    guard = MemoryGuard(args.out/"memory.jsonl").start()
    try:
        print("preflight:", guard.preflight(), flush=True)
        if args.repo_context:
            from spec_repo_context import build
            prompts=build(args.base,args.out,args.repo_context,guard)
            cases=select_cases(prompts,args.cases,args.prose_only)
            (args.out/'config.json').write_text(json.dumps({**vars(args),'out':str(args.out),'prompts':prompts},indent=2,default=str)+'\n')
        if args.integration:
            from spec_integration import run
            run(args.base,args.out,guard)
            return
        if args.context_smoke:
            from spec_context import run
            run(args.base,args.out,guard,costs)
            return
        if args.code_smoke:
            from spec_code_check import run
            run(args.base,args.out,guard,costs)
            return
        for repeat in range(args.repeats):
            for case_index,case in enumerate(cases):
                order = variants.copy()
                if len(order)==2:
                    # Counterbalance AB/BA over prompts as well as repetitions.
                    if (case_index+repeat)%2:
                        order.reverse()
                else:
                    random.Random(42+1009*repeat+case_index).shuffle(order)
                for mode,cap in order:
                    label = f"{args.out.name}-{case}-r{repeat}-{mode}-k{cap}"
                    if lossy_study and len(label) > 96:
                        raise ValueError('experiment label exceeds trace identity bound')
                    idle_check(args.base)
                    guard.preflight(seconds=4)
                    body = variant_body(prompts[case],mode,cap,label,args.tokens)
                    body['chat_template_kwargs']['enable_thinking'] = args.thinking or repeat in args.thinking_repeat
                    if costs:
                        add_costs(body,costs)
                    before = spec_metrics(args.base)
                    result = generate(args.base, body, guard,args.deadline_seconds)
                    after = spec_metrics(args.base)
                    result['spec_metric_delta'] = {k:after[k]-v for k,v in before.items() if k in after}
                    result['policy'] = mode
                    result.update(label=label, case=case, cap=cap, repeat=repeat,
                                  thinking=body['chat_template_kwargs']['enable_thinking'], max_tokens=args.tokens)
                    if lossy_study:
                        result.update(study='lossy', variant='fixed7' if mode == 'fixed' else mode,
                                      experiment_xargs=body['vllm_xargs'],
                                      message_sha256=hashlib.sha256(json.dumps(body['messages']).encode()).hexdigest())
                    (args.out/(label+".json")).write_text(json.dumps(result,indent=2)+"\n")
                    print(json.dumps({k:result[k] for k in ("label","decode_tps","ttft","token_sha256","usage")}), flush=True)
        print("final headroom:", guard.preflight(seconds=4), flush=True)
    finally:
        guard.close()


if __name__ == "__main__":
    main()
