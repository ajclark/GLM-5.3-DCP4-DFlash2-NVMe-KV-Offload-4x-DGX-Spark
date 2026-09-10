#!/usr/bin/env python3
"""Compare the deployed lane to the frozen repaired V5 adaptive build; restore exactly."""
import argparse
import asyncio
import contextlib
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import pi_humaneval as he
import spec_experiment as exp
from spec_memory import HOSTS, MemoryGuard

FROZEN = 'hints-conf-20260909-r2'
COSTS = he.ROOT / 'results/adaptive-next/cache-width-r1/costs-repaired-curve.json'
FILES = ('scheduler.py', 'adaptive.py', 'model_runner.py', 'cudagraph_utils.py',
         'v2_block_table.py', 'v2_async_utils.py', 'v1_outputs.py', 'confidence_trace.py')


def inventory():
    code = '''import json,subprocess,hashlib,pathlib
d=json.loads(subprocess.check_output(['docker','inspect','vllm_glm53big']))[0]
env=dict(e.split('=',1) for e in d['Config']['Env'])
keep={k:v for k,v in env.items() if k.startswith(('GLM_SPEC_','GLM_DCP_','NCCL_')) or k=='VLLM_MARLIN_USE_ATOMIC_ADD'}
mounts={m['Destination']:hashlib.sha256(pathlib.Path(m['Source']).read_bytes()).hexdigest() for m in d['Mounts'] if m['Source'].endswith('.py')}
gpu=subprocess.run(['nvidia-smi','--query-gpu=clocks.gr,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
print(json.dumps(dict(id=d['Id'],image=d['Image'],image_tag=d['Config']['Image'],cmd=d['Config']['Cmd'],environment=keep,mounts=mounts,started=d['State']['StartedAt'],running=d['State']['Running'],oom=d['State']['OOMKilled'],gpu=gpu.stdout.strip())))
'''
    return exp.parallel(lambda host: json.loads(exp.ssh(host, shlex.join(['python3', '-c', code]))))


def freeze_audit(out):
    local = {n: hashlib.sha256((he.ROOT/'stage/glm-dcp'/n).read_bytes()).hexdigest() for n in FILES}
    code = '''import pathlib,hashlib,json,sys
root=pathlib.Path.home()/'glm-spec'/sys.argv[1]/'changes'
print(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}))
'''
    remote = exp.parallel(lambda host: json.loads(exp.ssh(host, shlex.join(['python3', '-c', code, FROZEN]))))
    if any(row != local for row in remote.values()):
        raise RuntimeError('Current staging differs from the frozen V5 build; refusing an ambiguous comparison')
    he.write_json(out / 'adaptive-source-audit.json', {'frozen_label': FROZEN, 'local': local, 'nodes': remote,
                  'costs_sha256': hashlib.sha256(COSTS.read_bytes()).hexdigest(),
                  'policy': 'adaptive', 'confidence_trace': False, 'client_hints': False,
                  'trace': True, 'atomic_add': True})


def report(out):
    config = json.loads((out/'config.json').read_text()) if (out/'config.json').exists() else {}
    thinking = config.get('thinking', 'high')
    token_limit = config.get('max_tokens_override') or 32768
    rows = []
    for lane in ('deployed', 'adaptive', 'restored'):
        path = out / f'{lane}-summary.json'
        if path.exists(): rows += json.loads(path.read_text())
    he.write_json(out / 'summary.json', rows)
    aborted = (out/'ABORTED.json').exists()
    lines = ['# HumanEval through pi: deployed versus adaptive', '',
        'Run status: '+('aborted at user request; completed cells analyzed and incomplete cells excluded'
                       if aborted else 'complete' if (out/'COMPLETE').exists() else 'in progress; partial results')+'.', '',
        'Twelve preselected HumanEval tasks, one solve per task at every C1–C12. '
        'Real pi 0.85.1 RPC sessions launched by the benchmark in herdr; '
        f'GLM-5.3 settings, thinking {thinking}, temperature 0, top_p 1, {token_limit} output-token limit. '
        'The pi read/bash/edit/write tools and user SYSTEM.md are retained. '
        'Optional packages, skills, context files, and other extensions are disabled identically; '
        'the observer extension does not change provider payloads.', '',
        'The ten tasks from the previous HE10 benchmark are supplemented by HumanEval/0 and /22 '
        'so C12 can start twelve distinct tasks. Every cell uses all twelve tasks, a fixed shuffled '
        'order shared by both stacks, fresh sessions, a synchronized first wave, and at most C active '
        'sessions. A free slot takes the next task. This is a finite batch, not steady-state serving; '
        'tool gaps and the final draining tail reduce realized concurrency. Pi startup is excluded '
        'before the first wave; replacement-session startup is included in batch wall time.', '',
        'Aggregate tok/s = provider-reported output tokens / first task submission to last task completion. '
        'Per-request tok/s = output tokens / summed provider request times, including TTFT. '
        'Output tokens include any reasoning and tool-call serialization; reasoning is never added twice. '
        'Server decode tok/s additionally excludes server-reported queue/prefill intervals. '
        'These measure all agent calls, not just code in solution.py. The exported reasoning-token count '
        'can be zero when the backend omits that breakdown.', '',
        '| C | Deployed aggregate tok/s | Adaptive aggregate tok/s | Change | Deployed per-request tok/s | Adaptive per-request tok/s | Tests deployed / adaptive |',
        '|---:|---:|---:|---:|---:|---:|:---|']
    for c in range(1, 13):
        a = next((r for r in rows if r['lane']=='deployed' and r['concurrency']==c), None)
        b = next((r for r in rows if r['lane']=='adaptive' and r['concurrency']==c), None)
        if a and b:
            lines.append(f"| {c} | {a['aggregate_tok_s']:.2f} | {b['aggregate_tok_s']:.2f} | {(b['aggregate_tok_s']/a['aggregate_tok_s']-1)*100:+.1f}% | {a['request_tok_s']:.2f} | {b['request_tok_s']:.2f} | {a['passed']}/12 / {b['passed']}/12 |")
        elif a:
            state = ('incomplete; excluded' if (out/'adaptive'/f'c{c:02d}').exists() else 'not run') if aborted else 'pending'
            lines.append(f"| {c} | {a['aggregate_tok_s']:.2f} | {state} | — | {a['request_tok_s']:.2f} | — | {a['passed']}/12 / — |")
    lines += ['', 'The adaptive runtime is the exact source set saved as '+FROZEN+
              ', using its repaired measured cost curve. Confidence collection and client workload hints '
              'are off. Policy trace is on. The controller is eligible only for a single pure-decode '
              'request and otherwise falls back to K7; C>1 is a concurrency/fallback comparison, '
              'not a claim of multi-request adaptive verification. The experimental runtime also includes '
              'the replicated draft-cache repair, so this is a deployed-stack comparison, not an '
              'isolated controller-only A/B.', '',
              'All deployed cells run before all adaptive cells, with equal per-lane warmups; '
              'prefix caches are left enabled and can warm naturally. Cold cache state and temporal '
              'drift are not eliminated. '+
              ('After the requested abort, restoration uses one short native pi probe; the optional '
               'restored baseline C1/C12 repeatability batches are omitted. '
               if aborted else 'Restored baseline C1/C12 anchors follow when the run completes. ')+
              'Those restoration checks use fresh scratch paths, so their system prompts differ in '
              'the working-directory line; they are approximate repeatability checks. '
              'One small fixed sample per cell gives descriptive results, not a general HumanEval score '
              'or a confidence interval. Official hidden tests run after generation in a networkless '
              'bubblewrap process with CPU, memory, and wall limits, and no feedback goes to the agent.', '',
              'Dataset: [OpenAI HumanEval](https://github.com/openai/human-eval/tree/6d43fb980f9fee3c892a914eda09951f772ad10d). '
              'The pinned twelve prompts and tests, selection, source hash, and MIT license are in '
              '`bench/pi_throughput/`. Raw pi events, generated solutions, provider observations, '
              'server counters, source audits, and restoration evidence accompany this report.', '']
    if (out/'incomplete-attempts').exists():
        lines += ['An initial C3 batch exceeded the 600-second per-task watchdog on HumanEval/132. '
                  'Its incomplete events and metrics are preserved under `incomplete-attempts/`; '
                  'it is excluded from scored rates. The entire C3 batch was retried with a longer '
                  'watchdog and identical pi request settings. Resume records document the bound. '
                  'Completed C1/C2 cells were retained. This retry introduces additional cache warming '
                  'and must be considered when interpreting the small sample.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--resume', action='store_true', help='Resume completed cells on the unchanged original deployment')
    args = ap.parse_args()
    if not __import__('re').fullmatch(r'[a-z0-9-]{1,48}', args.label):
        ap.error('Invalid experiment label')
    if os.environ.get('HERDR_ENV') != '1': raise RuntimeError('Run this comparison in a herdr pane')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    setup = out / 'deployment'
    setup.mkdir(exist_ok=args.resume)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    freeze_audit(out)
    current = inventory()
    if args.resume:
        initial = json.loads((out/'deployed-inventory.json').read_text())
        if any(any(current[h][k] != initial[h][k] for k in ('id','image','cmd','environment','mounts')) for h in HOSTS):
            raise RuntimeError('Resume requires the unchanged original deployment')
        if (out/'adaptive').exists():
            raise RuntimeError('Resume after adaptive deployment needs a new comparison label')
        for lane in ('deployed','deployed-warmup'):
            for cell in (out/lane).glob('c*'):
                if not (cell/'summary.json').exists():
                    failed = out/'incomplete-attempts'/f'{lane}-{cell.name}-{int(time.time())}'
                    failed.parent.mkdir(exist_ok=True)
                    cell.rename(failed)
    else:
        initial = current
        he.write_json(out / 'deployed-inventory.json', initial)
        exp.prepare(args.label, setup, costs=COSTS)
    guard = MemoryGuard(out/'memory.jsonl').start()
    heartbeat = exp.Heartbeat(args.label)
    mutated = False
    complete = False
    try:
        print('preflight headroom MiB', guard.preflight(seconds=2), flush=True)
        workspace = (json.loads((out/'workspace.json').read_text())['path'] if args.resume
                     else tempfile.mkdtemp(prefix='pi-he-compare-'))
        Path(workspace).mkdir(exist_ok=True)
        he.write_json(out/'workspace.json', {'path':workspace})
        with contextlib.nullcontext(workspace) as tmp:
            work, config = Path(tmp)/'work', Path(tmp)/'config'
            if config.exists(): shutil.rmtree(config)
            hashes = he.prepare_config(config)
            profile = {'config_hashes': hashes, 'pi_version': subprocess.check_output(['pi','--version'],text=True).strip(),
                 'levels': list(range(1,13)), 'task_count': 12, 'thinking':os.environ.get('PI_HE_THINKING','high'), 'seed_order':9000,
                 'max_tokens_override':os.environ.get('PI_HE_MAX_TOKENS'),
                 'task_timeout_s':float(os.environ.get('PI_HE_TASK_TIMEOUT','600')),
                 'source_revision': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 'herdr_pane': os.environ.get('HERDR_PANE_ID')}
            if args.resume:
                previous = json.loads((out/'config.json').read_text())
                if any(previous.get(k) != profile.get(k) for k in ('config_hashes','pi_version','thinking','max_tokens_override')):
                    raise RuntimeError('Cannot resume with different pi settings')
                he.write_json(out/f'resume-{int(time.time())}.json', profile)
            else: he.write_json(out/'config.json', profile)
            def baseline_check(): guard.check()
            asyncio.run(he.run_sweep('deployed-warmup', out, work, config, [1], baseline_check, limit=1))
            asyncio.run(he.run_sweep('deployed', out, work, config, range(1,13), baseline_check))
            report(out)
            asyncio.run(he.idle())
            guard.preflight(seconds=2)
            print('Switching to frozen adaptive V5; original containers retained for exact restoration', flush=True)
            mutated = True
            guard.loading = True
            exp.parallel(lambda host: exp.rpc(host, args.label, 'stop'))
            heartbeat.thread.start()
            for rank in (3,2,1,0):
                guard.check(); heartbeat.check()
                print(exp.rpc(HOSTS[rank], args.label, 'launch', rank), flush=True)
            heartbeat.must_run = True
            exp.wait_healthy(guard, heartbeat, seconds=2400)
            exp.parallel(lambda host: exp.rpc(host, args.label, 'settled'))
            guard.loading = False
            guard.preflight(seconds=2)
            he.write_json(out/'adaptive-inventory.json', inventory())
            def adaptive_check(): guard.check(); heartbeat.check()
            asyncio.run(he.run_sweep('adaptive-warmup', out, work, config, [1], adaptive_check, limit=1))
            asyncio.run(he.run_sweep('adaptive', out, work, config, range(1,13), adaptive_check))
            complete = True
            report(out)
    finally:
        heartbeat.stop.set()
        if heartbeat.thread.is_alive(): heartbeat.thread.join(timeout=30)
        guard.close()
        if mutated:
            print('Restoring original serving containers', flush=True)
            print(exp.parallel(lambda host: exp.rpc(host, args.label, 'restore')), flush=True)
            restored_guard = MemoryGuard(out/'restore-memory.jsonl', loading=True).start()
            try:
                restored_guard.preflight(seconds=2, minimum_mib=512)
                exp.wait_healthy(restored_guard, seconds=2400)
                exp.parallel(lambda host: exp.rpc(host, args.label, 'settled'))
                restored_guard.loading = False
                restored_guard.preflight(seconds=2)
                final = inventory()
                same = all(all(initial[h][k] == final[h][k] for k in ('id','image','cmd','environment','mounts'))
                           and final[h]['running'] and not final[h]['oom'] for h in HOSTS)
                he.write_json(out/'restoration.json', {'exact_originals_restored': same, 'nodes':final})
                if not same: raise RuntimeError('Restored runtime does not match original inventory')
                with tempfile.TemporaryDirectory(prefix='pi-he-restored-') as tmp:
                    config = Path(tmp)/'config'
                    he.prepare_config(config)
                    asyncio.run(he.run_sweep('restored' if complete else 'restored-probe', out,
                               Path(tmp)/'work', config, [1,12] if complete else [1], restored_guard.check,
                               limit=None if complete else 1))
                print('Original stack restored and verified with pi', flush=True)
            finally:
                restored_guard.close()
                exp.collect(args.label, setup)
        report(out)
    (out/'COMPLETE').write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n')
    shutil.rmtree(workspace)
    print('COMPLETE '+str(out/'REPORT.md'), flush=True)


if __name__ == '__main__': main()
