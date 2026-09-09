#!/usr/bin/env python3
"""Prepare, run, and roll back one guarded fixed-cap Spark experiment.

Every run restores the original containers, including on test failure. Remote
watchdogs stop only labelled experimental containers if memory pressure rises
or the controller heartbeat disappears. Preparations do not stop serving.
"""
import argparse
import hashlib
import io
import json
import re
import shlex
import subprocess
import tarfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adaptive_spec import PROMPTS, generate, idle_check, request_body
from spec_memory import HOSTS, MemoryGuard

ROOT = Path(__file__).resolve().parents[1]
BASE = 'http://spark-06c4.local:8000'


def ssh(host, command, data=None, timeout=90):
    result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,command],
                            input=data, capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(host+': '+result.stderr.decode()[-2000:]+result.stdout.decode()[-2000:])
    return result.stdout.decode()


def parallel(fn):
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {host:pool.submit(fn,host) for host in HOSTS}
        results, errors = {}, []
        for host,future in futures.items():
            try:
                results[host] = future.result()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('\n'.join(errors))
        return results


def rpc(host, label, action, rank=None):
    args = ['python3',f'glm-spec/{label}/spec_node.py',action,label]
    if rank is not None:
        args += ['--rank',str(rank)]
    return ssh(host, shlex.join(args))


def package():
    files = {
        'spec_node.py':ROOT/'bench/spec_node.py', 'spec_memory.py':ROOT/'bench/spec_memory.py',
        'launch.sh':ROOT/'launch-glm53big-dcp.sh',
    }
    for name in ('scheduler.py','adaptive.py','model_runner.py','cudagraph_utils.py','v2_block_table.py',
                 'v2_async_utils.py','v1_outputs.py','confidence_trace.py'):
        files['changes/'+name] = ROOT/'stage/glm-dcp'/name
    full_manifest = json.loads((ROOT/'results/adaptive-spec/inventory/python-sha256.json').read_text())
    relevant = ['v1/core/sched/scheduler.py','v1/core/sched/async_scheduler.py',
                'config/vllm.py','config/speculative.py',
                'v1/worker/gpu/model_runner.py','v1/worker/gpu/cudagraph_utils.py',
                'v1/worker/gpu/input_batch.py']
    relevant += [p for p in full_manifest if p.startswith('v1/worker/gpu/spec_decode/')]
    manifest = {p:full_manifest[p] for p in relevant}
    # Added after the original inventory; keep that historical manifest frozen.
    for rel in ('v1/worker/gpu/block_table.py','v1/worker/gpu/async_utils.py','v1/outputs.py'):
        manifest[rel] = hashlib.sha256((ROOT/'baseline/vllm'/rel).read_bytes()).hexdigest()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for name,path in files.items():
            tar.add(path,arcname=name)
        data = json.dumps(manifest).encode()
        info = tarfile.TarInfo('expected-runtime.json')
        info.size = len(data)
        tar.addfile(info,io.BytesIO(data))
    return buf.getvalue()


def prepare(label, out, reuse_cache_from=None, trace_off=False, costs=None, hint_priors=None, confidence_trace=False):
    data = package()
    def node(host):
        dest = 'glm-spec/'+label
        # Refuse reuse, preserving all prior rollback material.
        ssh(host, f'mkdir -p glm-spec && mkdir {shlex.quote(dest)} && tar -xf - -C {shlex.quote(dest)}', data)
        return json.loads(rpc(host,label,'prepare'))
    result = parallel(node)
    (out/'prepared.json').write_text(json.dumps(result,indent=2)+'\n')
    for source, name in ((costs, 'boot-costs.json'), (hint_priors, 'hint-priors.json')):
        if source:
            raw = source.read_bytes()
            json.loads(raw)
            if len(raw) > 16384:
                raise ValueError('boot calibration exceeds 16 KiB')
            parallel(lambda host:ssh(host, shlex.join(['tee', f'glm-spec/{label}/kvcache/{name}']), raw))
            (out/name).write_bytes(raw)
    if trace_off:
        parallel(lambda host:ssh(host,shlex.join(['touch',f'glm-spec/{label}/trace-disabled'])))
        (out/'trace-disabled').touch()
    if confidence_trace:
        parallel(lambda host:ssh(host,shlex.join(['touch',f'glm-spec/{label}/confidence-trace-enabled'])))
        (out/'confidence-trace-enabled').touch()
    if reuse_cache_from:
        reused = parallel(lambda host:ssh(host,shlex.join(['python3',f'glm-spec/{label}/spec_node.py',
            'reuse-cache',label,'--source-label',reuse_cache_from])))
        (out/'cache-reuse.json').write_text(json.dumps({'source_label':reuse_cache_from,'nodes':reused},indent=2)+'\n')
    print('prepared and verified running source on all four ranks',flush=True)


class Heartbeat:
    def __init__(self,label):
        self.label, self.error = label, None
        self.must_run = False
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run,daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                states = parallel(lambda h:json.loads(rpc(h,self.label,'status')))
                for host,state in states.items():
                    if state['tripped'] or state['oom'] or (self.must_run and not state['running']):
                        self.error = host+': '+str(state)
            except Exception as exc:
                self.error = str(exc)
            self.stop.wait(10)

    def check(self):
        if self.error:
            raise RuntimeError(self.error)


def healthy():
    try:
        with urllib.request.urlopen(BASE+'/health',timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def validate_count_smoke(text):
    numbers = [int(n) for n in re.findall(r'^\s*(\d+)[ \t]*$',text,re.MULTILINE)]
    if len(numbers) < 10 or numbers != list(range(1,len(numbers)+1)):
        raise RuntimeError('verification smoke output failed sequential-count check: '+repr(text[:200]))


def wait_healthy(guard, heartbeat=None, seconds=1800):
    start = time.monotonic()
    while time.monotonic()-start < seconds:
        guard.check()
        if heartbeat:
            heartbeat.check()
        if healthy():
            print('healthy after',round(time.monotonic()-start),'seconds',flush=True)
            return
        if int(time.monotonic()-start) % 30 < 2:
            print('boot headroom:',{h:round(r['available_mib']) for h,r in guard.latest.items()},flush=True)
        time.sleep(2)
    raise RuntimeError('health deadline exceeded')


def collect(label, out):
    def node(host):
        # Snapshot the currently running experiment only when its label matches.
        code = '''import json,subprocess,sys
from pathlib import Path
label=sys.argv[1]
r=subprocess.run(['docker','inspect','vllm_glm53big'],capture_output=True,text=True)
if r.returncode==0 and (json.loads(r.stdout)[0]['Config'].get('Labels') or {}).get('glm.spec.experiment')==label:
    with (Path.home()/'glm-spec'/label/'live.log').open('w') as out:
        subprocess.run(['docker','logs','--tail','10000','vllm_glm53big'],stdout=out,stderr=out,timeout=30)
'''
        ssh(host,shlex.join(['python3','-c',code,label]))
        # Only diagnostic artifacts, never original.json (which contains env).
        cmd = f'cd glm-spec/{label} && tar -cf - --ignore-failed-read launch.log watchdog.log tripped experiment.log live.log kvcache/spec-trace.jsonl'
        result = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,cmd],capture_output=True,timeout=60)
        with tarfile.open(fileobj=io.BytesIO(result.stdout)) as tar:
            tar.extractall(out/host,filter='data')
        return 'collected'
    parallel(node)


def run(label, out, hold=False, hold_minutes=30, screen_repeats=2):
    guard = MemoryGuard(out/'memory.jsonl').start()
    heartbeat = Heartbeat(label)
    mutated = False
    try:
        print('preflight:',guard.preflight(),flush=True)
        idle_check(BASE)
        if not healthy():
            raise RuntimeError('baseline endpoint unhealthy')
        mutated = True
        guard.loading = True
        parallel(lambda h:rpc(h,label,'stop'))
        heartbeat.thread.start()
        # Worker-first; each original container is stopped before any allocation.
        for rank in (3,2,1,0):
            guard.check()
            heartbeat.check()
            print(rpc(HOSTS[rank],label,'launch',rank),flush=True)
        heartbeat.must_run = True
        wait_healthy(guard,heartbeat)
        parallel(lambda h:rpc(h,label,'settled'))
        guard.loading = False
        print('post-boot:',guard.preflight(),flush=True)
        # Warm small sizes with a bounded decode before longer measurements.
        for cap in (7,5,3,1):
            heartbeat.check()
            guard.preflight(seconds=4)
            result = generate(BASE,request_body('Count from 1 to 30, one number per line.',cap,
                                               f'{label}-smoke-k{cap}',64),guard)
            (out/f'smoke-k{cap}.json').write_text(json.dumps(result,indent=2)+'\n')
            validate_count_smoke(result['text'])
            print('smoke cap',cap,'tokens',len(result['token_ids']),'tps',result['decode_tps'],flush=True)
        # Interleave cap order across repeats to reduce drift and warm-up bias.
        for repeat,order in enumerate(((7,3,5,1),(1,5,3,7))[:screen_repeats]):
            for case,prompt in PROMPTS.items():
                for cap in order:
                    heartbeat.check()
                    idle_check(BASE)
                    guard.preflight(seconds=4)
                    run_label = f'{label}-{case}-r{repeat}-k{cap}'
                    result = generate(BASE,request_body(prompt,cap,run_label,256),guard)
                    result.update(label=run_label,case=case,repeat=repeat,cap=cap)
                    (out/(run_label+'.json')).write_text(json.dumps(result,indent=2)+'\n')
                    print(json.dumps({k:result[k] for k in ('label','decode_tps','ttft')}),flush=True)
        if hold:
            print('HOLD: experiment remains guarded for additional C1 tests; touch',str(out/'finish'),
                  f'to restore (automatic deadline {hold_minutes} minutes)',flush=True)
            deadline = time.monotonic()+60*hold_minutes
            while not (out/'finish').exists() and time.monotonic()<deadline:
                guard.check()
                heartbeat.check()
                time.sleep(2)
        print('experiment completed; restoring original serving containers',flush=True)
    finally:
        heartbeat.stop.set()
        if heartbeat.thread.is_alive():
            heartbeat.thread.join(timeout=30)
        guard.close()
        if mutated:
            # Node watchdogs only touch experiment-labelled containers. Restore
            # restarts the original image/config/mounts, on every participating node.
            print(parallel(lambda h:rpc(h,label,'restore')),flush=True)
            restoration = MemoryGuard(out/'restore-memory.jsonl',loading=True).start()
            try:
                restoration.preflight(seconds=2,minimum_mib=512)
                wait_healthy(restoration,seconds=2400)
                parallel(lambda h:rpc(h,label,'settled'))
                restoration.loading = False
                check = generate(BASE,request_body('Reply with OK.',7,'restore-check',16),restoration)
                if 'OK' not in check['text'].upper():
                    raise RuntimeError('restored server failed generation probe')
                (out/'restored.json').write_text(json.dumps(check,indent=2)+'\n')
                print('original DCP2 stack restored and generating',flush=True)
            finally:
                restoration.close()
                collect(label,out)


def verify_restore(label, out):
    guard = MemoryGuard(out/'restore-verified-memory.jsonl',loading=True).start()
    try:
        guard.preflight(seconds=2,minimum_mib=512)
        wait_healthy(guard,seconds=2400)
        parallel(lambda h:rpc(h,label,'settled'))
        guard.loading = False
        print('settled original headroom:',guard.preflight(seconds=10),flush=True)
        result = generate(BASE,request_body('Reply with OK.',7,'restore-check',16),guard)
        if 'OK' not in result['text'].upper():
            raise RuntimeError('original generation probe failed')
        (out/'restored.json').write_text(json.dumps(result,indent=2)+'\n')
        print('original DCP2 stack verified healthy and generating',flush=True)
    finally:
        guard.close()
        collect(label,out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['prepare','run','restore','verify-restore','collect'])
    ap.add_argument('label')
    ap.add_argument('--hold',action='store_true',help='Keep the guarded experiment available for up to 30 minutes after fixed-cap screening')
    ap.add_argument('--hold-minutes',type=int,default=30,help='Bounded additional test window (1..120 minutes); watchdogs remain active')
    ap.add_argument('--screen-repeats',type=int,choices=(0,1,2),default=2,help='Fixed-cap calibration repeats; zero is for durability-only checks after a validated deployment')
    ap.add_argument('--reuse-cache-from',help='Prepare a durability check using a stopped experiment with identical source, kernels, image, and lane')
    ap.add_argument('--trace-off',action='store_true',help='Prepare the identical policy with telemetry disabled for overhead controls')
    ap.add_argument('--costs',type=Path,help='Prepare server-owned cycle costs; enables adaptive policy for ordinary clients')
    ap.add_argument('--hint-priors',type=Path,help='Prepare server-owned weak workload priors; requires --costs')
    ap.add_argument('--confidence-trace',action='store_true',help='Prepare bounded previous-proposal confidence diagnostics; policy unchanged')
    args = ap.parse_args()
    if not re.fullmatch(r'[a-z0-9-]{1,48}',args.label):
        ap.error('invalid label')
    if not 1 <= args.hold_minutes <= 120:
        ap.error('hold minutes must be 1..120')
    if args.reuse_cache_from and (args.action!='prepare' or not re.fullmatch(r'[a-z0-9-]{1,48}',args.reuse_cache_from)):
        ap.error('cache reuse requires prepare and a valid source experiment label')
    if args.trace_off and args.action!='prepare':
        ap.error('trace-off is a prepare option')
    if args.confidence_trace and (args.action!='prepare' or args.trace_off or args.reuse_cache_from):
        ap.error('confidence collection requires a fresh prepare with telemetry enabled')
    if (args.costs or args.hint_priors) and args.action!='prepare':
        ap.error('boot calibration is a prepare option')
    if args.hint_priors and not args.costs:
        ap.error('hint priors require measured boot costs')
    if args.reuse_cache_from and (args.costs or args.hint_priors):
        ap.error('a durability-only boot must preserve its previous calibration files')
    out = ROOT/'results/adaptive-spec'/args.label
    if args.action == 'prepare':
        out.mkdir(exist_ok=False)
        prepare(args.label,out,args.reuse_cache_from,args.trace_off,args.costs,args.hint_priors,args.confidence_trace)
    elif args.action == 'run':
        if not (out/'prepared.json').exists():
            ap.error('prepare this label first')
        run(args.label,out,args.hold,args.hold_minutes,args.screen_repeats)
    elif args.action == 'verify-restore':
        verify_restore(args.label,out)
    elif args.action == 'collect':
        collect(args.label,out)
    else:
        print(parallel(lambda h:rpc(h,args.label,'restore')))


if __name__ == '__main__':
    main()
