#!/usr/bin/env python3
"""Deploy the staged release with a memory guard and exact-container rollback."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'bench'))
from spec_memory import HOSTS, MemoryGuard

NAME='vllm_glm53big'
IMAGE=os.environ.get('DCP_IMAGE','spark-vllm:0.29.0-dcp1')
LAUNCH_KEYS = ('DCP_IMAGE', 'DCP_SIZE', 'MAXLEN', 'MAXBATCHED', 'MAXSEQS', 'KVBYTES',
               'KVTIER', 'KVTIER_DIR', 'KVTIER_MODE', 'KVTIER_THREADS', 'KVTIER_BOUNCE',
               'KVTIER_DISK_BYTES', 'KVTIER_HASHSEED', 'DFLASH_K', 'NCCL_HOTPLUG',
               'HOTPLUG_SO', 'HOTPLUG_PORT', 'NCCL_IB_QPS_PER_CONNECTION', 'PROFILER_DIR',
               'GLM_SPEC_POLICY', 'GLM_SPEC_LOSSY', 'GLM_DCP_LSE_FOLD',
               'DCP_LSE_FOLD', 'DCP_RS_HEADMAJOR')
LAUNCH_ENV = shlex.join(f'{key}={os.environ[key]}' for key in LAUNCH_KEYS if key in os.environ)
BASE='http://spark-06c4.local:8000'


def ssh(host,args,timeout=90):
    cmd=args if isinstance(args,str) else shlex.join(str(x) for x in args)
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',host,cmd],
                          capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError(f'{host}: {result.stdout[-1500:]} {result.stderr[-1500:]}')
    return result.stdout


def parallel(fn):
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={h:pool.submit(fn,h) for h in HOSTS}
        results={};errors=[]
        for h,f in futures.items():
            try:results[h]=f.result()
            except Exception as exc:errors.append(str(exc))
        if errors:raise RuntimeError('\n'.join(errors))
        return results


def request(body=None,path='/health',timeout=10):
    req=urllib.request.Request(BASE+path,data=json.dumps(body).encode() if body else None,
                               headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        payload=response.read()
        return json.loads(payload) if payload else {}


def smoke():
    result=request({'model':'glm-5.3','messages':[{'role':'user','content':
        'Count from 1 to 100, one number per line. Output only the numbers.'}],
        'max_tokens':400,'temperature':0,'chat_template_kwargs':{'enable_thinking':False},
        'return_token_ids':True},'/v1/chat/completions',300)
    content=result['choices'][0]['message']['content']
    if [int(x) for x in re.findall(r'\d+',content)]!=list(range(1,101)):
        raise RuntimeError('count100 regression: missing, repeated or wrong numbers')
    return result


def logs(out,prefix):
    def capture(host):
        r=subprocess.run(['ssh',host,f'docker logs --tail 3000 {NAME} 2>&1'],capture_output=True,text=True,timeout=40)
        (out/f'{prefix}-{host}.log').write_text(r.stdout+r.stderr)
    parallel(capture)


def wait_health(out,guard,timeout=2400):
    start=time.monotonic();next_log=0
    while time.monotonic()-start<timeout:
        if guard:guard.check()
        status=parallel(lambda h:json.loads(ssh(h,['docker','inspect',NAME]))[0]['State'])
        if any(not s['Running'] for s in status.values()):
            raise RuntimeError('a vLLM rank exited during startup')
        try:
            request();return
        except Exception:pass
        if time.monotonic()>next_log:
            print(f'Waiting for engine: {int(time.monotonic()-start)}s',flush=True);next_log=time.monotonic()+30
        time.sleep(5)
    raise RuntimeError('engine health deadline expired')


def restore(out,state):
    print('Restoring exact original containers',flush=True)
    def stop_new(host):
        try:current=json.loads(ssh(host,['docker','inspect',NAME]))[0]
        except RuntimeError:return
        if current['Id']==state[host]['Id']:return
        if current['Config'].get('Labels',{}).get('spark.vllm.upgrade')!=out.name:
            raise RuntimeError('refusing to stop unrelated container on '+host)
        ssh(host,['docker','stop','-t','5',NAME])
        ssh(host,['docker','rename',NAME,NAME+'_failed_'+out.name])
    parallel(stop_new)
    for host in reversed(HOSTS):
        old=json.loads(ssh(host,['docker','inspect',state[host]['Id']]))[0]
        if old['Name']!='/'+NAME:ssh(host,['docker','rename',old['Id'],NAME])
        ssh(host,'$HOME/glm53big/start-flusher.sh')
        if not old['State']['Running']:ssh(host,['docker','start',old['Id']])
    wait_health(out,None)
    (out/'restored-smoke.json').write_text(json.dumps(smoke(),indent=2)+'\n')
    parallel(lambda h:ssh(h,"pkill -f '[c]ache_flusher.sh' || true"))
    print('Original service restored and generation verified',flush=True)


def guarded_process(cmd,outfile,guard,timeout):
    with outfile.open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        start=time.monotonic();next_log=0
        try:
            while proc.poll() is None:
                guard.check()
                if time.monotonic()-start>timeout:raise RuntimeError('regression deadline expired')
                if time.monotonic()>next_log:
                    print(f'Regression running: {int(time.monotonic()-start)}s, {outfile.name}',flush=True)
                    next_log=time.monotonic()+30
                time.sleep(2)
            if proc.returncode:raise RuntimeError(f'regression failed: {outfile}')
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=10)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('label');ap.add_argument('--restore',action='store_true')
    ap.add_argument('--skip-cuda',action='store_true')
    ap.add_argument('--resume-from', help='Reuse a recorded baseline after a failed restoration; current container IDs must match its originals')
    args=ap.parse_args()
    if not re.fullmatch('[a-z0-9-]{1,48}',args.label):raise SystemExit('invalid label')
    out=ROOT/'results/vllm029-upgrade'/args.label;out.mkdir(parents=True,exist_ok=True)
    state_file=out/'original.private.json'
    if args.restore:
        restore(out,json.loads(state_file.read_text()));return
    if state_file.exists():raise SystemExit('label already used; choose a fresh label or --restore')
    current=parallel(lambda h:json.loads(ssh(h,['docker','inspect',NAME]))[0])
    resume_source=None
    if args.resume_from:
        if not re.fullmatch('[a-z0-9-]{1,48}',args.resume_from) or args.resume_from==args.label:
            raise SystemExit('invalid baseline label')
        resume_source=out.parent/args.resume_from
        state=json.loads((resume_source/'original.private.json').read_text())
        if any(current[h]['Id']!=state[h]['Id'] for h in HOSTS):
            raise RuntimeError('resume requires the exact original containers on all nodes')
        if not (resume_source/'baseline-bench/bench.json').exists():
            raise RuntimeError('recorded baseline benchmark is required for resume')
        print(f'Resuming from retained original containers; baseline source: {args.resume_from}',flush=True)
    else:
        request()
        state=current
    manifest=json.loads((Path(__file__).parent/'manifest.json').read_text())
    def preflight(host):
        if not state[host]['State']['Running']:raise RuntimeError('original rank not running')
        image=json.loads(ssh(host,['docker','image','inspect',IMAGE]))[0]
        remote=json.loads(ssh(host, 'docker run --rm --entrypoint python3 '
            '-v "$HOME/glm-vllm029-build:/regression:ro" '+IMAGE+' /regression/verify_image.py'))
        if remote!=manifest:raise RuntimeError('staging manifest differs on '+host)
        ssh(host,'env '+LAUNCH_ENV+' DRYRUN=1 bash ~/glm-vllm029-build/launch.sh '+str(HOSTS.index(host)))
        return image['Id']
    images=parallel(preflight)
    print('Verifying hybrid cache allocation before downtime',flush=True)
    config_result=ssh(HOSTS[3], 'docker run --rm --memory=2g --entrypoint python3 '
        '-v "$HOME/glm-vllm029-build:/regression:ro" '+IMAGE+' /regression/config_regression.py',timeout=180)
    (out/'config-regression.log').write_text(config_result)
    with os.fdopen(os.open(state_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w') as saved:
        json.dump(state,saved,indent=2)
    (out/'images.json').write_text(json.dumps(images,indent=2)+'\n')
    if resume_source:
        shutil.copy2(resume_source/'baseline-count100.json',out/'baseline-count100.json')
        shutil.copytree(resume_source/'baseline-bench',out/'baseline-bench',dirs_exist_ok=True)
        (out/'resume.json').write_text(json.dumps({'baseline_source':args.resume_from,
            'original_ids_verified':{h:current[h]['Id'] for h in HOSTS}},indent=2)+'\n')
    else:
        (out/'baseline-count100.json').write_text(json.dumps(smoke(),indent=2)+'\n')
    guard=MemoryGuard(out/'memory.jsonl',loading=True).start()
    stopped=False;success=False
    def interrupt(signum,frame):raise RuntimeError(f'interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupt)
    try:
        guard.preflight(seconds=3)
        if not resume_source:
            baseline_bench = out/'baseline-bench'
            guarded_process([sys.executable, str(Path(__file__).with_name('service_checks.py')),
                             'bench', '--out', str(baseline_bench)],
                            out/'baseline-bench.log', guard, 1200)
        print('Stopping current service; exact containers and mounts are retained',flush=True)
        stopped=True
        parallel(lambda h:ssh(h,['docker','stop','-t','10',NAME]))
        parallel(lambda h:ssh(h,['docker','rename',NAME,NAME+'_pre_'+args.label]))
        if not args.skip_cuda:
            print('Running CUDA kernel and NVMe transfer regressions before model loading',flush=True)
            guarded_process(['ssh',HOSTS[3],
                'docker run --rm --name vllm029-regression --gpus all --ipc host '
                '-v /var/tmp/models/GLM-5.3-Int4-Int8Mix:/models/glm-5.3:ro '
                '-v /var/tmp/models/GLM-5.3-DFlash2-draft:/models/dflash2-draft:ro '
                '-v "$HOME/glm-vllm029-build:/regression:ro" '
                '-e HF_HUB_OFFLINE=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 '
                '--entrypoint python3 '+IMAGE+' /regression/model_config_regression.py'],
                out/'model-config-regression.log',guard,300)
            guarded_process(['ssh',HOSTS[3],
                'docker run --rm --name vllm029-regression --gpus all --ipc host '
                '-v "$HOME/glm-vllm029-build:/regression:ro" '
                '--entrypoint python3 '+IMAGE+' /regression/regression.py --cuda'],
                out/'cuda-regression.log',guard,1200)
            guarded_process(['ssh',HOSTS[3],
                'docker run --rm --name vllm029-regression --gpus all --ipc host '
                '-v "$HOME/glm-vllm029-build:/regression:ro" '
                '--entrypoint python3 '+IMAGE+' /regression/indexer_regression.py'],
                out/'indexer-regression.log',guard,1200)
        parallel(lambda h:ssh(h,'$HOME/glm53big/start-flusher.sh'))
        for rank in (3,2,1,0):
            print(f'Launching vLLM 0.29 rank {rank}',flush=True)
            result=ssh(HOSTS[rank],f'env {LAUNCH_ENV} UPGRADE_LABEL={args.label} bash ~/glm-vllm029-build/launch.sh {rank}',timeout=120)
            (out/f'launch-{rank}.log').write_text(result)
            time.sleep(2)
        wait_health(out,guard)
        guard.loading=False
        (out/'count100.json').write_text(json.dumps(smoke(),indent=2)+'\n')
        parallel(lambda h:ssh(h,"pkill -f '[c]ache_flusher.sh' || true"))
        guarded_process([sys.executable,str(ROOT/'bench/repro/probe_indexer_mixed_decode.py'),
                         '--out',str(out/'mixed')],out/'mixed.log',guard,600)
        logs(out,'running')
        success=True
        (out/'status.json').write_text(json.dumps({'ok':True,'version':'0.29.0','images':images},indent=2)+'\n')
        print('vLLM 0.29 is serving; count100 and concurrent mixed decoding passed',flush=True)
    finally:
        guard.close()
        if stopped and not success:
            try:
                logs(out,'failed')
            except Exception as exc:
                print(f'Log capture failed: {exc}', flush=True)
            try:
                ssh(HOSTS[3],'docker rm -f vllm029-regression >/dev/null 2>&1 || true')
            finally:
                restore(out,state)

if __name__=='__main__':main()
