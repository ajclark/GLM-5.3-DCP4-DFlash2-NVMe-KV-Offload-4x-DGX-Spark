#!/usr/bin/env python3
"""C1 coding benchmark, descending DFlash lengths, retaining exact serving containers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time

from rollout import ROOT, HOSTS, NAME, BASE, ssh, parallel, wait_health, restore, logs
sys.path.insert(0, str(ROOT/'bench'))
from adaptive_spec import generate, idle_check, spec_metrics
from spec_memory import MemoryGuard

CASES = {
    'merge_intervals': ('Write a Python function `merge_intervals(intervals)` that merges overlapping [start, end] intervals and returns them sorted. Do not mutate the input. Include a short docstring and three asserts as tests. Output only code.', '''
assert merge_intervals([]) == []
assert merge_intervals([[3,5],[1,3],[9,10]]) == [[1,5],[9,10]]
a=[[8,9],[1,2],[2,4],[3,7]]; saved=[x[:] for x in a]
assert merge_intervals(a)==[[1,7],[8,9]] and a==saved
assert merge_intervals([[1,8],[2,3],[8,10]])==[[1,10]]
'''),
    'lru_cache': ('Implement a Python class LRUCache(capacity) with get(key) returning the value or -1, and put(key, value). Both methods must be O(1). Reject capacity <= 0 with ValueError. Updating a key makes it most recently used. Include docstrings and a short usage example under if __name__ == "__main__". Output only Python code.', '''
for n in (0,-1):
    try: LRUCache(n)
    except ValueError: pass
    else: raise AssertionError('capacity')
c=LRUCache(2); c.put(1,10); c.put(2,20)
assert c.get(1)==10
c.put(3,30); assert c.get(2)==-1
c.put(1,11); c.put(4,40)
assert c.get(3)==-1 and c.get(1)==11 and c.get(4)==40
c=LRUCache(1); c.put('a',None); assert c.get('a') is None
c.put('b',False); assert c.get('a')==-1 and c.get('b') is False
'''),
    'topological_sort': ('Write a Python function topological_sort(graph) for a dictionary mapping string nodes to lists of outgoing neighbors. Include nodes appearing only as neighbors. Return a deterministic topological ordering by always selecting the lexicographically smallest currently available zero-indegree node. Raise ValueError on a cycle. Handle duplicate edges. Do not mutate graph. Use a heap, include type hints, a docstring, and three asserts. Output only Python code.', '''
assert topological_sort({})==[]
assert topological_sort({'b':['c'],'a':['c']})==['a','b','c']
g={'z':['a','a'],'b':['c'],'a':['c']}; saved={k:v[:] for k,v in g.items()}
assert topological_sort(g)==['b','z','a','c'] and g==saved
assert topological_sort({'a':[],'b':[]})==['a','b']
for g in ({'x':['x']},{'a':['b'],'b':['a']}):
    try: topological_sort(g)
    except ValueError: pass
    else: raise AssertionError('cycle')
'''),
}

def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def code_check(text, tests):
    fences=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S)
    source='\n'.join(fences) if fences else text
    def limits():
        resource.setrlimit(resource.RLIMIT_CPU,(3,3))
        resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
        resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))
    with tempfile.TemporaryDirectory(prefix='dflash-code-') as tmp:
        file=Path(tmp)/'check.py'
        (Path(tmp)/'generated.py').write_text(source)
        file.write_text("import runpy\nscope = runpy.run_path('generated.py', run_name='generated_module')\n"
                        + 'exec('+repr(tests)+', scope)\n')
        try:
            r=subprocess.run([sys.executable,'-I',str(file)],cwd=tmp,capture_output=True,text=True,
                             timeout=8,preexec_fn=limits,env={'PATH':'/usr/bin:/bin'})
            return {'ok':r.returncode==0,'returncode':r.returncode,'stderr':r.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            return {'ok':False,'error':'code test deadline exceeded'}


def measure(k, out, guard, repeats):
    rows=[]
    for rep in range(-1,repeats):
        for case,(prompt,tests) in CASES.items():
            idle_check(BASE)
            before=spec_metrics(BASE)
            body={'model':'glm-5.3','messages':[{'role':'user','content':prompt}],
                  'temperature':0,'top_p':1,'max_tokens':1200,'stream':True,
                  'stream_options':{'include_usage':True},'return_token_ids':True,
                  'chat_template_kwargs':{'enable_thinking':False}}
            result=generate(BASE,body,guard,deadline_seconds=300)
            after=spec_metrics(BASE)
            delta={key:value-before.get(key,0) for key,value in after.items()}
            def total(name):
                return sum(v for key,v in delta.items() if key.split('{')[0]==name)
            drafts=total('vllm:spec_decode_num_drafts_total')
            accepted=total('vllm:spec_decode_num_accepted_tokens_total')
            proposed=total('vllm:spec_decode_num_draft_tokens_total')
            assert drafts>0 and proposed>0, delta
            assert abs(proposed/drafts-k)<0.01, (k,proposed,drafts)
            check=code_check(result['text'],tests)
            row={'k':k,'case':case,'rep':rep,'warmup':rep<0,'decode_tok_s':result['decode_tps'],
                 'wall_tok_s':len(result['token_ids'])/result['seconds'],
                 'ttft_s':result['ttft'],'generated_tokens':len(result['token_ids']),
                 'wall_s':result['seconds'],'drafts':drafts,'proposed':proposed,'accepted':accepted,
                 'accepted_per_cycle':1+accepted/drafts,'acceptance_rate':accepted/proposed,
                 'cycle_ms_estimate':1000*(result['seconds']-result['ttft'])/drafts,
                 'token_sha256':result['token_sha256'],'code_check':check}
            save(out/f'{case}-rep{rep}.json',{'summary':row,'request':body,'result':result,'spec_metrics_delta':delta})
            rows.append(row);save(out/'rows.json',rows)
            print(json.dumps(row),flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('label');ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--ks',default='16,15,14,13,12,11,10,9,8',
                    help='Trial lengths in execution order; the current K=7 control is restored and measured last')
    args=ap.parse_args()
    if not re.fullmatch('[a-z0-9-]{1,48}',args.label):raise SystemExit('invalid label')
    trials=[int(k) for k in args.ks.split(',')]
    if not trials or len(set(trials))!=len(trials) or any(k<1 or k>16 or k==7 for k in trials):
        ap.error('--ks must contain distinct integers in 1..16, excluding the final K=7 control')
    if not 1<=args.repeats<=10:ap.error('--repeats must be 1..10')
    out=ROOT/'results/vllm029-upgrade'/args.label;out.mkdir(exist_ok=False)
    state=parallel(lambda h:json.loads(ssh(h,['docker','inspect',NAME]))[0])
    original_spec=json.loads(state[HOSTS[0]]['Config']['Cmd'][state[HOSTS[0]]['Config']['Cmd'].index('--speculative-config')+1])
    current=original_spec['num_speculative_tokens']
    assert current==7, original_spec
    order=trials+[current]
    idle_check(BASE)
    with os.fdopen(os.open(out/'original.private.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600),'w') as f:json.dump(state,f,indent=2)
    commands={}
    for k in trials:
        commands[k]={}
        for rank,host in enumerate(HOSTS):
            env={'DCP_IMAGE':state[host]['Image'],'DFLASH_K':str(k),'UPGRADE_LABEL':args.label,
                 'KVTIER_DIR':f'/var/tmp/kvcache-vllm029-c1-sweep/k{k}','DRYRUN':'1'}
            cmd=ssh(host,'env '+shlex.join(f'{a}={b}' for a,b in env.items())+' bash ~/glm-vllm029-build/launch.sh '+str(rank))
            parsed=shlex.split(cmd)
            actual=parsed[parsed.index('serve'):]
            expected=list(state[host]['Config']['Cmd']);idx=expected.index('--speculative-config')+1
            spec=json.loads(expected[idx]);spec['num_speculative_tokens']=k
            assert json.loads(actual[idx])==spec
            actual[idx]=expected[idx]
            assert actual==expected,(host,k,'launcher differs from running configuration')
            commands[k][host]=parsed
    save(out/'plan.json',{'order':order,'concurrency':1,'repetitions':args.repeats,
        'warmups_per_case':1,'max_tokens':1200,'thinking':False,'cases':{k:v[0] for k,v in CASES.items()},
        'original_spec':original_spec,'images':{h:s['Image'] for h,s in state.items()},
        'cache':'Dedicated sweep root per K; original cache retained; each prompt warmed before measurement'})
    guard=MemoryGuard(out/'memory.jsonl',loading=True).start()
    stopped=False;restored=False;results=[]
    def interrupt(signum,frame):raise RuntimeError(f'interrupted by {signum}')
    signal.signal(signal.SIGTERM,interrupt)
    def remove_trial(host):
        row=json.loads(ssh(host,['docker','inspect',NAME]))[0]
        assert row['Id']!=state[host]['Id'] and row['Config'].get('Labels',{}).get('spark.vllm.upgrade')==args.label
        ssh(host,['docker','rm','-f',NAME])
    try:
        guard.preflight(seconds=3)
        print(f'Starting C1 sweep in order {order}',flush=True)
        stopped=True
        parallel(lambda h:ssh(h,['docker','stop','-t','30',NAME]))
        parallel(lambda h:ssh(h,['docker','rename',NAME,NAME+'_pre_'+args.label]))
        for k in order:
            lane=out/f'k{k}';lane.mkdir()
            guard.loading=True
            if k==current:
                restore(out,state);restored=True
            else:
                parallel(lambda h:ssh(h,'$HOME/glm53big/start-flusher.sh'))
                for host in reversed(HOSTS):ssh(host,commands[k][host],timeout=120)
                wait_health(lane,guard)
            guard.loading=False
            parallel(lambda h:ssh(h,"pkill -f '[c]ache_flusher.sh' || true"))
            print(f'K={k} ready; warming coding tasks',flush=True)
            rows=measure(k,lane,guard,args.repeats)
            results.extend(rows);save(out/'all-rows.json',results)
            logs(lane,'measured')
            if k!=current:parallel(remove_trial)
            print(f'K={k} complete',flush=True)
        save(out/'status.json',{'ok':True,'restored_original_ids':{h:s['Id'] for h,s in state.items()}})
        print('Sweep complete; exact original K=7 containers are serving',flush=True)
    finally:
        guard.close()
        if stopped and not restored:
            try:logs(out,'failed')
            finally:restore(out,state)

if __name__=='__main__':main()
