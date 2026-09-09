"""Reload a fresh prompt cache after restart and continue its generated suffix.

The deployed offload_prompt_only=True excludes generated KV from disk stores.
The raw continuation therefore reuses prompt KV and recomputes the suffix.
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

from adaptive_spec import add_costs, generate, idle_check, request_body
from spec_memory import MemoryGuard

BOUNDARY = 102400  # Divisible by the deployed cache block alignments.
PREFIX_OUTPUT = 80


def cache_metrics(base):
    with urllib.request.urlopen(base+'/metrics',timeout=5) as response:
        lines=response.read().decode().splitlines()
    prefixes=('vllm:kv_offload_', 'vllm:prefix_cache_hits', 'vllm:external_prefix_cache_hits')
    return {line.rsplit(' ',1)[0]:float(line.rsplit(' ',1)[1]) for line in lines
            if line.startswith(prefixes) and '_created' not in line}


def metric_total(delta, name):
    return sum(value for key,value in delta.items()
               if key.split('{',1)[0] in (name, name+'_total'))


def validate(text):
    numbers=[int(n) for n in re.findall(r'^\s*(\d+)[ \t]*$',text,re.MULTILINE)]
    if 'CACHE_TEST_MARIGOLD' not in text or numbers!=list(range(1,101)):
        raise RuntimeError('Durable-context marker or complete count100 validation failed')


def prompt_ids(result):
    ids=next((row['data']['prompt_token_ids'] for row in result['chunks']
              if row['data'].get('prompt_token_ids')),None)
    if ids is None or len(ids)!=result['usage']['prompt_tokens']:
        raise RuntimeError('Missing exact original prompt token IDs')
    return ids


def reload_body(first, original, label):
    ids=prompt_ids(first)
    if not BOUNDARY-50<=len(ids)<=BOUNDARY-30 or len(first['token_ids'])<=PREFIX_OUTPUT:
        raise RuntimeError('Probe does not straddle a generated cache-block boundary')
    body={key:original[key] for key in ('model','temperature','seed','max_tokens','return_token_ids')}
    body.update(prompt=ids+first['token_ids'][:PREFIX_OUTPUT], add_special_tokens=False,
                stream=True, stream_options={'include_usage':True},
                vllm_xargs={**original['vllm_xargs'],'spec_label':label})
    return body


def complete_raw(base, body, guard, deadline=900):
    """Bounded completion stream; preserves integer prompt IDs across restart."""
    chunks=[]; errors=[]; responses=[]
    started=time.monotonic(); wall=time.time()
    def read():
        try:
            request=urllib.request.Request(base+'/v1/completions',json.dumps(body).encode(),
                                           {'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=min(deadline,600)) as response:
                responses.append(response)
                for line in response:
                    if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]':
                        chunks.append({'seconds':time.monotonic()-started,'data':json.loads(line[6:])})
        except urllib.error.HTTPError as exc:
            errors.append(f'HTTP {exc.code}: '+exc.read(4096).decode(errors='replace'))
        except Exception as exc:
            errors.append(str(exc))
    thread=threading.Thread(target=read,daemon=True); thread.start()
    try:
        while thread.is_alive():
            guard.check()
            if time.monotonic()-started>deadline:
                raise RuntimeError('Durable reload deadline exceeded')
            thread.join(.2)
    except BaseException:
        if responses:
            try: responses[0].fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except (AttributeError,OSError): pass
            responses[0].close()
        raise
    if errors: raise RuntimeError(errors[0])
    ids=[]; text=''; emitted=[]; usage=None
    for row in chunks:
        data=row['data']
        if 'error' in data: raise RuntimeError(str(data['error']))
        usage=data.get('usage') or usage
        for choice in data.get('choices',[]):
            new=choice.get('token_ids') or []; ids.extend(new)
            text+=choice.get('text') or ''
            if new: emitted.append((row['seconds'],len(new)))
    if not ids or not usage or usage['completion_tokens']!=len(ids):
        raise RuntimeError('Missing or inconsistent raw completion IDs/usage')
    elapsed=emitted[-1][0]-emitted[0][0]
    return {'started_at':wall,'seconds':time.monotonic()-started,'chunks':chunks,
            'token_ids':ids,'text':text,'usage':usage,'ttft':emitted[0][0],
            'decode_tps':(len(ids)-emitted[0][1])/elapsed if elapsed>0 else None}


def audit_reload(first, result, delta):
    external=metric_total(delta,'vllm:external_prefix_cache_hits')
    local=metric_total(delta,'vllm:prefix_cache_hits')
    loaded=metric_total(delta,'vllm:kv_offload_load_bytes')
    evidence={'identical_continuation_ids':result['token_ids']==first['token_ids'][PREFIX_OUTPUT:],
              'external_cached_tokens':external,'local_cached_tokens':local,
              'offload_load_bytes':loaded,'required_boundary':BOUNDARY,
              'original_prompt_tokens':len(prompt_ids(first))}
    # Retain the stronger evidence field without mistaking prompt loads for
    # generated-KV loads. The production prompt-only policy should make it false.
    evidence['generated_block_reload_passed']=(evidence['identical_continuation_ids']
        and loaded>0 and external>0 and external+local>=BOUNDARY
        and local<len(prompt_ids(first)))
    evidence['offload_prompt_only']=True
    evidence['prompt_cache_continuation_passed']=(evidence['identical_continuation_ids']
        and loaded>0 and external>0 and BOUNDARY-4096<=external+local<=len(prompt_ids(first))
        and local<len(prompt_ids(first)))
    return evidence


def fresh_body(base, costs, label, guard):
    rng=random.Random(202609091703)
    words='river window garden stone paper summer chair path station orange cloud table copper bridge'.split()
    data=[rng.choice(words) for _ in range(103000)]; length=102000
    for _ in range(8):
        guard.check()
        prompt=('Unique durability experiment 20260909-v4-once-store. The validation code is CACHE_TEST_MARIGOLD.\n'
                'The following words are arbitrary test data:\n'+' '.join(data[:length])
                +'\nEnd of test data. Print the validation code on its own line. Then count from 1 to 100, one number per line. Do not add anything else.')
        body=request_body(prompt,7,label,256); body['vllm_xargs']['spec_policy']='adaptive'
        add_costs(body,costs)
        payload={key:body[key] for key in ('model','messages','chat_template_kwargs')}
        req=urllib.request.Request(base+'/tokenize',json.dumps(payload).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as response: count=json.load(response)['count']
        if BOUNDARY-50<=count<=BOUNDARY-30: return body,count
        length+=BOUNDARY-40-count
        if not 100000<=length<=103000: break
    raise RuntimeError('Unable to align bounded durability prompt before cache boundary')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase',choices=('first','reload'))
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--record',type=Path,required=True)
    ap.add_argument('--costs',type=Path)
    ap.add_argument('--base',default='http://spark-06c4.local:8000')
    args=ap.parse_args()
    if args.phase=='first' and (args.record.exists() or not args.costs):
        ap.error('first use requires costs and a new record path')
    args.out.mkdir(parents=True,exist_ok=False)
    guard=MemoryGuard(args.out/'memory.jsonl').start()
    try:
        idle_check(args.base); print('preflight:',guard.preflight(),flush=True)
        if args.phase=='first':
            body,count=fresh_body(args.base,json.loads(args.costs.read_text()),args.out.name,guard)
            record={'body':body,'prompt_tokens':count,'first_output':str(args.out/'result.json'),
                    'prompt_sha256':hashlib.sha256(body['messages'][0]['content'].encode()).hexdigest(),
                    'costs_sha256':hashlib.sha256(args.costs.read_bytes()).hexdigest()}
            args.record.write_text(json.dumps(record,indent=2)+'\n')
        else:
            record=json.loads(args.record.read_text()); original=record['body']
            if hashlib.sha256(original['messages'][0]['content'].encode()).hexdigest()!=record['prompt_sha256']:
                raise RuntimeError('Durable prompt changed')
            first=json.loads(Path(record['first_output']).read_text()); validate(first['text'])
            body=reload_body(first,original,args.out.name)
        before=cache_metrics(args.base)
        print('starting',args.phase,'original prompt tokens',record['prompt_tokens'],flush=True)
        result=(generate(args.base,body,guard,deadline_seconds=900) if args.phase=='first'
                else complete_raw(args.base,body,guard))
        path=args.out/'result.json'; path.write_text(json.dumps(result,indent=2)+'\n')
        if args.phase=='first':
            validate(result['text']); reload_body(result,body,args.out.name)  # Validate restart inputs now.
            result['count100_passed']=True
        guard.preflight(seconds=12)
        after=cache_metrics(args.base)
        delta={key:value-before.get(key,0) for key,value in after.items()}
        result['cache_metric_delta']=delta
        if args.phase=='reload': result.update(audit_reload(first,result,delta))
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({key:value for key,value in result.items()
                          if key not in ('chunks','text','token_ids')}),flush=True)
        if args.phase=='reload' and not result['prompt_cache_continuation_passed']:
            raise RuntimeError('Reload lacks identical continuation or expected prompt-only disk-cache evidence')
    finally:
        guard.close()


if __name__=='__main__': main()
