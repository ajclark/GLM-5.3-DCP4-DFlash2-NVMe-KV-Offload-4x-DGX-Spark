"""Guarded C2 throughput control, measured only while both streams decode."""
import argparse
import json
import math
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adaptive_spec import add_costs,generate,idle_check,request_body
from spec_memory import MemoryGuard


def overlap_rate(rows):
    streams=[]
    for row in rows:
        chunks=[(row['started_at']+c['seconds'],sum(len(v.get('token_ids') or []) for v in c['data'].get('choices',[])))
                for c in row['chunks'] if any(v.get('token_ids') for v in c['data'].get('choices',[]))]
        streams.append(chunks)
    start=max(c[0][0] for c in streams);end=min(c[-1][0] for c in streams)
    if end<=start:raise ValueError('No shared decode interval')
    tokens=sum(n for stream in streams for time,n in stream if start<time<=end)
    return {'start':start,'end':end,'tokens':tokens,'seconds':end-start,'tokens_per_second':tokens/(end-start)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--costs',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=3,choices=range(1,9))
    ap.add_argument('--base',default='http://spark-06c4.local:8000')
    args=ap.parse_args()
    costs=json.loads(args.costs.read_text())
    prompts=['Count from 1 to 180, one number per line. Output only the numbers, without a code fence.',
             'Count from 201 to 380, one number per line. Output only the numbers, without a code fence.']
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'config.json').write_text(json.dumps({'prompts':prompts,'costs':costs,'repeats':args.repeats},indent=2)+'\n')
    guard=MemoryGuard(args.out/'memory.jsonl').start();results=[]
    try:
        print('preflight:',guard.preflight(),flush=True)
        for repeat in range(args.repeats):
            for mode in (('fixed','adaptive') if repeat%2==0 else ('adaptive','fixed')):
                idle_check(args.base);guard.preflight(seconds=4)
                bodies=[]
                for i,prompt in enumerate(prompts):
                    body=request_body(prompt,7,f'{args.out.name}-r{repeat}-{mode}-{i}',512)
                    body['vllm_xargs']['spec_policy']=mode;add_costs(body,costs);bodies.append(body)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures=[pool.submit(generate,args.base,body,guard) for body in bodies]
                    rows=[future.result() for future in futures]
                record={'repeat':repeat,'policy':mode,'streams':rows,'overlap':overlap_rate(rows)}
                (args.out/f'r{repeat}-{mode}.json').write_text(json.dumps(record,indent=2)+'\n')
                for row,start in zip(rows,(1,201)):
                    numbers=[int(n) for n in re.findall(r'^\s*(\d+)[ \t]*$',row['text'],re.MULTILINE)]
                    if numbers!=list(range(start,start+180)):
                        raise RuntimeError('C2 count output is incomplete or incorrect')
                results.append(record)
                print(json.dumps({k:record[k] for k in ('repeat','policy','overlap')}),flush=True)
                guard.preflight(seconds=4)
        ratios=[next(r['overlap']['tokens_per_second'] for r in results if r['repeat']==i and r['policy']=='adaptive')/
                next(r['overlap']['tokens_per_second'] for r in results if r['repeat']==i and r['policy']=='fixed')
                for i in range(args.repeats)]
        summary={'paired_ratios':ratios,'paired_geomean_ratio':math.exp(statistics.mean(map(math.log,ratios))),
                 'note':'C2 overlap only; initial prefill and C1 tails excluded. Small repeated deterministic workload, not a general concurrency benchmark.'}
        (args.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary),flush=True)
    finally:guard.close()


if __name__=='__main__':main()
