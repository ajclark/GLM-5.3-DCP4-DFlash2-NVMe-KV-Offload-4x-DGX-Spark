"""Compare identical fixed-seven requests with tracing enabled and disabled."""
import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def validate_count(case,text):
    lines=[line.strip() for line in text.strip().splitlines()]
    if case=='count_triples':
        expected=[' '.join([str(i)]*3) for i in range(1,101)]
    else:
        values=range(1,181) if case=='count_numbers' else range(2,361,2)
        if case not in ('count_numbers','count_evens'):raise ValueError('Unknown overhead control')
        expected=list(map(str,values))
    if lines!=expected:raise ValueError('Counting control is not complete and exact')


def interval(row):
    chunks=[chunk for chunk in row['chunks']
            if any(choice.get('token_ids') for choice in chunk['data'].get('choices',[]))]
    return row['started_at']+chunks[0]['seconds'],row['started_at']+chunks[-1]['seconds']


def compare(on,off):
    def index(rows):
        result={}
        for row in rows:
            validate_count(row['case'],row['text'])
            key=(row['case'],row['repeat'])
            if key in result:raise ValueError('Duplicate overhead pair')
            if row['cap']!=7 or row['policy']!='fixed':raise ValueError('Control must use fixed7')
            result[key]=row
        return result
    a,b=index(on),index(off)
    if not a or a.keys()!=b.keys():raise ValueError('Missing overhead pairs')
    groups=defaultdict(list);pairs=[]
    for key in sorted(a):
        x,y=a[key],b[key]
        if x['token_ids']!=y['token_ids']:raise ValueError('Output tokens differ between trace controls')
        on_start,on_end=interval(x);off_start,off_end=interval(y)
        if min(on_end-on_start,off_end-off_start)<=0:raise ValueError('Invalid decode interval')
        ratio=(on_end-on_start)/(off_end-off_start)
        groups[key[0]].append(math.log(ratio))
        pairs.append({'case':key[0],'repeat':key[1],'trace_on_decode_seconds':on_end-on_start,
                      'trace_off_decode_seconds':off_end-off_start,'on_over_off_time_ratio':ratio})
    means=[statistics.mean(v) for v in groups.values()]
    rng=random.Random(20260909)
    boot=sorted(math.exp(statistics.mean(rng.choices(means,k=len(means)))) for _ in range(10000)) if len(means)>1 else None
    return {'pairs':pairs,'paired_geomean_time_ratio':math.exp(statistics.mean(means)),
            'prompt_bootstrap_95':[boot[250],boot[9749]] if boot else None,
            'per_prompt_time_ratios':{k:math.exp(statistics.mean(v)) for k,v in groups.items()},
            'identical_output_tokens':True}


def clock_summary(rows,samples):
    summaries={}
    windows=[interval(row) for row in rows]
    for host,values in samples.items():
        values=sorted(values,key=lambda row:row['received_at'])
        covered=True
        for start,end in windows:
            before=[i for i,row in enumerate(values) if row['received_at']<=start]
            after=[i for i,row in enumerate(values) if row['received_at']>=end]
            if not before or not after:
                covered=False
                continue
            bracket=values[before[-1]:after[0]+1]
            if (any(row.get('graphics_mhz') is None or not math.isfinite(row['graphics_mhz']) or row['graphics_mhz']<=0 for row in bracket)
                    or any(b['received_at']-a['received_at']>5 for a,b in zip(bracket,bracket[1:]))):
                covered=False
        chosen=[s['graphics_mhz'] for start,end in windows for s in values
                if start<=s['received_at']<=end and s.get('graphics_mhz') is not None and math.isfinite(s['graphics_mhz'])]
        summaries[host]={'mean_mhz':statistics.mean(chosen),'min_mhz':min(chosen),'max_mhz':max(chosen),
                         'samples':len(chosen),'complete_coverage':covered} if chosen else None
    return summaries


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace_on',type=Path);ap.add_argument('trace_off',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--power',type=Path,action='append',required=True)
    args=ap.parse_args()
    configs=[json.loads((directory/'config.json').read_text()) for directory in (args.trace_on,args.trace_off)]
    keys=('prompts','caps','tokens','repeats','thinking','policy')
    if any(configs[0][k]!=configs[1][k] for k in keys):raise ValueError('Control configurations differ')
    def rows(directory):
        found=[]
        for path in directory.glob('*.json'):
            row=json.loads(path.read_text())
            if isinstance(row,dict) and {'case','cap','repeat','chunks','token_ids'}<=row.keys():found.append(row)
        expected={(case,i) for case in configs[0]['prompts'] for i in range(configs[0]['repeats'])}
        if {(r['case'],r['repeat']) for r in found}!=expected:raise ValueError('Incomplete control workload')
        return found
    on,off=rows(args.trace_on),rows(args.trace_off)
    result=compare(on,off);samples=defaultdict(list)
    for path in args.power:
        for line in path.read_text().splitlines():
            row=json.loads(line);samples[row['host']].append(row)
    clocks={'on':clock_summary(on,samples),'off':clock_summary(off,samples)}
    clock_ratios={host:clocks['on'][host]['mean_mhz']/clocks['off'][host]['mean_mhz']
                  for host in samples if clocks['on'][host] and clocks['off'][host]}
    comparable=(len(clock_ratios)==4 and all(abs(r-1)<=.005 for r in clock_ratios.values())
                and all(clocks[mode][host]['complete_coverage'] for mode in ('on','off') for host in clock_ratios))
    result.update(trace_on=str(args.trace_on),trace_off=str(args.trace_off),clocks=clocks,
                  mean_graphics_clock_ratios=clock_ratios,graphics_clocks_within_half_percent=comparable,
                  upper95_within_one_percent=(result['prompt_bootstrap_95'] is not None
                     and result['prompt_bootstrap_95'][1]<=1.01 and comparable),
                  source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__),)},
                  note='Three predictable prompts across separate boots. Time ratio above1 means tracing is slower. Identical outputs required. This is a fixed7 instrumentation control, not a complete adaptive-policy overhead or wall-power measurement; boot effects remain a limitation.')
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','clocks')},indent=2))


if __name__=='__main__':main()
