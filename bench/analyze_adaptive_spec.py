#!/usr/bin/env python3
"""Paired, prompt-clustered speculation analysis; device energy is not wall energy."""
import argparse
import json
import re
import statistics as S
import math
import random
from collections import defaultdict
from pathlib import Path


def variant(row):
    mode = row.get('policy','fixed')
    return f"fixed{row['cap']}" if mode == 'fixed' else mode


def gaps(row):
    times = [r['seconds'] for r in row['chunks']
             if any(c.get('token_ids') for c in r['data'].get('choices',[]))]
    return [b-a for a,b in zip(times,times[1:])]


def device_energy(row, samples):
    """Trapezoidal integration at stream token boundaries, rejecting gaps >5s."""
    chunks = [r for r in row['chunks'] if any(c.get('token_ids') for c in r['data'].get('choices',[]))]
    if len(chunks)<2 or 'started_at' not in row:
        return None
    start, end = [row['started_at']+r['seconds'] for r in (chunks[0],chunks[-1])]
    total = 0.0
    if len(samples)!=4:
        return None
    for host,values in samples.items():
        energy, covered = 0.0, 0.0
        for a,b in zip(values,values[1:]):
            t0,t1 = a['received_at'],b['received_at']
            lo,hi = max(start,t0),min(end,t1)
            if hi<=lo:
                continue
            if t1-t0>5 or a['power_w'] is None or b['power_w'] is None:
                return None
            slope = (b['power_w']-a['power_w'])/(t1-t0)
            energy += (hi-lo)*(a['power_w']+slope*((lo+hi)/2-t0))
            covered += hi-lo
        if abs(covered-(end-start))>0.01:
            return None
        total += energy
    first = sum(len(c.get('token_ids') or []) for c in chunks[0]['data']['choices'])
    return total/(len(row['token_ids'])-first)


def paired_groups(runs, group_depth=1):
    result = {}
    def group_of(row):
        return '_'.join(row['case'].split('_')[:group_depth])
    for group in sorted({group_of(r) for r in runs}):
        selected = [r for r in runs if group_of(r)==group]
        for mode in sorted({variant(r) for r in selected}-{'fixed7'}):
            pairs = defaultdict(list)
            for row in selected:
                if variant(row)!=mode:
                    continue
                base = next((b for b in selected if variant(b)=='fixed7' and b['case']==row['case'] and b['repeat']==row['repeat']),None)
                if base:
                    pairs[row['case']].append(math.log(row['decode_tps']/base['decode_tps']))
            if not pairs:
                continue
            means = [S.mean(v) for v in pairs.values()]
            rng = random.Random(20260908)
            boot = (sorted(math.exp(S.mean(rng.choices(means,k=len(means)))) for _ in range(10000))
                    if len(means)>1 else None)
            result[group+'/'+mode] = dict(prompts=len(means),pairs=sum(map(len,pairs.values())),
                paired_geomean_ratio=math.exp(S.mean(means)),
                prompt_bootstrap_95=[boot[250],boot[9749]] if boot else None,
                prompt_ratios={k:math.exp(S.mean(v)) for k,v in pairs.items()})
    return result


def summarize(root, trace_root=None, power=None):
    runs = []
    for path in sorted(root.glob('*.json')):
        row = json.loads(path.read_text())
        if isinstance(row,dict) and {'case','cap','repeat','decode_tps'} <= row.keys():
            runs.append(row)
    events = defaultdict(list)
    for path in (trace_root or root).rglob('spec-trace.jsonl'):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get('event') == 'verify':
                events[row['label']].append(row)
    samples = defaultdict(list)
    if power:
        for line in power.read_text().splitlines():
            r = json.loads(line)
            samples[r['host']].append(r)
    for values in samples.values():
        values.sort(key=lambda r:r['received_at'])
    by_case = defaultdict(dict)
    for case in sorted({r['case'] for r in runs}):
        for mode in sorted({variant(r) for r in runs if r['case']==case}):
            selected = [r for r in runs if r['case']==case and variant(r)==mode]
            rows = [e for r in selected for e in events[r['label']] if e['learned']]
            # Exclude each request's first 8 feedback observations from timings.
            cycles = [e['cycle_ms'] for e in rows if e['cycle_ms'] is not None and e['observations']>=8]
            intervals = sorted(g for r in selected for g in gaps(r))
            energy = [v for r in selected if (v:=device_energy(r,samples)) is not None]
            by_case[case][mode] = {
                'runs':len(selected), 'decode_tps':[r['decode_tps'] for r in selected],
                'median_tps':S.median(r['decode_tps'] for r in selected),
                'verified_cycles':len(rows),
                'mean_emitted':S.mean(e['sampled'] for e in rows) if rows else None,
                'median_cycle_ms':S.median(cycles) if cycles else None,
                'actual_caps':sorted({e['scheduled_k'] for e in rows}),
                'cap_counts':{k:sum(e['scheduled_k']==k for e in rows) for k in (1,3,5,7)},
                'median_ttft_s':S.median(r['ttft'] for r in selected),
                'p95_emission_gap_s':intervals[min(len(intervals)-1,math.ceil(.95*len(intervals))-1)] if intervals else None,
                'device_j_per_decode_token':energy,
            }
        base = by_case[case].get('fixed7')
        if base:
            for cap,record in by_case[case].items():
                record['median_ratio_to_7'] = record['median_tps']/base['median_tps']
    graphs = set()
    for path in (trace_root or root).rglob('*.log'):
        for line in path.read_text().splitlines():
            if 'SPEC_GRAPH' in line:
                graphs.add(line[line.index('SPEC_GRAPH'):])
    memory = {}
    path = root/'memory.jsonl'
    if path.exists():
        samples = [json.loads(s) for s in path.read_text().splitlines()]
        for host in sorted({r['host'] for r in samples}):
            rows = [r for r in samples if r['host']==host]
            memory[host] = {'minimum_available_mib':min(r['available_mib'] for r in rows),
                            'new_swap_out_mib':rows[-1]['swap_out_mib']-rows[0]['swap_out_mib'],
                            'oom_delta':rows[-1]['oom_kill']-rows[0]['oom_kill'],
                            'trips':[r['reason'] for r in rows if r.get('reason')]}
    return {'screening_only':True,'cases':dict(by_case),'paired':paired_groups(runs),'graphs':sorted(graphs),'memory':memory,
            'wall_energy_measured':False,
            'note':'Short synthetic prompts and few repeats are a screen, not a promotion gate. Greedy baseline is not bitwise deterministic.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--traces',type=Path)
    ap.add_argument('--power',type=Path)
    args = ap.parse_args()
    result = summarize(args.directory,args.traces,args.power)
    (args.directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('case                      variant  median tok/s  vs K7   emitted/cycle   cycle ms')
    for case,variants in result['cases'].items():
        for cap,row in variants.items():
            print(f"{case:25} {cap:9}  {row['median_tps']:12.2f}  {row.get('median_ratio_to_7',1):5.3f}  {str(row['mean_emitted']):>14}  {str(row['median_cycle_ms']):>9}")
    print(json.dumps(result['paired'],indent=2))
    print(result['note'])


if __name__ == '__main__':
    main()
