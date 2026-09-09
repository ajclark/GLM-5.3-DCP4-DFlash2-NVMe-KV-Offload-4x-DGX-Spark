"""Audit a locked paired evaluation; performance alone never enables deployment."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from analyze_adaptive_spec import gaps, paired_groups, summarize, variant


def audit_pairs(runs, cases, repeats):
    expected = {(case,repeat,mode) for case in cases for repeat in range(repeats)
                for mode in ('fixed7','adaptive')}
    actual = [(r['case'],r['repeat'],variant(r)) for r in runs]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Evaluation has missing, duplicate, or unexpected pairs')
    if any(not all(math.isfinite(r[key]) and r[key]>0 for key in ('decode_tps','ttft')) for r in runs):
        raise ValueError('Evaluation has invalid timing')


def latency_ratios(runs):
    """Compare per-request p95 burst gaps, taking medians across repeats."""
    result = {}
    for case in sorted({r['case'] for r in runs}):
        values = {}
        for mode in ('fixed7','adaptive'):
            selected = [r for r in runs if r['case']==case and variant(r)==mode]
            p95 = []
            for row in selected:
                intervals = sorted(gaps(row))
                if not intervals:
                    raise ValueError('Insufficient stream intervals for latency gate')
                p95.append(intervals[(95*len(intervals)+99)//100-1])
            values[mode] = {'ttft':statistics.median(r['ttft'] for r in selected),
                            'p95_gap':statistics.median(p95)}
        result[case] = {key:values['adaptive'][key]/values['fixed7'][key]
                        for key in ('ttft','p95_gap')}
    return result


def performance_gates(runs):
    paired = paired_groups(runs)
    prose,code = paired['prose/adaptive'],paired['code/adaptive']
    if not prose['prompt_bootstrap_95'] or not code['prompt_bootstrap_95']:
        raise ValueError('Multiple independent prompts per workload are required')
    latency = latency_ratios(runs)
    return {
        'prose_throughput':prose['paired_geomean_ratio']>=1.08 and prose['prompt_bootstrap_95'][0]>1,
        'coding_noninferiority':code['prompt_bootstrap_95'][0]>=.98,
        'ttft':all(r['ttft']<=1.05 for r in latency.values()),
        'p95_emission_gap':all(r['p95_gap']<=1.05 for r in latency.values()),
    },paired,latency


def aggregate_decode_rates(runs):
    totals = {}
    for row in runs:
        key = row['case'].split('_')[0]+'/'+variant(row)
        tokens = [chunk for chunk in row['chunks']
                  if any(choice.get('token_ids') for choice in chunk['data'].get('choices',[]))]
        first = sum(len(choice.get('token_ids') or []) for choice in tokens[0]['data']['choices'])
        total = totals.setdefault(key,{'decoded_tokens':0,'decode_seconds':0})
        total['decoded_tokens'] += len(row['token_ids'])-first
        total['decode_seconds'] += tokens[-1]['seconds']-tokens[0]['seconds']
    for total in totals.values():
        total['tokens_per_second'] = total['decoded_tokens']/total['decode_seconds']
    return totals


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--lock',type=Path,required=True)
    ap.add_argument('--traces',type=Path,required=True)
    ap.add_argument('--power',type=Path)
    args = ap.parse_args()
    lock = json.loads(args.lock.read_text())
    for name,sha in lock['source_sha256'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:
            raise ValueError('Locked source changed: '+name)
    corpus = json.loads(Path(lock.get('heldout_corpus','bench/spec-heldout.json')).read_text())
    config = json.loads((args.directory/'config.json').read_text())
    if (config['prompts']!=corpus or config['repeats']!=lock['heldout_repeats']
            or config['tokens']!=lock['heldout_output_tokens'] or config['thinking']):
        raise ValueError('Workload does not match locked thinking-off evaluation')
    if hashlib.sha256(Path(config['costs']).read_bytes()).hexdigest()!=lock['costs_sha256']:
        raise ValueError('Cost table differs from evaluation lock')
    runs = []
    for path in args.directory.glob('*.json'):
        row = json.loads(path.read_text())
        if isinstance(row,dict) and {'case','repeat','cap','decode_tps'}<=row.keys():
            runs.append(row)
    audit_pairs(runs,corpus,lock['heldout_repeats'])
    labels = {r['label'] for r in runs}
    events = []
    for path in args.traces.rglob('spec-trace.jsonl'):
        events += [r for line in path.read_text().splitlines()
                   if (r:=json.loads(line)).get('event')=='verify' and r.get('label') in labels]
    if {r['label'] for r in events}!=labels:
        raise ValueError('Missing runtime trace coverage')
    if any(r['version']!=lock['policy_version'] or r.get('dropped') or r.get('writer_error')
           or (r['learned'] and r['scheduled_k']!=r['cap']) for r in events):
        raise ValueError('Runtime version, scheduled cap, or telemetry integrity mismatch')
    gates,paired,latency = performance_gates(runs)
    summary = summarize(args.directory,args.traces,args.power)
    summary.update(screening_only=False,note='Complete locked synthetic evaluation; deployment requires separate correctness, reliability, and wall-energy review.')
    result = {'evaluation_lock_sha256':hashlib.sha256(args.lock.read_bytes()).hexdigest(),
              'complete_pairs':len(runs)//2,'performance_gates':gates,'paired':paired,
              'paired_subgroups':paired_groups(runs,group_depth=2),
              'aggregate_decode_rates':aggregate_decode_rates(runs),
              'matched_case_latency_ratios':latency,'summary':summary,
              'promotion_approved':False,'wall_energy_gate':'unmeasured',
              'note':'Locked synthetic workload. Correctness, C2, long-context, and reliability evidence must also be reviewed. NVIDIA device energy does not satisfy the wall-energy gate.'}
    (args.directory/'gate-report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'pairs':result['complete_pairs'],'performance_gates':gates,'paired':paired},indent=2))


if __name__=='__main__':
    main()
