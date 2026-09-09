#!/usr/bin/env python3
"""Fit one context-band cost table from verified FULL-graph calibration runs."""
import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--context-range',type=int,nargs=2,default=[0,512],metavar=('LOW','HIGH'))
    ap.add_argument('--label-prefix',help='Restrict calibration to predeclared fixed runs')
    args = ap.parse_args()
    low,high = args.context_range
    if not 0<=low<high<=180224:
        ap.error('invalid context range')
    graphs = set()
    for name in ('live.log','experiment.log'):
        for path in args.directory.rglob(name):
            for match in re.finditer(r'SPEC_GRAPH k=(\d+) target_tokens=(\d+) padded_tokens=(\d+) mode=(\w+)',path.read_text()):
                k,t,p,mode = match.groups()
                if mode=='FULL' and int(t)==int(p)==int(k)+1:
                    graphs.add(int(k))
    if graphs != {1,3,5,7}:
        raise SystemExit('cost fit requires actual FULL dispatch evidence for all four caps')
    rows, sources = [], {}
    for path in args.directory.rglob('spec-trace.jsonl'):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows += [json.loads(s) for s in path.read_text().splitlines()]
    selected = [r for r in rows if r.get('event')=='verify' and r['learned'] and r['mode']=='fixed'
                and r['observations']>=8 and r['cycle_ms'] is not None and 'smoke' not in r['label']
                and (not args.label_prefix or r['label'].startswith(args.label_prefix))]
    costs, counts = {}, {}
    for k in (1,3,5,7):
        times = [r['cycle_ms'] for r in selected if r['scheduled_k']==k and low<=r['context']<high]
        if len(times)<50:
            raise SystemExit(f'cap {k}: fewer than 50 eligible calibration cycles')
        costs[str(k)] = statistics.median(times)
        counts[str(k)] = len(times)
    full = [r for r in selected if r['scheduled_k']==7 and low<=r['context']<high]
    risks = [sum(r['accepted']>=j for r in full) for j in range(7)]
    successes = [sum(r['accepted']>j for r in full) for j in range(7)]
    # Jeffreys smoothing keeps an unobserved conditional position neutral and
    # avoids a degenerate zero/one prior from a finite calibration sample.
    prior = [(s+.5)/(n+1) for s,n in zip(successes,risks)]
    result = {'config':{'tp':4,'dcp':2,'max_model_len':180224,'draft_capacity':7},
              'lane':'tp4-dcp2-l180224-k7','context_range':[low,high],
              'cycle_ms':costs,'calibration_cycles':counts,'source_sha256':sources,
              'context_point':int(statistics.median(r['context'] for r in selected if low<=r['context']<high)),
              'conditional_acceptance_prior':prior,'prior_strength':2.0,
              'prior_risk_counts':risks,'prior_success_counts':successes,
              'label_prefix':args.label_prefix,
              'graph_mode':'FULL','screening_calibration':True}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
