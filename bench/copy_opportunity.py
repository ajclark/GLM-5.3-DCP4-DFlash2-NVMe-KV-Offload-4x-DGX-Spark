#!/usr/bin/env python3
"""Offline suffix-copy screen at recorded emission boundaries.

Proposal selection sees only the committed prefix. Future output is used only
to score that proposal on the baseline trajectory; this is not a live replay
or an estimate of a policy's end-to-end speedup.
"""
import argparse
from collections import deque
import json
from pathlib import Path


def propose(history, widths=(16,8,4), capacity=7, lookback=32768):
    end = len(history)
    for width in widths:
        if end < width+capacity:
            continue
        suffix = history[-width:]
        # Most recent complete match; all proposed tokens must already exist.
        for start in range(end-width-capacity, max(-1,end-lookback-1), -1):
            if history[start:start+width] == suffix:
                return history[start+width:start+width+capacity], width
    return [], 0


class CopyIndex:
    """Incremental exact suffix lookup over committed history, for offline scoring.

    Admit a match only after its entire continuation exists. The index retains
    the most recent complete occurrence and expires positions outside lookback.
    No generated future token is visible to proposal selection.
    """

    def __init__(self, widths=(16,8,4), capacity=7, lookback=32768):
        self.widths, self.capacity, self.lookback = widths, capacity, lookback
        self.history = []
        self.positions = {width:0 for width in widths}
        self.latest = {width:{} for width in widths}
        self.queues = {width:deque() for width in widths}

    def extend(self, tokens):
        self.history.extend(tokens)
        end = len(self.history)
        cutoff = max(0,end-self.lookback)
        for width in self.widths:
            latest, queue = self.latest[width], self.queues[width]
            while queue and queue[0][0] < cutoff:
                start, key = queue.popleft()
                if latest.get(key) == start:
                    del latest[key]
            start = max(self.positions[width],cutoff)
            limit = end-width-self.capacity+1
            for position in range(start,limit):
                key = tuple(self.history[position:position+width])
                latest[key] = position
                queue.append((position,key))
            self.positions[width] = max(start,limit)

    def propose(self):
        for width in self.widths:
            if len(self.history) < width+self.capacity:
                continue
            start = self.latest[width].get(tuple(self.history[-width:]))
            if start is not None:
                return self.history[start+width:start+width+self.capacity], width
        return [], 0


def score_run(run, lookback=32768):
    chunks = [row['data'] for row in run['chunks']]
    prompt = next((row['prompt_token_ids'] for row in chunks if row.get('prompt_token_ids')),None)
    if prompt is None:
        raise ValueError('recording lacks prompt token IDs')
    output = run['token_ids']
    index = CopyIndex(lookback=lookback)
    index.extend(prompt)
    offset, boundaries, matches, accepted, full = 0, 0, 0, [], 0
    for chunk in chunks:
        for choice in chunk.get('choices',[]):
            tokens = choice.get('token_ids') or []
            index.extend(tokens)
            offset += len(tokens)
            if not tokens or offset+7 > len(output):
                continue
            boundaries += 1
            candidate,width = index.propose()
            if candidate:
                matches += 1
                a = 0
                for x,y in zip(candidate,output[offset:]):
                    if x != y:
                        break
                    a += 1
                accepted.append(a)
                full += a==7
    return {'label':run['label'],'case':run['case'],'boundaries':boundaries,
            'copy_matches':matches,'accepted_prefixes_on_baseline':accepted,
            'full_seven_token_matches':full,
            'matched_fraction':matches/boundaries if boundaries else 0,
            'mean_accepted_when_matched':sum(accepted)/len(accepted) if accepted else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--lookback',type=int,default=32768)
    ap.add_argument('--cap',type=int,choices=(1,3,5,7),help='Restrict to a fixed-cap baseline')
    ap.add_argument('--out',type=Path)
    args = ap.parse_args()
    if not 1 <= args.lookback <= 180224:
        ap.error('lookback must be 1..180224')
    rows = []
    for path in sorted(args.directory.glob('*.json')):
        run = json.loads(path.read_text())
        if isinstance(run,dict) and {'case','label','chunks','token_ids'} <= run.keys():
            if args.cap is None or (run.get('cap')==args.cap and run.get('policy','fixed')=='fixed'):
                rows.append(score_run(run,args.lookback))
    result = {'screening_only':True,'lookback':args.lookback,'widths':[16,8,4],
              'capacity':7,'runs':rows,'warning':'Baseline-trajectory opportunity only; no throughput or energy claim.'}
    (args.out or args.directory/'copy-opportunity.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:
        print(row['label'],'matches',str(row['copy_matches'])+'/'+str(row['boundaries']),
              'mean accepted',row['mean_accepted_when_matched'])


if __name__ == '__main__':
    main()
