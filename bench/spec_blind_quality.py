"""Mask policy labels for a local review of held-out prose openings."""
import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    config=json.loads((args.directory/'config.json').read_text())
    cases=[case for case in config['prompts'] if case.startswith('prose_')]
    rng=random.SystemRandom();pairs=[];mapping={}
    for i,case in enumerate(cases):
        rows=[]
        for mode in ('fixed','adaptive'):
            candidates=list(args.directory.glob(f'*-{case}-r0-{mode}-k7.json'))
            if len(candidates)!=1:raise RuntimeError('First-repeat prose pairs are not complete')
            rows.append((mode,json.loads(candidates[0].read_text())))
        rng.shuffle(rows);identifier=f'prose-{i+1:02d}'
        pairs.append({'id':identifier,'prompt':config['prompts'][case],
                      **{slot:row['text'] for slot,(_,row) in zip(('A','B'),rows)}})
        mapping[identifier]={'case':case,**{slot:mode for slot,(mode,_) in zip(('A','B'),rows)}}
    args.out.mkdir(parents=True,exist_ok=False)
    blind={'rubric':'Compare coherence, instruction adherence, and clarity of these truncated openings. Do not penalize either for failing to complete a long requested answer within the shared 256-token budget. Record A/B/tie and a reason before opening policy-map.json. This is not a comprehensive factual or full-answer quality audit.',
           'pairs':pairs}
    path=args.out/'blind-pairs.json';path.write_text(json.dumps(blind,indent=2)+'\n')
    (args.out/'policy-map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    print('Masked',len(pairs),'prose pairs; review',str(path),'; SHA256',hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=='__main__':main()
