"""Mask policy labels for a local review of held-out prose openings."""
import argparse
import hashlib
import json
import random
from pathlib import Path

from spec_prose_results import paired_results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--variants',default='fixed,adaptive',help='Two variant names, control,treatment (fixed aliases fixed7)')
    ap.add_argument('--repeat',type=int,default=0)
    args=ap.parse_args()
    names=args.variants.split(',')
    if len(names)!=2 or names[0]==names[1]:ap.error('provide two distinct variant names')
    # Preserve the original fixed alias and default map labels for old reviews.
    variants=['fixed7' if name=='fixed' else name for name in names]
    paired=paired_results(args.directory,args.directory,*variants,repeats=[args.repeat])
    rng=random.SystemRandom();pairs=[];mapping={}
    for i,pair in enumerate(paired):
        case=pair[0]['case']
        rows=list(zip(names,pair))
        rng.shuffle(rows);identifier=f'prose-{i+1:02d}'
        pairs.append({'id':identifier,'prompt':pair[0]['prompt'],
                      **{slot:row['text'] for slot,(_,row) in zip(('A','B'),rows)}})
        mapping[identifier]={'case':case,**{slot:mode for slot,(mode,_) in zip(('A','B'),rows)}}
    args.out.mkdir(parents=True,exist_ok=False)
    blind={'rubric':'Compare coherence, instruction adherence, and clarity of these truncated openings. Do not penalize either for failing to complete a long requested answer within the shared 256-token budget. Record A/B/tie, whether either opening is incoherent, and a reason before opening policy-map.json. This is not a comprehensive factual or full-answer quality audit.',
           'pairs':pairs}
    path=args.out/'blind-pairs.json';path.write_text(json.dumps(blind,indent=2)+'\n')
    (args.out/'policy-map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    print('Masked',len(pairs),'prose pairs; review',str(path),'; SHA256',hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=='__main__':main()
