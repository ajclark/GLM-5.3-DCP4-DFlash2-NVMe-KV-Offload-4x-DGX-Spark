"""Combine measured context points into an explicitly bounded interpolation table."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('inputs',type=Path,nargs='+')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--max-context',type=int,required=True)
    args=ap.parse_args()
    rows=[json.loads(p.read_text()) for p in args.inputs]
    config=rows[0]['config'];lane=rows[0]['lane']
    if (not 1<=len(rows)<=8 or not 0<args.max_context<=config['max_model_len']
            or any(r['config']!=config or r['lane']!=lane for r in rows)):
        ap.error('inconsistent lane or invalid context bound')
    points=[]
    for row in sorted(rows,key=lambda r:r['context_point']):
        point={k:row[k] for k in ('cycle_ms','conditional_acceptance_prior','prior_strength')}
        point['context']=row['context_point']
        if not 0<=point['context']<args.max_context or (points and point['context']<=points[-1]['context']):
            ap.error('context points must be unique and inside the requested range')
        points.append(point)
    result={'config':config,'lane':lane,'context_range':[0,args.max_context],
            'calibration_points':points,
            'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs},
            'method':'Linear interpolation of measured cycle costs and conditional acceptance priors; nearest endpoint outside the measured points.',
            'note':'Performance estimates within this operating range, not a claim that every intermediate context or prompt was measured.'}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
