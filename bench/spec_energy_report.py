"""Paired NVIDIA-device energy, including request prefill; never wall energy."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from analyze_adaptive_spec import device_energy,paired_groups


def integrate(samples,start,end):
    if len(samples)!=4 or end<=start:return None
    total=0.0
    for values in samples.values():
        covered=0.0
        for left,right in zip(values,values[1:]):
            t0,t1=left['received_at'],right['received_at']
            lo,hi=max(start,t0),min(end,t1)
            if hi<=lo:continue
            if t1-t0>5 or left['power_w'] is None or right['power_w'] is None:return None
            slope=(right['power_w']-left['power_w'])/(t1-t0)
            total+=(hi-lo)*(left['power_w']+slope*((lo+hi)/2-t0))
            covered+=hi-lo
        if abs(covered-(end-start))>.01:return None
    return total


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory',type=Path)
    ap.add_argument('--power',type=Path,required=True)
    args=ap.parse_args()
    samples=defaultdict(list)
    for line in args.power.read_text().splitlines():
        row=json.loads(line);samples[row['host']].append(row)
    for values in samples.values():values.sort(key=lambda row:row['received_at'])
    requests=[];decode_rows=[];request_rows=[]
    for path in sorted(args.directory.glob('*.json')):
        row=json.loads(path.read_text())
        if not isinstance(row,dict) or not {'case','cap','repeat','decode_tps','chunks'}<=row.keys():continue
        end=row['started_at']+max(c['seconds'] for c in row['chunks'])
        energy=integrate(samples,row['started_at'],end)
        decode=device_energy(row,samples)
        request_per_token=None if energy is None else energy/len(row['token_ids'])
        requests.append({'label':row['label'],'device_request_joules':energy,
                         'device_request_j_per_output_token':request_per_token,
                         'device_decode_j_per_token':decode})
        # Reuse the predeclared prompt-cluster bootstrap for energy ratios;
        # here a smaller ratio is better. No throughput is substituted in output.
        if decode is not None:decode_rows.append({**row,'decode_tps':decode})
        if request_per_token is not None:request_rows.append({**row,'decode_tps':request_per_token})
    result={'sensor':'NVIDIA-reported device power, summed across four nodes; not wall power',
            'requests':requests,'paired_decode_energy':paired_groups(decode_rows),
            'paired_request_energy':paired_groups(request_rows),
            'lower_ratio_is_better':True,'wall_energy_measured':False,
            'missing_decode_records':len(requests)-len(decode_rows),
            'missing_request_records':len(requests)-len(request_rows),
            'note':'Two-second samples with <=5s gaps and bracketing coverage required. Whole-request interval starts at submission and ends at the last SSE event; prefill is included. Cold/warm cache conditions must still be compared separately.'}
    (args.directory/'device-energy-report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='requests'},indent=2))


if __name__=='__main__':main()
