#!/usr/bin/env python3
"""Combine the high/low DFlash sweeps, normalizing each to its own K=7 control."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--high',type=Path,required=True)
    ap.add_argument('--low',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--plot',action='store_true')
    args=ap.parse_args()
    rows=[];controls={};series={}
    for name,root in [('high',args.high),('low',args.low)]:
        assert json.loads((root/'status.json').read_text())['ok'],root
        summary=json.loads((root/'summary.json').read_text())
        control=next(r for r in summary if r['k']==7)
        controls[name]={'source':str(root),'summary':control}
        measured=[r for r in json.loads((root/'checked-rows.json').read_text()) if not r['warmup']]
        group=[]
        for row in sorted(summary,key=lambda r:r['k']):
            normalized=100*row['geomean_decode_tok_s']/control['geomean_decode_tok_s']
            repeats=[]
            kr=[r for r in measured if r['k']==row['k']]
            for rep in sorted({r['rep'] for r in kr}):
                repeats.append(100*statistics.geometric_mean(r['decode_tok_s'] for r in kr if r['rep']==rep)/control['geomean_decode_tok_s'])
            item={**row,'source_run':name,'source_directory':str(root),'normalized_throughput_percent':normalized,
                  'repetition_aggregate_range_percent':[min(repeats),max(repeats)]}
            group.append(item)
            if name=='low' or row['k']!=7:rows.append(item)
        series[name]=group
    rows.sort(key=lambda r:r['k'])
    assert [r['k'] for r in rows]==list(range(1,17))
    args.out.mkdir(exist_ok=True,parents=True)
    best=max(rows,key=lambda r:r['normalized_throughput_percent'])
    output={'rows':rows,'controls':controls,'highest_normalized_average_k':best['k'],
        'normalization':'Each sweep divided by its own K=7 geometric-mean control. Ranges are min/max of three repetition-level task geometric means, not confidence intervals.'}
    (args.out/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['# C1 coding: locating the DFlash sweet spot','',
        f"The highest normalized average is K={best['k']}, at {best['normalized_throughput_percent']:.1f}% of its K=7 control. The original K=7 configuration is restored.",
        '', 'The upper sweep ran 16 through 7; the lower sweep ran 6 through 1, then remeasured 7. Each used the same three short coding prompts, one warmup per task, three measured repetitions, thinking off, C1, TP4/DCP2 and the same vLLM 0.29.0 image. K is the number of draft tokens, excluding the bonus token.',
        '', f"K=7 control throughput: upper sweep {controls['high']['summary']['geomean_decode_tok_s']:.2f} tok/s; lower sweep {controls['low']['summary']['geomean_decode_tok_s']:.2f} tok/s. Each point is normalized to the control from its own sweep, so the table does not silently pool controls measured at different times.",
        '', '| K | Measured tok/s | Relative to its K7 control | Cycle ms (est.) | Tokens advanced/cycle | Correct outputs |',
        '|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['k']} | {r['geomean_decode_tok_s']:.2f} | {r['normalized_throughput_percent']:.1f}% | {r['cycle_ms_estimate']:.1f} | {r['accepted_per_cycle']:.2f} | {r['correct']}/{r['trials']} |")
    lines.extend(['', 'Throughput depends on useful tokens advanced per cycle divided by cycle time. Draft acceptance percentage alone does not measure speed: reducing K can raise acceptance percentage while reducing the number of tokens advanced.',
        '', 'Outputs vary between repetitions and can contain errors; all outputs remain in throughput averages. Small gaps are not evidence of a reliable winner. These short, thinking-off C1 measurements do not locate a universal optimum for long context, reasoning or multiple concurrent clients.'])
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5.3),layout='constrained')
        for name,color,label in [('high','#6680a6','K=16→7 sweep'),('low','#176c55','K=6→1, then K=7 control')]:
            group=series[name];xs=[r['k'] for r in group]
            ax.plot(xs,[r['normalized_throughput_percent'] for r in group],marker='o',color=color,label=label)
            ax.vlines(xs,[r['repetition_aggregate_range_percent'][0] for r in group],
                      [r['repetition_aggregate_range_percent'][1] for r in group],color=color,alpha=.5,lw=3)
        ax.axhline(100,color='#777777',ls='--',lw=1)
        ax.set(xlabel='DFlash K (speculative tokens)',ylabel='Decode throughput relative to K=7 (%)',
               title='C1 coding throughput across DFlash lengths',xticks=list(range(1,17)))
        ax.grid(axis='y',alpha=.2);ax.legend(frameon=False)
        fig.text(.5,-.025,'Each sweep uses its own K=7 control. Vertical bars: min–max of 3 repetitions, not confidence intervals.',ha='center',fontsize=9)
        fig.savefig(args.out/'throughput.png',dpi=180,bbox_inches='tight')
        fig.savefig(args.out/'throughput.svg',bbox_inches='tight')
        lines.extend(['', '![DFlash sweep throughput](throughput.png)'])
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:10]))

if __name__=='__main__':main()
