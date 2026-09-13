#!/usr/bin/env python3
"""Summarize C1 results, rerun isolated code checks, and retain per-case comparisons."""
import argparse
import json
from pathlib import Path
import statistics
from coding_sweep import CASES, code_check, save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path);args=ap.parse_args()
    root=args.directory
    plan=json.loads((root/'plan.json').read_text())
    abort=json.loads((root/'abort-status.json').read_text()) if (root/'abort-status.json').exists() else None
    rows=[]
    for path in sorted(root.glob('k*/*-rep*.json')):
        result=json.loads(path.read_text())
        row=dict(result['summary'])
        # runpy permits generated future imports and skips the usage example.
        row['code_check']=code_check(result['result']['text'],CASES[row['case']][1])
        rows.append(row)
    measured=[r for r in rows if not r['warmup']]
    summary=[]
    for k in sorted({r['k'] for r in measured},reverse=True):
        rr=[r for r in measured if r['k']==k]
        cases={}
        for case in CASES:
            cr=[r for r in rr if r['case']==case]
            if cr:
                cases[case]={'mean_decode_tok_s':statistics.mean(r['decode_tok_s'] for r in cr),
                    'min_decode_tok_s':min(r['decode_tok_s'] for r in cr),
                    'max_decode_tok_s':max(r['decode_tok_s'] for r in cr),
                    'generated_tokens':[r['generated_tokens'] for r in cr],
                    'unique_token_hashes':len({r['token_sha256'] for r in cr}),
                    'correct':sum(r['code_check']['ok'] for r in cr),'trials':len(cr)}
        summary.append({'k':k,'geomean_decode_tok_s':statistics.geometric_mean(v['mean_decode_tok_s'] for v in cases.values()),
            'accepted_per_cycle':1+sum(r['accepted'] for r in rr)/sum(r['drafts'] for r in rr),
            'acceptance_rate':sum(r['accepted'] for r in rr)/sum(r['proposed'] for r in rr),
            'cycle_ms_estimate':sum(r['cycle_ms_estimate']*r['drafts'] for r in rr)/sum(r['drafts'] for r in rr),
            'correct':sum(r['code_check']['ok'] for r in rr),'trials':len(rr),'cases':cases})
    baseline=next((r for r in summary if r['k']==7),None)
    if baseline:
        for row in summary:row['relative_to_k7_percent']=(row['geomean_decode_tok_s']/baseline['geomean_decode_tok_s']-1)*100
    save(root/'checked-rows.json',rows)
    save(root/'summary.json',summary)
    lines=['# C1 DFlash coding sweep','']
    if abort:
        restored='The exact original K=7 containers are restored and generation verified.' if abort.get('restored') else 'Restoration of the original K=7 service is in progress.'
        completed=', '.join(f'K={k}' for k in abort['completed_k'])
        skipped=', '.join(f'K={k}' for k in abort['skipped_benchmarks_k'])
        lines.extend([f'**Stopped at the user\'s request.** Completed {completed}. Skipped benchmarks: {skipped}. '+restored,
                      'Restoration evidence is recorded in abort-status.json. No fresh K=7 control was measured in this run.', ''])
        if abort.get('prior_k7_decode_tok_s'):
            best=max(summary,key=lambda r:r['geomean_decode_tok_s'])
            gap=100*(best['geomean_decode_tok_s']/abort['prior_k7_decode_tok_s']-1)
            lines.extend([f"The earlier high-K sweep measured K=7 at {abort['prior_k7_decode_tok_s']:.2f} decode tokens/second. The best average here, K={best['k']} at {best['geomean_decode_tok_s']:.2f}, differs by {gap:+.2f}% from that earlier control. This comparison spans separate runs; small differences do not demonstrate a performance advantage.", ''])
    if baseline and (root/'status.json').exists():
        best=max(summary,key=lambda r:r['geomean_decode_tok_s'])
        lines.extend([f"**K={best['k']} has the highest measured average on this coding corpus at {best['geomean_decode_tok_s']:.2f} decode tokens/second.** "
                      f"The exact original K=7 containers are restored and serving. "
                      f"Generated code passed {sum(r['correct'] for r in summary)}/{sum(r['trials'] for r in summary)} measured output checks.", ''])
        runner_up=sorted(summary,key=lambda r:r['geomean_decode_tok_s'],reverse=True)[1]
        gap=100*(best['geomean_decode_tok_s']/runner_up['geomean_decode_tok_s']-1)
        lines.extend([f"K={best['k']} and K={runner_up['k']} differ by only {gap:.2f}%. "
                      "Output lengths and timing vary across repetitions; a small gap should not be treated as a demonstrated performance advantage.", ''])
    lines.extend([
        ('Planned order: ' if abort else 'Order: ')+', '.join('K='+str(k) for k in plan['order'])+'. K counts speculative tokens, excluding the bonus token.',
        f"TP4/DCP2, vLLM 0.29.0, DFlash2, C1, greedy decoding, thinking off. Three coding tasks, one warmup each and {plan['repetitions']} measured repetitions each; 1200-token output limit. Other serving arguments are held fixed.",
        '', '| K | Merge tok/s | LRU tok/s | Toposort tok/s | Geomean tok/s | vs K7 | Tokens/cycle | Cycle ms (est.) | Acceptance | Correct |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for r in summary:
        vals=[f"{r['cases'][case]['mean_decode_tok_s']:.2f}" if case in r['cases'] else 'pending' for case in CASES]
        rel=f"{r['relative_to_k7_percent']:+.1f}%" if baseline else ('not measured' if abort else 'pending')
        lines.append(f"| {r['k']} | {' | '.join(vals)} | {r['geomean_decode_tok_s']:.2f} | {rel} | {r['accepted_per_cycle']:.2f} | {r['cycle_ms_estimate']:.1f} | {r['acceptance_rate']:.1%} | {r['correct']}/{r['trials']} |")
    lines.extend(['', 'Decode throughput excludes the first streamed token batch and its latency, using token IDs rather than text chunks. The geometric mean gives each coding task equal weight. Tokens/cycle means one bonus token plus mean accepted draft tokens; acceptance is accepted/proposed draft tokens. Engine metrics verify each configured K.',
        '', 'Correctness is checked by executing generated code with independent cases in a bounded subprocess. These checks cover behavior, not a formal complexity proof. Output lengths and token hashes are retained because different greedy responses can affect throughput comparisons. Three repetitions and short prompts do not establish long-context or concurrent performance.',
        '', 'Raw request/response and metric deltas are in each k*/ directory. summary.json contains per-case ranges and output lengths. checked-rows.json contains independent correctness results, including warmups. Restoration is recorded in abort-status.json for an interrupted sweep, or status.json for a completed sweep.'])
    if (root/'verification.json').exists():
        verification=json.loads((root/'verification.json').read_text())
        lines.extend(['', f"The completed run has {verification['warmups']} warmups and {verification['measured_requests']} measured requests in the configured order. "
                      "Restoration, truncation and memory observations are recorded in verification.json."])
    failures=[r for r in measured if not r['code_check']['ok']]
    if failures:
        lines.extend(['', 'Failed measured outputs remain included in throughput averages. The small sample does not establish a correctness difference between draft lengths.', ''])
        for row in failures:
            check=row['code_check']
            reason=(check.get('stderr') or check.get('error') or 'failed check').splitlines()[-1]
            lines.append(f"- K={row['k']}, {row['case']}, repetition {row['rep']+1}: {reason}")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(line for line in lines if line.startswith('|')))

if __name__=='__main__':main()
