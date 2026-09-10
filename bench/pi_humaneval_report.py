#!/usr/bin/env python3
"""Audit and tabulate the completed pi/HumanEval stack comparison."""
import argparse
import collections
import csv
import json
from pathlib import Path

import pi_humaneval as he
from pi_humaneval_compare import report


def full_concurrency_intervals(path, concurrency):
    """Conservative plateau estimate from existing one-second counter samples."""
    samples = [json.loads(line) for line in path.read_text().splitlines()]
    groups, current = [], []
    for row in samples:
        full = row.get('vllm:num_requests_running') == concurrency and row.get('vllm:num_requests_waiting') == 0
        if not full or (current and row['time']-current[-1]['time'] > 3):
            if current: groups.append(current)
            current = []
        if full: current.append(row)
    if current: groups.append(current)
    intervals = []
    for group in groups:
        # Discard a full five-second exporter tick after a concurrency change.
        stable = [r for r in group if r['time'] >= group[0]['time'] + 5]
        if len(stable) < 2 or stable[-1]['time']-stable[0]['time'] < 5: continue
        seconds = stable[-1]['time']-stable[0]['time']
        tokens = stable[-1]['vllm:generation_tokens_total']-stable[0]['vllm:generation_tokens_total']
        if tokens <= 0: continue
        intervals.append({'start':stable[0]['time'],'end':stable[-1]['time'],
                          'seconds':seconds,'output_tokens':tokens})
    seconds = sum(r['seconds'] for r in intervals)
    tokens = sum(r['output_tokens'] for r in intervals)
    return {'aggregate_tok_s':tokens/seconds if seconds else None,
            'seconds':seconds,'output_tokens':tokens,'intervals':intervals,
            'note':'Short telemetry estimate: gauge continuously at C, no queue, first 5s discarded, at least 5s retained per interval.'}


def plot(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for lane, label, color in [('deployed','Deployed','#2563eb'), ('adaptive','Adaptive V5','#d97706')]:
        data = sorted((r for r in rows if r['lane']==lane), key=lambda r:r['concurrency'])
        for ax, field, title in zip(axes, ['aggregate_tok_s','server_decode_tok_s'],
                ['Whole batch throughput', 'Per-request server decode throughput']):
            ax.plot([r['concurrency'] for r in data], [r[field] for r in data],
                    marker='o', color=color, label=label, linewidth=2)
            anchors = [r for r in rows if r['lane']=='restored']
            if lane=='deployed' and anchors:
                ax.scatter([r['concurrency'] for r in anchors], [r[field] for r in anchors],
                           marker='s', facecolors='none', edgecolors='#475569', s=70,
                           label='Restored baseline', zorder=4)
            ax.set(title=title, xlabel='Maximum concurrent pi sessions', ylabel='Output tokens / second')
            ax.set_xticks(range(1,13)); ax.set_ylim(bottom=0); ax.grid(alpha=.2)
    axes[0].legend(frameon=False)
    config = json.loads((out/'config.json').read_text())
    title = 'HumanEval through pi · GLM-5.3 · TP4/DCP2\n12 fixed tasks per cell · thinking '+config['thinking']+' · finite batches include tool pauses and draining tails'
    if (out/'ABORTED.json').exists(): title += '\nStopped during adaptive C10; paired results cover C1–C9'
    fig.suptitle(title, fontsize=12)
    fig.savefig(out/'throughput.png', dpi=180)
    fig.savefig(out/'throughput.svg')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('out', type=Path)
    ap.add_argument('--plot', action='store_true', help='Also create PNG/SVG charts; requires matplotlib')
    args = ap.parse_args()
    out = args.out.resolve()
    report(out)
    rows = json.loads((out/'summary.json').read_text())
    columns = ['lane','concurrency','tasks','passed','errors','output_tokens','wall_s',
               'aggregate_tok_s','request_tok_s','server_decode_tok_s','median_task_s',
               'median_ttft_s','api_calls','mean_running','max_running']
    with (out/'summary.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    expected = {r['task_id'] for r in json.loads((he.ASSETS/'humaneval12.json').read_text())['tasks']}
    audit = {'complete_marker': (out/'COMPLETE').exists(), 'aborted':(out/'ABORTED.json').exists(),
             'cells': [], 'incomplete_cells': [], 'prompt_pairs': [],
             'provider_settings_match': True, 'first_request_message_hashes_match': True,
             'reasoning_breakdown': 'Provider count may be absent; streamed thinking is counted as characters, never estimated tokens.'}
    tasks = {}
    for lane in ('deployed','adaptive','restored'):
        for cell in sorted((out/lane).glob('c*')):
            if not (cell/'summary.json').exists():
                audit['incomplete_cells'].append({'lane':lane,'cell':cell.name,
                    'completed_solves':len(list(cell.glob('HumanEval_*/result.json')))})
                continue
            actual = {}
            for path in cell.glob('HumanEval_*/result.json'):
                r = json.loads(path.read_text())
                actual[r['task_id']] = r
                tasks[lane,cell.name,r['task_id']] = r
            audit['cells'].append({'lane':lane,'cell':cell.name,'task_set_complete':set(actual)==expected,
                'passed':sum(r['check']['passed'] for r in actual.values()),
                'errors':sum(bool(r['errors']) for r in actual.values()),
                'thinking_characters':sum(q['delta_chars'].get('thinking_delta',0) for r in actual.values() for q in r['requests'])})
    audit['full_concurrency_intervals'] = [
        {'lane':r['lane'],'concurrency':r['concurrency'],
         **full_concurrency_intervals(out/r['lane']/f"c{r['concurrency']:02d}"/'metrics.jsonl',r['concurrency'])}
        for r in rows]
    for c in range(1,13):
        for tid in sorted(expected):
            a = tasks.get(('deployed',f'c{c:02d}',tid))
            b = tasks.get(('adaptive',f'c{c:02d}',tid))
            if a is None or b is None: continue
            pa, pb = a['requests'][0]['payload'], b['requests'][0]['payload']
            keys = ['model','provider','thinking','temperature','top_p','max_tokens',
                    'chat_template_kwargs','stream_options','tools','message_count']
            settings_equal = all(pa.get(k)==pb.get(k) for k in keys)
            messages_equal = pa['messages_sha256']==pb['messages_sha256']
            audit['provider_settings_match'] &= settings_equal
            audit['first_request_message_hashes_match'] &= messages_equal
            audit['prompt_pairs'].append({'concurrency':c,'task_id':tid,'settings_equal':settings_equal,
                'messages_equal':messages_equal,'deployed_tokens':a['output_tokens'],
                'adaptive_tokens':b['output_tokens'],'deployed_passed':a['check']['passed'],
                'adaptive_passed':b['check']['passed']})
    trace_path = out/'deployment/spark-06c4.local/kvcache/spec-trace.jsonl'
    if not trace_path.exists(): trace_path = out/'adaptive-policy-at-abort.jsonl'
    traces = [json.loads(line) for line in trace_path.read_text().splitlines()] if trace_path.exists() else []
    policy = []
    for c in range(1,13):
        selected = [r for (lane,cell,_),r in tasks.items() if lane=='adaptive' and cell==f'c{c:02d}']
        if not selected: continue
        start, end = min(r['start'] for r in selected), max(r['end'] for r in selected)
        samples = [r for r in traces if r.get('event')=='verify' and start<=r.get('time',0)<=end]
        policy.append({'concurrency':c,'rows':len(samples),
            'caps':dict(collections.Counter(r['scheduled_k'] for r in samples)),
            'eligible_rows':sum(r['eligible'] for r in samples),
            'ineligible_short_rows':sum(not r['eligible'] and r['scheduled_k']<7 for r in samples),
            'writer_errors':sum(bool(r.get('writer_error')) for r in samples),
            'max_dropped':max((r.get('dropped',0) for r in samples),default=0)})
    audit['policy'] = policy
    audit['functional_failures'] = [
        {'lane':lane, 'cell':cell, 'task_id':tid, 'result':r['check']['result'],
         'output_tokens':r['output_tokens'], 'seconds':r['seconds']}
        for (lane,cell,tid),r in tasks.items() if not r['check']['passed']]
    audit['measurement_flags'] = [
        {'lane':r['lane'], 'concurrency':r['concurrency'], 'flags':r['validity_flags']}
        for r in rows if r['validity_flags']]
    if (out/'restoration.json').exists():
        audit['exact_originals_restored'] = json.loads((out/'restoration.json').read_text())['exact_originals_restored']
    he.write_json(out/'audit.json', audit)
    lines = ['', '## Functional outcomes and measurement checks', '']
    for lane in ('deployed','adaptive','restored'):
        lane_rows = [r for r in rows if r['lane']==lane]
        if lane_rows:
            lines += [f"{lane}: {sum(r['passed'] for r in lane_rows)}/{sum(r['tasks'] for r in lane_rows)} "
                      f"official tests passed across {len(lane_rows)} completed cells.", '']
    lines += ['Functional failures remain in throughput measurements. The failed solves are:', '',
              '| Lane | C | Task | Output tokens | Task seconds |',
              '|:---|---:|:---|---:|---:|']
    for r in audit['functional_failures']:
        lines.append(f"| {r['lane']} | {int(r['cell'][1:])} | {r['task_id']} | {r['output_tokens']} | {r['seconds']:.2f} |")
    lines += ['', f"Completed cells with measurement validity flags: {len(audit['measurement_flags'])}.", '',
        '## Output lengths and batch durations', '',
        'Different reasoning lengths change the mix of work in each batch. Aggregate tok/s '
        'differences therefore combine serving speed, output length, tool use, and time spent '
        'at each realized concurrency; they are not pure engine speedup estimates.', '',
        '| Lane | C | Output tokens | Batch seconds |',
        '|:---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['lane']} | {r['concurrency']} | {r['output_tokens']} | {r['wall_s']:.2f} |")
    lines += ['', '## Additional measurements', '',
        '| Lane | C | Server decode tok/s | Median task seconds | Median first-output seconds | Mean / max running requests |',
        '|:---|---:|---:|---:|---:|:---|']
    for r in rows:
        lines.append(f"| {r['lane']} | {r['concurrency']} | {r['server_decode_tok_s']:.2f} | {r['median_task_s']:.2f} | {r['median_ttft_s']:.2f} | {r['mean_running']:.2f} / {r['max_running']:.0f} |")
    lines += ['', '## Adaptive policy observations', '',
        '| C | Completed verification rows | Caps (K: rows) | C1-eligible rows |',
        '|---:|---:|:---|---:|']
    for p in policy:
        lines.append(f"| {p['concurrency']} | {p['rows']} | {p['caps']} | {p['eligible_rows']} |")
    lines += ['', '## Intervals with all C requests running', '',
        'This supplementary estimate addresses the long draining tails. It uses the existing '
        'server generation counter only while the exported running-request gauge stays at C '
        'with no queued requests. The first five seconds after entering each interval are discarded '
        'for exporter lag, then at least five seconds must remain. Rates pool tokens and elapsed '
        'time across qualifying intervals. These short telemetry windows are not independently '
        'repeated steady-state tests; changing prompt mix and exporter granularity still matter.', '',
        '| Lane | C | Aggregate tok/s at full C | Retained seconds |',
        '|:---|---:|---:|---:|']
    for r in audit['full_concurrency_intervals']:
        rate = f"{r['aggregate_tok_s']:.2f}" if r['aggregate_tok_s'] is not None else 'insufficient interval'
        lines.append(f"| {r['lane']} | {r['concurrency']} | {rate} | {r['seconds']:.1f} |")
    lines += ['', 'Verification traces are joined by each cell’s wall-clock interval. They exclude some '
              'terminal/invalid-feedback cycles by design. Short caps can occur in C>1 cells when '
              'only one request remains active. Any ineligible short-cap row is retained in audit.json '
              'for inspection; the trace does not claim every launched session is always decoding.', '',
              f"Matched initial request pairs: {len(audit['prompt_pairs'])}; native provider settings equal: "
              f"{audit['provider_settings_match']}; serialized initial messages equal: "
              f"{audit['first_request_message_hashes_match']}.", '']
    if (out/'restoration.json').exists():
        restoration = json.loads((out/'restoration.json').read_text())
        lines += ['Exact original containers, images, commands, selected environment, and Python mounts restored: '
                  + str(restoration['exact_originals_restored']) + '.', '']
    with (out/'REPORT.md').open('a') as file: file.write('\n'.join(lines))
    if args.plot:
        plot(out, rows)
        with (out/'REPORT.md').open('a') as file:
            file.write('\n![Throughput comparison](throughput.png)\n')
    print(json.dumps({'cells':len(audit['cells']),'matched_prompt_pairs':len(audit['prompt_pairs']),
        'provider_settings_match':audit['provider_settings_match'],
        'first_request_message_hashes_match':audit['first_request_message_hashes_match'],
        'complete':audit['complete_marker']}))


if __name__ == '__main__': main()
