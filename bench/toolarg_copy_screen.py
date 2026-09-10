#!/usr/bin/env python3
"""CPU prompt-lookup falsification screen on private, recorded tool arguments.

This scores baseline continuations, not a live policy replay or measured speedup.
No generation calls, saved token IDs, request texts or response texts are written.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from harness_capture import (TokenizationUnavailable, category_counts, flatten_context,
                             load_capture, make_tokenizer, ratio, tokenizer_options, write_report)
from harness_call_attrib import analyze as attribute, fmt, markdown as attribution_markdown

POLICIES = {'n4': (4,), 'n8': (8,), 'n16': (16,), 'longest': (16, 8, 4)}


def validate_curve(row):
    curve = row.get('accept_rate_by_pos')
    if (not isinstance(curve, list) or len(curve) != 7
            or any(not isinstance(x, (int, float)) or not 0 <= x <= 1 for x in curve)
            or any(a < b for a, b in zip(curve, curve[1:]))):
        raise ValueError('copy screening needs a monotone seven-position acceptance curve per tool call')
    drafts, accepted = row.get('spec_drafts', 0), row.get('spec_accepted_tokens', 0)
    if drafts <= 0 or abs(sum(curve) - accepted / drafts) > .001:
        raise ValueError('acceptance curve disagrees with draft counters')
    return 1 + sum(curve)


def context_index(context, widths=(4, 8, 16), capacity=7):
    # Most recent occurrence with a full K7 proposal available, as copy_opportunity.py.
    return {n: {tuple(context[i:i + n]): i + n
                for i in range(len(context) - n - capacity + 1)} for n in widths}


def score_tokens(context, output, baseline, widths=(16, 8, 4)):
    """Score every full-cycle position; proposal selection cannot see its future.

    Longest *anchor* first, most recent complete occurrence. A >=8 match means
    eight future tokens actually agree, stronger than merely existing in context.
    Static request context only; the branch cannot copy generated output history.
    """
    indices = context_index(context, widths)
    positions = max(0, len(output) - 7)  # leave seven draft tokens + one bonus
    matched = eight = available_eight = prefix_sum = 0
    emitted_sum = 0.
    widths_used, histogram = Counter(), Counter()
    start = time.perf_counter()
    for position in range(positions):
        source, width = None, None
        for n in widths:
            if position < n:
                continue
            source = indices[n].get(tuple(output[position - n:position]))
            if source is not None:
                width = n
                break
        if source is None:
            emitted_sum += baseline
            continue
        matched += 1
        widths_used[str(width)] += 1
        agreed = 0
        # Diagnostic longest agreeing continuation; no future data selects source.
        while source + agreed < len(context) and position + agreed < len(output):
            if context[source + agreed] != output[position + agreed]:
                break
            agreed += 1
        available_eight += len(context) - source >= 8
        eight += agreed >= 8
        prefix_sum += agreed
        histogram[str(min(agreed, 8))] += 1  # last bucket means >=8
        emitted_sum += 1 + min(agreed, 7)
    elapsed = time.perf_counter() - start
    return {'positions': positions, 'copy_matches': matched,
            'context_continuation_ge8_positions': available_eight,
            'correct_copy_ge8_positions': eight, 'copy_ge8_share': ratio(eight, positions),
            'matching_anchor_share': ratio(matched, positions),
            'mean_agreeing_continuation_when_matched': ratio(prefix_sum, matched),
            'agreement_histogram_capped_at_8': dict(histogram), 'anchor_widths': dict(widths_used),
            'baseline_emitted_per_cycle': baseline,
            'candidate_emitted_per_cycle': (emitted_sum + (len(output) - positions) * baseline) / len(output) if output else baseline,
            'tail_positions_using_dflash': len(output) - positions,
            'scan_seconds': elapsed, 'scan_seconds_per_position': ratio(elapsed, positions)}


def pool(policy_calls, rows):
    tool_tokens = all_tokens = baseline_tool_cycles = candidate_tool_cycles = 0.
    baseline_all_cycles = candidate_all_cycles = 0.
    positions = matches = eight = 0
    maximum_scan = 0.
    for call, row in zip(policy_calls, rows):
        g, t, base = row.get('generation_tokens', 0), call['allocated_tool_tokens'], call['baseline_emitted_per_cycle']
        candidate = call['candidate_emitted_per_cycle']
        tool_tokens += t
        all_tokens += g
        baseline_tool_cycles += t / base
        candidate_tool_cycles += t / candidate
        baseline_all_cycles += g / base
        candidate_all_cycles += (g - t) / base + t / candidate
        positions += call['positions']
        matches += call['copy_matches']
        eight += call['correct_copy_ge8_positions']
        maximum_scan = max(maximum_scan, call['scan_seconds_per_position'] or 0.)
    baseline_tool = ratio(tool_tokens, baseline_tool_cycles)
    candidate_tool = ratio(tool_tokens, candidate_tool_cycles)
    base_all, candidate_all = ratio(all_tokens, baseline_all_cycles), ratio(all_tokens, candidate_all_cycles)
    all_gain = candidate_all / base_all - 1 if base_all else None
    decode = sum(r.get('server_decode_s') or 0 for r in rows)
    drafts = sum(r.get('spec_drafts') or 0 for r in rows)
    cycle_s = ratio(decode, drafts)
    # Conservative charge: worst per-position scoring cost at every modeled cycle.
    # Includes agreement scoring unavailable live, excludes one-off index construction.
    net_gain = ((1 + all_gain) / (1 + maximum_scan / cycle_s) - 1
                if all_gain is not None and cycle_s else None)
    return {'allocated_tool_tokens': tool_tokens, 'generation_tokens': all_tokens,
            'positions': positions, 'copy_matches': matches, 'correct_copy_ge8_positions': eight,
            'matching_anchor_share': ratio(matches, positions), 'copy_ge8_share': ratio(eight, positions),
            'baseline_tool_emitted_per_cycle': baseline_tool,
            'candidate_tool_emitted_per_cycle': candidate_tool,
            'tool_cycle_gain': candidate_tool / baseline_tool - 1 if baseline_tool else None,
            'baseline_all_emitted_per_cycle': base_all, 'candidate_all_emitted_per_cycle': candidate_all,
            'all_token_cycle_gain': all_gain, 'maximum_scan_seconds_per_position': maximum_scan,
            'observed_decode_seconds_per_cycle': cycle_s, 'cost_charged_all_token_gain': net_gain,
            'gate_above_3_percent': net_gain is not None and net_gain > .03,
            'decision': 'eligible for guarded integration measurement; not a measured speedup'
                        if net_gain is not None and net_gain > .03 else 'do not integrate from this screen'}


def screen(rows, tokenizer, provenance=None):
    if tokenizer is None:
        return {'status': 'skipped', 'source': provenance,
                'reason': 'exact token copy screen requires --tokenize and confirmed --no-hold',
                'decision': 'no evidence for the +3% pooled gate; do not integrate'}
    policy_rows = {name: [] for name in POLICIES}
    for i, row in enumerate(rows):
        counts = category_counts(row, tokenizer)
        g = row.get('generation_tokens') or 0
        t = g * counts[2] / sum(counts) if sum(counts) else 0.
        if row.get('tool_args_text'):
            base = validate_curve(row)
            context, output = tokenizer.encode(flatten_context(row)), tokenizer.encode(row['tool_args_text'])
            for name, widths in POLICIES.items():
                policy_rows[name].append({'call': i + 1, 'allocated_tool_tokens': t,
                                         **score_tokens(context, output, base, widths)})
        else:
            drafts = row.get('spec_drafts') or 0
            base = 1 + row.get('spec_accepted_tokens', 0) / drafts if drafts else 1.
            for name in POLICIES:
                policy_rows[name].append({'call': i + 1, 'allocated_tool_tokens': 0.,
                                         **score_tokens([], [], base)})
    return {'status': 'complete', 'source': provenance, 'capacity': 7,
            'policies': {name: {'pooled': pool(calls, rows), 'calls': calls}
                         for name, calls in policy_rows.items()},
            'tokenizer_post_requests': getattr(tokenizer, 'requests', None),
            'notes': [
                'Chat order is approximate: tools schema, then messages in order, literal arguments; no true chat delimiters.',
                'CPU /tokenize uses independent <=2048-character chunks; tokens near chunk boundaries can differ.',
                'Only static request context is searched; output anchor must already have 4/8/16 committed argument tokens.',
                'Choose longest anchor then most recent full K7 occurrence; future output scores it but never chooses it.',
                'All full-cycle output positions are screened, not observed cycle boundaries; short tails fall back to DFlash.',
                'tool_args_text concatenates separate argument streams; boundaries are not available in this capture.',
                'Baseline emitted/cycle is 1+sum(call accept_rate_by_pos); category acceptance and actual cycle starts are unavailable.',
                'Fallback uses DFlash on no match; a bad copy is charged even when DFlash would have done better.',
                'Pool cycle costs harmonically using tokenizer category shares normalized to generation_tokens.',
                'Copy >=8 means the selected continuation agrees with at least eight actual output tokens.',
                'Cost charge uses worst observed offline scan seconds/position at every cycle; indexing and server integration costs remain unmeasured.',
                'A positive gate permits a guarded measured prototype only; no training or generation is performed.',
            ]}


def markdown(report):
    lines = ['# Tool-argument prompt-lookup screen', '', 'Status: ' + report['status'] + '.', '']
    if report['status'] != 'complete':
        return '\n'.join(lines + [report['reason'], '', report['decision'], ''])
    lines += ['| Policy | Anchor hit share | Correct copy >=8 share | Tool cycle gain | All-token cycle gain | Cost-charged gain | >3% gate |',
              '|---|---:|---:|---:|---:|---:|---|']
    for name, row in report['policies'].items():
        p = row['pooled']
        values = [fmt(p[k] * 100 if p[k] is not None else None, 2) + '%' for k in
                  ('matching_anchor_share', 'copy_ge8_share', 'tool_cycle_gain', 'all_token_cycle_gain', 'cost_charged_all_token_gain')]
        lines.append('| ' + ' | '.join([name, *values, str(p['gate_above_3_percent'])]) + ' |')
    return '\n'.join(lines + ['', 'Limits:', ''] + ['- ' + n for n in report['notes']]) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--attribution-out', type=Path, help='Write attribution in the same process, reusing private in-memory tokenization')
    tokenizer_options(parser)
    args = parser.parse_args()
    rows, provenance = load_capture(args.capture)
    tokenizer = None
    try:
        tokenizer = make_tokenizer(args)
        report = screen(rows, tokenizer, provenance)
        attribution = attribute(rows, provenance, tokenizer, args.capture.parent) if args.attribution_out else None
    except TokenizationUnavailable as error:
        report = {'status': 'skipped', 'source': provenance, 'reason': str(error),
                  'decision': 'no evidence for the +3% pooled gate; do not integrate'}
        attribution = attribute(rows, provenance, source_dir=args.capture.parent) if args.attribution_out else None
        if attribution:
            attribution['notes'].append(str(error))
    write_report(args.out, report, markdown(report))
    if attribution is not None:
        write_report(args.attribution_out, attribution, attribution_markdown(attribution))
    print(markdown(report))


if __name__ == '__main__':
    main()
