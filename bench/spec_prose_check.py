#!/usr/bin/env python3
"""Deterministic constraints for complete prose; offline, with no model judge."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from spec_prose_results import read_results

CORPUS = Path(__file__).with_name('spec-prose-checks.json')
WORD = re.compile(r"\w+(?:['’\-]\w+)*", re.UNICODE)


def contains(text, phrase):
    pattern = r'(?<!\w)' + r'\s+'.join(re.escape(word) for word in phrase.split()) + r'(?!\w)'
    return re.search(pattern, text, re.IGNORECASE) is not None


def unfenced(text):
    return re.sub(r'^```[^\n]*\n.*?^```[ \t]*$', '', text, flags=re.MULTILINE | re.DOTALL)


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def check_constraint(text, constraint):
    kind = constraint['type']
    if kind == 'word_count':
        actual = len(WORD.findall(text))
        passed = constraint['min'] <= actual <= constraint['max']
    elif kind in ('required_items', 'forbidden_items'):
        found = [item for item in constraint['items'] if contains(text, item)]
        actual = {'found': found, 'missing': [item for item in constraint['items'] if item not in found]}
        passed = len(found) == len(constraint['items']) if kind == 'required_items' else not found
    elif kind == 'ordered_headings':
        actual = re.findall(r'^##[ \t]+(.+?)[ \t]*$', unfenced(text), re.MULTILINE)
        passed = actual == constraint['items']
    elif kind == 'bullet_list':
        actual = re.findall(r'^- (.+?)[ \t]*$', unfenced(text), re.MULTILINE)
        passed = actual == constraint['items']
    elif kind == 'table':
        lines = unfenced(text).splitlines()
        blocks, current = [], []
        for line in lines + ['']:
            if line.strip().startswith('|') and line.strip().endswith('|'):
                current.append([cell.strip() for cell in line.strip()[1:-1].split('|')])
            elif current:
                blocks.append(current)
                current = []
        actual = blocks
        expected = constraint['rows']
        passed = (len(blocks) == 1 and len(blocks[0]) == len(expected) + 2 and
                  blocks[0][0] == constraint['columns'] and
                  len(blocks[0][1]) == len(constraint['columns']) and
                  all(re.fullmatch(r':?-{3,}:?', cell) for cell in blocks[0][1]) and
                  blocks[0][2:] == expected)
    elif kind == 'embedded_json':
        blocks = re.findall(r'^```json[ \t]*\n(.*?)^```[ \t]*$', text, re.MULTILINE | re.DOTALL)
        try:
            if len(blocks) != 1:
                raise ValueError('expected exactly one fenced json block')
            # parse_constant rejects NaN/Infinity, which json.loads otherwise accepts.
            def bad_constant(value):
                raise ValueError('non-JSON number: ' + value)
            actual = json.loads(blocks[0], object_pairs_hook=strict_object, parse_constant=bad_constant)
            passed = json.dumps(actual, sort_keys=True) == json.dumps(constraint['value'], sort_keys=True)
        except (ValueError, RecursionError) as exc:
            actual, passed = {'error': str(exc)}, False
    else:
        raise ValueError('unknown prose constraint: ' + kind)
    return {'id': constraint['id'], 'type': kind, 'passed': bool(passed), 'actual': actual}


def check(text, case):
    """Score every declared constraint; one failure fails the completion."""
    constraints = case['constraints']
    if not constraints or len({c['id'] for c in constraints}) != len(constraints):
        raise ValueError('nonempty constraints with distinct ids required')
    results = [check_constraint(text, constraint) for constraint in constraints]
    return {'passed': all(row['passed'] for row in results),
            'passed_constraints': sum(row['passed'] for row in results),
            'total_constraints': len(results), 'constraints': results}


def evaluate(directory, corpus, variants=None, repeats=None):
    loaded = read_results(directory, repeats=repeats)
    rows, indexed = [], {}
    names = variants or sorted({key[2] for key in loaded})
    if len(names) != len(set(names)):
        raise ValueError('duplicate variant selection')
    for key, row in loaded.items():
        case, repeat, variant = key
        if variant not in names:
            continue
        if case not in corpus or row['prompt'] != corpus[case]['prompt']:
            raise ValueError('completion prompt does not match constrained corpus')
        result = {'case': case, 'repeat': repeat, 'variant': variant,
                  'source': row['source'], 'source_sha256': row['source_sha256'],
                  **check(row['text'], corpus[case])}
        rows.append(result)
        indexed[key] = result
    repeat_ids = repeats if repeats is not None else sorted({key[1] for key in indexed})
    if not repeat_ids or any((case, repeat, name) not in indexed
                            for case in corpus for repeat in repeat_ids for name in names):
        raise ValueError('incomplete constrained corpus/variant/repeat set')
    summaries = {}
    for name in names:
        selected = [row for row in rows if row['variant'] == name]
        constraints = defaultdict(list)
        for row in selected:
            for item in row['constraints']:
                constraints[item['type']].append(item['passed'])
        summaries[name] = {'completions': len(selected), 'passed': sum(r['passed'] for r in selected),
                           'pass_rate': sum(r['passed'] for r in selected) / len(selected),
                           'by_constraint_type': {kind: {'passed': sum(values), 'total': len(values)}
                                                  for kind, values in constraints.items()}}
    comparisons = {}
    if 'fixed7' in summaries:
        for name in names:
            if name == 'fixed7':
                continue
            regressions, improvements = [], []
            for case in corpus:
                for repeat in repeat_ids:
                    a, b = indexed[case, repeat, 'fixed7']['passed'], indexed[case, repeat, name]['passed']
                    if a and not b:
                        regressions.append({'case': case, 'repeat': repeat})
                    if b and not a:
                        improvements.append({'case': case, 'repeat': repeat})
            comparisons[name + '/fixed7'] = {
                'pass_rate_not_below_control': summaries[name]['pass_rate'] >= summaries['fixed7']['pass_rate'],
                'regressions': regressions, 'improvements': improvements}
    return {'summaries': summaries, 'comparisons': comparisons, 'completions': rows,
            'method': 'A completion passes only if every constraint passes. Named items use case-insensitive whole-item matching, headings/list/table use exact order, JSON uses exact typed values. Word counts include headings, tables, and JSON; words are Unicode word runs with internal apostrophes/hyphens. Scores check declared form and items, not factual or literary quality.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory', type=Path, help='adaptive_spec result directory')
    ap.add_argument('--corpus', type=Path, default=CORPUS)
    ap.add_argument('--variants', help='Comma-separated names; default all recorded variants')
    ap.add_argument('--repeats', help='Comma-separated repeat indices; default all recorded repeats')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    corpus = json.loads(args.corpus.read_text())
    result = evaluate(args.directory, corpus, args.variants.split(',') if args.variants else None,
                      [int(r) for r in args.repeats.split(',')] if args.repeats else None)
    (args.out or args.directory / 'prose-check-report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'summaries': result['summaries'], 'comparisons': result['comparisons']}, indent=2))


if __name__ == '__main__':
    main()
