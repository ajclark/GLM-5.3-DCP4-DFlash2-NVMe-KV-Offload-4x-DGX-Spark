"""Executable prose constraints and judge-neutral, swapped quality packets."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'bench'))
from spec_prose_check import CORPUS, check, check_constraint, evaluate
from spec_prose_judge import export_pairs, score, unique_object, wilson
from spec_prose_results import paired_results

CORPUS_DATA = json.loads(CORPUS.read_text())


def compliant(case):
    constraints = {c['type']: c for c in case['constraints']}
    sections = constraints['ordered_headings']['items']
    text = '\n\n'.join('## ' + section for section in sections)
    text += '\n' + '. '.join(constraints['required_items']['items']) + '.\n'
    if 'bullet_list' in constraints:
        text += '\n'.join('- ' + item for item in constraints['bullet_list']['items']) + '\n'
        table = constraints['table']
        text += '\n'.join('| ' + ' | '.join(row) + ' |' for row in
                          [table['columns'], ['---'] * len(table['columns']), *table['rows']]) + '\n'
        text += '```json\n' + json.dumps(constraints['embedded_json']['value']) + '\n```\n'
    # Padding leaves generous room within all declared windows; the unit tests
    # isolate mechanics. This deliberately makes no claim about literary merit.
    sentences = 16 if 'bullet_list' in constraints else (constraints['word_count']['min'] + 5) // 6
    text += ' '.join(['A practical choice needs patient discussion.'] * sentences)
    if 'bullet_list' in constraints:
        text += ' ' + ' '.join(['Clear notes help volunteers understand their responsibilities.'] * 7)
    return text


@pytest.mark.parametrize('name', list(CORPUS_DATA))
def test_all_twenty_corpus_entries_are_executable_and_satisfiable(name):
    assert len(CORPUS_DATA) == 20
    case = CORPUS_DATA[name]
    assert {'word_count', 'required_items', 'forbidden_items', 'ordered_headings'} <= {c['type'] for c in case['constraints']}
    result = check(compliant(case), case)
    assert result['passed'], result
    assert all(item['passed'] for item in result['constraints'])


def test_word_count_has_inclusive_bounds_and_declared_punctuation_rules():
    constraint = {'id': 'n', 'type': 'word_count', 'min': 4, 'max': 4}
    assert check_constraint("Nia's well-kept café costs 12.", constraint)['passed'] is False
    assert check_constraint("Nia's well-kept café: 12.", constraint)['passed'] is True
    assert not check_constraint('one two three', constraint)['passed']
    assert not check_constraint('one two three four five', constraint)['passed']


def test_named_and_forbidden_items_are_case_insensitive_whole_items():
    required = {'id': 'names', 'type': 'required_items', 'items': ['Nia', 'Reading Room']}
    assert check_constraint('NIA visits a reading\nroom.', required)['passed']
    assert not check_constraint('Niall visits the Reading Room.', required)['passed']
    forbidden = {'id': 'ban', 'type': 'forbidden_items', 'items': ['magic', 'zero risk']}
    assert check_constraint('magical thinking has risks.', forbidden)['passed']
    assert not check_constraint('This is ZERO\nRISK.', forbidden)['passed']


@pytest.mark.parametrize('headings', ['## Second\n## First', '## First\n## Second\n## First',
                                     '### First\n## Second', 'First\nSecond'])
def test_section_order_level_and_uniqueness_are_enforced(headings):
    constraint = {'id': 'sections', 'type': 'ordered_headings', 'items': ['First', 'Second']}
    assert not check_constraint(headings, constraint)['passed']
    assert check_constraint('## First\nText\n## Second\nText', constraint)['passed']


@pytest.mark.parametrize('replacement', ['{bad json}', '{"room":"Elm Reading Room","open":1,"volunteers":["Nia","Tomas"],"chairs":12,"lamps":2}',
                                       '{"room":"Elm Reading Room","open":true,"open":true,"volunteers":["Nia","Tomas"],"chairs":12,"lamps":2}',
                                       '{"room":"Elm Reading Room","open":true,"volunteers":["Nia","Tomas"],"chairs":NaN,"lamps":2}'])
def test_embedded_json_rejects_syntax_wrong_types_duplicate_keys_and_nan(replacement):
    case = CORPUS_DATA['prose_check_list_table_json']
    constraint = next(c for c in case['constraints'] if c['type'] == 'embedded_json')
    assert not check_constraint('```json\n' + replacement + '\n```', constraint)['passed']


def test_list_table_and_json_fail_independently_on_format_damage():
    case = CORPUS_DATA['prose_check_list_table_json']
    text = compliant(case)
    result = check(text, case)
    assert result['passed'], result
    for before, after, failed in [('- Check the door', '1. Check the door', 'list'),
                                  ('| Chairs | 12 |', '| Chairs | 13 |', 'table'),
                                  ('"open": true', '"open": "true"', 'json')]:
        result = check(text.replace(before, after), case)
        assert {c['id'] for c in result['constraints'] if not c['passed']} == {failed}
    json_constraint = next(c for c in case['constraints'] if c['type'] == 'embedded_json')
    assert not check_constraint(text + '\n```json\n{}\n```', json_constraint)['passed']


def write_results(path, variants=('fixed7', 'lossy-m1.0'), prompts=None, repeats=2, texts=None):
    prompts = prompts or {'prose_one': 'Explain a map.', 'prose_two': 'Write a story.'}
    path.mkdir()
    (path / 'config.json').write_text(json.dumps({'prompts': prompts, 'repeats': repeats}))
    for case, prompt in prompts.items():
        for repeat in range(repeats):
            for variant in variants:
                row = {'case': case, 'repeat': repeat, 'variant': variant, 'cap': 7,
                       'policy': 'fixed' if variant == 'fixed7' else variant,
                       'text': texts[case] if texts else ('Control prose.' if variant == 'fixed7' else 'Treatment prose.'),
                       'reasoning_text': 'PRIVATE REASONING',
                       'message_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
                (path / f'{case}-r{repeat}-{variant}.json').write_text(json.dumps(row))
    return path


def test_swapped_export_has_exactly_60_blind_packets_and_complete_provenance(tmp_path):
    prompts = {f'prose_{i:02d}': f'Prompt {i}' for i in range(15)}
    left = write_results(tmp_path / 'left', ('fixed7',), prompts)
    right = write_results(tmp_path / 'right', ('lossy-m1.0',), prompts)
    out = tmp_path / 'blind'
    manifest = export_pairs(left, right, out)
    assert manifest['judgements_expected'] == 60 and manifest['completion_pairs'] == 30
    assert len(manifest['sources']) == 60
    for first, second in zip(manifest['pairs'][::2], manifest['pairs'][1::2]):
        a, b = [json.loads((out / row['file']).read_text()) for row in (first, second)]
        assert a['prompt'] == b['prompt']
        assert a['A'] == b['B'] and a['B'] == b['A']
        assert first['A'] == second['B'] and first['B'] == second['A']
        assert first['sha256'] == hashlib.sha256((out / first['file']).read_bytes()).hexdigest()
        assert 'PRIVATE REASONING' not in json.dumps(a)
        assert not {'case', 'repeat', 'variant', 'control', 'treatment', 'source'} & a.keys()
    with pytest.raises(FileExistsError):
        export_pairs(left, right, out)


def test_scoring_unblinds_swaps_and_declares_ties_precision_and_position_bias(tmp_path):
    directory = write_results(tmp_path / 'runs')
    manifest = export_pairs(directory, directory, tmp_path / 'pairs')
    votes = {row['id']: ('A' if row['A'] == 'lossy' else 'B') for row in manifest['pairs']}
    result = score(manifest, votes)
    assert result['lossy_wins'] == 8 and result['win_rate'] == 1
    assert result['wilson_95'][0] == pytest.approx(.675592435, abs=1e-8)
    assert result['position_consistent_pairs'] == 4
    assert 'n=60 detects only gross loss' in result['precision_note']
    always_a = score(manifest, {key: 'A' for key in votes})
    assert always_a['win_rate'] == .5 and always_a['position_consistent_pairs'] == 0
    tied = score(manifest, {key: 'tie' for key in votes})
    assert tied['win_rate'] == 0 and tied['tie_adjusted_score'] == .5
    assert tied['decisive_win_rate'] is None and tied['decisive_wilson_95'] is None
    assert wilson(30, 60) == pytest.approx([.377350243, .622649757], abs=1e-8)
    assert wilson(0, 0) is None


def test_judgements_require_every_pair_once_and_only_declared_votes(tmp_path):
    directory = write_results(tmp_path / 'runs')
    manifest = export_pairs(directory, directory, tmp_path / 'pairs')
    good = {row['id']: 'tie' for row in manifest['pairs']}
    for bad in [{**good, 'unknown': 'A'}, dict(list(good.items())[1:]), {**good, next(iter(good)): 'lossy'}]:
        with pytest.raises(ValueError):
            score(manifest, bad)
    with pytest.raises(ValueError, match='duplicate JSON key'):
        json.loads('{"p":"A","p":"B"}', object_pairs_hook=unique_object)
    manifest['pairs'][1]['A'], manifest['pairs'][1]['B'] = manifest['pairs'][0]['A'], manifest['pairs'][0]['B']
    with pytest.raises(ValueError, match='swapped positions'):
        score(manifest, good)


def test_quality_pairing_rejects_duplicates_missing_repeats_and_prompt_drift(tmp_path):
    directory = write_results(tmp_path / 'runs')
    assert len(paired_results(directory, directory, repeats=[1])) == 2
    path = directory / 'prose_one-r0-lossy-m1.0.json'
    backup = path.read_text()
    (directory / 'duplicate.json').write_text(backup)
    with pytest.raises(ValueError, match='duplicate'):
        paired_results(directory, directory)
    (directory / 'duplicate.json').unlink()
    path.unlink()
    with pytest.raises(ValueError, match='incomplete'):
        paired_results(directory, directory)
    changed = json.loads(backup)
    changed['message_sha256'] = 'drift'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='different prompts'):
        paired_results(directory, directory)


def test_both_variants_missing_the_same_prompt_cannot_shrink_denominator(tmp_path):
    directory = write_results(tmp_path / 'runs')
    for path in directory.glob('prose_two-*.json'):
        path.unlink()
    with pytest.raises(ValueError, match='incomplete'):
        paired_results(directory, directory)


def test_deterministic_report_pairs_full_corpus_and_preserves_failures(tmp_path):
    corpus = {key: value for key, value in CORPUS_DATA.items() if key == 'prose_check_list_table_json'}
    texts = {key: compliant(case) for key, case in corpus.items()}
    directory = write_results(tmp_path / 'runs', prompts={key: value['prompt'] for key, value in corpus.items()}, texts=texts)
    path = directory / 'prose_check_list_table_json-r1-lossy-m1.0.json'
    row = json.loads(path.read_text())
    row['text'] = row['text'].replace('"open": true', '"open": "true"')
    path.write_text(json.dumps(row))
    result = evaluate(directory, corpus)
    assert result['summaries']['fixed7']['pass_rate'] == 1
    assert result['summaries']['lossy-m1.0']['pass_rate'] == .5
    assert result['comparisons']['lossy-m1.0/fixed7'] == {
        'pass_rate_not_below_control': False,
        'regressions': [{'case': 'prose_check_list_table_json', 'repeat': 1}], 'improvements': []}
    assert json.loads(path.read_text()) == row  # Evidence is not overwritten.
    with pytest.raises(ValueError, match='incomplete'):
        evaluate(directory, CORPUS_DATA)


def test_blind_cli_accepts_arbitrary_variants_and_keeps_mapping_out_of_packet(tmp_path):
    directory = write_results(tmp_path / 'runs', ('lossy-m0.5', 'lossy-m2.5-p0.1'))
    out = tmp_path / 'blind'
    result = subprocess.run([sys.executable, str(ROOT / 'bench/spec_blind_quality.py'), str(directory),
                             '--out', str(out), '--variants', 'lossy-m0.5,lossy-m2.5-p0.1'],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    packet = json.loads((out / 'blind-pairs.json').read_text())
    assert len(packet['pairs']) == 2
    assert 'lossy-m' not in json.dumps(packet)
    mapping = json.loads((out / 'policy-map.json').read_text())
    assert all({r['A'], r['B']} == {'lossy-m0.5', 'lossy-m2.5-p0.1'} for r in mapping.values())


def test_judge_export_and_score_cli_are_entirely_offline(tmp_path):
    directory = write_results(tmp_path / 'runs')
    out = tmp_path / 'pairs'
    result = subprocess.run([sys.executable, str(ROOT / 'bench/spec_prose_judge.py'), 'export', str(directory), str(directory),
                             '--out', str(out), '--treatment', 'lossy-m1.0', '--repeats', '0'],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / 'manifest.json').read_text())
    votes = tmp_path / 'votes.json'
    votes.write_text(json.dumps({r['id']: 'tie' for r in manifest['pairs']}))
    result = subprocess.run([sys.executable, str(ROOT / 'bench/spec_prose_judge.py'), 'score', str(out / 'manifest.json'), str(votes)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['ties'] == 4


@pytest.mark.parametrize('field,value', [('tokens', 512), ('thinking', True)])
def test_separate_directories_cannot_pair_different_generation_settings(tmp_path, field, value):
    left = write_results(tmp_path / 'left', ('fixed7',))
    right = write_results(tmp_path / 'right', ('lossy-m1.0',))
    for directory in (left, right):
        config = json.loads((directory / 'config.json').read_text())
        config.update(tokens=256, thinking=False)
        if directory == right:
            config[field] = value
        (directory / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='different token budgets or thinking'):
        paired_results(left, right)


def test_quality_pairing_checks_actual_prompt_ids_when_present(tmp_path):
    directory = write_results(tmp_path / 'runs')
    for path in directory.glob('prose_one-r0-*.json'):
        row = json.loads(path.read_text())
        row['chunks'] = [{'data': {'prompt_token_ids': [1, 2 if row['variant'] == 'fixed7' else 3]}}]
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match='different prompts'):
        paired_results(directory, directory)


def test_blind_default_still_reads_legacy_fixed_adaptive_result_names(tmp_path):
    directory = write_results(tmp_path / 'runs', ('fixed7', 'adaptive'))
    for path in directory.glob('prose*.json'):
        row = json.loads(path.read_text())
        row.pop('variant')
        path.write_text(json.dumps(row))
    out = tmp_path / 'blind'
    result = subprocess.run([sys.executable, str(ROOT / 'bench/spec_blind_quality.py'), str(directory), '--out', str(out)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    mapping = json.loads((out / 'policy-map.json').read_text())
    assert all({row['A'], row['B']} == {'fixed', 'adaptive'} for row in mapping.values())
