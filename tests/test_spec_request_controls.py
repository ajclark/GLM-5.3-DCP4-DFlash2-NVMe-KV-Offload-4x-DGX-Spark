import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from spec_request_controls import controlled_body


def captured():
    return {'model': 'glm-5.3', 'temperature': 0, 'max_tokens': 128,
            'messages': [{'role': 'system', 'content': 'private system context'},
                         {'role': 'user', 'content': 'Write Python code.'}],
            'vllm_xargs': {'spec_workload': 'code_generate', 'spec_phase': 'user_turn',
                           'spec_hint_strength': 'weak'}, 'chat_template_kwargs': {'enable_thinking': False}}


def test_hint_controls_preserve_captured_messages_and_never_increase_limit():
    original = captured()
    saved = copy.deepcopy(original)
    bodies = [controlled_body(original, 'hints', v, v, 256) for v in ('off', 'on', 'wrong', 'fixed7')]
    assert original == saved
    assert all(b['messages'] == original['messages'] and b['max_tokens'] == 128 for b in bodies)
    assert all(b['chat_template_kwargs'] == original['chat_template_kwargs'] for b in bodies)
    assert [b['vllm_xargs']['spec_use_hints'] for b in bodies] == [False, True, True, False]
    assert bodies[2]['vllm_xargs']['spec_workload'] == 'prose'
    assert all(b['vllm_xargs']['spec_confidence_trace'] is False for b in bodies)
    bodies[0]['messages'][0]['content'] = 'modified replay copy'
    assert original == saved


def test_confidence_controls_have_same_fixed_policy_and_no_hints():
    off = controlled_body(captured(), 'confidence', 'off', 'same', 256)
    on = controlled_body(captured(), 'confidence', 'on', 'same', 256)
    assert on['vllm_xargs']['spec_confidence_trace'] is True
    on['vllm_xargs']['spec_confidence_trace'] = False
    assert on == off
    assert off['vllm_xargs']['spec_policy'] == 'fixed'
    assert off['vllm_xargs']['spec_use_hints'] is False


def test_abstention_or_tool_followup_cannot_be_relabelled_as_eligible():
    for key, value in [('spec_workload', 'mixed'), ('spec_phase', 'tool_followup'), ('spec_hint_strength', 'abstain')]:
        original = captured()
        original['vllm_xargs'][key] = value
        with pytest.raises(ValueError):
            controlled_body(original, 'hints', 'on', 'x', 128)
