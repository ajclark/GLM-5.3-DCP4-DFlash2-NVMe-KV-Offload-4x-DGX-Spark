import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'bench'))
from spec_herdr_watch import assess


def state(status='idle', returncode=0):
    return {'returncode': returncode, 'response': {'result': {'agent': {'agent_status': status}}}}


def pane(text):
    return {'returncode': 0, 'response': {'text': text}}


def test_monitor_distinguishes_quiet_done_blocked_and_error_candidates():
    assert assess(state(), pane('Answer completed'))['assessment'] == 'done_or_idle'
    assert assess(state('working'), pane('Generating'))['assessment'] == 'working'
    assert assess(state(), pane('Waiting for approval'))['assessment'] == 'blocked_candidate'
    assert assess(state(), pane('API error: context length exceeded'))['assessment'] == 'error_candidate'
    assert assess(state(returncode=1), pane(''))['status'] == 'watch_error'


def test_old_error_text_remains_a_candidate_with_native_state_preserved():
    result = assess(state('working'), pane('Previous request failed\nRetrying'))
    assert result['status'] == 'working'
    assert result['assessment'] == 'error_candidate'
