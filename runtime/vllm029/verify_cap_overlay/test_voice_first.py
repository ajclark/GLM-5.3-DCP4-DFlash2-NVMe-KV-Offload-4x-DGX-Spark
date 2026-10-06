#!/usr/bin/env python3
"""CPU unit test for the voice-first scheduler (vLLM stubbed).

The stub AsyncScheduler models only the two gates the subclass relies on: the running
loop skips requests with current_step < next_decode_eligible_step, and the waiting loop
admits from the head of the priority queue. The real scheduler is exercised live.

    python3 test_voice_first.py
"""
import heapq
import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Req:
    def __init__(self, rid, priority=0, arrival=0.0, prefill=False):
        self.request_id, self.priority, self.arrival_time = rid, priority, arrival
        self.is_prefill_chunk, self.next_decode_eligible_step = prefill, 0

    def __lt__(self, o):
        return (self.priority, self.arrival_time) < (o.priority, o.arrival_time)


class PQ:
    def __init__(self, reqs=()):
        self._heap = list(reqs)
        heapq.heapify(self._heap)

    def add_request(self, r):
        heapq.heappush(self._heap, r)

    def remove_requests(self, rs):
        rs = set(rs)
        self._heap = [r for r in self._heap if r not in rs]
        heapq.heapify(self._heap)

    def __iter__(self):
        return iter(sorted(self._heap))

    def __bool__(self):
        return bool(self._heap)


class StubAsyncScheduler:
    def __init__(self, running=(), waiting=(), skipped=()):
        self.policy = types.SimpleNamespace(value="priority")
        self.running, self.waiting, self.skipped_waiting = list(running), PQ(waiting), PQ(skipped)
        self.current_step = 0

    def schedule(self, throttle_prefills=False):
        self.current_step += 1
        out = [r.request_id for r in self.running if self.current_step >= r.next_decode_eligible_step]
        while self.waiting:                         # admit everything visible, best first
            r = min(self.waiting._heap)
            self.waiting.remove_requests([r])
            self.running.append(r)
            out.append(r.request_id)
        return out


def load():
    logger = types.ModuleType("vllm.logger")
    logger.init_logger = lambda name: types.SimpleNamespace(info=lambda *a, **k: None)
    asched = types.ModuleType("vllm.v1.core.sched.async_scheduler")
    asched.AsyncScheduler = StubAsyncScheduler
    sys.modules.update({"vllm": types.ModuleType("vllm"), "vllm.logger": logger,
                        "vllm.v1.core.sched.async_scheduler": asched})
    spec = importlib.util.spec_from_file_location("voice_first", HERE / "vllm/v1/core/sched/voice_first.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.VoiceFirstAsyncScheduler


def main():
    VF = load()
    # No urgent request: behaves exactly like the base scheduler.
    agent = Req("agent", prefill=True)
    s = VF(running=[agent], waiting=[Req("other", arrival=1)])
    assert s.schedule() == ["agent", "other"]

    # A voice request arrives while an agent prefill runs: the agent chunk and other
    # waiting prefills are held, the voice request and running decodes are scheduled.
    agent, dec = Req("agent", prefill=True), Req("decode")
    voice, late = Req("voice", priority=-10, arrival=5), Req("late", arrival=6)
    skipped = Req("kvload", arrival=2)
    s = VF(running=[agent, dec], waiting=[voice, late], skipped=[skipped])
    assert s.schedule() == ["decode", "voice"], "first urgent step"
    assert agent.next_decode_eligible_step == 0, "eligibility restored after the step"
    assert [r.request_id for r in s.waiting] == ["late"]
    assert [r.request_id for r in s.skipped_waiting] == ["kvload"]
    # Still held while the voice request is running.
    voice.is_prefill_chunk = True
    assert s.schedule() == ["decode", "voice"]
    assert s._vf_held_steps == 2 and s._vf_held_reqs == {"agent", "late", "kvload"}
    # Voice finishes: held work resumes on the very next step and counters reset.
    s.running.remove(voice)
    assert s.schedule() == ["agent", "decode", "late"]
    assert s._vf_steps == 0 and not s._vf_held_reqs

    # An urgent request that is itself prefilling is never held; two urgent requests
    # both proceed.
    v1, v2 = Req("v1", priority=-10, prefill=True), Req("v2", priority=-5, prefill=True)
    s = VF(running=[v1, v2, Req("agent", prefill=True)])
    assert s.schedule() == ["v1", "v2"]

    # Exceptions inside the base schedule() still restore the queues.
    agent, voice, late = Req("agent", prefill=True), Req("voice", priority=-1), Req("late")
    s = VF(running=[agent, voice], waiting=[late])
    StubAsyncScheduler.schedule, orig = (lambda self, t=False: 1 / 0), StubAsyncScheduler.schedule
    try:
        s.schedule()
    except ZeroDivisionError:
        pass
    StubAsyncScheduler.schedule = orig
    assert agent.next_decode_eligible_step == 0 and [r.request_id for r in s.waiting] == ["late"]

    # Prefill cadence: with no urgent request, prefills are throttled except on every
    # Nth step and the capacity latch is cleared; an urgent period is unaffected.
    import os
    seen = []
    StubAsyncScheduler.schedule, orig = (
        lambda self, t=False: (setattr(self, "current_step", self.current_step + 1), seen.append(t))), orig
    os.environ["VLLM_PREFILL_CADENCE"] = "3"
    VF3 = load()
    s = VF3(running=[Req("agent", prefill=True), Req("decode")])
    s.prefill_capacity_bound = True
    for _ in range(6):
        s.schedule()
    assert seen == [True, True, False, True, True, False], seen
    assert s.prefill_capacity_bound is False
    seen.clear()
    s.running.append(Req("voice", priority=-10))
    s.schedule()
    assert seen == [False], seen
    os.environ["VLLM_PREFILL_CADENCE"] = "1"
    VF1 = load()
    seen.clear()
    s = VF1(running=[Req("agent", prefill=True), Req("decode")])
    for _ in range(3):
        s.schedule()
    assert seen == [False, False, False]
    # The control file overrides the cadence at runtime.
    import json, tempfile
    ctl = Path(tempfile.mkdtemp()) / "control.json"
    ctl.write_text(json.dumps({"mode": "auto", "prefill_cadence": 2}))
    os.environ["VLLM_VERIFY_CAP_CONTROL"] = str(ctl)
    VFc = load()
    seen.clear()
    s = VFc(running=[Req("agent", prefill=True), Req("decode")])
    for _ in range(4):
        s.schedule()
    assert s._cadence == 2 and seen == [True, False, True, False], seen
    del os.environ["VLLM_VERIFY_CAP_CONTROL"]
    StubAsyncScheduler.schedule = orig
    print("voice_first tests passed")


if __name__ == "__main__":
    main()
