# SPDX-License-Identifier: Apache-2.0
"""Voice-first scheduling: a request with priority < 0 (e.g. a voice client) never waits behind
other requests' prefills.

vLLM's priority policy only orders the waiting queue and picks preemption victims. A
running prefill still takes the whole step budget (2048 tokens here, ~3.6 s per chunk at
long context), so a voice request that arrives during a long agent prefill waits for all
of it. While any urgent request is running or waiting, this scheduler holds every other
request's prefill work: running prefill chunks are skipped for the step, and non-urgent
waiting requests are not admitted. Other requests' decodes continue. Held work resumes on
the first step with no urgent request left.

Prefill cadence (VLLM_PREFILL_CADENCE=N, default 1 = off): with no urgent request
active, prefill chunks run only on every Nth step while other requests are decoding, so
decoders are not starved by a long prefill (vLLM's own DP prefill-balancing gate, which
the single-engine core never turns on). A lone prefill is never held; the waiting-queue
latch that would release the gate is ignored. Urgent (voice) periods keep their own rule.
"prefill_cadence" in the verify-cap control file (VLLM_VERIFY_CAP_CONTROL) overrides N at
runtime; it is re-read about once a second.

Enable with --scheduling-policy priority
            --scheduler-cls vllm.v1.core.sched.voice_first.VoiceFirstAsyncScheduler
"""
import json
import os
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler

logger = init_logger(__name__)


class VoiceFirstAsyncScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._vf_steps = 0          # steps in the current urgent period
        self._vf_held_steps = 0     # of those, steps that held someone's prefill
        self._vf_held_reqs: set[str] = set()
        self._cadence = max(1, int(os.environ.get("VLLM_PREFILL_CADENCE", "1")))
        self._control = os.environ.get("VLLM_VERIFY_CAP_CONTROL") or None
        self._control_mtime = None
        self._control_next = 0.0
        logger.info("voice-first scheduling on (policy=%s, prefill cadence %d)",
                    self.policy.value, self._cadence)

    def _poll_control(self) -> None:
        now = time.monotonic()
        if self._control is None or now < self._control_next:
            return
        self._control_next = now + 1.0
        try:
            mtime = os.stat(self._control).st_mtime
            if mtime == self._control_mtime:
                return
            self._control_mtime = mtime
            n = json.loads(open(self._control).read()).get("prefill_cadence")
        except (OSError, ValueError):
            return
        if isinstance(n, int) and n >= 1 and n != self._cadence:
            logger.info("voice-first: prefill cadence %d -> %d (control file)", self._cadence, n)
            self._cadence = n

    def _urgent_active(self) -> bool:
        return any(r.priority < 0 for q in (self.running, self.waiting, self.skipped_waiting) for r in q)

    def schedule(self, throttle_prefills: bool = False):
        self._poll_control()
        if not self._urgent_active():
            if self._vf_steps:
                if self._vf_held_steps:
                    logger.info("voice-first: urgent period ended after %d steps; held %d requests' "
                                "prefills in %d steps", self._vf_steps, len(self._vf_held_reqs),
                                self._vf_held_steps)
                self._vf_steps = self._vf_held_steps = 0
                self._vf_held_reqs.clear()
            if self._cadence > 1:
                # schedule() increments current_step first; prefills run on steps that
                # are multiples of the cadence. The capacity latch would release the
                # gate whenever anything is waiting, so it is cleared every step.
                self.prefill_capacity_bound = False
                throttle_prefills = throttle_prefills or (self.current_step + 1) % self._cadence != 0
            return super().schedule(throttle_prefills)

        # Running prefill chunks: skip them this step through the eligibility gate the
        # V2 runner uses for pipeline microbatching (checked before any budget is spent).
        # schedule() increments current_step first, so +2 keeps them ineligible.
        held_running = [(r, r.next_decode_eligible_step) for r in self.running
                        if r.priority >= 0 and r.is_prefill_chunk]
        for r, _ in held_running:
            r.next_decode_eligible_step = self.current_step + 2
        # Waiting requests all need prefill; keep only urgent ones visible this step.
        held_waiting = [r for r in self.waiting if r.priority >= 0]
        held_skipped = [r for r in self.skipped_waiting if r.priority >= 0]
        if held_waiting:
            self.waiting.remove_requests(held_waiting)
        if held_skipped:
            self.skipped_waiting.remove_requests(held_skipped)
        self._vf_steps += 1
        if held_running or held_waiting or held_skipped:
            self._vf_held_steps += 1
            self._vf_held_reqs.update(r.request_id for r in (*[r for r, _ in held_running],
                                                              *held_waiting, *held_skipped))
        try:
            return super().schedule(throttle_prefills)
        finally:
            for r, step in held_running:
                r.next_decode_eligible_step = step
            for r in held_waiting:
                self.waiting.add_request(r)
            for r in held_skipped:
                self.skipped_waiting.add_request(r)
