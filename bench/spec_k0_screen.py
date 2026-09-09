#!/usr/bin/env python3
"""Offline K=0 ownership model, memory budget and conditional break-even screen.

This does not change serving or launch a model. The default cost table is
historical, before the replicated-cache repair. True K=0 and resume timings
remain unknown unless supplied as explicitly labeled scenario assumptions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

CAPS = (1, 3, 5, 7)
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Step:
    owner: tuple[str, int]
    sequence: int
    intent: int
    cap: int
    action: str


@dataclass(frozen=True)
class Ack:
    owner: tuple[str, int]
    sequence: int
    intent: int
    drafting_ready: bool
    anchor: int
    reason: str | None = None


class ParkingModel:
    """CPU protocol model; values stand in for immutable combined-state rows.

    Calls to execute are ordered worker steps. An Ack must only be published
    after the corresponding ordered GPU work has been safely enqueued/completed
    according to the implementation's explicit event contract. This model is
    not a CUDA concurrency proof.
    """
    def __init__(self, request="r", epoch=0, computed=4096, strategy="eager",
                 window=2048, slack=8, chunk=128):
        if strategy not in ("eager", "deferred") or min(window, chunk) < 1 or slack < 0:
            raise ValueError("invalid parking strategy or retention geometry")
        self.owner = (request, epoch)
        self.computed = computed
        self.materialized = computed  # known-valid initial draft-context cache
        self.proposal_anchor = computed
        self.strategy, self.window, self.slack, self.chunk = strategy, window, slack, chunk
        self.intent = 0
        self.desired = "active"
        self.worker_mode = "active"
        self.scheduler_ready = True
        self.next_sequence = 0
        self.pending = []
        self.rows = {}
        self.closed = False
        self.query_passes = 0
        self.rebuilt_positions = []
        self.rebuild_chunks = []

    def park(self):
        if self.closed:
            raise ValueError("request closed")
        self.intent += 1
        self.desired = "parked"
        self.scheduler_ready = False

    def arm(self):
        if self.closed:
            raise ValueError("request closed")
        self.intent += 1
        self.desired = "arming"
        self.scheduler_ready = False

    def schedule(self, desired_cap=7):
        if self.closed or desired_cap not in CAPS:
            raise ValueError("closed request or invalid positive cap")
        if self.desired == "parked":
            cap, action = 0, "park"
        elif self.desired == "arming" or not self.scheduler_ready:
            cap, action = 0, "arm"
        else:
            cap, action = desired_cap, "active"
        step = Step(self.owner, self.next_sequence, self.intent, cap, action)
        self.next_sequence += 1
        self.pending.append(step)
        return step

    def _retain(self, start, values):
        for offset, value in enumerate(values):
            # Copy list-valued rows to model the prohibition on retaining views
            # into target buffers that will be reused by the next invocation.
            self.rows[start + offset] = tuple(value) if isinstance(value, list) else value
        lower = self.computed - self.window - self.slack
        self.rows = {p: value for p, value in self.rows.items() if p >= lower}

    def execute(self, step, accepted=0, combined_rows=None):
        if not self.pending or self.pending[0] != step:
            raise ValueError("worker must execute immutable steps in FIFO order")
        self.pending.pop(0)
        if self.closed:
            return Ack(step.owner, step.sequence, step.intent, False, self.computed, "cancelled")
        if step.owner != self.owner or type(accepted) is not int or not 0 <= accepted <= step.cap:
            raise ValueError("invalid ownership or accepted-prefix count")
        if step.cap and self.proposal_anchor != self.computed:
            raise ValueError("positive verification would consume a stale/missing proposal")
        start = self.computed
        if combined_rows is None:
            combined_rows = list(range(start, start + step.cap + 1))
        if len(combined_rows) != step.cap + 1:
            raise ValueError("need one combined row for each actual target input row")
        # Rejected speculative rows never enter the committed retention ring.
        valid_rows = combined_rows[:accepted + 1]
        self.computed += accepted + 1
        self.rebuilt_positions = []
        self.rebuild_chunks = []

        if step.action == "park" and self.strategy == "deferred":
            self._retain(start, valid_rows)
            self.worker_mode = "parked"
            self.proposal_anchor = None
            return Ack(self.owner, step.sequence, step.intent, False, self.computed)

        if step.action == "park":
            # Eager mode performs FC + context-KV maintenance, no draft queries.
            self.materialized = self.computed
            self.worker_mode = "parked"
            self.proposal_anchor = None
            return Ack(self.owner, step.sequence, step.intent, False, self.computed)

        if self.strategy == "deferred" and self.materialized != start:
            self._retain(start, valid_rows)
            live_start = max(0, self.computed - self.window)
            dirty_start = max(live_start, self.materialized if self.materialized is not None else 0)
            required = list(range(dirty_start, self.computed))
            if any(p not in self.rows for p in required):
                self.proposal_anchor = None
                self.worker_mode = "parked"
                return Ack(self.owner, step.sequence, step.intent, False, self.computed,
                           "missing committed context for re-entry")
            self.rebuilt_positions = required
            self.rebuild_chunks = [required[i:i + self.chunk] for i in range(0, len(required), self.chunk)]
        # ACTIVE and ARMED both maintain context and draft after EVERY target
        # step. Queued K0 steps after arming must refresh the proposal anchor.
        self.materialized = self.computed
        self.query_passes += 1
        self.proposal_anchor = self.computed
        self.worker_mode = "armed" if step.action == "arm" else "active"
        return Ack(self.owner, step.sequence, step.intent, True, self.computed)

    def receive(self, ack):
        if self.closed or ack.owner != self.owner or ack.intent != self.intent:
            return False
        if self.desired == "arming" and ack.drafting_ready:
            self.desired = "active"
            self.scheduler_ready = True
            return True
        return False

    def publish_prefix(self, end):
        if self.materialized is None or end > self.materialized:
            raise ValueError("draft prefix is not materialized; publishing it would poison cache reuse")

    def cancel(self):
        self.closed = True
        self.scheduler_ready = False
        self.proposal_anchor = None

    @property
    def retention_lease_releasable(self):
        # In a real worker, draining includes GPU events and pending DMA, not
        # merely returning the Python schedule objects.
        return self.closed and not self.pending


def slots_for_positions(positions, block_table, block_size):
    result = []
    for pos in positions:
        column, offset = divmod(pos, block_size)
        if pos < 0 or column >= len(block_table):
            raise ValueError("current replicated block table cannot address replay position")
        result.append(block_table[column] * block_size + offset)
    return result


def memory_budget(config, kv_heads_per_rank=2, chunk=128, slack=8):
    if kv_heads_per_rank < 1 or chunk < 1 or slack < 0:
        raise ValueError("invalid per-rank KV heads or retention geometry")
    width = config["hidden_size"]
    aux = len(config["dflash_config"]["target_layer_ids"])
    layers, head = config["num_hidden_layers"], config["head_dim"]
    window, element = config["sliding_window"], 2
    capacity = window + slack
    combined = capacity * width * element
    raw = combined * aux

    def scratch(tokens):
        normalized_context = tokens * width * element
        kv = tokens * layers * 2 * kv_heads_per_rank * head * element
        normalized_k = kv // 2
        positions = tokens * layers * 8
        # Conservative explicit chunk copy for a wrapped retention ring.
        return 2 * normalized_context + 2 * kv + normalized_k + positions

    return {"combined_bytes_per_token": width * element,
            "raw_aux_bytes_per_token": width * aux * element,
            "ring_tokens": capacity, "combined_ring_bytes": combined,
            "raw_aux_ring_bytes": raw,
            "assumed_actual_kv_heads_per_rank": kv_heads_per_rank,
            "chunk_tokens": chunk, "estimated_chunk_scratch_bytes": scratch(chunk),
            "estimated_ring_plus_chunk_bytes": combined + scratch(chunk),
            "estimated_ring_plus_full_window_scratch_bytes": combined + scratch(window),
            "budget_bytes": 64 * 1024 ** 2,
            "combined_chunk_fits_budget": combined + scratch(chunk) <= 64 * 1024 ** 2,
            "note": "Source-derived tensor sizes, not measured allocation/peak memory; excludes existing model weights, KV pool, graph capture and allocator overhead."}


def expected_emitted(prior, cap):
    if len(prior) != 7 or cap not in CAPS or any(not math.isfinite(p) or not 0 <= p <= 1 for p in prior):
        raise ValueError("invalid conditional prior or positive cap")
    product, total = 1.0, 1.0
    for p in prior[:cap]:
        product *= p
        total += product
    return total


def break_even(costs, prior, k0_ms=None, resume_ms=None, dwell_tokens=None):
    if set(costs) != set(map(str, CAPS)) or any(not math.isfinite(v) or v <= 0 for v in costs.values()):
        raise ValueError("need four finite positive measured cycle costs")
    per_token = {k: costs[str(k)] / expected_emitted(prior, k) for k in CAPS}
    best = min(CAPS, key=lambda k: (per_token[k], -k))
    result = {"modelled_best_positive_cap": best,
              "modelled_positive_ms_per_emitted_token": per_token[best],
              "positive_ms_per_token_by_cap": per_token,
              "strict_k0_ms_per_token_ceiling_before_resume": per_token[best],
              "true_k0_timing": "unmeasured", "resume_timing": "unmeasured"}
    if k0_ms is None or resume_ms is None:
        return result
    if not math.isfinite(k0_ms) or k0_ms <= 0 or not math.isfinite(resume_ms) or resume_ms < 0:
        raise ValueError("scenario K0 must be positive and resume cost nonnegative")
    delta = per_token[best] - k0_ms
    minimum = math.floor(resume_ms / delta) + 1 if delta > 0 else None
    result["scenario_assumptions_not_measurements"] = {
        "k0_complete_ms_per_token": k0_ms,
        "resume_and_transition_ms": resume_ms,
        "minimum_tokens_for_strict_time_saving": minimum,
    }
    if dwell_tokens is not None:
        if type(dwell_tokens) is not int or dwell_tokens <= 0:
            raise ValueError("dwell_tokens must be a positive integer")
        result["scenario_assumptions_not_measurements"].update(
            dwell_tokens=dwell_tokens,
            modelled_stay_ms=dwell_tokens * per_token[best],
            modelled_park_ms=dwell_tokens * k0_ms + resume_ms,
            time_saved_ms=dwell_tokens * delta - resume_ms,
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--costs", type=Path, default=ROOT / "results/adaptive-spec/adaptive-v2-20260908-r4/costs-v3-curve.json")
    parser.add_argument("--config", type=Path, default=ROOT / "tests/fixtures/spec_k0/draft-config.json")
    parser.add_argument("--kv-heads-per-rank", type=int, default=2)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--k0-ms", type=float)
    parser.add_argument("--resume-ms", type=float)
    parser.add_argument("--dwell-tokens", type=int)
    args = parser.parse_args(argv)
    curve = json.loads(args.costs.read_text())
    config = json.loads(args.config.read_text())
    points = curve.get("calibration_points", [curve])
    result = {"status": "offline_feasibility_only",
              "cost_source": str(args.costs),
              "cost_provenance_warning": "Default v3 points predate the replicated-cache repair. Recalibrate before making a repaired-runtime decision; custom input provenance remains the caller's responsibility.",
              "memory": memory_budget(config, args.kv_heads_per_rank, args.chunk),
              "points": [{"context": p.get("context"), **break_even(
                  p["cycle_ms"], p["conditional_acceptance_prior"],
                  args.k0_ms, args.resume_ms, args.dwell_tokens)} for p in points],
              "decision": "Prefer eager context-KV maintenance if a real M1/no-query-draft trial is justified. Defer deferred caching and promotion until repaired acceptance, true K0 timing and resume overhead are measured."}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
