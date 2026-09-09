#!/usr/bin/env python3
"""Offline, causality-aware screening of DFlash2 selected-path confidence.

This consumes compact joined proposal/verification records, not production
traces that never captured scores. --demo uses explicitly synthetic records.
Same-boundary utility is not a rollout or a measured token-rate improvement.
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

CAPS = (1, 3, 5, 7)
BINS = (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)


def path_features(scores, candidate_ids=None, selected_tokens=None):
    """Reduce each chosen-predecessor score row, with no full-vocab operation.

    pmax and entropy describe only the candidate set. They are not calibrated
    probabilities that a greedy target will agree. Invalid rows abstain.
    """
    if len(scores) != 7:
        raise ValueError("expected seven selected-predecessor score rows")
    if (candidate_ids is None) != (selected_tokens is None):
        raise ValueError("candidate IDs and selected tokens must be supplied together")
    if candidate_ids is not None and (len(candidate_ids) != 7 or len(selected_tokens) != 7):
        raise ValueError("candidate/token row counts disagree")
    features = []
    for j, row in enumerate(scores):
        if len(row) != 16:
            raise ValueError("expected the loaded checkpoint's 16 candidates")
        values = [float(v) for v in row]
        if any(not math.isfinite(v) for v in values):
            features.append(None)
            continue
        chosen = max(range(16), key=values.__getitem__)
        if candidate_ids is not None:
            ids = candidate_ids[j]
            if len(ids) != 16 or len(set(ids)) != 16 or selected_tokens[j] not in ids:
                raise ValueError("selected token is not uniquely mapped to a candidate")
            chosen = ids.index(selected_tokens[j])
            if values[chosen] != max(values):
                raise ValueError("selected token is not a greedy maximum of this score row")
        maximum = values[chosen]
        weights = [math.exp(v - maximum) for v in values]
        total = sum(weights)
        probs = [v / total for v in weights]
        entropy = -sum(p * math.log(p) for p in probs if p) / math.log(16)
        features.append({
            "margin": maximum - max(v for i, v in enumerate(values) if i != chosen),
            "pmax": 1 / total, "entropy": entropy,
        })
    return features


def conditional_labels(k, accepted):
    """Only positions reached by the greedy verifier are conditional trials."""
    if k not in CAPS or not 0 <= accepted <= k:
        raise ValueError("invalid scheduled cap or accepted prefix")
    return [int(j < accepted) if j < k and j <= accepted else None for j in range(7)]


def prefix_labels(k, accepted):
    # After a rejection, all longer prefix-survival events are false. After an
    # all-accepted short cap, the unverified tail remains unknown.
    conditional_labels(k, accepted)
    return [int(j < accepted) if j < k or accepted < k else None for j in range(7)]


def owner(row):
    return tuple(row[k] for k in ("runtime_id", "run_id", "request", "epoch"))


def validate_records(records):
    seen = set()
    owner_cases = {}
    for row in records:
        key = (*owner(row), row["proposal_id"])
        if key in seen:
            raise ValueError("duplicate proposal identity")
        seen.add(key)
        identity = owner(row)
        if identity in owner_cases and owner_cases[identity] != row["case"]:
            raise ValueError("one request incarnation cannot cross prompt split groups")
        owner_cases[identity] = row["case"]
        if (row["feature_proposal_id"] != row["proposal_id"]
                or row["verified_proposal_id"] != row["proposal_id"]
                or row["feature_anchor"] != row["anchor"]
                or row["verified_anchor"] != row["anchor"]):
            raise ValueError("feature/verification ownership or anchor mismatch")
        if row["feedback_available_ns"] < row["decision_ns"]:
            raise ValueError("feedback cannot precede its scheduling decision")
        conditional_labels(row["scheduled_k"], row["accepted"])
        costs = row["costs_ms"]
        if set(costs) != set(map(str, CAPS)) or any(
                not math.isfinite(v) or v <= 0 for v in costs.values()):
            raise ValueError("need all four positive finite measured cap costs")
        if row.get("realized_scores") is not None:
            path_features(row["realized_scores"], row.get("candidate_ids"),
                          row.get("selected_tokens"))
    if len({r["runtime_id"] for r in records}) > 1:
        raise ValueError("fit repaired and original runtimes separately")


def select_feature(row, pool, *, causal=True, max_age=2):
    """Choose an immutable packet received by this scheduler's deadline.

    The timestamp is scheduler-local receipt time, not a cross-host wall clock
    or a CUDA event timestamp. Request-slot reuse is rejected by epoch identity.
    """
    candidates = []
    for source in pool:
        if owner(source) != owner(row) or source.get("realized_scores") is None:
            continue
        age = row["proposal_id"] - source["proposal_id"]
        if not 0 <= age <= max_age or source["anchor"] > row["anchor"]:
            continue
        if causal:
            ready = source.get("feature_available_ns")
            if ready is None or ready > row["decision_ns"]:
                continue
        elif source["proposal_id"] != row["proposal_id"]:
            continue
        candidates.append(source)
    if not candidates:
        return None
    source = max(candidates, key=lambda r: r["proposal_id"])
    return {"proposal_id": source["proposal_id"],
            "age": row["proposal_id"] - source["proposal_id"],
            "values": path_features(source["realized_scores"], source.get("candidate_ids"),
                                    source.get("selected_tokens"))}


def context_band(row):
    return bisect.bisect_right((4096, 32768, 90112, 131072), row["anchor"])


def _cell(row, packet, position):
    value = packet["values"][position]
    if value is None:
        return None
    return (context_band(row), packet["age"], position,
            bisect.bisect_right(BINS, value["pmax"]))


def fit(records, packets):
    # Small conditional-risk lookup, not a trained draft model. Each cell is
    # [successes, trials]; unknown tails never become failed observations.
    cells = defaultdict(lambda: [0, 0])
    priors = defaultdict(lambda: [0, 0])
    for row, packet in zip(records, packets):
        for j, label in enumerate(conditional_labels(row["scheduled_k"], row["accepted"])):
            if label is None:
                continue
            prior = priors[(context_band(row), j)]
            prior[0] += label
            prior[1] += 1
            key = _cell(row, packet, j) if packet else None
            if key is not None:
                cells[key][0] += label
                cells[key][1] += 1
    return dict(cells), dict(priors)


def probabilities(row, packet, model, history):
    cells, priors = model
    result = []
    visible = sorted((old for old in history if owner(old) == owner(row)
                      and old["proposal_id"] < row["proposal_id"]
                      and old["feedback_available_ns"] <= row["decision_ns"]),
                     key=lambda old: old["feedback_available_ns"])
    for j in range(7):
        success, count = priors.get((context_band(row), j), (0, 0))
        prior = (success + .5) / (count + 1)
        recent_success = recent_count = 0.0
        for old in visible:
            recent_success *= .97
            recent_count *= .97
            label = conditional_labels(old["scheduled_k"], old["accepted"])[j]
            if label is not None:
                recent_success += label
                recent_count += 1
        probability = (recent_success + 2 * prior) / (recent_count + 2)
        key = _cell(row, packet, j) if packet else None
        if key in cells:
            success, count = cells[key]
            cell_probability = (success + .5) / (count + 1)
            weight = min(count, 32)
            probability = (weight * cell_probability + 8 * probability) / (weight + 8)
        result.append(probability)
    return result


def survival(probabilities):
    result, product = [], 1.0
    for p in probabilities:
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("invalid conditional probability")
        product *= p
        result.append(product)
    return result


def choose_cap(probabilities, costs):
    cumulative = survival(probabilities)
    return max(CAPS, key=lambda k: ((1 + sum(cumulative[:k])) / costs[str(k)], k))


def screen(records, max_age=2):
    validate_records(records)
    rows = [r for r in records if r.get("eligible", False) and not r.get("terminal", True)]
    cases = sorted({r["case"] for r in rows})
    if len(cases) < 2:
        raise ValueError("leave-one-prompt-out fitting requires at least two cases")
    packets = {causal: [select_feature(r, rows, causal=causal, max_age=max_age) for r in rows]
               for causal in (True, False)}
    per_prompt = []
    for case in cases:
        train_indices = [i for i, r in enumerate(rows) if r["case"] != case]
        train = [rows[i] for i in train_indices]
        models = {causal: fit(train, [packets[causal][i] for i in train_indices])
                  for causal in (True, False)}
        losses = defaultdict(list)
        emitted, elapsed, choices = Counter(), Counter(), defaultdict(Counter)
        available_ages = Counter()
        evaluated = 0
        for i, row in enumerate(rows):
            if row["case"] != case:
                continue
            history = [r for r in rows if r["case"] == case]
            if packets[True][i]:
                available_ages[packets[True][i]["age"]] += 1
            modes = {
                "history_screen": probabilities(row, None, models[True], history),
                "available_confidence": probabilities(row, packets[True][i], models[True], history),
                "same_block_noncausal_diagnostic": probabilities(row, packets[False][i], models[False], history),
            }
            for mode, p in modes.items():
                for pred, label in zip(survival(p), prefix_labels(row["scheduled_k"], row["accepted"])):
                    if label is not None:
                        losses[mode].append((pred - label) ** 2)
            # Only full-seven records permit fair comparisons over every cap.
            if row["scheduled_k"] != 7:
                continue
            evaluated += 1
            caps = {mode: choose_cap(p, row["costs_ms"]) for mode, p in modes.items()}
            caps["fixed7"] = 7
            for mode, cap in caps.items():
                choices[mode][cap] += 1
                emitted[mode] += 1 + min(row["accepted"], cap)
                elapsed[mode] += row["costs_ms"][str(cap)]
        utility = {mode: emitted[mode] / elapsed[mode] for mode in emitted}
        per_prompt.append({"case": case, "full7_boundaries": evaluated,
            "feature_age_counts": dict(available_ages),
            "prefix_brier": {k: sum(v) / len(v) for k, v in losses.items()},
            "caps": {k: dict(v) for k, v in choices.items()},
            "same_boundary_utility_per_ms": utility,
            "available_over_history_utility_ratio": utility["available_confidence"] / utility["history_screen"]
                if utility else None})
    return {"status": "offline_screen_only", "cases": len(cases), "records": len(rows),
        "records_with_scores": sum(r.get("realized_scores") is not None for r in rows),
        "causally_available_records": sum(p is not None for p in packets[True]),
        "records_with_usable_available_features": sum(
            p is not None and any(v is not None for v in p["values"]) for p in packets[True]),
        "method": "Leave one prompt/case out, including all its repeats. Conditional-risk bins by context, feature age, position and candidate pmax; bounded shrinkage toward causal acceptance history.",
        "limitations": "Not a closed-loop rollout or V4 policy replica: warmup, probes and hysteresis are omitted. Same-block diagnostic ignores availability intentionally. No measured speedup or power claim.",
        "decision": "Collect and validate bounded shadow features before a live policy experiment; no production promotion.",
        "per_prompt": per_prompt}


def demo_records():
    result = []
    for case in ("code_a", "code_b", "prose_a", "prose_b"):
        for i in range(24):
            high = (i // 4) % 2 == 0
            anchor = 1000 + i * 8
            result.append(dict(runtime_id="synthetic", run_id="synthetic", case=case,
                request=case, epoch=1, proposal_id=i, feature_proposal_id=i,
                verified_proposal_id=i, anchor=anchor, feature_anchor=anchor,
                verified_anchor=anchor, decision_ns=i * 100, feature_available_ns=i * 100 + 150,
                feedback_available_ns=i * 100 + 180, scheduled_k=7,
                accepted=6 if high else 0, eligible=True, terminal=False,
                realized_scores=[([5.0] + [0.0] * 15) if high else [0.0] * 16 for _ in range(7)],
                costs_ms={"1": 100, "3": 120, "5": 140, "7": 160}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--trace", type=Path, help="joined proposal/verification JSONL; see feasibility doc")
    sources.add_argument("--demo", action="store_true", help="synthetic causality smoke test, not a benchmark")
    parser.add_argument("--max-age", type=int, default=2)
    args = parser.parse_args(argv)
    if args.max_age < 0:
        parser.error("max-age must be nonnegative")
    if args.demo:
        result = {"synthetic": True, **screen(demo_records(), args.max_age)}
    elif args.trace:
        records = [json.loads(line) for line in args.trace.read_text().splitlines() if line.strip()]
        result = screen(records, args.max_age)
    else:
        result = {"status": "no_feature_dataset_supplied",
                  "decision": "Historical acceptance-only traces cannot establish confidence quality. Use --trace with captured score packets or --demo for a synthetic smoke test."}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
