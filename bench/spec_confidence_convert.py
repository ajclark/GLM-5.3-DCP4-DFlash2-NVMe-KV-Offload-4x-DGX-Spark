#!/usr/bin/env python3
"""Convert genuine verify confidence packets into the causal offline format.

Runtime/run provenance and an explicit label-to-prompt case map are required.
No receipt/decision timestamps, score packets, or cap costs are synthesized.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def convert(rows, *, runtime_id, run_id, case_map):
    if not runtime_id or not run_id or not isinstance(case_map, dict):
        raise ValueError("explicit runtime, run and prompt case mapping required")
    output, counts, seen = [], Counter(), set()
    for row in rows:
        if row.get("event") != "verify":
            continue
        packet = row.get("confidence")
        if not packet:
            counts["no_packet"] += 1
            continue
        if packet.get("version") != 1 or not packet.get("valid"):
            counts["invalid_packet"] += 1
            continue
        if not row.get("learned") or not packet.get("learnable"):
            counts["not_learnable"] += 1
            continue
        decision, receipt = row.get("decision_ns"), row.get("receipt_ns")
        if (type(decision) is not int or type(receipt) is not int
                or decision < 0 or receipt < decision):
            raise ValueError("missing or invalid scheduler-local availability timestamps")
        k, accepted = row.get("scheduled_k"), row.get("accepted")
        if (type(k) is not int or k not in (1, 3, 5, 7)
                or type(accepted) is not int or not 0 <= accepted <= k
                or packet.get("verified_k") != k):
            raise ValueError("invalid acceptance join")
        anchor = packet.get("anchor")
        sample, positions = packet.get("sample_positions"), packet.get("target_positions")
        drafts, inputs = packet.get("draft_tokens"), packet.get("target_tokens")
        if (type(anchor) is not int or anchor < 0
                or sample != list(range(anchor + 1, anchor + 8))
                or positions != list(range(anchor, anchor + k + 1))
                or not isinstance(drafts, list) or len(drafts) != 7
                or not isinstance(inputs, list) or len(inputs) != k + 1
                or inputs[1:] != drafts[:k]):
            raise ValueError("invalid proposal/target anchor or token alignment")
        scores = packet.get("realized_scores")
        if (not isinstance(scores, list) or len(scores) != 7
                or any(not isinstance(values, list) or len(values) != 16
                       or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                       for values in scores)):
            raise ValueError("invalid realized score tensor")
        costs = row.get("costs_ms", {})
        if (set(map(str, costs)) != {"1", "3", "5", "7"}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in costs.values())):
            counts["missing_measured_costs"] += 1
            continue
        case = case_map.get(row.get("label"))
        if not isinstance(case, str) or not case:
            raise ValueError("explicit prompt case mapping missing for trace label")
        epoch, proposal = packet.get("epoch"), packet.get("proposal_id")
        if (type(epoch) is not int or epoch < 1 or type(proposal) is not int or proposal < 1
                or packet.get("proposal_age_steps") != 1):
            raise ValueError("invalid request incarnation or proposal age")
        key = (row["request"], epoch, proposal)
        if key in seen:
            raise ValueError("duplicate proposal packet")
        seen.add(key)
        output.append({
            "runtime_id": runtime_id, "run_id": run_id, "case": case,
            "request": row["request"], "epoch": epoch,
            "proposal_id": proposal, "feature_proposal_id": proposal,
            "verified_proposal_id": proposal, "anchor": anchor,
            "feature_anchor": sample[0] - 1, "verified_anchor": positions[0],
            "decision_ns": decision, "feature_available_ns": receipt,
            "feedback_available_ns": receipt, "scheduled_k": k, "accepted": accepted,
            "eligible": bool(row.get("eligible")), "terminal": bool(row.get("terminal")),
            "realized_scores": scores, "draft_tokens": drafts,
            "costs_ms": {str(key): value for key, value in costs.items()},
        })
    return output, {"converted": len(output), "excluded": dict(counts),
                    "availability": "Actual scheduler-local receipt; no same-decision feature availability assumed."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--case-map", type=Path, required=True,
                        help="JSON object mapping every experiment label to its prompt/case; repeats share one case")
    args = parser.parse_args(argv)
    with args.trace.open() as source:
        records, summary = convert((json.loads(line) for line in source if line.strip()),
            runtime_id=args.runtime_id, run_id=args.run_id,
            case_map=json.loads(args.case_map.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
