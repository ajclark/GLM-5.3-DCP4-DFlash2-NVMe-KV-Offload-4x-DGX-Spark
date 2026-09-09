#!/usr/bin/env python3
"""Offline DFlash input-geometry and recorded-acceptance diagnostics.

No model, network, CUDA, KV allocation, or prompt text is needed. A snapshot
contains the compact integer arrays passed to prepare_dflash_inputs; see
docs/research/LONG-CONTEXT-DIAGNOSTIC.md for the schema and its limits.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def audit_geometry(snapshot: dict) -> dict:
    """Validate a replicated draft group's absolute-position address space.

    A sliding window can free old physical pages, but its logical table must
    still address the live absolute positions. This audit deliberately rejects
    an out-of-row index instead of emulating the runtime's silent clamp.
    Block ID 0 is not rejected: it can be a valid/null page under the allocator.
    """
    positions = snapshot["target_positions"]
    starts = snapshot["target_query_start_loc"]
    rejected = snapshot["num_rejected"]
    tables = snapshot["block_table"]
    block_size = snapshot["block_size"]
    query_width = snapshot.get("num_query_per_req", 8)
    nreq = len(starts) - 1
    if block_size <= 0 or query_width <= 0:
        raise ValueError("block_size and num_query_per_req must be positive")
    if nreq <= 0 or len(rejected) != nreq or len(tables) != nreq:
        raise ValueError("request-array lengths disagree")
    if starts[0] != 0 or starts[-1] != len(positions):
        raise ValueError("query_start_loc does not cover target_positions")
    widths = {len(row) for row in tables}
    if len(widths) != 1 or 0 in widths:
        raise ValueError("block_table must have equally sized nonempty rows")

    issues = []
    expected = {k: [] for k in (
        "context_positions", "context_slots", "query_positions", "query_slots",
        "sample_positions", "seq_lens",
    )}
    requests = []
    for req in range(nreq):
        start, end = starts[req: req + 2]
        reject = rejected[req]
        if end <= start or not 0 <= reject < end - start:
            raise ValueError(f"request {req}: empty valid context or invalid rejection")
        context = positions[start:end]
        if any(p < 0 for p in context):
            raise ValueError(f"request {req}: negative absolute position")
        if any(b != a + 1 for a, b in zip(context, context[1:])):
            raise ValueError(f"request {req}: noncontiguous scheduled target positions")
        last_valid = positions[end - reject - 1]
        query = list(range(last_valid + 1, last_valid + 1 + query_width))
        table = tables[req]

        def slots(values, kind):
            result = []
            for pos in values:
                column, offset = divmod(pos, block_size)
                if column >= len(table):
                    issues.append({
                        "kind": "table_row_overflow", "request": req,
                        "array": kind, "position": pos, "column": column,
                        "row_columns": len(table),
                        "runtime_clamped_slot": table[-1] * block_size + offset,
                    })
                    result.append(None)
                else:
                    result.append(table[column] * block_size + offset)
            return result

        expected["context_positions"].extend(context)
        expected["context_slots"].extend(slots(context, "context"))
        expected["query_positions"].extend(query)
        expected["query_slots"].extend(slots(query, "query"))
        expected["sample_positions"].extend(query[1:])
        expected["seq_lens"].append(query[-1] + 1)
        requests.append({
            "request": req, "last_valid_position": last_valid,
            "query_position_range": [query[0], query[-1]],
            "rejected_context_rows": reject,
            "required_columns": max(context[-1], query[-1]) // block_size + 1,
            "row_columns": len(table),
            "addressable_positions": len(table) * block_size,
        })
    for name, values in snapshot.get("observed", {}).items():
        if name not in expected:
            raise ValueError(f"unknown observed array {name}")
        if values != expected[name]:
            issues.append({"kind": "output_mismatch", "array": name})
    return {"ok": not issues, "requests": requests, "issues": issues,
            "expected": expected}


def _metric(metrics, name, position=None):
    values = []
    for key, value in metrics.items():
        if key.split("{", 1)[0] != name:
            continue
        if position is not None:
            match = re.search(r'position="(\d+)"', key)
            if match is None or int(match.group(1)) != position:
                continue
        values.append(value)
    return sum(values) if values else None


def summarize_records(root: Path) -> list[dict]:
    """Read fixed-seven R4 records without exposing prompt/output contents.

    Prometheus deltas include terminal-boundary accounting and are not the
    controller's eligible nonterminal calibration counts. Keep them separate.
    """
    rows = []
    for path in sorted(root.glob("repo*k-fixed-r4/*-fixed-k7.json")):
        record = json.loads(path.read_text())
        if record.get("cap") != 7:
            continue
        metrics = record.get("spec_metric_delta", {})
        drafts = _metric(metrics, "vllm:spec_decode_num_drafts_total")
        first = _metric(metrics, "vllm:spec_decode_num_accepted_tokens_per_pos_total", 0)
        second = _metric(metrics, "vllm:spec_decode_num_accepted_tokens_per_pos_total", 1)
        accepted = _metric(metrics, "vllm:spec_decode_num_accepted_tokens_total")
        rows.append({
            "source": str(path), "case": record.get("case"),
            "prompt_tokens": record.get("usage", {}).get("prompt_tokens"),
            "decode_tps": record.get("decode_tps"), "ttft_s": record.get("ttft"),
            "draft_cycles_metric": drafts, "accepted_first_metric": first,
            "accepted_second_metric": second, "accepted_tokens_metric": accepted,
            "first_acceptance_fraction_metric": first / drafts
            if drafts and first is not None else None,
            "caveat": "one repeat; Prometheus counters, not eligible calibration cycles",
        })
    return sorted(rows, key=lambda row: (row["prompt_tokens"] or 0, row["case"] or ""))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/adaptive-spec"))
    parser.add_argument("--snapshot", type=Path,
                        help="compact integer-array snapshot to validate offline")
    args = parser.parse_args(argv)
    if args.snapshot:
        output = audit_geometry(json.loads(args.snapshot.read_text()))
        exit_code = 0 if output["ok"] else 2
    else:
        output = {"records": summarize_records(args.results)}
        exit_code = 0
    print(json.dumps(output, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
