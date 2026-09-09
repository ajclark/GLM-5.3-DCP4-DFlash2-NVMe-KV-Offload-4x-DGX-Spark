#!/usr/bin/env python3
"""Offline architecture estimates and exact ancestor-closed tree selection.

This never loads a model or contacts a server. Synthetic probabilities/costs
are explicitly labeled. Tree-shape costs cannot be inferred from chain costs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Node:
    name: str
    parent: str | None
    token: int
    conditional: float


def validate_tree(nodes):
    """Sibling tokens are mutually exclusive; leftover probability may abstain."""
    if len(nodes) > 128:
        raise ValueError("offline screen is limited to 128 candidate nodes")
    by_name = {}
    for node in nodes:
        if (not isinstance(node.name, str) or not node.name or node.name in by_name
                or (node.parent is not None and not isinstance(node.parent, str))
                or type(node.token) is not int or node.token < 0
                or type(node.conditional) not in (int, float)
                or not math.isfinite(node.conditional) or not 0 <= node.conditional <= 1):
            raise ValueError("invalid node identity, token or conditional probability")
        by_name[node.name] = node
    children = {name: [] for name in [None, *by_name]}
    for node in nodes:
        if node.parent not in children:
            raise ValueError("unknown parent")
        children[node.parent].append(node.name)
    for siblings in children.values():
        siblings.sort()
        if len({by_name[name].token for name in siblings}) != len(siblings):
            raise ValueError("sibling tokens must be distinct mutually exclusive outcomes")
        if sum((Fraction(str(by_name[name].conditional)) for name in siblings), Fraction()) > 1:
            raise ValueError("sibling conditional probabilities must sum to at most one")
    visiting, reach, order = set(), {}, []
    def visit(name):
        if name in visiting:
            raise ValueError("cycle in candidate tree")
        if name in reach:
            return reach[name]
        visiting.add(name)
        node = by_name[name]
        parent = visit(node.parent) if node.parent is not None else Fraction(1)
        reach[name] = parent * Fraction(str(node.conditional))
        visiting.remove(name)
        return reach[name]
    for name in by_name:
        visit(name)
    def preorder(parent):
        for name in children[parent]:
            order.append(name)
            preorder(name)
    preorder(None)
    return by_name, children, reach, order


def shape_key(selected, by_name, order):
    """Canonical verified topology: parent indices including virtual anchor 0."""
    chosen = set(selected)
    if any(name not in by_name or (by_name[name].parent is not None
                                  and by_name[name].parent not in chosen) for name in chosen):
        raise ValueError("a verified shape must be ancestor closed")
    positions = {name: i + 1 for i, name in enumerate(name for name in order if name in chosen)}
    return json.dumps([positions.get(by_name[name].parent, 0)
                       for name in order if name in chosen], separators=(",", ":"))


def tree_frontier(nodes, budget=7):
    """Exact best expected emissions for each node count, not best latency.

    A virtual anchor costs no candidate node and contributes one ordinary
    target token. Decimal inputs use exact Fraction arithmetic internally.
    """
    if type(budget) is not int or not 0 <= budget <= 128:
        raise ValueError("node budget must be an integer between zero and 128")
    by_name, children, reach, order = validate_tree(nodes)
    limit = min(budget, len(nodes))
    def merge(left, right):
        result = {}
        for a, (value_a, selected_a) in left.items():
            for b, (value_b, selected_b) in right.items():
                if a + b > limit:
                    continue
                value = value_a + value_b
                selected = tuple(sorted(selected_a + selected_b))
                old = result.get(a + b)
                if old is None or value > old[0] or (value == old[0] and selected < old[1]):
                    result[a + b] = value, selected
        return result
    def subtree(name):
        result = {1: (reach[name], (name,))} if limit else {}
        for child in children[name]:
            result = merge(result, {0: (Fraction(), ()), **subtree(child)})
        return result
    result = {0: (Fraction(), ())}
    for child in children[None]:
        result = merge(result, {0: (Fraction(), ()), **subtree(child)})
    return [{"nodes": size, "selected": list(selected),
             "expected_emitted": float(1 + benefit),
             "exact_expected_emitted": str(1 + benefit),
             "shape": shape_key(selected, by_name, order)}
            for size, (benefit, selected) in sorted(result.items())]


def rank_frontier(frontier, costs=None):
    if costs is None:
        return {"status": "tree_costs_unmeasured", "ranked": [],
                "missing_shapes": [plan["shape"] for plan in frontier]}
    if (costs.get("kind") != "tree_shape"
            or costs.get("provenance") not in ("measured", "synthetic")
            or not costs.get("runtime_id") or type(costs.get("context_tokens")) is not int
            or costs["context_tokens"] < 0):
        raise ValueError("need independently sourced tree-shape costs and explicit provenance/context")
    values = costs.get("cycle_ms_by_shape", {})
    if (not isinstance(values, dict) or any(type(v) not in (int, float)
            or not math.isfinite(v) or v <= 0 for v in values.values())):
        raise ValueError("invalid tree-shape cycle costs")
    ranked, missing = [], []
    for plan in frontier:
        if plan["shape"] not in values:
            missing.append(plan["shape"])
            continue
        cycle = values[plan["shape"]]
        ranked.append({**plan, "cycle_ms": cycle,
                       "modelled_expected_tokens_per_ms": plan["expected_emitted"] / cycle})
    ranked.sort(key=lambda plan: (-plan["modelled_expected_tokens_per_ms"], plan["nodes"]))
    return {"status": "synthetic_scenario" if costs["provenance"] == "synthetic" else "supplied_measured_costs_model",
            "ranked": ranked, "missing_shapes": missing,
            "warning": "Ranks only the emission-optimal node-count frontier using supplied topology costs; not a global latency optimum, rollout, measured tok/sec or energy result."}


def selector_memory(config, lora_rank=16):
    if type(lora_rank) is not int or lora_rank < 1:
        raise ValueError("LoRA rank must be a positive integer")
    vocab, hidden = config["vocab_size"], config["hidden_size"]
    rank = config["dflash_config"]["selector_rank"]
    codebooks, projection = 2 * vocab * rank, hidden * rank
    parameters = codebooks + projection
    lora = (hidden + rank) * lora_rank
    # Explicit mixed-precision Adam scenario: BF16 weight/gradient, FP32 master
    # and two FP32 moments. Activations/workspace and frozen drafter excluded.
    state_bytes = 2 + 2 + 4 + 4 + 4
    return {"codebook_parameters": codebooks, "projection_parameters": projection,
            "selector_parameters": parameters, "bf16_selector_weight_bytes": parameters * 2,
            "adam_parameter_state_bytes_per_trainable_parameter": state_bytes,
            "full_selector_adam_parameter_state_bytes": parameters * state_bytes,
            "projection_only_trainable_state_bytes": projection * state_bytes,
            "projection_only_with_frozen_codebooks_bytes": projection * state_bytes + codebooks * 2,
            "lora_rank": lora_rank, "lora_trainable_parameters": lora,
            "lora_adam_parameter_state_bytes": lora * state_bytes,
            "cached_hidden_bytes_per_block": (config["dflash_config"]["block_size"] - 1) * hidden * 2,
            "full_training_peak_bytes": None,
            "warning": "Parameter-state accounting is an explicit optimizer/dtype scenario, not peak training memory. Add frozen draft weights, activations, temporary gradients, data, communication and allocator workspace. No Spark training is implied."}


def draft_window_memory(config, heads=(2, 4, 8), block_size=64):
    layers, dim, window = config["num_hidden_layers"], config["head_dim"], config["sliding_window"]
    block = config["dflash_config"]["block_size"]
    if type(block_size) is not int or block_size < 1 or any(type(h) is not int or h < 1 for h in heads):
        raise ValueError("invalid allocation block or per-rank head scenario")
    rounded = ((window + block + block_size - 1) // block_size) * block_size
    result = []
    for count in heads:
        values_per_token = 2 * layers * count * dim
        result.append({"assumed_kv_heads_per_rank": count,
            "bf16_window_payload_bytes": window * values_per_token * 2,
            "window_plus_query_block_rounded_tokens": rounded,
            "bf16_rounded_payload_bytes": rounded * values_per_token * 2,
            "ideal_int8_rounded_payload_bytes": rounded * values_per_token,
            "ideal_payload_saving_bytes": rounded * values_per_token})
    return {"scenarios": result,
        "warning": "Actual per-rank heads and live allocator occupancy must be inspected. Int8 excludes scales/metadata/conversion workspace. Payload reduction does not imply RSS reduction when the configured KV pool stays fixed."}


def longer_blocks(config, candidates=(8, 12, 16)):
    trained = config["dflash_config"]["block_size"]
    topk = config["dflash_config"]["selector_top_k"]
    result = []
    for block in candidates:
        if type(block) is not int or block < 2:
            raise ValueError("block must include anchor and at least one proposal")
        result.append({"block": block, "proposals": block - 1,
            "matches_loaded_training_geometry": block == trained,
            "selected_path_fp32_bytes": (block - 1) * topk * 4,
            "all_parent_edges_fp32_bytes": (block - 1) * topk * topk * 4,
            "needs_new_checkpoint_validation": block != trained,
            "target_graph_rows_if_full_chain": block})
    return result


def copy_opportunity(report):
    if report.get("screening_only") is not True:
        raise ValueError("expect an explicitly trajectory-only copy report")
    rows = []
    for row in report["runs"]:
        count, hits = row["boundaries"], row["copy_matches"]
        accepted = row["accepted_prefixes_on_baseline"]
        if (type(count) is not int or count < 1 or type(hits) is not int or not 0 <= hits <= count
                or len(accepted) != hits or any(type(a) is not int or not 0 <= a <= 7 for a in accepted)):
            raise ValueError("inconsistent baseline-trajectory opportunity counts")
        rows.append({"case": row["case"], "boundaries": count, "hits": hits,
            "hit_fraction": hits / count,
            "accepted_when_matched": sum(accepted) / hits if hits else None,
            "copied_tokens_per_baseline_boundary": sum(accepted) / count})
    return {"rows": rows, "warning": "Observed continuation agreement on one baseline trajectory; not closed-loop acceptance, throughput or power."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "tests/fixtures/spec_k0/draft-config.json")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tree", type=Path, help="JSON list with name,parent,token,conditional")
    group.add_argument("--demo", action="store_true", help="explicitly synthetic tree and costs")
    parser.add_argument("--costs", type=Path, help="independent tree-shape cost/provenance JSON")
    parser.add_argument("--budget", type=int, default=7)
    parser.add_argument("--copy-report", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    output = {"status": "offline_architecture_screen", "selector": selector_memory(config),
              "draft_memory": draft_window_memory(config), "longer_blocks": longer_blocks(config)}
    if args.copy_report:
        output["exact_copy"] = copy_opportunity(json.loads(args.copy_report.read_text()))
    if args.demo or args.tree:
        if args.demo:
            nodes = [Node("a",None,10,.6),Node("b",None,11,.3),
                     Node("aa","a",12,.8),Node("ab","a",13,.1),Node("ba","b",14,.8)]
            output["synthetic_probabilities"] = True
        else:
            nodes = [Node(**row) for row in json.loads(args.tree.read_text())]
        frontier = tree_frontier(nodes,args.budget)
        costs = json.loads(args.costs.read_text()) if args.costs else None
        if args.demo and costs is None:
            costs = {"kind":"tree_shape","provenance":"synthetic","runtime_id":"synthetic",
                "context_tokens":100000,"cycle_ms_by_shape":{
                    plan["shape"]:100+15*plan["nodes"] for plan in frontier}}
        output["tree"] = {"frontier":frontier,**rank_frontier(frontier,costs)}
    print(json.dumps(output,indent=2,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
