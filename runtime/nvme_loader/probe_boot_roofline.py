#!/usr/bin/env python3
"""Offline boot accounting and explicit TP-layout hypothesis checks.

Uses saved evidence and synthetic tensors; does not read model payloads, contact
Sparks, or benchmark storage. This is not a generic placement planner. The TP
axes below are explicit hypotheses for the recorded expert projection examples.
"""
import argparse
import ast
import json
from pathlib import Path
import re


def page_bytes(ranges, page_size=4096):
    pages = set()
    for start, end in ranges:
        if end > start:
            pages.update(range(start // page_size, (end - 1) // page_size + 1))
    return len(pages) * page_size


def ranges_for_slice(rows, columns, item_size, axis, rank, tp):
    if axis == 0:
        count = rows // tp
        return [(rank * count * columns * item_size,
                 (rank + 1) * count * columns * item_size)]
    count = columns // tp
    return [(r * columns * item_size + rank * count * item_size,
             r * columns * item_size + (rank + 1) * count * item_size)
            for r in range(rows)]


def analyze(root, generation):
    import torch

    runs = json.loads((root / "stream-results.json").read_text())
    run = runs[generation]
    rank = run["ranks"]["spark-06c4.local"]
    target, draft = rank["streams"]
    model_seconds = float(re.search(r"([\d.]+) seconds", rank["model_loading"])[1])
    engine_seconds = float(re.search(r"took ([\d.]+) s", rank["engine_initialization"])[1])
    layout = json.loads((root / "checkpoint-layout.json").read_text())[0]
    tp = 4
    examples = []
    for projection, axis in (("gate_proj", 0), ("up_proj", 0), ("down_proj", 1)):
        example = layout["expert_examples"][projection + ".weight_packed"]
        rows, cols = example["shape"]
        assert example["dtype"] == "I32"
        assert (rows if axis == 0 else cols) % tp == 0
        tensor = torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols)
        count = tensor.shape[axis] // tp
        checks = []
        for rank_id in range(tp):
            old = tensor.t().contiguous().narrow(1 - axis, rank_id * count, count)
            new = tensor.narrow(axis, rank_id * count, count).t().contiguous()
            assert torch.equal(old, new), (projection, rank_id)
            ranges = ranges_for_slice(rows, cols, 4, axis, rank_id, tp)
            checks.append({
                "rank": rank_id,
                "logical_slice_bytes": sum(b - a for a, b in ranges),
                "uncoalesced_ranges": len(ranges),
                "physical_4k_bytes_tensor_aligned": page_bytes(ranges),
                "physical_4k_bytes_tensor_offset_123": page_bytes(
                    [(a + 123, b + 123) for a, b in ranges]),
                "transpose_then_slice_equals_slice_then_transpose": True,
                "old_transpose_storage_bytes": tensor.numel() * tensor.element_size(),
                "new_transpose_storage_bytes": new.untyped_storage().nbytes(),
            })
        examples.append({"projection": projection, "shape": [rows, cols],
                         "source_slice_axis_assumption": axis, "ranks": checks})
    source = target["bytes"]
    expert = layout["expert_named_bytes"]
    selective = expert * (2 / 3 / tp + 1 / 3) + source - expert + draft["bytes"]
    # Execute only these pure tensor helpers from the saved installed source,
    # without importing vLLM or running any model construction.
    marlin_path = root / "inventory/model_executor/layers/quantization/utils/marlin_utils.py"
    tree = ast.parse(marlin_path.read_text())
    functions = {"get_scale_perms", "marlin_permute_scales", "marlin_moe_permute_scales"}
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in functions]
    assert len(selected) == len(functions)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(marlin_path), "exec"), namespace)
    permutation_checks = []
    for experts in (1, 7, 256):
        for group_size, int8_activation in ((128, False), (-1, False), (128, True)):
            scales = torch.arange(experts * 8 * 128, dtype=torch.float32).reshape(experts, 8, 128)
            for noncontiguous in (False, True):
                sample = scales if not noncontiguous else scales.transpose(1, 2).contiguous().transpose(1, 2)
                original = namespace["marlin_moe_permute_scales"](sample, 1024, 128, group_size, int8_activation)
                perms = namespace["get_scale_perms"]()
                perm = perms[0 if group_size != -1 and not int8_activation else 1]
                batched = sample.reshape(experts, -1, len(perm))[:, :, perm].reshape(sample.shape).contiguous()
                assert torch.equal(original, batched)
                permutation_checks.append({"experts": experts, "group_size": group_size,
                                           "int8_activation": int8_activation,
                                           "noncontiguous": noncontiguous, "equal": True})
    return {
        "generation": generation,
        "status": "offline arithmetic and synthetic int32 equality; not a new activation",
        "baseline_seconds": {
            "healthy": run["health_seconds"], "target_stream": target["seconds"],
            "draft_stream": draft["seconds"],
            "remaining_model_load": model_seconds - target["seconds"] - draft["seconds"],
            "engine_initialization": engine_seconds,
            "outside_reported_model_and_engine": run["health_seconds"] - model_seconds - engine_seconds,
            "counterfactual_zero_target_stream": run["health_seconds"] - target["seconds"],
        },
        "ideal_tp4_target_bytes_per_rank": {
            "full_local_reads": source,
            "cooperative_reads": source / tp,
            "full_tensor_broadcast_receive": source * (tp - 1) / tp,
            "slice_scatter_send_and_receive_each": source * (tp - 1) / tp ** 2,
            "neighbor_forwarding_each_direction_balanced_shortest_paths": source / 8,
            "neighbor_forwarding_single_direction_ring": source * 3 / 8,
            "notes": "excludes replication, forwarding, protocol overhead, and draft",
        },
        "illustrative_local_selective_target_plus_draft_bytes": selective,
        "disk_only_seconds_at_supplied_rates": {
            str(rate): {"full_target": source / rate,
                        "cooperative_target": source / tp / rate}
            for rate in (4_000_000_000, 5_000_000_000)
        },
        "slice_examples": examples,
        "marlin_batched_permutation_checks": permutation_checks,
        "limitations": [
            "No physical storage or GPU measurement was performed.",
            "4 KiB touched pages are a read-amplification model, not observed device IO.",
            "Equality checks exclude quantizer packing, scales, padding, aliases, and native side effects.",
            "Timings are rank-0 arithmetic, not a synchronized dependency trace.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/nvme-loader"))
    parser.add_argument("--generation", default="nvme-stream-1789318252")
    args = parser.parse_args()
    print(json.dumps(analyze(args.results, args.generation), indent=2))
