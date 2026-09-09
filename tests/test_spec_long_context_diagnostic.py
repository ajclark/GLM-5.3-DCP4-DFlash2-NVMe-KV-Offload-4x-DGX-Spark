"""Actual pinned DFlash input kernel at long absolute positions, on CPU.

The reduced-width reproduction is an intentionally invalid table: no model
or CUDA allocation is involved. It demonstrates why metadata must be sized
for a replicated draft group, independently of acceptance/model quality.
"""
import ast
import importlib.util
import json
from types import SimpleNamespace as NS

import pytest
import torch

from harness import BASELINE, OVERLAY, ROOT, extract
from spec_harness import FIXTURES
from test_dcp_replicated_group import HELPER, MambaSpec, SlidingWindowSpec

spec = importlib.util.spec_from_file_location(
    "spec_long_context_diagnostic", ROOT / "bench/spec_long_context_diagnostic.py")
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)
DRAFT = extract(FIXTURES / "dflash_speculator.py", ["_prepare_dflash_inputs_kernel"])[
    "_prepare_dflash_inputs_kernel"]
PINNED_RUNNER = BASELINE / "v1/worker/gpu/model_runner.py"
PINNED_BUFFERS = BASELINE / "v1/worker/gpu/buffer_utils.py"


def _int(values, dtype=torch.int32):
    return torch.tensor(values, dtype=dtype)


def _run(snapshot, *, mapping=None, sampled=None):
    nreq = len(snapshot["num_rejected"])
    maxreq = nreq + 1
    numctx = len(snapshot["target_positions"])
    maxtokens = max(numctx, maxreq * 8)
    output = {
        name: torch.full((size,), -999, dtype=dtype)
        for name, size, dtype in [
            ("ids", maxtokens, torch.int32),
            ("query_positions", maxtokens, torch.int64),
            ("cu", maxreq + 1, torch.int32),
            ("seq_lens", maxreq, torch.int32),
            ("query_slots", maxtokens, torch.int64),
            ("context_positions", numctx, torch.int64),
            ("context_slots", numctx, torch.int64),
            ("sample_indices", maxreq * 7, torch.int64),
            ("sample_positions", maxreq * 7, torch.int64),
            ("sample_mapping", maxreq * 7, torch.int32),
        ]
    }
    mapping = mapping if mapping is not None else list(range(nreq))
    sampled = sampled if sampled is not None else [1] * nreq
    last = _int([1000 + i for i in range(maxreq)])
    prefill = _int([2000 + i for i in range(maxreq)])
    starts = snapshot["target_query_start_loc"]
    max_context = max(b - a for a, b in zip(starts, starts[1:]))
    threads = min(256, 1 << (max_context + 8 - 1).bit_length())
    DRAFT[(nreq, (max_context + 8 + threads - 1) // threads)](
        output["ids"], output["query_positions"], output["cu"], output["seq_lens"],
        output["query_slots"], output["context_positions"], output["context_slots"],
        output["sample_indices"], output["sample_positions"], output["sample_mapping"],
        _int(snapshot["target_positions"], torch.int64), _int(starts), _int(mapping),
        last, prefill, _int(sampled), _int(snapshot["num_rejected"]),
        _int(snapshot["block_table"]), len(snapshot["block_table"][0]),
        777, snapshot["block_size"], 8, 7, maxreq, maxtokens,
        PAD_SLOT_ID=-1, BLOCK_SIZE=threads,
    )
    for req in range(nreq):
        assert output["ids"][req * 8].item() == (
            last[mapping[req]] if sampled[req] else prefill[mapping[req]])
        assert output["ids"][req * 8 + 1: (req + 1) * 8].tolist() == [777] * 7
        assert output["sample_mapping"][req * 7: (req + 1) * 7].tolist() == [mapping[req]] * 7
    assert output["query_slots"][nreq * 8:].tolist() == [-1] * (maxtokens - nreq * 8)
    assert output["seq_lens"][nreq:].tolist() == [0]
    assert output["cu"].tolist() == [i * 8 for i in range(nreq + 1)] + [nreq * 8]
    observed = {}
    for name, size in [("query_positions", nreq * 8), ("query_slots", nreq * 8),
                       ("context_positions", numctx), ("context_slots", numctx),
                       ("sample_positions", nreq * 7), ("seq_lens", nreq)]:
        observed[name] = output[name][:size].tolist()
    return observed


def _snapshot(position, cap=7, accepted=0, width=2816, block_size=64):
    # Nonidentity physical IDs prevent accidental equality between a logical
    # index and its cache address. No corresponding physical pages are allocated.
    return {
        "target_positions": list(range(position, position + cap + 1)),
        "target_query_start_loc": [0, cap + 1], "num_rejected": [cap - accepted],
        "block_size": block_size,
        "block_table": [[5000 + 3 * i for i in range(width)]],
    }


@pytest.mark.parametrize("position", [2044, 32764, 65532, 90108, 100055, 131068, 170125, 180208])
@pytest.mark.parametrize("cap,accepted", [(k, a) for k in (1, 3, 5, 7) for a in range(k + 1)])
def test_long_positions_and_every_cap_rejection_anchor(position, cap, accepted):
    snapshot = _snapshot(position, cap, accepted)
    snapshot["observed"] = _run(snapshot)
    result = diagnostic.audit_geometry(snapshot)
    assert result["ok"], result["issues"]
    # Rejected target rows are preserved during precompute, then overwritten
    # by query K/V at the same absolute positions; their anchor must roll back.
    assert result["requests"][0]["last_valid_position"] == position + accepted


def test_chunked_prefill_at_long_position_and_permuted_request_state():
    snapshot = _snapshot(99745)
    snapshot["target_positions"] = list(range(99745, 100002)) + list(range(131066, 131070))
    snapshot["target_query_start_loc"] = [0, 257, 261]
    snapshot["num_rejected"] = [0, 2]
    snapshot["block_table"].append([19000 + 5 * i for i in range(2816)])
    snapshot["observed"] = _run(snapshot, mapping=[2, 0], sampled=[0, 2])
    result = diagnostic.audit_geometry(snapshot)
    assert result["ok"], result["issues"]
    assert result["expected"]["query_positions"][:8] == list(range(100002, 100010))
    assert result["expected"]["query_positions"][8:] == list(range(131068, 131076))


def test_physical_slot_math_exceeds_int32_without_overflow():
    snapshot = _snapshot(100055)
    snapshot["block_table"] = [[(1 << 25) + 3 * i for i in range(2816)]]
    snapshot["observed"] = _run(snapshot)
    assert min(snapshot["observed"]["query_slots"]) > (1 << 31)
    assert diagnostic.audit_geometry(snapshot)["ok"]


@pytest.mark.parametrize("path,columns", [
    (PINNED_RUNNER, 1408), (OVERLAY / "v1/worker/gpu/model_runner.py", 2816)])
def test_original_and_repaired_v2_sizing_drive_real_long_context_kernel(path, columns):
    # Execute actual source from both the committed original and repaired
    # worker; no worker/model import or initialization is performed.
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef) and node.name == "initialize_kv_cache")
    sizing_loop = next(node for node in method.body
                       if isinstance(node, ast.For)
                       and isinstance(node.target, ast.Name)
                       and node.target.id == "kv_cache_group")
    draft_spec = SlidingWindowSpec(block_size=64)
    env = {
        "block_table_max_model_len": 180224, "block_sizes": [],
        "max_num_blocks_per_group": [], "MambaSpec": MambaSpec, "cp_sizes": [],
        "cp_world_size_for_kv_cache_spec": HELPER,
        "self": NS(dcp_size=2), "cdiv": lambda a, b: -(-a // b),
        "kv_cache_config": NS(kv_cache_groups=[NS(kv_cache_spec=draft_spec)]),
    }
    module = ast.fix_missing_locations(ast.Module(body=[sizing_loop], type_ignores=[]))
    exec(compile(module, str(path), "exec"), env)
    assert env["max_num_blocks_per_group"] == [columns]
    group_cp = HELPER(draft_spec, 2)
    assert group_cp == 1
    assert 180224 // (64 * group_cp) == 2816
    snapshot = _snapshot(100055, width=columns)
    snapshot["observed"] = _run(snapshot)
    assert diagnostic.audit_geometry(snapshot)["ok"] == (columns == 2816)


def test_pinned_staged_write_can_cross_undersized_row_into_next_request():
    kernels = extract(PINNED_BUFFERS, ["_load_ptr", "_apply_write_kernel"])
    # Keep the write safely inside one CPU allocation while demonstrating
    # cross-request row corruption. This never performs an out-of-allocation
    # write, and intentionally does not launch against CUDA memory.
    rows = torch.full((2, 1408), -999, dtype=torch.int32)
    contents = torch.arange(1564, dtype=torch.int32) + 10
    kernels["_apply_write_kernel"][(1,)](
        rows, rows.stride(0), _int([0]), _int([0]), contents, _int([1564]),
        None, BLOCK_SIZE=1024, MULTI_GROUP=False,
    )
    assert torch.equal(rows[0], contents[:1408])
    assert torch.equal(rows[1, :156], contents[1408:])
    assert rows[1, 156:].tolist() == [-999] * (1408 - 156)


@pytest.mark.parametrize("position", [90108, 100055, 170125])
def test_uniform_dcp_sized_row_reproduces_silent_draft_slot_clamp(position):
    # Pinned V2 runner sizes every group for DCP2: 180224/(64*2)=1408.
    # The replicated draft requires 2816 logical columns. The real kernel
    # quietly redirects out-of-row positions to the last physical block.
    snapshot = _snapshot(position, width=1408)
    observed = _run(snapshot)
    result = diagnostic.audit_geometry(snapshot)
    issues = result["issues"]
    assert issues and all(x["kind"] == "table_row_overflow" for x in issues)
    for pos, slot in zip(observed["query_positions"], observed["query_slots"]):
        if pos >= 90112:
            assert slot == snapshot["block_table"][0][-1] * 64 + pos % 64
    # Increasing only table metadata to the replicated-group width removes
    # every alias; absolute positions and the trained eight-query block agree.
    full = _snapshot(position)
    full["observed"] = _run(full)
    assert diagnostic.audit_geometry(full)["ok"]


def test_snapshot_rejects_bad_rollback_and_detects_observed_position_drift():
    snapshot = _snapshot(100055)
    snapshot["num_rejected"] = [8]
    with pytest.raises(ValueError, match="invalid rejection"):
        diagnostic.audit_geometry(snapshot)
    snapshot["num_rejected"] = [7]
    snapshot["observed"] = _run(snapshot)
    snapshot["observed"]["query_positions"][0] += 1
    assert diagnostic.audit_geometry(snapshot)["issues"] == [
        {"kind": "output_mismatch", "array": "query_positions"}]


def test_record_summary_keeps_metric_denominators_and_zero_acceptance(tmp_path):
    folder = tmp_path / "repo100k-fixed-r4"
    folder.mkdir()
    record = {"cap": 7, "case": "code", "usage": {"prompt_tokens": 100055},
              "spec_metric_delta": {
                  'vllm:spec_decode_num_drafts_total{engine="0"}': 65,
                  'vllm:spec_decode_num_accepted_tokens_per_pos_total{position="0"}': 0,
              }}
    (folder / "fixture-fixed-k7.json").write_text(json.dumps(record))
    rows = diagnostic.summarize_records(tmp_path)
    assert len(rows) == 1
    assert rows[0]["first_acceptance_fraction_metric"] == 0
    assert rows[0]["draft_cycles_metric"] == 65
    assert rows[0]["accepted_second_metric"] is None
