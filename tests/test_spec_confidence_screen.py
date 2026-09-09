"""Confidence semantics, censored labels, causal joins and actual selector replay."""
import copy
import hashlib
import importlib.util
import json

import pytest
import torch

from harness import ROOT, extract

spec = importlib.util.spec_from_file_location("spec_confidence_screen", ROOT / "bench/spec_confidence_screen.py")
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)
FIXTURES = ROOT / "tests/fixtures/spec_confidence"
SELECTOR = FIXTURES / "dflash2_speculator.py"
MODEL = FIXTURES / "qwen3_dflash2.py"


def record(i=2):
    return copy.deepcopy(screen.demo_records()[i])


def test_uniform_truncated_scores_are_not_certainty():
    features = screen.path_features([[0] * 16 for _ in range(7)])
    assert features == [{"margin": 0, "pmax": 1 / 16, "entropy": pytest.approx(1)}] * 7


def test_features_ignore_logit_offset_but_respond_to_temperature_scale():
    rows = [[0, 1, 2, 3] * 4 for _ in range(7)]
    first = screen.path_features(rows)
    shifted = screen.path_features([[v + 73 for v in row] for row in rows])
    assert first == shifted
    scaled = screen.path_features([[v * 3 for v in row] for row in rows])
    assert scaled[0]["pmax"] > first[0]["pmax"]
    assert scaled[0]["entropy"] < first[0]["entropy"]


def test_selected_child_mapping_uses_token_ids_not_candidate_sort_order():
    ids = [list(reversed(range(16))) for _ in range(7)]
    scores = [[8] + [0] * 15 for _ in range(7)]
    good = screen.path_features(scores, ids, [15] * 7)
    assert good[0]["margin"] == 8
    with pytest.raises(ValueError, match="greedy maximum"):
        screen.path_features(scores, ids, [0] * 7)
    with pytest.raises(ValueError, match="uniquely mapped"):
        screen.path_features(scores, ids, [99] * 7)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_score_abstains_instead_of_producing_false_confidence(bad):
    scores = [[1] + [0] * 15 for _ in range(7)]
    scores[3][1] = bad
    result = screen.path_features(scores)
    assert result[3] is None and result[0] is not None


def test_conditional_risk_and_prefix_survival_have_different_censor_masks():
    assert screen.conditional_labels(3, 1) == [1, 0, None, None, None, None, None]
    assert screen.prefix_labels(3, 1) == [1, 0, 0, 0, 0, 0, 0]
    assert screen.conditional_labels(3, 3) == [1, 1, 1, None, None, None, None]
    assert screen.prefix_labels(3, 3) == [1, 1, 1, None, None, None, None]


def test_calibration_never_calls_unreached_tail_a_conditional_failure():
    rows = [record(0) for _ in range(20)]
    for r in rows:
        r["accepted"] = 0
    packets = [screen.select_feature(r, [r], causal=False) for r in rows]
    cells, priors = screen.fit(rows, packets)
    assert len(priors) == 1 and list(priors.values()) == [[0, 20]]
    assert all(key[2] == 0 for key in cells)


def test_current_packet_is_not_available_before_async_copy_receipt():
    rows = screen.demo_records()[:3]
    row = rows[2]
    causal = screen.select_feature(row, rows)
    noncausal = screen.select_feature(row, rows, causal=False)
    assert causal["proposal_id"] == 0 and causal["age"] == 2
    assert noncausal["proposal_id"] == 2 and noncausal["age"] == 0
    assert screen.select_feature(row, rows, max_age=1) is None


def test_epoch_and_run_identity_prevent_stale_request_slot_reuse():
    row, stale = record(2), record(0)
    for field, value in (("epoch", 2), ("request", "other"), ("run_id", "other"),
                         ("runtime_id", "original-broken-runtime")):
        changed = {**stale, field: value}
        assert screen.select_feature(row, [changed]) is None


def test_future_anchor_or_unreceived_packet_cannot_be_selected():
    row, source = record(2), record(0)
    source["anchor"] = row["anchor"] + 1
    assert screen.select_feature(row, [source]) is None
    source["anchor"] = 1
    source["feature_available_ns"] = None
    assert screen.select_feature(row, [source]) is None


@pytest.mark.parametrize("field", ["feature_proposal_id", "verified_proposal_id", "feature_anchor", "verified_anchor"])
def test_join_requires_exact_proposal_and_anchor_identity(field):
    row = record()
    row[field] += 1
    with pytest.raises(ValueError, match="ownership or anchor mismatch"):
        screen.validate_records([row])


def test_original_and_repaired_runtime_calibrations_are_separate():
    rows = [record(0), record(1)]
    rows[1]["runtime_id"] = "original"
    with pytest.raises(ValueError, match="runtimes separately"):
        screen.validate_records(rows)


def test_one_request_cannot_leak_across_prompt_training_split():
    rows = [record(0), record(1)]
    rows[1]["case"] = "other_prompt"
    with pytest.raises(ValueError, match="prompt split groups"):
        screen.validate_records(rows)


def test_future_acceptance_feedback_does_not_leak_into_history():
    row, prior = record(2), record(0)
    prior["feedback_available_ns"] = row["decision_ns"] + 1
    a = screen.probabilities(row, None, ({}, {}), [prior])
    prior["accepted"] = 0
    b = screen.probabilities(row, None, ({}, {}), [prior])
    assert a == b == [.5] * 7
    prior["feedback_available_ns"] = row["decision_ns"]
    assert screen.probabilities(row, None, ({}, {}), [prior])[0] < .5


def test_held_out_prompt_labels_do_not_train_its_confidence_model():
    rows = [record(0), record(24)]
    # Both are the first proposal of different prompts, with no prior history.
    before = screen.screen(rows)["per_prompt"][0]
    rows[0]["accepted"] = 0
    after = screen.screen(rows)["per_prompt"][0]
    assert before["caps"] == after["caps"]


def test_cost_aware_choice_uses_prefix_survival_and_known_cap_shapes():
    costs = {"1": 100, "3": 120, "5": 140, "7": 160}
    assert screen.choose_cap([0] * 7, costs) == 1
    assert screen.choose_cap([1] * 7, costs) == 7
    assert screen.survival([.5] * 7) == [.5 ** j for j in range(1, 8)]


def test_screen_marks_noncausal_upper_diagnostic_and_omits_partial_cap_utility():
    rows = [record(0), record(24)]
    for row in rows:
        row["scheduled_k"] = 3
        row["accepted"] = 3
    result = screen.screen(rows)
    assert result["status"] == "offline_screen_only"
    assert all(r["full7_boundaries"] == 0 and not r["caps"] for r in result["per_prompt"])
    assert "Not a closed-loop rollout" in result["limitations"]


@pytest.mark.parametrize("padded_state", [-1, 0])
def test_actual_edge_scores_and_greedy_walk_follow_chosen_predecessor_path(padded_state):
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == entry["sha256"]
    edge = extract(MODEL, ["_score_edges"])["_score_edges"]
    namespace = extract(SELECTOR, ["gumbel_noised_argmax", "_selector_walk_kernel"])
    kernel = namespace["_selector_walk_kernel"]
    generator = torch.Generator().manual_seed(438)
    batch, steps, topk, rank, vocab = 3, 7, 16, 4, 160
    predecessors = torch.randn(vocab, rank, generator=generator)
    successors = torch.randn(vocab, rank, generator=generator)
    hidden = torch.randn(batch, steps, rank, generator=generator)
    unary = torch.randn(batch, steps, topk, generator=generator)
    candidates = torch.arange(batch * steps * topk).reshape(batch, steps, topk) % vocab
    anchors = torch.tensor([133, 122, 111])
    scores = edge(predecessors, successors, candidates, unary, hidden, anchors, topk)
    # Independent scalar reference verifies that step0 uses the bonus anchor
    # and later edges are indexed by the actual preceding candidate index.
    for req, step, parent, child in ((0, 0, 9, 3), (1, 4, 7, 2)):
        token = anchors[req] if step == 0 else candidates[req, step - 1, parent]
        reference = unary[req, step, child] + sum(
            predecessors[token, d] * hidden[req, step, d]
            * successors[candidates[req, step, child], d] for d in range(rank))
        assert scores[req, step, parent, child].item() == pytest.approx(reference.item(), abs=1e-6)
    assert torch.equal(scores[0, 0, 0], scores[0, 0, 15])
    expected_ids, expected_scores = [], []
    for req in range(2):
        previous = 0
        ids, realized = [], []
        for step in range(steps):
            row = scores[req, step, previous]
            previous = int(row.argmax())
            ids.append(int(candidates[req, step, previous]))
            realized.append(row.tolist())
        expected_ids.append(ids)
        expected_scores.append(realized)
    # A negative state is masked, but real DFlash input preparation pads state
    # mapping with zero. Those padded graph rows can produce plausible finite
    # scores: export must use actual num_reqs, not only req_state >= 0.
    mappings = torch.tensor([[2] * 7, [0] * 7, [padded_state] * 7], dtype=torch.int32)
    temperatures = torch.zeros(3)
    seeds = torch.tensor([7, 11, 19], dtype=torch.int64)
    for anchor_position in (32, 100055):
        positions = torch.arange(anchor_position, anchor_position + batch * steps).reshape(batch, steps)
        tokens = torch.full((batch, steps), -999, dtype=torch.int64)
        realized = torch.full((batch, steps, topk), -999., dtype=torch.float32)
        kernel[(batch,)](scores, candidates, positions, mappings, temperatures, seeds,
                        tokens, realized, num_steps=7, top_k=16, BLOCK_K=16,
                        SAMPLE_PROBABILISTIC=False, USE_FP64=False)
        assert tokens[:2].tolist() == expected_ids
        assert realized[:2].tolist() == expected_scores
        if padded_state == -1:
            assert tokens[2].tolist() == [-999] * 7
            assert realized[2].flatten().tolist() == [-999.] * (7 * 16)
        else:
            assert (tokens[2] != -999).all()
            assert torch.isfinite(realized[2]).all()
        for req in range(2):
            features = screen.path_features(realized[req].tolist(), candidates[req].tolist(), tokens[req].tolist())
            assert all(f is not None and f["margin"] >= 0 for f in features)
