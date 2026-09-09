"""Tree DP against exhaustive enumeration and pinned model geometry estimates."""
import importlib.util
import itertools
import json
import random
import sys
from fractions import Fraction
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from harness import ROOT, extract
from spec_harness import source_class

spec=importlib.util.spec_from_file_location("architecture_screen_test",ROOT/"bench/spec_architecture_screen.py")
A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)
CONFIG=json.loads((ROOT/"tests/fixtures/spec_k0/draft-config.json").read_text())


def brute_force(nodes,budget):
    by_name={node.name:node for node in nodes}
    best={}
    for size in range(min(budget,len(nodes))+1):
        for choice in itertools.combinations(sorted(by_name),size):
            chosen=set(choice)
            if any(by_name[name].parent is not None and by_name[name].parent not in chosen for name in chosen):
                continue
            expected=Fraction(1)
            for name in chosen:
                probability=Fraction(1)
                while name is not None:
                    probability*=Fraction(str(by_name[name].conditional))
                    name=by_name[name].parent
                expected+=probability
            if size not in best or expected>best[size][0] or (expected==best[size][0] and choice<best[size][1]):
                best[size]=expected,choice
    return best


@pytest.mark.parametrize("seed",range(20))
def test_exact_tree_knapsack_matches_brute_force(seed):
    rng=random.Random(seed)
    nodes=[];remaining={None:16}
    for index in range(rng.randint(2,9)):
        parent=rng.choice([None]+[node.name for node in nodes])
        weight=rng.randint(0,remaining[parent]);remaining[parent]-=weight
        name=f"n{index}";remaining[name]=16
        nodes.append(A.Node(name,parent,index,weight/16))
    rng.shuffle(nodes)  # Source ordering must not imply parent-before-child.
    budget=rng.randrange(len(nodes)+1)
    expected=brute_force(nodes,budget)
    for plan in A.tree_frontier(nodes,budget):
        value,chosen=expected[plan["nodes"]]
        assert Fraction(plan["exact_expected_emitted"])==value
        assert tuple(plan["selected"])==chosen
        assert len(json.loads(plan["shape"]))==plan["nodes"]


def test_exclusive_branches_conserve_probability_and_expected_path_length():
    nodes=[A.Node("a",None,1,.6),A.Node("b",None,2,.4),
           A.Node("aa","a",3,.5),A.Node("ab","a",4,.5),
           A.Node("ba","b",3,.25),A.Node("bb","b",4,.75)]
    plan=A.tree_frontier(nodes,6)[-1]
    assert plan["expected_emitted"]==3
    assert plan["nodes"]==6  # Six verified candidates, one two-token accepted path.


def test_chain_reduces_to_one_plus_prefix_survival():
    nodes=[A.Node(str(i),str(i-1) if i else None,i,.5) for i in range(7)]
    frontier=A.tree_frontier(nodes,7)
    for plan in frontier:
        assert plan["expected_emitted"]==1+sum(.5**j for j in range(1,plan["nodes"]+1))


@pytest.mark.parametrize("nodes,message",[
    ([A.Node("a",None,1,.8),A.Node("b",None,2,.8)],"sum"),
    ([A.Node("a",None,1,.4),A.Node("b",None,1,.4)],"distinct"),
    ([A.Node("a","b",1,.5),A.Node("b","a",2,.5)],"cycle"),
    ([A.Node("a","missing",1,.5)],"parent"),
    ([A.Node("a",None,1,float("nan"))],"invalid"),
    ([A.Node("a",None,1,True)],"invalid"),
    ([A.Node("a",None,1,.5),A.Node("a",None,2,.5)],"invalid"),
])
def test_unusable_tree_estimates_are_rejected(nodes,message):
    with pytest.raises(ValueError,match=message): A.tree_frontier(nodes)


def test_exact_decimal_sibling_sum_and_deterministic_ties():
    nodes=[A.Node("c",None,3,.7),A.Node("b",None,2,.2),A.Node("a",None,1,.1)]
    assert A.tree_frontier(nodes,3)[-1]["expected_emitted"]==2
    tied=[A.Node("z",None,1,.5),A.Node("a",None,2,.5)]
    assert A.tree_frontier(tied,1)[1]["selected"]==["a"]


def test_unselected_ancestor_cannot_be_encoded_as_a_root_child():
    nodes=[A.Node("a",None,1,.5),A.Node("b","a",2,.5)]
    by_name,_,_,order=A.validate_tree(nodes)
    with pytest.raises(ValueError,match="ancestor closed"):
        A.shape_key(["b"],by_name,order)


def test_branch_and_chain_costs_are_not_interchangeable():
    chain=A.tree_frontier([A.Node("a",None,1,.9),A.Node("b","a",2,.9)],2)
    branch=A.tree_frontier([A.Node("a",None,1,.6),A.Node("b",None,2,.4)],2)
    assert chain[-1]["shape"]=="[0,1]" and branch[-1]["shape"]=="[0,0]"
    costs={"kind":"tree_shape","provenance":"measured","runtime_id":"test",
           "context_tokens":100000,"cycle_ms_by_shape":{"[0,1]":130}}
    assert A.rank_frontier(chain,costs)["ranked"]
    assert not A.rank_frontier(branch,costs)["ranked"]
    costs["kind"]="chain"
    with pytest.raises(ValueError,match="tree-shape"): A.rank_frontier(chain,costs)


def test_missing_tree_costs_abstain_and_synthetic_costs_stay_labeled():
    frontier=A.tree_frontier([A.Node("a",None,1,.5)],1)
    assert A.rank_frontier(frontier)["status"]=="tree_costs_unmeasured"
    costs={"kind":"tree_shape","provenance":"synthetic","runtime_id":"synthetic",
           "context_tokens":32,"cycle_ms_by_shape":{"[]":100,"[0]":120}}
    result=A.rank_frontier(frontier,costs)
    assert result["status"]=="synthetic_scenario"
    assert result["ranked"][0]["modelled_expected_tokens_per_ms"]==1.5/120
    assert "not a global latency optimum" in result["warning"]


def test_actual_selector_constructor_matches_parameter_estimate_without_allocating_weights():
    def linear(in_features,out_features,**kwargs):
        return torch.nn.Linear(in_features,out_features,bias=kwargs["bias"],
                               dtype=kwargs["params_dtype"],device="meta")
    def empty(*shape,**kwargs): return torch.empty(*shape,device="meta",**kwargs)
    cls=source_class(ROOT/"tests/fixtures/spec_confidence/qwen3_dflash2.py","CandidateSelector",["__init__"],
        {"nn":torch.nn,"torch":NS(empty=empty),"ReplicatedLinear":linear,
         "support_torch_compile":lambda cls:cls,"maybe_prefix":lambda prefix,name:name})
    model=cls(CONFIG["hidden_size"],CONFIG["vocab_size"],256,16,torch.bfloat16,"")
    estimate=A.selector_memory(CONFIG)
    assert sum(p.numel() for p in model.parameters())==estimate["selector_parameters"]==80871424
    assert model.predecessor_codebook.shape==model.successor_codebook.shape==(154880,256)
    assert model.hidden_projection.weight.shape==(256,6144)
    assert all(p.device.type=="meta" for p in model.parameters())


def test_training_accounting_includes_optimizer_and_abstains_on_unknown_peak():
    value=A.selector_memory(CONFIG)
    assert value["bf16_selector_weight_bytes"]==161742848
    assert value["full_selector_adam_parameter_state_bytes"]==1293942784
    assert value["full_training_peak_bytes"] is None
    assert value["lora_trainable_parameters"]==102400
    assert value["lora_adam_parameter_state_bytes"]==1638400
    assert value["cached_hidden_bytes_per_block"]==7*6144*2


def test_window_compression_is_a_payload_scenario_not_pool_release():
    rows=A.draft_window_memory(CONFIG)["scenarios"]
    assert [row["assumed_kv_heads_per_rank"] for row in rows]==[2,4,8]
    assert [row["bf16_window_payload_bytes"] for row in rows]==[12*2**20,24*2**20,48*2**20]
    assert all(row["window_plus_query_block_rounded_tokens"]==2112 for row in rows)
    assert all(row["ideal_int8_rounded_payload_bytes"]*2==row["bf16_rounded_payload_bytes"] for row in rows)
    assert "does not imply RSS" in A.draft_window_memory(CONFIG)["warning"]


def test_actual_grouped_conv_changes_at_trained_block_boundary():
    fn=extract(ROOT/"tests/fixtures/spec_confidence/qwen3_dflash2.py",["_grouped_conv"],{"F":F})["_grouped_conv"]
    hidden=torch.ones(16,16);delta=torch.zeros(16,2,1);base=torch.ones(2,1,16)
    trained=fn(hidden,delta,base,8,1,16,2)
    extended=fn(hidden,delta,base,16,1,16,2)
    assert trained[8].eq(1).all() and extended[8].eq(2).all()
    rows=A.longer_blocks(CONFIG)
    assert [r["matches_loaded_training_geometry"] for r in rows]==[True,False,False]
    assert rows[-1]["target_graph_rows_if_full_chain"]==16


def test_exact_copy_screen_preserves_baseline_trajectory_limitation():
    data={"screening_only":True,"runs":[{"case":"code","boundaries":86,"copy_matches":4,
          "accepted_prefixes_on_baseline":[0,0,2,1]}]}
    result=A.copy_opportunity(data);row=result["rows"][0]
    assert row["hit_fraction"]==4/86 and row["accepted_when_matched"]==.75
    assert row["copied_tokens_per_baseline_boundary"]==3/86
    assert "not closed-loop" in result["warning"]
    data["runs"][0]["copy_matches"]=5
    with pytest.raises(ValueError,match="inconsistent"): A.copy_opportunity(data)
