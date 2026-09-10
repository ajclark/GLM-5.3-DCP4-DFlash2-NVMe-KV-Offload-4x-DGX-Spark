# Task-level think-lossy report

Hold 9 protocol: pi thinking high; PI_HE_MAX_TOKENS=8192, reduced from production 32768. Observed budget deviations are reported below.

Status: **invalid**. Activation: **failed**.

| Arm | Pass | Tasks | Passed | Errors | Wall s | Median task s | Output | Reasoning | Calls | Length calls/tasks | Decode tok/s | Accepted/cycle | Reasoning char share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1 | 12 | 12 | 0 | 186.5 | 9.696 | 4892 | 0 | 36 | 0/0 | 34.21 | 5.394 | 0.3781 |
| control | 2 | 0 | 0 | 0 | 0 | n/a | 0 | 0 | 0 | 0/0 | n/a | n/a | n/a |
| control | pooled | 12 | 12 | 0 | 186.5 | 9.696 | 4892 | 0 | 36 | 0/0 | 34.21 | 5.394 | 0.3781 |
| m2.5 | 1 | 12 | 12 | 0 | 148.5 | 9.117 | 3949 | 0 | 37 | 0/0 | 37.57 | 5.944 | 0.214 |
| m2.5 | 2 | 0 | 0 | 0 | 0 | n/a | 0 | 0 | 0 | 0/0 | n/a | n/a | n/a |
| m2.5 | pooled | 12 | 12 | 0 | 148.5 | 9.117 | 3949 | 0 | 37 | 0/0 | 37.57 | 5.944 | 0.214 |
| m5.0 | 1 | 12 | 12 | 0 | 137 | 8.804 | 3553 | 0 | 36 | 0/0 | 37.79 | 5.981 | 0.2099 |
| m5.0 | 2 | 0 | 0 | 0 | 0 | n/a | 0 | 0 | 0 | 0/0 | n/a | n/a | n/a |
| m5.0 | pooled | 12 | 12 | 0 | 137 | 8.804 | 3553 | 0 | 36 | 0/0 | 37.79 | 5.981 | 0.2099 |

## m2.5 / control

Pass agreement: {"both_pass": 12, "control_only": 0, "treatment_only": 0, "both_fail": 0}.

| Ratio | Task geomean | Pair median | Bootstrap 95% | Undefined pairs |
|---|---:|---:|---|---:|
| wall_ratio | 0.9208 | 0.9765 | 0.8299, 1.002 | 0 |
| reasoning_token_ratio | n/a | n/a | n/a | 12 |
| output_token_ratio | 0.925 | 0.9982 | 0.8372, 1.003 | 0 |

| Task | Pass | Control pass | Treatment pass | Wall ratio | Reasoning ratio | Output ratio | Budget | Length-bound C/T |
|---|---:|---|---|---:|---:|---:|---|---|
| HumanEval/0 | 1 | True | True | 1.001 | n/a | 1 | matched | False/False |
| HumanEval/100 | 1 | True | True | 0.9987 | n/a | 1 | matched | False/False |
| HumanEval/122 | 1 | True | True | 0.9721 | n/a | 1 | matched | False/False |
| HumanEval/132 | 1 | True | True | 0.5861 | n/a | 0.608 | matched | False/False |
| HumanEval/154 | 1 | True | True | 0.879 | n/a | 0.9964 | matched | False/False |
| HumanEval/20 | 1 | True | True | 0.9136 | n/a | 0.981 | matched | False/False |
| HumanEval/22 | 1 | True | True | 1.16 | n/a | 1.143 | matched | False/False |
| HumanEval/48 | 1 | True | True | 0.9917 | n/a | 1 | matched | False/False |
| HumanEval/49 | 1 | True | True | 0.9809 | n/a | 0.9468 | matched | False/False |
| HumanEval/55 | 1 | True | True | 0.9031 | n/a | 0.7961 | matched | False/False |
| HumanEval/69 | 1 | True | True | 0.7604 | n/a | 0.7661 | matched | False/False |
| HumanEval/75 | 1 | True | True | 1.048 | n/a | 1 | matched | False/False |

## m5.0 / control

Pass agreement: {"both_pass": 12, "control_only": 0, "treatment_only": 0, "both_fail": 0}.

| Ratio | Task geomean | Pair median | Bootstrap 95% | Undefined pairs |
|---|---:|---:|---|---:|
| wall_ratio | 0.8994 | 0.9646 | 0.7363, 1.082 | 0 |
| reasoning_token_ratio | n/a | n/a | n/a | 12 |
| output_token_ratio | 0.8833 | 0.9928 | 0.7274, 1.062 | 0 |

| Task | Pass | Control pass | Treatment pass | Wall ratio | Reasoning ratio | Output ratio | Budget | Length-bound C/T |
|---|---:|---|---|---:|---:|---:|---|---|
| HumanEval/0 | 1 | True | True | 1.007 | n/a | 1 | matched | False/False |
| HumanEval/100 | 1 | True | True | 0.9974 | n/a | 1 | matched | False/False |
| HumanEval/122 | 1 | True | True | 0.9986 | n/a | 1 | matched | False/False |
| HumanEval/132 | 1 | True | True | 0.4244 | n/a | 0.445 | matched | False/False |
| HumanEval/154 | 1 | True | True | 0.8685 | n/a | 0.9856 | matched | False/False |
| HumanEval/20 | 1 | True | True | 1.035 | n/a | 1.017 | matched | False/False |
| HumanEval/22 | 1 | True | True | 0.9135 | n/a | 0.7206 | matched | False/False |
| HumanEval/48 | 1 | True | True | 0.9994 | n/a | 1 | matched | False/False |
| HumanEval/49 | 1 | True | True | 0.9318 | n/a | 0.984 | matched | False/False |
| HumanEval/55 | 1 | True | True | 0.8863 | n/a | 0.7882 | matched | False/False |
| HumanEval/69 | 1 | True | True | 0.5345 | n/a | 0.5298 | matched | False/False |
| HumanEval/75 | 1 | True | True | 1.817 | n/a | 1.707 | matched | False/False |

## Activation and guards

```json
{
  "activation": {
    "status": "failed",
    "arms": {
      "control": {
        "verify_rows": 401,
        "relaxed": 0,
        "failures": {},
        "proven": true
      },
      "m2.5": {
        "verify_rows": 0,
        "relaxed": 0,
        "failures": {
          "label_or_arm_window_mismatch": 334,
          "missing_label_rows": 1,
          "no_relaxation": 1
        },
        "proven": false
      },
      "m5.0": {
        "verify_rows": 0,
        "relaxed": 0,
        "failures": {
          "label_or_arm_window_mismatch": 272,
          "missing_label_rows": 1,
          "no_relaxation": 1
        },
        "proven": false
      }
    },
    "other_traffic_verify_rows": 1253,
    "non_verify_rows": 1,
    "invalid_time_rows": 0
  },
  "issues": {
    "control/pass1:proxy_client_call_count_mismatch": 1,
    "m2.5/pass1:proxy_client_call_count_mismatch": 1,
    "m5.0/pass1:proxy_client_call_count_mismatch": 1
  },
  "pending_counts": {
    "missing_or_unfinished_json": 3,
    "incomplete_arm_pass": 3,
    "unassigned_proxy_call": 37,
    "unmatched_task_sets": 1
  },
  "budget_deviation_arms": [],
  "unassigned_proxy_calls": 37
}
```

Conventions:

- Official check.passed only; check output, errors, request payloads and private texts are never copied.
- Wall is summed lane-summary elapsed time; pooled median is over individual task times.
- Pooled decode = sum proxy generation_tokens / sum proxy server_decode_s.
- Accepted/cycle = 1 + sum spec_accepted_tokens / sum spec_drafts; includes the bonus token.
- Reasoning character share uses saved counts; it is not a tokenizer-derived token share.
- Ratios are treatment/control; below one means less time or fewer tokens, not necessarily better quality.
- Geomean gives each task equal weight; 95% percentile bootstrap resamples task clusters with all available paired passes.
- Zero denominators are undefined and counted; zero numerators produce zero ratios/geomeans.
- Trace time uses proxy t as call START; windows are inclusive [t,t+wall_s].
- Verify rows outside every proxy window are other traffic, including known labels; labels inside the wrong arm window fail.
- Length-bound tasks have at least one proxy finish_reason=length; budgets must match for each paired task.
- Partial input may be an inconsistent live snapshot and is never promotion-ready.
