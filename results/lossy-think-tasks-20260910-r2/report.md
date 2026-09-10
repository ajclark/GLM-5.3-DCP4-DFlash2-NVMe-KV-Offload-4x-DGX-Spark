# Task-level bounded-lossy report

Task protocol: lossy_scope=think; pi thinking high; PI_HE_MAX_TOKENS=8192, reduced from production 32768. Observed budget deviations are reported below.

Status: **complete**. Activation: **proven**.

| Arm | Pass | Tasks | Passed | Errors | Wall s | Median task s | Output | Reasoning | Calls | Length calls/tasks | Decode tok/s | Accepted/cycle | Reasoning char share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1 | 12 | 12 | 0 | 202.6 | 12.89 | 5236 | 0 | 45 | 0/0 | 34.82 | 5.505 | 0.2531 |
| control | 2 | 12 | 12 | 0 | 219.2 | 12.37 | 5554 | 0 | 47 | 0/0 | 33.66 | 5.333 | 0.2959 |
| control | pooled | 24 | 24 | 0 | 421.8 | 12.89 | 1.079e+04 | 0 | 92 | 0/0 | 34.21 | 5.415 | 0.2754 |
| m2.5 | 1 | 12 | 11 | 0 | 203.4 | 14.93 | 5883 | 0 | 50 | 0/0 | 40.19 | 6.387 | 0.174 |
| m2.5 | 2 | 12 | 11 | 0 | 510.3 | 15.83 | 1.472e+04 | 0 | 50 | 1/1 | 32.45 | 5.188 | 0.6662 |
| m2.5 | pooled | 24 | 22 | 0 | 713.7 | 15.63 | 2.06e+04 | 0 | 100 | 1/1 | 34.34 | 5.482 | 0.5248 |
| m5.0 | 1 | 12 | 11 | 0 | 418.7 | 15.27 | 1.306e+04 | 0 | 46 | 1/1 | 35.63 | 5.715 | 0.7169 |
| m5.0 | 2 | 12 | 11 | 0 | 416.1 | 13.65 | 1.306e+04 | 0 | 47 | 1/1 | 36.04 | 5.764 | 0.7001 |
| m5.0 | pooled | 24 | 22 | 0 | 834.8 | 14.42 | 2.612e+04 | 0 | 93 | 2/2 | 35.84 | 5.74 | 0.7086 |

## m2.5 / control

Pass agreement: {"both_pass": 22, "control_only": 2, "treatment_only": 0, "both_fail": 0}.

| Ratio | Task geomean | Pair median | Bootstrap 95% | Undefined pairs |
|---|---:|---:|---|---:|
| wall_ratio | 1.206 | 1.198 | 1.073, 1.355 | 0 |
| reasoning_token_ratio | n/a | n/a | n/a | 24 |
| output_token_ratio | 1.358 | 1.462 | 1.164, 1.562 | 0 |

| Task | Pass | Control pass | Treatment pass | Wall ratio | Reasoning ratio | Output ratio | Budget | Length-bound C/T |
|---|---:|---|---|---:|---:|---:|---|---|
| HumanEval/0 | 1 | True | True | 0.9547 | n/a | 1.04 | matched | False/False |
| HumanEval/100 | 1 | True | True | 1.259 | n/a | 1.561 | matched | False/False |
| HumanEval/122 | 1 | True | True | 1.421 | n/a | 1.64 | matched | False/False |
| HumanEval/132 | 1 | True | False | 0.6321 | n/a | 0.7221 | matched | False/False |
| HumanEval/154 | 1 | True | True | 0.6868 | n/a | 0.9108 | matched | False/False |
| HumanEval/20 | 1 | True | True | 0.7569 | n/a | 0.6684 | matched | False/False |
| HumanEval/22 | 1 | True | True | 1.436 | n/a | 1.727 | matched | False/False |
| HumanEval/48 | 1 | True | True | 1.342 | n/a | 1.915 | matched | False/False |
| HumanEval/49 | 1 | True | True | 1.259 | n/a | 1.576 | matched | False/False |
| HumanEval/55 | 1 | True | True | 1.061 | n/a | 1.074 | matched | False/False |
| HumanEval/69 | 1 | True | True | 1.055 | n/a | 1.08 | matched | False/False |
| HumanEval/75 | 1 | True | True | 1.891 | n/a | 2.059 | matched | False/False |
| HumanEval/0 | 2 | True | True | 1.621 | n/a | 2.352 | matched | False/False |
| HumanEval/100 | 2 | True | True | 1.275 | n/a | 1.515 | matched | False/False |
| HumanEval/122 | 2 | True | True | 1.583 | n/a | 1.657 | matched | False/False |
| HumanEval/132 | 2 | True | False | 4.52 | n/a | 5.549 | matched | False/True |
| HumanEval/154 | 2 | True | True | 1.654 | n/a | 1.681 | matched | False/False |
| HumanEval/20 | 2 | True | True | 0.9083 | n/a | 0.929 | matched | False/False |
| HumanEval/22 | 2 | True | True | 1.015 | n/a | 1.119 | matched | False/False |
| HumanEval/48 | 2 | True | True | 1.066 | n/a | 1.049 | matched | False/False |
| HumanEval/49 | 2 | True | True | 1.136 | n/a | 1.409 | matched | False/False |
| HumanEval/55 | 2 | True | True | 0.8869 | n/a | 0.9052 | matched | False/False |
| HumanEval/69 | 2 | True | True | 0.8991 | n/a | 0.9004 | matched | False/False |
| HumanEval/75 | 2 | True | True | 1.429 | n/a | 1.536 | matched | False/False |

## m5.0 / control

Pass agreement: {"both_pass": 22, "control_only": 2, "treatment_only": 0, "both_fail": 0}.

| Ratio | Task geomean | Pair median | Bootstrap 95% | Undefined pairs |
|---|---:|---:|---|---:|
| wall_ratio | 1.248 | 1.165 | 1.058, 1.586 | 0 |
| reasoning_token_ratio | n/a | n/a | n/a | 24 |
| output_token_ratio | 1.412 | 1.279 | 1.139, 1.883 | 0 |

| Task | Pass | Control pass | Treatment pass | Wall ratio | Reasoning ratio | Output ratio | Budget | Length-bound C/T |
|---|---:|---|---|---:|---:|---:|---|---|
| HumanEval/0 | 1 | True | True | 1.363 | n/a | 1.518 | matched | False/False |
| HumanEval/100 | 1 | True | True | 1.245 | n/a | 1.557 | matched | False/False |
| HumanEval/122 | 1 | True | True | 1.037 | n/a | 1.139 | matched | False/False |
| HumanEval/132 | 1 | True | False | 4.493 | n/a | 5.65 | matched | False/True |
| HumanEval/154 | 1 | True | True | 0.9647 | n/a | 1.222 | matched | False/False |
| HumanEval/20 | 1 | True | True | 0.9739 | n/a | 1 | matched | False/False |
| HumanEval/22 | 1 | True | True | 1.557 | n/a | 1.947 | matched | False/False |
| HumanEval/48 | 1 | True | True | 1.02 | n/a | 1.113 | matched | False/False |
| HumanEval/49 | 1 | True | True | 1.199 | n/a | 1.52 | matched | False/False |
| HumanEval/55 | 1 | True | True | 1.081 | n/a | 1.214 | matched | False/False |
| HumanEval/69 | 1 | True | True | 0.9734 | n/a | 0.988 | matched | False/False |
| HumanEval/75 | 1 | True | True | 1.307 | n/a | 1.336 | matched | False/False |
| HumanEval/0 | 2 | True | True | 1.539 | n/a | 2.304 | matched | False/False |
| HumanEval/100 | 2 | True | True | 1.258 | n/a | 1.515 | matched | False/False |
| HumanEval/122 | 2 | True | True | 1.131 | n/a | 1.196 | matched | False/False |
| HumanEval/132 | 2 | True | False | 3.463 | n/a | 4.93 | matched | False/True |
| HumanEval/154 | 2 | True | True | 1.089 | n/a | 1.128 | matched | False/False |
| HumanEval/20 | 2 | True | True | 0.8637 | n/a | 0.7323 | matched | False/False |
| HumanEval/22 | 2 | True | True | 0.8073 | n/a | 0.7699 | matched | False/False |
| HumanEval/48 | 2 | True | True | 1.263 | n/a | 1.535 | matched | False/False |
| HumanEval/49 | 2 | True | True | 1.249 | n/a | 1.561 | matched | False/False |
| HumanEval/55 | 2 | True | True | 1.01 | n/a | 1.151 | matched | False/False |
| HumanEval/69 | 2 | True | True | 0.8251 | n/a | 0.7841 | matched | False/False |
| HumanEval/75 | 2 | True | True | 1.373 | n/a | 1.417 | matched | False/False |

## Activation and guards

```json
{
  "scope": "think",
  "labels": {
    "m2.5": "think-tasks-m2.5",
    "m5.0": "think-tasks-m5.0"
  },
  "window_slack_s": 2.0,
  "include_extra_calls": false,
  "extra_calls": {
    "control": {
      "output_tokens": 0,
      "reasoning_tokens": 0,
      "api_calls": 0,
      "tasks": 0,
      "passed": 0,
      "errors": 0,
      "wall_s": 0,
      "wall_complete": true,
      "median_task_s": null,
      "proxy_calls": 8,
      "proxy_api_calls": 12,
      "length_calls": 0,
      "length_bound_tasks": 0,
      "pooled_decode_tok_s": 15.96351197263398,
      "proxy_generation_tokens": 826,
      "server_decode_s": 51.743,
      "spec_drafts": 129,
      "spec_accepted_tokens": 696,
      "accepted_per_cycle": 6.395348837209302,
      "reasoning_chars_share": 0.047619047619047616,
      "completion_budgets": [
        8192
      ],
      "unknown_budget_calls": 0,
      "thinking_not_confirmed_calls": 0
    },
    "m2.5": {
      "output_tokens": 0,
      "reasoning_tokens": 0,
      "api_calls": 0,
      "tasks": 0,
      "passed": 0,
      "errors": 0,
      "wall_s": 0,
      "wall_complete": true,
      "median_task_s": null,
      "proxy_calls": 1,
      "proxy_api_calls": 1,
      "length_calls": 0,
      "length_bound_tasks": 0,
      "pooled_decode_tok_s": 38.288883020860844,
      "proxy_generation_tokens": 145,
      "server_decode_s": 3.787,
      "spec_drafts": 26,
      "spec_accepted_tokens": 119,
      "accepted_per_cycle": 5.576923076923077,
      "reasoning_chars_share": 0.18214285714285713,
      "completion_budgets": [
        200
      ],
      "unknown_budget_calls": 0,
      "thinking_not_confirmed_calls": 0
    },
    "m5.0": {
      "output_tokens": 0,
      "reasoning_tokens": 0,
      "api_calls": 0,
      "tasks": 0,
      "passed": 0,
      "errors": 0,
      "wall_s": 0,
      "wall_complete": true,
      "median_task_s": null,
      "proxy_calls": 1,
      "proxy_api_calls": 1,
      "length_calls": 0,
      "length_bound_tasks": 0,
      "pooled_decode_tok_s": 40.72790294627384,
      "proxy_generation_tokens": 94,
      "server_decode_s": 2.308,
      "spec_drafts": 16,
      "spec_accepted_tokens": 77,
      "accepted_per_cycle": 5.8125,
      "reasoning_chars_share": 0.06629834254143646,
      "completion_budgets": [
        200
      ],
      "unknown_budget_calls": 0,
      "thinking_not_confirmed_calls": 0
    }
  },
  "activation": {
    "status": "proven",
    "arms": {
      "control": {
        "verify_rows": 1998,
        "relaxed": 0,
        "failures": {},
        "proven": true
      },
      "m2.5": {
        "verify_rows": 3764,
        "relaxed": 1011,
        "failures": {},
        "proven": true
      },
      "m5.0": {
        "verify_rows": 4557,
        "relaxed": 1545,
        "failures": {},
        "proven": true
      }
    },
    "other_traffic_verify_rows": 64,
    "non_verify_rows": 1,
    "invalid_time_rows": 0,
    "overlapping_window_rows": 0,
    "excluded_probe_trial_trace": {
      "control": {
        "verify_rows": 0,
        "relaxed": 0
      },
      "m2.5": {
        "verify_rows": 26,
        "relaxed": 6
      },
      "m5.0": {
        "verify_rows": 16,
        "relaxed": 3
      }
    },
    "window_slack_s": 2.0
  },
  "issues": {},
  "pending_counts": {},
  "budget_deviation_arms": [],
  "unassigned_proxy_calls": 0
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
- Proxy t is recorded after forwarding/metrics; inferred windows are [t-wall_s,t], expanded by window_slack_s.
- Treatment labels own rows; windows check consistency. Control-window relaxation or treatment labels always fail.
- Calls outside selected arm/pass task spans are probes/trials, excluded by default and reported separately.
- --include-extra-calls includes these calls in pooled proxy metrics and activation only; paired task metrics remain task-only.
- Known treatment labels within selected pass spans must match their arm call windows; excluded probe/trial rows and other traffic are counted separately.
- Length-bound tasks have at least one proxy finish_reason=length; budgets must match for each paired task.
- Partial input may be an inconsistent live snapshot and is never promotion-ready.
