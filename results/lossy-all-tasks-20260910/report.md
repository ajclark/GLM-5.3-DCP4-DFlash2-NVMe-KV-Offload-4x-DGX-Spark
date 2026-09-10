# Task-level bounded-lossy report

Task protocol: lossy_scope=all; pi thinking high; PI_HE_MAX_TOKENS=8192, reduced from production 32768. Observed budget deviations are reported below.

Status: **complete**. Activation: **proven**.

| Arm | Pass | Tasks | Passed | Errors | Wall s | Median task s | Output | Reasoning | Calls | Length calls/tasks | Decode tok/s | Accepted/cycle | Reasoning char share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| control | 1 | 12 | 11 | 0 | 122.9 | 8.748 | 3187 | 0 | 36 | 0/0 | 39.95 | 6.282 | 0.05268 |
| control | 2 | 12 | 12 | 0 | 175.3 | 10.15 | 4666 | 0 | 40 | 0/0 | 36 | 5.709 | 0.267 |
| control | pooled | 24 | 23 | 0 | 298.2 | 9.44 | 7853 | 0 | 76 | 0/0 | 37.51 | 5.928 | 0.1824 |
| m2.5 | 1 | 12 | 11 | 0 | 238.8 | 11.55 | 6985 | 0 | 45 | 0/0 | 37.07 | 5.923 | 0.5458 |
| m2.5 | 2 | 12 | 12 | 0 | 180.4 | 11.24 | 4894 | 0 | 47 | 0/0 | 38.2 | 6.051 | 0.2468 |
| m2.5 | pooled | 24 | 23 | 0 | 419.2 | 11.24 | 1.188e+04 | 0 | 92 | 0/0 | 37.53 | 5.975 | 0.4261 |
| m5.0 | 1 | 12 | 12 | 0 | 264.2 | 11.96 | 7600 | 0 | 53 | 0/0 | 37.13 | 5.904 | 0.4456 |
| m5.0 | 2 | 12 | 11 | 0 | 147.7 | 12.49 | 4087 | 0 | 47 | 0/0 | 42.89 | 6.79 | 0.07157 |
| m5.0 | pooled | 24 | 23 | 0 | 411.9 | 12.3 | 1.169e+04 | 0 | 100 | 0/0 | 38.96 | 6.186 | 0.3195 |

## m2.5 / control

Pass agreement: {"both_pass": 23, "control_only": 0, "treatment_only": 0, "both_fail": 1}.

| Ratio | Task geomean | Pair median | Bootstrap 95% | Undefined pairs |
|---|---:|---:|---|---:|
| wall_ratio | 1.3 | 1.129 | 1.065, 1.6 | 0 |
| reasoning_token_ratio | n/a | n/a | n/a | 24 |
| output_token_ratio | 1.257 | 1.294 | 1.017, 1.565 | 0 |

| Task | Pass | Control pass | Treatment pass | Wall ratio | Reasoning ratio | Output ratio | Budget | Length-bound C/T |
|---|---:|---|---|---:|---:|---:|---|---|
| HumanEval/0 | 1 | True | True | 2.066 | n/a | 2.055 | matched | False/False |
| HumanEval/100 | 1 | True | True | 1.739 | n/a | 1.562 | matched | False/False |
| HumanEval/122 | 1 | True | True | 0.597 | n/a | 0.2901 | matched | False/False |
| HumanEval/132 | 1 | False | False | 7.391 | n/a | 8.229 | matched | False/False |
| HumanEval/154 | 1 | True | True | 1.142 | n/a | 1.354 | matched | False/False |
| HumanEval/20 | 1 | True | True | 0.9108 | n/a | 0.9282 | matched | False/False |
| HumanEval/22 | 1 | True | True | 0.7056 | n/a | 0.7085 | matched | False/False |
| HumanEval/48 | 1 | True | True | 1.85 | n/a | 1.585 | matched | False/False |
| HumanEval/49 | 1 | True | True | 1.018 | n/a | 1.082 | matched | False/False |
| HumanEval/55 | 1 | True | True | 0.9842 | n/a | 1.04 | matched | False/False |
| HumanEval/69 | 1 | True | True | 2.037 | n/a | 2.043 | matched | False/False |
| HumanEval/75 | 1 | True | True | 1.116 | n/a | 1.186 | matched | False/False |
| HumanEval/0 | 2 | True | True | 1.663 | n/a | 1.304 | matched | False/False |
| HumanEval/100 | 2 | True | True | 1.337 | n/a | 1.284 | matched | False/False |
| HumanEval/122 | 2 | True | True | 1.954 | n/a | 1.957 | matched | False/False |
| HumanEval/132 | 2 | True | True | 0.7688 | n/a | 0.8152 | matched | False/False |
| HumanEval/154 | 2 | True | True | 1.305 | n/a | 1.447 | matched | False/False |
| HumanEval/20 | 2 | True | True | 1.054 | n/a | 1.336 | matched | False/False |
| HumanEval/22 | 2 | True | True | 0.7231 | n/a | 0.6397 | matched | False/False |
| HumanEval/48 | 2 | True | True | 1.871 | n/a | 1.585 | matched | False/False |
| HumanEval/49 | 2 | True | True | 0.959 | n/a | 0.8933 | matched | False/False |
| HumanEval/55 | 2 | True | True | 0.8942 | n/a | 0.8577 | matched | False/False |
| HumanEval/69 | 2 | True | True | 2.108 | n/a | 2.061 | matched | False/False |
| HumanEval/75 | 2 | True | True | 0.9928 | n/a | 0.9974 | matched | False/False |

## m5.0 / control

Pass agreement: {"both_pass": 22, "control_only": 1, "treatment_only": 1, "both_fail": 0}.

| Ratio | Task geomean | Pair median | Bootstrap 95% | Undefined pairs |
|---|---:|---:|---|---:|
| wall_ratio | 1.312 | 1.198 | 1.075, 1.606 | 0 |
| reasoning_token_ratio | n/a | n/a | n/a | 24 |
| output_token_ratio | 1.335 | 1.44 | 1.103, 1.614 | 0 |

| Task | Pass | Control pass | Treatment pass | Wall ratio | Reasoning ratio | Output ratio | Budget | Length-bound C/T |
|---|---:|---|---|---:|---:|---:|---|---|
| HumanEval/0 | 1 | True | True | 2.055 | n/a | 2.055 | matched | False/False |
| HumanEval/100 | 1 | True | True | 1.673 | n/a | 1.586 | matched | False/False |
| HumanEval/122 | 1 | True | True | 2.204 | n/a | 2.179 | matched | False/False |
| HumanEval/132 | 1 | False | True | 8.887 | n/a | 9.322 | matched | False/False |
| HumanEval/154 | 1 | True | True | 1.11 | n/a | 1.422 | matched | False/False |
| HumanEval/20 | 1 | True | True | 0.7065 | n/a | 0.7259 | matched | False/False |
| HumanEval/22 | 1 | True | True | 0.6477 | n/a | 0.6268 | matched | False/False |
| HumanEval/48 | 1 | True | True | 1.865 | n/a | 1.585 | matched | False/False |
| HumanEval/49 | 1 | True | True | 1.023 | n/a | 1.082 | matched | False/False |
| HumanEval/55 | 1 | True | True | 0.9848 | n/a | 0.99 | matched | False/False |
| HumanEval/69 | 1 | True | True | 1.79 | n/a | 1.758 | matched | False/False |
| HumanEval/75 | 1 | True | True | 1.101 | n/a | 1.188 | matched | False/False |
| HumanEval/0 | 2 | True | True | 2.068 | n/a | 2.055 | matched | False/False |
| HumanEval/100 | 2 | True | True | 1.719 | n/a | 1.648 | matched | False/False |
| HumanEval/122 | 2 | True | True | 2.296 | n/a | 2.549 | matched | False/False |
| HumanEval/132 | 2 | True | False | 0.2014 | n/a | 0.2066 | matched | False/False |
| HumanEval/154 | 2 | True | True | 1.285 | n/a | 1.443 | matched | False/False |
| HumanEval/20 | 2 | True | True | 1.085 | n/a | 1.437 | matched | False/False |
| HumanEval/22 | 2 | True | True | 0.8442 | n/a | 0.8162 | matched | False/False |
| HumanEval/48 | 2 | True | True | 1.875 | n/a | 1.585 | matched | False/False |
| HumanEval/49 | 2 | True | True | 0.9877 | n/a | 1.013 | matched | False/False |
| HumanEval/55 | 2 | True | True | 0.8538 | n/a | 0.8261 | matched | False/False |
| HumanEval/69 | 2 | True | True | 1.837 | n/a | 1.775 | matched | False/False |
| HumanEval/75 | 2 | True | True | 0.9504 | n/a | 0.9474 | matched | False/False |

## Activation and guards

```json
{
  "scope": "all",
  "labels": {
    "m2.5": "all-tasks-m2.5",
    "m5.0": "all-tasks-m5.0"
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
      "proxy_calls": 0,
      "proxy_api_calls": 0,
      "length_calls": 0,
      "length_bound_tasks": 0,
      "pooled_decode_tok_s": null,
      "proxy_generation_tokens": 0,
      "server_decode_s": 0,
      "spec_drafts": 0,
      "spec_accepted_tokens": 0,
      "accepted_per_cycle": null,
      "reasoning_chars_share": null,
      "completion_budgets": [],
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
      "pooled_decode_tok_s": 40.019762845849804,
      "proxy_generation_tokens": 243,
      "server_decode_s": 6.072,
      "spec_drafts": 41,
      "spec_accepted_tokens": 202,
      "accepted_per_cycle": 5.926829268292683,
      "reasoning_chars_share": 0.0,
      "completion_budgets": [
        300
      ],
      "unknown_budget_calls": 0,
      "thinking_not_confirmed_calls": 1
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
      "pooled_decode_tok_s": 44.50084602368866,
      "proxy_generation_tokens": 263,
      "server_decode_s": 5.91,
      "spec_drafts": 40,
      "spec_accepted_tokens": 223,
      "accepted_per_cycle": 6.575,
      "reasoning_chars_share": 0.0,
      "completion_budgets": [
        300
      ],
      "unknown_budget_calls": 0,
      "thinking_not_confirmed_calls": 1
    }
  },
  "activation": {
    "status": "proven",
    "arms": {
      "control": {
        "verify_rows": 1325,
        "relaxed": 0,
        "failures": {},
        "proven": true
      },
      "m2.5": {
        "verify_rows": 1992,
        "relaxed": 436,
        "failures": {},
        "proven": true
      },
      "m5.0": {
        "verify_rows": 1893,
        "relaxed": 489,
        "failures": {},
        "proven": true
      }
    },
    "other_traffic_verify_rows": 63,
    "non_verify_rows": 1,
    "invalid_time_rows": 0,
    "overlapping_window_rows": 0,
    "excluded_probe_trial_trace": {
      "control": {
        "verify_rows": 0,
        "relaxed": 0
      },
      "m2.5": {
        "verify_rows": 41,
        "relaxed": 7
      },
      "m5.0": {
        "verify_rows": 40,
        "relaxed": 11
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
