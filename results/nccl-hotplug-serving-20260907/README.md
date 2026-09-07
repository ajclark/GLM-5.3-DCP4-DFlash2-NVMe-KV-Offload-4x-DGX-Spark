# Serving stack under the NCCL hot-plug plugin (2026-09-07 window)

Lane `dcp2-hotplug-1` (= production DCP=2 lane with `NCCL_HOTPLUG=1`).

| step | result |
|---|---|
| rollout | healthy after 545 s, real generation OK, plugin loaded once (NCCL 2.31.2), 10 comms per rank |
| count100 baseline (3 runs) | 200 tokens, 4.93-4.98 s wall, TTFT 0.5 s |
| `spark-idle.sh --down`, model resident | all four ranks prepared (nothing in flight while idle), suspended, adapters off, 27 s |
| request while off | held by the plugin; `wanted=1` on all four ranks within 3 s; `/health` 200 throughout |
| `spark-idle.sh --up` | 14 s (adapters, ring verified, 10 comms per rank resumed) |
| the held request | first token 32 s after send (includes the operator's delay), then 200 tokens at normal speed |
| count100 after the cycle (3 runs) | 4.83-5.16 s wall: unchanged |
| watcher, IDLE_MIN=1 | spin-down 71 s after the last request; 10-minute hold; request at 06:01:00, wake seen 06:01:01, `--up` done 06:01:16, first token 15.0 s, 200 tokens in 19.4 s |
| 4 concurrent count100 after the wake | all correct, 9.3 s wall each |

Files: `count100.jsonl` (every measurement), `watcher-idle1min-test.log`,
`step5-down-1.log`, `step5-up-1.log`, `rollout-dcp2-hotplug-1.console.log`.
