# Lever A development sweep (lossy-20260910-r1, 2026-09-10 00:49-01:24 UTC)

Guarded experiment boot `lossy-20260910-r1` (DCP2/180224/6e9/slab lane, V2
overlay set plus the three lossy-verification overlays, `GLM_SPEC_LOSSY=1`,
`GLM_SPEC_LOSSY_CHECK=1`, `GLM_SPEC_POLICY=shadow`, trace on). Six `prose_*`
prompts of `bench/spec-development.json`, two repeats, 256 tokens, thinking
off, nine interleaved arms: `fixed7` (lossless control) and
`lossy-m{0.5,1.0,1.5,2.5}[-p0.1]` (MARS margin rule, rank 2, optional
p_min 0.1). 108 requests. Report: `request-control-report.json` (activation
proof from the rank-0 trace: every lossy row echoes its margin and shows
`relaxed > 0`; every control row `relaxed = 0`).

| variant | paired geomean tok/s ratio | prompt-bootstrap 95% | relaxed/cycle | next-cycle p1 acceptance after relaxed vs exact |
|---|---:|---|---:|---|
| lossy-m0.5 | 1.021 | [0.996, 1.045] | 0.051 | 0.589 / 0.653 |
| lossy-m0.5-p0.1 | 1.008 | [0.980, 1.036] | 0.049 | 0.600 / 0.662 |
| lossy-m1.0 | 1.056 | [1.035, 1.078] | 0.095 | 0.679 / 0.680 |
| lossy-m1.0-p0.1 | 1.088 | [1.068, 1.106] | 0.102 | 0.717 / 0.701 |
| lossy-m1.5 | 1.088 | [1.072, 1.103] | 0.128 | 0.714 / 0.698 |
| lossy-m1.5-p0.1 | 1.061 | [1.051, 1.072] | 0.104 | 0.649 / 0.703 |
| **lossy-m2.5** | **1.126** | **[1.069, 1.183]** | 0.215 | 0.743 / 0.713 |
| lossy-m2.5-p0.1 | 1.112 | [1.055, 1.170] | 0.163 | 0.627 / 0.726 |

Frozen before the held-out set: `FROZEN-VARIANT` = `lossy-m2.5` (highest
ratio; the following-cycle acceptance shows no selector-interaction penalty;
the p_min floor never helped). First-divergence positions are early
(8-19 tokens) and first-64 token agreement low, as expected once a relaxed
token changes the trajectory; quality is judged by the gates in
`docs/LOSSY-VERIFICATION-PLAN.md` §5, not by agreement. Single boot, one
prompt set; device energy not sampled.
