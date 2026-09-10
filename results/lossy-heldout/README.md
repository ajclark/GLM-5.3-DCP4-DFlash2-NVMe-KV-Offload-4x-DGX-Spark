# Lever A held-out evaluation (lossy-20260910-r1, 2026-09-10 01:25-01:54 UTC)

Same boot as `results/lossy-dev/`. Fifteen `prose_*` prompts of
`bench/spec-heldout.json`, three repeats (the third with thinking on), 256
tokens, AB/BA counterbalanced, two arms: `fixed7` and the frozen
`lossy-m2.5`. 90 requests; no tuning on this set.

**Speed gate (pre-declared: paired lower bound >= +15%): passed.**
Paired geomean tok/s ratio **1.197, prompt-bootstrap 95% [1.170, 1.222]**;
every prompt positive (1.07-1.28). Pooled decode 15.93 -> 19.13 tok/s;
accepted/cycle 1.281 -> 1.725; relaxed accepts 0.217/cycle; TTFT 0.438 vs
0.439 s; p95 emission gap 0.151 s both; following-cycle first-position
acceptance 0.723 after a relaxed accept vs 0.728 after an exact one.
Activation proof holds for all 90 rows. `request-control-report.json`.

Quality so far: a blind, position-swapped screen of the 30 repeat-0/1 pairs
at 256 tokens (`../lossy-judge-256/`, judged by Claude without the manifest):
both sides coherent, on-prompt and fluent in every pair; 60 judgements:
54 ties, 2 lossy wins, 4 lossless wins (tie-adjusted score 0.48, win-or-tie
93%, all 30 swaps position-consistent; `score.json`). This screen detects only gross
loss at opening length; G3 (1024-token external judge), G4 (constrained
prose), G6 (2048-token drift) remain to be run, preferably against the
opt-in production switch. Not promoted to pi's prose persona yet.
