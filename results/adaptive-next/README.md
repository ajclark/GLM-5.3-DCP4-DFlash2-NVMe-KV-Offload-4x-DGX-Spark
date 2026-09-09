# Adaptive speculation follow-up evidence

These are actual four-Spark measurements and separately labelled sandbox
experiments from September 9, 2026. The goal is higher C1 coding/prose throughput
and lower active/loaded-idle power without expanding the resident model or KV
allocation. The [decision report](../../docs/ADAPTIVE-SPECULATION-NEXT.md) connects
these results to the architecture choices. Historical V4 evaluation data under
`results/adaptive-spec` remains frozen and is not replaced by these screens.

All declared experiments are complete. The original containers are restored and
generating; original GPU/CPU settings are verified, and owned experiment
controllers, watchdogs, samplers and the dedicated Pi session/watcher are
stopped. See the [restoration report](final-restoration-report.json).

## Reading the results

| Evidence | Location | Interpretation |
|---|---|---|
| Repaired absolute-position boundary | [boundary report](cache-width-r1/boundary-report.json) | Identical outputs across the old 90112-token draft-table limit, same model/KV capacity |
| Complete-cycle calibration | [repaired curve](cache-width-r1/costs-repaired-curve.json) | Measured short/100k/170k points; intermediate contexts are interpolation |
| Repaired 100k and 170k pairs | [100k summary](cache-width-r1/paired-100k/summary.json), [170k summary](cache-width-r1/paired-170k/summary.json) | About +4–5% coding and +10% prose; one prompt/domain/context, two repeats, not a promotion gate |
| Functional and target-logprob failures | [original diagnostic](cache-width-r1/quality-logprobs/logprob-divergence-report.json) | Failure also occurs under fixed K7; no quality promotion based on throughput alone |
| Fixed-K7 active power | [profile report](cache-width-r1/power-active/power-profile-report.json) | Four-prompt before/after brackets; 1600 MHz ratio 0.9537 tok/s and 0.8200 device J/token |
| Frozen-adaptive active power | [adaptive profile report](cache-width-r1/power-adaptive/power-profile-report.json) | 1600 MHz ratio 0.9290 tok/s and 0.8355 device J/token using the unchanged 2000 MHz cost curve |
| Loaded-idle power | [auto/governor](cache-width-r1/power-idle/power-profile-report.json), [explicit low clocks](cache-width-r1/power-idle-low/power-profile-report.json) | Baseline about 32 W, auto 47.08 W, 600 MHz 22.14 W, 300 MHz 21.56 W across four GPUs |
| Real Pi text requests | [live report](pi-hints-r1/live-pi-report.json) | Actual Pi/herdr code and prose completion with valid server hints; integration evidence |
| Explicit API activation | [activation gate](hints-conf-r2/activation/report.json) | Corrected true/false switches verified on the actual wire before comparisons |
| Corrected Pi hint controls | [hint report](pi-hints-r1/hint-controls-r2/request-control-report.json) | Correct-hint / inference-only ratios 0.9720 code, 1.0844 prose; two cases/two repeats, output variation retained |
| Confidence transport controls | [on/off report](pi-hints-r1/confidence-controls-r2/request-control-report.json) | 229 valid on packets, zero off; strict overhead bound unestablished |
| Coding quality of Pi controls | [quality report](pi-hints-r1/pi-control-quality-r2.json) | All twelve complete coding outputs pass |
| Real Pi tool transition | [tool-phase report](pi-tool-r1/tool-phase-report.json) | Actual read/edit fixes the fixture; both real tool-followups omit the domain prior |
| Causal confidence screen | [screen report](hints-conf-r2/confidence-screen-report.json) | 1050 learnable records; available scores are two steps old and the tested predictor loses to history |
| Isolated numerical flag control | [package difference](atomic-control-r1/package-difference-report.json), [paired result](atomic-control-r1/repeatability-report.json) | Atomic1/0 produce three/two output sequences and five/six passes in six requests each; disabling the flag is insufficient for repeatability |
| Request memory guards | [R1 guards](cache-width-r1/memory-screen-report.json), [follow-up guards](followup-memory-report.json) | Per-phase observations; loading and restoration have separate controller captures |

All energy figures use NVIDIA device sensors summed over four GPUs. They are
not CPU/package or wall measurements. Clock screens retain changing output
hashes and have no confidence interval. The post-idle generation followed a
settled baseline; it does not measure immediate wake latency. No permanent
frequency/governor or global Pi-hint policy was installed.

## Runtime and invalid comparisons

`cache-width-r1` used runtime `8f684a4`. The initial V5 transport experiment
`hints-conf-r1` used `542df7f`. Real Pi requests omitted explicit boolean switches
and activated the valid defaults. Its subsequent explicit HTTP on controls were
incorrectly disabled because the OpenAI schema normalizes booleans into integer
0/1. Those confidence/hint comparisons are preserved in their original directories
with `control-invalid.json`; **do not use their effect estimates**.

`hints-conf-r2` uses corrected runtime `f71eac0`. Its reporter requires actual
activation, C1 eligibility and intact telemetry. A first activation snapshot
preceded the one-second background-writer flush; the incomplete and completed
snapshots are both retained, and the original requests were not repeated.
The corrected results use fresh `*-controls-r2` directories.

The repaired cost curve predates the shadow collector. The causal confidence
screen uses that frozen atomic1 curve for same-boundary sensitivity, without
assuming zero collector overhead. It is not a live policy rollout, a V4 policy
reproduction or a measured throughput result. Current-proposal confidence is
explicitly noncausal at the existing host decision deadline. Atomic settings
must not be pooled into one calibration dataset.

## Archive and private evidence

`measurements.tar.gz` contains the curated source records, traces, power/memory
samples, frozen corpora and per-experiment scripts. Extract it from the repository
root:

```sh
tar -xzf results/adaptive-next/measurements.tar.gz
```

The archive paths are restricted to `results/adaptive-next/` and the four named
follow-up deployment directories under `results/adaptive-spec/`. It contains no
historical V4 replacement. Readable reports are also committed separately.
[ARTIFACT-PROVENANCE.json](ARTIFACT-PROVENANCE.json) records original and published
file hashes, transformations and archive identity. Embedded experimental source
hashes refer to original bytes; use that mapping after redaction.

Pi was invoked through herdr, with a 60-second native-state/pane watcher and a
separate memory guard for actual requests. Private Pi sessions, provider payloads,
full pane captures and Docker environment snapshots are excluded from publication.
Published Pi SSE records remove prompt token IDs/text after verifying a retained
prompt digest. Watcher publication retains only time, state and assessments.
Exact private originals are preserved locally in a restricted backup. Public
reports can reproduce paired arithmetic and verify opaque prompt identity; they
cannot reconstruct the private Pi system prompt for a fresh identical replay.

The reusable drivers and tests are under `bench/` and `tests/`. Archived per-run
scripts are historical records with portable path placeholders. Inference must
use the guarded controller, a healthy idle endpoint and a separately prepared
experiment; offline analysis does not require model access.
