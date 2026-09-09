# Adaptive speculation follow-up experiments

The user authorized autonomous investigation, implementation and benchmarks across
the research avenues on September 9, 2026, using the sandbox VM first and observing
Spark memory pressure before and during experiments. Pi is installed locally and
must be invoked through herdr for the harness investigation.

Historical implementation and evidence: commit `340c7ae`. Research inventory and
experiment designs: commit `53daa7f`. The completed evaluation and calibration stay
frozen. New policies compare against the repaired version 4 runtime on new data. The
original long-context measurements remain preserved but are confounded by the
subsequently reproduced cache-table defect.

| Avenue | Evidence and current decision | Remaining experiment |
|---|---|---|
| Pi workload/phase hints | Real Pi/herdr wire path validated; opt-in extension and eight-observation server prior implemented; disabled-hint decisions match frozen V4 over 4,497 drifting steps | Actual Pi payload controls with correct, absent and wrong hints on the repaired server |
| Candidate confidence | Bounded previous-proposal collector committed as `542df7f`; policy unchanged | Guarded shadow boot, transport identity, overhead and causal calibration |
| Long-context acceptance | Repair `8f684a4` passes same-capacity Spark boundary probes; healthy 100k/170k acceptance; paired exploration shows approximately +4–5% code and +10% prose | Preserve the functional failure and investigate target repeatability before any quality promotion |
| Target-only K=0 | [State-machine and memory screen](research/K0-FEASIBILITY.md) completed; repaired acceptance weakens the original opportunity | Defer serving integration pending measured target-only/context-maintenance opportunity |
| Energy operating points | Four-prompt GPU/CPU screen and separately bracketed idle screen; watchdogs restore original clocks and governors | Finish idle controls and report device energy separately from CPU proxies |
| Selector training | [Parameter and optimizer-state screen](research/ARCHITECTURE-SCREENS.md) completed | Defer training until candidate coverage and path-choice errors justify it |
| Trees/copy/compressed memory/longer blocks | Exact CPU tree selector and source/kernel screens completed; copy opportunity is sparse in these repo prompts | Defer serving integration; no live gain inferred from CPU or synthetic screens |
| Target repeatability | [Source and causal-kernel diagnostic](research/REPEATABILITY-DIAGNOSTIC.md); functional failure also occurs at fixed K7 | Isolated atomic-off run with fresh prompt cache after the separate hint/confidence experiment |

Initial read-only cluster preflight: healthy and idle; available memory about
4895 MiB on the head and 5982–6276 MiB on workers. These are starting observations,
not admission thresholds for every later experiment. The guard retains its
phase-aware pressure limits and checks all four nodes.

The dedicated pi agent uses the installed `glm53/glm-5.3` OpenAI-compatible
connection. Its initial probes use explicit output limits and a fresh session;
the user's existing pi pane and global settings are preserved. Harness labels
are request metadata, never instructions to bypass target verification. They do
not enable a server whose adaptive policy is off.

Only one cluster workload/experimental deployment runs at a time. Capture source
hashes and a new experiment declaration before active comparisons. Keep each run
bounded with monitoring and restoration. A failed hypothesis is recorded as a
result; new training or larger allocations require a concrete fit and expected
benefit within the existing serving capacity.

## Follow-up decisions, 06:32 UTC

The old long-context acceptance collapse is confounded by a reproduced V2 cache
table defect. New experiment `cache-width-20260909-r1` deploys commit `8f684a4`
with the same TP4/DCP2, maxlen180224, maxseq12 and 6 GB/rank KV pool. Its
package is already frozen remotely; ongoing local hint changes are not in that
experiment. Before boot, head available memory was 4269 MiB and workers
5539–6090 MiB. The original containers are retained for exact restoration.

Pi's two initial requests reached the existing server with HTTP200 and weak
workload metadata. The code probe completed; the prose probe hit its explicit
256-token experimental limit. These validate the wire path only. A precision
first classifier abstains on mixed/unknown intent; negated code words can cause
unnecessary abstention and are not treated as evidence of a speedup.

The 12-prompt development-only leave-one-prompt-out screen found lower Brier
error for domain priors than a pooled prior: code 0.2041 to 0.1906, prose 0.1167
to 0.1114. The hypothetical same-boundary utility ratios are not measured
throughput. A local server prototype therefore uses hints only for the first
eight eligible observations, retains the full-width probe every 16 steps, and
then returns to ordinary inference-only priors. An adversarial low-confidence
hint test exposed persistent tail bias if hints were retained indefinitely;
the eight-observation limit removes that failure mode. No hint policy has yet
been deployed or promoted.

Power inventory confirms the existing idle-power documentation: there is no
CPU/package/wall energy sensor available through NVML or hwmon. The already
measured ConnectX-7 saving is a separate mechanism; its hotplug configuration
is unchanged during these inference experiments. Small GPU clock and governor
experiments must not be represented as new wall-power measurements.

At 06:43 UTC, the repaired Spark stack passed both fixed-seven boundary probes.
At 89055 tokens: TTFT 188.21 s, decode 42.77 tok/s; at 92056: TTFT 194.97 s,
decode 43.00 tok/s. Each accepted 56 draft tokens in 10 draft cycles and emitted
identical token IDs, with the marker retained and sequential counting correct.
These are two short diagnostic outputs, not a broad workload estimate.
The 100k and 170k repository coding/prose screens have completed. New calibration
artifacts are isolated under results/adaptive-next/cache-width-r1; historical
priors remain frozen and are not reused for the repaired long-context policy.

At 100k, one complete-function adaptive request failed closed-interval semantics.
Its trace scheduled cap 7 throughout. Fixed 7 and 3 initial outputs passed with
identical 285 token IDs; all 10 declared follow-up controls passed, but same-policy
fixed repeats also changed output IDs. The 13-request audit records 12 passes
and retains the original failure. A separate subsequent experiment captured
target logprobs; the root cause remains unresolved. This is not a clean
end-to-end quality pass.

The confidence transport is off by default. Source clones are bounded to 656 bytes
per packet plus 656 bytes pinned host storage, capped at 128 packets per request.
The existing AsyncOutput event owns the copies; the scheduler records real
decision and receipt times for causal offline evaluation. Confidence changes
no cap or target sampling decision in the first live experiment.


The subsequent six-request top-two-logprob diagnostic also reproduced the
interval failure under fixed cap 7. All six requests used cap 7 throughout.
At a shared prompt/output prefix, reported target rankings switched between
`list` and `tuple`, with gaps around 0.5–0.75; the later semantic branch also
changed with substantial margins. The target sampler source confirms these
are target-side logprobs. This is not established as a tiny numerical tie.
A source and kernel repeatability investigation is now tracked separately;
no adaptive or fixed policy receives a clean quality promotion on this evidence.

Repaired-runtime two-repeat paired screens show +4.59% code/+10.05% prose
at 100k and +4.13%/+10.16% at 170k versus fixed seven. Device decode energy
ratios are 0.892/0.852 at 100k and 0.911/0.861 at 170k. Each domain/context
uses only one prompt, so no prompt-level confidence interval is estimable.
These are exploration results, not replacements for a broad held-out gate.
The 100k first baseline also has longer TTFT than its warm controls; whole
request energy should be interpreted separately from decode energy.

The clock/governor screen is now running on the repaired runtime, with its
own independent rollback watchdogs. The subsequent idle-only screen and
confidence/hint boot remain pending. Pi was restarted into a fresh private
session on the same herdr pane; the watcher recorded its brief name gap and
then returned to `done_or_idle`. No Pi inference overlaps these benchmarks.

The complete local suite at commit `85cecd0` passes **769 tests**. The isolated
atomic control (`d43f62f`) changes one environment setting in the experiment's
packaged launch copy and requires fresh persisted KV; the ordinary launcher
remains unchanged. Source inspection establishes eligibility of 75 shared-expert
gate/up projections, not that atomics caused the observed output variation.

The completed frequency brackets in the still-running active screen give the
following four-prompt geometric-mean ratios against the nearest original-setting
baseline before and after: GPU 1600 MHz throughput 0.9537, decode device J/token
0.8200; 1800 MHz 0.9483/0.9080; 2200 MHz 0.9893/1.1919. These are small development
screens without confidence intervals, with output hashes retained to expose
changing work. They do not justify a permanent clock change. The separate idle
screen will measure GPU-clock auto mode and CPU governor proxies.
