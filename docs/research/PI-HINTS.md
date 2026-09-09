# Pi workload hints: an opt-in C1 cold-start experiment

The Pi extension sends a weak workload label (`code_generate`, `code_edit`,
`code_review`, `prose`, `mixed` or `unknown`) and a user/tool phase through
vLLM's flat `vllm_xargs` request metadata. It changes neither the prompt nor
target verification. Mixed and unknown intent abstain; a pasted code block
does not override the user's opening instruction. It is a conservative rule
classifier, not a claim to recognize every task or phase.

## Server behavior

Version 5 optionally loads a small server-owned prior table from
`GLM_SPEC_HINT_PRIORS`, with exact TP/DCP/model-length/draft-capacity validation,
a measured context range, two domains and prior strength 2. Request metadata
cannot supply domain probabilities. The ordinary experimental cost table is
still required and must cover the current context. Off/shadow/fixed modes,
non-greedy requests, prefill, C2 and other existing exclusions retain their
behavior.

For a valid weak label on a user turn, the controller can choose among caps
1/3/5/7 before the original eight-observation warm-up finishes. The prior has
weight 2, contributes no fictitious trials, and is used only until eight
eligible observations arrive. The full-width probe every 16 steps remains.
After warm-up, ordinary inference-only priors and live acceptance take over.
`spec_use_hints=false` provides an experimental control. Missing, mixed,
unknown, malformed and tool-followup hints abstain.

An adversarial prior test showed that retaining a very wrong domain prior
indefinitely can bias seldom-observed tails despite ordinary feedback. The
bounded warm-up lifetime is deliberate. It also limits the maximum benefit:
long requests may gain very little, while short bursts are the natural test.

## Local Pi invocation

The user requested Pi through herdr. A dedicated sibling agent named
`spec-pi` uses the installed `glm53/glm-5.3` provider and a private experimental
session directory. Existing Pi panes and global settings are preserved. Load
`extensions/pi-speculation.ts` with Pi's `-e` option. Environment controls:

- `PI_SPEC_HINTS=1` enables labels; `/spec-hints on` and `/spec-hints off`
  toggle them during the session. Loading the extension alone is inert.
- `PI_SPEC_MAX_TOKENS` explicitly bounds an experiment's output to at most
 4096 tokens and only lowers an existing limit. It is independent of labels.
- `PI_SPEC_LOG` records request/completion hashes, timing and usage.
- `PI_SPEC_CAPTURE_DIR` explicitly captures private request payloads for
  controlled replay, with directory/file modes 0700/0600. These include Pi's
  system context and stay in ignored local results, outside publication.

`bench/spec_herdr_watch.py --agent spec-pi --out <private-jsonl> --interval 60`
reads native state and 80 lines of pane text every 60 seconds, with a bounded
duration. Native state remains in the record; textual error or blockage
matches are review candidates because scrollback may contain an old error.
It does not auto-approve anything. The separate `spec_pi_monitor.py` observes
all four nodes and cancels only the named experimental Pi agent if memory
pressure or its bounded duration trips.

## Evidence so far

Two actual Pi requests reached the original Spark endpoint with HTTP 200:
a Python function completed, and a prose probe hit its explicit 256-token
limit. They establish the wire path only. The original server policy was off,
so these requests cannot demonstrate adaptive performance.

A development-only leave-one-prompt-out screen used 12 prompts and 1057
full-seven eligible events. Domain priors reduced mean Brier error from
0.2041 to 0.1906 for coding and 0.1167 to 0.1114 for prose. Wrong-domain controls
were worse. Same-boundary utility is not a rollout or measured tok/s; the
classifier's abstentions are separate from these known corpus labels.

The server implementation and eligibility/adversarial tests pass locally.
Repaired-runtime short calibration, frozen hint/control comparisons and real
Pi payload replay are required before claiming benefit or enabling the
extension globally. The historical 100k/170k priors are excluded following
the replicated-cache table defect.

The subsequent V5 experiment ran two more **actual Pi/herdr** requests against
the repaired server. Both finished naturally with HTTP 200: 262 output tokens
for four Python functions, all functional checks passing, and 174 output
tokens for a complete harbor-rain scene. Server traces identify the requested
domains during the bounded warm-up. The code request used K7 throughout;
the prose request used 85 K3 and five K7 verification rows. All 125 collected
confidence packets across these two requests passed identity validation.
See the [live Pi report](../../results/adaptive-next/pi-hints-r1/live-pi-report.json).
These two requests demonstrate integration, not a paired speedup.

The first explicit on/off HTTP replay exposed a separate API-boundary error:
the live `vllm_xargs` schema normalizes JSON booleans into integer 0/1, while
the new switches initially checked Python `is True`. Explicit true controls
therefore stayed disabled. Both affected comparisons are retained and marked
invalid; no overhead or hint-effect claim may use them. The actual Pi requests
above omitted the switches and used their valid defaults, so their activation
evidence remains valid. Commit `f71eac0` accepts exactly boolean true or integer
one, tests the captured OpenAPI boundary, and makes the paired reporter require
server traces proving each requested treatment activated. Corrected replay
results must come from fresh experiment directories.
