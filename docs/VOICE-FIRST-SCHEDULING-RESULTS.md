# Voice-first scheduling — results (2026-09-30)

Voice-client requests (priority −10) no longer queue behind other requests' prefills.
Code: `runtime/vllm029/verify_cap_overlay/vllm/v1/core/sched/voice_first.py`
(test: `runtime/vllm029/verify_cap_overlay/test_voice_first.py`). Evidence:
`results/voice-first-20260930/contention.jsonl`. The client side (sending the priority) is not part of this
repository; the measurement client used a private voice app's configuration and is not published.

Measured on an abliterated graft of GLM-5.3 with the same quantization envelope as GLM-5.3
Int4-Int8Mix (Int4 group-128 routed experts, Int8 group-128 elsewhere, identical shapes).

## Problem

vLLM serves running requests before waiting ones, and a running prefill takes the whole
2048-token step budget. A voice turn that arrives while an agent's long prompt is
prefilling waits for all of it. vLLM's `--scheduling-policy priority` does not help: it only
orders the waiting queue and chooses preemption victims.

## Change

`VoiceFirstAsyncScheduler` subclasses the stock `AsyncScheduler`. While any request with
priority < 0 is running or waiting, it holds every other request's prefill work for the
step: running prefill chunks are skipped (through the eligibility gate the V2 runner
already has), and non-urgent waiting requests are hidden from admission. Other requests'
decodes continue. The held work resumes on the first step without an urgent request. When
no request has a negative priority, it behaves exactly like the stock scheduler, and
`priority` 0 requests order FCFS as before.

`launch.sh` passes `--scheduling-policy priority --scheduler-cls
vllm.v1.core.sched.voice_first.VoiceFirstAsyncScheduler` by default (`VOICE_FIRST=0`
restores the stock scheduler). It is in the same overlay image as the verification cap:
`spark-vllm:0.29.0-nvme4-stridefix-kvtier-verifycap5-voicefirst-20260930`. It is not in
`manifest.json`, so the NVMe slab salt and the warm slab are unchanged.

## Results

Voice request: a voice-assistant system prompt with its tool list, "Why is the sky
blue?", 60 tokens. Agent request: an uncached ~60K-token prompt (unique text, so no GPU or
NVMe prefix hits), started 5–8 s before the voice request.

| Scheduler | Voice time to first token | Agent total |
|---|---|---|
| Stock FCFS (production before) | **98.5 s** (waited for the whole agent prefill) | 102.6 s |
| Voice-first, priority −10 | **6.0 s**, 6.0 s | 106.3 s, 105.3 s |

Idle, the same voice request takes ~0.75 s to first token. Of the remaining ~6 s under
contention (per-request metrics in `results/voice-first-20260930/contention.jsonl`): ~1.7 s passes before the engine
core even accepts the request (it reads new requests between steps), ~3.5 s is queue
time behind the agent chunk the async scheduler had already submitted, and 0.75 s is the
voice prompt's own prefill. That is roughly 2.5 agent prefill chunks already in flight,
which cannot be interrupted. The agent pays about the voice turn's duration (+3–4 s).

## Limits and next steps

- The residual wait scales with agent prefill chunk time (up to ~3.6 s per 2048-token chunk
  at long context). Smaller chunks for non-urgent prefills would shrink it at some cost in
  agent prefill throughput (not measured yet).
- Urgent requests preempt nothing while there is KV room. When KV is short, vLLM's priority
  policy preempts the lowest-priority running request, which is the intended order.
- Any client can send a negative priority.
- Separate finding: a repeated identical voice prompt (638 tokens) hits only 384 tokens
  in the prefix caches (256 GPU + 128 NVMe), so every voice turn re-prefills ~250 tokens,
  ~0.75 s. Fixing that is the next latency lever for voice. The prefix-hit fix in
  `results/step1-batchcap-20260930/REPORT.md` later raised GPU hits on short prompts by 128–256 tokens.
