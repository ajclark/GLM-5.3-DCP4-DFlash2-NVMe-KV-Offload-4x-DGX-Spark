import test from "node:test";
import assert from "node:assert/strict";
import { classifyWorkload, preparePayload, lossyControls } from "../extensions/pi-speculation-core.mjs";

test("intent hints abstain for mixed, ambiguous and tool-only inputs", () => {
  for (const [text, workload] of [
    ["Write a Python function for merging sorted streams.", "code_generate"],
    ["Fix the bug in this function.", "code_edit"],
    ["Review this Python module.", "code_review"],
    ["Write a short story about rain.", "prose"],
    ["Explain this code and refactor it.", "mixed"],
    ["What next?", "unknown"],
    ["Inspect the following:\n```python\n# write a story\n```", "unknown"],
  ]) assert.equal(classifyWorkload([{ role: "user", content: text }]).workload, workload);
  assert.equal(classifyWorkload([{ role: "tool", content: "Write Python code" }]).workload, "unknown");
  assert.equal(classifyWorkload([{ role: "user", content: "Fix this Python bug" }, { role: "tool", content: "OK" }]).phase, "tool_followup");
});

test("disabled hints preserve payload identity and cannot select policy or cap", () => {
  const body = { messages: [{ role: "user", content: "Write Python code" }], temperature: 0.6, vllm_xargs: { spec_policy: "off" } };
  assert.equal(preparePayload(body), body);
  const hint = preparePayload(body, { enabled: true });
  assert.equal(hint.messages, body.messages);
  assert.equal(hint.temperature, 0.6);
  assert.equal(hint.vllm_xargs.spec_policy, "off");
  assert.equal(hint.vllm_xargs.spec_verify_cap, undefined);
  assert.equal(body.vllm_xargs.spec_workload, undefined);
  assert.equal(hint.vllm_xargs.spec_workload, "code_generate");
});

test("explicit experiment token cap only lowers a supported positive limit", () => {
  assert.deepEqual(preparePayload({ max_tokens: 100 }, { maxTokens: 256 }), { max_tokens: 100 });
  assert.deepEqual(preparePayload({ max_completion_tokens: 32768 }, { maxTokens: 256 }), { max_completion_tokens: 256 });
  assert.deepEqual(preparePayload({}, { maxTokens: -1 }), {});
  assert.deepEqual(preparePayload({}, { maxTokens: 5000 }), {});
});

test("lossy controls are flat numerics, greedy-only, and fail closed", () => {
  assert.deepEqual(lossyControls({ margin: 1.5 }), { spec_lossy_margin: 1.5, spec_lossy_rank: 2, spec_lossy_min_p: 0 });
  assert.deepEqual(lossyControls({ margin: "2", minP: 0.1 }), { spec_lossy_margin: 2, spec_lossy_rank: 2, spec_lossy_min_p: 0.1 });
  for (const bad of [null, {}, { margin: 0 }, { margin: 6 }, { margin: NaN }, { margin: 1, minP: 0.5 }, { margin: 1, minP: -1 }]) {
    assert.equal(lossyControls(bad), null);
  }
  const greedy = { messages: [], temperature: 0, vllm_xargs: { spec_policy: "off" } };
  const out = preparePayload(greedy, { lossy: { margin: 1.0 } });
  assert.deepEqual(out.vllm_xargs, { spec_policy: "off", spec_lossy_margin: 1, spec_lossy_rank: 2, spec_lossy_min_p: 0 });
  assert.equal(greedy.vllm_xargs.spec_lossy_margin, undefined);
  const sampled = { messages: [], temperature: 0.6 };
  assert.equal(preparePayload(sampled, { lossy: { margin: 1.0 } }), sampled);
  const unset = { messages: [] };
  assert.equal(preparePayload(unset, { lossy: { margin: 1.0 } }), unset);
  assert.equal(preparePayload(greedy, { lossy: { margin: 9 } }), greedy);
});
