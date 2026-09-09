// Pure request classification and metadata; no model calls or background work.
export function textContent(content) {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.filter(x => x?.type === "text" && typeof x.text === "string")
    .map(x => x.text).join("\n");
}

export function classifyWorkload(messages = []) {
  const lastUser = messages.findLast(m => m?.role === "user");
  // Examine intent before pasted blocks; file/tool contents are not instructions.
  const text = textContent(lastUser?.content).split("```")[0].slice(0, 4096).toLowerCase();
  const last = messages.at(-1);
  const phase = last?.role === "tool" || last?.role === "toolResult" ? "tool_followup" : "user_turn";
  const code = /\b(code|function|module|class|python|typescript|javascript|rust|sql|unit tests?|patch|refactor|debug|bug|compile|repository)\b/.test(text);
  const prose = /\b(explain|summari[sz]e|describe|essay|story|poem|prose|writeup|narrative|recap)\b/.test(text);
  let workload = "unknown";
  if (code && prose) workload = "mixed";
  else if (code) {
    if (/\b(fix|patch|refactor|edit|change|debug|bug)\b/.test(text)) workload = "code_edit";
    else if (/\b(review|inspect|audit)\b/.test(text)) workload = "code_review";
    else if (/\b(write|implement|create|build|complete|generate)\b/.test(text)) workload = "code_generate";
  } else if (prose) workload = "prose";
  return { workload, phase, strength: workload === "unknown" || workload === "mixed" ? "abstain" : "weak" };
}

export function preparePayload(payload, options = {}) {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return payload;
  const { enabled = false, maxTokens = null, label = null } = options;
  let out = payload;
  const changes = {};
  if (enabled) {
    const hint = classifyWorkload(payload.messages);
    changes.vllm_xargs = { ...payload.vllm_xargs, spec_workload: hint.workload,
      spec_phase: hint.phase, spec_hint_strength: hint.strength };
  }
  if (label) changes.vllm_xargs = { ...(changes.vllm_xargs ?? payload.vllm_xargs), spec_label: label.slice(0, 96) };
  if (Number.isInteger(maxTokens) && maxTokens > 0 && maxTokens <= 4096) {
    const key = "max_completion_tokens" in payload ? "max_completion_tokens" : "max_tokens";
    changes[key] = Math.min(Number.isInteger(payload[key]) && payload[key] > 0 ? payload[key] : maxTokens, maxTokens);
  }
  if (Object.keys(changes).length) out = { ...payload, ...changes };
  return out;
}
