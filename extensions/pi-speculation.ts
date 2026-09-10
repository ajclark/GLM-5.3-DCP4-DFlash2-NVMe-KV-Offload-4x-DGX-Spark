import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { classifyWorkload, preparePayload } from "./pi-speculation-core.mjs";

// Opt-in experiment extension. Loading it alone does not change requests.
export default function (pi: ExtensionAPI) {
  let enabled = process.env.PI_SPEC_HINTS === "1";
  // Bounded-lossy verification: off unless PI_SPEC_LOSSY_MARGIN is set (the
  // prose persona) or /spec-lossy on is issued; never sent for sampled requests.
  let lossy: { margin: number; minP: number } | null = process.env.PI_SPEC_LOSSY_MARGIN
    ? { margin: Number(process.env.PI_SPEC_LOSSY_MARGIN), minP: Number(process.env.PI_SPEC_LOSSY_MIN_P ?? "0") }
    : null;
  const cap = Number(process.env.PI_SPEC_MAX_TOKENS ?? "0");
  const log = process.env.PI_SPEC_LOG;
  const captureDir = process.env.PI_SPEC_CAPTURE_DIR;
  let serial = 0;
  let current: Record<string, any> | undefined;
  const record = (row: object) => {
    if (log) appendFileSync(log, JSON.stringify({ time: Date.now() / 1000, ...row }) + "\n", { mode: 0o600 });
  };
  pi.registerCommand("spec-hints", {
    description: "Set experimental GLM request hints: on or off",
    handler: async (args, ctx) => {
      if (!["on", "off"].includes(args.trim())) { ctx.ui.notify("Usage: /spec-hints on|off", "warning"); return; }
      enabled = args.trim() === "on";
      ctx.ui.notify(`Speculation request hints ${enabled ? "on" : "off"}`);
    },
  });
  pi.registerCommand("spec-lossy", {
    description: "Bounded-lossy prose verification: on [margin [minP]] or off",
    handler: async (args, ctx) => {
      const parts = args.trim().split(/\s+/);
      if (parts[0] === "off") { lossy = null; ctx.ui.notify("Lossy verification off"); return; }
      if (parts[0] !== "on") { ctx.ui.notify("Usage: /spec-lossy on [margin [minP]] | off", "warning"); return; }
      const margin = parts[1] ? Number(parts[1]) : Number(process.env.PI_SPEC_LOSSY_MARGIN ?? "1.0");
      const minP = parts[2] ? Number(parts[2]) : Number(process.env.PI_SPEC_LOSSY_MIN_P ?? "0");
      lossy = { margin, minP };
      ctx.ui.notify(`Lossy verification on: margin ${margin} nats, minP ${minP} (greedy requests only)`);
    },
  });
  pi.on("before_provider_request", (event, ctx) => {
    current = undefined;
    if (ctx.model?.provider !== "glm53" || ctx.model?.id !== "glm-5.3") return;
    if (!enabled && !cap && !log && !captureDir && !lossy) return;
    const original: any = event.payload;
    const label = log || captureDir ? `pi-spec-${Date.now()}-${++serial}` : null;
    const payload = preparePayload(original, { enabled, maxTokens: cap || null, label, lossy });
    // Explicit local experiment capture, including system/tool context. Keep
    // the directory private and excluded from published benchmark artifacts.
    if (captureDir) {
      mkdirSync(captureDir, { recursive: true, mode: 0o700 });
      writeFileSync(join(captureDir, `${label}.json`), JSON.stringify(payload) + "\n", { mode: 0o600, flag: "wx" });
    }
    current = { event: "request", label, started_at: Date.now() / 1000,
      hint: classifyWorkload(original?.messages), hints_enabled: enabled, lossy,
      prompt_sha256: createHash("sha256").update(JSON.stringify(original?.messages ?? [])).digest("hex"),
      model: ctx.model.id, thinking_level: pi.getThinkingLevel(),
      temperature: payload.temperature ?? null, max_tokens: payload.max_completion_tokens ?? payload.max_tokens,
      xargs: payload.vllm_xargs ?? {}, first_delta_at: null, last_delta_at: null, deltas: 0 };
    record(current);
    if (payload !== original) return payload;
  });
  pi.on("after_provider_response", event => {
    if (current) record({ event: "response", label: current.label, status: event.status });
  });
  pi.on("message_update", event => {
    if (!current || !event.assistantMessageEvent.type.endsWith("_delta")) return;
    const now = Date.now() / 1000;
    current.first_delta_at ??= now;
    current.last_delta_at = now;
    current.deltas++;
  });
  pi.on("message_end", event => {
    if (!current || event.message.role !== "assistant") return;
    record({ ...current, event: "completion", usage: event.message.usage,
      stop_reason: event.message.stopReason, output_sha256: createHash("sha256")
        .update(JSON.stringify(event.message.content)).digest("hex") });
    current = undefined;
  });
}
