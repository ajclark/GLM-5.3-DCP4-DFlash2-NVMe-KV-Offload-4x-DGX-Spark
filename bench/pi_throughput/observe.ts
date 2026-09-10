import { appendFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { createHash } from "node:crypto";

// Observe the native pi request without changing sampling, limits, or speculation.
export default function (pi: any) {
  const file = process.env.PI_BENCH_OBSERVATIONS;
  const emit = (event: string, data: any) => {
    if (file) appendFileSync(file, JSON.stringify({ event, epoch_ms: performance.timeOrigin + performance.now(), ...data }) + "\n");
  };
  pi.on("before_provider_request", (event: any, ctx: any) => {
    const p = event.payload;
    emit("request", {
      model: ctx.model?.id, provider: ctx.model?.provider,
      thinking: pi.getThinkingLevel(), temperature: p.temperature,
      top_p: p.top_p, max_tokens: p.max_tokens ?? p.max_completion_tokens,
      chat_template_kwargs: p.chat_template_kwargs, stream_options: p.stream_options,
      tools: p.tools?.map((t: any) => t.function?.name),
      messages_sha256: createHash("sha256").update(JSON.stringify(p.messages)).digest("hex"),
      message_count: p.messages?.length,
    });
  });
  pi.on("after_provider_response", (event: any) => emit("response", { status: event.status }));
  pi.on("message_update", (event: any) => {
    const e = event.assistantMessageEvent;
    if (e.type.endsWith("_delta") && e.delta) emit("delta", { kind: e.type, chars: e.delta.length });
  });
  pi.on("message_end", (event: any) => {
    if (event.message.role === "assistant") emit("completion", {
      usage: event.message.usage, stop_reason: event.message.stopReason,
      error: event.message.errorMessage,
      content_kinds: event.message.content.map((c: any) => c.type),
    });
  });
}
