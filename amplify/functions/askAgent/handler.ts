import type { Schema } from "../../data/resource";
import {
  BedrockAgentCoreClient,
  InvokeAgentRuntimeCommand,
} from "@aws-sdk/client-bedrock-agentcore";
import { env } from "$amplify/env/askAgent";

/**
 * Browser -> AppSync (Cognito) -> this function -> InvokeAgentRuntime (SigV4) -> the agent.
 *
 * The only job here is to cross the trust boundary safely. Deliberately NOT here: any decision
 * about what the agent may answer. That lives in the agent and in the OKF capability concept,
 * so there is one place to reason about grounding rather than two.
 */

const MAX_QUESTION_CHARS = 600;

export const handler: Schema["askAgent"]["functionHandler"] = async (event) => {
  const question = (event.arguments.question ?? "").trim();

  if (!question) {
    return { answer: "No question provided.", answered: false, mode: "rejected" };
  }
  // Bound the input before it reaches a metered model, not after.
  if (question.length > MAX_QUESTION_CHARS) {
    return {
      answer: `Question too long (limit ${MAX_QUESTION_CHARS} characters).`,
      answered: false,
      mode: "rejected",
    };
  }
  if (!env.AGENT_RUNTIME_ARN) {
    return {
      answer: "The agent runtime is not configured yet.",
      answered: false,
      mode: "unconfigured",
    };
  }

  const client = new BedrockAgentCoreClient({ region: env.AGENT_REGION });

  try {
    const response = await client.send(
      new InvokeAgentRuntimeCommand({
        agentRuntimeArn: env.AGENT_RUNTIME_ARN,
        // A fresh session per request: no conversation state crosses between callers.
        runtimeSessionId: crypto.randomUUID(),
        payload: new TextEncoder().encode(JSON.stringify({ question })),
      })
    );

    const raw = await collect(response.response);
    const parsed = safeParse(raw);
    return {
      answer: parsed?.answer ?? raw ?? "The agent returned nothing.",
      answered: parsed?.answered ?? Boolean(raw),
      mode: parsed?.mode ?? "agent",
    };
  } catch (error) {
    // Never surface a stack trace or an SDK message to the caller: it leaks internals and, in
    // an agent context, becomes text the model may treat as instructions. Log it instead.
    console.error("askAgent failed", error);
    return {
      answer: "The agent is unavailable right now. Please try again.",
      answered: false,
      mode: "error",
    };
  }
};

/** The runtime streams its reply; reassemble it. */
async function collect(body: unknown): Promise<string> {
  if (!body) return "";
  if (typeof body === "string") return body;

  const chunks: string[] = [];
  const decoder = new TextDecoder();
  for await (const chunk of body as AsyncIterable<Uint8Array>) {
    chunks.push(decoder.decode(chunk, { stream: true }));
  }
  const text = chunks.join("");
  // Server-sent events arrive as `data: {...}` lines.
  return text
    .split("\n")
    .map((line) => (line.startsWith("data:") ? line.slice(5).trim() : line))
    .join("")
    .trim();
}

function safeParse(text: string): { answer?: string; answered?: boolean; mode?: string } | null {
  try {
    const value = JSON.parse(text);
    return typeof value === "object" && value !== null ? value : null;
  } catch {
    return null;
  }
}
