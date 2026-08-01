import { defineFunction } from "@aws-amplify/backend";

/**
 * The thin proxy between the browser and the AgentCore runtime.
 *
 * It exists so the browser never invokes the agent directly: a Cognito-held credential must not
 * carry `bedrock-agentcore:InvokeAgentRuntime`. Auth stays where AppSync already enforces it,
 * and exactly one principal — this function's role — may reach the runtime.
 *
 * AGENT_RUNTIME_ARN is set after `agentcore deploy` prints it. Until then the function refuses
 * cleanly rather than failing at import.
 */
export const askAgent = defineFunction({
  name: "askAgent",
  entry: "./handler.ts",
  timeoutSeconds: 120, // an agent loop is several model calls; still far below the 15-min ceiling
  environment: {
    AGENT_RUNTIME_ARN: process.env.AGENT_RUNTIME_ARN ?? "",
    AGENT_REGION: "us-east-1",
  },
});
