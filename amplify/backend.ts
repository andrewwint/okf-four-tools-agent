import { defineBackend } from "@aws-amplify/backend";
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";
import { auth } from "./auth/resource";
import { data } from "./data/resource";
import { askAgent } from "./functions/askAgent/resource";
import { getNews } from "./functions/getNews/resource";

const backend = defineBackend({ auth, data, askAgent, getNews });

const REGION = "us-east-1";
// Set after `agentcore deploy` prints the runtime ARN. Empty until then, which keeps the grant
// unresolvable rather than wildcarded — a missing ARN must fail closed, never open.
const AGENT_RUNTIME_ARN = process.env.AGENT_RUNTIME_ARN ?? "";

/**
 * The proxy may invoke exactly ONE agent runtime, and nothing else may invoke it at all.
 *
 * `"Resource": "*"` is rejected outright by AgentCore resource policies, and an identity policy
 * should not wildcard the runtime either: the point of the proxy is that a single principal
 * crosses this boundary.
 */
if (AGENT_RUNTIME_ARN) {
  backend.askAgent.resources.lambda.addToRolePolicy(
    new PolicyStatement({
      effect: Effect.ALLOW,
      actions: ["bedrock-agentcore:InvokeAgentRuntime"],
      resources: [AGENT_RUNTIME_ARN, `${AGENT_RUNTIME_ARN}/runtime-endpoint/*`],
    })
  );
}

/**
 * Deliberately absent, and each absence is a finding from an independent security review:
 *
 *  - No `bedrock:*` on either Cognito role. A browser-held credential invoking a paid Bedrock
 *    resource is precisely the invariant the proxy exists to uphold; a sibling project granted
 *    it to `authenticatedUserIamRole` and that is the anti-pattern being avoided here.
 *  - No `bedrock:InvokeModel` on `askAgent`. The proxy does not call a model; the agent does.
 *  - No `lambda:InvokeFunction` on `getNews` here. That grant belongs to the AGENT's execution
 *    role, scoped to this one function ARN, and is applied on the AgentCore side.
 *
 * The agent runtime's own execution role must likewise be narrowed to the specific model,
 * knowledge base, and Lambda ARNs it uses. The CDK scaffold generates a wildcard
 * (`arn:aws:bedrock:*::foundation-model/*`) — that was exposure X2 and it does not ship.
 */

backend.addOutput({
  custom: {
    agentRuntimeArn: AGENT_RUNTIME_ARN,
    region: REGION,
  },
});
