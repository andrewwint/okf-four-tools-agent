# The agent runtime's execution role

**Why these files exist.** A security review found that the only thing preventing the deploy
toolkit from scaffolding `arn:aws:bedrock:*::foundation-model/*` — every model, in every region —
was a *comment* in `amplify/backend.ts`. A comment in a file that does not create the role.
So the policy lives here as JSON, and there is a command to verify it after deploy.

**Why the role is the whole blast radius.** AgentCore delivers the execution role's credentials
into the runtime sandbox via the instance metadata service, so *any* code execution inside the
runtime — including a fully prompt-injected agent — holds this role. With exactly this policy,
the worst a compromised agent can do is invoke one model, retrieve one knowledge base, and fetch
headlines. That containment is a design property worth stating out loud.

**Four model ARNs, still one model.** `us.anthropic.claude-sonnet-4-5-...` is a *cross-region
inference profile* that routes to us-east-1, us-east-2 and us-west-2. Bedrock authorises against
the profile **and** the backing foundation-model ARN in whichever region it lands. Granting only
us-east-1 produces intermittent `AccessDeniedException` under load — which is exactly the moment
someone reaches for a wildcard. Four explicit ARNs is the correct narrowing, not a compromise.

Switching to Haiku 4.5: replace all four ARNs with
`anthropic.claude-haiku-4-5-20251001-v1:0`, same three regions.

**Deliberately absent** — verify these stay absent: any `ssm:*` (this is the second reason the
agent cannot read the news API key, the first being that no code references it), any `iam:*`,
any S3 action, and `bedrock-agentcore:InvokeAgentRuntime` (the agent must not invoke itself).

## Apply and verify

```bash
export AWS_PROFILE=AdministratorAccess-790768631355
ROLE=<the role name agentcore created>

aws iam put-role-policy --role-name "$ROLE" \
  --policy-name okf-four-tools-least-privilege \
  --policy-document file://infra/agentcore-execution-policy.json

# Verify — this is the step that catches a scaffolded wildcard.
aws iam list-role-policies --role-name "$ROLE"
aws iam list-attached-role-policies --role-name "$ROLE"
aws iam get-role-policy --role-name "$ROLE" --policy-name okf-four-tools-least-privilege \
  | grep -c 'foundation-model/\*' # must print 0
```
