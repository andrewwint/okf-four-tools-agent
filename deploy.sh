#!/usr/bin/env bash
# Deploy the agent runtime — build, deploy, and re-apply what CDK reverts.
#
# Use this instead of `agentcore deploy`. Every deploy regenerates the runtime's CDK-managed
# configuration, which silently undoes two things that matter:
#
#   1. the least-privilege execution role. The scaffold emits
#      arn:aws:bedrock:*:<account>:inference-profile/* — every model, in every region — and
#      grants neither bedrock:Retrieve nor lambda:InvokeFunction, so two of the four tools
#      cannot work at all. A security review predicted this before the first deploy and it
#      happened exactly as described.
#   2. the runtime environment variables. Without them the knowledge-base and news tools report
#      themselves unconfigured. They fail closed and say so, which is the right failure mode,
#      but it means a "successful" deploy quietly runs on two tools.
#
# Both were re-applied by hand three times before this script existed. Remembering is not a
# control: the failure mode is a deploy that looks fine and runs with a wildcard role.
set -euo pipefail

: "${AWS_PROFILE:?set AWS_PROFILE}"
REGION="${AWS_REGION:-us-east-1}"
RUNTIME_ID="${OKF_RUNTIME_ID:-okffourtools_okffourtools-fyaxyV9GEz}"
MODEL="${OKF_MODEL_ID:-us.anthropic.claude-haiku-4-5-20251001-v1:0}"
KB_ID="${OKF_KB_ID:-U78LLZTZEL}"
NEWS_FN="${OKF_NEWS_FUNCTION:-amplify-okffourtoolsagent-an-getNewslambda832EFEA8-TYRI8OaXz1QH}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"

echo "==> build (stages dist/ from the concept; refuses to ship .env or a quarantined concept)"
./.venv/bin/python build.py

echo "==> tests"
./.venv/bin/python -m pytest tests/ -q

echo "==> deploy"
agentcore deploy --yes

echo "==> re-apply least-privilege execution role (CDK reverts this every deploy)"
role="$(aws iam list-roles \
  --query "Roles[?contains(RoleName,'ApplicationAgentOkffourto')].RoleName" \
  --output text | head -1)"
[ -n "$role" ] || { echo "could not find the execution role" >&2; exit 1; }
aws iam put-role-policy --role-name "$role" \
  --policy-name okf-four-tools-least-privilege \
  --policy-document "file://infra/agentcore-execution-policy.json"

echo "==> narrow the scaffold's model wildcard"
python3 - "$role" "$MODEL" <<'PY'
import json, os, re, subprocess, sys
role, profile = sys.argv[1], sys.argv[2]
model = re.sub(r"^(us|global)\.", "", profile)          # profile -> backing foundation model
acct = os.environ.get("AWS_ACCOUNT") or subprocess.run(
    ["aws", "sts", "get-caller-identity", "--query", "Account", "--output", "text"],
    capture_output=True, text=True, check=True).stdout.strip()

for name in subprocess.run(
        ["aws", "iam", "list-role-policies", "--role-name", role,
         "--query", "PolicyNames[]", "--output", "text"],
        capture_output=True, text=True, check=True).stdout.split():
    if "DefaultPolicy" not in name:
        continue
    doc = json.loads(subprocess.run(
        ["aws", "iam", "get-role-policy", "--role-name", role, "--policy-name", name,
         "--query", "PolicyDocument", "--output", "json"],
        capture_output=True, text=True, check=True).stdout)
    changed = False
    for st in doc.get("Statement", []):
        res = st.get("Resource")
        res = [res] if isinstance(res, str) else (res or [])
        if any("foundation-model" in r or "inference-profile" in r for r in res):
            # Four ARNs, still ONE model: a us. profile routes across three regions and Bedrock
            # authorises against the backing foundation-model ARN in whichever it lands. Grant
            # only us-east-1 and you get intermittent AccessDenied under load — which is exactly
            # the moment someone reaches for a wildcard.
            st["Resource"] = [
                f"arn:aws:bedrock:us-east-1:{acct}:inference-profile/{profile}",
                *[f"arn:aws:bedrock:{r}::foundation-model/{model}"
                  for r in ("us-east-1", "us-east-2", "us-west-2")],
            ]
            changed = True
    if changed:
        open("/tmp/okf-narrowed.json", "w").write(json.dumps(doc))
        subprocess.run(["aws", "iam", "put-role-policy", "--role-name", role,
                        "--policy-name", name,
                        "--policy-document", "file:///tmp/okf-narrowed.json"], check=True)
        print(f"    narrowed {name}")
PY

echo "==> restore runtime environment variables (CDK reverts these too)"
aws bedrock-agentcore-control get-agent-runtime \
  --agent-runtime-id "$RUNTIME_ID" --region "$REGION" --output json > /tmp/okf-rt.json
python3 - "$RUNTIME_ID" "$MODEL" "$KB_ID" "$NEWS_FN" <<'PY' > /tmp/okf-update.json
import json, sys
rid, model, kb, news = sys.argv[1:5]
d = json.load(open("/tmp/okf-rt.json"))
# Resource identifiers only — never secrets. get-agent-runtime exposes these, which is why the
# news API key lives in the Lambda (fetched from SSM at cold start) and never here.
print(json.dumps({
    "agentRuntimeId": rid,
    "agentRuntimeArtifact": d["agentRuntimeArtifact"],
    "roleArn": d["roleArn"],
    "networkConfiguration": d["networkConfiguration"],
    "environmentVariables": {
        "OKF_MODEL_ID": model, "OKF_KB_ID": kb, "OKF_NEWS_FUNCTION": news,
    },
}))
PY
aws bedrock-agentcore-control update-agent-runtime --region "$REGION" \
  --cli-input-json file:///tmp/okf-update.json --query 'status' --output text

echo "==> wait for READY"
until [ "$(aws bedrock-agentcore-control get-agent-runtime --agent-runtime-id "$RUNTIME_ID" \
        --region "$REGION" --query 'status' --output text)" = "READY" ]; do sleep 5; done

echo "==> verify no model wildcard survived"
for p in $(aws iam list-role-policies --role-name "$role" --query 'PolicyNames[]' --output text); do
  if aws iam get-role-policy --role-name "$role" --policy-name "$p" --output json \
      | grep -oE '"arn:aws:bedrock:[^"]*\*"' | grep -qE 'foundation-model|inference-profile'; then
    echo "    *** WILDCARD REMAINS in $p ***" >&2; exit 1
  fi
done
echo "    clean"

echo
echo "deployed and verified. model: $MODEL"
echo "smoke test:  agentcore invoke --json '{\"question\":\"What percent of diagnosed diabetics take insulin?\"}'"
