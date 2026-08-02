"""AgentCore Runtime entrypoint — the whole agent, assembled and served here.

Open this file to see the entire deployed surface at a glance:

  * the `@tool`s the agent may call — four kinds of knowing, five tools,
  * the Strands `Agent` on Amazon Bedrock, built per request, and
  * `invoke`, the `POST /invocations` entrypoint.

The tools' logic lives in `agent/tools.py`; what the query tool may compute is declared in
`concepts/query_adult23.md` and verified by execution at build time.

The thesis, in one line: an answer's trustworthiness depends on which tool produced it, and the
user cannot see the tools — so every tool stamps its answer with its mode, and the system prompt
forbids a number from coming from anywhere but a VERIFIED or COMPUTED one.
"""

from __future__ import annotations

import contextvars
import os
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.runtime.models import PingStatus
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from agent import provenance, tools

# What the tools actually returned during THIS request. The provenance check reads it to verify
# every number in the answer came from a tool allowed to produce one — which is what turns the
# system prompt's central rule into a property of the code instead of a request to the model.
_LEDGER: contextvars.ContextVar[provenance.Ledger] = contextvars.ContextVar("ledger")


def _record(result: tools.ToolResult) -> str:
    ledger = _LEDGER.get(None)
    if ledger is not None:
        ledger.record(result.mode, result.text, result.figures)
    return result.render()

app = BedrockAgentCoreApp()
log = app.logger

# Per-request bounds. The endpoint is authenticated, but "authenticated" is not "authorised to
# spend": a multi-tool loop is many model calls, so cap the input and the output.
MAX_QUESTION_CHARS = 600
MAX_OUTPUT_TOKENS = 800

MODEL_ID = os.environ.get("OKF_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")


# --- The tools. Four kinds of knowing; verify_claim is VERIFIED pointed at a claim ---------------------------------------------

@tool
def okf_facts(question: str) -> str:
    """VERIFIED. Look up a figure that was recomputed from the microdata and checked at build
    time. Use this first for any question asking for a number about diabetes, insulin use,
    prediabetes, or age at diagnosis. Refuses when no verified concept covers the question."""
    return _record(tools.okf_facts(question))


@tool
def okf_query(measure: str, universe: str, group_by: str = "") -> str:
    """COMPUTED. Calculate a survey-weighted figure now, for a combination the verified
    concepts do not already publish — for example a breakdown by sex. Arguments are keys, not
    SQL; the accepted keys are listed in the system prompt. Refuses an invalid combination."""
    return _record(tools.okf_query(measure, universe, group_by or None))


@tool
def kb_narrative(question: str) -> str:
    """RETRIEVED, NOT VERIFIED. Search CDC documentation for explanatory prose — survey
    methodology, what a variable means, how weighting works. Use for 'why' and 'how' questions.
    It cannot supply a figure: the survey-weighted numbers are computed from microdata and
    appear in no document. Anything it returns is unverified source text."""
    return _record(tools.kb_narrative(question))


@tool
def health_news(topic: str) -> str:
    """LIVE, NOT VERIFIED. Fetch recent third-party headlines for one of: diabetes, insulin,
    public_health. Use only for 'what is new' questions. Never treat a figure in a headline as
    a fact — cite it as an unverified news claim."""
    return _record(tools.health_news(topic))


@tool
def verify_claim(question: str, claim: float) -> str:
    """VERIFIED. Check a figure claimed by an unverified source (a headline, retrieved prose)
    against the verified bundle. Call this instead of repeating the claim. `question` describes
    what the claim is about, in the source's own terms including its population — it is matched
    against the verified concepts and REFUSED if none covers it. `claim` is the number the
    source stated. Report the verified figure and the population it describes, and whether the
    claim is supported; do NOT restate the claimed figure."""
    return _record(tools.verify_claim(question, claim))


# --- The agent ---------------------------------------------------------------------------

def build_agent() -> Agent:
    """Assemble the reasoning loop: a model, the rules, and the only things it may call.

    Built fresh per request, so each answer is stateless and `import main` needs no AWS
    credentials — which is what lets the tools be tested locally without a deploy.
    """
    return Agent(
        model=BedrockModel(
            model_id=MODEL_ID, region_name=REGION, max_tokens=MAX_OUTPUT_TOKENS
        ),
        system_prompt=tools.SYSTEM_PROMPT,
        tools=[okf_facts, okf_query, kb_narrative, health_news, verify_claim],
    )


def _question(payload: dict[str, Any]) -> str | None:
    for key in ("question", "query", "prompt"):
        value = (payload or {}).get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


@app.entrypoint
def invoke(payload: dict[str, Any], context: Any = None) -> dict[str, Any]:
    """Answer one question, grounded in whichever tool is appropriate — or refuse."""
    question = _question(payload)
    if question is None:
        return {"answered": False, "answer": "No question provided."}
    if len(question) > MAX_QUESTION_CHARS:
        return {
            "answered": False,
            "answer": f"Question too long (limit {MAX_QUESTION_CHARS} characters).",
        }

    log.info("question received (%d chars)", len(question))
    ledger = provenance.Ledger()
    _LEDGER.set(ledger)
    try:
        answer = str(build_agent()(question)).strip()

        # THE GATE. Every number in the answer must trace to a VERIFIED or COMPUTED tool result.
        # A figure the model invented, or copied out of a retrieved passage or a headline, does
        # not survive this — regardless of how convincing the surrounding text was.
        verdict = provenance.check(answer, ledger, question)
        if not verdict.ok:
            log.warning("withheld: ungrounded figures %s", sorted(verdict.ungrounded))
            return {
                "answered": False,
                "mode": "withheld-ungrounded",
                "answer": f"{verdict.note}\n\n{tools.SAFETY}",
            }

        return {"answered": True, "mode": "agent", "answer": f"{answer}\n\n{tools.SAFETY}"}
    except Exception as exc:
        # Never fail the request into a stack trace. Fall back to the verified bundle, which
        # needs no model at all — a degraded answer that is still grounded and cited.
        log.warning("agent unavailable (%s); falling back to the verified bundle", exc)
        fallback = tools.okf_facts(question)
        return {
            "answered": fallback.mode != "REFUSED",
            "mode": "fallback",
            "answer": f"{fallback.render()}\n\n{tools.SAFETY}",
        }


@app.ping
def ping() -> PingStatus:
    return PingStatus.HEALTHY


__all__ = ["app"]

if __name__ == "__main__":
    app.run()
