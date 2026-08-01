"""AgentCore Runtime entrypoint — the whole four-tool agent, assembled and served here.

Open this file to see the entire deployed surface at a glance:

  * the four `@tool`s the agent may call, one per KIND OF KNOWING,
  * the Strands `Agent` on Amazon Bedrock, built per request, and
  * `invoke`, the `POST /invocations` entrypoint.

The tools' logic lives in `agent/tools.py`; what the query tool may compute is declared in
`concepts/query_adult23.md` and verified by execution at build time.

The thesis, in one line: an answer's trustworthiness depends on which tool produced it, and the
user cannot see the tools — so every tool stamps its answer with its mode, and the system prompt
forbids a number from coming from anywhere but a VERIFIED or COMPUTED one.
"""

from __future__ import annotations

import os
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.runtime.models import PingStatus
from strands import Agent, tool
from strands.models.bedrock import BedrockModel

from agent import tools

app = BedrockAgentCoreApp()
log = app.logger

# Per-request bounds. The endpoint is authenticated, but "authenticated" is not "authorised to
# spend": a four-tool loop is many model calls, so cap the input and the output.
MAX_QUESTION_CHARS = 600
MAX_OUTPUT_TOKENS = 800

MODEL_ID = os.environ.get("OKF_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
REGION = os.environ.get("AWS_REGION", "us-east-1")


# --- The four tools, one per kind of knowing ---------------------------------------------

@tool
def okf_facts(question: str) -> str:
    """VERIFIED. Look up a figure that was recomputed from the microdata and checked at build
    time. Use this first for any question asking for a number about diabetes, insulin use,
    prediabetes, or age at diagnosis. Refuses when no verified concept covers the question."""
    return tools.okf_facts(question).render()


@tool
def okf_query(measure: str, universe: str, group_by: str = "") -> str:
    """COMPUTED. Calculate a survey-weighted figure now, for a combination the verified
    concepts do not already publish — for example a breakdown by sex. Arguments are keys, not
    SQL; the accepted keys are listed in the system prompt. Refuses an invalid combination."""
    return tools.okf_query(measure, universe, group_by or None).render()


@tool
def kb_narrative(question: str) -> str:
    """RETRIEVED, NOT VERIFIED. Search CDC documentation for explanatory prose — survey
    methodology, what a variable means, how weighting works. Use for 'why' and 'how' questions.
    It cannot supply a figure: the survey-weighted numbers are computed from microdata and
    appear in no document. Anything it returns is unverified source text."""
    return tools.kb_narrative(question).render()


@tool
def health_news(topic: str) -> str:
    """LIVE, NOT VERIFIED. Fetch recent third-party headlines for one of: diabetes, insulin,
    public_health. Use only for 'what is new' questions. Never treat a figure in a headline as
    a fact — cite it as an unverified news claim."""
    return tools.health_news(topic).render()


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
        tools=[okf_facts, okf_query, kb_narrative, health_news],
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

    log.info("question: %s", question)
    try:
        answer = str(build_agent()(question)).strip()
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
