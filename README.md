# okf-four-tools-agent

One agent, four kinds of knowing — and a hard line between them.

An answer's trustworthiness depends on **which tool produced it**, and the person reading it
cannot see the tools. So every tool stamps its answer with its mode, and one rule is enforced in
code rather than requested in a prompt:

> **A number may only ever come from a VERIFIED or COMPUTED tool.**

| Tool | Mode | Where the answer comes from |
| --- | --- | --- |
| `okf_facts` | **VERIFIED** | a figure recomputed from the microdata and checked at build time |
| `okf_query` | **COMPUTED** | calculated now, from a query the capability concept declares |
| `kb_narrative` | **RETRIEVED** | CDC documentation prose — grounded, but never checked |
| `health_news` | **LIVE** | third-party headlines — current, and unverifiable by construction |

The failure this prevents is **blending**: an agent that says *"31.96% of diagnosed adults take
insulin, and a recent study suggests that is rising"* in one breath, where the first clause was
executed against 29,522 records and the second is a headline. Both sound equally confident.

## Layout

```
concepts/query_adult23.md   the capability concept — declares what may be asked, and its
                            worked examples are verified by EXECUTION at build time
agent/query.py              reads that concept and builds the SQL
agent/facts.py              retrieval over the verified bundle (5 files: no vector DB, no sklearn)
agent/provenance.py         the ledger + the code-enforced no-ungrounded-numbers gate
agent/containment.py        the locked DuckDB connection (no fs, no network, no extensions)
agent/tools.py              the four tools and the system prompt
main.py                     AgentCore entrypoint
amplify/                    Cognito + AppSync + the askAgent proxy + the getNews Lambda
infra/                      the AgentCore execution role, least-privilege, as JSON
build.py                    stages a clean dist/ and derives its columns from the concept
```

## Run it

```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements-dev.txt
./.venv/bin/python build.py          # bundle + slim slice + dist/, from the OKF compiler
./.venv/bin/python -m pytest -q      # 96 tests, no AWS needed
```

Front door (needs Node ≥ 20 — `ampx` fails on 18):

```bash
npm install
npx ampx sandbox --profile <your-profile>
npx ampx sandbox secret set NEWS_API_KEY --profile <your-profile>
```

Model comparison: `./.venv/bin/python benchmark/models.py --models haiku-4.5,sonnet-4.6,opus-4.6`

## Two decisions that were earned, not chosen

**1. The model never writes SQL.** The obvious design is to let it write SQL and validate the
result. We built that, and three rounds of independent adversarial review broke it. The last
break has no fix: a steep rational function *is* an indicator function, so
`sum(AGE / (1 + (WEIGHT-k)^2 * 1e12))` returns one respondent's age using nothing but `+ - * /`.
Any expression language rich enough to compute a rate is rich enough to select a row. The tool
now takes enum keys declared by the concept — the attacks are **unsayable**, not filtered. Full
reasoning in [`docs/DECISION-sql-surface.md`](docs/DECISION-sql-surface.md).

**2. Provenance is enforced in code.** The no-blending rule lived in the system prompt until a
review pointed out that one sentence of instruction stood between a forged headline and a
fabricated health statistic wearing the VERIFIED badge. Now a ledger records what each tool
*computed*, and any answer containing a figure that traces to no VERIFIED/COMPUTED result is
withheld — regardless of how persuasive the surrounding text was.

Ground on **structural figures, never on digits scraped from rendered text**. That mistake let
the source filename `adult23.csv` ground a fabricated *"23% of diagnosed adults take insulin"*.

## What the security reviews found

Three independent rounds, each returning NOT-READY with a working exploit. Worth reading before
changing `agent/provenance.py` or `agent/facts.py`:

- **A forged fence.** The untrusted-source markers were static literals printed in the system
  prompt, in a public repo. One news headline containing the close marker plus a fake
  `[VERIFIED]` line passed off a fabricated 62.4% as verified. Markers are now per-process random
  and untrusted text is sanitised of markers *and* mode stamps.
- **Right number, wrong population.** A "title hit rescues a low relevance score" clause answered
  *"what percent of **children** take insulin?"* with the **adults** figure, stamped VERIFIED. The
  provenance ledger structurally cannot catch this — the number is grounded, only the denominator
  is wrong. The clause is gone and population qualifiers force a refusal.
- **Laundering through incidental digits.** Grouping by sex put the codes `1` and `2` into the
  grounded set, after which *"only 2% of U.S. adults have diabetes"* passed (true: 9.8%).
- **A stale artifact.** `dist/` once lagged two commits and would have deployed the very bug the
  review existed to close. A stale `dist/` is now a test failure, because "remember to run
  build.py" is not a control.

**Known residuals, accepted knowingly:** figures spelled without the word *percent* (*"one in
five"*) bypass the digit check; open Cognito signup is bounded by a budget alarm rather than a
quota; and a **grounded number can still carry a fabricated claim** — the ledger binds numerals,
not assertions, and no regex closes that. That last one is a real ceiling, not an oversight.

## Status

Deployable but **not deployed**. The AgentCore runtime is the remaining step; it is
approval-gated, its least-privilege policy is in [`infra/`](infra/README.md), and it should get
an independent security review of the built code before it goes up.
