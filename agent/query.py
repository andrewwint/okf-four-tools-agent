"""`okf_query` — the agent picks a measure and a population. It never writes SQL.

**Everything this tool will do is declared in an OKF capability concept**
(`concepts/query_adult23.md`), not in this file. The concept lists the measures, the named
universes, the groupings, and — crucially — worked examples with their expected answers. This
module reads that declaration and builds the SQL.

That has two payoffs, and they are the reason the capability concept exists at all:

1. **The tool is verified the same way a fact is.** `verify()` runs each declared example and
   checks it against the number the compiler proved. If an example stops matching, the
   capability is quarantined and the agent loses the tool rather than gaining a broken one.
2. **The agent cannot ask a wrong question.** The universe is a name from a list, not a filter
   it composes, so the classic skip-pattern error — computing insulin use over the whole sample
   and getting 3.66% instead of 31.96% — is not expressible as a request.

The obvious alternative was to let the model write SQL and validate it. We tried; three rounds
of independent review broke it, and the final break has no fix (a steep enough fraction is an
indicator function, so pure arithmetic can single out one respondent). See
`docs/DECISION-sql-surface.md`. Here the attacks are not blocked — they are unsayable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import yaml

from .containment import MAX_ROWS, RELATION, QueryRejected, execute_contained

CONCEPT_PATH = Path(__file__).resolve().parents[1] / "concepts" / "query_adult23.md"


def load_concept(path: Path = CONCEPT_PATH) -> dict:
    """Read the capability declaration out of the concept's YAML frontmatter."""
    match = re.match(r"^---\n(.*?)\n---\n", path.read_text(), re.DOTALL)
    if not match:
        raise ValueError(f"no frontmatter in {path}")
    return yaml.safe_load(match.group(1))


CONCEPT = load_concept()
MEASURES: dict = CONCEPT["measures"]
UNIVERSES: dict = CONCEPT["universes"]
GROUPINGS: dict = CONCEPT["groupings"]
WEIGHT: str = CONCEPT["weight"]
MIN_CELL_SIZE: int = CONCEPT["min_cell_size"]


def _codes(column: str, codes: list[int]) -> str:
    return f"{column} IN ({', '.join(str(c) for c in codes)})"


def build_sql(measure: str, universe: str, group_by: str | None = None) -> str:
    """Build the query from the declaration. Every value comes from the concept, not the caller.

    RULE for any future shape: return ratios and means, never an unnormalised weighted total.
    `SUM(WTFA_A)` looks like a harmless "estimated population count", but differencing it across
    two nested universes yields one respondent's exact survey weight — which is the seed an
    earlier review used to walk the entire file.
    """
    spec = MEASURES[measure]
    if spec["kind"] == "prevalence":
        figure = (f"SUM({WEIGHT}) FILTER (WHERE {measure} = 1) "
                  f"/ SUM({WEIGHT}) * 100 AS pct")
        where = [UNIVERSES[universe]["predicate"], _codes(measure, spec["valid"])]
    else:
        figure = f"SUM({WEIGHT} * {measure}) / SUM({WEIGHT}) AS mean"
        where = [UNIVERSES[universe]["predicate"], f"{measure} IS NOT NULL"]
        if spec.get("drop"):  # `NOT IN (NULL)` is NULL for every row — never emit an empty list
            where.append(f"{measure} NOT IN ({', '.join(str(c) for c in spec['drop'])})")

    select = [figure, "count(*) AS n_sample"]
    group_sql = ""
    if group_by:
        select.insert(0, group_by)
        where.append(_codes(group_by, GROUPINGS[group_by]["valid"]))
        group_sql = f" GROUP BY {group_by}"
    return f"SELECT {', '.join(select)} FROM {RELATION} WHERE {' AND '.join(where)}{group_sql}"


@dataclass
class Result:
    measure: str
    universe: str
    group_by: str | None
    sql: str
    columns: list[str]
    rows: list[tuple]

    @property
    def figure(self) -> float:
        return self.rows[0][0] if len(self.rows) == 1 else None

    def render(self) -> str:
        labels = (GROUPINGS.get(self.group_by, {}) or {}).get("value_labels") or {}

        def cell(value, index: int) -> str:
            # Substitute the group label for its code, and round the figure: full float
            # precision invites the model to quote 31.961243238265087 as if that were the
            # verified claim, when the concept states 31.96.
            if index == 0 and self.group_by and labels:
                return str(labels.get(value, value))
            return f"{value:.2f}" if isinstance(value, float) else str(value)

        head = " | ".join(self.columns)
        body = "\n".join(
            " | ".join(cell(v, i) for i, v in enumerate(row)) for row in self.rows
        )
        by = f" by {GROUPINGS[self.group_by]['label']}" if self.group_by else ""
        return (f"{MEASURES[self.measure]['label']} — {UNIVERSES[self.universe]['label']}{by}\n"
                f"Estimate is survey-weighted ({WEIGHT}); n_sample is a raw unweighted "
                f"respondent count, not a population figure.\n"
                f"Source: {CONCEPT['source']}\n\n{head}\n{body}")


def catalogue() -> str:
    """Exactly what may be asked for — this text goes into the agent's tool description."""
    return "\n".join(
        ["measures:"]
        + [f"  {k} ({v['kind']}) — {v['label']}; universes: "
           f"{', '.join(v.get('universes', list(UNIVERSES)))}" for k, v in MEASURES.items()]
        + ["universes:"]
        + [f"  {k} — {v['label']}" for k, v in UNIVERSES.items()]
        + ["group_by (optional):"]
        + [f"  {k} — {v['label']}" for k, v in GROUPINGS.items()]
    )


def _pick(kind: str, key, allowed) -> str:
    """Resolve a caller key to a declared one.

    Never echo `key` in the message: the agent's context carries text an outsider can influence,
    so reflecting it back as a tool result launders an injected instruction into a channel the
    model trusts. Report the slot and the valid options only.
    """
    if key is None:
        raise QueryRejected(f"{kind} is required; choose one of {sorted(allowed)}")
    if not isinstance(key, str) or key not in allowed:
        raise QueryRejected(f"unknown {kind}; choose one of {sorted(allowed)}")
    return key


def run_query(
    con: duckdb.DuckDBPyConnection,
    measure: str | None = None,
    universe: str | None = None,
    group_by: str | None = None,
) -> Result:
    """The tool entrypoint. Every argument is a key from the concept — never SQL."""
    measure = _pick("measure", measure, MEASURES)
    universe = _pick("universe", universe, UNIVERSES)
    group_by = _pick("group_by", group_by, GROUPINGS) if group_by is not None else None

    # The universe must be one the measure is actually ASKED in. For a skip-pattern item the
    # validity filter silently narrows the rows, so a broader universe would compute a real
    # number and label it with a population it was not computed over — e.g. insulin use over
    # "all adults" reads 16.75% when the true all-adults figure is nothing of the sort.
    allowed = MEASURES[measure].get("universes", list(UNIVERSES))
    if universe not in allowed:
        raise QueryRejected(
            f"{measure} is not asked of that population; valid universes: {sorted(allowed)}"
        )

    sql = build_sql(measure, universe, group_by)
    columns, rows = execute_contained(con, sql)

    if len(rows) > MAX_ROWS:
        raise QueryRejected("result exceeds the row bound")

    # Disclosure control. The template always puts count(*) last, so — unlike a model-authored
    # projection — this check cannot be pointed at the wrong column.
    for row in rows:
        if row[-1] < MIN_CELL_SIZE:
            raise QueryRejected(
                f"refused: a reported group holds fewer than {MIN_CELL_SIZE} respondents "
                "(statistical disclosure control)"
            )

    return Result(measure, universe, group_by, sql, columns, rows)


# --------------------------------------------------------------------------------------
# The gate: a capability is verified by EXECUTION, exactly like a verified figure
# --------------------------------------------------------------------------------------

@dataclass
class Check:
    measure: str
    universe: str
    expect: float
    actual: float | None
    ok: bool
    detail: str


def verify(con: duckdb.DuckDBPyConnection, tolerance: float = 0.01) -> list[Check]:
    """Run every example the concept declares and compare it to the verified figure.

    This is what makes a capability concept a *concept* rather than a prompt file: the claim
    "this tool computes 31.96% for insulin among diagnosed adults" is not documentation, it is
    an assertion that gets executed. If it fails, the compiler quarantines the capability.
    """
    checks: list[Check] = []
    for example in CONCEPT["examples"]:
        try:
            result = run_query(con, example["measure"], example["universe"])
            actual = round(result.figure, 2)
            ok = abs(actual - example["expect"]) <= tolerance
            detail = "matches" if ok else f"expected {example['expect']}, computed {actual}"
        except Exception as exc:
            actual, ok, detail = None, False, f"failed to execute: {exc}"
        checks.append(
            Check(example["measure"], example["universe"], example["expect"], actual, ok, detail)
        )
    return checks
