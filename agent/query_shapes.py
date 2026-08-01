"""`okf_query` — declared query shapes. The model never writes SQL.

Three rounds of independent adversarial review broke a free-form-SQL guard (see
`docs/DECISION-sql-surface.md`). The decisive finding was that a steep rational function is an
indicator function:

    sum(DIBAGETC_A / (1 + (WTFA_A - 3146.794) * (WTFA_A - 3146.794) * 1e12))  ->  61.0

That contains no comparison, no CASE, no FILTER — every column sits in legitimate value context
under `sum`. Any expression language rich enough to compute a rate is rich enough to select a
row, so no AST classifier can separate the two. The boundary was wrong, not the checks.

So the surface is inverted. The model chooses:

    shape     — one of a small set of declared, verified query templates
    measure   — one of a declared set of variables
    universe  — one of a declared set of NAMED populations (not a free predicate)
    group_by  — one of a declared set of grouping columns, or nothing

Every slot is an enum. Trusted code owns the SQL string; nothing the model produces is ever
concatenated into it. That removes the whole class:

  * no free arithmetic          -> the indicator-kernel route cannot be expressed
  * the template emits count(*) -> the cell-size floor cannot be desynchronised
  * universes are enumerable    -> differencing is bounded and auditable, not open-ended
  * the weight is in the template -> a figure cannot be mislabelled as weighted

Layer 1 containment (locked DuckDB connection, wall-clock bound) is kept underneath. It was
never broken in three rounds, and it is what makes the residual risk of *any* execution bug
survivable.

In the finished system these declarations come from the OKF capability concept, and each shape
carries an example verified by execution at compile time — which is the same rule that governs
every other OKF claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import duckdb

from .query_guard import (
    MAX_ROWS,
    MIN_CELL_SIZE,
    QueryRejected,
    RELATION,
    execute_contained,
)

WEIGHT_COLUMN = "WTFA_A"

# Non-substantive response codes are declared PER MEASURE, never globally: for a categorical
# item they are 7/8/9 (refused / not ascertained / don't know), but for age at diagnosis 7, 8 and
# 9 are real ages and 96+ are the sentinels. A blanket list would silently delete children
# diagnosed before 10 — the kind of quiet wrongness this whole project exists to catch.


@dataclass(frozen=True)
class Universe:
    """A NAMED population. The predicate is authored here, never supplied by the model."""

    id: str
    label: str
    predicate: str  # trusted SQL fragment


@dataclass(frozen=True)
class Measure:
    """A variable the tool may report on, with the semantics the verifier established."""

    id: str
    column: str
    kind: str                 # "prevalence" | "mean"
    label: str
    affirmative: int | None = None   # the code meaning "yes", for prevalence
    valid: tuple[int, ...] = ()      # substantive response codes, for prevalence
    drop_codes: tuple[int, ...] = ()  # non-substantive codes, for a continuous measure


@dataclass(frozen=True)
class GroupBy:
    id: str
    column: str
    label: str
    valid: tuple[int, ...]


UNIVERSES: dict[str, Universe] = {
    "all_adults": Universe("all_adults", "All U.S. adults", "TRUE"),
    "diagnosed_diabetes": Universe(
        "diagnosed_diabetes", "Adults ever told they had diabetes", "DIBEV_A = 1"
    ),
    "prediabetes": Universe(
        "prediabetes", "Adults ever told they had prediabetes", "PREDIB_A = 1"
    ),
    "diabetes_or_prediabetes": Universe(
        "diabetes_or_prediabetes",
        "Adults with diabetes or prediabetes (the insulin question's universe)",
        "(DIBEV_A = 1 OR PREDIB_A = 1)",
    ),
}

MEASURES: dict[str, Measure] = {
    "DIBEV_A": Measure(
        "DIBEV_A", "DIBEV_A", "prevalence", "Ever told had diabetes",
        affirmative=1, valid=(1, 2),
    ),
    "DIBINS_A": Measure(
        "DIBINS_A", "DIBINS_A", "prevalence", "Currently takes insulin",
        affirmative=1, valid=(1, 2),
    ),
    "PREDIB_A": Measure(
        "PREDIB_A", "PREDIB_A", "prevalence", "Ever told had prediabetes",
        affirmative=1, valid=(1, 2),
    ),
    "DIBAGETC_A": Measure(
        "DIBAGETC_A", "DIBAGETC_A", "mean", "Age first told had diabetes (years)",
        drop_codes=(96, 97, 98, 99),
    ),
}

GROUPINGS: dict[str, GroupBy] = {
    "SEX_A": GroupBy("SEX_A", "SEX_A", "Sex", valid=(1, 2)),
}


def _valid_clause(column: str, valid: tuple[int, ...]) -> str:
    codes = ", ".join(str(v) for v in valid)
    return f"{column} IN ({codes})"


@dataclass(frozen=True)
class Shape:
    id: str
    label: str
    kinds: tuple[str, ...]                     # measure kinds this shape accepts
    build: Callable[[Measure, Universe, GroupBy | None], str]


def _build_prevalence(measure: Measure, universe: Universe, group: GroupBy | None) -> str:
    """Survey-weighted percentage answering affirmatively, within a named universe."""
    select = [
        f"SUM({WEIGHT_COLUMN}) FILTER (WHERE {measure.column} = {measure.affirmative}) "
        f"/ SUM({WEIGHT_COLUMN}) * 100 AS pct",
        "count(*) AS n",
    ]
    where = [universe.predicate, _valid_clause(measure.column, measure.valid)]
    group_sql = ""
    if group is not None:
        select.insert(0, f"{group.column}")
        where.append(_valid_clause(group.column, group.valid))
        group_sql = f" GROUP BY {group.column}"
    return f"SELECT {', '.join(select)} FROM {RELATION} WHERE {' AND '.join(where)}{group_sql}"


def _build_mean(measure: Measure, universe: Universe, group: GroupBy | None) -> str:
    """Survey-weighted mean of a continuous measure, within a named universe."""
    dropped = ", ".join(str(c) for c in measure.drop_codes) or "NULL"
    select = [
        f"SUM({WEIGHT_COLUMN} * {measure.column}) / SUM({WEIGHT_COLUMN}) AS mean",
        "count(*) AS n",
    ]
    where = [
        universe.predicate,
        f"{measure.column} IS NOT NULL",
        f"{measure.column} NOT IN ({dropped})",
    ]
    group_sql = ""
    if group is not None:
        select.insert(0, f"{group.column}")
        where.append(_valid_clause(group.column, group.valid))
        group_sql = f" GROUP BY {group.column}"
    return f"SELECT {', '.join(select)} FROM {RELATION} WHERE {' AND '.join(where)}{group_sql}"


SHAPES: dict[str, Shape] = {
    "weighted_prevalence": Shape(
        "weighted_prevalence",
        "Survey-weighted % answering yes, within a named universe",
        ("prevalence",),
        _build_prevalence,
    ),
    "weighted_mean": Shape(
        "weighted_mean",
        "Survey-weighted mean of a continuous measure, within a named universe",
        ("mean",),
        _build_mean,
    ),
}


@dataclass
class ShapeResult:
    shape: str
    measure: str
    universe: str
    group_by: str | None
    sql: str
    columns: list[str]
    rows: list[tuple]

    def render(self) -> str:
        head = " | ".join(self.columns)
        body = "\n".join(" | ".join(f"{v}" for v in row) for row in self.rows)
        return (
            f"{MEASURES[self.measure].label} — {UNIVERSES[self.universe].label}"
            f"{f' by {GROUPINGS[self.group_by].label}' if self.group_by else ''}\n"
            f"(survey-weighted, WTFA_A)\n\n{head}\n{body}"
        )


def catalogue() -> str:
    """What the model is allowed to ask for. Goes into the tool description verbatim."""
    lines = ["shapes:"]
    lines += [f"  {s.id} — {s.label}" for s in SHAPES.values()]
    lines.append("measures:")
    lines += [f"  {m.id} ({m.kind}) — {m.label}" for m in MEASURES.values()]
    lines.append("universes:")
    lines += [f"  {u.id} — {u.label}" for u in UNIVERSES.values()]
    lines.append("group_by (optional):")
    lines += [f"  {g.id} — {g.label}" for g in GROUPINGS.values()]
    return "\n".join(lines)


def _pick(kind: str, key: str | None, table: dict):
    if key is None:
        raise QueryRejected(f"{kind} is required; choose one of {sorted(table)}")
    if not isinstance(key, str) or key not in table:
        raise QueryRejected(f"unknown {kind} {key!r}; choose one of {sorted(table)}")
    return table[key]


def run_shape(
    con: duckdb.DuckDBPyConnection,
    shape: str | None = None,
    measure: str | None = None,
    universe: str | None = None,
    group_by: str | None = None,
) -> ShapeResult:
    """The tool entrypoint. Every argument is an enum key; none reaches SQL as text."""
    shape_def = _pick("shape", shape, SHAPES)
    measure_def = _pick("measure", measure, MEASURES)
    universe_def = _pick("universe", universe, UNIVERSES)
    group_def = _pick("group_by", group_by, GROUPINGS) if group_by else None

    if measure_def.kind not in shape_def.kinds:
        raise QueryRejected(
            f"shape {shape_def.id!r} takes a {' or '.join(shape_def.kinds)} measure; "
            f"{measure_def.id!r} is a {measure_def.kind} measure"
        )

    sql = shape_def.build(measure_def, universe_def, group_def)

    columns, rows = execute_contained(con, sql)
    if len(rows) > MAX_ROWS:
        raise QueryRejected("result exceeds the row bound")

    # Disclosure control, defence in depth: the template always emits count(*) last, so the
    # floor cannot be pointed at the wrong column the way a model-authored projection could.
    count_index = columns.index("n")
    for row in rows:
        cell = row[count_index]
        if isinstance(cell, (int, float)) and cell < MIN_CELL_SIZE:
            raise QueryRejected(
                f"refused: a reported cell holds fewer than {MIN_CELL_SIZE} respondents "
                "(statistical disclosure control)"
            )

    return ShapeResult(
        shape=shape_def.id,
        measure=measure_def.id,
        universe=universe_def.id,
        group_by=group_def.id if group_def else None,
        sql=sql,
        columns=columns,
        rows=rows,
    )
