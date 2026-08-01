"""The `okf_query` guard — constrained aggregate SQL over the verified NHIS slice.

An LLM writes the SQL, so the SQL is untrusted. Worse than "user input": the agent's context
also carries text from the narrative and news tools, either of which an outsider can influence,
so treat every query as attacker-controlled.

Three layers, containment BEFORE parsing, because the parser is the layer that will have a bug:

  Layer 1  the DuckDB connection is locked down (no filesystem, no network, no extensions,
           bounded memory) and the data is bound by trusted code as relation `t`, so the model
           never needs — and cannot use — a file path. Execution is bounded by a wall clock,
           because `memory_limit` bounds memory and not CPU.
  Layer 2  the SQL is parsed by DuckDB's OWN parser (json_serialize_sql) and asserted
           structurally (below).
  Layer 3  the RESULT is bounded: row cap, minimum cell size, and a figure is reported as
           weighted only when the survey weight is genuinely summed.

Two rules do the real work, and both were learned by being broken:

**Rule 1 — a high-cardinality column may appear only in an aggregate's VALUE expression.**
The person-level weight, the design stratum and PSU, and age at diagnosis may never appear in a
projection, a GROUP BY, or *any predicate* — not a WHERE, not a FILTER clause, not a CASE WHEN,
not a HAVING comparison. The distinction that matters is value-vs-predicate, not
inside-vs-outside the aggregate node: `MIN(DIBAGETC_A) FILTER (WHERE WTFA_A = 3146.794)` is
syntactically "inside an aggregate" and is a single-person lookup.

**Rule 2 — every query must report its cell size, and small cells are refused.**
Low column cardinality is NOT the invariant; group size is. The four categorical columns here
have 3–4 values each, yet their 4-way cross product has 35 cells of which 8 hold exactly one
respondent — because the categories include the rare refusal/don't-know sentinels. Plain,
intended-shape SQL then returns `MIN()` over a cell of one, which is that person's value. So the
guard requires `count(*)` in the projection and refuses any cell below `MIN_CELL_SIZE`. This is
ordinary statistical disclosure control, and it is the only one of these rules that keeps working
when someone adds a column to the slice.

Both rules replaced weaker predecessors that an independent review broke: first an aggregate
appearing *anywhere* in the tree (an aggregate in a WHERE subquery satisfied it while the
projection stayed bare), then "inside the aggregate node" (FILTER and CASE walked straight
through it). Shape-matching is not a control; structure is.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

# The one relation the model may name. Bound by trusted code at startup.
RELATION = "t"

# Wall-clock bound on any single execution.
#
# Found by the hostile suite, not by design review: `memory_limit` bounds MEMORY, not CPU, so a
# recursive CTE runs to completion on an otherwise fully locked connection. Config alone does not
# satisfy "bounded cost" — containment needs a clock as well as a lock. The interrupt is checked
# at chunk boundaries, so this is a best-effort bound (sub-second overshoot), not a hard deadline.
QUERY_TIMEOUT_SECONDS = 5.0

# Layer 3 bounds.
MAX_ROWS = 50
MIN_CELL_SIZE = 30          # disclosure control: no reported cell may be smaller than this
WEIGHT_COLUMN = "WTFA_A"
UNWEIGHTED_BANNER = "UNWEIGHTED — a raw sample figure, NOT a population estimate."

# Columns that may appear outside an aggregate's value expression. Low cardinality is necessary
# but NOT sufficient (see Rule 2) — the cell-size floor is what actually prevents singling out.
#
# In the finished system this list is declared by the OKF capability concept; it is a module
# constant here so the guard can be built and attacked before the concept exists.
GROUPABLE_COLUMNS = frozenset({"DIBEV_A", "DIBINS_A", "PREDIB_A", "SEX_A"})

# Aggregates that constitute weighted ESTIMATION. count/min/max of the weight are not estimation:
# `min(WTFA_A)` is one respondent's weight, not a population figure.
_SUMMATION_AGGREGATES = {"sum", "avg", "mean", "median", "quantile_cont", "quantile_disc"}
_AGGREGATES = _SUMMATION_AGGREGATES | {
    "count", "count_star", "min", "max", "stddev", "stddev_samp", "var_samp",
}
_SCALARS = {
    "+", "-", "*", "/", "%", "==", "=", "!=", "<>", "<", "<=", ">", ">=",
    "and", "or", "not", "in", "abs", "round", "cast", "coalesce", "case",
    "float", "double", "integer", "bigint", "decimal",
}
_ALLOWED_FUNCTIONS = _AGGREGATES | _SCALARS

# Node types permitted in a FROM clause. Anything else — notably TABLE_FUNCTION — is out.
_ALLOWED_FROM_TYPES = {"BASE_TABLE", "SUBQUERY", "JOIN", "EMPTY", "EMPTY_FROM"}

# Predicate-forming nodes. Their operands are row selectors, never aggregated values.
_PREDICATE_PREFIXES = ("COMPARE", "CONJUNCTION", "OPERATOR_IN", "OPERATOR_NOT", "BETWEEN")

_ENGINE_ERROR = "the query could not be executed"


class QueryRejected(ValueError):
    """The query did not survive a guard layer, or failed in the engine. Nothing leaked."""


# --------------------------------------------------------------------------------------
# Layer 1 — the locked connection and the bounded execution
# --------------------------------------------------------------------------------------

def build_guarded_connection(parquet_path: str | Path) -> duckdb.DuckDBPyConnection:
    """A DuckDB connection that cannot touch the filesystem, the network, or extensions.

    Ordering is load-then-lock and it matters: `enable_external_access=false` blocks ALL file
    access, including the trusted read that loads the slice. So trusted code reads the parquet
    into an in-memory table FIRST, and only then closes the door — after which the setting
    cannot be reversed, because `lock_configuration` is set last.
    """
    path = Path(parquet_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"slice not found: {path}")

    con = duckdb.connect(":memory:")
    con.execute(f"CREATE TABLE {RELATION} AS SELECT * FROM read_parquet(?)", [str(path)])

    for setting in (
        "SET enable_external_access=false",       # filesystem, httpfs, ATTACH, COPY TO
        "SET autoinstall_known_extensions=false",
        "SET autoload_known_extensions=false",    # httpfs autoload evades keyword blocklists
        "SET allow_community_extensions=false",
        "SET memory_limit='512MB'",
        "SET threads=2",
        "SET max_expression_depth=100",
        "SET lock_configuration=true",            # LAST: nothing above can be re-enabled
    ):
        con.execute(setting)
    return con


def guarded_columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    """The real column names of the bound relation (the identifier allow-list source)."""
    cur = con.cursor()
    try:
        return [r[0] for r in cur.execute(f"DESCRIBE {RELATION}").fetchall()]
    finally:
        cur.close()


def execute_contained(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    timeout: float = QUERY_TIMEOUT_SECONDS,
    max_rows: int = MAX_ROWS,
) -> tuple[list[str], list[tuple]]:
    """Execute under Layer-1 containment only — locked connection, own cursor, wall clock.

    Deliberately performs NO validation, so the tests can prove containment holds with the parser
    stubbed out. Returns materialised results: the clock must cover the fetch as well as the
    execute, so this never hands back a live cursor.

    Each call gets its OWN cursor. `con.execute` returns the connection itself, so two callers
    sharing one connection overwrite each other's results — and `DESCRIBE` could then report
    another query's columns, corrupting validation itself.
    """
    cur = con.cursor()
    timer = threading.Timer(timeout, cur.interrupt)
    timer.daemon = True
    timer.start()
    try:
        cur.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        columns = [d[0] for d in cur.description] if cur.description else []
        return columns, [tuple(r) for r in rows]
    except duckdb.InterruptException as exc:
        raise QueryRejected(
            f"query exceeded the {timeout:g}s execution bound and was interrupted"
        ) from exc
    except QueryRejected:
        raise
    except Exception as exc:  # engine text is an injection channel and an inference oracle
        log.warning("guarded query failed: %s: %s", type(exc).__name__, exc)
        raise QueryRejected(_ENGINE_ERROR) from exc
    finally:
        timer.cancel()
        cur.close()


# --------------------------------------------------------------------------------------
# Layer 2 — structural AST validation using DuckDB's own parser
# --------------------------------------------------------------------------------------

@dataclass
class _Analysis:
    from_types: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    exposed_columns: list[str] = field(default_factory=list)  # projected, grouped, or compared
    value_columns: list[str] = field(default_factory=list)    # inside an aggregate's value expr
    qualifiers: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    stars: int = 0
    table_functions: int = 0
    aggregates: int = 0
    weight_is_summed: bool = False
    count_star_positions: list[int] = field(default_factory=list)


def _analyze(node, acc: _Analysis, in_value: bool = False, in_predicate: bool = False,
             agg: str | None = None) -> _Analysis:
    """Walk the AST tracking VALUE context versus PREDICATE context.

    A column is protected (may be high-cardinality) only when it is reached through an
    aggregate's value expression and never through a predicate.
    """
    if isinstance(node, dict):
        ntype = str(node.get("type") or "")

        if ntype in ("BASE_TABLE", "TABLE_FUNCTION", "SUBQUERY", "JOIN"):
            acc.from_types.append(ntype)
        if ntype == "TABLE_FUNCTION":
            acc.table_functions += 1
        if ntype == "BASE_TABLE" and node.get("table_name"):
            acc.tables.append(node["table_name"])
        if ntype == "STAR":
            acc.stars += 1
        if node.get("alias"):
            acc.aliases.append(str(node["alias"]))

        # Predicates: operands are row selectors, so nothing under them is protected.
        if any(ntype.startswith(p) for p in _PREDICATE_PREFIXES):
            for value in node.values():
                _analyze(value, acc, in_value, True, agg)
            return acc

        # CASE: the WHEN is a predicate; THEN/ELSE stay in whatever context we were in.
        if ntype == "CASE_EXPR":
            for check in node.get("case_checks") or []:
                _analyze(check.get("when_expr"), acc, in_value, True, agg)
                _analyze(check.get("then_expr"), acc, in_value, in_predicate, agg)
            _analyze(node.get("else_expr"), acc, in_value, in_predicate, agg)
            return acc

        if ntype == "COLUMN_REF":
            names = node.get("column_names") or []
            if names:
                if len(names) > 1:                      # ["t", "WTFA_A"] — qualifier first
                    acc.qualifiers.append(str(names[0]))
                column = str(names[-1])
                if in_value and not in_predicate:
                    acc.value_columns.append(column)
                    if (column.lower() == WEIGHT_COLUMN.lower()
                            and agg in _SUMMATION_AGGREGATES):
                        acc.weight_is_summed = True
                else:
                    acc.exposed_columns.append(column)
            return acc

        if ntype == "FUNCTION":
            name = str(node.get("function_name") or "").lower()
            acc.functions.append(name)
            is_aggregate = name in _AGGREGATES
            if is_aggregate:
                acc.aggregates += 1
            # Arguments are the value expression.
            for child in node.get("children") or []:
                _analyze(child, acc, in_value or is_aggregate, in_predicate,
                         name if is_aggregate else agg)
            # A FILTER clause is a ROW SELECTOR, not a value — this is the FILTER hole.
            if node.get("filter"):
                _analyze(node["filter"], acc, False, True, None)
            for key, value in node.items():
                if key not in ("children", "filter", "alias", "function_name"):
                    _analyze(value, acc, in_value, in_predicate, agg)
            return acc

        for value in node.values():
            _analyze(value, acc, in_value, in_predicate, agg)

    elif isinstance(node, list):
        for item in node:
            _analyze(item, acc, in_value, in_predicate, agg)

    return acc


def _count_star_index(select_list: list) -> int | None:
    """Index of the first projected expression that is a bare `count(*)`."""
    for index, entry in enumerate(select_list or []):
        found = _analyze(entry, _Analysis())
        if "count_star" in found.functions:
            return index
    return None


def validate_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    groupable: frozenset[str] = GROUPABLE_COLUMNS,
) -> tuple[_Analysis, int]:
    """Reject anything that is not a single, aggregate-only, `t`-only SELECT. Never executes.

    Returns the analysis and the result-column index carrying `count(*)`, which Layer 3 uses to
    enforce the cell-size floor.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise QueryRejected("empty query")

    cur = con.cursor()
    try:
        raw = cur.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
        parsed = json.loads(raw)
    except Exception as exc:
        raise QueryRejected("could not parse the query") from exc
    finally:
        cur.close()

    if parsed.get("error"):
        raise QueryRejected("only a single SELECT statement is allowed")

    statements = parsed.get("statements") or []
    if len(statements) != 1:
        raise QueryRejected(f"exactly one statement is allowed, found {len(statements)}")

    acc = _analyze(statements, _Analysis())

    if acc.table_functions:
        raise QueryRejected(
            "table functions are not allowed (that is how a query reaches files and the network)"
        )
    if acc.stars:
        raise QueryRejected("`SELECT *` is not allowed — name aggregate expressions explicitly")

    bad_tables = sorted({t for t in acc.tables if t != RELATION})
    if bad_tables:
        raise QueryRejected(f"only the relation {RELATION!r} may be queried; found {bad_tables}")
    if not acc.tables:
        raise QueryRejected(f"query must read from {RELATION!r}")

    bad_from = sorted({f for f in acc.from_types if f not in _ALLOWED_FROM_TYPES})
    if bad_from:
        raise QueryRejected(f"disallowed FROM construct: {bad_from}")

    bad_funcs = sorted({f for f in acc.functions if f not in _ALLOWED_FUNCTIONS})
    if bad_funcs:
        raise QueryRejected(f"function(s) not on the allow-list: {bad_funcs}")

    if not acc.aggregates:
        raise QueryRejected(
            "aggregate-only: the query must use an aggregate (this tool never returns records)"
        )

    known = {c.lower() for c in guarded_columns(con)}
    alias_names = {a.lower() for a in acc.aliases}
    allowed_qualifiers = {RELATION.lower()} | alias_names
    bad_qualifiers = sorted({q for q in acc.qualifiers if q.lower() not in allowed_qualifiers})
    if bad_qualifiers:
        raise QueryRejected(f"unknown table qualifier(s): {bad_qualifiers}")

    referenced = {c.lower() for c in acc.exposed_columns + acc.value_columns}
    unknown = sorted({c for c in referenced if c not in known and c not in alias_names})
    if unknown:
        raise QueryRejected(f"unknown column(s): {unknown}")

    # RULE 1. An exposed column is projected, grouped, or used in ANY predicate (WHERE, FILTER,
    # CASE WHEN, HAVING). High-cardinality columns are confined to aggregate value expressions.
    # NOTE: aliases are deliberately NOT exempt here — `SELECT WTFA_A AS WTFA_A` was a bypass.
    # The alias exemption applies only to the unknown-column check above.
    groupable_lower = {g.lower() for g in groupable}
    disclosive = sorted(
        {c for c in acc.exposed_columns if c.lower() not in groupable_lower and c.lower() in known}
    )
    if disclosive:
        raise QueryRejected(
            f"column(s) {disclosive} may only appear inside an aggregate's value expression — "
            f"never projected, grouped, or compared. Bare use is limited to {sorted(groupable)}"
        )

    # RULE 2. Every query reports its cell size so small cells can be refused.
    node = statements[0].get("node") or {}
    count_index = _count_star_index(node.get("select_list") or [])
    if count_index is None:
        raise QueryRejected(
            "every query must project count(*) (e.g. `, count(*) AS n`) so cell sizes below "
            f"{MIN_CELL_SIZE} can be refused"
        )

    return acc, count_index


# --------------------------------------------------------------------------------------
# Layer 3 — result shape, cell size, and the weight label
# --------------------------------------------------------------------------------------

@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    weighted: bool
    caveat: str = ""

    def render(self) -> str:
        head = " | ".join(self.columns)
        body = "\n".join(" | ".join(f"{v}" for v in row) for row in self.rows)
        out = f"{head}\n{body}"
        if self.caveat:
            out = f"{self.caveat}\n\n{out}"
        return out


def run_query(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    groupable: frozenset[str] = GROUPABLE_COLUMNS,
) -> QueryResult:
    """Validate, execute under containment, then bound the result. The tool entrypoint."""
    acc, count_index = validate_sql(con, sql, groupable=groupable)

    columns, rows = execute_contained(con, sql)
    if len(rows) > MAX_ROWS:
        raise QueryRejected(
            f"result exceeds {MAX_ROWS} rows — this tool returns aggregates, not records"
        )

    # Disclosure control: refuse the whole result if ANY reported cell is too small. Refusing
    # rather than suppressing is deliberate — a partially suppressed table can be differenced.
    for row in rows:
        if count_index < len(row):
            cell = row[count_index]
            if isinstance(cell, (int, float)) and cell < MIN_CELL_SIZE:
                raise QueryRejected(
                    f"refused: a reported cell holds fewer than {MIN_CELL_SIZE} respondents "
                    "(statistical disclosure control) — widen the grouping or the universe"
                )

    return QueryResult(
        columns=columns,
        rows=rows,
        weighted=acc.weight_is_summed,
        caveat="" if acc.weight_is_summed else UNWEIGHTED_BANNER,
    )
