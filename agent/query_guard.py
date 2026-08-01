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
           STRUCTURALLY: one statement, SELECT only, `t` is the only relation, no
           table-functions, no star, at least one aggregate, allow-listed functions — and the
           rule that does the real work below.
  Layer 3  the RESULT is bounded: row cap, and a figure is reported as weighted only when the
           survey weight is genuinely aggregated.

**The rule that makes this aggregate-only:** every *bare* column reference — one that is not an
argument to an aggregate — must be a low-cardinality categorical column. High-cardinality
columns (the person-level weight, the design stratum and PSU, age at diagnosis) may appear ONLY
inside an aggregate. One rule covers the projection, the GROUP BY, and the WHERE at once, which
is what closes every record-disclosure route: you cannot project a respondent's weight, you
cannot group by it to make groups of one, and you cannot filter on it to isolate a person.

An earlier version of this module checked only that an aggregate appeared *somewhere* in the
tree. An independent review broke it in minutes — an aggregate in a WHERE subquery satisfied the
check while the projection stayed bare, and 200 complete respondent records (including PSTRAT,
PPSU and the exact person-level weight) came out in four calls. Shape-matching is not a control;
structure is.
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
# satisfy "bounded cost" — containment needs a clock as well as a lock.
QUERY_TIMEOUT_SECONDS = 5.0

# Layer 3 bounds.
MAX_ROWS = 50
WEIGHT_COLUMN = "WTFA_A"
UNWEIGHTED_BANNER = "UNWEIGHTED — a raw sample figure, NOT a population estimate."

# Columns that may appear OUTSIDE an aggregate: low-cardinality categoricals whose groups are
# always many people. Everything else in the slice — WTFA_A, PSTRAT, PPSU, DIBAGETC_A — is
# aggregate-argument-only, which is what prevents singling out an individual.
#
# In the finished system this list is declared by the OKF capability concept (the concept
# declares which shapes are permitted, and its example query is verified by execution); it is a
# module constant here so the guard can be built and attacked before the concept exists.
GROUPABLE_COLUMNS = frozenset({"DIBEV_A", "DIBINS_A", "PREDIB_A", "SEX_A"})

_AGGREGATES = {
    "sum", "count", "count_star", "avg", "mean", "min", "max",
    "median", "quantile_cont", "quantile_disc", "stddev", "stddev_samp", "var_samp",
}
_SCALARS = {
    "+", "-", "*", "/", "%", "==", "=", "!=", "<>", "<", "<=", ">", ">=",
    "and", "or", "not", "in", "abs", "round", "cast", "coalesce", "case",
    "float", "double", "integer", "bigint", "decimal",
}
_ALLOWED_FUNCTIONS = _AGGREGATES | _SCALARS

# Node types permitted in a FROM clause. Anything else — notably TABLE_FUNCTION — is out.
_ALLOWED_FROM_TYPES = {"BASE_TABLE", "SUBQUERY", "JOIN", "EMPTY", "EMPTY_FROM"}

# A single fixed refusal for any engine-level failure. Engine text is never returned to the
# caller: it is both an injection channel and a blind-inference oracle (wrap a conditional CAST
# failure around a predicate and read the answer out of whether an error comes back).
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
    # Trusted load, by trusted code, with a path the model never sees.
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

    Deliberately performs NO validation, so the tests can prove containment holds with the
    parser stubbed out. Returns fully-materialised results: the clock must cover the fetch as
    well as the execute, so this never hands back a live cursor.

    Each call gets its OWN cursor. `con.execute` returns the connection itself, so two callers
    sharing one connection overwrite each other's results — and `DESCRIBE` could then report
    another query's columns, which would corrupt validation itself.
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
    except Exception as exc:  # never surface engine text to the caller
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
    bare_columns: list[str] = field(default_factory=list)   # outside every aggregate
    agg_columns: list[str] = field(default_factory=list)    # arguments to an aggregate
    qualifiers: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    stars: int = 0
    table_functions: int = 0
    aggregates: int = 0


def _analyze(node, acc: _Analysis, inside_aggregate: bool = False) -> _Analysis:
    if isinstance(node, dict):
        ntype = node.get("type")

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

        if ntype == "COLUMN_REF":
            names = node.get("column_names") or []
            if names:
                # A qualified reference serialises as ["t", "WTFA_A"]; the column is the last.
                if len(names) > 1:
                    acc.qualifiers.append(str(names[0]))
                column = str(names[-1])
                (acc.agg_columns if inside_aggregate else acc.bare_columns).append(column)

        if ntype == "FUNCTION" and node.get("function_name"):
            name = str(node["function_name"]).lower()
            acc.functions.append(name)
            is_aggregate = name in _AGGREGATES
            if is_aggregate:
                acc.aggregates += 1
            for value in node.values():
                _analyze(value, acc, inside_aggregate or is_aggregate)
            return acc

        for value in node.values():
            _analyze(value, acc, inside_aggregate)

    elif isinstance(node, list):
        for item in node:
            _analyze(item, acc, inside_aggregate)

    return acc


def validate_sql(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    groupable: frozenset[str] = GROUPABLE_COLUMNS,
) -> _Analysis:
    """Reject anything that is not a single, aggregate-only, `t`-only SELECT. Never executes.

    Uses DuckDB's own parser rather than a hand-rolled tokenizer: a bespoke lexer would have to
    re-implement comments, dollar-quoting and unicode escapes correctly to be safe, and that is
    exactly the parser-differential bet this design refuses to make.
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

    # DuckDB's serializer only emits SELECT; COPY/PRAGMA/ATTACH/INSTALL/SET fail here by design.
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

    # Every relation must be the one bound by trusted code. This also rejects CTEs, whose
    # references appear as base tables: a deliberate restriction that removes recursive-CTE
    # resource exhaustion along with them.
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
    allowed_qualifiers = {RELATION.lower()} | {a.lower() for a in acc.aliases}
    bad_qualifiers = sorted({q for q in acc.qualifiers if q.lower() not in allowed_qualifiers})
    if bad_qualifiers:
        raise QueryRejected(f"unknown table qualifier(s): {bad_qualifiers}")

    # Select-list aliases may legitimately be referenced by GROUP BY / ORDER BY.
    alias_names = {a.lower() for a in acc.aliases}
    referenced = {c.lower() for c in acc.bare_columns + acc.agg_columns}
    unknown = sorted({c for c in referenced if c not in known and c not in alias_names})
    if unknown:
        raise QueryRejected(f"unknown column(s): {unknown}")

    # THE rule. A bare column is one not wrapped in an aggregate: projected, grouped, or
    # filtered. Restricting those to low-cardinality categoricals is what makes singling out an
    # individual structurally impossible — you cannot project a person's weight, group by it to
    # make groups of one, or filter on it to isolate a row.
    groupable_lower = {g.lower() for g in groupable}
    disclosive = sorted(
        {
            c for c in acc.bare_columns
            if c.lower() not in groupable_lower and c.lower() not in alias_names
        }
    )
    if disclosive:
        raise QueryRejected(
            f"column(s) {disclosive} may only appear inside an aggregate — bare use is limited "
            f"to the low-cardinality columns {sorted(groupable)}"
        )

    return acc


# --------------------------------------------------------------------------------------
# Layer 3 — result shape and the weight label
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
    acc = validate_sql(con, sql, groupable=groupable)

    columns, rows = execute_contained(con, sql)
    if len(rows) > MAX_ROWS:
        raise QueryRejected(
            f"result exceeds {MAX_ROWS} rows — this tool returns aggregates, not records"
        )

    # Weighted only when the survey weight is genuinely AGGREGATED. Anything looser (a substring
    # scan of the SQL) is spoofable by a comment — `-- WTFA_A` would strip the caveat off an
    # unweighted figure, which is the one direction that matters for a statistical claim.
    weighted = any(c.lower() == WEIGHT_COLUMN.lower() for c in acc.agg_columns)

    return QueryResult(
        columns=columns,
        rows=rows,
        weighted=weighted,
        caveat="" if weighted else UNWEIGHTED_BANNER,
    )
