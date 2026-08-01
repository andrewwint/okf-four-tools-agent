"""The `okf_query` guard — constrained aggregate SQL over the verified NHIS slice.

An LLM writes the SQL, so the SQL is untrusted. Worse than "user input": the agent's context
also carries text from the narrative and news tools, either of which an outsider can influence,
so treat every query as attacker-controlled.

Three layers, containment BEFORE parsing, because the parser is the layer that will have a bug:

  Layer 1  the DuckDB connection itself is locked down (no filesystem, no network, no
           extensions, bounded memory) and the data is bound by trusted code as relation `t`,
           so the model never needs — and cannot use — a file path.
  Layer 2  the SQL is parsed by DuckDB's OWN parser (json_serialize_sql) and asserted against
           a whitelist of node shapes: one statement, SELECT only, `t` is the only relation,
           no table-functions, no star, at least one aggregate, allow-listed functions.
  Layer 3  the RESULT is bounded: row cap, and a figure computed without the survey weight is
           labeled rather than returned bare.

Layer 1 alone refuses every hostile case in the test-suite with the parser stubbed out; Layer 2
is defence in depth, not the primary control. That ordering is deliberate — see the tests.

Why Layer 3 exists is not primarily security: a free-form SELECT over the slice returns
INDIVIDUAL survey records, which would falsify the aggregate-only, row-free guarantee this
project publishes. Enforcing aggregate-only is what keeps that claim true.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

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

# Layer 2 allow-list. Aggregates the analysis actually needs, plus arithmetic/comparison
# operators (DuckDB serializes those as functions too) and a few safe scalars.
_AGGREGATES = {
    "sum", "count", "count_star", "avg", "mean", "min", "max",
    "median", "quantile_cont", "quantile_disc", "stddev", "stddev_samp", "var_samp",
}
_SCALARS = {
    "+", "-", "*", "/", "%", "==", "=", "!=", "<>", "<", "<=", ">", ">=",
    "and", "or", "not", "abs", "round", "cast", "coalesce", "case",
    "float", "double", "integer", "bigint", "decimal",
}
_ALLOWED_FUNCTIONS = _AGGREGATES | _SCALARS

# Node types that may appear in a FROM clause. Anything else (notably TABLE_FUNCTION) is out.
_ALLOWED_FROM_TYPES = {"BASE_TABLE", "SUBQUERY", "JOIN", "EMPTY", "EMPTY_FROM"}


class QueryRejected(ValueError):
    """The query did not survive a guard layer. It was never executed."""


# --------------------------------------------------------------------------------------
# Layer 1 — the locked connection
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
    return [r[0] for r in con.execute(f"DESCRIBE {RELATION}").fetchall()]


def execute_contained(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    timeout: float = QUERY_TIMEOUT_SECONDS,
) -> duckdb.DuckDBPyConnection:
    """Execute under Layer-1 containment only — locked connection plus a wall-clock bound.

    Deliberately performs NO validation, so the tests can prove the containment holds with the
    parser stubbed out. A watchdog interrupts the engine rather than trusting the query to end.
    """
    timer = threading.Timer(timeout, con.interrupt)
    timer.daemon = True
    timer.start()
    try:
        return con.execute(sql)
    except duckdb.InterruptException as exc:
        raise QueryRejected(
            f"query exceeded the {timeout:g}s execution bound and was interrupted"
        ) from exc
    finally:
        timer.cancel()


# --------------------------------------------------------------------------------------
# Layer 2 — AST validation using DuckDB's own parser
# --------------------------------------------------------------------------------------

@dataclass
class _Walk:
    from_types: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    stars: int = 0
    table_functions: int = 0


def _walk(node, acc: _Walk) -> _Walk:
    if isinstance(node, dict):
        ntype = node.get("type")
        if ntype in ("BASE_TABLE", "TABLE_FUNCTION", "SUBQUERY", "JOIN"):
            acc.from_types.append(ntype)
        if ntype == "TABLE_FUNCTION":
            acc.table_functions += 1
        if ntype == "BASE_TABLE" and node.get("table_name"):
            acc.tables.append(node["table_name"])
        if ntype == "FUNCTION" and node.get("function_name"):
            acc.functions.append(str(node["function_name"]).lower())
        if ntype == "STAR":
            acc.stars += 1
        names = node.get("column_names")
        if isinstance(names, list):
            acc.columns.extend(str(n) for n in names)
        for value in node.values():
            _walk(value, acc)
    elif isinstance(node, list):
        for item in node:
            _walk(item, acc)
    return acc


def validate_sql(con: duckdb.DuckDBPyConnection, sql: str) -> _Walk:
    """Reject anything that is not a single, aggregate, `t`-only SELECT. Never executes.

    Uses DuckDB's own parser rather than a hand-rolled tokenizer: a bespoke lexer would have to
    re-implement comments, dollar-quoting, and unicode escapes correctly to be safe, and that is
    precisely the kind of parser-differential bug this design refuses to bet on.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise QueryRejected("empty query")

    try:
        raw = con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0]
        parsed = json.loads(raw)
    except Exception as exc:  # a parse failure is a rejection, never a fallback
        raise QueryRejected(f"could not parse: {exc}") from exc

    # DuckDB's serializer only emits SELECT; COPY/PRAGMA/ATTACH/INSTALL fail here by design.
    if parsed.get("error"):
        raise QueryRejected(
            f"not a single SELECT statement: {parsed.get('error_message', 'parse error')}"
        )

    statements = parsed.get("statements") or []
    if len(statements) != 1:
        raise QueryRejected(
            f"exactly one statement is allowed, found {len(statements)}"
        )

    acc = _walk(statements, _Walk())

    if acc.table_functions:
        raise QueryRejected(
            "table functions are not allowed (this is how a query reaches files and the network)"
        )
    if acc.stars:
        raise QueryRejected("`SELECT *` is not allowed — name aggregate expressions explicitly")

    # Every relation must be the one bound by trusted code. This also rejects CTEs, whose
    # references appear as base tables: a deliberate v1 restriction that removes recursive-CTE
    # resource exhaustion along with them.
    bad_tables = sorted({t for t in acc.tables if t != RELATION})
    if bad_tables:
        raise QueryRejected(
            f"only the relation {RELATION!r} may be queried; found {bad_tables}"
        )
    if not acc.tables:
        raise QueryRejected(f"query must read from {RELATION!r}")

    bad_from = sorted({f for f in acc.from_types if f not in _ALLOWED_FROM_TYPES})
    if bad_from:
        raise QueryRejected(f"disallowed FROM construct: {bad_from}")

    bad_funcs = sorted({f for f in acc.functions if f not in _ALLOWED_FUNCTIONS})
    if bad_funcs:
        raise QueryRejected(f"function(s) not on the allow-list: {bad_funcs}")

    if not any(f in _AGGREGATES for f in acc.functions):
        raise QueryRejected(
            "aggregate-only: the query must use an aggregate (this tool never returns rows)"
        )

    known = {c.lower() for c in guarded_columns(con)}
    unknown = sorted({c for c in acc.columns if c.lower() not in known})
    if unknown:
        raise QueryRejected(f"unknown column(s): {unknown}")

    return acc


# --------------------------------------------------------------------------------------
# Layer 3 — result shape and the weight label
# --------------------------------------------------------------------------------------

def _uses_weight(sql: str, acc: _Walk) -> bool:
    if any(c.lower() == WEIGHT_COLUMN.lower() for c in acc.columns):
        return True
    return re.search(rf"\b{WEIGHT_COLUMN}\b", sql, re.IGNORECASE) is not None


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


def run_query(con: duckdb.DuckDBPyConnection, sql: str) -> QueryResult:
    """Validate, execute on the locked connection, then bound the result. The tool entrypoint."""
    acc = validate_sql(con, sql)

    cursor = execute_contained(con, sql)
    rows = cursor.fetchmany(MAX_ROWS + 1)
    if len(rows) > MAX_ROWS:
        raise QueryRejected(
            f"result exceeds {MAX_ROWS} rows — this tool returns aggregates, not records"
        )
    columns = [d[0] for d in cursor.description]

    weighted = _uses_weight(sql, acc)
    return QueryResult(
        columns=columns,
        rows=[tuple(r) for r in rows],
        weighted=weighted,
        caveat="" if weighted else UNWEIGHTED_BANNER,
    )
