"""Layer 1 containment for the query tool: a DuckDB connection that can only do one thing.

This module deliberately contains NO SQL validation. An earlier version did, and three rounds of
independent adversarial review established that validating model-authored SQL cannot be made
disclosure-safe (see `docs/DECISION-sql-surface.md`). The tool surface is now declared query
shapes (`agent/query_shapes.py`) where trusted code owns every SQL string.

What survived that finding — unbroken across all three rounds — is this layer:

  * the connection cannot touch the filesystem, the network, or extensions, and its
    configuration is locked so nothing can re-enable them
  * the slice is bound by trusted code as relation `t`, so no caller ever names a path
  * every execution is bounded by a wall clock, because `memory_limit` bounds memory and not CPU

It is kept because containment is what makes a bug in any higher layer survivable, and because
its guarantees are provable with everything above it stubbed out — which is exactly how the
hostile suite tests it.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import duckdb

log = logging.getLogger(__name__)

# The one relation the query tool may read. Bound by trusted code at startup.
RELATION = "t"

# Wall-clock bound on any single execution.
#
# Found by the hostile suite, not by design review: `memory_limit` bounds MEMORY, not CPU, so a
# recursive CTE runs to completion on an otherwise fully locked connection. Config alone does not
# satisfy "bounded cost" — containment needs a clock as well as a lock. The interrupt is checked
# at chunk boundaries, so this is a best-effort bound (sub-second overshoot), not a hard deadline.
QUERY_TIMEOUT_SECONDS = 5.0

# Result bounds, enforced by the shape layer.
MAX_ROWS = 50
MIN_CELL_SIZE = 30  # statistical disclosure control: no reported cell may be smaller

_ENGINE_ERROR = "the query could not be executed"


class QueryRejected(ValueError):
    """A query was refused, or failed in the engine. No engine detail reaches the caller."""


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
    """The real column names of the bound relation."""
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
    """Execute under containment: locked connection, own cursor, wall clock.

    Returns materialised results — the clock must cover the fetch as well as the execute, so this
    never hands back a live cursor.

    Each call gets its OWN cursor. `con.execute` returns the connection itself, so two callers
    sharing one connection overwrite each other's results, and a concurrent `DESCRIBE` could
    report another query's columns.
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
