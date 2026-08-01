"""Hostile-input suite for the `okf_query` guard.

The load-bearing test class is `TestLayer1EngineAloneWithParserStubbedOut`: it bypasses the
validator entirely and fires the same attacks straight at the connection. If those pass, the
containment is real and the parser is defence in depth. If they only fail through the parser,
the design is one parser bug away from arbitrary file read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.query_guard import (  # noqa: E402
    MAX_ROWS,
    QueryRejected,
    RELATION,
    UNWEIGHTED_BANNER,
    build_guarded_connection,
    execute_contained,
    run_query,
    validate_sql,
)

# The real verified slice, if it is available on this machine.
REAL_SLICE = Path(
    "/Users/andrewwint/Documents/PROJECTS/PERSONAL/nhisokfchat/app/nhisokfchat/"
    "nhis_okf/microdata/adult23_slice.parquet"
)

# Every one of these must be refused. Each is a real capability of a default DuckDB.
HOSTILE = {
    "arbitrary file read": "SELECT * FROM read_text('/etc/hostname')",
    "csv file read": "SELECT count(*) FROM read_csv('/etc/hosts')",
    "network fetch": "SELECT count(*) FROM read_csv('https://example.invalid/x.csv')",
    "file write": "COPY (SELECT 1) TO '/tmp/pwned.csv'",
    "attach database": "ATTACH '/tmp/x.db' AS x",
    "install extension": "INSTALL httpfs",
    "load extension": "LOAD httpfs",
    "re-enable access": "SET enable_external_access=true",
    "unlock config": "SET lock_configuration=false",
    "stacked statement": "SELECT count(*) FROM t; SELECT * FROM read_text('/etc/hostname')",
    "nested table fn": "SELECT count(*) FROM (SELECT * FROM read_csv('/etc/hosts'))",
    "cte table fn": "WITH x AS (SELECT * FROM read_text('/etc/hostname')) SELECT count(*) FROM x",
    "join table fn": "SELECT count(*) FROM t JOIN read_csv('/etc/hosts') ON 1=1",
    "recursive cte": (
        "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM r WHERE i<1000000) "
        "SELECT count(*) FROM r"
    ),
}


@pytest.fixture(scope="module")
def slice_path(tmp_path_factory) -> Path:
    """The real verified slice when present, else a synthetic stand-in with the same columns."""
    if REAL_SLICE.is_file():
        return REAL_SLICE
    out = tmp_path_factory.mktemp("data") / "slice.parquet"
    con = duckdb.connect(":memory:")  # unlocked: fixture setup is trusted code
    con.execute(
        f"""
        COPY (
          SELECT (i %% 2) + 1        AS DIBEV_A,
                 (i %% 2) + 1        AS DIBINS_A,
                 (i %% 3) + 1        AS PREDIB_A,
                 40 + (i %% 40)      AS DIBAGETC_A,
                 (i %% 2) + 1        AS SEX_A,
                 1000.0 + i          AS WTFA_A,
                 100 + (i %% 10)     AS PSTRAT,
                 (i %% 4) + 1        AS PPSU
          FROM range(5000) tbl(i)
        ) TO '{out}' (FORMAT parquet)
        """
    )
    con.close()
    return out


@pytest.fixture(scope="module")
def con(slice_path):
    connection = build_guarded_connection(slice_path)
    yield connection
    connection.close()


# ---------------------------------------------------------------------------------------
# Layer 1 — the control that must hold on its own
# ---------------------------------------------------------------------------------------

class TestLayer1EngineAloneWithParserStubbedOut:
    """Fire every attack directly at the connection, with NO validation in the path."""

    @pytest.mark.parametrize("name,sql", sorted(HOSTILE.items()))
    def test_containment_refuses_without_the_parser(self, con, name, sql):
        """Locked config + execution bound only. No validation anywhere in this path."""
        with pytest.raises(Exception) as excinfo:
            execute_contained(con, sql, timeout=2.0).fetchall()
        # A refusal, not an incidental error: a permission/config denial or the wall-clock bound.
        message = str(excinfo.value).lower()
        assert any(
            marker in message
            for marker in (
                "permission", "cannot change", "not allowed", "disabled",
                "no function", "execution bound",
            )
        ), f"{name!r} failed for an unexpected reason: {excinfo.value}"

    def test_resource_exhaustion_is_bounded_by_the_clock_not_the_config(self, con):
        """The defect the stubbed-parser suite found: memory_limit does not bound CPU."""
        runaway = (
            "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM r WHERE i<100000000) "
            "SELECT count(*) FROM r"
        )
        with pytest.raises(QueryRejected, match="execution bound"):
            execute_contained(con, runaway, timeout=2.0).fetchall()

    def test_the_bound_does_not_penalise_a_legitimate_query(self, con):
        cursor = execute_contained(con, f"SELECT count(*) FROM {RELATION}", timeout=2.0)
        assert cursor.fetchone()[0] > 0

    def test_the_data_is_still_reachable(self, con):
        assert con.execute(f"SELECT count(*) FROM {RELATION}").fetchone()[0] > 0

    def test_configuration_cannot_be_unlocked(self, con):
        for stmt in (
            "SET enable_external_access=true",
            "SET lock_configuration=false",
            "SET autoload_known_extensions=true",
        ):
            with pytest.raises(Exception):
                con.execute(stmt)


# ---------------------------------------------------------------------------------------
# Layer 2 — the validator, as defence in depth
# ---------------------------------------------------------------------------------------

class TestLayer2Validator:

    @pytest.mark.parametrize("name,sql", sorted(HOSTILE.items()))
    def test_validator_refuses_before_execution(self, con, name, sql):
        with pytest.raises(QueryRejected):
            validate_sql(con, sql)

    def test_star_is_refused(self, con):
        with pytest.raises(QueryRejected, match="SELECT \\*"):
            validate_sql(con, f"SELECT * FROM {RELATION}")

    def test_non_aggregate_is_refused(self, con):
        with pytest.raises(QueryRejected, match="aggregate-only"):
            validate_sql(con, f"SELECT DIBEV_A FROM {RELATION}")

    def test_unknown_column_is_refused(self, con):
        with pytest.raises(QueryRejected, match="unknown column"):
            validate_sql(con, f"SELECT count(SSN) FROM {RELATION}")

    def test_unknown_relation_is_refused(self, con):
        with pytest.raises(QueryRejected):
            validate_sql(con, "SELECT count(*) FROM sqlite_master")

    def test_empty_is_refused(self, con):
        with pytest.raises(QueryRejected):
            validate_sql(con, "   ")


# ---------------------------------------------------------------------------------------
# Layer 3 — result shape and the weight label
# ---------------------------------------------------------------------------------------

class TestLayer3ResultShape:

    def test_unweighted_result_is_labeled(self, con):
        result = run_query(con, f"SELECT count(*) AS n FROM {RELATION}")
        assert result.weighted is False
        assert UNWEIGHTED_BANNER in result.render()

    def test_weighted_result_is_not_labeled(self, con):
        result = run_query(
            con,
            f"SELECT SUM(WTFA_A) AS w FROM {RELATION} WHERE DIBEV_A = 1",
        )
        assert result.weighted is True
        assert UNWEIGHTED_BANNER not in result.render()

    def test_row_cap_rejects_wide_group_by(self, con):
        with pytest.raises(QueryRejected, match="exceeds"):
            run_query(
                con,
                f"SELECT WTFA_A, count(*) FROM {RELATION} GROUP BY WTFA_A",
            )

    def test_small_group_by_is_allowed(self, con):
        result = run_query(
            con,
            f"SELECT SEX_A, SUM(WTFA_A) AS w FROM {RELATION} GROUP BY SEX_A",
        )
        assert 0 < len(result.rows) <= MAX_ROWS


# ---------------------------------------------------------------------------------------
# The query the capability concept declares — the figure that must survive all three layers
# ---------------------------------------------------------------------------------------

class TestVerifiedExampleQuery:

    @pytest.mark.skipif(not REAL_SLICE.is_file(), reason="real verified slice not present")
    def test_insulin_prevalence_matches_the_verified_concept(self, con):
        sql = (
            "SELECT SUM(WTFA_A) FILTER (WHERE DIBINS_A = 1) / SUM(WTFA_A) * 100 AS pct "
            f"FROM {RELATION} WHERE DIBEV_A = 1 AND DIBINS_A IN (1, 2)"
        )
        result = run_query(con, sql)
        assert result.weighted is True
        assert round(result.rows[0][0], 2) == 31.96
