"""Hostile-input suite for the `okf_query` guard.

Two load-bearing classes:

  TestLayer1ContainmentWithParserStubbedOut  — fires every attack straight at the connection
      with NO validation in the path. If those pass, containment is real and the parser is
      defence in depth. If they only fail through the parser, the design is one parser bug away
      from arbitrary file read.

  TestRecordDisclosure — asserts the record *ban*, not the row *cap*. An earlier version of this
      suite tested only the cap, and an independent review walked 200 complete respondent
      records out through queries that satisfied every check. Those exact exploits are here.
"""

from __future__ import annotations

import sys
import threading
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

REAL_SLICE = Path(
    "/Users/andrewwint/Documents/PROJECTS/PERSONAL/nhisokfchat/app/nhisokfchat/"
    "nhis_okf/microdata/adult23_slice.parquet"
)

# Every one of these must be refused. Each is a real capability of a default DuckDB.
HOSTILE = {
    "arbitrary file read": "SELECT * FROM read_text('/etc/hostname')",
    "csv file read": "SELECT count(*) FROM read_csv('/etc/hosts')",
    "blob read": "SELECT count(*) FROM read_blob('/etc/hostname')",
    "glob": "SELECT count(*) FROM glob('/etc/*')",
    "network fetch": "SELECT count(*) FROM read_csv('https://example.invalid/x.csv')",
    "file write": "COPY (SELECT 1) TO '/tmp/pwned.csv'",
    "attach database": "ATTACH '/tmp/x.db' AS x",
    "attach remote": "ATTACH 'http://example.invalid/x.db' AS x",
    "install extension": "INSTALL httpfs",
    "load extension": "LOAD httpfs",
    "re-enable access": "SET enable_external_access=true",
    "unlock config": "SET lock_configuration=false",
    # PRAGMA forms of the same attacks — a separate syntax path from SET.
    "pragma re-enable": "PRAGMA enable_external_access=true",
    "pragma unlock": "PRAGMA lock_configuration=false",
    "pragma autoload": "PRAGMA autoload_known_extensions=true",
    "pragma extension dir": "PRAGMA extension_directory='/tmp'",
    "stacked statement": "SELECT count(*) FROM t; SELECT * FROM read_text('/etc/hostname')",
    "nested table fn": "SELECT count(*) FROM (SELECT * FROM read_csv('/etc/hosts'))",
    "cte table fn": "WITH x AS (SELECT * FROM read_text('/etc/hostname')) SELECT count(*) FROM x",
    "join table fn": "SELECT count(*) FROM t JOIN read_csv('/etc/hosts') ON 1=1",
    "recursive cte": (
        "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM r WHERE i<100000000) "
        "SELECT count(*) FROM r"
    ),
}

# The exploits an independent review used to extract complete respondent records past an
# aggregate check that only asked whether an aggregate appeared anywhere in the tree.
DISCLOSURE = {
    "bare projection via subquery aggregate": (
        "SELECT SEX_A, DIBEV_A, DIBINS_A, PREDIB_A, DIBAGETC_A, PSTRAT, PPSU, WTFA_A "
        f"FROM {RELATION} WHERE DIBAGETC_A >= (SELECT min(DIBAGETC_A) FROM {RELATION}) LIMIT 8"
    ),
    "group by the person-level weight": (
        "SELECT WTFA_A, MIN(SEX_A), MIN(DIBEV_A), MIN(DIBAGETC_A), MIN(PSTRAT), MIN(PPSU) "
        f"FROM {RELATION} GROUP BY WTFA_A LIMIT 12"
    ),
    "singleton isolation by weight": (
        f"SELECT MIN(DIBAGETC_A) FROM {RELATION} WHERE WTFA_A = 3146.794"
    ),
    "design fields projected": f"SELECT PSTRAT, PPSU, count(*) FROM {RELATION} GROUP BY 1, 2",
    "paging records": (
        f"SELECT SEX_A, DIBAGETC_A, WTFA_A FROM {RELATION} "
        "WHERE DIBEV_A = 1 LIMIT 8 OFFSET 24"
    ),
}

# Unweighted figures dressed up to look weighted. All must keep the caveat.
WEIGHT_SPOOFS = {
    "line comment": f"SELECT count(*) AS n FROM {RELATION} -- WTFA_A",
    "block comment": f"SELECT count(*) AS n FROM {RELATION} /* WTFA_A */",
    "alias": f'SELECT count(*) AS "WTFA_A" FROM {RELATION}',
    "mentioned only in where": (
        f"SELECT avg(DIBAGETC_A) AS a FROM {RELATION} WHERE DIBEV_A = 1"
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
          SELECT (i %% 2) + 1                                   AS DIBEV_A,
                 CASE WHEN i %% 2 = 0 THEN NULL ELSE (i %% 2) + 1 END AS DIBINS_A,
                 (i %% 3) + 1                                   AS PREDIB_A,
                 CAST(40 + (i %% 40) AS DOUBLE)                 AS DIBAGETC_A,
                 (i %% 2) + 1                                   AS SEX_A,
                 1000.0 + i                                     AS WTFA_A,
                 100 + (i %% 10)                                AS PSTRAT,
                 (i %% 4) + 1                                   AS PPSU
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

class TestLayer1ContainmentWithParserStubbedOut:
    """Locked config + execution bound only. No validation anywhere in this path."""

    @pytest.mark.parametrize("name,sql", sorted(HOSTILE.items()))
    def test_containment_refuses_without_the_parser(self, con, name, sql):
        with pytest.raises(Exception) as excinfo:
            execute_contained(con, sql, timeout=2.0)
        message = str(excinfo.value).lower()
        assert any(
            marker in message
            for marker in (
                "permission", "cannot change", "not allowed", "disabled",
                "no function", "execution bound", "could not be executed",
            )
        ), f"{name!r} failed for an unexpected reason: {excinfo.value}"

    def test_resource_exhaustion_is_bounded_by_the_clock_not_the_config(self, con):
        """memory_limit bounds memory, not CPU — so containment needs a clock too."""
        runaway = f"SELECT count(*) FROM {RELATION} a, {RELATION} b, {RELATION} c"
        with pytest.raises(QueryRejected, match="execution bound"):
            execute_contained(con, runaway, timeout=2.0)

    def test_the_bound_covers_the_fetch_not_just_the_execute(self, con):
        """A streaming cursor returns from execute instantly, so the clock must survive the fetch.

        The row count has to be large enough that the fetch genuinely cannot finish inside the
        bound — DuckDB materialises ~10M rows/sec here, so a smaller ask would complete honestly
        and prove nothing. Note the interrupt is checked at chunk boundaries, so it is a
        best-effort bound rather than a hard deadline: it lands shortly after the timer, not on it.
        """
        wide = f"SELECT a.SEX_A FROM {RELATION} a, {RELATION} b"
        with pytest.raises(QueryRejected):
            execute_contained(con, wide, timeout=1.0, max_rows=400_000_000)

    def test_the_data_is_still_reachable(self, con):
        columns, rows = execute_contained(con, f"SELECT count(*) AS n FROM {RELATION}")
        assert rows[0][0] > 0 and columns == ["n"]

    def test_configuration_cannot_be_unlocked(self, con):
        for stmt in (
            "SET enable_external_access=true",
            "SET lock_configuration=false",
            "PRAGMA enable_external_access=true",
        ):
            with pytest.raises(QueryRejected):
                execute_contained(con, stmt)


# ---------------------------------------------------------------------------------------
# The published invariant: aggregate-only, no individual records
# ---------------------------------------------------------------------------------------

class TestRecordDisclosure:
    """Assert the record BAN, not the row CAP. These are proven-working exploits."""

    @pytest.mark.parametrize("name,sql", sorted(DISCLOSURE.items()))
    def test_record_disclosure_is_refused(self, con, name, sql):
        with pytest.raises(QueryRejected):
            run_query(con, sql)

    def test_bare_high_cardinality_column_is_refused(self, con):
        for column in ("WTFA_A", "PSTRAT", "PPSU", "DIBAGETC_A"):
            with pytest.raises(QueryRejected, match="inside an aggregate"):
                run_query(con, f"SELECT {column}, count(*) FROM {RELATION} GROUP BY {column}")

    def test_high_cardinality_column_is_fine_inside_an_aggregate(self, con):
        result = run_query(con, f"SELECT SUM(WTFA_A) AS w FROM {RELATION}")
        assert len(result.rows) == 1

    def test_categorical_grouping_is_allowed(self, con):
        result = run_query(
            con,
            f"SELECT SEX_A, DIBEV_A, SUM(WTFA_A) AS w FROM {RELATION} GROUP BY SEX_A, DIBEV_A",
        )
        assert 0 < len(result.rows) <= MAX_ROWS


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

    # Legitimate SQL an LLM actually emits must NOT be refused — a high false-reject rate
    # pushes the model toward stranger queries to satisfy the tool.
    @pytest.mark.parametrize(
        "sql",
        [
            f"SELECT count({RELATION}.WTFA_A) AS n FROM {RELATION}",
            f"SELECT count(u.WTFA_A) AS n FROM {RELATION} AS u",
            f"SELECT SEX_A AS s, count(*) AS n FROM {RELATION} GROUP BY s",
            f"SELECT SEX_A, count(*) AS n FROM {RELATION} GROUP BY SEX_A ORDER BY n",
        ],
    )
    def test_legitimate_sql_is_accepted(self, con, sql):
        validate_sql(con, sql)


# ---------------------------------------------------------------------------------------
# Layer 3 — result shape and the weight label
# ---------------------------------------------------------------------------------------

class TestLayer3ResultShape:

    def test_unweighted_result_is_labeled(self, con):
        result = run_query(con, f"SELECT count(*) AS n FROM {RELATION}")
        assert result.weighted is False
        assert UNWEIGHTED_BANNER in result.render()

    @pytest.mark.parametrize("name,sql", sorted(WEIGHT_SPOOFS.items()))
    def test_the_weight_label_cannot_be_spoofed(self, con, name, sql):
        """`-- WTFA_A` must not strip the caveat: the weight has to be truly aggregated."""
        result = run_query(con, sql)
        assert result.weighted is False, f"{name!r} spoofed the weight label"
        assert UNWEIGHTED_BANNER in result.render()

    def test_genuinely_weighted_result_is_not_labeled(self, con):
        result = run_query(con, f"SELECT SUM(WTFA_A) AS w FROM {RELATION} WHERE DIBEV_A = 1")
        assert result.weighted is True
        assert UNWEIGHTED_BANNER not in result.render()

    def test_row_cap_still_applies(self, con):
        with pytest.raises(QueryRejected, match="exceeds|inside an aggregate"):
            run_query(con, f"SELECT DIBAGETC_A, count(*) FROM {RELATION} GROUP BY DIBAGETC_A")


# ---------------------------------------------------------------------------------------
# Fail closed: no engine text ever reaches the caller
# ---------------------------------------------------------------------------------------

class TestFailClosed:

    def test_engine_errors_become_QueryRejected(self, con):
        """A validated query that fails in the engine must not leak engine text."""
        sql = f"SELECT count(*) AS n FROM {RELATION} WHERE SEX_A = 'abc'"
        with pytest.raises(QueryRejected) as excinfo:
            run_query(con, sql)
        message = str(excinfo.value)
        for leak in ("Conversion Error", "LINE 1", "duckdb", "^"):
            assert leak not in message, f"engine text leaked: {message}"

    def test_parameter_marker_does_not_escape_as_a_raw_error(self, con):
        with pytest.raises(QueryRejected):
            run_query(con, f"SELECT count(*) AS n FROM {RELATION} WHERE SEX_A = ?")


# ---------------------------------------------------------------------------------------
# Concurrency: results must never cross between callers
# ---------------------------------------------------------------------------------------

class TestConcurrency:

    def test_results_do_not_cross_between_threads(self, con):
        errors: list[str] = []

        def worker(sex: int, expected_label: str) -> None:
            for _ in range(25):
                try:
                    result = run_query(
                        con,
                        f"SELECT count(*) AS {expected_label} FROM {RELATION} WHERE SEX_A = {sex}",
                    )
                    if result.columns != [expected_label]:
                        errors.append(f"columns crossed: {result.columns} != [{expected_label}]")
                except Exception as exc:  # a crash is also a failure of isolation
                    errors.append(f"{type(exc).__name__}: {exc}")

        threads = [
            threading.Thread(target=worker, args=(1, "males")),
            threading.Thread(target=worker, args=(2, "females")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"{len(errors)} isolation failures, first: {errors[0]}"


# ---------------------------------------------------------------------------------------
# The query the capability concept declares
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
