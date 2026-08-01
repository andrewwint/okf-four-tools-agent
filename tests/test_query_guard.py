"""Hostile-input suite for the `okf_query` guard.

Three load-bearing classes:

  TestLayer1ContainmentWithParserStubbedOut — fires every attack straight at the connection with
      NO validation in the path. If those pass, containment is real and the parser is defence in
      depth, not the primary control.

  TestRecordDisclosure — asserts the record BAN, not the row CAP, and parameterises over the
      *routes* to a person rather than specific exploit strings: a high-cardinality column is
      tried through every syntactic door (projection, GROUP BY, WHERE, FILTER, CASE WHEN,
      HAVING, self-alias). Two rounds of independent review broke this guard by relocating the
      same predicate into a new door, so the suite is written to catch relocation.

  TestCellSize — the invariant that low cardinality does not provide: group size.
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
    MIN_CELL_SIZE,
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
    "pragma re-enable": "PRAGMA enable_external_access=true",
    "pragma unlock": "PRAGMA lock_configuration=false",
    "pragma autoload": "PRAGMA autoload_known_extensions=true",
    "pragma extension dir": "PRAGMA extension_directory='/tmp'",
    "pragma profiling": "PRAGMA enable_profiling",
    "stacked statement": "SELECT count(*) FROM t; SELECT * FROM read_text('/etc/hostname')",
    "nested table fn": "SELECT count(*) FROM (SELECT * FROM read_csv('/etc/hosts'))",
    "cte table fn": "WITH x AS (SELECT * FROM read_text('/etc/hostname')) SELECT count(*) FROM x",
    "join table fn": "SELECT count(*) FROM t JOIN read_csv('/etc/hosts') ON 1=1",
    "recursive cte": (
        "WITH RECURSIVE r(i) AS (SELECT 1 UNION ALL SELECT i+1 FROM r WHERE i<100000000) "
        "SELECT count(*) FROM r"
    ),
}

# Honest carve-out: `lock_configuration` does NOT cover `enable_profiling` via PRAGMA, so this
# one is refused by Layer 2 only. It cannot reach the filesystem or the network (profiling output
# is locked and file writes are denied), so the consequence is log noise rather than a
# containment failure — but the Layer-1 class must not claim to block what it does not.
LAYER2_ONLY = {"pragma profiling"}
CONTAINED = {k: v for k, v in HOSTILE.items() if k not in LAYER2_ONLY}

# The columns that identify a person. None may reach a projection, a grouping, or any predicate.
PROTECTED = ["WTFA_A", "PSTRAT", "PPSU", "DIBAGETC_A"]

# Every syntactic door a protected column could walk through. Parameterised over the ROUTE so a
# relocated predicate is caught by construction — that is exactly how this guard was broken twice.
def _routes(column: str) -> dict[str, str]:
    return {
        "projected": f"SELECT {column}, count(*) AS n FROM {RELATION} GROUP BY {column}",
        "grouped": f"SELECT MIN(SEX_A) AS s, count(*) AS n FROM {RELATION} GROUP BY {column}",
        "bare where": f"SELECT MIN(SEX_A) AS s, count(*) AS n FROM {RELATION} WHERE {column} = 1",
        "filter clause": (
            f"SELECT MIN(SEX_A) FILTER (WHERE {column} = 1) AS s, count(*) AS n FROM {RELATION}"
        ),
        "case when": (
            f"SELECT sum(CASE WHEN {column} = 1 THEN 1 ELSE 0 END) AS s, count(*) AS n "
            f"FROM {RELATION}"
        ),
        "having": (
            f"SELECT SEX_A, count(*) AS n FROM {RELATION} GROUP BY SEX_A "
            f"HAVING MIN({column}) < 5000"
        ),
        "self alias": (
            f"SELECT {column} AS {column}, count(*) AS n FROM {RELATION} GROUP BY {column}"
        ),
        "aliased elsewhere": (
            f"SELECT count(*) AS {column}, {column}, count(*) AS n FROM {RELATION} "
            f"GROUP BY {column}"
        ),
        "subquery aggregate": (
            f"SELECT SEX_A, {column}, count(*) AS n FROM {RELATION} "
            f"WHERE {column} >= (SELECT min({column}) FROM {RELATION}) GROUP BY SEX_A, {column}"
        ),
    }


DISCLOSURE = {
    f"{column} via {route}": sql
    for column in PROTECTED
    for route, sql in _routes(column).items()
}

# Unweighted figures dressed up to look weighted. All must keep the caveat.
WEIGHT_SPOOFS = {
    "line comment": f"SELECT count(*) AS n FROM {RELATION} -- WTFA_A",
    "block comment": f"SELECT count(*) AS n FROM {RELATION} /* WTFA_A */",
    "alias": f'SELECT count(*) AS "WTFA_A", count(*) AS n FROM {RELATION}',
    "count of the weight": f"SELECT count(WTFA_A) AS c, count(*) AS n FROM {RELATION}",
    "min of the weight": f"SELECT min(WTFA_A) AS m, count(*) AS n FROM {RELATION}",
    "max of the weight": f"SELECT max(WTFA_A) AS m, count(*) AS n FROM {RELATION}",
    "weight only in a filter": (
        f"SELECT count(*) FILTER (WHERE DIBEV_A = 1) AS c, count(*) AS n FROM {RELATION}"
    ),
}


@pytest.fixture(scope="module")
def slice_path(tmp_path_factory) -> Path:
    if REAL_SLICE.is_file():
        return REAL_SLICE
    out = tmp_path_factory.mktemp("data") / "slice.parquet"
    con = duckdb.connect(":memory:")  # unlocked: fixture setup is trusted code
    con.execute(
        f"""
        COPY (
          SELECT (i %% 2) + 1                                   AS DIBEV_A,
                 CASE WHEN i %% 2 = 0 THEN NULL ELSE 1 END      AS DIBINS_A,
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

    @pytest.mark.parametrize("name,sql", sorted(CONTAINED.items()))
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
        runaway = f"SELECT count(*) FROM {RELATION} a, {RELATION} b, {RELATION} c"
        with pytest.raises(QueryRejected, match="execution bound"):
            execute_contained(con, runaway, timeout=2.0)

    def test_the_bound_covers_the_fetch_not_just_the_execute(self, con):
        """Streaming makes execute return instantly, so the clock must survive the fetch.

        The interrupt is checked at chunk boundaries: a best-effort bound, not a hard deadline.
        """
        wide = f"SELECT a.SEX_A FROM {RELATION} a, {RELATION} b"
        with pytest.raises(QueryRejected):
            execute_contained(con, wide, timeout=1.0, max_rows=400_000_000)

    def test_the_data_is_still_reachable(self, con):
        columns, rows = execute_contained(con, f"SELECT count(*) AS n FROM {RELATION}")
        assert rows[0][0] > 0 and columns == ["n"]


# ---------------------------------------------------------------------------------------
# The published invariant: aggregate-only, no individual records
# ---------------------------------------------------------------------------------------

class TestRecordDisclosure:
    """Assert the record BAN, parameterised over routes, not over exploit strings."""

    @pytest.mark.parametrize("name,sql", sorted(DISCLOSURE.items()))
    def test_no_route_to_a_protected_column(self, con, name, sql):
        with pytest.raises(QueryRejected):
            run_query(con, sql)

    def test_protected_columns_are_fine_inside_an_aggregate_value(self, con):
        result = run_query(con, f"SELECT SUM(WTFA_A) AS w, count(*) AS n FROM {RELATION}")
        assert len(result.rows) == 1

    def test_categorical_grouping_is_allowed(self, con):
        result = run_query(
            con,
            f"SELECT SEX_A, SUM(WTFA_A) AS w, count(*) AS n FROM {RELATION} "
            "WHERE DIBEV_A IN (1, 2) AND SEX_A IN (1, 2) GROUP BY SEX_A",
        )
        assert 0 < len(result.rows) <= MAX_ROWS


# ---------------------------------------------------------------------------------------
# Cell size — the invariant low cardinality does NOT provide
# ---------------------------------------------------------------------------------------

class TestCellSize:

    def test_count_star_is_mandatory(self, con):
        with pytest.raises(QueryRejected, match="count\\(\\*\\)"):
            run_query(con, f"SELECT SUM(WTFA_A) AS w FROM {RELATION}")

    @pytest.mark.skipif(not REAL_SLICE.is_file(), reason="real verified slice not present")
    def test_the_categorical_cross_product_is_refused_for_small_cells(self, con):
        """The route that needed no trick: 4 low-cardinality columns, cells of one person."""
        sql = (
            "SELECT DIBEV_A, DIBINS_A, PREDIB_A, SEX_A, MIN(SEX_A) AS s, count(*) AS n "
            f"FROM {RELATION} GROUP BY DIBEV_A, DIBINS_A, PREDIB_A, SEX_A"
        )
        with pytest.raises(QueryRejected, match="disclosure control"):
            run_query(con, sql)

    def test_a_large_cell_passes(self, con):
        result = run_query(con, f"SELECT count(*) AS n FROM {RELATION}")
        assert result.rows[0][0] >= MIN_CELL_SIZE


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
            validate_sql(con, f"SELECT count(SSN) AS c, count(*) AS n FROM {RELATION}")

    def test_unknown_relation_is_refused(self, con):
        with pytest.raises(QueryRejected):
            validate_sql(con, "SELECT count(*) AS n FROM sqlite_master")

    def test_empty_is_refused(self, con):
        with pytest.raises(QueryRejected):
            validate_sql(con, "   ")

    @pytest.mark.parametrize(
        "sql",
        [
            f"SELECT count({RELATION}.WTFA_A) AS c, count(*) AS n FROM {RELATION}",
            f"SELECT count(u.WTFA_A) AS c, count(*) AS n FROM {RELATION} AS u",
            f"SELECT SEX_A AS s, count(*) AS n FROM {RELATION} GROUP BY s",
            f"SELECT SEX_A, count(*) AS n FROM {RELATION} GROUP BY SEX_A ORDER BY n",
        ],
    )
    def test_legitimate_sql_is_accepted(self, con, sql):
        validate_sql(con, sql)


# ---------------------------------------------------------------------------------------
# Layer 3 — the weight label
# ---------------------------------------------------------------------------------------

class TestWeightLabel:

    @pytest.mark.parametrize("name,sql", sorted(WEIGHT_SPOOFS.items()))
    def test_the_weight_label_cannot_be_spoofed(self, con, name, sql):
        """Only genuine summation of the weight counts. count/min/max of it are not estimation."""
        result = run_query(con, sql)
        assert result.weighted is False, f"{name!r} spoofed the weight label"
        assert UNWEIGHTED_BANNER in result.render()

    def test_genuinely_weighted_result_is_not_labeled(self, con):
        result = run_query(
            con,
            f"SELECT SUM(WTFA_A) AS w, count(*) AS n FROM {RELATION} WHERE DIBEV_A = 1",
        )
        assert result.weighted is True
        assert UNWEIGHTED_BANNER not in result.render()


# ---------------------------------------------------------------------------------------
# Fail closed and concurrency
# ---------------------------------------------------------------------------------------

class TestFailClosed:

    def test_engine_errors_become_QueryRejected(self, con):
        sql = f"SELECT count(*) AS n FROM {RELATION} WHERE SEX_A = 'abc'"
        with pytest.raises(QueryRejected) as excinfo:
            run_query(con, sql)
        message = str(excinfo.value)
        for leak in ("Conversion Error", "LINE 1", "duckdb", "^"):
            assert leak not in message, f"engine text leaked: {message}"

    def test_parameter_marker_does_not_escape_as_a_raw_error(self, con):
        with pytest.raises(QueryRejected):
            run_query(con, f"SELECT count(*) AS n FROM {RELATION} WHERE SEX_A = ?")


class TestConcurrency:

    def test_results_do_not_cross_between_threads(self, con):
        errors: list[str] = []

        def worker(sex: int, label: str) -> None:
            for _ in range(25):
                try:
                    result = run_query(
                        con, f"SELECT count(*) AS {label} FROM {RELATION} WHERE SEX_A = {sex}"
                    )
                    if result.columns != [label]:
                        errors.append(f"columns crossed: {result.columns} != [{label}]")
                except Exception as exc:
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
            "SELECT SUM(WTFA_A) FILTER (WHERE DIBINS_A = 1) / SUM(WTFA_A) * 100 AS pct, "
            f"count(*) AS n FROM {RELATION} WHERE DIBEV_A = 1 AND DIBINS_A IN (1, 2)"
        )
        result = run_query(con, sql)
        assert result.weighted is True
        assert round(result.rows[0][0], 2) == 31.96
        assert result.rows[0][1] >= MIN_CELL_SIZE
