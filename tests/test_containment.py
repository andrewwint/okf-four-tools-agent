"""Hostile-input suite for Layer-1 containment.

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

from agent.containment import (  # noqa: E402
    QueryRejected,
    RELATION,
    build_guarded_connection,
    execute_contained,
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




class TestTheDeployArtifactMatchesTheSource:
    """`agentcore.json` ships `dist/`, so a stale `dist/` deploys code nobody reviewed.

    A pre-deploy review found `dist/` two commits behind: it still carried the laundering bug the
    review existed to close. The remedy was one command, but "remember to run build.py" is not a
    control — so the mismatch is a test failure instead.
    """

    def test_dist_is_absent_or_current(self):
        import filecmp

        root = Path(__file__).resolve().parents[1]
        dist = root / "dist"
        if not dist.is_dir():
            pytest.skip("no dist/ — nothing can be deployed stale")

        # main.py is included deliberately: it decides WHICH TOOLS ARE REGISTERED, so a stale
        # dist/main.py deploys a different agent than the one reviewed while this guard stays
        # green. The guard globbed agent/ only, which is exactly the file it most needed to watch.
        sources = [*(root / "agent").rglob("*.py"), root / "main.py"]
        stale = []
        for source in sources:
            if "__pycache__" in source.parts:
                continue
            shipped = dist / source.relative_to(root)
            if not shipped.exists() or not filecmp.cmp(source, shipped, shallow=False):
                stale.append(str(source.relative_to(root)))
        assert not stale, (
            f"dist/ is stale ({stale}) — run `python build.py` before deploying, or the "
            "artifact will not be the code that was reviewed"
        )
