"""Tests for the declared-shape query tool.

Note what is NOT here, and why: there is no hostile-SQL suite. The previous design needed 108
tests to police model-authored SQL, and three rounds of independent review still found routes to
an individual record. This design has no SQL parameter at all — every argument is an enum key —
so the entire attack class is *inexpressible* rather than *defended against*. The tests below
prove the enums are closed and the figures are right; `test_containment.py` still proves Layer 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.query_guard import MIN_CELL_SIZE, QueryRejected, build_guarded_connection  # noqa: E402
from agent.query_shapes import (  # noqa: E402
    GROUPINGS,
    MEASURES,
    SHAPES,
    UNIVERSES,
    catalogue,
    run_shape,
)

REAL_SLICE = Path(
    "/Users/andrewwint/Documents/PROJECTS/PERSONAL/nhisokfchat/app/nhisokfchat/"
    "nhis_okf/microdata/adult23_slice.parquet"
)


@pytest.fixture(scope="module")
def slice_path(tmp_path_factory) -> Path:
    if REAL_SLICE.is_file():
        return REAL_SLICE
    out = tmp_path_factory.mktemp("data") / "slice.parquet"
    con = duckdb.connect(":memory:")
    con.execute(
        f"""
        COPY (
          SELECT (i %% 2) + 1                            AS DIBEV_A,
                 (i %% 2) + 1                            AS DIBINS_A,
                 (i %% 2) + 1                            AS PREDIB_A,
                 CAST(40 + (i %% 40) AS DOUBLE)          AS DIBAGETC_A,
                 (i %% 2) + 1                            AS SEX_A,
                 1000.0 + i                              AS WTFA_A,
                 100 + (i %% 10)                         AS PSTRAT,
                 (i %% 4) + 1                            AS PPSU
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


class TestTheAttackSurfaceIsGone:
    """SQL cannot be supplied, so it cannot be validated wrongly."""

    def test_there_is_no_sql_parameter(self):
        import inspect

        params = set(inspect.signature(run_shape).parameters) - {"con"}
        assert params == {"shape", "measure", "universe", "group_by"}
        assert not any("sql" in p for p in params)

    @pytest.mark.parametrize(
        "hostile",
        [
            "all_adults; DROP TABLE t",
            "all_adults' OR '1'='1",
            "read_text('/etc/hostname')",
            "diagnosed_diabetes UNION SELECT 1",
            "DIBEV_A = 1 OR WTFA_A = 3146.794",
            "sum(DIBAGETC_A / (1 + (WTFA_A-3146.794)*(WTFA_A-3146.794)*1e12))",
        ],
    )
    def test_hostile_slot_values_are_unknown_keys(self, con, hostile):
        """The kernel that defeated the previous design is now just an unrecognised enum key."""
        with pytest.raises(QueryRejected, match="unknown universe"):
            run_shape(con, shape="weighted_prevalence", measure="DIBINS_A", universe=hostile)

    @pytest.mark.parametrize("slot", ["shape", "measure", "universe"])
    def test_unknown_keys_are_refused(self, con, slot):
        args = {"shape": "weighted_prevalence", "measure": "DIBINS_A", "universe": "all_adults"}
        args[slot] = "nope"
        with pytest.raises(QueryRejected, match=f"unknown {slot}"):
            run_shape(con, **args)

    @pytest.mark.parametrize("slot", ["shape", "measure", "universe"])
    def test_missing_keys_are_refused(self, con, slot):
        args = {"shape": "weighted_prevalence", "measure": "DIBINS_A", "universe": "all_adults"}
        args[slot] = None
        with pytest.raises(QueryRejected, match=f"{slot} is required"):
            run_shape(con, **args)

    def test_unknown_group_by_is_refused(self, con):
        with pytest.raises(QueryRejected, match="unknown group_by"):
            run_shape(
                con, shape="weighted_prevalence", measure="DIBINS_A",
                universe="all_adults", group_by="PSTRAT",
            )

    def test_a_protected_column_cannot_be_a_grouping_key(self):
        """The columns that identify a person are simply not in the declared sets."""
        for column in ("WTFA_A", "PSTRAT", "PPSU", "DIBAGETC_A"):
            assert column not in GROUPINGS

    def test_measure_kind_must_match_the_shape(self, con):
        with pytest.raises(QueryRejected, match="mean measure"):
            run_shape(
                con, shape="weighted_prevalence", measure="DIBAGETC_A",
                universe="diagnosed_diabetes",
            )


class TestGeneratedSql:
    """Trusted code owns every string, and it always weights and always counts."""

    def test_every_shape_weights_and_counts(self, con):
        for measure_id, measure in MEASURES.items():
            shape_id = "weighted_mean" if measure.kind == "mean" else "weighted_prevalence"
            result = run_shape(
                con, shape=shape_id, measure=measure_id, universe="all_adults"
            )
            assert "WTFA_A" in result.sql, "a shape must aggregate the survey weight"
            assert "count(*) AS n" in result.sql, "a shape must report its own cell size"
            assert result.columns[-1] == "n"

    def test_the_declared_universe_reaches_the_where_clause(self, con):
        result = run_shape(
            con, shape="weighted_prevalence", measure="DIBINS_A",
            universe="diagnosed_diabetes",
        )
        assert UNIVERSES["diagnosed_diabetes"].predicate in result.sql


class TestDisclosureControl:

    def test_cell_floor_is_enforced(self, con, slice_path):
        """A cell below the floor refuses the whole result rather than suppressing a row."""
        from agent import query_shapes

        original = query_shapes.MIN_CELL_SIZE
        try:
            query_shapes.MIN_CELL_SIZE = 10_000_000  # force every cell under the floor
            with pytest.raises(QueryRejected, match="disclosure control"):
                run_shape(
                    con, shape="weighted_prevalence", measure="DIBINS_A",
                    universe="diagnosed_diabetes",
                )
        finally:
            query_shapes.MIN_CELL_SIZE = original

    def test_a_normal_query_clears_the_floor(self, con):
        result = run_shape(
            con, shape="weighted_prevalence", measure="DIBEV_A", universe="all_adults"
        )
        assert result.rows[0][-1] >= MIN_CELL_SIZE


class TestCatalogue:

    def test_catalogue_lists_every_declared_key(self):
        text = catalogue()
        for keys in (SHAPES, MEASURES, UNIVERSES, GROUPINGS):
            for key in keys:
                assert key in text


class TestVerifiedFigures:
    """The shapes must reproduce the execution-verified concepts exactly."""

    @pytest.mark.skipif(not REAL_SLICE.is_file(), reason="real verified slice not present")
    @pytest.mark.parametrize(
        "shape,measure,universe,expected",
        [
            ("weighted_prevalence", "DIBINS_A", "diagnosed_diabetes", 31.96),
            ("weighted_prevalence", "DIBEV_A", "all_adults", 9.80),
            ("weighted_mean", "DIBAGETC_A", "diagnosed_diabetes", 47.41),
        ],
    )
    def test_matches_the_verified_concept(self, con, shape, measure, universe, expected):
        result = run_shape(con, shape=shape, measure=measure, universe=universe)
        assert round(result.rows[0][0], 2) == expected
        assert result.rows[0][-1] >= MIN_CELL_SIZE

    @pytest.mark.skipif(not REAL_SLICE.is_file(), reason="real verified slice not present")
    def test_the_naive_whole_sample_figure_is_unreachable(self, con):
        """3.66% is the wrong-universe answer the compiler quarantined; no shape can produce it.

        The insulin measure is only offered against universes where it is the right denominator,
        so the classic skip-pattern error cannot be expressed as a request.
        """
        for universe in UNIVERSES:
            result = run_shape(
                con, shape="weighted_prevalence", measure="DIBINS_A", universe=universe
            )
            assert round(result.rows[0][0], 2) != 3.66

    @pytest.mark.skipif(not REAL_SLICE.is_file(), reason="real verified slice not present")
    def test_grouping_returns_plausible_subgroup_figures(self, con):
        result = run_shape(
            con, shape="weighted_prevalence", measure="DIBINS_A",
            universe="diagnosed_diabetes", group_by="SEX_A",
        )
        assert len(result.rows) == 2
        for row in result.rows:
            assert 20 < row[1] < 45     # both sexes near the 31.96 headline
            assert row[-1] >= MIN_CELL_SIZE
