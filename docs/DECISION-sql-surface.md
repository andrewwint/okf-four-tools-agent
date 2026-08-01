# Decision record: can an LLM write SQL over person-level microdata?

**Status:** awaiting the human's call. Three rounds of independent adversarial review say no —
not with AST validation, at any level of care.

## What was attempted

A three-layer guard around LLM-generated DuckDB SQL over the verified NHIS slice (29,522
respondents, 8 columns). Each round was reviewed by an independent lane briefed with the seam's
invariants but none of the implementer's hypotheses, which re-attacked the code itself rather
than reading the tests.

| Round | What the guard asserted | How it was broken |
| --- | --- | --- |
| 1 | An aggregate appears somewhere in the AST | An aggregate in a `WHERE` subquery satisfied it while the projection stayed bare. **200 complete respondent records in 4 calls**, including PSTRAT, PPSU and the person-level weight. |
| 2 | A bare column must be low-cardinality | `FILTER (WHERE WTFA_A = …)` and `CASE WHEN WTFA_A = …` are row selectors that sit *syntactically inside* an aggregate. Also `WTFA_A AS WTFA_A` walked through the alias exemption. And with no trick at all: the 4 categorical columns have a 35-cell cross product, **8 cells holding exactly one respondent**, because the categories include rare refusal/don't-know sentinels. |
| 3 | Value context vs predicate context, plus a minimum cell size of 30 | **Three routes, one of them unpatchable.** |

## Round 3, the decisive result

**Route 3 — the arithmetic kernel.** A steep rational function is an indicator function:

```sql
SELECT count(*) AS n,
       sum(DIBAGETC_A / (1 + (WTFA_A-3146.794)*(WTFA_A-3146.794)*1000000000000)) AS dx,
       sum(PSTRAT     / (1 + (WTFA_A-3146.794)*(WTFA_A-3146.794)*1000000000000)) AS st
FROM t
→ ACCEPTED. n=29522 (floor passes trivially). dx ≈ 61.00000000005, st ≈ 122.00000001 — exact.
```

There is no `COMPARE`, no `CONJUNCTION`, no `CASE_EXPR`, no `FILTER`. Every column sits in pure
value context under `sum`, built only from `+ - * /`. The classifier is correct by its own
definition; the definition is what fails. **Any expression language rich enough to compute a rate
is rich enough to select a row**, so this cannot be fixed by extending the predicate node list.

Seeding is free: `min(WTFA_A)` is a legitimate published statistic and gives a starting weight,
and `count(DISTINCT WTFA_A)` = 29,318 of 29,522 confirms the weight is a near-unique key.

**Route 2 — differencing.** Two queries, each clearing the cell floor by 50×:

```
Q1 universe: (DIBEV_A=9 AND PREDIB_A=9 AND SEX_A=2) OR (DIBEV_A=1 AND SEX_A=1)  → n=1581
Q2 universe:  DIBEV_A=1 AND SEX_A=1                                             → n=1580
Q1 − Q2 → n=1, WTFA_A=8141.496, PSTRAT=111, PPSU=13   (exact)
```

A per-query floor cannot see a difference of universes. Statistical agencies pair cell floors
with query-set-overlap auditing or fixed universes precisely because of this.

**Route 1 — the floor read the wrong column.** `_count_star_index` matched any entry whose
subtree *contained* `count_star`, so `count(*) * 1000` — or, with no adversarial intent at all,
the ordinary analyst query `sum(WTFA_A)/count(*)` — pointed the floor at an inflated value and
disabled it. A real bug, and fixable; listed here because it shows the control is fragile even
where it is sound in principle.

## What did hold, across all three rounds

- **Layer 1 containment was never broken.** Locked DuckDB (no filesystem, no network, no
  extensions, `lock_configuration`), the slice bound by trusted code as relation `t`, and a
  wall-clock interrupt. Verified repeatedly with the parser stubbed out.
- The **value/predicate rule** killed every relocation attack in round 3 (FILTER, CASE, HAVING,
  alias). It is correct — it is simply not sufficient.
- Cursor isolation, fail-closed error handling, and the execution bound all hold.

## The recommendation

Do not ship free-form SQL over person-level microdata. Both the reviewer and the implementer
converged independently on the same exit:

**Let the OKF capability concept declare whole query *shapes*** — parameterised templates where
the model fills typed slots (universe from a declared set, grouping keys from a declared set,
measure from a declared set) and never emits raw SQL. This removes Route 3 entirely (no free
arithmetic), removes Route 1 (the template emits its own count), and bounds Route 2 (universes
are enumerable, so overlap is auditable or simply pre-cleared). It also *strengthens* the
article's thesis rather than weakening it: each shape carries a verified example, so the concept
still declares what is permitted and execution still proves it.

Worth doing regardless of the decision: recode the sentinel responses (7/8/9) at trusted load
time — all 8 singleton cells are sentinel-coded, so the easy differencing atoms disappear. It is
a mitigation, not a control.

## The honest reading

The guard got materially better each round and the reviewer confirmed every prior fix. That is
not the point. The point is that three rounds of competent hardening could not make the surface
safe, because the surface is the wrong shape — and no amount of care inside a wrong boundary
fixes a wrong boundary. The finding is worth more than the code: **an LLM writing SQL against
person-level data is not a validation problem, it is an architecture problem.**
