---
type: variable_definition
title: "Currently takes insulin (among adults with diagnosed diabetes)"
description: "Weighted % currently taking insulin among U.S. adults with diagnosed diabetes, 2023"
resource: "https://www.cdc.gov/nchs/nhis/2023nhis.htm"
tags: [nhis-2023, diabetes, DIBINS_A, prevalence]
timestamp: "2026-07-10T00:16:06Z"
# extension keys (OKF consumers tolerate unknown fields)
id: DIBINS_A
variable: DIBINS_A
question_universe: "Adults ever told they had diabetes (DIBEV_A == 1) or prediabetes (PREDIB_A == 1). The clinically meaningful 'among diagnosed diabetics' denominator is the narrower DIBEV_A == 1."
analytical_universe: "DIBEV_A == 1"
weight: WTFA_A
source: "NHIS 2023 Sample Adult public-use file (adult23.csv)"
statistic: "Weighted % currently taking insulin among U.S. adults with diagnosed diabetes, 2023"
value_pct: 31.96
verification:
  verdict: PASS
  method: execution-grounded
  correct_pct: 31.96
  claimed_pct: 31.96
  delta_pp: 0.0
  detail: "31.96% (95% CI 30.08-33.84; design-based SE 0.96pp, DEFF 1.39)"
  ci_95: [30.08, 33.84]
  se_pp: 0.96
  deff: 1.39
  variance_method: taylor-linearization (design-based)
  verified_at: 2026-07-10T00:16:06Z
---

# Currently takes insulin (among adults with diagnosed diabetes)

DIBINS_A records whether an adult currently takes insulin. The survey only asks it of
adults who report ever having diabetes ([DIBEV_A](./DIBEV_A.md) == 1) or prediabetes ([PREDIB_A](./PREDIB_A.md)
== 1) — it is a skip-pattern item, not asked of everyone.

The headline claim is insulin use **among people with diagnosed diabetes**, so the
denominator is DIBEV_A == 1, survey-weighted. Two denominators are wrong here: the whole
sample (most adults were never asked) and the full question universe (which dilutes the
rate with prediabetics who rarely use insulin).

## Verified statistic

**Weighted % currently taking insulin among U.S. adults with diagnosed diabetes, 2023: 31.96%**

- 95% CI: [30.08, 33.84] (design-based, Taylor linearization; SE 0.96pp; DEFF 1.39)
- Basis: 31.96% (95% CI 30.08-33.84; design-based SE 0.96pp, DEFF 1.39)
- Verification: executed against NHIS 2023 Sample Adult public-use file (adult23.csv); verdict **PASS**.

## Reproduce

Weighted, verified figure (aggregate only — the number to cite):

```bash
nhis analyze --variable DIBINS_A --universe "DIBEV_A == 1"
```

Raw row inspection (unweighted, not verified — for sanity-checking only; see the [tool reference](../references/parquet_query.md)):

```bash
nhis rows --columns "DIBEV_A,DIBINS_A" --universe "DIBEV_A == 1" --limit 10
```

## Related
- [DIBEV_A](./DIBEV_A.md)
- [DIBPILL_A](./DIBPILL_A.md)
