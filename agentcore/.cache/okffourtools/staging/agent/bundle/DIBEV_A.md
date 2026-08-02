---
type: variable_definition
title: "Ever told you had diabetes"
description: "Weighted prevalence of diagnosed diabetes among U.S. adults, 2023"
resource: "https://www.cdc.gov/nchs/nhis/2023nhis.htm"
tags: [nhis-2023, diabetes, DIBEV_A, prevalence]
timestamp: "2026-07-10T00:16:06Z"
# extension keys (OKF consumers tolerate unknown fields)
id: DIBEV_A
variable: DIBEV_A
question_universe: "All sample adults."
weight: WTFA_A
source: "NHIS 2023 Sample Adult public-use file (adult23.csv)"
statistic: "Weighted prevalence of diagnosed diabetes among U.S. adults, 2023"
value_pct: 9.8
verification:
  verdict: PASS
  method: execution-grounded
  correct_pct: 9.8
  claimed_pct: 9.8
  delta_pp: 0.0
  detail: "9.80% (95% CI 9.39-10.20; design-based SE 0.21pp, DEFF 1.41)"
  ci_95: [9.39, 10.20]
  se_pp: 0.21
  deff: 1.41
  variance_method: taylor-linearization (design-based)
  verified_at: 2026-07-10T00:16:06Z
---

# Ever told you had diabetes

DIBEV_A records whether a sample adult was ever told by a doctor or other health
professional that they had diabetes. "Diagnosed diabetes" is DIBEV_A == 1. It excludes
borderline/prediabetes (see [PREDIB_A](./PREDIB_A.md)) and gestational-only diabetes (GESDIB_A).

The population estimate must be survey-weighted (WTFA_A); the unweighted sample share
does not estimate the U.S. adult population.

## Verified statistic

**Weighted prevalence of diagnosed diabetes among U.S. adults, 2023: 9.8%**

- 95% CI: [9.39, 10.20] (design-based, Taylor linearization; SE 0.21pp; DEFF 1.41)
- Basis: 9.80% (95% CI 9.39-10.20; design-based SE 0.21pp, DEFF 1.41)
- Verification: executed against NHIS 2023 Sample Adult public-use file (adult23.csv); verdict **PASS**.

## Reproduce

Weighted, verified figure (aggregate only — the number to cite):

```bash
nhis analyze --variable DIBEV_A
```

Raw row inspection (unweighted, not verified — for sanity-checking only; see the [tool reference](../references/parquet_query.md)):

```bash
nhis rows --columns "DIBEV_A" --limit 10
```

## Related
- [DIBINS_A](./DIBINS_A.md)
- [DIBAGETC_A](./DIBAGETC_A.md)
- [PREDIB_A](./PREDIB_A.md)
