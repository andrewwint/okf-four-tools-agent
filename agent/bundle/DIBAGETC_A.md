---
type: variable_definition
title: "Age first told had diabetes (top-coded)"
description: "Weighted mean age first told had diabetes among U.S. adults with diagnosed diabetes, 2023"
resource: "https://www.cdc.gov/nchs/nhis/2023nhis.htm"
tags: [nhis-2023, diabetes, DIBAGETC_A, mean]
timestamp: "2026-07-10T00:16:06Z"
# extension keys (OKF consumers tolerate unknown fields)
id: DIBAGETC_A
variable: DIBAGETC_A
question_universe: "Adults ever told they had diabetes (DIBEV_A == 1)."
analytical_universe: "DIBEV_A == 1"
weight: WTFA_A
source: "NHIS 2023 Sample Adult public-use file (adult23.csv)"
statistic: "Weighted mean age first told had diabetes among U.S. adults with diagnosed diabetes, 2023"
kind: mean
value: 47.41
unit: "years"
verification:
  verdict: PASS
  method: execution-grounded
  correct_pct: 47.41
  claimed_pct: 47.41
  delta_pp: 0.0
  detail: "47.41 (95% CI 46.75-48.08; design-based SE 0.34)"
  verified_at: 2026-07-10T00:16:06Z
---

# Age first told had diabetes (top-coded)

DIBAGETC_A is the age at which an adult was first told they had diabetes, asked only of
adults with diagnosed diabetes ([DIBEV_A](./DIBEV_A.md) == 1) and top-coded at 85. Values of 96 and
above are non-substantive (refused / not ascertained / don't know) and must be dropped
before any age analysis. Any mean or distribution over this variable must be
survey-weighted (WTFA_A) and restricted to the DIBEV_A == 1 universe.

The headline claim is the survey-weighted mean age at diagnosis among adults with
diagnosed diabetes. Dropping the 96-99 codes and applying the survey weights are both
required: skipping either shifts the number in a way execution catches.

## Verified statistic

**Weighted mean age first told had diabetes among U.S. adults with diagnosed diabetes, 2023: 47.41 years**

- Basis: 47.41 (95% CI 46.75-48.08; design-based SE 0.34)
- Verification: executed against NHIS 2023 Sample Adult public-use file (adult23.csv); verdict **PASS**.

## Reproduce

Weighted, verified figure (aggregate only — the number to cite):

```bash
nhis analyze --variable DIBAGETC_A --universe "DIBEV_A == 1" --stat mean
```

Raw row inspection (unweighted, not verified — for sanity-checking only; see the [tool reference](../references/parquet_query.md)):

```bash
nhis rows --columns "DIBEV_A,DIBAGETC_A" --universe "DIBEV_A == 1" --limit 10
```

## Related
- [DIBEV_A](./DIBEV_A.md)
