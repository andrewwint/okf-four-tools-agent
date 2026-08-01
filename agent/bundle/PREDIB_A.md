---
type: variable_definition
title: "Ever told you had prediabetes"
description: "PREDIB_A records whether a sample adult was ever told they had prediabetes or borderline diabetes."
resource: "https://www.cdc.gov/nchs/nhis/2023nhis.htm"
tags: [nhis-2023, diabetes, PREDIB_A]
timestamp: "2026-07-10T00:16:06Z"
# extension keys (OKF consumers tolerate unknown fields)
id: PREDIB_A
variable: PREDIB_A
question_universe: "All sample adults."
weight: WTFA_A
source: "NHIS 2023 Sample Adult public-use file (adult23.csv)"
verification:
  verdict: DESCRIPTIVE
  method: execution-grounded
  verified_at: 2026-07-10T00:16:06Z
---

# Ever told you had prediabetes

PREDIB_A records whether a sample adult was ever told they had prediabetes or borderline
diabetes. It is a separate item from diagnosed diabetes ([DIBEV_A](./DIBEV_A.md)): prediabetics are
not counted as having diagnosed diabetes. It matters for the insulin universe — the
insulin item ([DIBINS_A](./DIBINS_A.md)) is asked of adults with diabetes **or** prediabetes, so the
clinically meaningful "among diagnosed diabetics" denominator is the narrower DIBEV_A == 1.

## Related
- [DIBEV_A](./DIBEV_A.md)
