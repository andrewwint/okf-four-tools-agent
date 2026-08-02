---
type: capability
id: query_adult23
title: "Survey-weighted queries over the NHIS 2023 adult slice"
kind: sql
relation: t
source: "NHIS 2023 Sample Adult public-use file (adult23.csv)"
weight: WTFA_A
min_cell_size: 30

# WHAT MAY BE ASKED. The agent picks keys from these lists; it never writes SQL. Everything
# here was established by the execution-grounded verifier, not by reading the codebook.
measures:
  DIBEV_A:
    label: "Ever told had diabetes"
    kind: prevalence
    valid: [1, 2]           # 1 yes, 2 no. 7/8/9 are refused / not ascertained / don't know
    universes: [all_adults]
  DIBINS_A:
    label: "Currently takes insulin"
    kind: prevalence
    valid: [1, 2]
    # A skip-pattern item is only asked of some people, so its VALID universes are part of its
    # meaning. Without this list the tool will happily compute the figure over "all adults",
    # label it that way, and be wrong by ~4.6x — the project's headline defect, reappearing at
    # the labelling layer after being closed at the computation layer.
    universes: [diagnosed_diabetes, diabetes_or_prediabetes]
  PREDIB_A:
    label: "Ever told had prediabetes"
    kind: prevalence
    valid: [1, 2]
    universes: [all_adults]
  DIBAGETC_A:
    label: "Age first told had diabetes (years)"
    kind: mean
    drop: [96, 97, 98, 99]  # NOT 7/8/9 — those are real ages for this variable
    universes: [diagnosed_diabetes]
    top_code: 85            # 85 means "85 or older"; it enters the mean as a literal 85

universes:
  all_adults:
    label: "All U.S. adults"
    predicate: "TRUE"
  diagnosed_diabetes:
    label: "Adults ever told they had diabetes"
    predicate: "DIBEV_A = 1"
  prediabetes:
    label: "Adults ever told they had prediabetes"
    predicate: "PREDIB_A = 1"
  diabetes_or_prediabetes:
    label: "Adults with diabetes or prediabetes"
    predicate: "(DIBEV_A = 1 OR PREDIB_A = 1)"

groupings:
  SEX_A:
    label: "Sex"
    valid: [1, 2]
    # Rendered instead of the raw code. A code is an identifier, not a figure, so it is never
    # grounded — and an answer that echoed "SEX_A 1" was withheld in full. Show words.
    value_labels: { 1: "Male", 2: "Female" }

# THE GATE. Each example is executed at compile time and must match. A capability whose
# examples stop matching is quarantined — the agent loses the tool rather than gaining a
# broken one. This is the same rule that governs every verified figure in the bundle.
examples:
  - measure: DIBINS_A
    universe: diagnosed_diabetes
    expect: 31.96
    links: DIBINS_A
  - measure: DIBEV_A
    universe: all_adults
    expect: 9.80
    links: DIBEV_A
  - measure: DIBAGETC_A
    universe: diagnosed_diabetes
    expect: 47.41
    links: DIBAGETC_A

verification:
  verdict: PASS
  method: execution
  detail: "each example executed against the shipped slice and matched its verified concept"
---

# Survey-weighted queries over the NHIS 2023 adult slice

This capability lets the agent compute a figure that is **not written down anywhere** — the
survey-weighted statistics in this file are calculated from 29,522 respondent records, so no
amount of document retrieval can produce them.

It answers two shapes of question:

- **prevalence** — the weighted share of a population answering *yes* to an item
- **mean** — the weighted average of a continuous measure

Ask by naming a `measure` and a `universe`, optionally broken down by a `group_by`. The agent
never writes SQL; it chooses from the lists above and this capability builds the query. That is
a safety decision (see [[query-surface]]) and a correctness one: the weight is applied and the
non-substantive codes are dropped every time, because they are part of the declaration rather
than something the caller must remember.

## Why the universe is a named list, not a free filter

The insulin item is the worked example. It is only asked of adults with diabetes or prediabetes,
so the denominator decides the answer:

- among adults with **diagnosed diabetes** — the clinically meaningful figure — it is **31.96%**
- computed over the **whole sample**, it reads 3.66%, because everyone never asked is silently
  counted as a non-user

Both are "the percentage taking insulin". One is wrong. Naming universes rather than accepting
arbitrary filters means the wrong denominator is not something the agent can pick by accident.

## Related

- [DIBINS_A](./DIBINS_A.md) — the verified insulin figure this capability reproduces
- [DIBEV_A](./DIBEV_A.md) — diagnosed diabetes prevalence
- [DIBAGETC_A](./DIBAGETC_A.md) — mean age at diagnosis
