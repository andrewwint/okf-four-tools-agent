#!/usr/bin/env python3
"""Package what the agent ships: the verified OKF bundle and the slim data slice.

Run this before deploying, and re-run it whenever the compiler produces a new bundle:

    python build.py

Why this file exists at all: the previous deploy shipped its bundle and its parquet as
**committed binaries with no generator** — the script that made them had been deleted, so
nobody could reproduce or re-derive them. A verified artifact you cannot rebuild is not really
verified; it is just a file you trust. This step is small, boring, and reproducible on purpose.

Note what it does NOT do: it never decides *what* the agent may query. That comes from
`concepts/query_adult23.md`, and this script reads the concept to work out which columns the
slice needs. The declaration stays in one place.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

import duckdb
import yaml

HERE = Path(__file__).resolve().parent

# Where the OKF compiler keeps its verified output. Override if your checkout lives elsewhere.
COMPILER = Path(
    os.environ.get(
        "OKF_COMPILER",
        HERE.parent / "nhis-okf-compiler",
    )
)

DIST = HERE / "dist"
BUNDLE_OUT = HERE / "agent" / "bundle"
DATA_OUT = HERE / "agent" / "data" / "adult23_slice.parquet"

# Survey design columns. Not queryable today, but they are what a design-based confidence
# interval needs, and re-cutting the slice later is more painful than carrying two columns now.
DESIGN_COLUMNS = ["PSTRAT", "PPSU"]


def read_concept() -> dict:
    text = (HERE / "concepts" / "query_adult23.md").read_text()
    return yaml.safe_load(re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL).group(1))


def required_columns(concept: dict) -> list[str]:
    """Exactly the columns the concept's declarations reference — nothing else ships."""
    columns = list(concept["measures"]) + list(concept["groupings"]) + [concept["weight"]]
    # Universe predicates name columns too (e.g. "DIBEV_A = 1").
    for universe in concept["universes"].values():
        columns += re.findall(r"\b([A-Z][A-Z0-9_]*_A)\b", universe["predicate"])
    return sorted(set(columns + DESIGN_COLUMNS))


def copy_bundle(concept: dict) -> list[str]:
    """Copy the verified concept files the capability links to."""
    source = COMPILER / ".okf" / "variables"
    if not source.is_dir():
        sys.exit(f"no verified bundle at {source} — run `nhis compile` in the compiler first")

    BUNDLE_OUT.mkdir(parents=True, exist_ok=True)
    for stale in BUNDLE_OUT.glob("*.md"):
        stale.unlink()  # start clean: a quarantined concept must not linger from a past build

    copied = []
    for measure in concept["measures"]:
        found = source / f"{measure}.md"
        if not found.is_file():
            sys.exit(f"{measure} is declared in the capability but not in the verified bundle "
                     f"— it may have been quarantined. Refusing to ship a broken tool.")
        shutil.copy2(found, BUNDLE_OUT / found.name)
        copied.append(found.name)

    # The capability concept ships alongside the facts it describes.
    shutil.copy2(HERE / "concepts" / "query_adult23.md", BUNDLE_OUT / "query_adult23.md")
    return copied


def build_slice(columns: list[str]) -> None:
    """Project the full microdata down to the columns the concept actually needs."""
    for candidate in (COMPILER / "data" / "adult23.parquet", COMPILER / "data" / "adult23.csv"):
        if candidate.is_file():
            source = candidate
            break
    else:
        sys.exit(f"no source microdata under {COMPILER / 'data'} — run `nhis fetch` first")

    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    reader = "read_parquet" if source.suffix == ".parquet" else "read_csv_auto"
    duckdb.sql(
        f"COPY (SELECT {', '.join(columns)} FROM {reader}('{source}')) "
        f"TO '{DATA_OUT}' (FORMAT parquet, COMPRESSION zstd)"
    )


# Exactly what the runtime needs. An allow-list, not an ignore-list: `codeLocation: "."` would
# make the deployment package the repo root, and the repo root is where `.env` lives. A security
# review flagged the live API key sitting inside the artifact — the boundary the whole
# remote-Lambda design exists to create, undone by a packaging default. Staging is free; verifying
# an exclusion you cannot see is not.
SHIPS = ["main.py", "requirements.txt", "agent", "concepts"]
NEVER_SHIP = {".env", ".env.example", "__pycache__", ".venv", "tests", "amplify", "node_modules"}


def stage() -> list[str]:
    """Copy only the allow-listed paths into dist/, then assert nothing secret came along."""
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    for name in SHIPS:
        source = HERE / name
        if not source.exists():
            sys.exit(f"cannot package: {name} is missing")
        if source.is_dir():
            shutil.copytree(
                source, DIST / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".env*"),
            )
        else:
            shutil.copy2(source, DIST / name)

    # Belt and braces: fail the build if anything on the deny-list reached the artifact.
    for path in DIST.rglob("*"):
        if path.name in NEVER_SHIP or path.name.startswith(".env"):
            sys.exit(f"REFUSING TO PACKAGE: {path.relative_to(DIST)} must never ship")
    return [str(p.relative_to(DIST)) for p in sorted(DIST.rglob("*")) if p.is_file()]


def main() -> None:
    concept = read_concept()
    columns = required_columns(concept)

    copied = copy_bundle(concept)
    build_slice(columns)

    staged = stage()
    rows = duckdb.sql(f"SELECT count(*) FROM read_parquet('{DATA_OUT}')").fetchone()[0]
    print(f"bundle  {BUNDLE_OUT.relative_to(HERE)}/  — {len(copied)} verified concepts "
          f"+ the capability")
    print(f"slice   {DATA_OUT.relative_to(HERE)} — {rows:,} rows x {len(columns)} columns, "
          f"{DATA_OUT.stat().st_size / 1024:.0f} KB")
    print(f"columns {', '.join(columns)}")
    print(f"package dist/ — {len(staged)} files, no .env, no tests, no amplify")


if __name__ == "__main__":
    main()
