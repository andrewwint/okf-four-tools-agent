"""`okf_facts` — retrieval over the verified OKF bundle.

The bundle is five markdown files. That is worth saying plainly, because the reflex is to reach
for a vector database and the arithmetic does not support it: embedding five documents to pick
one is more infrastructure than the problem has. A term-overlap score over the whole bundle costs
microseconds, ships as ~60 lines, and adds no dependency — this module deliberately does not
import sklearn, which would add tens of megabytes to a deployment package to rank five items.

The grounding does not come from the retrieval. It comes from the fact that only concepts which
passed execution-grounded verification are in the bundle at all: a quarantined figure is not a
file here, so it cannot be retrieved. Retrieval just picks which verified thing to show.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

BUNDLE = Path(__file__).resolve().parent / "bundle"

# Question scaffolding and measurement vocabulary. Strip these and what remains is the TOPIC
# the question is about — which is the only thing that should decide whether the bundle can
# answer it. "What percent of adults have asthma?" reduces to {asthma}; "how many take insulin"
# reduces to {insulin}. That is the comparison we want to make.
STOPWORDS = frozenset(
    "a an and are as at be by for from had has have in is it its of on or that the to was were "
    "with this these those been being not no can may must their there which who "
    # question words
    "what how why when where who whom which does did do can could would should tell show give me "
    # measurement vocabulary — true of every concept, so it never picks one out
    "percent percentage prevalence rate ratio share proportion average mean median number count "
    "many much total figure statistic estimate weighted survey national interview data file "
    "public use year 2023 nhis cdc sample adult adults among population people person us "
    # generic verbs
    "take taking takes use uses using get gets have has had report reports reported currently "
    # NHIS survey phrasing. "Ever told you had X" is how every one of these items is worded, so
    # the phrasing is scaffolding and X is the topic. Leaving it in made the tie-break prefer
    # whichever title happened to carry fewer of these filler words.
    "ever told you your about first top coded records whether item asked"
    .split()
)

# A concept must cover most of the question's topic words to answer it. This threshold is
# load-bearing: an early version scored raw word overlap and answered "what is the prevalence of
# asthma?" with the DIABETES figure, tagged VERIFIED — because "prevalence" matched and "asthma"
# simply went unnoticed. A wrong number wearing the verified badge is the worst thing this system
# can emit, so the tie-break is always toward refusing.
MIN_SCORE = 0.6


@dataclass
class Concept:
    id: str
    title: str
    frontmatter: dict
    body: str

    @property
    def statistic(self) -> str | None:
        fm = self.frontmatter
        if fm.get("value_pct") is not None:
            return f"{fm.get('statistic', self.title)}: {fm['value_pct']}%"
        if fm.get("value") is not None:
            return f"{fm.get('statistic', self.title)}: {fm['value']} {fm.get('unit', '')}".strip()
        return None

    @property
    def citation(self) -> str:
        return f"{self.id} ({self.frontmatter.get('source', 'NHIS 2023 Sample Adult')})"


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]{3,}", text.lower()) if w not in STOPWORDS}


def load_bundle(path: Path = BUNDLE) -> list[Concept]:
    """The verified FACTS in the bundle.

    Capability concepts are deliberately excluded: `query_adult23.md` describes what the query
    tool can do, which is not an answer to anything. Left in, it competes with the facts and
    wins questions about "queries" or "figures".
    """
    concepts = []
    for file in sorted(path.glob("*.md")):
        text = file.read_text()
        match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
        if not match:
            continue
        frontmatter = yaml.safe_load(match.group(1)) or {}
        if frontmatter.get("type") == "capability":
            continue
        concepts.append(
            Concept(
                id=frontmatter.get("id", file.stem),
                title=frontmatter.get("title", file.stem),
                frontmatter=frontmatter,
                body=match.group(2).strip(),
            )
        )
    return concepts


_CACHE: list[tuple[Concept, set[str], set[str]]] | None = None


def _indexed() -> list[tuple[Concept, set[str], set[str]]]:
    """Tokenise the bundle once per process: (concept, title terms, all terms)."""
    global _CACHE
    if _CACHE is None:
        _CACHE = [
            (c, _tokens(c.title), _tokens(f"{c.title} {c.body}")) for c in load_bundle()
        ]
    return _CACHE


def search(question: str, k: int = 1) -> list[tuple[Concept, float]]:
    """The verified concepts most relevant to the question, best first.

    Two different jobs, so two different measures — conflating them is what made an early
    version answer an asthma question with the diabetes figure:

    THE GATE (may we answer at all?) is topic COVERAGE — of the topic words left after stripping
    question scaffolding, how many does this concept mention? A question about asthma keeps the
    word "asthma", no concept mentions it, coverage is zero, and we refuse.

    THE RANK (which concept?) leans on the TITLE. Every concept's body mentions every other —
    the insulin concept explains prediabetes, the prediabetes concept explains insulin — so body
    overlap cannot tell them apart. A title is what the concept is *about*. Ties break toward the
    title carrying fewest words the question never asked for, which prefers the general concept
    ("Ever told you had diabetes") over a more specific one ("Age first told had diabetes").
    """
    asked = _tokens(question)
    if not asked:
        return []

    ranked = []
    for concept, title_terms, all_terms in _indexed():
        coverage = len(asked & all_terms) / len(asked)
        # A title hit is strong evidence on its own, and it rescues honest paraphrases the
        # bundle does not share vocabulary with — "how common is diabetes" keeps the unknown
        # word "common", which would otherwise drag coverage under the bar. It cannot rescue a
        # question about something the bundle has never heard of, because no title matches.
        if coverage < MIN_SCORE and not (asked & title_terms):
            continue
        ranked.append(
            (
                len(asked & title_terms),          # title match: what is it about?
                len(asked & all_terms),            # then body support
                -len(title_terms - asked),         # then the least off-topic title
                concept,
                coverage,
            )
        )
    ranked.sort(key=lambda r: r[:3], reverse=True)
    return [(r[3], r[4]) for r in ranked[:k]]
