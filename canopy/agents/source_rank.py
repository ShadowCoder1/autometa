"""Which of a paper's sources for one outcome is worth voting on — decided in CODE.

Heuer & Hegele 2008 has a figure (2a, p5) with SE bars and n = 20/20 per group. The mapper found
it, listed it, and then also listed a page-4 sentence with no n and no dispersion and an Experiment
2 figure measuring something else — and the vote treated all three as equals, so the paper produced
two `not_convertible` rows while carrying a perfectly usable figure.

Two rules, both deliberately dull:

* the rank is built from fields the mapper already returns, never from a self-reported ranking. A
  model's own "this is the best source" is one more unwitnessed number, and the mapper is the agent
  that got the choice wrong in the first place;
* the rank gates what enters the **vote**, not what gets extracted. Extraction keeps reading every
  figure source, because the losing sources are what a reviewer looks at to see that the choice was
  right, and because a source that loses on paper sometimes carries the only readable number.

Every source keeps its score AND the reason it lost, so the ranking is a thing a human can argue
with rather than a silent filter.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..models import Candidate, DispersionType, OutcomeSources, Source
from ..verify.figures import FIGURE_KINDS

__all__ = ["RankedSource", "rank_sources", "source_of", "keep_for_vote", "N_RE"]

#: "n = 20", "n=20 per group", "(N = 12)" — a group size printed beside the values
N_RE = re.compile(r"\bn\s*[=:]\s*\d+", re.IGNORECASE)

#: what each property is worth. Dispersion dominates because without it a figure cannot become an
#: effect size at all; the analysis metric comes next because a source measuring the wrong quantity
#: is worse than one measuring the right quantity badly.
DISPERSION_POINTS = 3
METRIC_POINTS = 2
N_POINTS = 2
VALUES_POINTS = 1


@dataclass
class RankedSource:
    """One source with its score and, when it lost, exactly what it lost on."""

    source: Source
    score: int = 0
    has_dispersion: bool = False
    has_n: bool = False
    matches_metric: bool = False
    reasons: list[str] = field(default_factory=list)
    losing_reasons: list[str] = field(default_factory=list)

    @property
    def locator(self) -> str:
        return self.source.locator or f"page {self.source.page}"

    def to_dict(self) -> dict:
        return {"locator": self.locator, "page": self.source.page,
                "figure_id": self.source.figure_id, "table_id": self.source.table_id,
                "kind": getattr(self.source.kind, "value", str(self.source.kind)),
                "score": self.score, "has_dispersion": self.has_dispersion,
                "has_n": self.has_n, "matches_metric": self.matches_metric,
                "reasons": list(self.reasons), "losing_reasons": list(self.losing_reasons)}


def rank_sources(sources: Sequence[Source], outcome: OutcomeSources | None = None,
                 ) -> list[RankedSource]:
    """Every source, best first, each carrying why it scored what it did."""
    wanted = (outcome.analysis_metric if outcome is not None else "unknown") or "unknown"
    ranked: list[RankedSource] = []
    for source in sources:
        row = RankedSource(source=source)
        row.has_dispersion = source.error_bar_type not in (DispersionType.UNKNOWN,
                                                           DispersionType.NONE)
        row.has_n = bool(N_RE.search(source.values_in_text or "")
                         or N_RE.search(source.quote or ""))
        row.matches_metric = (wanted != "unknown" and source.analysis_metric == wanted)
        if row.has_dispersion:
            row.score += DISPERSION_POINTS
            row.reasons.append(f"carries {source.error_bar_type.value} error bars")
        else:
            row.losing_reasons.append("no dispersion was determined here, so a value read from it "
                                      "cannot become an effect size on its own")
        if row.matches_metric:
            row.score += METRIC_POINTS
            row.reasons.append(f"measures the outcome's own metric ({wanted})")
        elif wanted != "unknown" and source.analysis_metric != "unknown":
            row.losing_reasons.append(
                f"measures {source.analysis_metric}, where this outcome is {wanted}")
        if row.has_n:
            row.score += N_POINTS
            row.reasons.append("states a group size beside the values")
        else:
            row.losing_reasons.append("no group size is printed with it")
        if source.values_in_text:
            row.score += VALUES_POINTS
            row.reasons.append("the values themselves are printed")
        ranked.append(row)
    ranked.sort(key=lambda r: (-r.score, r.source.page, r.locator))
    best = ranked[0].score if ranked else 0
    for row in ranked:
        if row.score < best:
            row.losing_reasons.insert(0, f"scored {row.score} against the best source's {best}")
    return ranked


def source_of(cand: Candidate, sources: Sequence[Source]) -> Source | None:
    """The source a candidate was read from — figure id first, then table id, then the page."""
    figure_id = (cand.pixel_provenance or {}).get("figure_id")
    for source in sources:
        if source.figure_id and figure_id and source.figure_id == figure_id:
            return source
    for source in sources:
        if source.table_id and source.table_id == (cand.pixel_provenance or {}).get("table_id"):
            return source
    same_page = [s for s in sources if cand.page is not None and s.page == cand.page]
    return same_page[0] if len(same_page) == 1 else None


def keep_for_vote(candidates: Sequence[Candidate], ranked: Sequence[RankedSource]
                  ) -> tuple[list[Candidate], list[str]]:
    """`(candidates the vote may see, why the others were held back)`.

    Deliberately narrow. The rank exists because Heuer & Hegele's map put a page-4 sentence with no
    n and no dispersion beside a figure with SE bars and n = 20/20, and the vote weighed them
    equally — so a candidate is held back only when its source **cannot produce an effect size at
    all** (no dispersion was ever determined there) while a better-scoring source can.

    Everything else votes. A source that merely scores lower still carries evidence: `verify.vote`
    already knows that a figure may corroborate a printed value and never replace it, and evicting
    the figure would remove the corroboration rather than the error. A candidate whose source
    cannot be identified votes too — the rank is a preference over known locations, not a licence
    to drop a reading nobody can place — and the filter never empties the cell.
    """
    if len(ranked) < 2 or not ranked[0].has_dispersion:
        return list(candidates), []
    best = ranked[0].score
    losers = {id(row.source): row for row in ranked
              if row.score < best and not row.has_dispersion}
    if not losers:
        return list(candidates), []
    sources = [row.source for row in ranked]
    kept: list[Candidate] = []
    held: list[str] = []
    for cand in candidates:
        source = source_of(cand, sources)
        row = losers.get(id(source)) if source is not None else None
        if row is None:
            kept.append(cand)
            continue
        held.append(f"{cand.candidate_id} (from {row.locator}): "
                    f"{'; '.join(row.losing_reasons)}")
    if not kept:
        return list(candidates), []
    return kept, held



_WORD_RE = re.compile(r"[^a-z0-9]+")
_PAGE_RE = re.compile(r"\bp(?:age|g)?\.?\s*(\d+)", re.IGNORECASE)


def _norm(text: str) -> str:
    lowered = (text or "").lower().replace("figure", "fig").replace("table", "tab")
    return _WORD_RE.sub("", lowered)


def match_named_source(named: str, sources: Sequence[Source]) -> Source | None:
    """The source in the mapper's own list that a verifier's `better_source` names, or `None`.

    The verifier's answer is prose ("Figure 2a, page 5, which has SE bars"), and acting on prose
    means acting on a page number a model invented. So extraction is only ever re-opened on a
    location the mapper ALREADY found: this matches the name against that list and refuses
    anything it cannot place.
    """
    from difflib import SequenceMatcher

    text = _norm(named)
    if not text:
        return None
    page = _PAGE_RE.search(named or "")
    want_page = int(page.group(1)) if page else None
    best, score = None, 0.0
    for source in sources:
        locator = _norm(source.locator)
        if not locator:
            continue
        ratio = 1.0 if locator and locator in text else SequenceMatcher(None, locator, text).ratio()
        if want_page is not None and source.page == want_page:
            ratio += 0.2
        if ratio > score:
            best, score = source, ratio
    return best if score >= 0.6 else None
