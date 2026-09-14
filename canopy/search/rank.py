"""Ranking the candidates before any cap, so a cap cuts the least likely papers and not the last
index's page.

The first real search screened the first 200 of 483 candidates IN ARRIVAL ORDER (design 01):
everything unique to the later queries sat past position 200 and was never read, and the one
answer-key paper the indexes did return late (Balitsky Thompson & Henriques 2010, position 444)
was cut by nothing but the order the indexes answered in. With the concept blocks the strings
match tens of thousands of records, so some cap always bites, and what it bites has to be a
judgement about the paper rather than about the network.

The score is deterministic and made of things the record already holds (design 03 §3):

    body_cov  = blocks of the string(s) that returned it whose terms appear in title+abstract / n
    ttl_cov   = the same, in the title alone / n
    hits      = log2(1 + rows the indexes returned for it)
    best      = 1 / (1 + best entry position / 100)
    seeds     = min(seeds citing or cited by it, 5) / 5        (citation rounds only)
    s = 3·body_cov + 2·ttl_cov + hits + best + 2·seeds + 0.25 if it has an abstract

Coverage is literal: an expanded term present, case-folded, hyphens as spaces. It is the same
test the index made, applied to the two fields a screener will read, so a paper that matched
the string in its full text but says none of it in the abstract ranks below one that says all
of it in the title. No model, no field knowledge, and the ordering is total (`(-s, key)`), so two
runs of one search rank the same list the same way.
"""
from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

from .models import Candidate

__all__ = ["coverage", "score", "rank", "why"]

_FOLD = re.compile(r"[\s\-]+")


def _fold(text: str) -> str:
    return " " + _FOLD.sub(" ", str(text or "").lower()) + " "


def coverage(text: str, terms: Sequence[str]) -> bool:
    """Is any of `terms` literally in `text` (case-folded, hyphens → space, whole-word)?"""
    folded = _fold(text)
    for term in terms:
        needle = _fold(term).strip()
        if needle and re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", folded):
            return True
    return False


def _blocks_for(candidate: Candidate, plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The non-empty blocks of every string that returned this candidate; every block when the
    record cannot say which string did (a candidate built by hand, an earlier record)."""
    from .blocks import QUERY_IDS, block_terms, blocks_of_query

    ranks = getattr(candidate, "ranks", None) or {}
    query_ids = {str(label).split(":")[1] for label in ranks if str(label).count(":") >= 2}
    query_ids = {q for q in query_ids if q in QUERY_IDS} or set(QUERY_IDS)
    seen: list[Mapping[str, Any]] = []
    for query_id in sorted(query_ids):
        for block in blocks_of_query(plan, query_id):
            if block_terms(block) and block not in seen:
                seen.append(block)
    return seen


def score(candidate: Candidate, plan: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    """`(s, parts)` — the number and the numbers it was made of, for `relevance_why`."""
    from .blocks import block_terms

    blocks = _blocks_for(candidate, plan)
    n = len(blocks) or 1
    title = candidate.title or ""
    body = f"{title} {candidate.abstract or ''}"
    body_cov = sum(1 for b in blocks if coverage(body, block_terms(b))) / n
    ttl_cov = sum(1 for b in blocks if coverage(title, block_terms(b))) / n
    hits = math.log2(1 + max(1, int(candidate.n_rows or len(candidate.found_by) or 1)))
    positions = [int(v) for v in (getattr(candidate, "ranks", None) or {}).values()
                 if isinstance(v, (int, float)) and v > 0]
    best = 1.0 / (1.0 + (min(positions) if positions else 1000) / 100.0)
    seeds = min(int(getattr(candidate, "cited_by_seeds", 0) or 0), 5) / 5.0
    has_abstract = 0.25 if (candidate.abstract or "").strip() else 0.0
    total = 3 * body_cov + 2 * ttl_cov + hits + best + 2 * seeds + has_abstract
    parts = {"body_cov": round(body_cov, 3), "ttl_cov": round(ttl_cov, 3),
             "hits": round(hits, 3), "best": round(best, 3), "seeds": round(seeds, 3),
             "abstract": has_abstract, "n_blocks": len(blocks),
             "best_position": min(positions) if positions else None}
    return round(total, 4), parts


def why(parts: Mapping[str, Any]) -> str:
    """One line a reader can check against the record."""
    return (f"{parts['n_blocks']} block(s): {parts['body_cov']:.2f} covered in title+abstract, "
            f"{parts['ttl_cov']:.2f} in the title; {parts['hits']:.2f} for the rows returned; "
            f"best entry position {parts['best_position'] or '—'}"
            + (f"; cited by {int(parts['seeds'] * 5)} seed(s)" if parts["seeds"] else "")
            + ("" if parts["abstract"] else "; no abstract"))


def rank(candidates: Sequence[Candidate], plan: Mapping[str, Any]) -> list[Candidate]:
    """Every candidate scored (`relevance`, `relevance_why` written on it) and sorted best first.
    Total order: ties break on the key, so the list is the same list every time."""
    for candidate in candidates:
        total, parts = score(candidate, plan)
        candidate.relevance = total
        candidate.relevance_why = why(parts)
    return sorted(candidates, key=lambda c: (-(c.relevance or 0.0), c.key))
