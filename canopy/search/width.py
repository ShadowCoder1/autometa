"""Width control: measure each string on Europe PMC and prune the widest block until it is a
search and not a crawl.

The adversarial review measured what the concept blocks look like before this step (design 04
BLOCKER-2b): a prompt that asks for "the broad words for the process" gets them — `learning`
2.1 M records, `transfer` 2.5 M, `dynamics` 2.1 M, a group word with a second meaning in
genetics 1.6 M, and in the second key's protocol a MeSH term that sits on every paper about
its population, 8.2 M — and the AND'd strings matched 1.1–2.1 million records with **none** of
the twenty-three answer-key papers in the top thousand. The string that reached 21 of 23 (`01` string B) was narrow because a person pruned it
by hand. This module does that by measurement:

    for every term in every block: its hit count on Europe PMC (free, one request each)
    while the AND'd string matches more than EPMC_MAX_HITS:
        the widest block (measured, not guessed) loses its highest-count term
    then, once on OpenAlex's title-and-abstract search: while over OA_TA_MAX_HITS, one more

Two rules decide the victim. **A protocol term is never pruned** — the user's own labels and
synonyms are the search's reason for existing, and the protection is what the person wrote,
not a list of a field's words. **An expander variant is sacrificed before a model term** when
its count is within a factor of two of the widest model term's: the v2 verification's stress
case lost a key paper to the model's `dynamics` when the expander's `dynamic` would have
narrowed the string almost as much (04 v2 verification §2). Every drop is written on the block
(`pruned: [{term, hits, kind, iteration, query}]`) with the count that condemned it, so the bench
can show exactly which paper a drop cost and a reader of `search.json` can see why a word they
expected is not in the string.

Requests: ≈ 100 per-term counts + up to 40 × (3 block widths + 1 total) on Europe PMC at 0.2 s,
and ≤ 10 OpenAlex counts at $0.001 each. A count that cannot be measured (an index down, a 429,
an offline replay with no recording) leaves the term `unmeasured` and never prunable — the
search proceeds with what it could measure and the plan says what it could not.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .expand import render_block, render_query, render_term

__all__ = ["EPMC_MAX_HITS", "OA_TA_MAX_HITS", "MAX_ITERATIONS", "MAX_OA_CHECKS",
           "VARIANT_MARGIN", "prune", "count"]

#: the AND'd string's ceiling on Europe PMC. String B — 21/23 reached, 8 in the top 1,000 — sat at
#: 110 k; the unpruned design strings at 1.1–2.1 M held 0/23.
EPMC_MAX_HITS = 150_000
#: …and on OpenAlex's title-and-abstract search, where B sat at 5 k and reached 18/23
OA_TA_MAX_HITS = 20_000
MAX_ITERATIONS = 40
MAX_OA_CHECKS = 10
#: a variant within this factor of the widest model term is dropped in its place
VARIANT_MARGIN = 2.0


def count(transport: Any, request: tuple[str, Mapping[str, Any]], *, context: Mapping[str, Any],
          read: Callable[[Any], int | None]) -> int | None:
    """One hit-count request, or None when it could not be measured. Never raises: a count the
    index would not give is a term that cannot be pruned, not a search that cannot run."""
    url, params = request
    try:
        response = transport.get_json(url, params=params, context=dict(context))
    except Exception:                                   # noqa: BLE001 - offline, or a bug
        return None
    if not getattr(response, "ok", False):
        return None
    try:
        return read(response.json())
    except Exception:                                   # noqa: BLE001 - not the JSON we expected
        return None


def _epmc_hits(payload: Any) -> int | None:
    return int(payload["hitCount"]) if isinstance(payload, dict) and "hitCount" in payload else None


def _oa_hits(payload: Any) -> int | None:
    meta = payload.get("meta") if isinstance(payload, dict) else None
    return int(meta["count"]) if isinstance(meta, dict) and "count" in meta else None


def _victim(block: Mapping[str, Any], terms: Sequence[str], df: Mapping[str, int | None],
            protected: frozenset[str]) -> tuple[str | None, str, int | None]:
    """`(term, kind, hits)` to drop from this block, or `(None, "", None)` when nothing may be.

    The highest-count term wins, with a variant beating a model term whose count is no more
    than `VARIANT_MARGIN` times its own. Protected and unmeasured terms are never candidates.
    """
    model = set(block.get("terms") or [])
    candidates = [(t, df.get(t)) for t in terms if t not in protected and df.get(t) is not None]
    if not candidates:
        return None, "", None
    top_model = max((c for c in candidates if c[0] in model), key=lambda c: c[1], default=None)
    top_variant = max((c for c in candidates if c[0] not in model), key=lambda c: c[1],
                      default=None)
    if top_variant is not None and (top_model is None
                                    or float(top_variant[1]) * VARIANT_MARGIN >= top_model[1]):
        return top_variant[0], "variant", int(top_variant[1])
    if top_model is not None:
        return top_model[0], "model", int(top_model[1])
    return None, "", None


def prune(plan: dict[str, Any], transport: Any, *, protected: frozenset[str] = frozenset(),
          epmc_max: int | None = None, oa_max: int | None = None) -> None:
    """Measure and prune the plan's blocks in place. Q1 first, then Q2 re-measured: the `task`
    and `phenomenon` blocks are shared, so a drop for Q1 narrows Q2 as well."""
    from .blocks import QUERY_IDS, block_terms, blocks_of_query
    from .indices import EuropePmc, OpenAlex

    epmc_max = EPMC_MAX_HITS if epmc_max is None else int(epmc_max)
    oa_max = OA_TA_MAX_HITS if oa_max is None else int(oa_max)

    epmc, openalex = EuropePmc(), OpenAlex()
    df: dict[str, int | None] = {}
    unmeasured: list[str] = []
    notes: list[str] = []

    def measure(term: str) -> int | None:
        if term not in df:
            rendered = render_term(term)
            df[term] = count(transport, epmc.count_request(rendered),
                             context={"index": "europepmc", "form": "count",
                                      "query_id": f"df:{term}", "page": 0,
                                      "query_text": rendered, "exact": True},
                             read=_epmc_hits)
            if df[term] is None:
                unmeasured.append(term)
        return df[term]

    def string_hits(query_id: str, blocks: Sequence[Mapping[str, Any]]) -> int | None:
        text = render_query([block_terms(b) for b in blocks])
        return count(transport, epmc.count_request(text),
                     context={"index": "europepmc", "form": "count", "query_id": query_id,
                              "page": 0, "query_text": text}, read=_epmc_hits)

    def block_hits(query_id: str, block: Mapping[str, Any]) -> int | None:
        text = render_block(block_terms(block))
        return count(transport, epmc.count_request(text),
                     context={"index": "europepmc", "form": "count",
                              "query_id": f"{query_id}:{block.get('name')}", "page": 0,
                              "query_text": text}, read=_epmc_hits)

    def oa_hits(query_id: str, blocks: Sequence[Mapping[str, Any]]) -> int | None:
        text = render_query([block_terms(b) for b in blocks])
        return count(transport, openalex.count_request(text, form="ta"),
                     context={"index": "openalex", "form": "count", "query_id": query_id,
                              "page": 0, "query_text": text}, read=_oa_hits)

    def drop_one(query_id: str, blocks: Sequence[dict[str, Any]], iteration: int,
                 stage: str) -> bool:
        widths = {b["name"]: block_hits(query_id, b) for b in blocks}
        measured = {k: v for k, v in widths.items() if v is not None}
        if not measured:
            notes.append(f"width control stopped on {query_id}: no block's width could be "
                         f"measured")
            return False
        widest = max(measured, key=lambda k: measured[k])
        block = next(b for b in blocks if b["name"] == widest)
        victim, kind, hits = _victim(block, block_terms(block), df, protected)
        if victim is None:
            notes.append(f"width control could not narrow {query_id} below "
                         f"{measured[widest]:,} hits in its widest block ({widest}) without "
                         f"dropping the protocol's own words, so it stopped there")
            return False
        block["pruned"].append({"term": victim, "hits": hits, "kind": kind,
                                "iteration": iteration, "query": query_id, "block": widest,
                                "stage": stage, "block_hits": measured[widest]})
        return True

    for block in plan.get("blocks") or []:
        for term in block_terms(block):
            measure(term)

    plan.setdefault("width", {})
    for query_id in QUERY_IDS:
        blocks = [b for b in blocks_of_query(plan, query_id)]
        if len(blocks) < len(QUERY_IDS[query_id]) or any(not block_terms(b) for b in blocks):
            continue
        total = string_hits(query_id, blocks)
        record: dict[str, Any] = {"epmc_hits_before": total, "epmc_hits_after": total,
                                  "oa_ta_hits_after": None, "iterations": 0, "oa_checks": 0}
        if total is None:
            notes.append(f"width control could not measure {query_id} on Europe PMC, so the "
                         f"string is sent as written")
            plan["width"][query_id] = record
            continue
        iterations = 0
        while total is not None and total > epmc_max and iterations < MAX_ITERATIONS:  # noqa: E501
            iterations += 1
            if not drop_one(query_id, blocks, iterations, "europepmc"):
                break
            total = string_hits(query_id, blocks)
        record["epmc_hits_after"] = total
        record["iterations"] = iterations
        if total is not None and total > epmc_max and iterations >= MAX_ITERATIONS:
            notes.append(f"width control stopped on {query_id} after {MAX_ITERATIONS} drops "
                         f"with {total:,} hits still on Europe PMC")

        oa_total = oa_hits(query_id, blocks)
        checks = 0
        while oa_total is not None and oa_total > oa_max and checks < MAX_OA_CHECKS:
            checks += 1
            if not drop_one(query_id, blocks, iterations + checks, "openalex"):
                break
            oa_total = oa_hits(query_id, blocks)
        if oa_total is None:
            notes.append(f"OpenAlex did not answer the width check for {query_id}, so its "
                         f"title-and-abstract width is unmeasured")
        record["oa_ta_hits_after"] = oa_total
        record["oa_checks"] = checks
        if checks:
            record["epmc_hits_after"] = string_hits(query_id, blocks)
        plan["width"][query_id] = record

    if unmeasured:
        plan["width"]["unmeasured"] = unmeasured
        notes.append(f"{len(unmeasured)} term(s) could not be measured on Europe PMC and were "
                     f"never candidates for pruning: {', '.join(unmeasured[:8])}"
                     + (" …" if len(unmeasured) > 8 else ""))
    plan.setdefault("notes", []).extend(notes)
