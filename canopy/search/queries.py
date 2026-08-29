"""Turning a research question into the queries an index will answer, and the criteria to screen by.

Two paths, and the record always says which one it took:

* **model** — one structured call turns the question into a handful of query strings and the
  inclusion criteria a screener will apply. One call, because query building is cheap thinking
  and the expensive part (reading 200 abstracts) is downstream.
* **template** — no model available: the queries are built from the protocol's OWN words (the
  group labels and their synonyms, the outcome labels), or, with no protocol, from the content
  words of the user's sentence. Crude, free, and — the point — made of words the user wrote, so
  a reader of the record can see exactly why each paper was proposed.

The second path exists because refusing to search when a model is missing would be refusing a
working feature over an optional part of it: the indexes are free, and a list of candidates with
their abstracts is still worth a person's afternoon. What must never happen is the tool implying
it screened when it did not — hence `SearchRecord.query_source`, which the page shows.

Nothing here knows a research field. The vocabulary comes from the user's protocol or the user's
sentence, never from a list of topics this module carries.
"""
from __future__ import annotations

import re
from typing import Any

__all__ = ["QUERY_SCHEMA", "PROMPT_VERSION", "SYSTEM", "build_queries", "template_queries",
           "MAX_QUERIES"]

PROMPT_VERSION = "search-queries-1"

#: More than this and the indexes are being asked the same thing in slightly different words:
#: each query costs a round trip and returns an overlapping page, and the dedupe pass then
#: throws most of it away. Measured against the record's `unique_contributed`, not guessed.
MAX_QUERIES = 6

#: The screener needs criteria that are about the STUDY, not about the wording of the question.
#: Five is enough to separate "compares two groups on an outcome" from "reviews the literature
#: about them", which is the distinction that does the work.
MAX_CRITERIA = 6

SYSTEM = ("You help a researcher find the papers for a meta-analysis. You write search queries "
          "and inclusion criteria. You never decide what a paper found, and you never invent a "
          "citation: you are choosing what to look for, not what is true.")

_PROMPT = """A researcher is running a meta-analysis. Their question, in their own words:

{QUESTION}
{PROTOCOL}
Write the queries that would find the primary studies answering it in a scholarly index
(Europe PMC, OpenAlex — plain keyword strings, no field prefixes, no boolean operators beyond
AND/OR, each 2-10 words). Vary the vocabulary the literature actually uses: a paper about the
same thing may call it by another name, and the queries together should cover those names.

Then write the inclusion criteria a screener will apply to a title and abstract. Criteria are
about the STUDY: what it must measure, on whom, and in what design. A review, an editorial or a
protocol without results is not a primary study.

Do not restrict by date or language unless the researcher's own words do."""

#: The house schema style: every field required, nothing extra, and no field could ever hold a
#: computed statistic (`llm.schemas.assert_no_derived_stats` audits this module).
QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["queries", "criteria"],
    "properties": {
        "queries": {
            "type": "array",
            "maxItems": MAX_QUERIES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "why"],
                "properties": {
                    "text": {"type": "string",
                             "description": "The query string, as it would be typed into a "
                                            "scholarly index."},
                    "why": {"type": "string",
                            "description": "What this query is reaching for that the others "
                                           "are not."},
                },
            },
        },
        "criteria": {
            "type": "array",
            "maxItems": MAX_CRITERIA,
            "items": {"type": "string",
                      "description": "One inclusion criterion, about the study rather than "
                                     "about the wording of the question."},
        },
    },
}


def build_queries(client: Any, question: str, *, model: str,
                  protocol: Any = None, max_tokens: int = 4000) -> dict[str, Any]:
    """One structured call: `{queries: [{text, why}], criteria: [...], cost_usd, source}`.

    Falls back to the template path — it does not raise — when the call fails: an index search
    the user asked for should not die because a model was briefly unavailable, and the record
    says which path produced the queries either way.
    """
    described = _protocol_block(protocol)
    try:
        result = client.structured(
            model=model, system=SYSTEM, schema=QUERY_SCHEMA, effort="medium",
            max_tokens=max_tokens, prompt_version=PROMPT_VERSION,
            messages=[{"role": "user",
                       "content": _PROMPT.format(QUESTION=question.strip()[:2000],
                                                 PROTOCOL=described)}])
    except Exception as exc:                               # noqa: BLE001 - reported, not raised
        fallback = template_queries(question, protocol)
        fallback["notes"] = [f"the query call failed ({type(exc).__name__}), so the queries "
                             f"below were built from your own words instead"]
        return fallback

    parsed = dict(result.parsed or {})
    queries = [{"text": " ".join(str(q.get("text") or "").split())[:200],
                "why": str(q.get("why") or "")[:300]}
               for q in (parsed.get("queries") or []) if str(q.get("text") or "").strip()]
    criteria = [str(c).strip()[:300] for c in (parsed.get("criteria") or []) if str(c).strip()]
    if not queries:
        # a call that returned nothing usable is the same situation as no call at all, and the
        # honest thing is the user's own words rather than an empty search
        fallback = template_queries(question, protocol)
        fallback["notes"] = ["the model returned no usable query, so your own words were used"]
        return fallback
    return {"queries": queries[:MAX_QUERIES], "criteria": criteria[:MAX_CRITERIA],
            "source": "model", "notes": [],
            "cost_usd": round(float(getattr(result, "cost_usd", 0.0)), 6)}


def _protocol_block(protocol: Any) -> str:
    """The protocol's own words, handed to the model as context — never as instructions."""
    if protocol is None:
        return ""
    lines = ["", "Their protocol compares these groups on these outcomes (their words):"]
    for key in ("group_a", "group_b"):
        group = getattr(protocol, key, None)
        if group is None:
            continue
        synonyms = ", ".join(getattr(group, "synonyms", None) or [])
        lines.append(f"- {getattr(group, 'label', '')}"
                     + (f" (also called: {synonyms})" if synonyms else ""))
    for outcome in getattr(protocol, "outcomes", None) or []:
        lines.append(f"- outcome: {getattr(outcome, 'label', '')}"
                     + (f" — {getattr(outcome, 'definition', '')[:160]}"
                        if getattr(outcome, "definition", "") else ""))
    return "\n".join(lines) + "\n"


#: words that carry no search signal. Deliberately short and general — a stopword list that
#: grew domain terms would be this module quietly deciding what a field is about.
_STOPWORDS = frozenset("""
a an the and or of in on for to from with without by is are was were be been being do does did
this that these those it its as at than then there their they them we our us you your i
does effect effects study studies research compared comparison versus vs between among any
what which who whom how why when where whether more most less least much many
""".split())


def template_queries(question: str, protocol: Any = None) -> dict[str, Any]:
    """Queries made only of words the user wrote — the keyless path.

    With a protocol: one block of group vocabulary (OR-ed, because a paper need only be about
    the comparison) crossed with each outcome's label, plus the outcome words alone as a wider
    net. Without one: the quoted phrases of the question kept whole, then its content words.
    """
    quoted = [m.strip() for m in re.findall(r'"([^"]{3,80})"', question)]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", question.lower())
             if w not in _STOPWORDS]
    queries: list[dict[str, str]] = []

    groups: list[str] = []
    outcomes: list[str] = []
    for key in ("group_a", "group_b"):
        group = getattr(protocol, key, None) if protocol is not None else None
        if group is None:
            continue
        # three words PER ARM, not six overall: a comparison query whose vocabulary is all one
        # arm's ("older OR elderly OR aged") finds papers about old people, not papers that
        # compared them with anybody — and the second arm is exactly what makes a study eligible
        mine = [str(getattr(group, "label", "") or "")]
        mine += [str(s) for s in (getattr(group, "synonyms", None) or [])]
        groups += [w.strip() for w in mine if w.strip()][:3]
    for outcome in (getattr(protocol, "outcomes", None) or []) if protocol is not None else []:
        label = str(getattr(outcome, "label", "") or "").strip()
        if label:
            outcomes.append(label)

    groups = [g for g in dict.fromkeys(groups) if g][:6]
    if groups and outcomes:
        group_block = " OR ".join(sorted(set(groups)))
        for label in outcomes[:3]:
            queries.append({"text": f"({group_block}) AND {label}",
                            "why": f"your protocol's groups, crossed with the outcome "
                                   f"{label!r}"})
    for label in outcomes[:2]:
        queries.append({"text": label, "why": f"the outcome {label!r} on its own, as a wider net"})
    for phrase in quoted[:2]:
        queries.append({"text": phrase, "why": "a phrase you put in quotation marks"})
    # …and the sentence's own content words, ALWAYS when the protocol contributed little. A
    # quoted phrase is a good query and a bad search on its own: it finds the papers that use
    # exactly those words and misses every paper that says the same thing differently.
    if words and len(queries) < 2:
        unique = list(dict.fromkeys(words))       # first occurrence wins, order preserved
        queries.append({"text": " ".join(unique[:8]),
                        "why": "the content words of your question"})
        if len(unique) >= 4:
            queries.append({"text": " ".join(unique[:4]),
                            "why": "the first few content words, as a narrower query"})

    criteria: list[str] = []
    if groups:
        criteria.append("compares the groups the protocol names (or their synonyms)")
    for label in outcomes[:3]:
        criteria.append(f"reports {label}")
    criteria.append("is a primary study with results, not a review, editorial or protocol")
    return {"queries": queries[:MAX_QUERIES], "criteria": criteria[:MAX_CRITERIA],
            "source": "template", "notes": [], "cost_usd": 0.0}
