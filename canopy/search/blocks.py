"""Turning a research question into concept blocks — the way a review team's librarian does it.

One model call writes four blocks (`groups`, `task`, `phenomenon`, `related_designs`: the words
the literature uses for each concept, ≤ 20 a block) and the rubric a screener will apply. This
module clamps them, expands every term into its spelling variants (`expand.py`), has the width
control measure and prune them (`width.py`), and renders the two Boolean strings the indexes are
asked: `Q1 = groups ∧ task ∧ phenomenon` and `Q2 = related_designs ∧ task ∧ phenomenon` (design
03 §1). Everything that happened to a term on the way is on the plan — its variants under
`expanded`, a pruned term under `pruned` with the hit count that condemned it — so a reader of
`search.json` can see why the string is what it is.

Why blocks and not the six short queries of v1 (design 01): fourteen of the twenty-three papers in
the first answer key were missed for VOCABULARY — each query AND'd five or six words, and a paper
that said the concept in another word than the query's was unreachable. A block OR's every form
of one concept, so a paper is found when it says the concept in any of its words; the AND across
blocks is what keeps the string about the question.

Two paths, and the record always says which: `model` (the call) and `template` (no model, or the
call failed: the blocks are the protocol's own labels and synonyms and the question's content
words). Nothing here knows a research field — every term came from the user's words or the
model's reading of them, and the clamps are counts.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .expand import (MAX_STRING_CHARS, STOPWORDS, head_words, known_words, render_block,
                     render_query, render_term, term_tokens, variants)

__all__ = ["PROMPT_VERSION", "SYSTEM", "PROMPT", "BLOCK_SCHEMA", "BLOCK_NAMES", "MAX_TERMS",
           "MAX_TERM_CHARS", "MAX_RUBRIC", "QUERY_IDS", "INDEX_FORMS", "build_plan",
           "template_plan", "plan_from_answer", "expand_plan", "render_strings",
           "block_terms", "protected_terms", "query_rows", "blocks_of_query", "check_seeds",
           "SEED_REWRITE_PROMPT", "MAX_SEEDS"]

PROMPT_VERSION = "search-blocks-3"

BLOCK_NAMES: tuple[str, ...] = ("groups", "task", "phenomenon", "related_designs")
#: the two strings and the blocks each AND's. Q2 swaps the groups block for the designs that
#: contain both groups without being about them; it is never AND'd with `groups`.
QUERY_IDS: dict[str, tuple[str, ...]] = {"Q1": ("groups", "task", "phenomenon"),
                                         "Q2": ("related_designs", "task", "phenomenon")}
#: the index forms each string is sent in, in the order the indexes are asked (design 03 §2:
#: by measured ranking quality — PubMed first, Europe PMC deep, OpenAlex full text only when the
#: string fits its 1,500-character limit)
INDEX_FORMS: tuple[tuple[str, str], ...] = (("pubmed", ""), ("openalex", "ta"),
                                            ("europepmc", ""), ("openalex", "ft"))

#: the model is told all three: terms a block, characters a term, rules in the rubric
MAX_TERMS = 20
MAX_TERM_CHARS = 60
MAX_RUBRIC = 12
#: a block with fewer terms than this is not a concept, it is a word: the template's block for
#: that name stands in, and the plan says so
MIN_TERMS = 2

SYSTEM = (
    "You help a researcher find the primary studies for a meta-analysis. You write the vocabulary "
    "a scholarly index must be asked with; you never decide what any paper found, and you never "
    "invent a citation. You are choosing what to look for, not what is true."
)

PROMPT = """A researcher is running a meta-analysis. Their question, in their own words:

{QUESTION}

{PROTOCOL}

Write the CONCEPT BLOCKS a systematic-review librarian would build for this question. A block is
one concept and the list of words and short phrases the literature uses for it. The search will OR
the terms inside a block and AND the blocks together, so a paper is found only when EVERY block
matches it somewhere in its title or abstract. Design for two consequences of that:

1. A block with too few terms loses every paper that says the same thing differently. List every
   word form and synonym you know for the concept: the noun, the adjective, the verb, the negated
   form, the lay term, the historical term, the abbreviation, and the vocabulary of neighbouring fields
   that study the same thing under another name.
2. A block that is not a genuine requirement loses papers that meet the question. Only make a block
   for a concept that EVERY eligible primary study must mention in its title or abstract.

Write exactly these four blocks, in this order:

- "groups": how papers name the two groups the protocol compares — both groups, and the words for
  the dimension that separates them (the property, the contrast, the factor). Start from the
  protocol's labels and synonyms and add the forms the literature uses that the protocol did not
  list.
- "task": the task, exposure, intervention or design every eligible study must have used — the
  words for what was done to participants or what they did.
- "phenomenon": the phenomenon or outcome family the studies measure. Include the broad words for
  the process as well as the names of specific measures, because a paper that reports an eligible
  outcome may only call it by the process name in its abstract. Do NOT write one required term per
  protocol outcome: any one outcome suffices, so this block must reach a paper that reports only
  one of them.
- "related_designs": designs that are ABOUT something else but must contain both of the protocol's
  groups to work — a design that transfers, generalizes, compares or counterbalances across the
  dimension that separates group A from group B; a manipulation replicated in both groups; a
  clinical study with control groups drawn from both. Write the words such papers use for the two
  groups when they are not the paper's stated topic. For example, depending on the field, that
  might be the untrained, opposite, other or second group; a placebo, sham or wait-list arm; a
  dose arm; an age-matched or comparison group — use the words YOUR field's literature uses. This
  block is used INSTEAD of "groups" in a second search string; it is not ANDed with it.

Each block holds AT MOST 20 terms. Start every block with the BARE WORDS — the single noun or
adjective your field's literature uses for that concept on its own, the word a paper would put
in its own title; at least five of them per block — and only then add multi-word phrases for
the forms that a bare word cannot capture. A quoted phrase matches only that exact phrase, so a
paper that names the concept in a sentence of its own words is reached by the bare word and
missed by the phrase. The search generates spelling, hyphen and plural variants itself, measures
how many records each term matches, and drops the broadest terms first if the string is too wide
— so a bare word that turns out too broad costs nothing, while a phrase that is too narrow
silently loses papers. Breadth is the search's problem to fix; narrowness is not.

For each block say in one sentence what it is reaching for.

Then write the RUBRIC a title-and-abstract screener will apply. Take each of the protocol's
eligibility rules in turn, and add any rule the protocol implies but does not state (for example
"a primary study with results, not a review, editorial or protocol"). Classify each rule as one of:
language, population, task, outcome, comparison, date, other. Say whether an abstract can
definitively FAIL it: that is true only for language, population, task and date rules, and only
when the abstract can prove the failure. Outcome and comparison rules can be met by an abstract
but never failed by one.

Rules:
- Terms are plain words or short phrases: no field prefixes, no wildcards, no boolean operators.
  Bare single words first, at least five per block; write a multi-word term exactly as it
  appears in prose. The search adds plural, hyphenated and spelling variants itself, so do not
  list those.
- Use only vocabulary that describes what the researcher asked for. Do not add a restriction the
  researcher did not make (date, language, species, age, setting) unless their words or their
  protocol do.
- Do not name any specific paper, author, laboratory or journal."""

#: the house schema style: every field required, nothing extra, no derived statistic anywhere.
#: `name` is a plain string (not an enum) so the house rule that every enum carries an `unknown`
#: member does not force a fifth block; `kind` carries `unknown` for the same rule.
BLOCK_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["blocks", "rubric"],
    "properties": {
        "blocks": {"type": "array", "maxItems": 4, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["name", "why", "terms"],
            "properties": {
                "name": {"type": "string",
                         "description": "groups | task | phenomenon | related_designs"},
                "why": {"type": "string",
                        "description": "One sentence: what this block reaches for."},
                "terms": {"type": "array", "maxItems": MAX_TERMS, "items": {
                    "type": "string",
                    "description": "One word or short phrase as it appears in prose. At most 20 "
                                   "per block."}}}}},
        "rubric": {"type": "array", "maxItems": MAX_RUBRIC, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["rule", "kind", "abstract_can_fail"],
            "properties": {
                "rule": {"type": "string",
                         "description": "The rule, in the protocol's words or yours."},
                "kind": {"type": "string",
                         "enum": ["language", "population", "task", "outcome", "comparison",
                                  "date", "other", "unknown"]},
                "abstract_can_fail": {
                    "type": "boolean",
                    "description": "True only for language, population, task and date rules."}}}},
    },
}


# ------------------------------------------------------------------------------ the protocol
def _protocol_block(protocol: Any) -> str:
    """The protocol's own words, handed to the model as context — never as instructions."""
    if protocol is None:
        return ""
    lines = ["Their protocol, in their words:", ""]
    for key, letter in (("group_a", "A"), ("group_b", "B")):
        group = getattr(protocol, key, None)
        if group is None:
            continue
        lines.append(f"GROUP {letter} — {getattr(group, 'label', '')}: "
                     f"{' '.join(str(getattr(group, 'definition', '') or '').split())}")
        synonyms = [str(s) for s in (getattr(group, "synonyms", None) or []) if str(s).strip()]
        if synonyms:
            lines.append(f"  also called: {', '.join(synonyms)}")
    outcomes = list(getattr(protocol, "outcomes", None) or [])
    if outcomes:
        lines.append("OUTCOMES (any one suffices):")
        for outcome in outcomes:
            definition = " ".join(str(getattr(outcome, "definition", "") or "").split())[:300]
            lines.append(f"  - {getattr(outcome, 'label', '')}: {definition}")
    rules = [str(r) for r in (getattr(protocol, "eligibility", None) or []) if str(r).strip()]
    if rules:
        lines.append("ELIGIBILITY RULES, verbatim:")
        lines.extend(f"  {i}. {rule}" for i, rule in enumerate(rules, start=1))
    return "\n".join(lines)


def protected_terms(protocol: Any) -> frozenset[str]:
    """The user's own vocabulary — the group labels and synonyms, verbatim, lower-cased — which
    width control may never prune. The protection is what the person wrote, not a domain list."""
    words: set[str] = set()
    if protocol is None:
        return frozenset()
    for key in ("group_a", "group_b"):
        group = getattr(protocol, key, None)
        if group is None:
            continue
        for term in [getattr(group, "label", "")] + list(getattr(group, "synonyms", None) or []):
            clean = " ".join(str(term or "").lower().split())
            if clean:
                words.add(clean)
    return frozenset(words)


def _clean_term(raw: Any) -> str:
    text = " ".join(str(raw or "").replace('"', " ").split()).strip(" ,;:.").lower()
    return text[:MAX_TERM_CHARS]


def _empty_block(name: str, why: str = "") -> dict[str, Any]:
    return {"name": name, "why": why[:300], "terms": [], "expanded": {}, "pruned": []}


# ------------------------------------------------------------------------------- the two paths
def plan_from_answer(parsed: Mapping[str, Any], *, question: str,
                     protocol: Any) -> dict[str, Any]:
    """The model's answer as a plan: names normalised, terms clamped, a thin block replaced by
    the template's block for that name (and the replacement written down). No expansion yet."""
    notes: list[str] = []
    blocks: dict[str, dict[str, Any]] = {}
    for raw in (parsed or {}).get("blocks") or []:
        if not isinstance(raw, Mapping):
            continue
        name = re.sub(r"[^a-z]", "_", str(raw.get("name") or "").strip().lower()).strip("_")
        if name not in BLOCK_NAMES:
            notes.append(f"the model wrote a block called {str(raw.get('name'))[:40]!r}, which "
                         f"is not one of the four this search uses, so it was not used")
            continue
        if name in blocks:
            continue
        terms: list[str] = []
        for term in (raw.get("terms") or [])[:MAX_TERMS]:
            clean = _clean_term(term)
            if clean and clean not in terms:
                terms.append(clean)
        block = _empty_block(name, str(raw.get("why") or ""))
        block["terms"] = terms
        blocks[name] = block
    fallback = template_plan(question, protocol)
    template_blocks = {b["name"]: b for b in fallback["blocks"]}
    replaced: list[str] = []
    for name in BLOCK_NAMES:
        thin = name not in blocks or len(blocks[name]["terms"]) < MIN_TERMS
        if thin and len(template_blocks.get(name, {}).get("terms") or []) >= MIN_TERMS:
            replaced.append(name)
            blocks[name] = dict(template_blocks[name])
            blocks[name]["why"] = ((blocks[name].get("why") or "") + " (from the protocol's own "
                                   "words: the model wrote fewer than two terms for this block)")
            notes.append(f"the model wrote fewer than two terms for the {name!r} block, so the "
                         f"protocol's own words stand in for it")
        elif name not in blocks:
            blocks[name] = _empty_block(name)
    rubric = []
    for raw in ((parsed or {}).get("rubric") or [])[:MAX_RUBRIC]:
        if not isinstance(raw, Mapping) or not str(raw.get("rule") or "").strip():
            continue
        rubric.append({"rule": " ".join(str(raw["rule"]).split())[:300],
                       "kind": str(raw.get("kind") or "unknown"),
                       "abstract_can_fail": bool(raw.get("abstract_can_fail"))})
    if not rubric:
        rubric = list(fallback["rubric"])
        notes.append("the model wrote no rubric, so the protocol's own rules stand in for it")
    return {"prompt_version": PROMPT_VERSION, "source": "model",
            "protocol_seen": protocol is not None, "blocks": [blocks[n] for n in BLOCK_NAMES],
            "rubric": rubric, "strings": {}, "width": {}, "seed_check": [], "notes": notes,
            "replaced": replaced, "cost_usd": 0.0}


_STOPWORDS = STOPWORDS


def template_plan(question: str, protocol: Any = None) -> dict[str, Any]:
    """Blocks made only of words the user wrote — the keyless path.

    `groups` is both arms' labels and synonyms; `phenomenon` the outcome labels and their head
    words; `task` the question's remaining content words. `related_designs` has no source in a
    protocol, so it is empty and the second string is not sent. Crude, free, and — the point —
    made of words the user wrote, so a reader can see exactly why each paper was proposed.
    """
    groups: list[str] = []
    outcomes: list[str] = []
    if protocol is not None:
        for key in ("group_a", "group_b"):
            group = getattr(protocol, key, None)
            if group is None:
                continue
            for term in [getattr(group, "label", "")] + list(getattr(group, "synonyms", None)
                                                              or []):
                clean = _clean_term(term)
                if clean and clean not in groups:
                    groups.append(clean)
        for outcome in getattr(protocol, "outcomes", None) or []:
            label = _clean_term(getattr(outcome, "label", ""))
            if label and label not in outcomes:
                outcomes.append(label)
            for word in term_tokens(label):
                if word not in _STOPWORDS and len(word) > 3 and word not in outcomes:
                    outcomes.append(word)
    quoted = [_clean_term(m) for m in re.findall(r'"([^"]{3,80})"', question or "")]
    taken = set(groups) | set(outcomes) | {w for g in groups + outcomes for w in term_tokens(g)}
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", (question or "").lower())
             if w not in _STOPWORDS and w not in taken]
    task = [q for q in quoted if q and q not in taken] + list(dict.fromkeys(words))
    blocks = [
        {**_empty_block("groups", "the protocol's own labels and synonyms for both groups"),
         "terms": groups[:MAX_TERMS]},
        {**_empty_block("task", "the content words of your question"), "terms": task[:MAX_TERMS]},
        {**_empty_block("phenomenon", "the protocol's outcome labels"),
         "terms": outcomes[:MAX_TERMS]},
        _empty_block("related_designs", "no model was available to name the designs that "
                                        "contain both groups without being about them"),
    ]
    rubric = []
    if groups:
        rubric.append({"rule": "compares the groups the protocol names (or their synonyms)",
                       "kind": "comparison", "abstract_can_fail": False})
    for label in outcomes[:3]:
        rubric.append({"rule": f"reports {label}", "kind": "outcome", "abstract_can_fail": False})
    rubric.append({"rule": "is a primary study with results, not a review, editorial or protocol",
                   "kind": "other", "abstract_can_fail": True})
    plan: dict[str, Any] = {
        "prompt_version": PROMPT_VERSION, "source": "template",
        "protocol_seen": protocol is not None, "blocks": blocks, "rubric": rubric,
        "strings": {}, "width": {}, "seed_check": [], "notes": [], "cost_usd": 0.0}
    if not groups and not outcomes:
        # no protocol: there is no concept to OR, only the sentence's words to require. The
        # words are AND'd — every one required, as the first-generation template did — and
        # the narrower first four make the second string. The plan says so.
        plan["words_required"] = True
        plan["notes"].append("no protocol was given, so the strings require every content word "
                             "of your question rather than a concept block each — name the "
                             "groups and the outcome in a protocol for a wider search")
    return plan


# --------------------------------------------------------------------------------- expansion
def expand_plan(plan: Mapping[str, Any], protocol: Any = None, question: str = "") -> None:
    """Every term's variants, under `expanded`, derived against the words the model and the
    protocol used (the prefix rule's boundary) — the ≤ 3 morphological variants first, then the
    bare head and modifier of a multi-word term (design 03 §1, v3 amendment), which are
    variants like any other: prunable by width control, never protected. Idempotent."""
    protocol_text = _protocol_block(protocol) + " " + str(question or "")
    model_terms = [t for b in plan.get("blocks") or [] for t in b.get("terms") or []]
    known = known_words(model_terms, protocol_text)
    for block in plan.get("blocks") or []:
        terms = list(block.get("terms") or [])
        expanded: dict[str, list[str]] = {}
        seen: set[str] = set(terms)
        for term in terms:
            forms = list(variants(term, known))
            for word in head_words(term):
                if word not in seen and word not in forms:
                    forms.append(word)
            seen.update(forms)
            expanded[term] = forms
        block["expanded"] = expanded


def block_terms(block: Mapping[str, Any]) -> list[str]:
    """The block as rendered: each model term followed by its variants, in the model's order,
    minus anything width control pruned. A pruned term lives under `pruned`, not here; its
    variants stay unless pruned on their own count — each form is measured and judged alone,
    which is what let the v2 verification drop `dynamic` and keep `dynamics`."""
    pruned = {p.get("term") for p in block.get("pruned") or []}
    out: list[str] = []
    for term in block.get("terms") or []:
        if term not in pruned and term not in out:
            out.append(term)
        for variant in (block.get("expanded") or {}).get(term) or []:
            if variant not in pruned and variant not in out:
                out.append(variant)
    return out


def blocks_of_query(plan: Mapping[str, Any], query_id: str) -> list[dict[str, Any]]:
    by_name = {b.get("name"): b for b in plan.get("blocks") or []}
    return [by_name[n] for n in QUERY_IDS.get(query_id, ()) if n in by_name]


def _fit(blocks: Sequence[Mapping[str, Any]], limit: int) -> tuple[str, list[str]]:
    """The AND'd string, dropping VARIANTS (never a model term) from the end until it fits
    `limit`; returns the string and the variants dropped, or ("", dropped) when even the bare
    model terms do not fit."""
    terms = [block_terms(b) for b in blocks]
    model = [set(b.get("terms") or []) for b in blocks]
    dropped: list[str] = []
    text = render_query(terms)
    while len(text) > limit:
        victim = None
        for i in range(len(terms) - 1, -1, -1):
            for j in range(len(terms[i]) - 1, -1, -1):
                if terms[i][j] not in model[i]:
                    victim = (i, j)
                    break
            if victim:
                break
        if victim is None:
            return "", dropped
        dropped.append(terms[victim[0]].pop(victim[1]))
        text = render_query(terms)
    return text, dropped


def render_strings(plan: dict[str, Any]) -> None:
    """Fill `plan["strings"]`, `plan["chars"]`, `plan["trimmed"]` from the blocks as they stand
    (after expansion and pruning). One string per query, the same for PubMed, OpenAlex's
    title-and-abstract filter and Europe PMC; the OpenAlex full-text form only under
    `MAX_STRING_CHARS`, trimmed of trailing variants to get there, else `None` (recorded)."""
    strings: dict[str, dict[str, str | None]] = {}
    chars: dict[str, int] = {}
    trimmed: dict[str, list[str]] = {}
    if plan.get("words_required"):
        # the protocol-less template: the question's words, all required (`blocks.template_plan`)
        words = [t for b in plan.get("blocks") or [] for t in b.get("terms") or []]
        for query_id, take in (("Q1", 8), ("Q2", 4)):
            chosen = words[:take]
            if not chosen or (query_id == "Q2" and len(words) < 4):
                continue
            base = " AND ".join(render_term(w) for w in chosen)
            strings[query_id] = {"pubmed": base, "openalex_ta": base, "europepmc": base,
                                 "openalex_ft": base if len(base) <= MAX_STRING_CHARS else None}
            chars[query_id] = len(base)
        plan["strings"], plan["chars"], plan["trimmed"] = strings, chars, trimmed
        return
    for query_id in QUERY_IDS:
        blocks = blocks_of_query(plan, query_id)
        if any(len(block_terms(b)) < 1 for b in blocks) or len(blocks) < len(QUERY_IDS[query_id]):
            continue
        base = render_query([block_terms(b) for b in blocks])
        full_text, dropped = _fit(blocks, MAX_STRING_CHARS)
        strings[query_id] = {"pubmed": base, "openalex_ta": base, "europepmc": base,
                             "openalex_ft": full_text or None}
        chars[query_id] = len(base)
        if dropped:
            trimmed[query_id] = dropped
    plan["strings"] = strings
    plan["chars"] = chars
    plan["trimmed"] = trimmed


def query_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """`record.queries`: one row per (query, index, form), in the order the indexes are asked.
    A full-text form that does not fit is a row with `skipped: True` and its length, so the
    page can say the string was too long for that index rather than that the index found
    nothing."""
    rows: list[dict[str, Any]] = []
    whys = {"Q1": "the groups the protocol compares, AND the task, AND the phenomenon",
            "Q2": "designs that contain both groups without being about them, AND the task, "
                  "AND the phenomenon"}
    for index, form in INDEX_FORMS:
        for query_id, forms in (plan.get("strings") or {}).items():
            key = f"{index}_{form}" if form else index
            text = forms.get(key)
            row: dict[str, Any] = {"query_id": query_id, "index": index, "form": form,
                                   "text": text or "", "why": whys.get(query_id, ""),
                                   "chars": len(text) if text else int(
                                       (plan.get("chars") or {}).get(query_id) or 0)}
            if not text:
                row["skipped"] = True
                row["why"] = (f"not sent: the string is {row['chars']} characters and OpenAlex's "
                              f"full-text search takes at most {MAX_STRING_CHARS}")
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------------- the call
def build_plan(client: Any, question: str, *, model: str, protocol: Any = None,
               transport: Any = None, seeds: Sequence[str] = (),
               max_tokens: int = 6000) -> dict[str, Any]:
    """One structured call, then expansion, width control and rendering. Falls back to the
    template path — it does not raise — when the call fails, and says so on the plan.

    `transport` is what width control measures with; `None` skips the measuring (the strings
    are rendered unpruned and the plan says width control did not run). `seeds` are recorded
    for step 7 of the design and not yet checked.
    """
    from . import width

    described = _protocol_block(protocol)
    plan: dict[str, Any]
    if client is None:
        plan = template_plan(question, protocol)
    else:
        try:
            result = client.structured(
                model=model, system=SYSTEM, schema=BLOCK_SCHEMA, effort="medium",
                max_tokens=max_tokens, prompt_version=PROMPT_VERSION,
                messages=[{"role": "user",
                           "content": PROMPT.format(QUESTION=question.strip()[:2000],
                                                    PROTOCOL=described)}])
            plan = plan_from_answer(dict(result.parsed or {}), question=question,
                                    protocol=protocol)
            plan["cost_usd"] = round(float(getattr(result, "cost_usd", 0.0) or 0.0), 6)
            usable = [b for b in plan["blocks"] if b["name"] in QUERY_IDS["Q1"]
                      and len(b["terms"]) >= MIN_TERMS]
            if len(usable) < 3 or set(plan.get("replaced") or ()) >= set(QUERY_IDS["Q1"]):
                fallback = template_plan(question, protocol)
                fallback["notes"].append("the model returned no usable blocks, so your own "
                                         "words were used")
                fallback["cost_usd"] = plan["cost_usd"]
                plan = fallback
        except Exception as exc:                        # noqa: BLE001 - reported, not raised
            plan = template_plan(question, protocol)
            plan["notes"].append(f"the query call failed ({type(exc).__name__}), so the blocks "
                                 f"below were built from your own words instead")
    plan["seeds"] = [str(s) for s in seeds if str(s).strip()]
    expand_plan(plan, protocol, question)
    render_strings(plan)
    if transport is not None:
        width.prune(plan, transport, protected=protected_terms(protocol))
        render_strings(plan)
        if plan["seeds"]:
            check_seeds(plan, transport, question=question, protocol=protocol, client=client,
                        model=model)
    else:
        plan["notes"].append("width control did not run: no index was available to measure "
                             "the strings, so they are sent unpruned")
    return plan


# ------------------------------------------------------------------------------------ seeds
SEED_REWRITE_PROMPT = """The search string built from your blocks did not reach these papers, which the researcher says
must be found. For each, the words of its own record are shown. Add to the blocks — never remove —
the terms that would reach them, keeping every term generic to the concept rather than to the paper:

{SEED_RECORDS}

Return the complete blocks and rubric again."""

MAX_SEEDS = 20


def _seed_record(transport: Any, doi: str) -> dict[str, Any] | None:
    """The seed's own Europe PMC record (`resultType=core`), or None when the index has none."""
    from .indices import EuropePmc

    epmc = EuropePmc()
    url, params = epmc.request(f"DOI:{doi}", limit=1)
    try:
        response = transport.get_json(url, params=params, context={
            "index": "europepmc", "form": "seed", "query_id": f"seed:{doi}", "page": 1,
            "query_text": params["query"], "exact": True})
    except Exception:                                   # noqa: BLE001 - offline, or a bug
        return None
    payload = response.json() if response.ok else None
    rows = ((payload.get("resultList") or {}).get("result") or []) if isinstance(payload, dict) \
        else []
    return rows[0] if rows and isinstance(rows[0], dict) else None


def _reaches(transport: Any, *, doi: str, pmid: str, terms: Sequence[str]) -> bool | None:
    """Does this seed's TITLE or ABSTRACT match this OR'd list of terms? PubMed `[uid]` when the
    seed has a PMID (its term mapping is the one the search will use), else Europe PMC fielded.
    Full-text matching would say "reached" for a term that only appears in the body (04 v2
    verification); the strings are answered against titles and abstracts, so the check is too."""
    from .indices import EuropePmc, PubMed
    from .width import _epmc_hits, count

    block = render_block(terms)
    if not block:
        return None
    if pmid:
        pubmed = PubMed()
        url, params = pubmed.request(f"{pmid}[uid] AND {block}", limit=1)

        def read(payload: Any) -> int | None:
            result = payload.get("esearchresult") if isinstance(payload, dict) else None
            return int(result.get("count") or 0) if isinstance(result, dict) else None

        try:
            response = transport.get_json(url, params={}, form_data=params, context={
                "index": "pubmed", "form": "seed", "query_id": f"seed:{doi}", "page": 0,
                "query_text": params["term"], "exact": True})
        except Exception:                               # noqa: BLE001
            return None
        if not response.ok:
            return None
        hits = read(response.json())
        return None if hits is None else hits > 0
    epmc = EuropePmc()
    probe = f"DOI:{doi} AND (TITLE:{block} OR ABSTRACT:{block})"
    hits = count(transport, epmc.count_request(probe), context={
        "index": "europepmc", "form": "seed", "query_id": f"seed:{doi}", "page": 0,
        "query_text": probe, "exact": True}, read=_epmc_hits)
    return None if hits is None else hits > 0


def check_seeds(plan: dict[str, Any], transport: Any, *, question: str, protocol: Any,
                client: Any = None, model: str = "") -> None:
    """Design 03 §1: every seed DOI checked against the pruned strings AT ABSTRACT LEVEL, block
    by block; a seed a block misses restores the block's pruned term the seed's record contains;
    still unreached → one rewrite call; still unreached → injected (`plan["seed_inject"]`, found
    by "seed", never counted as candidate recall). Everything lands in `plan["seed_check"]`."""
    from . import width

    seeds = [normalise_doi_(d) for d in plan.get("seeds") or []][:MAX_SEEDS]
    seeds = [d for d in seeds if d]
    if not seeds:
        plan["seed_check"] = []
        return
    records: dict[str, dict[str, Any]] = {}
    for doi in seeds:
        record = _seed_record(transport, doi)
        if record is not None:
            records[doi] = record

    def walk() -> dict[str, dict[str, Any]]:
        """Per seed: which strings reach it, which blocks miss it."""
        out: dict[str, dict[str, Any]] = {}
        for doi in seeds:
            record = records.get(doi)
            if record is None:
                out[doi] = {"reached_by": [], "missed": {}, "unindexed": True}
                continue
            pmid = str(record.get("pmid") or "").strip()
            reached_by: list[str] = []
            missed: dict[str, list[str]] = {}
            for query_id in plan.get("strings") or {}:
                blocks = blocks_of_query(plan, query_id)
                failing = []
                for block in blocks:
                    hit = _reaches(transport, doi=doi, pmid=pmid, terms=block_terms(block))
                    if hit is False:
                        failing.append(str(block.get("name")))
                if not failing:
                    reached_by.append(query_id)
                else:
                    missed[query_id] = failing
            out[doi] = {"reached_by": reached_by, "missed": missed, "unindexed": False}
        return out

    checked = walk()
    rows: dict[str, dict[str, Any]] = {
        doi: {"doi": doi, "title": _text_of(records.get(doi), "title"),
              "pmid": str((records.get(doi) or {}).get("pmid") or ""),
              "reached_by": list(checked[doi]["reached_by"]),
              "missed": dict(checked[doi]["missed"]), "restored": [], "rewritten": False,
              "injected": False, "unindexed": checked[doi]["unindexed"]}
        for doi in seeds}

    # ---- restore: a pruned term the unreached seed's own record contains
    restored_any = False
    for doi in seeds:
        row = rows[doi]
        if row["reached_by"] or row["unindexed"]:
            continue
        pmid = row["pmid"]
        for block_name in row["missed"].get("Q1") or next(iter(row["missed"].values()), []):
            block = next((b for b in plan.get("blocks") or [] if b.get("name") == block_name), None)
            if block is None:
                continue
            for pruned in list(block.get("pruned") or []):
                term = str(pruned.get("term") or "")
                if term and _reaches(transport, doi=doi, pmid=pmid, terms=[term]):
                    block["pruned"] = [p for p in block["pruned"] if p.get("term") != term]
                    row["restored"].append(term)
                    restored_any = True
                    break
    if restored_any:
        render_strings(plan)
        checked = walk()
        for doi in seeds:
            rows[doi]["reached_by"] = list(checked[doi]["reached_by"])
            rows[doi]["missed"] = dict(checked[doi]["missed"])
            for query_id, measure in (plan.get("width") or {}).items():
                if isinstance(measure, dict):
                    measure["after_seed_restore"] = True

    # ---- one rewrite call, for the seeds still unreached
    unreached = [doi for doi in seeds if not rows[doi]["reached_by"] and not rows[doi]["unindexed"]]
    if unreached and client is not None and model:
        described = _protocol_block(protocol)
        shown = []
        for doi in unreached:
            record = records[doi]
            missed = sorted({b for blocks in rows[doi]["missed"].values() for b in blocks})
            shown.append(f"- {_text_of(record, 'title')} ({_text_of(record, 'pubYear')})\n"
                         f"  {' '.join(str(record.get('abstractText') or '').split())[:1200]}\n"
                         f"  blocks that did not match: {', '.join(missed) or 'none'}")
        try:
            result = client.structured(
                model=model, system=SYSTEM, schema=BLOCK_SCHEMA, effort="medium",
                max_tokens=6000, prompt_version=PROMPT_VERSION + "-seeds",
                messages=[{"role": "user", "content": (
                    PROMPT.format(QUESTION=question.strip()[:2000], PROTOCOL=described)
                    + "\n\n" + SEED_REWRITE_PROMPT.format(SEED_RECORDS="\n".join(shown)))}])
            answer = plan_from_answer(dict(result.parsed or {}), question=question,
                                      protocol=protocol)
            # add-only: every term the rewrite wrote joins its block; nothing is removed
            for block in plan.get("blocks") or []:
                new = next((b for b in answer["blocks"] if b["name"] == block["name"]), None)
                if new is None:
                    continue
                for term in new["terms"]:
                    if term not in block["terms"] and len(block["terms"]) < MAX_TERMS:
                        block["terms"].append(term)
            plan["cost_usd"] = round(float(plan.get("cost_usd") or 0.0)
                                     + float(getattr(result, "cost_usd", 0.0) or 0.0), 6)
            expand_plan(plan, protocol, question)
            render_strings(plan)
            protected = protected_terms(protocol) | frozenset(
                t for r in rows.values() for t in r["restored"])
            width.prune(plan, transport, protected=protected)
            render_strings(plan)
            checked = walk()
            for doi in seeds:
                rows[doi]["reached_by"] = list(checked[doi]["reached_by"])
                rows[doi]["missed"] = dict(checked[doi]["missed"])
                if doi in unreached:
                    rows[doi]["rewritten"] = True
        except Exception as exc:                        # noqa: BLE001 - reported, not raised
            plan.setdefault("notes", []).append(
                f"the seed rewrite call failed ({type(exc).__name__}); the unreached seeds "
                f"are injected instead")

    # ---- inject what is still unreached
    inject: list[dict[str, Any]] = []
    for doi in seeds:
        row = rows[doi]
        if not row["reached_by"] and not row["unindexed"]:
            row["injected"] = True
            inject.append(records[doi])
    plan["seed_inject"] = inject
    plan["seed_check"] = [rows[doi] for doi in seeds]
    n_reached = sum(1 for r in plan["seed_check"] if r["reached_by"])
    plan.setdefault("notes", []).append(
        f"seed check: {n_reached} of {len(seeds)} seed(s) reached by a string at abstract level"
        + (f"; restored {sum(len(r['restored']) for r in rows.values())} pruned term(s)"
           if restored_any else "")
        + (f"; {len(inject)} injected as candidates found by \"seed\"" if inject else "")
        + (f"; {sum(1 for r in rows.values() if r['unindexed'])} not on Europe PMC"
           if any(r["unindexed"] for r in rows.values()) else ""))


def _text_of(record: Mapping[str, Any] | None, key: str) -> str:
    return " ".join(str((record or {}).get(key) or "").split())[:300]


def normalise_doi_(value: str) -> str:
    from .dedupe import normalise_doi

    return normalise_doi(str(value or ""))
