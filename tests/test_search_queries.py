"""Query building v2: concept blocks, the expander, rendering, width control, ranking.

No network, no model: the block call is scripted, the hit counts width control measures are
recorded at the exact keys the adapters ask for, and the keyless path is the protocol's own words.
What is pinned is the design's arithmetic (03 §1–3) and the three rules the adversarial review
folded in (04 v2 verification): the prefix boundary, the protocol-term protection, and the
variant-over-model-term preference.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from canopy.llm.schemas import assert_no_derived_stats, assert_valid_output_schema
from canopy.protocol import load_protocol
from canopy.search import width as width_module
from canopy.search.blocks import (BLOCK_SCHEMA, MAX_TERMS, PROMPT, block_terms, build_plan,
                                  expand_plan, plan_from_answer, protected_terms, query_rows,
                                  render_strings, template_plan)
from canopy.search.expand import (MAX_STRING_CHARS, MAX_VARIANTS, known_words, render_query,
                                  render_term, variants)
from canopy.search.indices import EuropePmc, OpenAlex
from canopy.search.models import Candidate
from canopy.search.rank import coverage, rank, score
from canopy.search.transport import HttpResponse, RecordedTransport
from tests.test_server import PROTOCOL as PROTOCOL_PATH

QUESTION = "Does ageing change sensorimotor adaptation to a visual perturbation?"

#: a scripted block answer with nothing in it that names a field: the words are the example
#: protocol's own labels and a few made-up task words
ANSWER = {
    "blocks": [
        {"name": "groups", "why": "how papers name the two age groups",
         "terms": ["older adults", "younger adults", "elderly", "young", "age-related"]},
        {"name": "task", "why": "the perturbation every eligible study used",
         "terms": ["visual perturbation", "rotation", "cursor rotation", "mirror reversal"]},
        {"name": "phenomenon", "why": "the outcome family",
         "terms": ["adaptation", "aftereffect", "recalibration"]},
        {"name": "related_designs", "why": "designs holding both groups without being about them",
         "terms": ["age-matched controls", "control group"]},
    ],
    "rubric": [
        {"rule": "participants are neurologically healthy adults", "kind": "population",
         "abstract_can_fail": True},
        {"rule": "reports an adaptation outcome", "kind": "outcome", "abstract_can_fail": False},
    ],
}


def _client(parsed, cost=0.0025):
    return SimpleNamespace(structured=lambda **kw: SimpleNamespace(parsed=parsed, cost_usd=cost))


@pytest.fixture
def protocol():
    return load_protocol(str(PROTOCOL_PATH))


# ------------------------------------------------------------------------------------ the schema
def test_the_block_schema_passes_both_audits():
    """The audit `client.structured` runs on the way to the wire — the search's own version of
    the blocker that made screening impossible before it was ever called."""
    assert_valid_output_schema(BLOCK_SCHEMA, "BLOCK_SCHEMA")
    assert_no_derived_stats(BLOCK_SCHEMA)
    assert BLOCK_SCHEMA["properties"]["blocks"]["items"]["properties"]["terms"]["maxItems"] == \
        MAX_TERMS == 20


def test_the_prompt_tells_the_model_the_clamp_and_marks_its_examples_as_examples():
    assert "AT MOST 20 terms" in PROMPT
    assert "depending on the field" in PROMPT and "placebo, sham or wait-list" in PROMPT
    assert "Do not name any specific paper, author, laboratory or journal" in PROMPT


# ------------------------------------------------------------------------------- the expander
def test_the_expander_table():
    """The forms the design lists, and the non-words the review found — never again."""
    known = known_words(["aftereffect", "effect", "adaptation", "manual"])
    assert variants("non-dominant")[:2] == ["non dominant", "nondominant"]
    assert "after-effect" in variants("aftereffect", known)
    assert variants("aftereffect") == ["aftereffects"], "no split without a known remainder"
    assert "co-mpensation" not in variants("compensation", known_words(["mpensation"] and []))
    assert "contra-st" not in variants("contrast", known) and "pre-ferred" not in variants(
        "preferred", known)
    assert "adaptate" not in variants("adaptation") and "adaptations" in variants("adaptation")
    assert "fource field" not in variants("force field")
    assert "adultss" not in variants("older adults") and "adultss" not in variants("adults")
    assert variants("adults") == ["adult"], "the singular of a plural head, and no plural of it"
    assert variants("dynamics") == ["dynamic"]
    assert variants("behaviour") == ["behavior", "behaviours"]
    assert variants("lateralisation")[0] == "lateralization"
    assert variants("dominant") == ["dominance", "dominants"]
    assert all(len(variants(t)) <= MAX_VARIANTS for t in ("force field", "visuo-motor",
                                                          "hand dominance", "older adults"))
    for term in ("x", "rotation", "force field", "non-dominant"):
        assert term not in variants(term), "never the term itself"
        assert variants(term) == variants(term), "order-stable"


def test_rendering_quotes_spaces_and_hyphens_and_nothing_else():
    """Europe PMC splits a bare hyphenated token into two AND'd words: `after-effect` bare was
    7.4 M records, quoted 4,360 (04 BLOCKER-2a)."""
    assert render_term("aftereffect") == "aftereffect"
    assert render_term("after-effect") == '"after-effect"'
    assert render_term("force field") == '"force field"'
    assert render_term('say "hi"') == '"say hi"'
    assert render_query([["a", "b-c"], ["d e"], []]) == '(a OR "b-c") AND ("d e")'


# ------------------------------------------------------------------------------- the clamp
def test_a_model_answer_becomes_a_plan_with_every_term_kept_and_every_variant_listed(protocol):
    plan = plan_from_answer(ANSWER, question=QUESTION, protocol=protocol)
    assert plan["source"] == "model" and plan["protocol_seen"] is True
    names = [b["name"] for b in plan["blocks"]]
    assert names == ["groups", "task", "phenomenon", "related_designs"]
    assert plan["blocks"][0]["terms"] == ["older adults", "younger adults", "elderly", "young",
                                          "age-related"]
    expand_plan(plan, protocol, QUESTION)
    groups = plan["blocks"][0]
    assert set(groups["expanded"]) == set(groups["terms"]), "every term has its entry"
    assert groups["expanded"]["age-related"] == ["age related", "agerelated"]
    assert groups["expanded"]["older adults"] == ["older-adults", "olderadults", "older adult",
                                                  "adults", "older"], "…then the bare words"
    assert block_terms(groups)[:3] == ["older adults", "older-adults", "olderadults"]
    assert [r["rule"] for r in plan["rubric"]] == ANSWER["rubric"][0]["rule"].split("|") + [
        ANSWER["rubric"][1]["rule"]]


def test_the_clamp_bounds_terms_per_block_and_never_drops_a_model_term_for_a_variant(protocol):
    many = dict(ANSWER)
    many["blocks"] = [dict(ANSWER["blocks"][0], terms=[f"term {i}" for i in range(30)])] + \
        ANSWER["blocks"][1:]
    plan = plan_from_answer(many, question=QUESTION, protocol=protocol)
    assert len(plan["blocks"][0]["terms"]) == MAX_TERMS
    expand_plan(plan, protocol, QUESTION)
    assert all(t in block_terms(plan["blocks"][0]) for t in plan["blocks"][0]["terms"])


def test_an_unknown_block_is_dropped_with_a_note_and_a_thin_block_falls_back_to_the_protocol(
        protocol):
    odd = {"blocks": [{"name": "Groups", "why": "", "terms": ["only-one"]},
                      {"name": "outcome", "why": "", "terms": ["a", "b"]},
                      {"name": "task", "why": "", "terms": ["rotation", "prism"]},
                      {"name": "phenomenon", "why": "", "terms": ["adaptation", "aftereffect"]}],
           "rubric": []}
    plan = plan_from_answer(odd, question=QUESTION, protocol=protocol)
    groups = plan["blocks"][0]
    assert "older" in " ".join(groups["terms"]).lower() and "younger" in " ".join(
        groups["terms"]).lower(), "the protocol's own labels stand in"
    assert any("fewer than two terms" in note for note in plan["notes"])
    assert any("'outcome'" in note for note in plan["notes"])
    assert plan["rubric"] and any("primary study" in r["rule"] for r in plan["rubric"])


# ------------------------------------------------------------------------------ the strings
def test_two_strings_are_rendered_and_the_full_text_form_is_trimmed_or_skipped(protocol):
    plan = plan_from_answer(ANSWER, question=QUESTION, protocol=protocol)
    expand_plan(plan, protocol, QUESTION)
    render_strings(plan)
    assert set(plan["strings"]) == {"Q1", "Q2"}
    q1 = plan["strings"]["Q1"]
    assert q1["pubmed"] == q1["openalex_ta"] == q1["europepmc"]
    assert q1["pubmed"].startswith('("older adults" OR "older-adults" OR olderadults')
    assert q1["pubmed"].count(" AND ") == 2 and plan["strings"]["Q2"]["pubmed"].count(" AND ") == 2
    assert q1["openalex_ft"] == q1["pubmed"], "short enough to send whole"
    rows = query_rows(plan)
    assert [(r["index"], r["form"], r["query_id"]) for r in rows] == [
        ("pubmed", "", "Q1"), ("pubmed", "", "Q2"), ("openalex", "ta", "Q1"),
        ("openalex", "ta", "Q2"), ("europepmc", "", "Q1"), ("europepmc", "", "Q2"),
        ("openalex", "ft", "Q1"), ("openalex", "ft", "Q2")]
    assert all(r["chars"] == len(r["text"]) and r["why"] for r in rows)

    # a wide block: the full-text form loses variants from the end until it fits, never a
    # model term, and is skipped (recorded) when even the model's own terms do not fit
    wide = dict(ANSWER)
    wide["blocks"] = [dict(ANSWER["blocks"][0],
                           terms=[f"a rather long group phrase {i}" for i in range(20)])] + \
        ANSWER["blocks"][1:]
    plan = plan_from_answer(wide, question=QUESTION, protocol=protocol)
    expand_plan(plan, protocol, QUESTION)
    render_strings(plan)
    assert len(plan["strings"]["Q1"]["pubmed"]) > MAX_STRING_CHARS
    full = plan["strings"]["Q1"]["openalex_ft"]
    assert full and len(full) <= MAX_STRING_CHARS
    assert all(f'"a rather long group phrase {i}"' in full for i in range(20)), "model terms stay"
    model_terms = {t for b in plan["blocks"] for t in b["terms"]}
    assert plan["trimmed"]["Q1"] and not (set(plan["trimmed"]["Q1"]) & model_terms)
    huge = dict(wide)
    huge["blocks"] = [dict(wide["blocks"][0],
                           terms=[f"an even longer group phrase written out in full {i}"
                                  for i in range(20)]),
                      dict(ANSWER["blocks"][1],
                           terms=[f"an even longer task phrase written out in full {i}"
                                  for i in range(20)])] + ANSWER["blocks"][2:]
    plan = plan_from_answer(huge, question=QUESTION, protocol=protocol)
    expand_plan(plan, protocol, QUESTION)
    render_strings(plan)
    assert plan["strings"]["Q1"]["openalex_ft"] is None
    skipped = [r for r in query_rows(plan) if r.get("skipped")]
    assert skipped and skipped[0]["chars"] > MAX_STRING_CHARS and "not sent" in skipped[0]["why"]


def test_the_keyless_plan_is_made_of_the_users_own_words(protocol):
    """Both arms of the comparison, the outcome labels, the question's content words — and no
    second string, because nothing in a protocol names the related designs."""
    plan = template_plan(QUESTION, protocol)
    assert plan["source"] == "template"
    groups = plan["blocks"][0]["terms"]
    assert any("older" in t for t in groups) and any("younger" in t for t in groups)
    assert plan["blocks"][2]["terms"][0] == "late adaptation"
    assert "adaptation" in plan["blocks"][2]["terms"]
    assert plan["blocks"][3]["terms"] == []
    expand_plan(plan, protocol, QUESTION)
    render_strings(plan)
    assert list(plan["strings"]) == ["Q1"]
    assert plan["strings"]["Q1"]["pubmed"].count(" AND ") == 2
    assert all(r["rule"] for r in plan["rubric"])
    assert template_plan('Does "cognitive behavioural therapy" reduce anxiety?')["blocks"][1][
        "terms"][0] == "cognitive behavioural therapy"


def test_a_failed_call_falls_back_to_the_users_own_words_and_says_so(protocol):
    def boom(**kw):
        raise RuntimeError("upstream 503")

    plan = build_plan(SimpleNamespace(structured=boom), QUESTION, model="m", protocol=protocol)
    assert plan["source"] == "template" and plan["strings"]
    assert any("failed" in note for note in plan["notes"])
    assert any("width control did not run" in note for note in plan["notes"])


def test_a_model_answer_with_no_usable_block_is_the_same_as_no_answer(protocol):
    plan = build_plan(_client({"blocks": [], "rubric": []}), QUESTION, model="m",
                      protocol=protocol)
    assert plan["source"] == "template" and any("usable" in n for n in plan["notes"])


# ------------------------------------------------------------------------------ width control
def counting_transport(hits: dict[str, int], oa_hits: int | None = 5) -> RecordedTransport:
    """Europe PMC answering `hitCount` for each rendered string in `hits`, and OpenAlex's
    title-and-abstract count; anything else unmeasured (a loud miss under the fake, which width
    control turns into an unmeasured term)."""
    transport = RecordedTransport()
    for text, n in hits.items():
        url, params = EuropePmc().count_request(text)
        transport.record(url, HttpResponse(url=url, status=200, outcome="ok",
                                           body=json.dumps({"hitCount": n}).encode()), params)
    if oa_hits is not None:
        transport.oa_hits = oa_hits                                  # type: ignore[attr-defined]
    return transport


class OaCounting(RecordedTransport):
    """OpenAlex's count for ANY string: one number, so the test is about Europe PMC's pruning."""

    def __init__(self, hits, oa_hits=5):
        super().__init__()
        self.oa_hits = oa_hits
        for text, n in hits.items():
            url, params = EuropePmc().count_request(text)
            self.record(url, HttpResponse(url=url, status=200, outcome="ok",
                                          body=json.dumps({"hitCount": n}).encode()), params)

    def get_json(self, url, *, params=None, headers=None, timeout=20.0,
                 accept="application/json", max_bytes=0, context=None):
        if url == OpenAlex.URL:
            self.calls.append({"method": "get_json", "url": url, "params": dict(params or {}),
                               "key": "oa", "context": dict(context or {})})
            return HttpResponse(url=url, status=200, outcome="ok",
                                body=json.dumps({"meta": {"count": self.oa_hits}}).encode())
        return super().get_json(url, params=params, headers=headers, timeout=timeout,
                                accept=accept, max_bytes=max_bytes, context=context)


def small_plan(protocol):
    """`older adults` is a protocol label (protected); `grownups` is the model's own word."""
    answer = {"blocks": [
        {"name": "groups", "why": "", "terms": ["older adults", "grownups"]},
        {"name": "task", "why": "", "terms": ["rotation", "prism"]},
        {"name": "phenomenon", "why": "", "terms": ["adaptation", "aftereffect"]},
        {"name": "related_designs", "why": "", "terms": []}], "rubric": []}
    plan = plan_from_answer(answer, question=QUESTION, protocol=protocol)
    expand_plan(plan, protocol, QUESTION)
    render_strings(plan)
    return plan


def test_width_control_prunes_the_widest_block_records_every_drop_and_spares_the_protocol(
        protocol, monkeypatch):
    """`grownups` is the widest unprotected term in the widest block and goes first, with its
    count on the record; `older adults` is a protocol label and can never go, however wide."""
    monkeypatch.setattr(width_module, "EPMC_MAX_HITS", 100)
    plan = small_plan(protocol)
    groups, task, phenomenon = plan["blocks"][:3]
    per_term = {"older adults": 900, "older-adults": 5, "olderadults": 1, "older adult": 30,
                "adults": 100, "older": 50,                # under half of `grownups`: it stays
                "grownups": 400, "grownup": 2, "rotation": 300, "rotations": 40,
                "prism": 50, "prisms": 3, "adaptation": 700, "adaptations": 60,
                "aftereffect": 100, "aftereffects": 10}
    hits = {render_term(t): n for t, n in per_term.items()}

    def string_of(*blocks):
        return render_query([block_terms(b) for b in blocks])

    def block_of(b):
        return "(" + " OR ".join(render_term(t) for t in block_terms(b)) + ")"

    # the AND'd string: 250 before the first drop, 90 after `elderly` goes
    hits[string_of(groups, task, phenomenon)] = 250
    hits[block_of(groups)] = 950
    hits[block_of(task)] = 320
    hits[block_of(phenomenon)] = 700
    transport = OaCounting(hits)
    # the string after the drop is asked at its new text: record that answer too
    import copy

    dropped = copy.deepcopy(groups)
    dropped["pruned"] = [{"term": "grownups"}]
    after = render_query([block_terms(b) for b in (dropped, task, phenomenon)])
    url, params = EuropePmc().count_request(after)
    transport.record(url, HttpResponse(url=url, status=200, outcome="ok",
                                       body=json.dumps({"hitCount": 90}).encode()), params)
    width_module.prune(plan, transport, protected=protected_terms(protocol))
    assert groups["pruned"] and groups["pruned"][0]["term"] == "grownups"
    assert groups["pruned"][0]["hits"] == 400 and groups["pruned"][0]["kind"] == "model"
    assert groups["pruned"][0]["query"] == "Q1" and groups["pruned"][0]["iteration"] == 1
    assert "grownups" not in block_terms(groups) and "older adults" in block_terms(groups)
    assert plan["width"]["Q1"] == {"epmc_hits_before": 250, "epmc_hits_after": 90,
                                   "oa_ta_hits_after": 5, "iterations": 1, "oa_checks": 0}


def test_width_control_prefers_a_variant_within_twice_the_widest_model_term(protocol,
                                                                            monkeypatch):
    """04 v2 verification: the stress case lost a key paper to the model's `dynamics` when the
    expander's `dynamic` would have narrowed the string almost as much."""
    monkeypatch.setattr(width_module, "EPMC_MAX_HITS", 100)
    monkeypatch.setattr(width_module, "MAX_ITERATIONS", 1)
    plan = small_plan(protocol)
    groups, task, phenomenon = plan["blocks"][:3]
    per_term = {"older adults": 50, "older-adults": 5, "olderadults": 1, "older adult": 30,
                "adults": 20, "older": 15,
                "grownups": 60, "grownup": 2, "rotation": 500, "rotations": 300,
                "prism": 50, "prisms": 3, "adaptation": 70, "adaptations": 60,
                "aftereffect": 10, "aftereffects": 1}
    hits = {render_term(t): n for t, n in per_term.items()}
    hits[render_query([block_terms(b) for b in (groups, task, phenomenon)])] = 250
    hits["(" + " OR ".join(render_term(t) for t in block_terms(groups)) + ")"] = 80
    hits["(" + " OR ".join(render_term(t) for t in block_terms(task)) + ")"] = 800
    hits["(" + " OR ".join(render_term(t) for t in block_terms(phenomenon)) + ")"] = 100
    width_module.prune(plan, OaCounting(hits), protected=protected_terms(protocol))
    assert task["pruned"][0]["term"] == "rotations", "the variant, at 300 against 500"
    assert task["pruned"][0]["kind"] == "variant"
    assert "rotation" in block_terms(task)


def test_width_control_stops_rather_than_drop_a_protocol_word_and_leaves_unmeasured_terms(
        protocol, monkeypatch):
    monkeypatch.setattr(width_module, "EPMC_MAX_HITS", 10)
    plan = small_plan(protocol)
    groups = plan["blocks"][0]
    groups["terms"] = ["older adults"]
    groups["expanded"] = {"older adults": []}
    plan["blocks"][1]["terms"] = ["rotation"]
    plan["blocks"][1]["expanded"] = {"rotation": []}
    plan["blocks"][2]["terms"] = ["adaptation"]
    plan["blocks"][2]["expanded"] = {"adaptation": []}
    hits = {'"older adults"': 5000, "rotation": 20, "adaptation": 30,
            '("older adults") AND (rotation) AND (adaptation)': 900,
            '("older adults")': 5000, "(rotation)": 20, "(adaptation)": 30}
    width_module.prune(plan, OaCounting(hits), protected=protected_terms(protocol))
    assert groups["pruned"] == []
    assert any("protocol's own words" in note for note in plan["notes"])
    assert plan["width"]["Q1"]["epmc_hits_before"] == 900
    # a term nobody could count is never a victim and is named
    plan = small_plan(protocol)
    width_module.prune(plan, OaCounting({}), protected=protected_terms(protocol))
    assert plan["width"]["unmeasured"] and any("could not be measured" in n
                                               for n in plan["notes"])
    assert all(b["pruned"] == [] for b in plan["blocks"])


# ------------------------------------------------------------------------------------ ranking
def test_coverage_is_literal_whole_word_and_hyphen_blind():
    assert coverage("Non-dominant arm adaptation", ["non dominant"])
    assert coverage("the NONDOMINANT arm", ["nondominant"])
    assert not coverage("predominant", ["dominant"]), "whole words only"
    assert not coverage("", ["x"]) and not coverage("x", [])


def test_the_score_is_made_of_the_record_and_the_order_is_total(protocol):
    plan = small_plan(protocol)
    covered = Candidate(key="c000000000002", title="Prism adaptation in older adults",
                        abstract="Older adults adapted to prisms.", n_rows=3,
                        ranks={"pubmed:Q1:": 12, "openalex:Q1:ta": 40})
    bare = Candidate(key="c000000000001", title="A paper about something else", n_rows=1,
                     ranks={"europepmc:Q1:": 900})
    no_abstract = Candidate(key="c000000000003", title="Rotation adaptation in the elderly",
                            n_rows=1, ranks={"pubmed:Q1:": 12})
    total, parts = score(covered, plan)
    assert parts["body_cov"] == 1.0 and parts["ttl_cov"] == 1.0 and parts["best_position"] == 12
    assert total > score(no_abstract, plan)[0] > score(bare, plan)[0]
    ordered = rank([bare, covered, no_abstract], plan)
    assert [c.key for c in ordered] == ["c000000000002", "c000000000003", "c000000000001"]
    assert ordered == rank([no_abstract, bare, covered], plan), "order-independent"
    assert all(c.relevance is not None and c.relevance_why for c in ordered)
    assert "3 block(s)" in covered.relevance_why and "no abstract" in bare.relevance_why


# ---------------------------------------------------------------- head words (03 §1, v3 amendment)
def test_a_multi_word_term_also_yields_its_bare_head_and_modifier():
    """Step 3 measured 19/23 with blocks made of phrases: `motor adaptation` never reaches a
    paper that says only `adaptation`. The bare words are variants — measured, prunable."""
    from canopy.search.expand import head_words

    assert head_words("motor adaptation") == ["adaptation", "motor"]
    assert head_words("reaching task") == ["reaching"], "`task` has four letters"
    assert head_words("other arm") == [], "a stopword and a three-letter word"
    assert head_words("adaptation") == [], "already bare"
    assert head_words("non-linear") == [], "a hyphenated compound is one word"
    assert head_words("force field adaptation") == ["adaptation", "force"]


def test_head_words_are_recorded_as_variants_never_duplicated_and_prunable(protocol,
                                                                            monkeypatch):
    answer = {"blocks": [
        {"name": "groups", "why": "", "terms": ["older adults", "grownups"]},
        {"name": "task", "why": "", "terms": ["cursor rotation", "rotation", "prism"]},
        {"name": "phenomenon", "why": "", "terms": ["motor adaptation", "aftereffect"]},
        {"name": "related_designs", "why": "", "terms": []}], "rubric": []}
    plan = plan_from_answer(answer, question=QUESTION, protocol=protocol)
    expand_plan(plan, protocol, QUESTION)
    task, phenomenon = plan["blocks"][1], plan["blocks"][2]
    # `rotation` is already a model term in the block: not added again as a head word
    assert task["expanded"]["cursor rotation"] == ["cursor-rotation", "cursorrotation",
                                                   "cursor rotations", "cursor"]
    assert block_terms(task).count("rotation") == 1
    assert phenomenon["expanded"]["motor adaptation"][-2:] == ["adaptation", "motor"]
    assert "adaptation" in block_terms(phenomenon) and "motor" in block_terms(phenomenon)

    # a head word is a variant: width control may drop it, and the model term stays
    monkeypatch.setattr(width_module, "EPMC_MAX_HITS", 10)
    monkeypatch.setattr(width_module, "MAX_ITERATIONS", 1)
    per_term = {t: 5 for b in plan["blocks"] for t in block_terms(b)}
    per_term.update({"motor": 900, "motor adaptation": 100, "adaptation": 400})
    hits = {render_term(t): n for t, n in per_term.items()}
    hits[render_query([block_terms(b) for b in plan["blocks"][:3]])] = 500
    for b in plan["blocks"][:3]:
        hits["(" + " OR ".join(render_term(t) for t in block_terms(b)) + ")"] = \
            900 if b["name"] == "phenomenon" else 10
    width_module.prune(plan, OaCounting(hits), protected=protected_terms(protocol))
    assert phenomenon["pruned"][0]["term"] == "motor"
    assert phenomenon["pruned"][0]["kind"] == "variant" and phenomenon["pruned"][0]["hits"] == 900
    assert "motor adaptation" in block_terms(phenomenon) and "motor" not in block_terms(
        phenomenon)


# ------------------------------------------------------------------- seeds (03 §1, step 7)
def test_the_seed_check_is_abstract_level_restores_a_pruned_term_and_injects_the_rest(protocol):
    """A seed the pruned string misses on one block gets that block's pruned term back when its
    own record contains it; a seed nothing reaches is injected, found by "seed"."""
    from canopy.search.blocks import check_seeds
    from canopy.search.expand import render_block

    plan = small_plan(protocol)
    groups, task, phenomenon = plan["blocks"][:3]
    # width control had pruned `grownups`; the first seed's record says it
    groups["pruned"] = [{"term": "grownups", "hits": 400, "kind": "model"}]
    render_strings(plan)
    plan["seeds"] = ["10.1000/seed1", "https://doi.org/10.1000/seed2", "10.1000/unindexed"]
    transport = RecordedTransport()
    epmc = EuropePmc()

    def record_seed(doi, row):
        url, params = epmc.request(f"DOI:{doi}", limit=1)
        transport.record(url, HttpResponse(url=url, status=200, outcome="ok", body=json.dumps(
            {"resultList": {"result": [row] if row else []}}).encode()), params)

    record_seed("10.1000/seed1", {"id": "1", "source": "MED", "doi": "10.1000/seed1",
                                  "title": "Grownups adapt to rotation", "abstractText": "x"})
    record_seed("10.1000/seed2", {"id": "2", "source": "MED", "doi": "10.1000/seed2",
                                  "title": "Nothing here", "abstractText": "y"})
    record_seed("10.1000/unindexed", None)

    def reach(doi, terms, hits):
        block = render_block(terms)
        probe = f"DOI:{doi} AND (TITLE:{block} OR ABSTRACT:{block})"
        url, params = epmc.count_request(probe)
        transport.record(url, HttpResponse(url=url, status=200, outcome="ok",
                                           body=json.dumps({"hitCount": hits}).encode()), params)

    # seed 1: groups misses (grownups pruned), task and phenomenon hit; after the restore the
    # groups block (now with grownups) hits
    reach("10.1000/seed1", block_terms(groups), 0)
    reach("10.1000/seed1", block_terms(task), 1)
    reach("10.1000/seed1", block_terms(phenomenon), 1)
    reach("10.1000/seed1", ["grownups"], 1)
    restored = dict(groups, pruned=[])
    reach("10.1000/seed1", block_terms(restored), 1)
    # seed 2: nothing reaches it, before or after
    for terms in (block_terms(groups), block_terms(restored), block_terms(task),
                  block_terms(phenomenon)):
        reach("10.1000/seed2", terms, 0)
    reach("10.1000/seed2", ["grownups"], 0)

    check_seeds(plan, transport, question=QUESTION, protocol=protocol, client=None)

    rows = {r["doi"]: r for r in plan["seed_check"]}
    assert rows["10.1000/seed1"]["restored"] == ["grownups"]
    assert rows["10.1000/seed1"]["reached_by"] == ["Q1"] and not rows["10.1000/seed1"]["injected"]
    assert groups["pruned"] == [] and "grownups" in block_terms(groups)
    assert "grownups" in plan["strings"]["Q1"]["pubmed"], "the string was re-rendered"
    assert rows["10.1000/seed2"]["reached_by"] == [] and rows["10.1000/seed2"]["injected"]
    assert rows["10.1000/unindexed"]["unindexed"] and not rows["10.1000/unindexed"]["injected"]
    assert [r["doi"] for r in plan["seed_inject"]] == ["10.1000/seed2"]
    assert any("seed check: 1 of 3" in n for n in plan["notes"])


def test_predict_is_the_design_arithmetic():
    from canopy.search.cost import predict

    p = predict(5.0, max_fetch_unsure=100, per_paper=7.0)
    assert (p["cap"], p["screen_usd"]) == (1166, 3.5)
    assert (p["snowball_records"], p["snowball_usd"]) == (333, 1.0)
    assert p["worst_case_usd"] == 5.18 and p["audit_usd"] == 0.35
    assert p["run_commit"] == {"n_read": 100, "n_wanted": 0, "n_unsure": 100, "per_paper_usd": 7.0,
                               "usd": 700.0, "unsure_usd": 700.0}
    assert predict(5.0, max_fetch_unsure=100, expected_unsure=40, per_paper=7.0)["run_commit"][
        "usd"] == 280.0
    assert predict(None)["cap"] is None and predict(2.0, snowball=False)["snowball_records"] == 0
