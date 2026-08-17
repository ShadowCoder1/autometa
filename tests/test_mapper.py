"""Task 4: the study mapper.

Two halves:
  * a replayed end-to-end map of Bock 2005 (fixtures in `tests/fixtures/llm`, recorded once live);
  * offline unit tests with `FakeProvider` for everything the mapper decides in *code* — roster
    completeness, cross-check diffing, adjudication, needs_human rules, id assignment.
"""
from __future__ import annotations

import pytest

from canopy.agents import load_prompt
from canopy.agents.mapper import (MAPPER_SCHEMA, MAPPER_SOURCES_SCHEMA, PROMPT_VERSION,
                                  map_study, protocol_text, roster_text)
from canopy.config import MODELS
from canopy.llm.client import LLMClient
from canopy.llm.errors import RefusalError
from canopy.llm.providers import FakeProvider
from canopy.llm.schemas import assert_no_derived_stats, assert_valid_output_schema
from canopy.models import DatasetSpec, DispersionType, Source, SourceKind, StudyMap

#: every Bock test replays fixtures; `@pytest.mark.replay` is what keeps `CANOPY_LIVE`/
#: `CANOPY_RECORD` visible during a recording run. `paper`, `protocol`, `client` and `bock_map`
#: are session fixtures shared by every agent test module (see tests/conftest.py).
replayed = pytest.mark.replay


# ------------------------------------------------------------------ schema hygiene
def test_schemas_are_valid_and_carry_no_derived_stats():
    from canopy.agents import mapper

    for name in ("MAPPER_SCHEMA", "MAPPER_SOURCES_SCHEMA", "MAPPER_CROSSCHECK_SCHEMA",
                 "MAPPER_ADJUDICATE_SCHEMA", "MAPPER_ROSTER_SCHEMA"):
        schema = getattr(mapper, name)
        assert_valid_output_schema(schema, name)
        assert_no_derived_stats(schema, name=name)


def test_mapper_schema_mirrors_studymap_without_bookkeeping():
    props = set(MAPPER_SCHEMA["properties"])
    assert props <= set(StudyMap.model_fields)
    assert not props & {"paper_id", "model", "prompt_version", "llm_call_ids", "disagreements",
                        "needs_human"}
    assert {"citation", "eligible", "datasets", "roster"} <= props
    dataset = MAPPER_SCHEMA["properties"]["datasets"]["items"]["properties"]
    assert set(dataset) <= set(DatasetSpec.model_fields) | {"moderators"}
    assert "outcomes" not in dataset                     # the second pass locates those
    source = (MAPPER_SOURCES_SCHEMA["properties"]["datasets"]["items"]["properties"]["outcomes"]
              ["items"]["properties"]["sources"]["items"]["properties"])
    assert set(source) <= set(Source.model_fields)


def test_prompts_carry_no_domain_knowledge():
    """The protocol supplies the domain; the prompts must work for any review."""
    banned = ["older adult", "younger adult", "visuomotor", "adaptation", "aftereffect",
              "ageing", "aging", "elderly", "perturbation", "sensorimotor"]
    for name in ("mapper", "mapper_sources", "mapper_crosscheck", "mapper_adjudicate",
                 "mapper_roster"):
        text = load_prompt(name).lower()
        assert not [word for word in banned if word in text], name


def test_protocol_text_renders_the_protocol(protocol):
    text = protocol_text(protocol)
    assert protocol.title in text
    for outcome in protocol.outcomes:
        assert outcome.key in text and outcome.definition.strip()[:40] in text
    assert protocol.group_a.label in text and protocol.group_b.label in text
    for rule in protocol.eligibility + protocol.dataset_rules:
        assert rule.strip()[:40] in text
    assert protocol.stats.multi_group_policy in text
    assert "cisneros2024" not in text                     # stats internals stay out of prompts


def test_roster_text_lists_every_ingested_figure_and_table(paper):
    text = roster_text(paper)
    for fig in paper.figures:
        assert fig.id in text and f"page {fig.page}" in text
    for table in paper.tables:
        assert table.id in text
    assert text == roster_text(paper)                     # deterministic (fixture keys depend on it)


# ------------------------------------------------------------------ replayed Bock 2005 map
@replayed
def test_bock_is_eligible(bock_map):
    assert bock_map.eligible is True
    assert bock_map.eligibility_rationale.strip()
    assert not bock_map.exclusion_reason


@replayed
def test_bock_has_a_dataset_with_twelve_per_group(bock_map, paper):
    sizes = [(d.group_a.n, d.group_b.n) for d in bock_map.datasets]
    assert (12, 12) in sizes, sizes
    dataset = next(d for d in bock_map.datasets if (d.group_a.n, d.group_b.n) == (12, 12))
    assert dataset.group_a.n_evidence and dataset.group_b.n_evidence
    assert dataset.dataset_id == f"{paper.sha256[:12]}:d{bock_map.datasets.index(dataset) + 1}"
    assert dataset.cluster_id == paper.sha256[:12]
    assert len(dataset.all_groups_listed) >= 2


@replayed
def test_bock_late_adaptation_has_a_figure_source_on_page_three_with_sd_error_bars(bock_map):
    figures = [s
               for d in bock_map.datasets
               for o in d.outcomes if o.outcome_key == "late_adaptation"
               for s in o.sources
               if s.kind in (SourceKind.figure_line, SourceKind.figure_bar, SourceKind.figure_points)]
    assert figures, "no figure source for late_adaptation"
    page3 = [s for s in figures if s.page == 3]
    assert page3, [(s.page, s.locator) for s in figures]
    assert any(s.error_bar_type is DispersionType.SD for s in page3), \
        [(s.locator, s.error_bar_type) for s in page3]
    assert any("standard deviation" in s.error_bar_evidence.lower() for s in page3), \
        [s.error_bar_evidence for s in page3]
    roster_ids = {r.id for r in bock_map.roster}
    assert all(s.figure_id in roster_ids for s in page3 if s.figure_id)


@replayed
def test_bock_outcome_keys_and_directions_come_from_the_protocol(bock_map, protocol):
    keys = {o.outcome_key for d in bock_map.datasets for o in d.outcomes}
    assert keys <= {o.key for o in protocol.outcomes} and keys
    for dataset in bock_map.datasets:
        for outcome in dataset.outcomes:
            assert outcome.measure_name and outcome.operationalization
            # a direction is never decided WITHOUT a quote — that is the dangerous case, and the
            # one the code guarantees against. "Unknown" may carry a quote: the 2026-08-17
            # recording cites the sentence it judged as NOT settling the after-effect's direction
            # ("similar in both age groups… gradually declined"), which is honest, not wrong.
            if outcome.higher_is_better is not None:
                assert outcome.higher_is_better_evidence, outcome.outcome_key


@replayed
def test_bock_figure_error_bars_were_independently_confirmed(bock_map):
    """Amendment D end-to-end: the second agent read every roster item, so figure bars are agreed."""
    late = bock_map.datasets[0].outcome("late_adaptation")
    figure = next(s for s in late.sources if s.figure_id == "fig01")
    assert figure.error_bar_type is DispersionType.SD
    assert figure.error_bar_agreement == "agreed"
    for dataset in bock_map.datasets:
        for outcome in dataset.outcomes:
            for source in outcome.sources:
                if source.figure_id or source.table_id:
                    assert source.error_bar_agreement in ("agreed", "conflict")
                else:                                   # text sources are Task 8's job
                    assert source.error_bar_agreement == "unconfirmed"
    assert not [f for f in bock_map.needs_human if "error-bar" in f], bock_map.needs_human


@replayed
def test_bock_keeps_the_curve_fit_the_taxonomy_cannot_route(bock_map):
    """The paper's exponential fits are numbers no `SourceKind` describes — kept, not dropped."""
    unknown = [s for d in bock_map.datasets for o in d.outcomes for s in o.sources
               if s.kind is SourceKind.unknown]
    assert unknown and any("exp" in s.quote.lower() or "fit" in s.locator.lower()
                           for s in unknown), [(s.locator, s.quote[:60]) for s in unknown]


@replayed
def test_bock_roster_decides_every_ingested_figure_and_table(bock_map, paper):
    decided = {r.id: r for r in bock_map.roster}
    assert set(decided) == {f.id for f in paper.figures} | {t.id for t in paper.tables}
    for fig in paper.figures:
        assert decided[fig.id].kind == "figure" and decided[fig.id].page == fig.page
    assert decided["fig01"].relevant is True and decided["fig01"].outcome_keys


@replayed
def test_bock_bookkeeping_is_filled_by_code(bock_map, paper):
    assert bock_map.paper_id == paper.sha256
    assert bock_map.model == MODELS["primary"]
    assert bock_map.prompt_version == PROMPT_VERSION
    assert len(bock_map.llm_call_ids) >= 2                # primary + cross-check
    assert len(set(bock_map.llm_call_ids)) == len(bock_map.llm_call_ids)
    assert bock_map.citation.year == 2005
    assert "bock" in bock_map.citation.first_author.lower()


# ------------------------------------------------------------------ offline unit tests
GROUP_A = {"label": "old", "n": 12, "n_evidence": "twelve old subjects", "age_mean": 69.5,
           "age_sd": 7.5, "age_range": "", "notes": ""}
GROUP_B = {"label": "young", "n": 12, "n_evidence": "twelve young subjects", "age_mean": 26.0,
           "age_sd": 5.0, "age_range": "", "notes": ""}
FIG_SOURCE = {"kind": "figure_line", "page": 3, "locator": "Fig 1 adaptation episodes",
              "quote": "Fig. 1 Mean pointing errors", "figure_id": "fig01", "table_id": "",
              "error_bar_type": "SD", "error_bar_scope": "between_subject",
              "error_bar_evidence": "Also shown are the standard deviations",
              "analysis_metric": "endpoint", "values_in_text": "", "notes": ""}


def _outcome(sources=(FIG_SOURCE,), key="late_adaptation"):
    return {"outcome_key": key, "measure_name": "pointing error", "units": "deg",
            "higher_is_better": "lower_is_better", "higher_is_better_evidence": "errors",
            "operationalization": "last adaptation episode", "analysis_metric": "endpoint",
            "sources": [dict(s) for s in sources]}


def _dataset(group_a=None, group_b=None):
    return {"label": "pointing", "experiment": "1", "condition": "rotation",
            "shared_control": False, "exposure_order": "first",
            "group_a": dict(group_a or GROUP_A), "group_b": dict(group_b or GROUP_B),
            "all_groups_listed": [dict(group_a or GROUP_A), dict(group_b or GROUP_B)],
            "chosen_pair_rationale": "the two age groups",
            "moderators": [{"name": "task_type", "value": "visuomotor"}], "notes": ""}


def _sources(outcomes=None, dataset_index=1):
    """The second primary pass: outcomes and their locations, attached to a dataset index."""
    return {"datasets": [{"dataset_index": dataset_index,
                          "outcomes": [dict(o) for o in (outcomes or [_outcome()])]}],
            "notes": ""}


def _primary(datasets=None, roster=("fig01", "fig02", "p1t1"), **over):
    payload = {
        "citation": {"authors": "Bock O", "year": 2005, "title": "Components", "journal": "EBR",
                     "doi": "10.1007/s00221-004-2133-5", "first_author": "Bock"},
        "eligible": True, "eligibility_rationale": "rotation, two age groups",
        "exclusion_reason": "", "design_notes": "same subjects across phases",
        "datasets": [dict(d) for d in ([_dataset()] if datasets is None else datasets)],
        "related_files": [],
        "roster": [{"id": rid, "relevant": rid.startswith("fig"), "reason": "shows the outcome",
                    "outcome_keys": ["late_adaptation"] if rid.startswith("fig") else []}
                   for rid in roster],
        "notes": "",
    }
    payload.update(over)
    return payload


def _check_source(**over):
    src = {"kind": "figure_line", "page": 3, "locator": "Fig 1", "figure_id": "fig01",
           "table_id": "", "error_bar_type": "SD", "quote": "standard deviations"}
    src.update(over)
    return src


def _roster_bars(bars=(("fig01", "SD"),)):
    return [{"id": rid, "error_bar_type": kind, "error_bar_scope": "between_subject",
             "evidence": "Also shown are the standard deviations"} for rid, kind in bars]


def _check(datasets=None, bars=(("fig01", "SD"),), **over):
    payload = {
        "eligible": True, "eligibility_rationale": "rotation, two age groups",
        "roster_error_bars": _roster_bars(bars),
        "datasets": datasets if datasets is not None else [{
            "label": "pointing", "experiment": "1", "condition": "rotation",
            "group_a": {"label": "old", "n": 12, "n_evidence": "twelve old"},
            "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"},
            "outcomes": [{"outcome_key": "late_adaptation", "sources": [_check_source()]}]}],
        "notes": "",
    }
    payload.update(over)
    return payload


def _adjudication(n_a=12, n_b=12, rulings=(), over_datasets=None, **over):
    payload = {
        "eligible": True, "eligibility_rationale": "two age groups, rotated feedback",
        "datasets": over_datasets if over_datasets is not None else [
                     {"primary_dataset_index": 1, "label": "pointing",
                      "group_a": {"label": "old", "n": n_a,
                                  "n_evidence": f"{n_a} analysed after exclusions"},
                      "group_b": {"label": "young", "n": n_b,
                                  "n_evidence": f"{n_b} analysed after exclusions"},
                      "rationale": "methods says twelve per group"}],
        "error_bar_rulings": [dict(r) for r in rulings], "notes": "",
    }
    payload.update(over)
    return payload


def _mapped(paper, protocol, payloads) -> tuple[StudyMap, FakeProvider]:
    provider = FakeProvider(payloads)
    client = LLMClient(provider=provider, cache_dir=None)
    return map_study(client, paper, protocol), provider


def test_code_fills_ids_bookkeeping_and_moderators(paper, protocol):
    study, provider = _mapped(paper, protocol, [_primary(), _sources(), _check()])
    assert len(provider.requests) == 3                  # study map, source map, cross-check
    sha12 = paper.sha256[:12]
    assert study.paper_id == paper.sha256
    assert [d.dataset_id for d in study.datasets] == [f"{sha12}:d1"]
    assert study.datasets[0].cluster_id == sha12
    assert study.datasets[0].moderators == {"task_type": "visuomotor"}
    assert study.datasets[0].outcomes[0].higher_is_better is False
    assert study.datasets[0].outcomes[0].sources[0].figure_id == "fig01"
    assert study.model == MODELS["primary"] and study.prompt_version == PROMPT_VERSION
    assert len(study.llm_call_ids) == 3
    assert not study.needs_human and not study.disagreements
    assert [r.model for r in provider.requests] == [MODELS["primary"], MODELS["primary"],
                                                    MODELS["secondary"]]
    assert [r.effort for r in provider.requests] == ["high", "high", "medium"]


def test_the_source_pass_is_told_which_datasets_to_map(paper, protocol):
    _, provider = _mapped(paper, protocol, [_primary(), _sources(), _check()])
    prompt = provider.requests[1].messages[0]["content"][-1]["text"]
    assert "dataset_index 1" in prompt and '"pointing"' in prompt
    assert "analysed n = 12" in prompt


def test_whole_pdf_is_sent_once_as_the_first_document_block(paper, protocol):
    """One document, first, and NOT marked cacheable — see `tests/test_prompt_cache.py`.

    Each mapper call carries a different output schema, and the schema is part of the cached
    prefix (measured live, task 15 §A1), so a marker here would write a fresh entry per call at
    1.25x the input price and never be read.
    """
    _, provider = _mapped(paper, protocol, [_primary(), _sources(), _check()])
    for request in provider.requests:
        blocks = request.messages[0]["content"]
        documents = [b for b in blocks if b.get("type") == "document"]
        assert len(documents) == 1 and blocks[0] is documents[0]
        assert "cache_control" not in documents[0]
        assert documents[0]["source"]["media_type"] == "application/pdf"
        assert not [b for b in blocks if b.get("type") == "image"]      # no page images


def test_pdf_file_id_is_used_with_the_files_beta(paper, protocol):
    provider = FakeProvider([_primary(), _sources(), _check()])
    client = LLMClient(provider=provider, cache_dir=None)
    map_study(client, paper, protocol, pdf_file_id="file_abc123")
    document = provider.requests[0].messages[0]["content"][0]
    assert document["source"] == {"type": "file", "file_id": "file_abc123"}
    assert provider.requests[0].betas == ["files-api-2025-04-14"]


def test_undecided_roster_ids_get_a_follow_up_call_then_needs_human(paper, protocol):
    follow_up = {"decisions": [{"id": "fig02", "relevant": False,
                                "reason": "tracking task, not an outcome", "outcome_keys": []}]}
    study, provider = _mapped(paper, protocol,
                              [_primary(roster=("fig01",)), _sources(), follow_up, _check()])
    assert len(provider.requests) == 4
    assert provider.requests[2].model == MODELS["secondary"]
    assert provider.requests[2].effort == "medium"
    decided = {r.id: r for r in study.roster}
    assert set(decided) == {"fig01", "fig02", "p1t1"}
    assert decided["fig02"].relevant is False and decided["fig02"].page == 4
    assert decided["p1t1"].relevant is False
    assert decided["p1t1"].reason == "mapper did not decide"
    assert decided["p1t1"].kind == "table"
    assert [f for f in study.needs_human if "p1t1" in f], study.needs_human
    assert not [f for f in study.needs_human if "fig02" in f]
    assert len(study.llm_call_ids) == 4


def test_group_n_mismatch_triggers_adjudication_and_adopts_its_values(paper, protocol):
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 11, "n_evidence": "eleven old"},
        "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [_check_source()]}]}])
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(), check, _adjudication(n_a=12)])
    assert len(provider.requests) == 4
    assert provider.requests[3].model == MODELS["adjudicator"]
    assert provider.requests[3].effort == "xhigh"
    assert study.datasets[0].group_a.n == 12 and study.datasets[0].group_b.n == 12
    assert [d for d in study.disagreements if "group_a" in d and "11" in d], study.disagreements
    assert [d for d in study.disagreements if d.startswith("adjudicated")], study.disagreements
    assert not study.needs_human


def test_adjudicator_can_overturn_the_primary_n(paper, protocol):
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 11, "n_evidence": "eleven old"},
        "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [_check_source()]}]}])
    study, _ = _mapped(paper, protocol, [_primary(), _sources(), check, _adjudication(n_a=11)])
    assert study.datasets[0].group_a.n == 11
    assert study.datasets[0].group_a.n_evidence == "11 analysed after exclusions"
    assert study.datasets[0].group_b.n_evidence == "twelve young subjects"   # unchanged n keeps it


def test_eligibility_disagreement_is_adjudicated(paper, protocol):
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(),
                               _check(eligible=False, eligibility_rationale="no age contrast"),
                               _adjudication()])
    assert len(provider.requests) == 4
    assert study.eligible is True
    assert [d for d in study.disagreements if "eligib" in d.lower()]
    assert not study.needs_human


def test_error_bar_conflict_without_adjudication_keeps_primary_and_asks_for_a_human(
        paper, protocol):
    """The cross-check read the same figure differently — nobody confirmed the primary's SD."""
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(), _check(bars=(("fig01", "SE"),))])
    assert len(provider.requests) == 3                       # no adjudication: eligibility+Ns agree
    source = study.datasets[0].outcomes[0].sources[0]
    assert source.error_bar_type is DispersionType.SD        # primary value survives
    assert source.error_bar_agreement == "conflict"
    flags = [f for f in study.needs_human if "error-bar" in f]
    assert flags and "SD" in flags[0] and "SE" in flags[0], study.needs_human
    assert "late_adaptation" in flags[0] and study.datasets[0].dataset_id in flags[0]
    assert [d for d in study.disagreements if "error bar" in d]


def test_error_bar_agreement_needs_coverage_not_just_absence_of_conflict(paper, protocol):
    """A roster id the cross-check said nothing about is `unconfirmed`, not silently accepted."""
    study, _ = _mapped(paper, protocol, [_primary(), _sources(), _check(bars=(("fig02", "SD"),))])
    source = study.datasets[0].outcomes[0].sources[0]
    assert source.error_bar_agreement == "unconfirmed"
    assert source.error_bar_type is DispersionType.SD
    flags = [f for f in study.needs_human if "unconfirmed" in f]
    assert flags and "fig01" not in flags[0]                 # the cell is named, not the id
    assert "Fig 1 adaptation episodes" in flags[0], study.needs_human


def test_error_bar_agreed_when_the_second_agent_read_the_same_type(paper, protocol):
    study, _ = _mapped(paper, protocol, [_primary(), _sources(), _check()])
    source = study.datasets[0].outcomes[0].sources[0]
    assert source.error_bar_agreement == "agreed"
    assert not study.needs_human


def test_error_bar_scope_mismatch_is_recorded_but_does_not_block(paper, protocol):
    bars = [{"id": "fig01", "error_bar_type": "SD", "error_bar_scope": "within_subject_normalized",
             "evidence": "normalised within subjects"}]
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(), _check(roster_error_bars=bars)])
    assert study.datasets[0].outcomes[0].sources[0].error_bar_agreement == "agreed"
    assert [d for d in study.disagreements if "scope" in d], study.disagreements
    assert not study.needs_human


def test_text_sources_are_exempt_from_the_coverage_rule(paper, protocol):
    """Text sources carry no roster id; Task 8's verifier is what checks them."""
    quote = "the young group averaged 9.9 ± 2.1 deg over the last two episodes"
    text_source = dict(FIG_SOURCE, kind="text_mean_sd", figure_id="", locator="Results ¶2",
                       quote=quote)
    check = _check(bars=(), datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 12, "n_evidence": ""},
        "group_b": {"label": "young", "n": 12, "n_evidence": ""},
        "outcomes": [{"outcome_key": "late_adaptation",
                      "sources": [_check_source(kind="text_mean_sd", figure_id="",
                                                locator="Results, second paragraph",
                                                quote=quote)]}]}])
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_outcome(sources=(text_source,))]), check])
    assert study.datasets[0].outcomes[0].sources[0].error_bar_agreement == "unconfirmed"
    assert not [f for f in study.needs_human if "error-bar" in f], study.needs_human


def test_adjudicated_error_bar_ruling_resolves_the_conflict(paper, protocol):
    check = _check(bars=(("fig01", "SE"),), datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 11, "n_evidence": "eleven old"},
        "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [_check_source()]}]}])
    ruling = {"dataset_index": 1, "outcome_key": "late_adaptation", "page": 3,
              "locator": "Fig 1 adaptation episodes", "figure_id": "fig01", "table_id": "",
              "error_bar_type": "SE", "evidence": "the legend says standard errors"}
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(), check, _adjudication(rulings=[ruling])])
    source = study.datasets[0].outcomes[0].sources[0]
    assert source.error_bar_type is DispersionType.SE
    assert source.error_bar_agreement == "agreed"
    assert source.error_bar_evidence == "the legend says standard errors"
    assert not [f for f in study.needs_human if "error-bar" in f], study.needs_human


def test_sources_only_the_cross_check_found_are_appended(paper, protocol):
    extra = _check_source(page=4, locator="Fig 2 tracking", figure_id="fig02")
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 12, "n_evidence": ""},
        "group_b": {"label": "young", "n": 12, "n_evidence": ""},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [_check_source(), extra]},
                     {"outcome_key": "aftereffect",
                      "sources": [_check_source(page=3, locator="Fig 1 after-effect")]}]}])
    study, _ = _mapped(paper, protocol, [_primary(), _sources(), check])
    late = study.datasets[0].outcome("late_adaptation").sources
    assert [s.locator for s in late] == ["Fig 1 adaptation episodes", "Fig 2 tracking"]
    assert late[1].notes == "added by cross-check" and late[1].figure_id == "fig02"
    after = study.datasets[0].outcome("aftereffect").sources          # outcome the primary missed
    assert [s.notes for s in after] == ["added by cross-check"]
    assert len([d for d in study.disagreements if "added by cross-check" in d]) == 2


def test_the_same_text_source_quoted_differently_is_not_duplicated(paper, protocol):
    """Text sources carry no figure id, so the quote is what identifies the location."""
    same = _check_source(kind="test_statistic", locator="Results, adaptation phase ANOVA",
                         figure_id="", error_bar_type="UNKNOWN",
                         quote="ANOVA yielded significant effects for Age (F(1,22)=7.58, P<0.05) "
                               "and Episode (F(14,308)=41.14, P<0.001), and their interaction.")
    text_source = dict(FIG_SOURCE, kind="test_statistic", page=3, figure_id="",
                       locator="Results, ANOVA on the adaptation phase (Age x Episode)",
                       error_bar_type="NONE",
                       quote="ANOVA yielded significant effects for Age (F(1,22)=7.58, P<0.05) "
                             "and Episode (F(14,308)=41.14, P<0.001)")
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 12, "n_evidence": ""},
        "group_b": {"label": "young", "n": 12, "n_evidence": ""},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [same]}]}])
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_outcome(sources=(text_source,))]), check])
    assert len(study.datasets[0].outcomes[0].sources) == 1
    assert not [d for d in study.disagreements if "added by cross-check" in d]


def _swapped_check(**over):
    return _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "young", "n": 12, "n_evidence": "twelve young"},
        "group_b": {"label": "old", "n": 12, "n_evidence": "twelve old"},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [_check_source()]}]}], **over)


def test_swapped_group_mapping_is_adjudicated_and_can_swap_the_groups(paper, protocol):
    swapped = _adjudication(over_datasets=[{
        "primary_dataset_index": 1, "label": "pointing",
        "group_a": {"label": "young", "n": 12, "n_evidence": "twelve young"},
        "group_b": {"label": "old", "n": 12, "n_evidence": "twelve old"},
        "rationale": "the protocol calls the younger group A"}])
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(), _swapped_check(), swapped])
    assert len(provider.requests) == 4                        # a mapping conflict is adjudicated
    assert provider.requests[3].model == MODELS["adjudicator"]
    assert (study.datasets[0].group_a.label, study.datasets[0].group_b.label) == ("young", "old")
    assert [d for d in study.disagreements if "swapped to" in d], study.disagreements
    assert not [f for f in study.needs_human if "group mapping" in f]


def test_swapped_group_mapping_can_be_settled_for_the_primary(paper, protocol):
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(), _swapped_check(), _adjudication()])
    assert study.datasets[0].group_a.label == "old"            # primary mapping confirmed
    assert [d for d in study.disagreements if "primary confirmed" in d]
    assert not [f for f in study.needs_human if "group mapping" in f], study.needs_human


def test_group_mapping_the_adjudicator_does_not_settle_needs_a_human(paper, protocol):
    unrelated = _adjudication(over_datasets=[{
        "primary_dataset_index": 1, "label": "pointing",
        "group_a": {"label": "left hand", "n": 12, "n_evidence": ""},
        "group_b": {"label": "right hand", "n": 12, "n_evidence": ""},
        "rationale": "different pair entirely"}])
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(), _swapped_check(), unrelated])
    assert study.datasets[0].group_a.label == "old"            # primary kept
    assert [f for f in study.needs_human if "group mapping" in f], study.needs_human


def test_group_mapping_without_a_counterpart_dataset_is_unconfirmed(paper, protocol):
    study, _ = _mapped(paper, protocol,
                       [_primary(datasets=[_dataset(), _dataset()]), _sources(), _check()])
    flags = [f for f in study.needs_human if "group mapping unconfirmed" in f]
    assert flags and study.datasets[1].dataset_id in flags[0], study.needs_human


def test_a_dataset_only_the_cross_check_found_is_described_not_invented(paper, protocol):
    extra = {"label": "force field", "experiment": "2", "condition": "curl field",
             "group_a": {"label": "old", "n": 9, "n_evidence": "nine older adults"},
             "group_b": {"label": "young", "n": 10, "n_evidence": "ten younger adults"},
             "outcomes": [{"outcome_key": "aftereffect",
                           "sources": [_check_source(page=4, locator="Fig 2", figure_id="fig02")]}]}
    check = _check(datasets=[_check()["datasets"][0], extra])
    study, _ = _mapped(paper, protocol, [_primary(), _sources(), check])
    assert len(study.datasets) == 1                            # never invented into the map
    described = [d for d in study.disagreements if "not in the map" in d]
    assert described and "force field" in described[0] and "n=9" in described[0]
    assert "aftereffect" in described[0] and "fig02" in described[0], described
    assert [f for f in study.needs_human if "does not contain" in f], study.needs_human


def test_dataset_count_mismatch_is_recorded(paper, protocol):
    study, _ = _mapped(paper, protocol,
                       [_primary(datasets=[_dataset(), _dataset()]), _sources(), _check()])
    assert [d for d in study.disagreements if "dataset count" in d], study.disagreements
    assert len(study.datasets) == 2
    assert [d.dataset_id[-2:] for d in study.datasets] == ["d1", "d2"]


def test_unknown_kind_sources_are_kept_and_flagged_when_they_stand_alone(paper, protocol):
    """A numeric location we cannot route (a fitted curve, say) is a human's problem, not a drop."""
    unknown = dict(FIG_SOURCE, kind="unknown", locator="Results, exponential fit",
                   figure_id="", quote="y=9.92+53.51*exp(-x/5.89) for young subjects")
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_outcome(sources=(unknown,))]),
                        _check(datasets=[])])
    kept = study.datasets[0].outcomes[0].sources
    assert [s.kind for s in kept] == [SourceKind.unknown]
    assert kept[0].locator == "Results, exponential fit"
    flags = [f for f in study.needs_human if "unlisted kind" in f]
    assert flags and "y=9.92" in flags[0], study.needs_human      # the quote travels with the flag


def test_unknown_kind_alongside_a_usable_source_is_not_flagged(paper, protocol):
    unknown = dict(FIG_SOURCE, kind="unknown", locator="Results, exponential fit", figure_id="")
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_outcome(sources=(FIG_SOURCE, unknown))]),
                        _check()])
    assert len(study.datasets[0].outcomes[0].sources) == 2
    assert not [f for f in study.needs_human if "unlisted kind" in f], study.needs_human


def test_outcome_key_outside_the_protocol_is_flagged(paper, protocol):
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_outcome(key="learning_rate")]), _check()])
    assert [f for f in study.needs_human if "learning_rate" in f], study.needs_human


def test_sources_for_an_unknown_dataset_index_are_flagged(paper, protocol):
    study, _ = _mapped(paper, protocol, [_primary(), _sources(dataset_index=7), _check()])
    assert [f for f in study.needs_human if "dataset_index 7" in f], study.needs_human
    kept = [s for o in study.datasets[0].outcomes for s in o.sources]
    assert [s.notes for s in kept] == ["added by cross-check"]   # only the check's copy survives


def test_ineligible_paper_keeps_its_exclusion_reason(paper, protocol):
    payload = _primary(datasets=[], eligible=False, eligibility_rationale="single age group",
                       exclusion_reason="no contrast between the protocol's groups")
    study, provider = _mapped(paper, protocol, [payload, _check(eligible=False, datasets=[])])
    assert study.eligible is False
    assert study.exclusion_reason == "no contrast between the protocol's groups"
    assert study.datasets == []
    assert len(provider.requests) == 2      # no datasets, so no source pass; the check still runs
    assert [r.model for r in provider.requests] == [MODELS["primary"], MODELS["secondary"]]


# ------------------------------------------------------------------ silent-failure paths
def test_empty_source_pass_is_retried_then_flagged(paper, protocol):
    empty = {"datasets": [], "notes": "nothing found"}
    study, provider = _mapped(paper, protocol, [_primary(), empty, empty, _check(datasets=[])])
    assert len(provider.requests) == 4                     # study map, source map, retry, check
    assert provider.requests[1].key != provider.requests[2].key      # the retry is a real call
    assert provider.requests[1].model == provider.requests[2].model == MODELS["primary"]
    assert study.datasets[0].outcomes == []
    flags = [f for f in study.needs_human if "no sources found" in f]
    assert flags and study.datasets[0].dataset_id in flags[0], study.needs_human
    assert len([d for d in study.disagreements if "no locations" in d]) == 2


def test_source_pass_retry_that_succeeds_is_used(paper, protocol):
    study, provider = _mapped(paper, protocol,
                              [_primary(), {"datasets": [], "notes": ""}, _sources(), _check()])
    assert len(provider.requests) == 4
    assert [o.outcome_key for o in study.datasets[0].outcomes] == ["late_adaptation"]
    assert not [f for f in study.needs_human if "no sources" in f], study.needs_human
    assert [d for d in study.disagreements if "no locations" in d]


def test_a_figure_id_ingestion_never_produced_is_cleared_and_flagged(paper, protocol):
    """`llm.context` raises on an unknown id, so a hallucinated one must never reach an extractor."""
    invented = dict(FIG_SOURCE, figure_id="fig99", locator="Fig 9, right panel",
                    quote="Fig. 9 shows the group means")
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_outcome(sources=(invented,))]),
                        _check(datasets=[])])
    source = study.datasets[0].outcomes[0].sources[0]
    assert source.figure_id is None                        # cleared, but the location is kept
    assert source.locator == "Fig 9, right panel"
    assert "not in the ingestion roster" in source.notes
    flags = [f for f in study.needs_human if "not in the ingestion roster" in f]
    assert flags and "fig99" in flags[0] and "Fig. 9 shows" in flags[0], study.needs_human


def test_a_bad_id_from_the_cross_check_is_validated_the_same_way(paper, protocol):
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 12, "n_evidence": ""},
        "group_b": {"label": "young", "n": 12, "n_evidence": ""},
        "outcomes": [{"outcome_key": "late_adaptation",
                      "sources": [_check_source(figure_id="fig42", locator="Fig 42", page=5)]}]}])
    study, _ = _mapped(paper, protocol, [_primary(), _sources(), check])
    added = [s for o in study.datasets[0].outcomes for s in o.sources if "cross-check" in s.notes]
    assert added and added[0].figure_id is None
    assert [f for f in study.needs_human if "fig42" in f], study.needs_human


def test_an_outcome_key_from_the_cross_check_is_validated_the_same_way(paper, protocol):
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 12, "n_evidence": ""},
        "group_b": {"label": "young", "n": 12, "n_evidence": ""},
        "outcomes": [{"outcome_key": "reaction_time", "sources": [_check_source()]}]}])
    study, _ = _mapped(paper, protocol, [_primary(), _sources(), check])
    assert [o.outcome_key for o in study.datasets[0].outcomes] == ["late_adaptation"]
    assert [f for f in study.needs_human if "reaction_time" in f], study.needs_human


def test_a_failing_roster_follow_up_does_not_kill_the_map(paper, protocol):
    def explode(request):
        raise RefusalError("the model refused")

    study, provider = _mapped(paper, protocol,
                              [_primary(roster=("fig01",)), _sources(), explode, _check()])
    assert len(provider.requests) == 4
    assert study.eligible is True                          # the map survived
    decided = {r.id: r for r in study.roster}
    assert set(decided) == {"fig01", "fig02", "p1t1"}
    assert decided["fig02"].reason == "mapper did not decide"
    assert [d for d in study.disagreements if "roster follow-up failed" in d], study.disagreements
    assert len([f for f in study.needs_human if "did not decide" in f]) == 2


def test_adjudicated_n_that_agrees_with_neither_agent_is_not_adopted(paper, protocol):
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 11, "n_evidence": "eleven old"},
        "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [_check_source()]}]}])
    study, _ = _mapped(paper, protocol, [_primary(), _sources(), check, _adjudication(n_a=7)])
    assert study.datasets[0].group_a.n == 12                # primary kept, nothing invented
    assert [d for d in study.disagreements if "agrees with neither" in d], study.disagreements
    flags = [f for f in study.needs_human if "n disagreement unresolved" in f]
    assert flags and "adjudicator=7" in flags[0], study.needs_human


def test_adjudicated_dataset_index_the_map_lacks_is_flagged(paper, protocol):
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 11, "n_evidence": "eleven old"},
        "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [_check_source()]}]}])
    stray = _adjudication(over_datasets=[{
        "primary_dataset_index": 9, "label": "second experiment",
        "group_a": {"label": "old", "n": 8, "n_evidence": ""},
        "group_b": {"label": "young", "n": 8, "n_evidence": ""}, "rationale": "extra"}])
    study, _ = _mapped(paper, protocol, [_primary(), _sources(), check, stray])
    assert len(study.datasets) == 1 and study.datasets[0].group_a.n == 12
    assert [f for f in study.needs_human
            if "adjudicator" in f and "second experiment" in f], study.needs_human


# ------------------------------------------------------------------ source matching (R4)
def test_two_panels_of_one_figure_are_not_the_same_location(paper, protocol):
    caption = "Fig. 2 Mean tracking errors of young (triangles) and old (squares) subjects"
    panel_a = dict(FIG_SOURCE, page=4, figure_id="fig02", quote=caption,
                   locator="Fig 2A, last block, filled squares")
    panel_b = _check_source(page=4, figure_id="fig02", quote=caption,
                            locator="Fig 2B, last block, open circles")
    check = _check(bars=(("fig02", "SD"),), datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 12, "n_evidence": ""},
        "group_b": {"label": "young", "n": 12, "n_evidence": ""},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [panel_b]}]}])
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_outcome(sources=(panel_a,))]), check])
    locators = [s.locator for s in study.datasets[0].outcomes[0].sources]
    assert locators == ["Fig 2A, last block, filled squares", "Fig 2B, last block, open circles"]


def test_two_rows_of_one_table_are_not_the_same_location(paper, protocol):
    row_old = dict(FIG_SOURCE, kind="table", figure_id="", table_id="p1t1", page=1,
                   locator="Table 1, row 'old'", quote="Table 1 Group means")
    row_young = _check_source(kind="table", figure_id="", table_id="p1t1", page=1,
                              locator="Table 1, row 'young'", quote="Table 1 Group means")
    check = _check(bars=(("p1t1", "SD"),), datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 12, "n_evidence": ""},
        "group_b": {"label": "young", "n": 12, "n_evidence": ""},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [row_young]}]}])
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_outcome(sources=(row_old,))]), check])
    assert [s.locator for s in study.datasets[0].outcomes[0].sources] == [
        "Table 1, row 'old'", "Table 1, row 'young'"]


def test_the_same_panel_described_differently_is_one_location(paper, protocol):
    panel = dict(FIG_SOURCE, page=4, figure_id="fig02", locator="Fig 2B, last block")
    same = _check_source(page=4, figure_id="fig02", locator="panel B of Figure 2, final block")
    check = _check(bars=(("fig02", "SD"),), datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 12, "n_evidence": ""},
        "group_b": {"label": "young", "n": 12, "n_evidence": ""},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [same]}]}])
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_outcome(sources=(panel,))]), check])
    assert len(study.datasets[0].outcomes[0].sources) == 1


# ------------------------------------------------------------------ misc
def test_one_outcome_reported_twice_keeps_both_rows_and_all_metadata(paper, protocol):
    first = dict(_outcome(sources=(FIG_SOURCE,)), units="", higher_is_better="unknown")
    second = dict(_outcome(sources=(dict(FIG_SOURCE, page=4, figure_id="fig02",
                                         locator="Fig 2, last block"),)),
                  measure_name="", units="deg", higher_is_better="lower_is_better",
                  operationalization="last adaptation episode")
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[first, second]), _check()])
    outcome = study.datasets[0].outcomes[0]
    assert len(study.datasets[0].outcomes) == 1
    assert [s.locator for s in outcome.sources] == ["Fig 1 adaptation episodes",
                                                    "Fig 2, last block"]
    assert outcome.measure_name == "pointing error"        # kept from the first row
    assert outcome.units == "deg"                          # filled in from the second
    assert outcome.higher_is_better is False               # filled in from the second
    assert not [f for f in study.needs_human if "reported twice" in f], study.needs_human


def test_conflicting_metadata_across_two_rows_of_one_outcome_is_flagged(paper, protocol):
    first = _outcome(sources=(FIG_SOURCE,))
    second = dict(_outcome(sources=()), measure_name="percent compensation",
                  higher_is_better="higher_is_better")
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[first, second]), _check()])
    assert [f for f in study.needs_human if "measure_name" in f], study.needs_human
    assert [f for f in study.needs_human if "higher_is_better" in f], study.needs_human


def test_prompt_version_tracks_the_prompt_files(tmp_path, monkeypatch):
    from canopy.agents import mapper

    assert PROMPT_VERSION.startswith("mapper/1@") and len(PROMPT_VERSION) == len("mapper/1@") + 8
    assert mapper.prompt_fingerprint() == PROMPT_VERSION.split("@")[1]
    original = mapper.load_prompt

    monkeypatch.setattr(mapper, "load_prompt", lambda name: original(name) + "\nedited")
    assert mapper.prompt_fingerprint() != PROMPT_VERSION.split("@")[1]


# ------------------------------------------------------------------ eligible, but empty
def test_an_eligible_paper_with_no_dataset_is_asked_a_second_time(paper, protocol):
    """Heuer & Hegele 2008 in `runs/proof`: eligible, 900 words of design notes, `datasets: []`.

    No dataset means no source pass, no candidate and no row, so the paper left that run without
    appearing anywhere — while the cross-check of the same PDF found two datasets. A
    contradiction that costs a whole paper is worth one more question.
    """
    study, provider = _mapped(paper, protocol,
                              [_primary(datasets=[]), _primary(), _sources(), _check()])
    assert len(provider.requests) == 4                   # map, map again, sources, cross-check
    assert study.datasets and study.datasets[0].dataset_id == f"{paper.sha256[:12]}:d1"
    assert any("asked again and got 1" in d for d in study.disagreements)
    assert not any("twice" in flag for flag in study.needs_human)


def test_the_second_empty_answer_is_flagged_for_a_human_rather_than_dropped(paper, protocol):
    study, provider = _mapped(paper, protocol,
                              [_primary(datasets=[]), _primary(datasets=[]), _check(datasets=[])])
    assert len(provider.requests) == 3                   # no source pass: there is nothing to map
    assert not study.datasets
    assert any("no dataset in a paper it called eligible, twice" in f for f in study.needs_human)


def test_an_ineligible_paper_is_not_asked_again(paper, protocol):
    """The retry answers a contradiction. "Ineligible, and so no dataset" is not one."""
    _, provider = _mapped(paper, protocol,
                          [_primary(datasets=[], eligible=False), _check(eligible=False)])
    assert len(provider.requests) == 2


def test_a_second_report_of_a_different_quantity_does_not_lend_this_outcome_its_pages(paper,
                                                                                      protocol):
    """Heuer & Hegele 2008 came back as the practice-block error, not the adaptive shift.

    The two reports were merged before the conflict was looked for, so both panels' locations sat
    under one outcome key and an extractor pointed at that outcome could read either. A source
    that measures a different quantity is not a source for this outcome.
    """
    first = _outcome(sources=(FIG_SOURCE,))
    second = dict(_outcome(sources=(dict(FIG_SOURCE, page=4, figure_id="fig02",
                                         locator="Fig 2, practice blocks"),)),
                  measure_name="initial direction error", units="deg")
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[first, second]), _check()])
    outcome = study.datasets[0].outcomes[0]
    assert [s.locator for s in outcome.sources] == ["Fig 1 adaptation episodes"]
    assert outcome.measure_name == "pointing error"        # the first reading is kept intact
    assert any("were NOT added to this outcome" in f for f in study.needs_human), study.needs_human
    assert any("different quantity" in f for f in study.needs_human)
