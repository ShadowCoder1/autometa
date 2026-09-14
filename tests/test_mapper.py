"""Task 4: the study mapper.

Two halves:
  * a replayed end-to-end map of Bock 2005 (fixtures in `tests/fixtures/llm`, recorded once live);
  * offline unit tests with `FakeProvider` for everything the mapper decides in *code* — roster
    completeness, cross-check diffing, adjudication, needs_human rules, id assignment.
"""
from __future__ import annotations

import pytest

from canopy.agents import load_prompt
from typing import get_args

from canopy.agents.mapper import (MAPPER_CROSSCHECK_SCHEMA, MAPPER_SCHEMA,
                                  MAPPER_SOURCES_SCHEMA, PROMPT_VERSION,
                                  _Conflicts, _diff_measures, apply_map_answers, c6_demoted,
                                  extraction_blocks, map_answer_effects,
                                  map_study, measure_of, named_alternatives,
                                  open_map_questions, protocol_text, readable_sources,
                                  reopened_outcome, roster_text, settled_measure,
                                  source_unreadable_reason, split_metrics)
from canopy.config import MODELS
from canopy.llm.client import LLMClient
from canopy.llm.errors import RefusalError
from canopy.llm.providers import FakeProvider
from canopy.llm.schemas import assert_no_derived_stats, assert_valid_output_schema
from canopy.models import (AnalysisMetric, DatasetSpec, DispersionType, OutcomeSources,
                           Source, SourceKind, StudyMap)

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
    # the cross-check reports no dataset at all, which is C7's case (a dataset only one agent
    # proposed), so the map now also buys the adjudication call that settles its inclusion
    study, provider = _mapped(paper, protocol,
                              [_primary(), empty, empty, _check(datasets=[]),
                               _adjudication(dataset_inclusions=_inclusion(index=1,
                                                                           verdict="include"))])
    assert len(provider.requests) == 5           # study map, source map, retry, check, adjudicate
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


# ------------------------------------------------------------------ C7: inclusion is a decision
def _two_datasets():
    """A second dataset the cross-check will not report — Bock's tracking-only control sample."""
    second = _dataset()
    second["label"] = "tracking only"
    second["experiment"] = "control"
    second["condition"] = "tracking, no pointing"
    return [_dataset(), second]


def _inclusion(index=2, verdict="exclude", rule="Dataset Rule 3", quote="never performed the "
               "pointing adaptation task", rationale="the control sample did another task"):
    return [{"primary_dataset_index": index, "verdict": verdict, "rule": rule, "quote": quote,
             "rationale": rationale}]


def test_a_dataset_only_one_agent_mapped_is_an_inclusion_question_not_a_warning(paper, protocol):
    """C7. `map.json disagreements[0] = "dataset count: primary=2 cross-check=1"` in the real run;
    the cross-check's rejection was rule-cited and the pipeline extracted the dataset anyway,
    producing a fully signed d = -2.9610 from a figure source scored 2/6."""
    study, provider = _mapped(paper, protocol,
                              [_primary(datasets=_two_datasets()), _sources(), _check(),
                               _adjudication(dataset_inclusions=_inclusion())])
    assert len(provider.requests) == 4                    # the count mismatch alone bought the call
    assert provider.requests[3].model == MODELS["adjudicator"]
    assert [d.included for d in study.datasets] == [True, False]
    excluded = study.datasets[1]
    assert excluded.exclusion_rule == "Dataset Rule 3"
    assert "pointing adaptation task" in excluded.exclusion_quote
    assert not study.open_questions                       # settled, so nothing to ask
    datasets, cells = extraction_blocks(study)
    assert excluded.dataset_id in datasets and not cells
    assert "Dataset Rule 3" in datasets[excluded.dataset_id]


def test_an_exclusion_that_cites_no_protocol_rule_is_not_an_exclusion(paper, protocol):
    """Reject only on a NAMED rule, with a quote. An adjudicator that cannot say which rule it is
    applying is guessing, and a guess that deletes a dataset is worse than a question."""
    study, _ = _mapped(paper, protocol,
                       [_primary(datasets=_two_datasets()), _sources(), _check(),
                        _adjudication(dataset_inclusions=_inclusion(rule="", quote=""))])
    assert [d.included for d in study.datasets] == [True, True]
    assert [q.kind for q in study.open_questions] == ["include_dataset"]
    assert study.open_questions[0].dataset_id == study.datasets[1].dataset_id
    assert [f for f in study.needs_human if "inclusion needs human" in f]


def test_a_dataset_the_adjudicator_cannot_place_becomes_an_include_dataset_question(paper,
                                                                                   protocol):
    study, _ = _mapped(paper, protocol,
                       [_primary(datasets=_two_datasets()), _sources(), _check(),
                        _adjudication(dataset_inclusions=_inclusion(verdict="unknown"))])
    assert [d.included for d in study.datasets] == [True, True]
    question = study.open_questions[0]
    assert question.kind == "include_dataset" and question.options == ["include it", "exclude it"]
    datasets, _ = extraction_blocks(study)
    assert study.datasets[1].dataset_id in datasets
    assert "include_dataset" in datasets[study.datasets[1].dataset_id]


def test_an_adjudicator_that_says_include_lets_the_dataset_through(paper, protocol):
    study, _ = _mapped(paper, protocol,
                       [_primary(datasets=_two_datasets()), _sources(), _check(),
                        _adjudication(dataset_inclusions=_inclusion(verdict="include",
                                                                    rule="Eligibility 1"))])
    assert all(d.included for d in study.datasets)
    assert not study.open_questions
    assert extraction_blocks(study) == ({}, {})


def test_the_single_mapper_dataset_is_found_by_identity_not_by_list_position(paper, protocol):
    """F3, reproduced from the review: the pairing was `zip(primary, cross_check)`.

    With the tracking-only control sample FIRST in the primary map and the cross-check proposing
    only the pointing dataset, position pairing compared tracking-only with pointing, blocked
    POINTING — the dataset both agents proposed — and let the tracking-only sample through, which
    is C7's own motivating case (`d = -2.9610`). Nothing makes a mapper's output order stable.
    """
    tracking, pointing = _two_datasets()[1], _two_datasets()[0]
    study, _ = _mapped(paper, protocol,
                       [_primary(datasets=[tracking, pointing]), _sources(dataset_index=2),
                        _check(), _adjudication(dataset_inclusions=[])])
    labels = {d.dataset_id: d.label for d in study.datasets}
    asked = [q.dataset_id for q in study.open_questions if q.kind == "include_dataset"]
    assert [labels[d] for d in asked] == ["tracking only"]
    blocked, _ = extraction_blocks(study)
    assert [labels[d] for d in blocked] == ["tracking only"]
    reason = next(iter(blocked.values()))
    assert "tracking only" in reason                             # …and it says which one it is
    assert "proposed" not in reason                              # not a claim it cannot check


def test_the_two_agents_real_descriptions_of_bocks_datasets_pair_the_way_a_reader_would():
    """The identity match, on the words the two agents really wrote (`runs/rerun-fixed` — the map
    stage file and the cross-check payload in the run's own LLM cache).

    The cross-check mapped ONE dataset, the pointing experiment, and described it in its own
    words; the primary mapped two. The pointing pair must be the match and the tracking-only
    control sample the one with no counterpart — it is the sample C7 exists to stop
    (`d = -2.9610`), and under list-position pairing it was the one that got through.
    """
    from canopy.agents.mapper import DATASET_MATCH, _dataset_similarity, _pair_datasets

    pointing = DatasetSpec(
        dataset_id="b511dbb76fa6:d1",
        label="Pointing adaptation to +60° rotation: old vs young (experimental sample)",
        experiment="Main experiment (pointing)",
        condition="+60° rotated visual feedback, pointing; after-effect without feedback")
    tracking = DatasetSpec(
        dataset_id="b511dbb76fa6:d2",
        label="Tracking under +60° rotation: old vs young control (naive) sample",
        experiment="Control sample (tracking only)",
        condition="+60° rotated visual feedback during tracking, no pointing pre-adaptation")
    check = {"label": "Bock (2005) Experiment 1 – Pointing adaptation to 60° visuomotor rotation",
             "experiment": "Pointing task, first (main) experiment on Day 2: baseline, "
                           "60°-rotated adaptation, after-effect, refresh phases",
             "condition": "60° rotated visual feedback (visuomotor rotation), 8 possible target "
                          "directions"}
    assert _dataset_similarity(pointing, check) >= DATASET_MATCH
    assert _dataset_similarity(tracking, check) < _dataset_similarity(pointing, check)
    for order in ([pointing, tracking], [tracking, pointing]):
        paired = _pair_datasets(order, [check])
        matched = {order[i].dataset_id for i in paired}
        assert matched == {"b511dbb76fa6:d1"}, order[0].dataset_id


def test_the_two_agents_real_descriptions_of_heuers_two_experiments_pair_one_to_one():
    """The same run's Heuer cross-check mapped both experiments, in its own words: both pair, and
    neither becomes an inclusion question."""
    from canopy.agents.mapper import _pair_datasets

    datasets = [
        DatasetSpec(dataset_id="3570e4ce2a9c:d1",
                    label="Exp 1a: 75° CCW rotation, 8 target directions — older vs younger",
                    experiment="Experiment 1a", condition="75° counterclockwise rotation"),
        DatasetSpec(dataset_id="3570e4ce2a9c:d2",
                    label="Exp 2: 30° CCW rotation, single practiced target — older vs younger",
                    experiment="Experiment 2", condition="30° counterclockwise rotation")]
    check = [{"label": "Experiment 1a (75° CCW rotation, 8 target directions)",
              "experiment": "Experiment 1a",
              "condition": "75° counterclockwise visuomotor rotation, 8 target directions during "
                           "practice"},
             {"label": "Experiment 2 (30° CCW rotation, single practice target, 5 test "
                       "directions)",
              "experiment": "Experiment 2",
              "condition": "30° counterclockwise visuomotor rotation, single target during "
                           "practice, tested across 5 target directions x 3 amplitudes"}]
    assert _pair_datasets(datasets, check) == {0: 0, 1: 1}
    assert _pair_datasets(datasets, list(reversed(check))) == {0: 1, 1: 0}


def test_a_cross_check_that_lists_its_datasets_in_the_other_order_matches_both(paper, protocol):
    """Identity, not order: the same two datasets described in the other sequence are the same
    two datasets, and neither becomes an inclusion question."""
    tracking, pointing = _two_datasets()[1], _two_datasets()[0]
    check = _check(datasets=[
        {"label": "tracking only", "experiment": "control", "condition": "tracking, no pointing",
         "group_a": {"label": "old", "n": 12, "n_evidence": "twelve old"},
         "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"}, "outcomes": []},
        {"label": "pointing", "experiment": "1", "condition": "rotation",
         "group_a": {"label": "old", "n": 12, "n_evidence": "twelve old"},
         "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"},
         "outcomes": [{"outcome_key": "late_adaptation", "sources": [_check_source()]}]}])
    study, provider = _mapped(paper, protocol,
                              [_primary(datasets=[pointing, tracking]), _sources(), check])
    assert len(provider.requests) == 3                # no adjudication: nothing disagrees
    assert not study.open_questions
    assert extraction_blocks(study) == ({}, {})


def test_an_adjudicator_silent_on_the_inclusion_leaves_the_question_open(paper, protocol):
    """No ruling at all is not consent."""
    study, _ = _mapped(paper, protocol,
                       [_primary(datasets=_two_datasets()), _sources(), _check(),
                        _adjudication(dataset_inclusions=[])])
    assert [q.kind for q in study.open_questions] == ["include_dataset"]


# ------------------------------------------------------------------ C6: one outcome, one measure
ALT_MEASURE = "adaptive shift; alternatively initial direction error in the last practice block"
ALT_OPERATIONALIZATION = ("Two candidate operationalizations. (a) the posttest-pretest difference; "
                          "(b) the error in the last block.")


def _two_measure_outcome():
    """The real shape of `3570e4ce2a9c:d1 late_adaptation`: two candidate operationalizations in
    the mapper's own words, and value sources split on `analysis_metric`."""
    endpoint = dict(FIG_SOURCE, locator="Results, Practice paragraph", kind="text_mean_sd",
                    figure_id="", analysis_metric="endpoint", page=4)
    change = dict(FIG_SOURCE, analysis_metric="change_from_baseline")
    outcome = _outcome(sources=(endpoint, change))
    outcome["measure_name"] = ALT_MEASURE
    outcome["operationalization"] = ALT_OPERATIONALIZATION
    outcome["analysis_metric"] = "change_from_baseline"
    return outcome


def _measure_ruling(verdict="winner", metric="change_from_baseline",
                    winner_quote="the difference between posttest and pretest",
                    loser_quote="initial direction error in the last practice block",
                    winning_location="", losing_locations=()):
    return [{"dataset_index": 1, "outcome_key": "late_adaptation", "verdict": verdict,
             "winning_analysis_metric": metric, "winning_location": winning_location,
             "losing_locations": list(losing_locations), "winner_quote": winner_quote,
             "loser_quote": loser_quote, "rationale": "the window names the open-loop test"}]


def test_two_measures_under_one_outcome_are_settled_once_and_the_loser_is_demoted(paper, protocol):
    """C6 acceptance: exactly one adjudication call, one `value`, one `alternate`."""
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(outcomes=[_two_measure_outcome()]), _check(),
                               _adjudication(measure_rulings=_measure_ruling())])
    assert len(provider.requests) == 4                    # ONE adjudication, not one per measure
    outcome = study.datasets[0].outcomes[0]
    roles = {s.analysis_metric: s.role for s in outcome.sources}
    assert roles == {"change_from_baseline": "value", "endpoint": "alternate"}
    assert outcome.analysis_metric == "change_from_baseline"
    assert "winner:" in outcome.measure_ruling and "loser:" in outcome.measure_ruling
    loser = next(s for s in outcome.sources if s.role == "alternate")
    assert "demoted to alternate" in loser.notes and "last practice block" in loser.notes
    assert not study.open_questions
    assert extraction_blocks(study) == ({}, {})


def test_a_measure_toss_up_asks_and_buys_no_extraction(paper, protocol):
    """Two expert readers of the same sentence give opposite answers, so the tool asks."""
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_two_measure_outcome()]), _check(),
                        _adjudication(measure_rulings=_measure_ruling(verdict="toss_up"))])
    outcome = study.datasets[0].outcomes[0]
    assert {s.role for s in outcome.sources} == {"value"}          # nothing demoted
    question = study.open_questions[0]
    assert question.kind == "which_measure" and question.outcome_key == "late_adaptation"
    assert [o.split(" — ")[0] for o in question.options] == ["endpoint", "change_from_baseline"]
    assert all(" — " in o for o in question.options)      # …and WHERE each one is read
    _, cells = extraction_blocks(study)
    assert cells[(study.datasets[0].dataset_id, "late_adaptation")].startswith("an unanswered "
                                                                              "which_measure")


def test_a_measure_ruling_without_both_quotes_settles_nothing(paper, protocol):
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_two_measure_outcome()]), _check(),
                        _adjudication(measure_rulings=_measure_ruling(loser_quote=""))])
    assert {s.role for s in study.datasets[0].outcomes[0].sources} == {"value"}
    assert [q.kind for q in study.open_questions] == ["which_measure"]


def test_a_ruling_for_a_metric_no_location_carries_is_not_applied(paper, protocol):
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_two_measure_outcome()]), _check(),
                        _adjudication(measure_rulings=_measure_ruling(metric="percent_of_perturbation"))])
    assert {s.role for s in study.datasets[0].outcomes[0].sources} == {"value"}
    assert [q.kind for q in study.open_questions] == ["which_measure"]
    assert [d for d in study.disagreements if "which_measure" in d and "not applied" in d]


def test_the_mappers_own_words_alone_open_the_question(paper, protocol):
    """`3570e4ce2a9c:d2 late_adaptation` names an alternative while both its value locations
    measure the same thing — the words are enough to ask."""
    outcome = _two_measure_outcome()
    outcome["sources"] = [dict(s, analysis_metric="change_from_baseline")
                          for s in outcome["sources"]]
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(outcomes=[outcome]), _check(),
                               _adjudication(measure_rulings=_measure_ruling(verdict="toss_up"))])
    assert len(provider.requests) == 4
    assert [q.kind for q in study.open_questions] == ["which_measure"]


def test_split_metrics_alone_open_the_question_without_the_word(paper, protocol):
    outcome = _two_measure_outcome()
    outcome["measure_name"] = "adaptive shift"
    outcome["operationalization"] = "the posttest-pretest difference"
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(outcomes=[outcome]), _check(),
                               _adjudication(measure_rulings=_measure_ruling())])
    assert len(provider.requests) == 4
    assert {s.role for s in study.datasets[0].outcomes[0].sources} == {"value", "alternate"}


def test_one_measure_and_no_alternative_words_buys_no_adjudication(paper, protocol):
    """The gate must not fire on every outcome: three calls, as before."""
    study, provider = _mapped(paper, protocol, [_primary(), _sources(), _check()])
    assert len(provider.requests) == 3
    assert not study.open_questions
    assert extraction_blocks(study) == ({}, {})


def test_a_baseline_location_is_not_a_second_measure(paper, protocol):
    """Only `value` locations are candidates for the outcome's measure — a baseline series with a
    different metric beside them is not a rival operationalization."""
    baseline = dict(FIG_SOURCE, locator="aligned-cursor baseline", analysis_metric="endpoint",
                    role="baseline", figure_id="", kind="text_mean_sd", page=4)
    outcome = _outcome(sources=(dict(FIG_SOURCE, analysis_metric="change_from_baseline"),
                                baseline))
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(outcomes=[outcome]), _check()])
    assert len(provider.requests) == 3 and not study.open_questions


def test_an_alternate_role_is_not_a_context_role(paper, protocol):
    """The `SourceRole` audit C6 asks for: `alternate` is its own answer, and the one consumer of
    the field (extraction) reads neither it nor `context` for a value."""
    from canopy.models import Source as SourceModel, SourceRole

    assert set(get_args(SourceRole)) == {"value", "baseline", "context", "alternate", "unknown"}
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_two_measure_outcome()]), _check(),
                        _adjudication(measure_rulings=_measure_ruling())])
    demoted = next(s for s in study.datasets[0].outcomes[0].sources if s.role == "alternate")
    assert demoted.role != "context"                       # kept as its own thing on the record
    readable = readable_sources(study.datasets[0].outcomes[0].sources)   # the rule, not a copy
    assert demoted not in readable and len(readable) == 1
    assert SourceModel(kind=demoted.kind, page=demoted.page).role == "value"   # default unchanged


# ------------------------------- C6 vs 4c59649: one measure written twice is not two measures
def _value_source(metric, locator, figure_id=None, kind=SourceKind.figure_bar):
    return Source(kind=kind, page=6, locator=locator, figure_id=figure_id,
                  analysis_metric=metric, role="value")


def _cressman_aftereffect() -> OutcomeSources:
    """Transcribed verbatim from `runs/rerun-fixed/papers/5039533c85ef/map.json`, `d1 aftereffect`.

    Fig. 3b's two bars are read twice — the left axis in degrees, the right axis as a percentage
    of the 30° distortion — and two further locations report the percentage form elsewhere.
    """
    return OutcomeSources(
        outcome_key="aftereffect",
        measure_name="Hand deviation at peak velocity on no-cursor (aftereffect) reaches, "
                     "expressed in degrees and as a percentage of the 30° distortion, with "
                     "aligned-cursor performance as baseline",
        operationalization="First set of 12 no-cursor reaches completed immediately after the 99 "
                           "reach training trials",
        analysis_metric="baseline_corrected",
        sources=[
            _value_source("baseline_corrected",
                          "Fig. 3b, left bar (Young, filled black) and right bar (Elderly, open "
                          "white); left y-axis 'Aftereffects at Peak Velocity (deg)'", "fig03"),
            _value_source("percent_of_perturbation",
                          "Fig. 3b, same two bars read against the right-hand y-axis "
                          "'Aftereffects at Peak Velocity (%)'", "fig03"),
            _value_source("baseline_corrected",
                          "Results, Reaching performance, final paragraph (aftereffect trials)",
                          kind=SourceKind.test_statistic),
            _value_source("percent_of_perturbation",
                          "Fig. 5, x-axis '% Visuomotor Adaptation' — individual subjects",
                          "fig05", kind=SourceKind.figure_points),
            _value_source("percent_of_perturbation",
                          "Results, Visuomotor adaptation versus proprioceptive recalibration, "
                          "first paragraph", kind=SourceKind.text_mean_sd),
        ])


def _heuer_late_adaptation() -> OutcomeSources:
    """Transcribed from `runs/rerun-fixed/papers/3570e4ce2a9c/map.json`, `d1 late_adaptation`."""
    return OutcomeSources(
        outcome_key="late_adaptation",
        measure_name="Adaptive shift (posttest−pretest difference of final movement direction) in "
                     "the open-loop test with cued presence of the 75° rotation; alternatively "
                     "initial direction error in the last practice block",
        operationalization="Two candidate operationalizations. (a) Adaptive shift: difference "
                           "between the visual open-loop posttest and the open-loop pretest. "
                           "(b) initial direction error in the last practice block.",
        analysis_metric="change_from_baseline",
        sources=[
            _value_source("endpoint", "Results, 'Practice' paragraph, first two sentences",
                          kind=SourceKind.unknown),
            _value_source("change_from_baseline",
                          "Figure 2, panel a ('adaptive shift'), mean adaptive shift at each of "
                          "the 8 target directions", "fig02", kind=SourceKind.figure_line),
        ])


def test_the_same_bars_read_against_two_axes_are_one_measure():
    """Controller ruling (corrected): "one measure in two units" is the SAME plotted element read
    against its other axis, and nothing wider.

    Cressman's Fig. 3b prints the two aftereffect bars against a degree axis on the left and a
    percent axis on the right: same `figure_id`, same panel, one measurement written twice, which
    is what the verify rule `unit_other_expression` (`4c59649`) treats as corroboration.
    """
    outcome = _cressman_aftereffect()
    same_bars = [s for s in outcome.sources if s.figure_id == "fig03"]
    assert [measure_of(s, outcome) for s in same_bars] == ["baseline_corrected"] * 2


def test_a_percentage_printed_somewhere_else_is_not_folded_into_the_reading_beside_it():
    """Branch (b) is DROPPED: a percentage is folded into a quantity only where the paper itself
    puts them on the same marks. Cressman's Fig. 5 and its p8 sentence print a percentage with
    nothing tying it to the Fig. 3b bars, so the cell now carries two candidate measures and asks
    — a question on a currently pooled cell (`5039533c85ef:d1 aftereffect`, d = -0.1179).
    """
    outcome = _cressman_aftereffect()
    assert named_alternatives(outcome) == ""              # the map's words name no alternative
    assert split_metrics(outcome) == ["baseline_corrected", "percent_of_perturbation"]
    study = StudyMap(paper_id="5039533c85ef",
                     datasets=[DatasetSpec(dataset_id="5039533c85ef:d1", outcomes=[outcome])])
    conflicts, disagreements = _Conflicts(), []
    _diff_measures(study, conflicts, disagreements)
    assert conflicts.measures == [(0, "aftereffect")]
    assert "two candidate measures" in disagreements[0]


def test_two_different_quantities_are_still_a_which_measure():
    """Heuer d1: a practice-block error and an adaptive shift are two quantities, in two places."""
    outcome = _heuer_late_adaptation()
    assert named_alternatives(outcome) == "alternatively"
    assert split_metrics(outcome) == ["change_from_baseline", "endpoint"]
    study = StudyMap(paper_id="3570e4ce2a9c",
                     datasets=[DatasetSpec(dataset_id="3570e4ce2a9c:d1", outcomes=[outcome])])
    conflicts, disagreements = _Conflicts(), []
    _diff_measures(study, conflicts, disagreements)
    assert conflicts.measures == [(0, "late_adaptation")]
    assert "two candidate measures" in disagreements[0]


def _bock_late_adaptation() -> OutcomeSources:
    """Transcribed from `runs/rerun-fixed/papers/b511dbb76fa6/map.json`, `d1 late_adaptation`.

    One quantity — the pointing error at the end of adaptation — printed in degrees off Fig. 1 and
    again as a fraction of the error the rotation imposed ("A=(I−F)/I"), in prose, on another page.
    """
    return OutcomeSources(
        outcome_key="late_adaptation",
        measure_name="Angular pointing error (median initial direction error per episode)",
        operationalization="Median initial movement direction error per 50-s episode; late "
                           "adaptation = end of the 15 adaptation episodes with +60° rotated "
                           "feedback (up to pointing episode 20 in Fig. 1).",
        analysis_metric="endpoint",
        sources=[
            _value_source("endpoint", "Fig. 1, last adaptation episode (pointing episode 20), "
                                      "filled triangles = young, filled squares = old", "fig01",
                          kind=SourceKind.figure_line),
            _value_source("endpoint", "Results, right column, ANOVA on the adaptation phase "
                                      "(between-factor Age, within-factor Episode)",
                          kind=SourceKind.test_statistic),
            _value_source("endpoint", "Results, right column, sentence reporting exponential fits "
                                      "to the adaptation curves", kind=SourceKind.unknown),
            _value_source("percent_of_perturbation",
                          "Results, left column, pooled-seniors analysis of adaptation magnitude "
                          "A=(I−F)/I", kind=SourceKind.text_mean_sd),
            _value_source("unknown", "Results, episode 20 vs 26 ANOVA",
                          kind=SourceKind.test_statistic),
        ])


def test_the_two_axes_of_one_panel_are_the_same_element_on_the_real_record():
    """Follow-up 2: is the Cressman aftereffect cell firing because branch (a) MISSES the two
    axes of Fig. 3b? No — it matches them.

    On `runs/rerun-fixed/papers/5039533c85ef/map.json` both readings carry `figure_id: fig03` and
    a locator whose panel qualifier parses to `b`, so `_element_key` is `('fig03', 'b')` for both
    and `measure_of` folds the percent reading into `baseline_corrected`. What makes the cell fire
    is the OTHER two percentage locations: Fig. 5 (a different figure, `('fig05', '')`) and a
    prose sentence with no figure at all (`None`). Those are not the same marks by any spelling,
    so the cell carries two candidate measures and the map settles it before extraction.
    """
    from canopy.agents.mapper import _element_key

    outcome = _cressman_aftereffect()
    keys = {s.locator[:20]: _element_key(s) for s in outcome.sources}
    fig3b = [s for s in outcome.sources if s.figure_id == "fig03"]
    assert len(fig3b) == 2
    assert _element_key(fig3b[0]) == _element_key(fig3b[1]) == ("fig03", "b")
    assert [measure_of(s, outcome) for s in fig3b] == ["baseline_corrected"] * 2
    elsewhere = [s for s in outcome.sources
                 if s.analysis_metric == "percent_of_perturbation" and s.figure_id != "fig03"]
    assert len(elsewhere) == 2 and all(_element_key(s) != ("fig03", "b") for s in elsewhere)
    assert [measure_of(s, outcome) for s in elsewhere] == ["percent_of_perturbation"] * 2
    assert keys  # the whole element map, for the reader of a failure


def test_a_panel_named_only_in_the_locator_still_matches_its_twin():
    """The matcher does not depend on how the two agents spelled the panel: "Fig. 3b" and
    "Figure 3, panel b" normalise to the same element, and so does a bare figure id with no panel
    named on either side."""
    from canopy.agents.mapper import _element_key

    a = _value_source("baseline_corrected", "Fig. 3b, left and right bars", "fig03")
    b = _value_source("percent_of_perturbation", "Figure 3, panel b, right-hand axis", "fig03")
    assert _element_key(a) == _element_key(b)
    outcome = OutcomeSources(outcome_key="aftereffect", sources=[a, b])
    assert split_metrics(outcome) == ["baseline_corrected"]
    plain_a = _value_source("endpoint", "Fig. 2, both series", "fig02")
    plain_b = _value_source("percent_of_perturbation", "Fig 2 right axis", "fig02")
    assert _element_key(plain_a) == _element_key(plain_b)
    assert split_metrics(OutcomeSources(outcome_key="o", sources=[plain_a, plain_b])) == ["endpoint"]


def test_a_prose_percentage_of_another_quantity_is_a_second_measure():
    """Bock d1, under the corrected C6 ruling: `A = (I - F)/I` is NOT a unit re-expression of the
    pointing error plotted in Fig. 1.

    Its denominator is the measured initial error, not the imposed rotation, so `d(A) != d(F)`
    whenever `I` differs between the groups; and the sentence computes it over the pooled seniors,
    not over the two groups Fig. 1 plots. Branch (b) — "the outcome's only other quantity" —
    merged them on the absence of a third candidate. It is gone: C6 fires, the adjudicator gets
    first crack, and a toss-up asks. Cost: a question on a currently pooled cell (d = -1.6306).
    """
    outcome = _bock_late_adaptation()
    assert named_alternatives(outcome) == ""
    assert split_metrics(outcome) == ["endpoint", "percent_of_perturbation"]
    study = StudyMap(paper_id="b511dbb76fa6",
                     datasets=[DatasetSpec(dataset_id="b511dbb76fa6:d1", outcomes=[outcome])])
    conflicts, disagreements = _Conflicts(), []
    _diff_measures(study, conflicts, disagreements)
    assert conflicts.measures == [(0, "late_adaptation")]


def test_a_percentage_beside_two_quantities_is_not_folded_into_either():
    """The collapse is a unit rule, not a licence to merge: with two quantities in question, a
    percentage nothing ties to either of them stays its own candidate and the question stays open.
    """
    outcome = OutcomeSources(
        outcome_key="late_adaptation", analysis_metric="endpoint",
        sources=[_value_source("endpoint", "Fig. 1, episode 20", "fig01"),
                 _value_source("change_from_baseline", "Fig. 2, the posttest−pretest difference",
                               "fig02"),
                 _value_source("percent_of_perturbation", "Results, adaptation magnitude",
                               kind=SourceKind.text_mean_sd)])
    assert split_metrics(outcome) == ["change_from_baseline", "endpoint",
                                      "percent_of_perturbation"]


def test_two_quantities_sharing_one_panel_do_not_collapse():
    """Only a re-expression collapses. Two different quantities plotted on one panel stay two."""
    outcome = OutcomeSources(
        outcome_key="late_adaptation", analysis_metric="endpoint",
        sources=[_value_source("endpoint", "Fig. 2a, filled squares", "fig02"),
                 _value_source("change_from_baseline", "Fig. 2a, open circles", "fig02")])
    assert split_metrics(outcome) == ["change_from_baseline", "endpoint"]


def test_a_winners_own_measure_in_its_other_expression_is_not_demoted(paper, protocol):
    """When a genuine split is settled, the winner's re-expression stays a `value` location."""
    endpoint = dict(FIG_SOURCE, locator="Results, Practice paragraph", kind="text_mean_sd",
                    figure_id="", analysis_metric="endpoint", page=4)
    change = dict(FIG_SOURCE, locator="Fig 1b, the two bars", analysis_metric="change_from_baseline")
    percent = dict(FIG_SOURCE, locator="Fig 1b, the same two bars, right-hand axis",
                   analysis_metric="percent_of_perturbation")
    outcome = _outcome(sources=(endpoint, change, percent))
    outcome["measure_name"] = ALT_MEASURE
    outcome["operationalization"] = ALT_OPERATIONALIZATION
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[outcome]), _check(),
                        _adjudication(measure_rulings=_measure_ruling())])
    roles = {s.analysis_metric: s.role for s in study.datasets[0].outcomes[0].sources
             if s.analysis_metric != "unknown"}     # the cross-check's own location records none
    assert roles == {"change_from_baseline": "value", "percent_of_perturbation": "value",
                     "endpoint": "alternate"}


#: the same plotted element read against its other axis — the only shape branch (a) collapses
OTHER_AXIS = dict(FIG_SOURCE, locator="Fig 1 adaptation episodes, right-hand percent axis",
                  analysis_metric="percent_of_perturbation")


def test_a_percentage_read_off_the_same_marks_buys_no_adjudication_call(paper, protocol):
    """The ruling's point: a unit re-expression must not buy the call that settles a real dispute.

    Three calls (study map, source map, cross-check) — the same as an outcome with one metric —
    and both locations stay readable, because they are the same marks written twice.
    """
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(outcomes=[_outcome(sources=(FIG_SOURCE,
                                                                                OTHER_AXIS))]),
                               _check()])
    assert len(provider.requests) == 3
    assert {s.role for s in study.datasets[0].outcomes[0].sources} == {"value"}
    assert not study.open_questions and extraction_blocks(study) == ({}, {})


def test_a_percentage_printed_in_prose_does_buy_the_call(paper, protocol):
    """…and the mirror of it, which is what the corrected ruling changed: a percentage the paper
    prints somewhere else is a second candidate measure, and the map settles it before extracting.
    """
    percent = dict(FIG_SOURCE, locator="Results, adaptation magnitude A=(I-F)/I",
                   kind="text_mean_sd", figure_id="", page=4,
                   analysis_metric="percent_of_perturbation")
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(outcomes=[_outcome(sources=(FIG_SOURCE,
                                                                                percent))]),
                               _check(),
                               _adjudication(measure_rulings=_measure_ruling(verdict="toss_up"))])
    assert len(provider.requests) == 4                    # ONE adjudication call, then a question
    assert [q.kind for q in study.open_questions] == ["which_measure"]
    assert split_metrics(study.datasets[0].outcomes[0]) == ["endpoint", "percent_of_perturbation"]


def test_the_mappers_own_words_open_the_question_whatever_the_metrics_are(paper, protocol):
    """The unit rule silences the metric half of the trigger, never the mapper's own words."""
    outcome = _outcome(sources=(FIG_SOURCE, OTHER_AXIS))
    outcome["measure_name"] = ALT_MEASURE
    outcome["operationalization"] = ALT_OPERATIONALIZATION
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(outcomes=[outcome]), _check(),
                               _adjudication(measure_rulings=_measure_ruling(verdict="toss_up"))])
    assert len(provider.requests) == 4
    assert [q.kind for q in study.open_questions] == ["which_measure"]
    assert split_metrics(study.datasets[0].outcomes[0]) == ["endpoint"]      # one measure, asked


# ------------------------------- F2: a which_measure decision is settled by LOCATION
HEUER_D2_MEASURE = ("Adaptive shift (posttest−pretest difference of final movement direction) in "
                    "the open-loop test with cued presence of the 30° rotation; alternatively "
                    "initial direction error across the five practice blocks")
HEUER_D2_OPERATIONALIZATION = (
    "Two candidate operationalizations. (a) Adaptive shift: difference between the visual "
    "open-loop posttest with cued presence of the 30° CCW rotation and the open-loop pretest, in "
    "final direction of the hand movement. (b) Initial direction error in the last of the five "
    "practice blocks with the rotation present.")
HEUER_D2_FIGURE = ("Figure 6, panel a ('adaptive shift'), mean adaptive shift at target "
                   "directions 0, 45, 90, 135, 180°; filled circles = young, open circles = old")
HEUER_D2_PROSE = ("Results, 'Tests' paragraph, adaptive shifts broken down by target amplitude "
                  "and age")


def _heuer_d2_outcome() -> dict:
    """`3570e4ce2a9c:d2 late_adaptation`, transcribed from `runs/rerun-fixed`.

    Two candidate operationalizations in the map's own words, and BOTH `value` locations carry
    `analysis_metric: change_from_baseline` — so a ruling that names the metric names both of
    them at once and demotes nothing.
    """
    figure = dict(FIG_SOURCE, kind="figure_line", page=10, figure_id="fig02",
                  locator=HEUER_D2_FIGURE, analysis_metric="change_from_baseline")
    prose = dict(FIG_SOURCE, kind="unknown", page=10, figure_id="",
                 locator=HEUER_D2_PROSE, analysis_metric="change_from_baseline")
    outcome = _outcome(sources=(figure, prose))
    outcome["measure_name"] = HEUER_D2_MEASURE
    outcome["operationalization"] = HEUER_D2_OPERATIONALIZATION
    outcome["analysis_metric"] = "change_from_baseline"
    return outcome


def _heuer_d2_map(paper, protocol, ruling):
    return _mapped(paper, protocol,
                   [_primary(), _sources(outcomes=[_heuer_d2_outcome()]), _check(),
                    _adjudication(measure_rulings=ruling)])


def test_a_winner_ruling_that_demotes_no_location_has_settled_nothing(paper, protocol):
    """F2, on the real Heuer d2 record: both measures share one `analysis_metric`.

    A fully valid `winner` ruling for `change_from_baseline` used to write "adjudicated …
    which_measure: change_from_baseline (0 location(s) demoted to alternate)", close the question
    and leave BOTH operationalizations readable — the run then extracts the two the question
    exists to choose between. A settlement that demotes nothing has settled nothing.
    """
    study, _ = _heuer_d2_map(paper, protocol, _measure_ruling())
    outcome = study.datasets[0].outcomes[0]
    assert {s.role for s in outcome.sources} == {"value"}          # nothing was demoted…
    assert [q.kind for q in study.open_questions] == ["which_measure"]      # …so nothing settled
    _, cells = extraction_blocks(study)
    assert (study.datasets[0].dataset_id, "late_adaptation") in cells
    assert [d for d in study.disagreements if "which_measure" in d and "demoted no" in d]


def test_a_ruling_that_names_the_winning_location_demotes_the_loser_by_location(paper, protocol):
    """The same record, settled the way the schema now allows: by naming WHERE the winner is."""
    study, _ = _heuer_d2_map(paper, protocol,
                             _measure_ruling(winning_location="fig02"))
    outcome = study.datasets[0].outcomes[0]
    roles = {s.locator: s.role for s in outcome.sources}
    assert roles[HEUER_D2_FIGURE] == "value" and roles[HEUER_D2_PROSE] == "alternate"
    assert not study.open_questions and extraction_blocks(study) == ({}, {})
    loser = next(s for s in outcome.sources if s.role == "alternate")
    assert "demoted to alternate" in loser.notes


def test_a_ruling_may_name_the_losing_locations_instead(paper, protocol):
    study, _ = _heuer_d2_map(paper, protocol,
                             _measure_ruling(losing_locations=[HEUER_D2_PROSE]))
    outcome = study.datasets[0].outcomes[0]
    roles = {s.locator: s.role for s in outcome.sources}
    assert roles[HEUER_D2_FIGURE] == "value" and roles[HEUER_D2_PROSE] == "alternate"
    assert not study.open_questions


def test_a_human_answer_that_names_a_location_demotes_by_location_not_by_metric(paper, protocol):
    """The review UI offers one option per (metric, location) pair and its answer carries BOTH.
    On a same-metric split the location is the only half that can settle it, so it decides."""
    study, _ = _heuer_d2_map(paper, protocol, _measure_ruling(verdict="toss_up"))
    assert [q.kind for q in study.open_questions] == ["which_measure"]

    answered = apply_map_answers(study, [
        {"kind": "which_measure", "paper_id": study.paper_id,
         "dataset_id": study.datasets[0].dataset_id, "outcome_key": "late_adaptation",
         "winning_analysis_metric": "change_from_baseline",
         "winning_location": HEUER_D2_FIGURE, "note": "panel a wins"}])
    outcome = answered.datasets[0].outcomes[0]
    roles = {s.locator: s.role for s in outcome.sources}
    assert roles[HEUER_D2_FIGURE] == "value" and roles[HEUER_D2_PROSE] == "alternate"
    assert not answered.open_questions and extraction_blocks(answered) == ({}, {})
    assert "a human reviewer" in outcome.measure_ruling


def test_a_human_answer_that_can_only_name_the_shared_metric_leaves_the_question_open(paper,
                                                                                     protocol):
    """The human path fails closed exactly where the ruling path does."""
    study, _ = _heuer_d2_map(paper, protocol, _measure_ruling(verdict="toss_up"))
    answered = apply_map_answers(study, [
        {"kind": "which_measure", "paper_id": study.paper_id,
         "dataset_id": study.datasets[0].dataset_id, "outcome_key": "late_adaptation",
         "winning_analysis_metric": "change_from_baseline", "note": "the adaptive shift"}])
    assert [q.kind for q in answered.open_questions] == ["which_measure"]
    assert {s.role for s in answered.datasets[0].outcomes[0].sources} == {"value"}


def test_a_ruling_that_names_the_percentage_form_keeps_both_expressions(paper, protocol):
    """A reading and the same marks in another unit are one measure, so a ruling for either name
    settles the same question and demotes nothing that reads that measure."""
    endpoint = dict(FIG_SOURCE, analysis_metric="endpoint")
    other = dict(FIG_SOURCE, locator="Results, Practice paragraph", kind="text_mean_sd",
                 figure_id="", page=4, analysis_metric="change_from_baseline")
    outcome = _outcome(sources=(endpoint, dict(OTHER_AXIS), other))
    outcome["measure_name"] = ALT_MEASURE
    outcome["operationalization"] = ALT_OPERATIONALIZATION
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[outcome]), _check(),
                        _adjudication(measure_rulings=_measure_ruling(
                            metric="percent_of_perturbation"))])
    roles = {s.analysis_metric: s.role for s in study.datasets[0].outcomes[0].sources
             if s.analysis_metric != "unknown"}
    assert roles == {"endpoint": "value", "percent_of_perturbation": "value",
                     "change_from_baseline": "alternate"}
    assert not study.open_questions


# ------------------------------- F6: a location that cannot say which measure it reads
def test_a_value_location_with_no_metric_is_a_second_candidate_measure():
    """F6: `_value_metrics` dropped `unknown`-metric locations, so an outcome whose second measure
    carries no metric never opened a `which_measure` question at all — while extraction read it
    (`run.py` reads `role in ("value", "unknown")`). A location that cannot say what it measures
    is unresolved, not agreement.
    """
    outcome = OutcomeSources(
        outcome_key="late_adaptation", analysis_metric="endpoint",
        sources=[_value_source("endpoint", "Fig. 1, episode 20", "fig01"),
                 _value_source("unknown", "Results, episode 20 vs 26 ANOVA",
                               kind=SourceKind.test_statistic)])
    study = StudyMap(paper_id="b511dbb76fa6",
                     datasets=[DatasetSpec(dataset_id="b511dbb76fa6:d1", outcomes=[outcome])])
    conflicts, disagreements = _Conflicts(), []
    _diff_measures(study, conflicts, disagreements)
    assert conflicts.measures == [(0, "late_adaptation")]
    assert "cannot say which measure" in disagreements[0]


def test_locations_that_all_carry_one_measure_and_no_metricless_sibling_still_ask_nothing():
    outcome = OutcomeSources(
        outcome_key="late_adaptation", analysis_metric="endpoint",
        sources=[_value_source("endpoint", "Fig. 1, episode 20", "fig01"),
                 _value_source("endpoint", "Results, the same episode",
                               kind=SourceKind.text_mean_sd)])
    study = StudyMap(paper_id="x", datasets=[DatasetSpec(dataset_id="x:d1", outcomes=[outcome])])
    conflicts, disagreements = _Conflicts(), []
    _diff_measures(study, conflicts, disagreements)
    assert conflicts.measures == [] and disagreements == []


def test_a_ruling_withholds_the_locations_that_cannot_say_which_measure_they_read(paper, protocol):
    """The real Heuer d1 record: after a winner ruling for `change_from_baseline` the losing
    operationalization's own paragraph was still read, through the sibling location whose metric
    the mapper left blank. C6's "one measure extracted" was not delivered on the record it was
    derived from.
    """
    endpoint = dict(FIG_SOURCE, locator="Results, 'Practice' paragraph, first two sentences",
                    kind="unknown", figure_id="", page=4, analysis_metric="endpoint")
    change = dict(FIG_SOURCE, locator="Figure 2, panel a ('adaptive shift')", figure_id="fig02",
                  analysis_metric="change_from_baseline")
    blank_a = dict(FIG_SOURCE, locator="Results, Tests paragraph", kind="unknown", figure_id="",
                   page=4, analysis_metric="unknown")
    blank_b = dict(FIG_SOURCE, locator="Practice paragraph", kind="unknown", figure_id="",
                   page=4, analysis_metric="unknown")
    outcome = _outcome(sources=(endpoint, change, blank_a, blank_b))
    outcome["measure_name"] = ALT_MEASURE
    outcome["operationalization"] = ALT_OPERATIONALIZATION
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[outcome]), _check(),
                        _adjudication(measure_rulings=_measure_ruling())])
    sources = study.datasets[0].outcomes[0].sources
    assert not study.open_questions                        # a winner ruling settled it
    readable = readable_sources(sources)
    assert [s.analysis_metric for s in readable] == ["change_from_baseline"]
    withheld = [s for s in sources if s.analysis_metric == "unknown"]
    assert withheld and all("cannot say which measure" in s.notes for s in withheld)


def test_the_cross_check_says_what_a_location_it_adds_measures(paper, protocol):
    """Follow-up 1: the schema gap behind two of F6's four new firings.

    `MAPPER_CROSSCHECK_SCHEMA` never asked the second agent what a location measures, so EVERY
    location it added arrived `analysis_metric: unknown` — an unresolved candidate that opens a
    `which_measure` question on a cell nobody disputed. Asked and answered, the added location is
    just another reading of the outcome's own measure and nothing is put in question.
    """
    source = MAPPER_CROSSCHECK_SCHEMA["properties"]["datasets"]["items"]["properties"]["outcomes"] \
        ["items"]["properties"]["sources"]["items"]
    assert "analysis_metric" in source["required"]
    assert set(source["properties"]["analysis_metric"]["enum"]) == set(get_args(AnalysisMetric))

    added = _check_source(kind="text_mean_sd", page=4, figure_id="",
                          locator="Results, the same episode printed in prose",
                          analysis_metric="endpoint")
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 12, "n_evidence": "twelve old"},
        "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"},
        "outcomes": [{"outcome_key": "late_adaptation",
                      "sources": [_check_source(), added]}]}])
    study, provider = _mapped(paper, protocol, [_primary(), _sources(), check])
    outcome = study.datasets[0].outcomes[0]
    metrics = {s.locator: s.analysis_metric for s in outcome.sources}
    assert metrics["Results, the same episode printed in prose"] == "endpoint"   # it said so
    assert len(provider.requests) == 3                    # …so no adjudication was bought
    assert not study.open_questions and extraction_blocks(study) == ({}, {})


def test_a_cross_check_location_that_still_names_no_measure_opens_the_question(paper, protocol):
    """`unknown` remains an honest answer, and it remains an unresolved candidate — the control
    for the test above, and the shape the recorded maps carry."""
    added = _check_source(kind="text_mean_sd", page=4, figure_id="",
                          locator="Results, the same episode printed in prose")
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 12, "n_evidence": "twelve old"},
        "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"},
        "outcomes": [{"outcome_key": "late_adaptation",
                      "sources": [_check_source(), added]}]}])
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(), check,
                               _adjudication(measure_rulings=_measure_ruling(verdict="toss_up"))])
    assert len(provider.requests) == 4
    assert [q.kind for q in study.open_questions] == ["which_measure"]
    assert [d for d in study.disagreements if "cannot say which measure" in d]


# ------------------------------- whose numbers are at this location? (`Source.sample`)
def test_a_value_location_reporting_a_pooled_sample_is_not_read_for_the_cell():
    """The reviewer's second Bock hole, as a general field: `A=(I-F)/I` is computed over the
    POOLED seniors, not over the two groups Fig. 1 plots, and nothing in the map could say so.
    A `value` location whose sample is explicitly not this contrast's two groups stays on the
    record — it is evidence about the paper — and is not read for the cell's number.
    """
    pooled = Source(kind=SourceKind.text_mean_sd, page=4, role="value", sample="pooled",
                    locator="Results, pooled-seniors analysis of adaptation magnitude",
                    sample_note="the 30 seniors of both groups taken together")
    both = Source(kind=SourceKind.figure_line, page=3, role="value", sample="both_groups",
                  locator="Fig. 1, last adaptation episode")
    silent = Source(kind=SourceKind.figure_line, page=3, role="value",
                    locator="Fig. 2, last episode")
    assert readable_sources([pooled, both, silent]) == [both, silent]
    assert "pooled" in source_unreadable_reason(pooled)
    assert source_unreadable_reason(both) == "" and source_unreadable_reason(silent) == ""
    assert silent.sample == "unknown"                      # the honest default reads


@pytest.mark.parametrize("sample,readable", [("both_groups", True), ("unknown", True),
                                             # a panel that plots ONE age group is the ordinary
                                             # layout (Fig. 1A young / Fig. 1B old): it is read for
                                             # the group it carries — the first nine-paper run
                                             # read `one_group` as unreadable and extracted nothing
                                             ("one_group", True), ("pooled", False),
                                             ("other", False)])
def test_every_sample_answer_decides_readability_the_same_way(sample, readable):
    source = Source(kind=SourceKind.text_mean_sd, page=4, role="value", sample=sample,
                    locator="Results")
    assert bool(readable_sources([source])) is readable
    assert (source_unreadable_reason(source) == "") is readable


def test_a_demoted_or_baseline_location_is_not_readable_either():
    """One helper, one rule: the role filter `run._extract_cell` applies lives here too, so the
    two reasons a location is not read are decided in one place."""
    for role in ("baseline", "context", "alternate"):
        source = Source(kind=SourceKind.figure_line, page=3, role=role, locator="Fig 1")
        assert readable_sources([source]) == []
        assert role in source_unreadable_reason(source)
    for role in ("value", "unknown"):
        assert readable_sources([Source(kind=SourceKind.figure_line, page=3, role=role)])


def test_the_source_schema_and_the_prompt_ask_whose_numbers_a_location_reports():
    from canopy.models import Source as SourceModel

    source = MAPPER_SOURCES_SCHEMA["properties"]["datasets"]["items"]["properties"]["outcomes"] \
        ["items"]["properties"]["sources"]["items"]
    assert "sample" in source["required"] and "sample_note" in source["required"]
    assert set(source["properties"]["sample"]["enum"]) == {"both_groups", "one_group", "pooled",
                                                           "other", "unknown"}
    assert set(source["properties"]) <= set(SourceModel.model_fields)
    text = load_prompt("mapper_sources")
    assert "`sample`" in text and "both_groups" in text and "pooled" in text
    assert "sample_note" in text


# --------------------------------- a human's answer to a map question (C6/C7, questions area)
def _open_inclusion(paper, protocol) -> StudyMap:
    """A map whose second dataset only one agent proposed, with no rule cited to reject it."""
    study, _ = _mapped(paper, protocol,
                       [_primary(datasets=_two_datasets()), _sources(), _check(),
                        _adjudication(dataset_inclusions=[])])
    return study


def _open_measure(paper, protocol) -> StudyMap:
    """A map whose late_adaptation outcome carries two measures nobody could rule between."""
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_two_measure_outcome()]), _check(),
                        _adjudication(measure_rulings=_measure_ruling(verdict="toss_up"))])
    return study


def _include_answer(study, decision="include", **over):
    answer = {"kind": "include_dataset", "paper_id": study.paper_id,
              "dataset_id": study.datasets[1].dataset_id, "decision": decision,
              "rule": "", "quote": "", "note": "the control sample did another task"}
    answer.update(over)
    return answer


def _measure_answer(study, **over):
    answer = {"kind": "which_measure", "paper_id": study.paper_id,
              "dataset_id": study.datasets[0].dataset_id, "outcome_key": "late_adaptation",
              "winning_analysis_metric": "change_from_baseline",
              "note": "the window names the open-loop posttest"}
    answer.update(over)
    return answer


def test_open_map_questions_lists_what_the_map_could_not_settle(paper, protocol):
    study = _open_inclusion(paper, protocol)
    assert [q.kind for q in open_map_questions(study)] == ["include_dataset"]
    assert open_map_questions(StudyMap(paper_id="x")) == []


def test_a_human_include_answer_unblocks_the_dataset(paper, protocol):
    study = _open_inclusion(paper, protocol)
    blocked, _ = extraction_blocks(study)
    assert study.datasets[1].dataset_id in blocked           # blocked before

    answered = apply_map_answers(study, [_include_answer(study)])
    assert extraction_blocks(answered) == ({}, {})           # not blocked after
    assert answered.datasets[1].included is True
    assert not answered.open_questions
    assert not [f for f in answered.needs_human if "inclusion needs human" in f]
    assert "control sample did another task" in answered.datasets[1].notes


def test_a_human_exclude_answer_excludes_the_dataset_with_its_rule(paper, protocol):
    study = _open_inclusion(paper, protocol)
    answered = apply_map_answers(study, [_include_answer(
        study, decision="exclude", rule="Dataset Rule 3",
        quote="never performed the pointing adaptation task")])
    excluded = answered.datasets[1]
    assert excluded.included is False and excluded.exclusion_rule == "Dataset Rule 3"
    assert "pointing adaptation task" in excluded.exclusion_quote
    assert not answered.open_questions
    blocked, cells = extraction_blocks(answered)
    assert "Dataset Rule 3" in blocked[excluded.dataset_id] and not cells


def test_a_human_may_exclude_without_a_rule_and_the_record_says_so(paper, protocol):
    """A person does not have to cite the protocol at an adjudicator's standard — but the record
    must not claim a rule that was never named."""
    study = _open_inclusion(paper, protocol)
    answered = apply_map_answers(study, [_include_answer(study, decision="exclude")])
    assert answered.datasets[1].included is False
    assert answered.datasets[1].exclusion_rule == "human decision"


def test_a_human_which_measure_answer_demotes_the_losing_locations(paper, protocol):
    study = _open_measure(paper, protocol)
    _, cells = extraction_blocks(study)
    assert (study.datasets[0].dataset_id, "late_adaptation") in cells      # blocked before

    answered = apply_map_answers(study, [_measure_answer(study)])
    outcome = answered.datasets[0].outcomes[0]
    roles = {s.analysis_metric: s.role for s in outcome.sources if s.analysis_metric != "unknown"}
    assert roles == {"change_from_baseline": "value", "endpoint": "alternate"}
    assert outcome.analysis_metric == "change_from_baseline"
    assert "human" in outcome.measure_ruling and "open-loop posttest" in outcome.measure_ruling
    loser = next(s for s in outcome.sources if s.role == "alternate")
    assert "a human" in loser.notes
    assert not answered.open_questions and extraction_blocks(answered) == ({}, {})
    assert not [f for f in answered.needs_human if "which_measure needs human" in f]


def test_a_which_measure_answer_can_name_the_winning_location_instead_of_the_metric(paper,
                                                                                    protocol):
    study = _open_measure(paper, protocol)
    winner = next(s for s in study.datasets[0].outcomes[0].sources
                  if s.analysis_metric == "endpoint")
    answered = apply_map_answers(study, [_measure_answer(study, winning_analysis_metric="",
                                                         winning_location=winner.locator)])
    outcome = answered.datasets[0].outcomes[0]
    roles = {s.analysis_metric: s.role for s in outcome.sources if s.analysis_metric != "unknown"}
    assert roles == {"endpoint": "value", "change_from_baseline": "alternate"}
    assert outcome.analysis_metric == "endpoint"


def test_an_answer_naming_a_measure_no_value_location_carries_is_ignored(paper, protocol):
    """Never invent: an answer the map cannot ground changes nothing;
    the question stays open."""
    study = _open_measure(paper, protocol)
    unchanged = apply_map_answers(study, [
        _measure_answer(study, winning_analysis_metric="percent_of_perturbation"),
        _measure_answer(study, winning_analysis_metric="", winning_location="Table 9, row 4"),
        _measure_answer(study, winning_analysis_metric="", winning_location=""),
    ])
    assert unchanged.model_dump() == study.model_dump()
    assert [q.kind for q in unchanged.open_questions] == ["which_measure"]


def test_answers_for_another_study_another_dataset_or_another_kind_are_ignored(paper, protocol):
    study = _open_inclusion(paper, protocol)
    unchanged = apply_map_answers(study, [
        _include_answer(study, paper_id="0" * 64),
        _include_answer(study, dataset_id="nothing:d9"),
        _include_answer(study, kind="value"),
        _include_answer(study, decision="maybe"),
        _include_answer(study, dataset_id=study.datasets[0].dataset_id),   # no question on d1
        {"note": "not an answer at all"},
    ])
    assert unchanged.model_dump() == study.model_dump()


def test_apply_map_answers_never_mutates_the_study_it_was_given(paper, protocol):
    study = _open_inclusion(paper, protocol)
    before = study.model_dump()
    answered = apply_map_answers(study, [_include_answer(study, decision="exclude",
                                                         rule="Dataset Rule 3", quote="tracking")])
    assert study.model_dump() == before                      # the input is untouched
    assert answered is not study and answered.datasets[1].included is False


def test_the_last_answer_to_one_question_is_the_one_that_stands(paper, protocol):
    """A reviewer who changes their mind appends a second answer; the log is never rewritten."""
    study = _open_inclusion(paper, protocol)
    answered = apply_map_answers(study, [_include_answer(study, decision="exclude",
                                                         rule="Dataset Rule 3", quote="tracking"),
                                         _include_answer(study, decision="include")])
    assert answered.datasets[1].included is True and not answered.open_questions
    assert extraction_blocks(answered) == ({}, {})


def test_a_short_paper_id_is_the_same_study(paper, protocol):
    """The run addresses papers by their 12-character prefix everywhere a dataset id appears."""
    study = _open_inclusion(paper, protocol)
    answered = apply_map_answers(study, [_include_answer(study, paper_id=study.paper_id[:12])])
    assert not answered.open_questions


def test_a_second_measure_answer_is_applied_to_the_map_it_arrived_with(paper, protocol):
    """Two answers to one `which_measure` question must not compose: the second is read against
    the locations as the map recorded them, not against the demotions the first one wrote."""
    study = _open_measure(paper, protocol)
    answered = apply_map_answers(study, [_measure_answer(study),
                                         _measure_answer(study,
                                                         winning_analysis_metric="endpoint")])
    outcome = answered.datasets[0].outcomes[0]
    roles = {s.analysis_metric: s.role for s in outcome.sources if s.analysis_metric != "unknown"}
    assert roles == {"endpoint": "value", "change_from_baseline": "alternate"}
    assert outcome.analysis_metric == "endpoint"


# --------------------------------- F8 / F9: the lows the review named
def test_open_map_questions_hands_out_copies_not_the_maps_own_models(paper, protocol):
    """A pure reader that returns live models lets its caller edit the study through it."""
    study = _open_inclusion(paper, protocol)
    questions = open_map_questions(study)
    questions[0].question = "something else entirely"
    questions[0].options.append("a third answer")
    assert study.open_questions[0].question != "something else entirely"
    assert "a third answer" not in study.open_questions[0].options


def test_answering_an_inclusion_question_clears_the_flag_the_diff_wrote_too(paper, protocol):
    """Two `needs_human` lines are written for a single-mapper dataset — the question's own, and
    the "group mapping unconfirmed" line from the cross-check diff. Answering the question left
    the second one standing, so the paper still went to a human for a settled question."""
    study = _open_inclusion(paper, protocol)
    dataset_id = study.datasets[1].dataset_id
    mine = [f for f in study.needs_human
            if dataset_id in f and ("inclusion needs human" in f or "mapping unconfirmed" in f)]
    assert len(mine) == 2
    answered = apply_map_answers(study, [_include_answer(study)])
    assert [f for f in answered.needs_human if f in mine] == []
    assert study.needs_human != answered.needs_human            # the input is untouched
    assert [f for f in answered.needs_human if dataset_id in f]  # the thin-outcome flag stays


def test_a_cell_with_an_open_map_question_buys_no_extraction_call(paper, protocol, tmp_path):
    """C6, asserted on the call count rather than on the blocks dict — "the dict names the cell"
    and "no call was made" are different claims, and only one of them is about money.

    This drives `run._extract` itself with a provider that would record any request, so removing
    the gate makes the count non-zero rather than leaving the test green.
    """
    from canopy.config import MODELS as _MODELS
    from canopy.models import PaperStatus
    from canopy.pipeline.run import RunContext, _extract

    study = _open_measure(paper, protocol)
    _, cells = extraction_blocks(study)
    assert (study.datasets[0].dataset_id, "late_adaptation") in cells
    provider = FakeProvider([{"statistics": [], "notes": ""}])
    ctx = RunContext(protocol=protocol, out_dir=tmp_path,
                     client=LLMClient(provider=provider, cache_dir=None),
                     models=dict(_MODELS), resume=False)
    status = PaperStatus(paper_id=paper.sha256, filename=paper.filename)
    candidates = _extract(ctx, paper, study, status)
    assert provider.requests == [] and candidates == []
    assert [w for w in status.warnings if "not extracted" in w and "which_measure" in w]


def test_the_source_prompt_still_refuses_the_statistics_p_b_excludes():
    """P-B's `role` wording is code as far as this pipeline is concerned: it is the only thing
    that keeps a test against zero, an omnibus F or a chi-square out of the `value` slot before
    any extraction is bought. Nothing else in the suite fails if the list is edited away."""
    text = " ".join(load_prompt("mapper_sources").split())
    for phrase in ("a test of one group against zero or any constant",
                   "even when its df equal n_a + n_b",
                   "an omnibus test over more than two levels (numerator df > 1)",
                   "A χ² is never a `value` source",
                   "within-subject factors",
                   "averaged over"):
        assert phrase in text, phrase
    assert "last resort" in text                      # a `value` statistic is fifth of seven


def test_the_adjudication_prompt_asks_for_the_location_when_the_metric_cannot_settle_it():
    text = load_prompt("mapper_adjudicate")
    assert "winning_location" in text and "losing_locations" in text
    assert "same" in text.lower()                     # the same-marks rule is still stated


def test_a_location_the_map_does_not_carry_falls_back_to_the_measure_the_answer_named(paper,
                                                                                      protocol):
    """The review UI sends both halves of the option it offered. A location string the map cannot
    match (an edited or truncated locator) must not throw the whole answer away — but it settles
    only if the measure it names demotes something, which is the same rule as everywhere else."""
    study = _open_measure(paper, protocol)
    answered = apply_map_answers(study, [_measure_answer(study, winning_location="Table 9, row 4")])
    outcome = answered.datasets[0].outcomes[0]
    roles = {s.analysis_metric: s.role for s in outcome.sources if s.analysis_metric != "unknown"}
    assert roles == {"change_from_baseline": "value", "endpoint": "alternate"}
    assert not answered.open_questions


def test_an_answer_that_would_demote_every_location_settles_nothing(paper, protocol):
    """C6 demotes, it never deletes: "the measure is X" has to leave X readable somewhere. A
    ruling that names every location as losing is a contradiction, and the question stays open."""
    study, _ = _heuer_d2_map(paper, protocol, _measure_ruling(
        losing_locations=[HEUER_D2_FIGURE, HEUER_D2_PROSE]))
    assert {s.role for s in study.datasets[0].outcomes[0].sources} == {"value"}
    assert [q.kind for q in study.open_questions] == ["which_measure"]



def test_the_mappers_own_words_do_not_open_a_measure_question_it_already_settled():
    """The first nine-paper run: "DE (primary); IEE (alternative operationalization)" with IEE's
    locations already `alternate` bought an adjudication and blocked BOTH late-adaptation cells
    behind a `which_measure` question the mapper had already answered. The words open the
    question only while the map still carries two readable candidate measures."""
    from canopy.agents.mapper import _Conflicts, _diff_measures
    from canopy.models import DatasetSpec, GroupSpec, OutcomeSources, Source, SourceKind, StudyMap
    ya = Source(kind=SourceKind.figure_points, page=4, role="value", sample="one_group",
                figure_id="fig01", locator="Fig. 1A (YA 30° DE), point at x = A3",
                analysis_metric="endpoint")
    oa = Source(kind=SourceKind.figure_points, page=4, role="value", sample="one_group",
                figure_id="fig01", locator="Fig. 1B (OA 30° DE), point at x = A3",
                analysis_metric="endpoint")
    iee = Source(kind=SourceKind.figure_points, page=4, role="alternate", sample="one_group",
                 figure_id="fig01", locator="Fig. 1C (YA 30° IEE), point at x = A3",
                 analysis_metric="endpoint")
    outcome = OutcomeSources(
        outcome_key="late_adaptation",
        measure_name="direction error (DE), degrees (primary); initial endpoint error (IEE), mm "
                     "(second, alternative operationalization)",
        operationalization="Two candidate measures are plotted for the same window: DE and IEE",
        sources=[ya, oa, iee])
    dataset = DatasetSpec(dataset_id="p:d1", label="30° rotation",
                          group_a=GroupSpec(label="older", n=9), group_b=GroupSpec(label="young", n=9),
                          outcomes=[outcome])
    study = StudyMap(paper_id="p", datasets=[dataset])
    conflicts, notes = _Conflicts(), []
    _diff_measures(study, conflicts, notes)
    assert conflicts.measures == [] and notes == [], "the mapper answered its own question"
    # …but the same words with NOTHING demoted still open it (Heuer's Experiment 2 shape)
    iee_value = iee.model_copy(update={"role": "value", "analysis_metric": "change_from_baseline"})
    outcome_open = outcome.model_copy(update={"sources": [ya, oa, iee_value]})
    study_open = StudyMap(paper_id="p", datasets=[dataset.model_copy(update={"outcomes": [outcome_open]})])
    conflicts, notes = _Conflicts(), []
    _diff_measures(study_open, conflicts, notes)
    assert conflicts.measures == [(0, "late_adaptation")]


def test_one_measure_left_standing_settles_however_many_locations_read_it():
    """A young-adults panel and an older-adults panel are two LOCATIONS of one measure; a ruling
    that leaves only them readable has settled the question even though it demoted nothing."""
    from canopy.agents.mapper import read_measure_answer
    from canopy.models import OutcomeSources, Source, SourceKind
    ya = Source(kind=SourceKind.figure_points, page=4, role="value", figure_id="fig01",
                locator="Fig. 1A (YA), point at x = A3", analysis_metric="endpoint")
    oa = Source(kind=SourceKind.figure_points, page=4, role="value", figure_id="fig01",
                locator="Fig. 1B (OA), point at x = A3", analysis_metric="endpoint")
    outcome = OutcomeSources(outcome_key="late_adaptation", measure_name="direction error",
                             sources=[ya, oa])
    settlement = read_measure_answer(outcome, winning_metric="endpoint")
    assert settlement.ok and settlement.settles and settlement.losers == []
    # two measures still readable after a metric-only answer: NOT settled (the Heuer d2 rule)
    other = Source(kind=SourceKind.text_mean_sd, page=5, role="value",
                   locator="Results, adaptive shift", analysis_metric="change_from_baseline")
    two = outcome.model_copy(update={"sources": [ya, oa, other]})
    assert read_measure_answer(two, winning_metric="endpoint").settles is True  # demotes `other`
    same_metric = other.model_copy(update={"analysis_metric": "endpoint"})
    still_two = outcome.model_copy(update={"measure_name": "DE; alternatively adaptive shift",
                                           "sources": [ya, oa, same_metric]})
    # the mapper's own words opened this one and nothing was demoted: two operationalizations
    # sharing one metric are still two measures, so a metric-only answer settles NOTHING (the F2
    # rule); naming the location does
    assert read_measure_answer(still_two, winning_metric="endpoint").settles is False
    assert read_measure_answer(still_two, winning_location="Fig. 1A").settles is True


# ------------------------------- C6 REVERSAL: a measure the tool chose is a decision, not a fact
def _settled_map(paper, protocol):
    """The Heuer d1 record, settled by the adjudicator exactly as the real runs settled it."""
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_two_measure_outcome()]), _check(),
                        _adjudication(measure_rulings=_measure_ruling())])
    return study


def test_a_settled_ruling_records_what_it_set_aside_and_which_side_may_be_put_back(paper,
                                                                                   protocol):
    """The record C6 leaves is what makes its decision reversible, and it must say who made it.

    `c6_demoted` is the whole test of "may an answer put this back": a location the MAPPER marked
    `alternate` is the map's reading of the paper and nobody's decision about this review, while one
    a `which_measure` settlement demoted is a choice somebody took and somebody else may take again.
    """
    outcome = _settled_map(paper, protocol).datasets[0].outcomes[0]
    loser = next(s for s in outcome.sources if s.role == "alternate")
    assert c6_demoted(loser) and loser.analysis_metric == "endpoint"
    assert not any(c6_demoted(s) for s in outcome.sources if s.role == "value")


def test_the_note_that_says_a_measure_decision_demoted_this_is_the_note_the_decision_writes():
    """The record and the reader of it, pinned together. `c6_demoted` decides what a reviewer may
    put back by reading a sentence `apply_measure_settlement` writes, and the two living in
    different modules is exactly how such a pair drifts into always saying no."""
    from canopy.agents.mapper import C6_DEMOTION_NOTE, WITHHELD_NOTE, _note
    from canopy.models import c6_demoted_note

    assert c6_demoted_note(f"{C6_DEMOTION_NOTE}: this outcome's measure is endpoint")
    assert c6_demoted_note(_note("added by cross-check", f"{C6_DEMOTION_NOTE}: because"))
    assert c6_demoted_note(WITHHELD_NOTE)
    assert not c6_demoted_note("the paper calls this the alternative operationalization")
    assert not c6_demoted_note("")


def test_a_measure_the_map_settled_for_itself_can_be_taken_again(paper, protocol):
    """THE BUG. A settled ruling closes the question, so the answer named nothing open; and the
    settlement had already demoted the rival out of `_value_metrics`, so an answer naming it came
    back "which none of the value locations carries". There was no record a reviewer could write
    that would ever put the other measure back.

    On the corpus this tool was validated against, the map settled 25 measures for itself and asked
    about 2 — and gave one paper's cell `change_from_baseline` on one run and `endpoint` on the
    next, a difference of an order of magnitude in the effect size, with no way to say which the
    review wanted.
    """
    study = _settled_map(paper, protocol)
    assert not study.open_questions                     # nothing blocked; the tool simply chose
    assert settled_measure(study, study.datasets[0].dataset_id, "late_adaptation")

    answered = apply_map_answers(study, [_measure_answer(study, winning_analysis_metric="endpoint",
                                                         note="the window is the practice phase")])
    outcome = answered.datasets[0].outcomes[0]
    assert outcome.analysis_metric == "endpoint"
    roles = {s.analysis_metric: s.role for s in outcome.sources if s.analysis_metric != "unknown"}
    assert roles == {"endpoint": "value", "change_from_baseline": "alternate"}
    assert "human" in outcome.measure_ruling
    # …and the reversal is itself reversible: the decision belongs to whoever takes it last
    back = apply_map_answers(answered, [_measure_answer(answered)])
    assert back.datasets[0].outcomes[0].analysis_metric == "change_from_baseline"


def test_reversing_a_measure_never_leaves_two_readings_readable(paper, protocol):
    """C6's invariant survives the reversal: putting a demoted location back is a RE-settlement,
    never an un-settlement. A state with both measures readable is the `metric_mixed` failure C6
    exists to prevent, and the mechanism that corrects C6 must not reintroduce it."""
    study = _settled_map(paper, protocol)
    for metric in ("endpoint", "change_from_baseline"):
        answered = apply_map_answers(study, [_measure_answer(study,
                                                             winning_analysis_metric=metric)])
        outcome = answered.datasets[0].outcomes[0]
        readable = {s.analysis_metric for s in outcome.sources
                    if s.role == "value" and s.analysis_metric != "unknown"}
        assert readable == {metric}, metric
        assert extraction_blocks(answered) == ({}, {})   # …and it still blocks nothing


def test_a_measure_answer_that_settles_nothing_leaves_the_ruling_exactly_as_it_was(paper,
                                                                                   protocol):
    """An answer is read against the reopened outcome on a COPY. One that names a reading the map
    does not carry must leave the map settled as the tool settled it — not half-reopened with both
    measures readable, which would be worse than the ruling it failed to overturn."""
    study = _settled_map(paper, protocol)
    before = study.model_dump(mode="json")
    answered = apply_map_answers(study, [_measure_answer(study,
                                                         winning_analysis_metric="not_a_metric")])
    assert answered.model_dump(mode="json") == before


def test_the_map_says_why_it_refused_an_answer_rather_than_only_that_it_did(paper, protocol):
    """D3's other half. `_consumed_seqs` may only retire an answer the map ACCEPTED, and a refused
    answer that is merely "not consumed" sits on the page as "waiting for a re-run" through every
    re-run there will ever be. The reason travels so the run can say what was wrong with it."""
    study = _settled_map(paper, protocol)
    key = ("which_measure", study.datasets[0].dataset_id, "late_adaptation")

    good = map_answer_effects(study, [_measure_answer(study,
                                                      winning_analysis_metric="endpoint")])
    assert good[key] == ""
    bad = map_answer_effects(study, [_measure_answer(study,
                                                     winning_analysis_metric="not_a_metric")])
    assert "not_a_metric" in bad[key] and bad[key] != ""
    elsewhere = map_answer_effects(study, [_measure_answer(study, dataset_id="nobody:d9")])
    assert elsewhere[("which_measure", "nobody:d9", "late_adaptation")]


def test_an_alternate_the_mapper_wrote_is_never_put_back_by_a_measure_answer():
    """A paper's own "DE (primary); IEE (alternative)" is the map's reading of the paper, not a
    decision anyone took about this review. Restoring it would make `words_open` true with three
    readings standing, the answer would settle nothing, and had it been persisted the run would
    read both operationalizations into one cell — C6's failure, reintroduced by C6's cure."""
    outcome = OutcomeSources(
        outcome_key="late_adaptation",
        measure_name="DE (primary); IEE (alternative operationalisation)",
        sources=[
            Source(kind=SourceKind.figure_bar, page=1, locator="Fig. 1A", figure_id="fig01",
                   analysis_metric="endpoint", role="value"),
            Source(kind=SourceKind.text_mean_sd, page=2, locator="Results, IEE paragraph",
                   analysis_metric="endpoint", role="alternate",
                   notes="the paper's own alternative operationalisation"),
        ])
    reopened = reopened_outcome(outcome)
    assert [s.role for s in reopened.sources] == ["value", "alternate"]


def test_agreeing_with_a_measure_the_tool_chose_changes_nothing_at_all(paper, protocol):
    """A confirmation is not a re-settlement, and `read_measure_answer` cannot express one.

    Its two shapes are both wrong for "yes, that one": a metric-only answer re-wins every location
    that reads that metric, so once the ruling's demotions are reopened it puts back the very
    readings the ruling set aside (one Vachon cell went from one readable location to three); and a
    location-named answer demotes every other reading, which on an outcome printed one panel per
    group sets aside the other group's panel and leaves a two-arm contrast with one arm. Agreement
    is authority over nothing, so it touches nothing.
    """
    study = _settled_map(paper, protocol)
    outcome = study.datasets[0].outcomes[0]
    before = study.model_dump(mode="json")

    effects: dict = {}
    answered = apply_map_answers(
        study, [_measure_answer(study, winning_analysis_metric=outcome.analysis_metric,
                                note="I opened the figure; the tool is right")],
        effects=effects)
    now = answered.datasets[0].outcomes[0]
    assert effects[("which_measure", study.datasets[0].dataset_id, "late_adaptation")] == ""
    assert {s.role for s in now.sources} == {s.role for s in outcome.sources}
    assert now.analysis_metric == outcome.analysis_metric
    assert study.model_dump(mode="json") == before          # …and the map handed in is untouched
    # the only thing that changed is the record of who has looked at it
    assert "a human reviewer" in now.measure_ruling and "confirmed" in now.measure_ruling


def test_a_metric_only_answer_still_settles_an_outcome_nobody_has_ruled_on(paper, protocol):
    """…and the confirmation short-circuit may not swallow the ORIGINAL question. On an outcome with
    no ruling there is nothing to agree with: `analysis_metric` is the mapper's own guess beside two
    readable measures, and treating "that one" as agreement would close the question while leaving
    both readable — the state C6 exists to prevent, reached by answering the question that prevents
    it."""
    study = _open_measure(paper, protocol)
    outcome = study.datasets[0].outcomes[0]
    assert not (outcome.measure_ruling or "").strip()
    answered = apply_map_answers(study, [_measure_answer(study)])
    settled = answered.datasets[0].outcomes[0]
    assert {s.analysis_metric: s.role for s in settled.sources
            if s.analysis_metric != "unknown"} == {"change_from_baseline": "value",
                                                   "endpoint": "alternate"}
    assert not answered.open_questions


# ------------------------------------------- ticket 2a: the caption as a bounded third voice
def _caption_fixture(primary="UNKNOWN", caption_type="SE"):
    from types import SimpleNamespace

    from canopy.agents.mapper import _Conflicts
    from canopy.models import (DatasetSpec, DispersionType, OutcomeSources, Source, SourceKind,
                               StudyMap)

    src = Source(kind=SourceKind.figure_line, page=4, locator="Fig 2a", figure_id="fig02",
                 error_bar_type=DispersionType(primary))
    study = StudyMap(paper_id="p", datasets=[DatasetSpec(
        dataset_id="p:d1", outcomes=[OutcomeSources(outcome_key="late_adaptation",
                                                    sources=[src])])])
    fig = SimpleNamespace(id="fig02", caption_dispersion=caption_type,
                          caption_dispersion_quote="mean ± SE across subjects",
                          caption_dispersion_panels={})
    paper = SimpleNamespace(figures=[fig])
    return study, src, paper, _Conflicts(), [], []


def test_the_caption_fills_an_unknown_and_never_settles_agreement():
    from canopy.agents.mapper import EBT_FROM_CAPTION, _agree_error_bars
    from canopy.models import DispersionType

    # cross-check silent (no determination) + primary UNKNOWN → filled, still unconfirmed
    study, src, paper, conflicts, disagreements, flags = _caption_fixture()
    _agree_error_bars(study, {}, conflicts, disagreements, flags, paper=paper)
    assert src.error_bar_type is DispersionType.SE
    assert EBT_FROM_CAPTION in src.notes
    assert src.error_bar_agreement == "unconfirmed", "one witness is one witness"
    assert src.error_bar_evidence == "mean ± SE across subjects"
    # both agents AGREED on UNKNOWN (the Kumar shape): agreement about ignorance is not
    # knowledge — the caption still fills, and the agreement stands as what the agents said
    study, src, paper, conflicts, disagreements, flags = _caption_fixture()
    _agree_error_bars(study, {"fig02": {"error_bar_type": "UNKNOWN"}}, conflicts,
                      disagreements, flags, paper=paper)
    assert src.error_bar_type is DispersionType.SE and EBT_FROM_CAPTION in src.notes
    assert src.error_bar_agreement == "agreed"


def test_the_caption_disputes_a_stated_type_and_rides_conflicts_as_evidence_only():
    from canopy.agents.mapper import _agree_error_bars
    from canopy.models import DispersionType

    # cross-check silent + primary SD + caption SE → a real conflict card, no silent preference
    study, src, paper, conflicts, disagreements, flags = _caption_fixture(primary="SD")
    _agree_error_bars(study, {}, conflicts, disagreements, flags, paper=paper)
    assert src.error_bar_type is DispersionType.SD, "no side is preferred"
    assert src.error_bar_agreement == "conflict"
    assert conflicts.error_bars and conflicts.error_bars[0]["check_type"] is DispersionType.SE
    # the two AGENTS conflict: the adjudicator keeps the ruling; the caption's statement rides
    # in the disagreement notes as evidence, never as a verdict
    study, src, paper, conflicts, disagreements, flags = _caption_fixture(primary="SD")
    _agree_error_bars(study, {"fig02": {"error_bar_type": "CI95"}}, conflicts,
                      disagreements, flags, paper=paper)
    assert src.error_bar_agreement == "conflict"
    assert conflicts.error_bars[0]["check_type"] is DispersionType.CI95
    assert any("caption states SE" in d for d in disagreements)


# --------------------------------------------------- a shared-control marking nothing acts on
def _marked(index: int, experiment: str, control: str, exposure: str = "first") -> DatasetSpec:
    from canopy.models import GroupSpec

    return DatasetSpec(dataset_id=f"834f53a347e0:d{index}", cluster_id="834f53a347e0",
                       shared_control=True, experiment=experiment,
                       # as the live map records them, and what `rows.own_sample` reads as "its own
                       # participants": a labelled experiment the mapper called a first exposure
                       exposure_order=exposure,
                       group_a=GroupSpec(label=control.replace("-", "+"), n=6),
                       group_b=GroupSpec(label=control, n=6),
                       outcomes=[OutcomeSources(outcome_key="late_adaptation")])


def test_a_shared_control_marking_nothing_shares_is_flagged():
    """`shared_control` is a bare bool — no quote, no page, no rule — so a wrong marking used to
    divide a control arm's n and leave nothing on the record to review.

    Galea 2010 (`834f53a347e0`) is the finding: both datasets marked, each experiment with its own
    control group of six, and both controls pooled at n=3. `pipeline.rows` no longer divides those
    arms; this is the other half, which is that the marking itself reaches a person.
    """
    from canopy.agents.mapper import _flag_shared_controls

    study = StudyMap(paper_id="b" * 64, eligible=True,
                     datasets=[_marked(1, "Experiment 1", "Rs-"),
                               _marked(2, "Experiment 2", "Rg-")])
    flags: list[str] = []
    _flag_shared_controls(study, flags)
    assert len(flags) == 2, flags
    assert all("needs human" in f and "shared_control" in f for f in flags)
    assert "'Rs-'" in flags[0] and "'Experiment 1'" in flags[0]
    assert "834f53a347e0:d2" in flags[1] and "'Rg-'" in flags[1]


def test_a_control_two_datasets_really_share_is_not_flagged():
    """The other side of it: a control group two comparisons really do share is exactly what
    `shared_control` is for, and flagging it would train a reviewer to ignore the flag.

    Both shapes, because they reach the answer by different signals — two arms of one experiment,
    and Liddy 2026's (`f84c51677f3b`), where Experiment 2 is compared against Experiment 1's own ST
    group and the map says so in its notes.
    """
    from canopy.agents.mapper import _flag_shared_controls

    for datasets in ([_marked(1, "Experiment 1", "Controls"),
                      _marked(2, "Experiment 1", "Controls")],
                     [_marked(1, "Experiment 1 (Exp 1)", "ST"),
                      _marked(2, "Experiment 2 (Exp 2)", "ST")]):
        flags: list[str] = []
        _flag_shared_controls(StudyMap(paper_id="b" * 64, eligible=True, datasets=datasets), flags)
        assert flags == [], flags


def test_control_arms_merged_because_no_sample_can_be_claimed_are_said_out_loud():
    """The gap the arithmetic cannot close, made reviewable instead of silent.

    `rows._shares_one_arm` merges two marked datasets whenever the map claims no separate
    participant sample for them — an unlabelled experiment, a repeated exposure, a counterbalanced
    set — because over-splitting is the conservative error while under-splitting double-counts. But
    differently-named control arms divided together is a judgement, not a default, so the record
    says so. It fires on nothing in the corpus today; the day it does, the map is genuinely
    ambiguous and a person should look.
    """
    from canopy.agents.mapper import _flag_shared_controls

    study = StudyMap(paper_id="b" * 64, eligible=True,
                     datasets=[_marked(1, "Exp 1", "Rs-", "counterbalanced_collapsed"),
                               _marked(2, "Exp 2", "Rg-", "counterbalanced_collapsed")])
    flags: list[str] = []
    _flag_shared_controls(study, flags)
    assert len(flags) == 1, flags
    assert "divided by 2" in flags[0] and "DIFFERENT control arms" in flags[0]
    assert "'Rs-' in 'Exp 1'" in flags[0] and "'Rg-' in 'Exp 2'" in flags[0]
    assert "needs human" in flags[0]
