"""Task 4: the study mapper.

Two halves:
  * a replayed end-to-end map of Bock 2005 (fixtures in `tests/fixtures/llm`, recorded once live);
  * offline unit tests with `FakeProvider` for everything the mapper decides in *code* — roster
    completeness, cross-check diffing, adjudication, needs_human rules, id assignment.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from canopy.agents import load_prompt
from canopy.agents.mapper import (MAPPER_SCHEMA, MAPPER_SOURCES_SCHEMA, PROMPT_VERSION,
                                  map_study, protocol_text, roster_text)
from canopy.config import MODELS, live_enabled, load_env, record_enabled
from canopy.ingest.pdf import PaperRecord, ingest_pdf
from canopy.llm.client import LLMClient
from canopy.llm.providers import FakeProvider
from canopy.llm.schemas import assert_no_derived_stats, assert_valid_output_schema
from canopy.models import DatasetSpec, DispersionType, Source, SourceKind, StudyMap
from canopy.protocol import load_protocol

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "tests" / "fixtures" / "pdfs" / "bock2005.pdf"
REPLAY = ROOT / "tests" / "fixtures" / "llm"
PROTOCOL_PATH = ROOT / "examples" / "protocols" / "aging_sensorimotor_adaptation.yaml"

# Read at import time: the autouse `offline_by_default` fixture clears both for every test that is
# not marked `live`, and the session fixtures below are built inside a test.
LIVE = live_enabled()
RECORD = record_enabled()
if LIVE:
    load_env()


@pytest.fixture(scope="session")
def paper(tmp_path_factory) -> PaperRecord:
    return ingest_pdf(PDF, tmp_path_factory.mktemp("bock2005-map"))


@pytest.fixture(scope="session")
def protocol():
    return load_protocol(PROTOCOL_PATH)


@pytest.fixture(scope="session")
def bock_map(paper, protocol) -> StudyMap:
    client = LLMClient(replay_dir=REPLAY, record_dir=REPLAY if RECORD else None,
                       allow_live=LIVE, cache_dir=None)
    study = map_study(client, paper, protocol)
    if LIVE:                                    # recording run: report what it cost
        print(f"\n[mapper] ${client.total_cost():.4f} over {len(client.calls())} calls: "
              f"{[c['model'] for c in client.calls()]}")
    return study


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
def test_bock_is_eligible(bock_map):
    assert bock_map.eligible is True
    assert bock_map.eligibility_rationale.strip()
    assert not bock_map.exclusion_reason


def test_bock_has_a_dataset_with_twelve_per_group(bock_map, paper):
    sizes = [(d.group_a.n, d.group_b.n) for d in bock_map.datasets]
    assert (12, 12) in sizes, sizes
    dataset = next(d for d in bock_map.datasets if (d.group_a.n, d.group_b.n) == (12, 12))
    assert dataset.group_a.n_evidence and dataset.group_b.n_evidence
    assert dataset.dataset_id == f"{paper.sha256[:12]}:d{bock_map.datasets.index(dataset) + 1}"
    assert dataset.cluster_id == paper.sha256[:12]
    assert len(dataset.all_groups_listed) >= 2


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


def test_bock_outcome_keys_and_directions_come_from_the_protocol(bock_map, protocol):
    keys = {o.outcome_key for d in bock_map.datasets for o in d.outcomes}
    assert keys <= {o.key for o in protocol.outcomes} and keys
    for dataset in bock_map.datasets:
        for outcome in dataset.outcomes:
            assert outcome.measure_name
            assert outcome.higher_is_better is not None
            assert outcome.higher_is_better_evidence


def test_bock_roster_decides_every_ingested_figure_and_table(bock_map, paper):
    decided = {r.id: r for r in bock_map.roster}
    assert set(decided) == {f.id for f in paper.figures} | {t.id for t in paper.tables}
    for fig in paper.figures:
        assert decided[fig.id].kind == "figure" and decided[fig.id].page == fig.page
    assert decided["fig01"].relevant is True and decided["fig01"].outcome_keys


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


def _check(datasets=None, **over):
    payload = {
        "eligible": True, "eligibility_rationale": "rotation, two age groups",
        "datasets": datasets if datasets is not None else [{
            "label": "pointing", "experiment": "1", "condition": "rotation",
            "group_a": {"label": "old", "n": 12, "n_evidence": "twelve old"},
            "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"},
            "outcomes": [{"outcome_key": "late_adaptation", "sources": [_check_source()]}]}],
        "notes": "",
    }
    payload.update(over)
    return payload


def _adjudication(n_a=12, n_b=12, rulings=(), **over):
    payload = {
        "eligible": True, "eligibility_rationale": "two age groups, rotated feedback",
        "datasets": [{"primary_dataset_index": 1, "label": "pointing",
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


def test_whole_pdf_is_sent_once_as_a_cached_document_block(paper, protocol):
    _, provider = _mapped(paper, protocol, [_primary(), _sources(), _check()])
    for request in provider.requests:
        blocks = request.messages[0]["content"]
        documents = [b for b in blocks if b.get("type") == "document"]
        assert len(documents) == 1
        assert documents[0]["cache_control"] == {"type": "ephemeral"}
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


def test_error_bar_disagreement_without_adjudication_keeps_primary_and_asks_for_a_human(
        paper, protocol):
    study, provider = _mapped(paper, protocol,
                              [_primary(), _sources(), _check(datasets=[{
                                  "label": "pointing", "experiment": "1", "condition": "rotation",
                                  "group_a": {"label": "old", "n": 12, "n_evidence": ""},
                                  "group_b": {"label": "young", "n": 12, "n_evidence": ""},
                                  "outcomes": [{"outcome_key": "late_adaptation",
                                                "sources": [_check_source(error_bar_type="SE")]}]}])])
    assert len(provider.requests) == 3                       # no adjudication: eligibility+Ns agree
    source = study.datasets[0].outcomes[0].sources[0]
    assert source.error_bar_type is DispersionType.SD        # primary value survives
    flags = [f for f in study.needs_human if "error-bar" in f]
    assert flags and "SD" in flags[0] and "SE" in flags[0], study.needs_human
    assert "late_adaptation" in flags[0] and study.datasets[0].dataset_id in flags[0]
    assert [d for d in study.disagreements if "error bar" in d]


def test_adjudicated_error_bar_ruling_resolves_the_disagreement(paper, protocol):
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "old", "n": 11, "n_evidence": "eleven old"},
        "group_b": {"label": "young", "n": 12, "n_evidence": "twelve young"},
        "outcomes": [{"outcome_key": "late_adaptation",
                      "sources": [_check_source(error_bar_type="SE")]}]}])
    ruling = {"dataset_index": 1, "outcome_key": "late_adaptation", "page": 3, "locator": "Fig 1",
              "figure_id": "fig01", "table_id": "", "error_bar_type": "SE",
              "evidence": "the legend says standard errors"}
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(), check, _adjudication(rulings=[ruling])])
    assert study.datasets[0].outcomes[0].sources[0].error_bar_type is DispersionType.SE
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


def test_swapped_group_mapping_is_flagged_for_a_human(paper, protocol):
    check = _check(datasets=[{
        "label": "pointing", "experiment": "1", "condition": "rotation",
        "group_a": {"label": "young", "n": 12, "n_evidence": "twelve young"},
        "group_b": {"label": "old", "n": 12, "n_evidence": "twelve old"},
        "outcomes": [{"outcome_key": "late_adaptation", "sources": [_check_source()]}]}])
    study, provider = _mapped(paper, protocol, [_primary(), _sources(), check])
    assert len(provider.requests) == 3                        # Ns agree, so no adjudication
    assert study.datasets[0].group_a.label == "old"           # primary mapping survives
    assert [f for f in study.needs_human if "group mapping" in f], study.needs_human
    assert [d for d in study.disagreements if "group mapping" in d]


def test_dataset_count_mismatch_is_recorded(paper, protocol):
    study, _ = _mapped(paper, protocol,
                       [_primary(datasets=[_dataset(), _dataset()]), _sources(), _check()])
    assert [d for d in study.disagreements if "dataset count" in d], study.disagreements
    assert len(study.datasets) == 2
    assert [d.dataset_id[-2:] for d in study.datasets] == ["d1", "d2"]


def test_unclassifiable_source_kind_is_flagged_not_dropped_silently(paper, protocol):
    unknown = dict(FIG_SOURCE, kind="unknown", locator="appendix plot")
    study, _ = _mapped(paper, protocol,
                       [_primary(), _sources(outcomes=[_outcome(sources=(unknown,))]), _check()])
    sources = study.datasets[0].outcomes[0].sources
    assert "appendix plot" not in [s.locator for s in sources]      # not silently kept as a guess
    assert [f for f in study.needs_human if "appendix plot" in f], study.needs_human
    assert [s.notes for s in sources] == ["added by cross-check"]   # the location survives


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
