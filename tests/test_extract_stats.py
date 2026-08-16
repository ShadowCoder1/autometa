"""Task 7: the test-statistic / reported-effect-size extractor.

A statistic is a fallback route to an effect size: when a paper prints no group means, an
independent-groups t or F on the outcome can still be converted — in code, in Task 9, and only for
designs that admit the conversion. So this extractor transcribes what is printed (statistic, dfs,
p, the design, which way the difference went) and code decides admissibility from the design; the
model's own judgement can only make a statistic *less* admissible, never more.

Replayed on Bock 2005, which prints exactly the two cases that matter: a mixed-design main effect
of the grouping factor (F(1,22)=7.58, inadmissible) and a plain two-group t test (t(22)=5.25).
"""
from __future__ import annotations

import pytest

from canopy.agents import load_prompt
from canopy.agents.extract_stats import (ADMISSIBLE_DESIGNS, EXTRACT_STATS_SCHEMA, PROMPT_FILES,
                                         PROMPT_VERSION, admissibility, extract_test_statistics)
from canopy.config import MODELS
from canopy.llm.client import LLMClient
from canopy.llm.providers import FakeProvider
from canopy.llm.schemas import assert_no_derived_stats, assert_valid_output_schema
from canopy.models import (Candidate, DatasetSpec, GroupSpec, OutcomeDef, Protocol, Source,
                           SourceKind)

replayed = pytest.mark.replay

# `paper`, `protocol`, `client`, `bock_map` and `dataset` (Bock 2005, ingested and mapped once per
# session) come from tests/conftest.py.


@pytest.fixture
def fake_dataset() -> DatasetSpec:
    """The offline tests never touch the replayed mapper fixtures (a session client built by a
    non-replay test cannot record)."""
    return DatasetSpec(
        dataset_id="b511dbb76fa6:d1", label="pointing", cluster_id="b511dbb76fa6",
        group_a=GroupSpec(label="old (older) subjects", n=12, n_evidence="twelve old subjects"),
        group_b=GroupSpec(label="young subjects", n=12, n_evidence="twelve young subjects"))


#: the same screening-test protocol the text extractor uses — the one outcome Bock analyses with a
#: plain two-group t test
SCREENING = OutcomeDef(
    key="screening_time",
    label="Screening test completion time",
    definition="Time to complete the paper-and-pencil screening test in which participants join "
               "numbered dots in ascending order.",
    measurement_window="The single administration of the screening test.",
    higher_is_better_hint="A longer completion time is worse performance.",
    units_hint="seconds",
)


@pytest.fixture(scope="session")
def screening_protocol(protocol) -> Protocol:
    return protocol.model_copy(update={"outcomes": [SCREENING]})


SCREENING_SOURCE = Source(
    kind=SourceKind.test_statistic, page=3, locator="Results, screening-test paragraph",
    quote="the difference was statistically significant")


# ------------------------------------------------------------------ schema and prompt hygiene
def test_schema_is_valid_and_carries_no_derived_stats():
    """`reported_*` fields are values the paper printed, which the guard allows; nothing here asks
    a model for a statistic we must compute."""
    assert_valid_output_schema(EXTRACT_STATS_SCHEMA, "EXTRACT_STATS_SCHEMA")
    assert_no_derived_stats(EXTRACT_STATS_SCHEMA, name="EXTRACT_STATS_SCHEMA")


def test_the_schema_stays_far_under_the_grammar_limit():
    leaves = EXTRACT_STATS_SCHEMA["properties"]["statistics"]["items"]["properties"]
    assert len(leaves) + 1 <= 40


def test_schema_fields_exist_on_the_candidate_model():
    leaves = set(EXTRACT_STATS_SCHEMA["properties"]["statistics"]["items"]["properties"])
    checked_in_code = {"effect_as_written", "compares_the_two_groups", "direction_quote"}
    assert leaves - checked_in_code <= set(Candidate.model_fields)


def test_prompt_carries_no_domain_knowledge():
    banned = ["older adult", "younger adult", "visuomotor", "adaptation", "aftereffect",
              "ageing", "aging", "elderly", "perturbation", "sensorimotor"]
    text = load_prompt(PROMPT_FILES[0]).lower()
    assert not [word for word in banned if word in text]


def test_prompt_version_tracks_the_prompt_file():
    assert PROMPT_VERSION.startswith("extract_stats/1@")
    assert len(PROMPT_VERSION) == len("extract_stats/1@") + 8


# ------------------------------------------------------------------ the admissibility rule table
@pytest.mark.parametrize("design,expected", [
    ("independent_t", True),
    ("one_way_between", True),
    ("mixed_main_effect", False),
    ("interaction", False),
    ("paired", False),
    ("ancova", False),
    ("welch", False),
    ("unknown", False),
])
def test_only_a_between_group_comparison_is_admissible(design, expected):
    ok, reason = admissibility("test_statistic", design, "yes", True, "")
    assert ok is expected
    assert (reason == "") is expected                 # every refusal explains itself
    assert (design in ADMISSIBLE_DESIGNS) is expected


def test_a_statistic_about_something_else_is_not_admissible():
    """The most useful reason first: a within-participant factor's F is not about these groups at
    all, whatever design label the extractor gave it."""
    ok, reason = admissibility("test_statistic", "independent_t", "no", True, "")
    assert ok is False and "two groups" in reason
    ok, reason = admissibility("test_statistic", "mixed_main_effect", "no", True, "")
    assert ok is False and "two groups" in reason


def test_an_unclear_target_is_not_admissible():
    ok, reason = admissibility("test_statistic", "independent_t", "unknown", True, "")
    assert ok is False


def test_the_extractor_can_veto_an_admissible_design():
    ok, reason = admissibility("test_statistic", "independent_t", "yes", False,
                               "this t test is for the baseline block")
    assert ok is False and "baseline block" in reason


def test_the_extractor_cannot_promote_an_inadmissible_design():
    ok, reason = admissibility("test_statistic", "interaction", "yes", True, "looks fine to me")
    assert ok is False and "interaction" in reason


@pytest.mark.parametrize("scale,standardizer,expected", [
    ("cohens_d", "pooled_sd_between", True),
    ("hedges_g", "pooled_sd_between", True),
    ("cohens_d", "dz_paired", False),
    ("partial_eta_squared", "partial_eta", False),
    ("unknown", "unknown", False),
])
def test_a_reported_effect_size_is_admissible_only_between_groups(scale, standardizer, expected):
    ok, reason = admissibility("reported_d", "unknown", "yes", True, "", reported_scale=scale,
                               standardizer=standardizer)
    assert ok is expected
    assert (reason == "") is expected


# ------------------------------------------------------------------ offline unit tests
def _stat(**over):
    row = {"kind": "test_statistic", "status": "found", "page": 3,
           "locator": "Results, ANOVA over the adaptation episodes",
           "quote": "significant effects were yielded for Age (F(1,22)=7.58, P<0.05), Episode "
                    "(F(14,308)=41.14, P<0.001), and their interaction (F(14,308)=3.44, P<0.001)",
           "effect_as_written": "Age", "compares_the_two_groups": "yes",
           "design": "mixed_main_effect", "stat_type": "F", "stat_value": 7.58, "df": None,
           "df1": 1.0, "df2": 22.0, "tails": None, "p_kind": "less_than", "p_value": 0.05,
           "direction": "a_greater",
           "direction_quote": "adaptive improvement was more pronounced for young than for old",
           "reported_value": None, "reported_scale": "unknown", "standardizer": "unknown",
           "reported_ci_low": None, "reported_ci_high": None, "positive_means": "unknown",
           "admissible": False, "admissible_reason": "mixed design", "notes": ""}
    row.update(over)
    return row


def _payload(rows=None, notes=""):
    return {"statistics": [dict(r) for r in (rows if rows is not None else [_stat()])],
            "notes": notes}


STAT_SOURCE = Source(kind=SourceKind.test_statistic, page=3,
                     locator="Results, ANOVA over the adaptation episodes",
                     quote="significant effects were yielded for Age")


def _extract(paper, protocol, fake_dataset, payload, *, sources=(STAT_SOURCE,),
             outcome_key="late_adaptation", model=None):
    provider = FakeProvider([payload])
    client = LLMClient(provider=provider, cache_dir=None)
    kwargs = {"model": model} if model else {}
    candidates = extract_test_statistics(client, paper, protocol, fake_dataset, outcome_key,
                                         list(sources), **kwargs)
    return candidates, provider


def test_a_printed_f_test_becomes_a_grounded_candidate(paper, protocol, fake_dataset):
    cands, provider = _extract(paper, protocol, fake_dataset, _payload())
    assert len(provider.requests) == 1 and len(cands) == 1
    cand = cands[0]
    assert cand.kind == "test_statistic" and cand.group is None
    assert (cand.stat_type, cand.stat_value) == ("F", 7.58)
    assert (cand.df1, cand.df2, cand.df) == (1.0, 22.0, None)
    assert cand.design == "mixed_main_effect" and cand.p_kind == "less_than"
    assert cand.p_value == 0.05 and cand.tails is None
    assert cand.direction == "a_greater"
    assert cand.grounded is True and cand.page == 3
    assert "more pronounced for young" in cand.notes            # the direction evidence travels
    assert cand.admissible is False and cand.admissible_reason


def test_code_fills_the_bookkeeping(paper, protocol, fake_dataset):
    cands, _ = _extract(paper, protocol, fake_dataset, _payload())
    cand = cands[0]
    assert cand.paper_id == paper.sha256 and cand.dataset_id == fake_dataset.dataset_id
    assert cand.outcome_key == "late_adaptation"
    assert cand.model == MODELS["primary"] and cand.prompt_version == PROMPT_VERSION
    assert cand.extractor_id == f"stats:{MODELS['primary']}" and cand.llm_call_id
    assert cand.source_kind is SourceKind.test_statistic
    assert cand.route == "test_statistic"
    assert cand.candidate_id and fake_dataset.dataset_id in cand.candidate_id


def test_the_context_is_text_only(paper, protocol, fake_dataset):
    _, provider = _extract(paper, protocol, fake_dataset, _payload())
    blocks = provider.requests[0].messages[0]["content"]
    assert not [b for b in blocks if b.get("type") == "image"]
    assert "[page 3 text]" in blocks[0]["text"]
    prompt = blocks[-1]["text"]
    assert "{{" not in prompt and "late_adaptation" in prompt
    assert "old (older) subjects" in prompt and "analysed n = 12" in prompt
    assert provider.requests[0].model == MODELS["primary"]


def test_a_source_kind_this_extractor_cannot_read_is_left_alone(paper, protocol, fake_dataset):
    figure = Source(kind=SourceKind.figure_line, page=3, locator="Fig 1", figure_id="fig01")
    cands, provider = _extract(paper, protocol, fake_dataset, _payload(), sources=(figure,))
    assert cands == [] and provider.requests == []


def test_several_statistics_all_come_back(paper, protocol, fake_dataset):
    rows = [_stat(), _stat(effect_as_written="Age x Episode", design="interaction",
                           stat_value=3.44, df1=14.0, df2=308.0, direction="unknown")]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    assert [c.design for c in cands] == ["mixed_main_effect", "interaction"]
    assert len({c.candidate_id for c in cands}) == 2
    assert all(c.admissible is False for c in cands)


def test_a_two_group_t_test_is_admissible(paper, protocol, fake_dataset):
    rows = [_stat(design="independent_t", stat_type="t", stat_value=5.25, df=22.0, df1=None,
                  df2=None, p_kind="less_than", p_value=0.001, admissible=True,
                  admissible_reason="", effect_as_written="group difference",
                  quote="the difference was statistically significant (t(22)=5.25; P<0.001)")]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    assert cands[0].admissible is True and cands[0].admissible_reason == ""
    assert (cands[0].stat_type, cands[0].stat_value, cands[0].df) == ("t", 5.25, 22.0)


def test_a_bare_p_value_is_routed_as_a_p_value(paper, protocol, fake_dataset):
    rows = [_stat(stat_type="p", stat_value=None, df1=None, df2=None, p_kind="exact",
                  p_value=0.03)]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    assert cands[0].route == "p_value" and cands[0].stat_type == "p"


def test_a_reported_effect_size_is_parsed_into_its_own_kind(paper, protocol, fake_dataset):
    rows = [_stat(kind="reported_effect_size", stat_type="unknown", stat_value=None, df1=None,
                  df2=None, p_kind="unknown", p_value=None, design="unknown",
                  reported_value=0.62, reported_scale="cohens_d",
                  standardizer="pooled_sd_between", reported_ci_low=0.41, reported_ci_high=0.83,
                  positive_means="a_greater", admissible=True, admissible_reason="",
                  quote="the difference between groups was significant")]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    cand = cands[0]
    assert cand.kind == "reported_d" and cand.route == "reported_d"
    assert cand.reported_value == 0.62 and cand.reported_scale == "cohens_d"
    assert cand.standardizer == "pooled_sd_between"
    assert (cand.reported_ci_low, cand.reported_ci_high) == (0.41, 0.83)
    assert cand.positive_means == "a_greater"
    assert cand.source_kind is SourceKind.reported_effect_size
    assert cand.admissible is True


def test_a_status_that_is_not_found_carries_no_numbers(paper, protocol, fake_dataset):
    """Degrees of freedom and a p kind are numbers too: left standing on an empty row they read
    downstream as a statistic the paper never printed."""
    rows = [_stat(status="ambiguous", quote="", stat_value=7.58, p_value=0.05, tails=2)]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    cand = cands[0]
    assert cand.status == "ambiguous"
    assert cand.stat_value is None and cand.p_value is None and cand.reported_value is None
    assert (cand.df, cand.df1, cand.df2, cand.tails) == (None, None, None, None)
    assert cand.p_kind == "unknown"
    assert "dropped because status is ambiguous" in cand.notes
    assert "7.58" in cand.notes and "df2" in cand.notes and "less_than" in cand.notes
    assert cand.admissible is False and cand.admissible_reason == "the extractor answered ambiguous"


def test_a_found_row_keeps_its_degrees_of_freedom(paper, protocol, fake_dataset):
    cands, _ = _extract(paper, protocol, fake_dataset, _payload())
    assert (cands[0].df1, cands[0].df2) == (1.0, 22.0) and cands[0].p_kind == "less_than"


def test_an_invented_quote_is_marked_ungrounded(paper, protocol, fake_dataset):
    rows = [_stat(quote="the groups differed on the final block (F(1,22)=99.9, P<0.001)")]
    cands, _ = _extract(paper, protocol, fake_dataset, _payload(rows=rows))
    assert cands[0].grounded is False and "not found" in cands[0].notes


def test_finding_nothing_is_an_explicit_answer(paper, protocol, fake_dataset):
    cands, _ = _extract(paper, protocol, fake_dataset, {"statistics": [], "notes": "no tests here"})
    assert len(cands) == 1
    assert cands[0].status == "not_on_these_pages" and cands[0].kind == "test_statistic"
    assert cands[0].stat_value is None and cands[0].admissible is False
    assert "no tests here" in cands[0].notes


def test_overall_notes_travel_with_every_candidate(paper, protocol, fake_dataset):
    cands, _ = _extract(paper, protocol, fake_dataset,
                        _payload(notes="the paper reports no post hoc tests"))
    assert all("no post hoc tests" in c.notes for c in cands)


# ------------------------------------------------------------------ replayed Bock 2005
def _stat_sources(dataset, outcome_key):
    return [s for s in dataset.outcome(outcome_key).sources
            if s.kind is SourceKind.test_statistic]


@replayed
def test_bock_age_main_effect_is_transcribed_and_flagged_inadmissible(client, paper, protocol,
                                                                     dataset):
    """"significant effects were yielded for Age (F(1,22)=7.58, P<0.05), Episode (F(14,308)=41.14,
    P<0.001), and their interaction" — a mixed ANOVA, so the Age main effect cannot stand in for
    the group means, and Task 9 will refuse it in code as well."""
    sources = _stat_sources(dataset, "late_adaptation")
    assert sources, "the mapper found no test-statistic source for late_adaptation"
    cands = extract_test_statistics(client, paper, protocol, dataset, "late_adaptation", sources)
    age = [c for c in cands if c.stat_value == 7.58]
    assert age, [(c.stat_type, c.stat_value, c.df1, c.df2, c.design) for c in cands]
    cand = age[0]
    assert cand.stat_type == "F" and (cand.df1, cand.df2) == (1.0, 22.0)
    assert cand.p_kind == "less_than" and cand.p_value == 0.05
    assert cand.design == "mixed_main_effect", cand.notes
    assert cand.admissible is False and cand.admissible_reason
    assert cand.grounded is True and cand.page == 3
    assert "F(1,22)=7.58" in cand.quote.replace(" ", "")
    assert cand.prompt_version == PROMPT_VERSION and cand.route == "test_statistic"
    for other in cands:                       # nothing in this paragraph is convertible
        assert other.admissible is False, (other.design, other.quote[:80])


@replayed
def test_bock_reports_no_group_effect_on_the_second_outcome(client, paper, protocol, dataset):
    """The after-effect ANOVA yielded Episode and the interaction only — no main effect of the
    grouping factor, so there is nothing here an effect size could be built from."""
    sources = _stat_sources(dataset, "aftereffect")
    assert sources
    cands = extract_test_statistics(client, paper, protocol, dataset, "aftereffect", sources)
    assert cands
    assert not [c for c in cands if c.admissible], [(c.design, c.quote[:60]) for c in cands]
    for cand in cands:
        assert cand.grounded in (None, True)
        assert cand.stat_value is None or cand.quote


@replayed
def test_bock_two_group_t_test_is_transcribed_with_its_direction(client, paper, screening_protocol,
                                                                 dataset):
    """"the completion time for young subjects was 27.4±7.2 s and that for old subjects was
    42.5±6.9 s; the difference was statistically significant (t(22)=5.25; P<0.001)"."""
    cands = extract_test_statistics(client, paper, screening_protocol, dataset, "screening_time",
                                    [SCREENING_SOURCE])
    ttest = [c for c in cands if c.stat_type == "t"]
    assert ttest, [(c.stat_type, c.stat_value, c.design) for c in cands]
    cand = ttest[0]
    assert cand.stat_value == 5.25 and cand.df == 22.0
    assert cand.p_kind == "less_than" and cand.p_value == 0.001
    assert cand.direction == "a_greater", cand.notes      # the older group took longer
    assert cand.grounded is True and cand.page == 3
    assert "t(22)=5.25" in cand.quote.replace(" ", "")
    # df 22 = 12 + 12 - 2, two independent groups: the one convertible statistic in the paper
    assert cand.design == "independent_t", cand.notes
    assert cand.admissible is True and cand.admissible_reason == ""
    assert cand.route == "test_statistic" and cand.kind == "test_statistic"


def test_no_statistic_source_means_no_call_at_all(paper, protocol, fake_dataset):
    """`unknown` alone is not evidence that the paper prints a statistic (task 15 §A3).

    The mapper has already read the paper and named where the numbers are. When none of those is
    a test statistic or a reported effect size, this extractor has nothing to transcribe, and its
    call is one of the more expensive ones in the pipeline.
    """
    provider = FakeProvider([{"statistics": [], "notes": ""}])
    client = LLMClient(provider=provider, cache_dir=None)
    only_unknown = [Source(kind=SourceKind.unknown, page=3, locator="a fitted parameter")]
    assert extract_test_statistics(client, paper, protocol, fake_dataset, "late_adaptation",
                                   only_unknown) == []
    assert provider.requests == [], "no source of the right kind must mean no model call"

    assert extract_test_statistics(client, paper, protocol, fake_dataset, "late_adaptation",
                                   [*only_unknown, STAT_SOURCE])
    assert len(provider.requests) == 1
