"""Task 8, steps 3–5: the three verification agents.

* `verify_candidate` — a *different* model, the whole paper, and one job: refute the reading.
* `adjudicate`      — the strongest model, once, on a cell the vote and the verifier could not settle.
* `orientation`     — what a larger raw value on this measure means, answered twice and independently.

The offline tests here drive each agent with `FakeProvider` payloads shaped like real answers, so
every parsing and guard path is covered without an API key. The replayed tests at the bottom run
the agents against Bock 2005 from recorded fixtures; until those exist they skip visibly rather
than weaken anything.
"""
from __future__ import annotations

import pytest

from canopy.agents import load_prompt
from canopy.agents.adjudicator import (ADJUDICATE_SCHEMA, PROMPT_VERSION as ADJ_VERSION,
                                       adjudicate)
from canopy.agents.orientation import (ORIENTATION_SCHEMA, PROMPT_VERSION as ORI_VERSION,
                                       combine_orientation, orientation, orientation_run)
from canopy.agents.verifier import (MAX_REOPENS, PROMPT_VERSION, REFUTATION_TARGETS,
                                    VERIFIER_SCHEMA, verifier_model_for, verify_candidate)
from canopy.config import MODELS
from canopy.llm.client import LLMClient
from canopy.llm.costs import PDF_TOKENS_PER_PAGE, clear_file_pages, file_document_tokens
from canopy.llm.errors import MissingFixture
from canopy.llm.providers import FakeProvider
from canopy.llm.schemas import assert_no_derived_stats, assert_valid_output_schema
from canopy.models import (Candidate, DatasetSpec, DispersionType, GroupSpec, OrientationRun,
                           OutcomeDef, OutcomeSources, Source, SourceKind, VerifierVerdict)

replayed = pytest.mark.replay

# `paper`, `protocol`, `client`, `bock_map` and `dataset` come from tests/conftest.py.

RECORD_HINT = ("fixture not recorded yet — run: CANOPY_LIVE=1 CANOPY_RECORD=1 "
               ".venv/bin/python -m pytest tests/test_verifier.py -q")

OPUS, SONNET = MODELS["primary"], MODELS["secondary"]

#: Bock 2005 prints these two screening-test values as "M ± SD" in the Results; they mirror the
#: shape of the recorded stats-extractor output without depending on the text-extractor fixtures.
SCREENING = OutcomeDef(
    key="screening_time", label="Screening test completion time",
    definition="Time to complete the paper-and-pencil screening test in which participants join "
               "numbered dots in ascending order.",
    measurement_window="The single administration of the screening test.",
    higher_is_better_hint="A longer completion time is worse performance.", units_hint="seconds")


@pytest.fixture
def bock_dataset() -> DatasetSpec:
    return DatasetSpec(
        dataset_id="b511dbb76fa6:d1", label="pointing", cluster_id="b511dbb76fa6",
        group_a=GroupSpec(label="old subjects", n=12, n_evidence="twelve old subjects"),
        group_b=GroupSpec(label="young subjects", n=12, n_evidence="twelve young subjects"),
        outcomes=[OutcomeSources(
            outcome_key="screening_time", measure_name="screening test completion time",
            units="s", higher_is_better=None,
            sources=[Source(kind=SourceKind.text_mean_sd, page=3,
                            locator="Results, screening-test paragraph",
                            quote="the completion time for young subjects was 27.4±7.2 s")])])


def screening_candidate(group="A", **kwargs) -> Candidate:
    base = dict(candidate_id=f"b511dbb76fa6:d1:screening_time:{group}:text#0",
                dataset_id="b511dbb76fa6:d1", outcome_key="screening_time", kind="group_stats",
                group=group, status="found", source_kind=SourceKind.text_mean_sd,
                mean=42.5, dispersion_value=6.9, dispersion_type=DispersionType.SD, n=12,
                unit="s", value_as_written="42.5±6.9 s", page=3,
                quote="that for old subjects was 42.5±6.9 s", grounded=True,
                grounding_similarity=1.0, route="text", model=OPUS,
                extractor_id=f"text:table_first:{OPUS}")
    base.update(kwargs)
    return Candidate(**base)


def verifier_payload(**overrides):
    payload = {"verdict": "confirmed",
               "reason": "page 3 prints 'that for old subjects was 42.5±6.9 s'",
               "checked": list(REFUTATION_TARGETS),
               "alt_mean": None, "alt_dispersion_value": None, "alt_dispersion_type": "UNKNOWN",
               "alt_n": None, "alt_page": None, "alt_quote": "", "better_source": "",
               "better_source_page": None, "notes": ""}
    payload.update(overrides)
    return payload


def run_verifier(paper, cand, payload, **kwargs):
    provider = FakeProvider([payload])
    client = LLMClient(provider=provider, cache_dir=None)
    verdict = verify_candidate(client, paper, cand, model=kwargs.pop("model", SONNET), **kwargs)
    return verdict, provider


# ------------------------------------------------------------------ schema and prompt hygiene
@pytest.mark.parametrize("name,schema", [("VERIFIER_SCHEMA", VERIFIER_SCHEMA),
                                         ("ADJUDICATE_SCHEMA", ADJUDICATE_SCHEMA),
                                         ("ORIENTATION_SCHEMA", ORIENTATION_SCHEMA)])
def test_schemas_are_valid_and_carry_no_derived_stats(name, schema):
    assert_valid_output_schema(schema, name)
    assert_no_derived_stats(schema, name=name)


@pytest.mark.parametrize("prompt", ["verifier", "adjudicator", "orientation"])
def test_prompts_carry_no_domain_knowledge(prompt):
    banned = ["older adult", "younger adult", "visuomotor", "adaptation", "aftereffect",
              "ageing", "aging", "elderly", "perturbation", "sensorimotor"]
    text = load_prompt(prompt).lower()
    assert not [word for word in banned if word in text]


@pytest.mark.parametrize("version,prefix", [(PROMPT_VERSION, "verifier/1@"),
                                            (ADJ_VERSION, "adjudicator/1@"),
                                            (ORI_VERSION, "orientation/1@")])
def test_prompt_versions_track_their_files(version, prefix):
    assert version.startswith(prefix) and len(version) == len(prefix) + 8


def test_the_verifier_prompt_names_every_refutation_target():
    text = load_prompt("verifier")
    assert [target for target in REFUTATION_TARGETS if target in text] == list(REFUTATION_TARGETS)


# ------------------------------------------------------------------ the verifier
def test_a_confirmation_is_parsed_with_its_provenance(paper, bock_dataset):
    cand = screening_candidate()
    verdict, provider = run_verifier(paper, cand, verifier_payload(), dataset=bock_dataset)
    assert verdict.verdict == "confirmed" and verdict.candidate_id == cand.candidate_id
    assert "42.5" in verdict.reason
    assert verdict.model == SONNET and verdict.prompt_version == PROMPT_VERSION
    assert verdict.llm_call_id and verdict.checked == list(REFUTATION_TARGETS)
    assert verdict.reopen == 0 and verdict.notes == ""
    assert provider.requests[0].model == SONNET


def test_a_refutation_carries_the_value_the_paper_actually_prints(paper, bock_dataset):
    payload = verifier_payload(verdict="refuted", reason="42.5 is the OLD group; this row is B",
                               alt_mean=27.4, alt_dispersion_value=7.2,
                               alt_dispersion_type="SD", alt_n=12, alt_page=3,
                               alt_quote="the completion time for young subjects was 27.4±7.2 s")
    verdict, _ = run_verifier(paper, screening_candidate(), payload, dataset=bock_dataset)
    assert verdict.verdict == "refuted"
    assert verdict.alt_mean == 27.4 and verdict.alt_dispersion_value == 7.2
    assert verdict.alt_dispersion_type is DispersionType.SD
    assert verdict.alt_n == 12 and verdict.alt_page == 3 and "27.4" in verdict.alt_quote


def test_a_refutation_without_a_reason_is_downgraded(paper, bock_dataset):
    payload = verifier_payload(verdict="refuted", reason="")
    verdict, _ = run_verifier(paper, screening_candidate(), payload, dataset=bock_dataset)
    assert verdict.verdict == "ambiguous" and "downgraded" in verdict.notes


def test_a_better_source_carries_its_page(paper, bock_dataset):
    payload = verifier_payload(better_source="Table 1 prints the same means per block",
                               better_source_page=4)
    verdict, _ = run_verifier(paper, screening_candidate(), payload, dataset=bock_dataset)
    assert "Table 1" in verdict.better_source and "page 4" in verdict.better_source


def test_traps_the_verifier_did_not_report_checking_are_recorded(paper, bock_dataset):
    payload = verifier_payload(checked=["Wrong Group", "units"])
    verdict, _ = run_verifier(paper, screening_candidate(), payload, dataset=bock_dataset)
    assert verdict.checked == ["wrong_group", "units"]
    assert "did not report checking" in verdict.notes and "se_vs_sd" in verdict.notes


def test_the_verifier_must_not_be_the_model_that_produced_the_candidate(paper, bock_dataset):
    with pytest.raises(ValueError, match="same family"):
        run_verifier(paper, screening_candidate(), verifier_payload(), model=OPUS,
                     dataset=bock_dataset)


def test_a_dated_model_id_is_still_the_same_family(paper, bock_dataset):
    cand = screening_candidate(model="claude-sonnet-5-20260401")
    with pytest.raises(ValueError):
        run_verifier(paper, cand, verifier_payload(), model=SONNET, dataset=bock_dataset)


def test_verifier_model_for_picks_a_different_family():
    assert verifier_model_for(screening_candidate(model=OPUS)) == SONNET
    assert verifier_model_for(screening_candidate(model=SONNET)) != SONNET


def test_a_cell_cannot_be_reopened_more_than_twice(paper, bock_dataset):
    with pytest.raises(ValueError, match="re-opened"):
        run_verifier(paper, screening_candidate(), verifier_payload(),
                     reopen=MAX_REOPENS + 1, dataset=bock_dataset)


def test_a_reopen_is_a_fresh_question_not_a_cached_answer(paper, bock_dataset):
    provider = FakeProvider([verifier_payload(), verifier_payload(verdict="ambiguous",
                                                                  reason="second look")])
    client = LLMClient(provider=provider, cache_dir=None)
    cand = screening_candidate()
    first = verify_candidate(client, paper, cand, model=SONNET, dataset=bock_dataset)
    second = verify_candidate(client, paper, cand, model=SONNET, dataset=bock_dataset, reopen=1)
    assert len(provider.requests) == 2
    assert first.reopen == 0 and second.reopen == 1
    assert "re-open 1 of 2" in second.notes
    assert provider.requests[0].key != provider.requests[1].key


def test_the_verifier_sees_the_whole_paper_and_the_extra_blocks(paper, bock_dataset):
    from canopy.llm.context import text_block

    extra = text_block("[overlay of the digitised marks]")
    _, provider = run_verifier(paper, screening_candidate(), verifier_payload(),
                               context_blocks=[extra], dataset=bock_dataset)
    blocks = provider.requests[0].messages[0]["content"]
    assert blocks[0]["type"] == "document"
    assert blocks[0]["source"]["media_type"] == "application/pdf"
    assert blocks[1] is extra
    prompt = blocks[-1]["text"]
    assert "{{" not in prompt
    assert "42.5" in prompt and "old subjects" in prompt and "screening_time" in prompt


@pytest.fixture
def file_page_registry():
    """`canopy.llm.costs` keeps a process-wide `file_id` -> page-count map so a whole-paper call
    reserves what the paper really costs. It is global state, so a test that writes to it clears
    it on both sides."""
    clear_file_pages()
    yield
    clear_file_pages()


def test_a_file_id_document_reserves_by_page_count(paper, bock_dataset, file_page_registry):
    _, provider = run_verifier(paper, screening_candidate(), verifier_payload(),
                               dataset=bock_dataset, pdf_file_id="file_abc")
    blocks = provider.requests[0].messages[0]["content"]
    assert blocks[0]["source"] == {"type": "file", "file_id": "file_abc"}
    assert file_document_tokens("file_abc") == paper.n_pages * PDF_TOKENS_PER_PAGE
    assert file_document_tokens("file_unknown") > 0


# ------------------------------------------------------------------ the adjudicator
def adjudication_payload(**overrides):
    payload = {
        "groups": [
            {"group": "A", "n": 12, "mean": 42.5, "dispersion_value": 6.9,
             "dispersion_type": "SD", "unit": "s",
             "quote": "that for old subjects was 42.5±6.9 s", "page": 3,
             "locator": "Results, screening-test paragraph", "chosen_candidate_ids": [],
             "reason": "page 3 prints it for the old group", "needs_human": False},
            {"group": "B", "n": 12, "mean": 27.4, "dispersion_value": 7.2,
             "dispersion_type": "SD", "unit": "s",
             "quote": "the completion time for young subjects was 27.4±7.2 s", "page": 3,
             "locator": "Results, screening-test paragraph", "chosen_candidate_ids": [],
             "reason": "page 3 prints it for the young group", "needs_human": False},
        ],
        "rationale": "both values are printed in the same sentence on page 3",
        "needs_human": False, "notes": ""}
    payload.update(overrides)
    return payload


def run_adjudicator(paper, dataset, candidates, payload, **kwargs):
    provider = FakeProvider([payload])
    client = LLMClient(provider=provider, cache_dir=None)
    ruling = adjudicate(client, paper, dataset, "screening_time", candidates, **kwargs)
    return ruling, provider


def test_a_ruling_is_parsed_per_group(paper, bock_dataset):
    cands = [screening_candidate("A"), screening_candidate("B", mean=27.4, dispersion_value=7.2)]
    ruling, provider = run_adjudicator(paper, bock_dataset, cands, adjudication_payload())
    assert ruling.dataset_id == bock_dataset.dataset_id and ruling.outcome_key == "screening_time"
    assert [g.group for g in ruling.groups] == ["A", "B"]
    assert ruling.group_values("A").mean == 42.5
    assert ruling.group_values("B").dispersion_type is DispersionType.SD
    assert ruling.needs_human is False and "page 3" in ruling.rationale
    assert ruling.model == MODELS["adjudicator"] and ruling.llm_call_id
    assert provider.requests[0].effort == "xhigh"


def test_a_ruling_may_only_cite_candidates_from_this_cell(paper, bock_dataset):
    cands = [screening_candidate("A")]
    payload = adjudication_payload()
    payload["groups"][0]["chosen_candidate_ids"] = [cands[0].candidate_id, "some:other:cell#3"]
    ruling, _ = run_adjudicator(paper, bock_dataset, cands, payload)
    assert ruling.group_values("A").chosen_candidate_ids == [cands[0].candidate_id]
    assert "some:other:cell#3" in ruling.notes
    assert ruling.chosen_candidate_ids == [cands[0].candidate_id]


def test_an_adjudicated_value_that_matches_a_candidate_inherits_its_provenance(paper,
                                                                              bock_dataset):
    """The adjudicator picked a reading somebody already evidenced, so it inherits that evidence
    rather than being grounded a second time."""
    cands = [screening_candidate("A")]
    ruling, _ = run_adjudicator(paper, bock_dataset, cands, adjudication_payload())
    group = ruling.group_values("A")
    assert group.grounded is True and group.page == 3
    assert group.chosen_candidate_ids == [cands[0].candidate_id]
    assert group.needs_human is False and ruling.needs_human is False


def test_an_adjudicated_value_nobody_proposed_must_stand_on_its_own_quote(paper, bock_dataset):
    payload = adjudication_payload()
    payload["groups"][0]["mean"] = 27.4                       # not what any candidate reported
    payload["groups"][0]["quote"] = "the completion time for young subjects was 27.4±7.2 s"
    ruling, _ = run_adjudicator(paper, bock_dataset, [screening_candidate("A")], payload)
    group = ruling.group_values("A")
    assert group.grounded is True and group.page == 3
    assert group.needs_human is False


def test_an_adjudicated_value_that_is_not_in_the_paper_needs_a_human(paper, bock_dataset):
    """A number that matches no candidate and cannot be found in the paper is an LLM's invention,
    and must never reach a pooled estimate as `accept_with_note`."""
    payload = adjudication_payload()
    payload["groups"][0]["mean"] = 39.9
    payload["groups"][0]["quote"] = "that for old subjects was 39.9±6.9 s"
    ruling, _ = run_adjudicator(paper, bock_dataset, [screening_candidate("A")], payload)
    group = ruling.group_values("A")
    assert group.grounded is False and group.needs_human is True
    assert "not in the paper" in group.reason
    assert ruling.needs_human is True


def test_an_adjudicated_value_with_no_quote_needs_a_human(paper, bock_dataset):
    payload = adjudication_payload()
    payload["groups"][0]["mean"] = 39.9
    payload["groups"][0]["quote"] = ""
    ruling, _ = run_adjudicator(paper, bock_dataset, [screening_candidate("A")], payload)
    group = ruling.group_values("A")
    assert group.needs_human is True and group.grounded is False
    assert "quoted nothing" in group.reason


def test_an_adjudicated_value_inherited_from_an_ungrounded_candidate_needs_a_human(paper,
                                                                                   bock_dataset):
    cand = screening_candidate("A", grounded=False, grounding_similarity=0.3)
    ruling, _ = run_adjudicator(paper, bock_dataset, [cand], adjudication_payload())
    group = ruling.group_values("A")
    assert group.needs_human is True and "not grounded" in group.reason


def test_grounding_a_ruling_is_pure_code(paper, bock_dataset):
    """`ground_adjudication` needs no model: Task 10 can re-run it after a human override."""
    from canopy.agents.adjudicator import ground_adjudication
    from canopy.models import Adjudication, AdjudicatedGroup

    ruling = Adjudication(groups=[AdjudicatedGroup(group="A", mean=42.5, quote="", page=3)])
    ground_adjudication(ruling, [], paper)
    assert ruling.groups[0].needs_human is True and ruling.needs_human is True


def test_the_adjudicator_prompt_demands_a_quote_for_every_value():
    text = load_prompt("adjudicator").lower()
    assert "quote" in text and "needs_human" in text
    assert "checked against the paper" in text


def test_a_group_the_adjudicator_skipped_needs_a_human(paper, bock_dataset):
    payload = adjudication_payload()
    payload["groups"] = payload["groups"][:1]
    ruling, _ = run_adjudicator(paper, bock_dataset, [screening_candidate("A")], payload)
    assert ruling.group_values("B").needs_human is True
    assert ruling.needs_human is True and "no ruling for group B" in ruling.notes


def test_a_ruling_without_a_rationale_needs_a_human(paper, bock_dataset):
    ruling, _ = run_adjudicator(paper, bock_dataset, [screening_candidate("A")],
                                adjudication_payload(rationale=""))
    assert ruling.needs_human is True and "cannot be reviewed" in ruling.notes


def test_the_adjudicator_sees_the_candidates_the_flags_and_the_verdicts(paper, bock_dataset):
    from canopy.models import CheckFlag
    from canopy.verify.vote import vote

    cands = [screening_candidate("A"),
             screening_candidate("A", candidate_id="other", mean=45.0, model=SONNET,
                                 extractor_id=f"text:narrative_first:{SONNET}")]
    verdicts = [VerifierVerdict(candidate_id="other", verdict="refuted",
                                reason="45.0 is not printed anywhere", model=OPUS)]
    flags = [CheckFlag(code="n_mismatch", severity="warn", message="n 9 vs 12")]
    _, provider = run_adjudicator(paper, bock_dataset, cands, adjudication_payload(),
                                  verdicts=verdicts, flags=flags,
                                  votes={"A": vote(cands, group="A")})
    prompt = provider.requests[0].messages[0]["content"][-1]["text"]
    assert "{{" not in prompt
    assert "42.5" in prompt and "45.0" in prompt
    assert "n_mismatch" in prompt and "45.0 is not printed anywhere" in prompt
    assert "VOTE for group A: disagree" in prompt


# ------------------------------------------------------------------ orientation
def orientation_payload(**overrides):
    payload = {"raw_value_semantics": "higher_more_error", "higher_is_better": "lower",
               "direction_stated_in_text": "a_greater",
               "quotes": ["the completion time for young subjects was 27.4±7.2 s"],
               "reason": "a completion time counts how long the task took, so more is worse"}
    payload.update(overrides)
    return payload


def run_orientation(paper, dataset, payloads, models=(OPUS, SONNET)):
    provider = FakeProvider(payloads)
    client = LLMClient(provider=provider, cache_dir=None)
    verdict = orientation(client, paper, dataset, dataset.outcomes[0], models, outcome=SCREENING)
    return verdict, provider


def test_two_agents_that_agree_settle_the_direction(paper, bock_dataset):
    verdict, provider = run_orientation(paper, bock_dataset,
                                        [orientation_payload(), orientation_payload()])
    assert verdict.higher_is_better is False and verdict.agreed is True
    assert verdict.needs_human is False
    assert verdict.raw_value_semantics == "higher_more_error"
    assert verdict.direction_stated_in_text == "a_greater"
    assert verdict.quotes and len(verdict.runs) == 2
    assert [r.model for r in verdict.runs] == [OPUS, SONNET]
    assert len(provider.requests) == 2 and len(verdict.llm_call_ids) == 2


def test_two_agents_that_disagree_need_a_human(paper, bock_dataset):
    verdict, _ = run_orientation(paper, bock_dataset,
                                 [orientation_payload(),
                                  orientation_payload(higher_is_better="higher",
                                                      raw_value_semantics="higher_more_construct")])
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert verdict.agreed is False and "disagree" in verdict.notes
    assert verdict.raw_value_semantics == "unknown"


def test_two_unknowns_need_a_human(paper, bock_dataset):
    payload = orientation_payload(higher_is_better="unknown", raw_value_semantics="unknown",
                                  reason="the paper never defines the measure")
    verdict, _ = run_orientation(paper, bock_dataset, [payload, payload])
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert "neither agent" in verdict.notes


def test_one_agent_alone_cannot_settle_the_direction(paper, bock_dataset):
    verdict, provider = run_orientation(paper, bock_dataset, [orientation_payload()],
                                        models=OPUS)
    assert len(provider.requests) == 1
    assert verdict.needs_human is True and "only one agent" in verdict.notes


def test_quotes_from_both_agents_travel_with_the_verdict(paper, bock_dataset):
    verdict, _ = run_orientation(
        paper, bock_dataset,
        [orientation_payload(quotes=["direction error, in degrees"]),
         orientation_payload(quotes=["direction error, in degrees", "larger values are worse"])])
    assert verdict.quotes == ["direction error, in degrees", "larger values are worse"]


def test_a_stated_direction_only_two_agents_share_is_kept(paper, bock_dataset):
    verdict, _ = run_orientation(
        paper, bock_dataset,
        [orientation_payload(direction_stated_in_text="a_greater"),
         orientation_payload(direction_stated_in_text="b_greater")])
    assert verdict.direction_stated_in_text == "unknown"
    assert "stated direction differently" in verdict.notes


def test_a_single_run_is_recorded_with_its_model_and_prompt(paper, bock_dataset):
    provider = FakeProvider([orientation_payload()])
    client = LLMClient(provider=provider, cache_dir=None)
    run = orientation_run(client, paper, bock_dataset, bock_dataset.outcomes[0], SONNET,
                          outcome=SCREENING)
    assert isinstance(run, OrientationRun)
    assert run.higher_is_better is False and run.model == SONNET
    assert run.prompt_version == ORI_VERSION and run.llm_call_id
    prompt = provider.requests[0].messages[0]["content"][-1]["text"]
    assert "{{" not in prompt and "screening test completion time" in prompt


def test_combine_orientation_is_pure_code():
    runs = [OrientationRun(higher_is_better=False, model=OPUS, reason="an error measure"),
            OrientationRun(higher_is_better=False, model=SONNET, reason="an error measure")]
    verdict = combine_orientation(runs, "screening_time", "completion time")
    assert verdict.higher_is_better is False and verdict.agreed is True
    assert verdict.outcome_key == "screening_time" and verdict.measure_name == "completion time"


# ------------------------------------------------------------------ replayed Bock 2005
@replayed
def test_bock_verifier_confirms_the_printed_screening_value(client, paper, bock_dataset):
    """"the completion time for young subjects was 27.4±7.2 s and that for old subjects was
    42.5±6.9 s" — a value printed in the running text, with the right group. A verifier that has
    the whole paper should fail to refute it."""
    cand = screening_candidate("A")
    try:
        verdict = verify_candidate(client, paper, cand, model=verifier_model_for(cand),
                                   dataset=bock_dataset, outcome=SCREENING)
    except MissingFixture:
        pytest.skip(RECORD_HINT)
    assert verdict.verdict == "confirmed", verdict.reason
    assert verdict.model != cand.model
    assert verdict.checked, "the verifier reported checking none of the traps"
    assert verdict.alt_mean is None


@replayed
def test_bock_verifier_refutes_a_swapped_group(client, paper, bock_dataset):
    """The same sentence, with the two groups exchanged: the verifier must catch it and offer the
    value the paper really prints for that group."""
    cand = screening_candidate("A", mean=27.4, dispersion_value=7.2,
                               value_as_written="27.4±7.2 s",
                               quote="the completion time for young subjects was 27.4±7.2 s")
    try:
        verdict = verify_candidate(client, paper, cand, model=verifier_model_for(cand),
                                   dataset=bock_dataset, outcome=SCREENING)
    except MissingFixture:
        pytest.skip(RECORD_HINT)
    assert verdict.verdict == "refuted", verdict.reason
    assert "wrong_group" in verdict.checked or "group" in verdict.reason.lower()
    assert verdict.alt_mean in (None, 42.5)


@replayed
def test_bock_orientation_says_a_longer_completion_time_is_worse(client, paper, bock_dataset):
    """Completion time: a larger raw value is more time, so LESS of the construct. Two agents
    have to reach that independently."""
    try:
        verdict = orientation(client, paper, bock_dataset, bock_dataset.outcomes[0],
                              outcome=SCREENING)
    except MissingFixture:
        pytest.skip(RECORD_HINT)
    assert verdict.higher_is_better is False, verdict.reason
    assert verdict.agreed is True and verdict.needs_human is False
    assert verdict.raw_value_semantics in ("higher_more_error", "unknown")
    assert verdict.quotes


@replayed
def test_bock_adjudicator_picks_the_printed_value(client, paper, bock_dataset):
    """One reader has the sentence right, one invented a number. The adjudicator has the paper."""
    from canopy.verify.checks import run_checks
    from canopy.verify.vote import vote

    right = screening_candidate("A")
    wrong = screening_candidate("A", candidate_id="wrong#1", mean=45.0,
                                value_as_written="45.0±6.9 s", model=SONNET,
                                extractor_id=f"text:narrative_first:{SONNET}",
                                quote="that for old subjects was 45.0±6.9 s", grounded=False)
    cands = [right, wrong]
    try:
        ruling = adjudicate(client, paper, bock_dataset, "screening_time", cands,
                            flags=run_checks(bock_dataset, "screening_time", cands),
                            votes={"A": vote(cands, group="A")}, outcome_def=SCREENING)
    except MissingFixture:
        pytest.skip(RECORD_HINT)
    group = ruling.group_values("A")
    assert group is not None and group.mean == 42.5, ruling.rationale
    assert ruling.rationale
