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
from canopy.agents.orientation import (MEANS_CHECK_NOTE, ORIENTATION_SCHEMA,
                                       PROMPT_VERSION as ORI_VERSION, TIEBREAK_EFFORT,
                                       TIEBREAK_MODEL, combine_orientation, orientation,
                                       orientation_run)
from canopy.agents.verifier import (MAX_REOPENS, PROMPT_VERSION, REFUTATION_TARGETS,
                                    VERIFIER_SCHEMA, verifier_model_for, verify_candidate)
from canopy.config import MODELS
from canopy.llm.client import REISSUE_CACHE_KEY, LLMClient
from canopy.llm.costs import PDF_TOKENS_PER_PAGE, clear_file_pages, file_document_tokens
from canopy.llm.errors import BudgetExceeded, MissingFixture
from canopy.llm.providers import FakeProvider
from canopy.llm.schemas import assert_no_derived_stats, assert_valid_output_schema
from canopy.models import (Candidate, DatasetSpec, DispersionType, GroupSpec, OrientationRun,
                           OutcomeDef, OutcomeSources, Source, SourceKind, VerifierVerdict)
from canopy.stats.effect_sizes import smd_from_means
from canopy.verify.checks import DISPUTED_MEANS_FLAGS, codes, run_checks
from canopy.verify.vote import model_family

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


def test_an_inherited_verdict_comes_with_the_quote_it_was_a_verdict_ON(paper, bock_dataset):
    """The quote and the grounding verdict travel together. Showing the adjudicator's own words
    beside a candidate's `grounded` would label an unchecked quote as checked."""
    payload = adjudication_payload()
    payload["groups"][0]["quote"] = "old subjects took about forty-two seconds"   # not in the paper
    ruling, _ = run_adjudicator(paper, bock_dataset, [screening_candidate("A")], payload)
    group = ruling.group_values("A")
    assert group.quote == screening_candidate("A").quote        # the checked one, not the model's
    assert group.grounded is True
    assert "forty-two seconds" in group.reason and "was checked" in group.reason


def test_a_quote_beside_a_digitised_candidate_is_checked_on_its_own(paper, bock_dataset):
    """A digitised candidate has pixels, not words, so there is no verdict to inherit: the
    adjudicator's own quote has to be grounded itself."""
    figure = screening_candidate("A", candidate_id="fig#1", quote="", grounded=None,
                                 grounding_similarity=None, model="",
                                 extractor_id="digitize:ensemble", route="figure")
    payload = adjudication_payload()
    payload["groups"][0]["quote"] = "that for old subjects was 42.5±6.9 s"
    ruling, _ = run_adjudicator(paper, bock_dataset, [figure], payload)
    group = ruling.group_values("A")
    assert group.grounded is True and group.grounding_similarity is not None
    assert group.needs_human is False

    payload["groups"][0]["quote"] = "old subjects averaged 42.5 furlongs per fortnight"
    ruling, _ = run_adjudicator(paper, bock_dataset, [figure], payload)
    group = ruling.group_values("A")
    # the VALUE still stands — a candidate reported it — but the justification does not, and it
    # says so; `confidence` reads `grounded` and takes the ungrounded penalty
    assert group.grounded is False
    assert "not in the paper" in group.reason and figure.candidate_id in group.reason
    assert group.needs_human is False


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
#: a justification long enough to be a justification: C12 records a stub reply as `not_run`, and a
#: fixture whose `reason` is three words would be testing the detector by accident.
AN_ERROR_MEASURE = ("a completion time counts how long the task took, so a larger number is worse "
                    "on this measure")


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
    payload = orientation_payload(
        higher_is_better="unknown", raw_value_semantics="unknown",
        reason="the paper never defines this measure, and nothing in the methods says which way "
               "a larger number points")
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
    runs = [OrientationRun(higher_is_better=False, model=OPUS, reason=AN_ERROR_MEASURE),
            OrientationRun(higher_is_better=False, model=SONNET, reason=AN_ERROR_MEASURE)]
    verdict = combine_orientation(runs, "screening_time", "completion time")
    assert verdict.higher_is_better is False and verdict.agreed is True
    assert verdict.outcome_key == "screening_time" and verdict.measure_name == "completion time"


# ------------------------------------------------- C3 / C12 / P-A: who may decide a direction
#: The four real cells the ceiling items argue about, as `smd_from_means` inputs. Every "must not
#: print this number" assertion below is written against the arithmetic, not against a paraphrase.
BOCK_D1 = (-22.4, 8.3, 12, -27.0, 5.5, 12)          # aftereffect: published -0.537
BOCK_D2 = (52.4905, 9.15, 12, 27.9830, 7.3, 12)     # tracking RMSE: two agreeing readers, -2.9610
CRESSMAN_LATE = (31.1, 6.0, 9, 33.1, 10.75, 10)     # pooled at -0.2263, must not move
ERROR_TYPE_12_6 = (12.0, 2.0, 20, 6.0, 2.0, 20)     # the reviewer's absolute-pointing-error case

REASON = ("the measure is defined in the methods section and the paper says which way a larger "
          "number points, which is what this answer rests on")


def ballot(model=OPUS, higher_is_better=None, raw_value_semantics="unknown",
           direction_stated_in_text="unknown", reason=REASON):
    return OrientationRun(higher_is_better=higher_is_better,
                          raw_value_semantics=raw_value_semantics,
                          direction_stated_in_text=direction_stated_in_text, reason=reason,
                          model=model, quotes=["a quote from the paper"])


def d_of(values, higher_is_better):
    return round(smd_from_means(*values, higher_is_better=higher_is_better).d, 4)


#: `run_checks` needs a dataset to hang an outcome on; these tests are about the orientation
#: verdict alone, so it is the emptiest one that exists.
BARE_DATASET = DatasetSpec(dataset_id="orientation-only")


def outcome_flags(verdict):
    """The check codes an orientation verdict puts in front of `confidence`."""
    return codes(run_checks(BARE_DATASET, "screening_time", [], orientation=verdict))


def orientation_bucket(verdict):
    """`(bucket, score)` for a cell whose ONLY doubt is how the direction of its measure was set.

    Two clean text readings from two model families: `auto_accept` at 0.80 when the orientation
    raises nothing at all, so every drop below that here is an orientation code and nothing else.
    A code that says "reviewed, never automatic" has to move this, and asserting the code alone
    would not notice if it were dropped from `confidence.CAPPING_FLAGS` tomorrow (F14).
    """
    from canopy.verify.confidence import confidence
    from canopy.verify.vote import vote

    cands = [screening_candidate("A"),
             screening_candidate("A", candidate_id="second", model=SONNET,
                                 extractor_id=f"text:narrative_first:{SONNET}")]
    flags = run_checks(BARE_DATASET, "screening_time", [], orientation=verdict)
    bucket, score, _reasons = confidence(vote(cands), (), flags, None, candidates=cands,
                                         n_a=12, n_b=12, orientation=verdict)
    return bucket, score


# --- C3: the check may discard or abstain. It may never choose.
def test_an_error_type_measure_two_readers_split_on_is_never_signed():
    """DECISION-v2 C3 test 1 — the 12/6 case that v1's semantics branch printed as +3.0.

    Both readers call the raw scale `higher_more_construct` (the construct IS pointing error), and
    the guard `|mean| >= 2.5*SD` passes on both groups, so nothing else in the pipeline is standing
    between this cell and a wrong sign. The verdict has to be an abstention.
    """
    verdict = combine_orientation([ballot(OPUS, True, "higher_more_construct"),
                                   ballot(SONNET, False, "higher_more_construct")],
                                  "pointing_error", "absolute pointing error (deg)")
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert verdict.agreed is False
    assert d_of(ERROR_TYPE_12_6, True) == 3.0        # the number v1 would have printed
    assert "orientation_unknown" in outcome_flags(verdict)


def test_the_same_split_on_an_error_semantics_reader_is_also_never_signed():
    """DECISION-v2 C3 test 2 — flipping the semantics field changes nothing, because no row of the
    table reads it. Polarity is not a thing arithmetic can recover."""
    verdict = combine_orientation([ballot(OPUS, True, "higher_more_error"),
                                   ballot(SONNET, False, "higher_more_error")],
                                  "pointing_error", "absolute pointing error (deg)")
    assert verdict.higher_is_better is None and verdict.needs_human is True


@pytest.mark.parametrize("semantics", ["higher_more_construct", "higher_more_error",
                                       "signed_direction", "unknown"])
def test_bock_tracking_rmse_cannot_be_flipped_by_what_the_readers_call_the_scale(semantics):
    """DECISION-v2 C3 test 3 — Bock d2, the cell v1's rule 2 would have turned into +2.9610.

    Its two real ballots agree on `hib=False`. Whatever they call the raw scale, the verdict must
    stay `False` and the effect must stay negative: `raw_value_semantics` appears in no row of the
    decision table, so there is no branch that can reverse a sign here.
    """
    verdict = combine_orientation([ballot(OPUS, False, semantics),
                                   ballot(SONNET, False, semantics)],
                                  "late_adaptation", "Tracking RMSE")
    assert verdict.higher_is_better is False and verdict.agreed is True
    assert verdict.needs_human is False
    assert d_of(BOCK_D2, verdict.higher_is_better) == -2.9610
    assert d_of(BOCK_D2, verdict.higher_is_better) != 2.9610


def test_bock_aftereffect_abstains_instead_of_pooling_a_positive_number(paper, bock_dataset):
    """DECISION-v2 C3 test 4 — the real Bock d1 ballots: opus `signed_direction/False`, sonnet
    `higher_more_construct/True`, neither stating a direction. Nothing is checkable, so nothing is
    discarded, so the readers still disagree: ABSTAIN, and buy nothing while doing it."""
    verdict, provider = run_orientation(paper, bock_dataset, [
        orientation_payload(higher_is_better="lower", raw_value_semantics="signed_direction",
                            direction_stated_in_text="unknown", reason=REASON),
        orientation_payload(higher_is_better="higher", raw_value_semantics="higher_more_construct",
                            direction_stated_in_text="unknown", reason=REASON)])
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert len(provider.requests) == 2               # zero paid tie-break calls before the question
    assert d_of(BOCK_D1, True) == 0.6534             # the number this must never reach
    assert "orientation_unknown" in outcome_flags(verdict)


def test_a_discard_never_leaves_the_silent_reader_deciding_alone():
    """ADVERSARIAL round 2 on C3, the one change that kept the item off the boat.

    opus states `b_greater`, which contradicts the raw means (A = -22.4 is greater than B = -27.0),
    so row 1 discards it — and the ballot it discards is the one holding the RIGHT answer
    (`hib=False` gives the published sign). Row 5 must not be reachable from here: the filter can
    only ever remove a reader that committed to a checkable claim, so handing the measure to the
    reader that stayed silent would prefer silence to evidence, every time.
    """
    verdict = combine_orientation(
        [ballot(OPUS, False, "signed_direction", "b_greater"),
         ballot(SONNET, True, "higher_more_construct", "unknown")],
        "aftereffect", "aftereffect (deg)", mean_a=-22.4, mean_b=-27.0)
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert d_of(BOCK_D1, True) == 0.6534             # what row 5 would have pooled
    assert d_of(BOCK_D1, False) == -0.6534           # what the DISCARDED reader was right about
    flags = outcome_flags(verdict)
    assert "orientation_reader_contradicts_values" in flags
    assert "orientation_single_witness" not in flags
    assert "b greater" in verdict.notes and "-22.4" in verdict.notes


@pytest.mark.parametrize("code", sorted(DISPUTED_MEANS_FLAGS))
def test_no_reader_is_discarded_while_which_series_is_which_is_disputed(code):
    """ADVERSARIAL round 2 change 2 — the discard compares a reader against the RESOLVED means, so
    it must not fire on a cell whose series identity is open. Bock d1 carries
    `series_marker_mismatch` today; under swapped series the reader row 1 would throw out is the
    one that read the paper correctly. `group_label_swapped` joins the set in this fix round: it
    is an `error` that says the two groups may be the wrong way round, which is the same claim
    `series_transposed` makes about a figure."""
    runs = [ballot(OPUS, False, "signed_direction", "b_greater"),
            ballot(SONNET, True, "higher_more_construct", "unknown")]
    verdict = combine_orientation(runs, "aftereffect", "aftereffect (deg)", mean_a=-22.4,
                                  mean_b=-27.0, open_flags=[code])
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert "orientation_reader_contradicts_values" not in outcome_flags(verdict)
    assert "which series is which is disputed" in verdict.notes


def test_a_reader_whose_stated_direction_matches_the_means_is_kept():
    """Row 2 — the filter has no appetite. A checkable claim that checks out is just a ballot."""
    verdict = combine_orientation([ballot(OPUS, False, "higher_more_error", "a_greater"),
                                   ballot(SONNET, False, "higher_more_error", "a_greater")],
                                  "late_adaptation", "pointing error", mean_a=-22.4, mean_b=-27.0)
    assert verdict.higher_is_better is False and verdict.agreed is True
    assert "orientation_reader_contradicts_values" not in outcome_flags(verdict)


def test_cressman_late_adaptation_still_pools_unchanged():
    """DECISION-v2 C3 test 6 — the regression guard on the paper supplying two of three pooled
    cells. Positive means, two agreeing readers, no stated direction: row 4, untouched."""
    verdict = combine_orientation([ballot(OPUS, True, "higher_more_construct"),
                                   ballot(SONNET, True, "higher_more_construct")],
                                  "late_adaptation", "hand deviation at peak velocity",
                                  mean_a=31.1, mean_b=33.1)
    assert verdict.higher_is_better is True and verdict.agreed is True
    assert verdict.needs_human is False
    assert d_of(CRESSMAN_LATE, True) == -0.2263
    assert outcome_flags(verdict) == []


def test_an_accepted_reversal_does_not_move_the_pooled_sd():
    """DECISION-v2 C3 test 7 — sign-flip invariance. Orientation decides a sign and nothing else;
    if it moved a dispersion it would be arithmetic, and it is not allowed to be."""
    both = [smd_from_means(*BOCK_D1, higher_is_better=hib) for hib in (True, False)]
    assert both[0].details["pooled_sd"] == both[1].details["pooled_sd"]
    assert both[0].details["pooled_sd"] == pytest.approx(7.0406, abs=1e-4)
    assert both[0].se == both[1].se and both[0].d == -both[1].d


def test_two_readers_reading_the_paper_backwards_is_a_flag_not_a_note():
    """C3 rule 6 — live on Heuer Exp 1a today: opus `a_greater` against sonnet `b_greater`,
    collapsed to `unknown` with `agreed: true, needs_human: false`, which also silently disables
    `sign_check`. The disagreement has to reach `confidence`."""
    verdict = combine_orientation([ballot(OPUS, False, "signed_direction", "a_greater"),
                                   ballot(SONNET, False, "signed_direction", "b_greater")],
                                  "late_adaptation", "adaptive shift")
    assert verdict.agreed is True and verdict.direction_stated_in_text == "unknown"
    assert "orientation_direction_conflict" in outcome_flags(verdict)


def test_a_disagreement_about_the_raw_scale_is_recorded_not_swallowed():
    """C3 rule 5 — the summary field cannot hold two answers, but it must not eat them either."""
    verdict = combine_orientation([ballot(OPUS, True, "signed_direction"),
                                   ballot(SONNET, True, "higher_more_construct")],
                                  "aftereffect", "aftereffect (deg)")
    assert verdict.raw_value_semantics == "unknown"
    assert f"{OPUS}=signed_direction" in verdict.notes
    assert f"{SONNET}=higher_more_construct" in verdict.notes


# --- C12: a degenerate reply is not a witness
def test_a_stub_reply_is_re_issued_once_and_the_cell_survives_it(paper, bock_dataset):
    """C12 acceptance test (i) — Cressman's aftereffect, the cell v1's C12 unpooled. The stub is
    re-asked past the cache, the re-issue is coherent, and the verdict is an ordinary agreement."""
    verdict, provider = run_orientation(paper, bock_dataset, [
        orientation_payload(reason="placeholder"),
        orientation_payload(reason=REASON),
        orientation_payload(reason=REASON)])
    assert len(provider.requests) == 3               # two readers, exactly one re-issue
    keys = [r.key for r in provider.requests]
    assert keys[0] != keys[1]                   # the re-issue cannot hit the stub's cache entry
    assert verdict.agreed is True and verdict.needs_human is False
    assert verdict.higher_is_better is False
    assert outcome_flags(verdict) == []              # nothing is capped: two real readers spoke
    # F14: the bucket, not the code list. A re-issue that worked leaves an ordinary agreement, so
    # this cell is not held back and not capped — that is the Cressman regression this guards.
    assert orientation_bucket(verdict) == ("auto_accept", 0.80)


def test_a_stub_that_stays_a_stub_leaves_one_flagged_witness(paper, bock_dataset):
    """C12 acceptance test (ii) — the ONLY route to a one-reader verdict. `hib` is set (so the
    cell is not deleted) and flagged (so it is not reported as an agreement)."""
    verdict, provider = run_orientation(paper, bock_dataset, [
        orientation_payload(reason="placeholder"),
        orientation_payload(reason="placeholder"),
        orientation_payload(reason=REASON)])
    assert len(provider.requests) == 3
    assert verdict.higher_is_better is False and verdict.needs_human is False
    assert verdict.agreed is False
    flags = outcome_flags(verdict)
    assert "orientation_single_witness" in flags and "orientation_unknown" not in flags
    assert len(verdict.runs) == 3                    # both raw replies stay on the record
    # F14: the cap C12 names. The cell still POOLS — `accept_with_note` is a pooling bucket — but
    # it can no longer be accepted without anybody looking at it.
    assert orientation_bucket(verdict) == ("accept_with_note", 0.70)


def test_two_degenerate_replies_are_a_question_not_a_verdict(paper, bock_dataset):
    """C12 acceptance test (iii) — no coherent ballot survives, so nobody decides."""
    verdict, provider = run_orientation(paper, bock_dataset,
                                        [orientation_payload(reason="placeholder")])
    assert len(provider.requests) == 4               # two readers, one re-issue each, and no more
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert "orientation_unknown" in outcome_flags(verdict)
    assert "not_run" in verdict.notes


def test_a_degenerate_reply_is_re_issued_at_most_once(paper, bock_dataset):
    """C12 acceptance test (v) — the re-issue is a bounded cost, not a retry loop."""
    _, provider = run_orientation(paper, bock_dataset,
                                  [orientation_payload(reason="placeholder")], models=OPUS)
    assert len(provider.requests) == 2
    assert provider.requests[0].key != provider.requests[1].key


# --- P-A residue: an abstention is not a dissent, and a third read is bought last or never
def test_a_reader_that_abstained_is_not_recorded_as_a_dissenter():
    """P-A residue (a). One reader answered, one said "I cannot tell". That is one answer and one
    abstention — and one answer still does not settle a direction."""
    verdict = combine_orientation([ballot(OPUS, True, "higher_more_construct"),
                                   ballot(SONNET, None, "unknown")],
                                  "aftereffect", "aftereffect (deg)")
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert "disagree" not in verdict.notes
    assert "one reader never decides" in verdict.notes


def test_no_third_read_is_bought_unless_the_caller_asks_for_one(paper, bock_dataset):
    """P-A: `tiebreak=None` is HEAD — the same two calls, whatever the readers said."""
    verdict, provider = run_orientation(paper, bock_dataset, [
        orientation_payload(higher_is_better="lower", raw_value_semantics="signed_direction"),
        orientation_payload(higher_is_better="higher", raw_value_semantics="signed_direction")])
    assert verdict.needs_human is True and len(provider.requests) == 2


def test_a_third_read_is_bought_only_after_the_free_check_abstains(paper, bock_dataset):
    """P-A: one extra ballot, at `xhigh`, from a family that failed differently, and what it
    settles is a MAJORITY — flagged and capped, never reported as an independent agreement."""
    provider = FakeProvider([
        orientation_payload(higher_is_better="lower", raw_value_semantics="signed_direction",
                            reason=REASON),
        orientation_payload(higher_is_better="higher", raw_value_semantics="signed_direction",
                            reason=REASON),
        orientation_payload(higher_is_better="lower", raw_value_semantics="signed_direction",
                            reason=REASON)])
    client = LLMClient(provider=provider, cache_dir=None)
    verdict = orientation(client, paper, bock_dataset, bock_dataset.outcomes[0], (OPUS, SONNET),
                          outcome=SCREENING, tiebreak=TIEBREAK_MODEL)
    assert len(provider.requests) == 3
    assert provider.requests[-1].model == TIEBREAK_MODEL
    assert provider.requests[-1].effort == TIEBREAK_EFFORT
    assert model_family(TIEBREAK_MODEL) not in {model_family(OPUS), model_family(SONNET)}
    assert verdict.higher_is_better is False and verdict.needs_human is False
    assert verdict.agreed is False                   # 2-1 is not two readers agreeing
    assert "orientation_by_majority" in outcome_flags(verdict)


def test_a_majority_that_read_a_different_scale_is_not_a_majority(paper, bock_dataset):
    """P-A: the three must share a `raw_value_semantics`, or their 2-1 is a coincidence."""
    provider = FakeProvider([
        orientation_payload(higher_is_better="lower", raw_value_semantics="signed_direction",
                            reason=REASON),
        orientation_payload(higher_is_better="higher", raw_value_semantics="higher_more_construct",
                            reason=REASON),
        orientation_payload(higher_is_better="lower", raw_value_semantics="higher_more_error",
                            reason=REASON)])
    client = LLMClient(provider=provider, cache_dir=None)
    verdict = orientation(client, paper, bock_dataset, bock_dataset.outcomes[0], (OPUS, SONNET),
                          outcome=SCREENING, tiebreak=TIEBREAK_MODEL)
    assert len(provider.requests) == 3
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert "do not agree on what the raw scale IS" in verdict.notes


def test_a_third_read_that_cannot_be_afforded_leaves_the_question_standing(paper, bock_dataset):
    """P-A: budget-gated. A tie-break nobody can pay for is an abstention, not a crash — the cell
    keeps the `orientation` question it already had."""
    def broke(request):
        raise BudgetExceeded("the run has spent its budget")

    provider = FakeProvider([
        orientation_payload(higher_is_better="lower", raw_value_semantics="signed_direction",
                            reason=REASON),
        orientation_payload(higher_is_better="higher", raw_value_semantics="signed_direction",
                            reason=REASON),
        broke])
    client = LLMClient(provider=provider, cache_dir=None)
    verdict = orientation(client, paper, bock_dataset, bock_dataset.outcomes[0], (OPUS, SONNET),
                          outcome=SCREENING, tiebreak=TIEBREAK_MODEL)
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert "no third read was bought" in verdict.notes


def test_a_majority_settled_direction_is_capped_below_automatic_acceptance():
    """P-A: `orientation_by_majority` is a code `confidence` can see; a cell settled 2-1 by a
    bought read must be reviewed, never auto-accepted."""
    verdict = combine_orientation([ballot(OPUS, False, "signed_direction"),
                                   ballot(SONNET, True, "signed_direction"),
                                   ballot(TIEBREAK_MODEL, False, "signed_direction")],
                                  "aftereffect", "aftereffect (deg)", third_read=True)
    assert verdict.higher_is_better is False and verdict.needs_human is False
    assert "orientation_by_majority" in outcome_flags(verdict)
    # F14: "must be reviewed, never auto-accepted" is a BUCKET claim. Asserting the code alone
    # would not notice if the code left `confidence.CAPPING_FLAGS`.
    assert orientation_bucket(verdict) == ("accept_with_note", 0.70)


def test_a_tie_among_three_readers_is_still_a_question():
    """An even split is not a majority, whatever it cost to buy."""
    verdict = combine_orientation([ballot(OPUS, False, "signed_direction"),
                                   ballot(SONNET, True, "signed_direction"),
                                   ballot(TIEBREAK_MODEL, None, "signed_direction")],
                                  "aftereffect", "aftereffect (deg)", third_read=True)
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert "a tie is not a majority" in verdict.notes


# ------------------------------------------------- fix round 1: who may decide, once a discard,
# ------------------------------------------------- a stub or a bought read is in the room
def test_a_discard_may_not_let_silent_readers_settle_a_direction_the_numbers_checked():
    """Fix round F1 — the reviewer's executed input, at n = 3, where row 4 used to sign it.

    opus is the only reader that made a checkable claim (`b_greater`, against A = -22.4 > B =
    -27.0) and it is the one the filter removes; the two readers left never stated a direction at
    all. Letting them agree +0.6534 into the pool is the sign inversion ADVERSARIAL round 2
    refused to ship C3 for, reached at n = 3 instead of n = 2 and with a worse outcome — `agreed`,
    uncapped — than the one the n = 2 rule blocked.
    """
    verdict = combine_orientation(
        [ballot(OPUS, False, "signed_direction", "b_greater"),
         ballot(SONNET, True, "higher_more_construct"),
         ballot(TIEBREAK_MODEL, True, "higher_more_construct")],
        "aftereffect", "aftereffect (deg)", mean_a=-22.4, mean_b=-27.0, third_read=True)
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert verdict.agreed is False
    assert d_of(BOCK_D1, True) == 0.6534             # the number this must never pool
    flags = outcome_flags(verdict)
    assert "orientation_reader_contradicts_values" in flags
    assert "orientation_unknown" in flags
    assert "orientation_by_majority" not in flags
    assert orientation_bucket(verdict)[0] == "needs_human"


def test_the_same_three_ballots_without_a_bought_read_are_no_more_able_to_sign_it():
    """The same shape reached the other way — three named models rather than a tie-break — because
    the bias the ruling names is in the DISCARD, not in who paid for the third ballot."""
    verdict = combine_orientation(
        [ballot(OPUS, False, "signed_direction", "b_greater"),
         ballot(SONNET, True, "higher_more_construct"),
         ballot(TIEBREAK_MODEL, True, "higher_more_construct")],
        "aftereffect", "aftereffect (deg)", mean_a=-22.4, mean_b=-27.0)
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert verdict.agreed is False


def test_a_polarity_every_reader_agrees_on_survives_a_discard_and_the_cell_is_still_held():
    """Fix round F1, the other half of the controller's ruling: a discard is not a veto on a
    polarity nobody contradicted.

    All three readers say `hib=False`; one of them also states a direction this cell's resolved
    means contradict. The polarity is recorded — the reviewer's question here is about the VALUES,
    not about which way the measure points — and the cell is held anyway, by an `error` flag that
    no score can absorb.
    """
    verdict = combine_orientation(
        [ballot(OPUS, False, "signed_direction", "b_greater"),
         ballot(SONNET, False, "higher_more_error"),
         ballot(TIEBREAK_MODEL, False, "higher_more_error")],
        "aftereffect", "aftereffect (deg)", mean_a=-22.4, mean_b=-27.0)
    assert verdict.higher_is_better is False
    assert d_of(BOCK_D1, False) == -0.6534
    flag = next(f for f in run_checks(BARE_DATASET, "screening_time", [], orientation=verdict)
                if f.code == "orientation_reader_contradicts_values")
    assert flag.severity == "error"
    assert orientation_bucket(verdict)[0] == "needs_human"


def test_a_bought_third_read_never_sets_the_direction_alone(paper, bock_dataset):
    """Fix round F2 — both readers' replies did not happen, twice each, and the only ballot left
    is the one that was BOUGHT.

    Row 5 exists for C12: ONE ORIGINAL reader decided and its partner's reply did not happen. A
    third read is not that reader. `orientation_single_witness` caps a cell at `accept_with_note`,
    which is a POOLING bucket, so this path would have put one model's single ballot behind the
    sign of a pooled effect.
    """
    provider = FakeProvider([orientation_payload(reason="placeholder"),
                             orientation_payload(reason="placeholder"),
                             orientation_payload(reason="placeholder"),
                             orientation_payload(reason="placeholder"),
                             orientation_payload(reason=REASON)])
    client = LLMClient(provider=provider, cache_dir=None)
    verdict = orientation(client, paper, bock_dataset, bock_dataset.outcomes[0], (OPUS, SONNET),
                          outcome=SCREENING, tiebreak=TIEBREAK_MODEL)
    assert len(provider.requests) == 5               # two readers, one re-issue each, one third
    assert provider.requests[-1].model == TIEBREAK_MODEL
    assert verdict.higher_is_better is None and verdict.needs_human is True
    flags = outcome_flags(verdict)
    assert "orientation_single_witness" not in flags
    assert "orientation_unknown" in flags
    assert orientation_bucket(verdict)[0] == "needs_human"


def test_an_original_reader_may_still_be_a_single_witness_when_its_partner_did_not_reply():
    """The path row 5 is FOR, kept working: one original reader decided, the other original's
    reply did not happen. This is C12's route and the only one, and it is what keeps Cressman's
    two pooled cells pooled."""
    verdict = combine_orientation(
        [ballot(OPUS, True, "higher_more_construct", reason="placeholder"),
         ballot(SONNET, True, "higher_more_construct")],
        "late_adaptation", "hand deviation at peak velocity", mean_a=31.1, mean_b=33.1)
    assert verdict.higher_is_better is True and verdict.needs_human is False
    assert "orientation_single_witness" in outcome_flags(verdict)
    assert orientation_bucket(verdict)[0] == "accept_with_note"


def test_a_third_read_that_settles_a_measure_is_never_reported_as_two_readers_agreeing():
    """Fix round F3 — the reviewer's executed input. opus names a direction, sonnet abstains, and
    the bought read agrees with opus about the direction while reading a DIFFERENT raw scale.

    Row 4 was evaluated first, so this came out `agreed=True, needs_human=False, flags=[]`: a
    verdict that exists only because money was spent, free to reach `auto_accept`, carrying none
    of P-A's four mandatory conditions. `third_read` now gates before row 4, so the verdict goes
    through the majority branch — and that branch's shared-scale condition refuses this one.
    """
    verdict = combine_orientation([ballot(OPUS, True, "signed_direction"),
                                   ballot(SONNET, None, "unknown"),
                                   ballot(TIEBREAK_MODEL, True, "higher_more_error")],
                                  "aftereffect", "aftereffect (deg)", third_read=True)
    assert verdict.agreed is False
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert "do not agree on what the raw scale IS" in verdict.notes
    assert outcome_flags(verdict) == ["orientation_unknown"]
    assert orientation_bucket(verdict)[0] == "needs_human"


def test_a_third_read_that_settles_a_measure_carries_its_flag_and_its_cap():
    """The same three ballots with P-A's mandatory condition met — one shared `raw_value_
    semantics` — so the third read really does settle it. What money settled is flagged
    `orientation_by_majority` and capped below automatic acceptance, never `agreed`."""
    verdict = combine_orientation([ballot(OPUS, True, "signed_direction"),
                                   ballot(SONNET, None, "signed_direction"),
                                   ballot(TIEBREAK_MODEL, True, "signed_direction")],
                                  "aftereffect", "aftereffect (deg)", third_read=True)
    assert verdict.higher_is_better is True and verdict.needs_human is False
    assert verdict.agreed is False
    assert "orientation_by_majority" in outcome_flags(verdict)
    assert orientation_bucket(verdict)[0] == "accept_with_note"


def test_a_stub_is_not_a_reader_whose_stated_direction_the_numbers_can_contradict():
    """Fix round F4 — the reviewer's executed input, on the Cressman-aftereffect shape.

    C12 has already said this reply did not happen, so its `direction_stated_in_text` is not a
    claim anybody made. Running the discard filter over it unpools a correct cell — v1's C12
    failure mode through the back door — and puts a reader that never replied on the record as
    having contradicted the paper.
    """
    verdict = combine_orientation(
        [ballot(SONNET, True, "higher_more_construct", "unknown"),
         ballot(OPUS, True, "higher_more_construct", "a_greater", reason="placeholder")],
        "late_adaptation", "hand deviation at peak velocity", mean_a=31.1, mean_b=33.1)
    assert verdict.higher_is_better is True and verdict.needs_human is False
    assert d_of(CRESSMAN_LATE, True) == -0.2263
    flags = outcome_flags(verdict)
    assert "orientation_single_witness" in flags
    assert "orientation_reader_contradicts_values" not in flags
    assert f"{OPUS} states" not in verdict.notes     # nobody is accused of a reply they never made


def test_two_readers_reading_the_paper_backwards_is_still_a_flag_once_the_means_arrive():
    """Fix round F7 — C3 rule 6 on the production path, which is now the only path.

    Heuer Exp 1a with the means the integration passes: opus `a_greater`, sonnet `b_greater`,
    A = 27.7 > B = 18.9. The discard used to consume the conflict — `directions` was taken from
    the survivors, one direction was left, and the flag never fired — so a flat contradiction
    between two readers about the paper's own sentence vanished from the record. A discarded
    reader's VOTE is what the filter removes; what it said about the paper still happened.
    """
    verdict = combine_orientation([ballot(OPUS, False, "signed_direction", "a_greater"),
                                   ballot(SONNET, False, "signed_direction", "b_greater")],
                                  "late_adaptation", "adaptive shift", mean_a=27.7, mean_b=18.9)
    assert verdict.direction_stated_in_text == "unknown"
    flags = outcome_flags(verdict)
    assert "orientation_direction_conflict" in flags
    assert "orientation_reader_contradicts_values" in flags
    assert verdict.higher_is_better is None          # one counting reader is not two


def test_a_stub_does_not_get_a_vote_on_what_the_raw_scale_is():
    """Fix round F10 — the summary field C3 rule 5 exists to protect. One real reader saying
    `higher_more_error` beside one stub parsed as `unknown` collapsed the summary to "unknown"
    and appended a per-reader note about a reader that never replied."""
    verdict = combine_orientation(
        [ballot(SONNET, False, "higher_more_error"),
         ballot(OPUS, True, "unknown", reason="placeholder")],
        "late_adaptation", "pointing error")
    assert verdict.raw_value_semantics == "higher_more_error"
    assert verdict.raw_value_semantics_by_model[SONNET] == "higher_more_error"
    assert "raw_value_semantics per reader" not in verdict.notes


@pytest.mark.parametrize("kwargs, state", [
    (dict(mean_a=31.1, mean_b=33.1), "ran"),
    (dict(), "no_means"),
    (dict(mean_a=31.1, mean_b=33.1, open_flags=["series_marker_mismatch"]), "disputed"),
    (dict(mean_a=31.1, mean_b=33.1, open_flags=["group_label_swapped"]), "disputed"),
])
def test_every_verdict_says_whether_the_discard_check_could_run(kwargs, state):
    """Fix round F11 — an `agreed` verdict used to read identically whether the filter ran and
    found nothing, the means were missing, or which series is which was in dispute. Those are
    three different claims about how well this direction was checked, and a reviewer cannot tell
    a checked cell from an unchecked one without being told."""
    verdict = combine_orientation([ballot(OPUS, True, "higher_more_construct"),
                                   ballot(SONNET, True, "higher_more_construct")],
                                  "late_adaptation", "hand deviation", **kwargs)
    assert f"{MEANS_CHECK_NOTE}: {state}" in verdict.notes
    assert verdict.higher_is_better is True          # saying so changes no verdict


def test_the_third_reads_first_reply_stays_on_the_record(paper, bock_dataset):
    """Fix round F12 — the two ordinary readers keep both replies when a stub stays a stub, and
    the bought read used to have its first reply overwritten by its own re-issue. Its re-issue was
    also never re-checked, so a doubly-degenerate third read was appended as a silent witness."""
    provider = FakeProvider([
        orientation_payload(higher_is_better="lower", raw_value_semantics="signed_direction",
                            reason=REASON),
        orientation_payload(higher_is_better="higher", raw_value_semantics="signed_direction",
                            reason=REASON),
        orientation_payload(reason="placeholder"),
        orientation_payload(reason="placeholder")])
    client = LLMClient(provider=provider, cache_dir=None)
    verdict = orientation(client, paper, bock_dataset, bock_dataset.outcomes[0], (OPUS, SONNET),
                          outcome=SCREENING, tiebreak=TIEBREAK_MODEL)
    assert len(provider.requests) == 4               # two readers, one third read, one re-issue
    assert [r.model for r in provider.requests[-2:]] == [TIEBREAK_MODEL, TIEBREAK_MODEL]
    assert provider.requests[-2].key != provider.requests[-1].key
    assert [r.model for r in verdict.runs].count(TIEBREAK_MODEL) == 2
    assert all(r.not_run for r in verdict.runs if r.model == TIEBREAK_MODEL)
    assert verdict.higher_is_better is None and verdict.needs_human is True


def test_a_reply_the_record_already_calls_a_non_reply_is_not_re_derived_from_its_prose():
    """Fix round F13 — `OrientationRun.not_run` was written and never read, while `models.py` says
    recording it is what makes a replayed `verify.json` state it rather than re-derive it. A
    replay must not be able to promote a ballot the live run classified as a non-reply back into a
    witness (a detector that is tuned later would do exactly that)."""
    stub = ballot(OPUS, True, "higher_more_construct").model_copy(update={"not_run": True})
    verdict = combine_orientation([stub, ballot(SONNET, True, "higher_more_construct")],
                                  "late_adaptation", "hand deviation")
    assert verdict.higher_is_better is True and verdict.agreed is False
    assert "orientation_single_witness" in outcome_flags(verdict)
    assert "not_run" in verdict.notes


def test_a_reader_that_could_not_tell_never_becomes_a_reply_that_did_not_happen():
    """Fix round F15's ruling, where it would have bitten: `not quotes` was NOT added to the
    degeneracy signatures. An honest "I cannot tell" is an ABSTENTION, and if the detector called
    it a non-reply, row 5 would open and the other reader would set the direction alone."""
    verdict = combine_orientation(
        [ballot(OPUS, True, "higher_more_construct"),
         ballot(SONNET, None, "unknown",
                reason="N/A. Not enough information was provided here.")],
        "aftereffect", "aftereffect (deg)")
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert "orientation_single_witness" not in outcome_flags(verdict)


# ---------------------------------------- C2: the tiebreak ballot, with the raw means in view
def test_the_tiebreak_ballot_has_a_prompt_and_a_version_of_its_own():
    """It asks a different question from `orientation.md` — the two readers' ballots and this
    cell's raw means are in front of it — so it is a different prompt, recorded as one. Putting
    it in `PROMPT_FILES` would move the ordinary readers' `prompt_version` too, and every
    recorded orientation fixture is keyed on that."""
    from canopy.agents import orientation as module

    assert "orientation_tiebreak" not in module.PROMPT_FILES
    assert module.TIEBREAK_PROMPT_VERSION.startswith("orientation_tiebreak/1@")
    assert module.TIEBREAK_PROMPT_VERSION != module.PROMPT_VERSION


def test_the_tiebreak_prompt_never_mentions_what_the_answer_would_do_to_the_pool():
    """A reader told which answer keeps a row in the analysis is being asked a different
    question. It sees the measure, the means and the ballots, and nothing about the consequence."""
    text = load_prompt("orientation_tiebreak").lower()
    for forbidden in ("pool", "meta-analys", "forest", "effect size", "significan"):
        assert forbidden not in text, forbidden


def test_the_ballot_is_shown_the_raw_means_and_both_prior_readers(paper, bock_dataset):
    from canopy.agents.orientation import TIEBREAK_PROMPT_VERSION, tiebreak_ballot

    provider = FakeProvider([orientation_payload(higher_is_better="lower")])
    client = LLMClient(provider=provider, cache_dir=None)
    run = tiebreak_ballot(
        client, paper, bock_dataset, bock_dataset.outcomes[0], protocol=None, outcome=SCREENING,
        pdf_file_id=None, mean_a=44.6, mean_b=30.2, group_a_label="old subjects",
        group_b_label="young subjects", n_a=12, n_b=12, unit="s",
        prior_runs=[ballot(OPUS, None, "signed_direction", "a_greater"),
                    ballot(SONNET, True, "higher_more_construct", "unknown")])

    prompt = provider.requests[0].messages[0]["content"][-1]["text"]
    assert "{{" not in prompt
    assert "screening test completion time" in prompt            # the measure
    assert "Screening test completion time" in prompt            # the outcome definition
    assert "44.6" in prompt and "30.2" in prompt                 # both raw means
    assert "old subjects" in prompt and "young subjects" in prompt
    assert OPUS in prompt and SONNET in prompt                   # both prior readers, by name
    assert "a_greater" in prompt and "higher_more_construct" in prompt
    assert REASON in prompt                                       # their reasons, verbatim
    assert run.model == TIEBREAK_MODEL and run.prompt_version == TIEBREAK_PROMPT_VERSION
    assert provider.requests[0].effort == TIEBREAK_EFFORT
    assert run.higher_is_better is False


def test_the_verdict_records_how_its_direction_was_settled():
    """`orientation_source` — the row carries it, and "two readers agreed" is a different claim
    from "a bought third read outvoted one of them"."""
    agreed = combine_orientation([ballot(OPUS, False, "higher_more_error"),
                                  ballot(SONNET, False, "higher_more_error")], "x", "m")
    assert (agreed.orientation_source, agreed.agreed) == ("agreed", True)

    majority = combine_orientation([ballot(OPUS, False, "signed_direction"),
                                    ballot(SONNET, True, "signed_direction"),
                                    ballot(TIEBREAK_MODEL, False, "signed_direction")],
                                   "x", "m", third_read=True)
    assert majority.orientation_source == "tiebreak_ballot" and majority.needs_human is False

    stub = ballot(SONNET, True, "higher_more_construct").model_copy(update={"not_run": True})
    single = combine_orientation([ballot(OPUS, True, "higher_more_construct"), stub], "x", "m")
    assert single.orientation_source == "single_witness"

    unsettled = combine_orientation([ballot(OPUS, None), ballot(SONNET, None)], "x", "m")
    assert unsettled.orientation_source == "", "an open question was settled by nobody"


def test_a_tiebreak_ballot_never_settles_a_direction_on_its_own():
    """Two abstentions and one answer is one ballot, whoever paid for it."""
    verdict = combine_orientation([ballot(OPUS, None, "signed_direction"),
                                   ballot(SONNET, None, "signed_direction"),
                                   ballot(TIEBREAK_MODEL, False, "signed_direction")],
                                  "x", "m", third_read=True)
    assert verdict.higher_is_better is None and verdict.needs_human is True
    assert verdict.orientation_source == ""


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
