"""The ceiling items that live between two areas — the wiring, not the rules.

Each area shipped a rule and a pure function to enforce it; every test here is about the ONE
question those tests could not ask: does the pipeline actually call it, with the right inputs, at
the only moment the inputs exist?

* **C3 row 1** (`_recheck_orientation`) — the discard filter compares a reader's stated direction
  against the RESOLVED raw means, and orientation is decided before any cell is resolved. Until
  this wiring the filter was inert on every real run.
* **C6/C7** (`_verify`) — a blocked cell buys no reader, and the two orientation reads are bought
  per measure rather than per candidate, so the extraction gate alone never stopped them.
* **C7** (`_run_paper`) — a dataset the map adjudicator rejected reaches the exclusions table.
* **C8** (`_verify_cell`) — a verifier that could not be run is recorded as `not_run`.
* **C9** (`resolve._finish`) — the conversion gate's own flags are priced on the ROW.
* **C11** (`resolve_cell`) — the margin is a field, not only a sentence.
* **C4/C6/C7** (`_extract`) — the reviewer's answers to the map's open questions are applied
  before the extraction gate reads them.

The orientation ballots are the real ones (`tests/fixtures/runs/orientation_ballots.json`); the
four cells are the real numbers, and every "must never print this" assertion is written against
the arithmetic rather than against a paraphrase.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from canopy.agents.orientation import combine_orientation
from canopy.config import MODELS
from canopy.models import (Candidate, CheckFlag, DatasetSpec, DispersionType, GroupSpec,
                           MapQuestion, OrientationRun, OutcomeSources, StudyMap, VerifierVerdict)
from canopy.pipeline.run import _recheck_orientation
from canopy.stats.effect_sizes import smd_from_means
from canopy.verify.vote import VoteResult

OPUS, SONNET = MODELS["primary"], MODELS["secondary"]
BALLOTS = (Path(__file__).resolve().parent / "fixtures" / "runs" / "orientation_ballots.json")

#: the four real cells, as `smd_from_means` inputs
BOCK_D1 = (-22.4, 8.3, 12, -27.0, 5.5, 12)          # aftereffect: published -0.537
BOCK_D2 = (52.4905, 9.15, 12, 27.9830, 7.3, 12)     # tracking RMSE: two agreeing readers


def d_of(values, higher_is_better) -> float:
    return round(smd_from_means(*values, higher_is_better=higher_is_better).d, 4)


def a_real_reason(index: int = 0) -> str:
    """A justification from the run itself, so no fixture accidentally tests C12's detector."""
    ballots = json.loads(BALLOTS.read_text())["ballots"]
    clean = [b["reason"] for b in ballots if len(b["reason"]) >= 403]
    return clean[index % len(clean)]


def ballot(model=OPUS, higher_is_better=None, raw_value_semantics="unknown",
           direction_stated_in_text="unknown", index: int = 0) -> OrientationRun:
    return OrientationRun(higher_is_better=higher_is_better,
                          raw_value_semantics=raw_value_semantics,
                          direction_stated_in_text=direction_stated_in_text,
                          reason=a_real_reason(index), model=model, quotes=["a quote"])


def votes_of(mean_a: float | None, mean_b: float | None) -> dict[str, VoteResult]:
    out = {}
    if mean_a is not None:
        out["A"] = VoteResult(group="A", agreement="agree", mean=mean_a, method="median")
    if mean_b is not None:
        out["B"] = VoteResult(group="B", agreement="agree", mean=mean_b, method="median")
    return out


def a_dataset(dataset_id: str) -> DatasetSpec:
    return DatasetSpec(dataset_id=dataset_id, cluster_id="b511dbb76fa6",
                       group_a=GroupSpec(label="old", n=12, n_evidence="twelve old subjects"),
                       group_b=GroupSpec(label="young", n=12, n_evidence="twelve young"))


def _means_check(verdict) -> str:
    """The state of the deterministic check as the VERDICT records it: `ran`, `no_means` or
    `disputed`. Every verdict carries it, because "checked, and nothing was wrong" and "could not
    be checked" are not the same claim about a direction."""
    from canopy.agents.orientation import MEANS_CHECK_NOTE

    marker = f"{MEANS_CHECK_NOTE}: "
    assert marker in verdict.notes, verdict.notes
    return verdict.notes.split(marker)[1].split(" ")[0].strip()


def _discard_note(verdict) -> str:
    """Just the sentence C3 row 1 wrote — the notes are `"; "`-joined and the tail names every
    reader's `raw_value_semantics`, which would make "who was thrown out" unfalsifiable."""
    marker = "[orientation_reader_contradicts_values]"
    assert marker in verdict.notes, verdict.notes
    return verdict.notes.split(marker)[1].split(";")[0]


def verdict_for(dataset_id: str, runs, **kwargs):
    return combine_orientation(runs, "aftereffect", "aftereffect (deg)",
                               dataset_id=dataset_id, **kwargs)


# ===================================================== C3 row 1, at the only place it can run
def test_the_discard_filter_is_inert_until_the_means_exist():
    """The premise of the whole item. Orientation is decided per (outcome, measure), before any
    cell of that measure is resolved, so at the moment the readers are asked there is nothing to
    check them against and `combine_orientation` cannot discard anybody."""
    runs = [ballot(OPUS, False, "signed_direction", "b_greater"),
            ballot(SONNET, True, "higher_more_construct", "unknown", index=1)]
    at_orientation_time = verdict_for("d1", runs)              # no means: what run.py had
    assert "orientation_reader_contradicts_values" not in at_orientation_time.notes
    assert at_orientation_time.higher_is_better is None        # readers disagree -> abstain


def test_a_reader_contradicted_by_the_resolved_means_is_discarded_at_verify_time():
    """C3 row 1, and ADVERSARIAL round 2's amendment to it, through the pipeline path.

    opus states `b_greater`; the resolved means say A = -22.4 is the greater of the two, so opus
    is discarded. The ballot discarded is the one holding the RIGHT answer (`hib=False` gives the
    published sign) — and row 5 must still not be reachable, because letting the silent reader
    decide would prefer silence to evidence every time. The cell abstains.
    """
    runs = [ballot(OPUS, False, "signed_direction", "b_greater"),
            ballot(SONNET, True, "higher_more_construct", "unknown", index=1)]
    before = verdict_for("d1", runs)
    after = _recheck_orientation(a_dataset("d1"), before, votes_of(-22.4, -27.0), [])

    assert after is not before                                 # the wiring did something
    assert after.higher_is_better is None and after.needs_human is True
    assert "orientation_reader_contradicts_values" in after.notes
    assert "orientation_single_witness" not in after.notes
    assert d_of(BOCK_D1, True) == 0.6534                       # what row 5 would have pooled
    assert d_of(BOCK_D1, False) == -0.6534                     # what the DISCARDED reader was right about


def test_two_readers_that_agree_are_not_touched_by_the_recheck():
    """Bock d2, tracking RMSE: both readers `higher_more_error`/`hib=False`, neither stating a
    direction. Nothing is checkable, so nothing is discarded and the cell keeps -2.9610. The
    recheck must be a no-op on every cell it has no evidence about."""
    runs = [ballot(OPUS, False, "higher_more_error", "unknown"),
            ballot(SONNET, False, "higher_more_error", "unknown", index=1)]
    before = verdict_for("d2", runs)
    assert before.higher_is_better is False and before.agreed is True
    after = _recheck_orientation(a_dataset("d2"), before, votes_of(52.4905, 27.9830), [])
    # the DECISION is untouched…
    assert (after.higher_is_better, after.needs_human, after.agreed) == (False, False, True)
    assert "orientation_reader_contradicts_values" not in after.notes
    # …and the RECORD is updated: the check ran here, where at orientation time it could not
    assert _means_check(before) == "no_means" and _means_check(after) == "ran"
    assert d_of(BOCK_D2, False) == -2.9610
    assert d_of(BOCK_D2, True) == 2.9610                       # the sign it must never acquire


def test_the_check_never_runs_against_another_datasets_means():
    """The verdict is a property of the MEASURE and is reused for every dataset carrying it. A
    reader checked against a different dataset's means is checked against a different comparison,
    and the reader thrown out would be whoever read THIS paper correctly."""
    runs = [ballot(OPUS, False, "signed_direction", "b_greater"),
            ballot(SONNET, True, "higher_more_construct", "unknown", index=1)]
    asked_about_d1 = verdict_for("d1", runs)
    # d2's means (A the greater) would contradict opus exactly as d1's do — and must not be used
    unchanged = _recheck_orientation(a_dataset("d2"), asked_about_d1, votes_of(52.4905, 27.9830),
                                     [])
    assert unchanged is asked_about_d1
    assert "orientation_reader_contradicts_values" not in unchanged.notes


def test_no_discard_happens_while_which_series_is_which_is_disputed():
    """ADVERSARIAL round 2, change 2. Under `series_marker_mismatch` the "resolved" means may be
    the other group's, so a discard run against them throws out the reader that was right. The
    open flags reach the filter from the cell, which is the whole reason this call site is after
    `run_checks` rather than before it."""
    runs = [ballot(OPUS, False, "signed_direction", "b_greater"),
            ballot(SONNET, True, "higher_more_construct", "unknown", index=1)]
    before = verdict_for("d1", runs)
    flags = [CheckFlag(code="series_marker_mismatch", severity="warn", message="")]
    after = _recheck_orientation(a_dataset("d1"), before, votes_of(-22.4, -27.0), flags)
    assert (after.higher_is_better, after.needs_human) == (before.higher_is_better,
                                                           before.needs_human)
    assert "orientation_reader_contradicts_values" not in after.notes
    # and the verdict says WHY nobody was checked, naming the flag that suspended it
    assert _means_check(after) == "disputed"
    assert "series_marker_mismatch" in after.notes


def test_a_measure_settled_by_a_bought_third_read_is_recombined_the_same_way():
    """`third_read` travels on the verdict. Without it the recheck would recombine three ballots
    under the two-reader rule and silently turn a settled measure back into a question — a paid
    read thrown away by the free check that runs after it."""
    runs = [ballot(OPUS, True, "higher_more_construct", "unknown"),
            ballot(SONNET, False, "higher_more_construct", "unknown", index=1),
            ballot("claude-fable-5", True, "higher_more_construct", "unknown", index=2)]
    settled = verdict_for("d1", runs, third_read=True)
    assert settled.higher_is_better is True and settled.third_read is True
    after = _recheck_orientation(a_dataset("d1"), settled, votes_of(-22.4, -27.0), [])
    assert after.higher_is_better is True and after.third_read is True
    assert "orientation_by_majority" in after.notes and _means_check(after) == "ran"


def test_the_recheck_can_only_take_a_witness_away_never_add_one():
    """The safety property the ruling rests on: passing means to a verdict that already agreed
    can never change it into a different agreement, only into a question. Checked over every
    combination of reader answers the two-ballot case can take."""
    for hib_a in (True, False, None):
        for hib_b in (True, False, None):
            for stated in ("unknown", "a_greater", "b_greater"):
                runs = [ballot(OPUS, hib_a, "higher_more_construct", stated),
                        ballot(SONNET, hib_b, "higher_more_construct", "unknown", index=1)]
                before = verdict_for("d1", runs)
                after = _recheck_orientation(a_dataset("d1"), before, votes_of(-22.4, -27.0), [])
                if before.higher_is_better is not None:
                    assert after.higher_is_better in (before.higher_is_better, None), (
                        hib_a, hib_b, stated)
                else:
                    assert after.higher_is_better is None, (hib_a, hib_b, stated)


# ============================================================ C9 on the row (resolve._finish)
def resolved_values(flags):
    """A cell whose paper printed a main effect averaged over the eight target directions — the
    outcome's own window — and printed no degrees of freedom with it.

    C5's aggregation gate PASSES here (the statistic averages over exactly what the outcome
    averages over), which is what makes this the case C9 is about: everything upstream is
    satisfied and the only thing missing is any evidence that the number is the comparison of
    THESE two groups. On the `within_factors=[]` path C5 refuses first, so this is the shape in
    which C9's cap is the rule that binds.
    """
    from canopy.pipeline.resolve import ResolvedValues, StatisticValues

    return ResolvedValues(
        dataset_id="d1", outcome_key="late_adaptation", higher_is_better=False,
        confidence="auto_accept", flags=list(flags),
        test_statistic=StatisticValues(
            stat_type="t", value=5.25, df=None if "df_missing" in flags else 22,
            p_kind="unknown", design="independent_t", direction="a_greater",
            contrast_kind="groups", within_factors=["target direction (8 levels)"],
            outcome_averages_over=["target direction"]))


def test_a_converted_row_carrying_df_missing_is_held_at_the_row_level():
    """C9's v1 defect, at the line it lives on. The gate's flags are raised while the effect size
    is being built — after `ResolvedValues.from_verdicts` has already set `confidence` from the
    two cells — so before this call a `df_missing` row carried the flag and pooled anyway.

    A post-hoc t from a three-group ANOVA prints exactly "t = 5.25" with no df, and nothing in the
    number says which two groups it compares.
    """
    from canopy.models import OutcomeDef, StatsSettings
    from canopy.pipeline.resolve import resolve_effect

    outcome = OutcomeDef(key="late_adaptation", label="late adaptation",
                         definition="the mean direction error over the last adaptation block")
    settings = StatsSettings(route_precedence=["test_statistic"])
    record = resolve_effect(a_dataset("d1"), outcome, resolved_values(["df_missing"]), settings)

    assert record.route == "test_statistic" and record.d is not None   # it converted
    assert record.confidence == "needs_human"                          # and is held anyway
    assert any("df_missing" in step for step in record.conversion_steps)


def test_the_same_row_without_the_flag_keeps_the_bucket_its_cells_earned():
    """The negative control: the cap is the flag's doing, not the route's."""
    from canopy.models import OutcomeDef, StatsSettings
    from canopy.pipeline.resolve import resolve_effect

    outcome = OutcomeDef(key="late_adaptation", label="late adaptation",
                         definition="the mean direction error over the last adaptation block")
    settings = StatsSettings(route_precedence=["test_statistic"])
    record = resolve_effect(a_dataset("d1"), outcome, resolved_values([]), settings)
    assert record.route == "test_statistic" and record.confidence == "auto_accept"


# ================================================================== C11 as a field (C6 of the brief)
def test_every_verdict_the_score_decided_publishes_its_distance_from_the_line():
    """Amended under review M3: the field is published on every cell whose BUCKET THE SCORE
    DECIDED. A cell held by an error or an unresolved direction cleared no boundary, and a
    distance printed for it reads as a claim about a decision the score never made."""
    from canopy.models import OrientationVerdict
    from canopy.verify.confidence import ACCEPT_WITH_NOTE, resolve_cell

    cand = Candidate(candidate_id="c1", dataset_id="d1", outcome_key="late_adaptation", group="A",
                     kind="group_stats", status="found", extractor_id="text:narrative:opus",
                     source_kind="text_mean_sd", mean=31.1, dispersion_value=6.0,
                     dispersion_type=DispersionType.SD, n=9, quote="31.1 +/- 6.0",
                     page=3, locator="Results")
    oriented = OrientationVerdict(outcome_key="late_adaptation", higher_is_better=True,
                                  agreed=True, needs_human=False)
    verdict = resolve_cell(a_dataset("d1"), "late_adaptation", "A", [cand], orientation=oriented)
    assert verdict.confidence_score is not None
    assert verdict.confidence_margin is not None and verdict.nearest_boundary
    assert verdict.confidence_margin == pytest.approx(
        min(abs(verdict.confidence_score - b)
            for b in (ACCEPT_WITH_NOTE, 0.70, 0.75)), abs=1e-4)
    # and the sentence and the field agree, so neither can drift from the other
    assert f"{verdict.confidence_margin:.4f}" in " ".join(verdict.confidence_reasons)


def test_the_review_queue_row_carries_the_margin_a_reviewer_would_otherwise_recompute():
    from canopy.pipeline.state import review_entry
    from canopy.verify.confidence import resolve_cell

    cand = Candidate(candidate_id="c1", dataset_id="d1", outcome_key="late_adaptation", group="A",
                     kind="group_stats", status="found", extractor_id="text:narrative:opus",
                     source_kind="text_mean_sd", mean=31.1, dispersion_value=6.0,
                     dispersion_type=DispersionType.SD, n=9, quote="31.1 +/- 6.0")
    verdict = resolve_cell(a_dataset("d1"), "late_adaptation", "A", [cand])
    row = review_entry(verdict, paper_id="b511dbb76fa6")
    assert row["confidence_margin"] == verdict.confidence_margin
    assert row["nearest_boundary"] == verdict.nearest_boundary
    assert row["confidence_score"] == verdict.confidence_score


def test_the_margin_reaches_the_file_a_reviewer_actually_opens(tmp_path):
    """Review M1: the three C11 keys were on the dict and dropped in silence by the writer's
    explicit column list (`csv.DictWriter(..., extrasaction="ignore")`), so the brief's "surface
    both in the review CSV" was undelivered while its test — this one's predecessor, three layers
    above the file — passed. Read the header the reviewer reads.
    """
    from canopy.pipeline.run import REVIEW_QUEUE_COLUMNS
    from canopy.pipeline.state import review_entry
    from canopy.report.tables import write_rows
    from canopy.verify.confidence import resolve_cell

    cand = Candidate(candidate_id="c1", dataset_id="d1", outcome_key="late_adaptation", group="A",
                     kind="group_stats", status="found", extractor_id="text:narrative:opus",
                     source_kind="text_mean_sd", mean=31.1, dispersion_value=6.0,
                     dispersion_type=DispersionType.SD, n=9, quote="31.1 +/- 6.0")
    from canopy.models import OrientationVerdict

    oriented = OrientationVerdict(outcome_key="late_adaptation", higher_is_better=True,
                                  agreed=True, needs_human=False)
    scored = review_entry(resolve_cell(a_dataset("d1"), "late_adaptation", "A", [cand],
                                       orientation=oriented), paper_id="b511dbb76fa6")
    held = review_entry(resolve_cell(a_dataset("d1"), "late_adaptation", "A", [cand]),
                        paper_id="b511dbb76fa6")            # no direction: the score decided none
    written = write_rows([scored, held], tmp_path / "human_review_queue", REVIEW_QUEUE_COLUMNS,
                         formats=("csv",))
    with open(written["csv"], newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header, rows = list(reader.fieldnames or []), list(reader)
    for column in ("confidence_score", "confidence_margin", "nearest_boundary"):
        assert column in header, header
    assert scored["confidence_margin"] is not None
    assert rows[0]["confidence_margin"] == str(scored["confidence_margin"])
    assert rows[0]["nearest_boundary"] == scored["nearest_boundary"]
    # …and M3's half in the same file: a cell the score did not decide claims no distance
    assert rows[1]["confidence_margin"] == "" and rows[1]["nearest_boundary"] == ""


# ========================================================= C8: a failure is an absence (ruling 2)
def test_a_verifier_that_could_not_be_run_is_recorded_as_not_run_not_as_a_doubt():
    """The two states are on the model now, and they are priced apart: `ambiguous` costs the cell
    0.05, `not_run` costs it nothing, so a cell whose verifier CRASHED no longer scores below one
    that was never verified at all."""
    from canopy.verify.confidence import NOT_RUN, confidence, verifier_state

    failed = VerifierVerdict(candidate_id="c1", verdict="not_run", reason="transport error")
    assert verifier_state(failed) == NOT_RUN

    vote = VoteResult(group="A", agreement="single", mean=31.1, method="single")
    _, with_failure, reasons = confidence(vote, [failed])
    _, never_scheduled, _ = confidence(vote, [])
    assert with_failure == never_scheduled
    assert any("could not be run" in r and "unverified rather than doubted" in r
               for r in reasons)


# ================================================ the two cross-area interfaces, at their call sites
def test_the_extract_stage_applies_the_reviewers_map_answers_before_the_gate_reads_them():
    """Ruling 3. `apply_map_answers` is the mapper's; `map_answers` is the review log's. This is
    the call site that joins them, and the order is the point: an answer that arrives after the
    gate has read the map unblocks nothing."""
    import inspect

    from canopy.agents import mapper
    from canopy.pipeline import overrides
    from canopy.pipeline import run as run_module

    for stage in (run_module._extract, run_module._verify):
        source = inspect.getsource(stage)
        answered = source.index("_answered_map(")
        gate = source.index("extraction_blocks(study)")
        assert answered < gate, "the answers must be applied before the gate reads the map"
        # …and before any `return`, which is the half a string-index comparison is blind to and
        # exactly the defect it missed: `--resume` returned the cached candidates at the top of
        # the stage, so the answered map below it was never read on the only path it exists for
        # (review L7 / H2). Comparing against the FIRST return is what binds that.
        assert answered < source.index("return"), (
            f"{stage.__name__} can return before it reads the reviewer's answers, so on --resume "
            f"the answers are never applied")

    # both halves are imported directly and neither is caught: an unreadable review log must stop
    # the run rather than read as "nobody answered anything", which blocks the same cells and
    # means the opposite thing
    helper = (inspect.getsource(run_module._map_answers)
              + inspect.getsource(run_module._answered_map))
    assert "apply_map_answers(study, " in helper
    assert "map_answers(ctx.out_dir, paper.sha256)" in helper
    assert "try:" not in helper and "except" not in helper
    assert run_module.map_answers is overrides.map_answers
    assert run_module.apply_map_answers is mapper.apply_map_answers


PAPER_ID = "b511dbb76fa6" + "0" * 52


def _study_with_an_open_inclusion_question() -> StudyMap:
    return StudyMap(
        paper_id=PAPER_ID,
        datasets=[DatasetSpec(dataset_id="d1", group_a=GroupSpec(label="old", n=12),
                              group_b=GroupSpec(label="young", n=12),
                              outcomes=[OutcomeSources(outcome_key="late_adaptation",
                                                       measure_name="mean direction error")])],
        open_questions=[MapQuestion(kind="include_dataset", dataset_id="d1",
                                    question="only one mapping agent proposed this dataset",
                                    options=["include", "exclude"])])


def test_an_answer_to_an_open_question_unblocks_the_dataset_it_names():
    """End to end over the two functions, with no pipeline: the gate blocks the dataset while the
    question is open and stops blocking it once the reviewer has answered."""
    from canopy.agents.mapper import apply_map_answers, extraction_blocks

    study = _study_with_an_open_inclusion_question()
    blocked_datasets, _ = extraction_blocks(study)
    assert "d1" in blocked_datasets

    answered = apply_map_answers(study, [
        {"kind": "include_dataset", "paper_id": PAPER_ID, "dataset_id": "d1",
         "decision": "include", "note": "the control sample the protocol asks for"}])
    assert extraction_blocks(answered) == ({}, {})
    assert study.open_questions, "the answer must not mutate the map it was applied to"


def test_an_unanswered_question_still_blocks_the_verify_stage_from_buying_a_direction():
    """C6/C7 at the stage the extraction gate never reached. A blocked cell produces no candidate,
    so no verifier and no digitiser call follows it — but orientation is bought per (outcome,
    measure) rather than per candidate, so two reads were spent on a dataset nobody had agreed to
    include. Bock's tracking-only control sample was read, digitised, verified and signed at
    d = -2.9610 while the objection to including it sat in a warning."""
    import inspect

    from canopy.pipeline import run as run_module

    source = inspect.getsource(run_module._verify)
    gate = source.index("extraction_blocks(study)")
    bought = source.index("orientation_verdict(")
    assert gate < bought, "the gate must be read before any direction is bought"
    # …and the gate is read before the stage can return at all: a `--resume` that returns above
    # it has not read it, which is the blind spot a string-index comparison has and the shape H2
    # turned out to be in the extract stage (review L7)
    assert gate < source.index("return"), "the verify stage can return before it reads the gate"
    assert "if dataset.dataset_id in blocked_datasets:" in source
    assert "in blocked_cells:" in source


# ================================ C7: a dataset the map rejected reaches the exclusions table
def test_a_dataset_the_map_adjudicator_rejected_is_recorded_in_the_exclusions_table(
        tmp_path, monkeypatch):
    """C7 says "Rejected -> recorded in `exclusions`", and until this wiring it was not.

    A dataset with `included=False` is never extracted, so it reaches no resolver, no row and no
    review queue — and it reached no table either, which is the same silence an ineligible paper
    used to leave. The rule it was rejected under and the paper's own words it rests on are what
    make the decision auditable, so both travel with the row.

    The map is stubbed rather than coaxed out of the fake, because what is under test is the
    orchestrator's bookkeeping, not the adjudicator that produced the ruling.
    """
    from canopy.pipeline import run as run_module
    from canopy.pipeline.run import run_pipeline
    from tests.test_pipeline_offline import PDFS, PROTOCOL, FakeSpec, fake_router

    from canopy.ingest.pdf import ingest_pdf
    from canopy.llm.client import LLMClient
    from canopy.llm.providers import FakeProvider

    work = tmp_path / "ingested"
    paper = ingest_pdf(PDFS[0], work / PDFS[0].stem)
    spec = FakeSpec(paper, 44.6, 30.2)
    client = LLMClient(provider=FakeProvider([fake_router([spec])]), allow_live=True,
                       cache_dir=None)
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    import shutil

    shutil.copyfile(PDFS[0], papers_dir / PDFS[0].name)

    real_map = run_module._map

    from canopy.agents.mapper import HUMAN_EXCLUSION_RULE

    def map_with_a_rejected_dataset(ctx, paper_record, group, status):
        study, file_id = real_map(ctx, paper_record, group, status)
        for suffix, rule, quote in (
                ("x", "the protocol's population rule: adults over 60",
                 "twelve young subjects aged 19-27 took part"),
                # what an ANSWERED map looks like when a person excluded the dataset and named no
                # protocol rule — `apply_map_answers` writes exactly this
                ("h", HUMAN_EXCLUSION_RULE, "")):
            rejected = study.datasets[0].model_copy(deep=True)
            rejected.dataset_id = f"{study.datasets[0].dataset_id}{suffix}"
            rejected.included = False
            rejected.exclusion_rule = rule
            rejected.exclusion_quote = quote
            study.datasets.append(rejected)
        return study, file_id

    monkeypatch.setattr(run_module, "_map", map_with_a_rejected_dataset)
    out = tmp_path / "run"
    run_pipeline(papers_dir, PROTOCOL, out, client=client, concurrency=1)

    rows = list(csv.DictReader((out / "exclusions.csv").open(newline="", encoding="utf-8")))
    mine = {r["dataset_id"][-1]: r
            for r in rows if r["reason_as_given"] == "map_adjudication:dataset_rule"}
    assert set(mine) == {"x", "h"}, rows

    by_model = mine["x"]
    assert by_model["decider"] == "map-adjudicator"
    assert by_model["detail"] == "the protocol's population rule: adults over 60"
    assert by_model["quote"] == "twelve young subjects aged 19-27 took part"
    assert by_model["stage"] == "map"
    assert by_model["reason"] == "map_adjudication"      # a named reason, never "other"

    # the person's own judgement is attributed to the person: putting it on the adjudicator would
    # claim a protocol rule that no model applied
    from canopy.agents.mapper import HUMAN_DECIDER

    # this map carries no review log behind it, so the placeholder rule is the only marker there
    # is — and it is a reliable one, because `apply_map_answers` is the only thing that writes it
    assert mine["h"]["decider"] == HUMAN_DECIDER      # the mapper's spelling, not a second one
    assert mine["h"]["detail"] == HUMAN_EXCLUSION_RULE

    # PRISMA still has to add up with the rejected dataset counted among the excluded ones
    prisma = json.loads((out / "prisma.json").read_text())
    counts = prisma["counts"] if "counts" in prisma else prisma
    assert counts["datasets"] - counts["datasets_excluded"] == counts["included_datasets"]
    assert counts["datasets_excluded"] >= 2


# ============================== the two stages, run for real against the pipeline's own fake
def _offline(tmp_path, router_wrapper=None, map_wrapper=None, monkeypatch=None, resume=True,
             **spec_kwargs):
    """One offline run of the Bock fixture paper, with optional hooks on the router and the map."""
    import shutil

    from canopy.ingest.pdf import ingest_pdf
    from canopy.llm.client import LLMClient
    from canopy.llm.providers import FakeProvider
    from canopy.pipeline import run as run_module
    from canopy.pipeline.run import run_pipeline
    from tests.test_pipeline_offline import PDFS, PROTOCOL, FakeSpec, fake_router

    paper = ingest_pdf(PDFS[0], tmp_path / "ingested" / PDFS[0].stem)
    router = fake_router([FakeSpec(paper, 44.6, 30.2, **spec_kwargs)])
    seen: list[Any] = []

    def recording(request):
        seen.append(request)
        return (router_wrapper(request, router) if router_wrapper else router(request))

    client = LLMClient(provider=FakeProvider([recording]), allow_live=True, cache_dir=None)
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir(exist_ok=True)
    shutil.copyfile(PDFS[0], papers_dir / PDFS[0].name)
    if map_wrapper is not None:
        real_map = run_module._map
        monkeypatch.setattr(run_module, "_map",
                            lambda ctx, p, g, s: map_wrapper(real_map(ctx, p, g, s)))
    out = tmp_path / "run"
    run_pipeline(papers_dir, PROTOCOL, out, client=client, concurrency=1, resume=resume)
    return out, seen


def _orientation_calls(requests) -> list[Any]:
    return [r for r in requests
            if "raw_value_semantics" in ((r.schema or {}).get("properties") or {})]


def _records(out: Path) -> list[dict]:
    paper_dir = next(p for p in (out / "papers").iterdir() if (p / "resolve.json").exists())
    return json.loads((paper_dir / "resolve.json").read_text())["records"]


def _extract_payload(out: Path) -> dict:
    paper_dir = next(p for p in (out / "papers").iterdir() if (p / "extract.json").exists())
    return json.loads((paper_dir / "extract.json").read_text())


def _verify_payload(out: Path) -> dict:
    paper_dir = next(p for p in (out / "papers").iterdir() if (p / "verify.json").exists())
    return json.loads((paper_dir / "verify.json").read_text())


def test_a_blocked_dataset_buys_no_direction_and_no_verifier(tmp_path, monkeypatch):
    """C6/C7 (m2). The extraction gate already stops a blocked cell being READ, but orientation is
    bought once per (outcome, measure) rather than once per candidate, so two reads per measure
    were still spent on a dataset nobody had agreed to include."""
    def add_a_blocked_dataset(mapped):
        study, file_id = mapped
        blocked = study.datasets[0].model_copy(deep=True)
        blocked.dataset_id = f"{study.datasets[0].dataset_id}q"
        for outcome in blocked.outcomes:                 # its own measure, so its own two reads
            outcome.measure_name = "peak velocity of the corrective movement"
        study.datasets.append(blocked)
        study.open_questions.append(MapQuestion(
            kind="include_dataset", dataset_id=blocked.dataset_id,
            question="only one mapping agent proposed this dataset"))
        return study, file_id

    out, requests = _offline(tmp_path, map_wrapper=add_a_blocked_dataset, monkeypatch=monkeypatch)
    payload = _verify_payload(out)

    measures = {key.split("|", 1)[1] for key in payload["orientation"]}
    assert "peak velocity of the corrective movement" not in measures
    assert not [v for v in payload["verdicts"] if v["dataset_id"].endswith("q")]
    for request in _orientation_calls(requests):
        assert "peak velocity of the corrective movement" not in json.dumps(request.messages)


def test_the_direction_of_a_measure_is_rechecked_against_the_cells_own_means(tmp_path):
    """C3 row 1 end to end (o1). The readers state `b_greater`; the resolved means say group A is
    the greater of the two. Every reader that made that claim is discarded, nobody is left to
    decide, and the measure becomes a question instead of a signed effect size."""
    def contradict_the_means(request, router):
        answer = router(request)
        if isinstance(answer, dict) and "raw_value_semantics" in answer:
            return {**answer, "direction_stated_in_text": "b_greater"}
        return answer

    out, _ = _offline(tmp_path, router_wrapper=contradict_the_means)
    payload = _verify_payload(out)
    verdict = next(iter(payload["orientation"].values()))

    assert "orientation_reader_contradicts_values" in verdict["notes"]
    assert verdict["higher_is_better"] is None and verdict["needs_human"] is True
    assert all(v["higher_is_better"] is None for v in payload["verdicts"])

    rows = _records(out)
    assert rows and all(r["route"] == "not_convertible" for r in rows)
    assert all("orientation_unresolved" in r["flags"] for r in rows)


def test_the_same_run_signs_its_rows_when_the_readers_agree_with_the_means(tmp_path):
    """The control for the test above, on the same paper and the same numbers: with the readers'
    stated direction matching the means nothing is discarded and the run pools as before. Without
    this, "the recheck holds every cell" would look like the rule working."""
    out, _ = _offline(tmp_path)
    payload = _verify_payload(out)
    verdict = next(iter(payload["orientation"].values()))
    assert "orientation_reader_contradicts_values" not in verdict["notes"]
    assert verdict["higher_is_better"] is False and verdict["needs_human"] is False
    rows = _records(out)
    assert any(r["route"] != "not_convertible" and r["d"] is not None for r in rows)


# ================== the four guarantees the controller's fix-round ruling names for this call
def test_a_group_that_did_not_resolve_gives_the_filter_nothing_rather_than_a_default():
    """`None`, never a default and never the other group's number. An absent mean disables the
    filter, which is the safe direction: its only power is to remove a witness."""
    runs = [ballot(OPUS, False, "signed_direction", "b_greater"),
            ballot(SONNET, True, "higher_more_construct", "unknown", index=1)]
    before = verdict_for("d1", runs)
    # group B's readers disagreed, so the vote resolved no mean for it
    half = {"A": VoteResult(group="A", agreement="agree", mean=-22.4, method="median"),
            "B": VoteResult(group="B", agreement="disagree", mean=None, method="median")}
    for votes in (half, votes_of(-22.4, None), {}):
        after = _recheck_orientation(a_dataset("d1"), before, votes, [])
        assert "orientation_reader_contradicts_values" not in after.notes   # nobody was discarded
        assert (after.higher_is_better, after.needs_human) == (before.higher_is_better,
                                                               before.needs_human)
        assert _means_check(after) == "no_means"        # and the record says so, truthfully


def test_the_means_handed_to_the_filter_are_raw_and_in_the_datasets_own_a_b_assignment():
    """Nothing orientates the means before the comparison. `higher_is_better` is applied in
    `resolve_effect`, far later; a sign applied here would be the comparison arguing with itself,
    and the reader discarded would flip with the answer under test."""
    import inspect

    from canopy.pipeline import run as run_module

    body = inspect.getsource(run_module._recheck_orientation)
    assert "mean_a=vote_a.mean" in body and "mean_b=vote_b.mean" in body

    # the means are not touched by the answer under test: which reader is discarded is the same
    # whichever direction the ballots claim, because only `direction_stated_in_text` is compared
    for hib_first, hib_second in ((False, True), (True, False)):
        runs = [ballot(OPUS, hib_first, "signed_direction", "b_greater"),
                ballot(SONNET, hib_second, "higher_more_construct", "unknown", index=1)]
        checked = _recheck_orientation(a_dataset("d1"), verdict_for("d1", runs),
                                       votes_of(-22.4, -27.0), [])
        thrown_out = _discard_note(checked)
        assert OPUS in thrown_out and SONNET not in thrown_out

    # and the same ballots against swapped means discard the OTHER reader — which is why the
    # assignment has to be the dataset's own
    runs = [ballot(OPUS, False, "signed_direction", "b_greater"),
            ballot(SONNET, True, "higher_more_construct", "a_greater", index=1)]
    before = verdict_for("d1", runs)
    a_greater = _recheck_orientation(a_dataset("d1"), before, votes_of(-22.4, -27.0), [])
    b_greater = _recheck_orientation(a_dataset("d1"), before, votes_of(-27.0, -22.4), [])
    assert OPUS in _discard_note(a_greater) and SONNET not in _discard_note(a_greater)
    assert SONNET in _discard_note(b_greater) and OPUS not in _discard_note(b_greater)


def test_the_open_flags_come_from_the_same_candidates_the_means_did():
    """`_verify_cell` recomputes `votes` and `flags` from one `cell` list and hands both to the
    recheck, so a `series_marker_mismatch` raised by the readings the means came from is the flag
    that suppresses the discard against them."""
    import inspect

    from canopy.pipeline import run as run_module

    body = inspect.getsource(run_module._verify_cell)
    recheck = body.index("_recheck_orientation(dataset, orientation, votes, flags)")
    last_votes = body.rindex("votes = vote_groups(cell", 0, recheck)
    last_flags = body.rindex("flags = [*run_checks(dataset, key, cell", 0, recheck)
    assert last_votes < recheck and last_flags < recheck
    assert "vote_groups(cell" in body[last_votes:recheck]


# =========================== a source that does not report THIS contrast's two groups (mapper review)
class _Located:
    """A `value` location, as much of one as the sample filter reads. Deliberately not a `Source`:
    the filter is `getattr`-based so that a map written before `Source.sample` existed still reads,
    and testing it through the model would make the test skip until that field lands."""

    role = "value"
    locator = "Results, para 2"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_a_location_that_reports_one_group_or_a_pooled_sample_is_not_read_for_the_cell():
    """`Source.sample`, via the mapper's own `source_unreadable_reason`. A `value` location whose
    sample is explicitly `one_group`, `pooled` or `other` is evidence about the paper, not about
    THIS contrast: reading it fills the cell with a different set of people. `unknown` and
    `both_groups` are read — `unknown` is the honest absence a map written before the field
    existed carries, and refusing on it would stop reading every paper in the corpus."""
    from canopy.agents.mapper import readable_sources, source_unreadable_reason
    from canopy.models import Source

    def located(**kwargs):
        return Source(role="value", kind="text_mean_sd", page=3, locator="Results", **kwargs)

    for value in ("one_group", "pooled", "other"):
        assert source_unreadable_reason(located(sample=value)), value
    for value in ("unknown", "both_groups"):
        assert source_unreadable_reason(located(sample=value)) == "", value

    readable = readable_sources([located(sample="both_groups"), located(sample="pooled"),
                                 located(sample="unknown")])
    assert [s.sample for s in readable] == ["both_groups", "unknown"]


def test_a_role_that_is_not_value_is_refused_by_the_same_one_rule():
    """The other half `source_unreadable_reason` decides, and the reason the pipeline must not keep
    its own copy: role and sample are one question — may a number be read here — answered once."""
    from canopy.agents.mapper import source_unreadable_reason
    from canopy.models import Source

    for role in ("baseline", "context", "alternate"):
        why = source_unreadable_reason(Source(role=role, kind="text_mean_sd", page=3,
                                              locator="Results"))
        assert why and role in why, (role, why)
    assert source_unreadable_reason(Source(role="value", kind="text_mean_sd", page=3,
                                           locator="Results")) == ""


def test_the_extract_stage_asks_the_map_which_locations_are_readable():
    """Update #7 (b): `_extract_cell` calls the mapper's exported rule rather than re-deriving it.
    A second copy of "may a number be read here" is how the pipeline and the map came to disagree
    about Cressman's aligned-cursor curves in the first place."""
    import inspect

    from canopy.agents import mapper
    from canopy.pipeline import run as run_module

    body = inspect.getsource(run_module._extract_cell)
    assert "readable_sources(sources.sources)" in body
    assert "source_unreadable_reason(skipped)" in body
    assert "not read for the value" in body
    assert run_module.readable_sources is mapper.readable_sources
    assert run_module.source_unreadable_reason is mapper.source_unreadable_reason
    # and no second copy of the rule survives in this module
    assert not hasattr(run_module, "_sample_is_this_contrast")
    assert not hasattr(run_module, "NOT_THIS_CONTRASTS_SAMPLE")


#: a location the map keeps for the record and refuses to read a value at (`role="baseline"`)
BASELINE_LOCATOR = "Results \u00b60 (the pre-adaptation baseline block)"


def _adds_a_baseline_source(request, router):
    """The shape the mapper re-review names: a baseline plotted beside the outcome, which
    `readable_sources` keeps on the record and refuses to read the VALUE at."""
    answer = router(request)
    if set(((request.schema or {}).get("properties") or {})) == {"datasets", "notes"}:
        outcome = answer["datasets"][0]["outcomes"][0]
        baseline = dict(outcome["sources"][0])
        baseline.update(role="baseline", locator=BASELINE_LOCATOR)
        outcome["sources"] = [*outcome["sources"], baseline]
    return answer


def test_the_bought_third_reading_is_offered_only_the_readable_locations(tmp_path):
    """Mapper re-review, amendment G: `_extract_cell` asks `readable_sources` which locations may
    be read for the value, and the third reading the vote buys when two readers disagree passed
    the mapper's WHOLE list instead — so the one reading bought to settle a disagreement was the
    only one allowed to read the baseline the other two were kept away from."""
    from tests.test_pipeline_offline import _request_text

    out, requests = _offline(tmp_path, router_wrapper=_adds_a_baseline_source, disagreement=5.0)
    readers = [r for r in requests
               if {"groups"} <= set(((r.schema or {}).get("properties") or {}))
               and "rationale" not in ((r.schema or {}).get("properties") or {})]
    assert len(readers) >= 3, "the vote did not buy a third reading, so nothing is under test"
    offered = [_request_text(r) for r in readers]
    assert not any(BASELINE_LOCATOR in text for text in offered), \
        "a reader was offered a location the map says carries no value for this outcome"
    # …and the value source really was offered, so the assertion above is not vacuous
    assert all("Results" in text for text in offered)


def test_a_dataset_a_person_excluded_is_attributed_to_the_person_not_the_adjudicator():
    """The mapper review's refinement of m3. `apply_map_answers` lets a reviewer answer an open
    `include_dataset` question with `exclude`, and writes `HUMAN_EXCLUSION_RULE` when they named no
    protocol rule. The exclusions row must say who decided — attributing a person's judgement to
    the adjudicator would put a rule in the record that no model applied."""
    from canopy.agents.mapper import HUMAN_EXCLUSION_RULE, apply_map_answers

    study = _study_with_an_open_inclusion_question()
    answered = apply_map_answers(study, [
        {"kind": "include_dataset", "paper_id": PAPER_ID, "dataset_id": "d1",
         "decision": "exclude", "note": "these are the same people as d0"}])
    dataset = answered.datasets[0]
    assert dataset.included is False
    assert dataset.exclusion_rule == HUMAN_EXCLUSION_RULE
    # …and the orchestrator's rule for the `decider` column reads exactly that
    assert (dataset.exclusion_rule.strip() == HUMAN_EXCLUSION_RULE)

    ruled = apply_map_answers(study, [
        {"kind": "include_dataset", "paper_id": PAPER_ID, "dataset_id": "d1",
         "decision": "exclude", "rule": "the protocol's population rule: adults over 60",
         "quote": "twelve young subjects aged 19-27"}])
    assert ruled.datasets[0].exclusion_rule != HUMAN_EXCLUSION_RULE


def test_a_reviewers_answer_in_the_run_log_unblocks_the_cell_the_pipeline_would_have_skipped(
        tmp_path, monkeypatch):
    """Ruling 3, end to end over the real interface now that both halves exist: the answer is
    written to the run's own review log, and the pipeline — not the UI — applies it.

    Without the answer the dataset is blocked and buys nothing; with it the same run extracts,
    verifies and resolves the dataset as if the question had never been open.
    """
    from canopy.pipeline.overrides import append_override

    seen: dict[str, str] = {}

    def leave_the_dataset_unsettled(mapped):
        study, file_id = mapped
        seen["paper_id"] = study.paper_id
        seen["dataset_id"] = study.datasets[0].dataset_id
        study.open_questions.append(MapQuestion(
            kind="include_dataset", dataset_id=study.datasets[0].dataset_id,
            question="only one mapping agent proposed this dataset"))
        return study, file_id

    blocked_out, _ = _offline(tmp_path / "unanswered", map_wrapper=leave_the_dataset_unsettled,
                              monkeypatch=monkeypatch)
    assert _verify_payload(blocked_out)["verdicts"] == []      # nothing was read, nothing bought
    assert _records(blocked_out) == []

    answered_root = tmp_path / "answered"
    run_dir = answered_root / "run"
    run_dir.mkdir(parents=True)
    append_override(run_dir, {
        "kind": "include_dataset", "paper_id": seen["paper_id"],
        "dataset_id": seen["dataset_id"], "decision": "include", "actor": "a reviewer",
        "justification": "the protocol asks for this sample"})

    answered_out, _ = _offline(answered_root, map_wrapper=leave_the_dataset_unsettled,
                               monkeypatch=monkeypatch)
    assert answered_out == run_dir
    assert _verify_payload(answered_out)["verdicts"], "the answered dataset was still skipped"
    assert any(r["d"] is not None for r in _records(answered_out))


def test_an_answer_written_after_a_run_is_acted_on_by_resuming_that_same_run(tmp_path,
                                                                              monkeypatch):
    """Review H2, end to end on the only path the ruling exists for.

    The run finishes with a dataset blocked by an open `include_dataset` question, so extraction
    was never bought for it and the review page tells the reviewer to re-run with `--resume`. The
    answer is written into that run's own log and the SAME directory is resumed. Before this fix
    the resume returned the cached (empty) candidates before it ever read the answers, nothing
    wrote `consumed_override_seqs`, and the reviewer got the identical "re-run with --resume"
    message back for ever — with no exit but a full re-run that re-pays for every stage of every
    paper.
    """
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool, consumed_seqs

    seen: dict[str, str] = {}

    def leave_the_dataset_unsettled(mapped):
        study, file_id = mapped
        seen["paper_id"] = study.paper_id
        seen["dataset_id"] = study.datasets[0].dataset_id
        study.open_questions.append(MapQuestion(
            kind="include_dataset", dataset_id=study.datasets[0].dataset_id,
            question="only one mapping agent proposed this dataset"))
        return study, file_id

    out, _ = _offline(tmp_path, map_wrapper=leave_the_dataset_unsettled, monkeypatch=monkeypatch)
    assert _extract_payload(out)["candidates"] == []       # blocked: nothing was bought
    assert _verify_payload(out)["verdicts"] == []
    assert consumed_seqs(out) == set()

    append_override(out, {
        "kind": "include_dataset", "paper_id": seen["paper_id"],
        "dataset_id": seen["dataset_id"], "decision": "include", "actor": "a reviewer",
        "justification": "the protocol asks for this sample"})
    assert apply_overrides_and_repool(out)["applied"] == 0        # …and it is pending, correctly

    resumed, _ = _offline(tmp_path, map_wrapper=leave_the_dataset_unsettled,
                          monkeypatch=monkeypatch, resume=True)
    assert resumed == out                                        # the SAME run directory

    payload = _extract_payload(out)
    assert payload["candidates"], "the answered dataset was still skipped on the resume"
    assert payload["consumed_override_seqs"], "nothing recorded acting on the answer"
    assert consumed_seqs(out) == set(payload["consumed_override_seqs"])
    assert any(r["d"] is not None for r in _records(out)), "no signed row for the included dataset"

    summary = apply_overrides_and_repool(out)
    assert summary["applied"] == 1 and not summary.get("pending"), summary.get("pending")


def test_a_row_refused_for_its_denominator_is_findable_in_the_review_queue(tmp_path):
    """C9's screen is on the ROW, and the review queue is built from CELLS. A row the pipeline
    refuses and names nowhere a reviewer works is a refusal nobody can act on: the queue file is
    what the review page and `questions_for_run` read.

    Both readers agree perfectly on values whose spread implies |d| = 143 — the shape the screen
    exists for, and one no amount of agreement can settle.
    """
    out, _ = _offline(tmp_path, sd=0.1)
    row = next(r for r in _records(out))
    assert abs(row["d"]) > 3.0 and row["confidence"] == "needs_human"
    assert "implausible_dispersion" in row["flags"]
    assert all(not v["needs_human"] for v in _verify_payload(out)["verdicts"])  # the cells pooled

    with open(out / "human_review_queue.csv", newline="", encoding="utf-8") as handle:
        queued = list(csv.DictReader(handle))
    mine = [q for q in queued if q["dataset_id"] == row["dataset_id"]
            and q["outcome_key"] == row["outcome_key"]]
    assert {q["group"] for q in mine} == {"A", "B"}
    assert all("implausible_dispersion" in q["reason"] for q in mine), mine


# ------------------------------ the queue a fresh run builds and the queue a re-pool rebuilds
def _cell(group: str, *, needs_human: bool, dataset_id: str = "d1") -> Any:
    from canopy.models import Verdict

    return Verdict(dataset_id=dataset_id, outcome_key="late_adaptation", group=group,
                   confidence="needs_human" if needs_human else "accept_with_note",
                   needs_human=needs_human)


def _row(*flags: str, route: str = "text_mean_sd", dataset_id: str = "d1") -> Any:
    from canopy.models import EffectSizeRecord

    return EffectSizeRecord(paper_id="p1", dataset_id=dataset_id,
                            outcome_key="late_adaptation", route=route,
                            confidence="needs_human", flags=list(flags))


def test_both_cells_of_a_row_the_resolver_refused_stay_in_the_queue():
    """Consistency item 2. The queue is built from cells and the refusal is on the ROW, so the
    "held in its own right" subtraction — right for an ordinary cell finding, whose healthy
    sibling has no question — drops the sibling of a cell held BY THE ROW. Answer the first cell
    and the second disappears, while the row is still refused: half the question, gone.

    Pinned against the orchestrator's own function, which the re-pool applies too.
    """
    from canopy.pipeline.run import cells_for_review
    from canopy.verify.confidence import ROW_REFUSAL_CODES

    cells = [_cell("A", needs_human=True), _cell("B", needs_human=False)]
    for code in ROW_REFUSAL_CODES:
        queued = cells_for_review(cells, [_row(code)])
        assert {v.group for v in queued} == {"A", "B"}, code
    # …and with no refusal on the row, the healthy sibling is left alone exactly as before
    ordinary = cells_for_review(cells, [_row("group_label_swapped")])
    assert {v.group for v in ordinary} == {"A"}
    # a row held by a rule NEITHER cell carries still brings both cells, as it did before
    healthy = [_cell("A", needs_human=False), _cell("B", needs_human=False)]
    assert {v.group for v in cells_for_review(healthy, [_row("df_missing")])} == {"A", "B"}
    # …and a `not_convertible` row is a paper that did not report enough: it belongs in
    # `exclusions`, not in the queue
    assert cells_for_review(healthy, [_row(route="not_convertible")]) == []


def test_a_cell_a_person_excluded_is_not_a_cell_awaiting_review():
    """The other half of consistency item 2, and the other half of a re-pool's rule: a decision a
    reviewer took is on the record and in the exclusion table, so counting it as outstanding
    counts it for ever. Only a HUMAN exclusion settles a cell this way — a row the code dropped
    as `not_convertible` still needs whatever review its cells need."""
    from canopy.agents.mapper import HUMAN_DECIDER
    from canopy.pipeline.run import cells_for_review

    cells = [_cell("A", needs_human=True), _cell("B", needs_human=True)]
    assert len(cells_for_review(cells, [], [])) == 2

    by_dataset = [{"dataset_id": "d1", "outcome_key": "", "decider": HUMAN_DECIDER}]
    assert cells_for_review(cells, [], by_dataset) == []
    by_cell = [{"dataset_id": "d1", "outcome_key": "late_adaptation", "decider": HUMAN_DECIDER}]
    assert cells_for_review(cells, [], by_cell) == []
    # a different cell of the same paper is untouched
    other = [{"dataset_id": "d1", "outcome_key": "aftereffect", "decider": HUMAN_DECIDER}]
    assert len(cells_for_review(cells, [], other)) == 2
    # and an exclusion the CODE made never removes a question from a person
    for decider in ("code", "map-adjudicator"):
        rows = [{"dataset_id": "d1", "outcome_key": "late_adaptation", "decider": decider}]
        assert len(cells_for_review(cells, [], rows)) == 2, decider


def _every_location_is_a_baseline(request, router):
    """The map finds only locations that carry no value for this outcome, so nothing is read."""
    answer = router(request)
    if set(((request.schema or {}).get("properties") or {})) == {"datasets", "notes"}:
        outcome = answer["datasets"][0]["outcomes"][0]
        outcome["sources"] = [{**dict(source), "role": "baseline"}
                              for source in outcome["sources"]]
    return answer


def test_a_cell_with_no_candidate_buys_no_direction(tmp_path):
    """Review H2's second half, as a call count. The two orientation reads are per
    (outcome, measure) rather than per candidate, so a cell the extract stage read nothing for
    still bought them — two calls, and two verdicts whose only content was a direction for
    numbers that do not exist. There is nothing to sign, so nothing is bought."""
    out, requests = _offline(tmp_path, router_wrapper=_every_location_is_a_baseline)
    assert _extract_payload(out)["candidates"] == []
    assert _orientation_calls(requests) == []
    assert _verify_payload(out)["verdicts"] == []
    assert _records(out) == []


# ============ controller ruling: an orientation abstention must not buy an adjudication
def _adjudicator_calls(requests) -> list[Any]:
    """The fake routes the adjudicator by its schema: `groups` AND `rationale` together."""
    out = []
    for request in requests:
        props = set(((request.schema or {}).get("properties") or {}))
        if {"groups", "rationale"} <= props:
            out.append(request)
    return out


def test_a_cell_whose_only_error_is_the_orientation_contradiction_buys_no_adjudicator():
    """The trigger itself, at the predicate. The adjudicator settles VALUE disputes — which of two
    readings of a number is right — and by ruling it may not decide a direction. So a cell whose
    only error is C3's contradiction has nothing for it to settle, and is already waiting for the
    person who will answer the orientation question.

    Asserted through `ORIENTATION_FLAGS` rather than a literal, so a code the orientation area adds
    there is excluded here without anyone remembering to.
    """
    from canopy.verify.checks import ORIENTATION_FLAGS, severity_of

    # the premise: this really is an `error`, so before the ruling it really did buy a call
    assert severity_of("orientation_reader_contradicts_values") == "error"
    assert _trigger(["orientation_reader_contradicts_values"]) is False
    for code in ORIENTATION_FLAGS:
        assert _trigger([code]) is False, code
    # …and anything else still adjudicates exactly as before
    assert _trigger(["orientation_reader_contradicts_values", "sign_mismatch"]) is True
    assert _trigger(["sign_mismatch"]) is True


def _trigger(codes) -> bool:
    """The predicate `_verify_cell` really uses, called — not a copy of it written here.

    The review's own criticism of the first cut of this test: a test that re-implements the
    expression it is checking passes when the expression moves. `buys_adjudication` is the
    orchestrator's own function.
    """
    from canopy.pipeline.run import buys_adjudication
    from canopy.verify.checks import severity_of

    return buys_adjudication([CheckFlag(code=c, severity=severity_of(c), message="")
                              for c in codes])


def test_a_missing_df_buys_no_adjudicator_because_no_model_can_print_the_paper_a_df():
    """Review M5, widened by the fix round's controller ruling. All three codes are `error` — the
    row must not pool — and none of them is a VALUE dispute: "this t is printed with no degrees of
    freedom" is a fact about the paper, and the flags are computed before the call and passed
    unchanged into `resolve_cell`, so the cell is held whatever the adjudicator answers.

    The two C9 codes did not exist at HEAD and shipping them as errors bought the adjudicator
    model, whole paper in context, for an answer it cannot give. `test_stat_missing_df` DID exist
    and bought it at HEAD too, for the same non-answer — the ruling stops that as well, so the
    exclusion is not inert on the commonest shape (a t with no df at all raises both codes)."""
    from canopy.verify.checks import DF_PROVENANCE_FLAGS, severity_of

    assert set(DF_PROVENANCE_FLAGS) == {"df_missing", "df_shortfall_unexplained",
                                        "test_stat_missing_df"}
    for code in DF_PROVENANCE_FLAGS:
        assert severity_of(code) == "error", code       # the premise: still an error, still held
        assert _trigger([code]) is False, code
    # a real VALUE dispute on the same cell still adjudicates
    assert _trigger(["df_missing", "sd_nonpositive"]) is True


def _prints(statistic: dict):
    """A router wrapper that makes the paper print one statistic for this outcome.

    It remembers the paper's own sentence and quotes the statistic against it, so the invented row
    is grounded in the real text — an ungrounded quote is itself an `error`, and would adjudicate,
    which would make every call count below say nothing.
    """
    quoted: dict[str, str] = {}

    def wrapper(request, router):
        answer = router(request)
        props = set(((request.schema or {}).get("properties") or {}))
        if props == {"datasets", "notes"}:    # the mapper's source list: add a place to read it
            outcome = answer["datasets"][0]["outcomes"][0]
            printed = dict(outcome["sources"][0])
            quoted["quote"] = printed["quote"]
            printed.update(kind="test_statistic", locator="Results \u00b62")
            outcome["sources"] = [*outcome["sources"], printed]
            return answer
        if "statistics" in props:
            return {"notes": "", "statistics": [{**statistic,
                                                 "quote": quoted.get("quote", "")}]}
        return answer
    return wrapper


#: one statistic row in `EXTRACT_STATS_SCHEMA`'s shape, to be varied per test
_A_STATISTIC: dict[str, Any] = {
    "kind": "test_statistic", "status": "found", "page": 3, "locator": "Results \u00b62",
    "quote": "", "effect_as_written": "", "compares_the_two_groups": "yes",
    "contrast_kind": "groups", "design": "independent_t", "stat_type": "t",
    "stat_value": 5.25, "df": None, "df1": None, "df2": None, "within_factors": [],
    "outcome_averages_over": [], "model_fitted_to": "", "tails": 2, "p_kind": "less_than",
    "p_value": 0.001, "direction": "a_greater", "direction_quote": "", "reported_value": None,
    "reported_scale": "unknown", "standardizer": "unknown", "reported_ci_low": None,
    "reported_ci_high": None, "positive_means": "unknown", "admissible": True,
    "admissible_reason": "", "notes": ""}

#: `F(1, ?) = 27.6` — a numerator df and no error df. `df_missing` catches it;
#: `test_stat_missing_df` does not, because `df1` is not None.
_F_WITH_NO_ERROR_DF: dict[str, Any] = {**_A_STATISTIC, "stat_type": "F", "stat_value": 27.6,
                                       "df1": 1.0, "design": "one_way_between",
                                       "effect_as_written": "F(1) = 27.6"}
#: `t = 5.25, p < .001` with NO degrees of freedom at all — the failing input C9 was written
#: against, and the shape that raises BOTH codes.
_T_WITH_NO_DF: dict[str, Any] = {**_A_STATISTIC, "effect_as_written": "t = 5.25"}


@pytest.mark.parametrize("statistic, expected", [
    (_F_WITH_NO_ERROR_DF, {"df_missing"}),
    (_T_WITH_NO_DF, {"df_missing", "test_stat_missing_df"}),
], ids=["F(1,?)", "t with no df"])
def test_a_statistic_printed_without_degrees_of_freedom_buys_no_adjudicator_call(
        tmp_path, statistic, expected):
    """M5's call count, end to end, on both shapes: the codes are raised, the cell is held for
    them, and the adjudicator model — the whole paper in context — is not bought for an answer it
    cannot give.

    The second case is the controller's fix-round ruling. `test_stat_missing_df` is an `error` at
    HEAD and says the same thing as `df_missing` about the same statistic, so while it was off the
    excluded list the ruling was inert on the commonest shape it exists for: a t printed with no
    df at all still bought the adjudicator, and the cell was held whatever it answered.
    """
    out, requests = _offline(tmp_path, router_wrapper=_prints(statistic))
    payload = _verify_payload(out)
    raised = {f["code"] for v in payload["verdicts"] for f in v["flags"]}
    assert raised & _DF_CODES == expected, sorted(raised)
    assert all(v["confidence"] == "needs_human" for v in payload["verdicts"])
    assert payload["adjudications"] == []
    assert _adjudicator_calls(requests) == []


#: every code that can say "the degrees of freedom do not establish this contrast", so the
#: parametrisation above asserts which of them fired and which did not
_DF_CODES = frozenset({"df_missing", "df_shortfall_unexplained", "test_stat_missing_df"})


def test_an_implausible_denominator_no_longer_buys_an_adjudicator_at_the_cell():
    """The third of the three new `error` codes. It is a `warn` now (H1): the binding screen is on
    the RESOLVED |d|, which is computed after the adjudication — so an adjudicator bought here to
    settle the denominator would supply a value this check never looks at."""
    from canopy.verify.checks import severity_of

    assert severity_of("implausible_dispersion") == "warn"
    assert _trigger(["implausible_dispersion"]) is False


def _readers_split_on_the_stated_direction(request, router):
    """Heuer & Hegele d1 `late_adaptation`, as the real run recorded it: opus reads the paper's own
    sentence as `a_greater` and sonnet as `b_greater`.

    The means say A is the greater, so sonnet is discarded (C3 row 1); the two stated directions
    conflict, so the verdict carries NO agreed `direction_stated_in_text` and `sign_check` has
    nothing to compare — which leaves `orientation_reader_contradicts_values` as the cell's only
    error. That is the shape the ruling is about.
    """
    answer = router(request)
    if isinstance(answer, dict) and "raw_value_semantics" in answer:
        stated = "a_greater" if request.model == MODELS["primary"] else "b_greater"
        return {**answer, "direction_stated_in_text": stated}
    return answer


def test_a_run_whose_orientation_was_discarded_buys_zero_adjudicator_calls(tmp_path):
    """The ruling end to end, on the one real cell in `runs/rerun-fixed` that C3 row 1 changes.

    Before the ruling this bought an adjudicator call on a cell whose only problem the adjudicator
    is forbidden to decide — and whose values nobody disputed.
    """
    out, requests = _offline(tmp_path, router_wrapper=_readers_split_on_the_stated_direction)
    payload = _verify_payload(out)

    verdict = next(iter(payload["orientation"].values()))
    assert "orientation_reader_contradicts_values" in verdict["notes"]   # the discard happened
    for cell in payload["verdicts"]:
        codes = {f["code"] for f in cell["flags"] if f["severity"] == "error"}
        assert codes == {"orientation_reader_contradicts_values"}, codes   # and is the ONLY error
    assert payload["adjudications"] == []                                  # so nothing was bought
    assert _adjudicator_calls(requests) == []


def test_the_same_cell_with_a_refuted_reading_still_buys_one(tmp_path):
    """The control, and the half of the ruling that must NOT change: a cell that also has a real
    VALUE problem is exactly what the adjudicator is for, and it is bought as before.

    A refutation rather than a vote disagreement, because two text readers that disagree leave the
    cell with no resolved mean at all — and with no means the discard filter is disabled, so that
    route cannot produce the two conditions together.
    """
    out, requests = _offline(tmp_path, router_wrapper=_readers_split_on_the_stated_direction,
                             refute=True)
    payload = _verify_payload(out)

    assert "orientation_reader_contradicts_values" in next(
        iter(payload["orientation"].values()))["notes"]
    assert payload["adjudications"], "a refuted reading must still reach the adjudicator"
    assert _adjudicator_calls(requests)


def test_an_ordinary_error_on_the_same_cell_still_adjudicates(tmp_path):
    """And so does any OTHER error. Both readers stating a direction the means contradict raises
    `sign_mismatch` beside the orientation code, and `sign_mismatch` is not an orientation flag —
    so the cell adjudicates exactly as before. (Reported to the controller: this is the COMMON
    shape, and it means the ruling bites only where the readers also disagree with each other.)"""
    def both_readers_contradict(request, router):
        answer = router(request)
        if isinstance(answer, dict) and "raw_value_semantics" in answer:
            return {**answer, "direction_stated_in_text": "b_greater"}
        return answer

    out, requests = _offline(tmp_path, router_wrapper=both_readers_contradict)
    payload = _verify_payload(out)
    errors = {f["code"] for v in payload["verdicts"] for f in v["flags"]
              if f["severity"] == "error"}
    assert errors == {"orientation_reader_contradicts_values", "sign_mismatch"}
    assert payload["adjudications"] and _adjudicator_calls(requests)


# ================= controller ruling: `panel_not_isolated` is doubt (caps), not a hold
def _figure_candidate(provenance: dict, cid: str = "ens-A"):
    return Candidate(
        candidate_id=cid, dataset_id="d1", outcome_key="late_adaptation", group="A",
        kind="group_stats", status="found", extractor_id="digitize:ensemble",
        source_kind="figure_bar", mean=31.1, dispersion_value=6.0,
        dispersion_type=DispersionType.SD, n=9, locator="Fig. 3b",
        pixel_provenance=provenance)


def test_a_panel_that_could_not_be_isolated_is_flagged_from_the_digitizers_own_provenance():
    """The flag has to REACH `confidence`, and the digitiser records it as a top-level key on the
    ensemble row's provenance (`fix-ingest-round1.md` F1). A code nothing lifts out of provenance
    is a code that caps nothing."""
    from canopy.verify.checks import run_checks, severity_of

    cand = _figure_candidate({"figure_id": "fig03", "panel_not_isolated": True,
                              "needs_review": True, "needs_review_kind": "calibration",
                              "needs_review_reason": "Fig. 3b names panel b and ingestion "
                                                     "produced one panel"})
    flags = run_checks(a_dataset("d1"), "late_adaptation", [cand])
    mine = [f for f in flags if f.code == "panel_not_isolated"]
    assert len(mine) == 1, [f.code for f in flags]
    assert mine[0].severity == "warn" == severity_of("panel_not_isolated")
    assert "Fig. 3b names panel b" in mine[0].message     # the digitiser's own reason travels
    assert mine[0].candidate_ids == [cand.candidate_id]

    # the negative twin: a figure whose locator named no panel is not flagged
    assert not [f for f in run_checks(a_dataset("d1"), "late_adaptation",
                                      [_figure_candidate({"figure_id": "fig05"})])
                if f.code == "panel_not_isolated"]


def test_the_union_crop_caps_the_cell_and_does_not_withhold_it():
    """DOUBT caps, CONTRADICTION withholds. The union crop may hold a neighbouring panel's ladder,
    which is a reason for a person to look — but the axis-identity, overlay and verifier nets all
    still apply to the number, so withholding on this alone would bury correct readings. (The fix
    round proposed a 0.44 cap, which WOULD hold cells; the ruling rejected it.)

    Capped at `ADJUDICATED_CAP` and floored at `ACCEPT_WITH_NOTE`, exactly like its neighbours.
    """
    from canopy.models import OrientationVerdict
    from canopy.verify.checks import run_checks
    from canopy.verify.confidence import (ACCEPT_WITH_NOTE, ADJUDICATED_CAP, CAPPING_FLAGS,
                                          CAP_REASONS, CONTRADICTING_FLAGS, confidence)
    from canopy.verify.vote import RouteValue

    assert "panel_not_isolated" in CAPPING_FLAGS
    assert "panel_not_isolated" not in CONTRADICTING_FLAGS
    assert CAPPING_FLAGS.isdisjoint(CONTRADICTING_FLAGS)      # the import-time assertion, restated

    orientation = OrientationVerdict(outcome_key="late_adaptation", dataset_id="d1",
                                     higher_is_better=False, agreed=True, needs_human=False)
    base = {"figure_id": "fig03", "model_families": ["claude-opus", "claude-sonnet"]}

    def score_of(provenance):
        cand = _figure_candidate(provenance)
        vote = VoteResult(group="A", agreement="single", mean=31.1, dispersion_value=6.0, n=9,
                          method="single", agreeing_ids=[cand.candidate_id],
                          routes=[RouteValue(route_key="figure/claude-opus", value=31.1,
                                             candidate_ids=[cand.candidate_id])])
        flags = run_checks(a_dataset("d1"), "late_adaptation", [cand], orientation=orientation)
        return confidence(vote, [], flags, candidates=[cand], n_a=12, n_b=12,
                          orientation=orientation)

    bucket, score, reasons = score_of({**base, "panel_not_isolated": True,
                                       "needs_review_reason": "the panel could not be isolated"})
    assert bucket != "auto_accept" and score <= ADJUDICATED_CAP    # it cannot be automatic…
    assert bucket != "needs_human" and score >= ACCEPT_WITH_NOTE   # …and it is not withheld
    assert CAP_REASONS["panel_not_isolated"] in " ".join(reasons)
    assert f"caps this cell at {ADJUDICATED_CAP:.2f}" in " ".join(reasons)

    # and the cap is THIS flag's doing: the same cell without it scores strictly higher, on the
    # same route, the same readers and the same number
    clean_bucket, clean_score, clean_reasons = score_of(base)
    assert clean_score > score and clean_bucket != "needs_human"
    assert CAP_REASONS["panel_not_isolated"] not in " ".join(clean_reasons)


# ============ controller update #5: the PERSISTED verdict must say what the check actually did
def test_the_persisted_verdict_says_the_means_check_ran(tmp_path):
    """Re-review finding N1. `_recheck_orientation` used to return the PRE-check verdict whenever
    `(hib, needs_human, agreed)` was unchanged — true of 11 of the 13 replayed cells, including all
    three pooled ones — so `verify.json` kept `means check: no_means`, written when orientation was
    decided and there were no means yet, on a cell where the check had since run.

    A verdict that says the check could not run, on a cell where it did, is a false statement about
    the evidence — and it is the statement someone would rely on to conclude the check had never
    been wired in at all.
    """
    from canopy.agents.orientation import MEANS_CHECK_NOTE

    out, _ = _offline(tmp_path)
    notes = next(iter(_verify_payload(out)["orientation"].values()))["notes"]
    assert f"{MEANS_CHECK_NOTE}: ran" in notes
    assert f"{MEANS_CHECK_NOTE}: no_means" not in notes
    assert "resolved raw means" in notes            # and it names the numbers it checked against


def test_the_persisted_verdict_says_disputed_when_the_series_identity_was_open(tmp_path,
                                                                               monkeypatch):
    """The other state a reviewer must be able to tell apart: the check was SUSPENDED because
    which series is which was itself in dispute, not skipped for want of means."""
    from canopy.agents.orientation import MEANS_CHECK_NOTE
    from canopy.verify import checks as checks_module

    real_run_checks = checks_module.run_checks

    def with_a_disputed_series(*args, **kwargs):
        flags = real_run_checks(*args, **kwargs)
        if not any(f.code == "series_marker_mismatch" for f in flags):
            flags.append(CheckFlag(code="series_marker_mismatch", severity="warn",
                                   message="the markers the reader named do not line up"))
        return flags

    from canopy.pipeline import run as run_module

    monkeypatch.setattr(run_module, "run_checks", with_a_disputed_series)
    out, _ = _offline(tmp_path)
    notes = next(iter(_verify_payload(out)["orientation"].values()))["notes"]
    assert f"{MEANS_CHECK_NOTE}: disputed" in notes
    assert "series_marker_mismatch" in notes
    assert "orientation_reader_contradicts_values" not in notes     # and nobody was discarded


# ============ update #7 (a): a printed effect size carries its CONTRAST to the route that uses it
def _reported_candidate(contrast_kind: str):
    return Candidate(
        candidate_id="rep-1", dataset_id="d1", outcome_key="late_adaptation",
        kind="reported_d", status="found", extractor_id="text:narrative:opus",
        source_kind="reported_effect_size", reported_value=1.30, reported_scale="cohens_d",
        standardizer="pooled_sd_between", positive_means="a_greater", contrast_kind=contrast_kind,
        page=6, locator="Results", quote="the groups differed, d = 1.30")


def test_a_printed_effect_size_carries_its_contrast_to_the_resolver():
    """P-B. `_reported_values()` dropped `contrast_kind`, so every printed d reached the resolver
    as `unknown` — which the gate refuses. Fail-closed, so never a wrong number, but no printed
    effect size could ever fill a cell."""
    from canopy.pipeline.run import _reported_values

    assert _reported_values([_reported_candidate("groups")]).contrast_kind == "groups"
    assert _reported_values([_reported_candidate("against_constant")]).contrast_kind == \
        "against_constant"
    # an extraction that recorded nothing stays the refusing default, never a permission
    blank = _reported_candidate("groups")
    blank.contrast_kind = ""
    assert _reported_values([blank]).contrast_kind == "unknown"


def test_a_reported_d_converts_only_when_it_contrasts_the_two_groups():
    """The consequence, at the route. "The aftereffect differed from zero, d = 1.30" is a
    one-sample effect and pooled as the between-group difference once; it is refused now, and a
    genuine two-group d still converts."""
    from canopy.models import OutcomeDef, StatsSettings
    from canopy.pipeline.resolve import ResolvedValues, resolve_effect
    from canopy.pipeline.run import _reported_values

    outcome = OutcomeDef(key="late_adaptation", label="late adaptation",
                         definition="the mean direction error over the last adaptation block")
    settings = StatsSettings(route_precedence=["reported_d"])

    def record_for(contrast_kind):
        return resolve_effect(a_dataset("d1"), outcome, ResolvedValues(
            dataset_id="d1", outcome_key="late_adaptation", higher_is_better=False,
            confidence="auto_accept",
            reported=_reported_values([_reported_candidate(contrast_kind)])), settings)

    converted = record_for("groups")
    assert converted.route == "reported_d" and converted.d is not None

    refused = record_for("against_constant")
    assert refused.route == "not_convertible"
    assert "against_constant" in refused.not_convertible_reason


# ===================================================== the whole-diff review's cross-area seams
# H1/H2/M1/M2/M3 and the low findings of `ceiling/review-whole-diff.md`. Each one is a seam: two
# areas that each pass their own tests and disagree about the row that reaches the forest plot.
REPO = Path(__file__).resolve().parents[1]
RERUN = REPO / "runs" / "rerun-fixed"
_HAS_RERUN = (RERUN / "manifest.json").exists()


def _clone_rerun(tmp_path: Path, name: str = "rerun") -> Path:
    import shutil

    clone = tmp_path / name
    shutil.copytree(RERUN, clone,
                    ignore=shutil.ignore_patterns("cache", "thumbnails", "*.png", "*.svg"))
    (clone / "overrides.jsonl").unlink(missing_ok=True)
    return clone


def _shared_control_run(out: Path, *, split: str = "split_n"):
    """Two comparisons against ONE control arm, resolved exactly as `run._resolve` resolves them.

    The construction the whole-diff review used for H1: `shared_control=True` on both datasets, a
    control of n = 20 shared between them, and the protocol's `split_n` strategy — so the run
    gives each row n_B = 10 and the flag that says so.
    """
    import shutil

    from canopy.models import (Citation, DatasetSpec, GroupSpec, OutcomeSources, PaperStatus,
                               RunManifest, Source, SourceKind, StudyMap, Verdict)
    from canopy.pipeline.rows import prepare_rows
    from canopy.pipeline.resolve import resolve_effect
    from canopy.pipeline.state import save_manifest, write_stage
    from canopy.protocol import load_protocol
    from canopy.report import extraction_table
    from tests.test_pipeline_offline import PROTOCOL

    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROTOCOL, out / "protocol.yaml")
    protocol = load_protocol(out / "protocol.yaml")
    protocol.stats.shared_control_strategy = split
    key = protocol.outcomes[0].key
    paper = "b" * 64

    def spec(i: int) -> DatasetSpec:
        return DatasetSpec(
            dataset_id=f"bbbbbbbbbbbb:d{i}", label=f"arm {i} vs control", cluster_id=paper,
            shared_control=True,
            group_a=GroupSpec(label=f"trained arm {i}", n=10, n_evidence="10"),
            group_b=GroupSpec(label="one control group", n=20, n_evidence="20"),
            outcomes=[OutcomeSources(outcome_key=key, measure_name="angular error",
                                     higher_is_better=False,
                                     sources=[Source(kind=SourceKind.table, page=4,
                                                     locator="Table 2", role="value")])])

    datasets = [spec(1), spec(2)]
    study = StudyMap(paper_id=paper, citation=Citation(first_author="Synthetic", year=2021),
                     eligible=True, datasets=datasets)

    def verdict(dataset, group, mean) -> Verdict:
        return Verdict(dataset_id=dataset.dataset_id, outcome_key=key, group=group,
                       agreement="agree", agreeing_ids=["x", "y"], mean=mean,
                       dispersion_value=4.0, dispersion_type=DispersionType.SD,
                       n=(10 if group == "A" else 20), route="table", higher_is_better=False,
                       confidence="accept_with_note", confidence_score=0.6, needs_human=False,
                       verifier_verdict="confirmed")

    verdicts = [verdict(d, g, m) for d in datasets for g, m in (("A", 12.0), ("B", 8.0))]
    by = {(v.dataset_id, v.group): v for v in verdicts}
    prepared = prepare_rows([(d, key, by[(d.dataset_id, "A")], by[(d.dataset_id, "B")])
                             for d in datasets], [], protocol.stats,
                            cluster_of=lambda d: d.cluster_id or paper)
    records = []
    for row in prepared:
        record = resolve_effect(row.dataset, protocol.outcome(key), row.values, protocol.stats)
        record.paper_id = record.cluster_id = paper
        record.citation, record.label = study.citation, row.dataset.label
        records.append(record)
    write_stage(out, paper, "map", {"study": study.model_dump(mode="json")})
    write_stage(out, paper, "extract", {"candidates": [], "complete": True, "cells_extracted": []})
    write_stage(out, paper, "verify", {"verdicts": [v.model_dump(mode="json") for v in verdicts],
                                       "extra_candidates": [], "orientation": {},
                                       "adjudications": [], "source_rank": {}, "reopens": 0})
    write_stage(out, paper, "resolve", {"records": [r.model_dump(mode="json") for r in records]})
    save_manifest(out, RunManifest(
        run_id="sc", created_at="2026-08-18T00:00:00+00:00",
        protocol_path=str(out / "protocol.yaml"), protocol_hash="x",
        papers=[PaperStatus(paper_id=paper, filename="synthetic.pdf", status="resolved",
                            stages={"map": "done", "extract": "done", "verify": "done",
                                    "resolve": "done"})]))
    extraction_table(records, out / "results" / "extraction_table_all", verdicts=verdicts,
                     candidates=[], primary=records)
    (out / "exclusions.json").write_text("[]")
    (out / "human_review_queue.json").write_text("[]")
    return paper, key, records


def test_a_reviewer_confirming_one_cell_does_not_un_split_a_shared_control(tmp_path):
    """Whole-diff H1. `_rebuild_row` was `ResolvedValues.from_verdicts(a, b)` and nothing else, and
    it is on EVERY human release path. The Cochrane 16.5.4 split lives one level up — it needs
    every comparison that uses the control arm at once — so the commonest answer in the whole tool
    ("yes, this value is right") gave the row back the control's full n: variance 0.22778 → 0.16786,
    +36% weight in the pooled estimate, and the flag that says the arm is shared, gone.
    """
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool

    out = tmp_path / "run"
    paper, key, records = _shared_control_run(out)
    assert [r.n_b for r in records] == [10, 10]
    assert all("shared_control_split" in r.flags for r in records)
    before = {r.dataset_id: (r.n_b, round(r.var, 6), r.d) for r in records}

    append_override(out, {"kind": "mark_reviewed", "paper_id": paper,
                          "dataset_id": "bbbbbbbbbbbb:d1", "outcome_key": key, "group": "A",
                          "justification": "checked Table 2 for arm 1",
                          "confidence": "accept_with_note"})
    summary = apply_overrides_and_repool(out)
    assert summary["applied"] == 1 and not summary["pending"], summary["pending"]

    rows = {r["dataset_id"]: r for r in
            json.loads((out / "results" / "extraction_table_all.json").read_text())}
    answered = rows["bbbbbbbbbbbb:d1"]
    assert round(answered["var"], 6) == before["bbbbbbbbbbbb:d1"][1], \
        "the reviewer's answer changed the row's variance by un-splitting the shared control"
    assert "shared_control_split" in answered["flags"]
    assert answered["es"] == before["bbbbbbbbbbbb:d1"][2]
    # …and the sibling nobody answered is untouched, which is what makes the first assertion mean
    # something: both rows still carry the split, so the answer moved neither.
    assert round(rows["bbbbbbbbbbbb:d2"]["var"], 6) == before["bbbbbbbbbbbb:d2"][1]
    assert "shared_control_split" in rows["bbbbbbbbbbbb:d2"]["flags"]


def test_the_same_answer_through_a_value_and_a_direction_keeps_the_split_too(tmp_path):
    """The other two release paths (`value`, `orientation`) go through the same rebuild, so the
    same construction has to hold for them — the review found the defect on `mark_reviewed`
    because that is the newest path, not because the others are safe by construction."""
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool

    out = tmp_path / "run"
    paper, key, records = _shared_control_run(out)
    var = round(records[0].var, 6)

    append_override(out, {"kind": "value", "paper_id": paper, "dataset_id": "bbbbbbbbbbbb:d1",
                          "outcome_key": key, "group": "A", "mean": 12.0,
                          "justification": "Table 2 row 1 reads 12.0"})
    append_override(out, {"kind": "orientation", "paper_id": paper,
                          "dataset_id": "bbbbbbbbbbbb:d1", "outcome_key": key,
                          "measure_name": "angular error", "higher_is_better": False,
                          "justification": "a larger angular error is a worse score"})
    apply_overrides_and_repool(out)
    rows = {r["dataset_id"]: r for r in
            json.loads((out / "results" / "extraction_table_all.json").read_text())}
    for dataset_id, row in rows.items():
        assert round(row["var"], 6) == var, dataset_id
        assert "shared_control_split" in row["flags"], dataset_id


def test_a_re_pool_rebuilds_the_row_the_run_resolved_and_not_a_different_one(tmp_path):
    """The rule the split, the statistic and the policy flags are all instances of: **the row the
    review layer rebuilds for a cell nobody answered is the row the run resolved.**

    Asserted on the whole `ResolvedValues` — the thing `_resolve` prepares and `_rebuild_row` used
    to re-derive from two verdicts alone — and then on the whole record, so the next input
    `_resolve` learns to assemble cannot quietly go missing from the review path again. The only
    fields allowed to differ are the ones an override is FOR: the human's own flag and note.
    """
    from canopy.pipeline import overrides as ov
    from canopy.pipeline.resolve import resolve_effect
    from canopy.pipeline.state import load_manifest
    from canopy.protocol import load_protocol

    out = tmp_path / "run"
    paper, key, records = _shared_control_run(out)
    protocol = load_protocol(out / "protocol.yaml")
    protocol.stats.shared_control_strategy = "split_n"
    state = ov._RunState(out, load_manifest(out))
    live = {(v.dataset_id, v.outcome_key, v.group): v for v in state.verdicts}

    for record in records:
        dataset = state.datasets[record.dataset_id]
        prepared = ov._prepare(dataset, key, live[(record.dataset_id, key, "A")],
                               live[(record.dataset_id, key, "B")], protocol,
                               state=state, verdicts=live)
        rebuilt = resolve_effect(dataset, protocol.outcome(key), prepared.values, protocol.stats)
        was, now = record.model_dump(mode="json"), rebuilt.model_dump(mode="json")
        for field in ("paper_id", "cluster_id", "sample_id", "citation", "label", "moderators",
                      "analysis_metric", "notes"):
            now.pop(field, None), was.pop(field, None)
        assert now == was, record.dataset_id


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_marking_a_pooled_row_reviewed_keeps_the_rows_own_policy_flags(tmp_path):
    """H1 on the corpus. Bock d1 late adaptation is one of the three rows that pool, and it carries
    `multi_group_closest_to_definition` — the record that this paper reported more than the two
    groups being contrasted and which pair was chosen. A reviewer confirming the figure reading
    dropped it: the row went into the analysis without the note that says the pair was picked.
    """
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool

    run = _clone_rerun(tmp_path)
    bock, key = "b511dbb76fa6", "late_adaptation"
    apply_overrides_and_repool(run)
    rows = {(r["dataset_id"], r["outcome_key"]): r for r in
            json.loads((run / "results" / "extraction_table_all.json").read_text())}
    before = rows[(f"{bock}:d1", key)]
    assert "multi_group_closest_to_definition" in before["flags"]

    append_override(run, {"kind": "mark_reviewed", "paper_id": bock + "0" * 52,
                          "dataset_id": f"{bock}:d1", "outcome_key": key, "group": "A",
                          "justification": "I opened Figure 1 and this is what it shows",
                          "confidence": "accept_with_note"})
    apply_overrides_and_repool(run)
    after = {(r["dataset_id"], r["outcome_key"]): r for r in
             json.loads((run / "results" / "extraction_table_all.json").read_text())
             }[(f"{bock}:d1", key)]
    assert "multi_group_closest_to_definition" in after["flags"]
    assert after["es"] == before["es"] and after["route"] == before["route"]

    # …and the OTHER row flag the corpus carries: a dispersion the digitiser built rather than
    # read. Heuer d1 late adaptation has both halves of it, and a sensitivity analysis that pools
    # with and without such rows needs it to survive a reviewer's answer.
    heuer = "3570e4ce2a9c"
    approximated = {(r["dataset_id"], r["outcome_key"]): r for r in
                    json.loads((run / "results" / "extraction_table_all.json").read_text())
                    }[(f"{heuer}:d1", key)]["flags"]
    assert {"dispersion_approximated", "dispersion_approximated:mean_of_point_sd"} <= set(
        approximated)
    append_override(run, {"kind": "mark_reviewed", "paper_id": heuer + "0" * 52,
                          "dataset_id": f"{heuer}:d1", "outcome_key": key, "group": "A",
                          "justification": "I opened the figure and this is what it shows",
                          "confidence": "accept_with_note"})
    apply_overrides_and_repool(run)
    kept = {(r["dataset_id"], r["outcome_key"]): r for r in
            json.loads((run / "results" / "extraction_table_all.json").read_text())
            }[(f"{heuer}:d1", key)]["flags"]
    assert {"dispersion_approximated", "dispersion_approximated:mean_of_point_sd"} <= set(kept)


# --------------------------- H2: a row whose value the paper printed as a statistic, not as means
def _statistic_run(out: Path, *, df: float | None = 22.0, decoy: bool = False):
    """A cell whose ONLY source is a printed `t(22) = 2.50` — the review's H2 construction.

    `df=None` is the variant the conversion refuses: nothing establishes that the t is the
    contrast between these two groups, so the row is `not_convertible` and no answer may release
    it.
    """
    import shutil

    from canopy.models import (Candidate, Citation, DatasetSpec, GroupSpec, OrientationVerdict,
                               OutcomeSources, PaperStatus, RunManifest, Source, SourceKind,
                               StudyMap)
    from canopy.pipeline.resolve import resolve_effect
    from canopy.pipeline.rows import prepare_rows
    from canopy.pipeline.run import cells_for_review
    from canopy.pipeline.state import review_entry, save_manifest, write_stage
    from canopy.protocol import load_protocol
    from canopy.report import extraction_table
    from canopy.verify.checks import run_checks
    from canopy.verify.confidence import resolve_cell
    from canopy.verify.vote import vote_groups
    from tests.test_pipeline_offline import PROTOCOL

    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROTOCOL, out / "protocol.yaml")
    protocol = load_protocol(out / "protocol.yaml")
    key = protocol.outcomes[0].key
    paper = "a" * 64
    dataset = DatasetSpec(
        dataset_id="aaaaaaaaaaaa:d1", label="older vs younger",
        group_a=GroupSpec(label="older", n=12, n_evidence="12 older adults"),
        group_b=GroupSpec(label="younger", n=12, n_evidence="12 younger adults"),
        outcomes=[OutcomeSources(outcome_key=key, measure_name="angular error",
                                 higher_is_better=False,
                                 sources=[Source(kind=SourceKind.test_statistic, page=5,
                                                 locator="Results ¶3", role="value")])])
    study = StudyMap(paper_id=paper, citation=Citation(first_author="Synthetic", year=2020),
                     eligible=True, datasets=[dataset])
    cand = Candidate(candidate_id="c1", paper_id=paper, dataset_id=dataset.dataset_id,
                     outcome_key=key, kind="test_statistic", group=None, status="found",
                     source_kind=SourceKind.test_statistic, stat_type="t", stat_value=2.5, df=df,
                     design="independent_t", contrast_kind="groups", direction="a_greater",
                     admissible=True, page=5, quote="t(22) = 2.50, p = .02",
                     extractor_id="stats/opus")
    orient = OrientationVerdict(outcome_key=key, measure_name="angular error",
                                higher_is_better=False, agreed=True, needs_human=False,
                                dataset_id=dataset.dataset_id)
    # the review's N1 construction: an INADMISSIBLE F for a different effect, listed BEFORE the
    # t the row converts from. `checks.best_statistic` skips it; "the first one" does not.
    decoy_cand = Candidate(
        candidate_id="c0", paper_id=paper, dataset_id=dataset.dataset_id, outcome_key=key,
        kind="test_statistic", group=None, status="found",
        source_kind=SourceKind.test_statistic, stat_type="F", stat_value=9.0, df1=1, df2=22,
        design="mixed_main_effect", contrast_kind="unknown", within_factors=["block"],
        direction="a_greater", admissible=False, page=5,
        quote="F(1,22) = 9.0 for the block main effect", extractor_id="stats/opus")
    cell = [decoy_cand, cand] if decoy else [cand]
    flags = run_checks(dataset, key, cell, other_candidates=[], orientation=orient)
    votes = vote_groups(cell, unit_hint="deg")
    verdicts = [resolve_cell(dataset, key, group, cell, vote_result=votes.get(group), verdicts=[],
                             flags=flags, orientation=orient, n_a=12, n_b=12)
                for group in ("A", "B")]
    prepared = prepare_rows([(dataset, key, verdicts[0], verdicts[1])], cell, protocol.stats)
    record = resolve_effect(dataset, protocol.outcome(key), prepared[0].values, protocol.stats)
    record.paper_id = record.cluster_id = paper
    record.citation, record.label = study.citation, dataset.label

    write_stage(out, paper, "map", {"study": study.model_dump(mode="json")})
    write_stage(out, paper, "extract", {"candidates": [c.model_dump(mode="json") for c in cell],
                                        "complete": True,
                                        "cells_extracted": [f"{dataset.dataset_id}/{key}"]})
    write_stage(out, paper, "verify",
                {"verdicts": [v.model_dump(mode="json") for v in verdicts],
                 "extra_candidates": [],
                 "orientation": {f"{key}|angular error": orient.model_dump(mode="json")},
                 "adjudications": [], "source_rank": {}, "reopens": 0})
    write_stage(out, paper, "resolve", {"records": [record.model_dump(mode="json")]})
    manifest = RunManifest(run_id="stat", created_at="2026-08-18T00:00:00+00:00",
                           protocol_path=str(out / "protocol.yaml"), protocol_hash="x",
                           papers=[PaperStatus(paper_id=paper, filename="synthetic.pdf",
                                               status="resolved",
                                               stages={"map": "done", "extract": "done",
                                                       "verify": "done", "resolve": "done"})])
    held = [record] if record.confidence == "needs_human" else []
    queue = [review_entry(v, paper_id=paper, candidates=cell, record=record, primary=[],
                          settings=protocol.stats)
             for v in cells_for_review(verdicts, held, [])]
    manifest.human_review_queue = queue
    save_manifest(out, manifest)
    extraction_table([record], out / "results" / "extraction_table_all", verdicts=verdicts,
                     candidates=cell, primary=[])
    (out / "exclusions.json").write_text("[]")
    (out / "human_review_queue.json").write_text(json.dumps(queue, default=str))
    return paper, key, record


def test_a_row_converted_from_a_printed_statistic_asks_about_the_statistic(tmp_path):
    """Whole-diff H2. Neither cell of such a row has a value and neither ever will — the paper
    prints a t and no group means — so the page asked "where is it, or is it not reported?", whose
    only answer excludes the row. Combined with a rebuild that dropped the statistic, the whole
    `test_statistic` / `p_value` / `reported_d` family could not reach the plot by ANY path.
    """
    from canopy.review.questions import questions_for_run

    out = tmp_path / "run"
    paper, key, record = _statistic_run(out)
    assert record.route == "test_statistic" and record.confidence == "needs_human"
    assert round(record.d, 4) == -1.0206

    questions = questions_for_run(out)
    assert {q["kind"] for q in questions} == {"converted_statistic"}, \
        [q["kind"] for q in questions]
    asked = questions[0]
    # the statistic itself, as extracted — not a paraphrase and not the cell's missing value
    for part in ("t(22) = 2.5", "independent_t", "contrast groups", "-1.021"):
        assert part in asked["prompt"], (part, asked["prompt"])
    assert [o["key"] for o in asked["options"]] == ["accept", "not_usable"]
    assert "no_group_values" in asked["options"][0]["overrules"]
    assert asked["answer_writes"] == "mark_reviewed"


def test_accepting_the_converted_effect_pools_the_row_the_statistic_implies(tmp_path):
    """The other half: the accept answer is a decision the log records, the rebuild carries the
    statistic through `pipeline.rows`, and the row reaches the analysis with the t-derived d."""
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool
    from canopy.review.questions import answer_to_override, questions_for_run

    out = tmp_path / "run"
    paper, key, record = _statistic_run(out)
    for group in ("A", "B"):
        question = next(q for q in questions_for_run(out) if q["group"] == group)
        override = answer_to_override(question, {
            "option": "accept",
            "note": "the Results paragraph names the two age groups this t compares"})
        assert override["kind"] == "mark_reviewed"
        assert "no_group_values" in override["overrules"]
        append_override(out, override)
        apply_overrides_and_repool(out)

    row = json.loads((out / "results" / "extraction_table_all.json").read_text())[0]
    assert row["route"] == "test_statistic"
    assert round(row["es"], 4) == -1.0206
    assert row["confidence"] == "accept_with_note" and row["primary_row"] is True
    assert json.loads((out / "human_review_queue.json").read_text()) == []
    assert questions_for_run(out) == [] or all(q["status"] != "open"
                                               for q in questions_for_run(out))


def test_a_statistic_whose_contrast_is_unestablished_is_not_releasable_at_all(tmp_path):
    """C5/C9 still bind on the human path. The same construction with no printed degrees of
    freedom: nothing says the t is these two groups' comparison rather than a post-hoc from a
    larger model, the resolver refuses the route outright, and there is no accept to give — the
    question is the honest "where is this value" one and the log refuses an answer that claims to
    have accepted a conversion that never happened."""
    from canopy.pipeline.overrides import OverrideRejected, append_override
    from canopy.review.questions import questions_for_run

    out = tmp_path / "run"
    paper, key, record = _statistic_run(out, df=None)
    assert record.route == "not_convertible" and record.d is None
    assert "df_missing" in record.flags

    # …and it is asked for the group values, not "where is the value" (re-review N3): the paper
    # printed a number, the gate refused its provenance, and no `accept` is on offer.
    asked = questions_for_run(out)
    assert {q["kind"] for q in asked} == {"needs_group_values"}, [q["kind"] for q in asked]
    assert all("accept" not in [o["key"] for o in q["options"]] for q in asked)
    with pytest.raises(OverrideRejected):
        append_override(out, {"kind": "mark_reviewed", "paper_id": paper,
                              "dataset_id": "aaaaaaaaaaaa:d1", "outcome_key": key, "group": "A",
                              "overrules": ["no_group_values"],
                              "justification": "I accept the printed t"})


def test_no_statistic_only_row_pools_without_a_person(tmp_path):
    """Ruling H2 (c): this build does NOT pool a converted row on its own. A cell with no group
    value scores 0.0 and is forced to a human, and nothing here changes that — the release is a
    decision on the record, not a score."""
    out = tmp_path / "run"
    paper, key, record = _statistic_run(out)
    verify = json.loads((out / "papers" / "aaaaaaaaaaaa" / "verify.json").read_text())
    for verdict in verify["verdicts"]:
        assert verdict["confidence"] == "needs_human" and verdict["confidence_score"] == 0.0
    assert record.confidence == "needs_human"


# ------------------------------- M1: the rank must not see the locations nobody is allowed to read
def _m1_sources():
    """The review's construction: a rival operationalization's table outscores both readable
    locations, and one of them has no error-bar type of its own."""
    from canopy.models import DispersionType, OutcomeSources, Source, SourceKind

    alternate = Source(kind=SourceKind.table, page=6, locator="Table 3", role="alternate",
                       error_bar_type=DispersionType.SD, analysis_metric="endpoint",
                       values_in_text="15.2 (3.1) and 11.8 (2.9), n = 12 per group")
    printed = Source(kind=SourceKind.text_mean_sd, page=4, locator="Results ¶2", role="value",
                     analysis_metric="endpoint",
                     values_in_text="12.3 and 13.9 degrees, n = 12 per group")
    figure = Source(kind=SourceKind.figure_line, page=5, locator="Figure 2a", figure_id="fig02",
                    role="value", error_bar_type=DispersionType.SE, analysis_metric="endpoint")
    return OutcomeSources(outcome_key="late_adaptation", measure_name="angular error",
                          analysis_metric="endpoint",
                          sources=[alternate, printed, figure]), printed, figure


def _m1_candidates(printed, figure):
    from canopy.models import Candidate, DispersionType, SourceKind

    text = Candidate(candidate_id="text-A", paper_id="p", dataset_id="d1",
                     outcome_key="late_adaptation", group="A", kind="group_stats", status="found",
                     source_kind=SourceKind.text_mean_sd, mean=12.3, n=12, page=printed.page,
                     quote="12.3 degrees", extractor_id="text/opus")
    fig = Candidate(candidate_id="fig-A", paper_id="p", dataset_id="d1",
                    outcome_key="late_adaptation", group="A", kind="group_stats", status="found",
                    source_kind=SourceKind.figure_line, mean=13.9, n=12,
                    dispersion_value=2.0, dispersion_type=DispersionType.SE, page=figure.page,
                    pixel_provenance={"figure_id": "fig02"}, extractor_id="digitize:ensemble")
    return [text, fig]


def test_a_location_nobody_may_read_cannot_evict_the_printed_reading_from_the_vote():
    """Whole-diff M1. `keep_for_vote` holds a candidate back when a HIGHER-scoring source carries
    a dispersion its own source lacks — and the rank was built over the mapper's whole list, so a
    C6 `alternate` table (a rival operationalization, kept for the record and never read) scored
    top and pushed the printed 12.3 out of the vote. The figure's 13.9 then resolved alone.
    """
    from canopy.agents.mapper import readable_sources
    from canopy.agents.source_rank import keep_for_vote, rank_sources

    sources, printed, figure = _m1_sources()
    cell = _m1_candidates(printed, figure)

    whole_list = rank_sources(sources.sources, sources)
    evicted, held = keep_for_vote(cell, whole_list)
    assert whole_list[0].source.role == "alternate", [r.source.role for r in whole_list]
    assert [c.candidate_id for c in evicted] == ["fig-A"] and held, \
        "the construction no longer evicts anything, so the assertion below proves nothing"

    readable = rank_sources(readable_sources(sources.sources), sources)
    kept, held_back = keep_for_vote(cell, readable)
    assert [c.candidate_id for c in kept] == ["text-A", "fig-A"]
    assert held_back == []
    assert all(row.source.role == "value" for row in readable)


def test_the_verify_stage_ranks_the_readable_locations():
    """…and the pipeline calls it that way. The rank decides what the VOTE weighs, so a second
    reading of "may a number be read here" here would put back exactly the defect above."""
    import inspect

    from canopy.pipeline import run as run_module

    body = inspect.getsource(run_module._verify_cell)
    assert "rank_sources(readable_sources(sources.sources), sources)" in body
    assert "rank_sources(sources.sources" not in body


# ------------------------------------- M2: a garbled dissent is still a dissent about the sign
def _garbled(reason: str, higher_is_better: bool) -> Any:
    """A ballot that DECIDED a direction and whose prose trips C12's detector."""
    from canopy.llm.client import degenerate_reply

    run = OrientationRun(higher_is_better=higher_is_better, raw_value_semantics="unknown",
                         direction_stated_in_text="unknown", reason=reason, model=SONNET,
                         quotes=["a quote"])
    assert degenerate_reply(reason), "this reason is not degenerate, so nothing is under test"
    return run


#: three shapes the detector catches, each on a reason that is otherwise a real sentence
_GARBLES = (
    a_real_reason(0) + " The measure is an error \\theta that grows with adaptation.",
    a_real_reason(1) + " The value is the the size of the aftereffect in degrees.",
    a_real_reason(2) + " The scale is defined as {deg of visual angle.",
)


@pytest.mark.parametrize("reason", _GARBLES)
def test_a_dissenting_ballot_c12_calls_a_non_reply_still_stops_the_single_witness(reason):
    """Whole-diff M2. C12's detector reads the JUSTIFICATION, and it is deliberately tuned to
    over-fire so a garbled reply is re-issued rather than trusted. `higher_is_better` is a parsed
    enum, not prose — so a reader that named the OPPOSITE direction disagreed about the sign
    however mangled its sentences are, and row 5 handing the measure to the other reader alone is
    a single model deciding the sign of a pooled effect over a recorded disagreement.
    """
    from canopy.agents.orientation import combine_orientation

    coherent = ballot(model=OPUS, higher_is_better=True)
    verdict = combine_orientation([coherent, _garbled(reason, higher_is_better=False)],
                                  "aftereffect", "aftereffect (deg)")
    assert verdict.higher_is_better is None, verdict.notes
    assert verdict.needs_human is True
    assert "orientation_single_witness" not in verdict.notes
    assert "OPPOSITE" in verdict.notes


def test_a_garbled_reply_that_agreed_or_abstained_still_leaves_the_single_witness_standing():
    """The other side, and the reason this is not simply "never trust row 5": C12 exists because a
    reply really can be debris, and a ballot that named nothing — or named the same direction —
    contradicts nobody. Row 5 is unchanged for those."""
    from canopy.agents.orientation import combine_orientation

    coherent = ballot(model=OPUS, higher_is_better=True)
    for other in (_garbled(_GARBLES[0], higher_is_better=True),
                  OrientationRun(higher_is_better=None, reason="ok ok", model=SONNET)):
        verdict = combine_orientation([coherent, other], "aftereffect", "aftereffect (deg)")
        assert verdict.higher_is_better is True, verdict.notes
        assert "orientation_single_witness" in verdict.notes


# ------------------------ M3: an answer that changes nothing is not an answer (C4's own property)
def test_the_denominator_question_offers_no_answer_that_leaves_the_row_where_it_was():
    """Whole-diff M3. `dispersion_doubt.both_right` recorded a `mark_reviewed` naming
    `implausible_dispersion` — which counts as an ANSWER — and changed nothing, because C9's
    screen is arithmetic and is re-derived from the same numbers. Answer it on both cells and the
    row was still refused with every question on it ticked: a held row with nothing open anywhere,
    which is the exact state C4 exists to remove."""
    from canopy.review.questions import _dispersion_options

    options = _dispersion_options(["implausible_dispersion"])
    assert [o["key"] for o in options] == ["se_not_sd", "within_subject"]
    # each remaining answer changes the analysis: one converts the spread, one removes the cell
    assert options[0]["dispersion_type"] == "SE"
    assert "leaves the analysis" in options[1]["label"]
    assert not any(o.get("clears") and not o.get("dispersion_type") for o in options), \
        "an option that only names what it clears is an answer that changes nothing"


# ------------------------------------------------------------------- the low findings, L1 and L2
def test_a_cell_with_no_value_at_all_publishes_no_margin_and_no_boundary(tmp_path):
    """Whole-diff L2. C11's margin says how close the SCORE came to a boundary, and the M3-margin
    ruling already says it must be empty on a cell the score did not decide. `confidence()`'s very
    first branch — "no value was resolved for this cell" — returned a plain tuple, so
    `forced_human` read False and every such cell in the review queue carried
    `confidence_margin = 0.45, nearest_boundary = accept_with_note`.
    """
    out = tmp_path / "run"
    _statistic_run(out)
    verdicts = json.loads((out / "papers" / "aaaaaaaaaaaa" / "verify.json").read_text())["verdicts"]
    for verdict in verdicts:
        assert verdict["confidence_reasons"] == ["no value was resolved for this cell"]
        assert verdict["confidence_margin"] is None, verdict
        assert verdict["nearest_boundary"] == "", verdict


def test_the_direction_recheck_warns_only_when_the_direction_actually_changed(tmp_path):
    """Whole-diff L1. `_recheck_orientation` returns a FRESH object whenever the dataset matches,
    so the identity test behind this warning fired for every measure of every paper — including
    the ones where the check ran and found nothing wrong."""
    from canopy.pipeline.run import _orientation_state

    same = ballot(model=OPUS, higher_is_better=True)
    other = ballot(model=SONNET, higher_is_better=True, index=1)
    verdict = verdict_for("d1", [same, other])
    assert _orientation_state(verdict) == _orientation_state(verdict.model_copy(deep=True))
    assert _orientation_state(verdict) != _orientation_state(
        verdict.model_copy(update={"higher_is_better": False}))
    assert _orientation_state(None) == ()


def test_the_contradiction_question_quotes_the_reader_that_was_actually_discarded(tmp_path):
    """Whole-diff L3. The question exists because ONE reader's stated direction is contradicted by
    the means — which is also the case in which the verdict's summary `direction_stated_in_text`
    collapses to "unknown", because the two readers stated opposite directions. So the reviewer
    was shown 'The reader said "unknown"' about the one thing they were being asked to arbitrate.
    The discarded ballot's own direction is on the record per reader now, and the prompt reads it.
    """
    from canopy.review.questions import questions_for_run

    out, _ = _offline(tmp_path, router_wrapper=_readers_split_on_the_stated_direction)
    verdict = next(iter(_verify_payload(out)["orientation"].values()))
    assert "orientation_reader_contradicts_values" in verdict["notes"]
    thrown_out = [r for r in verdict["runs"] if r["discarded"]]
    assert len(thrown_out) == 1, [(r["model"], r["discarded"]) for r in verdict["runs"]]
    assert verdict["direction_stated_in_text"] == "unknown"     # the summary really is empty

    # …and the question is reached the way a reviewer reaches it: the discard leaves the direction
    # undecided, so the cell asks `orientation` first; answering that is what brings the
    # contradiction to the top of the same cell's list.
    from canopy.pipeline.overrides import append_override
    from canopy.review.questions import answer_to_override

    direction = next(q for q in questions_for_run(out) if q["kind"] == "orientation")
    append_override(out, answer_to_override(
        direction, {"option": "lower_is_more",
                    "note": "a larger angular error is a worse score on this measure"}))

    asked = [q for q in questions_for_run(out) if q["kind"] == "reader_contradicts_values"]
    assert asked, [q["kind"] for q in questions_for_run(out)]
    said = {"a_greater": "group A came out higher",
            "b_greater": "group B came out higher"}[thrown_out[0]["direction_stated_in_text"]]
    for question in asked:
        assert f"The reader {thrown_out[0]['model']} said {said}" in question["prompt"], \
            question["prompt"]
        assert 'The reader said "unknown"' not in question["prompt"]


def _leave_the_dataset_unsettled(seen):
    def wrapper(mapped):
        study, file_id = mapped
        seen["paper_id"] = study.paper_id
        seen["dataset_id"] = study.datasets[0].dataset_id
        study.open_questions.append(MapQuestion(
            kind="include_dataset", dataset_id=study.datasets[0].dataset_id,
            question="only one mapping agent proposed this dataset"))
        return study, file_id
    return wrapper


def test_a_dataset_a_reviewer_excluded_appears_once_in_the_exclusions_table(tmp_path,
                                                                            monkeypatch):
    """Whole-diff L5. Two writers record the same decision: the run's C7 loop, which reads the
    ANSWERED map and writes `map_adjudication:dataset_rule` with `decider = a human reviewer`, and
    the re-pool's `_apply_exclude`, which writes `human_override`. `_rewrite_exclusions` de-duped
    on `(dataset, outcome, reason)`, so the two reasons kept both rows — the dataset appeared
    twice in the table a reader reads and twice in the PRISMA `exclusion_reasons` counts.
    """
    from canopy.pipeline.overrides import append_override

    seen: dict[str, str] = {}
    out, _ = _offline(tmp_path, map_wrapper=_leave_the_dataset_unsettled(seen),
                      monkeypatch=monkeypatch)
    append_override(out, {
        "kind": "include_dataset", "paper_id": seen["paper_id"],
        "dataset_id": seen["dataset_id"], "decision": "exclude", "actor": "a reviewer",
        "rule": "the protocol's population rule: adults over 60",
        "justification": "these are the same people as the other dataset"})

    resumed, _ = _offline(tmp_path, map_wrapper=_leave_the_dataset_unsettled(seen),
                          monkeypatch=monkeypatch, resume=True)
    assert resumed == out
    rows = json.loads((out / "exclusions.json").read_text())
    mine = [r for r in rows if r.get("dataset_id") == seen["dataset_id"]]
    assert len(mine) == 1, [(r.get("reason"), r.get("decider"), r.get("outcome_key")) for r in mine]
    # …and the row that survives says a person decided it and names the rule they gave — a
    # reviewer who cites the protocol used to be attributed to the map adjudicator, because the
    # only test for "a person decided this" was that they had named NO rule.
    from canopy.agents.mapper import HUMAN_DECIDER

    assert mine[0]["decider"] == HUMAN_DECIDER, mine[0]
    assert "population rule" in json.dumps(mine[0])
    # and the PRISMA counts see it once too
    counts = json.loads((out / "prisma.json").read_text()) if (out / "prisma.json").exists() else {}
    reasons = (counts.get("counts") or counts).get("exclusion_reasons") or {}
    assert sum(reasons.values()) == len(rows), (reasons, len(rows))


def test_an_inclusion_answer_is_not_pending_for_ever_because_a_sibling_cell_is_still_blocked():
    """Whole-diff L4. An answer is "acted on" when every cell it asks for has been extracted. The
    list included cells another OPEN map question still blocks — which no resume may buy — so
    `include_dataset: include` on a dataset whose second outcome has an unanswered `which_measure`
    stayed `pending re-run` after every resume, and the page told the reviewer to run `--resume`
    again for ever.
    """
    from canopy.models import DatasetSpec, GroupSpec, OutcomeSources, StudyMap
    from canopy.pipeline.run import _answer_cells

    dataset = DatasetSpec(dataset_id="d1", group_a=GroupSpec(label="old", n=12),
                          group_b=GroupSpec(label="young", n=12),
                          outcomes=[OutcomeSources(outcome_key="late_adaptation"),
                                    OutcomeSources(outcome_key="aftereffect")])
    study = StudyMap(paper_id=PAPER_ID, eligible=True, datasets=[dataset])
    keys = {"late_adaptation", "aftereffect"}
    answer = {"kind": "include_dataset", "dataset_id": "d1", "decision": "include"}

    assert _answer_cells(answer, study, keys) == ["d1/late_adaptation", "d1/aftereffect"]
    blocked = {("d1", "aftereffect"): "which_measure needs human"}
    assert _answer_cells(answer, study, keys, blocked) == ["d1/late_adaptation"]
    # …and the answer is still not consumed while the cell it CAN un-block is unread
    assert _answer_cells(answer, study, keys, {("d1", "late_adaptation"): "blocked",
                                               ("d1", "aftereffect"): "blocked"}) == []


def test_a_reader_is_told_which_of_the_maps_locations_are_not_places_the_value_is_read():
    """Whole-diff L6. `measure_prompt` listed the outcome's first six locations with no role, so a
    C6 `alternate` operationalization's table and a `pooled`-sample sentence were offered to the
    orientation and verifier readers exactly like the outcome's own figure. The rule is the
    mapper's own, so the prompt cannot disagree with the extractor about what may be read — and a
    map whose locations are all this contrast's own `value` prints exactly what it printed before.
    """
    from canopy.agents.verify_common import measure_prompt
    from canopy.models import OutcomeSources, Source, SourceKind

    plain = OutcomeSources(outcome_key="late_adaptation", measure_name="angular error",
                           sources=[Source(kind=SourceKind.figure_line, page=5,
                                           locator="Figure 2a")])
    assert "NOT A PLACE" not in measure_prompt(plain)

    mixed = OutcomeSources(
        outcome_key="late_adaptation", measure_name="angular error",
        sources=[Source(kind=SourceKind.figure_line, page=5, locator="Figure 2a"),
                 Source(kind=SourceKind.table, page=6, locator="Table 3", role="alternate"),
                 Source(kind=SourceKind.text_mean_sd, page=4, locator="Results",
                        sample="pooled")])
    text = measure_prompt(mixed)
    assert text.count("NOT A PLACE THIS MEASURE IS READ") == 2, text
    assert "Figure 2a" in text and "NOT A PLACE" not in text.split("Figure 2a")[1].split("\n")[0]


def test_an_answer_is_validated_against_the_stage_file_not_against_the_last_re_pools_output(
        tmp_path):
    """Whole-diff L7. `recorded_holds` names the row's own refusals so an answer may address them,
    and it read `results/extraction_table_all.json` — an artefact the re-pool REWRITES. So whether
    an answer was admissible depended on the previous re-pool's output, and a log replayed onto
    fresh stage files with no table at all fell back to the record's own `carried` list alone.
    The resolve stage file is the run's record and no re-pool touches it.
    """
    from canopy.pipeline.overrides import _row_of, recorded_holds

    out = tmp_path / "run"
    paper, key, record = _statistic_run(out)
    ref = {"paper_id": paper, "dataset_id": "aaaaaaaaaaaa:d1", "outcome_key": key, "group": "A"}
    assert _row_of(out, ref)["route"] == "test_statistic"
    assert "no_group_values" in recorded_holds(out, ref)

    (out / "results" / "extraction_table_all.json").unlink()
    assert _row_of(out, ref)["route"] == "test_statistic", "the table was the only reference"
    assert "no_group_values" in recorded_holds(out, ref)


def test_the_other_two_answers_to_a_converted_row_do_what_they_say(tmp_path):
    """C4's per-option property on the new kind: every answer changes the analysis. Typing the
    group values the paper prints after all moves the row onto the group route (and the cell asks
    the ordinary "is it right?" next, not the conversion question again); "not usable" removes it.
    """
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool
    from canopy.review.questions import answer_to_override, questions_for_run

    typed = tmp_path / "typed"
    _statistic_run(typed)
    for group, mean in (("A", 15.2), ("B", 11.8)):
        question = next(q for q in questions_for_run(typed) if q["group"] == group)
        override = answer_to_override(question, {
            "mean": mean, "dispersion_value": 3.0, "dispersion_type": "SD", "n": 12,
            "note": "Table 2 prints both groups after all"})
        assert override["kind"] == "value"
        append_override(typed, override)
        apply_overrides_and_repool(typed)
    assert {q["kind"] for q in questions_for_run(typed)} == {"confirm_value"}
    row = json.loads((typed / "results" / "extraction_table_all.json").read_text())[0]
    assert row["route"] == "text_mean_sd" and round(row["es"], 4) == -1.1333

    gone = tmp_path / "gone"
    _statistic_run(gone)
    question = questions_for_run(gone)[0]
    override = answer_to_override(question, {"option": "not_usable",
                                             "note": "the t is the block × age interaction"})
    assert override["kind"] == "exclude_dataset"
    append_override(gone, override)
    apply_overrides_and_repool(gone)
    assert json.loads((gone / "results" / "extraction_table_all.json").read_text()) == []
    assert json.loads((gone / "exclusions.json").read_text()), "no exclusion was recorded"


# ============================== the re-review's findings (N1, N2, N3)
def test_the_question_names_the_statistic_the_row_actually_converted_from(tmp_path):
    """Re-review N1. A paper prints several statistics near one outcome and the row converts from
    exactly one of them — `checks.best_statistic`, which skips the inadmissible ones and ranks
    t before F before p. The question took the FIRST `test_statistic` in the stage file instead,
    so an inadmissible `F(1,22) = 9` for a block main effect, listed above the `t(22) = 2.50` the
    row used, was what the reviewer was asked to accept — beside a d computed from the t.
    """
    from canopy.review.questions import questions_for_run

    out = tmp_path / "run"
    paper, key, record = _statistic_run(out, decoy=True)
    assert record.route == "test_statistic" and round(record.d, 4) == -1.0206

    prompt = questions_for_run(out)[0]["prompt"]
    assert "t(22) = 2.5" in prompt and "independent_t" in prompt, prompt
    assert "F(1, 22)" not in prompt and "mixed_main_effect" not in prompt, prompt


def test_the_page_and_the_resolver_pick_the_statistic_with_the_same_function():
    """…and they must not be able to drift: one selection rule, `pipeline.rows`, used by the row
    the resolver builds and by the question the page asks about it."""
    from canopy.models import Candidate, SourceKind
    from canopy.pipeline.rows import converting_candidate, statistic_values

    def stat(cid, **kw):
        return Candidate(candidate_id=cid, paper_id="p", dataset_id="d1", outcome_key="k",
                         kind="test_statistic", status="found",
                         source_kind=SourceKind.test_statistic, **kw)

    inadmissible = stat("c0", stat_type="F", stat_value=9.0, df1=1, df2=22, admissible=False)
    usable = stat("c1", stat_type="t", stat_value=2.5, df=22, admissible=True)
    cell = [inadmissible, usable]
    assert converting_candidate(cell).candidate_id == "c1"
    assert statistic_values(cell).value == converting_candidate(cell).stat_value
    # …and an F is chosen when it is the only admissible one
    only_f = [inadmissible, stat("c2", stat_type="F", stat_value=4.0, df1=1, df2=22,
                                 admissible=True)]
    assert converting_candidate(only_f).candidate_id == "c2"


def test_a_cell_whose_statistic_the_gate_refused_is_asked_for_the_group_values(tmp_path):
    """Re-review N3. "No usable value was found — where is it, or is it genuinely not reported?"
    is a true question only when the paper printed NOTHING. A cell whose paper printed a t that
    the conversion gate then refused has a number in it; what it lacks is the provenance of that
    statistic to these two groups, and the only thing that replaces it is both groups' own
    numbers. It is asked for those, with the statistic and the resolver's own refusal shown.
    """
    from canopy.review.questions import questions_for_run

    out = tmp_path / "run"
    paper, key, record = _statistic_run(out, df=None)
    assert record.route == "not_convertible"

    asked = questions_for_run(out)
    assert {q["kind"] for q in asked} == {"needs_group_values"}, [q["kind"] for q in asked]
    prompt = asked[0]["prompt"]
    assert "t = 2.5" in prompt and "NO degrees of freedom" in prompt, prompt
    assert "The resolver refused it" in prompt, prompt
    assert asked[0]["answer_writes"] == "value"          # the group values, typed


def test_a_cell_with_nothing_printed_at_all_is_still_asked_where_the_value_is(tmp_path):
    """The control for N3, and the reason it is not "never ask `no_value`": a cell whose paper
    prints no statistic and no group values has exactly the question it always had."""
    from canopy.pipeline.state import read_stage, write_stage
    from canopy.review.questions import questions_for_run

    out = tmp_path / "run"
    paper, key, record = _statistic_run(out, df=None)
    payload = read_stage(out, paper, "extract")
    write_stage(out, paper, "extract", {**payload, "candidates": []})   # nothing was printed
    assert {q["kind"] for q in questions_for_run(out)} == {"no_value"}


def test_who_settled_a_datasets_inclusion_is_read_from_the_log_not_from_the_rule(tmp_path,
                                                                                  monkeypatch):
    """Re-review N2. The exclusions table's `decider` column was derived by testing whether the
    `exclusion_rule` was the "no rule named" placeholder — so a reviewer who CITED the protocol
    was recorded as the map adjudicator, and their exclusion then appeared twice in the table.

    It is read from the review LOG instead, which is the record of who decided. NOT from a new
    field on `DatasetSpec`: the study map is dumped verbatim into the map adjudicator's prompt, so
    a field there changes that prompt and re-buys the call for every paper in the run — measured,
    not assumed (adding one turned the recorded map-adjudicate fixture into a `MissingFixture`).
    """
    import inspect

    from canopy.agents.mapper import HUMAN_DECIDER, MAP_ADJUDICATOR
    from canopy.pipeline import run as run_module
    from canopy.pipeline.overrides import append_override

    seen: dict[str, str] = {}
    out, _ = _offline(tmp_path, map_wrapper=_leave_the_dataset_unsettled(seen),
                      monkeypatch=monkeypatch)
    append_override(out, {
        "kind": "include_dataset", "paper_id": seen["paper_id"],
        "dataset_id": seen["dataset_id"], "decision": "exclude", "actor": "a reviewer",
        "rule": "the protocol's population rule: adults over 60",
        "justification": "these are the same people as the other dataset"})
    _offline(tmp_path, map_wrapper=_leave_the_dataset_unsettled(seen), monkeypatch=monkeypatch,
             resume=True)

    row = next(r for r in json.loads((out / "exclusions.json").read_text())
               if r.get("dataset_id") == seen["dataset_id"])
    assert row["decider"] == HUMAN_DECIDER, row
    assert "population rule" in row["detail"], row     # the rule they cited, still recorded

    body = inspect.getsource(run_module._run_paper)
    assert "dataset.dataset_id in by_hand" in body      # the log is what it reads
    assert "excluded by" not in body                    # …not the prose of a note
    assert "decided_by" not in body                     # and no model field was added for it
    assert MAP_ADJUDICATOR == "map-adjudicator"


def test_the_study_map_a_prompt_carries_is_the_one_the_fixtures_were_recorded_for():
    """The measurement behind that ruling, kept as a test: `DatasetSpec` is dumped verbatim into
    the map adjudicator's prompt, so its field list is part of a prompt hash. A field added for
    the review layer's bookkeeping re-buys one adjudication per paper — money, on a build whose
    fixtures were re-recorded hours earlier."""
    import inspect

    from canopy.agents import mapper
    from canopy.models import DatasetSpec

    body = inspect.getsource(mapper.map_study)
    assert "MAP_A=json.dumps(study.model_dump(mode=\"json\")" in body
    assert "decided_by" not in set(DatasetSpec.model_fields)
