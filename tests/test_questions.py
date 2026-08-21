"""The questions page: every held cell as a thing to decide, with its screenshot and its answers.

Two halves: the pure module over real run records (`runs/proof` when present, otherwise the
fixtures), and the server round-trip — ask, answer, see the override and the re-pool.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from canopy.review.questions import (QUESTION_KINDS, answer_to_override,
                                     answers_to_overrides, questions_for_run,
                                     write_questions)

REPO = Path(__file__).resolve().parents[1]
PROOF = REPO / "runs" / "proof"


# ------------------------------------------------------------------ the module, on real records
@pytest.mark.skipif(not (PROOF / "manifest.json").exists(), reason="runs/proof is not on this machine")
def test_every_held_cell_of_a_real_run_becomes_a_question_with_a_picture_and_a_reason():
    qs = questions_for_run(PROOF, fold=False)       # the per-cell questions the page is built from
    queue = json.loads((PROOF / "human_review_queue.json").read_text())
    assert len(qs) == len(queue), "one question per held cell — nothing dropped, nothing invented"
    for q in qs:
        assert q["kind"] in QUESTION_KINDS
        assert "?" in q["prompt"] or q["kind"] == "other"     # it asks something
        assert q["why"], "a question always says why the tool could not decide"
        assert q["free_text"] is True
        # a figure cell shows the picture the tool read — the overlay, so the marks are visible
        if q["route"] == "figure":
            assert q["image"].get("path"), q["id"]
            assert (PROOF / q["image"]["path"]).exists()
    # the two-value question offers the candidates' own numbers, most-backed first, never invented
    which = [q for q in qs if q["kind"] == "which_value"]
    for q in which:
        assert len(q["options"]) >= 2
        assert q["options"][0]["n_backers"] >= q["options"][-1]["n_backers"]
        assert all(o["mean"] is not None for o in q["options"])


@pytest.mark.skipif(not (PROOF / "manifest.json").exists(), reason="runs/proof is not on this machine")
def test_the_questions_file_is_written_beside_the_queue(tmp_path):
    import shutil

    clone = tmp_path / "proof"
    shutil.copytree(PROOF, clone, ignore=shutil.ignore_patterns("cache", "*.png"))
    paths = write_questions(clone)
    assert paths["json"].exists() and paths["md"].exists()
    md = paths["md"].read_text(encoding="utf-8")
    assert "# Questions for the reviewer" in md
    assert "why the tool could not decide" in md
    assert json.loads(paths["json"].read_text())[0]["number"] == 1


# ------------------------------------------------------------------ answers → overrides
def _q(kind: str, **over: Any) -> dict[str, Any]:
    q = {"number": 3, "kind": kind, "paper_id": "p", "dataset_id": "p:d1",
         "outcome_key": "late_adaptation", "group": "A", "unit": "deg",
         "options": [{"key": "v1", "label": "31.5 deg", "mean": 31.5, "dispersion_value": 11.0,
                      "dispersion_type": "SD", "n": 12},
                     {"key": "v2", "label": "12.4 deg", "mean": 12.4, "dispersion_value": 12.7,
                      "dispersion_type": "SD", "n": 12}]}
    q.update(over)
    return q


def test_choosing_a_candidate_value_becomes_a_value_override_that_names_the_question():
    payload = answer_to_override(_q("which_value"), {"option": "v2", "note": "the open square"})
    assert payload["kind"] == "value" and payload["group"] == "A"
    assert payload["mean"] == 12.4 and payload["dispersion_value"] == 12.7
    assert payload["dispersion_type"] == "SD" and payload["n"] == 12
    assert "question #3" in payload["justification"] and "the open square" in payload["justification"]


def test_a_typed_value_is_a_value_override_and_an_empty_answer_is_rejected_by_the_log():
    from canopy.pipeline.overrides import OverrideRejected, _validate

    typed = answer_to_override(_q("which_value"), {"mean": "17.6", "dispersion_value": "3.0",
                                                    "dispersion_type": "SE", "n": "9",
                                                    "note": "Table 2, row 'older', p. 5"})
    assert typed["kind"] == "value" and typed["mean"] == "17.6"
    _validate(typed)                                          # the log accepts it
    with pytest.raises(OverrideRejected):
        _validate(answer_to_override(_q("which_value", options=[]), {}) | {"kind": "value"})


def test_a_typed_number_with_no_note_is_refused_because_the_note_is_its_only_provenance():
    """M6. A picked option carries the candidate's backers, its page and its quote; a typed number
    carries nothing. Before this, "answered question #2 (which_axis) — typed value" was a complete
    justification and the number entered the pooled analysis with no reviewer words at all."""
    from canopy.pipeline.overrides import OverrideRejected

    with pytest.raises(OverrideRejected) as refused:
        answer_to_override(_q("which_value"), {"mean": "17.6"})
    assert "note" in str(refused.value)
    with pytest.raises(OverrideRejected):
        answer_to_override(_q("which_value"), {"n": "12", "note": "   "})
    # the option path is untouched: what it carries IS the provenance
    assert answer_to_override(_q("which_value"), {"option": "v1"})["kind"] == "value"


def test_the_other_kinds_map_onto_the_right_override():
    assert answer_to_override(_q("error_bar_type", options=[
        {"key": "SE", "label": "standard error", "dispersion_type": "SE"}]),
        {"option": "SE"})["dispersion_type"] == "SE"
    mapping = _q("group_mapping", options=[{"key": "as_mapped", "label": "the mapping is right"},
                                           {"key": "swapped", "label": "the two groups are swapped"}])
    assert answer_to_override(mapping, {"option": "swapped"})["kind"] == "exclude_dataset"
    assert answer_to_override(mapping, {"option": "as_mapped"})["confidence"] == "accept_with_note"
    assert answer_to_override(_q("no_value", options=[]), {"hint": "Table 2, p. 5"})["kind"] == "re_extract"
    assert answer_to_override(_q("no_value", options=[]), {})["confidence"] == "needs_human"
    assert answer_to_override(_q("which_value"), {"exclude": True})["kind"] == "exclude_dataset"
    assert answer_to_override(_q("confirm_value"), {"option": "yes"})["confidence"] == "accept_with_note"


# ------------------------------------------------------------------ C4: a question you can answer
RERUN = REPO / "runs" / "rerun-fixed"
_HAS_RERUN = (RERUN / "manifest.json").exists()


def _clone_run(tmp_path: Path, name: str = "run") -> Path:
    import shutil

    clone = tmp_path / name
    shutil.copytree(RERUN, clone,
                    ignore=shutil.ignore_patterns("cache", "thumbnails", "*.png", "*.svg"))
    (clone / "overrides.jsonl").unlink(missing_ok=True)
    return clone


def _cell_state(run: Path) -> dict[tuple, Any]:
    """What a re-pool WROTE about every cell: the rows, the queue, the exclusions.

    Stage files are never rewritten by a re-pool, so this is the only honest place to read the
    effect of an answer — and it is where a reader of the review would look.
    """
    state: dict[tuple, Any] = {}
    for row in json.loads((run / "results" / "extraction_table_all.json").read_text()):
        state[("row", row.get("dataset_id"), row.get("outcome_key"))] = {
            key: row.get(key) for key in ("confidence", "es", "var", "se", "route", "n_a", "n_b",
                                          "higher_is_better", "mean_a", "mean_b", "dispersion_a",
                                          "dispersion_b", "dispersion_type_a", "dispersion_type_b",
                                          "primary_row")}
    for entry in json.loads((run / "human_review_queue.json").read_text()):
        state[("held", entry.get("dataset_id"), entry.get("outcome_key"),
               entry.get("group"))] = entry.get("reason")
    for gone in json.loads((run / "exclusions.json").read_text()):
        state[("gone", gone.get("dataset_id"), gone.get("outcome_key"),
               gone.get("reason"))] = True
    return state


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_every_answer_to_every_real_question_is_accepted_and_changes_its_cell(tmp_path):
    """C4's invariant, on the eleven real questions of `runs/rerun-fixed`.

    For every question the tool emits and *every* answer it offers: the override the answer
    becomes is one the log accepts, and re-pooling under it changes the cell — its numbers, its
    bucket, or its presence in the analysis. An answer that leaves the cell exactly as it was is
    not an answer; before this, 7 of these 11 questions had no answer that could change anything.
    """
    from canopy.pipeline.overrides import (OVERRIDES_FILE, _validate,
                                           apply_overrides_and_repool)

    baseline = _clone_run(tmp_path, "baseline")
    apply_overrides_and_repool(baseline)
    before = _cell_state(baseline)
    questions = questions_for_run(baseline, fold=False)
    assert len(questions) == 11, "the recorded run holds eleven cells"

    checked = 0
    for question in questions:
        assert question["options"], f"#{question['number']} ({question['kind']}) offers no answer"
        for option in question["options"]:
            run = _clone_run(tmp_path, f"q{question['number']}-{option['key']}")
            payload = answer_to_override(question, {"option": option["key"],
                                                    "note": "read off the figure"})
            record = _validate(payload)          # (1) the log accepts it — no OverrideRejected
            (run / OVERRIDES_FILE).write_text(
                json.dumps({**record, "seq": 1, "at": "2026-08-17T00:00:00+00:00",
                            "actor": "test"}) + "\n", encoding="utf-8")
            summary = apply_overrides_and_repool(run)
            assert summary["applied"] == 1 and not summary["pending"], \
                f"#{question['number']}/{option['key']} was recorded but not applied: {summary}"
            after = _cell_state(run)
            moved = [key for key in set(before) | set(after)
                     if before.get(key) != after.get(key)
                     and key[1] == question["dataset_id"] and key[2] == question["outcome_key"]]
            assert moved, (f"#{question['number']} ({question['kind']}) answered "
                           f"{option['key']!r} left its own cell untouched")
            checked += 1
    assert checked >= 11


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_the_question_is_chosen_by_what_is_holding_the_cell_not_by_a_fixed_list(tmp_path):
    """Bock's aftereffect is held because its direction is null, and that is what it is asked.

    It also carries `series_marker_mismatch`, which is what the old fixed-order scan matched: the
    cell was asked `which_series`, and answering that correctly left `higher_is_better` null and
    the cell exactly where it was.
    """
    run = _clone_run(tmp_path)
    by_cell = {(q["dataset_id"], q["outcome_key"], q["group"]): q
               for q in questions_for_run(run, fold=False)}
    for group in ("A", "B"):
        question = by_cell[("b511dbb76fa6:d1", "aftereffect", group)]
        assert question["kind"] == "orientation", question["kind"]
        assert question["answer_writes"] == "orientation"
        assert "series" not in question["prompt"]
        assert [o["key"] for o in question["options"]] == ["higher_is_more", "lower_is_more"]
        assert "Angular pointing error" in question["prompt"]      # the measure, in the map's words
    # the cell whose blocker really is the spread's type still asks about the spread's type
    assert by_cell[("b511dbb76fa6:d2", "late_adaptation", "B")]["kind"] == "error_bar_type"


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_an_orientation_answer_signs_bocks_aftereffect_and_does_not_wave_through_its_other_blocker(
        tmp_path):
    """The answer settles the direction and nothing else.

    `higher_is_better = False` turns a row the resolver refused (`not_convertible`,
    `orientation_unresolved`) into a signed effect at the published sign; and because group A of
    the same cell is *also* held — for a series identity the answer says nothing about — the row
    stays in the review queue rather than pooling on one settled blocker.
    """
    from canopy.pipeline.overrides import (OVERRIDES_FILE, _validate,
                                           apply_overrides_and_repool)
    from canopy.stats.effect_sizes import smd_from_means

    run = _clone_run(tmp_path)
    question = next(q for q in questions_for_run(run, fold=False)
                    if q["dataset_id"] == "b511dbb76fa6:d1" and q["outcome_key"] == "aftereffect"
                    and q["group"] == "A")
    record = _validate(answer_to_override(question, {
        "option": "lower_is_more", "note": "aftereffect is an error measure: less is more shift"}))
    assert record["kind"] == "orientation" and record["higher_is_better"] is False
    assert record["group"] is None, "a direction is a property of the measure, not of one cell"
    assert record["measure_name"].startswith("Angular pointing error")
    assert record["quote"], "the reviewer's own words are on the record with the direction"

    (run / OVERRIDES_FILE).write_text(
        json.dumps({**record, "seq": 1, "at": "2026-08-17T00:00:00+00:00", "actor": "t"}) + "\n",
        encoding="utf-8")
    apply_overrides_and_repool(run)
    row = next(r for r in json.loads((run / "results" / "extraction_table_all.json").read_text())
               if r["dataset_id"] == "b511dbb76fa6:d1" and r["outcome_key"] == "aftereffect")
    assert row["higher_is_better"] is False and row["orientation_applied"] is True
    assert row["route"] != "not_convertible", "the resolver can sign it now"
    expected = smd_from_means(-22.4, 8.3, 12, -27.0, 5.5, 12, higher_is_better=False)
    assert expected.d == pytest.approx(-0.6534, abs=1e-4)      # DECISION-v2 §C's number
    assert row["es"] == pytest.approx(expected.d, abs=1e-9)
    assert "so d = -0.65335" in row["conversion_chain"]
    # the other direction would have printed the published sign backwards, which is why this is
    # a question and not an inference
    assert smd_from_means(-22.4, 8.3, 12, -27.0, 5.5, 12,
                          higher_is_better=True).d == pytest.approx(+0.6534, abs=1e-4)
    # the second blocker is untouched: group A is still held, so the row does not pool
    still_held = {(e["dataset_id"], e["outcome_key"], e["group"])
                  for e in json.loads((run / "human_review_queue.json").read_text())}
    assert ("b511dbb76fa6:d1", "aftereffect", "A") in still_held
    assert row["confidence"] == "needs_human"
    # and no other measure of the same paper was re-signed by it
    other = next(r for r in json.loads((run / "results" / "extraction_table_all.json").read_text())
                 if r["dataset_id"] == "b511dbb76fa6:d2")
    assert other["higher_is_better"] is False and "human override" not in (other["notes"] or "")


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_after_the_direction_is_answered_the_cell_asks_about_what_is_still_holding_it(tmp_path):
    """An answer retires the blocker it names, and the next one becomes the question.

    Bock's aftereffect is held for two reasons. Answering the direction pools group B out of the
    queue and leaves group A asking about the series identity — an open question about its real
    blocker, not an "answered" tick on a cell nothing has settled.
    """
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool

    run = _clone_run(tmp_path)
    before = questions_for_run(run, fold=False)
    assert sum(1 for q in before if not q["answered"]) == 11
    question = next(q for q in before if q["kind"] == "orientation" and q["group"] == "A")
    append_override(run, answer_to_override(question, {"option": "lower_is_more",
                                                       "note": "an error measure"}))
    apply_overrides_and_repool(run)

    after = {(q["dataset_id"], q["outcome_key"], q["group"]): q
             for q in questions_for_run(run, fold=False)}
    assert ("b511dbb76fa6:d1", "aftereffect", "B") not in after, "group B is settled and pooled"
    still = after[("b511dbb76fa6:d1", "aftereffect", "A")]
    assert still["kind"] == "which_series", "the cell now asks about its remaining blocker"
    assert still["answered"] is False, "a direction does not tick off a series identity"
    assert any(o["key"] == "as_read" and o.get("mean") is not None for o in still["options"])


# ------------------------------------------------------------------ the rules, without a run
def test_the_kind_follows_the_finding_that_withholds_the_cell_not_the_order_of_the_list():
    """`calibration_disputed` sits above `series_transposed` in `_FLAG_TO_KIND`; the cell is held
    by the transposition. The question must be about the transposition."""
    from canopy.review.questions import _holding_codes, _kind

    verdict = {"higher_is_better": False, "flags": [
        {"code": "calibration_disputed", "severity": "warn"},     # costs score, cannot withhold
        {"code": "series_transposed", "severity": "warn"}]}       # CONTRADICTING: withholds
    flags = [f["code"] for f in verdict["flags"]]
    valued = [{"mean": 1.0, "candidate_id": "c1"}]
    assert _kind(verdict, flags, valued, _holding_codes(verdict)) == "which_series"
    # and with the same two codes, when neither withholds, the list order stands
    plain = {"higher_is_better": False, "flags": [
        {"code": "calibration_disputed", "severity": "warn"},
        {"code": "figure_error_bar_unknown", "severity": "warn"}]}
    assert _kind(plain, [f["code"] for f in plain["flags"]], valued,
                 _holding_codes(plain)) == "which_axis"


def test_a_null_direction_outranks_every_capping_flag_because_nothing_else_can_sign_the_cell():
    from canopy.review.questions import _holding_codes, _kind

    verdict = {"higher_is_better": None, "flags": [
        {"code": "series_marker_mismatch", "severity": "warn"},
        {"code": "orientation_unknown", "severity": "warn"}]}
    flags = [f["code"] for f in verdict["flags"]]
    assert _kind(verdict, flags, [{"mean": 1.0, "candidate_id": "c"}],
                 _holding_codes(verdict)) == "orientation"
    # C3's own decisions are not re-asked: a single-witness direction is set, so it is not null
    settled = {"higher_is_better": True, "flags": [
        {"code": "orientation_single_witness", "severity": "warn"},
        {"code": "series_marker_mismatch", "severity": "warn"}]}
    assert _kind(settled, [f["code"] for f in settled["flags"]],
                 [{"mean": 1.0, "candidate_id": "c"}], _holding_codes(settled)) == "which_series"


def test_no_answer_to_an_axis_or_series_question_is_a_note_that_changes_nothing():
    """Every option of these two kinds writes a value, an exclusion, or a re-extraction."""
    axis = _q("which_axis", options=[
        {"key": "v1", "label": "27.7 deg — read against the left ladder", "mean": 27.7,
         "dispersion_value": 3.0, "dispersion_type": "SD", "n": 20},
        {"key": "v2", "label": "51.1 deg — read against the upper panel", "mean": 51.1,
         "dispersion_value": 5.6, "dispersion_type": "SE", "n": 20}])
    for key, mean in (("v1", 27.7), ("v2", 51.1)):
        payload = answer_to_override(axis, {"option": key})
        assert payload["kind"] == "value" and payload["mean"] == mean
        assert payload.get("confidence") != "needs_human"
    series = _q("which_series", options=[
        {"key": "as_read", "label": "this series is this group — -22.4 deg is right",
         "mean": -22.4, "dispersion_value": 8.3, "dispersion_type": "SD", "n": 12},
        {"key": "other_series", "label": "the value belongs to the other group"}])
    assert answer_to_override(series, {"option": "as_read"})["kind"] == "value"
    assert answer_to_override(series, {"option": "other_series"})["kind"] == "exclude_dataset"


def test_the_log_accepts_an_error_bar_answer_that_only_names_the_spreads_type():
    """The one question in the run carrying an `impact` produced an override the log refused."""
    from canopy.pipeline.overrides import OverrideRejected, _validate

    payload = answer_to_override(
        _q("error_bar_type", options=[{"key": "SE", "label": "standard error",
                                       "dispersion_type": "SE"}]), {"option": "SE"})
    record = _validate(payload)
    assert record["kind"] == "value" and record["dispersion_type"] == "SE"
    assert record["mean"] is None and record["dispersion_value"] is None and record["n"] is None
    with pytest.raises(OverrideRejected):        # a value override still has to say *something*
        _validate({**payload, "dispersion_type": ""})


def test_an_orientation_override_is_scoped_to_the_measure_and_needs_a_direction():
    from canopy.pipeline.overrides import KINDS, OverrideRejected, _validate

    assert "orientation" in KINDS
    question = _q("orientation", group="A", measure_name="Tracking RMSE",
                  options=[{"key": "higher_is_more", "label": "larger is more",
                            "higher_is_better": True},
                           {"key": "lower_is_more", "label": "smaller is more",
                            "higher_is_better": False}])
    record = _validate(answer_to_override(question, {"option": "lower_is_more", "note": "“RMSE”"}))
    assert record["kind"] == "orientation" and record["higher_is_better"] is False
    assert record["group"] is None and record["measure_name"] == "Tracking RMSE"
    with pytest.raises(OverrideRejected):
        _validate({**record, "higher_is_better": None})
    with pytest.raises(OverrideRejected):
        _validate({**record, "paper_id": ""})
    # an answer with no direction at all is not recorded as one
    assert answer_to_override(question, {})["kind"] == "mark_reviewed"


def test_a_direction_settles_a_clean_cell_and_leaves_a_doubly_blocked_one_held():
    """V2's acceptance test, on the function that decides it.

    One blocker (a null direction) → `accept_with_note` and a signed effect. A second blocker of
    any kind — a low score, an error, a contradiction, a refuting verifier, an adjudication —
    → still `needs_human`, because an override clears the blocker it names and no other.
    """
    from canopy.models import CheckFlag, Verdict
    from canopy.pipeline.overrides import _bucket_after_orientation

    def cell(**over: Any) -> Verdict:
        base = dict(dataset_id="d", outcome_key="o", group="A", agreement="agree",
                    verifier_verdict="confirmed", mean=12.0, dispersion_value=2.0, n=20,
                    confidence="needs_human", confidence_score=0.62, needs_human=True)
        return Verdict.model_validate({**base, **over})

    assert _bucket_after_orientation(cell()) == "accept_with_note"
    assert _bucket_after_orientation(cell(confidence_score=0.44)) == "needs_human"
    assert _bucket_after_orientation(cell(mean=None)) == "needs_human"
    assert _bucket_after_orientation(cell(agreement="disagree")) == "needs_human"
    assert _bucket_after_orientation(cell(verifier_verdict="refuted")) == "needs_human"
    assert _bucket_after_orientation(cell(adjudicated=True)) == "needs_human"
    assert _bucket_after_orientation(
        cell(flags=[CheckFlag(code="quote_not_grounded", severity="error")])) == "needs_human"
    assert _bucket_after_orientation(
        cell(flags=[CheckFlag(code="series_transposed", severity="warn")])) == "needs_human"
    # a cap, not a promotion: an already-accepted cell of the same measure is brought into view
    assert _bucket_after_orientation(
        cell(confidence="auto_accept", confidence_score=0.88, needs_human=False)) \
        == "accept_with_note"


# ------------------------------------------------- C4's two run-level acceptance tests, on the run
#: the finding a question of each kind claims to answer, as the record itself words it. A question
#: whose kind is not justified by anything on its own cell is a question about something else.
_JUSTIFIED_BY: dict[str, tuple[str, ...]] = {
    "which_axis": ("axis_conflict", "calibration_disputed", "calibration_refuted"),
    "which_series": ("series_transposed", "series_identity_conflict", "series_marker_mismatch"),
    "error_bar_type": ("dispersion_type_from_legend", "figure_error_bar_unknown",
                       "dispersion_type_conflict"),
    "group_mapping": ("group_label_swapped", "multi_group_closest_to_definition"),
    "verifier_refuted": ("refuted",),
    "no_value": ("no value was resolved", "not_convertible"),
    "which_value": ("no value was resolved", "disagree", "only one independent route",
                    "the readers"),
    "confirm_value": ("only one independent route", "adjudicat"),
    # asked of a cell whose degrees of freedom cannot be reconciled, and of one whose ROW converts
    # to nothing — there the answer that changes the row is both groups' own statistics, never a
    # confirmation of a number the conversion already refused.
    "needs_group_values": ("df_missing", "df_shortfall_unexplained", "test_stat_missing_df",
                           "no value was resolved", "not_convertible"),
    "quote_not_found": ("quote_not_grounded",),
    "number_unusable": ("df_missing", "df_shortfall_unexplained", "test_stat_missing_df",
                        "n_not_integer", "n_too_small", "sd_nonpositive",
                        "implausible_dispersion"),
    "reader_contradicts_values": ("orientation_reader_contradicts_values",),
}


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_every_held_cell_asks_a_question_whose_answer_writes_the_field_its_reason_names(tmp_path):
    """C4's run-level invariant, on the real queue.

    Every `needs_human` row has exactly one question; the question's kind is justified by the
    finding that row's own reason names (or, for `orientation`, by a direction that is null); and
    what its answer writes is a field of the cell — never a note saying a human looked.
    """
    run = _clone_run(tmp_path)
    queue = json.loads((run / "human_review_queue.json").read_text())
    questions = questions_for_run(run, fold=False)     # per CELL: the path the cards are built from
    assert len(questions) == len(queue), "one question per held cell — none dropped, none invented"
    by_cell = {(q["dataset_id"], q["outcome_key"], q["group"]): q for q in questions}
    for entry in queue:
        key = (entry["dataset_id"], entry["outcome_key"], entry["group"])
        question = by_cell.get(key)
        assert question is not None, f"{key} is held and asks nothing"
        reason = str(entry.get("reason") or "")
        if question["kind"] == "orientation":
            verdict = _verdict_of(run, entry)
            assert verdict.get("higher_is_better") is None or "orientation" in reason, key
        else:
            markers = _JUSTIFIED_BY[question["kind"]]
            assert any(m in reason for m in markers), \
                f"{key} asks {question['kind']!r}, which its reason does not name: {reason[:200]}"
        assert question["answer_writes"] in ("value", "orientation", "re_extract"), \
            f"{key} advertises {question['answer_writes']!r} — a note is not an answer to a held cell"


def _verdict_of(run: Path, entry: dict) -> dict:
    from canopy.review.questions import _verdict

    return _verdict(run, entry["paper_id"], entry["dataset_id"], entry["outcome_key"],
                    entry["group"])


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_answering_every_question_the_run_asks_leaves_no_cell_held(tmp_path):
    """The other half of C4's property test: a cell IS unblockable by the questions it asks.

    Per question that is not the claim — a cell held on both groups needs both answered, which is
    exactly the "an override clears only the named blocker" rule. So the honest run-level test is:
    answer every open question, re-pool, ask again, until nothing is open — and then no cell is
    still `needs_human`. Two rounds is what this run takes, because answering Bock's direction
    uncovers the series identity underneath it.
    """
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool

    run = _clone_run(tmp_path)
    apply_overrides_and_repool(run)
    rounds = 0
    while rounds < 5:
        open_now = [q for q in questions_for_run(run) if not q["answered"]]
        if not open_now:
            break
        rounds += 1
        for question in open_now:
            slots = [s for s in question["slots"] if s.get("answerable") and s.get("options")]
            assert question["options"] or slots, \
                f"#{question['number']} ({question['kind']}) asks nothing"
            # §C1: one answer, one override per cell the card names — appending only the first
            # would leave a folded card's other group unanswered and the run would never converge.
            # A card answered one slot at a time (second fold) is answered on every slot.
            answer = ({"option": question["options"][0]["key"], "note": "answered in test"}
                      if question["options"] else
                      {"slots": [{"slot": s["member_id"], "option": s["options"][0]["key"],
                                  "note": "answered in test"} for s in slots]})
            for record in answers_to_overrides(question, answer):
                append_override(run, record)
        apply_overrides_and_repool(run)
    assert 1 <= rounds <= 3, f"the run needed {rounds} rounds of answering"
    rows = json.loads((run / "results" / "extraction_table_all.json").read_text())
    still = [(r["dataset_id"], r["outcome_key"]) for r in rows if r["confidence"] == "needs_human"]
    assert not still, f"answering everything left {still} held"
    gone = {(e["dataset_id"], e["outcome_key"]) for e in
            json.loads((run / "exclusions.json").read_text()) if e["reason"] == "human_override"}
    assert gone, "the cells answered 'not reported' left the analysis with a stated reason"
    assert not json.loads((run / "human_review_queue.json").read_text()), \
        "nothing is held, so nothing is queued — the two now agree"


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_the_orientation_question_quotes_both_readers_ballots(tmp_path):
    """ADVERSARIAL Round 2, C3 change 1: the abstention becomes a C4 question, *both ballots
    quoted*. A reviewer cannot decide a direction from "nobody established it" — they decide it
    from what the two readers said and which of them read the paper right."""
    run = _clone_run(tmp_path)
    question = next(q for q in questions_for_run(run) if q["kind"] == "orientation")
    why = question["why"]
    assert "claude-opus-5" in why and "claude-sonnet-5" in why, why[-400:]
    # each reader's own answer is on the page, not just its prose
    assert "a larger value is MORE" in why and "a larger value is LESS" in why, why[-400:]
    assert "after-effect episodes without feedback are plotted as negative values" in why
    assert "a larger residual error reflects more retained adaptation" in why


# --------------------------------------------- C6/C7: the questions the MAP could not settle
BOCK, HEUER = "b511dbb76fa6", "3570e4ce2a9c"
#: what `apply_overrides_and_repool` must say about an answer it cannot act on itself
MAP_PENDING = "extraction was never bought for this; re-run with --resume to extract it"


def _inclusion_question() -> Any:
    """Bock's real C7 case: `map.json disagreements[0] = "dataset count: primary=2 cross-check=1"`,
    with the cross-check's own rule-cited rejection as the quote. The recorded map predates
    `open_questions`, so the question is built from the model the mapper now writes."""
    from canopy.models import MapQuestion

    return MapQuestion(
        kind="include_dataset", dataset_id=f"{BOCK}:d2",
        question=("Only one of the two mapping agents proposed 'Tracking under +60° rotation: old "
                  "vs young control (naive) sample', and no protocol rule was cited to exclude "
                  "it. Does this review include it?"),
        options=["include it", "exclude it"],
        quotes=["The separate tracking-only control groups … never performed the pointing "
                "adaptation task"])


def _measure_question() -> Any:
    """Heuer d1 `late_adaptation`: the map's own words name an alternative ("; alternatively",
    "Two candidate operationalizations."), and its value locations measure two different things."""
    from canopy.models import MapQuestion

    return MapQuestion(
        kind="which_measure", dataset_id=f"{HEUER}:d1", outcome_key="late_adaptation",
        question=("The map names two measures for late_adaptation in 'Exp 1a: 75° CCW rotation, 8 "
                  "target directions — older vs younger'. Which one does this review's definition "
                  "and measurement window ask for?"),
        options=["adaptive shift", "initial direction error"],
        quotes=["Two candidate operationalizations."])


def _ask_at_map(run: Path, paper: str, *questions: Any) -> None:
    path = run / "papers" / paper / "map.json"
    payload = json.loads(path.read_text())
    payload["study"]["open_questions"] = [q.model_dump(mode="json") for q in questions]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _with_figure(run: Path, paper: str, figure: str) -> None:
    """The clone drops images to stay cheap; a question about a figure needs its crop."""
    import shutil

    src = RERUN / "papers" / paper / "ingest" / "figures" / f"{figure}.png"
    dst = run / "papers" / paper / "ingest" / "figures" / f"{figure}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _asked(run: Path) -> tuple[dict, dict]:
    qs = questions_for_run(run)
    return (next(q for q in qs if q["kind"] == "include_dataset"),
            next(q for q in qs if q["kind"] == "which_measure"))


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_map_question_is_asked_on_the_questions_page_with_the_options_the_map_named(tmp_path):
    """C6/C7 questions block extraction, so they never reach the review queue — and before this
    they reached nothing at all: the mapper wrote them and no page in the tool asked them."""
    run = _clone_run(tmp_path)
    _ask_at_map(run, BOCK, _inclusion_question())
    _ask_at_map(run, HEUER, _measure_question())
    _with_figure(run, HEUER, "fig02")
    inclusion, measure = _asked(run)

    assert inclusion["dataset_id"] == f"{BOCK}:d2" and inclusion["paper"].startswith("Bock")
    assert inclusion["answer_writes"] == "include_dataset"
    assert "Does this review include it?" in inclusion["prompt"]
    assert "never performed the pointing adaptation task" in inclusion["why"]
    keys = [o["key"] for o in inclusion["options"]]
    assert keys[0] == "include" and keys[-1] == "exclude"
    # the protocol's own dataset rules are the reason menu: choosing one IS citing it
    rules = [o["rule"] for o in inclusion["options"] if o["decision"] == "exclude" and o["rule"]]
    assert any("include only the first experiment" in r for r in rules), rules
    assert any("contextual change" in r for r in rules), rules

    assert measure["dataset_id"] == f"{HEUER}:d1" and measure["outcome_key"] == "late_adaptation"
    assert measure["answer_writes"] == "which_measure"
    metrics = {o["analysis_metric"] for o in measure["options"]}
    assert {"endpoint", "change_from_baseline"} <= metrics, metrics
    figure_option = next(o for o in measure["options"]
                         if o["analysis_metric"] == "change_from_baseline")
    assert "Figure 2" in figure_option["location"] and figure_option["quote"]
    assert "adaptive shift" in figure_option["quote"].lower()
    # a named figure shows its crop, like every other kind of question
    assert measure["image"]["path"].endswith("fig02.png"), measure["image"]
    assert (run / measure["image"]["path"]).exists()
    # and both are open until they are answered
    assert inclusion["answered"] is False and measure["answered"] is False


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_each_map_answer_becomes_the_record_the_extract_stage_reads(tmp_path):
    from canopy.pipeline.overrides import _validate

    run = _clone_run(tmp_path)
    _ask_at_map(run, BOCK, _inclusion_question())
    _ask_at_map(run, HEUER, _measure_question())
    inclusion, measure = _asked(run)

    rule = next(o for o in inclusion["options"]
                if o.get("rule") and "contextual change" in o["rule"])
    record = _validate(answer_to_override(inclusion, {"option": rule["key"],
                                                      "note": "tracking-only controls"}))
    assert record["kind"] == "include_dataset" and record["decision"] == "exclude"
    assert record["paper_id"] and record["dataset_id"] == f"{BOCK}:d2"
    assert "contextual change" in record["rule"] and record["note"] == "tracking-only controls"
    assert record["group"] is None and record["justification"], "the log records why, always"

    keep = _validate(answer_to_override(inclusion, {"option": "include", "note": "it is a "
                                                                                "separate sample"}))
    assert keep["decision"] == "include" and keep["rule"] == ""

    chosen = next(o for o in measure["options"] if o["analysis_metric"] == "change_from_baseline")
    picked = _validate(answer_to_override(measure, {"option": chosen["key"],
                                                    "note": "the window asks for the shift"}))
    assert picked["kind"] == "which_measure" and picked["outcome_key"] == "late_adaptation"
    assert picked["winning_analysis_metric"] == "change_from_baseline"
    assert "Figure 2" in picked["winning_location"]
    assert picked["note"] == "the window asks for the shift"


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_map_answers_returns_this_papers_answers_in_log_order_with_the_later_one_winning(tmp_path):
    from canopy.pipeline.overrides import append_override, map_answers

    run = _clone_run(tmp_path)
    _ask_at_map(run, BOCK, _inclusion_question())
    _ask_at_map(run, HEUER, _measure_question())
    inclusion, measure = _asked(run)
    bock = json.loads((run / "papers" / BOCK / "map.json").read_text())["study"]["paper_id"]
    heuer = json.loads((run / "papers" / HEUER / "map.json").read_text())["study"]["paper_id"]

    assert map_answers(run, bock) == [], "no overrides file is no answers, not a crash"
    append_override(run, answer_to_override(inclusion, {"option": "include", "note": "first"}))
    append_override(run, answer_to_override(measure, {"option": measure["options"][0]["key"],
                                                      "note": "heuer"}))
    append_override(run, answer_to_override(inclusion, {"option": "exclude", "note": "second"}))

    mine = map_answers(run, bock)
    assert [a["kind"] for a in mine] == ["include_dataset"], "one answer per question, the last"
    assert mine[0]["decision"] == "exclude" and mine[0]["note"] == "second"
    assert map_answers(run, heuer)[0]["kind"] == "which_measure"
    assert len(map_answers(run, heuer)) == 1, "another paper's answers are not this paper's"
    # a sha12 names the same paper as its sha256
    assert map_answers(run, bock[:12]) == mine


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_excluding_a_dataset_at_the_map_question_takes_it_out_of_the_analysis(tmp_path):
    """`exclude` is the one map answer the tool can act on with no model call: the dataset leaves
    the analysis exactly as an `exclude_dataset` decision, and says which rule excluded it."""
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool

    run = _clone_run(tmp_path)
    _ask_at_map(run, BOCK, _inclusion_question())
    apply_overrides_and_repool(run)
    before = {(r["dataset_id"], r["outcome_key"])
              for r in json.loads((run / "results" / "extraction_table_all.json").read_text())}
    assert (f"{BOCK}:d2", "late_adaptation") in before, "the run bought this extraction anyway"

    inclusion = next(q for q in questions_for_run(run) if q["kind"] == "include_dataset")
    rule = next(o for o in inclusion["options"] if o.get("rule"))
    append_override(run, answer_to_override(inclusion, {"option": rule["key"],
                                                        "note": "not the adaptation task"}))
    summary = apply_overrides_and_repool(run)
    assert summary["applied"] == 1 and not summary["pending"], summary
    after = {(r["dataset_id"], r["outcome_key"])
             for r in json.loads((run / "results" / "extraction_table_all.json").read_text())}
    assert (f"{BOCK}:d2", "late_adaptation") not in after
    gone = [e for e in json.loads((run / "exclusions.json").read_text())
            if e["dataset_id"] == f"{BOCK}:d2" and e["reason"] == "human_override"]
    assert gone and rule["rule"][:20] in gone[0]["detail"], gone
    # and the question now reads as answered
    assert next(q for q in questions_for_run(run)
                if q["kind"] == "include_dataset")["answered"] is True


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_include_and_which_measure_are_pending_because_the_extraction_was_never_bought(tmp_path):
    """The honest answer to "include it": the tool cannot, here. It says so instead of recording
    a decision that changed nothing — the cell is read on the next `--resume`."""
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool

    run = _clone_run(tmp_path)
    _ask_at_map(run, BOCK, _inclusion_question())
    _ask_at_map(run, HEUER, _measure_question())
    inclusion, measure = _asked(run)
    append_override(run, answer_to_override(inclusion, {"option": "include", "note": "keep it"}))
    append_override(run, answer_to_override(measure, {"option": measure["options"][0]["key"],
                                                      "note": "the printed one"}))
    summary = apply_overrides_and_repool(run)
    assert summary["applied"] == 0
    assert [p["why"] for p in summary["pending"]] == [MAP_PENDING, MAP_PENDING], summary["pending"]
    assert {p["kind"] for p in summary["pending"]} == {"include_dataset", "which_measure"}


def test_the_log_refuses_a_map_answer_with_an_unknown_decision_no_target_or_no_winner():
    from canopy.pipeline.overrides import KINDS, OverrideRejected, _validate

    assert "include_dataset" in KINDS and "which_measure" in KINDS
    good = {"kind": "include_dataset", "paper_id": "a" * 64, "dataset_id": "p:d2",
            "decision": "exclude", "rule": "Exclude experiments in which…", "quote": "",
            "note": "the controls never adapted"}
    record = _validate(good)                    # the spec's own record shape, with no justification
    assert record["decision"] == "exclude" and record["justification"]
    for broken in ({**good, "decision": "maybe"}, {**good, "decision": ""},
                   {**good, "dataset_id": ""}, {**good, "paper_id": ""},
                   {**good, "note": "", "rule": "", "justification": ""}):
        with pytest.raises(OverrideRejected):
            _validate(broken)

    measure = {"kind": "which_measure", "paper_id": "a" * 64, "dataset_id": "p:d1",
               "outcome_key": "late_adaptation", "winning_analysis_metric": "change_from_baseline",
               "note": "the window asks for the shift"}
    assert _validate(measure)["winning_analysis_metric"] == "change_from_baseline"
    assert _validate({**measure, "winning_analysis_metric": "",
                      "winning_location": "Figure 2, panel a"})["winning_location"]
    for broken in ({**measure, "winning_analysis_metric": ""},
                   {**measure, "outcome_key": ""}, {**measure, "dataset_id": ""}):
        with pytest.raises(OverrideRejected):
            _validate(broken)


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_run_that_has_been_copied_shows_its_own_pictures_and_not_the_originals(tmp_path):
    """Paths in the records are relative to the working directory the run was made in, so a run
    directory that has been moved or copied — which is what serving a run from anywhere but the
    machine that made it means — must find its own crop. Before this the question pointed at the
    *original* run's file: an absolute path outside the run, which the server cannot serve (it
    refuses to serve anything outside the run) and which a markdown reader cannot open."""
    import shutil

    run = _clone_run(tmp_path)
    for rel in ("papers/3570e4ce2a9c/ingest/figures/fig02.png",
                "papers/3570e4ce2a9c/figures/fig02.overlay1.png"):
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RERUN / rel, run / rel)
    question = next(q for q in questions_for_run(run)
                    if q["dataset_id"] == "3570e4ce2a9c:d1" and q["outcome_key"] == "late_adaptation")
    path = question["image"]["path"]
    assert path and not Path(path).is_absolute(), path
    assert "rerun-fixed" not in path, "that is the original run's file, not this run's"
    assert (run / path).exists()


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_map_answer_does_not_tick_off_the_questions_of_a_cell_that_was_read(tmp_path):
    """"This dataset belongs in the review" says nothing about what its error bars are.

    The two decisions share a dataset_id, and the cell questions match any override on their
    dataset — so without this an inclusion would mark every question of that dataset answered,
    which is the record saying "answered" about a cell nothing has settled. An exclusion is the
    exception: it takes the cell out of the analysis, so it does answer it.
    """
    from canopy.pipeline.overrides import append_override

    run = _clone_run(tmp_path)
    _ask_at_map(run, BOCK, _inclusion_question())
    inclusion = next(q for q in questions_for_run(run) if q["kind"] == "include_dataset")
    held = (f"{BOCK}:d2", "late_adaptation", "B")
    append_override(run, answer_to_override(inclusion, {"option": "include", "note": "keep it"}))
    after = {(q["dataset_id"], q["outcome_key"], q["group"]): q
             for q in questions_for_run(run, fold=False)}
    assert after[held]["answered"] is False and after[held]["kind"] == "error_bar_type"
    assert next(q for q in questions_for_run(run)
                if q["kind"] == "include_dataset")["answered"] is True

    rule = next(o for o in inclusion["options"] if o.get("rule"))
    append_override(run, answer_to_override(inclusion, {"option": rule["key"], "note": "out"}))
    settled = {(q["dataset_id"], q["outcome_key"], q["group"]): q for q in questions_for_run(run)}
    assert settled[held]["answered"] is True, "an exclusion does settle the cell it removes"


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_an_answer_whose_consequence_has_not_happened_reads_as_pending_a_re_run(tmp_path):
    """The third state. "Include it" and "read it again" are recorded decisions that change
    nothing until the model call they ask for is bought, so they must not wear the same tick as a
    decision that moved a number — that is the overclaim C4 took out of the answers themselves.
    An exclusion is not one of them: it takes effect at the re-pool, with no model call."""
    from canopy.pipeline.overrides import (MAP_PENDING, RE_EXTRACT_PENDING, append_override,
                                           apply_overrides_and_repool)
    from canopy.review.questions import PENDING_RERUN

    run = _clone_run(tmp_path)
    _ask_at_map(run, BOCK, _inclusion_question())
    _ask_at_map(run, HEUER, _measure_question())
    inclusion, measure = _asked(run)
    nothing_read = next(q for q in questions_for_run(run) if q["kind"] == "no_value")
    assert nothing_read["status"] == "open" and nothing_read["pending_why"] == ""

    append_override(run, answer_to_override(inclusion, {"option": "include", "note": "keep it"}))
    append_override(run, answer_to_override(measure, {"option": measure["options"][0]["key"],
                                                      "note": "the printed one"}))
    append_override(run, answer_to_override(nothing_read, {"hint": "Table 2, row 'older', p. 5"}))
    apply_overrides_and_repool(run)

    by_kind = {q["kind"]: q for q in questions_for_run(run)}
    for kind, why in (("include_dataset", MAP_PENDING), ("which_measure", MAP_PENDING),
                      ("no_value", RE_EXTRACT_PENDING)):
        question = by_kind[kind]
        assert question["answered"] is True, kind
        assert question["status"] == PENDING_RERUN, kind
        assert question["pending_why"] == why, (kind, question["pending_why"])

    # an exclusion is a decision the re-pool acts on, so it is answered outright
    rule = next(o for o in inclusion["options"] if o.get("rule"))
    append_override(run, answer_to_override(inclusion, {"option": rule["key"], "note": "out"}))
    apply_overrides_and_repool(run)
    settled = next(q for q in questions_for_run(run) if q["kind"] == "include_dataset")
    assert settled["status"] == "answered" and settled["pending_why"] == ""

    # and the written record carries it, for a reader who never opens the page
    paths = write_questions(run)
    recorded = {q["kind"]: q for q in json.loads(paths["json"].read_text())}
    assert recorded["which_measure"]["status"] == PENDING_RERUN
    assert recorded["include_dataset"]["status"] == "answered"
    assert f"**not applied yet**: {MAP_PENDING}" in paths["md"].read_text(encoding="utf-8")


# ===================================================== fix round 1 — the review's findings, pinned
from canopy.review.questions import PENDING_RERUN            # noqa: E402  (the third state)


def _repool(run: Path) -> dict:
    from canopy.pipeline.overrides import apply_overrides_and_repool

    return apply_overrides_and_repool(run)


def _rows(run: Path) -> dict[tuple[str, str], dict]:
    return {(r["dataset_id"], r["outcome_key"]): r
            for r in json.loads((run / "results" / "extraction_table_all.json").read_text())}


def _ask(run: Path, dataset_id: str, outcome_key: str, group: str | None) -> dict:
    """One CELL's own question — the unfolded path (§C1).

    The page folds these into cards (a dataset's pair, a measure's direction), and a card has no
    group of its own. Every test that reaches for a cell by group is about the question the cell
    asks, which is what `fold=False` returns and what every card is built out of.
    """
    return next(q for q in questions_for_run(run, fold=False) if q["dataset_id"] == dataset_id
                and q["outcome_key"] == outcome_key and q["group"] == group)


def _answer(run: Path, question: dict, option: str, note: str = "answered in test") -> None:
    from canopy.pipeline.overrides import append_override

    # one answer, one override per cell it names (§C1): a folded card writes one record per group,
    # and appending only the first would leave the other group unanswered for ever.
    for record in answers_to_overrides(question, {"option": option, "note": note}):
        append_override(run, record)


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_direction_with_no_measure_named_is_refused_instead_of_re_signing_the_whole_paper(
        tmp_path):
    """H1. Bock's map names two measures under `late_adaptation` — Angular pointing error (d1) and
    Tracking RMSE (d2). A direction recorded with a blank `measure_name` used to mean "every
    measure of this outcome in this paper": it flipped both rows and inverted the pooled estimate
    from −0.9233 to +0.6983. The log now refuses to record one, and a record that reaches the
    re-pool without a measure — a hand-edited log, one written before this rule — is refused there
    too, with the two measures it would have landed on named."""
    from canopy.pipeline.overrides import OVERRIDES_FILE, _validate

    run = _clone_run(tmp_path)
    _repool(run)
    before = _rows(run)
    paper_id = json.loads((run / "papers" / BOCK / "map.json").read_text())["study"]["paper_id"]
    blank = {"kind": "orientation", "paper_id": paper_id, "outcome_key": "late_adaptation",
             "higher_is_better": True, "measure_name": "",
             "justification": "a larger pointing error is more adaptation"}
    _validate(blank)          # recordable: an outcome whose map named no measure has a blank one
    (run / OVERRIDES_FILE).write_text(json.dumps({**blank, "seq": 1, "actor": "hand-edited",
                                                  "at": "2026-08-17T00:00:00+00:00"}) + "\n",
                                      encoding="utf-8")
    summary = _repool(run)
    assert summary["applied"] == 0 and len(summary["pending"]) == 1
    why = summary["pending"][0]["why"]
    assert "different measures" in why and "Tracking" in why, why
    after = _rows(run)
    for key in (f"{BOCK}:d1", f"{BOCK}:d2"):
        assert after[(key, "late_adaptation")]["es"] == before[(key, "late_adaptation")]["es"]


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_value_answer_does_not_release_a_cell_held_by_a_blocker_it_never_named(tmp_path):
    """H2/M1, §C4's mandatory clause for the override kind 7 of the 11 questions write.

    Heuer d1 late adaptation carries three `error` flags — `calibration_disputed`,
    `quote_not_grounded`, `value_outside_axis` — and was settled by an adjudicator. The page asks
    one question about it, `which_axis`. Answering it used to stamp the cell `accept_with_note`
    and pool a row whose quote is not printed in the paper, with nobody asked about the quote.
    """
    run = _clone_run(tmp_path)
    _repool(run)
    cell = (f"{HEUER}:d1", "late_adaptation")
    assert _rows(run)[cell]["confidence"] == "needs_human"
    for group in ("A", "B"):
        question = _ask(run, *cell, group)
        assert question["kind"] == "which_axis"
        _answer(run, question, question["options"][0]["key"], "read off the left ladder")
    _repool(run)

    row = _rows(run)[cell]
    assert row["confidence"] == "needs_human", "the quote and the adjudication were never answered"
    assert row["primary_row"] is not True
    # and the cell now asks about what is still holding it, as an OPEN question
    nxt = _ask(run, *cell, "A")
    assert nxt["kind"] == "quote_not_found" and nxt["status"] == "open"
    assert "could not be found in the paper" in nxt["prompt"]
    # answering that one too still leaves the adjudication, which is asked as a confirmation
    _answer(run, nxt, nxt["options"][0]["key"], "it is printed on p. 4")
    _repool(run)
    assert _rows(run)[cell]["confidence"] == "needs_human"
    assert _ask(run, *cell, "A")["kind"] == "confirm_value"


def test_each_override_kind_clears_only_the_blocker_it_names(tmp_path):
    """§C4's regression test written per override KIND, which is where the rule actually lives.

    `value` and `mark_reviewed` derive the bucket from what is left; `orientation` does the same
    through its own rule; `re_extract` changes nothing at all until a model has run.
    """
    from canopy.models import CheckFlag, Verdict
    from canopy.pipeline.overrides import (VALUE_CLEARS_MEAN, _bucket_after_orientation,
                                           _bucket_after_value, _derived_bucket,
                                           codes_cleared_by_value)

    def cell(**over: Any) -> Verdict:
        base = dict(dataset_id="d", outcome_key="o", group="A", agreement="agree",
                    verifier_verdict="confirmed", mean=12.0, dispersion_value=2.0, n=20,
                    higher_is_better=False, confidence="needs_human", confidence_score=0.62,
                    needs_human=True)
        return Verdict.model_validate({**base, **over})

    axis_answer = {"mean": 27.7, "clears": []}
    # what a value answer names, and nothing else
    assert "axis_conflict" in codes_cleared_by_value(axis_answer)
    assert "quote_not_grounded" not in codes_cleared_by_value(axis_answer)
    assert codes_cleared_by_value({"dispersion_type": "SE"}) == frozenset(
        {"dispersion_type_from_legend", "figure_error_bar_unknown", "dispersion_type_conflict"})
    assert codes_cleared_by_value({}) == frozenset()
    # …and "UNKNOWN" answers nothing: it is the error-bar question restated. The test was the
    # field's truthiness and the string is truthy, so "the paper never labels these bars" used to
    # retire `figure_error_bar_unknown` — the finding that asks — and the cell stopped asking while
    # its row divided by a spread of no known kind and converted to nothing.
    assert codes_cleared_by_value({"dispersion_type": "UNKNOWN"}) == frozenset()
    assert codes_cleared_by_value({"mean": 1.0, "dispersion_type": "UNKNOWN"}) == VALUE_CLEARS_MEAN
    assert "series_transposed" in codes_cleared_by_value({"mean": 1.0,
                                                          "clears": ["series_transposed"]})

    # VALUE: one blocker → released; a second of any kind → still held
    assert _bucket_after_value(cell(), axis_answer) == "accept_with_note"
    for second in (dict(confidence_score=0.44), dict(mean=None), dict(adjudicated=True),
                   dict(verifier_verdict="refuted"),
                   dict(flags=[CheckFlag(code="quote_not_grounded", severity="error")]),
                   dict(flags=[CheckFlag(code="series_transposed", severity="warn")])):
        assert _bucket_after_value(cell(**second), axis_answer) == "needs_human", second
    # a disagreement about the number is answered by a number and by nothing else
    assert _bucket_after_value(cell(agreement="disagree"), axis_answer) == "accept_with_note"
    assert _bucket_after_value(cell(agreement="disagree"),
                               {"dispersion_type": "SE"}) == "needs_human"

    # MARK_REVIEWED that names codes derives too; the plain one is the wider "I have checked it"
    assert _derived_bucket(cell(adjudicated=True)) == "needs_human"
    assert _derived_bucket(cell()) == "accept_with_note"
    # ORIENTATION is unchanged and still one-sided
    assert _bucket_after_orientation(cell()) == "accept_with_note"
    assert _bucket_after_orientation(cell(adjudicated=True)) == "needs_human"
    # a cell already accepted is capped by any of them, never promoted
    healthy = cell(confidence="auto_accept", confidence_score=0.88, needs_human=False)
    assert _bucket_after_value(healthy, axis_answer) == "accept_with_note"
    assert _derived_bucket(healthy) == "accept_with_note"


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_re_extraction_changes_nothing_until_a_run_says_it_consumed_it(tmp_path):
    """The `re_extract` half of the per-kind regression, and M8's clearing rule.

    A hint is a request for a model call: until a stage records the seq it acted on, the cell is
    exactly where it was and the question reads "answered — pending re-run".

    And once a stage HAS recorded it on a cell that still has no value — which is what this
    fixture's hand-written seq stands for, a reading bought that came back with nothing — the
    question is open again, carrying what the re-read returned. A green tick there would be the
    overclaim §C4 removed from the answers themselves: a settled question on a cell nothing
    changed about, and a `needs_human` row with nothing left on the page to ask about it.
    """
    run = _clone_run(tmp_path)
    _repool(run)
    cell = (f"{HEUER}:d1", "aftereffect")
    before = _rows(run).get(cell, {}).get("confidence")
    question = _ask(run, *cell, "A")
    assert question["kind"] == "no_value"
    from canopy.pipeline.overrides import append_override

    record = append_override(run, answer_to_override(question, {"hint": "Table 2, row 'older'"}))
    summary = _repool(run)
    assert summary["applied"] == 0 and summary["pending"][0]["kind"] == "re_extract"
    assert _rows(run).get(cell, {}).get("confidence") == before, "nothing moved"
    assert _ask(run, *cell, "A")["status"] == PENDING_RERUN

    stage = run / "papers" / HEUER / "extract.json"
    payload = json.loads(stage.read_text())
    payload["consumed_override_seqs"] = [record["seq"]]
    stage.write_text(json.dumps(payload), encoding="utf-8")
    assert _repool(run)["applied"] == 1
    reopened = _ask(run, *cell, "A")
    assert reopened["status"] == "open"
    assert "Table 2, row 'older'" in reopened["why"] and "absence" in reopened["why"]


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_an_answer_names_the_question_it_answers_so_a_hint_for_one_group_is_not_the_others(
        tmp_path):
    """M2. `already` used to read "any kind that is not `value`" as "carries no group", so a hint
    given for group A marked group B answered — one decision ticking off two cells."""
    from canopy.pipeline.overrides import append_override

    run = _clone_run(tmp_path)
    _repool(run)
    cell = (f"{HEUER}:d1", "aftereffect")
    append_override(run, answer_to_override(_ask(run, *cell, "A"), {"hint": "Table 2, p. 5"}))
    _repool(run)
    assert _ask(run, *cell, "A")["answered"] is True
    other = _ask(run, *cell, "B")
    assert other["answered"] is False and other["status"] == "open"



@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_every_map_option_is_one_the_mapper_can_actually_settle(tmp_path):
    """H4 + the criterion the review asked for: for a map question, "accepted and moves its cell"
    has to mean "accepted AND `apply_map_answers` applies it to the map" — the consequence is
    deferred, so a pending record nothing will ever act on satisfies the weaker form trivially.

    Heuer d1 `late_adaptation` has four `value` locations; two carry no metric at all. Those were
    offered: one was refused by `apply_map_answers` ("it names 'unknown'…") and the other matched
    a *different* location by substring, so the map recorded a measure the reviewer never picked.
    """
    from canopy.agents.mapper import apply_map_answers, extraction_blocks
    from canopy.models import StudyMap
    from canopy.pipeline.overrides import _validate

    run = _clone_run(tmp_path)
    _ask_at_map(run, HEUER, _measure_question())
    _ask_at_map(run, BOCK, _inclusion_question())
    measure = next(q for q in questions_for_run(run) if q["kind"] == "which_measure")
    inclusion = next(q for q in questions_for_run(run) if q["kind"] == "include_dataset")
    assert measure["options"], "a question with no answer is not a question"
    assert all(o["analysis_metric"] not in ("", "unknown") for o in measure["options"]), \
        [o["analysis_metric"] for o in measure["options"]]

    study = StudyMap.model_validate(json.loads((run / "papers" / HEUER / "map.json").read_text())
                                    ["study"])
    for option in measure["options"]:
        record = _validate(answer_to_override(measure, {"option": option["key"], "note": "picked"}))
        answered = apply_map_answers(study, [record])
        assert extraction_blocks(answered) == ({}, {}), option["key"]
        outcome = next(o for o in answered.datasets[0].outcomes
                       if o.outcome_key == "late_adaptation")
        assert outcome.analysis_metric == option["analysis_metric"], \
            f"{option['key']} settled {outcome.analysis_metric!r}, not what was picked"
        assert option["analysis_metric"] in outcome.measure_ruling

    bock = StudyMap.model_validate(json.loads((run / "papers" / BOCK / "map.json").read_text())
                                   ["study"])
    for option in inclusion["options"]:
        record = _validate(answer_to_override(inclusion, {"option": option["key"],
                                                          "note": "decided"}))
        answered = apply_map_answers(bock, [record])
        assert extraction_blocks(answered)[0] == ({} if option["decision"] == "include"
                                                  else {f"{BOCK}:d2": ANY_REASON}), option["key"]


class _AnyReason(str):
    def __eq__(self, other: object) -> bool:                 # any reason, as long as there is one
        return isinstance(other, str) and bool(other)

    def __hash__(self) -> int:
        return 0


ANY_REASON = _AnyReason()


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_the_re_pool_acts_on_the_answer_that_is_still_standing(tmp_path):
    """H5. `map_answers` keeps one answer per question and the re-pool applied every one of them
    in order, so a log reading include → exclude → include left the reviewer's current answer as
    "include", the analysis with the dataset excluded, and the next resume paying to extract a
    dataset the closing re-pool would drop again."""
    from canopy.pipeline.overrides import append_override, map_answers

    run = _clone_run(tmp_path)
    _ask_at_map(run, BOCK, _inclusion_question())
    _repool(run)
    inclusion = next(q for q in questions_for_run(run) if q["kind"] == "include_dataset")
    rule = next(o for o in inclusion["options"] if o.get("rule"))
    paper_id = json.loads((run / "papers" / BOCK / "map.json").read_text())["study"]["paper_id"]

    for option, note in (("include", "keep it"), (rule["key"], "out"), ("include", "back in")):
        append_override(run, answer_to_override(inclusion, {"option": option, "note": note}))
    summary = _repool(run)

    assert [a["decision"] for a in map_answers(run, paper_id)] == ["include"]
    assert summary["applied"] == 0, "the exclusion it changed its mind about must not act"
    assert [p["why"] for p in summary["pending"]] == [MAP_PENDING]
    assert (f"{BOCK}:d2", "late_adaptation") in _rows(run), "the dataset the reviewer kept"
    assert not [e for e in json.loads((run / "exclusions.json").read_text())
                if e["dataset_id"] == f"{BOCK}:d2" and e["reason"] == "human_override"]


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_map_answer_naming_a_dataset_this_run_never_mapped_is_pending_not_an_exclusion(tmp_path):
    """M3. `always=True` wrote an exclusion row whenever there was no row to drop — which is the
    normal state of a map-blocked dataset, but also the state of a typo. An invented dataset was
    reported `applied` and carried into the exclusion table and the PRISMA recount."""
    from canopy.pipeline.overrides import append_override

    run = _clone_run(tmp_path)
    _ask_at_map(run, BOCK, _inclusion_question())
    _repool(run)
    inclusion = next(q for q in questions_for_run(run) if q["kind"] == "include_dataset")
    rule = next(o for o in inclusion["options"] if o.get("rule"))
    payload = answer_to_override(inclusion, {"option": rule["key"], "note": "out"})
    append_override(run, {**payload, "dataset_id": f"{BOCK}:d99"})
    summary = _repool(run)

    assert summary["applied"] == 0 and "no such dataset" in summary["pending"][0]["why"]
    assert not [e for e in json.loads((run / "exclusions.json").read_text())
                if e["dataset_id"] == f"{BOCK}:d99"]


def test_every_forcing_finding_has_a_question_that_can_express_it():
    """M5. §C4: "A hold reason no question kind can express is a bug in the review layer." Eleven
    of the fourteen `error` codes had no entry, so a cell held only by them fell through to
    `confirm_value`, whose answer is a note — a note that released the cell."""
    from canopy.review.questions import _FLAG_TO_KIND
    from canopy.verify.checks import CHECK_SEVERITY
    from canopy.verify.confidence import CONTRADICTING_FLAGS

    asked = {code for code, _ in _FLAG_TO_KIND}
    forcing = {code for code, severity in CHECK_SEVERITY.items() if severity == "error"}
    assert forcing <= asked, sorted(forcing - asked)
    assert CONTRADICTING_FLAGS <= asked, sorted(CONTRADICTING_FLAGS - asked)
    assert {kind for _, kind in _FLAG_TO_KIND} <= set(QUESTION_KINDS)


def test_the_two_locator_findings_ask_a_question_that_states_them():
    """D2's two contradictions, and the question each one raises.

    Neither may fall to `confirm_value`: its answer is a note, and a note does not settle whose
    number this is. "Two places, two numbers" is answered by naming the right number; "every
    reading came off another group's panel" is the same doubt a transposed series raises — which
    thing in this picture is this group? — and has the same two answers. Each prompt states the
    finding that raised it, and each answer clears exactly that finding and nothing else.
    """
    from canopy.review.questions import _answer_kind, _kind, _options, _prompt

    valued = [{"mean": 6.0, "candidate_id": "c1:digitize:ensemble", "route": "figure", "n": 20},
              {"mean": 6.2, "candidate_id": "c2:digitize:ensemble", "route": "figure", "n": 20}]
    verdict = {"agreement": "single", "mean": 6.0, "higher_is_better": True, "flags": []}
    expected = {"locator_reads_conflict": ("which_value", "different numbers"),
                "locator_panel_mismatch": ("which_series", "own panel")}
    for code, (kind, said) in expected.items():
        flags = [code]
        assert _kind(verdict, flags, valued, holding={code}) == kind
        assert _answer_kind(kind) == "value", "a note cannot settle whose number this is"
        options = _options(kind, valued, verdict, flags, "deg", ())
        prompt = _prompt(kind, "young adults", "aftereffect", "Fig. 1", "deg", "", options,
                         verdict, "", (), flags)
        assert said in prompt, prompt
        cleared = {c for option in options for c in (option.get("clears") or [])}
        assert cleared == {code}, cleared


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_reader_contradicting_the_numbers_is_asked_with_the_quote_and_the_two_means(tmp_path):
    """M5, the new CONTRADICTING error: a reader's stated direction disagrees with this cell's own
    means. A reviewer cannot arbitrate that from a code, so the question carries the sentence the
    reader quoted and both resolved means — and offers the only three answers there are."""
    run = _clone_run(tmp_path)
    held = (f"{BOCK}:d2", "late_adaptation", "B")            # a cell the run really does hold
    verify = run / "papers" / BOCK / "verify.json"
    payload = json.loads(verify.read_text())
    for verdict in payload["verdicts"]:
        if (verdict["dataset_id"], verdict["outcome_key"]) == held[:2]:
            verdict["flags"] = [*(verdict.get("flags") or []),
                                {"code": "orientation_reader_contradicts_values",
                                 "severity": "error",
                                 "message": "the reader says group A came out higher; the "
                                            "resolved means say the opposite"}]
    verify.write_text(json.dumps(payload), encoding="utf-8")

    question = _ask(run, *held)
    assert question["kind"] == "reader_contradicts_values"
    assert [o["key"] for o in question["options"][:2]] == ["numbers_right", "groups_swapped"]
    assert any(o.get("mean") is not None for o in question["options"][2:]), "and 'a value is wrong'"
    assert "The reader said" in question["prompt"] and "resolved to" in question["prompt"]
    study = json.loads((run / "papers" / BOCK / "map.json").read_text())["study"]
    dataset = next(d for d in study["datasets"] if d["dataset_id"] == held[0])
    for side in ("group_a", "group_b"):                       # both groups' own labels and means
        assert dataset[side]["label"] in question["prompt"], dataset[side]["label"]

    # (a) clears that finding and nothing else; (b) is the group-mapping path; (c) writes a value
    from canopy.pipeline.overrides import _validate

    kept = _validate(answer_to_override(question, {"option": "numbers_right", "note": "misread"}))
    assert kept["kind"] == "mark_reviewed"
    assert kept["clears"] == ["orientation_reader_contradicts_values"]
    swapped = _validate(answer_to_override(question, {"option": "groups_swapped", "note": "swap"}))
    assert swapped["kind"] == "exclude_dataset"
    number = next(o for o in question["options"] if o.get("mean") is not None)
    assert _validate(answer_to_override(question, {"option": number["key"],
                                                   "note": "the printed one"}))["kind"] == "value"


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_an_answer_may_only_retire_a_finding_its_cell_actually_carries(tmp_path):
    """`clears` is the one field of an answer that RETIRES evidence, so it may name only what the
    run recorded on that cell. Without the check, a record — hand-written, or an option built from
    a question the cell has moved past — could retire a finding nobody ever raised, and the log
    would read as though a reviewer had settled it.

    The options build themselves from the cell for the same reason: Bock's series question names
    the marker mismatch that was found and says nothing about a transposition that was not.
    """
    from canopy.pipeline.overrides import (OVERRIDES_FILE, OverrideRejected, append_override,
                                           recorded_flags)

    run = _clone_run(tmp_path)
    _repool(run)
    cell = (f"{HEUER}:d1", "late_adaptation")
    _answer(run, _ask(run, *cell, "A"), "v1", "read off the left ladder")
    _repool(run)

    question = _ask(run, *cell, "A")
    assert question["kind"] == "quote_not_found"
    carried = recorded_flags(run, {"paper_id": question["paper_id"], "dataset_id": cell[0],
                                   "outcome_key": cell[1], "group": "A"})
    assert "quote_not_grounded" in carried and "series_transposed" not in carried
    # every option names only findings that are on the cell
    for option in question["options"]:
        assert set(option.get("clears") or []) <= carried, option["key"]
    assert next(o for o in question["options"] if o.get("mean") is not None)["clears"] == [
        "quote_not_grounded"]

    payload = answer_to_override(question, {"option": question["options"][0]["key"],
                                            "note": "it is printed on p. 4"})
    with pytest.raises(OverrideRejected) as refused:
        append_override(run, {**payload, "clears": ["series_transposed"]})
    assert "does not carry" in str(refused.value) and "series_transposed" in str(refused.value)
    append_override(run, payload)                       # the honest one is recorded as before

    # and a record that reaches the log another way is refused where it would have acted
    before = {(k, v.get("confidence")) for k, v in _rows(run).items()}
    with (run / OVERRIDES_FILE).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**payload, "clears": ["series_transposed"], "seq": 99,
                                 "actor": "hand-edited",
                                 "at": "2026-08-17T00:00:00+00:00"}) + "\n")
    summary = _repool(run)
    hand = [p for p in summary["pending"] if p.get("seq") == 99]
    assert hand and "does not carry" in hand[0]["why"]
    assert {(k, v.get("confidence")) for k, v in _rows(run).items()} == before


# ===================================================== fix round 2 — the re-review's findings
def _refute(run: Path, paper: str, dataset_id: str, outcome_key: str) -> None:
    """Mark a real cell refuted, the way the verify stage records a refutation."""
    path = run / "papers" / paper / "verify.json"
    payload = json.loads(path.read_text())
    for verdict in payload["verdicts"]:
        if verdict["dataset_id"] == dataset_id and verdict["outcome_key"] == outcome_key:
            verdict["verifier_verdict"] = "refuted"
            verdict["verifier_reason"] = "the printed value for this group is 3.9, not 31.4"
    path.write_text(json.dumps(payload), encoding="utf-8")


def _excludes(question: dict, option: dict) -> bool:
    """Does picking this option take the cell out of the analysis? Asked of the translator rather
    than of the option's name, because §C1's cards spell the exclusion differently (`exclude`,
    `not_reported`, `not_usable`) and all three mean the same thing here."""
    try:
        return any(r.get("kind") == "exclude_dataset"
                   for r in answers_to_overrides(question, {"option": option["key"],
                                                            "note": "answered in test"}))
    except Exception:
        return False


def _answer_everything(run: Path, *, prefer_last: bool = False, rounds: int = 6) -> int:
    """Answer every open question until none is open. `prefer_last` never picks the head option,
    which is where an overrule lives — so it answers everything a *number* can answer."""
    from canopy.pipeline.overrides import append_override

    done = 0
    while done < rounds:
        open_now = [q for q in questions_for_run(run) if q["status"] == "open"
                    and (q["options"] or any(s.get("answerable") and s.get("options")
                                             for s in q["slots"]))]
        if not open_now:
            break
        done += 1
        for question in open_now:
            if question["options"]:
                option = question["options"][-1 if prefer_last else 0]
                if prefer_last and _excludes(question, option):
                    option = question["options"][0]       # excluding the cell is not "a number"
                _answer(run, question, option["key"])
                continue
            # a card answered one slot at a time (§C1, second fold): every slot, by the same rule
            picks = []
            for slot in question["slots"]:
                if not (slot.get("answerable") and slot.get("options")):
                    continue
                option = slot["options"][-1 if prefer_last else 0]
                if prefer_last and _slot_excludes(question, slot, option):
                    option = slot["options"][0]
                picks.append({"slot": slot["member_id"], "option": option["key"],
                              "note": "answered in test"})
            for record in answers_to_overrides(question, {"slots": picks}):
                append_override(run, record)
        _repool(run)
    return done


def _slot_excludes(question: dict, slot: dict, option: dict) -> bool:
    try:
        return any(r.get("kind") == "exclude_dataset"
                   for r in answers_to_overrides(question, {"slots": [
                       {"slot": slot["member_id"], "option": option["key"],
                        "note": "answered in test"}]}))
    except Exception:
        return False


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_refutation_and_an_adjudication_hold_until_an_answer_says_it_overrules_them(tmp_path):
    """N1. A verifier's refutation and an adjudicator's ruling are not flag codes, so no `clears`
    can name them and no number retires them. The terminal confirmation used to assign its own
    bucket, which released both: the refuted Heuer d2 aftereffect pooled at −0.0544 and the
    adjudicated Heuer d1 late row at +0.2683, with nobody asked about either finding."""
    run = _clone_run(tmp_path)
    _refute(run, HEUER, f"{HEUER}:d2", "aftereffect")
    _repool(run)
    refuted, adjudicated = (f"{HEUER}:d2", "aftereffect"), (f"{HEUER}:d1", "late_adaptation")

    _answer_everything(run, prefer_last=True)
    for cell in (refuted, adjudicated):
        assert _rows(run)[cell]["confidence"] == "needs_human", cell
        assert _rows(run)[cell]["primary_row"] is not True, cell
    # the refutation is still being asked — it is not switched off by the answers that landed
    assert _ask(run, *refuted, "A")["kind"] == "verifier_refuted"

    # …and each cell offers the overrule as its honest last step, naming what it overrules
    stands = next(o for o in _ask(run, *refuted, "A")["options"] if o["key"] == "stands")
    assert stands["overrules"] == ["verifier_refuted"]
    confirm = _ask(run, *adjudicated, "A")
    assert confirm["kind"] == "confirm_value"
    assert "adjudicated" in next(o for o in confirm["options"] if o["key"] == "yes")["overrules"]

    # N1(c): the prompt says why THIS cell is held, not a hard-coded sentence
    assert "a verifier reading the whole paper says this value is wrong" \
        in _ask(run, *refuted, "A")["prompt"] or _ask(run, *refuted, "A")["kind"] == "verifier_refuted"
    assert "an adjudicator had to settle it" in confirm["prompt"]
    assert "only one route produced it" not in confirm["prompt"]

    # answering with the overrule releases them, and the record says exactly what it overruled
    rounds = _answer_everything(run)
    assert rounds >= 1
    for cell in (refuted, adjudicated):
        assert _rows(run)[cell]["confidence"] == "accept_with_note", cell
    logged = [json.loads(line) for line in
              (run / "overrides.jsonl").read_text().splitlines() if line.strip()]
    overruled = {name for record in logged for name in record.get("overrules") or []}
    assert {"verifier_refuted", "adjudicated"} <= overruled


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_an_answer_may_not_overrule_a_finding_the_cell_does_not_have(tmp_path):
    """The `overrules` half of the validator: the three non-code findings are checked against the
    cell's own record exactly as `clears` is, and the record stores what it was shown."""
    from canopy.pipeline.overrides import OverrideRejected, append_override

    run = _clone_run(tmp_path)
    _repool(run)
    cell = (f"{HEUER}:d1", "late_adaptation")
    _answer(run, _ask(run, *cell, "A"), "v1", "the left ladder")
    _repool(run)
    _answer(run, _ask(run, *cell, "A"), _ask(run, *cell, "A")["options"][0]["key"], "p. 4")
    _repool(run)

    confirm = _ask(run, *cell, "A")
    assert confirm["kind"] == "confirm_value"
    payload = answer_to_override(confirm, {"option": "yes", "note": "checked"})
    assert "adjudicated" in payload["overrules"]
    assert "verifier_refuted" not in payload["overrules"], "this cell was never refuted"
    with pytest.raises(OverrideRejected) as refused:
        append_override(run, {**payload, "overrules": ["verifier_refuted"]})
    assert "does not carry" in str(refused.value)
    recorded = append_override(run, payload)
    assert "adjudicated" in recorded["carried"], "what the cell was holding, on the record"


def test_a_group_size_does_not_answer_a_statistics_degrees_of_freedom(tmp_path):
    """N2. §C9's df findings say the printed statistic's estimand is unestablished — that this may
    not be the contrast between these two groups at all. A group size is not evidence about that,
    and a spread is not an argument that the denominator is plausible."""
    from canopy.models import CheckFlag, Verdict
    from canopy.pipeline.overrides import (VALUE_CLEARS_DF, _derived_bucket,
                                           codes_cleared_by_value)

    assert not (codes_cleared_by_value({"n": 20}) & VALUE_CLEARS_DF)
    assert not (codes_cleared_by_value({"dispersion_value": 2.0}) & VALUE_CLEARS_DF)
    assert "implausible_dispersion" not in codes_cleared_by_value(
        {"mean": 1.0, "dispersion_value": 2.0, "n": 20, "dispersion_type": "SD"})
    # only a complete pair for BOTH groups retires them, because only then does the row stop
    # converting from the statistic
    assert VALUE_CLEARS_DF <= codes_cleared_by_value({"n": 20}, both_groups_complete=True)

    def cell(**over: Any) -> Verdict:
        base = dict(dataset_id="d", outcome_key="o", group="A", agreement="agree",
                    verifier_verdict="confirmed", mean=40.0, dispersion_value=1.0, n=12,
                    dispersion_type="SD", higher_is_better=False, confidence="needs_human",
                    confidence_score=0.62, needs_human=True)
        return Verdict.model_validate({**base, **over})

    # `implausible_dispersion` is never cleared: it is a finding about the ROW, re-derived by the
    # resolver on every rebuild, and the cells are held while it stands
    other = cell(group="B", mean=10.0)
    assert _derived_bucket(cell(), other=other,
                           row_flags=["implausible_dispersion"]) == "needs_human"
    assert _derived_bucket(cell(), other=other, row_flags=["human_override"]) == "accept_with_note"
    # the cell-level warn is an early warning and not the ruling — the row's flag is
    assert _derived_bucket(cell(flags=[CheckFlag(code="implausible_dispersion", severity="warn")]),
                           other=other) == "accept_with_note"

    # N3: an unresolved direction is forcing, exactly as `confidence.py` says
    assert _derived_bucket(cell(higher_is_better=None)) == "needs_human"


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_statistics_cell_asks_for_both_groups_numbers_and_nothing_else(tmp_path):
    """N2's question side: the df family gets a kind whose only offered answer is "the paper does
    not print them" — everything else has to be typed, for both groups."""
    run = _clone_run(tmp_path)
    held = (f"{HEUER}:d2", "aftereffect", "A")
    path = run / "papers" / HEUER / "verify.json"
    payload = json.loads(path.read_text())
    for verdict in payload["verdicts"]:
        if (verdict["dataset_id"], verdict["outcome_key"]) == held[:2]:
            verdict["flags"] = [*(verdict.get("flags") or []),
                                {"code": "df_missing", "severity": "error",
                                 "message": "the statistic's degrees of freedom were not recorded"}]
    path.write_text(json.dumps(payload), encoding="utf-8")

    question = _ask(run, *held)
    assert question["kind"] == "needs_group_values"
    assert [o["key"] for o in question["options"]] == ["not_reported"]
    assert "both groups' own numbers" in question["prompt"]
    assert question["answer_writes"] == "value"


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_resume_that_rewrites_the_stage_file_cannot_un_apply_an_answer(tmp_path):
    """N5. `clears` was validated against the verify stage file at apply time, so deleting the
    finding — which is exactly what a re-extraction that finds the quote produces — flipped
    already-applied answers to pending and dropped their numbers out of the analysis."""
    run = _clone_run(tmp_path)
    _repool(run)
    cell = (f"{HEUER}:d1", "late_adaptation")
    _answer(run, _ask(run, *cell, "A"), "v1", "the left ladder")
    _repool(run)
    question = _ask(run, *cell, "A")
    assert question["kind"] == "quote_not_found"
    _answer(run, question, question["options"][0]["key"], "it is printed on p. 4")
    before = _repool(run)
    assert before["applied"] == 2 and not before["pending"]
    rows = _rows(run)

    path = run / "papers" / HEUER / "verify.json"
    payload = json.loads(path.read_text())
    for verdict in payload["verdicts"]:                   # the resume found the quote
        if (verdict["dataset_id"], verdict["outcome_key"]) == cell:
            verdict["flags"] = [f for f in verdict["flags"]
                                if f["code"] != "quote_not_grounded"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    after = _repool(run)
    assert after["applied"] == 2 and not after["pending"], after["pending"]
    assert _rows(run)[cell]["mean_a"] == rows[cell]["mean_a"]


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_marking_a_cell_reviewed_never_stamps_a_bucket_the_record_does_not_support(tmp_path):
    """N1(d)/N7. `mark_reviewed` assigned whatever bucket its payload asked for, so a decision
    posted straight to `/overrides` could put a cell at `auto_accept` — above the cap every other
    reviewed cell carries — and release one held by findings the decision never mentioned. It is
    now always derived and never above `accept_with_note`; "keep holding it" is still honoured."""
    from canopy.pipeline.overrides import append_override

    run = _clone_run(tmp_path)
    _repool(run)
    cell = (f"{HEUER}:d1", "late_adaptation")
    assert _rows(run)[cell]["confidence"] == "needs_human"
    append_override(run, {"kind": "mark_reviewed", "paper_id": _ask(run, *cell, "A")["paper_id"],
                          "dataset_id": cell[0], "outcome_key": cell[1],
                          "confidence": "auto_accept",
                          "justification": "I have looked at this cell and it is fine"})
    _repool(run)
    row = _rows(run)[cell]
    assert row["confidence"] == "needs_human", "it named none of the findings holding this cell"
    assert row["confidence"] != "auto_accept"

    # on a cell nothing else holds, the same decision lands at the cap and no higher
    from canopy.models import CheckFlag, Verdict
    from canopy.pipeline.overrides import _derived_bucket

    healthy = Verdict.model_validate(
        dict(dataset_id="d", outcome_key="o", group="A", agreement="agree", mean=1.0,
             dispersion_value=1.0, n=20, higher_is_better=True, verifier_verdict="confirmed",
             confidence="needs_human", confidence_score=0.62, needs_human=True))
    assert _derived_bucket(healthy) == "accept_with_note"
    assert _derived_bucket(healthy.model_copy(update={"confidence": "auto_accept"})) \
        == "accept_with_note", "a cell already accepted is capped, never promoted"


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_re_pool_writes_the_review_queue_csv_with_the_runs_own_columns(tmp_path):
    """The CSV a reviewer opens is written twice — once by the run, once by every re-pool — and the
    re-pool had its own column list. `csv.DictWriter(extrasaction="ignore")` drops a key that list
    omits without a word, so C11's score, margin and boundary were in the JSON and the HTML table
    and gone from the CSV after any answer. One list, imported from the writer that owns it."""
    import csv

    from canopy.pipeline.run import REVIEW_QUEUE_COLUMNS

    run = _clone_run(tmp_path)
    _repool(run)
    with (run / "human_review_queue.csv").open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == list(REVIEW_QUEUE_COLUMNS), header
    for column in ("confidence_score", "confidence_margin", "nearest_boundary"):
        assert column in header, column


# ===================================================== fix round 3 — the row is a thing that holds
CRESSMAN = "5039533c85ef"


def _shrink_dispersions(run: Path, paper: str, dataset_id: str, outcome_key: str,
                        value: float = 0.05) -> None:
    """Make a real row's denominator tiny, so its RESOLVED |d| is one C9's screen must refuse.

    The cells stay exactly as the run read them — a mean, a spread, an n, both `accept_with_note`.
    Only the arithmetic they imply is impossible, which is the whole point of a row-level screen:
    no cell can see it.
    """
    path = run / "papers" / paper / "verify.json"
    payload = json.loads(path.read_text())
    for verdict in payload["verdicts"]:
        if verdict["dataset_id"] == dataset_id and verdict["outcome_key"] == outcome_key:
            verdict["dispersion_value"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_marking_both_cells_reviewed_does_not_pool_a_row_the_screen_refused(tmp_path):
    """C9's screen lives on the ROW, because it needs the number the conversion produced. Every
    release path now rebuilds the row through `resolve_effect` and derives the cells' bucket from
    what comes back — so "I opened the figure and this value is what it shows", which is true of
    both cells, no longer pools a row whose denominator cannot be the authors'.

    And the run-level invariant holds in the direction that matters: a held row always has an open
    question. Its cells stay in the queue rather than being released under it."""
    from canopy.pipeline.overrides import append_override

    run = _clone_run(tmp_path)
    cell = (f"{CRESSMAN}:d1", "late_adaptation")
    _shrink_dispersions(run, CRESSMAN, *cell)
    _repool(run)
    assert _rows(run)[cell]["confidence"] == "accept_with_note", "the run itself pooled it"
    assert not [q for q in questions_for_run(run) if (q["dataset_id"], q["outcome_key"]) == cell]

    for group in ("A", "B"):
        append_override(run, {"kind": "mark_reviewed", "paper_id": f"{CRESSMAN}" + "0" * 52,
                              "dataset_id": cell[0], "outcome_key": cell[1], "group": group,
                              "confidence": "accept_with_note",
                              "justification": "I opened the figure and this is what it shows"})
    summary = _repool(run)
    assert summary["applied"] == 2 and not summary["pending"], summary

    row = _rows(run)[cell]
    assert "implausible_dispersion" in row["flags"], row["flags"]
    assert row["confidence"] == "needs_human" and row["primary_row"] is not True
    # …and the row that is held has a question: both cells are still in the queue
    held = [(e["dataset_id"], e["outcome_key"], e["group"])
            for e in json.loads((run / "human_review_queue.json").read_text())]
    assert [g for d, o, g in held if (d, o) == cell] == ["A", "B"], held
    asked = [q for q in questions_for_run(run, fold=False)
             if (q["dataset_id"], q["outcome_key"]) == cell]
    assert len(asked) == 2 and all(q["status"] == "open" for q in asked)
    # …and the page shows the two as ONE card of two slots (§C1, second fold), still open
    shown = [q for q in questions_for_run(run) if (q["dataset_id"], q["outcome_key"]) == cell]
    assert len(shown) == 1 and shown[0]["kind"] == "cell" and shown[0]["status"] == "open"
    assert [s["group"] for s in shown[0]["slots"] if s["answerable"]] == ["A", "B"]


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_a_re_pool_keeps_the_cells_of_a_row_held_by_a_rule_no_cell_carries(tmp_path):
    """The queue is built from cells, so a row refused by a ROW-level rule was queued by the run
    (`run.py` adds its two cells) and deleted by the first re-pool — which `run_pipeline` itself
    performs as soon as an overrides file exists. Queue 2 → 2, not 2 → 0."""
    run = _clone_run(tmp_path)
    cell = (f"{CRESSMAN}:d1", "late_adaptation")
    path = run / "papers" / CRESSMAN / "resolve.json"
    payload = json.loads(path.read_text())
    for record in payload["records"]:
        if (record["dataset_id"], record["outcome_key"]) == cell:
            record["confidence"] = "needs_human"
            record["flags"] = sorted({*record["flags"], "implausible_dispersion"})
    path.write_text(json.dumps(payload), encoding="utf-8")

    def queued() -> list[tuple[str, str, str]]:
        return [(e["dataset_id"], e["outcome_key"], e["group"])
                for e in json.loads((run / "human_review_queue.json").read_text())
                if (e["dataset_id"], e["outcome_key"]) == cell]

    _repool(run)
    assert queued() == [(cell[0], cell[1], "A"), (cell[0], cell[1], "B")]
    _repool(run)
    assert queued() == [(cell[0], cell[1], "A"), (cell[0], cell[1], "B")], "the re-pool kept them"
    # and the cells the row put in the queue are asked about the denominator, not about a number
    asked = [q for q in questions_for_run(run, fold=False)
             if (q["dataset_id"], q["outcome_key"]) == cell]
    assert [q["kind"] for q in asked] == ["dispersion_doubt", "dispersion_doubt"]


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_the_denominator_question_names_the_dispersion_and_every_answer_settles_something(
        tmp_path):
    """N3 + the per-option property, extended to the new kind. The cell's VALUE is reported — the
    doubt is about what it was divided by — so "it is not reported in the paper" was the only
    answer on offer and it was false. Each answer now names the dispersion, and each is accepted by
    the log and does what it says."""
    from canopy.pipeline.overrides import OVERRIDES_FILE, _validate, append_override

    def prepared(name: str) -> Path:
        run = _clone_run(tmp_path, name)
        _shrink_dispersions(run, CRESSMAN, f"{CRESSMAN}:d1", "late_adaptation")
        path = run / "papers" / CRESSMAN / "resolve.json"
        payload = json.loads(path.read_text())
        for record in payload["records"]:
            if record["outcome_key"] == "late_adaptation" and record["dataset_id"].startswith(
                    CRESSMAN):
                record["confidence"] = "needs_human"
                record["flags"] = sorted({*record["flags"], "implausible_dispersion"})
        path.write_text(json.dumps(payload), encoding="utf-8")
        _repool(run)
        return run

    cell = (f"{CRESSMAN}:d1", "late_adaptation")
    base = prepared("base")
    question = _ask(base, *cell, "A")
    assert question["kind"] == "dispersion_doubt" and question["answer_writes"] == "value"
    # `both_right` is gone (whole-diff M3): its own label said it changed nothing, and recording
    # it counted the question as answered, so a row C9 still refuses ended with every question on
    # it ticked and nothing open anywhere. Both remaining answers change the analysis.
    assert [o["key"] for o in question["options"]] == ["se_not_sd", "within_subject"]
    assert "standard errors" in question["options"][0]["label"]
    assert "within-subject" in question["options"][1]["label"]
    assert "cannot be the one the authors used" in question["prompt"]

    for option in question["options"]:
        run = prepared(f"opt-{option['key']}")
        asked = _ask(run, *cell, "A")
        payload = answer_to_override(asked, {"option": option["key"], "note": "read the caption"})
        record = _validate(payload)                       # (1) the log accepts every one of them
        append_override(run, payload)
        summary = _repool(run)
        assert summary["applied"] == 1 and not summary["pending"], (option["key"], summary)
        if option["key"] == "se_not_sd":
            assert record["kind"] == "value" and record["dispersion_type"] == "SE"
            assert _rows(run)[cell]["dispersion_type_a"] == "SE"
        else:
            assert option["key"] == "within_subject"
            assert record["kind"] == "exclude_dataset"
            assert cell not in _rows(run), "the cell leaves the analysis"


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_an_answer_may_name_a_finding_the_cell_holds_now_even_if_the_record_forgot_it(tmp_path):
    """P2. `carried` is written by `append_override`, so a hand-edited record could assert its own
    permission. It is now one of two references, not the only one: a code in `carried` OR in the
    cell's current holds is accepted, a code in neither is refused."""
    from canopy.pipeline.overrides import OverrideRejected, _check_clears

    run = _clone_run(tmp_path)
    _repool(run)
    cell = (f"{HEUER}:d1", "late_adaptation")
    paper_id = _ask(run, *cell, "A")["paper_id"]
    base = {"kind": "mark_reviewed", "paper_id": paper_id, "dataset_id": cell[0],
            "outcome_key": cell[1], "group": "A", "justification": "checked"}

    # in the cell's current holds, absent from a record that never stored one
    assert _check_clears(run, {**base, "clears": ["quote_not_grounded"]}) == ""
    # in `carried` only — a resume has since rewritten the stage file
    assert _check_clears(run, {**base, "clears": ["series_transposed"],
                               "carried": ["series_transposed"]}) == ""
    # in neither: refused, whatever the record says about itself
    refused = _check_clears(run, {**base, "clears": ["sd_nonpositive"],
                                  "carried": ["quote_not_grounded"]})
    assert "does not carry" in refused and "sd_nonpositive" in refused


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
def test_an_excluded_cell_leaves_the_queue_and_shows_one_answered_question(tmp_path):
    """P3, the decision this area asked for. An excluded cell is settled: counting it as
    outstanding for ever made the queue disagree with the exclusion table, and it showed the
    decision twice — once per group."""
    run = _clone_run(tmp_path)
    _repool(run)
    cell = (f"{HEUER}:d1", "aftereffect")
    before = questions_for_run(run, fold=False)
    assert len([q for q in before if (q["dataset_id"], q["outcome_key"]) == cell]) == 2

    _answer(run, _ask(run, *cell, "A"), "not_reported", "the paper does not print it")
    _repool(run)

    assert not [e for e in json.loads((run / "human_review_queue.json").read_text())
                if (e["dataset_id"], e["outcome_key"]) == cell], "settled, so not outstanding"
    mine = [q for q in questions_for_run(run) if (q["dataset_id"], q["outcome_key"]) == cell]
    assert len(mine) == 1 and mine[0]["answered"] is True and mine[0]["status"] == "answered"
    assert "excluded from the analysis" in mine[0]["prompt"]
    assert "not print it" in mine[0]["why"]
    assert [e for e in json.loads((run / "exclusions.json").read_text())
            if (e["dataset_id"], e["outcome_key"]) == cell]


@pytest.mark.skipif(not _HAS_RERUN, reason="runs/rerun-fixed is not on this machine")
@pytest.mark.parametrize("order", ["reviewed_first", "values_first"])
def test_a_row_is_never_held_with_every_question_on_it_answered(tmp_path, order):
    """Q1. The row and its cells are decided by different rules and answered by different records,
    so a sequence could leave them disagreeing — and a held row whose cells are all released is a
    row with no open question anywhere, which is the state C4 exists to remove.

    The sequence is the manual override form's, where nothing re-derives the row afterwards:
    mark both cells reviewed and then supply both values (or the other way round). The row used to
    freeze at `needs_human` with both queue lines reading `accept_with_note` and both questions
    reading answered — permanently, over any number of re-pools.
    """
    from canopy.pipeline.overrides import append_override

    run = _clone_run(tmp_path, f"seq-{order}")
    cell = (f"{CRESSMAN}:d1", "late_adaptation")
    _shrink_dispersions(run, CRESSMAN, *cell)
    _repool(run)
    paper_id = f"{CRESSMAN}" + "0" * 52
    reviewed = [{"kind": "mark_reviewed", "paper_id": paper_id, "dataset_id": cell[0],
                 "outcome_key": cell[1], "group": group, "confidence": "accept_with_note",
                 "justification": "I opened the figure and this is what it shows"}
                for group in ("A", "B")]
    values = [{"kind": "value", "paper_id": paper_id, "dataset_id": cell[0],
               "outcome_key": cell[1], "group": group, "dispersion_value": 5.0,
               "dispersion_type": "SD", "justification": f"the error bars for {group} are ±5 SD"}
              for group in ("A", "B")]
    for record in (reviewed + values if order == "reviewed_first" else values + reviewed):
        append_override(run, record)
    _repool(run)

    for _ in range(2):                       # and it stays consistent over further re-pools
        row = _rows(run)[cell]
        queued = [e for e in json.loads((run / "human_review_queue.json").read_text())
                  if (e["dataset_id"], e["outcome_key"]) == cell]
        asked = [q for q in questions_for_run(run)
                 if (q["dataset_id"], q["outcome_key"]) == cell and q["status"] == "open"]
        if row["confidence"] == "needs_human":
            assert queued and asked, "a held row with no open question is the C4 failure"
        else:
            assert not queued, "a released row does not keep its cells in the queue"
        assert bool(queued) == bool(asked), (row["confidence"], len(queued), len(asked))
        _repool(run)
    # the spread the reviewer gave makes the arithmetic possible again, so the row is released
    assert _rows(run)[cell]["confidence"] != "needs_human"
    assert abs(_rows(run)[cell]["es"]) < 3.0


def test_the_consistency_pass_pulls_the_cells_of_a_still_held_row_back_into_the_queue():
    """Q1's safety net, directly. The repair only goes one way: since the rebuild is the last step
    of every apply, a row still held is one the RESOLVER held — and this module cannot name the
    rule that did it, so it gives the row its questions back rather than releasing it."""
    from canopy.models import EffectSizeRecord, Verdict
    from canopy.pipeline.overrides import _reconcile

    def cells(bucket: str = "accept_with_note") -> dict[tuple[str, str, str], Verdict]:
        return {("d", "o", group): Verdict.model_validate(
            dict(dataset_id="d", outcome_key="o", group=group, mean=1.0, confidence=bucket,
                 needs_human=bucket == "needs_human")) for group in ("A", "B")}

    def row(**over: Any) -> EffectSizeRecord:
        return EffectSizeRecord.model_validate(
            {"dataset_id": "d", "outcome_key": "o", "es": -0.4, "confidence": "needs_human",
             "flags": [], **over})

    # (1) a row-level refusal: the cells follow the row back into the queue and keep asking
    records, verdicts = {("d", "o"): row(flags=["implausible_dispersion"])}, cells()
    _reconcile(records, verdicts, [("d", "o")], set())
    assert records[("d", "o")].confidence == "needs_human"
    assert all(v.needs_human for v in verdicts.values())
    # a row with no effect size at all is the same case: nothing to release
    records, verdicts = {("d", "o"): row(es=None)}, cells()
    _reconcile(records, verdicts, [("d", "o")], set())
    assert all(v.needs_human for v in verdicts.values())

    # (2) …and a row held by something this module cannot even name is treated the same way: the
    # row is never released on the cells' word, only ever given its questions back
    records, verdicts = {("d", "o"): row(flags=["df_missing"])}, cells()
    _reconcile(records, verdicts, [("d", "o")], set())
    assert records[("d", "o")].confidence == "needs_human"
    assert all(v.needs_human for v in verdicts.values())

    # …and it says nothing about a row nobody answered, or one whose cells are still held
    records, verdicts = {("d", "o"): row()}, cells()
    _reconcile(records, verdicts, [], set())
    assert records[("d", "o")].confidence == "needs_human", "untouched rows are the run's business"
    records, verdicts = {("d", "o"): row()}, cells("needs_human")
    _reconcile(records, verdicts, [("d", "o")], set())
    assert records[("d", "o")].confidence == "needs_human"


def test_marking_a_cell_reviewed_keeps_the_stricter_of_the_row_and_the_cells():
    """R1. `mark_reviewed` is the one applier with cell buckets of its own to combine, so it is the
    one that could overwrite the resolver's. The conversion gate is real and it lives on the ROW —
    it raises `df_missing` while the effect size is being built, after the two cells have been
    scored — so taking the cells' minimum discarded a row the gate had refused."""
    from canopy.pipeline.overrides import _stricter
    from canopy.verify.confidence import conversion_gate_bucket

    # the gate, on a row whose cells say nothing about it
    gated, said = conversion_gate_bucket("accept_with_note", "t_stat", ["df_missing"])
    assert gated == "needs_human" and said, (gated, said)
    assert conversion_gate_bucket("accept_with_note", "figure", ["df_missing"])[0] \
        == "accept_with_note", "the gate only speaks about converted routes"

    # …and the composition keeps it, in either position and whatever the cells say
    assert _stricter(gated, "accept_with_note") == "needs_human"
    assert _stricter("accept_with_note", gated) == "needs_human"
    assert _stricter("accept_with_note", "auto_accept") == "accept_with_note"
    assert _stricter("auto_accept", "auto_accept") == "auto_accept"


def test_a_resumed_runs_questions_come_from_the_queue_it_just_built_not_the_stale_manifest(tmp_path):
    """`questions_for_run` must not read the queue off a manifest the run has not rewritten yet.

    While a resumed run writes its outputs, `manifest.json` on disk is still the previous pass's
    manifest: `runs/ceiling-3` wrote THREE questions for a NINE-cell queue that way. The pipeline
    now passes the queue it just built; without one, the freshest record on disk wins.
    """
    import json, os, shutil, time
    from canopy.review.questions import questions_for_run
    if not _HAS_RERUN:
        pytest.skip("runs/rerun-fixed is not on this machine")
    src = RERUN
    run = tmp_path / "run"
    shutil.copytree(src, run, ignore=shutil.ignore_patterns("_before_*", "cache", "*.log"))
    manifest = json.loads((run / "manifest.json").read_text())
    full = list(manifest["human_review_queue"])
    assert len(full) >= 2
    # a stale manifest that knows only the first held cell…
    manifest["human_review_queue"] = full[:1]
    (run / "manifest.json").write_text(json.dumps(manifest))
    # …while the queue file the run has just written carries every cell
    (run / "human_review_queue.json").write_text(json.dumps(full))
    time.sleep(0.02)
    os.utime(run / "human_review_queue.json", None)
    explicit = questions_for_run(run, queue=full)
    assert len(explicit) == len(questions_for_run(src)), "the passed queue is what is asked"
    freshest = questions_for_run(run)
    assert len(freshest) == len(explicit), "the freshest file wins over a stale manifest"
    # and when the manifest IS the newer record, it is trusted again
    manifest["human_review_queue"] = full
    (run / "manifest.json").write_text(json.dumps(manifest))
    (run / "human_review_queue.json").write_text(json.dumps(full[:1]))
    time.sleep(0.02)
    os.utime(run / "manifest.json", None)
    assert len(questions_for_run(run)) == len(explicit)


def test_a_confirm_question_asks_about_the_number_its_yes_button_confirms():
    """#1 of the nine-paper run asked "Is 30.57 degrees right?" beside a button saying
    "yes — 30.2 degrees is right": the prompt took the first candidate carrying a mean (a rival
    reading) while the button took the run's resolved value. #21 had no rival at all and asked
    about "this value". Both come from one place now."""
    from canopy.review.questions import _confirmed_value

    head = {"key": "yes", "label": "yes — 30.2 degrees is right, I have checked it — this "
                                   "overrules a confidence score below the acceptance line"}
    rival = {"key": "v2", "label": "30.57 degrees", "mean": 30.565}
    assert _confirmed_value([head, rival], {"mean": 30.2}, " degrees") == "30.2 degrees"
    # no rival, and no resolved mean on the verdict: the button still knows the number
    only = {"key": "yes", "label": "yes — 15.89 degrees is right, I have checked it"}
    assert _confirmed_value([only], {}, " degrees") == "15.89 degrees"
    # nothing anywhere names one: say so rather than inventing
    assert _confirmed_value([{"key": "yes", "label": "yes — this value is right, I have "
                                                     "checked it"}], {}, "") == "this value"


def test_model_written_text_shown_to_a_reviewer_carries_no_markup():
    """Question #15 offered "read against </antml_parameter> <parameter name="axis_read">Left
    y-axis…" — a tool-call fragment the model left in a free-text field, put in front of a person
    being asked which axis a number was read against."""
    from canopy.review.questions import _short

    leaked = ('</antml_parameter> <parameter name="axis_read">Left y-axis, printed title '
              "'relative direction [deg]'")
    cleaned = _short(leaked, 200)
    assert "<" not in cleaned and "antml" not in cleaned
    assert cleaned.startswith("Left y-axis")
    assert _short("a < b and c > d", 40) == "a < b and c > d"      # plain prose is untouched


# ------------------------------------------------------------------ D4-lite: the analysed n
def _nine(tmp_path: Path) -> Path:
    """A writable copy of the fixture run with its recorded answer cleared.

    The fixture carries one `value` override of its own (Vachon d2's series identity), which is
    part of what makes it a real run — and it would be counted beside the answer these tests are
    about, so they start from the log the reviewer they describe would have.
    """
    from tests.helpers import nine

    run = nine.copy_to(tmp_path)
    (run / "overrides.jsonl").unlink(missing_ok=True)
    return run


def test_a_group_n_answer_rebuilds_every_row_of_the_dataset(tmp_path):
    """Vachon's non-instructed arms were recruited at 19 and 20 and analysed at 18 and 16.

    The answer is one record about the DATASET, and both of its outcomes are rebuilt with it: the
    two numbers came from the same people, so a denominator corrected for late adaptation and not
    for the aftereffect would say the study had two different sample sizes.
    """
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool

    run = _nine(tmp_path)
    append_override(run, {"kind": "group_n", "paper_id": "b7523a41b03a",
                          "dataset_id": "b7523a41b03a:d1", "n_a": 18, "n_b": 16,
                          "quote": "We excluded 4 younger (all from the non-instructed group) "
                                   "and 3 older (1 non-instructed, 2 instructed) participants.",
                          "justification": "the analysed n, after the stated exclusions"})
    summary = apply_overrides_and_repool(run)
    assert summary["applied"] == 1 and not summary["pending"]

    rows = {(r["dataset_id"], r["outcome_key"]): r
            for r in json.loads((run / "results" / "extraction_table_all.json").read_text())}
    late = rows[("b7523a41b03a:d1", "late_adaptation")]
    assert (late["n_a"], late["n_b"]) == (18, 16)
    assert late["se"] == pytest.approx(0.344, abs=0.002)
    # the dataset's other outcome is rebuilt too, even though nothing about it was asked
    assert (rows[("b7523a41b03a:d1", "aftereffect")]["n_a"],
            rows[("b7523a41b03a:d1", "aftereffect")]["n_b"]) == (18, 16)
    # …and the paper's OTHER dataset keeps the sizes the mapper read: the answer named one
    assert (rows[("b7523a41b03a:d2", "late_adaptation")]["n_a"],
            rows[("b7523a41b03a:d2", "late_adaptation")]["n_b"]) == (19, 21)


def test_an_analysed_n_survives_a_later_answer_about_the_same_cell(tmp_path):
    """The n is a fact about the arms, so the next answer's rebuild must not undo it."""
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool

    run = _nine(tmp_path)
    append_override(run, {"kind": "group_n", "paper_id": "b7523a41b03a",
                          "dataset_id": "b7523a41b03a:d1", "n_a": 18, "n_b": 16,
                          "justification": "the analysed n, after the stated exclusions"})
    append_override(run, {"kind": "mark_reviewed", "paper_id": "b7523a41b03a",
                          "dataset_id": "b7523a41b03a:d1", "outcome_key": "late_adaptation",
                          "confidence": "accept_with_note",
                          "justification": "I have opened the figure and checked this cell"})
    apply_overrides_and_repool(run)
    row = next(r for r in json.loads((run / "results" / "extraction_table_all.json").read_text())
               if r["dataset_id"] == "b7523a41b03a:d1" and r["outcome_key"] == "late_adaptation")
    assert (row["n_a"], row["n_b"]) == (18, 16)


def test_an_analysed_n_the_run_has_no_row_for_is_pending_not_applied(tmp_path):
    from canopy.pipeline.overrides import append_override, apply_overrides_and_repool

    run = _nine(tmp_path)
    append_override(run, {"kind": "group_n", "dataset_id": "nobody:d9", "n_a": 3, "n_b": 4,
                          "justification": "a dataset this run never mapped"})
    summary = apply_overrides_and_repool(run)
    assert summary["applied"] == 0 and len(summary["pending"]) == 1
    assert "nobody:d9" in summary["pending"][0]["why"]


# ============================================ a typed pair on a cell the run itself never produced
def _forget_cell(run: Path, dataset_id: str, outcome_key: str) -> None:
    """Take a mapped cell out of every stage file — no candidate, no verdict, no row.

    The shape a cell really has when the extract stage never reached it: the map still asks for
    it (Roller's E1a and E3, Wolpe's second experiment), and the three files below hold nothing
    about it at all. Written by deletion rather than by hand, so what is left is the run's own
    record of every other cell.
    """
    paper = dataset_id.split(":")[0]
    for stage, key, test in (("extract", "candidates", lambda c: c.get("dataset_id") == dataset_id
                              and c.get("outcome_key") == outcome_key),
                             ("verify", "verdicts", lambda v: v.get("dataset_id") == dataset_id
                              and v.get("outcome_key") == outcome_key),
                             ("resolve", "records", lambda r: r.get("dataset_id") == dataset_id
                              and r.get("outcome_key") == outcome_key)):
        path = run / "papers" / paper / f"{stage}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload[key] = [item for item in payload.get(key) or [] if not test(item)]
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _pair(run: Path, dataset_id: str, outcome_key: str,
          a: tuple[float, float, int], b: tuple[float, float, int]) -> None:
    from canopy.pipeline.overrides import append_override

    for group, (mean, spread, n) in (("A", a), ("B", b)):
        append_override(run, {
            "kind": "value", "paper_id": dataset_id.split(":")[0], "dataset_id": dataset_id,
            "outcome_key": outcome_key, "group": group, "mean": mean,
            "dispersion_value": spread, "dispersion_type": "SE", "n": n,
            "justification": f"read off the printed figure for group {group}"})


def test_a_typed_pair_builds_a_row_for_a_mapped_cell_the_run_never_read(tmp_path):
    """§C4: a pair of human values is the whole of a cell, whether or not a reader ever saw it.

    Roller's E1a and E3 and Wolpe's second experiment are mapped cells with no candidate, no
    verdict and no row — the extract stage never reached them. A reviewer who reads the figure
    and types both groups' numbers has supplied everything the resolver needs, and the answer
    came back "no row for this in this run": four complete answers, no row, and a pending line
    that no re-run could ever clear.
    """
    run = _nine(tmp_path)
    cell = ("592b3b55a318:d2", "aftereffect")
    _forget_cell(run, *cell)
    _repool(run)
    assert cell not in _rows(run), "the fixture was not stripped"

    _pair(run, *cell, a=(5.0, 0.463, 13), b=(2.593, 0.463, 14))
    summary = _repool(run)
    assert not [p for p in summary["pending"] if p.get("dataset_id") == cell[0]
                and p.get("outcome_key") == cell[1]], summary["pending"]
    row = _rows(run).get(cell)
    assert row is not None, "two complete group answers built no row"
    assert row["es"] is not None, row
    assert (row["mean_a"], row["mean_b"]) == (5.0, 2.593)


def test_a_typed_pair_survives_every_later_repool_and_every_unrelated_answer(tmp_path):
    """…and it stays. A row a human's numbers built is not undone by a decision about another cell.

    Panouillères d1's aftereffect went into the best-guess line when its pair was typed and was
    gone from both lines after the next round of answers — an exclusion of a SIBLING dataset and
    an analysed n somewhere else — because nothing recreated what the typed values had made.
    """
    from canopy.pipeline.overrides import append_override

    run = _nine(tmp_path)
    cell = ("592b3b55a318:d2", "aftereffect")
    _forget_cell(run, *cell)
    _pair(run, *cell, a=(5.0, 0.463, 13), b=(2.593, 0.463, 14))
    _repool(run)
    es = _rows(run)[cell]["es"]
    assert es is not None

    append_override(run, {"kind": "exclude_dataset", "dataset_id": "592b3b55a318:d1",
                          "outcome_key": "aftereffect",
                          "justification": "a sibling dataset this review does not pool"})
    append_override(run, {"kind": "group_n", "paper_id": "b7523a41b03a",
                          "dataset_id": "b7523a41b03a:d1", "n_a": 18, "n_b": 16,
                          "justification": "the analysed n, after the stated exclusions"})
    for _ in range(2):
        _repool(run)
        row = _rows(run).get(cell)
        assert row is not None, "the row a human's numbers built disappeared on a later repool"
        assert row["es"] == es, (row["es"], es)


# ============================================ a settled cell is not asked again on the next repool
def test_a_confirmed_value_raises_no_further_card_however_often_the_run_is_repooled(tmp_path):
    """§C4's terminus. Langan's two cells and Heuer's aftereffect pair came back after every round.

    Answering with a value spawns a `confirm_value` card; confirming it spawns another, because
    the rebuild puts the same row in front of the same rule. Once a person has typed a number and
    confirmed it, the cell has nothing left to ask — until the row's value changes again.
    """
    from canopy.pipeline.overrides import append_override

    run = _nine(tmp_path)
    dataset_id, outcome_key = "b7523a41b03a:d1", "aftereffect"
    _pair(run, dataset_id, outcome_key, a=(30.57, 2.1, 12), b=(24.4, 1.9, 12))
    _repool(run)
    open_cards = [q for q in questions_for_run(run, fold=False)
                  if q["dataset_id"] == dataset_id and q["outcome_key"] == outcome_key]
    assert open_cards, "the typed value settled the cell before it was confirmed"
    assert {q["kind"] for q in open_cards} == {"confirm_value"}, [q["kind"] for q in open_cards]

    for card in open_cards:                                # confirmed the way the page confirms
        for written in answers_to_overrides(card, {"option": "yes",
                                                   "note": "I opened the figure: this is it"}):
            append_override(run, written)
    for _ in range(2):
        _repool(run)
        assert not [q for q in questions_for_run(run, fold=False)
                    if q["dataset_id"] == dataset_id and q["outcome_key"] == outcome_key], \
            "a settled cell was asked again"

    # …and a NEW number un-settles it: the confirmation was about the value that stood when it
    # was made, and a cell holding a number nobody has confirmed may be asked about again.
    from canopy.pipeline.overrides import read_overrides
    from canopy.review.questions import _value_settled

    def log_for(group: str) -> list[dict]:
        return [o for o in read_overrides(run) if o.get("dataset_id") == dataset_id
                and o.get("outcome_key") in ("", outcome_key)
                and o.get("group") in (None, group)]

    assert _value_settled(log_for("A")) and _value_settled(log_for("B"))
    _pair(run, dataset_id, outcome_key, a=(31.9, 2.1, 12), b=(24.4, 1.9, 12))
    _repool(run)
    assert not _value_settled(log_for("A")), "a value nobody has confirmed is not a settled cell"
    assert not _value_settled(log_for("B"))


# ================================================== fix round 2: the map's word, and the last card
def _map_excludes(run: Path, dataset_id: str, rule: str = "") -> None:
    """Mark a dataset excluded on the copied run's map, the way the adjudicator marks one."""
    path = run / "papers" / dataset_id.split(":")[0] / "map.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for dataset in payload["study"]["datasets"]:
        if dataset["dataset_id"] == dataset_id:
            dataset["included"] = False
            dataset["exclusion_rule"] = rule
            dataset["exclusion_quote"] = "the same participants took part in both experiments"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_a_typed_value_never_resurrects_a_dataset_the_map_excluded(tmp_path):
    """One rule for both answers. The hint on Roller's E1a is refused because the map excluded the
    dataset under a protocol rule; a typed value on the same cell was accepted, built the row, put
    its cells back in the queue and let the ordinary confirmation pool it. An answer that never
    named the exclusion cannot undo it — that is the standing rule everywhere else in this module.
    """
    run = _nine(tmp_path)
    cell = ("592b3b55a318:d2", "aftereffect")
    _forget_cell(run, *cell)
    _map_excludes(run, cell[0], "include only the first experiment")
    _pair(run, *cell, a=(5.0, 0.463, 13), b=(2.593, 0.463, 14))
    summary = _repool(run)

    refused = [p for p in summary["pending"]
               if p.get("dataset_id") == cell[0] and p.get("outcome_key") == cell[1]]
    assert len(refused) == 2, summary["pending"]
    assert "include only the first experiment" in refused[0]["why"], refused[0]["why"]
    assert cell not in _rows(run), "a row was built for a dataset the map excluded"
    assert not [q for q in questions_for_run(run, fold=False) if q["dataset_id"] == cell[0]
                and q["outcome_key"] == cell[1]]


def test_the_block_names_a_rule_even_when_the_map_recorded_none(tmp_path):
    """"the map excluded this dataset under ''" tells a reviewer nothing they can act on."""
    run = _nine(tmp_path)
    cell = ("592b3b55a318:d2", "aftereffect")
    _forget_cell(run, *cell)
    _map_excludes(run, cell[0], "")
    _pair(run, *cell, a=(5.0, 0.463, 13), b=(2.593, 0.463, 14))
    why = [p["why"] for p in _repool(run)["pending"] if p.get("dataset_id") == cell[0]][0]
    assert "''" not in why, why
    assert "does not record" in why, why


def test_an_inclusion_ruling_makes_the_same_refused_answer_actionable(tmp_path):
    """…and the refusal is not a dead end: `include_dataset` is the answer it points at.

    A dataset the map adjudicator excluded carries no open question, so `apply_map_answers` used
    to ignore an inclusion for it and there was no record a reviewer could write that would ever
    change the message. A person overruling the map at dataset level is §C3's ruling one level
    down, and it is now honoured — so the same typed pair, unchanged, builds its row.
    """
    from canopy.pipeline.overrides import append_override

    run = _nine(tmp_path)
    cell = ("592b3b55a318:d2", "aftereffect")
    _forget_cell(run, *cell)
    _map_excludes(run, cell[0], "include only the first experiment")
    _pair(run, *cell, a=(5.0, 0.463, 13), b=(2.593, 0.463, 14))
    _repool(run)
    assert cell not in _rows(run)

    append_override(run, {"kind": "include_dataset", "paper_id": "592b3b55a318",
                          "dataset_id": cell[0], "decision": "include",
                          "rule": "criterion 3: both experiments used different participants",
                          "note": "the map read the design wrong; these are separate samples",
                          "justification": "answered the map's exclusion: include this dataset"})
    summary = _repool(run)
    assert not [p for p in summary["pending"]
                if p.get("dataset_id") == cell[0] and p.get("kind") == "value"], summary["pending"]
    row = _rows(run).get(cell)
    assert row is not None and row["es"] is not None, row


def test_a_confirmed_value_still_asks_while_a_flag_code_withholds_the_cell(tmp_path):
    """MAJOR 2: the terminus may not be reached by dropping the question off a held cell.

    `_overrulable` knows three findings and no flag codes, so a cell forced to `needs_human` by a
    contradiction — `sign_mismatch` and every other `CONTRADICTING_FLAGS` member — passed the
    suppression while still held, still in the queue and with nothing on the page to answer. A
    repeated question is visible; a vanished one is not, so that is worse than the loop it
    replaced.
    """
    from canopy.pipeline.overrides import append_override
    from tests.helpers import nine as nine_helper

    run = _nine(tmp_path)
    dataset_id, outcome_key = "b7523a41b03a:d1", "aftereffect"
    for group in ("A", "B"):
        nine_helper.with_flags(run, dataset_id, outcome_key, group, ["sign_mismatch"])
    # …and nothing `_overrulable` knows about: a healthy score, no refutation, no adjudication.
    # The contradiction is then the ONLY thing withholding the cell, which is the shape the guard
    # could not see.
    path = run / "papers" / "b7523a41b03a" / "verify.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for verdict in payload["verdicts"]:
        if verdict["dataset_id"] == dataset_id and verdict["outcome_key"] == outcome_key:
            verdict.update({"confidence_score": 0.95, "verifier_verdict": "confirmed",
                            "adjudicated": False})
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    _pair(run, dataset_id, outcome_key, a=(30.57, 2.1, 12), b=(24.4, 1.9, 12))
    _repool(run)
    cards = [q for q in questions_for_run(run, fold=False)
             if q["dataset_id"] == dataset_id and q["outcome_key"] == outcome_key]
    assert cards, "the contradiction is not being asked about at all"

    append_override(run, {"kind": "mark_reviewed", "paper_id": "b7523a41b03a",
                          "dataset_id": dataset_id, "outcome_key": outcome_key,
                          "confidence": "accept_with_note",
                          "justification": "confirmed the number, said nothing about the sign"})
    _repool(run)
    row = _rows(run)[(dataset_id, outcome_key)]
    assert row["confidence"] == "needs_human", "the contradiction stopped withholding the cell"
    assert "sign_mismatch" in " ".join(row["flags"] or []) or row["confidence"] == "needs_human"
    assert [q for q in questions_for_run(run, fold=False)
            if q["dataset_id"] == dataset_id and q["outcome_key"] == outcome_key], \
        "a held cell with no question anywhere is the state C4 exists to remove"
    assert [c for c in questions_for_run(run) if c["dataset_id"] == dataset_id
            and c["outcome_key"] == outcome_key], "…and none on the folded page either"
    queued = json.loads((run / "human_review_queue.json").read_text())
    assert [q for q in queued if q.get("dataset_id") == dataset_id
            and q.get("outcome_key") == outcome_key], "the cells are queued with nothing to answer"


def test_only_a_spread_a_row_can_be_built_from_answers_the_error_bar_question():
    """§C4: a type the row still cannot divide by is not an answer to "what do these bars show?".

    The test in `codes_cleared_by_value` was the field's TRUTHINESS, and every type string is
    truthy — so answering "the paper never labels them" (UNKNOWN), or picking the card's own
    "range" button, retired `figure_error_bar_unknown`, the finding that asks. Retiring it moves
    the cell off `error_bar_type` and on to `confirm_value`, which the §C4 terminus is allowed to
    suppress once the value is confirmed; meanwhile `resolve._has_spread` refuses the spread, so
    the row converts to nothing and stands in neither analysis line. Held, with no question
    anywhere — Langan's four rows left the forest exactly that way.

    The KIND is the assertion, not the mere existence of a card: before the fix every type below
    returned `confirm_value`.
    """
    from canopy.review.questions import _codes_answered, _holding_codes, _kind
    from canopy.verify.checks import severity_of

    def asks_after(dispersion_type: str) -> str:
        already = [{"kind": "value", "mean": 12.5, "dispersion_value": 1.5, "n": 9,
                    "dispersion_type": dispersion_type}]
        codes = [code for code in ("figure_error_bar_unknown", "dispersion_unknown")
                 if code not in _codes_answered(already)]
        verdict = {"flags": [{"code": code, "severity": severity_of(code), "message": code,
                              "candidate_ids": [], "detail": {}} for code in codes],
                   "higher_is_better": False, "confidence_score": 0.9,
                   "verifier_verdict": "confirmed", "adjudicated": False}
        return _kind(verdict, codes, valued=[{"mean": 12.5}], holding=_holding_codes(verdict))

    for named in ("SD", "SE", "CI95", "CI90", "IQR"):
        assert asks_after(named) != "error_bar_type", f"{named} answers the question and is ignored"
    for unusable in ("UNKNOWN", "RANGE", "NONE"):
        assert asks_after(unusable) == "error_bar_type", \
            f"{unusable} builds no row, so the cell must keep asking what the bars are"


def test_the_spread_types_that_answer_the_error_bar_question_are_the_ones_a_row_converts_from():
    """The two halves of that rule, pinned against each other so they cannot drift.

    `SPREAD_TYPES_A_VALUE_CONVERTS` is a list of strings beside the clearing rule; `_has_spread` is
    the resolver's own test. If a type is ever added to one and not the other, either an answer
    silences a question while the row stays unconvertible (the bug above), or a perfectly good
    answer stops retiring the finding it settles.
    """
    from canopy.models import DispersionType
    from canopy.pipeline.overrides import SPREAD_TYPES_A_VALUE_CONVERTS
    from canopy.pipeline.resolve import GroupValues, _has_spread

    for kind in DispersionType:
        # the four fields a `value` override can write, and nothing else
        group = GroupValues(mean=12.5, dispersion_value=1.5, n=9, dispersion_type=kind)
        assert (kind.value in SPREAD_TYPES_A_VALUE_CONVERTS) is _has_spread(group), \
            f"{kind.value}: the clearing rule and the resolver disagree about this spread"


def test_no_row_refusal_can_ever_reach_the_confirmed_value_terminus():
    """The interlock that keeps `questions_for_run`'s row-refusal guard from being needed.

    A row refusal reaches `_kind` through `on_the_row`, so the cell asks that code's own question.
    That question is not one a confirmation answers, so the §C4 terminus is never reached while a
    row is refused. The guard in `questions_for_run` holds the invariant if this stops being true —
    and it would stop being true silently, because a code with no `_FLAG_TO_KIND` entry falls
    through to `confirm_value`, which IS a kind the terminus suppresses.
    """
    from canopy.pipeline.overrides import ROW_REFUSALS
    from canopy.review.questions import _ASKED_ONCE, _FLAG_TO_KIND

    mapped = dict(_FLAG_TO_KIND)
    for code in ROW_REFUSALS:
        assert code in mapped, f"{code} has no question kind, so it falls through to confirm_value"
        assert mapped[code] not in _ASKED_ONCE, \
            f"{code} asks {mapped[code]!r}, which a confirmation can end while the row is refused"


def test_a_row_the_resolver_refuses_keeps_its_question_after_the_value_is_confirmed(tmp_path):
    """§C4, the ROW's half: a refused row keeps asking, however settled its cells are.

    Typed, confirmed, and still refused — `dispersion_plausibility_bucket` re-derives the screen
    from the numbers the reviewer themselves supplied, so the row is in neither analysis line. The
    cells must still be asking. (The suppression that ends a confirmed value's question now reads
    the row's refusals as well as the verdict's codes — the two halves `_question` itself unions —
    so a row refusal can never take the last card off a cell. No row-refusal code reaches that
    branch today, because the only one maps to `dispersion_doubt`; the guard is what keeps it true
    if one is ever added, since an unmapped code falls through to `confirm_value`.)
    """
    from canopy.pipeline.overrides import append_override

    run = _nine(tmp_path)
    dataset_id, outcome_key = "b7523a41b03a:d1", "aftereffect"
    # a spread so tight that the implied effect is impossible — the screen refuses the ROW
    for group, mean in (("A", 30.57), ("B", 24.4)):
        append_override(run, {"kind": "value", "paper_id": "b7523a41b03a",
                              "dataset_id": dataset_id, "outcome_key": outcome_key,
                              "group": group, "mean": mean, "dispersion_value": 0.05,
                              "dispersion_type": "SD", "n": 12,
                              "justification": "typed off the figure"})
    append_override(run, {"kind": "mark_reviewed", "paper_id": "b7523a41b03a",
                          "dataset_id": dataset_id, "outcome_key": outcome_key,
                          "confidence": "accept_with_note",
                          "justification": "confirmed the numbers I typed"})
    _repool(run)
    row = _rows(run)[(dataset_id, outcome_key)]
    assert "implausible_dispersion" in (row.get("flags") or []), "the screen did not refuse the row"
    assert row.get("in_best_guess") is False, "a refused row is in neither analysis line"
    assert [q for q in questions_for_run(run, fold=False)
            if q["dataset_id"] == dataset_id and q["outcome_key"] == outcome_key], \
        "a row in neither line was left with no question on either of its cells"
    assert [c for c in questions_for_run(run) if c["dataset_id"] == dataset_id
            and c["outcome_key"] == outcome_key], "…and none on the folded page either"


def test_a_borrowed_direction_comes_from_the_same_measure_or_from_nowhere(tmp_path):
    """MINOR 3: two datasets of one paper can measure different things under one outcome.

    The orientation is decided once per (paper, outcome, MEASURE) — `_apply_orientation` refuses a
    direction that would land on more than one measure for exactly this reason — so a sibling
    measuring something else is not this cell's witness. An error measure and a magnitude measure
    have opposite `higher_is_better`, and borrowing across them signs the row backwards silently.
    """
    run = _nine(tmp_path)
    cell = ("592b3b55a318:d2", "aftereffect")
    _forget_cell(run, *cell)
    path = run / "papers" / "592b3b55a318" / "map.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for dataset in payload["study"]["datasets"]:
        for outcome in dataset["outcomes"]:
            if outcome["outcome_key"] == "aftereffect" and dataset["dataset_id"] == cell[0]:
                outcome["measure_name"] = "initial endpoint error (IEE), mm"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    _pair(run, *cell, a=(5.0, 0.463, 13), b=(2.593, 0.463, 14))
    _repool(run)
    row = _rows(run)[cell]
    assert row["higher_is_better"] is None, \
        "a direction was borrowed from a sibling measuring something else"
    assert row["es"] is None and "direction" in (row["not_convertible_reason"] or "")


def test_a_refused_hint_becomes_a_promised_re_read_once_the_dataset_is_included(tmp_path):
    """MINOR 4: the honest message names an answer, and the answer actually works.

    Telling a reviewer what is in the way is only half the fix — the other half is that there is
    something they can do about it. A dataset the map adjudicator excluded asked no question, so
    `apply_map_answers` ignored an inclusion for it and the message could never change. Now the
    same hint, untouched and with its seq still unconsumed, becomes one the next resume will buy.
    """
    from canopy.pipeline.overrides import append_override, consumed_seqs

    run = _nine(tmp_path)
    dataset_id, outcome_key = "592b3b55a318:d2", "aftereffect"
    _map_excludes(run, dataset_id, "include only the first experiment")
    record = append_override(run, {
        "kind": "re_extract", "paper_id": "592b3b55a318", "dataset_id": dataset_id,
        "outcome_key": outcome_key, "hint": "Figure 3b, the open bars",
        "justification": "the value is plotted rather than printed"})
    why = next(p["why"] for p in _repool(run)["pending"] if p["seq"] == record["seq"])
    assert "include only the first experiment" in why, why
    assert "include_dataset" in why, why

    append_override(run, {"kind": "include_dataset", "paper_id": "592b3b55a318",
                          "dataset_id": dataset_id, "decision": "include",
                          "rule": "criterion 3: the two experiments used different participants",
                          "justification": "answered the map's exclusion: include this dataset"})
    after = next(p["why"] for p in _repool(run)["pending"] if p["seq"] == record["seq"])
    assert "re-reads this cell" in after, after
    assert record["seq"] not in consumed_seqs(run), "nothing has bought the reading yet"


def test_a_dataset_ruling_never_resurrects_a_paper_the_screen_excluded():
    """Re-review MAJOR 7. `include_dataset` is dataset-scoped; a paper with `eligible=False`
    stays out of reach for hints and typed values alike until the PAPER's eligibility card says
    otherwise."""
    from canopy.agents.mapper import unreadable_cell
    from canopy.models import StudyMap

    study = StudyMap.model_validate({
        "paper_id": "p" * 64, "eligible": False, "eligibility_rationale": "no older group",
        "citation": {"first_author": "Any", "year": 2020},
        "datasets": [{"dataset_id": "aaaaaaaaaaaa:d1", "label": "d1", "included": True,
                      "outcomes": [{"outcome_key": "late_adaptation", "sources": []}]}]})
    why = unreadable_cell(study, "aaaaaaaaaaaa:d1", "late_adaptation")
    assert "screen excluded this paper" in why and "eligibility" in why
