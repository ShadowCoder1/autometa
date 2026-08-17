"""The questions page: every held cell as a thing to decide, with its screenshot and its answers.

Two halves: the pure module over real run records (`runs/proof` when present, otherwise the
fixtures), and the server round-trip — ask, answer, see the override and the re-pool.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from canopy.review.questions import (QUESTION_KINDS, answer_to_override, questions_for_run,
                                     write_questions)

REPO = Path(__file__).resolve().parents[1]
PROOF = REPO / "runs" / "proof"


# ------------------------------------------------------------------ the module, on real records
@pytest.mark.skipif(not (PROOF / "manifest.json").exists(), reason="runs/proof is not on this machine")
def test_every_held_cell_of_a_real_run_becomes_a_question_with_a_picture_and_a_reason():
    qs = questions_for_run(PROOF)
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
                                                    "dispersion_type": "SE", "n": "9"})
    assert typed["kind"] == "value" and typed["mean"] == "17.6"
    _validate(typed)                                          # the log accepts it
    with pytest.raises(OverrideRejected):
        _validate(answer_to_override(_q("which_value", options=[]), {}) | {"kind": "value"})


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
