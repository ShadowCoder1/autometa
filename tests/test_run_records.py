"""Task 16: the two real run records, replayed against the hardened code.

These are not new measurements. They are the stage files the first live runs actually wrote
(`runs/20260816-075831-…/papers/5039533c85ef` and
`validation/out/run_cisneros_dev/papers/b511dbb76fa6`, trimmed of their tool-call logs), replayed
through the functions that produced them so that the specific failures those runs exposed cannot
come back:

* **Cressman 2010 Fig. 3a** — tesseract read the tick labels 45/35/25/15 as 4/3/2/1 (the trailing
  digit fell outside the label band) and fitted them to 0.2 px. Every pixel route then produced no
  usable value and `value_outside_axis` fired, as an `error`, on a CORRECT read of 31.3.
* **Bock 2005 Fig. 1** — the error bar is drawn on one side only; the cap walk stopped on the
  marker's own lower edge and the two "arms" were averaged, halving an 11-unit SD to 6.

The Bock record predates `ad17509` (the one-armed fix), so its stored route-C value is the bug.
Nothing here asserts the stored candidate: the pixels are replayed and the answer recomputed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canopy.digitize.calibrate import AxisCalibration
from canopy.digitize.cv import Axes
from canopy.digitize.digitizer import (_Core, _choose_calibration, _values_from_pixels,
                                       ensemble_stats, resolve_arms)
from canopy.digitize.vlm import GroupReadOut, ReadOut
from canopy.stats.effect_sizes import cohens_d

RECORDS = Path(__file__).resolve().parent / "fixtures" / "runs"


def _record(name: str) -> list[dict]:
    return json.loads((RECORDS / name / "extract.json").read_text())["candidates"]


def _cell(name: str, outcome: str, extractor: str, group: str, figure: str = "") -> dict:
    for cand in _record(name):
        if (cand["outcome_key"] == outcome and cand["extractor_id"] == extractor
                and cand["group"] == group
                and (not figure or cand["pixel_provenance"].get("figure_id") == figure)):
            return cand
    raise AssertionError(f"no {extractor} {group} for {outcome} in the {name} record")


def _cal_from(node: dict) -> AxisCalibration:
    return AxisCalibration(axis=node["axis"], scale=node["scale"], a=node["a"], b=node["b"],
                           rmse=node["rmse"], ticks=[tuple(t) for t in node["ticks"]],
                           dropped=[tuple(t) for t in node.get("dropped") or []])


def _core_from(provenance: dict) -> _Core:
    axes = Axes(**{**provenance["axes"],
                   "plot_bbox": tuple(provenance["axes"]["plot_bbox"]),
                   "y_axis_span": tuple(provenance["axes"]["y_axis_span"] or ()) or None,
                   "x_axis_span": tuple(provenance["axes"]["x_axis_span"] or ()) or None})
    return _Core(gray=None, colour=None, axes=axes, tick_rows=list(provenance["tick_rows"]),
                 labels=[], cal=_cal_from(provenance["cal"]) if provenance.get("cal") else None,
                 ocr_status=provenance["ocr_status"], bars=[], markers=[])


def _readouts_from(*provenances: dict) -> list[ReadOut]:
    """The path-D samples the run recorded, rebuilt as `ReadOut`s (means + the ladder they read).

    One ensemble's `per_route` holds one group, so both groups' provenance is merged here — the
    magnitude rule is about the whole figure, not about one series.
    """
    by_call: dict[tuple[str, str], ReadOut] = {}
    for provenance in provenances:
        for row in provenance["per_route"]:
            if row["route"] != "D":
                continue
            key = (row["model"], row["variant"])
            reading = by_call.setdefault(key, ReadOut(model=row["model"], variant=row["variant"]))
            reading.tick_labels = list(row["extra"].get("tick_labels") or [])
            reading.groups.append(GroupReadOut(group=row["group"], mean=row["mean"],
                                               error_half_length=row["error"],
                                               x_read=row["extra"].get("x_read", "")))
    return list(by_call.values())


# ------------------------------------------------------------------ Cressman 2010 (F1)
@pytest.fixture(scope="module")
def cressman_late() -> dict:
    return _cell("cressman", "late_adaptation", "digitize:ensemble", "A")["pixel_provenance"]


@pytest.fixture(scope="module")
def cressman_readouts(cressman_late) -> list[ReadOut]:
    other = _cell("cressman", "late_adaptation", "digitize:ensemble", "B")["pixel_provenance"]
    return _readouts_from(cressman_late, other)


def test_the_cressman_record_is_the_failure_it_is_kept_for(cressman_late):
    """Guard the fixture itself: if this stops holding, the regression below tests nothing."""
    assert [v for _, v in cressman_late["cal"]["ticks"]] == [4.0, 3.0, 2.0, 1.0]
    assert cressman_late["cal"]["rmse"] < 0.25            # a perfect fit of wrong numbers
    assert cressman_late["cal_note"] == "no model ticks to cross-check against"
    ladders = [row["extra"]["tick_labels"] for row in cressman_late["per_route"]
               if row["route"] == "D"]
    assert [45.0, 40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0, -5.0] in ladders


def test_cressman_the_read_outs_own_ladder_calibrates_the_axis(cressman_late, cressman_readouts):
    """Acceptance item 1: the witness vote reaches the true 45..-5 ladder, at no extra cost."""
    choice = _choose_calibration(_core_from(cressman_late), None, None, cressman_readouts)
    assert choice.source == "readout_ticks"
    assert choice.cal is not None
    values = sorted(v for _, v in choice.cal.ticks)
    assert values[-1] == 45.0 and values[0] == -5.0
    assert values[-1] - values[0] == pytest.approx(50.0, abs=1.0)      # tick span, not axis_range
    assert choice.cal.a == pytest.approx(-0.0691, rel=0.02)
    assert "cv_ocr" in choice.witnesses and "refuted" in choice.witnesses["cv_ocr"]


def test_cressman_the_magnitude_rule_names_both_numbers(cressman_late, cressman_readouts):
    """Acceptance item 2: with only the stale ladder, the calibration is refuted, not the value."""
    core = _core_from(cressman_late)
    readouts = [ReadOut(model=r.model, variant=r.variant, groups=r.groups)   # ladders withheld
                for r in cressman_readouts]
    choice = _choose_calibration(core, None, None, readouts)
    assert choice.status == "cal_refuted"
    assert choice.usable_for_pixels is None                 # the pixel routes emit nothing
    assert choice.refutation["tick_max"] == 4.0
    assert choice.refutation["readout_max_abs_mean"] == 33.3


def test_cressman_late_adaptation_still_implies_the_published_effect(cressman_late):
    """The read-outs were right all along; nothing in the fix moves their number."""
    import math

    a = _cell("cressman", "late_adaptation", "digitize:ensemble", "A")
    b = _cell("cressman", "late_adaptation", "digitize:ensemble", "B")
    d = cohens_d(a["mean"], a["dispersion_value"] * math.sqrt(a["n"]), a["n"],
                 b["mean"], b["dispersion_value"] * math.sqrt(b["n"]), b["n"])
    assert d == pytest.approx(-0.224, abs=0.02)


# ------------------------------------------------------------------ Bock 2005 (F3 / item 8-10)
@pytest.fixture(scope="module")
def bock_route_c() -> dict:
    return _cell("bock", "late_adaptation", "digitize:vlm_coords:claude-opus-5", "A",
                 figure="fig01")["pixel_provenance"]


def test_the_bock_record_still_holds_the_pre_fix_value(bock_route_c):
    """The stored candidate is the BUG (this record predates ad17509) — never assert it as truth."""
    assert bock_route_c["route_sample"]["error"] == pytest.approx(5.998, abs=0.01)


def test_bock_one_armed_whisker_replays_to_eleven_units(bock_route_c):
    """Acceptance items 8 and 9, from the recorded pixels rather than the recorded answer."""
    sample = bock_route_c["route_sample"]
    cal = _cal_from(bock_route_c["cal"])
    mean, error, side = _values_from_pixels(cal, sample["y_px"], sample["cap_top_px"],
                                            sample["cap_bottom_px"])
    assert mean == pytest.approx(31.67, abs=0.01)
    assert (error, side) == (pytest.approx(11.00, abs=0.02), "up")
    # …and the same answer straight out of `resolve_arms`, in data units
    up, down = 42.671 - 31.668, 31.668 - 30.671
    assert resolve_arms(up, down, floor=2.0 * abs(cal.a)) == (pytest.approx(11.00, abs=0.02), "up")
    # within 10 % of what the two read-outs said (11.0 and 11.2)
    assert abs(error - 11.1) <= 0.1 * 11.1


def test_bock_route_c_no_longer_halves_the_dispersion_of_either_group():
    """Both groups: the arm that is really the marker's own edge is dropped, not averaged in."""
    for group, half in (("A", 11.00), ("B", 10.88)):
        provenance = _cell("bock", "late_adaptation", "digitize:vlm_coords:claude-opus-5", group,
                           figure="fig01")["pixel_provenance"]
        sample = provenance["route_sample"]
        _, error, side = _values_from_pixels(_cal_from(provenance["cal"]), sample["y_px"],
                                             sample["cap_top_px"], sample["cap_bottom_px"])
        assert error == pytest.approx(half, abs=0.05)
        assert side in ("up", "down")
        assert error > 1.7 * sample["error"] - 0.5          # the recorded value was ~half of this


def test_bock_the_repaired_ensemble_moves_towards_the_published_effect():
    """Acceptance item 10, with the residual gap named rather than asserted away.

    The gold effect for Bock 2005 late adaptation is |d| = 1.676 (Cisneros' human read: older
    31.51 ± 11.12, young 12.28 ± 11.82). Repairing route C's half-length moves the digitiser's
    ensemble from |d| = 1.545 to |d| = 1.61; the remaining 0.07 is in the READ-OUTS' own numbers
    (they put the young group's SD at 13-14.5 where the human read 11.82), not in route C, so it
    is recorded here instead of being hidden behind a loose tolerance.
    """
    ensembles, routes = {}, {}
    for group in ("A", "B"):
        ensembles[group] = _cell("bock", "late_adaptation", "digitize:ensemble", group,
                                 figure="fig01")
        provenance = _cell("bock", "late_adaptation", "digitize:vlm_coords:claude-opus-5", group,
                           figure="fig01")["pixel_provenance"]
        sample = provenance["route_sample"]
        routes[group] = _values_from_pixels(_cal_from(provenance["cal"]), sample["y_px"],
                                            sample["cap_top_px"], sample["cap_bottom_px"])

    repaired = {}
    for group in ("A", "B"):
        values = ensembles[group]["pixel_provenance"]["route_values"]
        # every route EXCEPT the stale route C, whose repaired answer is appended in its place
        others = {k: v for k, v in values.items() if "vlm_coords" not in k}
        assert others, f"group {group}: the record holds no route besides vlm_coords"
        assert len(others) < len(values) or group == "B", (
            f"group {group}: the record no longer holds a vlm_coords route to repair")
        means = [v["mean"] for v in others.values()] + [routes[group][0]]
        errors = [v["error"] for v in others.values()] + [routes[group][1]]
        assert len(means) == len(others) + 1
        repaired[group] = (ensemble_stats(means)[0], ensemble_stats(errors)[0])

    d = cohens_d(repaired["A"][0], repaired["A"][1], 12, repaired["B"][0], repaired["B"][1], 12)
    stored = cohens_d(ensembles["A"]["mean"], ensembles["A"]["dispersion_value"], 12,
                      ensembles["B"]["mean"], ensembles["B"]["dispersion_value"], 12)
    assert abs(d) > abs(stored)                          # the repair moves towards the gold
    assert abs(d) == pytest.approx(1.61, abs=0.05)
    assert abs(abs(d) - 1.676) < 0.08


def _dataset_for(outcome_key: str, metric: str = "unknown"):
    from canopy.models import DatasetSpec, GroupSpec, OutcomeSources

    return DatasetSpec(dataset_id="d1", cluster_id="5039533c85ef", label="Cressman 2010",
                       group_a=GroupSpec(label="younger", n=9),
                       group_b=GroupSpec(label="older", n=10),
                       outcomes=[OutcomeSources(outcome_key=outcome_key, units="deg",
                                                analysis_metric=metric)])


def _candidates(name: str, outcome: str) -> list:
    from canopy.models import Candidate

    return [Candidate.model_validate(c) for c in _record(name)
            if c["outcome_key"] == outcome]


def test_cressman_the_two_aftereffect_figures_are_not_the_same_quantity():
    """Acceptance item 6: Fig 3b (deg, baseline-corrected) and Fig 5 (% of perturbation)."""
    from canopy.pipeline.run import vote_candidates
    from canopy.verify.checks import codes, run_checks

    cell = vote_candidates(_candidates("cressman", "aftereffect"))
    others = vote_candidates(_candidates("cressman", "late_adaptation"))
    metrics = {c.analysis_metric for c in cell if c.status == "found"}
    assert metrics == {"baseline_corrected"}, "the fixture no longer holds both metrics"
    # …because Fig 5's own ensemble came out ambiguous. Add it back the way the vote would see it
    # if it had resolved, and the per-outcome scope catches it:
    fig5 = [c for c in _candidates("cressman", "aftereffect")
            if c.extractor_id == "digitize:readout:claude-opus-5:direct"
            and (c.pixel_provenance or {}).get("figure_id") == "fig05"]
    assert fig5 and {c.analysis_metric for c in fig5} == {"percent_of_perturbation"}
    flags = run_checks(_dataset_for("aftereffect"), "aftereffect", [*cell, *fig5],
                       other_candidates=others)
    assert "metric_mixed" in codes(flags)
    mixed = next(f for f in flags if f.code == "metric_mixed")
    assert "percent_of_perturbation" in mixed.message and "baseline_corrected" in mixed.message
    # and the paper-wide comparison, which used to be the one that fired, is now only a note
    across = [f for f in flags if f.code == "metric_mixed_across_outcomes"]
    assert all(f.severity == "info" for f in across)


def test_cressman_the_fig5_ensemble_says_its_routes_are_not_in_one_unit():
    """Acceptance item 6, second half: 61.6 and 0.061 are not two reads of one number."""
    from canopy.verify.checks import codes, run_checks

    fig5 = [c for c in _candidates("cressman", "aftereffect")
            if c.extractor_id == "digitize:ensemble"
            and (c.pixel_provenance or {}).get("figure_id") == "fig05" and c.group == "A"]
    assert len(fig5) == 1
    means = sorted(row["mean"] for row in fig5[0].pixel_provenance["per_route"]
                   if row["mean"] is not None)
    assert means[0] < 0.1 and means[-1] > 60, f"the fixture no longer holds the split: {means}"
    # the ensemble is `ambiguous`, so give the check a `found` copy of it — what is under test is
    # the diagnosis, not the status
    found = fig5[0].model_copy(update={"status": "found", "mean": 61.6})
    flags = run_checks(_dataset_for("aftereffect"), "aftereffect", [found])
    assert "unit_incoherent" in codes(flags)
    flag = next(f for f in flags if f.code == "unit_incoherent")
    assert "another unit or off another axis" in flag.message


def test_cressman_late_adaptation_is_no_longer_convicted_by_its_own_bad_ladder(cressman_late):
    """The F1 end state: the same record, after the fix, holds no `value_outside_axis` error."""
    from canopy.digitize.digitizer import _choose_calibration
    from canopy.verify.checks import codes, run_checks

    cell = [c for c in _candidates("cressman", "late_adaptation")
            if c.extractor_id == "digitize:ensemble"]
    assert len(cell) == 2 and all(c.mean is not None for c in cell)
    # the record's stored provenance still carries the 1..4 ladder and no status …
    stale = run_checks(_dataset_for("late_adaptation"), "late_adaptation", cell)
    assert "value_outside_axis" in codes(stale), "the fixture no longer shows the failure"
    # … and with what `_choose_calibration` produces today, it does not fire at all
    readouts = _readouts_from(*[c.pixel_provenance for c in cell])
    choice = _choose_calibration(_core_from(cressman_late), None, None, readouts)
    repaired = [c.model_copy(update={"pixel_provenance": {
        **c.pixel_provenance, "cal_status": choice.status,
        "cal": choice.usable_for_pixels.to_dict() if choice.usable_for_pixels else None}})
        for c in cell]
    flags = run_checks(_dataset_for("late_adaptation"), "late_adaptation", repaired)
    assert "value_outside_axis" not in codes(flags)
    assert not [f for f in flags if f.severity == "error"], codes(flags)


def test_cressman_the_record_shows_the_one_family_stop_the_plan_now_prevents(cressman_late):
    """Acceptance item 3, against the run that exposed it (F2).

    The live run stopped at two read-outs, and both were Opus — one voter agreeing with itself.
    The Sonnet read-out was third in the plan and never ran, so the cell could not be accepted by
    agreement however right 31.3 was.
    """
    from canopy.digitize.digitizer import _readout_plan, model_families

    plan = cressman_late["readout_plan"]
    assert [(s["model"], s["variant"]) for s in plan[:2]] == [
        ("claude-opus-5", "direct"), ("claude-opus-5", "ticks_first")]
    assert cressman_late["call_plan"]["readouts_run"] == 2
    ran = {row["model"] for row in cressman_late["per_route"] if row["route"] == "D"}
    assert ran == {"claude-opus-5"}, "the record no longer shows the one-family stop"

    today = _readout_plan(("claude-opus-5",), 3)
    assert [(s.model, s.variant) for s in today[:2]] == [
        ("claude-opus-5", "direct"), ("claude-sonnet-5", "direct")]
    assert len(today) == 3, "three read-outs remains the ceiling"


def test_cressman_the_family_gate_would_have_bought_the_second_read_out(cressman_late):
    """The same samples, through today's gate: one family is a reason to buy, agreement is not."""
    from canopy.digitize.digitizer import RouteSample, _needs_another_readout

    samples = [RouteSample(route=row["route"], group=row["group"], model=row["model"],
                           variant=row["variant"], mean=row["mean"], error=row["error"])
               for row in cressman_late["per_route"]]
    needed, why = _needs_another_readout(samples, axis_range=50.0, tick_spacing=5.0, px_units=0.07)
    assert needed and "one model family" in why
    # add the Sonnet read the plan now buys first, and the gate is satisfied
    samples.append(RouteSample(route="D", group="A", model="claude-sonnet-5", variant="direct",
                               mean=31.4, error=2.0))
    needed, why = _needs_another_readout(samples, axis_range=50.0, tick_spacing=5.0, px_units=0.07)
    assert not needed and "two model families" in why


def test_cressman_the_two_block_33_reads_agree_once_compared_in_pixels(cressman_late):
    """Acceptance item 5: "x ≈ 1451 px" and "x=1449 px" are the same block, two pixels apart."""
    from canopy.digitize.digitizer import RouteSample, _late_window_provenance
    from canopy.digitize.vlm import TargetSpec

    assert cressman_late["late_window_x_agrees"] is False, "the record no longer shows the failure"
    assert len(cressman_late["late_window_x_read"]) == 2

    samples = [RouteSample(route=row["route"], group=row["group"], model=row["model"],
                           mean=row["mean"], extra={"x_read": row["extra"].get("x_read", "")})
               for row in cressman_late["per_route"]]
    out = _late_window_provenance(TargetSpec(outcome_key="late_adaptation",
                                             x_hint="last adaptation block"),
                                  samples, [], x_tick_px=80.0)
    assert out["late_window_x_compared"] == "pixels"
    assert out["late_window_x_px"] == [1449.0, 1451.0]
    assert out["late_window_x_spread_px"] == 2.0
    assert out["late_window_x_agrees"] is True


def test_cressman_fig3b_names_two_value_axes_and_only_one_of_them_is_pooled():
    """Acceptance item 7: the locator itself says the panel has a deg axis AND a per-cent axis."""
    from canopy.digitize.digitizer import RouteSample, _reconcile_axes
    from canopy.digitize.vlm import TargetSpec

    fig3b = _cell("cressman", "aftereffect", "digitize:ensemble", "A", figure="fig03")
    assert "left y-axis" in fig3b["locator"] and "right y-axis" in fig3b["locator"]
    assert "(deg)" in fig3b["locator"] and "(%)" in fig3b["locator"]
    # the read-outs of that run carried no `axis_read` at all — the field did not exist
    assert all("axis_read" not in (row.get("extra") or {})
               for row in fig3b["pixel_provenance"]["per_route"])

    # with it, a reader that answers off the right-hand ladder is dropped rather than averaged in
    deg, pct = "left y-axis 'Aftereffects at Peak Velocity (deg)'", "right y-axis '… (%)'"
    samples = [RouteSample(route="D", group="A", model="claude-opus-5", mean=17.5,
                           extra={"axis_read": deg}),
               RouteSample(route="D", group="A", model="claude-sonnet-5", mean=17.4,
                           extra={"axis_read": deg}),
               RouteSample(route="D", group="A", model="claude-haiku-4-5", mean=58.0,
                           extra={"axis_read": pct})]
    info = _reconcile_axes(samples, TargetSpec(outcome_key="aftereffect", unit_hint="deg"))
    assert info["axis_agreement"] == "conflict"
    assert samples[2].dropped is True and not samples[0].dropped
    assert [s.mean for s in samples if not s.dropped] == [17.5, 17.4]
