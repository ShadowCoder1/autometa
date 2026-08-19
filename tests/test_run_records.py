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


# ------------------------------------------------------------------ the hardened re-run (F11)
#: The live re-run of the hardened pipeline. The digitiser did BETTER than the first run here —
#: `cal_status: confirmed`, two witnesses agreeing on the mapping, both model families answering,
#: means of 31.3/33.4 — and the paper still came out `not_convertible`, because `_reconcile_axes`
#: compared two readers' free-text axis descriptions with a string-similarity ratio and called one
#: verbose and one terse description of the SAME axis a conflict. It evicted the Sonnet read-out
#: from six of the eight cells and left two of them with no usable route at all.
RERUN = "cressman_rerun"


def _rerun_samples() -> list:
    """The re-run's own per-route samples, with ONLY the axis eviction undone.

    The overlay-verification drops are a real judgement about a mark and are kept; what is replayed
    is the axis rule, which is the thing that was wrong.
    """
    from canopy.digitize.digitizer import RouteSample

    samples = []
    for cand in _record(RERUN):
        if cand["extractor_id"] != "digitize:ensemble" or cand["mean"] is None:
            continue
        for row in cand["pixel_provenance"]["per_route"]:
            sample = RouteSample(route=row["route"], group=row["group"], model=row["model"],
                                 variant=row["variant"], mean=row["mean"], error=row["error"],
                                 snap_conf=row["snap_conf"], label_read=row["label_read"],
                                 extra=row["extra"])
            if row["dropped"] and "read off" not in (row["drop_reason"] or ""):
                sample.dropped, sample.drop_reason = True, row["drop_reason"]
            samples.append(sample)
    return samples


def test_the_rerun_record_is_the_regression_it_is_kept_for():
    """Guard the fixture: the digitiser succeeded and the row was lost anyway."""
    ensembles = [c for c in _record(RERUN) if c["extractor_id"] == "digitize:ensemble"]
    assert len(ensembles) == 4
    assert all(c["pixel_provenance"]["cal_status"] == "confirmed" for c in ensembles)
    assert all(c["pixel_provenance"]["axis_agreement"] == "conflict" for c in ensembles)
    # …every one of them evicted the same reader, and two lost every route they had
    assert {tuple(c["pixel_provenance"]["axis_dropped_samples"]) for c in ensembles} == {
        ("digitize:readout:claude-sonnet-5:direct",)}
    assert sum(1 for c in ensembles if c["mean"] is None) == 2
    assert all(c["status"] == "ambiguous" for c in ensembles), \
        "an `ambiguous` ensemble never reaches the resolved values — that is how the row was lost"


def test_two_wordings_of_one_axis_are_one_axis():
    """The fix, on the strings the models actually wrote."""
    from canopy.digitize.digitizer import _axis_similar

    terse = 'left y-axis, "Mean Hand Deviation Angles at Peak Velocity (deg)"'
    verbose = ("left y-axis, 'Mean Hand Deviation Angles at Peak Velocity (deg)', linear, "
               "ticks 45, 35, 25, 15, 5 above the axis line")
    panelled = ('Left y-axis of panel a, "Mean Hand Deviation Angles at Peak Velocity (deg)", '
                'ticks 45, 35, 25, 15')
    assert _axis_similar(terse, verbose) and _axis_similar(terse, panelled)
    # Bock's d2, which raised `axis_conflict` on three descriptions of one RMSE axis
    assert _axis_similar('left y-axis, "RMSE (mm)"',
                         'Left y-axis, printed title "RMSE (mm)". Linear. Tick ladder: 0 (y=899.5)')
    # …and the case the rule exists for still splits
    assert not _axis_similar("left y-axis 'Aftereffects at Peak Velocity (deg)'",
                             "right y-axis 'Aftereffects at Peak Velocity (%)'")
    assert not _axis_similar("y-axis, 'adaptive shift (deg)'",
                             "x-axis, '% Visuomotor Adaptation' (bottom axis)")


def test_the_rerun_cell_resolves_to_the_published_effect_through_the_figure_route():
    """The regression test the re-run earned: this cell must produce a number again.

    The old code got d = -0.230 on this paper. The hardened digitiser reads it slightly differently
    (31.15 / 33.20 against 31.3 / 33.3), so the replay lands at -0.2115 — the assertion is on the
    effect being recovered at all and on its agreeing with the published value, not on reproducing
    a particular route's arithmetic to the last digit.
    """
    import math

    from canopy.digitize.digitizer import (_drop_zero_confidence, _reconcile_axes, ensemble_stats)
    from canopy.digitize.vlm import TargetSpec

    samples = _rerun_samples()
    info = _reconcile_axes(samples, TargetSpec(outcome_key="late_adaptation", unit_hint="deg"))
    assert info["axis_agreement"] == "agreed", "the false conflict is back"

    resolved = {}
    for group in ("A", "B"):
        live = [s for s in samples if s.group == group and s.usable]
        assert len(live) >= 2, f"group {group} lost its readings again"
        assert len({s.model for s in live if s.model}) >= 2, \
            f"group {group} is down to one model family"
        live, _notes = _drop_zero_confidence(live)
        resolved[group] = (ensemble_stats([s.mean for s in live])[0],
                           ensemble_stats([s.error for s in live if s.error is not None])[0])

    n_a, n_b = 9, 10
    d = cohens_d(resolved["A"][0], resolved["A"][1] * math.sqrt(n_a), n_a,
                 resolved["B"][0], resolved["B"][1] * math.sqrt(n_b), n_b)
    assert d == pytest.approx(-0.230, abs=0.02), f"the figure route resolved to d = {d}"
    assert resolved["A"][0] == pytest.approx(31.15, abs=0.5)
    assert resolved["B"][0] == pytest.approx(33.20, abs=0.5)


def test_an_evicted_reader_is_kept_when_it_is_the_only_one_a_group_has():
    """Two of the re-run's cells lost EVERY route: a conflict must not empty a group."""
    from canopy.digitize.digitizer import RouteSample, _reconcile_axes
    from canopy.digitize.vlm import TargetSpec

    left = "left y-axis 'angle (deg)'"
    right = "right y-axis 'percent (%)'"
    samples = [
        RouteSample(route="D", group="A", model="opus", mean=31.0, extra={"axis_read": left}),
        RouteSample(route="D", group="A", model="sonnet", mean=31.2, extra={"axis_read": left}),
        # group B was read by the minority reader ONLY — evicting it leaves the group empty
        RouteSample(route="D", group="B", model="haiku", mean=58.0, extra={"axis_read": right}),
    ]
    info = _reconcile_axes(samples, TargetSpec(outcome_key="x", unit_hint="deg"))
    assert info["axis_agreement"] == "conflict"
    assert samples[2].dropped is False
    assert "no other reading" in samples[2].notes
    assert info["axis_dropped_samples"] == []


# --------------------------------------------------------------------------- C10: a partial read
#
# `runs/rerun-fixed/papers/b511dbb76fa6` — Bock 2005, dataset d1, the after-effect. Three readers
# read the figure; for group A one of them came back with a spread and no mean, and the cell then
# reported "only one independent route produced this value" — the same sentence a genuinely
# single-family cell prints. Group B, read by the same three, scored twice as high. The trimmed
# record is `tests/fixtures/runs/bock_aftereffect/extract.json`.
def _bock_aftereffect(group: str) -> list[dict]:
    return [c for c in _record("bock_aftereffect")
            if c["group"] == group and c["extractor_id"] != "digitize:ensemble"]


def _samples_of(candidates: list[dict]) -> list:
    from canopy.digitize.digitizer import RouteSample

    out = []
    for cand in candidates:
        model = cand["extractor_id"].split(":")[2]
        variant = cand["extractor_id"].split(":")[3]
        out.append(RouteSample(route="D", group=cand["group"], model=model, variant=variant,
                               mean=cand["mean"], error=cand["dispersion_value"],
                               status=cand["status"]))
    return out


def test_the_bock_record_holds_the_partial_read_c10_is_for():
    """The record itself, before any rule: sonnet read the after-effect figure, gave group B a
    full answer and group A a spread with no mean."""
    a = {c["extractor_id"]: (c["mean"], c["dispersion_value"]) for c in _bock_aftereffect("A")}
    b = {c["extractor_id"]: (c["mean"], c["dispersion_value"]) for c in _bock_aftereffect("B")}
    assert a["digitize:readout:claude-sonnet-5:direct"] == (None, 6.5)
    assert b["digitize:readout:claude-sonnet-5:direct"] == (-27.0, 7.0)
    assert len(a) == len(b) == 3                    # the same three readers on both groups


def test_the_partial_read_is_named_by_route_and_group():
    """C10 (a): the reader that produced half an answer is named — route AND group — so the cell
    can say "two families read it and one came back empty" instead of "one family read this"."""
    from canopy.digitize.digitizer import _partial_reads

    named = _partial_reads(_samples_of(_bock_aftereffect("A")))
    assert [(x["route"], x["group"], x["missing"]) for x in named] == [
        ("digitize:readout:claude-sonnet-5:direct", "A", "mean")]
    assert named[0]["has"]["error"] == 6.5
    assert named[0]["model"] == "claude-sonnet-5"   # who to re-ask, not just that someone failed
    assert _partial_reads(_samples_of(_bock_aftereffect("B"))) == []


def test_the_partial_read_does_not_credit_the_empty_reader_as_a_witness():
    """The other half of the rule, and the one that must not be undone: a reader that produced no
    mean corroborates no mean. C10 buys the missing number back; it never counts the absence as
    agreement."""
    from canopy.digitize.digitizer import model_families

    samples = _samples_of(_bock_aftereffect("A"))
    # `digitize()` counts families over the samples that produced a value, and this is the reason:
    # the sonnet reader is present, it answered, and it has no mean for A. Counting it here would
    # credit an abstention as agreement and pool group A on one reader.
    assert sorted(model_families([s for s in samples if s.usable])) == ["claude-opus"]
    assert sorted(model_families(samples)) == ["claude-opus", "claude-sonnet"]
    assert sorted(model_families([s for s in _samples_of(_bock_aftereffect("B"))
                                  if s.usable])) == ["claude-opus", "claude-sonnet"]


def _ensemble_cell(group: str, partial: list[dict], *, families: list[str]):
    """One `digitize:ensemble` candidate for `group`, carrying a real `partial_read` list.

    Built from the Bock record's own readings — `partial` comes out of `_partial_reads` over the
    real per-reader candidates, so nothing here invents the half-answer the rule is about.
    """
    from canopy.models import Candidate, DispersionType

    return Candidate(
        candidate_id=f"ens-{group}", dataset_id="b511dbb76fa6:d1", outcome_key="aftereffect",
        group=group, kind="group_stats", status="found", extractor_id="digitize:ensemble",
        source_kind="figure_bar", mean=-22.4, dispersion_value=8.3,
        dispersion_type=DispersionType.SD, n=12, locator="Fig. 1",
        pixel_provenance={"partial_read": partial, "model_families": families})


def _single_route_vote(candidate):
    from canopy.verify.vote import RouteValue, VoteResult

    return VoteResult(
        group=candidate.group, agreement="single", mean=candidate.mean,
        dispersion_value=candidate.dispersion_value, n=candidate.n, method="single",
        agreeing_ids=[candidate.candidate_id],
        routes=[RouteValue(route_key="figure/claude-opus", value=candidate.mean,
                           candidate_ids=[candidate.candidate_id])])


#: the sentence a genuinely single-family cell prints, in full — the thing C10 (b) says a partial
#: read must NOT print
SINGLE_FAMILY_LINE = "only one independent route produced this value"


def test_a_reader_that_came_back_without_a_mean_is_not_reported_as_a_missing_family():
    """C10 (b): Bock d1 aftereffect group A.

    Three readers read the figure; `claude-sonnet-5` returned `mean=None, error=6.5` for group A
    and a full reading for group B. The cell printed the same sentence a cell no second family
    ever looked at prints, which points a reviewer at the wrong repair: there IS a second family,
    it answered, and one number is missing from its answer.
    """
    from canopy.digitize.digitizer import _partial_reads
    from canopy.verify.confidence import confidence

    partial = _partial_reads(_samples_of(_bock_aftereffect("A")))
    assert [(x["group"], x["missing"]) for x in partial] == [("A", "mean")]   # the real record

    cand = _ensemble_cell("A", partial, families=["claude-opus"])
    _, _, reasons = confidence(_single_route_vote(cand), candidates=[cand])
    assert "a second model family read this figure and returned a spread but no mean for group A" \
        in " ".join(reasons)
    assert "claude-sonnet-5" in " ".join(reasons)          # WHICH reader to re-ask
    assert SINGLE_FAMILY_LINE not in " ".join(reasons)
    # exactly one reason line carries it, and it is the partial-read line — `next()` over a
    # mis-bound `or` used to stand here, and `assert line` on its result can never fail
    named = [r for r in reasons if "no mean" in r]
    assert len(named) == 1, reasons


def test_a_figure_with_no_whisker_to_read_still_prints_the_single_family_reason():
    """The other half, and the reason the rule is keyed on `missing == "mean"`.

    Cressman d1 aftereffect is a scatter of individual subjects: five routes read a mean and no
    error bar, because there is no error bar drawn. Keying the distinct reason on "a partial read
    exists" would tell a reviewer a second family came back empty about the MEAN on a pooled cell
    where every reader produced one.
    """
    from canopy.verify.confidence import confidence

    no_whisker = [{"route": f"digitize:readout:claude-sonnet-5:direct#{i}", "group": "A",
                   "missing": "error", "has": {"mean": 17.4}, "model": "claude-sonnet-5",
                   "variant": "direct", "sample": i} for i in range(5)]
    cand = _ensemble_cell("A", no_whisker, families=["claude-opus"])
    _, _, reasons = confidence(_single_route_vote(cand), candidates=[cand])
    assert SINGLE_FAMILY_LINE in " ".join(reasons)
    assert "no mean" not in " ".join(reasons)


def test_the_partial_read_of_another_group_does_not_speak_for_this_one():
    """Group B of the same cell was read in full by both families. A `partial_read` entry naming
    group A must not change what group B's cell says about itself."""
    from canopy.digitize.digitizer import _partial_reads
    from canopy.verify.confidence import confidence

    partial = _partial_reads(_samples_of(_bock_aftereffect("A")))
    cand = _ensemble_cell("B", partial, families=["claude-opus", "claude-sonnet"])
    _, _, reasons = confidence(_single_route_vote(cand), candidates=[cand])
    assert "no mean" not in " ".join(reasons)


# --------------------------------------------------------------------------- C1/C2: Heuer d1 late
#
# `runs/rerun-fixed/papers/3570e4ce2a9c` — Heuer & Hegele 2008, dataset d1, late adaptation, read
# off "Figure 2, panel a". The crop the run sent held 2 words and 0 numeric words, and panel a was
# not in it at all: the caption sat 296.2 pt below the panel and the 260 pt gate dropped it. Two
# readers said so in prose and returned nothing; the third returned numbers off the panels that
# WERE in the crop. Those numbers became the cell's candidates.
def _heuer_late(group: str) -> list[dict]:
    return [c for c in _record("heuer_late")
            if c["group"] == group and c["extractor_id"].startswith("digitize:readout")]


def test_the_heuer_record_shows_two_readers_saying_the_panel_is_not_there():
    """The evidence C2 turns into an action — today it exists only as free text."""
    notes = {c["extractor_id"]: c["notes"] for c in _heuer_late("A")}
    absent = [k for k, v in notes.items() if "not contained" in v or "not visible" in v]
    assert sorted(absent) == ["digitize:readout:claude-opus-5:direct",
                              "digitize:readout:claude-opus-5:ticks_first"]
    assert all(c["mean"] is None for c in _heuer_late("A") if c["extractor_id"] in absent)
    lone = next(c for c in _heuer_late("A") if c["extractor_id"] not in absent)
    assert lone["mean"] == 51.125
    assert "inferred calibration" in lone["notes"]      # it built its own ladder, and says so
    ensemble = next(c for c in _record("heuer_late")
                    if c["group"] == "A" and c["extractor_id"] == "digitize:ensemble")
    assert ensemble["mean"] == 51.125, "the minority reading IS the cell today"


def test_the_two_structured_fields_the_record_dictates_make_this_cell_abstain():
    """C2, replayed on the real reading. The enums are reconstructed from each reader's OWN
    sentence — "Panel 2a is not contained in this crop" is `target_visible: no`, "y-values
    estimated using inferred calibration" is `calibration_source: inferred` — which is the whole
    argument for having the enums: the sentences are already there and nothing can act on them.
    Either rule alone removes 51.125 from this cell; together they leave nothing to pool."""
    from canopy.digitize.digitizer import _mark_illegible, legibility
    from canopy.digitize.digitizer import RouteSample
    from canopy.digitize.vlm import ReadOut

    readouts, samples = [], []
    for cand in _heuer_late("A"):
        note = cand["notes"]
        visible = "no" if ("not contained" in note or "not visible" in note) else "yes"
        cal = "inferred" if "inferred calibration" in note else "printed_labels"
        model = cand["extractor_id"].split(":")[2]
        variant = cand["extractor_id"].split(":")[3]
        reading = ReadOut(model=model, variant=variant, target_visible=visible,
                          calibration_source=cal, notes=note)
        mine = [RouteSample(route="D", group="A", model=model, variant=variant,
                            mean=cand["mean"], error=cand["dispersion_value"], notes=note)]
        _mark_illegible(reading, mine)
        readouts.append(reading)
        samples.extend(mine)
    seen = legibility(readouts)
    assert seen["abstain"] is True and len(seen["target_not_visible"]) == 2
    assert seen["calibration_inferred"] == ["digitize:readout:claude-sonnet-5:direct"]
    kept = [s for s in samples if not s.dropped]
    assert kept == [], "51.125 was the only number here and it is not the cell's answer"


def test_the_deleted_keyword_rule_would_have_been_a_mass_abstain_switch():
    """Why C2 gates on the enum and not on prose.

    The decision's argument rests on a count over the WHOLE of this paper's extract —
    inferred/estimated/assumed 152 times against 115 for not-contained/not-visible — which this
    trimmed fixture cannot check: it holds the two cells the C2 tests need and, in them, the two
    families of phrase occur equally often. So the fixture's real counts are asserted here and
    the 152/115 claim is left where it can be checked, in the decision. What the fixture DOES
    prove is the thing the rule turns on: prose that says "estimated" off a printed ladder is a
    legitimate reading and survives."""
    import re

    raw = (RECORDS / "heuer_late" / "extract.json").read_text()
    hedge = len(re.findall(r"inferr|estimated|assumed", raw, re.I))
    absent = len(re.findall(r"not contained|not visible", raw, re.I))
    assert (hedge, absent) == (4, 4), (hedge, absent)
    from canopy.digitize.digitizer import _mark_illegible, RouteSample
    from canopy.digitize.vlm import ReadOut

    honest = ReadOut(model="m1", target_visible="yes", calibration_source="printed_labels",
                     notes="bar top estimated to the nearest half degree; caps assumed symmetric")
    mine = [RouteSample(route="D", group="A", model="m1", mean=31.5, error=11.0)]
    _mark_illegible(honest, mine)
    assert not mine[0].dropped, "prose is not evidence; the structured field is"


# --------------------------------------------------------------- D1: the override, on both paths
def test_the_precedence_override_row_is_the_same_row_on_the_run_and_the_re_pool_paths():
    """The standing rule, applied to D1: run and re-pool build the SAME row.

    Heuer d1 late adaptation is the cell the override exists for — the adjudicator kept the two
    printed numbers, the paper prints no spread for either, and the figure both readers measured
    sits in the same cell with a mean, an SE and an n. The run reaches it through
    `rows.prepare_rows → resolve.resolve_effect_with_fallback`; every human answer reaches it
    through `overrides._rebuild_row`, which calls the same two functions over the same cluster. If
    only one of them fell back, a reviewer confirming an unrelated cell would silently move the
    row's effect size — which is precisely the class of defect the one-row-path rule is for.

    The only fields allowed to differ are the ones an override IS: the human's flag and note.
    """
    from canopy.pipeline import overrides as ov
    from canopy.pipeline.resolve import resolve_effect_with_fallback
    from canopy.pipeline.rows import prepare_rows
    from canopy.pipeline.state import load_manifest
    from tests.helpers import nine

    protocol = nine.protocol()
    state = ov._RunState(nine.NINE, load_manifest(nine.NINE))   # read-only: nothing writes
    live = {(v.dataset_id, v.outcome_key, v.group): v for v in state.verdicts}
    ds_id, key = "3570e4ce2a9c:d1", "late_adaptation"
    dataset = state.datasets[ds_id]
    verdict_a, verdict_b = live[(ds_id, key, "A")], live[(ds_id, key, "B")]

    prepared = prepare_rows([(dataset, key, verdict_a, verdict_b)], state.candidates,
                            protocol.stats,
                            cluster_of=lambda d: d.cluster_id or state.paper_of[d.dataset_id])[0]
    ran = resolve_effect_with_fallback(dataset, protocol.outcome(key), prepared.values,
                                       prepared.alternatives, protocol.stats)
    assert ran.route == "figure" and "precedence_override" in ran.flags

    rebuilt = ov._rebuild_row(nine.record(ds_id, key), dataset, verdict_a, verdict_b, protocol,
                              key, "I opened Figure 2a and this is what it shows",
                              state=state, verdicts=live)
    was, now = ran.model_dump(mode="json"), rebuilt.model_dump(mode="json")
    for field in ("paper_id", "cluster_id", "sample_id", "citation", "label", "moderators",
                  "analysis_metric", "notes", "flags"):
        was.pop(field, None), now.pop(field, None)
    assert now == was
    assert set(rebuilt.flags) - set(ran.flags) == {"human_override"}


def test_a_reading_the_caption_set_aside_is_off_the_alternatives_on_both_paths():
    """Fix round 2, finding 5. Standing ruling (d) again: run and re-pool must offer the resolver
    the SAME alternatives.

    D2's caption check marks a reading whose panel the caption gives to the other group by writing
    `pixel_provenance["locator_dropped"]` on the candidate — in memory, on the objects
    `run._verify_cell` happens to hold. `run._verify` then writes the verdicts and its
    `extra_candidates`, and `extract.json` is never rewritten: every later reader (a re-pool after
    a human answer, `_Preview`'s pricing, a resumed `resolve`) loads the candidates from disk
    WITHOUT the mark and offers the fallback a pair the run had set aside. Measured on Langan d1
    late adaptation, where the re-pool offered two pairs and the run offered none.

    What survives the write is the verdict's `locator_reads_set_aside` flag, which names the very
    candidate ids the check dropped, so that is what both paths read now.
    """
    from canopy.pipeline import overrides as ov
    from canopy.pipeline.resolve import resolve_effect_with_fallback
    from canopy.pipeline.rows import prepare_rows
    from canopy.pipeline.state import load_manifest
    from canopy.verify.checks import CHECK_SEVERITY
    from canopy.verify.panels import LOCATOR_SET_ASIDE
    from canopy.verify.vote import LOCATOR_DROPPED
    from canopy.models import CheckFlag
    from tests.helpers import nine

    protocol = nine.protocol()
    state = ov._RunState(nine.NINE, load_manifest(nine.NINE))   # read-only: nothing writes
    live = {(v.dataset_id, v.outcome_key, v.group): v for v in state.verdicts}

    for ds_id, key in (("d1f2946e7e81:d1", "late_adaptation"), ("592b3b55a318:d2", "aftereffect")):
        dataset = state.datasets[ds_id]
        cell = [c for c in state.candidates
                if c.dataset_id == ds_id and c.outcome_key == key and c.group == "A"
                and c.extractor_id.startswith("digitize:")]
        assert cell, "the cell is read off a figure, so a panel can be given away"
        dropped = sorted({c.candidate_id for c in cell})
        flag = CheckFlag(code=LOCATOR_SET_ASIDE, severity=CHECK_SEVERITY[LOCATOR_SET_ASIDE],
                         message="the caption gives that panel to the other group",
                         candidate_ids=dropped)
        verdicts = {}
        for group in ("A", "B"):
            verdict = live[(ds_id, key, group)].model_copy(deep=True)
            verdict.flags = [*verdict.flags, flag]           # what `apply_panel_check` returns
            verdicts[group] = verdict

        # the RUN: the check marked the candidate objects it held, and the run resolved from those
        marked = []
        for cand in state.candidates:
            if cand.candidate_id not in dropped:
                marked.append(cand)
                continue
            copy = cand.model_copy(deep=True)
            copy.pixel_provenance = {**(copy.pixel_provenance or {}),
                                     LOCATOR_DROPPED: "the caption gives panel a to the other "
                                                      "group"}
            marked.append(copy)
        # the RE-POOL: the same verdicts, and the candidates as `extract.json` holds them
        cells = [(dataset, key, verdicts["A"], verdicts["B"])]
        ran = prepare_rows(cells, marked, protocol.stats)[0]
        repooled = prepare_rows(cells, state.candidates, protocol.stats)[0]

        assert [a.model_dump() for a in repooled.alternatives] \
            == [a.model_dump() for a in ran.alternatives], f"{ds_id}/{key} alternatives differ"
        assert all(a.group_a.candidate_id not in dropped for a in ran.alternatives)
        rows = [resolve_effect_with_fallback(dataset, protocol.outcome(key), row.values,
                                             row.alternatives, protocol.stats)
                for row in (ran, repooled)]
        assert (rows[0].es, rows[0].var, rows[0].flags) == (rows[1].es, rows[1].var, rows[1].flags)


# ------------------------------- D4-lite: an answered analysed n over a shared control arm
def _shared_row(*, flags: list[str], n_a: int = 19, n_b: int = 20):
    """One prepared row of a two-comparison cluster, at the sizes the mapper read."""
    from canopy.models import DatasetSpec
    from canopy.pipeline.resolve import GroupValues, ResolvedValues
    from canopy.pipeline.rows import PreparedRow

    values = ResolvedValues(dataset_id="p:d1", outcome_key="late_adaptation",
                            group_a=GroupValues(n=n_a), group_b=GroupValues(n=n_b), flags=flags)
    return PreparedRow(dataset=DatasetSpec(dataset_id="p:d1"), outcome_key="late_adaptation",
                       values=values)


def _answered(sizes=(18, 16), *, when: int = 1, per_cell: dict | None = None):
    """The run state as `_apply_group_n` reads it: the answered sizes, and WHEN each answer about
    a group size was made — a per-cell `n` answered later is not overwritten by this one."""
    from types import SimpleNamespace

    return SimpleNamespace(group_n={"p:d1": sizes}, group_n_seq={"p:d1": when},
                           n_answered=dict(per_cell or {}), keep_printed=set())


def test_an_answered_size_for_a_split_control_arm_is_re_split_not_handed_back_whole():
    """Cochrane 16.5.4 survives the answer: the arm two comparisons share still contributes n/k.

    Written against `resolve.SHARED_CONTROL_ARM` rather than a literal "B", because that constant
    is the whole of the coupling — `rows.prepare_rows` hands it to `apply_shared_control` and this
    function has to undo exactly the arm it adjusted (review finding 3).
    """
    from canopy.pipeline.overrides import _apply_group_n
    from canopy.pipeline.resolve import SHARED_CONTROL_ARM
    from canopy.stats.conversions import split_control

    answered = {"A": 18, "B": 16}
    shared = SHARED_CONTROL_ARM
    other = "A" if shared == "B" else "B"

    row = _shared_row(flags=["shared_control_split"])
    _apply_group_n(row, _answered(), 2)
    assert row.values.group(shared).n == int(round(split_control(answered[shared], 2)))
    assert row.values.group(other).n == answered[other]


def test_a_merged_control_arm_keeps_the_n_the_strategy_built():
    """Under `combine_arms` the surviving row's other arm is two arms added together, and a
    per-arm size a reviewer answered is not that number — so it is left alone."""
    from canopy.pipeline.overrides import _apply_group_n
    from canopy.pipeline.resolve import SHARED_CONTROL_ARM

    shared = SHARED_CONTROL_ARM
    other = "A" if shared == "B" else "B"
    row = _shared_row(flags=["shared_control_combined"])
    was = row.values.group(other).n

    _apply_group_n(row, _answered(), 1)
    assert row.values.group(other).n == was, "a merged arm is not one arm"
    assert row.values.group(shared).n == {"A": 18, "B": 16}[shared]


def test_a_row_that_shares_no_control_simply_takes_the_answered_sizes():
    from canopy.pipeline.overrides import _apply_group_n

    row = _shared_row(flags=[])
    _apply_group_n(row, _answered(), 1)
    assert (row.values.group_a.n, row.values.group_b.n) == (18, 16)
