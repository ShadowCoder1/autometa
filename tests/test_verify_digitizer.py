"""The verification layer against candidates a REAL digitizer run produced.

Everything else in `tests/test_checks.py`, `tests/test_vote.py` and `tests/test_confidence.py`
hand-builds its candidates, which is how a layer ends up expecting keys the producer never writes.
This file closes that loop: it runs `canopy.digitize.digitize()` on the synthetic bar chart from
`tests/test_digitizer.py` (whose bar tops and ticks are known exactly through matplotlib's
transform, with a scripted `FakeProvider` standing in for the models) and pushes the candidates it
emits through `run_checks` → `vote` → `confidence`.

It caught a real bug: the verification layer used to look for `y_min` / `y_max` / `y_tick` inside
`pixel_provenance["cal"]`, and the digitizer writes none of those — it writes `cal.ticks` as
`[pixel, value]` pairs, an `axis_range`, and, on the ensemble candidate, the dual tolerance it
already computed under `agreement.mean_tolerance`. Every figure tolerance was silently falling
back to 2% of the value and `value_outside_axis` could never fire.
"""
from __future__ import annotations

import pytest

from canopy.digitize.digitizer import digitize
from canopy.digitize.vlm import FigureView
from canopy.models import DatasetSpec, GroupSpec, OutcomeSources, Source, SourceKind
from canopy.verify.checks import codes, run_checks
from canopy.verify.confidence import confidence, figure_gate
from canopy.verify.figures import axis_limits, figure_calibration, figure_tolerance
from canopy.verify.vote import route_key, vote

# the synthetic figure, its paper record and the scripted model come from the digitizer's own tests
from tests.test_digitizer import (DATASET, SOURCE, TARGET, _client, _coord_payload, _paper_for,
                                  _readout_payload, _scripted, bar_figure)  # noqa: F401

TICK_LOW, TICK_HIGH, TICK_STEP = 0.0, 60.0, 10.0        # ax.set_ylim(0, 60), ticks every 10


@pytest.fixture(scope="module")
def digitised(bar_figure, tmp_path_factory) -> list:
    """One real `digitize()` run: five per-route candidates and an ensemble, per group."""
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                         _coord_payload(bar_figure, view.scale))
    return digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                    out_dir=tmp_path_factory.mktemp("verify_digitizer"))


def dataset_with_outcome() -> DatasetSpec:
    return DatasetSpec(
        dataset_id="ds1", cluster_id="paper1",
        group_a=GroupSpec(label="old", n=10), group_b=GroupSpec(label="young", n=12),
        outcomes=[OutcomeSources(
            outcome_key="late_adaptation", units="deg", higher_is_better=False,
            sources=[Source(kind=SourceKind.figure_bar, page=1, figure_id="fig01",
                            locator="Fig 1", error_bar_type=SOURCE.error_bar_type,
                            error_bar_agreement="agreed")])])


# --------------------------------------------------------------------------- the real keys
def test_the_calibration_parser_reads_what_the_digitizer_actually_writes(digitised):
    per_route = [c for c in digitised if c.extractor_id != "digitize:ensemble"]
    assert per_route, "the digitizer produced no per-route candidates"
    for cand in per_route:
        cal = figure_calibration(cand.pixel_provenance)
        assert cal.source == "cal_ticks", cand.extractor_id
        assert (cal.low, cal.high) == (TICK_LOW, TICK_HIGH)
        assert cal.tick == pytest.approx(TICK_STEP)
        assert cal.span == pytest.approx(TICK_HIGH - TICK_LOW, rel=0.02)


def test_the_ensemble_carries_the_digitizers_own_dual_tolerance(digitised):
    """`agreement.mean_tolerance` IS max(2% of the axis range, half a tick); when the digitizer
    has already computed it we use its number rather than recomputing a slightly different one."""
    ensemble = next(c for c in digitised if c.extractor_id == "digitize:ensemble")
    cal = figure_calibration(ensemble.pixel_provenance)
    assert cal.source == "agreement"
    assert cal.mean_tolerance == pytest.approx(
        ensemble.pixel_provenance["agreement"]["mean_tolerance"])
    assert figure_tolerance(ensemble) == pytest.approx(max(0.02 * cal.span, 0.5 * TICK_STEP))


def test_every_route_agrees_on_the_tolerance(digitised):
    """Whether a candidate carries the digitizer's own `agreement` block or only `cal` + the axis
    range, the tolerance the verification layer derives is the same number."""
    tolerances = sorted({round(figure_tolerance(c), 9) for c in digitised})
    assert tolerances == [pytest.approx(max(0.02 * 60.26, 0.5 * TICK_STEP), rel=1e-3)]


def test_the_axis_limits_come_from_the_labelled_ticks_with_the_plot_box_as_slack(digitised):
    low, high = axis_limits(digitised[0].pixel_provenance)
    assert low <= TICK_LOW and high >= TICK_HIGH
    assert low == pytest.approx(TICK_LOW, abs=1.0) and high == pytest.approx(TICK_HIGH, abs=1.0)


# --------------------------------------------------------------------------- checks
def test_real_digitised_values_raise_no_axis_or_dispersion_flags(digitised):
    flags = run_checks(dataset_with_outcome(), "late_adaptation", digitised)
    assert "value_outside_axis" not in codes(flags)
    assert "dispersion_unknown" not in codes(flags)
    assert "figure_error_bar_unknown" not in codes(flags)
    assert "figure_n_mismatch" not in codes(flags)


def test_a_value_off_the_real_axis_is_caught(digitised):
    off = digitised[0].model_copy(update={"candidate_id": "off", "mean": 500.0})
    flags = run_checks(dataset_with_outcome(), "late_adaptation", [*digitised, off])
    flag = next(f for f in flags if f.code == "value_outside_axis")
    assert flag.candidate_ids == ["off"] and "60" in flag.message


# --------------------------------------------------------------------------- vote
def test_the_digitizer_routes_vote_as_separate_voters(digitised):
    result = vote(digitised, group="A")
    keys = {r.route_key for r in result.routes}
    assert len({route_key(c) for c in digitised if c.group == "A"}) == len(keys)
    assert any(k.startswith("figure:readout") for k in keys)
    assert any(k.startswith("figure:vlm_coords") for k in keys)
    assert any(k.startswith("figure:raster_cv") for k in keys)
    assert any(k.startswith("figure:ensemble") for k in keys)


def test_the_real_routes_agree_within_the_real_tolerance(digitised):
    result = vote(digitised, group="A")
    assert result.agreement == "agree" and result.method == "figure_tolerance"
    assert result.tolerance == pytest.approx(5.0, rel=1e-3)
    assert result.mean == pytest.approx(31.5, abs=1.0)
    assert result.dispersion_value == pytest.approx(11.0, abs=1.5)
    assert result.needs_third_candidate is False
    assert result.n == 10 and result.unit == "deg"


# --------------------------------------------------------------------------- amendment F gate
def test_the_figure_gate_runs_on_real_provenance(digitised):
    ok, delta, share, reasons = figure_gate(digitised, 10, 12)
    assert delta is not None, reasons          # several routes read BOTH groups
    assert delta < 0.1, reasons                # they were fed the true pixels, so they agree
    assert share is not None and share >= 0.0


def test_a_digitised_cell_is_scored_with_the_gate(digitised):
    from canopy.models import OrientationVerdict

    oriented = OrientationVerdict(outcome_key="late_adaptation", higher_is_better=False,
                                  agreed=True, needs_human=False)
    flags = run_checks(dataset_with_outcome(), "late_adaptation", digitised)
    bucket, score, reasons = confidence(vote(digitised, group="A"), [], flags, None,
                                        candidates=digitised, n_a=10, n_b=12,
                                        orientation=oriented)
    assert bucket in ("auto_accept", "accept_with_note")
    assert not [r for r in reasons if "could not be compared" in r]
    assert score > 0.5
