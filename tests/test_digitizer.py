"""Task 6b: the vision routes and `digitize()`.

Two kinds of test here:

* **synthetic** — a matplotlib bar chart whose bar tops, error-bar caps and y ticks are known
  exactly through `ax.transData`. A scripted `FakeProvider` answers each tool loop with the TRUE
  pixel coordinates, so paths B and C are exercised end to end without a model in the loop: what
  is under test is the calibration/snap/ensemble code, not the model.
* **replay** — the real Bock 2005 Fig. 1 crop with recorded fixtures, compared against the human
  WebPlotDigitizer read (young 12.28 +/- 11.82, old 31.51 +/- 11.12). The read-out fixtures are
  recorded; the coords/verify ones are NOT, because the API account ran out of credit part-way
  through the recording run. Those tests skip (loudly) until someone re-runs
  `CANOPY_LIVE=1 CANOPY_RECORD=1 pytest tests/test_digitizer.py -k bock` with a funded key —
  recorded turns replay for free, so the re-run only pays for what is still missing.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from canopy.config import live_enabled, load_env, record_enabled
from canopy.digitize.digitizer import (ReadoutSpec, RouteSample, digitize, dual_tolerance,
                                       ensemble_stats, _absent_status, _drop_zero_confidence,
                                       _legend_dispersion, _overlay_marks, _readout_plan)
from canopy.digitize.vlm import (FigureView, MAX_ZOOM, READOUT_SCHEMA, READOUT_VARIANTS,
                                 TargetSpec, coords, overlay_verify, read_out, render_prompt)
from canopy.ingest.pdf import Bbox, FigureRegion, PaperRecord, ingest_pdf
from canopy.llm.client import LLMClient, MissingFixture
from canopy.llm.providers import FakeProvider
from canopy.llm.schemas import assert_no_derived_stats, assert_valid_output_schema
from canopy.models import DatasetSpec, DispersionType, GroupSpec, Source, SourceKind

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPLAY = FIXTURES / "llm"
PDF_DIR = FIXTURES / "pdfs"

# Bock 2005 Fig. 1, adaptation phase, last pointing episode (WebPlotDigitizer, human):
HUMAN = {"B": (12.28, 11.82), "A": (31.51, 11.12)}     # A = older, B = young
MEAN_TOL, SD_TOL = 1.0, 1.5


# ------------------------------------------------------------------ synthetic figure
def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@pytest.fixture(scope="module")
def bar_figure(tmp_path_factory) -> dict:
    """Two bars with error bars; truth = plotted values and their exact image pixels."""
    plt = _mpl()
    values = {"A": 31.5, "B": 12.25}
    errors = {"A": 11.0, "B": 11.75}
    dpi = 150
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=dpi)
    bars = ax.bar([0, 1], [values["A"], values["B"]], width=0.5,
                  yerr=[errors["A"], errors["B"]], capsize=8,
                  color=["#333333", "#bbbbbb"], ecolor="#000000")
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["old", "young"])
    ax.set_ylabel("error (deg)")
    fig.canvas.draw()
    out = tmp_path_factory.mktemp("digitizer")
    path = out / "figures" / "fig01.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    h_px = int(round(fig.get_size_inches()[1] * dpi))

    def row(value: float) -> float:
        return h_px - ax.transData.transform((0, value))[1]

    truth = {
        "path": path, "out_dir": out, "values": values, "errors": errors, "height": h_px,
        "ticks": [(row(v), v) for v in ax.get_yticks()],
        "bars": {},
    }
    for key, patch in zip(("A", "B"), bars):
        bb = patch.get_window_extent()
        truth["bars"][key] = {
            "x_center": (bb.x0 + bb.x1) / 2.0, "x0": bb.x0, "x1": bb.x1,
            "top": row(values[key]),
            "cap_top": row(values[key] + errors[key]),
            "cap_bottom": row(values[key] - errors[key]),
        }
    plt.close(fig)
    return truth


def _paper_for(truth: dict, kind: str = "raster") -> tuple[PaperRecord, FigureRegion]:
    fig = FigureRegion(
        id="fig01", page=1, bbox=Bbox(0.0, 0.0, 360.0, 288.0),
        caption="Fig. 1 Pointing error of old and young subjects. Bars are means, "
                "whiskers are standard deviations.",
        label="Fig. 1", kind=kind, n_images=1, n_drawings=0, native_px=None,
        crop_png="figures/fig01.png", claude_png="figures/fig01.png",
        crop_dpi=150.0, claude_scale=1.0, confidence=0.9)
    paper = PaperRecord(
        sha256="0" * 64, source_path="synthetic.pdf", filename="synthetic.pdf", n_pages=1,
        title="synthetic", doi="", first_page_text="", pages=[], figures=[fig], tables=[],
        has_text_layer=False, out_dir=str(truth["out_dir"]))
    return paper, fig


TARGET = TargetSpec(outcome_key="late_adaptation", group_a_label="old", group_b_label="young",
                    series_hint="dark bar = old, light bar = young",
                    x_hint="the plotted bars", panel_hint="Fig 1",
                    error_bar_type_hint="SD", unit_hint="deg",
                    late_window_sd="block_closest_to_end")
SOURCE = Source(kind=SourceKind.figure_bar, page=1, locator="Fig 1",
                figure_id="fig01", error_bar_type=DispersionType.SD)
DATASET = DatasetSpec(dataset_id="ds1", group_a=GroupSpec(label="old", n=10, n_evidence="n=10"),
                      group_b=GroupSpec(label="young", n=12, n_evidence="n=12"))


# ------------------------------------------------------------------ scripted model
def _submit(payload: dict) -> list[dict]:
    return [{"type": "tool_use", "id": "toolu_submit", "name": "submit", "input": payload}]


def _system_of(request) -> str:
    return request.system if isinstance(request.system, str) else json.dumps(request.system)


def _scripted(readout: dict, coord: dict, overlay: dict | None = None,
              readout_by_call: list[dict] | None = None):
    """One FakeProvider payload that answers each of the three prompts with canned JSON."""
    state = {"readouts": list(readout_by_call or [])}

    def respond(request):
        system = _system_of(request)
        if "read numeric values" in system:
            payload = state["readouts"].pop(0) if state["readouts"] else readout
        elif "locate features" in system:
            payload = coord
        else:
            payload = overlay if overlay is not None else {"marks": [], "notes": ""}
        return _submit(payload)

    return FakeProvider([respond])


def _client(provider: FakeProvider) -> LLMClient:
    return LLMClient(provider=provider, cache_dir=None)


def _readout_payload(a_mean: float | None, a_err: float | None, b_mean: float | None,
                     b_err: float | None, legend: str = "whiskers are standard deviations",
                     status: str = "found") -> dict:
    def cap(mean, err, sign):
        return None if mean is None or err is None else mean + sign * err

    return {
        "status": status, "panel": "Fig 1", "unit": "deg", "legend_says": legend,
        "tick_labels": [0, 10, 20, 30, 40, 50, 60], "pixel_resolution_estimate": 0.1,
        "confidence": 0.85, "notes": "",
        "groups": [
            {"group": "A", "label_read": "dark bar", "mean": a_mean,
             "error_half_length": a_err, "error_upper": cap(a_mean, a_err, 1),
             "error_lower": cap(a_mean, a_err, -1), "x_read": "old",
             "confidence": 0.85, "notes": ""},
            {"group": "B", "label_read": "light bar", "mean": b_mean,
             "error_half_length": b_err, "error_upper": cap(b_mean, b_err, 1),
             "error_lower": cap(b_mean, b_err, -1), "x_read": "young",
             "confidence": 0.85, "notes": ""},
        ],
    }


def _coord_payload(truth: dict, scale: float, jitter: float = 0.0) -> dict:
    def sent(px: float) -> float:
        return px * scale

    groups = []
    for key in ("A", "B"):
        bar = truth["bars"][key]
        groups.append({
            "group": key, "label_read": "dark bar" if key == "A" else "light bar",
            "x_px": sent(bar["x_center"]), "y_px": sent(bar["top"] + jitter),
            "bar_x0_px": sent(bar["x0"]), "bar_x1_px": sent(bar["x1"]),
            "cap_top_px": sent(bar["cap_top"]), "cap_bottom_px": sent(bar["cap_bottom"]),
            "notes": ""})
    return {"status": "found", "panel": "Fig 1", "unit": "deg", "confidence": 0.9, "notes": "",
            "ticks": [{"value": v, "y_px": sent(row)} for row, v in truth["ticks"]],
            "groups": groups}


# ------------------------------------------------------------------ schemas / prompts
def test_digitizer_schemas_are_valid_and_carry_no_derived_stats():
    assert_valid_output_schema(READOUT_SCHEMA)
    assert_no_derived_stats(READOUT_SCHEMA)
    submit = {"name": "submit", "strict": True, "input_schema": READOUT_SCHEMA}
    assert submit["input_schema"]["additionalProperties"] is False
    assert set(submit["input_schema"]["required"]) == set(READOUT_SCHEMA["properties"])


def test_prompts_render_with_the_target_and_refuse_unfilled_placeholders():
    text = render_prompt("digitize_readout", TARGET=TARGET.describe(), CAPTION="cap", VARIANT="")
    assert "late_adaptation" in text and "block_closest_to_end" in text
    with pytest.raises(KeyError):
        render_prompt("digitize_readout", TARGET="t")


def test_readout_plan_is_two_model_families_and_two_variants():
    assert _readout_plan(("claude-opus-5",), 3) == [
        ReadoutSpec("claude-opus-5", "direct"), ReadoutSpec("claude-opus-5", "ticks_first"),
        ReadoutSpec("claude-sonnet-5", "direct")]
    assert _readout_plan(("claude-opus-5",), 0) == []


@pytest.mark.parametrize("models", [("claude-opus-5",), ("claude-opus-5", "claude-sonnet-5"),
                                    ("claude-opus-5", "claude-haiku-4-5")])
@pytest.mark.parametrize("n", [1, 3, 5, 6, 7, 11])
def test_readout_plan_never_asks_the_same_prompt_twice_unflagged(models, n):
    plan = _readout_plan(models, n)
    assert len(plan) == n
    keys = [(s.model, s.variant, s.sample) for s in plan]
    assert len(set(keys)) == n, f"duplicate sample in {keys}"
    firsts = [(s.model, s.variant) for s in plan if not s.resample]
    assert len(set(firsts)) == len(firsts), "a (model, variant) pair was used twice unflagged"
    # re-samples only appear once every distinct pair is spent
    if any(s.resample for s in plan):
        distinct = len({(s.model, s.variant) for s in plan})
        assert len(firsts) == distinct


def test_readout_plan_spends_every_variant_before_repeating_a_prompt():
    plan = _readout_plan(("claude-opus-5",), 6)
    assert {s.variant for s in plan} == set(READOUT_VARIANTS)
    assert not any(s.resample for s in plan)
    assert _readout_plan(("claude-opus-5",), 7)[-1].resample is True


# ------------------------------------------------------------------ FigureView / zoom tools
def test_figure_view_maps_sent_pixels_back_to_crop_pixels(bar_figure):
    view = FigureView(bar_figure["path"])
    assert view.scale > 0
    x, y = view.to_sent(100.0, 50.0)
    assert view.to_crop(x, y) == pytest.approx((100.0, 50.0))


def test_crop_image_tool_returns_an_image_and_its_mapping(bar_figure):
    view = FigureView(bar_figure["path"])
    w, h = view.image.size
    blocks = view.handle_crop_image({"x0": 10, "y0": 20, "x1": 110, "y1": 120, "zoom": 3})
    assert [b["type"] for b in blocks] == ["image", "text"]
    note = blocks[1]["text"]
    assert "[10..110]" in note and "[20..120]" in note
    zoom = float(note.split("at zoom ")[1].split("x.")[0])
    assert 1.0 < zoom <= MAX_ZOOM + 1e-6
    assert view.handle_crop_image({"x0": 0, "y0": 0, "x1": w, "y1": h})     # full frame is legal


def test_crop_image_tool_rejects_a_degenerate_or_off_image_box(bar_figure):
    view = FigureView(bar_figure["path"])
    with pytest.raises(ValueError, match="at least"):
        view.handle_crop_image({"x0": 10, "y0": 10, "x1": 12, "y1": 200})
    with pytest.raises(ValueError, match="numeric"):
        view.handle_crop_image({"x0": 10, "y0": 10, "x1": None, "y1": 200})


def test_list_regions_reports_sent_pixel_coordinates_of_the_cv_pass(bar_figure):
    view = FigureView(bar_figure["path"])
    text = view.handle_list_regions({})
    assert "image you were sent" in text
    assert "y axis" in text and "x axis" in text
    assert "bar #" in text or "bar " in text
    axis_x = float(text.split("y axis (vertical line) at x = ")[1].split("\n")[0])
    # the y axis is at the left edge of the plot box, scaled into the sent image
    assert axis_x == pytest.approx(view.to_sent(*view.to_crop(axis_x, 0))[0], abs=1e-6)
    assert 0 < axis_x < view.image.size[0]


def test_overlay_points_tool_draws_and_returns_the_figure(bar_figure, tmp_path):
    view = FigureView(bar_figure["path"], work_dir=tmp_path)
    blocks = view.handle_overlay_points({"points": [{"x": 100, "y": 100, "label": "old mean"}]})
    assert [b["type"] for b in blocks] == ["image", "text"]
    assert "old mean" in blocks[1]["text"]
    assert list(tmp_path.glob("*model_overlay1.png"))
    with pytest.raises(ValueError):
        view.handle_overlay_points({"points": []})


# ------------------------------------------------------------------ the three vision calls
def test_read_out_parses_the_submitted_values(bar_figure):
    client = _client(_scripted(_readout_payload(31.5, 11.0, 12.25, 11.75), {}))
    reading = read_out(client, bar_figure["path"], "cap", TARGET, "claude-opus-5", "ticks_first")
    assert reading.status == "found" and reading.variant == "ticks_first"
    assert reading.group("A").mean == 31.5 and reading.group("B").error_half_length == 11.75
    assert reading.tick_labels[-1] == 60
    assert reading.call_ids and reading.cost_usd > 0


def test_read_out_falls_back_to_the_cap_values_when_no_half_length_is_given(bar_figure):
    payload = _readout_payload(31.5, 11.0, 12.25, 11.75)
    for row in payload["groups"]:
        row["error_half_length"] = None
    client = _client(_scripted(payload, {}))
    reading = read_out(client, bar_figure["path"], "cap", TARGET)
    from canopy.digitize.digitizer import _samples_from_readout
    samples = {s.group: s for s in _samples_from_readout(reading)}
    assert samples["A"].error == pytest.approx(11.0)
    assert samples["B"].error == pytest.approx(11.75)


def test_read_out_rejects_an_unknown_variant(bar_figure):
    with pytest.raises(ValueError, match="variant"):
        read_out(_client(_scripted({}, {})), bar_figure["path"], "", TARGET, variant="nope")


def test_coords_maps_sent_pixels_into_crop_pixels(bar_figure):
    view = FigureView(bar_figure["path"])
    payload = _coord_payload(bar_figure, view.scale)
    client = _client(_scripted({}, payload))
    out = coords(client, bar_figure["path"], "cap", TARGET, view=view)
    assert out.scale == pytest.approx(view.scale)
    assert out.group("A").y_px == pytest.approx(bar_figure["bars"]["A"]["top"], abs=1e-6)
    assert out.ticks[0].y_px == pytest.approx(bar_figure["ticks"][0][0], abs=1e-6)


def test_overlay_verify_returns_one_verdict_per_mark_and_the_overlay_path(bar_figure, tmp_path):
    marks = [{"x": 100.0, "y": 100.0, "label": "group A mean"},
             {"x": 200.0, "y": 150.0, "label": "group B mean"}]
    client = _client(_scripted({}, {}, overlay={"marks": [
        {"number": 1, "verdict": "ok", "reason": "on the bar top"},
        {"number": 2, "verdict": "not_on_datum", "reason": "sits on the cap"}], "notes": ""}))
    verdict = overlay_verify(client, bar_figure["path"], marks, TARGET,
                             out_png=tmp_path / "ov.png")
    assert [m.verdict for m in verdict] == ["ok", "not_on_datum"]
    assert [m.number for m in verdict.mismatches] == [2]
    assert Path(verdict.overlay_path).exists()
    assert verdict.call_ids


def test_overlay_verify_marks_unjudged_numbers_unknown(bar_figure, tmp_path):
    marks = [{"x": 10.0, "y": 10.0, "label": "one"}, {"x": 20.0, "y": 20.0, "label": "two"}]
    client = _client(_scripted({}, {}, overlay={
        "marks": [{"number": 1, "verdict": "ok", "reason": ""},
                  {"number": 9, "verdict": "ok", "reason": "no such mark"}], "notes": ""}))
    verdict = overlay_verify(client, bar_figure["path"], marks, TARGET,
                             out_png=tmp_path / "ov.png")
    assert [(m.number, m.verdict) for m in verdict] == [(1, "ok"), (2, "unknown")]


def test_overlay_verify_with_no_marks_makes_no_call(bar_figure):
    provider = FakeProvider([])
    verdict = overlay_verify(_client(provider), bar_figure["path"], [], TARGET)
    assert list(verdict) == [] and provider.requests == []


# ------------------------------------------------------------------ ensemble maths
def test_ensemble_stats_is_the_median_and_a_robust_spread():
    assert ensemble_stats([10.0, 10.0, 10.0]) == (10.0, 0.0)
    med, sigma = ensemble_stats([10.0, 11.0, 12.0, 40.0])
    assert med == pytest.approx(11.5)                       # the blunder does not move the median
    assert sigma == pytest.approx(1.4826 * 1.0)
    with pytest.raises(ValueError):
        ensemble_stats([])


def test_dual_tolerance_uses_the_larger_of_two_percent_and_half_a_tick():
    ok = dual_tolerance([30.0, 30.4], [11.0, 11.2], axis_range=60, tick_spacing=10, px_units=0.1)
    assert ok["agrees"] and ok["mean_tolerance"] == pytest.approx(5.0)      # 0.5 x 10 tick beats 1.2
    bad = dual_tolerance([30.0, 38.0], [11.0, 11.2], axis_range=60, tick_spacing=10, px_units=0.1)
    assert not bad["agrees"] and not bad["mean_agrees"] and "means span" in bad["reasons"][0]


def test_dual_tolerance_flags_error_bars_that_disagree_by_more_than_ten_percent():
    out = dual_tolerance([30.0, 30.1], [11.0, 14.0], axis_range=60, tick_spacing=10, px_units=0.1)
    assert out["mean_agrees"] and not out["error_agrees"]
    assert "error half-lengths span" in out["reasons"][0]


def test_dual_tolerance_of_a_single_route_cannot_disagree():
    assert dual_tolerance([30.0], [11.0], axis_range=60, tick_spacing=10, px_units=0.1)["agrees"]


def test_legend_reading_is_mapped_to_a_dispersion_type():
    assert _legend_dispersion("error bars are standard deviations") is DispersionType.SD
    assert _legend_dispersion("whiskers show the standard error of the mean") is DispersionType.SE
    assert _legend_dispersion("bars show 95% confidence intervals") is DispersionType.CI95
    assert _legend_dispersion("bars show something") is None


def test_overlay_marks_merge_routes_that_agree_to_the_pixel():
    samples = [RouteSample(route="D", group="A", model="claude-opus-5", variant="direct",
                           mean=30.0, x_px=100.0, y_px=200.0),
               RouteSample(route="C", group="A", mean=30.01, x_px=100.4, y_px=200.3),
               RouteSample(route="C", group="B", mean=12.0, x_px=300.0, y_px=400.0)]

    class _Axes:
        plot_bbox = (0.0, 0.0, 500.0, 500.0)

    class _Core:
        axes = _Axes()

    marks, owners = _overlay_marks(samples, None, _Core(), {"A": "old", "B": "young"})
    assert len(marks) == 2
    assert owners[0] == [0, 1] and owners[1] == [2]
    assert "old" in marks[0]["label"] and "readout" in marks[0]["label"]


# ------------------------------------------------------------------ digitize() on the synthetic
def test_digitize_recovers_the_plotted_values_through_paths_b_and_c(bar_figure, tmp_path):
    """The FakeProvider hands back the TRUE pixels, so any error here is in our code."""
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(_readout_payload(31.0, 11.0, 12.0, 12.0),
                         _coord_payload(bar_figure, view.scale))
    out = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                   out_dir=tmp_path, result=True)
    by_route = {}
    for s in out.samples:
        by_route.setdefault((s.route, s.group), s)
    assert {"C", "D", "B"} <= {r for r, _ in by_route}, "paths B, C and D must all vote"
    for group, value in (("A", 31.5), ("B", 12.25)):
        for route in ("B", "C"):
            sample = by_route[(route, group)]
            assert sample.mean == pytest.approx(value, abs=0.3), (route, group)
            assert sample.error == pytest.approx(bar_figure["errors"][group], abs=0.5)
    ensemble = {c.group: c for c in out.candidates if c.extractor_id == "digitize:ensemble"}
    assert ensemble["A"].mean == pytest.approx(31.5, abs=MEAN_TOL)
    assert ensemble["B"].mean == pytest.approx(12.25, abs=MEAN_TOL)
    assert ensemble["A"].dispersion_value == pytest.approx(11.0, abs=SD_TOL)
    assert ensemble["A"].status == "found"
    assert ensemble["A"].sigma is not None and ensemble["A"].sigma > 0


def test_digitize_emits_one_candidate_per_route_sample_plus_an_ensemble(bar_figure, tmp_path):
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(_readout_payload(31.4, 11.1, 12.3, 11.7),
                         _coord_payload(bar_figure, view.scale))
    candidates = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                          out_dir=tmp_path)
    per_route = [c for c in candidates if c.extractor_id != "digitize:ensemble"]
    ensembles = [c for c in candidates if c.extractor_id == "digitize:ensemble"]
    assert len(ensembles) == 2
    assert {c.group for c in candidates} == {"A", "B"}
    assert all(c.route == "figure" and c.kind == "group_stats" for c in candidates)
    assert all(c.source_kind is SourceKind.figure_bar for c in candidates)
    assert all(c.dispersion_type is DispersionType.SD for c in candidates)
    assert all(c.unit == "deg" and c.page == 1 and c.locator == "Fig 1" for c in candidates)
    assert {c.n for c in candidates if c.group == "A"} == {10}
    assert {c.n for c in candidates if c.group == "B"} == {12}
    assert any(c.extractor_id.startswith("digitize:readout:claude-opus-5:") for c in per_route)
    assert any(c.extractor_id == "digitize:vlm_coords:claude-opus-5" for c in per_route)
    assert any(c.extractor_id == "digitize:raster_cv" for c in per_route)
    assert all(c.prompt_version and c.crop_path for c in candidates)
    assert all(c.overlay_path and Path(c.overlay_path).exists() for c in candidates)
    prov = ensembles[0].pixel_provenance
    for key in ("cal", "axes", "ocr_status", "per_route", "route_values", "snap_confidences",
                "agreement", "overlay_iterations", "late_window_rule", "narrow_bar",
                "vector_warnings", "pixel_resolution", "n_routes"):
        assert key in prov, key
    assert prov["late_window_rule"] == "block_closest_to_end"
    assert prov["cal"]["axis"] == "y"


def test_digitize_flags_disagreement_between_routes_as_ambiguous(bar_figure, tmp_path):
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    good = _readout_payload(31.5, 11.0, 12.25, 11.75)
    provider = _scripted(good, _coord_payload(bar_figure, view.scale),
                         readout_by_call=[_readout_payload(45.0, 11.0, 12.25, 11.75), good, good])
    out = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                   out_dir=tmp_path, result=True)
    ensemble = {c.group: c for c in out.candidates if c.extractor_id == "digitize:ensemble"}
    assert ensemble["A"].status == "ambiguous"
    assert ensemble["A"].pixel_provenance["needs_review"] is True
    assert "means span" in ensemble["A"].pixel_provenance["needs_review_reason"]
    assert ensemble["B"].status == "found"
    assert ensemble["A"].mean == pytest.approx(31.5, abs=1.0)             # median resists it


def test_digitize_flags_a_legend_that_contradicts_the_mapper(bar_figure, tmp_path):
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(_readout_payload(31.5, 11.0, 12.25, 11.75,
                                          legend="whiskers are the standard error of the mean"),
                         _coord_payload(bar_figure, view.scale))
    out = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                   out_dir=tmp_path, result=True)
    ensemble = {c.group: c for c in out.candidates if c.extractor_id == "digitize:ensemble"}
    assert ensemble["A"].status == "ambiguous"
    assert "legend reads SE" in ensemble["A"].notes
    assert ensemble["A"].pixel_provenance["legend_dispersion"] == "SE"
    assert ensemble["A"].dispersion_type is DispersionType.SD     # the mapper still decides
    assert ensemble["A"].pixel_provenance["mapper_dispersion"] == "SD"


def test_digitize_drops_a_route_the_overlay_verifier_rejects_and_recomputes(bar_figure, tmp_path):
    """Path C is told a wrong y; the verifier condemns that mark; the ensemble must recover."""
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    wrong = _coord_payload(bar_figure, view.scale, jitter=-60.0)   # 60 px above the true tops
    seen = {"verify": 0, "condemned": []}

    def respond(request):
        system = _system_of(request)
        if "read numeric values" in system:
            return _submit(_readout_payload(31.5, 11.0, 12.25, 11.75))
        if "locate features" in system:
            return _submit(wrong)
        seen["verify"] += 1
        # the mark list is in the prompt, and each label names the routes behind that mark
        verdicts = []
        for line in system.splitlines():
            head, _, rest = line.partition(". ")
            if not head.strip().isdigit():
                continue
            number = int(head.strip())
            if "vlm_coords" in rest:
                seen["condemned"].append(number)
                verdicts.append({"number": number, "verdict": "not_on_datum",
                                 "reason": "floats well above the bar top"})
            else:
                verdicts.append({"number": number, "verdict": "ok", "reason": "on the bar top"})
        return _submit({"marks": verdicts, "notes": ""})

    out = digitize(_client(FakeProvider([respond])), paper, fig, TARGET, source=SOURCE,
                   dataset=DATASET, out_dir=tmp_path, result=True)
    dropped = [s for s in out.samples if s.dropped]
    assert dropped, "the verifier's mismatch must drop at least one route sample"
    assert all("overlay verify" in s.drop_reason for s in dropped)
    assert seen["verify"] >= 2, "a mismatch must trigger a recompute + re-verify"
    ensemble = {c.group: c for c in out.candidates if c.extractor_id == "digitize:ensemble"}
    assert ensemble["A"].mean == pytest.approx(31.5, abs=MEAN_TOL)
    assert ensemble["A"].pixel_provenance["dropped_samples"]
    assert len(out.provenance["overlay_iterations"]) >= 2


def test_digitize_flags_narrow_bars(tmp_path):
    plt = _mpl()
    fig_, ax = plt.subplots(figsize=(5.0, 4.0), dpi=100)
    ax.bar([0, 1], [30.0, 12.0], width=0.02, color=["#333333", "#bbbbbb"])
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 20, 40, 60])
    fig_.canvas.draw()
    out = tmp_path / "narrow"
    (out / "figures").mkdir(parents=True)
    fig_.savefig(out / "figures" / "fig01.png", dpi=100)
    plt.close(fig_)
    truth = {"out_dir": out, "path": out / "figures" / "fig01.png"}
    paper, fig = _paper_for(truth)
    provider = _scripted(_readout_payload(30.0, 0.0, 12.0, 0.0),
                         {"status": "found", "panel": "", "unit": "", "confidence": 0.5,
                          "notes": "", "ticks": [], "groups": []})
    res = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                   out_dir=out, result=True)
    from canopy.digitize.cv import MIN_BAR_WIDTH_PX
    bars = res.provenance["bars"]
    assert bars, "the narrow bars must still be detected, only flagged"
    assert all(b["x1"] - b["x0"] < MIN_BAR_WIDTH_PX for b in bars)
    assert res.provenance["narrow_bar"] is True


def test_digitize_reports_not_found_when_the_figure_holds_no_answer(bar_figure, tmp_path):
    paper, fig = _paper_for(bar_figure)
    empty = _readout_payload(None, None, None, None, status="not_on_these_pages")
    provider = _scripted(empty, {"status": "not_on_these_pages", "panel": "", "unit": "",
                                 "confidence": 0.0, "notes": "wrong panel", "ticks": [],
                                 "groups": []})
    candidates = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                          out_dir=tmp_path)
    ensemble = {c.group: c for c in candidates if c.extractor_id == "digitize:ensemble"}
    assert ensemble["A"].status == "not_on_these_pages" and ensemble["A"].mean is None
    assert ensemble["A"].pixel_provenance["needs_review"] is True


def test_digitize_never_emits_an_effect_size(bar_figure, tmp_path):
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    candidates = digitize(_client(_scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                                            _coord_payload(bar_figure, view.scale))),
                          paper, fig, TARGET, source=SOURCE, dataset=DATASET, out_dir=tmp_path)
    for c in candidates:
        assert c.reported_value is None and c.stat_value is None
        blob = json.dumps(c.model_dump(mode="json"))
        assert '"d":' not in blob and '"g":' not in blob


# ------------------------------------------------------------------ real figure, replay
@pytest.fixture(scope="session")
def bock(tmp_path_factory) -> PaperRecord:
    return ingest_pdf(PDF_DIR / "bock2005.pdf", tmp_path_factory.mktemp("bock"))


@pytest.fixture(scope="session")
def live_flags() -> tuple[bool, bool]:
    """Whether this run may call the API and record. Read here, not at import: conftest keeps
    `CANOPY_LIVE`/`CANOPY_RECORD` only for tests marked `live` or `replay`."""
    return live_enabled(), record_enabled()


@pytest.fixture()
def replay_client(live_flags) -> LLMClient:
    live, record = live_flags
    if live:
        load_env()
    return LLMClient(replay_dir=REPLAY, record_dir=REPLAY if record else None,
                     allow_live=live, cache_dir=None,
                     budget_usd=6.0 if live else None)    # recording guard; fixtures are free


BOCK_TARGET = TargetSpec(
    outcome_key="late_adaptation",
    group_a_label="older subjects", group_b_label="young subjects",
    series_hint="open symbols = young, filled symbols = old (see the legend)",
    x_hint="the last pointing episode of the adaptation phase (episode 20)",
    panel_hint="Fig. 1", quantity="mean_and_error", error_bar_type_hint="SD",
    unit_hint="deg", late_window_sd="block_closest_to_end")
BOCK_SOURCE = Source(kind=SourceKind.figure_line, page=3, locator="Fig. 1", figure_id="fig01",
                     error_bar_type=DispersionType.SD)
BOCK_DATASET = DatasetSpec(dataset_id="bock2005:main",
                           group_a=GroupSpec(label="old", n=10),
                           group_b=GroupSpec(label="young", n=10))


@pytest.mark.replay
def test_bock_fig1_read_outs_match_the_human_digitisation(bock, replay_client, tmp_path):
    """Path D on the real figure: every recorded read-out variant, against the human numbers."""
    fig = next(f for f in bock.figures if f.id == "fig01")
    assert fig.page == 3
    crop = Path(bock.out_dir) / fig.crop_png
    view = FigureView(crop, work_dir=tmp_path)
    readings = []
    missing = []
    for variant in ("direct", "ticks_first"):
        try:
            readings.append(read_out(replay_client, crop, fig.caption, BOCK_TARGET,
                                     "claude-opus-5", variant, view=view,
                                     cell_key=f"bock/fig01/D/{variant}"))
        except MissingFixture as exc:                       # see the module docstring
            missing.append(f"{variant}: {exc}")
    if not readings:
        pytest.skip("no recorded Bock read-out at all: " + "; ".join(missing))
    for reading in readings:
        assert reading.status == "found", reading.notes
        assert "standard deviation" in reading.legend_says.lower()
        assert set(reading.tick_labels) >= {-40.0, 0.0, 60.0}
        for group, (mean, sd) in HUMAN.items():
            row = reading.group(group)
            assert row is not None and row.mean is not None, (reading.variant, group)
            assert row.mean == pytest.approx(mean, abs=MEAN_TOL), (reading.variant, group,
                                                                   row.mean, row.notes)
            assert row.error_half_length == pytest.approx(sd, abs=SD_TOL), (
                reading.variant, group, row.error_half_length)
        assert any(c["name"] == "crop_image" for c in reading.tool_calls), "the model must zoom"
        assert reading.turns >= 2 and reading.call_ids


@pytest.mark.replay
def test_bock_fig1_matches_the_human_digitisation(bock, replay_client, tmp_path):
    """The whole of `digitize()` on the real figure (skips until the fixture set is complete)."""
    fig = next(f for f in bock.figures if f.id == "fig01")
    try:
        candidates = digitize(replay_client, bock, fig, BOCK_TARGET, source=BOCK_SOURCE,
                              dataset=BOCK_DATASET, out_dir=tmp_path)
    except MissingFixture as exc:                           # see the module docstring
        pytest.skip(f"Bock fixture set is incomplete — re-record with "
                    f"CANOPY_LIVE=1 CANOPY_RECORD=1: {exc}")
    ensemble = {c.group: c for c in candidates if c.extractor_id == "digitize:ensemble"}
    assert set(ensemble) == {"A", "B"}
    for group, (mean, sd) in HUMAN.items():
        got = ensemble[group]
        assert got.mean == pytest.approx(mean, abs=MEAN_TOL), (group, got.mean, got.notes)
        assert got.dispersion_value == pytest.approx(sd, abs=SD_TOL), (group, got.dispersion_value)
        assert got.sigma is not None and got.sigma == got.sigma and got.sigma >= 0
        assert got.crop_path and Path(got.overlay_path).exists()
        prov = got.pixel_provenance
        assert prov["ocr_status"] in ("ok", "missing", "failed", "timeout")
        assert prov["cal"] is not None and prov["n_routes"] >= 3
        assert prov["late_window_rule"] == "block_closest_to_end"
    per_route = [c for c in candidates if c.extractor_id != "digitize:ensemble"]
    assert len({c.extractor_id for c in per_route}) >= 4


# ------------------------------------------------------------------ item 1: fixture hygiene
#: Every recorded Bock fixture must be reachable by one of these documented calls. A fixture no
#: call can reach is an orphan: dead weight in the repo that also inflates the "already recorded"
#: figure in the report. Loops that are only partly recorded still reach their first N turns.
DOCUMENTED_BOCK_CALLS = [("read_out", "claude-opus-5", "direct"),
                         ("read_out", "claude-opus-5", "ticks_first")]


def _is_tool_loop_fixture(path: Path) -> bool:
    content = (json.loads(path.read_text()).get("response") or {}).get("content") or []
    return any(isinstance(b, dict) and b.get("type") == "tool_use" for b in content)


def test_every_recorded_fixture_is_reachable_by_a_documented_call(bock, tmp_path):
    # Deliberately NOT marked `replay`: it must only ever read fixtures (a live run here would
    # spend unbudgeted money continuing the ticks_first loop).
    # Only tool-loop fixtures belong to the digitizer: the same directory also holds the mapper /
    # extractor / verifier fixtures (plain structured calls, no `tool_use` block).
    committed = {path.stem for path in REPLAY.glob("*.json") if _is_tool_loop_fixture(path)}
    if not committed:
        pytest.skip("no digitizer fixtures recorded yet")
    fig = next(f for f in bock.figures if f.id == "fig01")
    crop = Path(bock.out_dir) / fig.crop_png
    reached: set[str] = set()
    for kind, model, variant in DOCUMENTED_BOCK_CALLS:
        assert kind == "read_out"
        client = LLMClient(replay_dir=REPLAY, cache_dir=None)
        view = FigureView(crop, work_dir=tmp_path / f"{model}-{variant}")
        try:
            read_out(client, crop, fig.caption, BOCK_TARGET, model, variant, view=view)
        except MissingFixture:
            pass                                    # a partly-recorded loop still reaches its head
        reached |= {call["key"] for call in client.calls()}
    orphans = sorted(committed - reached)
    assert not orphans, (
        f"{len(orphans)} recorded fixture(s) no documented call can reach: {orphans}. "
        f"Either delete them or add the call that uses them to DOCUMENTED_BOCK_CALLS.")


# ------------------------------------------------------------------ item 2: no duplicated votes
def test_digitize_gives_every_sample_a_distinct_extractor_and_candidate_id(bar_figure, tmp_path):
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                         _coord_payload(bar_figure, view.scale))
    out = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                   out_dir=tmp_path, n_readouts=7, result=True)          # forces re-samples
    ids = [c.candidate_id for c in out.candidates]
    assert len(set(ids)) == len(ids), "candidate ids collided"
    for group in ("A", "B"):
        mine = [s for s in out.samples if s.group == group]
        assert len({s.extractor_id for s in mine}) == len(mine), "extractor ids collided"
    resampled = [s for s in out.samples if s.route == "D" and s.sample > 0]
    assert resampled, "n_readouts=7 must exhaust the distinct prompts and re-sample"
    assert all(s.extra["same_prompt_resample"] for s in resampled)
    assert all(s.extractor_id.endswith(("#2", "#3")) for s in resampled)
    prov = next(c for c in out.candidates
                if c.extractor_id == "digitize:ensemble" and c.group == "A").pixel_provenance
    assert prov["resampled_routes"], "a re-sample must be visible to task 8"
    assert len(prov["readout_plan"]) == 7


# ------------------------------------------------------------------ item 3: absent vs unreadable
def test_absent_status_separates_not_on_the_page_from_unreadable():
    absent = [RouteSample(route="D", group="A", status="not_on_these_pages")]
    status, why = _absent_status(absent)
    assert status == "not_on_these_pages" and "not plotted" in why

    dropped = [RouteSample(route="C", group="A", mean=1.0, dropped=True, drop_reason="x")]
    status, why = _absent_status(dropped)
    assert status == "ambiguous" and "overlay verification" in why

    assert _absent_status([])[0] == "ambiguous"


def test_digitize_is_ambiguous_not_absent_when_verification_drops_every_route(bar_figure,
                                                                             tmp_path):
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])

    def respond(request):
        system = _system_of(request)
        if "read numeric values" in system:
            return _submit(_readout_payload(31.5, 11.0, 12.25, 11.75))
        if "locate features" in system:
            return _submit(_coord_payload(bar_figure, view.scale))
        verdicts = [{"number": int(line.split(".")[0]), "verdict": "wrong_series",
                     "reason": "that is the other group"}
                    for line in system.splitlines() if line.split(".")[0].strip().isdigit()]
        return _submit({"marks": verdicts, "notes": ""})

    out = digitize(_client(FakeProvider([respond])), paper, fig, TARGET, source=SOURCE,
                   dataset=DATASET, out_dir=tmp_path, result=True)
    ensemble = {c.group: c for c in out.candidates if c.extractor_id == "digitize:ensemble"}
    for group in ("A", "B"):
        assert ensemble[group].status == "ambiguous", "a failed read is not 'not on these pages'"
        assert ensemble[group].mean is None
        prov = ensemble[group].pixel_provenance
        assert prov["needs_review"] is True
        assert "overlay verification" in prov["needs_review_reason"]
        assert prov["dropped_samples"] and prov["tool_calls"]


# ------------------------------------------------------------------ items 4-7: provenance
def test_ensemble_provenance_carries_the_aggregated_tool_calls(bar_figure, tmp_path):
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    out = digitize(_client(_scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                                     _coord_payload(bar_figure, view.scale))),
                   paper, fig, TARGET, source=SOURCE, dataset=DATASET, out_dir=tmp_path,
                   result=True)
    prov = next(c for c in out.candidates
                if c.extractor_id == "digitize:ensemble" and c.group == "A").pixel_provenance
    calls = prov["tool_calls"]
    assert calls and all("route" in c and "name" in c for c in calls)
    assert {c["name"] for c in calls} == {"submit"}          # the scripted model calls no zoom tool
    assert any(c["route"].startswith("digitize:readout") for c in calls)


def test_late_window_provenance_records_the_rule_applied_and_the_x_read(bar_figure, tmp_path):
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    out = digitize(_client(_scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                                     _coord_payload(bar_figure, view.scale))),
                   paper, fig, TARGET, source=SOURCE, dataset=DATASET, out_dir=tmp_path,
                   result=True)
    prov = out.provenance
    assert prov["late_window_rule"] == "block_closest_to_end"          # x_hint set -> applied
    assert prov["late_window_rule_configured"] == "block_closest_to_end"
    assert prov["late_window_x_read"] == ["old", "young"]              # what the model says it read
    assert prov["late_window_x_agrees"] is False                       # two groups, two x labels

    still = replace(TARGET, x_hint="")                                 # not a time series
    out2 = digitize(_client(_scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                                      _coord_payload(bar_figure, view.scale))),
                    paper, fig, still, source=SOURCE, dataset=DATASET, out_dir=tmp_path,
                    result=True)
    assert out2.provenance["late_window_rule"] == "not_a_time_series"
    assert out2.provenance["late_window_rule_configured"] == "block_closest_to_end"


def test_axis_range_uses_the_plotted_span_and_says_so(bar_figure, tmp_path):
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    out = digitize(_client(_scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                                     _coord_payload(bar_figure, view.scale))),
                   paper, fig, TARGET, source=SOURCE, dataset=DATASET, out_dir=tmp_path,
                   result=True)
    assert out.provenance["axis_range_source"] == "plot_bbox"
    # the ticks span 0..60, and the drawn plot box is a little taller than the outermost ticks
    assert out.provenance["axis_range"] >= 60.0
    assert out.provenance["axis_range"] < 75.0


def test_a_kept_zero_confidence_snap_makes_the_ensemble_ambiguous():
    from canopy.digitize.digitizer import _ensemble_status
    assert _ensemble_status(True, False, []) == "found"
    assert _ensemble_status(True, False, ["digitize:raster_cv excluded: snap found no ink (confidence 0)"]) == "found"
    assert _ensemble_status(True, False, ["digitize:vlm_coords snapped with zero confidence (kept: too few routes left)"]) == "ambiguous"
    assert _ensemble_status(False, False, []) == "ambiguous"
    assert _ensemble_status(True, True, []) == "ambiguous"


def test_zero_confidence_snaps_are_excluded_unless_they_are_all_we_have():
    good = [RouteSample(route="D", group="A", mean=30.0, snap_conf=0.9),
            RouteSample(route="C", group="A", mean=30.1, snap_conf=0.8),
            RouteSample(route="B", group="A", mean=99.0, snap_conf=0.0)]
    kept, notes = _drop_zero_confidence(good)
    assert [s.route for s in kept] == ["D", "C"]
    assert notes and "no ink" in notes[0]

    thin = [RouteSample(route="C", group="A", mean=30.0, snap_conf=0.0),
            RouteSample(route="B", group="A", mean=30.2, snap_conf=0.0)]
    kept, notes = _drop_zero_confidence(thin)
    assert len(kept) == 2, "never drop everything — flag instead"
    assert all("too few routes left" in n for n in notes)

    # a read-out's confidence is the model's own, not a snap: never used to exclude
    only_d = [RouteSample(route="D", group="A", mean=30.0, snap_conf=0.0),
              RouteSample(route="D", group="A", mean=30.1, snap_conf=0.0),
              RouteSample(route="D", group="A", mean=30.2, snap_conf=0.0)]
    assert _drop_zero_confidence(only_d) == (only_d, [])


def test_conftest_keeps_the_recording_env_only_for_live_and_replay(monkeypatch):
    """Item 11's mechanism: without this, a recording run silently records nothing."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "canopy_test_conftest", Path(__file__).resolve().parent / "conftest.py")
    conftest = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conftest)
    fixture_fn = conftest.offline_by_default.__wrapped__

    class _Request:
        def __init__(self, marker: str | None):
            self.marker = marker
            self.node = self

        def get_closest_marker(self, name: str):
            return object() if name == self.marker else None

    for marker in ("live", "replay"):
        monkeypatch.setenv("CANOPY_LIVE", "1")
        fixture_fn(_Request(marker), monkeypatch)
        assert live_enabled(), f"{marker}-marked tests must still see CANOPY_LIVE"

    monkeypatch.setenv("CANOPY_LIVE", "1")
    fixture_fn(_Request(None), monkeypatch)
    assert not live_enabled(), "an unmarked test must be forced offline"
