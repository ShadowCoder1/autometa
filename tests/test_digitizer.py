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
import statistics
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
from canopy.models import (DatasetSpec, DigitizeSettings, DispersionType, GroupSpec, Source,
                          SourceKind)

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
    """The target travels in the USER turn (task 15 §A2a), so the system prompt is variant-free."""
    from canopy.digitize.vlm import load_prompt

    text = render_prompt("digitize_task", TARGET=TARGET.describe(), CAPTION="cap",
                         VARIANT="read the ticks first")
    assert "late_adaptation" in text and "block_closest_to_end" in text
    assert "cap" in text and "read the ticks first" in text
    with pytest.raises(KeyError):
        render_prompt("digitize_task", TARGET="t")
    # …and the two system prompts are the same bytes for every call on a figure
    for name in ("digitize_readout", "digitize_coords"):
        assert "{{" not in load_prompt(name), f"{name} still interpolates a per-call value"


def test_readout_plan_buys_a_second_model_family_before_a_second_prompt():
    """F2: two prompts of one model that agree are one voter agreeing with itself."""
    assert _readout_plan(("claude-opus-5",), 3) == [
        ReadoutSpec("claude-opus-5", "direct"), ReadoutSpec("claude-sonnet-5", "direct"),
        ReadoutSpec("claude-opus-5", "ticks_first")]
    # …and the first two, which is where the adaptive plan stops when they agree, are two families
    first_two = _readout_plan(("claude-opus-5",), 2)
    assert len({spec.model for spec in first_two}) == 2
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
                   out_dir=tmp_path, n_readouts=7, result=True,         # forces re-samples
                   settings=DigitizeSettings(readouts_min=7, readouts_max=7))
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
                   dataset=DATASET, out_dir=tmp_path, result=True,
                   settings=DigitizeSettings(overlay_verify="always"))
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


# ------------------------------------------------------------------ task 15: calls are bought
def _n_readouts(out) -> int:
    return len({(s.model, s.variant, s.sample) for s in out.samples if s.route == "D"})


def test_digitize_stops_at_two_read_outs_when_the_routes_agree(bar_figure, tmp_path):
    """The third vision pass is bought only when the first two disagree (task 15 §A3).

    A read-out is the most expensive call in the pipeline; a third one that confirms two agreeing
    reads changes neither the median nor the confidence gate.
    """
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                         _coord_payload(bar_figure, view.scale))
    out = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                   out_dir=tmp_path, result=True)
    assert _n_readouts(out) == 2
    plan = next(c for c in out.candidates
                if c.extractor_id == "digitize:ensemble").pixel_provenance["call_plan"]
    agreed = ("the routes agreed across two model families, so no further read-out was bought")
    assert plan == {"readouts_min": 2, "readouts_max": 3, "readouts_run": 2,
                    "extra_readouts_bought": 0,
                    "extra_readout_reason": agreed, "readout_stop_reason": agreed,
                    "overlay_verify": False,
                    "overlay_verify_reason": "every route agreed and none was dropped",
                    "list_regions_offered": plan["list_regions_offered"]}


def test_digitize_buys_a_third_read_out_when_the_first_two_disagree(bar_figure, tmp_path):
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    good = _readout_payload(31.5, 11.0, 12.25, 11.75)
    provider = _scripted(good, _coord_payload(bar_figure, view.scale),
                         readout_by_call=[_readout_payload(45.0, 11.0, 12.25, 11.75), good, good])
    out = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                   out_dir=tmp_path, result=True)
    assert _n_readouts(out) == 3
    plan = next(c for c in out.candidates
                if c.extractor_id == "digitize:ensemble").pixel_provenance["call_plan"]
    assert plan["extra_readouts_bought"] == 1 and "means span" in plan["extra_readout_reason"]
    # a disagreement is also what buys the overlay-verification call
    assert plan["overlay_verify"] is True


def test_digitize_still_draws_the_overlay_it_did_not_pay_to_verify(bar_figure, tmp_path):
    """Drawing the marks is free and is what a reviewer opens; only the CALL is conditional."""
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    out = digitize(_client(_scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                                     _coord_payload(bar_figure, view.scale))),
                   paper, fig, TARGET, source=SOURCE, dataset=DATASET, out_dir=tmp_path,
                   result=True)
    assert out.overlay_path and Path(out.overlay_path).exists()
    assert not next(c for c in out.candidates
                    if c.extractor_id == "digitize:ensemble").pixel_provenance["overlay_iterations"]


def test_readout_tool_list_drops_list_regions_when_the_cv_pass_found_nothing(tmp_path):
    """The tool list is part of the cached prefix, so it is decided once per figure."""
    from PIL import Image

    blank = tmp_path / "blank.png"
    Image.new("RGB", (300, 200), "white").save(blank)
    view = FigureView(blank, work_dir=tmp_path)
    assert view.has_regions is False
    assert "list_regions" not in {t["name"] for t in view.tools()}
    assert "list_regions" not in view.handlers()


def test_figure_view_sends_the_image_first_and_marks_it_cacheable(bar_figure, tmp_path):
    """Block order IS the cache design: header, image (marked), then everything that varies."""
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"], work_dir=tmp_path)
    provider = _scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                         _coord_payload(bar_figure, view.scale))
    client = _client(provider)
    systems = []
    for variant in ("direct", "ticks_first"):
        read_out(client, bar_figure["path"], "cap", TARGET, "claude-opus-5", variant, view=view)
        request = provider.requests[-1]
        systems.append(request.system)
        content = request.messages[0]["content"]
        assert [b["type"] for b in content] == ["text", "image", "text", "text"]
        assert "cache_control" in content[1] and content[1]["cache_control"]["type"] == "ephemeral"
        assert "cache_control" not in content[2] and "cache_control" not in content[3]
        assert TARGET.outcome_key in content[2]["text"], "the target belongs after the image"
    assert systems[0] == systems[1], "two variants must share one system prompt, or nothing caches"


# ------------------------------------------------------------------ task 15 §B: one-armed bars
def test_resolve_arms_never_halves_a_one_armed_whisker():
    from canopy.digitize.digitizer import resolve_arms

    assert resolve_arms(11.0, 11.2) == (pytest.approx(11.1), None)     # symmetric: the average
    assert resolve_arms(11.0, None) == (11.0, "up")                    # only an upper arm
    assert resolve_arms(None, 9.5) == (9.5, "down")
    # the killer case: the cap walk stopped on the marker's own edge a pixel below the datum
    assert resolve_arms(11.0, 0.99) == (11.0, "up")
    assert resolve_arms(0.4, 12.0) == (12.0, "down")
    # an arm inside the floor (two pixels' worth) is not an arm at all
    assert resolve_arms(0.05, None, floor=0.2) == (None, None)
    assert resolve_arms(None, None) == (None, None)


@pytest.fixture(scope="module")
def one_armed_figure(tmp_path_factory) -> dict:
    """Bock's figure form: the old series' whisker points UP, the young series' points DOWN."""
    plt = _mpl()
    dpi = 150
    values = {"A": 31.5, "B": 12.25}
    errors = {"A": 11.0, "B": 11.75}
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=dpi)
    ax.errorbar([0], [values["A"]], yerr=[[0.0], [errors["A"]]], fmt="o", color="#333333",
                capsize=8, markersize=9)
    ax.errorbar([1], [values["B"]], yerr=[[errors["B"]], [0.0]], fmt="s", color="#999999",
                capsize=8, markersize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_ylabel("error (deg)")
    fig.canvas.draw()
    out = tmp_path_factory.mktemp("one_armed")
    path = out / "figures" / "fig01.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    h_px = int(round(fig.get_size_inches()[1] * dpi))

    def row(value: float) -> float:
        return h_px - ax.transData.transform((0, value))[1]

    truth = {"path": path, "out_dir": out, "values": values, "errors": errors,
             "x": {"A": ax.transData.transform((0, 0))[0],
                   "B": ax.transData.transform((1, 0))[0]},
             "y": {k: row(v) for k, v in values.items()},
             "cap_down": {k: row(values[k] - errors[k]) for k in values}}
    plt.close(fig)
    return truth


def test_a_one_armed_whisker_reads_its_true_length_not_half_of_it(one_armed_figure):
    """End to end on pixels: the arm that exists IS the half-length (task 15 §B5).

    Averaging the arms read 6.0 where the figure showed 11.0 on the first live run, which is what
    made four agreeing routes look like a disagreement.
    """
    from canopy.digitize.calibrate import fit_axis, pair_ticks
    from canopy.digitize.cv import (find_axes, find_cap_ends, find_tick_marks, load_gray,
                                    ocr_tick_labels)
    from canopy.digitize.digitizer import _values_from_pixels

    gray = load_gray(one_armed_figure["path"])
    axes = find_axes(gray)
    rows = find_tick_marks(gray, axes).get("left", [])
    cal = fit_axis(pair_ticks(list(ocr_tick_labels(gray, axes, side="left", ticks=rows or None)),
                              rows, axis="y"), axis="y")
    span = abs(axes.plot_bbox[3] - axes.plot_bbox[1])
    # group A: the whole raster path, cap search included
    x, y = one_armed_figure["x"]["A"], one_armed_figure["y"]["A"]
    top, bottom = find_cap_ends(gray, x, y, max_len_px=span)
    assert bottom is not None, "the cap walk stops on the marker's own lower edge"
    mean, error, one_sided = _values_from_pixels(cal, y, top, bottom)
    assert mean == pytest.approx(one_armed_figure["values"]["A"], abs=0.5)
    assert error == pytest.approx(one_armed_figure["errors"]["A"], abs=1.0), (
        f"{error} — a one-armed whisker must not be halved")
    assert one_sided == "up"

    # group B (the arm points DOWN): `find_cap_ends` does not find this one at all on a square
    # marker, so the caps come from the model's own coordinates, as path C supplies them.
    y_b = one_armed_figure["y"]["B"]
    down_cap = one_armed_figure["cap_down"]["B"]
    mean, error, one_sided = _values_from_pixels(cal, y_b, None, down_cap)
    assert mean == pytest.approx(one_armed_figure["values"]["B"], abs=0.5)
    assert error == pytest.approx(one_armed_figure["errors"]["B"], abs=1.0)
    assert one_sided == "down"


def test_agreed_means_survive_a_dispersion_disagreement(bar_figure, tmp_path):
    """Amendment F per QUANTITY: the cell is `found`, the dispersion carries the doubt (§B4)."""
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    # same means, wildly different half-lengths — Bock's shape
    first = _readout_payload(31.5, 11.0, 12.25, 11.75)
    second = _readout_payload(31.6, 16.0, 12.30, 13.00)
    provider = _scripted(first, _coord_payload(bar_figure, view.scale),
                         readout_by_call=[first, second, second])
    out = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                   out_dir=tmp_path, result=True)
    ensemble = {c.group: c for c in out.candidates if c.extractor_id == "digitize:ensemble"}
    got = ensemble["A"]
    prov = got.pixel_provenance
    assert got.status == "found", got.notes
    assert got.mean == pytest.approx(31.5, abs=0.5)
    assert prov["mean_agreement"] is True and prov["error_agreement"] is False
    assert prov["needs_review"] is True and prov["needs_review_kind"] == "dispersion"
    assert "agree about the mean and disagree about the error half-length" in got.notes
    # the median of the routes that FOUND a whisker, with its own uncertainty widened
    assert got.dispersion_value is not None
    assert got.dispersion_sigma is not None and got.dispersion_sigma > (got.sigma or 0)
    assert prov["n_routes_with_error"] >= 2


# ------------------------------------------------------------------ task 16 (a): calibration truth
def test_fit_best_scale_infers_a_log_axis_from_the_ticks_alone():
    """Nothing in the raster path used to ask whether an axis is logarithmic (critique miss 2)."""
    from canopy.digitize.digitizer import fit_best_scale

    # a log ladder: 1, 3, 10, 30, 100 drawn at even pixel spacing
    log_pairs = [(400.0, 1.0), (300.0, 3.16227766), (200.0, 10.0), (100.0, 31.6227766),
                 (0.0, 100.0)]
    from canopy.digitize.calibrate import px_to_value

    cal, note = fit_best_scale(log_pairs)
    assert note == "log" and cal.scale == "log"
    assert px_to_value(cal, 200.0) == pytest.approx(10.0, rel=1e-6)

    linear_pairs = [(400.0, 0.0), (300.0, 10.0), (200.0, 20.0), (100.0, 30.0), (0.0, 40.0)]
    cal, note = fit_best_scale(linear_pairs)
    assert note == "linear" and cal.scale == "linear"
    assert px_to_value(cal, 200.0) == pytest.approx(20.0, abs=1e-9)


def test_fit_best_scale_refuses_to_guess_when_two_ticks_fit_both_scales():
    from canopy.digitize.digitizer import fit_best_scale

    cal, note = fit_best_scale([(100.0, 10.0), (0.0, 100.0)])
    # two ticks fit a line AND an exponential exactly; the honest prior is linear, said out loud
    assert note == "linear" and cal.scale == "linear"


def test_fit_best_scale_says_scale_ambiguous_when_neither_fit_wins():
    from canopy.digitize.digitizer import fit_best_scale

    # a ladder neither scale reproduces: 1, 2, 3.5, 5 is not a line and not a decade series
    _, note = fit_best_scale([(300.0, 1.0), (200.0, 2.0), (100.0, 3.5), (0.0, 5.0)])
    assert note == "scale_ambiguous"
    # a clean doubling IS log, and is named as such
    _, note = fit_best_scale([(300.0, 1.0), (200.0, 2.0), (100.0, 4.0), (0.0, 8.0)])
    assert note == "log"


def test_readout_tick_values_pair_with_the_detected_tick_rows():
    """The read-outs already report the ladder; pairing it with the CV rows costs nothing."""
    from canopy.digitize.digitizer import _ladder_from_values

    rows = [268.4, 340.5, 412.6, 485.4, 557.5, 629.6, 702.4, 774.5, 846.6, 918.5, 991.5]
    exact = _ladder_from_values([45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5], rows)
    assert exact is not None and exact[0] == (268.4, 45.0) and exact[-1] == (991.5, -5.0)

    strided = _ladder_from_values([-5, 5, 15, 25, 35, 45], rows)      # every second tick labelled
    assert strided is not None and len(strided) == 6
    assert strided[0] == (268.4, 45.0) and strided[-1] == (991.5, -5.0)

    assert _ladder_from_values([1, 2, 3, 4], rows) is None            # 4 values, 11 rows: refuse
    assert _ladder_from_values([45.0], rows) is None
    assert _ladder_from_values([45, 35], []) is None


def _cal(pairs, scale="linear"):
    from canopy.digitize.calibrate import fit_axis
    return fit_axis(list(pairs), scale=scale, axis="y")


def _core_with(cal, rows=(), scale_note="linear"):
    from canopy.digitize.cv import Axes
    from canopy.digitize.digitizer import _Core

    axes = Axes(y_axis_x=10.0, x_axis_y=None, y_axis_span=None, x_axis_span=None,
                y_axis_width=1.0, x_axis_width=0.0, plot_bbox=(10.0, 0.0, 400.0, 300.0),
                confidence=0.6)
    return _Core(gray=None, colour=None, axes=axes, tick_rows=list(rows), labels=[], cal=cal,
                 ocr_status="ok", bars=[], markers=[], scale_note=scale_note)


def _readout(means, ticks, model="claude-opus-5", variant="direct"):
    from canopy.digitize.vlm import GroupReadOut, ReadOut

    return ReadOut(groups=[GroupReadOut(group=g, mean=m) for g, m in means.items()],
                   tick_labels=list(ticks), model=model, variant=variant)


def test_choose_calibration_confirms_a_mapping_two_witnesses_agree_on():
    from canopy.digitize.digitizer import _choose_calibration

    ladder = [(100.0, 40.0), (200.0, 30.0), (300.0, 20.0), (400.0, 10.0)]
    core = _core_with(_cal(ladder), rows=[100.0, 200.0, 300.0, 400.0])
    choice = _choose_calibration(core, None, None,
                                 [_readout({"A": 25.0}, [40, 30, 20, 10])])
    assert choice.status == "confirmed"
    assert choice.source == "cv_ocr"
    assert set(choice.agreeing) == {"cv_ocr", "readout_ticks"}


def test_choose_calibration_is_a_single_witness_when_nothing_corroborates_it():
    from canopy.digitize.digitizer import _choose_calibration

    ladder = [(100.0, 40.0), (200.0, 30.0), (300.0, 20.0), (400.0, 10.0)]
    core = _core_with(_cal(ladder), rows=[100.0, 200.0, 300.0, 400.0])
    choice = _choose_calibration(core, None, None, [])
    assert choice.status == "single_witness" and choice.source == "cv_ocr"
    assert "only the cv_ocr ladder" in choice.why


def test_an_ambiguous_scale_is_never_a_confirmed_calibration():
    from canopy.digitize.digitizer import _choose_calibration

    ladder = [(100.0, 40.0), (200.0, 30.0), (300.0, 20.0), (400.0, 10.0)]
    core = _core_with(_cal(ladder), rows=[100.0, 200.0, 300.0, 400.0],
                      scale_note="scale_ambiguous")
    choice = _choose_calibration(core, None, None,
                                 [_readout({"A": 25.0}, [40, 30, 20, 10])])
    assert choice.status == "single_witness"
    assert "linear or logarithmic" in choice.why


def test_the_magnitude_rule_refutes_a_ladder_that_cannot_draw_the_read_values():
    """Cressman 2010 Fig. 3a: OCR read 45/35/25/15 as 4/3/2/1 and fitted it to 0.2 px."""
    from canopy.digitize.digitizer import _choose_calibration

    stale = _cal([(268.0, 4.0), (412.0, 3.0), (557.0, 2.0), (702.0, 1.0)])
    core = _core_with(stale, rows=[268.0, 412.0, 557.0, 702.0])
    choice = _choose_calibration(core, None, None, [
        _readout({"A": 31.3, "B": 33.3}, [], variant="direct"),
        _readout({"A": 31.3, "B": 33.3}, [], variant="ticks_first")])
    assert choice.status == "cal_refuted"
    assert choice.usable_for_pixels is None
    assert choice.refutation["tick_max"] == 4.0
    assert choice.refutation["readout_max_abs_mean"] == 33.3
    assert "33.3" in choice.why and "4" in choice.why


def test_the_magnitude_rule_needs_two_agreeing_read_outs():
    from canopy.digitize.digitizer import _choose_calibration

    stale = _cal([(268.0, 4.0), (412.0, 3.0), (557.0, 2.0), (702.0, 1.0)])
    core = _core_with(stale, rows=[268.0, 412.0, 557.0, 702.0])
    # one read-out cannot refute anything: it may itself have read the wrong panel
    lonely = _choose_calibration(core, None, None, [_readout({"A": 31.3}, [])])
    assert lonely.status == "single_witness"
    # two read-outs that disagree with EACH OTHER cannot either
    noisy = _choose_calibration(core, None, None, [
        _readout({"A": 31.3}, [], variant="direct"),
        _readout({"A": 3.1}, [], variant="ticks_first")])
    assert noisy.status == "single_witness"


def test_a_correct_ladder_is_not_refuted_by_values_inside_it():
    from canopy.digitize.digitizer import _choose_calibration

    good = _cal([(54.8, 60.0), (209.8, 40.0), (368.2, 20.0), (524.8, 0.0), (684.0, -20.0)])
    core = _core_with(good, rows=[54.8, 209.8, 368.2, 524.8, 684.0])
    choice = _choose_calibration(core, None, None, [
        _readout({"A": 32.2, "B": 12.4}, [], variant="direct"),
        _readout({"A": 31.8, "B": 11.8}, [], variant="ticks_first")])
    assert choice.status != "cal_refuted"


def test_the_printed_tick_ladder_outranks_an_ocr_read_of_the_same_axis():
    """F3: the PDF text layer is the number the publisher typeset; OCR is a guess about it.

    Both ladders here describe the SAME mapping, so the vote confirms either one; what is under
    test is which witness's numbers are then used. Measured on the audited corpus the vector fit
    residual is 0.003 px against tesseract's 0.20-0.92 px, and tesseract produced two
    order-of-magnitude misreads on three figures. Ranking a read of a rendered glyph above the
    glyph's own source is backwards for any figure whose PDF still carries its text.
    """
    from canopy.digitize.digitizer import CAL_PREFERENCE, _choose_calibration

    assert CAL_PREFERENCE.index("vector") < CAL_PREFERENCE.index("cv_ocr")
    ladder = [(100.0, 40.0), (200.0, 30.0), (300.0, 20.0), (400.0, 10.0)]
    core = _core_with(_cal(ladder), rows=[100.0, 200.0, 300.0, 400.0])
    choice = _choose_calibration(core, None, _cal(ladder), [])
    assert choice.status == "confirmed"
    assert choice.source == "vector"
    assert set(choice.agreeing) == {"vector", "cv_ocr"}


def test_a_two_tick_ladder_never_supplies_the_numbers_when_a_checkable_one_agrees():
    """A ladder fitted through two ticks reproduces a linear AND a log axis exactly and equally,
    so nothing about it can be checked against itself. It is a witness, not the source of record —
    whatever route built it. (Same threshold as `_MIN_TICKS_FOR_SCALE`.)"""
    from canopy.digitize.digitizer import _choose_calibration

    ladder = [(100.0, 40.0), (200.0, 30.0), (300.0, 20.0), (400.0, 10.0)]
    core = _core_with(_cal(ladder), rows=[100.0, 200.0, 300.0, 400.0])
    thin_vector = _cal([(100.0, 40.0), (400.0, 10.0)])
    choice = _choose_calibration(core, None, thin_vector, [])
    assert choice.status == "confirmed"
    assert set(choice.agreeing) == {"vector", "cv_ocr"}
    assert choice.source == "cv_ocr", "a 2-tick ladder outranked a 4-tick one"


def _ensemble_of(bar_figure, samples, base, **kw):
    """`_build_candidates` on a hand-made sample list — the ensemble candidate for group A."""
    from canopy.digitize.digitizer import _build_candidates

    paper, fig = _paper_for(bar_figure)
    out = _build_candidates(
        list(samples), target=kw.get("target", TARGET), fig=fig, paper=paper,
        source=kw.get("source", SOURCE), dataset=DATASET, core=_core_with(None), cal=None,
        crop=bar_figure["path"], overlay_path="", base=base, axis_range=60.0, tick_spacing=10.0,
        px_units=0.1, readouts=kw.get("readouts", []), want_uncertainty=True,
        labels={"A": "old", "B": "young"})
    return next(c for c in out
                if c.extractor_id == "digitize:ensemble" and c.group == "A")


def _d_sample(mean, error, **kw):
    """One route-D sample for group A (the ensemble arithmetic under test needs nothing else)."""
    kw.setdefault("model", "claude-opus-5")
    return RouteSample(route="D", group="A", variant="direct", mean=mean, error=error, **kw)


def test_a_figure_with_no_calibration_at_all_says_so_on_the_row(bar_figure):
    """F3: `cal_status="none"` used to be the quietest state in the system.

    The read-out routes need no ladder to produce a number, so a cell whose axis was never
    calibrated was released with an empty reason — while a cell with ONE witness was capped and
    flagged. Zero witnesses cannot be less suspicious than one.
    """
    pair = [_d_sample(31.5, 11.0), _d_sample(31.4, 11.1, model="claude-sonnet-5")]
    blind = _ensemble_of(bar_figure, pair, {"cal_status": "none", "figure_id": "fig01"})
    assert blind.pixel_provenance["cal_missing"] is True
    assert blind.pixel_provenance["needs_review"] is True
    assert blind.pixel_provenance["needs_review_kind"] == "calibration"
    assert "no y calibration could be built" in blind.notes

    seeing = _ensemble_of(bar_figure, pair, {"cal_status": "confirmed", "figure_id": "fig01"})
    assert seeing.pixel_provenance["cal_missing"] is False
    assert seeing.pixel_provenance["needs_review"] is False
    assert "no y calibration" not in seeing.notes


# ------------------------------------------------------------------ F2: dispersion arms
def test_two_arms_that_do_not_agree_are_not_a_symmetric_bar():
    """F2, with the numbers from `runs/proof` Bock 2005 Fig. 1, group A (aftereffect).

    Route C reported `y=692.137, cap_top=631.0, cap_bottom=730.0` on a ladder of 0.1272 deg/px:
    a 7.78-deg up arm and a 4.82-deg "down arm" that is really the young group's triangle. At a
    ratio of 0.62 the old rule kept both and returned their mean, 6.30 — a bar that corresponds
    to no ink in the panel. An error bar is `mean ± half-length`, so its arms are equal by
    construction; a 61% disparity is not measurement noise.
    """
    from canopy.digitize.digitizer import resolve_arms

    error, side = resolve_arms(7.783, 4.821, floor=0.254)
    assert error == pytest.approx(7.783) and side == "up"


def test_two_arms_that_differ_only_by_pixel_noise_are_still_averaged():
    """The other half of the same rule: a genuinely symmetric bar must not become one-armed."""
    from canopy.digitize.digitizer import resolve_arms

    error, side = resolve_arms(11.0, 10.6, floor=0.254)
    assert error == pytest.approx(10.8) and side is None
    # a difference under the caller's floor is noise however small the arms are
    error, side = resolve_arms(0.9, 0.7, floor=0.4)
    assert error == pytest.approx(0.8) and side is None


def _marker(x, y, size=15.0):
    from canopy.digitize.cv import Marker
    return Marker(x=x, y=y, colour="#000000", kind="square", size=size)


def test_a_cap_that_landed_on_the_other_series_mark_is_not_this_series_whisker():
    """F2: two series stacked in one column, and the cap walk stops on the wrong one.

    Real geometry from `runs/proof` Bock 2005 Fig. 1, episode 21: the square sits at y 692 and the
    triangle at y 723.6, 6.4 px from the "lower cap" route C reported for the square. The panel's
    only real caps are at y 630 (above the square) and y 769 (below the triangle) — I scanned the
    column. A cap inside another series' glyph is that series' ink.
    """
    from canopy.digitize.digitizer import _drop_caps_on_other_series

    core = _core_with(None)
    core.markers = [_marker(1336.0, 692.1), _marker(1336.0, 723.6)]
    a = RouteSample(route="C", group="A", x_px=1336.0, y_px=692.137,
                    cap_top_px=631.0, cap_bottom_px=730.0)
    b = RouteSample(route="C", group="B", x_px=1336.0, y_px=723.616,
                    cap_top_px=730.0, cap_bottom_px=769.5)
    _drop_caps_on_other_series([a, b], core)
    assert a.cap_top_px == 631.0 and a.cap_bottom_px is None
    assert "group B" in a.notes and "whisker" in a.notes
    # …and group B loses only the cap that sits on group A's square, keeping its real lower one
    assert b.cap_bottom_px == 769.5


def test_a_cap_found_past_the_other_series_is_that_series_whisker():
    """The same guard, the other failure direction: the walk did not stop on the neighbour's mark,
    it ran through it and stopped on the neighbour's own CAP. Found by building a stacked-series
    figure with known truth — the short-arm case alone left this one reading 12.05 for a bar of
    8.0."""
    from canopy.digitize.digitizer import _drop_caps_on_other_series

    core = _core_with(None)
    core.markers = [_marker(384.0, 226.0), _marker(384.0, 257.0)]
    a = RouteSample(route="C", group="A", x_px=384.0, y_px=226.0,
                    cap_top_px=164.0, cap_bottom_px=318.0)
    b = RouteSample(route="C", group="B", x_px=384.0, y_px=257.0,
                    cap_top_px=164.0, cap_bottom_px=318.0)
    _drop_caps_on_other_series([a, b], core)
    assert a.cap_top_px == 164.0 and a.cap_bottom_px is None      # 318 is past group B
    assert b.cap_bottom_px == 318.0 and b.cap_top_px is None      # 164 is past group A


def test_a_cap_a_column_away_is_left_alone():
    """The guard is about a vertical walk running into a mark, so it needs the same column."""
    from canopy.digitize.digitizer import _drop_caps_on_other_series

    core = _core_with(None)
    core.markers = [_marker(100.0, 200.0), _marker(400.0, 260.0)]
    a = RouteSample(route="C", group="A", x_px=100.0, y_px=200.0,
                    cap_top_px=160.0, cap_bottom_px=258.0)
    b = RouteSample(route="C", group="B", x_px=400.0, y_px=260.0,
                    cap_top_px=220.0, cap_bottom_px=300.0)
    _drop_caps_on_other_series([a, b], core)
    assert a.cap_bottom_px == 258.0 and b.cap_top_px == 220.0


@pytest.mark.parametrize("note, side", [
    ("Marker overlaps the filled young square; upper error cap hidden, half-length inferred "
     "from the visible lower arm.", "up"),
    ("lower cap inferred by symmetry", "down"),
    # a clause naming BOTH sides does not say which cap is missing, so the prose guard declines
    # to act on it. This real note (`runs/proof`, route C on Bock Fig. 1) is caught instead by
    # `_drop_caps_on_other_series`, which can see that the cap sits on the other group's mark.
    ("lower cap taken as the upper of the coincident cap pair near y=729", None),
    ("both caps clearly visible", None),
    ("", None),
    ("the lower panel is not visible in this crop", None),      # no cap named: not about a cap
    ("upper cap hidden and lower cap hidden", None),            # a bar with no arms is not usable
    ("error bars are standard deviations", None),
])
def test_a_cap_the_reader_says_it_did_not_see_is_read_out_of_its_prose(note, side):
    """F2/F5: the read-out prompt asks for an undrawn cap to be left null; readers fill it in
    anyway and then say so in `notes`. A cap the reader admits it invented is not evidence."""
    from canopy.digitize.digitizer import _unmeasured_cap_side

    assert _unmeasured_cap_side(note) == side


def test_a_fabricated_cap_does_not_make_a_one_armed_bar_look_symmetric():
    """`runs/proof` cache 7258771c…: `error_sides: "both"`, `error_lower: -29.0`, note "lower cap
    inferred by symmetry". The half-length it measured off the visible arm stands; the topology
    it asserted does not."""
    from canopy.digitize.digitizer import _samples_from_readout
    from canopy.digitize.vlm import GroupReadOut, ReadOut

    reading = ReadOut(groups=[GroupReadOut(
        group="A", mean=-21.2, error_half_length=7.8, error_upper=-13.4, error_lower=-29.0,
        error_sides="both", notes="lower cap inferred by symmetry")],
        model="claude-opus-5", variant="direct")
    sample = _samples_from_readout(reading)[0]
    assert sample.error == pytest.approx(7.8), "the measured arm is still the half-length"
    assert sample.one_sided == "up"
    assert sample.extra["unmeasured_cap"] == "down"


@pytest.fixture(scope="module")
def stacked_figure(tmp_path_factory) -> dict:
    """Two series in ONE column, each with a whisker drawn on one side only.

    This is the shape of Bock 2005 Fig. 1 (and of any figure that plots two overlapping groups at
    the same x): the upper series' whisker goes up, the lower series' goes down, and the space
    between the two marks contains nothing but the other group's ink. The synthetic corpus under
    `validation/synthetic/` has no case of this class — every one of its cases puts its series in
    separate columns — so the truth here is built the same way that corpus builds its own:
    matplotlib, with the plotted values as ground truth.
    """
    plt = _mpl()
    truth = {"A": (40.0, 8.0), "B": (36.0, 8.0)}       # value, one-armed half-length
    dpi, x, w = 150, 1.0, 0.12
    fig, ax = plt.subplots(figsize=(5.0, 4.0), dpi=dpi)
    for key, (value, err), sign, colour in (("A", truth["A"], 1, "#111111"),
                                            ("B", truth["B"], -1, "#666666")):
        cap = value + sign * err
        ax.plot([x, x], [value, cap], color="black", lw=1.2)
        ax.plot([x - w, x + w], [cap, cap], color="black", lw=1.6)
        ax.plot([x], [value], marker="s", color=colour, markersize=13, linestyle="none")
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 60)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xticks([1])
    ax.set_xticklabels(["ep 21"])
    ax.set_ylabel("error (deg)")
    fig.canvas.draw()
    out = tmp_path_factory.mktemp("stacked")
    path = out / "figures" / "fig01.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    h_px = int(round(fig.get_size_inches()[1] * dpi))
    rows = {k: h_px - ax.transData.transform((0, v))[1] for k, (v, _) in truth.items()}
    column = ax.transData.transform((x, 0))[0]
    plt.close(fig)
    return {"path": path, "out_dir": out, "truth": truth, "rows": rows, "column": column,
            "px_per_unit": (rows["B"] - rows["A"]) / (truth["A"][0] - truth["B"][0])}


def test_a_cap_on_the_neighbouring_series_no_longer_halves_a_one_armed_whisker(stacked_figure):
    """F2, end to end on a figure whose half-lengths are known by construction.

    The caps here are the ones route C really reported for Bock 2005 in `runs/proof`: group A's
    genuine upper cap, and a "lower cap" that has landed just past group B's mark. At a ratio of
    0.62 the old rule averaged the two and reported 6.37 for a bar that is 8.0 — a 20% error in
    the denominator of every effect size computed from it.
    """
    from canopy.digitize.digitizer import (_choose_calibration, _samples_from_coords)
    from canopy.digitize.vlm import CoordReadout, GroupCoords

    truth, rows, per_unit = (stacked_figure["truth"], stacked_figure["rows"],
                             stacked_figure["px_per_unit"])
    core = _cv_core_of(stacked_figure["path"])
    coord = CoordReadout(status="found", model="claude-opus-5", groups=[
        GroupCoords(group="A", x_px=stacked_figure["column"], y_px=rows["A"],
                    cap_top_px=rows["A"] - truth["A"][1] * per_unit,
                    cap_bottom_px=rows["B"] + 0.85 * per_unit),
        GroupCoords(group="B", x_px=stacked_figure["column"], y_px=rows["B"],
                    cap_bottom_px=rows["B"] + truth["B"][1] * per_unit)])
    choice = _choose_calibration(core, None, None, [])
    samples = {s.group: s for s in
               _samples_from_coords(coord, core, choice.usable_for_pixels, choice.source)}

    assert samples["A"].error == pytest.approx(truth["A"][1], abs=0.1)     # was 6.37
    assert samples["A"].one_sided == "up"
    assert samples["A"].cap_bottom_px is None
    assert "group B" in samples["A"].notes
    # the series that was never misread is untouched
    assert samples["B"].error == pytest.approx(truth["B"][1], abs=0.1)
    assert samples["B"].one_sided == "down"


def _cv_core_of(path):
    from canopy.digitize.digitizer import _cv_core
    return _cv_core(path, prefer_markers=True)


# ------------------------------------------------------------------ F4: the adaptive stop
#: `runs/proof/papers/5039533c85ef/extract.json`, `…:late_adaptation:{A,B}:digitize:ensemble`.
#: Two read-outs, agreeing on the means and 1.2 deg apart on group A's half-length, on an axis
#: whose ticks are 10 deg apart and whose fitted resolution is 0.0691 deg/px.
_CRESSMAN_AXIS = dict(axis_range=50.26, tick_spacing=10.0, px_units=0.0691)


def _proof_pair(group, opus, sonnet):
    return [RouteSample(route="D", group=group, model="claude-opus-5", variant="direct",
                        mean=opus[0], error=opus[1]),
            RouteSample(route="D", group=group, model="claude-sonnet-5", variant="direct",
                        mean=sonnet[0], error=sonnet[1])]


def test_the_stop_rule_buys_another_read_out_when_the_dispersions_disagree():
    """F4: `dual_tolerance` was handed an empty error list, so the plan stopped on the means.

    The record shows what that cost: `error_agrees: false, error_spread: 1.2` against a tolerance
    of 0.24, `extra_readouts_bought: 0`, a third read-out planned and inside `readouts_max`, and
    the two half-lengths implying d = -0.2548 or d = -0.1478 depending on which is believed. A
    spread the effect size divides by is as load-bearing as the mean it subtracts.
    """
    from canopy.digitize.digitizer import _needs_another_readout

    disputed = _proof_pair("A", (31.2, 1.8), (31.0, 3.0))
    needed, why = _needs_another_readout(disputed, **_CRESSMAN_AXIS)
    assert needed and "error half-lengths span" in why
    assert "means span" not in why, "the means agreed; it is the spread that is in dispute"

    # the same cell's other group agrees on BOTH quantities and still stops at two read-outs
    settled = _proof_pair("B", (33.3, 3.2), (32.5, 3.5))
    needed, why = _needs_another_readout(settled, **_CRESSMAN_AXIS)
    assert not needed and "two model families" in why


def test_a_disputed_dispersion_also_buys_the_overlay_check():
    """The overlay call exists to look at the picture when the routes disagree — and they do."""
    from canopy.digitize.digitizer import _overlay_wanted

    disputed = _proof_pair("A", (31.2, 1.8), (31.0, 3.0))
    wanted, why = _overlay_wanted(True, "on_disagreement", disputed, **_CRESSMAN_AXIS)
    assert wanted and "error half-lengths span" in why


# ------------------------------------------------------------------ F5: the dispersion combiner
def test_a_one_armed_read_outvotes_a_two_armed_one_on_the_half_length(bar_figure):
    """F5: the ensemble was measurably less accurate than its best member, and always on the
    spread. Routes that disagree about the whisker's TOPOLOGY are not measuring the same object.

    A one-armed read is the half-length whether the bar is one-armed (it measured the only arm)
    or two-armed (the arms of `mean ± half-length` are equal). A two-armed read is the half-length
    only in the second case: in the first, its "other arm" is whatever its cap walk stopped on.
    So when the two conflict the one-armed reads carry the evidence — and this is about the
    reading, not about which model or route produced it.
    """
    one_armed = _d_sample(31.2, 1.8, one_sided="down")
    two_armed = _d_sample(31.0, 3.0, model="claude-sonnet-5")
    base = {"cal_status": "confirmed", "figure_id": "fig03"}
    ens = _ensemble_of(bar_figure, [one_armed, two_armed], base)
    assert ens.dispersion_value == pytest.approx(1.8)
    assert ens.pixel_provenance["n_routes_with_error"] == 2
    assert ens.pixel_provenance["n_routes_voting_on_error"] == 1
    assert "one side only" in ens.pixel_provenance["dispersion_topology_note"]
    # the disagreement is settled, not erased: the record still shows both reads and the
    # dispersion keeps an uncertainty that spans them
    assert ens.pixel_provenance["error_agreement"] is False
    assert ens.dispersion_sigma >= 0.5 * (3.0 - 1.8)
    assert sorted(r["error"] for r in
                  ens.pixel_provenance["dispersion_route_errors"].values()) == [1.8, 3.0]


def test_routes_that_agree_about_the_topology_are_all_still_medianed(bar_figure):
    """The segregation only fires on a conflict; two one-armed reads are two votes as before."""
    both_one_armed = [_d_sample(31.2, 1.8, one_sided="down"),
                      _d_sample(31.0, 2.2, one_sided="down", model="claude-sonnet-5")]
    base = {"cal_status": "confirmed", "figure_id": "fig03"}
    ens = _ensemble_of(bar_figure, both_one_armed, base)
    assert ens.dispersion_value == pytest.approx(2.0)
    assert ens.pixel_provenance["n_routes_voting_on_error"] == 2
    assert ens.pixel_provenance["dispersion_topology_note"] == ""


def test_an_asymmetric_whisker_kind_is_left_alone(bar_figure):
    """An IQR box has genuinely unequal arms, so one arm is not a half-length and the argument
    for preferring a one-armed read does not hold. Nothing is segregated there."""
    from canopy.models import DispersionType

    mixed = [_d_sample(31.2, 1.8, one_sided="down"),
             _d_sample(31.0, 3.0, model="claude-sonnet-5")]
    iqr_source = SOURCE.model_copy(update={"error_bar_type": DispersionType.IQR})
    ens = _ensemble_of(bar_figure, mixed, {"cal_status": "confirmed", "figure_id": "fig03"},
                       source=iqr_source)
    assert ens.dispersion_value == pytest.approx(2.4)          # the plain median of both
    assert ens.pixel_provenance["dispersion_topology_note"] == ""


def test_the_marker_floor_rejects_a_cap_inside_the_marker(bar_figure):
    """Bock 2005 route C stopped on the square's own lower edge (acceptance item 9)."""
    from canopy.digitize.digitizer import _values_from_pixels

    cal = _cal([(54.77215189873418, 60.0), (209.75, 40.0), (368.2278481012658, 20.0),
                (524.780487804878, 0.0), (684.016393442623, -20.0), (839.4375, -40.0)])
    mean, error, side = _values_from_pixels(cal, 276.5, 190.0, 284.2957446575165, floor_px=7.5)
    assert mean == pytest.approx(31.6677, abs=1e-3)
    assert error == pytest.approx(11.00, abs=0.02)          # not 5.998
    assert side == "up"


def test_the_family_rule_buys_a_second_family_and_records_which_ones_answered(bar_figure,
                                                                              tmp_path):
    """F2, end to end: `readouts_min=1` leaves one family, so the adaptive step buys another."""
    from canopy.digitize.digitizer import model_families

    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                         _coord_payload(bar_figure, view.scale))
    out = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                   out_dir=tmp_path, result=True,
                   settings=DigitizeSettings(readouts_min=1, readouts_max=3))
    plan = out.provenance["call_plan"]
    assert plan["readouts_run"] == 2 and plan["extra_readouts_bought"] == 1
    assert "one model family" in plan["extra_readout_reason"]
    assert out.provenance["model_families"] == ["claude-opus", "claude-sonnet"]
    assert plan["readouts_run"] <= 3, "the cost guard: never more than three read-outs"
    # …and the ensemble does not pretend to be one of them
    ensemble = next(c for c in out.candidates if c.extractor_id == "digitize:ensemble")
    assert ensemble.model == ""
    assert ensemble.pixel_provenance["model_families"] == ["claude-opus", "claude-sonnet"]
    assert model_families([s for s in out.samples if s.group == "A"]) == ["claude-opus",
                                                                         "claude-sonnet"]


def test_a_reader_that_produced_no_value_is_not_counted_as_an_agreeing_family(bar_figure,
                                                                                tmp_path):
    """`runs/proof`: Bock group A was credited with a sonnet reader whose `mean` was null.

    The confidence score pays +0.25 for "two independent model families read it and agreed", and
    that bonus is the only term that lifts a figure cell over the acceptance line. Counting a
    reader that abstained makes the deciding bit a lie — group A was released and group B, on the
    same figure of the same paper, was withheld, purely on that phantom vote. A vote may only
    credit readers that cast a ballot.
    """
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                         _coord_payload(bar_figure, view.scale),
                         readout_by_call=[_readout_payload(31.5, 11.0, 12.25, 11.75),
                                          _readout_payload(None, None, None, None,
                                                           status="ambiguous")])
    out = digitize(_client(provider), paper, fig, TARGET, source=SOURCE, dataset=DATASET,
                   out_dir=tmp_path, result=True,
                   settings=DigitizeSettings(readouts_min=1, readouts_max=2))
    assert out.provenance["call_plan"]["readouts_run"] == 2      # the second family WAS bought…
    assert out.provenance["model_families"] == ["claude-opus"]   # …and it answered with nothing
    ensemble = next(c for c in out.candidates if c.extractor_id == "digitize:ensemble")
    assert ensemble.pixel_provenance["model_families"] == ["claude-opus"]


# ------------------------------------------------------------------ task 16 (e): the misses
def _readout_sample(group, model, axis_read="", label_read="", mean=10.0, x_read=""):
    return RouteSample(route="D", group=group, model=model, variant="direct", mean=mean,
                       label_read=label_read,
                       extra={"axis_read": axis_read, "x_read": x_read,
                              "axis_direction_note": ""})


def test_two_readers_on_two_different_value_axes_are_not_pooled():
    """Miss 1: Cressman Fig. 3b has a left axis in degrees and a right one in per cent."""
    from canopy.digitize.digitizer import _reconcile_axes

    left = "left y-axis 'Aftereffects at Peak Velocity (deg)'"
    right = "right y-axis 'Aftereffects at Peak Velocity (%)'"
    samples = [_readout_sample("A", "claude-opus-5", left, mean=17.5),
               _readout_sample("A", "claude-sonnet-5", left, mean=17.4),
               _readout_sample("A", "claude-haiku-4-5", right, mean=58.0)]
    info = _reconcile_axes(samples, replace(TARGET, unit_hint="deg"))
    assert info["axis_agreement"] == "conflict"
    assert "deg" in info["axis_kept"]
    assert len(info["axis_dropped_samples"]) == 1
    assert [s.dropped for s in samples] == [False, False, True]
    assert "two value axes are not one number" in samples[2].drop_reason


def test_readers_who_name_the_same_axis_differently_are_still_pooled():
    from canopy.digitize.digitizer import _reconcile_axes

    samples = [_readout_sample("A", "claude-opus-5", "y-axis: angular error (deg)"),
               _readout_sample("A", "claude-sonnet-5", "the y axis, angular error in deg")]
    info = _reconcile_axes(samples, TARGET)
    assert info["axis_agreement"] == "agreed"
    assert not any(s.dropped for s in samples)
    # and a reader that did not say cannot be said to disagree
    quiet = [_readout_sample("A", "claude-opus-5", "y-axis (deg)"),
             _readout_sample("A", "claude-sonnet-5", "")]
    assert _reconcile_axes(quiet, TARGET)["axis_agreement"] == "not_enough_readers"


def test_both_groups_described_as_the_same_marker_is_a_conflict():
    """Misses 4 and 5: group assignment rests on one free-text legend read."""
    from canopy.digitize.cv import Marker
    from canopy.digitize.digitizer import _series_identity

    core = _core_with(None)
    core.markers = [Marker(x=10.0, y=20.0, colour="#000000", kind="square", size=6.0),
                    Marker(x=40.0, y=25.0, colour="#ffffff", kind="open", size=6.0)]
    same = [_readout_sample("A", "m", label_read="Young: open white squares"),
            _readout_sample("B", "m", label_read="Elderly: open white squares")]
    info = _series_identity(same, core)
    assert info["conflict"] is True
    assert "same marker" in " ".join(info["notes"])

    distinct = [_readout_sample("A", "m", label_read="Young: filled black squares"),
                _readout_sample("B", "m", label_read="Elderly: open white squares")]
    assert _series_identity(distinct, core)["conflict"] is False


def test_a_marker_the_pixel_pass_never_found_is_reported():
    from canopy.digitize.cv import Marker
    from canopy.digitize.digitizer import _series_identity

    core = _core_with(None)
    core.markers = [Marker(x=10.0, y=20.0, colour="#000000", kind="square", size=6.0)]
    described = [_readout_sample("A", "m", label_read="Young: open white circles"),
                 _readout_sample("B", "m", label_read="Elderly: filled black squares")]
    info = _series_identity(described, core)
    assert info["conflict"] is False
    assert any("OPEN marker" in note for note in info["notes"])
    assert info["detected_marker_kinds"] == ["square"]


def test_the_late_window_compares_pixels_not_prose():
    """Miss 6: "x ≈ 1451 px" and "x=1449 px" are the same block, and used to score as a clash."""
    from canopy.digitize.digitizer import _late_window_provenance

    cressman = [_readout_sample("A", "m", x_read="Block 33 (last block, x ≈ 1451 px)"),
                _readout_sample("A", "m", x_read="Block 33 (last block, x=1449 px)")]
    out = _late_window_provenance(replace(TARGET, x_hint="last block"), cressman, [],
                                  x_tick_px=40.0)
    assert out["late_window_x_agrees"] is True
    assert out["late_window_x_compared"] == "pixels"
    assert out["late_window_x_spread_px"] == 2.0
    assert len(out["late_window_x_read"]) == 2       # the prose still disagreed, and is recorded

    # …and two genuinely different blocks phrased identically do NOT agree
    apart = [_readout_sample("A", "m", x_read="the last block (x=1451 px)"),
             _readout_sample("A", "m", x_read="the last block (x=980 px)")]
    assert _late_window_provenance(replace(TARGET, x_hint="last block"), apart, [],
                                   x_tick_px=40.0)["late_window_x_agrees"] is False


def test_the_legend_supplies_the_dispersion_type_the_mapper_could_not(bar_figure, tmp_path):
    """Miss 10: UNKNOWN dispersion is a needs_human factory — `_sd_of` returns None for it."""
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    payload = _readout_payload(31.5, 11.0, 12.25, 11.75)
    payload["legend_says"] = "error bars are the standard deviation of the mean"
    provider = _scripted(payload, _coord_payload(bar_figure, view.scale))
    blind = SOURCE.model_copy(update={"error_bar_type": DispersionType.UNKNOWN})
    out = digitize(_client(provider), paper, fig, TARGET, source=blind, dataset=DATASET,
                   out_dir=tmp_path, result=True)
    ensemble = next(c for c in out.candidates if c.extractor_id == "digitize:ensemble")
    assert ensemble.dispersion_type is DispersionType.SD
    assert ensemble.pixel_provenance["dispersion_type_from"] == "legend"
    assert ensemble.status == "found", "the legend agreeing with itself is not a conflict"


# ------------------------------------------------------------------ task 16 (g): categorical x
def _categorical_payload(a_points, b_points, unit="deg"):
    """A read-out that reports EVERY point of each series, as a categorical axis needs."""
    def series(group, label, points):
        return {"group": group, "label_read": label,
                "mean": statistics.fmean([m for m, _ in points]),
                "error_half_length": None, "error_upper": None, "error_lower": None,
                "error_sides": "both", "x_read": "all eight target directions",
                "points": [{"x_label": f"{i * 45}deg", "mean": m, "error_half_length": e}
                           for i, (m, e) in enumerate(points)],
                "confidence": 0.8, "notes": ""}

    return {"status": "found", "panel": "Fig 2a", "unit": unit,
            "axis_read": "left y-axis, adaptation (deg)", "axis_direction_note": "",
            "legend_says": "error bars are SE", "tick_labels": [0, 10, 20, 30, 40],
            "pixel_resolution_estimate": 0.1, "confidence": 0.8, "notes": "",
            "groups": [series("A", "young: filled circles", a_points),
                       series("B", "older: open circles", b_points)]}


CATEGORICAL_SOURCE = SOURCE.model_copy(update={"x_axis_kind": "categorical",
                                               "kind": SourceKind.figure_points,
                                               "error_bar_type": DispersionType.SE})


def test_a_categorical_x_axis_with_the_mode_OFF_is_never_a_wrong_number(bar_figure, tmp_path):
    """Acceptance item 15: `not_convertible`, with the reason, and without spending a penny."""
    from canopy.digitize.digitizer import CATEGORICAL_UNSUPPORTED

    paper, fig = _paper_for(bar_figure)
    provider = FakeProvider([])                       # any model call at all would raise
    out = digitize(_client(provider), paper, fig, TARGET, source=CATEGORICAL_SOURCE,
                   dataset=DATASET, out_dir=tmp_path, result=True)
    assert out.cost_usd == 0.0 and out.samples == []
    assert {c.group for c in out.candidates} == {"A", "B"}
    for cand in out.candidates:
        assert cand.mean is None and cand.dispersion_value is None
        assert cand.status == "ambiguous"
        assert cand.pixel_provenance[CATEGORICAL_UNSUPPORTED] is True
        # the refusal now names BOTH shapes a "categorical" x axis can be, because the remedy for
        # one of them (turn the collapse on) destroys the other (F1): on a group chart averaging
        # across the axis gives both arms the same mean
        assert "average across them" in cand.notes
        assert "GROUPS THEMSELVES" in cand.notes


def test_a_categorical_x_axis_with_the_mode_ON_reads_every_point_and_averages_them(bar_figure,
                                                                                   tmp_path):
    """Acceptance item 13: eight points per series, averaged in CODE, dispersion named as built."""
    from canopy.digitize.digitizer import MEAN_OF_POINT_SD

    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    a = [(12.0, 2.0), (14.0, 2.4), (11.0, 1.8), (13.0, 2.2),
         (12.5, 2.1), (13.5, 2.3), (11.5, 1.9), (14.5, 2.5)]
    b = [(20.0, 3.0), (22.0, 3.4), (19.0, 2.8), (21.0, 3.2),
         (20.5, 3.1), (21.5, 3.3), (19.5, 2.9), (22.5, 3.5)]
    provider = _scripted(_categorical_payload(a, b), _coord_payload(bar_figure, view.scale))
    target = replace(TARGET, collapse_across_x=True)
    out = digitize(_client(provider), paper, fig, target, source=CATEGORICAL_SOURCE,
                   dataset=DATASET, out_dir=tmp_path, result=True)

    ensembles = {c.group: c for c in out.candidates if c.extractor_id == "digitize:ensemble"}
    assert ensembles["A"].mean == pytest.approx(statistics.fmean([m for m, _ in a]), abs=1e-9)
    assert ensembles["A"].dispersion_value == pytest.approx(
        statistics.fmean([e for _, e in a]), abs=1e-9)
    for group in ("A", "B"):
        provenance = ensembles[group].pixel_provenance
        assert provenance["collapsed_across_x"] is True
        assert provenance["n_points"] == 8
        assert provenance["dispersion_approximation"] == MEAN_OF_POINT_SD
    # the pixel routes read ONE point, so they are not a vote on the average across the axis
    dropped = [s for s in out.samples if s.route in ("B", "C") and s.dropped]
    assert dropped and all("average across the categorical" in s.drop_reason for s in dropped)


def test_the_collapsed_row_is_flagged_capped_and_can_be_excluded_by_the_pooler():
    """Acceptance item 13's tail: the statistic is an approximation and the row says so."""
    from canopy.pipeline.run import DISPERSION_APPROXIMATED, _approximation_flags
    from canopy.verify.checks import codes, run_checks
    from canopy.verify.confidence import CAPPING_FLAGS
    from canopy.models import (Candidate, DatasetSpec as DS, GroupSpec as GS,
                               OutcomeSources)

    dataset = DS(dataset_id="d1", cluster_id="p", group_a=GS(label="young", n=20),
                 group_b=GS(label="older", n=20),
                 outcomes=[OutcomeSources(outcome_key="late_adaptation", units="deg")])
    collapsed = Candidate(
        candidate_id="c1", paper_id="p", dataset_id="d1", outcome_key="late_adaptation",
        kind="group_stats", group="A", status="found", source_kind=SourceKind.figure_points,
        n=20, mean=12.75, dispersion_value=2.15, dispersion_type=DispersionType.SE, unit="deg",
        route="figure", extractor_id="digitize:ensemble",
        pixel_provenance={"collapsed_across_x": True, "n_points": 8,
                          "dispersion_approximation": "mean_of_point_sd"})
    flags = run_checks(dataset, "late_adaptation", [collapsed])
    assert "collapsed_across_x" in codes(flags)
    assert "collapsed_across_x" in CAPPING_FLAGS
    assert "overstates the denominator" in next(
        f for f in flags if f.code == "collapsed_across_x").message
    row_flags = _approximation_flags([collapsed])
    assert DISPERSION_APPROXIMATED in row_flags
    assert f"{DISPERSION_APPROXIMATED}:mean_of_point_sd" in row_flags


# ---------------------------------------------------- F1: when the x categories ARE the groups
def _group_chart_payload(points_a, points_b, mean_a=None, mean_b=None, x_read=None):
    """A read-out of a figure whose x axis is the comparison itself: one bar per group.

    `points` carries what the reader was asked for when the collapse mode is on — every point on
    the x axis. On this figure shape those points are the two groups' own bars.
    """
    def series(group, label, points, mean):
        return {"group": group, "label_read": label, "mean": mean,
                "error_half_length": None, "error_upper": None, "error_lower": None,
                "error_sides": "both",
                "x_read": x_read if x_read is not None else f"the {label} bar",
                "points": [{"x_label": x, "mean": m, "error_half_length": e}
                           for x, m, e in points],
                "confidence": 0.8, "notes": ""}

    return {"status": "found", "panel": "Fig 1", "unit": "deg",
            "axis_read": "left y-axis (deg)", "axis_direction_note": "",
            "legend_says": "error bars are SD", "tick_labels": [0, 10, 20, 30, 40, 50, 60],
            "pixel_resolution_estimate": 0.1, "confidence": 0.8, "notes": "",
            "groups": [series("A", "old", points_a, mean_a),
                       series("B", "young", points_b, mean_b)]}


def _ensembles(out):
    return {c.group: c for c in out.candidates if c.extractor_id == "digitize:ensemble"}


def test_a_categorical_axis_whose_categories_are_the_groups_is_read_as_a_group_chart(bar_figure,
                                                                                     tmp_path):
    """F1: one bar per group is not an axis to average over — it is the comparison itself.

    With the collapse mode on, `_collapse_points` needed two points per series and each group has
    exactly one bar, so the cell produced nothing after paying for the reads, and every pixel
    route — which had read the bars correctly — was dropped as "reading one point". The categories
    the reader names are what settles it: they are the two groups the protocol is comparing.
    """
    from canopy.digitize.digitizer import CATEGORICAL_GROUPS

    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    payload = _group_chart_payload([("old", 31.5, 11.0)], [("young", 12.25, 11.75)])
    provider = _scripted(payload, _coord_payload(bar_figure, view.scale))
    target = replace(TARGET, collapse_across_x=True)
    out = digitize(_client(provider), paper, fig, target, source=CATEGORICAL_SOURCE,
                   dataset=DATASET, out_dir=tmp_path, result=True)

    ens = _ensembles(out)
    assert ens["A"].mean == pytest.approx(31.5, abs=0.6)
    assert ens["B"].mean == pytest.approx(12.25, abs=0.6)
    assert ens["A"].mean != ens["B"].mean
    provenance = ens["A"].pixel_provenance
    assert provenance["categorical_x_role"] == CATEGORICAL_GROUPS
    assert "each category is a group's own value" in provenance["categorical_x_role_why"] or \
           "one point per group" in provenance["categorical_x_role_why"]
    assert not provenance.get("collapsed_across_x")
    # the pixel routes read one datum per group, which on this figure IS the quantity
    assert not [s for s in out.samples if s.route in ("B", "C") and s.dropped]


def test_a_reader_that_lists_both_bars_for_both_groups_cannot_produce_d_equal_zero(bar_figure,
                                                                                   tmp_path):
    """F1's worst branch: "read every point on the x axis" is followed literally on a group chart.

    Both group rows come back with the same two points, so both averages are `(A + B) / 2` — one
    number reported twice, Cohen's d exactly 0.0, every route in perfect agreement with every
    other, and nothing downstream comparing group A with group B to catch it. The categories name
    the groups, so each group is read at its own.
    """
    both = [("old", 31.5, 11.0), ("young", 12.25, 11.75)]
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(_group_chart_payload(both, list(both)),
                         _coord_payload(bar_figure, view.scale))
    target = replace(TARGET, collapse_across_x=True)
    out = digitize(_client(provider), paper, fig, target, source=CATEGORICAL_SOURCE,
                   dataset=DATASET, out_dir=tmp_path, result=True)
    ens = _ensembles(out)
    assert ens["A"].mean == pytest.approx(31.5, abs=0.6)
    assert ens["B"].mean == pytest.approx(12.25, abs=0.6)
    assert ens["A"].mean != ens["B"].mean, "both arms took the average of the same two bars"


def test_a_reader_that_names_its_own_category_settles_the_axis_without_listing_points(bar_figure,
                                                                                        tmp_path):
    """The commonest shape of the same evidence: the reader gives one mean per group and says
    which x category it read it at. Each group naming its OWN category — and not the other's — is
    the axis telling us the categories are the groups."""
    from canopy.digitize.digitizer import CATEGORICAL_GROUPS

    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    payload = _readout_payload(31.5, 11.0, 12.25, 11.75)
    for row, x_read in zip(payload["groups"], ("the 'old' bar", "the 'young' bar")):
        row["points"] = []
        row["x_read"] = x_read
    provider = _scripted(payload, _coord_payload(bar_figure, view.scale))
    target = replace(TARGET, collapse_across_x=True)
    out = digitize(_client(provider), paper, fig, target, source=CATEGORICAL_SOURCE,
                   dataset=DATASET, out_dir=tmp_path, result=True)
    ens = _ensembles(out)
    assert ens["A"].pixel_provenance["categorical_x_role"] == CATEGORICAL_GROUPS
    assert ens["A"].mean == pytest.approx(31.5, abs=0.6)
    assert ens["B"].mean == pytest.approx(12.25, abs=0.6)


def test_two_groups_read_off_the_same_points_produce_no_value_at_all(bar_figure, tmp_path):
    """The same shape with categories that name nothing — the role cannot be resolved from them,
    so the code falls back to the structural rule: two arms whose points are the very same points
    are one series read twice, whatever the axis turns out to be."""
    from canopy.digitize.digitizer import CATEGORICAL_CONDITIONS

    twins = [("1", 31.5, 11.0), ("2", 12.25, 11.75)]
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(
        _group_chart_payload(twins, list(twins), x_read="all the categories on the x axis"),
        _coord_payload(bar_figure, view.scale))
    target = replace(TARGET, collapse_across_x=True)
    out = digitize(_client(provider), paper, fig, target, source=CATEGORICAL_SOURCE,
                   dataset=DATASET, out_dir=tmp_path, result=True)
    ens = _ensembles(out)
    assert ens["A"].mean is None and ens["B"].mean is None
    assert ens["A"].status == "ambiguous"
    assert any("SAME points for both groups" in s.notes
               for s in out.samples if s.route == "D")
    assert ens["A"].pixel_provenance["categorical_x_role"] == CATEGORICAL_CONDITIONS


def test_the_last_net_refuses_two_arms_that_collapsed_to_one_number(bar_figure):
    """Belt and braces for the same failure, stated on the OUTPUT rather than on the input: two
    collapsed arms that agree to the last digit on the mean AND the spread are one series
    reported twice, and the effect size between them is zero by construction."""
    from canopy.digitize.digitizer import _refuse_identical_collapse

    paper, fig = _paper_for(bar_figure)
    collapsed = {"collapsed_across_x": True, "n_points": 2}
    twins = [_ensemble_of(bar_figure, [_d_sample(17.7, 2.45, extra=dict(collapsed))],
                          {"cal_status": "confirmed"}) for _ in range(2)]
    twins[1].group = "B"
    _refuse_identical_collapse(twins)
    for cand in twins:
        assert cand.mean is None and cand.dispersion_value is None
        assert cand.status == "ambiguous"
        assert cand.pixel_provenance["identical_collapse_refused"] is True
        assert "one series reported twice" in cand.notes


def test_an_explicit_categorical_x_of_groups_skips_the_refusal_without_the_collapse_flag(
        bar_figure, tmp_path):
    """The data-model path: when a caller states the categories ARE the groups, the figure is an
    ordinary group chart and the free refusal — which exists for the conditions case — does not
    apply to it."""
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(_readout_payload(31.5, 11.0, 12.25, 11.75),
                         _coord_payload(bar_figure, view.scale))
    target = replace(TARGET, categorical_x="groups")       # collapse mode still OFF
    out = digitize(_client(provider), paper, fig, target, source=CATEGORICAL_SOURCE,
                   dataset=DATASET, out_dir=tmp_path, result=True)
    ens = _ensembles(out)
    assert ens["A"].mean == pytest.approx(31.5, abs=0.6)
    assert ens["B"].mean == pytest.approx(12.25, abs=0.6)
    assert not any(c.pixel_provenance.get("categorical_x_unsupported") for c in out.candidates)


def test_the_sign_of_two_panels_of_one_paper_is_read_off_the_figure_not_assumed(bar_figure,
                                                                                tmp_path):
    """Acceptance item 14: Heuer 2a and 2b have OPPOSITE signs on the same paper."""
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    aftereffect_a = [(-4.0, 1.0), (-3.0, 0.9), (-5.0, 1.1), (-3.5, 1.0),
                     (-4.5, 1.05), (-3.2, 0.95), (-4.8, 1.15), (-3.8, 1.0)]
    aftereffect_b = [(-2.0, 0.8), (-1.5, 0.7), (-2.5, 0.9), (-1.8, 0.75),
                     (-2.2, 0.85), (-1.6, 0.72), (-2.4, 0.88), (-1.9, 0.78)]
    provider = _scripted(_categorical_payload(aftereffect_a, aftereffect_b),
                         _coord_payload(bar_figure, view.scale))
    target = replace(TARGET, collapse_across_x=True, outcome_key="aftereffect")
    out = digitize(_client(provider), paper, fig, target, source=CATEGORICAL_SOURCE,
                   dataset=DATASET, out_dir=tmp_path, result=True)
    ensembles = {c.group: c for c in out.candidates if c.extractor_id == "digitize:ensemble"}
    assert ensembles["A"].mean < 0 and ensembles["B"].mean < 0, "the sign was not preserved"
    assert ensembles["A"].mean == pytest.approx(
        statistics.fmean([m for m, _ in aftereffect_a]), abs=1e-9)
    assert ensembles["A"].dispersion_value > 0, "a half-length is a magnitude"


# ------------------------------------------------------------------ review fix: symmetric ladder
def test_the_magnitude_rule_refuses_a_ladder_the_data_sits_BELOW_as_well_as_above():
    """The first cut tested the upper side only, in absolute value, so this walked through."""
    from canopy.digitize.digitizer import _choose_calibration

    ladder = _cal([(268.0, 4.0), (412.0, 3.0), (557.0, 2.0), (702.0, 1.0)])
    core = _core_with(ladder, rows=[268.0, 412.0, 557.0, 702.0])
    below = _choose_calibration(core, None, None, [
        _readout({"A": -31.3, "B": -33.3}, [], variant="direct"),
        _readout({"A": -31.3, "B": -33.3}, [], variant="ticks_first")])
    assert below.status == "cal_refuted" and below.refutation["side"] == "below"
    above = _choose_calibration(core, None, None, [
        _readout({"A": 31.3, "B": 33.3}, [], variant="direct"),
        _readout({"A": 31.3, "B": 33.3}, [], variant="ticks_first")])
    assert above.status == "cal_refuted" and above.refutation["side"] == "above"


def test_a_group_that_is_off_the_end_alone_does_not_condemn_the_ladder():
    """One series off the frame is a bad read of that series; the whole cell off it is the ladder."""
    from canopy.digitize.digitizer import _choose_calibration

    ladder = _cal([(100.0, 40.0), (200.0, 30.0), (300.0, 20.0), (400.0, 10.0)])
    core = _core_with(ladder, rows=[100.0, 200.0, 300.0, 400.0])
    choice = _choose_calibration(core, None, None, [
        _readout({"A": 25.0, "B": 900.0}, [], variant="direct"),
        _readout({"A": 25.0, "B": 900.0}, [], variant="ticks_first")])
    assert choice.status != "cal_refuted"


def test_two_readers_reporting_the_same_ladder_outrank_one_tesseract_pass():
    """A ladder read uniformly too LARGE has values inside its own frame — nothing refutes it.

    What settles it is the other witness: two models reading the same printed labels and agreeing
    beat one OCR pass. Preferring `cv_ocr` on a tie is what read 45/35/25/15 as 4/3/2/1.
    """
    from canopy.digitize.digitizer import _choose_calibration

    inflated = _cal([(500.0, 10.0), (400.0, 20.0), (300.0, 30.0), (200.0, 40.0), (100.0, 50.0)])
    rows = [100.0, 200.0, 300.0, 400.0, 500.0]
    core = _core_with(inflated, rows=rows)
    truth = [5, 4, 3, 2, 1]
    choice = _choose_calibration(core, None, None, [
        _readout({"A": 3.0, "B": 4.2}, truth, variant="direct"),
        _readout({"A": 3.0, "B": 4.2}, truth, variant="ticks_first")])
    assert choice.source == "readout_ticks"
    assert sorted(v for _, v in choice.cal.ticks) == [1.0, 2.0, 3.0, 4.0, 5.0]
    # …and a lone read-out ladder does NOT get to overturn the OCR pass on its own
    lonely = _choose_calibration(core, None, None,
                                 [_readout({"A": 3.0}, truth, variant="direct")])
    assert lonely.source == "cv_ocr"


def test_two_series_read_onto_each_others_rows_are_caught():
    """A clean A/B transposition: both readings are real, both are on the wrong group."""
    from canopy.digitize.cv import Marker
    from canopy.digitize.digitizer import _series_identity

    core = _core_with(None)
    # the OPEN marker is high on the plot (small y), the FILLED one low
    core.markers = [Marker(x=100.0, y=50.0, colour="#ffffff", kind="open", size=6.0),
                    Marker(x=100.0, y=250.0, colour="#000000", kind="square", size=6.0)]
    swapped = [RouteSample(route="C", group="A", model="m", mean=40.0, x_px=100.0, y_px=50.0,
                           label_read="Young: filled black squares"),
               RouteSample(route="C", group="B", model="m", mean=10.0, x_px=100.0, y_px=250.0,
                           label_read="Elderly: open white squares")]
    info = _series_identity(swapped, core)
    assert info["transposed"] is True
    assert "transposed" in " ".join(info["notes"])
    assert info["conflict"] is False

    right_way = [RouteSample(route="C", group="A", model="m", mean=40.0, x_px=100.0, y_px=50.0,
                             label_read="Young: open white squares"),
                 RouteSample(route="C", group="B", model="m", mean=10.0, x_px=100.0, y_px=250.0,
                             label_read="Elderly: filled black squares")]
    assert _series_identity(right_way, core)["transposed"] is False


def test_a_described_marker_the_pixel_pass_cannot_find_is_actionable_not_prose():
    """These notes were written and nothing ever read them — the branch was dead."""
    from canopy.digitize.cv import Marker
    from canopy.digitize.digitizer import _series_identity

    core = _core_with(None)
    core.markers = [Marker(x=10.0, y=20.0, colour="#000000", kind="square", size=6.0)]
    info = _series_identity(
        [RouteSample(route="D", group="A", model="m", mean=1.0,
                     label_read="Young: open white circles"),
         RouteSample(route="D", group="B", model="m", mean=2.0,
                     label_read="Elderly: filled black squares")], core)
    assert info["marker_mismatch"] is True
    assert any("OPEN marker" in note for note in info["notes"])


# ------------------------------------------------------------------ acceptance items 13 / 14
#: Heuer & Hegele 2008, shaped like Fig 2a (adaptation, n = 20/20, SE bars over 8 target
#: directions) and Fig 2b (aftereffect, same design, OPPOSITE sign).
#:
#: **What these payloads are and are not.** They are not a stand-in for the figure: nobody here has
#: Heuer's per-direction values, and choosing point values that land on the published d and then
#: asserting that d would be asserting this file's own arithmetic back at itself. What is under
#: test is the CHAIN — collapse eight points to a mean, average the per-point half-lengths, carry
#: SE with n through to Cohen's d, and keep the sign — against expectations computed independently
#: from the same inputs. Whether the digitiser READS Heuer as these numbers is a question only the
#: live re-run can answer, and the report says so.
HEUER_2A = {
    "A": [(19.0, 1.05), (20.5, 1.10), (18.0, 1.00), (21.0, 1.15),
          (19.5, 1.05), (20.0, 1.10), (18.5, 1.00), (21.5, 1.15)],
    "B": [(22.0, 1.15), (23.5, 1.20), (21.0, 1.10), (24.0, 1.25),
          (22.5, 1.15), (23.0, 1.20), (21.5, 1.10), (24.5, 1.25)]}
HEUER_2B = {
    "A": [(-5.0, 0.62), (-4.5, 0.60), (-5.5, 0.64), (-4.8, 0.61),
          (-5.2, 0.63), (-4.6, 0.60), (-5.4, 0.64), (-5.0, 0.62)],
    "B": [(-5.8, 0.66), (-5.4, 0.64), (-6.2, 0.68), (-5.6, 0.65),
          (-6.0, 0.67), (-5.5, 0.64), (-6.1, 0.68), (-5.9, 0.66)]}


def _expected_d(points: dict, n: int = 20) -> float:
    """Cohen's d from the per-point readings, computed here rather than taken from the code."""
    import math

    from canopy.stats.effect_sizes import cohens_d

    def summary(rows):
        return (statistics.fmean([m for m, _ in rows]),
                statistics.fmean([e for _, e in rows]) * math.sqrt(n))

    (mean_a, sd_a), (mean_b, sd_b) = summary(points["A"]), summary(points["B"])
    return cohens_d(mean_a, sd_a, n, mean_b, sd_b, n)


def _collapsed_ensembles(points, bar_figure, tmp_path, outcome_key):
    paper, fig = _paper_for(bar_figure)
    view = FigureView(bar_figure["path"])
    provider = _scripted(_categorical_payload(points["A"], points["B"]),
                         _coord_payload(bar_figure, view.scale))
    dataset = DatasetSpec(dataset_id="d1", group_a=GroupSpec(label="young", n=20, n_evidence="20"),
                          group_b=GroupSpec(label="older", n=20, n_evidence="20"))
    target = replace(TARGET, collapse_across_x=True, outcome_key=outcome_key)
    out = digitize(_client(provider), paper, fig, target,
                   source=CATEGORICAL_SOURCE.model_copy(
                       update={"error_bar_type": DispersionType.SE}),
                   dataset=dataset, out_dir=tmp_path, result=True)
    return {c.group: c for c in out.candidates if c.extractor_id == "digitize:ensemble"}


def test_the_collapsed_chain_carries_eight_points_through_to_an_effect_size(bar_figure, tmp_path):
    """Acceptance item 13, on the arithmetic the categorical mode is responsible for."""
    import math

    from canopy.stats.effect_sizes import cohens_d

    ensembles = _collapsed_ensembles(HEUER_2A, bar_figure, tmp_path / "a", "late_adaptation")
    for group in ("A", "B"):
        assert ensembles[group].pixel_provenance["n_points"] == 8
        assert ensembles[group].dispersion_type is DispersionType.SE
        assert ensembles[group].n == 20
    d = cohens_d(ensembles["A"].mean, ensembles["A"].dispersion_value * math.sqrt(20), 20,
                 ensembles["B"].mean, ensembles["B"].dispersion_value * math.sqrt(20), 20)
    assert d == pytest.approx(_expected_d(HEUER_2A), abs=1e-9)
    assert d < 0, "the adaptation panel has the older group higher, so d is negative"


def test_the_two_heuer_panels_come_out_with_opposite_signs(bar_figure, tmp_path):
    """Acceptance item 14: 2a and 2b are the same design and the sign flips between them.

    A collapse that dropped or normalised the sign would pass every magnitude assertion in this
    file and still put the aftereffect on the wrong side of zero.
    """
    import math

    from canopy.stats.effect_sizes import cohens_d

    def effect(points, key, where):
        ens = _collapsed_ensembles(points, bar_figure, tmp_path / where, key)
        return cohens_d(ens["A"].mean, ens["A"].dispersion_value * math.sqrt(20), 20,
                        ens["B"].mean, ens["B"].dispersion_value * math.sqrt(20), 20), ens

    adaptation, ens_a = effect(HEUER_2A, "late_adaptation", "2a")
    aftereffect, ens_b = effect(HEUER_2B, "aftereffect", "2b")
    assert adaptation == pytest.approx(_expected_d(HEUER_2A), abs=1e-9)
    assert aftereffect == pytest.approx(_expected_d(HEUER_2B), abs=1e-9)
    assert adaptation < 0 < aftereffect, (adaptation, aftereffect)
    # the aftereffect panel's own values are negative, and that survives the collapse
    assert ens_b["A"].mean < 0 and ens_b["B"].mean < 0
    assert ens_b["A"].dispersion_value > 0, "a half-length is a magnitude, whatever the mean's sign"
    assert ens_a["A"].mean > 0
