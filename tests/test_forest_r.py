"""Task 9 — the R `meta::forest.meta` renderer, its labels, its cross-check and its fallback.

Everything here is offline. The tests that need R are marked `slow` and skip themselves when
`Rscript` or `meta` is missing, so the suite is the same suite on a machine that has neither —
which is the point of the fallback they are testing.

`runs/nine`'s late-adaptation rows are the input throughout: they are two real rows from a real
run, with the moderator strings, the missing values and the wide interval a hand-written pair
would not have.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

import pytest

from canopy.report import forest_r, forest_render, labels
from tests.helpers import nine

REPO = Path(__file__).resolve().parents[1]
FIX = REPO / "tests" / "fixtures" / "forest_r"


def _strict_late():
    """`(rows, pooled, outcome, settings, protocol)` — the strict late-adaptation analysis."""
    from canopy.pipeline.run import _split_rows
    from canopy.report.tables import pool_rows

    protocol = nine.protocol()
    split = _split_rows(nine.records("late_adaptation"), protocol.stats)
    return (split.primary, pool_rows(split.primary, protocol.stats),
            protocol.outcome("late_adaptation"), protocol.stats, protocol)


def _best_guess_late():
    """`(rows, pooled, added_ids, outcome, settings, protocol)` — the second line's own rows."""
    from canopy.pipeline.run import _split_rows
    from canopy.report.outputs import _best_guess_line
    from canopy.report.tables import pool_rows

    protocol = nine.protocol()
    outcome = protocol.outcome("late_adaptation")
    split = _split_rows(nine.records("late_adaptation"), protocol.stats)
    rows, _decisions, added, _cells, _cell_guesses, _fired = _best_guess_line(
        split.primary_pre_agg, split.held, outcome, protocol.stats)
    return (rows, pool_rows(rows, protocol.stats), [r.dataset_id for r in added], outcome,
            protocol.stats, protocol)


def _r_result(pooled, **overrides):
    """What `forest_meta.R` would print if R agreed with us exactly, minus any override."""
    result = {"k": pooled.k, "TE_random": pooled.estimate, "lower_random": pooled.ci_low,
              "upper_random": pooled.ci_high, "tau2": pooled.tau2, "I2": pooled.I2,
              "Q": pooled.Q, "Q_df": pooled.Q_df, "pi_low": None, "pi_high": None,
              "df_predict": pooled.k - 2, "df_predict_from": "meta"}
    result.update(overrides)
    return result


# --------------------------------------------------------------------------- labels
def test_leftcols_come_from_protocol_moderators():
    p = nine.protocol()
    c = labels.forest_columns(p, p.outcome("late_adaptation"), p.stats)
    assert c.leftcols == ["studlab", "year", "mod_task_type", "mod_perturbation_size_deg",
                          "mod_n_targets", "n_label"]
    assert c.leftlabs == ["Author", "Year", "Task type", "Perturbation size (deg)", "N targets",
                          "N (O/Y)"]
    assert c.rightcols == ["effect", "ci", "w.random"]
    assert c.rightlabs == ["Cohen's d", "95% CI", "Weight"]
    assert (c.label_left, c.label_right) == ("Reduced in Old", "Enhanced in Old")
    p.moderators = []
    assert labels.forest_columns(p, p.outcome("late_adaptation"),
                                 p.stats).leftcols == ["studlab", "year", "n_label"]


def test_both_renderers_head_their_columns_the_same_way(tmp_path):
    """The R plot's headings and the matplotlib plot's are one list, not two that agree today."""
    rows, pooled, outcome, settings, protocol = _strict_late()
    columns = labels.forest_columns(protocol, outcome, settings)
    from canopy.report.forest import forest_plot

    svg = forest_plot(rows, pooled, outcome, settings, tmp_path / "forest",
                      moderators=protocol.moderators,
                      columns=columns)["svg"].read_text(encoding="utf-8")
    for heading in ("Author", "Year", "Task type", "N (O/Y)", "Cohen's d", "Weight"):
        assert heading in svg, heading


# --------------------------------------------------------------------------- the x axis
def test_xlim_is_data_driven_unless_overridden():
    assert forest_r.xlim_for([(-2.47, 1.00)], None) == [-2.5, 2.5]
    assert forest_r.xlim_for([(-0.2, 0.3)], None) == [-1.0, 1.0]
    assert forest_r.xlim_for([(-2.47, 1.0)], [-4, 4]) == [-4.0, 4.0]
    with pytest.raises(ValueError):
        forest_r.xlim_for([(-1.0, 1.0)], [4, -4])


# --------------------------------------------------------------------------- the inputs
def test_forest_inputs_match_golden(tmp_path):
    rows, pooled, outcome, settings, protocol = _strict_late()
    csv, opts = forest_r.write_forest_inputs(rows, pooled, outcome, settings, protocol,
                                             line="strict", best_guess_ids=(), out_dir=tmp_path)
    assert csv.read_bytes() == (FIX / "late_adaptation.rows.csv").read_bytes()
    assert opts.read_bytes() == (FIX / "late_adaptation.options.json").read_bytes()
    options = json.loads(opts.read_text(encoding="utf-8"))
    assert options["method_predict"] == "HTS"
    assert options["digits"] == 1
    assert options["xlim"] == [-4.0, 4.0]
    assert options["sortvar"] == "TE" and options["weight_study"] == "random"
    assert options["print_tau2"] is False and options["fontfamily"] == "Helvetica"
    # sorted by TE, so R's own `sortvar` has nothing left to reorder under the per-row vectors
    written = [line.split(",")[0] for line in csv.read_text(encoding="utf-8").splitlines()[1:]]
    assert written == [r.dataset_id for r in forest_r.row_order(rows)]


def test_best_guess_options_use_subgroup_and_distinct_style(tmp_path):
    rows, pooled, added, outcome, settings, protocol = _best_guess_late()
    assert len(added) >= 1 and pooled.k >= 3
    csv, opts = forest_r.write_forest_inputs(rows, pooled, outcome, settings, protocol,
                                             line="best_guess", best_guess_ids=added,
                                             out_dir=tmp_path)
    options = json.loads(opts.read_text(encoding="utf-8"))
    assert options["subgroup"] == "analysis_line"
    assert options["subgroup_levels"] == ["Confirmed", "Best guess"]
    assert options["weight_study"] == "random"
    ordered = forest_r.row_order(rows)
    assert options["type_study"] == ["circle" if r.dataset_id in added else "square"
                                     for r in ordered]
    assert {shape for shape in options["type_study"]} == {"circle", "square"}
    assert len(set(options["col_square"])) == 2          # a guessed square is not the plain one
    assert options["text_addline1"].startswith("Best guess, not the primary analysis")
    lines = csv.read_text(encoding="utf-8").splitlines()
    column = lines[0].split(",").index("analysis_line")
    levels = {row.split(",")[0]: row.split(",")[column] for row in lines[1:]}
    assert {k: v for k, v in levels.items() if v == "Best guess"}.keys() == set(added)


def test_the_r_forest_marks_an_overridden_row_and_keys_the_marker(tmp_path):
    """An override has to be visible on the PUBLISHED plot, which is R's whenever R is installed.

    `forest_render.render_forest` prefers this renderer, and its inputs carried no override marking
    at all: a row a human had corrected came out byte-identical to one the tool read, while the
    matplotlib fallback marked it — so one review said two different things about its own provenance
    depending on which renderer the machine could reach. `forest.meta` has no glyph legend to key
    the marker in, so the key goes on the additional line it does have.
    """
    import csv

    from canopy.report.theme import OVERRIDE_KEY, OVERRIDE_MARK

    rows, pooled, outcome, settings, protocol = _strict_late()
    marked = rows[0].model_copy(update={"flags": [*rows[0].flags, "human_override"]})
    rows_csv, opts = forest_r.write_forest_inputs([marked, *rows[1:]], pooled, outcome, settings,
                                                  protocol, out_dir=tmp_path)
    written = {row["dataset_id"]: row["studlab"]
               for row in csv.DictReader(rows_csv.open(newline="", encoding="utf-8"))}
    assert written[marked.dataset_id].endswith(OVERRIDE_MARK)
    assert [label for label in written.values() if OVERRIDE_MARK in label] == \
           [written[marked.dataset_id]]
    options = json.loads(opts.read_text(encoding="utf-8"))
    assert OVERRIDE_KEY in options["text_addline2"]

    # …and nothing is marked or keyed on the same rows without the flag
    plain, _opts = forest_r.write_forest_inputs(rows, pooled, outcome, settings, protocol,
                                                out_dir=tmp_path / "plain")
    assert OVERRIDE_MARK not in plain.read_text(encoding="utf-8")


def test_the_override_key_does_not_displace_the_prediction_interval(tmp_path):
    """Both belong under the plot and meta has two lines: the key joins the PI, it does not evict it."""
    from canopy.report.theme import OVERRIDE_KEY

    rows, pooled, added, outcome, settings, protocol = _best_guess_late()
    # `S` is a prediction interval canopy computes and `PREDICT` does not map, so the figure prints
    # ours under the plot — the one case where both addlines are already spoken for
    settings = settings.model_copy(update={"pi_method": "S"})
    marked = [rows[0].model_copy(update={"flags": [*rows[0].flags, "human_override"]}), *rows[1:]]
    _csv, opts = forest_r.write_forest_inputs(marked, pooled, outcome, settings, protocol,
                                              line="best_guess", best_guess_ids=added,
                                              out_dir=tmp_path)
    options = json.loads(opts.read_text(encoding="utf-8"))
    assert options["text_addline1"].startswith("Best guess, not the primary analysis")
    assert "prediction interval" in options["text_addline2"]
    assert options["text_addline2"].endswith(OVERRIDE_KEY)


def test_the_strict_line_has_no_subgroup_and_no_caveat(tmp_path):
    rows, pooled, outcome, settings, protocol = _strict_late()
    _csv, opts = forest_r.write_forest_inputs(rows, pooled, outcome, settings, protocol,
                                              out_dir=tmp_path, line="strict")
    options = json.loads(opts.read_text(encoding="utf-8"))
    assert options["subgroup"] is None and options["subgroup_levels"] is None
    assert options["text_addline1"] == "" and options["type_study"] == ["square", "square"]


def test_unmapped_pi_method_disables_r_prediction(tmp_path):
    assert forest_r.PREDICT == {"HTS": "HTS", "V": "V", "z": "S"}
    assert forest_r.PREDICT["z"] == "S"
    rows, pooled, added, outcome, settings, protocol = _best_guess_late()
    settings = settings.model_copy(update={"pi_method": "bootstrap"})   # not in PREDICT
    _csv, opts = forest_r.write_forest_inputs(rows, pooled, outcome, settings, protocol,
                                              line="best_guess", best_guess_ids=added,
                                              out_dir=tmp_path)
    options = json.loads(opts.read_text(encoding="utf-8"))
    assert options["prediction"] is False
    assert forest_r.pi_drawn_by(settings) == "canopy"
    assert forest_r.pi_drawn_by(nine.protocol().stats) == "meta"


def test_no_prediction_is_asked_of_meta_below_three_studies(tmp_path):
    """We refuse a prediction interval below k = 3 and so does meta; both plots say the same."""
    rows, pooled, outcome, settings, protocol = _strict_late()
    _csv, opts = forest_r.write_forest_inputs(rows, pooled, outcome, settings, protocol,
                                              out_dir=tmp_path)
    assert pooled.k == 2 and json.loads(opts.read_text())["prediction"] is False


# --------------------------------------------------------------------------- the cross-check
def test_crosscheck_raises_on_disagreement():
    _rows, pooled, _o, settings, _p = _strict_late()
    with pytest.raises(forest_r.ForestCrossCheckError) as e:
        forest_r._crosscheck({"k": 2, "TE_random": -0.7358, "lower_random": -2.4714,
                              "upper_random": 0.9958, "tau2": 1.4013, "I2": 0.8941,
                              "Q": 9.4417, "Q_df": 1, "pi_low": None, "pi_high": None,
                              "df_predict": None}, pooled, settings)
    assert "estimate" in str(e.value) and "-0.7358" in str(e.value)
    assert e.value.crosscheck["ok"] is False
    assert e.value.crosscheck["failed"] == ["estimate"]


def test_crosscheck_passes_within_tolerance():
    _rows, pooled, _o, settings, _p = _strict_late()
    check = forest_r._crosscheck(_r_result(pooled, TE_random=pooled.estimate + 5e-4), pooled,
                                 settings)
    assert check["ok"] and check["failed"] == []
    assert check["df_predict"] == pooled.k - 2
    assert check["max_abs_diff"] == pytest.approx(5e-4)


def test_crosscheck_fails_when_k_differs():
    _rows, pooled, _o, settings, _p = _strict_late()
    with pytest.raises(forest_r.ForestCrossCheckError) as e:
        forest_r._crosscheck(_r_result(pooled, k=pooled.k + 1), pooled, settings)
    assert "k" in e.value.crosscheck["failed"]


def test_crosscheck_tau2_near_zero_uses_absolute():
    """τ² = 0 vs 5e-7 is the same answer; 0 vs 2e-3 is not, and no ratio can tell them apart."""
    _rows, pooled, _o, settings, _p = _strict_late()
    # k = 2 makes a τ² gap a warning (REML on two studies sits on the boundary), so this is
    # checked on a pool of three, where a disagreement is an error
    three = dataclasses.replace(pooled, k=3, tau2=0.0)
    passes = forest_r._crosscheck(_r_result(three, tau2=5e-7), three, settings)
    assert passes["ok"]
    with pytest.raises(forest_r.ForestCrossCheckError) as e:
        forest_r._crosscheck(_r_result(three, tau2=2e-3), three, settings)
    assert e.value.crosscheck["failed"] == ["tau2"]


def test_a_tau2_gap_on_two_studies_is_a_warning_not_a_failure():
    _rows, pooled, _o, settings, _p = _strict_late()
    check = forest_r._crosscheck(_r_result(pooled, tau2=pooled.tau2 + 0.5), pooled, settings)
    assert check["ok"] and any("tau2" in w for w in check["warnings"])


def test_a_moved_confirmed_diamond_is_a_warning():
    """The best-guess forest's Confirmed subgroup IS the strict line; a shift is worth saying."""
    _rows, pooled, _o, settings, _p = _strict_late()
    check = forest_r._crosscheck(_r_result(pooled, subgroup_TE={"Confirmed": pooled.estimate}),
                                 pooled, settings, strict_pooled=pooled)
    assert check["ok"] and not check["warnings"]
    moved = forest_r._crosscheck(_r_result(pooled,
                                           subgroup_TE={"Confirmed": pooled.estimate + 0.4}),
                                 pooled, settings, strict_pooled=pooled)
    assert moved["ok"] and any("Confirmed" in w for w in moved["warnings"])


# --------------------------------------------------------------------------- the fallback
def test_falls_back_when_rscript_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    forest_r.r_available.cache_clear()
    try:
        rows, pooled, o, s, p = _strict_late()
        paths, info = forest_render.render_forest(rows, pooled, o, s, p, tmp_path / "forest")
    finally:
        forest_r.r_available.cache_clear()
    assert info.renderer == "canopy.report.forest (matplotlib)"
    assert info.reason == "Rscript not found"
    assert (tmp_path / "forest.png").exists()
    assert set(paths) == {"png", "svg", "pdf"}
    assert info.crosscheck is None


def test_falls_back_without_a_protocol(tmp_path):
    """No protocol means no labels to draw the R plot with, so it is not drawn — and it says so."""
    rows, pooled, o, s, _p = _strict_late()
    _paths, info = forest_render.render_forest(rows, pooled, o, s, None, tmp_path / "forest")
    assert info.renderer == "canopy.report.forest (matplotlib)"
    assert "protocol" in info.reason


def test_a_cross_check_failure_falls_back_and_is_recorded_loudly(monkeypatch, tmp_path):
    rows, pooled, o, s, p = _strict_late()

    def explode(*args, **kwargs):
        raise forest_r.ForestCrossCheckError("estimate: canopy -0.7378, R -0.7358",
                                             {"ok": True, "failed": ["estimate"]})

    monkeypatch.setattr(forest_render, "render_forest_r", explode)
    # pretend this machine's R can write every format, so the dispatcher gets as far as calling it
    monkeypatch.setattr(forest_render, "r_available", lambda: forest_r.RInfo(
        rscript="Rscript", r_version="4.4.0", meta_version="8.2.1", formats=forest_r.FORMATS))
    warnings: list[str] = []
    paths, info = forest_render.render_forest(rows, pooled, o, s, p, tmp_path / "forest",
                                              warnings=warnings)
    assert info.renderer == "canopy.report.forest (matplotlib)"
    assert info.crosscheck == {"ok": False, "failed": ["estimate"]}
    assert set(paths) == {"png", "svg", "pdf"} and paths["png"].exists()
    assert warnings and "-0.7358" in warnings[0] and "validate" in warnings[0]


def test_validate_exits_non_zero_when_a_cross_check_failed(tmp_path):
    from canopy.pipeline.run import _forest_crosscheck_failures

    results = tmp_path / "results" / "late_adaptation"
    results.mkdir(parents=True)
    (results / "pooled.json").write_text(json.dumps(
        {"renderer": {"renderer": "canopy.report.forest (matplotlib)",
                      "crosscheck": {"ok": False, "failed": ["estimate", "tau2"]}},
         "renderer_best_guess": None}), encoding="utf-8")
    assert _forest_crosscheck_failures(tmp_path) == ["late_adaptation/strict: estimate, tau2"]
    (results / "pooled.json").write_text(json.dumps(
        {"renderer": {"crosscheck": {"ok": True}}}), encoding="utf-8")
    assert _forest_crosscheck_failures(tmp_path) == []


def test_the_report_says_who_drew_the_plot():
    from canopy.report.html import forest_caption

    caption = forest_caption({"renderer": "R meta::forest.meta", "reason": ""})
    assert caption.endswith("Drawn by R meta::forest.meta")
    reasoned = forest_caption({"renderer": "canopy.report.forest (matplotlib)",
                               "reason": "Rscript not found"})
    assert reasoned.endswith("Drawn by canopy.report.forest (matplotlib) — Rscript not found")
    assert "hollow" in reasoned.lower()                   # …and how to read THAT plot
    assert forest_render.methods_line(
        {"renderer": "R meta::forest.meta", "r_version": "4.4.0", "meta_version": "8.2.1"}) == (
        "Forest plots: R meta::forest.meta (R 4.4.0, meta 8.2.1); REML cross-checked against "
        "canopy.stats.meta to 1e-3.")
    # a fallback has nothing to cross-check, and does not claim to
    fell_back = forest_render.methods_line({"renderer": "canopy.report.forest (matplotlib)",
                                            "reason": "Rscript not found"})
    assert "cross-checked" not in fell_back and "Rscript not found" in fell_back


def test_a_failed_cross_check_gets_a_red_callout_in_the_report():
    from canopy.report.html import _renderer_callout

    callout = _renderer_callout({"renderer": "canopy.report.forest (matplotlib)",
                                 "reason": "estimate: canopy -0.7378, R -0.7358",
                                 "crosscheck": {"ok": False, "failed": ["estimate"]}})
    assert 'class="callout"' in callout and "estimate" in callout
    assert "validate" in callout and "NOT drawn by R" in callout
    assert _renderer_callout({"renderer": "R meta::forest.meta",
                              "crosscheck": {"ok": True}}) == ""
    assert _renderer_callout(None) == ""


# --------------------------------------------------------------------------- the R script
def test_r_script_guards_meta_version():
    assert 'packageVersion("meta") >= "6.0"' in (REPO / "canopy/report/forest_meta.R").read_text()


@pytest.mark.skipif(forest_r.r_available() is None, reason="Rscript/meta not installed")
@pytest.mark.slow
def test_r_renders_and_agrees_on_runs_nine_late(tmp_path):
    """The one test that actually runs R: it draws the fixture's strict forest and agrees."""
    rows, pooled, o, s, p = _strict_late()
    result = forest_r.render_forest_r(rows, pooled, o, s, p, tmp_path / "forest")
    info = result.info
    assert info.renderer == "R meta::forest.meta"
    assert info.crosscheck["ok"]
    assert info.crosscheck["df_predict"] == pooled.k - 2      # HTS: t(k − 2)
    assert info.crosscheck["max_abs_diff"] < 1e-6            # …and not merely inside tolerance
    assert info.r_version and info.meta_version
    assert info.pi_drawn_by == "meta"
    assert result.paths and set(result.paths) <= set(forest_r.FORMATS)
    assert all(path.stat().st_size > 0 for path in result.paths.values())
    # every format this machine could not write is named, so the report can say why
    for fmt in set(forest_r.FORMATS) - set(result.paths):
        assert fmt in info.reason


@pytest.mark.skipif(forest_r.r_available() is None, reason="Rscript/meta not installed")
@pytest.mark.slow
def test_the_marker_and_its_key_reach_the_drawn_r_figure(tmp_path):
    """Writing △ into rows.csv is only half of it: the device has to draw it.

    The SVG is the format a test can read as text, and `svglite` carries the glyph and the key line
    under the plot. The PDF device is a Type1 path that cannot encode it — it drops U+25B3 with a
    `mbcsToSbcs` warning, the same way it already substitutes "..." for the ellipsis every elided
    label carries — so the PDF shows the key's words without its glyph.
    """
    from canopy.report.theme import OVERRIDE_KEY, OVERRIDE_MARK, study_label

    rows, pooled, o, s, p = _strict_late()
    marked = [rows[0].model_copy(update={"flags": [*rows[0].flags, "human_override"]}), *rows[1:]]
    result = forest_r.render_forest_r(marked, pooled, o, s, p, tmp_path / "forest")
    if "svg" not in result.paths:
        pytest.skip("this R has no svg device")
    svg = result.paths["svg"].read_text(encoding="utf-8")
    assert OVERRIDE_KEY in svg
    label = labels.elide(study_label(marked[0]), labels.MAX_LABEL_CH)
    assert f">{label} {OVERRIDE_MARK}<" in svg
    assert f">{label}<" not in svg                     # the marked label is the only one drawn


@pytest.mark.skipif(forest_r.r_available() is None, reason="Rscript/meta not installed")
@pytest.mark.slow
def test_r_draws_the_best_guess_line_as_two_subgroups(tmp_path):
    rows, pooled, added, o, s, p = _best_guess_late()
    strict_pooled = _strict_late()[1]
    result = forest_r.render_forest_r(rows, pooled, o, s, p, tmp_path / "forest_best_guess",
                                      line="best_guess", best_guess_ids=added,
                                      strict_pooled=strict_pooled)
    check = result.info.crosscheck
    assert check["ok"] and check["df_predict"] == pooled.k - 2
    # the prediction interval meta drew is the one canopy computed
    assert any(c["quantity"] == "pi_low" and c["ok"] for c in check["checks"])
    # …and the Confirmed subgroup's diamond is still the strict analysis
    assert not check["warnings"]


@pytest.mark.skipif(forest_r.r_available() is None, reason="Rscript/meta not installed")
@pytest.mark.slow
def test_the_dispatcher_only_keeps_r_when_every_format_was_written(tmp_path):
    """A report links png, svg and pdf, and all three have to be the SAME picture."""
    rows, pooled, o, s, p = _strict_late()
    result = forest_r.render_forest_r(rows, pooled, o, s, p, tmp_path / "probe")
    complete = set(forest_r.FORMATS) <= set(result.paths)
    paths, info = forest_render.render_forest(rows, pooled, o, s, p, tmp_path / "forest")
    if complete:
        assert info.renderer == "R meta::forest.meta"
    else:
        assert info.renderer == "canopy.report.forest (matplotlib)"
        assert "cannot write every format" in info.reason
        assert set(paths) == {"png", "svg", "pdf"}
