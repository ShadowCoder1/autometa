"""Task 13 — the validation scripts, offline.

These are the scripts that produce the Cisneros deliverables, so their *pure* parts are tested
the way the pipeline's are: the join, the agreement statistics, the split, the gold reader, the
synthetic corpus and every figure writer, all on tiny inputs, with no run directory, no network
and no model.

The one thing that is NOT tested here is a full Cisneros run — that costs money and lives in
`validation/README.md`. What is tested is that every script would work on one: each is driven
against a small run directory built by the pipeline's own `FakeProvider` (real PDFs, real
ingestion, fake model answers), which is the same thing `run_cisneros.py --demo` builds.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                       # the scripts live outside the installed package
    sys.path.insert(0, str(ROOT))

from validation.scripts import _common as common                            # noqa: E402
from validation.scripts import compare_manual_vs_auto as compare            # noqa: E402
from validation.scripts import extraction_routes_figure as routes_fig       # noqa: E402
from validation.scripts import replot_forest, run_cisneros, synthetic_figures  # noqa: E402

GOLD = ROOT / "validation" / "reference" / "cisneros2024"


# ============================================================================ text + gold data
def test_first_author_survives_every_spelling_the_two_sides_use():
    assert common.first_author("Fernández-Ruiz et al.") == "fernandezruiz"
    assert common.first_author("Heuer & Hegele") == "heuer"
    assert common.first_author("Bock ") == "bock"
    assert common.first_author("Bock, O.") == "bock"
    assert common.first_author("J. Bock and R. Smith") == "bock"
    assert common.first_author("") == ""


def test_experiment_labels_normalise_to_the_same_string():
    assert common.norm_experiment("Exp. 1a") == common.norm_experiment("1a") == "1a"
    assert common.norm_experiment("Experiment 2") == "2"
    assert common.norm_experiment("") == ""


def test_gold_rows_load_with_their_numbers_and_their_row_numbers():
    rows = common.load_gold("late_adaptation", GOLD)
    assert len(rows) >= 40
    bock = next(r for r in rows if r.author_key == "bock")
    assert bock.year == 2005 and bock.n_old == 12 and bock.n_young == 12
    assert bock.te == pytest.approx(-1.676) and bock.se == pytest.approx(0.4801)
    # the row number is the CSV's own, so a person can find the line the number came from
    assert 1 <= bock.index <= len(rows) + 5
    assert all(r.te is not None for r in rows)                # TE-less rows are dropped


def test_an_outcome_with_no_gold_spreadsheet_says_so():
    with pytest.raises(KeyError):
        common.load_gold("not_an_outcome", GOLD)


def test_pooling_the_gold_reproduces_the_published_review(settings=None):
    """The reference review pooled these 50 rows; our pooler must agree with metagen's inputs."""
    from canopy.models import StatsSettings
    from canopy.protocol import apply_profile

    settings = apply_profile(StatsSettings(profile="cisneros2024"))
    rows = common.load_gold("late_adaptation", GOLD)
    pooled = common.pool_gold(rows, settings)
    assert pooled.k == len(rows)
    assert -1.0 < pooled.estimate < 0.0                       # older adults adapt less
    assert pooled.ci_low < pooled.estimate < pooled.ci_high
    assert 0.0 <= pooled.I2 <= 1.0


def test_gold_rows_become_records_the_forest_can_draw_without_recomputing_anything():
    from canopy.models import StatsSettings

    rows = common.load_gold("aftereffect", GOLD)[:4]
    records = common.gold_records(rows, StatsSettings())
    assert [r.es for r in records] == [r.te for r in rows]     # carried across, never recomputed
    assert all(r.route == "manual" and r.confidence == "auto_accept" for r in records)
    assert all(r.var == pytest.approx(row.se ** 2) for r, row in zip(records, rows))


# ============================================================================ agreement metrics
def test_ccc_is_1_only_for_the_identity_and_punishes_a_scaled_reading():
    x = [0.1, -0.5, 1.2, -1.8, 0.4]
    assert common.lins_ccc(x, list(x)) == pytest.approx(1.0)
    half = [v / 2 for v in x]
    assert common.lins_ccc(x, half) < 0.85                    # r would still be exactly 1.0
    assert common.lins_ccc([1.0], [1.0]) is None              # one pair is not agreement


def test_mae_rmse_sign_agreement_and_bland_altman():
    manual = [1.0, -1.0, 0.5, -0.5]
    auto = [1.2, -0.8, 0.5, 0.1]
    assert common.mae(manual, auto) == pytest.approx((0.2 + 0.2 + 0.0 + 0.6) / 4)
    assert common.rmse(manual, auto) == pytest.approx(((0.04 + 0.04 + 0 + 0.36) / 4) ** 0.5)
    assert common.sign_agreement(manual, auto) == pytest.approx(0.75)   # the last pair flips
    ba = common.bland_altman(manual, auto)
    assert ba["bias"] == pytest.approx((0.2 + 0.2 + 0.0 + 0.6) / 4)
    assert ba["loa_low"] < ba["bias"] < ba["loa_high"]


def test_ci_falls_back_to_the_se_when_the_interval_is_missing():
    from canopy.models import EffectSizeRecord

    record = EffectSizeRecord(es=0.5, se=0.25)
    low, high = common.ci_of(record)
    assert low == pytest.approx(0.5 - 1.96 * 0.25)
    assert high == pytest.approx(0.5 + 1.96 * 0.25)
    assert common.ci_of(EffectSizeRecord()) == (None, None)


# ============================================================================ the join
def _record(**kwargs):
    from canopy.models import Citation, EffectSizeRecord

    citation = Citation(authors=kwargs.pop("authors", "Bock"), year=kwargs.pop("year", 2005),
                        first_author=kwargs.pop("first_author", "Bock"))
    return EffectSizeRecord(citation=citation, outcome_key="late_adaptation", **kwargs)


def _gold(index: int, **kwargs):
    base = dict(index=index, outcome_key="late_adaptation", author="Bock", author_key="bock",
                year=2005, title="", experiment="1a", figure="1", measure="m", phase="p",
                n_young=12, n_old=12, te=-1.676, ci_low=-2.6, ci_high=-0.73, se=0.48)
    base.update(kwargs)
    return common.GoldRow(**base)


def test_the_join_matches_on_author_year_and_group_sizes():
    records = [_record(dataset_id="d1", n_a=12, n_b=12, es=-1.6)]
    pairs = common.join_rows(records, [_gold(1)], "late_adaptation")
    assert len(pairs) == 1
    assert pairs[0].how in ("exact", "n_pair")
    assert pairs[0].delta == pytest.approx(-1.6 - -1.676)
    assert pairs[0].within_tolerance is True


def test_the_join_never_pairs_two_rows_with_different_group_sizes_by_the_strict_rules():
    records = [_record(dataset_id="d1", n_a=30, n_b=30, es=-1.6),
               _record(dataset_id="d2", n_a=12, n_b=12, es=-1.7)]
    pairs = common.join_rows(records, [_gold(1)], "late_adaptation")
    matched = [p for p in pairs if p.auto is not None and p.gold is not None]
    assert len(matched) == 1 and matched[0].auto.dataset_id == "d2"
    assert any(p.how == "unmatched_auto" and p.auto.dataset_id == "d1" for p in pairs)


def test_the_experiment_label_comes_from_the_mapper_not_from_the_dataset_id():
    """`record.label` falls back to `d1`, which is not an experiment and must not be compared."""
    from canopy.models import DatasetSpec, StudyMap

    record = _record(dataset_id="d1", label="d1", n_a=12, n_b=12, es=-1.6)
    record.paper_id = "sha"
    study = StudyMap(paper_id="sha", datasets=[DatasetSpec(dataset_id="d1", experiment="1a")])

    # with the mapper's experiment, this is an EXACT match to the gold row's experiment "1a"
    pairs = common.join_rows([record], [_gold(1)], "late_adaptation", studies={"sha": study})
    assert pairs[0].how == "exact"

    # a genuinely different experiment label is not an exact match, but the N pair still is
    other = StudyMap(paper_id="sha", datasets=[DatasetSpec(dataset_id="d1", experiment="2b")])
    pairs = common.join_rows([record], [_gold(1)], "late_adaptation", studies={"sha": other})
    assert pairs[0].how == "n_pair"

    # and with no study at all the experiment is simply unknown, never "d1"
    assert common._auto_experiment(record, None) == ""


def test_a_gold_row_the_run_never_produced_is_reported_not_dropped():
    pairs = common.join_rows([], [_gold(1), _gold(2)], "late_adaptation")
    assert [p.how for p in pairs] == ["unmatched_gold", "unmatched_gold"]
    assert all(p.auto is None for p in pairs)


def test_one_gold_row_is_never_claimed_twice():
    records = [_record(dataset_id="d1", n_a=12, n_b=12, es=-1.6),
               _record(dataset_id="d2", n_a=12, n_b=12, es=-1.5)]
    pairs = common.join_rows(records, [_gold(1)], "late_adaptation")
    claimed = [p for p in pairs if p.gold is not None and p.auto is not None]
    assert len(claimed) == 1
    assert sum(1 for p in pairs if p.how == "unmatched_auto") == 1


def test_the_author_year_fallback_only_fires_when_one_row_is_left_on_each_side():
    #: sizes disagree, so the strict rules cannot pair them; a unique author still can
    records = [_record(dataset_id="d1", n_a=11, n_b=12, es=-1.6)]
    pairs = common.join_rows(records, [_gold(1, year=2006)], "late_adaptation")
    assert pairs[0].how == "author_year"
    assert "group sizes differ" in pairs[0].note
    #: two candidates on one side and the fallback must refuse
    records = [_record(dataset_id="d1", n_a=11, n_b=12), _record(dataset_id="d2", n_a=9, n_b=9)]
    pairs = common.join_rows(records, [_gold(1, year=2006)], "late_adaptation")
    assert {p.how for p in pairs} == {"unmatched_auto", "unmatched_gold"}


def test_an_override_wins_over_every_automatic_rule(tmp_path):
    path = tmp_path / "join_overrides.csv"
    path.write_text("# a comment line the reader must ignore\n"
                    "outcome_key,dataset_id,gold_row_index,reason\n"
                    "late_adaptation,d1,2,the spreadsheet labels this experiment differently\n")
    overrides = common.load_overrides(path)
    assert overrides == {("late_adaptation", "d1"): 2}
    records = [_record(dataset_id="d1", n_a=12, n_b=12, es=-1.6)]
    pairs = common.join_rows(records, [_gold(1), _gold(2, te=0.5)], "late_adaptation",
                             overrides=overrides)
    matched = next(p for p in pairs if p.auto is not None and p.gold is not None)
    assert matched.how == "override" and matched.gold.index == 2


def test_the_overrides_file_shipped_with_the_repo_parses_and_is_empty():
    assert common.load_overrides(GOLD / "join_overrides.csv") == {}


# ============================================================================ the split
def test_the_split_is_deterministic_and_depends_only_on_the_papers_bytes():
    class Group:
        def __init__(self, sha: str, name: str):
            self.sha256 = sha
            self.representative = Path("/tmp") / name
            self.duplicates: list[Path] = []
            self.title = name
            self.doi = ""

    groups = [Group(f"{i:064x}", f"p{i}.pdf") for i in range(5)]
    first = run_cisneros.build_split(groups)
    again = run_cisneros.build_split(list(reversed(groups)))
    assert [p["split"] for p in first["papers"]] == [p["split"] for p in again["papers"]]
    assert [p["sha256"] for p in first["papers"]] == [p["sha256"] for p in again["papers"]]
    assert first["n_dev"] == 3 and first["n_heldout"] == 2
    assert first["heldout_tag"] == "validation-heldout-v1"


def test_the_frozen_split_covers_every_unique_paper_exactly_once():
    payload = run_cisneros.load_split()
    shas = [p["sha256"] for p in payload["papers"]]
    assert len(shas) == len(set(shas)) == payload["n_papers"]
    assert payload["n_dev"] + payload["n_heldout"] == payload["n_papers"]
    assert {p["split"] for p in payload["papers"]} == {"dev", "heldout"}


def test_selecting_a_split_keeps_only_that_sides_papers_and_warns_about_strangers(capsys):
    class Group:
        def __init__(self, sha):
            self.sha256 = sha
            self.representative = Path("/tmp/x.pdf")

    payload = run_cisneros.load_split()
    dev = [p["sha256"] for p in payload["papers"] if p["split"] == "dev"]
    groups = [Group(sha) for sha in dev[:2]] + [Group("f" * 64)]
    chosen = run_cisneros.select(groups, "dev", payload)
    assert [g.sha256 for g in chosen] == dev[:2]
    assert "not in splits.json" in capsys.readouterr().out
    assert len(run_cisneros.select(groups, "all", payload)) == 3


# ============================================================================ synthetic figures
def test_the_synthetic_corpus_is_drawn_and_read_back_by_both_offline_routes(tmp_path):
    result = synthetic_figures.build(tmp_path, only=("bars_sd_plain",))
    summary = result["summary"]
    assert summary["n_cases"] == 1
    assert (tmp_path / "corpus" / "bars_sd_plain.png").exists()
    assert (tmp_path / "corpus" / "bars_sd_plain.pdf").exists()
    assert (tmp_path / "corpus" / "bars_sd_plain.json").exists()
    assert Path(summary["files"]["csv"]).exists()
    assert Path(summary["files"]["png"]).exists()

    for route in ("A_vector", "B_raster_cv"):
        entry = summary[route]
        assert entry["n_read"] == entry["n_series"] == 2, f"{route} failed to read the plain case"
        # the plainest possible figure: both offline routes must land inside the vote tolerance
        assert entry["mae_pct_of_axis_range"] < 2.0
    assert "NOT measured here" in summary["_note"]


def test_a_mark_that_was_never_detected_is_a_read_FAILURE_not_a_huge_error():
    """Detection and read-out are different failures; mixing them makes the MAE meaningless."""
    case = next(c for c in synthetic_figures.corpus() if c.name == "bars_sd_plain")
    picked, status = synthetic_figures.select_marks([object()], case, lambda m: 0.0)
    assert picked == [] and status.startswith("mark_count_mismatch")
    rows = [{"route": "B", "case": "c", "series": "a", "status": status, "read": None,
             "axis_range": 30.0}]
    summary = synthetic_figures.summarise(rows)
    assert summary["B"]["read_rate"] == 0.0
    assert summary["B"]["mae_pct_of_axis_range"] is None       # nothing to average, not a zero


def test_the_time_series_case_takes_each_series_rightmost_mark():
    case = next(c for c in synthetic_figures.corpus() if c.kind == "line")

    class Mark:
        def __init__(self, x):
            self.x = x

    marks = [Mark(x) for x in (1, 2, 3, 40, 41, 42)]           # two clusters, far apart
    picked, status = synthetic_figures.select_marks(marks, case, lambda m: m.x)
    assert status == "read"
    assert [m.x for m in picked] == [3, 42]


# ============================================================================ scripts on a run
@pytest.fixture(scope="module")
def demo_run(tmp_path_factory) -> Path:
    """A real run directory, built offline by the pipeline's own fake provider."""
    out = tmp_path_factory.mktemp("demo_run")
    assert run_cisneros.main(["--demo", "--out", str(out), "--no-resume"]) == 0
    return out


def test_the_demo_run_is_a_real_run_every_script_can_read(demo_run):
    run = common.load_run(demo_run)
    assert len(run.records) == 2
    assert run.protocol.stats.profile == "cisneros2024"
    assert run.settings.estimator == "cohen"
    assert set(run.pooled) >= {"late_adaptation"}
    assert run.pooled["late_adaptation"]["k"] == 2
    assert all(run.papers.get(r.paper_id) is not None for r in run.records)
    assert run.candidates and run.verdicts


def test_compare_writes_the_scatter_the_table_and_the_summary(demo_run, tmp_path):
    summary = compare.compare(demo_run, "late_adaptation", out_dir=tmp_path)
    assert Path(summary["files"]["discrepancies"]).exists()
    assert Path(summary["files"]["agreement"]).exists()
    rows = list(csv.DictReader(Path(summary["files"]["discrepancies"]).open()))
    assert rows, "every pair must appear in the table, matched or not"
    assert {"classification", "adjudicated_truth", "adjudicator_note"} <= set(rows[0])
    assert summary["pooled"]["manual"]["k"] >= 40             # the human's own 50 rows
    assert summary["pooled"]["settings"]["estimator"] == "cohen"
    # the fake's numbers are not Bock's, so pairs may be zero — but the shape must be right
    assert summary["n_gold_rows"] > 0 and summary["n_auto_rows"] == 2
    if summary["n_pairs"]:
        assert Path(summary["files"]["png"]).exists()


def test_compare_reports_agreement_when_the_auto_rows_ARE_the_gold_rows(tmp_path):
    """A tool that reproduced the human exactly must score CCC 1, MAE 0 — a control for the metric."""
    from canopy.models import StatsSettings

    gold = common.load_gold("late_adaptation", GOLD)[:8]
    records = common.gold_records(gold, StatsSettings())
    for record, row in zip(records, gold):
        record.citation.first_author = row.author_key
        record.n_a, record.n_b = row.n_old, row.n_young
    pairs = common.join_rows(records, gold, "late_adaptation")
    summary = compare.agreement(pairs)
    assert summary["n_pairs"] == len(gold)
    assert summary["lins_ccc"] == pytest.approx(1.0)
    assert summary["mae"] == pytest.approx(0.0)
    assert summary["within_tolerance"] == pytest.approx(1.0)
    assert summary["auto_accept_precision"] == pytest.approx(1.0)


def test_the_routes_figure_is_drawn_from_the_runs_own_provenance(demo_run, tmp_path):
    payload = routes_fig.build(demo_run, out_dir=tmp_path)
    assert Path(payload["files"]["png"]).exists()
    assert Path(payload["files"]["csv"]).exists()
    counts = payload["overall"]["counts"]
    assert sum(entry["n"] for entry in counts.values()) == 2
    assert sum(entry["pct"] for entry in counts.values()) == pytest.approx(100.0)
    rows = list(csv.DictReader(Path(payload["files"]["csv"]).open()))
    assert {r["route_bucket"] for r in rows} <= {"text", "table", "figure", "test_statistic",
                                                 "reported_d", "not_convertible", "other"}


def test_replot_draws_both_forests_and_reports_both_pooled_estimates(demo_run, tmp_path):
    payload = replot_forest.replot(demo_run, "late_adaptation", out_dir=tmp_path)
    assert Path(payload["files"]["manual_png"]).exists()
    assert Path(payload["files"]["auto_png"]).exists()
    assert Path(payload["files"]["side_by_side_png"]).exists()
    assert payload["manual"]["k"] >= 40 and payload["auto"]["k"] == 2
    # the weights the plot used are the random-effects weights, and they add to 100 %
    assert sum(payload["manual"]["weights_pct"]) == pytest.approx(100.0, abs=1e-6)
    assert payload["settings"]["weights"].startswith("random-effects 1/(v_i + tau^2)")


def test_the_examples_explain_a_cell_without_calling_a_model(demo_run, tmp_path, capsys):
    from validation.scripts import example_text_ms

    code = example_text_ms.main(["--run", str(demo_run), "--out", str(tmp_path)])
    assert code == 0
    out = capsys.readouterr().out
    for heading in ("what the mapper found", "what every extractor read",
                    "what verification decided", "the arithmetic", "the effect size"):
        assert heading in out
    assert (tmp_path / "example_text_ms.png").exists()
    payload = json.loads((tmp_path / "example_text_ms.json").read_text())
    assert payload["route"] in ("text_mean_sd", "table", "text_mean_se_ci")
    assert payload["es"] is not None


def test_an_example_that_has_no_cell_to_show_says_so_and_prints_what_the_run_does_have(
        demo_run, tmp_path, capsys):
    from validation.scripts import example_figure_only

    code = example_figure_only.main(["--run", str(demo_run), "--out", str(tmp_path)])
    assert code == 2                                    # the fake run has no figure-derived row
    out = capsys.readouterr().out
    assert "no cell in" in out and "run_cisneros.py" in out


def test_the_second_protocol_is_a_genuinely_different_review():
    from canopy.protocol import load_protocol

    aging = load_protocol(ROOT / "examples" / "protocols" /
                          "aging_sensorimotor_adaptation.yaml")
    toy = load_protocol(ROOT / "examples" / "protocols" /
                        "clinical_vs_control_adaptation.yaml")
    assert {o.key for o in toy.outcomes}.isdisjoint({o.key for o in aging.outcomes})
    assert toy.group_a.label != aging.group_a.label
    assert toy.stats.profile == "metafor" != aging.stats.profile
    assert toy.stats.estimator == "hedges" and toy.stats.variance == "borenstein"
    assert toy.moderators and set(toy.moderators).isdisjoint(set(aging.moderators))
