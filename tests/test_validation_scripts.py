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
    # a particle belongs to the surname: "Van De Plas" is `vandeplas`, not `van` — and `van`
    # would collide with every other Dutch first author in the corpus
    assert common.first_author("Van De Plas et al.") == "vandeplas"
    assert common.first_author("von der Heydt") == "vonderheydt"
    assert common.first_author("De Xivry, J.") == "dexivry"
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


def test_every_row_of_BOTH_gold_sheets_carries_the_numbers_the_join_and_the_pooling_need():
    """The regression that made this test exist.

    `aft_gsheet.csv` spells the column `N_yng`; `late_gsheet.csv` spells it `N_young`. Reading one
    spelling left all 40 aftereffect rows with `n_young = None`, which silently made `_n_distance`
    return None and every join rule that needs the group sizes unreachable — no error, no warning,
    just a column of blanks in the committed discrepancy table.
    """
    from canopy.models import StatsSettings

    for outcome in ("late_adaptation", "aftereffect"):
        rows = common.load_gold(outcome, GOLD)
        assert rows, outcome
        for row in rows:
            assert row.n_old is not None, f"{outcome} row {row.index}: n_old"
            assert row.n_young is not None, f"{outcome} row {row.index}: n_young"
            assert row.te is not None and row.se not in (None, 0.0), f"{outcome} row {row.index}"
            assert row.author_key, f"{outcome} row {row.index}: author"
        records = common.gold_records(rows, StatsSettings())
        assert all(r.es is not None and r.var not in (None, 0) for r in records), outcome
        assert all(r.n_a is not None and r.n_b is not None for r in records), outcome


def test_the_one_gold_row_with_no_year_is_known_about():
    """A gap in the DATA, not in the reader — pinned so it cannot grow unnoticed.

    `aft_gsheet.csv` row 35 (Pacheco et al.) has a blank Year. Such a row can never match on
    `author + year`, so it reaches the `author_year` fallback or stays unmatched. One row is a
    known gap; a test that fails when it becomes ten is worth more than one that hides it.
    """
    undated = {outcome: [r.author_key for r in common.load_gold(outcome, GOLD) if r.year is None]
               for outcome in ("late_adaptation", "aftereffect")}
    assert undated["late_adaptation"] == []
    assert undated["aftereffect"] == ["pacheco", "vandeplas"]


def test_the_gold_column_aliases_resolve_for_every_shipped_spreadsheet():
    import csv as _csv

    for name in common.GOLD_FOR_OUTCOME.values():
        path = GOLD / name
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = _csv.DictReader(handle).fieldnames or []
        column = common.resolve_columns(header, path)
        assert column["n_young"] in ("N_young", "N_yng"), name
        assert column["n_old"] == "N_old", name
    # late and aft really do disagree — that is the whole point of the alias table
    with (GOLD / "late_gsheet.csv").open(newline="", encoding="utf-8-sig") as handle:
        late = _csv.DictReader(handle).fieldnames or []
    with (GOLD / "aft_gsheet.csv").open(newline="", encoding="utf-8-sig") as handle:
        aft = _csv.DictReader(handle).fieldnames or []
    assert common.resolve_columns(late)["n_young"] != common.resolve_columns(aft)["n_young"]

    # `combined_gsheet.csv` is a WIDE summary (TE_Late / TE_Aft, no per-outcome TE), so it is not
    # a `load_gold` input at all and the reader says so rather than half-reading it
    import csv as _csv2

    with (GOLD / "combined_gsheet.csv").open(newline="", encoding="utf-8-sig") as handle:
        combined = _csv2.DictReader(handle).fieldnames or []
    with pytest.raises(common.GoldSchemaError):
        common.resolve_columns(combined, "combined_gsheet.csv")


def test_a_spreadsheet_missing_a_required_column_fails_loudly(tmp_path):
    # `load_gold` builds the filename from GOLD_FOR_OUTCOME, so the broken file has to use it
    path = tmp_path / common.GOLD_FOR_OUTCOME["late_adaptation"]
    path.write_text("Author,Year,TE,CI_low,CI_high,seTE\nBock,2005,-1.6,-2.6,-0.7,0.48\n")
    with pytest.raises(common.GoldSchemaError) as excinfo:
        common.load_gold("late_adaptation", tmp_path)
    assert "n_young" in str(excinfo.value) and "N_yng" in str(excinfo.value)


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


def test_the_empty_scatter_says_why_it_is_empty(tmp_path):
    """Deliverable (a) owes the reader a file that explains itself, not a missing one."""
    from canopy.protocol import load_protocol

    outcome = load_protocol(ROOT / "examples" / "protocols" /
                            "aging_sensorimotor_adaptation.yaml").outcome("late_adaptation")
    summary = {"n_pairs": 0, "n_auto_rows": 2, "n_auto_without_effect": 2, "n_gold_rows": 50,
               "gold_file": "late_gsheet.csv", "outcome_key": "late_adaptation",
               "join_rules": {"author_year": 2, "unmatched_gold": 48}}
    files = compare.empty_scatter(outcome, tmp_path / "manual_vs_auto_late_adaptation", summary)
    assert Path(files["png"]).exists() and Path(files["svg"]).exists()
    text = Path(files["svg"]).read_text()
    assert "0 matched pairs" in text and "2 produced NO effect size" in text


def test_a_ci_that_does_not_bracket_its_own_estimate_is_reported_not_hidden():
    assert compare._brackets(0.5, 0.1, 0.9) == "true"
    assert compare._brackets(0.5, 0.6, 0.9) == "false"      # drawn as a magnitude, flagged here
    assert compare._brackets(None, 0.1, 0.9) == ""
    assert compare._brackets(0.5, None, 0.9) == ""


def test_the_synthetic_corpus_contains_the_log_axis_case_the_docstring_promises():
    log_cases = [c for c in synthetic_figures.corpus() if c.log_y]
    assert log_cases, "the module docstring claims a log-axis case"
    assert "log" in synthetic_figures.__doc__


def test_both_offline_routes_read_a_log_axis_without_falling_back_to_linear(tmp_path):
    result = synthetic_figures.build(tmp_path, only=("points_log_axis",))
    for route in ("A_vector", "B_raster_cv"):
        entry = result["summary"][route]
        assert entry["n_read"] == 2, f"{route} could not read the log-axis case"
        # a linear fit through 1,3,10,30,100 is wrong by an order of magnitude, so this number
        # only stays small if the log scale was actually chosen
        assert entry["mae_pct_of_axis_range"] < 2.0, f"{route} read the log axis as linear"


def test_the_example_group_labels_come_from_the_protocol():
    """A label that says "older" when the protocol compares patients to controls is just wrong."""
    from canopy.protocol import load_protocol
    from validation.scripts import _example

    aging = load_protocol(ROOT / "examples" / "protocols" /
                          "aging_sensorimotor_adaptation.yaml")
    clinical = load_protocol(ROOT / "examples" / "protocols" /
                             "clinical_vs_control_adaptation.yaml")
    assert _example.group_labels(aging) == {"A": "Older adults (A)", "B": "Younger adults (B)"}
    assert _example.group_labels(clinical) == {"A": "Clinical group (A)",
                                               "B": "Control group (B)"}


def test_the_frozen_split_carries_no_absolute_path():
    payload = run_cisneros.load_split()
    for paper in payload["papers"]:
        for key, value in paper.items():
            assert not (isinstance(value, str) and value.startswith("/")), (key, value)
        assert paper["source"] in ("validation/papers_oa", "papers_folder")


def test_the_run_wrapper_never_overwrites_an_explicit_stats_field(tmp_path):
    """`--profile cisneros2024` is a floor, not a bulldozer: an explicit `stats:` field wins."""
    import yaml
    from canopy.models import StatsSettings
    from canopy.protocol import apply_profile

    source = yaml.safe_load(
        (ROOT / "examples" / "protocols" / "aging_sensorimotor_adaptation.yaml").read_text())
    source["stats"] = {"profile": "cisneros2024", "hakn": True, "tau2_method": "DL"}
    path = tmp_path / "explicit.yaml"
    path.write_text(yaml.safe_dump(source))

    raw = yaml.safe_load(path.read_text())
    explicit = {k: v for k, v in raw["stats"].items() if k != "profile"}
    settings = apply_profile(StatsSettings(profile="cisneros2024", **explicit))
    assert settings.hakn is True                     # the profile says False; the protocol wins
    assert settings.tau2_method == "DL"              # the profile says REML
    assert settings.estimator == "cohen"             # unset -> the profile fills it in


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


# ============================================================================ task 16: calibration
@pytest.fixture(scope="module")
def drawn_corpus(tmp_path_factory) -> dict:
    """Every case rendered once, and the digitiser's own CV core run over each PNG."""
    from canopy.digitize.digitizer import _cv_core

    out = tmp_path_factory.mktemp("corpus")
    drawn = {}
    for case in synthetic_figures.corpus():
        synthetic_figures.draw(case, out)
        core = _cv_core(Path(case.files["png"]), prefer_markers=case.kind != "bar")
        drawn[case.name] = (case, core)
    return drawn


def test_the_whole_corpus_calibrates_with_no_failures(drawn_corpus):
    """Acceptance item 16: not one case may end up without a y calibration, or with a wrong one.

    What is asserted is CORRECTNESS, not completeness: tesseract legitimately misses a label now
    and then, and a ladder of five true ticks out of seven calibrates the axis perfectly well.
    A ladder holding a value the figure never printed does not — that is F1 (45/35/25/15 read as
    4/3/2/1) and F8 (15/10/5/0 read as 5/0/5/0), and both are caught by the same line.
    """
    failed = [name for name, (_case, core) in drawn_corpus.items() if core.cal is None]
    assert failed == [], f"cases with no calibration: {failed}"
    for name, (case, core) in drawn_corpus.items():
        got = sorted(v for _, v in core.cal.ticks)
        truth = sorted(float(t) for t in case.ticks)
        assert set(got) <= set(truth), f"{name}: a tick value the figure never printed: {got}"
        assert len(got) == len(set(got)), f"{name}: the same tick value twice: {got}"
        assert len(got) >= max(2, 0.6 * len(truth)), f"{name}: only {got} of {truth}"
        # …and the mapping those ticks imply is the one the figure was drawn with
        implied = (got[-1] - got[0]) / (truth[-1] - truth[0])
        assert implied >= 0.5, f"{name}: the ladder covers only {implied:.0%} of the axis"


def test_the_negative_range_case_keeps_its_minus_sign_end_to_end(drawn_corpus):
    """Acceptance item 16, second half — a dropped minus is a sign error, not a rounding one."""
    from canopy.digitize.calibrate import px_to_value

    case, core = drawn_corpus["bars_negative_range"]
    ticks = sorted(v for _, v in core.cal.ticks)
    assert min(ticks) < 0, f"every recovered tick is positive on a -20..5 axis: {ticks}"
    assert set(ticks) <= set(float(t) for t in case.ticks)
    assert core.cal.a < 0, "on a y axis the value falls as the pixel row grows"
    px = dict((p, v) for p, v in core.cal.ticks)
    for pixel, value in px.items():
        assert px_to_value(core.cal, pixel) == pytest.approx(value, abs=0.5)


def test_a_log_axis_is_inferred_by_the_pipeline_not_handed_to_it(drawn_corpus):
    """Acceptance item 17: `_cv_core` used to hard-code linear; the scale is now read off the ticks.

    The corpus already had a log case, but both routes were told `scale="log"` before they fitted
    anything — so what was tested was `fit_axis`, never the inference.
    """
    from canopy.digitize.calibrate import px_to_value
    from canopy.digitize.digitizer import fit_best_scale

    case, core = drawn_corpus["points_log_axis"]
    assert case.log_y
    assert core.scale_note == "log" and core.cal.scale == "log", (
        f"the scale was not inferred: {core.scale_note}, ticks {core.cal.ticks}")
    # the linear fit through the SAME ticks is rejected, not merely not chosen
    again, note = fit_best_scale(core.cal.ticks, axis="y")
    assert note == "log" and again.scale == "log"
    top = max(core.cal.ticks, key=lambda t: t[1])
    assert px_to_value(core.cal, top[0]) == pytest.approx(top[1], rel=0.02)
    # every other case stays linear: the inference must not turn a straight axis into a curved one
    assert {name for name, (_c, cr) in drawn_corpus.items() if cr.cal.scale == "log"} == \
        {"points_log_axis"}


def test_wide_two_digit_labels_at_a_small_font_keep_both_digits(drawn_corpus):
    """Acceptance item 19: the band cut used to fall INSIDE a wide label (Cressman's 45 -> 4)."""
    case, core = drawn_corpus["points_wide_two_digit_small_font"]
    got = sorted(v for _, v in core.cal.ticks)
    assert set(got) <= set(float(t) for t in case.ticks), f"a value never printed: {got}"
    # a label cut to its first digit reads 1..9; the surviving ladder must reach the tens
    assert max(got) >= 60.0, f"the two-digit labels were cut to one digit: {got}"
    assert len(got) >= 6


def test_the_band_edge_retry_is_a_repair_not_the_normal_path(drawn_corpus):
    """The retry re-cuts the label band; it must not fire on every figure, and it is recorded."""
    retried = {name for name, (_c, core) in drawn_corpus.items() if core.band_retried}
    assert len(retried) < len(drawn_corpus) / 2, f"the retry fired on {sorted(retried)}"
    # and wherever it did fire, the ladder it produced is still made of real tick values
    for name in retried:
        case, core = drawn_corpus[name]
        assert set(v for _, v in core.cal.ticks) <= set(float(t) for t in case.ticks)


def test_a_line_plot_with_tiny_markers_still_reaches_the_marker_detector(drawn_corpus):
    """Acceptance item 20: `detect_bars` claimed a bar, so `detect_markers` never ran (miss 8)."""
    from canopy.digitize.digitizer import _cv_core

    case, core = drawn_corpus["line_tiny_markers"]
    assert core.markers, "the line's own markers were not detected"
    without = _cv_core(Path(case.files["png"]), prefer_markers=False)
    assert without.bars and not without.markers, "this case no longer demonstrates the failure"
