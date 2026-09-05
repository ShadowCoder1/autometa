"""Task 1: protocol/profile schemas and study/candidate models."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from canopy.models import (
    Candidate,
    Citation,
    DatasetSpec,
    DispersionType,
    EffectSizeRecord,
    GroupSpec,
    OutcomeSources,
    Protocol,
    RunManifest,
    Source,
    SourceKind,
    StatsSettings,
    StudyMap,
    Verdict,
)
from canopy.protocol import apply_profile, available_profiles, load_protocol
from tests.helpers import nine

REPO = Path(__file__).resolve().parents[1]
EXAMPLE = REPO / "examples" / "protocols" / "aging_sensorimotor_adaptation.yaml"


# --------------------------------------------------------------- example protocol
def test_example_protocol_loads():
    p = load_protocol(EXAMPLE)
    assert isinstance(p, Protocol)
    assert [o.key for o in p.outcomes] == ["late_adaptation", "aftereffect"]
    assert len(p.outcomes) == 2
    assert "Older" in p.group_a.label
    assert p.group_a.key == "A" and p.group_b.key == "B"
    assert "Younger" in p.group_b.label


def test_example_protocol_methods_content():
    p = load_protocol(EXAMPLE)
    assert len(p.eligibility) == 4
    assert len(p.dataset_rules) == 3
    assert set(p.moderators) == {"task_type", "perturbation_size_deg", "n_targets"}
    late = p.outcome("late_adaptation")
    assert "Enhanced in Old" == late.positive_direction_label
    assert "Reduced in Old" == late.negative_direction_label
    assert "last measure" in late.definition
    aft = p.outcome("aftereffect")
    assert "implicit recalibration" in aft.definition
    with pytest.raises(KeyError):
        p.outcome("nope")


def test_a_repeated_key_in_a_protocol_is_an_error_not_a_silent_overwrite(tmp_path):
    """The failure this prevents: `runs/proof` lost `digitize.collapse_across_categorical_x`.

    The protocol had two `digitize:` blocks — the flag in the first, the read-out settings in
    the second — and PyYAML kept only the last, so the setting was never applied and the cell it
    governed produced no value. Nothing in the run said a word about it.
    """
    path = tmp_path / "protocol.yaml"
    body = EXAMPLE.read_text() + "\ndigitize:\n  collapse_across_categorical_x: true\n"
    path.write_text(body + "digitize:\n  readouts_min: 2\n", encoding="utf-8")
    with pytest.raises(ValueError) as exc:
        load_protocol(path)
    assert "duplicate key 'digitize'" in str(exc.value)
    assert "dropped in silence" in str(exc.value)


def test_a_protocol_without_repeated_keys_still_loads(tmp_path):
    path = tmp_path / "protocol.yaml"
    path.write_text(EXAMPLE.read_text(), encoding="utf-8")
    assert isinstance(load_protocol(path), Protocol)


def test_example_protocol_uses_cisneros_profile():
    p = load_protocol(EXAMPLE)
    assert p.stats.profile == "cisneros2024"
    # load_protocol resolves the profile
    assert p.stats.variance == "hedges_olkin_df"
    assert p.stats.pi_method == "HTS"


def test_example_protocol_is_study_agnostic():
    """Nothing study-specific: the protocol may not name papers/authors/results."""
    raw = EXAMPLE.read_text().lower()
    for banned in ("bock", "heuer", "hegele", "et al.", "table 1", "figure 2", "n = 12"):
        assert banned not in raw


# --------------------------------------------------------------- profiles
def test_profiles_available():
    names = available_profiles()
    assert "metafor" in names and "cisneros2024" in names


def test_apply_profile_cisneros():
    s = apply_profile(StatsSettings(profile="cisneros2024"))
    assert s.estimator == "cohen"
    assert s.variance == "hedges_olkin_df"
    assert s.pi_method == "HTS"
    assert s.tau2_method == "REML"
    assert s.hakn is False
    assert s.profile == "cisneros2024"


def test_apply_profile_metafor():
    s = apply_profile(StatsSettings(profile="metafor"))
    assert s.estimator == "hedges"
    assert s.variance == "borenstein"
    assert s.tau2_method == "REML"
    assert s.hakn is False
    assert s.pi_method == "z"


def test_apply_profile_is_idempotent():
    once = apply_profile(StatsSettings(profile="cisneros2024"))
    twice = apply_profile(once)
    assert once.model_dump() == twice.model_dump()


def test_explicit_settings_win_over_profile():
    s = apply_profile(StatsSettings(profile="cisneros2024", pi_method="z"))
    assert s.pi_method == "z"           # user value survives
    assert s.variance == "hedges_olkin_df"  # profile still fills the rest


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        apply_profile(StatsSettings(profile="does_not_exist"))


# --------------------------------------------------------------- stats settings (amendment A)
def test_stats_settings_defaults_follow_amendment():
    s = StatsSettings()
    assert s.route_precedence == ["text_mean_sd", "table", "text_mean_se_ci", "figure",
                                  "test_statistic", "p_value", "reported_d"]
    assert s.late_window_sd == "paper_reported_block"
    assert s.ci_to_sd_dist == "auto"
    assert s.primary_analysis_includes == ["auto_accept", "accept_with_note"]
    assert s.shared_control_strategy == "split_n"
    assert s.multi_group_policy == "closest_to_definition"
    assert s.digitization_variance == "sensitivity"
    assert s.one_row_per_paper is True
    # the pre-amendment duplicates are gone: late_window_sd / digitization_variance are the
    # authoritative fields
    assert "late_window_rule" not in StatsSettings.model_fields
    assert "add_digitization_variance" not in StatsSettings.model_fields
    with pytest.raises(Exception):
        StatsSettings(late_window_rule="paper_reported_block_else_last_point")


def test_stats_settings_rejects_bad_enum():
    with pytest.raises(Exception):
        StatsSettings(variance="nope")


# --------------------------------------------------------------- hashing
def test_protocol_hash_is_stable_and_content_sensitive():
    a = load_protocol(EXAMPLE)
    b = load_protocol(EXAMPLE)
    assert a.hash() == b.hash()
    assert len(a.hash()) == 64
    c = a.model_copy(deep=True)
    c.title = a.title + " (v2)"
    assert c.hash() != a.hash()
    # hash survives a JSON round-trip
    assert Protocol.model_validate_json(a.model_dump_json()).hash() == a.hash()


# --------------------------------------------------------------- candidates etc.
def test_candidate_round_trips_through_json():
    c = Candidate(
        candidate_id="cand-1",
        dataset_id="ds1",
        outcome_key="late_adaptation",
        kind="group_stats",
        group="A",
        source_kind=SourceKind.figure_line,
        status="found",
        n=12,
        n_quote="twelve elderly subjects",
        mean=8.5,
        dispersion_value=3.1,
        dispersion_type=DispersionType.SD,
        unit="deg",
        page=3,
        quote="the older group showed 8.5 +/- 3.1 deg",
        locator="Fig 2A",
        crop_path="figures/fig02_claude.png",
        pixel_provenance={"x_px": 431.0, "y_px": 208.5, "image": "fig02_claude.png"},
        sigma=0.4,
        model="claude-opus-5",
        prompt_version="digitize_vlm@1",
        llm_call_id="abc123",
        extractor_id="digitizer_C",
    )
    blob = c.model_dump_json()
    json.loads(blob)  # valid JSON
    back = Candidate.model_validate_json(blob)
    assert back == c
    assert back.dispersion_type is DispersionType.SD
    assert back.pixel_provenance["x_px"] == 431.0


def test_candidate_amendment_fields():
    c = Candidate(kind="group_stats", group="B")
    # defaults exist and are the "unknown"-ish members
    assert c.status == "found"
    assert c.raw_value_semantics == "unknown"
    assert c.analysis_metric == "unknown"
    assert c.error_bar_scope == "unknown"
    assert c.whisker_definition == "unknown"
    assert c.page_corrected is False
    assert c.mean is None and c.n is None          # numerics nullable
    c2 = Candidate(kind="group_stats", group="B", status="not_on_these_pages",
                   raw_value_semantics="higher_more_error", analysis_metric="change_from_baseline",
                   error_bar_scope="within_subject_normalized", whisker_definition="iqr_1_5",
                   page_corrected=True)
    assert Candidate.model_validate_json(c2.model_dump_json()) == c2


def test_test_statistic_candidate():
    c = Candidate(kind="test_statistic", group=None, stat_type="t", stat_value=2.31, df=22,
                  direction="a_greater", page=4, quote="t(22) = 2.31")
    assert Candidate.model_validate_json(c.model_dump_json()) == c
    d = Candidate(kind="test_statistic", stat_type="F", stat_value=7.58, design="mixed_main_effect",
                  df1=1, df2=22, tails=2, p_kind="less_than", admissible=False,
                  admissible_reason="mixed design interaction")
    assert Candidate.model_validate_json(d.model_dump_json()) == d


def test_reported_effect_size_candidate():
    c = Candidate(kind="reported_d", reported_value=-1.68, reported_scale="cohens_d",
                  standardizer="pooled_sd_between", reported_ci_low=-2.6, reported_ci_high=-0.7,
                  positive_means="a_greater", page=4, quote="d = -1.68")
    assert Candidate.model_validate_json(c.model_dump_json()) == c


def test_dataset_spec_amendment_fields():
    ds = DatasetSpec(dataset_id="ds1", cluster_id="0123456789ab", shared_control=True,
                     exposure_order="first",
                     all_groups_listed=[GroupSpec(label="old", n=12), GroupSpec(label="young", n=12),
                                        GroupSpec(label="middle", n=10)],
                     chosen_pair_rationale="closest to protocol age definitions")
    assert len(ds.all_groups_listed) == 3
    assert ds.exposure_order == "first"
    assert DatasetSpec.model_validate_json(ds.model_dump_json()) == ds
    assert DatasetSpec(dataset_id="x").exposure_order == "unknown"      # default


def test_studymap_round_trip():
    sm = StudyMap(
        paper_id="sha256abc",
        citation=Citation(authors="Bock", year=2005, title="Components", journal="EBR"),
        eligible=True,
        eligibility_rationale="visuomotor rotation, older vs younger, English",
        datasets=[
            DatasetSpec(
                dataset_id="ds1",
                label="Experiment 1",
                experiment="1",
                group_a=GroupSpec(label="older", n=12, n_evidence="twelve older adults"),
                group_b=GroupSpec(label="younger", n=12, n_evidence="twelve younger adults"),
                moderators={"task_type": "visuomotor", "perturbation_size_deg": "60"},
                outcomes=[
                    OutcomeSources(
                        outcome_key="late_adaptation",
                        measure_name="directional error",
                        units="deg",
                        higher_is_better=False,
                        sources=[Source(kind=SourceKind.figure_line, page=3, locator="Fig 2A",
                                        error_bar_type=DispersionType.SD)],
                    )
                ],
            )
        ],
    )
    back = StudyMap.model_validate_json(sm.model_dump_json())
    assert back == sm
    assert back.datasets[0].outcomes[0].sources[0].kind is SourceKind.figure_line
    assert StudyMap(paper_id="x").citation.year is None      # citation optional


def test_source_kinds_and_dispersion_members():
    assert {k.value for k in SourceKind} == {
        "text_mean_sd", "text_mean_se", "text_mean_ci", "table", "figure_bar", "figure_line",
        "figure_points", "figure_box", "test_statistic", "reported_effect_size", "author_data",
        "unknown"}
    assert {d.value for d in DispersionType} == {
        "SD", "SE", "CI95", "CI90", "IQR", "RANGE", "NONE", "UNKNOWN"}


def test_verdict_effect_size_and_manifest_round_trip():
    v = Verdict(dataset_id="ds1", outcome_key="aftereffect", group="A",
                agreement="agree", verifier_verdict="confirmed", confidence="auto_accept")
    assert Verdict.model_validate_json(v.model_dump_json()) == v

    e = EffectSizeRecord(dataset_id="ds1", outcome_key="aftereffect", route="means_sd",
                         n_a=12, n_b=12, d=0.5, g=0.48, es=0.48, var=0.17, se=0.412,
                         ci_low=-0.33, ci_high=1.29, estimator="hedges",
                         variance_method="hedges_olkin_df", higher_is_better=False,
                         conversion_chain="means+SD -> d -> g")
    assert EffectSizeRecord.model_validate_json(e.model_dump_json()) == e

    m = RunManifest(run_id="run1", created_at="2026-08-15T00:00:00Z", protocol_hash="deadbeef")
    assert RunManifest.model_validate_json(m.model_dump_json()) == m


def test_models_reject_unknown_fields():
    with pytest.raises(Exception):
        Candidate(kind="group_stats", group="A", bogus_field=1)


def test_everything_is_json_serialisable():
    """pydantic models must dump to plain JSON (enums as values, no python objects)."""
    p = load_protocol(EXAMPLE)
    assert isinstance(json.loads(p.model_dump_json()), dict)
    d = p.model_dump(mode="json")
    assert d["stats"]["profile"] == "cisneros2024"
    c = Candidate(kind="group_stats", group="A", source_kind=SourceKind.table,
                  dispersion_type=DispersionType.SE)
    assert json.loads(c.model_dump_json())["dispersion_type"] == "SE"


# --------------------------------------------------------------- config
def test_config_models_map():
    from canopy.config import MODELS

    assert MODELS["primary"] == "claude-opus-5"
    assert MODELS["secondary"] == "claude-sonnet-5"
    assert MODELS["adjudicator"] == "claude-opus-5"
    assert MODELS["adjudicator_max"] == "claude-fable-5"


def test_load_env_does_not_leak_key(capsys):
    from canopy.config import load_env

    load_env()                      # must not print anything
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------- C8: the two verdict literals
def test_every_verifier_state_round_trips_through_the_cell_verdict():
    """H3: `Verdict.verifier_verdict` must hold everything `VerifierVerdict.verdict` can say.

    `_verifier_summary` copies the verifier's state onto the cell verdict, and `CanopyModel` does
    not validate on assignment — so a member the cell's literal lacks is written happily, dumped
    happily, and then raises on the FIRST `--resume`, after the whole run has been paid for. C8's
    `no_value_printed` was exactly that. The two literals are compared here rather than listed, so
    a member added to one and not the other fails at the model instead of on a reviewer's resume.
    """
    import typing

    from canopy.models import VerifierVerdict

    states = typing.get_args(VerifierVerdict.model_fields["verdict"].annotation)
    holds = typing.get_args(Verdict.model_fields["verifier_verdict"].annotation)
    assert set(states) <= set(holds), (
        f"a verifier can report {sorted(set(states) - set(holds))}, which no Verdict can hold")
    for state in states:
        verdict = Verdict(verifier_verdict=state)
        assert Verdict.model_validate(verdict.model_dump(mode="json")).verifier_verdict == state


# --------------------------------------------------------------- the runs/nine real records
def test_nine_fixture_loads_twenty_records():
    """`tests/fixtures/runs/nine` is a real run, not a hand-written one.

    Every later test that needs a record, a verdict, a candidate or a study map reads it through
    `tests.helpers.nine`, so a model change that a hand-made fixture would have shrugged off
    fails here against numbers a run actually produced.
    """
    assert len(nine.records()) == 20 and nine.record("3570e4ce2a9c:d1", "late_adaptation").route == "not_convertible"


def test_record_and_settings_fields_default():
    from canopy.models import EffectSizeRecord, StatsSettings

    r = EffectSizeRecord()
    assert (r.best_guess_rule, r.best_guess_reason, r.route_overridden_from, r.precedence_override_reason, r.orientation_source) == ("",) * 5
    assert StatsSettings().forest_xlim is None and StatsSettings().forest_digits == 1


def test_cisneros_profile_pins_the_humans_axis():
    from canopy.models import StatsSettings; from canopy.protocol import apply_profile
    assert apply_profile(StatsSettings(profile="cisneros2024")).forest_xlim == [-4.0, 4.0]


def test_protocol_hash_change_is_deliberate():   # Major 9: the consequence is pinned, not discovered
    """New settings fields change every protocol's hash — including the one `runs/nine` recorded.

    That is a `--resume` of an older run refusing to reuse its stages, which is the correct
    behaviour and not a thing to discover in the middle of a re-run. It is asserted here so the
    cost is a decision someone made rather than a surprise someone hit.
    """
    from canopy.models import Protocol
    assert nine.protocol().hash() != json.load(open(nine.NINE / "manifest.json"))["protocol_hash"]


def test_stats_settings_dependency_defaults_and_validation():
    """The dependency treatment defaults off; contradictory requests fail at settings load —
    before any spend — with messages that say the fix (RVE design synthesis, C2/M7)."""
    import pytest
    from canopy.models import StatsSettings

    s = StatsSettings()
    assert s.dependency == "independent"
    assert s.rve_rho == 0.8

    ok = StatsSettings(dependency="cluster_robust", hakn=False, one_row_per_paper=False)
    assert ok.dependency == "cluster_robust"

    with pytest.raises(ValueError, match="hakn: false"):
        StatsSettings(dependency="cluster_robust", hakn=True, one_row_per_paper=False)
    with pytest.raises(ValueError, match="one_row_per_paper: false"):
        StatsSettings(dependency="cluster_robust", hakn=False, one_row_per_paper=True)
    with pytest.raises(ValueError, match="between 0 and 1"):
        StatsSettings(dependency="cluster_robust", hakn=False, one_row_per_paper=False,
                      rve_rho=1.5)
    # rho = 1.0 inclusive is legal (the golden fixture F4c is pinned AT 1.0 — review M7)
    assert StatsSettings(dependency="cluster_robust", hakn=False, one_row_per_paper=False,
                         rve_rho=1.0).rve_rho == 1.0
    # the bounds do not bite when the feature is off (nothing reads the knob then)
    assert StatsSettings(rve_rho=1.5).dependency == "independent"
