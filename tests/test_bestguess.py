"""Task 7 / DECISION A — the best-guess line: strict, plus the rows a rule admits.

Every case here is a REAL row of `runs/nine`: the fixture's records are what the resolver actually
produced, so a veto that fires here fires on evidence rather than on a hand-written flag list. The
one constructed row is D1's precedence override (Task 4 lands it on the run; the fixture predates
it), and it is constructed by copying the fixture's own record and applying exactly the fields the
resolver's fallback sets — so it agrees with the model, not with this test's imagination.
"""
from __future__ import annotations

import pytest

from tests.helpers import nine

from canopy.pipeline.bestguess import (BEST_GUESS_FLAG, RULES, VETOES, VETO_ROW_FLAGS,
                                       best_guess_payload, best_guess_rows, mark_composites)
from canopy.pipeline.run import _split_rows


def _lines(key, one_row_per_paper=False):
    p = nine.protocol()
    p.stats.one_row_per_paper = one_row_per_paper
    split = _split_rows(nine.records(key), p.stats)
    rows, dec = best_guess_rows(split.primary_pre_agg, split.held, outcome=p.outcome(key),
                                settings=p.stats)
    return split, rows, {d.dataset_id: d for d in dec}


@pytest.mark.parametrize("agg", [False, True])
def test_strict_input_rows_are_never_changed_or_removed(agg):
    for key in ("late_adaptation", "aftereffect"):
        split, rows, _ = _lines(key, agg)
        for s in split.primary_pre_agg:
            same = next(r for r in rows if r.dataset_id == s.dataset_id)
            # the module promises the OBJECT, not a row that happens to hold the same numbers:
            # a refactor that copied and renormalised a strict row would pass a value comparison
            assert same is s
            assert (same.es, same.var) == (s.es, s.var)


def test_low_confidence_rows_enter_at_their_own_value():
    _, rows, d = _lines("late_adaptation")
    for ds, es in (("5039533c85ef:d1", -0.22561429632968197),
                   ("592b3b55a318:d1", -0.4889045145111591),
                   ("592b3b55a318:d2", -0.09938079899999072)):
        assert d[ds].admitted and d[ds].rule == "low_confidence_value"
        assert next(r for r in rows if r.dataset_id == ds).es == es


def test_contradicting_flags_veto_unless_retired():
    _, _, d = _lines("late_adaptation")
    assert d["b7523a41b03a:d2"].veto == "contradicted_value"
    assert "series_identity_conflict" in d["b7523a41b03a:d2"].reason
    assert d["3570e4ce2a9c:d2"].veto == "contradicted_value"
    assert "axis_conflict" in d["3570e4ce2a9c:d2"].reason
    _, _, d = _lines("aftereffect")
    assert d["d1f2946e7e81:d1"].veto == "contradicted_value"


def test_a_retired_flag_no_longer_vetoes_the_rebuilt_row():
    """Controller ruling (d): the veto reads the row it is given, not the row's history.

    A human answer rebuilds the row without the flag it retired; nothing here remembers that the
    flag was ever there, which is what makes answering a question able to change an analysis.
    """
    held = nine.record("b7523a41b03a:d2", "late_adaptation")
    retired = held.model_copy(update={
        "flags": [f for f in held.flags if f not in ("series_identity_conflict", "axis_conflict")]})
    rows, dec = best_guess_rows([], [retired], outcome=nine.protocol().outcome("late_adaptation"),
                                settings=nine.protocol().stats)
    assert dec[0].admitted and dec[0].rule == "low_confidence_value"
    assert rows[0].es == held.es and rows[0].var == held.var


def test_unsigned_rows_are_vetoed_orientation_unresolvable():
    _, _, d = _lines("aftereffect")
    assert d["b511dbb76fa6:d1"].veto == "orientation_unresolvable"


def test_row_refusal_codes_are_the_resolvers():
    from canopy.verify.confidence import ROW_REFUSAL_CODES

    assert VETO_ROW_FLAGS is ROW_REFUSAL_CODES


def test_a_refused_row_is_vetoed_by_the_resolvers_own_code():
    """`row_refusal` wins over every other veto — an refused value is not a value at all."""
    code = sorted(VETO_ROW_FLAGS)[0]
    held = nine.record("5039533c85ef:d1", "late_adaptation")
    refused = held.model_copy(update={"flags": [*held.flags, code]})
    _, dec = best_guess_rows([], [refused], outcome=nine.protocol().outcome("late_adaptation"),
                             settings=nine.protocol().stats)
    assert dec[0].veto == "row_refusal" and code in dec[0].reason


def test_every_held_row_has_a_reason_and_a_closed_enum_code():
    for key in ("late_adaptation", "aftereffect"):
        _, _, d = _lines(key)
        for x in d.values():
            assert x.reason and ((x.rule in RULES) if x.admitted else (x.veto in VETOES))


def test_a_row_with_no_group_statistics_is_vetoed_one_group_only():
    _, _, d = _lines("aftereffect")
    assert d["b7523a41b03a:d1"].veto == "one_group_only"
    assert "have no mean" in d["b7523a41b03a:d1"].reason


def _override_row():
    """Heuer 1a late as D1's fallback builds it: the figure pair under the printed value's locator."""
    return nine.record("3570e4ce2a9c:d1", "late_adaptation").model_copy(update={
        "route": "figure", "es": -0.627, "var": 0.1, "se": 0.316, "confidence": "needs_human",
        "flags": ["precedence_override", "group_statistics_missing", "collapsed_across_x"],
        "precedence_override_reason": "text_mean_se_ci resolved 27.7/18.9 with no dispersion; …"})


def test_precedence_override_rule_admits_the_rebuilt_row():
    r = _override_row()
    rows, dec = best_guess_rows([], [r], outcome=nine.protocol().outcome("late_adaptation"),
                                settings=nine.protocol().stats)
    assert dec[0].rule == "precedence_override" and rows[0].es == -0.627


def test_the_override_reason_quotes_the_route_it_overruled_and_the_objection():
    _, dec = best_guess_rows([], [_override_row()],
                             outcome=nine.protocol().outcome("late_adaptation"),
                             settings=nine.protocol().stats)
    assert "text_mean_se_ci resolved 27.7/18.9" in dec[0].reason


def test_group_statistics_missing_does_not_veto_a_row_that_has_a_value():
    """The flag says why the printed route lost, not that the contrast is missing a group."""
    _, dec = best_guess_rows([], [_override_row()],
                             outcome=nine.protocol().outcome("late_adaptation"),
                             settings=nine.protocol().stats)
    assert dec[0].admitted and dec[0].veto == ""


def test_an_admitted_row_is_a_copy_that_carries_the_rule_and_the_reason():
    held = nine.record("5039533c85ef:d1", "late_adaptation")
    rows, dec = best_guess_rows([], [held], outcome=nine.protocol().outcome("late_adaptation"),
                                settings=nine.protocol().stats)
    assert rows[0] is not held and held.best_guess_rule == "" and BEST_GUESS_FLAG not in held.flags
    assert BEST_GUESS_FLAG in rows[0].flags
    assert f"best_guess_rule:{dec[0].rule}" in rows[0].flags
    assert rows[0].best_guess_rule == dec[0].rule and rows[0].best_guess_reason == dec[0].reason


def test_k_bg_is_never_below_k_strict_on_either_outcome():
    for key in ("late_adaptation", "aftereffect"):
        split, rows, _ = _lines(key)
        assert len(rows) >= len(split.primary_pre_agg)


def test_a_composite_with_a_guessed_member_is_a_guess_wholesale():
    from canopy.pipeline.aggregate import aggregate_one_row_per_paper

    p = nine.protocol()
    split = _split_rows(nine.records("late_adaptation"), p.stats)
    rows, dec = best_guess_rows(split.primary_pre_agg, split.held,
                                outcome=p.outcome("late_adaptation"), settings=p.stats)
    combined = mark_composites(aggregate_one_row_per_paper(rows, p.stats).rows, dec)
    composite = next(r for r in combined if "+" in r.dataset_id)
    assert BEST_GUESS_FLAG in composite.flags and composite.best_guess_reason
    for member in composite.dataset_id.split("+"):
        if member in dec and dec[member].admitted:
            assert member in composite.best_guess_reason


def test_the_payload_names_every_added_row_and_every_row_it_could_not_add():
    from canopy.report.tables import leave_one_out_rows, pool_rows, poolable_rows

    p = nine.protocol()
    split = _split_rows(nine.records("late_adaptation"), p.stats)
    rows, dec = best_guess_rows(split.primary_pre_agg, split.held,
                                outcome=p.outcome("late_adaptation"), settings=p.stats)
    added = [r for r in rows if BEST_GUESS_FLAG in r.flags]
    bg = pool_rows(rows, p.stats)
    weights = {r.dataset_id: float(w) for r, w in zip(poolable_rows(rows), bg.weights_pct)}
    payload = best_guess_payload(pool_rows(split.primary, p.stats), bg, dec, added_rows=added,
                                 loo_bg=leave_one_out_rows(rows, p.stats), rows=rows,
                                 weights=weights, settings=p.stats)
    assert payload["k"] == bg.k and payload["k_rows"] == len(rows)
    assert payload["n_added"] == len(added)
    assert {a["dataset_id"] for a in payload["added"]} == {r.dataset_id for r in added}
    assert {n["dataset_id"] for n in payload["not_added"]} == {
        d.dataset_id for d in dec if not d.admitted}
    assert payload["n_still_held"] == len(payload["not_added"])
    assert all(a["reason"] and a["rule"] in RULES for a in payload["added"])
    assert all(n["reason"] and n["veto"] in VETOES for n in payload["not_added"])
    assert payload["delta_vs_strict"] is not None and payload["note"]
    assert payload["max_abs_delta_from_one_best_guess_row"] > 0
    assert sum(a["weight_pct"] for a in payload["added"]) < 100


# ------------------------------------------------------- what "the dispute is quoted" means
def test_a_disputed_row_quotes_every_flag_the_checks_declare_as_a_dispute():
    """DECISION admits a disputed row only WITH the dispute quoted — all of it.

    Buch's two rows carry `calibration_disputed` (the check layer declares it `error`) and
    `unit_mismatch` (`warn`). Both are things a reader has to see before reading the value: one
    says the axis calibration is contested, the other that the two groups may not be in the same
    unit. Reading the flag's SPELLING quoted the first only because the letters "disput" sit in
    the middle of its name, and never quoted the second at all.
    """
    _, _, d = _lines("late_adaptation")
    for ds in ("592b3b55a318:d1", "592b3b55a318:d2"):
        assert d[ds].admitted and d[ds].reason.startswith("disputed (")
        assert "calibration_disputed" in d[ds].reason and "unit_mismatch" in d[ds].reason
        assert d[ds].evidence["disputed"] == ["calibration_disputed", "unit_mismatch"]


def test_the_dispute_mark_reads_the_declared_severity_not_the_flags_name():
    """Both directions: an undeclared code does not become a dispute by being named like one,
    and a declared one is quoted however innocuous its name reads."""
    held = nine.record("5039533c85ef:d1", "late_adaptation")
    row = held.model_copy(update={"flags": [*held.flags, "looks_disputed_but_undeclared"]})
    _, dec = best_guess_rows([], [row], outcome=nine.protocol().outcome("late_adaptation"),
                             settings=nine.protocol().stats)
    assert "looks_disputed_but_undeclared" not in dec[0].reason
    assert "panel_not_isolated" in dec[0].reason          # declared `warn`, quoted for it


def test_the_group_statistics_sentence_this_veto_reads_is_the_one_the_resolver_writes():
    """`one_group_only` matches prose on records resolved before D1's flag existed, so the two
    spellings have to be tied together by something the suite runs — not by a frozen fixture."""
    from canopy.pipeline.bestguess import _NO_GROUP_STATS
    from canopy.pipeline.resolve import ResolvedValues, resolve_effect

    p = nine.protocol()
    live = {(v.dataset_id, v.outcome_key, v.group): v for v in nine.verdicts("b7523a41b03a")}
    group_a = live[("b7523a41b03a:d1", "aftereffect", "A")]
    group_b = live[("b7523a41b03a:d1", "aftereffect", "B")]
    record = resolve_effect(nine.dataset("b7523a41b03a", "b7523a41b03a:d1"),
                            p.outcome("aftereffect"),
                            ResolvedValues.from_verdicts(group_a, group_b, higher_is_better=True),
                            p.stats)
    assert record.route == "not_convertible" and record.es is None
    assert _NO_GROUP_STATS.search(record.not_convertible_reason)
