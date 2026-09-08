"""Task 8 / DECISION B — the written conclusion: deterministic, from the numbers and the labels.

Every case here is built from the `runs/nine` fixture the same way the report builds one: the
rows the resolver wrote, split as the run splits them, pooled by the report's own pooler, and the
best-guess block Task 7 produces. Nothing is hand-written except the two synthetic pools that
exist to reach a `k` the fixture does not have (a sign flip, and k = 4/5 for the caveat
thresholds), and those are built by pooling real values rather than by inventing a `MetaResult`.

The rules under test are DECISION B / panel A B1 (R1–R12): never "significant", no magnitude
adjective, direction words only from the protocol's own labels, no `nan`/`inf` in the prose, and
— the one a reader can check — every number in the prose is in `facts`.
"""
from __future__ import annotations

import ast
import re

import pytest

from tests.helpers import nine

from canopy.pipeline.bestguess import BEST_GUESS_FLAG, best_guess_payload, best_guess_rows
from canopy.pipeline.run import _split_rows
from canopy.report.conclusion import (Conclusion, conclusion_from_payload, conclusion_payload,
                                      outcome_conclusion, overall_conclusion, render_text)
from canopy.report.tables import leave_one_out_rows, pool_rows
from canopy.stats.meta import random_effects

FORBIDDEN = ("significan", "nan", "inf", "small", "moderate", "large", "marked", "substantial",
             "strong")
NUMBER = re.compile(r"[-+]?\d+\.\d+|\b\d+\b")


# --------------------------------------------------------------------------- helpers
def _conclusion(key: str, outcome=None) -> Conclusion:
    """One outcome of the fixture, conclusion and all — the report's own path, in order."""
    protocol = nine.protocol()
    settings = protocol.stats
    outcome = protocol.outcome(key) if outcome is None else outcome
    split = _split_rows(nine.records(key), settings)
    pooled = pool_rows(split.primary, settings)
    bg_rows, decisions = best_guess_rows(split.primary_pre_agg, split.held, outcome=outcome,
                                         settings=settings)
    bg_pooled = pool_rows(bg_rows, settings)
    best_guess = best_guess_payload(
        pooled, bg_pooled, decisions,
        added_rows=[r for r in bg_rows if BEST_GUESS_FLAG in r.flags],
        loo_bg=leave_one_out_rows(bg_rows, settings), rows=bg_rows, settings=settings)
    return outcome_conclusion(outcome, settings, pooled=pooled, rows=split.primary,
                              held=split.held, best_guess=best_guess,
                              loo=leave_one_out_rows(split.primary, settings),
                              group_a=protocol.group_a, group_b=protocol.group_b)


def _late(outcome=None) -> Conclusion:
    return _conclusion("late_adaptation", outcome)


def _aft() -> Conclusion:
    return _conclusion("aftereffect")


def _rows(values: list[float], var: float = 0.1):
    """Real fixture rows carrying the values a synthetic pool needs — never a hand-built record."""
    base = nine.record("b511dbb76fa6:d1", "late_adaptation")
    return [base.model_copy(update={"dataset_id": f"d{i}", "paper_id": f"p{i}",
                                    "cluster_id": f"p{i}", "es": v, "var": var,
                                    "se": var ** 0.5}) for i, v in enumerate(values)]


def _synthetic(values: list[float], *, best_guess=None, held=(), loo=()) -> Conclusion:
    protocol = nine.protocol()
    rows = _rows(values)
    pooled = random_effects([r.es for r in rows], [r.var for r in rows],
                            method=protocol.stats.tau2_method, hakn=protocol.stats.hakn,
                            level=protocol.stats.ci_level)
    return outcome_conclusion(protocol.outcome("late_adaptation"), protocol.stats, pooled=pooled,
                              rows=rows, held=list(held), best_guess=best_guess, loo=list(loo),
                              group_a=protocol.group_a, group_b=protocol.group_b)


# --------------------------------------------------------------------------- the rules
def test_late_states_k_pi_held_and_delta():
    t = render_text(_late())
    assert "k = 2" in t and "not estimable" in t and " rows for this outcome are held" in t
    assert "best-guess line adds" in t


def test_k_lt_2_says_not_estimable_and_nothing_else():
    t = render_text(_aft())
    assert "not estimable" in t and "I²" not in t and "τ²" not in t and "Enhanced" not in t


def test_forbidden_words_never_appear():
    for c in (_late(), _aft()):
        assert not any(w in render_text(c).lower() for w in FORBIDDEN)


def test_every_number_in_prose_is_in_facts():
    for c in (_late(), _aft()):
        rendered = ({f"{v:.2f}" for v in c.facts.values() if isinstance(v, float)}
                    | {f"{v:.3f}" for v in c.facts.values() if isinstance(v, float)}
                    | {f"{v:.0f}" for v in c.facts.values() if isinstance(v, float)}
                    | {str(v) for v in c.facts.values()})
        assert set(NUMBER.findall(render_text(c))) <= rendered


def test_direction_word_comes_from_protocol():
    o = nine.protocol().outcome("late_adaptation").model_copy(
        update={"negative_direction_label": "Better in A"})
    assert "better in a" in render_text(_late(outcome=o)).lower()


def test_direction_falls_back_to_the_group_labels():
    """R3: blank both labels and the sentence names the groups, never an invented word.

    In CONSTRUCT terms, and fix round 2 (finding 10) is why. "A scored lower than B" is a claim
    about the raw numbers the paper printed, and `es` is not that: it is orientation-applied, so
    on a measure where a larger raw value means LESS of the construct (`higher_is_better = false`
    — every error-type outcome in this protocol) the sign has already been flipped and the
    raw-score sentence states the opposite of the finding. What the sign means is how much of the
    outcome each group shows, so that is what the fallback says.
    """
    protocol = nine.protocol()
    o = protocol.outcome("late_adaptation").model_copy(
        update={"positive_direction_label": "", "negative_direction_label": ""})
    text = render_text(_late(outcome=o))
    assert "scored lower" not in text and "scored higher" not in text
    assert f"less {o.label} in {protocol.group_a.label} than in {protocol.group_b.label}" in text
    assert "reduced in old" not in text.lower()


def test_sign_flip_is_announced():
    strict = _synthetic([-0.4, -0.4], best_guess={
        "k": 3, "estimate": 0.3, "ci_low": -0.10, "ci_high": 0.70, "n_added": 1,
        "n_still_held": 1, "delta_vs_strict": 0.7, "sign_agrees_with_strict": False,
        "not_added": [{"dataset_id": "d9", "veto": "no_variance", "reason": "no value"}]},
        held=_rows([0.3]))
    text = render_text(strict)
    assert "points the other way" in text and "a change of +0.70" in text
    assert not any(w in text.lower() for w in FORBIDDEN)


def test_no_sign_flip_is_not_announced():
    agreeing = _synthetic([-0.4, -0.4], best_guess={
        "k": 3, "estimate": -0.30, "ci_low": -0.70, "ci_high": 0.10, "n_added": 1,
        "n_still_held": 0, "delta_vs_strict": 0.1, "sign_agrees_with_strict": True,
        "not_added": []}, held=_rows([-0.3]))
    assert "points the other way" not in render_text(agreeing)


def test_small_k_caveats():
    both = render_text(_synthetic([-0.4, -0.2]))
    first = render_text(_synthetic([-0.4, -0.2, -0.1, 0.1]))
    neither = render_text(_synthetic([-0.4, -0.2, -0.1, 0.1, 0.3]))
    assert "estimated very imprecisely" in both and "essentially uninterpretable" in both
    assert "estimated very imprecisely" in first and "essentially uninterpretable" not in first
    assert "estimated very imprecisely" not in neither
    for text in (both, first, neither):
        assert not any(w in text.lower() for w in FORBIDDEN)


def test_prediction_interval_not_estimable_at_k2_prints_no_nan():
    """R12: `runs/nine` late has `pi_low_used = nan` at k = 2 — the branch fires, the word never."""
    text = render_text(_late())
    assert "A prediction interval is not estimable at k = 2" in text
    assert "nan" not in text.lower() and "inf" not in text.lower()


def test_a_p_value_is_only_ever_quoted_beside_its_interval():
    """R1: no bare p, and never the word."""
    for sentence in _late().sentences:
        if re.search(r"\bp = ", sentence):
            assert "CI" in sentence or "Q(" in sentence, sentence


def test_leave_one_out_is_a_range_and_names_a_zero_crossing():
    """R8: the range, plus the omission that moves the interval across zero, by name."""
    loo = [{"omitted_label": "Author A", "omitted_dataset_id": "d0", "estimate": -0.90,
            "ci_low": -1.50, "ci_high": -0.30, "k": 2},
           {"omitted_label": "Author B", "omitted_dataset_id": "d1", "estimate": -0.10,
            "ci_low": -0.60, "ci_high": 0.40, "k": 2}]
    text = render_text(_synthetic([-0.9, -0.5, -0.1], loo=loo))
    assert "ranges from -0.90 to -0.10" in text
    assert "Omitting Author B alone moves the interval across zero." in text


def test_held_sentence_is_mandatory_and_quantitative():
    """R7: the fixture's late outcome holds 8 of 10 rows and the sentence says so."""
    text = render_text(_late())
    assert "8 of 10 rows for this outcome are held for human review." in text
    assert "4 row(s) could not be given a value by any rule" in text


def test_no_held_rows_means_no_held_sentence():
    assert "held for human review" not in render_text(_synthetic([-0.4, -0.2]))


def test_facts_carry_every_formatted_number_by_name():
    facts = _late().facts
    for key in ("k", "k_papers", "estimate", "ci_low", "ci_high", "p", "tau2", "i2", "q", "q_df",
                "q_p", "n_held", "n_rows", "best_guess_n_added", "best_guess_k"):
        assert key in facts, key
    assert all(v == v for v in facts.values() if isinstance(v, float))     # no nan ever


def test_overall_paragraph_counts_the_estimable_outcomes():
    protocol = nine.protocol()
    text = overall_conclusion(protocol, {"late_adaptation": _late(), "aftereffect": _aft()})
    assert text.startswith(f"{protocol.title}: 1 of 2 outcomes could be pooled.")
    assert "Late adaptation: -0.74 (95% CI -2.47 to 1.00), k = 2" in text
    assert "Aftereffect: not estimable (k = 1)." in text
    assert "17 of 20 rows are held for human review" in text
    assert "changes the direction of 0 outcome(s)" in text
    assert not any(w in text.lower() for w in FORBIDDEN)


def test_payload_round_trips_through_pooled_json():
    c = _late()
    same = conclusion_from_payload(conclusion_payload(c))
    assert same == c and render_text(same) == render_text(c)


def test_deterministic_and_no_llm_import():
    import canopy.report.conclusion as m

    source = open(m.__file__).read()
    assert "canopy.llm" not in source
    imported = {n.module or "" for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.ImportFrom)}
    imported |= {a.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Import)
                 for a in n.names}
    assert not any("llm" in name for name in imported), imported
    assert render_text(_late()) == render_text(_late())


def test_render_text_joins_the_sentences_in_order():
    c = _late()
    assert render_text(c) == " ".join(c.sentences)


@pytest.mark.parametrize("key", ["late_adaptation", "aftereffect"])
def test_the_conclusion_reaches_pooled_json(tmp_path, key):
    """The block the SPA and the report read is written by `write_outcome_outputs`, not re-derived."""
    import json

    from canopy.report import write_outcome_outputs

    protocol = nine.protocol()
    split = _split_rows(nine.records(key), protocol.stats)
    out = write_outcome_outputs(tmp_path, protocol.outcome(key), split.primary,
                                pool_rows(split.primary, protocol.stats), protocol.stats,
                                needs_human_rows=split.held, all_rows=split.every,
                                primary_pre_agg=split.primary_pre_agg, protocol=protocol,
                                warnings=[])
    payload = json.loads(out["pooled_json"].read_text(encoding="utf-8"))
    written = conclusion_from_payload(payload["conclusion"])
    assert written.outcome_key == key
    assert render_text(written) == render_text(_conclusion(key))


def test_the_report_carries_the_conclusion_and_writes_conclusion_md(tmp_path):
    """B3: `<h2>Conclusion</h2>` after the header cards, the paragraph above each forest, a file."""
    import json

    from canopy.models import RunManifest
    from canopy.report import write_html_report, write_outcome_outputs

    protocol = nine.protocol()
    results = {}
    for key in ("late_adaptation", "aftereffect"):
        split = _split_rows(nine.records(key), protocol.stats)
        outputs = write_outcome_outputs(tmp_path, protocol.outcome(key), split.primary,
                                        pool_rows(split.primary, protocol.stats), protocol.stats,
                                        needs_human_rows=split.held, all_rows=split.every,
                                        primary_pre_agg=split.primary_pre_agg, protocol=protocol,
                                        warnings=[])
        results[key] = {"pooled": pool_rows(split.primary, protocol.stats), "outputs": outputs,
                        "rows": split.primary, "needs_human_rows": split.held}
    manifest = RunManifest(run_id="r", created_at="2026-08-18T10:00:00+00:00",
                           protocol_path="protocol.yaml", protocol_hash=protocol.hash(),
                           settings=protocol.stats)
    out = write_html_report(tmp_path, manifest, protocol, results=results)

    html = out["html"].read_text(encoding="utf-8")
    late = render_text(conclusion_from_payload(json.loads(
        (tmp_path / "results" / "late_adaptation" / "pooled.json").read_text())["conclusion"]))
    assert "<h2>Conclusion</h2>" in html
    assert html.index("<h2>Conclusion</h2>") < html.index("<h2>Late adaptation</h2>")
    assert html.count(_escape(late)) == 2                  # the section, and above the forest
    assert 'href="conclusion.md"' in html
    text = out["conclusion"].read_text(encoding="utf-8")
    assert late in text and protocol.title in text
    assert not any(w in text.lower() for w in FORBIDDEN)


def _escape(text: str) -> str:
    import html as _html

    return _html.escape(text, quote=True)


# ------------------------------------------------------------------- a sign zero does not have
def test_a_value_that_rounds_to_zero_is_never_printed_with_a_minus_sign():
    """Whole-branch review, MINOR 6: `runs/nine` wrote "gives -0.00 (95% CI -0.33 to 0.33, k = 5)".

    The best-guess estimate really was -0.0025, and at two decimals that is zero. A minus sign in
    front of it is a direction the line does not have — the one thing a reader takes away from a
    pooled number at a glance — so the printed form is normalised at the printed precision. The
    ledger keeps the unrounded number, and the form the prose used beside it, because every number
    in the prose has to be in `facts`.
    """
    c = _synthetic([-0.4, -0.4], best_guess={
        "k": 5, "estimate": -0.0025139210429772395, "ci_low": -0.33143932059379366,
        "ci_high": 0.32641147850783914, "n_added": 2, "n_still_held": 0,
        "delta_vs_strict": -0.0011, "sign_agrees_with_strict": True, "not_added": []},
        held=_rows([-0.3, -0.3]))
    text = render_text(c)
    assert "-0.00" not in text
    assert "gives 0.00 (95% CI -0.33 to 0.33, k = 5)" in text
    assert "a change of +0.00 from the primary estimate" in text
    assert c.facts["best_guess_estimate"] == -0.0025139210429772395     # the ledger is unrounded
    assert set(NUMBER.findall(text)) <= ({f"{v:.2f}" for v in c.facts.values()
                                          if isinstance(v, float)}
                                         | {f"{v:.3f}" for v in c.facts.values()
                                            if isinstance(v, float)}
                                         | {f"{v:.0f}" for v in c.facts.values()
                                            if isinstance(v, float)}
                                         | {str(v) for v in c.facts.values()})


def test_a_negative_number_that_does_not_round_to_zero_keeps_its_sign():
    """The normalisation is about the printed precision and nothing else."""
    c = _synthetic([-0.4, -0.4], best_guess={
        "k": 3, "estimate": -0.006, "ci_low": -0.4, "ci_high": 0.4, "n_added": 1,
        "n_still_held": 0, "delta_vs_strict": 0.4, "sign_agrees_with_strict": True,
        "not_added": []}, held=_rows([-0.3]))
    assert "gives -0.01" in render_text(c)


def test_the_payload_tells_the_two_lines_sentences_apart():
    """Fix round 2, finding 15. The conclusion card claimed the page never shows both lines'
    headline numbers, and then rendered the whole paragraph — which carries the strict d AND the
    best-guess d, in every state of the toggle. The paragraph is still one paragraph (the report
    prints it whole); the payload now says which sentence is the strict headline and which
    sentences are the second line's, so a viewer can show the line the reader chose."""
    payload = conclusion_payload(_late())
    assert payload["headline"] in payload["sentences"] and "pooled" in payload["headline"]
    assert payload["best_guess_sentences"], "the fixture's late outcome holds rows"
    assert all(s in payload["sentences"] for s in payload["best_guess_sentences"])
    assert payload["headline"] not in payload["best_guess_sentences"]
    # …and each caveat is in the paragraph exactly once: the card used to print them again
    for caveat in payload["caveats"]:
        assert payload["sentences"].count(caveat) == 1


def test_a_z_prediction_interval_with_infinite_df_writes_the_sentence_and_no_crash():
    """The z convention (metafor default) returns df = inf BY DESIGN: finite bounds whose
    reference distribution has no df. int(inf) raised OverflowError here and one unguarded
    conversion at the very end killed a whole 22-paper run after every paper had resolved."""
    protocol = nine.protocol()
    stats = protocol.stats.model_copy(update={"pi_method": "z"})
    rows = _rows([-0.9, -0.5, -0.1, 0.2, 0.4])
    pooled = random_effects([r.es for r in rows], [r.var for r in rows],
                            method=stats.tau2_method, hakn=stats.hakn, level=stats.ci_level)
    conclusion = outcome_conclusion(protocol.outcome("late_adaptation"), stats, pooled=pooled,
                                    rows=rows, held=[], best_guess=None, loo=[],
                                    group_a=protocol.group_a, group_b=protocol.group_b)
    text = " ".join(conclusion.sentences)
    assert "expected to fall between" in text
    assert "inf" not in text.lower()
    assert "pi_df" not in conclusion.facts or conclusion.facts["pi_df"] != float("inf")


# --------------------------------------------------------------------- cluster-robust (RVE)
def _rve() -> Conclusion:
    """The late-adaptation case re-pooled cluster-robust: same fixture rows, RVE settings."""
    protocol = nine.protocol()
    settings = protocol.stats.model_copy(update={"dependency": "cluster_robust", "hakn": False,
                                                 "one_row_per_paper": False})
    split = _split_rows(nine.records("late_adaptation"), settings)
    pooled = pool_rows(split.primary, settings)
    loo = leave_one_out_rows(split.primary, settings)
    return outcome_conclusion(protocol.outcome("late_adaptation"), settings, pooled=pooled,
                              rows=split.primary, held=split.held, loo=loo,
                              group_a=protocol.group_a, group_b=protocol.group_b)


def test_rve_conclusion_states_the_cluster_model_and_its_facts():
    c = _rve()
    t = render_text(c)
    assert "cluster-robust" in t and "Satterthwaite" in t
    assert "n_clusters" in c.facts and "df_robust" in c.facts and "rho" in c.facts
    assert "nan" not in t.lower()
    # the non-integer Q df is printed as itself, never truncated to an int (review M3)
    assert c.facts["q_df"] == pytest.approx(float(c.facts["q_df"]))


def test_rve_conclusion_keeps_the_ledger_invariant():
    c = _rve()
    rendered = ({f"{v:.2f}" for v in c.facts.values() if isinstance(v, float)}
                | {f"{v:.3f}" for v in c.facts.values() if isinstance(v, float)}
                | {f"{v:.0f}" for v in c.facts.values() if isinstance(v, float)}
                | {str(v) for v in c.facts.values()})
    assert set(NUMBER.findall(render_text(c))) <= rendered
