"""Task 16 (f): which of a paper's locations the VOTE weighs, decided in code.

Heuer & Hegele 2008 (`paper_1755470.pdf`) is the case these rules exist for. Its map lists, for the
adaptation outcome: a page-4 sentence with no n and no dispersion, an Experiment 2 figure measuring
something else, and — the source the published review used — Fig 2a on page 5 with SE bars and
n = 20/20. All three were weighed equally, and the paper produced two `not_convertible` rows while
carrying a perfectly usable figure.

Nothing here calls a model: the rank is built from fields the mapper already returns, because a
model's own "this is the best source" is one more unwitnessed number — and the mapper is the agent
that got the choice wrong.
"""
from __future__ import annotations

import pytest

from canopy.agents.source_rank import (keep_for_vote, match_named_source, rank_sources, source_of)
from canopy.models import (Candidate, DispersionType, OutcomeSources, Source, SourceKind)


def _sources() -> list[Source]:
    """Heuer & Hegele's three locations for the adaptation outcome, as the mapper listed them."""
    return [
        Source(kind=SourceKind.text_mean_sd, page=4, locator="Results, first paragraph",
               quote="adaptation was smaller in the older group",
               analysis_metric="endpoint"),
        Source(kind=SourceKind.figure_line, page=8, locator="Fig 6 (Experiment 2)",
               figure_id="fig06", error_bar_type=DispersionType.SE,
               analysis_metric="change_from_baseline"),
        Source(kind=SourceKind.figure_line, page=5, locator="Fig 2a", figure_id="fig02",
               error_bar_type=DispersionType.SE, analysis_metric="endpoint",
               values_in_text="n = 20 per group"),
    ]


def _outcome() -> OutcomeSources:
    return OutcomeSources(outcome_key="late_adaptation", measure_name="adaptation",
                          units="deg", analysis_metric="endpoint", sources=_sources())


def test_the_figure_with_error_bars_and_an_n_outranks_the_sentence_without_either():
    """Acceptance item 12."""
    ranked = rank_sources(_sources(), _outcome())
    assert [r.locator for r in ranked] == ["Fig 2a", "Fig 6 (Experiment 2)",
                                           "Results, first paragraph"]
    best = ranked[0]
    assert best.source.figure_id == "fig02" and best.source.page == 5
    assert best.has_dispersion and best.has_n and best.matches_metric
    assert best.score > ranked[1].score > ranked[2].score


def test_every_source_that_lost_records_what_it_lost_on():
    ranked = {r.locator: r for r in rank_sources(_sources(), _outcome())}
    text = ranked["Results, first paragraph"]
    assert any("no dispersion" in why for why in text.losing_reasons)
    assert any("no group size" in why for why in text.losing_reasons)
    assert any("against the best source's 8" in why for why in text.losing_reasons)

    exp2 = ranked["Fig 6 (Experiment 2)"]
    assert any("measures change_from_baseline, where this outcome is endpoint" in why
               for why in exp2.losing_reasons)
    assert exp2.has_dispersion, "Exp 2 does have error bars — it loses on the metric, not on them"
    assert not ranked["Fig 2a"].losing_reasons


def _cand(cid: str, page: int, figure_id: str | None = None, mean: float = 12.0) -> Candidate:
    return Candidate(candidate_id=cid, paper_id="p", dataset_id="d1",
                     outcome_key="late_adaptation", kind="group_stats", group="A",
                     status="found", mean=mean, page=page,
                     source_kind=SourceKind.figure_line if figure_id else SourceKind.text_mean_sd,
                     extractor_id="digitize:ensemble" if figure_id else "text:table_first:opus",
                     pixel_provenance={"figure_id": figure_id} if figure_id else {})


def test_a_source_that_cannot_make_an_effect_size_is_held_out_of_the_vote():
    ranked = rank_sources(_sources(), _outcome())
    cands = [_cand("fig2a", 5, "fig02"), _cand("text_p4", 4)]
    kept, held = keep_for_vote(cands, ranked)
    assert [c.candidate_id for c in kept] == ["fig2a"]
    assert len(held) == 1 and "text_p4" in held[0] and "no dispersion" in held[0]


def test_a_source_that_merely_scores_lower_still_votes():
    """Evicting a figure that scores lower removes corroboration, not error."""
    ranked = rank_sources(_sources(), _outcome())
    cands = [_cand("fig2a", 5, "fig02"), _cand("fig6", 8, "fig06", mean=9.0)]
    kept, held = keep_for_vote(cands, ranked)
    assert {c.candidate_id for c in kept} == {"fig2a", "fig6"}
    assert held == []


def test_the_filter_never_empties_a_cell_and_never_drops_what_it_cannot_place():
    ranked = rank_sources(_sources(), _outcome())
    only_loser = [_cand("text_p4", 4)]
    kept, held = keep_for_vote(only_loser, ranked)
    assert [c.candidate_id for c in kept] == ["text_p4"] and held == []
    # a candidate on a page no source claims cannot be attributed, so it votes
    stray = [_cand("fig2a", 5, "fig02"), _cand("mystery", 99)]
    kept, _ = keep_for_vote(stray, ranked)
    assert {c.candidate_id for c in kept} == {"fig2a", "mystery"}


def test_a_candidate_is_attributed_by_figure_id_first_then_page():
    sources = _sources()
    assert source_of(_cand("a", 5, "fig02"), sources).figure_id == "fig02"
    assert source_of(_cand("b", 4), sources).page == 4
    assert source_of(_cand("c", 99), sources) is None


# --------------------------------------------------------------------------- better_source
def test_a_better_source_is_only_acted_on_when_the_mapper_already_had_it():
    """`better_source` was written, stored, printed — and consumed nowhere (critique §1 P5)."""
    sources = _sources()
    named = match_named_source("Figure 2a on page 5, which carries SE bars and n = 20", sources)
    assert named is not None and named.figure_id == "fig02"
    # …and a location the mapper never found is refused rather than believed
    assert match_named_source("Table 7 (page 12)", sources) is None
    assert match_named_source("", sources) is None
    assert match_named_source("somewhere in the discussion", sources) is None


@pytest.mark.parametrize("named,figure", [
    ("Fig. 2a", "fig02"), ("figure 2A (p. 5)", "fig02"), ("Fig 6 (Experiment 2)", "fig06")])
def test_the_matcher_reads_the_ways_a_verifier_writes_a_locator(named, figure):
    matched = match_named_source(named, _sources())
    assert matched is not None and matched.figure_id == figure
