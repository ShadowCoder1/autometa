"""Confidence (spec §3.3(6)) and `resolve_cell` — the one call Task 10 makes per cell.

`confidence` turns everything the verification layer produced into one of three buckets. It is
additive and boring on purpose: a score built from named contributions can be explained to a
reviewer line by line, and the reasons it returns are exactly those lines.

    where the value came from   text/table 0.40 · figure 0.35 · statistic 0.30
    agreement                   two heterogeneous routes +0.25 (+0.05 per further route, ≤ +0.10)
    grounding                   every quote in the paper +0.15 · a short quote +0.05 · missing −0.30
    adversarial verifier        confirmed +0.20 · ambiguous −0.05 · refuted → human
    spread                      MAD > 5% of the value −0.05 · digitisation σ > 10% −0.05
    consistency flags           warn −0.08 each (≤ −0.24) · info −0.01 each (≤ −0.03)

Four things override the arithmetic and send the cell to a human: an `error` flag, a refutation,
an unresolved direction for the measure, and an adjudicator that asked for one. Two things cap it:
a value only one route produced, and a value an adjudicator had to settle — neither may be
accepted automatically. Figure-derived cells have their own gate (amendment F): the digitizer
routes must imply the same effect to within |Δd| < 0.1 and the digitisation uncertainty must be
under 10% of the sampling error, or the cell is `accept_with_note` at best.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from ..models import (Adjudication, Candidate, CheckFlag, ConfidenceBucket, DatasetSpec,
                      DispersionType, OrientationVerdict, VerifierVerdict, Verdict)
from ..stats.effect_sizes import cohens_d, pooled_sd, se_smd
from .checks import MAX_PLAUSIBLE_D, run_checks, severity_of
from .grounding import is_short_quote
from .vote import VoteResult, digitizer_path, is_figure_route, modality, vote

__all__ = ["confidence", "resolve_cell", "figure_gate", "AUTO_ACCEPT", "ACCEPT_WITH_NOTE",
           "DELTA_D_LIMIT", "DIGITIZATION_SE_SHARE", "SINGLE_ROUTE_CAP", "ADJUDICATED_CAP",
           "partial_mean_readers", "PARTIAL_READ_MISSING_MEAN",
           "verifier_state", "NOT_RUN", "NO_VALUE_PRINTED", "NO_EVIDENCE_VERDICTS",
           "confidence_margin", "MARGIN_BAND", "DECIDED_BY_A_HAIR", "BUCKET_BOUNDARIES",
           "BUCKET_NOT_SCORED",
           "conversion_gate_bucket", "CONVERTED_ROUTES", "UNVERIFIED_CONTRAST_FLAGS",
           "DF_SHORTFALL_PREFIX", "dispersion_plausibility_bucket", "IMPLAUSIBLE_DISPERSION",
           "ROW_REFUSAL_CODES",
           "unverified_variance_bucket", "UNVERIFIED_VARIANCE_FLAGS"]

AUTO_ACCEPT = 0.75              # at or above this the pipeline pools the row without a human
ACCEPT_WITH_NOTE = 0.45         # below this a human decides
SINGLE_ROUTE_CAP = 0.70         # one route is never "accepted by vote" (spec §3.3(2))
ADJUDICATED_CAP = 0.70          # a cell an LLM had to settle is always reviewed at least once

#: amendment F, moved here by controller ruling: what a digitised cell must satisfy to auto-accept
DELTA_D_LIMIT = 0.1             # implied |Δd| across digitizer routes
DIGITIZATION_SE_SHARE = 0.1     # digitisation SE as a share of the sampling SE

BASE_SCORE = {"text": 0.40, "table": 0.40, "figure": 0.35, "statistic": 0.30}
AGREE_BONUS = 0.25
EXTRA_ROUTE_BONUS, EXTRA_ROUTE_CAP = 0.05, 0.10
DISAGREE_PENALTY = 0.20
GROUNDED_BONUS, SHORT_QUOTE_BONUS, UNGROUNDED_PENALTY = 0.15, 0.05, 0.30
CONFIRMED_BONUS, AMBIGUOUS_PENALTY = 0.20, 0.05

#: ---------------------------------------------------------------- C8: what a verifier RECORD is
#: `ambiguous` is a reader that looked at the paper and could not tell. It costs 0.05 because it
#: is a doubt about the reading. Two other things used to be written into that same word, and
#: neither is a doubt about anything:
#:
#: `not_run` — no agent produced a verdict at all. `pipeline/run.py` records a `VerifierVerdict`
#:   when the call raised `TruncatedOutput` or `LLMError`, so the failure is on the record and the
#:   cell is not silently unverified — but a transport error, an output cut off at its token limit
#:   and an exhausted budget are facts about this laptop, not about the paper. A cell whose only
#:   verifier failed must score exactly what a cell whose verifier was never scheduled scores.
#: `no_value_printed` — an agent DID run and reported, successfully, that the paper prints no
#:   independent value for this cell. That is a completed cross-check whose answer is "there is
#:   nothing here to check this against": the reading is exactly as corroborated as it was before
#:   the check ran, and charging it 0.05 punishes a paper for publishing a figure without a table.
#:
#: Both move the score by ZERO and both say so in the reasons, because a reviewer must be able to
#: tell "nothing contradicted this" from "nobody looked".
NOT_RUN, NO_VALUE_PRINTED = "not_run", "no_value_printed"
NO_EVIDENCE_VERDICTS: frozenset[str] = frozenset({NOT_RUN, NO_VALUE_PRINTED})

#: ------------------------------------------------------------------------ C11: publish the margin
#: The two thresholds that decide a bucket. A cell sitting within `MARGIN_BAND` of either one was
#: decided by a hair, whichever side of it the cell fell on: 0.4500 pools by 0.0000 and 0.7400
#: fails to auto-accept by 0.0100, and both facts belong on the record rather than in the head of
#: whoever reads the score. The caps (`SINGLE_ROUTE_CAP`, `ADJUDICATED_CAP`) are deliberately NOT
#: boundaries here — a capped cell lands exactly ON its ceiling by construction, so every one of
#: them would carry a margin of 0.0000 and the label would mean "this cell was capped", which the
#: cap reason already says.
BUCKET_BOUNDARIES: dict[str, float] = {"accept_with_note": ACCEPT_WITH_NOTE,
                                       "auto_accept": AUTO_ACCEPT}
MARGIN_BAND = 0.03
DECIDED_BY_A_HAIR = "decided_by_a_hair"

#: ------------------------------------------------- C9: the conversion gate's own flags, priced
#: Routes whose value is a printed test statistic turned into a standardised mean difference.
#: Both vocabularies are here on purpose: `canopy.pipeline.resolve` names the route it *tried*
#: (`test_statistic`, `p_value`) and `canopy.stats.effect_sizes` names the route it *took*
#: (`t_stat`, `f_stat`, `p_value`), and the gate has to fire on either.
CONVERTED_ROUTES: frozenset[str] = frozenset({"test_statistic", "statistic", "p_value",
                                              "t_stat", "f_stat"})
#: Flags that say the statistic's provenance TO THESE TWO GROUPS was never established. A paper
#: printing "t = 5.25, p < .001" as a post-hoc from a three-group ANOVA satisfies every arithmetic
#: check there is; nothing in it says the 5.25 is the contrast this row pools. `convertibility`
#: raises `df_missing` and passes, and `canopy.verify.checks` raises the cell-level counterparts.
UNVERIFIED_CONTRAST_FLAGS: frozenset[str] = frozenset({"df_missing", "test_stat_missing_df",
                                                       "df_shortfall_unexplained"})
#: `df_off_by_1` and friends: a shortfall the paper EXPLAINS (a stated exclusion that also reduces
#: n, or an n that was inferred rather than printed). Allowed, never automatic.
DF_SHORTFALL_PREFIX = "df_off_by_"
WARN_PENALTY, WARN_CAP = 0.08, 0.24
INFO_PENALTY, INFO_CAP = 0.01, 0.03
MAD_SHARE, SIGMA_SHARE, SPREAD_PENALTY = 0.05, 0.10, 0.05
FIGURE_CONFLICT_PENALTY = 0.05
#: TWO DOUBTS, and they are not the same doubt. Both used to live in `CAPPING_FLAGS` under one
#: floor, which made the second incapable of withholding anything (controller ruling R2 as first
#: written; overturned in part after a wrong-panel figure read landed on exactly 0.4500).
#:
#: `CAPPING_FLAGS` — **"this reading is under-corroborated."** Nothing here says the number is the
#: wrong number: each one says the scale, the error-bar type, the location or the dispersion rests
#: on one witness, one model's choice or an approximation. A reviewer should LOOK at such a cell;
#: withholding it on that alone would bury correct readings, which is the failure task 16 exists
#: to remove. So these are deducted, capped, and then floored back at `ACCEPT_WITH_NOTE`.
CAPPING_FLAGS: frozenset[str] = frozenset({
    #: one witness calibrated the axis: the value may be right and nothing corroborates the scale
    "calibration_single_witness",
    #: the read-outs agree on a value the ladder cannot draw — their number stands, the ladder does not
    "calibration_refuted",
    #: an average across a categorical axis the protocol ASKED for, with a dispersion the code
    #: approximated: a declared transformation whose spread is not the paper's own
    "collapsed_across_x",
    #: …and its opposite number (D3): the value is the ONE point at the x category the source
    #: names, so nothing was averaged and nothing approximated — whatever spread it carries is
    #: the paper's own band at that category, or there is none. What is thin here is
    #: corroboration: a series with one point in the frame has no neighbouring point to agree
    #: with it. A reviewer should see the figure; the number itself is not in question.
    "categorical_point_read",
    #: what the categorical x axis IS was resolved from the readers' own category reports, and
    #: two or more readings support the ruling on their own — figure evidence, corroborated. Its
    #: single-witness sibling is NOT here: one reading's word about what the axis is is an
    #: inferred premise (`INFERRED_PREMISE_FLAGS` below) and holds the row for a person.
    "categorical_x_resolved_from_readings",
    #: extraction was re-opened on a source the verifier named, so a model chose the location
    "reopened_on_better_source",
    #: no ladder for the value axis could be built at all: the numbers rest on the readers' own
    #: sense of the scale, and nothing contradicts them either
    "calibration_missing",
    "calibration_two_point",
    #: the error-bar type came from the figure's legend because the map never determined one
    "dispersion_type_from_legend",
    #: the reader's words about this series and the markers the pixel pass found do not line up.
    #: A soft doubt about a shape vocabulary — the *transposition* it used to share a code with
    #: is `series_transposed`, below, because the two have opposite consequences.
    "series_marker_mismatch",
    #: how thin the agreement behind the measure's DIRECTION was (ceiling items C3 / C12 / the
    #: P-A residue, raised in `canopy.verify.checks`). Neither says the direction is wrong — one
    #: says a single ballot set it after the other was re-issued or discarded, the other says a
    #: majority of three did. Both are reasons for a reviewer to look, which is why they cap
    #: rather than convict: `orientation.higher_is_better is None` is what withholds a cell.
    "orientation_single_witness",
    "orientation_by_majority",
    #: …and the third of the same kind (review L3): the two readers read the paper's own sentence
    #: about which group came out higher in OPPOSITE directions, so `sign_check` has nothing left
    #: to compare the extracted numbers with. It is a statement about how thin the agreement on
    #: the direction was, not evidence that this number is the wrong number — its sibling
    #: `orientation_reader_contradicts_values` is the one that says the numbers themselves are in
    #: question, and that one is a contradiction. In neither family it was deducted in the generic
    #: warn bucket ABOVE the floor, so unlike its three siblings it could withhold a cell on its
    #: own at the margin, and it printed as a bare code with nothing said to the reviewer.
    "orientation_direction_conflict",
    #: the named panel could not be isolated from its neighbours, so the reader was handed the
    #: union crop and a neighbouring panel's ladder was in the frame. Doubt about the SCALE, not
    #: evidence that this number came from somewhere else — the axis-identity, overlay and
    #: verifier nets all still apply — so it caps like its neighbours here rather than withholding.
    "panel_not_isolated",
    #: fix E's sibling of the line above: the reading was re-acquired from the full page after
    #: the panel crop was refused — wider than any panel, a reason to look, never to withhold
    "crop_reacquired",
    #: a reading taken off another group's panel was set aside, and this group's OWN panel still
    #: has one. The number that survives came from where the caption says it should have — the
    #: doubt left is that a reader went to the wrong panel at all, which is a reason to look at
    #: the cell, not evidence against the reading that stands (D2).
    "locator_reads_set_aside",
    #: the group size behind this cell is one the paper prints BEFORE the exclusions it reports,
    #: so the analysed arm may be smaller than the row's denominator (D4-lite). It caps rather
    #: than contradicts for the same reason its neighbours do: the MEAN is not in question, the
    #: value came from where it was asked for, and what an n four people too large moves is the
    #: variance — a reason to look at the cell, not evidence that the number is another quantity.
    "n_before_exclusions",
})

#: `CONTRADICTING_FLAGS` — **"this may be a different quantity."** Each one is evidence that the
#: number was measured somewhere other than where it was asked for: off another value axis, off
#: the other series, out of another column, with the two groups swapped. No amount of agreement
#: about a number establishes that it is the right number, and nothing downstream can recover: the
#: meta-analysis pools whatever it is handed, at whatever sign it carries. These are therefore
#: EXEMPT from the floor below and force a human — they are evidence, not a cap.
CONTRADICTING_FLAGS: frozenset[str] = frozenset({
    #: the numbers are in the named table row but not the named column — the other group's cell
    "quote_row_only",
    #: both groups resolve to the same plotted marker — one series was read twice, so one group's
    #: number is the other's
    "series_identity_conflict",
    #: each group's value was measured on the OTHER group's marker: the effect's sign is inverted
    "series_transposed",
    #: the readers answered off different value axes and one cluster was kept — a value off the
    #: wrong ladder is wrong by a factor, and keeping the majority does not prove it was right
    "axis_conflict",
    #: the reader that made the only checkable claim about this measure says the opposite of what
    #: this cell's resolved means say. It is the one contradiction the orientation stage can
    #: measure, and it is about the NUMBERS (which group is higher), not about how thin the
    #: agreement on the direction was — so it belongs here rather than beside its three
    #: `orientation_*` neighbours in `CAPPING_FLAGS` (fix round F5).
    "orientation_reader_contradicts_values",
    #: the readings that agreed came from different places in one figure. A figure tolerance is
    #: wide enough to span two panels, so their agreement is a coincidence of scale rather than
    #: corroboration, and their middle is a number neither panel contains (D2).
    "locator_reads_conflict",
    #: every reading for this group was taken off a panel the caption gives to another group, and
    #: this group has none of its own — so nothing could be set aside, and the value that stands
    #: may be the other group's
    "locator_panel_mismatch",
})

#: A code belongs to exactly one of the two, and every one of them tells the reviewer why it
#: fired: a code in neither is scored as an ordinary warning by accident, and a code in both would
#: be deducted twice and floored inconsistently.
WITHHOLDING_FLAGS: frozenset[str] = CAPPING_FLAGS | CONTRADICTING_FLAGS
assert CAPPING_FLAGS.isdisjoint(CONTRADICTING_FLAGS), \
    "a doubt is either under-corroboration or contradiction, never both"
assert all((code in CAPPING_FLAGS) ^ (code in CONTRADICTING_FLAGS) for code in WITHHOLDING_FLAGS)

#: every cap is at or above `ACCEPT_WITH_NOTE`, so no COMBINATION of caps can push a cell that
#: scored well enough on the evidence down into `needs_human` (controller ruling R2, task 16):
#: only disagreement, a refutation, a contradiction, an unresolved direction or an `error` flag
#: does that.
_CAPS = (SINGLE_ROUTE_CAP, ADJUDICATED_CAP)
assert all(cap >= ACCEPT_WITH_NOTE for cap in _CAPS), "a cap must never mean needs_human"

#: How much a figure's axis costs when it is not fully corroborated — an ORDERED ladder, scored
#: once per cell, and deliberately kept out of the generic warn arithmetic below.
#:
#: The first cut of this got it backwards twice over. `run_checks` raises the calibration flag on
#: every candidate of a cell, the warn penalty counted FLAGS rather than codes, and the capping
#: codes were `warn` — so six ensemble candidates carrying one `calibration_single_witness` spent
#: the entire warn budget (-0.24) on one problem and dropped a clean figure cell from 0.60 to
#: 0.36, i.e. `needs_human`. That is precisely the failure task 16 exists to remove: Cressman's
#: late adaptation, with the TRUE ladder recovered, would have gone to a human anyway. Worse, the
#: order inverted — `cal_refuted` (ladder known wrong, non-forcing, no penalty) outscored
#: `single_witness` (ladder probably right).
#:
#: So the axis is scored here, once, on a ladder that is monotone by construction:
#:     confirmed (nothing) > single_witness > refuted
#: and the deduction is floored at `ACCEPT_WITH_NOTE`, because an uncorroborated axis is a reason
#: to have a human LOOK at a cell, never a reason to withhold it on its own (R2).
CALIBRATION_PENALTY: dict[str, float] = {
    "calibration_single_witness": 0.08,
    "calibration_refuted": 0.16,
    #: no ladder was built at all. It shares the bottom rung with a ladder the readers disproved,
    #: and for the same reason: in both, no calibration is left standing and the value rests on
    #: the read-outs alone. "We could not establish the scale" must never cost less than "we
    #: established it with one witness".
    "calibration_missing": 0.16,
    "calibration_two_point": 0.16,
}

#: `error`-severity codes that do NOT force a human on their own. `calibration_refuted` is an
#: error against the CALIBRATION, not against the value: it is only ever raised when two or more
#: read-outs agreed on a number the ladder cannot draw, so the number has two witnesses and the
#: ladder has none. It caps the cell (above) instead of convicting it. When the read-outs did not
#: agree the checks raise `calibration_disputed`, which is not on this list.
NON_FORCING_ERRORS: frozenset[str] = frozenset({"calibration_refuted"})

#: why each flag above holds a cell back — a reviewer reads these lines, so each one names the
#: actual doubt, and names the comparison that was really made rather than the one it would be
#: nice to have made
#: row-policy flags for a value built on a premise THE TOOL INFERRED from the paper (never a
#: field prior): the paper supplied the evidence, a person has not confirmed the reading of it.
#: These are ROW flags like `N_FROM_MAP` — absent from `CHECK_SEVERITY` (no verdict writes them),
#: absent from `CAPPING_FLAGS` (whose floor is accept_with_note, the wrong semantics: an inferred
#: row must be needs_human unconditionally, which `resolve._finish` enforces), and their rows are
#: admitted to the BEST-GUESS line under their own named rule, with the question still open.
SPREAD_TYPE_HOUSE_STYLE = "spread_type_inferred_house_style"
#: what a categorical x axis IS, resolved from ONE reading's category report (or one in tension
#: with the protocol's own measurement window). Unlike its `..._resolved_from_readings` sibling
#: (two or more readings agree — a capping flag), a single witness to the axis's identity is a
#: premise the tool inferred, so the row is held for a person and admitted to the best-guess
#: line under the `inferred_premise` rule with its question open. This one IS check-emitted
#: (`categorical_x_single_witness` in `checks.CHECK_SEVERITY`), unlike the house-style flag,
#: which only the row funnel stamps — membership here is about the fence, not the origin.
CATEGORICAL_X_SINGLE_WITNESS = "categorical_x_single_witness"
INFERRED_PREMISE_FLAGS: frozenset[str] = frozenset({SPREAD_TYPE_HOUSE_STYLE,
                                                    CATEGORICAL_X_SINGLE_WITNESS})

CAP_REASONS: dict[str, str] = {
    "crop_reacquired": ("a majority of readers refused the panel crop — the named target was "
                        "not in that image — so the reading was re-acquired from the full page "
                        "render, which is wider than any panel; a reviewer should see the "
                        "figure"),
    "panel_not_isolated": ("the named panel could not be isolated from its neighbours; the "
                           "reading was made on the whole figure and is capped below automatic "
                           "acceptance"),
    "quote_row_only": ("the numbers are somewhere in the named table row but not in the column "
                       "this reading claims, so they may be the other group's"),
    "calibration_single_witness": ("only one witness calibrated this figure's axis, so the scale "
                                   "the value was read against is uncorroborated"),
    "calibration_refuted": ("the readers agree on a value the tick ladder cannot draw, so the "
                            "axis calibration was discarded and only the read-outs stand"),
    "calibration_missing": ("no calibration of this figure's value axis could be built at all, "
                            "so the scale these numbers were read against rests entirely on the "
                            "readers and nothing checked it"),
    "calibration_two_point": ("the axis calibration kept exactly two ticks — they define the "
                              "scale exactly, and nothing checks that the axis is linear between "
                              "them"),
    "collapsed_across_x": ("this is an average across a categorical axis and its dispersion is an "
                           "approximation, not the paper's own"),
    "categorical_point_read": ("this is the single plotted point at the x category the locator "
                               "names rather than an average across the axis, so there is one "
                               "point per group and nothing else on the axis corroborates it"),
    "categorical_x_resolved_from_readings": ("what this figure's categorical x axis is — "
                                             "conditions or the groups themselves — was resolved "
                                             "from the readers' own category reports rather than "
                                             "stated by the protocol or the mapper; two or more "
                                             "readings support the ruling, and a reviewer should "
                                             "see the figure"),
    "reopened_on_better_source": ("extraction was re-opened on a source the verifier named, so "
                                  "the location itself was decided by a model"),
    "series_identity_conflict": ("both groups resolve to the same plotted marker, so this number "
                                 "may belong to the other series"),
    "series_marker_mismatch": ("the reader's description of this series and the markers the "
                               "pixel pass found do not line up, so which series this number was "
                               "measured on rests on the reader's words alone"),
    "series_transposed": ("each group's described marker is the one found where the OTHER "
                          "group's value was measured, so the two series are swapped and the "
                          "sign of the effect is inverted"),
    "dispersion_type_from_legend": ("the error-bar type was read off the figure's legend, not "
                                    "determined by the map, so nothing independent confirms it"),
    "axis_conflict": ("the readers answered off different value axes and only one of them was "
                      "pooled, so which ladder this number is on rests on a majority"),
    "orientation_single_witness": ("the direction of this measure was set by one ballot, because "
                                   "the only other reader did not produce a usable one — so the "
                                   "sign of this effect rests on a witness nothing corroborated"),
    "orientation_by_majority": ("the direction of this measure was settled by a majority of "
                                "three readers rather than by two that agreed, so one reader "
                                "read the paper the other way and was outvoted"),
    "orientation_direction_conflict": ("the readers read the paper's own sentence about which "
                                       "group came out higher in opposite directions, so nothing "
                                       "the paper states is left for the sign check to compare "
                                       "the extracted numbers with"),
    "locator_reads_conflict": ("the readings that agreed were taken at different places in the "
                               "same figure, and a figure tolerance is wide enough to span two "
                               "panels — so their agreement is a coincidence of scale and their "
                               "middle is a value neither place contains"),
    "locator_panel_mismatch": ("every reading for this group was taken off a panel the caption "
                               "gives to another group, and this group has no reading from its "
                               "own panel, so this number may be the other group's"),
    "locator_reads_set_aside": ("a reading taken off another group's panel was set aside because "
                                "the caption says which panel is whose; the value that stands "
                                "came from this group's own panel, and a reader went to the "
                                "wrong one"),
    "orientation_reader_contradicts_values": ("a reader states which group came out higher on "
                                              "this measure and this cell's resolved means say "
                                              "the opposite, so the reader that made the one "
                                              "checkable claim about these numbers was discarded "
                                              "— and the numbers themselves are in question"),
    "n_before_exclusions": ("the group size behind this cell is a number the paper prints before "
                            "the exclusions it then reports, so the arm that was ANALYSED may be "
                            "smaller than the denominator this row was divided by"),
}


# ----------------------------------------------------------------------------- amendment F gate
def _as_sd(value: float | None, kind: DispersionType, n: int | None) -> float | None:
    """A spread as a standard deviation, when it is one or converts trivially."""
    if value is None or value <= 0:
        return None
    if kind is DispersionType.SD:
        return value
    if kind is DispersionType.SE and n:
        return value * math.sqrt(n)
    return None


def _sd_of(cand: Candidate) -> float | None:
    """The candidate's spread as a standard deviation, when it is one or converts trivially."""
    return _as_sd(cand.dispersion_value, cand.dispersion_type, cand.n)


class _Reading(NamedTuple):
    """One digitizer path's answer for one group, in data units."""

    mean: float
    sd: float
    sigma: float | None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _recorded_paths(cand: Candidate) -> dict[str, list[_Reading]] | None:
    """The digitizer paths recorded INSIDE one ensemble candidate, or `None` when it records none.

    `run.py:vote_candidates` admits exactly one `digitize:ensemble` candidate per group — the
    digitiser's four or five ways of measuring one picture must not outvote a printed value — so
    the several routes amendment F wants to compare never arrive here as candidates. They arrive
    as `pixel_provenance["per_route"]`, which is where they are read from. A route's `error` is
    the error-bar half-length in the ensemble's own dispersion type, because that is what
    `digitize._build_candidates` builds each per-route candidate with.

    A dropped reading is not a witness: the ensemble threw it away, and counting it here would
    let a value that lost an axis conflict corroborate the one that won it.
    """
    rows = (cand.pixel_provenance or {}).get("per_route")
    if not isinstance(rows, (list, tuple)):
        return None
    out: dict[str, list[_Reading]] = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("dropped") or row.get("status", "found") != "found":
            continue
        mean = _number(row.get("mean"))
        sd = _as_sd(_number(row.get("error")), cand.dispersion_type, cand.n)
        if mean is None or sd is None:
            continue
        out.setdefault(digitizer_path(str(row.get("extractor_id") or "")), []).append(
            _Reading(mean, sd, _number(row.get("sigma")) or cand.sigma))
    return out


def _per_route(candidates: Iterable[Candidate], group: str
               ) -> tuple[dict[str, list[_Reading]], bool]:
    """`(readings by digitizer path, whether any candidate recorded its paths at all)`.

    An ensemble that records its paths is REPLACED by them — counting the median of a set of
    routes as a further route would let one measured path plus its own summary look like two
    paths agreeing perfectly.
    """
    out: dict[str, list[_Reading]] = {}
    recorded = False
    for cand in candidates:
        if not (cand.group == group and cand.kind == "group_stats" and cand.status == "found"):
            continue
        paths = _recorded_paths(cand)
        if paths is not None:
            recorded = True
            for name, readings in paths.items():
                out.setdefault(name, []).extend(readings)
            continue
        sd = _sd_of(cand)
        if cand.mean is not None and sd is not None:
            out.setdefault(modality(cand), []).append(_Reading(cand.mean, sd, cand.sigma))
    return out, recorded


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values))


def figure_gate(candidates: Sequence[Candidate], n_a: int | None, n_b: int | None
                ) -> tuple[bool, float | None, float | None, list[str]]:
    """Amendment F: may a digitised cell be accepted without a human?

    Returns `(ok, delta_d, digitisation_se_share, reasons)`. `delta_d` is the spread of the effect
    size implied by the digitizer routes that read BOTH groups — the quantity the review actually
    pools, so two routes that differ on a mean but imply the same effect are not a problem. The
    share is the delta-method digitisation standard error over the sampling standard error.
    """
    reasons: list[str] = []
    if not n_a or not n_b:
        return False, None, None, ["the analysed group sizes are unknown, so the digitisation "
                                   "uncertainty cannot be compared with the sampling error"]
    (routes_a, recorded_a), (routes_b, recorded_b) = _per_route(candidates, "A"), \
        _per_route(candidates, "B")
    shared = sorted(set(routes_a) & set(routes_b))
    effects: dict[str, float] = {}
    for route in shared:
        a, b = routes_a[route], routes_b[route]
        mean_a, sd_a = _median([r.mean for r in a]), _median([r.sd for r in a])
        mean_b, sd_b = _median([r.mean for r in b]), _median([r.sd for r in b])
        try:
            effects[route] = cohens_d(mean_a, sd_a, n_a, mean_b, sd_b, n_b)
        except ValueError:                                # pragma: no cover - guarded by _as_sd
            continue

    delta = (max(effects.values()) - min(effects.values())) if len(effects) >= 2 else None
    paths = set(routes_a) | set(routes_b)
    # An absence of evidence must not be printed as though it were a finding about the reading:
    # "nothing here says how this picture was measured" and "the ways it was measured do not
    # agree" are opposite states, and the reviewer is the one who has to tell them apart.
    if delta is None and not paths and not (recorded_a or recorded_b):
        reasons.append("this cell records no per-route digitisation detail, so the digitisation "
                       "gate could not be evaluated either way")
    elif delta is None and not paths:
        reasons.append("no digitizer path recorded both a value and a usable spread, so the "
                       "digitisation gate could not be evaluated either way")
    elif delta is None:
        reasons.append(f"{len(effects)} of the digitizer's {len(paths)} measurement path(s) read "
                       f"both groups with a usable spread, so the effect this figure implies is "
                       f"corroborated by nothing — one path cannot agree with another")
    elif delta >= DELTA_D_LIMIT:
        reasons.append(f"the digitizer's measurement paths imply effects that differ by "
                       f"{delta:.3f} (limit {DELTA_D_LIMIT}), so how this picture was measured "
                       f"changes the answer")

    share = None
    if effects:
        d = _median(list(effects.values()))
        sds_a = [r.sd for row in routes_a.values() for r in row]
        sds_b = [r.sd for row in routes_b.values() for r in row]
        sigma_a = [r.sigma for row in routes_a.values() for r in row if r.sigma is not None]
        sigma_b = [r.sigma for row in routes_b.values() for r in row if r.sigma is not None]
        if sds_a and sds_b and (sigma_a or sigma_b):
            sp = pooled_sd(_median(sds_a), n_a, _median(sds_b), n_b)
            se_dig = math.sqrt(((_median(sigma_a) if sigma_a else 0.0) / sp) ** 2
                               + ((_median(sigma_b) if sigma_b else 0.0) / sp) ** 2)
            share = se_dig / se_smd(d, n_a, n_b)
            if share >= DIGITIZATION_SE_SHARE:
                reasons.append(f"the digitisation uncertainty is {share:.0%} of the sampling "
                               f"error (limit {DIGITIZATION_SE_SHARE:.0%})")
        else:
            reasons.append("the digitizer recorded no per-value uncertainty, so its share of the "
                           "sampling error is unknown")
    ok = (delta is not None and delta < DELTA_D_LIMIT
          and share is not None and share < DIGITIZATION_SE_SHARE)
    return ok, delta, share, reasons


# ------------------------------------------------------- C8 / C11 / C9: three public rulings
def verifier_state(verdict: VerifierVerdict) -> str:
    """What a verifier record IS — which is not always what its `verdict` field says (C8).

    A verdict is recorded only when an agent produced one. When the call raised `TruncatedOutput`
    or `LLMError`, `pipeline/run.py` writes a `VerifierVerdict` so the failure is on the record
    and the warning reaches the manifest — but that record names no model, no prompt version and
    no call id, because there was no call. Nothing else in the pipeline can produce a verdict with
    all three empty: every real one carries them (verified against all 14 verdicts in
    `runs/rerun-fixed`, where exactly the two truncated Heuer records are bare).

    So the structural fact is read here rather than the label, and the label wins when it is
    already one of the honest ones — which is what lets the extractor and `run.py` start writing
    `not_run` / `no_value_printed` directly without this function changing again.
    """
    if verdict.verdict in NO_EVIDENCE_VERDICTS:
        return verdict.verdict
    if not (verdict.llm_call_id or verdict.model or verdict.prompt_version):
        return NOT_RUN
    return verdict.verdict


def confidence_margin(score: float) -> tuple[float, str, bool]:
    """`(margin, nearest_boundary, decided_by_a_hair)` for one score (C11).

    `margin` is the UNSIGNED distance to the nearest bucket boundary and `nearest_boundary` names
    it, so "0.4500, pooled by 0.0000" and "0.7400, held back by 0.0100" are both sayable. Ties go
    to the lower boundary, which is the one such a cell cleared.
    """
    nearest, margin = min(BUCKET_BOUNDARIES.items(), key=lambda kv: (abs(score - kv[1]), kv[1]))
    distance = round(abs(score - margin), 4)
    return distance, nearest, distance <= MARGIN_BAND


#: what `confidence` says instead of a margin when the score decided nothing (C11 / review M3)
BUCKET_NOT_SCORED = ("this cell was held by the evidence, not by its score, so the score is not "
                     "compared with any boundary: the bucket was not decided by the score")


class _Scored(tuple):
    """`(bucket, score, reasons)` — plus WHO decided the bucket.

    Every caller unpacks the three-tuple `confidence` has always returned, and this is one. The
    fourth fact rides alongside because `resolve_cell` cannot recover it: a `needs_human` cell may
    have been forced by an error at a score of 1.0 or sent there by a score of 0.20, and only the
    second of those has a margin worth publishing.
    """

    forced_human: bool

    def __new__(cls, bucket: ConfidenceBucket, score: float, reasons: list[str], *,
                forced_human: bool) -> "_Scored":
        self = super().__new__(cls, (bucket, score, reasons))
        self.forced_human = forced_human
        return self


def _margin_reason(score: float) -> str:
    distance, nearest, hair = confidence_margin(score)
    cleared = score >= BUCKET_BOUNDARIES[nearest]
    side = "clears" if cleared else "misses"
    line = (f"margin {distance:.4f} {side} {nearest} ({BUCKET_BOUNDARIES[nearest]:.2f})")
    if not hair:
        return line
    return (f"{DECIDED_BY_A_HAIR}: {line} — this cell is inside {MARGIN_BAND:.2f} of the line "
            f"that decided it, so which side it landed on is not a finding about the paper")


def conversion_gate_bucket(bucket: ConfidenceBucket, route: str, flags: Sequence[str]
                           ) -> tuple[ConfidenceBucket, list[str]]:
    """The bucket a row keeps once the conversion gate's own flags are priced (C9).

    Called by whoever resolves a ROW (`canopy.pipeline.resolve`), because the gate's flags are
    raised while the effect size is being built, after this module has scored the two cells. It is
    a cap on the BUCKET and not on the score on purpose: "below `auto_accept`" still includes
    `accept_with_note`, which pools, so a score cap could never withhold the row it exists to
    withhold. Nothing whose provenance to these two groups is unverified may pool.
    """
    if route not in CONVERTED_ROUTES:
        return bucket, []
    codes = list(flags)
    unverified = sorted(set(codes) & UNVERIFIED_CONTRAST_FLAGS)
    shortfall = sorted(c for c in codes if c.startswith(DF_SHORTFALL_PREFIX))
    if unverified:
        return "needs_human", [
            f"{', '.join(unverified)}: this row was converted from a printed test statistic and "
            f"nothing establishes that the statistic is the comparison of THESE two groups — a "
            f"post-hoc from a three-group analysis reports the same t. A human decides"]
    if shortfall and bucket == "auto_accept":
        return "accept_with_note", [
            f"{', '.join(shortfall)}: the printed degrees of freedom do not equal n_a + n_b - 2 "
            f"and the shortfall is explained, so the conversion is allowed but never automatic"]
    return bucket, []


#: ------------------------------------------- the denominator's own provenance, weighed as a pair
#: The gate above asks whether the row's NUMERATOR is the contrast it claims to be. This one asks
#: whether anything on the record establishes the DENOMINATOR it was divided by, and it is keyed on
#: two codes `canopy.verify.checks` already raises:
#:
#: `dispersion_unknown` — a spread was read and nobody could say what kind it is. An SD and an SE
#:   are printed identically ("41.5 ± 3.1"), and reading one as the other scales the effect size by
#:   √n: the same pair at n = 19 is |d| = 0.31 or |d| = 1.35 depending on which it was.
#: `n_missing` — no group size was transcribed beside the value, so the size the conversion scaled
#:   by came from the map's reading of the methods rather than from the number's own neighbourhood.
#:
#: Either one ALONE is an ordinary warning, priced by the score and floored like every other, and
#: that is right: a typed SD beside a guessed n has a magnitude that is settled and a variance that
#: is approximate, and an untyped ± beside a printed n can be checked against a reported statistic
#: or against the paper's own interval. Both were left poolable on purpose — holding on either
#: alone withholds readings that agree with a hand-built reference analysis, which is the failure
#: the whole calibration exists to avoid.
#:
#: TOGETHER they are not two warnings. They are the same doubt arriving twice, and it is the doubt
#: that decides the row's magnitude: the only number on the record that could tell an SD from an SE
#: is the n, and the n was guessed too. Nothing says what |d| this row has — so, like the gate
#: above, it is a cap on the BUCKET rather than on the score ("below auto_accept" still includes
#: `accept_with_note`, which pools), and a human decides.
#:
#: It deliberately adds NO code of its own. Both codes are already on the row, put there by the
#: cells that raised them, so a `ROW_REFUSAL_CODES` entry would be a third name for a doubt the
#: record already carries — and a row refusal vetoes the best-guess line (`bestguess.VETO_ROW_FLAGS`)
#: and the release paths treat it as unanswerable. This row's value may well be the best estimate
#: anyone has of the contrast; what is unverified is its SCALE. So the strict line may not take it
#: and the best-guess line still may, under `low_confidence_value` — held, not refused.
#:
#: And it is a CONJUNCTION rather than a set with a threshold, so the hold is answerable — see the
#: function, which says which answer, on which arm, because only one of the two codes has one.
#:
#: **Read on ONE arm, never across the pair** (adversarial review, fix round). The row's `flags`
#: are the UNION of both cells' codes (`resolve.ResolvedValues.from_verdicts`), so a row-level
#: reading of this conjunction fired whenever the two doubts sat on DIFFERENT arms — an arm whose
#: untyped ± sits beside a printed n (so the spread is checkable against it) paired with an arm
#: whose spread is typed and only its n is guessed. Neither of those arms is "the same doubt
#: arriving twice", which is the entire justification above: the reason is a statement about ONE
#: number, so the rule has to be evaluated on one number.
UNVERIFIED_VARIANCE_FLAGS: frozenset[str] = frozenset({"dispersion_unknown", "n_missing"})

#: …and the codes are real ones. A bare `assert` is stripped under `python -O`, which is exactly
#: when a gate keyed on a typo would go quiet, so `severity_of` is called for its own KeyError.
for _code in sorted(UNVERIFIED_VARIANCE_FLAGS):
    severity_of(_code)
del _code


def unverified_variance_bucket(bucket: ConfidenceBucket,
                               arm_flags: Mapping[str, Sequence[str]]
                               ) -> tuple[ConfidenceBucket, list[str]]:
    """The bucket a row keeps once the provenance of its DENOMINATOR has been weighed.

    `arm_flags` is the codes each independently-read number carried, keyed by the arm it was read
    for (`{"A": [...], "B": [...]}` for a resolved row; `{"<member>|A": [...], ...}` for a
    composite). It is a MAPPING and not the row's flat flag list on purpose: unioning arm sets can
    never manufacture a conjunction, while unioning code sets — which is what the row's own `flags`
    are, and what `aggregate.composite_row` does again a level up — silently can.

    Called by whoever builds a row that can pool: `canopy.pipeline.resolve._finish` beside the
    conversion gate and C9's `|d|` screen, and `canopy.pipeline.aggregate.composite_row`, which
    builds an `EffectSizeRecord` without passing through the resolver at all. Every rebuild an
    answer triggers goes back through `_finish`, so this re-derives from the codes the cells then
    carry rather than from anything a reviewer stamped.

    Route-independent on purpose. Whatever the row converted through, `dispersion_unknown` and
    `n_missing` are statements about the numbers the cells resolved, and no route makes an untyped
    spread beside a guessed n into a verified variance.
    """
    held = [arm for arm in sorted(arm_flags)
            if UNVERIFIED_VARIANCE_FLAGS <= set(arm_flags[arm] or ())]
    if not held:
        return bucket, []
    named = ", ".join(sorted(UNVERIFIED_VARIANCE_FLAGS))
    where = ", ".join(f"group {arm}" for arm in held)
    return "needs_human", [
        f"{named} on {where}: that arm's spread was never typed and its group size was never "
        f"transcribed, so nothing on the record establishes the denominator the effect size was "
        f"divided by. An SD read as an SE (or the reverse) changes |d| by a factor of √n, and the "
        f"n that would have told them apart is the number that is missing — so the magnitude is "
        f"unverified rather than merely uncorroborated. The answer asked for is the group size "
        f"for {where}: nothing a reviewer can type retires {sorted(UNVERIFIED_VARIANCE_FLAGS)[0]} "
        f"(it is in no `overrides.VALUE_CLEARS_*` family), so the n is the half of the pair that "
        f"can be broken. A human decides"]


#: the code the ROW carries when its resolved |d| fails the plausibility screen. Same code the
#: cell-level early warning uses, because it is the same doubt about the same denominator — a
#: reviewer who has learned what it means at one level has learned it at both.
IMPLAUSIBLE_DISPERSION = "implausible_dispersion"

#: EVERY code the resolver can put on a ROW that neither cell carries — the findings that are
#: about the number the CONVERSION produced, so they cannot exist until both cells are in.
#:
#: This is a contract with the review layer, which is why it is one constant and not two lists.
#: A row refusal is exactly the case where both cells read as fine, so every rule that asks "is
#: this cell releasable?" has to consult the row first: `canopy.pipeline.overrides` imports this
#: set to decide a rebuilt bucket, to refuse an answer that would release such a cell, and to
#: keep BOTH of the row's cells in the review queue; `canopy.pipeline.run` uses it for the same
#: queue rule on a fresh run. A second copy of the membership in the review layer is a set kept
#: in sync by hand, and the first code that was added to one and not the other would release
#: cells under a refused row again (questions area, fix round 3, concern 1).
#:
#: `resolve._add_row_refusal` is the only writer and it refuses a code that is not here, so the
#: set cannot fall behind the resolver — `tests/test_resolve.py` pins that both ways.
ROW_REFUSAL_CODES: frozenset[str] = frozenset({IMPLAUSIBLE_DISPERSION})


def dispersion_plausibility_bucket(bucket: ConfidenceBucket, d: float | None, *,
                                   denominator: str = "", route: str = ""
                                   ) -> tuple[ConfidenceBucket, list[str]]:
    """The bucket a row keeps once its RESOLVED `|d|` has been screened (C9, second half).

    DECISION-v2 §C9 specifies this on the **resolved** `|d|`, "separately and independently of
    route", and that is the only place it can be honest: the number that reaches the plot is post
    vote, post adjudication and post unit conversion, and none of those three existed when
    `canopy.verify.checks` screened the raw candidates. Screening candidates instead missed
    (review H1, each executed): an SE-typed pair whose SE→SD conversion implies |d| = 17.3 — SE is
    the MODAL shape in the live corpus, 32 against 25 SD of the 57 `found` group_stats
    candidates that carry either — a group whose
    first SD-typed candidate was sane while the two agreeing readers were not, and a dispersion
    the adjudicator itself supplied. All three pooled, one of them with both cells at
    `auto_accept`.

    Two means eight standard deviations apart are almost never two means eight standard deviations
    apart: they are two means divided by a standard error, a within-subject error bar, a range, or
    the other group's spread. The numerator is usually fine, so the arithmetic looks plausible all
    the way down and only the size of the answer gives it away — which is why the cell-level check
    remains as an early warning (it is the cheapest place to SAY it) and this one binds.

    `denominator` is what the caller actually divided by, in the caller's own words, so the
    reviewer reads the two dispersions the conversion used rather than the ones the paper printed.
    """
    if d is None or not math.isfinite(d) or abs(d) <= MAX_PLAUSIBLE_D:
        return bucket, []
    names = denominator or (f"the row was converted through {route!r}, which has no dispersion of "
                            f"its own" if route else "")
    return "needs_human", [
        f"{IMPLAUSIBLE_DISPERSION}: the resolved values imply |d| = {abs(d):.2f}, above the "
        f"plausibility threshold of {MAX_PLAUSIBLE_D:g}. The suspect number is the denominator"
        + (f" — {names}" if names else "") +
        f". An SE printed as an SD, a within-subject error bar, a range read as a spread or the "
        f"other group's dispersion all look exactly like this, and the screen is on the value "
        f"that reaches the plot rather than on any one reading of it. A human decides"]


# ----------------------------------------------------------------------------- confidence
def _grounding(vote_result: VoteResult, candidates: Sequence[Candidate] | None,
               settled: Adjudication | None) -> tuple[bool | None, bool]:
    """`(grounded, any_short_quote)` for the readers whose numbers were actually kept.

    Normally that is the vote's agreeing set, which the vote already summarised. When an
    adjudicator settled the cell the vote has no agreeing set, so the candidates it chose are read
    from the ruling instead.
    """
    chosen: set[str] = set()
    if settled is not None:
        chosen = set(settled.chosen_candidate_ids)
        chosen |= {cid for group in settled.groups for cid in group.chosen_candidate_ids}
    if chosen and candidates:
        quoted = [c for c in candidates if c.candidate_id in chosen and c.quote.strip()]
        if quoted:
            return (all(c.grounded is not False for c in quoted),
                    any(is_short_quote(c.quote) for c in quoted))
    if settled is not None:
        # a ruling that quoted the paper itself: `ground_adjudication` already checked it, and a
        # failed check set `needs_human`, so anything still here is grounded evidence
        ruled = [g for g in settled.groups if g.quote.strip() and g.grounded is not None]
        if ruled:
            return (all(g.grounded for g in ruled), any(is_short_quote(g.quote) for g in ruled))
    return vote_result.grounded, vote_result.short_quote


def _agreeing_routes(result: VoteResult) -> list:
    agreeing = set(result.agreeing_ids)
    return [r for r in result.routes if agreeing & set(r.candidate_ids)]


def _agreeing_model_families(result: VoteResult,
                             candidates: Sequence[Candidate] | None) -> list[str]:
    """The model families recorded INSIDE the candidates the vote kept.

    `route_key` is one modality and one model family per candidate, and the digitiser's ensemble
    is one candidate however many models read the figure — so a figure that two independent model
    families read and agreed on arrived at the vote looking like a single route (F2). The
    digitiser records the families it actually ran in `pixel_provenance["model_families"]`; this
    is where they are cashed in.
    """
    agreeing = set(result.agreeing_ids)
    families: set[str] = set()
    for cand in candidates or []:
        if cand.candidate_id not in agreeing:
            continue
        names = (cand.pixel_provenance or {}).get("model_families")
        if isinstance(names, (list, tuple)):
            families |= {str(name) for name in names if name}
    return sorted(families)


#: C10: which half of a reading is missing. The distinct reason below is keyed on THIS value and
#: never on "there is a partial read at all" — a scatter of individual subjects has no whisker to
#: read, and five routes honestly reporting no error bar (Cressman d1 aftereffect, a pooled cell)
#: are an absent feature of the picture, not a reader that came back empty-handed.
PARTIAL_READ_MISSING_MEAN = "mean"


def partial_mean_readers(group: str | None,
                         candidates: Sequence[Candidate] | None) -> list[str]:
    """Readers that read this figure for THIS group and returned a spread but no mean (C10).

    Such a reader is not a family that never looked: it looked, it answered, and its answer had a
    hole in it. It is still not a witness to the mean — `_agreeing_model_families` counts only
    readers that produced the quantity being corroborated, and that rule is what stops a cell
    pooling on one reader — but the cell must not print the sentence a genuinely single-family
    cell prints, because the two call for different actions (re-ask this reader for this one
    number, versus find a second reader at all).
    """
    out: list[str] = []
    for cand in candidates or []:
        entries = (cand.pixel_provenance or {}).get("partial_read") or []
        for entry in entries if isinstance(entries, (list, tuple)) else []:
            if not isinstance(entry, dict):
                continue
            if entry.get("group") != group:
                continue
            if entry.get("missing") != PARTIAL_READ_MISSING_MEAN:
                continue
            name = str(entry.get("model") or entry.get("route") or "").strip()
            if name and name not in out:
                out.append(name)
    return out


def _witness_families(result: VoteResult, candidates: Sequence[Candidate] | None) -> set[str]:
    """Every model family behind the agreeing readings — from the route keys and from inside them.

    A digitiser ensemble is one candidate however many models read the figure, so the families it
    ran are recorded in its provenance; a text or table route carries its family in its route key.
    An empty family is "not stated", never a second one.
    """
    # index 1, not "everything after the first slash": a route key is `modality/family` and the
    # modality of a digitised reading has a colon in it, never a slash — and one model reading two
    # panels is one family, which is why the PLACE is not part of the key (fix round 2, finding 7)
    families = {parts[1] for parts in (r.route_key.split("/")
                                       for r in _agreeing_routes(result)) if len(parts) > 1}
    families |= set(_agreeing_model_families(result, candidates))
    return {name for name in families if name}


def _witness_locations(result: VoteResult, candidates: Sequence[Candidate] | None) -> set[str]:
    """The distinct places in the document the agreeing readings came from.

    A quote, or the locator when a value was measured rather than transcribed. NOT the page, and
    NOT `source_kind`: both are fields the reading model filled in itself, and a self-declared
    label is not a second place in the paper. Anything that states no location at all is not a
    location.
    """
    agreeing = set(result.agreeing_ids)
    places = {" ".join((c.quote or c.locator or "").split()).casefold()
              for c in candidates or [] if c.candidate_id in agreeing}
    return {place for place in places if place}


def _base_kind(routes: Sequence) -> str:
    kinds = {r.route_key.split("/", 1)[0] for r in routes}
    if any(k == "text" for k in kinds):
        return "text"
    if any(k == "table" for k in kinds):
        return "table"
    if any(k.startswith("figure") for k in kinds):
        return "figure"
    if any(k == "statistic" for k in kinds):
        return "statistic"
    return "text"


def confidence(vote_result: VoteResult, verdicts: Sequence[VerifierVerdict] = (),
               flags: Sequence[CheckFlag] = (), adjudication: Adjudication | None = None, *,
               candidates: Sequence[Candidate] | None = None, n_a: int | None = None,
               n_b: int | None = None, orientation: OrientationVerdict | None = None
               ) -> tuple[ConfidenceBucket, float, list[str]]:
    """`(bucket, score, reasons)` for one resolved cell. Pure code; nothing here calls a model."""
    reasons: list[str] = []
    forced_human = False
    settled = adjudication is not None and not adjudication.needs_human

    adjudicated_value = settled and any(g.mean is not None for g in adjudication.groups)
    if vote_result.agreement == "none" and not adjudicated_value:
        # `_Scored`, not a bare tuple: this cell's bucket was decided by the EVIDENCE (there is
        # none), not by the score, so `resolve_cell` must publish no margin and no nearest
        # boundary for it. A plain tuple made `forced_human` read False, and every "no value was
        # resolved" cell in the queue carried "0.45 from accept_with_note" — a claim about a
        # decision the score never made (whole-diff L2, the M3-margin ruling's own rule).
        return _Scored("needs_human", 0.0, ["no value was resolved for this cell"],
                       forced_human=True)

    routes = _agreeing_routes(vote_result)
    score = BASE_SCORE[_base_kind(routes)]
    reasons.append(f"source kind {_base_kind(routes)} (+{BASE_SCORE[_base_kind(routes)]:.2f})")

    # Two readings are two WITNESSES when something outside the reading model separates them: a
    # second model family (a different failure mode) or a second place in the document (the paper
    # printed the value twice). `route_key` alone does not establish that, because half of it is
    # `source_kind` — a label the reading model wrote about its own source. One model reporting
    # one sentence as `text` and again as `table` was two routes, +0.25, no cap, and a
    # maximum-score automatic acceptance off one model reading one sentence.
    families = _witness_families(vote_result, candidates)
    places = _witness_locations(vote_result, candidates)
    independent = len(families) >= 2 or len(places) >= 2
    one_witness = ""

    # --- agreement
    if settled:
        reasons.append("the adjudicator settled a cell the vote could not, so the vote neither "
                       "credits nor penalises it")
    elif vote_result.agreement == "agree" and independent:
        extra = min(EXTRA_ROUTE_CAP, EXTRA_ROUTE_BONUS * max(0, len(routes) - 2))
        score += AGREE_BONUS + extra
        reasons.append(f"{len(routes)} independent routes agree within {vote_result.tolerance:.4g} "
                       f"(+{AGREE_BONUS + extra:.2f})")
    elif vote_result.agreement == "agree":
        one_witness = (f"{len(routes)} routes agree, but they are one witness: one model family "
                       f"({', '.join(sorted(families)) or 'unnamed'}) reading one place in the "
                       f"paper. Two prompts of one model share a failure mode, and how a reading "
                       f"labels its own source is not a second place — so this is not agreement "
                       f"and cannot be accepted by vote")
        reasons.append(one_witness)
    elif vote_result.agreement == "single":
        heterogeneous = _agreeing_model_families(vote_result, candidates)
        if len(heterogeneous) >= 2:
            score += AGREE_BONUS
            reasons.append(f"one route, but {len(heterogeneous)} independent model families read "
                           f"it and agreed ({', '.join(heterogeneous)}) — a second family is a "
                           f"different failure mode, which is what agreement is for "
                           f"(+{AGREE_BONUS:.2f})")
        else:
            empty_handed = partial_mean_readers(vote_result.group, candidates)
            if empty_handed:
                # C10 (b): textually distinct from the genuine single-family line below, because
                # the cheapest repair is different — re-ask THIS reader for THIS one number.
                reasons.append(
                    f"a second model family read this figure and returned a spread but no mean "
                    f"for group {vote_result.group or '?'} ({', '.join(empty_handed)}), so the "
                    f"value rests on one reader — a half-answer, not a family that never looked")
            else:
                reasons.append("only one independent route produced this value, so it cannot be "
                               "accepted by vote")
    else:
        score -= DISAGREE_PENALTY
        forced_human = True
        reasons.append(f"the routes disagree ({-DISAGREE_PENALTY:.2f}) — a human decides")

    # --- grounding of the readers whose numbers were kept
    grounded, short = _grounding(vote_result, candidates, adjudication if settled else None)
    if grounded is False:
        score -= UNGROUNDED_PENALTY
        reasons.append(f"a quote behind this value is not in the paper (-{UNGROUNDED_PENALTY:.2f})")
    elif grounded and short:
        score += SHORT_QUOTE_BONUS
        reasons.append(f"the quotes are grounded but one is too short to stand alone "
                       f"(+{SHORT_QUOTE_BONUS:.2f})")
    elif grounded:
        score += GROUNDED_BONUS
        reasons.append(f"every quote behind this value is printed in the paper "
                       f"(+{GROUNDED_BONUS:.2f})")

    # --- the adversarial verifier. `verifier_state` and not `.verdict`, because a record that
    # names no model, no prompt version and no call id is an infrastructure failure written down,
    # not a reader who looked (C8): it is priced at zero, and said out loud.
    kinds = {verifier_state(v) for v in verdicts}
    if "refuted" in kinds:
        forced_human = True
        reasons.append("a verifier refuted this reading — a human decides")
    elif "confirmed" in kinds:
        score += CONFIRMED_BONUS
        reasons.append(f"a verifier confirmed it — it tried to refute the reading and could not "
                       f"(+{CONFIRMED_BONUS:.2f})")
    elif "ambiguous" in kinds:
        score -= AMBIGUOUS_PENALTY
        reasons.append(f"the verifier could not settle it either way (-{AMBIGUOUS_PENALTY:.2f})")
    if NOT_RUN in kinds:
        failures = sorted({(v.reason or "no reason recorded").strip()
                           for v in verdicts if verifier_state(v) == NOT_RUN})
        reasons.append(f"a verifier for this cell could not be run, so this reading is "
                       f"unverified rather than doubted and the score is unchanged (0.00): "
                       + "; ".join(failures))
    if NO_VALUE_PRINTED in kinds:
        reasons.append("a verifier ran and reported that the paper prints no independent value "
                       "for this cell — a completed cross-check with nothing to check against, "
                       "so the score is unchanged (0.00)")

    # --- spread
    value = vote_result.mean if vote_result.mean is not None else 0.0
    if vote_result.mad and value and vote_result.mad / abs(value) > MAD_SHARE:
        score -= SPREAD_PENALTY
        reasons.append(f"the routes are {vote_result.mad / abs(value):.0%} apart around the value "
                       f"(-{SPREAD_PENALTY:.2f})")
    if vote_result.figure_conflict:
        score -= FIGURE_CONFLICT_PENALTY
        reasons.append(f"a figure read disagrees with the printed value that carried the vote "
                       f"(-{FIGURE_CONFLICT_PENALTY:.2f})")
    if vote_result.sigma and value and vote_result.sigma / abs(value) > SIGMA_SHARE:
        score -= SPREAD_PENALTY
        reasons.append(f"the digitisation uncertainty is {vote_result.sigma / abs(value):.0%} of "
                       f"the value (-{SPREAD_PENALTY:.2f})")

    # --- consistency flags. Penalties count distinct CODES, never flags: `run_checks` raises a
    # code on every candidate it applies to, and one problem reported six times is one problem.
    errors = [f for f in flags if f.severity == "error" and f.code not in NON_FORCING_ERRORS]
    noted = sorted({f.code for f in flags
                    if f.severity == "error" and f.code in NON_FORCING_ERRORS})
    if noted:
        reasons.append(f"{', '.join(noted)}: the calibration is wrong, not the value — the "
                       f"reading stands, capped for review")
    codes = {f.code for f in flags}
    warns = sorted({f.code for f in flags if f.severity == "warn"} - WITHHOLDING_FLAGS
                   - set(CALIBRATION_PENALTY))
    capping_warns = sorted({f.code for f in flags if f.severity == "warn"}
                           & (CAPPING_FLAGS - set(CALIBRATION_PENALTY)))
    infos = sorted({f.code for f in flags if f.severity == "info"})
    if errors:
        forced_human = True
        reasons.append("consistency errors stand: " + ", ".join(sorted({f.code for f in errors})))
    if warns:
        penalty = min(WARN_CAP, WARN_PENALTY * len(warns))
        score -= penalty
        reasons.append(f"warnings ({', '.join(warns)}) -{penalty:.2f}")
    if infos:
        penalty = min(INFO_CAP, INFO_PENALTY * len(infos))
        score -= penalty
        reasons.append(f"notes ({', '.join(infos)}) -{penalty:.2f}")

    # --- "this may be a different quantity" (CONTRADICTING_FLAGS). Deducted HERE, ABOVE the
    # floor, and forcing: a contradiction is evidence about what was measured, not a cap on how
    # well corroborated it is, so the floor below must not be able to restore it. Under one shared
    # floor these were arithmetically incapable of withholding anything — a figure read off the
    # wrong panel in the wrong unit, carrying `axis_conflict`, landed on exactly the acceptance
    # threshold and would have been pooled.
    contradicting = sorted(codes & CONTRADICTING_FLAGS)
    if contradicting:
        penalty = min(WARN_CAP, WARN_PENALTY * len(contradicting))
        score -= penalty
        forced_human = True
        reasons.append(f"{', '.join(contradicting)} (-{penalty:.2f}) — a human decides: "
                       + "; ".join(CAP_REASONS.get(code, "this may be a different quantity")
                                   for code in contradicting)
                       + ". Agreement about a number never establishes that it is the right "
                         "number, so this is not something a score can settle")

    # Everything from here to the end of the caps is the CAPPING mechanism, and R2 says a
    # combination of caps clamps AT `accept_with_note` — it never composes downwards into
    # `needs_human`. So the score as it stands is remembered, the capping deductions and ceilings
    # are applied, and the result is floored back to it: a cap can lower a cell to
    # `accept_with_note` and no further, while a deduction that was already there for other
    # reasons (an ungrounded quote, routes that are far apart, a contradiction above) still
    # stands on its own.
    floor = min(score, ACCEPT_WITH_NOTE)

    if capping_warns:
        penalty = min(WARN_CAP, WARN_PENALTY * len(capping_warns))
        score -= penalty
        # said plainly because it is true: under the floor below, this deduction ranks a cell
        # inside the accept band and can never move it out of one (see the restore line)
        reasons.append(f"warnings ({', '.join(capping_warns)}) -{penalty:.2f}, which orders this "
                       f"cell in the review queue but cannot withhold it")

    # --- how well corroborated the axis was, scored once and ordered (see CALIBRATION_PENALTY)
    axis_codes = sorted(codes & set(CALIBRATION_PENALTY))
    if axis_codes:
        deduction = max(CALIBRATION_PENALTY[code] for code in axis_codes)
        score -= deduction
        reasons.append(f"{', '.join(axis_codes)}: the axis this value was read against is not "
                       f"fully corroborated (-{deduction:.2f})")

    # --- orientation
    if orientation is None or orientation.higher_is_better is None:
        forced_human = True
        reasons.append("the direction of this measure is unresolved, so the sign of any effect "
                       "size built on it is undecided")

    # --- caps
    if adjudication is not None:
        if adjudication.needs_human:
            forced_human = True
            reasons.append("the adjudicator asked for a human")
        score = min(score, ADJUDICATED_CAP)
        reasons.append(f"adjudicated cells are capped at {ADJUDICATED_CAP:.2f}")
    if not settled and (one_witness or (
            vote_result.agreement == "single"
            and len(_agreeing_model_families(vote_result, candidates)) < 2)):
        score = min(score, SINGLE_ROUTE_CAP)
    capping = sorted(codes & CAPPING_FLAGS)
    if capping or contradicting:
        score = min(score, ADJUDICATED_CAP)
    if capping:
        reasons.append(f"{', '.join(capping)} caps this cell at {ADJUDICATED_CAP:.2f}: "
                       + "; ".join(CAP_REASONS.get(code, "this reading is unconfirmed")
                                   for code in capping))
    if score < floor:                    # R2: the caps clamp at accept_with_note, never below it
        reasons.append(f"the caps above stop at {floor:.2f}: an under-corroborated reading is a "
                       f"reason for a human to look at this cell, not on its own a reason to "
                       f"withhold it")
        score = floor

    # --- amendment F: a purely digitised cell has to earn its automatic acceptance
    if routes and all(is_figure_route(r.route_key) for r in routes) and not settled:
        ok, delta, share, gate_reasons = figure_gate(candidates or [], n_a, n_b)
        reasons += gate_reasons
        if not ok:
            score = min(score, ADJUDICATED_CAP)
            reasons.append(f"a digitised cell that does not meet the digitisation gate cannot "
                           f"be accepted automatically, so it stops at {ADJUDICATED_CAP:.2f}")

    score = round(max(0.0, min(1.0, score)), 4)
    # --- C11: publish the margin — but only where the SCORE decided the bucket. C11's rule is
    # "within 0.03 of any boundary the cell CLEARED", and a cell held by an error, a refutation, a
    # contradiction or an unresolved direction cleared nothing: its score is a number the decision
    # never consulted. Publishing a distance for it printed "margin 0.2500 clears auto_accept" in
    # the review queue beside a held cell — the exact claim about "which side it landed on" that
    # C11 exists to make checkable, made about a cell where the score landed on no side at all
    # (review M3).
    if forced_human:
        reasons.append(BUCKET_NOT_SCORED)
        return _Scored("needs_human", score, reasons, forced_human=True)
    reasons.append(_margin_reason(score))
    if score >= AUTO_ACCEPT:
        return _Scored("auto_accept", score, reasons, forced_human=False)
    if score >= ACCEPT_WITH_NOTE:
        return _Scored("accept_with_note", score, reasons, forced_human=False)
    reasons.append(f"score {score:.2f} is below {ACCEPT_WITH_NOTE:.2f}")
    return _Scored("needs_human", score, reasons, forced_human=False)


# ----------------------------------------------------------------------------- resolve_cell
#: worst first. The two no-evidence states rank BELOW `confirmed` — a cell with one failed call
#: and one confirmation was confirmed, and a cell with nothing but failed calls summarises as
#: `not_run`, which is the same word `_verifier_summary` already used for "no verdicts at all".
_VERIFIER_ORDER = {"refuted": 0, "ambiguous": 1, "confirmed": 2, NOT_RUN: 3, NO_VALUE_PRINTED: 3}


def _verifier_summary(verdicts: Sequence[VerifierVerdict], ids: set[str]) -> tuple[str, str]:
    relevant = [v for v in verdicts if not v.candidate_id or v.candidate_id in ids]
    if not relevant:
        return NOT_RUN, ""
    worst = min(relevant, key=lambda v: _VERIFIER_ORDER.get(verifier_state(v), 1))
    return verifier_state(worst), worst.reason


def _route_name(routes: Sequence) -> str:
    return _base_kind(routes) if routes else ""


def resolve_cell(dataset: DatasetSpec, outcome_key: str, group: str,
                 candidates: Sequence[Candidate], *, vote_result: VoteResult | None = None,
                 verdicts: Sequence[VerifierVerdict] = (),
                 flags: Sequence[CheckFlag] | None = None,
                 adjudication: Adjudication | None = None,
                 orientation: OrientationVerdict | None = None,
                 other_candidates: Sequence[Candidate] = (), axis_range: float | None = None,
                 n_a: int | None = None, n_b: int | None = None) -> Verdict:
    """One `Verdict` for one (dataset × outcome × group): vote, checks, verifier, adjudication.

    Everything is optional except the candidates: pass what the orchestrator has already run and
    this fills in the rest (the vote and the consistency checks), so Task 10 makes one call per
    cell and gets back the row Task 9 resolves into an effect size.
    """
    mine = [c for c in candidates if c.group == group or c.kind != "group_stats"]
    result = vote_result if vote_result is not None else vote(mine, axis_range, group=group)
    flag_list = list(flags) if flags is not None else run_checks(
        dataset, outcome_key, candidates, other_candidates=other_candidates,
        orientation=orientation)

    ruling = adjudication.group_values(group) if adjudication is not None else None
    routes = _agreeing_routes(result)
    verdict = Verdict(
        dataset_id=dataset.dataset_id, outcome_key=outcome_key, group=group,
        agreement=result.agreement, agreeing_ids=list(result.agreeing_ids),
        disagreeing_ids=list(result.disagreeing_ids), vote_method=result.method,
        vote_tolerance=result.tolerance, needs_third_candidate=result.needs_third_candidate,
        flags=flag_list, mean=result.mean, dispersion_value=result.dispersion_value,
        dispersion_type=result.dispersion_type, ci_low=result.ci_low, ci_high=result.ci_high,
        points=list(result.points), n=result.n, unit=result.unit, sigma=result.sigma,
        mad=result.mad, route=_route_name(routes),
        candidate_ids=list(result.agreeing_ids),
        analysis_metric=_analysis_metric(mine, result.agreeing_ids))

    verdict.verifiers = [v for v in verdicts if not v.candidate_id
                         or v.candidate_id in {c.candidate_id for c in mine}]
    verdict.verifier_verdict, verdict.verifier_reason = _verifier_summary(
        verdicts, {c.candidate_id for c in mine})
    verdict.reopens = max((v.reopen for v in verdict.verifiers), default=0)

    if ruling is not None and ruling.mean is not None:
        verdict.adjudicated = True
        verdict.mean = ruling.mean
        verdict.dispersion_value = ruling.dispersion_value
        verdict.dispersion_type = ruling.dispersion_type
        verdict.n = ruling.n if ruling.n is not None else verdict.n
        verdict.unit = ruling.unit or verdict.unit
        verdict.candidate_ids = list(ruling.chosen_candidate_ids)
        verdict.route = "adjudicated"
    if adjudication is not None:
        verdict.adjudicated = True
        verdict.adjudication_rationale = "; ".join(
            part for part in (adjudication.rationale, ruling.reason if ruling else "") if part)

    if orientation is not None:
        verdict.higher_is_better = orientation.higher_is_better
        verdict.orientation_source = orientation.orientation_source
        verdict.orientation_evidence = "; ".join(
            part for part in (orientation.reason, *orientation.quotes) if part)

    scored = confidence(result, verdict.verifiers, flag_list, adjudication,
                        candidates=list(candidates), n_a=n_a, n_b=n_b, orientation=orientation)
    bucket, score, reasons = scored
    verdict.confidence, verdict.confidence_score, verdict.confidence_reasons = bucket, score, reasons
    # C11: the same numbers `confidence()` already publishes in prose, as fields — so the review
    # CSV and the report can sort and filter on them instead of a reviewer parsing a sentence.
    # Left empty on a cell whose bucket the score did not decide (M3): "0.2500 from auto_accept"
    # sorts and filters as a claim about a decision, and on such a cell no such decision was made.
    if getattr(scored, "forced_human", False):
        verdict.confidence_margin, verdict.nearest_boundary = None, ""
    else:
        verdict.confidence_margin, verdict.nearest_boundary, _ = confidence_margin(score)
    verdict.needs_human = bucket == "needs_human"
    return verdict


def _analysis_metric(candidates: Sequence[Candidate], agreeing_ids: Sequence[str]) -> str:
    agreeing = set(agreeing_ids)
    metrics = [c.analysis_metric for c in candidates
               if c.candidate_id in agreeing and c.analysis_metric != "unknown"]
    return metrics[0] if len(set(metrics)) == 1 and metrics else "unknown"
