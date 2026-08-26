"""Consistency checks (spec §3.3(1) + amendment G) — code only, no model, no repair.

Everything here is arithmetic or bookkeeping over candidates that already exist. A check states
what is wrong, which candidates it is wrong on, and how badly; it never edits a candidate and
never drops one, because a value quietly "fixed" by the pipeline is a value nobody reviewed.

The rules fall into three families:

* *is this a number at all* — n ≥ 2, dispersion > 0, an SE that implies the reported SD, a
  symmetric interval, a figure value inside its own axis, a statistic with degrees of freedom;
* *is it the number we asked for* — the group label the extractor echoed, the unit, the analysis
  metric, the dispersion type the mapper saw at that location, the same numbers turning up under
  two different outcomes;
* *can it be used at all* — is the direction of this measure known, did two agents ever agree on
  what the error bars in a figure mean.

Severity is what `canopy.verify.confidence` weighs: an `error` sends the cell to a human, a `warn`
costs it points, an `info` is recorded for the provenance bundle.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Any, Iterable, Sequence

from ..models import (Candidate, CheckFlag, DatasetSpec, DispersionType, GroupSpec,
                      OrientationVerdict, OutcomeSources, Source)
from ..stats.effect_sizes import CONVERTIBLE_DESIGNS, cohens_d
from .figures import (CAL_STATUSES, FIGURE_KINDS, axis_limits, calibration_status, is_figure,
                      routes_agree)
from .grounding import ROW_ONLY, SIGN_NOTE, is_short_quote

__all__ = ["run_checks", "sign_check", "codes", "CHECK_SEVERITY", "GROUP_LABEL_MISMATCH_NOTE",
           "ROW_ONLY_MARKER", "SIGN_NOTE_MARKER", "SEVERITY_RANK", "MIN_N", "MAX_PLAUSIBLE_D",
           "AXIS_TESTABLE", "ORIENTATION_FLAGS", "DF_PROVENANCE_FLAGS", "DISPUTED_MEANS_FLAGS",
           "orientation_note", "n_before_exclusions", "excluded_count", "EXCLUSION_SPAN",
           "EXCLUSION_COUNT_SPAN",
           "CHECK_SEVERITY_PREFIXES", "severity_of", "DF_SHORTFALL_TOLERANCE", "best_statistic"]

#: the marker `canopy.agents.extract_common.group_label_check` writes into a candidate's notes when
#: the label the extractor echoed belongs to the *other* group (tests/test_checks.py pins it).
GROUP_LABEL_MISMATCH_NOTE = "group label mismatch"
#: the exact shapes `grounding.check_table_cell` writes its two table-cell markers in — the bare
#: words would also match a reviewer's prose, so the punctuation that follows them is part of it
ROW_ONLY_MARKER = f"{ROW_ONLY}:"
SIGN_NOTE_MARKER = f"{SIGN_NOTE} for "

#: An `OrientationVerdict` carries no flag list of its own, so `combine_orientation` records the
#: codes it decided on in the verdict's `notes`, bracketed, and `_check_outcome` reads them back
#: out to where `canopy.verify.confidence` can weigh them — the same note-marker contract
#: `GROUP_LABEL_MISMATCH_NOTE` and `ROW_ONLY_MARKER` above use. Brackets keep them out of prose.
ORIENTATION_FLAGS: tuple[str, ...] = ("orientation_reader_contradicts_values",
                                      "orientation_single_witness", "orientation_by_majority",
                                      "orientation_direction_conflict")

#: C9 codes about the printed DEGREES OF FREEDOM. Both are errors — a statistic whose provenance
#: to these two groups is unverified may not pool — but none of them is a VALUE dispute, which is
#: the only thing the adjudicator is for. "This t is printed with no df" is a fact about the paper,
#: and no model can supply a number the paper does not print; asking one buys the adjudicator
#: model with the whole paper in context for a cell that is held whatever it answers (review M5).
#: Excluded from the adjudication trigger for exactly the reason `ORIENTATION_FLAGS` is, and
#: exported as a name so that the trigger cannot drift from the codes.
#:
#: `test_stat_missing_df` is the third (controller ruling, fix round 1): it says the same thing as
#: `df_missing` about the same statistic — a t or F printed with no degrees of freedom at all —
#: and it is the code that fires on the very failing input C9 was written against, so leaving it
#: out left the ruling inert on the commonest shape it exists for. It is the third member of
#: `UNVERIFIED_CONTRAST_FLAGS` for the same reason.
DF_PROVENANCE_FLAGS: tuple[str, ...] = ("df_missing", "df_shortfall_unexplained",
                                        "test_stat_missing_df")

#: The deterministic orientation check compares a reader's stated direction against the RESOLVED
#: raw means, so it must not fire while WHICH SERIES IS WHICH is itself in dispute: under any of
#: these the means it would judge the reader against may be the other group's (ADVERSARIAL round 2
#: on C3). The check abstains instead of discarding. `group_label_swapped` is here for the same
#: reason as `series_transposed` and nothing weaker: it is an `error` saying the label an
#: extractor echoed belongs to the OTHER group, so a discard run against those means throws out
#: the reader that read the paper correctly.
DISPUTED_MEANS_FLAGS: frozenset[str] = frozenset({"series_marker_mismatch", "series_transposed",
                                                  "axis_conflict", "group_label_swapped"})


def orientation_note(code: str, message: str) -> str:
    """One bracketed orientation note, as `combine_orientation` writes it into `verdict.notes`."""
    if code not in ORIENTATION_FLAGS:                # pragma: no cover - programming error
        raise KeyError(f"{code!r} is not an orientation note code")
    return f"[{code}] {message}"

MIN_N = 2                       # a group of one has no within-group variance
#: |d| above this is nearly always a wrong DENOMINATOR — an SE printed as an SD, a within-subject
#: error bar read as a between-subject one, the other group's spread. C9 makes it forcing rather
#: than a warning: the numerator is usually fine, so the arithmetic looks plausible all the way
#: down and only the size of the answer gives it away. Live corpus check: Bock d2 late adaptation
#: sits at |d| = 2.9610, just under the line, so nothing in the nine-paper run changes today.
MAX_PLAUSIBLE_D = 3.0
#: how far printed degrees of freedom may sit from n_a + n_b - 2 when the shortfall is EXPLAINED
#: (a stated exclusion that also reduces the participant total, or an n nobody printed). When it
#: is not explained, exact equality is required — see `_check_conversion_gate`.
DF_SHORTFALL_TOLERANCE = 2.0
SE_SD_TOLERANCE = 0.05          # relative gap allowed between SE·√n and a reported SD
CI_ASYMMETRY_TOLERANCE = 0.05   # relative gap allowed between the two halves of an interval
SD_NEAR_ZERO_RATIO = 1e-3       # dispersion this small next to the mean reads as "no variance"
SD_NEAR_ZERO_ABS = 1e-9
AXIS_MARGIN = 0.01              # a digitised value may sit this far outside the axis (1% of range)

SEVERITY_RANK = {"error": 0, "warn": 1, "info": 2}

#: every code this module can emit, with the severity `confidence` weighs it at
CHECK_SEVERITY: dict[str, str] = {
    # --- is this a number at all
    "n_not_integer": "error",
    "n_too_small": "error",
    "n_missing": "warn",
    "n_mismatch": "warn",
    "n_sum_mismatch": "warn",
    #: D4-lite. The group size this cell was scored on is a size the paper prints BEFORE the
    #: exclusions it then reports, so the analysed group may be smaller than the row's denominator
    #: says. A `warn`, and a cap in `confidence`: an n four people too large moves a variance, not
    #: a mean, and nothing here says the value came from the wrong place. The number is not
    #: repaired — a reviewer answers with the analysed sizes (`group_n`).
    "n_before_exclusions": "warn",
    "sd_nonpositive": "error",
    "sd_near_zero": "warn",
    "se_sd_inconsistent": "warn",
    "ci_asymmetric": "warn",
    "value_outside_axis": "error",
    #: the axis a figure value was read against, and how much corroboration it had (task 16 P1/P3)
    "calibration_single_witness": "warn",
    "calibration_refuted": "error",
    "calibration_disputed": "error",
    "calibration_missing": "warn",
    #: the winning calibration kept exactly two ticks: they define the affine map, and nothing
    #: verifies its linearity — the reading is usable but rests on an unchecked assumption
    "calibration_two_point": "warn",
    "mean_missing": "warn",
    "dispersion_unknown": "warn",
    "dispersion_missing": "warn",
    "test_stat_missing_df": "error",
    #: ceiling item C9. The first two are errors because both mean the row must not pool:
    #: `df_missing` says nothing establishes that the printed statistic is the contrast between
    #: THESE two groups (a post-hoc from a three-group analysis prints the same t), and
    #: `df_shortfall_unexplained` says the printed df contradicts the analysed group sizes and the
    #: paper offers no reason for the gap.
    "df_missing": "error",
    "df_shortfall_unexplained": "error",
    #: …and `implausible_dispersion` is an EARLY WARNING here, not the ruling (review H1). The
    #: binding screen is `confidence.dispersion_plausibility_bucket`, on the resolved |d| that
    #: reaches the plot: this one can only see raw candidates, before the vote, before the
    #: adjudicator and before any SE/IQR/range conversion, so on the majority shape in the live
    #: corpus it is blind while the resolver is not. It stays because it is the cheapest place to
    #: NAME the suspect denominator to a reviewer — and it is a `warn` because an early warning
    #: that also withheld the cell bought an adjudicator call whose own answer this check would
    #: then never screen (M5).
    "implausible_dispersion": "warn",
    # --- is it the number we asked for
    "quote_not_grounded": "error",
    "quote_short": "info",
    "quote_row_only": "warn",
    "sign_not_confirmed": "warn",
    "group_label_swapped": "error",
    "unit_mismatch": "warn",
    "unit_other_expression": "info",
    "metric_mixed": "warn",
    "metric_mixed_across_outcomes": "info",
    "unit_incoherent": "warn",
    "dispersion_type_conflict": "warn",
    "dispersion_type_from_legend": "warn",
    #: the named panel could not be isolated from its neighbours, so the reader was handed the
    #: union crop. DOUBT, not evidence of error: the value may be perfectly right, and the
    #: axis-identity, overlay and verifier nets still apply to it. Doubt caps; contradiction
    #: withholds — so this is a `warn` in `CAPPING_FLAGS`, never a hold.
    "panel_not_isolated": "warn",
    #: fix E: the panel crop was refused by a majority of readers and the reading was
    #: re-acquired from the full page render. Same doubt family as `panel_not_isolated`,
    #: never stacked with it (the digitiser suppresses that flag on a re-acquire).
    "crop_reacquired": "warn",
    #: fix F: the figure's panel letters could not be verified against its caption at ingest and
    #: no page render existed to prefer, so a letter-addressed crop was read under the doubt
    #: that its letter belongs to a sibling. Same family; the page path carries `crop_reacquired`
    #: instead (one doubt, one price).
    "panel_labels_disputed": "warn",
    #: fix G: a verifier refuted a figure read against printed values and the reading was
    #: re-acquired from the full page render. The re-read's own `crop_reacquired` is the cap;
    #: this code is the cell-level record that the repair happened, priced as information only.
    "reacquired_on_refutation": "warn",
    #: WHERE in a figure a reading was taken, and whether the caption agrees it is this group's
    #: panel (D2). All three are `warn`: a reading off the wrong panel is a claim about the
    #: LOCATION, and the location is decided by the caption and the vote, not by an error budget
    #: — `confidence` weighs the first two as contradictions and the third as a cap.
    "locator_reads_conflict": "warn",
    "locator_panel_mismatch": "warn",
    "locator_reads_set_aside": "warn",
    "duplicate_across_outcomes": "warn",
    "figure_n_mismatch": "warn",
    "reopened_on_better_source": "warn",
    "collapsed_across_x": "warn",
    "categorical_x_unsupported": "warn",
    #: the value is the ONE point at the x category the source names (D3). Nothing here says the
    #: number is wrong — it says the cell rests on a single plotted point rather than on a series,
    #: which is a reason for a reviewer to look at the figure. `warn`, and a cap in `confidence`.
    "categorical_point_read": "warn",
    #: what the categorical x axis IS was resolved from the readers' own category reports, with
    #: two or more readings supporting the ruling on their own — figure evidence, corroborated,
    #: so it caps like `categorical_point_read` rather than withholding
    "categorical_x_resolved_from_readings": "warn",
    #: …and the same resolution resting on ONE reading (or one in tension with the protocol's own
    #: measurement window): an inferred premise. The row is held for a person
    #: (`confidence.INFERRED_PREMISE_FLAGS`) and lands in the best-guess line under its named rule.
    "categorical_x_single_witness": "warn",
    "points_undercount": "warn",
    # --- can it be used at all
    "orientation_unknown": "warn",
    #: how the direction of this measure was settled, when it was not simply two agreeing readers
    #: (ceiling items C3, C12 and the P-A residue). The last three do not say the direction is
    #: WRONG: each says a reviewer should see how thin the agreement behind it was.
    #:
    #: `orientation_reader_contradicts_values` is the exception and is an `error`: the reader the
    #: deterministic check discarded is the one that made the ONLY mechanically checkable claim on
    #: this measure, and it said the opposite of what this cell's own resolved means say. What is
    #: in question afterwards is the VALUES, not merely how well corroborated the direction is, so
    #: it holds the cell instead of costing it 0.08 that a good score absorbs (fix round F5); it
    #: sits in `confidence.CONTRADICTING_FLAGS` for the same reason.
    "orientation_reader_contradicts_values": "error",
    "orientation_single_witness": "warn",
    "orientation_by_majority": "warn",
    "orientation_direction_conflict": "warn",
    "sign_mismatch": "error",
    "figure_error_bar_unknown": "warn",
    "series_identity_conflict": "warn",
    "series_marker_mismatch": "warn",
    "series_transposed": "warn",
    "axis_conflict": "warn",
    "error_bar_unconfirmed": "info",
}

#: dispersion types that are a spread and must therefore be strictly positive
#: Codes that carry a measured quantity in their tail (`df_off_by_1`, `df_off_by_1.5`), so the
#: severity is declared once for the FAMILY. A family is a prefix and nothing more clever: a code
#: whose severity cannot be found either way is a typo, and `severity_of` raises on it rather than
#: letting `confidence` weigh an undeclared code as an ordinary warning by accident.
CHECK_SEVERITY_PREFIXES: dict[str, str] = {"df_off_by_": "warn"}


def severity_of(code: str) -> str:
    """The declared severity of one code, family codes included."""
    if code in CHECK_SEVERITY:
        return CHECK_SEVERITY[code]
    for prefix, severity in CHECK_SEVERITY_PREFIXES.items():
        if code.startswith(prefix):
            return severity
    raise KeyError(f"{code!r} has no declared severity")


_POSITIVE_DISPERSIONS = frozenset({DispersionType.SD, DispersionType.SE, DispersionType.IQR,
                                   DispersionType.RANGE})
_INTERVALS = frozenset({DispersionType.CI95, DispersionType.CI90})
_FIGURE_KINDS = FIGURE_KINDS


def codes(flags: Iterable[CheckFlag]) -> list[str]:
    """The distinct codes in a flag list, sorted — what tests and the review queue read."""
    return sorted({flag.code for flag in flags})


# ----------------------------------------------------------------------------- small helpers
def _flag(out: list[CheckFlag], code: str, message: str, *candidate_ids: str) -> None:
    out.append(CheckFlag(code=code, severity=severity_of(code), message=message,
                         candidate_ids=[cid for cid in candidate_ids if cid]))


def _found_stats(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Group statistics an extractor says it actually read (a gap is not a bad value)."""
    return [c for c in candidates if c.kind == "group_stats" and c.status == "found"]


def _norm_unit(unit: str) -> str:
    return "".join((unit or "").split()).casefold()


def _group_spec(dataset: DatasetSpec, group: str | None) -> GroupSpec | None:
    if group == "A":
        return dataset.group_a
    if group == "B":
        return dataset.group_b
    return None


def _outcome(dataset: DatasetSpec, outcome_key: str) -> OutcomeSources | None:
    try:
        return dataset.outcome(outcome_key)
    except KeyError:
        return None


def _close(a: float, b: float, rel: float) -> bool:
    scale = max(abs(a), abs(b), 1e-12)
    return abs(a - b) <= rel * scale


# ----------------------------------------------------------------------------- per-candidate
def _check_one(cand: Candidate, dataset: DatasetSpec, outcome: OutcomeSources | None,
               out: list[CheckFlag]) -> None:
    cid = cand.candidate_id
    if cand.quote.strip():
        if cand.grounded is False:
            _flag(out, "quote_not_grounded",
                  f"the quote behind this value is not printed in the paper "
                  f"(best similarity {cand.grounding_similarity})", cid)
        elif cand.grounded and is_short_quote(cand.quote):
            _flag(out, "quote_short",
                  f"the quote {cand.quote.strip()!r} is too short to stand as evidence on its own",
                  cid)
    if ROW_ONLY_MARKER in cand.notes:
        _flag(out, "quote_row_only",
              f"the transcribed numbers are somewhere in the named table row but NOT in the "
              f"column this candidate claims — which is what reading the other group's cell looks "
              f"like ({cand.notes})", cid)
    if SIGN_NOTE_MARKER in cand.notes:
        _flag(out, "sign_not_confirmed",
              f"a transcribed value matched the table only without its sign, so the direction of "
              f"this number is not confirmed ({cand.notes})", cid)
    if GROUP_LABEL_MISMATCH_NOTE in cand.notes:
        _flag(out, "group_label_swapped",
              f"the group label this extractor echoed belongs to the other group ({cand.notes})",
              cid)

    # …before the `found` gate: a cell that REFUSED to read a categorical axis has no value, and
    # the whole point of the refusal is that it says why rather than going quiet (task 16 P6)
    _check_categorical_x(cand, out)
    _check_panel_isolation(cand, out)
    _check_dispersion_source(cand, out)
    if cand.kind in ("test_statistic", "reported_d"):
        _check_statistic(cand, dataset, out)
        return
    if cand.status != "found":
        return

    spec = _group_spec(dataset, cand.group)
    analysed_n = spec.n if spec is not None else None

    # --- n
    if cand.n is None:
        _flag(out, "n_missing", "no group size was transcribed with this value", cid)
    elif not isinstance(cand.n, int) or isinstance(cand.n, bool):
        _flag(out, "n_not_integer", f"group size {cand.n!r} is not a whole number", cid)
    elif cand.n < MIN_N:
        _flag(out, "n_too_small",
              f"group size {cand.n} is below {MIN_N}: a group of one has no within-group "
              f"variance, so no effect size can be built from it", cid)
    elif analysed_n is not None and cand.n != analysed_n:
        code = "figure_n_mismatch" if is_figure(cand) else "n_mismatch"
        _flag(out, code, f"this value carries n = {cand.n}, but the map analysed "
                         f"n = {analysed_n} in group {cand.group}", cid)

    # --- the value and its spread
    if cand.mean is None and not cand.points:
        _flag(out, "mean_missing", "the extractor reported a found value with no mean", cid)
    if cand.points:
        wanted = cand.n if cand.n is not None else analysed_n
        if wanted is not None and len(cand.points) < wanted:
            _flag(out, "points_undercount",
                  f"only {len(cand.points)} points were read where {wanted} participants were "
                  f"analysed", cid)

    dispersion, kind = cand.dispersion_value, cand.dispersion_type
    has_interval = cand.ci_low is not None and cand.ci_high is not None
    if dispersion is None and not has_interval and not cand.points:
        _flag(out, "dispersion_missing", "this value carries no dispersion of any kind", cid)
    elif dispersion is not None and kind is DispersionType.UNKNOWN:
        _flag(out, "dispersion_unknown",
              f"a spread of {dispersion} was read but nobody could say what kind it is", cid)
    if dispersion is not None and kind in _POSITIVE_DISPERSIONS:
        if dispersion <= 0:
            _flag(out, "sd_nonpositive",
                  f"{kind.value} = {dispersion} is not a positive spread", cid)
        elif dispersion <= SD_NEAR_ZERO_ABS or (
                cand.mean is not None and abs(cand.mean) > 0
                and dispersion / abs(cand.mean) < SD_NEAR_ZERO_RATIO):
            _flag(out, "sd_near_zero",
                  f"{kind.value} = {dispersion} is effectively zero next to a mean of "
                  f"{cand.mean} — an effect size built on it would be unbounded", cid)

    if has_interval and cand.mean is not None:
        lower, upper = cand.mean - cand.ci_low, cand.ci_high - cand.mean
        width = cand.ci_high - cand.ci_low
        if width > 0 and not _close(lower, upper, CI_ASYMMETRY_TOLERANCE):
            _flag(out, "ci_asymmetric",
                  f"the interval [{cand.ci_low}, {cand.ci_high}] is not symmetric about the mean "
                  f"{cand.mean} ({lower} below, {upper} above)", cid)

    # --- a digitised value has to be inside the axis it was read from — but only when we know
    # what the axis is. Convicting a value on a calibration nobody corroborated is exactly how a
    # correct read of 31.3 was sent to a human by a ladder that had been misread as 1..4 (F1).
    _check_calibration(cand, out)
    _check_series_identity(cand, out)
    # …on a calibration that is worth testing against — see `AXIS_TESTABLE`. A `single_witness`
    # ladder is tested, but ONLY while the routes that read the figure agree with each other: two
    # readers agreeing on a value the frame cannot draw is evidence about the reading, and
    # dropping that check outright — as the first cut did — left a ladder misread the other way
    # with nothing at all to stop it.
    status = calibration_status(cand.pixel_provenance)
    testable = AXIS_TESTABLE[status] and (
        status != "single_witness" or routes_agree(cand.pixel_provenance) is not False)
    if cand.mean is not None and testable:
        limits = axis_limits(cand.pixel_provenance)
        if limits is not None:
            low, high = limits
            margin = AXIS_MARGIN * (high - low)
            if not (low - margin <= cand.mean <= high + margin):
                _flag(out, "value_outside_axis",
                      f"the value {cand.mean} is outside the calibrated axis [{low}, {high}]", cid)

    # --- what the mapper saw at this location
    conflict = _dispersion_conflict(cand, outcome)
    if conflict is not None:
        mapped, where = conflict
        _flag(out, "dispersion_type_conflict",
              f"the extractor read {kind.value} at {where}, but the map determined "
              f"{mapped.value} there", cid)


#: Can a `value_outside_axis` test be made against a figure in this calibration state? Written as
#: a TOTAL map rather than a list of the passing cases, because the list-of-passing-cases form
#: silently skipped any state nobody had thought about — which is how `none` ("we could not
#: establish the scale at all") ended up checked LESS than `single_witness` ("we established it
#: with one witness"). A state added to `CAL_STATUSES` without a decision here fails at import.
AXIS_TESTABLE: dict[str, bool] = {
    #: two or more witnesses agree on the mapping — a value outside THAT frame is a real error
    "confirmed": True,
    #: a record from before the ladder was scored: the `cal` it carries is all there is to test
    "unknown": True,
    #: one witness built it; tested only while the routes that read the figure agree (below)
    "single_witness": True,
    #: `digitize` writes no `cal` for a ladder the readers disproved, so there are no limits
    "cal_refuted": False,
    #: no ladder was built at all, so likewise there is nothing to test the value against — which
    #: is exactly why `_check_calibration` has to say so out loud instead
    "none": False,
}
assert set(AXIS_TESTABLE) == set(CAL_STATUSES) | {"unknown"}, \
    "every calibration state must be decided about, not fall through to 'not tested'"


def _check_calibration(cand: Candidate, out: list[CheckFlag]) -> None:
    """What the axis this figure value was read against is worth (task 16, P1/P3 as corrected).

    Three different things used to arrive as one `value_outside_axis` error:

    * the calibration two or more witnesses agree on — a value outside THAT is a real error, and
      is flagged above, not here;
    * a calibration the readers contradict — `calibration_refuted`. The error is against the
      LADDER: two readers agreed on a number the ticks cannot draw, so their number has two
      witnesses and the ladder has none. It caps the cell instead of convicting it;
    * a calibration only one witness built, which nothing corroborates. On its own that is a
      `warn` and a cap; when the readers of that figure also disagree with each other, nothing is
      left standing and it is `calibration_disputed` — a human's problem.
    """
    if cand.status != "found" or not cand.pixel_provenance:
        return
    status = calibration_status(cand.pixel_provenance)
    cid = cand.candidate_id
    where = (cand.pixel_provenance or {}).get("figure_id") or cand.locator or "this figure"
    note = str((cand.pixel_provenance or {}).get("cal_note") or "")
    if status == "cal_refuted":
        _flag(out, "calibration_refuted",
              f"the tick ladder read off {where} cannot draw the values the readers agree on, so "
              f"the axis calibration was discarded and the reading rests on the read-outs alone "
              f"({note})", cid)
    elif status == "single_witness":
        pp = cand.pixel_provenance or {}
        source = str(pp.get("cal_source") or "")
        disputed = [str(x) for x in (pp.get("cal_disputed") or [])]
        overlay_disputed = any((entry or {}).get("not_applied")
                               for entry in (pp.get("overlay_iterations") or []))
        if source and source in disputed:
            _flag(out, "calibration_disputed",
                  f"the ladder {where} was read against ({source}) carries none of the tick "
                  f"values the readers report for this axis — it is a ladder of another axis of "
                  f"the crop, and it was used anyway because nothing better was built ({note})",
                  cid)
        elif overlay_disputed:
            _flag(out, "calibration_disputed",
                  f"the overlay judged a mark off the datum on {where}, but the calibration the "
                  f"mark was drawn with has one witness, so the mark and the reading disagree "
                  f"and nothing says which is wrong ({note})", cid)
        elif routes_agree(cand.pixel_provenance) is False:
            _flag(out, "calibration_disputed",
                  f"only one witness calibrated {where} AND the routes that read it disagree "
                  f"about the value, so neither the scale nor the number is corroborated "
                  f"({note})", cid)
        else:
            _flag(out, "calibration_single_witness",
                  f"only one witness calibrated the axis of {where}, so the scale this value was "
                  f"read against is uncorroborated ({note})", cid)
    elif status == "none":
        # zero witnesses cannot be quieter than one. The read-out routes need no ladder to produce
        # a number, so this cell reached the score with nothing said about its axis at all, while
        # a cell whose axis ONE witness had established was flagged and capped. Absence of a check
        # is a finding: `AXIS_TESTABLE` records that `value_outside_axis` cannot run here either.
        _flag(out, "calibration_missing",
              f"no calibration of the value axis of {where} could be built at all, so nothing "
              f"independent checked that this number lies on the axis it was read from, and the "
              f"scale rests entirely on the readers ({note})", cid)
    # …and INDEPENDENTLY of who corroborated it: a calibration that kept exactly two ticks is an
    # exact line through two points, so its residual is zero BY CONSTRUCTION and says nothing.
    # Two ticks are `fit_axis`'s own precondition — the reading is legitimate — but the third
    # tick is the only thing that ever VERIFIES linearity, and a two-tick fit of a log axis
    # reads out linear without a murmur. Flagged whatever the witness count, because a second
    # witness agreeing on the same two rungs corroborates the rungs, not the shape of the axis
    # between them.
    # …but never on a record from before `cal_status` existed: those were scored under the old
    # rules, and a new flag on an old stage file re-scores a run nobody re-ran (status "unknown"
    # is exactly the old-record case; every current digitize write records a status).
    cal = (cand.pixel_provenance or {}).get("cal")
    if status != "unknown" and isinstance(cal, dict) and len(cal.get("ticks") or []) == 2:
        _flag(out, "calibration_two_point",
              f"the calibration of {where} kept exactly two ticks: they define the scale "
              f"exactly, and nothing checks that the axis is linear between them", cid)


def _descriptor(value: Any) -> tuple[str, str]:
    """`(fill, shape)` from a recorded descriptor pair; `("", "")` for anything unusable."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return str(value[0] or ""), str(value[1] or "")
    return "", ""


def _positively_matches(described: tuple[str, str], detected: tuple[str, str]) -> bool:
    """Do these two agree on everything both state, AND state at least one thing in common?

    The producer's own test is the first half only, which is vacuously true against a marker whose
    fill and shape the pixel pass could not resolve — the detector's "I could not tell" read as a
    positive identification. Convicting a reading on that is how a correct row gets withheld.
    """
    if not any(a and b for a, b in zip(described, detected)):
        return False
    return all(not a or not b or a == b for a, b in zip(described, detected))


def _transposition_is_corroborated(series: dict[str, Any]) -> bool:
    """Was the swap SEEN at both measured points, or inferred from a marker nobody could read?"""
    described, detected = series.get("described"), series.get("detected")
    if not isinstance(described, dict) or not isinstance(detected, dict):
        return False
    groups = ("A", "B")
    if not all(g in described and g in detected for g in groups):
        return False
    return (_positively_matches(_descriptor(described["A"]), _descriptor(detected["B"]))
            and _positively_matches(_descriptor(described["B"]), _descriptor(detected["A"])))


def _check_series_identity(cand: Candidate, out: list[CheckFlag]) -> None:
    """Which SERIES this value came from, and which AXIS it was read against (misses 1, 4, 5).

    Two routes that agree on a number can still both be reading the wrong curve: group assignment
    rests on one free-text legend read, and the numeric agreement the plan stops on says nothing
    about it. Likewise a panel with two value axes gives two readers two correct answers that
    differ by a factor.
    """
    provenance = cand.pixel_provenance or {}
    series = provenance.get("series_identity")
    if isinstance(series, dict):
        notes = "; ".join(str(note) for note in series.get("notes") or [])
        if series.get("conflict"):
            _flag(out, "series_identity_conflict",
                  notes or "both groups resolve to the same plotted marker", cand.candidate_id)
        # A transposition and a marker quibble are NOT one finding, though they arrived here as
        # one code. "The described shape is not in this figure's marker vocabulary" is a doubt
        # about words; "each group's value was measured on the OTHER group's marker" is a
        # determination that the effect's sign is inverted, and nothing downstream can undo it.
        # They are split so the second can withhold the row and the first cannot.
        if series.get("transposed") and _transposition_is_corroborated(series):
            _flag(out, "series_transposed",
                  notes or "each group's value was measured on the other group's marker, which "
                           "inverts the sign of the effect",
                  cand.candidate_id)
        elif series.get("transposed"):
            # …and a conviction that inverts a sign may not rest on a marker the pixel pass could
            # not resolve: `_descriptors_match` is vacuously true against an unknown descriptor,
            # so an uncorroborated transposition is reported as the doubt it actually is.
            _flag(out, "series_marker_mismatch",
                  f"the two series may be transposed, but that could not be corroborated at both "
                  f"measured points — the pixel pass resolved no marker to compare against "
                  f"({notes or 'no descriptors were recorded'})",
                  cand.candidate_id)
        elif series.get("marker_mismatch"):
            _flag(out, "series_marker_mismatch",
                  notes or "the described marker is not the one found at this value",
                  cand.candidate_id)
    if provenance.get("axis_agreement") == "conflict":
        _flag(out, "axis_conflict",
              f"the readers of this figure answered off different value axes; the ensemble kept "
              f"{provenance.get('axis_kept')!r} and dropped "
              f"{', '.join(provenance.get('axis_dropped_samples') or [])}",
              cand.candidate_id)


def _check_dispersion_source(cand: Candidate, out: list[CheckFlag]) -> None:
    """The mapper could not say what the error bars are, and the legend was believed instead.

    That substitution is worth making — `confidence._sd_of` returns None for UNKNOWN, so no route
    reaches the figure gate and a cell fails on a spread the figure states plainly. But it is a
    reading of the figure's own words by the same model that read its values, and it replaces the
    determination two mapper agents are supposed to make. A legend that says "SEM" over SD bars
    would otherwise pool unflagged where UNKNOWN went to a human, so it is flagged and capped.
    """
    provenance = cand.pixel_provenance or {}
    if provenance.get("dispersion_type_from") != "legend":
        return
    kind = getattr(cand.dispersion_type, "value", cand.dispersion_type)
    _flag(out, "dispersion_type_from_legend",
          f"the map never determined what the error bars at this location are; {kind} was taken "
          f"from the figure's own legend ({str(provenance.get('legend_says') or '')[:120]!r}) "
          f"rather than from the two agents that are supposed to agree on it",
          cand.candidate_id)


def _check_panel_isolation(cand: Candidate, out: list[CheckFlag]) -> None:
    """The digitiser named a panel that ingestion could not isolate (`panel_not_isolated`).

    The union crop is still READ — refusing would be worse, and the axis-identity rules repaired
    exactly this figure once — but the row has to say that the picture the reader was handed is
    wider than the panel the map asked for, so a neighbouring panel's ladder is in the frame. That
    is a reason to look, not a reason to withhold: `confidence.CAPPING_FLAGS` caps it below
    automatic acceptance and floors it at `ACCEPT_WITH_NOTE`.
    """
    provenance = cand.pixel_provenance or {}
    if provenance.get("crop_reacquired"):
        # fix E: the panel crop was refused by a majority of readers and the reading was
        # re-acquired from the full page render — wider than any panel, so a reviewer should
        # see the figure. The same doubt family as `panel_not_isolated`, priced once (the
        # digitiser suppresses that flag on a re-acquire so the two never stack).
        _flag(out, "crop_reacquired",
              str(provenance.get("reacquire_reason")
                  or "the panel crop was refused and the reading re-acquired from the page"),
              cand.candidate_id)
    if provenance.get("panel_labels_disputed"):
        # fix F's no-page fallback: the letters of this figure could not be verified against
        # its caption and there was no page render to prefer, so the letter-addressed crop was
        # read under the doubt that it answers to a sibling's letter
        _flag(out, "panel_labels_disputed",
              f"the figure's panel lettering is disputed at ingest "
              f"({str(provenance.get('panel_labels_disputed'))[:200]}) and the reading was "
              f"taken from the letter-addressed crop anyway (no page render existed to prefer)",
              cand.candidate_id)
    if not provenance.get("panel_not_isolated"):
        return
    _flag(out, "panel_not_isolated",
          str(provenance.get("needs_review_reason")
              or "the panel this cell names could not be isolated from its neighbours, so the "
                 "reading was made on the whole figure"),
          cand.candidate_id)


#: `digitizer.CATEGORICAL_POINT_AT_CATEGORY`, spelled here so this layer does not import the
#: digitiser (and, with it, OpenCV) to read one word out of a provenance dict. The pair is pinned
#: by `tests/test_digitizer.py::test_a_point_at_category_read_is_capped_and_says_why`, which
#: builds the provenance from the digitiser's own constant and asserts this check fires on it.
CATEGORICAL_POINT_AT_CATEGORY = "point_at_category"
#: `digitizer.CATEGORICAL_UNRESOLVED`, spelled here for the same no-OpenCV-import reason.
CATEGORICAL_UNRESOLVED = "unknown"


def _check_categorical_x(cand: Candidate, out: list[CheckFlag]) -> None:
    """A value averaged across a categorical x axis, read at one category of it, or the refusal
    to invent either (task 16 P6, D3)."""
    provenance = cand.pixel_provenance or {}
    if provenance.get("categorical_x_role") == CATEGORICAL_POINT_AT_CATEGORY:
        _flag(out, "categorical_point_read",
              f"this value is the single plotted point at the x category this cell's locator "
              f"names, not an average across the axis "
              f"({provenance.get('categorical_x_role_why') or 'the locator named one category'})"
              f" — one point per group, so nothing on the axis corroborates it",
              cand.candidate_id)
    if provenance.get("categorical_x_resolved_from_readings"):
        support = int(provenance.get("categorical_x_role_support") or 0)
        role = str(provenance.get("categorical_x_role") or "")
        why = str(provenance.get("categorical_x_role_why") or "")
        tension = str(provenance.get("categorical_x_hint_tension") or "")
        if support >= 2 and not tension:
            _flag(out, "categorical_x_resolved_from_readings",
                  f"what this figure's categorical x axis IS ({role!r}) was resolved from the "
                  f"readers' own category reports — {support} readings support the ruling on "
                  f"their own ({why})",
                  cand.candidate_id)
        else:
            _flag(out, "categorical_x_single_witness",
                  (f"what this figure's categorical x axis IS ({role!r}) rests on a single "
                   f"reading's category report ({why})"
                   + (f"; {tension}" if tension else "")),
                  cand.candidate_id)
    if provenance.get("categorical_x_unsupported"):
        _flag(out, "categorical_x_unsupported",
              str(provenance.get("needs_review_reason")
                  or "this figure's x axis is categorical and the collapse mode is off"),
              cand.candidate_id)
        return
    if (provenance.get("collapse_across_x") and cand.mean is None
            and str(provenance.get("categorical_x_role") or "") in ("", CATEGORICAL_UNRESOLVED)):
        # the collapse was PERMITTED but the role never resolved and no value came out — the same
        # open question as the refusal above, and it must raise the same card rather than a blank
        # "type the number" one (found on a real run: a cell nulled this way carried no flag at
        # all, so the review page could only ask "where is it?")
        _flag(out, "categorical_x_unsupported",
              str(provenance.get("categorical_x_role_why")
                  or "the x axis is categorical and nothing settled whether its categories are "
                     "conditions to average across or the groups themselves"),
              cand.candidate_id)
        return
    if provenance.get("collapsed_across_x"):
        _flag(out, "collapsed_across_x",
              f"this value is the average of {provenance.get('n_points')} points across a "
              f"categorical x axis, and its spread is "
              f"{provenance.get('dispersion_approximation') or 'approximated'} — the SD of one "
              f"point, not of a participant's mean across them, so it overstates the denominator "
              f"unless the between-point variance is fully shared",
              cand.candidate_id)


def _check_statistic(cand: Candidate, dataset: DatasetSpec, out: list[CheckFlag]) -> None:
    """Spec §3.3(1): a t or F may only be used with degrees of freedom and both group sizes."""
    if cand.status != "found" or cand.kind != "test_statistic":
        return
    if cand.stat_type not in ("t", "F"):
        return
    if cand.df is None and cand.df1 is None and cand.df2 is None:
        sizes = f"n = {dataset.group_a.n}/{dataset.group_b.n}"
        _flag(out, "test_stat_missing_df",
              f"{cand.stat_type} = {cand.stat_value} was printed without degrees of freedom, so "
              f"the design behind it cannot be checked against the analysed group sizes ({sizes})",
              cand.candidate_id)


def _error_df(cand: Candidate) -> float | None:
    """The degrees of freedom a t or F contrast is tested on: `df`, or an F's ERROR df (`df2`)."""
    if cand.df is not None:
        return float(cand.df)
    if cand.stat_type == "F" and cand.df2 is not None:
        return float(cand.df2)
    return None


def _shortfall_is_explained(dataset: DatasetSpec) -> str:
    """Why a df that is not exactly n_a + n_b - 2 may still be this contrast (C9), or `""`.

    ONE reason, because it is the only one anything on the record can carry: an n that was never
    printed. The mapper inferred it, so `n_a + n_b - 2` is itself an estimate and demanding
    equality against it would refuse papers for the extractor's uncertainty.

    C9 licensed a second — a participant total the paper STATES, larger than the analysed group
    sizes, so at least `gap` people were excluded after being tested. That branch was deleted
    rather than left standing (review M2): nothing in a `StudyMap` records a stated participant
    total, and the orchestrator's only call passed `n_a + n_b` back in as one, which made
    `excluded` identically zero. A documented rule wired to a constant is worse than an absent
    one — it reads as live in the code and in the acceptance test, and refuses papers in the run.
    Restoring it is MAPPER work: a field for the participant total the paper prints, extracted
    with its quote, and then this function takes it again. Until that exists, `t(37)` at
    n = 20/20 with a stated single dropout is REFUSED — fail-closed, a fill-rate cost and never a
    wrong number.

    Everything else — and in particular a paper that simply prints df two below what its own
    stated group sizes imply — is NOT explained. `F(1,36)` at n = 20/20 is the shape of a
    two-covariate ANCOVA reported as a one-way, and the ±2 tolerance admitted it silently.
    """
    if not (dataset.group_a.n_evidence.strip() and dataset.group_b.n_evidence.strip()):
        return ("at least one analysed group size was inferred rather than printed, so "
                "n_a + n_b - 2 is itself an estimate")
    return ""


#: how a t, an F and a p rank when a cell prints more than one: a t is the most direct statement
#: of the contrast, a p the least (it has lost the statistic's own precision)
_STAT_RANK = {"t": 0, "F": 1, "p": 2}


def best_statistic(candidates: Sequence[Candidate]) -> Candidate | None:
    """The one statistic a row would be converted FROM, or `None` when a cell prints none.

    One selection rule, in one place, used by both the check that screens a statistic and the
    orchestrator that resolves the row from it (`canopy.pipeline.run._statistic_values`). Two
    copies is how the gate came to hold a cell for the degrees of freedom of a statistic the row
    never used: a paper printing an unrelated one-way F beside the outcome raised `df_missing` on
    a cell that resolves from a printed t (review L4).
    """
    stats = [c for c in candidates
             if c.kind == "test_statistic" and c.status == "found" and c.admissible]
    ranked = sorted(stats, key=lambda c: (_STAT_RANK.get(str(c.stat_type), 3), c.candidate_id))
    return next((c for c in ranked if c.stat_value is not None or c.p_value is not None), None)


def _check_conversion_gate(candidates: Sequence[Candidate], dataset: DatasetSpec,
                           out: list[CheckFlag]) -> None:
    """C9: price the conversion gate's own flags, on the cell, before any statistic route runs.

    `canopy.stats.effect_sizes.convertibility` answers "may this become an SMD?" while the effect
    size is being built — long after this module has scored the two cells, and its answer reaches
    the row as a flag string that changes no bucket. Two of its outcomes are not bookkeeping:

    * **no degrees of freedom at all.** `convertibility` returns ok with `df_missing`. But a paper
      printing "t = 5.25, p < .001" as a post-hoc from a three-group ANOVA prints exactly that,
      and nothing in the number says which two groups it compares. Unverified provenance to THESE
      two groups may not pool, so it is an error here.
    * **a df that contradicts the analysed group sizes.** Exact equality, unless the shortfall is
      explained (`_shortfall_is_explained`) — in which case it is admitted, named with the size of
      the gap, and capped by `confidence.conversion_gate_bucket` rather than accepted.
    """
    expected: float | None = None
    if dataset.group_a.n is not None and dataset.group_b.n is not None:
        expected = float(dataset.group_a.n + dataset.group_b.n - 2)
    # only the statistic the ROW would convert from, not every statistic the paper prints near
    # this outcome (review L4) — a cell held for a number nothing would have used is a fill-rate
    # cost with no correctness behind it
    for cand in [c for c in [best_statistic(candidates)] if c is not None]:
        if cand.stat_type not in ("t", "F") or cand.design not in CONVERTIBLE_DESIGNS:
            continue
        df = _error_df(cand)
        if df is None:
            _flag(out, "df_missing",
                  f"{cand.stat_type} = {cand.stat_value} is printed with no degrees of freedom, "
                  f"so nothing establishes that it is the comparison of these two groups rather "
                  f"than a post-hoc from a larger analysis", cand.candidate_id)
            continue
        if expected is None:
            continue
        gap = abs(df - expected)
        if gap == 0:
            continue
        why = _shortfall_is_explained(dataset)
        if why and gap <= DF_SHORTFALL_TOLERANCE:
            _flag(out, f"df_off_by_{gap:g}",
                  f"the printed degrees of freedom ({df:g}) are {gap:g} from n_a + n_b - 2 = "
                  f"{expected:g}, which this paper explains: {why}", cand.candidate_id)
        else:
            _flag(out, "df_shortfall_unexplained",
                  f"the printed degrees of freedom ({df:g}) do not equal n_a + n_b - 2 = "
                  f"{expected:g} and the paper explains no shortfall, so this statistic was not "
                  f"computed on these two groups as analysed", cand.candidate_id)


def _dispersion_conflict(cand: Candidate, outcome: OutcomeSources | None
                         ) -> tuple[DispersionType, str] | None:
    """The mapper's error-bar determination for this location against what was transcribed."""
    if outcome is None or cand.dispersion_type in (DispersionType.UNKNOWN, DispersionType.NONE):
        return None
    for source in outcome.sources:
        if not _same_location(cand, source):
            continue
        mapped = source.error_bar_type
        if mapped in (DispersionType.UNKNOWN, DispersionType.NONE) or mapped is cand.dispersion_type:
            continue
        return mapped, source.locator or f"page {source.page}"
    return None


def _same_location(cand: Candidate, source: Source) -> bool:
    figure_id = (cand.pixel_provenance or {}).get("figure_id")
    if source.figure_id and figure_id and source.figure_id == figure_id:
        return True
    if source.table_id and source.table_id == (cand.pixel_provenance or {}).get("table_id"):
        return True
    return cand.page is not None and cand.page == source.page


# ----------------------------------------------------------------------------- across candidates
def _check_se_against_sd(found: Sequence[Candidate], out: list[CheckFlag]) -> None:
    """SE·√n ≈ SD when one reader called the spread SE and another called it SD."""
    for group in ("A", "B"):
        rows = [c for c in found if c.group == group and c.dispersion_value is not None
                and c.n is not None and c.n > 0]
        sds = [c for c in rows if c.dispersion_type is DispersionType.SD]
        ses = [c for c in rows if c.dispersion_type is DispersionType.SE]
        for sd_cand in sds:
            for se_cand in ses:
                if se_cand.n != sd_cand.n:
                    continue
                implied = se_cand.dispersion_value * math.sqrt(se_cand.n)
                if not _close(implied, sd_cand.dispersion_value, SE_SD_TOLERANCE):
                    _flag(out, "se_sd_inconsistent",
                          f"group {group}: SE {se_cand.dispersion_value} × √{se_cand.n} = "
                          f"{implied:.4g} does not match the SD {sd_cand.dispersion_value} read "
                          f"from the same paper", sd_cand.candidate_id, se_cand.candidate_id)


def _check_units(found: Sequence[Candidate], outcome: OutcomeSources | None,
                 out: list[CheckFlag]) -> None:
    from .units import unit_key

    units = {unit_key(c.unit): c for c in found if unit_key(c.unit)}
    mapped = unit_key(outcome.units) if outcome is not None else ""
    if len(units) > 1:
        # "another expression" only when EVERY group that was read in another unit was also read
        # in the recorded one — the same bars off both axes. A group read only in mm beside a
        # group read only in deg is a disagreement, whatever the map says.
        by_group: dict[str | None, set[str]] = {}
        for c in found:
            if unit_key(c.unit):
                by_group.setdefault(c.group, set()).add(unit_key(c.unit))
        same_bars = mapped and mapped in units and all(
            mapped in keys for keys in by_group.values() if keys - {mapped})
        if same_bars:
            # the map says which unit this outcome is in, and readings in it exist; the others
            # are the same bars read off another axis (Cressman's Fig. 3b: degrees on the left,
            # percent of the perturbation on the right) — another expression of the quantity,
            # which the vote sets aside. That is a note for the record, not a disagreement.
            others = sorted(c.unit for k, c in units.items() if k != mapped)
            _flag(out, "unit_other_expression",
                  f"some readings are in {others} where the outcome is recorded in "
                  f"{outcome.units!r}; the vote keeps the recorded unit and sets the others "
                  f"aside as another expression of the same quantity",
                  *sorted(c.candidate_id for k, c in units.items() if k != mapped))
        else:
            _flag(out, "unit_mismatch",
                  f"the readers disagree about the unit of this outcome: "
                  f"{sorted(c.unit for c in units.values())}",
                  *sorted(c.candidate_id for c in units.values()))
    elif units and mapped and mapped not in units:
        only = next(iter(units.values()))
        _flag(out, "unit_mismatch",
              f"values were transcribed in {only.unit!r}, but the map recorded this outcome in "
              f"{outcome.units!r}", *sorted(c.candidate_id for c in units.values()))


def _check_metric(found: Sequence[Candidate], others: Sequence[Candidate],
                  out: list[CheckFlag]) -> None:
    """Amendment G: an endpoint and a change from baseline are not the same quantity.

    Scoped per (paper, OUTCOME), because a paper whose outcomes are legitimately different metrics
    is not a problem — Cressman's late adaptation is an endpoint and its aftereffect a
    baseline-subtracted difference, and warning about that penalised a correct map. What IS a
    problem is one outcome fed by two different metrics: within `aftereffect`, Fig 3b (degrees,
    endpoint) and Fig 5 (`% Visuomotor Adaptation`, a correlation scatter) are both sources of the
    same number.

    The cross-outcome comparison is kept, demoted to `info`: it is the only thing that would catch
    a paper whose outcomes silently drift metric, and deleting it costs that.
    """
    rows = [c for c in found if c.analysis_metric != "unknown"]
    metrics = sorted({c.analysis_metric for c in rows})
    if len(metrics) > 1:
        _flag(out, "metric_mixed",
              f"this outcome's values mix analysis metrics ({', '.join(metrics)}), which are not "
              f"the same quantity", *sorted(c.candidate_id for c in rows))
    across = [c for c in list(found) + list(others) if c.analysis_metric != "unknown"]
    other_metrics = sorted({c.analysis_metric for c in across})
    if len(other_metrics) > len(metrics):
        _flag(out, "metric_mixed_across_outcomes",
              f"this paper's outcomes are read on different analysis metrics "
              f"({', '.join(other_metrics)}) — legitimate when the outcomes really are different "
              f"quantities, worth a look when they are not",
              *sorted(c.candidate_id for c in across))


#: two readings of one datum a factor of ten apart are not two readings of one datum: one of them
#: is in another unit, off another axis, or off another panel
_UNIT_DECADE = 10.0
#: a spread this many times its own mean, next to a route whose spread is proportionate, is a mean
#: and a dispersion recorded in different units on the same candidate
_SPREAD_OVER_MEAN = 20.0


def _check_unit_coherence(found: Sequence[Candidate], out: list[CheckFlag]) -> None:
    """Is one cell's arithmetic internally consistent — mean against mean, mean against spread?

    `_check_units` compares the unit STRINGS the readers wrote down, which agree perfectly while
    the numbers do not: Cressman's aftereffect group A carries read-outs at 61.6 and 62.0, a
    `vlm_coords` at 0.0610 and a `vector` at 0.0383 whose dispersion is 2.106 — a mean in one unit
    and a spread in another, on one candidate, all labelled "degrees (deg)".
    """
    for group in ("A", "B"):
        rows = [c for c in found if c.group == group and c.mean is not None and c.mean != 0]
        by_size = sorted(rows, key=lambda c: (abs(c.mean), c.candidate_id))
        if len(by_size) >= 2 and abs(by_size[0].mean) > 0:
            small, large = by_size[0], by_size[-1]
            ratio = abs(large.mean) / abs(small.mean)
            if ratio >= _UNIT_DECADE:
                _flag(out, "unit_incoherent",
                      f"group {group}: two readings of the same value are a factor of "
                      f"{ratio:.0f} apart ({small.mean} and {large.mean}) — that is a unit, an "
                      f"axis or a panel, not a reading error",
                      small.candidate_id, large.candidate_id)
        proportionate = [c for c in rows if c.dispersion_value is not None
                         and abs(c.dispersion_value) <= 2.0 * abs(c.mean)]
        for cand in rows:
            spread = cand.dispersion_value
            if spread is None or abs(spread) <= _SPREAD_OVER_MEAN * abs(cand.mean):
                continue
            if not proportionate:
                continue
            _flag(out, "unit_incoherent",
                  f"group {group}: this reading pairs a mean of {cand.mean} with a spread of "
                  f"{spread}, while another route of the same cell reads a spread proportionate "
                  f"to its mean — the mean and the spread are not in the same unit",
                  cand.candidate_id)
    for cand in found:
        _check_route_coherence(cand, out)


def _route_means(pixel_provenance: dict | None) -> list[float]:
    """The per-route means behind one digitised candidate, however that candidate recorded them."""
    provenance = pixel_provenance if isinstance(pixel_provenance, dict) else {}
    values: list[float] = []
    for row in provenance.get("per_route") or []:
        if isinstance(row, dict) and isinstance(row.get("mean"), (int, float)):
            values.append(float(row["mean"]))
    if values:
        return values
    routes = provenance.get("route_values")
    if isinstance(routes, dict):
        for row in routes.values():
            if isinstance(row, dict) and isinstance(row.get("mean"), (int, float)):
                values.append(float(row["mean"]))
    return values


def _check_route_coherence(cand: Candidate, out: list[CheckFlag]) -> None:
    """The routes BEHIND one digitised value, against each other — a decade apart is a unit.

    An ensemble whose routes span a factor of a thousand (Cressman's Fig. 5 group A: 61.6, 62.0,
    50.0 and 0.061, 0.038) is not a noisy read: some routes answered in percent and some in
    fractions, or off two different axes. The ensemble already reports that as a disagreement;
    this says WHAT KIND, which is what tells a reviewer where to look.
    """
    means = [m for m in _route_means(cand.pixel_provenance) if m != 0]
    if len(means) < 2:
        return
    low, high = min(abs(m) for m in means), max(abs(m) for m in means)
    if low <= 0 or high / low < _UNIT_DECADE:
        return
    _flag(out, "unit_incoherent",
          f"the routes behind this value span a factor of {high / low:.0f} "
          f"({', '.join(f'{m:g}' for m in sorted(means))}) — some of them answered in another "
          f"unit or off another axis, which is not a disagreement a median can settle",
          cand.candidate_id)


def _check_duplicates(found: Sequence[Candidate], others: Sequence[Candidate],
                      out: list[CheckFlag]) -> None:
    """The same mean and spread under two outcomes means one of the two was read off the wrong row."""
    for cand in found:
        if cand.mean is None:
            continue
        for other in others:
            if other.outcome_key == cand.outcome_key or other.status != "found":
                continue
            if other.mean is None or not math.isclose(other.mean, cand.mean, rel_tol=1e-9):
                continue
            if (cand.dispersion_value is None) != (other.dispersion_value is None):
                continue
            if cand.dispersion_value is not None and not math.isclose(
                    cand.dispersion_value, other.dispersion_value, rel_tol=1e-9):
                continue
            _flag(out, "duplicate_across_outcomes",
                  f"the same value ({cand.mean} ± {cand.dispersion_value}) is also recorded for "
                  f"outcome {other.outcome_key!r}", cand.candidate_id, other.candidate_id)


def _check_effect_size(found: Sequence[Candidate], dataset: DatasetSpec,
                       out: list[CheckFlag]) -> None:
    """Spec §3.3(1) + C9: |d| ≤ 3 — computed here to screen the DENOMINATOR, never stored.

    The flag names the dispersions and not the effect, because the effect is not what is wrong.
    Two means eight standard deviations apart are almost never two means eight standard deviations
    apart: they are two means divided by an error bar that is a standard error, or a within-subject
    normalisation, or the other group's. Which is why this fires whatever the route agreement
    says — six readers can agree perfectly on a number that was divided by the wrong thing.
    """
    usable: dict[str, tuple[Candidate, float]] = {}
    for group in ("A", "B"):
        rows = [(c, sd) for c in found if c.group == group and c.mean is not None
                for sd in [_sd_like(c)] if sd is not None]
        if rows:
            usable[group] = rows[0]
    if set(usable) != {"A", "B"}:
        return
    (a, sd_a), (b, sd_b) = usable["A"], usable["B"]
    try:
        d = cohens_d(a.mean, sd_a, a.n, b.mean, sd_b, b.n)
    except ValueError:                                   # pragma: no cover - guarded above
        return
    if abs(d) > MAX_PLAUSIBLE_D:
        _flag(out, "implausible_dispersion",
              f"these values imply |d| = {abs(d):.2f}, above the plausibility threshold of "
              f"{MAX_PLAUSIBLE_D:g}. The suspect number is the denominator: the pooled SD of "
              f"{sd_a:g} (group A) and {sd_b:g} (group B) against "
              f"means of {a.mean:g} and {b.mean:g}. An SE printed as an SD, a within-subject "
              f"error bar or the other group's spread all look exactly like this",
              a.candidate_id, b.candidate_id)


def _sd_like(cand: Candidate) -> float | None:
    """This candidate's spread as a standard deviation, when it IS one or converts arithmetically.

    An SE with an n converts exactly (`SD = SE × √n`) and costs nothing to include — and SE is the
    modal shape in the live corpus (32 against 25 SD among the 57 `found` group_stats candidates
    that carry either), so a screen that looked only at `SD` was blind on the majority of real
    readings (review H1). Everything else (IQR, range, an interval) needs a distributional
    assumption `canopy.stats.conversions` makes on the ROW, where the binding screen now is.
    """
    value, n = cand.dispersion_value, (cand.n or 0)
    if value is None or value <= 0 or n < MIN_N:
        return None
    if cand.dispersion_type is DispersionType.SD:
        return float(value)
    if cand.dispersion_type is DispersionType.SE:
        return float(value) * math.sqrt(n)
    return None


def sign_check(direction: str, mean_a: float | None, mean_b: float | None,
               candidate_ids: Sequence[str] = ()) -> CheckFlag | None:
    """Spec §3.3(5): the direction the paper *states* against the sign the numbers imply.

    `direction` is the paper's own words about which group scored higher on the RAW measure, so it
    is compared with the raw difference — before `orient()`, which is about the measure, not about
    the result. Returns `None` when the paper said nothing or a mean is missing.
    """
    if direction not in ("a_greater", "b_greater") or mean_a is None or mean_b is None:
        return None
    if mean_a == mean_b:
        # identical means carry no direction to contradict: the effect size is zero, and telling a
        # reviewer the SIGN is inverted would be wrong. A stated difference that the extracted
        # numbers do not show at all is a magnitude problem, and the vote and the verifier are
        # what catch it.
        return None
    stated_a_greater = direction == "a_greater"
    if (mean_a > mean_b) != stated_a_greater:
        return CheckFlag(
            code="sign_mismatch", severity=CHECK_SEVERITY["sign_mismatch"],
            message=(f"the paper states {direction.replace('_', ' ')} on this measure, but the "
                     f"extracted means say A = {mean_a} and B = {mean_b}"),
            candidate_ids=[cid for cid in candidate_ids if cid])
    return None


def _best_mean(found: Sequence[Candidate], group: str) -> Candidate | None:
    rows = [c for c in found if c.group == group and c.mean is not None]
    return rows[0] if rows else None


# ----------------------------------------------------------------------------- outcome level
def _check_outcome(outcome: OutcomeSources | None, orientation: OrientationVerdict | None,
                   out: list[CheckFlag]) -> None:
    higher_is_better = outcome.higher_is_better if outcome is not None else None
    if orientation is not None:
        higher_is_better = orientation.higher_is_better
    if higher_is_better is None:
        _flag(out, "orientation_unknown",
              "nobody has established the direction of this measure (does a larger raw value mean "
              "more of the construct?), so the sign of any effect size is undecided")
    if orientation is not None:
        for code in ORIENTATION_FLAGS:              # what `combine_orientation` already decided
            marker = f"[{code}]"
            if marker in (orientation.notes or ""):
                said = [part.split(marker, 1)[1].strip()
                        for part in (orientation.notes or "").split("; ") if marker in part]
                _flag(out, code, said[0] if said and said[0] else code.replace("_", " "))
    if outcome is None:
        return
    for source in outcome.sources:
        if source.kind not in _FIGURE_KINDS:
            continue
        where = source.locator or f"page {source.page}"
        if source.error_bar_type is DispersionType.UNKNOWN:
            _flag(out, "figure_error_bar_unknown",
                  f"the error bars at {where} were never identified, so a digitised spread there "
                  f"cannot be converted")
        elif source.error_bar_agreement != "agreed":
            _flag(out, "error_bar_unconfirmed",
                  f"only one agent determined that the error bars at {where} are "
                  f"{source.error_bar_type.value} ({source.error_bar_agreement})")


# ----------------------------------------------------------------------------- entry point
# --------------------------------------------------- D4-lite: an n printed before the exclusions
#: How far past a printed group size an exclusion sentence may sit and still be about it. The real
#: case (Vachon 2020) prints the four group sizes in one paragraph and the exclusions in the next,
#: and a PDF's text layer has no paragraph marks worth trusting — so "the same paragraph or the
#: next one" is measured in characters, which is a rule that reads the same on every paper.
EXCLUSION_SPAN = 1200
#: …and how far from the exclusion cue a number may sit and still be ITS count. Wide enough for
#: "excluded 4 younger (all from the non-instructed group) and 3 older", narrow enough that the
#: next sentence's numbers are not read as exclusions.
EXCLUSION_COUNT_SPAN = 60

#: a size the paper PRINTS: "n = 20", or "20 younger" / "38 older adults" / "12 participants".
#: Spelled-out counts ("Forty-one younger adults were recruited") are deliberately not matched:
#: this check only ever speaks about a number a reader can see beside the group it belongs to.
#:
#: The words are NOT a fixed list. `young|old|healthy` are this review's arm vocabulary, and a
#: check that carries them is a check that reads one study: "Twenty PD patients (20 patients)…
#: 3 controls were excluded" found nothing at all, and "20 younger… excluded 4 younger" fired
#: (review finding 8). So the pattern is built per call from the person nouns plus the words the
#: protocol itself uses for THIS arm — the same vocabulary `excluded_count` already reads.
_ANY_SIZE = r"\bn\s*=\s*(\d+)\b"


def _group_size(group_terms: Sequence[str]) -> re.Pattern[str]:
    """`"n = 20"`, or a count standing beside a word this review calls its people by."""
    words = _PERSON_NOUNS | _group_words(group_terms)
    return _size_pattern(frozenset(words))


@lru_cache(maxsize=None)
def _size_pattern(words: frozenset[str]) -> re.Pattern[str]:
    beside = "|".join(re.escape(word) for word in sorted(words, key=lambda w: (-len(w), w)))
    return re.compile(rf"{_ANY_SIZE}|\b(\d+)\s+(?:{beside})\w*", re.I) if beside \
        else re.compile(_ANY_SIZE, re.I)


#: a size sentence that is already the ANALYSED one — the paper has done the subtraction, and
#: doing it again offers the reviewer a number lower than either ("the final sample comprised 20
#: … 4 did not complete" → "may be 16", review MINOR 25). The check says nothing about such a
#: size rather than saying something wrong about it.
_ALREADY_ANALYSED = re.compile(
    r"\b(?:final|analys\w*|analyz\w*|remain\w*|included in the analys\w*|after (?:the )?"
    r"exclusion\w*)\b", re.I)
#: the words a paper takes people OUT with
_EXCLUSION_CUE = re.compile(
    r"\b(?:exclud|remov|withdrew|withdrawn|drop(?:ped)?[- ]?out|discontinued|did not complete"
    r"|were not included|data (?:were|was) lost)\w*", re.I)
#: "4 younger", "153 of" — a count and the word immediately after it
_COUNTED = re.compile(r"\b(\d+)\s+([A-Za-z][\w-]*)")
#: "…, 4 were excluded" / "…, 4 participants were excluded" — the count that is the cue's own
#: subject. Group 2 is the word immediately after the count, and it is what decides whether the
#: count is a count of PEOPLE: "5 trials were excluded" is the same grammar about a different
#: noun (Heuer 2008 prints exactly that), and reading it as an arm is the one mistake this whole
#: function exists to avoid.
_COUNT_INTO_CUE = re.compile(r"\b(\d+)\s+([A-Za-z][\w-]*)(?:\s+\w+)?\s*$")
#: the followers that make a bare count a count of people: a person noun, or the auxiliary of the
#: cue's own verb ("4 were excluded"). It overlaps `_GENERIC_GROUP_WORDS` below and the two sets
#: do opposite jobs — there these words are useless because BOTH arms share them, here they are
#: the whole evidence that the number counts participants at all.
#: …the NOUNS of it. A printed size stands beside a noun ("20 patients"), never beside the
#: auxiliary — "20 were excluded" is a loss, not a group size — so the size pattern above takes
#: this half and `excluded_count` below takes both.
_PERSON_NOUNS: frozenset[str] = frozenset({
    "adult", "adults", "participant", "participants", "subject", "subjects", "person", "people",
    "volunteer", "volunteers", "patient", "patients"})
_PERSON_WORDS: frozenset[str] = _PERSON_NOUNS | frozenset({"were", "was", "had"})
#: words every arm of every review shares, so a count standing next to one says nothing about
#: WHICH group lost it. They are dropped from a group's vocabulary before the count is read.
_GENERIC_GROUP_WORDS: frozenset[str] = frozenset({
    "adult", "adults", "participant", "participants", "subject", "subjects", "person", "people",
    "volunteer", "volunteers", "patient", "patients", "control", "controls", "group", "groups",
    "arm", "arms", "sample", "samples", "and", "the", "of", "who", "were", "was"})


def _group_words(group_terms: Sequence[str]) -> frozenset[str]:
    """The words that identify THIS arm, out of the review's names for it.

    A dataset's label is a phrase ("non-instructed older adults"), and the words it shares with
    the other arm — "adults", "participants", "group" — cannot tell one arm's exclusion count from
    the other's. What is left is the arm's own vocabulary: "older", "young", "healthy".
    """
    words = {word for term in group_terms for word in re.split(r"[^\w]+", str(term or "").casefold())
             if len(word) > 2}
    return frozenset(words - _GENERIC_GROUP_WORDS)


def _sentence(text: str, start: int, end: int) -> str:
    """The sentence the exclusion was written in, whitespace normalised (a PDF wraps mid-phrase)."""
    left = text.rfind(".", 0, start) + 1
    right = text.find(".", end)
    return " ".join(text[left:(right + 1) if right != -1 else len(text)].split())


def excluded_count(text: str, group_terms: Sequence[str]) -> tuple[int | None, str, str]:
    """How many of THIS arm an exclusion sentence in `text` names — `(count, phrase, sentence)`.

    The count has to stand NEXT TO one of the arm's own words: Heuer 2008 reports that "153 of
    13,920 trials (1.1%) were excluded" for the younger group, and every part of that sentence
    except the number itself is a participant exclusion. Requiring "<count> <arm word>" is what
    separates the two, and it is also what makes the answer usable — a card that offers
    "recruited − excluded" must have parsed a count that belongs to the group it is offering it
    for. When no such pair sits beside the cue this returns `(None, "", "")` and the caller says
    nothing: a cue on its own is not evidence about this arm.

    **The cue is read forwards first.** English writes the exclusion after the word that announces
    it — "we excluded 4 younger participants" — while what sits BEFORE the cue is very often the
    size being excluded FROM: "Of the 20 younger participants who were recruited, 4 were excluded"
    put `20 younger` next to the cue and the first cut of this function read it as the exclusion
    count. So the window from the cue onwards is scanned on its own, and the 60 characters before
    it are a fallback consulted only when nothing followed the cue at all. The remaining ambiguity
    — a recruited size read out of the fallback — is what `n_before_exclusions`' `0 < count <
    printed` guard is for: a count that is not a strict part of the size it is taken from is not
    an exclusion count, whatever it stands next to.

    A bare count read out of the fallback must still be a count of PEOPLE: the word after it has
    to be a person noun, an arm word, or the auxiliary of the cue's own verb. Without that test
    "In the younger group's session, 5 trials were excluded" reads as five lost participants —
    the trial-versus-participant confusion this function was written to refuse, arriving through
    the back door.
    """
    words = _group_words(group_terms)
    if not words:
        return None, "", ""
    for cue in _EXCLUSION_CUE.finditer(text):
        before = max(0, cue.start() - EXCLUSION_COUNT_SPAN)
        ahead, behind = (text[cue.start():cue.end() + EXCLUSION_COUNT_SPAN],
                         text[before:cue.start()])
        # 1. the ordinary shape, read forwards: "we excluded 4 younger participants"
        for match in _COUNTED.finditer(ahead):
            if match.group(2).casefold() in words:
                return (int(match.group(1)), match.group(0).strip(),
                        _sentence(text, cue.start(), cue.start() + match.end()))
        # 2. the count that IS the cue's own subject: "…, 4 were excluded". Two conditions, and
        #    both are load-bearing: the word after the count must name PEOPLE or be the auxiliary
        #    of the cue's verb (so "5 trials were excluded" is not an arm's loss), and the same
        #    window must name THIS arm (so a clause about the other group cannot supply a number
        #    for this one). The arm test is on WORDS, not on substrings: "old" inside "household"
        #    is not this arm being named.
        into = _COUNT_INTO_CUE.search(behind)
        named = set(re.split(r"[^\w]+", behind.casefold())) & words
        if (into is not None and named
                and (into.group(2).casefold() in _PERSON_WORDS
                     or into.group(2).casefold() in words)):
            # the phrase is the bare count: the words between it and the cue are the sentence's
            # own grammar ("4 were excluded"), and quoting them back after "it excluded" would
            # read as nonsense. The sentence itself travels in the quote.
            return (int(into.group(1)), into.group(1),
                    _sentence(text, cue.start(), before + into.end()))
        # 3. and last, an arm-labelled count behind the cue. It is where a RECRUITED size hides
        #    ("Of the 20 younger participants…"), which is why it is the fallback and why the
        #    caller still has to check that what comes back is a strict part of the printed size.
        for match in _COUNTED.finditer(behind):
            if match.group(2).casefold() in words:
                return (int(match.group(1)), match.group(0).strip(),
                        _sentence(text, cue.start(), before + match.end()))
    return None, "", ""


def n_before_exclusions(pages: Sequence[str], cand: Candidate,
                        group_terms: Sequence[str]) -> CheckFlag | None:
    """D4-lite: is this candidate's `n` the size the paper RECRUITED, not the size it analysed?

    Deterministic, free, and it never repairs anything. The rule is three facts in a row, in the
    paper's own text: the paper prints this exact number as a group size; it says within the next
    `EXCLUSION_SPAN` characters that it excluded people; and a count beside that cue belongs to
    THIS arm. Vachon 2020 is the case it was written from — "non-instructed younger adults
    (n = 20, 14 female)" in one paragraph, "We excluded 4 younger … participants" in the next, and
    an analysed group of sixteen behind a row whose variance was computed from twenty.

    What it does NOT do is decide the analysed n. `recruited − excluded` is often right and is
    exactly what the review card offers, but a paper may break its own totals down further (Vachon
    excluded 3 older *across two datasets*), so the number a row is rebuilt with comes from a
    human's `group_n` answer and never from here. The flag caps the cell and asks.

    `group_terms` is the review's vocabulary for this candidate's arm (`run._group_vocabulary`);
    `pages` is the ingested page text, page 1 first.
    """
    if cand.n is None or cand.n < MIN_N or cand.group not in ("A", "B"):
        return None
    pattern = _group_size(group_terms)
    for text in pages:
        for size in pattern.finditer(text):
            printed = int(size.group(1) or size.group(2))
            if printed != cand.n:
                continue
            if _ALREADY_ANALYSED.search(_sentence(text, size.start(), size.end())):
                continue        # the paper has already taken its losses off this one (MINOR 25)
            count, phrase, quote = excluded_count(
                text[size.end():size.end() + EXCLUSION_SPAN], group_terms)
            if count is None:
                continue
            # …and the count must be a strict PART of the size it is taken out of. "Of the 20
            # younger participants who were recruited, 4 were excluded" stands the recruited size
            # next to this arm's own word, so a fallback read behind the cue can come back with
            # the size itself — and `recruited − excluded` would then offer the reviewer a group
            # of nobody, which `overrides._validate` refuses ("a group of nobody is an exclusion,
            # not a size"): a card option nobody can answer. The FINDING still stands, because it
            # rests on the printed size and the exclusion sentence, not on the subtraction; what
            # is dropped is the subtraction, so the card offers no `recruited_minus_excluded`.
            if not 0 < count < printed:
                count = None
            detail = {"recruited": printed, "excluded": count, "quote": quote}
            if count is None:
                return CheckFlag(
                    code="n_before_exclusions", severity=severity_of("n_before_exclusions"),
                    message=(f"n = {printed} is a group size this paper prints before its "
                             f"exclusions, and how many of THIS group were left out cannot be "
                             f"read from the sentence that reports them — so the analysed group "
                             f"may be smaller than {printed} by an amount only a reader can say "
                             f"— \"{quote}\""),
                    candidate_ids=[cand.candidate_id], detail=detail)
            return CheckFlag(
                code="n_before_exclusions", severity=severity_of("n_before_exclusions"),
                message=(f"n = {printed} is a group size this paper prints before its exclusions: "
                         f"it then says it excluded {phrase}, so the analysed group may be "
                         f"{printed - count} rather than {printed} — \"{quote}\""),
                candidate_ids=[cand.candidate_id], detail=detail)
    return None


def run_checks(dataset: DatasetSpec, outcome_key: str, candidates: Sequence[Candidate], *,
               other_candidates: Sequence[Candidate] = (),
               orientation: OrientationVerdict | None = None,
               total_n: int | None = None) -> list[CheckFlag]:
    """Every consistency problem in one (dataset × outcome) cell, worst first.

    `other_candidates` are this paper's candidates for *other* outcomes — the cross-outcome rules
    (duplicated numbers, mixed analysis metric) need them. `orientation` overrides the mapper's
    reading of the measure's direction once Task 8's orientation agents have ruled. `total_n` is a
    participant total the paper states for this dataset, if one was found — nothing produces one
    today (review M2: no `StudyMap` field records it), so `n_sum_mismatch` waits for the mapper
    to grow one rather than being fed the analysed sizes back as if they were the paper's total.
    """
    outcome = _outcome(dataset, outcome_key)
    flags: list[CheckFlag] = []
    for cand in candidates:
        _check_one(cand, dataset, outcome, flags)

    found = _found_stats(candidates)
    others = [c for c in other_candidates if c.kind == "group_stats"]
    _check_se_against_sd(found, flags)
    _check_units(found, outcome, flags)
    _check_metric(found, others, flags)
    _check_unit_coherence(found, flags)
    _check_duplicates(found, others, flags)
    _check_effect_size(found, dataset, flags)
    _check_conversion_gate(candidates, dataset, flags)
    _check_outcome(outcome, orientation, flags)

    if orientation is not None:
        a, b = _best_mean(found, "A"), _best_mean(found, "B")
        mismatch = sign_check(orientation.direction_stated_in_text,
                              a.mean if a else None, b.mean if b else None,
                              [c.candidate_id for c in (a, b) if c is not None])
        if mismatch is not None:
            flags.append(mismatch)

    if total_n is not None:
        sizes = [n for n in (dataset.group_a.n, dataset.group_b.n) if n is not None]
        if len(sizes) == 2 and sum(sizes) != total_n:
            _flag(flags, "n_sum_mismatch",
                  f"the analysed group sizes sum to {sum(sizes)}, but the paper reports "
                  f"{total_n} participants for this dataset")

    flags.sort(key=lambda f: (SEVERITY_RANK[f.severity], f.code, f.candidate_ids))
    return flags
