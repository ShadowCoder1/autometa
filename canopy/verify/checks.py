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
from typing import Any, Iterable, Sequence

from ..models import (Candidate, CheckFlag, DatasetSpec, DispersionType, GroupSpec,
                      OrientationVerdict, OutcomeSources, Source)
from ..stats.effect_sizes import cohens_d
from .figures import (CAL_STATUSES, FIGURE_KINDS, axis_limits, calibration_status, is_figure,
                      routes_agree)
from .grounding import ROW_ONLY, SIGN_NOTE, is_short_quote

__all__ = ["run_checks", "sign_check", "codes", "CHECK_SEVERITY", "GROUP_LABEL_MISMATCH_NOTE",
           "ROW_ONLY_MARKER", "SIGN_NOTE_MARKER", "SEVERITY_RANK", "MIN_N", "MAX_PLAUSIBLE_D",
           "AXIS_TESTABLE"]

#: the marker `canopy.agents.extract_common.group_label_check` writes into a candidate's notes when
#: the label the extractor echoed belongs to the *other* group (tests/test_checks.py pins it).
GROUP_LABEL_MISMATCH_NOTE = "group label mismatch"
#: the exact shapes `grounding.check_table_cell` writes its two table-cell markers in — the bare
#: words would also match a reviewer's prose, so the punctuation that follows them is part of it
ROW_ONLY_MARKER = f"{ROW_ONLY}:"
SIGN_NOTE_MARKER = f"{SIGN_NOTE} for "

MIN_N = 2                       # a group of one has no within-group variance
MAX_PLAUSIBLE_D = 3.0           # |d| above this is nearly always a transcription error
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
    "mean_missing": "warn",
    "dispersion_unknown": "warn",
    "dispersion_missing": "warn",
    "test_stat_missing_df": "error",
    "effect_implausible": "warn",
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
    "duplicate_across_outcomes": "warn",
    "figure_n_mismatch": "warn",
    "reopened_on_better_source": "warn",
    "collapsed_across_x": "warn",
    "categorical_x_unsupported": "warn",
    "points_undercount": "warn",
    # --- can it be used at all
    "orientation_unknown": "warn",
    "sign_mismatch": "error",
    "figure_error_bar_unknown": "warn",
    "series_identity_conflict": "warn",
    "series_marker_mismatch": "warn",
    "series_transposed": "warn",
    "axis_conflict": "warn",
    "error_bar_unconfirmed": "info",
}

#: dispersion types that are a spread and must therefore be strictly positive
_POSITIVE_DISPERSIONS = frozenset({DispersionType.SD, DispersionType.SE, DispersionType.IQR,
                                   DispersionType.RANGE})
_INTERVALS = frozenset({DispersionType.CI95, DispersionType.CI90})
_FIGURE_KINDS = FIGURE_KINDS


def codes(flags: Iterable[CheckFlag]) -> list[str]:
    """The distinct codes in a flag list, sorted — what tests and the review queue read."""
    return sorted({flag.code for flag in flags})


# ----------------------------------------------------------------------------- small helpers
def _flag(out: list[CheckFlag], code: str, message: str, *candidate_ids: str) -> None:
    out.append(CheckFlag(code=code, severity=CHECK_SEVERITY[code], message=message,
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


def _check_categorical_x(cand: Candidate, out: list[CheckFlag]) -> None:
    """A value averaged across a categorical x axis, or the refusal to invent one (task 16 P6)."""
    provenance = cand.pixel_provenance or {}
    if provenance.get("categorical_x_unsupported"):
        _flag(out, "categorical_x_unsupported",
              str(provenance.get("needs_review_reason")
                  or "this figure's x axis is categorical and the collapse mode is off"),
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
    """Spec §3.3(1): |d| ≤ 3 — computed here only to flag a transcription error, never stored."""
    usable = {}
    for group in ("A", "B"):
        rows = [c for c in found
                if c.group == group and c.dispersion_type is DispersionType.SD
                and c.mean is not None and c.dispersion_value is not None
                and c.dispersion_value > 0 and (c.n or 0) >= MIN_N]
        if rows:
            usable[group] = rows[0]
    if set(usable) != {"A", "B"}:
        return
    a, b = usable["A"], usable["B"]
    try:
        d = cohens_d(a.mean, a.dispersion_value, a.n, b.mean, b.dispersion_value, b.n)
    except ValueError:                                   # pragma: no cover - guarded above
        return
    if abs(d) > MAX_PLAUSIBLE_D:
        _flag(out, "effect_implausible",
              f"these values imply |d| = {abs(d):.2f}, above the plausibility threshold of "
              f"{MAX_PLAUSIBLE_D:g} — usually a mis-read row or a wrong dispersion type",
              a.candidate_id, b.candidate_id)


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
def run_checks(dataset: DatasetSpec, outcome_key: str, candidates: Sequence[Candidate], *,
               other_candidates: Sequence[Candidate] = (),
               orientation: OrientationVerdict | None = None,
               total_n: int | None = None) -> list[CheckFlag]:
    """Every consistency problem in one (dataset × outcome) cell, worst first.

    `other_candidates` are this paper's candidates for *other* outcomes — the cross-outcome rules
    (duplicated numbers, mixed analysis metric) need them. `orientation` overrides the mapper's
    reading of the measure's direction once Task 8's orientation agents have ruled. `total_n` is a
    participant total the paper states for this dataset, if one was found.
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
