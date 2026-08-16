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
from typing import Iterable, Sequence

from ..models import (Candidate, CheckFlag, DatasetSpec, DispersionType, GroupSpec,
                      OrientationVerdict, OutcomeSources, Source)
from ..stats.effect_sizes import cohens_d
from .figures import FIGURE_KINDS, axis_limits, is_figure
from .grounding import ROW_ONLY, SIGN_NOTE, is_short_quote

__all__ = ["run_checks", "sign_check", "codes", "CHECK_SEVERITY", "GROUP_LABEL_MISMATCH_NOTE",
           "ROW_ONLY_MARKER", "SIGN_NOTE_MARKER", "SEVERITY_RANK", "MIN_N", "MAX_PLAUSIBLE_D"]

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
    "metric_mixed": "warn",
    "dispersion_type_conflict": "warn",
    "duplicate_across_outcomes": "warn",
    "figure_n_mismatch": "warn",
    "points_undercount": "warn",
    # --- can it be used at all
    "orientation_unknown": "warn",
    "sign_mismatch": "error",
    "figure_error_bar_unknown": "warn",
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

    # --- a digitised value has to be inside the axis it was read from
    if cand.mean is not None:
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
    units = {_norm_unit(c.unit): c for c in found if _norm_unit(c.unit)}
    mapped = _norm_unit(outcome.units) if outcome is not None else ""
    if len(units) > 1:
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
    """Amendment G: an endpoint and a change from baseline are not the same quantity."""
    rows = [c for c in list(found) + list(others) if c.analysis_metric != "unknown"]
    metrics = sorted({c.analysis_metric for c in rows})
    if len(metrics) > 1:
        _flag(out, "metric_mixed",
              f"this paper's values mix analysis metrics ({', '.join(metrics)}), which are not "
              f"the same quantity", *sorted(c.candidate_id for c in rows))


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
