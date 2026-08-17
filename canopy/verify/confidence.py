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
from typing import Any, Iterable, NamedTuple, Sequence

from ..models import (Adjudication, Candidate, CheckFlag, ConfidenceBucket, DatasetSpec,
                      DispersionType, OrientationVerdict, VerifierVerdict, Verdict)
from ..stats.effect_sizes import cohens_d, pooled_sd, se_smd
from .checks import run_checks
from .grounding import is_short_quote
from .vote import VoteResult, digitizer_path, is_figure_route, modality, vote

__all__ = ["confidence", "resolve_cell", "figure_gate", "AUTO_ACCEPT", "ACCEPT_WITH_NOTE",
           "DELTA_D_LIMIT", "DIGITIZATION_SE_SHARE", "SINGLE_ROUTE_CAP", "ADJUDICATED_CAP"]

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
    #: extraction was re-opened on a source the verifier named, so a model chose the location
    "reopened_on_better_source",
    #: no ladder for the value axis could be built at all: the numbers rest on the readers' own
    #: sense of the scale, and nothing contradicts them either
    "calibration_missing",
    #: the error-bar type came from the figure's legend because the map never determined one
    "dispersion_type_from_legend",
    #: the reader's words about this series and the markers the pixel pass found do not line up.
    #: A soft doubt about a shape vocabulary — the *transposition* it used to share a code with
    #: is `series_transposed`, below, because the two have opposite consequences.
    "series_marker_mismatch",
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
CAP_REASONS: dict[str, str] = {
    "quote_row_only": ("the numbers are somewhere in the named table row but not in the column "
                       "this reading claims, so they may be the other group's"),
    "calibration_single_witness": ("only one witness calibrated this figure's axis, so the scale "
                                   "the value was read against is uncorroborated"),
    "calibration_refuted": ("the readers agree on a value the tick ladder cannot draw, so the "
                            "axis calibration was discarded and only the read-outs stand"),
    "calibration_missing": ("no calibration of this figure's value axis could be built at all, "
                            "so the scale these numbers were read against rests entirely on the "
                            "readers and nothing checked it"),
    "collapsed_across_x": ("this is an average across a categorical axis and its dispersion is an "
                           "approximation, not the paper's own"),
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


def _witness_families(result: VoteResult, candidates: Sequence[Candidate] | None) -> set[str]:
    """Every model family behind the agreeing readings — from the route keys and from inside them.

    A digitiser ensemble is one candidate however many models read the figure, so the families it
    ran are recorded in its provenance; a text or table route carries its family in its route key.
    An empty family is "not stated", never a second one.
    """
    families = {key.split("/", 1)[1] for key in (r.route_key for r in _agreeing_routes(result))}
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
        return "needs_human", 0.0, ["no value was resolved for this cell"]

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

    # --- the adversarial verifier
    kinds = {v.verdict for v in verdicts}
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
    if forced_human:
        return "needs_human", score, reasons
    if score >= AUTO_ACCEPT:
        return "auto_accept", score, reasons
    if score >= ACCEPT_WITH_NOTE:
        return "accept_with_note", score, reasons
    reasons.append(f"score {score:.2f} is below {ACCEPT_WITH_NOTE:.2f}")
    return "needs_human", score, reasons


# ----------------------------------------------------------------------------- resolve_cell
_VERIFIER_ORDER = {"refuted": 0, "ambiguous": 1, "confirmed": 2}


def _verifier_summary(verdicts: Sequence[VerifierVerdict], ids: set[str]) -> tuple[str, str]:
    relevant = [v for v in verdicts if not v.candidate_id or v.candidate_id in ids]
    if not relevant:
        return "not_run", ""
    worst = min(relevant, key=lambda v: _VERIFIER_ORDER.get(v.verdict, 1))
    return worst.verdict, worst.reason


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
        verdict.orientation_evidence = "; ".join(
            part for part in (orientation.reason, *orientation.quotes) if part)

    bucket, score, reasons = confidence(result, verdict.verifiers, flag_list, adjudication,
                                        candidates=list(candidates), n_a=n_a, n_b=n_b,
                                        orientation=orientation)
    verdict.confidence, verdict.confidence_score, verdict.confidence_reasons = bucket, score, reasons
    verdict.needs_human = bucket == "needs_human"
    return verdict


def _analysis_metric(candidates: Sequence[Candidate], agreeing_ids: Sequence[str]) -> str:
    agreeing = set(agreeing_ids)
    metrics = [c.analysis_metric for c in candidates
               if c.candidate_id in agreeing and c.analysis_metric != "unknown"]
    return metrics[0] if len(set(metrics)) == 1 and metrics else "unknown"
