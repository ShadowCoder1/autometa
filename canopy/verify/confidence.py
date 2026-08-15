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
from typing import Iterable, Sequence

from ..models import (Adjudication, Candidate, CheckFlag, ConfidenceBucket, DatasetSpec,
                      DispersionType, OrientationVerdict, VerifierVerdict, Verdict)
from ..stats.effect_sizes import cohens_d, pooled_sd, se_smd
from .checks import run_checks
from .grounding import is_short_quote
from .vote import VoteResult, is_figure_route, modality, vote

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


# ----------------------------------------------------------------------------- amendment F gate
def _sd_of(cand: Candidate) -> float | None:
    """The candidate's spread as a standard deviation, when it is one or converts trivially."""
    if cand.dispersion_value is None or cand.dispersion_value <= 0:
        return None
    if cand.dispersion_type is DispersionType.SD:
        return cand.dispersion_value
    if cand.dispersion_type is DispersionType.SE and cand.n:
        return cand.dispersion_value * math.sqrt(cand.n)
    return None


def _per_route(candidates: Iterable[Candidate], group: str) -> dict[str, list[Candidate]]:
    out: dict[str, list[Candidate]] = {}
    for cand in candidates:
        if (cand.group == group and cand.kind == "group_stats" and cand.status == "found"
                and cand.mean is not None and _sd_of(cand) is not None):
            out.setdefault(modality(cand), []).append(cand)
    return out


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
    routes_a, routes_b = _per_route(candidates, "A"), _per_route(candidates, "B")
    shared = sorted(set(routes_a) & set(routes_b))
    effects: dict[str, float] = {}
    for route in shared:
        a, b = routes_a[route], routes_b[route]
        mean_a, sd_a = _median([c.mean for c in a]), _median([_sd_of(c) for c in a])
        mean_b, sd_b = _median([c.mean for c in b]), _median([_sd_of(c) for c in b])
        try:
            effects[route] = cohens_d(mean_a, sd_a, n_a, mean_b, sd_b, n_b)
        except ValueError:                                # pragma: no cover - guarded by _sd_of
            continue

    delta = (max(effects.values()) - min(effects.values())) if len(effects) >= 2 else None
    if delta is None:
        reasons.append(f"{len(effects)} digitizer route(s) read both groups, so the effect "
                       f"implied across routes could not be compared — one route cannot agree "
                       f"with another")
    elif delta >= DELTA_D_LIMIT:
        reasons.append(f"the digitizer routes imply effects that differ by {delta:.3f} across "
                       f"routes (limit {DELTA_D_LIMIT})")

    share = None
    if effects:
        d = _median(list(effects.values()))
        sds_a = [_sd_of(c) for row in routes_a.values() for c in row]
        sds_b = [_sd_of(c) for row in routes_b.values() for c in row]
        sigma_a = [c.sigma for row in routes_a.values() for c in row if c.sigma is not None]
        sigma_b = [c.sigma for row in routes_b.values() for c in row if c.sigma is not None]
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
    return vote_result.grounded, vote_result.short_quote


def _agreeing_routes(result: VoteResult) -> list:
    agreeing = set(result.agreeing_ids)
    return [r for r in result.routes if agreeing & set(r.candidate_ids)]


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

    # --- agreement
    if settled:
        reasons.append("the adjudicator settled a cell the vote could not, so the vote neither "
                       "credits nor penalises it")
    elif vote_result.agreement == "agree":
        extra = min(EXTRA_ROUTE_CAP, EXTRA_ROUTE_BONUS * max(0, len(routes) - 2))
        score += AGREE_BONUS + extra
        reasons.append(f"{len(routes)} independent routes agree within {vote_result.tolerance:.4g} "
                       f"(+{AGREE_BONUS + extra:.2f})")
    elif vote_result.agreement == "single":
        reasons.append("only one independent route produced this value, so it cannot be accepted "
                       "by vote")
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
    if vote_result.sigma and value and vote_result.sigma / abs(value) > SIGMA_SHARE:
        score -= SPREAD_PENALTY
        reasons.append(f"the digitisation uncertainty is {vote_result.sigma / abs(value):.0%} of "
                       f"the value (-{SPREAD_PENALTY:.2f})")

    # --- consistency flags
    errors = [f for f in flags if f.severity == "error"]
    warns = [f for f in flags if f.severity == "warn"]
    infos = [f for f in flags if f.severity == "info"]
    if errors:
        forced_human = True
        reasons.append("consistency errors stand: " + ", ".join(sorted({f.code for f in errors})))
    if warns:
        penalty = min(WARN_CAP, WARN_PENALTY * len(warns))
        score -= penalty
        reasons.append(f"warnings ({', '.join(sorted({f.code for f in warns}))}) -{penalty:.2f}")
    if infos:
        penalty = min(INFO_CAP, INFO_PENALTY * len(infos))
        score -= penalty
        reasons.append(f"notes ({', '.join(sorted({f.code for f in infos}))}) -{penalty:.2f}")

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
    if vote_result.agreement == "single" and not settled:
        score = min(score, SINGLE_ROUTE_CAP)

    # --- amendment F: a purely digitised cell has to earn its automatic acceptance
    if routes and all(is_figure_route(r.route_key) for r in routes) and not settled:
        ok, delta, share, gate_reasons = figure_gate(candidates or [], n_a, n_b)
        reasons += gate_reasons
        if not ok:
            score = min(score, ADJUDICATED_CAP)
            reasons.append("a digitised cell that does not meet the digitisation gate cannot be "
                           "accepted automatically")

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
