"""Section A's best-guess line: the primary analysis, PLUS the rows it had to hold back.

A review that can only report what it is sure of reports two rows and calls the question open.
The rows it held back are not noise — most of them carry a number the resolver actually built,
withheld because one corroboration was missing, not because the value is wrong. The best-guess
line says what those rows imply IF each held value is taken at face value, and it says it as a
second analysis standing beside the first, never in place of it.

**Invariant A0.** Best guess = strict, plus rows. Nothing here modifies or removes a strict row:
`best_guess_rows` returns the strict rows it was given, unchanged and identical objects apart from
ordering, and appends copies of the held rows a rule admitted. `k_bg ≥ k_strict`, always.

**No rebuild ever happens here.** The value the line uses is the held row's OWN `es`/`var` — the
number the resolver already produced. Orientation is the same: a row is signed by a human answer
or by the tiebreak ballot in the verify/rebuild path, long before this module sees it, and a row
that is still unsigned is vetoed rather than guessed at. This module reads records; it never
builds one.

Two things decide each held row, and every held row gets exactly one of them with a reason:

* a **veto** (`VETOES`, first match wins) — a named reason the row's value may not be borrowed at
  all. The two flag-driven vetoes import their vocabulary from `verify.confidence` rather than
  restating it, so a flag added to `ROW_REFUSAL_CODES` or `CONTRADICTING_FLAGS` starts vetoing
  here on the same commit. A flag a human answer retired is no longer on the rebuilt row, so it
  does not veto — which is the whole point of answering.
* a **rule** (`RULES`, first match wins) — the named ground on which the value is admitted, with
  the row's own value and variance carried through untouched.

A verifier's refutation is NOT a veto. A refutation is a dispute about a magnitude or a quantity,
and the best-guess line's job is to say what the reading implies while showing the dispute: the
reason opens with `disputed` and names it. What the line never resolves is a SIGN it does not
have — an unsigned row is vetoed, because guessing a direction would be inventing the finding,
and so is a row whose sign is CONTESTED (`sign_mismatch`: the direction the paper states and the
direction the extracted means imply disagree). A disputed magnitude is a number to show with its
dispute; a disputed direction is a different finding.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..models import EffectSizeRecord, OutcomeDef, StatsSettings
from ..stats.meta import MetaResult, prediction_interval
from ..verify.checks import severity_of
from .resolve import GROUP_ROUTES
from ..verify.confidence import (CONTRADICTING_FLAGS, INFERRED_PREMISE_FLAGS,
                                 ROW_REFUSAL_CODES, WITHHOLDING_FLAGS)
from .resolve import GROUP_STATISTICS_MISSING, PRECEDENCE_OVERRIDE

__all__ = ["BEST_GUESS_FLAG", "RULES", "VETOES", "VETO_ROW_FLAGS", "CONTRADICTED",
           "BestGuessDecision", "best_guess_rows", "best_guess_payload", "best_guess_cells",
           "mark_composites"]

#: on every row the best-guess line added, and on any composite one of them went into
BEST_GUESS_FLAG = "best_guess"
#: the closed set of grounds on which a held row may enter the line, in match order
RULES: tuple[str, ...] = ("inferred_premise", "low_confidence_value", "precedence_override")
#: the closed set of reasons a held row may not, in match order (first wins)
VETOES: tuple[str, ...] = ("row_refusal", "contradicted_value", "orientation_unresolvable",
                           "one_group_only", "no_variance")
#: the resolver owns the refusal vocabulary — imported, never copied (standing ruling)
VETO_ROW_FLAGS = ROW_REFUSAL_CODES
#: "this may be a different quantity" — `verify.confidence`'s own set, likewise imported
CONTRADICTED = CONTRADICTING_FLAGS
#: a contested SIGN, not a contested magnitude: `checks.sign_check` raises it when the direction
#: the paper STATES and the direction the extracted means imply disagree. It vetoes under
#: `contradicted_value` (amendment, whole-branch review MAJOR 2) — the closed enum is unchanged,
#: because what a disputed sign contradicts is the row's direction, and DECISION A's invariant is
#: that this line never resolves a direction it does not have. Named and checked rather than
#: spelled inline, so renaming the code in `verify.checks` fails here on the same commit.
SIGN_DISPUTED = "sign_mismatch"
assert severity_of(SIGN_DISPUTED) == "error"
#: codes that record HOW a decision was made rather than a doubt about a reading. They are not
#: disputes and the reason must not quote them as ones (whole-branch review, MINOR 7).
PROVENANCE_PREFIX = "orientation_"

#: `resolve.available_routes` writes this sentence into `routes_rejected`/`not_convertible_reason`
#: when a group is not ready; from D1 onward the same fact is also a flag
#: (`GROUP_STATISTICS_MISSING`), but records resolved before it carry only the sentence.
#: An alternation, because the sentence changed once: records written before the per-group
#: diagnosis said "have no mean, group size and dispersion" whatever was actually missing, and
#: both spellings must keep matching or old runs' held rows change meaning on reload.
_NO_GROUP_STATS = re.compile(r"group\(s\)[^;]*\bhave no mean\b|group\(s\) not ready —")


@dataclass(frozen=True)
class BestGuessDecision:
    """What the line did with ONE held row, and why — the record a reviewer is owed.

    `rule` is set iff `admitted`, `veto` iff not; `reason` is never empty. `label` is the row's
    citation name, carried so the payload can name the row without the record.
    """

    dataset_id: str
    outcome_key: str
    admitted: bool
    rule: str = ""
    veto: str = ""
    reason: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    es: float | None = None
    var: float | None = None
    label: str = ""


# ----------------------------------------------------------------------------- reading a row
def _label(record: EffectSizeRecord) -> str:
    """`First author Year` — the same name `report.tables` puts on a forest row.

    Restated here rather than imported: `canopy.report` draws with matplotlib at import time and
    the analysis must not pull a renderer in to name a row.
    """
    cite = record.citation
    name = ((cite.first_author or "").strip() or (cite.authors or "").strip()
            or (record.label or record.dataset_id or record.paper_id or "").strip() or "—")
    return f"{name} {cite.year}" if cite.year else name


def _usable(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _has_value(record: EffectSizeRecord) -> bool:
    """Did the resolver actually build an effect size on this row?"""
    return (record.route != "not_convertible" and _usable(record.es)
            and _usable(record.var) and float(record.var) > 0)


def _group_statistics_missing(record: EffectSizeRecord) -> bool:
    """Are the two groups' statistics NOT both present on this row?

    Three witnesses, in order of how durable they are: D1's flag, the numbers the record kept,
    and the sentence the resolver wrote when it rejected every group route.
    """
    if GROUP_STATISTICS_MISSING in record.flags:
        return True
    inputs = record.inputs or {}
    if "mean_a" in inputs or "mean_b" in inputs:
        return not (_usable(inputs.get("mean_a")) and _usable(inputs.get("mean_b")))
    prose = "; ".join([record.not_convertible_reason, *record.routes_rejected.values()])
    return bool(_NO_GROUP_STATS.search(prose))


def _severity(flag: str) -> str:
    """The severity `verify.checks` DECLARED for a code, or `""` for one it does not declare."""
    try:
        return severity_of(flag)
    except KeyError:
        return ""


def _disputes(record: EffectSizeRecord) -> list[str]:
    """The doubts recorded ON the row — what the reason has to show when it admits it anyway.

    A dispute is what the check layer DECLARED as one: a code it raised at `error` or `warn`, plus
    the withholding sets. Reading the flag's spelling instead (`"disput" in name`) made the mark a
    coincidence — `unit_mismatch` was never quoted because its name says nothing, and
    `calibration_disputed` was quoted only because of four letters in the middle of its name, so
    renaming that code would have silently dropped the whole mark from a row whose two groups may
    be on different scales. Severity is the one place that fact is stated on purpose.

    The `orientation_*` codes are the exception, and for the same reason: they carry a severity
    because a cell settled by a majority must be reviewed, but what they record is HOW the
    direction was decided, not a doubt about a reading. Quoting `orientation_by_majority` under
    "disputed (…)" told a reader the tiebreak itself was contested (MINOR 7).
    """
    flags = set(record.flags)
    return sorted(((flags & WITHHOLDING_FLAGS)
                   | {f for f in flags if _severity(f) in ("error", "warn")})
                  - {f for f in flags if f.startswith(PROVENANCE_PREFIX)})


def _evidence(record: EffectSizeRecord, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"route": record.route, "confidence": str(record.confidence),
                           "flags": list(record.flags)}
    out.update({k: v for k, v in extra.items() if v})
    return out


# ----------------------------------------------------------------------------- the vetoes
def _veto(record: EffectSizeRecord) -> tuple[str, str, dict[str, Any]] | None:
    """`(veto, reason, evidence)` for a row the line may not take, or `None`. First match wins."""
    refused = sorted(set(record.flags) & VETO_ROW_FLAGS)
    if refused:
        return ("row_refusal",
                f"the resolver refused this row ({', '.join(refused)}); a value its own "
                f"conversion refused is not one the best-guess line may borrow",
                _evidence(record, refusal_flags=refused))

    contradicting = sorted(set(record.flags) & CONTRADICTED)
    if contradicting:
        return ("contradicted_value",
                f"{', '.join(contradicting)}: the evidence says this reading may not be the "
                f"quantity the cell asks for, and a best guess about the wrong quantity is not a "
                f"guess about this contrast — a human has to settle it first",
                _evidence(record, contradicting_flags=contradicting))

    if SIGN_DISPUTED in record.flags:
        return ("contradicted_value",
                "sign_mismatch: the paper's stated direction contradicts the extracted means, so "
                "the sign of this row is disputed — the best-guess line never resolves a sign it "
                "does not have",
                _evidence(record, contradicting_flags=[SIGN_DISPUTED]))

    if record.higher_is_better is None:
        return ("orientation_unresolvable",
                "which direction counts as better was never settled for this measure, so this "
                "row has no sign; the best-guess line resolves magnitudes, never a direction it "
                "does not have",
                _evidence(record))

    if not _has_value(record):
        if _group_statistics_missing(record):
            detail = record.not_convertible_reason or "; ".join(record.routes_rejected.values())
            return ("one_group_only",
                    f"the two groups' statistics are not both present, so there is no contrast "
                    f"to take a guess at: {detail[:400] or 'no group statistics were resolved'}",
                    _evidence(record, not_convertible_reason=record.not_convertible_reason))
        detail = record.not_convertible_reason or "; ".join(record.routes_rejected.values())
        return ("no_variance",
                f"this row carries no usable effect size and variance (route {record.route or '—'}"
                f", es {record.es}, var {record.var})"
                + (f": {detail[:400]}" if detail else ""),
                _evidence(record, not_convertible_reason=record.not_convertible_reason))
    return None


# ----------------------------------------------------------------------------- the admit rules
def _precedence_override(record: EffectSizeRecord) -> bool:
    return PRECEDENCE_OVERRIDE in record.flags


def _held_only_by_bucket(record: EffectSizeRecord) -> bool:
    """Held because of its confidence bucket alone — no row-level rule put it here."""
    return not _precedence_override(record)


def _rule(record: EffectSizeRecord) -> tuple[str, str, dict[str, Any]] | None:
    """`(rule, reason, evidence)` for a row the line admits, or `None`. First match wins."""
    value = (f"{record.route} → d = {float(record.es):+.4g} (var {float(record.var):.4g}, "
             f"se {'—' if record.se is None else format(float(record.se), '.4g')})")
    disputed = _disputes(record)
    mark = f"disputed ({', '.join(disputed)}) — " if disputed else ""

    inferred = sorted(set(record.flags) & INFERRED_PREMISE_FLAGS)
    if inferred and record.route not in GROUP_ROUTES:
        inferred = []                     # this conversion consumed no inferred value (see resolve)
    if inferred:
        # FIRST, deliberately: an inferred-premise row also satisfies `low_confidence_value`
        # (the resolver did build a value), and admitting it under that rule would hide WHY it is
        # held — the premise is the tool's inference from the paper's other captions, not the
        # paper's label for this number, and the reviewer deciding whether to trust the line is
        # owed that sentence. The vetoes have already run: an inferred row that is also refuted,
        # refused or unsigned never reaches any rule.
        return ("inferred_premise",
                f"{mark}built on a premise the tool inferred from the paper "
                f"({', '.join(inferred)}): {value}. The paper supplied the evidence; no person "
                f"has confirmed the reading of it, and the question is still open.",
                _evidence(record, disputed=disputed, inferred=inferred))

    if _held_only_by_bucket(record):
        return ("low_confidence_value",
                f"{mark}held for review ({record.confidence}) but the resolver did build a "
                f"value: {value}. The best-guess line takes it at its own value — nothing is "
                f"recomputed, re-signed or rescaled.",
                _evidence(record, disputed=disputed))

    if _precedence_override(record):
        from_route = record.route_overridden_from or "the route the precedence list chose"
        return ("precedence_override",
                f"{mark}the precedence list chose {from_route}, which converts to nothing, so "
                f"the row was rebuilt from a same-locator reading and held: {value}. "
                f"{record.precedence_override_reason}".strip(),
                _evidence(record, disputed=disputed,
                          route_overridden_from=record.route_overridden_from,
                          precedence_override_reason=record.precedence_override_reason))
    return None                                            # pragma: no cover - a rule always fits


def _decide(record: EffectSizeRecord) -> BestGuessDecision:
    common = {"dataset_id": record.dataset_id, "outcome_key": record.outcome_key,
              "label": _label(record)}
    vetoed = _veto(record)
    if vetoed is not None:
        veto, reason, evidence = vetoed
        return BestGuessDecision(admitted=False, veto=veto, reason=reason, evidence=evidence,
                                 es=record.es, var=record.var, **common)
    admitted = _rule(record)
    if admitted is None:                                   # pragma: no cover - unreachable today
        return BestGuessDecision(
            admitted=False, veto="no_variance",
            reason=f"no admit rule fits this row (route {record.route or '—'})",
            evidence=_evidence(record), es=record.es, var=record.var, **common)
    rule, reason, evidence = admitted
    return BestGuessDecision(admitted=True, rule=rule, reason=reason, evidence=evidence,
                             es=record.es, var=record.var, **common)


def best_guess_rows(strict: Sequence[EffectSizeRecord], held: Sequence[EffectSizeRecord], *,
                    outcome: OutcomeDef, settings: StatsSettings,
                    ) -> tuple[list[EffectSizeRecord], list[BestGuessDecision]]:
    """`(the line's rows, one decision per held row)` — the strict rows first, then the admitted.

    `strict` are the primary analysis's rows BEFORE any within-paper aggregation and `held` the
    rows the confidence filter kept out of it; both come from `run._split_rows`. Aggregation is
    the caller's (`report.outputs`), once per line, so that a composite is built from the line it
    belongs to rather than from a mixture of the two.

    The returned strict rows are the objects passed in — not copies, not reordered among
    themselves, never edited (invariant A0). Each admitted row is a COPY carrying the flag, the
    rule and the reason; the record on disk is untouched, because best guess is a way of reading
    the analysis and not a stage that changes it.
    """
    rows = list(strict)
    decisions: list[BestGuessDecision] = []
    for record in held:
        decision = _decide(record)
        decisions.append(decision)
        if decision.admitted:
            rows.append(record.model_copy(update={
                "flags": [*record.flags, BEST_GUESS_FLAG, f"best_guess_rule:{decision.rule}"],
                "best_guess_rule": decision.rule, "best_guess_reason": decision.reason}))
    return rows, decisions


def mark_composites(rows: Sequence[EffectSizeRecord],
                    decisions: Sequence[BestGuessDecision]) -> list[EffectSizeRecord]:
    """After aggregation: a composite with any guessed member is best guess wholesale.

    `aggregate_one_row_per_paper` builds a fresh record whose flags are the union of its members'
    — so `best_guess` arrives on the composite by itself — but its `best_guess_rule`/`reason`
    fields start empty. This fills them in, naming the members that were guessed, because a
    composite half of which is a guess is not a confirmed row and must not be drawn as one.
    """
    admitted = {d.dataset_id: d for d in decisions if d.admitted}
    out: list[EffectSizeRecord] = []
    for row in rows:
        members = [admitted[part] for part in row.dataset_id.split("+") if part in admitted]
        if not members or row.best_guess_rule:
            out.append(row)
            continue
        rules = sorted({d.rule for d in members})
        out.append(row.model_copy(update={
            "flags": sorted(set(row.flags) | {BEST_GUESS_FLAG}),
            "best_guess_rule": rules[0] if len(rules) == 1 else "composite",
            "best_guess_reason": (
                "this paper's combined row includes "
                + ", ".join(f"{d.dataset_id} ({d.rule})" for d in members)
                + " — a composite with a guessed member is a best guess wholesale")}))
    return out


def best_guess_cells(rows: Sequence[EffectSizeRecord],
                     decisions: Sequence[BestGuessDecision],
                     strict: Sequence[EffectSizeRecord] = ()) -> dict[tuple[str, str], dict]:
    """`(dataset_id, outcome_key) -> the best-guess columns` for the extraction tables.

    Every row of the line gets an entry (a strict row is in the line, with no rule of its own),
    and so does every held row the line refused — with the veto and its reason, because "why is
    this row in neither analysis" is the question the table has to answer.

    `strict` is the line's PRE-aggregation membership. Without it, a strict row that the
    best-guess aggregation folded into a composite appears in neither the decisions nor the
    post-aggregation rows, and its cell would read `in_best_guess: None` — which the extraction
    table defines as "this run computed no second line", a different and false claim.
    """
    cells: dict[tuple[str, str], dict] = {}
    for record in strict:
        cells[(record.dataset_id, record.outcome_key)] = {
            "in_best_guess": True, "best_guess_rule": "", "best_guess_reason": "",
            "best_guess_es": record.es, "best_guess_se": record.se}
    for decision in decisions:
        cells[(decision.dataset_id, decision.outcome_key)] = (
            {"in_best_guess": True, "best_guess_rule": decision.rule,
             "best_guess_reason": decision.reason, "best_guess_es": decision.es,
             "best_guess_se": (math.sqrt(decision.var)
                               if decision.var and decision.var > 0 else None)}
            if decision.admitted else
            {"in_best_guess": False, "best_guess_rule": "",
             "best_guess_reason": f"{decision.veto}: {decision.reason}",
             "best_guess_es": None, "best_guess_se": None})
    # the line's own rows last: an admitted row that a within-paper aggregation combined away
    # keeps the entry its decision gave it above, and the composite that replaced it gets its own
    for record in rows:
        cells[(record.dataset_id, record.outcome_key)] = {
            "in_best_guess": True, "best_guess_rule": record.best_guess_rule,
            "best_guess_reason": record.best_guess_reason,
            "best_guess_es": record.es, "best_guess_se": record.se}
    return cells


# ----------------------------------------------------------------------------- the payload
def _cluster(record: EffectSizeRecord) -> str:
    return record.cluster_id or record.paper_id or record.dataset_id


def _sign_agrees(a: float | None, b: float | None) -> bool | None:
    if a is None or b is None or not math.isfinite(a) or not math.isfinite(b):
        return None
    return math.copysign(1.0, a) == math.copysign(1.0, b)


def best_guess_payload(strict_pooled: MetaResult | None, bg_pooled: MetaResult | None,
                       decisions: Sequence[BestGuessDecision], *,
                       added_rows: Sequence[EffectSizeRecord],
                       loo_bg: Sequence[Mapping[str, Any]],
                       rows: Sequence[EffectSizeRecord] = (),
                       weights: Mapping[str, float] | None = None,
                       settings: StatsSettings | None = None) -> dict[str, Any]:
    """The `best_guess` block of `pooled.json` — the second line, next to the first.

    `added_rows` are the line's rows that are guesses (after aggregation, a composite with a
    guessed member is one of them); `rows` is the whole line; `loo_bg` is
    `tables.leave_one_out_rows` over it, so `max_abs_delta_from_one_best_guess_row` can say how
    much of the line's estimate rests on a single guessed row.

    `weights` is `dataset_id -> weight %`, built by the caller from the SAME filter that fed the
    pooler. Re-deriving the poolable subset here and zipping the weights onto it by index would
    put a plausible number beside the wrong row the first time the two filters disagreed, with
    nothing raised — so the map is passed in rather than recomputed.
    """
    line = list(rows) if rows else list(added_rows)
    added_ids = {r.dataset_id for r in added_rows}
    not_added = [d for d in decisions if not d.admitted]
    weight_of = dict(weights or {})

    payload: dict[str, Any] = {
        # the same rule the strict payload uses: `k` is what was POOLED, and is 0 when nothing
        # was. How many rows the line has is a different question, and has its own key.
        "k": bg_pooled.k if bg_pooled is not None else 0,
        "k_rows": len(line),
        "k_papers": len({_cluster(r) for r in line}),
        "estimate": None, "se": None, "ci_low": None, "ci_high": None, "z": None, "p": None,
        "tau2": None, "I2": None, "Q": None, "Q_df": None, "Q_p": None,
        "pi_method": str(settings.pi_method) if settings is not None else "",
        "pi_low": None, "pi_high": None, "pi_df": None,
        "n_added": len(added_rows), "n_still_held": len(not_added),
        "delta_vs_strict": None, "sign_agrees_with_strict": None,
        "max_abs_delta_from_one_best_guess_row": None,
        "added": [{"dataset_id": r.dataset_id, "label": _label(r),
                   "rule": r.best_guess_rule, "reason": r.best_guess_reason,
                   "es": r.es, "se": r.se, "ci_low": r.ci_low, "ci_high": r.ci_high,
                   "weight_pct": weight_of.get(r.dataset_id)}
                  for r in added_rows],
        "not_added": [{"dataset_id": d.dataset_id, "label": d.label, "veto": d.veto,
                       "reason": d.reason} for d in not_added],
        "note": "",
    }

    if bg_pooled is not None:
        payload.update({
            "estimate": bg_pooled.estimate, "se": bg_pooled.se, "ci_low": bg_pooled.ci_low,
            "ci_high": bg_pooled.ci_high, "z": bg_pooled.z, "p": bg_pooled.p,
            "tau2": bg_pooled.tau2, "I2": bg_pooled.I2, "Q": bg_pooled.Q,
            "Q_df": bg_pooled.Q_df, "Q_p": bg_pooled.Q_p})
        pi_low, pi_high, pi_df = bg_pooled.pi_low, bg_pooled.pi_high, bg_pooled.pi_df
        if settings is not None:
            try:
                pi_low, pi_high, pi_df = prediction_interval(bg_pooled, str(settings.pi_method))
            except ValueError:                             # pragma: no cover - guarded upstream
                pi_low = pi_high = float("nan")
                pi_df = 0
        # …as `null` when it is not estimable, never as the bare token `NaN`: at k = 2 under
        # Hartung-Knapp (and under `pi_t` at any k the method cannot serve) the interval has no
        # width, and `NaN` is not JSON — every consumer that is not Python fails on the FILE
        # rather than on the number (review MINOR 21).
        payload.update({"pi_low": pi_low if _usable(pi_low) else None,
                        "pi_high": pi_high if _usable(pi_high) else None, "pi_df": pi_df})
        if strict_pooled is not None:
            payload["delta_vs_strict"] = bg_pooled.estimate - strict_pooled.estimate
            payload["sign_agrees_with_strict"] = _sign_agrees(bg_pooled.estimate,
                                                              strict_pooled.estimate)
        deltas = [abs(float(entry["estimate"]) - bg_pooled.estimate) for entry in loo_bg
                  if entry.get("estimate") is not None
                  and str(entry.get("omitted_dataset_id", "")) in added_ids]
        payload["max_abs_delta_from_one_best_guess_row"] = max(deltas) if deltas else None

    counts: dict[str, int] = {}
    for row in added_rows:
        counts[row.best_guess_rule or "composite"] = counts.get(row.best_guess_rule
                                                                or "composite", 0) + 1
    vetoes: dict[str, int] = {}
    for decision in not_added:
        vetoes[decision.veto] = vetoes.get(decision.veto, 0) + 1
    added_note = (", ".join(f"{rule} ×{n}" for rule, n in sorted(counts.items()))
                  if counts else "no held row could be admitted")
    held_note = (", ".join(f"{veto} ×{n}" for veto, n in sorted(vetoes.items()))
                 if vetoes else "nothing was held back")
    payload["note"] = (
        f"best guess = the primary analysis plus {len(added_rows)} of "
        f"{len(added_rows) + len(not_added)} held row(s), admitted by rule ({added_note}); "
        f"still held: {held_note}. This is not the primary analysis: every added row is one a "
        f"human has not confirmed, taken at its own value.")
    return payload
